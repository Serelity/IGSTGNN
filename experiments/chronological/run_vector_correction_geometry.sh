#!/usr/bin/env bash
# Local/GitHub delivery; run this foreground workflow in the server igstgnn environment.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd -- "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_DIR"
ACTION=${1:-help}
case "$ACTION" in
  run|check|preflight) if [[ $# -gt 3 ]]; then echo 'Expected ACTION [RUN_NAME] [V12M_SOURCE_NAME].' >&2; exit 2; fi ;;
  status|report) if [[ $# -gt 2 ]]; then echo 'Expected ACTION [RUN_NAME].' >&2; exit 2; fi ;;
  _worker) if [[ $# -ne 6 ]]; then echo 'Invalid worker arguments.' >&2; exit 2; fi ;;
  help) if [[ $# -gt 1 ]]; then echo 'help accepts no arguments.' >&2; exit 2; fi ;;
  *) echo "Unknown action: $ACTION" >&2; exit 2 ;;
esac
if [[ "$ACTION" == help ]]; then
  echo 'Usage: bash experiments/chronological/run_vector_correction_geometry.sh {preflight|check|run} [contra_v12n_geometry_01] [contra_v12m_output_scope_01]'
  echo 'Inspect: bash experiments/chronological/run_vector_correction_geometry.sh {status|report} [RUN_NAME]'
  echo 'Activate igstgnn on the allocated V100 server. run performs tests, input preflight, frozen subset replay, then full frozen inference.'
  echo 'No training, SSH, network fetch, lambda search or test-set access. V12N_DEVICE defaults to cuda:0.'
  exit 0
fi
RUN_NAME=${2:-contra_v12n_geometry_01}
if [[ ! "$RUN_NAME" =~ ^contra_v12n_geometry_[A-Za-z0-9_-]+$ ]]; then
  echo 'Use a fresh contra_v12n_geometry_* name containing only letters, digits, underscores or hyphens.' >&2; exit 2
fi
OUT="$REPO_DIR/experiments/chronological_runs/$RUN_NAME"
JOB="${OUT}.job"
if [[ "$ACTION" == _worker ]]; then
  PYTHON_BIN=$3
  DEVICE=$4
  SOURCE=$5
  MODE=$6
  if [[ "$MODE" != run && "$MODE" != check ]]; then echo 'Invalid worker mode.' >&2; exit 2; fi
else
  SOURCE_NAME=${3:-contra_v12m_output_scope_01}
  if [[ ! "$SOURCE_NAME" =~ ^contra_v12m_output_scope_[A-Za-z0-9_-]+$ ]]; then
    echo 'Source must be a completed contra_v12m_output_scope_* run.' >&2; exit 2
  fi
  SOURCE="$REPO_DIR/experiments/chronological_runs/$SOURCE_NAME"
  DEVICE=${V12N_DEVICE:-cuda:0}
fi
ARGS=(--source "$SOURCE"
  --data-dir "${V12N_DATA_DIR:-../data/chronological/Contra_Costa_v8_dev}"
  --primary-control-dir "${V12N_PRIMARY_DIR:-../research_artifacts/v6_inputs_20260920/v3_materialized_01}"
  --secondary-control-dir "${V12N_SECONDARY_DIR:-../research_artifacts/v6_inputs_20260920/v5b_second_materialized_01}"
  --checkpoint "${V12N_CHECKPOINT:-experiments/chronological_runs/contra_fixed_s2025_full_01/best_model.pt}")
case "$ACTION" in
  preflight)
    python "$SCRIPT_DIR/audit_vector_correction_geometry.py" preflight "${ARGS[@]}"
    ;;
  run|check)
    for TARGET in "$OUT" "${OUT}.partial" "$JOB"; do
      if [[ -e "$TARGET" || -L "$TARGET" ]]; then echo "Preserve existing path: $TARGET" >&2; exit 1; fi
    done
    PYTHON_BIN=$(python -c 'import sys; print(sys.executable)')
    # Reject a symlinked parent before creating any job record.
    "$PYTHON_BIN" -c 'from pathlib import Path; import sys; p=Path(sys.argv[1]); assert not any(x.is_symlink() for x in (p,*p.parents)), "Symlinked output parent"' "$OUT"
    mkdir -p -- "$(dirname -- "$OUT")"
    mkdir -- "$JOB"
    hostname > "$JOB/host"
    bash "$SCRIPT_DIR/run_vector_correction_geometry.sh" _worker "$RUN_NAME" "$PYTHON_BIN" "$DEVICE" "$SOURCE" "$ACTION" 2>&1 | tee "$JOB/run.log"
    ;;
  _worker)
    if [[ ! -d "$JOB" || -L "$JOB" ]]; then echo 'Missing regular job directory.' >&2; exit 1; fi
    trap 'STATUS=$?; printf "%s\n" "$STATUS" > "$JOB/exit_code"; printf "Workflow exit code: %s\nFinished: %s\n" "$STATUS" "$(date -Is)"' EXIT
    printf '%s\n' "$$" > "$JOB/pid"
    printf '%s\n' "${SLURM_JOB_ID:-unset}" > "$JOB/slurm_job_id"
    export CUBLAS_WORKSPACE_CONFIG=:4096:8
    export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3 NUMEXPR_NUM_THREADS=3
    printf 'Started: %s\nHost: %s\nSlurm job: %s\nPython: %s\nDevice: %s\nRead-only source: %s\n' \
      "$(date -Is)" "$(hostname)" "${SLURM_JOB_ID:-unset}" "$PYTHON_BIN" "$DEVICE" "$SOURCE"
    git log -1 --format='commit=%H'
    "$PYTHON_BIN" -c 'import sys,torch,numpy; assert sys.version_info >= (3,10); print("Torch:",torch.__version__,"NumPy:",numpy.__version__); d=torch.device(sys.argv[1]); assert d.type != "cuda" or torch.cuda.is_available(), "Activate igstgnn in an allocated GPU session"; torch.empty(1,device=d); print("GPU:",torch.cuda.get_device_name(d) if d.type=="cuda" else "CPU engineering check")' "$DEVICE"
    "$PYTHON_BIN" -m unittest discover -s tests -p 'test_vector_correction_geometry*.py' -v
    "$PYTHON_BIN" -u "$SCRIPT_DIR/audit_vector_correction_geometry.py" preflight "${ARGS[@]}"
    if [[ "$MODE" == check ]]; then
      "$PYTHON_BIN" -u "$SCRIPT_DIR/audit_vector_correction_geometry.py" run "${ARGS[@]}" --device "$DEVICE" --check --output "$OUT"
    else
      "$PYTHON_BIN" -u "$SCRIPT_DIR/audit_vector_correction_geometry.py" run "${ARGS[@]}" --device "$DEVICE" --check --output "$JOB/check"
      echo 'Engineering replay passed; starting six fixed selected endpoints. Zero optimizer steps.'
      time "$PYTHON_BIN" -u "$SCRIPT_DIR/audit_vector_correction_geometry.py" run "${ARGS[@]}" --device "$DEVICE" --output "$OUT"
    fi
    ;;
  status)
    if [[ ! -d "$JOB" || -L "$JOB" ]]; then echo "No regular job record: $JOB" >&2; exit 1; fi
    if [[ -f "$JOB/host" ]]; then printf 'Run host: %s\n' "$(< "$JOB/host")"; fi
    if [[ -f "$JOB/exit_code" ]]; then
      printf 'Workflow exit code: %s\n' "$(< "$JOB/exit_code")"
    elif [[ -f "$JOB/host" && "$(< "$JOB/host")" == "$(hostname)" && -f "$JOB/pid" ]]; then
      ps -o pid,lstart,etime,%cpu,%mem,stat,cmd -p "$(< "$JOB/pid")" || true
    fi
    if [[ -f "$JOB/run.log" ]]; then tail -n 20 "$JOB/run.log"; fi
    if [[ -f "$OUT/summary.json" ]]; then printf 'Summary: %s\n' "$OUT/summary.json"; else echo 'No completed result; preserve partial and inspect the log.'; fi
    ;;
  report)
    if [[ ! -f "$OUT/summary.json" ]]; then echo 'No completed v12n result; use status.' >&2; exit 1; fi
    python "$SCRIPT_DIR/audit_vector_correction_geometry.py" report "$OUT/summary.json"
    ;;
esac
