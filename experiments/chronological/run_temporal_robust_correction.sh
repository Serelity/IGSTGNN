#!/usr/bin/env bash
# Foreground v12o: paired ERM/temporal-excess fresh adapter training on V100.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd -- "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_DIR"
ACTION=${1:-help}
case "$ACTION" in
  run|check|status|report)
    if [[ $# -gt 2 ]]; then echo "$ACTION accepts only an optional run name." >&2; exit 2; fi
    ;;
  resume)
    if [[ $# -ne 3 ]]; then echo 'Recovery requires: resume NEW_RUN SOURCE_RUN.' >&2; exit 2; fi
    ;;
  help)
    if [[ $# -gt 1 ]]; then echo 'help accepts no additional arguments.' >&2; exit 2; fi
    ;;
  _worker)
    if [[ $# -ne 6 ]]; then echo 'Invalid internal worker arguments.' >&2; exit 2; fi
    ;;
  *) echo "Unknown action: $ACTION" >&2; exit 2 ;;
esac
RUN_NAME=${2:-contra_v12o_temporal_robust_01}
if [[ ! "$RUN_NAME" =~ ^contra_v12o_temporal_robust_[A-Za-z0-9_-]+$ ]]; then
  echo 'Run name must start with contra_v12o_temporal_robust_ and contain only letters, digits, underscores or hyphens.' >&2
  exit 2
fi
OUT="$REPO_DIR/experiments/chronological_runs/$RUN_NAME"
JOB="${OUT}.job"

case "$ACTION" in
  run|check|resume)
    SOURCE=''
    if [[ "$ACTION" == resume ]]; then
      SOURCE_NAME=$3
      if [[ ! "$SOURCE_NAME" =~ ^contra_v12o_temporal_robust_[A-Za-z0-9_-]+$ || "$SOURCE_NAME" == "$RUN_NAME" ]]; then
        echo 'Recovery source must be a different contra_v12o_temporal_robust_* run name.' >&2
        exit 2
      fi
      SOURCE="$REPO_DIR/experiments/chronological_runs/${SOURCE_NAME}.partial"
      if [[ ! -d "$SOURCE" || -L "$SOURCE" || ! -f "$SOURCE/run_identity.json" || -L "$SOURCE/run_identity.json" ]]; then
        printf 'Missing regular recovery identity, or symlinked source: %s\n' "$SOURCE/run_identity.json" >&2
        exit 1
      fi
    fi
    for TARGET in "$OUT" "${OUT}.partial" "$JOB"; do
      if [[ -e "$TARGET" || -L "$TARGET" ]]; then
        printf 'Already exists; preserve it and use a new run name: %s\n' "$TARGET" >&2
        exit 1
      fi
    done
    PYTHON_BIN=$(python -c 'import sys; print(sys.executable)')
    DEVICE=${V12O_DEVICE:-cuda:0}
    mkdir -p -- "$(dirname -- "$OUT")"
    mkdir -- "$JOB"
    hostname > "$JOB/host"
    bash "$SCRIPT_DIR/run_temporal_robust_correction.sh" _worker "$RUN_NAME" "$PYTHON_BIN" "$DEVICE" "$SOURCE" "$ACTION" 2>&1 | tee "$JOB/run.log"
    ;;
  _worker)
    PYTHON_BIN=$3
    DEVICE=$4
    SOURCE=$5
    MODE=$6
    if [[ ! -d "$JOB" || -L "$JOB" ]]; then echo 'Missing regular worker job directory.' >&2; exit 1; fi
    trap 'STATUS=$?; printf "%s\n" "$STATUS" > "$JOB/exit_code"; printf "Workflow exit code: %s\nFinished: %s\n" "$STATUS" "$(date -Is)"' EXIT
    printf '%s\n' "$$" > "$JOB/pid"
    printf '%s\n' "${SLURM_JOB_ID:-unset}" > "$JOB/slurm_job_id"
    export CUBLAS_WORKSPACE_CONFIG=:4096:8
    export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3 NUMEXPR_NUM_THREADS=3
    printf 'Started: %s\nHost: %s\nSlurm job: %s\nPython: %s\nDevice: %s\n' \
      "$(date -Is)" "$(hostname)" "${SLURM_JOB_ID:-unset}" "$PYTHON_BIN" "$DEVICE"
    if [[ -n "$SOURCE" ]]; then printf 'Read-only recovery source: %s\n' "$SOURCE"; fi
    git log -1 --format='commit=%H'
    "$PYTHON_BIN" -c 'import sys,torch,numpy; assert sys.version_info >= (3,10), "Python 3.10+ required"; print("Torch:",torch.__version__,"NumPy:",numpy.__version__); device=torch.device(sys.argv[1]); assert device.type != "cuda" or torch.cuda.is_available(), "CUDA unavailable: use an allocated GPU and activate igstgnn"; torch.empty(1, device=device)' "$DEVICE"
    "$PYTHON_BIN" -m unittest discover -s tests -p 'test_temporal_robust_correction*.py' -v
    ARGS=(
      --data-dir ../data/chronological/Contra_Costa_v8_dev
      --primary-control-dir ../research_artifacts/v6_inputs_20260920/v3_materialized_01
      --secondary-control-dir ../research_artifacts/v6_inputs_20260920/v5b_second_materialized_01
      --checkpoint experiments/chronological_runs/contra_fixed_s2025_full_01/best_model.pt
      --device "$DEVICE"
    )
    "$PYTHON_BIN" -u "$SCRIPT_DIR/train_temporal_robust_correction.py" run "${ARGS[@]}" --output "$JOB/check" --check
    if [[ "$MODE" == check ]]; then
      echo 'Engineering check complete. No formal experiment was run.'
      exit 0
    fi
    echo 'Engineering check passed; starting 12 fresh adapter fits: 144 epochs, 27360 Adam steps, 12 protected endpoints. All endpoints freeze before audit.'
    if [[ -n "$SOURCE" ]]; then ARGS+=(--resume-from "$SOURCE"); fi
    time "$PYTHON_BIN" -u "$SCRIPT_DIR/train_temporal_robust_correction.py" run "${ARGS[@]}" --output "$OUT"
    ;;
  status)
    if [[ ! -d "$JOB" ]]; then echo "No run record: $JOB" >&2; exit 1; fi
    if [[ -f "$JOB/host" ]]; then printf 'Run host: %s\n' "$(< "$JOB/host")"; fi
    printf 'Current host: %s\n' "$(hostname)"
    if [[ -f "$JOB/slurm_job_id" ]]; then printf 'Slurm job: %s\n' "$(< "$JOB/slurm_job_id")"; fi
    if [[ -f "$JOB/exit_code" ]]; then
      printf 'Workflow exit code: %s\n' "$(< "$JOB/exit_code")"
    elif [[ -f "$JOB/host" && "$(< "$JOB/host")" == "$(hostname)" && -f "$JOB/pid" ]]; then
      ps -o pid,lstart,etime,%cpu,%mem,stat,cmd -p "$(< "$JOB/pid")" || true
    fi
    if [[ -f "$JOB/run.log" ]]; then tail -n 20 "$JOB/run.log"; fi
    if [[ -f "$OUT/summary.json" ]]; then
      printf 'Summary: %s\n' "$OUT/summary.json"
    else
      echo 'No completed summary yet; inspect exit code/log and preserve partial.'
      if [[ -d "${OUT}.partial" ]]; then printf 'Partial: %s\n' "${OUT}.partial"; fi
    fi
    ;;
  report)
    if [[ ! -f "$OUT/summary.json" ]]; then
      printf 'No completed result: %s\nUse status to inspect the log; resume an interrupted run into a new name.\n' "$OUT/summary.json" >&2
      exit 1
    fi
    python "$SCRIPT_DIR/train_temporal_robust_correction.py" report "$OUT/summary.json"
    ;;
  help)
    echo 'Usage: bash experiments/chronological/run_temporal_robust_correction.sh {run|check|status|report} [contra_v12o_temporal_robust_01]'
    echo 'Recovery: bash experiments/chronological/run_temporal_robust_correction.sh resume contra_v12o_temporal_robust_02 contra_v12o_temporal_robust_01'
    echo 'run/resume are foreground commands: tests and a real small-package check precede 12 paired fresh adapter fits (ERM and temporal excess).'
    echo 'V12O_DEVICE defaults to cuda:0. Full budget: 144 epochs, 27360 Adam steps, 12 endpoints; freeze all before audit.'
    ;;
esac
