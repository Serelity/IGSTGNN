#!/usr/bin/env bash
# Foreground v13b in the existing igstgnn environment on an allocated V100.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd -- "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_DIR"
ACTION=${1:-help}
case "$ACTION" in
  run|check|pilot|preflight|status|report)
    if [[ $# -gt 2 ]]; then echo 'Only an optional run name is accepted.' >&2; exit 2; fi ;;
  resume)
    if [[ $# -ne 3 ]]; then echo 'Usage: resume NEW_RUN SOURCE_RUN' >&2; exit 2; fi ;;
  _worker)
    if [[ $# -ne 6 ]]; then echo 'Invalid worker arguments.' >&2; exit 2; fi ;;
  help)
    if [[ $# -gt 1 ]]; then echo 'help accepts no arguments.' >&2; exit 2; fi ;;
  *) echo "Unknown action: $ACTION" >&2; exit 2 ;;
esac
RUN_NAME=${2:-contra_v13b_information_01}
if [[ ! "$RUN_NAME" =~ ^contra_v13b_information_[A-Za-z0-9_-]+$ ]]; then
  echo 'Use a contra_v13b_information_* name with letters, digits, underscores or hyphens.' >&2
  exit 2
fi
OUT="$REPO_DIR/experiments/chronological_runs/$RUN_NAME"
JOB="${OUT}.job"
case "$ACTION" in
  run|check|pilot|preflight|resume)
    SOURCE=''
    if [[ "$ACTION" == resume ]]; then
      SOURCE_NAME=$3
      if [[ ! "$SOURCE_NAME" =~ ^contra_v13b_information_[A-Za-z0-9_-]+$ || "$SOURCE_NAME" == "$RUN_NAME" ]]; then
        echo 'Source must be a different v13b run name.' >&2; exit 2
      fi
      SOURCE="$REPO_DIR/experiments/chronological_runs/${SOURCE_NAME}.partial"
      if [[ ! -d "$SOURCE" || -L "$SOURCE" || ! -f "$SOURCE/run_identity.json" || -L "$SOURCE/run_identity.json" ]]; then
        echo 'Missing regular partial recovery identity.' >&2; exit 1
      fi
    fi
    for TARGET in "$OUT" "${OUT}.partial" "$JOB"; do
      if [[ -e "$TARGET" || -L "$TARGET" ]]; then
        printf 'Preserve existing result; choose a new run name: %s\n' "$TARGET" >&2; exit 1
      fi
    done
    PYTHON_BIN=$(python -c 'import sys; print(sys.executable)')
    DEVICE=${V13B_DEVICE:-cuda:0}
    if [[ "$ACTION" == preflight ]]; then DEVICE=cpu; fi
    mkdir -p -- "$(dirname -- "$OUT")"
    mkdir -- "$JOB"
    hostname > "$JOB/host"
    bash "$SCRIPT_DIR/run_minimal_information.sh" _worker "$RUN_NAME" "$PYTHON_BIN" "$DEVICE" "$ACTION" "$SOURCE" 2>&1 | tee "$JOB/run.log"
    ;;
  _worker)
    PYTHON_BIN=$3
    DEVICE=$4
    MODE=$5
    SOURCE=$6
    if [[ ! -d "$JOB" || -L "$JOB" ]]; then echo 'Missing regular job directory.' >&2; exit 1; fi
    case "$MODE" in run|check|pilot|preflight|resume) ;; *) echo 'Invalid worker mode.' >&2; exit 2 ;; esac
    trap 'STATUS=$?; printf "%s\n" "$STATUS" > "$JOB/exit_code"; printf "Workflow exit code: %s\nFinished: %s\n" "$STATUS" "$(date -Is)"' EXIT
    printf '%s\n' "$$" > "$JOB/pid"
    printf '%s\n' "${SLURM_JOB_ID:-unset}" > "$JOB/slurm_job_id"
    export CUBLAS_WORKSPACE_CONFIG=:4096:8
    export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3 NUMEXPR_NUM_THREADS=3
    printf 'Started: %s\nHost: %s\nSlurm job: %s\nPython: %s\nMode: %s\nDevice: %s\n' \
      "$(date -Is)" "$(hostname)" "${SLURM_JOB_ID:-unset}" "$PYTHON_BIN" "$MODE" "$DEVICE"
    git log -1 --format='commit=%H'
    "$PYTHON_BIN" -c 'import sys,torch,numpy; print("Torch:",torch.__version__,"NumPy:",numpy.__version__); device=torch.device(sys.argv[1]); assert device.type != "cuda" or torch.cuda.is_available(), "CUDA unavailable: activate igstgnn on an allocated GPU node"; torch.empty(1, device=device)' "$DEVICE"
    "$PYTHON_BIN" -m unittest discover -s tests -p 'test_minimal_information*.py' -v
    ARGS=(--data-dir "${V13B_DATA_DIR:-../data/chronological/Contra_Costa_v8_dev}"
      --primary-control-dir "${V13B_PRIMARY_DIR:-../research_artifacts/v6_inputs_20260920/v3_materialized_01}"
      --secondary-control-dir "${V13B_SECONDARY_DIR:-../research_artifacts/v6_inputs_20260920/v5b_second_materialized_01}"
      --device "$DEVICE")
    if [[ "$MODE" == preflight || "$MODE" == check ]]; then
      "$PYTHON_BIN" -u "$SCRIPT_DIR/train_minimal_information.py" run "${ARGS[@]}" --mode "$MODE" --output "$OUT"
    else
      "$PYTHON_BIN" -u "$SCRIPT_DIR/train_minimal_information.py" run "${ARGS[@]}" --mode check --output "$JOB/check"
      if [[ "$MODE" == resume ]]; then
        MODE=run
        ARGS+=(--resume-from "$SOURCE")
      fi
      printf 'Starting %s: fresh full-backbone training, gradients and Adam steps enabled. No old A/scaler.\n' "$MODE"
      "$PYTHON_BIN" -u "$SCRIPT_DIR/train_minimal_information.py" run "${ARGS[@]}" --mode "$MODE" --output "$OUT"
    fi
    ;;
  status)
    if [[ ! -d "$JOB" ]]; then echo "No run record: $JOB" >&2; exit 1; fi
    printf 'Run host: %s\nCurrent host: %s\n' "$(< "$JOB/host")" "$(hostname)"
    if [[ -f "$JOB/exit_code" ]]; then
      printf 'Workflow exit code: %s\n' "$(< "$JOB/exit_code")"
    elif [[ "$(< "$JOB/host")" == "$(hostname)" && -f "$JOB/pid" ]]; then
      ps -o pid,lstart,etime,%cpu,%mem,stat,cmd -p "$(< "$JOB/pid")" || true
    fi
    tail -n 20 "$JOB/run.log"
    ;;
  report)
    if [[ ! -f "$OUT/summary.json" ]]; then echo 'No completed result; inspect status and preserve partial.' >&2; exit 1; fi
    python "$SCRIPT_DIR/train_minimal_information.py" report "$OUT/summary.json"
    ;;
  help)
    echo 'Usage: bash experiments/chronological/run_minimal_information.sh {check|pilot|run|preflight|status|report} [contra_v13b_information_01]'
    echo 'Resume: bash experiments/chronological/run_minimal_information.sh resume NEW_RUN SOURCE_RUN'
    echo 'pilot: one full fit/selection epoch per arm on CUDA, no audit; review time before the 9-fit, 60-epoch run.'
    echo 'run: 9 fresh full-backbone fits, 102600 Adam steps; all selected endpoints freeze before audit.'
    echo 'Foreground execution, existing igstgnn/V100, no SSH. Outputs are never overwritten.'
    ;;
esac
