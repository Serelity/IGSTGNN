#!/usr/bin/env bash
# Foreground v13a; run from an allocated campus node in the existing igstgnn env.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd -- "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_DIR"
ACTION=${1:-help}
case "$ACTION" in
  run|check|inventory|status|report)
    if [[ $# -gt 2 ]]; then echo 'Only an optional run name is accepted.' >&2; exit 2; fi ;;
  _worker)
    if [[ $# -ne 5 ]]; then echo 'Invalid worker arguments.' >&2; exit 2; fi ;;
  help)
    if [[ $# -gt 1 ]]; then echo 'help accepts no arguments.' >&2; exit 2; fi ;;
  *) echo "Unknown action: $ACTION" >&2; exit 2 ;;
esac
RUN_NAME=${2:-contra_v13a_inputs_01}
if [[ ! "$RUN_NAME" =~ ^contra_v13a_inputs_[A-Za-z0-9_-]+$ ]]; then
  echo 'Run name must start with contra_v13a_inputs_ and contain letters/digits/underscores/hyphens.' >&2
  exit 2
fi
OUT="$REPO_DIR/experiments/chronological_runs/$RUN_NAME"
JOB="${OUT}.job"
case "$ACTION" in
  run|check|inventory)
    for TARGET in "$OUT" "${OUT}.partial" "$JOB"; do
      if [[ -e "$TARGET" || -L "$TARGET" ]]; then
        printf 'Preserve existing result; use a new run name: %s\n' "$TARGET" >&2
        exit 1
      fi
    done
    PYTHON_BIN=$(python -c 'import sys; print(sys.executable)')
    DEVICE=${V13A_DEVICE:-cuda:0}
    if [[ "$ACTION" == inventory ]]; then DEVICE=cpu; fi
    mkdir -p -- "$(dirname -- "$OUT")"
    mkdir -- "$JOB"
    hostname > "$JOB/host"
    bash "$SCRIPT_DIR/run_effective_input_audit.sh" _worker "$RUN_NAME" "$PYTHON_BIN" "$DEVICE" "$ACTION" 2>&1 | tee "$JOB/run.log"
    ;;
  _worker)
    PYTHON_BIN=$3
    DEVICE=$4
    MODE=$5
    if [[ ! -d "$JOB" || -L "$JOB" ]]; then echo 'Missing regular job directory.' >&2; exit 1; fi
    case "$MODE" in run|check|inventory) ;; *) echo 'Invalid worker mode.' >&2; exit 2 ;; esac
    trap 'STATUS=$?; printf "%s\n" "$STATUS" > "$JOB/exit_code"; printf "Workflow exit code: %s\nFinished: %s\n" "$STATUS" "$(date -Is)"' EXIT
    printf '%s\n' "$$" > "$JOB/pid"
    printf '%s\n' "${SLURM_JOB_ID:-unset}" > "$JOB/slurm_job_id"
    export CUBLAS_WORKSPACE_CONFIG=:4096:8
    export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3 NUMEXPR_NUM_THREADS=3
    printf 'Started: %s\nHost: %s\nSlurm job: %s\nPython: %s\nMode: %s\nDevice: %s\n' \
      "$(date -Is)" "$(hostname)" "${SLURM_JOB_ID:-unset}" "$PYTHON_BIN" "$MODE" "$DEVICE"
    git log -1 --format='commit=%H'
    "$PYTHON_BIN" -c 'import sys,torch,numpy; print("Torch:",torch.__version__,"NumPy:",numpy.__version__); device=torch.device(sys.argv[1]); assert device.type != "cuda" or torch.cuda.is_available(), "CUDA unavailable: activate igstgnn on an allocated GPU node"; torch.empty(1, device=device)' "$DEVICE"
    "$PYTHON_BIN" -m unittest discover -s tests -p 'test_effective_input*.py' -v
    ARGS=(--data-dir "${V13A_DATA_DIR:-../data/chronological/Contra_Costa_v8_dev}" --device "$DEVICE")
    if [[ "$MODE" == inventory ]]; then
      "$PYTHON_BIN" -u "$SCRIPT_DIR/audit_effective_inputs.py" run "${ARGS[@]}" --inventory-only --output "$OUT"
    else
      ARGS+=(--checkpoint "${V13A_CHECKPOINT:-experiments/chronological_runs/contra_fixed_s2025_full_01/best_model.pt}")
      "$PYTHON_BIN" -u "$SCRIPT_DIR/audit_effective_inputs.py" run "${ARGS[@]}" --check --output "$JOB/check"
      if [[ "$MODE" == run ]]; then
        echo 'Check passed. Comparing original/simplified frozen A on 1520 fit X windows. No optimizer or Y loss.'
        "$PYTHON_BIN" -u "$SCRIPT_DIR/audit_effective_inputs.py" run "${ARGS[@]}" --output "$OUT"
      fi
    fi
    ;;
  status)
    if [[ ! -d "$JOB" ]]; then echo "No run record: $JOB" >&2; exit 1; fi
    printf 'Run host: %s\nCurrent host: %s\n' "$(< "$JOB/host")" "$(hostname)"
    if [[ -f "$JOB/exit_code" ]]; then
      printf 'Workflow exit code: %s\n' "$(< "$JOB/exit_code")"
    elif [[ "$(< "$JOB/host")" == "$(hostname)" && -f "$JOB/pid" ]]; then
      ps -o pid,lstart,etime,%cpu,%mem,stat,cmd -p "$(< "$JOB/pid")" || true
    else
      echo 'A different host cannot determine the recorded process status.'
    fi
    tail -n 18 "$JOB/run.log"
    ;;
  report)
    SUMMARY="$OUT/summary.json"
    if [[ ! -f "$SUMMARY" ]]; then SUMMARY="$JOB/check/summary.json"; fi
    python "$SCRIPT_DIR/audit_effective_inputs.py" report "$SUMMARY"
    ;;
  help)
    echo 'Usage: bash experiments/chronological/run_effective_input_audit.sh {run|check|inventory|status|report} [contra_v13a_inputs_01]'
    echo 'run: tests, two-window check, then 1520-window inference audit on CUDA. inventory: CPU input inventory only.'
    echo 'Foreground execution; keep the allocated job alive. Existing outputs are never overwritten.'
    ;;
esac
