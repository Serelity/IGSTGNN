#!/usr/bin/env bash
# Foreground CPU support diagnostic of completed v12i and v12f artifacts.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd -- "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_DIR"
ACTION=${1:-help}
RUN_NAME=${2:-contra_v12j_support_audit_01}
if [[ ! "$RUN_NAME" =~ ^contra_v12j_support_audit_[A-Za-z0-9_-]+$ ]]; then
  echo 'Run name must match contra_v12j_support_audit_01.' >&2
  exit 2
fi
OUT="$REPO_DIR/experiments/chronological_runs/$RUN_NAME"
JOB="${OUT}.job"

case "$ACTION" in
  run)
    if [[ $# -gt 4 ]]; then echo 'run accepts only run name, v12i name and v12f name.' >&2; exit 2; fi
    FIT_NAME=${3:-contra_v12i_full_fit_audit_01}
    SOURCE_NAME=${4:-contra_v12f_state_interaction_01}
    if [[ ! "$FIT_NAME" =~ ^contra_v12i_full_fit_audit_[A-Za-z0-9_-]+$ ||
          ! "$SOURCE_NAME" =~ ^contra_v12f_state_interaction_[A-Za-z0-9_-]+$ ]]; then
      echo 'Sources must be completed contra_v12i_full_fit_audit_* and contra_v12f_state_interaction_* runs, not partial directories.' >&2
      exit 2
    fi
    FIT="$REPO_DIR/experiments/chronological_runs/$FIT_NAME"
    SOURCE="$REPO_DIR/experiments/chronological_runs/$SOURCE_NAME"
    for DIRECTORY in "$FIT" "$SOURCE"; do
      if [[ ! -d "$DIRECTORY" || -L "$DIRECTORY" ]]; then
        printf 'Missing completed source directory, or source is a symlink: %s\n' "$DIRECTORY" >&2
        exit 1
      fi
    done
    FIT_FILES=(summary.json fit_A.npz)
    SOURCE_FILES=(summary.json run_identity.json eligibility.json audit_A_incident_full.npz)
    for ARM in strength state_vector interaction_vector; do
      for SEED in 2025 2026 2027; do
        FIT_FILES+=("fit_${ARM}_s${SEED}.npz")
        SOURCE_FILES+=("${ARM}_s${SEED}/audit_incident_full.npz")
        if [[ -L "$SOURCE/${ARM}_s${SEED}" ]]; then
          printf 'Source model directory must not be a symlink: %s\n' "$SOURCE/${ARM}_s${SEED}" >&2
          exit 1
        fi
      done
    done
    for FILE in "${FIT_FILES[@]}"; do
      if [[ ! -f "$FIT/$FILE" || -L "$FIT/$FILE" ]]; then
        printf 'Missing regular v12i result: %s\n' "$FIT/$FILE" >&2
        exit 1
      fi
    done
    for FILE in "${SOURCE_FILES[@]}"; do
      if [[ ! -f "$SOURCE/$FILE" || -L "$SOURCE/$FILE" ]]; then
        printf 'Missing regular v12f result: %s\n' "$SOURCE/$FILE" >&2
        exit 1
      fi
    done
    for TARGET in "$OUT" "${OUT}.partial" "$JOB"; do
      if [[ -e "$TARGET" || -L "$TARGET" ]]; then
        printf 'Already exists; preserve it and use a new run name: %s\n' "$TARGET" >&2
        exit 1
      fi
    done
    PYTHON_BIN=$(python -c 'import sys; print(sys.executable)')
    mkdir -p -- "$(dirname -- "$OUT")"
    mkdir -- "$JOB"
    hostname > "$JOB/host"
    bash "$SCRIPT_DIR/run_state_interaction_support_audit.sh" _worker "$RUN_NAME" "$PYTHON_BIN" "$FIT" "$SOURCE" 2>&1 | tee "$JOB/run.log"
    ;;
  _worker)
    PYTHON_BIN=$3
    FIT=$4
    SOURCE=$5
    trap 'STATUS=$?; printf "%s\n" "$STATUS" > "$JOB/exit_code"; printf "Workflow exit code: %s\nFinished: %s\n" "$STATUS" "$(date -Is)"' EXIT
    printf '%s\n' "$$" > "$JOB/pid"
    printf '%s\n' "${SLURM_JOB_ID:-unset}" > "$JOB/slurm_job_id"
    export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3 NUMEXPR_NUM_THREADS=3
    printf 'Started: %s\nHost: %s\nSlurm job: %s\nPython: %s\nDevice: CPU (NumPy)\nFit source: %s\nModel source: %s\n' \
      "$(date -Is)" "$(hostname)" "${SLURM_JOB_ID:-unset}" "$PYTHON_BIN" "$FIT" "$SOURCE"
    git log -1 --format='commit=%H'
    "$PYTHON_BIN" -c 'import numpy; print("NumPy:", numpy.__version__)'
    "$PYTHON_BIN" -m unittest discover -s tests -p 'test_state_interaction_support*.py' -v
    echo 'Preflight passed; analyzing saved errors and report-time states. No training, GPU inference or model selection.'
    time "$PYTHON_BIN" -u "$SCRIPT_DIR/audit_state_interaction_support.py" run \
      --fit-dir "$FIT" \
      --source-dir "$SOURCE" \
      --data-dir ../data/chronological/Contra_Costa_v8_dev \
      --output "$OUT"
    ;;
  status)
    if [[ ! -d "$JOB" ]]; then echo "No run record: $JOB" >&2; exit 1; fi
    printf 'Run host: %s\nCurrent host: %s\n' "$(< "$JOB/host")" "$(hostname)"
    if [[ -f "$JOB/slurm_job_id" ]]; then printf 'Slurm job: %s\n' "$(< "$JOB/slurm_job_id")"; fi
    if [[ -f "$JOB/exit_code" ]]; then
      printf 'Workflow exit code: %s\n' "$(< "$JOB/exit_code")"
    elif [[ "$(< "$JOB/host")" == "$(hostname)" && -f "$JOB/pid" ]]; then
      ps -o pid,lstart,etime,%cpu,%mem,stat,cmd -p "$(< "$JOB/pid")" || true
    fi
    if [[ -f "$JOB/run.log" ]]; then tail -n 20 "$JOB/run.log"; fi
    if [[ -f "$OUT/summary.json" ]]; then
      printf 'Summary: %s\n' "$OUT/summary.json"
    else
      printf 'No completed summary.json yet; inspect the exit code and log.\n'
      if [[ -d "${OUT}.partial" ]]; then printf 'Partial: %s\n' "${OUT}.partial"; fi
    fi
    ;;
  report)
    if [[ ! -f "$OUT/summary.json" ]]; then
      printf 'No completed result: %s\nUse status to inspect the exit code and log; after interruption use a new run name.\n' "$OUT/summary.json" >&2
      exit 1
    fi
    python "$SCRIPT_DIR/audit_state_interaction_support.py" report "$OUT/summary.json"
    ;;
  help)
    echo 'Usage: bash experiments/chronological/run_state_interaction_support_audit.sh {run|status|report} [contra_v12j_support_audit_01]'
    echo 'Source overrides: run contra_v12j_support_audit_02 contra_v12i_full_fit_audit_01 contra_v12f_state_interaction_01'
    echo 'run is a foreground CPU NumPy diagnostic; it reads completed v12i/v12f artifacts without changing them.'
    echo 'No training, GPU, check or resume entry point; after interruption use a new run name.'
    ;;
  *) echo "Unknown action: $ACTION" >&2; exit 2 ;;
esac
