#!/usr/bin/env bash
# Foreground JSON-only CPU audit of an already completed v12k run.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd -- "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_DIR"
ACTION=${1:-help}
RUN_NAME=${2:-contra_v12l_selection_audit_01}
if [[ ! "$RUN_NAME" =~ ^contra_v12l_selection_audit_[A-Za-z0-9_-]+$ ]]; then
  echo 'Run name must match contra_v12l_selection_audit_01.' >&2
  exit 2
fi
OUT="$REPO_DIR/experiments/chronological_runs/$RUN_NAME"
JOB="${OUT}.job"

case "$ACTION" in
  run)
    if [[ $# -gt 3 ]]; then echo 'run accepts only run name and v12k source name.' >&2; exit 2; fi
    SOURCE_NAME=${3:-contra_v12k_objective_alignment_01}
    if [[ ! "$SOURCE_NAME" =~ ^contra_v12k_objective_alignment_[A-Za-z0-9_-]+$ ]]; then
      echo 'Source must be a completed contra_v12k_objective_alignment_* run.' >&2
      exit 2
    fi
    SOURCE="$REPO_DIR/experiments/chronological_runs/$SOURCE_NAME"
    if [[ ! -d "$SOURCE" || -L "$SOURCE" ]]; then echo "Missing regular source directory: $SOURCE" >&2; exit 1; fi
    SOURCE_FILES=(summary.json run_identity.json selected_endpoints_frozen.json)
    for ARM in state_vector interaction_vector; do
      for LOSS in global candidate_early; do
        for SEED in 2025 2026 2027; do
          FIT="${ARM}__loss_${LOSS}_s${SEED}"
          if [[ -L "$SOURCE/$FIT" ]]; then echo "Symlinked fit directory: $FIT" >&2; exit 1; fi
          SOURCE_FILES+=("$FIT/history.json" "$FIT/fit_summary.json")
        done
      done
    done
    for FILE in "${SOURCE_FILES[@]}"; do
      if [[ ! -f "$SOURCE/$FILE" || -L "$SOURCE/$FILE" ]]; then
        printf 'Missing regular source JSON: %s\n' "$SOURCE/$FILE" >&2
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
    bash "$SCRIPT_DIR/run_vector_selection_audit.sh" _worker "$RUN_NAME" "$PYTHON_BIN" "$SOURCE" 2>&1 | tee "$JOB/run.log"
    ;;
  _worker)
    PYTHON_BIN=$3
    SOURCE=$4
    trap 'STATUS=$?; printf "%s\n" "$STATUS" > "$JOB/exit_code"; printf "Workflow exit code: %s\nFinished: %s\n" "$STATUS" "$(date -Is)"' EXIT
    printf '%s\n' "$$" > "$JOB/pid"
    printf '%s\n' "${SLURM_JOB_ID:-unset}" > "$JOB/slurm_job_id"
    printf 'Started: %s\nHost: %s\nSlurm job: %s\nPython: %s\nDevice: CPU (standard library)\nSource: %s\n' \
      "$(date -Is)" "$(hostname)" "${SLURM_JOB_ID:-unset}" "$PYTHON_BIN" "$SOURCE"
    git log -1 --format='commit=%H'
    "$PYTHON_BIN" -m unittest discover -s tests -p 'test_vector_selection_audit*.py' -v
    echo 'Preflight passed; replaying saved decisions. No training, inference or new model selection.'
    time "$PYTHON_BIN" -u "$SCRIPT_DIR/audit_vector_selection.py" run --source-dir "$SOURCE" --output "$OUT"
    ;;
  status)
    if [[ $# -gt 2 ]]; then exit 2; fi
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
      echo 'No completed summary yet; inspect exit code/log and preserve partial.'
    fi
    ;;
  report)
    if [[ $# -gt 2 ]]; then exit 2; fi
    if [[ ! -f "$OUT/summary.json" ]]; then
      echo 'No completed result; use status. After interruption choose a new run name.' >&2
      exit 1
    fi
    python "$SCRIPT_DIR/audit_vector_selection.py" report "$OUT/summary.json"
    ;;
  help)
    echo 'Usage: bash experiments/chronological/run_vector_selection_audit.sh {run|status|report} [contra_v12l_selection_audit_01]'
    echo 'Source override: run contra_v12l_selection_audit_02 contra_v12k_objective_alignment_01'
    echo 'Foreground CPU standard-library audit. Completed source stays read-only. No GPU/data arrays/checkpoint reads.'
    ;;
  *) echo "Unknown action: $ACTION" >&2; exit 2 ;;
esac
