#!/usr/bin/env bash
# M2.1 engineering only; run inside the existing Python environment/allocation.
set -euo pipefail
if [[ $# -eq 1 && ( "$1" == --help || "$1" == -h ) ]]; then
    echo 'Usage: bash experiments/chronological/run_incident_relative_capacity.sh [CANDIDATE_X_DIR]'
    echo 'Always runs module tests, synthetic supply cases, gradients and checkpoint checks.'
    echo 'Optional directory enables all-event/all-node train-X interface checks without real-data training.'
    echo 'CAPACITY_PYTHON defaults to python; CAPACITY_DEVICE defaults to cpu.'
    exit 0
fi
[[ $# -le 1 ]] || { echo 'Use --help for usage.' >&2; exit 2; }
CANDIDATE=${1:-}
if [[ -n "$CANDIDATE" ]]; then
    [[ -f "$CANDIDATE/summary.json" ]] || { echo 'Candidate X summary missing.' >&2; exit 2; }
    CANDIDATE=$(cd -- "$CANDIDATE" && pwd)
fi
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd -- "$SCRIPT_DIR/../.." && pwd)
PYTHON=${CAPACITY_PYTHON:-python}
DEVICE=${CAPACITY_DEVICE:-cpu}
cd "$REPO_DIR"
export PYTHONUTF8=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=3 CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p experiments/chronological_runs
RUN_DIR=$(mktemp -d "$REPO_DIR/experiments/chronological_runs/contra_capacity_m21_$(date +%Y%m%d_%H%M%S)_XXXXXX")
exec > >(tee "$RUN_DIR/run.log") 2>&1
trap 'STATUS=$?; printf "%s\n" "$STATUS" > "$RUN_DIR/exit_code"; printf "Workflow exit code: %s\nOutput directory: %s\n" "$STATUS" "$RUN_DIR"' EXIT
printf 'Started: %s\nHost: %s\nSlurm job: %s\nOutput directory: %s\n' "$(date -Is)" "$(hostname)" "${SLURM_JOB_ID:-unset}" "$RUN_DIR"
"$PYTHON" -m unittest discover -s tests -p 'test_incident_relative_capacity.py' -v
ARGS=(--device "$DEVICE" --output-dir "$RUN_DIR/check")
if [[ -n "$CANDIDATE" ]]; then ARGS+=(--candidate-dir "$CANDIDATE"); fi
"$PYTHON" -u experiments/chronological/check_incident_relative_capacity.py "${ARGS[@]}"
