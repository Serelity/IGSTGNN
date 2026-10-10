#!/usr/bin/env bash
# M3.0 engineering verification within an existing environment/allocation.
set -euo pipefail
if [[ $# -eq 1 && ( "$1" == --help || "$1" == -h ) ]]; then
    echo 'Usage: bash experiments/chronological/run_incident_capacity_exchange.sh'
    echo 'Runs capacity regressions, directed-exchange tests and synthetic 496-axis checks.'
    echo 'No data package is required. This does not train IGSTGNN or certify real road edges.'
    echo 'EXCHANGE_PYTHON defaults to python; EXCHANGE_DEVICE defaults to cpu.'
    exit 0
fi
[[ $# -eq 0 ]] || { echo 'Use --help for usage.' >&2; exit 2; }
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd -- "$SCRIPT_DIR/../.." && pwd)
PYTHON=${EXCHANGE_PYTHON:-python}
DEVICE=${EXCHANGE_DEVICE:-cpu}
cd "$REPO_DIR"
export PYTHONUTF8=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=3 CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p experiments/chronological_runs
RUN_DIR=$(mktemp -d "$REPO_DIR/experiments/chronological_runs/contra_exchange_m30_$(date +%Y%m%d_%H%M%S)_XXXXXX")
exec > >(tee "$RUN_DIR/run.log") 2>&1
trap 'STATUS=$?; printf "%s\n" "$STATUS" > "$RUN_DIR/exit_code"; printf "Workflow exit code: %s\nOutput directory: %s\n" "$STATUS" "$RUN_DIR"' EXIT
printf 'Started: %s\nHost: %s\nSlurm job: %s\nOutput directory: %s\n' "$(date -Is)" "$(hostname)" "${SLURM_JOB_ID:-unset}" "$RUN_DIR"
"$PYTHON" -m unittest discover -s tests -p 'test_incident_*capacity*.py' -v
"$PYTHON" -u experiments/chronological/check_incident_capacity_exchange.py --device "$DEVICE" --output-dir "$RUN_DIR/check"
