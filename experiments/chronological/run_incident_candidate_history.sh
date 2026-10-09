#!/usr/bin/env bash
# CPU-only candidate X preparation in the user's existing environment/allocation.
set -euo pipefail
if [[ $# -eq 1 && ( "$1" == --help || "$1" == -h ) ]]; then
    echo 'Usage: bash experiments/chronological/run_incident_candidate_history.sh DATA_DIR AUDIT_DIR RAW_EVENTS'
    echo 'AUDIT_DIR must contain the previously audited candidate_events.csv.'
    echo 'Use the existing Python environment; optionally set CANDIDATE_PYTHON and CANDIDATE_SENSORS.'
    echo 'Builds train X only from the existing local row cache; CPU only.'
    exit 0
fi
if [[ $# -ne 3 ]]; then
    echo 'Expected DATA_DIR AUDIT_DIR RAW_EVENTS; use --help.' >&2
    exit 2
fi
[[ -d "$1" && -d "$2" && -f "$3" ]] || { echo 'Input directory or raw event file missing.' >&2; exit 2; }
DATA_DIR=$(cd -- "$1" && pwd)
AUDIT_DIR=$(cd -- "$2" && pwd)
RAW_EVENTS=$(cd -- "$(dirname -- "$3")" && pwd)/$(basename -- "$3")
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd -- "$SCRIPT_DIR/../.." && pwd)
PYTHON=${CANDIDATE_PYTHON:-python}
SENSORS=${CANDIDATE_SENSORS:-$REPO_DIR/../data/xtraffic/Contra_Costa/sensors.csv}
[[ -f "$SENSORS" ]] || { echo 'Sensor table missing; set CANDIDATE_SENSORS.' >&2; exit 2; }
SENSORS=$(cd -- "$(dirname -- "$SENSORS")" && pwd)/$(basename -- "$SENSORS")
cd "$REPO_DIR"
export PYTHONUTF8=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=3
mkdir -p experiments/chronological_runs
RUN_DIR=$(mktemp -d "$REPO_DIR/experiments/chronological_runs/contra_candidate_history_$(date +%Y%m%d_%H%M%S)_XXXXXX")
exec > >(tee "$RUN_DIR/run.log") 2>&1
trap 'STATUS=$?; printf "%s\n" "$STATUS" > "$RUN_DIR/exit_code"; printf "Workflow exit code: %s\nOutput directory: %s\n" "$STATUS" "$RUN_DIR"' EXIT
printf 'Started: %s\nHost: %s\nSlurm job: %s\nOutput directory: %s\n' "$(date -Is)" "$(hostname)" "${SLURM_JOB_ID:-unset}" "$RUN_DIR"
"$PYTHON" -m unittest discover -s tests -p 'test_incident_candidate_history.py' -v
"$PYTHON" -u experiments/chronological/prepare_incident_candidate_history.py \
    --data-dir "$DATA_DIR" --audit-dir "$AUDIT_DIR" --raw-events "$RAW_EVENTS" \
    --sensors "$SENSORS" --output-dir "$RUN_DIR/candidates"
