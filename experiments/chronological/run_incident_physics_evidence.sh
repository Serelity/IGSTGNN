#!/usr/bin/env bash
# CPU evidence preparation, optionally followed by a caller-supplied source comparison.
set -euo pipefail

if [[ $# -eq 1 && ( "$1" == --help || "$1" == -h ) ]]; then
    echo 'Usage: bash experiments/chronological/run_incident_physics_evidence.sh [official_station_5min.txt.gz]'
    echo 'Prepare metadata checks, unique train-X profiles, and exact source-record request dates.'
    echo 'Optional input: compare official Station 5-Minute records against train X.'
    echo 'No full training, downloads, installs, or scheduler submission; CPU only.'
    echo 'PHYSICS_DATA_DIR, PHYSICS_SENSORS and PHYSICS_PYTHON may be set.'
    exit 0
fi
if [[ $# -gt 1 || ( $# -eq 1 && "$1" == -* ) ]]; then
    echo 'Expected at most one source-file path; use --help.' >&2
    exit 2
fi
SOURCE_FILE=${1:-}
if [[ -n "$SOURCE_FILE" ]]; then
    [[ -f "$SOURCE_FILE" ]] || { printf 'Missing source file: %s\n' "$SOURCE_FILE" >&2; exit 2; }
    SOURCE_FILE=$(cd -- "$(dirname -- "$SOURCE_FILE")" && printf '%s/%s' "$PWD" "$(basename -- "$SOURCE_FILE")")
fi
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd -- "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_DIR"

if [[ -z "${PHYSICS_PYTHON:-}" ]]; then
    if [[ "${CONDA_DEFAULT_ENV:-}" != igstgnn ]]; then
        if ! command -v conda >/dev/null 2>&1; then
            echo 'Activate igstgnn or set PHYSICS_PYTHON to its Python executable.' >&2
            exit 1
        fi
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate igstgnn
    fi
    PYTHON="${CONDA_PREFIX:?Activate igstgnn}/bin/python"
else
    PYTHON=$PHYSICS_PYTHON
fi
export OMP_NUM_THREADS=3
export PYTHONUNBUFFERED=1
export PYTHONUTF8=1
DATA_DIR=${PHYSICS_DATA_DIR:-../data/chronological/Contra_Costa_v8_dev}
SENSORS=${PHYSICS_SENSORS:-../data/xtraffic/Contra_Costa/sensors.csv}
[[ -d "$DATA_DIR" ]] || { printf 'Missing data: %s\n' "$DATA_DIR" >&2; exit 1; }
[[ -f "$SENSORS" ]] || { printf 'Missing sensors: %s\n' "$SENSORS" >&2; exit 1; }
mkdir -p experiments/chronological_runs
RUN_DIR=$(mktemp -d "$REPO_DIR/experiments/chronological_runs/contra_physics_evidence_$(date +%Y%m%d_%H%M%S)_XXXXXX")
exec > >(tee "$RUN_DIR/run.log") 2>&1
trap 'STATUS=$?; printf "%s\n" "$STATUS" > "$RUN_DIR/exit_code"; printf "Workflow exit code: %s\nOutput directory: %s\n" "$STATUS" "$RUN_DIR"' EXIT
printf 'Started: %s\nHost: %s\nSlurm job: %s\nOutput directory: %s\n' "$(date -Is)" "$(hostname)" "${SLURM_JOB_ID:-unset}" "$RUN_DIR"

"$PYTHON" -m unittest discover -s tests -p 'test_incident_physics_evidence*.py' -v
ARGS=(--data-dir "$DATA_DIR" --published-sensors "$SENSORS"
      --metadata-bundle "$SCRIPT_DIR/physics_metadata" --output-dir "$RUN_DIR/evidence")
if [[ -n "$SOURCE_FILE" ]]; then ARGS+=(--pems-5min "$SOURCE_FILE"); fi
"$PYTHON" -u experiments/chronological/prepare_incident_physics_evidence.py "${ARGS[@]}"
"$PYTHON" -u - "$RUN_DIR" <<'PY'
import json
from pathlib import Path
import sys
root = Path(sys.argv[1])
report = json.loads((root / 'evidence/summary.json').read_text(encoding='utf-8'))
request = json.loads((root / 'evidence/source_record_request.json').read_text(encoding='utf-8'))
compact = {key: report[key] for key in (
    'status', 'readiness', 'main_training_ready', 'training_started',
    'unique_train_x_nominal_slots', 'candidate_pairs', 'metadata_prefilter_pass_pairs',
    'historical_lane_road_coordinate_matches', 'optional_source_record_comparison')}
compact['suggested_pems_dates'] = [row['date'] for row in request['suggested_dates']]
compact['source_request'] = str(root / 'evidence/source_record_request.json')
compact['output_dir'] = str(root)
(root / 'summary.json').write_text(json.dumps(compact, indent=2) + '\n', encoding='utf-8')
print(json.dumps(compact, indent=2))
PY
