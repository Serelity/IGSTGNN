#!/usr/bin/env bash
# Prepare candidate corridor inputs with CPU resources in an existing allocation.
set -euo pipefail

if [[ $# -eq 1 && ( "$1" == --help || "$1" == -h ) ]]; then
    echo 'Usage: bash experiments/chronological/run_incident_corridors.sh [v11a_history_directory]'
    echo 'CPU only. Reuse v11a histories, or materialize them from the existing local row cache.'
    echo 'No training, downloads, installs or scheduler submission.'
    echo 'PHYSICS_DATA_DIR, PHYSICS_HISTORY_DIR, PHYSICS_SENSORS, PHYSICS_PYTHON may be set.'
    exit 0
fi
if [[ $# -gt 1 || ( $# -eq 1 && "$1" == -* ) ]]; then
    echo 'Expected at most one history directory; use --help.' >&2
    exit 2
fi
HISTORY_DIR=${1:-${PHYSICS_HISTORY_DIR:-}}
if [[ -n "$HISTORY_DIR" ]]; then
    [[ -d "$HISTORY_DIR" ]] || { printf 'Missing history directory: %s\n' "$HISTORY_DIR" >&2; exit 2; }
    HISTORY_DIR=$(cd -- "$HISTORY_DIR" && pwd)
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
RUN_DIR=$(mktemp -d "$REPO_DIR/experiments/chronological_runs/contra_incident_corridors_$(date +%Y%m%d_%H%M%S)_XXXXXX")
exec > >(tee "$RUN_DIR/run.log") 2>&1
trap 'STATUS=$?; printf "%s\n" "$STATUS" > "$RUN_DIR/exit_code"; printf "Workflow exit code: %s\nOutput directory: %s\n" "$STATUS" "$RUN_DIR"' EXIT
printf 'Started: %s\nHost: %s\nSlurm job: %s\nOutput directory: %s\n' "$(date -Is)" "$(hostname)" "${SLURM_JOB_ID:-unset}" "$RUN_DIR"
"$PYTHON" -m unittest discover -s tests -p 'test_incident_corridor*.py' -v
MATERIALIZED=false
if [[ -z "$HISTORY_DIR" ]]; then
    HISTORY_DIR=$("$PYTHON" - "$DATA_DIR" <<'PY'
from pathlib import Path
import sys
from experiments.chronological.prepare_incident_corridors import discover_history
from src.utils.incident_corridor import sha256
data = Path(sys.argv[1])
roots = [data, data.parent, Path('experiments/chronological_runs'),
         Path('../论文学习/研究开发_20260911')]
found = discover_history(roots, sha256(data / 'summary.json'))
print(str(found) if found else '')
PY
)
fi
if [[ -z "$HISTORY_DIR" ]]; then
    [[ -d "$DATA_DIR/row_cache/blobs" ]] || {
        echo 'No completed v11a history or local row cache found. Supply the existing v11a directory as the sole argument.' >&2
        exit 1
    }
    HISTORY_DIR="$RUN_DIR/multichannel"
    echo 'Materializing v11a train/validation histories from the existing cache; corridor selection remains train-only.'
    "$PYTHON" -u experiments/chronological/materialize_multichannel_history.py \
        --data-dir "$DATA_DIR" --output "$HISTORY_DIR"
    MATERIALIZED=true
fi
printf 'History directory: %s\n' "$HISTORY_DIR"
"$PYTHON" -u experiments/chronological/prepare_incident_corridors.py \
    --data-dir "$DATA_DIR" --history-dir "$HISTORY_DIR" --sensors "$SENSORS" \
    --output-dir "$RUN_DIR/corridors"
"$PYTHON" - "$RUN_DIR" "$HISTORY_DIR" "$MATERIALIZED" <<'PY'
from pathlib import Path
import json
import sys
root = Path(sys.argv[1])
report = json.loads((root / 'corridors/summary.json').read_text(encoding='utf-8'))
result = {key: report[key] for key in ('status', 'samples', 'packed_stations', 'history_shape',
                                      'physics_ready', 'model_training_performed', 'corridors')}
result.update(history_directory=sys.argv[2], history_materialized_this_run=sys.argv[3] == 'true',
              output_dir=str(root / 'corridors'))
(root / 'summary.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
print(json.dumps(result, indent=2))
PY
