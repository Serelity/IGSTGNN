#!/usr/bin/env bash
# M4.1 original TRAIN-X input audit and optional information profiles. No Git.
set -euo pipefail
if [[ $# -eq 1 && ( "$1" == --help || "$1" == -h ) ]]; then
    echo 'Usage: bash experiments/chronological/run_incident_capacity_network.sh [original_v11a_history_directory]'
    echo 'M41_PYTHON, M41_DEVICE, M41_DATA_DIR, M41_SENSORS, M41_HISTORY_DIR may override defaults.'
    echo 'Uses bundled train report locations; no downloads, targets, val/test or real training.'
    exit 0
fi
[[ $# -le 1 && ( $# -eq 0 || "$1" != -* ) ]] || { echo 'Use --help.' >&2; exit 2; }
if [[ -n "${M41_REPO_DIR:-}" ]]; then
    REPO_DIR=$(cd -- "$M41_REPO_DIR" && pwd)
elif [[ -n "${SLURM_JOB_ID:-}" ]]; then
    REPO_DIR=$(pwd)
else
    SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
    REPO_DIR=$(cd -- "$SCRIPT_DIR/../.." && pwd)
fi
[[ -f "$REPO_DIR/src/models/igstgnn.py" && -f "$REPO_DIR/experiments/chronological/prepare_incident_capacity_network.py" ]] || {
    printf 'Invalid repository directory: %s. Set M41_REPO_DIR or submit with --chdir.\n' "$REPO_DIR" >&2; exit 1;
}
cd "$REPO_DIR"
printf 'Repository directory: %s\n' "$REPO_DIR"
PYTHON=${M41_PYTHON:-${M40_PYTHON:-python}}
DEVICE=${M41_DEVICE:-cpu}
DATA_DIR=${M41_DATA_DIR:-${M40_DATA_DIR:-../data/chronological/Contra_Costa_v8_dev}}
SENSORS=${M41_SENSORS:-${M40_SENSORS:-../data/xtraffic/Contra_Costa/sensors.csv}}
HISTORY_DIR=${1:-${M41_HISTORY_DIR:-${M40_HISTORY_DIR:-}}}
export PYTHONUTF8=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=3 CUBLAS_WORKSPACE_CONFIG=:4096:8
RUNS_DIR="$REPO_DIR/experiments/chronological_runs"
mkdir -p "$RUNS_DIR"
[[ -w "$RUNS_DIR" ]] || { printf 'Output directory not writable: %s\n' "$RUNS_DIR" >&2; exit 1; }
RUN_DIR=$(mktemp -d "$RUNS_DIR/contra_network_m41_$(date +%Y%m%d_%H%M%S)_XXXXXX")
exec > >(tee "$RUN_DIR/run.log") 2>&1
trap 'STATUS=$?; printf "%s\n" "$STATUS" > "$RUN_DIR/exit_code"; printf "Workflow exit code: %s\nOutput directory: %s\n" "$STATUS" "$RUN_DIR"' EXIT
printf 'Started: %s\nHost: %s\nSlurm job: %s\nDevice: %s\nOutput directory: %s\n' "$(date -Is)" "$(hostname)" "${SLURM_JOB_ID:-unset}" "$DEVICE" "$RUN_DIR"
"$PYTHON" -m unittest discover -s tests -p 'test_capacity_network_inputs.py' -v
"$PYTHON" -m unittest discover -s tests -p 'test_incident_capacity_exchange.py' -v
"$PYTHON" -m unittest discover -s tests -p 'test_incident_capacity_fusion.py' -v
"$PYTHON" -m unittest discover -s tests -p 'test_capacity_fusion_inputs.py' -v
[[ -d "$DATA_DIR" && -f "$SENSORS" ]] || { echo 'Set M41_DATA_DIR and M41_SENSORS to the existing original data.' >&2; exit 1; }
if [[ -z "$HISTORY_DIR" ]]; then
    HISTORY_DIR=$("$PYTHON" - "$DATA_DIR" <<'PY'
from pathlib import Path
import sys
from experiments.chronological.prepare_incident_corridors import discover_history
from src.utils.incident_corridor import sha256
data = Path(sys.argv[1])
found = discover_history([data, data.parent, Path('experiments/chronological_runs'),
                          Path('../论文学习/研究开发_20260911')], sha256(data/'summary.json'))
print(str(found) if found else '')
PY
)
fi
[[ -n "$HISTORY_DIR" && -d "$HISTORY_DIR" ]] || { echo 'Set M41_HISTORY_DIR to the original v11a history directory.' >&2; exit 1; }
printf 'History directory: %s\n' "$HISTORY_DIR"
"$PYTHON" -u experiments/chronological/prepare_incident_capacity_network.py \
    --data-dir "$DATA_DIR" --history-dir "$HISTORY_DIR" --sensors "$SENSORS" \
    --output-dir "$RUN_DIR/pack" --device "$DEVICE"
