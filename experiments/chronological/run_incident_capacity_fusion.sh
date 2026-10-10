#!/usr/bin/env bash
# M4.0 integration and original train-X checks. No Git or scheduler operations.
set -euo pipefail
if [[ $# -eq 1 && ( "$1" == --help || "$1" == -h ) ]]; then
    echo 'Usage: bash experiments/chronological/run_incident_capacity_fusion.sh [original_v11a_history_directory]'
    echo 'M40_PYTHON=python, M40_DEVICE=cpu; M40_DATA_DIR, M40_SENSORS, M40_HISTORY_DIR may be set.'
    echo 'M40_REPO_DIR overrides the root; Slurm jobs otherwise use their working directory.'
    echo 'Checks original 3604 train histories, not candidate expansion. No Y, val/test, downloads or training.'
    exit 0
fi
[[ $# -le 1 && ( $# -eq 0 || "$1" != -* ) ]] || { echo 'Use --help.' >&2; exit 2; }
if [[ -n "${M40_REPO_DIR:-}" ]]; then
    REPO_DIR=$(cd -- "$M40_REPO_DIR" && pwd)
elif [[ -n "${SLURM_JOB_ID:-}" ]]; then
    # sbatch --chdir selects the repository; BASH_SOURCE points into Slurm's spool.
    REPO_DIR=$(pwd)
else
    SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
    REPO_DIR=$(cd -- "$SCRIPT_DIR/../.." && pwd)
fi
[[ -f "$REPO_DIR/src/models/igstgnn.py" && -f "$REPO_DIR/experiments/chronological/check_incident_capacity_fusion.py" ]] || {
    printf 'Invalid repository directory: %s. Set M40_REPO_DIR to the code repository or submit with --chdir.\n' "$REPO_DIR" >&2
    exit 1
}
cd "$REPO_DIR"
printf 'Repository directory: %s\n' "$REPO_DIR"
PYTHON=${M40_PYTHON:-python}
DEVICE=${M40_DEVICE:-cpu}
DATA_DIR=${M40_DATA_DIR:-../data/chronological/Contra_Costa_v8_dev}
SENSORS=${M40_SENSORS:-../data/xtraffic/Contra_Costa/sensors.csv}
HISTORY_DIR=${1:-${M40_HISTORY_DIR:-}}
export PYTHONUTF8=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=3 CUBLAS_WORKSPACE_CONFIG=:4096:8
RUNS_DIR="$REPO_DIR/experiments/chronological_runs"
mkdir -p "$RUNS_DIR" || { printf 'Cannot create output directory: %s\n' "$RUNS_DIR" >&2; exit 1; }
[[ -w "$RUNS_DIR" ]] || { printf 'Output directory is not writable: %s\n' "$RUNS_DIR" >&2; exit 1; }
RUN_DIR=$(mktemp -d "$RUNS_DIR/contra_fusion_m40_$(date +%Y%m%d_%H%M%S)_XXXXXX")
exec > >(tee "$RUN_DIR/run.log") 2>&1
trap 'STATUS=$?; printf "%s\n" "$STATUS" > "$RUN_DIR/exit_code"; printf "Workflow exit code: %s\nOutput directory: %s\n" "$STATUS" "$RUN_DIR"' EXIT
printf 'Started: %s\nHost: %s\nSlurm job: %s\nDevice: %s\nOutput directory: %s\n' "$(date -Is)" "$(hostname)" "${SLURM_JOB_ID:-unset}" "$DEVICE" "$RUN_DIR"
"$PYTHON" -m unittest discover -s tests -p 'test_incident_*capacity*.py' -v
"$PYTHON" -m unittest discover -s tests -p 'test_capacity_fusion_inputs.py' -v
"$PYTHON" -m unittest discover -s tests -p 'test_acdg.py' -v
"$PYTHON" -m unittest discover -s tests -p 'test_incident_response.py' -v
"$PYTHON" -m unittest discover -s tests -p 'test_paper_modules.py' -v
[[ -d "$DATA_DIR" && -f "$SENSORS" ]] || { echo 'Set M40_DATA_DIR and M40_SENSORS to existing original data.' >&2; exit 1; }
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
[[ -n "$HISTORY_DIR" && -d "$HISTORY_DIR" ]] || {
    echo 'Original v11a history not found. Set M40_HISTORY_DIR to the existing directory containing train_history.npy.' >&2
    exit 1
}
printf 'History directory: %s\n' "$HISTORY_DIR"
"$PYTHON" -u experiments/chronological/check_incident_capacity_fusion.py \
    --device "$DEVICE" --output-dir "$RUN_DIR/check" --data-dir "$DATA_DIR" \
    --history-dir "$HISTORY_DIR" --sensors "$SENSORS"
