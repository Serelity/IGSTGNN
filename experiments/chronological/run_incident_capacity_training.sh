#!/usr/bin/env bash
# M4.2 paired exploratory training. Run inside an allocated GPU job.
set -euo pipefail
if [[ $# -eq 1 && ( "$1" == --help || "$1" == -h ) ]]; then
    echo 'Usage: bash experiments/chronological/run_incident_capacity_training.sh [original_v11a_history_directory]'
    echo 'Defaults: six arms, seed 11, one full epoch, original train/val, test sealed.'
    echo 'M42_EPOCHS is the absolute stopping epoch; M42_RESUME_RUN resumes an existing run.'
    echo 'M42_CHECK=1 uses first four train/val rows for engineering only.'
    exit 0
fi
[[ $# -le 1 && ( $# -eq 0 || "$1" != -* ) ]] || { echo 'Use --help.' >&2; exit 2; }
if [[ -n "${M42_REPO_DIR:-}" ]]; then
    REPO_DIR=$(cd -- "$M42_REPO_DIR" && pwd)
elif [[ -n "${SLURM_JOB_ID:-}" ]]; then
    REPO_DIR=$(pwd)
else
    SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
    REPO_DIR=$(cd -- "$SCRIPT_DIR/../.." && pwd)
fi
[[ -f "$REPO_DIR/src/models/igstgnn.py" && -f "$REPO_DIR/experiments/chronological/train_incident_capacity.py" ]] || {
    printf 'Invalid repository directory: %s. Set M42_REPO_DIR or submit with --chdir.\n' "$REPO_DIR" >&2; exit 1;
}
cd "$REPO_DIR"
printf 'Repository directory: %s\n' "$REPO_DIR"
PYTHON=${M42_PYTHON:-${M41_PYTHON:-${M40_PYTHON:-python}}}
DEVICE=${M42_DEVICE:-cuda:0}
DATA_DIR=${M42_DATA_DIR:-${M41_DATA_DIR:-${M40_DATA_DIR:-../data/chronological/Contra_Costa_v8_dev}}}
SENSORS=${M42_SENSORS:-${M41_SENSORS:-${M40_SENSORS:-../data/xtraffic/Contra_Costa/sensors.csv}}}
HISTORY_DIR=${1:-${M42_HISTORY_DIR:-${M41_HISTORY_DIR:-${M40_HISTORY_DIR:-}}}}
export PYTHONUTF8=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=3 CUBLAS_WORKSPACE_CONFIG=:4096:8
RUNS_DIR="$REPO_DIR/experiments/chronological_runs"
mkdir -p "$RUNS_DIR"
[[ -w "$RUNS_DIR" ]] || { printf 'Output directory not writable: %s\n' "$RUNS_DIR" >&2; exit 1; }
RESUME_ARGS=()
if [[ -n "${M42_RESUME_RUN:-}" ]]; then
    [[ -f "$M42_RESUME_RUN/identity.json" ]] || { echo 'Resume directory has no M4.2 identity.' >&2; exit 1; }
    RUN_DIR=$(cd -- "$M42_RESUME_RUN" && pwd)
    RESUME_ARGS+=(--resume)
else
    RUN_DIR=$(mktemp -d "$RUNS_DIR/contra_training_m42_$(date +%Y%m%d_%H%M%S)_XXXXXX")
fi
exec > >(tee -a "$RUN_DIR/run.log") 2>&1
trap 'STATUS=$?; printf "%s\n" "$STATUS" > "$RUN_DIR/exit_code"; printf "Workflow exit code: %s\nOutput directory: %s\n" "$STATUS" "$RUN_DIR"' EXIT
printf 'Started: %s\nHost: %s\nSlurm job: %s\nDevice: %s\nOutput directory: %s\n' "$(date -Is)" "$(hostname)" "${SLURM_JOB_ID:-unset}" "$DEVICE" "$RUN_DIR"
"$PYTHON" -m unittest discover -s tests -p 'test_capacity_training.py' -v
"$PYTHON" -m unittest discover -s tests -p 'test_incident_capacity_exchange.py' -v
"$PYTHON" -m unittest discover -s tests -p 'test_incident_capacity_fusion.py' -v
"$PYTHON" -m unittest discover -s tests -p 'test_capacity_network_inputs.py' -v
"$PYTHON" -m unittest discover -s tests -p 'test_capacity_fusion_inputs.py' -v
[[ -d "$DATA_DIR" && -f "$SENSORS" ]] || { echo 'Set M42_DATA_DIR and M42_SENSORS to the existing original data.' >&2; exit 1; }
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
[[ -n "$HISTORY_DIR" && -d "$HISTORY_DIR" ]] || { echo 'Set M42_HISTORY_DIR to the original v11a history directory.' >&2; exit 1; }
printf 'History directory: %s\n' "$HISTORY_DIR"
EXTRA_ARGS=()
[[ "${M42_CHECK:-0}" == 0 ]] || EXTRA_ARGS+=(--check)
[[ "${M42_RAMPS:-1}" == 1 ]] || EXTRA_ARGS+=(--without-ramp-exchanges)
[[ -z "${M42_NETWORK_DIR:-}" ]] || EXTRA_ARGS+=(--network-dir "$M42_NETWORK_DIR")
read -r -a GROUPS_TO_RUN <<< "${M42_GROUPS:-F N0 N1 P0 P1 L1}"
# mktemp reserves the directory; Python accepts this explicitly empty run root.
"$PYTHON" -u experiments/chronological/train_incident_capacity.py \
    --data-dir "$DATA_DIR" --history-dir "$HISTORY_DIR" --sensors "$SENSORS" \
    --output-dir "$RUN_DIR" --device "$DEVICE" --allow-exploratory \
    --seed "${M42_SEED:-11}" --epochs "${M42_EPOCHS:-1}" --groups "${GROUPS_TO_RUN[@]}" \
    --launcher-created-root "${RESUME_ARGS[@]}" "${EXTRA_ARGS[@]}"
