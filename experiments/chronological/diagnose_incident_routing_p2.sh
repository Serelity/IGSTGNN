#!/usr/bin/env bash
# Analyze saved predictions on CPU; no training or repository updates.
set -euo pipefail
if [[ $# -eq 1 && ( "$1" == --help || "$1" == -h ) ]]; then
    echo 'Usage: bash experiments/chronological/diagnose_incident_routing_p2.sh RUN_NAME'
    echo 'Reads the completed pair; creates a new diagnostics directory. CPU and NumPy only.'
    exit 0
fi
if [[ $# -ne 1 || ! "$1" =~ ^contra_p2_first_epoch_[A-Za-z0-9_]+$ ]]; then
    echo 'Pass the existing contra_p2_first_epoch_* directory name, without a path.' >&2
    exit 2
fi
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd -- "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_DIR"
RUN_ROOT="$REPO_DIR/experiments/chronological_runs/$1"
[[ -d "$RUN_ROOT" ]] || { echo "Missing pair directory: $RUN_ROOT" >&2; exit 1; }
if [[ "${CONDA_DEFAULT_ENV:-}" != igstgnn ]]; then
    command -v conda >/dev/null 2>&1 || { echo 'Activate igstgnn first.' >&2; exit 1; }
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate igstgnn
fi
export PATH="${CONDA_PREFIX:?Activate igstgnn}/bin:$PATH"
export OMP_NUM_THREADS=3
export PYTHONUNBUFFERED=1
SESSION_DIR=$(mktemp -d "$RUN_ROOT/diagnostics_$(date +%Y%m%d_%H%M%S)_XXXXXX")
exec > >(tee "$SESSION_DIR/run.log") 2>&1
trap 'STATUS=$?; printf "%s\n" "$STATUS" > "$SESSION_DIR/exit_code"; printf "Workflow exit code: %s\nSession directory: %s\n" "$STATUS" "$SESSION_DIR"' EXIT
printf 'Started: %s\nHost: %s\nSlurm job: %s\nPair directory: %s\n' \
    "$(date -Is)" "$(hostname)" "${SLURM_JOB_ID:-unset}" "$RUN_ROOT"
python -u experiments/chronological/diagnose_incident_routing_p2.py \
    --run-dir "$RUN_ROOT" \
    --data-dir "${P2_DATA_DIR:-../data/chronological/Contra_Costa_v8_dev}" \
    --output-dir "$SESSION_DIR/report"
