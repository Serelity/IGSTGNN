#!/usr/bin/env bash
# Inference only, inside an allocated V100 job. Update the repository separately.
set -euo pipefail
if [[ $# -eq 1 && ( "$1" == --help || "$1" == -h ) ]]; then
    echo 'Usage: bash experiments/chronological/probe_incident_routing_p2_layers.sh RUN_NAME'
    echo 'Runs five independent single-layer gate-OFF cases at the existing best ACDG checkpoint.'
    echo 'All-ON replay before and after. Original igstgnn/V100 environment; no training or Git operations.'
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
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=3
export PYTHONUNBUFFERED=1
SESSION_DIR=$(mktemp -d "$RUN_ROOT/gate_layerwise_$(date +%Y%m%d_%H%M%S)_XXXXXX")
exec > >(tee "$SESSION_DIR/run.log") 2>&1
trap 'STATUS=$?; printf "%s\n" "$STATUS" > "$SESSION_DIR/exit_code"; printf "Workflow exit code: %s\nSession directory: %s\n" "$STATUS" "$SESSION_DIR"' EXIT
printf 'Started: %s\nHost: %s\nSlurm job: %s\nPair directory: %s\n' \
    "$(date -Is)" "$(hostname)" "${SLURM_JOB_ID:-unset}" "$RUN_ROOT"
python -m unittest discover -s tests -p 'test_incident_routing_p2_probe*.py' -v
python -u experiments/chronological/probe_incident_routing_p2_layers.py \
    --run-dir "$RUN_ROOT" \
    --data-dir "${P2_DATA_DIR:-../data/chronological/Contra_Costa_v8_dev}" \
    --output-dir "$SESSION_DIR/report"
