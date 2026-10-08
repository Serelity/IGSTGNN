#!/usr/bin/env bash
# Continue the existing pair inside a GPU job. Update the repository separately.
set -euo pipefail

if [[ $# -eq 1 && ( "$1" == --help || "$1" == -h ) ]]; then
    echo 'Usage: bash experiments/chronological/resume_incident_routing_p2.sh RUN_NAME'
    echo 'RUN_NAME is the existing contra_p2_first_epoch_* directory name.'
    echo 'Continues fixed/acdg to the original early stop or 100-epoch limit.'
    echo 'Completed variants are verified and skipped. No Git commands are run.'
    exit 0
fi
if [[ $# -ne 1 || ! "$1" =~ ^contra_p2_first_epoch_[A-Za-z0-9_]+$ ]]; then
    echo 'Pass the existing contra_p2_first_epoch_* run name, without a path.' >&2
    exit 2
fi
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd -- "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_DIR"
RUN_ROOT="$REPO_DIR/experiments/chronological_runs/$1"
if [[ ! -d "$RUN_ROOT" ]]; then
    printf 'Existing run not found: %s\n' "$RUN_ROOT" >&2
    exit 1
fi
if [[ "${CONDA_DEFAULT_ENV:-}" != igstgnn ]]; then
    if ! command -v conda >/dev/null 2>&1; then
        echo 'Activate the existing igstgnn environment before running.' >&2
        exit 1
    fi
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate igstgnn
fi
export PATH="${CONDA_PREFIX:?Activate igstgnn}/bin:$PATH"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=3
export PYTHONUNBUFFERED=1

# One writer per pair; flock releases the lock even if Slurm kills the job.
command -v flock >/dev/null 2>&1 || { echo 'Missing flock utility.' >&2; exit 1; }
exec 9>"$RUN_ROOT/.resume.lock"
flock -n 9 || { echo 'This pair already has a running continuation.' >&2; exit 1; }
SESSION_DIR=$(mktemp -d "$RUN_ROOT/resume_$(date +%Y%m%d_%H%M%S)_XXXXXX")
exec > >(tee "$SESSION_DIR/run.log") 2>&1
trap 'STATUS=$?; printf "%s\n" "$STATUS" > "$SESSION_DIR/exit_code"; printf "Workflow exit code: %s\nSession directory: %s\n" "$STATUS" "$SESSION_DIR"' EXIT
printf 'Started: %s\nHost: %s\nSlurm job: %s\nPair directory: %s\n' \
    "$(date -Is)" "$(hostname)" "${SLURM_JOB_ID:-unset}" "$RUN_ROOT"
python -u experiments/chronological/continue_incident_routing_p2.py \
    --run-dir "$RUN_ROOT" \
    --data-dir "${P2_DATA_DIR:-../data/chronological/Contra_Costa_v8_dev}"
