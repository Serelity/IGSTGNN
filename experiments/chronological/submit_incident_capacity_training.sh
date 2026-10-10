#!/usr/bin/env bash
# Slurm entry; source updates are performed separately by the user.
set -euo pipefail
if [[ $# -eq 1 && ( "$1" == --help || "$1" == -h ) ]]; then
    echo 'Usage: bash experiments/chronological/submit_incident_capacity_training.sh [original_v11a_history_directory]'
    echo 'Submits six paired arms for one full epoch: one GPU, 3 CPUs, 2 hours.'
    echo 'M42_EPOCHS, M42_SEED, M42_RAMPS, M42_CHECK, M42_RESUME_RUN may override defaults.'
    exit 0
fi
[[ $# -le 1 && ( $# -eq 0 || "$1" != -* ) ]] || { echo 'Use --help.' >&2; exit 2; }
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd -- "$SCRIPT_DIR/../.." && pwd)
command -v sbatch >/dev/null 2>&1 || { echo 'Run on the Slurm login node.' >&2; exit 1; }
[[ -f "$REPO_DIR/src/models/igstgnn.py" && -w "$REPO_DIR" ]] || { printf 'Invalid/unwritable repository: %s\n' "$REPO_DIR" >&2; exit 1; }
mkdir -p "$REPO_DIR/experiments/chronological_runs"
[[ -w "$REPO_DIR/experiments/chronological_runs" ]] || { echo 'Output directory is not writable.' >&2; exit 1; }
export M42_REPO_DIR="$REPO_DIR"
export M42_PYTHON=${M42_PYTHON:-${M41_PYTHON:-${M40_PYTHON:-/seu_share/home/huangkai/220243809/.conda/envs/igstgnn/bin/python}}}
export M42_DEVICE=${M42_DEVICE:-cuda:0}
command -v "$M42_PYTHON" >/dev/null 2>&1 || { printf 'Python unavailable: %s\n' "$M42_PYTHON" >&2; exit 1; }
if [[ $# -eq 1 ]]; then
    [[ -d "$1" ]] || { printf 'Missing history: %s\n' "$1" >&2; exit 1; }
    export M42_HISTORY_DIR
    M42_HISTORY_DIR=$(cd -- "$1" && pwd)
fi
sbatch --partition="${M42_PARTITION:-gpu_v100}" --job-name=capacity_m42 \
    --nodes=1 --ntasks=1 --cpus-per-task=3 --gres=gpu:1 \
    --time="${M42_TIME:-02:00:00}" --export=ALL --chdir="$REPO_DIR" \
    --output="$REPO_DIR/slurm-%x-%j.out" \
    "$SCRIPT_DIR/run_incident_capacity_training.sh"
