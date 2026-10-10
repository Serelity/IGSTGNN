#!/usr/bin/env bash
# Submit the M4.0 runner to Slurm; no version-control operations.
set -euo pipefail
if [[ $# -eq 1 && ( "$1" == --help || "$1" == -h ) ]]; then
    echo 'Usage: bash experiments/chronological/submit_incident_capacity_fusion.sh [original_v11a_history_directory]'
    echo 'Submits one V100 job: 1 GPU, 3 CPUs, 15 minutes; RAM is assigned by the cluster.'
    echo 'M40_PARTITION, M40_PYTHON and M40_DEVICE may override server defaults.'
    echo 'Existing M40 data/history environment variables are passed to the job.'
    exit 0
fi
[[ $# -le 1 && ( $# -eq 0 || "$1" != -* ) ]] || { echo 'Use --help.' >&2; exit 2; }
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd -- "$SCRIPT_DIR/../.." && pwd)
[[ -f "$REPO_DIR/src/models/igstgnn.py" && -f "$REPO_DIR/experiments/chronological/check_incident_capacity_fusion.py" ]] || {
    printf 'Invalid repository directory: %s\n' "$REPO_DIR" >&2; exit 1;
}
command -v sbatch >/dev/null 2>&1 || { echo 'sbatch unavailable: run on the Slurm login node.' >&2; exit 1; }
[[ -w "$REPO_DIR" ]] || { printf 'Repository is not writable for Slurm logs: %s\n' "$REPO_DIR" >&2; exit 1; }
mkdir -p "$REPO_DIR/experiments/chronological_runs" || {
    printf 'Cannot create output directory under repository: %s\n' "$REPO_DIR" >&2; exit 1;
}
[[ -w "$REPO_DIR/experiments/chronological_runs" ]] || {
    printf 'Output directory is not writable: %s/experiments/chronological_runs\n' "$REPO_DIR" >&2; exit 1;
}
# Slurm executes a spool copy; the runner must not infer its root from that copy.
export M40_REPO_DIR="$REPO_DIR"
export M40_PYTHON=${M40_PYTHON:-/seu_share/home/huangkai/220243809/.conda/envs/igstgnn/bin/python}
export M40_DEVICE=${M40_DEVICE:-cuda:0}
command -v "$M40_PYTHON" >/dev/null 2>&1 || { printf 'Python unavailable: %s\n' "$M40_PYTHON" >&2; exit 1; }
if [[ $# -eq 1 ]]; then
    [[ -d "$1" ]] || { printf 'Missing history directory: %s\n' "$1" >&2; exit 1; }
    export M40_HISTORY_DIR
    M40_HISTORY_DIR=$(cd -- "$1" && pwd)
fi
# This cluster assigns RAM from CPU count and rejects explicit memory requests.
sbatch --partition="${M40_PARTITION:-gpu_v100}" --job-name=capacity_m40 \
    --nodes=1 --ntasks=1 --cpus-per-task=3 --gres=gpu:1 \
    --time=00:15:00 --export=ALL --chdir="$REPO_DIR" \
    --output="$REPO_DIR/slurm-%x-%j.out" \
    "$SCRIPT_DIR/run_incident_capacity_fusion.sh"
