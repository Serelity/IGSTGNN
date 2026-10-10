#!/usr/bin/env bash
# Slurm entry for M4.1; no version-control operations or explicit memory request.
set -euo pipefail
if [[ $# -eq 1 && ( "$1" == --help || "$1" == -h ) ]]; then
    echo 'Usage: bash experiments/chronological/submit_incident_capacity_network.sh [original_v11a_history_directory]'
    echo 'Submits one V100, 3 CPUs, 15 minutes; RAM is assigned by the cluster.'
    echo 'M41_PARTITION, M41_PYTHON, M41_DEVICE and M41 data variables may override defaults.'
    exit 0
fi
[[ $# -le 1 && ( $# -eq 0 || "$1" != -* ) ]] || { echo 'Use --help.' >&2; exit 2; }
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd -- "$SCRIPT_DIR/../.." && pwd)
command -v sbatch >/dev/null 2>&1 || { echo 'Run on the Slurm login node.' >&2; exit 1; }
[[ -f "$REPO_DIR/src/models/igstgnn.py" && -w "$REPO_DIR" ]] || { printf 'Invalid/unwritable repository: %s\n' "$REPO_DIR" >&2; exit 1; }
mkdir -p "$REPO_DIR/experiments/chronological_runs"
[[ -w "$REPO_DIR/experiments/chronological_runs" ]] || { echo 'Output directory is not writable.' >&2; exit 1; }
export M41_REPO_DIR="$REPO_DIR"
export M41_PYTHON=${M41_PYTHON:-${M40_PYTHON:-/seu_share/home/huangkai/220243809/.conda/envs/igstgnn/bin/python}}
export M41_DEVICE=${M41_DEVICE:-cuda:0}
command -v "$M41_PYTHON" >/dev/null 2>&1 || { printf 'Python unavailable: %s\n' "$M41_PYTHON" >&2; exit 1; }
if [[ $# -eq 1 ]]; then
    [[ -d "$1" ]] || { printf 'Missing history: %s\n' "$1" >&2; exit 1; }
    export M41_HISTORY_DIR
    M41_HISTORY_DIR=$(cd -- "$1" && pwd)
fi
sbatch --partition="${M41_PARTITION:-gpu_v100}" --job-name=capacity_m41 \
    --nodes=1 --ntasks=1 --cpus-per-task=3 --gres=gpu:1 \
    --time=00:15:00 --export=ALL --chdir="$REPO_DIR" \
    --output="$REPO_DIR/slurm-%x-%j.out" \
    "$SCRIPT_DIR/run_incident_capacity_network.sh"
