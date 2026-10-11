#!/usr/bin/env bash
#SBATCH --job-name=igst-is11
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=3
#SBATCH --time=06:00:00

set -euo pipefail
REPO_DIR="${IS11_REPO_DIR:-/seu_share/home/huangkai/220243809/paper/IGSTGNN/IGSTGNN-code}"
PYTHON_BIN="${PYTHON_BIN:-/seu_share/home/huangkai/220243809/.conda/envs/igstgnn/bin/python}"
[[ -d "$REPO_DIR" ]] || { echo "Repository not found: $REPO_DIR" >&2; exit 2; }
[[ -x "$PYTHON_BIN" ]] || { echo "Python not executable: $PYTHON_BIN" >&2; exit 2; }
export IS11_REPO_DIR="$REPO_DIR" PYTHON_BIN
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PYTHONUNBUFFERED=1

# From a login shell, `bash ...` submits this script; within an allocation it runs directly.
if [[ -z "${SLURM_JOB_ID:-}" && "${IS11_DIRECT:-0}" != 1 ]]; then
    exec sbatch --chdir="$REPO_DIR" --output="$REPO_DIR/is11-%j.out" \
        "$REPO_DIR/experiments/chronological/run_internal_spatial.sh" "$@"
fi
cd "$REPO_DIR"
printf 'Repository: %s\nStarted: %s\nHost: %s\nSlurm job: %s\n' \
    "$PWD" "$(date -Is)" "$(hostname)" "${SLURM_JOB_ID:-direct}"
exec "$PYTHON_BIN" -u experiments/chronological/train_internal_spatial.py "$@"
