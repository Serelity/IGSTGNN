#!/usr/bin/env bash
#SBATCH --job-name=min-stock-flow
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=3
#SBATCH --time=06:00:00
set -euo pipefail

REPO_DIR="${MIN_I_REPO_DIR:-/seu_share/home/huangkai/220243809/paper/IGSTGNN/IGSTGNN-code}"
PYTHON_BIN="${PYTHON_BIN:-/seu_share/home/huangkai/220243809/.conda/envs/igstgnn/bin/python}"
DIRECT="${MIN_STOCK_FLOW_DIRECT:-0}"
ARGUMENTS=()
for argument in "$@"; do
  if [[ "$argument" == --direct ]]; then DIRECT=1; else ARGUMENTS+=("$argument"); fi
done
[[ -d "$REPO_DIR" ]] || { printf 'Repository not found: %s\n' "$REPO_DIR" >&2; exit 1; }
[[ -x "$PYTHON_BIN" ]] || { printf 'Python not executable: %s\n' "$PYTHON_BIN" >&2; exit 1; }
export MIN_I_REPO_DIR="$REPO_DIR" PYTHON_BIN
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}" PYTHONUNBUFFERED=1
if [[ -z "${SLURM_JOB_ID:-}" && "$DIRECT" != 1 ]]; then
  exec sbatch --chdir="$REPO_DIR" --output="$REPO_DIR/min-stock-flow-%j.out" "$REPO_DIR/experiments/chronological/run_min_stock_flow.sh" "${ARGUMENTS[@]}"
fi
cd "$REPO_DIR"
printf 'Repository: %s\nStarted: %s\nHost: %s\nSlurm job: %s\n' "$REPO_DIR" "$(date --iso-8601=seconds)" "$(hostname)" "${SLURM_JOB_ID:-direct}"
exec "$PYTHON_BIN" -u experiments/chronological/train_min_stock_flow.py "${ARGUMENTS[@]}"
