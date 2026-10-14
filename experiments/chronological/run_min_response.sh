#!/usr/bin/env bash
#SBATCH --job-name=mini-response
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=3
#SBATCH --time=06:00:00
set -euo pipefail

REPO_DIR="${MIN_I_REPO_DIR:-/seu_share/home/huangkai/220243809/paper/IGSTGNN/IGSTGNN-code}"
PYTHON_BIN="${PYTHON_BIN:-/seu_share/home/huangkai/220243809/.conda/envs/igstgnn/bin/python}"
[[ -d "$REPO_DIR" ]] || { printf 'Repository not found: %s\n' "$REPO_DIR" >&2; exit 1; }
[[ -x "$PYTHON_BIN" ]] || { printf 'Python not executable: %s\n' "$PYTHON_BIN" >&2; exit 1; }
export MIN_I_REPO_DIR="$REPO_DIR" PYTHON_BIN
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}" PYTHONUNBUFFERED=1
if [[ -z "${SLURM_JOB_ID:-}" && "${MIN_RESPONSE_DIRECT:-0}" != 1 ]]; then
  exec sbatch --chdir="$REPO_DIR" --output="$REPO_DIR/mini-response-%j.out" "$REPO_DIR/experiments/chronological/run_min_response.sh" "$@"
fi
cd "$REPO_DIR"
printf 'Repository: %s\nStarted: %s\nHost: %s\nSlurm job: %s\n' "$REPO_DIR" "$(date --iso-8601=seconds)" "$(hostname)" "${SLURM_JOB_ID:-direct}"
for argument in "$@"; do
  if [[ "$argument" == --check ]]; then
    "$PYTHON_BIN" -X utf8 -m unittest tests.test_min_response_model tests.test_min_response_metrics tests.test_min_response_runner -v
    break
  fi
done
exec "$PYTHON_BIN" -u experiments/chronological/train_min_response.py "$@"
