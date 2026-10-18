#!/usr/bin/env bash
#SBATCH --job-name=igst-gw4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=3
#SBATCH --time=04:00:00
set -euo pipefail
SCRIPT_REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
REPO_DIR="${ORIGINAL_IGSTGNN_REPO_DIR:-$SCRIPT_REPO_DIR}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python || true)}"
[[ -n "$PYTHON_BIN" && -x "$PYTHON_BIN" ]] || { printf 'Activate igstgnn or set executable PYTHON_BIN.\n' >&2; exit 1; }
[[ "$REPO_DIR" -ef "$SCRIPT_REPO_DIR" ]] || { printf 'Repository override differs from launcher.\n' >&2; exit 1; }
export ORIGINAL_IGSTGNN_REPO_DIR="$REPO_DIR" PYTHON_BIN
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}" PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
if [[ -z "${SLURM_JOB_ID:-}" && "${ORIGINAL_IGSTGNN_DIRECT:-0}" != 1 ]]; then
  exec sbatch --chdir="$REPO_DIR" --output="$REPO_DIR/original-gw4-%j.out" "$SCRIPT_REPO_DIR/experiments/chronological/run_original_receiver_residual.sh" "$@"
fi
cd "$REPO_DIR"
printf 'Repository: %s\nStarted: %s\nHost: %s\nSlurm job: %s\n' "$REPO_DIR" "$(date --iso-8601=seconds)" "$(hostname)" "${SLURM_JOB_ID:-direct}"
exec "$PYTHON_BIN" -u experiments/chronological/train_original_receiver_residual.py "$@"
