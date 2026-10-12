#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${B3R_REPO_DIR:-/seu_share/home/huangkai/220243809/paper/IGSTGNN/IGSTGNN-code}"
PYTHON_BIN="${PYTHON_BIN:-/seu_share/home/huangkai/220243809/.conda/envs/igstgnn/bin/python}"
[[ -d "$REPO_DIR" ]] || { printf 'Repository not found: %s\n' "$REPO_DIR" >&2; exit 1; }
[[ -x "$PYTHON_BIN" ]] || { printf 'Python not executable: %s\n' "$PYTHON_BIN" >&2; exit 1; }
cd "$REPO_DIR"
export PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
printf 'Repository: %s\nStarted: %s\nHost: %s\n' "$REPO_DIR" "$(date --iso-8601=seconds)" "$(hostname)"
printf 'Evaluation: CPU artifact analysis; zero inference, zero training, no test data.\n'
exec "$PYTHON_BIN" -u experiments/chronological/analyze_model_evaluation.py "$@"
