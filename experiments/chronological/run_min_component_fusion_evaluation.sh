#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${MIN_COMPONENT_REPO_DIR:-${MIN_FUSION_REPO_DIR:-/seu_share/home/huangkai/220243809/paper/IGSTGNN/IGSTGNN-code}}"
PYTHON_BIN="${PYTHON_BIN:-/seu_share/home/huangkai/220243809/.conda/envs/igstgnn/bin/python}"
[[ -d "$REPO_DIR" ]] || { printf 'Repository not found: %s\n' "$REPO_DIR" >&2; exit 1; }
[[ -x "$PYTHON_BIN" ]] || { printf 'Python not executable: %s\n' "$PYTHON_BIN" >&2; exit 1; }
cd "$REPO_DIR"
export PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
printf 'Repository: %s\nStarted: %s\nHost: %s\nSlurm job: %s\n' "$REPO_DIR" "$(date --iso-8601=seconds)" "$(hostname)" "${SLURM_JOB_ID:-none}"
exec "$PYTHON_BIN" -u experiments/chronological/evaluate_min_component_fusion.py "$@"
