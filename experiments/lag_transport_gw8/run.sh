#!/usr/bin/env bash
#SBATCH --job-name=igst-gw8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=3
#SBATCH --time=24:00:00
set -euo pipefail

SCRIPT_REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
REPO_DIR="${GW8_REPO_DIR:-$SCRIPT_REPO_DIR}"
[[ -f "$REPO_DIR/experiments/lag_transport_gw8/train.py" && -f "$REPO_DIR/GW8_SOURCE_MANIFEST.json" ]] || { printf 'Set GW8_REPO_DIR to the GW8 checkout.\n' >&2; exit 1; }
export GW8_REPO_DIR="$REPO_DIR"
MODE="${1:-check}"
if [[ $# -gt 0 ]]; then shift; fi
[[ $# -eq 0 ]] || { printf 'Configure this launcher through environment variables.\n' >&2; exit 2; }
case "$MODE" in check|train|screen|all|resume|evaluate) ;; *) printf 'Mode must be check, train, screen, all, resume or evaluate.\n' >&2; exit 2 ;; esac
PYTHON_BIN="${PYTHON_BIN:-$(command -v python || true)}"
[[ -n "$PYTHON_BIN" && -x "$PYTHON_BIN" ]] || { printf 'Activate igstgnn or set PYTHON_BIN.\n' >&2; exit 1; }
GW8_GW7_RUN="${GW8_GW7_RUN:-${GW7_RUN:-}}"
[[ -n "$GW8_GW7_RUN" ]] || { printf 'Set GW8_GW7_RUN to the completed GW7 five-epoch run.\n' >&2; exit 1; }
GW8_DATA="${GW8_DATA:-$REPO_DIR/data/xtraffic/Contra_Costa}"
GW8_RUN="${GW8_RUN:-$REPO_DIR/experiments/chronological_runs/contra_original_gw8_screen_s2025}"
GW8_TAG="${GW8_TAG:-$(date +%Y%m%d_%H%M%S)}"
GW8_DEVICE="${GW8_DEVICE:-cuda:0}"
GW8_THREADS="${GW8_THREADS:-3}"
GW8_SEED="${GW8_SEED:-2025}"
GW8_NODE_CHUNK="${GW8_NODE_CHUNK:-32}"
GW8_CHECK_OUTPUT="${GW8_CHECK_OUTPUT:-${GW8_RUN}_verification_${GW8_TAG}.json}"
export PYTHON_BIN GW8_GW7_RUN GW8_DATA GW8_RUN GW8_TAG GW8_DEVICE GW8_THREADS GW8_SEED
export GW8_NODE_CHUNK GW8_CHECK_OUTPUT
export ORIGINAL_IGSTGNN_REPO_DIR="$REPO_DIR"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
if [[ -z "${SLURM_JOB_ID:-}" && "${GW8_DIRECT:-0}" != 1 ]]; then
  command -v sbatch >/dev/null 2>&1 || { printf 'Obtain a GPU allocation, or set GW8_DIRECT=1 on an allocated GPU host.\n' >&2; exit 1; }
  exec sbatch --chdir="$REPO_DIR" --export=ALL --output="$REPO_DIR/original-gw8-%j.out" "$REPO_DIR/experiments/lag_transport_gw8/run.sh" "$MODE"
fi
cd "$REPO_DIR"
printf 'Repository: %s\nMode: %s\nHost: %s\nSlurm: %s\nRun: %s\n' "$REPO_DIR" "$MODE" "$(hostname)" "${SLURM_JOB_ID:-direct}" "$GW8_RUN"
COMMON=(--gw7-run "$GW8_GW7_RUN" --data-dir "$GW8_DATA" --device "$GW8_DEVICE" --threads "$GW8_THREADS" --seed "$GW8_SEED" --hidden 16 --edges 8 --node-chunk "$GW8_NODE_CHUNK")
if [[ -n "${GW8_GW7_BEST_EVAL:-}" ]]; then COMMON+=(--gw7-best-eval "$GW8_GW7_BEST_EVAL"); fi
if [[ -n "${GW8_GW7_LAST_EVAL:-}" ]]; then COMMON+=(--gw7-last-eval "$GW8_GW7_LAST_EVAL"); fi
run_check() {
  "$PYTHON_BIN" verify_gw8_bundle.py
  "$PYTHON_BIN" -m unittest tests.test_lag_transport_gw8_model tests.test_lag_transport_gw8_protocol tests.test_gw8_historical_replay -v
  "$PYTHON_BIN" -u -m experiments.lag_transport_gw8.verify "${COMMON[@]}" --batch-size 48 --output "$GW8_CHECK_OUTPUT"
}
run_train() {
  "$PYTHON_BIN" -u -m experiments.lag_transport_gw8.train "${COMMON[@]}" --run-dir "$GW8_RUN" --epochs 5 "$@"
}
run_evaluate() {
  "$PYTHON_BIN" -u -m experiments.lag_transport_gw8.evaluate --run-dir "$GW8_RUN" --data-dir "$GW8_DATA" --device "$GW8_DEVICE" --checkpoint best --output-dir "$GW8_RUN/eval_best_$GW8_TAG"
  "$PYTHON_BIN" -u -m experiments.lag_transport_gw8.evaluate --run-dir "$GW8_RUN" --data-dir "$GW8_DATA" --device "$GW8_DEVICE" --checkpoint last --output-dir "$GW8_RUN/eval_last_$GW8_TAG"
  "$PYTHON_BIN" -u -m experiments.lag_transport_gw8.evaluate --run-dir "$GW8_RUN" --screen --best-eval "$GW8_RUN/eval_best_$GW8_TAG" --last-eval "$GW8_RUN/eval_last_$GW8_TAG" --output-dir "$GW8_RUN/screen_$GW8_TAG"
}
case "$MODE" in
  check) run_check ;;
  train) run_check; run_train ;;
  screen|all) run_check; run_train; run_evaluate ;;
  resume) run_check; run_train --resume; run_evaluate ;;
  evaluate) "$PYTHON_BIN" verify_gw8_bundle.py; run_evaluate ;;
esac
