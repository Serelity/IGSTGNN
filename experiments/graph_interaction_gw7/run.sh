#!/usr/bin/env bash
#SBATCH --job-name=igst-gw7
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=3
#SBATCH --time=24:00:00
set -euo pipefail

SCRIPT_REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
REPO_DIR="${GW7_REPO_DIR:-$SCRIPT_REPO_DIR}"
[[ -f "$REPO_DIR/experiments/graph_interaction_gw7/train.py" && -f "$REPO_DIR/GW7_SOURCE_MANIFEST.json" ]] || { printf 'Set GW7_REPO_DIR to the GW7 checkout.\n' >&2; exit 1; }
export GW7_REPO_DIR="$REPO_DIR"
MODE="${1:-diagnose}"
if [[ $# -gt 0 ]]; then shift; fi
[[ $# -eq 0 ]] || { printf 'Configure this launcher through environment variables.\n' >&2; exit 2; }
case "$MODE" in diagnose|check|train|screen|all|resume|evaluate) ;; *) printf 'Mode must be diagnose, check, train, screen, all, resume or evaluate.\n' >&2; exit 2 ;; esac
PYTHON_BIN="${PYTHON_BIN:-$(command -v python || true)}"
[[ -n "$PYTHON_BIN" && -x "$PYTHON_BIN" ]] || { printf 'Activate igstgnn or set PYTHON_BIN.\n' >&2; exit 1; }
GW7_GW6_RUN="${GW7_GW6_RUN:-${GW6_RUN:-}}"
[[ -n "$GW7_GW6_RUN" ]] || { printf 'Set GW7_GW6_RUN to the completed 20-epoch GW6 run.\n' >&2; exit 1; }
GW7_DATA="${GW7_DATA:-$REPO_DIR/data/xtraffic/Contra_Costa}"
GW7_RUN="${GW7_RUN:-$REPO_DIR/experiments/chronological_runs/contra_original_gw7_screen_s2025}"
GW7_TAG="${GW7_TAG:-$(date +%Y%m%d_%H%M%S)}"
GW7_DEVICE="${GW7_DEVICE:-cuda:0}"
GW7_THREADS="${GW7_THREADS:-3}"
GW7_SEED="${GW7_SEED:-2025}"
GW7_HIDDEN="${GW7_HIDDEN:-32}"
GW7_NODE_CHUNK="${GW7_NODE_CHUNK:-32}"
GW7_CHECK_OUTPUT="${GW7_CHECK_OUTPUT:-${GW7_RUN}_verification_${GW7_TAG}.json}"
if [[ "$MODE" == diagnose ]]; then
  GW7_DIAGNOSTIC_DIR="${GW7_DIAGNOSTIC_DIR:-$GW7_GW6_RUN/layers_$GW7_TAG}"
elif [[ "$MODE" == train || "$MODE" == screen || "$MODE" == all || "$MODE" == resume ]]; then
  [[ -n "${GW7_DIAGNOSTIC_DIR:-}" ]] || { printf 'Set GW7_DIAGNOSTIC_DIR explicitly to the completed layer diagnostic; training does not follow diagnose automatically.\n' >&2; exit 1; }
fi
export PYTHON_BIN GW7_GW6_RUN GW7_DATA GW7_RUN GW7_TAG GW7_DEVICE GW7_THREADS GW7_SEED
export GW7_HIDDEN GW7_NODE_CHUNK GW7_CHECK_OUTPUT GW7_DIAGNOSTIC_DIR
export ORIGINAL_IGSTGNN_REPO_DIR="$REPO_DIR"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
if [[ -z "${SLURM_JOB_ID:-}" && "${GW7_DIRECT:-0}" != 1 ]]; then
  command -v sbatch >/dev/null 2>&1 || { printf 'Obtain a GPU allocation, or set GW7_DIRECT=1 on an allocated GPU host.\n' >&2; exit 1; }
  exec sbatch --chdir="$REPO_DIR" --export=ALL --output="$REPO_DIR/original-gw7-%j.out" "$REPO_DIR/experiments/graph_interaction_gw7/run.sh" "$MODE"
fi
cd "$REPO_DIR"
printf 'Repository: %s\nMode: %s\nHost: %s\nSlurm: %s\nRun: %s\n' "$REPO_DIR" "$MODE" "$(hostname)" "${SLURM_JOB_ID:-direct}" "$GW7_RUN"
COMMON=(--gw6-run "$GW7_GW6_RUN" --data-dir "$GW7_DATA" --device "$GW7_DEVICE" --threads "$GW7_THREADS" --seed "$GW7_SEED" --hidden "$GW7_HIDDEN" --node-chunk "$GW7_NODE_CHUNK")
if [[ -n "${GW7_GW6_BEST_EVAL:-}" ]]; then COMMON+=(--gw6-best-eval "$GW7_GW6_BEST_EVAL"); fi
if [[ -n "${GW7_GW6_LAST_EVAL:-}" ]]; then COMMON+=(--gw6-last-eval "$GW7_GW6_LAST_EVAL"); fi
run_check() {
  "$PYTHON_BIN" verify_gw7_bundle.py
  "$PYTHON_BIN" -m unittest tests.test_graph_interaction_gw7_model tests.test_graph_interaction_gw7_protocol -v
  "$PYTHON_BIN" -u -m experiments.graph_interaction_gw7.verify "${COMMON[@]}" --batch-size 48 --output "$GW7_CHECK_OUTPUT"
}
run_train() {
  "$PYTHON_BIN" -u -m experiments.graph_interaction_gw7.train "${COMMON[@]}" --diagnostic-dir "$GW7_DIAGNOSTIC_DIR" --run-dir "$GW7_RUN" --epochs 5 "$@"
}
run_evaluate() {
  "$PYTHON_BIN" -u -m experiments.graph_interaction_gw7.evaluate --run-dir "$GW7_RUN" --data-dir "$GW7_DATA" --device "$GW7_DEVICE" --checkpoint best --output-dir "$GW7_RUN/eval_best_$GW7_TAG"
  "$PYTHON_BIN" -u -m experiments.graph_interaction_gw7.evaluate --run-dir "$GW7_RUN" --data-dir "$GW7_DATA" --device "$GW7_DEVICE" --checkpoint last --output-dir "$GW7_RUN/eval_last_$GW7_TAG"
  "$PYTHON_BIN" -u -m experiments.graph_interaction_gw7.evaluate --run-dir "$GW7_RUN" --screen --best-eval "$GW7_RUN/eval_best_$GW7_TAG" --last-eval "$GW7_RUN/eval_last_$GW7_TAG" --output-dir "$GW7_RUN/screen_$GW7_TAG"
}
case "$MODE" in
  diagnose)
    "$PYTHON_BIN" verify_gw7_bundle.py
    "$PYTHON_BIN" -m unittest tests.test_graph_memory_gw6_layers -v
    "$PYTHON_BIN" -u -m experiments.graph_memory_gw6.diagnose_layers --run-dir "$GW7_GW6_RUN" --output-dir "$GW7_DIAGNOSTIC_DIR" --data-dir "$GW7_DATA" --device "$GW7_DEVICE" --threads "$GW7_THREADS" --batches 2 --batch-size 48
    ;;
  check) run_check ;;
  train) run_check; run_train ;;
  screen|all) run_check; run_train; run_evaluate ;;
  resume) run_check; run_train --resume; run_evaluate ;;
  evaluate) "$PYTHON_BIN" verify_gw7_bundle.py; run_evaluate ;;
esac
