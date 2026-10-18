#!/usr/bin/env bash
#SBATCH --job-name=igst-gw6
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=3
#SBATCH --time=24:00:00
set -euo pipefail

SCRIPT_REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
# Slurm executes a spool copy. Keep the original checkout across --export=ALL.
REPO_DIR="${GW6_REPO_DIR:-$SCRIPT_REPO_DIR}"
[[ -f "$REPO_DIR/experiments/graph_memory_gw6/train.py" && -f "$REPO_DIR/GW6_SOURCE_MANIFEST.json" ]] || { printf 'GW6_REPO_DIR must point to the GW6 checkout.\n' >&2; exit 1; }
export GW6_REPO_DIR="$REPO_DIR"
MODE="${1:-all}"
if [[ $# -gt 0 ]]; then shift; fi
if [[ $# -ne 0 ]]; then
  printf 'Use environment variables for launcher settings, or invoke Python CLI directly.\n' >&2
  exit 2
fi
case "$MODE" in check|train|resume|evaluate|all) ;; *) printf 'Mode must be check, train, resume, evaluate or all.\n' >&2; exit 2 ;; esac

PYTHON_BIN="${PYTHON_BIN:-$(command -v python || true)}"
[[ -n "$PYTHON_BIN" && -x "$PYTHON_BIN" ]] || { printf 'Activate igstgnn or set executable PYTHON_BIN.\n' >&2; exit 1; }
GW6_GW5_RUN="${GW6_GW5_RUN:-${GW5_RUN:-}}"
[[ -n "$GW6_GW5_RUN" ]] || { printf 'Set GW6_GW5_RUN to the completed GW5 run directory.\n' >&2; exit 1; }
GW6_DATA="${GW6_DATA:-$REPO_DIR/data/xtraffic/Contra_Costa}"
GW6_RUN="${GW6_RUN:-$REPO_DIR/experiments/chronological_runs/contra_original_gw6_joint_s2025}"
GW6_TAG="${GW6_TAG:-$(date +%Y%m%d_%H%M%S)}"
GW6_EPOCHS="${GW6_EPOCHS:-5}"
GW6_DEVICE="${GW6_DEVICE:-cuda:0}"
GW6_THREADS="${GW6_THREADS:-3}"
GW6_SEED="${GW6_SEED:-2025}"
GW6_HIDDEN="${GW6_HIDDEN:-32}"
GW6_NODE_CHUNK="${GW6_NODE_CHUNK:-32}"
GW6_STAGE="${GW6_STAGE:-joint}"
GW6_CURRICULUM="${GW6_CURRICULUM:-full}"
GW6_ARMS="${GW6_ARMS:-RECURRENT_CONTROL POOLED_MEMORY ADDITIVE_MEMORY JOINT_MEMORY}"
GW6_CHECK_OUTPUT="${GW6_CHECK_OUTPUT:-${GW6_RUN}_verification_${GW6_TAG}.json}"
export PYTHON_BIN GW6_GW5_RUN GW6_DATA GW6_RUN GW6_TAG GW6_EPOCHS GW6_DEVICE
export GW6_THREADS GW6_SEED GW6_HIDDEN GW6_NODE_CHUNK GW6_STAGE GW6_CURRICULUM GW6_ARMS GW6_CHECK_OUTPUT
export ORIGINAL_IGSTGNN_REPO_DIR="$REPO_DIR"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8

# Submit the entire requested sequence as one job. An asynchronous sbatch
# return must never trigger evaluation before that job has finished training.
if [[ -z "${SLURM_JOB_ID:-}" && "${GW6_DIRECT:-0}" != 1 ]]; then
  command -v sbatch >/dev/null 2>&1 || { printf 'No Slurm allocation found. Obtain a GPU allocation, or set GW6_DIRECT=1 on an allocated GPU host.\n' >&2; exit 1; }
  exec sbatch --chdir="$REPO_DIR" --export=ALL --output="$REPO_DIR/original-gw6-%j.out" "$REPO_DIR/experiments/graph_memory_gw6/run.sh" "$MODE"
fi
cd "$REPO_DIR"
printf 'Repository: %s\nMode: %s\nHost: %s\nSlurm job: %s\nRun: %s\n' "$REPO_DIR" "$MODE" "$(hostname)" "${SLURM_JOB_ID:-direct}" "$GW6_RUN"
read -r -a ARMS <<< "$GW6_ARMS"
COMMON=(--gw5-run "$GW6_GW5_RUN" --data-dir "$GW6_DATA" --device "$GW6_DEVICE" --threads "$GW6_THREADS" --seed "$GW6_SEED" --stage "$GW6_STAGE" --curriculum "$GW6_CURRICULUM" --hidden "$GW6_HIDDEN" --node-chunk "$GW6_NODE_CHUNK" --arms "${ARMS[@]}")
if [[ -n "${GW6_GW5_BEST_EVAL:-}" ]]; then COMMON+=(--gw5-best-eval "$GW6_GW5_BEST_EVAL"); fi
if [[ -n "${GW6_GW5_LAST_EVAL:-}" ]]; then COMMON+=(--gw5-last-eval "$GW6_GW5_LAST_EVAL"); fi

run_check() {
  "$PYTHON_BIN" verify_gw6_bundle.py
  "$PYTHON_BIN" -m unittest tests.test_graph_memory_gw6_model tests.test_graph_memory_gw6_workflow tests.test_graph_memory_gw6_evaluation -v
  "$PYTHON_BIN" -u -m experiments.graph_memory_gw6.verify "${COMMON[@]}" --batch-size 48 --output "$GW6_CHECK_OUTPUT"
}
run_train() {
  "$PYTHON_BIN" -u -m experiments.graph_memory_gw6.train "${COMMON[@]}" --run-dir "$GW6_RUN" --epochs "$GW6_EPOCHS" "$@"
}
run_evaluate() {
  "$PYTHON_BIN" -u -m experiments.graph_memory_gw6.evaluate --run-dir "$GW6_RUN" --data-dir "$GW6_DATA" --device "$GW6_DEVICE" --checkpoint best --output-dir "$GW6_RUN/eval_best_$GW6_TAG"
  "$PYTHON_BIN" -u -m experiments.graph_memory_gw6.evaluate --run-dir "$GW6_RUN" --data-dir "$GW6_DATA" --device "$GW6_DEVICE" --checkpoint last --output-dir "$GW6_RUN/eval_last_$GW6_TAG"
}
case "$MODE" in
  check) run_check ;;
  train) run_check; run_train ;;
  resume) run_check; run_train --resume; run_evaluate ;;
  evaluate) run_evaluate ;;
  all) run_check; run_train; run_evaluate ;;
esac
