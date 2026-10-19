#!/usr/bin/env bash
#SBATCH --job-name=igst-gw9
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=3
#SBATCH --time=24:00:00
set -euo pipefail

SCRIPT_REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
REPO_DIR="${GW9_REPO_DIR:-$SCRIPT_REPO_DIR}"
[[ -f "$REPO_DIR/experiments/branch_readout_gw9/probe.py" && -f "$REPO_DIR/GW9_SOURCE_MANIFEST.json" ]] || { printf 'Set GW9_REPO_DIR to the GW9 checkout.\n' >&2; exit 1; }
export GW9_REPO_DIR="$REPO_DIR"
MODE="${1:-probe}"
if [[ $# -gt 0 ]]; then shift; fi
[[ $# -eq 0 ]] || { printf 'Configure this launcher through environment variables.\n' >&2; exit 2; }
case "$MODE" in check|probe|probe-resume|probe-evaluate|joint|joint-resume|evaluate) ;; *) printf 'Mode must be check, probe, probe-resume, probe-evaluate, joint, joint-resume or evaluate.\n' >&2; exit 2 ;; esac
PYTHON_BIN="${PYTHON_BIN:-$(command -v python || true)}"
[[ -n "$PYTHON_BIN" && -x "$PYTHON_BIN" ]] || { printf 'Activate igstgnn or set PYTHON_BIN.\n' >&2; exit 1; }
GW9_GW8_RUN="${GW9_GW8_RUN:-${GW8_RUN:-}}"
[[ -n "$GW9_GW8_RUN" ]] || { printf 'Set GW9_GW8_RUN to the completed corrected GW8 run.\n' >&2; exit 1; }
GW9_DATA="${GW9_DATA:-$REPO_DIR/data/xtraffic/Contra_Costa}"
GW9_PROBE_RUN="${GW9_PROBE_RUN:-$REPO_DIR/experiments/chronological_runs/contra_original_gw9_probe_s2025}"
GW9_JOINT_RUN="${GW9_JOINT_RUN:-$REPO_DIR/experiments/chronological_runs/contra_original_gw9_joint_s2025}"
GW9_TAG="${GW9_TAG:-$(date +%Y%m%d_%H%M%S)}"
GW9_DEVICE="${GW9_DEVICE:-cuda:0}"
GW9_THREADS="${GW9_THREADS:-3}"
GW9_SEED="${GW9_SEED:-2025}"
GW9_CHECK_OUTPUT="${GW9_CHECK_OUTPUT:-${GW9_PROBE_RUN}_verification_${GW9_TAG}_${MODE}.json}"
export PYTHON_BIN GW9_GW8_RUN GW9_DATA GW9_PROBE_RUN GW9_JOINT_RUN GW9_TAG
export GW9_DEVICE GW9_THREADS GW9_SEED GW9_CHECK_OUTPUT
export ORIGINAL_IGSTGNN_REPO_DIR="$REPO_DIR"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
if [[ -z "${SLURM_JOB_ID:-}" && "${GW9_DIRECT:-0}" != 1 ]]; then
  command -v sbatch >/dev/null 2>&1 || { printf 'Obtain a GPU allocation; use GW9_DIRECT=1 only on an already allocated GPU host.\n' >&2; exit 1; }
  exec sbatch --chdir="$REPO_DIR" --export=ALL --output="$REPO_DIR/original-gw9-${MODE}-%j.out" "$REPO_DIR/experiments/branch_readout_gw9/run.sh" "$MODE"
fi
cd "$REPO_DIR"
printf 'Repository: %s\nMode: %s\nHost: %s\nSlurm: %s\nProbe: %s\nJoint: %s\n' "$REPO_DIR" "$MODE" "$(hostname)" "${SLURM_JOB_ID:-direct}" "$GW9_PROBE_RUN" "$GW9_JOINT_RUN"
COMMON=(--gw8-run "$GW9_GW8_RUN" --data-dir "$GW9_DATA" --device "$GW9_DEVICE" --threads "$GW9_THREADS" --seed "$GW9_SEED" --hidden 16)
if [[ -n "${GW9_GW8_BEST_EVAL:-}" ]]; then COMMON+=(--gw8-best-eval "$GW9_GW8_BEST_EVAL"); fi
if [[ -n "${GW9_GW8_LAST_EVAL:-}" ]]; then COMMON+=(--gw8-last-eval "$GW9_GW8_LAST_EVAL"); fi
run_check() {
  "$PYTHON_BIN" verify_gw9_bundle.py
  "$PYTHON_BIN" -m unittest tests.test_branch_readout_gw9_model tests.test_branch_readout_gw9_protocol tests.test_branch_readout_gw9_probe -v
  "$PYTHON_BIN" -u -m experiments.branch_readout_gw9.verify "${COMMON[@]}" --batch-size 48 --output "$GW9_CHECK_OUTPUT"
}
run_probe() {
  "$PYTHON_BIN" -u -m experiments.branch_readout_gw9.probe "${COMMON[@]}" --run-dir "$GW9_PROBE_RUN" "$@"
  printf 'Probe finished. Read %s/report.md and %s/screening.json. Joint training requires an eligible probe and a separate joint command.\n' "$GW9_PROBE_RUN" "$GW9_PROBE_RUN"
}
run_joint() {
  "$PYTHON_BIN" -u -m experiments.branch_readout_gw9.train "${COMMON[@]}" --probe-dir "$GW9_PROBE_RUN" --run-dir "$GW9_JOINT_RUN" --epochs 5 "$@"
}
run_evaluate() {
  "$PYTHON_BIN" -u -m experiments.branch_readout_gw9.evaluate --run-dir "$GW9_JOINT_RUN" --data-dir "$GW9_DATA" --device "$GW9_DEVICE" --checkpoint best --output-dir "$GW9_JOINT_RUN/eval_best_$GW9_TAG"
  "$PYTHON_BIN" -u -m experiments.branch_readout_gw9.evaluate --run-dir "$GW9_JOINT_RUN" --data-dir "$GW9_DATA" --device "$GW9_DEVICE" --checkpoint last --output-dir "$GW9_JOINT_RUN/eval_last_$GW9_TAG"
  "$PYTHON_BIN" -u -m experiments.branch_readout_gw9.evaluate --run-dir "$GW9_JOINT_RUN" --screen --best-eval "$GW9_JOINT_RUN/eval_best_$GW9_TAG" --last-eval "$GW9_JOINT_RUN/eval_last_$GW9_TAG" --output-dir "$GW9_JOINT_RUN/screen_$GW9_TAG"
}
case "$MODE" in
  check) run_check ;;
  probe) run_check; run_probe ;;
  probe-resume) run_check; run_probe --resume ;;
  probe-evaluate) "$PYTHON_BIN" verify_gw9_bundle.py; run_probe --evaluate-only ;;
  joint) run_check; run_joint; run_evaluate ;;
  joint-resume) run_check; run_joint --resume; run_evaluate ;;
  evaluate) "$PYTHON_BIN" verify_gw9_bundle.py; run_evaluate ;;
esac
