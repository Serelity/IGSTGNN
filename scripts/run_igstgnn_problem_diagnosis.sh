#!/usr/bin/env bash
# Same validation samples, original BEST and PASE BEST. Never train or test.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
TASK_PARENT="${PASE_PROBLEM_ROOT:-/seu_share/home/huangkai/220243809/paper/IGSTGNN}"
MODEL_CODE="${PASE_PROBLEM_MODEL_CODE:-$TASK_PARENT/IGSTGNN-GW16_BASELINE_1d0lEe0V}"
DATA_DIR="${PASE_PROBLEM_DATA:-$TASK_PARENT/data/xtraffic/Contra_Costa}"
ORIGINAL_EVAL="${PASE_PROBLEM_ORIGINAL_EVAL:-$MODEL_CODE/experiments/chronological_runs/original_native_val_20261019_223356}"
PASE_RUN="${PASE_PROBLEM_PASE_RUN:-$MODEL_CODE/experiments/chronological_runs/contra_gw16_state_main_s2025}"
PASE_EVAL="${PASE_PROBLEM_PASE_EVAL:-$PASE_RUN/eval_best_val_epoch_100/STATE_EXCHANGE}"
ORIGINAL_SOURCE="${PASE_PROBLEM_ORIGINAL_SOURCE:-$TASK_PARENT/IGSTGNN-code/experiments/igstgnn/Contra_Costa_igstgnn_s2025_20260908_193127}"
OUTPUT="${PASE_PROBLEM_OUTPUT:-$TASK_PARENT/paper_diagnostics/igstgnn_pase_best_$(date +%Y%m%d_%H%M%S)_$$}"
DEVICE="${PASE_PROBLEM_DEVICE:-cuda:0}"
THREADS="${PASE_PROBLEM_THREADS:-3}"
PYTHON_BIN="${PYTHON_BIN:-python}"
REPREDICT="${PASE_PROBLEM_REPREDICT:-0}"

test "$REPREDICT" = 0 || test "$REPREDICT" = 1 || { printf 'PASE_PROBLEM_REPREDICT must be 0 or 1\n' >&2; exit 2; }
test -f "$DATA_DIR/sensors.csv"
if test -e "$OUTPUT" || test -e "${OUTPUT}_original_val" || test -e "${OUTPUT}_pase_val"; then
  printf 'Output already exists; set a fresh PASE_PROBLEM_OUTPUT. Existing files preserved.\n' >&2
  exit 2
fi

# Numerical diagnostic checks run on the server, not on the authoring computer.
"$PYTHON_BIN" -m unittest discover -s "$SCRIPT_DIR/tests" -p 'test_igstgnn_problem_diagnosis.py' -v

if test "$REPREDICT" = 1 || ! test -f "$ORIGINAL_EVAL/predictions.npz"; then
  test -f "$MODEL_CODE/experiments/original_baseline_gw16/evaluate.py"
  test -f "$ORIGINAL_SOURCE/final_model_s2025.pt"
  test -f "$ORIGINAL_SOURCE/record_igstgnn_s2025_20260908_193127.log"
  ORIGINAL_EVAL="${OUTPUT}_original_val"
  printf 'Predicting original validation-selected BEST from checkpoint/log...\n'
  (
    cd "$MODEL_CODE"
    export PYTHONPATH="$MODEL_CODE${PYTHONPATH:+:$PYTHONPATH}"
    "$PYTHON_BIN" -u -m experiments.original_baseline_gw16.evaluate \
      --checkpoint "$ORIGINAL_SOURCE/final_model_s2025.pt" \
      --training-log "$ORIGINAL_SOURCE/record_igstgnn_s2025_20260908_193127.log" \
      --data-dir "$DATA_DIR" --output-dir "$ORIGINAL_EVAL" \
      --device "$DEVICE" --threads "$THREADS"
  )
else
  printf 'Reusing certified original BEST validation predictions: %s\n' "$ORIGINAL_EVAL"
fi

if test "$REPREDICT" = 1 || ! test -f "$PASE_EVAL/predictions.npz"; then
  test -f "$MODEL_CODE/experiments/position_state_gw16/evaluate.py"
  test -f "$PASE_RUN/identity.json"
  test -f "$PASE_RUN/STATE_EXCHANGE/last.pt"
  printf 'Predicting PASE validation-selected BEST (no LAST/test selection)...\n'
  (
    cd "$MODEL_CODE"
    export PYTHONPATH="$MODEL_CODE${PYTHONPATH:+:$PYTHONPATH}"
    "$PYTHON_BIN" -u -m experiments.position_state_gw16.evaluate \
      --run-dir "$PASE_RUN" --data-dir "$DATA_DIR" --arms STATE_EXCHANGE \
      --selection best --output-dir "${OUTPUT}_pase_val" \
      --device "$DEVICE" --threads "$THREADS"
  )
  PASE_EVAL="${OUTPUT}_pase_val/STATE_EXCHANGE"
else
  printf 'Reusing certified PASE BEST validation predictions: %s\n' "$PASE_EVAL"
fi

"$PYTHON_BIN" -u "$SCRIPT_DIR/diagnose_igstgnn_local_problem.py" \
  --root "$TASK_PARENT" --original-eval "$ORIGINAL_EVAL" \
  --pase-eval "$PASE_EVAL" --data-dir "$DATA_DIR" --output-dir "$OUTPUT" --bundle "$@"

printf '\nResults: %s\n' "$OUTPUT"
cat "$OUTPUT/report.md"
