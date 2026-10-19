#!/usr/bin/env bash
# Run only on the existing igstgnn server environment. No git commands.
set -euo pipefail
TASK_CODE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
GW16_PARENT="/seu_share/home/huangkai/220243809/paper/IGSTGNN"
GW16_PRIOR_CODE="${GW16_PRIOR_CODE:-$GW16_PARENT/IGSTGNN-GW16_BASELINE_1d0lEe0V}"
GW16_DATA="${GW16_DATA:-$GW16_PARENT/data/xtraffic/Contra_Costa}"
GW16_FINAL_SOURCE_RUN="${GW16_FINAL_SOURCE_RUN:-$GW16_PRIOR_CODE/experiments/chronological_runs/contra_gw16_state_main_s2025}"
GW16_FINAL_VAL="${GW16_FINAL_VAL:-$GW16_FINAL_SOURCE_RUN/eval_best_val_epoch_100/STATE_EXCHANGE}"
GW16_FINAL_ORIGINAL_VAL="${GW16_FINAL_ORIGINAL_VAL:-$GW16_PRIOR_CODE/experiments/chronological_runs/original_native_val_20261019_223356}"
GW16_TEST_RUN="${GW16_TEST_RUN:-${GW16_FINAL_SOURCE_RUN}_final_test_locked}"
export PYTHONPATH="$TASK_CODE${PYTHONPATH:+:$PYTHONPATH}"
cd -- "$TASK_CODE"
test -f "$GW16_FINAL_SOURCE_RUN/STATE_EXCHANGE/last.pt"
test -f "$GW16_FINAL_VAL/metadata.json"
test -f "$GW16_FINAL_VAL/predictions.npz"
test -f "$GW16_FINAL_ORIGINAL_VAL/metadata.json"
test -f "$GW16_FINAL_ORIGINAL_VAL/predictions.npz"
test -f "$GW16_DATA/incident_all.npy"
printf 'Frozen candidate run: %s\nCertified BEST: %s\nOriginal replay: %s\nTest output: %s\n' \
  "$GW16_FINAL_SOURCE_RUN" "$GW16_FINAL_VAL" "$GW16_FINAL_ORIGINAL_VAL" "$GW16_TEST_RUN"
declare -p TASK_CODE GW16_PRIOR_CODE GW16_DATA GW16_FINAL_SOURCE_RUN GW16_FINAL_VAL \
  GW16_FINAL_ORIGINAL_VAL GW16_TEST_RUN > "$TASK_CODE/gw16_final_test_paths.env"
python -m unittest discover -s "$TASK_CODE/tests" -p 'test_final_test_gw16_*.py' -v
python -m experiments.final_test_gw16.evaluate \
  --gw16-run "$GW16_FINAL_SOURCE_RUN" \
  --gw16-val-eval "$GW16_FINAL_VAL" \
  --original-eval "$GW16_FINAL_ORIGINAL_VAL" \
  --data-dir "$GW16_DATA" --output-dir "$GW16_TEST_RUN" \
  --device "${GW16_DEVICE:-cuda:0}" --threads "${GW16_THREADS:-3}"
cat "$GW16_TEST_RUN/report.md"
