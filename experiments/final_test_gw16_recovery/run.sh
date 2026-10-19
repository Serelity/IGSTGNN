#!/usr/bin/env bash
# Server-only; keep the old e5dac19 download untouched to preserve its freeze.
set -euo pipefail
RECOVERY_CODE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
GW16_PARENT="/seu_share/home/huangkai/220243809/paper/IGSTGNN"
GW16_FROZEN_CODE="$GW16_PARENT/IGSTGNN-GW16_TEST_e5dac19"
GW16_PRIOR_CODE="$GW16_PARENT/IGSTGNN-GW16_BASELINE_1d0lEe0V"
GW16_SOURCE_RUN="$GW16_PRIOR_CODE/experiments/chronological_runs/contra_gw16_state_main_s2025"
GW16_TEST_RUN="${GW16_SOURCE_RUN}_final_test_locked"
export PYTHONPATH="$RECOVERY_CODE${PYTHONPATH:+:$PYTHONPATH}"
cd -- "$RECOVERY_CODE"
test -f "$GW16_FROZEN_CODE/experiments/final_test_gw16/evaluate.py"
test -f "$GW16_TEST_RUN/ORIGINAL_IGSTGNN/predictions.npz"
test -f "$GW16_TEST_RUN/ORIGINAL_IGSTGNN/metadata.json"
test -f "$GW16_TEST_RUN/freeze.json"
python -m unittest discover -s "$RECOVERY_CODE/tests" -p 'test_final_test_gw16_recovery.py' -v
python "$RECOVERY_CODE/experiments/final_test_gw16_recovery/recover.py" \
  --frozen-code "$GW16_FROZEN_CODE" \
  --expected-freeze-id 5a9d88244cbe9d0f5e2c6c1fd42a271a920ece32eda5012befe8bd21eb30af06 \
  --gw16-run "$GW16_SOURCE_RUN" \
  --gw16-val-eval "$GW16_SOURCE_RUN/eval_best_val_epoch_100/STATE_EXCHANGE" \
  --original-eval "$GW16_PRIOR_CODE/experiments/chronological_runs/original_native_val_20261019_223356" \
  --data-dir "$GW16_PARENT/data/xtraffic/Contra_Costa" \
  --output-dir "$GW16_TEST_RUN" --device cuda:0 --threads 3
cat "$GW16_TEST_RUN/report.md"
