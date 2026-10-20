#!/usr/bin/env bash
# Server-only development; same command resumes committed epochs automatically.
set -euo pipefail
GW17_CODE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
GW17_PARENT="/seu_share/home/huangkai/220243809/paper/IGSTGNN"
GW17_DATA="${GW17_DATA:-$GW17_PARENT/data/xtraffic/Contra_Costa}"
GW17_REFERENCE="${GW17_REFERENCE:-$GW17_PARENT/IGSTGNN-GW16_BASELINE_1d0lEe0V/experiments/chronological_runs/contra_gw16_state_main_s2025/eval_best_val_epoch_100/STATE_EXCHANGE}"
GW17_RUN="${GW17_RUN:-$GW17_CODE/experiments/chronological_runs/contra_gw17_observed_state_s2025}"
GW17_CHECK_RUN="${GW17_CHECK_RUN:-${GW17_RUN}_engineering}"
GW17_EPOCHS="${GW17_EPOCHS:-100}"
GW17_DEVICE="${GW17_DEVICE:-cuda:0}"
GW17_THREADS="${GW17_THREADS:-3}"
export PYTHONPATH="$GW17_CODE${PYTHONPATH:+:$PYTHONPATH}"
cd -- "$GW17_CODE"
test -f "$GW17_DATA/incident_all.npy"
test -f "$GW17_REFERENCE/metadata.json"
test -f "$GW17_REFERENCE/predictions.npz"
printf 'GW17 code: %s\nData: %s\nGW16 validation reference: %s\nGW17 run: %s\nTotal target epochs per arm: %s\n' \
  "$GW17_CODE" "$GW17_DATA" "$GW17_REFERENCE" "$GW17_RUN" "$GW17_EPOCHS"
declare -p GW17_CODE GW17_DATA GW17_REFERENCE GW17_RUN GW17_CHECK_RUN GW17_EPOCHS \
  GW17_DEVICE GW17_THREADS > "$GW17_CODE/gw17_paths.env"
python -m unittest discover -s "$GW17_CODE/tests" -p 'test_observed_state_gw17_*.py' -v

run_training() {
  local task_directory="$1"
  local target_epochs="$2"
  shift 2
  local resume_arguments=()
  if [ -d "$task_directory" ]; then
    test -f "$task_directory/identity.json" || {
      printf 'Initialization is incomplete: %s. Preserve it and set a fresh run path.\n' "$task_directory" >&2
      return 1
    }
    resume_arguments=(--resume)
  fi
  python -m experiments.observed_state_gw17.train \
    --data-dir "$GW17_DATA" --gw16-reference "$GW17_REFERENCE" \
    --run-dir "$task_directory" --epochs "$target_epochs" --seed 2025 \
    --device "$GW17_DEVICE" --threads "$GW17_THREADS" \
    "${resume_arguments[@]}" "$@"
}

# Two tiny real-data arms, each three epochs; their state cannot become formal.
run_training "$GW17_CHECK_RUN" 3 --check
for selection in best last; do
  python -m experiments.observed_state_gw17.evaluate \
    --data-dir "$GW17_DATA" --run-dir "$GW17_CHECK_RUN" --selection "$selection" \
    --output-dir "$GW17_CHECK_RUN/eval_${selection}_val_epoch_3" \
    --device "$GW17_DEVICE" --threads "$GW17_THREADS"
done
run_training "$GW17_RUN" "$GW17_EPOCHS"

if [ "$GW17_EPOCHS" -eq 100 ]; then
  python -m experiments.observed_state_gw17.evaluate \
    --data-dir "$GW17_DATA" --run-dir "$GW17_RUN" --selection best \
    --output-dir "$GW17_RUN/eval_best_val_epoch_100" \
    --device "$GW17_DEVICE" --threads "$GW17_THREADS"
  python -m experiments.observed_state_gw17.evaluate \
    --data-dir "$GW17_DATA" --run-dir "$GW17_RUN" --selection last \
    --output-dir "$GW17_RUN/eval_last_val_epoch_100" \
    --device "$GW17_DEVICE" --threads "$GW17_THREADS"
  python -m experiments.observed_state_gw17.screen --run-dir "$GW17_RUN" \
    --best-dir "$GW17_RUN/eval_best_val_epoch_100" \
    --last-dir "$GW17_RUN/eval_last_val_epoch_100" \
    --output-dir "$GW17_RUN/screening"
  cat "$GW17_RUN/eval_best_val_epoch_100/report.md"
  cat "$GW17_RUN/eval_last_val_epoch_100/report.md"
  cat "$GW17_RUN/screening/screening.json"
else
  printf 'Partial development budget complete. Re-run with GW17_EPOCHS=100 to reach the fixed full budget.\n'
  cat "$GW17_RUN/summary.json"
fi
