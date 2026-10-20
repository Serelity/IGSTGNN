#!/usr/bin/env bash
# Only the selected GW16 model; completed Contra Costa predictions are reused.
set -euo pipefail
GW16_ONLY_CODE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
GW16_PARENT="/seu_share/home/huangkai/220243809/paper/IGSTGNN"
GW16_ONLY_DATA="${GW16_ONLY_DATA:-$GW16_PARENT/data/xtraffic}"
GW16_ONLY_RUN="${GW16_ONLY_RUN:-$GW16_ONLY_CODE/experiments/chronological_runs/gw16_candidate_three_datasets_s2025}"
GW16_ONLY_CONTRA_TEST="${GW16_ONLY_CONTRA_TEST:-$GW16_PARENT/IGSTGNN-GW16_BASELINE_1d0lEe0V/experiments/chronological_runs/contra_gw16_state_main_s2025_final_test_locked}"
GW16_ONLY_DATASETS="${GW16_ONLY_DATASETS:-Alameda Orange}"
GW16_ONLY_EPOCHS="${GW16_ONLY_EPOCHS:-100}"
GW16_ONLY_DEVICE="${GW16_ONLY_DEVICE:-cuda:0}"
GW16_ONLY_THREADS="${GW16_ONLY_THREADS:-3}"
export PYTHONPATH="$GW16_ONLY_CODE${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="$GW16_ONLY_THREADS" MKL_NUM_THREADS="$GW16_ONLY_THREADS"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd -- "$GW16_ONLY_CODE"
read -r -a datasets <<< "$GW16_ONLY_DATASETS"
test "${#datasets[@]}" -ge 1
for dataset in "${datasets[@]}"; do
  case "$dataset" in Alameda|Orange) ;; *) printf 'This launcher trains Alameda/Orange; Contra Costa is reused.\n' >&2; exit 1 ;; esac
  test -f "$GW16_ONLY_DATA/$dataset/incident_all.npy"
  test -f "$GW16_ONLY_DATA/$dataset/sensors.csv"
done
printf 'Method: STATE_EXCHANGE\nCode: %s\nData root: %s\nResults: %s\nNew datasets: %s\nTotal epochs per dataset: %s\n' \
  "$GW16_ONLY_CODE" "$GW16_ONLY_DATA" "$GW16_ONLY_RUN" "$GW16_ONLY_DATASETS" "$GW16_ONLY_EPOCHS"
declare -p GW16_ONLY_CODE GW16_ONLY_DATA GW16_ONLY_RUN GW16_ONLY_CONTRA_TEST \
  GW16_ONLY_DATASETS GW16_ONLY_EPOCHS GW16_ONLY_DEVICE GW16_ONLY_THREADS \
  > "$GW16_ONLY_CODE/gw16_candidate_paths.env"

# Data, selected model and transaction checks; baseline tests are not invoked.
for area in data model training; do
  python -m unittest discover -s "$GW16_ONLY_CODE/tests" \
    -p "test_state_exchange_multidata_${area}.py" -v
done
python -m unittest discover -s "$GW16_ONLY_CODE/tests" -p 'test_state_exchange_candidate_*.py' -v

train_candidate() {
  local dataset="$1" directory="$2" epochs="$3"
  shift 3
  local resume_args=()
  if [ -d "$directory" ]; then
    test -f "$directory/identity.json" || {
      printf 'Initialization is incomplete: %s. Preserve it and set a fresh GW16_ONLY_RUN.\n' "$directory" >&2
      return 1
    }
    resume_args=(--resume)
  fi
  mkdir -p "$(dirname -- "$directory")"
  python -m experiments.state_exchange_multidata.train \
    --dataset "$dataset" --data-dir "$GW16_ONLY_DATA/$dataset" --run-dir "$directory" \
    --epochs "$epochs" --seed 2025 --device "$GW16_ONLY_DEVICE" --threads "$GW16_ONLY_THREADS" \
    "${resume_args[@]}" "$@"
}

for dataset in "${datasets[@]}"; do
  check="$GW16_ONLY_RUN/_engineering/$dataset"
  train_candidate "$dataset" "$check/candidate" 3 --engineering
  python -m experiments.state_exchange_multidata_candidate.evaluate \
    --dataset "$dataset" --data-dir "$GW16_ONLY_DATA/$dataset" --run-dir "$check/candidate" \
    --output-dir "$check/candidate_validation" --split val --engineering \
    --device "$GW16_ONLY_DEVICE" --threads "$GW16_ONLY_THREADS"
done

for dataset in "${datasets[@]}"; do
  python -m experiments.state_exchange_multidata_candidate.batch_check \
    --dataset "$dataset" --data-dir "$GW16_ONLY_DATA/$dataset" \
    --output-dir "$GW16_ONLY_RUN/_candidate_batch_check/$dataset" \
    --device "$GW16_ONLY_DEVICE" --threads "$GW16_ONLY_THREADS"
done

for dataset in "${datasets[@]}"; do
  train_candidate "$dataset" "$GW16_ONLY_RUN/$dataset/candidate" "$GW16_ONLY_EPOCHS"
done
if [ "$GW16_ONLY_EPOCHS" -ne 100 ]; then
  printf 'Partial budget complete. Repeat with GW16_ONLY_EPOCHS=100 to reach the fixed total.\n'
  exit 0
fi

for dataset in "${datasets[@]}"; do
  run="$GW16_ONLY_RUN/$dataset"
  for split in val test; do
    python -m experiments.state_exchange_multidata_candidate.evaluate \
      --dataset "$dataset" --data-dir "$GW16_ONLY_DATA/$dataset" --run-dir "$run/candidate" \
      --output-dir "$run/candidate_$split" --split "$split" \
      --device "$GW16_ONLY_DEVICE" --threads "$GW16_ONLY_THREADS"
  done
done

if [ -f "$GW16_ONLY_RUN/Alameda/candidate_test/result.json" ] && \
   [ -f "$GW16_ONLY_RUN/Orange/candidate_test/result.json" ] && \
   [ -f "$GW16_ONLY_CONTRA_TEST/STATE_EXCHANGE/metadata.json" ]; then
  python -m experiments.state_exchange_multidata_candidate.collect \
    --alameda-result "$GW16_ONLY_RUN/Alameda/candidate_test" \
    --orange-result "$GW16_ONLY_RUN/Orange/candidate_test" \
    --contra-test "$GW16_ONLY_CONTRA_TEST" \
    --output-dir "$GW16_ONLY_RUN/three_dataset_candidate_test"
  cat "$GW16_ONLY_RUN/three_dataset_candidate_test/report.md"
else
  for dataset in "${datasets[@]}"; do cat "$GW16_ONLY_RUN/$dataset/candidate_test/report.md"; done
  printf 'Collection needs both new candidate test caches and the existing Contra STATE_EXCHANGE cache.\n'
fi
