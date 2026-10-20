#!/usr/bin/env bash
# New Alameda/Orange training; the completed Contra Costa experiment is reused.
set -euo pipefail
GW16_MULTI_CODE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
GW16_PARENT="/seu_share/home/huangkai/220243809/paper/IGSTGNN"
GW16_MULTI_DATA="${GW16_MULTI_DATA:-$GW16_PARENT/data/xtraffic}"
GW16_MULTI_RUN="${GW16_MULTI_RUN:-$GW16_MULTI_CODE/experiments/chronological_runs/gw16_three_datasets_s2025}"
GW16_CONTRA_TEST="${GW16_CONTRA_TEST:-$GW16_PARENT/IGSTGNN-GW16_BASELINE_1d0lEe0V/experiments/chronological_runs/contra_gw16_state_main_s2025_final_test_locked}"
GW16_MULTI_DATASETS="${GW16_MULTI_DATASETS:-Alameda Orange}"
GW16_MULTI_EPOCHS="${GW16_MULTI_EPOCHS:-100}"
GW16_MULTI_DEVICE="${GW16_MULTI_DEVICE:-cuda:0}"
GW16_MULTI_THREADS="${GW16_MULTI_THREADS:-3}"
export PYTHONPATH="$GW16_MULTI_CODE${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="$GW16_MULTI_THREADS" MKL_NUM_THREADS="$GW16_MULTI_THREADS"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd -- "$GW16_MULTI_CODE"
read -r -a datasets <<< "$GW16_MULTI_DATASETS"
test "${#datasets[@]}" -ge 1
for dataset in "${datasets[@]}"; do
  case "$dataset" in Alameda|Orange) ;; *) printf 'New training supports Alameda/Orange here; Contra Costa stays frozen.\n' >&2; exit 1 ;; esac
  test -f "$GW16_MULTI_DATA/$dataset/incident_all.npy"
  test -f "$GW16_MULTI_DATA/$dataset/sensors.csv"
done
printf 'Code: %s\nData root: %s\nResults: %s\nNew datasets: %s\nCandidate total epoch target: %s\n' \
  "$GW16_MULTI_CODE" "$GW16_MULTI_DATA" "$GW16_MULTI_RUN" "$GW16_MULTI_DATASETS" "$GW16_MULTI_EPOCHS"
declare -p GW16_MULTI_CODE GW16_MULTI_DATA GW16_MULTI_RUN GW16_CONTRA_TEST \
  GW16_MULTI_DATASETS GW16_MULTI_EPOCHS GW16_MULTI_DEVICE GW16_MULTI_THREADS \
  > "$GW16_MULTI_CODE/gw16_multidata_paths.env"
python -m unittest discover -s "$GW16_MULTI_CODE/tests" -p 'test_state_exchange_multidata_*.py' -v

train_one() {
  local module="$1" dataset="$2" directory="$3" epochs="$4"
  shift 4
  local resume_args=()
  if [ -d "$directory" ]; then
    test -f "$directory/identity.json" || {
      printf 'Initialization is incomplete: %s. Preserve it and set a fresh GW16_MULTI_RUN.\n' "$directory" >&2
      return 1
    }
    resume_args=(--resume)
  fi
  mkdir -p "$(dirname -- "$directory")"
  python -m "experiments.state_exchange_multidata.$module" \
    --dataset "$dataset" --data-dir "$GW16_MULTI_DATA/$dataset" --run-dir "$directory" \
    --epochs "$epochs" --seed 2025 --device "$GW16_MULTI_DEVICE" --threads "$GW16_MULTI_THREADS" \
    "${resume_args[@]}" "$@"
}

# Every dataset first exercises both models and the saved validation archives.
for dataset in "${datasets[@]}"; do
  check="$GW16_MULTI_RUN/_engineering/$dataset"
  train_one baseline "$dataset" "$check/baseline" 3 --check
  train_one train "$dataset" "$check/candidate" 3 --engineering
  python -m experiments.state_exchange_multidata.evaluate \
    --dataset "$dataset" --data-dir "$GW16_MULTI_DATA/$dataset" \
    --baseline-run "$check/baseline" --candidate-run "$check/candidate" \
    --output-dir "$check/validation" --split val --engineering \
    --device "$GW16_MULTI_DEVICE" --threads "$GW16_MULTI_THREADS"
done

# Check batch 48, including backward and another forward with Adam states resident.
for dataset in "${datasets[@]}"; do
  python -m experiments.state_exchange_multidata.batch_check \
    --dataset "$dataset" --data-dir "$GW16_MULTI_DATA/$dataset" \
    --output-dir "$GW16_MULTI_RUN/_batch_check/$dataset" \
    --device "$GW16_MULTI_DEVICE" --threads "$GW16_MULTI_THREADS"
done

for dataset in "${datasets[@]}"; do
  run="$GW16_MULTI_RUN/$dataset"
  train_one baseline "$dataset" "$run/baseline" "$GW16_MULTI_EPOCHS"
  train_one train "$dataset" "$run/candidate" "$GW16_MULTI_EPOCHS"
done

if [ "$GW16_MULTI_EPOCHS" -ne 100 ]; then
  printf 'Partial training complete. Repeat with GW16_MULTI_EPOCHS=100 for the fixed full protocol.\n'
  exit 0
fi

# Both new datasets finish training before any new test evaluation begins.
for dataset in "${datasets[@]}"; do
  run="$GW16_MULTI_RUN/$dataset"
  for split in val test; do
    python -m experiments.state_exchange_multidata.evaluate \
      --dataset "$dataset" --data-dir "$GW16_MULTI_DATA/$dataset" \
      --baseline-run "$run/baseline" --candidate-run "$run/candidate" \
      --output-dir "$run/$split" --split "$split" \
      --device "$GW16_MULTI_DEVICE" --threads "$GW16_MULTI_THREADS"
  done
done

if [ -f "$GW16_MULTI_RUN/Alameda/test/comparison.json" ] && \
   [ -f "$GW16_MULTI_RUN/Orange/test/comparison.json" ] && \
   [ -f "$GW16_CONTRA_TEST/comparison.json" ]; then
  python -m experiments.state_exchange_multidata.collect \
    --alameda-result "$GW16_MULTI_RUN/Alameda/test" \
    --orange-result "$GW16_MULTI_RUN/Orange/test" \
    --contra-test "$GW16_CONTRA_TEST" \
    --output-dir "$GW16_MULTI_RUN/three_dataset_test"
  cat "$GW16_MULTI_RUN/three_dataset_test/report.md"
else
  printf 'Completed new dataset reports:\n'
  for dataset in "${datasets[@]}"; do cat "$GW16_MULTI_RUN/$dataset/test/report.md"; done
  printf 'Three-dataset collection needs both new test caches and GW16_CONTRA_TEST/comparison.json.\n'
fi
