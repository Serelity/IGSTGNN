#!/usr/bin/env bash
# Current compact FULL only: reuse CC; train Alameda and Orange, then test BEST.
set -euo pipefail
task_code="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd -- "$task_code"
task_dataset="${1:-all}"
if [ "$#" -gt 1 ]; then printf 'Usage: bash %s [all|Contra_Costa|Alameda|Orange|check]\n' "$0" >&2; exit 2; fi
case "$task_dataset" in all|Contra_Costa|Alameda|Orange|check) ;; *) printf 'Unknown dataset: %s\n' "$task_dataset" >&2; exit 2 ;; esac
task_parent='/seu_share/home/huangkai/220243809/paper/IGSTGNN'
task_python="${PASE_MAIN_PYTHON:-python}"
task_data_root="${PASE_MAIN_DATA_ROOT:-$task_parent/data/xtraffic}"
task_run_root="${PASE_MAIN_RUN_ROOT:-$task_code/experiments/chronological_runs/pase_compact_main}"
task_cc_root="${PASE_MAIN_CC_ROOT:-$task_parent/IGSTGNN-code/PASE-compact-code/experiments/chronological_runs/pase_compact_v3/Contra_Costa}"
task_device="${PASE_MAIN_DEVICE:-cuda:0}"
task_threads="${PASE_MAIN_THREADS:-3}"
task_environment="${CONDA_DEFAULT_ENV:-}"
if [ "${task_environment##*/}" != igstgnn ]; then
  printf 'Activate the existing environment: conda activate igstgnn\n' >&2; exit 1
fi
export PYTHONPATH="$task_code${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$task_threads" OPENBLAS_NUM_THREADS="$task_threads" MKL_NUM_THREADS="$task_threads"
"$task_python" -m unittest discover -s "$task_code/tests" -p 'test_pase_compact_main_*.py' -v
if [ "$task_dataset" = check ]; then exit 0; fi
if [ "$task_dataset" = all ]; then
  task_datasets=(Contra_Costa Alameda Orange)
else
  task_datasets=("$task_dataset")
fi
# Preflight selected inputs before starting a long training job.
for task_name in "${task_datasets[@]}"; do
  test -d "$task_data_root/$task_name" || { printf 'Dataset missing: %s\n' "$task_data_root/$task_name" >&2; exit 1; }
  if [ "$task_name" = Contra_Costa ]; then
    test -f "$task_cc_root/s2025/identity.json" || { printf 'Existing CC identity missing: %s\nSet PASE_MAIN_CC_ROOT to the completed compact run.\n' "$task_cc_root" >&2; exit 1; }
  fi
done
printf 'Code: %s\nResults: %s\nExisting CC: %s\n' "$task_code" "$task_run_root" "$task_cc_root"
for task_name in "${task_datasets[@]}"; do
  task_root="$task_run_root/$task_name"
  task_data="$task_data_root/$task_name"
  if [ "$task_name" = Contra_Costa ]; then
    # Keep the exact original numerical source/identity and trained FULL weights.
    "$task_python" -m experiments.pase_compact_v3.evaluate \
      --run-root "$task_cc_root" --data-dir "$task_data" \
      --output-dir "$task_root/evaluation_test_s2025" \
      --arms FULL --seeds 2025 --split test --save-predictions \
      --device "$task_device" --threads "$task_threads"
  else
    task_resume=()
    if [ -f "$task_root/s2025/identity.json" ]; then task_resume+=(--resume); fi
    "$task_python" -m experiments.pase_compact_main.train \
      --dataset "$task_name" --data-dir "$task_data" --run-dir "$task_root/s2025" \
      --arms FULL --seed 2025 --device "$task_device" --threads "$task_threads" \
      "${task_resume[@]}"
    "$task_python" -m experiments.pase_compact_main.evaluate \
      --run-root "$task_root" --data-dir "$task_data" \
      --output-dir "$task_root/evaluation_val_s2025" \
      --arms FULL --seeds 2025 --split val --device "$task_device" --threads "$task_threads"
    "$task_python" -m experiments.pase_compact_main.evaluate \
      --run-root "$task_root" --data-dir "$task_data" \
      --output-dir "$task_root/evaluation_test_s2025" \
      --arms FULL --seeds 2025 --split test --save-predictions \
      --device "$task_device" --threads "$task_threads"
  fi
  printf 'Completed %s: %s/evaluation_test_s2025/report.md\n' "$task_name" "$task_root"
done
if [ "$task_dataset" = all ]; then
  "$task_python" -m experiments.pase_compact_main.summarize --run-root "$task_run_root"
fi
