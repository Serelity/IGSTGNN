#!/usr/bin/env bash
# May be called from IGSTGNN-code or any other scheduler working directory.
set -euo pipefail

task_code="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd -- "$task_code"
task_phase="${1:-run}"
if [ "$#" -gt 1 ]; then
  printf 'Usage: bash scripts/run_pase_paper_experiments.sh [run|resume|evaluate|test|env|check]\n' >&2
  exit 2
fi
case "$task_phase" in run|resume|evaluate|test|env|check) ;; *) printf 'Unknown phase: %s\n' "$task_phase" >&2; exit 2 ;; esac

task_python="${PASE_PYTHON:-python}"
task_parent="/seu_share/home/huangkai/220243809/paper/IGSTGNN"
task_data="${PASE_DATA_DIR:-$task_parent/data/xtraffic/Contra_Costa}"
task_root="${PASE_RUN_ROOT:-$task_code/experiments/chronological_runs/pase_paper_cc_core}"
task_device="${PASE_DEVICE:-cuda:0}"
task_threads="${PASE_THREADS:-3}"
read -r -a task_seeds <<< "${PASE_SEEDS:-2025}"
read -r -a task_arms <<< "${PASE_ARMS:-FULL CONCAT_MLP REGIONAL_SOFTMAX NO_EXCHANGE UP_ONLY DOWN_ONLY ORIGIN_EXCHANGE NO_POSITION_BRANCH NO_INTERIOR_CENTER}"
task_environment="${CONDA_DEFAULT_ENV:-}"
if [ "${task_environment##*/}" != igstgnn ]; then
  printf 'Activate the existing environment first: conda activate igstgnn\nCurrent CONDA_DEFAULT_ENV=%s\n' "$task_environment" >&2
  exit 1
fi
command -v "$task_python" >/dev/null
export PYTHONPATH="$task_code${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="$task_threads" OPENBLAS_NUM_THREADS="$task_threads" MKL_NUM_THREADS="$task_threads"
export PYTHONUNBUFFERED=1

if [ "$task_phase" = check ]; then
  exec "$task_python" -m unittest discover -s "$task_code/tests" -p 'test_pase_paper_experiments_*.py' -v
fi

"$task_python" -m experiments.pase_paper_experiments.environment \
  --output-dir "$task_root/_environment" --device "$task_device"
if [ "$task_phase" = env ]; then exit 0; fi

test -d "$task_data" || { printf 'Dataset directory is missing: %s\nSet PASE_DATA_DIR to its existing location.\n' "$task_data" >&2; exit 1; }
printf 'Code: %s\nData: %s\nResults: %s\nDevice: %s\nSeeds: %s\nArms: %s\n' \
  "$task_code" "$task_data" "$task_root" "$task_device" "${task_seeds[*]}" "${task_arms[*]}"

if [ "$task_phase" = run ] || [ "$task_phase" = resume ]; then
  "$task_python" -m unittest discover -s "$task_code/tests" -p 'test_pase_paper_experiments_*.py' -v
  for task_seed in "${task_seeds[@]}"; do
    task_seed_dir="$task_root/s$task_seed"
    task_resume=()
    if [ -f "$task_seed_dir/identity.json" ]; then
      task_resume=(--resume)
    elif [ "$task_phase" = resume ]; then
      printf 'No initialized run identity at %s; starting the fixed protocol from stage zero.\n' "$task_seed_dir"
    fi
    "$task_python" -m experiments.pase_paper_experiments.train \
      --data-dir "$task_data" --run-dir "$task_seed_dir" --seed "$task_seed" \
      --device "$task_device" --threads "$task_threads" --arms "${task_arms[@]}" "${task_resume[@]}"
  done
fi

task_split=val
if [ "$task_phase" = test ]; then task_split=test; fi
task_scope="$(IFS=_; printf '%s' "${task_seeds[*]}")"
task_eval="${PASE_EVAL_DIR:-$task_root/evaluation_${task_split}_s$task_scope}"
"$task_python" -m experiments.pase_paper_experiments.evaluate \
  --data-dir "$task_data" --run-root "$task_root" --seeds "${task_seeds[@]}" \
  --arms "${task_arms[@]}" --device "$task_device" --threads "$task_threads" --split "$task_split" \
  --output-dir "$task_eval"
printf 'Read: %s/report.md\n' "$task_eval"
