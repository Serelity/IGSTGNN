#!/usr/bin/env bash
# Versioned MLP model. Default: reuse the completed Contra Costa BEST, no fitting.
set -euo pipefail

task_code="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd -- "$task_code"
task_phase="${1:-promote}"
if [ "$#" -gt 1 ]; then printf 'Usage: bash %s [promote|train|resume|evaluate|check]\n' "$0" >&2; exit 2; fi
case "$task_phase" in promote|train|resume|evaluate|check) ;; *) printf 'Unknown phase: %s\n' "$task_phase" >&2; exit 2 ;; esac
task_parent='/seu_share/home/huangkai/220243809/paper/IGSTGNN'
task_dataset="${PASE_DATASET:-Contra_Costa}"
case "$task_dataset" in Contra_Costa|Alameda|Orange) ;; *) printf 'Unknown dataset: %s\n' "$task_dataset" >&2; exit 2 ;; esac
task_python="${PASE_PYTHON:-python}"
task_root="${PASE_V2_RUN_ROOT:-$task_code/experiments/chronological_runs/pase_mlp_v2/$task_dataset}"
task_data="${PASE_DATA_DIR:-$task_parent/data/xtraffic/$task_dataset}"
task_device="${PASE_DEVICE:-cuda:0}"
task_threads="${PASE_THREADS:-3}"
task_source_root="${PASE_MLP_SOURCE_ROOT:-$task_parent/IGSTGNN-code/PASE-paper-code/experiments/chronological_runs/pase_paper_cc_core}"
task_reference="${PASE_MLP_REFERENCE:-$task_root/promoted_full}"
read -r -a task_arms <<< "${PASE_V2_ARMS:-FULL}"
task_environment="${CONDA_DEFAULT_ENV:-}"
if [ "${task_environment##*/}" != igstgnn ]; then
  printf 'Activate the existing environment: conda activate igstgnn\n' >&2; exit 1
fi
export PYTHONPATH="$task_code${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$task_threads" OPENBLAS_NUM_THREADS="$task_threads" MKL_NUM_THREADS="$task_threads"
if [ "$task_phase" = check ]; then
  exec "$task_python" -m unittest discover -s "$task_code/tests" -p 'test_pase_mlp_v2_*.py' -v
fi

if [ "$task_phase" = promote ]; then
  test "$task_dataset" = Contra_Costa || { printf 'Completed source promotion is Contra Costa only.\n' >&2; exit 2; }
  "$task_python" -m unittest discover -s "$task_code/tests" -p 'test_pase_mlp_v2_*.py' -v
  "$task_python" -m experiments.pase_mlp_v2.promote \
    --source-run "$task_source_root/s2025" \
    --source-evaluation "$task_source_root/evaluation_mechanism_val_s2025" \
    --output-dir "$task_reference"
  printf 'Versioned MLP model: %s/model.pt\nRead: %s/report.md\n' "$task_reference" "$task_reference"
  exit 0
fi

test -d "$task_data" || { printf 'Dataset is missing: %s\n' "$task_data" >&2; exit 1; }
"$task_python" -m experiments.pase_paper_experiments.environment \
  --output-dir "$task_root/_environment" --device "$task_device"
task_run="$task_root/s2025"
if [ "$task_phase" = train ] || [ "$task_phase" = resume ]; then
  "$task_python" -m unittest discover -s "$task_code/tests" -p 'test_pase_mlp_v2_*.py' -v
  task_arguments=()
  if [ -f "$task_run/identity.json" ]; then task_arguments+=(--resume); fi
  if [ -f "$task_reference/metadata.json" ]; then task_arguments+=(--full-reference "$task_reference"); fi
  # Reusing existing Contra Costa FULL is mandatory through this convenience
  # entry. Explicit direct train CLI remains available for controlled fresh runs.
  if [ "$task_dataset" = Contra_Costa ] && [ ! -f "$task_reference/metadata.json" ]; then
    printf 'First run this launcher with promote to reuse the completed Contra Costa FULL.\n' >&2
    exit 1
  fi
  "$task_python" -m experiments.pase_mlp_v2.train \
    --dataset "$task_dataset" --data-dir "$task_data" --run-dir "$task_run" \
    --seed 2025 --device "$task_device" --threads "$task_threads" \
    --arms "${task_arms[@]}" "${task_arguments[@]}"
fi
task_eval="${PASE_V2_EVAL_DIR:-$task_root/evaluation_val}"
"$task_python" -m experiments.pase_mlp_v2.evaluate \
  --data-dir "$task_data" --run-root "$task_root" --output-dir "$task_eval" \
  --seeds 2025 --arms "${task_arms[@]}" --split val --device "$task_device" --threads "$task_threads"
printf 'Read: %s/report.md\n' "$task_eval"
