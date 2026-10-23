#!/usr/bin/env bash
# Official compact PASE: reuse FULL; train only the two matching one-way arms.
set -euo pipefail

task_code="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd -- "$task_code"
task_phase="${1:-run}"
if [ "$#" -gt 1 ]; then printf 'Usage: bash %s [run|resume|promote|evaluate|check]\n' "$0" >&2; exit 2; fi
case "$task_phase" in run|resume|promote|evaluate|check) ;; *) printf 'Unknown phase: %s\n' "$task_phase" >&2; exit 2 ;; esac
task_parent='/seu_share/home/huangkai/220243809/paper/IGSTGNN'
task_python="${PASE_COMPACT_PYTHON:-python}"
task_root="${PASE_COMPACT_RUN_ROOT:-$task_code/experiments/chronological_runs/pase_compact_v3/Contra_Costa}"
task_source="${PASE_COMPACT_SOURCE_ROOT:-$task_parent/IGSTGNN-code/PASE-MLP-code/experiments/chronological_runs/pase_mlp_v2/Contra_Costa}"
task_reference="${PASE_COMPACT_REFERENCE:-$task_root/promoted_full}"
task_data="${PASE_COMPACT_DATA_DIR:-$task_parent/data/xtraffic/Contra_Costa}"
task_device="${PASE_COMPACT_DEVICE:-cuda:0}"
task_threads="${PASE_COMPACT_THREADS:-3}"
task_run="$task_root/s2025"
task_replay="$task_root/evaluation_full_conversion_val_s2025"
task_eval="$task_root/evaluation_ablation_val_s2025"
task_environment="${CONDA_DEFAULT_ENV:-}"
if [ "${task_environment##*/}" != igstgnn ]; then
  printf 'Activate the existing environment: conda activate igstgnn\n' >&2; exit 1
fi
export PYTHONPATH="$task_code${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$task_threads" OPENBLAS_NUM_THREADS="$task_threads" MKL_NUM_THREADS="$task_threads"

if [ "$task_phase" = check ]; then
  exec "$task_python" -m unittest discover -s "$task_code/tests" -p 'test_pase_compact_v3_*.py' -v
fi
if [ "$task_phase" != evaluate ]; then
  "$task_python" -m unittest discover -s "$task_code/tests" -p 'test_pase_compact_v3_*.py' -v
  "$task_python" -m experiments.pase_compact_v3.promote \
    --source-run "$task_source/s2025" \
    --source-evaluation "$task_source/evaluation_ablation_val_s2025" \
    --output-dir "$task_reference"
  if [ "$task_phase" = promote ]; then
    printf 'Compact PASE inference weights: %s/model.pt\n' "$task_reference"
    printf 'Conversion validation is pending; run this launcher on the GPU to finish.\n'
    exit 0
  fi
fi
test -d "$task_data" || { printf 'Dataset missing: %s\n' "$task_data" >&2; exit 1; }
"$task_python" -m experiments.pase_paper_experiments.environment \
  --output-dir "$task_root/_environment" --device "$task_device"

if [ "$task_phase" != evaluate ]; then
  task_arguments=()
  if [ -f "$task_run/identity.json" ]; then task_arguments+=(--resume); fi
  # Register the external FULL role. No FULL optimizer or training state is made.
  "$task_python" -m experiments.pase_compact_v3.train \
    --data-dir "$task_data" --run-dir "$task_run" --arms FULL \
    --full-reference "$task_reference" --device "$task_device" --threads "$task_threads" \
    "${task_arguments[@]}"
  # One fixed-BEST replay checks compact conversion before any new training.
  # Later runs authenticate and reuse this complete cache without FULL forwards.
  "$task_python" -m experiments.pase_compact_v3.evaluate \
    --data-dir "$task_data" --run-root "$task_root" --output-dir "$task_replay" \
    --arms FULL --split val --device "$task_device" --threads "$task_threads"
  "$task_python" -m experiments.pase_compact_v3.train \
    --data-dir "$task_data" --run-dir "$task_run" --arms UP_ONLY DOWN_ONLY \
    --full-reference "$task_reference" --conversion-evaluation "$task_replay" \
    --resume --device "$task_device" --threads "$task_threads"
fi
"$task_python" -m experiments.pase_compact_v3.evaluate \
  --data-dir "$task_data" --run-root "$task_root" --output-dir "$task_eval" \
  --conversion-evaluation "$task_replay" \
  --arms FULL UP_ONLY DOWN_ONLY --split val --device "$task_device" --threads "$task_threads"
printf 'Read: %s/report.md\n' "$task_eval"
