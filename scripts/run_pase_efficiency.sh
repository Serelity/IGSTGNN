#!/usr/bin/env bash
set -euo pipefail
task_code="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
task_parent='/seu_share/home/huangkai/220243809/paper/IGSTGNN'
task_original="${PASE_EFF_ORIGINAL_SOURCE:-$task_parent/IGSTGNN-code/experiments/igstgnn/Contra_Costa_igstgnn_s2025_20260908_193127}"
task_output="${PASE_EFF_OUTPUT:-$task_parent/paper_efficiency/pase_compact_$(date +%Y%m%d_%H%M%S)_$$}"
task_python="${PASE_EFF_PYTHON:-python}"
export PYTHONPATH="$task_code${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3
printf 'Code: %s\nOutput: %s\n' "$task_code" "$task_output"
"$task_python" "$task_code/scripts/benchmark_pase_compact.py" \
  --code-dir "$task_code" \
  --data-dir "${PASE_EFF_DATA:-$task_parent/data/xtraffic/Contra_Costa}" \
  --pase-run "${PASE_EFF_RUN:-$task_parent/IGSTGNN-code/PASE-compact-code/experiments/chronological_runs/pase_compact_v3/Contra_Costa/s2025}" \
  --original-checkpoint "$task_original/final_model_s2025.pt" \
  --original-log "$task_original/record_igstgnn_s2025_20260908_193127.log" \
  --output-dir "$task_output" --device "${PASE_EFF_DEVICE:-cuda:0}" --threads 3
cat "$task_output/report.md"
