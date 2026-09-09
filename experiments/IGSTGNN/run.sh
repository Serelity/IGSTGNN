#!/bin/bash

set -euo pipefail

if (( $# > 2 )) || [[ -n "${2:-}" && "${2:-}" != "--smoke" ]]; then
    echo "Usage: bash experiments/IGSTGNN/run.sh [Alameda|Contra_Costa|Orange] [--smoke]" >&2
    exit 2
fi

dataset="${1:-Alameda}"
max_epochs=100
if [[ "${2:-}" == "--smoke" ]]; then
    max_epochs=1
fi
case "$dataset" in
    Alameda|Contra_Costa) batch_size=48 ;;
    Orange) batch_size=24 ;;
    *) echo "Unsupported dataset: $dataset" >&2; exit 2 ;;
esac

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
data_dir="data/xtraffic/$dataset"
for filename in adj_matrix.npy desc_mapping.json type_mapping.json incident_stats.npz sensors.csv; do
    if [[ ! -f "$data_dir/$filename" ]]; then
        echo "Missing dataset file: $PWD/$data_dir/$filename" >&2
        exit 1
    fi
done

split_count=0
for split in train val test; do
    if [[ -f "$data_dir/incident_$split.npy" ]]; then
        split_count=$((split_count + 1))
    fi
done
if (( split_count != 0 && split_count != 3 )); then
    echo "Incomplete splits in $data_dir. Restore a complete set before training; no files were overwritten." >&2
    exit 1
fi
if (( split_count == 0 )) && [[ ! -f "$data_dir/incident_all.npy" ]]; then
    echo "Missing dataset file: $PWD/$data_dir/incident_all.npy" >&2
    exit 1
fi

printf 'Dataset: %s, batch size: %s, seed: 2025, max epochs: %s, implementation: paper_aligned_v1\n' "$dataset" "$batch_size" "$max_epochs"
printf 'Host: %s, Slurm job: %s\n' "$(hostname)" "${SLURM_JOB_ID:-none}"
date -Is
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git log -1 --format='Code commit: %H %s'
    git status --short
fi

python -u - <<'PY'
import os
import sys

import numpy
import scipy
import torch

print("Python:", sys.version)
print("Executable:", sys.executable)
print("PyTorch:", torch.__version__, "CUDA runtime:", torch.version.cuda)
print("NumPy:", numpy.__version__, "SciPy:", scipy.__version__)
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", "not set"))
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable. Submit to gpu_v100; do not train on a login node.")
print("GPU:", torch.cuda.get_device_name(0))
print("VRAM GiB:", torch.cuda.get_device_properties(0).total_memory / 1024**3)
print("CUDA tensor check:", torch.ones(1, device="cuda:0").sum().item())
PY

if (( split_count == 0 )); then
    python -u data/xtraffic/prepare_splits.py --dataset "$dataset"
fi

exec python -u experiments/IGSTGNN/main.py \
    --dataset "$dataset" \
    --model_name igstgnn_paper \
    --seed 2025 \
    --bs "$batch_size" \
    --incident \
    --device cuda:0 \
    --use_sensor_info \
    --max_epochs "$max_epochs" \
    --patience 20 \
    --warm_epoch 30 \
    --cl_epoch 3
