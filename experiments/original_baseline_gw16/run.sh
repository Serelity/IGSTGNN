#!/usr/bin/env bash
# Server only. An independent original-model run; never rewrites GW16.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_DIR="${1:?Usage: bash run.sh DATA_DIR ORIGINAL_RUN_DIR}"
RUN_DIR="${2:?Usage: bash run.sh DATA_DIR ORIGINAL_RUN_DIR}"
PYTHON="${PYTHON:-python}"
RESUME="${RESUME:-0}"
DEVICE="${DEVICE:-cuda:0}"
THREADS="${THREADS:-3}"
SEED="${SEED:-2025}"
PRECHECK="${PRECHECK:-1}"
for VALUE in "$RESUME" "$PRECHECK"; do
    if [[ "$VALUE" != 0 && "$VALUE" != 1 ]]; then
        printf '%s\n' 'RESUME and PRECHECK must be 0 or 1.' >&2
        exit 2
    fi
done
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"
DATA_DIR="$(cd -- "$DATA_DIR" && pwd -P)"
if [[ -L "$RUN_DIR" ]]; then
    printf '%s\n' 'The original-model run directory cannot be a symbolic link.' >&2
    exit 2
fi
RUN_NAME="$(basename -- "$RUN_DIR")"
if [[ "$RUN_NAME" == . || "$RUN_NAME" == .. || "$RUN_NAME" == / ]]; then
    printf '%s\n' 'Use a named independent run directory.' >&2
    exit 2
fi
mkdir -p -- "$(dirname -- "$RUN_DIR")"
RUN_PARENT="$(cd -- "$(dirname -- "$RUN_DIR")" && pwd -P)"
RUN_DIR="$RUN_PARENT/$RUN_NAME"
if [[ -L "$RUN_DIR" ]]; then
    printf '%s\n' 'The original-model run directory cannot be a symbolic link.' >&2
    exit 2
fi
if [[ "$RUN_DIR" == "$DATA_DIR" || "$RUN_DIR" == "$DATA_DIR/"* || "$DATA_DIR" == "$RUN_DIR/"* ]]; then
    printf '%s\n' 'The original-model run must be outside the dataset directory.' >&2
    exit 2
fi
ANCESTOR="$RUN_PARENT"
while :; do
    if [[ -f "$ANCESTOR/identity.json" ]]; then
        printf 'Use an independent original-model run, outside %s\n' "$ANCESTOR" >&2
        exit 2
    fi
    [[ "$ANCESTOR" == / ]] && break
    ANCESTOR="$(dirname -- "$ANCESTOR")"
done
if [[ "$RESUME" == 0 && -e "$RUN_DIR" ]]; then
    printf '%s\n' 'Use a fresh original run, or RESUME=1 for its own committed run.' >&2
    exit 2
fi
if [[ "$RESUME" == 1 ]]; then
    if [[ ! -f "$RUN_DIR/identity.json" || ! -f "$RUN_DIR/last.pt" ]]; then
        printf '%s\n' 'Resume needs an existing original-model identity and committed last.pt.' >&2
        exit 2
    fi
    # Read JSON only, before the old trainer can open a log or write a failure.
    "$PYTHON" -c 'import json,sys; value=json.load(open(sys.argv[1])); sys.exit(0 if value.get("schema")=="original_igstgnn_baseline_training_v1" else "Resume target is not an original IGSTGNN run; source was not modified")' "$RUN_DIR/identity.json"
fi
if [[ "$PRECHECK" == 1 ]]; then
    for TEST_PATTERN in 'test_original_baseline_gw16_*.py' \
        'test_original_igstgnn_runner.py' 'test_original_igstgnn_source.py' \
        'test_original_igstgnn_runtime.py' 'test_original_igstgnn_checkpoint_compat.py'; do
        "$PYTHON" -m unittest discover -s "$ROOT/tests" -p "$TEST_PATTERN" -v
    done
fi
ARGS=()
if [[ "$RESUME" == 1 ]]; then
    ARGS+=(--resume)
fi
"$PYTHON" -m experiments.chronological.original_igstgnn_checkpoint_compat train \
    --data-dir "$DATA_DIR" --run-dir "$RUN_DIR" --epochs 100 \
    --seed "$SEED" --threads "$THREADS" --device "$DEVICE" "${ARGS[@]}" \
    2>&1 | tee -a "${RUN_DIR}.console.log"
EVAL_DIR="${RUN_DIR}_val_$(date +%Y%m%d_%H%M%S)"
"$PYTHON" -m experiments.original_baseline_gw16.evaluate \
    --source-run "$RUN_DIR" --data-dir "$DATA_DIR" --output-dir "$EVAL_DIR" \
    --threads "$THREADS" --device "$DEVICE" 2>&1 | tee -a "${RUN_DIR}.console.log"
printf 'Original summary: %s\nOriginal validation: %s\n' \
    "$RUN_DIR/summary.json" "$EVAL_DIR/report.md"
