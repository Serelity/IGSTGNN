#!/usr/bin/env bash
set -euo pipefail

# Server only. EPOCHS is a TOTAL epoch target, never an additional budget.
# bash experiments/position_state_gw16/run.sh DATA_DIR NEW_RUN_DIR
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_DIR="${1:?Usage: bash run.sh DATA_DIR NEW_RUN_DIR}"
RUN_DIR="${2:?Usage: bash run.sh DATA_DIR NEW_RUN_DIR}"
PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda:0}"
SEED="${SEED:-2025}"
THREADS="${THREADS:-3}"
RESUME="${RESUME:-0}"
ENGINEERING="${ENGINEERING:-0}"
DO_EVAL="${DO_EVAL:-1}"
if [[ "$RESUME" == 1 ]]; then
    PRECHECK="${PRECHECK:-0}"
else
    PRECHECK="${PRECHECK:-1}"
fi
for VALUE in "$RESUME" "$ENGINEERING" "$DO_EVAL" "$PRECHECK"; do
    if [[ "$VALUE" != 0 && "$VALUE" != 1 ]]; then
        printf '%s\n' 'RESUME, ENGINEERING, DO_EVAL and PRECHECK must be 0 or 1.' >&2
        exit 2
    fi
done
read -r -a ARM_LIST <<< "${ARMS:-NATIVE_POS DIRECT FREE_RECURRENT ORIGIN_EXCHANGE STATE_EXCHANGE}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
cd "$ROOT"
DATA_DIR="$(cd -- "$DATA_DIR" && pwd)"
mkdir -p -- "$(dirname -- "$RUN_DIR")"
RUN_PARENT="$(cd -- "$(dirname -- "$RUN_DIR")" && pwd)"
RUN_DIR="$RUN_PARENT/$(basename -- "$RUN_DIR")"

CONFIG_ARGS=()
if [[ -n "${CONFIG_FILE:-}" ]]; then
    CONFIG_ARGS=(--config "$CONFIG_FILE")
fi
TRAIN_ARGS=()
if [[ "$RESUME" == 1 ]]; then
    TRAIN_ARGS+=(--resume)
fi
if [[ -n "${EPOCHS:-}" ]]; then
    TRAIN_ARGS+=(--epochs "$EPOCHS")
fi
if [[ "$ENGINEERING" == 1 ]]; then
    TRAIN_ARGS+=(--engineering)
fi

if [[ "$PRECHECK" == 1 ]]; then
    "$PYTHON" -m unittest discover -s tests -p 'test_position_state_gw16_*.py' -v 2>&1 | tee -a "${RUN_DIR}.precheck.log"
    "$PYTHON" -m experiments.position_state_gw16.verify \
        --data-dir "$DATA_DIR" --output-dir "${RUN_DIR}_verify_$(date +%Y%m%d_%H%M%S)" \
        --device "$DEVICE" --threads "$THREADS" --seed "$SEED" "${CONFIG_ARGS[@]}" 2>&1 | tee -a "${RUN_DIR}.precheck.log"
fi

"$PYTHON" -m experiments.position_state_gw16.train \
    --data-dir "$DATA_DIR" --run-dir "$RUN_DIR" --seed "$SEED" \
    --device "$DEVICE" --threads "$THREADS" --arms "${ARM_LIST[@]}" \
    "${CONFIG_ARGS[@]}" "${TRAIN_ARGS[@]}" 2>&1 | tee -a "${RUN_DIR}.console.log"

if [[ "$DO_EVAL" != 1 ]]; then
    printf 'Training summary: %s\n' "$RUN_DIR/summary.json"
    exit 0
fi
# Read the committed summary only. This executes on the server, not on the
# development workstation, and never imports a model or opens test data.
read -r COMPLETE TARGET < <("$PYTHON" -c 'import json,sys; s=json.load(open(sys.argv[1])); arms=s["arms"]; required={"NATIVE_POS","DIRECT","FREE_RECURRENT","ORIGIN_EXCHANGE","STATE_EXCHANGE"}; complete=not s["engineering_only"] and s["fixed_full_budget"]==100 and required.issubset(arms) and all(arms[a]["completed_epoch"]==100 and arms[a]["full_protocol_reached"] for a in required); print(int(complete),s["requested_total_epochs"])' "$RUN_DIR/summary.json")
if [[ "$COMPLETE" != 0 && "$COMPLETE" != 1 ]]; then
    printf '%s\n' 'Could not determine the committed training budget.' >&2
    exit 2
fi
if [[ "$COMPLETE" == 1 ]]; then
    EVAL_SUFFIX=''
    EVALUATION_ARMS=(NATIVE_POS DIRECT FREE_RECURRENT ORIGIN_EXCHANGE STATE_EXCHANGE)
else
    printf -v EVAL_SUFFIX '_epoch_%03d' "$TARGET"
    EVALUATION_ARMS=("${ARM_LIST[@]}")
fi
for SELECTION in best last; do
    "$PYTHON" -m experiments.position_state_gw16.evaluate \
        --data-dir "$DATA_DIR" --run-dir "$RUN_DIR" --selection "$SELECTION" \
        --output-dir "$RUN_DIR/eval_${SELECTION}_val${EVAL_SUFFIX}" \
        --arms "${EVALUATION_ARMS[@]}" --device "$DEVICE" --threads "$THREADS" 2>&1 | tee -a "${RUN_DIR}.console.log"
done
if [[ "$COMPLETE" == 1 ]]; then
    "$PYTHON" -m experiments.position_state_gw16.screen --run-dir "$RUN_DIR" 2>&1 | tee -a "${RUN_DIR}.console.log"
    printf 'Screen: %s\n' "$RUN_DIR/screening.json"
else
    printf '%s\n' 'Partial budget or engineering run: no confirmation candidate and no automatic screen.'
fi
printf 'BEST report: %s\nLAST report: %s\n' \
    "$RUN_DIR/eval_best_val${EVAL_SUFFIX}/report.md" "$RUN_DIR/eval_last_val${EVAL_SUFFIX}/report.md"
