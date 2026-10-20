#!/usr/bin/env bash
# Original IGSTGNN only: one independent Alameda/Orange process per visible GPU.
set -euo pipefail
IGSTGNN_ORIGINAL_CODE="${IGSTGNN_ORIGINAL_CODE:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)}"
IGSTGNN_PARENT="/seu_share/home/huangkai/220243809/paper/IGSTGNN"
IGSTGNN_ORIGINAL_DATA="${IGSTGNN_ORIGINAL_DATA:-$IGSTGNN_PARENT/data/xtraffic}"
IGSTGNN_ORIGINAL_RUN="${IGSTGNN_ORIGINAL_RUN:-$IGSTGNN_ORIGINAL_CODE/experiments/chronological_runs/original_alameda_orange_s2025}"
IGSTGNN_ORIGINAL_ALAMEDA_DEVICE="${IGSTGNN_ORIGINAL_ALAMEDA_DEVICE:-cuda:0}"
IGSTGNN_ORIGINAL_ORANGE_DEVICE="${IGSTGNN_ORIGINAL_ORANGE_DEVICE:-cuda:1}"
IGSTGNN_ORIGINAL_THREADS="${IGSTGNN_ORIGINAL_THREADS:-3}"
IGSTGNN_ORIGINAL_EPOCHS="${IGSTGNN_ORIGINAL_EPOCHS:-100}"
export IGSTGNN_ORIGINAL_CODE IGSTGNN_ORIGINAL_DATA IGSTGNN_ORIGINAL_RUN
export IGSTGNN_ORIGINAL_ALAMEDA_DEVICE IGSTGNN_ORIGINAL_ORANGE_DEVICE
export IGSTGNN_ORIGINAL_THREADS IGSTGNN_ORIGINAL_EPOCHS
export PYTHONPATH="$IGSTGNN_ORIGINAL_CODE${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="$IGSTGNN_ORIGINAL_THREADS" MKL_NUM_THREADS="$IGSTGNN_ORIGINAL_THREADS"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1
cd -- "$IGSTGNN_ORIGINAL_CODE"

train_original() {
  local dataset="$1" device="$2" directory="$3" epochs="$4"
  shift 4
  local resume_args=()
  if [ -d "$directory" ]; then
    test -f "$directory/identity.json" || {
      printf 'Incomplete initialization: %s. Preserve it and choose a fresh IGSTGNN_ORIGINAL_RUN.\n' "$directory" >&2
      return 1
    }
    resume_args=(--resume)
  fi
  mkdir -p -- "$(dirname -- "$directory")"
  python -m experiments.state_exchange_multidata.baseline \
    --dataset "$dataset" --data-dir "$IGSTGNN_ORIGINAL_DATA/$dataset" --run-dir "$directory" \
    --epochs "$epochs" --seed 2025 --device "$device" --threads "$IGSTGNN_ORIGINAL_THREADS" \
    "${resume_args[@]}" "$@"
}

worker_pipeline() {
  local dataset="$1" device="$2"
  local check="$IGSTGNN_ORIGINAL_RUN/_engineering/$dataset"
  local run="$IGSTGNN_ORIGINAL_RUN/$dataset"
  printf '[%s] Original IGSTGNN on %s; results %s\n' "$dataset" "$device" "$run"
  train_original "$dataset" "$device" "$check/original" 3 --check
  python -m experiments.original_multidata_only.evaluate \
    --dataset "$dataset" --data-dir "$IGSTGNN_ORIGINAL_DATA/$dataset" --run-dir "$check/original" \
    --output-dir "$check/original_val" --split val --engineering \
    --device "$device" --threads "$IGSTGNN_ORIGINAL_THREADS"
  python -m experiments.original_multidata_only.batch_check \
    --dataset "$dataset" --data-dir "$IGSTGNN_ORIGINAL_DATA/$dataset" \
    --output-dir "$IGSTGNN_ORIGINAL_RUN/_original_batch_check/$dataset" \
    --device "$device" --threads "$IGSTGNN_ORIGINAL_THREADS"
  train_original "$dataset" "$device" "$run/original" "$IGSTGNN_ORIGINAL_EPOCHS"
  if [ "$IGSTGNN_ORIGINAL_EPOCHS" -ne 100 ]; then
    printf '[%s] Partial total %s reached. Repeat with IGSTGNN_ORIGINAL_EPOCHS=100 to finish the registered budget.\n' \
      "$dataset" "$IGSTGNN_ORIGINAL_EPOCHS"
    return 0
  fi
  python -m experiments.original_multidata_only.evaluate \
    --dataset "$dataset" --data-dir "$IGSTGNN_ORIGINAL_DATA/$dataset" --run-dir "$run/original" \
    --output-dir "$run/original_val" --split val \
    --device "$device" --threads "$IGSTGNN_ORIGINAL_THREADS"
  python -m experiments.original_multidata_only.evaluate \
    --dataset "$dataset" --data-dir "$IGSTGNN_ORIGINAL_DATA/$dataset" --run-dir "$run/original" \
    --output-dir "$run/original_test" --split test --validation-dir "$run/original_val" \
    --device "$device" --threads "$IGSTGNN_ORIGINAL_THREADS"
  cat -- "$run/original_test/report.md"
}

if [ "${1:-}" = --worker ]; then
  test "$#" -eq 4
  case "$2" in Alameda|Orange) ;; *) exit 2 ;; esac
  # The worker owns its session. The master can terminate Python and tee together.
  set +e
  ( set -e; worker_pipeline "$2" "$3" ) 2>&1 | tee -a -- "$4"
  pipeline_status=("${PIPESTATUS[@]}")
  if [ "${pipeline_status[0]}" -ne 0 ]; then exit "${pipeline_status[0]}"; fi
  exit "${pipeline_status[1]}"
fi
if [ "$#" -ne 0 ]; then printf 'Run bash experiments/original_multidata_only/run.sh without arguments.\n' >&2; exit 2; fi
[[ "$IGSTGNN_ORIGINAL_THREADS" =~ ^[1-9][0-9]*$ ]] || { printf 'Threads must be positive.\n' >&2; exit 2; }
[[ "$IGSTGNN_ORIGINAL_EPOCHS" =~ ^[1-9][0-9]*$ ]] && \
  [ "$IGSTGNN_ORIGINAL_EPOCHS" -le 100 ] || { printf 'Epoch total must be in 1..100.\n' >&2; exit 2; }
for command in python setsid flock tee; do command -v "$command" >/dev/null; done
for dataset in Alameda Orange; do
  test -f "$IGSTGNN_ORIGINAL_DATA/$dataset/incident_all.npy"
  test -f "$IGSTGNN_ORIGINAL_DATA/$dataset/sensors.csv"
done
mkdir -p -- "$IGSTGNN_ORIGINAL_RUN"
IGSTGNN_ORIGINAL_RUN="$(cd -- "$IGSTGNN_ORIGINAL_RUN" && pwd -P)"
export IGSTGNN_ORIGINAL_RUN
exec 9>"$IGSTGNN_ORIGINAL_RUN/.launcher.lock"
flock -n 9 || { printf 'This output root already has a running launcher: %s\n' "$IGSTGNN_ORIGINAL_RUN" >&2; exit 1; }

# cuda:N is relative to CUDA_VISIBLE_DEVICES; do not replace a scheduler's mapping.
python - "$IGSTGNN_ORIGINAL_ALAMEDA_DEVICE" "$IGSTGNN_ORIGINAL_ORANGE_DEVICE" <<'PY'
import re
import sys
import torch
devices = sys.argv[1:]
if any(re.fullmatch(r'cuda:(0|[1-9][0-9]*)', device) is None for device in devices):
    raise SystemExit('Choose two distinct visible CUDA devices, for example cuda:0 and cuda:1')
indices = [int(device.split(':')[1]) for device in devices]
if len(set(indices)) != 2:
    raise SystemExit('Alameda and Orange must use different visible CUDA devices')
if not torch.cuda.is_available():
    raise SystemExit('CUDA is unavailable; run inside the allocated two-GPU server job')
for dataset, device in zip(('Alameda', 'Orange'), devices):
    index = int(device.split(':')[1])
    if index >= torch.cuda.device_count():
        raise SystemExit(f'{device} is not visible; request two GPUs or correct the device variables')
    print(f'{dataset}: {device} / {torch.cuda.get_device_name(index)}', flush=True)
PY

declare -p IGSTGNN_ORIGINAL_CODE IGSTGNN_ORIGINAL_DATA IGSTGNN_ORIGINAL_RUN \
  IGSTGNN_ORIGINAL_ALAMEDA_DEVICE IGSTGNN_ORIGINAL_ORANGE_DEVICE \
  IGSTGNN_ORIGINAL_THREADS IGSTGNN_ORIGINAL_EPOCHS \
  > "$IGSTGNN_ORIGINAL_CODE/original_multidata_paths.env"
printf 'Method: ORIGINAL_IGSTGNN\nCode: %s\nData root: %s\nResults: %s\n' \
  "$IGSTGNN_ORIGINAL_CODE" "$IGSTGNN_ORIGINAL_DATA" "$IGSTGNN_ORIGINAL_RUN"
python -m unittest discover -s "$IGSTGNN_ORIGINAL_CODE/tests" \
  -p 'test_state_exchange_multidata_baseline.py' -v
python -m unittest discover -s "$IGSTGNN_ORIGINAL_CODE/tests" \
  -p 'test_original_multidata_only*.py' -v

worker_pids=()
worker_names=()
cleanup_children() {
  trap - INT TERM EXIT
  local pid
  for pid in "${worker_pids[@]}"; do kill -TERM -- "-$pid" 2>/dev/null || true; done
  for pid in "${worker_pids[@]}"; do wait "$pid" 2>/dev/null || true; done
}
trap 'cleanup_children; exit 130' INT
trap 'cleanup_children; exit 143' TERM
trap cleanup_children EXIT
script="$IGSTGNN_ORIGINAL_CODE/experiments/original_multidata_only/run.sh"
for dataset in Alameda Orange; do
  case "$dataset" in
    Alameda) device="$IGSTGNN_ORIGINAL_ALAMEDA_DEVICE" ;;
    Orange) device="$IGSTGNN_ORIGINAL_ORANGE_DEVICE" ;;
  esac
  logfile="$IGSTGNN_ORIGINAL_RUN/${dataset}.launcher.log"
  printf 'Starting %s on %s; log: %s\n' "$dataset" "$device" "$logfile"
  setsid bash "$script" --worker "$dataset" "$device" "$logfile" &
  worker_pids+=("$!")
  worker_names+=("$dataset")
done
failed=0
for index in "${!worker_pids[@]}"; do
  if wait "${worker_pids[$index]}"; then
    printf '[%s] Pipeline completed.\n' "${worker_names[$index]}"
  else
    status="$?"
    printf '[%s] Pipeline failed (exit %s). The other dataset continues; preserve this run and rerun the same launcher to resume.\n' \
      "${worker_names[$index]}" "$status" >&2
    failed=1
  fi
  unset 'worker_pids[index]'
done
trap - INT TERM EXIT
if [ "$failed" -ne 0 ]; then exit 1; fi
if [ "$IGSTGNN_ORIGINAL_EPOCHS" -eq 100 ]; then
  printf 'Both original IGSTGNN pipelines completed. Results: %s/{Alameda,Orange}/original_test/report.md\n' \
    "$IGSTGNN_ORIGINAL_RUN"
else
  printf 'Both partial training budgets completed. Keep the same output root to resume.\n'
fi
