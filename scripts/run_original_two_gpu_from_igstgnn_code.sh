#!/usr/bin/env bash
# Scheduler work directory: .../IGSTGNN/IGSTGNN-code.
# Keep the existing checkout untouched; prepare and run the pinned dual-GPU code.
set -euo pipefail
task_parent="/seu_share/home/huangkai/220243809/paper/IGSTGNN"
task_workdir="${IGSTGNN_ORIGINAL_WORKDIR:-$task_parent/IGSTGNN-code}"
task_revision="ea13bc4d67a3611da57e9c05920ec52bfa9e7d0e"
task_branch="research/state-exchange-three-datasets"
task_repository="https://github.com/Serelity/IGSTGNN.git"
task_code="${IGSTGNN_ORIGINAL_CODE:-$task_parent/IGSTGNN-ORIGINAL-2GPU_ea13bc4}"
task_prepare_only=0
if [ "${1:-}" = --prepare-only ] && [ "$#" -eq 1 ]; then
  task_prepare_only=1
elif [ "$#" -ne 0 ]; then
  printf 'Usage: bash run_original_two_gpu.sh [--prepare-only]\n' >&2
  exit 2
fi
test -d "$task_workdir" || { printf 'Working directory is missing: %s\n' "$task_workdir" >&2; exit 1; }
cd -- "$task_workdir"
for task_command in git flock; do command -v "$task_command" >/dev/null; done

# Serialize preparation, including a second submission arriving during clone.
exec 8>"$task_workdir/.original_two_gpu_prepare.lock"
flock -n 8 || { printf 'Another process is preparing the same work directory. Retry after it finishes.\n' >&2; exit 1; }
task_staging=""
cleanup_preparation() {
  if [ -n "$task_staging" ] && [ -d "$task_staging" ]; then
    printf 'Incomplete download preserved: %s\n' "$task_staging" >&2
  fi
}
trap cleanup_preparation EXIT
if [ ! -e "$task_code" ]; then
  mkdir -p -- "$(dirname -- "$task_code")"
  task_staging="$(mktemp -d "$(dirname -- "$task_code")/.original_two_gpu_download.XXXXXXXX")"
  git clone --single-branch --branch "$task_branch" "$task_repository" "$task_staging"
  git -C "$task_staging" checkout --detach "$task_revision"
  test ! -e "$task_code" || { printf 'Code directory appeared during preparation; both copies preserved.\n' >&2; exit 1; }
  mv -T -- "$task_staging" "$task_code"
  task_staging=""
fi
test -d "$task_code/.git" || { printf 'Expected the pinned Git checkout at %s; preserve this directory and check the path.\n' "$task_code" >&2; exit 1; }
test "$(git -C "$task_code" rev-parse HEAD)" = "$task_revision" || {
  printf 'Code revision differs at %s. Expected %s; existing files were preserved.\n' "$task_code" "$task_revision" >&2
  exit 1
}
git -C "$task_code" diff --quiet HEAD -- || { printf 'Tracked code was modified at %s; existing files were preserved.\n' "$task_code" >&2; exit 1; }
test -f "$task_code/experiments/original_multidata_only/run.sh"
flock -u 8
exec 8>&-
trap - EXIT

export IGSTGNN_ORIGINAL_CODE="$task_code"
export IGSTGNN_ORIGINAL_DATA="${IGSTGNN_ORIGINAL_DATA:-$task_parent/data/xtraffic}"
export IGSTGNN_ORIGINAL_RUN="${IGSTGNN_ORIGINAL_RUN:-$task_code/experiments/chronological_runs/original_alameda_orange_s2025}"
export IGSTGNN_ORIGINAL_ALAMEDA_DEVICE="${IGSTGNN_ORIGINAL_ALAMEDA_DEVICE:-cuda:0}"
export IGSTGNN_ORIGINAL_ORANGE_DEVICE="${IGSTGNN_ORIGINAL_ORANGE_DEVICE:-cuda:1}"
export IGSTGNN_ORIGINAL_THREADS="${IGSTGNN_ORIGINAL_THREADS:-3}"
export IGSTGNN_ORIGINAL_EPOCHS="${IGSTGNN_ORIGINAL_EPOCHS:-100}"
declare -p IGSTGNN_ORIGINAL_CODE IGSTGNN_ORIGINAL_DATA IGSTGNN_ORIGINAL_RUN \
  IGSTGNN_ORIGINAL_ALAMEDA_DEVICE IGSTGNN_ORIGINAL_ORANGE_DEVICE \
  IGSTGNN_ORIGINAL_THREADS IGSTGNN_ORIGINAL_EPOCHS > "$task_workdir/original_multidata_job_paths.env"
printf 'Submission work directory: %s\nTraining code: %s\nResults: %s\n' \
  "$task_workdir" "$IGSTGNN_ORIGINAL_CODE" "$IGSTGNN_ORIGINAL_RUN"
if [ "$task_prepare_only" -eq 1 ]; then
  printf 'Preparation complete. Submit from this work directory with: bash run_original_two_gpu.sh\n'
  exit 0
fi
# The portal supplies its selected igstgnn environment and two visible GPUs.
# Existing code needs no network access here; the mature runner handles resume.
exec bash "$IGSTGNN_ORIGINAL_CODE/experiments/original_multidata_only/run.sh"
