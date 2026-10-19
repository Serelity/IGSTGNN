#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
[[ $# -eq 0 ]] || { printf 'Set GW10 environment variables; this launcher accepts no arguments.\n' >&2; exit 2; }
PYTHON_BIN="${PYTHON_BIN:-$(command -v python || true)}"
[[ -n "$PYTHON_BIN" && -x "$PYTHON_BIN" ]] || { printf 'Activate igstgnn first.\n' >&2; exit 1; }
GW10_PROBE_RUN="${GW10_PROBE_RUN:-$REPO_DIR/experiments/chronological_runs/contra_original_gw9_probe_replayfix_s2025}"
GW10_TAG="${GW10_TAG:-$(date +%Y%m%d_%H%M%S)}"
GW10_CACHE_DIR="${GW10_CACHE_DIR:-${GW10_PROBE_RUN}_gw10_cache_${GW10_TAG}}"
GW10_RUN="${GW10_RUN:-${GW10_PROBE_RUN}_gw10_probe_${GW10_TAG}}"
GW10_DATA="${GW10_DATA:-$REPO_DIR/../data/xtraffic/Contra_Costa}"
GW10_STAGE="${GW10_STAGE:-all}"
case "$GW10_STAGE" in all|export|fit) ;; *) printf 'GW10_STAGE must be all, export or fit.\n' >&2; exit 2;; esac
[[ -d "$GW10_PROBE_RUN" ]] || { printf 'Missing completed GW9 run: %s\n' "$GW10_PROBE_RUN" >&2; exit 1; }
if [[ "$GW10_STAGE" != export && -e "$GW10_RUN" ]]; then
  printf 'Use a fresh GW10_RUN; preserve existing output: %s\n' "$GW10_RUN" >&2; exit 1
fi
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-3}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-3}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-3}"
cd "$REPO_DIR"
printf 'Host: %s\nGW9 source: %s\nGW10 cache: %s\nGW10 result: %s\nStage: %s\n' "$(hostname)" "$GW10_PROBE_RUN" "$GW10_CACHE_DIR" "$GW10_RUN" "$GW10_STAGE"
"$PYTHON_BIN" verify_gw10_bundle.py
"$PYTHON_BIN" -m unittest tests.test_gw10_export_contract tests.test_gw10_median_probe tests.test_gw10_screen -v
if [[ "$GW10_STAGE" != fit ]]; then
  if [[ -f "$GW10_CACHE_DIR/cache.json" ]]; then
    printf 'Reusing completed cache; the probe will authenticate all source and cache bytes.\n'
  else
    [[ ! -e "$GW10_CACHE_DIR" ]] || { printf 'Incomplete cache exists. Keep it and set a fresh GW10_CACHE_DIR.\n' >&2; exit 1; }
    [[ -d "$GW10_DATA" ]] || { printf 'Set GW10_DATA to the original Contra_Costa data.\n' >&2; exit 1; }
    "$PYTHON_BIN" -u -m experiments.conditional_residual_gw10.export_cache \
      --probe-dir "$GW10_PROBE_RUN" --data-dir "$GW10_DATA" --cache-dir "$GW10_CACHE_DIR" \
      --device "${GW10_DEVICE:-cuda:0}" --threads "${GW10_THREADS:-3}"
  fi
fi
if [[ "$GW10_STAGE" != export ]]; then
  "$PYTHON_BIN" -u -m experiments.conditional_residual_gw10.run_probe \
    --probe-dir "$GW10_PROBE_RUN" --cache-dir "$GW10_CACHE_DIR" --output-dir "$GW10_RUN"
  printf 'Read %s/report.md and %s/screening.json.\n' "$GW10_RUN" "$GW10_RUN"
fi
