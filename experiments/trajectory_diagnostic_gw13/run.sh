#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
[[ $# -eq 0 ]] || { printf 'Configure GW13 through environment variables.\n' >&2; exit 2; }
GW13_PROBE_RUN="${GW13_PROBE_RUN:-$REPO_DIR/experiments/chronological_runs/contra_original_gw9_probe_replayfix_s2025}"
[[ -n "${GW13_CACHE:-}" ]] || {
  printf 'Set GW13_CACHE explicitly to the completed original GW10 cache directory.\n' >&2
  exit 2
}
GW13_DATA="${GW13_DATA:-$REPO_DIR/../data/xtraffic/Contra_Costa}"
GW13_RUN="${GW13_RUN:-$REPO_DIR/experiments/chronological_runs/contra_gw13_trajectory_s2025_$(date +%Y%m%d_%H%M%S)}"
for variable in GW13_PROBE_RUN GW13_CACHE GW13_DATA GW13_RUN; do
  value="${!variable}"
  [[ "$value" == /* ]] || { printf '%s must be an absolute path.\n' "$variable" >&2; exit 2; }
done
[[ -d "$GW13_PROBE_RUN" && -f "$GW13_PROBE_RUN/identity.json" \
    && -f "$GW13_PROBE_RUN/last.pt" && -f "$GW13_PROBE_RUN/screening.json" ]] || {
  printf 'GW13_PROBE_RUN must contain the completed original GW9 probe identity, last.pt and screening.json: %s\n' "$GW13_PROBE_RUN" >&2
  exit 2
}
[[ -d "$GW13_CACHE" && -f "$GW13_CACHE/cache.json" \
    && -f "$GW13_CACHE/train.npz" && -f "$GW13_CACHE/val.npz" ]] || {
  printf 'GW13_CACHE must contain completed cache.json, train.npz and val.npz: %s\n' "$GW13_CACHE" >&2
  exit 2
}
[[ -d "$GW13_DATA" && -f "$GW13_DATA/incident_all.npy" ]] || {
  printf 'Original authenticated GW13_DATA directory is missing: %s\n' "$GW13_DATA" >&2
  exit 2
}
[[ ! -e "$GW13_RUN" && ! -L "$GW13_RUN" ]] || {
  printf 'GW13_RUN already exists: %s\nKeep the original output and select a fresh directory; GW13 has no training resume mode.\n' "$GW13_RUN" >&2
  exit 2
}
PYTHON_BIN="${PYTHON_BIN:-$(command -v python || true)}"
[[ -n "$PYTHON_BIN" && -x "$PYTHON_BIN" ]] || {
  printf 'Activate the existing igstgnn environment or set PYTHON_BIN.\n' >&2
  exit 1
}
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3
cd "$REPO_DIR"
"$PYTHON_BIN" verify_gw13_bundle.py
"$PYTHON_BIN" -m unittest tests.test_trajectory_gw13_features \
  tests.test_trajectory_gw13_model tests.test_trajectory_gw13_protocol \
  tests.test_trajectory_gw13_run -v
printf 'Host: %s\nGW9 source: %s\nGW10 cache: %s\nOriginal data: %s\nGW13 output: %s\nCPU NumPy diagnostic; no new neural inference or training.\n' \
  "$(hostname)" "$GW13_PROBE_RUN" "$GW13_CACHE" "$GW13_DATA" "$GW13_RUN"
"$PYTHON_BIN" -u -m experiments.trajectory_diagnostic_gw13.run \
  --probe-dir "$GW13_PROBE_RUN" --cache-dir "$GW13_CACHE" \
  --data-dir "$GW13_DATA" --output-dir "$GW13_RUN"
printf 'Read %s/report.md, %s/screening.json and %s/coverage.csv.\n' "$GW13_RUN" "$GW13_RUN" "$GW13_RUN"
