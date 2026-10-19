#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
[[ $# -eq 0 ]] || { printf 'Configure GW14 through environment variables.\n' >&2; exit 2; }
[[ -n "${GW14_GW13_RUN:-}" ]] || {
  printf 'Set GW14_GW13_RUN explicitly to the completed original GW13 run directory.\n' >&2
  exit 2
}
[[ -n "${GW14_CACHE:-}" ]] || {
  printf 'Set GW14_CACHE explicitly to the completed original GW10 cache directory.\n' >&2
  exit 2
}
GW14_PROBE_RUN="${GW14_PROBE_RUN:-$REPO_DIR/experiments/chronological_runs/contra_original_gw9_probe_replayfix_s2025}"
GW14_DATA="${GW14_DATA:-$REPO_DIR/../data/xtraffic/Contra_Costa}"
GW14_RUN="${GW14_RUN:-$REPO_DIR/experiments/chronological_runs/contra_gw14_trajectory_check_s2025_$(date +%Y%m%d_%H%M%S)}"
for variable in GW14_GW13_RUN GW14_PROBE_RUN GW14_CACHE GW14_DATA GW14_RUN; do
  value="${!variable}"
  [[ "$value" == /* ]] || { printf '%s must be an absolute path.\n' "$variable" >&2; exit 2; }
done
[[ -d "$GW14_GW13_RUN" && -f "$GW14_GW13_RUN/analysis.json" \
    && -f "$GW14_GW13_RUN/model.json" && -f "$GW14_GW13_RUN/audit_model.json" \
    && -f "$GW14_GW13_RUN/screening.json" ]] || {
  printf 'GW14_GW13_RUN must contain the completed original GW13 analysis.json, model.json, audit_model.json and screening.json: %s\n' "$GW14_GW13_RUN" >&2
  exit 2
}
[[ -d "$GW14_PROBE_RUN" && -f "$GW14_PROBE_RUN/identity.json" \
    && -f "$GW14_PROBE_RUN/last.pt" && -f "$GW14_PROBE_RUN/screening.json" ]] || {
  printf 'GW14_PROBE_RUN must contain the completed original GW9 probe identity, last.pt and screening.json: %s\n' "$GW14_PROBE_RUN" >&2
  exit 2
}
[[ -d "$GW14_CACHE" && -f "$GW14_CACHE/cache.json" \
    && -f "$GW14_CACHE/train.npz" && -f "$GW14_CACHE/val.npz" ]] || {
  printf 'GW14_CACHE must contain completed cache.json, train.npz and val.npz: %s\n' "$GW14_CACHE" >&2
  exit 2
}
[[ -d "$GW14_DATA" && -f "$GW14_DATA/incident_all.npy" ]] || {
  printf 'Original authenticated GW14_DATA directory is missing: %s\n' "$GW14_DATA" >&2
  exit 2
}
[[ ! -e "$GW14_RUN" && ! -L "$GW14_RUN" ]] || {
  printf 'GW14_RUN already exists: %s\nKeep the original output and select a fresh directory; GW14 has no resume mode.\n' "$GW14_RUN" >&2
  exit 2
}
PYTHON_BIN="${PYTHON_BIN:-$(command -v python || true)}"
[[ -n "$PYTHON_BIN" && -x "$PYTHON_BIN" ]] || {
  printf 'Activate the existing igstgnn environment or set PYTHON_BIN.\n' >&2
  exit 1
}
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3
export NUMEXPR_NUM_THREADS=3 VECLIB_MAXIMUM_THREADS=3
cd "$REPO_DIR"
"$PYTHON_BIN" verify_gw14_bundle.py
"$PYTHON_BIN" -m unittest tests.test_trajectory_gw14_source \
  tests.test_trajectory_gw14_matching tests.test_trajectory_gw14_cross \
  tests.test_trajectory_gw14_run -v
printf 'Host: %s\nGW13 source: %s\nGW9 source: %s\nGW10 cache: %s\nOriginal data: %s\nGW14 output: %s\nCPU median lookup diagnostic; no new neural inference or optimizer steps.\n' \
  "$(hostname)" "$GW14_GW13_RUN" "$GW14_PROBE_RUN" "$GW14_CACHE" "$GW14_DATA" "$GW14_RUN"
"$PYTHON_BIN" -u -m experiments.trajectory_check_gw14.run \
  --gw13-dir "$GW14_GW13_RUN" --probe-dir "$GW14_PROBE_RUN" \
  --cache-dir "$GW14_CACHE" --data-dir "$GW14_DATA" --output-dir "$GW14_RUN"
printf 'Read %s/report.md, %s/checks.json and %s/matching_summary.csv.\n' "$GW14_RUN" "$GW14_RUN" "$GW14_RUN"
