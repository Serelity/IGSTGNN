#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
[[ $# -eq 0 ]] || { printf 'Configure GW11 through environment variables.\n' >&2; exit 2; }
PYTHON_BIN="${PYTHON_BIN:-$(command -v python || true)}"
[[ -n "$PYTHON_BIN" && -x "$PYTHON_BIN" ]] || { printf 'Activate igstgnn or set PYTHON_BIN.\n' >&2; exit 1; }
GW11_PROBE_RUN="${GW11_PROBE_RUN:-$REPO_DIR/experiments/chronological_runs/contra_original_gw9_probe_replayfix_s2025}"
GW11_TAG="${GW11_TAG:-$(date +%Y%m%d_%H%M%S)}"
GW11_RUN="${GW11_RUN:-$REPO_DIR/experiments/chronological_runs/contra_gw11_state_gate_s2025_${GW11_TAG}}"
GW11_MASK_DIR="${GW11_MASK_DIR:-${GW11_RUN}_mask_audit}"
GW11_DATA="${GW11_DATA:-$REPO_DIR/../data/xtraffic/Contra_Costa}"
GW11_STAGE="${GW11_STAGE:-all}"
case "$GW11_STAGE" in all|audit|train) ;; *) printf 'GW11_STAGE must be all, audit or train.\n' >&2; exit 2;; esac
[[ ! ( "${GW11_RESUME:-0}" == 1 && "${GW11_EVALUATE_ONLY:-0}" == 1 ) ]] || { printf 'Choose resume or evaluate-only.\n' >&2; exit 2; }
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-3}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-3}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-3}"
cd "$REPO_DIR"
"$PYTHON_BIN" verify_gw11_bundle.py
"$PYTHON_BIN" -m unittest tests.test_state_gate_gw11_model tests.test_state_gate_gw11_protocol \
  tests.test_state_gate_gw11_reference tests.test_state_gate_gw11_mask \
  tests.test_state_gate_gw11_training tests.test_state_gate_gw11_discovery -v
DISCOVERY_ARGS=(--probe-dir "$GW11_PROBE_RUN")
if [[ -n "${GW11_CACHE_DIR:-}" ]]; then DISCOVERY_ARGS+=(--cache-dir "$GW11_CACHE_DIR"); fi
if [[ -n "${GW11_GW10_RUN:-}" ]]; then DISCOVERY_ARGS+=(--gw10-dir "$GW11_GW10_RUN"); fi
DISCOVERED="$("$PYTHON_BIN" -m experiments.state_gate_gw11.discover "${DISCOVERY_ARGS[@]}")"
mapfile -t DISCOVERED_PATHS <<< "$DISCOVERED"
[[ ${#DISCOVERED_PATHS[@]} -eq 2 ]] || { printf 'Source discovery did not return exactly two paths.\n' >&2; exit 1; }
GW11_CACHE_DIR="${DISCOVERED_PATHS[0]}"
GW11_GW10_RUN="${DISCOVERED_PATHS[1]}"
printf 'Host: %s\nGW9 source: %s\nGW10 cache: %s\nGW10 result: %s\nMask audit: %s\nGW11 run: %s\n' \
  "$(hostname)" "$GW11_PROBE_RUN" "$GW11_CACHE_DIR" "$GW11_GW10_RUN" "$GW11_MASK_DIR" "$GW11_RUN"
if [[ "$GW11_STAGE" != train ]]; then
  if [[ ! -f "$GW11_MASK_DIR/mask_audit.json" ]]; then
    [[ ! -e "$GW11_MASK_DIR" ]] || { printf 'Partial mask audit exists. Keep it and set a fresh GW11_MASK_DIR.\n' >&2; exit 1; }
    "$PYTHON_BIN" -u -m experiments.state_gate_gw11.mask_audit --probe-dir "$GW11_PROBE_RUN" \
      --cache-dir "$GW11_CACHE_DIR" --gw10-dir "$GW11_GW10_RUN" --output-dir "$GW11_MASK_DIR"
  fi
fi
if [[ "$GW11_STAGE" != audit ]]; then
  "$PYTHON_BIN" - "$GW11_MASK_DIR" <<'PY'
import json
from pathlib import Path
import sys
path=Path(sys.argv[1])/'mask_audit.json'
if not path.is_file() or json.loads(path.read_text(encoding='utf-8')).get('eligible_for_training') is not True:
    raise SystemExit('GW11 stopped before training. Read '+str(path.parent/'report.md')+'; masks were not changed.')
PY
  TRAIN_ARGS=(--probe-dir "$GW11_PROBE_RUN" --cache-dir "$GW11_CACHE_DIR" --gw10-dir "$GW11_GW10_RUN" \
    --audit-dir "$GW11_MASK_DIR" --data-dir "$GW11_DATA" --run-dir "$GW11_RUN" \
    --device "${GW11_DEVICE:-cuda:0}" --threads "${GW11_THREADS:-3}")
  if [[ "${GW11_RESUME:-0}" == 1 ]]; then TRAIN_ARGS+=(--resume); fi
  if [[ "${GW11_EVALUATE_ONLY:-0}" == 1 ]]; then TRAIN_ARGS+=(--evaluate-only); fi
  "$PYTHON_BIN" -u -m experiments.state_gate_gw11.train "${TRAIN_ARGS[@]}"
  printf 'Read %s/report.md and %s/screening.json.\n' "$GW11_RUN" "$GW11_RUN"
fi
