#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
[[ $# -eq 0 ]] || { printf 'Configure GW12 through environment variables.\n' >&2; exit 2; }
[[ -n "${GW11_RUN:-}" && "${GW11_RUN:-}" == /* ]] || {
  printf 'Set GW11_RUN to the original absolute, completed GW11 training directory.\n' >&2
  exit 2
}
[[ -d "$GW11_RUN" && -f "$GW11_RUN/identity.json" ]] || {
  printf 'No identity.json in GW11_RUN: %s\n' "$GW11_RUN" >&2
  exit 2
}
GW12_RESUME="${GW12_RESUME:-0}"
GW12_EVALUATE_ONLY="${GW12_EVALUATE_ONLY:-0}"
[[ "$GW12_RESUME" =~ ^[01]$ && "$GW12_EVALUATE_ONLY" =~ ^[01]$ ]] || {
  printf 'GW12_RESUME and GW12_EVALUATE_ONLY must be 0 or 1.\n' >&2
  exit 2
}
[[ "$GW12_RESUME" != 1 || "$GW12_EVALUATE_ONLY" != 1 ]] || {
  printf 'Choose resume or evaluate-only, not both.\n' >&2
  exit 2
}
MODE_ARGS=()
if [[ "$GW12_RESUME" == 1 || "$GW12_EVALUATE_ONLY" == 1 ]]; then
  [[ -n "${GW12_RUN:-}" && "${GW12_RUN:-}" == /* ]] || {
    printf 'Resume/evaluate-only requires an explicit original absolute GW12_RUN; no default is used.\n' >&2
    exit 2
  }
  [[ -d "$GW12_RUN" && -f "$GW12_RUN/identity.json" && -f "$GW12_RUN/last.pt" ]] || {
    printf 'Resume/evaluate-only requires identity.json and last.pt in %s.\nKeep partial initialization; do not rebuild it as a resumed run.\n' "$GW12_RUN" >&2
    exit 2
  }
  if [[ "$GW12_RESUME" == 1 ]]; then MODE_ARGS=(--resume); else MODE_ARGS=(--evaluate-only); fi
else
  GW12_RUN="${GW12_RUN:-$REPO_DIR/experiments/chronological_runs/contra_gw12_frozen_readout_s2025_$(date +%Y%m%d_%H%M%S)}"
  [[ "$GW12_RUN" == /* ]] || {
    printf 'GW12_RUN must be an absolute, fresh output directory.\n' >&2
    exit 2
  }
  [[ ! -e "$GW12_RUN" && ! -L "$GW12_RUN" ]] || {
    printf 'Fresh output already exists: %s\nUse its original resume command, or keep it and choose a new GW12_RUN.\n' "$GW12_RUN" >&2
    exit 2
  }
fi
PYTHON_BIN="${PYTHON_BIN:-$(command -v python || true)}"
[[ -n "$PYTHON_BIN" && -x "$PYTHON_BIN" ]] || {
  printf 'Activate igstgnn or set PYTHON_BIN.\n' >&2
  exit 1
}
GW12_DATA="${GW12_DATA:-$REPO_DIR/../data/xtraffic/Contra_Costa}"
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-3}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-3}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-3}"
cd "$REPO_DIR"
"$PYTHON_BIN" verify_gw12_bundle.py
"$PYTHON_BIN" -m unittest tests.test_frozen_readout_gw12_model \
  tests.test_frozen_readout_gw12_protocol tests.test_frozen_readout_gw12_state \
  tests.test_frozen_readout_gw12_probe -v
printf 'Host: %s\nCompleted GW11 source: %s\nData: %s\nGW12 run: %s\nResume: %s\nEvaluate only: %s\n' \
  "$(hostname)" "$GW11_RUN" "$GW12_DATA" "$GW12_RUN" "$GW12_RESUME" "$GW12_EVALUATE_ONLY"
"$PYTHON_BIN" -u -m experiments.frozen_readout_gw12.probe \
  --gw11-run "$GW11_RUN" --data-dir "$GW12_DATA" --run-dir "$GW12_RUN" \
  --device "${GW12_DEVICE:-cuda:0}" --threads "${GW12_THREADS:-3}" "${MODE_ARGS[@]}"
printf 'Read %s/report.md, %s/screening.json, and %s/summary.json.\n' "$GW12_RUN" "$GW12_RUN" "$GW12_RUN"
