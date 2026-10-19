#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
[[ $# -eq 0 ]] || { printf 'Configure GW15 through environment variables.\n' >&2; exit 2; }
[[ -n "${GW11_RUN:-}" && "${GW11_RUN:-}" == /* ]] || {
  printf 'Set GW11_RUN to the original absolute, completed GW11 training directory.\n' >&2
  exit 2
}
[[ -d "$GW11_RUN" && -f "$GW11_RUN/identity.json" ]] || {
  printf 'No identity.json in GW11_RUN: %s\n' "$GW11_RUN" >&2
  exit 2
}
GW15_RESUME="${GW15_RESUME:-0}"
GW15_EVALUATE_ONLY="${GW15_EVALUATE_ONLY:-0}"
[[ "$GW15_RESUME" =~ ^[01]$ && "$GW15_EVALUATE_ONLY" =~ ^[01]$ ]] || {
  printf 'GW15_RESUME and GW15_EVALUATE_ONLY must be 0 or 1.\n' >&2
  exit 2
}
[[ "$GW15_RESUME" != 1 || "$GW15_EVALUATE_ONLY" != 1 ]] || {
  printf 'Choose resume or evaluate-only, not both.\n' >&2
  exit 2
}
MODE_ARGS=()
if [[ "$GW15_RESUME" == 1 || "$GW15_EVALUATE_ONLY" == 1 ]]; then
  [[ -n "${GW15_RUN:-}" && "${GW15_RUN:-}" == /* ]] || {
    printf 'Resume/evaluate-only requires an explicit original absolute GW15_RUN; no default is used.\n' >&2
    exit 2
  }
  [[ -d "$GW15_RUN" && -f "$GW15_RUN/identity.json" && -f "$GW15_RUN/last.pt" ]] || {
    printf 'Resume/evaluate-only requires identity.json and last.pt in %s.\nKeep partial initialization; do not rebuild it as a resumed run.\n' "$GW15_RUN" >&2
    exit 2
  }
  if [[ "$GW15_RESUME" == 1 ]]; then MODE_ARGS=(--resume); else MODE_ARGS=(--evaluate-only); fi
else
  GW15_RUN="${GW15_RUN:-$REPO_DIR/experiments/chronological_runs/contra_gw15_trajectory_encoder_s2025_$(date +%Y%m%d_%H%M%S)}"
  [[ "$GW15_RUN" == /* ]] || {
    printf 'GW15_RUN must be an absolute, fresh output directory.\n' >&2
    exit 2
  }
  [[ ! -e "$GW15_RUN" && ! -L "$GW15_RUN" ]] || {
    printf 'Fresh output already exists: %s\nUse its original resume command, or keep it and choose a new GW15_RUN.\n' "$GW15_RUN" >&2
    exit 2
  }
fi
PYTHON_BIN="${PYTHON_BIN:-$(command -v python || true)}"
[[ -n "$PYTHON_BIN" && -x "$PYTHON_BIN" ]] || {
  printf 'Activate igstgnn or set PYTHON_BIN.\n' >&2
  exit 1
}
GW15_DATA="${GW15_DATA:-$REPO_DIR/../data/xtraffic/Contra_Costa}"
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-3}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-3}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-3}"
cd "$REPO_DIR"
"$PYTHON_BIN" verify_gw15_bundle.py
"$PYTHON_BIN" -m unittest tests.test_trajectory_encoder_gw15_model \
  tests.test_trajectory_encoder_gw15_protocol tests.test_trajectory_encoder_gw15_state \
  tests.test_trajectory_encoder_gw15_probe -v
printf 'Host: %s\nCompleted GW11 source: %s\nData: %s\nGW15 run: %s\nResume: %s\nEvaluate only: %s\n' \
  "$(hostname)" "$GW11_RUN" "$GW15_DATA" "$GW15_RUN" "$GW15_RESUME" "$GW15_EVALUATE_ONLY"
"$PYTHON_BIN" -u -m experiments.trajectory_encoder_gw15.probe \
  --gw11-run "$GW11_RUN" --data-dir "$GW15_DATA" --run-dir "$GW15_RUN" \
  --device "${GW15_DEVICE:-cuda:0}" --threads "${GW15_THREADS:-3}" "${MODE_ARGS[@]}"
printf 'Read %s/report.md, %s/screening.json, and %s/summary.json.\n' "$GW15_RUN" "$GW15_RUN" "$GW15_RUN"
