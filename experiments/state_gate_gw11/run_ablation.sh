#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
[[ $# -eq 0 ]] || { printf 'Configure GW11 LAST ablation through environment variables.\n' >&2; exit 2; }
[[ -n "${GW11_RUN:-}" && "${GW11_RUN:-}" == /* ]] || {
  printf 'Set GW11_RUN to the original absolute GW11 training directory.\n' >&2
  exit 2
}
[[ -d "$GW11_RUN" && -f "$GW11_RUN/identity.json" ]] || {
  printf 'No identity.json in GW11_RUN: %s\n' "$GW11_RUN" >&2
  exit 2
}
GW11_ABLATION_DIR="${GW11_ABLATION_DIR:-${GW11_RUN%/}_last_ablation_$(date +%Y%m%d_%H%M%S)}"
[[ "$GW11_ABLATION_DIR" == /* ]] || {
  printf 'GW11_ABLATION_DIR must be an absolute, fresh output directory.\n' >&2
  exit 2
}
[[ ! -e "$GW11_ABLATION_DIR" && ! -L "$GW11_ABLATION_DIR" ]] || {
  printf 'Output already exists: %s\nKeep it and choose a fresh GW11_ABLATION_DIR.\n' "$GW11_ABLATION_DIR" >&2
  exit 2
}
PYTHON_BIN="${PYTHON_BIN:-$(command -v python || true)}"
[[ -n "$PYTHON_BIN" && -x "$PYTHON_BIN" ]] || {
  printf 'Activate igstgnn or set PYTHON_BIN.\n' >&2
  exit 1
}
GW11_DATA="${GW11_DATA:-$REPO_DIR/../data/xtraffic/Contra_Costa}"
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-3}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-3}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-3}"
cd "$REPO_DIR"
"$PYTHON_BIN" verify_gw11_ablation_bundle.py
"$PYTHON_BIN" -m unittest tests.test_gw11_ablation_auth \
  tests.test_gw11_ablation_metrics tests.test_gw11_ablation_runner -v
printf 'Host: %s\nOriginal GW11 run: %s\nValidation data: %s\nNew LAST-ablation output: %s\n' \
  "$(hostname)" "$GW11_RUN" "$GW11_DATA" "$GW11_ABLATION_DIR"
"$PYTHON_BIN" -u -m experiments.state_gate_gw11.ablate_last \
  --run-dir "$GW11_RUN" --data-dir "$GW11_DATA" --output-dir "$GW11_ABLATION_DIR" \
  --device "${GW11_DEVICE:-cuda:0}" --threads "${GW11_THREADS:-3}"
printf 'Read %s/report.md and %s/decomposition.csv.\n' "$GW11_ABLATION_DIR" "$GW11_ABLATION_DIR"
