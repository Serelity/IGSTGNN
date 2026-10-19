#!/usr/bin/env bash
set -euo pipefail

SCRIPT_REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
REPO_DIR="${GW9_REPO_DIR:-$SCRIPT_REPO_DIR}"
[[ $# -eq 0 ]] || { printf 'Configure this launcher through environment variables.\n' >&2; exit 2; }
[[ -f "$REPO_DIR/experiments/branch_readout_gw9/analyze_readouts.py" && -f "$REPO_DIR/GW9_ANALYSIS_SOURCE_MANIFEST.json" ]] || { printf 'Set GW9_REPO_DIR to the updated GW9 checkout.\n' >&2; exit 1; }
PYTHON_BIN="${PYTHON_BIN:-$(command -v python || true)}"
[[ -n "$PYTHON_BIN" && -x "$PYTHON_BIN" ]] || { printf 'Activate igstgnn or set PYTHON_BIN.\n' >&2; exit 1; }
GW9_PROBE_RUN="${GW9_PROBE_RUN:-$REPO_DIR/experiments/chronological_runs/contra_original_gw9_probe_replayfix_s2025}"
GW9_ANALYSIS_TAG="${GW9_ANALYSIS_TAG:-$(date +%Y%m%d_%H%M%S)}"
GW9_ANALYSIS_DIR="${GW9_ANALYSIS_DIR:-${GW9_PROBE_RUN}_readout_analysis_${GW9_ANALYSIS_TAG}}"
[[ -d "$GW9_PROBE_RUN" ]] || { printf 'Probe directory does not exist: %s\n' "$GW9_PROBE_RUN" >&2; exit 1; }
[[ ! -e "$GW9_ANALYSIS_DIR" ]] || { printf 'Use a fresh analysis output directory: %s\n' "$GW9_ANALYSIS_DIR" >&2; exit 1; }
if [[ -n "${GW9_ANALYSIS_DATA:-}" ]]; then
  [[ -d "$GW9_ANALYSIS_DATA" ]] || { printf 'GW9_ANALYSIS_DATA does not exist: %s\n' "$GW9_ANALYSIS_DATA" >&2; exit 1; }
fi
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

cd "$REPO_DIR"
printf 'Repository: %s\nHost: %s\nProbe: %s\nAnalysis: %s\nHistory data: %s\n' "$REPO_DIR" "$(hostname)" "$GW9_PROBE_RUN" "$GW9_ANALYSIS_DIR" "${GW9_ANALYSIS_DATA:-omitted; input-history associations will be skipped}"
"$PYTHON_BIN" verify_gw9_analysis_bundle.py
"$PYTHON_BIN" -m unittest \
  tests.test_gw9_readout_metrics \
  tests.test_gw9_readout_auth \
  tests.test_gw9_readout_history \
  tests.test_gw9_readout_analysis -v
ANALYSIS_ARGS=(--probe-dir "$GW9_PROBE_RUN" --output-dir "$GW9_ANALYSIS_DIR")
if [[ -n "${GW9_ANALYSIS_DATA:-}" ]]; then ANALYSIS_ARGS+=(--data-dir "$GW9_ANALYSIS_DATA"); fi
"$PYTHON_BIN" -u -m experiments.branch_readout_gw9.analyze_readouts "${ANALYSIS_ARGS[@]}"
printf 'Read %s/report.md and %s/analysis.json. This diagnostic does not authorize joint training.\n' "$GW9_ANALYSIS_DIR" "$GW9_ANALYSIS_DIR"
