#!/usr/bin/env bash
# Run inside an existing allocation or on CPU. No package installation.
set -euo pipefail

if [[ $# -eq 1 && ( "$1" == --help || "$1" == -h ) ]]; then
    echo 'Usage: bash experiments/chronological/run_incident_physics_h1.sh [declared_contract.json]'
    echo 'Default: input audit + numerical tests + synthetic H1 integration check.'
    echo 'With a declared contract: also check two predetermined real training windows.'
    echo 'No full training, automatic capacity fitting, downloads, or CTM network run.'
    echo 'PHYSICS_DATA_DIR, PHYSICS_SENSORS, PHYSICS_DEVICE and PHYSICS_PYTHON may be set.'
    exit 0
fi
if [[ $# -gt 1 ]]; then
    echo 'Expected at most one declared contract path; use --help.' >&2
    exit 2
fi
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd -- "$SCRIPT_DIR/../.." && pwd)
CONTRACT=${1:-}
if [[ -n "$CONTRACT" ]]; then
    CONTRACT=$(cd -- "$(dirname -- "$CONTRACT")" && printf '%s/%s' "$PWD" "$(basename -- "$CONTRACT")")
    [[ -f "$CONTRACT" ]] || { echo 'Declared contract is missing.' >&2; exit 2; }
fi
cd "$REPO_DIR"

if [[ -z "${PHYSICS_PYTHON:-}" ]]; then
    if [[ "${CONDA_DEFAULT_ENV:-}" != igstgnn ]]; then
        if ! command -v conda >/dev/null 2>&1; then
            echo 'Activate igstgnn or set PHYSICS_PYTHON to its Python executable.' >&2
            exit 1
        fi
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate igstgnn
    fi
    PYTHON="${CONDA_PREFIX:?Activate igstgnn}/bin/python"
else
    PYTHON=$PHYSICS_PYTHON
fi
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=3
export PYTHONUNBUFFERED=1
export PYTHONUTF8=1
DATA_DIR=${PHYSICS_DATA_DIR:-../data/chronological/Contra_Costa_v8_dev}
SENSORS=${PHYSICS_SENSORS:-../data/xtraffic/Contra_Costa/sensors.csv}
DEVICE=${PHYSICS_DEVICE:-auto}
[[ -d "$DATA_DIR" ]] || { printf 'Missing data: %s\n' "$DATA_DIR" >&2; exit 1; }
mkdir -p experiments/chronological_runs
RUN_DIR=$(mktemp -d "$REPO_DIR/experiments/chronological_runs/contra_physics_h1_$(date +%Y%m%d_%H%M%S)_XXXXXX")
exec > >(tee "$RUN_DIR/run.log") 2>&1
trap 'STATUS=$?; printf "%s\n" "$STATUS" > "$RUN_DIR/exit_code"; printf "Workflow exit code: %s\nOutput directory: %s\n" "$STATUS" "$RUN_DIR"' EXIT
printf 'Started: %s\nHost: %s\nSlurm job: %s\nOutput directory: %s\n' "$(date -Is)" "$(hostname)" "${SLURM_JOB_ID:-unset}" "$RUN_DIR"

"$PYTHON" -m unittest discover -s tests -p 'test_incident_physics*.py' -v
AUDIT_ARGS=(--data-dir "$DATA_DIR" --sensors "$SENSORS" --output-dir "$RUN_DIR/input_audit")
if [[ -n "$CONTRACT" ]]; then AUDIT_ARGS+=(--contract "$CONTRACT"); fi
"$PYTHON" -u experiments/chronological/audit_incident_physics.py "${AUDIT_ARGS[@]}"
"$PYTHON" -u experiments/chronological/check_incident_physics.py --device "$DEVICE" --output-dir "$RUN_DIR/synthetic_check"
if [[ -n "$CONTRACT" ]]; then
    "$PYTHON" -u experiments/chronological/check_incident_physics.py --device "$DEVICE" \
        --data-dir "$DATA_DIR" --contract "$RUN_DIR/input_audit/declared_contract.json" --output-dir "$RUN_DIR/real_check"
fi
"$PYTHON" -u - "$RUN_DIR" <<'PY'
import json
from pathlib import Path
import sys
root = Path(sys.argv[1])
audit = json.loads((root / 'input_audit/summary.json').read_text(encoding='utf-8'))
report = {'status': 'H1_PREPARATION_COMPLETE', 'readiness': audit['readiness'],
          'full_training_started': False, 'ctm_real_data_enabled': False,
          'missing_physical_evidence': audit['missing_physical_evidence'],
          'output_dir': str(root)}
(root / 'summary.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
print(json.dumps(report, indent=2))
PY
