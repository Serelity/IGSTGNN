#!/usr/bin/env bash
# Run inside an allocated V100 job; Slurm resources are selected by the caller.
set -euo pipefail

if [[ $# -eq 1 && ( "$1" == --help || "$1" == -h ) ]]; then
    echo 'Usage: bash experiments/chronological/run_incident_routing_p2.sh'
    echo 'Runs P2 tests, then one full epoch each of fixed/acdg (seed 2025).'
    echo 'Use the existing igstgnn environment on an allocated V100.'
    echo 'Optional P2_DATA_DIR overrides ../data/chronological/Contra_Costa_v8_dev.'
    echo 'Fresh outputs and run.log are saved under experiments/chronological_runs/.'
    exit 0
fi
if [[ $# -ne 0 ]]; then
    echo 'This first-epoch launcher accepts no arguments; use --help.' >&2
    exit 2
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd -- "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_DIR"

if [[ "${CONDA_DEFAULT_ENV:-}" != igstgnn ]]; then
    if ! command -v conda >/dev/null 2>&1; then
        echo 'Activate the existing igstgnn environment before starting this job.' >&2
        exit 1
    fi
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate igstgnn
fi
export PATH="${CONDA_PREFIX:?Activate igstgnn before running}/bin:$PATH"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=3
export PYTHONUNBUFFERED=1

# Allow launcher/documentation commits, but preserve the validated model,
# data handling, test, training entry point and scientific protocol.
VALIDATED_COMMIT=782f8e0ad44ffcbd738b8573ae680c4f998d25af
FROZEN_PATHS=(
    src
    experiments/chronological/train.py
    experiments/chronological/smoke.py
    experiments/chronological/incident_routing_p2.json
    tests/test_acdg.py
)
verify_sources() {
    if ! git diff --quiet "$VALIDATED_COMMIT" -- "${FROZEN_PATHS[@]}"; then
        echo 'P2 model/training sources differ from the validated version. Preserve the run and review the changes.' >&2
        exit 1
    fi
}
verify_sources

DATA_DIR=${P2_DATA_DIR:-../data/chronological/Contra_Costa_v8_dev}
if [[ ! -d "$DATA_DIR" ]]; then
    printf 'Missing data directory: %s\n' "$DATA_DIR" >&2
    exit 1
fi

mkdir -p experiments/chronological_runs
export P2_RUN_ROOT
P2_RUN_ROOT=$(mktemp -d "$REPO_DIR/experiments/chronological_runs/contra_p2_first_epoch_$(date +%Y%m%d_%H%M%S)_XXXXXX")
exec > >(tee "$P2_RUN_ROOT/run.log") 2>&1
trap 'STATUS=$?; printf "%s\n" "$STATUS" > "$P2_RUN_ROOT/exit_code"; printf "Workflow exit code: %s\nOutput directory: %s\n" "$STATUS" "$P2_RUN_ROOT"' EXIT

printf 'Started: %s\nHost: %s\nSlurm job: %s\nOutput directory: %s\n' \
    "$(date -Is)" "$(hostname)" "${SLURM_JOB_ID:-unset}" "$P2_RUN_ROOT"
git log -1 --format='commit=%H%nsubject=%s'

python -u - preflight <<'PY'
import sys
import torch

if not torch.cuda.is_available():
    raise SystemExit('CUDA unavailable: run this script in an allocated GPU job.')
gpu = torch.cuda.get_device_name(0)
if 'V100' not in gpu:
    raise SystemExit(f'This comparison requires V100; found {gpu}')
print({'python': sys.executable, 'torch': torch.__version__,
       'cuda': torch.version.cuda, 'gpu': gpu}, flush=True)
PY

python -m unittest discover -s tests -p 'test_acdg.py' -v

for variant in fixed acdg; do
    verify_sources
    printf 'Starting full first epoch: %s\n' "$variant"
    python -u experiments/chronological/train.py \
        --data-dir "$DATA_DIR" \
        --output-dir "$P2_RUN_ROOT/$variant" \
        --protocol experiments/chronological/incident_routing_p2.json \
        --variant "$variant" \
        --device cuda:0 \
        --seed 2025 \
        --stop-after-epoch 1
done
verify_sources

python -u - "$P2_RUN_ROOT" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
results = {}

def require(condition, message):
    if not condition:
        raise RuntimeError(message)

for variant in ('fixed', 'acdg'):
    directory = root / variant
    summary = json.loads((directory / 'summary.json').read_text())
    require(summary['variant'] == variant, 'Variant mismatch')
    require(summary['status'] == 'PAUSED_AT_EPOCH_BOUNDARY', 'Unexpected run status')
    require(summary['completed_epoch'] == 1, 'Expected exactly one full epoch')
    require(summary['global_updates'] == 76, 'Expected 76 Adam updates')
    require(summary['identity']['check'] is False, 'This must not be a small --check run')
    require(summary['identity']['seed'] == 2025, 'Seed mismatch')
    require(summary['identity']['batch_size'] == 48, 'Batch size mismatch')
    require(summary['initial_max_abs_difference_from_A'] <= 0.001, 'Initial output mismatch')
    require((directory / 'last_checkpoint.pt').is_file(), 'Missing resume checkpoint')
    results[variant] = summary

baseline, candidate = results['fixed'], results['acdg']
for key in ('common_initialization_sha256', 'protocol_sha256',
            'package_sha256', 'source_sha256'):
    require(baseline['identity'][key] == candidate['identity'][key], f'Paired identity mismatch: {key}')
require(baseline['history'][0]['train_order_sha256'] ==
        candidate['history'][0]['train_order_sha256'], 'Training sample order mismatch')

report = {'status': 'PAIR_IDENTITY_CHECK_PASS', 'runs': {},
          'additional_parameters': candidate['parameters'] - baseline['parameters'],
          'scientific_status': 'FIRST_EPOCH_TIMING_ONLY_NO_PREDICTIVE_GAIN_CLAIM'}
for variant, summary in results.items():
    report['runs'][variant] = {
        'output_dir': str(root / variant), 'status': summary['status'],
        'updates': summary['global_updates'], 'parameters': summary['parameters'],
        'epoch_seconds': summary['runtime_epochs'][0]['seconds'],
        'train_mae': summary['history'][0]['train']['mae_macro'],
        'validation_mae': summary['best_metric'],
        'initial_difference': summary['initial_max_abs_difference_from_A'],
    }
(root / 'pair_report.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2), flush=True)
PY
