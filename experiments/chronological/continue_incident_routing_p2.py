"""Resume an existing P2 pair with the unchanged chronological trainer."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


REPO = Path(__file__).resolve().parents[2]
PROTOCOL = REPO / 'experiments/chronological/incident_routing_p2.json'
VARIANTS = ('fixed', 'acdg')
COMPLETE = 'CONDITIONAL_OFFLINE_SCREENING_COMPLETE'
PAUSED = 'PAUSED_AT_EPOCH_BOUNDARY'
SOURCE_NAMES = {
    'experiments/chronological/train.py', 'experiments/chronological/smoke.py',
    'src/models/igstgnn.py', 'src/models/incident_response.py',
    'src/utils/chronological.py',
}
PAIRED_KEYS = ('seed', 'check', 'device', 'protocol_sha256', 'package_sha256',
               'common_initialization_sha256', 'batch_size', 'max_epochs',
               'source_sha256', 'torch_version', 'numpy_version')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_summaries(root):
    return {variant: json.loads((root / variant / 'summary.json').read_text())
            for variant in VARIANTS}


def validate_pair(summaries, repo=None, protocol_path=None):
    repo = REPO if repo is None else repo
    protocol_path = PROTOCOL if protocol_path is None else protocol_path
    for variant, summary in summaries.items():
        identity = summary['identity']
        require(summary['variant'] == identity['variant'] == variant, 'Variant mismatch')
        require(summary['status'] in (PAUSED, COMPLETE), 'Expected a paused or completed full run')
        require(identity['check'] is False and identity['seed'] == 2025, 'Expected full seed2025 run')
        require(identity['batch_size'] == 48 and identity['max_epochs'] == 100, 'Training budget changed')
        require(identity['device'] == 'cuda:0', 'Expected the original CUDA device')
        require(summary['selection_metric'] == 'all_nodes.mae_macro', 'Selection metric changed')
        require(summary['initial_max_abs_difference_from_A'] <= .001, 'Initial prediction mismatch')
        require(identity['protocol_sha256'] == digest(protocol_path), 'Protocol changed since checkpoint')
        sources = identity['source_sha256']
        require(set(sources) == SOURCE_NAMES, 'Unexpected checkpoint source manifest')
        for name, expected in sources.items():
            require(digest(repo / name) == expected, f'Training source changed: {name}')
    a, b = summaries['fixed'], summaries['acdg']
    for key in PAIRED_KEYS:
        require(a['identity'][key] == b['identity'][key], f'Paired identity mismatch: {key}')
    for left, right in zip(a['history'], b['history']):
        require(left['epoch'] == right['epoch'] and
                left['train_order_sha256'] == right['train_order_sha256'],
                'Paired epoch/sample order mismatch')


def checkpoint_action(summary, checkpoint, directory, patience=20):
    require(checkpoint.get('format_version') == 1, 'Unsupported checkpoint format')
    require(checkpoint['identity'] == summary['identity'], 'Checkpoint/summary identity mismatch')
    epoch = checkpoint['completed_epoch']
    require(1 <= summary['completed_epoch'] <= epoch <= 100, 'Invalid checkpoint epoch')
    require(checkpoint['global_updates'] == epoch * 76, 'Checkpoint update count mismatch')
    require(len(checkpoint['history']) == epoch, 'Checkpoint history length mismatch')
    require(checkpoint['history'][:summary['completed_epoch']] == summary['history'],
            'Checkpoint and saved summary histories disagree')
    terminal = epoch == 100 or checkpoint['wait'] >= patience
    if summary['status'] == COMPLETE:
        require(terminal and epoch == summary['completed_epoch'], 'Completed summary is inconsistent')
        require(summary['stop_reason'] in ('max_epochs', 'early_stopping'), 'Invalid completion reason')
        require(summary['global_updates'] == checkpoint['global_updates'], 'Completed update count mismatch')
        require(summary['best_epoch'] == checkpoint['best_epoch'] and
                summary['best_metric'] == checkpoint['best_metric'], 'Selected checkpoint mismatch')
        for name in ('best_model.pt', 'best_validation_predictions.npz'):
            require((directory / name).is_file(), f'Missing completed artifact: {name}')
        require(digest(directory / 'best_validation_predictions.npz') ==
                summary['best_validation_predictions_sha256'], 'Prediction artifact checksum mismatch')
        return 'skip'
    require(not terminal,
            'Checkpoint already reached its stopping boundary but final export is incomplete. '
            'Preserve this directory and request export recovery; do not train extra epochs.')
    return 'resume'


def completion_report(summaries, root):
    report = {'status': 'P2_PAIRED_SCREENING_COMPLETE', 'runs': {},
              'scientific_status': 'SINGLE_SEED_REUSED_VALIDATION_SCREENING_NOT_CONFIRMATION'}
    for variant, summary in summaries.items():
        require(summary['status'] == COMPLETE, 'Both runs must complete before comparison')
        all_nodes = summary['best_validation_metrics']['all_nodes']
        require(summary['best_metric'] == all_nodes['mae_macro'], 'Selected metric mismatch')
        regions = {}
        for region in ('all_nodes', 'associated_nodes'):
            metrics = summary['best_validation_metrics'][region]
            mae = metrics['per_horizon_mae']
            require(len(mae) == 12, 'Expected 12 forecast horizons')
            regions[region] = {
                'mae_macro': metrics['mae_macro'], 'rmse_macro': metrics['rmse_macro'],
                'mape_macro': metrics['mape_macro'],
                'h1_h3_mae_macro': sum(mae[:3]) / 3,
                'h4_h6_mae_macro': sum(mae[3:6]) / 3,
                'h7_h12_mae_macro': sum(mae[6:]) / 6,
            }
        report['runs'][variant] = {
            'output_dir': str(root / variant), 'completed_epoch': summary['completed_epoch'],
            'best_epoch': summary['best_epoch'], 'stop_reason': summary['stop_reason'],
            'updates': summary['global_updates'], 'parameters': summary['parameters'],
            'epoch_runtime_seconds_total': sum(x['seconds'] for x in summary['runtime_epochs']),
            'metrics': regions,
        }
    a = report['runs']['fixed']['metrics']['all_nodes']['mae_macro']
    b = report['runs']['acdg']['metrics']['all_nodes']['mae_macro']
    report['selection_mae_gain_fixed_minus_acdg'] = a - b
    report['selection_relative_gain_percent'] = 100 * (a - b) / a if a else None
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.run_dir.resolve()
    summaries = load_summaries(root)
    validate_pair(summaries)
    protocol = json.loads(PROTOCOL.read_text())
    require(protocol['patience'] == 20 and protocol['max_epochs'] == 100, 'Frozen budget changed')

    import numpy as np
    import torch

    require(torch.__version__ == summaries['fixed']['identity']['torch_version'], 'PyTorch version changed')
    require(np.__version__ == summaries['fixed']['identity']['numpy_version'], 'NumPy version changed')
    plans = {}
    for variant in VARIANTS:
        directory = root / variant
        checkpoint = torch.load(directory / 'last_checkpoint.pt', map_location='cpu', weights_only=False)
        plans[variant] = checkpoint_action(summaries[variant], checkpoint, directory)
        print(json.dumps({'variant': variant, 'action': plans[variant],
                          'checkpoint_epoch': checkpoint['completed_epoch'],
                          'checkpoint_updates': checkpoint['global_updates']}), flush=True)
        del checkpoint

    if 'resume' in plans.values():
        require(torch.cuda.is_available(), 'CUDA unavailable: use an allocated V100 job')
        gpu = torch.cuda.get_device_name(0)
        require('V100' in gpu, f'Expected V100; found {gpu}')
        print(json.dumps({'python': sys.executable, 'torch': torch.__version__,
                          'numpy': np.__version__, 'gpu': gpu}), flush=True)
    for variant in VARIANTS:
        if plans[variant] == 'skip':
            continue
        validate_pair(load_summaries(root))
        # No --stop-after-epoch 100: that would leave the existing trainer
        # PAUSED at epoch 100 and bypass its selected-model/prediction export.
        subprocess.run([
            sys.executable, '-u', str(REPO / 'experiments/chronological/train.py'),
            '--data-dir', str(args.data_dir.resolve()), '--output-dir', str(root / variant),
            '--protocol', str(PROTOCOL), '--variant', variant, '--device', 'cuda:0',
            '--seed', '2025', '--resume',
        ], cwd=REPO, check=True)
        summary = json.loads((root / variant / 'summary.json').read_text())
        checkpoint = torch.load(root / variant / 'last_checkpoint.pt', map_location='cpu', weights_only=False)
        require(checkpoint_action(summary, checkpoint, root / variant) == 'skip', 'Run did not finish')
        del checkpoint

    summaries = load_summaries(root)
    validate_pair(summaries)
    report = completion_report(summaries, root)
    temporary = root / 'completion_report.json.partial'
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    temporary.replace(root / 'completion_report.json')
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
