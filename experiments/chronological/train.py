"""Train one frozen chronological time-response screening run."""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import torch
from torch.utils.data import default_collate

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.chronological.smoke import (
    device_batch, forecast, make_model, set_seed, sha256, verify_package,
)
from src.utils.chronological import ChronologicalDataset, masked_flow_mae


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--variant', choices=('fixed', 'shared', 'conditioned', 'phase'), required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--protocol', type=Path,
                        default=Path(__file__).with_name('screening_v1.json'))
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stop-after-epoch', type=int)
    parser.add_argument('--check', action='store_true')
    return parser.parse_args(argv)


def epoch_plan(sample_indices, batch_size, seed, epoch):
    indices = [int(value) for value in sample_indices]
    if not indices or len(indices) != len(set(indices)) or batch_size < 1 or epoch < 1:
        raise ValueError('Epoch planning requires unique samples, positive batch size and epoch')
    random.Random(int(seed) * 1_000_003 + int(epoch)).shuffle(indices)
    batches = [indices[first:first + batch_size] for first in range(0, len(indices), batch_size)]
    encoded = json.dumps(indices, separators=(',', ':')).encode()
    return {'order': indices, 'batches': batches, 'tail_batch_size': len(batches[-1]),
            'order_sha256': hashlib.sha256(encoded).hexdigest()}


def validate_protocol(protocol, package_hashes, train_samples, val_samples, stations):
    if protocol.get('scope') != 'conditional_offline_screening' or protocol.get('main_training_ready') is not False:
        raise ValueError('Protocol must retain the conditional screening boundary')
    if (protocol.get('data_summary_sha256') != package_hashes.get('summary.json') or
            protocol.get('context_summary_sha256') != package_hashes.get('context_manifest.json')):
        raise ValueError('Data package fingerprint differs from the frozen protocol')
    if [protocol.get(k) for k in ('train_samples', 'val_samples', 'stations')] != [
            train_samples, val_samples, stations]:
        raise ValueError('Data shape differs from the frozen protocol')
    if protocol.get('supervised_horizons') != 12 or protocol.get('selection_metric') != 'all_nodes.mae_macro':
        raise ValueError('Training target or selection metric differs from the frozen protocol')
    positive = ('batch_size', 'max_epochs', 'patience', 'learning_rate', 'adam_eps',
                'clip_grad_norm', 'lr_gamma', 'free_test_session_hours')
    if any(not isinstance(protocol.get(key), (int, float)) or protocol[key] <= 0 for key in positive):
        raise ValueError('Protocol contains a non-positive training setting')
    milestones = protocol.get('lr_milestones')
    if (not isinstance(milestones, list) or not milestones or milestones != sorted(set(milestones))
            or any(not isinstance(value, int) or value < 1 for value in milestones)):
        raise ValueError('Protocol learning-rate milestones are invalid')
    if protocol.get('min_delta') != 0:
        raise ValueError('Screening protocol requires strict validation improvement')
    return dict(protocol)


def save_checkpoint(path, payload):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    torch.save(payload, temporary)
    temporary.replace(path)


def restore_checkpoint(path, expected_identity, model, optimizer, scheduler):
    import torch
    checkpoint = torch.load(Path(path), map_location='cpu', weights_only=False)
    if checkpoint.get('format_version') != 1 or checkpoint.get('identity') != expected_identity:
        raise ValueError('Checkpoint identity differs from this run')
    model.load_state_dict(checkpoint['model_state'], strict=True)
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    scheduler.load_state_dict(checkpoint['scheduler_state'])
    return checkpoint


class MetricTotals:
    def __init__(self, horizons=12):
        self.absolute = np.zeros(horizons, dtype=np.float64)
        self.squared = np.zeros(horizons, dtype=np.float64)
        self.counts = np.zeros(horizons, dtype=np.int64)
        self.percent = np.zeros(horizons, dtype=np.float64)
        self.positive_counts = np.zeros(horizons, dtype=np.int64)

    def update(self, prediction, target, valid):
        valid = valid.bool()
        difference = prediction.double() - target.double()
        absolute = difference.abs()
        dimensions = (0, 2, 3)
        selected_absolute = torch.where(valid, absolute, torch.zeros_like(absolute))
        positive = valid & (target > 0)
        denominator = torch.where(positive, target, torch.ones_like(target))
        selected_percent = torch.where(positive, absolute / denominator, torch.zeros_like(absolute))
        self.absolute += selected_absolute.sum(dim=dimensions).detach().cpu().double().numpy()
        self.squared += selected_absolute.square().sum(dim=dimensions).detach().cpu().double().numpy()
        self.counts += valid.sum(dim=dimensions).detach().cpu().numpy()
        self.percent += selected_percent.sum(dim=dimensions).detach().cpu().double().numpy()
        self.positive_counts += positive.sum(dim=dimensions).detach().cpu().numpy()

    def result(self):
        if np.any(self.counts == 0):
            raise ValueError('At least one horizon has no valid targets')
        mae = self.absolute / self.counts
        rmse = np.sqrt(self.squared / self.counts)
        mape = np.divide(
            self.percent,
            self.positive_counts,
            out=np.full_like(self.percent, np.nan),
            where=self.positive_counts > 0,
        )
        return {
            'mae_macro': float(mae.mean()),
            'rmse_macro': float(rmse.mean()),
            'mape_macro': float(mape.mean()) if np.isfinite(mape).all() else None,
            'mae_pooled': float(self.absolute.sum() / self.counts.sum()),
            'rmse_pooled': float(np.sqrt(self.squared.sum() / self.counts.sum())),
            'mape_pooled': (float(self.percent.sum() / self.positive_counts.sum())
                            if self.positive_counts.sum() else None),
            'per_horizon_mae': mae.tolist(),
            'per_horizon_rmse': rmse.tolist(),
            'per_horizon_mape': [float(value) if np.isfinite(value) else None for value in mape],
            'valid_count_per_horizon': self.counts.tolist(),
            'positive_target_count_per_horizon': self.positive_counts.tolist(),
        }


def batches(dataset, sample_indices, batch_size):
    position = {int(row['sample_index']): i for i, row in enumerate(dataset.rows)}
    for first in range(0, len(sample_indices), batch_size):
        selected = sample_indices[first:first + batch_size]
        yield default_collate([dataset[position[index]] for index in selected])


def train_epoch(model, optimizer, dataset, plan, device, scaler, clip_grad_norm):
    model.train()
    totals = MetricTotals()
    updates = 0
    for batch in batches(dataset, plan['order'], plan['batch_size']):
        batch = device_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        prediction = forecast(model, batch, scaler)
        loss = masked_flow_mae(prediction, batch['y_flow'], batch['y_valid'])
        loss.backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
            raise ValueError('Training produced absent or non-finite gradients')
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        optimizer.step()
        if not torch.isfinite(loss) or not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
            raise ValueError('Training produced a non-finite loss or parameter')
        totals.update(prediction.detach(), batch['y_flow'], batch['y_valid'])
        updates += 1
    return totals.result(), updates


def evaluate(model, dataset, sample_indices, batch_size, device, scaler, collect=False):
    model.eval()
    totals = MetricTotals()
    associated_totals = MetricTotals()
    saved = ({'prediction': [], 'target': [], 'valid': [], 'associated': []}
             if collect else None)
    with torch.inference_mode():
        for batch in batches(dataset, sample_indices, batch_size):
            batch = device_batch(batch, device)
            prediction = forecast(model, batch, scaler)
            if not torch.isfinite(prediction).all():
                raise ValueError('Validation produced non-finite predictions')
            totals.update(prediction, batch['y_flow'], batch['y_valid'])
            associated = (batch['incident']['distances'].abs().sum(-1) > 0)
            associated = associated[:, None, :, None].expand_as(batch['y_valid'])
            associated_valid = batch['y_valid'] & associated
            associated_totals.update(prediction, batch['y_flow'], associated_valid)
            if collect:
                saved['prediction'].append(prediction.detach().cpu().numpy())
                saved['target'].append(batch['y_flow'].cpu().numpy())
                saved['valid'].append(batch['y_valid'].cpu().numpy())
                saved['associated'].append(associated.cpu().numpy())
    all_nodes = totals.result()
    associated_nodes = associated_totals.result()
    for metrics in (all_nodes, associated_nodes):
        metrics['h7_h12_mae_macro'] = float(np.mean(metrics['per_horizon_mae'][6:]))
        metrics['h7_h12_rmse_macro'] = float(np.mean(metrics['per_horizon_rmse'][6:]))
    arrays = ({key: np.concatenate(value, axis=0) for key, value in saved.items()}
              if collect else None)
    return {'all_nodes': all_nodes, 'associated_nodes': associated_nodes}, arrays


def cpu_state(state):
    return {key: value.detach().cpu().clone() for key, value in state.items()}


def state_sha256(state):
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str(value.dtype).encode())
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def atomic_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def atomic_npz(path, **arrays):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.partial')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def build_model(data_dir, node_count, device, variant, seed):
    set_seed(seed)
    reference = make_model(data_dir, node_count, device, 'fixed')
    common = cpu_state(reference.state_dict())
    if variant == 'fixed':
        return reference, common, None
    set_seed(seed)
    model = make_model(data_dir, node_count, device, variant)
    extras = set(model.state_dict()) - set(common)
    missing, unexpected = model.load_state_dict(copy.deepcopy(common), strict=False)
    if unexpected or set(missing) != extras or any(
            not key.startswith('tiid_module.time_response.') for key in extras):
        raise ValueError('Unexpected differences in common time-response state')
    for key, value in common.items():
        if not torch.equal(model.state_dict()[key].cpu(), value):
            raise ValueError(f'Common initialization mismatch: {key}')
    return model, common, reference


def make_summary(status, args, protocol, identity, state):
    return {
        'status': status,
        'scope': protocol['scope'],
        'scientific_support': ('NOT_EVALUATED' if args.check else
                               'CONDITIONAL_OFFLINE_SCREENING_ONLY'),
        'main_training_ready': False,
        'variant': args.variant,
        'seed': args.seed,
        'completed_epoch': state['completed_epoch'],
        'best_epoch': state['best_epoch'],
        'best_metric': state['best_metric'],
        'wait': state['wait'],
        'global_updates': state['global_updates'],
        'history': state['history'],
        'runtime_epochs': state['runtime_epochs'],
        'identity': identity,
        'selection_metric': protocol['selection_metric'],
    }


def configure_determinism(device):
    if (device.type == 'cuda' and
            os.environ.get('CUBLAS_WORKSPACE_CONFIG') not in (':4096:8', ':16:8')):
        raise RuntimeError(
            'CUDA runs require CUBLAS_WORKSPACE_CONFIG=:4096:8 before Python starts')
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends.cuda.matmul, 'allow_tf32'):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends.cudnn, 'allow_tf32'):
        torch.backends.cudnn.allow_tf32 = False


def main(argv=None):
    args = parse_args(argv)
    if args.seed < 1:
        raise ValueError('Seed must be positive')
    if args.stop_after_epoch is not None and args.stop_after_epoch < 1:
        raise ValueError('Pause epoch must be positive')
    output_dir = args.output_dir.resolve()
    if args.resume:
        if not output_dir.is_dir():
            raise FileNotFoundError('Resume output directory does not exist')
    elif output_dir.exists():
        raise FileExistsError('Output directory already exists; use a new run directory')

    thread_count = max(1, int(os.environ.get('OMP_NUM_THREADS', '3')))
    torch.set_num_threads(thread_count)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but not available')
    configure_determinism(device)

    data_dir = args.data_dir.resolve()
    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text())
    package_hashes = verify_package(data_dir)
    train = ChronologicalDataset(data_dir, 'train')
    val = ChronologicalDataset(data_dir, 'val')
    validate_protocol(protocol, package_hashes, len(train), len(val), len(train.station_ids))
    if protocol.get('candidate_variant') not in (None, args.variant):
        raise ValueError('Protocol candidate variant differs from the requested model')
    if not np.array_equal(train.station_ids, val.station_ids):
        raise ValueError('Train and validation station axes differ')

    full_train_indices = [int(row['sample_index']) for row in train.rows]
    full_val_indices = [int(row['sample_index']) for row in val.rows]
    train_indices = full_train_indices[:2] if args.check else full_train_indices
    val_indices = full_val_indices[:2] if args.check else full_val_indices
    batch_size = min(2, len(train_indices), len(val_indices)) if args.check else protocol['batch_size']
    max_epochs = 2 if args.check else protocol['max_epochs']
    patience = 2 if args.check else protocol['patience']
    if batch_size < 1 or not train_indices or not val_indices:
        raise ValueError('Training and validation require at least one sample')
    if args.stop_after_epoch is not None and args.stop_after_epoch > max_epochs:
        raise ValueError('Pause epoch exceeds this run maximum')

    model, common, reference = build_model(
        data_dir, len(train.station_ids), device, args.variant, args.seed)
    initial_batch = device_batch(next(batches(train, train_indices[:batch_size], batch_size)), device)
    model.eval()
    with torch.inference_mode():
        initial_prediction = forecast(model, initial_batch, train.scaler)
        if reference is None:
            initial_difference = 0.
        else:
            reference.eval()
            reference_prediction = forecast(reference, initial_batch, train.scaler)
            initial_difference = float((initial_prediction - reference_prediction).abs().max())
    if initial_difference > .001:
        raise ValueError(f'{args.variant} initial output differs from A by {initial_difference}')
    del reference, initial_batch, initial_prediction
    optimizer = torch.optim.Adam(
        model.parameters(), lr=protocol['learning_rate'], weight_decay=protocol['weight_decay'],
        eps=protocol['adam_eps'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=protocol['lr_milestones'], gamma=protocol['lr_gamma'])
    source_paths = [Path(__file__), REPO / 'experiments/chronological/smoke.py',
                    REPO / 'src/models/igstgnn.py', REPO / 'src/models/incident_response.py',
                    REPO / 'src/utils/chronological.py']
    identity = {
        'variant': args.variant,
        'seed': args.seed,
        'check': args.check,
        'device': str(device),
        'protocol_sha256': sha256(protocol_path),
        'package_sha256': package_hashes,
        'common_initialization_sha256': state_sha256(common),
        'batch_size': batch_size,
        'max_epochs': max_epochs,
        'source_sha256': {str(path.relative_to(REPO)): sha256(path) for path in source_paths},
        'torch_version': torch.__version__,
        'numpy_version': np.__version__,
    }
    state = {
        'completed_epoch': 0,
        'best_model_state': cpu_state(model.state_dict()),
        'best_epoch': 0,
        'best_metric': float('inf'),
        'wait': 0,
        'global_updates': 0,
        'history': [],
        'runtime_epochs': [],
    }
    checkpoint_path = output_dir / 'last_checkpoint.pt'
    if args.resume:
        restored = restore_checkpoint(checkpoint_path, identity, model, optimizer, scheduler)
        state.update({key: restored[key] for key in state})
    else:
        output_dir.mkdir(parents=True, exist_ok=False)

    start_epoch = state['completed_epoch'] + 1
    if start_epoch > max_epochs:
        raise ValueError('Run already reached its maximum epoch')
    if args.stop_after_epoch is not None and args.stop_after_epoch < start_epoch:
        raise ValueError('Pause epoch precedes the next epoch to run')

    stop_reason = 'max_epochs'
    for epoch in range(start_epoch, max_epochs + 1):
        plan = epoch_plan(train_indices, batch_size, args.seed, epoch)
        plan['batch_size'] = batch_size
        set_seed(args.seed * 1_000_003 + epoch)
        learning_rate = optimizer.param_groups[0]['lr']
        began = time.perf_counter()
        train_metrics, updates = train_epoch(
            model, optimizer, train, plan, device, train.scaler, protocol['clip_grad_norm'])
        validation_metrics, _ = evaluate(model, val, val_indices, batch_size, device, val.scaler)
        elapsed = time.perf_counter() - began
        metric = validation_metrics['all_nodes']['mae_macro']
        improved = metric < state['best_metric'] - protocol['min_delta']
        if improved:
            state['best_metric'] = metric
            state['best_epoch'] = epoch
            state['best_model_state'] = cpu_state(model.state_dict())
            state['wait'] = 0
        else:
            state['wait'] += 1
        scheduler.step()
        state['completed_epoch'] = epoch
        state['global_updates'] += updates
        state['history'].append({
            'epoch': epoch,
            'learning_rate': learning_rate,
            'next_learning_rate': optimizer.param_groups[0]['lr'],
            'train_order_sha256': plan['order_sha256'],
            'train_samples': len(plan['order']),
            'train_unique_samples': len(set(plan['order'])),
            'tail_batch_size': plan['tail_batch_size'],
            'train': train_metrics,
            'validation': validation_metrics,
            'improved': improved,
        })
        state['runtime_epochs'].append({'epoch': epoch, 'seconds': elapsed})
        payload = {
            'format_version': 1,
            'identity': identity,
            'completed_epoch': state['completed_epoch'],
            'model_state': cpu_state(model.state_dict()),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'best_model_state': state['best_model_state'],
            'best_epoch': state['best_epoch'],
            'best_metric': state['best_metric'],
            'wait': state['wait'],
            'global_updates': state['global_updates'],
            'history': state['history'],
            'runtime_epochs': state['runtime_epochs'],
        }
        save_checkpoint(checkpoint_path, payload)
        print(json.dumps({'epoch': epoch, 'train_mae_macro': train_metrics['mae_macro'],
                          'validation_mae_macro': metric, 'best_metric': state['best_metric'],
                          'seconds': elapsed}), flush=True)
        if args.stop_after_epoch == epoch:
            stop_reason = 'paused'
            break
        if state['wait'] >= patience:
            stop_reason = 'early_stopping'
            break

    if stop_reason == 'paused':
        status = 'PAUSED_AT_EPOCH_BOUNDARY'
    elif args.check:
        status = 'ENGINEERING_CHECK_PASS'
    else:
        status = 'CONDITIONAL_OFFLINE_SCREENING_COMPLETE'
    summary = make_summary(status, args, protocol, identity, state)
    summary['stop_reason'] = stop_reason
    summary['parameters'] = sum(parameter.numel() for parameter in model.parameters())
    summary['initial_max_abs_difference_from_A'] = initial_difference
    summary['environment'] = {
        'python_version': sys.version,
        'torch_version': torch.__version__,
        'numpy_version': np.__version__,
        'cuda_version': torch.version.cuda,
        'device': str(device),
        'gpu_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'threads': torch.get_num_threads(),
        'cudnn_deterministic': torch.backends.cudnn.deterministic,
        'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
        'tf32': bool(torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32),
        'git_head': subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
    }
    if stop_reason != 'paused':
        model.load_state_dict(state['best_model_state'], strict=True)
        best_metrics, arrays = evaluate(
            model, val, val_indices, batch_size, device, val.scaler, collect=True)
        save_checkpoint(output_dir / 'best_model.pt', state['best_model_state'])
        atomic_npz(
            output_dir / 'best_validation_predictions.npz',
            **arrays,
            sample_indices=np.asarray(val_indices, dtype=np.int64),
            station_ids=np.asarray(val.station_ids, dtype=np.int64),
        )
        summary['best_validation_metrics'] = best_metrics
        summary['best_validation_predictions_sha256'] = sha256(
            output_dir / 'best_validation_predictions.npz')
    atomic_json(output_dir / 'summary.json', summary)
    print(f"Saved run summary: {output_dir / 'summary.json'}", flush=True)


if __name__ == '__main__':
    main()
