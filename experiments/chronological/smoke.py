"""Run equal-initialization A/B/C checks on real conditional train/val batches."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.models.igstgnn import IGSTGNN
from src.utils.chronological import ChronologicalDataset, masked_flow_mae, flow_metrics


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verify_package(directory):
    report = {}
    for name, key in [('summary.json', 'files'), ('context_manifest.json', 'outputs')]:
        path = directory / name
        meta = json.loads(path.read_text())
        if name == 'summary.json' and (meta.get('source_version') != 8 or not meta.get('build_complete')):
            raise ValueError('Raw flow package is incomplete or has a different source version')
        for filename, expected in meta[key].items():
            if sha256(directory / filename) != expected:
                raise ValueError(f'Package checksum mismatch: {filename}')
        report[name] = sha256(path)
    context = json.loads((directory / 'context_manifest.json').read_text())
    if context['schema'] != 'report_location_v1' or context['scope'] != 'conditional_development':
        raise ValueError('Unexpected event input schema or scope')
    return report


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def make_model(data_dir, node_count, device, variant):
    adjacency = np.load(data_dir / 'adjacency.npy', allow_pickle=False)
    if adjacency.shape != (node_count, node_count) or not np.isfinite(adjacency).all():
        raise ValueError('Adjacency and package node axis disagree')
    supports = []
    for a in (adjacency, adjacency.T):
        sums = a.sum(axis=1, keepdims=True)
        transition = np.divide(a, sums, out=np.zeros_like(a), where=sums > 0)
        supports.append(torch.as_tensor(transition, dtype=torch.float32, device=device))
    return IGSTGNN(
        model_args=dict(num_feat=1, num_hidden=32, node_hidden=12, time_emb_dim=12,
                        layer=5, k_s=2, k_t=3, tpd=288, dropout=.1, gap=3, sigma_t=1.,
                        lambda_incident=1., adjs=supports, incident_schema='report_location_v1',
                        time_response=variant),
        node_num=node_count, input_dim=3, output_dim=1, seq_len=12, horizon=12,
        dataset='Contra_Costa', data_path=str(data_dir), use_sensor_info=False).to(device)


def device_batch(batch, device):
    return {k: ({key: value.to(device) for key, value in v.items()} if isinstance(v, dict)
                else v.to(device)) for k, v in batch.items()}


def forecast(model, batch, scaler):
    return model(batch['x'], incident_data=batch['incident']) * scaler['std'] + scaler['mean']


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--bs', type=int, default=2)
    parser.add_argument('--steps', type=int, default=2)
    parser.add_argument('--seed', type=int, default=2025)
    args = parser.parse_args()
    if args.steps < 2 or args.bs < 1:
        parser.error('Use at least two gradient updates and a positive batch size')
    if args.output_dir.exists():
        raise FileExistsError('Output directory already exists; use a new run directory')
    torch.set_num_threads(3)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but not available')
    package_hashes = verify_package(args.data_dir)
    train = ChronologicalDataset(args.data_dir, 'train')
    val = ChronologicalDataset(args.data_dir, 'val')
    if args.bs > min(len(train), len(val)):
        raise ValueError('Requested batch is larger than the development split')
    # First chronological rows are chosen before inspecting targets or errors.
    train_batch = device_batch(next(iter(DataLoader(Subset(train, range(args.bs)), batch_size=args.bs))), device)
    val_batch = device_batch(next(iter(DataLoader(Subset(val, range(args.bs)), batch_size=args.bs))), device)
    if not torch.any(train_batch['incident']['distances'] != 0):
        raise ValueError('Fixed real training batch has no spatially connected nodes')
    set_seed(args.seed)
    reference = make_model(args.data_dir, len(train.station_ids), device, 'fixed')
    common = {k: v.detach().cpu().clone() for k, v in reference.state_dict().items()}
    reference.eval()
    with torch.inference_mode():
        initial_prediction = forecast(reference, train_batch, train.scaler).detach().cpu().numpy()
    del reference
    results, predictions = {}, {'train_initial_A': initial_prediction}
    for variant in ('fixed', 'shared', 'conditioned'):
        set_seed(args.seed)
        model = make_model(args.data_dir, len(train.station_ids), device, variant)
        extras = set(model.state_dict()) - set(common)
        missing, unexpected = model.load_state_dict(copy.deepcopy(common), strict=False)
        if unexpected or set(missing) != extras or any(not k.startswith('tiid_module.time_response.') for k in extras):
            raise ValueError('Unexpected differences in common A/B/C state')
        for key, value in common.items():
            if not torch.equal(model.state_dict()[key].cpu(), value):
                raise ValueError(f'Common initialization mismatch: {key}')
        model.eval()
        with torch.inference_mode():
            actual = forecast(model, train_batch, train.scaler).detach().cpu().numpy()
        initial_diff = float(np.max(np.abs(actual - initial_prediction)))
        if initial_diff > .001:
            raise ValueError(f'{variant} initial output differs from A by {initial_diff}')
        optimizer = torch.optim.Adam(model.parameters(), lr=.002, weight_decay=1e-5, eps=1e-8)
        before = {name: p.detach().clone() for name, p in model.named_parameters()}
        losses, grad_norms, response_gradients = [], [], []
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        synchronize(device)
        began = time.perf_counter()
        for step in range(args.steps):
            set_seed(args.seed + step)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            prediction = forecast(model, train_batch, train.scaler)
            loss = masked_flow_mae(prediction, train_batch['y_flow'], train_batch['y_valid'])
            loss.backward()
            gradients = [p.grad for p in model.parameters() if p.grad is not None]
            if not gradients or not all(torch.isfinite(g).all() for g in gradients):
                raise ValueError(f'{variant} has non-finite or absent gradients')
            # Inspect loss gradients before Adam weight decay can move parameters.
            response = {name: p for name, p in model.named_parameters()
                        if name.startswith('tiid_module.time_response.')}
            current_response_gradients = {}
            if response:
                if any(p.grad is None for p in response.values()):
                    raise ValueError(f'{variant} time response is disconnected from loss')
                current_response_gradients = {name: float(p.grad.abs().sum()) for name, p in response.items()}
                h12 = float(model.tiid_module.time_response.b.grad[-1].abs())
                current_response_gradients['h12_offset_gradient_abs'] = h12
                if h12 == 0:
                    raise ValueError(f'{variant} real batch has no H12 offset loss gradient')
                if variant == 'conditioned' and step >= 1:
                    if float(model.tiid_module.time_response.mlp[0].weight.grad.abs().sum()) == 0:
                        raise ValueError('C early layer has no loss gradient after the first update')
            response_gradients.append(current_response_gradients)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            if not all(torch.isfinite(p).all() for p in model.parameters()):
                raise ValueError(f'{variant} has non-finite updated parameters')
            losses.append(float(loss.detach()))
            grad_norms.append(float(grad_norm))
        synchronize(device)
        elapsed = time.perf_counter() - began
        changed = [name for name, p in model.named_parameters() if not torch.equal(p.detach(), before[name])]
        if not changed:
            raise ValueError('Optimizer did not update any model parameter')
        model.eval()
        with torch.inference_mode():
            validation = forecast(model, val_batch, val.scaler).detach().cpu().numpy()
        predictions[f'{variant}_val_after_updates'] = validation
        results[variant] = {
            'parameters': sum(p.numel() for p in model.parameters()),
            'extra_time_response_parameters': sum(p.numel() for name, p in model.named_parameters()
                                                  if name.startswith('tiid_module.time_response.')),
            'common_initialization_exact': True, 'initial_max_abs_difference_from_A': initial_diff,
            'training_batch_losses': losses, 'unclipped_gradient_norms': grad_norms,
            'time_response_loss_gradient_l1_before_optimizer': response_gradients,
            'updated_time_response_parameter_names': [name for name in changed
                                                       if name.startswith('tiid_module.time_response.')],
            'updated_parameter_tensors': len(changed), 'update_seconds': elapsed,
            'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
            'fixed_validation_batch_metrics_NOT_model_comparison': flow_metrics(
                validation, val_batch['y_flow'].cpu().numpy(), val_batch['y_valid'].cpu().numpy())}
        print(json.dumps({'variant': variant, 'initial_difference': initial_diff,
                          'finite_updates': len(losses), 'parameters': results[variant]['parameters'],
                          'seconds': elapsed}), flush=True)
        del model, optimizer, before
    args.output_dir.mkdir(parents=True, exist_ok=False)
    predictions.update(val_y=val_batch['y_flow'].cpu().numpy(), val_valid=val_batch['y_valid'].cpu().numpy())
    np.savez_compressed(args.output_dir / 'batch_predictions.npz', **predictions)
    source_paths = [Path(__file__), REPO / 'src/models/igstgnn.py', REPO / 'src/utils/chronological.py']
    response_file = REPO / 'src/models/incident_response.py'
    if response_file.exists():
        source_paths.append(response_file)
    report = {'status': 'CONDITIONAL_REAL_BATCH_SMOKE_PASS', 'scientific_support': 'NOT_EVALUATED',
              'scope': 'Engineering only: no full epoch, no full validation, no test, no accuracy ranking',
              'main_training_ready': False, 'data_package': str(args.data_dir.resolve()),
              'package_summary_sha256': package_hashes,
              'train_sample_indices': [int(r['sample_index']) for r in train.rows[:args.bs]],
              'val_sample_indices': [int(r['sample_index']) for r in val.rows[:args.bs]],
              'batch_size': args.bs, 'updates_per_variant': args.steps, 'seed': args.seed,
              'device': str(device), 'torch_version': torch.__version__, 'numpy_version': np.__version__,
              'code_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
              'source_sha256': {str(p.relative_to(REPO)): sha256(p) for p in source_paths},
              'train_valid_target_count': int(train_batch['y_valid'].sum()),
              'train_true_zero_target_count': int(((train_batch['y_flow'] == 0) & train_batch['y_valid']).sum()),
              'train_connected_nodes_per_sample': (train_batch['incident']['distances'].abs().sum(-1) > 0)
                                                    .sum(-1).cpu().tolist(),
              'variants': results, 'prediction_sha256': sha256(args.output_dir / 'batch_predictions.npz')}
    (args.output_dir / 'summary.json').write_text(json.dumps(report, indent=2, ensure_ascii=False,
                                                           allow_nan=False) + '\n')
    print(f"Saved engineering evidence: {args.output_dir / 'summary.json'}", flush=True)


if __name__ == '__main__':
    main()
