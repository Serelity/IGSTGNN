"""H1 engineering check only; no full epoch or model-selection claim."""

import argparse
import copy
import json
from pathlib import Path
import platform
import sys

import numpy as np
import torch
from torch.utils.data import default_collate

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.chronological.smoke import device_batch, make_model, set_seed, sha256, verify_package
from src.models.igstgnn import IGSTGNN
from src.models.incident_queue import QueueAugmentedIGSTGNN
from src.utils.chronological import ChronologicalDataset, masked_flow_mae
from src.utils.traffic_physics_contract import build_physical_inputs, queue_config, validate_contract


def synthetic_fixture(device):
    """Three artificial detectors, arbitrary toy units, not a real bottleneck."""
    args = dict(num_feat=1, num_hidden=32, node_hidden=12, time_emb_dim=12,
                layer=5, k_s=2, k_t=3, tpd=288, dropout=.1, gap=3, sigma_t=1.,
                lambda_incident=1., adjs=[torch.eye(3, device=device)] * 2,
                incident_schema='report_location_v1', time_response='fixed',
                dataset='synthetic', use_sensor_info=False)
    set_seed(2025)
    reference = IGSTGNN(args, node_num=3, input_dim=3, output_dim=1, seq_len=12, horizon=12).to(device)
    x = torch.zeros(2, 12, 3, 3, device=device)
    x[..., 0] = torch.linspace(.2, 1.4, 12, device=device)[None, :, None]
    x[..., 1], x[..., 2] = .25, 2 / 7
    incident = {'report_age_minutes': torch.tensor([2., 4.], device=device),
                'forecast_tod': torch.tensor([80, 90], device=device),
                'forecast_dow': torch.tensor([2, 3], device=device),
                'distances': torch.tensor([[[0., .5, 1.], [0., .8, 0.], [0., 0., 0.]]] * 2, device=device)}
    batch = {'x': x, 'incident': incident, 'x_valid': torch.ones_like(x[..., :1], dtype=torch.bool),
             'y_flow': torch.full((2, 12, 3, 1), 5., device=device),
             'y_valid': torch.ones(2, 12, 3, 1, device=device, dtype=torch.bool)}
    config = {'capacities': [600.], 'queue_scales': [120.],
              'readout_weights': [[0., 1., 0.]], 'step_minutes': 1, 'gap_minutes': 10}
    physical = {'history_rates': torch.full((2, 12, 1, 2), 400., device=device),
                'history_valid': torch.ones(2, 12, 1, 2, device=device, dtype=torch.bool),
                'event_features': torch.tensor([[[.4, 0., .8, 0.]], [[.8, 0., .8, 0.]]], device=device),
                'report_support': torch.ones(2, 1, device=device, dtype=torch.bool)}
    return reference, batch, physical, config, {'mean': 0., 'std': 1.}


def augmented(reference, config, mode, device):
    set_seed(2025)
    model = QueueAugmentedIGSTGNN(
        copy.deepcopy(reference._model_args), {**config, 'mode': mode},
        node_num=reference.node_num, input_dim=3, output_dim=1, seq_len=12, horizon=12).to(device)
    missing, unexpected = model.load_state_dict(reference.state_dict(), strict=False)
    if unexpected or any(not name.startswith('queue_branch.') for name in missing):
        raise ValueError('Unexpected difference from the complete fixed backbone')
    for key, value in reference.state_dict().items():
        if not torch.equal(value, model.state_dict()[key]):
            raise ValueError(f'Common state mismatch: {key}')
    return model


def run_check(output_dir, device='cpu', data_dir=None, contract_path=None):
    if device == 'auto':
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    output_dir, device = Path(output_dir), torch.device(device)
    if output_dir.exists():
        raise FileExistsError('Use a fresh engineering output directory')
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable')
    if (data_dir is None) != (contract_path is None):
        raise ValueError('Real data requires both package and declared contract')
    torch.set_num_threads(3)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    set_seed(2025)
    package_hashes = None
    if data_dir is None:
        reference, batch, physical, config, scaler = synthetic_fixture(device)
        input_mode = 'SYNTHETIC_ONLY_NOT_REAL_PHYSICS_EVIDENCE'
    else:
        data_dir, contract_path = Path(data_dir), Path(contract_path)
        package_hashes = verify_package(data_dir)
        dataset = ChronologicalDataset(data_dir, 'train')
        contract = validate_contract(json.loads(contract_path.read_text(encoding='utf-8')), dataset.station_ids, package_hashes)
        if len(dataset) < 2:
            raise ValueError('Engineering check requires two predetermined training samples')
        batch = device_batch(default_collate([dataset[0], dataset[1]]), device)
        reference = make_model(data_dir, len(dataset.station_ids), device, 'fixed')
        # Use only X; invalid filled values are discarded by the physical mask.
        history = batch['x'][..., 0] * dataset.scaler['std'] + dataset.scaler['mean']
        physical = build_physical_inputs(history, batch['x_valid'][..., 0], batch['incident'], contract)
        if not physical['report_support'].any():
            raise ValueError('Predetermined first two samples do not exercise any declared bottleneck; no target-based search performed')
        config, scaler = queue_config(contract, 'queue'), dataset.scaler
        config.pop('mode')
        input_mode = 'TWO_REAL_TRAIN_WINDOWS_DECLARED_PHYSICAL_CONTRACT'
    reference.eval()
    with torch.no_grad():
        expected = reference(batch['x'], incident_data=batch['incident'])
    results, arrays = {}, {'fixed_initial': expected.cpu().numpy()}
    for mode in ('queue', 'recurrent'):
        model = augmented(reference, config, mode, device).eval()
        with torch.no_grad():
            initial = model(batch['x'], incident_data=batch['incident'], physical_inputs=physical)
        difference = float((initial - expected).abs().max())
        if difference != 0:
            raise ValueError(f'{mode}: zero-initialized output differs from native fixed by {difference}')
        optimizer = torch.optim.Adam(model.parameters(), lr=.002, weight_decay=0)
        history = []
        for step in range(2):
            set_seed(2025 + step)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch['x'], incident_data=batch['incident'], physical_inputs=physical)
            prediction = prediction * scaler['std'] + scaler['mean']
            loss = masked_flow_mae(prediction, batch['y_flow'], batch['y_valid'])
            loss.backward()
            grads = {n: p.grad for n, p in model.queue_branch.named_parameters()}
            if any(g is None or not torch.isfinite(g).all() for g in grads.values()):
                raise ValueError('Auxiliary branch has disconnected or nonfinite gradients')
            l1 = {n: float(g.abs().sum()) for n, g in grads.items()}
            if l1['projection.weight'] <= 0 or (step == 1 and l1['history_encoder.weight_ih_l0'] <= 0):
                raise ValueError('Required projection/history gradient did not become active')
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            if any(not torch.isfinite(p).all() for p in model.parameters()):
                raise ValueError('Nonfinite updated parameters')
            history.append({'loss_NOT_comparison': float(loss.detach()), 'branch_gradient_l1': l1})
        model.eval()
        with torch.no_grad():
            actual = model(batch['x'], incident_data=batch['incident'], physical_inputs=physical)
        arrays[f'{mode}_after_two_updates'] = actual.cpu().numpy()
        trace = model.last_queue_trace
        balance = None
        if mode == 'queue':
            residual = trace['queue'][:, 1:] - trace['queue'][:, :-1] - model.queue_branch.dt_hours * (trace['arrival'] - trace['departure'])
            balance = float(residual.abs().max())
            scale = max(1., float(trace['queue'].abs().max()))
            if balance > 64 * torch.finfo(actual.dtype).eps * scale:
                raise ValueError('Internal point-queue conservation check failed')
            for key in ('queue', 'arrival', 'capacity', 'departure'):
                arrays[key] = trace[key].cpu().numpy()
        results[mode] = {'initial_max_abs_difference': difference, 'optimizer_updates': 2,
                         'total_parameters': sum(p.numel() for p in model.parameters()),
                         'branch_parameters': sum(p.numel() for p in model.queue_branch.parameters()),
                         'queue_balance_abs_max': balance, 'updates': history}
        print(json.dumps({'mode': mode, 'initial_difference': difference, 'updates': 2,
                          'branch_parameters': results[mode]['branch_parameters']}, allow_nan=False), flush=True)
    report = {'status': 'H1_ENGINEERING_CHECK_PASS', 'input_mode': input_mode,
              'scientific_status': 'NO_PREDICTIVE_GAIN_OR_PHYSICAL_IDENTIFICATION_CLAIM',
              'full_training_started': False, 'ctm_real_data_enabled': False,
              'runs': results, 'package_sha256': package_hashes,
              'contract_sha256': sha256(contract_path) if contract_path else None,
              'environment': {'python': platform.python_version(), 'torch': torch.__version__,
                              'device': str(device), 'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None},
              'source_sha256': {name: sha256(REPO / name) for name in (
                  'src/models/igstgnn.py', 'src/models/incident_queue.py', 'src/models/traffic_physics.py',
                  'src/utils/traffic_physics_contract.py', 'experiments/chronological/check_incident_physics.py')}}
    output_dir.mkdir(parents=True)
    np.savez_compressed(output_dir / 'engineering_arrays.npz', **arrays)
    (output_dir / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'runs'}, indent=2), flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--contract', type=Path)
    args = parser.parse_args(argv)
    run_check(args.output_dir, args.device, args.data_dir, args.contract)


if __name__ == '__main__':
    main()
