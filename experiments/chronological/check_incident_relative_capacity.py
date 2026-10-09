"""M2.1 synthetic behavior + optional complete candidate train-X interface check."""
import argparse
import copy
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.models.incident_relative_capacity import SharedIncidentState, supply_limited_flow
from src.utils.incident_corridor import require, sha256, write_json, write_rows
from src.utils.relative_capacity_inputs import CapacityHistoryInputs

SOURCE_FILES = ('src/models/incident_relative_capacity.py', 'src/utils/relative_capacity_inputs.py',
                'experiments/chronological/check_incident_relative_capacity.py',
                'experiments/chronological/run_incident_relative_capacity.sh', 'tests/test_incident_relative_capacity.py',
                'src/models/igstgnn.py')


def synthetic_inputs(device):
    generator = torch.Generator().manual_seed(1701)
    history = .2 + torch.rand(2, 12, 4, 3, generator=generator)
    report = torch.zeros(2, 4, 3)
    report[:, ::2, 1] = .8
    report[:, ::2, 2] = 1
    return {'history': history.to(device), 'valid': torch.ones_like(history, dtype=torch.bool, device=device),
            'references': torch.ones(4, 3, device=device), 'report_features': report.to(device),
            'report_age_minutes': torch.full((2,), 3., device=device),
            'elapsed_minutes': torch.arange(0., 65., 5., device=device)}


def paired_models(device):
    torch.manual_seed(20261009)
    capacity = SharedIncidentState().to(device)
    ordinary = SharedIncidentState(mode='ordinary').to(device)
    ordinary.load_state_dict(capacity.state_dict())
    return {'capacity': capacity, 'ordinary': ordinary}


def synthetic_check(output, device):
    inputs = synthetic_inputs(device)
    controlled = SharedIncidentState().to(device).eval()
    with torch.no_grad():
        controlled.condition_head[-1].weight.zero_()
        controlled.condition_head[-1].bias.copy_(torch.tensor([2., np.log(25/150)], device=device))
        retention = controlled(**inputs)['state'][0, :, 0]
        baseline = torch.ones_like(retention)
        low, high, blocked = [torch.full_like(retention, v) for v in (.2, 2., .1)]
        low_flow = supply_limited_flow(low, baseline, retention, high)
        high_flow = supply_limited_flow(high, baseline, retention, high)
        blocked_flow = supply_limited_flow(high, baseline, retention, blocked)
        expected = 1 - .5 * torch.exp(-inputs['elapsed_minutes'] / 30)
        torch.testing.assert_close(retention, expected)
        torch.testing.assert_close(low_flow, low)
        torch.testing.assert_close(high_flow, retention)
        torch.testing.assert_close(blocked_flow, blocked)
    times, r, lo, hi, bl = [v.cpu().tolist() for v in (inputs['elapsed_minutes'], retention, low_flow, high_flow, blocked_flow)]
    rows = [dict(elapsed_minutes=t, capacity_retention=a, low_demand_flow=b, high_demand_flow=c, downstream_limited_flow=d)
            for t, a, b, c, d in zip(times, r, lo, hi, bl)]
    write_rows(output / 'synthetic_supply_cases.csv', rows)

    results = {}
    for mode, model in paired_models(device).items():
        optimizer = torch.optim.Adam(model.parameters(), lr=.003)
        initial = copy.deepcopy(model.state_dict())
        gradients, losses = [], []
        def step(current, opt):
            opt.zero_grad(set_to_none=True)
            # Artificial target ONLY tests the optimizer path, never real traffic.
            target = .7 + .1 * (inputs['elapsed_minutes'][None, :, None] / 60)
            loss = (current(**inputs)['state'] - target).square().mean()
            loss.backward()
            norms = {}
            for name, parameter in current.named_parameters():
                require(parameter.grad is not None and torch.isfinite(parameter.grad).all(), 'Disconnected or nonfinite gradient')
                norms[name] = float(parameter.grad.abs().sum())
                require(norms[name] > 0, 'No synthetic gradient: ' + name)
            opt.step()
            require(all(torch.isfinite(p).all() for p in current.parameters()), 'Nonfinite updated parameter')
            return float(loss.detach()), norms
        for _ in range(2):
            loss, norms = step(model, optimizer)
            losses.append(loss)
            gradients.append(norms)
        checkpoint = output / (mode + '_synthetic_only.pt')
        torch.save({'schema': 'm21_synthetic_only_v1', 'config': model.configuration(),
                    'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'updates': 2}, checkpoint)
        saved = torch.load(checkpoint, weights_only=True, map_location=device)
        restored = SharedIncidentState(**saved['config']).to(device)
        restored.load_state_dict(saved['model'])
        resumed = torch.optim.Adam(restored.parameters(), lr=.003)
        resumed.load_state_dict(saved['optimizer'])
        with torch.no_grad():
            prediction = model(**inputs)['state']
            difference = float((restored(**inputs)['state'] - prediction).abs().max())
            require(difference == 0, 'Checkpoint replay mismatch')
            require((model(**inputs, incident_enabled=False)['state'] == 1).all(), 'Off intervention failed after updates')
            require((model(**inputs)['state'] == prediction).all(), 'On restoration failed after updates')
        step(model, optimizer)
        step(restored, resumed)
        resume_difference = max(float((v - restored.state_dict()[k]).abs().max()) for k, v in model.state_dict().items())
        require(resume_difference == 0, 'Optimizer continuation mismatch')
        changed = [k for k, v in model.state_dict().items() if not torch.equal(v, initial[k])]
        require(len(changed) == len(initial), 'Some shared parameters never updated')
        results[mode] = {'parameters': sum(p.numel() for p in model.parameters()), 'saved_optimizer_updates': 2,
                         'continued_updates_checked': 1, 'synthetic_losses_NOT_prediction_metrics': losses,
                         'gradient_l1': gradients, 'changed_parameter_tensors': len(changed),
                         'checkpoint_replay_abs_max': difference, 'continuation_parameter_abs_max': resume_difference}
    require(results['capacity']['parameters'] == results['ordinary']['parameters'], 'Parameter budget mismatch')
    return {'status': 'PASS', 'same_inputs_and_initial_weights': True, 'cases': rows, 'arms': results}


def candidate_check(directory, device, batch_size):
    adapter = CapacityHistoryInputs(directory)
    try:
        models = paired_models(device)  # fresh, NOT the synthetically updated models
        for model in models.values():
            model.eval()
        before = {mode: copy.deepcopy(model.state_dict()) for mode, model in models.items()}
        n = len(adapter.pack.station_ids)
        total = len(adapter.pack.events)
        support_seen, history_seen = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
        minima, maxima = {m: float('inf') for m in models}, {m: -float('inf') for m in models}
        start = time.perf_counter()
        with torch.no_grad():
            for first in range(0, total, batch_size):
                end = min(first + batch_size, total)
                inputs = adapter.batch(np.arange(first, end), np.arange(0., 65., 5.), device)
                for mode, model in models.items():
                    on = model(**inputs)
                    off = model(**inputs, incident_enabled=False)
                    state = on['state']
                    require(state.shape == (end - first, 13, n) and torch.isfinite(state).all(), 'Invalid full-network output')
                    require((off['state'] == 1).all(), 'Nonneutral disabled state')
                    require(torch.equal(on['history_state'], off['history_state']), 'Report switch changed traffic encoder')
                    unsupported = (~on['effective_support'])[:, None].expand_as(state)
                    require((state[unsupported] == 1).all(), 'Nonneutral unsupported state')
                    if mode == 'capacity':
                        require((state >= .05 - 1e-6).all() and (state <= 1).all(), 'Capacity bound violated')
                        require((torch.diff(state, dim=1) >= -1e-6).all(), 'Monotone reference failed')
                        support_seen |= on['report_support'].any(0).cpu().numpy()
                        history_seen |= on['history_available'].any(0).cpu().numpy()
                    minima[mode] = min(minima[mode], float(state.min()))
                    maxima[mode] = max(maxima[mode], float(state.max()))
                if first == 0 or end == total or end % 1024 == 0:
                    print(f'Candidate X interface: {end}/{total} events, {n} nodes, no real-data updates', flush=True)
        for mode, model in models.items():
            require(all(torch.equal(v, before[mode][k]) for k, v in model.state_dict().items()), 'Real-data check modified weights')
            require(all(p.grad is None for p in model.parameters()), 'Real-data check created gradients')
        return {'status': 'ALL_CANDIDATE_X_INTERFACE_PASS', 'candidate_summary_sha256': adapter.summary_sha256,
                'events': total, 'stations': n, 'roads': sorted({r['road'] for r in adapter.pack.events}),
                'station_ids': adapter.pack.station_ids.tolist(), 'nodes_ever_history_available': int(history_seen.sum()),
                'nodes_ever_report_supported': int(support_seen.sum()), 'real_optimizer_updates': 0,
                'weights_unchanged': True, 'fresh_initialization_not_synthetic_checkpoint': True,
                'state_min_UNTRAINED_NOT_mechanism_estimate': minima, 'state_max_UNTRAINED_NOT_mechanism_estimate': maxima,
                'elapsed_minutes_from_cutoff': list(range(0, 65, 5)), 'seconds': time.perf_counter() - start}
    finally:
        adapter.close()


def run_check(output, device_name='cpu', candidate_dir=None, batch_size=32):
    require(batch_size > 0, 'batch_size must be positive')
    require(not output.exists(), 'Use a fresh output directory')
    device = torch.device('cuda:0' if device_name == 'auto' and torch.cuda.is_available() else
                          'cpu' if device_name == 'auto' else device_name)
    require(device.type in ('cpu', 'cuda'), 'Expected CPU or CUDA')
    torch.set_num_threads(3)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    output.mkdir(parents=True)
    synthetic = synthetic_check(output, device)
    real = candidate_check(candidate_dir, device, batch_size) if candidate_dir else {'status': 'NOT_REQUESTED'}
    result = {'schema': 'incident_relative_capacity_m21_v1', 'status': 'M21_ENGINEERING_CHECK_PASS',
              'scientific_status': 'NO_PREDICTIVE_GAIN_OR_PHYSICAL_IDENTIFICATION_CLAIM',
              'synthetic': synthetic, 'candidate': real, 'full_training_started': False,
              'new_Y_or_validation_test_used': False, 'real_capacity_fitted': False,
              'igstgnn_fusion_implemented': False, 'spatial_propagation_implemented': False,
              'environment': {'python': platform.python_version(), 'torch': str(torch.__version__),
                              'device': str(device), 'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None},
              'source_sha256': {name: sha256(REPO / name) for name in SOURCE_FILES},
              'outputs_sha256': {p.name: sha256(p) for p in output.iterdir()}}
    write_json(output / 'summary.json', result)
    print(f"M21_ENGINEERING_CHECK_PASS; candidate={real['status']}; summary={output / 'summary.json'}", flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--candidate-dir', type=Path)
    p.add_argument('--device', default='cpu')
    p.add_argument('--batch-size', type=int, default=32)
    a = p.parse_args()
    run_check(a.output_dir, a.device, a.candidate_dir, a.batch_size)


if __name__ == '__main__':
    main()
