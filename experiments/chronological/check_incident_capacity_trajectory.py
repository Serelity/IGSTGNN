"""M2.2 bounded trajectory checks; real candidate X is never used for updates."""
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
from experiments.chronological.check_incident_relative_capacity import synthetic_inputs
from src.models.incident_relative_capacity import SharedIncidentState, supply_limited_flow
from src.models.incident_capacity_trajectory import TrajectoryIncidentState, capacity_curve
from src.utils.relative_capacity_inputs import CapacityHistoryInputs
from src.utils.incident_corridor import require, sha256, write_json, write_rows

SOURCES = ('src/models/incident_relative_capacity.py', 'src/models/incident_capacity_trajectory.py',
           'src/utils/relative_capacity_inputs.py', 'src/utils/incident_candidate_history.py',
           'experiments/chronological/check_incident_relative_capacity.py',
           'experiments/chronological/check_incident_capacity_trajectory.py',
           'experiments/chronological/run_incident_capacity_trajectory.sh',
           'tests/test_incident_relative_capacity.py', 'tests/test_incident_capacity_trajectory.py',
           'src/models/igstgnn.py')


def matched_models(device):
    torch.manual_seed(20261010)
    legacy = SharedIncidentState().to(device)
    pulse = TrajectoryIncidentState().to(device)
    pulse.initialize_common_from(legacy)
    models = {'exponential': legacy, 'pulse': pulse}
    for name in ('mixture', 'ordinary'):
        models[name] = TrajectoryIncidentState(trajectory=name).to(device)
        models[name].load_state_dict(pulse.state_dict())
    return models


def synthetic_check(output, device):
    times = torch.tensor([0., 5., 15., 30., 60., 120., 600.], device=device)
    params = [torch.tensor([[x]], device=device) for x in (.2, 30., .6, 30.)]
    curves = {kind: capacity_curve(*params, times, kind)[0, :, 0] for kind in ('pulse', 'mixture')}
    require(curves['pulse'][3] > curves['pulse'][0] and curves['pulse'][-1] < 1e-6, 'Expected worsening and recovery')
    require((torch.diff(curves['mixture']) <= 0).all(), 'Mixture must remain monotone')
    r = 1-curves['pulse']
    base, low, high, supply = [torch.full_like(r, x) for x in (1., .02, 2., .01)]
    low_out = supply_limited_flow(low, base, r, high)
    high_out = supply_limited_flow(high, base, r, high)
    blocked_out = supply_limited_flow(high, base, r, supply)
    require(torch.equal(low_out, low) and torch.equal(high_out, r) and torch.equal(blocked_out, supply), 'Supply behavior failed')
    values = [x.cpu().tolist() for x in (times, r, 1-curves['mixture'], low_out, high_out, blocked_out)]
    rows = [dict(zip(('elapsed_minutes', 'pulse_retention', 'mixture_retention', 'low_demand_flow',
                      'high_demand_flow', 'downstream_limited_flow'), v)) for v in zip(*values)]
    write_rows(output/'synthetic_trajectory_cases.csv', rows)
    inputs, models, result = synthetic_inputs(device), matched_models(device), {}
    for name in ('pulse', 'mixture', 'ordinary'):
        model = models[name]
        opt = torch.optim.Adam(model.parameters(), lr=.003)
        def step(m, optimizer):
            optimizer.zero_grad(set_to_none=True)
            state = m(**inputs)['state']
            loss = (state-.5).square().mean()  # artificial target, never traffic Y
            loss.backward()
            require(all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
                        for p in m.parameters()), 'Gradient connection failed')
            head_rows = m.condition_head[-1].weight.grad.abs().sum(-1)
            require((head_rows > 0).all(), 'A shape parameter has no gradient')
            optimizer.step()
            require(all(torch.isfinite(p).all() for p in m.parameters()), 'Nonfinite update')
            return {'synthetic_loss_NOT_prediction_metric': float(loss.detach()), 'head_gradient_l1': head_rows.tolist()}
        updates = [step(model, opt) for _ in range(2)]
        checkpoint = output/(name+'_synthetic_only.pt')
        torch.save({'schema': 'm22_synthetic_only_v1', 'config': model.configuration(),
                    'model': model.state_dict(), 'optimizer': opt.state_dict()}, checkpoint)
        saved = torch.load(checkpoint, weights_only=True, map_location=device)
        restored = TrajectoryIncidentState(**saved['config']).to(device)
        restored.load_state_dict(saved['model'])
        resumed = torch.optim.Adam(restored.parameters(), lr=.003)
        resumed.load_state_dict(saved['optimizer'])
        with torch.no_grad():
            on = model(**inputs)['state']
            replay = float((on-restored(**inputs)['state']).abs().max())
            require(replay == 0, 'Checkpoint replay failed')
            require((model(**inputs, incident_enabled=False)['state'] == 1).all(), 'Condition-off failed')
            require(torch.equal(on, model(**inputs)['state']), 'Condition-on restoration failed')
        step(model, opt)
        step(restored, resumed)
        continuation = max(float((v-restored.state_dict()[k]).abs().max()) for k, v in model.state_dict().items())
        require(continuation == 0, 'Optimizer replay failed')
        result[name] = dict(parameters=sum(p.numel() for p in model.parameters()), updates=updates,
                            checkpoint_replay_abs_max=replay, continuation_parameter_abs_max=continuation)
    require(len({r['parameters'] for r in result.values()}) == 1, 'Matched parameter budget failed')
    return {'status': 'PASS', 'cases': rows, 'arms': result,
            'legacy_parameters': sum(p.numel() for p in models['exponential'].parameters()),
            'known_nonidentifiability': ['shape time is invisible when strength is zero',
                                       'mixture weight is invisible when the two time scales coincide']}


def candidate_check(directory, device, batch_size):
    adapter = CapacityHistoryInputs(directory)
    try:
        models = matched_models(device)  # discard all synthetic updates
        before = {k: copy.deepcopy(m.state_dict()) for k, m in models.items()}
        for model in models.values():
            model.eval()
        total, n = len(adapter.pack.events), len(adapter.pack.station_ids)
        stats = {k: dict(minimum=float('inf'), maximum=-float('inf'), worsening_supported_curves=0,
                          supported_curves=0, amplitude_saturated=0 if k in ('pulse', 'mixture') else None,
                          strength_saturated=0 if k in ('pulse', 'mixture') else None) for k in models}
        ever = np.zeros(n, dtype=bool)
        start = time.perf_counter()
        with torch.no_grad():
            for first in range(0, total, batch_size):
                end = min(total, first+batch_size)
                inputs = adapter.batch(np.arange(first, end), np.arange(0., 65., 5.), device)
                legacy_start = None
                for name, model in models.items():
                    on, off = model(**inputs), model(**inputs, incident_enabled=False)
                    state, supported = on['state'], on['effective_support']
                    require(state.shape == (end-first, 13, n) and torch.isfinite(state).all(), 'State interface failed')
                    require((off['state'] == 1).all() and torch.equal(on['history_state'], off['history_state']), 'Off/history invariants failed')
                    require((state[(~supported)[:, None].expand_as(state)] == 1).all(), 'Unsupported increment')
                    if name != 'ordinary':
                        require((state >= .05-1e-6).all() and (state <= 1+1e-6).all(), 'Capacity bound failed')
                        if name == 'exponential':
                            legacy_start = state[:, 0]
                        else:
                            require(torch.equal(state[:, 0], legacy_start), 'Common cutoff capacity mismatch')
                        if name != 'pulse':
                            require((torch.diff(state, dim=1) >= -1e-6).all(), 'Monotone reference failed')
                    s = stats[name]
                    s['minimum'], s['maximum'] = min(s['minimum'], float(state.min())), max(s['maximum'], float(state.max()))
                    s['supported_curves'] += int(supported.sum())
                    s['worsening_supported_curves'] += int(((torch.diff(state, dim=1) < -1e-6).any(1) & supported).sum())
                    for field in ('amplitude_saturated', 'strength_saturated'):
                        if field in on:
                            s[field] += int(on[field].sum())
                    ever |= supported.any(0).cpu().numpy()
                if first == 0 or end == total or end % 1024 == 0:
                    print(f'M2.2 candidate X: {end}/{total}, {n} stations, four arms, no updates', flush=True)
        for name, model in models.items():
            require(all(torch.equal(v, before[name][k]) for k, v in model.state_dict().items()), 'Real-data weight change')
            require(all(p.grad is None for p in model.parameters()), 'Unexpected real-data gradients')
        return dict(status='ALL_CANDIDATE_X_INTERFACE_PASS', events=total, stations=n,
                    road_direction_groups=len({e['road'] for e in adapter.pack.events}), nodes_ever_supported=int(ever.sum()),
                    real_optimizer_updates=0, weights_unchanged=True, seconds=time.perf_counter()-start,
                    candidate_summary_sha256=adapter.summary_sha256, initialization_only_NOT_mechanism_estimates=stats)
    finally:
        adapter.close()


def run_check(output, device_name='cpu', candidate_dir=None, batch_size=32):
    require(batch_size > 0 and not output.exists(), 'Use positive batch size and fresh output directory')
    device = torch.device(device_name)
    require(device.type in ('cpu', 'cuda'), 'Expected CPU or CUDA')
    torch.set_num_threads(3)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    output.mkdir(parents=True)
    synthetic = synthetic_check(output, device)
    candidate = candidate_check(candidate_dir, device, batch_size) if candidate_dir else {'status': 'NOT_REQUESTED'}
    result = dict(schema='incident_capacity_trajectory_m22_v1', status='M22_ENGINEERING_CHECK_PASS',
                  synthetic=synthetic, candidate=candidate, full_training_started=False,
                  new_Y_or_validation_test_used=False, spatial_propagation_implemented=False,
                  scientific_status='NO_PREDICTIVE_GAIN_OR_IDENTIFIED_CAPACITY_CLAIM',
                  environment=dict(python=platform.python_version(), torch=str(torch.__version__), device=str(device),
                                   gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else None),
                  source_sha256={p: sha256(REPO/p) for p in SOURCES},
                  outputs_sha256={p.name: sha256(p) for p in output.iterdir()})
    write_json(output/'summary.json', result)
    print(f"M22_ENGINEERING_CHECK_PASS; candidate={candidate['status']}; output={output}", flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--candidate-dir', type=Path)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()
    run_check(args.output_dir, args.device, args.candidate_dir, args.batch_size)
