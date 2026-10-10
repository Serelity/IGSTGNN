"""M3.0 engineering evidence, synthetic graphs only; never opens traffic Y."""
import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

import torch
from torch import nn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.models.incident_capacity_exchange import (
    ExchangeGraph, CapacityLimitedExchange, OrdinaryDirectedRecurrence,
    condition_coefficients, edge_capacities, target_window_features)
from src.models.incident_relative_capacity import require

SOURCES = ('src/models/incident_capacity_exchange.py', 'tests/test_incident_capacity_exchange.py',
           'experiments/chronological/check_incident_capacity_exchange.py',
           'experiments/chronological/run_incident_capacity_exchange.sh',
           'experiments/chronological/incident_capacity_exchange_v21.json', 'src/models/igstgnn.py')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')


def fixture(device, nodes=4, channels=4, substeps=4):
    edge_index = torch.tensor([[-1, 0]] + [[i, i+1] for i in range(nodes-1)] + [[nodes-1, -1]],
                              dtype=torch.long, device=device).T.contiguous()
    eligible = torch.ones(nodes, dtype=torch.bool, device=device)
    boundary_nodes = torch.zeros_like(eligible)
    boundary_nodes[[0, nodes-1]] = True
    graph = ExchangeGraph(nodes, edge_index, eligible, boundary_nodes, evidence_scope='synthetic_only')
    rate = torch.ones(nodes, channels, device=device)
    weights = torch.ones(nodes+1, device=device)
    weights[0] = 0
    times = torch.arange(13*substeps+1, device=device, dtype=rate.dtype)*(5/substeps)
    initial = torch.full((1, nodes, channels), .5, device=device)
    boundary = torch.zeros(1, times.numel()-1, nodes+1, channels, device=device)
    boundary[:, :, 0] = .5
    coefficients = torch.zeros(1, nodes+1, 4, device=device)
    return graph, rate, weights, times, initial, boundary, coefficients


def scenario_check(output, device):
    graph, rate, weights, times, initial, boundary, coefficients = fixture(device)
    model = CapacityLimitedExchange(graph, rate, rate)
    capacities = edge_capacities(coefficients, times[:-1], 65., graph, rate, weights)
    base = model(initial, capacities, boundary, times, weights)
    reduced = capacities.clone()
    reduced[:, :16, 2] = .05
    affected = model(initial, reduced, boundary, times, weights)
    require(affected['states'][0, 12, 0, 0] > base['states'][0, 12, 0, 0], 'Missing upstream response')
    low = torch.full_like(initial, .01)
    no_inlet = torch.zeros_like(boundary)
    low_base = model(low, capacities, no_inlet, times, weights)
    low_reduced = capacities.clone()
    low_reduced[:, :, 2] = .5
    low_case = model(low, low_reduced, no_inlet, times, weights)
    require(torch.equal(low_base['states'], low_case['states']), 'Low-demand neutrality failed')
    rows = []
    for k in range(times.numel()):
        rows.append(dict(elapsed_minutes=float(times[k]), baseline_upstream=float(base['states'][0, k, 0, 0]),
                         bottleneck_upstream=float(affected['states'][0, k, 0, 0]),
                         bottleneck_source=float(affected['states'][0, k, 1, 0])))
    with (output/'synthetic_responses.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    # Same fixed continuous parameters, different numerical resolution.
    comparison = []
    for substeps in (4, 8):
        g, r, w, ts, x, b, c = fixture(device, substeps=substeps)
        c[:, 2] = torch.tensor([.8, .8, .1, .1], device=device)
        result = CapacityLimitedExchange(g, r, r)(x, edge_capacities(c, ts[:-1], 65., g, r, w), b, ts, w)
        comparison.append(result['states'][:, ::substeps])
    sensitivity = float((comparison[0]-comparison[1]).abs().max())
    # A large difference blocks this default numerical configuration.
    require(sensitivity <= .05, 'Default substeps have excessive synthetic step dependence')
    return dict(status='PASS', low_demand_exactly_unchanged=True,
                upstream_difference_at_15_minutes=float(affected['states'][0, 12, 0, 0]-base['states'][0, 12, 0, 0]),
                maximum_balance_residual=float(affected['balance_residual'].abs().max()),
                four_vs_eight_substep_state_abs_max=sensitivity, sensitivity_tolerance=.05,
                tolerance_scope='synthetic development diagnostic in latent state units, not empirical accuracy')


def gradient_and_replay_check(output, device):
    graph, rate, weights, times, initial, boundary, history_coefficients = fixture(device)
    initial[:, :2] = .8
    initial[:, 2:] = .1
    # This small shared head is a synthetic optimizer fixture, not a trained
    # report/history encoder, and its targets are never traffic observations.
    head = nn.Linear(1, 4).to(device)
    nn.init.normal_(head.weight, std=.005)
    nn.init.constant_(head.bias, .75)
    features = torch.ones(1, graph.edges, 1, device=device)
    association = torch.zeros(1, graph.edges, device=device)
    association[:, 2] = 1
    operator = CapacityLimitedExchange(graph, rate, rate)
    optimizer = torch.optim.Adam(head.parameters(), lr=.001)

    def forward(current, enabled=True):
        report = current(features).clamp(0, .95)
        coefs = condition_coefficients(history_coefficients, report, association, incident_enabled=enabled)
        capacities = edge_capacities(coefs, times[:-1], 65., graph, rate, weights)
        return operator(initial, capacities, boundary, times, weights)

    def update(current, opt):
        opt.zero_grad(set_to_none=True)
        result = forward(current)
        loss = (result['edge_flux'][:, :, 2]-.15).square().mean()
        loss.backward()
        require(all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
                    for p in current.parameters()), 'Synthetic head gradient failed')
        rows = current.weight.grad.abs().sum(-1)
        require((rows > 0).all(), 'A Bernstein coefficient received no flow gradient')
        opt.step()
        require(all(torch.isfinite(p).all() for p in current.parameters()), 'Nonfinite optimizer state')
        return dict(artificial_response_loss_NOT_forecast_metric=float(loss.detach()), gradient_l1=rows.detach().tolist())

    updates = [update(head, optimizer) for _ in range(2)]
    checkpoint = output/'shared_head_synthetic_only.pt'
    torch.save(dict(schema='m30_synthetic_only_v1', model=head.state_dict(), optimizer=optimizer.state_dict()), checkpoint)
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    restored = nn.Linear(1, 4).to(device)
    restored.load_state_dict(saved['model'])
    resumed = torch.optim.Adam(restored.parameters(), lr=.001)
    resumed.load_state_dict(saved['optimizer'])
    with torch.no_grad():
        on = forward(head)
        off = forward(head, False)
        require(torch.equal(on['states'], forward(restored)['states']), 'Checkpoint replay failed')
        require(torch.equal(on['states'], forward(head)['states']), 'On/off restoration failed')
        require(not torch.equal(on['states'], off['states']), 'High-demand condition switch has no response')
    update(head, optimizer)
    update(restored, resumed)
    difference = max(float((p-restored.state_dict()[name]).abs().max()) for name, p in head.state_dict().items())
    require(difference == 0, 'Optimizer continuation replay failed')
    return dict(status='PASS', synthetic_updates_before_checkpoint=2, coefficient_checks=updates,
                checkpoint_state_replay_abs_max=0., continuation_parameter_abs_max=difference,
                condition_off_is_history_capacity_NOT_unit_capacity=True)


def full_axis_check(device):
    graph, rate, weights, times, initial, boundary, coefficients = fixture(device, nodes=496)
    capacities = edge_capacities(coefficients, times[:-1], 65., graph, rate, weights)
    capacities[:, :16, 250] = .1
    windows_start = torch.arange(5., 65., 5., device=device)
    windows = torch.stack((windows_start, windows_start+5), -1)
    operators = dict(capacity_limited=CapacityLimitedExchange(graph, rate, rate),
                     ordinary=OrdinaryDirectedRecurrence(copy.deepcopy(graph), rate, rate).to(device))
    arms = {}
    start = time.perf_counter()
    with torch.no_grad():
        for name, model in operators.items():
            result = model(initial, capacities, boundary, times, weights)
            features = target_window_features(result, windows)
            require(features.shape == (1, 12, 496, 16) and torch.isfinite(features).all(), 'Full-axis feature interface failed')
            require((result['states'] >= -1e-6).all() and (result['states'] <= 1+1e-6).all(), 'State bounds failed')
            arms[name] = dict(learned_operator_parameters=sum(p.numel() for p in model.parameters()),
                              feature_shape=list(features.shape), state_shape=list(result['states'].shape),
                              balance_residual_abs_max=float(result['balance_residual'].abs().max()) if name == 'capacity_limited' else None)
    return dict(status='SYNTHETIC_496_AXIS_PASS', graph_scope=graph.evidence_scope, real_road_mapping_used=False,
                real_joint_qualification_assessed=False, arms=arms, seconds=time.perf_counter()-start,
                full_model_parameter_matching_complete=False,
                budget_note='Operator-only control; shared encoders, observation head and IGSTGNN fusion are M4 work')


def run_check(output, device_name='cpu'):
    require(not output.exists(), 'Use a fresh output directory')
    device = torch.device(device_name)
    require(device.type in ('cpu', 'cuda'), 'Expected CPU or CUDA')
    if device.type == 'cuda':
        require(torch.cuda.is_available(), 'CUDA requested but unavailable')
    torch.set_num_threads(3)
    torch.manual_seed(20261010)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    output.mkdir(parents=True)
    scenarios = scenario_check(output, device)
    gradients = gradient_and_replay_check(output, device)
    full_axis = full_axis_check(device)
    result = dict(schema='incident_capacity_exchange_m30_v1', status='M30_ENGINEERING_CHECK_PASS',
                  scenarios=scenarios, gradients=gradients, full_axis=full_axis,
                  kappa=1., real_traffic_used=False, real_optimizer_updates=0,
                  igstgnn_fusion_implemented=False, full_training_started=False,
                  physical_contract_status='PHYSICAL_CONTRACT_REQUIRED',
                  scientific_status='NO_PREDICTIVE_GAIN_OR_REAL_NETWORK_PHYSICS_CLAIM',
                  environment=dict(python=platform.python_version(), torch=str(torch.__version__),
                                   device=str(device), gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                                   cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES')),
                  source_sha256={path: digest(REPO/path) for path in SOURCES},
                  output_sha256={path.name: digest(path) for path in output.iterdir()})
    write_json(output/'summary.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    run_check(args.output_dir, args.device)
