"""Check the review's local capacity-path examples against the unchanged B3 operator.

Synthetic CPU calculations only; no B3-R model, real data, checkpoints or training.
"""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from src.models.incident_capacity_exchange import ExchangeGraph
from src.models.incident_spatial import exchange_step


def graph(nodes, pairs):
    return ExchangeGraph(
        nodes, torch.tensor(pairs, dtype=torch.long).T,
        torch.ones(nodes, dtype=torch.bool), torch.zeros(nodes, dtype=torch.bool),
        evidence_scope='synthetic_only')


def number(value):
    return float(value.detach())


def single(capacity, source, destination, proposal=.9, conditioned=False):
    c = torch.tensor(capacity, dtype=torch.float64, requires_grad=True)
    x = torch.tensor([[[source], [destination]]], dtype=torch.float64)
    q = torch.sigmoid(1-2*c) if conditioned else c.new_tensor(proposal)
    result = exchange_step(x, q.reshape(1, 1, 1), c.reshape(1, 1, 1),
                           graph(2, ((0, 1),)), restricted=True)
    f = result['flux'].reshape(())
    derivative = torch.autograd.grad(f, c)[0]
    assert result['balance'].abs().max() < 1e-12
    assert ((result['state'] >= 0) & (result['state'] <= 1)).all()
    return dict(capacity=capacity, proposal=number(q), flux=number(f),
                total_flux_derivative_wrt_capacity=number(derivative),
                capacity_cap_triggered=number(q) > capacity)


def main():
    torch.set_num_threads(1)
    endpoints = torch.tensor([.3, .8], dtype=torch.float64, requires_grad=True)
    edge = endpoints.min()
    endpoint_gradient = torch.autograd.grad(edge, endpoints)[0]
    endpoint_case = dict(before=[.3, .8], after=[.3, .6],
                         edge_before=number(edge), edge_after=min(.3, .6),
                         gradient_before=endpoint_gradient.tolist())
    assert endpoint_case['edge_before'] == endpoint_case['edge_after']
    assert endpoint_case['gradient_before'] == [1., 0.]

    sending = [single(c, .1, .5) for c in (.4, .2)]
    receiving = [single(c, .8, .9) for c in (.4, .2)]
    for rows in (sending, receiving):
        assert abs(rows[0]['flux']-rows[1]['flux']) < 1e-12
        assert all(row['capacity_cap_triggered'] for row in rows)
        assert all(abs(row['total_flux_derivative_wrt_capacity']) < 1e-12 for row in rows)

    positive = single(.2, .8, .1, proposal=.7)
    assert abs(positive['flux']-.2) < 1e-12
    assert abs(positive['total_flux_derivative_wrt_capacity']-1) < 1e-12
    epsilon = 1e-6
    finite_difference = (single(.2+epsilon, .8, .1, proposal=.7)['flux']
                         - single(.2-epsilon, .8, .1, proposal=.7)['flux'])/(2*epsilon)
    assert abs(finite_difference-1) < 1e-9
    positive['finite_difference_derivative'] = finite_difference

    fork = graph(3, ((0, 1), (0, 2)))
    fork_rows = []
    for value in (1., .5):
        factor = torch.tensor(value, dtype=torch.float64, requires_grad=True)
        capacities = factor * torch.tensor([[[.4], [.2]]], dtype=torch.float64)
        result = exchange_step(torch.tensor([[[.1], [.1], [.1]]], dtype=torch.float64),
                               torch.full_like(capacities, .9), capacities, fork,
                               restricted=True)
        derivative = torch.autograd.grad(result['flux'][0, 0, 0], factor)[0]
        fork_rows.append(dict(common_capacity_factor=value,
                              flux=result['flux'].detach().flatten().tolist(),
                              first_edge_derivative_wrt_factor=number(derivative)))
    assert max(abs(a-b) for a, b in zip(fork_rows[0]['flux'], fork_rows[1]['flux'])) < 1e-12
    assert all(abs(row['first_edge_derivative_wrt_factor']) < 1e-12 for row in fork_rows)

    nonmonotone = [single(c, .95, .05, conditioned=True) for c in (.8, .7)]
    assert nonmonotone[1]['flux'] > nonmonotone[0]['flux']
    assert all(row['flux'] < row['capacity'] for row in nonmonotone)
    assert all(row['total_flux_derivative_wrt_capacity'] < 0 for row in nonmonotone)

    output = dict(
        schema='b3_r_capacity_path_examples_v1', date='2026-10-12',
        status='LOCAL_SYNTHETIC_CAPACITY_PATH_EXAMPLES_PASS',
        scope='local examples on the existing B3 exchange_step, not integrated B3-R validation',
        device='cpu', dtype='float64', torch_version=str(torch.__version__),
        sources_sha256={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                        for name in ('src/models/incident_spatial.py',
                                     'src/models/incident_capacity_exchange.py')},
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        cases=dict(endpoint_minimum_masks_node_change=endpoint_case,
                   sending_budget_cancels_capacity_change=sending,
                   receiving_budget_cancels_capacity_change=receiving,
                   capacity_sensitive_positive=positive,
                   shared_sending_cancels_common_factor=fork_rows,
                   conditioned_proposal_can_reverse_local_monotonicity=dict(
                       constructed_proposal='sigmoid(1-2*C), not a trained network', rows=nonmonotone)),
        full_main_forecast_loss_checked=False, b3_r_model_implemented=False,
        real_data_or_checkpoints_read=False, server_run_changed=False)
    target = Path(__file__).with_name('igstgnn_b3_r_capacity_path_examples_20261012.json')
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False)+'\n',
                      encoding='utf-8')
    print(json.dumps(dict(status=output['status'], cases=len(output['cases']),
                         output=str(target)), ensure_ascii=False))


if __name__ == '__main__':
    main()
