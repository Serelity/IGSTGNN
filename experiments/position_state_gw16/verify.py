"""Server-only real-batch checks; never produces research accuracy claims."""
import argparse
import logging
from pathlib import Path

import torch

from . import ARMS
from .common import load_config, protected_output, require, runtime, tensor_tree_hash, write_json
from .data import device_batch, load_data
from .model import build_model
from .objective import loss_terms
from .train import seed_all


def module_gradient(module):
    terms = [p.grad.detach().double().square().sum() for p in module.parameters() if p.grad is not None]
    return float(torch.stack(terms).sum().sqrt()) if terms else 0.


def check_arm(data, config, arm, device, seed):
    seed_all(seed)
    model = build_model(data, config, arm, device)
    raw = next(data.loaders['train_loader'].get_iterator())
    x, target, event = device_batch(raw, data, device)
    model.set_stage('joint')
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    gradients = {}
    for _ in range(3):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        result = model(x, event)
        require(all(bool(torch.isfinite(result[k]).all()) for k in
            ('prediction', 'reference', 'regional', 'alpha', 'beta', 'state_rms')), 'Nonfinite output: '+arm)
        loss = loss_terms(result, target, model.geometry, config)['loss']
        require(bool(torch.isfinite(loss)), 'Nonfinite engineering loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['clip_grad_norm'], error_if_nonfinite=True)
        gradients = {name:module_gradient(getattr(model, name)) for name in
            ('position', 'encoder', 'incident', 'origin_condition') if hasattr(model, name)}
        if arm in ('ORIGIN_EXCHANGE', 'STATE_EXCHANGE'):
            for name, module in (('exchange_position', model.head.exchange.position),
                                 ('exchange_dynamic', model.head.exchange.dynamic),
                                 ('pool_map', model.head.pool_map),
                                 ('lift_map', model.head.lift_map)):
                gradients[name] = module_gradient(module)
        require(any(p.grad is not None and bool(p.grad.abs().sum() > 0) for p in model.parameters()),
                'Model receives no gradient: '+arm)
        optimizer.step()
    if arm in ('ORIGIN_EXCHANGE', 'STATE_EXCHANGE'):
        require(all(gradients[name] > 0 for name in
            ('exchange_position', 'exchange_dynamic', 'pool_map', 'lift_map')),
            'The proposed exchange path receives no real-batch training signal: '+arm)
    model.eval()
    with torch.no_grad():
        output = model(x, event)
        blank = dict(event, report_present=torch.zeros_like(event['report_present']))
        absent = model(x, blank)
        require(bool(torch.isfinite(absent['prediction']).all()), 'No-report forward failed')
        g = model.geometry
        coherence = float((g.pool(output['prediction'])-output['regional']).abs().max())
        center = float(g.pool(output['beta']).abs().max())
        require(torch.allclose(g.pool(output['prediction']), output['regional'], atol=2e-3, rtol=1e-5),
                'Region decoding contract failed: '+arm)
        require(center < 1e-4, 'Local output is not region-centered: '+arm)
        permutation = torch.arange(len(x)-1, -1, -1, device=x.device)
        reordered = model(x[permutation], {k:v[permutation] for k,v in event.items()})
        require(torch.allclose(reordered['prediction'], output['prediction'][permutation], atol=2e-3, rtol=1e-5),
                'Batch order changes prediction: '+arm)
    capacity = getattr(model, 'capacity', None)
    return dict(passed=True, engineering_optimizer_updates=3,
        parameters=sum(p.numel() for p in model.parameters()), capacity=capacity,
        last_gradient_norms=gradients, coherence_max_abs=coherence,
        local_center_max_abs=center, no_report_forward_finite=True,
        checkpoint_written=False, research_accuracy_result=False)


def paired_initial_check(data, config, device, seed):
    seed_all(seed)
    origin = build_model(data, config, 'ORIGIN_EXCHANGE', device).eval()
    seed_all(seed)
    state = build_model(data, config, 'STATE_EXCHANGE', device).eval()
    require(tensor_tree_hash(origin.state_dict()) == tensor_tree_hash(state.state_dict()),
            'ORIGIN/STATE initial tensors differ')
    x, _, event = device_batch(next(data.loaders['train_loader'].get_iterator()), data, device)
    with torch.no_grad():
        a, b = origin(x, event), state(x, event)
    require(torch.allclose(a['prediction'][:,0], b['prediction'][:,0], atol=1e-4, rtol=1e-6),
            'ORIGIN/STATE first step differs before states diverge')
    return dict(initial_tensors_identical=True, first_step_equal_within_float32_tolerance=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--threads', type=int, default=3)
    parser.add_argument('--seed', type=int, default=2025)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    output = protected_output(args.output_dir, args.data_dir)
    require(not output.exists(), 'Verification needs a fresh output directory')
    environment = runtime(args.device, args.threads, args.seed)
    data = load_data(args.data_dir, config, logging.getLogger('gw16-verify'), engineering=True)
    paired = paired_initial_check(data, config, args.device, args.seed)
    records = {arm:check_arm(data, config, arm, args.device, args.seed) for arm in ARMS}
    output.mkdir(parents=True)
    report = dict(schema='position_state_gw16_engineering_v1', passed=True,
        arms=records, paired_initialization=paired, runtime=environment, data=data.identity,
        engineering_only=True, test_evaluated=False, research_accuracy_results=False)
    write_json(output/'verification.json', report)
    print(output/'verification.json')
    return report


if __name__ == '__main__':
    main()
