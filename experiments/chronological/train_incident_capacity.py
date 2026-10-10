"""M4.2 paired six-arm exploratory training on the unchanged development task."""
import argparse
import copy
import hashlib
import json
import os
import platform
from pathlib import Path
import sys
import time

import numpy as np
import torch

REPO = Path(__file__).absolute().parents[2]
sys.path.insert(0, str(REPO))
from experiments.chronological.smoke import make_model, set_seed
from experiments.chronological.train import MetricTotals, epoch_plan, save_checkpoint
from experiments.chronological.prepare_incident_capacity_network import prepare
from src.models.incident_capacity_fusion import CapacityAugmentedIGSTGNN, IncidentCapacityBranch
from src.utils.capacity_fusion_inputs import OriginalCapacityInputs
from src.utils.capacity_training_inputs import CapacityDevelopmentInputs
from src.utils.chronological import masked_flow_mae
from src.utils.incident_corridor import read_json, require, sha256, write_json

ARMS = ('F', 'N0', 'N1', 'P0', 'P1', 'L1')
MODES = dict(N0='ordinary', N1='ordinary', P0='capacity', P1='capacity', L1='local')
SOURCE_FILES = (
    'experiments/chronological/train_incident_capacity.py', 'experiments/chronological/train.py',
    'experiments/chronological/smoke.py', 'experiments/chronological/prepare_incident_capacity_network.py',
    'experiments/chronological/prepare_incident_corridors.py', 'experiments/chronological/prepare_context.py',
    'experiments/chronological/audit_incident_expansion.py',
    'src/models/igstgnn.py', 'src/models/incident_response.py',
    'src/models/incident_capacity_exchange.py', 'src/models/incident_capacity_fusion.py',
    'src/models/incident_relative_capacity.py', 'src/utils/capacity_network_inputs.py',
    'src/utils/capacity_fusion_inputs.py', 'src/utils/capacity_training_inputs.py',
    'src/utils/chronological.py', 'src/utils/incident_corridor.py',
)


def state_digest(state):
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        h.update(key.encode())
        h.update(str((value.dtype, tuple(value.shape))).encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def build_arm(backbone, branch_state, graph, weights, arm, seed):
    require(arm in ARMS, 'Unknown arm')
    native = copy.deepcopy(backbone)
    if arm == 'F':
        return native
    set_seed(seed+4200)
    branch = IncidentCapacityBranch(copy.deepcopy(graph), weights, mode=MODES[arm],
                                    forecast_dim=backbone._forecast_dim).to(weights.device)
    missing, unexpected = branch.load_state_dict(branch_state, strict=False)
    require(not unexpected and all(k.startswith(('operator.message.', 'operator.transition.', 'operator.gate.'))
                                   for k in missing), 'Unexpected common branch mismatch')
    for key, value in branch_state.items():
        require(torch.equal(branch.state_dict()[key], value), 'Common branch initialization changed: '+key)
    return CapacityAugmentedIGSTGNN(native, branch)


def forward_arm(model, batch, arm, scaler, *, details=False):
    if arm == 'F':
        prediction = model(batch['x'], incident_data=batch['incident'])
        result = dict(prediction=prediction)
    else:
        result = model(batch['x'], incident_data=batch['incident'], capacity_inputs=batch['capacity_inputs'],
                       incident_enabled=arm.endswith('1'), return_details=True)
    result['prediction'] = result['prediction']*scaler['std']+scaler['mean']
    return result if details else result['prediction']


def masked_auxiliary(prediction, target, valid, operator_mask, scaler):
    valid = valid & operator_mask[None, None, :, None]
    if not valid.any():
        return prediction.sum()*0
    return masked_flow_mae(prediction, (target-scaler['mean'])/scaler['std'], valid)


def capacity_loss(model, batch, target, valid, prefix_batch, prefix_target, prefix_valid,
                  arm, scaler, protocol):
    result = forward_arm(model, batch, arm, scaler, details=True)
    main = masked_flow_mae(result['prediction'], target, valid)
    losses = dict(main=main)
    if arm == 'F':
        return main, losses, result
    branch = model.branch
    auxiliary = masked_auxiliary(result['auxiliary_prediction'], target, valid, branch.graph.operator_mask, scaler)
    prepared = branch.prepare(**prefix_batch['capacity_inputs'], incident_enabled=arm.endswith('1'))
    windows = target.new_tensor([[5., 10.], [10., 15.]])
    prefix = branch.rollout_prepared(prepared, windows)
    prefix_loss = masked_auxiliary(prefix['auxiliary_prediction'], prefix_target, prefix_valid,
                                   branch.graph.operator_mask, scaler)
    # Smooth the used capacity trajectory, including the earlier-prefix task.
    def curvature(coefficients):
        if not coefficients.numel():
            return coefficients.sum()*0
        return torch.diff(coefficients, n=2, dim=-1).square().mean()
    curve = (curvature(result['coefficients'])+curvature(prefix['coefficients']))/2
    losses.update(auxiliary=auxiliary, prefix=prefix_loss, curve=curve)
    # Main loss remains the original raw-unit pooled MAE. Multiplying normalized
    # auxiliary terms by the frozen train std gives the specified relative scale.
    total = main + scaler['std']*(protocol['lambda_obs']*(auxiliary+prefix_loss)/2
                                + protocol['lambda_curve']*curve)
    return total, losses, result


def train_update(model, optimizer, batch, target, valid, prefix_batch, prefix_target, prefix_valid,
                 arm, scaler, protocol, *, diagnose=False):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss, terms, result = capacity_loss(model, batch, target, valid, prefix_batch, prefix_target,
                                       prefix_valid, arm, scaler, protocol)
    require(torch.isfinite(loss), 'Nonfinite training loss')
    main_gradients = {}
    if diagnose and arm != 'F':
        selected = {n: p for n, p in model.named_parameters() if n.startswith('branch.coefficients.')}
        main_grads = torch.autograd.grad(terms['main'], tuple(selected.values()), retain_graph=True, allow_unused=True)
        main_gradients = {'main_loss:'+name: (0. if g is None else float(g.detach().abs().sum()))
                         for name, g in zip(selected, main_grads)}
    loss.backward()
    grads = {name: p.grad for name, p in model.named_parameters() if p.grad is not None}
    require(grads and all(torch.isfinite(g).all() for g in grads.values()), 'Nonfinite/absent gradients')
    gradients = {name: float(g.detach().abs().sum()) for name, g in grads.items() if name.startswith('branch.')}
    gradients.update(main_gradients)
    torch.nn.utils.clip_grad_norm_(model.parameters(), protocol['clip_grad_norm'])
    optimizer.step()
    require(all(torch.isfinite(p).all() for p in model.parameters()), 'Nonfinite model parameter')
    diagnostic = {}
    if arm != 'F':
        states = result['rollout']['states'].detach()
        diagnostic.update(state_min=float(states.min()), state_max=float(states.max()),
                          forecast_delta_abs_max=float(result['forecast_delta'].detach().abs().max()))
        require(diagnostic['state_min'] >= -1e-5 and diagnostic['state_max'] <= 1+1e-5, 'Latent state range violated')
        for key in ('capacity_limited', 'receiving_limited', 'sending_bid_limited'):
            flags = result['rollout'].get(key)
            if flags is not None:
                diagnostic[key+'_fraction'] = float(flags.float().mean()) if flags.numel() else None
        residual = result['rollout'].get('balance_residual')
        if residual is not None:
            diagnostic['balance_residual_abs_max'] = float(residual.detach().abs().max())
    return result['prediction'].detach(), {k: float(v.detach()) for k, v in terms.items()}, gradients, diagnostic


def evaluate(model, inputs, arm, indices, batch_size, device, scaler):
    model.eval()
    regions = dict(all_nodes=np.ones(len(inputs.structure_mask), bool), common_structure=inputs.common_mask,
                   added_structure=inputs.structure_mask & ~inputs.common_mask,
                   candidate_structure=inputs.structure_mask,
                   candidate_boundary=np.asarray(inputs.network.structure['boundary_nodes'], bool),
                   **{'road_'+key: mask for key, mask in inputs.road_groups.items()})
    metrics = {key: MetricTotals() for key, mask in regions.items() if mask.any()}
    predictions, targets, validity = [], [], []
    for start in range(0, len(indices), batch_size):
        selected = indices[start:start+batch_size]
        batch = inputs.batch('val', selected, device, new_reports=arm.endswith('1'))
        target, valid = inputs.targets('val', selected, device)
        with torch.inference_mode():
            prediction = forward_arm(model, batch, arm, scaler)
        for key, total in metrics.items():
            mask = torch.tensor(regions[key], device=device)[None, None, :, None]
            total.update(prediction, target, valid & mask)
        predictions.append(prediction.cpu().numpy())
        targets.append(target.cpu().numpy())
        validity.append(valid.cpu().numpy())
    result = {key: (metrics[key].result() if key in metrics and (metrics[key].counts > 0).all() else None)
              for key in regions}
    arrays = dict(prediction=np.concatenate(predictions), target=np.concatenate(targets), valid=np.concatenate(validity),
                  sample_indices=np.array([int(inputs.events['val'][i]['sample_index']) for i in indices]))
    difference = np.abs(arrays['prediction'].astype(np.float64)-arrays['target'])
    for key, mask in regions.items():
        if result[key] is not None:
            available = arrays['valid'] & mask[None, None, :, None]
            count = available.sum(0)
            error = np.where(available, difference, 0).sum(0)
            result[key]['station_horizon_macro_mae'] = float((error[count > 0]/count[count > 0]).mean())
    return result, arrays


def train_arm(model, inputs, arm, root, identity, protocol, seed, epochs, batch_size, device, check, resume):
    directory = root/arm
    directory.mkdir(exist_ok=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=protocol['learning_rate'],
                                 weight_decay=protocol['weight_decay'], eps=protocol['adam_eps'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, protocol['lr_milestones'], protocol['lr_gamma'])
    checkpoint = directory/'last_checkpoint.pt'
    history, completed, updates, best, best_epoch, wait = [], 0, 0, float('inf'), 0, 0
    if resume and checkpoint.is_file():
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        require(saved['identity'] == identity, 'Resume identity changed; preserve this run and start a new one')
        model.load_state_dict(saved['model_state'])
        optimizer.load_state_dict(saved['optimizer_state'])
        scheduler.load_state_dict(saved['scheduler_state'])
        history, completed, updates, best, best_epoch, wait = [saved[k] for k in
            ('history', 'completed_epoch', 'global_updates', 'best_metric', 'best_epoch', 'wait')]
    elif checkpoint.exists():
        raise ValueError('Checkpoint exists; use --resume explicitly')
    selected_train = list(range(min(4, len(inputs.events['train'])) if check else len(inputs.events['train'])))
    selected_val = list(range(min(4, len(inputs.events['val'])) if check else len(inputs.events['val'])))
    for epoch in range(completed+1, min(epochs, protocol['max_epochs'])+1):
        if wait >= protocol['patience']:
            break
        plan = epoch_plan(selected_train, batch_size, seed, epoch)
        began = time.perf_counter()
        total, losses, gradients, diagnostics = MetricTotals(), [], {}, []
        for step, indices in enumerate(plan['batches']):
            set_seed(seed*1_000_003+epoch*10_000+step)
            batch = inputs.batch('train', indices, device, new_reports=arm.endswith('1'))
            target, valid = inputs.targets('train', indices, device)
            prefix_batch = prefix_target = prefix_valid = None
            if arm != 'F':
                prefix_batch = inputs.batch('train', indices, device, new_reports=arm.endswith('1'), prefix=True)
                prefix_target, prefix_valid = inputs.targets('train', indices, device, prefix=True)
            prediction, terms, grads, diag = train_update(model, optimizer, batch, target, valid,
                prefix_batch, prefix_target, prefix_valid, arm, inputs.original.scaler, protocol, diagnose=step < 2)
            total.update(prediction, target, valid)
            losses.append(terms)
            diagnostics.append(diag)
            for name, value in grads.items():
                gradients[name] = max(gradients.get(name, 0.), value)
            updates += 1
        metrics, arrays = evaluate(model, inputs, arm, selected_val, batch_size, device, inputs.original.scaler)
        score = metrics['all_nodes']['mae_macro']
        improved = score < best
        if improved:
            best, best_epoch, wait = score, epoch, 0
            save_checkpoint(directory/'best_model.pt', dict(model_state=model.state_dict(), identity=identity, epoch=epoch))
            temporary = directory/'best_validation_predictions.partial.npz'
            np.savez_compressed(temporary, **arrays)
            temporary.replace(directory/'best_validation_predictions.npz')
            write_json(directory/'best_validation_metrics.json', metrics)
        else:
            wait += 1
        scheduler.step()
        completed = epoch
        row = dict(epoch=epoch, global_updates=updates, train_order_sha256=plan['order_sha256'],
                   train_mae_macro=total.result()['mae_macro'], validation_mae_macro=score,
                   best_metric=best, seconds=time.perf_counter()-began,
                   losses={k: float(np.mean([r[k] for r in losses])) for k in losses[0]},
                   branch_gradient_max_l1=gradients, diagnostics=diagnostics,
                   validation_metrics=metrics)
        history.append(row)
        save_checkpoint(checkpoint, dict(identity=identity, model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict(), scheduler_state=scheduler.state_dict(), history=history,
            completed_epoch=completed, global_updates=updates, best_metric=best, best_epoch=best_epoch, wait=wait))
        print(json.dumps(dict(arm=arm, **{k: row[k] for k in ('epoch', 'train_mae_macro', 'validation_mae_macro', 'seconds')})), flush=True)
    require(completed > 0, 'No completed epoch')
    terminal = completed == protocol['max_epochs'] or wait >= protocol['patience']
    summary = dict(arm=arm, identity=identity, status='EXPLORATORY_TRAINING_COMPLETE' if terminal else 'PAUSED_AT_EPOCH_BOUNDARY',
        scientific_status='EXPLORATORY_CHECK_SUBSET_NO_GAIN_CLAIM' if check else 'SINGLE_SEED_EXPLORATORY_REUSED_VALIDATION',
        main_training_ready=False, online_semantics_certified=False, jointly_certified_nodes=0,
        completed_epoch=completed, global_updates=updates, best_epoch=best_epoch, best_metric=best,
        parameters=sum(p.numel() for p in model.parameters()),
        branch_parameters=0 if arm == 'F' else sum(p.numel() for p in model.branch.parameters()),
        best_validation_metrics=read_json(directory/'best_validation_metrics.json'), history=history,
        prediction_sha256=sha256(directory/'best_validation_predictions.npz'), test_accessed=False)
    write_json(directory/'summary.json', summary)
    return summary


def pair_report(summaries):
    pairs = [('P1', 'N1'), ('P1', 'F'), ('P1', 'P0'), ('P1', 'L1'), ('N1', 'N0')]
    result = dict(status='M42_PAIRED_EXPLORATORY_RUN_COMPLETE', runs={a: {
        k: r[k] for k in ('completed_epoch', 'global_updates', 'best_metric', 'best_epoch', 'parameters', 'branch_parameters')}
        for a, r in summaries.items()}, comparisons={}, test_accessed=False, predictive_gain_claim=False)
    for candidate, control in pairs:
        if candidate not in summaries or control not in summaries:
            continue
        a, b = summaries[candidate], summaries[control]
        for key in ('seed', 'check', 'batch_size', 'common_backbone_sha256', 'common_branch_sha256',
                    'inputs_sha256', 'protocol_sha256', 'source_sha256', 'ramp_exchanges'):
            require(a['identity'][key] == b['identity'][key], 'Paired identity differs: '+key)
        require(all(x['train_order_sha256'] == y['train_order_sha256'] for x, y in zip(a['history'], b['history'])),
                'Paired sample orders differ')
        result['comparisons'][candidate+'_vs_'+control] = {
            region: (None if a['best_validation_metrics'][region] is None or b['best_validation_metrics'][region] is None else
                     b['best_validation_metrics'][region]['mae_macro']-a['best_validation_metrics'][region]['mae_macro'])
            for region in a['best_validation_metrics']}
    if all(a in summaries for a in ('N0', 'N1', 'P0', 'P1')):
        s = {a: summaries[a]['best_metric'] for a in ('N0', 'N1', 'P0', 'P1')}
        result['report_structure_interaction'] = (s['P0']-s['P1'])-(s['N0']-s['N1'])
    result['difference_sign'] = 'control_MAE_minus_candidate_MAE_positive_is_candidate_better'
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('data-dir', 'history-dir', 'sensors', 'output-dir'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--network-dir', type=Path)
    p.add_argument('--report-bundle', type=Path, default=Path(__file__).with_name('report_metadata_m42'))
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--seed', type=int, default=11)
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--groups', nargs='+', choices=ARMS, default=list(ARMS))
    p.add_argument('--without-ramp-exchanges', action='store_true')
    p.add_argument('--allow-exploratory', action='store_true')
    p.add_argument('--check', action='store_true')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--launcher-created-root', action='store_true', help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    require(args.allow_exploratory, 'Candidate graph/source assumptions require --allow-exploratory')
    require(1 <= args.epochs <= 100 and args.seed > 0 and len(set(args.groups)) == len(args.groups), 'Invalid budget/seed/groups')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(3)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    if device.type == 'cuda':
        require(torch.cuda.is_available(), 'CUDA unavailable')
    root = args.output_dir
    reserved = (args.launcher_created_root and root.is_dir()
                and {p.name for p in root.iterdir()} <= {'run.log'})
    require(not root.exists() or args.resume or reserved, 'Output exists; explicitly resume or use a fresh directory')
    root.mkdir(parents=True, exist_ok=True)
    original = OriginalCapacityInputs(args.data_dir, args.history_dir, args.sensors,
        Path(__file__).with_name('physics_metadata'), Path(__file__).with_name('incident_corridor_selection_v1.json'))
    inputs = None
    try:
        network_dir = args.network_dir or root/'network_pack'
        if not network_dir.exists():
            prepare(original, Path(__file__).with_name('report_metadata'), network_dir, str(device))
        inputs = CapacityDevelopmentInputs(original, network_dir, args.history_dir, args.report_bundle,
                                           ramp_exchanges=not args.without_ramp_exchanges)
        protocol_path = Path(__file__).with_name('incident_capacity_training_m42.json')
        protocol = read_json(protocol_path)
        require(len(original.events) == protocol['train_samples'] and len(inputs.events['val']) == protocol['val_samples']
                and len(original.station_ids) == protocol['stations'], 'Frozen task shape changed')
        set_seed(args.seed)
        backbone = make_model(args.data_dir, len(original.station_ids), device, 'fixed')
        graph, weights = inputs.network.graph(device)
        set_seed(args.seed+4200)
        reference_branch = IncidentCapacityBranch(copy.deepcopy(graph), weights, forecast_dim=backbone._forecast_dim).to(device)
        common_branch = copy.deepcopy(reference_branch.state_dict())
        del reference_branch
        batch_size = 2 if args.check else protocol['batch_size']
        identity = dict(schema='capacity_training_m42_v1', seed=args.seed, check=args.check, batch_size=batch_size,
            common_backbone_sha256=state_digest(backbone.state_dict()), common_branch_sha256=state_digest(common_branch),
            inputs_sha256=inputs.fingerprints, protocol_sha256=sha256(protocol_path),
            source_sha256={name: sha256(REPO/name) for name in SOURCE_FILES}, device=str(device),
            torch_version=torch.__version__, numpy_version=np.__version__, ramp_exchanges=not args.without_ramp_exchanges,
            environment=dict(python=platform.python_version(), cuda=torch.version.cuda,
                             gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                             deterministic=torch.are_deterministic_algorithms_enabled(), tf32=False),
            sample_scope='first_4_train_val' if args.check else 'original_full_train_val',
            candidate_structure_nodes=int(inputs.structure_mask.sum()), certified_structure_nodes=0)
        if (root/'identity.json').exists():
            require(read_json(root/'identity.json') == identity, 'Run identity changed')
        else:
            write_json(root/'identity.json', identity)
        summaries = {}
        first = inputs.batch('train', [0, 1], device)
        backbone.eval()
        with torch.inference_mode():
            expected = forward_arm(backbone, first, 'F', original.scaler)
        for arm in args.groups:
            model = build_arm(backbone, common_branch, graph, weights, arm, args.seed).to(device).eval()
            with torch.inference_mode():
                actual = forward_arm(model, first, arm, original.scaler)
            require(torch.equal(actual, expected), 'Zero-fusion initial prediction mismatch: '+arm)
            summaries[arm] = train_arm(model, inputs, arm, root, dict(identity, arm=arm), protocol,
                args.seed, args.epochs, batch_size, device, args.check, args.resume)
            del model
        report = pair_report(summaries)
        write_json(root/'paired_report.json', report)
        print(json.dumps(report, indent=2, allow_nan=False), flush=True)
    finally:
        if inputs is not None:
            inputs.close()
        original.close()


if __name__ == '__main__':
    main()
