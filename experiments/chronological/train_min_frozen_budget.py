"""Train one zero-initialized location adapter on a frozen published-data MIN.

Source A is read-only. Epoch zero competes with all trained epochs. OFF must
remain the original best MIN prediction at every validation position.
"""
import argparse
from contextlib import ExitStack
from datetime import datetime
import gc
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
import time
import traceback
import uuid

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.chronological import train_min_event_budget as original
from experiments.chronological import probe_min_event_budget as probe
from experiments.chronological import train_internal_spatial as legacy
from experiments.chronological.min_frozen_budget_metrics import select_hard_windows, summarize_pair, write_report
from experiments.chronological.smoke import set_seed, sha256
from experiments.chronological.train import cpu_state, save_checkpoint
from src.models.incident_spatial import require
from src.models.incident_spatial_residual import tensor_state_hash
from src.models.incident_spatial_frozen_budget import (
    build_frozen_location, frozen_state_hash, assert_frozen,
    trainable_named_parameters, set_frozen_train_mode,
)
from src.utils.igstgnn_published_events import EventPublishedData
from src.utils.metrics import compute_all_metrics, masked_mae

HERE = Path(__file__).resolve().parent
RECIPE = HERE/'min_frozen_budget_v1.json'
SCHEMA = 'frozen_MIN_location_budget_pilot_v1'
FREEZE_NORMALIZED_ATOL, FREEZE_NORMALIZED_RTOL = 1e-6, 1e-5


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run', type=Path, required=True, help='Original MIN-I run containing A/last.pt')
    p.add_argument('--run-dir', type=Path, help='Independent output; default is stable and resumable')
    p.add_argument('--data-dir', type=Path, default=original.paper.default_data())
    p.add_argument('--graph-dir', type=Path, default=HERE/'internal_spatial_graph_is11')
    p.add_argument('--source-protocol', type=Path, default=original.PROTOCOL)
    p.add_argument('--recipe', type=Path, default=RECIPE)
    p.add_argument('--epochs', type=int, help='Absolute completed adapter epoch, 1..10; default 10, --check 1')
    p.add_argument('--seed', type=int, default=2025)
    p.add_argument('--device', help='Default is the source device type; GPU name is audit-only')
    p.add_argument('--check', action='store_true', help='Requires an existing engineering-check source; never subsets formal validation')
    args = p.parse_args(argv)
    args.epochs = (1 if args.check else 10) if args.epochs is None else args.epochs
    require(1 <= args.epochs <= 10 and 0 <= args.seed < 2**32, 'Expected 1..10 epochs and a valid seed')
    for key in ('source_run', 'data_dir', 'graph_dir', 'source_protocol', 'recipe'):
        setattr(args, key, getattr(args, key).resolve())
    if args.run_dir is not None:
        args.run_dir = args.run_dir.resolve()
    return args


def source_paths(args, identity):
    paths = {'source/'+name: args.source_run/name for name in (
        'identity.json', 'initialization.json', 'data_audit.json',
        'A/last.pt', 'A/history.json', 'A/summary.json')}
    if (args.source_run/'A/test_metrics.json').is_file():
        paths['source/A/test_metrics.json'] = args.source_run/'A/test_metrics.json'
    paths.update({'training_code/'+name: REPO/name for name in identity['source_sha256']})
    paths.update({'package/'+name: args.data_dir/name for name in identity['package']})
    paths.update({'graph/'+name: args.graph_dir/name for name in identity['graph_inputs_sha256']})
    paths['source_protocol'] = args.source_protocol
    paths.update({'experiment_code/'+name: REPO/name for name in experiment_sources()})
    paths['experiment_recipe'] = args.recipe
    require(all(path.is_file() for path in paths.values()), 'A protected source input is missing')
    return paths


def verify_source(args, identity):
    require(identity.get('schema') == original.SCHEMA
            and identity.get('arm_modes', {}).get('A') == 'minimal', 'Expected a published MIN-I A source')
    require(original.source_fingerprints() == identity['source_sha256'], 'Source training code changed')
    require(sha256(args.source_protocol) == identity['protocol_sha256'], 'Source protocol changed')
    effective = original.load_protocol(argparse.Namespace(
        protocol=args.source_protocol, batch_size=identity['protocol']['batch_size']))
    require(effective == identity['protocol'], 'Source protocol contents changed')
    require(original.paper.verify_release(args.data_dir, effective) == identity['package'], 'Source release changed')
    for name, expected in identity['graph_inputs_sha256'].items():
        require(sha256(args.graph_dir/name) == expected, 'Source graph changed: '+name)
    require(legacy.read_json(args.source_run/'initialization.json') == identity['initialization'],
            'Source initialization certificate changed')


def experiment_sources():
    names = ('experiments/chronological/train_min_frozen_budget.py',
             'experiments/chronological/min_frozen_budget_metrics.py',
             'experiments/chronological/probe_min_event_budget.py',
             'experiments/chronological/min_event_budget_probe_metrics.py',
             'experiments/chronological/run_min_frozen_budget.sh',
             'src/models/incident_spatial_frozen_budget.py')
    return {name: sha256(REPO/name) for name in names}


def make_identity(args, source_identity, source_saved, certificate, recipe, paths):
    return dict(schema=SCHEMA, source_run=str(args.source_run),
        source_artifacts_sha256=probe.snapshot_files(paths),
        source_best_epoch=probe.best_row(source_saved)['epoch'],
        source_best_model_sha256=tensor_state_hash(source_saved['best_model_state']),
        frozen_state_sha256=certificate['frozen_state_sha256'],
        adapter_initial_sha256=certificate['adapter_initial_sha256'],
        source_identity=source_identity, recipe=recipe, recipe_sha256=sha256(args.recipe),
        experiment_sources_sha256=experiment_sources(), seed=args.seed,
        batch_size=source_identity['batch_size'], threads=source_identity['threads'],
        runtime=source_identity['runtime'], engineering_check=source_identity['engineering_check'],
        event_scope='published_archive_attributes_conditional_offline',
        capacity_scope='dimensionless_latent_exchange_budget_not_vehicle_capacity',
        test_payload_deserialized=True, test_evaluated=False)


def verify_output_identity(root, identity):
    path = root/'identity.json'
    if path.is_file():
        require(legacy.read_json(path) == identity, 'Frozen experiment identity changed; existing checkpoints preserved')
    else:
        require({p.name for p in root.iterdir()} <= {'active.lock'}, 'Use a fresh independent frozen run directory')
        legacy.write_json(path, identity)


def torch_metrics(prediction, target):
    labels = torch.from_numpy(target)
    null = float(labels.min()) if float(labels.min()) < 1 else 0.
    values = compute_all_metrics(torch.from_numpy(prediction), labels, torch.tensor(null))
    return dict(zip(('mae', 'mape', 'rmse'), values)), null


def source_baseline(base, data, saved):
    predictions, targets, partition = [], [], []
    base.requires_grad_(True).eval()
    before, rng = tensor_state_hash(base.state_dict()), probe.rng_digest(next(base.parameters()).device)
    for batch in data.loaders['val_loader'].get_iterator():
        row = probe.forward_case(base, 'A', batch, data.scaler, 'A_ON')
        predictions.append(row['prediction'])
        targets.append(row['target'])
        partition.append(len(row['prediction']))
    original_prediction, target = np.concatenate(predictions), np.concatenate(targets)
    metrics, null = torch_metrics(original_prediction, target)
    expected = probe.best_row(saved)
    require(np.allclose([metrics[key] for key in ('mae', 'mape', 'rmse')],
            [expected['validation_'+key] for key in ('mae', 'mape', 'rmse')],
            atol=probe.REPLAY_ATOL, rtol=probe.REPLAY_RTOL), 'Source A best validation replay failed; no adapter training permitted')
    require(tensor_state_hash(base.state_dict()) == before and probe.rng_digest(next(base.parameters()).device) == rng,
            'Source baseline inference changed state or RNG')
    require(len(original_prediction) == data.loaders['val_loader'].size, 'Incomplete source validation replay')
    # requires_grad flags can change fused-kernel dispatch by a few ULPs. The
    # strict reference therefore uses identical frozen flags, unchanged weights.
    base.requires_grad_(False).eval()
    frozen = []
    for batch in data.loaders['val_loader'].get_iterator():
        row = probe.forward_case(base, 'A', batch, data.scaler, 'A_ON')
        frozen.append(row['prediction'])
    prediction = np.concatenate(frozen)
    mean, std = float(data.scaler.mean), float(data.scaler.std)
    normalized = (prediction.astype(np.float64)-mean)/std
    original_normalized = (original_prediction.astype(np.float64)-mean)/std
    require(np.allclose(normalized, original_normalized, atol=FREEZE_NORMALIZED_ATOL, rtol=FREEZE_NORMALIZED_RTOL),
            'Freezing source MIN changed normalized predictions beyond fixed model-test roundoff tolerance')
    require(tensor_state_hash(base.state_dict()) == before, 'Freezing modified source MIN tensor values')
    freeze_change = float(np.max(np.abs(prediction.astype(np.float64)-original_prediction)))
    indices = data.audit['counts']['train_samples']+np.arange(len(prediction), dtype=np.int64)
    return dict(prediction=prediction, target=target, valid=np.isfinite(target) & (target != null),
                sample_indices=indices, mask_value=null, saved_pooled_replay=metrics,
                batch_partition=partition, source_best_epoch=expected['epoch'],
                original_to_frozen_prediction_abs_max=freeze_change,
                original_to_frozen_normalized_abs_max=float(np.max(np.abs(normalized-original_normalized))),
                original_to_frozen_normalized_atol=FREEZE_NORMALIZED_ATOL,
                original_to_frozen_normalized_rtol=FREEZE_NORMALIZED_RTOL)


def evaluate_pair(model, data, baseline, frozen_hash, hard):
    model.eval()
    assert_frozen(model, frozen_hash)
    before, rng = tensor_state_hash(model.state_dict()), probe.rng_digest(next(model.parameters()).device)
    on, position, partition, mechanisms = [], 0, [], {}
    for batch in data.loaders['val_loader'].get_iterator():
        enabled = probe.forward_case(model, 'L', batch, data.scaler, 'L_ON')
        disabled = probe.forward_case(model, 'L', batch, data.scaler, 'L_OFF')
        restored = probe.forward_case(model, 'L', batch, data.scaler, 'L_ON')
        end = position+len(enabled['prediction'])
        require(np.array_equal(disabled['prediction'], baseline['prediction'][position:end]),
                'OFF differs from the frozen source MIN; no training/commit permitted')
        require(all(np.array_equal(row['target'], baseline['target'][position:end]) for row in (enabled, disabled, restored)),
                'Paired validation target changed')
        require(np.array_equal(enabled['prediction'], restored['prediction']), 'Paired ON restoration failed')
        on.append(enabled['prediction'])
        probe.merge_mechanisms(mechanisms, enabled['mechanisms'])
        partition.append(end-position)
        position = end
    require(position == len(baseline['prediction']) and partition == baseline['batch_partition'], 'Changed original validation batches')
    require(tensor_state_hash(model.state_dict()) == before and probe.rng_digest(next(model.parameters()).device) == rng,
            'Paired validation changed model or RNG')
    assert_frozen(model, frozen_hash)
    prediction = np.concatenate(on)
    report = summarize_pair(prediction, baseline['prediction'], baseline['target'], baseline['valid'], hard)
    report.update(off_exactly_matches_source=True, on_exactly_restored=True,
                  validation_rows=position, batch_partition=partition, mechanisms=mechanisms)
    return report, prediction


def optimizer_for(model, recipe):
    named = trainable_named_parameters(model)
    optimizer = torch.optim.Adam([p for _, p in named], lr=recipe['learning_rate'],
                                 weight_decay=recipe['weight_decay'], eps=recipe['adam_eps'])
    verify_optimizer(model, optimizer)
    return optimizer


def verify_optimizer(model, optimizer):
    expected = {id(p) for _, p in trainable_named_parameters(model)}
    actual = [id(p) for group in optimizer.param_groups for p in group['params']]
    require(len(actual) == len(expected) and set(actual) == expected, 'Optimizer must contain only and all adapter parameters')


def train_epoch(model, data, optimizer, recipe, frozen_hash):
    set_frozen_train_mode(model)
    assert_frozen(model, frozen_hash)
    verify_optimizer(model, optimizer)
    loader, device = data.loaders['train_loader'], next(model.parameters()).device
    loader.shuffle()
    losses, active_batches, optimizer_steps, gradient_max = [], 0, 0, 0.
    before = tensor_state_hash(model.event_budget_adapter.state_dict())
    for number, batch in enumerate(loader.get_iterator(), 1):
        optimizer.zero_grad(set_to_none=True)
        digest = probe.input_digest(batch)
        x = torch.as_tensor(batch['x_data'], dtype=torch.float32, device=device)
        y = torch.as_tensor(batch['y_data'], dtype=torch.float32, device=device)
        events = original.event_tensors(batch['event_data'], device)
        pred = model(x, incident_data=events, report_enabled=True)
        pred = pred*data.scaler.std.to(device)+data.scaler.mean.to(device)
        target = y*data.scaler.std.to(device)+data.scaler.mean.to(device)
        require(pred.shape == target.shape and pred.shape[1] == 12 and bool(torch.isfinite(pred).all()), 'Invalid 12-step training prediction')
        null = target.min() if target.min() < 1 else target.new_zeros(())
        loss = masked_mae(pred, target, null)
        require(bool(torch.isfinite(loss)), 'Non-finite adapter loss')
        if loss.requires_grad:
            loss.backward()
            gradients = [p.grad for _, p in trainable_named_parameters(model) if p.grad is not None]
            require(gradients and all(bool(torch.isfinite(g).all()) for g in gradients), 'Missing/non-finite adapter gradients')
            peak = max(float(g.abs().max()) for g in gradients)
            gradient_max = max(gradient_max, peak)
            active_batches += int(peak > 0)
            torch.nn.utils.clip_grad_norm_([p for _, p in trainable_named_parameters(model)], recipe['clip_grad_norm'])
            optimizer.step()
            optimizer_steps += 1
        else:
            support = (events['relation'].abs().sum(-1) > 0) & events['report_present'][:, None] & events['relation_available'][:, None]
            src, dst = model.internal_spatial_graph.edge_index
            eligible = (support[:, src.clamp_min(0)] | support[:, dst.clamp_min(0)]) & ((src >= 0) & (dst >= 0))[None]
            require(not bool(eligible.any()), 'Eligible adapter branch is disconnected from training loss')
        require(probe.input_digest(batch) == digest, 'Training changed an input payload')
        require(all(p.grad is None for name, p in model.named_parameters() if not name.startswith('event_budget_adapter.')),
                'Frozen backbone acquired gradients')
        losses.append(float(loss.detach().cpu()))
        if number % 16 == 0 or number == loader.num_batch:
            print(f'Adapter training: {number}/{loader.num_batch} batches; full 12-step loss', flush=True)
    optimizer.zero_grad(set_to_none=True)
    assert_frozen(model, frozen_hash)
    return dict(train_mae_batch_mean=float(np.mean(losses)), full_supervised_horizons=12,
                batches=loader.num_batch, batches_with_nonzero_adapter_gradient=active_batches,
                optimizer_steps=optimizer_steps,
                adapter_gradient_abs_max=gradient_max,
                adapter_state_changed=tensor_state_hash(model.event_budget_adapter.state_dict()) != before,
                train_order_sha256=hashlib.sha256(loader.order.tobytes()).hexdigest())


def capture(model, optimizer, data, identity, history, best_state, best_epoch, best_metric, baseline_sha):
    device = next(model.parameters()).device
    return dict(schema=SCHEMA, identity=identity, completed_epoch=history[-1]['epoch'], history=history,
        adapter_state=cpu_state(model.event_budget_adapter.state_dict()), best_adapter_state=cpu_state(best_state),
        best_epoch=best_epoch, best_metric=best_metric, optimizer_state=optimizer.state_dict(),
        train_order=data.loaders['train_loader'].order.copy(), numpy_rng=np.random.get_state(),
        torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else [],
        baseline_sha256=baseline_sha, frozen_state_sha256=identity['frozen_state_sha256'])


def validate_checkpoint(saved, identity, model, baseline_sha):
    require(saved.get('schema') == SCHEMA and saved.get('identity') == identity, 'Frozen checkpoint identity changed')
    history, epoch = saved['history'], saved['completed_epoch']
    require(type(epoch) is int and 0 <= epoch <= identity['recipe']['max_epochs']
            and len(history) == epoch+1 and [r['epoch'] for r in history] == list(range(epoch+1)), 'Invalid frozen checkpoint history')
    chosen = min(range(len(history)), key=lambda i: history[i]['validation']['on']['mae_pooled'])
    require(saved['best_epoch'] == chosen and saved['best_metric'] == history[chosen]['validation']['on']['mae_pooled'],
            'Best adapter selection disagrees with history')
    require(saved['baseline_sha256'] == baseline_sha and saved['frozen_state_sha256'] == frozen_state_hash(model), 'Frozen baseline changed')
    order = np.asarray(saved['train_order'])
    if epoch:
        require(hashlib.sha256(order.tobytes()).hexdigest() == history[-1]['training']['train_order_sha256'],
                'Saved training order differs from its committed digest')
    else:
        require(np.array_equal(order, np.arange(len(order), dtype=np.int64)), 'Epoch-zero training order is not canonical')
    spec = model.event_budget_adapter.state_dict()
    for key in ('adapter_state', 'best_adapter_state'):
        state = saved[key]
        require(state.keys() == spec.keys() and all(state[n].shape == p.shape and state[n].dtype == p.dtype
                and bool(torch.isfinite(state[n]).all()) for n, p in spec.items()), 'Invalid saved adapter tensors')


def repair_history(root, saved):
    path = root/'history.json'
    if path.is_file():
        try:
            old = legacy.read_json(path)
        except (json.JSONDecodeError, UnicodeDecodeError):
            path.with_name('history.interrupted_'+uuid.uuid4().hex[:6]+'.json').write_bytes(path.read_bytes())
        else:
            require(isinstance(old, list) and len(old) <= len(saved['history']) and old == saved['history'][:len(old)],
                    'Frozen run history conflicts with checkpoint; refusing overwrite')
    legacy.write_json(path, saved['history'])


def restore(saved, model, optimizer, data):
    model.event_budget_adapter.load_state_dict(saved['adapter_state'], strict=True)
    optimizer.load_state_dict(saved['optimizer_state'])
    verify_optimizer(model, optimizer)
    data.loaders['train_loader'].set_order(saved['train_order'])
    np.random.set_state(saved['numpy_rng'])
    torch.set_rng_state(saved['torch_rng'])
    if next(model.parameters()).device.type == 'cuda':
        torch.cuda.set_rng_state_all(saved['cuda_rng'])


def fit(model, base, data, source_saved, identity, root, target_epoch, recipe, *, protected_paths=None):
    """Single-process epochs; last.pt is authoritative after interruptions."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    require(1 <= target_epoch <= recipe['max_epochs'], 'Target exceeds the fixed pilot budget')
    if protected_paths:
        probe.require_preserved(identity['source_artifacts_sha256'], protected_paths)
    if not (root/'last.pt').is_file():
        require(tensor_state_hash(model.event_budget_adapter.state_dict()) == identity['adapter_initial_sha256'],
                'Fresh experiment requires the canonical zero-start adapter')
        data.loaders['train_loader'].set_order(np.arange(data.loaders['train_loader'].size))
        set_seed(identity['seed'])
    frozen_hash = identity['frozen_state_sha256']
    assert_frozen(model, frozen_hash)
    baseline = source_baseline(base, data, source_saved)
    hard = select_hard_windows(baseline['prediction'], baseline['target'], baseline['valid'],
                               baseline['sample_indices'], fraction=recipe['fixed_hard_fraction'])
    baseline['fixed_hard_indices'] = hard
    archive = root/'baseline_predictions.npz'
    if (root/'last.pt').is_file():
        require(archive.is_file(), 'Committed frozen run is missing baseline predictions')
        with np.load(archive, allow_pickle=False) as original_baseline:
            require(all(np.array_equal(original_baseline[key], baseline[key]) for key in (
                'prediction', 'target', 'valid', 'sample_indices', 'fixed_hard_indices')), 'Stored source baseline replay differs')
    else:
        partial = root/'baseline_predictions.partial.npz'
        np.savez_compressed(partial, **{key: baseline[key] for key in (
            'prediction', 'target', 'valid', 'sample_indices', 'fixed_hard_indices')})
        partial.replace(archive)
    baseline_sha = sha256(archive)
    optimizer = optimizer_for(model, recipe)
    path = root/'last.pt'
    if path.is_file():
        saved = torch.load(path, map_location='cpu', weights_only=False)
        validate_checkpoint(saved, identity, model, baseline_sha)
        repair_history(root, saved)
        restore(saved, model, optimizer, data)
        replay, _ = evaluate_pair(model, data, baseline, frozen_hash, hard)
        require(replay == saved['history'][-1]['validation'], 'Last adapter validation replay differs; refusing continuation')
        print(f'Resuming committed adapter epoch {saved["completed_epoch"]}', flush=True)
    else:
        initial, on = evaluate_pair(model, data, baseline, frozen_hash, hard)
        require(np.array_equal(on, baseline['prediction']), 'Zero adapter differs from source A; no training permitted')
        epoch0 = dict(epoch=0, validation=initial, global_updates=0, optimizer_updates=0,
                      full_supervised_horizons=12, frozen_state_sha256=frozen_hash)
        saved = capture(model, optimizer, data, identity, [epoch0], model.event_budget_adapter.state_dict(),
                        0, initial['on']['mae_pooled'], baseline_sha)
        legacy.write_json(root/'baseline.json', dict(source_best_epoch=baseline['source_best_epoch'],
            saved_pooled_replay=baseline['saved_pooled_replay'], mask_value=baseline['mask_value'],
            validation_rows=len(baseline['prediction']), batch_partition=baseline['batch_partition'],
            original_to_frozen_prediction_abs_max=baseline['original_to_frozen_prediction_abs_max'],
            original_to_frozen_normalized_abs_max=baseline['original_to_frozen_normalized_abs_max'],
            original_to_frozen_normalized_atol=baseline['original_to_frozen_normalized_atol'],
            original_to_frozen_normalized_rtol=baseline['original_to_frozen_normalized_rtol'],
            fixed_hard_sample_indices=baseline['sample_indices'][hard].tolist(),
            zero_ON_and_OFF_exactly_match_frozen_MIN=True,
            original_source_metrics_replay_passed=True, validation=initial))
        save_checkpoint(path, saved)
        legacy.write_json(root/'history.json', saved['history'])
        print('FROZEN_MIN_EPOCH_ZERO_REPLAY_PASS; only the location adapter may update', flush=True)
    history, best_epoch, best_metric = saved['history'], saved['best_epoch'], saved['best_metric']
    best_state, start_epoch = saved['best_adapter_state'], saved['completed_epoch']
    for epoch in range(start_epoch+1, target_epoch+1):
        start = time.perf_counter()
        train = train_epoch(model, data, optimizer, recipe, frozen_hash)
        validation, _ = evaluate_pair(model, data, baseline, frozen_hash, hard)
        metric = validation['on']['mae_pooled']
        if metric < best_metric:
            best_metric, best_epoch, best_state = metric, epoch, cpu_state(model.event_budget_adapter.state_dict())
        row = dict(epoch=epoch, training=train, validation=validation,
            global_updates=history[-1]['global_updates']+train['batches'],
            optimizer_updates=history[-1]['optimizer_updates']+train['optimizer_steps'],
            full_supervised_horizons=12, frozen_state_sha256=frozen_hash, seconds=time.perf_counter()-start)
        history.append(row)
        saved = capture(model, optimizer, data, identity, history, best_state, best_epoch, best_metric, baseline_sha)
        save_checkpoint(path, saved)
        legacy.write_json(root/'history.json', history)
        print(json.dumps(dict(epoch=epoch, validation_mae=metric, best_epoch=best_epoch,
            best_mae=best_metric, off_minus_on_mae=validation['mae_gain_pooled'],
            frozen_backbone_preserved=True, seconds=row['seconds']), allow_nan=False), flush=True)
        gc.collect()
    # Read the best adapter without replacing the authoritative current checkpoint.
    current = cpu_state(model.event_budget_adapter.state_dict())
    model.event_budget_adapter.load_state_dict(best_state, strict=True)
    best_report, best_predictions = evaluate_pair(model, data, baseline, frozen_hash, hard)
    model.event_budget_adapter.load_state_dict(current, strict=True)
    require(best_report == history[best_epoch]['validation'], 'Best adapter validation replay differs')
    assert_frozen(model, frozen_hash)
    if protected_paths:
        preservation = probe.require_preserved(identity['source_artifacts_sha256'], protected_paths)
        legacy.write_json(root/'source_preservation.json', preservation)
    np.savez_compressed(root/'best_predictions.npz', on=best_predictions, off=baseline['prediction'],
        target=baseline['target'], valid=baseline['valid'], sample_indices=baseline['sample_indices'],
        fixed_hard_indices=hard)
    baseline_metrics = history[0]['validation']['on']
    selected = best_report['on']
    gain = baseline_metrics['mae_pooled']-selected['mae_pooled']
    summary = dict(status='FROZEN_MIN_LOCATION_ENGINEERING_PASS' if identity['engineering_check'] else
        'FROZEN_MIN_LOCATION_PILOT_COMPLETE' if saved['completed_epoch'] == recipe['max_epochs'] else
        'FROZEN_MIN_LOCATION_PAUSED_AT_EPOCH_BOUNDARY',
        completed_epoch=saved['completed_epoch'], requested_epoch=target_epoch, pilot_max_epochs=recipe['max_epochs'],
        source_best_epoch=baseline['source_best_epoch'], best_adapter_epoch=best_epoch,
        baseline_mae=baseline_metrics['mae_pooled'], best_mae=selected['mae_pooled'], baseline_minus_best_mae=gain,
        baseline_metrics=baseline_metrics, best_metrics=selected,
        best_validation=best_report, best_adapter_selected=best_epoch > 0,
        exploratory_validation_increment=best_epoch > 0 and gain > 0,
        selection_includes_epoch_zero=True, fixed_hard_windows=len(hard),
        trainable_parameters=sum(p.numel() for _, p in trainable_named_parameters(model)),
        frozen_parameters=sum(p.numel() for p in model.parameters() if not p.requires_grad),
        global_updates=saved['history'][-1]['global_updates'], optimizer_updates=saved['history'][-1]['optimizer_updates'],
        optimizer_updates_this_invocation=saved['history'][-1]['optimizer_updates']-saved['history'][start_epoch]['optimizer_updates'],
        frozen_state_sha256=frozen_hash, frozen_backbone_preserved=True,
        off_exactly_matches_source=True, source_artifacts_preserved=bool(protected_paths),
        exact_OFF_reference='unchanged_source_weights_in_frozen_MIN_execution_mode',
        engineering_only=identity['engineering_check'], test_evaluated=False, predictive_gain_claim=False,
        interpretation='frozen_backbone_adapter_path_increment_not_unique_location_content_or_real_incident_causality',
        output_dir=str(root))
    legacy.write_json(root/'summary.json', summary)
    write_report(root, history[0]['validation'], history, summary)
    return summary


def main(argv=None):
    args = parse_args(argv)
    require(args.source_run.is_dir(), 'Source run does not exist')
    require((args.source_run/'A').is_dir(), 'Source A directory does not exist')
    source_identity = legacy.read_json(args.source_run/'identity.json')
    require(source_identity.get('schema') == original.SCHEMA, 'Expected a MIN-I published source run')
    require(not args.check or source_identity['engineering_check'], '--check requires an engineering source; formal validation cannot be cut')
    args.check = bool(source_identity['engineering_check'])
    args.device = args.device or ('cuda:0' if source_identity['runtime']['device_type'] == 'cuda' else 'cpu')
    args.threads = source_identity['threads']
    if args.run_dir is None:
        args.run_dir = REPO/'experiments/chronological_runs'/(
            'contra_min_i_frozen_loc_v1_'+('check_' if args.check else '')+
            f'bs{source_identity["batch_size"]}_seed{args.seed}')
    require(args.run_dir != args.source_run and args.source_run not in args.run_dir.parents
            and args.run_dir not in args.source_run.parents, 'Use an independent directory outside the source run')
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    require(device.type in ('cpu', 'cuda') and (device.type != 'cuda' or torch.cuda.is_available()), 'Requested device unavailable')
    probe.verify_runtime(source_identity, device)
    recipe = legacy.read_json(args.recipe)
    require(recipe == legacy.read_json(RECIPE) and recipe['supervised_horizons'] == 12
            and not recipe['curriculum']
            and recipe['freeze_dispatch_normalized_atol'] == FREEZE_NORMALIZED_ATOL
            and recipe['freeze_dispatch_normalized_rtol'] == FREEZE_NORMALIZED_RTOL, 'Frozen pilot recipe changed')
    logger = logging.getLogger('min_frozen_budget')
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler(sys.stdout))
    args.run_dir.mkdir(parents=True, exist_ok=True)
    with ExitStack() as locks:
        locks.enter_context(legacy.run_lock(args.source_run))
        locks.enter_context(legacy.run_lock(args.source_run/'A'))
        locks.enter_context(legacy.run_lock(args.run_dir))
        verify_source(args, source_identity)
        paths = source_paths(args, source_identity)
        before = probe.snapshot_files(paths)
        source_saved = torch.load(args.source_run/'A/last.pt', map_location='cpu', weights_only=False)
        original.validate_saved(source_saved, source_identity, 'A')
        probe.verify_saved_sidecars(args.source_run/'A', source_saved)
        data = EventPublishedData(args.data_dir, source_identity['protocol'], logger, check=args.check)
        require(data.audit == legacy.read_json(args.source_run/'data_audit.json'), 'Source data audit differs')
        graph, _, graph_info = legacy.load_graph(args.graph_dir, data.station_ids, source_identity['protocol']['graph_sha256'])
        require(graph_info == source_identity['graph'], 'Source graph audit differs')
        base_args = argparse.Namespace(data_dir=args.data_dir, seed=source_identity['seed'])
        base, _ = original.build_arm(base_args, 'A', len(data.station_ids), graph, device)
        base.load_state_dict(source_saved['best_model_state'], strict=True)
        base.eval()
        set_seed(args.seed)
        model, certificate = build_frozen_location(base)
        identity = make_identity(args, source_identity, source_saved, certificate, recipe, paths)
        require(identity['source_artifacts_sha256'] == before, 'Source changed during preparation')
        verify_output_identity(args.run_dir, identity)
        initialization_path = args.run_dir/'initialization.json'
        if initialization_path.is_file():
            require(legacy.read_json(initialization_path) == certificate, 'Frozen initialization certificate changed')
        else:
            require(not (args.run_dir/'last.pt').exists(), 'Committed run is missing its initialization certificate')
            legacy.write_json(initialization_path, certificate)
        session = args.run_dir/('session_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6])
        session.mkdir()
        legacy.write_json(session/'environment.json', legacy.environment(device))
        legacy.write_json(session/'request.json', dict(source_run=str(args.source_run), epochs=args.epochs,
            recipe=recipe, batch_size=source_identity['batch_size'], seed=args.seed, test_evaluated=False))
        print('Source best MIN: '+str(args.source_run/'A/last.pt'), flush=True)
        print('Run directory: '+str(args.run_dir), flush=True)
        print('Session directory: '+str(session), flush=True)
        try:
            result = fit(model, base, data, source_saved, identity, args.run_dir, args.epochs, recipe, protected_paths=paths)
            probe.require_preserved(before, paths)
            print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
        except BaseException as error:
            failed = dict(status='FROZEN_MIN_LOCATION_FAILED', error_type=type(error).__name__, error=str(error),
                          traceback=traceback.format_exc(), checkpoint_preserved=(args.run_dir/'last.pt').is_file())
            legacy.write_json(session/'failure.json', failed)
            legacy.write_json(args.run_dir/'summary.json', failed)
            raise


if __name__ == '__main__':
    main()
