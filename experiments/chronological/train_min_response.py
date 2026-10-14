"""Stage 1 diagnostic: train a conditional response on an immutable MIN.

The two arms share backbone, support, initialization of common parameters and
sample order. Only location reads numerical relation details. Epoch zero remains
eligible; this frozen stage is not the final joint optimization. Validation and
checkpoints never imply real capacity identification.
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
import subprocess
import sys
import time
import traceback
import uuid

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.chronological import train_min_frozen_budget as frozen
from experiments.chronological import train_min_event_budget as original
from experiments.chronological import probe_min_event_budget as probe
from experiments.chronological import train_internal_spatial as legacy
from experiments.chronological.min_response_metrics import paired_training_loss, summarize_response_pair, write_diagnostics
from experiments.chronological.smoke import set_seed, sha256
from experiments.chronological.train import cpu_state, save_checkpoint
from src.models.incident_spatial import require
from src.models.incident_spatial_residual import tensor_state_hash
from src.models.incident_spatial_response import build_frozen_response
from src.utils.igstgnn_published_events import EventPublishedData

HERE = Path(__file__).resolve().parent
RECIPE = HERE/'min_response_v1.json'
SCHEMA = 'frozen_MIN_conditional_response_pair_v1'
ARMS = ('location', 'state')
WORKER_ENV = 'IGSTGNN_MIN_RESPONSE_CHILD'
BASELINE_KEYS = ('prediction', 'target', 'valid', 'sample_indices', 'fixed_hard_indices', 'history_low_mask')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--data-dir', type=Path, default=original.paper.default_data())
    parser.add_argument('--graph-dir', type=Path, default=HERE/'internal_spatial_graph_is11')
    parser.add_argument('--source-protocol', type=Path, default=original.PROTOCOL)
    parser.add_argument('--recipe', type=Path, default=RECIPE)
    parser.add_argument('--arms', choices=ARMS, nargs='+', default=list(ARMS))
    parser.add_argument('--worker', choices=ARMS, help='Prepared arm only; epochs is an absolute target')
    parser.add_argument('--epochs', type=int, default=1, help='Absolute completed response epoch, 1..10')
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--device', help='Source device type by default; GPU name is audit-only')
    parser.add_argument('--check', action='store_true', help='Engineering source only; formal validation is never subset')
    args = parser.parse_args(argv)
    require(len(set(args.arms)) == len(args.arms), 'Duplicate arms are not permitted')
    require(args.worker is None or args.worker in args.arms, 'Worker must belong to the recorded arm set')
    args.arms = [arm for arm in ARMS if arm in args.arms]
    require(1 <= args.epochs <= 10 and 0 <= args.seed < 2**32, 'Expected 1..10 epochs and a valid seed')
    for key in ('source_run', 'data_dir', 'graph_dir', 'source_protocol', 'recipe'):
        setattr(args, key, getattr(args, key).resolve())
    if args.run_dir is not None:
        args.run_dir = args.run_dir.resolve()
    return args


def experiment_sources():
    result = frozen.experiment_sources()
    names = ('experiments/chronological/train_min_response.py',
             'experiments/chronological/min_response_metrics.py',
             'experiments/chronological/run_min_response.sh',
             'src/models/incident_spatial_response.py')
    result.update({name: sha256(REPO/name) for name in names})
    return result


def source_paths(args, identity):
    paths = frozen.source_paths(args, identity)
    paths.update({'experiment_code/'+name: REPO/name for name in experiment_sources()})
    require(all(path.is_file() for path in paths.values()), 'A protected response experiment input is missing')
    return paths


def require_independent_directory(args):
    root, source = args.run_dir.resolve(), args.source_run.resolve()
    require(root != source and source not in root.parents and root not in source.parents,
            'Use an independent directory outside the source run')
    for ancestor in root.parents:
        require(not (ancestor/'identity.json').is_file(), 'Output is inside an existing experiment')
    if (root/'identity.json').is_file():
        require(legacy.read_json(root/'identity.json').get('schema') == SCHEMA,
                'Use a fresh response experiment; legacy runs cannot be resumed')


def load_source(args, logger):
    identity = legacy.read_json(args.source_run/'identity.json')
    frozen.verify_source(args, identity)
    saved = torch.load(args.source_run/'A/last.pt', map_location='cpu', weights_only=False)
    original.validate_saved(saved, identity, 'A')
    probe.verify_saved_sidecars(args.source_run/'A', saved)
    data = EventPublishedData(args.data_dir, identity['protocol'], logger, check=args.check)
    require(data.audit == legacy.read_json(args.source_run/'data_audit.json'), 'Source data audit differs')
    graph, _, audit = legacy.load_graph(args.graph_dir, data.station_ids, identity['protocol']['graph_sha256'])
    require(audit == identity['graph'], 'Source graph audit differs')
    base_args = argparse.Namespace(data_dir=args.data_dir, seed=identity['seed'])
    base, _ = original.build_arm(base_args, 'A', len(data.station_ids), graph, torch.device(args.device))
    base.load_state_dict(saved['best_model_state'], strict=True)
    return identity, saved, data, base.eval()


def build_response(base, arm, recipe):
    return build_frozen_response(base, mode=arm, **{key: recipe[key] for key in
        ('latent', 'feature_width', 'interaction_rank', 'steps', 'eta')})


def baseline_digest(baseline):
    digest = hashlib.sha256()
    for key in BASELINE_KEYS:
        value = np.ascontiguousarray(baseline[key])
        digest.update(key.encode())
        digest.update(str(value.shape).encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def response_baseline(base, data, saved, recipe):
    baseline = frozen.source_baseline(base, data, saved)
    baseline['fixed_hard_indices'] = frozen.select_hard_windows(
        baseline['prediction'], baseline['target'], baseline['valid'], baseline['sample_indices'],
        fraction=recipe['fixed_hard_fraction'])
    low = []
    mean, std = float(data.scaler.mean), float(data.scaler.std)
    for batch in data.loaders['val_loader'].get_iterator():
        history = np.asarray(batch['x_data'], dtype=np.float64)[..., 0]*std+mean
        average = history.mean(axis=1)
        group = (average >= 0.) & (average < recipe['low_history_threshold'])
        low.append(group)
    baseline['history_low_mask'] = np.concatenate(low)
    require(baseline['history_low_mask'].shape == (len(baseline['target']), baseline['target'].shape[2]),
            'Historical group alignment changed')
    return baseline


def forward_case(model, batch, scaler, enabled):
    require(not any(module.training for module in model.modules()), 'Response replay requires eval mode')
    device = next(model.parameters()).device
    before = probe.input_digest(batch)
    x = torch.as_tensor(batch['x_data'], dtype=torch.float32, device=device)
    y = torch.as_tensor(batch['y_data'], dtype=torch.float32, device=device)
    events = original.event_tensors(batch['event_data'], device)
    try:
        with torch.inference_mode():
            result = model(x, incident_data=events, report_enabled=enabled, return_details=True)
            prediction = result['prediction']*scaler.std.to(device)+scaler.mean.to(device)
            target = y*scaler.std.to(device)+scaler.mean.to(device)
        require(prediction.shape == target.shape and prediction.shape[1] == 12,
                'Invalid full-horizon response replay')
        require(bool(torch.isfinite(prediction).all()), 'Non-finite response replay')
        return dict(prediction=prediction[..., 0].cpu().numpy(), target=target[..., 0].cpu().numpy(),
                    mechanisms=result['response'])
    finally:
        require(probe.input_digest(batch) == before, 'Response replay changed an input payload')


def evaluate_pair(model, data, baseline, frozen_hash, hard):
    model.eval()
    model.assert_backbone_frozen(frozen_hash)
    before, rng = tensor_state_hash(model.state_dict()), probe.rng_digest(next(model.parameters()).device)
    predictions, partition, diagnostics, position = [], [], [], 0
    for batch in data.loaders['val_loader'].get_iterator():
        on = forward_case(model, batch, data.scaler, True)
        off = forward_case(model, batch, data.scaler, False)
        restored = forward_case(model, batch, data.scaler, True)
        end = position+len(on['prediction'])
        require(np.array_equal(off['prediction'], baseline['prediction'][position:end]),
                'OFF differs from frozen MIN; no training/commit permitted')
        require(all(np.array_equal(row['target'], baseline['target'][position:end]) for row in (on, off, restored)),
                'Paired validation targets changed')
        require(np.array_equal(on['prediction'], restored['prediction']), 'Paired ON restoration failed')
        require(on['mechanisms'] == restored['mechanisms'], 'Paired response diagnostics changed on restoration')
        predictions.append(on['prediction'])
        diagnostics.append(on['mechanisms'])
        partition.append(end-position)
        position = end
    require(position == len(baseline['prediction']) and partition == baseline['batch_partition'],
            'Changed original validation batch partition')
    require(tensor_state_hash(model.state_dict()) == before and probe.rng_digest(next(model.parameters()).device) == rng,
            'Response validation changed model tensors or RNG')
    model.assert_backbone_frozen(frozen_hash)
    prediction = np.concatenate(predictions)
    report = summarize_response_pair(prediction, baseline['prediction'], baseline['target'], baseline['valid'],
                                     hard, history_low_mask=baseline['history_low_mask'])
    report.update(off_exactly_matches_source=True, on_exactly_restored=True,
        validation_rows=position, batch_partition=partition,
        mechanisms=dict(scope='dimensionless_latent_response_not_real_capacity', batches=diagnostics))
    return report, prediction


def preflight(args, source_identity, saved, data, base, recipe):
    baseline = response_baseline(base, data, saved, recipe)
    certificates = {}
    for arm in ARMS:
        set_seed(args.seed)
        model, certificate = build_response(base, arm, recipe)
        report, prediction = evaluate_pair(model, data, baseline, certificate['frozen_state_sha256'],
                                          baseline['fixed_hard_indices'])
        require(np.array_equal(prediction, baseline['prediction']), 'Initial response ON differs from frozen MIN')
        require(report['off_exactly_matches_source'] and report['on_exactly_restored'], 'Initial pair replay failed')
        certificates[arm] = certificate
        del model
    require(len({c['frozen_state_sha256'] for c in certificates.values()}) == 1, 'Paired source states differ')
    require(len({c['common_initial_sha256'] for c in certificates.values()}) == 1, 'Common response initialization differs')
    require(certificates['location']['frozen_state_sha256'] == tensor_state_hash(base.state_dict()), 'Source backbone changed')
    return dict(status='MIN_RESPONSE_INITIALIZATION_PAIR_PASS', arms=certificates,
        source_best_epoch=probe.best_row(saved)['epoch'], baseline_sha256=baseline_digest(baseline),
        validation_rows=len(baseline['prediction']), batch_partition=baseline['batch_partition'],
        fixed_hard_sample_indices=baseline['sample_indices'][baseline['fixed_hard_indices']].tolist(),
        all_initial_ON_OFF_exactly_match_MIN=True, common_initialization_exact=True,
        trainable_parameters={arm: certificates[arm]['trainable_parameters'] for arm in ARMS},
        original_source_metrics_replay_passed=True, test_evaluated=False)


def make_identity(args, source_identity, saved, initialization, recipe, paths):
    return dict(schema=SCHEMA, source_run=str(args.source_run), arms=args.arms,
        training_stage=recipe['training_stage'],
        source_artifacts_sha256=probe.snapshot_files(paths), source_identity=source_identity,
        source_best_epoch=probe.best_row(saved)['epoch'], source_best_model_sha256=tensor_state_hash(saved['best_model_state']),
        initialization=initialization, recipe=recipe, recipe_sha256=sha256(args.recipe),
        experiment_sources_sha256=experiment_sources(), seed=args.seed,
        batch_size=source_identity['batch_size'], threads=source_identity['threads'], runtime=source_identity['runtime'],
        engineering_check=source_identity['engineering_check'], event_scope='published_archive_attributes_conditional_offline',
        capacity_scope=recipe['capacity_scope'], time_scope=recipe['time_scope'],
        reference_scope='shared_latent_computation_reference_not_no_incident_counterfactual',
        test_payload_deserialized=True, test_evaluated=False)


def arm_identity(identity, arm):
    require(arm in identity['arms'], 'Arm is absent from the prepared experiment')
    certificate = identity['initialization']['arms'][arm]
    return dict(identity, arm=arm, response_mode=arm, response_certificate=certificate,
        frozen_state_sha256=certificate['frozen_state_sha256'],
        response_initial_sha256=certificate['response_initial_sha256'])


def verify_identity(root, identity, allow_new):
    path = root/'identity.json'
    if path.is_file():
        require(legacy.read_json(path) == identity, 'Response experiment identity changed; checkpoints preserved')
        require(legacy.read_json(root/'initialization.json') == identity['initialization'], 'Initialization sidecar changed')
    else:
        require(allow_new, 'Worker requires a completed parent initialization')
        require({p.name for p in root.iterdir()} <= {'active.lock'}, 'Use a fresh independent response run')
        legacy.write_json(path, identity)
        legacy.write_json(root/'initialization.json', identity['initialization'])


def verify_optimizer(model, optimizer):
    expected = {id(p) for p in model.response.parameters() if p.requires_grad}
    actual = [id(p) for group in optimizer.param_groups for p in group['params']]
    require(expected and len(actual) == len(expected) and set(actual) == expected,
            'Optimizer must contain only and all trainable response parameters')


def optimizer_for(model, recipe):
    optimizer = torch.optim.Adam([p for p in model.response.parameters() if p.requires_grad],
        lr=recipe['learning_rate'], weight_decay=recipe['weight_decay'], eps=recipe['adam_eps'])
    verify_optimizer(model, optimizer)
    return optimizer


def train_epoch(model, data, optimizer, recipe, frozen_hash):
    model.train()
    model.backbone.eval()
    model.assert_backbone_frozen(frozen_hash)
    verify_optimizer(model, optimizer)
    loader, device = data.loaders['train_loader'], next(model.parameters()).device
    loader.shuffle()
    losses, main_losses, risk_losses, active, condition_active, steps, gradient_max = [], [], [], 0, 0, 0, 0.
    before = tensor_state_hash(model.response.state_dict())
    for number, batch in enumerate(loader.get_iterator(), 1):
        optimizer.zero_grad(set_to_none=True)
        digest = probe.input_digest(batch)
        x = torch.as_tensor(batch['x_data'], dtype=torch.float32, device=device)
        y = torch.as_tensor(batch['y_data'], dtype=torch.float32, device=device)
        events = original.event_tensors(batch['event_data'], device)
        result = model(x, incident_data=events, report_enabled=True, return_details=True)
        prediction, off = result['prediction'], result['baseline_prediction']
        require(prediction.shape == y.shape and prediction.shape[1] == 12 and bool(torch.isfinite(prediction).all()),
                'Invalid full-horizon response training prediction')
        loss, stats = paired_training_loss(prediction, off, y, x, data.scaler.mean, data.scaler.std,
            low_history_threshold=recipe['low_history_threshold'], risk_weight=recipe['risk_weight'])
        require(bool(torch.isfinite(loss)), 'Non-finite response loss')
        if loss.requires_grad:
            loss.backward()
            gradients = [p.grad for p in model.response.parameters() if p.requires_grad and p.grad is not None]
            require(gradients and all(bool(torch.isfinite(g).all()) for g in gradients), 'Missing/non-finite response gradients')
            peak = max(float(g.abs().max()) for g in gradients)
            gradient_max = max(gradient_max, peak)
            active += int(peak > 0)
            condition_active += int(any(p.grad is not None and bool((p.grad != 0).any())
                                        for p in model.response.condition_output.parameters()))
            torch.nn.utils.clip_grad_norm_([p for p in model.response.parameters() if p.requires_grad], recipe['clip_grad_norm'])
            optimizer.step()
            steps += 1
        else:
            require(result['response']['eligible_edge_positions'] == 0,
                    'Eligible response branch is disconnected from training loss')
        require(probe.input_digest(batch) == digest, 'Response training changed an input payload')
        require(all(p.grad is None for p in model.backbone.parameters()), 'Frozen MIN acquired gradients')
        losses.append(float(loss.detach().cpu()))
        main_losses.append(float(stats['mae_on']))
        risk_losses.append(float(stats.get('risk_penalty', 0.)))
        if number % 16 == 0 or number == loader.num_batch:
            print(f'Response training: {number}/{loader.num_batch} batches; full 12-step loss', flush=True)
    optimizer.zero_grad(set_to_none=True)
    model.assert_backbone_frozen(frozen_hash)
    return dict(loss_batch_mean=float(np.mean(losses)), main_mae_batch_mean=float(np.mean(main_losses)),
        risk_penalty_batch_mean=float(np.mean(risk_losses)), full_supervised_horizons=12,
        batches=loader.num_batch, batches_with_nonzero_response_gradient=active, optimizer_steps=steps,
        batches_with_nonzero_condition_gradient=condition_active,
        response_gradient_abs_max=gradient_max, response_state_changed=tensor_state_hash(model.response.state_dict()) != before,
        train_order_sha256=hashlib.sha256(loader.order.tobytes()).hexdigest())


def capture(model, optimizer, data, identity, history, best_state, best_epoch, best_metric, baseline_sha):
    device = next(model.parameters()).device
    return dict(schema=SCHEMA, identity=identity, completed_epoch=history[-1]['epoch'], history=history,
        response_state=cpu_state(model.response.state_dict()), best_response_state=cpu_state(best_state),
        best_epoch=best_epoch, best_metric=best_metric, optimizer_state=optimizer.state_dict(),
        train_order=data.loaders['train_loader'].order.copy(), numpy_rng=np.random.get_state(),
        torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else [],
        baseline_sha256=baseline_sha, frozen_state_sha256=identity['frozen_state_sha256'])


def validate_checkpoint(saved, identity, model, baseline_sha):
    require(saved.get('schema') == SCHEMA and saved.get('identity') == identity, 'Response checkpoint identity changed')
    history, epoch = saved['history'], saved['completed_epoch']
    require(type(epoch) is int and 0 <= epoch <= identity['recipe']['max_epochs'] and len(history) == epoch+1
            and [row['epoch'] for row in history] == list(range(epoch+1)), 'Invalid committed response history')
    chosen = min(range(len(history)), key=lambda e: history[e]['validation']['on']['mae_pooled'])
    require(saved['best_epoch'] == chosen and saved['best_metric'] == history[chosen]['validation']['on']['mae_pooled'],
            'Best response selection disagrees with committed history')
    require(saved['baseline_sha256'] == baseline_sha and saved['frozen_state_sha256'] == identity['frozen_state_sha256'],
            'Response baseline certificate changed')
    model.assert_backbone_frozen(saved['frozen_state_sha256'])
    order = np.asarray(saved['train_order'])
    if epoch:
        require(hashlib.sha256(order.tobytes()).hexdigest() == history[-1]['training']['train_order_sha256'],
                'Saved response train order differs from committed digest')
    else:
        require(np.array_equal(order, np.arange(len(order), dtype=np.int64)), 'Epoch-zero train order is not canonical')
    spec = model.response.state_dict()
    for key in ('response_state', 'best_response_state'):
        state = saved[key]
        require(state.keys() == spec.keys() and all(state[name].shape == value.shape and state[name].dtype == value.dtype
            and bool(torch.isfinite(state[name]).all()) for name, value in spec.items()), 'Invalid saved response tensors')


def restore(saved, model, optimizer, data):
    model.response.load_state_dict(saved['response_state'], strict=True)
    optimizer.load_state_dict(saved['optimizer_state'])
    verify_optimizer(model, optimizer)
    data.loaders['train_loader'].set_order(saved['train_order'])
    np.random.set_state(saved['numpy_rng'])
    torch.set_rng_state(saved['torch_rng'])
    if next(model.parameters()).device.type == 'cuda':
        torch.cuda.set_rng_state_all(saved['cuda_rng'])


def write_archive(path, **arrays):
    path = Path(path)
    temporary = path.with_name(path.stem+'.partial.npz')
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def fit(model, base, data, source_saved, identity, root, target_epoch, recipe, *, protected_paths=None):
    """Commit complete epochs atomically; last.pt is the continuation authority."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    require(1 <= target_epoch <= recipe['max_epochs'], 'Target exceeds fixed response pilot budget')
    if protected_paths:
        probe.require_preserved(identity['source_artifacts_sha256'], protected_paths)
    path = root/'last.pt'
    if not path.is_file():
        require(tensor_state_hash(model.response.state_dict()) == identity['response_initial_sha256'],
                'Fresh response experiment requires canonical initialization')
        data.loaders['train_loader'].set_order(np.arange(data.loaders['train_loader'].size))
        set_seed(identity['seed'])
    frozen_hash = identity['frozen_state_sha256']
    model.assert_backbone_frozen(frozen_hash)
    baseline = response_baseline(base, data, source_saved, recipe)
    hard = baseline['fixed_hard_indices']
    archive = root/'baseline_predictions.npz'
    if path.is_file():
        require(archive.is_file(), 'Committed response run is missing baseline predictions')
        with np.load(archive, allow_pickle=False) as previous:
            require(all(np.array_equal(previous[key], baseline[key]) for key in BASELINE_KEYS), 'Stored MIN baseline replay differs')
    else:
        write_archive(archive, **{key: baseline[key] for key in BASELINE_KEYS})
    baseline_sha = sha256(archive)
    optimizer = optimizer_for(model, recipe)
    if path.is_file():
        saved = torch.load(path, map_location='cpu', weights_only=False)
        validate_checkpoint(saved, identity, model, baseline_sha)
        frozen.repair_history(root, saved)
        restore(saved, model, optimizer, data)
        replay, _ = evaluate_pair(model, data, baseline, frozen_hash, hard)
        require(replay == saved['history'][-1]['validation'], 'Last response replay differs; continuation blocked')
        print(f'Resuming committed response epoch {saved["completed_epoch"]}', flush=True)
    else:
        initial, on = evaluate_pair(model, data, baseline, frozen_hash, hard)
        require(np.array_equal(on, baseline['prediction']), 'Zero response differs from source MIN')
        epoch0 = dict(epoch=0, validation=initial, global_updates=0, optimizer_updates=0,
                      full_supervised_horizons=12, frozen_state_sha256=frozen_hash)
        saved = capture(model, optimizer, data, identity, [epoch0], model.response.state_dict(), 0,
                        initial['on']['mae_pooled'], baseline_sha)
        legacy.write_json(root/'baseline.json', dict(source_best_epoch=baseline['source_best_epoch'],
            saved_pooled_replay=baseline['saved_pooled_replay'], mask_value=baseline['mask_value'],
            original_to_frozen_normalized_abs_max=baseline['original_to_frozen_normalized_abs_max'],
            batch_partition=baseline['batch_partition'], zero_ON_OFF_exactly_match_frozen_MIN=True,
            original_source_metrics_replay_passed=True, validation=initial))
        save_checkpoint(path, saved)
        legacy.write_json(root/'history.json', saved['history'])
        print('FROZEN_MIN_RESPONSE_EPOCH_ZERO_REPLAY_PASS; complete response branch may update', flush=True)
    history, best_epoch, best_metric = saved['history'], saved['best_epoch'], saved['best_metric']
    best_state, start_epoch = saved['best_response_state'], saved['completed_epoch']
    for epoch in range(start_epoch+1, target_epoch+1):
        start = time.perf_counter()
        training = train_epoch(model, data, optimizer, recipe, frozen_hash)
        validation, _ = evaluate_pair(model, data, baseline, frozen_hash, hard)
        metric = validation['on']['mae_pooled']
        if metric < best_metric:
            best_metric, best_epoch, best_state = metric, epoch, cpu_state(model.response.state_dict())
        row = dict(epoch=epoch, training=training, validation=validation,
            global_updates=history[-1]['global_updates']+training['batches'],
            optimizer_updates=history[-1]['optimizer_updates']+training['optimizer_steps'],
            full_supervised_horizons=12, frozen_state_sha256=frozen_hash, seconds=time.perf_counter()-start)
        history.append(row)
        saved = capture(model, optimizer, data, identity, history, best_state, best_epoch, best_metric, baseline_sha)
        save_checkpoint(path, saved)
        legacy.write_json(root/'history.json', history)
        print(json.dumps(dict(arm=identity['arm'], epoch=epoch, validation_mae=metric, best_epoch=best_epoch,
            best_mae=best_metric, off_minus_on_mae=validation['mae_gain_pooled'],
            frozen_backbone_preserved=True, seconds=row['seconds']), allow_nan=False), flush=True)
        gc.collect()
    current = cpu_state(model.response.state_dict())
    model.response.load_state_dict(best_state, strict=True)
    try:
        best_report, best_predictions = evaluate_pair(model, data, baseline, frozen_hash, hard)
    finally:
        model.response.load_state_dict(current, strict=True)
    require(best_report == history[best_epoch]['validation'], 'Selected response replay differs')
    model.assert_backbone_frozen(frozen_hash)
    if protected_paths:
        legacy.write_json(root/'source_preservation.json', probe.require_preserved(identity['source_artifacts_sha256'], protected_paths))
    write_archive(root/'best_predictions.npz', on=best_predictions, off=baseline['prediction'],
        **{key: baseline[key] for key in BASELINE_KEYS if key != 'prediction'})
    baseline_metric = history[0]['validation']['on']['mae_pooled']
    status = ('MIN_RESPONSE_ENGINEERING_PASS' if identity['engineering_check'] else
              'MIN_RESPONSE_PILOT_COMPLETE' if saved['completed_epoch'] == recipe['max_epochs'] else
              'MIN_RESPONSE_PAUSED_AT_EPOCH_BOUNDARY')
    summary = dict(schema=SCHEMA, status=status, arm=identity['arm'], response_mode=identity['arm'],
        training_stage=recipe['training_stage'], final_joint_model=False,
        completed_epoch=saved['completed_epoch'], requested_epoch=target_epoch, pilot_max_epochs=recipe['max_epochs'],
        source_best_epoch=baseline['source_best_epoch'], best_epoch=best_epoch, best_response_epoch=best_epoch,
        best_mae=best_metric, baseline_mae=baseline_metric, baseline_minus_best_mae=baseline_metric-best_metric,
        best_metrics=best_report['on'], best_validation=best_report, baseline_metrics=history[0]['validation']['on'],
        selection_includes_epoch_zero=True, trainable_parameters=sum(p.numel() for p in model.response.parameters() if p.requires_grad),
        frozen_parameters=sum(p.numel() for p in model.backbone.parameters()),
        global_updates=history[-1]['global_updates'], optimizer_updates=history[-1]['optimizer_updates'],
        optimizer_updates_this_invocation=history[-1]['optimizer_updates']-history[start_epoch]['optimizer_updates'],
        frozen_state_sha256=frozen_hash, frozen_backbone_preserved=True, off_exactly_matches_source=True,
        source_artifacts_preserved=bool(protected_paths), engineering_only=identity['engineering_check'],
        baseline_predictions_sha256=sha256(archive), best_predictions_sha256=sha256(root/'best_predictions.npz'),
        test_evaluated=False, predictive_gain_claim=False,
        interpretation='conditional_offline_response_increment_not_real_capacity_or_causal_effect', output_dir=str(root))
    legacy.write_json(root/'summary.json', summary)
    return summary


def child_command(args, arm):
    command = [sys.executable, '-u', str(Path(__file__).resolve()), '--source-run', str(args.source_run),
        '--run-dir', str(args.run_dir), '--data-dir', str(args.data_dir), '--graph-dir', str(args.graph_dir),
        '--source-protocol', str(args.source_protocol), '--recipe', str(args.recipe), '--epochs', str(args.epochs),
        '--seed', str(args.seed), '--device', args.device, '--arms', *args.arms, '--worker', arm]
    if args.check:
        command.append('--check')
    return command


def run_worker(args, arm, session):
    with (session/(arm+'.log')).open('w', encoding='utf-8') as stream:
        process = subprocess.Popen(child_command(args, arm), cwd=REPO,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8',
            env=dict(os.environ, **{WORKER_ENV: '1', 'PYTHONUNBUFFERED': '1', 'PYTHONIOENCODING': 'utf-8'}))
        try:
            for line in process.stdout:
                print(line, end='', flush=True)
                stream.write(line)
                stream.flush()
            code = process.wait()
        except BaseException:
            if process.poll() is None:
                try:
                    process.terminate()
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
            legacy.write_json(session/(arm+'_process_result.json'), dict(exit_code=process.returncode, launcher_interrupted=True))
            raise
        finally:
            process.stdout.close()
    legacy.write_json(session/(arm+'_process_result.json'), dict(exit_code=code, launcher_interrupted=False))
    return code


def verify_arm_identity(root, identity):
    path = root/'identity.json'
    if path.is_file():
        require(legacy.read_json(path) == identity, 'Response arm identity changed')
    else:
        require({p.name for p in root.iterdir()} <= {'active.lock'}, 'Use a fresh response arm directory')
        legacy.write_json(path, identity)


def train_worker(args, identity, logger, recipe):
    root = args.run_dir/args.worker
    root.mkdir(exist_ok=True)
    with legacy.run_lock(root):
        source_identity, saved, data, base = load_source(args, logger)
        paths = source_paths(args, source_identity)
        require(probe.snapshot_files(paths) == identity['source_artifacts_sha256'], 'Protected input changed before worker')
        set_seed(args.seed)
        model, certificate = build_response(base, args.worker, recipe)
        require(certificate == identity['initialization']['arms'][args.worker], 'Worker initialization differs from paired preflight')
        current_identity = arm_identity(identity, args.worker)
        verify_arm_identity(root, current_identity)
        result = fit(model, base, data, saved, current_identity, root, args.epochs, recipe, protected_paths=paths)
        print(json.dumps({key: result[key] for key in ('status', 'arm', 'completed_epoch', 'best_epoch', 'best_mae',
            'baseline_minus_best_mae', 'trainable_parameters', 'optimizer_updates_this_invocation',
            'frozen_backbone_preserved', 'test_evaluated')}, allow_nan=False), flush=True)
        print('Saved arm summary: '+str(root/'summary.json'), flush=True)
        return result


def comparison(args, identity):
    histories, summaries = {}, {}
    for arm in args.arms:
        path = args.run_dir/arm/'last.pt'
        if not path.is_file():
            continue
        saved = torch.load(path, map_location='cpu', weights_only=False)
        require(saved.get('schema') == SCHEMA and saved.get('identity') == arm_identity(identity, arm),
                'Response checkpoint identity changed during comparison')
        history = saved['history']
        require([row['epoch'] for row in history] == list(range(saved['completed_epoch']+1)), 'Incomplete committed response history')
        chosen = min(range(len(history)), key=lambda e: history[e]['validation']['on']['mae_pooled'])
        require(chosen == saved['best_epoch'] and saved['best_metric'] == history[chosen]['validation']['on']['mae_pooled'],
                'Response selection certificate differs')
        histories[arm] = history
        summaries[arm] = dict(completed_epoch=saved['completed_epoch'], best_epoch=saved['best_epoch'], best_mae=saved['best_metric'])
    common = min((len(rows)-1 for rows in histories.values()), default=0) if len(histories) == len(args.arms) else None
    contrasts = {}
    if common is not None and {'location', 'state'} <= histories.keys():
        location, state = histories['location'][:common+1], histories['state'][:common+1]
        require(all(location[e]['training']['train_order_sha256'] == state[e]['training']['train_order_sha256']
                    for e in range(1, common+1)), 'Paired response training sample orders differ')
        metric = lambda row: row['validation']['on']['mae_pooled']
        contrasts['location_vs_state'] = dict(common_epoch_mae_gain=metric(state[-1])-metric(location[-1]),
            common_prefix_best_mae_gain=min(map(metric, state))-min(map(metric, location)),
            candidate_common_last_five=[metric(row) for row in location[-5:]],
            control_common_last_five=[metric(row) for row in state[-5:]])
    complete = len(histories) == len(args.arms) and all(row['completed_epoch'] >= args.epochs for row in summaries.values())
    return dict(schema=SCHEMA, status='MIN_RESPONSE_PAIRED_STAGE_COMPLETE' if complete else 'MIN_RESPONSE_PAIRED_STAGE_PARTIAL',
        training_stage='frozen_diagnostic_stage1', final_joint_model=False,
        requested_epoch=args.epochs, common_epoch=common, runs=summaries, comparisons=contrasts,
        interpretation='shared_frozen_MIN_common_initialization_same_support_conditional_response_pair',
        predictive_gain_claim=False, test_evaluated=False)


def main(argv=None):
    args = parse_args(argv)
    require(args.source_run.is_dir() and (args.source_run/'A').is_dir(), 'Source MIN-I A run does not exist')
    source_identity = legacy.read_json(args.source_run/'identity.json')
    require(source_identity.get('schema') == original.SCHEMA, 'Expected original MIN-I source run')
    require(not args.check or source_identity['engineering_check'], '--check requires an engineering source')
    args.check = bool(source_identity['engineering_check'])
    args.threads = source_identity['threads']
    args.device = args.device or ('cuda:0' if source_identity['runtime']['device_type'] == 'cuda' else 'cpu')
    if args.run_dir is None:
        args.run_dir = REPO/'experiments/chronological_runs'/('contra_min_response_v1_'+
            ('check_' if args.check else '')+f'bs{source_identity["batch_size"]}_seed{args.seed}')
    require_independent_directory(args)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    require(device.type in ('cpu', 'cuda') and (device.type != 'cuda' or torch.cuda.is_available()), 'Requested device unavailable')
    probe.verify_runtime(source_identity, device)
    recipe = legacy.read_json(args.recipe)
    require(recipe == legacy.read_json(RECIPE) and recipe['max_epochs'] == 10 and recipe['supervised_horizons'] == 12
            and not recipe['curriculum'] and recipe['arms'] == list(ARMS), 'Response recipe changed')
    logger = logging.getLogger('min_response')
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler(sys.stdout))
    args.run_dir.mkdir(parents=True, exist_ok=True)
    internal_worker = args.worker and os.environ.get(WORKER_ENV) == '1'
    with ExitStack() as locks:
        if not internal_worker:
            for directory in (args.source_run, args.source_run/'A', args.run_dir):
                locks.enter_context(legacy.run_lock(directory))
        if args.worker:
            require((args.run_dir/'identity.json').is_file(), 'Worker requires prepared parent identity')
            identity = legacy.read_json(args.run_dir/'identity.json')
            require(identity['schema'] == SCHEMA and identity['arms'] == args.arms and identity['seed'] == args.seed
                    and identity['source_run'] == str(args.source_run) and identity['recipe'] == recipe,
                    'Worker request differs from prepared response experiment')
            train_worker(args, identity, logger, recipe)
            if not internal_worker:
                report = comparison(args, identity)
                legacy.write_json(args.run_dir/'comparison.json', report)
                legacy.write_json(args.run_dir/'latest_stage.json', report)
                write_diagnostics(args.run_dir, arms=args.arms)
                print(json.dumps(report, indent=2, allow_nan=False), flush=True)
            return
        for arm in args.arms:
            if (args.run_dir/arm).is_dir():
                with legacy.run_lock(args.run_dir/arm):
                    pass
        source_identity, saved, data, base = load_source(args, logger)
        paths = source_paths(args, source_identity)
        before = probe.snapshot_files(paths)
        initialization = preflight(args, source_identity, saved, data, base, recipe)
        identity = make_identity(args, source_identity, saved, initialization, recipe, paths)
        require(identity['source_artifacts_sha256'] == before, 'Protected input changed during response preflight')
        verify_identity(args.run_dir, identity, allow_new=True)
        legacy.write_json(args.run_dir/'data_audit.json', data.audit)
        session = args.run_dir/('session_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6])
        session.mkdir()
        legacy.write_json(session/'environment.json', legacy.environment(device))
        legacy.write_json(session/'request.json', dict(source_run=str(args.source_run), arms=args.arms, epochs=args.epochs,
            batch_size=source_identity['batch_size'], test_evaluated=False))
        print('Run directory: '+str(args.run_dir), flush=True)
        print('Session directory: '+str(session), flush=True)
        print('Initialization: '+initialization['status'], flush=True)
        del data, base, saved
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        results = {}
        try:
            for arm in args.arms:
                checkpoint = args.run_dir/arm/'last.pt'
                if checkpoint.is_file():
                    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
                    require(state.get('schema') == SCHEMA and state.get('identity') == arm_identity(identity, arm),
                            'Existing response checkpoint identity changed')
                    # A finished arm can need sidecar regeneration after an interrupted
                    # export, so only skip when its current summary is committed too.
                    summary_path = args.run_dir/arm/'summary.json'
                    summary = legacy.read_json(summary_path) if summary_path.is_file() else {}
                    if state['completed_epoch'] >= args.epochs and summary.get('completed_epoch') == state['completed_epoch']:
                        results[arm] = dict(exit_code=0, skipped_completed=True)
                        legacy.write_json(session/'process_results.json', results)
                        print(f'Skipping {arm}; committed epoch {state["completed_epoch"]} already meets target', flush=True)
                        continue
                print('Starting '+arm+' persistent response branch', flush=True)
                code = run_worker(args, arm, session)
                results[arm] = dict(exit_code=code, skipped_completed=False)
                legacy.write_json(session/'process_results.json', results)
                require(code == 0, 'Worker failed; committed checkpoints preserved. See '+str(session/(arm+'.log')))
        except BaseException as error:
            legacy.write_json(session/'failure.json', dict(error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc()))
            try:
                probe.require_preserved(before, paths)
                partial = comparison(args, identity)
                legacy.write_json(session/'comparison.json', partial)
                legacy.write_json(args.run_dir/'comparison.json', partial)
                write_diagnostics(args.run_dir, arms=args.arms)
            except Exception as diagnostic_error:
                legacy.write_json(session/'partial_diagnostics_error.json', dict(error_type=type(diagnostic_error).__name__, error=str(diagnostic_error)))
            raise
        probe.require_preserved(before, paths)
        report = comparison(args, identity)
        legacy.write_json(session/'comparison.json', report)
        legacy.write_json(args.run_dir/'comparison.json', report)
        legacy.write_json(args.run_dir/'latest_stage.json', report)
        write_diagnostics(args.run_dir, arms=args.arms)
        print(json.dumps(dict(report, output=str(session/'comparison.json')), indent=2, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
