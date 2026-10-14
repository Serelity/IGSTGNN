"""Stage 2: jointly fine-tune MIN and its complete conditional response.

State/location start from independent copies of the same source best MIN. Both
responses start at zero, without stage-one training. Source MIN is never trained.
The immutable source is the
training risk reference; current-checkpoint OFF is a separate path diagnostic.
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
from experiments.chronological import train_min_response as diagnostic
from experiments.chronological import train_min_frozen_budget as frozen
from experiments.chronological import train_min_event_budget as original
from experiments.chronological import probe_min_event_budget as probe
from experiments.chronological import train_internal_spatial as legacy
from experiments.chronological.min_response_metrics import paired_training_loss, summarize_response_pair
from experiments.chronological.min_response_joint_metrics import write_diagnostics
from experiments.chronological.smoke import set_seed, sha256
from experiments.chronological.train import cpu_state, save_checkpoint
from src.models.incident_spatial import require
from src.models.incident_spatial_residual import tensor_state_hash

HERE = Path(__file__).resolve().parent
RECIPE = HERE/'min_response_joint_v1.json'
SCHEMA = 'MIN_conditional_response_joint_pair_v1'
STAGE = 'joint_finetune_stage2'
ARMS = ('state', 'location')
WORKER_ENV = 'IGSTGNN_MIN_RESPONSE_JOINT_CHILD'
BASELINE_KEYS = diagnostic.BASELINE_KEYS


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
    parser.add_argument('--epochs', type=int, default=1, help='Absolute joint epoch, 1..10')
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--device', help='Source device type by default; GPU name is audit-only')
    parser.add_argument('--check', action='store_true', help='Engineering source only; never subsets formal validation')
    args = parser.parse_args(argv)
    require(len(set(args.arms)) == len(args.arms), 'Duplicate joint arms are not permitted')
    require(args.worker is None or args.worker in args.arms, 'Worker must belong to the recorded joint arm set')
    args.arms = [arm for arm in ARMS if arm in args.arms]
    require(1 <= args.epochs <= 10 and 0 <= args.seed < 2**32, 'Expected 1..10 epochs and a valid seed')
    for key in ('source_run', 'data_dir', 'graph_dir', 'source_protocol', 'recipe'):
        setattr(args, key, getattr(args, key).resolve())
    if args.run_dir is not None:
        args.run_dir = args.run_dir.resolve()
    return args


def experiment_sources():
    result = diagnostic.experiment_sources()
    names = ('experiments/chronological/train_min_response_joint.py',
             'experiments/chronological/min_response_joint_metrics.py',
             'experiments/chronological/run_min_response_joint.sh')
    result.update({name: sha256(REPO/name) for name in names})
    return result


def source_paths(args, identity):
    paths = diagnostic.source_paths(args, identity)
    paths.update({'experiment_code/'+name: REPO/name for name in experiment_sources()})
    paths['stage1_recipe_dependency'] = diagnostic.RECIPE
    require(all(path.is_file() for path in paths.values()), 'A protected joint experiment input is missing')
    return paths


def require_independent_directory(args):
    root, source = args.run_dir.resolve(), args.source_run.resolve()
    require(root != source and source not in root.parents and root not in source.parents,
            'Use an independent joint directory outside the source run')
    for ancestor in root.parents:
        require(not (ancestor/'identity.json').is_file(), 'Joint output is inside an existing experiment')
    if (root/'identity.json').is_file():
        identity = legacy.read_json(root/'identity.json')
        require(identity.get('schema') == SCHEMA and identity.get('training_stage') == STAGE,
                'Use a fresh joint run; frozen and legacy stages cannot be resumed here')


def build_joint(base, arm, recipe):
    require(arm in ARMS, 'Unknown joint arm')
    source_hash = tensor_state_hash(base.state_dict())
    model, certificate = diagnostic.build_response(base, arm, recipe)
    model.set_joint_training(True)
    certificate = dict(certificate, source_state_sha256=source_hash,
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        trainable_names=[name for name, p in model.named_parameters() if p.requires_grad],
        frozen_parameters=sum(p.numel() for p in model.parameters() if not p.requires_grad),
        backbone_parameters=sum(p.numel() for p in model.backbone.parameters()),
        optimizer_parameter_scope='joint_backbone_and_complete_response')
    for parameter in model.parameters():
        parameter.grad = None
    model.eval()
    certificate.update(training_stage=STAGE, arm=arm, model_initial_sha256=tensor_state_hash(model.state_dict()),
        initialization='same_source_MIN_best_zero_response_no_stage1_pretraining',
        source_reference_scope=recipe['reference_scope'], current_OFF_scope='joint_checkpoint_backbone_not_original_MIN')
    require(tensor_state_hash(base.state_dict()) == source_hash, 'Joint construction changed original source')
    return model, certificate


def assert_reference(base, expected_hash):
    require(all(not module.training for module in base.modules()), 'Immutable source reference must stay eval')
    require(all(parameter.grad is None for parameter in base.parameters()), 'Immutable source reference acquired gradients')
    require(tensor_state_hash(base.state_dict()) == expected_hash, 'Immutable source reference parameters or buffers changed')


def baseline_reference(base, data, source_saved, recipe):
    # Verify the author's original selected validation metrics and fixed freeze
    # roundoff tolerance first; joint optimization uses native flags thereafter.
    baseline = diagnostic.response_baseline(base, data, source_saved, recipe)
    base.requires_grad_(True).eval()
    values, targets, partition = [], [], []
    before, rng = tensor_state_hash(base.state_dict()), probe.rng_digest(next(base.parameters()).device)
    for batch in data.loaders['val_loader'].get_iterator():
        row = probe.forward_case(base, 'A', batch, data.scaler, 'A_ON')
        values.append(row['prediction'])
        targets.append(row['target'])
        partition.append(len(row['prediction']))
    prediction, target = np.concatenate(values), np.concatenate(targets)
    require(partition == baseline['batch_partition'] and np.array_equal(target, baseline['target']), 'Native source batch alignment changed')
    metrics, null = frozen.torch_metrics(prediction, target)
    expected = probe.best_row(source_saved)
    require(np.allclose([metrics[key] for key in ('mae', 'mape', 'rmse')],
        [expected['validation_'+key] for key in ('mae', 'mape', 'rmse')],
        atol=probe.REPLAY_ATOL, rtol=probe.REPLAY_RTOL), 'Native source MIN validation replay failed')
    require(probe.rng_digest(next(base.parameters()).device) == rng, 'Source replay changed RNG')
    assert_reference(base, before)
    baseline.update(prediction=prediction, target=target, valid=np.isfinite(target) & (target != null),
        mask_value=null, saved_pooled_replay=metrics,
        reference_scope='immutable_reference_no_optimizer_no_grad_native_dispatch')
    baseline['fixed_hard_indices'] = frozen.select_hard_windows(prediction, target, baseline['valid'],
        baseline['sample_indices'], fraction=recipe['fixed_hard_fraction'])
    return baseline


def forward_case(model, arm, batch, scaler, enabled):
    require(not any(module.training for module in model.modules()), 'Joint replay requires eval mode')
    device = next(model.parameters()).device
    before = probe.input_digest(batch)
    x = torch.as_tensor(batch['x_data'], dtype=torch.float32, device=device)
    y = torch.as_tensor(batch['y_data'], dtype=torch.float32, device=device)
    events = original.event_tensors(batch['event_data'], device)
    try:
        with torch.inference_mode():
            result = model(x, incident_data=events, report_enabled=enabled, return_details=True)
            prediction, mechanisms = result['prediction'], result['response']
            prediction = prediction*scaler.std.to(device)+scaler.mean.to(device)
            target = y*scaler.std.to(device)+scaler.mean.to(device)
        require(prediction.shape == target.shape and prediction.shape[1] == 12 and bool(torch.isfinite(prediction).all()),
                'Invalid full-horizon joint replay')
        return dict(prediction=prediction[..., 0].cpu().numpy(), target=target[..., 0].cpu().numpy(), mechanisms=mechanisms)
    finally:
        require(probe.input_digest(batch) == before, 'Joint replay changed an input payload')


def evaluate_pair(model, arm, data, baseline, hard):
    model.eval()
    before, rng = tensor_state_hash(model.state_dict()), probe.rng_digest(next(model.parameters()).device)
    on_values, off_values, partition, diagnostics, position = [], [], [], [], 0
    for batch in data.loaders['val_loader'].get_iterator():
        on = forward_case(model, arm, batch, data.scaler, True)
        off = forward_case(model, arm, batch, data.scaler, False)
        restored = forward_case(model, arm, batch, data.scaler, True)
        end = position+len(on['prediction'])
        require(all(np.array_equal(row['target'], baseline['target'][position:end]) for row in (on, off, restored)),
                'Joint validation target changed')
        require(np.array_equal(on['prediction'], restored['prediction']) and on['mechanisms'] == restored['mechanisms'],
                'Joint ON restoration failed')
        on_values.append(on['prediction'])
        off_values.append(off['prediction'])
        partition.append(end-position)
        diagnostics.append(on['mechanisms'])
        position = end
    require(position == len(baseline['prediction']) and partition == baseline['batch_partition'], 'Joint validation batches changed')
    require(tensor_state_hash(model.state_dict()) == before and probe.rng_digest(next(model.parameters()).device) == rng,
            'Joint replay changed model tensors or RNG')
    on, current_off = np.concatenate(on_values), np.concatenate(off_values)
    report = summarize_response_pair(on, baseline['prediction'], baseline['target'], baseline['valid'], hard,
                                     history_low_mask=baseline['history_low_mask'])
    report.update(current_off=summarize_response_pair(on, current_off, baseline['target'], baseline['valid'], hard,
        history_low_mask=baseline['history_low_mask']), on_exactly_restored=True,
        current_OFF_reference='same_joint_checkpoint_backbone', original_OFF_reference=baseline['reference_scope'],
        current_OFF_exactly_matches_original=bool(np.array_equal(current_off, baseline['prediction'])),
        validation_rows=position, batch_partition=partition,
        mechanisms=dict(scope='dimensionless_latent_response_not_real_capacity', batches=diagnostics))
    report['scope'] = 'joint_candidate_vs_immutable_original_MIN_conditional_offline'
    report['current_off']['scope'] = 'same_joint_checkpoint_response_path_sensitivity_conditional_offline'
    return report, on, current_off


def preflight(args, source_identity, saved, data, base, recipe):
    baseline = baseline_reference(base, data, saved, recipe)
    certificates = {}
    for arm in ARMS:
        set_seed(args.seed)
        model, certificate = build_joint(base, arm, recipe)
        report, on, off = evaluate_pair(model, arm, data, baseline, baseline['fixed_hard_indices'])
        require(np.array_equal(on, baseline['prediction']) and np.array_equal(off, baseline['prediction']),
                'Initial joint ON/OFF differs from native source MIN')
        certificates[arm] = certificate
        del model
    require(certificates['state']['common_initial_sha256'] == certificates['location']['common_initial_sha256'],
            'Common joint response initialization differs')
    require(len({c['source_state_sha256'] for c in certificates.values()}) == 1, 'Joint source initialization differs')
    assert_reference(base, tensor_state_hash(saved['best_model_state']))
    return dict(status='MIN_RESPONSE_JOINT_INITIALIZATION_PASS', training_stage=STAGE, arms=certificates,
        source_best_epoch=probe.best_row(saved)['epoch'], baseline_sha256=diagnostic.baseline_digest(baseline),
        validation_rows=len(baseline['prediction']), batch_partition=baseline['batch_partition'],
        all_initial_ON_OFF_exactly_match_native_source_MIN=True, no_stage1_pretraining=True,
        common_response_initialization_exact=True,
        trainable_parameters={arm: c['trainable_parameters'] for arm, c in certificates.items()}, test_evaluated=False)


def optimizer_for(model, arm, recipe):
    require(arm in ARMS, 'Unknown joint optimizer arm')
    groups = [dict(params=list(model.backbone.parameters()), lr=recipe['backbone_learning_rate'], name='backbone'),
              dict(params=list(model.response.parameters()), lr=recipe['response_learning_rate'], name='response')]
    optimizer = torch.optim.Adam(groups, weight_decay=recipe['weight_decay'], eps=recipe['adam_eps'])
    verify_optimizer(model, arm, optimizer, recipe)
    return optimizer


def verify_optimizer(model, arm, optimizer, recipe):
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    actual = [id(p) for group in optimizer.param_groups for p in group['params']]
    require(expected and len(actual) == len(expected) and set(actual) == expected, 'Joint optimizer parameter set differs')
    require(arm in ARMS, 'Unknown joint optimizer arm')
    expected_groups = [('backbone', recipe['backbone_learning_rate']), ('response', recipe['response_learning_rate'])]
    require([(g.get('name'), g['lr']) for g in optimizer.param_groups] == expected_groups, 'Joint learning-rate groups differ')


def train_epoch(model, arm, base, data, optimizer, recipe, source_hash):
    model.train()
    base.eval()
    assert_reference(base, source_hash)
    verify_optimizer(model, arm, optimizer, recipe)
    loader, device = data.loaders['train_loader'], next(model.parameters()).device
    loader.shuffle()
    losses, main_losses, risks, gradient_max = [], [], [], 0.
    backbone_before = tensor_state_hash(model.backbone.state_dict())
    response_before = tensor_state_hash(model.response.state_dict())
    active_response, active_condition = 0, 0
    for number, batch in enumerate(loader.get_iterator(), 1):
        optimizer.zero_grad(set_to_none=True)
        digest = probe.input_digest(batch)
        x = torch.as_tensor(batch['x_data'], dtype=torch.float32, device=device)
        y = torch.as_tensor(batch['y_data'], dtype=torch.float32, device=device)
        events = original.event_tensors(batch['event_data'], device)
        with torch.no_grad():
            original_off = base(x).detach()
        prediction = model(x, incident_data=events, report_enabled=True)
        loss, stats = paired_training_loss(prediction, original_off, y, x, data.scaler.mean, data.scaler.std,
            low_history_threshold=recipe['low_history_threshold'], risk_weight=recipe['risk_weight'])
        require(bool(torch.isfinite(loss)) and loss.requires_grad, 'Invalid or disconnected joint loss')
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        require(gradients and all(bool(torch.isfinite(g).all()) for g in gradients), 'Missing/non-finite joint gradients')
        gradient_max = max(gradient_max, max(float(g.abs().max()) for g in gradients))
        active_response += int(any(p.grad is not None and bool((p.grad != 0).any()) for p in model.response.parameters()))
        active_condition += int(any(p.grad is not None and bool((p.grad != 0).any())
                                    for p in model.response.condition_output.parameters()))
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], recipe['clip_grad_norm'])
        optimizer.step()
        require(probe.input_digest(batch) == digest, 'Joint training changed input payload')
        require(all(p.grad is None for p in base.parameters()), 'Immutable source acquired training gradients')
        losses.append(float(loss.detach().cpu()))
        main_losses.append(stats['mae_on'])
        risks.append(stats['risk_penalty'])
        if number % 16 == 0 or number == loader.num_batch:
            print(f'Joint {arm}: {number}/{loader.num_batch} batches; full 12-step loss', flush=True)
    optimizer.zero_grad(set_to_none=True)
    assert_reference(base, source_hash)
    return dict(loss_batch_mean=float(np.mean(losses)), main_mae_batch_mean=float(np.mean(main_losses)),
        risk_penalty_batch_mean=float(np.mean(risks)), batches=loader.num_batch, optimizer_steps=loader.num_batch,
        full_supervised_horizons=12, joint_gradient_abs_max=gradient_max,
        backbone_state_changed=tensor_state_hash(model.backbone.state_dict()) != backbone_before,
        response_state_changed=tensor_state_hash(model.response.state_dict()) != response_before,
        batches_with_nonzero_response_gradient=active_response,
        batches_with_nonzero_condition_gradient=active_condition,
        train_order_sha256=hashlib.sha256(loader.order.tobytes()).hexdigest())


def capture(model, optimizer, data, identity, history, best_state, best_epoch, best_metric, baseline_sha):
    device = next(model.parameters()).device
    return dict(schema=SCHEMA, identity=identity, completed_epoch=history[-1]['epoch'], history=history,
        model_state=cpu_state(model.state_dict()), best_model_state=cpu_state(best_state),
        best_epoch=best_epoch, best_metric=best_metric, optimizer_state=optimizer.state_dict(),
        train_order=data.loaders['train_loader'].order.copy(), numpy_rng=np.random.get_state(),
        torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else [],
        baseline_sha256=baseline_sha, source_state_sha256=identity['source_state_sha256'])


def validate_checkpoint(saved, identity, model, baseline_sha):
    require(saved.get('schema') == SCHEMA and saved.get('identity') == identity, 'Joint checkpoint identity changed')
    require(identity['training_stage'] == STAGE, 'Checkpoint is not a joint training stage')
    history, epoch = saved['history'], saved['completed_epoch']
    require(type(epoch) is int and 0 <= epoch <= identity['recipe']['max_epochs'] and len(history) == epoch+1
            and [row['epoch'] for row in history] == list(range(epoch+1)), 'Invalid committed joint history')
    chosen = min(range(len(history)), key=lambda e: history[e]['validation']['on']['mae_pooled'])
    require(saved['best_epoch'] == chosen and saved['best_metric'] == history[chosen]['validation']['on']['mae_pooled'],
            'Best joint selection disagrees with history')
    require(saved['baseline_sha256'] == baseline_sha and saved['source_state_sha256'] == identity['source_state_sha256'],
            'Joint source reference certificate changed')
    order = np.asarray(saved['train_order'])
    if epoch:
        require(hashlib.sha256(order.tobytes()).hexdigest() == history[-1]['training']['train_order_sha256'],
                'Saved joint sample order differs')
    else:
        require(np.array_equal(order, np.arange(len(order), dtype=np.int64)), 'Joint epoch-zero order is not canonical')
    spec = model.state_dict()
    for key in ('model_state', 'best_model_state'):
        state = saved[key]
        require(state.keys() == spec.keys() and all(state[name].shape == value.shape and state[name].dtype == value.dtype
            and bool(torch.isfinite(state[name]).all()) for name, value in spec.items()), 'Invalid saved joint model tensors')


def restore(saved, model, optimizer, data, arm, recipe):
    model.load_state_dict(saved['model_state'], strict=True)
    optimizer.load_state_dict(saved['optimizer_state'])
    verify_optimizer(model, arm, optimizer, recipe)
    data.loaders['train_loader'].set_order(saved['train_order'])
    np.random.set_state(saved['numpy_rng'])
    torch.set_rng_state(saved['torch_rng'])
    if next(model.parameters()).device.type == 'cuda':
        torch.cuda.set_rng_state_all(saved['cuda_rng'])


def fit(model, base, data, source_saved, identity, root, target_epoch, recipe, *, protected_paths=None):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    require(identity['training_stage'] == STAGE and 1 <= target_epoch <= recipe['max_epochs'], 'Invalid joint stage or target')
    if protected_paths:
        probe.require_preserved(identity['source_artifacts_sha256'], protected_paths)
    arm, path = identity['arm'], root/'last.pt'
    if not path.is_file():
        require(tensor_state_hash(model.state_dict()) == identity['model_initial_sha256'], 'Fresh joint run requires canonical initialization')
        data.loaders['train_loader'].set_order(np.arange(data.loaders['train_loader'].size))
        set_seed(identity['seed'])
    source_hash = identity['source_state_sha256']
    baseline = baseline_reference(base, data, source_saved, recipe)
    assert_reference(base, source_hash)
    hard, archive = baseline['fixed_hard_indices'], root/'baseline_predictions.npz'
    if path.is_file():
        require(archive.is_file(), 'Committed joint run is missing its fixed source baseline')
        with np.load(archive, allow_pickle=False) as previous:
            require(all(np.array_equal(previous[key], baseline[key]) for key in BASELINE_KEYS), 'Stored joint source replay differs')
    else:
        diagnostic.write_archive(archive, **{key: baseline[key] for key in BASELINE_KEYS})
    baseline_sha = sha256(archive)
    optimizer = optimizer_for(model, arm, recipe)
    if path.is_file():
        saved = torch.load(path, map_location='cpu', weights_only=False)
        validate_checkpoint(saved, identity, model, baseline_sha)
        frozen.repair_history(root, saved)
        restore(saved, model, optimizer, data, arm, recipe)
        report, _, _ = evaluate_pair(model, arm, data, baseline, hard)
        require(report == saved['history'][-1]['validation'], 'Last joint replay differs; continuation blocked')
        print(f'Resuming committed joint {arm} epoch {saved["completed_epoch"]}', flush=True)
    else:
        report, on, current_off = evaluate_pair(model, arm, data, baseline, hard)
        require(np.array_equal(on, baseline['prediction']) and np.array_equal(current_off, baseline['prediction']),
                'Joint epoch zero ON/OFF differs from native source MIN')
        epoch0 = dict(epoch=0, validation=report, global_updates=0, optimizer_updates=0, full_supervised_horizons=12)
        saved = capture(model, optimizer, data, identity, [epoch0], model.state_dict(), 0, report['on']['mae_pooled'], baseline_sha)
        legacy.write_json(root/'baseline.json', dict(source_best_epoch=baseline['source_best_epoch'],
            reference_scope=baseline['reference_scope'], saved_pooled_replay=baseline['saved_pooled_replay'],
            mask_value=baseline['mask_value'], batch_partition=baseline['batch_partition'],
            original_source_metrics_replay_passed=True, zero_ON_OFF_exactly_match_native_MIN=True, validation=report))
        save_checkpoint(path, saved)
        legacy.write_json(root/'history.json', saved['history'])
        print('MIN_RESPONSE_JOINT_EPOCH_ZERO_REPLAY_PASS; backbone and response may update', flush=True)
    history, best_epoch, best_metric = saved['history'], saved['best_epoch'], saved['best_metric']
    best_state, start_epoch = saved['best_model_state'], saved['completed_epoch']
    for epoch in range(start_epoch+1, target_epoch+1):
        start = time.perf_counter()
        training = train_epoch(model, arm, base, data, optimizer, recipe, source_hash)
        validation, _, _ = evaluate_pair(model, arm, data, baseline, hard)
        metric = validation['on']['mae_pooled']
        if metric < best_metric:
            best_metric, best_epoch, best_state = metric, epoch, cpu_state(model.state_dict())
        row = dict(epoch=epoch, training=training, validation=validation,
            global_updates=history[-1]['global_updates']+training['batches'],
            optimizer_updates=history[-1]['optimizer_updates']+training['optimizer_steps'],
            full_supervised_horizons=12, seconds=time.perf_counter()-start)
        history.append(row)
        saved = capture(model, optimizer, data, identity, history, best_state, best_epoch, best_metric, baseline_sha)
        save_checkpoint(path, saved)
        legacy.write_json(root/'history.json', history)
        print(json.dumps(dict(arm=arm, epoch=epoch, validation_mae=metric, best_epoch=best_epoch, best_mae=best_metric,
            original_MIN_minus_on_mae=validation['mae_gain_pooled'], current_OFF_minus_ON_mae=validation['current_off']['mae_gain_pooled'],
            source_preserved=True, seconds=row['seconds']), allow_nan=False), flush=True)
        gc.collect()
    current = cpu_state(model.state_dict())
    model.load_state_dict(best_state, strict=True)
    try:
        best_report, on, current_off = evaluate_pair(model, arm, data, baseline, hard)
    finally:
        model.load_state_dict(current, strict=True)
    require(best_report == history[best_epoch]['validation'], 'Selected joint model replay differs')
    assert_reference(base, source_hash)
    if protected_paths:
        legacy.write_json(root/'source_preservation.json', probe.require_preserved(identity['source_artifacts_sha256'], protected_paths))
    diagnostic.write_archive(root/'best_predictions.npz', on=on, current_off=current_off,
        original_off=baseline['prediction'], off=baseline['prediction'],
        **{key: baseline[key] for key in BASELINE_KEYS if key != 'prediction'})
    baseline_metric = history[0]['validation']['on']['mae_pooled']
    status = ('MIN_RESPONSE_JOINT_ENGINEERING_PASS' if identity['engineering_check'] else
              'MIN_RESPONSE_JOINT_PILOT_COMPLETE' if saved['completed_epoch'] == recipe['max_epochs'] else
              'MIN_RESPONSE_JOINT_PAUSED_AT_EPOCH_BOUNDARY')
    summary = dict(schema=SCHEMA, training_stage=STAGE, status=status, arm=arm,
        completed_epoch=saved['completed_epoch'], requested_epoch=target_epoch, pilot_max_epochs=recipe['max_epochs'],
        best_epoch=best_epoch, best_mae=best_metric, baseline_mae=baseline_metric,
        baseline_minus_best_mae=baseline_metric-best_metric, best_metrics=best_report['on'], best_validation=best_report,
        baseline_metrics=history[0]['validation']['on'], selection_includes_epoch_zero=True,
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        backbone_trainable=True, response_trainable=True, no_stage1_pretraining=True,
        source_MIN_training_updates=0,
        global_updates=history[-1]['global_updates'], optimizer_updates=history[-1]['optimizer_updates'],
        optimizer_updates_this_invocation=history[-1]['optimizer_updates']-history[start_epoch]['optimizer_updates'],
        source_state_sha256=source_hash, source_reference_preserved=True, source_artifacts_preserved=bool(protected_paths),
        current_OFF_exactly_matches_original=best_report['current_OFF_exactly_matches_original'],
        baseline_predictions_sha256=sha256(archive), best_predictions_sha256=sha256(root/'best_predictions.npz'),
        engineering_only=identity['engineering_check'], test_evaluated=False, predictive_gain_claim=False,
        interpretation='candidate_plus_extra_finetuning_vs_existing_MIN_not_pure_module_gain', output_dir=str(root))
    legacy.write_json(root/'summary.json', summary)
    return summary


def make_identity(args, source_identity, saved, initialization, recipe, paths):
    return dict(schema=SCHEMA, training_stage=STAGE, source_run=str(args.source_run), arms=args.arms,
        source_artifacts_sha256=probe.snapshot_files(paths), source_identity=source_identity,
        source_best_epoch=probe.best_row(saved)['epoch'], source_state_sha256=tensor_state_hash(saved['best_model_state']),
        initialization=initialization, recipe=recipe, recipe_sha256=sha256(args.recipe),
        experiment_sources_sha256=experiment_sources(), seed=args.seed, batch_size=source_identity['batch_size'],
        threads=source_identity['threads'], runtime=source_identity['runtime'], engineering_check=source_identity['engineering_check'],
        event_scope='published_archive_attributes_conditional_offline', capacity_scope=recipe['capacity_scope'],
        time_scope=recipe['time_scope'], no_stage1_pretraining=True, source_MIN_training_updates=0,
        test_payload_deserialized=True, test_evaluated=False)


def arm_identity(identity, arm):
    require(arm in identity['arms'], 'Arm is absent from prepared joint experiment')
    certificate = identity['initialization']['arms'][arm]
    return dict(identity, arm=arm, model_certificate=certificate,
                model_initial_sha256=certificate['model_initial_sha256'])


def verify_identity(root, identity, allow_new):
    if (root/'identity.json').is_file():
        require(legacy.read_json(root/'identity.json') == identity, 'Joint experiment identity changed; checkpoints preserved')
        require(legacy.read_json(root/'initialization.json') == identity['initialization'], 'Joint initialization sidecar changed')
    else:
        require(allow_new and {p.name for p in root.iterdir()} <= {'active.lock'}, 'Use a fresh independent joint run')
        legacy.write_json(root/'identity.json', identity)
        legacy.write_json(root/'initialization.json', identity['initialization'])


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
        process = subprocess.Popen(child_command(args, arm), cwd=REPO, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding='utf-8',
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


def train_worker(args, identity, logger, recipe):
    root = args.run_dir/args.worker
    root.mkdir(exist_ok=True)
    with legacy.run_lock(root):
        source_identity, saved, data, base = diagnostic.load_source(args, logger)
        paths = source_paths(args, source_identity)
        require(probe.snapshot_files(paths) == identity['source_artifacts_sha256'], 'Protected joint input changed before worker')
        set_seed(args.seed)
        model, certificate = build_joint(base, args.worker, recipe)
        require(certificate == identity['initialization']['arms'][args.worker], 'Joint worker initialization differs')
        current_identity = arm_identity(identity, args.worker)
        diagnostic.verify_arm_identity(root, current_identity)
        summary = fit(model, base, data, saved, current_identity, root, args.epochs, recipe, protected_paths=paths)
        print(json.dumps({key: summary[key] for key in ('status', 'arm', 'completed_epoch', 'best_epoch', 'best_mae',
            'baseline_minus_best_mae', 'trainable_parameters', 'optimizer_updates_this_invocation', 'test_evaluated')}, allow_nan=False), flush=True)
        return summary


def comparison(args, identity):
    histories, summaries = {}, {}
    for arm in args.arms:
        path = args.run_dir/arm/'last.pt'
        if not path.is_file():
            continue
        saved = torch.load(path, map_location='cpu', weights_only=False)
        require(saved.get('schema') == SCHEMA and saved.get('identity') == arm_identity(identity, arm), 'Joint checkpoint identity changed during comparison')
        history = saved['history']
        require([row['epoch'] for row in history] == list(range(saved['completed_epoch']+1)), 'Incomplete committed joint history')
        chosen = min(range(len(history)), key=lambda e: history[e]['validation']['on']['mae_pooled'])
        require(chosen == saved['best_epoch'] and saved['best_metric'] == history[chosen]['validation']['on']['mae_pooled'], 'Joint selection certificate differs')
        histories[arm] = history
        summaries[arm] = dict(completed_epoch=saved['completed_epoch'], best_epoch=saved['best_epoch'], best_mae=saved['best_metric'])
    common = min((len(rows)-1 for rows in histories.values()), default=0) if len(histories) == len(args.arms) else None
    contrasts = {}
    if common is not None:
        for candidate, control in (('location', 'state'),):
            if not {candidate, control} <= histories.keys():
                continue
            left, right = histories[candidate][:common+1], histories[control][:common+1]
            require(all(left[e]['training']['train_order_sha256'] == right[e]['training']['train_order_sha256']
                        for e in range(1, common+1)), 'Joint matched training sample orders differ')
            metric = lambda row: row['validation']['on']['mae_pooled']
            contrasts[candidate+'_vs_'+control] = dict(common_epoch_mae_gain=metric(right[-1])-metric(left[-1]),
                common_prefix_best_mae_gain=min(map(metric, right))-min(map(metric, left)),
                candidate_common_last_five=[metric(row) for row in left[-5:]],
                control_common_last_five=[metric(row) for row in right[-5:]])
    complete = len(histories) == len(args.arms) and all(row['completed_epoch'] >= args.epochs for row in summaries.values())
    return dict(schema=SCHEMA, training_stage=STAGE,
        status='MIN_RESPONSE_JOINT_PAIRED_STAGE_COMPLETE' if complete else 'MIN_RESPONSE_JOINT_PAIRED_STAGE_PARTIAL',
        requested_epoch=args.epochs, common_epoch=common, runs=summaries, comparisons=contrasts,
        interpretation='location_vs_same_support_state_matched_joint_budget_existing_MIN_reference_not_pure_module_gain',
        source_MIN_training_updates=0, predictive_gain_claim=False, test_evaluated=False)


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
        args.run_dir = REPO/'experiments/chronological_runs'/('contra_min_response_joint_v1_'+
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
    require(recipe == legacy.read_json(RECIPE) and recipe['training_stage'] == STAGE and recipe['max_epochs'] == 10
            and recipe['supervised_horizons'] == 12 and not recipe['curriculum'] and recipe['arms'] == list(ARMS), 'Joint recipe changed')
    logger = logging.getLogger('min_response_joint')
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
            require((args.run_dir/'identity.json').is_file(), 'Joint worker requires prepared parent identity')
            identity = legacy.read_json(args.run_dir/'identity.json')
            require(identity['schema'] == SCHEMA and identity['training_stage'] == STAGE and identity['arms'] == args.arms
                and identity['seed'] == args.seed and identity['source_run'] == str(args.source_run) and identity['recipe'] == recipe,
                'Worker request differs from prepared joint experiment')
            train_worker(args, identity, logger, recipe)
            if not internal_worker:
                result = comparison(args, identity)
                legacy.write_json(args.run_dir/'comparison.json', result)
                legacy.write_json(args.run_dir/'latest_stage.json', result)
                write_diagnostics(args.run_dir, arms=args.arms)
                print(json.dumps(result, indent=2, allow_nan=False), flush=True)
            return
        for arm in args.arms:
            if (args.run_dir/arm).is_dir():
                with legacy.run_lock(args.run_dir/arm):
                    pass
        source_identity, saved, data, base = diagnostic.load_source(args, logger)
        paths = source_paths(args, source_identity)
        before = probe.snapshot_files(paths)
        initialization = preflight(args, source_identity, saved, data, base, recipe)
        identity = make_identity(args, source_identity, saved, initialization, recipe, paths)
        require(identity['source_artifacts_sha256'] == before, 'Protected input changed during joint preflight')
        verify_identity(args.run_dir, identity, allow_new=True)
        legacy.write_json(args.run_dir/'data_audit.json', data.audit)
        session = args.run_dir/('session_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6])
        session.mkdir()
        legacy.write_json(session/'environment.json', legacy.environment(device))
        legacy.write_json(session/'request.json', dict(source_run=str(args.source_run), arms=args.arms, epochs=args.epochs,
            batch_size=source_identity['batch_size'], training_stage=STAGE, test_evaluated=False))
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
                    require(state.get('schema') == SCHEMA and state.get('identity') == arm_identity(identity, arm), 'Existing joint identity changed')
                    summary_path = args.run_dir/arm/'summary.json'
                    summary = legacy.read_json(summary_path) if summary_path.is_file() else {}
                    if state['completed_epoch'] >= args.epochs and summary.get('completed_epoch') == state['completed_epoch']:
                        results[arm] = dict(exit_code=0, skipped_completed=True)
                        legacy.write_json(session/'process_results.json', results)
                        print(f'Skipping {arm}; committed epoch {state["completed_epoch"]} already meets target', flush=True)
                        continue
                print('Starting joint '+arm, flush=True)
                code = run_worker(args, arm, session)
                results[arm] = dict(exit_code=code, skipped_completed=False)
                legacy.write_json(session/'process_results.json', results)
                require(code == 0, 'Joint worker failed; committed checkpoints preserved. See '+str(session/(arm+'.log')))
        except BaseException as error:
            legacy.write_json(session/'failure.json', dict(error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc()))
            try:
                probe.require_preserved(before, paths)
                result = comparison(args, identity)
                legacy.write_json(session/'comparison.json', result)
                legacy.write_json(args.run_dir/'comparison.json', result)
                write_diagnostics(args.run_dir, arms=args.arms)
            except Exception as diagnostic_error:
                legacy.write_json(session/'partial_diagnostics_error.json', dict(error_type=type(diagnostic_error).__name__, error=str(diagnostic_error)))
            raise
        probe.require_preserved(before, paths)
        result = comparison(args, identity)
        legacy.write_json(session/'comparison.json', result)
        legacy.write_json(args.run_dir/'comparison.json', result)
        legacy.write_json(args.run_dir/'latest_stage.json', result)
        write_diagnostics(args.run_dir, arms=args.arms)
        print(json.dumps(dict(result, output=str(session/'comparison.json')), indent=2, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
