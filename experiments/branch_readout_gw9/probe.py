"""Three-epoch frozen-R readout screen; streaming features and paired resume.

The two heads share each actual backbone forward and training batch. The frozen
reference receives zero updates and is not called a same-budget trained control.
No probe weights are imported into the optional, separately invoked joint stage.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
from time import perf_counter
import uuid

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from torch import nn

from experiments.branch_readout_gw9 import train
from experiments.branch_readout_gw9.model import DESIGN, PROBE_ARMS, ReadoutHead
from experiments.branch_readout_gw9.reference import (
    certify, recertify, protected_paths, read_json, require, verify_identity_digest)
from experiments.branch_readout_gw9.replay import compare_arrays
from experiments.chronological import evaluate_original_graph_message as metrics
from experiments.chronological.graph_message_reuse import RUNTIME_KEYS
from experiments.chronological.original_igstgnn_runtime import device_batch

base, epoch_tools, REPO = train.base, train.epoch_tools, train.REPO
SCHEMA = 'original_igstgnn_branch_readout_gw9_probe_v1'
ARCHIVE_SCHEMA = 'original_igstgnn_branch_readout_gw9_probe_archive_v1'
PRODUCER_SCHEMA = 'gw9_frozen_branch_readout_probe_producer_v1'
PROBE_EPOCHS = 3
ARMS = PROBE_ARMS
CASES = ('FROZEN_R', *ARMS)
AXIS_KEYS = ('y', 'valid', 'sample_ids', 'node_ids', 'road_labels', 'direct_support')
GROUPS = ('high_value_ge314', 'long_horizon_7_12')
EXECUTION_FILES = train.EXECUTION_FILES


def source_hashes():
    return {name: base.sha256(REPO/name) for name in EXECUTION_FILES}


def record(prediction, y, valid, modules=None):
    result = metrics.point_metrics(prediction, y, valid)
    error = prediction[valid].astype(np.float64)-y[valid].astype(np.float64)
    result['signed_bias'] = float(error.mean())
    scores, complete = metrics.window_scores(prediction, y, valid)
    result['window_p95'] = float(np.quantile(scores[complete], .95)) if complete.any() else None
    result['groups'] = {}
    for name, mask in (('high_value_ge314', valid & (y >= 314)),
            ('long_horizon_7_12', valid & (np.arange(12)[None, :, None] >= 6))):
        count = int(mask.sum())
        result['groups'][name] = dict(valid_count=count, mae=float(np.abs(
            prediction[mask].astype(np.float64)-y[mask].astype(np.float64)).mean()) if count else None)
    if modules is not None:
        result['native'] = metrics.native_metrics(prediction, y, modules)
    return result


def load_baseline(reference):
    saved = reference['archives']['best']; directory = Path(saved['directory'])
    require(base.sha256(directory/'metadata.json') == saved['metadata_sha256']
        and base.sha256(directory/'paired_predictions.npz') == saved['archive_sha256'],
        'Certified GW8 BEST cache changed')
    with np.load(directory/'paired_predictions.npz', allow_pickle=False) as payload:
        result = {name: payload[name].copy() for name in AXIS_KEYS}
        result['pred__FROZEN_R'] = payload['pred__R_CONTINUE'].copy()
    result['_cache_runtime'] = read_json(directory/'metadata.json').get('runtime', {})
    require(base.sha256(directory/'metadata.json') == saved['metadata_sha256']
        and base.sha256(directory/'paired_predictions.npz') == saved['archive_sha256'],
        'GW8 BEST cache changed while reading')
    return result


class ProbeEngine:
    def __init__(self, extractor, data, modules, device, baseline, hidden=16):
        require(extractor.stage == 'probe' and extractor.head is None,
            'Probe requires a frozen R extractor without a readout')
        self.extractor, self.data, self.modules = extractor, data, modules
        self.device, self.baseline = torch.device(device), baseline
        self.loader = data.loaders['train_loader']
        self.backbone_tensor_sha256 = base.tensor_tree_hash(extractor.anchor.state_dict())
        self.heads = nn.ModuleDict()
        devices = [self.device.index or 0] if self.device.type == 'cuda' else []
        for arm in ARMS:
            with torch.random.fork_rng(devices=devices):
                self.heads[arm] = ReadoutHead(extractor.channels, extractor.gap, arm, hidden).to(self.device)
        config = data.config
        self.optimizers = {arm: torch.optim.Adam(self.heads[arm].parameters(),
            lr=config['lrate'], weight_decay=config['wdecay'], eps=1e-8) for arm in ARMS}
        self.schedulers = {arm: torch.optim.lr_scheduler.MultiStepLR(self.optimizers[arm],
            milestones=base.MILESTONES, gamma=.5) for arm in ARMS}
        self.check_backbone()

    def check_backbone(self):
        require(not any(p.requires_grad or p.grad is not None for p in self.extractor.anchor.parameters()),
            'Frozen backbone acquired trainability or gradients')
        require(not any(module.training for module in self.extractor.anchor.modules()),
            'Frozen backbone left evaluation mode')
        require(base.tensor_tree_hash(self.extractor.anchor.state_dict()) == self.backbone_tensor_sha256,
            'Frozen backbone parameters or buffers changed')
        self.extractor.assert_cache_clear()
        return True

    def _start(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device); torch.cuda.reset_peak_memory_stats(self.device)
        return perf_counter()

    def _timing(self, start, batches):
        if self.device.type == 'cuda': torch.cuda.synchronize(self.device)
        return dict(seconds=perf_counter()-start, batches=batches,
            backbone_complete_forwards=batches, shared_forward_between_heads=True,
            peak_allocated_bytes=int(torch.cuda.max_memory_allocated(self.device))
                if self.device.type == 'cuda' else None,
            scope='streaming_loader_H2D_one_frozen_forward_two_readouts_and_native_loss')

    def train_epoch(self, epoch):
        self.check_backbone(); self.extractor.eval(); self.heads.train(); self.loader.shuffle()
        start = self._start(); losses = {arm: [] for arm in ARMS}; batches = 0
        for batches, raw in enumerate(self.loader.get_iterator(), 1):
            x, y, incident, sensors = device_batch(raw, self.device)
            with torch.no_grad():
                features = self.extractor.capture_features(x, incident_data=incident, sensor_data=sensors)
                target = self.data.scaler.inverse_transform(y)
            require(all(bool(torch.isfinite(value).all()) for value in (*features.values(), target)),
                'Nonfinite frozen features or training targets')
            null = target.min() if target.min() < 1 else target.new_tensor(0.)
            for arm in ARMS:
                optimizer = self.optimizers[arm]; optimizer.zero_grad(set_to_none=True)
                normalized = features['base']+self.heads[arm](features['S'], features['Q'])
                prediction = self.data.scaler.inverse_transform(normalized)
                require(bool(torch.isfinite(prediction).all()), 'Nonfinite probe prediction')
                loss = self.modules.metrics.masked_mae(prediction, target, null)
                require(bool(torch.isfinite(loss)), 'Nonfinite probe training loss')
                loss.backward()
                require(all(p.grad is None or bool(torch.isfinite(p.grad).all())
                    for p in self.heads[arm].parameters()), 'Nonfinite probe gradient')
                if self.data.config['clip_grad_value'] != 0:
                    torch.nn.utils.clip_grad_norm_(self.heads[arm].parameters(), self.data.config['clip_grad_value'])
                optimizer.step(); losses[arm].append(float(loss.detach()))
            if batches == 1 or batches % 10 == 0 or batches == self.loader.num_batch:
                print(json.dumps(dict(phase='frozen_probe_training', epoch=epoch,
                    batch=batches, total=self.loader.num_batch)), flush=True)
        require(batches == self.loader.num_batch, 'Probe loader batch budget differs')
        self.check_backbone()
        return ({arm: dict(mean_batch_train_mae=float(np.mean(losses[arm])),
            updates=batches, learning_rate=self.schedulers[arm].get_last_lr()[0]) for arm in ARMS},
            self._timing(start, batches))

    def evaluate(self):
        self.check_backbone(); self.extractor.eval(); self.heads.eval()
        start = self._start()
        predictions = {name: [] for name in CASES}; truths, supports = [], []
        with torch.no_grad():
            for index, raw in enumerate(self.data.loaders['val_loader'].get_iterator(), 1):
                x, y, incident, sensors = device_batch(raw, self.device)
                f = self.extractor.capture_features(x, incident_data=incident, sensor_data=sensors)
                require(all(bool(torch.isfinite(value).all()) for value in f.values()),
                    'Nonfinite validation features')
                normalized = {'FROZEN_R': f['base']}
                normalized.update({arm: f['base']+self.heads[arm](f['S'], f['Q']) for arm in ARMS})
                for arm, value in normalized.items():
                    predictions[arm].append(self.data.scaler.inverse_transform(value).squeeze(-1).cpu().numpy())
                truths.append(self.data.scaler.inverse_transform(y).squeeze(-1).cpu().numpy())
                supports.append(incident['distances'].abs().sum(-1).cpu().numpy() > 0)
                if index == 1 or index == self.data.loaders['val_loader'].num_batch:
                    print(json.dumps(dict(phase='frozen_probe_validation', batch=index,
                        total=self.data.loaders['val_loader'].num_batch)), flush=True)
        predictions = {name: np.concatenate(values) for name, values in predictions.items()}
        truth, support = np.concatenate(truths), np.concatenate(supports)
        require(np.array_equal(truth, self.baseline['y'])
            and np.array_equal(support, self.baseline['direct_support']), 'Probe validation axes differ')
        require(all(np.isfinite(value).all() for value in predictions.values()), 'Nonfinite probe validation')
        self.last_frozen_replay = compare_arrays(predictions['FROZEN_R'], self.baseline['pred__FROZEN_R'])
        self.last_frozen_replay.update(
            actual_native=metrics.native_metrics(predictions['FROZEN_R'], truth, self.modules),
            cached_native=metrics.native_metrics(self.baseline['pred__FROZEN_R'], truth, self.modules),
            anchor_requires_grad_at_boundary=any(p.requires_grad for p in self.extractor.anchor.parameters()),
            device_name=torch.cuda.get_device_name(self.device) if self.device.type == 'cuda' else 'CPU',
            torch=str(torch.__version__), cuda=torch.version.cuda,
            cached_runtime=self.baseline.get('_cache_runtime', {}),
            policy='strict_dtype_shape_and_values_no_tolerance_relaxation')
        require(self.last_frozen_replay['exact_equal'],
            'Frozen R prediction differs from certified GW8 BEST cache: '+json.dumps(self.last_frozen_replay))
        self.check_backbone()
        records = {name: record(value, truth, self.baseline['valid'], self.modules)
            for name, value in predictions.items()}
        self.last_validation_timing = self._timing(start, self.data.loaders['val_loader'].num_batch)
        self.last_validation_timing['scope'] = 'all_validation_batches_shared_frozen_forward_two_heads_inverse_global_metrics'
        return records, predictions


def head_initializations(runner):
    return {arm: dict(head_tensor_sha256=base.tensor_tree_hash(runner.heads[arm].state_dict()),
        registered_parameters=sum(p.numel() for p in runner.heads[arm].parameters()),
        trainable_parameters=sum(p.numel() for p in runner.heads[arm].parameters() if p.requires_grad))
        for arm in ARMS}


def commit_state(runner, state, directory, identity, get_order):
    state.update(identity_sha256=identity['identity_sha256'],
        # Detached copies remain LAST even when final evaluation loads BEST into
        # the live heads. state_dict() alone aliases those parameter storages.
        heads_state={arm: {key: value.detach().cpu().clone()
            for key, value in runner.heads[arm].state_dict().items()} for arm in ARMS},
        optimizers_state={arm: runner.optimizers[arm].state_dict() for arm in ARMS},
        schedulers_state={arm: runner.schedulers[arm].state_dict() for arm in ARMS},
        train_order=get_order(), rng_state=base.rng_state(),
        backbone_tensor_sha256=runner.backbone_tensor_sha256)
    state['heads_tensor_sha256'] = {arm: base.tensor_tree_hash(state['heads_state'][arm]) for arm in ARMS}
    epoch_tools.atomic_save(Path(directory)/'last.pt', state)
    base.write_json(Path(directory)/'history.json', state['history'])


def validate_saved(state, identity, directory, batches):
    verify_identity_digest(identity)
    require(identity.get('schema') == SCHEMA and identity.get('design') == DESIGN
        and identity.get('arms') == list(ARMS) and identity.get('fixed_screen_epochs') == PROBE_EPOCHS,
        'Expected the fixed three-epoch GW9 probe identity')
    require(state.get('schema') == SCHEMA and state.get('identity_sha256') == identity['identity_sha256'],
        'Saved paired probe identity differs')
    completed, history = state.get('completed_epoch'), state.get('history', [])
    require(type(completed) is int and 0 <= completed <= PROBE_EPOCHS and len(history) == completed
        and [row['epoch'] for row in history] == list(range(1, completed+1)), 'Saved probe history differs')
    require(state['global_updates'] == completed*batches
        and state['head_updates'] == {arm: completed*batches for arm in ARMS}
        and all(row['head_updates'] == {arm: row['epoch']*batches for arm in ARMS} for row in history),
        'Saved probe head budgets differ')
    require(state['backbone_tensor_sha256'] == identity['backbone_tensor_sha256'],
        'Saved frozen backbone identity differs')
    for records in (state['stage_zero_validation'], *(row['validation'] for row in history)):
        require(set(records) == set(CASES), 'Incomplete probe validation records')
        require(records['FROZEN_R'] == state['stage_zero_validation']['FROZEN_R'],
            'Frozen R validation changed across epochs')
        for item in records.values():
            require(all(np.isfinite(item[key]) and item[key] >= 0
                for key in ('mae_pooled', 'rmse_pooled', 'mape_pooled'))
                and np.isfinite(item['signed_bias'])
                and all(np.isfinite(item['native'][key]) and item['native'][key] >= 0
                    for key in ('mae', 'rmse', 'mape_fraction')),
                'Nonfinite or negative probe validation metrics')
            for group in GROUPS:
                row = item['groups'][group]
                require(type(row['valid_count']) is int and row['valid_count'] >= 0
                    and (row['mae'] is None if row['valid_count'] == 0
                        else row['mae'] is not None and np.isfinite(row['mae']) and row['mae'] >= 0),
                    'Invalid probe evaluation-only group')
    order = np.asarray(state['train_order']); count = identity['data']['splits']['train']['count']
    require(order.shape == (count,) and np.array_equal(np.sort(order), np.arange(count)),
        'Saved probe training permutation differs')
    require(all(set(state[key]) == set(ARMS) for key in ('heads_state', 'heads_tensor_sha256',
        'optimizers_state', 'schedulers_state', 'initial_head_files', 'best')), 'Incomplete paired probe state')
    for arm in ARMS:
        require(base.tensor_tree_hash(state['heads_state'][arm]) == state['heads_tensor_sha256'][arm]
            and state['schedulers_state'][arm]['last_epoch'] == completed, 'Saved head/scheduler differs: '+arm)
        rows = [(0, state['stage_zero_validation'][arm]),
            *((row['epoch'], row['validation'][arm]) for row in history)]
        require(all(np.isfinite(row['mae_pooled']) and row['mae_pooled'] > 0 for _, row in rows),
            'Invalid probe validation MAE')
        epoch, selected = min(rows, key=lambda item: item[1]['mae_pooled'])
        best = state['best'][arm]
        require(best['epoch'] == epoch and best['mae_pooled'] == selected['mae_pooled'],
            'Probe BEST must include stage zero: '+arm)
        for item in (best, state['initial_head_files'][arm]):
            name = item['checkpoint_file']
            require(isinstance(name, str) and Path(name).name == name
                and base.sha256(Path(directory)/name) == item['checkpoint_sha256'], 'Probe head checkpoint changed')
        initial = base.load_checkpoint(Path(directory)/state['initial_head_files'][arm]['checkpoint_file'])
        require(base.tensor_tree_hash(initial) == identity['initializations'][arm]['head_tensor_sha256'],
            'Probe initial head tensors changed')


def run_epochs(runner, directory, identity, target_epoch=PROBE_EPOCHS, state=None):
    require(type(target_epoch) is int and 0 <= target_epoch <= PROBE_EPOCHS,
        'Probe is capped at three total epochs; no budget extension')
    directory = Path(directory)
    get_order = epoch_tools.tracked_shuffle(runner.loader, None if state is None else state['train_order'])
    if state is None:
        require(head_initializations(runner) == identity['initializations'], 'Actual head initialization differs')
        records, _ = runner.evaluate()
        require(all(records[arm] == records['FROZEN_R'] for arm in ARMS), 'Zero-head stage zero must equal frozen R')
        expected = identity.get('source_reference', {}).get('source_validation_metrics')
        if expected is not None:
            require(all(np.isclose(records['FROZEN_R']['native'][key], value, atol=1e-5, rtol=1e-6)
                for key, value in expected.items()), 'Frozen R native metric replay differs')
        state = dict(schema=SCHEMA, completed_epoch=0, global_updates=0,
            head_updates={arm: 0 for arm in ARMS}, stage_zero_validation=records,
            stage_zero_validation_timing=getattr(runner, 'last_validation_timing', None),
            initial_head_files={}, best={}, history=[])
        for arm in ARMS:
            name = arm+'_best_epoch_000_'+uuid.uuid4().hex[:12]+'.pt'
            epoch_tools.atomic_save(directory/name, runner.heads[arm].state_dict())
            item = dict(checkpoint_file=name, checkpoint_sha256=base.sha256(directory/name))
            state['initial_head_files'][arm] = dict(item)
            state['best'][arm] = dict(item, epoch=0, mae_pooled=records[arm]['mae_pooled'])
        commit_state(runner, state, directory, identity, get_order)
    else:
        validate_saved(state, identity, directory, runner.loader.num_batch)
        require(state['completed_epoch'] <= target_epoch, 'Cannot resume backwards')
        for arm in ARMS:
            runner.heads[arm].load_state_dict(state['heads_state'][arm], strict=True)
            runner.optimizers[arm].load_state_dict(state['optimizers_state'][arm])
            runner.schedulers[arm].load_state_dict(state['schedulers_state'][arm])
        epoch_tools.restore_rng(state['rng_state'])
    runner.check_backbone()
    for epoch in range(state['completed_epoch']+1, target_epoch+1):
        train_metrics, timing = runner.train_epoch(epoch)
        records, _ = runner.evaluate()
        row = dict(epoch=epoch, train_metrics=train_metrics, train_timing=timing,
            validation_timing=getattr(runner, 'last_validation_timing', None),
            validation=records, head_updates={arm: epoch*runner.loader.num_batch for arm in ARMS},
            train_order_sha256=hashlib.sha256(np.asarray(get_order(), dtype=np.int64).tobytes()).hexdigest())
        for arm in ARMS:
            runner.schedulers[arm].step()
            if records[arm]['mae_pooled'] < state['best'][arm]['mae_pooled']:
                name = arm+'_best_epoch_{:03d}_{}.pt'.format(epoch, uuid.uuid4().hex[:12])
                epoch_tools.atomic_save(directory/name, runner.heads[arm].state_dict())
                state['best'][arm] = dict(epoch=epoch, mae_pooled=records[arm]['mae_pooled'],
                    checkpoint_file=name, checkpoint_sha256=base.sha256(directory/name))
        state['history'].append(row)
        state.update(completed_epoch=epoch, global_updates=epoch*runner.loader.num_batch,
            head_updates=dict(row['head_updates']))
        runner.check_backbone(); commit_state(runner, state, directory, identity, get_order)
        base.write_json(directory/'summary.json', training_summary(state, identity))
        print(json.dumps(dict(status='GW9_PROBE_EPOCH_COMMITTED', **row)), flush=True)
    return state


def training_summary(state, identity):
    return dict(schema=SCHEMA, status='GW9_PROBE_TRAINING_COMPLETE' if state['completed_epoch'] == PROBE_EPOCHS
        else 'GW9_PROBE_TRAINING_PARTIAL', completed_epoch=state['completed_epoch'],
        target_completed_epochs=PROBE_EPOCHS, head_updates=state['head_updates'], best=state['best'],
        backbone_updates=0, frozen_R_is_equal_budget_control=False, same_budget_comparison=list(ARMS),
        source_reference=identity.get('source_reference'), best_selection_includes_stage_zero=True,
        optimizer_reset_at_start=True, optimizer_reset_on_resume=False, early_stopping_enabled=False,
        stage_zero_validation_timing=state.get('stage_zero_validation_timing'),
        phase_timings=[dict(epoch=row['epoch'], train=row['train_timing'],
            validation=row.get('validation_timing')) for row in state['history']],
        extension_allowed=False, joint_training_started=False, test_evaluated=False)


def probe_screening(best, last):
    value = best['BRANCH_READOUT']['mae_pooled']
    require(value > 0 and all(best[arm]['mae_pooled'] > 0 for arm in CASES), 'Positive probe MAE required')
    checks = {'BEST_relative_gain_vs_SUM_at_least_0_005':
        (best['SUM_CONTROL']['mae_pooled']-value)/best['SUM_CONTROL']['mae_pooled'] >= .005,
        'BEST_relative_gain_vs_frozen_R_at_least_0_005':
        (best['FROZEN_R']['mae_pooled']-value)/best['FROZEN_R']['mae_pooled'] >= .005,
        'LAST_own_degradation_at_most_0_01': (last['BRANCH_READOUT']['mae_pooled']-value)/value <= .01}
    for name in GROUPS:
        a, b = best['BRANCH_READOUT']['groups'][name]['mae'], best['FROZEN_R']['groups'][name]['mae']
        checks['BEST_'+name+'_no_worse_than_frozen_R'] = a is not None and b is not None and a <= b
    checks = {name: bool(value) for name, value in checks.items()}
    return dict(status='GW9_PROBE_SCREENING_COMPLETE', checks=checks,
        eligible_for_joint_screen=all(checks.values()), candidate_for_confirmation=False,
        joint_training_started=False, frozen_R_is_equal_budget_control=False,
        interpretation='single_seed_development_probe_not_independent_confirmation',
        screening_failure_does_not_authorize_budget_extension=True, test_evaluated=False)


def selected_role(state, directory, arm, role):
    if role == 'best':
        item = state['best'][arm]; path = Path(directory)/item['checkpoint_file']
        require(base.sha256(path) == item['checkpoint_sha256'], 'Probe BEST file changed')
        weights, epoch = base.load_checkpoint(path), item['epoch']
    else:
        path, weights, epoch = Path(directory)/'last.pt', state['heads_state'][arm], state['completed_epoch']
    return weights, dict(selection=role, epoch=epoch, checkpoint_sha256=base.sha256(path),
        head_tensor_sha256=base.tensor_tree_hash(weights), policy=arm)


def finalize(runner, directory, identity, state):
    directory = Path(directory); validate_saved(state, identity, directory, runner.loader.num_batch)
    require(state['completed_epoch'] == PROBE_EPOCHS, 'Finish both heads at three epochs')
    last_sha = base.sha256(directory/'last.pt')
    output = directory/('evaluation_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'_'+uuid.uuid4().hex[:8])
    output.mkdir(); roles, all_records, digests = {}, {}, {}
    for role in ('best', 'last'):
        roles[role] = {}
        for arm in ARMS:
            weights, roles[role][arm] = selected_role(state, directory, arm, role)
            runner.heads[arm].load_state_dict(weights, strict=True)
        records, predictions = runner.evaluate(); all_records[role] = records
        replay = {}
        for arm in ARMS:
            epoch = roles[role][arm]['epoch']
            expected = state['stage_zero_validation'][arm] if epoch == 0 else state['history'][epoch-1]['validation'][arm]
            passed = all(np.isclose(records[arm][key], expected[key], atol=1e-10, rtol=1e-10)
                for key in ('mae_pooled', 'rmse_pooled', 'mape_pooled'))
            passed = passed and all(np.isclose(records[arm]['native'][key], expected['native'][key],
                atol=1e-5, rtol=1e-6) for key in ('mae', 'rmse', 'mape_fraction'))
            require(passed, 'Probe selected validation replay differs: '+arm)
            replay[arm] = dict(passed=True, actual=records[arm]['native'], expected=expected['native'])
        child = output/role; child.mkdir()
        archive = {key: runner.baseline[key] for key in AXIS_KEYS}
        archive.update(schema=np.array(ARCHIVE_SCHEMA), axis_order=np.array('sample,horizon,node'),
            registered_arms=np.asarray(ARMS, dtype=str),
            **{'pred__'+arm: predictions[arm] for arm in CASES})
        metrics.validate_archive(archive, CASES, ARCHIVE_SCHEMA)
        np.savez_compressed(child/'paired_predictions.npz', **archive)
        metadata = dict(schema=PRODUCER_SCHEMA, status='GW9_PROBE_PREDICTION_PASS',
            run_identity_sha256=identity['identity_sha256'], source_reference=identity['source_reference'],
            checkpoint_selection=role, head_roles=roles[role], validation_replay=replay,
            frozen_R_exact_cache_replay=True, backbone_tensor_sha256=runner.backbone_tensor_sha256,
            frozen_R_replay=getattr(runner, 'last_frozen_replay', None),
            backbone_updates=0, head_updates=state['head_updates'], completed_epoch=PROBE_EPOCHS,
            execution_source=source_hashes(), archive_sha256=base.sha256(child/'paired_predictions.npz'),
            timing=getattr(runner, 'last_validation_timing', None),
            optimizer_updates_this_evaluation=0, test_evaluated=False)
        base.write_json(child/'metadata.json', metadata); base.write_json(child/'analysis.json', records)
        digests[role] = dict(metadata_sha256=base.sha256(child/'metadata.json'), archive_sha256=metadata['archive_sha256'])
    recertify(identity['source_reference']); runner.check_backbone()
    require(base.sha256(directory/'last.pt') == last_sha and identity['execution_source'] == source_hashes(),
        'Probe state or source changed during final evaluation')
    result = probe_screening(all_records['best'], all_records['last'])
    result.update(run_identity_sha256=identity['identity_sha256'], last_checkpoint_sha256=last_sha,
        evaluation_directory=output.name, evaluations=digests)
    lines = ['# GW9 frozen-R paired development probe', '',
        'Two equal-capacity heads, three epochs each; shared streaming frozen eval backbone. BEST includes stage zero.',
        'FROZEN_R has zero updates and is not a trained equal-budget control. No test evaluation.', '',
        '| Selection | Case | MAE | RMSE | MAPE % | Signed bias |', '|---|---|---:|---:|---:|---:|']
    for role, records in all_records.items():
        for arm, row in records.items():
            lines.append('| {} | {} | {:.9f} | {:.9f} | {:.9f} | {:.9f} |'.format(
                role, arm, row['mae_pooled'], row['rmse_pooled'], 100*row['mape_pooled'], row['signed_bias']))
    lines += ['', 'eligible_for_joint_screen: '+str(result['eligible_for_joint_screen']).lower(),
        'Passing permits the separately invoked five-epoch joint screen; it is not independent confirmation.',
        'Failure stops this branch without extending epochs or changing the locked criteria.', '']
    (output/'report.md').write_text('\n'.join(lines), encoding='utf-8')
    (directory/'report.md').write_text('\n'.join(lines), encoding='utf-8')
    base.write_json(output/'screening.json', result)
    # Publish this pointer last; interrupted evaluation is repeated with a new child directory.
    base.write_json(directory/'screening.json', result)
    base.write_json(directory/'summary.json', dict(training_summary(state, identity), screening=result))
    print(json.dumps(result), flush=True)
    return result


def validate_gate(probe_dir, reference):
    """Recompute the gate from bound prediction archives, never trust an edited report."""
    directory = Path(probe_dir).resolve(); identity = read_json(directory/'identity.json')
    identity_sha = base.sha256(directory/'identity.json')
    verify_identity_digest(identity)
    require(identity.get('source_reference') == reference and identity.get('execution_source') == source_hashes(),
        'Passing probe must use the unchanged certified GW8 source and GW9 code')
    old_identity, _, _ = recertify(reference)
    require(identity.get('data') == old_identity['data']
        and identity.get('backbone_tensor_sha256') == reference['source_initialization']['model_tensor_sha256']
        and all(identity['runtime'].get(key) == old_identity['runtime'].get(key) for key in RUNTIME_KEYS),
        'Probe/source data, backbone or runtime identity differs')
    require(read_json(directory/'budget.json') == dict(target_completed_epochs=PROBE_EPOCHS,
        extension_allowed=False, backbone_updates=0, registered_heads=list(ARMS), test_evaluated=False),
        'Probe budget file differs')
    budget_sha = base.sha256(directory/'budget.json')
    modules = base.get_original_modules()
    require(modules.source_fingerprint == identity['source_fingerprint'], 'Probe gate native source differs')
    last_sha = base.sha256(directory/'last.pt'); state = base.load_checkpoint(directory/'last.pt')
    validate_saved(state, identity, directory, len(identity['data']['splits']['train']['partition']))
    require(state['completed_epoch'] == PROBE_EPOCHS, 'Joint stage requires a completed three-epoch probe')
    pointer = read_json(directory/'screening.json'); pointer_sha = base.sha256(directory/'screening.json')
    require(pointer.get('run_identity_sha256') == identity['identity_sha256']
        and pointer.get('last_checkpoint_sha256') == last_sha, 'Probe screening state binding differs')
    name = pointer.get('evaluation_directory')
    require(isinstance(name, str) and Path(name).name == name and name.startswith('evaluation_'),
        'Probe evaluation must be a direct child')
    output = directory/name
    require(output.resolve().parent == directory, 'Probe evaluation path escapes the run')
    baseline = load_baseline(reference); records = {}
    for role in ('best', 'last'):
        child = output/role; info = pointer['evaluations'][role]
        require(base.sha256(child/'metadata.json') == info['metadata_sha256']
            and base.sha256(child/'paired_predictions.npz') == info['archive_sha256'], 'Probe evaluation changed')
        metadata = read_json(child/'metadata.json')
        require(metadata.get('schema') == PRODUCER_SCHEMA and metadata.get('status') == 'GW9_PROBE_PREDICTION_PASS'
            and metadata.get('checkpoint_selection') == role
            and metadata.get('run_identity_sha256') == identity['identity_sha256']
            and metadata.get('source_reference') == reference and metadata.get('execution_source') == source_hashes()
            and metadata.get('backbone_tensor_sha256') == identity['backbone_tensor_sha256']
            and metadata.get('archive_sha256') == info['archive_sha256']
            and metadata.get('backbone_updates') == 0 and metadata.get('completed_epoch') == PROBE_EPOCHS
            and metadata.get('head_updates') == state['head_updates']
            and metadata.get('frozen_R_exact_cache_replay') is True
            and metadata.get('optimizer_updates_this_evaluation') == 0
            and metadata.get('test_evaluated') is False, 'Probe producer certificate differs')
        for arm in ARMS:
            _, selected = selected_role(state, directory, arm, role)
            require(metadata['head_roles'][arm] == selected
                and metadata['validation_replay'][arm]['passed'] is True, 'Probe checkpoint selection/replay differs')
        with np.load(child/'paired_predictions.npz', allow_pickle=False) as payload:
            archive = dict(payload)
        require(np.array_equal(archive.get('registered_arms'), np.asarray(ARMS)), 'Probe registered arms differ')
        y, valid, predictions = metrics.validate_archive(archive, CASES, ARCHIVE_SCHEMA)
        for key in (*AXIS_KEYS, 'pred__FROZEN_R'):
            require(archive[key].shape == baseline[key].shape and archive[key].dtype == baseline[key].dtype
                and np.array_equal(archive[key], baseline[key]), 'Probe baseline axes/prediction changed: '+key)
        records[role] = {arm: record(predictions[arm], y, valid, modules) for arm in CASES}
        # Bind actual aggregate outputs to recorded selection metrics as well as files.
        for arm in ARMS:
            epoch = metadata['head_roles'][arm]['epoch']
            expected = state['stage_zero_validation'][arm] if epoch == 0 else state['history'][epoch-1]['validation'][arm]
            require(all(np.isclose(records[role][arm][key], expected[key], atol=1e-10, rtol=1e-10)
                for key in ('mae_pooled', 'rmse_pooled', 'mape_pooled')), 'Probe archive metric replay differs')
            require(all(np.isclose(records[role][arm]['native'][key], expected['native'][key],
                atol=1e-5, rtol=1e-6) for key in ('mae', 'rmse', 'mape_fraction')),
                'Probe archive native metric replay differs')
        require(base.sha256(child/'metadata.json') == info['metadata_sha256']
            and base.sha256(child/'paired_predictions.npz') == info['archive_sha256'], 'Probe evaluation changed while reading')
    result = probe_screening(records['best'], records['last'])
    require(all(pointer.get(key) == value for key, value in result.items()), 'Probe locked screening results differ')
    require(result['eligible_for_joint_screen'], 'Frozen branch probe failed; joint training is not permitted by this protocol')
    require(base.sha256(directory/'last.pt') == last_sha and base.sha256(directory/'screening.json') == pointer_sha,
        'Probe changed during gate certification')
    require(base.sha256(directory/'identity.json') == identity_sha
        and base.sha256(directory/'budget.json') == budget_sha, 'Probe identity/budget changed during certification')
    return dict(schema='gw9_authenticated_passing_probe_gate_v1', directory=str(directory),
        identity_sha256=identity['identity_sha256'], identity_file_sha256=identity_sha, budget_file_sha256=budget_sha,
        last_checkpoint_sha256=last_sha, screening_file_sha256=pointer_sha,
        evaluation_directory=name, evaluations=pointer['evaluations'], checks=result['checks'],
        eligible_for_joint_screen=True, probe_weights_loaded_into_joint=False, test_evaluated=False)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gw8-run', type=Path, required=True)
    parser.add_argument('--gw8-best-eval', type=Path)
    parser.add_argument('--gw8-last-eval', type=Path)
    parser.add_argument('--data-dir', type=Path, default=base.default_data_dir())
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--hidden', type=int, choices=(16,), default=16)
    parser.add_argument('--epochs', type=int, choices=(PROBE_EPOCHS,), default=PROBE_EPOCHS)
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--threads', type=int, default=3)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--evaluate-only', action='store_true')
    args = parser.parse_args(argv)
    if args.threads < 1: parser.error('threads must be positive')
    for key in ('gw8_run', 'data_dir', 'run_dir'): setattr(args, key, getattr(args, key).resolve())
    return args


def main(argv=None):
    args = parse_args(argv); existing = args.resume or args.evaluate_only
    require(args.run_dir.exists() == existing, 'Use a fresh probe directory or resume/evaluate the same run')
    identity_path = args.run_dir/'identity.json'
    if existing and identity_path.is_file():
        saved = read_json(identity_path); require(saved.get('schema') == SCHEMA, 'Expected GW9 probe identity')
        reference = saved['source_reference']
        require(reference['source_run'] == str(args.gw8_run), 'Cannot replace probe source run')
        old_identity, gw4_weights, recurrent_weights = recertify(reference)
    else:
        require(not args.evaluate_only, 'Evaluate-only requires an existing probe identity')
        if existing:
            require(all(path.name in ('active.lock', 'record.log') or
                (path.name.startswith('failure_') and path.suffix == '.json')
                for path in args.run_dir.iterdir()), 'Identity-less resume directory contains unknown files')
        reference, old_identity, gw4_weights, recurrent_weights = certify(args.gw8_run,
            {role: getattr(args, 'gw8_'+role+'_eval') for role in ('best', 'last')})
    for protected in protected_paths(reference):
        require(args.run_dir != protected and protected not in args.run_dir.parents,
            'Probe output must be outside immutable source runs')
    args.run_dir.mkdir(parents=True, exist_ok=existing)
    with base.run_lock(args.run_dir):
        try:
            runtime = base.configure_runtime(args.device, args.threads, args.seed)
            for key in RUNTIME_KEYS:
                require(runtime.get(key) == old_identity['runtime'].get(key), 'Probe/source runtime differs: '+key)
            modules = base.get_original_modules()
            require(modules.source_fingerprint == old_identity['source_fingerprint'], 'Native source differs')
            config = dict(base.CONFIG, device=args.device, seed=args.seed, warm_epoch=101)
            data = base.load_data(args.data_dir, config, modules, logging.getLogger('gw9-probe'), check=False, training=True)
            require(data.identity == old_identity['data'], 'Probe data identity differs')
            baseline = load_baseline(reference)
            require(np.array_equal(baseline['sample_ids'], data.identity['splits']['val']['selected_indices'])
                and np.array_equal(baseline['node_ids'], data.station_ids), 'Probe sample/node axes differ')
            roads = np.full(len(data.station_ids), 'unknown', dtype='<U40')
            for road, indices in data.roads.items(): roads[indices] = road
            require(np.array_equal(roads, baseline['road_labels']), 'Probe road axis differs')
            extractor = train.model_for_arm(data, modules, args.device, 'R_CONTINUE', gw4_weights,
                recurrent_weights, hidden=args.hidden, stage='probe')
            runner = ProbeEngine(extractor, data, modules, args.device, baseline, args.hidden)
            initializations = head_initializations(runner)
            require(len({row['head_tensor_sha256'] for row in initializations.values()}) == 1
                and all(row['trainable_parameters'] == row['registered_parameters'] == 8256
                    for row in initializations.values()), 'Probe initial head tensors/capacities differ')
            require(runner.backbone_tensor_sha256 == reference['source_initialization']['model_tensor_sha256'],
                'Probe backbone differs from certified GW8 R BEST')
            identity = dict(schema=SCHEMA, design=DESIGN, arms=list(ARMS), fixed_screen_epochs=PROBE_EPOCHS,
                model_options=dict(hidden=args.hidden), training_config=config, runtime=runtime,
                source_fingerprint=modules.source_fingerprint, data=data.identity, source_reference=reference,
                initializations=initializations, backbone_tensor_sha256=runner.backbone_tensor_sha256,
                execution_source=source_hashes(), scheduler=dict(milestones=base.MILESTONES, gamma=.5),
                common_post_construction_training_rng='configure_runtime(seed)_after_all_constructors',
                streaming_features=True, best_selection_includes_stage_zero=True,
                optimizer_reset_at_start=True, frozen_R_is_equal_budget_control=False, test_evaluated=False)
            identity['identity_sha256'] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
            budget = dict(target_completed_epochs=PROBE_EPOCHS, extension_allowed=False,
                backbone_updates=0, registered_heads=list(ARMS), test_evaluated=False)
            if identity_path.is_file():
                require(read_json(identity_path) == identity and read_json(args.run_dir/'budget.json') == budget,
                    'Probe identity or fixed budget changed')
            else:
                base.write_json(args.run_dir/'budget.json', budget); base.write_json(identity_path, identity)
            state = base.load_checkpoint(args.run_dir/'last.pt') if (args.run_dir/'last.pt').is_file() else None
            require(not args.evaluate_only or state is not None, 'Evaluate-only requires a completed paired state')
            base.configure_runtime(args.device, args.threads, args.seed)
            if args.evaluate_only:
                validate_saved(state, identity, args.run_dir, runner.loader.num_batch)
            else:
                state = run_epochs(runner, args.run_dir, identity, state=state)
            return finalize(runner, args.run_dir, identity, state)
        except BaseException as error:
            replay = getattr(locals().get('runner'), 'last_frozen_replay', None)
            if replay is not None:
                base.write_json(args.run_dir/'frozen_replay.json', replay)
            base.write_json(args.run_dir/('failure_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')+'.json'),
                dict(status='GW9_PROBE_FAILED', error_type=type(error).__name__, error=str(error),
                    resume_semantics='paired_last_atomic_epoch_incomplete_epoch_replayed_for_both_heads',
                    frozen_replay=replay,
                    test_evaluated=False))
            raise


if __name__ == '__main__': main()
