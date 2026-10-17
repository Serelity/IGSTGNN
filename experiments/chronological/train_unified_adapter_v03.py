"""Matched v03 information/state controls; immutable recipe and epoch resumes.

Published-order point experiments and optional training-internal calibration
holdouts have distinct identities. No test loader or validation-fitted policy.
"""
import argparse
import copy
from datetime import datetime, timezone
import gc
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
import uuid

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.chronological.original_igstgnn_runtime import (
    CONFIG, build_model, configure_runtime, default_data_dir, load_data,
    run_lock, sha256, tensor_tree_hash, write_json)
from experiments.chronological.original_igstgnn_source import get_original_modules
from experiments.chronological.train_original_isp import (
    MILESTONES, atomic_save, rng_state, restore_rng, tracked_shuffle,
    load_checkpoint, arm_summary, batch_progress, timed_phase, commit_state,
    validate_saved as legacy_validate_saved)
from experiments.chronological.unified_adapter_data_v03 import (
    make_loaders, device_batch_v03, audit_data)
from experiments.chronological.unified_incident_adapter_v03 import (
    ARMS, DEFAULT_ARMS, ADAPTER_CONFIG, arm_spec, build_variant, common_traffic_state)

SCHEMA = 'unified_adapter_matched_training_v03'
EXECUTION_FILES = (
    'experiments/chronological/train_unified_adapter_v03.py',
    'experiments/chronological/unified_incident_adapter_v03.py',
    'experiments/chronological/unified_adapter_data_v03.py',
    'experiments/chronological/unified_incident_adapter.py',
    'experiments/chronological/train_original_isp.py',
    'experiments/chronological/original_isp.py',
    'experiments/chronological/original_igstgnn_runtime.py',
    'experiments/chronological/original_igstgnn_source.py',
    'experiments/chronological/train_original_igstgnn_baseline.py',
    'experiments/chronological/original_igstgnn_checkpoint_compat.py')
PAIRS = (('ORIGINAL', 'BOTH'), ('TRAFFIC_CORE', 'H_ONLY'),
         ('H_ONLY', 'R_ONLY'), ('R_ONLY', 'BOTH'), ('H_ONLY', 'BOTH'),
         ('DIRECT', 'BOTH'), ('TRAFFIC_CORE', 'TIME'),
         ('TRAFFIC_CORE', 'GRAPH'), ('SEPARATE', 'BOTH'), ('STACKED', 'BOTH'))
PROTOCOL = dict(primary_comparison='BOTH_vs_ORIGINAL', primary_metric='pooled_MAE',
    checkpoint_rule='earliest_minimum_validation_MAE_within_common_completed_budget',
    secondary_checkpoint='common_last_epoch', formal_epochs=100,
    paired_seeds=[2025, 2026, 2027],
    development_screen_only=True, independent_final_confirmation_completed=False,
    multiple_comparisons='primary_fixed_others_descriptive_no_pointwise_significance',
    requires_paired_prediction_diagnostics=True, test_evaluated=False)


def model_for_arm(data, modules, device, arm):
    original = build_model(data, modules, device)
    original_hash = tensor_tree_hash(original.state_dict())
    model = build_variant(original, arm)
    model._original_initial_tensor_sha256 = original_hash
    return model


def initialization(model, arm):
    adapter = getattr(model, 'adapter', None)
    return dict(original_initial_tensor_sha256=model._original_initial_tensor_sha256,
        traffic_tensor_sha256=tensor_tree_hash(common_traffic_state(model)),
        adapter_tensor_sha256=None if adapter is None else tensor_tree_hash(adapter.state_dict()),
        post_construction_rng_sha256=tensor_tree_hash(rng_state()),
        registered_parameters=sum(p.numel() for p in model.parameters()),
        adapter_registered_parameters=0 if adapter is None else sum(p.numel() for p in adapter.parameters()),
        arm_spec=arm_spec(arm),
        active_capacity_caveat='input_information_and_active_parameters_are_not_equal_capacity',
        zero_reference='ORIGINAL' if arm == 'STACKED' else 'TRAFFIC_CORE')


def verify_initializations(records):
    for key in ('original_initial_tensor_sha256', 'traffic_tensor_sha256', 'post_construction_rng_sha256'):
        if len({row[key] for row in records.values()}) != 1:
            raise ValueError('Common initialization failed: '+key)
    # All information/state controls register the same tensors, with different
    # execution switches. This controls initialization, not effective capacity.
    added = [r['adapter_tensor_sha256'] for r in records.values() if r['adapter_tensor_sha256'] is not None]
    if len(set(added)) > 1:
        raise ValueError('Adapter controls do not share initial tensors')


def partition_training(data, modules, logger, holdout_rows):
    """A declared array-tail head holdout; no invented date/event independence."""
    if not holdout_rows:
        data.identity = copy.deepcopy(data.identity)
        data.identity['calibration'] = dict(rows=0, candidate_fit_excludes_calibration=False,
            eligible_for_auxiliary_fit=False)
        return data
    old = data.loaders['train_loader']
    if not 0 < holdout_rows <= len(old.samples)-2:
        raise ValueError('holdout_rows must leave at least two fitting rows')
    fit_count = len(old.samples)-holdout_rows
    fit, reserved = list(old.samples[:fit_count]), list(old.samples[fit_count:])
    def loader(samples):
        return modules.loader.IncidentDataLoader(samples, True, old.bs, logger,
            input_dim=old.input_dim, sensor_info=old.sensor_info)
    data.loaders = dict(data.loaders, train_loader=loader(fit), calibration_loader=loader(reserved))
    data.identity = copy.deepcopy(data.identity)
    original_ids = data.identity['splits']['train']['selected_indices']
    selected = original_ids[:fit_count]
    data.identity['splits']['train'].update(count=fit_count, selected_indices=selected,
        bounds_half_open=[selected[0], selected[-1]+1],
        partition=[min(old.bs, fit_count-i) for i in range(0, fit_count, old.bs)])
    ids = original_ids[fit_count:]
    data.identity['splits']['calibration'] = dict(origin='declared_training_array_tail_not_certified_temporal_split',
        count=holdout_rows, selected_indices=ids, batch_size=old.bs,
        partition=[min(old.bs, holdout_rows-i) for i in range(0, holdout_rows, old.bs)])
    data.identity['calibration'] = dict(rows=holdout_rows, candidate_fit_excludes_calibration=True,
        calibration_not_used_for_checkpoint_selection=True, eligible_for_auxiliary_fit=True,
        event_independence_certified=False, absolute_order_certified=False)
    return data


class DiagnosticsAccumulator:
    """Accumulate declared sufficient statistics, never average batch means."""
    SUM_KEYS = ('state_abs_sum', 'state_elements', 'supported_positions', 'node_positions',
                'active_samples', 'output_delta_abs_sum', 'output_elements', 'edge_change_abs_sum',
                'edge_elements', 'local_gate_sum', 'local_gate_elements', 'local_gate_saturated')
    MAX_KEYS = ('output_delta_abs_max', 'log_strength_abs_max')

    def __init__(self):
        self.sums = {key: 0. for key in self.SUM_KEYS}
        self.maxima = {key: 0. for key in self.MAX_KEYS}
        self.batches = 0

    def add(self, record):
        self.batches += 1
        for key in self.SUM_KEYS:
            value = record.get(key)
            if value is not None:
                value = float(np.asarray(value).sum())
                if not np.isfinite(value):
                    raise ValueError('Nonfinite mechanism statistic: '+key)
                self.sums[key] += value
        for key in self.MAX_KEYS:
            value = record.get(key)
            if value is not None:
                value = float(np.asarray(value).max())
                if not np.isfinite(value):
                    raise ValueError('Nonfinite mechanism statistic: '+key)
                self.maxima[key] = max(self.maxima[key], value)

    def result(self):
        result = dict(scope='all_validation_forward_batches', batches=self.batches,
                      **self.sums, **self.maxima)
        for output, total, count in (
                ('state_abs_mean', 'state_abs_sum', 'state_elements'),
                ('supported_fraction', 'supported_positions', 'node_positions'),
                ('output_delta_abs_mean', 'output_delta_abs_sum', 'output_elements'),
                ('edge_change_abs_mean', 'edge_change_abs_sum', 'edge_elements'),
                ('local_gate_mean', 'local_gate_sum', 'local_gate_elements'),
                ('local_gate_saturation_fraction', 'local_gate_saturated', 'local_gate_elements')):
            result[output] = self.sums[total]/self.sums[count] if self.sums[count] else None
        return result


def _gradient_group(name):
    if not name.startswith('adapter.'):
        return None
    for prefix, group in (('encoder', 'report_encoder'), ('summary', 'history_summary'),
            ('graph.query', 'graph_query'), ('graph.key', 'graph_key'),
            ('graph.strength', 'graph_strength'), ('output', 'prediction_output'),
            ('decoder', 'prediction_decoder'), ('local_gate', 'local_gate'),
            ('transition', 'state_transition'), ('initial', 'state_initial')):
        if name[len('adapter.'):].startswith(prefix):
            return group
    return 'other_adapter'


def engine_class(base):
    class Engine(base):
        def __init__(self, *, candidate_batches=False, curriculum='original', **kwargs):
            super().__init__(**kwargs)
            self._candidate_batches, self._curriculum_mode = candidate_batches, curriculum
            self.validation_mechanisms, self.last_gradients = {}, {}
            self._collect_mechanisms, self._watch_gradients = False, False
            self._gradient_squares, self._gradient_calls = {}, {}
            self._mechanisms = DiagnosticsAccumulator()
            self._gradient_handles = []
            self._gradient_groups = set()
            for name, parameter in self.model.named_parameters():
                group = _gradient_group(name)
                if group is not None and parameter.requires_grad:
                    self._gradient_groups.add(group)
                    def record(gradient, group=group):
                        if self._watch_gradients:
                            norm = gradient.detach().float().square().sum()
                            self._gradient_squares[group] = self._gradient_squares.get(group, 0.)+norm
                            self._gradient_calls[group] = self._gradient_calls.get(group, 0)+1
                    self._gradient_handles.append(parameter.register_hook(record))

        def _prepare_batch(self, batch):
            if self._candidate_batches:
                # Information controls also exist inside the model, so direct
                # API calls cannot accidentally bypass the declared condition.
                information = getattr(getattr(self.model, 'adapter', None), 'information', 'RC')
                return device_batch_v03(batch, self._device, information)
            return super()._prepare_batch(batch)

        def _update_curriculum_length(self):
            if self._curriculum_mode == 'full':
                self._iter_cnt += 1
                self._cl_len = self._horizon
            else:
                super()._update_curriculum_length()

        def train_batch(self):
            self._gradient_squares, self._gradient_calls = {}, {}
            self._watch_gradients = True
            try:
                values = super().train_batch()
            finally:
                self._watch_gradients = False
            self.last_gradients = {key: dict(squared_gradient_sum=(float(self._gradient_squares[key].cpu())
                if key in self._gradient_squares else 0.), backward_parameter_calls=self._gradient_calls.get(key, 0))
                for key in sorted(self._gradient_groups)}
            if any(not np.isfinite(value['squared_gradient_sum']) for value in self.last_gradients.values()):
                raise ValueError('Nonfinite real-objective adapter gradient')
            return values

        def _predict(self, *args):
            prediction = super()._predict(*args)
            if self._collect_mechanisms and hasattr(self.model, 'batch_diagnostics'):
                self._mechanisms.add(self.model.batch_diagnostics())
            return prediction

        def evaluate(self, mode):
            if mode != 'val':
                raise ValueError('Training entry evaluates validation only')
            self._mechanisms = DiagnosticsAccumulator()
            self._collect_mechanisms = True
            try:
                result = super().evaluate(mode)
            finally:
                self._collect_mechanisms = False
            self.validation_mechanisms = self._mechanisms.result()
            return result
    return Engine


def validate_saved(state, identity, arm, run, batches):
    if identity['curriculum'] == 'original':
        return legacy_validate_saved(state, identity, arm, run, batches)
    actual = 0 if state['global_updates'] == 0 else 12
    if state['curriculum_length'] != actual:
        raise ValueError('Full-horizon curriculum state is inconsistent')
    # Reuse identity/hash/permutation/best-checkpoint checks, replacing only
    # the legacy curriculum sentinel after separately validating our contract.
    updates = state['global_updates']
    warm, step = (identity['training_config'][key]*batches for key in ('warm_epoch', 'cl_epoch'))
    legacy_length = 0 if not updates else (12 if updates < warm else min(12, 1+(updates-warm)//step))
    legacy_validate_saved(dict(state, curriculum_length=legacy_length), identity, arm, run, batches)


def run_epochs(engine, run, identity, arm, target_epoch, state=None):
    run = Path(run)
    get_order = tracked_shuffle(engine._dataloader['train_loader'], None if state is None else state['train_order'])
    if state is None:
        state = dict(arm=arm, completed_epoch=0, best_epoch=None, best_validation_mae=None,
            wait=0, best_checkpoint_file=None, best_checkpoint_sha256=None, global_updates=0,
            history=[], initialization=identity['initializations'][arm])
        commit_state(engine, state, run, identity, get_order)
    else:
        engine.model.load_state_dict(state['model_state'], strict=True)
        engine._optimizer.load_state_dict(state['optimizer_state'])
        engine._lr_scheduler.load_state_dict(state['scheduler_state'])
        engine._iter_cnt, engine._cl_len = state['global_updates'], state['curriculum_length']
        restore_rng(state['rng_state'])
    for epoch in range(state['completed_epoch']+1, target_epoch+1):
        print(json.dumps(dict(arm=arm, phase='training', epoch=epoch, target_epochs=target_epoch)), flush=True)
        with batch_progress(engine._dataloader['train_loader'], arm, 'training', epoch):
            (train_mae, train_mape, train_rmse), train_time = timed_phase(engine, engine.train_batch)
        print(json.dumps(dict(arm=arm, phase='validation', epoch=epoch)), flush=True)
        with batch_progress(engine._dataloader['val_loader'], arm, 'validation', epoch):
            (mae, mape, rmse), val_time = timed_phase(engine, lambda: engine.evaluate('val'))
        lr = engine._lr_scheduler.get_last_lr()[0]
        engine._lr_scheduler.step()
        row = dict(epoch=epoch, train_mae=train_mae, train_mape_fraction=train_mape, train_rmse=train_rmse,
            validation_mae=mae, validation_mape_fraction=mape, validation_rmse=rmse,
            train_seconds=train_time['seconds'], validation_seconds=val_time['seconds'],
            train_timing=train_time, validation_timing=val_time, learning_rate=lr,
            global_updates=engine._iter_cnt, curriculum_length=engine._cl_len,
            supervised_horizons=list(range(1, engine._cl_len+1)),
            graph_directly_supervised=engine._cl_len > 3,
            real_objective_gradients=getattr(engine, 'last_gradients', {}),
            mechanisms=getattr(engine, 'validation_mechanisms', {}),
            train_order_sha256=hashlib.sha256(np.asarray(get_order(), dtype=np.int64).tobytes()).hexdigest(),
            test_evaluated=False)
        state['history'].append(row)
        if state['best_validation_mae'] is None or mae < state['best_validation_mae']:
            name = 'best_epoch_{:03d}_{}.pt'.format(epoch, uuid.uuid4().hex[:12])
            atomic_save(run/name, engine.model.state_dict())
            state.update(best_epoch=epoch, best_validation_mae=mae, wait=0,
                         best_checkpoint_file=name, best_checkpoint_sha256=sha256(run/name))
        else:
            state['wait'] += 1
        state['completed_epoch'] = epoch
        commit_state(engine, state, run, identity, get_order)
        write_json(run/'summary.json', arm_summary(state, target_epoch, identity['engineering_only']))
        print(json.dumps(dict(arm=arm, **row)), flush=True)
    return state


def paired_summary(states, identity, target):
    all_present = bool(states) and set(states) == set(identity['arms'])
    common = min((s['completed_epoch'] for s in states.values()), default=0)
    comparisons = {}
    for control, candidate in PAIRS:
        if not (all_present and common and control in states and candidate in states):
            continue
        a, b = (states[arm]['history'][:common] for arm in (control, candidate))
        ba, bb = (min(records, key=lambda row: row['validation_mae']) for records in (a, b))
        comparisons[candidate+'_vs_'+control] = dict(best_epochs={control: ba['epoch'], candidate: bb['epoch']},
            independent_best_at_common_budget={key+'_gain': ba['validation_'+key]-bb['validation_'+key]
                for key in ('mae', 'rmse', 'mape_fraction')},
            common_last_epoch={key+'_gain': a[-1]['validation_'+key]-b[-1]['validation_'+key]
                for key in ('mae', 'rmse', 'mape_fraction')})
    return dict(schema=SCHEMA, status='V03_TRAINING_COMPLETE' if all_present and common >= target else 'V03_TRAINING_PARTIAL',
        engineering_only=identity['engineering_only'], requested_completed_epochs=target,
        common_completed_epochs=common if all_present else None,
        same_completed_training_budget=all_present and len({s['completed_epoch'] for s in states.values()}) == 1,
        arms={a: arm_summary(s, target, identity['engineering_only']) for a,s in states.items()},
        comparisons=comparisons, primary_baseline_present='ORIGINAL' in states,
        protocol=PROTOCOL, seed=identity['training_config']['seed'],
        formal_replication_completed=False, exceeds_baseline_confirmed=False,
        calibration=identity['data'].get('calibration'), curriculum=identity['curriculum'],
        common_initialization=True, optimizer_reset_on_resume=False, test_evaluated=False)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, default=default_data_dir())
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--threads', type=int, default=3)
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--arms', nargs='+', choices=ARMS, default=list(DEFAULT_ARMS))
    parser.add_argument('--only', nargs='+', choices=ARMS)
    parser.add_argument('--curriculum', choices=('original', 'full'), default='original')
    parser.add_argument('--holdout-rows', type=int, default=0)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    if not 1 <= args.epochs <= CONFIG['max_epochs'] or args.threads < 1 or args.holdout_rows < 0:
        parser.error('Invalid epochs, threads, or holdout_rows')
    if len(set(args.arms)) != len(args.arms) or (args.only and (
            len(set(args.only)) != len(args.only) or not set(args.only) <= set(args.arms))):
        parser.error('arms must be unique; only must be a unique registered subset')
    args.run_dir, args.data_dir = args.run_dir.resolve(), args.data_dir.resolve()
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.run_dir.exists() != args.resume:
        raise ValueError('Use a fresh v03 directory or --resume its unchanged identity')
    args.run_dir.mkdir(parents=True, exist_ok=args.resume)
    with run_lock(args.run_dir):
        logger = logging.getLogger('unified-v03-'+str(args.run_dir))
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(args.run_dir/'record.log', encoding='utf-8')
        logger.addHandler(handler)
        try:
            runtime = configure_runtime(args.device, args.threads, args.seed)
            modules = get_original_modules()
            config = dict(CONFIG, seed=args.seed, device=args.device, bs=2 if args.check else CONFIG['bs'])
            data = load_data(args.data_dir, config, modules, logger, check=args.check, training=True)
            audit = audit_data(data)
            data = partition_training(data, modules, logger, args.holdout_rows)
            candidates = make_loaders(data, modules, logger)
            original_samples = tuple(data.loaders['train_loader'].samples)
            candidate_samples = tuple(candidates['train_loader'].samples)
            records = {}
            for arm in args.arms:
                configure_runtime(args.device, args.threads, args.seed)
                model = model_for_arm(data, modules, args.device, arm)
                records[arm] = initialization(model, arm)
                del model
            verify_initializations(records)
            identity = dict(schema=SCHEMA, arms=args.arms, training_config=config, adapter_config=ADAPTER_CONFIG,
                data=data.identity, source_fingerprint=modules.source_fingerprint,
                execution_source={p: sha256(REPO/p) for p in EXECUTION_FILES}, runtime=runtime,
                initializations=records, engineering_only=args.check, curriculum=args.curriculum,
                holdout_rows=args.holdout_rows, data_audit=audit, protocol=PROTOCOL,
                scheduler=dict(milestones=MILESTONES, gamma=.5), early_stopping_enabled=False,
                from_scratch=True, test_evaluated=False)
            identity['identity_sha256'] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
            if args.resume:
                if json.loads((args.run_dir/'identity.json').read_text(encoding='utf-8')) != identity:
                    raise ValueError('Code/data/runtime/protocol identity changed; use a fresh directory')
                if args.epochs < json.loads((args.run_dir/'budget.json').read_text())['target_completed_epochs']:
                    raise ValueError('Cannot decrease an existing epoch budget')
            else:
                write_json(args.run_dir/'identity.json', identity)
                write_json(args.run_dir/'data_audit.json', audit)
                write_json(args.run_dir/'protocol.json', PROTOCOL)
            write_json(args.run_dir/'budget.json', dict(target_completed_epochs=args.epochs, total_not_additional=True))
            states, batches = {}, data.loaders['train_loader'].num_batch
            for arm in args.arms:
                run = args.run_dir/arm
                if (run/'last.pt').is_file():
                    states[arm] = load_checkpoint(run/'last.pt')
                    validate_saved(states[arm], identity, arm, run, batches)
            Engine = engine_class(modules.engine.IGSTGNN_Engine)
            for arm in args.only or args.arms:
                run = args.run_dir/arm
                run.mkdir(exist_ok=True)
                state = states.get(arm)
                if state and state['completed_epoch'] >= args.epochs:
                    continue
                configure_runtime(args.device, args.threads, args.seed)
                data.loaders['train_loader'].samples = list(original_samples)
                candidates['train_loader'].samples = list(candidate_samples)
                model = model_for_arm(data, modules, args.device, arm)
                if initialization(model, arm) != records[arm]:
                    raise ValueError('Actual initialization differs from preflight')
                optimizer = torch.optim.Adam(model.parameters(), lr=config['lrate'], weight_decay=config['wdecay'], eps=1e-8)
                scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=MILESTONES, gamma=.5)
                engine = Engine(candidate_batches=arm != 'ORIGINAL', curriculum=args.curriculum,
                    device=torch.device(args.device), model=model,
                    dataloader=data.loaders if arm == 'ORIGINAL' else candidates,
                    scaler=data.scaler, sampler=None, loss_fn=modules.metrics.masked_mae,
                    lrate=config['lrate'], optimizer=optimizer, scheduler=scheduler,
                    clip_grad_value=config['clip_grad_value'], max_epochs=CONFIG['max_epochs'],
                    patience=config['patience'], log_dir=str(run), logger=logger, seed=args.seed,
                    cl_step=config['cl_epoch']*batches, warm_step=config['warm_epoch']*batches,
                    horizon=12, incident=True)
                states[arm] = run_epochs(engine, run, identity, arm, args.epochs, state)
                write_json(args.run_dir/'summary.json', paired_summary(states, identity, args.epochs))
                del engine, model, optimizer, scheduler
                gc.collect()
                if torch.device(args.device).type == 'cuda':
                    torch.cuda.empty_cache()
            summary = paired_summary(states, identity, args.epochs)
            write_json(args.run_dir/'summary.json', summary)
            print(json.dumps(summary), flush=True)
            return summary
        except BaseException as error:
            write_json(args.run_dir/('failure_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')+'.json'),
                dict(status='V03_TRAINING_FAILED', error_type=type(error).__name__, error=str(error),
                    checkpoint_semantics='last_committed_epoch_replay', test_evaluated=False))
            raise
        finally:
            logger.removeHandler(handler)
            handler.close()


if __name__ == '__main__':
    main()
