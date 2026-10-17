"""Matched from-scratch original/ISP-Lite pilots; original recipe, no test evaluation.

Every arm starts from the same original tensors and RNG/sample order. Epoch
checkpoints include optimizer, scheduler, curriculum and RNG; an interrupted
partial epoch is replayed from its last committed epoch. Patience is recorded
but never truncates the specified common completed-epoch budget.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import gc
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
import time
import uuid

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.chronological.original_igstgnn_source import get_original_modules
from experiments.chronological.original_igstgnn_runtime import (
    CONFIG, build_model, configure_runtime, default_data_dir, load_data, run_lock,
    sha256, tensor_tree_hash, write_json)
from experiments.chronological.original_igstgnn_checkpoint_compat import counter_weights_only_compat
from experiments.chronological.train_original_igstgnn_baseline import (
    MILESTONES, atomic_save, rng_state, restore_rng, tracked_shuffle)
from experiments.chronological.original_isp import ISPIGSTGNN

SCHEMA = 'original_igstgnn_isp_lite_training_v1'
ARMS = ('ORIGINAL', 'STATE', 'STATE_PROPAGATION')
ISP_CONFIG = dict(state_dim=16, graph_rank=8, max_graph_mix=.25, max_state_delta=.1)
EXECUTION_FILES = (
    'experiments/chronological/train_original_isp.py',
    'experiments/chronological/original_isp.py',
    'experiments/chronological/original_igstgnn_runtime.py',
    'experiments/chronological/original_igstgnn_source.py',
    'experiments/chronological/train_original_igstgnn_baseline.py',
    'experiments/chronological/original_igstgnn_checkpoint_compat.py')


def load_checkpoint(path):
    with counter_weights_only_compat():
        return torch.load(path, map_location='cpu', weights_only=True)


def model_for_arm(data, modules, device, arm):
    if arm not in ARMS:
        raise ValueError('Unknown ISP arm')
    original = build_model(data, modules, device)
    return original if arm == 'ORIGINAL' else ISPIGSTGNN(original, mode=arm, **ISP_CONFIG)


def initialization(model, arm):
    original = model if arm == 'ORIGINAL' else model.original
    added = {} if arm == 'ORIGINAL' else {
        name: value for name, value in model.state_dict().items() if not name.startswith('original.')}
    graph_parameters = 0 if arm == 'ORIGINAL' else sum(p.numel() for p in model.graph.parameters())
    return dict(original_tensor_sha256=tensor_tree_hash(original.state_dict()),
        added_tensor_sha256=tensor_tree_hash(added),
        post_construction_rng_sha256=tensor_tree_hash(rng_state()),
        registered_parameters=sum(p.numel() for p in model.parameters()),
        added_registered_parameters=0 if arm == 'ORIGINAL' else sum(
            p.numel() for name, p in model.named_parameters() if not name.startswith('original.')),
        inactive_graph_registered_parameters=graph_parameters if arm == 'STATE' else 0,
        registered_parameter_caveat='equal_registered_tensors_do_not_imply_equal_active_capacity')


def validate_saved(state, identity, arm, run, train_batches):
    """Reject altered identity, contradictory budgets and corrupted best files."""
    if state.get('identity_sha256') != identity['identity_sha256'] or state.get('arm') != arm:
        raise ValueError('Saved ISP identity or arm differs')
    completed, history = state.get('completed_epoch'), state.get('history', [])
    if (type(completed) is not int or not 0 <= completed <= CONFIG['max_epochs']
            or len(history) != completed
            or [r.get('epoch') for r in history] != list(range(1, completed + 1))):
        raise ValueError('Saved ISP epoch history is invalid')
    if state['global_updates'] != completed * train_batches or any(
            row['global_updates'] != row['epoch'] * train_batches for row in history):
        raise ValueError('Saved ISP update budget is inconsistent')
    if any(not np.isfinite([row[key] for key in (
            'validation_mae', 'validation_rmse', 'validation_mape_fraction')]).all() for row in history):
        raise ValueError('Saved ISP validation metrics are nonfinite')
    expected_init = identity['initializations'][arm]
    if state['initialization'] != expected_init:
        raise ValueError('Saved ISP initialization differs')
    length = identity['data']['splits']['train']['count']
    order = np.asarray(state['train_order'])
    if order.shape != (length,) or not np.array_equal(np.sort(order), np.arange(length)):
        raise ValueError('Saved ISP training permutation is invalid')
    warm = identity['training_config']['warm_epoch'] * train_batches
    step = identity['training_config']['cl_epoch'] * train_batches
    updates = state['global_updates']
    curriculum = 0 if updates == 0 else (12 if updates < warm else min(12, 1 + (updates - warm)//step))
    if (state['curriculum_length'] != curriculum
            or state['scheduler_state']['last_epoch'] != completed):
        raise ValueError('Saved ISP curriculum or scheduler is inconsistent')
    if tensor_tree_hash(state['model_state']) != state['model_tensor_sha256']:
        raise ValueError('Saved ISP model tensors are inconsistent')
    if history:
        best = min(history, key=lambda row: row['validation_mae'])
        name = state['best_checkpoint_file']
        if (best['epoch'] != state['best_epoch']
                or best['validation_mae'] != state['best_validation_mae']
                or state['wait'] != completed-state['best_epoch']
                or not isinstance(name, str) or Path(name).name != name
                or sha256(Path(run)/name) != state['best_checkpoint_sha256']):
            raise ValueError('Saved ISP best checkpoint or selection is inconsistent')
    elif any(state[key] is not None for key in (
            'best_epoch', 'best_validation_mae', 'best_checkpoint_file', 'best_checkpoint_sha256')):
        raise ValueError('Epoch-zero ISP state has an invalid best selection')


@contextmanager
def batch_progress(loader, arm, phase, epoch):
    """Show validation progress as well as the unchanged engine train progress."""
    previous = loader.get_iterator
    def iterator():
        for index, batch in enumerate(previous(), 1):
            if index == 1 or index % 10 == 0 or index == loader.num_batch:
                print(json.dumps(dict(arm=arm, phase=phase, epoch=epoch,
                    batch=index, total_batches=loader.num_batch)), flush=True)
            yield batch
    loader.get_iterator = iterator
    try:
        yield
    finally:
        loader.get_iterator = previous


def timed_phase(engine, call):
    device = torch.device(engine._device)
    cuda = device.type == 'cuda'
    if cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        baseline = torch.cuda.memory_allocated(device)
    else:
        baseline = 0
    started = time.perf_counter()
    values = list(map(float, call()))
    if cuda:
        torch.cuda.synchronize(device)
    result = dict(seconds=time.perf_counter()-started,
        peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device)) if cuda else None,
        incremental_peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device)-baseline) if cuda else None,
        peak_reserved_bytes=int(torch.cuda.max_memory_reserved(device)) if cuda else None,
        scope='full_engine_phase_including_loader_H2D_inverse_transform_metrics_and_training_backward')
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite ISP metrics')
    return values, result


def commit_state(engine, state, run, identity, get_order):
    state.update(global_updates=engine._iter_cnt, curriculum_length=engine._cl_len,
        train_order=get_order(), rng_state=rng_state(), model_state=engine.model.state_dict(),
        optimizer_state=engine._optimizer.state_dict(), scheduler_state=engine._lr_scheduler.state_dict(),
        identity_sha256=identity['identity_sha256'])
    state['model_tensor_sha256'] = tensor_tree_hash(state['model_state'])
    atomic_save(Path(run)/'last.pt', state)
    write_json(Path(run)/'history.json', state['history'])


def arm_summary(state, target_epoch, engineering):
    best = next((r for r in state['history'] if r['epoch'] == state['best_epoch']), None)
    timings = {}
    for phase in ('train', 'validation'):
        records = [row[phase+'_timing'] for row in state['history'] if phase+'_timing' in row]
        peaks = [row['peak_allocated_bytes'] for row in records if row['peak_allocated_bytes'] is not None]
        increments = [row['incremental_peak_allocated_bytes'] for row in records
                      if row['incremental_peak_allocated_bytes'] is not None]
        timings[phase] = dict(completed_epochs=len(records), total_seconds=sum(row['seconds'] for row in records),
            mean_seconds_per_epoch=float(np.mean([row['seconds'] for row in records])) if records else None,
            peak_allocated_bytes=max(peaks, default=None), incremental_peak_allocated_bytes=max(increments, default=None),
            scope='full_engine_phase_including_loader_H2D_inverse_transform_metrics_and_training_backward')
    return dict(arm=state['arm'], completed_epoch=state['completed_epoch'],
        target_completed_epochs=target_epoch, target_reached=state['completed_epoch'] >= target_epoch,
        best_epoch=state['best_epoch'], best_validation_mae=state['best_validation_mae'],
        best_checkpoint_file=state['best_checkpoint_file'], best_checkpoint_sha256=state['best_checkpoint_sha256'],
        last_checkpoint_file='last.pt', last_model_tensor_sha256=state['model_tensor_sha256'],
        best_metrics=metric_record(best), last_metrics=metric_record(state['history'][-1]) if state['history'] else None,
        global_updates=state['global_updates'], initialization=state['initialization'], timing=timings,
        engineering_only=engineering, optimizer_reset_on_resume=False,
        best_selection_includes_stage_zero=False, early_stopping_enabled=False,
        test_evaluated=False)


def metric_record(row):
    if row is None:
        return None
    return dict(mae=row['validation_mae'], rmse=row['validation_rmse'],
        mape_fraction=row['validation_mape_fraction'], mape_percent=100*row['validation_mape_fraction'])


def run_epochs(engine, run, identity, arm, target_epoch, state=None):
    run = Path(run)
    get_order = tracked_shuffle(engine._dataloader['train_loader'], None if state is None else state['train_order'])
    if state is None:
        state = dict(arm=arm, completed_epoch=0, best_epoch=None, best_validation_mae=None, wait=0,
            best_checkpoint_file=None, best_checkpoint_sha256=None, global_updates=0,
            history=[], initialization=identity['initializations'][arm])
        commit_state(engine, state, run, identity, get_order)
    else:
        engine.model.load_state_dict(state['model_state'], strict=True)
        engine._optimizer.load_state_dict(state['optimizer_state'])
        engine._lr_scheduler.load_state_dict(state['scheduler_state'])
        engine._iter_cnt = state['global_updates']
        engine._cl_len = state['curriculum_length']
        restore_rng(state['rng_state'])
    for epoch in range(state['completed_epoch']+1, target_epoch+1):
        print(json.dumps(dict(arm=arm, phase='training', epoch=epoch, target_epochs=target_epoch)), flush=True)
        with batch_progress(engine._dataloader['train_loader'], arm, 'training', epoch):
            (train_mae, train_mape, train_rmse), train_timing = timed_phase(engine, engine.train_batch)
        print(json.dumps(dict(arm=arm, phase='validation', epoch=epoch)), flush=True)
        with batch_progress(engine._dataloader['val_loader'], arm, 'validation', epoch):
            (mae, mape, rmse), val_timing = timed_phase(engine, lambda: engine.evaluate('val'))
        lr = engine._lr_scheduler.get_last_lr()[0]
        engine._lr_scheduler.step()
        row = dict(epoch=epoch, train_mae=train_mae, train_mape_fraction=train_mape, train_rmse=train_rmse,
            validation_mae=mae, validation_mape_fraction=mape, validation_rmse=rmse,
            train_seconds=train_timing['seconds'], validation_seconds=val_timing['seconds'],
            train_timing=train_timing, validation_timing=val_timing, learning_rate=lr,
            global_updates=engine._iter_cnt, curriculum_length=engine._cl_len,
            train_order_sha256=hashlib.sha256(np.asarray(get_order(), dtype=np.int64).tobytes()).hexdigest(),
            mechanisms=engine.model.diagnostics() if hasattr(engine.model, 'diagnostics') else {},
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


def paired_summary(states, identity, target_epoch):
    complete = len(states) == len(identity['arms']) and all(
        state['completed_epoch'] >= target_epoch for state in states.values())
    common = min((state['completed_epoch'] for state in states.values()), default=0)
    all_present = set(states) == set(identity['arms'])
    comparisons = {}
    for control, candidate in (('ORIGINAL', 'STATE'), ('STATE', 'STATE_PROPAGATION'), ('ORIGINAL', 'STATE_PROPAGATION')):
        if all_present and common and control in states and candidate in states:
            # Prefixes give an equal completed budget even after --only extends
            # one arm. Do not compare a 10-epoch winner to a 5-epoch winner.
            a = states[control]['history'][:common]
            b = states[candidate]['history'][:common]
            best_a, best_b = min(a, key=lambda r: r['validation_mae']), min(b, key=lambda r: r['validation_mae'])
            comparisons[candidate+'_vs_'+control] = dict(
                independent_best_at_common_budget={
                    metric+'_gain': best_a['validation_'+metric]-best_b['validation_'+metric]
                    for metric in ('mae', 'rmse', 'mape_fraction')},
                best_epochs={control: best_a['epoch'], candidate: best_b['epoch']},
                common_last_epoch={metric+'_gain': a[-1]['validation_'+metric]-b[-1]['validation_'+metric]
                    for metric in ('mae', 'rmse', 'mape_fraction')})
    return dict(schema=SCHEMA, status='ORIGINAL_ISP_TRAINING_PASS' if complete else 'ORIGINAL_ISP_TRAINING_PARTIAL',
        run_kind='from_scratch_matched_original_recipe_pilot' if target_epoch < 100 else 'from_scratch_matched_original_recipe',
        requested_completed_epochs=target_epoch, common_completed_epochs=common if all_present else None,
        same_completed_training_budget=all_present and len({s['completed_epoch'] for s in states.values()}) == 1,
        engineering_only=identity['engineering_only'], arms={arm: arm_summary(state, target_epoch,
            identity['engineering_only']) for arm, state in states.items()}, comparisons=comparisons,
        primary_baseline=('original_IGSTGNN_558ab01_trained_from_scratch_in_same_run'
            if 'ORIGINAL' in identity['arms'] else 'original_IGSTGNN_558ab01_not_present_in_this_run'),
        primary_baseline_present='ORIGINAL' in states,
        common_original_initialization=True, common_added_initialization=True,
        shared_seed_and_initial_train_order=True, optimizer_reset_on_resume=False,
        early_stopping_enabled=False, validation_used_for_model_selection=True,
        test_evaluated=False,
        interpretation='predictive_architecture_comparison_not_identified_incident_causality_or_physics',
        data_boundaries=identity['data'])


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, default=default_data_dir())
    parser.add_argument('--epochs', type=int, default=5, help='Total completed epochs per arm, not additional epochs; default pilot 5')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--threads', type=int, default=3)
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--arms', nargs='+', choices=ARMS, default=list(ARMS))
    parser.add_argument('--only', nargs='+', choices=ARMS, help='Execute only these registered arms, useful for partial resume')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--check', action='store_true', help='Four train/validation rows, batch two; engineering only')
    args = parser.parse_args(argv)
    if not 1 <= args.epochs <= CONFIG['max_epochs']:
        parser.error('epochs must be in 1..100')
    if len(set(args.arms)) != len(args.arms) or (args.only and (
            len(set(args.only)) != len(args.only) or not set(args.only) <= set(args.arms))):
        parser.error('arms must be unique; only must be a unique subset of arms')
    args.run_dir, args.data_dir = args.run_dir.resolve(), args.data_dir.resolve()
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.run_dir.exists() != args.resume:
        raise ValueError('Use a fresh ISP run directory, or --resume an existing run')
    args.run_dir.mkdir(parents=True, exist_ok=args.resume)
    with run_lock(args.run_dir):
        handler = None
        try:
            runtime = configure_runtime(args.device, args.threads, args.seed)
            modules = get_original_modules()
            logger = logging.getLogger('original-isp-'+str(args.run_dir))
            logger.setLevel(logging.INFO)
            handler = logging.FileHandler(args.run_dir/'record.log', encoding='utf-8')
            logger.addHandler(handler)
            config = dict(CONFIG, seed=args.seed, device=args.device)
            if args.check:
                config['bs'] = 2
            data = load_data(args.data_dir, config, modules, logger, check=args.check, training=True)
            original_samples = tuple(data.loaders['train_loader'].samples)
            initializations = {}
            for arm in args.arms:
                configure_runtime(args.device, args.threads, args.seed)
                model = model_for_arm(data, modules, args.device, arm)
                initializations[arm] = initialization(model, arm)
                del model
            if len({v['original_tensor_sha256'] for v in initializations.values()}) != 1:
                raise ValueError('ISP arms do not share original initialization')
            if len({v['post_construction_rng_sha256'] for v in initializations.values()}) != 1:
                raise ValueError('ISP construction changed the common training RNG')
            added = [v['added_tensor_sha256'] for arm, v in initializations.items() if arm != 'ORIGINAL']
            if len(set(added)) > 1:
                raise ValueError('ISP candidate arms do not share new initialization')
            identity = dict(schema=SCHEMA, arms=args.arms, training_config=config, isp_config=ISP_CONFIG,
                data=data.identity, source_fingerprint=modules.source_fingerprint,
                execution_source={p: sha256(REPO/p) for p in EXECUTION_FILES},
                runtime=runtime, initializations=initializations, engineering_only=args.check,
                initial_train_order_sha256=hashlib.sha256(np.arange(len(original_samples), dtype=np.int64).tobytes()).hexdigest(),
                scheduler=dict(milestones=MILESTONES, gamma=.5), early_stopping_enabled=False,
                from_scratch=True, test_evaluated=False)
            identity['identity_sha256'] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
            if args.resume:
                if json.loads((args.run_dir/'identity.json').read_text(encoding='utf-8')) != identity:
                    raise ValueError('ISP run identity differs; use a fresh run directory')
                previous_target = json.loads((args.run_dir/'budget.json').read_text(encoding='utf-8'))['target_completed_epochs']
                if args.epochs < previous_target:
                    raise ValueError('Cannot decrease an existing ISP completed-epoch budget')
            else:
                write_json(args.run_dir/'identity.json', identity)
                write_json(args.run_dir/'runtime.json', runtime)
            write_json(args.run_dir/'budget.json', dict(target_completed_epochs=args.epochs,
                total_not_additional=True, early_stopping_enabled=False, test_evaluated=False))
            states = {}
            batches = data.loaders['train_loader'].num_batch
            for arm in args.arms:
                run = args.run_dir/arm
                if (run/'last.pt').is_file():
                    states[arm] = load_checkpoint(run/'last.pt')
                    validate_saved(states[arm], identity, arm, run, batches)
            for arm in args.only or args.arms:
                run = args.run_dir/arm
                run.mkdir(exist_ok=True)
                state = states.get(arm)
                if state and state['completed_epoch'] >= args.epochs:
                    write_json(run/'summary.json', arm_summary(state, args.epochs, args.check))
                    continue
                configure_runtime(args.device, args.threads, args.seed)
                data.loaders['train_loader'].samples = list(original_samples)
                model = model_for_arm(data, modules, args.device, arm)
                if initialization(model, arm) != initializations[arm]:
                    raise ValueError('Actual ISP initialization differs')
                optimizer = torch.optim.Adam(model.parameters(), lr=config['lrate'], weight_decay=config['wdecay'], eps=1e-8)
                scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=MILESTONES, gamma=.5)
                engine = modules.engine.IGSTGNN_Engine(device=torch.device(args.device), model=model,
                    dataloader=data.loaders, scaler=data.scaler, sampler=None, loss_fn=modules.metrics.masked_mae,
                    lrate=config['lrate'], optimizer=optimizer, scheduler=scheduler,
                    clip_grad_value=config['clip_grad_value'], max_epochs=CONFIG['max_epochs'],
                    patience=config['patience'], log_dir=str(run), logger=logger, seed=args.seed,
                    cl_step=config['cl_epoch']*batches, warm_step=config['warm_epoch']*batches,
                    horizon=12, incident=True)
                print(json.dumps(dict(arm=arm, phase='arm_start', from_scratch=state is None,
                    completed_epoch=0 if state is None else state['completed_epoch'], initialization=initializations[arm])), flush=True)
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
                dict(status='ORIGINAL_ISP_TRAINING_FAILED', error_type=type(error).__name__, error=str(error),
                    checkpoint_semantics='resume_last_committed_epoch_replays_incomplete_epoch', test_evaluated=False))
            raise
        finally:
            if handler is not None:
                logger.removeHandler(handler)
                handler.close()


if __name__ == '__main__':
    main()
