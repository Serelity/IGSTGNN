"""Frozen GW16 STATE_EXCHANGE on author releases: one model per dataset.

No Python command in this package is intended to run on the development PC.
The server owns numerical tests, engineering checks and research training.
"""
import argparse
from datetime import datetime, timezone
import gc
import logging
from pathlib import Path
import random
import time
import uuid

import numpy as np
import torch

from . import ARMS
from .common import (SCHEMA, atomic_save, digest_json, exclusive_run, load_config,
    load_state, phase_for, protected_output, read_json, require, restore_rng,
    rng_state, runtime, sha256, source_hashes, synchronize, tensor_tree_hash,
    validate_config, write_json)
from .data import device_batch, load_data
from .model import build_model
from experiments.position_state_gw16.objective import PooledStatistics, loss_terms, metric_contract


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def initialization(model):
    result = dict(model_tensor_sha256=tensor_tree_hash(model.state_dict()),
        registered_parameters=sum(v.numel() for v in model.parameters()),
        joint_trainable_parameters=sum(v.numel() for v in model.parameters() if v.requires_grad),
        capacity=getattr(model, 'capacity', None))
    for name in ('encoder', 'incident', 'position', 'origin_condition'):
        module = getattr(model, name, None)
        result[name+'_tensor_sha256'] = tensor_tree_hash(module.state_dict()) if module is not None else None
    return result


def make_identity(data, config, seed, engineering):
    validate_config(config, engineering=engineering)
    initializations = {}
    for arm in ARMS:
        seed_all(seed)
        model = build_model(data, config, arm, 'cpu')
        initializations[arm] = initialization(model)
        del model

    identity = dict(schema=SCHEMA, config=config, seed=seed, data=data.identity,
        position=data.position_identity,
        execution_source=source_hashes(), initializations=initializations,
        comparison_arms=list(ARMS), engineering_only=engineering,
        metric_contract=metric_contract(config), history_scope='unchanged_GW16_STATE_EXCHANGE_per_dataset_from_scratch',
        best_selection='validation_pooled_mae_includes_stage_zero',
        selection_includes_stage_zero=True, all_training_stages='joint',
        optimizer_reset_between_stages=False, optimizer_reset_on_resume=False,
        sample_order_policy='RandomState((seed+1009*epoch) mod 2**32); identical across arms',
        training_rng_policy='seed+104729 after model construction and stage-zero evaluation',
        old_baseline_new_training_updates=0, old_baseline_new_inference_calls=0,
        test_evaluated=False, independent_confirmation=False)
    identity['identity_sha256'] = digest_json(identity)
    return identity


def validate_identity(identity):
    payload = dict(identity)
    expected = payload.pop('identity_sha256', None)
    require(expected == digest_json(payload) and identity.get('schema') == SCHEMA,
            'Invalid multi-dataset GW16 run identity')
    validate_config(identity['config'], engineering=identity['engineering_only'])
    require(identity.get('comparison_arms') == list(ARMS)
            and identity.get('metric_contract') == metric_contract(identity['config'])
            and identity.get('selection_includes_stage_zero') is True,
            'Multi-dataset GW16 identity roles, metrics or selection differ')


def epoch_order(seed, epoch, size):
    require(epoch >= 1, 'Only trained epochs are shuffled')
    return np.random.RandomState((seed+1009*epoch) % (2**32)).permutation(size)


def validate_runtime(previous, current):
    keys = ('torch', 'numpy', 'threads', 'dtype', 'cuda', 'device_name', 'python_device',
            'deterministic_algorithms', 'tf32', 'cudnn_benchmark', 'python', 'seed',
            'scipy', 'cudnn', 'cublas_workspace_config', 'default_dtype')
    require(all(key in previous and key in current and previous[key] == current[key] for key in keys),
            'Resume requires the recorded numerical runtime, device, threads and seed')


def validate_artifacts(directory, identity, data):
    expected = {'config.json':identity['config'], 'partition.json':data.partition,
                'position.json':data.position_identity,
                'metric_contract.json':identity['metric_contract']}
    for name, value in expected.items():
        require((directory/name).is_file() and read_json(directory/name) == value,
                'Run artifact missing or changed: '+name)


def state_digest(state):
    return tensor_tree_hash({key:value for key, value in state.items() if key != 'state_payload_sha256'})


def validate_saved(state, identity, arm, directory, batches):
    validate_identity(identity)
    require(state.get('schema') == SCHEMA and state.get('identity_sha256') == identity['identity_sha256']
            and state.get('arm') == arm, 'Checkpoint run/arm identity mismatch')
    require(state.get('state_payload_sha256') == state_digest(state), 'Checkpoint state payload changed')
    epoch, history = state['completed_epoch'], state['history']
    require(type(epoch) is int and 0 <= epoch <= identity['config']['max_epochs']
            and len(history) == epoch+1 and [r['epoch'] for r in history] == list(range(epoch+1)),
            'Invalid checkpoint epoch history, including stage zero')
    require(state['global_updates'] == epoch*batches
            and all(r['global_updates'] == r['epoch']*batches and r['stage'] == 'joint' for r in history)
            and history[0]['training'] is None, 'Checkpoint update count/stage history differs')
    config = identity['config']
    require(state['stage'] == 'joint' and state['scheduler_state']['last_epoch'] == epoch,
            'Checkpoint scheduler/phase is inconsistent')
    expected_lr = config['learning_rate']*config['lr_gamma']**sum(v <= epoch for v in config['lr_milestones'])
    require(all(abs(group['lr']-expected_lr) <= 1e-15 for group in state['optimizer_state']['param_groups']),
            'Checkpoint optimizer learning rate differs from uninterrupted schedule')
    require(tensor_tree_hash(state['model_state']) == state['model_tensor_sha256'], 'Model tensors changed')
    expected_parameters = identity['initializations'][arm]
    require(state['registered_parameters'] == expected_parameters['registered_parameters']
            and state['joint_trainable_parameters'] == expected_parameters['joint_trainable_parameters'],
            'Checkpoint parameter accounting differs')
    order = np.asarray(state['train_order'])
    length = identity['data']['published']['used_counts']['train_loader']
    expected_order = epoch_order(identity['seed'], epoch, length) if epoch else np.arange(length)
    require(order.shape == (length,) and np.array_equal(order, expected_order),
            'Checkpoint sample order differs from the registered epoch permutation')
    for row in history[1:]:
        expected = epoch_order(identity['seed'], row['epoch'], length)
        lr_used = config['learning_rate']*config['lr_gamma']**sum(v < row['epoch'] for v in config['lr_milestones'])
        require(row['train_order_sha256'] == tensor_tree_hash(expected)
                and abs(row['learning_rate']-lr_used) <= 1e-15,
                'Saved epoch sample order or learning rate differs')
    best = min(history, key=lambda row:row['validation']['mae'])
    require(state['best_epoch'] == best['epoch'] and state['best_metrics'] == best['validation'],
            'BEST selection differs from pooled validation MAE including stage zero')
    name = state['best_checkpoint_file']
    require(isinstance(name, str) and Path(name).name == name
            and sha256(Path(directory)/name) == state['best_checkpoint_sha256'],
            'BEST checkpoint missing or changed')


def run_phase(model, loader, data, config, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    stats = PooledStatistics(config)
    if torch.device(device).type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
        baseline = torch.cuda.memory_allocated(device)
    else:
        baseline = 0
    synchronize(device)
    started = time.perf_counter()
    max_state, total_loss, batches, max_grad = 0., 0., 0, 0.
    for raw in loader.get_iterator():
        history, target, event = device_batch(raw, data, device)
        with torch.set_grad_enabled(training):
            if training:
                optimizer.zero_grad(set_to_none=True)
            output = model(history, event)
            require(all(bool(torch.isfinite(output[key]).all()) for key in
                ('prediction', 'reference', 'regional', 'alpha', 'beta', 'state_rms')),
                'Nonfinite model output/state')
            losses = loss_terms(output, target, model.geometry, config, stage='joint')
            require(bool(torch.isfinite(losses['loss'])), 'Nonfinite training/validation loss')
            if training:
                losses['loss'].backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['clip_grad_norm'],
                                                      error_if_nonfinite=True)
                max_grad = max(max_grad, float(norm))
                optimizer.step()
        with torch.no_grad():
            stats.update(output, target, model.geometry)
            max_state = max(max_state, float(output['state_rms'].max()))
        total_loss += float(losses['loss'].detach())
        batches += 1
    require(batches == loader.num_batch and batches > 0, 'Incomplete or empty phase loader')
    synchronize(device)
    seconds = time.perf_counter()-started
    peak = torch.cuda.max_memory_allocated(device) if torch.device(device).type == 'cuda' else 0
    return dict(**stats.result(), mean_batch_loss=total_loss/batches, batches=batches,
        max_state_rms=max_state, max_preclip_gradient_norm=max_grad if training else None), dict(
        seconds=seconds, peak_allocated_bytes=peak, incremental_peak_allocated_bytes=max(0, peak-baseline),
        scope='full_phase_loader_H2D_forward_raw_unit_losses_metrics_and_training_backward')


def arm_summary(state, target, config):
    return dict(completed_epoch=state['completed_epoch'], target_completed_epochs=target,
        target_reached=state['completed_epoch'] == target,
        full_protocol_reached=state['completed_epoch'] == config['max_epochs'],
        stage=state['stage'], best_epoch=state['best_epoch'], best_metrics=state['best_metrics'],
        best_checkpoint_file=state['best_checkpoint_file'], best_checkpoint_sha256=state['best_checkpoint_sha256'],
        stage_zero_metrics=state['history'][0]['validation'],
        last_metrics=state['history'][-1]['validation'], global_updates=state['global_updates'],
        registered_parameters=state['registered_parameters'],
        joint_trainable_parameters=state['joint_trainable_parameters'],
        train_seconds=sum(row['timing']['train']['seconds'] for row in state['history']),
        validation_seconds=sum(row['timing']['validation']['seconds'] for row in state['history']))


def collect_summary(run, identity, target, batches):
    arms = {}
    for arm in ARMS:
        path = run/arm/'last.pt'
        if path.is_file():
            state = load_state(path)
            validate_saved(state, identity, arm, path.parent, batches)
            arms[arm] = arm_summary(state, target, identity['config'])
    complete = len(arms) == len(ARMS) and all(record['full_protocol_reached'] for record in arms.values())
    summary = dict(schema=SCHEMA, status='STATE_EXCHANGE_MULTIDATA_COMPLETE' if complete else 'STATE_EXCHANGE_MULTIDATA_PARTIAL',
        identity_sha256=identity['identity_sha256'], arms=arms, from_scratch=True,
        fixed_full_budget=identity['config']['max_epochs'], requested_total_epochs=target,
        engineering_only=identity['engineering_only'], early_stopping_enabled=False,
        best_selection='validation_pooled_mae_includes_stage_zero',
        historical_runs_reused_as_training_sources=False, old_baseline_new_training_updates=0,
        old_baseline_new_inference_calls=0, test_evaluated=False, independent_confirmation=False)
    write_json(run/'summary.json', summary)
    return summary


def commit_state(path, state, model, optimizer, scheduler, order, identity, arm):
    state.update(schema=SCHEMA, identity_sha256=identity['identity_sha256'], arm=arm,
        stage='joint', model_state=model.state_dict(),
        model_tensor_sha256=tensor_tree_hash(model.state_dict()), optimizer_state=optimizer.state_dict(),
        scheduler_state=scheduler.state_dict(), rng_state=rng_state(), train_order=order.tolist())
    state['state_payload_sha256'] = state_digest(state)
    atomic_save(path, state)


def save_best(directory, identity, arm, epoch, model, validation):
    # last.pt alone commits an epoch. An interruption may leave this immutable
    # file orphaned; retry uses a new name and never overwrites a committed BEST.
    name = 'best_epoch_{:03d}_{}.pt'.format(epoch, uuid.uuid4().hex[:10])
    atomic_save(directory/name, dict(schema=SCHEMA, identity_sha256=identity['identity_sha256'],
        arm=arm, epoch=epoch, stage='joint', model_state=model.state_dict(), validation=validation))
    return dict(best_epoch=epoch, best_metrics=validation, best_checkpoint_file=name,
                best_checkpoint_sha256=sha256(directory/name))


def train_arm(args, data, config, identity, arm, logger):
    directory = args.run_dir/arm
    directory.mkdir(exist_ok=True)
    path = directory/'last.pt'
    saved = load_state(path) if path.is_file() else None
    batches = data.loaders['train_loader'].num_batch
    if saved is not None:
        validate_saved(saved, identity, arm, directory, batches)
        require(saved['completed_epoch'] <= args.epochs, 'Cannot reduce a completed run budget')
        if saved['completed_epoch'] == args.epochs:
            logger.info('%s already completed %s epochs; skipped', arm, args.epochs)
            return
    seed_all(args.seed)
    model = build_model(data, config, arm, args.device)
    require(initialization(model) == identity['initializations'][arm], 'Initialization changed: '+arm)
    model.set_stage('joint')
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'],
        weight_decay=config['weight_decay'], eps=config['adam_eps'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
        milestones=config['lr_milestones'], gamma=config['lr_gamma'])
    seed_all(args.seed+104729)
    if saved is not None:
        state = saved
        model.load_state_dict(state['model_state'], strict=True)
        optimizer.load_state_dict(state['optimizer_state'])
        scheduler.load_state_dict(state['scheduler_state'])
        data.loaders['train_loader'].set_order(state['train_order'])
        restore_rng(state['rng_state'])
    else:
        order = np.arange(data.loaders['train_loader'].size, dtype=np.int64)
        data.loaders['train_loader'].set_order(order)
        validation, timing = run_phase(model, data.loaders['val_loader'], data, config, args.device)
        # Constructor and validation RNG consumption cannot alter the common
        # initial training stream. Subsequent resume restores the committed RNG.
        seed_all(args.seed+104729)
        stage_zero = dict(epoch=0, stage='joint', training=None, validation=validation,
            learning_rate=config['learning_rate'], global_updates=0,
            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            train_order_sha256=tensor_tree_hash(order),
            timing=dict(train=dict(seconds=0.), validation=timing))
        initial = identity['initializations'][arm]
        state = dict(completed_epoch=0, global_updates=0, history=[stage_zero],
            registered_parameters=initial['registered_parameters'],
            joint_trainable_parameters=initial['joint_trainable_parameters'])
        state.update(save_best(directory, identity, arm, 0, model, validation))
        commit_state(path, state, model, optimizer, scheduler, order, identity, arm)
        write_json(directory/'history.json', state['history'])
        collect_summary(args.run_dir, identity, args.epochs, batches)
        logger.info('%s epoch=0 val_MAE=%.8f; committed stage-zero BEST', arm, validation['mae'])
    for epoch in range(state['completed_epoch']+1, args.epochs+1):
        model.set_stage('joint')
        order = epoch_order(args.seed, epoch, data.loaders['train_loader'].size)
        data.loaders['train_loader'].set_order(order)
        logger.info('%s epoch=%s stage=joint training', arm, epoch)
        training, train_time = run_phase(model, data.loaders['train_loader'], data, config, args.device, optimizer)
        validation, val_time = run_phase(model, data.loaders['val_loader'], data, config, args.device)
        lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        row = dict(epoch=epoch, stage='joint', training=training, validation=validation,
            learning_rate=lr, global_updates=epoch*batches,
            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            train_order_sha256=tensor_tree_hash(order), timing=dict(train=train_time, validation=val_time))
        state['history'].append(row)
        if validation['mae'] < state['best_metrics']['mae']:
            state.update(save_best(directory, identity, arm, epoch, model, validation))
        state.update(completed_epoch=epoch, global_updates=epoch*batches)
        commit_state(path, state, model, optimizer, scheduler, order, identity, arm)
        write_json(directory/'history.json', state['history'])
        collect_summary(args.run_dir, identity, args.epochs, batches)
        logger.info('%s epoch=%s val_MAE=%.8f bias=%s region_MAE=%s train_s=%.2f',
            arm, epoch, validation['mae'], validation.get('signed_bias'),
            validation.get('region_mae'), train_time['seconds'])
    del model, optimizer, scheduler, state, saved
    gc.collect()
    if torch.device(args.device).type == 'cuda':
        torch.cuda.empty_cache()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=('Alameda', 'Contra_Costa', 'Orange'), required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--arms', nargs='+', choices=ARMS, default=list(ARMS))
    parser.add_argument('--epochs', type=int, help='Total completed epochs, not additional epochs')
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--threads', type=int, default=3)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--engineering', action='store_true', help='Four train/four validation windows, three epochs; engineering only')
    args = parser.parse_args(argv)
    require(args.seed == 2025 and len(set(args.arms)) == len(args.arms),
            'GW16 fixes seed=2025 and disallows duplicate arms')
    require(args.threads > 0, 'Threads must be positive')
    args.data_dir = args.data_dir.resolve()
    args.run_dir = protected_output(args.run_dir, args.data_dir)
    return args


def main(argv=None):
    args = parse_args(argv)
    config = load_config(args.dataset, args.config)
    if args.engineering:
        config.update(max_epochs=3, lr_milestones=[])
    validate_config(config, engineering=args.engineering)
    args.epochs = config['max_epochs'] if args.epochs is None else args.epochs
    require(1 <= args.epochs <= config['max_epochs'], 'Epoch target outside fixed protocol budget')
    require(args.run_dir.exists() == args.resume, 'Use a fresh run directory, or --resume an existing run')
    if args.resume:
        require((args.run_dir/'identity.json').is_file() and (args.run_dir/'runtime.json').is_file(),
                'No committed run identity/runtime: initialization was interrupted; preserve it and use a fresh directory')
        validate_identity(read_json(args.run_dir/'identity.json'))
    args.run_dir.mkdir(parents=True, exist_ok=args.resume)
    with exclusive_run(args.run_dir):
        logger = logging.getLogger('gw16-'+str(args.run_dir))
        logger.setLevel(logging.INFO)
        handlers = [logging.FileHandler(args.run_dir/'record.log', encoding='utf-8'), logging.StreamHandler()]
        for handler in handlers:
            logger.addHandler(handler)
        try:
            environment = runtime(args.device, args.threads, args.seed)
            data = load_data(args.data_dir, config, logger, engineering=args.engineering)
            identity = make_identity(data, config, args.seed, args.engineering)
            if args.resume:
                saved_identity = read_json(args.run_dir/'identity.json')
                validate_identity(saved_identity)
                require(saved_identity == identity, 'Data, position, source or protocol changed; cannot resume')
                previous = read_json(args.run_dir/'runtime.json')
                validate_runtime(previous, environment)
                validate_artifacts(args.run_dir, identity, data)
                completed = [load_state(path)['completed_epoch'] for path in args.run_dir.glob('*/last.pt')]
                require(not completed or max(completed) <= args.epochs,
                        'A saved arm exceeds the requested total budget; do not reduce it')
            else:
                write_json(args.run_dir/'runtime.json', environment)
                write_json(args.run_dir/'partition.json', data.partition)
                write_json(args.run_dir/'position.json', data.position_identity)
                write_json(args.run_dir/'config.json', config)
                write_json(args.run_dir/'metric_contract.json', metric_contract(config))
                # Identity is the initialization commit, after every required
                # companion artifact. A partial setup cannot silently resume.
                write_json(args.run_dir/'identity.json', identity)
            write_json(args.run_dir/'budget.json', dict(total_epochs=args.epochs,
                maximum_epochs=config['max_epochs'], total_not_additional=True,
                early_stopping=False, selection_includes_stage_zero=True))
            for arm in args.arms:
                train_arm(args, data, config, identity, arm, logger)
            require(source_hashes() == identity['execution_source'], 'Source files changed during training')
            summary = collect_summary(args.run_dir, identity, args.epochs, data.loaders['train_loader'].num_batch)
            logger.info('Finished requested arms. Summary: %s', args.run_dir/'summary.json')
            return summary
        except BaseException as error:
            write_json(args.run_dir/('failure_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')+'.json'),
                dict(status='STATE_EXCHANGE_MULTIDATA_FAILED', error_type=type(error).__name__, error=str(error),
                    resume_boundary='last_committed_epoch_including_zero', test_evaluated=False))
            raise
        finally:
            for handler in handlers:
                logger.removeHandler(handler)
                handler.close()


if __name__ == '__main__':
    main()
