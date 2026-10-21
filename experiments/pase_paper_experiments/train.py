"""Fixed 100-epoch, from-scratch PASE mechanism/ablation training (server only)."""
import argparse
from datetime import datetime, timezone
import gc
from itertools import islice
import json
import logging
import os
from pathlib import Path
import random
import uuid

import numpy as np
import torch

from experiments.position_state_gw16.data import load_data
from experiments.position_state_gw16.objective import metric_contract
from experiments.position_state_gw16.train import run_phase
from .model import build_model, parameter_accounting
from .protocol import CORE_ARMS, arm_spec, configuration, protocol_record
from .state import (SCHEMA, atomic_save, digest_json, epoch_order, exclusive_run,
    expected_batches, load_state, protected_output, read_json, require, restore_rng,
    rng_state, runtime, sha256, source_hashes, state_digest, tensor_tree_hash,
    validate_artifacts, validate_config, validate_identity, validate_runtime,
    validate_saved, write_json)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class LimitedLoader:
    """Engineering-only streaming prefix; never materializes GPU batches."""
    def __init__(self, loader, maximum):
        self.loader = loader
        self.num_batch = min(loader.num_batch, maximum) if maximum is not None else loader.num_batch
        self.size = loader.size

    @property
    def order(self):
        return self.loader.order

    def get_iterator(self):
        return islice(self.loader.get_iterator(), self.num_batch)

    def set_order(self, order):
        self.loader.set_order(order)


def initialization(model):
    value = dict(model_tensor_sha256=tensor_tree_hash(model.state_dict()),
                 parameters=parameter_accounting(model))
    for name in ('encoder', 'incident', 'position', 'origin_condition'):
        module = getattr(model, name, None)
        value[name+'_tensor_sha256'] = tensor_tree_hash(module.state_dict()) if module is not None else None
    return value


def make_identity(data, config, seed, engineering=False, train_limit=None, val_limit=None):
    validate_config(config, engineering)
    initializations = {}
    for arm in CORE_ARMS:
        seed_all(seed)
        model = build_model(data, config, arm, 'cpu')
        initializations[arm] = initialization(model)
        del model
    for name in ('encoder', 'incident', 'position', 'origin_condition'):
        values = {value[name+'_tensor_sha256'] for value in initializations.values()}
        require(len(values) == 1 and None not in values, 'Common initial tensors differ: '+name)
    identity = dict(schema=SCHEMA, config=config, seed=seed, data=data.identity,
        position=data.position_identity, execution_source=source_hashes(),
        initializations=initializations, comparison_arms=list(CORE_ARMS),
        arm_specs={arm: arm_spec(arm) for arm in CORE_ARMS},
        protocol=protocol_record(),
        engineering_only=engineering,
        phase_batch_limits={'train': train_limit, 'val': val_limit},
        metric_contract=metric_contract(config),
        history_scope='PASE_paper_from_scratch_equal_fixed_epoch_budget',
        best_selection='validation_pooled_mae_includes_stage_zero',
        selection_includes_stage_zero=True, all_training_stages='joint',
        optimizer_reset_on_resume=False, from_scratch=True,
        sample_order_policy='RandomState((seed+1009*epoch) mod 2**32); identical across arms',
        training_rng_policy='seed+104729 after construction and stage-zero evaluation',
        loader_generator_policy='published loader has explicit saved order and no independent generator',
        historical_trained_weights_reused=False,
        test_evaluated=False, independent_confirmation=False)
    identity['identity_sha256'] = digest_json(identity)
    validate_identity(identity)
    return identity


def save_best(directory, identity, arm, epoch, model, validation):
    # Only last.pt commits an epoch. Orphan immutable BEST files after a kill
    # remain harmless; resume never overwrites a committed BEST.
    name = 'best_epoch_{:03d}_{}.pt'.format(epoch, uuid.uuid4().hex[:10])
    atomic_save(directory/name, dict(schema=SCHEMA, identity_sha256=identity['identity_sha256'],
        arm=arm, epoch=epoch, stage='joint', model_state=model.state_dict(),
        model_tensor_sha256=tensor_tree_hash(model.state_dict()), validation=validation))
    return dict(best_epoch=epoch, best_metrics=validation, best_checkpoint_file=name,
                best_checkpoint_sha256=sha256(directory/name))


def commit_state(path, state, model, optimizer, scheduler, order, identity, arm):
    state.update(schema=SCHEMA, identity_sha256=identity['identity_sha256'], arm=arm,
        stage='joint', model_state=model.state_dict(),
        model_tensor_sha256=tensor_tree_hash(model.state_dict()),
        optimizer_state=optimizer.state_dict(), scheduler_state=scheduler.state_dict(),
        rng_state=rng_state(), train_order=order.tolist(), loader_generator_state=None)
    state['state_payload_sha256'] = state_digest(state)
    atomic_save(path, state)


def write_history(directory, history):
    """Derived views are regenerated from committed state, so resume cannot duplicate rows."""
    write_json(directory/'history.json', history)
    path = directory/'history.jsonl'
    temporary = path.with_name(path.name+'.tmp')
    with temporary.open('w', encoding='utf-8', newline='\n') as stream:
        for row in history:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def arm_summary(state, config):
    return dict(completed_epoch=state['completed_epoch'],
        fixed_budget_epochs=config['max_epochs'],
        budget_complete=state['completed_epoch'] == config['max_epochs'],
        best_epoch=state['best_epoch'], best_metrics=state['best_metrics'],
        best_checkpoint_file=state['best_checkpoint_file'],
        best_checkpoint_sha256=state['best_checkpoint_sha256'],
        stage_zero_metrics=state['history'][0]['validation'],
        last_metrics=state['history'][-1]['validation'],
        global_updates=state['global_updates'], parameters=state['parameters'],
        train_seconds=sum(row['timing']['train']['seconds'] for row in state['history']))


def collect_summary(run, identity):
    arms = {}
    for arm in CORE_ARMS:
        path = run/arm/'last.pt'
        if path.is_file():
            state = load_state(path)
            validate_saved(state, identity, arm, path.parent)
            arms[arm] = arm_summary(state, identity['config'])
    complete = len(arms) == len(CORE_ARMS) and all(row['budget_complete'] for row in arms.values())
    status = 'PASE_PAPER_TRAINING_COMPLETE' if complete else 'PASE_PAPER_TRAINING_PARTIAL'
    if identity['engineering_only']:
        status = 'PASE_PAPER_ENGINEERING_COMPLETE' if complete else 'PASE_PAPER_ENGINEERING_PARTIAL'
    summary = dict(schema=SCHEMA, status=status, identity_sha256=identity['identity_sha256'],
        seed=identity['seed'], arms=arms, registered_arms=list(CORE_ARMS),
        fixed_full_budget=identity['config']['max_epochs'],
        from_scratch=True, historical_trained_weights_reused=False,
        engineering_only=identity['engineering_only'],
        early_stopping_enabled=False, test_evaluated=False, independent_confirmation=False)
    write_json(run/'summary.json', summary)
    return summary


def train_arm(args, data, config, identity, arm, logger):
    directory = args.run_dir/arm
    directory.mkdir(exist_ok=True)
    path = directory/'last.pt'
    saved = load_state(path) if path.is_file() else None
    batches = expected_batches(identity, 'train')
    if saved is not None:
        validate_saved(saved, identity, arm, directory, batches)
        write_history(directory, saved['history'])
        if saved['completed_epoch'] == config['max_epochs']:
            logger.info('%s completed fixed %s epochs; skipped', arm, config['max_epochs'])
            return
    seed_all(args.seed)
    model = build_model(data, config, arm, args.device)
    require(initialization(model) == identity['initializations'][arm],
            'Model initialization differs from registered common tensors: '+arm)
    model.set_stage('joint')
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'],
        weight_decay=config['weight_decay'], eps=config['adam_eps'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
        milestones=config['lr_milestones'], gamma=config['lr_gamma'])
    train_loader = LimitedLoader(data.loaders['train_loader'], identity['phase_batch_limits']['train'])
    val_loader = LimitedLoader(data.loaders['val_loader'], identity['phase_batch_limits']['val'])
    if saved is not None:
        state = saved
        model.load_state_dict(state['model_state'], strict=True)
        optimizer.load_state_dict(state['optimizer_state'])
        scheduler.load_state_dict(state['scheduler_state'])
        train_loader.set_order(state['train_order'])
        restore_rng(state['rng_state'])
        logger.info('%s resume from committed epoch=%s; optimizer/scheduler/RNG restored',
                    arm, state['completed_epoch'])
    else:
        order = np.arange(train_loader.size, dtype=np.int64)
        train_loader.set_order(order)
        validation, timing = run_phase(model, val_loader, data, config, args.device)
        seed_all(args.seed+104729)
        stage_zero = dict(epoch=0, stage='joint', training=None, validation=validation,
            learning_rate=config['learning_rate'], global_updates=0,
            train_order_sha256=tensor_tree_hash(order),
            timing={'train': {'seconds': 0.}, 'validation': timing})
        state = dict(completed_epoch=0, global_updates=0, history=[stage_zero],
                     parameters=identity['initializations'][arm]['parameters'])
        state.update(save_best(directory, identity, arm, 0, model, validation))
        commit_state(path, state, model, optimizer, scheduler, order, identity, arm)
        write_history(directory, state['history'])
        collect_summary(args.run_dir, identity)
        logger.info('%s epoch=0 val_MAE=%.8f; committed stage-zero BEST', arm, validation['mae'])
    for epoch in range(state['completed_epoch']+1, config['max_epochs']+1):
        order = epoch_order(args.seed, epoch, train_loader.size)
        train_loader.set_order(order)
        logger.info('%s epoch=%s/%s joint training', arm, epoch, config['max_epochs'])
        training, train_time = run_phase(model, train_loader, data, config, args.device, optimizer)
        validation, val_time = run_phase(model, val_loader, data, config, args.device)
        lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        row = dict(epoch=epoch, stage='joint', training=training, validation=validation,
            learning_rate=lr, global_updates=epoch*batches,
            train_order_sha256=tensor_tree_hash(order),
            timing={'train': train_time, 'validation': val_time})
        state['history'].append(row)
        if validation['mae'] < state['best_metrics']['mae']:
            state.update(save_best(directory, identity, arm, epoch, model, validation))
        state.update(completed_epoch=epoch, global_updates=epoch*batches)
        commit_state(path, state, model, optimizer, scheduler, order, identity, arm)
        write_history(directory, state['history'])
        collect_summary(args.run_dir, identity)
        logger.info('%s epoch=%s val_MAE=%.8f bias=%s region_MAE=%s train_s=%.2f',
            arm, epoch, validation['mae'], validation.get('signed_bias'),
            validation.get('region_mae'), train_time['seconds'])
    del model, optimizer, scheduler, state, saved
    gc.collect()
    if torch.device(args.device).type == 'cuda':
        torch.cuda.empty_cache()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--arms', nargs='+', default=list(CORE_ARMS))
    parser.add_argument('--seed', type=int, choices=(2025, 2026, 2027), default=2025)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--threads', type=int, default=3)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--engineering', action='store_true')
    parser.add_argument('--engineering-epochs', type=int, default=2)
    parser.add_argument('--engineering-max-train-batches', type=int, default=2)
    parser.add_argument('--engineering-max-val-batches', type=int, default=2)
    args = parser.parse_args(argv)
    args.arms = [part for value in args.arms for part in value.split(',') if part]
    require(args.arms and len(set(args.arms)) == len(args.arms)
            and set(args.arms) <= set(CORE_ARMS), 'Unknown or duplicate comparison arms')
    require(args.threads > 0, 'Threads must be positive')
    require(1 <= args.engineering_epochs <= 3
            and args.engineering_max_train_batches > 0 and args.engineering_max_val_batches > 0,
            'Invalid engineering-only budget')
    require(args.engineering or (args.engineering_epochs == 2
            and args.engineering_max_train_batches == 2 and args.engineering_max_val_batches == 2),
            'Engineering options require --engineering; formal budget is fixed at 100 full epochs')
    args.data_dir = args.data_dir.resolve()
    args.run_dir = protected_output(args.run_dir, args.data_dir)
    return args


def main(argv=None):
    args = parse_args(argv)
    config = configuration('FULL')
    if args.engineering:
        config.update(max_epochs=args.engineering_epochs, lr_milestones=[])
    validate_config(config, args.engineering)
    if args.resume:
        require((args.run_dir/'identity.json').is_file() and (args.run_dir/'runtime.json').is_file(),
            'No committed identity/runtime: preserve partial initialization and use a fresh directory')
        validate_identity(read_json(args.run_dir/'identity.json'))
    elif args.run_dir.exists():
        remaining = [path.name for path in args.run_dir.iterdir()
                     if path.name not in ('active.lock', 'run.lock')]
        require(not remaining,
            'Run directory already contains artifacts. If identity.json exists use --resume; '
            'otherwise preserve this partial initialization and select a fresh run directory. '
            'Existing artifacts: '+', '.join(remaining))
    args.run_dir.mkdir(parents=True, exist_ok=True)
    with exclusive_run(args.run_dir):
        logger = logging.getLogger('pase-paper-'+str(args.run_dir))
        logger.setLevel(logging.INFO)
        handlers = [logging.FileHandler(args.run_dir/'record.log', encoding='utf-8'), logging.StreamHandler()]
        for handler in handlers:
            logger.addHandler(handler)
        try:
            environment = runtime(args.device, args.threads, args.seed)
            environment.update(conda_default_env=os.environ.get('CONDA_DEFAULT_ENV'),
                               conda_prefix=os.environ.get('CONDA_PREFIX'))
            data = load_data(args.data_dir, config, logger, engineering=args.engineering)
            identity = make_identity(data, config, args.seed, args.engineering,
                args.engineering_max_train_batches if args.engineering else None,
                args.engineering_max_val_batches if args.engineering else None)
            if args.resume:
                previous = read_json(args.run_dir/'identity.json')
                validate_identity(previous)
                require(previous == identity, 'Data, source, initialization or protocol changed; cannot resume')
                validate_runtime(read_json(args.run_dir/'runtime.json'), environment)
                validate_artifacts(args.run_dir, identity, data)
            else:
                write_json(args.run_dir/'runtime.json', environment)
                write_json(args.run_dir/'environment.json', environment)
                write_json(args.run_dir/'partition.json', data.partition)
                write_json(args.run_dir/'position.json', data.position_identity)
                write_json(args.run_dir/'config.json', config)
                write_json(args.run_dir/'metric_contract.json', identity['metric_contract'])
                write_json(args.run_dir/'budget.json', dict(total_epochs=config['max_epochs'],
                    total_not_additional=True, early_stopping=False,
                    selection_includes_stage_zero=True, engineering_only=args.engineering,
                    phase_batch_limits=identity['phase_batch_limits']))
                # Written last: partial initialization cannot masquerade as a committed run.
                write_json(args.run_dir/'identity.json', identity)
            for arm in args.arms:
                train_arm(args, data, config, identity, arm, logger)
            require(source_hashes() == identity['execution_source'], 'Training source changed during execution')
            result = collect_summary(args.run_dir, identity)
            logger.info('%s: %s', result['status'], args.run_dir/'summary.json')
            return result
        except BaseException as error:
            write_json(args.run_dir/('failure_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')+'.json'),
                dict(status='PASE_PAPER_TRAINING_FAILED', error_type=type(error).__name__,
                    error=str(error), resume_boundary='last_committed_epoch_including_zero',
                    original_directory_preserved=True, test_evaluated=False))
            raise
        finally:
            for handler in handlers:
                logger.removeHandler(handler)
                handler.close()


if __name__ == '__main__':
    main()
