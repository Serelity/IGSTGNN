"""Train one frozen original IGSTGNN baseline; never evaluate a test loader."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import random
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

SCHEMA = 'original_igstgnn_baseline_training_v1'
MILESTONES = [1, 38, 46, 54, 62, 70, 80]


def atomic_save(path, payload):
    temp = path.with_suffix(path.suffix+'.tmp')
    torch.save(payload, temp)
    temp.replace(path)


def rng_state():
    state = np.random.get_state()
    return dict(torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        numpy=[state[0], state[1].tolist(), state[2], state[3], state[4]], python=random.getstate())


def restore_rng(state):
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:
        torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda']])
    value = state['numpy']
    np.random.set_state((value[0], np.asarray(value[1], dtype=np.uint32), value[2], value[3], value[4]))
    random.setstate(state['python'])


def tracked_shuffle(loader, saved_order=None):
    original = tuple(loader.samples)
    order = np.arange(len(original), dtype=np.int64) if saved_order is None else np.asarray(saved_order, dtype=np.int64)
    if order.shape != (len(original),) or not np.array_equal(np.sort(order), np.arange(len(original))):
        raise ValueError('Invalid saved train sample permutation')
    loader.samples = [original[i] for i in order]
    def shuffle():
        nonlocal order
        order = order[np.random.permutation(len(order))]
        loader.samples = [original[i] for i in order]
    loader.shuffle = shuffle
    return lambda: order.tolist()


def summary_for(state, identity):
    complete = state['completed_epoch'] >= CONFIG['max_epochs'] or state['wait'] >= CONFIG['patience']
    return dict(schema=SCHEMA, status='ORIGINAL_IGSTGNN_BASELINE_COMPLETE' if complete else 'ORIGINAL_IGSTGNN_BASELINE_PARTIAL',
        completed_epoch=state['completed_epoch'], best_epoch=state['best_epoch'],
        best_validation_mae=state['best_validation_mae'], best_checkpoint_file=state['best_checkpoint_file'],
        best_checkpoint_sha256=state['best_checkpoint_sha256'], global_updates=state['global_updates'],
        wait=state['wait'], engineering_only=identity['engineering_only'],
        best_selection_includes_stage_zero=False, registered_parameters=state['registered_parameters'],
        training_config=identity['training_config'], source_fingerprint=identity['source_fingerprint'],
        test_evaluated=False)


def validate_saved(state, identity, run, train_batches):
    """Reject contradictory epoch/selection/order state before any updates."""
    if state.get('identity_sha256') != identity['identity_sha256']:
        raise ValueError('Saved training state identity differs')
    completed = state.get('completed_epoch')
    history = state.get('history', [])
    if (type(completed) is not int or not 1 <= completed <= 100 or len(history) != completed
            or [row.get('epoch') for row in history] != list(range(1, completed+1))
            or not all(np.isfinite(row['validation_mae']) for row in history)):
        raise ValueError('Saved training history is invalid')
    best = min(history, key=lambda row: row['validation_mae'])
    if (best['epoch'] != state['best_epoch'] or best['validation_mae'] != state['best_validation_mae']
            or state['wait'] != completed-state['best_epoch']
            or state['global_updates'] != completed*train_batches
            or [row['global_updates'] for row in history] != [epoch*train_batches for epoch in range(1, completed+1)]
            or state['registered_parameters'] != 455220):
        raise ValueError('Saved best selection or update budget is inconsistent')
    length = identity['data']['splits']['train']['count']
    order = np.asarray(state['train_order'])
    if order.shape != (length,) or not np.array_equal(np.sort(order), np.arange(length)):
        raise ValueError('Saved training permutation is invalid')
    warm = identity['training_config']['warm_epoch']*train_batches
    step = identity['training_config']['cl_epoch']*train_batches
    updates = state['global_updates']
    curriculum = 12 if updates < warm else min(12, 1+(updates-warm)//step)
    if state['curriculum_length'] != curriculum or state['scheduler_state']['last_epoch'] != completed:
        raise ValueError('Saved curriculum/scheduler epoch is inconsistent')
    name = state['best_checkpoint_file']
    if Path(name).name != name or sha256(Path(run)/name) != state['best_checkpoint_sha256']:
        raise ValueError('Saved best checkpoint fingerprint differs')


def run_epochs(engine, run, identity, target_epoch, state=None):
    """Original engine updates/evaluation, with a test-free outer loop."""
    run = Path(run)
    get_order = tracked_shuffle(engine._dataloader['train_loader'], None if state is None else state['train_order'])
    if state is None:
        state = dict(completed_epoch=0, best_epoch=None, best_validation_mae=float('inf'), wait=0,
                     global_updates=0, history=[], registered_parameters=sum(p.numel() for p in engine.model.parameters()))
    else:
        engine.model.load_state_dict(state['model_state'], strict=True)
        engine._optimizer.load_state_dict(state['optimizer_state'])
        engine._lr_scheduler.load_state_dict(state['scheduler_state'])
        engine._iter_cnt = state['global_updates']
        engine._cl_len = state['curriculum_length']
        restore_rng(state['rng_state'])
    def synchronize():
        if torch.device(engine._device).type == 'cuda':
            torch.cuda.synchronize(engine._device)
    for epoch in range(state['completed_epoch']+1, target_epoch+1):
        if state['wait'] >= engine._patience:
            break
        print(json.dumps(dict(phase='training', epoch=epoch, total_batches=engine._dataloader['train_loader'].num_batch)), flush=True)
        synchronize()
        started = time.perf_counter()
        train_mae, train_mape, train_rmse = map(float, engine.train_batch())
        synchronize()
        train_seconds = time.perf_counter()-started
        print(json.dumps(dict(phase='validation', epoch=epoch, total_batches=engine._dataloader['val_loader'].num_batch)), flush=True)
        synchronize()
        started = time.perf_counter()
        val_mae, val_mape, val_rmse = map(float, engine.evaluate('val'))
        synchronize()
        validation_seconds = time.perf_counter()-started
        if not np.isfinite([train_mae, train_mape, train_rmse, val_mae, val_mape, val_rmse]).all():
            raise ValueError('Nonfinite original baseline metrics')
        lr = engine._lr_scheduler.get_last_lr()[0]
        engine._lr_scheduler.step()
        row = dict(epoch=epoch, train_mae=train_mae, train_mape_fraction=train_mape, train_rmse=train_rmse,
            validation_mae=val_mae, validation_mape_fraction=val_mape, validation_rmse=val_rmse,
            train_seconds=train_seconds, validation_seconds=validation_seconds, learning_rate=lr,
            global_updates=engine._iter_cnt, curriculum_length=engine._cl_len,
            train_order_sha256=hashlib.sha256(np.asarray(get_order(), dtype=np.int64).tobytes()).hexdigest())
        state['history'].append(row)
        if val_mae < state['best_validation_mae']:
            best_name = 'best_epoch_{:03d}_{}.pt'.format(epoch, uuid.uuid4().hex[:12])
            atomic_save(run/best_name, engine.model.state_dict())
            state.update(best_epoch=epoch, best_validation_mae=val_mae, wait=0,
                         best_checkpoint_file=best_name, best_checkpoint_sha256=sha256(run/best_name))
        else:
            state['wait'] += 1
        state.update(completed_epoch=epoch, global_updates=engine._iter_cnt, curriculum_length=engine._cl_len,
            train_order=get_order(), rng_state=rng_state(), model_state=engine.model.state_dict(),
            optimizer_state=engine._optimizer.state_dict(), scheduler_state=engine._lr_scheduler.state_dict(),
            identity_sha256=identity['identity_sha256'])
        # The immutable best file is written first. last.pt commits the epoch;
        # derived JSON can be regenerated after an interrupted write.
        atomic_save(run/'last.pt', state)
        write_json(run/'history.json', state['history'])
        write_json(run/'summary.json', summary_for(state, identity))
        engine._logger.info('Epoch: %03d, Train Loss: %.4f, Train RMSE: %.4f, Train MAPE: %.4f, Valid Loss: %.4f, Valid RMSE: %.4f, Valid MAPE: %.4f, Train Time: %.4fs/epoch, Valid Time: %.4fs, LR: %.4e',
            epoch, train_mae, train_rmse, train_mape, val_mae, val_rmse, val_mape, train_seconds, validation_seconds, lr)
        print(json.dumps(row), flush=True)
    return state


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, default=default_data_dir())
    parser.add_argument('--epochs', type=int, default=100, help='Total completed epochs, not additional epochs; original maximum 100')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--threads', type=int, default=3)
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--check', action='store_true', help='Four train/val rows, batch two; engineering only')
    args = parser.parse_args(argv)
    if not 1 <= args.epochs <= 100:
        parser.error('epochs must be in 1..100')
    args.run_dir = args.run_dir.resolve()
    args.data_dir = args.data_dir.resolve()
    return args


def main(argv=None):
    args = parse_args(argv)
    exists = args.run_dir.exists()
    if exists != args.resume:
        raise ValueError('Use a fresh run directory, or explicitly --resume an existing run')
    args.run_dir.mkdir(parents=True, exist_ok=args.resume)
    with run_lock(args.run_dir):
        handler = None
        try:
            runtime = configure_runtime(args.device, args.threads, args.seed)
            modules = get_original_modules()
            logger = logging.getLogger('original-igstgnn-baseline')
            logger.setLevel(logging.INFO)
            handler = logging.FileHandler(args.run_dir/'record.log', encoding='utf-8')
            handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
            logger.addHandler(handler)
            config = dict(CONFIG, seed=args.seed, device=args.device)
            if args.check:
                config['bs'] = 2
            if not args.resume:
                from argparse import Namespace
                logger.info(Namespace(**config))
            data = load_data(args.data_dir, config, modules, logger, check=args.check, training=True)
            identity = dict(schema=SCHEMA, training_config=config, data=data.identity, source_fingerprint=modules.source_fingerprint,
                execution_source={p: sha256(REPO/p) for p in ('experiments/chronological/train_original_igstgnn_baseline.py',
                    'experiments/chronological/original_igstgnn_runtime.py', 'experiments/chronological/original_igstgnn_source.py')},
                runtime={key: value for key, value in runtime.items() if key != 'device_name'},
                engineering_only=args.check, scheduler=dict(milestones=MILESTONES, gamma=.5),
                original_training_rng_caveat='deterministic algorithms and TF32 disabled; old launcher did not enforce these',
                test_evaluated=False, best_selection_includes_stage_zero=False)
            identity['identity_sha256'] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
            state = None
            if args.resume:
                saved_identity = json.loads((args.run_dir/'identity.json').read_text(encoding='utf-8'))
                if saved_identity != identity:
                    raise ValueError('Baseline training identity differs; use a fresh directory')
                state = torch.load(args.run_dir/'last.pt', map_location='cpu', weights_only=True)
                validate_saved(state, identity, args.run_dir, data.loaders['train_loader'].num_batch)
                if state['completed_epoch'] >= args.epochs or state['wait'] >= config['patience']:
                    summary = summary_for(state, identity)
                    write_json(args.run_dir/'history.json', state['history'])
                    write_json(args.run_dir/'summary.json', summary)
                    print(json.dumps(dict(summary, no_op=True)), flush=True)
                    return summary
            else:
                write_json(args.run_dir/'identity.json', identity)
                write_json(args.run_dir/'runtime.json', runtime)
            model = build_model(data, modules, args.device)
            if sum(p.numel() for p in model.parameters()) != 455220:
                raise ValueError('Original Contra model parameter count differs')
            optimizer = torch.optim.Adam(model.parameters(), lr=config['lrate'], weight_decay=config['wdecay'], eps=1e-8)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=MILESTONES, gamma=.5)
            batches = data.loaders['train_loader'].num_batch
            engine = modules.engine.IGSTGNN_Engine(device=torch.device(args.device), model=model,
                dataloader=data.loaders, scaler=data.scaler, sampler=None, loss_fn=modules.metrics.masked_mae,
                lrate=config['lrate'], optimizer=optimizer, scheduler=scheduler,
                clip_grad_value=config['clip_grad_value'], max_epochs=100, patience=config['patience'],
                log_dir=str(args.run_dir), logger=logger, seed=args.seed, cl_step=config['cl_epoch']*batches,
                warm_step=config['warm_epoch']*batches, horizon=12, incident=True)
            run_epochs(engine, args.run_dir, identity, args.epochs, state)
            summary = json.loads((args.run_dir/'summary.json').read_text(encoding='utf-8'))
            print(json.dumps(summary), flush=True)
            return summary
        except BaseException as error:
            write_json(args.run_dir/('failure_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')+'.json'),
                dict(status='ORIGINAL_IGSTGNN_BASELINE_FAILED', error_type=type(error).__name__, error=str(error),
                     test_evaluated=False))
            raise
        finally:
            if handler is not None:
                logger.removeHandler(handler)
                handler.close()


if __name__ == '__main__':
    main()
