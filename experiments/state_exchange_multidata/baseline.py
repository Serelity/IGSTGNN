"""Dataset-generic training of the byte-frozen original IGSTGNN model.

The original model, sensor parser, batch builder, engine, optimizer recipe,
curriculum and patience remain unchanged. Only the test-free outer transaction
and the authenticated dataset dimensions differ from the Contra-only runner.
"""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
from types import SimpleNamespace
import time
import uuid

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch

from experiments.chronological.original_igstgnn_runtime import (
    CONFIG, configure_runtime, load_array, run_lock, sha256, tensor_tree_hash, write_json)
from experiments.chronological.original_igstgnn_source import get_original_modules
from experiments.chronological.train_original_igstgnn_baseline import (
    atomic_save, restore_rng, rng_state, tracked_shuffle)
from .common import source_hashes, spec

SCHEMA = 'state_exchange_multidata_original_training_v1'
MILESTONES = [1, 38, 46, 54, 62, 70, 80]
REPO = Path(__file__).resolve().parents[2]


def original_config(dataset, seed=2025, device='cpu', *, check=False):
    spec(dataset)
    return dict(CONFIG, dataset=dataset, seed=seed, device=str(device), bs=2 if check else 48)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _validate_samples(samples, nodes, split):
    for sample in samples:
        for key in ('x_data', 'y_data'):
            value = np.asarray(sample[key])
            _require(value.shape == (12, nodes, 3) and value.dtype == np.float32
                     and np.isfinite(value).all(), 'Invalid published '+split+' '+key)


def original_loader(data, modules, split, batch_size, logger):
    """Construct a frozen original loader; test access requires this explicit call."""
    _require(split in ('train', 'val', 'test'), 'Unknown original split')
    if split == 'test':
        samples = data.test_samples
        _validate_samples(samples, data.config['node_num'], split)
    else:
        samples = data.samples[split]
    _require(type(batch_size) is int and batch_size > 0, 'Invalid batch size')
    return modules.loader.IncidentDataLoader(samples, True, batch_size, logger,
        input_dim=3, sensor_info=data.sensors)


def load_original_data(directory, dataset, config, modules, logger, *, check=False, training=False):
    directory = Path(directory).resolve()
    registered = spec(dataset)
    nodes, counts = registered['nodes'], registered.get('counts', registered)
    hashes = {name: sha256(directory/name) for name in registered['hashes']}
    _require(hashes == registered['hashes'], 'Published release fingerprint mismatch')
    _require(config['dataset'] == dataset and config['seq_len'] == config['horizon'] == 12
             and config['input_dim'] == 3 and config['output_dim'] == 1,
             'Original dataset configuration differs')
    with (directory/'sensors.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    station_ids = np.asarray([int(row['station_id']) for row in rows], dtype=np.int64)
    _require(len(station_ids) == nodes and len(set(station_ids.tolist())) == nodes,
             'Original station axis differs')
    args = SimpleNamespace(**dict(config, node_num=nodes))
    sensors = modules.loader._load_sensor_info(str(directory), nodes, args, logger)
    with np.load(directory/'incident_stats.npz', allow_pickle=False) as stats:
        mean, std = float(stats['mean']), float(stats['std'])
        _require(bool(stats['normalized']) and np.isfinite([mean, std]).all() and std > 0,
                 'Expected unchanged normalized author release')
    all_samples = load_array(directory/'incident_all.npy')
    _require(len(all_samples) == counts['total_samples'], 'Published sample count differs')
    train_stop = counts['train_samples']
    val_stop = train_stop+counts['val_samples']
    _require(val_stop+counts['test_samples'] == len(all_samples), 'Registered split counts differ')
    wanted = ('train', 'val') if training else ('val',)
    samples, identities = {}, {}
    for split in wanted:
        start, stop = (0, train_stop) if split == 'train' else (train_stop, val_stop)
        chosen = tuple(all_samples[start:stop])
        if check:
            chosen = chosen[:4]
        _validate_samples(chosen, nodes, split)
        samples[split] = chosen
        batch = 2 if check else config['bs']
        identities[split] = dict(origin='incident_all.npy author-order slice',
            bounds_half_open=[start, stop], selected_indices=list(range(start, start+len(chosen))),
            count=len(chosen), batch_size=batch,
            partition=[min(batch, len(chosen)-i) for i in range(0, len(chosen), batch)])
    roads = {}
    for index, row in enumerate(rows):
        road = str(row.get('Fwy', row.get('Freeway', row.get('Road', 'unknown'))))
        direction = str(row.get('Direction', '')).strip()
        name = road+('-'+direction if direction and not road.upper().endswith('-'+direction.upper()) else '')
        roads.setdefault(name, []).append(index)
    data = SimpleNamespace(loaders={}, samples=samples, scaler=modules.loader.StandardScaler(mean, std),
        sensors=sensors, config=vars(args), station_ids=station_ids, roads=roads, directory=directory,
        test_samples=tuple(all_samples[val_stop:]),
        identity=dict(dataset=dataset, data_source='author_model_ready_release', hashes=hashes,
            splits=identities, normalization=dict(mean=mean, std=std, refitted=False),
            sensor_parser='unchanged_558ab01', test_file_opened=False,
            test_payload_deserialized=True, test_loader_constructed=False, test_evaluated=False,
            absolute_calendar_order_independently_verified=False))
    for split in wanted:
        data.loaders[split+'_loader'] = original_loader(data, modules, split,
                                                       2 if check else config['bs'], logger)
    return data


def build_original_model(data, modules, device):
    graphs = modules.graph.normalize_adj_mx(
        np.load(data.directory/'adj_matrix.npy', allow_pickle=False), 'doubletransition')
    config = dict(data.config, data_path=str(data.directory), device=str(device),
                  adjs=[torch.as_tensor(np.asarray(graph), device=device) for graph in graphs])
    model = modules.model.IGSTGNN(model_args=config, node_num=config['node_num'], input_dim=3,
        output_dim=1, seq_len=12, horizon=12, dataset=config['dataset'],
        data_path=str(data.directory), use_sensor_info=True)
    return model.to(device)


def _complete(state, identity):
    config = identity['training_config']
    budget = 3 if identity['engineering_only'] else config['max_epochs']
    return state['completed_epoch'] >= budget or state['wait'] >= config['patience']


def summary_for(state, identity):
    return dict(schema=SCHEMA, dataset=identity['training_config']['dataset'],
        status='ORIGINAL_MULTIDATA_COMPLETE' if _complete(state, identity) else 'ORIGINAL_MULTIDATA_PARTIAL',
        completed_epoch=state['completed_epoch'], best_epoch=state['best_epoch'],
        best_validation_mae=state['best_validation_mae'], best_checkpoint_file=state['best_checkpoint_file'],
        best_checkpoint_sha256=state['best_checkpoint_sha256'], global_updates=state['global_updates'],
        wait=state['wait'], engineering_only=identity['engineering_only'],
        best_selection_includes_stage_zero=False, registered_parameters=state['registered_parameters'],
        training_config=identity['training_config'], source_fingerprint=identity['source_fingerprint'],
        equal_recipe_candidate_control=False, test_evaluated=False)


def state_digest(state):
    return tensor_tree_hash({key:value for key, value in state.items() if key != 'state_payload_sha256'})


def scheduler_snapshot(scheduler):
    """Keep checkpoints readable by older weights-only unpicklers.

    MultiStepLR exposes a Counter in state_dict. Store only its integer mapping;
    the live scheduler gets a Counter back before any resumed step is executed.
    No deserialization allowlist or unrestricted pickle fallback is required.
    """
    saved = scheduler.state_dict()
    saved['milestones'] = dict(saved['milestones'])
    return saved


def restore_scheduler(scheduler, saved):
    milestones = saved.get('milestones')
    _require(type(milestones) is dict and milestones == dict(Counter(MILESTONES))
        and all(type(key) is int and type(value) is int for key, value in milestones.items()),
        'Saved scheduler milestones differ from the original recipe')
    scheduler.load_state_dict(dict(saved, milestones=Counter(milestones)))


def validate_saved(state, identity, run, train_batches):
    """Authenticate the committed epoch, optimizer schedule and immutable BEST."""
    _require(state.get('identity_sha256') == identity['identity_sha256'], 'Saved identity differs')
    _require(state.get('state_payload_sha256') == state_digest(state), 'Saved state payload changed')
    completed, history = state.get('completed_epoch'), state.get('history', [])
    _require(type(completed) is int and 0 <= completed <= 100 and len(history) == completed
        and [row.get('epoch') for row in history] == list(range(1, completed+1))
        and all(np.isfinite(row['validation_mae']) for row in history), 'Saved training history is invalid')
    _require(state['global_updates'] == completed*train_batches
        and [row['global_updates'] for row in history] == [epoch*train_batches for epoch in range(1, completed+1)]
        and state['registered_parameters'] == identity['registered_parameters'],
        'Saved best selection or update budget is inconsistent')
    length = identity['data']['splits']['train']['count']
    order = np.asarray(state['train_order'])
    _require(order.shape == (length,) and np.array_equal(np.sort(order), np.arange(length)),
             'Saved training permutation is invalid')
    config = identity['training_config']
    milestones = state['scheduler_state'].get('milestones')
    _require(type(milestones) is dict and milestones == dict(Counter(MILESTONES))
        and all(type(key) is int and type(value) is int for key, value in milestones.items()),
        'Saved scheduler milestones differ from the original recipe')
    warm, step, updates = config['warm_epoch']*train_batches, config['cl_epoch']*train_batches, state['global_updates']
    curriculum = 12 if updates < warm else min(12, 1+(updates-warm)//step)
    _require(state['curriculum_length'] == curriculum and state['scheduler_state']['last_epoch'] == completed,
             'Saved curriculum/scheduler epoch is inconsistent')
    expected_lr = config['lrate']*(.5**sum(completed >= milestone for milestone in MILESTONES))
    _require(state['scheduler_state']['_last_lr'] == [expected_lr]
        and [group['lr'] for group in state['optimizer_state']['param_groups']] == [expected_lr],
        'Saved optimizer learning rate differs')
    if completed == 0:
        _require(state['best_epoch'] is None and state['best_validation_mae'] is None and state['wait'] == 0
            and state['best_checkpoint_file'] is None and state['best_checkpoint_sha256'] is None
            and state['best_model_state_sha256'] is None and state['optimizer_state']['state'] == {},
            'Initial checkpoint cannot select an untrained BEST or contain optimizer updates')
        return
    best = min(history, key=lambda row: row['validation_mae'])
    _require(best['epoch'] == state['best_epoch'] and best['validation_mae'] == state['best_validation_mae']
        and state['wait'] == completed-state['best_epoch'], 'Saved best selection is inconsistent')
    name = state['best_checkpoint_file']
    _require(Path(name).name == name and sha256(Path(run)/name) == state['best_checkpoint_sha256'],
             'Saved best checkpoint fingerprint differs')


def _commit(engine, run, identity, state, order):
    state.update(global_updates=engine._iter_cnt, curriculum_length=engine._cl_len,
        train_order=list(order), rng_state=rng_state(), model_state=engine.model.state_dict(),
        optimizer_state=engine._optimizer.state_dict(), scheduler_state=scheduler_snapshot(engine._lr_scheduler),
        identity_sha256=identity['identity_sha256'])
    state['state_payload_sha256'] = state_digest(state)
    atomic_save(Path(run)/'last.pt', state)
    return state


def initial_state(engine, run, identity):
    """Commit the initial update boundary without adding stage zero to selection."""
    engine._cl_len = 12  # First original update already sets 12 before slicing its loss.
    state = dict(completed_epoch=0, best_epoch=None, best_validation_mae=None, wait=0,
        global_updates=0, history=[], registered_parameters=identity['registered_parameters'],
        best_checkpoint_file=None, best_checkpoint_sha256=None, best_model_state_sha256=None)
    order = np.arange(len(engine._dataloader['train_loader'].samples), dtype=np.int64).tolist()
    return _commit(engine, run, identity, state, order)


def run_epochs(engine, run, identity, target_epoch, state=None):
    run = Path(run)
    get_order = tracked_shuffle(engine._dataloader['train_loader'], None if state is None else state['train_order'])
    if state is None:
        state = initial_state(engine, run, identity)
    else:
        engine.model.load_state_dict(state['model_state'], strict=True)
        engine._optimizer.load_state_dict(state['optimizer_state'])
        restore_scheduler(engine._lr_scheduler, state['scheduler_state'])
        engine._iter_cnt, engine._cl_len = state['global_updates'], state['curriculum_length']
        restore_rng(state['rng_state'])
    def synchronize():
        if torch.device(engine._device).type == 'cuda':
            torch.cuda.synchronize(engine._device)
    for epoch in range(state['completed_epoch']+1, target_epoch+1):
        if state['wait'] >= engine._patience:
            break
        print(json.dumps(dict(dataset=identity['training_config']['dataset'], model='ORIGINAL_IGSTGNN',
                              phase='training', epoch=epoch,
                              total_batches=engine._dataloader['train_loader'].num_batch)), flush=True)
        synchronize()
        started = time.perf_counter()
        train_mae, train_mape, train_rmse = map(float, engine.train_batch())
        synchronize()
        train_seconds = time.perf_counter()-started
        synchronize()
        started = time.perf_counter()
        val_mae, val_mape, val_rmse = map(float, engine.evaluate('val'))
        synchronize()
        validation_seconds = time.perf_counter()-started
        _require(np.isfinite([train_mae, train_mape, train_rmse, val_mae, val_mape, val_rmse]).all(),
                 'Nonfinite original baseline metrics')
        lr = engine._lr_scheduler.get_last_lr()[0]
        engine._lr_scheduler.step()
        row = dict(epoch=epoch, train_mae=train_mae, train_mape_fraction=train_mape, train_rmse=train_rmse,
            validation_mae=val_mae, validation_mape_fraction=val_mape, validation_rmse=val_rmse,
            train_seconds=train_seconds, validation_seconds=validation_seconds, learning_rate=lr,
            global_updates=engine._iter_cnt, curriculum_length=engine._cl_len,
            train_order_sha256=hashlib.sha256(np.asarray(get_order(), dtype=np.int64).tobytes()).hexdigest())
        state['history'].append(row)
        if state['best_epoch'] is None or val_mae < state['best_validation_mae']:
            best_name = 'best_epoch_{:03d}_{}.pt'.format(epoch, uuid.uuid4().hex[:12])
            atomic_save(run/best_name, engine.model.state_dict())
            state.update(best_epoch=epoch, best_validation_mae=val_mae, wait=0,
                         best_checkpoint_file=best_name, best_checkpoint_sha256=sha256(run/best_name),
                         best_model_state_sha256=tensor_tree_hash(engine.model.state_dict()))
        else:
            state['wait'] += 1
        state.update(completed_epoch=epoch)
        # Committed last.pt points only to a previously completed immutable BEST.
        _commit(engine, run, identity, state, get_order())
        write_json(run/'history.json', state['history'])
        write_json(run/'summary.json', summary_for(state, identity))
        engine._logger.info('Epoch: %03d, Train Loss: %.4f, Train RMSE: %.4f, Train MAPE: %.4f, '
            'Valid Loss: %.4f, Valid RMSE: %.4f, Valid MAPE: %.4f, LR: %.4e',
            epoch, train_mae, train_rmse, train_mape, val_mae, val_rmse, val_mape, lr)
        print(json.dumps(row), flush=True)
    return state


def _read_identity(run):
    identity = json.loads((run/'identity.json').read_text(encoding='utf-8'))
    _require(identity.get('schema') == SCHEMA, 'Not a multidataset original baseline')
    digest = identity.pop('identity_sha256', None)
    _require(digest == hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest(),
             'Baseline identity hash differs')
    identity['identity_sha256'] = digest
    _require(identity['execution_source'] == source_hashes(), 'Baseline execution source changed')
    _require(identity['source_fingerprint'] == get_original_modules().source_fingerprint,
             'Frozen original source changed')
    config = identity['training_config']
    _require(type(identity['engineering_only']) is bool and identity['seed'] == 2025
        and config == identity['config']
        and config == original_config(config['dataset'], identity['seed'], config['device'],
                                      check=identity['engineering_only']),
        'Baseline registered training recipe differs')
    return identity


def selected_checkpoint(run, data=None):
    """Return only validation-selected BEST; caller separately enforces formal completion."""
    run = Path(run).resolve()
    identity = _read_identity(run)
    state = torch.load(run/'last.pt', map_location='cpu', weights_only=True)
    batches = len(identity['data']['splits']['train']['partition'])
    validate_saved(state, identity, run, batches)
    _require(state['completed_epoch'] > 0 and state['best_epoch'] is not None,
             'No validation-selected BEST exists before the first trained epoch')
    if data is not None:
        _require(identity['training_config']['dataset'] == data.config['dataset']
            and identity['data']['hashes'] == data.identity['hashes']
            and identity['data']['normalization'] == data.identity['normalization']
            and identity['station_ids'] == data.station_ids.tolist(),
            'Baseline training and evaluation data differ')
    checkpoint = run/state['best_checkpoint_file']
    weights = torch.load(checkpoint, map_location='cpu', weights_only=True)
    _require(tensor_tree_hash(weights) == state['best_model_state_sha256'], 'BEST model tensor hash differs')
    role = dict(schema=SCHEMA, selection='best', case='ORIGINAL_IGSTGNN',
        dataset=identity['training_config']['dataset'], identity_sha256=identity['identity_sha256'],
        completed_epoch=state['completed_epoch'], selected_epoch=state['best_epoch'],
        best_epoch=state['best_epoch'], training_complete=_complete(state, identity),
        training_budget_epochs=3 if identity['engineering_only'] else 100,
        engineering_only=identity['engineering_only'], registered_parameters=state['registered_parameters'],
        checkpoint_sha256=state['best_checkpoint_sha256'], model_state_sha256=tensor_tree_hash(weights),
        source_fingerprint=identity['source_fingerprint'], training_config=identity['training_config'],
        input_data_identity=identity['data'], source_run=str(run),
        last_checkpoint_sha256=sha256(run/'last.pt'), global_updates=state['global_updates'],
        wait=state['wait'], expected_validation_mae=state['best_validation_mae'],
        best_selection_includes_stage_zero=False, equal_recipe_candidate_control=False,
        test_evaluated=False)
    return weights, role, state['best_validation_mae'], checkpoint


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=('Alameda', 'Orange', 'Contra_Costa'), required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--threads', type=int, default=3)
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args(argv)
    if not 1 <= args.epochs <= (3 if args.check else 100):
        parser.error('epochs must be in 1..3 for engineering or 1..100 for formal training')
    if args.seed != 2025:
        parser.error('This registered cross-dataset run uses seed 2025')
    args.run_dir, args.data_dir = args.run_dir.resolve(), args.data_dir.resolve()
    return args


def main(argv=None):
    args = parse_args(argv)
    _require(args.run_dir != args.data_dir and args.data_dir not in args.run_dir.parents,
             'Output must be outside the dataset')
    for parent in args.run_dir.parents:
        _require(not (parent/'identity.json').is_file(), 'Output intersects an existing experiment')
    _require(args.run_dir.exists() == args.resume,
             'Use a fresh run directory, or explicitly --resume an existing run')
    args.run_dir.mkdir(parents=True, exist_ok=args.resume)
    with run_lock(args.run_dir):
        handler = None
        try:
            runtime = configure_runtime(args.device, args.threads, args.seed)
            modules = get_original_modules()
            logger = logging.getLogger('original-multidata-'+args.dataset)
            logger.setLevel(logging.INFO)
            handler = logging.FileHandler(args.run_dir/'record.log', encoding='utf-8')
            handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
            logger.addHandler(handler)
            config = original_config(args.dataset, args.seed, args.device, check=args.check)
            data = load_original_data(args.data_dir, args.dataset, config, modules, logger,
                                      check=args.check, training=True)
            model = build_original_model(data, modules, args.device)
            parameters = sum(parameter.numel() for parameter in model.parameters())
            identity = dict(schema=SCHEMA, training_config=config, config=config, seed=args.seed, data=data.identity,
                station_ids=data.station_ids.tolist(), registered_parameters=parameters,
                source_fingerprint=modules.source_fingerprint, execution_source=source_hashes(),
                runtime={key: value for key, value in runtime.items() if key != 'device_name'},
                engineering_only=args.check, scheduler=dict(milestones=MILESTONES, gamma=.5),
                original_training_rng_caveat='deterministic algorithms and TF32 disabled; old launcher did not enforce these',
                test_evaluated=False, best_selection_includes_stage_zero=False,
                selection='original_native_pooled_validation_mae', equal_recipe_candidate_control=False)
            identity['identity_sha256'] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
            state = None
            if args.resume:
                saved = json.loads((args.run_dir/'identity.json').read_text(encoding='utf-8'))
                _require(saved == identity, 'Baseline training identity differs; use a fresh directory')
                state = torch.load(args.run_dir/'last.pt', map_location='cpu', weights_only=True)
                validate_saved(state, identity, args.run_dir, data.loaders['train_loader'].num_batch)
                if state['completed_epoch'] >= args.epochs or _complete(state, identity):
                    summary = summary_for(state, identity)
                    write_json(args.run_dir/'history.json', state['history'])
                    write_json(args.run_dir/'summary.json', summary)
                    print(json.dumps(dict(summary, no_op=True)), flush=True)
                    return summary
            optimizer = torch.optim.Adam(model.parameters(), lr=config['lrate'], weight_decay=config['wdecay'], eps=1e-8)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=MILESTONES, gamma=.5)
            batches = data.loaders['train_loader'].num_batch
            engine = modules.engine.IGSTGNN_Engine(device=torch.device(args.device), model=model,
                dataloader=data.loaders, scaler=data.scaler, sampler=None, loss_fn=modules.metrics.masked_mae,
                lrate=config['lrate'], optimizer=optimizer, scheduler=scheduler,
                clip_grad_value=config['clip_grad_value'], max_epochs=100, patience=config['patience'],
                log_dir=str(args.run_dir), logger=logger, seed=args.seed,
                cl_step=config['cl_epoch']*batches, warm_step=config['warm_epoch']*batches,
                horizon=12, incident=True)
            if not args.resume:
                state = initial_state(engine, args.run_dir, identity)
                write_json(args.run_dir/'runtime.json', runtime)
                # Identity is committed only after the resumable pre-update state.
                write_json(args.run_dir/'identity.json', identity)
                logger.info(SimpleNamespace(**config))
            state = run_epochs(engine, args.run_dir, identity, args.epochs, state)
            summary = summary_for(state, identity)
            print(json.dumps(summary), flush=True)
            return summary
        except BaseException as error:
            write_json(args.run_dir/('failure_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')+'.json'),
                dict(status='ORIGINAL_MULTIDATA_FAILED', error_type=type(error).__name__, error=str(error),
                     test_evaluated=False))
            raise
        finally:
            if handler is not None:
                logger.removeHandler(handler)
                handler.close()


if __name__ == '__main__':
    main()
