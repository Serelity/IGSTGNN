"""Paired HISTORY/M_ONLY/EVENT adaptation with a replacement DIF operator.

This is a new architecture. Its own stage-zero validation is eligible for
selection; the historical MIN validation metric is never substituted for it.
Only train and development validation loaders are iterated.
"""
import argparse
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
import uuid

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.chronological import train_min_forecast_fusion as shared
from experiments.chronological.smoke import sha256, set_seed
from experiments.chronological.train import cpu_state, save_checkpoint
from src.engines.igstgnn_engine import IGSTGNN_Engine
from src.models.incident_spatial import require
from src.models.incident_spatial_residual import tensor_state_hash
from src.utils.igstgnn_published_events import EventPublishedData
from src.utils.metrics import masked_mae

paper, legacy = shared.paper, shared.legacy
HERE = Path(__file__).resolve().parent
PROTOCOL = HERE/'min_conditional_propagation_v1.json'
SCHEMA = 'min_conditional_propagation_warmstart_run_v1'
ARMS = {'HISTORY': 'history', 'M_ONLY': 'mask_only', 'EVENT': 'event'}
WORKER_ENV = 'MIN_CONDITIONAL_PROPAGATION_WORKER'
DEFAULT_SOURCE = shared.DEFAULT_SOURCE
DEFAULT_REFERENCE = REPO/'experiments/chronological_runs/contra_min_forecast_fusion_f1_bs64_seed2025'
load_source, build_core = shared.load_source, shared.build_core
source_snapshot, readonly_source_lock = shared.source_snapshot, shared.readonly_source_lock
event_tensors = shared.event_tensors


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.'+uuid.uuid4().hex+'.partial')
    try:
        with temporary.open('w', encoding='utf-8', newline='\n') as stream:
            stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in (('data-dir', paper.default_data()), ('graph-dir', HERE/'internal_spatial_graph_is11'),
                          ('protocol', PROTOCOL), ('source-run', DEFAULT_SOURCE), ('reference-run', DEFAULT_REFERENCE)):
        parser.add_argument('--'+name, type=Path, default=default)
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--arms', nargs='+', choices=ARMS, default=list(ARMS))
    parser.add_argument('--only', nargs='+', choices=ARMS)
    parser.add_argument('--epochs', type=int, default=1, help='Absolute completed stage target; resume with 5 or 10')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--threads', type=int, default=3)
    parser.add_argument('--receiver-chunk-size', type=int, default=4)
    parser.add_argument('--no-checkpoint-messages', action='store_true')
    parser.add_argument('--check', action='store_true', help='Four train/validation rows, batch two; engineering only')
    parser.add_argument('--worker', choices=ARMS, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    require(1 <= args.epochs <= 100 and args.threads > 0 and 0 <= args.seed < 2**32, 'Invalid epoch/threads/seed')
    require(args.receiver_chunk_size > 0 and (args.batch_size is None or args.batch_size > 0), 'Invalid chunk/batch size')
    require(len(set(args.arms)) == len(args.arms), 'Duplicate arms')
    args.arms = [arm for arm in ARMS if arm in args.arms]
    require(args.only is None or (len(set(args.only)) == len(args.only) and set(args.only) <= set(args.arms)),
            '--only must be a unique subset of the saved --arms')
    require(not args.check or args.epochs == 1, 'Engineering check requires exactly one epoch')
    require(not args.worker or (os.environ.get(WORKER_ENV) == '1' and args.run_dir is not None
            and args.worker in args.arms), 'Internal worker requires parent launcher; use --only')
    if args.run_dir is None:
        args.run_dir = REPO/'experiments/chronological_runs'/('contra_min_conditional_'+('check_' if args.check else '')
            +datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6])
    for name in ('data_dir', 'graph_dir', 'protocol', 'source_run', 'reference_run', 'run_dir'):
        setattr(args, name, getattr(args, name).resolve())
    for source in (args.source_run, args.reference_run):
        require(args.run_dir != source and source not in args.run_dir.parents and args.run_dir not in source.parents,
                'Run must be separate from source/reference and their ancestors')
    args.checkpoint_messages = not args.no_checkpoint_messages
    return args


def load_protocol(args):
    protocol = legacy.read_json(args.protocol)
    require(protocol == legacy.read_json(PROTOCOL), 'Frozen conditional-propagation protocol changed')
    return dict(protocol, batch_size=args.batch_size) if args.batch_size is not None else protocol


def build_arm(args, arm, nodes, graph, device, source=None, adjacency=None):
    from src.models.incident_spatial_conditional_propagation import from_minimal
    require(arm in ARMS, 'Unknown conditional-propagation arm')
    source = load_source(args) if source is None else source
    adjacency = np.load(args.data_dir/'adj_matrix.npy', allow_pickle=False) if adjacency is None else adjacency
    core = build_core(args, nodes, graph, device, source)
    set_seed(args.seed)
    model, certificate = from_minimal(core, mode=ARMS[arm], adjacency=adjacency,
        receiver_chunk_size=args.receiver_chunk_size, checkpoint_messages=args.checkpoint_messages)
    model.to(device).requires_grad_(True)
    set_seed(args.seed)
    return model, dict(certificate, source_best_epoch=source['selected']['epoch'],
        source_best_model_sha256=source['best_model_sha256'], fresh_optimizer_and_scheduler=True,
        old_MIN_stage_zero_equivalence_claim=False)


def source_fingerprints():
    names = ('experiments/chronological/train_min_conditional_propagation.py',
             'src/models/incident_spatial_conditional_propagation.py',
             'src/models/incident_conditional_propagation.py', 'src/utils/igstgnn_published_events.py',
             'src/engines/igstgnn_engine.py')
    return dict(shared.source_fingerprints(), **{name: sha256(REPO/name) for name in names})


def preflight(args, data, graph, source, adjacency):
    batch = next(data.loaders['train_loader'].get_iterator())
    x = torch.as_tensor(batch['x_data'][:2], dtype=torch.float32, device=args.device)
    events = event_tensors({k: v[:2] for k, v in batch['event_data'].items()}, args.device)
    states, prediction, manifests = None, None, {}
    for arm in args.arms:
        model, certificate = build_arm(args, arm, len(data.station_ids), graph, args.device, source, adjacency)
        model.eval()
        state = cpu_state(model.state_dict())
        require(all(p.requires_grad for p in model.parameters()), 'Whole candidate must train jointly')
        if states is not None:
            require(state.keys() == states.keys() and all(torch.equal(state[k], states[k]) for k in state),
                    'Candidate initialization differs between arms')
        with torch.inference_mode():
            actual = model(x, incident_data=None if arm == 'HISTORY' else events)
        require(torch.isfinite(actual).all(), 'Nonfinite stage-zero candidate')
        if prediction is not None:
            require(torch.equal(actual, prediction), 'Zero-initialized event paths must match HISTORY at stage zero')
        states, prediction = state, actual.detach().clone()
        manifests[arm] = dict(parameters=sum(p.numel() for p in model.parameters()),
            initial_state_sha256=tensor_state_hash(state), conversion=certificate)
        del model
    return dict(status='MIN_CONDITIONAL_PROPAGATION_PAIRED_INITIALIZATION_PASS', arms=manifests,
        registered_state_and_predictions_equal=True, initialization_examples=len(x), initialization_split='train',
        source_validation_forward_calls=0, own_stage_zero_validation_required=True,
        old_MIN_stage_zero_equivalence_claim=False)


def make_identity(args, protocol, hashes, graph_info, initialization, source):
    return dict(schema=SCHEMA, protocol=protocol, protocol_sha256=sha256(args.protocol),
        paths={name: str(getattr(args, name)) for name in ('data_dir', 'graph_dir', 'source_run', 'reference_run', 'protocol')},
        arm_modes={arm: ARMS[arm] for arm in args.arms}, package=hashes, graph=graph_info,
        graph_inputs_sha256={name: sha256(args.graph_dir/name) for name in ('graph.json', 'stations.csv', 'source_summary.json')},
        source_sha256=source_fingerprints(), seed=args.seed, engineering_check=args.check,
        batch_size=2 if args.check else protocol['batch_size'], threads=args.threads, initialization=initialization,
        train_batches=(min(4, protocol['train_samples']) if args.check else protocol['train_samples'])//
            (2 if args.check else protocol['batch_size'])+
            int((min(4, protocol['train_samples']) if args.check else protocol['train_samples']) %
                (2 if args.check else protocol['batch_size']) != 0),
        source=dict(files_sha256=source['files_sha256'], best_epoch=source['selected']['epoch'],
            completed_epoch=source['saved']['completed_epoch'], best_model_sha256=source['best_model_sha256'],
            best_metrics=[source['selected'][k] for k in ('validation_mae', 'validation_mape', 'validation_rmse')]),
        receiver_chunk_size=args.receiver_chunk_size, checkpoint_messages=args.checkpoint_messages,
        reference_run=str(args.reference_run), source_optimizer_restored=False, source_exact_resume=False,
        source_scheduler_restored=False, full_horizon_loss_every_update=True,
        same_registered_parameter_layout=True, own_stage_zero_eligible=True,
        old_MIN_stage_zero_equivalence_claim=False, clock_offset_slots=0,
        event_scope='published_archive_attributes_conditional_offline',
        graph_inputs='original_unconditioned_history', physical_capacity_claim=False,
        optimizer_recipe='fixed_shared_Adam_1e-4_no_new_parameter_lr_search',
        primary_baseline='original_IGSTGNN_not_in_this_training_stage',
        runtime=dict(torch=str(torch.__version__), cuda=torch.version.cuda, device_type=torch.device(args.device).type,
            cublas_workspace_config=os.environ['CUBLAS_WORKSPACE_CONFIG']),
        test_evaluated=False, test_payload_deserialized=True)


def verify_identity(args, identity, initialization, allow_new=False):
    path = args.run_dir/'identity.json'
    if path.is_file():
        require(legacy.read_json(path) == identity, 'Run identity changed; saved checkpoints preserved')
        require(legacy.read_json(args.run_dir/'initialization.json') == initialization, 'Initialization changed')
    else:
        require(allow_new and {p.name for p in args.run_dir.iterdir()} <= {'active.lock'}, 'Use a fresh run directory')
        write_json(path, identity)
        write_json(args.run_dir/'initialization.json', initialization)


def merge_statistics(total, statistics):
    for key, value in statistics.items():
        if torch.is_tensor(value):
            require(value.numel() == 1, 'Only scalar propagation diagnostics can aggregate')
            value = value.detach().cpu().item()
        value = float(value)
        require(np.isfinite(value), 'Nonfinite propagation diagnostic: '+key)
        if key.endswith('_max'):
            total[key] = max(total.get(key, 0.), value)
        elif key.endswith('_min'):
            total[key] = min(total.get(key, value), value)
        else:
            total[key] = total.get(key, 0.)+value


class _ValidationProgressLoader:
    """Observe the unchanged iterator around the parent's validation body.

    The parent has inverse-transformed predictions and copied both arrays to
    CPU before it requests the next batch. Resuming this generator therefore
    marks that batch completed without a new forward or explicit CUDA sync.
    """
    def __init__(self, loader, arm, phase):
        self.loader, self.arm, self.phase = loader, arm, phase
        self.total, self.completed, self.active_batch = loader.num_batch, 0, None
        self.started = time.perf_counter()

    def emit(self, status, **extra):
        print(json.dumps(dict(event='validation_progress', arm=self.arm, phase=self.phase,
            status=status, completed=self.completed, total=self.total,
            elapsed_seconds=time.perf_counter()-self.started, **extra), allow_nan=False), flush=True)

    def get_iterator(self):
        for index, batch in enumerate(self.loader.get_iterator(), 1):
            self.active_batch = index
            self.emit('batch_start', batch=index)
            yield batch
            self.completed = index
            self.active_batch = None
            self.emit('batch_completed', batch=index)

    def __getattr__(self, name):
        return getattr(self.loader, name)


class PropagationEngine(IGSTGNN_Engine):
    def __init__(self, arm, **kwargs):
        self.arm, self.collect_diagnostics, self.diagnostics = arm, False, {}
        self.validation_phase = 'validation'
        super().__init__(**kwargs)
        self._cl_len = self._horizon

    def _update_curriculum_length(self):
        self._iter_cnt += 1
        self._cl_len = self._horizon

    def _prepare_batch(self, batch):
        events = None if self.arm == 'HISTORY' else event_tensors(batch['event_data'], self._device)
        return self._as_float_tensor(batch['x_data']), self._as_float_tensor(batch['y_data']), events, None

    def _predict(self, X, label, incident_data, sensor_data):
        if not self.collect_diagnostics:
            return self.model(X, label, incident_data=incident_data)
        result = self.model(X, label, incident_data=incident_data, return_spatial_details=True)
        self.diagnostics['forward_calls'] = self.diagnostics.get('forward_calls', 0)+1
        merge_statistics(self.diagnostics, result['propagation']['statistics'])
        return result['prediction']

    def evaluate(self, mode):
        require(mode == 'val', 'This stage never evaluates test')
        original = self._dataloader
        progress = _ValidationProgressLoader(original['val_loader'], self.arm, self.validation_phase)
        self._dataloader = dict(original, val_loader=progress)
        self.collect_diagnostics, self.diagnostics = True, {}
        try:
            progress.emit('phase_start')
            result = super().evaluate(mode)
            progress.emit('phase_end')
            return result
        except BaseException as error:
            progress.emit('phase_failed', active_batch=progress.active_batch, error_type=type(error).__name__)
            raise
        finally:
            self._dataloader = original
            self.collect_diagnostics = False


def make_engine(args, arm, protocol, data, model, logger):
    require(all(p.requires_grad for p in model.parameters()), 'Complete candidate must train jointly')
    optimizer = torch.optim.Adam(model.parameters(), lr=protocol['learning_rate'],
        weight_decay=protocol['weight_decay'], eps=protocol['adam_eps'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, protocol['lr_milestones'], gamma=protocol['lr_gamma'])
    return PropagationEngine(arm=arm, device=torch.device(args.device), model=model, dataloader=data.loaders,
        scaler=data.scaler, sampler=None, loss_fn=masked_mae, lrate=protocol['learning_rate'], optimizer=optimizer,
        scheduler=scheduler, clip_grad_value=protocol['clip_grad_norm'], max_epochs=protocol['max_epochs'],
        patience=protocol['patience'], log_dir=str(args.run_dir/arm), logger=logger, seed=args.seed,
        cl_step=0, warm_step=0, horizon=12, incident=False)


def validate_saved(saved, identity, arm):
    require(saved.get('identity') == identity and saved.get('arm') == arm, 'Checkpoint identity/arm mismatch')
    history = saved.get('history')
    require(isinstance(history, list) and history and len(history) == saved.get('completed_epoch')
        and [r['epoch'] for r in history] == list(range(1, len(history)+1)), 'Invalid checkpoint history')
    require(saved.get('curriculum_length') == 12 and all(r['curriculum_length'] == 12 for r in history), 'Full horizon required')
    require(saved['global_updates'] == len(history)*identity['train_batches'], 'Optimizer update budget changed')
    require(all(saved[k] == history[-1][k] for k in ('best_epoch', 'best_metric', 'global_updates', 'wait')), 'Checkpoint counters disagree')
    stage0 = saved['stage_zero']
    require(stage0['initial_state_sha256'] == identity['initialization']['arms'][arm]['initial_state_sha256'], 'Stage-zero state changed')
    eligible = [stage0['validation_mae']]+[r['validation_mae'] for r in history]
    require(np.isfinite(eligible).all() and saved['best_metric'] == min(eligible)
        and type(saved['best_epoch']) is int and 0 <= saved['best_epoch'] < len(eligible)
        and eligible[saved['best_epoch']] == saved['best_metric'], 'Best selection differs from own stage-zero/prefix')
    require(saved['model_state'].keys() == saved['best_model_state'].keys(), 'Checkpoint tensor keys differ')
    conversion = identity['initialization']['arms'][arm]['conversion']
    expected_keys = set(conversion['retained_state_keys']) | set(conversion['added_state_keys'])
    require(set(saved['model_state']) == expected_keys, 'Checkpoint replacement architecture keys changed')
    require(saved['parameters'] == identity['initialization']['arms'][arm]['parameters'], 'Checkpoint parameter count changed')
    if saved['best_epoch'] == 0:
        require(tensor_state_hash(saved['best_model_state']) == stage0['initial_state_sha256'], 'Selected stage-zero tensors differ')
    for key, value in saved['model_state'].items():
        best = saved['best_model_state'][key]
        require(value.shape == best.shape and value.dtype == best.dtype and torch.isfinite(value).all()
            and torch.isfinite(best).all(), 'Invalid checkpoint tensor: '+key)


def summary_for(saved, args, protocol):
    stopped = saved['wait'] >= protocol['patience']
    return dict(status='ENGINEERING_CHECK_PASS' if args.check else 'EARLY_STOPPED' if stopped else 'PAUSED_AT_EPOCH_BOUNDARY',
        arm=saved['arm'], mode=ARMS[saved['arm']], completed_epoch=saved['completed_epoch'], requested_epoch=args.epochs,
        best_epoch=saved['best_epoch'], best_metric=saved['best_metric'], wait=saved['wait'], early_stopped=stopped,
        global_updates=saved['global_updates'], curriculum_length=12, parameters=saved['parameters'],
        stage_zero=saved['stage_zero'], own_stage_zero_eligible=True, source_stage_zero_equivalence_claim=False,
        batch_size=2 if args.check else protocol['batch_size'], all_backbone_parameters_trainable=True,
        event_inputs_used=saved['arm'] != 'HISTORY', semantic_inputs_used=saved['arm'] == 'EVENT',
        source_best_epoch=saved['identity']['source']['best_epoch'], fresh_stage_optimizer=True,
        receiver_chunk_size=args.receiver_chunk_size, checkpoint_messages=args.checkpoint_messages,
        primary_baseline='original_IGSTGNN_not_in_this_training_stage', predictive_gain_claim=False, test_evaluated=False,
        checkpoint='last.pt', best_model_state_key='best_model_state')


def repair_sidecars(root, saved, args, protocol):
    path = root/'history.json'
    if path.is_file():
        try:
            previous = legacy.read_json(path)
        except (json.JSONDecodeError, UnicodeDecodeError):
            path.with_name('history.interrupted_'+uuid.uuid4().hex[:6]+'.json').write_bytes(path.read_bytes())
        else:
            require(isinstance(previous, list) and previous == saved['history'][:len(previous)], 'History conflicts with checkpoint')
    write_json(path, saved['history'])
    write_json(root/'stage_zero.json', saved['stage_zero'])
    write_json(root/'summary.json', summary_for(saved, args, protocol))


def train_arm(args, arm, protocol, data, identity, graph, logger, source=None, adjacency=None):
    root = args.run_dir/arm
    root.mkdir(exist_ok=True)
    path = root/'last.pt'
    saved = torch.load(path, map_location='cpu', weights_only=False) if path.is_file() else None
    if saved is not None:
        validate_saved(saved, identity, arm)
        repair_sidecars(root, saved, args, protocol)
        if saved['completed_epoch'] >= args.epochs or saved['wait'] >= protocol['patience']:
            return dict(summary_for(saved, args, protocol), optimizer_updates_this_invocation=0)
    model, _ = build_arm(args, arm, len(data.station_ids), graph, args.device, source, adjacency)
    engine = make_engine(args, arm, protocol, data, model, logger)
    if saved is None:
        engine.validation_phase = 'stage_zero'
        values = engine.evaluate('val')
        require(np.isfinite(values).all(), 'Nonfinite own stage zero')
        stage0 = dict(zip(('validation_mae', 'validation_mape', 'validation_rmse'), map(float, values)))
        stage0.update(initial_state_sha256=tensor_state_hash(model.state_dict()), optimizer_updates=0,
            source_MIN_equivalence_claim=False, validation_forward_calls=data.loaders['val_loader'].num_batch)
        require(stage0['initial_state_sha256'] == identity['initialization']['arms'][arm]['initial_state_sha256'], 'Stage-zero hash differs')
        history, best, best_epoch, wait, best_state = [], values[0], 0, 0, cpu_state(model.state_dict())
        old = root/'stage_zero.json'
        if old.is_file():
            require(legacy.read_json(old) == stage0, 'Interrupted stage-zero certificate differs')
        write_json(old, stage0)
        set_seed(args.seed)
    else:
        model.load_state_dict(saved['model_state'], strict=True)
        engine._optimizer.load_state_dict(saved['optimizer_state'])
        engine._lr_scheduler.load_state_dict(saved['scheduler_state'])
        engine._iter_cnt, engine._cl_len = saved['global_updates'], 12
        data.loaders['train_loader'].set_order(saved['train_order'])
        np.random.set_state(saved['numpy_rng'])
        torch.set_rng_state(saved['torch_rng'])
        if torch.device(args.device).type == 'cuda':
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
        history, best, best_epoch, wait, best_state, stage0 = (saved[k] for k in
            ('history', 'best_metric', 'best_epoch', 'wait', 'best_model_state', 'stage_zero'))
    updates_before = engine._iter_cnt
    for epoch in range(len(history)+1, args.epochs+1):
        if wait >= protocol['patience']:
            break
        if torch.device(args.device).type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        train = engine.train_batch()
        engine.validation_phase = 'epoch_'+str(epoch)
        val = engine.evaluate('val')
        require(np.isfinite([*train, *val]).all(), 'Nonfinite candidate metrics')
        if val[0] < best:
            best, best_epoch, wait, best_state = float(val[0]), epoch, 0, cpu_state(model.state_dict())
        else:
            wait += 1
        lr = engine._optimizer.param_groups[0]['lr']
        engine._lr_scheduler.step()
        row = dict(epoch=epoch, **dict(zip(('train_mae', 'train_mape', 'train_rmse'), map(float, train))),
            **dict(zip(('validation_mae', 'validation_mape', 'validation_rmse'), map(float, val))),
            best_metric=best, best_epoch=best_epoch, wait=wait, global_updates=engine._iter_cnt, curriculum_length=12,
            learning_rate_used=lr, next_learning_rate=engine._optimizer.param_groups[0]['lr'],
            train_order_sha256=hashlib.sha256(data.loaders['train_loader'].order.tobytes()).hexdigest(),
            validation_diagnostics=engine.diagnostics, seconds=time.perf_counter()-started,
            cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated() if torch.device(args.device).type == 'cuda' else None)
        history.append(row)
        saved = dict(identity=identity, arm=arm, completed_epoch=epoch, history=history, stage_zero=stage0,
            parameters=sum(p.numel() for p in model.parameters()), model_state=cpu_state(model.state_dict()),
            best_model_state=best_state, best_metric=best, best_epoch=best_epoch, wait=wait,
            optimizer_state=engine._optimizer.state_dict(), scheduler_state=engine._lr_scheduler.state_dict(),
            global_updates=engine._iter_cnt, curriculum_length=12, train_order=data.loaders['train_loader'].order.copy(),
            numpy_rng=np.random.get_state(), torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if torch.device(args.device).type == 'cuda' else [])
        validate_saved(saved, identity, arm)
        save_checkpoint(path, saved)
        repair_sidecars(root, saved, args, protocol)
        print(json.dumps(dict(arm=arm, **row), allow_nan=False), flush=True)
        gc.collect()
    result = dict(summary_for(saved, args, protocol), optimizer_updates_this_invocation=saved['global_updates']-updates_before)
    write_json(root/'summary.json', result)
    return result


def prepare(args, logger, skip_preflight=False):
    protocol = load_protocol(args)
    hashes = paper.verify_release(args.data_dir, protocol)
    data = EventPublishedData(args.data_dir, protocol, logger, check=args.check)
    graph, _, info = legacy.load_graph(args.graph_dir, data.station_ids, protocol['graph_sha256'])
    adjacency = np.load(args.data_dir/'adj_matrix.npy', allow_pickle=False)
    source = load_source(args, hashes)
    require(data.loaders['train_loader'].bs == data.loaders['val_loader'].bs == source['identity']['batch_size'],
            'Preserve source original batch size')
    initialization = legacy.read_json(args.run_dir/'initialization.json') if skip_preflight else preflight(args, data, graph, source, adjacency)
    identity = make_identity(args, protocol, hashes, info, initialization, source)
    identity['train_batches'] = data.loaders['train_loader'].num_batch
    return protocol, data, graph, adjacency, initialization, identity, source


def comparison(args, identity):
    saved = {}
    for arm in args.arms:
        path = args.run_dir/arm/'last.pt'
        if path.is_file():
            saved[arm] = torch.load(path, map_location='cpu', weights_only=False)
            validate_saved(saved[arm], identity, arm)
    common = min((s['completed_epoch'] for s in saved.values()), default=0) if len(saved) == len(args.arms) else 0
    if common:
        require(all(len({s['history'][e]['train_order_sha256'] for s in saved.values()}) == 1 for e in range(common)), 'Paired train order differs')
    comparisons = {}
    for left, right in (('HISTORY', 'M_ONLY'), ('M_ONLY', 'EVENT'), ('HISTORY', 'EVENT')):
        if common and left in saved and right in saved:
            eligible = lambda s: [s['stage_zero']['validation_mae']]+[r['validation_mae'] for r in s['history'][:common]]
            comparisons[right+'_vs_'+left] = dict(common_epoch_mae_gain=saved[left]['history'][common-1]['validation_mae']-
                saved[right]['history'][common-1]['validation_mae'], common_prefix_best_mae_gain=min(eligible(saved[left]))-min(eligible(saved[right])))
    complete = len(saved) == len(args.arms) and all(s['completed_epoch'] >= args.epochs or s['wait'] >= identity['protocol']['patience'] for s in saved.values())
    return dict(schema=SCHEMA, status='MIN_CONDITIONAL_PROPAGATION_PAIRED_STAGE_COMPLETE' if complete else 'MIN_CONDITIONAL_PROPAGATION_PAIRED_STAGE_PARTIAL',
        common_epoch=common, runs={a: summary_for(s, args, identity['protocol']) for a, s in saved.items()}, comparisons=comparisons,
        same_completed_training_budget=len(saved) == len(args.arms) and len({s['global_updates'] for s in saved.values()}) == 1,
        primary_baseline='original_IGSTGNN_not_in_this_training_stage', predictive_gain_claim=False, test_evaluated=False)


def child_command(args, arm):
    command = [sys.executable, '-u', str(Path(__file__).resolve()), '--run-dir', str(args.run_dir),
        '--data-dir', str(args.data_dir), '--graph-dir', str(args.graph_dir), '--protocol', str(args.protocol),
        '--source-run', str(args.source_run), '--reference-run', str(args.reference_run), '--epochs', str(args.epochs),
        '--seed', str(args.seed), '--device', args.device, '--threads', str(args.threads),
        '--receiver-chunk-size', str(args.receiver_chunk_size), '--arms', *args.arms, '--worker', arm]
    if not args.checkpoint_messages:
        command.append('--no-checkpoint-messages')
    if args.batch_size is not None:
        command += ['--batch-size', str(args.batch_size)]
    if args.check:
        command.append('--check')
    return command


def run_worker(args, arm, session):
    with (session/(arm+'.log')).open('w', encoding='utf-8') as stream:
        process = subprocess.Popen(child_command(args, arm), cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding='utf-8', env=dict(os.environ, **{WORKER_ENV: '1', 'PYTHONUNBUFFERED': '1', 'PYTHONIOENCODING': 'utf-8'}))
        try:
            for line in process.stdout:
                print(line, end='', flush=True)
                stream.write(line)
                stream.flush()
            code = process.wait()
        except BaseException:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
            raise
        finally:
            process.stdout.close()
    write_json(session/(arm+'_process_result.json'), dict(exit_code=code))
    return code


def main(argv=None):
    args = parse_args(argv)
    existing = (args.run_dir/'identity.json').is_file()
    require(not existing or legacy.read_json(args.run_dir/'identity.json').get('schema') == SCHEMA, 'Use a new conditional-propagation run directory')
    require(not args.worker or existing, 'Worker requires prepared identity')
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    require(torch.device(args.device).type in ('cpu', 'cuda') and (not args.device.startswith('cuda') or torch.cuda.is_available()), 'Device unavailable')
    logger = logging.getLogger('min_conditional_propagation')
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler(sys.stdout))
    args.run_dir.mkdir(parents=True, exist_ok=True)
    lock_root = args.run_dir/args.worker if args.worker else args.run_dir
    lock_root.mkdir(exist_ok=True)
    with legacy.run_lock(lock_root):
        before = source_snapshot(args)
        try:
            with readonly_source_lock(args.source_run):
                protocol, data, graph, adjacency, initialization, identity, source = prepare(args, logger, skip_preflight=existing)
                verify_identity(args, identity, initialization, allow_new=not args.worker)
                if args.worker:
                    train_arm(args, args.worker, protocol, data, identity, graph, logger, source, adjacency)
                    return
            write_json(args.run_dir/'data_audit.json', data.audit)
            session = args.run_dir/('session_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6])
            session.mkdir()
            write_json(session/'environment.json', legacy.environment(torch.device(args.device)))
            write_json(session/'request.json', dict(epochs=args.epochs, arms=args.arms, only=args.only,
                receiver_chunk_size=args.receiver_chunk_size, checkpoint_messages=args.checkpoint_messages))
            print('Run directory: '+str(args.run_dir), flush=True)
            del data, graph, adjacency, source
            gc.collect()
            for arm in (args.only or args.arms):
                code = run_worker(args, arm, session)
                require(code == 0, 'Worker failed; committed checkpoints preserved. See '+str(session/(arm+'.log')))
            report = comparison(args, identity)
            write_json(session/'comparison.json', report)
            write_json(args.run_dir/'latest_stage.json', report)
            print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
        finally:
            after = source_snapshot(args)
            require(before == after, 'Source MIN artifacts changed')
            write_json(lock_root/'source_preservation.json', dict(source_artifacts_preserved=True, files_sha256=before))


if __name__ == '__main__':
    main()
