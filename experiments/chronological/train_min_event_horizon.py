"""Paired EVENT-best warmstarts with one bounded forecast-horizon response gate.

The source run stays read-only. CONTROL and HORIZON reset optimizer/scheduler
equally and jointly update the existing core; this is not exact source resume.
"""
import argparse
from contextlib import contextmanager
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
from types import SimpleNamespace
import uuid

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.chronological import train_b3_r_minimal_paper as paper
from experiments.chronological import train_min_early_event as early
from experiments.chronological import train_internal_spatial as legacy
from experiments.chronological.smoke import sha256, set_seed
from experiments.chronological.train import cpu_state, save_checkpoint
from src.engines.igstgnn_engine import IGSTGNN_Engine
from src.models.incident_spatial import STAT_NAMES, require
from src.models.incident_spatial_residual import tensor_state_hash
from src.models.incident_spatial_event_horizon import EventHorizonIGSTGNN
from src.utils.igstgnn_published_events import EventPublishedData
from src.utils.metrics import compute_all_metrics, masked_mae

HERE = Path(__file__).resolve().parent
PROTOCOL = HERE/'min_event_horizon_v1.json'
SCHEMA = 'min_event_horizon_warmstart_run_v1'
ARMS = {'CONTROL': 'control', 'HORIZON': 'horizon'}
WORKER_ENV = 'MIN_EVENT_HORIZON_WORKER'
SOURCE_FILES = ('identity.json', 'initialization.json', 'data_audit.json',
                'EVENT/last.pt', 'EVENT/history.json', 'EVENT/summary.json')
DEFAULT_SOURCE = REPO/'experiments/chronological_runs/contra_min_early_e1_bs64_seed2025'


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=paper.default_data())
    parser.add_argument('--graph-dir', type=Path, default=HERE/'internal_spatial_graph_is11')
    parser.add_argument('--protocol', type=Path, default=PROTOCOL)
    parser.add_argument('--source-run', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--arms', nargs='+', choices=ARMS, default=list(ARMS))
    parser.add_argument('--only', nargs='+', choices=ARMS, help='Run only these members of the saved arm set')
    parser.add_argument('--epochs', type=int, help='Absolute target epoch: default 5, or 1 with --check')
    parser.add_argument('--batch-size', type=int, help='Default 64; fixed 2 for --check')
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--threads', type=int, default=3)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--evaluate-after', action='store_true',
                        help='After training, replay saved CONTROL/HORIZON best validation and audit responses in this job')
    parser.add_argument('--worker', choices=ARMS, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    args.epochs = (1 if args.check else 5) if args.epochs is None else args.epochs
    require(1 <= args.epochs <= 100 and args.threads >= 1 and 0 <= args.seed < 2**32,
            'Invalid epoch/threads/seed')
    require(args.batch_size is None or args.batch_size >= 1, 'Batch size must be positive')
    require(len(set(args.arms)) == len(args.arms), 'Duplicate arms')
    args.arms = [arm for arm in ARMS if arm in args.arms]
    require(args.only is None or (len(set(args.only)) == len(args.only)
            and set(args.only) <= set(args.arms)), '--only must be a unique subset of --arms')
    require(not args.check or args.epochs == 1,
            'Engineering check is one small epoch without test evaluation')
    require(not args.worker or (os.environ.get(WORKER_ENV) == '1' and args.run_dir is not None
            and args.worker in args.arms), 'Internal worker requires parent launcher; use --only')
    require(not args.worker or not args.evaluate_after, 'Workers cannot recursively evaluate-after')
    if args.run_dir is None:
        args.run_dir = REPO/'experiments/chronological_runs'/('contra_min_event_horizon_'+('check_' if args.check else '')
                       +datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6])
    for name in ('data_dir', 'graph_dir', 'protocol', 'source_run', 'run_dir'):
        setattr(args, name, getattr(args, name).resolve())
    require(args.run_dir != args.source_run and args.source_run not in args.run_dir.parents
            and args.run_dir not in args.source_run.parents, 'New run must be separate from source run and its ancestors')
    return args


def load_protocol(args):
    protocol = legacy.read_json(args.protocol)
    require(protocol == legacy.read_json(PROTOCOL), 'Horizon protocol changed; use a separate version')
    return dict(protocol, batch_size=args.batch_size) if args.batch_size is not None else protocol


def source_snapshot(args):
    require(all((args.source_run/name).is_file() for name in SOURCE_FILES), 'Missing source EVENT experiment artifacts')
    return {name: sha256(args.source_run/name) for name in SOURCE_FILES}


@contextmanager
def readonly_source_lock(root):
    path = Path(root)/'active.lock'
    require(path.is_file() and path.stat().st_size > 0, 'Source run requires its existing lock; no source files will be created')
    stream = path.open('rb')
    try:
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as error:
        stream.close()
        raise ValueError('Source EVENT run is active; finish its job before this warmstart') from error
    try:
        yield
    finally:
        stream.close()


def load_source(args, hashes=None):
    before = source_snapshot(args)
    identity = legacy.read_json(args.source_run/'identity.json')
    require(identity.get('schema') == early.SCHEMA and identity.get('arm_modes', {}).get('EVENT') == 'event',
            'Expected the existing E1 EVENT run; MIN and older candidates cannot initialize this experiment')
    require(identity.get('engineering_check') is args.check,
            '--check requires an engineering source run; formal source requires full validation')
    require(identity['runtime']['torch'] == str(torch.__version__)
            and identity['runtime']['cuda'] == torch.version.cuda
            and identity['runtime']['device_type'] == torch.device(args.device).type,
            'Source torch/CUDA/device type changed; GPU model names may differ')
    require(identity['source_sha256'] == early.source_fingerprints(), 'Source E1 code fingerprint changed')
    require(legacy.read_json(args.source_run/'initialization.json') == identity['initialization'], 'Source initialization certificate changed')
    require(identity['protocol_sha256'] == sha256(early.PROTOCOL), 'Source E1 protocol changed')
    if hashes is not None:
        require(identity['package'] == hashes, 'Source release fingerprints differ')
    require(identity['graph_inputs_sha256'] == {name: sha256(args.graph_dir/name)
            for name in ('graph.json', 'stations.csv', 'source_summary.json')}, 'Source graph fingerprints differ')
    saved = torch.load(args.source_run/'EVENT/last.pt', map_location='cpu', weights_only=False)
    early.validate_saved(saved, identity, 'EVENT')
    require(legacy.read_json(args.source_run/'EVENT/history.json') == saved['history'], 'Source history differs from checkpoint')
    selected = min(saved['history'], key=lambda row: row['validation_mae'])
    companion = legacy.read_json(args.source_run/'EVENT/summary.json')
    require(selected['epoch'] == companion.get('best_epoch') == saved['history'][-1]['best_epoch']
            and selected['validation_mae'] == companion.get('best_metric') == saved['best_metric']
            and companion.get('completed_epoch') == saved['completed_epoch'], 'Source selected-best counters disagree')
    require(source_snapshot(args) == before, 'Source artifacts changed while loading')
    return dict(identity=identity, saved=saved, selected=selected, files_sha256=before,
                best_model_sha256=tensor_state_hash(saved['best_model_state']))


def build_core(args, nodes, graph, device, source):
    setup = SimpleNamespace(**vars(args))
    setup.protocol, setup.seed = early.PROTOCOL, source['identity']['seed']
    setup.batch_size = source['identity']['protocol']['batch_size']
    core, _ = early.build_arm(setup, 'EVENT', nodes, graph, device)
    core.load_state_dict(source['saved']['best_model_state'], strict=True)
    core.requires_grad_(True)
    return core


def build_arm(args, arm, nodes, graph, device, source=None):
    require(arm in ARMS, 'Unknown horizon arm')
    source = load_source(args) if source is None else source
    core = build_core(args, nodes, graph, device, source)
    model = EventHorizonIGSTGNN(core, mode=ARMS[arm], checkpoint_branches=load_protocol(args)['horizon_spec']['checkpoint_branches'])
    set_seed(args.seed)
    return model, dict(mode=ARMS[arm], source_best_epoch=source['selected']['epoch'],
                       source_completed_epoch=source['saved']['completed_epoch'],
                       source_best_model_sha256=source['best_model_sha256'],
                       shared_core_warmstart=True, fresh_optimizer_and_scheduler=True)


def event_tensors(payload, device):
    long_names = {'type', 'description'}
    return {key: torch.as_tensor(value, device=device,
            dtype=torch.long if key in long_names else torch.float32 if key == 'relation' else torch.bool)
            for key, value in payload.items()}


def preflight(args, data, graph, source):
    reference = build_core(args, len(data.station_ids), graph, torch.device('cpu'), source)
    reference.eval()
    batch = next(data.loaders['train_loader'].get_iterator())
    x = torch.as_tensor(batch['x_data'][:2], dtype=torch.float32)
    events = event_tensors({k: v[:2] for k, v in batch['event_data'].items()}, 'cpu')
    base_state = reference.state_dict()
    with torch.inference_mode():
        expected = reference(x, incident_data=events)
    manifests, shared_state = {}, None
    for arm in args.arms:
        model, certificate = build_arm(args, arm, len(data.station_ids), graph, torch.device('cpu'), source)
        state = model.state_dict()
        require(all(torch.equal(value, state['core.'+name]) for name, value in base_state.items()), 'Source core weights changed: '+arm)
        model.eval()
        with torch.inference_mode():
            actual = model(x, incident_data=events)
        require(torch.equal(expected, actual), 'Gate-one warmstart differs from selected source EVENT: '+arm)
        require(all(parameter.requires_grad for parameter in model.core.parameters()), 'Entire existing EVENT core must train jointly')
        if shared_state is not None:
            require(state.keys() == shared_state.keys()
                    and all(torch.equal(state[k], shared_state[k]) for k in state),
                    'CONTROL/HORIZON parameter layout or initial tensors differ')
        shared_state = state
        manifests[arm] = dict(parameters=sum(p.numel() for p in model.parameters()),
                              common_state_sha256=tensor_state_hash(base_state),
                              initial_state_sha256=tensor_state_hash(state),
                              added_parameters=model.horizon_offset.numel(), conversion=certificate)
        del model
    return dict(status='MIN_EVENT_HORIZON_INITIALIZATION_PAIR_PASS', arms=manifests,
                gate_one_source_prediction_abs_max=0.,
                common_parameters_exact=True, same_parameter_layout_and_initialization=True,
                training_initialization='shared_source_EVENT_best_warmstart_fresh_paired_optimizer', loaded_source_checkpoint=True,
                event_scope='published_archive_attributes_conditional_offline')


def source_fingerprints():
    names = ('experiments/chronological/train_min_event_horizon.py', 'src/models/incident_spatial_event_horizon.py')
    return dict(early.source_fingerprints(), **{name: sha256(REPO/name) for name in names})


def make_identity(args, protocol, hashes, graph_info, initialization, source):
    return dict(schema=SCHEMA, arm_modes={arm: ARMS[arm] for arm in args.arms}, protocol=protocol,
                protocol_sha256=sha256(args.protocol), package=hashes, graph=graph_info,
                graph_inputs_sha256={name: sha256(args.graph_dir/name)
                                     for name in ('graph.json', 'stations.csv', 'source_summary.json')},
                source_sha256=source_fingerprints(), seed=args.seed, engineering_check=args.check,
                batch_size=2 if args.check else protocol['batch_size'], threads=args.threads,
                initialization=initialization, clock_policy='last_published_history_slot_no_extra_gap',
                training_initialization='shared_source_EVENT_best_warmstart_fresh_paired_optimizer', loaded_source_checkpoint=True,
                source=dict(files_sha256=source['files_sha256'], best_epoch=source['selected']['epoch'],
                            completed_epoch=source['saved']['completed_epoch'],
                            best_model_sha256=source['best_model_sha256'], original_schema=source['identity']['schema'],
                            best_metrics=[source['selected'][key] for key in ('validation_mae', 'validation_mape', 'validation_rmse')]),
                source_optimizer_restored=False, source_scheduler_restored=False, source_exact_resume=False,
                source_reference_files_sha256={name: sha256(args.run_dir/name) for name in ('source_reference.json', 'source_reference.npz')}
                    if all((args.run_dir/name).is_file() for name in ('source_reference.json', 'source_reference.npz')) else None,
                full_horizon_loss_every_update=True,
                event_scope='published_archive_attributes_conditional_offline',
                excluded_event_fields=['Incident Time', 'durations', 'Holiday', 'incident_position'],
                runtime=dict(torch=str(torch.__version__), cuda=torch.version.cuda,
                             device_type=torch.device(args.device).type,
                             cublas_workspace_config=os.environ['CUBLAS_WORKSPACE_CONFIG']),
                test_evaluation_policy='explicit_only', test_payload_deserialized=True)


def verify_identity(args, identity, initialization, allow_new):
    path = args.run_dir/'identity.json'
    if path.exists():
        require(legacy.read_json(path) == identity, 'Run identity changed; existing checkpoints preserved')
        require(legacy.read_json(args.run_dir/'initialization.json') == initialization,
                'Initialization certificate changed')
    else:
        require(allow_new and {p.name for p in args.run_dir.iterdir()} <= {'active.lock'},
                'Use a fresh horizon directory; old experiments are preserved')
        legacy.write_json(path, identity)
        legacy.write_json(args.run_dir/'initialization.json', initialization)


class HorizonEngine(early.EarlyEventEngine):
    def __init__(self, arm, **kwargs):
        super().__init__(arm=arm, **kwargs)
        self._cl_len = self._horizon
        self._projection_handle = self._optimizer.register_step_post_hook(lambda optimizer, args, kwargs: self.model.project_horizon_weights_())

    def _update_curriculum_length(self):
        self._iter_cnt += 1
        self._cl_len = self._horizon

    def _prepare_batch(self, batch):
        if not isinstance(batch, dict):
            return super()._prepare_batch(batch)
        events = event_tensors(batch['event_data'], self._device)
        return self._as_float_tensor(batch['x_data']), self._as_float_tensor(batch['y_data']), events, None

    def _predict(self, X, label, incident_data, sensor_data):
        if not self.collect_diagnostics:
            return self.model(X, label, incident_data=incident_data)
        result = self.model(X, label, incident_data=incident_data, return_spatial_details=True)
        self._record(result['spatial'])
        details = result['horizon']
        self.diagnostics['paired_core_calls'] = self.diagnostics.get('paired_core_calls', 0)+details['paired_core_calls']
        for name in ('raw_delta', 'applied_delta'):
            value = details[name]
            if value is None:
                continue
            self.diagnostics[name+'_abs_sum'] = self.diagnostics.get(name+'_abs_sum', 0.)+float(value.abs().sum().cpu())
            self.diagnostics[name+'_abs_max'] = max(self.diagnostics.get(name+'_abs_max', 0.), float(value.abs().max().cpu()))
        return result['prediction']

    def _record(self, context):
        self.diagnostics['forward_calls'] = self.diagnostics.get('forward_calls', 0)+1
        for key, value in context.get('early_event_diagnostics', {}).items():
            value = float(value.detach().cpu()) if torch.is_tensor(value) else float(value)
            require(np.isfinite(value), 'Non-finite early-event diagnostic: '+key)
            if key.endswith('_max'):
                self.diagnostics[key] = max(self.diagnostics.get(key, 0.), value)
            else:
                self.diagnostics[key] = self.diagnostics.get(key, 0.)+value
        for records in context.get('records', []):
            self.diagnostics['local_substeps'] = self.diagnostics.get('local_substeps', 0)+records.shape[0]
            for index, name in enumerate(STAT_NAMES):
                values = records[:, index].detach()
                if name.endswith('_max'):
                    value = float(values.max().cpu())
                    self.diagnostics[name] = max(self.diagnostics.get(name, 0.), value)
                elif name.endswith('_min'):
                    value = float(values.min().cpu())
                    self.diagnostics[name] = min(self.diagnostics.get(name, value), value)
                else:
                    value = float(values.sum().cpu())
                    self.diagnostics[name] = self.diagnostics.get(name, 0.)+value
                require(np.isfinite(value), 'Non-finite MIN exchange diagnostic: '+name)

    def evaluate(self, mode):
        require(mode == 'val', 'Use explicit best-checkpoint test evaluator')
        self.diagnostics = {}
        self.collect_diagnostics = True
        try:
            return super().evaluate(mode)
        finally:
            self.collect_diagnostics = False


def make_engine(args, arm, protocol, data, model, logger):
    optimizer = torch.optim.Adam([
        dict(params=list(model.core.parameters()), lr=protocol['learning_rate'], weight_decay=protocol['weight_decay']),
        dict(params=[model.horizon_offset], lr=protocol['horizon_learning_rate'], weight_decay=0.)],
        eps=protocol['adam_eps'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, protocol['lr_milestones'], gamma=protocol['lr_gamma'])
    batches = data.loaders['train_loader'].num_batch
    return HorizonEngine(arm=arm, device=torch.device(args.device), model=model,
        dataloader=data.loaders, scaler=data.scaler, sampler=None, loss_fn=masked_mae,
        lrate=protocol['learning_rate'], optimizer=optimizer, scheduler=scheduler,
        clip_grad_value=protocol['clip_grad_norm'], max_epochs=protocol['max_epochs'],
        patience=protocol['patience'], log_dir=str(args.run_dir/arm), logger=logger, seed=args.seed,
        cl_step=protocol['cl_epoch']*batches, warm_step=protocol['warm_epoch']*batches,
        horizon=protocol['horizon'], incident=False)


def summary_for(saved, args, protocol):
    early = saved['wait'] >= protocol['patience']
    return dict(status='ENGINEERING_CHECK_PASS' if args.check else 'EARLY_STOPPED' if early else 'PAUSED_AT_EPOCH_BOUNDARY',
                arm=saved['arm'], mode=ARMS[saved['arm']], completed_epoch=saved['completed_epoch'],
                requested_epoch=args.epochs, best_epoch=saved['history'][-1]['best_epoch'],
                best_metric=saved['best_metric'], wait=saved['wait'], early_stopped=early,
                global_updates=saved['global_updates'], curriculum_length=saved['curriculum_length'],
                parameters=saved['parameters'], batch_size=2 if args.check else protocol['batch_size'],
                event_inputs_used=True, semantic_inputs_used=True,
                training_initialization='shared_source_EVENT_best_warmstart_fresh_paired_optimizer', loaded_source_checkpoint=True,
                source_exact_resume=False, source_optimizer_restored=False, source_scheduler_restored=False,
                trainable_scope='all_existing_EVENT_core_plus_horizon_offsets_for_candidate',
                source_best_epoch=saved['identity']['source']['best_epoch'],
                best_stage_epoch=saved['best_epoch'], stage_epoch_zero_eligible_for_selection=True,
                best_is_source_epoch_zero=saved['best_epoch'] == 0,
                last_horizon_weights=saved['history'][-1]['horizon_weights'],
                best_horizon_weights=(1.+saved['best_model_state']['horizon_offset']).tolist(),
                full_horizon_loss_every_update=True,
                graph_and_capacity_inputs='original_unconditioned_MIN_history',
                event_scope='published_archive_attributes_conditional_offline',
                capacity_scope='dimensionless_latent_exchange_budget', future_capacity_trajectory=False,
                selection_metric='published_engine_validation_masked_mae', test_evaluated=False,
                predictive_gain_claim=False, checkpoint='last.pt', best_model_state_key='best_model_state')


def validate_saved(saved, identity, arm):
    require(saved.get('identity') == identity and saved.get('arm') == arm, 'Checkpoint identity/arm mismatch')
    history = saved.get('history')
    require(isinstance(history, list) and history and len(history) == saved.get('completed_epoch')
            and [row['epoch'] for row in history] == list(range(1, len(history)+1)), 'Invalid checkpoint history')
    require(history[-1]['best_metric'] == saved['best_metric']
            and history[-1]['global_updates'] == saved['global_updates'], 'Checkpoint counters disagree')
    require(saved.get('best_epoch') == history[-1]['best_epoch'] and 0 <= saved['best_epoch'] <= saved['completed_epoch'],
            'Invalid horizon best-stage epoch')
    require(saved.get('curriculum_length') == 12 and all(row.get('curriculum_length') == 12 for row in history),
            'Horizon experiment must always train all twelve forecast slots')
    require(saved['model_state'].keys() == saved['best_model_state'].keys(), 'Checkpoint state keys differ')
    for key in ('model_state', 'best_model_state'):
        offset = saved[key].get('horizon_offset')
        require(isinstance(offset, torch.Tensor) and offset.shape == (12,) and bool(torch.isfinite(offset).all())
                and bool(((offset >= -1.) & (offset <= 0.)).all())
                and (arm != 'CONTROL' or bool((offset == 0.).all())), 'Invalid saved horizon offsets: '+key)
    for key, value in saved['model_state'].items():
        best = saved['best_model_state'][key]
        require(value.shape == best.shape and value.dtype == best.dtype
                and bool(torch.isfinite(value).all()) and bool(torch.isfinite(best).all()),
                'Invalid checkpoint tensor: '+key)


def repair_sidecars(root, saved, args, protocol):
    def read_companion(path):
        if not path.exists():
            return None
        try:
            return legacy.read_json(path)
        except (json.JSONDecodeError, UnicodeDecodeError):
            path.with_name(path.name+'.interrupted_'+uuid.uuid4().hex[:6]).write_bytes(path.read_bytes())
            return None
    previous = read_companion(root/'history.json')
    history = saved['history']
    require(previous is None or (isinstance(previous, list) and len(previous) <= len(history)
            and previous == history[:len(previous)]), 'JSON history conflicts with checkpoint; refusing overwrite')
    summary = read_companion(root/'summary.json')
    require(summary is None or (summary.get('arm') == saved['arm']
            and summary.get('completed_epoch', 0) <= saved['completed_epoch']), 'JSON summary conflicts with checkpoint')
    if summary is not None:
        completed = summary.get('completed_epoch')
        require(type(completed) is int and 1 <= completed <= len(history), 'Invalid JSON summary epoch')
        committed = history[completed-1]
        require(all(summary.get(key) == committed[key] for key in
                    ('best_epoch', 'best_metric', 'global_updates', 'curriculum_length')),
                'JSON summary counters conflict with checkpoint; refusing overwrite')
    legacy.write_json(root/'history.json', history)
    result = summary_for(saved, args, protocol)
    # Preserve a previous explicit test report when training made no updates.
    if summary and summary.get('test_evaluated') and summary.get('completed_epoch') == saved['completed_epoch']:
        result.update(test_evaluated=True, test_metrics=summary['test_metrics'])
    legacy.write_json(root/'summary.json', result)
    return result


def restore(saved, identity, arm, engine, data):
    validate_saved(saved, identity, arm)
    engine.model.load_state_dict(saved['model_state'], strict=True)
    engine._optimizer.load_state_dict(saved['optimizer_state'])
    engine._lr_scheduler.load_state_dict(saved['scheduler_state'])
    engine._iter_cnt, engine._cl_len = saved['global_updates'], saved['curriculum_length']
    data.loaders['train_loader'].set_order(saved['train_order'])
    np.random.set_state(saved['numpy_rng'])
    torch.set_rng_state(saved['torch_rng'])
    if torch.device(engine._device).type == 'cuda':
        torch.cuda.set_rng_state_all(saved['cuda_rng'])


def train_arm(args, arm, protocol, data, identity, graph, logger, source=None):
    root = args.run_dir/arm
    root.mkdir(exist_ok=True)
    model, _ = build_arm(args, arm, len(data.station_ids), graph, torch.device(args.device), source)
    engine = make_engine(args, arm, protocol, data, model, logger)
    path = root/'last.pt'
    baseline = legacy.read_json(args.run_dir/'source_reference.json')
    history, best, wait, saved = [], baseline['actual'][0], 0, None
    best_state, best_epoch, prior_summary = cpu_state(model.state_dict()), 0, None
    if path.is_file():
        saved = torch.load(path, map_location='cpu', weights_only=False)
        validate_saved(saved, identity, arm)
        prior_summary = repair_sidecars(root, saved, args, protocol)
        restore(saved, identity, arm, engine, data)
        history, best, wait, best_state = saved['history'], saved['best_metric'], saved['wait'], saved['best_model_state']
        best_epoch = saved['best_epoch']
    starting_epoch = len(history)
    parameters = sum(p.numel() for p in model.parameters())
    for epoch in range(starting_epoch+1, args.epochs+1):
        if wait >= protocol['patience']:
            break
        start = time.perf_counter()
        train_mae, train_mape, train_rmse = engine.train_batch()
        val_mae, val_mape, val_rmse = engine.evaluate('val')
        require(np.isfinite([train_mae, train_mape, train_rmse, val_mae, val_mape, val_rmse]).all(),
                'Non-finite published training/validation metrics')
        improved = val_mae < best
        if improved:
            best, wait, best_state, best_epoch = val_mae, 0, cpu_state(model.state_dict()), epoch
        else:
            wait += 1
        lr = engine._optimizer.param_groups[0]['lr']
        horizon_lr = engine._optimizer.param_groups[1]['lr']
        engine._lr_scheduler.step()
        row = dict(epoch=epoch, train_mae=float(train_mae), train_rmse=float(train_rmse), train_mape=float(train_mape),
                   validation_mae=float(val_mae), validation_rmse=float(val_rmse), validation_mape=float(val_mape),
                   best_metric=best, best_epoch=best_epoch,
                   global_updates=engine._iter_cnt, curriculum_length=engine._cl_len,
                   learning_rate_used=lr, next_learning_rate=engine._optimizer.param_groups[0]['lr'],
                   horizon_learning_rate_used=horizon_lr,
                   horizon_weights=model.horizon_weights().detach().cpu().tolist(),
                   train_order_sha256=hashlib.sha256(data.loaders['train_loader'].order.tobytes()).hexdigest(),
                   validation_diagnostics=engine.diagnostics, seconds=time.perf_counter()-start)
        history.append(row)
        saved = dict(identity=identity, arm=arm, completed_epoch=epoch, history=history, parameters=parameters,
                     model_state=cpu_state(model.state_dict()), best_model_state=best_state, best_metric=best, best_epoch=best_epoch, wait=wait,
                     optimizer_state=engine._optimizer.state_dict(), scheduler_state=engine._lr_scheduler.state_dict(),
                     global_updates=engine._iter_cnt, curriculum_length=engine._cl_len,
                     train_order=data.loaders['train_loader'].order.copy(), numpy_rng=np.random.get_state(),
                     torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if args.device.startswith('cuda') else [])
        save_checkpoint(path, saved)
        legacy.write_json(root/'history.json', history)
        legacy.write_json(root/'summary.json', summary_for(saved, args, protocol))
        print(json.dumps(dict(arm=arm, **row), allow_nan=False), flush=True)
        gc.collect()
    require(saved is not None, 'No committed training checkpoint')
    result = summary_for(saved, args, protocol)
    result['optimizer_updates_this_invocation'] = saved['global_updates']-(0 if starting_epoch == 0 else history[starting_epoch-1]['global_updates'])
    if len(history) == starting_epoch and prior_summary and prior_summary.get('test_evaluated'):
        result.update(test_evaluated=True, test_metrics=prior_summary['test_metrics'])
    legacy.write_json(root/'summary.json', result)
    print(json.dumps(dict(arm=arm, status=result['status'], completed_epoch=result['completed_epoch'],
                         best_epoch=result['best_epoch'], best_metric=result['best_metric'],
                         optimizer_updates_this_invocation=result['optimizer_updates_this_invocation'],
                         output=str(root/'summary.json')), allow_nan=False), flush=True)
    return result


def comparison(args, identity):
    stage_zero_metric = legacy.read_json(args.run_dir/'source_reference.json')['actual'][0]
    histories, summaries = {}, {}
    for arm in args.arms:
        path = args.run_dir/arm/'last.pt'
        if not path.is_file():
            continue
        saved = torch.load(path, map_location='cpu', weights_only=False)
        validate_saved(saved, identity, arm)
        histories[arm], summaries[arm] = saved['history'], summary_for(saved, args, identity['protocol'])
    common = min((len(h) for h in histories.values()), default=0) if len(histories) == len(args.arms) else 0
    comparisons = {}
    if common:
        for control, candidate in [('CONTROL', 'HORIZON')]:
            if control not in histories or candidate not in histories:
                continue
            c, r = histories[control][:common], histories[candidate][:common]
            comparisons[candidate+'_vs_'+control] = dict(
                common_epoch_mae_gain=c[-1]['validation_mae']-r[-1]['validation_mae'],
                common_prefix_best_mae_gain=min([stage_zero_metric]+[x['validation_mae'] for x in c])
                    -min([stage_zero_metric]+[x['validation_mae'] for x in r]),
                candidate_common_last_five=[x['validation_mae'] for x in r[-5:]],
                control_common_last_five=[x['validation_mae'] for x in c[-5:]])
        require(all(len({histories[a][e]['train_order_sha256'] for a in histories}) == 1 for e in range(common)),
                'Paired training sample order differs')
    complete = len(summaries) == len(args.arms) and all(
        s['completed_epoch'] >= args.epochs or s['early_stopped'] for s in summaries.values())
    budgets = {arm: row['completed_epoch'] for arm, row in summaries.items()}
    updates = {arm: row['global_updates'] for arm, row in summaries.items()}
    equal_budget = len(summaries) == len(args.arms) and len(set(budgets.values())) == 1 and len(set(updates.values())) == 1
    return dict(status='MIN_EVENT_HORIZON_PAIRED_STAGE_COMPLETE' if complete else 'MIN_EVENT_HORIZON_PAIRED_STAGE_PARTIAL',
                common_epoch=common, runs=summaries, comparisons=comparisons,
                completed_stage_epochs=budgets, optimizer_updates=updates, same_completed_training_budget=equal_budget,
                interpretation='shared_EVENT_best_warmstart_with_fresh_paired_optimizer_common_prefix_comparison',
                independent_best_scope='equal_completed_stage_budget' if equal_budget else 'unequal_completed_stage_budget_exploratory',
                training_initialization='shared_source_EVENT_best_warmstart_fresh_paired_optimizer',
                source_best_epoch=identity['source']['best_epoch'], source_exact_resume=False, existing_MIN_retrained=False,
                predictive_gain_claim=False)


def prepare(args, logger):
    protocol = load_protocol(args)
    hashes = paper.verify_release(args.data_dir, protocol)
    data = EventPublishedData(args.data_dir, protocol, logger, check=args.check)
    graph, _, graph_info = legacy.load_graph(args.graph_dir, data.station_ids, protocol['graph_sha256'])
    source = load_source(args, hashes)
    require(data.loaders['val_loader'].bs == source['identity']['batch_size'],
            'This paired stage keeps the source original validation/training batch size')
    initialization = preflight(args, data, graph, source)
    identity = make_identity(args, protocol, hashes, graph_info, initialization, source)
    return protocol, data, graph, initialization, identity, source


def validation_predictions(model, data, device):
    model.eval()
    predictions, labels, partitions = [], [], []
    with torch.inference_mode():
        for batch in data.loaders['val_loader'].get_iterator():
            x = torch.as_tensor(batch['x_data'], dtype=torch.float32, device=device)
            y = torch.as_tensor(batch['y_data'], dtype=torch.float32, device=device)
            events = event_tensors(batch['event_data'], device)
            prediction = model(x, incident_data=events)
            predictions.append(data.scaler.inverse_transform(prediction).squeeze(-1).cpu())
            labels.append(data.scaler.inverse_transform(y).squeeze(-1).cpu())
            partitions.append(len(x))
    prediction, target = torch.cat(predictions), torch.cat(labels)
    require(bool(torch.isfinite(prediction).all()) and bool(torch.isfinite(target).all()), 'Nonfinite full validation reference')
    null = target.min() if target.min() < 1 else torch.tensor(0.)
    actual = compute_all_metrics(prediction, target, null)
    return dict(prediction=prediction.numpy(), target=target.numpy(), valid=(target != null).numpy(),
                sample_indices=data.audit['counts']['train_samples']+np.arange(len(target), dtype=np.int64),
                station_ids=np.asarray(data.station_ids, np.int64)), dict(actual=actual, mask_value=float(null), batch_partition=partitions)


def create_source_reference(args, data, graph, source):
    reference = build_core(args, len(data.station_ids), graph, torch.device(args.device), source)
    arrays, certificate = validation_predictions(reference, data, torch.device(args.device))
    expected = [source['selected'][key] for key in ('validation_mae', 'validation_mape', 'validation_rmse')]
    require(np.allclose(certificate['actual'], expected, atol=1e-5, rtol=1e-6),
            'Source EVENT best full ON replay failed; new optimizer updates are blocked')
    np.savez_compressed(args.run_dir/'source_reference.npz', **arrays)
    result = dict(certificate, status='SOURCE_EVENT_BEST_FULL_ON_REPLAY_PASS', expected=expected,
        source_best_epoch=source['selected']['epoch'], source_best_model_sha256=source['best_model_sha256'],
        validation_rows=len(arrays['target']), batch_size=data.loaders['val_loader'].bs,
        engineering_only=args.check, atol=1e-5, rtol=1e-6,
        archive_sha256=sha256(args.run_dir/'source_reference.npz'), optimizer_updates=0, test_evaluated=False)
    legacy.write_json(args.run_dir/'source_reference.json', result)
    print('Source EVENT best full ON replay: PASS; new stage uses a fresh paired optimizer', flush=True)
    del reference
    gc.collect()
    return result


def verify_source_reference(args, identity):
    expected_hashes = identity.get('source_reference_files_sha256')
    require(expected_hashes and all(sha256(args.run_dir/name) == value for name, value in expected_hashes.items()),
            'Saved source reference artifacts changed')
    baseline = legacy.read_json(args.run_dir/'source_reference.json')
    require(baseline.get('status') == 'SOURCE_EVENT_BEST_FULL_ON_REPLAY_PASS'
            and baseline.get('source_best_epoch') == identity['source']['best_epoch']
            and baseline.get('source_best_model_sha256') == identity['source']['best_model_sha256']
            and baseline.get('expected') == identity['source']['best_metrics']
            and baseline.get('engineering_only') is args.check
            and baseline.get('archive_sha256') == expected_hashes['source_reference.npz']
            and np.allclose(baseline['actual'], baseline['expected'], atol=1e-5, rtol=1e-6), 'Invalid source ON replay certificate')
    return baseline


def child_command(args, arm):
    command = [sys.executable, '-u', str(Path(__file__).resolve()), '--run-dir', str(args.run_dir),
               '--data-dir', str(args.data_dir), '--graph-dir', str(args.graph_dir), '--protocol', str(args.protocol),
               '--epochs', str(args.epochs), '--seed', str(args.seed), '--device', args.device,
               '--threads', str(args.threads), '--source-run', str(args.source_run), '--arms', *args.arms, '--worker', arm]
    if args.batch_size is not None:
        command += ['--batch-size', str(args.batch_size)]
    if args.check:
        command.append('--check')
    return command


def run_worker(args, arm, session):
    """Do not release the parent's run lock while an interrupted child lives."""
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
            legacy.write_json(session/(arm+'_process_result.json'),
                              dict(exit_code=process.returncode, launcher_interrupted=True))
            raise
        finally:
            process.stdout.close()
    legacy.write_json(session/(arm+'_process_result.json'), dict(exit_code=code, launcher_interrupted=False))
    return code


def evaluate_after_training(args):
    """Called outside the parent run lock; the evaluator obtains its own lock."""
    require(all((args.run_dir/arm/'last.pt').is_file() for arm in ('CONTROL', 'HORIZON')),
            'Automatic evaluation requires saved CONTROL and HORIZON checkpoints; training results are preserved')
    from experiments.chronological import evaluate_min_event_horizon as evaluation
    command = ['--run-dir', str(args.run_dir), '--data-dir', str(args.data_dir),
               '--graph-dir', str(args.graph_dir), '--protocol', str(args.protocol),
               '--source-run', str(args.source_run), '--device', args.device, '--threads', str(args.threads)]
    if args.check:
        command.append('--check')
    return evaluation.main(command)


def main(argv=None):
    args = parse_args(argv)
    if (args.run_dir/'identity.json').is_file():
        require(legacy.read_json(args.run_dir/'identity.json').get('schema') == SCHEMA,
                'Use a fresh horizon directory; old experiment checkpoints cannot be resumed')
    if args.worker:
        require((args.run_dir/'identity.json').is_file(), 'Worker requires prepared parent identity')
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    require(device.type in ('cpu', 'cuda') and (device.type != 'cuda' or torch.cuda.is_available()), 'Requested device unavailable')
    logger = logging.getLogger('min_event_horizon')
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler(sys.stdout))
    args.run_dir.mkdir(parents=True, exist_ok=True)
    if args.worker:
        worker_root = args.run_dir/args.worker
        worker_root.mkdir(exist_ok=True)
        # A surviving child keeps its own lock even if the launcher is killed.
        with legacy.run_lock(worker_root), readonly_source_lock(args.source_run):
            before = source_snapshot(args)
            protocol, data, graph, initialization, identity, source = prepare(args, logger)
            verify_identity(args, identity, initialization, allow_new=False)
            verify_source_reference(args, identity)
            try:
                train_arm(args, args.worker, protocol, data, identity, graph, logger, source)
            finally:
                after = source_snapshot(args)
                legacy.write_json(worker_root/'source_preservation.json', dict(status='SOURCE_ARTIFACTS_PRESERVED' if before == after else 'SOURCE_ARTIFACTS_CHANGED',
                    before=before, after=after, source_artifacts_preserved=before == after))
                require(before == after, 'Source EVENT artifacts changed during worker')
        return
    with legacy.run_lock(args.run_dir):
        with readonly_source_lock(args.source_run):
            before = source_snapshot(args)
            protocol, data, graph, initialization, identity, source = prepare(args, logger)
            if not (args.run_dir/'identity.json').is_file():
                require({path.name for path in args.run_dir.iterdir()} <= {'active.lock'}, 'Use a fresh warmstart run directory')
                create_source_reference(args, data, graph, source)
                # The reference files are immutable run inputs, included in identity.
                identity = make_identity(args, protocol, identity['package'], identity['graph'], initialization, source)
                legacy.write_json(args.run_dir/'identity.json', identity)
                legacy.write_json(args.run_dir/'initialization.json', initialization)
        verify_identity(args, identity, initialization, allow_new=True)
        verify_source_reference(args, identity)
        legacy.write_json(args.run_dir/'data_audit.json', data.audit)
        session = args.run_dir/('session_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6])
        session.mkdir()
        legacy.write_json(session/'environment.json', legacy.environment(device))
        legacy.write_json(session/'request.json', dict(epochs=args.epochs, arms=args.arms, only=args.only,
                                                      source_run=str(args.source_run), evaluate_after=args.evaluate_after,
                                                      batch_size=identity['batch_size']))
        print('Data source: unchanged author release; 12 history / 12 target; clock offset 0', flush=True)
        print('Run directory: '+str(args.run_dir), flush=True)
        print('Session directory: '+str(session), flush=True)
        print('Initialization: '+initialization['status'], flush=True)
        # Release the full object-array package before sequential workers load it.
        del data, graph, source
        gc.collect()
        results = {}
        for arm in (args.only or args.arms):
            print('Starting '+arm+' ('+ARMS[arm]+')', flush=True)
            code = run_worker(args, arm, session)
            results[arm] = dict(exit_code=code)
            legacy.write_json(session/'process_results.json', results)
            require(code == 0, 'Worker failed; checkpoints preserved. See '+str(session/(arm+'.log')))
            require(source_snapshot(args) == before, 'Source EVENT artifacts changed while workers ran')
        report = comparison(args, identity)
        legacy.write_json(session/'comparison.json', report)
        legacy.write_json(args.run_dir/'latest_stage.json', report)
        legacy.write_json(session/'source_preservation.json', dict(status='SOURCE_ARTIFACTS_PRESERVED', before=before, after=source_snapshot(args)))
        print(json.dumps(dict(report, output=str(session/'comparison.json')), indent=2, allow_nan=False), flush=True)
    # Evaluation enters the same run lock. Never dispatch it from within the
    # training lock or the current process would block its own read-only audit.
    if args.evaluate_after:
        evaluate_after_training(args)


if __name__ == '__main__':
    main()
