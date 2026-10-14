"""From-scratch joint MIN training with one early state/event conditioning site.

Separate workers own each arm. Training windows, normalization, curriculum and
metrics are inherited unchanged; event attributes have no certified as-of time.
No existing MIN or adapter checkpoint is loaded to initialize these arms.
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
from experiments.chronological import train_b3_r_minimal_paper as paper
from experiments.chronological import train_internal_spatial as legacy
from experiments.chronological.smoke import sha256
from experiments.chronological.train import cpu_state, save_checkpoint
from src.engines.igstgnn_engine import IGSTGNN_Engine
from src.models.incident_spatial import STAT_NAMES, require
from src.models.incident_spatial_residual import tensor_state_hash
from src.models.incident_spatial_early_event import from_minimal
from src.utils.igstgnn_published_events import EventPublishedData
from src.utils.metrics import compute_all_metrics, masked_mae

HERE = Path(__file__).resolve().parent
PROTOCOL = HERE/'min_early_event_v1.json'
SCHEMA = 'min_early_event_published_run_v1'
ARMS = {'STATE': 'state', 'LOCATION': 'location', 'EVENT': 'event'}
WORKER_ENV = 'MIN_EARLY_EVENT_WORKER'


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=paper.default_data())
    parser.add_argument('--graph-dir', type=Path, default=HERE/'internal_spatial_graph_is11')
    parser.add_argument('--protocol', type=Path, default=PROTOCOL)
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--arms', nargs='+', choices=ARMS, default=['STATE', 'EVENT'])
    parser.add_argument('--only', nargs='+', choices=ARMS, help='Run only these members of the saved arm set')
    parser.add_argument('--epochs', type=int, help='Absolute target epoch: default 5, or 1 with --check')
    parser.add_argument('--batch-size', type=int, help='Default 64; fixed 2 for --check')
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--threads', type=int, default=3)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--evaluate-test', action='store_true', help='Explicit best-checkpoint test evaluation')
    parser.add_argument('--evaluate-after', action='store_true',
                        help='After training, replay saved STATE/EVENT best validation and audit modules in this job')
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
    require(not args.check or (args.epochs == 1 and not args.evaluate_test),
            'Engineering check is one small epoch without test evaluation')
    require(not args.worker or (os.environ.get(WORKER_ENV) == '1' and args.run_dir is not None
            and args.worker in args.arms), 'Internal worker requires parent launcher; use --only')
    require(not args.worker or not args.evaluate_after, 'Workers cannot recursively evaluate-after')
    if args.run_dir is None:
        args.run_dir = REPO/'experiments/chronological_runs'/('contra_min_early_'+('check_' if args.check else '')
                       +datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6])
    for name in ('data_dir', 'graph_dir', 'protocol', 'run_dir'):
        setattr(args, name, getattr(args, name).resolve())
    return args


def load_protocol(args):
    protocol = legacy.read_json(args.protocol)
    require(protocol == legacy.read_json(PROTOCOL), 'MIN early-event protocol changed; use a separate version')
    return dict(protocol, batch_size=args.batch_size) if args.batch_size is not None else protocol


def build_arm(args, arm, nodes, graph, device):
    require(arm in ARMS, 'Unknown MIN early-event arm')
    model, min_certificate = paper.build_model(args, nodes, graph, device)
    # All modes register the exact same parameter set. Preserve starting data
    # order and dropout streams even when initialization draws extra tensors.
    numpy_rng, torch_rng = np.random.get_state(), torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.device(device).type == 'cuda' else None
    specification = load_protocol(args)['condition_spec']
    model, certificate = from_minimal(model, mode=ARMS[arm], width=specification['width'],
                                      max_scale=specification['max_scale'])
    np.random.set_state(numpy_rng)
    torch.set_rng_state(torch_rng)
    if cuda_rng is not None:
        torch.cuda.set_rng_state_all(cuda_rng)
    certificate = dict(certificate, MIN_initialization=min_certificate,
                       training_initialization='from_scratch_joint', loaded_source_checkpoint=False)
    return model, certificate


def event_tensors(payload, device):
    long_names = {'type', 'description'}
    return {key: torch.as_tensor(value, device=device,
            dtype=torch.long if key in long_names else torch.float32 if key == 'relation' else torch.bool)
            for key, value in payload.items()}


def preflight(args, data, graph):
    reference, _ = paper.build_model(args, len(data.station_ids), graph, torch.device('cpu'))
    reference.eval()
    batch = next(data.loaders['train_loader'].get_iterator())
    x = torch.as_tensor(batch['x_data'][:2], dtype=torch.float32)
    events = event_tensors({k: v[:2] for k, v in batch['event_data'].items()}, 'cpu')
    base_state = reference.state_dict()
    with torch.inference_mode():
        expected = reference(x)
    manifests, shared_state = {}, None
    for arm in args.arms:
        model, certificate = build_arm(args, arm, len(data.station_ids), graph, torch.device('cpu'))
        state = model.state_dict()
        require(base_state.keys() <= state.keys() and all(torch.equal(v, state[k]) for k, v in base_state.items()),
                'Common MIN initialization changed: '+arm)
        model.eval()
        with torch.inference_mode():
            actual = model(x, incident_data=events)
            off = model(x, incident_data=events, report_enabled=False)
        require(torch.equal(expected, actual) and torch.equal(expected, off),
                'Zero initialized/OFF model differs from MIN: '+arm)
        added = {k: v for k, v in state.items() if k not in base_state}
        require(added, 'Early conditioning parameters were not registered')
        require(all(parameter.requires_grad for parameter in model.parameters()),
                'All backbone and conditioning parameters must train jointly')
        if shared_state is not None:
            require(state.keys() == shared_state.keys()
                    and all(torch.equal(state[k], shared_state[k]) for k in state),
                    'STATE/LOCATION/EVENT parameter layout or initialization differs')
        shared_state = state
        manifests[arm] = dict(parameters=sum(p.numel() for p in model.parameters()),
                              common_state_sha256=tensor_state_hash(base_state),
                              initial_state_sha256=tensor_state_hash(state),
                              added_parameters=sum(v.numel() for v in added.values()), conversion=certificate)
        del model
    return dict(status='MIN_EARLY_INITIALIZATION_PAIR_PASS', arms=manifests,
                zero_initialized_prediction_abs_max=0., report_off_prediction_abs_max=0.,
                common_parameters_exact=True, same_parameter_layout_and_initialization=True,
                training_initialization='from_scratch_joint', loaded_source_checkpoint=False,
                event_scope='published_archive_attributes_conditional_offline')


def source_fingerprints():
    names = ('experiments/chronological/train_min_early_event.py', 'src/utils/igstgnn_published_events.py',
             'src/models/incident_spatial_early_event.py', 'src/models/incident_spatial_event_budget.py',
             'experiments/chronological/train.py',
             'experiments/chronological/smoke.py')
    return dict(paper.source_fingerprints(), **{name: sha256(REPO/name) for name in names})


def make_identity(args, protocol, hashes, graph_info, initialization):
    return dict(schema=SCHEMA, arm_modes={arm: ARMS[arm] for arm in args.arms}, protocol=protocol,
                protocol_sha256=sha256(args.protocol), package=hashes, graph=graph_info,
                graph_inputs_sha256={name: sha256(args.graph_dir/name)
                                     for name in ('graph.json', 'stations.csv', 'source_summary.json')},
                source_sha256=source_fingerprints(), seed=args.seed, engineering_check=args.check,
                batch_size=2 if args.check else protocol['batch_size'], threads=args.threads,
                initialization=initialization, clock_policy='last_published_history_slot_no_extra_gap',
                training_initialization='from_scratch_joint', loaded_source_checkpoint=False,
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
                'Use a fresh MIN early-event directory; old experiments are preserved')
        legacy.write_json(path, identity)
        legacy.write_json(args.run_dir/'initialization.json', initialization)


class EarlyEventEngine(IGSTGNN_Engine):
    def __init__(self, arm, **kwargs):
        self.arm, self.collect_diagnostics = arm, False
        self.diagnostics = {}
        super().__init__(**kwargs)

    def _prepare_batch(self, batch):
        if not isinstance(batch, dict):
            return super()._prepare_batch(batch)
        events = event_tensors(batch['event_data'], self._device) if self.arm != 'STATE' else None
        return self._as_float_tensor(batch['x_data']), self._as_float_tensor(batch['y_data']), events, None

    def _predict(self, X, label, incident_data, sensor_data):
        if not self.collect_diagnostics:
            return self.model(X, label, incident_data=incident_data)
        result = self.model(X, label, incident_data=incident_data, return_spatial_details=True)
        self._record(result['spatial'])
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
    optimizer = legacy.optimizer_for(model, protocol)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, protocol['lr_milestones'], gamma=protocol['lr_gamma'])
    batches = data.loaders['train_loader'].num_batch
    return EarlyEventEngine(arm=arm, device=torch.device(args.device), model=model,
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
                event_inputs_used=saved['arm'] != 'STATE', semantic_inputs_used=saved['arm'] == 'EVENT',
                training_initialization='from_scratch_joint', loaded_source_checkpoint=False,
                trainable_scope='all_MIN_backbone_and_single_early_conditioner',
                conditioning_site='latest_normalized_history_slot_before_decoupled_layers',
                conditioning_injections_per_forward=1,
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
    require(saved['model_state'].keys() == saved['best_model_state'].keys(), 'Checkpoint state keys differ')
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


def evaluate_test(engine, data, saved, root, logger):
    engine.model.load_state_dict(saved['best_model_state'], strict=True)
    engine.model.eval()
    loader = data.test_loader(engine._dataloader['val_loader'].bs, logger)
    predictions, labels = [], []
    with torch.inference_mode():
        for batch in loader.get_iterator():
            x, y, event, sensor = engine._prepare_batch(batch)
            pred, target = engine._inverse_transform([engine._predict(x, y, event, sensor), y])
            predictions.append(pred.cpu())
            labels.append(target.cpu())
    pred, target = torch.cat(predictions), torch.cat(labels)
    null = engine._mask_value(target)
    metrics = [compute_all_metrics(pred[:, h], target[:, h], null) for h in range(engine._horizon)]
    report = dict(status='MIN_EARLY_PUBLISHED_TEST_EVALUATION_COMPLETE', arm=saved['arm'],
                  best_epoch=saved['history'][-1]['best_epoch'], samples=loader.size, mask_value=float(null),
                  by_horizon=[dict(horizon=h+1, mae=m[0], mape=m[1], rmse=m[2]) for h, m in enumerate(metrics)],
                  average=dict(zip(('mae', 'mape', 'rmse'), np.mean(metrics, axis=0).tolist())))
    legacy.write_json(root/'test_metrics.json', report)
    print(json.dumps(report, ensure_ascii=False, allow_nan=False), flush=True)
    return report


def train_arm(args, arm, protocol, data, identity, graph, logger):
    root = args.run_dir/arm
    root.mkdir(exist_ok=True)
    model, _ = build_arm(args, arm, len(data.station_ids), graph, torch.device(args.device))
    engine = make_engine(args, arm, protocol, data, model, logger)
    path = root/'last.pt'
    history, best, wait, saved, best_state, prior_summary = [], float('inf'), 0, None, None, None
    if path.is_file():
        saved = torch.load(path, map_location='cpu', weights_only=False)
        validate_saved(saved, identity, arm)
        prior_summary = repair_sidecars(root, saved, args, protocol)
        restore(saved, identity, arm, engine, data)
        history, best, wait, best_state = saved['history'], saved['best_metric'], saved['wait'], saved['best_model_state']
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
            best, wait, best_state = val_mae, 0, cpu_state(model.state_dict())
        else:
            wait += 1
        lr = engine._optimizer.param_groups[0]['lr']
        engine._lr_scheduler.step()
        row = dict(epoch=epoch, train_mae=float(train_mae), train_rmse=float(train_rmse), train_mape=float(train_mape),
                   validation_mae=float(val_mae), validation_rmse=float(val_rmse), validation_mape=float(val_mape),
                   best_metric=best, best_epoch=epoch if improved else history[-1]['best_epoch'],
                   global_updates=engine._iter_cnt, curriculum_length=engine._cl_len,
                   learning_rate_used=lr, next_learning_rate=engine._optimizer.param_groups[0]['lr'],
                   train_order_sha256=hashlib.sha256(data.loaders['train_loader'].order.tobytes()).hexdigest(),
                   validation_diagnostics=engine.diagnostics, seconds=time.perf_counter()-start)
        history.append(row)
        saved = dict(identity=identity, arm=arm, completed_epoch=epoch, history=history, parameters=parameters,
                     model_state=cpu_state(model.state_dict()), best_model_state=best_state, best_metric=best, wait=wait,
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
    if args.evaluate_test:
        result.update(test_evaluated=True, test_metrics=evaluate_test(engine, data, saved, root, logger))
    legacy.write_json(root/'summary.json', result)
    print(json.dumps(dict(arm=arm, status=result['status'], completed_epoch=result['completed_epoch'],
                         best_epoch=result['best_epoch'], best_metric=result['best_metric'],
                         optimizer_updates_this_invocation=result['optimizer_updates_this_invocation'],
                         output=str(root/'summary.json')), allow_nan=False), flush=True)
    return result


def comparison(args, identity):
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
        for control, candidate in [('STATE', 'LOCATION'), ('LOCATION', 'EVENT'), ('STATE', 'EVENT')]:
            if control not in histories or candidate not in histories:
                continue
            c, r = histories[control][:common], histories[candidate][:common]
            comparisons[candidate+'_vs_'+control] = dict(
                common_epoch_mae_gain=c[-1]['validation_mae']-r[-1]['validation_mae'],
                common_prefix_best_mae_gain=min(x['validation_mae'] for x in c)-min(x['validation_mae'] for x in r),
                candidate_common_last_five=[x['validation_mae'] for x in r[-5:]],
                control_common_last_five=[x['validation_mae'] for x in c[-5:]])
        require(all(len({histories[a][e]['train_order_sha256'] for a in histories}) == 1 for e in range(common)),
                'Paired training sample order differs')
    complete = len(summaries) == len(args.arms) and all(
        s['completed_epoch'] >= args.epochs or s['early_stopped'] for s in summaries.values())
    return dict(status='MIN_EARLY_PAIRED_STAGE_COMPLETE' if complete else 'MIN_EARLY_PAIRED_STAGE_PARTIAL',
                common_epoch=common, runs=summaries, comparisons=comparisons,
                interpretation={'LOCATION_vs_STATE': 'location_support_and_relation_conditioning_increment_same_registered_parameters',
                                'EVENT_vs_LOCATION': 'semantic_information_increment_same_architecture_and_parameters',
                                'EVENT_vs_STATE': 'complete_event_support_relation_semantics_conditioning_increment_same_registered_parameters'},
                active_parameter_caveat='same_registered_size_does_not_imply_equal_active_semantic_parameters_or_support_extent',
                training_initialization='from_scratch_joint', old_MIN_comparison='unpaired_historical_reference_only',
                predictive_gain_claim=False)


def prepare(args, logger):
    protocol = load_protocol(args)
    hashes = paper.verify_release(args.data_dir, protocol)
    data = EventPublishedData(args.data_dir, protocol, logger, check=args.check)
    graph, _, graph_info = legacy.load_graph(args.graph_dir, data.station_ids, protocol['graph_sha256'])
    initialization = preflight(args, data, graph)
    identity = make_identity(args, protocol, hashes, graph_info, initialization)
    return protocol, data, graph, initialization, identity


def child_command(args, arm):
    command = [sys.executable, '-u', str(Path(__file__).resolve()), '--run-dir', str(args.run_dir),
               '--data-dir', str(args.data_dir), '--graph-dir', str(args.graph_dir), '--protocol', str(args.protocol),
               '--epochs', str(args.epochs), '--seed', str(args.seed), '--device', args.device,
               '--threads', str(args.threads), '--arms', *args.arms, '--worker', arm]
    if args.batch_size is not None:
        command += ['--batch-size', str(args.batch_size)]
    if args.check:
        command.append('--check')
    if args.evaluate_test:
        command.append('--evaluate-test')
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
    require(all((args.run_dir/arm/'last.pt').is_file() for arm in ('STATE', 'EVENT')),
            'Automatic evaluation requires saved STATE and EVENT checkpoints; training results are preserved')
    from experiments.chronological import evaluate_min_early_event as evaluation
    command = ['--run-dir', str(args.run_dir), '--data-dir', str(args.data_dir),
               '--graph-dir', str(args.graph_dir), '--protocol', str(args.protocol),
               '--device', args.device, '--threads', str(args.threads)]
    if args.check:
        command.append('--check')
    return evaluation.main(command)


def main(argv=None):
    args = parse_args(argv)
    if (args.run_dir/'identity.json').is_file():
        require(legacy.read_json(args.run_dir/'identity.json').get('schema') == SCHEMA,
                'Use a fresh MIN early-event directory; old experiment checkpoints cannot be resumed')
    if args.worker:
        require((args.run_dir/'identity.json').is_file(), 'Worker requires prepared parent identity')
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    require(device.type in ('cpu', 'cuda') and (device.type != 'cuda' or torch.cuda.is_available()), 'Requested device unavailable')
    logger = logging.getLogger('min_early_event')
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler(sys.stdout))
    args.run_dir.mkdir(parents=True, exist_ok=True)
    if args.worker:
        worker_root = args.run_dir/args.worker
        worker_root.mkdir(exist_ok=True)
        # A surviving child keeps its own lock even if the launcher is killed.
        with legacy.run_lock(worker_root):
            protocol, data, graph, initialization, identity = prepare(args, logger)
            verify_identity(args, identity, initialization, allow_new=False)
            train_arm(args, args.worker, protocol, data, identity, graph, logger)
        return
    with legacy.run_lock(args.run_dir):
        protocol, data, graph, initialization, identity = prepare(args, logger)
        verify_identity(args, identity, initialization, allow_new=True)
        legacy.write_json(args.run_dir/'data_audit.json', data.audit)
        session = args.run_dir/('session_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6])
        session.mkdir()
        legacy.write_json(session/'environment.json', legacy.environment(device))
        legacy.write_json(session/'request.json', dict(epochs=args.epochs, arms=args.arms, only=args.only,
                                                      evaluate_test=args.evaluate_test, evaluate_after=args.evaluate_after,
                                                      batch_size=identity['batch_size']))
        print('Data source: unchanged author release; 12 history / 12 target; clock offset 0', flush=True)
        print('Run directory: '+str(args.run_dir), flush=True)
        print('Session directory: '+str(session), flush=True)
        print('Initialization: '+initialization['status'], flush=True)
        # Release the full object-array package before sequential workers load it.
        del data, graph
        gc.collect()
        results = {}
        for arm in (args.only or args.arms):
            print('Starting '+arm+' ('+ARMS[arm]+')', flush=True)
            code = run_worker(args, arm, session)
            results[arm] = dict(exit_code=code)
            legacy.write_json(session/'process_results.json', results)
            require(code == 0, 'Worker failed; checkpoints preserved. See '+str(session/(arm+'.log')))
        report = comparison(args, identity)
        legacy.write_json(session/'comparison.json', report)
        legacy.write_json(args.run_dir/'latest_stage.json', report)
        print(json.dumps(dict(report, output=str(session/'comparison.json')), indent=2, allow_nan=False), flush=True)
    # Evaluation enters the same run lock. Never dispatch it from within the
    # training lock or the current process would block its own read-only audit.
    if args.evaluate_after:
        evaluate_after_training(args)


if __name__ == '__main__':
    main()
