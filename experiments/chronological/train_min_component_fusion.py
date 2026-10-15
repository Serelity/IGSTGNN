"""F2: paired CONTROL/STATE/EVENT adaptation from the existing MIN A best.

CONTROL is a pure MIN additional fine-tuning control, not a new original-MIN
training run. Cached certified F1 source predictions are reused when available.
"""
import argparse
from datetime import datetime
import gc
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
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
from src.models.incident_spatial import require
from src.models.incident_spatial_residual import tensor_state_hash
from src.models.incident_spatial_component_fusion import from_minimal
from src.utils.igstgnn_published_events import EventPublishedData
from src.utils.metrics import compute_all_metrics, masked_mae

paper, legacy, source_training = shared.paper, shared.legacy, shared.source_training
HERE = Path(__file__).resolve().parent
PROTOCOL = HERE/'min_component_fusion_v1.json'
SCHEMA = 'min_component_content_fusion_warmstart_run_v1'
ARMS = {'CONTROL': 'minimal', 'STATE': 'state', 'EVENT': 'event'}
WORKER_ENV = 'MIN_COMPONENT_FUSION_WORKER'
DEFAULT_SOURCE = shared.DEFAULT_SOURCE
DEFAULT_REFERENCE = REPO/'experiments/chronological_runs/contra_min_forecast_fusion_f1_bs64_seed2025'
SOURCE_FILES = shared.SOURCE_FILES
source_snapshot = shared.source_snapshot
readonly_source_lock = shared.readonly_source_lock
load_source = shared.load_source
build_core = shared.build_core
event_tensors = shared.event_tensors
validation_predictions = shared.validation_predictions
verify_source_reference = shared.verify_source_reference
load_source_reference = shared.load_source_reference


def write_json(path, value):
    """Commit one JSON sidecar atomically; checkpoints remain authoritative."""
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
    p = argparse.ArgumentParser(description=__doc__)
    for name, default in (('data-dir', paper.default_data()), ('graph-dir', HERE/'internal_spatial_graph_is11'),
                          ('protocol', PROTOCOL), ('source-run', DEFAULT_SOURCE), ('reference-run', DEFAULT_REFERENCE)):
        p.add_argument('--'+name, type=Path, default=default)
    p.add_argument('--run-dir', type=Path)
    p.add_argument('--arms', nargs='+', choices=ARMS, default=list(ARMS))
    p.add_argument('--only', nargs='+', choices=ARMS, help='Continue only these members of the immutable saved arm set')
    p.add_argument('--epochs', type=int, help='Absolute stage target; default 5, or 1 with --check')
    p.add_argument('--batch-size', type=int)
    p.add_argument('--seed', type=int, default=2025)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--threads', type=int, default=3)
    p.add_argument('--check', action='store_true')
    p.add_argument('--evaluate-after', action='store_true')
    p.add_argument('--worker', choices=ARMS, help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    args.epochs = (1 if args.check else 5) if args.epochs is None else args.epochs
    require(1 <= args.epochs <= 100 and args.threads >= 1 and 0 <= args.seed < 2**32, 'Invalid epoch/threads/seed')
    require(args.batch_size is None or args.batch_size > 0, 'Batch size must be positive')
    require(len(set(args.arms)) == len(args.arms), 'Duplicate arms')
    args.arms = [arm for arm in ARMS if arm in args.arms]
    require(args.only is None or (len(set(args.only)) == len(args.only) and set(args.only) <= set(args.arms)),
            '--only must be a unique subset of --arms')
    require(not args.check or args.epochs == 1, 'Engineering check requires exactly one small epoch')
    require(not args.worker or (os.environ.get(WORKER_ENV) == '1' and args.run_dir is not None
            and args.worker in args.arms), 'Internal worker requires parent launcher; use --only')
    require(not args.worker or not args.evaluate_after, 'Workers cannot recursively evaluate-after')
    if args.run_dir is None:
        args.run_dir = REPO/'experiments/chronological_runs'/('contra_min_component_fusion_'+('check_' if args.check else '')
            +datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6])
    for name in ('data_dir', 'graph_dir', 'protocol', 'source_run', 'reference_run', 'run_dir'):
        setattr(args, name, getattr(args, name).resolve())
    for source in (args.source_run, args.reference_run):
        require(args.run_dir != source and source not in args.run_dir.parents and args.run_dir not in source.parents,
                'New run must be separate from source/reference runs and their ancestors')
    return args


def load_protocol(args):
    protocol = legacy.read_json(args.protocol)
    require(protocol == legacy.read_json(PROTOCOL), 'Component-fusion protocol changed; use a separate version')
    return dict(protocol, batch_size=args.batch_size) if args.batch_size is not None else protocol


def build_arm(args, arm, nodes, graph, device, source=None):
    require(arm in ARMS, 'Unknown component-fusion arm')
    source = load_source(args) if source is None else source
    core = build_core(args, nodes, graph, device, source)
    set_seed(args.seed)
    if arm == 'CONTROL':
        model, certificate = core, {'architecture_version': 'MIN', 'added_parameters': 0}
    else:
        spec = load_protocol(args)['fusion_spec']
        model, certificate = from_minimal(core, mode=ARMS[arm], width=spec['width'],
            component_dim=spec['component_dim'], context_dim=spec['context_dim'], max_delta=spec['max_delta'],
            type_count=spec['type_classes'], description_count=spec['description_classes'])
    model.requires_grad_(True)
    set_seed(args.seed)
    return model, dict(certificate, mode=ARMS[arm], source_best_epoch=source['selected']['epoch'],
        source_best_model_sha256=source['best_model_sha256'], shared_core_warmstart=True,
        fresh_optimizer_and_scheduler=True)


def preflight(args, data, graph, source):
    """Only two training examples: never repeats a full source validation pass."""
    reference = build_core(args, len(data.station_ids), graph, torch.device('cpu'), source).eval()
    batch = next(data.loaders['train_loader'].get_iterator())
    x = torch.as_tensor(batch['x_data'][:2], dtype=torch.float32)
    events = event_tensors({k: v[:2] for k, v in batch['event_data'].items()}, 'cpu')
    base = reference.state_dict()
    with torch.inference_mode():
        expected = reference(x)
    manifests, fused_state = {}, None
    for arm in args.arms:
        model, conversion = build_arm(args, arm, len(data.station_ids), graph, 'cpu', source)
        model.eval()
        state = model.state_dict()
        require(base.keys() <= state.keys() and all(torch.equal(v, state[k]) for k, v in base.items()),
                'Source common tensors changed: '+arm)
        require((state.keys() == base.keys()) if arm == 'CONTROL' else (base.keys() < state.keys()),
                'CONTROL must be pure MIN; fused groups must add their single router')
        with torch.inference_mode():
            actual = model(x, incident_data=events if arm == 'EVENT' else None)
        require(torch.equal(expected, actual), 'Stage-zero prediction differs from source MIN: '+arm)
        require(all(p.requires_grad for p in model.parameters()), 'All backbone/router parameters must train jointly')
        if arm != 'CONTROL':
            if fused_state is not None:
                require(state.keys() == fused_state.keys() and all(torch.equal(state[k], fused_state[k]) for k in state),
                        'STATE/EVENT initialization differs')
            fused_state = cpu_state(state)
        manifests[arm] = dict(parameters=sum(p.numel() for p in model.parameters()),
            common_state_sha256=tensor_state_hash(base), initial_state_sha256=tensor_state_hash(state),
            added_parameters=sum(p.numel() for n, p in model.named_parameters() if n.startswith('forecast_fusion.')),
            conversion=conversion)
        del model
    return dict(status='MIN_COMPONENT_FUSION_INITIALIZATION_PAIR_PASS', arms=manifests,
        common_parameters_exact=True, same_fused_parameter_layout_and_initialization=True,
        zero_fusion_source_prediction_abs_max=0., initialization_examples=2,
        initialization_source_forward_calls=1, initialization_arm_forward_calls=len(args.arms),
        initialization_split='train', full_source_validation_calls=0,
        training_initialization='shared_source_MIN_A_best_warmstart_fresh_paired_optimizer')


def source_fingerprints():
    names = ('experiments/chronological/train_min_component_fusion.py',
             'src/models/incident_spatial_component_fusion.py')
    return dict(shared.source_fingerprints(), **{name: sha256(REPO/name) for name in names})


def make_identity(args, protocol, hashes, graph_info, initialization, source):
    setup = argparse.Namespace(**vars(args))
    setup.arms = [arm for arm in args.arms if arm in shared.ARMS]
    identity = shared.make_identity(setup, protocol, hashes, graph_info, initialization, source)
    identity.update(schema=SCHEMA, arm_modes={arm: ARMS[arm] for arm in args.arms}, source_sha256=source_fingerprints(),
        architecture_version='MIN-COMPONENT-FUSION2.0', same_registered_parameter_layout=False,
        same_fused_registered_parameter_layout=True, CONTROL_scope='pure_MIN_additional_finetuning_same_stage_budget',
        existing_MIN_retrained_from_scratch=False, primary_baseline='original_IGSTGNN_not_in_this_training_stage',
        content_context='ten_native_forecast_components_available_to_router',
        weight_rule='one_plus_bounded_zero_sum_component_relative_increment')
    count = min(protocol['train_samples'], 4) if args.check else protocol['train_samples']
    identity['train_batches'] = (count+identity['batch_size']-1)//identity['batch_size']
    return identity


def verify_identity(args, identity, initialization, allow_new):
    path = args.run_dir/'identity.json'
    if path.is_file():
        require(legacy.read_json(path) == identity, 'Run identity changed; existing checkpoints preserved')
        require(legacy.read_json(args.run_dir/'initialization.json') == initialization, 'Initialization certificate changed')
    else:
        require(allow_new and {p.name for p in args.run_dir.iterdir()} <=
                {'active.lock', 'source_reference.json', 'source_reference.npz'}, 'Use a fresh F2 run directory')
        write_json(path, identity)
        write_json(args.run_dir/'initialization.json', initialization)


def _validation_axes(data):
    targets, partitions = [], []
    for batch in data.loaders['val_loader'].get_iterator():
        value = data.scaler.inverse_transform(torch.as_tensor(batch['y_data'], dtype=torch.float32)).squeeze(-1)
        targets.append(value.numpy())
        partitions.append(len(value))
    target = np.concatenate(targets)
    mask_value = float(target.min()) if target.min() < 1 else 0.
    return dict(target=target, valid=target != mask_value,
        sample_indices=data.audit['counts']['train_samples']+np.arange(len(target), dtype=np.int64),
        station_ids=np.asarray(data.station_ids, np.int64)), partitions, mask_value


def create_source_reference(args, data, graph, source):
    """Reuse a certified F1 archive; only an absent archive permits one replay."""
    names = ('identity.json', 'source_reference.json', 'source_reference.npz')
    present = [(args.reference_run/name).is_file() for name in names]
    if not any(present):
        print('No existing certified source archive found: one full validation inference, no source training', flush=True)
        certificate = shared.create_source_reference(args, data, graph, source)
        certificate.update(reference_origin='new_single_source_validation_replay', source_full_validation_calls=1,
                           reused_archive=False, origin_run=None)
        write_json(args.run_dir/'source_reference.json', certificate)
        return certificate
    require(all(present), 'Incomplete reference-run archive; refusing silent replacement inference')
    before = {name: sha256(args.reference_run/name) for name in names}
    previous = legacy.read_json(args.reference_run/'identity.json')
    require(previous.get('schema') == shared.SCHEMA, 'Reference must be a certified previous F1 run')
    require(previous.get('protocol_sha256') == sha256(shared.PROTOCOL), 'Previous F1 protocol fingerprint changed')
    require(previous.get('source_sha256') == shared.source_fingerprints(), 'Previous F1 source code fingerprints changed')
    require(previous.get('package') == source['identity']['package'], 'Reference release fingerprints changed')
    require(previous.get('runtime') == dict(torch=str(torch.__version__), cuda=torch.version.cuda,
            device_type=torch.device(args.device).type, cublas_workspace_config=os.environ['CUBLAS_WORKSPACE_CONFIG']),
            'Reference runtime changed; GPU model name is not an identity field')
    require(previous.get('batch_size') == data.loaders['val_loader'].bs and previous.get('engineering_check') is args.check,
            'Reference original validation batch/check scope differs')
    require(previous.get('source', {}).get('files_sha256') == source['files_sha256']
            and previous['source']['best_model_sha256'] == source['best_model_sha256']
            and previous['source']['best_epoch'] == source['selected']['epoch'], 'Reference selected source differs')
    reference_args = argparse.Namespace(**vars(args))
    reference_args.run_dir = args.reference_run
    certificate, arrays = shared.load_source_reference(reference_args, previous)
    axes, partitions, mask_value = _validation_axes(data)
    require(all(np.array_equal(arrays[name], value) for name, value in axes.items()),
            'Reference validation targets, validity mask, sample or station axes differ')
    require(certificate['batch_partition'] == partitions and certificate['mask_value'] == mask_value,
            'Reference original batch partition or mask convention differs')
    actual = compute_all_metrics(torch.from_numpy(arrays['prediction']), torch.from_numpy(arrays['target']), torch.tensor(mask_value))
    require(np.allclose(actual, certificate['actual'], atol=1e-5, rtol=1e-6), 'Reference archived metrics differ')
    require(before == {name: sha256(args.reference_run/name) for name in names}, 'Reference artifacts changed while reading')
    destination = args.run_dir/'source_reference.npz'
    temporary = destination.with_suffix('.npz.partial')
    shutil.copyfile(args.reference_run/'source_reference.npz', temporary)
    require(sha256(temporary) == before['source_reference.npz'], 'Copied reference archive changed')
    temporary.replace(destination)
    result = dict(certificate, archive_sha256=sha256(destination), reused_archive=True,
        reference_origin='certified_previous_F1_source_reference_reuse_not_new_replay',
        source_full_validation_calls=0, origin_run=str(args.reference_run), origin_files_sha256=before,
        reuse_checks=['source_selected_weights', 'source_files', 'release_hashes', 'torch_cuda_device_type',
                     'F1_code_fingerprints', 'validation_targets', 'mask', 'sample_axis', 'station_axis', 'original_batches'])
    write_json(args.run_dir/'source_reference.json', result)
    print('Certified existing SOURCE_MIN archive reused: 0 new source validation forwards', flush=True)
    return result


class ComponentEngine(shared.FusionEngine):
    def _predict(self, X, label, incident_data, sensor_data):
        if not self.collect_diagnostics:
            return self.model(X, label, incident_data=incident_data)
        result = self.model(X, label, incident_data=incident_data, return_spatial_details=True)
        self._record(result['spatial'])
        for key, value in result.get('fusion', {}).get('statistics', {}).items():
            value = float(value.detach().cpu()) if torch.is_tensor(value) else float(value)
            require(np.isfinite(value), 'Non-finite component-fusion statistic: '+key)
            if key.endswith('_max'):
                self.diagnostics[key] = max(self.diagnostics.get(key, 0.), value)
            elif key.endswith('_min'):
                self.diagnostics[key] = min(self.diagnostics.get(key, value), value)
            else:
                self.diagnostics[key] = self.diagnostics.get(key, 0.)+value
        return result['prediction']


def make_engine(args, arm, protocol, data, model, logger):
    core = [p for n, p in model.named_parameters() if not n.startswith('forecast_fusion.')]
    fusion = [p for n, p in model.named_parameters() if n.startswith('forecast_fusion.')]
    require(core and all(p.requires_grad for p in core+fusion), 'Complete backbone and router must train jointly')
    require(bool(fusion) == (arm != 'CONTROL'), 'CONTROL must have no fusion parameters')
    groups = [dict(params=core, lr=protocol['learning_rate'], weight_decay=protocol['weight_decay'])]
    if fusion:
        groups.append(dict(params=fusion, lr=protocol['fusion_learning_rate'], weight_decay=protocol['weight_decay']))
    optimizer = torch.optim.Adam(groups, eps=protocol['adam_eps'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, protocol['lr_milestones'], gamma=protocol['lr_gamma'])
    return ComponentEngine(arm=arm, device=torch.device(args.device), model=model, dataloader=data.loaders,
        scaler=data.scaler, sampler=None, loss_fn=masked_mae, lrate=protocol['learning_rate'], optimizer=optimizer,
        scheduler=scheduler, clip_grad_value=protocol['clip_grad_norm'], max_epochs=protocol['max_epochs'],
        patience=protocol['patience'], log_dir=str(args.run_dir/arm), logger=logger, seed=args.seed,
        cl_step=0, warm_step=0, horizon=12, incident=False)


def summary_for(saved, args, protocol):
    stopped = saved['wait'] >= protocol['patience']
    return dict(status='ENGINEERING_CHECK_PASS' if args.check else 'EARLY_STOPPED' if stopped else 'PAUSED_AT_EPOCH_BOUNDARY',
        arm=saved['arm'], mode=ARMS[saved['arm']], completed_epoch=saved['completed_epoch'], requested_epoch=args.epochs,
        best_epoch=saved['best_epoch'], best_metric=saved['best_metric'], wait=saved['wait'], early_stopped=stopped,
        global_updates=saved['global_updates'], curriculum_length=saved['curriculum_length'], parameters=saved['parameters'],
        batch_size=2 if args.check else protocol['batch_size'], event_inputs_used=saved['arm'] == 'EVENT',
        semantic_inputs_used=saved['arm'] == 'EVENT', source_best_epoch=saved['identity']['source']['best_epoch'],
        best_stage_epoch=saved['best_epoch'], best_is_source_epoch_zero=saved['best_epoch'] == 0,
        stage_epoch_zero_eligible_for_selection=True, training_initialization='shared_source_MIN_A_best_warmstart_fresh_paired_optimizer',
        source_exact_resume=False, source_optimizer_restored=False, source_scheduler_restored=False,
        existing_MIN_retrained_from_scratch=False, all_backbone_parameters_trainable=True,
        trainable_scope='pure_MIN_additional_finetuning' if saved['arm'] == 'CONTROL' else 'all_MIN_backbone_and_component_content_fusion',
        full_horizon_loss_every_update=True, core_forwards_per_prediction=1, activation_checkpointing=False,
        graph_and_capacity_inputs='original_unconditioned_MIN_history',
        event_scope='published_archive_attributes_conditional_offline', capacity_scope='dimensionless_latent_exchange_budget',
        primary_baseline='original_IGSTGNN_not_in_this_training_stage', predictive_gain_claim=False,
        test_evaluated=False, checkpoint='last.pt', best_model_state_key='best_model_state')


def validate_saved(saved, identity, arm):
    require(saved.get('identity') == identity and saved.get('arm') == arm, 'Checkpoint identity/arm mismatch')
    history = saved.get('history')
    require(isinstance(history, list) and history and len(history) == saved.get('completed_epoch')
            and [r['epoch'] for r in history] == list(range(1, len(history)+1)), 'Invalid checkpoint history')
    require(saved['best_metric'] == history[-1]['best_metric'] and saved['global_updates'] == history[-1]['global_updates']
            and saved['best_epoch'] == history[-1]['best_epoch'], 'Checkpoint counters disagree')
    require(saved.get('curriculum_length') == 12 and all(r.get('curriculum_length') == 12 for r in history),
            'F2 always trains all twelve forecast horizons')
    eligible = [identity['source']['reference_metrics'][0]]+[r['validation_mae'] for r in history]
    require(np.isfinite(eligible).all() and saved['best_metric'] == min(eligible)
            and type(saved['best_epoch']) is int and 0 <= saved['best_epoch'] < len(eligible)
            and eligible[saved['best_epoch']] == saved['best_metric'], 'Saved best metric differs from eligible stage-zero/prefix')
    require(saved['model_state'].keys() == saved['best_model_state'].keys(), 'Checkpoint tensor keys differ')
    require(any(n.startswith('forecast_fusion.') for n in saved['model_state']) == (arm != 'CONTROL'),
            'Checkpoint CONTROL/fusion semantics changed')
    require(saved['global_updates'] == saved['completed_epoch']*identity['train_batches'], 'Optimizer update budget changed')
    for key, value in saved['model_state'].items():
        best = saved['best_model_state'][key]
        require(value.shape == best.shape and value.dtype == best.dtype and bool(torch.isfinite(value).all())
                and bool(torch.isfinite(best).all()), 'Invalid checkpoint tensor: '+key)


def repair_sidecars(root, saved, args, protocol):
    for name, expected in (('history.json', saved['history']), ('summary.json', summary_for(saved, args, protocol))):
        path = root/name
        if path.is_file():
            try:
                previous = legacy.read_json(path)
            except (json.JSONDecodeError, UnicodeDecodeError):
                path.with_name(name+'.interrupted_'+uuid.uuid4().hex[:6]).write_bytes(path.read_bytes())
                previous = None
            if previous is not None:
                if name == 'history.json':
                    require(isinstance(previous, list) and previous == saved['history'][:len(previous)],
                            'JSON history conflicts with checkpoint; refusing overwrite')
                else:
                    epoch = previous.get('completed_epoch')
                    require(previous.get('arm') == saved['arm'] and type(epoch) is int and 1 <= epoch <= saved['completed_epoch'],
                            'JSON summary conflicts with checkpoint')
                    row = saved['history'][epoch-1]
                    require(all(previous.get(k) == row[k] for k in ('best_epoch', 'best_metric', 'global_updates', 'curriculum_length')),
                            'JSON summary counters conflict with checkpoint')
        write_json(path, expected)


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
    history, best, wait, saved = [], identity['source']['reference_metrics'][0], 0, None
    best_state, best_epoch = cpu_state(model.state_dict()), 0
    if path.is_file():
        saved = torch.load(path, map_location='cpu', weights_only=False)
        validate_saved(saved, identity, arm)
        repair_sidecars(root, saved, args, protocol)
        restore(saved, identity, arm, engine, data)
        history, best, wait, best_state, best_epoch = (saved[k] for k in ('history', 'best_metric', 'wait', 'best_model_state', 'best_epoch'))
    starting_epoch = len(history)
    updates_before = engine._iter_cnt
    for epoch in range(starting_epoch+1, args.epochs+1):
        if wait >= protocol['patience']:
            break
        start = time.perf_counter()
        train_mae, train_mape, train_rmse = engine.train_batch()
        val_mae, val_mape, val_rmse = engine.evaluate('val')
        require(np.isfinite([train_mae, train_mape, train_rmse, val_mae, val_mape, val_rmse]).all(), 'Non-finite F2 metrics')
        if val_mae < best:
            best, wait, best_state, best_epoch = val_mae, 0, cpu_state(model.state_dict()), epoch
        else:
            wait += 1
        lr = engine._optimizer.param_groups[0]['lr']
        fusion_lr = engine._optimizer.param_groups[1]['lr'] if arm != 'CONTROL' else None
        engine._lr_scheduler.step()
        row = dict(epoch=epoch, train_mae=float(train_mae), train_mape=float(train_mape), train_rmse=float(train_rmse),
            validation_mae=float(val_mae), validation_mape=float(val_mape), validation_rmse=float(val_rmse),
            best_metric=best, best_epoch=best_epoch, global_updates=engine._iter_cnt, curriculum_length=engine._cl_len,
            learning_rate_used=lr, next_learning_rate=engine._optimizer.param_groups[0]['lr'],
            fusion_learning_rate_used=fusion_lr,
            train_order_sha256=hashlib.sha256(data.loaders['train_loader'].order.tobytes()).hexdigest(),
            validation_diagnostics=engine.diagnostics, seconds=time.perf_counter()-start)
        history.append(row)
        saved = dict(identity=identity, arm=arm, completed_epoch=epoch, history=history,
            parameters=sum(p.numel() for p in model.parameters()), model_state=cpu_state(model.state_dict()),
            best_model_state=best_state, best_metric=best, best_epoch=best_epoch, wait=wait,
            optimizer_state=engine._optimizer.state_dict(), scheduler_state=engine._lr_scheduler.state_dict(),
            global_updates=engine._iter_cnt, curriculum_length=engine._cl_len,
            train_order=data.loaders['train_loader'].order.copy(), numpy_rng=np.random.get_state(),
            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if args.device.startswith('cuda') else [])
        validate_saved(saved, identity, arm)
        save_checkpoint(path, saved)
        repair_sidecars(root, saved, args, protocol)
        print(json.dumps(dict(arm=arm, **row), allow_nan=False), flush=True)
        gc.collect()
    require(saved is not None, 'No committed stage checkpoint')
    result = dict(summary_for(saved, args, protocol), optimizer_updates_this_invocation=saved['global_updates']-updates_before)
    write_json(root/'summary.json', result)
    print(json.dumps(dict(arm=arm, status=result['status'], completed_epoch=result['completed_epoch'],
        best_epoch=result['best_epoch'], best_metric=result['best_metric'],
        optimizer_updates_this_invocation=result['optimizer_updates_this_invocation'], output=str(root/'summary.json')), allow_nan=False), flush=True)
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
    common = min(map(len, histories.values()), default=0) if len(histories) == len(args.arms) else 0
    comparisons = {}
    baseline = identity['source']['reference_metrics'][0]
    if common:
        for control, candidate in (('CONTROL', 'STATE'), ('STATE', 'EVENT'), ('CONTROL', 'EVENT')):
            if control not in histories or candidate not in histories:
                continue
            left, right = histories[control][:common], histories[candidate][:common]
            comparisons[candidate+'_vs_'+control] = dict(common_epoch_mae_gain=left[-1]['validation_mae']-right[-1]['validation_mae'],
                common_prefix_best_mae_gain=min([baseline]+[r['validation_mae'] for r in left])-min([baseline]+[r['validation_mae'] for r in right]),
                candidate_common_last_five=[r['validation_mae'] for r in right[-5:]],
                control_common_last_five=[r['validation_mae'] for r in left[-5:]])
        require(all(len({histories[a][e]['train_order_sha256'] for a in histories}) == 1 for e in range(common)),
                'Paired sample order differs')
    budgets = {a: s['completed_epoch'] for a, s in summaries.items()}
    updates = {a: s['global_updates'] for a, s in summaries.items()}
    equal = len(summaries) == len(args.arms) and len(set(budgets.values())) == len(set(updates.values())) == 1
    complete = len(summaries) == len(args.arms) and all(s['completed_epoch'] >= args.epochs or s['early_stopped'] for s in summaries.values())
    return dict(schema=SCHEMA, status='MIN_COMPONENT_FUSION_PAIRED_STAGE_COMPLETE' if complete else 'MIN_COMPONENT_FUSION_PAIRED_STAGE_PARTIAL',
        common_epoch=common, runs=summaries, comparisons=comparisons, completed_stage_epochs=budgets,
        optimizer_updates=updates, same_completed_training_budget=equal,
        STATE_vs_CONTROL_scope='component_fusion_architecture_plus_parameter_increment_same_backbone_finetuning_budget',
        EVENT_vs_STATE_scope='report_conditioning_increment_same_registered_fused_architecture',
        versus_source_scope='includes_additional_backbone_finetuning_not_structure_only',
        primary_baseline='original_IGSTGNN_not_in_this_training_stage', source_best_epoch=identity['source']['best_epoch'],
        predictive_gain_claim=False, test_evaluated=False)


def prepare(args, logger, *, skip_preflight=False):
    protocol = load_protocol(args)
    hashes = paper.verify_release(args.data_dir, protocol)
    data = EventPublishedData(args.data_dir, protocol, logger, check=args.check)
    graph, _, info = legacy.load_graph(args.graph_dir, data.station_ids, protocol['graph_sha256'])
    source = load_source(args, hashes)
    require(data.loaders['train_loader'].bs == data.loaders['val_loader'].bs == source['identity']['batch_size'],
            'Keep the source original training/validation batch size')
    if skip_preflight:
        initialization = legacy.read_json(args.run_dir/'initialization.json')
    else:
        initialization = preflight(args, data, graph, source)
    identity = make_identity(args, protocol, hashes, info, initialization, source)
    identity['train_batches'] = data.loaders['train_loader'].num_batch
    return protocol, data, graph, initialization, identity, source


def child_command(args, arm):
    command = [sys.executable, '-u', str(Path(__file__).resolve()), '--run-dir', str(args.run_dir),
        '--data-dir', str(args.data_dir), '--graph-dir', str(args.graph_dir), '--protocol', str(args.protocol),
        '--source-run', str(args.source_run), '--reference-run', str(args.reference_run), '--epochs', str(args.epochs),
        '--seed', str(args.seed), '--device', args.device, '--threads', str(args.threads), '--arms', *args.arms, '--worker', arm]
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
            write_json(session/(arm+'_process_result.json'), dict(exit_code=process.returncode, launcher_interrupted=True))
            raise
        finally:
            process.stdout.close()
    write_json(session/(arm+'_process_result.json'), dict(exit_code=code, launcher_interrupted=False))
    return code


def evaluate_after_training(args):
    require(all((args.run_dir/a/'last.pt').is_file() for a in ARMS), 'Automatic evaluation requires all three saved arms')
    command = [sys.executable, '-u', str(HERE/'evaluate_min_component_fusion.py'), '--run-dir', str(args.run_dir),
        '--source-run', str(args.source_run), '--reference-run', str(args.reference_run), '--data-dir', str(args.data_dir),
        '--graph-dir', str(args.graph_dir), '--protocol', str(args.protocol), '--device', args.device, '--threads', str(args.threads)]
    if args.check:
        command.append('--check')
    result = subprocess.run(command, cwd=REPO)
    require(result.returncode == 0, 'Read-only F2 evaluation failed; training checkpoints preserved')


def main(argv=None):
    args = parse_args(argv)
    existing = (args.run_dir/'identity.json').is_file()
    if existing:
        require(legacy.read_json(args.run_dir/'identity.json').get('schema') == SCHEMA, 'Use a fresh F2 directory; old checkpoints are preserved')
    require(not args.worker or existing, 'Worker requires prepared parent identity')
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    require(device.type in ('cpu', 'cuda') and (device.type != 'cuda' or torch.cuda.is_available()), 'Requested device unavailable')
    logger = logging.getLogger('min_component_fusion')
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler(sys.stdout))
    args.run_dir.mkdir(parents=True, exist_ok=True)
    if args.worker:
        root = args.run_dir/args.worker
        root.mkdir(exist_ok=True)
        with legacy.run_lock(root), readonly_source_lock(args.source_run):
            before = source_snapshot(args)
            protocol, data, graph, initialization, identity, source = prepare(args, logger, skip_preflight=True)
            verify_identity(args, identity, initialization, allow_new=False)
            verify_source_reference(args, identity)
            try:
                train_arm(args, args.worker, protocol, data, identity, graph, logger, source)
            finally:
                after = source_snapshot(args)
                write_json(root/'source_preservation.json', dict(before=before, after=after, source_artifacts_preserved=before == after))
                require(before == after, 'Source MIN A artifacts changed during worker')
        return
    with legacy.run_lock(args.run_dir):
        with readonly_source_lock(args.source_run):
            before = source_snapshot(args)
            protocol, data, graph, initialization, identity, source = prepare(args, logger, skip_preflight=existing)
            if not existing:
                require({p.name for p in args.run_dir.iterdir()} <= {'active.lock'}, 'Use a fresh F2 run directory')
                create_source_reference(args, data, graph, source)
                identity = make_identity(args, protocol, identity['package'], identity['graph'], initialization, source)
                identity['train_batches'] = data.loaders['train_loader'].num_batch
            verify_identity(args, identity, initialization, allow_new=True)
            verify_source_reference(args, identity)
        write_json(args.run_dir/'data_audit.json', data.audit)
        session = args.run_dir/('session_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6])
        session.mkdir()
        write_json(session/'environment.json', legacy.environment(device))
        write_json(session/'request.json', dict(epochs=args.epochs, arms=args.arms, only=args.only,
            source_run=str(args.source_run), reference_run=str(args.reference_run), batch_size=identity['batch_size'], evaluate_after=args.evaluate_after))
        print('Run directory: '+str(args.run_dir), flush=True)
        print('Session directory: '+str(session), flush=True)
        print('Initialization: '+initialization['status']+'; CONTROL is extra fine-tuning of the existing MIN', flush=True)
        del data, graph, source
        gc.collect()
        results = {}
        for arm in (args.only or args.arms):
            print('Starting '+arm+' ('+ARMS[arm]+')', flush=True)
            code = run_worker(args, arm, session)
            results[arm] = dict(exit_code=code)
            write_json(session/'process_results.json', results)
            require(code == 0, 'Worker failed; checkpoints preserved. See '+str(session/(arm+'.log')))
            require(source_snapshot(args) == before, 'Source MIN artifacts changed while workers ran')
        report = comparison(args, identity)
        write_json(session/'comparison.json', report)
        write_json(args.run_dir/'latest_stage.json', report)
        write_json(session/'source_preservation.json', dict(before=before, after=source_snapshot(args), source_artifacts_preserved=True))
        print(json.dumps(dict(report, output=str(session/'comparison.json')), indent=2, allow_nan=False), flush=True)
    if args.evaluate_after:
        evaluate_after_training(args)


if __name__ == '__main__':
    main()
