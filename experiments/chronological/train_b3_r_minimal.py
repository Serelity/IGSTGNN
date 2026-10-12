"""Train the pruned history-only B3R_MIN in a fresh, resumable run.

TIID, incident encoders and D are absent from the registered model. Historical
traffic, calendar time, the original graph backbone, P and constrained latent
exchange remain. Report metadata serves package audits and validation cohorts.
"""
import argparse
from datetime import datetime
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.chronological import train_b3_r as base
from experiments.chronological import train_internal_spatial as legacy
from experiments.chronological.audit_b3_r_development import audit_development
from experiments.chronological.b3_r_diagnostics import CapacityDiagnostics
from experiments.chronological.smoke import device_batch, sha256, verify_package
from experiments.chronological.train import batches, validate_protocol
from src.models.incident_spatial import require
from src.models.incident_spatial_minimal import from_residual, cutoff_clock_from_history
from src.models.incident_spatial_residual import tensor_state_hash
from src.utils.chronological import ChronologicalDataset

HERE = Path(__file__).resolve().parent
ARM = 'B3R_MIN'
ARMS = {ARM: 'history_P_restricted_exchange_no_TIID_no_D_no_incident_encoder'}
SCHEMA = 'b3r_minimal_run_v1'
MAPPING = 'b3r_prune_tiid_d_incident_encoders_v1'
DIAGNOSTIC_SCHEMA = 'b3r_minimal_diagnostics_v1'


class MinimalDiagnostics(CapacityDiagnostics):
    """Preserve the exchange audit without inventing an incident correction."""
    def update(self, context):
        require(not context['report_path_executed'] and context['report_logit_delta'] is None,
                'Minimal model executed report correction')
        require(not bool(context['report_mask'].any())
                and not bool(context['effective_logit_delta'].any()), 'Minimal model has report effects')
        super().update(context)

    def result(self):
        result = super().result()
        if result is not None:
            result.update(schema=DIAGNOSTIC_SCHEMA, incident_inputs_used=False,
                          tiid_module_present=False, D_module_present=False,
                          capacity_scope='history_conditioned_dimensionless_latent_budget',
                          physics_capacity_identification_claim=False)
        return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=legacy.default_data())
    parser.add_argument('--graph-dir', type=Path, default=HERE/'internal_spatial_graph_is11')
    parser.add_argument('--protocol', type=Path, default=HERE/'internal_spatial_b3_r_minimal_v1.json')
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--epochs', type=int, help='Absolute target epoch; default 10, or 1 with --check')
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--threads', type=int, default=3)
    parser.add_argument('--check', action='store_true', help='First four train/validation rows; engineering only')
    parser.add_argument('--worker', choices=ARMS, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    args.epochs = (1 if args.check else 10) if args.epochs is None else args.epochs
    args.arms = [ARM]
    require(args.epochs >= 1 and args.threads >= 1 and 0 <= args.seed < 2**32, 'Invalid epoch/threads/seed')
    require(not args.check or args.epochs == 1, 'Engineering check is exactly one small epoch')
    if args.run_dir is None:
        require(not args.worker, 'Internal worker requires an existing run directory')
        args.run_dir = REPO/'experiments/chronological_runs'/('contra_b3r_min_'+('check_' if args.check else '')
                        +datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6])
    for name in ('data_dir', 'graph_dir', 'protocol', 'run_dir'):
        setattr(args, name, getattr(args, name).resolve())
    return args


def build_arm(args, arm, nodes, graph, device):
    require(arm == ARM, 'Unknown minimal arm')
    reference = base.build_arm(args, 'B3R', nodes, graph, device)
    return from_residual(reference)[0]


def training_loop(arms=None):
    require(arms is None or list(arms) == [ARM], 'Minimal run contains exactly one arm')
    loop = base.training_loop(['B3R'])
    loop.ARMS = dict(ARMS)
    loop.build_arm, loop.Diagnostics = build_arm, MinimalDiagnostics
    return loop


def preflight(args, train, graph):
    batch = device_batch(next(batches(train, [int(r['sample_index']) for r in train.rows[:2]], 2)), 'cpu')
    reference = base.build_arm(args, 'B3R', len(train.station_ids), graph, torch.device('cpu')).eval()
    reference_manifest = legacy.parameter_manifest(reference)
    model, conversion = from_residual(reference)
    model.eval()
    ref_state, minimal_state = reference.state_dict(), model.state_dict()
    retained_exact = minimal_state.keys() <= ref_state.keys() and all(
        torch.equal(value, ref_state[name]) for name, value in minimal_state.items())
    require(retained_exact, 'Minimal conversion changed retained parameters or buffers')
    tod, dow = cutoff_clock_from_history(batch['x'])
    require(torch.equal(tod, batch['incident']['forecast_tod'])
            and torch.equal(dow, batch['incident']['forecast_dow']),
            'History-derived forecast clock disagrees with development metadata')
    with torch.inference_mode():
        expected = reference(batch['x'], incident_data=batch['incident'],
                             spatial_report_enabled=False, tiid_report_enabled=False,
                             return_spatial_details=True)
        actual = model(batch['x'], incident_data=None, return_spatial_details=True)
        ignored = model(batch['x'], incident_data={'ignored': object()})
    require(torch.equal(expected['prediction'], actual['prediction']),
            'Pruned model differs from reference with both incident paths disabled')
    require(torch.equal(actual['prediction'], ignored), 'Minimal prediction depends on incident input')
    require(torch.equal(expected['spatial']['capacity'], actual['spatial']['capacity']),
            'Pruned historical capacity differs from reference')
    require(not actual['spatial']['report_path_executed'] and actual['spatial']['report_logit_delta'] is None,
            'Minimal model executed D')
    manifest = legacy.parameter_manifest(model)
    require(manifest['total_parameters'] == manifest['trainable_parameters']
            and manifest['total_parameters'] < reference_manifest['total_parameters'],
            'Pruning did not remove trainable parameters')
    return dict(status='B3RMIN_INITIALIZATION_PAIR_PASS', mapping=MAPPING, conversion=conversion,
                arms={ARM: manifest}, reference_arm='B3R',
                reference_common_initial_sha256=reference_manifest['common_initial_sha256'],
                reference_initial_parameters_sha256=reference_manifest['initial_parameters_sha256'],
                reference_seed=args.seed, reference_source_sha256=base.source_fingerprints(),
                retained_state_exact_verified=retained_exact,
                retained_state_sha256=tensor_state_hash(minimal_state),
                removed_parameters=reference_manifest['total_parameters']-manifest['total_parameters'],
                reference_parameter_count=reference_manifest['total_parameters'],
                initial_both_off_prediction_abs_max=0., initial_historical_capacity_abs_max=0.,
                history_clock_matches_metadata=True, incident_payload_ignored_exactly=True,
                check_device='cpu', task_input_scope='history_flow_and_calendar_time_only',
                incident_metadata_scope='validation_cohort_masks_only')


def source_fingerprints():
    names = ('src/models/incident_spatial_minimal.py', 'experiments/chronological/train_b3_r_minimal.py',
             'experiments/chronological/internal_spatial_b3_r_v11.json')
    return dict(base.source_fingerprints(), **{name: sha256(REPO/name) for name in names})


def validate_architecture_protocol(protocol):
    original = legacy.read_json(HERE/'internal_spatial_b3_r_v11.json')
    changed = {'protocol_id', 'architecture_version', 'initialization_mapping', 'diagnostic_schema', 'arms', 'report_control'}
    require(all(protocol.get(key) == value for key, value in original.items() if key not in changed),
            'Minimal protocol changed frozen training/data/exchange settings')
    require(protocol.get('architecture_version') == 'B3-R-MIN1.0'
            and protocol.get('initialization_mapping') == MAPPING
            and protocol.get('diagnostic_schema') == DIAGNOSTIC_SCHEMA
            and protocol.get('arms') == [ARM]
            and protocol.get('report_control') == 'TIID_D_and_incident_encoders_physically_removed'
            and protocol.get('minimal_spec') == dict(
                prediction_inputs='history_flow_and_calendar_time_only',
                retained=['original_temporal_and_graph_backbone', 'P', 'inlet_exit', 'constrained_latent_exchange', 'forecast_head'],
                removed=['TIID', 'D', 'incident_encoders', 'incident_key_projection'],
                clock='last_history_slot_plus_two_five_minute_slots_with_day_rollover',
                report_metadata='validation_cohort_masks_only'), 'Minimal architecture protocol mismatch')


def make_identity(args, protocol, hashes, graph_info, initialization):
    identity = legacy.make_identity(args, protocol, hashes, graph_info)
    identity.update(schema=SCHEMA, source_sha256=source_fingerprints(), arm_modes=dict(ARMS),
                    initialization_mapping=MAPPING, initialization_sha256=base.json_hash(initialization),
                    diagnostic_schema=DIAGNOSTIC_SCHEMA, tiid_report_enabled=False, D_module_present=False,
                    incident_inputs_used=False, trainable_scope='all_retained_parameters_trainable',
                    report_metadata_scope='validation_cohort_masks_only')
    return identity


def prepare(args, device):
    protocol = legacy.read_json(args.protocol)
    validate_architecture_protocol(protocol)
    require(args.epochs <= protocol['max_epochs'], 'Requested epochs exceed frozen budget')
    for filename, key in (('summary.json', 'files'), ('context_manifest.json', 'outputs')):
        names = legacy.read_json(args.data_dir/filename)[key]
        require(all(Path(name).name == name and not name.lower().startswith('test') for name in names),
                'Development manifest attempts a non-development path')
    hashes = verify_package(args.data_dir)
    train, val = ChronologicalDataset(args.data_dir, 'train'), ChronologicalDataset(args.data_dir, 'val')
    validate_protocol(protocol, hashes, len(train), len(val), len(train.station_ids))
    graph, roads, graph_info = legacy.load_graph(args.graph_dir, train.station_ids, protocol['graph_sha256'])
    initialization = preflight(args, train, graph)
    identity = make_identity(args, protocol, hashes, graph_info, initialization)
    return protocol, train, val, graph, roads, initialization, identity


def verify_identity(args, identity, initialization, allow_new):
    path = args.run_dir/'identity.json'
    if path.exists():
        require(legacy.read_json(path) == identity, 'Run identity changed; original checkpoints preserved')
        require(legacy.read_json(args.run_dir/'initialization.json') == initialization, 'Initialization certificate changed')
    else:
        require(allow_new and {p.name for p in args.run_dir.iterdir()} <= {'active.lock'},
                'Use a fresh minimal directory; old B3-R runs cannot be resumed here')
        legacy.write_json(path, identity)
        legacy.write_json(args.run_dir/'initialization.json', initialization)


def repair_sidecars(args, identity, protocol, device):
    """Rebuild interrupted JSON writes from the committed checkpoint only."""
    root, arm = args.run_dir/ARM, ARM
    path = root/'last.pt'
    if not path.exists():
        return False
    saved = torch.load(path, map_location='cpu', weights_only=False)
    require(saved.get('identity') == identity and saved.get('arm') == arm,
            'Checkpoint identity/arm mismatch before sidecar repair')
    history = saved.get('history')
    require(isinstance(history, list) and history and len(history) == saved.get('completed_epoch')
            and [row['epoch'] for row in history] == list(range(1, len(history)+1)), 'Invalid checkpoint history')
    require(history[-1]['best_metric'] == saved['best_metric']
            and history[-1]['global_updates'] == saved['global_updates'], 'Checkpoint history/state counters disagree')
    state, best_state = saved.get('model_state'), saved.get('best_model_state')
    require(isinstance(state, dict) and state and isinstance(best_state, dict) and state.keys() == best_state.keys(),
            'Invalid checkpoint model state')
    for name, value in state.items():
        best_value = best_state[name]
        require(isinstance(value, torch.Tensor) and isinstance(best_value, torch.Tensor)
                and value.shape == best_value.shape and value.dtype == best_value.dtype
                and bool(torch.isfinite(value).all()) and bool(torch.isfinite(best_value).all()),
                'Invalid checkpoint model tensor: '+name)
    specification = legacy.read_json(args.run_dir/'initialization.json')['arms'][arm]['specification']
    require(all(name in state and list(state[name].shape) == shape for name, shape in specification.items()),
            'Checkpoint model does not match initialization parameter shapes')

    def read_companion(path):
        if not path.exists():
            return None
        try:
            return legacy.read_json(path)
        except (json.JSONDecodeError, UnicodeDecodeError):
            path.with_name(path.name+'.interrupted_'+uuid.uuid4().hex[:6]).write_bytes(path.read_bytes())
            return None

    history_path, summary_path = root/'history.json', root/'summary.json'
    previous = read_companion(history_path)
    require(previous is None or (isinstance(previous, list) and len(previous) <= len(history)
                                and previous == history[:len(previous)]),
            'JSON history conflicts with authoritative checkpoint; refusing overwrite')
    summary = read_companion(summary_path)
    require(summary is None or isinstance(summary, dict), 'Invalid companion summary structure')
    if summary is None:
        parameters = dict(legacy.read_json(args.run_dir/'initialization.json')['arms'][arm])
        parameters.pop('initial_parameters_sha256', None)
        parameters.pop('common_initial_sha256', None)
        parameters['optimizer_parameters'] = parameters['total_parameters']
        summary = dict(parameters=parameters, environment=legacy.environment(device),
                       environment_scope='sidecar_repair_worker_not_recovered_training_provenance')
    summary.update(status='ENGINEERING_CHECK_PASS' if args.check else 'PAUSED_AT_EPOCH_BOUNDARY',
                   arm=arm, mode=ARMS[arm], completed_epoch=len(history), best_epoch=history[-1]['best_epoch'],
                   best_metric=saved['best_metric'], global_updates=saved['global_updates'],
                   early_stopped=saved['wait'] >= protocol['patience'], checkpoint='last.pt',
                   best_model_state_key='best_model_state', test_read=False, predictive_gain_claim=False,
                   incident_inputs_used=False, tiid_module_present=False, D_module_present=False)
    for target, value in ((history_path, history), (summary_path, summary)):
        temporary = target.with_suffix(target.suffix+'.repair.partial')
        legacy.write_json(temporary, value)
        temporary.replace(target)
    return True


def stage_report(root):
    summary = legacy.read_json(root/ARM/'summary.json')
    return dict(status='B3RMIN_STAGE_COMPLETE', completed_epoch=summary['completed_epoch'],
                best_epoch=summary['best_epoch'], best_metric=summary['best_metric'],
                arm=ARM, summary=summary, comparisons={},
                scope='engineering_only' if legacy.read_json(root/'identity.json')['engineering_check'] else 'exploratory_validation',
                attribution='fresh_training_with_TIID_D_and_incident_encoders_physically_removed',
                predictive_gain_claim=False, test_read=False)


def launch(args, device, initialization, audit):
    session = args.run_dir/('session_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6])
    session.mkdir()
    environment = legacy.environment(device)
    legacy.write_json(session/'environment.json', environment)
    initial_path = args.run_dir/'initial_environment.json'
    if initial_path.exists():
        original = legacy.read_json(initial_path)
        changes = {key: dict(saved=original.get(key), current=value) for key, value in environment.items()
                   if original.get(key) != value}
        legacy.write_json(session/'environment_changes.json', changes)
        if 'gpu' in changes:
            print('RESUME_GPU_NAME_CHANGE_ACCEPTED '+json.dumps(changes['gpu']), flush=True)
    else:
        legacy.write_json(initial_path, environment)
    legacy.write_json(session/'data_time_event_audit.json', audit)
    require(not audit['blocking_contract_conflict'], 'Development time contract conflict; audit preserved')
    legacy.write_json(session/'initialization_verified.json', initialization)
    print('Run directory: '+str(args.run_dir), flush=True)
    print('Session directory: '+str(session), flush=True)
    print('Initialization: '+initialization['status']+'; input scope: history flow/calendar only', flush=True)
    command = [sys.executable, '-X', 'utf8', '-u', str(Path(__file__).resolve()), '--worker', ARM,
               '--run-dir', str(args.run_dir), '--data-dir', str(args.data_dir), '--graph-dir', str(args.graph_dir),
               '--protocol', str(args.protocol), '--epochs', str(args.epochs), '--seed', str(args.seed),
               '--device', args.device, '--threads', str(args.threads)]
    if args.check:
        command.append('--check')
    env = dict(os.environ, B3R_MIN_INTERNAL_WORKER=str(args.run_dir)+'|'+ARM)
    print('Starting '+ARM, flush=True)
    with (session/(ARM+'.log')).open('w', encoding='utf-8') as log:
        process = subprocess.Popen(command, cwd=REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding='utf-8', errors='replace')
        try:
            for line in process.stdout:
                print(line, end='', flush=True)
                log.write(line)
                log.flush()
            code = process.wait()
        except BaseException:
            process.terminate()
            process.wait()
            raise
    legacy.write_json(session/(ARM+'_process_result.json'), dict(arm=ARM, exit_code=code))
    require(code == 0, f'{ARM} process failed ({code}); checkpoints preserved; see {session/(ARM+".log")}')
    report = stage_report(args.run_dir)
    legacy.write_json(session/'stage.json', report)
    legacy.write_json(args.run_dir/'latest_stage.json', report)
    print(json.dumps(dict(status=report['status'], completed_epoch=report['completed_epoch'],
                         best_epoch=report['best_epoch'], best_metric=report['best_metric'],
                         output=str(session/'stage.json')), indent=2), flush=True)


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    require(device.type in ('cpu', 'cuda') and (device.type != 'cuda' or torch.cuda.is_available()),
            'Requested device unavailable')
    if args.worker:
        require(os.environ.get('B3R_MIN_INTERNAL_WORKER') == str(args.run_dir)+'|'+ARM,
                'Use the launcher; direct internal workers bypass its lock')
        protocol, train, val, graph, roads, initialization, identity = prepare(args, device)
        verify_identity(args, identity, initialization, False)
        repair_sidecars(args, identity, protocol, device)
        training_loop().worker(args, identity, protocol, train, val, graph, roads)
        repair_sidecars(args, identity, protocol, device)
    else:
        args.run_dir.mkdir(parents=True, exist_ok=True)
        with legacy.run_lock(args.run_dir):
            protocol, train, val, graph, roads, initialization, identity = prepare(args, device)
            verify_identity(args, identity, initialization, True)
            audit = audit_development(train, val, legacy.read_json(args.data_dir/'context_manifest.json'))
            del train, val, graph, roads
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            launch(args, device, initialization, audit)


if __name__ == '__main__':
    main()
