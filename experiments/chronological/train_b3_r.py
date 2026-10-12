"""Independent B3/B3-R paired runner using the unchanged IS1.1 epoch/checkpoint loop."""
import argparse
from datetime import datetime
import gc
import hashlib
import importlib.util
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
from experiments.chronological import train_internal_spatial as legacy
from experiments.chronological.audit_b3_r_development import audit_development
from experiments.chronological.b3_r_diagnostics import CapacityDiagnostics, SCHEMA as DIAGNOSTIC_SCHEMA
from experiments.chronological.smoke import device_batch, sha256, verify_package
from experiments.chronological.train import batches, validate_protocol
from src.models.incident_spatial import require, state_capacity
from src.models.incident_spatial_residual import MAPPING_VERSION, from_canonical_b3
from src.utils.chronological import ChronologicalDataset

HERE = Path(__file__).resolve().parent
ARMS = {'B3': 'canonical_b3', 'B3R': 'pd_report_on', 'B3R_OFF': 'pd_report_off_tiid_on'}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, default=legacy.default_data())
    p.add_argument('--run-dir', type=Path)
    p.add_argument('--protocol', type=Path, default=HERE/'internal_spatial_b3_r_v11.json')
    p.add_argument('--graph-dir', type=Path, default=HERE/'internal_spatial_graph_is11')
    p.add_argument('--arms', nargs='+', choices=ARMS, default=['B3', 'B3R'])
    p.add_argument('--epochs', type=int, default=1, help='Absolute target epoch, including completed epochs')
    p.add_argument('--seed', type=int, default=2025)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--threads', type=int, default=3)
    p.add_argument('--check', action='store_true', help='Only first four train/val rows, batch two; not a ranking')
    p.add_argument('--worker', choices=ARMS, help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    require(args.epochs >= 1 and args.threads >= 1 and 0 <= args.seed < 2**32, 'Invalid epoch/threads/seed')
    require(not args.check or args.epochs == 1, 'Engineering check is exactly one small epoch')
    require(len(args.arms) == len(set(args.arms)) and (not args.worker or args.worker in args.arms), 'Invalid arm selection')
    if args.run_dir is None:
        require(not args.worker, 'Worker requires an existing run directory')
        args.run_dir = REPO/'experiments/chronological_runs'/('contra_b3r_'+('check_' if args.check else '')
                       +datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6])
    for name in ('data_dir', 'run_dir', 'protocol', 'graph_dir'):
        setattr(args, name, getattr(args, name).resolve())
    return args


def build_arm(args, arm, nodes, graph, device):
    canonical = legacy.build_arm(args, 'B3', nodes, graph, device)
    if arm == 'B3':
        return canonical
    require(arm in ARMS, 'Unknown B3-R arm')
    model, _ = from_canonical_b3(canonical, report_path_enabled=arm == 'B3R')
    return model


def training_loop(arms):
    """An isolated module namespace supplies only three explicit extension points.

    The old module and its B0-B3 globals are never patched. Its worker, resume,
    optimizer, epoch, metrics, data order and checkpoint functions are reused.
    """
    spec = importlib.util.spec_from_file_location('_b3r_frozen_is11_loop', Path(legacy.__file__))
    loop = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loop)
    old_diagnostics = loop.Diagnostics

    class Diagnostics:
        def __init__(self):
            self.collector = None

        def update(self, context):
            if self.collector is None:
                self.collector = CapacityDiagnostics() if 'base_node_capacity' in context else old_diagnostics()
            self.collector.update(context)

        def result(self):
            return self.collector.result() if self.collector is not None else None

    loop.ARMS = {arm: ARMS[arm] for arm in arms}
    loop.build_arm, loop.Diagnostics = build_arm, Diagnostics
    return loop


def preflight(args, train, graph):
    # Stable CPU certificate allows compatible GPU SKU changes on resume. Device
    # environment is audited separately, never hashed via measured GPU outputs.
    device = torch.device('cpu')
    batch = device_batch(next(batches(train, [int(r['sample_index']) for r in train.rows[:2]], 2)), device)
    canonical = legacy.build_arm(args, 'B3', len(train.station_ids), graph, device).eval()
    candidate, certificate = from_canonical_b3(canonical)
    candidate.eval()
    with torch.inference_mode():
        reference = canonical(batch['x'], incident_data=batch['incident'], spatial_report_enabled=False, return_spatial_details=True)
        on = candidate(batch['x'], incident_data=batch['incident'], return_spatial_details=True)
        off = candidate(batch['x'], incident_data=batch['incident'], spatial_report_enabled=False, return_spatial_details=True)
    require(torch.equal(on['prediction'], off['prediction']), 'B3-R initial ON/OFF predictions differ')
    require(torch.equal(on['spatial']['node_capacity'], off['spatial']['node_capacity']), 'B3-R initial ON/OFF capacity differs')
    old_capacity = state_capacity(reference['spatial']['prior'], reference['spatial']['graph'])
    capacity_error = float((old_capacity-off['spatial']['capacity']).abs().max())
    prediction_error = float((reference['prediction']-off['prediction']).abs().max())
    require(torch.allclose(old_capacity, off['spatial']['capacity'], rtol=2e-6, atol=2e-7), 'Mapped historical capacities differ')
    require(torch.allclose(reference['prediction'], off['prediction'], rtol=1e-5, atol=2e-6), 'Mapped historical predictions differ')
    if len(train.station_ids) == 496:
        require(sum(p.numel() for p in canonical.parameters()) == 458360 and certificate['parameter_count'] == 455780,
                'Full model parameter counts changed')
    manifests = {'B3': legacy.parameter_manifest(canonical), 'B3R': legacy.parameter_manifest(candidate)}
    manifests['B3R_OFF'] = manifests['B3R']
    return dict(status='B3R_INITIALIZATION_PAIR_PASS', mapping=certificate,
                arms={arm: manifests[arm] for arm in args.arms},
                canonical_spatial_off_capacity_abs_max=capacity_error,
                canonical_spatial_off_prediction_abs_max=prediction_error, initial_on_off_prediction_abs_max=0.,
                tolerances=dict(capacity_atol=2e-7, capacity_rtol=2e-6, prediction_atol=2e-6, prediction_rtol=1e-5),
                check_device=str(device), tiid_report_enabled=True)


def source_fingerprints():
    names = ('src/models/incident_spatial_residual.py', 'experiments/chronological/train_b3_r.py',
             'experiments/chronological/b3_r_diagnostics.py', 'experiments/chronological/audit_b3_r_development.py',
             'experiments/chronological/internal_spatial_is11.json')
    return dict(legacy.source_fingerprints(), **{name: sha256(REPO/name) for name in names})


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def validate_architecture_protocol(protocol):
    require((protocol.get('architecture_version'), protocol.get('initialization_mapping'), protocol.get('diagnostic_schema'),
             protocol.get('latent_channels'), protocol.get('spatial_steps'), protocol.get('eta'), protocol.get('arms')) ==
            ('B3-R1.1', MAPPING_VERSION, DIAGNOSTIC_SCHEMA, 4, 2, .5, list(ARMS)), 'B3-R architecture protocol mismatch')
    original = legacy.read_json(HERE/'internal_spatial_is11.json')
    changed_fields = {'protocol_id', 'architecture_version', 'arms', 'report_control'}
    require(all(protocol.get(key) == value for key, value in original.items() if key not in changed_fields),
            'B3-R must retain the original B3 training/input/exchange protocol')


def initialization_identity(initialization):
    # Measured discrepancies must pass tolerance on every invocation, but their
    # last bits can vary across CPU hosts. Only the exact copied state/shape and
    # mapping contract belong in strict resume identity.
    measured = {'canonical_spatial_off_capacity_abs_max', 'canonical_spatial_off_prediction_abs_max'}
    return {key: value for key, value in initialization.items() if key not in measured}


def make_identity(args, protocol, hashes, graph_info, initialization):
    identity = legacy.make_identity(args, protocol, hashes, graph_info)
    identity.update(schema='b3r_run_v1', source_sha256=source_fingerprints(), initialization_mapping=MAPPING_VERSION,
                    initialization_sha256=json_hash(initialization_identity(initialization)), diagnostic_schema=DIAGNOSTIC_SCHEMA,
                    arm_modes={arm: ARMS[arm] for arm in args.arms}, tiid_report_enabled=True,
                    trainable_scope='all_registered_parameters_no_freezing_no_detach')
    return identity


def paired_report(root, arms):
    summaries = {arm: legacy.read_json(root/arm/'summary.json') for arm in arms}
    history = {arm: legacy.read_json(root/arm/'history.json') for arm in arms}
    common = min(v['completed_epoch'] for v in summaries.values())
    comparisons = {}
    for candidate, control in (('B3R', 'B3'), ('B3R', 'B3R_OFF')):
        if candidate in arms and control in arms:
            comparisons[candidate+'_vs_'+control] = dict(
                common_epoch_mae_gain=history[control][common-1]['validation_mae_macro']-history[candidate][common-1]['validation_mae_macro'],
                common_prefix_best_mae_gain=min(v['validation_mae_macro'] for v in history[control][:common])
                                           -min(v['validation_mae_macro'] for v in history[candidate][:common]),
                best_mae_gain=summaries[control]['best_metric']-summaries[candidate]['best_metric'],
                best_metrics_completed_epochs={arm: summaries[arm]['completed_epoch'] for arm in (control, candidate)},
                candidate_common_last_five=[v['validation_mae_macro'] for v in history[candidate][max(0, common-5):common]],
                control_common_last_five=[v['validation_mae_macro'] for v in history[control][max(0, common-5):common]])
    return dict(status='B3R_PAIRED_STAGE_COMPLETE', common_epoch=common, arms=summaries, comparisons=comparisons,
                gain_sign='control_minus_candidate_mae',
                scope='engineering_only' if legacy.read_json(root/'identity.json')['engineering_check'] else 'exploratory_validation',
                attribution='B3R/B3 report-update parameterization; B3R/OFF entire D path with original TIID on',
                predictive_gain_claim=False, test_read=False)


def launch(args, train, val, graph, device, initialization):
    session = args.run_dir/('session_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6])
    session.mkdir()
    environment = legacy.environment(device)
    legacy.write_json(session/'environment.json', environment)
    legacy.write_json(session/'initialization_verified.json', initialization)
    initial_path = args.run_dir/'initial_environment.json'
    if initial_path.exists():
        original = legacy.read_json(initial_path)
        changes = {k: dict(saved=original.get(k), current=v) for k, v in environment.items() if original.get(k) != v}
        legacy.write_json(session/'environment_changes.json', changes)
        if 'gpu' in changes:
            print('RESUME_GPU_NAME_CHANGE_ACCEPTED '+json.dumps(changes['gpu']), flush=True)
    else:
        legacy.write_json(initial_path, environment)
    audit = audit_development(train, val, legacy.read_json(args.data_dir/'context_manifest.json'))
    legacy.write_json(session/'data_time_event_audit.json', audit)
    require(not audit['blocking_contract_conflict'], 'Development time contract conflict; audit preserved, do not silently change old data')
    print('Run directory: '+str(args.run_dir), flush=True)
    print('Session directory: '+str(session), flush=True)
    print('Initialization: '+initialization['status']+'; time scope: '+audit['status'], flush=True)
    # Workers run sequentially and release their model/optimizer/activation allocations on exit.
    del train, val, graph
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    for arm in args.arms:
        command = [sys.executable, '-X', 'utf8', '-u', str(Path(__file__).resolve()), '--worker', arm,
                   '--run-dir', str(args.run_dir), '--data-dir', str(args.data_dir), '--graph-dir', str(args.graph_dir),
                   '--protocol', str(args.protocol), '--epochs', str(args.epochs), '--seed', str(args.seed),
                   '--device', args.device, '--threads', str(args.threads), '--arms', *args.arms]
        if args.check:
            command.append('--check')
        print('Starting '+arm, flush=True)
        with (session/(arm+'.log')).open('w', encoding='utf-8') as log:
            process = subprocess.Popen(command, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, encoding='utf-8', errors='replace')
            for line in process.stdout:
                print(line, end='', flush=True)
                log.write(line)
                log.flush()
            code = process.wait()
        legacy.write_json(session/(arm+'_process_result.json'), dict(arm=arm, exit_code=code))
        require(code == 0, f'{arm} process failed ({code}); checkpoints preserved; see {session/(arm+".log")}')
    report = paired_report(args.run_dir, args.arms)
    legacy.write_json(session/'comparison.json', report)
    legacy.write_json(args.run_dir/'latest_comparison.json', report)
    print(json.dumps(dict(status=report['status'], common_epoch=report['common_epoch'], comparisons=report['comparisons'],
                         output=str(session/'comparison.json')), indent=2), flush=True)


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    require(device.type in ('cpu', 'cuda') and (device.type != 'cuda' or torch.cuda.is_available()), 'Requested device unavailable')
    protocol = legacy.read_json(args.protocol)
    validate_architecture_protocol(protocol)
    for filename, key in (('summary.json', 'files'), ('context_manifest.json', 'outputs')):
        names = legacy.read_json(args.data_dir/filename)[key]
        require(all(Path(name).name == name and not name.lower().startswith('test') for name in names),
                'Development manifest attempts to read a non-development path')
    hashes = verify_package(args.data_dir)
    train, val = ChronologicalDataset(args.data_dir, 'train'), ChronologicalDataset(args.data_dir, 'val')
    validate_protocol(protocol, hashes, len(train), len(val), len(train.station_ids))
    require(args.epochs <= protocol['max_epochs'], 'Requested epochs exceed frozen budget')
    graph, roads, graph_info = legacy.load_graph(args.graph_dir, train.station_ids, protocol['graph_sha256'])
    # Every worker checks the full initialization certificate before loading a checkpoint.
    initialization = preflight(args, train, graph)
    identity = make_identity(args, protocol, hashes, graph_info, initialization)
    if (args.run_dir/'identity.json').exists():
        require(legacy.read_json(args.run_dir/'identity.json') == identity, 'Run identity changed; preserved checkpoints')
        require(initialization_identity(legacy.read_json(args.run_dir/'initialization.json')) == initialization_identity(initialization),
                'Reconstructed initialization differs')
    else:
        require(not args.worker and not args.run_dir.exists(), 'Use a fresh B3-R directory; never resume an old B3 run here')
        args.run_dir.mkdir(parents=True)
        legacy.write_json(args.run_dir/'identity.json', identity)
        legacy.write_json(args.run_dir/'initialization.json', initialization)
    if args.worker:
        training_loop(args.arms).worker(args, identity, protocol, train, val, graph, roads)
    else:
        with legacy.run_lock(args.run_dir):
            launch(args, train, val, graph, device, initialization)


if __name__ == '__main__':
    main()
