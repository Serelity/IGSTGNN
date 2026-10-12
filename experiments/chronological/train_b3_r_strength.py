"""B3-R G1.1: identity-centered D strength and independent no-D training.

Fresh paired runs only. Old training files and checkpoints are never modified.
Only original history/report fields are used; the conditional offline time audit
is retained. Future bias labels never enter model inputs.
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
from src.models.incident_spatial_strength import from_residual
from src.models.incident_spatial_residual import tensor_state_hash
from src.utils.chronological import ChronologicalDataset

HERE = Path(__file__).resolve().parent
ARMS = {'B3R': 'original_D', 'B3R_HALF': 'fixed_half',
        'B3R_GATE': 'identity_gate', 'B3R_NO_D': 'pd_report_off_tiid_on'}
SCHEMA = 'b3r_strength_run_g11_v1'
MAPPING = 'b3r_common_pd_identity_gate_nod_g11_v1'
DIAGNOSTIC_SCHEMA = 'b3r_strength_diagnostics_g11_v1'


class GateDiagnostics(CapacityDiagnostics):
    """Count scalar node gates once per forward, separately from latent channels."""
    @torch.no_grad()
    def update(self, context):
        super().update(context)
        strength = context.get('d_strength')
        if strength is None and context['report_path_executed']:
            strength = torch.ones_like(context['report_mask'])
        if strength is not None:
            mask = context['report_mask'].bool()
            require(strength.shape == mask.shape and bool(torch.isfinite(strength).all())
                    and bool(((strength >= 0) & (strength <= 2)).all()), 'Invalid D strength')
            for name, value in (('d_strength', strength), ('d_strength_squared', strength.square()),
                                ('d_strength_below_0_1', (strength < .1).to(strength.dtype)),
                                ('d_strength_above_1_9', (strength > 1.9).to(strength.dtype)),
                                ('d_strength_below_one', (strength < 1).to(strength.dtype)),
                                ('d_strength_above_one', (strength > 1).to(strength.dtype))):
                self.add(name+':report_nodes', value, mask)
            raw = context.get('raw_effective_logit_delta', context['effective_logit_delta'])
            self.add('raw_effective_logit_delta_abs:report_nodes', raw.abs(), mask)
            self.add('applied_effective_logit_delta_abs:report_nodes', context['effective_logit_delta'].abs(), mask)
            self.add('D_abs_for_strength_weighting:report_nodes', raw.abs(), mask)
            self.add('strength_times_D_abs:report_nodes', strength*raw.abs(), mask)

    def result(self):
        result = super().result()
        if result is not None:
            result['schema'] = DIAGNOSTIC_SCHEMA
            result['d_strength_denominator'] = 'associated sample-node positions once per forward; not latent channels'
            values = result['capacity_statistics']
            mean = values.get('d_strength:report_nodes', {}).get('mean')
            squared = values.get('d_strength_squared:report_nodes', {}).get('mean')
            result['d_strength_population_std'] = None if mean is None else max(0., squared-mean*mean)**.5
            result['d_strength_status'] = ('not_computed' if result['raw_delta_status'] == 'not_computed'
                                           else 'available' if mean is not None else 'no_associated_nodes')
            denominator = values.get('D_abs_for_strength_weighting:report_nodes', {}).get('mean')
            numerator = values.get('strength_times_D_abs:report_nodes', {}).get('mean')
            result['d_magnitude_weighted_strength'] = None if denominator is None or denominator == 0 else numerator/denominator
            result['calibrated_reliability_claim'] = False
            result['strength_range'] = [0., 2.]
            result['neutral_strength'] = 1.
        return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=legacy.default_data())
    parser.add_argument('--graph-dir', type=Path, default=HERE/'internal_spatial_graph_is11')
    parser.add_argument('--protocol', type=Path, default=HERE/'internal_spatial_b3_r_strength_g11.json')
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--arms', nargs='+', choices=ARMS, default=list(ARMS))
    parser.add_argument('--only', nargs='+', choices=ARMS, help='Run/resume these arms; preserve the full --arms identity')
    parser.add_argument('--epochs', type=int, default=1, help='Absolute target epoch')
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--threads', type=int, default=3)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--worker', choices=ARMS, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    require(args.epochs >= 1 and args.threads >= 1 and 0 <= args.seed < 2**32, 'Invalid epoch/threads/seed')
    require(not args.check or args.epochs == 1, 'Engineering check is exactly one small epoch')
    require(len(args.arms) == len(set(args.arms)), 'Duplicate arms')
    args.only = args.arms if args.only is None else args.only
    require(len(args.only) == len(set(args.only)) and set(args.only) <= set(args.arms), 'Invalid --only selection')
    require(not args.worker or args.worker in args.arms, 'Worker is outside the run identity')
    if args.run_dir is None:
        require(not args.worker, 'Internal worker requires an existing run directory')
        args.run_dir = REPO/'experiments/chronological_runs'/('contra_b3rg11_'+('check_' if args.check else '')
                        +datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6])
    for name in ('data_dir', 'graph_dir', 'protocol', 'run_dir'):
        setattr(args, name, getattr(args, name).resolve())
    return args


def build_arm(args, arm, nodes, graph, device):
    require(arm in ARMS, 'Unknown strength arm')
    model = base.build_arm(args, 'B3R_OFF' if arm == 'B3R_NO_D' else 'B3R', nodes, graph, device)
    return model if arm in ('B3R', 'B3R_NO_D') else from_residual(model, ARMS[arm])[0]


def training_loop(arms):
    # Isolated IS1.1 namespace; no mutable globals in old runners are patched.
    loop = base.training_loop(['B3R'])
    loop.ARMS = {arm: ARMS[arm] for arm in arms}
    loop.build_arm, loop.Diagnostics = build_arm, GateDiagnostics
    return loop


def shared_state(model):
    return {name: value for name, value in model.state_dict().items()
            if not name.startswith('report_spatial_state.gate.')}


def preflight(args, train, graph):
    batch = device_batch(next(batches(train, [int(r['sample_index']) for r in train.rows[:2]], 2)), 'cpu')
    manifests, shared_hash, reference, nonzero_reference, counts = {}, None, None, None, {}
    for arm in ARMS:
        model = build_arm(args, arm, len(train.station_ids), graph, torch.device('cpu')).eval()
        shared = tensor_state_hash(shared_state(model))
        shared_hash = shared if shared_hash is None else shared_hash
        require(shared == shared_hash, 'Shared P/D/backbone/buffer initialization mismatch')
        with torch.inference_mode():
            on = model(batch['x'], incident_data=batch['incident'], return_spatial_details=True)
            off = model(batch['x'], incident_data=batch['incident'], spatial_report_enabled=False, return_spatial_details=True)
        require(torch.equal(on['prediction'], off['prediction']), 'Initial D zero must give exact ON/OFF equality')
        reference = on['prediction'] if reference is None else reference
        require(torch.equal(on['prediction'], reference), 'Initial four-arm predictions differ')
        if arm in ('B3R_HALF', 'B3R_GATE'):
            initial_strength = .5 if arm == 'B3R_HALF' else 1.
            require(torch.equal(on['spatial']['d_strength'], torch.full_like(on['spatial']['d_strength'], initial_strength)),
                    'Half/gate strength initialization mismatch')
        if arm == 'B3R_NO_D':
            require(not on['spatial']['report_path_executed']
                    and on['spatial']['report_logit_delta'] is None, 'NO_D executed report correction')
        if arm in ('B3R', 'B3R_GATE'):
            # A nonzero signed D fixture verifies the identity gate, not just
            # the trivial equality obtained from the zero output initialization.
            bias = model.report_spatial_state.D[-1].bias
            saved_bias = bias.detach().clone()
            try:
                with torch.no_grad():
                    bias.copy_((1-2*(torch.arange(bias.numel()) % 2)).to(bias)*.25)
                with torch.inference_mode():
                    fixture = model(batch['x'], incident_data=batch['incident'], return_spatial_details=True)
                if nonzero_reference is None:
                    nonzero_reference = (fixture['prediction'], fixture['spatial']['node_capacity'])
                else:
                    require(torch.equal(fixture['prediction'], nonzero_reference[0])
                            and torch.equal(fixture['spatial']['node_capacity'], nonzero_reference[1]),
                            'Identity gate differs from original for nonzero signed D')
            finally:
                with torch.no_grad():
                    bias.copy_(saved_bias)
            del fixture
        manifests[arm] = legacy.parameter_manifest(model)
        require(manifests[arm]['total_parameters'] == manifests[arm]['trainable_parameters'], 'Frozen parameter found')
        counts[arm] = manifests[arm]['total_parameters']
        del model, on, off
    require(counts['B3R'] == counts['B3R_HALF'] == counts['B3R_NO_D']
            and counts['B3R_GATE'] > counts['B3R'], 'Incorrect gate parameter control')
    if len(train.station_ids) == 496:
        require(counts == {'B3R': 455780, 'B3R_HALF': 455780, 'B3R_GATE': 458309, 'B3R_NO_D': 455780}, 'Full model counts changed')
    return dict(status='B3RSTRENGTH_INITIALIZATION_PAIR_PASS', mapping=MAPPING,
                arms={arm: manifests[arm] for arm in args.arms},
                shared_state_sha256=shared_hash, initial_prediction_abs_max=0.,
                half_initial_strength=.5, gate_initial_strength=1.,
                nonzero_D_gate_prediction_and_capacity_exact=True,
                no_D_training_path_disabled=True, original_tiid_on=True,
                check_device='cpu', task_input_scope='original_history_clock_report_distance_age_only')


def source_fingerprints():
    names = ('src/models/incident_spatial_strength.py', 'experiments/chronological/train_b3_r_strength.py')
    return dict(base.source_fingerprints(), **{name: sha256(REPO/name) for name in names},
                **{'experiments/chronological/internal_spatial_b3_r_v11.json': sha256(HERE/'internal_spatial_b3_r_v11.json')})


def validate_architecture_protocol(protocol):
    original = legacy.read_json(HERE/'internal_spatial_b3_r_v11.json')
    changed = {'protocol_id', 'architecture_version', 'initialization_mapping', 'diagnostic_schema', 'arms', 'report_control'}
    require(all(protocol.get(key) == value for key, value in original.items() if key not in changed),
            'Gate protocol changed original training/data/exchange settings')
    require(protocol.get('architecture_version') == 'B3-R-G1.1' and protocol.get('initialization_mapping') == MAPPING
            and protocol.get('diagnostic_schema') == DIAGNOSTIC_SCHEMA and protocol.get('arms') == list(ARMS)
            and protocol.get('gate_spec') == dict(input='history_summary_report_embedding_distance_age', hidden=16,
                                                output='one_scalar_per_node_shared_four_latents', initial_strength=1.,
                                                fixed_control=.5, range=[0., 2.], position='masked_D_logit_only',
                                                activation='twice_sigmoid')
            and protocol.get('report_control') == 'original_TIID_on_D_strength_and_independent_no_D',
            'Gate architecture protocol mismatch')


def make_identity(args, protocol, hashes, graph_info, initialization):
    identity = legacy.make_identity(args, protocol, hashes, graph_info)
    identity.update(schema=SCHEMA, source_sha256=source_fingerprints(), arm_modes={arm: ARMS[arm] for arm in args.arms},
                    initialization_mapping=MAPPING, initialization_sha256=base.json_hash(initialization),
                    diagnostic_schema=DIAGNOSTIC_SCHEMA, tiid_report_enabled=True,
                    trainable_scope='all_registered_parameters_trainable_NO_D_skips_D_no_detach', gate_target='D_logit_only',
                    future_bias_labels_used_as_inputs=False)
    return identity


def paired_report(root, arms):
    available = [arm for arm in arms if (root/arm/'summary.json').is_file()]
    missing = [arm for arm in arms if arm not in available]
    summaries = {arm: legacy.read_json(root/arm/'summary.json') for arm in available}
    history = {arm: legacy.read_json(root/arm/'history.json') for arm in available}
    common = min(s['completed_epoch'] for s in summaries.values()) if summaries and not missing else 0
    comparisons = {}
    if common:
        for candidate, control in (('B3R_HALF', 'B3R'), ('B3R_GATE', 'B3R_HALF'), ('B3R_GATE', 'B3R'),
                                   ('B3R', 'B3R_NO_D'), ('B3R_HALF', 'B3R_NO_D'), ('B3R_GATE', 'B3R_NO_D')):
            if candidate not in available or control not in available:
                continue
            comparisons[candidate+'_vs_'+control] = dict(
                common_epoch_mae_gain=history[control][common-1]['validation_mae_macro']-history[candidate][common-1]['validation_mae_macro'],
                common_prefix_best_mae_gain=min(r['validation_mae_macro'] for r in history[control][:common])
                                           -min(r['validation_mae_macro'] for r in history[candidate][:common]),
                common_last_five_mean_gain=sum(history[control][i]['validation_mae_macro']-history[candidate][i]['validation_mae_macro']
                                              for i in range(max(0, common-5), common))/min(5, common),
                candidate_common_last_five=[r['validation_mae_macro'] for r in history[candidate][max(0, common-5):common]],
                control_common_last_five=[r['validation_mae_macro'] for r in history[control][max(0, common-5):common]])
    return dict(status='B3RSTRENGTH_PAIRED_STAGE_COMPLETE' if not missing else 'B3RSTRENGTH_PARTIAL_STAGE_COMPLETE',
                common_epoch=common, completed_epochs={arm: s['completed_epoch'] for arm, s in summaries.items()},
                missing_arms=missing, comparisons=comparisons, summaries=summaries,
                scope='engineering_only' if legacy.read_json(root/'identity.json')['engineering_check'] else 'exploratory_validation',
                attribution='identity_centered_D_parameterization_and_independent_no_D_with_original_TIID_on',
                predictive_gain_claim=False, calibrated_reliability_claim=False, test_read=False)


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
                'Use a fresh G1.1 directory; old B3-R/G1.0 runs cannot be resumed here')
        legacy.write_json(path, identity)
        legacy.write_json(args.run_dir/'initialization.json', initialization)


def repair_sidecars(args, identity, protocol, device):
    """Recover JSON companions after checkpoint commit, without inventing epochs."""
    arm = args.worker
    root = args.run_dir/arm
    path = root/'last.pt'
    if not path.exists():
        return False
    saved = torch.load(path, map_location='cpu', weights_only=False)
    require(saved.get('identity') == identity and saved.get('arm') == arm, 'Checkpoint identity/arm mismatch before sidecar repair')
    history = saved.get('history')
    require(isinstance(history, list) and len(history) == saved.get('completed_epoch') and bool(history)
            and [row['epoch'] for row in history] == list(range(1, len(history)+1)), 'Invalid checkpoint history')
    require(history[-1]['best_metric'] == saved['best_metric'] and history[-1]['global_updates'] == saved['global_updates'],
            'Checkpoint history/state counters disagree')
    state, best_state = saved.get('model_state'), saved.get('best_model_state')
    require(isinstance(state, dict) and bool(state) and isinstance(best_state, dict) and state.keys() == best_state.keys(),
            'Invalid checkpoint model state')
    for name, value in state.items():
        best_value = best_state[name]
        require(isinstance(value, torch.Tensor) and isinstance(best_value, torch.Tensor)
                and value.shape == best_value.shape and value.dtype == best_value.dtype
                and bool(torch.isfinite(value).all()) and bool(torch.isfinite(best_value).all()),
                'Invalid checkpoint model tensor: '+name)
    initial_path = args.run_dir/'initialization.json'
    if initial_path.exists():
        specification = legacy.read_json(initial_path)['arms'][arm]['specification']
        require(all(name in state and list(state[name].shape) == shape for name, shape in specification.items()),
                'Checkpoint model does not match initialization parameter shapes')
    # The initialization certificate includes the complete parameter manifest.
    # No forward pass or optimizer update is needed to repair companion JSON.
    def read_companion(path):
        if not path.exists():
            return None
        try:
            return legacy.read_json(path)
        except (json.JSONDecodeError, UnicodeDecodeError):
            # Preserve the exact interrupted bytes before rebuilding from the
            # already validated authoritative checkpoint.
            backup = path.with_name(path.name+'.interrupted_'+uuid.uuid4().hex[:6])
            backup.write_bytes(path.read_bytes())
            return None
    summary_path = root/'summary.json'
    summary = read_companion(summary_path)
    summary_missing = summary is None
    summary = {} if summary_missing else summary
    require(isinstance(summary, dict), 'Invalid companion summary structure')
    summary.update(status='ENGINEERING_CHECK_PASS' if args.check else 'PAUSED_AT_EPOCH_BOUNDARY',
                   arm=arm, mode=ARMS[arm], completed_epoch=len(history), best_epoch=history[-1]['best_epoch'],
                   best_metric=saved['best_metric'], global_updates=saved['global_updates'],
                   early_stopped=saved['wait'] >= protocol['patience'], checkpoint='last.pt',
                   best_model_state_key='best_model_state', test_read=False, predictive_gain_claim=False)
    if summary_missing:
        if initial_path.exists():
            parameters = dict(legacy.read_json(initial_path)['arms'][arm])
            parameters.pop('initial_parameters_sha256', None)
            parameters.pop('common_initial_sha256', None)
            parameters['optimizer_parameters'] = parameters['total_parameters']
            summary['parameters'] = parameters
        else:
            summary['parameters_status'] = 'initialization_manifest_unavailable_in_fixture'
    history_path = root/'history.json'
    previous = read_companion(history_path)
    if previous is not None:
        require(isinstance(previous, list) and len(previous) <= len(history) and previous == history[:len(previous)],
                'JSON history conflicts with authoritative checkpoint; refusing overwrite')
    for target, value in ((history_path, history), (summary_path, summary)):
        temporary = target.with_suffix(target.suffix+'.repair.partial')
        legacy.write_json(temporary, value)
        temporary.replace(target)
    return True


def launch(args, device, initialization, audit):
    session = args.run_dir/('session_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6])
    session.mkdir()
    environment = legacy.environment(device)
    legacy.write_json(session/'environment.json', environment)
    initial_path = args.run_dir/'initial_environment.json'
    if initial_path.exists():
        original = legacy.read_json(initial_path)
        changes = {key: dict(saved=original.get(key), current=value) for key, value in environment.items() if original.get(key) != value}
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
    print('Initialization: '+initialization['status']+'; time scope: '+audit['status'], flush=True)
    for arm in args.only:
        command = [sys.executable, '-X', 'utf8', '-u', str(Path(__file__).resolve()), '--worker', arm,
                   '--run-dir', str(args.run_dir), '--data-dir', str(args.data_dir), '--graph-dir', str(args.graph_dir),
                   '--protocol', str(args.protocol), '--epochs', str(args.epochs), '--seed', str(args.seed),
                   '--device', args.device, '--threads', str(args.threads), '--arms', *args.arms]
        if args.check:
            command.append('--check')
        env = dict(os.environ, B3R_STRENGTH_INTERNAL_WORKER=str(args.run_dir)+'|'+arm)
        print('Starting '+arm, flush=True)
        with (session/(arm+'.log')).open('w', encoding='utf-8') as log:
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
        legacy.write_json(session/(arm+'_process_result.json'), dict(arm=arm, exit_code=code))
        require(code == 0, f'{arm} process failed ({code}); checkpoints preserved; see {session/(arm+".log")}')
    report = paired_report(args.run_dir, args.arms)
    legacy.write_json(session/'comparison.json', report)
    legacy.write_json(args.run_dir/'latest_comparison.json', report)
    print(json.dumps(dict(status=report['status'], common_epoch=report['common_epoch'], completed_epochs=report['completed_epochs'],
                         comparisons=report['comparisons'], output=str(session/'comparison.json')), indent=2), flush=True)


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    require(device.type in ('cpu', 'cuda') and (device.type != 'cuda' or torch.cuda.is_available()), 'Requested device unavailable')
    if args.worker:
        require(os.environ.get('B3R_STRENGTH_INTERNAL_WORKER') == str(args.run_dir)+'|'+args.worker,
                'Use --only through the launcher; direct internal workers bypass its lock')
        protocol, train, val, graph, roads, initialization, identity = prepare(args, device)
        verify_identity(args, identity, initialization, False)
        repair_sidecars(args, identity, protocol, device)
        training_loop(args.arms).worker(args, identity, protocol, train, val, graph, roads)
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
