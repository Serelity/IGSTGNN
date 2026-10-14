"""Train matched location/state service-limit adapters on one immutable MIN.

The established frozen pilot owns the replay, optimization and atomic checkpoint
implementation. This entry point adds a distinct experiment identity, a paired
initialization gate and sequential process isolation; it never modifies legacy
modules or resumes their old experiments.
"""
import argparse
from contextlib import ExitStack
from datetime import datetime
import gc
import hashlib
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import traceback
import uuid

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.chronological import train_min_frozen_budget as frozen
from experiments.chronological import train_min_event_budget as original
from experiments.chronological import probe_min_event_budget as probe
from experiments.chronological import train_internal_spatial as legacy
from experiments.chronological.min_service_budget_metrics import write_diagnostics
from experiments.chronological.smoke import set_seed, sha256
from src.models.incident_spatial import require
from src.models.incident_spatial_residual import tensor_state_hash
from src.models.incident_spatial_service_budget import build_frozen_service
from src.utils.igstgnn_published_events import EventPublishedData

HERE = Path(__file__).resolve().parent
RECIPE = HERE/'min_service_budget_v1.json'
SCHEMA = 'frozen_MIN_service_limit_pair_v1'
ARMS = ('location', 'state')
WORKER_ENV = 'IGSTGNN_MIN_SERVICE_LIMIT_CHILD'


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run', type=Path, required=True)
    p.add_argument('--run-dir', type=Path)
    p.add_argument('--data-dir', type=Path, default=original.paper.default_data())
    p.add_argument('--graph-dir', type=Path, default=HERE/'internal_spatial_graph_is11')
    p.add_argument('--source-protocol', type=Path, default=original.PROTOCOL)
    p.add_argument('--recipe', type=Path, default=RECIPE)
    p.add_argument('--arms', choices=ARMS, nargs='+', default=list(ARMS))
    p.add_argument('--worker', choices=ARMS, help='Resume only one prepared arm; target epoch remains absolute')
    p.add_argument('--epochs', type=int, default=1, help='Absolute completed adapter epoch, 1..10; default 1')
    p.add_argument('--seed', type=int, default=2025)
    p.add_argument('--device', help='Source device type by default; GPU name is audit-only')
    p.add_argument('--check', action='store_true', help='Requires an engineering source; formal data are never subset')
    args = p.parse_args(argv)
    require(len(set(args.arms)) == len(args.arms), 'Duplicate arms are not permitted')
    require(args.worker is None or args.worker in args.arms, 'Worker must belong to the recorded arm set')
    args.arms = [arm for arm in ARMS if arm in args.arms]
    require(1 <= args.epochs <= 10 and 0 <= args.seed < 2**32, 'Expected 1..10 epochs and a valid seed')
    for key in ('source_run', 'data_dir', 'graph_dir', 'source_protocol', 'recipe'):
        setattr(args, key, getattr(args, key).resolve())
    if args.run_dir is not None:
        args.run_dir = args.run_dir.resolve()
    return args


def experiment_sources():
    result = frozen.experiment_sources()
    names = ('experiments/chronological/train_min_service_budget.py',
             'experiments/chronological/min_service_budget_metrics.py',
             'experiments/chronological/run_min_service_budget.sh',
             'src/models/incident_spatial_service_budget.py')
    result.update({name: sha256(REPO/name) for name in names})
    return result


def source_paths(args, identity):
    paths = frozen.source_paths(args, identity)
    paths.update({'experiment_code/'+name: REPO/name for name in experiment_sources()})
    require(all(path.is_file() for path in paths.values()), 'A protected service experiment input is missing')
    return paths


def require_independent_directory(args):
    root, source = args.run_dir.resolve(), args.source_run.resolve()
    require(root != source and source not in root.parents and root not in source.parents,
            'Use an independent directory outside the source run')
    # An empty subdirectory of an old run is still not an independent experiment.
    for ancestor in root.parents:
        candidate = ancestor/'identity.json'
        require(not candidate.is_file(), 'Output is inside an existing experiment; use an independent directory')
    if (root/'identity.json').is_file():
        require(legacy.read_json(root/'identity.json').get('schema') == SCHEMA,
                'Use a fresh service experiment; old frozen runs cannot be resumed')


def load_source(args, logger):
    identity = legacy.read_json(args.source_run/'identity.json')
    frozen.verify_source(args, identity)
    saved = torch.load(args.source_run/'A/last.pt', map_location='cpu', weights_only=False)
    original.validate_saved(saved, identity, 'A')
    probe.verify_saved_sidecars(args.source_run/'A', saved)
    data = EventPublishedData(args.data_dir, identity['protocol'], logger, check=args.check)
    require(data.audit == legacy.read_json(args.source_run/'data_audit.json'), 'Source data audit differs')
    graph, _, audit = legacy.load_graph(args.graph_dir, data.station_ids, identity['protocol']['graph_sha256'])
    require(audit == identity['graph'], 'Source graph audit differs')
    base_args = argparse.Namespace(data_dir=args.data_dir, seed=identity['seed'])
    base, _ = original.build_arm(base_args, 'A', len(data.station_ids), graph, torch.device(args.device))
    base.load_state_dict(saved['best_model_state'], strict=True)
    base.eval()
    return identity, saved, data, base


def baseline_digest(baseline):
    digest = hashlib.sha256()
    for key in ('prediction', 'target', 'valid', 'sample_indices'):
        value = np.ascontiguousarray(baseline[key])
        digest.update(key.encode())
        digest.update(str(value.shape).encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def build_service(base, arm, recipe):
    # The state design has fewer active inputs. Its integer hidden width is
    # chosen by the model builder to approach the location parameter budget.
    return build_frozen_service(base, arm,
                                hidden_width=recipe['adapter_hidden_width'] if arm == 'location' else None)


def preflight(args, source_identity, saved, data, base, recipe):
    """Verify both designs before permitting any optimizer in either arm."""
    baseline = frozen.source_baseline(base, data, saved)
    hard = frozen.select_hard_windows(baseline['prediction'], baseline['target'], baseline['valid'],
                                      baseline['sample_indices'], fraction=recipe['fixed_hard_fraction'])
    certificates = {}
    for arm in ARMS:
        set_seed(args.seed)
        model, certificate = build_service(base, arm, recipe)
        report, prediction = frozen.evaluate_pair(model, data, baseline, certificate['frozen_state_sha256'], hard)
        require(np.array_equal(prediction, baseline['prediction']), 'Initial service ON differs from frozen MIN')
        require(report['off_exactly_matches_source'] and report['on_exactly_restored'], 'Initial replay certificate failed')
        certificates[arm] = certificate
        del model
    require(len({c['frozen_state_sha256'] for c in certificates.values()}) == 1,
            'Paired arms retained different MIN states')
    require(certificates['location']['frozen_state_sha256'] == tensor_state_hash(base.state_dict()),
            'Paired copies changed the source backbone')
    return dict(status='MIN_SERVICE_LIMIT_INITIALIZATION_PAIR_PASS', arms=certificates,
                source_best_epoch=probe.best_row(saved)['epoch'], baseline_sha256=baseline_digest(baseline),
                validation_rows=len(baseline['prediction']), batch_partition=baseline['batch_partition'],
                fixed_hard_sample_indices=baseline['sample_indices'][hard].tolist(),
                all_initial_ON_OFF_exactly_match_MIN=True, shared_adapter_initialization=False,
                parameter_comparison='active_inputs_near_parameter_budget_not_identical_architecture',
                trainable_parameters={arm: certificates[arm]['trainable_parameters'] for arm in ARMS},
                original_source_metrics_replay_passed=True, test_evaluated=False)


def make_identity(args, source_identity, saved, initialization, recipe, paths):
    return dict(schema=SCHEMA, source_run=str(args.source_run), arms=args.arms,
        source_artifacts_sha256=probe.snapshot_files(paths), source_identity=source_identity,
        source_best_epoch=probe.best_row(saved)['epoch'],
        source_best_model_sha256=tensor_state_hash(saved['best_model_state']),
        initialization=initialization, recipe=recipe, recipe_sha256=sha256(args.recipe),
        experiment_sources_sha256=experiment_sources(), seed=args.seed,
        batch_size=source_identity['batch_size'], threads=source_identity['threads'],
        runtime=source_identity['runtime'], engineering_check=source_identity['engineering_check'],
        checkpoint_schema_reused=frozen.SCHEMA,
        checkpoint_compatibility='legacy_storage_helpers_only_new_identity_required',
        event_scope='published_archive_attributes_conditional_offline',
        capacity_scope='dimensionless_latent_service_limit_not_vehicle_capacity',
        proposal_scope='unchanged_baseline_C0', service_scope='adapter_changes_only_final_exchange_limit',
        test_payload_deserialized=True, test_evaluated=False)


def arm_identity(identity, arm):
    require(arm in identity['arms'], 'Arm is absent from the prepared experiment')
    certificate = identity['initialization']['arms'][arm]
    return dict(identity, arm=arm, service_mode=arm, adapter_certificate=certificate,
        frozen_state_sha256=certificate['frozen_state_sha256'],
        adapter_initial_sha256=certificate['adapter_initial_sha256'])


def verify_identity(root, identity, allow_new):
    path = root/'identity.json'
    if path.is_file():
        require(legacy.read_json(path) == identity, 'Service experiment identity changed; checkpoints preserved')
        require(legacy.read_json(root/'initialization.json') == identity['initialization'],
                'Service initialization sidecar changed')
    else:
        require(allow_new, 'Worker requires a completed parent initialization')
        require({p.name for p in root.iterdir()} <= {'active.lock'}, 'Use a fresh independent service run')
        legacy.write_json(root/'identity.json', identity)
        legacy.write_json(root/'initialization.json', identity['initialization'])


def child_command(args, arm):
    command = [sys.executable, '-u', str(Path(__file__).resolve()), '--source-run', str(args.source_run),
        '--run-dir', str(args.run_dir), '--data-dir', str(args.data_dir), '--graph-dir', str(args.graph_dir),
        '--source-protocol', str(args.source_protocol), '--recipe', str(args.recipe),
        '--epochs', str(args.epochs), '--seed', str(args.seed), '--device', args.device,
        '--arms', *args.arms, '--worker', arm]
    if args.check:
        command.append('--check')
    return command


def run_worker(args, arm, session):
    """Keep arm locks alive until a surviving child has terminated."""
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


def train_worker(args, identity, logger, recipe):
    arm = args.worker
    root = args.run_dir/arm
    root.mkdir(exist_ok=True)
    with legacy.run_lock(root):
        source_identity, saved, data, base = load_source(args, logger)
        paths = source_paths(args, source_identity)
        require(probe.snapshot_files(paths) == identity['source_artifacts_sha256'], 'Protected input changed before worker')
        set_seed(args.seed)
        model, certificate = build_service(base, arm, recipe)
        require(certificate == identity['initialization']['arms'][arm], 'Worker initialization differs from paired preflight')
        current_identity = arm_identity(identity, arm)
        frozen.verify_output_identity(root, current_identity)
        # Reuse safe fit without changing its module globals or old source hashes.
        summary = frozen.fit(model, base, data, saved, current_identity, root, args.epochs,
                             recipe, protected_paths=paths)
        summary.update(schema=SCHEMA, service_mode=arm,
            status='MIN_SERVICE_LIMIT_ENGINEERING_PASS' if args.check else
                   'MIN_SERVICE_LIMIT_PILOT_COMPLETE' if summary['completed_epoch'] == recipe['max_epochs'] else
                   'MIN_SERVICE_LIMIT_PAUSED_AT_EPOCH_BOUNDARY',
            proposal_scope='unchanged_baseline_C0',
            service_scope='adapter_changes_only_final_exchange_limit',
            relation_features_used=arm == 'location', original_support_used=True,
            best_predictions_sha256=sha256(root/'best_predictions.npz'),
            baseline_predictions_sha256=sha256(root/'baseline_predictions.npz'),
            interpretation='location_vs_same_support_state_control_not_real_capacity_or_causal_effect')
        legacy.write_json(root/'summary.json', summary)
        report = root/'report.md'
        title = '冻结 MIN：'+('关系条件' if arm == 'location' else '仅状态、相同支持范围')+'服务上界适配器'
        text = report.read_text(encoding='utf-8')
        text = text.replace('# 冻结 MIN：事故位置预算分支验证', '# '+title, 1)
        text = text.replace('固定 MIN 最佳检查点，只训练新增位置预算适配器。',
            '固定 MIN 最佳检查点，只训练服务上界适配器；交换提议继续使用基线 C0。'+
            ('本组读取事故关系细节。' if arm == 'location' else '本组不读取关系特征，保留相同事故支持范围并采用接近的有效参数预算。'), 1)
        report.write_text(text, encoding='utf-8')
        print(json.dumps({key: summary[key] for key in ('status', 'service_mode', 'completed_epoch',
            'best_adapter_epoch', 'best_mae', 'baseline_minus_best_mae', 'trainable_parameters',
            'optimizer_updates_this_invocation', 'frozen_backbone_preserved', 'test_evaluated')},
            allow_nan=False), flush=True)
        print('Saved arm summary: '+str(root/'summary.json'), flush=True)
        return summary


def comparison(args, identity):
    histories, summaries = {}, {}
    for arm in args.arms:
        root = args.run_dir/arm
        if not (root/'last.pt').is_file():
            continue
        saved = torch.load(root/'last.pt', map_location='cpu', weights_only=False)
        require(saved.get('schema') == frozen.SCHEMA and saved.get('identity') == arm_identity(identity, arm),
                'Service checkpoint identity changed during comparison')
        history = saved['history']
        require([row['epoch'] for row in history] == list(range(saved['completed_epoch']+1)),
                'Incomplete committed service history')
        histories[arm] = history
        summaries[arm] = dict(completed_epoch=saved['completed_epoch'], best_epoch=saved['best_epoch'],
                              best_mae=saved['best_metric'])
    common = min((len(rows)-1 for rows in histories.values()), default=0) if len(histories) == len(args.arms) else None
    contrasts = {}
    if common is not None and {'location', 'state'} <= histories.keys():
        location, state = histories['location'][:common+1], histories['state'][:common+1]
        require(all(location[e]['training']['train_order_sha256'] == state[e]['training']['train_order_sha256']
                    for e in range(1, common+1)), 'Paired service training sample orders differ')
        get_mae = lambda row: row['validation']['on']['mae_pooled']
        contrasts['location_vs_state'] = dict(
            common_epoch_mae_gain=get_mae(state[-1])-get_mae(location[-1]),
            common_prefix_best_mae_gain=min(map(get_mae, state))-min(map(get_mae, location)),
            candidate_common_last_five=[get_mae(row) for row in location[-5:]],
            control_common_last_five=[get_mae(row) for row in state[-5:]])
    complete = len(histories) == len(args.arms) and all(s['completed_epoch'] >= args.epochs for s in summaries.values())
    return dict(schema=SCHEMA, status='MIN_SERVICE_LIMIT_PAIRED_STAGE_COMPLETE' if complete else
                'MIN_SERVICE_LIMIT_PAIRED_STAGE_PARTIAL', requested_epoch=args.epochs,
                common_epoch=common, runs=summaries, comparisons=contrasts,
                interpretation='independently_trained_conditional_offline_adapters_shared_frozen_MIN',
                predictive_gain_claim=False, test_evaluated=False)


def main(argv=None):
    args = parse_args(argv)
    require(args.source_run.is_dir() and (args.source_run/'A').is_dir(), 'Source MIN-I A run does not exist')
    source_identity = legacy.read_json(args.source_run/'identity.json')
    require(source_identity.get('schema') == original.SCHEMA, 'Expected the original MIN-I source run')
    require(not args.check or source_identity['engineering_check'], '--check requires an engineering source')
    args.check = bool(source_identity['engineering_check'])
    args.threads = source_identity['threads']
    args.device = args.device or ('cuda:0' if source_identity['runtime']['device_type'] == 'cuda' else 'cpu')
    if args.run_dir is None:
        args.run_dir = REPO/'experiments/chronological_runs'/('contra_min_service_v1_'+
            ('check_' if args.check else '')+f'bs{source_identity["batch_size"]}_seed{args.seed}')
    require_independent_directory(args)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    require(device.type in ('cpu', 'cuda') and (device.type != 'cuda' or torch.cuda.is_available()), 'Requested device unavailable')
    probe.verify_runtime(source_identity, device)
    recipe = legacy.read_json(args.recipe)
    require(recipe == legacy.read_json(RECIPE) and recipe['max_epochs'] == 10
            and recipe['supervised_horizons'] == 12 and not recipe['curriculum'], 'Service recipe changed')
    logger = logging.getLogger('min_service_budget')
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler(sys.stdout))
    args.run_dir.mkdir(parents=True, exist_ok=True)
    internal_worker = args.worker and os.environ.get(WORKER_ENV) == '1'
    with ExitStack() as locks:
        if not internal_worker:
            for directory in (args.source_run, args.source_run/'A', args.run_dir):
                locks.enter_context(legacy.run_lock(directory))
        if args.worker:
            require((args.run_dir/'identity.json').is_file(), 'Worker requires prepared parent identity')
            identity = legacy.read_json(args.run_dir/'identity.json')
            require(identity['schema'] == SCHEMA and identity['arms'] == args.arms and identity['seed'] == args.seed
                    and identity['source_run'] == str(args.source_run) and identity['recipe'] == recipe,
                    'Worker request differs from prepared service experiment')
            train_worker(args, identity, logger, recipe)
            if not internal_worker:
                report = comparison(args, identity)
                legacy.write_json(args.run_dir/'comparison.json', report)
                legacy.write_json(args.run_dir/'latest_stage.json', report)
                write_diagnostics(args.run_dir, arms=args.arms)
                print(json.dumps(report, indent=2, allow_nan=False), flush=True)
            return
        # An orphaned child still owns its arm lock after the parent is killed.
        # Detect it before loading another full package or allocating a model.
        for arm in args.arms:
            arm_root = args.run_dir/arm
            if arm_root.is_dir():
                with legacy.run_lock(arm_root):
                    pass
        source_identity, saved, data, base = load_source(args, logger)
        paths = source_paths(args, source_identity)
        before = probe.snapshot_files(paths)
        initialization = preflight(args, source_identity, saved, data, base, recipe)
        identity = make_identity(args, source_identity, saved, initialization, recipe, paths)
        require(identity['source_artifacts_sha256'] == before, 'Protected input changed during preflight')
        verify_identity(args.run_dir, identity, allow_new=True)
        legacy.write_json(args.run_dir/'data_audit.json', data.audit)
        session = args.run_dir/('session_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6])
        session.mkdir()
        legacy.write_json(session/'environment.json', legacy.environment(device))
        legacy.write_json(session/'request.json', dict(source_run=str(args.source_run), arms=args.arms,
            epochs=args.epochs, batch_size=source_identity['batch_size'], test_evaluated=False))
        print('Run directory: '+str(args.run_dir), flush=True)
        print('Session directory: '+str(session), flush=True)
        print('Initialization: '+initialization['status'], flush=True)
        del data, base, saved
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        results = {}
        try:
            for arm in args.arms:
                print('Starting '+arm+' service-limit adapter', flush=True)
                code = run_worker(args, arm, session)
                results[arm] = dict(exit_code=code)
                legacy.write_json(session/'process_results.json', results)
                require(code == 0, 'Worker failed; checkpoints preserved. See '+str(session/(arm+'.log')))
        except BaseException as error:
            legacy.write_json(session/'failure.json', dict(error_type=type(error).__name__, error=str(error),
                                                         traceback=traceback.format_exc()))
            # Preserve the original training failure even if a killed child
            # left incomplete derived sidecars. Committed last.pt stays valid.
            try:
                probe.require_preserved(before, paths)
                report = comparison(args, identity)
                legacy.write_json(session/'comparison.json', report)
                legacy.write_json(args.run_dir/'comparison.json', report)
                legacy.write_json(args.run_dir/'latest_stage.json', report)
                write_diagnostics(args.run_dir, arms=args.arms)
            except Exception as diagnostic_error:
                legacy.write_json(session/'partial_diagnostics_error.json',
                    dict(error_type=type(diagnostic_error).__name__, error=str(diagnostic_error)))
            raise
        probe.require_preserved(before, paths)
        report = comparison(args, identity)
        legacy.write_json(session/'comparison.json', report)
        legacy.write_json(args.run_dir/'comparison.json', report)
        legacy.write_json(args.run_dir/'latest_stage.json', report)
        write_diagnostics(args.run_dir, arms=args.arms)
        print(json.dumps(dict(report, output=str(session/'comparison.json')), indent=2, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
