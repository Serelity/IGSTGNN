"""Audit and continue the existing six M4.2 arms without changing their identity."""
import argparse
from datetime import datetime
import hashlib
import os
from pathlib import Path
import platform
import subprocess
import sys
import uuid

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.chronological import train_incident_capacity as training
from experiments.chronological.prepare_incident_corridors import discover_history
from experiments.chronological.train import epoch_plan
from src.utils.incident_corridor import read_json, read_rows, require, sha256, write_json

PROTOCOL = REPO/'experiments/chronological/incident_capacity_training_m42.json'
PAIRS = (('P1', 'F'), ('P1', 'N1'), ('P1', 'P0'), ('P1', 'L1'), ('N1', 'N0'))
INPUT_FILES = {
    'data': ('summary.json', 'context_manifest.json', 'station_ids.npy', 'train_manifest.csv',
             'train_context.npz', 'scaler.json', 'adjacency.npy', 'train_flow.npy',
             'val_manifest.csv', 'val_flow.npy', 'val_context.npz'),
    'history': ('summary.json', 'train_history.npy', 'train_multichannel_scaler.json', 'val_history.npy'),
    'metadata': ('manifest.json', 'source_sensor_subset.tsv'),
    'network': ('summary.json',), 'reports': ('manifest.json', 'locations.tsv'),
}


def discover_run(runs_dir):
    candidates = []
    for path in sorted(Path(runs_dir).glob('contra_training_m42_*/identity.json')):
        identity = read_json(path)
        if identity.get('schema') == 'capacity_training_m42_v1' and identity.get('check') is False:
            candidates.append(path.parent.resolve())
    require(len(candidates) == 1, 'Expected one full M4.2 run; use --run-dir explicitly. Candidates: '
            + ', '.join(map(str, candidates)))
    return candidates[0]


def array_digest(value):
    value = np.ascontiguousarray(value)
    h = hashlib.sha256(str((value.dtype.str, value.shape)).encode())
    h.update(value.tobytes())
    return h.hexdigest()


def verify_sources(identity):
    require(identity['schema'] == 'capacity_training_m42_v1', 'Unsupported run identity')
    require(set(identity['source_sha256']) == set(training.SOURCE_FILES), 'Source file set changed')
    for name, expected in identity['source_sha256'].items():
        require(sha256(REPO/name) == expected, 'Frozen training source changed: '+name)
    require(sha256(PROTOCOL) == identity['protocol_sha256'], 'Frozen training protocol changed')


def verify_inputs(identity, directories, sensors):
    expected_keys = {'sensors'} | {group+'/'+name for group, names in INPUT_FILES.items() for name in names}
    require(set(identity['inputs_sha256']) == expected_keys, 'Unexpected input file set')
    for key, expected in identity['inputs_sha256'].items():
        if key == 'sensors':
            path = sensors
        else:
            group, name = key.split('/')
            path = directories[group]/name
        require(path.is_file() and sha256(path) == expected, 'Input checksum mismatch: '+str(path))
    pack = directories['network']
    for name, expected in read_json(pack/'summary.json')['outputs_sha256'].items():
        require(Path(name).name == name, 'Unexpected network filename')
        require(sha256(pack/name) == expected, 'Network pack changed: '+name)


def regions_from_pack(pack, identity, station_ids):
    graph = read_json(pack/('graph.json' if identity['ramp_exchanges'] else 'conservative_graph.json'))
    require(np.array_equal(graph['station_ids'], station_ids), 'Graph station axis changed')
    candidate = np.asarray(graph['operator_mask'], bool)
    common = np.load(pack/'common_structure_mask.npy', allow_pickle=False)
    boundary = np.asarray(graph['boundary_nodes'], bool)
    require(candidate.shape == common.shape == boundary.shape == station_ids.shape
            and not (common & ~candidate).any(), 'Invalid graph masks')
    require(int(candidate.sum()) == identity['candidate_structure_nodes'], 'Candidate coverage changed')
    regions = dict(all_nodes=np.ones_like(candidate), common_structure=common,
                   added_structure=candidate & ~common, candidate_structure=candidate,
                   outside_candidate=~candidate, candidate_boundary=boundary)
    rows = read_rows(pack/'stations.csv')
    require([int(r['station_id']) for r in rows] == station_ids.tolist(), 'Station metadata order changed')
    for i, row in enumerate(rows):
        regions.setdefault('road_'+row['road'], np.zeros_like(candidate))[i] = True
    return regions


def error_metrics(error, count, mask):
    """Same horizon-macro pooled MAE as the trainer; empty horizons are unavailable."""
    e, c = error[:, mask].sum(1), count[:, mask].sum(1)
    if not (c > 0).all():
        return None
    available = count[:, mask] > 0
    return dict(mae_macro=float((e/c).mean()), per_horizon_mae=(e/c).tolist(),
                valid_count=int(c.sum()), per_horizon_count=c.tolist(),
                absolute_error_sum=float(e.sum()), per_horizon_absolute_error_sum=e.tolist(),
                station_horizon_macro_mae=float((error[:, mask][available]/count[:, mask][available]).mean()))


def epoch_diagnostic(row, protocol, scaler_std, train_samples, batch_size):
    plan_sizes = [min(batch_size, train_samples-start) for start in range(0, train_samples, batch_size)]
    diagnostics = row['diagnostics']
    require(len(diagnostics) == len(plan_sizes), 'Diagnostic batch count changed')
    result = dict(epoch=row['epoch'], train_mae_macro=row['train_mae_macro'],
                  validation_mae_macro=row['validation_mae_macro'], seconds=row['seconds'],
                  losses_batch_mean=row['losses'], branch_gradient_max_l1=row['branch_gradient_max_l1'],
                  main_loss_gradient_scope='max_over_first_two_batches_only',
                  total_loss_gradient_scope='max_over_all_batches_before_clipping')
    terms = row['losses']
    if 'auxiliary' in terms:
        extra = dict(future_auxiliary=scaler_std*protocol['lambda_obs']*terms['auxiliary']/2,
                     prefix_auxiliary=scaler_std*protocol['lambda_obs']*terms['prefix']/2,
                     curve=scaler_std*protocol['lambda_curve']*terms['curve'])
        result['weighted_extra_loss_raw_units_batch_mean'] = extra
        result['extra_to_main_loss_ratio'] = sum(extra.values())/terms['main'] if terms['main'] else None
    for key in ('capacity_limited_fraction', 'receiving_limited_fraction', 'sending_bid_limited_fraction'):
        values = [r.get(key) for r in diagnostics]
        if all(v is not None for v in values):
            result[key] = float(np.average(values, weights=plan_sizes))
    result['limitation_fraction_denominator'] = 'all_recorded_edges_substeps_channels_sample_weighted; not_only_active_edges'
    for key, reduce in (('state_min', min), ('state_max', max),
                        ('forecast_delta_abs_max', max), ('balance_residual_abs_max', max)):
        values = [r[key] for r in diagnostics if key in r]
        if values:
            require(np.isfinite(values).all(), 'Nonfinite diagnostic: '+key)
            result[key] = reduce(values)
    return result


def audit_arm(directory, arm, identity, protocol, regions, expected_arrays, scaler_std):
    summary = read_json(directory/'summary.json')
    checkpoint = torch.load(directory/'last_checkpoint.pt', map_location='cpu', weights_only=False)
    require(checkpoint['identity'] == summary['identity'] == dict(identity, arm=arm), 'Arm identity changed: '+arm)
    require(summary['arm'] == arm and summary['test_accessed'] is False, 'Arm summary scope changed: '+arm)
    history, completed = checkpoint['history'], checkpoint['completed_epoch']
    require(completed > 0 and len(history) == completed, 'Incomplete checkpoint history: '+arm)
    require(0 < summary['completed_epoch'] <= completed
            and summary['history'] == history[:summary['completed_epoch']], 'Summary/history mismatch: '+arm)
    lagged = summary['completed_epoch'] < completed
    samples = min(4, protocol['train_samples']) if identity['check'] else protocol['train_samples']
    batches = (samples+identity['batch_size']-1)//identity['batch_size']
    best, best_epoch = float('inf'), 0
    for epoch, row in enumerate(history, 1):
        plan = epoch_plan(range(samples), identity['batch_size'], identity['seed'], epoch)
        require(row['epoch'] == epoch and row['global_updates'] == epoch*batches
                and row['train_order_sha256'] == plan['order_sha256'], 'Training order/update mismatch: '+arm)
        score = row['validation_mae_macro']
        require(np.isfinite(score), 'Nonfinite validation metric')
        if score < best:
            best, best_epoch = score, epoch
        require(row['best_metric'] == best, 'Best metric history mismatch: '+arm)
    summary_history = history[:summary['completed_epoch']]
    summary_best_epoch = min(range(len(summary_history)), key=lambda i: summary_history[i]['validation_mae_macro'])+1
    require(summary['global_updates'] == summary['completed_epoch']*batches
            and summary['best_epoch'] == summary_best_epoch
            and summary['best_metric'] == summary_history[summary_best_epoch-1]['validation_mae_macro'],
            'Summary counters do not match its own history: '+arm)
    require(checkpoint['global_updates'] == completed*batches and checkpoint['best_epoch'] == best_epoch
            and checkpoint['best_metric'] == best and checkpoint['wait'] == completed-best_epoch,
            'Checkpoint counters changed: '+arm)
    require(checkpoint['optimizer_state']['state'] and checkpoint['scheduler_state']['last_epoch'] == completed,
            'Optimizer/scheduler state missing: '+arm)
    if not lagged:
        require(all(summary[k] == checkpoint[k] for k in ('global_updates', 'best_epoch', 'best_metric')),
                'Summary/checkpoint counters differ: '+arm)
    best_model = torch.load(directory/'best_model.pt', map_location='cpu', weights_only=False)
    require(best_model['identity'] == checkpoint['identity'] and best_model['epoch'] == best_epoch,
            'Best model identity/epoch mismatch: '+arm)
    if best_epoch == completed:
        require(training.state_digest(best_model['model_state']) == training.state_digest(checkpoint['model_state']),
                'Best/last weights differ at the same epoch: '+arm)
    del best_model, checkpoint
    path = directory/'best_validation_predictions.npz'
    prediction_hash = sha256(path)
    if not lagged:
        require(prediction_hash == summary['prediction_sha256'], 'Prediction file checksum changed: '+arm)
    with np.load(path, allow_pickle=False) as stored:
        require(set(stored.files) == {'prediction', 'target', 'valid', 'sample_indices'}, 'Prediction schema changed')
        for key, expected in expected_arrays.items():
            require(array_digest(stored[key]) == expected, 'Validation target/mask/order mismatch: '+arm+'/'+key)
        prediction, target, valid = stored['prediction'], stored['target'], stored['valid']
        require(prediction.shape == target.shape == valid.shape and prediction.ndim == 4
                and prediction.shape[-1] == 1 and prediction.shape[1:3] == (12, len(regions['all_nodes']))
                and valid.dtype == bool and np.isfinite(prediction).all(), 'Prediction axes/values changed: '+arm)
        difference = np.abs(prediction.astype(np.float64)-target)
        error = np.where(valid, difference, 0).sum(0)[..., 0]
        count = valid.sum(0)[..., 0]
    metrics = {name: error_metrics(error, count, mask) for name, mask in regions.items()}
    saved_metrics = read_json(directory/'best_validation_metrics.json')
    require(saved_metrics == history[best_epoch-1]['validation_metrics'], 'Best metrics/history mismatch: '+arm)
    if not lagged:
        require(summary['best_validation_metrics'] == saved_metrics, 'Summary best metrics mismatch: '+arm)
    for name, metric in saved_metrics.items():
        actual = metrics[name]
        require((metric is None and actual is None) or (metric is not None and actual is not None
                and np.isclose(metric['mae_macro'], actual['mae_macro'], rtol=1e-10, atol=1e-9)
                and np.allclose(metric['per_horizon_mae'], actual['per_horizon_mae'], rtol=1e-10, atol=1e-9)),
                'Prediction/metric readback differs: '+arm+'/'+name)
    return dict(completed_epoch=completed, global_updates=completed*batches, best_epoch=best_epoch,
                best_metric=best, summary_lagged=lagged, prediction_sha256=prediction_hash,
                best_prediction_regions=metrics,
                best_station_horizon_absolute_error_sum=error.tolist(), best_station_horizon_count=count.tolist(),
                epochs=[epoch_diagnostic(row, protocol, scaler_std, samples, identity['batch_size']) for row in history],
                epoch_validation_metrics=[row['validation_metrics'] for row in history])


def comparisons(runs):
    best, same_epoch = {}, {}
    for candidate, control in PAIRS:
        a, b = runs[candidate], runs[control]
        name = candidate+'_vs_'+control
        best[name] = dict(candidate_best_epoch=a['best_epoch'], control_best_epoch=b['best_epoch'],
                          regions={key: None if value is None or b['best_prediction_regions'][key] is None else
                                   b['best_prediction_regions'][key]['mae_macro']-value['mae_macro']
                                   for key, value in a['best_prediction_regions'].items()})
        same_epoch[name] = [dict(epoch=i+1, regions={key: None if value is None or y[key] is None else
                                                   y[key]['mae_macro']-value['mae_macro'] for key, value in x.items()})
                           for i, (x, y) in enumerate(zip(a['epoch_validation_metrics'], b['epoch_validation_metrics']))]
    return dict(best_selected=best, same_epoch=same_epoch,
                difference_sign='control_MAE_minus_candidate_MAE_positive_is_candidate_better',
                outside_candidate_available_for='best_prediction_epoch_only; before_snapshot_preserves_previous_best')


def audit_run(root, identity, directories):
    protocol = read_json(PROTOCOL)
    station_ids = np.load(directories['data']/'station_ids.npy', allow_pickle=False)
    regions = regions_from_pack(directories['network'], identity, station_ids)
    require(len(station_ids) == protocol['stations'], 'Station count changed')
    require(identity['sample_scope'] == ('first_4_train_val' if identity['check'] else 'original_full_train_val')
            and identity['batch_size'] == (2 if identity['check'] else protocol['batch_size']), 'Training scope changed')
    rows = read_rows(directories['data']/'val_manifest.csv')
    require(len(rows) == protocol['val_samples'], 'Validation sample count changed')
    n = min(4, len(rows)) if identity['check'] else len(rows)
    raw = np.load(directories['data']/'val_flow.npy', mmap_mode='r', allow_pickle=False)
    try:
        target = np.asarray(raw[:n, 14:26]).copy()[..., None]
    finally:
        raw._mmap.close()
    valid = np.isfinite(target) & (target >= 0)
    expected = dict(target=array_digest(np.where(valid, target, 0)), valid=array_digest(valid),
                    sample_indices=array_digest(np.array([int(r['sample_index']) for r in rows[:n]])))
    del target, valid
    std = read_json(directories['data']/'scaler.json')['std']
    runs = {arm: audit_arm(root/arm, arm, identity, protocol, regions, expected, std) for arm in training.ARMS}
    return dict(status='M42_SIX_ARM_AUDIT_PASS', run_dir=str(root), identity=identity,
                station_ids=station_ids.tolist(), check=identity['check'], seed=identity['seed'],
                region_station_counts={k: int(v.sum()) for k, v in regions.items()},
                shared_validation_arrays_sha256=expected, runs=runs, comparisons=comparisons(runs),
                predictive_gain_claim=False, test_accessed=False, main_training_ready=False,
                online_semantics_certified=False, jointly_certified_nodes=0)


def trainer_command(root, identity, directories, sensors, target_epoch):
    command = [sys.executable, '-u', str(REPO/'experiments/chronological/train_incident_capacity.py'),
               '--data-dir', str(directories['data']), '--history-dir', str(directories['history']),
               '--sensors', str(sensors), '--output-dir', str(root), '--network-dir', str(directories['network']),
               '--report-bundle', str(directories['reports']), '--device', identity['device'],
               '--seed', str(identity['seed']), '--epochs', str(target_epoch), '--allow-exploratory', '--resume']
    if not identity['ramp_exchanges']:
        command.append('--without-ramp-exchanges')
    return command


def print_table(report):
    print('arm  last_epoch  best_epoch  best_MAE  last_MAE  outside_best_MAE', flush=True)
    for arm, run in report['runs'].items():
        outside = run['best_prediction_regions']['outside_candidate']
        print(f"{arm:4} {run['completed_epoch']:10} {run['best_epoch']:11} {run['best_metric']:9.5f} "
              f"{run['epochs'][-1]['validation_mae_macro']:9.5f} "
              + ('unavailable' if outside is None else f"{outside['mae_macro']:.5f}"), flush=True)
        if arm != 'F':
            last = run['epochs'][-1]
            print('diagnostic '+arm+': '+str({key: last[key] for key in (
                'capacity_limited_fraction', 'receiving_limited_fraction', 'sending_bid_limited_fraction',
                'state_min', 'state_max', 'balance_residual_abs_max', 'extra_to_main_loss_ratio') if key in last}), flush=True)
            print('capacity_output_gradients '+arm+': '+str({k: v for k, v in last['branch_gradient_max_l1'].items()
                                                           if 'coefficients.2.' in k}), flush=True)
    for name, comparison in report['comparisons']['best_selected'].items():
        print(name+' best_selected: '+str({k: comparison['regions'][k] for k in
              ('all_nodes', 'candidate_structure', 'outside_candidate')}), flush=True)
    for name, rows in report['comparisons']['same_epoch'].items():
        print(name+' same_epoch '+str(rows[-1]['epoch'])+': '+str({k: rows[-1]['regions'][k] for k in
              ('all_nodes', 'common_structure', 'added_structure', 'candidate_structure')}), flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path)
    p.add_argument('--runs-dir', type=Path, default=REPO/'experiments/chronological_runs')
    p.add_argument('--data-dir', type=Path, default=REPO.parent/'data/chronological/Contra_Costa_v8_dev')
    p.add_argument('--history-dir', type=Path)
    p.add_argument('--sensors', type=Path, default=REPO.parent/'data/xtraffic/Contra_Costa/sensors.csv')
    p.add_argument('--network-dir', type=Path)
    p.add_argument('--report-bundle', type=Path, default=Path(__file__).with_name('report_metadata_m42'))
    p.add_argument('--epochs', type=int, default=5, help='Absolute stopping epoch, not additional epochs')
    p.add_argument('--audit-only', action='store_true', help='No optimizer updates; also accepts an explicit check run')
    args = p.parse_args(argv)
    root = args.run_dir.resolve() if args.run_dir else discover_run(args.runs_dir)
    identity = read_json(root/'identity.json')
    verify_sources(identity)
    require(args.audit_only or identity['check'] is False, 'Subset check cannot be continued by this full-run entry')
    require(1 <= args.epochs <= read_json(PROTOCOL)['max_epochs'], 'Invalid target epoch')
    history = args.history_dir or discover_history(
        [args.data_dir, args.data_dir.parent, REPO/'experiments/chronological_runs', REPO.parent/'论文学习/研究开发_20260911'],
        identity['inputs_sha256']['data/summary.json'])
    require(history is not None, 'Original v11a history not found; set --history-dir')
    directories = {k: v.resolve() for k, v in dict(data=args.data_dir, history=history,
        network=args.network_dir or root/'network_pack', metadata=Path(__file__).with_name('physics_metadata'),
        reports=args.report_bundle).items()}
    sensors = args.sensors.resolve()
    verify_inputs(identity, directories, sensors)
    lock = root/'capacity_continuation.lock'
    # Exclusive creation prevents two instances of this entry from updating the same run.
    with lock.open('x', encoding='utf-8') as stream:
        stream.write(f'pid={os.getpid()} host={platform.node()}\n')
    try:
        session = root/('continuation_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6])
        session.mkdir()
        print('Run directory: '+str(root)+'\nSession directory: '+str(session), flush=True)
        before = audit_run(root, identity, directories)
        write_json(session/'before.json', before)
        print_table(before)
        if args.audit_only:
            print('Audit complete; no training updates. Report: '+str(session/'before.json'), flush=True)
            return
        require(all(r['completed_epoch'] <= args.epochs for r in before['runs'].values()),
                'A checkpoint is already beyond the requested stopping epoch')
        command = trainer_command(root, identity, directories, sensors, args.epochs)
        write_json(session/'invocation.json', dict(command=command, identity=identity, target_epoch=args.epochs,
                   wrapper_sha256=sha256(Path(__file__)), test_accessed=False))
        print('Continuing all six arms to absolute epoch '+str(args.epochs), flush=True)
        with (session/'training.log').open('w', encoding='utf-8') as log:
            with subprocess.Popen(command, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, encoding='utf-8', errors='replace') as process:
                for line in process.stdout:
                    print(line, end='', flush=True)
                    log.write(line)
                    log.flush()
                code = process.wait()
        write_json(session/'process_result.json', dict(exit_code=code))
        require(code == 0, 'Trainer failed; preserved checkpoints and before audit. See '+str(session/'training.log'))
        verify_sources(identity)
        verify_inputs(identity, directories, sensors)
        after = audit_run(root, identity, directories)
        require(all(r['completed_epoch'] == args.epochs for r in after['runs'].values()), 'Target epoch was not reached')
        write_json(session/'after.json', after)
        print_table(after)
        print('M42_CONTINUATION_AND_DIAGNOSTICS_PASS\nReport: '+str(session/'after.json'), flush=True)
    finally:
        lock.unlink()


if __name__ == '__main__':
    main()
