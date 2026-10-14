"""Analyze archived early-event validation predictions with NumPy only.

No model, checkpoint, raw release loader, accelerator, or new inference is used.
Recorded replay certificates are checked against saved arrays; this is not a
new checkpoint replay or an independently signed historical archive audit.
"""
import argparse
from contextlib import contextmanager, nullcontext
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import uuid

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.chronological.min_early_event_gain_metrics import analyze_predictions, flat_rows
from experiments.chronological.min_event_budget_probe_metrics import metric_result, tail_summary, window_scores

HERE = Path(__file__).resolve().parent
SCHEMA = 'min_early_event_saved_gain_analysis_v1'
SOURCE_SCHEMA = 'min_early_event_evaluation_v1'
SOURCE_STATUS = 'MIN_EARLY_EVENT_EVALUATION_PASS'
BASE_CASES = ('STATE_ON', 'EVENT_ON', 'EVENT_OFF', 'EVENT_SEMANTIC_UNK', 'EVENT_LOCATION_SHUFFLE')
PARTITION = ('direct_report_nodes', 'potential_only', 'remaining_nodes')
SOURCE_FILES = ('summary.json', 'analysis.json', 'replay.json', 'preservation.json',
                'comparability.json', 'region_definitions.json', 'paired_predictions.npz')
FLOW_EDGES = (0., 5., 20., 50., 100., 200.)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    with Path(path).open(encoding='utf-8-sig') as stream:
        return json.load(stream)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def locked_stream_hash(stream):
    # Windows byte-range locking blocks a second handle from reading byte 0,
    # even inside this process. Hash through the already locked read-only handle.
    position, digest = stream.tell(), hashlib.sha256()
    stream.seek(0)
    try:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block)
    finally:
        stream.seek(position)
    return digest.hexdigest()


def snapshot(paths):
    return {key: sha256(path) if Path(path).is_file() else '<missing>' for key, path in paths.items()}


def is_hash(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def assert_close(actual, expected, label, *, atol=1e-9, rtol=1e-9):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and actual.keys() == expected.keys(), 'Saved metric fields differ: '+label)
        for key in expected:
            assert_close(actual[key], expected[key], label+'.'+key, atol=atol, rtol=rtol)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), 'Saved metric vector differs: '+label)
        for i, (left, right) in enumerate(zip(actual, expected)):
            assert_close(left, right, label+'.'+str(i), atol=atol, rtol=rtol)
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        require(isinstance(actual, (int, float)) and not isinstance(actual, bool)
                and np.isfinite(actual) and np.isfinite(expected)
                and np.isclose(actual, expected, atol=atol, rtol=rtol), 'Saved metric mismatch: '+label)
    else:
        require(actual == expected, 'Saved certificate mismatch: '+label)


@contextmanager
def readonly_run_lock(root):
    """Use the trainer's existing OS lock without creating or writing a file."""
    path = Path(root)/'active.lock'
    require(path.is_file() and path.stat().st_size > 0, 'Missing existing run lock; use --evaluation-dir for an archived copy')
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
        raise ValueError('Source run is active; retry after its launcher finishes') from error
    try:
        yield path, stream
    finally:
        stream.close()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--run-dir', type=Path, help='Choose the latest completed early_event_evaluation_* archive')
    source.add_argument('--evaluation-dir', type=Path, help='Explicit completed evaluation archive; no raw data required')
    parser.add_argument('--output-dir', type=Path, help='Fresh directory outside the source archive')
    parser.add_argument('--flow-edges', nargs='+', type=float, default=FLOW_EDGES)
    parser.add_argument('--response-tol', type=float, default=.001)
    args = parser.parse_args(argv)
    for name in ('run_dir', 'evaluation_dir', 'output_dir'):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    edges = np.asarray(args.flow_edges, dtype=np.float64)
    require(edges.ndim == 1 and edges.size >= 2 and np.isfinite(edges).all()
            and edges[0] == 0. and (np.diff(edges) > 0.).all(), 'Flow edges must increase from zero')
    require(np.isfinite(args.response_tol) and args.response_tol >= 0., 'Response tolerance must be finite nonnegative')
    return args


def select_source(args):
    if args.evaluation_dir is not None:
        require(args.evaluation_dir.is_dir(), 'Evaluation directory does not exist')
        return args.evaluation_dir
    root = args.run_dir
    require(root.is_dir(), 'Run directory does not exist')
    candidates = []
    for directory in root.glob('early_event_evaluation_*'):
        if not directory.is_dir() or not (directory/'summary.json').is_file():
            continue
        try:
            summary = read_json(directory/'summary.json')
        except (ValueError, UnicodeDecodeError):
            continue
        if summary.get('schema') == SOURCE_SCHEMA and summary.get('status') == SOURCE_STATUS:
            candidates.append(directory)
    require(candidates, 'No completed PASS early-event evaluation archive found; no predictions will be fabricated')
    return max(candidates, key=lambda path: path.name).resolve()


def output_path(args, source):
    root = args.run_dir if args.run_dir is not None else source.parent
    output = args.output_dir or root/('early_event_gain_analysis_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6])
    output = output.resolve()
    require(output != source and source not in output.parents and output not in source.parents,
            'Output must not replace the source evaluation, a descendant, or an ancestor')
    for arm in ('STATE', 'LOCATION', 'EVENT'):
        checkpoint_root = source.parent/arm
        require(output != checkpoint_root and checkpoint_root not in output.parents,
                'Output must not be inside a checkpoint directory')
    require(not output.exists(), 'Output directory already exists; refusing overwrite')
    return output


def budget_certificate(source, summary, analysis, replay):
    comparable = read_json(source/'comparability.json')
    keys = ('status', 'common_epoch', 'completed_epochs', 'global_updates', 'best_epochs',
            'same_completed_training_budget', 'common_prefix_training_order_verified', 'whole_model_scope')
    require(all(key in comparable for key in keys), 'Missing archived training-budget evidence')
    require(comparable['status'] == 'COMMON_PREFIX_TRAIN_ORDER_VERIFIED'
            and comparable['common_prefix_training_order_verified'] is True,
            'Archived common-prefix training-order certificate did not pass')
    for other, name in ((summary.get('training_comparability'), 'summary'),
                        (analysis.get('training_comparability'), 'analysis')):
        require(isinstance(other, dict), 'Missing training comparability in '+name)
        assert_close({key: other.get(key) for key in keys}, {key: comparable[key] for key in keys}, name+'.training_comparability')
    completed, updates, best = (comparable[key] for key in ('completed_epochs', 'global_updates', 'best_epochs'))
    require(all(isinstance(value, dict) and value.keys() == replay.keys() for value in (completed, updates, best)),
            'Budget arm axes differ from replay certificates')
    require(all(type(completed[arm]) is int and completed[arm] > 0
                and type(updates[arm]) is int and updates[arm] > 0
                and type(best[arm]) is int and 1 <= best[arm] <= completed[arm]
                and best[arm] == replay[arm]['best_epoch'] for arm in replay), 'Invalid archived training budgets or selected epochs')
    require(comparable['common_epoch'] == min(completed.values()), 'Archived common epoch changed')
    equal = len(set(completed.values())) == 1 and len(set(updates.values())) == 1
    require(comparable['same_completed_training_budget'] is equal, 'Archived equal-budget flag disagrees with counters')
    expected_scope = ('same_protocol_equal_completed_budget_independent_best_validation' if equal else
                      'unequal_completed_budget_exploratory_independent_best_validation')
    require(comparable['whole_model_scope'] == expected_scope, 'Archived whole-model scope disagrees with budgets')
    require(comparable.get('test_evaluated', False) is False, 'Evaluation archive unexpectedly used test data')
    return comparable


def original_all_nodes(prediction, target, valid):
    rows, horizons, _ = target.shape
    counts, positive = valid.sum((0, 2)), (valid & (target > 0.)).sum((0, 2))
    absolute, squared, percentage = (np.zeros(horizons, np.float64) for _ in range(3))
    window_absolute = np.zeros((rows, horizons), np.float64)
    window_counts = valid.sum(2)
    for horizon in range(horizons):
        y = target[:, horizon].astype(np.float64)
        error = np.abs(prediction[:, horizon].astype(np.float64)-y)
        mask = valid[:, horizon]
        absolute[horizon] = error[mask].sum(dtype=np.float64)
        squared[horizon] = np.square(error[mask]).sum(dtype=np.float64)
        pct = np.divide(error, y, out=np.zeros_like(y), where=y > 0.)
        percentage[horizon] = pct[mask].sum(dtype=np.float64)
        window_absolute[:, horizon] = np.where(mask, error, 0.).sum(1, dtype=np.float64)
    result = metric_result(absolute, squared, counts, percentage, positive)
    result['window_mae'] = tail_summary(window_scores(window_absolute, window_counts))
    return result


def original_region_metrics(prediction, target, valid, regions):
    """Recheck archived region masks without repeating model or target reads."""
    totals = {name: [np.zeros(12, np.float64), np.zeros(12, np.float64), np.zeros(12, np.float64),
                     np.zeros(12, np.int64), np.zeros(12, np.int64)] for name in regions if name != 'all_nodes'}
    for horizon in range(12):
        y = target[:, horizon].astype(np.float64)
        error = np.abs(prediction[:, horizon].astype(np.float64)-y)
        percentage = np.divide(error, y, out=np.zeros_like(y), where=y > 0.)
        for name, (absolute, squared, pct, counts, positive) in totals.items():
            mask = valid[:, horizon] & regions[name]
            selected = error[mask]
            absolute[horizon] = selected.sum(dtype=np.float64)
            squared[horizon] = np.square(selected).sum(dtype=np.float64)
            pct[horizon] = percentage[mask].sum(dtype=np.float64)
            counts[horizon], positive[horizon] = mask.sum(), (mask & (y > 0.)).sum()
    return {name: metric_result(absolute, squared, counts, percentage, positive)
            for name, (absolute, squared, percentage, counts, positive) in totals.items()}


def load_and_verify(source):
    require(all((source/name).is_file() for name in SOURCE_FILES), 'Missing required evaluation archive artifacts')
    require(not (source/'failure.json').exists(), 'Evaluation archive has a failure record; refusing stale PASS')
    summary, analysis, replay, preservation, definitions = [read_json(source/name) for name in
        ('summary.json', 'analysis.json', 'replay.json', 'preservation.json', 'region_definitions.json')]
    require(summary.get('schema') == SOURCE_SCHEMA and summary.get('status') == SOURCE_STATUS, 'Unsupported evaluation schema or non-PASS status')
    require(analysis.get('schema') == 'min_early_event_metrics_v1'
            and analysis.get('status') == 'MIN_EARLY_EVENT_ANALYSIS_PASS', 'Unsupported saved analysis schema or status')
    require(summary.get('replay_passed') is True and summary.get('on_exactly_restored') is True
            and summary.get('source_artifacts_preserved') is True and summary.get('training_updates') == 0
            and summary.get('test_evaluated') is False and type(summary.get('engineering_only')) is bool,
            'Evaluation completion/protection/engineering certificate is incomplete')
    require(isinstance(replay, dict) and {'STATE', 'EVENT'} <= replay.keys()
            and replay.keys() <= {'STATE', 'EVENT', 'LOCATION'}, 'Missing or unknown replay arm')
    for arm, row in replay.items():
        require(isinstance(row, dict) and row.get('status') == 'BEST_ON_REPLAY_PASS'
                and type(row.get('best_epoch')) is int and row['best_epoch'] > 0,
                'Recorded best ON replay did not pass: '+arm)
    require(summary.get('best_epochs') == {arm: row['best_epoch'] for arm, row in replay.items()}, 'Summary best epochs differ from replay')
    files = preservation.get('files', {})
    require(files.get('status') == 'SOURCE_ARTIFACTS_PRESERVED' and isinstance(files.get('before'), dict)
            and files['before'] and files.get('after') == files['before']
            and files.get('files') == len(files['before'])
            and all(is_hash(value) for value in files['before'].values()), 'Archived source-file preservation did not pass')
    models = preservation.get('models', {})
    require(isinstance(models, dict) and models.keys() == replay.keys()
            and all(row.get('status') == 'MODEL_STATE_PRESERVED'
                    and row.get('mode_gradients_hooks_graph_rng_preserved') is True
                    and is_hash(row.get('model_state_sha256')) for row in models.values()),
            'Archived model preservation did not pass')
    comparable = budget_certificate(source, summary, analysis, replay)
    cases = summary.get('cases')
    require(isinstance(cases, list) and len(cases) == len(set(cases)) and set(BASE_CASES) <= set(cases), 'Missing or repeated archived prediction cases')
    extra = set(cases)-set(BASE_CASES)
    forecast_cases = {f'EVENT_FORECAST_{branch}_{layer}_OFF' for branch in ('DIF', 'INH') for layer in range(5)}
    forecast_cases |= {f'EVENT_FORECAST_{branch}_ALL_OFF' for branch in ('DIF', 'INH')}
    expected_extra = (forecast_cases if summary.get('forecast_audit') is True else set())
    if 'LOCATION' in replay:
        expected_extra.add('LOCATION_ON')
    require(type(summary.get('forecast_audit')) is bool and extra == expected_extra,
            'Archived prediction cases disagree with forecast/replay certificates')
    require(isinstance(definitions.get('masks'), dict) and definitions['masks']
            and all(isinstance(name, str) and name and isinstance(key, str) and key.startswith('region_')
                    for name, key in definitions['masks'].items())
            and len(set(definitions['masks'].values())) == len(definitions['masks']), 'Invalid region definitions mapping')
    region_keys = definitions['masks']
    with np.load(source/'paired_predictions.npz', allow_pickle=False) as archive:
        metadata = {'target', 'valid', 'sample_indices', 'station_ids', 'location_donor_station_indices'}
        require(set(archive.files) == set(cases) | set(region_keys.values()) | metadata, 'NPZ fields disagree with archived cases/regions')
        predictions = {case: archive[case] for case in BASE_CASES}
        target, valid, indices, stations = [archive[name] for name in ('target', 'valid', 'sample_indices', 'station_ids')]
        regions = {name: archive[key] for name, key in region_keys.items()}
    require(target.ndim == 3 and target.shape[1] == 12 and target.shape[0] > 0 and target.shape[2] > 0
            and target.dtype.kind in 'fiu' and np.isfinite(target).all(), 'Invalid numeric target axes or values')
    rows, horizons, nodes = target.shape
    require(type(summary.get('validation_rows')) is int and summary['validation_rows'] == rows
            and analysis.get('validation_rows') == rows and analysis.get('horizons') == horizons
            and analysis.get('stations') == nodes, 'Archived sample/station/horizon counts disagree')
    require(valid.shape == target.shape and valid.dtype == np.bool_ and valid.any(), 'Invalid explicit target validity mask')
    require(indices.shape == (rows,) and indices.dtype.kind in 'iu' and (indices >= 0).all()
            and np.unique(indices).size == rows and (rows == 1 or (np.diff(indices) == 1).all()), 'Invalid original sample IDs')
    require(stations.shape == (nodes,) and stations.dtype.kind in 'iu' and np.unique(stations).size == nodes,
            'Invalid unique station IDs')
    require(all(value.shape == target.shape and value.dtype.kind in 'fiu' and np.isfinite(value).all()
                for value in predictions.values()), 'Invalid paired prediction numeric axes or values')
    normalized = {}
    for name, mask in regions.items():
        require(mask.dtype == np.bool_ and mask.shape in ((nodes,), (rows, nodes)), 'Invalid bool region axes: '+name)
        normalized[name] = np.broadcast_to(mask, (rows, nodes))
    require('all_nodes' in normalized and normalized['all_nodes'].all() and all(name in normalized for name in PARTITION),
            'Missing complete all_nodes/report partitions')
    require((sum(normalized[name].astype(np.uint8) for name in PARTITION) == 1).all(), 'Report regions are not disjoint/exhaustive')
    if 'candidate_potential' in normalized:
        require(np.array_equal(normalized['candidate_potential'], normalized[PARTITION[0]] | normalized[PARTITION[1]]),
                'Potential region disagrees with report partitions')
    batch = summary.get('batch_size')
    require(type(batch) is int and batch > 0, 'Invalid original validation batch size')
    partitions = [min(batch, rows-i) for i in range(0, rows, batch)]
    mask_values = []
    for arm, row in replay.items():
        require(row.get('batch_partition') == partitions, 'Recorded original batch partition changed')
        require(isinstance(row.get('mask_value'), (int, float)) and np.isfinite(row['mask_value']), 'Invalid recorded mask value')
        mask_values.append(row['mask_value'])
        for name in ('expected', 'actual'):
            require(isinstance(row.get(name), list) and len(row[name]) == 3 and np.isfinite(row[name]).all(), 'Invalid recorded selection metric vector')
        require(isinstance(row.get('atol'), (int, float)) and 0 <= row['atol'] <= 1e-5
                and isinstance(row.get('rtol'), (int, float)) and 0 <= row['rtol'] <= 1e-6, 'Recorded replay tolerances were loosened')
        require(np.allclose(row['actual'], row['expected'], atol=row['atol'], rtol=row['rtol']), 'Recorded selected checkpoint metrics do not match')
    require(len(set(mask_values)) == 1 and np.array_equal(valid, target != mask_values[0]), 'Saved validity differs from original replay mask')
    native_mask = float(target.min()) if target.min() < 1. else 0.
    require(mask_values[0] == native_mask and (target[valid] >= 0.).all(), 'Recorded mask differs from original native target convention')
    recomputed = {case: original_all_nodes(value, target, valid) for case, value in predictions.items()}
    for case, metrics in recomputed.items():
        require(case in analysis.get('cases', {}) and case in summary.get('all_nodes', {}), 'Missing saved all_nodes metrics: '+case)
        assert_close(metrics, analysis['cases'][case]['metrics']['all_nodes'], 'analysis.'+case)
        assert_close(metrics, summary['all_nodes'][case], 'summary.'+case)
        archived_regions = analysis['cases'][case]['metrics']
        require(archived_regions.keys() == normalized.keys(), 'Archived region metric axes differ: '+case)
        for name, value in original_region_metrics(predictions[case], target, valid, normalized).items():
            assert_close(value, archived_regions[name], 'analysis.'+case+'.region.'+name)
        arm = case.removesuffix('_ON')
        if arm in ('STATE', 'EVENT'):
            actual = [metrics[key] for key in ('mae_pooled', 'mape_pooled', 'rmse_pooled')]
            for name in ('actual', 'expected'):
                require(np.allclose(actual, replay[arm][name], atol=replay[arm]['atol'], rtol=replay[arm]['rtol']),
                        'Archived predictions disagree with selected ON metric certificate: '+arm+'.'+name)
    fixed = analysis.get('fixed_hard_windows', {})
    hard = np.asarray(fixed.get('sample_indices', []))
    require(fixed.get('selector') == 'STATE_ON' and fixed.get('fraction') == .1
            and hard.ndim == 1 and hard.dtype.kind in 'iu' and hard.size > 0
            and np.unique(hard).size == hard.size and np.isin(hard, indices).all()
            and fixed.get('selected_count') == hard.size, 'Invalid archived fixed difficult-window sample IDs')
    selected = np.isin(indices, hard)
    require((valid[selected].sum(2) > 0).all(), 'Fixed difficult windows contain an incomplete horizon')
    for case, prediction in predictions.items():
        recorded = fixed.get('cases', {}).get(case)
        require(isinstance(recorded, dict), 'Missing fixed difficult-window metric certificate: '+case)
        fixed_metrics = original_all_nodes(prediction[selected], target[selected], valid[selected])
        fixed_windows = fixed_metrics.pop('window_mae')
        assert_close(recorded, dict(window_mean_mae=fixed_windows['mean'], metrics=fixed_metrics), 'fixed_hard_windows.'+case)
    identity = comparable.get('identity')
    axis_scope = 'archived_axes_only_no_new_release_axis_certification'
    if identity is not None:
        require(isinstance(identity, dict) and identity.get('schema') == 'min_early_event_published_run_v1', 'Unsupported archived training identity schema')
        require(identity.get('engineering_check') is summary['engineering_only'], 'Engineering flag differs from archived training identity')
        protocol = identity.get('protocol', {})
        train_rows = protocol.get('train_samples')
        require(type(train_rows) is int and train_rows >= 0 and indices[0] == train_rows
                and (rows <= protocol.get('val_samples', -1) if summary['engineering_only'] else rows == protocol.get('val_samples')),
                'Original sample IDs/count do not match archived release split identity')
        require(identity.get('batch_size') == batch and protocol.get('stations') == nodes, 'Archived identity station/batch axes changed')
        axis_scope = 'saved_release_identity_axes_verified_without_raw_data_read'
    provenance = read_json(source/'provenance.json') if (source/'provenance.json').is_file() else {}
    if 'station_ids' in provenance:
        require(provenance['station_ids'] == stations.tolist(), 'Station IDs differ from archived provenance')
    return dict(summary=summary, analysis=analysis, comparable=comparable, predictions=predictions,
                target=target, valid=valid, regions=regions, indices=indices, stations=stations,
                fixed_hard_ids=hard, recomputed_all_nodes=recomputed, axis_scope=axis_scope)


def write_csv(path, rows):
    rows = list(rows)
    keys = list(dict.fromkeys(key for row in rows for key in row)) or ['comparison', 'group', 'horizon']
    with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False, allow_nan=False) if isinstance(value, (dict, list, tuple)) else value
                             for key, value in row.items()})


def write_tables(output, result):
    rows = flat_rows(result)
    families = {
        'flow.csv': lambda row: str(row.get('group', '')).startswith('flow:') and row.get('horizon') == 0,
        'bias.csv': lambda row: str(row.get('group', '')).startswith('bias:') and row.get('horizon') == 0,
        'roads.csv': lambda row: str(row.get('group', '')).startswith('road:') and row.get('horizon') == 0,
        'cross.csv': lambda row: str(row.get('group', '')).startswith('cross:') and row.get('horizon') == 0,
        'cross_horizons.csv': lambda row: str(row.get('group', '')).startswith('cross:') and row.get('horizon', 0) > 0,
        'road_horizons.csv': lambda row: str(row.get('group', '')).startswith('road:') and row.get('horizon', 0) > 0,
        'region_horizons.csv': lambda row: row.get('family') == 'region' and row.get('group') != 'all_nodes' and row.get('horizon', 0) > 0,
        'horizons.csv': lambda row: row.get('group') == 'all_nodes' and row.get('horizon', 0) > 0,
        'fixed_hard.csv': lambda row: row.get('group') == 'fixed_hard_windows' and row.get('horizon') == 0,
        'all_nodes.csv': lambda row: row.get('group') == 'all_nodes' and row.get('horizon') == 0,
        'regions.csv': lambda row: row.get('horizon') == 0 and row.get('group') in result.get('group_definitions', {})
            and not str(row.get('group', '')).startswith(('flow:', 'bias:', 'road:', 'cross:'))
            and row.get('group') not in ('all_nodes', 'fixed_hard_windows'),
    }
    for name, selector in families.items():
        write_csv(output/name, (row for row in rows if selector(row)))
    flattened = []
    for group, families in result.get('quadrants', {}).get('groups', {}).items():
        for family, values in families.items():
            if not isinstance(values, dict):
                continue
            if 'valid_count' in values:
                flattened.append(dict(group=group, family=family, label='all', **values))
            else:
                for label, metrics in values.items():
                    if isinstance(metrics, dict):
                        flattened.append(dict(group=group, family=family, label=label, **metrics))
    write_csv(output/'gain_overlap.csv', flattened)


def write_report(output, result, source_info):
    fmt = lambda value: '不可用' if value is None else f'{value:.9f}'
    summary, comparable = source_info['summary'], source_info['comparable']
    lines = ['# MIN 早期事故模块：已有预测收益分解', '',
        f"源评估验证窗口：{summary['validation_rows']}；工程检查：{summary['engineering_only']}。", '',
        '仅分析已保存的五组配对预测，不读取原始交通数据或检查点，不加载模型，不增加推理，不训练，不评价测试集。', '',
        'T = STATE误差 − EVENT_ON误差；P = EVENT_OFF误差 − EVENT_ON误差；R = STATE误差 − EVENT_OFF误差。',
        '三者在同一预测点满足 T = P + R。T是独立训练模型的总体差异，P是EVENT检查点对事故路径的依赖，R是其余差异；R不能单独归因为训练正则化。', '',
        '## 全网比较', '', '| 比较 | MAE收益 pooled | MAE收益 horizon macro | 正收益 pooled | 负收益 pooled | 抵消比例 | 候选预测升高均值 |', '|---|---:|---:|---:|---:|---:|---:|']
    for name, row in result['comparisons'].items():
        metrics = row['groups']['all_nodes']
        values = [metrics.get(key) for key in ('mae_gain_pooled', 'mae_gain_macro', 'gain_positive_pooled',
                                              'gain_negative_pooled', 'cancellation_pooled')]
        values.append(metrics['prediction_change']['signed_mean_candidate_minus_control_pooled'])
        lines.append('| '+name+' | '+' | '.join(fmt(value) for value in values)+' |')
    p_change = result['comparisons']['P']['groups']['all_nodes']['prediction_change']
    lines += ['', 'P 的预测变化方向统一为 EVENT_ON − EVENT_OFF；正数表示开启路径后预测升高。',
        '上升点占比：'+fmt(p_change['up_fraction'])+'；下降点占比：'+fmt(p_change['down_fraction'])+
        '；超过响应阈值的点占比：'+fmt(p_change['response_fraction'])+'。',
        '逐点 T=P+R 与分层贡献守恒检查：'+result['identity_check']['status']+'。',
        '正收益、负收益和抵消比例分别由每种比较自己的逐点误差变化计算，不能把 T/P/R 的正负部分相加，也不使用净 P/T 比例作事故归因。']
    lines += ['分层表中的 MAE收益使用组内有效预测点等权 pooled 均值；全网贡献统一除以全网有效点数。horizon macro 是12个时距均值等权，分组缺失时距时不可用。',
        'MAE/RMSE沿用作者原流量尺度，未新增物理单位认证；JSON/CSV中的MAPE为比例。']
    lines += ['', '## RMSE与MAPE权衡', '',
        '收益均为对照减候选；正数表示误差更低。MAPE收益在本表换算为百分点，RMSE收益不满足T=P+R的线性分解。', '',
        '| 比较 | RMSE收益 pooled | RMSE收益 horizon macro | MAPE收益 pooled（百分点） | MAPE收益 horizon macro（百分点） |',
        '|---|---:|---:|---:|---:|']
    for name, row in result['comparisons'].items():
        metrics = row['groups']['all_nodes']
        values = [metrics['rmse_gain_pooled'], metrics['rmse_gain_macro']]
        values += [None if metrics[key] is None else 100.*metrics[key]
                   for key in ('mape_gain_pooled', 'mape_gain_macro')]
        lines.append('| '+name+' | '+' | '.join(fmt(value) for value in values)+' |')
    definitions = result['group_definitions']
    for title, names in (
        ('真实流量区间', [name for name in definitions if name.startswith('flow:')]),
        ('固定 STATE 高估／低估分组', ['bias:over', 'bias:under', 'bias:exact']),
        ('事故关联三分区', list(PARTITION)),
        ('道路分组', [name for name in definitions if name.startswith('road:')]),
    ):
        lines += ['', '## '+title, '', '| 分组 | 有效点数 | T MAE收益 | P MAE收益 | R MAE收益 | P全网贡献 |', '|---|---:|---:|---:|---:|---:|']
        for group in names:
            metrics = {key: result['comparisons'][key]['groups'][group] for key in ('T', 'P', 'R')}
            lines.append('| '+group+' | '+str(metrics['P']['valid_count'])+' | '+
                ' | '.join(fmt(metrics[key]['mae_gain_pooled']) for key in ('T', 'P', 'R'))+' | '+
                fmt(metrics['P']['global_mae_contribution_pooled'])+' |')
    lines += ['', '## 逐预测时距', '', '| 步 | 有效点数 | T MAE收益 | P MAE收益 | R MAE收益 |', '|---|---:|---:|---:|---:|']
    all_metrics = {key: result['comparisons'][key]['groups']['all_nodes'] for key in ('T', 'P', 'R')}
    for horizon in range(12):
        lines.append('| '+str(horizon+1)+' | '+str(all_metrics['P']['valid_count_per_horizon'][horizon])+' | '+
            ' | '.join(fmt(all_metrics[key]['per_horizon_mae_gain'][horizon]) for key in ('T', 'P', 'R'))+' |')
    fixed = result['fixed_hard_windows']
    lines += ['', '## 原评估固定困难窗口', '',
        '直接复用样本ID：'+str(fixed['sample_indices'])+'；完整窗口数：'+str(fixed['complete_window_count'])+'。', '',
        '| 比较 | 窗口等权 MAE收益 | 逐点 MAE收益 |', '|---|---:|---:|']
    for key in ('T', 'P', 'R', 'Q', 'L'):
        row = fixed['comparisons'][key]
        lines.append('| '+key+' | '+fmt(row['window_equal_weight_mean_gain'])+' | '+fmt(row['point_metrics']['mae_gain_pooled'])+' |')
    quadrants = result['quadrants']['groups']['all_nodes']
    lines += ['', '## 总体收益与路径收益是否重合', '',
        '| T／P同点符号 | 有效点数 | 全网点数占比 | T全网贡献 | P全网贡献 | R全网贡献 |', '|---|---:|---:|---:|---:|---:|']
    for label, row in quadrants['T_P_positive_nonpositive'].items():
        lines.append('| '+label.replace('|', ' · ')+' | '+str(row['valid_count'])+' | '+fmt(row['global_point_fraction'])+' | '+
            ' | '.join(fmt(row[key+'_global_mae_contribution_pooled']) for key in ('T', 'P', 'R'))+' |')
    opposed = quadrants['P_R_opposed']
    lines += ['', 'P与R在同一点反向作用的点数：'+str(opposed['valid_count'])+'；全网点数占比：'+fmt(opposed['global_point_fraction'])+'。',
        '该集合的逐点抵消量 Σ(|P|+|R|−|T|)：'+fmt(opposed['P_R_pointwise_cancellation_sum'])+
        '；除以 Σ(|P|+|R|) 的比例：'+fmt(opposed['P_R_pointwise_cancellation_fraction'])+'。',
        '这是同点算术抵消，不能解释为一个模块训练造成另一个模块的因果变化。']
    cross = [(name, row) for name, row in result['comparisons']['P']['groups'].items()
             if name.startswith('cross:') and row['valid_count'] and row['global_mae_contribution_pooled'] is not None]
    positive = sorted((item for item in cross if item[1]['global_mae_contribution_pooled'] > 0.),
                      key=lambda item: (-item[1]['global_mae_contribution_pooled'], item[0]))[:3]
    negative = sorted((item for item in cross if item[1]['global_mae_contribution_pooled'] < 0.),
                      key=lambda item: (item[1]['global_mae_contribution_pooled'], item[0]))[:3]
    lines += ['', '## P贡献最大的正／负交叉组', '',
        '按共同全网有效点数分母的贡献排序，各列至多三组；用于定位改进对象，不把小组均值直接相加。', '',
        '| 方向 | 流量 × STATE偏差 × 事故区域 | 点数 | 组内P收益 | P全网贡献 |', '|---|---|---:|---:|---:|']
    for direction, selected in (('正', positive), ('负', negative)):
        for name, row in selected:
            lines.append('| '+direction+' | '+name.replace('|', ' · ')+' | '+str(row['valid_count'])+' | '+
                fmt(row['mae_gain_pooled'])+' | '+fmt(row['global_mae_contribution_pooled'])+' |')
    lines += ['', '## 可比性与解释边界', '',
        '完成轮次：'+str(comparable['completed_epochs'])+'；最佳轮次：'+str(comparable['best_epochs'])+'。',
        '优化器更新：'+str(comparable['global_updates'])+'；整体模型比较范围：'+comparable['whole_model_scope']+'。',
        '选模证书、原掩码、原样本和站点轴、已保存全网指标已与归档预测核对。这是归档内部一致性检查，不是新增检查点回放，也不是对历史文件的独立签名认证。',
        '固定困难窗口直接复用原评估保存的样本ID；没有重新挑选。流量桶和高估/低估分组为真值可见的事后诊断，不可作为推理输入或门控。',
        '事故关联区域来自原关系编码；候选范围是弱连通分量包络，不是已认证事故影响范围。其余区域不等于自然无事故。',
        'Q为语义UNK敏感性，L为全站事故关系重分配敏感性，均可能产生分布外输入。路径依赖、模型间收益和真实事故因果不可互换。',
        'gain_overlap.csv 中27种严格符号组合、4种T/P正或非正组合、9种P/R符号组合各自互斥完备；各family之间不可相加，P_R_opposed是另外的重叠汇总。',
        '窗口和预测点可能重叠，不作为独立事故重复。验证集已用于研究和选模，结果属于探索性诊断。', '',
        '数值明细：analysis.json；all_nodes.csv、flow.csv、bias.csv、regions.csv、roads.csv、horizons.csv、cross.csv、gain_overlap.csv、fixed_hard.csv；逐时距交叉与道路/区域明细：cross_horizons.csv、road_horizons.csv、region_horizons.csv。', '']
    (output/'report.md').write_text('\n'.join(lines), encoding='utf-8')


def analyze_run(args):
    source = select_source(args)
    output = output_path(args, source)
    lock_root = args.run_dir if args.run_dir is not None else source.parent if (source.parent/'active.lock').is_file() else None
    lock = readonly_run_lock(lock_root) if lock_root is not None else nullcontext(None)
    with lock as lock_state:
        def source_paths():
            files = {str(path.relative_to(source)): path for path in source.rglob('*') if path.is_file()}
            require(all(source == path.resolve().parent or source in path.resolve().parents for path in files.values()), 'Source archive contains an external file link')
            return files
        paths = source_paths()
        def protected_snapshot():
            values = snapshot(source_paths())
            if lock_state is not None:
                values['<existing_source_run_lock>'] = locked_stream_hash(lock_state[1])
            return values
        before = protected_snapshot()
        output.mkdir(parents=True)
        print('Source evaluation: '+str(source), flush=True)
        print('Analysis directory: '+str(output), flush=True)
        try:
            info = load_and_verify(source)
            result = analyze_predictions(info['predictions'], info['target'], info['valid'], info['regions'],
                info['indices'], info['stations'], info['fixed_hard_ids'], flow_edges=tuple(args.flow_edges), response_tol=args.response_tol)
            result.update(source_scope=info['comparable']['whole_model_scope'], engineering_only=info['summary']['engineering_only'],
                source_best_epochs=info['summary']['best_epochs'], training_comparability=info['comparable'],
                archive_axis_scope=info['axis_scope'], source_validation_rows=info['summary']['validation_rows'],
                fixed_hard_selection='reused_original_archived_sample_ids_no_reselection',
                new_inference_calls=0, optimizer_updates=0, test_evaluated=False,
                source_certificate_check='saved_arrays_and_recorded_replay_internal_consistency_not_new_checkpoint_replay')
            write_json(output/'analysis.json', result)
            write_tables(output, result)
            write_report(output, result, info)
            provenance = dict(schema=SCHEMA, source_evaluation=str(source), source_files_sha256_before=before,
                source_summary_schema=info['summary']['schema'], source_summary_status=info['summary']['status'],
                source_validation_rows=info['summary']['validation_rows'], source_best_epochs=info['summary']['best_epochs'],
                analysis_source_sha256={str(path.relative_to(REPO)): sha256(path) for path in
                    (Path(__file__).resolve(), HERE/'min_early_event_gain_metrics.py', HERE/'min_event_budget_probe_metrics.py')},
                original_fixed_hard_sample_ids=info['fixed_hard_ids'].tolist(), flow_edges=list(args.flow_edges), response_tol=args.response_tol,
                imported_runtime='numpy_and_standard_library_only', raw_data_read=False, checkpoint_loaded=False,
                historical_signature_verified=False, new_checkpoint_replay=False, new_inference_calls=0, optimizer_updates=0, test_evaluated=False)
            write_json(output/'provenance.json', provenance)
            after = protected_snapshot()
            require(before == after, 'Source artifacts changed during read-only analysis')
            write_json(output/'preservation.json', dict(status='SOURCE_ARTIFACTS_PRESERVED', source_artifacts_preserved=True,
                files=len(before), before=before, after=after,
                scope='this_analysis_read_interval_byte_hashes_not_original_archive_signature'))
            summary = dict(schema=SCHEMA, status='MIN_EARLY_EVENT_GAIN_ANALYSIS_PASS', source_evaluation=str(source),
                validation_rows=info['summary']['validation_rows'], engineering_only=info['summary']['engineering_only'],
                source_best_epochs=info['summary']['best_epochs'], source_scope=info['comparable']['whole_model_scope'],
                all_nodes={name: row['groups']['all_nodes'] for name, row in result['comparisons'].items()},
                new_inference_calls=0, optimizer_updates=0, test_evaluated=False, source_artifacts_preserved=True,
                source_certificate_check='archived_internal_consistency_not_new_checkpoint_replay', output_dir=str(output))
            write_json(output/'summary.json', summary)
            return summary
        except BaseException as error:
            after = protected_snapshot()
            write_json(output/'preservation.json', dict(status='SOURCE_ARTIFACTS_PRESERVED' if before == after else 'SOURCE_ARTIFACTS_CHANGED',
                source_artifacts_preserved=before == after, files=len(before), before=before, after=after,
                analysis_completed=False, scope='this_analysis_read_interval_byte_hashes_not_original_archive_signature'))
            write_json(output/'failure.json', dict(schema=SCHEMA, status='MIN_EARLY_EVENT_GAIN_ANALYSIS_FAILED',
                error_type=type(error).__name__, error=str(error), source_evaluation=str(source),
                new_inference_calls=0, optimizer_updates=0, test_evaluated=False))
            raise


def main(argv=None):
    result = analyze_run(parse_args(argv))
    concise = {key: value for key, value in result.items() if key != 'all_nodes'}
    concise['all_nodes'] = {name: {key: row.get(key) for key in ('mae_gain_pooled', 'mae_gain_macro')}
                            for name, row in result['all_nodes'].items()}
    print(json.dumps(concise, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
    return result


if __name__ == '__main__':
    main()
