"""Decompose saved frozen-MIN ON/OFF validation predictions without model calls.

Uses NumPy and the standard library only. Raw targets define retrospective
diagnostic strata; neither targets nor baseline errors become model inputs.
"""
import argparse
from contextlib import contextmanager
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
from experiments.chronological.min_frozen_budget_metrics import select_hard_windows, summarize_pair
from experiments.chronological.min_frozen_gain_metrics import summarize_strata

HERE = Path(__file__).resolve().parent
SCHEMA = 'frozen_MIN_saved_gain_analysis_v1'
RUN_SCHEMA = 'frozen_MIN_location_budget_pilot_v1'
RUN_FILES = ('identity.json', 'initialization.json', 'baseline.json', 'history.json',
             'summary.json', 'source_preservation.json', 'baseline_predictions.npz',
             'best_predictions.npz', 'last.pt')
PARTITION = ('direct_report_nodes', 'potential_only', 'remaining_nodes')


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


def snapshot(paths):
    return {key: sha256(path) for key, path in paths.items()}


def default_data():
    for directory in (REPO/'data/xtraffic/Contra_Costa', REPO.parent/'data/xtraffic/Contra_Costa'):
        if (directory/'incident_all.npy').is_file():
            return directory
    return REPO.parent/'data/xtraffic/Contra_Costa'


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True, help='Completed or paused frozen location pilot')
    parser.add_argument('--data-dir', type=Path, default=default_data())
    parser.add_argument('--graph-dir', type=Path, default=HERE/'internal_spatial_graph_is11')
    parser.add_argument('--output-dir', type=Path, help='New directory only; default is a timestamped run subdirectory')
    parser.add_argument('--flow-edges', type=float, nargs='+', default=(0., 5., 20., 50., 100., 200.),
                        help='Fixed left edges in original published flow scale, starting at zero; no quantile fitting')
    args = parser.parse_args(argv)
    for name in ('run_dir', 'data_dir', 'graph_dir'):
        setattr(args, name, getattr(args, name).resolve())
    edges = np.asarray(args.flow_edges, dtype=np.float64)
    require(edges.ndim == 1 and len(edges) >= 2 and np.isfinite(edges).all()
            and edges[0] == 0. and (np.diff(edges) > 0).all(), 'Flow edges must increase from zero')
    args.output_dir = (args.output_dir.resolve() if args.output_dir else args.run_dir/(
        'frozen_gain_analysis_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6]))
    require(args.output_dir != args.run_dir and args.output_dir not in args.run_dir.parents,
            'Output must not replace the source run or an ancestor')
    for directory in (args.data_dir, args.graph_dir):
        require(args.output_dir != directory and directory not in args.output_dir.parents
                and args.output_dir not in directory.parents, 'Output must be separate from release and graph inputs')
    require(not args.output_dir.exists(), 'Output directory already exists; refusing overwrite')
    return args


@contextmanager
def readonly_run_lock(root):
    """Coordinate with the existing trainer without creating or writing a lock."""
    path = Path(root)/'active.lock'
    require(path.is_file() and path.stat().st_size > 0, 'Missing existing run lock; use a completed frozen pilot')
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
        raise ValueError('Source run is active; retry analysis after its launcher finishes') from error
    try:
        yield
    finally:
        stream.close()


def load_archives(root):
    expected = {'baseline_predictions.npz': {'prediction', 'target', 'valid', 'sample_indices', 'fixed_hard_indices'},
                'best_predictions.npz': {'on', 'off', 'target', 'valid', 'sample_indices', 'fixed_hard_indices'}}
    result = {}
    for name, keys in expected.items():
        with np.load(Path(root)/name, allow_pickle=False) as archive:
            require(set(archive.files) == keys, 'Unexpected archive fields: '+name)
            result[name] = {key: archive[key] for key in keys}
    baseline, best = result['baseline_predictions.npz'], result['best_predictions.npz']
    for key in ('target', 'valid', 'sample_indices', 'fixed_hard_indices'):
        require(np.array_equal(best[key], baseline[key], equal_nan=True), 'Paired archive mismatch: '+key)
    require(np.array_equal(best['off'], baseline['prediction']), 'Saved OFF differs from the fixed MIN baseline')
    return best, baseline


def assert_close(actual, expected, label):
    """Tight float64 consistency check, never a model replay tolerance."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and actual.keys() == expected.keys(), 'Metric fields changed: '+label)
        for key in expected:
            assert_close(actual[key], expected[key], label+'.'+key)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), 'Metric vector changed: '+label)
        for index, (left, right) in enumerate(zip(actual, expected)):
            assert_close(left, right, label+'.'+str(index))
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        require(isinstance(actual, (int, float)) and not isinstance(actual, bool)
                and np.isfinite(actual) and np.isfinite(expected)
                and np.isclose(actual, expected, atol=1e-12, rtol=1e-12), 'Saved metric mismatch: '+label)
    else:
        require(actual == expected, 'Saved certificate mismatch: '+label)


def verify_certificates(root, identity, summary, history, baseline_info, best, baseline):
    require(identity.get('schema') == RUN_SCHEMA, 'Expected a frozen MIN location pilot')
    require(summary.get('status') in ('FROZEN_MIN_LOCATION_PILOT_COMPLETE',
            'FROZEN_MIN_LOCATION_PAUSED_AT_EPOCH_BOUNDARY', 'FROZEN_MIN_LOCATION_ENGINEERING_PASS'),
            'Source pilot has no successful committed summary')
    for key in ('frozen_backbone_preserved', 'off_exactly_matches_source', 'source_artifacts_preserved',
                'selection_includes_epoch_zero'):
        require(summary.get(key) is True, 'Missing successful training certificate: '+key)
    require(summary.get('test_evaluated') is False, 'This analysis expects the validation-only pilot')
    require(summary.get('engineering_only') == identity.get('engineering_check'), 'Engineering scope differs')
    require(summary.get('frozen_state_sha256') == identity.get('frozen_state_sha256'), 'Frozen state certificate differs')
    require(summary.get('source_best_epoch') == identity.get('source_best_epoch')
            == baseline_info.get('source_best_epoch'), 'Source best epoch certificate differs')
    require(baseline_info.get('zero_ON_and_OFF_exactly_match_frozen_MIN') is True
            and baseline_info.get('original_source_metrics_replay_passed') is True, 'Missing epoch-zero replay certificate')
    completed = summary.get('completed_epoch')
    require(type(completed) is int and 0 <= completed <= identity['recipe']['max_epochs']
            and isinstance(history, list) and len(history) == completed+1
            and [row.get('epoch') for row in history] == list(range(completed+1)), 'Invalid complete epoch-zero history')
    for row in history:
        require(row.get('frozen_state_sha256') == identity['frozen_state_sha256'], 'History frozen state changed')
        value = row['validation']['on']['mae_pooled']
        require(isinstance(value, (float, int)) and np.isfinite(value), 'Invalid historical selection metric')
        require(row['validation'].get('off_exactly_matches_source') is True
                and row['validation'].get('on_exactly_restored') is True, 'Historical pairing certificate failed')
    selected = min(range(len(history)), key=lambda i: history[i]['validation']['on']['mae_pooled'])
    require(type(summary.get('best_adapter_epoch')) is int and summary['best_adapter_epoch'] == selected,
            'Best adapter epoch differs from earliest history minimum')
    shape = best['target'].shape
    require(len(shape) == 3 and shape[1] == 12 and shape[0] > 0
            and shape[2] == identity['source_identity']['protocol']['stations'], 'Unexpected validation axes')
    require(best['valid'].shape == shape and best['valid'].dtype == np.bool_, 'Expected explicit Boolean validity mask')
    for key in ('on', 'off', 'target'):
        require(best[key].shape == shape and best[key].dtype == np.float32
                and np.isfinite(best[key]).all(), 'Expected finite original float32 archive: '+key)
    indices = best['sample_indices']
    require(indices.shape == (shape[0],) and indices.dtype.kind in 'iu'
            and len(np.unique(indices)) == len(indices), 'Invalid or duplicate sample axis')
    target = best['target']
    require(np.isfinite(target).all(), 'Saved target contains nonfinite values')
    null = float(target.min()) if float(target.min()) < 1. else 0.
    require(baseline_info.get('mask_value') == null
            and np.array_equal(best['valid'], np.isfinite(target) & (target != null)), 'Saved validity mask differs from source rule')
    hard = select_hard_windows(best['off'], target, best['valid'], indices,
                               fraction=identity['recipe']['fixed_hard_fraction'])
    require(best['fixed_hard_indices'].dtype.kind in 'iu'
            and np.array_equal(best['fixed_hard_indices'], hard), 'Fixed baseline difficult-window selection differs')
    require(baseline_info.get('fixed_hard_sample_indices') == indices[hard].tolist(), 'Fixed difficult-window IDs differ')
    require(summary.get('fixed_hard_windows') == len(hard), 'Fixed difficult-window count differs')
    require(baseline_info.get('validation_rows') == shape[0], 'Baseline row certificate differs')
    partition = [min(identity['batch_size'], shape[0]-i) for i in range(0, shape[0], identity['batch_size'])]
    require(baseline_info.get('batch_partition') == partition, 'Baseline batch partition differs')
    for name, certificate in (('best', summary['best_validation']), ('baseline', baseline_info['validation'])):
        require(certificate.get('off_exactly_matches_source') is True and certificate.get('on_exactly_restored') is True
                and certificate.get('validation_rows') == shape[0] and certificate.get('batch_partition') == partition,
                'Saved pairing/row/batch certificate failed: '+name)
    for row in history:
        require(row['validation'].get('validation_rows') == shape[0]
                and row['validation'].get('batch_partition') == partition, 'Historical row/batch certificate differs')
    pair = summarize_pair(best['on'], best['off'], target, best['valid'], hard)
    zero = summarize_pair(best['off'], best['off'], target, best['valid'], hard)
    for name, recomputed in (('best_validation', pair),):
        for key, value in recomputed.items():
            assert_close(summary[name][key], value, name+'.'+key)
            assert_close(history[selected]['validation'][key], value, 'history_best.'+key)
    for key, value in zero.items():
        assert_close(history[0]['validation'][key], value, 'history_zero.'+key)
        assert_close(baseline_info['validation'][key], value, 'baseline_zero.'+key)
    for row in history:
        assert_close(row['validation']['off'], zero['off'], 'history_OFF')
    assert_close(summary['baseline_metrics'], pair['off'], 'baseline_metrics')
    assert_close(summary['best_metrics'], pair['on'], 'best_metrics')
    assert_close(summary['baseline_mae'], pair['off']['mae_pooled'], 'baseline_mae')
    assert_close(summary['best_mae'], pair['on']['mae_pooled'], 'best_mae')
    assert_close(summary['baseline_minus_best_mae'], pair['mae_gain_pooled'], 'baseline_minus_best')
    preservation = read_json(Path(root)/'source_preservation.json')
    require(preservation.get('status') == 'SOURCE_ARTIFACTS_PRESERVED'
            and preservation.get('before') == preservation.get('after') == identity['source_artifacts_sha256'],
            'Original source-preservation certificate failed')
    return pair


def report_support(sample, nodes):
    """Match original encode_event float32 relation/presence policy exactly."""
    require(isinstance(sample, dict), 'Expected a published sample dictionary')
    available = 'incident_distances' in sample
    present = sample.get('report_present', 'incident_features' in sample or available)
    require(isinstance(present, (bool, np.bool_)), 'Non-Boolean report presence flag')
    if not available:
        return np.zeros(nodes, dtype=bool)
    raw = np.asarray(sample['incident_distances'])
    require(raw.shape == (nodes, 3) and raw.dtype.kind in 'fiu' and np.isfinite(raw).all(), 'Invalid published report relation')
    with np.errstate(over='ignore', invalid='ignore'):
        relation = raw.astype(np.float32)
    require(np.isfinite(relation).all(), 'Report relation overflows float32')
    return (np.abs(relation).sum(-1) > 0) & bool(present)


def partition_regions(direct, graph, station_ids):
    nodes = len(station_ids)
    require(graph.get('station_ids') == list(station_ids), 'Graph station axis differs from release')
    require(direct.ndim == 2 and direct.shape[1] == nodes and direct.dtype == np.bool_, 'Invalid direct-support axis')
    neighbors = [set() for _ in range(nodes)]
    require(isinstance(graph.get('edges'), list), 'Missing graph edge list')
    for edge in graph['edges']:
        u, v = edge['source'], edge['destination']
        require(type(u) is int and type(v) is int and -1 <= u < nodes and -1 <= v < nodes
                and (u >= 0 or v >= 0), 'Graph edge outside the station axis')
        if u >= 0 and v >= 0:
            neighbors[u].add(v)
            neighbors[v].add(u)
    potential, unseen = direct.copy(), set(range(nodes))
    while unseen:
        seed = min(unseen)
        unseen.remove(seed)
        component, queue = [seed], [seed]
        while queue:
            for node in neighbors[queue.pop()]:
                if node in unseen:
                    unseen.remove(node)
                    component.append(node)
                    queue.append(node)
        potential[:, component] |= direct[:, component].any(1)[:, None]
    return dict(direct_report_nodes=direct, potential_only=potential & ~direct, remaining_nodes=~potential)


def read_metadata(data_dir, graph_dir, identity, best, baseline_info):
    protocol = identity['source_identity']['protocol']
    with (Path(data_dir)/'sensors.csv').open(encoding='utf-8-sig', newline='') as stream:
        stations = [int(row['station_id']) for row in csv.DictReader(stream)]
    require(len(stations) == protocol['stations'] and len(set(stations)) == len(stations), 'Invalid station axis')
    with np.load(Path(data_dir)/'incident_stats.npz', allow_pickle=False) as archive:
        require(bool(archive['normalized']), 'Expected normalized published release')
        mean, std = np.float32(archive['mean']), np.float32(archive['std'])
    require(np.isfinite(mean) and np.isfinite(std) and std > 0, 'Invalid published normalization')
    # The authenticated author format is an object array. Its entire payload is
    # deserialized, but only selected validation records are inspected below.
    raw = np.load(Path(data_dir)/'incident_all.npy', allow_pickle=True)
    require(raw.ndim == 1 and len(raw) == protocol['total_samples'], 'Published sample count/axis differs')
    start = int(len(raw)*.70)
    end = start+int(len(raw)*.15)
    require((start, end-start, len(raw)-end) == (protocol['train_samples'], protocol['val_samples'], protocol['test_samples']),
            'Published split differs')
    expected_end = min(end, start+4) if identity['engineering_check'] else end
    indices = best['sample_indices']
    require(np.array_equal(indices, np.arange(start, expected_end, dtype=np.int64)), 'Saved samples are not the full original validation axis')
    direct = np.zeros((len(indices), len(stations)), dtype=bool)
    for position, index in enumerate(indices):
        sample = raw[int(index)]
        require(isinstance(sample, dict), 'Invalid selected published record')
        y = np.asarray(sample['y_data'])
        require(y.shape == (12, len(stations), 3) and y.dtype == np.float32 and np.isfinite(y).all(), 'Invalid published target')
        restored = np.add(np.multiply(y[..., 0], std, dtype=np.float32), mean, dtype=np.float32)
        require(np.array_equal(restored, best['target'][position]), 'Saved targets differ from original validation targets')
        direct[position] = report_support(sample, len(stations))
    del raw
    graph = read_json(Path(graph_dir)/'graph.json')
    regions = partition_regions(direct, graph, stations)
    audit = dict(validation_rows=len(indices), stations=len(stations), engineering_only=identity['engineering_check'],
                 sample_axis='original_author_validation_order_full_or_original_engineering_prefix',
                 selected_original_targets_exact=True, test_payload_deserialized=True,
                 test_records_inspected=False, test_evaluated=False, raw_release_loaded_for='validation_target_and_report_mask_alignment',
                 direct_support='float32_abs_relation_sum_positive_and_report_present_and_relation_available',
                 potential_only='weak_internal_graph_component_envelope_excluding_direct_support_not_certified_propagation',
                 remaining_nodes='outside_this_candidate_envelope_not_a_natural_no_accident_control',
                 graph_evidence_scope=graph.get('evidence_scope', 'not_recorded'),
                 report_first_availability_verified=False, flow_unit='original_published_flow_scale_physical_unit_not_recertified')
    return regions, np.asarray(stations, dtype=np.int64), audit


def write_tables(output, analysis):
    fields = ('family', 'stratum', 'status', 'valid_count', 'valid_fraction_pooled', 'positive_target_count',
              'off_mae', 'on_mae', 'mae_gain', 'off_rmse', 'on_rmse', 'rmse_gain', 'off_mape_fraction',
              'on_mape_fraction', 'mape_gain_fraction', 'global_mae_gain_contribution', 'global_mape_gain_contribution',
              'gain_positive', 'gain_negative', 'cancellation_fraction', 'mean_OFF_signed_error', 'mean_ON_signed_error',
              'prediction_mean_on_minus_off', 'prediction_abs_mean', 'prediction_up_fraction', 'prediction_down_fraction',
              'mae_improved_fraction', 'mae_worsened_fraction')
    with (output/'strata.csv').open('w', encoding='utf-8-sig', newline='') as stream, (output/'horizons.csv').open('w', encoding='utf-8-sig', newline='') as horizons:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        hwriter = csv.DictWriter(horizons, fieldnames=('family', 'stratum', 'horizon', 'valid_count', 'mae_gain',
            'mape_gain_fraction', 'global_mae_gain_contribution', 'global_mape_gain_contribution'))
        hwriter.writeheader()
        for family, groups in [('global', {'all_nodes': analysis['global']})]+[(key, analysis[key]) for key in ('flow', 'bias', 'regions', 'cross')]:
            for name, group in groups.items():
                writer.writerow(dict(family=family, stratum=name, status=group['status'], valid_count=group['valid_count'],
                    valid_fraction_pooled=group['valid_fraction_pooled'], positive_target_count=group['positive_target_count'],
                    off_mae=group['off']['mae_pooled'], on_mae=group['on']['mae_pooled'], mae_gain=group['mae_gain_pooled'],
                    off_rmse=group['off']['rmse_pooled'], on_rmse=group['on']['rmse_pooled'], rmse_gain=group['rmse_gain_pooled'],
                    off_mape_fraction=group['off']['mape_pooled'], on_mape_fraction=group['on']['mape_pooled'], mape_gain_fraction=group['mape_gain_pooled'],
                    global_mae_gain_contribution=group['global_mae_gain_contribution'], global_mape_gain_contribution=group['global_mape_gain_contribution'],
                    gain_positive=group['gain_positive_pooled'], gain_negative=group['gain_negative_pooled'], cancellation_fraction=group['cancellation_fraction'],
                    mean_OFF_signed_error=group['mean_OFF_signed_error'], mean_ON_signed_error=group['mean_ON_signed_error'],
                    prediction_mean_on_minus_off=group['prediction_change']['signed_mean_on_minus_off'],
                    prediction_abs_mean=group['prediction_change']['absolute_mean'], prediction_up_fraction=group['prediction_change']['up_fraction'],
                    prediction_down_fraction=group['prediction_change']['down_fraction'], mae_improved_fraction=group['mae_response']['improved_fraction'],
                    mae_worsened_fraction=group['mae_response']['worsened_fraction']))
                for i, count in enumerate(group['valid_count_per_horizon']):
                    hwriter.writerow(dict(family=family, stratum=name, horizon=i+1, valid_count=count,
                        mae_gain=group['per_horizon_mae_gain'][i], mape_gain_fraction=group['per_horizon_mape_gain'][i],
                        global_mae_gain_contribution=group['per_horizon_global_mae_gain_contribution'][i],
                        global_mape_gain_contribution=group['per_horizon_global_mape_gain_contribution'][i]))


def fmt(value, *, percent=False):
    return '不可用' if value is None else f'{value*100:.6f}%' if percent else f'{value:.9f}'


def write_report(output, analysis, summary):
    lines = ['# 冻结 MIN 事故预算适配器：已有预测收益分解', '',
        f"源 MIN 最佳轮次：{summary['source_best_epoch']}；适配器最佳轮次：{summary['best_adapter_epoch']}；验证窗口：{summary['validation_rows']}。", '',
        '范围：'+('工程检查，仅验收分析流程，不代表正式预测效果。' if summary['engineering_only'] else '正式验证集上的探索性诊断。'), '',
        '仅分析已保存的 ON/OFF 预测。没有加载模型、重新推理、继续训练或评价测试集。已有制品一致性通过不等于新增检查点回放。', '',
        '收益统一为 OFF 误差 − ON 误差，正数表示开启适配器更好。MAE/RMSE 使用作者原流量尺度；物理单位未重新认证。MAPE 在报告中为百分数，在 JSON/CSV 中为比例。', '',
        '所有分组共享源有效掩码；MAPE 另外排除零目标。全网 MAPE 贡献使用全网正目标点数作共同分母，不能直接按 MAE 点数占比加权。当前原掩码是否已排除零值以保存记录为准，不重新纳入。', '',
        '流量桶由真实目标值按固定边界划分；高估/低估由 OFF − 真实目标固定划分。这些是真值可见的事后诊断，不能作为实际预测门控或输入。', '',
        '直接关联区域来自作者关系编码；候选范围仅是有直接关联节点的内部图弱连通分量，不能解释为真实事故传播范围。其余区域也不是已认证的无事故区域。', '']
    labels = {'global': '全网', 'flow': '真实流量区间', 'bias': '原 MIN 高估／低估', 'regions': '事故关联区域'}
    for family in ('global', 'flow', 'bias', 'regions'):
        groups = {'all_nodes': analysis['global']} if family == 'global' else analysis[family]
        lines.extend(['## '+labels[family], '',
            '| 分组 | 有效点数 | 全网点数占比 | OFF MAE | ON MAE | MAE收益 | OFF MAPE | ON MAPE | 全网MAE贡献 | 全网MAPE贡献（百分点） |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|'])
        for name, group in groups.items():
            mc = group['global_mape_gain_contribution']
            lines.append('| '+' | '.join((name, str(group['valid_count']), fmt(group['valid_fraction_pooled'], percent=True),
                fmt(group['off']['mae_pooled']), fmt(group['on']['mae_pooled']), fmt(group['mae_gain_pooled']),
                fmt(group['off']['mape_pooled'], percent=True), fmt(group['on']['mape_pooled'], percent=True),
                fmt(group['global_mae_gain_contribution']), fmt(None if mc is None else 100*mc)))+' |')
        lines.append('')
    lines.extend(['## 预测变化与误差抵消', '',
        '| 分组 | ON−OFF均值 | 绝对变化均值 | 预测上升点占比 | 预测下降点占比 | MAE改善点占比 | MAE变差点占比 | 正收益 | 负收益 | 抵消比例 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|'])
    for name, group in [('all_nodes', analysis['global'])]+list(analysis['bias'].items()):
        change, response = group['prediction_change'], group['mae_response']
        lines.append('| '+' | '.join((name, fmt(change['signed_mean_on_minus_off']), fmt(change['absolute_mean']),
            fmt(change['up_fraction'], percent=True), fmt(change['down_fraction'], percent=True),
            fmt(response['improved_fraction'], percent=True), fmt(response['worsened_fraction'], percent=True),
            fmt(group['gain_positive_pooled']), fmt(group['gain_negative_pooled']), fmt(group['cancellation_fraction'], percent=True)))+' |')
    lines.extend(['', '上升／下降占比按有效预测点计数；抵消比例按误差变化量计算，两者不能互换。', '',
        '## 流量 × 原 MIN 偏差 × 事故区域', '',
        '以下仅列非空交叉组；全部交叉组及空组状态见 analysis.json 和 strata.csv。', '',
        '| 流量区间 | OFF偏差 | 区域 | 点数 | MAE收益 | MAPE收益（百分点） | 全网MAE贡献 | 全网MAPE贡献（百分点） |',
        '|---|---|---|---:|---:|---:|---:|---:|'])
    for name, group in analysis['cross'].items():
        if not group['valid_count']:
            continue
        flow, bias, region = name.split('|')
        mg, mc = group['mape_gain_pooled'], group['global_mape_gain_contribution']
        lines.append('| '+' | '.join((flow, bias, region, str(group['valid_count']), fmt(group['mae_gain_pooled']),
            fmt(None if mg is None else mg*100), fmt(group['global_mae_gain_contribution']), fmt(None if mc is None else mc*100)))+' |')
    lines.extend(['', '## 解释边界与一致性', '',
        '各流量桶、偏差分组、事故三分区以及完整交叉分组分别互斥完备，其点数及全网 MAE/MAPE 贡献已核对守恒。局部组内 MAE 收益不能不加权相加；RMSE 不具备此线性贡献分解。', '',
        '流量分层能定位百分比误差代价，但不能单凭分组证明适配器使用了正确事故位置。适配器还读取主干内部状态；事故内容的独特贡献需要另设内容对照。', '',
        '本结果使用同一个验证集选出的最佳适配器，属于探索性诊断。窗口及预测点可能重叠，不按独立事故重复计算显著性，不自动建议新训练门控或扩大轮数。', '',
        f"固定困难窗口：{summary['fixed_hard_windows']}；已有配对MAE收益：{fmt(summary['fixed_hard_mae_gain'])}。没有重新选择困难集。", '',
        '数值明细：analysis.json；全部分组：strata.csv；逐预测步：horizons.csv；区域掩码：region_masks.npz；输入与代码来源：provenance.json；源文件保护：preservation.json。', ''])
    (output/'report.md').write_text('\n'.join(lines), encoding='utf-8')


def analyze_run(args):
    root, output = Path(args.run_dir), Path(args.output_dir)
    require(root.is_dir() and not output.exists(), 'Expected an existing source run and a new output directory')
    with readonly_run_lock(root):
        identity = read_json(root/'identity.json')
        require(identity.get('schema') == RUN_SCHEMA, 'Expected a frozen MIN pilot identity')
        source = identity['source_identity']
        require(source['package'] == source['protocol']['data_sha256'], 'Release hashes differ from the recorded protocol')
        paths = {'run/'+name: root/name for name in RUN_FILES}
        for family, directory, records in (('release', Path(args.data_dir), source['package']),
                                            ('graph', Path(args.graph_dir), source['graph_inputs_sha256'])):
            for name, digest in records.items():
                require(Path(name).name == name, 'Unsafe recorded input filename')
                path = directory/name
                require(path.is_file() and sha256(path) == digest, 'Recorded '+family+' input differs: '+name)
                paths[family+'/'+name] = path
        require(sha256(Path(args.graph_dir)/'graph.json') == source['protocol']['graph_sha256'], 'Protocol graph differs')
        for name in ('experiments/chronological/min_frozen_budget_metrics.py',):
            require(sha256(REPO/name) == identity['experiment_sources_sha256'][name], 'Original metric implementation changed')
            paths['original_code/'+name] = REPO/name
        for name in ('analyze_min_frozen_gain.py', 'min_frozen_gain_metrics.py', 'run_min_frozen_gain_analysis.sh'):
            paths['analysis_code/'+name] = HERE/name
        require(all(path.is_file() for path in paths.values()), 'A required saved input is missing')
        before = snapshot(paths)
        print('Checking saved ON/OFF archives and original selection certificates', flush=True)
        best, baseline = load_archives(root)
        summary, history, baseline_info = (read_json(root/name) for name in ('summary.json', 'history.json', 'baseline.json'))
        pair = verify_certificates(root, identity, summary, history, baseline_info, best, baseline)
        del baseline
        print('Matching original validation targets and report regions; no model calls', flush=True)
        regions, stations, metadata = read_metadata(args.data_dir, args.graph_dir, identity, best, baseline_info)
        analysis = summarize_strata(best['on'], best['off'], best['target'], best['valid'], regions, flow_edges=args.flow_edges)
        for key, paired_key in (('mae_gain_pooled', 'mae_gain_pooled'), ('rmse_gain_pooled', 'rmse_gain_pooled'), ('mape_gain_pooled', 'mape_gain_pooled')):
            assert_close(analysis['global'][key], pair['paired'][paired_key], 'reconstructed_global.'+key)
        require(snapshot(paths) == before, 'Source changed during analysis; refusing completed output')
        output.mkdir(parents=True, exist_ok=False)
        try:
            result = dict(status='FROZEN_MIN_GAIN_ANALYSIS_PASS', schema=SCHEMA,
                source_run=str(root), analysis_dir=str(output), source_best_epoch=summary['source_best_epoch'],
                best_adapter_epoch=summary['best_adapter_epoch'], validation_rows=len(best['target']),
                engineering_only=identity['engineering_check'], scope='exploratory_saved_validation_pair_diagnostics',
                global_metrics=analysis['global'], flow=analysis['flow'], bias=analysis['bias'], regions=analysis['regions'],
                fixed_hard_windows=len(best['fixed_hard_indices']), fixed_hard_mae_gain=pair['fixed_hard_mae_gain'],
                saved_artifact_consistency_passed=True, new_checkpoint_replay=False,
                new_inference_calls=0, training_updates=0, test_evaluated=False, source_artifacts_preserved=True,
                reconciliation=analysis['reconciliation'])
            analysis.update(source_scope=result['scope'], source_best_epoch=summary['source_best_epoch'],
                            best_adapter_epoch=summary['best_adapter_epoch'], metadata=metadata, original_pair=pair)
            write_json(output/'analysis.json', analysis)
            write_tables(output, analysis)
            write_report(output, analysis, result)
            np.savez_compressed(output/'region_masks.npz', sample_indices=best['sample_indices'], station_ids=stations, **regions)
            write_json(output/'provenance.json', dict(schema=SCHEMA, source_run=str(root), input_paths={key: str(path) for key, path in paths.items()},
                input_sha256=before, flow_edges=list(args.flow_edges), retrospective_grouping=True,
                checkpoint_policy='last.pt_hash_only_not_deserialized_or_independently_replayed', **metadata))
            after = snapshot(paths)
            require(before == after, 'Analysis modified a source artifact; refusing PASS')
            write_json(output/'preservation.json', dict(status='SOURCE_ARTIFACTS_PRESERVED', files=len(paths), before=before, after=after))
            write_json(output/'summary.json', result)
            return result
        except BaseException as error:
            write_json(output/'failure.json', dict(status='FROZEN_MIN_GAIN_ANALYSIS_FAILED', error=str(error),
                                                  source_artifacts_preserved=snapshot(paths) == before))
            raise


def main(argv=None):
    args = parse_args(argv)
    print('Source frozen run: '+str(args.run_dir), flush=True)
    print('Analysis directory: '+str(args.output_dir), flush=True)
    result = analyze_run(args)
    compact = {key: result[key] for key in ('status', 'source_best_epoch', 'best_adapter_epoch', 'validation_rows',
        'engineering_only', 'new_inference_calls', 'training_updates', 'source_artifacts_preserved', 'analysis_dir')}
    compact.update(global_metrics=result['global_metrics'], flow=result['flow'], bias=result['bias'], regions=result['regions'])
    # Console uses compact fields; full horizon/cross detail remains on disk.
    for family in ('global_metrics', 'flow', 'bias', 'regions'):
        groups = {'all_nodes': compact[family]} if family == 'global_metrics' else compact[family]
        compact[family] = {name: {key: group[key] for key in ('status', 'valid_count', 'valid_fraction_pooled',
            'mae_gain_pooled', 'mape_gain_pooled', 'global_mae_gain_contribution', 'global_mape_gain_contribution')}
            for name, group in groups.items()}
    print(json.dumps(compact, ensure_ascii=False, indent=2, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
