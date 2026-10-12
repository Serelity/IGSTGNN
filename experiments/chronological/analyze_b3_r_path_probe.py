"""Offline B3-R D-path gain decomposition using saved validation predictions only.

No model, checkpoint, training data, Torch, inference or optimizer is loaded.
OFF skips the entire D path; original TIID remains ON in the source probe.
"""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import platform
import uuid

import numpy as np


SCHEMA = 'b3r_gain_analysis_v1'
SOURCE_SCHEMA = 'b3r_d_path_probe_v1'
ATOL, RTOL = 1e-9, 1e-10
PARTITIONS = ('direct_report_nodes', 'potential_propagation_only_nodes', 'remaining_nodes')
ARRAY_KEYS = {'on', 'off', 'target', 'valid', 'sample_indices', 't0',
              'report_associated', 'candidate_potential', 'operator_mask', 'final_on_exactly_restored'}
SOURCE_FILES = ('summary.json', 'provenance.json', 'replay.json', 'preservation.json', 'paired_predictions.npz')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def snapshot(paths):
    return {str(path): sha256(path) for path in paths}


def close(actual, expected, label):
    if actual is None or expected is None:
        require(actual is None and expected is None, 'Unavailable metric mismatch: '+label)
        return
    if isinstance(actual, (list, tuple, np.ndarray)) or isinstance(expected, (list, tuple, np.ndarray)):
        require(isinstance(actual, (list, tuple, np.ndarray)) and isinstance(expected, (list, tuple, np.ndarray))
                and len(actual) == len(expected), 'Source metric axes mismatch: '+label)
        for i, (a, b) in enumerate(zip(actual, expected)):
            close(a, b, label+f'[{i}]')
        return
    a, b = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
    require(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
            and np.allclose(a, b, atol=ATOL, rtol=RTOL), 'Source metric mismatch: '+label)


def write_csv(path, rows):
    require(bool(rows), 'Cannot write empty table')
    with Path(path).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def verify_source(probe, allow_engineering):
    report, provenance, replay, preservation = [read_json(probe/name) for name in SOURCE_FILES[:4]]
    require(not (probe/'failure.json').exists(), 'Source probe has a failure record')
    require(report.get('status') == 'B3R_PATH_PROBE_PASS' and report.get('schema') == SOURCE_SCHEMA,
            'Source probe is incomplete or has the wrong schema')
    for key in ('source_artifacts_preserved', 'final_on_exactly_restored', 'inputs_preserved',
                'inference_only', 'tiid_report_enabled'):
        require(report.get(key) is True, 'Source invariant absent/failed: '+key)
    require(report.get('test_read') is False and provenance.get('test_read') is False,
            'Source is not validation-only')
    require(provenance.get('schema') == SOURCE_SCHEMA and report.get('best_epoch') == provenance.get('best_epoch')
            and isinstance(report.get('best_epoch'), int) and report['best_epoch'] > 0,
            'Source best-epoch provenance mismatch')
    identity = provenance.get('identity', {})
    engineering = identity.get('engineering_check')
    require(isinstance(engineering, bool), 'Missing source engineering scope')
    require(report.get('scope') == ('engineering_only' if engineering else 'conditional_offline_checkpoint_sensitivity'),
            'Source scientific scope mismatch')
    require(not engineering or allow_engineering, 'Engineering source requires --allow-engineering')
    require(report.get('replay') == replay and replay.get('status') == 'BEST_ON_WINDOW_REPLAY_PASS',
            'Source ON replay is missing or inconsistent')
    require(preservation.get('source_preserved') is True
            and isinstance(preservation.get('before'), dict)
            and bool(preservation['before'])
            and preservation['before'] == preservation.get('after')
            and isinstance(preservation.get('graph_before'), dict)
            and bool(preservation['graph_before'])
            and preservation['graph_before'] == preservation.get('graph_after'),
            'Source preservation audit failed')
    require(provenance.get('saved_files_sha256') == preservation['before'], 'Source saved-file provenance mismatch')
    state_hash = provenance.get('best_model_state_sha256')
    require(isinstance(state_hash, str) and bool(state_hash) and report.get('model_state_sha256') == state_hash,
            'Source best-model state hash mismatch')
    return report, provenance, replay


def load_arrays(probe, report, provenance):
    with np.load(probe/'paired_predictions.npz', allow_pickle=False) as packed:
        require(set(packed.files) == ARRAY_KEYS, 'Unexpected paired-prediction schema')
        arrays = {key: packed[key] for key in packed.files}
    on, off, target, valid = [arrays[key] for key in ('on', 'off', 'target', 'valid')]
    require(on.ndim == 4 and on.shape == off.shape == target.shape == valid.shape
            and on.shape[0] > 0 and on.shape[1] == 12 and on.shape[2] > 0 and on.shape[3] == 1,
            'Expected equal [rows,12,nodes,1] prediction/target/valid axes')
    require(valid.dtype == np.bool_ and all(a.dtype.kind == 'f' for a in (on, off, target))
            and np.isfinite(on).all() and np.isfinite(off).all() and np.isfinite(target[valid]).all(),
            'Invalid prediction, valid target or boolean mask')
    rows, _, nodes, _ = on.shape
    require(provenance['identity']['engineering_check'] or (rows == 917 and nodes == 496),
            'Full validation must contain 917 rows and 496 nodes')
    indices, t0 = arrays['sample_indices'], arrays['t0']
    require(indices.shape == (rows,) and indices.dtype.kind in 'iu' and len(np.unique(indices)) == rows,
            'Invalid sample identity axis')
    require(t0.shape == (rows,) and t0.dtype.kind == 'U', 'Invalid cutoff time axis')
    for value in t0:
        require(bool(value), 'Empty cutoff time')
        datetime.fromisoformat(str(value))
    for key in ('report_associated', 'candidate_potential'):
        require(arrays[key].shape == (rows, nodes) and arrays[key].dtype == np.bool_, 'Invalid region axes: '+key)
    operator = arrays['operator_mask']
    require(operator.shape == (nodes,) and operator.dtype == np.bool_, 'Invalid operator mask axis')
    require(not (arrays['candidate_potential'] & ~operator[None]).any(), 'Candidate envelope extends outside operator')
    restored = arrays['final_on_exactly_restored']
    require(restored.shape == () and restored.dtype == np.bool_ and bool(restored), 'Final ON not exactly restored')
    batch_size = provenance['identity'].get('batch_size')
    require(isinstance(batch_size, int) and batch_size > 0, 'Invalid original batch size')
    expected_partition = [min(batch_size, rows-i) for i in range(0, rows, batch_size)]
    require(report.get('batch_partition') == expected_partition, 'Source batch partition differs from rows')
    require((valid.sum((0, 2, 3)) > 0).all(), 'Missing globally valid forecast horizon')
    return arrays


def region_masks(arrays):
    direct, potential = arrays['report_associated'], arrays['candidate_potential']
    operator = arrays['operator_mask'][None]
    masks = dict(all_nodes=np.ones_like(direct), direct_report_nodes=direct,
                 potential_propagation_only_nodes=potential & ~direct,
                 remaining_nodes=~(direct | potential), candidate_potential=potential,
                 direct_outside_operator=direct & ~operator,
                 outside_operator=np.broadcast_to(~operator, direct.shape),
                 no_report_samples=np.broadcast_to(~direct.any(1)[:, None], direct.shape))
    require(np.all(sum(masks[key].astype(np.int8) for key in PARTITIONS) == 1), 'Regions are not an exact partition')
    return masks


def raw_statistics(errors, valid, mask):
    """Retain row x horizon sums; masks may select nodes or individual horizons."""
    mask = np.asarray(mask)
    if mask.ndim == 2:
        mask = mask[:, None, :, None]
    require(mask.dtype == np.bool_, 'Region mask must be boolean')
    selected = valid & np.broadcast_to(mask, valid.shape)
    result = {'count': selected.sum((2, 3))}
    for name, value in errors.items():
        result[name] = np.where(selected, value, 0.).sum((2, 3))
    result['maximum'] = np.where(selected, errors['change'], 0.).max((1, 2, 3))
    return result


def aggregate(raw, global_counts, indices=None, weights=None):
    indices = np.arange(len(raw['count'])) if indices is None else np.asarray(indices)
    w = np.ones(len(indices)) if weights is None else np.asarray(weights)[indices]
    require(w.shape == (len(indices),) and np.isfinite(w).all() and (w > 0).all(), 'Invalid row weights')
    sums = {key: (value[indices]*w[:, None]).sum(0) for key, value in raw.items() if key != 'maximum'}
    counts = sums['count']
    global_counts = np.asarray(global_counts, dtype=np.float64)
    require(global_counts.shape == counts.shape and (global_counts > 0).all(), 'Invalid global denominators')
    available = bool((counts > 0).all())
    def per_horizon(value):
        return [float(x/n) if n else None for x, n in zip(value, counts)]
    def macro(value):
        return float(np.mean(value/counts)) if available else None
    def contribution(value):
        return float(np.mean(value/global_counts))
    gain, positive, negative = [macro(sums[key]) for key in ('gain', 'positive', 'negative')]
    gross = None if positive is None else positive+negative
    global_positive, global_negative = [contribution(sums[key]) for key in ('positive', 'negative')]
    global_gross = global_positive+global_negative
    global_gain = contribution(sums['gain'])
    require(np.all(sums['positive']+sums['negative'] <= sums['change']+ATOL+RTOL*sums['change']),
            'Absolute-error triangle bound failed')
    close(global_gain, global_positive-global_negative, 'positive-negative identity')
    return dict(region_metric_available=available, on_mae_macro=macro(sums['on_error']),
        off_mae_macro=macro(sums['off_error']), off_minus_on_mae=gain,
        gain_positive=positive, gain_negative=negative, gross_error_gain=gross,
        cancellation_fraction=None if gross is None or gross == 0 else max(0., 1.-abs(gain)/gross),
        relative_gain_percent=None if not available or macro(sums['off_error']) == 0 else 100.*gain/macro(sums['off_error']),
        global_gain_contribution=global_gain, global_gain_positive=global_positive,
        global_gain_negative=global_negative, global_gross_error_gain=global_gross,
        global_cancellation_fraction=None if global_gross == 0 else max(0., 1.-abs(global_gain)/global_gross),
        global_valid_weight_fraction=contribution(counts),
        global_prediction_abs_change_contribution=contribution(sums['change']),
        per_horizon_global_gain_contribution=(sums['gain']/global_counts).tolist(),
        per_horizon_global_gain_positive=(sums['positive']/global_counts).tolist(),
        per_horizon_global_gain_negative=(sums['negative']/global_counts).tolist(),
        per_horizon_global_valid_weight_fraction=(counts/global_counts).tolist(),
        per_horizon_valid_count=raw['count'][indices].sum(0).astype(int).tolist(),
        per_horizon_weighted_count=counts.tolist(), valid_count=int(raw['count'][indices].sum()),
        per_horizon_on_mae=per_horizon(sums['on_error']), per_horizon_off_mae=per_horizon(sums['off_error']),
        per_horizon_off_minus_on=per_horizon(sums['gain']),
        per_horizon_gain_positive=per_horizon(sums['positive']), per_horizon_gain_negative=per_horizon(sums['negative']),
        per_horizon_prediction_abs_change_mean=per_horizon(sums['change']),
        prediction_abs_change_horizon_macro_mean=macro(sums['change']),
        prediction_abs_change_pooled_mean=None if counts.sum() == 0 else float(sums['change'].sum()/counts.sum()),
        prediction_signed_change_horizon_macro_mean=macro(sums['signed_change']),
        prediction_abs_change_max=None if raw['count'][indices].sum() == 0 else float(raw['maximum'][indices].max()))


def verify_metrics(report, replay, metrics):
    aliases = dict(all_nodes='all_nodes', report_associated='direct_report_nodes',
                   candidate_potential='candidate_potential', potential_only='potential_propagation_only_nodes',
                   direct_outside_operator='direct_outside_operator', outside_operator='outside_operator',
                   no_report_samples='no_report_samples')
    saved_metrics = report.get('metrics', {})
    require('all_nodes' in saved_metrics, 'Missing source all_nodes metrics')
    mapped = dict(counts='per_horizon_valid_count', on_mae_by_horizon='per_horizon_on_mae',
                  off_mae_by_horizon='per_horizon_off_mae', off_minus_on_by_horizon='per_horizon_off_minus_on',
                  on_mae_macro='on_mae_macro', off_mae_macro='off_mae_macro', off_minus_on_mae='off_minus_on_mae')
    for name, key in aliases.items():
        if name not in saved_metrics:
            continue
        actual, saved = metrics[key], saved_metrics[name]
        require(saved.get('status') == ('available' if actual['region_metric_available'] else 'unavailable'),
                'Source region availability mismatch: '+name)
        for original, current in mapped.items():
            close(actual[current], saved.get(original), name+'.'+original)
        change = saved.get('prediction_change', {})
        if change:
            close(actual['valid_count'], change['count'], name+'.prediction count')
            close(actual['prediction_abs_change_pooled_mean'], change['absolute_mean'], name+'.pooled prediction change')
            close(actual['prediction_abs_change_max'], change['absolute_max'], name+'.maximum prediction change')
    close(metrics['all_nodes']['on_mae_macro'], replay.get('on_mae_macro'), 'ON replay MAE')
    saved_mae = replay.get('saved_mae_macro')
    require(isinstance(saved_mae, (int, float)) and np.isfinite(saved_mae)
            and abs(metrics['all_nodes']['on_mae_macro']-saved_mae) <= 1e-4, 'ON differs from saved best metric')


def check_partition(metrics, keys, total):
    for field in ('global_gain_contribution', 'global_gain_positive', 'global_gain_negative',
                  'global_valid_weight_fraction', 'global_prediction_abs_change_contribution'):
        close(sum(metrics[key][field] for key in keys), total[field], 'Partition sum: '+field)


def decomposition_rows(metrics, category):
    rows = []
    for name, metric in metrics.items():
        rows.append(dict(category=category, group=name, horizon='macro', valid_count=metric['valid_count'],
                         on_mae=metric['on_mae_macro'], off_mae=metric['off_mae_macro'],
                         off_minus_on=metric['off_minus_on_mae'], gain_positive=metric['gain_positive'],
                         gain_negative=metric['gain_negative'], cancellation_fraction=metric['cancellation_fraction'],
                         contribution_to_global_macro_mae=metric['global_gain_contribution'],
                         prediction_abs_change=metric['prediction_abs_change_horizon_macro_mean']))
        for h in range(12):
            p, n = metric['per_horizon_gain_positive'][h], metric['per_horizon_gain_negative'][h]
            gross = None if p is None else p+n
            gain = metric['per_horizon_off_minus_on'][h]
            rows.append(dict(category=category, group=name, horizon=h+1,
                             valid_count=metric['per_horizon_valid_count'][h],
                             on_mae=metric['per_horizon_on_mae'][h], off_mae=metric['per_horizon_off_mae'][h],
                             off_minus_on=gain, gain_positive=p, gain_negative=n,
                             cancellation_fraction=None if gross is None or gross == 0 else max(0., 1.-abs(gain)/gross),
                             contribution_to_global_macro_mae=metric['per_horizon_global_gain_contribution'][h]/12.,
                             prediction_abs_change=metric['per_horizon_prediction_abs_change_mean'][h]))
    return rows


TABLE_FIELDS = ('valid_count', 'on_mae_macro', 'off_mae_macro', 'off_minus_on_mae',
                'gain_positive', 'gain_negative', 'cancellation_fraction', 'global_gain_contribution',
                'global_gain_positive', 'global_gain_negative', 'global_valid_weight_fraction',
                'prediction_abs_change_horizon_macro_mean', 'prediction_abs_change_max')


def sample_tables(arrays, raw, global_counts):
    windows, groups = [], defaultdict(list)
    for i, (sample, cutoff) in enumerate(zip(arrays['sample_indices'], arrays['t0'])):
        groups[str(cutoff)].append(i)
        metric = aggregate(raw['all_nodes'], global_counts, [i])
        row = dict(sample_index=int(sample), t0=str(cutoff),
                   direct_nodes=int(arrays['report_associated'][i].sum()),
                   candidate_potential_nodes=int(arrays['candidate_potential'][i].sum()))
        row.update({key: metric[key] for key in TABLE_FIELDS})
        for name in PARTITIONS:
            row[name+'_global_contribution'] = aggregate(raw[name], global_counts, [i])['global_gain_contribution']
        windows.append(row)
    cutoffs = []
    for cutoff, indices in sorted(groups.items()):
        metric = aggregate(raw['all_nodes'], global_counts, indices)
        row = dict(t0=cutoff, rows=len(indices))
        row.update({key: metric[key] for key in TABLE_FIELDS})
        for name in PARTITIONS:
            row[name+'_global_contribution'] = aggregate(raw[name], global_counts, indices)['global_gain_contribution']
        cutoffs.append(row)
    close(sum(row['global_gain_contribution'] for row in windows),
          aggregate(raw['all_nodes'], global_counts)['global_gain_contribution'], 'Window contributions')
    close(sum(row['global_gain_contribution'] for row in cutoffs),
          aggregate(raw['all_nodes'], global_counts)['global_gain_contribution'], 'Cutoff contributions')
    return windows, cutoffs


def explain(result, windows):
    m = result['metrics']['all_nodes']
    value = lambda x: '不可用' if x is None else f'{x:.12g}'
    lines = ['# B3-R D路径收益分解', '',
             '范围：'+result['source_scope']+'。仅分析已有验证预测；没有新增训练或推理。', '',
             f"最佳轮次：{result['source_best_epoch']}；验证行数：{result['validation_rows']}。",
             f"ON MAE={value(m['on_mae_macro'])}；OFF MAE={value(m['off_mae_macro'])}；OFF−ON={value(m['off_minus_on_mae'])}。",
             f"正收益G+={value(m['gain_positive'])}；负收益G−={value(m['gain_negative'])}；抵消率={value(m['cancellation_fraction'])}。",
             f"误差变化总量G++G−={value(m['gross_error_gain'])}；预测绝对变化宏平均={value(m['prediction_abs_change_horizon_macro_mean'])}。", '',
             '| 互斥区域 | 组内OFF−ON | 全网贡献 | 全网权重占比 |', '|---|---:|---:|---:|']
    for key in PARTITIONS:
        row = result['metrics'][key]
        lines.append(f"| {key} | {value(row['off_minus_on_mae'])} | {value(row['global_gain_contribution'])} | {value(row['global_valid_weight_fraction'])} |")
    lines += ['', '## 指标定义', '',
              '- g=|OFF−y|−|ON−y|，正值表示开启D后的绝对误差更小。',
              '- 主指标保留全部原验证行：每个有效元素权重为1/(12×该时距全网有效元素数)。',
              '- G+=sum(w×max(g,0))，G−=sum(w×max(−g,0))，净收益=G+−G−。',
              '- 抵消率=1−|G+−G−|/(G++G−)；总误差变化为0时不可用，不能以预测变化替代分母。',
              '- 组内指标按该组各时距有效数归一化；全网贡献使用全网分母。空组贡献为0、组内MAE不可用。',
              '- 三个互斥区域为A=直接关联、B=候选包络且非A、C=其余；部分A位于算子之外，仍属于A。',
              '- 诊断组可能重叠，不能再次相加。候选包络不等于实测事故影响范围。',
              '- bias按OFF预测相对真实值的高估/低估/精确命中分组；OFF并非原始B0。',
              '- movement的crossing仅含严格异号残差；恰好命中目标另列。对称跨越可能改变预测但不改变绝对误差。',
              '- CSV的contribution_to_global_macro_mae在单时距行已除以12，12行相加等于对应macro行。',
              '- 唯一截止点敏感性用每行1/同t0行数的权重重新计算每时距分母；主结果没有去重。', '',
              '## 解释边界', '',
              'OFF关闭整个历史/报告条件D路径，原TIID仍开启。结果是同一检查点的路径敏感性，不是事故信息的纯因果收益，也不是OFF从头训练结果。',
              '重叠窗口、站点及时距不视为独立重复；本工具不做显著性检验，不自动选择动态容量或软化约束。',
              '源NPZ未保存站点ID、道路及逐边/逐窗口交换轨迹，本工具不推断道路身份或个体容量→交换→收益归因。',
              '旧探针未保存NPZ的完成时哈希；本分析记录读入和结束哈希并复核源指标，只证明本次保全及数值一致性。', '',
              '## 代表窗口（按全网净收益贡献排序，改善/恶化各最多5个）', '']
    for title, positive in (('改善', True), ('恶化', False)):
        chosen = [row for row in windows if (row['global_gain_contribution'] > 0 if positive else row['global_gain_contribution'] < 0)]
        chosen.sort(key=lambda row: row['global_gain_contribution'], reverse=positive)
        lines.append(title+'：'+('；'.join(f"sample={row['sample_index']}，t0={row['t0']}，贡献={row['global_gain_contribution']:.8g}" for row in chosen[:5]) or '无'))
    return '\n'.join(lines)+'\n'


def analyze(probe, output, allow_engineering=False):
    probe, output = Path(probe).resolve(), Path(output).resolve()
    require(probe.is_dir(), 'Probe directory does not exist')
    require(output != probe and not output.is_relative_to(probe) and not probe.is_relative_to(output),
            'Output overlaps the protected probe directory')
    output.mkdir(parents=True, exist_ok=False)
    before = {}
    try:
        paths = [probe/name for name in SOURCE_FILES]
        before = snapshot(paths)
        report, provenance, replay = verify_source(probe, allow_engineering)
        arrays = load_arrays(probe, report, provenance)
        # Invalid targets may be NaN. Remove them before residual arithmetic.
        valid = arrays['valid']
        target = np.where(valid, arrays['target'], 0.).astype(np.float64)
        on_residual, off_residual = [arrays[key].astype(np.float64)-target for key in ('on', 'off')]
        on_error, off_error = np.abs(on_residual), np.abs(off_residual)
        gain = off_error-on_error
        errors = dict(on_error=on_error, off_error=off_error, gain=gain,
                      positive=np.maximum(gain, 0.), negative=np.maximum(-gain, 0.),
                      change=np.abs(arrays['off'].astype(np.float64)-arrays['on']),
                      signed_change=arrays['off'].astype(np.float64)-arrays['on'])
        global_counts = valid.sum((0, 2, 3))
        masks = region_masks(arrays)
        raw = {key: raw_statistics(errors, valid, mask) for key, mask in masks.items()}
        metrics = {key: aggregate(value, global_counts) for key, value in raw.items()}
        verify_metrics(report, replay, metrics)
        check_partition(metrics, PARTITIONS, metrics['all_nodes'])
        bias_masks = dict(over=off_residual > 0, under=off_residual < 0, exact=off_residual == 0)
        movement_masks = dict(same_side=((on_residual > 0) & (off_residual > 0)) | ((on_residual < 0) & (off_residual < 0)),
                              crossing=((on_residual > 0) & (off_residual < 0)) | ((on_residual < 0) & (off_residual > 0)),
                              off_at_target=(off_residual == 0) & (on_residual != 0),
                              on_at_target=(on_residual == 0) & (off_residual != 0),
                              both_at_target=(on_residual == 0) & (off_residual == 0))
        families = {}
        for family, chosen in (('bias_metrics', bias_masks), ('movement_metrics', movement_masks)):
            require(np.all(sum(mask.astype(np.int8) for mask in chosen.values())[valid] == 1),
                    'Incomplete residual partition: '+family)
            families[family] = {key: aggregate(raw_statistics(errors, valid, mask), global_counts) for key, mask in chosen.items()}
            check_partition(families[family], chosen, metrics['all_nodes'])
        multiplicity = Counter(str(t) for t in arrays['t0'])
        cutoff_weights = np.array([1./multiplicity[str(t)] for t in arrays['t0']])
        weighted_counts = (raw['all_nodes']['count']*cutoff_weights[:, None]).sum(0)
        sensitivity = {key: aggregate(value, weighted_counts, weights=cutoff_weights) for key, value in raw.items()}
        check_partition(sensitivity, PARTITIONS, sensitivity['all_nodes'])
        windows, cutoffs = sample_tables(arrays, raw, global_counts)
        result = dict(status='B3R_GAIN_ANALYSIS_PASS', schema=SCHEMA, source_probe=str(probe),
                      source_scope=report['scope'], source_best_epoch=report['best_epoch'],
                      validation_rows=len(arrays['sample_indices']), nodes=arrays['on'].shape[2],
                      distinct_cutoffs=len(cutoffs), metric_sign='OFF_minus_ON_positive_means_D_helpful',
                      weighting='original_rows_horizon_macro_1_over_12N_h', partitions=list(PARTITIONS),
                      metrics=metrics, unique_cutoff_sensitivity_metrics=sensitivity, **families,
                      new_inference_calls=0, optimizer_updates=0, test_accessed=False,
                      predictive_gain_claim=False, accident_only_attribution=False,
                      source_artifacts_preserved=True,
                      original_artifact_hash_manifest_available=False,
                      source_sha256=before, analysis_source_sha256=sha256(Path(__file__)),
                      environment=dict(python=platform.python_version(), numpy=np.__version__))
        write_csv(output/'gain_decomposition.csv', decomposition_rows(metrics, 'region'))
        write_csv(output/'bias_decomposition.csv', decomposition_rows(families['bias_metrics'], 'OFF_residual'))
        write_csv(output/'movement_decomposition.csv', decomposition_rows(families['movement_metrics'], 'residual_movement'))
        write_csv(output/'window_summary.csv', windows)
        write_csv(output/'cutoff_summary.csv', cutoffs)
        (output/'README.md').write_text(explain(result, windows), encoding='utf-8')
        after = snapshot(paths)
        require(before == after, 'Source artifacts changed during analysis; completion blocked')
        write_json(output/'preservation.json', dict(before=before, after=after, source_artifacts_preserved=True))
        result['outputs_sha256'] = {p.name: sha256(p) for p in sorted(output.iterdir())}
        write_json(output/'analysis.json', result)
        return result
    except BaseException as error:
        write_json(output/'failure.json', dict(status='B3R_GAIN_ANALYSIS_FAILED', completion_claim=False,
                   error_type=type(error).__name__, message=str(error), new_inference_calls=0,
                   optimizer_updates=0, test_accessed=False, source_sha256_before=before))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--allow-engineering', action='store_true', help='Allow explicitly marked engineering fixtures only')
    args = parser.parse_args(argv)
    probe = args.probe_dir.resolve()
    output = args.output_dir or probe.parent/('b3r_gain_analysis_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6])
    print('Source probe: '+str(probe), flush=True)
    print('Analysis directory: '+str(output.resolve()), flush=True)
    result = analyze(probe, output, args.allow_engineering)
    print(json.dumps(dict(status=result['status'], source_best_epoch=result['source_best_epoch'],
                          source_scope=result['source_scope'], validation_rows=result['validation_rows'],
                          all_nodes={key: result['metrics']['all_nodes'][key] for key in
                                     ('on_mae_macro', 'off_mae_macro', 'off_minus_on_mae', 'gain_positive', 'gain_negative',
                                      'gross_error_gain', 'cancellation_fraction', 'prediction_abs_change_horizon_macro_mean')},
                          partition_contributions={key: result['metrics'][key]['global_gain_contribution'] for key in PARTITIONS},
                          new_inference_calls=0, source_artifacts_preserved=True,
                          analysis=str(output.resolve()/'analysis.json')), ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
