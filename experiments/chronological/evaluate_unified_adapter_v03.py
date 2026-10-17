"""Read-only paired analysis of v0.3 saved predictions; NumPy only.

An archive is a *single* shared target/mask/axis snapshot.  It never selects a
different mask for a case, silently transposes axes, fits on validation labels,
loads a checkpoint, or treats overlapping points as independent experiments.
All gains are control error minus candidate error. MAPE is a fraction in JSON.
"""
from argparse import ArgumentParser
from hashlib import sha256
from io import BytesIO
import csv
import json
import math
from pathlib import Path

import numpy as np


ARCHIVE_SCHEMA = 'igstgnn_unified_adapter_archive_v03'
SCHEMA = 'igstgnn_unified_adapter_evaluation_v03'
AXIS_ORDER = 'sample,horizon,node'
LEVELS = np.array([.1, .5, .9], dtype=np.float64)
FLOW_EDGES = (0., 5., 20., 50., 100., 200., float('inf'))
PAIRS = (
    ('TRAFFIC_CORE_vs_ORIGINAL', 'ORIGINAL', 'TRAFFIC_CORE'),
    ('H_ONLY_vs_TRAFFIC_CORE', 'TRAFFIC_CORE', 'H_ONLY'),
    ('R_ONLY_vs_H_ONLY', 'H_ONLY', 'R_ONLY'),
    ('BOTH_vs_R_ONLY', 'R_ONLY', 'BOTH'),
    ('BOTH_vs_H_ONLY', 'H_ONLY', 'BOTH'),
    ('BOTH_vs_TRAFFIC_CORE', 'TRAFFIC_CORE', 'BOTH'),
    ('BOTH_vs_ORIGINAL', 'ORIGINAL', 'BOTH'),
    ('BOTH_vs_DIRECT', 'DIRECT', 'BOTH'),
    ('time_dependency', 'BOTH_TIME_OFF', 'BOTH'),
    ('graph_dependency', 'BOTH_GRAPH_OFF', 'BOTH'),
    ('report_dependency', 'BOTH_REPORT_OFF', 'BOTH'),
    ('semantic_UNK_sensitivity', 'BOTH_SEMANTIC_UNK', 'BOTH'),
    ('semantic_shuffle_sensitivity', 'BOTH_SEMANTIC_SHUFFLE', 'BOTH'),
    ('ALWAYS_CANDIDATE_vs_ALWAYS_BASE', 'ALWAYS_BASE', 'ALWAYS_CANDIDATE'),
    ('HALF_CORRECTION_vs_ALWAYS_BASE', 'ALWAYS_BASE', 'HALF_CORRECTION'),
    ('GLOBAL_CALIBRATION_vs_ALWAYS_BASE', 'ALWAYS_BASE', 'GLOBAL_CALIBRATION'),
    ('LEARNED_vs_ALWAYS_BASE', 'ALWAYS_BASE', 'LEARNED'),
    ('RANDOM_MATCHED_vs_ALWAYS_BASE', 'ALWAYS_BASE', 'RANDOM_MATCHED'),
    ('ORACLE_UNDEPLOYABLE_vs_ALWAYS_BASE', 'ALWAYS_BASE', 'ORACLE_UNDEPLOYABLE'),
    ('LEARNED_vs_HALF_CORRECTION', 'HALF_CORRECTION', 'LEARNED'),
    ('LEARNED_vs_GLOBAL_CALIBRATION', 'GLOBAL_CALIBRATION', 'LEARNED'),
    ('LEARNED_vs_RANDOM_MATCHED', 'RANDOM_MATCHED', 'LEARNED'),
    ('RISK_H_vs_ORIGINAL', 'ORIGINAL', 'RISK_H'),
    ('RISK_H_R_vs_RISK_H_same_center', 'RISK_H', 'RISK_H_R'),
    ('RISK_H_R_C_vs_RISK_H_R_same_center', 'RISK_H_R', 'RISK_H_R_C'),
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def array_hash(value):
    value = np.ascontiguousarray(value)
    return sha256(str(value.dtype).encode()+str(value.shape).encode()+value.tobytes()).hexdigest()


def _text(value, name):
    value = np.asarray(value)
    require(value.ndim == 0 and value.dtype.kind in 'US', name+' must be a scalar string')
    return str(value.item().decode('utf-8') if value.dtype.kind == 'S' else value.item())


def _ids(value, size, name):
    value = np.asarray(value)
    require(value.shape == (size,) and value.dtype.kind in 'iuUS' and len(np.unique(value)) == size,
            name+' must identify the axis uniquely, without object arrays')
    return value


def _mask(value, rows, nodes, name):
    value = np.asarray(value)
    require(value.dtype == np.bool_ and value.shape in ((nodes,), (rows, nodes)),
            name+' must be Boolean [node] or [sample,node]')
    return np.broadcast_to(value, (rows, nodes))


def validate_archive(archive):
    """Accept a mapping or NPZ path, with exact explicit sample/horizon/node axes."""
    source_hash = None
    if isinstance(archive, (str, Path)):
        snapshot = Path(archive).read_bytes()
        source_hash = sha256(snapshot).hexdigest()
        with np.load(BytesIO(snapshot), allow_pickle=False) as loaded:
            archive = {key: loaded[key] for key in loaded.files}
    require(isinstance(archive, dict), 'archive must be a mapping or NPZ path')
    require(_text(archive.get('schema'), 'schema') == ARCHIVE_SCHEMA, 'Unsupported archive schema')
    require(_text(archive.get('axis_order'), 'axis_order') == AXIS_ORDER, 'Axes must be explicit; no transpose inference')
    y, valid = np.asarray(archive.get('y')), np.asarray(archive.get('valid'))
    require(y.ndim == 3 and y.shape[1] == 12 and min(y.shape) > 0 and y.dtype.kind in 'fiu',
            'y must be numeric [sample,12,node]')
    require(valid.shape == y.shape and valid.dtype == np.bool_ and valid.any(),
            'valid must be the saved shared Boolean target mask')
    require(np.isfinite(y[valid]).all() and (y[valid] > 0).all(),
            'Original masked MAPE requires finite strictly positive valid targets')
    rows, _, nodes = y.shape
    sample_ids = _ids(archive.get('sample_ids'), rows, 'sample_ids')
    node_ids = _ids(archive.get('node_ids'), nodes, 'node_ids')
    direct = np.asarray(archive.get('direct_support'))
    require(direct.shape == (rows, nodes) and direct.dtype == np.bool_,
            'direct_support must retain the original Boolean [sample,node] support')
    predictions = {key[6:]: np.asarray(value) for key, value in archive.items() if key.startswith('pred__')}
    require('ORIGINAL' in predictions and all(predictions), 'ORIGINAL prediction is required for the fixed hard set')
    for name, prediction in predictions.items():
        require(prediction.shape == y.shape and prediction.dtype.kind in 'fiu'
                and np.isfinite(prediction[valid]).all(),
                name+': predictions must follow the same axes and be finite on every valid point')
    groups = {'all_nodes': np.ones((rows, nodes), bool), 'direct_report_nodes': direct,
              'outside_direct_support': ~direct}
    for key, value in archive.items():
        if key.startswith('group__'):
            name = key[7:]
            require(bool(name) and name not in groups, 'Group names must be unique and nonempty')
            groups[name] = _mask(value, rows, nodes, key)
    if 'road_labels' in archive:
        labels = np.asarray(archive['road_labels'])
        require(labels.shape == (nodes,) and labels.dtype.kind in 'US', 'road_labels must follow the node axis')
        for label in np.unique(labels):
            name = 'road:'+str(label)
            require(name not in groups, 'Duplicate road group')
            groups[name] = np.broadcast_to(labels == label, (rows, nodes))
    if 'candidate_potential' in groups:
        potential = groups['candidate_potential']
        require((~direct | potential).all(), 'candidate_potential must contain direct support')
        for name, value in [('potential_only', potential & ~direct), ('remaining_nodes', ~potential)]:
            require(name not in groups, 'Do not supply duplicate derived region '+name)
            groups[name] = value
    quantiles = {key[3:]: np.asarray(value) for key, value in archive.items() if key.startswith('q__')}
    if quantiles:
        levels = np.asarray(archive.get('quantile_levels'))
        require(levels.shape == (3,) and levels.dtype.kind in 'fiu' and np.allclose(levels, LEVELS, rtol=0., atol=1e-7),
                'Only fixed quantiles .1/.5/.9 are supported')
    for name, quantile in quantiles.items():
        require(name in predictions and quantile.shape == (*y.shape, 3) and quantile.dtype.kind in 'fiu'
                and np.isfinite(quantile[valid]).all(), name+': quantile axes or values disagree')
        require((np.diff(quantile[valid], axis=-1) >= 0).all(), name+': quantile crossing is not permitted')
        require(np.allclose(quantile[..., 1][valid], predictions[name][valid], rtol=1e-6, atol=1e-6),
                name+': median must match the saved point forecast')
    return dict(y=y, valid=valid, sample_ids=sample_ids, node_ids=node_ids,
                predictions=predictions, quantiles=quantiles, groups=groups,
                source_sha256=source_hash, axis_hashes={key: array_hash(value) for key, value in
                    [('y', y), ('valid', valid), ('sample_ids', sample_ids), ('node_ids', node_ids), ('direct_support', direct)]})


def _ratio(total, count):
    return float(total/count) if count else None


def _macro(values):
    return float(np.mean(values)) if all(value is not None for value in values) else None


def point_metrics(prediction, y, valid):
    """Pooled valid-point and twelve-horizon equal-weight metrics."""
    horizons = []
    totals = np.zeros(3, np.float64)
    count = 0
    for h in range(y.shape[1]):
        mask = valid[:, h]
        target = y[:, h][mask].astype(np.float64)
        error = prediction[:, h][mask].astype(np.float64)-target
        n = int(mask.sum())
        sums = np.array([np.abs(error).sum(), np.square(error).sum(), (np.abs(error)/target).sum()])
        totals += sums
        count += n
        horizons.append(dict(step=h+1, valid_count=n, mae=_ratio(sums[0], n),
                             rmse=None if not n else float(np.sqrt(sums[1]/n)), mape=_ratio(sums[2], n)))
    return dict(valid_count=count, mae_pooled=_ratio(totals[0], count),
                rmse_pooled=None if not count else float(np.sqrt(totals[1]/count)),
                mape_pooled=_ratio(totals[2], count), mae_macro=_macro([r['mae'] for r in horizons]),
                rmse_macro=_macro([r['rmse'] for r in horizons]), mape_macro=_macro([r['mape'] for r in horizons]),
                horizons=horizons)


def window_scores(prediction, y, valid):
    counts = valid.sum(2)
    # Only complete twelve-horizon windows enter fixed selection/window quantiles.
    complete = (counts > 0).all(1)
    scores = np.full(len(y), np.nan)
    errors = np.zeros(y.shape, np.float64)
    errors[valid] = np.abs(prediction[valid].astype(np.float64)-y[valid].astype(np.float64))
    scores[complete] = (errors.sum(2)[complete]/counts[complete]).mean(1)
    return scores, complete


def fixed_hard_indices(original, y, valid, sample_ids, fraction=.1):
    require(0 < fraction <= 1, 'hard fraction must lie in (0,1]')
    scores, complete = window_scores(original, y, valid)
    eligible = np.flatnonzero(complete)
    if not len(eligible):
        return np.array([], np.int64)
    # Sample identity ascending is a deterministic tie-break, never candidate error.
    order = np.lexsort((sample_ids[eligible], -scores[eligible]))
    return eligible[order[:math.ceil(len(eligible)*fraction)]]


def paired_metrics(control, candidate, y, valid, global_count=None, response_tol=.001):
    c = control[valid].astype(np.float64)
    p = candidate[valid].astype(np.float64)
    target = y[valid].astype(np.float64)
    gain = np.abs(c-target)-np.abs(p-target)
    change = p-c
    n = len(gain)
    positive, negative = np.maximum(gain, 0).sum(), np.maximum(-gain, 0).sum()
    denominator = float(positive+negative)
    by_horizon = []
    for h in range(y.shape[1]):
        mask = valid[:, h]
        g = np.abs(control[:, h][mask].astype(np.float64)-y[:, h][mask])-np.abs(candidate[:, h][mask].astype(np.float64)-y[:, h][mask])
        by_horizon.append(dict(step=h+1, valid_count=int(mask.sum()), mae_gain=_ratio(g.sum(), len(g))))
    control_metrics = point_metrics(control, y, valid)
    candidate_metrics = point_metrics(candidate, y, valid)
    directions = {}
    error_before, error_after = target-c, target-p
    crossed = error_before*error_after < 0
    category = np.full(n, 'away_from_target', dtype='<U30')
    category[change == 0] = 'unchanged'
    category[(change != 0) & (error_before == 0)] = 'exact_control_changed'
    category[(change != 0) & (error_before != 0) & (error_after == 0)] = 'reached_target'
    category[(change != 0) & (gain > 0) & ~crossed & (error_after != 0)] = 'toward_without_crossing'
    category[crossed & (gain > 0)] = 'overshoot_improved'
    category[crossed & (gain == 0)] = 'overshoot_equal_error'
    category[crossed & (gain < 0)] = 'overshoot_worsened'
    total_n = n if global_count is None else global_count
    for direction in ('unchanged', 'exact_control_changed', 'reached_target', 'toward_without_crossing',
                      'overshoot_improved', 'overshoot_equal_error', 'overshoot_worsened', 'away_from_target'):
        m = category == direction
        directions[direction] = dict(valid_count=int(m.sum()), mae_gain=_ratio(gain[m].sum(), int(m.sum())),
                                     global_mae_contribution=_ratio(gain[m].sum(), total_n))
    return dict(valid_count=n, mae_gain_pooled=_ratio(gain.sum(), n),
                mae_gain_macro=_macro([r['mae_gain'] for r in by_horizon]),
                rmse_gain_pooled=None if not n else control_metrics['rmse_pooled']-candidate_metrics['rmse_pooled'],
                rmse_gain_macro=None if candidate_metrics['rmse_macro'] is None else control_metrics['rmse_macro']-candidate_metrics['rmse_macro'],
                mape_gain=None if not n else control_metrics['mape_pooled']-candidate_metrics['mape_pooled'],
                positive_gain=_ratio(positive, n), negative_gain=_ratio(negative, n),
                cancellation_fraction=float(np.clip(1.-abs(float(gain.sum()))/denominator, 0., 1.)) if denominator else (0. if n else None),
                global_mae_contribution=_ratio(gain.sum(), total_n),
                prediction_change_mean=_ratio(change.sum(), n), prediction_change_abs_mean=_ratio(np.abs(change).sum(), n),
                up_fraction=_ratio(np.count_nonzero(change > 0), n), down_fraction=_ratio(np.count_nonzero(change < 0), n),
                response_fraction=_ratio(np.count_nonzero(np.abs(change) > response_tol), n), horizons=by_horizon, directions=directions)


def quantile_metrics(quantile, y, valid):
    """Three quantile scores; this is not a full predictive distribution or CRPS."""
    q = quantile[valid].astype(np.float64)
    target = y[valid].astype(np.float64)
    n = len(target)
    if not n:
        return dict(valid_count=0, pinball_by_quantile=None, pinball_mean=None,
                    coverage80=None, width80=None, interval_score80=None)
    residual = target[:, None]-q
    pinball = np.maximum(LEVELS*residual, (LEVELS-1)*residual).mean(0)
    lower, upper = q[:, 0], q[:, 2]
    width = upper-lower
    interval_score = width+10*np.maximum(lower-target, 0)+10*np.maximum(target-upper, 0)
    return dict(valid_count=n, pinball_by_quantile=pinball.tolist(), pinball_mean=float(pinball.mean()),
                coverage80=float(((target >= lower) & (target <= upper)).mean()),
                width80=float(width.mean()), interval_score80=float(interval_score.mean()))


def aggregate_mechanisms(batches):
    """Aggregate *raw* flat batch counters: suffix _max/_min, otherwise sum.

    Average-only records are refused: store numerator *_sum and denominator
    *_count, then compute their ratio. Never average per-batch averages.
    """
    result = {}
    for record in batches:
        require(isinstance(record, dict), 'Each mechanism batch must be a raw counter mapping')
        for key, value in record.items():
            require(not key.endswith(('_mean', '_average', '_fraction')), 'Store raw sum/count, not batch means: '+key)
            require(isinstance(value, (int, float, np.number)) and not isinstance(value, (bool, np.bool_))
                    and np.isfinite(value), 'Mechanism counters must be finite numeric scalars')
            if key.endswith('_max'):
                result[key] = max(result.get(key, -float('inf')), float(value))
            elif key.endswith('_min'):
                result[key] = min(result.get(key, float('inf')), float(value))
            else:
                result[key] = result.get(key, 0.)+float(value)
    means = {}
    for key, value in result.items():
        if key.endswith('_sum') and key[:-4]+'_count' in result:
            means[key[:-4]+'_mean'] = _ratio(value, result[key[:-4]+'_count'])
    return dict(raw=result, means=means, batch_count=len(batches), aggregation='raw_sum_count_max_min')


def _write_csv(path, records):
    if not records:
        return
    columns = list(dict.fromkeys(key for row in records for key in row))
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, columns)
        writer.writeheader()
        writer.writerows(records)


def analyze_archive(archive, output_dir, metadata=None):
    """Validate one archived validation snapshot and write auditable paired tables.

    Existing files are never overwritten. Input arrays/checkpoints are unchanged.
    ``metadata`` records producer-side identity; it is not independent proof of
    online availability, equal training budget, or chronological independence.
    """
    metadata = {} if metadata is None else dict(metadata)
    require(metadata.get('split', 'validation') == 'validation', 'This development evaluator never evaluates the test split')
    require(metadata.get('test_evaluated', False) is False, 'test_evaluated must be false')
    checked = validate_archive(archive)
    output = Path(output_dir)
    require(not output.exists() or not any(output.iterdir()), 'Output directory must be new or empty; source artifacts are preserved')
    output.mkdir(parents=True, exist_ok=True)
    y, valid, predictions = checked['y'], checked['valid'], checked['predictions']
    total_count = int(valid.sum())
    hard = fixed_hard_indices(predictions['ORIGINAL'], y, valid, checked['sample_ids'])
    hard_mask = np.zeros(len(y), bool)
    hard_mask[hard] = True
    groups = dict(checked['groups'])
    groups['fixed_hard_windows'] = np.broadcast_to(hard_mask[:, None], (len(y), y.shape[2]))
    cases, risk_summary = {}, {}
    case_rows, horizon_rows, group_rows, direction_rows, comparison_rows, risk_rows = [], [], [], [], [], []
    for name, prediction in predictions.items():
        metrics = point_metrics(prediction, y, valid)
        windows, complete = window_scores(prediction, y, valid)
        metrics.update(window_p95=float(np.quantile(windows[complete], .95)) if complete.any() else None,
                       fixed_hard_window_mae=float(windows[hard].mean()) if len(hard) else None)
        cases[name] = metrics
        case_rows.append(dict(case=name, **{key: value for key, value in metrics.items() if key != 'horizons'}))
        horizon_rows.extend(dict(case=name, **row) for row in metrics['horizons'])
        for group, mask in groups.items():
            group_mask = valid & mask[:, None, :]
            values = point_metrics(prediction, y, group_mask)
            group_rows.append(dict(kind='case', name=name, group=group,
                                   **{key: value for key, value in values.items() if key != 'horizons'}))
        for lower, upper in zip(FLOW_EDGES[:-1], FLOW_EDGES[1:]):
            mask = valid & (y >= lower) & (y < upper)
            values = point_metrics(prediction, y, mask)
            group_rows.append(dict(kind='case', name=name, group=f'flow:[{lower:g},{upper:g})',
                                   **{key: value for key, value in values.items() if key != 'horizons'}))
        if name in checked['quantiles']:
            q = checked['quantiles'][name]
            risk_summary[name] = quantile_metrics(q, y, valid)
            for group, mask in groups.items():
                risk_rows.append(dict(case=name, group=group, **quantile_metrics(q, y, valid & mask[:, None, :])))
            for h in range(12):
                risk_rows.append(dict(case=name, group='horizon:'+str(h+1),
                                      **quantile_metrics(q[:, h:h+1], y[:, h:h+1], valid[:, h:h+1])))
            for lower, upper in zip(FLOW_EDGES[:-1], FLOW_EDGES[1:]):
                risk_rows.append(dict(case=name, group=f'flow:[{lower:g},{upper:g})',
                                      **quantile_metrics(q, y, valid & (y >= lower) & (y < upper))))
    pairs = [(name, control, candidate) for name, control, candidate in PAIRS if control in predictions and candidate in predictions]
    comparisons = {}
    for name, control_name, candidate_name in pairs:
        control, candidate = predictions[control_name], predictions[candidate_name]
        values = paired_metrics(control, candidate, y, valid, total_count)
        values.update(control=control_name, candidate=candidate_name,
                      fixed_hard_window_gain=None if not len(hard) else cases[control_name]['fixed_hard_window_mae']-cases[candidate_name]['fixed_hard_window_mae'])
        comparisons[name] = values
        comparison_rows.append(dict(comparison=name, **{key: value for key, value in values.items() if key not in ('horizons', 'directions')}))
        horizon_rows.extend(dict(comparison=name, **row) for row in values['horizons'])
        direction_rows.extend(dict(comparison=name, direction=direction, **row) for direction, row in values['directions'].items())
        for group, mask in groups.items():
            item = paired_metrics(control, candidate, y, valid & mask[:, None, :], total_count)
            group_rows.append(dict(kind='comparison', name=name, group=group,
                                   **{key: value for key, value in item.items() if key not in ('horizons', 'directions')}))
        for lower, upper in zip(FLOW_EDGES[:-1], FLOW_EDGES[1:]):
            mask = valid & (y >= lower) & (y < upper)
            item = paired_metrics(control, candidate, y, mask, total_count)
            group_rows.append(dict(kind='comparison', name=name, group=f'flow:[{lower:g},{upper:g})',
                                   **{key: value for key, value in item.items() if key not in ('horizons', 'directions')}))
    mechanism_batches = metadata.get('mechanism_batches_by_case', {})
    require(isinstance(mechanism_batches, dict), 'mechanism_batches_by_case must map cases to raw batch counter lists')
    mechanisms = {name: aggregate_mechanisms(records) for name, records in mechanism_batches.items()}
    if mechanisms:
        (output/'mechanisms.json').write_text(json.dumps(mechanisms, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    analysis = dict(schema=SCHEMA, status='UNIFIED_ADAPTER_ARCHIVE_ANALYSIS_PASS',
                    primary_baseline='ORIGINAL_IGSTGNN', cases=cases, comparisons=comparisons, quantile_scores=risk_summary,
                    axes=list(y.shape), axis_order=AXIS_ORDER, axis_hashes=checked['axis_hashes'],
                    archive_sha256=checked['source_sha256'], source_metadata=metadata, mechanisms=mechanisms,
                    hard_set=dict(anchor='ORIGINAL', fraction=.1, complete_windows=int(window_scores(predictions['ORIGINAL'], y, valid)[1].sum()),
                                  sample_ids=checked['sample_ids'][hard].tolist(), rule='twelve_horizon_macro_mae_desc_sample_id_asc'),
                    engineering_only=bool(metadata.get('engineering_only', True)),
                    new_inference_calls=0, optimizer_updates=0, test_evaluated=False,
                    boundaries=['exploratory_reused_validation', 'same_checkpoint_ablations_not_retrained_deletions',
                                'gain_not_causal_information_value', 'true_flow_groups_are_posthoc_only',
                                'overlapping_windows_not_independent_events', 'three_quantiles_not_full_distribution_or_CRPS',
                                'producer_metadata_not_independent_identity_certification'])
    (output/'analysis.json').write_text(json.dumps(analysis, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    for filename, rows in [('cases.csv', case_rows), ('comparisons.csv', comparison_rows), ('horizons.csv', horizon_rows),
                           ('groups.csv', group_rows), ('directions.csv', direction_rows), ('quantiles.csv', risk_rows)]:
        _write_csv(output/filename, rows)
    f = lambda value: '不可用' if value is None else f'{value:.9f}'
    lines = ['# IGSTGNN v0.3 统一事故状态适配器：归档配对评估', '',
             '原 IGSTGNN ORIGINAL 为主基线。只读共享真值、原掩码和预测，不加载模型、不训练、不评价测试集。', '',
             '开发验证集探索性结果；同检查点屏蔽不能证明删模块后重训结果。报告有无、支持范围及机制状态不等于真实事故因果。', '',
             '| 状态 | MAE pooled | MAE macro | RMSE pooled | MAPE (%) | 窗口P95 |',
             '|---|---:|---:|---:|---:|---:|']
    for name, values in cases.items():
        lines.append('| '+name+' | '+' | '.join(f(value) for value in
                     [values['mae_pooled'], values['mae_macro'], values['rmse_pooled'],
                      None if values['mape_pooled'] is None else values['mape_pooled']*100, values['window_p95']])+' |')
    lines += ['', '收益=对照误差−候选误差。正数表示候选更好；不可线性相加RMSE或独立模块贡献。', '',
              '| 比较 | MAE收益 pooled | MAE收益 macro | MAPE收益百分点 | 正收益 | 负收益 | 固定困难窗口收益 |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for name, values in comparisons.items():
        lines.append('| '+name+' | '+' | '.join(f(value) for value in
                     [values['mae_gain_pooled'], values['mae_gain_macro'],
                      None if values['mape_gain'] is None else values['mape_gain']*100, values['positive_gain'],
                      values['negative_gain'], values['fixed_hard_window_gain']])+' |')
    lines += ['', f'固定困难窗口：仅由 ORIGINAL 选择一次，共 {len(hard)} 个；样本ID保存在 analysis.json。', '',
              'groups.csv 包含道路、直接支持、候选区域、低流量和固定困难窗口；horizons.csv 与 directions.csv 保存时距及修正方向。',
              'quantiles.csv 报告三固定分位点的 pinball、80%覆盖、宽度与区间评分；不据此计算精确CRPS，也不保证条件覆盖。',
              '没有独立、已验证的训练内保留集身份时，禁止在本验证集上拟合采用决策器。', '']
    (output/'report.md').write_text('\n'.join(lines), encoding='utf-8')
    return analysis


def main(argv=None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--archive', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--metadata', type=Path)
    args = parser.parse_args(argv)
    metadata = None if args.metadata is None else json.loads(args.metadata.read_text(encoding='utf-8'))
    result = analyze_archive(args.archive, args.output_dir, metadata)
    print(json.dumps({key: result[key] for key in ['schema', 'status', 'new_inference_calls', 'optimizer_updates', 'test_evaluated']}, indent=2))


if __name__ == '__main__':
    main()
