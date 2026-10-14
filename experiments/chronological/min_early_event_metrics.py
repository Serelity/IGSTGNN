"""Prediction-level diagnostics for the early event model; no model imports.

All cases share explicit target validity and original sample/station axes.
STATE_ON defines difficult windows and over/under groups for every comparison.
"""
import csv
import math
from pathlib import Path

import numpy as np

from experiments.chronological.min_event_budget_probe_metrics import (
    CaseTotals, ComparisonTotals, metric_result, require, tail_summary, window_scores,
)

SCHEMA = 'min_early_event_metrics_v1'
BASE_CASES = ('STATE_ON', 'EVENT_ON', 'EVENT_OFF', 'EVENT_SEMANTIC_UNK', 'EVENT_LOCATION_SHUFFLE')
PARTITION = ('direct_report_nodes', 'potential_only', 'remaining_nodes')
FLOW_EDGES = (0., 5., 20., 50., 100., 200.)


def default_comparisons(cases):
    values = {
        'EVENT_vs_STATE': ('STATE_ON', 'EVENT_ON', 'independently_trained_best_validation'),
        'EVENT_dependency': ('EVENT_OFF', 'EVENT_ON', 'same_checkpoint_report_path_dependency'),
        'semantic_dependency': ('EVENT_SEMANTIC_UNK', 'EVENT_ON', 'same_checkpoint_UNK_sensitivity_potential_OOD'),
        'location_dependency': ('EVENT_LOCATION_SHUFFLE', 'EVENT_ON', 'same_checkpoint_full_station_shuffle_potential_OOD'),
    }
    if 'LOCATION_ON' in cases:
        values.update(LOCATION_vs_STATE=('STATE_ON', 'LOCATION_ON', 'independently_trained_best_validation'),
                      EVENT_vs_LOCATION=('LOCATION_ON', 'EVENT_ON', 'independently_trained_best_validation'))
    for name in cases:
        if name.startswith('EVENT_FORECAST_'):
            values['forecast_'+name.removeprefix('EVENT_FORECAST_')] = (
                name, 'EVENT_ON', 'same_checkpoint_direct_readout_only_backcasts_and_later_layers_preserved')
    return values


def summarize(predictions, target, valid, regions, sample_indices, *, comparisons=None):
    target, valid, indices = np.asarray(target), np.asarray(valid), np.asarray(sample_indices)
    require(target.ndim == 3 and target.shape[1] == 12 and target.shape[0] > 0 and target.shape[2] > 0,
            'Expected nonempty target [sample,12,station]')
    require(valid.shape == target.shape and valid.dtype == np.bool_ and valid.any(), 'Explicit nonempty bool validity required')
    require(np.isfinite(target[valid]).all() and (target[valid] >= 0).all(), 'Valid targets must be finite nonnegative')
    rows, horizons, nodes = target.shape
    require(indices.shape == (rows,) and indices.dtype.kind in 'iu' and np.unique(indices).size == rows,
            'Unique original sample indices required')
    arrays = {name: np.asarray(value) for name, value in predictions.items()}
    require(set(BASE_CASES) <= arrays.keys(), 'Missing early-event prediction cases')
    require(all(v.shape == target.shape and v.dtype.kind in 'fiu' and np.isfinite(v[valid]).all()
                for v in arrays.values()), 'Prediction axes/dtypes/finite values differ')
    masks = {}
    for name, mask in regions.items():
        mask = np.asarray(mask)
        require(mask.dtype == np.bool_ and mask.shape in ((nodes,), (rows, nodes)), 'Region must be bool [V] or [N,V]')
        masks[name] = np.broadcast_to(mask, (rows, nodes))
    require('all_nodes' in masks and masks['all_nodes'].all(), 'Complete all_nodes mask required')
    require(all(name in masks for name in PARTITION), 'Three report partitions required')
    require((sum(masks[name].astype(np.uint8) for name in PARTITION) == 1).all(), 'Report partitions must exhaust disjointly')
    comparisons = default_comparisons(arrays) if comparisons is None else comparisons
    require(all(left in arrays and right in arrays for left, right, _ in comparisons.values()), 'Unknown comparison case')
    counts = {name: np.zeros(horizons, np.int64) for name in masks}
    positive_counts = {name: np.zeros(horizons, np.int64) for name in masks}
    window_counts = np.zeros((rows, horizons), np.int64)
    window_positive = np.zeros((rows, horizons), np.int64)
    totals = {case: {name: CaseTotals(rows, horizons, global_region=name == 'all_nodes') for name in masks}
              for case in arrays}
    paired = {key: {name: ComparisonTotals(horizons) for name in masks} for key in comparisons}
    # Strata use the same STATE reference, even when the comparison control is OFF/UNK/shuffled.
    strata_names = [f'flow:[{lo:g},{FLOW_EDGES[i+1]:g})' if i+1 < len(FLOW_EDGES) else f'flow:[{lo:g},inf)'
                    for i, lo in enumerate(FLOW_EDGES)]+['bias:over', 'bias:under', 'bias:exact']
    strata = {key: {name: ComparisonTotals(horizons) for name in strata_names} for key in comparisons}
    for h in range(horizons):
        y, active = target[:, h].astype(np.float64), valid[:, h]
        values = {case: array[:, h].astype(np.float64) for case, array in arrays.items()}
        errors = {case: np.abs(value-y) for case, value in values.items()}
        percentages = {case: np.divide(error, y, out=np.zeros_like(y), where=y > 0.) for case, error in errors.items()}
        window_counts[:, h], window_positive[:, h] = active.sum(1), (active & (y > 0)).sum(1)
        for name, region in masks.items():
            mask = active & region
            counts[name][h], positive_counts[name][h] = mask.sum(), (mask & (y > 0)).sum()
            for case in arrays:
                totals[case][name].update(h, errors[case], np.square(errors[case]), percentages[case], mask,
                                         global_region=name == 'all_nodes')
            for key, (left, right, _) in comparisons.items():
                paired[key][name].update(h, errors[left], errors[right], errors[left]-errors[right], values[left]-values[right], mask)
        flow_ids = np.searchsorted(FLOW_EDGES, y, side='right')-1
        state_signed = values['STATE_ON']-y
        groups = {name: flow_ids == i for i, name in enumerate(strata_names[:len(FLOW_EDGES)])}
        groups.update({'bias:over': state_signed > 0., 'bias:under': state_signed < 0., 'bias:exact': state_signed == 0.})
        for key, (left, right, _) in comparisons.items():
            for name, mask in groups.items():
                strata[key][name].update(h, errors[left], errors[right], errors[left]-errors[right], values[left]-values[right], active & mask)
    cases, scores = {}, {}
    for case in arrays:
        metrics = {}
        for name, value in totals[case].items():
            metrics[name] = metric_result(value.absolute, value.squared, counts[name], value.percentage, positive_counts[name])
        scores[case] = window_scores(totals[case]['all_nodes'].window_absolute, window_counts)
        metrics['all_nodes']['window_mae'] = tail_summary(scores[case])
        cases[case] = dict(metrics=metrics)
    eligible = np.flatnonzero(np.isfinite(scores['STATE_ON']))
    order = np.lexsort((indices[eligible], -scores['STATE_ON'][eligible]))
    chosen = eligible[order[:max(1, math.ceil(.1*eligible.size))]] if eligible.size else np.array([], np.int64)
    fixed = dict(selector='STATE_ON', fraction=.1, selected_count=int(chosen.size), sample_indices=indices[chosen].tolist(),
                 tie_break='ascending_original_sample_id', cases={})
    for case in arrays:
        value = totals[case]['all_nodes']
        fixed['cases'][case] = dict(window_mean_mae=float(scores[case][chosen].mean()) if chosen.size else None,
            metrics=metric_result(value.window_absolute[chosen].sum(0), value.global_window_squared[chosen].sum(0),
                window_counts[chosen].sum(0), value.global_window_percentage[chosen].sum(0), window_positive[chosen].sum(0)))
    results = {}
    for key, (left, right, scope) in comparisons.items():
        metrics = {name: value.result(counts['all_nodes']) for name, value in paired[key].items()}
        groups = {name: value.result(counts['all_nodes']) for name, value in strata[key].items()}
        complete = np.isfinite(scores[left]) & np.isfinite(scores[right])
        gain = scores[left][complete]-scores[right][complete]
        results[key] = dict(control=left, candidate=right, scope=scope, metrics=metrics, strata=groups,
            paired_windows=dict(count=int(complete.sum()), mean_gain=float(gain.mean()) if gain.size else None,
                                win_fraction=float((gain > 0).mean()) if gain.size else None),
            fixed_hard_window_gain=float((scores[left][chosen]-scores[right][chosen]).mean()) if chosen.size else None)
        net = metrics['all_nodes']['global_gain_contribution']
        for family in (PARTITION, strata_names[:len(FLOW_EDGES)], strata_names[len(FLOW_EDGES):]):
            source = metrics if family == PARTITION else groups
            total = sum(source[name]['global_gain_contribution'] or 0. for name in family)
            require(net is None or np.isclose(total, net, atol=1e-10, rtol=1e-9), 'Gain partition does not reconcile: '+key)
    return dict(schema=SCHEMA, status='MIN_EARLY_EVENT_ANALYSIS_PASS', cases=cases, comparisons=results,
                fixed_hard_windows=fixed, validation_rows=rows, horizons=horizons, stations=nodes,
                validity='one_shared_explicit_original_mask', gain_sign='control_error_minus_candidate_error',
                bias_reference='STATE_ON_for_all_comparisons', flow_edges=list(FLOW_EDGES), mape_unit='fraction',
                scope='exploratory_validation_checkpoint_sensitivity_not_incident_causality',
                forecast_scope='direct_readout_only_not_whole_module_deletion', partitions=list(PARTITION))


def write_report(output, analysis):
    output = Path(output)
    def fmt(x):
        return '不可用' if x is None else f'{x:.9f}'
    lines = ['# MIN 早期事故条件化：模型与数据流评估', '',
        '原作者验证集上的探索性评估。STATE 不读事故输入；EVENT_OFF 保留 EVENT 训练后的主干，并不等于旧 MIN。', '',
        '收益为对照误差减候选误差。MASK、站点、时距、原样本轴统一；本报告 MAPE 为百分数，JSON/CSV 为比例。事故关联不等于真实事故影响。', '',
        '| 状态 | MAE pooled | MAE horizon macro | RMSE pooled | RMSE horizon macro | MAPE pooled (%) | 窗口 P95 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for case, row in analysis['cases'].items():
        m = row['metrics']['all_nodes']
        lines.append('| '+' | '.join([case]+[fmt(m[k]) for k in ('mae_pooled', 'mae_macro', 'rmse_pooled', 'rmse_macro')]
                                     +[fmt(None if m['mape_pooled'] is None else 100*m['mape_pooled']), fmt(m['window_mae']['p95'])])+' |')
    if 'training_comparability' in analysis:
        comparable = analysis['training_comparability']
        lines += ['', '完成轮次：'+str(comparable['completed_epochs'])+'；最佳轮次：'+str(comparable['best_epochs'])+'。',
                  '优化器更新：'+str(comparable['global_updates'])+'；共同训练前缀：'+str(comparable['common_epoch'])+' 轮，样本顺序已逐轮核对。',
                  '整体模型比较范围：'+comparable['whole_model_scope']+'。']
    lines += ['', '| 比较 | MAE收益 macro | 正收益 macro | 负收益 macro | 抵消比例 | 固定困难窗口收益 |', '|---|---:|---:|---:|---:|---:|']
    for name, row in analysis['comparisons'].items():
        m = row['metrics']['all_nodes']
        lines.append('| '+' | '.join([name]+[fmt(m[k]) for k in ('mae_gain_macro', 'gain_positive', 'gain_negative', 'cancellation_fraction')]
                                     +[fmt(row['fixed_hard_window_gain'])])+' |')
    primary = analysis['comparisons']['EVENT_vs_STATE']
    lines += ['', '## 道路和事故关联区域', '', '| 区域 | MAE收益 | 全网 MAE 贡献 |', '|---|---:|---:|']
    for name, m in primary['metrics'].items():
        lines.append('| '+name+' | '+fmt(m['mae_gain_macro'])+' | '+fmt(m['global_gain_contribution'])+' |')
    lines += ['', '## 固定流量与 STATE 偏差分组', '', '| 分组 | MAE收益 | 全网 MAE 贡献 |', '|---|---:|---:|']
    for name, m in primary['strata'].items():
        lines.append('| '+name+' | '+fmt(m['mae_gain_macro'])+' | '+fmt(m['global_gain_contribution'])+' |')
    lines += ['', '## 逐预测时距', '', '| 步 | STATE MAE | EVENT MAE | EVENT收益 |', '|---|---:|---:|---:|']
    for h in range(12):
        lines.append(f'| {h+1} | '+fmt(analysis['cases']['STATE_ON']['metrics']['all_nodes']['per_horizon_mae'][h])+' | '
            +fmt(analysis['cases']['EVENT_ON']['metrics']['all_nodes']['per_horizon_mae'][h])+' | '
            +fmt(primary['metrics']['all_nodes']['per_horizon_mae_gain'][h])+' |')
    lines += ['', '## 解释边界', '',
        'SEMANTIC_UNK 固定 M/D；LOCATION_SHUFFLE 在每个样本全站重分配完整 D 行，包含零行并移动 M，可能产生分布外输入。',
        'FORECAST 扰动只从最终汇总中移除特征；原 backcast、同层另一分支和后续层保持 ON，不能直接证明整个模块可删。',
        'FORECAST 单层标签使用代码索引 0–4，分别对应第 1–5 层；ALL_OFF 仅移除该分支最终读出。',
        '困难窗口由 STATE 固定选择；所有偏差分组也由 STATE 定义。真值分层与误差符号只用于事后诊断，不作推理门控。',
        '窗口可重叠，预测点不是独立事故重复。无天然无事故组；隐藏特征变化或预测敏感性不等于正确事故因果或容量识别。',
        '明细：analysis.json、paired_predictions.npz、horizons.csv、regions.csv、strata.csv、mechanisms.json、replay.json、preservation.json。', '']
    (output/'report.md').write_text('\n'.join(lines), encoding='utf-8')
    for filename, family in [('regions.csv', 'metrics'), ('strata.csv', 'strata')]:
        with (output/filename).open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['comparison', 'group', 'valid_count', 'mae_gain_macro', 'mae_gain_pooled', 'global_gain_contribution',
                             'gain_positive', 'gain_negative', 'cancellation_fraction'])
            for name, comparison in analysis['comparisons'].items():
                for group, m in comparison[family].items():
                    writer.writerow([name, group]+[m[key] for key in ('valid_count', 'mae_gain_macro', 'mae_gain_pooled',
                        'global_gain_contribution', 'gain_positive', 'gain_negative', 'cancellation_fraction')])
    with (output/'horizons.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['case', 'horizon', 'mae', 'rmse', 'mape'])
        for case, row in analysis['cases'].items():
            m = row['metrics']['all_nodes']
            for h in range(12):
                writer.writerow([case, h+1]+[m[key][h] for key in ('per_horizon_mae', 'per_horizon_rmse', 'per_horizon_mape')])
