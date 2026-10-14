"""Validation metrics for a frozen MIN and its trainable location adapter.

The explicit validity mask is shared by ON and OFF. A horizon is accumulated at
a time to avoid retaining full float64 point-error arrays. MAPE is a ratio and
excludes zero targets in addition to the supplied mask. Positive paired gains
mean that ON lowers error relative to OFF; they are descriptive, not causal.
"""
import math
from pathlib import Path

import numpy as np


SCHEMA = 'min_frozen_budget_pair_metrics_v1'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _inputs(predictions, target, valid):
    target, valid = np.asarray(target), np.asarray(valid)
    require(target.ndim == 3 and target.shape[1] > 0 and target.shape[2] > 0,
            'Expected target [sample, horizon, station] with positive horizons and stations')
    require(valid.dtype == np.bool_ and valid.shape == target.shape,
            'Expected explicit boolean validity mask matching target')
    require(target.dtype.kind in 'fiu', 'Target must be real numeric data')
    require(np.isfinite(target[valid]).all(), 'Nonfinite valid target')
    require((target[valid] >= 0).all(), 'Negative valid target')
    arrays = []
    for prediction in predictions:
        prediction = np.asarray(prediction)
        require(prediction.shape == target.shape and prediction.dtype.kind in 'fiu',
                'Prediction shape or dtype differs from target')
        require(np.isfinite(prediction[valid]).all(), 'Nonfinite valid prediction')
        arrays.append(prediction)
    return arrays, target, valid


def _vector(values, counts):
    return [float(value) if count else None for value, count in zip(values, counts)]


def _means(totals, counts):
    return np.divide(totals, counts, out=np.zeros_like(totals, dtype=np.float64), where=counts > 0)


def _macro(values, counts):
    return float(np.mean(values)) if bool((counts > 0).all()) else None


def _status(counts):
    return 'available' if bool((counts > 0).all()) else 'partial_horizons' if bool((counts > 0).any()) else 'unavailable'


def _metrics(absolute, squared, percentage, counts, percentage_counts):
    mae, rmse = _means(absolute, counts), np.sqrt(_means(squared, counts))
    mape = _means(percentage, percentage_counts)
    count, percentage_count = int(counts.sum()), int(percentage_counts.sum())
    result = dict(status=_status(counts), valid_count=count,
                  valid_count_per_horizon=counts.tolist(),
                  positive_target_count=percentage_count,
                  positive_target_count_per_horizon=percentage_counts.tolist(),
                  mae_pooled=float(absolute.sum()/count) if count else None,
                  rmse_pooled=float(np.sqrt(squared.sum()/count)) if count else None,
                  mape_pooled=float(percentage.sum()/percentage_count) if percentage_count else None,
                  mae_macro=_macro(mae, counts), rmse_macro=_macro(rmse, counts),
                  mape_macro=_macro(mape, percentage_counts), mape_unit='fraction',
                  per_horizon_mae=_vector(mae, counts),
                  per_horizon_rmse=_vector(rmse, counts),
                  per_horizon_mape=_vector(mape, percentage_counts), selected_horizons={})
    for horizon in (3, 6, 12):
        i = horizon-1
        available = i < counts.size and counts[i] > 0
        result['selected_horizons']['H'+str(horizon)] = dict(
            status='available' if available else 'unavailable',
            valid_count=int(counts[i]) if i < counts.size else 0,
            mae=result['per_horizon_mae'][i] if available else None,
            rmse=result['per_horizon_rmse'][i] if available else None,
            mape=result['per_horizon_mape'][i] if available else None)
    return result


def _window_scores(absolute, counts):
    complete = (counts > 0).all(axis=1)
    scores = np.full(counts.shape[0], np.nan)
    scores[complete] = _means(absolute[complete], counts[complete]).mean(axis=1)
    return scores


def select_hard_windows(off, target, valid, sample_indices, fraction=.1):
    """Select worst complete OFF windows, returning positional sample indices.

    Each window score equally averages its horizon MAEs, each of which pools
    valid stations. Ties use the original, unique sample IDs in ascending order.
    Incomplete windows are excluded; the selection uses no ON predictions.
    """
    (off,), target, valid = _inputs((off,), target, valid)
    require(np.isfinite(fraction) and 0. < fraction <= 1., 'Fraction must be in (0, 1]')
    indices = np.asarray(sample_indices)
    require(indices.ndim == 1 and indices.shape[0] == target.shape[0] and indices.dtype.kind in 'iu',
            'Expected one integer sample ID per window')
    require(np.unique(indices).size == indices.size, 'Sample IDs must be unique')
    window_absolute = np.zeros(target.shape[:2], dtype=np.float64)
    counts = valid.sum(axis=2, dtype=np.int64)
    for h in range(target.shape[1]):
        mask = valid[:, h]
        difference = np.zeros(mask.shape, dtype=np.float64)
        difference[mask] = np.abs(off[:, h][mask].astype(np.float64)-target[:, h][mask].astype(np.float64))
        window_absolute[:, h] = difference.sum(axis=1)
    scores = _window_scores(window_absolute, counts)
    eligible = np.flatnonzero(np.isfinite(scores))
    if not eligible.size:
        return np.empty(0, dtype=np.int64)
    ordering = np.lexsort((indices[eligible], -scores[eligible]))
    count = max(1, math.ceil(float(fraction)*eligible.size))
    return eligible[ordering[:count]].astype(np.int64, copy=False)


def summarize_pair(on, off, target, valid, fixed_hard_indices):
    """Return ON/OFF metrics and paired gains on a preselected fixed hard set."""
    (on, off), target, valid = _inputs((on, off), target, valid)
    rows, horizons, _ = target.shape
    fixed = np.asarray(fixed_hard_indices)
    require(fixed.ndim == 1 and (fixed.dtype.kind in 'iu' or fixed.size == 0),
            'Fixed hard indices must be a one-dimensional integer array')
    fixed = fixed.astype(np.int64, copy=False)
    require(np.unique(fixed).size == fixed.size and ((fixed >= 0) & (fixed < rows)).all(),
            'Fixed hard indices must be unique and in range')
    counts = valid.sum(axis=(0, 2), dtype=np.int64)
    percentage_counts = np.zeros(horizons, dtype=np.int64)
    window_counts = valid.sum(axis=2, dtype=np.int64)
    accumulators = {name: {key: np.zeros(horizons) for key in ('absolute', 'squared', 'percentage')}
                    for name in ('on', 'off')}
    window_absolute = {name: np.zeros((rows, horizons)) for name in ('on', 'off')}
    positive, negative = np.zeros(horizons), np.zeros(horizons)
    prediction_absolute, prediction_signed = np.zeros(horizons), np.zeros(horizons)
    prediction_max = np.zeros(horizons)
    for h in range(horizons):
        mask = valid[:, h]
        y = target[:, h][mask].astype(np.float64)
        percentage_mask = y > 0.
        percentage_counts[h] = np.count_nonzero(percentage_mask)
        errors = {}
        for name, prediction in (('on', on), ('off', off)):
            error = np.abs(prediction[:, h][mask].astype(np.float64)-y)
            errors[name] = error
            accumulators[name]['absolute'][h] = error.sum()
            accumulators[name]['squared'][h] = np.square(error).sum()
            accumulators[name]['percentage'][h] = (error[percentage_mask]/y[percentage_mask]).sum()
            per_point = np.zeros(mask.shape)
            per_point[mask] = error
            window_absolute[name][:, h] = per_point.sum(axis=1)
        gains = errors['off']-errors['on']
        positive[h], negative[h] = np.maximum(gains, 0.).sum(), np.maximum(-gains, 0.).sum()
        change = off[:, h][mask].astype(np.float64)-on[:, h][mask].astype(np.float64)
        prediction_absolute[h], prediction_signed[h] = np.abs(change).sum(), change.sum()
        prediction_max[h] = np.abs(change).max() if change.size else 0.
    results = {name: _metrics(**totals, counts=counts, percentage_counts=percentage_counts)
               for name, totals in accumulators.items()}
    count = int(counts.sum())
    net = positive-negative
    macro_positive, macro_negative, macro_net = (_macro(_means(a, counts), counts)
                                                  for a in (positive, negative, net))
    gross = macro_positive+macro_negative if macro_positive is not None else None
    pooled_positive = float(positive.sum()/count) if count else None
    pooled_negative = float(negative.sum()/count) if count else None
    pooled_net = float(net.sum()/count) if count else None
    pooled_gross = pooled_positive+pooled_negative if count else None
    difference = lambda field: (results['off'][field]-results['on'][field]
                                if results['off'][field] is not None and results['on'][field] is not None else None)
    paired = dict(status=_status(counts), mae_gain_macro=macro_net, mae_gain_pooled=pooled_net,
                  rmse_gain_macro=difference('rmse_macro'), rmse_gain_pooled=difference('rmse_pooled'),
                  mape_gain_macro=difference('mape_macro'), mape_gain_pooled=difference('mape_pooled'),
                  gain_positive=macro_positive, gain_negative=macro_negative, gross_error_gain=gross,
                  cancellation_fraction=max(0., min(1., 1.-abs(macro_net)/gross)) if gross else None,
                  gain_positive_pooled=pooled_positive, gain_negative_pooled=pooled_negative,
                  cancellation_fraction_pooled=max(0., min(1., 1.-abs(pooled_net)/pooled_gross)) if pooled_gross else None,
                  per_horizon_mae_gain=_vector(_means(net, counts), counts),
                  per_horizon_gain_positive=_vector(_means(positive, counts), counts),
                  per_horizon_gain_negative=_vector(_means(negative, counts), counts),
                  prediction_change=dict(
                      absolute_mean_pooled=float(prediction_absolute.sum()/count) if count else None,
                      absolute_mean_horizon_macro=_macro(_means(prediction_absolute, counts), counts),
                      absolute_max=float(prediction_max.max()) if count else None,
                      signed_mean_off_minus_on=float(prediction_signed.sum()/count) if count else None,
                      per_horizon_absolute_mean=_vector(_means(prediction_absolute, counts), counts)))
    scores = {name: _window_scores(absolute, window_counts) for name, absolute in window_absolute.items()}
    complete_fixed = bool(fixed.size and np.isfinite(scores['off'][fixed]).all())
    hard = dict(status='available' if complete_fixed else 'partial_windows' if fixed.size else 'unavailable',
                count=int(fixed.size), complete_count=int(np.count_nonzero(np.isfinite(scores['off'][fixed]))),
                positional_indices=fixed.tolist(),
                on_mae=float(scores['on'][fixed].mean()) if complete_fixed else None,
                off_mae=float(scores['off'][fixed].mean()) if complete_fixed else None,
                mae_gain=float((scores['off'][fixed]-scores['on'][fixed]).mean()) if complete_fixed else None,
                on_per_window_mae=[float(x) if np.isfinite(x) else None for x in scores['on'][fixed]],
                off_per_window_mae=[float(x) if np.isfinite(x) else None for x in scores['off'][fixed]],
                weighting='equal_complete_windows_then_equal_horizons_then_valid_stations',
                selection='fixed_baseline_OFF_worst_windows_not_reselected_per_epoch')
    return dict(schema=SCHEMA, on=results['on'], off=results['off'], paired=paired,
                fixed_hard_windows=hard, mae_gain_pooled=paired['mae_gain_pooled'],
                mae_gain_macro=paired['mae_gain_macro'], fixed_hard_mae_gain=hard['mae_gain'],
                signed_gain='OFF_absolute_error_minus_ON_absolute_error_positive_is_better',
                scope='frozen_backbone_same_validation_samples_conditional_offline_location_adapter')


def _number(value, *, percent=False):
    return '不可用' if value is None else f'{100*value if percent else value:.9f}'


def write_report(root, baseline, history, summary):
    """Write a compact report; each history row contains ``epoch`` and ``metrics``.

    ``baseline`` is the epoch-zero pair returned by :func:`summarize_pair`.
    ``summary`` may provide ``best_epoch`` and a completion ``status``.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    lines = ['# 冻结 MIN：事故位置预算分支验证', '',
             '固定 MIN 最佳检查点，只训练新增位置预算适配器。正收益为 OFF 误差减 ON 误差；MAPE 表中为百分比。', '',
             '固定困难集由训练前 OFF 验证预测选择，按完整窗口等权统计；每轮使用同一集合。H3/H6/H12 是单独预测步。', '',
             '事故输入来自作者发布归档，范围为条件离线评估；潜在交换预算不是已测道路容量。', '',
             '| 轮次 | ON pooled MAE | OFF pooled MAE | MAE 收益 | ON pooled RMSE | ON MAPE % | ON 困难集 MAE | 困难集收益 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    entries = [(0, baseline)]
    for row in history:
        if row['epoch'] != 0:
            entries.append((row['epoch'], row.get('metrics', row.get('validation', row.get('pair', row)))))
    for epoch, metrics in entries:
        on, off = metrics['on'], metrics['off']
        hard = metrics['fixed_hard_windows']
        lines.append('| '+str(epoch)+' | '+' | '.join((
            _number(on['mae_pooled']), _number(off['mae_pooled']), _number(metrics['paired']['mae_gain_pooled']),
            _number(on['rmse_pooled']), _number(on['mape_pooled'], percent=True),
            _number(hard['on_mae']), _number(hard['mae_gain'])))+' |')
    lines += ['', '最佳轮次按验证 ON pooled MAE 选择；RMSE、MAPE 和困难窗口是代价及收益的共同诊断，不能只报告有利指标。', '',
              '状态：'+str(summary.get('status', '未提供'))+'；最佳轮次：'+str(summary.get('best_adapter_epoch', summary.get('best_epoch', '未提供')))+'。', '',
              '第 0 轮也是候选；最佳仍为 0 时，表示本轮没有取得优于冻结 MIN 的新增修正收益。', '',
              '仅凭这一固定主干与一个种子的试验，不宣称真实事故因果效应，也不宣称超过原论文。', '']
    path = root/'report.md'
    path.write_text('\n'.join(lines), encoding='utf-8')
    return path
