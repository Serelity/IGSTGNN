"""Read-only point metrics for MIN-I validation interventions.

Inputs are inverse-scaled flow [sample, horizon, station] with an explicit valid
mask. Work is performed a horizon at a time; no full float64 prediction copies
or pointwise gain arrays are retained. Results are descriptive checkpoint and
training-run comparisons, not independent incident replicates or causal effects.
"""
import csv
import math
from pathlib import Path

import numpy as np


SCHEMA = 'min_event_budget_probe_metrics_v1'
CASES = ('A_ON', 'L_ON', 'L_OFF', 'S_ON', 'S_OFF', 'S_SEMANTIC_UNK', 'S_SEMANTIC_SHUFFLE')
THRESHOLDS = (.0001, .001, .01, .1, 1.)
COMPARISONS = {
    'L_vs_A': ('A_ON', 'L_ON', 'independently_trained_best_validation'),
    'S_vs_A': ('A_ON', 'S_ON', 'independently_trained_best_validation'),
    'S_vs_L': ('L_ON', 'S_ON', 'independently_trained_best_validation'),
    'L_report_dependency': ('L_OFF', 'L_ON', 'same_checkpoint_dependency'),
    'S_report_dependency': ('S_OFF', 'S_ON', 'same_checkpoint_dependency'),
    'S_semantic_UNK_dependency': ('S_SEMANTIC_UNK', 'S_ON', 'same_checkpoint_UNK_sensitivity_potential_OOD'),
    'S_semantic_shuffle_dependency': ('S_SEMANTIC_SHUFFLE', 'S_ON', 'same_checkpoint_shuffle_sensitivity'),
}
PARTITION = ('direct_report_nodes', 'potential_only', 'remaining_nodes')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def ratio(numerator, denominator):
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator)
    result = np.zeros_like(numerator, dtype=np.float64)
    np.divide(numerator, denominator, out=result, where=denominator > 0)
    return result


def vector(values, counts):
    return [float(value) if count else None for value, count in zip(values, counts)]


def macro(values, counts):
    return float(np.mean(values)) if bool((np.asarray(counts) > 0).all()) else None


def status_for(counts):
    return 'available' if bool((counts > 0).all()) else 'partial_horizons' if bool((counts > 0).any()) else 'unavailable'


def tail_summary(scores):
    scores = np.asarray(scores, dtype=np.float64)
    scores = scores[np.isfinite(scores)]
    if not scores.size:
        return dict(status='unavailable', complete_windows=0, p95=None,
                    own_worst_5_percent_count=0, own_worst_5_percent_mean=None, mean=None)
    count = max(1, math.ceil(.05*scores.size))
    # The mean of the selected tail does not depend on how ties are ordered.
    worst = np.partition(scores, scores.size-count)[-count:]
    return dict(status='available', complete_windows=int(scores.size), mean=float(scores.mean()),
                p95=float(np.quantile(scores, .95)), own_worst_5_percent_count=count,
                own_worst_5_percent_mean=float(worst.mean()),
                scope='each_case_own_complete_window_tail_not_a_paired_fixed_set')


def metric_result(absolute, squared, counts, percentage, percentage_counts):
    counts, percentage_counts = np.asarray(counts, dtype=np.int64), np.asarray(percentage_counts, dtype=np.int64)
    mae, rmse = ratio(absolute, counts), np.sqrt(ratio(squared, counts))
    mape = ratio(percentage, percentage_counts)
    count, positive = int(counts.sum()), int(percentage_counts.sum())
    result = dict(status=status_for(counts), valid_count_per_horizon=counts.tolist(), valid_count=count,
        complete_horizons=int(np.count_nonzero(counts)),
        mae_macro=macro(mae, counts), rmse_macro=macro(rmse, counts),
        mae_pooled=float(np.sum(absolute)/count) if count else None,
        rmse_pooled=float(np.sqrt(np.sum(squared)/count)) if count else None,
        per_horizon_mae=vector(mae, counts), per_horizon_rmse=vector(rmse, counts),
        mape_macro=macro(mape, percentage_counts),
        mape_pooled=float(np.sum(percentage)/positive) if positive else None,
        per_horizon_mape=vector(mape, percentage_counts),
        positive_target_count_per_horizon=percentage_counts.tolist(), mape_unit='fraction',
        selected_horizons={})
    for horizon in (3, 6, 12):
        i = horizon-1
        result['selected_horizons']['H'+str(horizon)] = dict(
            valid_count=int(counts[i]), mae=result['per_horizon_mae'][i], rmse=result['per_horizon_rmse'][i],
            mape=result['per_horizon_mape'][i])
    return result


def window_scores(absolute, counts):
    complete = (counts > 0).all(1)
    values = np.full(counts.shape[0], np.nan, dtype=np.float64)
    values[complete] = ratio(absolute[complete], counts[complete]).mean(1)
    return values


class CaseTotals:
    def __init__(self, rows, horizons, *, global_region=False):
        self.absolute = np.zeros(horizons)
        self.squared = np.zeros(horizons)
        self.percentage = np.zeros(horizons)
        self.window_absolute = np.zeros((rows, horizons))
        # Only global fields are needed for the fixed difficult-window set.
        self.global_window_squared = np.zeros((rows, horizons)) if global_region else None
        self.global_window_percentage = np.zeros((rows, horizons)) if global_region else None

    def update(self, horizon, error, squared, percentage, mask, *, global_region=False):
        selected = error[mask]
        self.absolute[horizon] = selected.sum(dtype=np.float64)
        self.squared[horizon] = squared[mask].sum(dtype=np.float64)
        self.percentage[horizon] = percentage[mask].sum(dtype=np.float64)
        self.window_absolute[:, horizon] = np.where(mask, error, 0.).sum(1, dtype=np.float64)
        if global_region:
            self.global_window_squared[:, horizon] = np.where(mask, squared, 0.).sum(1, dtype=np.float64)
            self.global_window_percentage[:, horizon] = np.where(mask, percentage, 0.).sum(1, dtype=np.float64)


class ComparisonTotals:
    def __init__(self, horizons):
        for field in ('counts', 'positive', 'negative', 'net', 'prediction_absolute', 'prediction_signed',
                      'prediction_max', 'control_absolute', 'candidate_absolute', 'control_squared', 'candidate_squared'):
            setattr(self, field, np.zeros(horizons, dtype=np.int64 if field == 'counts' else np.float64))
        self.above = {str(value): np.zeros(horizons, dtype=np.int64) for value in THRESHOLDS}

    def update(self, horizon, control_error, candidate_error, gain, prediction_change, mask):
        selected = gain[mask]
        change = prediction_change[mask]
        absolute_change = np.abs(change)
        self.counts[horizon] = selected.size
        self.positive[horizon] = np.maximum(selected, 0.).sum(dtype=np.float64)
        self.negative[horizon] = np.maximum(-selected, 0.).sum(dtype=np.float64)
        self.net[horizon] = selected.sum(dtype=np.float64)
        self.prediction_absolute[horizon] = absolute_change.sum(dtype=np.float64)
        self.prediction_signed[horizon] = change.sum(dtype=np.float64)
        self.prediction_max[horizon] = float(absolute_change.max()) if selected.size else 0.
        left, right = control_error[mask], candidate_error[mask]
        self.control_absolute[horizon], self.candidate_absolute[horizon] = left.sum(), right.sum()
        self.control_squared[horizon], self.candidate_squared[horizon] = np.square(left).sum(), np.square(right).sum()
        for threshold in THRESHOLDS:
            self.above[str(threshold)][horizon] = np.count_nonzero(absolute_change > threshold)

    def result(self, global_counts):
        count = int(self.counts.sum())
        net, positive, negative = (ratio(values, self.counts) for values in (self.net, self.positive, self.negative))
        mean_net, mean_positive, mean_negative = (macro(values, self.counts) for values in (net, positive, negative))
        gross = None if mean_positive is None else mean_positive+mean_negative
        control_mae, candidate_mae = ratio(self.control_absolute, self.counts), ratio(self.candidate_absolute, self.counts)
        control_rmse, candidate_rmse = np.sqrt(ratio(self.control_squared, self.counts)), np.sqrt(ratio(self.candidate_squared, self.counts))
        prediction = ratio(self.prediction_absolute, self.counts)
        global_complete = bool((global_counts > 0).all())
        contribution = lambda values: float(np.mean(ratio(values, global_counts))) if count and global_complete else None
        result = dict(status=status_for(self.counts), valid_count_per_horizon=self.counts.tolist(), valid_count=count,
            mae_gain_macro=mean_net, mae_gain_pooled=float(self.net.sum()/count) if count else None,
            control_mae_macro=macro(control_mae, self.counts), candidate_mae_macro=macro(candidate_mae, self.counts),
            control_rmse_macro=macro(control_rmse, self.counts), candidate_rmse_macro=macro(candidate_rmse, self.counts),
            rmse_gain_macro=macro(control_rmse-candidate_rmse, self.counts),
            rmse_gain_pooled=(float(np.sqrt(self.control_squared.sum()/count)-np.sqrt(self.candidate_squared.sum()/count))
                              if count else None),
            gain_positive=mean_positive, gain_negative=mean_negative, gross_error_gain=gross,
            cancellation_fraction=max(0., min(1., 1-abs(mean_net)/gross)) if gross else None,
            global_gain_positive=contribution(self.positive), global_gain_negative=contribution(self.negative),
            global_gain_contribution=contribution(self.net), global_valid_weight_fraction=contribution(self.counts),
            per_horizon_mae_gain=vector(net, self.counts),
            per_horizon_gain_positive=vector(positive, self.counts), per_horizon_gain_negative=vector(negative, self.counts),
            per_horizon_global_gain_contribution=vector(ratio(self.net, global_counts), global_counts) if count else [None]*12,
            prediction_change=dict(absolute_mean_horizon_macro=macro(prediction, self.counts),
                absolute_mean_pooled=float(self.prediction_absolute.sum()/count) if count else None,
                signed_mean_control_minus_candidate=float(self.prediction_signed.sum()/count) if count else None,
                absolute_max=float(self.prediction_max.max()) if count else None,
                per_horizon_absolute_mean=vector(prediction, self.counts),
                response_fraction_above={key: float(values.sum()/count) if count else None for key, values in self.above.items()},
                per_horizon_response_fraction_above={key: vector(ratio(values, self.counts), self.counts)
                                                     for key, values in self.above.items()},
                response_fraction_denominator='pooled_explicit_valid_prediction_positions',
                signed_change='control_prediction_minus_candidate_prediction'), selected_horizons={})
        for horizon in (3, 6, 12):
            i = horizon-1
            result['selected_horizons']['H'+str(horizon)] = dict(valid_count=int(self.counts[i]),
                mae_gain=result['per_horizon_mae_gain'][i], gain_positive=result['per_horizon_gain_positive'][i],
                gain_negative=result['per_horizon_gain_negative'][i])
        return result


def prepare(predictions, target, valid, regions, sample_indices, roads):
    require(isinstance(predictions, dict) and set(predictions) == set(CASES), 'Expected exactly seven MIN-I prediction cases')
    target, valid = np.asarray(target), np.asarray(valid)
    require(target.ndim == 3 and target.shape[1] == 12 and target.shape[0] > 0 and target.shape[2] > 0,
            'Expected target [N,12,V] with nonempty sample/station axes')
    require(target.dtype.kind in 'fiu' and valid.shape == target.shape and valid.dtype == np.bool_,
            'Explicit bool valid mask must match numeric target [N,12,V]')
    require(bool(valid.any()), 'At least one explicitly valid target is required')
    arrays = {name: np.asarray(predictions[name]) for name in CASES}
    require(all(value.shape == target.shape and value.dtype.kind in 'fiu' for value in arrays.values()),
            'Prediction shapes/numeric dtypes must match target axes exactly')
    rows, _, stations = target.shape
    indices = np.asarray(sample_indices)
    require(indices.shape == (rows,) and indices.dtype.kind in 'iu' and np.unique(indices).size == rows,
            'sample_indices must be unique integer [N] values')
    require(isinstance(regions, dict) and 'all_nodes' in regions, 'all_nodes region is required')
    combined = dict(regions)
    if roads is not None:
        require(isinstance(roads, dict), 'roads must be a dictionary of bool station masks')
        for road, value in roads.items():
            name = str(road) if str(road).startswith('road:') else 'road:'+str(road)
            require(name not in combined or np.array_equal(combined[name], value), 'Conflicting road region: '+name)
            combined[name] = value
    masks = {}
    for name, mask in combined.items():
        mask = np.asarray(mask)
        require(isinstance(name, str) and bool(name) and mask.dtype == np.bool_
                and mask.shape in ((stations,), (rows, stations)), 'Region must be bool [V] or [N,V]: '+str(name))
        masks[name] = np.broadcast_to(mask, (rows, stations))
    require(bool(masks['all_nodes'].all()), 'all_nodes must cover every sample-station position')
    partition = dict(regions=list(PARTITION), status='not_provided', additive=False)
    if all(name in masks for name in PARTITION):
        total = sum(masks[name].astype(np.uint8) for name in PARTITION)
        require(bool((total == 1).all()), 'Explicit report partition must be disjoint and exhaust all sample-stations')
        partition.update(status='verified_disjoint_exhaustive', additive=True)
    return arrays, target, valid, masks, indices, partition


def summarize(predictions, target, valid, regions, sample_indices, roads=None):
    arrays, target, valid, masks, indices, partition = prepare(predictions, target, valid, regions, sample_indices, roads)
    rows, horizons, _ = target.shape
    counts = {name: np.zeros(horizons, dtype=np.int64) for name in masks}
    positive_counts = {name: np.zeros(horizons, dtype=np.int64) for name in masks}
    window_counts = {name: np.zeros((rows, horizons), dtype=np.int64) for name in masks}
    window_positive_counts = np.zeros((rows, horizons), dtype=np.int64)
    totals = {case: {name: CaseTotals(rows, horizons, global_region=name == 'all_nodes') for name in masks} for case in CASES}
    comparisons = {key: {name: ComparisonTotals(horizons) for name in masks} for key in COMPARISONS}
    bias = {key: {name: ComparisonTotals(horizons) for name in ('over', 'under', 'exact')} for key in COMPARISONS}
    for horizon in range(horizons):
        y = np.asarray(target[:, horizon], dtype=np.float64)
        active = valid[:, horizon]
        require(bool(np.isfinite(y[active]).all()) and bool((y[active] >= 0).all()),
                'Valid target values must be finite nonnegative flow')
        positive = active & (y > 0)
        window_positive_counts[:, horizon] = positive.sum(1)
        frames, errors, squares, percentages = {}, {}, {}, {}
        for case in CASES:
            prediction = np.asarray(arrays[case][:, horizon], dtype=np.float64)
            require(bool(np.isfinite(prediction[active]).all()), 'Non-finite explicitly valid prediction: '+case)
            difference = np.zeros_like(y)
            with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
                np.subtract(prediction, y, out=difference, where=active)
                error = np.abs(difference)
                squared = np.square(error)
                percentage = np.zeros_like(y)
                np.divide(error, y, out=percentage, where=positive)
            require(bool(np.isfinite(error).all() and np.isfinite(squared).all() and np.isfinite(percentage).all()),
                    'Non-finite derived error statistic: '+case)
            frames[case], errors[case], squares[case], percentages[case] = prediction, error, squared, percentage
        for name, region in masks.items():
            selected = active & region
            counts[name][horizon] = selected.sum()
            positive_counts[name][horizon] = (positive & region).sum()
            window_counts[name][:, horizon] = selected.sum(1)
            for case in CASES:
                totals[case][name].update(horizon, errors[case], squares[case], percentages[case], selected,
                                         global_region=name == 'all_nodes')
        for key, (control, candidate, _) in COMPARISONS.items():
            gain = errors[control]-errors[candidate]
            prediction_change = np.zeros_like(y)
            with np.errstate(over='ignore', invalid='ignore'):
                np.subtract(frames[control], frames[candidate], out=prediction_change, where=active)
            require(bool(np.isfinite(prediction_change).all()), 'Non-finite prediction-change statistic')
            for name, region in masks.items():
                comparisons[key][name].update(horizon, errors[control], errors[candidate], gain, prediction_change, active & region)
            residual = np.zeros_like(y)
            np.subtract(frames[control], y, out=residual, where=active)
            for name, sign in (('over', residual > 0), ('under', residual < 0), ('exact', residual == 0)):
                bias[key][name].update(horizon, errors[control], errors[candidate], gain, prediction_change, active & sign)
    cases, scores = {}, {}
    for case in CASES:
        result, scores[case] = {}, {}
        for name, total in totals[case].items():
            metric = metric_result(total.absolute, total.squared, counts[name], total.percentage, positive_counts[name])
            scores[case][name] = window_scores(total.window_absolute, window_counts[name])
            metric['window_mae'] = tail_summary(scores[case][name])
            metric['window_mae']['excluded_incomplete_windows'] = int(rows-np.isfinite(scores[case][name]).sum())
            result[name] = metric
        cases[case] = dict(metrics=result)
    comparison_results = {}
    for key, (control, candidate, scope) in COMPARISONS.items():
        values = {}
        for name, total in comparisons[key].items():
            result = total.result(counts['all_nodes'])
            left, right = scores[control][name], scores[candidate][name]
            complete = np.isfinite(left) & np.isfinite(right)
            if complete.any():
                gain = left[complete]-right[complete]
                result['paired_complete_windows'] = dict(status='available', count=int(complete.sum()),
                    mean_gain=float(gain.mean()), win_fraction=float(np.mean(gain > 0)),
                    loss_fraction=float(np.mean(gain < 0)), tie_fraction=float(np.mean(gain == 0)))
            else:
                result['paired_complete_windows'] = dict(status='unavailable', count=0, mean_gain=None,
                    win_fraction=None, loss_fraction=None, tie_fraction=None)
            values[name] = result
        comparison_results[key] = dict(control=control, candidate=candidate, comparison_scope=scope, metrics=values,
            bias_metrics={name: total.result(counts['all_nodes']) for name, total in bias[key].items()})
    eligible = np.flatnonzero(np.isfinite(scores['A_ON']['all_nodes']))
    chosen = np.array([], dtype=np.int64)
    if eligible.size:
        order = np.lexsort((indices[eligible], -scores['A_ON']['all_nodes'][eligible]))
        chosen = eligible[order[:max(1, math.ceil(.1*eligible.size))]]
    fixed = dict(status='available' if chosen.size else 'unavailable', selector='A_ON', fraction=.1,
        complete_candidate_windows=int(eligible.size), selected_count=int(chosen.size),
        sample_indices=indices[chosen].tolist(), tie_break='ascending_sample_index',
        definition='each_horizon_at_least_one_valid_target_then_mean_of_12_window_horizon_MAEs', cases={})
    for case in CASES:
        total = totals[case]['all_nodes']
        selected_counts = window_counts['all_nodes'][chosen].sum(0)
        metric = metric_result(total.window_absolute[chosen].sum(0), total.global_window_squared[chosen].sum(0),
            selected_counts, total.global_window_percentage[chosen].sum(0), window_positive_counts[chosen].sum(0))
        mean = float(scores[case]['all_nodes'][chosen].mean()) if chosen.size else None
        fixed['cases'][case] = dict(metrics=metric, window_mean_mae=mean,
            A_minus_case_window_mean_gain=(float(np.mean(scores['A_ON']['all_nodes'][chosen]-scores[case]['all_nodes'][chosen]))
                                          if chosen.size else None))
    for key, (control, candidate, _) in COMPARISONS.items():
        comparison_results[key]['fixed_hard_windows'] = dict(
            status=fixed['status'], count=int(chosen.size),
            window_mean_gain=(float(np.mean(scores[control]['all_nodes'][chosen]-scores[candidate]['all_nodes'][chosen]))
                              if chosen.size else None))
    output = dict(schema=SCHEMA, status='MIN_EVENT_BUDGET_PROBE_ANALYSIS_PASS', validation_rows=rows, horizons=horizons,
        stations=target.shape[2], cases=cases, comparisons=comparison_results, fixed_hard_windows=fixed,
        partitions=partition, region_names=list(masks), gain_sign='control_absolute_error_minus_candidate_absolute_error',
        metric_definitions=dict(mae_macro='mean_of_12_horizon_point_MAEs_only_if_every_horizon_has_valid_points',
            mae_pooled='sum_absolute_error_over_all_explicit_valid_points_divided_by_their_count',
            rmse_macro='mean_of_12_horizon_RMSEs', rmse_pooled='sqrt_all_valid_squared_error_sum_over_all_valid_count',
            mape='fraction_using_strictly_positive_nonnegative_flow_targets_no_epsilon_clip_zero_excluded',
            cancellation='1_abs_net_gain_over_positive_plus_negative_gain_null_for_zero_gross_gain',
            global_contribution='mean_horizon_region_gain_sum_divided_by_all_nodes_valid_count_at_that_horizon',
            bias='control_prediction_minus_target_over_under_exact_analysis_only',
            H3_H6_H12='individual_horizon_3_6_12_not_prefix_averages',
            windows='mean_of_12_window_horizon_MAEs_only_if_each_horizon_has_one_or_more_valid_points'),
        interpretation=dict(independent_training_comparisons='different_best_checkpoints_and_training_paths',
            interventions='dependency_or_sensitivity_at_same_weights_not_retraining_or_real_incident_causality',
            semantic_UNK='may_be_out_of_distribution_not_a_retrained_semantic_removal',
            candidate_potential='region_definition_supplied_by_collector_not_certified_directed_physical_reach',
            overlapping_regions='only_verified_explicit_partition_contributions_are_additive',
            windows='may_overlap_not_independent_events', bias_uses_future_labels_only_in_analysis=True),
        new_inference_calls=0)
    # Catch numerical overflow in sums as well as pointwise arithmetic.
    def finite_tree(value):
        if isinstance(value, dict):
            return all(finite_tree(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return all(finite_tree(item) for item in value)
        return not isinstance(value, (float, np.floating)) or math.isfinite(value)
    require(finite_tree(output), 'Non-finite aggregate statistic')
    return output


def write_report(output_dir, analysis):
    """Write compact Markdown and flat CSVs; caller owns analysis.json."""
    require(analysis.get('schema') == SCHEMA, 'Unexpected analysis schema')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    def fmt(value):
        return '不可用' if value is None else f'{value:.9f}' if isinstance(value, (int, float)) else str(value)
    global_rows, horizon_rows, region_rows = [], [], []
    for case, group in analysis['cases'].items():
        for region, metric in group['metrics'].items():
            row = dict(kind='case', case=case, region=region, status=metric['status'],
                mae_macro=metric['mae_macro'], mae_pooled=metric['mae_pooled'], rmse_macro=metric['rmse_macro'],
                rmse_pooled=metric['rmse_pooled'], mape_macro_percent=None if metric['mape_macro'] is None else metric['mape_macro']*100,
                mape_pooled_percent=None if metric['mape_pooled'] is None else metric['mape_pooled']*100,
                valid_count=metric['valid_count'], window_p95=metric['window_mae']['p95'],
                own_worst_5_percent_mean=metric['window_mae']['own_worst_5_percent_mean'])
            region_rows.append(row)
            if region == 'all_nodes':
                global_rows.append(row)
            for h in range(12):
                horizon_rows.append(dict(kind='case', case=case, region=region, horizon=h+1,
                    valid_count=metric['valid_count_per_horizon'][h], mae=metric['per_horizon_mae'][h],
                    rmse=metric['per_horizon_rmse'][h],
                    mape_percent=None if metric['per_horizon_mape'][h] is None else metric['per_horizon_mape'][h]*100))
    for name, comparison in analysis['comparisons'].items():
        for region, metric in comparison['metrics'].items():
            row = dict(kind='comparison', comparison=name, scope=comparison['comparison_scope'], region=region,
                control=comparison['control'], candidate=comparison['candidate'], status=metric['status'],
                mae_gain_macro=metric['mae_gain_macro'], mae_gain_pooled=metric['mae_gain_pooled'],
                rmse_gain_macro=metric['rmse_gain_macro'], rmse_gain_pooled=metric['rmse_gain_pooled'],
                gain_positive=metric['gain_positive'], gain_negative=metric['gain_negative'],
                cancellation_fraction=metric['cancellation_fraction'], global_gain_contribution=metric['global_gain_contribution'],
                global_valid_weight_fraction=metric['global_valid_weight_fraction'],
                prediction_abs_mean=metric['prediction_change']['absolute_mean_horizon_macro'],
                prediction_abs_max=metric['prediction_change']['absolute_max'], valid_count=metric['valid_count'])
            region_rows.append(row)
            if region == 'all_nodes':
                global_rows.append(row)
            for h in range(12):
                horizon_rows.append(dict(kind='comparison', comparison=name, region=region, horizon=h+1,
                    valid_count=metric['valid_count_per_horizon'][h], mae_gain=metric['per_horizon_mae_gain'][h],
                    gain_positive=metric['per_horizon_gain_positive'][h], gain_negative=metric['per_horizon_gain_negative'][h],
                    global_gain_contribution=metric['per_horizon_global_gain_contribution'][h]))
    def csv_file(name, rows):
        keys = list(dict.fromkeys(key for row in rows for key in row))
        with (output_dir/name).open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
    for name, rows in (('global.csv', global_rows), ('horizon.csv', horizon_rows), ('region.csv', region_rows)):
        csv_file(name, rows)
    lines = ['# MIN-I 验证集模型效果与机制依赖回读', '',
        '全部指标来自相同样本、站点顺序和显式有效掩码。独立训练最佳权重比较与同权重干预分别报告。',
        '正收益表示 candidate 比 control 的绝对误差更低；干预不是重新训练消融，也不构成真实事故因果证据。',
        'UNK 可能超出训练分布。道路/候选区域可能重叠，只有已核验的显式互斥分区贡献可相加。', '',
        '语义打乱只有一组固定置换，结果是配对敏感性；固定困难集按 A_ON 误差选取，用于诊断 A 的失败窗口，不是独立困难集泛化证据。', '',
        '## 七种状态的全网结果', '',
        '| 状态 | MAE macro | MAE pooled | RMSE macro | RMSE pooled | MAPE macro % | 窗口P95 | 各自最差5%窗口均值 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for case in CASES:
        m = analysis['cases'][case]['metrics']['all_nodes']
        values = (m['mae_macro'], m['mae_pooled'], m['rmse_macro'], m['rmse_pooled'],
            None if m['mape_macro'] is None else m['mape_macro']*100, m['window_mae']['p95'], m['window_mae']['own_worst_5_percent_mean'])
        lines.append('| '+case+' | '+' | '.join(fmt(value) for value in values)+' |')
    lines += ['', '## 收益与抵消', '', '| 比较 | 范围 | MAE macro收益 | MAE pooled收益 | 正收益macro | 负收益macro | 抵消比例 | 预测平均绝对变化 |',
              '|---|---|---:|---:|---:|---:|---:|---:|']
    for name, comparison in analysis['comparisons'].items():
        m = comparison['metrics']['all_nodes']
        values = (m['mae_gain_macro'], m['mae_gain_pooled'], m['gain_positive'], m['gain_negative'], m['cancellation_fraction'],
                  m['prediction_change']['absolute_mean_horizon_macro'])
        lines.append('| '+name+' | '+comparison['comparison_scope']+' | '+' | '.join(fmt(value) for value in values)+' |')
    lines += ['', '## H3 / H6 / H12', '',
              '这里是第 3、6、12 个预测步的误差和收益，完整 12 步宏平均见上表。', '',
              '| 比较 | H3收益 | H6收益 | H12收益 |', '|---|---:|---:|---:|']
    for name, comparison in analysis['comparisons'].items():
        m = comparison['metrics']['all_nodes']['selected_horizons']
        lines.append('| '+name+' | '+' | '.join(fmt(m['H'+str(h)]['mae_gain']) for h in (3, 6, 12))+' |')
    lines += ['', '## 道路与区域', '', '| 比较 | 区域 | MAE收益 | 全网加权贡献 | 区域有效权重占比 |',
              '|---|---|---:|---:|---:|']
    for name, comparison in analysis['comparisons'].items():
        for region, m in comparison['metrics'].items():
            if region != 'all_nodes':
                lines.append('| '+name+' | '+region+' | '+' | '.join(fmt(m[field]) for field in
                    ('mae_gain_macro', 'global_gain_contribution', 'global_valid_weight_fraction'))+' |')
    fixed = analysis['fixed_hard_windows']
    lines += ['', '## 固定困难窗口', '',
        f"由 A_ON 完整窗口的 12 步窗口宏 MAE 选最差 10%，固定 {fixed['selected_count']} 个窗口，平局按样本编号。",
        '完整指每步至少一个有效点；不要求全部站点均有效。七组使用同一集合。窗口等权均值和有效点加权宏平均分别列出。', '',
        '| 状态 | 窗口等权MAE | 有效点加权MAE macro | RMSE macro | 相对A的窗口等权收益 |', '|---|---:|---:|---:|---:|']
    for case, entry in fixed['cases'].items():
        values = (entry['window_mean_mae'], entry['metrics']['mae_macro'], entry['metrics']['rmse_macro'],
                  entry['A_minus_case_window_mean_gain'])
        lines.append('| '+case+' | '+' | '.join(fmt(value) for value in values)+' |')
    lines += ['', '## 基于对照误差方向的分解', '',
        'over / under / exact 由 control 预测减真实值的符号决定，只用于分析，不进入模型。下表是对全网宏收益的贡献。', '',
        '| 比较 | over贡献 | under贡献 | exact贡献 |', '|---|---:|---:|---:|']
    for name, comparison in analysis['comparisons'].items():
        lines.append('| '+name+' | '+' | '.join(fmt(comparison['bias_metrics'][key]['global_gain_contribution'])
                                               for key in ('over', 'under', 'exact'))+' |')
    lines += ['', 'MAE/RMSE 遵循显式有效掩码；MAPE 在此基础上再排除零流量标签，JSON 用比例、表格用百分数。',
        '每个预测时距先按有效点数求均值，再跨 12 步取宏平均。缺有效点的时距和完整宏指标记不可用，不填零。',
        '全网贡献按各时距全网分母计算，不能一般性地用区域宏收益乘一个总体权重代替。',
        '各自窗口尾部是分布比较；固定困难窗口是同一批窗口。窗口可能重叠，不能视作独立事故重复。',
        '数值明细：analysis.json；平表：global.csv、horizon.csv、region.csv。本分析新增推理调用为 0。', '']
    (output_dir/'report.md').write_text('\n'.join(lines), encoding='utf-8')
    return dict(report='report.md', global_csv='global.csv', horizon_csv='horizon.csv', region_csv='region.csv')
