"""Pure NumPy, descriptive strata for saved frozen-MIN paired predictions.

Groups use observed target flow and the fixed OFF error sign only after
prediction. They are diagnostics, never proposed prediction-time features.
All summaries share the supplied validity mask. Zero targets participate in
MAE/RMSE and are explicitly excluded from MAPE; MAPE uses fraction units.
"""
from collections.abc import Mapping

import numpy as np


SCHEMA = 'min_frozen_gain_strata_v1'
REGIONS = ('direct_report_nodes', 'potential_only', 'remaining_nodes')
BIASES = ('over', 'under', 'exact')
_SUMS = ('on_absolute', 'off_absolute', 'on_squared', 'off_squared',
         'on_percentage', 'off_percentage', 'positive', 'negative',
         'prediction_signed', 'prediction_absolute', 'on_signed_error',
         'off_signed_error')
_COUNTS = ('count', 'positive_target_count', 'up', 'down', 'unchanged',
           'improved', 'worsened', 'same_error')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _check_inputs(on, off, target, valid, regions, flow_edges):
    target, valid = np.asarray(target), np.asarray(valid)
    require(target.ndim == 3 and target.shape[1] == 12 and target.shape[2] > 0,
            'Expected target [sample, 12, station] with positive station count')
    require(target.dtype.kind in 'fiu', 'Target must contain real numbers')
    require(valid.dtype == np.bool_ and valid.shape == target.shape,
            'Expected explicit boolean validity mask matching target')
    require(np.isfinite(target[valid]).all() and (target[valid] >= 0).all(),
            'Valid targets must be finite and nonnegative')
    arrays = []
    for value in (on, off):
        value = np.asarray(value)
        require(value.shape == target.shape and value.dtype.kind in 'fiu',
                'Prediction shape or real numeric dtype differs from target')
        require(np.isfinite(value[valid]).all(), 'Valid predictions must be finite')
        arrays.append(value)
    require(isinstance(regions, Mapping) and set(regions) == set(REGIONS),
            'Expected exactly direct_report_nodes, potential_only, remaining_nodes')
    region_arrays = [np.asarray(regions[name]) for name in REGIONS]
    for value in region_arrays:
        require(value.dtype == np.bool_ and value.shape == (target.shape[0], target.shape[2]),
                'Each region must be boolean [sample, station]')
    coverage = sum(value.astype(np.uint8) for value in region_arrays)
    require((coverage == 1).all(), 'Regions must be mutually exclusive and exhaustive')
    raw_edges = np.asarray(flow_edges)
    require(raw_edges.ndim == 1 and raw_edges.size >= 2 and raw_edges.dtype.kind in 'fiu',
            'Flow edges must be a one-dimensional real numeric sequence')
    edges = raw_edges.astype(np.float64)
    require(np.isfinite(edges).all() and edges[0] == 0. and (np.diff(edges) > 0).all(),
            'Flow edges must be finite, strictly increasing, and start at zero')
    return arrays[0], arrays[1], target, valid, region_arrays, edges


def _label(lower, upper):
    return '['+format(float(lower), 'g')+','+('inf' if upper is None else format(float(upper), 'g'))+')'


def _mean(total, count):
    return float(total/count) if count else None


def _per_horizon(totals, counts):
    return [_mean(total, count) for total, count in zip(totals, counts)]


def _summary(a, index, global_count, global_percentage_count, global_h, global_percentage_h):
    row = {key: a[key][index] for key in (*_SUMS, *_COUNTS, 'prediction_max')}
    count_h, percentage_h = row['count'], row['positive_target_count']
    count, percentage_count = int(count_h.sum()), int(percentage_h.sum())
    totals = {key: row[key].sum() for key in _SUMS}
    net_h = row['off_absolute']-row['on_absolute']
    percentage_net_h = row['off_percentage']-row['on_percentage']
    net, percentage_net = float(net_h.sum()), float(percentage_net_h.sum())
    on, off = {}, {}
    for name, result in (('on', on), ('off', off)):
        result.update(mae_pooled=_mean(totals[name+'_absolute'], count),
                      rmse_pooled=float(np.sqrt(totals[name+'_squared']/count)) if count else None,
                      mape_pooled=_mean(totals[name+'_percentage'], percentage_count),
                      per_horizon_mae=_per_horizon(row[name+'_absolute'], count_h),
                      per_horizon_mape=_per_horizon(row[name+'_percentage'], percentage_h))
    difference = lambda key: off[key]-on[key] if off[key] is not None and on[key] is not None else None
    gain = {key: difference(key) for key in ('mae_pooled', 'rmse_pooled', 'mape_pooled')}
    gross = float(totals['positive']+totals['negative'])
    fraction = lambda key: _mean(int(row[key].sum()), count)
    return dict(
        status='available' if count else 'unavailable', count=count, valid_count=count,
        valid_count_per_horizon=count_h.tolist(), positive_target_count=percentage_count,
        positive_target_count_per_horizon=percentage_h.tolist(), zero_target_count=count-percentage_count,
        valid_fraction_pooled=_mean(count, global_count),
        positive_target_fraction_pooled=_mean(percentage_count, global_percentage_count),
        on=on, off=off, gain=gain, mae_gain_pooled=gain['mae_pooled'],
        rmse_gain_pooled=gain['rmse_pooled'], mape_gain_pooled=gain['mape_pooled'],
        per_horizon_mae_gain=_per_horizon(net_h, count_h),
        per_horizon_mape_gain=_per_horizon(percentage_net_h, percentage_h),
        gain_positive_pooled=_mean(totals['positive'], count),
        gain_negative_pooled=_mean(totals['negative'], count),
        gross_error_gain_pooled=_mean(gross, count),
        cancellation_fraction=max(0., min(1., 1.-abs(net)/gross)) if gross else None,
        global_mae_gain_contribution=_mean(net, global_count),
        global_mape_gain_contribution=_mean(percentage_net, global_percentage_count),
        per_horizon_global_mae_gain_contribution=_per_horizon(net_h, global_h),
        per_horizon_global_mape_gain_contribution=_per_horizon(percentage_net_h, global_percentage_h),
        mean_OFF_signed_error=_mean(totals['off_signed_error'], count),
        mean_ON_signed_error=_mean(totals['on_signed_error'], count),
        prediction_change=dict(signed_mean_on_minus_off=_mean(totals['prediction_signed'], count),
                               absolute_mean=_mean(totals['prediction_absolute'], count),
                               absolute_max=float(row['prediction_max'].max()) if count else None,
                               up_fraction=fraction('up'), down_fraction=fraction('down'),
                               unchanged_fraction=fraction('unchanged')),
        mae_response=dict(improved_fraction=fraction('improved'), worsened_fraction=fraction('worsened'),
                          unchanged_fraction=fraction('same_error')))


def _reconcile(global_metrics, families):
    output = {}
    for name, family in families.items():
        groups = list(family.values())
        count = sum(group['count'] for group in groups)
        percentage_count = sum(group['positive_target_count'] for group in groups)
        count_h = np.sum([group['valid_count_per_horizon'] for group in groups], axis=0)
        percentage_h = np.sum([group['positive_target_count_per_horizon'] for group in groups], axis=0)
        require(count == global_metrics['count'] and percentage_count == global_metrics['positive_target_count'],
                'Stratum mass does not reconcile: '+name)
        require(np.array_equal(count_h, global_metrics['valid_count_per_horizon']) and
                np.array_equal(percentage_h, global_metrics['positive_target_count_per_horizon']),
                'Stratum horizon mass does not reconcile: '+name)
        result = dict(status='PASS', count_sum=count, positive_target_count_sum=percentage_count,
                      valid_count_per_horizon_sum=count_h.tolist(),
                      positive_target_count_per_horizon_sum=percentage_h.tolist())
        for metric in ('mae', 'mape'):
            field = 'global_'+metric+'_gain_contribution'
            expected = global_metrics[field]
            summed = sum(group[field] for group in groups) if expected is not None else None
            residual = summed-expected if expected is not None else None
            require(expected is None or np.isclose(summed, expected, rtol=1e-10, atol=1e-12),
                    'Stratum gain does not reconcile: '+name+' '+metric)
            result[metric+'_gain_contribution_sum'] = summed
            result[metric+'_gain_residual'] = residual
            per_field = 'per_horizon_global_'+metric+'_gain_contribution'
            per_sums, per_residuals = [], []
            for h, wanted in enumerate(global_metrics[per_field]):
                total = sum(group[per_field][h] for group in groups) if wanted is not None else None
                require(wanted is None or np.isclose(total, wanted, rtol=1e-10, atol=1e-12),
                        'Stratum horizon gain does not reconcile: '+name+' '+metric)
                per_sums.append(total)
                per_residuals.append(total-wanted if wanted is not None else None)
            result['per_horizon_'+metric+'_gain_contribution_sum'] = per_sums
            result['per_horizon_'+metric+'_gain_residual'] = per_residuals
        output[name] = result
    return dict(status='PASS', families=output,
                empty_group_contribution='zero_mass_has_zero_global_contribution_if_global_denominator_exists')


def summarize_strata(on, off, target, valid, regions, *, flow_edges=(0., 5., 20., 50., 100., 200.)):
    """Summarize paired errors by fixed flow bins, OFF bias, regions, and intersections.

    Positive error gains are OFF minus ON; positive signed prediction change is
    ON minus OFF. Empty local statistics are None. Empty-group contributions
    are zero only when the global denominator exists, making partitions add up.
    No model imports, inference, checkpoint loading, quantile fitting, or tests
    of causal/statistical significance occur here.
    """
    on, off, target, valid, region_arrays, edges = _check_inputs(on, off, target, valid, regions, flow_edges)
    definitions = [dict(key=_label(lower, edges[i+1] if i+1 < edges.size else None),
                        lower=float(lower), upper=float(edges[i+1]) if i+1 < edges.size else None,
                        lower_inclusive=True, upper_exclusive=True)
                   for i, lower in enumerate(edges)]
    flow_names = [item['key'] for item in definitions]
    cross_names = [flow+'|'+bias+'|'+region for flow in flow_names for bias in BIASES for region in REGIONS]
    sizes = (1, len(flow_names), len(BIASES), len(REGIONS), len(cross_names))
    offsets = np.cumsum((0, *sizes[:-1])).tolist()
    group_count, horizons = sum(sizes), target.shape[1]
    accumulators = {key: np.zeros((group_count, horizons), dtype=np.float64) for key in (*_SUMS, 'prediction_max')}
    accumulators.update({key: np.zeros((group_count, horizons), dtype=np.int64) for key in _COUNTS})
    region_ids = np.zeros(region_arrays[0].shape, dtype=np.int8)
    for i, mask in enumerate(region_arrays):
        region_ids[mask] = i
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        try:
            for h in range(horizons):
                mask = valid[:, h]
                y = target[:, h][mask].astype(np.float64)
                current_on, current_off = on[:, h][mask].astype(np.float64), off[:, h][mask].astype(np.float64)
                if not y.size:
                    continue
                signed_on, signed_off = current_on-y, current_off-y
                error_on, error_off = np.abs(signed_on), np.abs(signed_off)
                gain, delta = error_off-error_on, current_on-current_off
                positive_target = y > 0.
                percentage_on, percentage_off = np.zeros(y.size), np.zeros(y.size)
                percentage_on[positive_target] = error_on[positive_target]/y[positive_target]
                percentage_off[positive_target] = error_off[positive_target]/y[positive_target]
                flow_ids = np.searchsorted(edges, y, side='right')-1
                bias_ids = np.where(signed_off > 0., 0, np.where(signed_off < 0., 1, 2))
                region = region_ids[mask]
                cross_ids = (flow_ids*len(BIASES)+bias_ids)*len(REGIONS)+region
                values = dict(on_absolute=error_on, off_absolute=error_off,
                              on_squared=np.square(error_on), off_squared=np.square(error_off),
                              on_percentage=percentage_on, off_percentage=percentage_off,
                              positive=np.maximum(gain, 0.), negative=np.maximum(-gain, 0.),
                              prediction_signed=delta, prediction_absolute=np.abs(delta),
                              on_signed_error=signed_on, off_signed_error=signed_off)
                counts = dict(count=np.ones(y.size, dtype=np.int8), positive_target_count=positive_target,
                              up=delta > 0., down=delta < 0., unchanged=delta == 0.,
                              improved=gain > 0., worsened=gain < 0., same_error=gain == 0.)
                ids = (np.zeros(y.size, dtype=np.int8), flow_ids, bias_ids, region, cross_ids)
                for offset, size, labels in zip(offsets, sizes, ids):
                    selection = slice(offset, offset+size)
                    for key, value in values.items():
                        accumulators[key][selection, h] = np.bincount(labels, weights=value, minlength=size)
                    for key, value in counts.items():
                        accumulators[key][selection, h] = np.bincount(labels[value.astype(bool)], minlength=size)
                    np.maximum.at(accumulators['prediction_max'][:, h], labels+offset, np.abs(delta))
        except FloatingPointError as error:
            raise ValueError('Finite inputs overflowed float64 diagnostic accumulation') from error
    require(all(np.isfinite(value).all() for value in accumulators.values()),
            'Nonfinite float64 diagnostic accumulation')
    global_count_h, global_percentage_h = accumulators['count'][0], accumulators['positive_target_count'][0]
    global_count, global_percentage_count = int(global_count_h.sum()), int(global_percentage_h.sum())
    summaries = [_summary(accumulators, i, global_count, global_percentage_count,
                          global_count_h, global_percentage_h) for i in range(group_count)]
    names = (('global',), flow_names, BIASES, REGIONS, cross_names)
    families = [dict(zip(keys, summaries[offset:offset+size])) for keys, offset, size in zip(names, offsets, sizes)]
    result = dict(schema=SCHEMA, flow_bin_definitions=definitions, global_metrics=families[0]['global'],
                  flow=families[1], bias=families[2], regions=families[3], cross=families[4],
                  mape_unit='fraction', zero_target_policy='included_in_MAE_RMSE_excluded_from_MAPE',
                  grouping_scope='posthoc_observed_target_flow_and_fixed_OFF_error_sign',
                  signed_gain='OFF_error_minus_ON_error_positive_is_better',
                  signed_prediction_change='ON_minus_OFF_positive_is_upward',
                  interpretation_scope='descriptive_saved_prediction_association_not_causal_identification')
    result['global'] = result.pop('global_metrics')
    result['reconciliation'] = _reconcile(result['global'], {name: result[name] for name in ('flow', 'bias', 'regions', 'cross')})
    return result
