"""Pure NumPy pointwise error decomposition of saved early-event predictions.

This module does not import a model, torch, training or evaluation entrypoints.
It works one horizon at a time and uses one STATE reference for every group.
"""
from collections.abc import Mapping
import itertools

import numpy as np


SCHEMA = 'min_early_event_gain_decomposition_v1'
BASE_CASES = ('STATE_ON', 'EVENT_ON', 'EVENT_OFF', 'EVENT_SEMANTIC_UNK', 'EVENT_LOCATION_SHUFFLE')
PARTITION = ('direct_report_nodes', 'potential_only', 'remaining_nodes')
FLOW_EDGES = (0., 5., 20., 50., 100., 200.)
BIASES = ('over', 'under', 'exact')
COMPARISONS = {
    'T': ('STATE_ON', 'EVENT_ON', 'total_independently_trained_model_difference'),
    'P': ('EVENT_OFF', 'EVENT_ON', 'same_checkpoint_report_path_sensitivity'),
    'R': ('STATE_ON', 'EVENT_OFF', 'OFF_relative_to_STATE_remaining_difference_not_training_causality'),
    'Q': ('EVENT_SEMANTIC_UNK', 'EVENT_ON', 'same_checkpoint_semantic_UNK_sensitivity_potential_OOD'),
    'L': ('EVENT_LOCATION_SHUFFLE', 'EVENT_ON', 'same_checkpoint_full_station_location_shuffle_sensitivity_potential_OOD'),
}
SUM_FIELDS = ('control_absolute', 'candidate_absolute', 'control_squared', 'candidate_squared',
              'control_percentage', 'candidate_percentage', 'gain_net', 'gain_positive', 'gain_negative',
              'prediction_signed', 'prediction_absolute')
COUNT_FIELDS = ('count', 'positive_target_count', 'prediction_up', 'prediction_down', 'prediction_unchanged',
                'prediction_response', 'gain_improved', 'gain_worsened', 'gain_equal')
SIGN_NAMES = ('negative', 'equal', 'positive')
TRIPLETS = list(itertools.product(range(3), repeat=3))
TRIPLET_NAMES = ['T_'+SIGN_NAMES[t]+'|P_'+SIGN_NAMES[p]+'|R_'+SIGN_NAMES[r] for t, p, r in TRIPLETS]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _divide(numerator, denominator):
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator)
    result = np.zeros_like(numerator, dtype=np.float64)
    np.divide(numerator, denominator, out=result, where=denominator > 0)
    return result


def _values(values, counts):
    return [float(value) if count else None for value, count in zip(values, counts)]


def _mean(total, count):
    return float(total/count) if count else None


def _macro(values, counts):
    return float(np.mean(values)) if bool((np.asarray(counts) > 0).all()) else None


def _cancellation(positive, negative):
    if positive is None or negative is None or positive+negative == 0.:
        return None
    return float(np.clip(1.-abs(positive-negative)/(positive+negative), 0., 1.))


def _label(lower, upper):
    return '['+format(float(lower), 'g')+','+('inf' if upper is None else format(float(upper), 'g'))+')'


def _validate(predictions, target, valid, regions, sample_indices, station_ids, fixed_ids, flow_edges, response_tol):
    require(isinstance(predictions, Mapping) and set(BASE_CASES) <= predictions.keys(), 'Five base prediction cases are required')
    target, valid = np.asarray(target), np.asarray(valid)
    require(target.ndim == 3 and target.shape[0] > 0 and target.shape[1] == 12 and target.shape[2] > 0,
            'Target must be nonempty [sample,12,station]')
    require(target.dtype.kind in 'fiu' and valid.shape == target.shape and valid.dtype == np.bool_ and valid.any(),
            'Explicit nonempty Boolean validity must match numeric target axes')
    require(np.isfinite(target[valid]).all() and (target[valid] >= 0.).all(), 'Valid targets must be finite nonnegative')
    arrays = {case: np.asarray(predictions[case]) for case in BASE_CASES}
    require(all(value.shape == target.shape and value.dtype.kind in 'fiu' and np.isfinite(value[valid]).all()
                for value in arrays.values()), 'Predictions must share numeric axes and finite valid positions')
    rows, _, nodes = target.shape
    sample_indices, station_ids, fixed_ids = np.asarray(sample_indices), np.asarray(station_ids), np.asarray(fixed_ids)
    require(sample_indices.shape == (rows,) and sample_indices.dtype.kind in 'iu'
            and np.unique(sample_indices).size == rows and (sample_indices >= 0).all(), 'Unique nonnegative original sample IDs required')
    require(station_ids.shape == (nodes,) and station_ids.dtype.kind in 'iu' and np.unique(station_ids).size == nodes,
            'Unique integer station IDs must match the station axis')
    require(fixed_ids.ndim == 1 and (fixed_ids.size == 0 or fixed_ids.dtype.kind in 'iu')
            and np.unique(fixed_ids).size == fixed_ids.size and np.isin(fixed_ids, sample_indices).all(),
            'Fixed hard sample IDs must be a unique subset of the saved sample axis')
    fixed_ids = fixed_ids.astype(np.int64)
    require(isinstance(regions, Mapping) and 'all_nodes' in regions and set(PARTITION) <= regions.keys(),
            'all_nodes and the three report partitions are required')
    masks = {}
    for name, value in regions.items():
        value = np.asarray(value)
        require(isinstance(name, str) and bool(name) and value.dtype == np.bool_
                and value.shape in ((nodes,), (rows, nodes)), 'Region mask must be bool [station] or [sample,station]')
        masks[name] = np.broadcast_to(value, (rows, nodes))
    require(masks['all_nodes'].all(), 'all_nodes must cover the entire sample/station axis')
    require((sum(masks[name].astype(np.uint8) for name in PARTITION) == 1).all(),
            'The three report partitions must be disjoint and exhaustive')
    edges = np.asarray(flow_edges)
    require(edges.ndim == 1 and edges.size >= 2 and edges.dtype.kind in 'fiu', 'Flow edges must be a numeric vector')
    edges = edges.astype(np.float64)
    require(np.isfinite(edges).all() and edges[0] == 0. and (np.diff(edges) > 0).all(),
            'Flow edges must be finite increasing bounds starting at zero')
    require(isinstance(response_tol, (int, float, np.integer, np.floating)) and not isinstance(response_tol, (bool, np.bool_))
            and np.isfinite(response_tol) and response_tol >= 0., 'Prediction response tolerance must be finite nonnegative')
    return arrays, target, valid, masks, sample_indices, station_ids, fixed_ids, edges, float(response_tol)


def _groups(masks, edges):
    names, definitions = [], {}
    def add(name, family, **fields):
        require(name not in definitions, 'Reserved group name conflicts with a supplied region: '+name)
        names.append(name)
        definitions[name] = dict(family=family, **fields)
    for name in masks:
        add(name, 'road' if name.startswith('road:') else 'region', region=name)
    flow_names = [_label(lo, edges[i+1] if i+1 < len(edges) else None) for i, lo in enumerate(edges)]
    offsets = dict(flow=len(names))
    for name in flow_names:
        add('flow:'+name, 'flow', flow=name)
    offsets['bias'] = len(names)
    for bias in BIASES:
        add('bias:'+bias, 'STATE_bias', bias=bias)
    offsets['cross'] = len(names)
    for flow, bias, region in itertools.product(flow_names, BIASES, PARTITION):
        add('cross:flow='+flow+'|bias='+bias+'|region='+region, 'flow_STATE_bias_report_partition',
            flow=flow, bias=bias, region=region)
    offsets['fixed'] = len(names)
    add('fixed_hard_windows', 'fixed_STATE_hard_windows')
    return names, definitions, offsets, flow_names


def _case_metrics(absolute, squared, percentage, counts, positive_counts):
    mae, rmse, mape = _divide(absolute, counts), np.sqrt(_divide(squared, counts)), _divide(percentage, positive_counts)
    count, positive = int(counts.sum()), int(positive_counts.sum())
    return dict(mae_pooled=_mean(absolute.sum(), count), mae_macro=_macro(mae, counts),
        rmse_pooled=float(np.sqrt(squared.sum()/count)) if count else None, rmse_macro=_macro(rmse, counts),
        mape_pooled=_mean(percentage.sum(), positive), mape_macro=_macro(mape, positive_counts),
        per_horizon_mae=_values(mae, counts), per_horizon_rmse=_values(rmse, counts), per_horizon_mape=_values(mape, positive_counts))


def _summary(acc, group, global_counts, global_positive_counts):
    row = {name: value[group] for name, value in acc.items()}
    counts, positive_counts = row['count'], row['positive_target_count']
    count, positive_count = int(counts.sum()), int(positive_counts.sum())
    control = _case_metrics(row['control_absolute'], row['control_squared'], row['control_percentage'], counts, positive_counts)
    candidate = _case_metrics(row['candidate_absolute'], row['candidate_squared'], row['candidate_percentage'], counts, positive_counts)
    gain_h = _divide(row['gain_net'], counts)
    percentage_net = row['control_percentage']-row['candidate_percentage']
    positive_h, negative_h = _divide(row['gain_positive'], counts), _divide(row['gain_negative'], counts)
    positive_pooled, negative_pooled = _mean(row['gain_positive'].sum(), count), _mean(row['gain_negative'].sum(), count)
    positive_macro, negative_macro = _macro(positive_h, counts), _macro(negative_h, counts)
    global_count, global_positive = int(global_counts.sum()), int(global_positive_counts.sum())
    global_macro = lambda value: _macro(_divide(value, global_counts), global_counts)
    global_pct_macro = lambda value: _macro(_divide(value, global_positive_counts), global_positive_counts)
    prediction = dict(signed_mean_candidate_minus_control_pooled=_mean(row['prediction_signed'].sum(), count),
        signed_mean_candidate_minus_control_macro=_macro(_divide(row['prediction_signed'], counts), counts),
        absolute_mean_pooled=_mean(row['prediction_absolute'].sum(), count),
        absolute_mean_macro=_macro(_divide(row['prediction_absolute'], counts), counts),
        absolute_max=float(row['prediction_max'].max()) if count else None,
        up_fraction=_mean(row['prediction_up'].sum(), count), down_fraction=_mean(row['prediction_down'].sum(), count),
        unchanged_fraction=_mean(row['prediction_unchanged'].sum(), count),
        response_fraction=_mean(row['prediction_response'].sum(), count),
        per_horizon_signed_mean=_values(_divide(row['prediction_signed'], counts), counts),
        per_horizon_absolute_mean=_values(_divide(row['prediction_absolute'], counts), counts),
        per_horizon_up_fraction=_values(_divide(row['prediction_up'], counts), counts),
        per_horizon_down_fraction=_values(_divide(row['prediction_down'], counts), counts),
        per_horizon_unchanged_fraction=_values(_divide(row['prediction_unchanged'], counts), counts),
        per_horizon_response_fraction=_values(_divide(row['prediction_response'], counts), counts))
    result = dict(status='available' if (counts > 0).all() else 'partial_horizons' if count else 'unavailable',
        valid_count=count, valid_count_per_horizon=counts.astype(int).tolist(),
        positive_target_count=positive_count, positive_target_count_per_horizon=positive_counts.astype(int).tolist(),
        control=control, candidate=candidate,
        mae_gain_pooled=_mean(row['gain_net'].sum(), count), mae_gain_macro=_macro(gain_h, counts),
        rmse_gain_pooled=None if not count else control['rmse_pooled']-candidate['rmse_pooled'],
        rmse_gain_macro=None if control['rmse_macro'] is None else control['rmse_macro']-candidate['rmse_macro'],
        mape_gain_pooled=_mean(percentage_net.sum(), positive_count),
        mape_gain_macro=_macro(_divide(percentage_net, positive_counts), positive_counts),
        gain_positive_pooled=positive_pooled, gain_negative_pooled=negative_pooled,
        gain_positive_macro=positive_macro, gain_negative_macro=negative_macro,
        gross_gain_pooled=None if positive_pooled is None else positive_pooled+negative_pooled,
        gross_gain_macro=None if positive_macro is None else positive_macro+negative_macro,
        cancellation_pooled=_cancellation(positive_pooled, negative_pooled), cancellation_macro=_cancellation(positive_macro, negative_macro),
        improved_fraction=_mean(row['gain_improved'].sum(), count), worsened_fraction=_mean(row['gain_worsened'].sum(), count),
        equal_error_fraction=_mean(row['gain_equal'].sum(), count),
        per_horizon_improved_fraction=_values(_divide(row['gain_improved'], counts), counts),
        per_horizon_worsened_fraction=_values(_divide(row['gain_worsened'], counts), counts),
        per_horizon_equal_error_fraction=_values(_divide(row['gain_equal'], counts), counts),
        global_mae_contribution_pooled=_mean(row['gain_net'].sum(), global_count),
        global_mae_contribution_macro=global_macro(row['gain_net']),
        global_mape_contribution_pooled=_mean(percentage_net.sum(), global_positive),
        global_mape_contribution_macro=global_pct_macro(percentage_net),
        global_gain_positive_pooled=_mean(row['gain_positive'].sum(), global_count),
        global_gain_negative_pooled=_mean(row['gain_negative'].sum(), global_count),
        global_gain_positive_macro=global_macro(row['gain_positive']), global_gain_negative_macro=global_macro(row['gain_negative']),
        global_valid_weight_fraction_pooled=_mean(count, global_count), global_valid_weight_fraction_macro=global_macro(counts),
        per_horizon_mae_gain=_values(gain_h, counts),
        per_horizon_rmse_gain=[None if left is None else left-right for left, right in zip(control['per_horizon_rmse'], candidate['per_horizon_rmse'])],
        per_horizon_mape_gain=_values(_divide(percentage_net, positive_counts), positive_counts),
        per_horizon_gain_positive=_values(positive_h, counts), per_horizon_gain_negative=_values(negative_h, counts),
        per_horizon_global_mae_contribution=_values(_divide(row['gain_net'], global_counts), global_counts),
        per_horizon_global_mape_contribution=_values(_divide(percentage_net, global_positive_counts), global_positive_counts),
        per_horizon_global_valid_weight=_values(_divide(counts, global_counts), global_counts), prediction_change=prediction)
    return result


def analyze_predictions(predictions, target, valid, regions, sample_indices, station_ids, fixed_hard_sample_ids,
                        flow_edges=FLOW_EDGES, response_tol=.001):
    """Describe T=P+R on identical saved points; this is not causal attribution.

    Target flow bins and STATE errors are posthoc diagnostic groups. The response
    tolerance affects prediction-response counts only; gain signs use exact zero.
    Group macro means require all 12 group horizons to be nonempty. Common global
    contributions have global denominators and can have different signs.
    """
    arrays, target, valid, masks, samples, stations, hard_ids, edges, response_tol = _validate(
        predictions, target, valid, regions, sample_indices, station_ids, fixed_hard_sample_ids, flow_edges, response_tol)
    names, definitions, offsets, flow_names = _groups(masks, edges)
    group_count, horizons = len(names), 12
    accumulators = {key: {field: np.zeros((group_count, horizons), dtype=np.int64 if field in COUNT_FIELDS else np.float64)
                          for field in (*SUM_FIELDS, *COUNT_FIELDS, 'prediction_max')} for key in COMPARISONS}
    quadrant_counts = np.zeros((group_count, horizons, 27), np.int64)
    # T/P/R net sums, followed by same-point |P|+|R| and its reduction to |T|.
    quadrant_sums = np.zeros((group_count, horizons, 27, 5), np.float64)
    partition_ids = np.zeros(target.shape[::2], np.int8)
    for i, name in enumerate(PARTITION):
        partition_ids[masks[name]] = i
    fixed_rows = np.isin(samples, hard_ids)
    window_counts = np.zeros((len(samples), horizons), np.int64)
    window_errors = {case: np.zeros((len(samples), horizons)) for case in BASE_CASES}
    maximum_identity_residual = 0.
    identity_checked = 0
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        try:
            for h in range(horizons):
                active = valid[:, h]
                window_counts[:, h] = active.sum(1)
                y = target[:, h][active].astype(np.float64)
                values = {case: arrays[case][:, h][active].astype(np.float64) for case in BASE_CASES}
                if not len(y):
                    continue
                errors = {case: np.abs(value-y) for case, value in values.items()}
                percentage = {case: np.divide(error, y, out=np.zeros_like(y), where=y > 0.) for case, error in errors.items()}
                gains = {key: errors[left]-errors[right] for key, (left, right, _) in COMPARISONS.items()}
                residual = gains['T']-gains['P']-gains['R']
                maximum_identity_residual = max(maximum_identity_residual, float(np.abs(residual).max()))
                identity_scale = errors['STATE_ON']+errors['EVENT_OFF']+errors['EVENT_ON']+1.
                require((np.abs(residual) <= 1e-12+32*np.finfo(np.float64).eps*identity_scale).all(),
                        'Pointwise T=P+R identity failed')
                identity_checked += len(y)
                sign_ids = {key: np.where(gains[key] > 0., 2, np.where(gains[key] < 0., 0, 1)) for key in ('T', 'P', 'R')}
                triplet = (sign_ids['T']*3+sign_ids['P'])*3+sign_ids['R']
                flow_ids = np.searchsorted(edges, y, side='right')-1
                state_error = values['STATE_ON']-y
                bias_ids = np.where(state_error > 0., 0, np.where(state_error < 0., 1, 2))
                cross_ids = (flow_ids*3+bias_ids)*3+partition_ids[active]
                # Region groups may overlap; the categorical families each exhaust the same valid points.
                selections = [(i, 1, mask[active].astype(np.int8), True) for i, mask in enumerate(masks.values())]
                selections += [(offsets['flow'], len(edges), flow_ids, False), (offsets['bias'], 3, bias_ids, False),
                    (offsets['cross'], len(edges)*9, cross_ids, False),
                    (offsets['fixed'], 1, np.broadcast_to(fixed_rows[:, None], active.shape)[active].astype(np.int8), True)]
                row_ids = np.broadcast_to(np.arange(len(samples))[:, None], active.shape)[active]
                for case in BASE_CASES:
                    window_errors[case][:, h] = np.bincount(row_ids, weights=errors[case], minlength=len(samples))
                quadrant_values = [gains[key] for key in ('T', 'P', 'R')]
                pair_absolute = np.abs(gains['P'])+np.abs(gains['R'])
                quadrant_values += [pair_absolute, np.maximum(pair_absolute-np.abs(gains['T']), 0.)]
                for offset, size, labels, is_mask in selections:
                    selection = labels.astype(bool) if is_mask else np.ones(len(y), bool)
                    group_labels = np.zeros(int(selection.sum()), np.int64) if is_mask else labels
                    quad_labels = group_labels*27+triplet[selection]
                    quadrant_counts[offset:offset+size, h] = np.bincount(quad_labels, minlength=size*27).reshape(size, 27)
                    for i, value in enumerate(quadrant_values):
                        quadrant_sums[offset:offset+size, h, :, i] = np.bincount(
                            quad_labels, weights=value[selection], minlength=size*27).reshape(size, 27)
                for key, (left, right, _) in COMPARISONS.items():
                    # Build each comparison's arithmetic vectors only once for this horizon.
                    prediction_change, gain = values[right]-values[left], gains[key]
                    sums = dict(control_absolute=errors[left], candidate_absolute=errors[right],
                        control_squared=np.square(errors[left]), candidate_squared=np.square(errors[right]),
                        control_percentage=percentage[left], candidate_percentage=percentage[right],
                        gain_net=gain, gain_positive=np.maximum(gain, 0.), gain_negative=np.maximum(-gain, 0.),
                        prediction_signed=prediction_change, prediction_absolute=np.abs(prediction_change))
                    counts = dict(count=np.ones(len(y), bool), positive_target_count=y > 0.,
                        prediction_up=prediction_change > 0., prediction_down=prediction_change < 0.,
                        prediction_unchanged=prediction_change == 0., prediction_response=np.abs(prediction_change) > response_tol,
                        gain_improved=gain > 0., gain_worsened=gain < 0., gain_equal=gain == 0.)
                    acc = accumulators[key]
                    for offset, size, labels, is_mask in selections:
                        selection = labels.astype(bool) if is_mask else np.ones(len(y), bool)
                        group_labels = np.zeros(int(selection.sum()), np.int64) if is_mask else labels
                        for field, value in sums.items():
                            acc[field][offset:offset+size, h] = np.bincount(group_labels, weights=value[selection], minlength=size)
                        for field, value in counts.items():
                            acc[field][offset:offset+size, h] = np.bincount(group_labels[value[selection]], minlength=size)
                        np.maximum.at(acc['prediction_max'][:, h], group_labels+offset, np.abs(prediction_change[selection]))
        except FloatingPointError as error:
            raise ValueError('Finite saved values overflowed float64 diagnostic arithmetic') from error
    require(all(np.isfinite(value).all() for acc in accumulators.values() for value in acc.values())
            and np.isfinite(quadrant_sums).all(), 'Nonfinite diagnostic accumulation')
    global_index = names.index('all_nodes')
    global_counts = accumulators['T']['count'][global_index]
    global_positive = accumulators['T']['positive_target_count'][global_index]
    comparisons = {key: dict(control=left, candidate=right, scope=scope,
        groups={name: _summary(accumulators[key], i, global_counts, global_positive) for i, name in enumerate(names)})
        for key, (left, right, scope) in COMPARISONS.items()}
    reconciliation = {}
    families = dict(report_partition=[names.index(name) for name in PARTITION],
        flow=list(range(offsets['flow'], offsets['bias'])), bias=list(range(offsets['bias'], offsets['cross'])),
        cross=list(range(offsets['cross'], offsets['fixed'])))
    for key, acc in accumulators.items():
        reconciliation[key] = {}
        for family, ids in families.items():
            count_residual = acc['count'][ids].sum(0)-global_counts
            net_residual = acc['gain_net'][ids].sum(0)-acc['gain_net'][global_index]
            positive_residual = acc['positive_target_count'][ids].sum(0)-global_positive
            percentage_net = acc['control_percentage']-acc['candidate_percentage']
            percentage_residual = percentage_net[ids].sum(0)-percentage_net[global_index]
            scale = acc['control_absolute'][global_index]+acc['candidate_absolute'][global_index]+1.
            percentage_scale = acc['control_percentage'][global_index]+acc['candidate_percentage'][global_index]+1.
            require((count_residual == 0).all() and (positive_residual == 0).all()
                    and (np.abs(net_residual) <= 1e-10+128*np.finfo(np.float64).eps*scale).all()
                    and (np.abs(percentage_residual) <= 1e-10+128*np.finfo(np.float64).eps*percentage_scale).all(),
                    'Stratum count/gain reconciliation failed: '+key+' '+family)
            reconciliation[key][family] = dict(status='PASS', count_residual_per_horizon=count_residual.tolist(),
                maximum_absolute_gain_sum_residual=float(np.abs(net_residual).max()),
                maximum_absolute_percentage_gain_sum_residual=float(np.abs(percentage_residual).max()))
    aggregate_residual = accumulators['T']['gain_net']-accumulators['P']['gain_net']-accumulators['R']['gain_net']
    aggregate_scale = (accumulators['T']['control_absolute']+accumulators['T']['candidate_absolute']
                       +accumulators['P']['control_absolute']+1.)
    require((np.abs(aggregate_residual) <= 1e-10+128*np.finfo(np.float64).eps*aggregate_scale).all(),
            'Grouped T=P+R identity failed')
    quadrant_results = {}
    for i, name in enumerate(names):
        count_h = accumulators['T']['count'][i]
        require(np.array_equal(quadrant_counts[i].sum(1), count_h), 'Strict sign triplets do not exhaust group')
        def quadrant_summary(ids):
            counts = quadrant_counts[i][:, ids].sum(1)
            sums = quadrant_sums[i][:, ids].sum(1)
            count = int(counts.sum())
            return dict(valid_count=count, valid_count_per_horizon=counts.tolist(),
                within_group_fraction=_mean(count, int(count_h.sum())),
                global_point_fraction=_mean(count, int(global_counts.sum())),
                T_gain_sum=float(sums[:, 0].sum()), P_gain_sum=float(sums[:, 1].sum()), R_gain_sum=float(sums[:, 2].sum()),
                T_gain_mean=_mean(sums[:, 0].sum(), count), P_gain_mean=_mean(sums[:, 1].sum(), count),
                R_gain_mean=_mean(sums[:, 2].sum(), count),
                P_R_absolute_sum=float(sums[:, 3].sum()), P_R_pointwise_cancellation_sum=float(sums[:, 4].sum()),
                P_R_pointwise_cancellation_fraction=_mean(sums[:, 4].sum(), sums[:, 3].sum()),
                P_R_pointwise_cancellation_global_contribution=_mean(sums[:, 4].sum(), int(global_counts.sum())),
                T_global_mae_contribution_pooled=_mean(sums[:, 0].sum(), int(global_counts.sum())),
                P_global_mae_contribution_pooled=_mean(sums[:, 1].sum(), int(global_counts.sum())),
                R_global_mae_contribution_pooled=_mean(sums[:, 2].sum(), int(global_counts.sum())))
        strict = {label: quadrant_summary([j]) for j, label in enumerate(TRIPLET_NAMES)}
        overlap = {}
        for t_positive, p_positive in itertools.product((False, True), repeat=2):
            label = ('T_positive' if t_positive else 'T_nonpositive')+'|'+('P_positive' if p_positive else 'P_nonpositive')
            ids = [j for j, (t, p, _) in enumerate(TRIPLETS) if (t == 2) == t_positive and (p == 2) == p_positive]
            overlap[label] = quadrant_summary(ids)
        require(sum(value['valid_count'] for value in overlap.values()) == int(count_h.sum()), 'Four overlaps do not exhaust group')
        pair_signs = {('P_'+SIGN_NAMES[p]+'|R_'+SIGN_NAMES[r]): quadrant_summary(
            [j for j, (_, jp, jr) in enumerate(TRIPLETS) if jp == p and jr == r])
            for p, r in itertools.product(range(3), repeat=2)}
        opposed = quadrant_summary([j for j, (_, p, r) in enumerate(TRIPLETS) if (p, r) in ((0, 2), (2, 0))])
        quadrant_results[name] = dict(strict_triplets=strict, T_P_positive_nonpositive=overlap,
                                     P_R_sign_cross=pair_signs, P_R_opposed=opposed,
                                     strict_triplets_and_four_overlaps_each_disjoint_exhaustive=True)
    complete_windows = (window_counts > 0).all(1)
    window_values = {case: np.where(complete_windows, _divide(error, window_counts).mean(1), np.nan)
                     for case, error in window_errors.items()}
    fixed_complete = fixed_rows & complete_windows
    fixed_report = dict(selector='saved_STATE_fixed_sample_ids_no_reselection', sample_indices=hard_ids.tolist(),
        selected_count=int(len(hard_ids)), complete_window_count=int(fixed_complete.sum()), comparisons={})
    for key, (left, right, _) in COMPARISONS.items():
        gain = window_values[left][fixed_complete]-window_values[right][fixed_complete]
        fixed_report['comparisons'][key] = dict(window_equal_weight_mean_gain=float(gain.mean()) if gain.size else None,
            window_improved_fraction=float((gain > 0).mean()) if gain.size else None,
            window_worsened_fraction=float((gain < 0).mean()) if gain.size else None,
            point_metrics=comparisons[key]['groups']['fixed_hard_windows'])
    return dict(schema=SCHEMA, status='MIN_EARLY_EVENT_GAIN_ANALYSIS_PASS', comparisons=comparisons,
        group_definitions=definitions, quadrants=dict(gain_sign_threshold=0., groups=quadrant_results),
        fixed_hard_windows=fixed_report, reconciliation=reconciliation,
        identity_check=dict(status='POINTWISE_AND_GROUPED_T_EQUALS_P_PLUS_R_PASS', checked_points=int(identity_checked),
            maximum_absolute_point_residual=maximum_identity_residual,
            maximum_absolute_group_gain_sum_residual=float(np.abs(aggregate_residual).max()),
            numerical_tolerance='pointwise_and_sum_checks_scale_with_original_absolute_errors_and_float64_epsilon',
            interpretation='arithmetic_same_point_error_decomposition_not_causal_or_identified_training_attribution'),
        axis=dict(samples=len(samples), horizons=12, stations=len(stations), sample_indices=samples.astype(int).tolist(),
                  station_ids=stations.astype(int).tolist()),
        definitions=dict(gain='control_absolute_error_minus_candidate_absolute_error',
            T='abs(STATE-y)-abs(EVENT_ON-y)', P='abs(EVENT_OFF-y)-abs(EVENT_ON-y)',
            R='abs(STATE-y)-abs(EVENT_OFF-y); OFF_relative_to_STATE_remaining_difference',
            Q='abs(EVENT_SEMANTIC_UNK-y)-abs(EVENT_ON-y)', L='abs(EVENT_LOCATION_SHUFFLE-y)-abs(EVENT_ON-y)',
            bias='STATE_ON_minus_target_over_under_exact_fixed_for_every_comparison',
            group_macro='equal_mean_of_12_group_horizon_means_only_when_every_group_horizon_nonempty',
            pooled='all_valid_points_equal_weight',
            global_macro_contribution='mean_horizon_group_gain_sum_divided_by_all_nodes_valid_count_at_same_horizon',
            global_pooled_contribution='group_gain_sum_divided_by_all_nodes_valid_count',
            group_macro_and_global_contribution_may_have_opposite_signs=True,
            group_contributions_additive_only_for_disjoint_exhaustive_families=['flow', 'bias', 'report_partition', 'cross'],
            mape_unit='fraction', mape_zero_policy='zero_targets_in_MAE_RMSE_but_excluded_from_MAPE',
            prediction_change='candidate_prediction_minus_control_prediction_positive_is_upward',
            prediction_response_tol=response_tol, prediction_response='abs_prediction_change_strictly_above_tol',
            gain_sign_threshold=0., grouping_uses_targets_for_posthoc_analysis_only=True,
            P_R_pointwise_cancellation='sum(abs(P)+abs(R)-abs(T)); differs_from_positive_negative_aggregate_gain_cancellation',
            positive_negative_gain_parts_not_additive_between_T_P_R=True,
            net_path_over_total_gain_ratio_not_computed=True,
            independence='overlapping_windows_and_points_not_independent_incident_replicates',
            hidden_scale_or_capacity_not_observed_vehicle_flow=True,
            empty_groups='local_means_unavailable_zero_mass_common_denominator_contributions_zero'))


def flat_rows(result):
    """Compact group/horizon CSV rows; never emit individual prediction points."""
    rows = []
    for comparison, item in result['comparisons'].items():
        for group, metric in item['groups'].items():
            definition = result['group_definitions'][group]
            common = dict(comparison=comparison, group=group, family=definition['family'],
                          flow=definition.get('flow', ''), bias=definition.get('bias', ''), region=definition.get('region', ''))
            pooled = dict(common, horizon=0, aggregate='pooled_and_macro', valid_count=metric['valid_count'],
                positive_target_count=metric['positive_target_count'],
                mae_gain=metric['mae_gain_pooled'], mae_gain_macro=metric['mae_gain_macro'],
                rmse_gain=metric['rmse_gain_pooled'], mape_gain=metric['mape_gain_pooled'],
                rmse_gain_macro=metric['rmse_gain_macro'], mape_gain_macro=metric['mape_gain_macro'],
                control_mae=metric['control']['mae_pooled'], candidate_mae=metric['candidate']['mae_pooled'],
                control_rmse=metric['control']['rmse_pooled'], candidate_rmse=metric['candidate']['rmse_pooled'],
                control_mape=metric['control']['mape_pooled'], candidate_mape=metric['candidate']['mape_pooled'],
                gain_positive=metric['gain_positive_pooled'], gain_negative=metric['gain_negative_pooled'],
                cancellation=metric['cancellation_pooled'],
                global_mae_contribution=metric['global_mae_contribution_pooled'],
                global_mae_contribution_macro=metric['global_mae_contribution_macro'],
                global_mape_contribution=metric['global_mape_contribution_pooled'],
                global_mape_contribution_macro=metric['global_mape_contribution_macro'],
                prediction_signed_mean=metric['prediction_change']['signed_mean_candidate_minus_control_pooled'],
                prediction_abs_mean=metric['prediction_change']['absolute_mean_pooled'],
                response_fraction=metric['prediction_change']['response_fraction'],
                prediction_up_fraction=metric['prediction_change']['up_fraction'],
                prediction_down_fraction=metric['prediction_change']['down_fraction'],
                improved_fraction=metric['improved_fraction'], worsened_fraction=metric['worsened_fraction'])
            rows.append(pooled)
            for h in range(12):
                positive, negative = metric['per_horizon_gain_positive'][h], metric['per_horizon_gain_negative'][h]
                rows.append(dict(common, horizon=h+1, aggregate='individual_horizon',
                    valid_count=metric['valid_count_per_horizon'][h], mae_gain=metric['per_horizon_mae_gain'][h],
                    positive_target_count=metric['positive_target_count_per_horizon'][h],
                    mae_gain_macro=None, rmse_gain=metric['per_horizon_rmse_gain'][h],
                    mape_gain=metric['per_horizon_mape_gain'][h], gain_positive=positive, gain_negative=negative,
                    rmse_gain_macro=None, mape_gain_macro=None,
                    control_mae=metric['control']['per_horizon_mae'][h], candidate_mae=metric['candidate']['per_horizon_mae'][h],
                    control_rmse=metric['control']['per_horizon_rmse'][h], candidate_rmse=metric['candidate']['per_horizon_rmse'][h],
                    control_mape=metric['control']['per_horizon_mape'][h], candidate_mape=metric['candidate']['per_horizon_mape'][h],
                    cancellation=_cancellation(positive, negative),
                    global_mae_contribution=metric['per_horizon_global_mae_contribution'][h], global_mae_contribution_macro=None,
                    global_mape_contribution=metric['per_horizon_global_mape_contribution'][h], global_mape_contribution_macro=None,
                    prediction_signed_mean=metric['prediction_change']['per_horizon_signed_mean'][h],
                    prediction_abs_mean=metric['prediction_change']['per_horizon_absolute_mean'][h],
                    response_fraction=metric['prediction_change']['per_horizon_response_fraction'][h],
                    prediction_up_fraction=metric['prediction_change']['per_horizon_up_fraction'][h],
                    prediction_down_fraction=metric['prediction_change']['per_horizon_down_fraction'][h],
                    improved_fraction=metric['per_horizon_improved_fraction'][h],
                    worsened_fraction=metric['per_horizon_worsened_fraction'][h]))
    return rows
