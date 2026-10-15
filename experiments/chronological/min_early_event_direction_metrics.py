"""Pure NumPy direction audit of saved EVENT_ON relative to EVENT_OFF.

OFF, not STATE, defines the baseline error whose correction is being examined.
STATE is used only for a separately labelled STATE x OFF diagnostic cross.
"""
import itertools

import numpy as np

from experiments.chronological.min_early_event_gain_metrics import (
    COUNT_FIELDS, FLOW_EDGES, PARTITION, SUM_FIELDS, _divide, _label, _mean,
    _summary, _validate, require,
)


SCHEMA = 'min_early_event_OFF_direction_metrics_v1'
BIASES = ('over', 'under', 'exact')
DIRECTIONS = ('unchanged', 'toward_without_crossing', 'reached_target', 'overshoot_improved',
              'overshoot_equal_error', 'overshoot_worsened', 'away_from_target', 'exact_OFF_changed')
DELTAS = ('up', 'down', 'unchanged')
P_SIGNS = ('positive', 'negative', 'equal')


def direction_codes(off, on, target):
    """Eight mutually exclusive, exhaustive direction categories (exact zero).

    A step initially toward the target can overshoot so far that error grows.
    Response tolerance is deliberately not part of this classification.
    """
    off, on, target = np.asarray(off), np.asarray(on), np.asarray(target)
    require(off.shape == on.shape == target.shape and off.dtype.kind in 'fiu'
            and on.dtype.kind in 'fiu' and target.dtype.kind in 'fiu', 'Direction arrays must share numeric axes')
    off, on, target = off.astype(np.float64, copy=False), on.astype(np.float64, copy=False), target.astype(np.float64, copy=False)
    require(np.isfinite(off).all() and np.isfinite(on).all() and np.isfinite(target).all(), 'Direction values must be finite')
    residual, actual, delta = off-target, on-target, on-off
    unchanged, exact = delta == 0., residual == 0.
    toward = ((residual > 0.) & (delta < 0.)) | ((residual < 0.) & (delta > 0.))
    same_side = ((residual > 0.) & (actual > 0.)) | ((residual < 0.) & (actual < 0.))
    result = np.full(off.shape, -1, np.int8)
    result[unchanged] = 0
    result[toward & same_side & ~unchanged] = 1
    result[toward & (actual == 0.) & ~unchanged] = 2
    crossed = toward & ~same_side & (actual != 0.) & ~unchanged
    off_error, on_error = np.abs(residual), np.abs(actual)
    result[crossed & (on_error < off_error)] = 3
    result[crossed & (on_error == off_error)] = 4
    result[crossed & (on_error > off_error)] = 5
    result[~toward & ~exact & ~unchanged] = 6
    result[exact & ~unchanged] = 7
    require((result >= 0).all(), 'Unclassified correction direction')
    return result


def analyze_direction(predictions, target, valid, regions, sample_indices, station_ids,
                      flow_edges=FLOW_EDGES, response_tol=.001, *, fixed_hard_sample_ids=()):
    arrays, target, valid, masks, samples, stations, hard_ids, edges, response_tol = _validate(
        predictions, target, valid, regions, sample_indices, station_ids, fixed_hard_sample_ids, flow_edges, response_tol)
    names, definitions, families = [], {}, {}
    def add(name, family, **fields):
        require(name not in definitions, 'Reserved direction-group name conflicts with a source region: '+name)
        names.append(name)
        definitions[name] = dict(family=family, **fields)
    for name in masks:
        add(name, 'road' if name.startswith('road:') else 'region', region=name)
    flow_names = [_label(lo, edges[i+1] if i+1 < len(edges) else None) for i, lo in enumerate(edges)]
    def category(family, values):
        offset = len(names)
        for name, fields in values:
            add(name, family, **fields)
        families[family] = list(range(offset, len(names)))
        return offset
    flow_offset = category('flow', [('flow:'+name, dict(flow=name)) for name in flow_names])
    off_offset = category('OFF_bias', [('off_bias:'+bias, dict(OFF_bias=bias)) for bias in BIASES])
    state_off_offset = category('STATE_OFF_bias', [('state_off_bias:STATE_'+state+'|OFF_'+off,
        dict(STATE_bias=state, OFF_bias=off)) for state, off in itertools.product(BIASES, repeat=2)])
    flow_off_offset = category('flow_OFF_bias', [('flow_off_bias:'+flow+'|OFF_'+off,
        dict(flow=flow, OFF_bias=off)) for flow, off in itertools.product(flow_names, BIASES)])
    partition_off_offset = category('partition_OFF_bias', [('partition_off_bias:'+region+'|OFF_'+off,
        dict(region=region, OFF_bias=off)) for region, off in itertools.product(PARTITION, BIASES)])
    direction_offset = category('direction', [('direction:'+name, dict(direction=name)) for name in DIRECTIONS])
    delta_offset = category('prediction_delta', [('delta:'+name, dict(delta=name)) for name in DELTAS])
    p_offset = category('P_sign', [('P_sign:'+name, dict(P_sign=name)) for name in P_SIGNS])
    fixed_offset = len(names)
    add('fixed_hard_windows', 'fixed_STATE_hard_windows')
    group_count, horizons = len(names), 12
    accumulator = {field: np.zeros((group_count, horizons), dtype=np.int64 if field in COUNT_FIELDS else np.float64)
                   for field in (*SUM_FIELDS, *COUNT_FIELDS, 'prediction_max')}
    motion_counts = np.zeros((group_count, horizons, len(DIRECTIONS)), np.int64)
    motion_gains = np.zeros_like(motion_counts, dtype=np.float64)
    partition_ids = np.zeros((target.shape[0], target.shape[2]), np.int8)
    for i, region in enumerate(PARTITION):
        partition_ids[masks[region]] = i
    fixed_rows = np.isin(samples, hard_ids)
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        try:
            for h in range(horizons):
                active = valid[:, h]
                y = target[:, h][active].astype(np.float64)
                if not len(y):
                    continue
                off = arrays['EVENT_OFF'][:, h][active].astype(np.float64)
                on = arrays['EVENT_ON'][:, h][active].astype(np.float64)
                state = arrays['STATE_ON'][:, h][active].astype(np.float64)
                off_error, on_error, delta = np.abs(off-y), np.abs(on-y), on-off
                gain = off_error-on_error
                direction = direction_codes(off, on, y)
                require(np.array_equal(gain > 0., np.isin(direction, (1, 2, 3)))
                        and np.array_equal(gain < 0., np.isin(direction, (5, 6, 7))),
                        'Direction categories disagree with absolute-error improvement signs')
                off_bias = np.where(off > y, 0, np.where(off < y, 1, 2))
                state_bias = np.where(state > y, 0, np.where(state < y, 1, 2))
                flow = np.searchsorted(edges, y, side='right')-1
                delta_ids = np.where(delta > 0., 0, np.where(delta < 0., 1, 2))
                p_ids = np.where(gain > 0., 0, np.where(gain < 0., 1, 2))
                selections = [(i, 1, mask[active], True) for i, mask in enumerate(masks.values())]
                selections += [(flow_offset, len(edges), flow, False), (off_offset, 3, off_bias, False),
                    (state_off_offset, 9, state_bias*3+off_bias, False),
                    (flow_off_offset, len(edges)*3, flow*3+off_bias, False),
                    (partition_off_offset, 9, partition_ids[active]*3+off_bias, False),
                    (direction_offset, len(DIRECTIONS), direction, False), (delta_offset, 3, delta_ids, False),
                    (p_offset, 3, p_ids, False),
                    (fixed_offset, 1, np.broadcast_to(fixed_rows[:, None], active.shape)[active], True)]
                sums = dict(control_absolute=off_error, candidate_absolute=on_error,
                    control_squared=np.square(off_error), candidate_squared=np.square(on_error),
                    control_percentage=np.divide(off_error, y, out=np.zeros_like(y), where=y > 0.),
                    candidate_percentage=np.divide(on_error, y, out=np.zeros_like(y), where=y > 0.),
                    gain_net=gain, gain_positive=np.maximum(gain, 0.), gain_negative=np.maximum(-gain, 0.),
                    prediction_signed=delta, prediction_absolute=np.abs(delta))
                counts = dict(count=np.ones(len(y), bool), positive_target_count=y > 0., prediction_up=delta > 0.,
                    prediction_down=delta < 0., prediction_unchanged=delta == 0., prediction_response=np.abs(delta) > response_tol,
                    gain_improved=gain > 0., gain_worsened=gain < 0., gain_equal=gain == 0.)
                for offset, size, labels, is_mask in selections:
                    selected = labels if is_mask else np.ones(len(y), bool)
                    ids = np.zeros(int(selected.sum()), np.int64) if is_mask else labels
                    for field, value in sums.items():
                        accumulator[field][offset:offset+size, h] = np.bincount(ids, weights=value[selected], minlength=size)
                    for field, value in counts.items():
                        accumulator[field][offset:offset+size, h] = np.bincount(ids[value[selected]], minlength=size)
                    np.maximum.at(accumulator['prediction_max'][:, h], ids+offset, np.abs(delta[selected]))
                    motion_ids = ids*len(DIRECTIONS)+direction[selected]
                    motion_counts[offset:offset+size, h] = np.bincount(motion_ids, minlength=size*len(DIRECTIONS)).reshape(size, -1)
                    motion_gains[offset:offset+size, h] = np.bincount(motion_ids, weights=gain[selected], minlength=size*len(DIRECTIONS)).reshape(size, -1)
        except FloatingPointError as error:
            raise ValueError('Finite saved values overflowed float64 direction arithmetic') from error
    require(all(np.isfinite(value).all() for value in accumulator.values()) and np.isfinite(motion_gains).all(),
            'Nonfinite direction accumulation')
    global_index = names.index('all_nodes')
    global_counts, global_positive = accumulator['count'][global_index], accumulator['positive_target_count'][global_index]
    groups = {name: _summary(accumulator, i, global_counts, global_positive) for i, name in enumerate(names)}
    for i, name in enumerate(names):
        count_h = accumulator['count'][i]
        require(np.array_equal(motion_counts[i].sum(1), count_h), 'Direction categories do not exhaust group counts')
        tolerance = 1e-10+128*np.finfo(np.float64).eps*(accumulator['control_absolute'][i]+accumulator['candidate_absolute'][i]+1.)
        require((np.abs(motion_gains[i].sum(1)-accumulator['gain_net'][i]) <= tolerance).all(), 'Direction gains do not reconcile')
        motions = {}
        for j, label in enumerate(DIRECTIONS):
            counts, net = motion_counts[i, :, j], motion_gains[i, :, j]
            motions[label] = dict(valid_count=int(counts.sum()), valid_count_per_horizon=counts.tolist(),
                within_group_fraction=_mean(counts.sum(), count_h.sum()), global_point_fraction=_mean(counts.sum(), global_counts.sum()),
                P_gain_sum=float(net.sum()), P_gain_pooled=_mean(net.sum(), counts.sum()),
                global_P_contribution_pooled=_mean(net.sum(), global_counts.sum()),
                global_P_contribution_macro=float(_divide(net, global_counts).mean()) if (global_counts > 0).all() else None,
                per_horizon_P_gain=[float(value) if count else None for value, count in zip(_divide(net, counts), counts)],
                per_horizon_global_P_contribution=[float(value) if count else None for value, count in zip(_divide(net, global_counts), global_counts)])
        groups[name]['directions'] = motions
    families['report_partition'] = [names.index(name) for name in PARTITION]
    reconciliations = {}
    for family, ids in families.items():
        count_residual = accumulator['count'][ids].sum(0)-global_counts
        net_residual = accumulator['gain_net'][ids].sum(0)-accumulator['gain_net'][global_index]
        scale = accumulator['control_absolute'][global_index]+accumulator['candidate_absolute'][global_index]+1.
        require((count_residual == 0).all() and (np.abs(net_residual) <= 1e-10+128*np.finfo(np.float64).eps*scale).all(),
                'Direction family does not reconcile: '+family)
        reconciliations[family] = dict(status='PASS', count_residual_per_horizon=count_residual.tolist(),
                                      maximum_absolute_gain_sum_residual=float(np.abs(net_residual).max()))
    return dict(schema=SCHEMA, status='MIN_EARLY_EVENT_OFF_DIRECTION_ANALYSIS_PASS', groups=groups,
        group_definitions=definitions, reconciliation=reconciliations,
        axis=dict(samples=len(samples), horizons=12, stations=len(stations), sample_indices=samples.tolist(), station_ids=stations.tolist()),
        fixed_hard_sample_ids=hard_ids.tolist(), definitions=dict(control='EVENT_OFF', candidate='EVENT_ON',
            P='abs(EVENT_OFF-y)-abs(EVENT_ON-y); positive_is_improvement',
            bias_reference='EVENT_OFF_minus_target_for_over_under_exact',
            STATE_role='additional_STATE_x_OFF_error_sign_cross_only_not_the_correction_baseline',
            prediction_delta='EVENT_ON_minus_EVENT_OFF_positive_is_upward', prediction_response_tol=response_tol,
            direction_gain_and_delta_sign_threshold=0., response_tol_changes_only_prediction_response_fraction=True,
            direction_categories=list(DIRECTIONS), directions_disjoint_exhaustive=True,
            toward_includes_overshoot_even_if_error_grows=True,
            flow_grouping='observed_target_flow_posthoc_only',
            contribution_denominator='all_nodes_valid_count_pooled_or_same_horizon_counts_for_macro',
            group_macro_and_global_contribution_can_have_opposite_signs=True,
            MAPE='fraction_in_JSON_CSV_positive_targets_only', zero_targets='original_validity_preserved_not_reintroduced',
            interpretation='saved_checkpoint_direction_sensitivity_not_incident_causality_or_online_gating',
            natural_no_incident_group_verified=False, windows_points_are_not_independent_incidents=True))


def group_rows(result):
    """One group/horizon row, without any individual prediction records."""
    rows = []
    for group, metric in result['groups'].items():
        definition = result['group_definitions'][group]
        common = dict(group=group, family=definition['family'], region=definition.get('region', ''),
            flow=definition.get('flow', ''), OFF_bias=definition.get('OFF_bias', ''), STATE_bias=definition.get('STATE_bias', ''),
            direction=definition.get('direction', ''), delta=definition.get('delta', ''), P_sign=definition.get('P_sign', ''))
        rows.append(dict(common, horizon=0, valid_count=metric['valid_count'], P_gain=metric['mae_gain_pooled'],
            P_gain_macro=metric['mae_gain_macro'], global_P_contribution=metric['global_mae_contribution_pooled'],
            global_P_contribution_macro=metric['global_mae_contribution_macro'],
            gain_positive=metric['gain_positive_pooled'], gain_negative=metric['gain_negative_pooled'],
            cancellation=metric['cancellation_pooled'], prediction_delta_mean=metric['prediction_change']['signed_mean_candidate_minus_control_pooled'],
            prediction_up_fraction=metric['prediction_change']['up_fraction'], prediction_down_fraction=metric['prediction_change']['down_fraction'],
            improved_fraction=metric['improved_fraction'], worsened_fraction=metric['worsened_fraction']))
        for h in range(12):
            positive, negative = metric['per_horizon_gain_positive'][h], metric['per_horizon_gain_negative'][h]
            cancellation = None if positive is None or positive+negative == 0. else 1.-abs(positive-negative)/(positive+negative)
            rows.append(dict(common, horizon=h+1, valid_count=metric['valid_count_per_horizon'][h], P_gain=metric['per_horizon_mae_gain'][h],
                P_gain_macro=None, global_P_contribution=metric['per_horizon_global_mae_contribution'][h], global_P_contribution_macro=None,
                gain_positive=positive, gain_negative=negative, cancellation=cancellation,
                prediction_delta_mean=metric['prediction_change']['per_horizon_signed_mean'][h],
                prediction_up_fraction=metric['prediction_change']['per_horizon_up_fraction'][h],
                prediction_down_fraction=metric['prediction_change']['per_horizon_down_fraction'][h],
                improved_fraction=metric['per_horizon_improved_fraction'][h], worsened_fraction=metric['per_horizon_worsened_fraction'][h]))
    return rows


def direction_rows(result):
    rows = []
    for group, metric in result['groups'].items():
        for direction, value in metric['directions'].items():
            rows.append(dict(group=group, direction=direction, horizon=0, valid_count=value['valid_count'],
                within_group_fraction=value['within_group_fraction'], P_gain=value['P_gain_pooled'],
                global_P_contribution=value['global_P_contribution_pooled']))
            for h in range(12):
                n = value['valid_count_per_horizon'][h]
                rows.append(dict(group=group, direction=direction, horizon=h+1, valid_count=n,
                    within_group_fraction=_mean(n, metric['valid_count_per_horizon'][h]),
                    P_gain=value['per_horizon_P_gain'][h], global_P_contribution=value['per_horizon_global_P_contribution'][h]))
    return rows
