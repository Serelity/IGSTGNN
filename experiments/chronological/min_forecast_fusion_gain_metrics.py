"""Pure NumPy diagnostics of eight archived F1 prediction states.

No model, raw traffic, checkpoint, torch, or training entrypoint is imported.
All differences use the same saved targets and validity, one horizon at a time.
An arithmetic decomposition describes predictions, not training or incident causality.
"""
from collections.abc import Mapping
import itertools

import numpy as np

from experiments.chronological.min_early_event_gain_metrics import (
    BIASES, COUNT_FIELDS, FLOW_EDGES, PARTITION, SUM_FIELDS, _divide, _label,
    _macro, _mean, _summary, _values, _validate as _validate_saved_axes,
    flat_rows, require,
)
from experiments.chronological.min_early_event_direction_metrics import DIRECTIONS, direction_codes


SCHEMA = 'min_forecast_fusion_saved_gain_metrics_v1'
BASE_CASES = ('STATE_ON', 'EVENT_ON', 'EVENT_REPORT_OFF', 'EVENT_FUSION_OFF',
              'EVENT_UNIFORM', 'EVENT_SEMANTIC_UNK', 'EVENT_LOCATION_SHUFFLE', 'SOURCE_MIN')
COMPARISONS = {
    'EVENT_vs_STATE': ('STATE_ON', 'EVENT_ON', 'independently_jointly_trained_model_difference'),
    'report_content_dependency': ('EVENT_REPORT_OFF', 'EVENT_ON', 'same_checkpoint_report_context_sensitivity'),
    'fusion_dependency': ('EVENT_FUSION_OFF', 'EVENT_ON', 'same_checkpoint_entire_fusion_sensitivity'),
    'state_fusion_dependency': ('EVENT_FUSION_OFF', 'EVENT_REPORT_OFF', 'same_checkpoint_report_free_state_fusion_sensitivity'),
    'component_differentiation': ('EVENT_UNIFORM', 'EVENT_ON', 'same_checkpoint_differentiated_vs_mean_coefficients'),
    'common_scaling_dependency': ('EVENT_FUSION_OFF', 'EVENT_UNIFORM', 'same_checkpoint_mean_coefficient_vs_fixed_sum'),
    'semantic_UNK_sensitivity': ('EVENT_SEMANTIC_UNK', 'EVENT_ON', 'same_checkpoint_UNK_sensitivity_potential_OOD'),
    'location_shuffle_sensitivity': ('EVENT_LOCATION_SHUFFLE', 'EVENT_ON', 'same_checkpoint_location_shuffle_potential_OOD'),
    'EVENT_vs_SOURCE': ('SOURCE_MIN', 'EVENT_ON', 'source_MIN_to_jointly_adapted_model_not_structure_only'),
    'STATE_vs_SOURCE': ('SOURCE_MIN', 'STATE_ON', 'source_MIN_to_jointly_adapted_state_model'),
    'current_backbone_vs_SOURCE': ('SOURCE_MIN', 'EVENT_FUSION_OFF', 'updated_MIN_vs_source_MIN_not_training_causal_attribution'),
    'remaining_model_difference': ('STATE_ON', 'EVENT_REPORT_OFF', 'remaining_independent_model_difference_not_training_attribution'),
}
IDENTITIES = {
    'source_to_EVENT': ('EVENT_vs_SOURCE', 'current_backbone_vs_SOURCE', 'fusion_dependency'),
    'fusion_state_and_report': ('fusion_dependency', 'state_fusion_dependency', 'report_content_dependency'),
    'fusion_common_and_differentiated': ('fusion_dependency', 'common_scaling_dependency', 'component_differentiation'),
    'model_difference_and_report': ('EVENT_vs_STATE', 'remaining_model_difference', 'report_content_dependency'),
}
DIRECTION_COMPARISONS = ('fusion_dependency', 'report_content_dependency')


def _validate(predictions, target, valid, regions, samples, stations, hard_ids, edges, response_tol):
    require(isinstance(predictions, Mapping) and set(BASE_CASES) <= predictions.keys(),
            'All eight F1 archived prediction cases are required')
    # Reuse the established axis/mask/ID contract, with a local alias dictionary;
    # never change the old module's globals or the caller's prediction mapping.
    aliases = {name: predictions[source] for name, source in (
        ('STATE_ON', 'STATE_ON'), ('EVENT_ON', 'EVENT_ON'), ('EVENT_OFF', 'EVENT_REPORT_OFF'),
        ('EVENT_SEMANTIC_UNK', 'EVENT_SEMANTIC_UNK'), ('EVENT_LOCATION_SHUFFLE', 'EVENT_LOCATION_SHUFFLE'))}
    _, target, valid, masks, samples, stations, hard_ids, edges, response_tol = _validate_saved_axes(
        aliases, target, valid, regions, samples, stations, hard_ids, edges, response_tol)
    arrays = {case: np.asarray(predictions[case]) for case in BASE_CASES}
    require(all(a.shape == target.shape and a.dtype.kind in 'fiu' and np.isfinite(a[valid]).all()
                for a in arrays.values()), 'All eight predictions must share numeric axes and finite valid positions')
    return arrays, target, valid, masks, samples, stations, hard_ids, edges, response_tol


def _groups(masks, edges):
    names, definitions, offsets = [], {}, {}
    def add(name, family, **fields):
        require(name not in definitions, 'Reserved F1 group conflicts with a supplied region: '+name)
        names.append(name)
        definitions[name] = dict(family=family, **fields)
    for name in masks:
        add(name, 'road' if name.startswith('road:') else 'region', region=name)
    flow_names = [_label(lo, edges[i+1] if i+1 < len(edges) else None) for i, lo in enumerate(edges)]
    offsets['flow'] = len(names)
    for flow in flow_names:
        add('flow:'+flow, 'flow', flow=flow)
    offsets['bias'] = len(names)
    for bias in BIASES:
        add('bias:'+bias, 'current_MIN_bias', bias=bias)
    offsets['source_bias'] = len(names)
    for bias in BIASES:
        add('source_bias:'+bias, 'source_MIN_bias', bias=bias)
    offsets['cross'] = len(names)
    for flow, bias, region in itertools.product(flow_names, BIASES, PARTITION):
        add('cross:flow='+flow+'|bias='+bias+'|region='+region, 'flow_current_MIN_bias_report_partition',
            flow=flow, bias=bias, region=region)
    offsets['fixed'] = len(names)
    add('fixed_hard_windows', 'fixed_SOURCE_MIN_hard_windows')
    return names, definitions, offsets


def _accumulate(acc, h, selections, sums, counts, absolute_delta):
    for offset, size, selected, labels in selections:
        for field, value in sums.items():
            acc[field][offset:offset+size, h] = np.bincount(labels, weights=value[selected], minlength=size)
        for field, value in counts.items():
            acc[field][offset:offset+size, h] = np.bincount(labels[value[selected]], minlength=size)
        np.maximum.at(acc['prediction_max'][:, h], labels+offset, absolute_delta[selected])


def analyze_predictions(predictions, target, valid, regions, sample_indices, station_ids,
                        fixed_hard_sample_ids, flow_edges=FLOW_EDGES, response_tol=.001):
    """Locate fusion/report benefits and harms without new inference or fitting.

    Primary bias groups use current EVENT_FUSION_OFF, while source bias remains
    separately labelled. Direction categories use each comparison's own control.
    Target flow and residual signs are posthoc groups, never inference inputs.
    """
    arrays, target, valid, masks, samples, stations, hard_ids, edges, response_tol = _validate(
        predictions, target, valid, regions, sample_indices, station_ids, fixed_hard_sample_ids, flow_edges, response_tol)
    names, definitions, offsets = _groups(masks, edges)
    g, horizons = len(names), 12
    accumulators = {key: {field: np.zeros((g, horizons), np.int64 if field in COUNT_FIELDS else np.float64)
        for field in (*SUM_FIELDS, *COUNT_FIELDS, 'prediction_max')} for key in COMPARISONS}
    direction_counts = {key: np.zeros((g, horizons, 8), np.int64) for key in DIRECTION_COMPARISONS}
    direction_gains = {key: np.zeros((g, horizons, 8), np.float64) for key in DIRECTION_COMPARISONS}
    # Four mutually exclusive sign quadrants for each two-part decomposition.
    overlap_counts = {key: np.zeros((horizons, 4), np.int64) for key in IDENTITIES}
    overlap_gains = {key: np.zeros((horizons, 4, 3), np.float64) for key in IDENTITIES}
    max_point = dict.fromkeys(IDENTITIES, 0.)
    partition_ids = np.zeros(target.shape[::2], np.int8)
    for index, name in enumerate(PARTITION):
        partition_ids[masks[name]] = index
    fixed_rows = np.isin(samples, hard_ids)
    window_counts = np.zeros((len(samples), horizons), np.int64)
    window_errors = {case: np.zeros((len(samples), horizons)) for case in BASE_CASES}
    eps = np.finfo(np.float64).eps
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        try:
            for h in range(horizons):
                active = valid[:, h]
                window_counts[:, h] = active.sum(1)
                y = target[:, h][active].astype(np.float64)
                if not len(y):
                    continue
                values = {case: arrays[case][:, h][active].astype(np.float64) for case in BASE_CASES}
                errors = {case: np.abs(value-y) for case, value in values.items()}
                percentage = {case: np.divide(error, y, out=np.zeros_like(y), where=y > 0.) for case, error in errors.items()}
                gains = {key: errors[left]-errors[right] for key, (left, right, _) in COMPARISONS.items()}
                scale = sum(errors.values())+1.
                for key, (total, first, second) in IDENTITIES.items():
                    residual = gains[total]-gains[first]-gains[second]
                    max_point[key] = max(max_point[key], float(np.abs(residual).max()))
                    require((np.abs(residual) <= 1e-12+32*eps*scale).all(), 'Pointwise decomposition failed: '+key)
                    labels = (gains[first] > 0).astype(np.int8)*2+(gains[second] > 0).astype(np.int8)
                    overlap_counts[key][h] = np.bincount(labels, minlength=4)
                    for j, comparison in enumerate((total, first, second)):
                        overlap_gains[key][h, :, j] = np.bincount(labels, weights=gains[comparison], minlength=4)
                flow_ids = np.searchsorted(edges, y, side='right')-1
                bias_ids = np.where(values['EVENT_FUSION_OFF'] > y, 0, np.where(values['EVENT_FUSION_OFF'] < y, 1, 2))
                source_ids = np.where(values['SOURCE_MIN'] > y, 0, np.where(values['SOURCE_MIN'] < y, 1, 2))
                cross_ids = (flow_ids*3+bias_ids)*3+partition_ids[active]
                selections = []
                for i, mask in enumerate(masks.values()):
                    selected = mask[active]
                    selections.append((i, 1, selected, np.zeros(int(selected.sum()), np.int64)))
                entire = np.ones(len(y), bool)
                selections += [(offsets['flow'], len(edges), entire, flow_ids),
                    (offsets['bias'], 3, entire, bias_ids), (offsets['source_bias'], 3, entire, source_ids),
                    (offsets['cross'], len(edges)*9, entire, cross_ids)]
                selected = np.broadcast_to(fixed_rows[:, None], active.shape)[active]
                selections.append((offsets['fixed'], 1, selected, np.zeros(int(selected.sum()), np.int64)))
                row_ids = np.broadcast_to(np.arange(len(samples))[:, None], active.shape)[active]
                for case in BASE_CASES:
                    window_errors[case][:, h] = np.bincount(row_ids, weights=errors[case], minlength=len(samples))
                for key, (left, right, _) in COMPARISONS.items():
                    delta, gain = values[right]-values[left], gains[key]
                    sums = dict(control_absolute=errors[left], candidate_absolute=errors[right],
                        control_squared=np.square(errors[left]), candidate_squared=np.square(errors[right]),
                        control_percentage=percentage[left], candidate_percentage=percentage[right],
                        gain_net=gain, gain_positive=np.maximum(gain, 0.), gain_negative=np.maximum(-gain, 0.),
                        prediction_signed=delta, prediction_absolute=np.abs(delta))
                    counts = dict(count=entire, positive_target_count=y > 0., prediction_up=delta > 0.,
                        prediction_down=delta < 0., prediction_unchanged=delta == 0.,
                        prediction_response=np.abs(delta) > response_tol, gain_improved=gain > 0.,
                        gain_worsened=gain < 0., gain_equal=gain == 0.)
                    _accumulate(accumulators[key], h, selections, sums, counts, np.abs(delta))
                    if key in DIRECTION_COMPARISONS:
                        directions = direction_codes(values[left], values[right], y)
                        for offset, size, selected, labels in selections:
                            bins = labels*8+directions[selected]
                            direction_counts[key][offset:offset+size, h] = np.bincount(bins, minlength=size*8).reshape(size, 8)
                            direction_gains[key][offset:offset+size, h] = np.bincount(bins, weights=gain[selected], minlength=size*8).reshape(size, 8)
        except FloatingPointError as error:
            raise ValueError('Finite archived values overflowed float64 F1 arithmetic') from error
    require(all(np.isfinite(a).all() for acc in accumulators.values() for a in acc.values())
            and all(np.isfinite(a).all() for a in (*direction_gains.values(), *overlap_gains.values())),
            'Nonfinite F1 accumulation')
    global_index = names.index('all_nodes')
    first_acc = accumulators['EVENT_vs_STATE']
    global_counts, global_positive = first_acc['count'][global_index], first_acc['positive_target_count'][global_index]
    comparisons = {key: dict(control=left, candidate=right, scope=scope,
        groups={name: _summary(accumulators[key], i, global_counts, global_positive) for i, name in enumerate(names)})
        for key, (left, right, scope) in COMPARISONS.items()}
    cases = {}
    for key, (left, right, _) in COMPARISONS.items():
        cases[left] = comparisons[key]['groups']['all_nodes']['control']
        cases[right] = comparisons[key]['groups']['all_nodes']['candidate']
    families = dict(report_partition=[names.index(name) for name in PARTITION],
        flow=list(range(offsets['flow'], offsets['bias'])), current_MIN_bias=list(range(offsets['bias'], offsets['source_bias'])),
        source_MIN_bias=list(range(offsets['source_bias'], offsets['cross'])),
        cross=list(range(offsets['cross'], offsets['fixed'])))
    reconciliation = {}
    for key, acc in accumulators.items():
        reconciliation[key] = {}
        for family, ids in families.items():
            count = acc['count'][ids].sum(0)-global_counts
            positive = acc['positive_target_count'][ids].sum(0)-global_positive
            net = acc['gain_net'][ids].sum(0)-acc['gain_net'][global_index]
            pct = acc['control_percentage']-acc['candidate_percentage']
            pct_residual = pct[ids].sum(0)-pct[global_index]
            scale = acc['control_absolute'][global_index]+acc['candidate_absolute'][global_index]+1.
            pct_scale = acc['control_percentage'][global_index]+acc['candidate_percentage'][global_index]+1.
            require((count == 0).all() and (positive == 0).all() and (np.abs(net) <= 1e-10+128*eps*scale).all()
                    and (np.abs(pct_residual) <= 1e-10+128*eps*pct_scale).all(), 'F1 family conservation failed: '+key+' '+family)
            reconciliation[key][family] = dict(status='PASS', count_residual_per_horizon=count.tolist(),
                maximum_absolute_gain_sum_residual=float(np.abs(net).max()),
                maximum_absolute_percentage_gain_sum_residual=float(np.abs(pct_residual).max()))
    decompositions, overlap = {}, {}
    for key, (total, first, second) in IDENTITIES.items():
        acc = accumulators[total]
        residual = acc['gain_net']-accumulators[first]['gain_net']-accumulators[second]['gain_net']
        scale = sum(accumulators[name]['control_absolute']+accumulators[name]['candidate_absolute']
                    for name in (total, first, second))+1.
        require((np.abs(residual) <= 1e-10+128*eps*scale).all(), 'Grouped decomposition failed: '+key)
        decompositions[key] = dict(status='POINTWISE_AND_GROUPED_IDENTITY_PASS', total=total, parts=[first, second],
            max_pointwise_residual=max_point[key], max_grouped_residual=float(np.abs(residual).max()),
            checked_points=int(global_counts.sum()), interpretation='same_point_arithmetic_not_causal_attribution')
        require(np.array_equal(overlap_counts[key].sum(1), global_counts), 'Overlap counts do not exhaust all_nodes')
        quadrants = {}
        for j, (a_positive, b_positive) in enumerate(itertools.product((False, True), repeat=2)):
            label = ('first_positive' if a_positive else 'first_nonpositive')+'|'+('second_positive' if b_positive else 'second_nonpositive')
            count_h, sums = overlap_counts[key][:, j], overlap_gains[key][:, j]
            quadrants[label] = dict(valid_count=int(count_h.sum()), within_all_nodes_fraction=_mean(count_h.sum(), global_counts.sum()),
                contributions={comparison: dict(pooled=_mean(sums[:, k].sum(), global_counts.sum()),
                    macro=_macro(_divide(sums[:, k], global_counts), global_counts)) for k, comparison in enumerate((total, first, second))})
        overlap[key] = dict(total=total, first=first, second=second, quadrants=quadrants,
            disjoint_exhaustive=True, sign_threshold=0.)
    directions_result = {}
    for key in DIRECTION_COMPARISONS:
        direction_groups = {}
        acc = accumulators[key]
        for i, name in enumerate(names):
            count_h = acc['count'][i]
            net_h = direction_gains[key][i].sum(1)
            require(np.array_equal(direction_counts[key][i].sum(1), count_h), 'Directions do not exhaust group counts')
            scale = acc['control_absolute'][i]+acc['candidate_absolute'][i]+1.
            require((np.abs(net_h-acc['gain_net'][i]) <= 1e-10+128*eps*scale).all(), 'Direction gain conservation failed')
            motions = {}
            for j, label in enumerate(DIRECTIONS):
                counts, net = direction_counts[key][i, :, j], direction_gains[key][i, :, j]
                motions[label] = dict(valid_count=int(counts.sum()), valid_count_per_horizon=counts.tolist(),
                    within_group_fraction=_mean(counts.sum(), count_h.sum()), global_point_fraction=_mean(counts.sum(), global_counts.sum()),
                    gain_sum=float(net.sum()), gain_pooled=_mean(net.sum(), counts.sum()),
                    global_mae_contribution_pooled=_mean(net.sum(), global_counts.sum()),
                    global_mae_contribution_macro=_macro(_divide(net, global_counts), global_counts),
                    per_horizon_mae_gain=_values(_divide(net, counts), counts),
                    per_horizon_global_mae_contribution=_values(_divide(net, global_counts), global_counts))
            direction_groups[name] = dict(valid_count=int(count_h.sum()), directions=motions)
        left, right, _ = COMPARISONS[key]
        directions_result[key] = dict(control=left, candidate=right, direction_baseline=left, groups=direction_groups,
            definitions=dict(categories=list(DIRECTIONS), exact_zero_classification=True,
                response_tolerance_does_not_change_directions=True, toward_can_overshoot_and_worsen=True))
    complete = (window_counts > 0).all(1)
    fixed_complete = fixed_rows & complete
    window_values = {case: _divide(error, window_counts).mean(1) for case, error in window_errors.items()}
    fixed = dict(selector='saved_SOURCE_MIN_fixed_sample_ids_no_reselection', sample_indices=hard_ids.tolist(),
        selected_count=int(len(hard_ids)), complete_window_count=int(fixed_complete.sum()), comparisons={})
    for key, (left, right, _) in COMPARISONS.items():
        gain = window_values[left][fixed_complete]-window_values[right][fixed_complete]
        fixed['comparisons'][key] = dict(window_equal_weight_mean_gain=float(gain.mean()) if gain.size else None,
            window_improved_fraction=float((gain > 0).mean()) if gain.size else None,
            window_worsened_fraction=float((gain < 0).mean()) if gain.size else None,
            point_metrics=comparisons[key]['groups']['fixed_hard_windows'])
    return dict(schema=SCHEMA, status='MIN_FORECAST_FUSION_GAIN_ANALYSIS_PASS', cases=cases, comparisons=comparisons,
        group_definitions=definitions, decompositions=decompositions,
        identity_check=dict(status='FOUR_POINTWISE_AND_GROUPED_IDENTITIES_PASS', decompositions=decompositions),
        reconciliation=reconciliation, directions=directions_result, overlap=overlap, fixed_hard_windows=fixed,
        axis=dict(samples=len(samples), horizons=12, stations=len(stations), sample_indices=samples.astype(int).tolist(), station_ids=stations.astype(int).tolist()),
        definitions=dict(gain='control_absolute_error_minus_candidate_absolute_error',
            primary_paper_baseline='original_IGSTGNN_not_in_this_eight_case_archive',
            SOURCE_MIN_role='intermediate_structural_control_not_original_IGSTGNN',
            current_MIN_bias='EVENT_FUSION_OFF_minus_target_fixed_for_every_comparison',
            source_MIN_bias='SOURCE_MIN_minus_target_separately_labelled',
            report_direction_bias='EVENT_REPORT_OFF_minus_target_not_FUSION_OFF_or_SOURCE',
            group_macro='equal_mean_of_12_group_horizon_means_only_when_all_nonempty',
            pooled='all_saved_valid_points_equal_weight', global_contribution='common_all_nodes_denominator_pooled_or_per_horizon_macro',
            additive_group_families=list(families), group_macro_and_global_contribution_may_have_opposite_signs=True,
            mape_unit='fraction', mape_zero_policy='original_validity_preserved_positive_targets_only_for_MAPE',
            prediction_change='candidate_minus_control_positive_is_upward', prediction_response_tol=response_tol,
            response_tolerance_affects_response_counts_only=True, grouping_uses_targets_posthoc_only=True,
            positive_negative_parts_not_additive_across_decompositions=True,
            net_path_over_total_gain_ratio_not_computed=True, training_and_incident_causal_attribution=False,
            empty_groups='local_means_unavailable_common_denominator_zero_mass_contribution_zero',
            independence='overlapping_windows_and_points_are_not_independent_incident_replicates',
            scope='exploratory_validation_saved_predictions_no_training_no_new_inference_no_test'))
