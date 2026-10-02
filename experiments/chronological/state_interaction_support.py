"""Fit-frozen observable-state support diagnostics for fixed adapter predictions."""

from itertools import product

import numpy as np

from experiments.chronological import audit_architecture_regions as regional
from experiments.chronological import state_interaction_fit as fit_stats


FEATURE_NAMES = ('history_mean', 'history_trend', 'history_volatility',
                 'history_missing_fraction', 'report_age_minutes', 'candidate_node_count')
SUPPORT_GROUPS = ('missing_state', 'outside_fit_range', 'low_joint_support', 'supported')
SUPPORT_COMPARISONS = {'state_vector_vs_A': ['A', 'state_vector'],
                       'interaction_vector_vs_A': ['A', 'interaction_vector']}
STATE_SPEC = {'quantiles': [.25, .5, .75],
              'joint_features': ['history_mean', 'history_trend', 'candidate_node_count'],
              'joint_quantiles': [.5], 'minimum_fit_windows': 30, 'minimum_fit_weeks': 4}
PATHS, REGIONS, PHASES = fit_stats.PATHS, fit_stats.REGIONS, fit_stats.PHASES
ESTIMANDS = {'pooled': 'gain_raw_mae',
             'equal_forecast_window': 'equal_forecast_window_gain_raw_mae'}


def _number(value):
    return regional.finite_number(value)


def _settings(spec):
    if not isinstance(spec, dict) or set(spec) != set(STATE_SPEC):
        raise ValueError('State settings must contain the original five fields')
    for key in ('quantiles', 'joint_features', 'joint_quantiles'):
        if not isinstance(spec[key], list) or spec[key] != STATE_SPEC[key]:
            raise ValueError(f'State {key} must match the frozen v12j definition')
    for key in ('minimum_fit_windows', 'minimum_fit_weeks'):
        fit_stats._integer(spec[key], f'State {key}', 1)
        if spec[key] != STATE_SPEC[key]:
            raise ValueError(f'State {key} must match the frozen v12j definition')


def _validate_inputs(records, features, clocks, bootstrap_spec, state_spec):
    if any(not isinstance(value, dict) or set(value) != set(PHASES)
           for value in (records, features, clocks)):
        raise ValueError('Require exactly fit and audit records, features and timestamps')
    fit_stats._bootstrap_settings(bootstrap_spec)
    _settings(state_spec)
    parsed, all_ids, all_indices, nodes = {}, set(), set(), None
    weeks, week_index = {}, {}
    for phase in PHASES:
        record = records[phase]
        parsed[phase] = fit_stats._validate(record, clocks[phase], phase)
        current_nodes = record['candidate_mask'].shape[1]
        if nodes is not None and current_nodes != nodes:
            raise ValueError('Fit and audit must use the same node axis')
        nodes = current_nodes
        ids, indices = set(record['ids'].tolist()), set(record['source_indices'].tolist())
        if all_ids.intersection(ids) or all_indices.intersection(indices):
            raise ValueError('Forecast identities/source indices must be disjoint across phases')
        all_ids.update(ids)
        all_indices.update(indices)
        values = features[phase]
        if not isinstance(values, dict) or set(values) != set(FEATURE_NAMES):
            raise ValueError(f'{phase} requires the exact six observable state features')
        for name in FEATURE_NAMES:
            array = values[name]
            if (not isinstance(array, np.ndarray) or array.dtype.kind != 'f'
                    or array.shape != record['ids'].shape or np.isinf(array).any()):
                raise ValueError(f'{phase} {name} must be an aligned float vector with no infinity')
        finite = np.isfinite(values['candidate_node_count'])
        if not np.array_equal(values['candidate_node_count'][finite],
                              record['candidate_mask'].sum(1)[finite]):
            raise ValueError(f'{phase} candidate_node_count disagrees with candidate mask')
        fraction = values['history_missing_fraction']
        if np.any((fraction < 0) | (fraction > 1)):
            raise ValueError(f'{phase} history_missing_fraction must be in [0, 1] or missing')
        if np.any(values['history_volatility'] < 0):
            raise ValueError(f'{phase} volatility must be nonnegative or missing')
        age = values['report_age_minutes']
        if np.any((age <= 0) | (age > 5)):
            raise ValueError(f'{phase} report age must be in (0, 5] minutes or missing')
        weeks[phase], positions = regional.week_grid(clocks[phase])
        if bootstrap_spec['sensitivity_circular_block_weeks'] > len(weeks[phase]):
            raise ValueError(f'{phase} has fewer calendar weeks than configured block')
        week_index[phase] = np.asarray([positions[int(sample)] for sample in record['ids']],
                                      dtype=np.int64)
    try:
        ordered = max(parsed['fit']) < min(parsed['audit'])
    except TypeError as error:
        raise ValueError('Forecast timestamps must have consistent timezone awareness') from error
    if not ordered:
        raise ValueError('Fit and audit forecast clocks must be chronologically disjoint')
    return weeks, week_index


def _bin_definition(values, quantiles):
    finite = values[np.isfinite(values)]
    if not len(finite):
        return {'fit_finite_windows': 0, 'fit_minimum': None, 'fit_maximum': None,
                'cuts': [], 'in_range_bins': 0, 'finite_reference_available': False}
    minimum, maximum = float(finite.min()), float(finite.max())
    cuts = np.unique(np.quantile(finite, quantiles))
    # With right ties a minimum cut is always empty; a maximum cut can isolate
    # a nonempty tied upper state and must not merge its minority lower state.
    cuts = cuts[cuts > minimum]
    return {'fit_finite_windows': len(finite), 'fit_minimum': minimum,
            'fit_maximum': maximum, 'cuts': cuts.tolist(), 'in_range_bins': len(cuts) + 1,
            'finite_reference_available': True}


def _bin_values(values, definition):
    labels = np.full(len(values), 'missing_state', dtype=object)
    finite = np.isfinite(values)
    labels[finite] = 'no_finite_fit_reference'
    if definition['finite_reference_available']:
        below, above = finite & (values < definition['fit_minimum']), finite & (values > definition['fit_maximum'])
        inside = finite & ~below & ~above
        labels[below], labels[above] = 'below_fit_range', 'above_fit_range'
        indices = np.searchsorted(definition['cuts'], values[inside], side='right')
        labels[inside] = [f'in_range_{value}' for value in indices]
    return labels


def _cell_name(indices):
    return ':'.join(str(index) for index in indices)


def _states(records, features, times, weeks, week_index, spec):
    marginal = {name: _bin_definition(features['fit'][name], spec['quantiles'])
                for name in FEATURE_NAMES}
    joint = {name: _bin_definition(features['fit'][name], spec['joint_quantiles'])
             for name in spec['joint_features']}
    membership, cells, eligible = {}, {}, {}
    for phase in PHASES:
        membership[phase] = {name: _bin_values(features[phase][name], marginal[name])
                             for name in FEATURE_NAMES}
        finite = np.logical_and.reduce([np.isfinite(features[phase][name]) for name in FEATURE_NAMES])
        inside = np.logical_and.reduce([np.asarray([label.startswith('in_range_') for label in membership[phase][name]])
                                        for name in FEATURE_NAMES])
        eligible[phase] = finite & inside
        joint_labels = [_bin_values(features[phase][name], joint[name]) for name in spec['joint_features']]
        cells[phase] = np.full(len(records[phase]['ids']), None, dtype=object)
        for index in np.flatnonzero(eligible[phase]):
            cells[phase][index] = _cell_name([int(labels[index].removeprefix('in_range_')) for labels in joint_labels])
    cell_definitions = {}
    for indices in product(*[range(joint[name]['in_range_bins']) for name in spec['joint_features']]):
        cell = _cell_name(indices)
        mask = cells['fit'] == cell
        windows, nonempty_weeks = int(mask.sum()), len(np.unique(week_index['fit'][mask]))
        cell_definitions[cell] = {'indices': list(indices), 'fit_windows': windows,
                                  'fit_nonempty_weeks': nonempty_weeks,
                                  'supported': windows >= spec['minimum_fit_windows']
                                  and nonempty_weeks >= spec['minimum_fit_weeks']}
    support, state_rows = {}, []
    for phase in PHASES:
        missing = np.logical_or.reduce([np.isnan(features[phase][name]) for name in FEATURE_NAMES])
        labels = np.full(len(missing), 'low_joint_support', dtype=object)
        labels[~eligible[phase]], labels[missing] = 'outside_fit_range', 'missing_state'
        for index in np.flatnonzero(eligible[phase]):
            if cell_definitions[cells[phase][index]]['supported']:
                labels[index] = 'supported'
        support[phase] = labels
        for index, sample in enumerate(records[phase]['ids']):
            cell = cells[phase][index]
            info = cell_definitions.get(cell, {})
            row = {'phase': phase, 'sample_index': int(sample),
                   'source_index': int(records[phase]['source_indices'][index]),
                   't0': times[phase][int(sample)],
                   'positive_week': weeks[phase][week_index[phase][index]],
                   'support_group': labels[index], 'joint_cell': cell,
                   'joint_cell_fit_windows': info.get('fit_windows'),
                   'joint_cell_fit_nonempty_weeks': info.get('fit_nonempty_weeks')}
            for name in FEATURE_NAMES:
                row[name], row[f'{name}_bin'] = _number(features[phase][name][index]), membership[phase][name][index]
            state_rows.append(row)
    return marginal, joint, cell_definitions, membership, cells, support, state_rows


def _group_definitions(marginal):
    groups = []
    for name, definition in marginal.items():
        labels = ['below_fit_range', *[f'in_range_{index}' for index in range(definition['in_range_bins'])],
                  'above_fit_range', 'missing_state']
        if not definition['finite_reference_available']:
            labels.append('no_finite_fit_reference')
        groups.extend(('marginal', name, label) for label in labels)
    groups.extend(('support', '', label) for label in SUPPORT_GROUPS)
    return groups


def _phase_analysis(record, weeks, indices, membership, support, groups, spec, phase):
    counts, errors, prediction = (record[key] for key in ('counts', 'errors', 'prediction_counts'))
    gains = np.stack([errors[:, PATHS.index(pair[0])] - errors[:, PATHS.index(pair[1])]
                      for pair in SUPPORT_COMPARISONS.values()], axis=1)
    window_gains = regional.ratio(gains, counts[:, None, :])
    ng, nw, nc, nr = len(groups), len(weeks), len(SUPPORT_COMPARISONS), len(REGIONS)
    week_gain, week_equal = np.zeros((nw, ng, nc, nr)), np.zeros((nw, ng, nc, nr))
    week_cells, week_windows = np.zeros((nw, ng, nr)), np.zeros((nw, ng, nr))
    masks = []
    for group_index, (kind, feature, label) in enumerate(groups):
        mask = (membership[feature] if kind == 'marginal' else support) == label
        masks.append(mask)
        np.add.at(week_gain[:, group_index], indices[mask], gains[mask])
        np.add.at(week_equal[:, group_index], indices[mask], np.nan_to_num(window_gains[mask], nan=0.))
        np.add.at(week_cells[:, group_index], indices[mask], counts[mask])
        np.add.at(week_windows[:, group_index], indices[mask], counts[mask] > 0)
    group_cells, group_windows = week_cells.sum(0), week_windows.sum(0)
    total_cells, total_windows = counts.sum(0), (counts > 0).sum(0)
    group_gain, group_equal = week_gain.sum(0), week_equal.sum(0)
    pooled = regional.ratio(group_gain, group_cells[:, None, :])
    equal = regional.ratio(group_equal, group_windows[:, None, :])
    contribution = regional.ratio(group_gain, total_cells[None, None, :])
    equal_contribution = regional.ratio(group_equal, total_windows[None, None, :])
    global_contribution = regional.ratio(group_gain, total_cells[0])
    draws = {}
    for method, block in (('week', 1), ('four_week_block', spec['sensitivity_circular_block_weeks'])):
        weights = regional.bootstrap_weights(nw, spec['draws'], spec['seed'] + (method != 'week'), block)
        numerator = (weights @ week_gain.reshape(nw, -1)).reshape(-1, ng, nc, nr)
        equal_num = (weights @ week_equal.reshape(nw, -1)).reshape(numerator.shape)
        denominator = (weights @ week_cells.reshape(nw, -1)).reshape(-1, ng, nr)
        equal_den = (weights @ week_windows.reshape(nw, -1)).reshape(denominator.shape)
        draws[method] = {'pooled': regional.ratio(numerator, denominator[:, :, None, :]),
                         'equal_forecast_window': regional.ratio(equal_num, equal_den[:, :, None, :])}
    analysis = {'weeks': weeks, 'samples': len(indices), 'groups': [], 'partition_accounting': [],
                'interval_scope': ('descriptive_in_sample_conditional_on_selected_fitted_weights' if phase == 'fit'
                                   else 'posthoc_reused_development_period_conditional_on_selected_fitted_weights')}
    gain_rows, weekly_rows = [], []
    for group_index, ((kind, feature, label), mask) in enumerate(zip(groups, masks)):
        identity = {'grouping': kind, 'feature': feature, 'group': label}
        item = {**identity, 'forecast_windows': int(mask.sum()),
                'forecast_window_fraction': float(mask.mean()),
                'nonempty_forecast_weeks': len(np.unique(indices[mask])), 'regions': {}}
        for region_index, region in enumerate(REGIONS):
            cell_count = int(group_cells[group_index, region_index])
            region_item = {'valid_cells': cell_count,
                           'prediction_cells': int(prediction[mask, region_index].sum()),
                           'evaluable_forecast_windows': int(group_windows[group_index, region_index]),
                           'phase_region_valid_cell_fraction': _number(regional.ratio(cell_count, total_cells[region_index])),
                           'phase_region_evaluable_window_fraction': _number(regional.ratio(group_windows[group_index, region_index], total_windows[region_index])),
                           'mae': {path: _number(regional.ratio(errors[mask, path_index, region_index].sum(), cell_count))
                                   for path_index, path in enumerate(PATHS)}, 'comparisons': {}}
            for comparison_index, (name, pair) in enumerate(SUPPORT_COMPARISONS.items()):
                confidence = {method: {estimand: regional.interval(array[:, group_index, comparison_index, region_index],
                                         spec['confidence'], spec['minimum_valid_draw_fraction'])
                                       for estimand, array in values.items()} for method, values in draws.items()}
                effect = {'gain_raw_mae': _number(pooled[group_index, comparison_index, region_index]),
                          'equal_forecast_window_gain_raw_mae': _number(equal[group_index, comparison_index, region_index]),
                          'gain_contribution_to_phase_region_raw_mae': _number(contribution[group_index, comparison_index, region_index]),
                          'equal_forecast_window_gain_contribution_to_phase_region_raw_mae': _number(equal_contribution[group_index, comparison_index, region_index]),
                          'gain_contribution_to_phase_global_raw_mae': _number(global_contribution[group_index, comparison_index, region_index]),
                          'intervals': confidence}
                region_item['comparisons'][name] = effect
                row = {'phase': phase, **identity, 'region': region, 'comparison': name,
                       'left_path': pair[0], 'right_path': pair[1],
                       'forecast_windows': item['forecast_windows'],
                       **{key: value for key, value in region_item.items() if key not in ('mae', 'comparisons')},
                       'mae_left': region_item['mae'][pair[0]], 'mae_right': region_item['mae'][pair[1]],
                       **{key: value for key, value in effect.items() if key != 'intervals'},
                       'status': 'DEFINED' if cell_count else 'UNDEFINED_EMPTY_SUPPORT'}
                for method, intervals in confidence.items():
                    for estimand, interval in intervals.items():
                        row.update({f'{method}_{estimand}_{key}': value for key, value in interval.items()})
                gain_rows.append(row)
                for week_index, week in enumerate(weeks):
                    wg = week_gain[week_index, group_index, comparison_index, region_index]
                    we = week_equal[week_index, group_index, comparison_index, region_index]
                    wc, wn = week_cells[week_index, group_index, region_index], week_windows[week_index, group_index, region_index]
                    value = _number(regional.ratio(wg, wc))
                    weekly_rows.append({'phase': phase, **identity, 'positive_week': week,
                        'region': region, 'comparison': name,
                        'forecast_windows': int((mask & (indices == week_index)).sum()),
                        'valid_cells': int(wc), 'evaluable_forecast_windows': int(wn),
                        'gain_raw_mae': value, 'equal_forecast_window_gain_raw_mae': _number(regional.ratio(we, wn)),
                        'gain_sign': ('UNDEFINED' if value is None else 'POSITIVE' if value > 0 else 'NEGATIVE' if value < 0 else 'ZERO'),
                        'status': 'DEFINED' if value is not None else 'UNDEFINED_EMPTY_SUPPORT'})
            item['regions'][region] = region_item
        analysis['groups'].append(item)
    for kind, feature in [('marginal', name) for name in FEATURE_NAMES] + [('support', '')]:
        selected = [index for index, group in enumerate(groups) if group[:2] == (kind, feature)]
        if not np.all(np.stack([masks[index] for index in selected]).sum(0) == 1):
            raise ValueError('State groups must partition every forecast window exactly once')
        for comparison_index, name in enumerate(SUPPORT_COMPARISONS):
            for region_index, region in enumerate(REGIONS):
                full_pooled = _number(regional.ratio(gains[:, comparison_index, region_index].sum(), total_cells[region_index]))
                full_equal = _number(regional.ratio(np.nan_to_num(window_gains[:, comparison_index, region_index], nan=0.).sum(), total_windows[region_index]))
                part_pooled = _number(contribution[selected, comparison_index, region_index].sum())
                part_equal = _number(equal_contribution[selected, comparison_index, region_index].sum())
                full_global = _number(regional.ratio(gains[:, comparison_index, region_index].sum(), total_cells[0]))
                part_global = _number(global_contribution[selected, comparison_index, region_index].sum())
                for expected, actual in ((full_pooled, part_pooled), (full_equal, part_equal),
                                         (full_global, part_global)):
                    if (expected is None) != (actual is None) or (expected is not None and not np.isclose(expected, actual, rtol=1e-10, atol=1e-8)):
                        raise ValueError('State contributions do not reconstruct the full phase region gain')
                analysis['partition_accounting'].append({'grouping': kind, 'feature': feature,
                    'region': region, 'comparison': name, 'full_phase_gain_raw_mae': full_pooled,
                    'sum_group_gain_contributions_raw_mae': part_pooled,
                    'full_phase_equal_forecast_window_gain_raw_mae': full_equal,
                    'sum_group_equal_forecast_window_gain_contributions_raw_mae': part_equal,
                    'full_phase_region_gain_in_global_mae_units': full_global,
                    'sum_group_gain_contributions_to_phase_global_raw_mae': part_global,
                    'reconstruction_passed': True})
    return analysis, gain_rows, weekly_rows


def _composition(records, cells, support, cell_definitions):
    rows = []
    supported_cells = [cell for cell, definition in cell_definitions.items() if definition['supported']]
    for region_index, region in enumerate(REGIONS):
        shared = [cell for cell in supported_cells if all(
            records[phase]['counts'][cells[phase] == cell, region_index].sum() > 0 for phase in PHASES)]
        phase_masks = {phase: np.isin(cells[phase], shared) for phase in PHASES}
        for comparison, pair in SUPPORT_COMPARISONS.items():
            left, right = (PATHS.index(path) for path in pair)
            for estimand in ESTIMANDS:
                masses, cell_gains, coverage = {}, {}, {}
                for phase in PHASES:
                    counts = records[phase]['counts'][:, region_index]
                    numerator = records[phase]['errors'][:, left, region_index] - records[phase]['errors'][:, right, region_index]
                    if estimand == 'equal_forecast_window':
                        numerator = np.nan_to_num(regional.ratio(numerator, counts), nan=0.)
                        weights = (counts > 0).astype(np.int64)
                    else:
                        weights = counts
                    masses[phase] = np.asarray([weights[cells[phase] == cell].sum() for cell in shared], dtype=float)
                    cell_gains[phase] = np.asarray([regional.ratio(numerator[cells[phase] == cell].sum(), mass)
                                                   for cell, mass in zip(shared, masses[phase])])
                    total, supported_mass, common_mass = int(weights.sum()), int(weights[support[phase] == 'supported'].sum()), int(weights[phase_masks[phase]].sum())
                    coverage[phase] = {'total_region_mass': total, 'supported_region_mass': supported_mass,
                        'shared_supported_region_mass': common_mass, 'excluded_unsupported_region_mass': total - supported_mass,
                        'excluded_supported_unshared_region_mass': supported_mass - common_mass,
                        'shared_supported_fraction_of_all_region_mass': _number(regional.ratio(common_mass, total)),
                        'shared_supported_fraction_of_supported_region_mass': _number(regional.ratio(common_mass, supported_mass))}
                values = dict.fromkeys(('fit_common_gain_raw_mae', 'audit_common_gain_raw_mae',
                    'audit_fit_mass_standardized_gain_raw_mae', 'composition_component_raw_mae',
                    'within_cell_component_raw_mae', 'audit_minus_fit_common_gain_raw_mae', 'decomposition_residual_raw_mae'))
                if shared:
                    fit_mass = masses['fit'] / masses['fit'].sum()
                    audit_mass = masses['audit'] / masses['audit'].sum()
                    fit_gain, audit_gain = float(fit_mass @ cell_gains['fit']), float(audit_mass @ cell_gains['audit'])
                    standardized = float(fit_mass @ cell_gains['audit'])
                    composition, within = audit_gain - standardized, standardized - fit_gain
                    gap, residual = audit_gain - fit_gain, audit_gain - fit_gain - composition - within
                    if not np.isclose(residual, 0., rtol=0., atol=1e-10):
                        raise ValueError('Composition accounting does not reconstruct common-cell phase gap')
                    values = dict(zip(values, (fit_gain, audit_gain, standardized, composition, within, gap, residual)))
                row = {'region': region, 'comparison': comparison, 'estimand': estimand,
                    'mass_unit': 'valid_target_cell' if estimand == 'pooled' else 'evaluable_forecast_window',
                    'shared_supported_cells': len(shared), 'shared_supported_cell_ids': '|'.join(shared),
                    'status': 'DEFINED' if shared else 'UNDEFINED_NO_SHARED_SUPPORTED_REGION_CELLS',
                    **values, 'paired_windows': False, 'confidence_interval_constructed': False,
                    'scope': 'descriptive_point_accounting_on_shared_supported_cells_not_causal_decomposition'}
                for phase in PHASES:
                    row.update({f'{phase}_{key}': value for key, value in coverage[phase].items()})
                rows.append(row)
    return rows


def analyze_support(phase_records, features_by_phase, times_by_phase, bootstrap_spec, state_spec):
    """Analyze fit-frozen states without training, selection or a deployable router.

    Feature vectors must be in the exact sample order of each phase record. All
    state definitions use features and calendar support only, never error sums.
    Phase intervals share calendar bootstrap weights across groups and models;
    cross-phase accounting is an unpaired, descriptive point comparison only.
    """
    weeks, indices = _validate_inputs(phase_records, features_by_phase, times_by_phase,
                                      bootstrap_spec, state_spec)
    marginal, joint, cell_definitions, membership, cells, support, state_rows = _states(
        phase_records, features_by_phase, times_by_phase, weeks, indices, state_spec)
    groups = _group_definitions(marginal)
    analysis = {'feature_names': list(FEATURE_NAMES), 'marginal_bins': marginal,
        'joint_bins': joint, 'joint_cells': cell_definitions, 'support_group_priority': list(SUPPORT_GROUPS),
        'state_definition_fit_only': True, 'state_definition_uses_errors': False,
        'quantile_tie_side': 'right', 'minimum_endpoint_and_duplicate_cuts_removed': True,
        'nonconstant_maximum_cut_retained': True,
        'automatic_development_gate': False, 'deployable_router_constructed': False,
        'support_is_coarse_feature_space_diagnostic_not_full_input_support_guarantee': True,
        'multiple_comparisons_adjusted': False,
        'new_model_selection_performed': False, 'equal_weight_unit': 'forecast_window_not_unique_accident',
        'gain_direction': 'MAE_A_minus_MAE_vector_positive_is_vector_improvement',
        'composition_scope': 'point_only_shared_supported_cells_no_cross_phase_pairing_or_causal_claim',
        'phases': {}}
    gain_rows, weekly_rows = [], []
    for phase in PHASES:
        result, gains, weekly = _phase_analysis(phase_records[phase], weeks[phase], indices[phase],
                                               membership[phase], support[phase], groups, bootstrap_spec, phase)
        analysis['phases'][phase] = result
        gain_rows.extend(gains)
        weekly_rows.extend(weekly)
    composition_rows = _composition(phase_records, cells, support, cell_definitions)
    analysis['composition_accounting'] = composition_rows
    return analysis, state_rows, gain_rows, weekly_rows, composition_rows
