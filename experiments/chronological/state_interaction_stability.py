"""NumPy-only diagnostics of saved v12g weekly sufficient statistics."""

import re
from datetime import date, timedelta

import numpy as np

from experiments.chronological.audit_architecture_regions import (
    bootstrap_weights, finite_number, interval, ratio,
)


COHORTS = ('incident_full', 'incident', 'primary_control', 'secondary_control',
           'incident_full_common', 'incident_full_complement')
GROUPS = COHORTS[-2:]
REGIONS = ('all', 'candidate_h1_h6', 'candidate_h7_h12', 'noncandidate_all')
COMPARISONS = ('strength_vs_A', 'state_vector_vs_A', 'interaction_vector_vs_A',
               'state_vector_vs_strength', 'interaction_vector_vs_strength',
               'interaction_vector_vs_state_vector')
ESTIMANDS = ('pooled', 'equal_forecast_window')


def _close(left, right, label, atol=1e-8):
    if not np.allclose(left, right, rtol=1e-10, atol=atol):
        raise ValueError(f'{label} disagrees')


def _number_close(left, right, label):
    if ((left is None) != (right is None) or
            (left is not None and (not isinstance(right, (int, float)) or
                                  not np.isfinite(right) or
                                  not np.isclose(left, right, rtol=1e-10, atol=1e-9)))):
        raise ValueError(f'{label} disagrees')


def _interval_close(observed, declared, label):
    if not isinstance(declared, dict):
        raise ValueError(f'{label} missing')
    for key in ('status', 'valid_draws'):
        if observed[key] != declared.get(key):
            raise ValueError(f'{label}/{key} disagrees')
    for key in ('ci_low', 'ci_high'):
        _number_close(observed[key], declared.get(key), f'{label}/{key}')


def _axis(arrays, name, expected):
    value = np.asarray(arrays[name])
    if value.ndim != 1 or value.dtype.kind not in 'US' or value.tolist() != list(expected):
        raise ValueError(f'Saved {name} axis/order changed')


def _validate(arrays, source_analysis, comparisons, spec):
    if list(comparisons) != list(COMPARISONS):
        raise ValueError('Require all six frozen comparisons in source order')
    _axis(arrays, 'comparisons', comparisons)
    _axis(arrays, 'regions', REGIONS)
    weeks_array = np.asarray(arrays['weeks'])
    if weeks_array.ndim != 1 or weeks_array.dtype.kind not in 'US':
        raise ValueError('Invalid saved calendar-week axis')
    weeks = weeks_array.tolist()
    if (len(weeks) < 4 or len(set(weeks)) != len(weeks) or
            any(not re.fullmatch(r'\d{4}-W\d{2}', week) for week in weeks) or
            source_analysis.get('weeks') != weeks):
        raise ValueError('Saved/source calendar weeks changed')
    try:
        mondays = [date.fromisocalendar(int(week[:4]), int(week[-2:]), 1) for week in weeks]
    except ValueError as error:
        raise ValueError('Invalid ISO calendar-week value') from error
    if any(right - left != timedelta(weeks=1) for left, right in zip(mondays, mondays[1:])):
        raise ValueError('Saved calendar grid must retain consecutive weeks')
    required_spec = {'draws': 2000, 'seed': 12028, 'confidence': 0.95,
                     'sensitivity_circular_block_weeks': 4,
                     'minimum_valid_draw_fraction': 0.95}
    if any(spec.get(key) != value for key, value in required_spec.items()):
        raise ValueError('Frozen source bootstrap settings changed')
    weights = {}
    for method, seed, block in (('week', spec['seed'], 1),
                                ('four_week_block', spec['seed'] + 1, 4)):
        value = np.asarray(arrays[f'bootstrap_{method}_weights'])
        expected = bootstrap_weights(len(weeks), spec['draws'], seed, block)
        if value.dtype.kind not in 'iu' or not np.array_equal(value, expected):
            raise ValueError(f'Saved {method} bootstrap weights changed')
        weights[method] = value
    if set(source_analysis.get('results', {})) != set(COHORTS):
        raise ValueError('Saved/source cohort set changed')
    values = {}
    shape = (len(weeks), len(comparisons), len(REGIONS))
    for cohort in COHORTS:
        source_result = source_analysis['results'][cohort]
        sample_count = source_result.get('samples')
        if (not isinstance(sample_count, (int, np.integer)) or
                isinstance(sample_count, (bool, np.bool_)) or sample_count < 0):
            raise ValueError(f'{cohort} source sample budget must be a nonnegative integer')
        source_regions = source_result.get('regions', {})
        if (set(source_regions) != set(REGIONS) or
                any(set(source_regions[region].get('comparisons', {})) != set(COMPARISONS)
                    for region in REGIONS)):
            raise ValueError(f'{cohort} source-summary region/comparison set changed')
        item = {}
        for field, expected_shape in (('gain_sums', shape), ('event_gain_sums', shape),
                                       ('valid_counts', (len(weeks), len(REGIONS))),
                                       ('evaluable_events', (len(weeks), len(REGIONS)))):
            value = np.asarray(arrays[f'{cohort}_{field}'])
            if (value.shape != expected_shape or value.dtype.kind not in 'iuf' or
                    not np.isfinite(value).all()):
                raise ValueError(f'{cohort}/{field} geometry or finite-value mismatch')
            if field in ('valid_counts', 'evaluable_events'):
                if (np.any(value < 0) or np.any(value != np.floor(value)) or
                        np.any(value > 2 ** 53)):
                    raise ValueError(f'{cohort}/{field} must be nonnegative integers')
            item[field] = value.astype(np.float64, copy=False)
            if not np.isfinite(item[field]).all():
                raise ValueError(f'{cohort}/{field} exceeds finite float64 range')
        counts, windows = item['valid_counts'], item['evaluable_events']
        if (np.any(windows > counts) or np.any(windows[:, 1:] > windows[:, :1]) or
                np.any((counts == 0) != (windows == 0))):
            raise ValueError(f'{cohort} evaluable-window support disagrees with valid cells')
        if np.any(windows.sum(0) > sample_count):
            raise ValueError(f'{cohort} evaluable forecast windows exceed the source sample budget')
        for field in ('gain_sums', 'event_gain_sums'):
            if np.any(np.where(counts[:, None, :] == 0, item[field], 0.) != 0):
                raise ValueError(f'{cohort} empty regions have nonzero saved gains')
            for destination, left, right in ((3, 1, 0), (4, 2, 0), (5, 2, 1)):
                _close(item[field][:, destination], item[field][:, left] - item[field][:, right],
                       f'{cohort}/{field} comparison algebra', atol=1e-7)
        if not np.array_equal(counts[:, 0], counts[:, 1:].sum(1)):
            raise ValueError(f'{cohort} regional count partition disagrees')
        _close(item['gain_sums'][:, :, 0], item['gain_sums'][:, :, 1:].sum(2),
               f'{cohort} regional gain partition', atol=1e-7)
        values[cohort] = item
    for field in ('valid_counts', 'evaluable_events'):
        if not np.array_equal(values['incident'][field], values['incident_full_common'][field]):
            raise ValueError(f'Matched incident/full-positive common {field} replay support disagrees')
    if values['incident_full']['valid_counts'][:, 0].sum() <= 0:
        raise ValueError('Full-positive global valid-cell denominator is empty')
    for field in ('valid_counts', 'evaluable_events', 'gain_sums', 'event_gain_sums'):
        partition = sum(values[group][field] for group in GROUPS)
        if field in ('valid_counts', 'evaluable_events'):
            if not np.array_equal(values['incident_full'][field], partition):
                raise ValueError(f'Common/complement {field} partition disagrees')
        else:
            _close(values['incident_full'][field], partition,
                   f'Common/complement {field} partition', atol=1e-7)
    return weeks, values, weights


def _statistics(values, weights):
    points, draws = {}, {}
    gain, cells = values['gain_sums'], values['valid_counts']
    event_gain, windows = values['event_gain_sums'], values['evaluable_events']
    points['pooled'] = ratio(gain.sum(0), cells.sum(0)[None, :])
    points['equal_forecast_window'] = ratio(event_gain.sum(0), windows.sum(0)[None, :])
    points['global_units'] = ratio(gain.sum(0), cells.sum(0)[None, :1])
    for method, weight in weights.items():
        sampled_gain = np.einsum('bw,wcr->bcr', weight, gain)
        sampled_event_gain = np.einsum('bw,wcr->bcr', weight, event_gain)
        draws[method] = {'pooled': ratio(sampled_gain, (weight @ cells)[:, None, :]),
                        'equal_forecast_window': ratio(sampled_event_gain, (weight @ windows)[:, None, :]),
                        'global_units': ratio(sampled_gain, (weight @ cells)[:, None, :1])}
    return points, draws


def _sign_counts(values):
    values = np.asarray(values)
    valid = np.isfinite(values)
    return {'positive': int((values[valid] > 0).sum()),
            'negative': int((values[valid] < 0).sum()),
            'zero': int((values[valid] == 0).sum()),
            'undefined': int((~valid).sum())}


def _stability(point, weekly, omitted, weeks):
    valid = omitted[np.isfinite(omitted)]
    return {'point_gain_raw_mae': finite_number(point),
            'weekly_sign_counts': _sign_counts(weekly),
            'leave_one_week_out': {
                'sign_counts': _sign_counts(omitted),
                'minimum_gain_raw_mae': finite_number(valid.min()) if len(valid) else None,
                'maximum_gain_raw_mae': finite_number(valid.max()) if len(valid) else None,
                'sign_flip_from_full_point_count': int(((valid * point) < 0).sum())
                    if np.isfinite(point) else None,
                'weeks': [{'omitted_week': week, 'gain_raw_mae': finite_number(value),
                           'status': 'OK' if np.isfinite(value) else 'UNDEFINED_EMPTY_SUPPORT'}
                          for week, value in zip(weeks, omitted)]}}


def analyze_weekly(arrays, source_analysis, comparisons, bootstrap_spec):
    """Return reconciled stability, paired contrasts, weeks and global contributions.

    All estimates are conditional on the saved models and calendar. The inherited
    ``event`` statistic averages forecast windows, not unique accident identities.
    """
    weeks, values, weights = _validate(arrays, source_analysis, comparisons, bootstrap_spec)
    ci = lambda value: interval(value, bootstrap_spec['confidence'],
                                bootstrap_spec['minimum_valid_draw_fraction'])
    statistics = {cohort: _statistics(item, weights) for cohort, item in values.items()}
    weekly, omitted = {}, {}
    result, weekly_rows = {}, []
    source_fields = {'pooled': 'gain_raw_mae', 'equal_forecast_window': 'equal_event_gain_raw_mae',
                     'global_units': 'regional_gain_in_global_mae_units'}
    for cohort, item in values.items():
        point, sampled = statistics[cohort]
        weekly[cohort], omitted[cohort] = {}, {}
        for estimand, numerator, denominator in (
                ('pooled', item['gain_sums'], item['valid_counts']),
                ('equal_forecast_window', item['event_gain_sums'], item['evaluable_events'])):
            weekly[cohort][estimand] = ratio(numerator, denominator[:, None, :])
            omitted[cohort][estimand] = ratio(numerator.sum(0)[None, :, :] - numerator,
                                             (denominator.sum(0)[None, :] - denominator)[:, None, :])
        result[cohort] = {'regions': {}}
        for r, region in enumerate(REGIONS):
            declared = source_analysis['results'][cohort]['regions'][region]
            if (declared['valid_cells'] != int(item['valid_counts'][:, r].sum()) or
                    declared['evaluable_samples'] != int(item['evaluable_events'][:, r].sum())):
                raise ValueError(f'{cohort}/{region} source summary count mismatch')
            result[cohort]['regions'][region] = {'comparisons': {}}
            for c, comparison in enumerate(comparisons):
                source_effect = declared['comparisons'][comparison]
                for estimand, source_field in source_fields.items():
                    _number_close(finite_number(point[estimand][c, r]), source_effect[source_field],
                                  f'{cohort}/{region}/{comparison}/{source_field}')
                    for method in weights:
                        source_name = 'equal_event' if estimand == 'equal_forecast_window' else estimand
                        _interval_close(ci(sampled[method][estimand][:, c, r]),
                                        source_effect['intervals'][method][source_name],
                                        f'{cohort}/{region}/{comparison}/{method}/{estimand}')
                item_stability = {}
                for estimand in ESTIMANDS:
                    item_stability[estimand] = _stability(point[estimand][c, r],
                        weekly[cohort][estimand][:, c, r], omitted[cohort][estimand][:, c, r], weeks)
                    item_stability[estimand]['intervals'] = {
                        method: ci(sampled[method][estimand][:, c, r]) for method in weights}
                result[cohort]['regions'][region]['comparisons'][comparison] = item_stability
                for w, week in enumerate(weeks):
                    n, windows = (int(item[key][w, r]) for key in ('valid_counts', 'evaluable_events'))
                    weekly_rows.append({'cohort': cohort, 'positive_week': week, 'region': region,
                        'comparison': comparison, 'valid_cells': n, 'evaluable_forecast_windows': windows,
                        'gain_sum': float(item['gain_sums'][w, c, r]),
                        'forecast_window_gain_sum': float(item['event_gain_sums'][w, c, r]),
                        'pooled_gain_raw_mae': finite_number(weekly[cohort]['pooled'][w, c, r]),
                        'equal_forecast_window_gain_raw_mae': finite_number(weekly[cohort]['equal_forecast_window'][w, c, r]),
                        'status': 'OK' if n else 'UNDEFINED_EMPTY_SUPPORT',
                        'leave_one_week_out_pooled_gain_raw_mae': finite_number(omitted[cohort]['pooled'][w, c, r]),
                        'leave_one_week_out_equal_forecast_window_gain_raw_mae': finite_number(omitted[cohort]['equal_forecast_window'][w, c, r])})
    contrasts, contrast_rows = {}, []
    common, complement = GROUPS
    for r, region in enumerate(REGIONS):
        contrasts[region] = {}
        for c, comparison in enumerate(comparisons):
            effects = {}
            for estimand in ESTIMANDS:
                point = statistics[common][0][estimand][c, r] - statistics[complement][0][estimand][c, r]
                week_values = weekly[common][estimand][:, c, r] - weekly[complement][estimand][:, c, r]
                loo = omitted[common][estimand][:, c, r] - omitted[complement][estimand][:, c, r]
                effect = _stability(point, week_values, loo, weeks)
                effect['intervals'] = {method: ci(statistics[common][1][method][estimand][:, c, r] -
                                                 statistics[complement][1][method][estimand][:, c, r])
                                       for method in weights}
                effect['weekly_contrasts'] = [
                    {'positive_week': week, 'gain_raw_mae': finite_number(week_values[w]),
                     'common_valid_cells': int(values[common]['valid_counts'][w, r]),
                     'complement_valid_cells': int(values[complement]['valid_counts'][w, r]),
                     'common_evaluable_forecast_windows': int(values[common]['evaluable_events'][w, r]),
                     'complement_evaluable_forecast_windows': int(values[complement]['evaluable_events'][w, r]),
                     'status': 'OK' if np.isfinite(week_values[w]) else 'UNDEFINED_EMPTY_GROUP_SUPPORT'}
                    for w, week in enumerate(weeks)]
                effects[estimand] = effect
                contrast_rows.append({'region': region, 'comparison': comparison, 'estimand': estimand,
                    'common_minus_complement_gain_raw_mae': effect['point_gain_raw_mae'],
                    'week_ci_status': effect['intervals']['week']['status'],
                    'week_valid_draws': effect['intervals']['week']['valid_draws'],
                    'week_ci_low': effect['intervals']['week']['ci_low'],
                    'week_ci_high': effect['intervals']['week']['ci_high'],
                    'block_ci_status': effect['intervals']['four_week_block']['status'],
                    'block_valid_draws': effect['intervals']['four_week_block']['valid_draws'],
                    'block_ci_low': effect['intervals']['four_week_block']['ci_low'],
                    'block_ci_high': effect['intervals']['four_week_block']['ci_high'],
                    **{f'weekly_{key}_count': value for key, value in effect['weekly_sign_counts'].items()},
                    'leave_one_week_out_minimum': effect['leave_one_week_out']['minimum_gain_raw_mae'],
                    'leave_one_week_out_maximum': effect['leave_one_week_out']['maximum_gain_raw_mae'],
                    'leave_one_week_out_sign_flip_count': effect['leave_one_week_out']['sign_flip_from_full_point_count']})
            contrasts[region][comparison] = effects
    full_cells = values['incident_full']['valid_counts'][:, 0]
    denominator = float(full_cells.sum())
    accounting, contribution_rows = {}, []
    for c, comparison in enumerate(comparisons):
        contributions = {}
        for cohort in ('incident_full', *GROUPS):
            gain = values[cohort]['gain_sums'][:, c]
            contributions[cohort] = {region: float(gain[:, r].sum() / denominator)
                                      for r, region in enumerate(REGIONS)}
            _close(contributions[cohort]['all'], sum(contributions[cohort][region] for region in REGIONS[1:]),
                   'Full-positive-denominator regional contributions')
            for r, region in enumerate(REGIONS):
                contribution_rows.append({'cohort': cohort, 'comparison': comparison, 'region': region,
                    'period': 'overall', 'positive_week': None, 'full_positive_global_valid_cells': int(denominator),
                    'group_region_valid_cells': int(values[cohort]['valid_counts'][:, r].sum()),
                    'gain_sum': float(gain[:, r].sum()), 'contribution_to_full_global_gain': contributions[cohort][region],
                    'status': 'OK'})
                for w, week in enumerate(weeks):
                    contribution_rows.append({'cohort': cohort, 'comparison': comparison, 'region': region,
                        'period': 'weekly', 'positive_week': week, 'full_positive_global_valid_cells': int(full_cells[w]),
                        'group_region_valid_cells': int(values[cohort]['valid_counts'][w, r]),
                        'gain_sum': float(gain[w, r]),
                        'contribution_to_full_global_gain': finite_number(ratio(gain[w, r], full_cells[w])),
                        'status': 'OK' if full_cells[w] else 'UNDEFINED_EMPTY_FULL_POSITIVE_WEEK'})
        for region in REGIONS:
            _close(contributions['incident_full'][region], sum(contributions[group][region] for group in GROUPS),
                   'Full-positive-denominator group contributions')
        accounting[comparison] = {'full_global_gain_raw_mae': contributions['incident_full']['all'],
                                   'contribution_to_full_global_gain': contributions}
    return ({'weeks': weeks, 'cohort_stability': result, 'common_minus_complement': contrasts,
             'full_positive_contributions': accounting,
             'source_statistics_reconciled': True,
             'equal_weight_unit': 'forecast_window_not_unique_accident',
             'contrast_direction': 'positive means common forecast windows benefit more than complement',
             'interval_scope': 'pointwise paired saved-week bootstrap; no multiple-comparison adjustment',
             'leave_one_week_out_scope': 'descriptive influence check, not a new model or time selection'},
            contrast_rows, weekly_rows, contribution_rows)
