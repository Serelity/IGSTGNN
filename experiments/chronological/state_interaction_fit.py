"""NumPy-only regional accounting for fixed selected adapters across phases."""

from datetime import datetime

import numpy as np

from experiments.chronological import audit_architecture_regions as regions_audit


PATHS = ('A', 'strength', 'state_vector', 'interaction_vector')
REGIONS = ('all', 'candidate_h1_h6', 'candidate_h7_h12', 'noncandidate_all')
PHASES = ('fit', 'audit')
COMPARISONS = {
    'strength_vs_A': ['A', 'strength'],
    'state_vector_vs_A': ['A', 'state_vector'],
    'interaction_vector_vs_A': ['A', 'interaction_vector'],
    'state_vector_vs_strength': ['strength', 'state_vector'],
    'interaction_vector_vs_strength': ['strength', 'interaction_vector'],
    'interaction_vector_vs_state_vector': ['state_vector', 'interaction_vector'],
}
ESTIMANDS = {
    'pooled': 'gain_raw_mae',
    'equal_forecast_window': 'equal_forecast_window_gain_raw_mae',
    'global_units': 'regional_gain_in_global_mae_units',
}


def _integer(value, label, minimum=0):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f'{label} must be an integer >= {minimum}')


def _bootstrap_settings(spec):
    names = {'draws', 'seed', 'confidence', 'sensitivity_circular_block_weeks',
             'minimum_valid_draw_fraction'}
    if not isinstance(spec, dict) or set(spec) != names:
        raise ValueError('Bootstrap settings must contain the original five fields')
    _integer(spec['draws'], 'Bootstrap draws', 1)
    _integer(spec['seed'], 'Bootstrap seed')
    _integer(spec['sensitivity_circular_block_weeks'], 'Circular block weeks', 1)
    for name in ('confidence', 'minimum_valid_draw_fraction'):
        value = spec[name]
        if (isinstance(value, bool) or not isinstance(value, (int, float, np.number))
                or not np.isfinite(value) or not 0 < value <= 1):
            raise ValueError(f'Bootstrap {name} must be finite and in (0, 1]')
    if spec['confidence'] == 1:
        raise ValueError('Bootstrap confidence must be less than 1')


def _validate(record, times, phase):
    if (not isinstance(record, dict) or not isinstance(record.get('regions'), list)
            or record['regions'] != list(REGIONS)):
        raise ValueError(f'{phase} must contain the exact four-region axis')
    required = ('ids', 'source_indices', 'candidate_mask', 'counts', 'prediction_counts', 'errors')
    if any(not isinstance(record.get(name), np.ndarray) for name in required):
        raise ValueError(f'{phase} must contain NumPy identity, support and error arrays')
    ids, indices, mask, counts, predicted, errors = (record[k] for k in required)
    n = len(ids) if ids.ndim else 0
    if (n == 0 or ids.dtype.kind not in 'iu' or indices.dtype.kind not in 'iu'
            or ids.shape != (n,) or indices.shape != (n,) or np.any(indices < 0)
            or np.any(ids < 0) or len(np.unique(ids)) != n or len(np.unique(indices)) != n):
        raise ValueError(f'{phase} has invalid or duplicate sample identities/source indices')
    if mask.dtype.kind != 'b' or mask.ndim != 2 or mask.shape[0] != n or mask.shape[1] == 0:
        raise ValueError(f'{phase} candidate support must be a nonempty-node boolean matrix')
    if (counts.shape != (n, 4) or predicted.shape != counts.shape or errors.shape != (n, 4, 4)
            or counts.dtype.kind not in 'iu' or predicted.dtype.kind not in 'iu'
            or errors.dtype.kind != 'f'):
        raise ValueError(f'{phase} has invalid count/error shapes or dtypes')
    if (not np.isfinite(errors).all() or np.any(errors < 0) or np.any(counts < 0)
            or np.any(predicted < counts) or np.any(predicted < 0)
            or np.any(np.where(counts[:, None, :] == 0, errors, 0.) != 0)):
        raise ValueError(f'{phase} has invalid error sums or target support')
    geometry = np.stack((np.full(n, mask.shape[1] * 12), mask.sum(1) * 6,
                         mask.sum(1) * 6, (~mask).sum(1) * 12), axis=-1)
    if not np.array_equal(predicted, geometry):
        raise ValueError(f'{phase} prediction counts disagree with 12-horizon candidate geometry')
    for value in (counts, predicted):
        if not np.array_equal(value[:, 0], value[:, 1:].sum(1)):
            raise ValueError(f'{phase} regional counts do not partition all cells')
    if counts[:, 0].sum() == 0:
        raise ValueError(f'{phase} full-positive global target support is empty')
    if not np.allclose(errors[:, :, 0], errors[:, :, 1:].sum(-1), rtol=1e-10, atol=1e-7):
        raise ValueError(f'{phase} regional error sums do not partition all cells')
    if (not isinstance(times, dict) or set(times) != set(ids.tolist())
            or any(isinstance(key, bool) or not isinstance(key, (int, np.integer)) for key in times)):
        raise ValueError(f'{phase} timestamps must match the exact sample identity set')
    try:
        parsed = [datetime.fromisoformat(times[int(sample)]) for sample in ids]
    except (TypeError, ValueError) as error:
        raise ValueError(f'{phase} has invalid ISO forecast timestamps') from error
    return parsed


def _number(value):
    return regions_audit.finite_number(value)


def _status(value):
    return 'DEFINED' if value is not None else 'UNDEFINED_EMPTY_SUPPORT'


def _phase_analysis(record, times, comparisons, spec, phase):
    weeks, positions = regions_audit.week_grid(times)
    if spec['sensitivity_circular_block_weeks'] > len(weeks):
        raise ValueError(f'{phase} has fewer calendar weeks than the configured block')
    index = np.asarray([positions[int(sample)] for sample in record['ids']], dtype=np.int64)
    errors, counts, predicted = (record[k] for k in ('errors', 'counts', 'prediction_counts'))
    pairs = list(comparisons.values())
    gains = np.stack([errors[:, PATHS.index(left)] - errors[:, PATHS.index(right)]
                      for left, right in pairs], axis=1)
    shape = (len(weeks), len(pairs), len(REGIONS))
    wg, we = np.zeros(shape), np.zeros(shape)
    wc, wn = np.zeros((len(weeks), 4)), np.zeros((len(weeks), 4))
    np.add.at(wg, index, gains)
    np.add.at(wc, index, counts)
    window_gain = regions_audit.ratio(gains, counts[:, None, :])
    np.add.at(we, index, np.nan_to_num(window_gain, nan=0.))
    np.add.at(wn, index, counts > 0)
    pooled = regions_audit.ratio(wg.sum(0), wc.sum(0)[None, :])
    equal_window = regions_audit.ratio(we.sum(0), wn.sum(0)[None, :])
    global_units = regions_audit.ratio(wg.sum(0), wc.sum(0)[None, :1])
    if wc[:, 0].sum() > 0 and not np.allclose(global_units[:, 0], global_units[:, 1:].sum(-1),
                                             rtol=1e-10, atol=1e-8):
        raise ValueError(f'{phase} valid-cell contributions do not reconstruct global gains')
    draws = {}
    for method, block in (('week', 1), ('four_week_block', spec['sensitivity_circular_block_weeks'])):
        weights = regions_audit.bootstrap_weights(len(weeks), spec['draws'],
                                                  spec['seed'] + (method != 'week'), block)
        sampled_gains = (weights @ wg.reshape(len(weeks), -1)).reshape(-1, len(pairs), 4)
        sampled_window = (weights @ we.reshape(len(weeks), -1)).reshape(sampled_gains.shape)
        sampled_counts = weights @ wc
        draws[method] = {
            'pooled': regions_audit.ratio(sampled_gains, sampled_counts[:, None, :]),
            'equal_forecast_window': regions_audit.ratio(sampled_window, (weights @ wn)[:, None, :]),
            'global_units': regions_audit.ratio(sampled_gains, sampled_counts[:, None, :1]),
        }
    total_cells, total_prediction = counts.sum(0), predicted.sum(0)
    analysis = {'weeks': weeks, 'samples': len(index), 'nonempty_positive_weeks': len(set(index)),
                'interval_scope': ('descriptive_in_sample_conditional_on_selected_fitted_weights'
                                   if phase == 'fit' else
                                   'posthoc_reused_development_period_conditional_on_selected_fitted_weights'),
                'regions': {}}
    phase_rows, weekly_rows = [], []
    weekly_estimates = {
        'pooled': regions_audit.ratio(wg, wc[:, None, :]),
        'equal_forecast_window': regions_audit.ratio(we, wn[:, None, :]),
        'global_units': regions_audit.ratio(wg, wc[:, None, :1]),
    }
    for r, region in enumerate(REGIONS):
        cells = int(total_cells[r])
        item = {'valid_cells': cells, 'prediction_cells': int(total_prediction[r]),
                'evaluable_samples': int((counts[:, r] > 0).sum()),
                'valid_cell_fraction': _number(regions_audit.ratio(total_cells[r], total_cells[0])),
                'mae': {path: _number(regions_audit.ratio(errors[:, p, r].sum(), cells))
                        for p, path in enumerate(PATHS)}, 'comparisons': {}}
        for c, (name, pair) in enumerate(comparisons.items()):
            effect = {'gain_raw_mae': _number(pooled[c, r]),
                      'equal_forecast_window_gain_raw_mae': _number(equal_window[c, r]),
                      'regional_gain_in_global_mae_units': _number(global_units[c, r]),
                      'intervals': {method: {estimand: regions_audit.interval(values[:, c, r],
                                       spec['confidence'], spec['minimum_valid_draw_fraction'])
                                       for estimand, values in sampled.items()}
                                    for method, sampled in draws.items()}}
            item['comparisons'][name] = effect
            row = {'phase': phase, 'region': region, 'comparison': name,
                   'left_path': pair[0], 'right_path': pair[1], 'mae_left': item['mae'][pair[0]],
                   'mae_right': item['mae'][pair[1]], 'valid_cells': cells,
                   'prediction_cells': item['prediction_cells'], 'evaluable_samples': item['evaluable_samples'],
                   'valid_cell_fraction': item['valid_cell_fraction'],
                   **{field: effect[field] for field in ESTIMANDS.values()}}
            for method, intervals in effect['intervals'].items():
                for estimand, confidence in intervals.items():
                    row.update({f'{method}_{estimand}_{key}': value for key, value in confidence.items()})
            phase_rows.append(row)
            for w, week in enumerate(weeks):
                estimates = {field: _number(weekly_estimates[estimand][w, c, r])
                             for estimand, field in ESTIMANDS.items()}
                weekly_rows.append({'phase': phase, 'positive_week': week, 'region': region,
                    'comparison': name, 'forecast_windows': int((index == w).sum()),
                    'valid_cells': int(wc[w, r]), 'global_valid_cells': int(wc[w, 0]),
                    'evaluable_samples': int(wn[w, r]),
                    'valid_cell_fraction': _number(regions_audit.ratio(wc[w, r], wc[w, 0])),
                    **estimates, 'status': _status(estimates['gain_raw_mae'])})
        analysis['regions'][region] = item
    return analysis, phase_rows, weekly_rows


def analyze_phases(phase_records, comparisons, bootstrap_spec, times_by_phase):
    """Compare fixed predictions in disjoint phases; never select models or epochs.

    Counts/masks/identities are shared across the four paths in each input record.
    The reader must check that path-specific source metadata agrees before stacking
    errors. Cross-phase differences are descriptive unpaired point differences.
    """
    if (not isinstance(phase_records, dict) or set(phase_records) != set(PHASES)
            or not isinstance(times_by_phase, dict) or set(times_by_phase) != set(PHASES)):
        raise ValueError('Require exactly fit and audit with matching timestamp phases')
    if (not isinstance(comparisons, dict) or list(comparisons) != list(COMPARISONS)
            or comparisons != COMPARISONS):
        raise ValueError('Require all six fixed directed path comparisons in original order')
    _bootstrap_settings(bootstrap_spec)
    all_ids, all_indices, parsed, node_count = set(), set(), {}, None
    order = [phase for phase in PHASES if phase in phase_records]
    for phase in order:
        record = phase_records[phase]
        parsed[phase] = _validate(record, times_by_phase[phase], phase)
        current_nodes = record['candidate_mask'].shape[1]
        if node_count is not None and current_nodes != node_count:
            raise ValueError('Fit and audit must use the same node axis')
        node_count = current_nodes
        ids, indices = set(record['ids'].tolist()), set(record['source_indices'].tolist())
        if all_ids.intersection(ids) or all_indices.intersection(indices):
            raise ValueError('Forecast identities/source indices must be disjoint across phases')
        all_ids.update(ids)
        all_indices.update(indices)
    try:
        ordered = all(max(parsed[left]) < min(parsed[right]) for left, right in zip(order, order[1:]))
    except TypeError as error:
        raise ValueError('Forecast timestamps must have consistent timezone awareness') from error
    if not ordered:
        raise ValueError('Fit and audit forecast clocks must be chronologically disjoint')
    analysis = {'phases': {}, 'equal_weight_unit': 'forecast_window_not_unique_accident',
                'gain_direction': 'MAE_left_minus_MAE_right_positive_is_right_path_improvement',
                'cross_phase_gap_scope': 'descriptive_unpaired_point_difference_no_gap_confidence_interval',
                'new_model_selection_performed': False, 'automatic_development_gate': False}
    phase_rows, weekly_rows, gap_rows = [], [], []
    for phase in order:
        result, rows, weeks = _phase_analysis(phase_records[phase], times_by_phase[phase],
                                              comparisons, bootstrap_spec, phase)
        analysis['phases'][phase] = result
        phase_rows.extend(rows)
        weekly_rows.extend(weeks)
    for phase in order[1:]:
        for region in REGIONS:
            fit = analysis['phases']['fit']['regions'][region]
            current = analysis['phases'][phase]['regions'][region]
            for comparison in comparisons:
                for estimand, field in ESTIMANDS.items():
                    f, c = fit['comparisons'][comparison][field], current['comparisons'][comparison][field]
                    gap = c - f if c is not None and f is not None else None
                    gap_rows.append({'phase': phase, 'reference_phase': 'fit', 'region': region,
                        'comparison': comparison, 'estimand': estimand, 'fit_gain_raw_mae': f,
                        'phase_gain_raw_mae': c, 'phase_minus_fit_gain_raw_mae': gap,
                        'status': _status(gap), 'paired_windows': False,
                        'gap_confidence_interval_constructed': False})
    return analysis, phase_rows, weekly_rows, gap_rows
