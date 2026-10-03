"""v12n fixed-output geometry and paired statistics (NumPy only)."""

from datetime import datetime, timedelta

import numpy as np

from experiments.chronological import audit_architecture_regions as regional

CATEGORIES = ('no_change', 'r_zero', 'wrong_direction', 'same_direction_beneficial',
              'neutral_at_2r', 'harmful_overshoot')
QUANTITIES = ('gain', 'slope', 'crossing_penalty')
ESTIMANDS = ('pooled_valid_cells', 'equal_forecast_window')
COHORTS = ('incident', 'primary_control', 'secondary_control')
CONTRASTS = {'incident_minus_mean_controls': (1., -.5, -.5),
             'incident_minus_primary': (1., -1., 0.),
             'incident_minus_secondary': (1., 0., -1.)}
META = ('ids', 'source_indices', 'incident_ids', 'positive_t0', 'cohort_t0',
        'support_start', 'support_end_exclusive')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(left, right, label):
    require(np.allclose(left, right, rtol=1e-10, atol=1e-8, equal_nan=False), label)


def cell_geometry(a, y, d, valid):
    """Invalid Y never enters arithmetic; no thresholding or lambda search."""
    a, y, d, valid = map(np.asarray, (a, y, d, valid))
    require(a.shape == y.shape == d.shape == valid.shape and a.ndim == 1,
            'Cell arrays must be aligned vectors')
    require(valid.dtype == np.bool_ and a.dtype == y.dtype == np.float32
            and d.dtype == np.float64, 'Require FP32 A/Y, FP64 d and boolean validity')
    require(np.isfinite(a).all() and np.isfinite(d).all()
            and np.isfinite(y[valid]).all(), 'Nonfinite prediction/correction/valid target')
    r = y[valid].astype(np.float64) - a[valid].astype(np.float64)
    delta = d[valid]
    require(np.isfinite(r).all(), 'Nonfinite residual')
    same = (r != 0) & (delta != 0) & (np.signbit(r) == np.signbit(delta))
    masks = np.stack((delta == 0, (delta != 0) & (r == 0),
        (r != 0) & (delta != 0) & ~same,
        same & (np.abs(delta) < 2 * np.abs(r)),
        same & (np.abs(delta) == 2 * np.abs(r)),
        same & (np.abs(delta) > 2 * np.abs(r))), axis=1)
    require(np.all(masks.sum(1) == 1), 'Categories do not partition valid cells')
    gain = np.abs(r) - np.abs(r - delta)
    slope = np.where(r == 0, -np.abs(delta), np.sign(r) * delta)
    penalty = np.where(same, 2 * np.maximum(np.abs(delta) - np.abs(r), 0), 0.)
    close(gain, slope - penalty, 'Cell G=S-O failed')
    values = {'gain': gain, 'slope': slope, 'crossing_penalty': penalty,
              'a_error': np.abs(r), 'p_error': np.abs(r - delta)}
    result = {key: np.zeros(len(a), dtype=np.float64) for key in values}
    for key, value in values.items():
        require(np.isfinite(value).all(), 'Nonfinite cell statistic')
        result[key][valid] = value
    result['category'] = np.full(len(a), -1, dtype=np.int64)
    result['category'][valid] = np.argmax(masks, axis=1)
    return result


def validate_packed(record):
    ids = np.asarray(record['ids'])
    n = len(ids)
    require(ids.ndim == 1 and ids.dtype == np.int64 and len(np.unique(ids)) == n
            and np.all(ids >= 0), 'Invalid window IDs')
    for key in META:
        require(np.asarray(record[key]).shape == (n,), f'Invalid window metadata: {key}')
    require(record['source_indices'].dtype == np.int64 and
            np.all(record['source_indices'] >= 0), 'Invalid source indices')
    for key in META[2:]:
        require(record[key].dtype.kind in 'US', f'Metadata must be strings: {key}')
    for start, end, t0 in zip(record['support_start'], record['support_end_exclusive'], record['cohort_t0']):
        first, last, origin = map(datetime.fromisoformat, (str(start), str(end), str(t0)))
        require(first <= origin < last, 'Invalid complete support interval')
    offsets = np.asarray(record['window_offsets'])
    candidate = np.asarray(record['candidate_mask'])
    require(candidate.ndim == 2 and candidate.shape[0] == n and candidate.shape[1] > 0
            and candidate.dtype == np.bool_, 'Invalid candidate mask')
    require(offsets.shape == (n + 1,) and offsets.dtype == np.int64 and offsets[0] == 0
            and np.array_equal(np.diff(offsets), 6 * candidate.sum(1)),
            'Packing must retain every candidate H1-H6 cell')
    k = int(offsets[-1])
    for key in ('candidate_node_indices', 'horizon_indices', 'A', 'Y', 'd', 'valid'):
        require(record[key].shape == (k,), f'Invalid packed length: {key}')
    for key in ('candidate_node_indices', 'horizon_indices'):
        require(record[key].dtype == np.int64, 'Coordinates must be int64')
    for i in range(n):
        span = slice(offsets[i], offsets[i + 1])
        nodes = np.flatnonzero(candidate[i])
        require(np.array_equal(record['candidate_node_indices'][span], np.tile(nodes, 6))
                and np.array_equal(record['horizon_indices'][span], np.repeat(np.arange(1, 7), len(nodes))),
                'Packed cells omitted, duplicated or reordered')
    return np.repeat(np.arange(n), np.diff(offsets))


def window_geometry(record):
    owner = validate_packed(record)
    cells = cell_geometry(record['A'], record['Y'], record['d'], record['valid'])
    n = len(record['ids'])
    out = {key: record[key].copy() for key in META}
    out['category_names'] = np.asarray(CATEGORIES)
    out['valid_counts'] = np.bincount(owner[record['valid']], minlength=n).astype(np.int64)
    for key in (*QUANTITIES, 'a_error', 'p_error'):
        out[key + '_sums'] = np.bincount(owner, weights=cells[key], minlength=n)
    out['category_counts'] = np.stack([np.bincount(owner[cells['category'] == c], minlength=n)
                                      for c in range(6)], axis=1).astype(np.int64)
    for key in QUANTITIES:
        out['category_' + key + '_sums'] = np.stack([
            np.bincount(owner, weights=np.where(cells['category'] == c, cells[key], 0.), minlength=n)
            for c in range(6)], axis=1)
        close(out[key + '_sums'], out['category_' + key + '_sums'].sum(1),
              f'Category {key} contributions do not reconstruct windows')
    require(np.array_equal(out['valid_counts'], out['category_counts'].sum(1)),
            'Category counts do not reconstruct windows')
    close(out['gain_sums'], out['slope_sums'] - out['crossing_penalty_sums'], 'Window G=S-O failed')
    close(out['gain_sums'], out['a_error_sums'] - out['p_error_sums'], 'Window MAE gain failed')
    require(np.all(out['crossing_penalty_sums'] >= 0), 'Negative crossing penalty')
    return out


def calendar(bounds, spec):
    first, last = map(datetime.fromisoformat, bounds)
    require(first.weekday() == last.weekday() == 0 and first < last
            and first.time() == last.time() == datetime.min.time(), 'Require Monday phase boundaries')
    days = [first + timedelta(weeks=i) for i in range((last - first).days // 7)]
    require(len(days) >= 4, 'Need at least four calendar weeks')
    weeks = [f'{day.isocalendar().year}-W{day.isocalendar().week:02d}' for day in days]
    weights = {method: regional.bootstrap_weights(len(weeks), spec['draws'], seed, block)
        for method, seed, block in (('week', spec['week_seed'], 1),
            ('four_week_block', spec['four_week_block_seed'], spec['sensitivity_circular_block_weeks']))}
    return weeks, weights


def week_indices(record, weeks):
    positions = {week: i for i, week in enumerate(weeks)}
    result = []
    for value in record['positive_t0']:
        day = datetime.fromisoformat(str(value)).isocalendar()
        key = f'{day.year}-W{day.week:02d}'
        require(key in positions, 'Window is outside original phase calendar')
        result.append(positions[key])
    return np.asarray(result, dtype=np.int64)


def weighted_mean(values, counts, estimand):
    if estimand == ESTIMANDS[0]:
        return regional.ratio(values.sum(axis=0), counts.sum())
    require(estimand == ESTIMANDS[1], 'Unknown estimand')
    valid = counts > 0
    return regional.ratio((values[valid] / counts[valid, None]).sum(0), valid.sum())


def slope_status(gain, slope, band):
    if not np.isfinite(gain) or not np.isfinite(slope):
        return 'UNDEFINED_SUPPORT'
    if abs(slope) <= band:
        return 'NEAR_ZERO_UNRESOLVED'
    if slope < 0:
        return 'FIXED_SAMPLE_DIRECTION_UNFAVORABLE'
    return 'LOCAL_OPPORTUNITY_WITH_CROSSING_PENALTY' if gain < 0 else 'POSITIVE_LOCAL_DIRECTION'


def analyze_windows(record, weeks, weights, protocol):
    counts = record['valid_counts']
    fields = (*QUANTITIES, 'a_error', 'p_error')
    values = np.stack([record[key + '_sums'] for key in fields], axis=1)
    index = week_indices(record, weeks)
    wc = np.zeros(len(weeks)); wn = np.zeros(len(weeks))
    ws = np.zeros((len(weeks), len(fields))); we = np.zeros_like(ws)
    np.add.at(wc, index, counts); np.add.at(wn, index, counts > 0)
    np.add.at(ws, index, values)
    np.add.at(we, index, np.nan_to_num(regional.ratio(values, counts[:, None]), nan=0.))
    spec = protocol['bootstrap']
    results, category_rows = {}, []
    for estimand in ESTIMANDS:
        point = weighted_mean(values, counts, estimand)
        sums, den = (ws, wc) if estimand == ESTIMANDS[0] else (we, wn)
        intervals = {}
        for method, w in weights.items():
            draws = regional.ratio(w @ sums, (w @ den)[:, None])
            intervals[method] = {key: regional.interval(draws[:, j], spec['confidence'], spec['minimum_valid_draw_fraction'])
                                 for j, key in enumerate(QUANTITIES)}
        results[estimand] = {'point': {key: regional.finite_number(point[j]) for j, key in enumerate(fields)},
            'intervals': intervals,
            'description': slope_status(point[0], point[1], protocol['replay']['slope_near_zero_reporting_band_raw_mae'])}
        for c, category in enumerate(CATEGORIES):
            numerators = np.stack([record['category_' + key + '_sums'][:, c] for key in QUANTITIES], 1)
            contribution = weighted_mean(numerators, counts, estimand)
            within = regional.ratio(numerators.sum(0), record['category_counts'][:, c].sum())
            share = weighted_mean(record['category_counts'][:, c:c+1], counts, estimand)[0]
            category_rows.append({'estimand': estimand, 'category': category,
                'cells': int(record['category_counts'][:, c].sum()), 'mass_fraction': regional.finite_number(share),
                **{key + '_contribution': regional.finite_number(contribution[j]) for j, key in enumerate(QUANTITIES)},
                **{key + '_within_category_cell_mean': regional.finite_number(within[j]) for j, key in enumerate(QUANTITIES)}})
        if counts.sum():
            for j, key in enumerate(QUANTITIES):
                close(sum(row[key + '_contribution'] for row in category_rows if row['estimand'] == estimand),
                      point[j], 'Aggregate category contribution failed')
            close(point[0], point[1] - point[2], 'Aggregate G=S-O failed')
    return {'forecast_windows': len(counts), 'unique_incidents': len(set(record['incident_ids'])),
            'evaluable_windows': int((counts > 0).sum()), 'valid_cells': int(counts.sum()),
            'zero_support_windows': int((counts == 0).sum()), 'estimands': results}, category_rows


def matched_arrays(records):
    ids = list(map(int, records['incident']['ids']))
    aligned = {}
    for cohort in COHORTS:
        record = records[cohort]
        positions = {int(sample): i for i, sample in enumerate(record['ids'])}
        require(len(positions) == len(ids) and set(positions) == set(ids), 'Matched window IDs differ')
        order = np.asarray([positions[sample] for sample in ids], dtype=np.int64)
        aligned[cohort] = {key: record[key][order] for key in (*META, 'valid_counts', 'gain_sums')}
        for key in ('incident_ids', 'positive_t0'):
            require(np.array_equal(aligned[cohort][key], records['incident'][key]), 'Matched identity/clock changed')
    counts = np.stack([aligned[c]['valid_counts'] for c in COHORTS], 1)
    gains = np.stack([aligned[c]['gain_sums'] for c in COHORTS], 1)
    return aligned, counts, gains


def contrast_points(counts, gains, keep):
    n, g = counts[keep], gains[keep]
    pooled = regional.ratio(g.sum(0), n.sum(0))
    complete = (n > 0).all(1)
    per_group = g[complete] / n[complete]
    equal = regional.ratio(per_group.sum(0), complete.sum())
    result = {}
    for name, coefficients in CONTRASTS.items():
        active = np.flatnonzero(coefficients)
        result[name] = {label: regional.finite_number(np.dot(values[active], np.asarray(coefficients)[active]))
                       for label, values in (('pooled_valid_cells', pooled), ('equal_triad', equal))}
    return result


def analyze_matched(records, weeks, weights, protocol):
    aligned, counts, gains = matched_arrays(records)
    n = len(counts)
    complete = (counts > 0).all(1)
    index = week_indices(aligned['incident'], weeks)
    points = contrast_points(counts, gains, np.ones(n, dtype=bool))
    wc, wg, we = [np.zeros((len(weeks), 3)) for _ in range(3)]
    wn = np.zeros(len(weeks))
    np.add.at(wc, index, counts); np.add.at(wg, index, gains)
    np.add.at(we, index[complete], gains[complete] / counts[complete])
    np.add.at(wn, index[complete], 1)
    spec = protocol['bootstrap']
    intervals = {name: {} for name in CONTRASTS}
    for method, w in weights.items():
        sampled = {'pooled_valid_cells': regional.ratio(w @ wg, w @ wc),
                   'equal_triad': regional.ratio(w @ we, (w @ wn)[:, None])}
        for name, coefficients in CONTRASTS.items():
            active = np.flatnonzero(coefficients)
            intervals[name][method] = {estimand: regional.interval(values[:, active] @ np.asarray(coefficients)[active],
                spec['confidence'], spec['minimum_valid_draw_fraction']) for estimand, values in sampled.items()}
    coverage = {cohort: {'all_valid_cells': int(counts[:, j].sum()),
                'retained_valid_cells': int(counts[complete, j].sum()),
                'retained_cell_fraction': regional.finite_number(regional.ratio(counts[complete, j].sum(), counts[:, j].sum()))}
                for j, cohort in enumerate(COHORTS)}
    rows = []
    for week in weeks:
        year, number = week.split('-W')
        first = datetime.fromisocalendar(int(year), int(number), 1)
        last = first + timedelta(weeks=1)
        drop = np.zeros(n, dtype=bool)
        for record in aligned.values():
            drop |= np.asarray([datetime.fromisoformat(str(start)) < last and datetime.fromisoformat(str(end)) > first
                                for start, end in zip(record['support_start'], record['support_end_exclusive'])])
        remaining = contrast_points(counts, gains, ~drop)
        for name in CONTRASTS:
            for estimand, point in remaining[name].items():
                rows.append({'deleted_week': week, 'contrast': name, 'estimand': estimand, 'gain_difference': point,
                    'status': 'DEFINED' if point is not None else 'UNDEFINED_SUPPORT',
                    'removed_triplets': int(drop.sum()), 'remaining_triplets': int((~drop).sum()),
                    'remaining_complete_triplets': int((~drop & complete).sum()),
                    **{c + '_removed_cells': int(counts[drop, j].sum()) for j, c in enumerate(COHORTS)},
                    **{c + '_remaining_cells': int(counts[~drop, j].sum()) for j, c in enumerate(COHORTS)}})
    return {'triplets': n, 'complete_triplets': int(complete.sum()), 'excluded_triplets': int((~complete).sum()),
            'coverage': coverage, 'points': points, 'intervals': intervals}, rows
