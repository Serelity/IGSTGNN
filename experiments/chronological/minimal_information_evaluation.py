"""v13b paired raw-unit errors with shared calendar-week resampling."""

from datetime import datetime, timedelta

import numpy as np

from experiments.chronological.minimal_information_data import require

REGIONS = ('all', 'candidate_early', 'candidate_late', 'noncandidate')


def region_statistics(prediction, batch):
    target, valid = batch['y'], batch['valid']
    require(prediction.shape == target.shape == valid.shape and valid.dtype == np.bool_, 'Invalid target/mask shape')
    require(np.isfinite(prediction).all() and np.isfinite(target[valid]).all(), 'Nonfinite prediction/target')
    candidate = np.broadcast_to(batch['candidate'][:, None], valid.shape)
    early = np.arange(12)[None, :, None] < 6
    masks = (valid, valid & candidate & early, valid & candidate & ~early, valid & ~candidate)
    error = np.abs(prediction.astype(np.float64) - target.astype(np.float64))
    return (np.stack([np.where(mask, error, 0).sum((1, 2)) for mask in masks], -1),
            np.stack([mask.sum((1, 2)) for mask in masks], -1))


def metrics(record):
    result = {}
    for column, region in enumerate(REGIONS):
        counts, errors = record['counts'][:, column], record['errors'][:, column]
        valid = counts > 0
        result[region] = {'valid_cells': int(counts.sum()), 'valid_windows': int(valid.sum()),
            'mae_pooled': float(errors.sum() / counts.sum()) if counts.sum() else None,
            'mae_equal_window': float((errors[valid] / counts[valid]).mean()) if valid.any() else None}
    return result


def bootstrap_weights(weeks, draws, seed, block):
    require(1 <= block <= weeks and draws > 0, 'Invalid bootstrap dimensions')
    starts = np.random.default_rng(seed).integers(weeks, size=(draws, (weeks + block - 1) // block))
    selected = ((starts[..., None] + np.arange(block)) % weeks).reshape(draws, -1)[:, :weeks]
    weights = np.zeros((draws, weeks), dtype=np.int64)
    np.add.at(weights, (np.arange(draws)[:, None], selected), 1)
    return weights


def interval(numerator, denominator, spec):
    valid = denominator > 0
    if valid.mean() < spec['minimum_valid_fraction']:
        return {'status': 'INSUFFICIENT_VALID_DRAWS', 'low': None, 'high': None, 'valid_draws': int(valid.sum())}
    draws = numerator[valid] / denominator[valid]
    alpha = (1 - spec['confidence']) / 2
    low, high = np.quantile(draws, [alpha, 1 - alpha])
    return {'status': 'OK', 'low': float(low), 'high': float(high), 'valid_draws': len(draws)}


def compare_all(records, times, protocol):
    """records[seed][arm][cohort], paired by source and positive event identity."""
    spec, rule = protocol['bootstrap'], protocol['evaluation']
    first, end = map(datetime.fromisoformat, protocol['periods']['audit'])
    require(first.weekday() == 0 and (end - first).days % 7 == 0, 'Expected complete Monday week grid')
    weeks = (end - first).days // 7
    draws = {'week': bootstrap_weights(weeks, spec['draws'], spec['seed'], 1),
             'four_week_block': bootstrap_weights(weeks, spec['draws'], spec['seed'] + 1, spec['block_weeks'])}
    calendar = [(first + timedelta(weeks=i)).date().isoformat() for i in range(weeks)]
    rows, weekly_rows, decisions = [], [], {}
    for reference, added in rule['comparisons']:
        name = added + '-' + reference
        seed_decisions = []
        for seed in protocol['seeds']:
            per_case = {}
            for cohort, before in records[seed][reference].items():
                after = records[seed][added][cohort]
                for key in ('ids', 'source_indices', 'counts'):
                    require(np.array_equal(before[key], after[key]), f'Paired {key} differs')
                positions = np.array([(datetime.fromisoformat(times[int(i)]) - first).days // 7 for i in before['ids']])
                require(((positions >= 0) & (positions < weeks)).all(), 'Origin week outside audit')
                for column, region in enumerate(REGIONS):
                    count = before['counts'][:, column]
                    error_before, error_after = before['errors'][:, column], after['errors'][:, column]
                    valid = count > 0
                    weekly = np.zeros((weeks, 5), dtype=np.float64)
                    np.add.at(weekly[:, 0], positions, error_before)
                    np.add.at(weekly[:, 1], positions, error_after)
                    np.add.at(weekly[:, 2], positions, count)
                    np.add.at(weekly[:, 3], positions[valid], (error_before[valid] - error_after[valid]) / count[valid])
                    np.add.at(weekly[:, 4], positions[valid], 1)
                    for i, label in enumerate(calendar):
                        weekly_rows.append(dict(seed=seed, comparison=name, cohort=cohort, region=region,
                            origin_week=label, reference_error=weekly[i, 0], added_error=weekly[i, 1],
                            valid_cells=int(weekly[i, 2]), window_gain_sum=weekly[i, 3], valid_windows=int(weekly[i, 4])))
                    enough = (valid.sum() >= rule['minimum_windows'] and
                              np.count_nonzero(weekly[:, 2]) >= rule['minimum_nonempty_weeks'] and error_before.sum() > 0)
                    total = int(count.sum())
                    gain = float((error_before - error_after).sum() / total) if total else None
                    relative = float((error_before - error_after).sum() / error_before.sum()) if error_before.sum() > 0 else None
                    result = dict(seed=seed, comparison=name, cohort=cohort, region=region, valid_cells=total,
                        valid_windows=int(valid.sum()), nonempty_origin_weeks=int(np.count_nonzero(weekly[:, 2])),
                        support_sufficient=bool(enough), reference_mae=float(error_before.sum() / total) if total else None,
                        added_mae=float(error_after.sum() / total) if total else None, gain=gain, relative_gain=relative,
                        equal_window_gain=float(weekly[:, 3].sum() / valid.sum()) if valid.any() else None)
                    for kind, weights in draws.items():
                        ci = interval(weights @ (weekly[:, 0] - weekly[:, 1]), weights @ weekly[:, 2], spec)
                        eq = interval(weights @ weekly[:, 3], weights @ weekly[:, 4], spec)
                        result.update({f'{kind}_{key}': value for key, value in ci.items()})
                        result.update({f'{kind}_equal_window_{key}': value for key, value in eq.items()})
                    per_case[(cohort, region)] = result
                    rows.append(result)
            primary = per_case[('incident_full', 'candidate_early')]
            protected = [per_case[('incident_full', region)] for region in ('all', 'noncandidate')]
            protected += [per_case[(cohort, region)] for cohort in ('primary_control', 'secondary_control') for region in REGIONS]
            matched = per_case[('incident', 'candidate_early')]
            sufficient = all(item['support_sufficient'] for item in [primary, matched, *protected])
            supported = (sufficient and primary['relative_gain'] >= rule['practical_relative_gain']
                and all(primary[kind + '_status'] == 'OK' and primary[kind + '_low'] > 0 for kind in draws)
                and matched['gain'] > 0
                and all(item['relative_gain'] >= -rule['maximum_relative_harm'] for item in protected))
            seed_decisions.append({'seed': seed, 'status': ('SUPPORTED_DEVELOPMENT_INCREMENT' if supported else
                'NOT_SUPPORTED' if sufficient else 'INSUFFICIENT_SUPPORT')})
        statuses = [item['status'] for item in seed_decisions]
        decisions[name] = {'by_seed': seed_decisions, 'status': ('SUPPORTED_DEVELOPMENT_INCREMENT'
            if all(s == 'SUPPORTED_DEVELOPMENT_INCREMENT' for s in statuses) else
            'INSUFFICIENT_SUPPORT' if 'INSUFFICIENT_SUPPORT' in statuses else 'NOT_SUPPORTED')}
    return {'comparisons': decisions, 'rows': rows, 'weekly_rows': weekly_rows, 'calendar': calendar,
            'interpretation': 'Conditional paired development intervals; reused cohort, pointwise not simultaneous, seeds not independent replicates.'}
