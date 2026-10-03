"""v12m output projection, shared-trajectory selectors and paired regional evidence."""

import copy
import json

import numpy as np
import torch

from experiments.chronological import vector_objective_alignment as alignment

base, regional = alignment.base, alignment.regional
ARMS = ('state_vector', 'interaction_vector')
POLICIES = ('unrestricted', 'candidate_early_only')
OUTPUT_PATHS = ('unrestricted_at_unrestricted', 'protected_at_unrestricted',
                'protected_at_protected')
REGIONS = alignment.REGIONS
ADDITIVITY_RTOL = 1e-10
ADDITIVITY_ATOL = 1e-8


def project_prediction(prediction, baseline_prediction, candidate_mask):
    """Choose the vector prediction on report-location candidates at H1--H6.

    The caller supplies the original report-location node mask, never target validity
    or cohort membership. The strict [B,N] Boolean contract prevents horizon/target
    masks and per-cohort flags from silently broadcasting into this intervention.
    """
    if not isinstance(prediction, torch.Tensor) or not isinstance(baseline_prediction, torch.Tensor):
        raise ValueError('Prediction and baseline must be tensors')
    if (prediction.ndim != 4 or prediction.shape[1] != 12 or prediction.shape[3] != 1
            or prediction.shape[0] == 0 or prediction.shape[2] == 0):
        raise ValueError('Prediction shape must be [B,12,N,1] with nonempty B/N')
    if (baseline_prediction.shape != prediction.shape or baseline_prediction.dtype != prediction.dtype
            or baseline_prediction.device != prediction.device):
        raise ValueError('Baseline prediction shape, dtype and device must match')
    if not prediction.is_floating_point() or not baseline_prediction.is_floating_point():
        raise ValueError('Predictions must be floating point')
    if baseline_prediction.requires_grad or baseline_prediction.grad_fn is not None:
        raise ValueError('Baseline prediction must be computed without gradients')
    if (not isinstance(candidate_mask, torch.Tensor) or candidate_mask.dtype != torch.bool
            or candidate_mask.shape != (prediction.shape[0], prediction.shape[2])
            or candidate_mask.device != prediction.device):
        raise ValueError('Candidate mask must be the Boolean [B,N] report-location support')
    if not torch.isfinite(prediction).all() or not torch.isfinite(baseline_prediction).all():
        raise ValueError('Nonfinite prediction or baseline')
    early = torch.arange(12, device=prediction.device)[None, :, None, None] < 6
    support = candidate_mask[:, None, :, None] & early
    return torch.where(support, prediction, baseline_prediction)


def endpoint(arm, path):
    if arm not in ARMS or path not in OUTPUT_PATHS:
        raise ValueError('Unknown v12m architecture or output path')
    return f'{arm}__{path}'


def selection_decision(current, baseline, best_metrics, protocol):
    return alignment.selection_decision(current, baseline, best_metrics, 'candidate_early', protocol)


def _identical(left, right):
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(right, sort_keys=True, allow_nan=False)


def replay_selection(history, baseline, protocol):
    """Restore each output policy's own best epoch, including ties and A fallback."""
    best = {policy: {'epoch': 0, 'selection_metrics': copy.deepcopy(baseline)} for policy in POLICIES}
    for epoch, row in enumerate(history, 1):
        if type(row['epoch']) is not int or row['epoch'] != epoch:
            raise ValueError('Recovery history is not contiguous')
        if set(row['selection']) != set(POLICIES) or set(row['decisions']) != set(POLICIES):
            raise ValueError('Recovery history requires exactly both output policies')
        for policy in POLICIES:
            current = row['selection'][policy]
            decision = selection_decision(current, baseline, best[policy]['selection_metrics'], protocol)
            if decision['replace_best']:
                best[policy] = {'epoch': epoch, 'selection_metrics': copy.deepcopy(current)}
            expected = {**decision, 'best_epoch': best[policy]['epoch']}
            if not _identical(row['decisions'][policy], expected):
                raise ValueError(f'Recovery history disagrees with frozen output-policy selectors: {policy}')
    return best


def comparisons():
    """Reference minus compared MAE: positive is improvement, not causal attribution."""
    result = {}
    for arm in ARMS:
        u, pu, pp = (endpoint(arm, path) for path in OUTPUT_PATHS)
        for name in (u, pu, pp):
            result[name + '_vs_A'] = ('A', name)
        result[f'{arm}__output_scope_effect'] = (u, pu)
        result[f'{arm}__selection_effect'] = (pu, pp)
        result[f'{arm}__total_policy_effect'] = (u, pp)
    result['protected_at_protected__context_effect'] = tuple(endpoint(arm, OUTPUT_PATHS[2]) for arm in ARMS)
    return result


def _require_close(left, right, message):
    if not np.allclose(left, right, rtol=ADDITIVITY_RTOL, atol=ADDITIVITY_ATOL, equal_nan=False):
        raise ValueError(message)


def _validate_record(record):
    regions = list(record['regions'])
    if len(regions) != len(set(regions)) or not set(REGIONS).issubset(regions):
        raise ValueError('Missing or duplicate output region')
    ids, sources = np.asarray(record['ids']), np.asarray(record['source_indices'])
    if (ids.ndim != 1 or sources.shape != ids.shape or ids.dtype.kind not in 'iu'
            or sources.dtype.kind not in 'iu' or (ids < 0).any() or (sources < 0).any()
            or len(np.unique(ids)) != len(ids)):
        raise ValueError('Invalid output sample identities')
    candidate = np.asarray(record['candidate_mask'])
    if (candidate.ndim != 2 or candidate.shape[0] != len(ids) or candidate.shape[1] == 0
            or candidate.dtype != np.bool_):
        raise ValueError('Invalid output candidate mask')
    errors, counts, predicted = (np.asarray(record[k]) for k in ('errors', 'counts', 'prediction_counts'))
    expected_shape = (len(ids), len(regions))
    if (errors.shape != expected_shape or counts.shape != expected_shape or predicted.shape != expected_shape
            or errors.dtype.kind != 'f' or counts.dtype.kind not in 'iu' or predicted.dtype.kind not in 'iu'):
        raise ValueError('Invalid output errors/counts shape or dtype')
    if (not np.isfinite(errors).all() or (errors < 0).any() or (counts < 0).any()
            or (predicted < counts).any() or (errors[counts == 0] != 0).any()):
        raise ValueError('Invalid output errors/counts or empty-region errors')
    columns = [regions.index(region) for region in REGIONS]
    candidate_nodes = candidate.sum(1)
    geometry = np.stack((np.full(len(ids), 12 * candidate.shape[1]), 6 * candidate_nodes,
                         6 * candidate_nodes, 12 * (candidate.shape[1] - candidate_nodes)), 1)
    if not np.array_equal(predicted[:, columns], geometry):
        raise ValueError('Prediction support does not match report-location candidates and twelve horizons')
    for values in (counts, predicted):
        if not np.array_equal(values[:, columns[0]], values[:, columns[1:]].sum(-1)):
            raise ValueError('Output regional counts do not partition all cells')
    _require_close(errors[:, columns[0]], errors[:, columns[1:]].sum(-1),
                   'Output regional errors do not partition all cells')
    return columns


def _validate_paths(anchor, records, names):
    columns = _validate_record(anchor)
    for record in records[1:]:
        _validate_record(record)
        for key in ('ids', 'source_indices', 'counts', 'prediction_counts', 'candidate_mask', 'regions'):
            if not np.array_equal(anchor[key], record[key]):
                raise ValueError(f'Endpoint sample/support mismatch: {key}')
    a = np.asarray(anchor['errors'])[:, columns]
    for arm in ARMS:
        u, pu, pp = (np.asarray(records[names.index(endpoint(arm, path))]['errors'])[:, columns]
                     for path in OUTPUT_PATHS)
        if not np.array_equal(u[:, 1], pu[:, 1]):
            raise ValueError('Same-weight output scope changed candidate early errors')
        for protected in (pu, pp):
            if not np.array_equal(protected[:, 2:], a[:, 2:]):
                raise ValueError('Protected late/noncandidate errors differ from A')
            _require_close(a[:, 0] - protected[:, 0], a[:, 1] - protected[:, 1],
                           'Protected global gain does not equal candidate early error gain')
        _require_close(u - pp, (u - pu) + (pu - pp), 'Output and selection effects do not add to total')
    return columns


def compare_phase(reference, endpoints, times, protocol):
    """Six prespecified outputs on one calendar and shared paired bootstrap draws.

    No unrestricted prediction at the protected-selected epoch is accepted. Pure
    output replacement, changed selection, and their total each receive a direct
    paired interval; interval endpoints are never added.
    """
    names = ['A', *[endpoint(arm, path) for arm in ARMS for path in OUTPUT_PATHS]]
    if set(endpoints) != set(names[1:]):
        raise ValueError('Require exactly the six v12m output paths; no unrestricted-at-protected path')
    if 'incident_full' not in reference or len(reference['incident_full']['ids']) == 0:
        raise ValueError('A nonempty incident_full cohort is required for the shared calendar')
    if any(set(value) != set(reference) for value in endpoints.values()):
        raise ValueError('Endpoint cohort set mismatch')
    selected_times = {int(sample): times[int(sample)] for sample in reference['incident_full']['ids']}
    weeks, positions = regional.week_grid(selected_times)
    spec, contrasts = protocol['bootstrap'], comparisons()
    weights = {method: regional.bootstrap_weights(len(weeks), spec['draws'], spec['seed'] + offset, block)
        for method, offset, block in (('week', 0, 1), ('four_week_block', 1, spec['sensitivity_circular_block_weeks']))}
    arrays = {'weeks': np.asarray(weeks), 'regions': np.asarray(REGIONS),
              'comparisons': np.asarray(list(contrasts)), **{f'{key}_weights': value for key, value in weights.items()}}
    results, rows = {}, []
    for cohort, anchor in reference.items():
        records = [anchor] + [endpoints[name][cohort] for name in names[1:]]
        columns = _validate_paths(anchor, records, names)
        if not set(map(int, anchor['ids'])).issubset(positions):
            raise ValueError('Cohort sample is outside the full-positive calendar')
        counts = np.asarray(anchor['counts'])[:, columns]
        errors = np.stack([np.asarray(record['errors'])[:, columns] for record in records], 1)
        gains = np.stack([errors[:, names.index(left)] - errors[:, names.index(right)]
                          for left, right in contrasts.values()], 1)
        window_gains = regional.ratio(gains, counts[:, None, :])
        index = np.asarray([positions[int(i)] for i in anchor['ids']], dtype=np.int64)
        wg = np.zeros((len(weeks), len(contrasts), len(REGIONS)))
        we, wc = np.zeros_like(wg), np.zeros((len(weeks), len(REGIONS)))
        wn = np.zeros_like(wc)
        np.add.at(wg, index, gains)
        np.add.at(we, index, np.nan_to_num(window_gains, nan=0.))
        np.add.at(wc, index, counts)
        np.add.at(wn, index, counts > 0)
        for key, value in (('gain_sums', wg), ('window_gain_sums', we), ('valid_counts', wc), ('evaluable_windows', wn)):
            arrays[f'{cohort}_{key}'] = value
        sampled = {}
        for method, w in weights.items():
            shape = (len(w), len(contrasts), len(REGIONS))
            sampled[method] = {
                'pooled': regional.ratio((w @ wg.reshape(len(weeks), -1)).reshape(shape), (w @ wc)[:, None, :]),
                'equal_forecast_window': regional.ratio((w @ we.reshape(len(weeks), -1)).reshape(shape), (w @ wn)[:, None, :])}
        result = {'forecast_windows': len(index), 'regions': {}}
        for r, region in enumerate(REGIONS):
            item = {'valid_cells': int(counts[:, r].sum()),
                    'evaluable_windows': int((counts[:, r] > 0).sum()),
                    'mae': {name: regional.finite_number(regional.ratio(errors[:, i, r].sum(), counts[:, r].sum()))
                            for i, name in enumerate(names)}, 'comparisons': {}}
            for c, name in enumerate(contrasts):
                effect = {'gain_raw_mae': regional.finite_number(regional.ratio(wg[:, c, r].sum(), wc[:, r].sum())),
                    'equal_forecast_window_gain_raw_mae': regional.finite_number(regional.ratio(we[:, c, r].sum(), wn[:, r].sum())),
                    'regional_gain_in_global_mae_units': regional.finite_number(regional.ratio(wg[:, c, r].sum(), wc[:, 0].sum())),
                    'intervals': {method: {estimand: regional.interval(value[:, c, r], spec['confidence'], spec['minimum_valid_draw_fraction'])
                        for estimand, value in draws.items()} for method, draws in sampled.items()}}
                item['comparisons'][name] = effect
                rows.append({'cohort': cohort, 'region': region, 'comparison': name,
                    **{key: value for key, value in effect.items() if key != 'intervals'},
                    **{f'{method}_{estimand}_{bound}': ci[bound]
                       for method, values in effect['intervals'].items() for estimand, ci in values.items()
                       for bound in ('ci_low', 'ci_high')}})
            result['regions'][region] = item
        results[cohort] = result
    return {'weeks': weeks, 'paths': names, 'contrasts': contrasts, 'results': results,
            'scope_identities_verified': True, 'effect_decomposition_verified': True,
            'additivity_tolerance': {'rtol': ADDITIVITY_RTOL, 'atol': ADDITIVITY_ATOL}}, rows, arrays
