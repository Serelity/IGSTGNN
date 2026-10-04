"""v12o fit-only temporal excess reweighting and shared paired inference statistics."""

from contextlib import contextmanager
from datetime import datetime

import numpy as np
import torch

from experiments.chronological import train_vector_output_scope as original
from experiments.chronological import vector_correction_geometry as geometry

base, alignment, scope = original.base, original.alignment, original.scope
regional, require = geometry.regional, geometry.require
ARMS = scope.ARMS
OBJECTIVES = ('erm', 'temporal_excess')
REGIONS = scope.REGIONS


def fit_groups(positive, indices, protocol):
    blocks = [tuple(map(datetime.fromisoformat, pair)) for pair in protocol['fit_groups']['blocks']]
    require(len(blocks) == 4 and all(a < b for a, b in blocks)
            and all(a[1] == b[0] for a, b in zip(blocks, blocks[1:])), 'Invalid fixed temporal blocks')
    require([blocks[0][0], blocks[-1][1]] == list(map(datetime.fromisoformat, protocol['periods']['fit'])),
            'Blocks must cover exactly the original fit period')
    result = {}
    for index in indices:
        stamp = datetime.fromisoformat(positive[index]['t0'])
        matches = [g for g, (start, end) in enumerate(blocks) if start <= stamp < end]
        require(len(matches) == 1 and index not in result, 'Fit window outside blocks or duplicated')
        result[index] = matches[0]
    require(set(result.values()) == set(range(4)), 'All four fit groups must be represented')
    return result


def group_statistics(record, groups):
    scope._validate_record(record)
    require(set(map(int, record['source_indices'])) == set(groups), 'Group statistics require full fit only')
    column = list(record['regions']).index('candidate_h1_h6')
    index = np.asarray([groups[int(i)] for i in record['source_indices']])
    counts = np.bincount(index, weights=record['counts'][:, column], minlength=4)
    errors = np.bincount(index, weights=record['errors'][:, column], minlength=4)
    require(np.all(counts > 0), 'Every fit block needs valid early cells')
    return {'counts': counts.astype(np.int64).tolist(), 'errors': errors.tolist(),
            'mae': (errors / counts).tolist()}


def distribution(values):
    result = np.asarray(values, dtype=np.float64)
    require(result.shape == (4,) and np.isfinite(result).all() and np.all(result > 0)
            and np.isclose(result.sum(), 1., rtol=0, atol=1e-12), 'Invalid positive four-group distribution')
    return result


def reference_weights(stats, spec):
    counts, errors = np.asarray(stats['counts']), np.asarray(stats['errors'])
    require(counts.shape == errors.shape == (4,) and np.all(counts > 0)
            and np.isfinite(errors).all() and np.all(errors >= 0), 'Invalid baseline group support/errors')
    scale = float(errors.sum() / counts.sum())
    require(scale > spec['baseline_scale_min_raw_mae'], 'Fit baseline scale is too small')
    pi = distribution(counts / counts.sum())
    return pi, scale


def update_weights(q, pi, current, baseline, spec):
    q, pi = distribution(q), distribution(pi)
    expected_pi, scale = reference_weights(baseline, spec)
    require(np.array_equal(pi, expected_pi) and current['counts'] == baseline['counts'],
            'Fit support or reference mass changed')
    errors = np.asarray(current['errors'], dtype=np.float64)
    require(errors.shape == (4,) and np.isfinite(errors).all() and np.all(errors >= 0),
            'Invalid fit group errors')
    mae = errors / np.asarray(current['counts'])
    require(np.array_equal(mae, np.asarray(current['mae'])), 'Fit MAE does not reconcile')
    excess = (mae - np.asarray(baseline['mae'])) / scale
    logits = np.log(q) + spec['update_rate'] * excess
    require(np.isfinite(logits).all(), 'Nonfinite temporal logits')
    v = np.exp(logits - logits.max())
    v /= v.sum()
    mixture = spec['coverage_mixture']
    require(0 < mixture < 1 and spec['update_rate'] > 0, 'Invalid frozen update settings')
    return distribution(mixture * pi + (1 - mixture) * v)


@contextmanager
def checked_device(model, device, progress):
    """Observe actual forwards, including unmodified legacy ERM optimizer forwards."""
    device = torch.device(device)
    seen = {'forwards': 0, 'gradient_forwards': 0}
    def tensors(value):
        if isinstance(value, torch.Tensor):
            yield value
        elif isinstance(value, dict):
            for item in value.values():
                yield from tensors(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                yield from tensors(item)
    def hook(module, args, kwargs, output):
        require(all(p.device == device for p in module.parameters()), 'Model parameter device mismatch')
        require(all(t.device == device for t in tensors((args, kwargs, output))), 'Input/prediction device mismatch')
        seen['forwards'] += 1
        seen['gradient_forwards'] += int(torch.is_grad_enabled())
        if seen['forwards'] == 1 or (torch.is_grad_enabled() and seen['gradient_forwards'] == 1):
            progress('actual_device_verified', parameter_device=str(next(module.parameters()).device),
                     input_device=str(args[0].device), prediction_device=str(output.device),
                     autograd_enabled=torch.is_grad_enabled())
    handle = model.register_forward_hook(hook, with_kwargs=True)
    try:
        yield seen
    finally:
        handle.remove()


def weighted_loss(prediction, target, valid, candidate, groups, q, pi):
    q, pi = distribution(q), distribution(pi)
    require(groups.shape == (len(prediction),) and groups.dtype == torch.int64
            and bool(((groups >= 0) & (groups < 4)).all()), 'Invalid batch group membership')
    # This branch preserves exactly the original floating-point operations at q=pi.
    if np.array_equal(q, pi):
        return alignment.early_loss(prediction, target, valid, candidate)
    mask = base.masks_for(prediction, candidate)[base.REGIONS.index('candidate_h1_h6')] & valid
    require(bool(mask.any()) and bool(torch.isfinite(prediction).all())
            and bool(torch.isfinite(target[valid]).all()), 'Invalid weighted early batch')
    multiplier = torch.as_tensor(q / pi, dtype=prediction.dtype, device=prediction.device)[groups]
    # Mask BEFORE arithmetic: missing targets must not produce NaN gradients.
    error = (prediction[mask] - target[mask]).abs()
    weights = multiplier[:, None, None, None].expand_as(prediction)[mask]
    return (error * weights).sum() / mask.sum(), int(mask.sum())


def train_epoch(model, adapter, optimizer, dataset, indices, settings, device, seed, epoch,
                progress, objective, groups, q, pi):
    require(objective in OBJECTIVES and set(indices) == set(groups), 'Unknown objective or fit indices')
    if objective == 'erm' or np.array_equal(distribution(q), distribution(pi)):
        return alignment.early_train_epoch(model, adapter, optimizer, dataset, indices,
                                           settings, device, seed, epoch, progress)
    model.eval()
    ordered = np.random.default_rng(seed * 1000 + epoch).permutation(indices).tolist()
    before = alignment.tensor_hash(adapter.state_dict())
    total, cells, penalties, max_norm, steps = 0., 0, 0., 0., 0
    for raw in base.loader(dataset, ordered, settings['batch_size']):
        batch = base.device_batch(raw, device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(batch['x'], incident_data=batch['incident'])
        group = torch.tensor([groups[int(i)] for i in raw['source_index']], device=device)
        target = (batch['y_flow'] - dataset.scaler['mean']) / dataset.scaler['std']
        loss, count = weighted_loss(prediction, target, batch['y_valid'], batch['candidate_mask'], group, q, pi)
        penalty = model.icsf_module.last_unit_residual[batch['candidate_mask']].square().mean()
        (loss + settings['gate_identity_penalty'] * penalty).backward()
        require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in adapter.parameters()),
                'Absent/nonfinite weighted gradient')
        norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), settings['clip_grad_norm'])
        require(bool(torch.isfinite(norm)), 'Nonfinite weighted gradient norm')
        optimizer.step()
        require(all(bool(torch.isfinite(p).all()) for p in adapter.parameters()), 'Nonfinite adapter')
        total += float(loss.detach()) * count
        cells += count
        penalties += float(penalty.detach())
        max_norm = max(max_norm, float(norm))
        steps += 1
        model.icsf_module.clear_observations()
        if (steps - 1) % settings['progress_every_batches'] == 0:
            progress('training_progress', epoch=epoch, batches_completed=steps,
                     samples_total=len(indices), loss_standardized=float(loss.detach()))
    require(cells > 0, 'Empty weighted epoch')
    return {'mae_standardized': total / cells, 'maximum_gradient_norm': max_norm,
            'mean_minibatch_identity_penalty': penalties / steps, 'optimizer_steps': steps,
            'gate_parameters_changed': before != alignment.tensor_hash(adapter.state_dict())}


def endpoint(arm, objective):
    require(arm in ARMS and objective in OBJECTIVES, 'Unknown endpoint')
    return f'{arm}__{objective}'


def compare_phase(reference, endpoints, times, phase, protocol):
    """Positive = lower MAE; every contrast uses the same complete calendar draws."""
    names = ['A', *[endpoint(a, o) for a in ARMS for o in OBJECTIVES]]
    require(set(endpoints) == set(names[1:]), 'Incomplete paired endpoints')
    contrasts = {name + '_vs_A': ('A', name) for name in names[1:]}
    contrasts.update({arm + '__temporal_vs_erm': tuple(endpoint(arm, o) for o in OBJECTIVES) for arm in ARMS})
    weeks, weights = geometry.calendar(protocol['periods'][phase], protocol['bootstrap'])
    arrays = {'weeks': np.asarray(weeks), **{k + '_weights': v for k, v in weights.items()},
              'comparisons': np.asarray(list(contrasts)), 'regions': np.asarray(REGIONS)}
    spec, results, rows = protocol['bootstrap'], {}, []
    for cohort, anchor in reference.items():
        records = [anchor] + [endpoints[name][cohort] for name in names[1:]]
        columns = scope._validate_record(anchor)
        for record in records[1:]:
            scope._validate_record(record)
            for key in ('ids', 'source_indices', 'counts', 'prediction_counts', 'candidate_mask', 'regions'):
                require(np.array_equal(record[key], anchor[key]), 'Paired support/identity mismatch')
        counts = anchor['counts'][:, columns]
        errors = np.stack([r['errors'][:, columns] for r in records], axis=1)
        gains = np.stack([errors[:, names.index(a)] - errors[:, names.index(b)] for a, b in contrasts.values()], 1)
        index = geometry.week_indices({'positive_t0': [times[int(i)] for i in anchor['ids']]}, weeks)
        wg = np.zeros((len(weeks), len(contrasts), len(REGIONS)))
        we, wc = np.zeros_like(wg), np.zeros((len(weeks), len(REGIONS)))
        wn = np.zeros_like(wc)
        np.add.at(wg, index, gains)
        np.add.at(we, index, np.nan_to_num(regional.ratio(gains, counts[:, None, :]), nan=0.))
        np.add.at(wc, index, counts)
        np.add.at(wn, index, counts > 0)
        for key, value in (('gain_sums', wg), ('window_gain_sums', we), ('valid_counts', wc), ('windows', wn)):
            arrays[cohort + '_' + key] = value
        sampled = {}
        for method, w in weights.items():
            shape = (len(w), len(contrasts), len(REGIONS))
            sampled[method] = {
                'pooled': regional.ratio((w @ wg.reshape(len(weeks), -1)).reshape(shape), (w @ wc)[:, None, :]),
                'equal_forecast_window': regional.ratio((w @ we.reshape(len(weeks), -1)).reshape(shape), (w @ wn)[:, None, :])}
        results[cohort] = {}
        for r, region in enumerate(REGIONS):
            items = {}
            for c, name in enumerate(contrasts):
                effect = {'gain': regional.finite_number(regional.ratio(gains[:, c, r].sum(), counts[:, r].sum())),
                    'equal_forecast_window_gain': regional.finite_number(regional.ratio(we[:, c, r].sum(), wn[:, r].sum())),
                    'intervals': {method: {estimand: regional.interval(v[:, c, r], spec['confidence'], spec['minimum_valid_draw_fraction'])
                        for estimand, v in draws.items()} for method, draws in sampled.items()}}
                items[name] = effect
                rows.append({'phase': phase, 'cohort': cohort, 'region': region, 'comparison': name,
                    'gain': effect['gain'], 'equal_forecast_window_gain': effect['equal_forecast_window_gain'],
                    **{f'{method}_{estimand}_{bound}': ci[bound] for method, values in effect['intervals'].items()
                       for estimand, ci in values.items() for bound in ('ci_low', 'ci_high')}})
            results[cohort][region] = {'valid_cells': int(counts[:, r].sum()), 'comparisons': items}
    return {'weeks': weeks, 'contrasts': contrasts, 'results': results}, rows, arrays
