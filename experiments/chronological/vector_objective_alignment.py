"""Frozen v12k selection policies, early objective and paired regional statistics."""

import hashlib

import numpy as np
import torch

from experiments.chronological import train_incident_state_interaction as vector
from experiments.chronological import audit_architecture_regions as regional

base = vector.base
ARMS = ('state_vector', 'interaction_vector')
LOSSES = SELECTORS = ('global', 'candidate_early')
REGIONS = ('all', 'candidate_h1_h6', 'candidate_h7_h12', 'noncandidate_all')


def tensor_hash(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def selection_decision(current, baseline, best_metrics, selector, protocol):
    if selector not in SELECTORS:
        raise ValueError('Unknown v12k selector')
    eligible, checks = base.selection_eligible(current, baseline, protocol)
    full, anchor = current['incident_full']['mae'], baseline['incident_full']['mae']
    global_improves = bool(eligible and full['all'] < anchor['all'])
    region = 'all' if selector == 'global' else 'candidate_h1_h6'
    early_improves = bool(eligible and full['candidate_h1_h6'] < anchor['candidate_h1_h6'])
    allowed = global_improves and (selector == 'global' or early_improves)
    replace = bool(allowed and full[region] < best_metrics['incident_full']['mae'][region])
    return {'protected': eligible, 'protection_checks': checks,
            'full_global_strictly_better_than_A': global_improves,
            'full_early_strictly_better_than_A': early_improves,
            'eligible': allowed, 'replace_best': replace}


def replay_selection(history, baseline, protocol):
    best = {name: {'epoch': 0, 'selection_metrics': baseline} for name in SELECTORS}
    for epoch, row in enumerate(history, 1):
        if row['epoch'] != epoch:
            raise ValueError('Recovery history is not contiguous')
        for selector in SELECTORS:
            expected = selection_decision(row['selection'], baseline,
                best[selector]['selection_metrics'], selector, protocol)
            if expected['replace_best']:
                best[selector] = {'epoch': epoch, 'selection_metrics': row['selection']}
            if row['decisions'][selector] != {**expected, 'best_epoch': best[selector]['epoch']}:
                raise ValueError('Recovery history disagrees with frozen selectors')
    return best


def early_loss(prediction, target, valid, candidate):
    mask = base.masks_for(prediction, candidate)[base.REGIONS.index('candidate_h1_h6')] & valid
    if not mask.any():
        raise ValueError('Fit batch has no valid candidate H1-H6 targets')
    if not torch.isfinite(prediction).all() or not torch.isfinite(target[valid]).all():
        raise ValueError('Nonfinite prediction or valid target')
    return base.masked_flow_mae(prediction, target, mask), int(mask.sum())


def early_train_epoch(model, adapter, optimizer, dataset, indices, settings, device, seed, epoch, progress):
    model.eval()
    ordered = np.random.default_rng(seed * 1000 + epoch).permutation(indices).tolist()
    before = base.cpu_tree(adapter.state_dict())
    total, cells, penalty_sum, maximum_gradient, steps = 0., 0, 0., 0., 0
    for step, raw in enumerate(base.loader(dataset, ordered, settings['batch_size'])):
        batch = base.device_batch(raw, device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(batch['x'], incident_data=batch['incident'])
        target = (batch['y_flow'] - dataset.scaler['mean']) / dataset.scaler['std']
        loss, count = early_loss(prediction, target, batch['y_valid'], batch['candidate_mask'])
        penalty = model.icsf_module.last_unit_residual[batch['candidate_mask']].square().mean()
        (loss + settings['gate_identity_penalty'] * penalty).backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in adapter.parameters()):
            raise ValueError('Absent/nonfinite early-objective gradient')
        norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), settings['clip_grad_norm'])
        if not torch.isfinite(norm):
            raise ValueError('Nonfinite early-objective gradient norm')
        maximum_gradient = max(maximum_gradient, float(norm))
        optimizer.step()
        if any(not torch.isfinite(p).all() for p in adapter.parameters()):
            raise ValueError('Nonfinite adapter parameter')
        total += float(loss.detach()) * count
        cells += count
        penalty_sum += float(penalty.detach())
        steps += 1
        model.icsf_module.clear_observations()
        if step % settings['progress_every_batches'] == 0:
            progress('training_progress', epoch=epoch, batches_completed=steps,
                     samples_total=len(indices), loss_standardized=float(loss.detach()))
    changed = tensor_hash(before) != tensor_hash(adapter.state_dict())
    if not cells or (epoch == 1 and (not changed or maximum_gradient == 0)):
        raise ValueError('Early-objective adapter had no effective learning update')
    return {'mae_standardized': total / cells, 'maximum_gradient_norm': maximum_gradient,
            'mean_minibatch_identity_penalty': penalty_sum / steps,
            'gate_parameters_changed': changed, 'optimizer_steps': steps}


def endpoint(arm, loss, selector):
    return f'{arm}__loss_{loss}__select_{selector}'


def comparisons():
    """Each contrast changes one factor, apart from the explicitly named A references."""
    result = {}
    for arm in ARMS:
        for loss in LOSSES:
            for selector in SELECTORS:
                name = endpoint(arm, loss, selector)
                result[name + '_vs_A'] = ('A', name)
            result[f'{arm}__loss_{loss}__selector_effect'] = tuple(endpoint(arm, loss, s) for s in SELECTORS)
        for selector in SELECTORS:
            result[f'{arm}__select_{selector}__loss_effect'] = tuple(endpoint(arm, loss, selector) for loss in LOSSES)
    for loss in LOSSES:
        for selector in SELECTORS:
            result[f'loss_{loss}__select_{selector}__context_effect'] = tuple(endpoint(arm, loss, selector) for arm in ARMS)
    return result


def compare_phase(reference, endpoints, times, protocol):
    """Shared calendar and draws; supports fit-only positives as well as four cohorts."""
    names = ['A', *[endpoint(a, l, s) for a in ARMS for l in LOSSES for s in SELECTORS]]
    if set(endpoints) != set(names[1:]):
        raise ValueError('Incomplete v12k endpoint set')
    selected_times = {int(sample): times[int(sample)] for sample in reference['incident_full']['ids']}
    weeks, positions = regional.week_grid(selected_times)
    spec, contrasts = protocol['bootstrap'], comparisons()
    weights = {method: regional.bootstrap_weights(len(weeks), spec['draws'], spec['seed'] + offset, block)
        for method, offset, block in (('week', 0, 1), ('four_week_block', 1, spec['sensitivity_circular_block_weeks']))}
    arrays = {'weeks': np.asarray(weeks), 'regions': np.asarray(REGIONS),
              'comparisons': np.asarray(list(contrasts)), **{f'{k}_weights': v for k, v in weights.items()}}
    results, rows = {}, []
    for cohort, anchor in reference.items():
        records = [anchor] + [endpoints[name][cohort] for name in names[1:]]
        for other in records[1:]:
            for key in ('ids', 'source_indices', 'counts', 'prediction_counts', 'candidate_mask', 'regions'):
                if not np.array_equal(anchor[key], other[key]):
                    raise ValueError('Endpoint sample/support mismatch')
        columns = [list(anchor['regions']).index(r) for r in REGIONS]
        counts = anchor['counts'][:, columns]
        errors = np.stack([r['errors'][:, columns] for r in records], 1)
        gains = np.stack([errors[:, names.index(a)] - errors[:, names.index(b)] for a, b in contrasts.values()], 1)
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
                    'mae': {name: regional.finite_number(regional.ratio(errors[:, p, r].sum(), counts[:, r].sum())) for p, name in enumerate(names)},
                    'comparisons': {}}
            for c, name in enumerate(contrasts):
                effect = {'gain_raw_mae': regional.finite_number(regional.ratio(wg[:, c, r].sum(), wc[:, r].sum())),
                    'equal_forecast_window_gain_raw_mae': regional.finite_number(regional.ratio(we[:, c, r].sum(), wn[:, r].sum())),
                    'regional_gain_in_global_mae_units': regional.finite_number(regional.ratio(wg[:, c, r].sum(), wc[:, 0].sum())),
                    'intervals': {method: {estimand: regional.interval(v[:, c, r], spec['confidence'], spec['minimum_valid_draw_fraction'])
                        for estimand, v in draws.items()} for method, draws in sampled.items()}}
                item['comparisons'][name] = effect
                rows.append({'cohort': cohort, 'region': region, 'comparison': name,
                    **{k: v for k, v in effect.items() if k != 'intervals'},
                    **{f'{method}_{estimand}_{bound}': ci[bound]
                       for method, values in effect['intervals'].items() for estimand, ci in values.items()
                       for bound in ('ci_low', 'ci_high')}})
            result['regions'][region] = item
        results[cohort] = result
    return {'weeks': weeks, 'paths': names, 'contrasts': contrasts, 'results': results}, rows, arrays
