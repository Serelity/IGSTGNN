"""Replay saved v12f epoch selection without importing training or checkpoint code."""

import copy

import numpy as np


COHORTS = ('incident_full', 'incident', 'primary_control', 'secondary_control')
REGIONS = ('all', 'candidate_h1_h6', 'candidate_h7_h12', 'noncandidate_all')
PARTITION = REGIONS[1:]


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(value)


def _finite(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f'{name} must be finite numeric data')
    value = float(value)
    if not np.isfinite(value) or value < minimum:
        raise ValueError(f'{name} must be finite and >= {minimum}')
    return value


def _metric(value, name):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f'{name} must be a numeric MAE or null')
    value = float(value)
    if not np.isfinite(value):
        return None
    if value < 0:
        raise ValueError(f'{name} MAE cannot be negative')
    return value


def _metrics(metrics, name):
    if not isinstance(metrics, dict) or set(metrics) != set(COHORTS):
        raise ValueError(f'{name} must contain the four protected cohorts')
    result = {}
    for cohort in COHORTS:
        values = metrics[cohort]['mae']
        if not set(REGIONS).issubset(values):
            raise ValueError(f'{name}/{cohort} must contain the four protected regions')
        result[cohort] = {region: _metric(values[region], f'{name}/{cohort}/{region}')
                          for region in REGIONS}
    return result


def _partition(mae, fractions, name):
    if mae['all'] is None or any(mae[r] is None and fractions[r] > 0 for r in PARTITION):
        return {'defined': False, 'weighted_mae': None, 'reported_all_mae': mae['all'],
                'difference': None}
    weighted = sum(fractions[r] * mae[r] for r in PARTITION if fractions[r])
    difference = weighted - mae['all']
    if not np.isclose(weighted, mae['all'], rtol=1e-10, atol=1e-9):
        raise ValueError(f'{name} regional MAEs do not reconcile with all-cell MAE')
    return {'defined': True, 'weighted_mae': float(weighted), 'reported_all_mae': mae['all'],
            'difference': float(difference)}


def _gains(current, baseline, fractions):
    return {region: {
        'mae_A': baseline[region], 'mae': current[region],
        'gain_vs_A_raw': (baseline[region] - current[region]
                          if baseline[region] is not None and current[region] is not None else None),
        'valid_cell_fraction': fractions[region],
        'weighted_global_contribution_raw': (fractions[region] * (baseline[region] - current[region])
                                             if baseline[region] is not None and current[region] is not None
                                             else (0.0 if fractions[region] == 0 else None)),
    } for region in REGIONS}


def analyze_history(history, detail, baseline, protocol, expected_steps):
    """Validate a complete history and replay the original epoch-0-backed selector.

    ``expected_steps`` is the number of optimizer updates in each epoch. Baseline
    valid-cell fractions must come from the saved, aligned selection support.
    Representation observations do not participate in the historical selector.
    """
    settings, selection = protocol['training'], protocol['selection']
    epochs = _integer(settings['epochs'], 'training epochs', minimum=1)
    expected_steps = _integer(expected_steps, 'expected steps per epoch', minimum=1)
    if (set(selection['protected_cohorts']) != set(COHORTS)
            or set(selection['protected_regions']) != set(REGIONS)
            or len(selection['protected_cohorts']) != len(COHORTS)
            or len(selection['protected_regions']) != len(REGIONS)
            or selection['maximum_relative_harm'] != .001):
        raise ValueError('Original 16 protection constraints and .001 harm bound are required')
    clip_limit = _finite(settings['clip_grad_norm'], 'gradient clipping limit')
    if clip_limit <= 0:
        raise ValueError('Gradient clipping limit must be positive')
    if not isinstance(history, list) or len(history) != epochs:
        raise ValueError('History must contain every configured epoch')
    a = _metrics(baseline, 'baseline')
    fractions, baseline_reconciliation = {}, {}
    for cohort in COHORTS:
        fraction = baseline[cohort]['valid_cell_fraction']
        if set(fraction) != set(REGIONS):
            raise ValueError(f'Baseline valid-cell fractions incomplete: {cohort}')
        fractions[cohort] = {r: _finite(fraction[r], f'{cohort}/{r} fraction') for r in REGIONS}
        if (not np.isclose(fractions[cohort]['all'], 1., rtol=0, atol=1e-12)
                or not np.isclose(sum(fractions[cohort][r] for r in PARTITION), 1., rtol=0, atol=1e-12)):
            raise ValueError(f'Baseline region fractions must partition all cells: {cohort}')
        baseline_reconciliation[cohort] = _partition(a[cohort], fractions[cohort], f'baseline/{cohort}')
    if a['incident_full']['all'] is None:
        raise ValueError('Baseline incident_full/all must be defined for epoch selection')
    if detail.get('initial_prediction_exactly_A') is not True or detail.get('backbone_state_unchanged') is not True:
        raise ValueError('Initial A identity and frozen backbone flags must be true')

    rows, epoch_results, rejected, selected_updates = [], [], [], []
    best_epoch, best_metrics = 0, a
    best_saved_metrics = baseline
    candidate_worsened = {c: [] for c in COHORTS}
    global_improved_early_worse = {c: [] for c in COHORTS}
    gradient_norms, changed_epochs, steps, penalties = [], [], 0, []
    for expected_epoch, entry in enumerate(history, 1):
        epoch = _integer(entry['epoch'], 'history epoch', minimum=1)
        if epoch != expected_epoch:
            raise ValueError('History epochs must be contiguous and ordered from 1')
        train = entry['training']
        loss = _finite(train['mae_standardized'], f'epoch {epoch} training MAE')
        gradient = _finite(train['maximum_gradient_norm'], f'epoch {epoch} gradient norm')
        epoch_steps = _integer(train['optimizer_steps'], f'epoch {epoch} optimizer steps', minimum=1)
        if epoch_steps != expected_steps:
            raise ValueError(f'Epoch {epoch} optimizer steps do not match the fixed budget')
        changed = train['gate_parameters_changed']
        if not isinstance(changed, bool):
            raise ValueError('Parameter-change flags must be boolean')
        if epoch == 1 and (not changed or gradient == 0):
            raise ValueError('First epoch must contain an effective learning update')
        if 'mean_minibatch_identity_penalty' in train:
            penalties.append(_finite(train['mean_minibatch_identity_penalty'], f'epoch {epoch} identity penalty'))
        steps += epoch_steps
        gradient_norms.append(gradient)
        if changed:
            changed_epochs.append(epoch)

        current = _metrics(entry['selection'], f'epoch {epoch} selection')
        checks, failures, reconciliation = {}, [], {}
        for cohort in COHORTS:
            reconciliation[cohort] = _partition(current[cohort], fractions[cohort], f'epoch {epoch}/{cohort}')
            gains = _gains(current[cohort], a[cohort], fractions[cohort])
            for region in REGIONS:
                aa, bb = a[cohort][region], current[cohort][region]
                key = f'{cohort}/{region}'
                allowed = aa * (1 + .001) + 1e-12 if aa is not None else None
                passed = bool(aa is not None and bb is not None and bb <= allowed)
                checks[key] = passed
                if not passed:
                    failures.append({'cohort': cohort, 'region': region, 'mae_A': aa, 'mae': bb,
                                     'maximum_allowed_mae': allowed,
                                     'reason': 'undefined_metric' if aa is None or bb is None else 'harm_bound_exceeded'})
                rows.append({'epoch': epoch, 'cohort': cohort, 'region': region,
                             **gains[region], 'protection_passed': passed})
            if gains['candidate_h1_h6']['gain_vs_A_raw'] is not None and gains['candidate_h1_h6']['gain_vs_A_raw'] < 0:
                candidate_worsened[cohort].append(epoch)
                if gains['all']['gain_vs_A_raw'] is not None and gains['all']['gain_vs_A_raw'] > 0:
                    global_improved_early_worse[cohort].append(epoch)
        eligible = all(checks.values())
        if type(entry.get('eligible')) is not bool or entry['eligible'] != eligible:
            raise ValueError(f'Epoch {epoch} saved eligibility disagrees with original selector')
        if (set(entry['protection_checks']) != set(checks)
                or any(type(entry['protection_checks'][k]) is not bool or entry['protection_checks'][k] != v
                       for k, v in checks.items())):
            raise ValueError(f'Epoch {epoch} saved protection checks disagree with original selector')
        global_mae = current['incident_full']['all']
        improved_best = global_mae is not None and global_mae < best_metrics['incident_full']['all']
        updated = eligible and improved_best
        if updated:
            best_epoch, best_metrics, best_saved_metrics = epoch, current, entry['selection']
            selected_updates.append(epoch)
        if _integer(entry['best_epoch'], f'epoch {epoch} saved best epoch') != best_epoch:
            raise ValueError(f'Epoch {epoch} saved best epoch disagrees with strict selection replay')
        if not eligible:
            rejected.append({'epoch': epoch, 'failed_constraints': [f'{f["cohort"]}/{f["region"]}' for f in failures],
                             'failures': failures})
        for row in rows[-len(COHORTS) * len(REGIONS):]:
            row.update(eligible=eligible, selected_as_new_best=updated, best_epoch=best_epoch)
        epoch_results.append({'epoch': epoch, 'eligible': eligible, 'selected_as_new_best': updated,
                              'best_epoch': best_epoch, 'training_mae_standardized': loss,
                              'partition_reconciliation': reconciliation})

    if _integer(detail['optimizer_steps'], 'saved total optimizer steps') != steps:
        raise ValueError('Saved total optimizer steps disagree with complete history')
    if _integer(detail['selected_epoch'], 'saved selected epoch') != best_epoch:
        raise ValueError('Saved selected epoch disagrees with selection replay')
    saved = _metrics(detail['selection_metrics'], 'saved selected metrics')
    if saved != best_metrics:
        raise ValueError('Saved selected metrics disagree with selection replay')
    for cohort in COHORTS:
        if ('samples' in best_saved_metrics[cohort] and 'samples' in detail['selection_metrics'][cohort]
                and best_saved_metrics[cohort]['samples'] != detail['selection_metrics'][cohort]['samples']):
            raise ValueError('Saved selected metric sample count disagrees with history')
    eligible_count = epochs - len(rejected)
    reason = None
    if best_epoch == 0:
        reason = ('NO_ELIGIBLE_EPOCH' if eligible_count == 0
                  else 'NO_STRICT_FULL_POSITIVE_GLOBAL_IMPROVEMENT_AMONG_ELIGIBLE_EPOCHS')
    last = _metrics(history[-1]['selection'], 'last selection')
    result = {
        'epochs': epochs, 'eligible_epochs': eligible_count, 'rejected_epochs_count': len(rejected),
        'selected_epoch': best_epoch, 'last_epoch': epochs, 'selection_update_epochs': selected_updates,
        'optimizer_steps': steps, 'optimizer_steps_per_epoch': expected_steps,
        'fallback_reason': reason, 'rejected_epochs': rejected, 'epoch_summary': epoch_results,
        'baseline_partition_reconciliation': baseline_reconciliation,
        'cohorts': {c: {'selected_selection': _gains(best_metrics[c], a[c], fractions[c]),
                        'last_selection': _gains(last[c], a[c], fractions[c]),
                        'candidate_h1_h6_worsened_epochs': candidate_worsened[c],
                        'global_improved_while_candidate_h1_h6_worse_epochs': global_improved_early_worse[c]}
                    for c in COHORTS},
        'training': {'parameters_changed_epochs': changed_epochs,
                     'maximum_reported_gradient_norm': max(gradient_norms),
                     'minimum_reported_gradient_norm': min(gradient_norms),
                     'clip_grad_norm_limit': clip_limit,
                     'epochs_with_some_preclip_gradient_norm_above_limit':
                         [i + 1 for i, v in enumerate(gradient_norms) if v > clip_limit],
                     'gradient_norm_interpretation': 'Saved maxima are pre-clipping norms; they do not measure clipped norms or the number of clipped batches.',
                     'mean_minibatch_identity_penalty_by_available_epoch': copy.deepcopy(penalties)},
        'candidate_gate_used_for_selection': False, 'candidate_representation_used_for_selection': False,
        'interpretation': 'Selection-period point estimates replay the original rule. Regional weighted contributions partition each cohort global gain; they are accounting identities, not causal attributions. No training or checkpoint state is loaded.',
    }
    return result, rows
