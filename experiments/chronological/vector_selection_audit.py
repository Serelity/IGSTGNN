"""v12l selection accounting; standard library only, no model or data access."""

from collections import Counter
from itertools import combinations
import json
import math

SELECTORS = ('global', 'candidate_early')
EARLY = 'candidate_h1_h6'
REASONS = ('invalid_protection_metric', 'protection_rejected', 'global_not_improved',
           'early_not_improved', 'eligible_not_better_than_incumbent', 'best_updated')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def identical(left, right):
    # Reject bool/int substitutions in recorded decision fields as well as value changes.
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(right, sort_keys=True, allow_nan=False)


def number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def gain(a, b):
    return a - b if number(a) and number(b) else None


def strict_gain(a, b):
    value = gain(a, b)
    return value is not None and value > 0


def metric_rows(current, baseline, spec):
    rows = []
    for cohort in spec['protected_cohorts']:
        for region in spec['protected_regions']:
            a, b = baseline[cohort]['mae'][region], current[cohort]['mae'][region]
            valid = number(a) and number(b)
            limit = a * (1 + spec['maximum_relative_harm']) if number(a) else None
            decision_limit = limit + 1e-12 if limit is not None else None
            harm = b - a if valid else None
            passed = bool(valid and b <= decision_limit)
            rows.append({'check': f'{cohort}/{region}', 'cohort': cohort, 'region': region,
                'baseline_mae': a, 'current_mae': b, 'gain_raw_mae': gain(a, b),
                'harm_raw_mae': harm,
                'relative_harm_percent': 100 * harm / a if valid and a > 0 else None,
                'nominal_limit': limit, 'decision_limit': decision_limit,
                'nominal_slack_raw_mae': limit - b if valid else None,
                'decision_slack_raw_mae': decision_limit - b if valid else None,
                'excess_over_decision_limit': max(0., b - decision_limit) if valid else None,
                'valid': valid, 'passed': passed})
    return rows


def audit_history(history, baseline, spec):
    """Replay both selectors; all diagnostics condition only on saved selection MAEs."""
    best_metrics = {s: baseline for s in SELECTORS}
    best_epochs = dict.fromkeys(SELECTORS, 0)
    checks, decisions, epochs = [], [], []
    for epoch, saved in enumerate(history, 1):
        require(type(saved['epoch']) is int and saved['epoch'] == epoch, 'Noncontiguous epoch history')
        current = saved['selection']
        rows = metric_rows(current, baseline, spec)
        failed = [r['check'] for r in rows if not r['passed']]
        protected = not failed
        full, anchor = current['incident_full']['mae'], baseline['incident_full']['mae']
        global_gain, early_gain = gain(anchor['all'], full['all']), gain(anchor[EARLY], full[EARLY])
        raw_global, raw_early = strict_gain(anchor['all'], full['all']), strict_gain(anchor[EARLY], full[EARLY])
        joint = raw_global and raw_early
        checks.extend({'epoch': epoch, 'joint_positive': joint, **row} for row in rows)
        epochs.append({'epoch': epoch, 'global_gain': global_gain, 'early_gain': early_gain,
            'raw_global_improves': raw_global, 'raw_early_improves': raw_early,
            'joint_positive': joint, 'protected': protected, 'failed_checks': failed,
            'joint_positive_protection_rejected': joint and not protected})
        require(set(saved['decisions']) == set(SELECTORS), 'Missing/extra stored selector')
        for selector in SELECTORS:
            region = 'all' if selector == 'global' else EARLY
            allowed = protected and raw_global and (selector == 'global' or raw_early)
            replace = bool(allowed and full[region] < best_metrics[selector]['incident_full']['mae'][region])
            previous = best_epochs[selector]
            incumbent_mae = best_metrics[selector]['incident_full']['mae'][region]
            if replace:
                best_metrics[selector], best_epochs[selector] = current, epoch
            expected = {'protected': protected, 'protection_checks': {r['check']: r['passed'] for r in rows},
                # Original v12k fields are protection-conditioned. Raw diagnostics above are not.
                'full_global_strictly_better_than_A': protected and raw_global,
                'full_early_strictly_better_than_A': protected and raw_early,
                'eligible': allowed, 'replace_best': replace, 'best_epoch': best_epochs[selector]}
            require(identical(saved['decisions'][selector], expected),
                    f'Stored decision mismatch at epoch {epoch}, selector {selector}')
            blockers = [f'protection:{key}' for key in failed]
            if not raw_global:
                blockers.append('global_not_improved')
            if selector == 'candidate_early' and not raw_early:
                blockers.append('early_not_improved')
            if any(not r['valid'] for r in rows):
                reason = REASONS[0]
            elif not protected:
                reason = REASONS[1]
            elif not raw_global:
                reason = REASONS[2]
            elif selector == 'candidate_early' and not raw_early:
                reason = REASONS[3]
            else:
                reason = REASONS[5] if replace else REASONS[4]
            decisions.append({'epoch': epoch, 'selector': selector, 'reason': reason,
                'raw_global_improves': raw_global, 'raw_early_improves': raw_early,
                'protected': protected, 'eligible': allowed, 'replace_best': replace,
                'previous_best_epoch': previous, 'best_epoch': best_epochs[selector],
                'objective_mae': full[region], 'previous_best_objective_mae': incumbent_mae,
                'failed_checks': failed, 'eligibility_blockers': blockers})
    return {'epochs': epochs, 'checks': checks, 'decisions': decisions,
            'replayed_selectors': {s: {'selected_epoch': best_epochs[s], 'selection_metrics': best_metrics[s]}
                                   for s in SELECTORS}}


def accounting(epochs, checks, decisions, spec):
    """Epoch-level protection counts; selector-level procedural counts are separate."""
    keys = [f'{c}/{r}' for c in spec['protected_cohorts'] for r in spec['protected_regions']]
    failure_counts = {}
    for key in keys:
        subset = [r for r in checks if r['check'] == key]
        failures = [r for r in subset if not r['passed']]
        joint = [r for r in subset if r['joint_positive']]
        joint_failures = [r for r in joint if not r['passed']]
        finite_excess = [r['excess_over_decision_limit'] for r in failures if r['valid']]
        failure_counts[key] = {'evaluated_epochs': len(subset), 'failed_epochs': len(failures),
            'undefined_metric_epochs': sum(not r['valid'] for r in subset),
            'joint_positive_epochs': len(joint), 'joint_positive_failed_epochs': len(joint_failures),
            'sole_failed_protection_epochs': sum(e['failed_checks'] == [key] for e in epochs),
            'joint_positive_sole_failed_protection_epochs': sum(e['joint_positive'] and e['failed_checks'] == [key] for e in epochs),
            'maximum_finite_excess_raw_mae': max(finite_excess, default=None)}
    cofailures = [{'first': a, 'second': b,
        'epochs': sum(a in e['failed_checks'] and b in e['failed_checks'] for e in epochs),
        'joint_positive_epochs': sum(e['joint_positive'] and a in e['failed_checks'] and b in e['failed_checks'] for e in epochs)}
        for a, b in combinations(keys, 2)]
    union_counts = {}
    for dimension, values in (('cohort', spec['protected_cohorts']), ('region', spec['protected_regions'])):
        for value in values:
            position = 0 if dimension == 'cohort' else 1
            affected = [e for e in epochs if any(k.split('/')[position] == value for k in e['failed_checks'])]
            union_counts[f'{dimension}:{value}'] = {'failed_epochs': len(affected),
                'joint_positive_failed_epochs': sum(e['joint_positive'] for e in affected)}
    reason_counts = {s: dict.fromkeys(REASONS, 0) for s in SELECTORS}
    for row in decisions:
        reason_counts[row['selector']][row['reason']] += 1
    patterns = Counter(tuple(e['failed_checks']) for e in epochs if e['failed_checks'])
    return {'trajectory_epochs': len(epochs), 'selector_decisions': len(decisions),
        'raw_global_improvement_epochs': sum(e['raw_global_improves'] for e in epochs),
        'raw_early_improvement_epochs': sum(e['raw_early_improves'] for e in epochs),
        'joint_positive_epochs': sum(e['joint_positive'] for e in epochs),
        'protection_rejected_epochs': sum(not e['protected'] for e in epochs),
        'joint_positive_protection_rejected_epochs': sum(e['joint_positive_protection_rejected'] for e in epochs),
        'protection_failure_counts': failure_counts, 'cofailures': cofailures,
        'overlapping_group_unions': union_counts, 'selector_reason_counts': reason_counts,
        'failure_patterns': [{'checks': list(k), 'epochs': v} for k, v in sorted(patterns.items())]}
