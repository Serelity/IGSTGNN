"""Locked screens for a train-fitted frozen-SUM trajectory diagnostic.

The internal audit holds out labels from every fitted correction table. The
already-trained SUM source is not an independent model on that internal tail.
Passing these screens licenses architecture design only, never confirmation.
"""
from numbers import Integral, Real

import numpy as np


SCHEMA = 'gw13_trajectory_information_protocol_v1'
ARMS = ('SUM', 'ORIGIN', 'OWN4', 'CONTEXT', 'MAGNITUDE16',
    'DIRECTION16', 'TRAJECTORY16')
CANDIDATE = 'TRAJECTORY16'
CHILD_ARMS = ('MAGNITUDE16', 'DIRECTION16', 'TRAJECTORY16')
COVERAGE_ARMS = ('CONTEXT',) + CHILD_ARMS
GROUPS = ('high_value_ge314', 'long_horizon_7_12', 'flow_50_200')
AUDIT_CONTROLS = ('OWN4', 'CONTEXT', 'MAGNITUDE16', 'DIRECTION16')
VALIDATION_GAIN_CONTROLS = ('SUM', 'OWN4', 'CONTEXT')
MIN_ACTIVE_FRACTION = .25
MIN_RELATIVE_GAIN = .005
AUDIT_CHECKS = tuple('candidate_beats_'+arm for arm in AUDIT_CONTROLS) + tuple(
    'active_fraction_'+arm+'_at_least_0_25' for arm in CHILD_ARMS)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _number(value, name, nonnegative=False):
    require(isinstance(value, Real) and not isinstance(value, (bool, np.bool_))
        and np.isfinite(value), 'Finite real metric required: '+name)
    value = float(value)
    require(np.isfinite(value), 'Metric must remain finite in float64: '+name)
    require(not nonnegative or value >= 0., 'Nonnegative metric required: '+name)
    return value


def _count(value, name, allow_empty=False):
    require(isinstance(value, Integral) and not isinstance(value, (bool, np.bool_))
        and value >= (0 if allow_empty else 1), 'Invalid integer count: '+name)
    return int(value)


def validate_records(records, phase='records'):
    """Require complete paired records; the runner authenticates mask bytes."""
    require(isinstance(records, dict) and set(records) == set(ARMS),
        phase+' requires exactly the seven predeclared arms')
    for arm in ARMS:
        row = records[arm]
        require(isinstance(row, dict), 'Metric record must be a dictionary: '+arm)
        _count(row.get('valid_count'), phase+'/'+arm+'/valid_count')
        for name in ('mae', 'rmse', 'mape_fraction', 'signed_bias'):
            _number(row.get(name), phase+'/'+arm+'/'+name,
                nonnegative=name != 'signed_bias')
    reference_count = records['SUM']['valid_count']
    require(all(row['valid_count'] == reference_count for row in records.values()),
        'Paired global valid_count differs across arms: '+phase)


def validate_coverage(coverage):
    """Validate held-out point coverage of each arm's own eligible table level.

    An active point uses that arm's supported table directly, rather than a
    fallback. Training cell counts and held-out point coverage are different
    quantities: an eligible training cell need not receive a held-out point.
    """
    require(isinstance(coverage, dict) and set(COVERAGE_ARMS).issubset(coverage),
        'Coverage requires CONTEXT and all three child arms')
    for arm in COVERAGE_ARMS:
        row = coverage[arm]
        require(isinstance(row, dict), 'Coverage record must be a dictionary: '+arm)
        fraction = _number(row.get('active_fraction'), arm+'/active_fraction', True)
        require(fraction <= 1., 'Active fraction exceeds one: '+arm)
        eligible = _count(row.get('eligible_cells'), arm+'/eligible_cells', True)
        observed = _count(row.get('observed_cells'), arm+'/observed_cells', True)
        require(eligible <= observed, 'Eligible cell count exceeds observed cells: '+arm)
        require(fraction == 0. or eligible > 0,
            'Positive active coverage requires eligible cells: '+arm)
    context_fraction = float(coverage['CONTEXT']['active_fraction'])
    require(all(float(coverage[arm]['active_fraction']) <= context_fraction+1e-12
        for arm in CHILD_ARMS), 'Child own-table coverage cannot exceed its CONTEXT parent')


def validate_groups(groups, records):
    """Guard masks are shared across all arms and never define fit features."""
    require(isinstance(groups, dict) and set(groups) == set(ARMS),
        'Group records require exactly the seven predeclared arms')
    for arm in ARMS:
        rows = groups[arm]
        require(isinstance(rows, dict) and set(GROUPS).issubset(rows),
            'All three fixed guard groups are required: '+arm)
        for name in GROUPS:
            row = rows[name]
            require(isinstance(row, dict), 'Group record must be a dictionary: '+arm+'/'+name)
            count = _count(row.get('valid_count'), arm+'/'+name+'/valid_count')
            require(count <= records[arm]['valid_count'],
                'Group valid_count exceeds global count: '+arm+'/'+name)
            _number(row.get('mae'), arm+'/'+name+'/mae', True)
            _number(row.get('signed_bias'), arm+'/'+name+'/signed_bias')
            for metric in ('rmse', 'mape_fraction'):
                if metric in row:
                    _number(row[metric], arm+'/'+name+'/'+metric, True)
    for name in GROUPS:
        count = groups['SUM'][name]['valid_count']
        require(all(groups[arm][name]['valid_count'] == count for arm in ARMS),
            'Paired guard valid_count differs across arms: '+name)


def audit_screen(records, coverage):
    """Screen the fixed candidate on the internal correction-fit holdout only.

    The 25 percent activity rule is an operational support safeguard. It is not
    a confidence bound or a claim that the nominal cell budgets are equally
    expressive. Inadequate support leaves the information question unresolved.
    """
    validate_records(records, 'internal_audit')
    validate_coverage(coverage)
    candidate_mae = float(records[CANDIDATE]['mae'])
    checks = {'candidate_beats_'+arm: candidate_mae < float(records[arm]['mae'])
        for arm in AUDIT_CONTROLS}
    support_checks = {'active_fraction_'+arm+'_at_least_0_25':
        float(coverage[arm]['active_fraction']) >= MIN_ACTIVE_FRACTION
        for arm in CHILD_ARMS}
    checks.update(support_checks)
    checks = {name: bool(value) for name, value in checks.items()}
    supported = all(support_checks.values())
    passed = all(checks.values())
    outcome = ('inconclusive_support' if not supported else
        'increment_on_internal_audit' if passed else 'no_increment_under_fixed_probe')
    return dict(schema=SCHEMA, status='GW13_INTERNAL_AUDIT_COMPLETE',
        phase='audit_completed', candidate=CANDIDATE, checks=checks,
        support_sufficient=bool(supported), outcome=outcome,
        audit_passed=passed, validation_scoring_permitted=passed,
        full_training_fit_permitted=passed,
        allow_full_training_fit_without_passed_audit=False,
        min_active_fraction=MIN_ACTIVE_FRACTION,
        activity_threshold_is_operational_not_statistical=True,
        candidate_for_confirmation=False, independent_confirmation_authorized=False,
        statistical_significance_claimed=False, causal_effect_claimed=False,
        fitting_based_on_validation=False, winner_selection=False,
        prior_failed_screens_unchanged=True, test_evaluated=False,
        interpretation='correction_fit_holdout_under_already_trained_frozen_SUM')


def _require_passed_audit(decision):
    require(isinstance(decision, dict) and decision.get('schema') == SCHEMA
        and decision.get('status') == 'GW13_INTERNAL_AUDIT_COMPLETE'
        and decision.get('phase') == 'audit_completed'
        and decision.get('candidate') == CANDIDATE,
        'A complete GW13 internal audit decision is required before validation scoring')
    checks = decision.get('checks')
    require(isinstance(checks, dict) and set(checks) == set(AUDIT_CHECKS)
        and all(value is True for value in checks.values()),
        'Every locked internal audit check must pass before validation scoring')
    require(all(decision.get(name) is True for name in
        ('audit_passed', 'support_sufficient', 'validation_scoring_permitted',
         'full_training_fit_permitted'))
        and decision.get('outcome') == 'increment_on_internal_audit'
        and decision.get('allow_full_training_fit_without_passed_audit') is False,
        'Failed or incomplete internal audit cannot authorize full fitting or validation')
    require(decision.get('min_active_fraction') == MIN_ACTIVE_FRACTION
        and decision.get('activity_threshold_is_operational_not_statistical') is True
        and all(decision.get(name) is False for name in
            ('candidate_for_confirmation', 'independent_confirmation_authorized',
             'statistical_significance_claimed', 'causal_effect_claimed',
             'fitting_based_on_validation', 'winner_selection', 'test_evaluated')),
        'Audit boundaries or non-confirmation constraints differ from the locked protocol')


def validation_screen(records, groups, audit_decision):
    """Evaluate the one locked candidate after a successful internal audit.

    The runner must check audit authorization before scoring newly fitted
    corrections on development labels. Historical source/cache authentication
    already reads those labels. No comparison here authorizes retuning or test
    access, and an audit failure must skip full-training correction fitting.
    """
    _require_passed_audit(audit_decision)
    validate_records(records, 'development_validation')
    validate_groups(groups, records)
    candidate_mae = float(records[CANDIDATE]['mae'])
    checks, gains = {}, {}
    for arm in VALIDATION_GAIN_CONTROLS:
        baseline = float(records[arm]['mae'])
        gains[arm] = (baseline-candidate_mae)/baseline if baseline > 0. else None
        # Comparing the score to the declared boundary avoids cancellation in
        # the relative-gain calculation exactly at a half-percent improvement.
        checks['relative_gain_vs_'+arm+'_at_least_0_005'] = (
            baseline > 0. and candidate_mae <= baseline*(1.-MIN_RELATIVE_GAIN))
    for arm in ('MAGNITUDE16', 'DIRECTION16'):
        baseline = float(records[arm]['mae'])
        gains[arm] = (baseline-candidate_mae)/baseline if baseline > 0. else None
        checks['candidate_beats_'+arm] = candidate_mae < baseline
    for name in GROUPS:
        checks[name+'_no_worse_than_SUM'] = (
            float(groups[CANDIDATE][name]['mae']) <= float(groups['SUM'][name]['mae']))
    checks = {name: bool(value) for name, value in checks.items()}
    passed = all(checks.values())
    return dict(schema=SCHEMA, status='GW13_TRAJECTORY_INFORMATION_SCREEN_COMPLETE',
        phase='completed', candidate=CANDIDATE, checks=checks, relative_gains=gains,
        trajectory_information_screen_passed=passed,
        eligible_for_architecture_design=passed,
        candidate_for_confirmation=False, independent_confirmation_authorized=False,
        confirmation_performed=False, statistical_significance_claimed=False,
        causal_effect_claimed=False, fitting_based_on_validation=False,
        prior_failed_screens_unchanged=True, winner_selection=False,
        validation_hyperparameter_search=False,
        screening_failure_does_not_authorize_retuning=True,
        test_evaluated=False,
        interpretation='single_development_screen_of_train_fitted_trajectory_tables_not_confirmation')
