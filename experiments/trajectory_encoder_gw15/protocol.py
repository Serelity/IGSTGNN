"""Predeclared internal history-encoder probe, without fitting/model imports.

GW13 and GW14 provide a small, development-only ordered-history clue; neither
established a structural improvement. GW15 tests that separate question with
two equal-capacity adapters before the existing ICSF/ST interactions. The
authenticated GW10 OWN4 score remains historical context, not a trained arm.
"""
from numbers import Integral, Real

import numpy as np


ARMS = ('SORTED_DELTA', 'ORDERED_DELTA')
REFERENCE = 'FROZEN_SUM'
CASES = (REFERENCE,) + ARMS
CANDIDATE = 'ORDERED_DELTA'
EPOCHS = 3
GROUPS = ('high_value_ge314', 'long_horizon_7_12', 'flow_50_200')
BATCH_SIZE = 48
TRAIN_COUNT = 3909
TRAIN_BATCHES = 82
UPDATES_PER_ARM = EPOCHS * TRAIN_BATCHES
SEED = 2025
MIN_RELATIVE_GAIN = .005
MAX_LAST_DEGRADATION = .01
FROZEN_GW10_OWN4_MAE = 13.246264357


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _number(value, name, nonnegative=False):
    require(isinstance(value, Real) and not isinstance(value, (bool, np.bool_))
        and np.isfinite(value), 'Finite real metric required: '+name)
    value = float(value)
    require(not nonnegative or value >= 0., 'Nonnegative metric required: '+name)
    return value


def _count(value, name, allow_empty=False):
    require(isinstance(value, Integral) and not isinstance(value, (bool, np.bool_))
        and value >= (0 if allow_empty else 1), 'Invalid valid_count: '+name)
    return int(value)


def validate_records(records, selection):
    """Check complete paired metric records; support hashes are runner checks."""
    require(isinstance(records, dict) and set(records) == set(CASES),
        selection+' must contain exactly FROZEN_SUM and the two predefined adapters')
    for arm in CASES:
        row = records[arm]
        require(isinstance(row, dict), 'Metric record must be a dictionary: '+arm)
        prefix = selection+'/'+arm
        _count(row.get('valid_count'), prefix)
        for name in ('mae', 'rmse', 'mape_fraction', 'signed_bias', 'window_p95'):
            _number(row.get(name), prefix+'/'+name, nonnegative=name != 'signed_bias')
        groups = row.get('groups')
        require(isinstance(groups, dict) and set(GROUPS).issubset(groups),
            'All fixed guard groups are required: '+prefix)
        for name in GROUPS:
            group = groups[name]
            require(isinstance(group, dict), 'Invalid group record: '+prefix+'/'+name)
            count = _count(group.get('valid_count'), prefix+'/'+name, allow_empty=True)
            require(count <= row['valid_count'], 'Group support exceeds global support: '+prefix+'/'+name)
            if count:
                _number(group.get('mae'), prefix+'/'+name+'/mae', nonnegative=True)
                _number(group.get('signed_bias'), prefix+'/'+name+'/signed_bias')
            else:
                require(group.get('mae') is None and group.get('signed_bias') is None,
                    'An empty group must report unavailable metrics: '+prefix+'/'+name)
    assert_same_support(records, selection)


def assert_same_support(records, selection):
    reference = records[REFERENCE]
    require(all(row['valid_count'] == reference['valid_count'] for row in records.values()),
        'Global masks/counts differ across cases: '+selection)
    for name in GROUPS:
        count = reference['groups'][name]['valid_count']
        require(all(row['groups'][name]['valid_count'] == count for row in records.values()),
            'Guard masks/counts differ across cases: '+selection+'/'+name)


def assert_same_reference(first, second, context):
    """FROZEN_SUM must reproduce the same fixed validation metric record."""
    for name in ('mae', 'rmse', 'mape_fraction', 'signed_bias', 'window_p95', 'valid_count'):
        require(first[name] == second[name], 'Frozen source metric changed: '+context+'/'+name)
    for group in GROUPS:
        for name in ('valid_count', 'mae', 'signed_bias'):
            require(first['groups'][group][name] == second['groups'][group][name],
                'Frozen source group metric changed: '+context+'/'+group+'/'+name)


def screen(best, last):
    """Screen ordered delta injection under a fixed source, never select a winner.

    Both dictionaries contain exactly CASES, using the GW11 runner record
    format. BEST includes stage zero independently for each trained adapter.
    Signed bias and the historical OWN4 score are reported, without gates or
    changes to the native MAE objective. The runner authenticates budgets,
    caches, replay, source tensors and optimizer membership before screening.
    """
    validate_records(best, 'BEST')
    validate_records(last, 'LAST')
    for arm in CASES:
        require(best[arm]['valid_count'] == last[arm]['valid_count'],
            'BEST/LAST global mask/count differs: '+arm)
        require(all(best[arm]['groups'][name]['valid_count'] ==
            last[arm]['groups'][name]['valid_count'] for name in GROUPS),
            'BEST/LAST guard mask/count differs: '+arm)
    assert_same_reference(best[REFERENCE], last[REFERENCE], 'BEST/LAST')
    candidate_mae = float(best[CANDIDATE]['mae'])
    last_mae = float(last[CANDIDATE]['mae'])
    checks, gains = {}, {}
    for control in (REFERENCE, 'SORTED_DELTA'):
        baseline = float(best[control]['mae'])
        gain = (baseline-candidate_mae)/baseline if baseline > 0. else None
        gains[control] = gain
        checks['BEST_relative_gain_vs_'+control+'_at_least_0_005'] = (
            gain is not None and gain >= MIN_RELATIVE_GAIN)
    degradation = (last_mae-candidate_mae)/candidate_mae if candidate_mae > 0. else None
    checks['LAST_own_degradation_at_most_0_01'] = (
        degradation <= MAX_LAST_DEGRADATION if degradation is not None else last_mae == 0.)
    for name in GROUPS:
        candidate = best[CANDIDATE]['groups'][name]
        reference = best[REFERENCE]['groups'][name]
        checks['BEST_'+name+'_no_worse_than_FROZEN_SUM'] = (
            candidate['valid_count'] > 0 and candidate['mae'] <= reference['mae'])
    checks = {name: bool(value) for name, value in checks.items()}
    passed = all(checks.values())
    return dict(status='GW15_TRAJECTORY_ENCODER_SCREEN_COMPLETE', candidate=CANDIDATE,
        checks=checks, BEST_relative_gains=gains, LAST_own_relative_degradation=degradation,
        history_encoder_utility_screen_passed=passed, eligible_for_internal_encoder_followup=passed,
        candidate_for_confirmation=False, independent_confirmation_authorized=False,
        confirmation_performed=False, statistical_significance_claimed=False,
        frozen_GW10_OWN4_MAE=FROZEN_GW10_OWN4_MAE,
        BEST_no_worse_than_historical_OWN4_report_only=candidate_mae <= FROZEN_GW10_OWN4_MAE,
        historical_OWN4_is_a_screening_gate=False,
        historical_lookup_is_equal_gradient_budget_control=False,
        interpretation='predeclared_single_seed_internal_history_encoder_probe_not_confirmation',
        prior_GW11_failed_screen_unchanged=True,
        prior_GW12_failed_screen_unchanged=True,
        prior_GW13_failed_screen_unchanged=True,
        prior_GW14_diagnostic_only=True,
        previous_lookup_evidence_is_diagnostic_only=True,
        internal_structure_probe=True, winner_selection=False,
        validation_hyperparameter_search=False, target_completed_epochs=EPOCHS,
        new_updates_per_arm={arm:UPDATES_PER_ARM for arm in ARMS},
        frozen_source_optimizer_updates=0, frozen_reference_optimizer_updates=0,
        source_requires_grad_flags_are_numerical_replay_only=False,
        source_requires_grad_flags_preserved_for_certified_replay=True,
        frozen_source_eval_flags_unchanged=True,
        training_backward_through_frozen_source_required=True,
        only_history_delta_adapters_are_optimizer_parameters=True,
        injection_site='embedding_before_existing_ICSF_ST',
        sorted_and_ordered_delta_capacity_identical=True,
        embedding_residual_is_bounded=True,
        best_selection_includes_stage_zero=True, last_completed_epoch=EPOCHS,
        optimizer_reset_at_start=True, optimizer_reset_on_resume=False,
        native_MAE_objective_unchanged=True, additional_bias_loss=False,
        high_value_group_uses_future_truth_for_evaluation_only=True,
        screening_failure_does_not_authorize_budget_extension=True,
        screening_failure_does_not_authorize_retuning=True, test_evaluated=False)
