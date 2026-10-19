"""Predeclared GW11 development screen; no model, Torch, or fitting imports.

The runner authenticates checkpoints, masks, completed epochs and optimizer
updates before calling this screen. These checks authorize consideration for a
separate independent confirmation; they never constitute that confirmation.
"""
from numbers import Integral, Real

import numpy as np


ARMS = ('SUM_CONTINUE', 'STATIC_GATE', 'STATE_READOUT', 'STATE_GATE')
CANDIDATE = 'STATE_GATE'
EPOCHS = 3
GROUPS = ('high_value_ge314', 'long_horizon_7_12', 'flow_50_200')
BATCH_SIZE = 48
TRAIN_BATCHES = 82
UPDATES_PER_ARM = EPOCHS * TRAIN_BATCHES
SEED = 2025
MIN_RELATIVE_GAIN = .005
MAX_LAST_DEGRADATION = .01
FROZEN_GW10_OWN4_MAE = 13.246264357


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _number(value, name, nonnegative=False):
    _require(isinstance(value, Real) and not isinstance(value, (bool, np.bool_))
        and np.isfinite(value), 'Finite real metric required: '+name)
    value = float(value)
    _require(not nonnegative or value >= 0., 'Nonnegative metric required: '+name)
    return value


def _count(value, name, allow_empty=False):
    _require(isinstance(value, Integral) and not isinstance(value, (bool, np.bool_))
        and value >= (0 if allow_empty else 1), 'Invalid valid_count: '+name)
    return int(value)


def _validate_records(records, selection):
    _require(isinstance(records, dict) and set(records) == set(ARMS),
        selection+' must contain exactly the four predefined arms')
    for arm in ARMS:
        row = records[arm]
        _require(isinstance(row, dict), 'Metric record must be a dictionary: '+arm)
        prefix = selection+'/'+arm
        _count(row.get('valid_count'), prefix)
        for name in ('mae', 'rmse', 'mape_fraction', 'signed_bias'):
            _number(row.get(name), prefix+'/'+name, nonnegative=name != 'signed_bias')
        groups = row.get('groups')
        _require(isinstance(groups, dict) and set(GROUPS).issubset(groups),
            'All fixed guard groups are required: '+prefix)
        for name in GROUPS:
            group = groups[name]
            _require(isinstance(group, dict), 'Invalid group record: '+prefix+'/'+name)
            count = _count(group.get('valid_count'), prefix+'/'+name, allow_empty=True)
            if count:
                _number(group.get('mae'), prefix+'/'+name+'/mae', nonnegative=True)
                _number(group.get('signed_bias'), prefix+'/'+name+'/signed_bias')
            else:
                _require(group.get('mae') is None and group.get('signed_bias') is None,
                    'An empty group must report unavailable metrics: '+prefix+'/'+name)
    reference = records['SUM_CONTINUE']
    _require(all(row['valid_count'] == reference['valid_count'] for row in records.values()),
        'Global masks/counts differ across arms: '+selection)
    for name in GROUPS:
        count = reference['groups'][name]['valid_count']
        _require(all(row['groups'][name]['valid_count'] == count for row in records.values()),
            'Guard masks/counts differ across arms: '+selection+'/'+name)


def _historical_own4(historical):
    _require(isinstance(historical, dict) and isinstance(historical.get('OWN4'), dict),
        'Pass the authenticated historical record as {OWN4: {mae: ...}}')
    mae = _number(historical['OWN4'].get('mae'), 'historical/OWN4/mae', nonnegative=True)
    # The published value was rounded to nine decimal places. The runner also
    # authenticates the original GW10 record and source/mask identities.
    _require(abs(mae-FROZEN_GW10_OWN4_MAE) <= 5.1e-10,
        'Historical OWN4 differs from the locked GW10 result')
    return mae


def screen(best, last, historical):
    """Evaluate the sole STATE_GATE candidate using complete paired records.

    ``best`` and ``last`` map every ARMS member to a record with mae, rmse,
    mape_fraction, signed_bias, valid_count and GROUPS dictionaries. Every
    group has valid_count, mae and signed_bias; empty groups use None metrics.
    ``historical`` is {"OWN4": {"mae": authenticated_original_mae, ...}}.
    Signed biases are reported without changing the native MAE objective.
    """
    _validate_records(best, 'BEST')
    _validate_records(last, 'LAST')
    for arm in ARMS:
        _require(best[arm]['valid_count'] == last[arm]['valid_count'],
            'BEST/LAST global mask/count differs: '+arm)
        _require(all(best[arm]['groups'][name]['valid_count'] ==
            last[arm]['groups'][name]['valid_count'] for name in GROUPS),
            'BEST/LAST guard mask/count differs: '+arm)
    own4_mae = _historical_own4(historical)
    candidate_mae = float(best[CANDIDATE]['mae'])
    last_mae = float(last[CANDIDATE]['mae'])
    checks, gains = {}, {}
    for control in ARMS[:-1]:
        baseline = float(best[control]['mae'])
        gain = (baseline-candidate_mae)/baseline if baseline > 0. else None
        gains[control] = gain
        checks['BEST_relative_gain_vs_'+control+'_at_least_0_005'] = (
            gain is not None and gain >= MIN_RELATIVE_GAIN)
    checks['BEST_no_worse_than_frozen_GW10_OWN4'] = candidate_mae <= own4_mae
    degradation = (last_mae-candidate_mae)/candidate_mae if candidate_mae > 0. else None
    checks['LAST_own_degradation_at_most_0_01'] = (
        degradation <= MAX_LAST_DEGRADATION if degradation is not None else last_mae == 0.)
    for name in GROUPS:
        candidate = best[CANDIDATE]['groups'][name]
        reference = best['SUM_CONTINUE']['groups'][name]
        checks['BEST_'+name+'_no_worse_than_SUM_CONTINUE'] = (
            candidate['valid_count'] > 0 and candidate['mae'] <= reference['mae'])
    checks = {name: bool(value) for name, value in checks.items()}
    passed = all(checks.values())
    return dict(status='GW11_STATE_GATE_SCREEN_COMPLETE', candidate=CANDIDATE,
        checks=checks, BEST_relative_gains=gains, LAST_own_relative_degradation=degradation,
        frozen_GW10_OWN4_MAE=own4_mae, historical_lookup_is_equal_gradient_budget_control=False,
        candidate_for_confirmation=passed, eligible_for_independent_confirmation=passed,
        confirmation_performed=False, statistical_significance_claimed=False,
        interpretation='predeclared_single_seed_development_screen_not_independent_confirmation',
        winner_selection=False, validation_hyperparameter_search=False,
        target_completed_epochs=EPOCHS, new_updates_per_arm={arm:UPDATES_PER_ARM for arm in ARMS},
        update_budget_scope='fixed_completed_run_budget_authenticated_by_runner',
        best_selection_includes_stage_zero=True, last_completed_epoch=EPOCHS,
        optimizer_reset_at_start=True, optimizer_reset_on_resume=False,
        native_MAE_objective_unchanged=True, additional_bias_loss=False,
        high_value_group_uses_future_truth_for_evaluation_only=True,
        screening_failure_does_not_authorize_budget_extension=True,
        screening_failure_does_not_authorize_retuning=True, test_evaluated=False)
