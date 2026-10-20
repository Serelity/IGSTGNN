"""Server-only locked development decision rules; no fitting or model forwards."""
from copy import deepcopy
import json
import unittest

import numpy as np

from experiments.observed_state_gw17 import ARMS
from experiments.observed_state_gw17.screen import (CANDIDATE, CONTROL, REFERENCE,
    RISK_METRICS, require_same_axes, screen_records)


def fixture():
    identity = dict(identity_sha256='run', config=dict(max_epochs=100), engineering_only=False,
        data=dict(loader_batching=dict(train_loader=dict(batches=82))),
        gw16_reference=dict(directory='historical-reference'),
        initializations={arm:dict(registered_parameters=10000, model_tensor_sha256='paired') for arm in ARMS})
    best, last = {}, {}
    for selection, records in (('best', best), ('last', last)):
        for arm in ARMS:
            records[arm] = dict(identity_sha256='run', arm=arm, split='val', test_evaluated=False,
                gw16_reference=identity['gw16_reference'],
                checkpoint_role=dict(selection=selection, stage='joint', best_includes_stage_zero=True,
                    epoch=80 if selection == 'best' else 100, completed_training_epochs=100,
                    registered_parameters=10000, training_batches_per_epoch=82, global_updates=8200,
                    full_protocol_reached=True),
                diagnostics=dict(overall=dict(mae=9.9 if arm == CANDIDATE else 10., rmse=15.,
                    mape_percent=20.), window_p95=18., history_preupdate=dict(mae=5.)))
    reference = dict(arm='STATE_EXCHANGE', split='val', test_evaluated=False, engineering_only=False,
        checkpoint_role=dict(selection='best', full_protocol_reached=True, completed_training_epochs=100,
            best_includes_stage_zero=True),
        diagnostics=dict(overall=dict(mae=10., rmse=15., mape_percent=20.), window_p95=18.))
    return best, last, identity, reference


class GW17ScreenTests(unittest.TestCase):
    def test_complete_paired_candidate_passes_development_only(self):
        result = screen_records(*fixture())
        self.assertTrue(result['candidate_for_confirmation'])
        self.assertTrue(result['development_eligible'])
        self.assertFalse(result['independent_confirmation'])
        self.assertFalse(result['test_evaluated'])
        self.assertTrue(result['screening_failure_does_not_authorize_budget_extension'])
        self.assertTrue(result['screening_pass_does_not_open_test'])
        self.assertFalse(result['historical_reference_is_equal_budget_control'])
        self.assertEqual(result['historical_reference_new_model_forwards'], 0)
        self.assertTrue(result['matched_capacity_does_not_claim_equal_compute'])
        json.dumps(result, allow_nan=False)

    def test_gain_against_both_comparators_is_required(self):
        for comparator in (CONTROL, REFERENCE):
            with self.subTest(comparator=comparator):
                best, last, identity, reference = fixture()
                record = best[CONTROL] if comparator == CONTROL else reference
                record['diagnostics']['overall']['mae'] = 9.92
                result = screen_records(best, last, identity, reference)
                self.assertFalse(result['candidate_for_confirmation'])
                self.assertFalse(result['checks']['BEST_gain_vs_'+comparator+'_at_least_0_005'])

    def test_each_risk_metric_must_not_regress_against_either_comparator(self):
        for comparator in (CONTROL, REFERENCE):
            for name in RISK_METRICS:
                with self.subTest(comparator=comparator, metric=name):
                    best, last, identity, reference = fixture()
                    record = best[CONTROL] if comparator == CONTROL else reference
                    if name == 'window_p95':
                        record['diagnostics'][name] = 17.9
                    else:
                        record['diagnostics']['overall'][name] -= .1
                    result = screen_records(best, last, identity, reference)
                    self.assertFalse(result['candidate_for_confirmation'])
                    self.assertFalse(result['checks']['BEST_'+name+'_no_worse_than_'+comparator])

    def test_tied_risk_metrics_are_allowed_without_relaxing_mae_gate(self):
        result = screen_records(*fixture())
        for comparator in (CONTROL, REFERENCE):
            for name in RISK_METRICS:
                self.assertTrue(result['checks']['BEST_'+name+'_no_worse_than_'+comparator])

    def test_last_regression_does_not_trigger_extra_epochs(self):
        best, last, identity, reference = fixture()
        last[CANDIDATE]['diagnostics']['overall']['mae'] = 10.1
        result = screen_records(best, last, identity, reference)
        self.assertFalse(result['checks']['LAST_own_degradation_at_most_0_01'])
        self.assertFalse(result['candidate_for_confirmation'])

    def test_partial_budget_and_engineering_never_eligible(self):
        for engineering, completed, maximum in ((False, 50, 100), (True, 3, 3)):
            with self.subTest(engineering=engineering):
                best, last, identity, reference = fixture()
                identity.update(engineering_only=engineering)
                identity['config']['max_epochs'] = maximum
                for selection, records in (('best', best), ('last', last)):
                    for record in records.values():
                        record['checkpoint_role'].update(completed_training_epochs=completed,
                            epoch=0 if selection == 'best' else completed, global_updates=82*completed,
                            full_protocol_reached=completed == maximum)
                result = screen_records(best, last, identity, reference)
                self.assertFalse(result['candidate_for_confirmation'])

    def test_capacity_must_be_exact_not_approximate(self):
        best, last, identity, reference = fixture()
        identity['initializations'][CONTROL]['registered_parameters'] = 9999
        for records in (best, last):
            records[CONTROL]['checkpoint_role']['registered_parameters'] = 9999
        result = screen_records(best, last, identity, reference)
        self.assertFalse(result['checks']['exact_equal_parameter_counts'])
        self.assertFalse(result['candidate_for_confirmation'])

    def test_identical_parameter_count_does_not_replace_paired_tensors(self):
        best, last, identity, reference = fixture()
        identity['initializations'][CONTROL]['model_tensor_sha256'] = 'unpaired'
        result = screen_records(best, last, identity, reference)
        self.assertFalse(result['checks']['exact_paired_initialization'])
        self.assertFalse(result['candidate_for_confirmation'])

    def test_missing_arm_or_last_is_rejected(self):
        best, last, identity, reference = fixture()
        del last[CONTROL]
        with self.assertRaisesRegex(ValueError, 'Both GW17'):
            screen_records(best, last, identity, reference)

    def test_best_may_be_epoch_zero_but_last_must_be_final(self):
        best, last, identity, reference = fixture()
        best[CANDIDATE]['checkpoint_role']['epoch'] = 0
        self.assertTrue(screen_records(best, last, identity, reference)['candidate_for_confirmation'])
        last[CANDIDATE]['checkpoint_role']['epoch'] = 0
        with self.assertRaisesRegex(ValueError, 'LAST'):
            screen_records(best, last, identity, reference)

    def test_mixed_run_reference_or_split_is_rejected(self):
        for field, value in (('identity_sha256', 'other'), ('gw16_reference', {'directory':'other'}),
                             ('split', 'test'), ('test_evaluated', True)):
            with self.subTest(field=field):
                best, last, identity, reference = fixture()
                last[CANDIDATE][field] = value
                with self.assertRaisesRegex(ValueError, 'identity, split or reference'):
                    screen_records(best, last, identity, reference)

    def test_best_and_last_at_different_budget_are_rejected(self):
        best, last, identity, reference = fixture()
        last[CANDIDATE]['checkpoint_role']['global_updates'] -= 1
        with self.assertRaisesRegex(ValueError, 'different training budgets'):
            screen_records(best, last, identity, reference)

    def test_reference_cannot_be_test_partial_last_or_engineering(self):
        for field, value in (('split', 'test'), ('engineering_only', True), ('test_evaluated', True)):
            with self.subTest(field=field):
                best, last, identity, reference = fixture()
                reference[field] = value
                with self.assertRaisesRegex(ValueError, 'formal GW16'):
                    screen_records(best, last, identity, reference)
        for field, value in (('selection', 'last'), ('completed_training_epochs', 99),
                             ('full_protocol_reached', False), ('best_includes_stage_zero', False)):
            with self.subTest(field=field):
                best, last, identity, reference = fixture()
                reference['checkpoint_role'][field] = value
                with self.assertRaisesRegex(ValueError, 'full-100 BEST'):
                    screen_records(best, last, identity, reference)

    def test_unavailable_or_nonfinite_risk_metric_is_rejected(self):
        for value in (None, float('nan'), float('inf')):
            with self.subTest(value=value):
                best, last, identity, reference = fixture()
                best[CANDIDATE]['diagnostics']['overall']['mape_percent'] = value
                with self.assertRaisesRegex(ValueError, 'Invalid screening metric'):
                    screen_records(best, last, identity, reference)

    def test_history_auxiliary_mae_cannot_rescue_bad_future_mae(self):
        best, last, identity, reference = fixture()
        best[CANDIDATE]['diagnostics']['overall']['mae'] = 10.1
        best[CANDIDATE]['diagnostics']['history_preupdate']['mae'] = 0.
        self.assertFalse(screen_records(best, last, identity, reference)['candidate_for_confirmation'])

    def test_pairing_covers_observed_history_masks_and_future_axes(self):
        left = {key:np.arange(4, dtype=np.float32) for key in
            ('target', 'reference', 'sample_ids', 'station_ids', 'assignment',
             'support', 'available', 'history_target', 'history_valid')}
        require_same_axes(left, deepcopy(left))
        for key in left:
            with self.subTest(key=key):
                right = deepcopy(left)
                right[key] = right[key][::-1]
                with self.assertRaisesRegex(ValueError, key):
                    require_same_axes(left, right)


if __name__ == '__main__':
    unittest.main()
