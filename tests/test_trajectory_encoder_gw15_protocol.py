"""Server-only fixed GW15 utility-screen contract tests; no model imports.

All records below are synthetic bookkeeping fixtures, not experiment results.
"""
from copy import deepcopy
import unittest

from experiments.trajectory_encoder_gw15.protocol import (
    ARMS, CANDIDATE, CASES, EPOCHS, FROZEN_GW10_OWN4_MAE, GROUPS,
    REFERENCE, screen)


def record(mae=13.5):
    return dict(mae=mae,rmse=23.,mape_fraction=.2,signed_bias=-.1,
        window_p95=18.,valid_count=100,groups={name:dict(
            valid_count=20,mae=20.,signed_bias=-3.) for name in GROUPS})


def fixture():
    best = {arm:record() for arm in CASES}
    best[CANDIDATE]['mae'] = 13.3
    last = deepcopy(best)
    last[CANDIDATE]['mae'] = 13.35
    return best,last


class TrajectoryEncoderProtocolTests(unittest.TestCase):
    def test_pass_is_only_internal_encoder_followup_with_locked_budget(self):
        result = screen(*fixture())
        self.assertTrue(all(result['checks'].values()))
        self.assertTrue(result['history_encoder_utility_screen_passed'])
        self.assertTrue(result['eligible_for_internal_encoder_followup'])
        for name in ('candidate_for_confirmation','independent_confirmation_authorized',
                'confirmation_performed','statistical_significance_claimed','test_evaluated',
                'source_requires_grad_flags_are_numerical_replay_only'):
            self.assertFalse(result[name],name)
        self.assertEqual(result['candidate'],CANDIDATE)
        self.assertEqual(ARMS,('SORTED_DELTA','ORDERED_DELTA'))
        self.assertEqual(result['target_completed_epochs'],EPOCHS)
        self.assertEqual(result['new_updates_per_arm'],dict.fromkeys(ARMS,246))
        self.assertEqual(result['frozen_source_optimizer_updates'],0)
        self.assertEqual(result['frozen_reference_optimizer_updates'],0)
        self.assertTrue(result['training_backward_through_frozen_source_required'])
        self.assertTrue(result['only_history_delta_adapters_are_optimizer_parameters'])

    def test_each_best_control_requires_half_percent_and_winner_is_fixed(self):
        for control in (REFERENCE,'SORTED_DELTA'):
            best,last = fixture()
            best[control]['mae'] = 13.31
            if control == REFERENCE:
                last[control]['mae'] = best[control]['mae']
            else:
                last[control]['mae'] = 50.
            with self.subTest(control=control):
                result = screen(best,last)
                self.assertFalse(result['checks'][
                    'BEST_relative_gain_vs_'+control+'_at_least_0_005'])
                self.assertFalse(result['history_encoder_utility_screen_passed'])
                self.assertEqual(result['candidate'],'ORDERED_DELTA')
                self.assertFalse(result['winner_selection'])

    def test_historical_own4_and_bias_are_reports_not_new_gates(self):
        best,last = fixture()
        best[CANDIDATE]['signed_bias'] = -11.
        result = screen(best,last)
        self.assertGreater(best[CANDIDATE]['mae'],FROZEN_GW10_OWN4_MAE)
        self.assertTrue(result['history_encoder_utility_screen_passed'])
        self.assertFalse(result['BEST_no_worse_than_historical_OWN4_report_only'])
        self.assertFalse(result['historical_OWN4_is_a_screening_gate'])
        self.assertFalse(result['historical_lookup_is_equal_gradient_budget_control'])
        self.assertFalse(any('OWN4' in key for key in result['checks']))
        self.assertTrue(result['native_MAE_objective_unchanged'])
        self.assertFalse(result['additional_bias_loss'])

    def test_last_compares_to_own_best_and_each_guard_compares_to_frozen(self):
        best,last = fixture(); last[CANDIDATE]['mae'] = 13.44
        self.assertFalse(screen(best,last)['checks']['LAST_own_degradation_at_most_0_01'])
        for group in GROUPS:
            for mode in ('regression','empty'):
                best,last = fixture()
                best['SORTED_DELTA']['groups'][group]['mae'] = 1.
                self.assertTrue(screen(best,last)['history_encoder_utility_screen_passed'])
                if mode=='regression':
                    best[CANDIDATE]['groups'][group]['mae'] = 20.001
                else:
                    for rows in (best,last):
                        for arm in CASES:
                            rows[arm]['groups'][group] = dict(
                                valid_count=0,mae=None,signed_bias=None)
                with self.subTest(group=group,mode=mode):
                    result = screen(best,last)
                    self.assertFalse(result['checks']['BEST_'+group+'_no_worse_than_FROZEN_SUM'])

    def test_metrics_support_and_frozen_reference_must_be_valid(self):
        changes = (lambda b,l:b[CANDIDATE].update(mae=float('nan')),
            lambda b,l:b[CANDIDATE].update(mae=True),
            lambda b,l:b[CANDIDATE].update(valid_count=101),
            lambda b,l:b.pop('SORTED_DELTA'),
            lambda b,l:b[CANDIDATE]['groups'].pop('flow_50_200'),
            lambda b,l:l[REFERENCE].update(mae=13.6),
            lambda b,l:[l[a].update(valid_count=101) for a in CASES])
        for index,change in enumerate(changes):
            best,last = fixture(); change(best,last)
            with self.subTest(change=index),self.assertRaises(ValueError): screen(best,last)

    def test_zero_control_cannot_create_a_relative_gain(self):
        best,last = fixture()
        best[REFERENCE]['mae'] = last[REFERENCE]['mae'] = 0.
        best[CANDIDATE]['mae'] = last[CANDIDATE]['mae'] = 0.
        result = screen(best,last)
        self.assertIsNone(result['BEST_relative_gains'][REFERENCE])
        self.assertFalse(result['history_encoder_utility_screen_passed'])
        self.assertTrue(result['checks']['LAST_own_degradation_at_most_0_01'])

    def test_failure_preserves_previous_negative_results_and_diagnostic_limits(self):
        best,last = fixture(); best[CANDIDATE]['mae'] = 14.
        result = screen(best,last)
        self.assertFalse(result['history_encoder_utility_screen_passed'])
        for name in ('prior_GW11_failed_screen_unchanged','prior_GW12_failed_screen_unchanged',
                'prior_GW13_failed_screen_unchanged','prior_GW14_diagnostic_only',
                'previous_lookup_evidence_is_diagnostic_only',
                'screening_failure_does_not_authorize_budget_extension',
                'screening_failure_does_not_authorize_retuning'):
            self.assertTrue(result[name],name)
        self.assertFalse(result['validation_hyperparameter_search'])
        self.assertFalse(result['independent_confirmation_authorized'])


if __name__=='__main__':
    unittest.main()
