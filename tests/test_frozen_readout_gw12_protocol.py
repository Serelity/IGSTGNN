"""Server-only GW12 information-screen tests; no model or Torch dependency."""
from copy import deepcopy
import unittest

from experiments.frozen_readout_gw12.protocol import (
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


class FrozenReadoutProtocolTests(unittest.TestCase):
    def test_pass_answers_only_the_fixed_source_information_question(self):
        result = screen(*fixture())
        self.assertTrue(all(result['checks'].values()))
        self.assertTrue(result['state_information_probe_passed'])
        self.assertTrue(result['eligible_for_frozen_readout_followup'])
        self.assertFalse(result['candidate_for_confirmation'])
        self.assertFalse(result['independent_confirmation_authorized'])
        self.assertFalse(result['confirmation_performed'])
        self.assertFalse(result['statistical_significance_claimed'])
        self.assertFalse(result['test_evaluated'])
        self.assertEqual(result['candidate'],CANDIDATE)
        self.assertEqual(result['target_completed_epochs'],EPOCHS)
        self.assertEqual(result['new_updates_per_arm'],dict.fromkeys(ARMS,246))
        self.assertEqual(result['frozen_source_optimizer_updates'],0)
        self.assertEqual(result['frozen_reference_optimizer_updates'],0)

    def test_each_reference_must_clear_half_percent_relative_gain(self):
        for control in (REFERENCE,'STATIC_READOUT'):
            best,last = fixture()
            best[control]['mae'] = 13.31
            if control == REFERENCE:
                last[control]['mae'] = best[control]['mae']
            with self.subTest(control=control):
                result = screen(best,last)
                self.assertFalse(result['checks'][
                    'BEST_relative_gain_vs_'+control+'_at_least_0_005'])
                self.assertFalse(result['state_information_probe_passed'])

    def test_own4_is_reported_without_being_a_hard_gate(self):
        result = screen(*fixture())
        self.assertGreater(13.3,FROZEN_GW10_OWN4_MAE)
        self.assertTrue(result['state_information_probe_passed'])
        self.assertEqual(result['frozen_GW10_OWN4_MAE'],FROZEN_GW10_OWN4_MAE)
        self.assertFalse(result['historical_OWN4_is_a_screening_gate'])
        self.assertFalse(result['BEST_no_worse_than_historical_OWN4_report_only'])
        self.assertFalse(result['historical_lookup_is_equal_gradient_budget_control'])
        self.assertFalse(any('OWN4' in key for key in result['checks']))

    def test_a_better_static_control_is_not_promoted_after_the_results(self):
        best,last = fixture()
        best['STATIC_READOUT']['mae'] = 12.
        result = screen(best,last)
        self.assertFalse(result['state_information_probe_passed'])
        self.assertEqual(result['candidate'],'STATE_READOUT')
        self.assertFalse(result['winner_selection'])

    def test_last_degradation_is_measured_against_candidate_own_best(self):
        best,last = fixture()
        last[CANDIDATE]['mae'] = 13.44
        result = screen(best,last)
        self.assertFalse(result['checks']['LAST_own_degradation_at_most_0_01'])
        self.assertFalse(result['state_information_probe_passed'])

    def test_static_best_is_the_control_even_if_its_last_is_worse(self):
        best,last = fixture()
        best['STATIC_READOUT']['mae'] = 13.31
        last['STATIC_READOUT']['mae'] = 50.
        result = screen(best,last)
        self.assertFalse(result['checks']['BEST_relative_gain_vs_STATIC_READOUT_at_least_0_005'])

    def test_each_predeclared_guard_blocks_regression(self):
        for group in GROUPS:
            best,last = fixture()
            best[CANDIDATE]['groups'][group]['mae'] = 20.001
            with self.subTest(group=group):
                result = screen(best,last)
                self.assertFalse(result['checks']['BEST_'+group+'_no_worse_than_FROZEN_SUM'])
                self.assertFalse(result['state_information_probe_passed'])

    def test_group_guards_compare_to_frozen_not_static(self):
        best,last = fixture()
        for group in GROUPS:
            best['STATIC_READOUT']['groups'][group]['mae'] = 1.
        self.assertTrue(screen(best,last)['state_information_probe_passed'])

    def test_bias_does_not_add_a_new_loss_or_screening_threshold(self):
        best,last = fixture()
        best[CANDIDATE]['signed_bias'] = -11.
        best[CANDIDATE]['groups']['high_value_ge314']['signed_bias'] = -20.
        result = screen(best,last)
        self.assertTrue(result['state_information_probe_passed'])
        self.assertTrue(result['native_MAE_objective_unchanged'])
        self.assertFalse(result['additional_bias_loss'])

    def test_empty_guard_fails_instead_of_passing_vacuously(self):
        best,last = fixture()
        for rows in (best,last):
            for arm in CASES:
                rows[arm]['groups']['flow_50_200'] = dict(
                    valid_count=0,mae=None,signed_bias=None)
        result = screen(best,last)
        self.assertFalse(result['checks']['BEST_flow_50_200_no_worse_than_FROZEN_SUM'])

    def test_nonfinite_and_boolean_global_metrics_are_rejected(self):
        for field in ('mae','rmse','mape_fraction','signed_bias','window_p95'):
            for invalid in (float('nan'),float('inf'),True):
                best,last = fixture(); best[CANDIDATE][field] = invalid
                with self.subTest(field=field,invalid=invalid),self.assertRaises(ValueError):
                    screen(best,last)

    def test_negative_errors_and_noninteger_support_are_rejected(self):
        for field in ('mae','rmse','mape_fraction','window_p95'):
            best,last = fixture(); best[CANDIDATE][field] = -.01
            with self.subTest(field=field),self.assertRaises(ValueError): screen(best,last)
        for invalid in (0,-1,100.,True):
            best,last = fixture(); best[CANDIDATE]['valid_count'] = invalid
            with self.subTest(count=invalid),self.assertRaises(ValueError): screen(best,last)

    def test_invalid_or_incomplete_group_metrics_are_rejected(self):
        changes = (dict(mae=float('nan')),dict(signed_bias=float('inf')),
            dict(valid_count=101),dict(valid_count=True),dict(valid_count=0,mae=0.,signed_bias=0.))
        for change in changes:
            best,last = fixture()
            best[CANDIDATE]['groups']['flow_50_200'].update(change)
            with self.subTest(change=change),self.assertRaises(ValueError): screen(best,last)

    def test_exact_cases_and_all_guard_groups_are_required(self):
        for mode in ('missing','extra','missing_frozen','missing_group','missing_p95'):
            best,last = fixture()
            if mode=='missing': del best['STATIC_READOUT']
            elif mode=='extra': best['POSTHOC'] = record()
            elif mode=='missing_frozen': del best[REFERENCE]
            elif mode=='missing_group': del best[CANDIDATE]['groups']['flow_50_200']
            else: del best[CANDIDATE]['window_p95']
            with self.subTest(mode=mode),self.assertRaises(ValueError): screen(best,last)

    def test_support_must_match_cases_and_best_last(self):
        for mode in ('case_global','case_group','selection_global','selection_group'):
            best,last = fixture()
            if mode=='case_global': best[CANDIDATE]['valid_count'] += 1
            elif mode=='case_group': best[CANDIDATE]['groups']['flow_50_200']['valid_count'] += 1
            elif mode=='selection_global':
                for arm in CASES: last[arm]['valid_count'] += 1
            else:
                for arm in CASES: last[arm]['groups']['flow_50_200']['valid_count'] += 1
            with self.subTest(mode=mode),self.assertRaises(ValueError): screen(best,last)

    def test_frozen_source_metrics_cannot_drift_between_best_and_last(self):
        for field in ('mae','rmse','mape_fraction','signed_bias','window_p95'):
            best,last = fixture(); last[REFERENCE][field] += .001
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,'Frozen source'):
                screen(best,last)
        best,last = fixture()
        last[REFERENCE]['groups']['high_value_ge314']['mae'] += .001
        with self.assertRaisesRegex(ValueError,'Frozen source'): screen(best,last)

    def test_zero_error_control_does_not_create_a_relative_gain(self):
        best,last = fixture()
        best[REFERENCE]['mae'] = last[REFERENCE]['mae'] = 0.
        best[CANDIDATE]['mae'] = last[CANDIDATE]['mae'] = 0.
        result = screen(best,last)
        self.assertIsNone(result['BEST_relative_gains'][REFERENCE])
        self.assertFalse(result['state_information_probe_passed'])
        self.assertTrue(result['checks']['LAST_own_degradation_at_most_0_01'])

    def test_failure_does_not_extend_budget_or_rewrite_gw11(self):
        best,last = fixture(); best[CANDIDATE]['mae'] = 14.
        result = screen(best,last)
        self.assertFalse(result['state_information_probe_passed'])
        self.assertTrue(result['prior_GW11_failed_screen_unchanged'])
        self.assertTrue(result['screening_failure_does_not_authorize_budget_extension'])
        self.assertTrue(result['screening_failure_does_not_authorize_retuning'])
        self.assertFalse(result['validation_hyperparameter_search'])
        self.assertFalse(result['independent_confirmation_authorized'])


if __name__=='__main__':
    unittest.main()
