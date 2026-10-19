"""Server-only tests of the fixed GW11 decision; no Torch/model dependency."""
import copy
import unittest

from experiments.state_gate_gw11.protocol import (
    ARMS, CANDIDATE, EPOCHS, FROZEN_GW10_OWN4_MAE, GROUPS, screen)


def fixture():
    best = {}
    for arm in ARMS:
        best[arm] = dict(mae=13.5, rmse=23., mape_fraction=.2, signed_bias=-.1,
            valid_count=100, groups={name:dict(valid_count=10,mae=20.,signed_bias=-3.)
                for name in GROUPS})
    best[CANDIDATE]['mae'] = 13.2
    last = copy.deepcopy(best)
    last[CANDIDATE]['mae'] = 13.25
    return best,last,{'OWN4':dict(mae=FROZEN_GW10_OWN4_MAE)}


class StateGateProtocolTests(unittest.TestCase):
    def test_pass_only_identifies_a_confirmation_candidate(self):
        result = screen(*fixture())
        self.assertTrue(all(result['checks'].values()))
        self.assertTrue(result['candidate_for_confirmation'])
        self.assertTrue(result['eligible_for_independent_confirmation'])
        self.assertFalse(result['confirmation_performed'])
        self.assertFalse(result['statistical_significance_claimed'])
        self.assertFalse(result['test_evaluated'])
        self.assertEqual(result['candidate'],CANDIDATE)
        self.assertEqual(result['target_completed_epochs'],EPOCHS)
        self.assertEqual(result['new_updates_per_arm'],dict.fromkeys(ARMS,246))
        self.assertFalse(result['historical_lookup_is_equal_gradient_budget_control'])

    def test_each_same_budget_control_must_clear_the_relative_gain(self):
        for control in ARMS[:-1]:
            with self.subTest(control=control):
                best,last,historical = fixture()
                best[control]['mae'] = 13.21
                result = screen(best,last,historical)
                self.assertFalse(result['eligible_for_independent_confirmation'])
                self.assertFalse(result['checks'][
                    'BEST_relative_gain_vs_'+control+'_at_least_0_005'])

    def test_a_winning_control_is_never_promoted_to_candidate(self):
        best,last,historical = fixture()
        best['STATE_READOUT']['mae'] = 13.
        result = screen(best,last,historical)
        self.assertFalse(result['eligible_for_independent_confirmation'])
        self.assertEqual(result['candidate'],'STATE_GATE')
        self.assertFalse(result['winner_selection'])

    def test_frozen_lookup_is_a_separate_reference_guard(self):
        best,last,historical = fixture()
        best[CANDIDATE]['mae'] = 13.25
        last[CANDIDATE]['mae'] = 13.25
        result = screen(best,last,historical)
        self.assertFalse(result['checks']['BEST_no_worse_than_frozen_GW10_OWN4'])
        self.assertFalse(result['eligible_for_independent_confirmation'])

    def test_last_degradation_is_against_own_best(self):
        best,last,historical = fixture()
        last[CANDIDATE]['mae'] = 13.34
        result = screen(best,last,historical)
        self.assertFalse(result['checks']['LAST_own_degradation_at_most_0_01'])

    def test_comparisons_use_best_controls_not_their_last_records(self):
        best,last,historical = fixture()
        for arm in ARMS[:-1]:
            last[arm]['mae'] = 1.
        self.assertTrue(screen(best,last,historical)['eligible_for_independent_confirmation'])

    def test_each_predeclared_guard_blocks_regression(self):
        for name in GROUPS:
            with self.subTest(group=name):
                best,last,historical = fixture()
                best[CANDIDATE]['groups'][name]['mae'] = 20.001
                result = screen(best,last,historical)
                self.assertFalse(result['checks']['BEST_'+name+'_no_worse_than_SUM_CONTINUE'])
                self.assertFalse(result['eligible_for_independent_confirmation'])

    def test_group_guards_use_same_run_sum_best(self):
        best,last,historical = fixture()
        for name in GROUPS:
            best['STATE_READOUT']['groups'][name]['mae'] = 1.
            last['SUM_CONTINUE']['groups'][name]['mae'] = 1.
        self.assertTrue(screen(best,last,historical)['eligible_for_independent_confirmation'])

    def test_more_negative_high_value_bias_does_not_change_mae_objective(self):
        best,last,historical = fixture()
        best[CANDIDATE]['groups']['high_value_ge314']['signed_bias'] = -10.
        best[CANDIDATE]['signed_bias'] = -1.
        self.assertTrue(screen(best,last,historical)['eligible_for_independent_confirmation'])

    def test_empty_guard_is_a_failure_not_a_vacuous_pass(self):
        best,last,historical = fixture()
        for records in (best,last):
            for arm in ARMS:
                records[arm]['groups']['high_value_ge314'] = dict(
                    valid_count=0,mae=None,signed_bias=None)
        result = screen(best,last,historical)
        self.assertFalse(result['eligible_for_independent_confirmation'])

    def test_wrong_historical_record_is_rejected(self):
        for historical in (13.246264357,{'OWN4':13.246264357},
                {'OWN4':{'mae':14.}}, {'OWN4':{'mae':float('nan')}}):
            with self.subTest(historical=historical):
                best,last,_ = fixture()
                with self.assertRaises(ValueError):
                    screen(best,last,historical)

    def test_nonfinite_global_or_group_metrics_are_rejected(self):
        for field in ('mae','rmse','mape_fraction','signed_bias'):
            best,last,historical = fixture()
            best[CANDIDATE][field] = float('nan')
            with self.subTest(field=field), self.assertRaises(ValueError):
                screen(best,last,historical)
        best,last,historical = fixture()
        best[CANDIDATE]['groups']['high_value_ge314']['mae'] = float('inf')
        with self.assertRaises(ValueError):
            screen(best,last,historical)

    def test_missing_extra_arms_or_groups_are_rejected(self):
        for mode in ('missing_arm','extra_arm','missing_group'):
            best,last,historical = fixture()
            if mode == 'missing_arm':
                del best['STATIC_GATE']
            elif mode == 'extra_arm':
                best['UNREGISTERED'] = copy.deepcopy(best[CANDIDATE])
            else:
                del best[CANDIDATE]['groups']['flow_50_200']
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                screen(best,last,historical)

    def test_different_masks_counts_or_best_last_support_are_rejected(self):
        for mode in ('arm','group','last'):
            best,last,historical = fixture()
            if mode == 'arm':
                best[CANDIDATE]['valid_count'] += 1
            elif mode == 'group':
                best[CANDIDATE]['groups']['flow_50_200']['valid_count'] += 1
            else:
                for arm in ARMS:
                    last[arm]['valid_count'] += 1
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                screen(best,last,historical)


if __name__ == '__main__':
    unittest.main()
