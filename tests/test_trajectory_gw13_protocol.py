"""Server-only protocol checks; no model imports or optimization."""
import unittest

import numpy as np

from experiments.trajectory_diagnostic_gw13.protocol import (
    ARMS, AUDIT_CONTROLS, CANDIDATE, CHILD_ARMS, COVERAGE_ARMS, GROUPS,
    VALIDATION_GAIN_CONTROLS, audit_screen, validation_screen)


def record(mae=10.):
    return dict(valid_count=100, mae=mae, rmse=15., mape_fraction=.2, signed_bias=-.1)


def fixture():
    records = {arm: record() for arm in ARMS}
    records[CANDIDATE]['mae'] = 9.9
    coverage = {arm: dict(active_fraction=.5, eligible_cells=10, observed_cells=20)
        for arm in COVERAGE_ARMS}
    groups = {arm: {name: dict(valid_count=20, mae=12., signed_bias=-.2)
        for name in GROUPS} for arm in ARMS}
    return records, coverage, groups


def passed_audit():
    records, coverage, _ = fixture()
    return audit_screen(records, coverage)


class TrajectoryProtocolTests(unittest.TestCase):
    def test_passed_audit_only_permits_fixed_full_fit_and_development_scoring(self):
        decision = passed_audit()
        for name in ('audit_passed', 'support_sufficient',
                'full_training_fit_permitted', 'validation_scoring_permitted'):
            self.assertTrue(decision[name])
        self.assertEqual(decision['candidate'], CANDIDATE)
        self.assertEqual(decision['outcome'], 'increment_on_internal_audit')
        self.assertFalse(decision['allow_full_training_fit_without_passed_audit'])
        self.assertFalse(decision['candidate_for_confirmation'])
        self.assertFalse(decision['independent_confirmation_authorized'])

    def test_internal_candidate_must_strictly_beat_every_fixed_control(self):
        for control in AUDIT_CONTROLS:
            records, coverage, _ = fixture()
            records[control]['mae'] = records[CANDIDATE]['mae']
            with self.subTest(control=control):
                result = audit_screen(records, coverage)
                self.assertFalse(result['audit_passed'])
                self.assertTrue(result['support_sufficient'])
                self.assertEqual(result['outcome'], 'no_increment_under_fixed_probe')
                self.assertEqual(result['candidate'], CANDIDATE)
                self.assertFalse(result['winner_selection'])

    def test_support_boundary_is_inclusive_for_each_child(self):
        for arm in CHILD_ARMS:
            records, coverage, _ = fixture()
            coverage[arm]['active_fraction'] = .25
            with self.subTest(arm=arm):
                self.assertTrue(audit_screen(records, coverage)['audit_passed'])

    def test_low_support_is_inconclusive_even_if_mae_is_better(self):
        for arm in CHILD_ARMS:
            records, coverage, groups = fixture()
            coverage[arm]['active_fraction'] = .249999999
            decision = audit_screen(records, coverage)
            with self.subTest(arm=arm):
                self.assertEqual(decision['outcome'], 'inconclusive_support')
                self.assertFalse(decision['support_sufficient'])
                self.assertFalse(decision['validation_scoring_permitted'])
                self.assertFalse(decision['full_training_fit_permitted'])
                with self.assertRaises(ValueError):
                    validation_screen(records, groups, decision)

    def test_zero_observed_and_eligible_cells_are_valid_but_inconclusive(self):
        records, coverage, _ = fixture()
        coverage[CANDIDATE] = dict(active_fraction=0., eligible_cells=0, observed_cells=0)
        self.assertEqual(audit_screen(records, coverage)['outcome'], 'inconclusive_support')

    def test_child_coverage_cannot_exceed_context_parent(self):
        records, coverage, _ = fixture()
        coverage['CONTEXT'] = dict(active_fraction=0., eligible_cells=0, observed_cells=0)
        with self.assertRaisesRegex(ValueError, 'CONTEXT parent'):
            audit_screen(records, coverage)
        coverage['CONTEXT'] = dict(active_fraction=.5-5e-13,
            eligible_cells=10, observed_cells=20)
        self.assertTrue(audit_screen(records, coverage)['audit_passed'])
        coverage['CONTEXT']['active_fraction'] = .5-2e-12
        with self.assertRaisesRegex(ValueError, 'CONTEXT parent'):
            audit_screen(records, coverage)

    def test_missing_coverage_and_malformed_coverage_are_rejected(self):
        changes = (dict(active_fraction=-.1), dict(active_fraction=1.01),
            dict(active_fraction=True), dict(active_fraction=np.bool_(True)),
            dict(active_fraction=float('nan')), dict(active_fraction=float('inf')),
            dict(eligible_cells=-1), dict(eligible_cells=True),
            dict(eligible_cells=1.5), dict(observed_cells=-1),
            dict(observed_cells=True), dict(eligible_cells=21),
            dict(eligible_cells=0, active_fraction=.1))
        for change in changes:
            records, coverage, _ = fixture()
            coverage[CANDIDATE].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                audit_screen(records, coverage)
        for arm in COVERAGE_ARMS:
            records, coverage, _ = fixture(); del coverage[arm]
            with self.subTest(missing=arm), self.assertRaises(ValueError):
                audit_screen(records, coverage)

    def test_global_metrics_reject_nonfinite_boolean_and_negative_errors(self):
        for field in ('mae', 'rmse', 'mape_fraction', 'signed_bias'):
            for invalid in (float('nan'), float('inf'), True, np.bool_(False)):
                records, coverage, _ = fixture(); records[CANDIDATE][field] = invalid
                with self.subTest(field=field, value=invalid), self.assertRaises(ValueError):
                    audit_screen(records, coverage)
        for field in ('mae', 'rmse', 'mape_fraction'):
            records, coverage, _ = fixture(); records[CANDIDATE][field] = -.01
            with self.subTest(field=field), self.assertRaises(ValueError):
                audit_screen(records, coverage)

    def test_signed_bias_can_be_negative_and_is_not_an_extra_gate(self):
        records, coverage, groups = fixture()
        records[CANDIDATE]['signed_bias'] = -99.
        groups[CANDIDATE][GROUPS[0]]['signed_bias'] = -99.
        self.assertTrue(audit_screen(records, coverage)['audit_passed'])
        self.assertTrue(validation_screen(records, groups,
            passed_audit())['trajectory_information_screen_passed'])

    def test_global_support_is_positive_integral_and_shared(self):
        for invalid in (0, -1, 100., True, 99):
            records, coverage, _ = fixture(); records[CANDIDATE]['valid_count'] = invalid
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                audit_screen(records, coverage)

    def test_exact_arms_and_complete_global_fields_are_required(self):
        for arm in ARMS:
            records, coverage, _ = fixture(); del records[arm]
            with self.subTest(missing=arm), self.assertRaises(ValueError):
                audit_screen(records, coverage)
        records, coverage, _ = fixture(); records['POSTHOC'] = record(1.)
        with self.assertRaises(ValueError): audit_screen(records, coverage)
        for field in ('valid_count', 'mae', 'rmse', 'mape_fraction', 'signed_bias'):
            records, coverage, _ = fixture(); del records[CANDIDATE][field]
            with self.subTest(missing=field), self.assertRaises(ValueError):
                audit_screen(records, coverage)

    def test_validation_pass_is_architecture_permission_never_confirmation(self):
        records, _, groups = fixture()
        result = validation_screen(records, groups, passed_audit())
        self.assertTrue(result['trajectory_information_screen_passed'])
        self.assertTrue(result['eligible_for_architecture_design'])
        self.assertEqual(result['phase'], 'completed')
        for name in ('candidate_for_confirmation', 'independent_confirmation_authorized',
                'confirmation_performed', 'statistical_significance_claimed',
                'causal_effect_claimed', 'fitting_based_on_validation',
                'winner_selection', 'validation_hyperparameter_search', 'test_evaluated'):
            self.assertFalse(result[name], name)
        self.assertTrue(result['prior_failed_screens_unchanged'])

    def test_half_percent_boundary_is_inclusive(self):
        records, _, groups = fixture()
        records[CANDIDATE]['mae'] = 9.95
        result = validation_screen(records, groups, passed_audit())
        self.assertTrue(result['trajectory_information_screen_passed'])
        self.assertTrue(all(result['checks'].values()))

    def test_just_below_half_percent_improvement_fails_each_gain_control(self):
        for control in VALIDATION_GAIN_CONTROLS:
            records, _, groups = fixture()
            records[CANDIDATE]['mae'] = 9.95
            records[control]['mae'] = 9.9999999
            with self.subTest(control=control):
                result = validation_screen(records, groups, passed_audit())
                self.assertFalse(result['checks']['relative_gain_vs_'+control+'_at_least_0_005'])
                self.assertFalse(result['eligible_for_architecture_design'])

    def test_validation_candidate_must_strictly_beat_both_capacity_controls(self):
        for control in ('MAGNITUDE16', 'DIRECTION16'):
            records, _, groups = fixture()
            records[control]['mae'] = records[CANDIDATE]['mae']
            with self.subTest(control=control):
                result = validation_screen(records, groups, passed_audit())
                self.assertFalse(result['checks']['candidate_beats_'+control])
                self.assertFalse(result['trajectory_information_screen_passed'])

    def test_each_guard_allows_equality_but_rejects_any_regression(self):
        for name in GROUPS:
            records, _, groups = fixture()
            self.assertTrue(validation_screen(records, groups,
                passed_audit())['trajectory_information_screen_passed'])
            groups[CANDIDATE][name]['mae'] += .000001
            with self.subTest(group=name):
                result = validation_screen(records, groups, passed_audit())
                self.assertFalse(result['checks'][name+'_no_worse_than_SUM'])

    def test_guards_use_sum_rather_than_other_controls(self):
        records, _, groups = fixture()
        for arm in ARMS:
            if arm not in ('SUM', CANDIDATE):
                for name in GROUPS: groups[arm][name]['mae'] = 1.
        self.assertTrue(validation_screen(records, groups,
            passed_audit())['trajectory_information_screen_passed'])

    def test_group_counts_and_metrics_are_checked_before_screening(self):
        changes = (dict(valid_count=0), dict(valid_count=-1), dict(valid_count=True),
            dict(valid_count=20.), dict(valid_count=101), dict(valid_count=21),
            dict(mae=None), dict(mae=float('nan')), dict(mae=-1.), dict(mae=True),
            dict(signed_bias=float('inf')), dict(signed_bias=True),
            dict(rmse=float('nan')), dict(mape_fraction=-.1))
        for change in changes:
            records, _, groups = fixture()
            groups[CANDIDATE][GROUPS[0]].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                validation_screen(records, groups, passed_audit())

    def test_missing_group_arm_or_group_metric_is_rejected(self):
        for mode in ('arm', 'group', 'mae', 'signed_bias'):
            records, _, groups = fixture()
            if mode == 'arm': del groups[CANDIDATE]
            elif mode == 'group': del groups[CANDIDATE][GROUPS[0]]
            else: del groups[CANDIDATE][GROUPS[0]][mode]
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                validation_screen(records, groups, passed_audit())

    def test_forged_partial_audit_and_false_permissions_are_rejected(self):
        decisions = (None, {}, dict(audit_passed=True), passed_audit())
        decisions[-1]['checks']['candidate_beats_OWN4'] = False
        for decision in decisions:
            records, _, groups = fixture()
            with self.subTest(decision=decision), self.assertRaises(ValueError):
                validation_screen(records, groups, decision)
        for field in ('audit_passed', 'support_sufficient', 'validation_scoring_permitted',
                'full_training_fit_permitted'):
            decision = passed_audit(); decision[field] = 1
            records, _, groups = fixture()
            with self.subTest(field=field), self.assertRaises(ValueError):
                validation_screen(records, groups, decision)
        decision = passed_audit(); decision['checks']['POSTHOC'] = True
        records, _, groups = fixture()
        with self.assertRaises(ValueError): validation_screen(records, groups, decision)

    def test_changed_audit_support_threshold_and_confirmation_flags_are_rejected(self):
        for field, invalid in (('min_active_fraction', .1),
                ('activity_threshold_is_operational_not_statistical', False),
                ('candidate_for_confirmation', True),
                ('independent_confirmation_authorized', True),
                ('fitting_based_on_validation', True), ('test_evaluated', True)):
            decision = passed_audit(); decision[field] = invalid
            records, _, groups = fixture()
            with self.subTest(field=field), self.assertRaises(ValueError):
                validation_screen(records, groups, decision)

    def test_no_gain_is_defined_against_a_zero_error_reference(self):
        records, _, groups = fixture()
        records['SUM']['mae'] = records[CANDIDATE]['mae'] = 0.
        result = validation_screen(records, groups, passed_audit())
        self.assertIsNone(result['relative_gains']['SUM'])
        self.assertFalse(result['trajectory_information_screen_passed'])

    def test_validation_rejects_invalid_global_metrics_too(self):
        for field, invalid in (('mae', float('nan')), ('valid_count', 99),
                ('mape_fraction', True)):
            records, _, groups = fixture(); records[CANDIDATE][field] = invalid
            with self.subTest(field=field), self.assertRaises(ValueError):
                validation_screen(records, groups, passed_audit())


if __name__ == '__main__':
    unittest.main()
