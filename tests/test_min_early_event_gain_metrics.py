"""Hand-computed saved-prediction decomposition, grouping and overlap checks."""
import copy
import json
import unittest

import numpy as np

from experiments.chronological import min_early_event_gain_metrics as metrics


def fixture(rows=2, nodes=4):
    target = np.full((rows, 12, nodes), 10., np.float32)
    cases = dict(STATE_ON=12., EVENT_ON=11., EVENT_OFF=13., EVENT_SEMANTIC_UNK=12., EVENT_LOCATION_SHUFFLE=14.)
    predictions = {key: np.full_like(target, value) for key, value in cases.items()}
    direct, potential = np.zeros((rows, nodes), bool), np.zeros((rows, nodes), bool)
    direct[:, 0] = True
    if nodes > 1:
        potential[:, 1] = True
    regions = dict(all_nodes=np.ones((rows, nodes), bool), direct_report_nodes=direct,
        potential_only=potential, remaining_nodes=~(direct | potential), empty=np.zeros((rows, nodes), bool),
        **{'road:west': np.arange(nodes) < max(1, nodes//2)})
    samples, stations = np.arange(rows, dtype=np.int64)*100+100, np.arange(nodes, dtype=np.int64)+500
    return predictions, target, np.ones_like(target, bool), regions, samples, stations, samples[-1:]


class EarlyEventGainMetricsTests(unittest.TestCase):
    def analyze(self, args):
        return metrics.analyze_predictions(*args)

    def test_hand_computed_T_P_R_identity_gain_metrics_and_prediction_direction(self):
        result = self.analyze(fixture())
        expected = dict(T=1., P=2., R=-1., Q=1., L=3.)
        for name, value in expected.items():
            group = result['comparisons'][name]['groups']['all_nodes']
            self.assertEqual(group['mae_gain_pooled'], value)
            self.assertEqual(group['mae_gain_macro'], value)
            self.assertEqual(group['global_mae_contribution_pooled'], value)
            self.assertEqual(group['global_mae_contribution_macro'], value)
        self.assertEqual(result['identity_check']['checked_points'], 96)
        self.assertEqual(result['identity_check']['maximum_absolute_point_residual'], 0.)
        self.assertEqual(result['comparisons']['T']['groups']['all_nodes']['prediction_change']['down_fraction'], 1.)
        self.assertEqual(result['comparisons']['R']['groups']['all_nodes']['prediction_change']['up_fraction'], 1.)
        self.assertEqual(result['comparisons']['R']['groups']['all_nodes']['gain_negative_pooled'], 1.)
        self.assertAlmostEqual(result['comparisons']['T']['groups']['all_nodes']['mape_gain_pooled'], .1)
        # Positive parts are not an additive decomposition, even when signed net gains are.
        total = result['comparisons']['T']['groups']['all_nodes']['gain_positive_pooled']
        path = result['comparisons']['P']['groups']['all_nodes']['gain_positive_pooled']
        rest = result['comparisons']['R']['groups']['all_nodes']['gain_positive_pooled']
        self.assertNotEqual(total, path+rest)
        json.dumps(result, allow_nan=False)

    def test_all_disjoint_families_reconcile_and_empty_group_keeps_zero_contribution(self):
        result = self.analyze(fixture())
        definitions = result['group_definitions']
        family_names = ('flow', 'STATE_bias', 'flow_STATE_bias_report_partition')
        for key, comparison in result['comparisons'].items():
            groups, global_metric = comparison['groups'], comparison['groups']['all_nodes']
            for family in family_names:
                names = [name for name, definition in definitions.items() if definition['family'] == family]
                for field in ('global_mae_contribution_pooled', 'global_mae_contribution_macro',
                              'global_mape_contribution_pooled', 'global_mape_contribution_macro'):
                    self.assertAlmostEqual(sum(groups[name][field] for name in names), global_metric[field])
                self.assertEqual(sum(groups[name]['valid_count'] for name in names), global_metric['valid_count'])
            self.assertEqual(groups['empty']['status'], 'unavailable')
            self.assertIsNone(groups['empty']['mae_gain_macro'])
            self.assertEqual(groups['empty']['global_mae_contribution_macro'], 0.)
            self.assertEqual(groups['empty']['global_mae_contribution_pooled'], 0.)
        cross = [name for name, value in definitions.items() if value['family'] == 'flow_STATE_bias_report_partition']
        self.assertEqual(len(cross), 6*3*3)
        rows = metrics.flat_rows(result)
        self.assertEqual(len(rows), len(definitions)*5*13)
        self.assertEqual({row['horizon'] for row in rows}, set(range(13)))
        self.assertTrue(all(set(row) == set(rows[0]) for row in rows))

    def test_every_comparison_uses_STATE_bias_even_when_OFF_has_opposite_bias(self):
        args = fixture()
        args[0]['STATE_ON'][:] = 8.  # STATE under, but OFF remains 13 (over).
        result = self.analyze(args)
        for comparison in result['comparisons'].values():
            self.assertEqual(comparison['groups']['bias:under']['valid_count'], 96)
            self.assertEqual(comparison['groups']['bias:over']['valid_count'], 0)
        self.assertEqual(result['comparisons']['P']['groups']['bias:under']['mae_gain_pooled'], 2.)

    def test_horizon_macro_and_global_weighted_contribution_can_have_opposite_signs(self):
        args = list(fixture(rows=1, nodes=100))
        predictions = args[0]
        predictions['STATE_ON'][:] = 8.
        predictions['EVENT_ON'][:] = 8.
        for h in range(12):
            count, candidate = (10, 13.) if h < 6 else (90, 11.5)
            predictions['STATE_ON'][0, h, :count] = 12.
            predictions['EVENT_ON'][0, h, :count] = candidate
        predictions['EVENT_OFF'] = predictions['STATE_ON'].copy()
        result = self.analyze(args)
        group = result['comparisons']['T']['groups']['bias:over']
        self.assertEqual(group['mae_gain_macro'], -.25)
        self.assertEqual(group['mae_gain_pooled'], .35)
        self.assertAlmostEqual(group['global_mae_contribution_macro'], .175)
        self.assertEqual(group['valid_count_per_horizon'], [10]*6+[90]*6)
        self.assertEqual(group['per_horizon_mae_gain'], [-1.]*6+[.5]*6)

    def test_strict_triplets_four_overlaps_and_P_R_same_point_opposition(self):
        args = fixture(rows=1)
        for case, values in dict(STATE_ON=[12, 12, 11, 10], EVENT_OFF=[13, 10.5, 13, 10],
                                 EVENT_ON=[11, 11, 12, 10]).items():
            args[0][case][:] = np.array(values, np.float32)
        result = self.analyze(args)
        group = result['quadrants']['groups']['all_nodes']
        self.assertEqual(sum(row['valid_count'] for row in group['strict_triplets'].values()), 48)
        self.assertEqual(sum(row['valid_count'] for row in group['P_R_sign_cross'].values()), 48)
        overlaps = group['T_P_positive_nonpositive']
        self.assertEqual(overlaps['T_positive|P_nonpositive']['valid_count'], 12)
        self.assertEqual(overlaps['T_nonpositive|P_positive']['valid_count'], 12)
        self.assertEqual(overlaps['T_nonpositive|P_nonpositive']['valid_count'], 12)
        opposed = group['P_R_opposed']
        self.assertEqual(opposed['valid_count'], 36)
        self.assertEqual(opposed['within_group_fraction'], .75)
        self.assertEqual(opposed['P_R_pointwise_cancellation_sum'], 60.)
        self.assertEqual(opposed['P_R_absolute_sum'], 96.)
        self.assertEqual(opposed['P_R_pointwise_cancellation_fraction'], .625)
        for row in group['strict_triplets'].values():
            self.assertAlmostEqual(row['T_gain_sum'], row['P_gain_sum']+row['R_gain_sum'])

    def test_fixed_window_equal_weight_is_separate_from_fixed_point_weight(self):
        args = list(fixture())
        args[0]['STATE_ON'][1] = 15.
        args[2][1, :, 1:] = False
        args[6] = args[4][::-1].copy()  # Preserve this saved fixed set; do not reselect or reorder by errors.
        result = self.analyze(args)
        fixed = result['fixed_hard_windows']
        self.assertEqual(fixed['sample_indices'], [200, 100])
        self.assertEqual(fixed['complete_window_count'], 2)
        self.assertEqual(fixed['comparisons']['T']['window_equal_weight_mean_gain'], 2.5)
        self.assertEqual(fixed['comparisons']['T']['point_metrics']['mae_gain_pooled'], 1.6)

    def test_response_tolerance_changes_only_prediction_response_counts_not_gain_signs(self):
        args = fixture(rows=1)
        args[0]['EVENT_ON'][:] = 11.9995
        low = metrics.analyze_predictions(*args, response_tol=.0001)
        high = metrics.analyze_predictions(*args, response_tol=.001)
        low_group, high_group = (value['comparisons']['T']['groups']['all_nodes'] for value in (low, high))
        self.assertEqual(low_group['prediction_change']['response_fraction'], 1.)
        self.assertEqual(high_group['prediction_change']['response_fraction'], 0.)
        self.assertEqual(low_group['improved_fraction'], 1.)
        self.assertEqual(high_group['improved_fraction'], 1.)
        self.assertEqual(low_group['mae_gain_pooled'], high_group['mae_gain_pooled'])

    def test_zero_target_mape_missing_horizons_and_invalid_axes(self):
        args = list(fixture(rows=1))
        args[1][0, 0, 0] = 0.
        result = self.analyze(args)
        group = result['comparisons']['T']['groups']['all_nodes']
        self.assertEqual(group['valid_count'], 48)
        self.assertEqual(group['positive_target_count'], 47)
        self.assertIsNotNone(group['candidate']['mape_pooled'])
        args[2][0, 0] = False
        result = self.analyze(args)
        group = result['comparisons']['T']['groups']['all_nodes']
        self.assertEqual(group['status'], 'partial_horizons')
        self.assertIsNone(group['mae_gain_macro'])
        self.assertIsNone(group['global_mae_contribution_macro'])
        self.assertIsNotNone(group['mae_gain_pooled'])
        self.assertEqual(result['fixed_hard_windows']['complete_window_count'], 0)
        malformed = list(fixture())
        malformed[6] = np.array([999])
        with self.assertRaisesRegex(ValueError, 'subset'):
            self.analyze(malformed)
        malformed = list(fixture())
        malformed[3]['remaining_nodes'][:] = True
        with self.assertRaisesRegex(ValueError, 'disjoint'):
            self.analyze(malformed)
        malformed = list(fixture())
        malformed[0]['EVENT_OFF'] = malformed[0]['EVENT_OFF'][:, :, :3]
        with self.assertRaisesRegex(ValueError, 'axes'):
            self.analyze(malformed)

    def test_fractional_roundoff_identity_and_sources_are_unchanged(self):
        args = list(fixture())
        rng = np.random.default_rng(42)
        for case in args[0]:
            args[0][case] = rng.normal(10., 4., args[1].shape)
        before = copy.deepcopy(args)
        result = self.analyze(args)
        self.assertEqual(result['identity_check']['status'], 'POINTWISE_AND_GROUPED_T_EQUALS_P_PLUS_R_PASS')
        for case in args[0]:
            np.testing.assert_array_equal(args[0][case], before[0][case])
        for actual, original in zip(args[1:3], before[1:3]):
            np.testing.assert_array_equal(actual, original)
        json.dumps(result, allow_nan=False)


if __name__ == '__main__':
    unittest.main()
