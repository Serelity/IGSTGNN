import ast
import copy
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np

from experiments.chronological import min_forecast_fusion_gain_metrics as old
from experiments.chronological import min_forecast_fusion_relative_metrics as metrics


def fixture(nodes=8):
    target = np.full((2, 12, nodes), 10., dtype=np.float64)
    valid = np.ones(target.shape, bool)
    predictions = {case: np.full(target.shape, 9., dtype=np.float64) for case in metrics.BASE_CASES}
    predictions['EVENT_FUSION_OFF'][:] = 8.
    predictions['SOURCE_MIN'][:] = 12.
    predictions['EVENT_ON'][:] = 11.
    predictions['EVENT_UNIFORM'][:] = 7.
    predictions['EVENT_RELATIVE_EXACT'][:] = 10.
    predictions['EVENT_RELATIVE_BOUNDED'][:] = 9.5
    direct = np.zeros((2, nodes), bool)
    direct[:, :max(nodes//3, 1)] = True
    potential = np.zeros((2, nodes), bool)
    potential[:, max(nodes//3, 1):max(2*nodes//3, 2)] = True
    regions = dict(all_nodes=np.ones(nodes, bool), direct_report_nodes=direct,
        potential_only=potential, remaining_nodes=~(direct | potential), **{'road:test': np.ones(nodes, bool)})
    return predictions, target, valid, regions, np.array([10, 20]), np.arange(nodes), np.array([20])


class RelativeMetricsTests(unittest.TestCase):
    def test_hand_calculated_cases_comparison_factorial_and_sign(self):
        result = metrics.summarize(*fixture())
        self.assertEqual(result['status'], 'MIN_FORECAST_FUSION_RELATIVE_METRICS_PASS')
        self.assertEqual(len(result['cases']), 10)
        self.assertEqual(result['cases']['EVENT_FUSION_OFF']['mae_pooled'], 2.)
        self.assertEqual(result['cases']['EVENT_RELATIVE_EXACT']['rmse_macro'], 0.)
        self.assertAlmostEqual(result['cases']['EVENT_RELATIVE_BOUNDED']['mape_pooled'], .05)
        gain = result['comparisons']['EXACT_vs_FUSION_OFF']['groups']['all_nodes']
        self.assertEqual(gain['mae_gain_macro'], 2.)
        self.assertAlmostEqual(gain['mape_gain_pooled'], .2)
        self.assertEqual(gain['prediction_change']['signed_mean_candidate_minus_control_pooled'], 2.)
        factorial = result['factorial']
        # |7-10| + |10-10| - |11-10| - |8-10| = 0.
        self.assertEqual(factorial['groups']['all_nodes']['interaction_mae_pooled'], 0.)
        args = fixture()
        args[0]['EVENT_ON'][:] = 10.
        args[0]['EVENT_RELATIVE_EXACT'][:] = 7.
        result = metrics.summarize(*args)
        # 3+3-0-2 = +4; joint improves more than individual reductions.
        self.assertEqual(result['factorial']['groups']['all_nodes']['interaction_mae_macro'], 4.)
        self.assertTrue(result['factorial']['bounded_excluded'])
        self.assertFalse(result['definitions']['training_and_incident_causal_attribution'])

    def test_directions_use_own_control_and_do_not_use_source_bias(self):
        args = fixture()
        args[0]['EVENT_FUSION_OFF'][:] = [8., 8., 8., 8., 8., 8., 8., 10.]
        args[0]['EVENT_RELATIVE_EXACT'][:] = [8., 9., 10., 11., 12., 13., 7., 11.]
        args[0]['EVENT_ON'][:] = 14.
        result = metrics.summarize(*args, response_tol=100.)
        direction = result['directions']['EXACT_vs_FUSION_OFF']
        self.assertEqual(direction['direction_baseline'], 'EVENT_FUSION_OFF')
        for name in metrics.DIRECTIONS:
            self.assertEqual(direction['groups']['all_nodes']['directions'][name]['valid_count'], 24)
        current = result['comparisons']['EXACT_vs_FUSION_OFF']['groups']
        self.assertEqual(current['bias:under']['valid_count'], 168)
        self.assertEqual(current['source_bias:over']['valid_count'], 192)
        self.assertEqual(current['all_nodes']['prediction_change']['response_fraction'], 0.)
        # A different control has a different direction classification.
        on_direction = result['directions']['EXACT_vs_ON']['groups']['all_nodes']['directions']
        self.assertEqual(on_direction['toward_without_crossing']['valid_count'], 96)
        self.assertEqual(on_direction['overshoot_equal_error']['valid_count'], 0)

    def test_group_counts_and_weighted_contributions_conserve_with_all_flow_bins(self):
        args = fixture(nodes=6)
        args[1][:] = [0., 8., 30., 70., 150., 300.]
        args[0].update({case: args[1]+(i-3)*.125 for i, case in enumerate(metrics.BASE_CASES)})
        args[2][0, 0, 1] = False
        result = metrics.summarize(*args)
        groups = result['comparisons']['EXACT_vs_FUSION_OFF']['groups']
        for family in ('flow', 'current_MIN_bias', 'source_MIN_bias', 'flow_current_MIN_bias_report_partition'):
            selected = [name for name, definition in result['group_definitions'].items() if definition['family'] == family]
            self.assertEqual(sum(groups[name]['valid_count'] for name in selected), groups['all_nodes']['valid_count'])
            self.assertAlmostEqual(sum(groups[name]['global_mae_contribution_pooled'] for name in selected), groups['all_nodes']['mae_gain_pooled'])
            self.assertAlmostEqual(sum(groups[name]['global_mape_contribution_pooled'] for name in selected), groups['all_nodes']['mape_gain_pooled'])
        cross = [name for name in groups if name.startswith('cross:')]
        self.assertEqual(len(cross), 54)
        self.assertEqual(groups['flow:[0,5)']['positive_target_count'], 0)
        self.assertIsNone(groups['flow:[0,5)']['mape_gain_pooled'])
        self.assertEqual(groups['flow:[0,5)']['global_mape_contribution_pooled'], 0.)

    def test_empty_groups_missing_horizons_and_saved_fixed_ids_are_not_reselected(self):
        args = fixture()
        args[2][1, 4, :] = False
        args[3]['empty'] = np.zeros(args[1].shape[::2], bool)
        args[3]['row_20'] = np.array([[False]*8, [True]*8])
        result = metrics.summarize(*args)
        groups = result['comparisons']['BOUNDED_vs_ON']['groups']
        self.assertEqual(groups['empty']['status'], 'unavailable')
        self.assertIsNone(groups['empty']['mae_gain_pooled'])
        self.assertEqual(groups['empty']['global_mae_contribution_pooled'], 0.)
        self.assertEqual(groups['row_20']['status'], 'partial_horizons')
        self.assertIsNone(groups['row_20']['mae_gain_macro'])
        self.assertIsNone(result['case_groups']['EVENT_RELATIVE_EXACT']['row_20']['per_horizon_mae'][4])
        fixed = result['fixed_hard_windows']
        self.assertEqual(fixed['sample_indices'], [20])
        self.assertEqual(fixed['complete_window_count'], 0)
        self.assertIsNone(fixed['comparisons']['EXACT_vs_ON']['window_equal_weight_mean_gain'])
        self.assertIsNone(fixed['factorial_interaction_window_equal_weight'])

    def test_new_cases_axes_masks_and_finiteness_are_checked(self):
        mutations = [
            lambda args: args[0].pop('EVENT_RELATIVE_EXACT'),
            lambda args: args[0].__setitem__('EVENT_RELATIVE_BOUNDED', np.zeros((2, 12, 7))),
            lambda args: args[0]['EVENT_RELATIVE_EXACT'].__setitem__((0, 0, 0), np.nan),
            lambda args: args[3]['remaining_nodes'].__setitem__((0, 0), True),
            lambda args: args[4].__setitem__(1, 10),
            lambda args: args[6].__setitem__(0, 999),
        ]
        for mutation in mutations:
            args = fixture()
            mutation(args)
            with self.assertRaises(ValueError):
                metrics.summarize(*args)

    def test_no_mutation_flat_rows_and_report_are_standalone(self):
        args = fixture()
        saved = copy.deepcopy(args)
        old_cases, old_comparisons = old.BASE_CASES, copy.deepcopy(old.COMPARISONS)
        result = metrics.summarize(*args)
        for left, right in zip(args, saved):
            if isinstance(left, dict):
                for name in left:
                    np.testing.assert_array_equal(left[name], right[name])
            else:
                np.testing.assert_array_equal(left, right)
        self.assertEqual(old.BASE_CASES, old_cases)
        self.assertEqual(old.COMPARISONS, old_comparisons)
        rows = metrics.flat_rows(result)
        self.assertTrue(rows)
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path = metrics.write_report(directory, result, {'engineering_only': True})
            content = path.read_text(encoding='utf-8')
            self.assertIn('原 IGSTGNN', content)
            self.assertIn('EXACT_vs_FUSION_OFF', content)
            self.assertIn('error(U)+error(R)', content)
            self.assertIn('事故内容增益', content)
        tree = ast.parse(Path(metrics.__file__).read_text(encoding='utf-8'))
        imports = [node.module or '' for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        imports += [name.name for node in ast.walk(tree) if isinstance(node, ast.Import) for name in node.names]
        self.assertFalse(any('torch' in name or 'train_' in name or '.models' in name for name in imports))

    def test_large_fractional_partition_uses_stable_pairwise_sums(self):
        rng = np.random.default_rng(20261015)
        args = fixture(nodes=2049)
        args[1][:] = rng.uniform(.0001, 100., args[1].shape)
        for value in args[0].values():
            value[:] = rng.uniform(.01, 110., value.shape)
        result = metrics.summarize(*args)
        self.assertEqual(result['factorial']['identity_status'], 'POINTWISE_AND_GROUPED_FACTORIAL_IDENTITY_PASS')
        self.assertTrue(all(row == 'PASS' for row in result['reconciliation']['counts'].values()))

    def test_comparison_validity_drift_is_rejected(self):
        original = metrics._accumulate
        calls = [0]
        def changed_counts(acc, h, selections, sums, counts, absolute_delta):
            original(acc, h, selections, sums, counts, absolute_delta)
            calls[0] += 1
            if calls[0] == 2:
                acc['count'][0, h] += 1
        with mock.patch.object(metrics, '_accumulate', changed_counts):
            with self.assertRaisesRegex(ValueError, 'identical saved validity: EXACT_vs_ON'):
                metrics.summarize(*fixture())

    def test_release_sized_ten_cases_six_comparisons_and_factorial_reconcile(self):
        from tests.test_min_forecast_fusion_summation import _fixture
        n = 338686
        args = _fixture(837, 496, n)
        shape = args[1].shape
        args[0]['EVENT_RELATIVE_EXACT'] = np.broadcast_to(np.array(10.9998), shape)
        args[0]['EVENT_RELATIVE_BOUNDED'] = np.broadcast_to(np.array(10.9999), shape)
        result = metrics.summarize(*args)
        self.assertEqual(set(result['comparisons']), set(metrics.COMPARISONS))
        self.assertEqual(set(result['cases']), set(metrics.BASE_CASES))
        for name, comparison in result['comparisons'].items():
            with self.subTest(comparison=name):
                row = comparison['groups']['all_nodes']
                self.assertEqual(row['valid_count'], n)
                self.assertEqual(row['valid_count_per_horizon'], [n]+[0]*11)
                self.assertIsNone(row['mae_gain_macro'])
                self.assertIsNone(row['rmse_gain_macro'])
                self.assertIsNone(row['mape_gain_macro'])
                for quantity in result['reconciliation']['comparisons'][name].values():
                    for family in quantity.values():
                        self.assertEqual(family['status'], 'PASS')
        for name, row in result['cases'].items():
            with self.subTest(case=name):
                self.assertEqual(row['valid_count'], n)
                self.assertEqual(row['valid_count_per_horizon'], [n]+[0]*11)
                self.assertIsNone(row['mae_macro'])
                self.assertIsNone(row['rmse_macro'])
                self.assertIsNone(row['mape_macro'])
                for quantity in result['reconciliation']['cases'][name].values():
                    for family in quantity.values():
                        self.assertEqual(family['status'], 'PASS')
        for status in result['reconciliation']['counts'].values():
            self.assertEqual(status, 'PASS')
        for family in result['reconciliation']['interaction'].values():
            self.assertEqual(family['status'], 'PASS')
        factorial = result['factorial']
        self.assertEqual(factorial['identity_status'], 'POINTWISE_AND_GROUPED_FACTORIAL_IDENTITY_PASS')
        self.assertIsNone(factorial['groups']['all_nodes']['interaction_mae_macro'])
        self.assertIsNone(factorial['groups']['all_nodes']['interaction_mape_macro'])
        exact = result['cases']['EVENT_RELATIVE_EXACT']
        self.assertAlmostEqual(exact['mae_pooled'], abs(10.9998-10.), places=13)
        self.assertAlmostEqual(exact['rmse_pooled'], abs(10.9998-10.), places=13)
        self.assertAlmostEqual(exact['mape_pooled'], abs(10.9998-10.)/10., places=14)


if __name__ == '__main__':
    unittest.main()
