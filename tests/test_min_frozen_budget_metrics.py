import importlib.util
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1]/'experiments/chronological/min_frozen_budget_metrics.py'
SPEC = importlib.util.spec_from_file_location('min_frozen_budget_metrics', MODULE_PATH)
metrics = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(metrics)


class FrozenBudgetMetricsTests(unittest.TestCase):
    def fixture(self):
        target = np.ones((2, 2, 2), dtype=np.float32)
        off = target+np.array([[[1, 3], [4, 0]], [[0, 0], [8, 12]]], dtype=np.float32)
        on = target+np.array([[[0, 4], [2, 0]], [[0, 0], [9, 10]]], dtype=np.float32)
        valid = np.array([[[1, 1], [1, 0]], [[0, 0], [1, 1]]], dtype=bool)
        return on, off, target, valid

    def test_nonuniform_mask_keeps_pooled_macro_and_window_weighting_distinct(self):
        on, off, target, valid = self.fixture()
        result = metrics.summarize_pair(on, off, target, valid, np.array([0]))
        self.assertEqual(result['off']['valid_count_per_horizon'], [2, 3])
        self.assertAlmostEqual(result['off']['mae_pooled'], 28/5)
        self.assertAlmostEqual(result['off']['mae_macro'], 5.)
        self.assertAlmostEqual(result['off']['rmse_pooled'], math.sqrt(234/5))
        self.assertAlmostEqual(result['off']['rmse_macro'], (math.sqrt(5)+math.sqrt(224/3))/2)
        self.assertAlmostEqual(result['on']['mae_pooled'], 5.)
        self.assertAlmostEqual(result['on']['mae_macro'], 4.5)
        self.assertAlmostEqual(result['fixed_hard_windows']['off_mae'], 3.)
        self.assertAlmostEqual(result['fixed_hard_windows']['on_mae'], 2.)
        self.assertAlmostEqual(result['fixed_hard_mae_gain'], 1.)

    def test_point_gain_decomposition_and_prediction_change_use_off_minus_on(self):
        result = metrics.summarize_pair(*self.fixture(), np.array([0]))
        pair = result['paired']
        self.assertAlmostEqual(pair['mae_gain_pooled'], .6)
        self.assertAlmostEqual(pair['mae_gain_macro'], .5)
        self.assertAlmostEqual(pair['gain_positive_pooled'], 1.)
        self.assertAlmostEqual(pair['gain_negative_pooled'], .4)
        self.assertAlmostEqual(pair['cancellation_fraction_pooled'], 1-.6/1.4)
        self.assertAlmostEqual(pair['gain_positive'], 11/12)
        self.assertAlmostEqual(pair['gain_negative'], 5/12)
        self.assertAlmostEqual(pair['cancellation_fraction'], .625)
        self.assertAlmostEqual(pair['prediction_change']['absolute_mean_pooled'], 1.4)
        self.assertAlmostEqual(pair['prediction_change']['signed_mean_off_minus_on'], .6)
        self.assertEqual(pair['prediction_change']['absolute_max'], 2.)

    def test_hard_selection_excludes_incomplete_and_ties_use_original_sample_id(self):
        target = np.ones((5, 3, 1), dtype=np.float32)
        off = target+np.array([4, 3, 3, 1, 9], dtype=np.float32)[:, None, None]
        valid = np.ones(target.shape, dtype=bool)
        valid[4, 2, 0] = False
        ids = np.array([40, 30, 20, 10, 0])
        chosen = metrics.select_hard_windows(off, target, valid, ids, fraction=.5)
        np.testing.assert_array_equal(chosen, [0, 2])
        np.testing.assert_array_equal(metrics.select_hard_windows(off, target, valid, ids), [0])

    def test_fixed_set_is_not_reselected_for_changed_on_errors(self):
        target = np.ones((2, 12, 1), dtype=np.float32)
        off = target+np.array([10, 1], dtype=np.float32)[:, None, None]
        on = target+np.array([0, 100], dtype=np.float32)[:, None, None]
        valid = np.ones(target.shape, dtype=bool)
        fixed = metrics.select_hard_windows(off, target, valid, np.array([1, 2]))
        result = metrics.summarize_pair(on, off, target, valid, fixed)
        self.assertEqual(result['fixed_hard_windows']['positional_indices'], [0])
        self.assertEqual(result['fixed_hard_windows']['on_mae'], 0.)
        self.assertEqual(result['fixed_hard_windows']['mae_gain'], 10.)
        self.assertLess(result['mae_gain_pooled'], 0.)

    def test_selected_horizon_is_single_step_not_prefix_average(self):
        target = np.ones((1, 12, 1), dtype=np.float32)
        off = target+np.arange(1, 13, dtype=np.float32)[None, :, None]
        result = metrics.summarize_pair(target, off, target, np.ones(target.shape, dtype=bool), [0])
        for h in (3, 6, 12):
            self.assertEqual(result['off']['selected_horizons']['H'+str(h)]['mae'], float(h))
        self.assertEqual(result['on']['selected_horizons']['H12']['mape'], 0.)

    def test_zero_targets_count_for_mae_but_not_mape(self):
        target = np.array([[[0., 2., 4.]]])
        on = np.array([[[3., 3., 2.]]])
        off = target.copy()
        result = metrics.summarize_pair(on, off, target, np.ones(target.shape, dtype=bool), [0])
        self.assertEqual(result['on']['valid_count'], 3)
        self.assertEqual(result['on']['positive_target_count'], 2)
        self.assertEqual(result['on']['mae_pooled'], 2.)
        self.assertEqual(result['on']['mape_pooled'], .5)
        self.assertEqual(result['on']['mape_unit'], 'fraction')

    def test_all_zero_percentage_targets_are_unavailable_not_zero(self):
        target = np.zeros((1, 12, 1), dtype=np.float32)
        result = metrics.summarize_pair(target+1, target, target, np.ones(target.shape, dtype=bool), [0])
        self.assertEqual(result['on']['mae_pooled'], 1.)
        self.assertIsNone(result['on']['mape_pooled'])
        self.assertIsNone(result['on']['mape_macro'])
        self.assertEqual(result['on']['per_horizon_mape'], [None]*12)

    def test_empty_horizon_blocks_full_macro_but_preserves_pooled_and_available_steps(self):
        on, off, target, valid = self.fixture()
        valid[:, 1] = False
        result = metrics.summarize_pair(on, off, target, valid, [0])
        self.assertEqual(result['off']['status'], 'partial_horizons')
        self.assertIsNone(result['off']['mae_macro'])
        self.assertEqual(result['off']['mae_pooled'], 2.)
        self.assertIsNone(result['paired']['mae_gain_macro'])
        self.assertIsNone(result['fixed_hard_windows']['mae_gain'])
        self.assertEqual(result['off']['selected_horizons']['H3']['status'], 'unavailable')
        self.assertEqual(metrics.select_hard_windows(off, target, valid, np.array([0, 1])).size, 0)

    def test_all_invalid_and_empty_fixed_set_report_unavailable(self):
        on, off, target, valid = self.fixture()
        valid[:] = False
        result = metrics.summarize_pair(on, off, target, valid, [])
        self.assertEqual(result['on']['status'], 'unavailable')
        self.assertIsNone(result['on']['mae_pooled'])
        self.assertIsNone(result['paired']['prediction_change']['absolute_max'])
        self.assertEqual(result['fixed_hard_windows']['status'], 'unavailable')

    def test_invalid_points_can_be_nonfinite_but_valid_nonfinite_and_negative_targets_fail(self):
        on, off, target, valid = self.fixture()
        on[~valid] = np.nan
        off[~valid] = np.inf
        target[~valid] = np.nan
        result = metrics.summarize_pair(on, off, target, valid, [0])
        self.assertTrue(np.isfinite(result['on']['mae_pooled']))
        for kind in ('on', 'target', 'negative'):
            current_on, current_off, current_target, current_valid = self.fixture()
            if kind == 'on':
                current_on[0, 0, 0] = np.nan
            elif kind == 'target':
                current_target[0, 0, 0] = np.inf
            else:
                current_target[0, 0, 0] = -1.
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                metrics.summarize_pair(current_on, current_off, current_target, current_valid, [0])

    def test_rejects_bad_shapes_masks_hard_indices_and_selection_contract(self):
        on, off, target, valid = self.fixture()
        for bad_fixed in ([0, 0], [-1], [2], [.5], [[0]]):
            with self.subTest(fixed=bad_fixed), self.assertRaises(ValueError):
                metrics.summarize_pair(on, off, target, valid, bad_fixed)
        with self.assertRaises(ValueError):
            metrics.summarize_pair(on[:, :1], off, target, valid, [])
        with self.assertRaises(ValueError):
            metrics.summarize_pair(on, off, target, valid.astype(int), [])
        for fraction in (0., -1., 1.1, np.nan):
            with self.subTest(fraction=fraction), self.assertRaises(ValueError):
                metrics.select_hard_windows(off, target, valid, [1, 2], fraction)
        with self.assertRaises(ValueError):
            metrics.select_hard_windows(off, target, valid, [1, 1])

    def test_identical_predictions_have_zero_change_and_undefined_cancellation(self):
        on, _, target, valid = self.fixture()
        result = metrics.summarize_pair(on, on, target, valid, [0])
        self.assertEqual(result['mae_gain_pooled'], 0.)
        self.assertEqual(result['paired']['gain_positive'], 0.)
        self.assertEqual(result['paired']['prediction_change']['absolute_max'], 0.)
        self.assertIsNone(result['paired']['cancellation_fraction'])

    def test_report_uses_pair_metrics_and_formats_mape_as_percent(self):
        baseline = metrics.summarize_pair(*self.fixture(), [0])
        with tempfile.TemporaryDirectory() as directory:
            path = metrics.write_report(directory, baseline, [{'epoch': 1, 'metrics': baseline}],
                                        {'status': 'ENGINEERING_ONLY', 'best_epoch': 1})
            text = path.read_text(encoding='utf-8')
            self.assertIn('| 0 | 5.000000000 | 5.600000000 | 0.600000000 |', text)
            self.assertIn('500.000000000', text)
            self.assertIn('最佳轮次：1', text)


if __name__ == '__main__':
    unittest.main()
