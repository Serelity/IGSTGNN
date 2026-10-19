"""Server-only synthetic checks for the NumPy cache-analysis arithmetic.

These validate diagnostics, not forecasting accuracy. Do not execute locally.
"""
import json
import unittest

import numpy as np

from experiments.branch_readout_gw9 import readout_metrics as metrics


def cases(baseline, sum_prediction, branch_prediction=None):
    return dict(zip(metrics.CASES, (
        np.asarray(baseline), np.asarray(sum_prediction),
        np.asarray(sum_prediction if branch_prediction is None else branch_prediction))))


class ReadoutMetricsTests(unittest.TestCase):
    def test_float64_subtraction_precedes_absolute_error_difference(self):
        # A float32 error subtraction would round both absolute errors to 2e8.
        y = np.array([1e8], dtype=np.float32)
        prediction = cases(np.array([-1e8], dtype=np.float32),
            np.array([-99999992.], dtype=np.float32))
        result = metrics.analyze_group(prediction, y, np.ones(1, dtype=bool))
        self.assertEqual(result['comparisons']['SUM_vs_R']['mae_gain'], 8.)

    def test_balanced_help_and_harm_cancel_in_pooled_mae(self):
        y = np.array([1., 1.], dtype=np.float32)
        prediction = cases(np.array([2., 2.], dtype=np.float32),
            np.array([1.5, 2.5], dtype=np.float32))
        result = metrics.analyze_group(prediction, y, np.ones(2, dtype=bool))
        pair = result['comparisons']['SUM_vs_R']
        self.assertEqual(pair['mae_gain'], 0.)
        self.assertEqual(pair['gross_help'], .25)
        self.assertEqual(pair['gross_harm'], .25)
        self.assertEqual(pair['cancellation_ratio'], 1.)
        self.assertEqual(pair['improved_fraction'], .5)
        self.assertEqual(pair['worsened_fraction'], .5)

    def test_groups_distinguish_cancellation_from_identical_predictions(self):
        y = np.ones(4)
        predictions = cases([2., 2., 2., 2.], [2., 2., 2., 2.], [1., 3., 1., 3.])
        valid = np.ones(4, dtype=bool)
        global_result = metrics.analyze_group(predictions, y, valid)
        self.assertEqual(global_result['comparisons']['BRANCH_vs_SUM']['mae_gain'], 0.)
        self.assertEqual(global_result['comparisons']['BRANCH_vs_SUM']['cancellation_ratio'], 1.)
        tied = global_result['comparisons']['SUM_vs_R']
        self.assertIsNone(tied['cancellation_ratio'])
        self.assertEqual(tied['tied_fraction'], 1.)
        helpful = metrics.analyze_group(predictions, y, valid,
            mask=np.array([True, False, True, False]), total_valid_count=4)
        harmful = metrics.analyze_group(predictions, y, valid,
            mask=np.array([False, True, False, True]), total_valid_count=4)
        self.assertEqual(helpful['comparisons']['BRANCH_vs_SUM']['mae_gain'], 1.)
        self.assertEqual(helpful['comparisons']['BRANCH_vs_SUM']['global_mae_contribution'], .5)
        self.assertEqual(harmful['comparisons']['BRANCH_vs_SUM']['global_mae_contribution'], -.5)

    def test_mse_decomposition_distinguishes_translation_from_centered_correction(self):
        y = np.zeros(2)
        valid = np.ones(2, dtype=bool)
        # Baseline error [1, 3], constant correction [-1, -1]: improvement is
        # entirely explained by the actual mean correction interacting with bias.
        translated = metrics.analyze_group(cases([1., 3.], [0., 2.]), y, valid)
        translation = translated['comparisons']['SUM_vs_R']['mse_decomposition']
        self.assertEqual(translation['before_mse'], 5.)
        self.assertEqual(translation['after_mse'], 2.)
        self.assertEqual(translation['mse_gain'], 3.)
        self.assertEqual(translation['mean_only_mse_change'], -3.)
        self.assertEqual(translation['centered_mse_change'], 0.)
        # Correction [1, -1] has zero mean but removes all centered variation.
        centered = metrics.analyze_group(cases([1., 3.], [2., 2.]), y, valid)
        covariance = centered['comparisons']['SUM_vs_R']['mse_decomposition']
        self.assertEqual(covariance['before_signed_bias'], 2.)
        self.assertEqual(covariance['mean_correction'], 0.)
        self.assertEqual(covariance['mean_only_mse_change'], 0.)
        self.assertEqual(covariance['centered_mse_change'], -1.)
        self.assertEqual(covariance['total_mse_change'], -1.)

    def test_mse_decomposition_empty_and_paired_identity(self):
        y = np.array([1., 7., 20., 314.])
        predictions = cases([4., 5., 18., 320.], [2., 8., 21., 310.], [3., 6., 23., 315.])
        result = metrics.analyze_group(predictions, y, np.ones(4, dtype=bool))
        for pair in result['comparisons'].values():
            decomposition = pair['mse_decomposition']
            self.assertAlmostEqual(decomposition['mean_only_mse_change']
                + decomposition['centered_mse_change'], decomposition['total_mse_change'])
            self.assertAlmostEqual(decomposition['total_mse_change'],
                decomposition['after_mse'] - decomposition['before_mse'])
            self.assertAlmostEqual(decomposition['mse_gain'], -decomposition['total_mse_change'])
        empty = metrics.analyze_group(predictions, y, np.zeros(4, dtype=bool))
        for pair in empty['comparisons'].values():
            self.assertTrue(all(value is None for value in pair['mse_decomposition'].values()))
        json.dumps(empty, allow_nan=False)

    def test_subset_count_and_global_contribution_use_separate_denominators(self):
        y = np.full((2, 2, 2), 10.)
        baseline = y + 4.
        improved = y + 2.
        valid = np.ones(y.shape, dtype=bool)
        valid[0, 0, 0] = False
        predictions = cases(baseline, improved)
        totals = metrics.analyze_group(predictions, y, valid)
        contributions = []
        for horizon in range(2):
            mask = np.zeros((1, 2, 1), dtype=bool)
            mask[:, horizon, :] = True
            group = metrics.analyze_group(predictions, y, valid, mask, total_valid_count=7)
            self.assertEqual(group['cases']['SUM_CONTROL']['mae'], 2.)
            contributions.append(group['comparisons']['SUM_vs_R']['global_mae_contribution'])
        self.assertAlmostEqual(sum(contributions),
            totals['comparisons']['SUM_vs_R']['mae_gain'])
        self.assertEqual(totals['cases']['FROZEN_R']['valid_count'], 7)

    def test_relative_gain_is_undefined_for_perfect_before(self):
        result = metrics.analyze_group(cases([1.], [2.]), np.array([1.]), np.array([True]))
        self.assertIsNone(result['comparisons']['SUM_vs_R']['relative_gain'])
        self.assertEqual(result['comparisons']['SUM_vs_R']['mae_gain'], -1.)

    def test_correction_statistics_and_correlations(self):
        y = np.array([10., 10., 10.])
        result = metrics.analyze_group(cases([9., 8., 7.], [10., 10., 10.], [11., 12., 13.]),
            y, np.ones(3, dtype=bool))
        correction = result['corrections']
        self.assertAlmostEqual(correction['common_correction_correlation'], 1.)
        self.assertAlmostEqual(correction['SUM_minus_R']['correlation_with_before_signed_error'], -1.)
        self.assertAlmostEqual(correction['branch_delta_to_sum_rms_ratio'], 1.)
        self.assertEqual(correction['SUM_minus_R']['mean'], 2.)
        self.assertAlmostEqual(correction['SUM_minus_R']['rms'], np.sqrt(14. / 3.))

    def test_constant_correlation_and_zero_energy_are_undefined(self):
        result = metrics.analyze_group(cases([1., 2.], [2., 3.], [3., 4.]),
            np.ones(2), np.ones(2, dtype=bool))
        self.assertIsNone(result['corrections']['common_correction_correlation'])
        decomposition = metrics.decompose_intercept(np.zeros(3))
        self.assertEqual(decomposition['mean'], 0.)
        self.assertIsNone(decomposition['global_constant_energy_share'])

    def test_empty_group_is_json_safe_without_nan(self):
        result = metrics.analyze_group(cases([1.], [2.]), np.array([1.]),
            np.array([False]), total_valid_count=7)
        self.assertEqual(result['cases']['SUM_CONTROL']['count'], 0)
        self.assertIsNone(result['cases']['SUM_CONTROL']['mae'])
        self.assertIsNone(result['comparisons']['SUM_vs_R']['gross_help'])
        self.assertIsNone(result['corrections']['branch_delta_to_sum_rms_ratio'])
        json.dumps(result, allow_nan=False)

    def test_mape_explicitly_reports_zero_truth_exclusion(self):
        result = metrics.analyze_group(cases([2., 3., 3.], [1., 1., 1.]),
            np.array([0., 2., -2.]), np.ones(3, dtype=bool))
        record = result['cases']['SUM_CONTROL']
        self.assertEqual(record['valid_count'], 3)
        self.assertEqual(record['mape_valid_count'], 2)
        self.assertEqual(record['mape_fraction'], 1.)

    def test_invalid_masks_shapes_and_denominators_fail(self):
        y = np.ones(2)
        valid = np.ones(2, dtype=bool)
        predictions = cases(y, y)
        for invalid_valid in (np.ones(2, dtype=int), np.ones(3, dtype=bool)):
            with self.assertRaises(ValueError):
                metrics.analyze_group(predictions, y, invalid_valid)
        for total in (True, 1, 2.5, -1):
            with self.assertRaises(ValueError):
                metrics.analyze_group(predictions, y, valid, total_valid_count=total)
        with self.assertRaises(ValueError):
            metrics.analyze_group(cases([1.], [1.]), y, valid)
        with self.assertRaises(ValueError):
            metrics.analyze_group(predictions, y, valid, mask=np.ones(2, dtype=int))

    def test_nonfinite_invalid_points_are_ignored_but_selected_points_fail(self):
        y = np.array([1., np.nan])
        predictions = cases([2., np.nan], [1., np.inf])
        result = metrics.analyze_group(predictions, y, np.array([True, False]))
        self.assertEqual(result['cases']['SUM_CONTROL']['mae'], 0.)
        with self.assertRaises(ValueError):
            metrics.analyze_group(predictions, y, np.ones(2, dtype=bool))

    def test_global_and_horizon_constant_energy_are_not_additive_twice(self):
        values = np.array([[[1., 3.], [3., 5.]]])
        result = metrics.correction_decomposition(values, np.ones(values.shape, dtype=bool))
        self.assertEqual(result['count'], 4)
        self.assertEqual(result['mean'], 3.)
        self.assertEqual([row['mean'] for row in result['horizons']], [2., 4.])
        self.assertAlmostEqual(result['global_constant_energy_share'], 36. / 44.)
        self.assertAlmostEqual(result['horizon_constant_energy_share'], 40. / 44.)
        self.assertAlmostEqual(result['additional_horizon_energy_share'], 4. / 44.)
        self.assertAlmostEqual(result['within_horizon_centered_energy_share'], 4. / 44.)
        self.assertAlmostEqual(result['global_constant_energy_share']
            + result['additional_horizon_energy_share']
            + result['within_horizon_centered_energy_share'], 1.)

    def test_horizon_decomposition_weights_valid_count_and_handles_empty_horizon(self):
        values = np.array([[[2., 2.], [4., 100.], [10., 20.]]])
        valid = np.array([[[True, True], [True, False], [False, False]]])
        result = metrics.correction_decomposition(values, valid)
        self.assertAlmostEqual(result['mean'], 8. / 3.)
        self.assertEqual(result['horizon_constant_energy_share'], 1.)
        self.assertEqual(result['within_horizon_centered_energy_share'], 0.)
        self.assertEqual(result['horizons'][2]['count'], 0)
        self.assertIsNone(result['horizons'][2]['mean'])

    def test_empty_decomposition_and_wrong_axes(self):
        values = np.zeros((1, 2, 1))
        result = metrics.correction_decomposition(values, np.zeros_like(values, dtype=bool))
        self.assertEqual(result['count'], 0)
        self.assertIsNone(result['mean'])
        self.assertIsNone(result['horizon_constant_energy_share'])
        json.dumps(result, allow_nan=False)
        with self.assertRaises(ValueError):
            metrics.correction_decomposition(np.ones(2), np.ones(2, dtype=bool))


if __name__ == '__main__':
    unittest.main()
