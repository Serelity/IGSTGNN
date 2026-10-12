"""Metric contracts for same-checkpoint module interventions, using synthetic data."""

import json
import unittest

import numpy as np

from experiments.chronological import b3_r_module_metrics as metrics


def predictions(values, shape=(3, 2, 2)):
    return {case: np.full(shape, value, dtype=np.float64) for case, value in zip(metrics.CASES, values)}


class FactorialMetricsTests(unittest.TestCase):
    def summarize(self, values, **kwargs):
        arrays = predictions(values)
        return metrics.summarize_factorial(arrays, np.zeros((3, 2, 2)),
                                           {"all_nodes": np.ones((3, 2), dtype=bool)}, **kwargs)

    def test_complementarity_and_redundancy_have_opposite_interaction_sign(self):
        complementary = self.summarize([3., 8., 7., 10.])
        redundant = self.summarize([7., 7., 7., 10.])
        self.assertEqual(complementary["interaction"]["metrics"]["all_nodes"]["mae_interaction_macro"], 2.)
        self.assertEqual(redundant["interaction"]["metrics"]["all_nodes"]["mae_interaction_macro"], -3.)
        self.assertEqual(complementary["comparisons"]["D_given_TIID_ON"]["metrics"]["all_nodes"]["baseline_minus_candidate_mae"], 5.)
        self.assertEqual(complementary["comparisons"]["TIID_given_D_OFF"]["metrics"]["all_nodes"]["baseline_minus_candidate_mae"], 2.)
        self.assertEqual(complementary["comparisons"]["combined_vs_both_off"]["metrics"]["all_nodes"]["baseline_minus_candidate_mae"], 7.)

    def test_nonuniform_valid_counts_use_horizon_macro_and_keep_zero_targets(self):
        shape = (2, 2, 2)
        arrays = predictions([0., 0., 0., 0.], shape)
        arrays[metrics.CASES[0]][:, 0] = [[2., 2.], [8., 99.]]
        arrays[metrics.CASES[0]][:, 1] = [[3., 99.], [99., 99.]]
        valid = np.asarray([[[1, 1], [1, 0]], [[1, 0], [0, 0]]], dtype=bool)
        labels = np.zeros(shape)
        labels[~valid] = np.nan
        report = metrics.summarize_factorial(arrays, labels, {"all_nodes": np.ones((2, 2), bool)}, valid=valid)
        result = report["cases"][metrics.CASES[0]]["metrics"]["all_nodes"]
        self.assertEqual(result["valid_count_per_horizon"], [3, 1])
        self.assertEqual(result["per_horizon_mae"], [4., 3.])
        self.assertEqual(result["mae_macro"], 3.5)
        self.assertEqual(result["mae_pooled"], 3.75)
        self.assertAlmostEqual(result["rmse_macro"], (np.sqrt(24.) + 3.) / 2)
        self.assertEqual(result["window_mae"]["available_count"], 1)
        self.assertEqual(result["window_mae"]["partial_count"], 1)
        self.assertEqual(result["window_mae"]["mean"], 2.5)
        self.assertFalse(report["interpretation"]["zero_targets_automatically_masked"])
        json.dumps(report, allow_nan=False)

    def test_point_gains_reveal_complete_cancellation_despite_prediction_change(self):
        arrays = predictions([1., 1., 1., 1.], (1, 2, 2))
        arrays[metrics.CASES[0]][:] = [0., 2.]
        regions = {"all_nodes": np.ones((1, 2), bool), "left": np.asarray([[True, False]]),
                   "right": np.asarray([[False, True]])}
        report = metrics.summarize_factorial(arrays, np.zeros((1, 2, 2)), regions)
        comparison = report["comparisons"]["D_given_TIID_ON"]["metrics"]
        total = comparison["all_nodes"]
        self.assertEqual(total["baseline_minus_candidate_mae"], 0.)
        self.assertEqual(total["gain_positive"], .5)
        self.assertEqual(total["gain_negative"], .5)
        self.assertEqual(total["cancellation_fraction"], 1.)
        self.assertEqual(total["prediction_absolute_change_macro"], 1.)
        self.assertEqual(comparison["left"]["global_gain_contribution"], .5)
        self.assertEqual(comparison["right"]["global_gain_contribution"], -.5)
        self.assertEqual(sum(comparison[key]["global_valid_weight_fraction"] for key in ("left", "right")), 1.)

    def test_empty_region_is_unavailable_and_json_safe(self):
        report = metrics.summarize_factorial(predictions([1., 2., 3., 4.]), np.zeros((3, 2, 2)),
                                             {"all_nodes": np.ones((3, 2), bool), "empty": np.zeros((3, 2), bool)})
        for case in metrics.CASES:
            empty = report["cases"][case]["metrics"]["empty"]
            self.assertEqual(empty["status"], "unavailable")
            self.assertIsNone(empty["mae_macro"])
            self.assertIsNone(empty["window_mae"]["p95"])
            self.assertEqual(empty["per_horizon_mae"], [None, None])
        paired = report["comparisons"]["D_given_TIID_ON"]["metrics"]["empty"]
        self.assertIsNone(paired["baseline_minus_candidate_mae"])
        self.assertEqual(paired["global_gain_contribution"], 0.)
        self.assertIsNone(report["interaction"]["metrics"]["empty"]["mae_interaction_macro"])
        json.dumps(report, allow_nan=False)

    def test_hard_set_is_selected_once_from_full_on_and_uses_index_ties(self):
        arrays = predictions([0., 0., 0., 0.], (4, 2, 1))
        arrays[metrics.CASES[0]][:, :, 0] = np.asarray([5., 5., 1., 0.])[:, None]
        arrays[metrics.CASES[1]][:, :, 0] = np.asarray([1., 3., 40., 0.])[:, None]
        report = metrics.summarize_factorial(arrays, np.zeros((4, 2, 1)), {"all_nodes": np.ones((4, 1), bool)},
                                             sample_indices=np.asarray([20, 10, 30, 40]), fixed_hard_fraction=.25)
        hard = report["fixed_hard_windows"]
        self.assertEqual(hard["sample_indices"], [10])
        self.assertEqual(hard["row_positions"], [1])
        intervention = hard["cases"][metrics.CASES[1]]["metrics"]["all_nodes"]
        self.assertEqual(intervention["mae_macro"], 3.)
        self.assertEqual(intervention["full_on_minus_case_window_mae_mean"], 2.)
        own_tail = report["cases"][metrics.CASES[1]]["metrics"]["all_nodes"]["window_mae"]["top_5_percent"]
        self.assertEqual(own_tail["sample_indices"], [30])

    def test_partial_windows_never_enter_full_on_hard_set(self):
        arrays = predictions([1., 2., 3., 4.], (3, 2, 1))
        arrays[metrics.CASES[0]][0] = 100.
        valid = np.ones((3, 2, 1), bool)
        valid[0, 1] = False
        report = metrics.summarize_factorial(arrays, np.zeros((3, 2, 1)), {"all_nodes": np.ones((3, 1), bool)},
                                             valid=valid, fixed_hard_fraction=1.)
        self.assertEqual(report["fixed_hard_windows"]["sample_indices"], [1, 2])
        self.assertGreater(report["cases"][metrics.CASES[0]]["metrics"]["all_nodes"]["mae_macro"], 1.)

    def test_rejects_bad_axes_masks_nonfinite_and_missing_cases(self):
        arrays = predictions([1., 2., 3., 4.])
        labels = np.zeros((3, 2, 2))
        regions = {"all_nodes": np.ones((3, 2), bool)}
        bad_cases = [dict(arrays, TIID_ON_D_ON=np.ones((3, 2, 2, 1))),
                     {case: value for case, value in arrays.items() if case != metrics.CASES[0]},
                     dict(arrays, TIID_ON_D_ON=np.full(labels.shape, np.nan))]
        for bad in bad_cases:
            with self.subTest(keys=list(bad)), self.assertRaises(ValueError):
                metrics.summarize_factorial(bad, labels, regions)
        for invalid_regions in ({"all_nodes": np.ones((3, 2), int)},
                                {"all_nodes": np.zeros((3, 2), bool)},
                                {"all_nodes": np.ones((3, 1), bool)}):
            with self.assertRaises(ValueError):
                metrics.summarize_factorial(arrays, labels, invalid_regions)
        for options in ({"valid": np.ones((3, 2, 2), int)}, {"sample_indices": [0, 0, 1]},
                        {"fixed_hard_fraction": 0}, {"fixed_hard_fraction": np.nan}):
            with self.assertRaises(ValueError):
                metrics.summarize_factorial(arrays, labels, regions, **options)
        labels[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            metrics.summarize_factorial(arrays, labels, regions)

    def test_no_input_mutation_and_all_six_comparisons_exist(self):
        arrays = predictions([1., 2., 3., 4.])
        before = {key: value.copy() for key, value in arrays.items()}
        labels = np.zeros((3, 2, 2))
        valid = np.ones(labels.shape, bool)
        region = np.ones((3, 2), bool)
        report = metrics.summarize_factorial(arrays, labels, {"all_nodes": region}, valid=valid)
        self.assertEqual(set(report["comparisons"]), set(metrics.COMPARISONS))
        for case in metrics.CASES:
            np.testing.assert_array_equal(arrays[case], before[case])
        np.testing.assert_array_equal(labels, np.zeros(labels.shape))
        self.assertTrue(valid.all() and region.all())


if __name__ == "__main__":
    unittest.main()
