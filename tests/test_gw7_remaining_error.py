"""Server-only numeric contracts for the GW7 remaining-error diagnostic.

These small arrays check the analysis definitions, not model performance.  Run
this module on the server; importing it locally is outside the run contract.
"""
import unittest

import numpy as np

from experiments.graph_interaction_gw7.diagnose_remaining_error import (
    assign_bins,
    conditional_contrast,
    fit_bins,
    graph_weights,
    neighbor_features,
    temporal_features,
)


class RemainingErrorFeatureTests(unittest.TestCase):
    def test_graphs_remove_self_edges_and_preserve_direction(self):
        adjacency = np.array([[9., 2., 0.], [0., 7., 3.], [4., 0., 5.]])
        original = adjacency.copy()
        graphs = graph_weights(adjacency)
        self.assertEqual(set(graphs), {"symmetric", "row_A", "row_AT"})
        np.testing.assert_array_equal(adjacency, original)
        np.testing.assert_allclose(
            graphs["row_A"], [[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]]
        )
        np.testing.assert_allclose(
            graphs["row_AT"], [[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]]
        )
        np.testing.assert_allclose(
            graphs["symmetric"],
            [[0., 2. / 6., 4. / 6.],
             [2. / 5., 0., 3. / 5.],
             [4. / 7., 3. / 7., 0.]],
        )
        for weights in graphs.values():
            self.assertEqual(weights.dtype, np.dtype("float64"))
            np.testing.assert_array_equal(np.diag(weights), 0.)
            np.testing.assert_allclose(weights.sum(axis=1), 1.)

    def test_graph_empty_row_stays_zero_after_normalization(self):
        adjacency = np.array([[5., 2., 0.], [0., 7., 0.], [0., 0., 9.]])
        graphs = graph_weights(adjacency)
        np.testing.assert_array_equal(graphs["row_A"][1:], 0.)
        np.testing.assert_array_equal(graphs["row_AT"][[0, 2]], 0.)
        np.testing.assert_array_equal(graphs["symmetric"][2], 0.)
        self.assertTrue(all(np.isfinite(weights).all() for weights in graphs.values()))

    def test_temporal_features_use_real_zeros_and_last_three_minus_first_three(self):
        flow = np.array([[[0., 4.], [0., 4.], [0., 4.],
                          [0., 4.], [0., 4.], [6., 4.]]])
        features = temporal_features(flow, np.array([2., 1.]))
        np.testing.assert_allclose(features["level"], [[1., 4.]])
        np.testing.assert_allclose(features["volatility"], [[.6, 0.]])
        np.testing.assert_allclose(features["trend"], [[1., 0.]])
        for values in features.values():
            self.assertEqual(values.shape, (1, 2))

    def test_temporal_change_features_are_invariant_to_node_units(self):
        flow = np.array([[[0., 4.], [2., 2.], [4., 0.],
                          [3., 1.], [2., 2.], [1., 3.]]])
        scale = np.array([2., 4.])
        multiplier = np.array([7., 3.])
        original = temporal_features(flow, scale)
        converted = temporal_features(flow * multiplier, scale * multiplier)
        for name in ("volatility", "trend"):
            np.testing.assert_allclose(converted[name], original[name])
        np.testing.assert_allclose(converted["level"], original["level"] * multiplier)

    def test_edgewise_absolute_difference_prevents_neighbor_cancellation(self):
        # The neighbors move in opposite directions; their average delta is
        # zero, but each differs from this flat receiver by exactly one.
        flow = np.array([[[5., 0., 5.], [5., 1., 4.], [5., 2., 3.],
                          [5., 3., 2.], [5., 4., 1.], [5., 5., 0.]]])
        weights = np.array([[0., .5, .5], [0., 0., 0.], [0., 0., 0.]])
        features = neighbor_features(flow, np.ones(3), weights, chunk_size=1)
        self.assertAlmostEqual(float(features["disagreement"][0, 0]), 1.)
        self.assertAlmostEqual(float(features["neighbor_volatility"][0, 0]), 1.)
        self.assertAlmostEqual(float(features["discordance_ratio"][0, 0]), 1.)
        for name in ("disagreement", "neighbor_volatility", "discordance_ratio"):
            self.assertTrue(np.isnan(features[name][0, 1:]).all(), name)

    def test_neighbor_disagreement_uses_per_node_training_scales(self):
        flow = np.array([[[0., 0., 0.], [1., 2., 4.], [2., 4., 8.],
                          [3., 6., 12.], [4., 8., 16.], [5., 10., 20.]]])
        scale = np.array([1., 2., 4.])
        weights = np.array([[0., .25, .75], [.5, 0., .5], [1., 0., 0.]])
        features = neighbor_features(flow, scale, weights, chunk_size=2)
        np.testing.assert_allclose(features["disagreement"], 0.)
        np.testing.assert_allclose(features["neighbor_volatility"], 1.)
        np.testing.assert_allclose(features["discordance_ratio"], 0.)
        converted = neighbor_features(flow * scale, scale * scale, weights, chunk_size=1)
        for name in ("disagreement", "neighbor_volatility", "discordance_ratio"):
            np.testing.assert_allclose(converted[name], features[name])

    def test_flat_neighbors_have_zero_ratio_and_empty_neighbors_are_missing(self):
        flow = np.array([[[0., 5., 0.], [0., 5., 0.], [0., 5., 0.],
                          [0., 5., 0.], [0., 5., 0.], [0., 5., 0.]]])
        weights = np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., 0.]])
        features = neighbor_features(flow, np.ones(3), weights)
        for name in ("disagreement", "neighbor_volatility", "discordance_ratio"):
            np.testing.assert_array_equal(features[name][0, :2], 0.)
            self.assertTrue(np.isnan(features[name][0, 2]), name)

    def test_neighbor_features_are_chunk_size_invariant(self):
        flow = np.array([
            [[0., 3., 1.], [1., 1., 4.], [3., 2., 2.],
             [2., 4., 3.], [4., 2., 1.], [3., 1., 2.]],
            [[4., 1., 2.], [2., 2., 3.], [1., 3., 0.],
             [5., 1., 4.], [3., 2., 2.], [2., 4., 1.]],
        ])
        scale = np.array([2., 3., 4.])
        weights = np.array([[0., .5, .5], [.2, 0., .8], [1., 0., 0.]])
        one = neighbor_features(flow, scale, weights, chunk_size=1)
        all_windows = neighbor_features(flow, scale, weights, chunk_size=20)
        for name in one:
            np.testing.assert_allclose(one[name], all_windows[name], rtol=0., atol=1e-14)


class RemainingErrorStratificationTests(unittest.TestCase):
    def test_cutpoints_are_fitted_on_training_values_and_reused_unchanged(self):
        training = np.array([0., 3., 6., 9.])
        cuts = fit_bins(training)
        np.testing.assert_allclose(cuts, [3., 6.])
        validation = np.array([-100., 3., 4., 6., 100., np.nan, np.inf])
        np.testing.assert_array_equal(assign_bins(validation, cuts), [0, 0, 1, 1, 2, -1, -1])
        np.testing.assert_allclose(cuts, [3., 6.])
        self.assertEqual(assign_bins(validation, cuts).dtype, np.dtype("int64"))

    def test_tied_quantiles_collapse_without_splitting_equal_values(self):
        cuts = fit_bins(np.array([0., 0., 0., 0., np.nan]))
        np.testing.assert_array_equal(cuts, [0.])
        np.testing.assert_array_equal(assign_bins(np.array([-1., 0., 0., 1.]), cuts), [0, 0, 0, 1])
        self.assertEqual(cuts.dtype, np.dtype("float64"))

    def test_zero_heavy_training_retains_zero_versus_active_comparison(self):
        training = np.array([0.] * 90 + [1.] * 10)
        cuts = fit_bins(training)
        np.testing.assert_array_equal(cuts, [0.])
        labels = assign_bins(training, cuts)[:, None]
        scores = (1. + training)[:, None]
        summary, _ = conditional_contrast(scores, scores, np.zeros_like(scores),
            labels, [], min_count=5, high_label=len(cuts))
        self.assertEqual(summary['matched_strata'], 1)
        self.assertEqual(summary['low_units'], 90)
        self.assertEqual(summary['high_units'], 10)
        self.assertAlmostEqual(summary['weighted_mae_high_minus_low'], 1.)

    def test_continuous_control_imbalance_is_reported_within_coarse_bin(self):
        labels = np.array([0] * 5 + [2] * 5)[:, None]
        scores = np.ones((10, 1))
        continuous = np.array([1.] * 5 + [1.5] * 5)[:, None]
        summary, _ = conditional_contrast(scores, scores, np.zeros_like(scores), labels,
            [np.zeros((10, 1), dtype=np.int64)], control_values={'activity': continuous})
        self.assertAlmostEqual(summary['weighted_control_activity_low'], 1.)
        self.assertAlmostEqual(summary['weighted_control_activity_high'], 1.5)
        self.assertAlmostEqual(summary['weighted_control_activity_high_minus_low'], .5)

    def test_conditional_contrast_recovers_within_stratum_simpson_reversal(self):
        # Marginal high error is lower because high-exposure windows mostly
        # occupy the easy stratum.  Within both strata high error is +1.
        labels = np.array([0] * 5 + [2] * 20 + [0] * 20 + [2] * 5)[:, None]
        controls = np.array([0] * 25 + [1] * 25)[:, None]
        scores = np.array([1.] * 5 + [2.] * 20 + [10.] * 20 + [11.] * 5)[:, None]
        bias = np.where(labels == 2, 0., 2.)
        self.assertLess(float(scores[labels == 2].mean() - scores[labels == 0].mean()), 0.)
        summary, cells = conditional_contrast(
            scores, scores / 2., bias, labels, [controls], min_count=5, high_label=2
        )
        self.assertAlmostEqual(summary["weighted_mae_high_minus_low"], 1.)
        self.assertAlmostEqual(summary["weighted_scaled_mae_high_minus_low"], .5)
        self.assertAlmostEqual(summary["weighted_bias_high_minus_low"], -2.)
        self.assertEqual(summary["matched_strata"], 2)
        self.assertEqual(summary["matched_nodes"], 1)
        self.assertEqual(sum(bool(row["matched"]) for row in cells), 2)

    def test_conditional_contrast_cannot_match_across_different_nodes(self):
        scores = np.tile([1., 9.], (10, 1))
        labels = np.tile([0, 2], (10, 1))
        controls = np.zeros((10, 2), dtype=np.int64)
        summary, cells = conditional_contrast(
            scores, scores, np.zeros_like(scores), labels, [controls], min_count=5, high_label=2
        )
        self.assertEqual(summary["matched_strata"], 0)
        self.assertEqual(summary["matched_nodes"], 0)
        self.assertIsNone(summary["weighted_mae_high_minus_low"])
        self.assertFalse(any(row["matched"] for row in cells))

    def test_conditional_stratum_weights_use_the_smaller_extreme_count(self):
        labels = np.array([0] * 5 + [2] * 50 + [0] * 10 + [2] * 10)[:, None]
        controls = np.array([0] * 55 + [1] * 20)[:, None]
        scores = np.array([1.] * 5 + [3.] * 50 + [5.] * 10 + [4.] * 10)[:, None]
        summary, _ = conditional_contrast(
            scores, scores / 4., np.zeros_like(scores), labels, [controls], min_count=5, high_label=2
        )
        # The effects are +2 and -1, with weights 5 and 10 respectively.
        self.assertAlmostEqual(summary["weighted_mae_high_minus_low"], 0.)
        self.assertAlmostEqual(summary["weighted_scaled_mae_high_minus_low"], 0.)
        self.assertEqual(summary["matched_low_units"], 15)
        self.assertEqual(summary["matched_high_units"], 60)
        self.assertAlmostEqual(summary["matched_extreme_fraction"], 1.)

    def test_high_exposure_absence_does_not_relabel_middle_as_high(self):
        labels = np.array([0] * 6 + [1] * 6)[:, None]
        scores = np.array([1.] * 6 + [8.] * 6)[:, None]
        summary, cells = conditional_contrast(
            scores, scores, np.zeros_like(scores), labels, [], min_count=5, high_label=2
        )
        self.assertEqual(summary["matched_strata"], 0)
        self.assertEqual(summary["high_units"], 0)
        self.assertIsNone(summary["weighted_mae_high_minus_low"])
        self.assertFalse(any(row["matched"] for row in cells))

    def test_inadequate_strata_are_reported_and_do_not_enter_contrast(self):
        labels = np.array([0] * 5 + [2] * 5 + [0] * 4 + [2] * 7)[:, None]
        controls = np.array([0] * 10 + [1] * 11)[:, None]
        scores = np.array([1.] * 5 + [3.] * 5 + [100.] * 4 + [1.] * 7)[:, None]
        summary, cells = conditional_contrast(
            scores, scores, np.zeros_like(scores), labels, [controls], min_count=5, high_label=2
        )
        self.assertEqual(summary["matched_strata"], 1)
        self.assertEqual(len(cells), 2)
        self.assertAlmostEqual(summary["weighted_mae_high_minus_low"], 2.)
        self.assertEqual(sum(bool(row["matched"]) for row in cells), 1)
        self.assertLess(summary["matched_extreme_fraction"], 1.)


if __name__ == "__main__":
    unittest.main()
