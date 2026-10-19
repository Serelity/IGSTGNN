"""Server-only numeric contracts for the GW7 trend-response diagnostic.

The tests use small fabricated arrays and import no model or training code.
Run them on the server, together with the read-only cached-data diagnostic.
"""
import unittest

import numpy as np

from experiments.graph_interaction_gw7.diagnose_trend_response import (
    caliper_pairs,
    classify_trend,
    pair_statistics,
    summarize,
)


class TrendResponseMatchingTests(unittest.TestCase):
    def test_augmenting_path_recovers_maximum_cardinality(self):
        # Greedily assigning left 0 to its closest tied right 2 would strand
        # left 1.  Reassigning that first pair allows both left units to match.
        features = np.array([[.5], [0.], [0.], [1.]])
        pairs = caliper_pairs(features, np.array([0, 1]),
                              np.array([2, 3]), np.array([.6]))
        self.assertEqual(pairs.dtype, np.dtype("int64"))
        self.assertEqual(pairs.shape, (2, 2))
        self.assertEqual(set(map(tuple, pairs.tolist())), {(0, 3), (1, 2)})

    def test_all_continuous_controls_must_satisfy_inclusive_calipers(self):
        features = np.array([[0., 0.], [.5, .5], [.2, .6]])
        pairs = caliper_pairs(features, np.array([0]),
                              np.array([2, 1]), np.array([.5, .5]))
        np.testing.assert_array_equal(pairs, [[0, 1]])

    def test_local_preference_uses_sum_of_caliper_normalized_distances(self):
        # Raw Euclidean distance favors right 1, whereas matching controls in
        # their own caliper units favors right 2.
        features = np.array([[0., 0.], [.09, 1.], [0., 5.]])
        pairs = caliper_pairs(features, np.array([0]),
                              np.array([1, 2]), np.array([.1, 10.]))
        np.testing.assert_array_equal(pairs, [[0, 2]])

    def test_ties_are_deterministic_and_neither_side_is_reused(self):
        features = np.zeros((7, 2))
        left, right = np.array([2, 0, 1]), np.array([6, 4, 5, 3])
        first = caliper_pairs(features, left, right, np.ones(2))
        for _ in range(4):
            np.testing.assert_array_equal(
                caliper_pairs(features, left, right, np.ones(2)), first)
        self.assertEqual(len(first), 3)
        self.assertEqual(len(np.unique(first[:, 0])), 3)
        self.assertEqual(len(np.unique(first[:, 1])), 3)
        self.assertTrue(set(first[:, 0]).issubset(set(left)))
        self.assertTrue(set(first[:, 1]).issubset(set(right)))

    def test_no_support_and_empty_sides_return_typed_empty_pairs(self):
        features = np.array([[0.], [5.]])
        for left, right in (([0], [1]), ([], [1]), ([0], []), ([], [])):
            with self.subTest(left=left, right=right):
                pairs = caliper_pairs(features, np.array(left, dtype=np.int64),
                                      np.array(right, dtype=np.int64), np.array([1.]))
                self.assertEqual(pairs.shape, (0, 2))
                self.assertEqual(pairs.dtype, np.dtype("int64"))

    def test_invalid_index_sets_are_rejected(self):
        features = np.arange(4., dtype=np.float64)[:, None]
        for left, right in (([0, 0], [2]), ([0], [2, 2]),
                            ([0, 1], [1, 2]), ([-1], [2]), ([0], [4])):
            with self.subTest(left=left, right=right):
                with self.assertRaises(ValueError):
                    caliper_pairs(features, np.array(left), np.array(right),
                                  np.array([1.]))

    def test_invalid_feature_and_caliper_contracts_are_rejected(self):
        valid_features = np.array([[0., 1.], [1., 2.]])
        cases = [
            (np.array([0., 1.]), np.array([1.])),
            (np.array([[0., np.nan], [1., 2.]]), np.ones(2)),
            (np.array([[0., 1.], [np.inf, 2.]]), np.ones(2)),
            (valid_features, np.array([1.])),
            (valid_features, np.array([1., 0.])),
            (valid_features, np.array([1., -1.])),
            (valid_features, np.array([1., np.nan])),
            (valid_features, np.array([1., np.inf])),
        ]
        for features, calipers in cases:
            with self.subTest(features=features, calipers=calipers):
                with self.assertRaises(ValueError):
                    caliper_pairs(features, np.array([0]), np.array([1]), calipers)

    def test_matching_does_not_modify_controls_or_group_indices(self):
        features = np.array([[0., 0.], [.3, .2], [.1, .4], [.2, .1]])
        left, right, calipers = np.array([0, 1]), np.array([2, 3]), np.ones(2)
        inputs = [features, left, right, calipers]
        originals = [value.copy() for value in inputs]
        caliper_pairs(*inputs)
        for actual, original in zip(inputs, originals):
            np.testing.assert_array_equal(actual, original)


class TrendResponseDirectionTests(unittest.TestCase):
    def test_deadband_boundaries_are_flat_and_classification_keeps_shape(self):
        trend = np.array([[-.0500001, -.05, -0.], [.0, .05, .0500001]])
        actual = classify_trend(trend)
        np.testing.assert_array_equal(actual, [[-1, 0, 0], [0, 0, 1]])
        self.assertEqual(actual.dtype, np.dtype("int8"))
        self.assertEqual(actual.shape, trend.shape)

    def test_explicit_deadband_changes_labels_without_changing_values(self):
        trend = np.array([-.2, -.1, 0., .1, .2])
        original = trend.copy()
        np.testing.assert_array_equal(classify_trend(trend, deadband=.1),
                                      [-1, 0, 0, 0, 1])
        np.testing.assert_array_equal(trend, original)


class TrendResponsePairStatisticsTests(unittest.TestCase):
    def test_common_horizons_are_averaged_within_pair_then_pairs_equal_weighted(self):
        left = np.array([[1., 3.], [10., 1000.]])
        right = np.array([[2., 4.], [4., -1000.]])
        valid = np.array([[True, True], [True, False]])
        stats = pair_statistics(left, right, valid, np.array([2., 5.]))
        self.assertEqual(stats["pairs_proposed"], 2)
        self.assertEqual(stats["pairs_scored"], 2)
        self.assertEqual(stats["common_valid_points"], 3)
        self.assertAlmostEqual(stats["left_mae"], 6.)
        self.assertAlmostEqual(stats["right_mae"], 3.5)
        self.assertAlmostEqual(stats["mae_left_minus_right"], 2.5)
        self.assertAlmostEqual(stats["left_bias"], 6.)
        self.assertAlmostEqual(stats["right_bias"], 3.5)
        self.assertAlmostEqual(stats["scaled_left_mae"], 1.5)
        self.assertAlmostEqual(stats["scaled_right_mae"], 1.15)
        self.assertAlmostEqual(stats["scaled_mae_left_minus_right"], .35)

    def test_actual_signed_bias_is_not_replaced_by_mae_or_between_group_difference(self):
        left = np.array([[-6., 2.], [2., 0.]])
        right = np.array([[4., -2.], [-8., 2.]])
        stats = pair_statistics(left, right, np.ones((2, 2), dtype=bool), np.ones(2))
        self.assertAlmostEqual(stats["left_mae"], 2.5)
        self.assertAlmostEqual(stats["right_mae"], 4.)
        self.assertAlmostEqual(stats["left_bias"], -.5)
        self.assertAlmostEqual(stats["right_bias"], -1.)
        self.assertAlmostEqual(stats["mae_left_minus_right"], -1.5)

    def test_directional_alignment_uses_up_and_down_biases_and_strict_pair_signs(self):
        left = np.array([[-4.], [2.], [0.]])
        right = np.array([[4.], [-2.], [3.]])
        stats = pair_statistics(left, right, np.ones((3, 1), dtype=bool),
                                np.ones(3), directional=True)
        self.assertAlmostEqual(stats["left_bias"], -2. / 3.)
        self.assertAlmostEqual(stats["right_bias"], 5. / 3.)
        self.assertAlmostEqual(stats["aligned_error"], 7. / 6.)
        # A flat up-group error does not satisfy the strict underreaction test.
        self.assertAlmostEqual(stats["underreaction_pair_fraction"], 1. / 3.)

    def test_masked_nonfinite_values_and_unscorable_pairs_are_excluded(self):
        left = np.array([[-2., np.nan], [np.nan, np.inf], [100., 4.]])
        right = np.array([[3., np.inf], [np.inf, np.nan], [-100., -6.]])
        valid = np.array([[True, False], [False, False], [False, True]])
        stats = pair_statistics(left, right, valid, np.ones(3), directional=True)
        self.assertEqual(stats["pairs_proposed"], 3)
        self.assertEqual(stats["pairs_scored"], 2)
        self.assertEqual(stats["common_valid_points"], 2)
        self.assertAlmostEqual(stats["left_mae"], 3.)
        self.assertAlmostEqual(stats["right_mae"], 4.5)
        self.assertAlmostEqual(stats["left_bias"], 1.)
        self.assertAlmostEqual(stats["right_bias"], -1.5)
        self.assertAlmostEqual(stats["underreaction_pair_fraction"], .5)

    def test_no_common_valid_observations_produces_missing_metrics_not_zero_effect(self):
        stats = pair_statistics(np.ones((3, 2)), -np.ones((3, 2)),
                                np.zeros((3, 2), dtype=bool), np.ones(3),
                                directional=True)
        self.assertEqual(stats["pairs_proposed"], 3)
        self.assertEqual(stats["pairs_scored"], 0)
        self.assertEqual(stats["common_valid_points"], 0)
        for key in ("left_mae", "right_mae", "mae_left_minus_right", "left_bias",
                    "right_bias", "scaled_left_mae", "scaled_right_mae",
                    "scaled_mae_left_minus_right", "aligned_error",
                    "underreaction_pair_fraction"):
            self.assertIsNone(stats[key], key)

    def test_empty_pair_set_is_valid_and_has_no_estimated_effect(self):
        stats = pair_statistics(np.empty((0, 4)), np.empty((0, 4)),
                                np.empty((0, 4), dtype=bool), np.empty(0))
        self.assertEqual(stats["pairs_proposed"], 0)
        self.assertEqual(stats["pairs_scored"], 0)
        self.assertEqual(stats["common_valid_points"], 0)
        self.assertIsNone(stats["mae_left_minus_right"])

    def test_scaled_metrics_are_invariant_to_pair_specific_unit_conversion(self):
        left = np.array([[-2., 3.], [4., -8.]])
        right = np.array([[1., -4.], [-6., 2.]])
        valid = np.array([[True, False], [True, True]])
        scale, multiplier = np.array([2., 4.]), np.array([7., 3.])
        original = pair_statistics(left, right, valid, scale)
        converted = pair_statistics(left * multiplier[:, None],
                                    right * multiplier[:, None], valid,
                                    scale * multiplier)
        for key in ("scaled_left_mae", "scaled_right_mae", "scaled_mae_left_minus_right"):
            self.assertAlmostEqual(converted[key], original[key], msg=key)

    def test_invalid_array_shapes_and_scales_are_rejected(self):
        left, right = np.ones((2, 3)), np.ones((2, 3))
        valid, scale = np.ones((2, 3), dtype=bool), np.ones(2)
        cases = [
            (left[0], right, valid, scale),
            (left, right[:, :2], valid, scale),
            (left, right, valid[:, :2], scale),
            (left, right, valid, np.ones(3)),
            (left, right, valid, np.array([1., 0.])),
            (left, right, valid, np.array([1., -1.])),
            (left, right, valid, np.array([1., np.nan])),
            (left, right, valid, np.array([1., np.inf])),
        ]
        for arguments in cases:
            with self.subTest(shapes=[value.shape for value in arguments]):
                with self.assertRaises(ValueError):
                    pair_statistics(*arguments)


class TrendResponseSummaryIntegrationTests(unittest.TestCase):
    @staticmethod
    def example():
        """Two nodes with five versus four fixed history-matched pairs."""
        pairs = np.array([[left, left + 5, node, 0]
                          for node, count in ((0, 5), (1, 4))
                          for left in range(count)], dtype=np.int64)
        y = np.full((12, 3, 2), 100., dtype=np.float64)
        prediction = y.copy()
        for left, right, node, _ in pairs:
            # The unsupported node deliberately has far larger error, exposing
            # accidental horizon/node indexing or support-gate aggregation.
            change = np.array([1., 2., 3.]) * (1. if node == 0 else 100.)
            prediction[left, :, node] -= change
            prediction[right, :, node] += change
        left_mask, right_mask = np.zeros((12, 2), bool), np.zeros((12, 2), bool)
        left_mask[pairs[:, 0], pairs[:, 2]] = True
        right_mask[pairs[:, 1], pairs[:, 2]] = True
        spec = dict(pairs=pairs, feature=np.zeros((12, 2, 5)),
                    columns=(2, 3, 4), calipers=np.array([.05, .02, .05]),
                    directional=True, left_mask=left_mask, right_mask=right_mask)
        archive = dict(y=y, pred__R_CONTINUE=prediction,
                       valid=np.ones_like(y, dtype=bool), node_ids=np.array([42, 77]))
        return spec, archive, np.ones(2), np.zeros(12, dtype=np.int8)

    def test_summary_indexes_pairs_horizons_nodes_and_enforces_five_pair_gate(self):
        spec, archive, scale, blocks = self.example()
        result, rows = summarize(spec, archive, scale, blocks)
        self.assertEqual(result["pairs_proposed"], 9)
        self.assertEqual(result["pairs_scored"], 5)
        self.assertEqual(result["common_valid_points"], 15)
        self.assertEqual(result["nodes"], 1)
        self.assertEqual(result["node_blocks"], 1)
        self.assertEqual(result["excluded_node_blocks_below_minimum"], 1)
        self.assertAlmostEqual(result["left_bias"], -2.)
        self.assertAlmostEqual(result["right_bias"], 2.)
        self.assertAlmostEqual(result["aligned_error"], 2.)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["node_index"], 0)
        self.assertEqual(rows[0]["station_id"], 42)
        self.assertEqual(rows[0]["pairs_scored"], 5)

    def test_horizon_support_gate_drops_slice_without_rematching_fixed_pairs(self):
        spec, archive, scale, blocks = self.example()
        original_pairs = spec["pairs"].copy()
        archive["valid"][4, 2, 0] = False
        overall, _ = summarize(spec, archive, scale, blocks)
        first, first_rows = summarize(spec, archive, scale, blocks, horizon=0)
        third, third_rows = summarize(spec, archive, scale, blocks, horizon=2)
        np.testing.assert_array_equal(spec["pairs"], original_pairs)
        self.assertEqual(overall["pairs_scored"], 5)
        self.assertEqual(overall["common_valid_points"], 14)
        self.assertEqual(first["pairs_proposed"], 9)
        self.assertEqual(first["pairs_scored"], 5)
        self.assertEqual(first["common_valid_points"], 5)
        self.assertEqual(len(first_rows), 1)
        self.assertAlmostEqual(first["left_bias"], -1.)
        self.assertAlmostEqual(first["right_bias"], 1.)
        self.assertEqual(third["pairs_proposed"], 9)
        self.assertEqual(third["pairs_with_common_targets_before_minimum"], 8)
        self.assertEqual(third["pairs_scored"], 0)
        self.assertEqual(third["common_valid_points"], 0)
        self.assertEqual(third["nodes"], 0)
        self.assertEqual(third_rows, [])
        self.assertIsNone(third["left_bias"])
        self.assertIsNone(third["right_bias"])
        self.assertIsNone(third["aligned_error"])

    def test_pair_minimum_is_per_node_block_not_pooled_across_blocks(self):
        spec, archive, scale, blocks = self.example()
        # Node 0 still has five pairs overall, but neither block reaches five.
        spec["pairs"][4, 3] = 1
        blocks[[4, 9]] = 1
        result, rows = summarize(spec, archive, scale, blocks)
        block_zero, _ = summarize(spec, archive, scale, blocks, block=0)
        block_one, _ = summarize(spec, archive, scale, blocks, block=1)
        self.assertEqual(result["pairs_proposed"], 9)
        self.assertEqual(result["pairs_with_common_targets_before_minimum"], 9)
        self.assertEqual(result["excluded_node_blocks_below_minimum"], 3)
        self.assertEqual(result["pairs_scored"], 0)
        self.assertEqual(result["nodes"], 0)
        self.assertEqual(rows, [])
        self.assertEqual(block_zero["pairs_proposed"], 8)
        self.assertEqual(block_one["pairs_proposed"], 1)
        self.assertEqual(block_zero["pairs_scored"], 0)
        self.assertEqual(block_one["pairs_scored"], 0)


if __name__ == "__main__":
    unittest.main()
