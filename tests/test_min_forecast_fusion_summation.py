"""Large synthetic aggregation regressions; no model, traffic, or test-set data."""
import math
import unittest
from unittest import mock

import numpy as np

from experiments.chronological import min_forecast_fusion_gain_metrics as metrics


def _fixture(rows=2, nodes=4, valid_points=None):
    shape = (rows, 12, nodes)
    target = np.broadcast_to(np.array(10., dtype=np.float64), shape)
    # Broadcasting keeps the full-release-sized test lightweight. Only one
    # horizon is active; invalid values never enter numerical reductions.
    values = dict(STATE_ON=11., EVENT_ON=11.0001, EVENT_REPORT_OFF=11.00005,
        EVENT_UNIFORM=11.0002, EVENT_SEMANTIC_UNK=11.00015,
        EVENT_LOCATION_SHUFFLE=11.00012, SOURCE_MIN=11.1)
    predictions = {name: np.broadcast_to(np.array(value), shape) for name, value in values.items()}
    current = np.full((rows, nodes), 11., dtype=np.float64)
    current.reshape(-1)[:rows*nodes//2] = 9.
    predictions['EVENT_FUSION_OFF'] = np.broadcast_to(current[:, None, :], shape)
    valid = np.zeros(shape, dtype=bool)
    valid[:, 0, :].flat[:rows*nodes if valid_points is None else valid_points] = True
    station_group = np.arange(nodes) % 3
    regions = dict(all_nodes=np.ones(nodes, bool), direct_report_nodes=station_group == 0,
        potential_only=station_group == 1, remaining_nodes=station_group == 2)
    return (predictions, target, valid, regions, np.arange(rows, dtype=np.int64),
        np.arange(nodes, dtype=np.int64), np.array([], dtype=np.int64))


class ForecastFusionSummationTests(unittest.TestCase):
    def test_release_sized_fractional_MAPE_linear_accumulation_fails_original_bound(self):
        n = 338686
        labels = np.random.default_rng(20261015).integers(0, 2, n)
        zeros = np.zeros(n, dtype=np.int64)
        control = np.full(n, .1)
        candidate = np.full(n, .10001)
        linear_control = np.bincount(zeros, weights=control)[0]
        linear_candidate = np.bincount(zeros, weights=candidate)[0]
        linear_all = linear_control-linear_candidate
        linear_parts = (np.bincount(labels, weights=control, minlength=3)
            -np.bincount(labels, weights=candidate, minlength=3)).sum()
        original_limit = 1e-10+128*np.finfo(float).eps*(linear_control+linear_candidate+1.)
        # Every input and label is valid; the historical failure is numerical.
        self.assertGreater(abs(linear_parts-linear_all), original_limit)

        whole_plan = metrics._sum_plan(zeros, 1)
        groups_plan = metrics._sum_plan(labels, 3)
        whole = (metrics._pairwise_bincount(control, whole_plan)
            -metrics._pairwise_bincount(candidate, whole_plan))[0]
        groups = (metrics._pairwise_bincount(control, groups_plan)
            -metrics._pairwise_bincount(candidate, groups_plan)).sum()
        reference = math.fsum(control)-math.fsum(candidate)
        self.assertLessEqual(abs(groups-whole), original_limit)
        self.assertLessEqual(abs(whole-reference), original_limit)
        self.assertLessEqual(abs(groups-reference), original_limit)
        self.assertEqual(metrics._pairwise_bincount(control, groups_plan)[2], 0.)

    def test_signed_cancellation_odd_blocks_and_empty_groups_against_fsum(self):
        rng = np.random.default_rng(739)
        for count in (0, 1, 63, 64, 65, 129, 193, 415152):
            for size in (1, 3, 54):
                with self.subTest(count=count, size=size):
                    labels = rng.integers(0, size, count, dtype=np.int64)
                    weights = rng.normal(0., .25, count)
                    weights[::3] += 100.
                    weights[1::3] -= 100.
                    result = metrics._pairwise_bincount(weights, metrics._sum_plan(labels, size))
                    reference = np.array([math.fsum(weights[labels == group]) for group in range(size)])
                    absolute = np.array([math.fsum(np.abs(weights[labels == group])) for group in range(size)])
                    blocks = max(1, (count+63)//64)
                    depth = 63+math.ceil(math.log2(blocks))
                    unit_roundoff = np.finfo(float).eps/2.
                    gamma = depth*unit_roundoff/(1.-depth*unit_roundoff)
                    # Independent forward-error bound for 64-item sequential
                    # leaves and an explicit pairwise tree; no relaxed check.
                    bound = gamma*absolute+np.spacing(absolute)
                    self.assertTrue(np.all(np.abs(result-reference) <= bound))
                    self.assertEqual(result.dtype, np.dtype('float64'))
                    self.assertEqual(result.shape, (size,))

    def test_full_release_axes_one_valid_horizon_all_twelve_comparisons_reconcile(self):
        n = 338686
        result = metrics.analyze_predictions(*_fixture(837, 496, n))
        self.assertEqual(set(result['comparisons']), set(metrics.COMPARISONS))
        for comparison, detail in result['comparisons'].items():
            with self.subTest(comparison=comparison):
                group = detail['groups']['all_nodes']
                self.assertEqual(group['valid_count'], n)
                self.assertEqual(group['valid_count_per_horizon'], [n]+[0]*11)
                self.assertIsNone(group['mae_gain_macro'])
                for family in result['reconciliation'][comparison].values():
                    self.assertEqual(family['status'], 'PASS')
                    self.assertEqual(family['count_residual_per_horizon'], [0]*12)
        source = result['comparisons']['EVENT_vs_STATE']['groups']['all_nodes']
        expected_mae = abs(11.-10.)-abs(11.0001-10.)
        self.assertAlmostEqual(source['mae_gain_pooled'], expected_mae, places=14)
        self.assertAlmostEqual(source['mape_gain_pooled'], expected_mae/10., places=14)
        for decomposition in result['identity_check']['decompositions'].values():
            self.assertEqual(decomposition['status'], 'POINTWISE_AND_GROUPED_IDENTITY_PASS')

    def test_weighted_partition_corruption_is_still_rejected(self):
        original = metrics._pairwise_bincount

        def corrupted(weights, plan):
            values = original(weights, plan)
            if plan[2] == 3:
                values[0] += .01
            return values

        with mock.patch.object(metrics, '_pairwise_bincount', side_effect=corrupted):
            with self.assertRaisesRegex(ValueError, 'F1 family conservation failed: EVENT_vs_STATE current_MIN_bias'):
                metrics.analyze_predictions(*_fixture())

    def test_integer_partition_count_corruption_is_still_rejected(self):
        original = np.bincount

        def corrupted(labels, weights=None, minlength=0):
            values = original(labels, weights=weights, minlength=minlength)
            if weights is None and minlength == 3:
                values[0] += 1
            return values

        with mock.patch.object(metrics.np, 'bincount', side_effect=corrupted):
            with self.assertRaisesRegex(ValueError, 'F1 family conservation failed: EVENT_vs_STATE current_MIN_bias'):
                metrics.analyze_predictions(*_fixture())


if __name__ == '__main__':
    unittest.main()
