import importlib.util
import json
import math
from pathlib import Path
import unittest

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1]/'experiments/chronological/min_frozen_gain_metrics.py'
SPEC = importlib.util.spec_from_file_location('min_frozen_gain_metrics', MODULE_PATH)
metrics = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(metrics)


class FrozenGainMetricsTests(unittest.TestCase):
    def regions(self, rows, stations):
        code = np.broadcast_to(np.arange(stations) % 3, (rows, stations))
        return {name: code == i for i, name in enumerate(metrics.REGIONS)}

    def fixture(self):
        target = np.full((2, 12, 3), np.nan)
        on, off = target.copy(), target.copy()
        valid = np.zeros(target.shape, dtype=bool)
        # Each tuple is sample, horizon, station, truth, OFF, ON. Values are
        # deliberately hand specified, including two sign-crossing forecasts.
        for n, h, v, y, before, after in (
            (0, 0, 0, 0., 2., 3.), (0, 0, 1, 4., 6., 5.),
            (0, 0, 2, 5., 3., 6.), (1, 0, 0, 20., 25., 15.),
            (1, 0, 1, 50., 50., 49.), (1, 0, 2, 200., 190., 200.),
            (0, 1, 0, 100., 105., 100.), (1, 1, 2, 1., 1., 1.)):
            target[n, h, v], off[n, h, v], on[n, h, v] = y, before, after
            valid[n, h, v] = True
        return on, off, target, valid, self.regions(2, 3)

    def test_independent_hand_computed_pooled_metrics_and_signs(self):
        report = metrics.summarize_strata(*self.fixture())
        row = report['global']
        self.assertEqual(row['count'], 8)
        self.assertEqual(row['positive_target_count'], 7)
        self.assertEqual(row['zero_target_count'], 1)
        self.assertAlmostEqual(row['off']['mae_pooled'], 26/8)
        self.assertAlmostEqual(row['on']['mae_pooled'], 11/8)
        self.assertAlmostEqual(row['off']['rmse_pooled'], math.sqrt(162/8))
        self.assertAlmostEqual(row['on']['rmse_pooled'], math.sqrt(37/8))
        self.assertAlmostEqual(row['off']['mape_pooled'], 1.25/7)
        self.assertAlmostEqual(row['on']['mape_pooled'], .72/7)
        self.assertAlmostEqual(row['mae_gain_pooled'], 15/8)
        self.assertAlmostEqual(row['mape_gain_pooled'], .53/7)
        self.assertAlmostEqual(row['rmse_gain_pooled'], 4.5-math.sqrt(37/8))
        self.assertAlmostEqual(row['mean_OFF_signed_error'], 2/8)
        self.assertAlmostEqual(row['mean_ON_signed_error'], -1/8)
        self.assertAlmostEqual(row['gain_positive_pooled'], 17/8)
        self.assertAlmostEqual(row['gain_negative_pooled'], 2/8)
        self.assertAlmostEqual(row['cancellation_fraction'], 4/19)
        self.assertEqual(report['mape_unit'], 'fraction')

    def test_nonuniform_horizon_counts_are_pooled_not_equal_horizon_means(self):
        row = metrics.summarize_strata(*self.fixture())['global']
        self.assertEqual(row['valid_count_per_horizon'], [6, 2]+[0]*10)
        self.assertAlmostEqual(row['per_horizon_mae_gain'][0], 10/6)
        self.assertAlmostEqual(row['per_horizon_mae_gain'][1], 5/2)
        self.assertEqual(row['per_horizon_mae_gain'][2:], [None]*10)
        self.assertNotAlmostEqual(row['mae_gain_pooled'], (10/6+5/2)/2)
        self.assertAlmostEqual(row['per_horizon_global_mae_gain_contribution'][0], 10/6)

    def test_prediction_up_fraction_is_count_not_signed_or_absolute_magnitude(self):
        row = metrics.summarize_strata(*self.fixture())['global']
        change = row['prediction_change']
        self.assertEqual(change['up_fraction'], 3/8)
        self.assertEqual(change['down_fraction'], 4/8)
        self.assertEqual(change['unchanged_fraction'], 1/8)
        self.assertEqual(change['absolute_max'], 10.)
        self.assertAlmostEqual(change['signed_mean_on_minus_off'], -3/8)
        self.assertAlmostEqual(change['absolute_mean'], 31/8)
        self.assertEqual(row['mae_response']['improved_fraction'], 4/8)
        self.assertEqual(row['mae_response']['worsened_fraction'], 2/8)
        self.assertEqual(row['mae_response']['unchanged_fraction'], 2/8)

    def test_bias_is_fixed_by_off_even_when_on_crosses_truth(self):
        report = metrics.summarize_strata(*self.fixture())
        over, under, exact = (report['bias'][name] for name in ('over', 'under', 'exact'))
        self.assertEqual((over['count'], under['count'], exact['count']), (4, 2, 2))
        self.assertAlmostEqual(over['global_mae_gain_contribution'], 5/8)
        self.assertAlmostEqual(under['global_mae_gain_contribution'], 11/8)
        self.assertAlmostEqual(exact['global_mae_gain_contribution'], -1/8)
        # y=5, OFF=3, ON=6 remains in under, even though ON is now over.
        crossed = report['cross']['[5,20)|under|remaining_nodes']
        self.assertEqual(crossed['count'], 1)
        self.assertEqual(crossed['mae_gain_pooled'], 1.)
        self.assertEqual(crossed['mean_OFF_signed_error'], -2.)
        self.assertEqual(crossed['mean_ON_signed_error'], 1.)
        # y=20, OFF=25, ON=15 remains over and has exactly unchanged MAE.
        crossed = report['cross']['[20,50)|over|direct_report_nodes']
        self.assertEqual(crossed['mae_response']['unchanged_fraction'], 1.)

    def test_fixed_flow_boundaries_are_lower_inclusive_and_upper_exclusive(self):
        y = [0., 4.999, 5., 19.999, 20., 49.999, 50., 99.999, 100., 199.999, 200.]
        target = np.repeat(np.array(y)[None, None, :], 12, axis=1)
        valid = np.zeros(target.shape, dtype=bool)
        valid[:, 0] = True
        report = metrics.summarize_strata(target, target+1., target, valid, self.regions(1, len(y)))
        self.assertEqual(list(report['flow']), ['[0,5)', '[5,20)', '[20,50)', '[50,100)', '[100,200)', '[200,inf)'])
        self.assertEqual([row['count'] for row in report['flow'].values()], [2, 2, 2, 2, 2, 1])
        self.assertIsNone(report['flow_bin_definitions'][-1]['upper'])
        self.assertEqual(report['flow']['[0,5)']['zero_target_count'], 1)
        self.assertEqual(report['flow']['[0,5)']['positive_target_count'], 1)

    def test_region_and_cross_contributions_reconcile_to_global_with_shared_denominators(self):
        report = metrics.summarize_strata(*self.fixture())
        direct, potential, remaining = (report['regions'][name] for name in metrics.REGIONS)
        self.assertEqual([direct['count'], potential['count'], remaining['count']], [3, 2, 3])
        self.assertAlmostEqual(direct['global_mae_gain_contribution'], 4/8)
        self.assertEqual(potential['global_mae_gain_contribution'], 0.)
        self.assertAlmostEqual(remaining['global_mae_gain_contribution'], 11/8)
        self.assertAlmostEqual(direct['global_mape_gain_contribution'], .05/7)
        self.assertAlmostEqual(potential['global_mape_gain_contribution'], .23/7)
        self.assertAlmostEqual(remaining['global_mape_gain_contribution'], .25/7)
        self.assertAlmostEqual(direct['valid_fraction_pooled'], 3/8)
        self.assertAlmostEqual(direct['positive_target_fraction_pooled'], 2/7)
        self.assertEqual(report['reconciliation']['status'], 'PASS')
        for name in ('flow', 'bias', 'regions', 'cross'):
            with self.subTest(family=name):
                rows = list(report[name].values())
                self.assertEqual(sum(row['count'] for row in rows), 8)
                self.assertAlmostEqual(sum(row['global_mae_gain_contribution'] for row in rows), 15/8)
                self.assertAlmostEqual(sum(row['global_mape_gain_contribution'] for row in rows), .53/7)
                self.assertAlmostEqual(report['reconciliation']['families'][name]['mae_gain_residual'], 0.)

    def test_invalid_nan_and_infinity_do_not_contaminate_or_change_inputs(self):
        inputs = self.fixture()
        on, off, target, valid, regions = inputs
        on[~valid], off[~valid], target[~valid] = np.nan, np.inf, -np.inf
        saved = [array.copy() for array in (on, off, target, valid)]
        report = metrics.summarize_strata(*inputs)
        self.assertAlmostEqual(report['global']['mae_gain_pooled'], 15/8)
        json.dumps(report, allow_nan=False)
        for original, before in zip((on, off, target, valid), saved):
            np.testing.assert_array_equal(original, before)

    def test_empty_stratum_has_none_local_metrics_and_zero_global_contribution(self):
        report = metrics.summarize_strata(*self.fixture())
        row = report['cross']['[200,inf)|exact|direct_report_nodes']
        self.assertEqual(row['status'], 'unavailable')
        self.assertEqual(row['count'], 0)
        self.assertEqual(row['valid_fraction_pooled'], 0.)
        self.assertIsNone(row['on']['mae_pooled'])
        self.assertIsNone(row['off']['rmse_pooled'])
        self.assertIsNone(row['mae_gain_pooled'])
        self.assertIsNone(row['prediction_change']['absolute_max'])
        self.assertIsNone(row['mae_response']['improved_fraction'])
        self.assertEqual(row['per_horizon_mae_gain'], [None]*12)
        self.assertEqual(row['global_mae_gain_contribution'], 0.)
        self.assertEqual(row['global_mape_gain_contribution'], 0.)

    def test_valid_zero_targets_are_in_mae_but_no_mape_denominator(self):
        target = np.zeros((1, 12, 3))
        report = metrics.summarize_strata(target+2., target+1., target,
                                          np.ones(target.shape, dtype=bool), self.regions(1, 3))
        row = report['global']
        self.assertEqual(row['count'], 36)
        self.assertEqual(row['positive_target_count'], 0)
        self.assertEqual(row['mae_gain_pooled'], -1.)
        self.assertIsNone(row['on']['mape_pooled'])
        self.assertIsNone(row['mape_gain_pooled'])
        self.assertIsNone(row['global_mape_gain_contribution'])
        self.assertEqual(row['per_horizon_mape_gain'], [None]*12)
        self.assertIsNone(report['reconciliation']['families']['flow']['mape_gain_contribution_sum'])
        json.dumps(report, allow_nan=False)

    def test_all_invalid_or_zero_samples_remain_unavailable(self):
        for rows in (0, 2):
            with self.subTest(rows=rows):
                target = np.full((rows, 12, 3), np.nan)
                report = metrics.summarize_strata(target, target, target,
                                                  np.zeros(target.shape, dtype=bool), self.regions(rows, 3))
                row = report['global']
                self.assertEqual(row['status'], 'unavailable')
                self.assertEqual(row['count'], 0)
                self.assertIsNone(row['valid_fraction_pooled'])
                self.assertIsNone(row['global_mae_gain_contribution'])
                self.assertIsNone(row['prediction_change']['up_fraction'])
                self.assertEqual(report['reconciliation']['status'], 'PASS')
                json.dumps(report, allow_nan=False)

    def test_identical_forecasts_are_exactly_unchanged_and_cancellation_is_undefined(self):
        on, _, target, valid, regions = self.fixture()
        report = metrics.summarize_strata(on, on, target, valid, regions)
        row = report['global']
        self.assertEqual(row['mae_gain_pooled'], 0.)
        self.assertEqual(row['gain_positive_pooled'], 0.)
        self.assertEqual(row['gain_negative_pooled'], 0.)
        self.assertIsNone(row['cancellation_fraction'])
        self.assertEqual(row['prediction_change']['unchanged_fraction'], 1.)
        self.assertEqual(row['mae_response']['unchanged_fraction'], 1.)

    def test_nonfinite_predictions_targets_and_negative_valid_truth_are_rejected(self):
        for array, value in ((0, np.nan), (1, np.inf), (2, np.inf), (2, -1.)):
            with self.subTest(array=array, value=value):
                args = list(self.fixture())
                args[array][0, 0, 0] = value
                with self.assertRaises(ValueError):
                    metrics.summarize_strata(*args)

    def test_wrong_shapes_dtypes_masks_and_horizon_contract_are_rejected(self):
        for case in ('on_shape', 'off_complex', 'target_bool', 'mask_int', 'mask_shape', 'horizons', 'empty_stations'):
            with self.subTest(case=case):
                args = list(self.fixture())
                if case == 'on_shape':
                    args[0] = args[0][:, :, :1]
                elif case == 'off_complex':
                    args[1] = args[1].astype(complex)
                elif case == 'target_bool':
                    args[2] = args[2].astype(bool)
                elif case == 'mask_int':
                    args[3] = args[3].astype(int)
                elif case == 'mask_shape':
                    args[3] = args[3][:, :, :1]
                elif case == 'horizons':
                    args[:4] = [value[:, :11] for value in args[:4]]
                else:
                    args[:4] = [value[:, :, :0] for value in args[:4]]
                with self.assertRaises(ValueError):
                    metrics.summarize_strata(*args)

    def test_region_contract_rejects_missing_extra_wrong_dtype_overlap_and_gap(self):
        for case in ('missing', 'extra', 'dtype', 'shape', 'overlap', 'gap'):
            with self.subTest(case=case):
                args = list(self.fixture())
                region = args[4]
                if case == 'missing':
                    del region['potential_only']
                elif case == 'extra':
                    region['unknown'] = region['potential_only']
                elif case == 'dtype':
                    region['potential_only'] = region['potential_only'].astype(int)
                elif case == 'shape':
                    region['potential_only'] = region['potential_only'][:1]
                elif case == 'overlap':
                    region['potential_only'][0, 0] = True
                else:
                    region['direct_report_nodes'][0, 0] = False
                with self.assertRaises(ValueError):
                    metrics.summarize_strata(*args)

    def test_invalid_edges_are_rejected_and_custom_fixed_edges_are_supported(self):
        for edges in ((5., 20.), (0., 5., 5.), (0., -1.), (0., np.inf),
                      (0., np.nan), (0.,), [[0., 5.]], ('0', '5'), (False, True)):
            with self.subTest(edges=edges), self.assertRaises(ValueError):
                metrics.summarize_strata(*self.fixture(), flow_edges=edges)
        report = metrics.summarize_strata(*self.fixture(), flow_edges=(0., 10.))
        self.assertEqual(list(report['flow']), ['[0,10)', '[10,inf)'])
        self.assertEqual([row['count'] for row in report['flow'].values()], [4, 4])

    def test_float64_accumulation_preserves_fractional_differences_and_rejects_overflow(self):
        target = np.ones((1, 12, 3), dtype=np.float64)
        valid = np.ones(target.shape, dtype=bool)
        report = metrics.summarize_strata(target, target+1.e-10, target, valid, self.regions(1, 3))
        self.assertGreater(report['global']['mae_gain_pooled'], 0.)
        self.assertEqual(report['global']['mae_response']['improved_fraction'], 1.)
        target[:] = 1.e308
        with self.assertRaises(ValueError):
            metrics.summarize_strata(-target, target, target, valid, self.regions(1, 3))


if __name__ == '__main__':
    unittest.main()
