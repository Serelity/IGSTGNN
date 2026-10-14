import csv
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


HERE = Path(__file__).resolve().parents[1]/'experiments/chronological'
sys.path.insert(0, str(HERE))
import min_frozen_location_probe_metrics as metrics


class LocationProbeMetricsTests(unittest.TestCase):
    def fixture(self):
        target = np.tile(np.array([0., 4., 200.])[None, None], (2, 12, 1))
        baseline = np.tile(np.array([2., 6., 190.])[None, None], (2, 12, 1))
        real = np.tile(np.array([1., 5., 199.])[None, None], (2, 12, 1))
        shuffled = np.tile(np.array([3., 1., 202.])[None, None], (2, 12, 1))
        valid = np.ones(target.shape, dtype=bool)
        codes = np.broadcast_to(np.arange(3), (2, 3))
        regions = {name: codes == i for i, name in enumerate(('direct_report_nodes', 'potential_only', 'remaining_nodes'))}
        return real, shuffled, baseline, target, valid, regions, np.array([1], dtype=np.int64)

    def test_independent_hand_computed_three_case_pooled_metrics_and_signs(self):
        report = metrics.summarize_location_probe(*self.fixture())
        real, shuffled, content = (report['pairs'][key] for key in ('real_vs_MIN', 'shuffle_vs_MIN', 'real_vs_shuffle'))
        self.assertAlmostEqual(real['off']['mae_pooled'], 14/3)
        self.assertEqual(real['on']['mae_pooled'], 1.)
        self.assertAlmostEqual(shuffled['on']['mae_pooled'], 8/3)
        self.assertAlmostEqual(real['mae_gain_pooled'], 11/3)
        self.assertAlmostEqual(shuffled['mae_gain_pooled'], 2.)
        self.assertAlmostEqual(content['mae_gain_pooled'], 5/3)
        self.assertAlmostEqual(content['paired']['rmse_gain_pooled'], math.sqrt(22/3)-1.)
        self.assertAlmostEqual(content['paired']['mape_gain_pooled'], .2525)
        self.assertAlmostEqual(content['fixed_hard_mae_gain'], 5/3)
        self.assertEqual(report['schema'], metrics.SCHEMA)

    def test_min_bias_is_fixed_when_shuffle_reverses_both_error_signs(self):
        report = metrics.summarize_location_probe(*self.fixture())
        content = report['content_strata']
        self.assertEqual(content['bias']['over']['valid_count'], 48)
        self.assertEqual(content['bias']['under']['valid_count'], 24)
        # The y=4 shuffled prediction becomes under and the y=200 prediction
        # becomes over, but both stay in their original MIN groups.
        self.assertEqual(content['cross']['[0,5)|over|potential_only']['valid_count'], 24)
        self.assertEqual(content['cross']['[200,inf)|under|remaining_nodes']['valid_count'], 24)
        self.assertEqual(content['bias']['over']['mae_gain_pooled'], 2.)
        self.assertEqual(content['bias']['under']['mae_gain_pooled'], 1.)
        self.assertEqual(content['cross']['[0,5)|under|potential_only']['valid_count'], 0)
        self.assertEqual(report['fixed_bias_reference'], 'MIN_prediction_minus_observed_target')

    def test_contributions_and_horizon_counts_reconcile_for_all_fixed_families(self):
        report = metrics.summarize_location_probe(*self.fixture())
        content = report['content_strata']
        self.assertEqual(report['reconciliation']['status'], 'PASS')
        for family in metrics.FAMILIES:
            with self.subTest(family=family):
                rows = list(content[family].values())
                self.assertEqual(sum(row['valid_count'] for row in rows), 72)
                self.assertEqual(sum(row['positive_target_count'] for row in rows), 48)
                self.assertAlmostEqual(sum(row['global_mae_content_contribution'] for row in rows), 5/3)
                self.assertAlmostEqual(sum(row['global_mape_content_contribution'] for row in rows), .2525)
                for h in range(12):
                    self.assertAlmostEqual(sum(row['per_horizon_global_mae_content_contribution'][h] for row in rows), 5/3)
                    self.assertAlmostEqual(sum(row['per_horizon_global_mape_content_contribution'][h] for row in rows), .2525)

    def test_actual_content_positive_negative_gains_are_not_subtracted_branch_parts(self):
        args = list(self.fixture())
        # Both cases improve MIN, but improve different points.
        args[0][:] = args[3]+np.array([1., 4., 1.])
        args[1][:] = args[3]+np.array([2., 1., 3.])
        report = metrics.summarize_location_probe(*args)
        pair = report['pairs']['real_vs_shuffle']['paired']
        self.assertEqual(pair['gain_positive_pooled'], 1.)
        self.assertEqual(pair['gain_negative_pooled'], 1.)
        self.assertEqual(pair['mae_gain_pooled'], 0.)
        self.assertEqual(pair['cancellation_fraction_pooled'], 1.)
        self.assertNotIn('gain_positive_pooled', report['content_strata']['global'])

    def test_identical_real_and_shuffled_predictions_produce_zero_content_effect(self):
        args = list(self.fixture())
        args[1] = args[0].copy()
        report = metrics.summarize_location_probe(*args)
        pair = report['pairs']['real_vs_shuffle']['paired']
        self.assertEqual(pair['mae_gain_pooled'], 0.)
        self.assertEqual(pair['prediction_change']['absolute_max'], 0.)
        self.assertIsNone(pair['cancellation_fraction_pooled'])
        for family, name, row in metrics._rows(report):
            if row['valid_count']:
                self.assertEqual(row['mae_gain_pooled'], 0.)
            self.assertEqual(row['global_mae_content_contribution'], 0.)

    def test_empty_group_and_zero_target_use_none_local_and_zero_global_contribution(self):
        report = metrics.summarize_location_probe(*self.fixture())
        row = report['content_strata']['cross']['[5,20)|exact|direct_report_nodes']
        self.assertEqual(row['status'], 'unavailable')
        self.assertIsNone(row['mae_gain_pooled'])
        self.assertIsNone(row['MIN']['rmse_pooled'])
        self.assertEqual(row['global_mae_content_contribution'], 0.)
        self.assertEqual(row['global_mape_content_contribution'], 0.)
        zero = report['content_strata']['regions']['direct_report_nodes']
        self.assertEqual(zero['mae_gain_pooled'], 2.)
        self.assertIsNone(zero['mape_gain_pooled'])
        self.assertEqual(zero['global_mape_content_contribution'], 0.)

    def test_all_zero_target_or_all_invalid_remain_json_serializable(self):
        for case in ('zero', 'invalid'):
            with self.subTest(case=case):
                args = list(self.fixture())
                args[3][:] = 0.
                if case == 'invalid':
                    args[4][:] = False
                    args[3][:] = np.nan
                report = metrics.summarize_location_probe(*args)
                self.assertIsNone(report['content_strata']['global']['mape_gain_pooled'])
                self.assertIsNone(report['content_strata']['global']['global_mape_content_contribution'])
                json.dumps(report, allow_nan=False)
                self.assertEqual(report['reconciliation']['status'], 'PASS')

    def test_nonuniform_mask_has_correct_per_horizon_and_no_input_mutation(self):
        args = list(self.fixture())
        args[4][:, 1:, 1:] = False
        snapshots = [value.copy() for value in args[:5]]
        report = metrics.summarize_location_probe(*args)
        global_row = report['content_strata']['global']
        self.assertEqual(global_row['valid_count_per_horizon'], [6]+[2]*11)
        self.assertAlmostEqual(global_row['mae_gain_pooled'], 54/28)
        np.testing.assert_allclose(global_row['per_horizon_mae_gain'], [5/3]+[2.]*11, rtol=0., atol=1.e-14)
        for before, after in zip(snapshots, args[:5]):
            np.testing.assert_array_equal(before, after)

    def test_report_utf8_csv_fraction_units_and_empty_groups(self):
        report = metrics.summarize_location_probe(*self.fixture())
        metadata = dict(source_best_epoch=28, best_adapter_epoch=10, shuffle_seed=2025,
                        perturbation_coverage=dict(changed_supported_fraction=.75), source_artifacts_preserved=True)
        with tempfile.TemporaryDirectory(dir=HERE.parents[1]) as folder:
            path = metrics.write_report(folder, report, metadata)
            text = path.read_text(encoding='utf-8')
            self.assertIn('原 MIN 高估／低估', text)
            self.assertIn('25.250000000', text)
            self.assertIn('"shuffle_seed": 2025', text)
            self.assertIn('"changed_supported_fraction": 0.75', text)
            self.assertIn('不能识别支持范围本身的贡献', text)
            with (Path(folder)/'content_strata.csv').open(encoding='utf-8', newline='') as stream:
                rows = list(csv.DictReader(stream))
            global_row = next(row for row in rows if row['family'] == 'global')
            self.assertAlmostEqual(float(global_row['mape_gain_pooled']), .2525)
            empty = next(row for row in rows if row['family'] == 'cross' and row['valid_count'] == '0')
            self.assertEqual(empty['mae_gain_pooled'], '')
            self.assertEqual(float(empty['global_mae_content_contribution']), 0.)
            with (Path(folder)/'horizons.csv').open(encoding='utf-8', newline='') as stream:
                horizon_rows = list(csv.DictReader(stream))
            self.assertEqual(len(horizon_rows), len(rows)*12)
            json.dumps(report, ensure_ascii=False, allow_nan=False)

    def test_invalid_shuffle_values_mask_or_regions_are_rejected(self):
        for case in ('shuffle_nan', 'mask_int', 'region_overlap'):
            with self.subTest(case=case):
                args = list(self.fixture())
                if case == 'shuffle_nan':
                    args[1][0, 0, 0] = np.nan
                elif case == 'mask_int':
                    args[4] = args[4].astype(int)
                else:
                    args[5]['potential_only'][0, 0] = True
                with self.assertRaises(ValueError):
                    metrics.summarize_location_probe(*args)

    def test_report_marks_no_effective_perturbation_and_keeps_metadata_unmodified(self):
        report = metrics.summarize_location_probe(*self.fixture())
        metadata = dict(source_best_epoch=None, best_adapter_epoch=None, validation_rows=2, shuffle_seed=7,
                        shuffle_audit=dict(rows=2, eligible_rows=2, effective_changed_rows=0,
                                           supported_station_count=6, moved_station_count=6, changed_vector_count=0,
                                           per_sample=[dict(sample_index=1234567)]))
        before = json.dumps(metadata, sort_keys=True)
        with tempfile.TemporaryDirectory(dir=HERE.parents[1]) as folder:
            text = metrics.write_report(folder, report, metadata).read_text(encoding='utf-8')
            self.assertIn('不能判定关联细节是否有作用', text)
            self.assertIn('原 MIN 最佳轮次：None', text)
            self.assertIn('| 2 | 2 | 0 | 6 | 6 | 0 |', text)
            self.assertNotIn('1234567', text)
            self.assertIn('shuffle_audit.json', text)
        self.assertEqual(json.dumps(metadata, sort_keys=True), before)

    def test_final_report_recovers_window_count_from_saved_shuffle_audit(self):
        report = metrics.summarize_location_probe(*self.fixture())
        # Older successful probes omitted validation_rows in the final report
        # metadata, while their authenticated shuffle audit retained it.
        metadata = dict(source_best_epoch=28, best_adapter_epoch=10, shuffle_seed=20261014,
                        shuffle_audit=dict(rows=837, eligible_rows=818, effective_changed_rows=818))
        before = json.dumps(metadata, sort_keys=True)
        with tempfile.TemporaryDirectory(dir=HERE.parents[1]) as folder:
            text = metrics.write_report(folder, report, metadata).read_text(encoding='utf-8')
            self.assertIn('验证窗口：837', text)
            self.assertNotIn('验证窗口：None', text)
        self.assertEqual(json.dumps(metadata, sort_keys=True), before)


if __name__ == '__main__':
    unittest.main()
