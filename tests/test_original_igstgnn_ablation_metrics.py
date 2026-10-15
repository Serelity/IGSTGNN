"""CPU-only evidence checks for twelve-state checkpoint screen statistics."""
import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.chronological import original_igstgnn_ablation_metrics as metrics


def fixture(rows=4, horizons=3, nodes=4):
    target = np.arange(rows*horizons*nodes, dtype=np.float32).reshape(rows, horizons, nodes)+1.
    valid = np.ones_like(target, bool)
    predictions = {case: target+2. for case in metrics.CASE_NAMES}
    predictions['DIF_FORECAST_READOUT_OFF'] = target+3.
    predictions['INH_FORECAST_READOUT_OFF'] = target+1.
    indices = np.arange(rows, dtype=np.int64)+100
    stations = np.arange(nodes, dtype=np.int64)+1000
    return predictions, target, valid, indices, stations


class OriginalIGSTGNNMetricsTests(unittest.TestCase):
    def test_twelve_cases_required_axes_and_original_mask_unchanged(self):
        predictions, target, valid, indices, stations = fixture()
        before = copy.deepcopy((predictions, target, valid, indices, stations))
        result = metrics.summarize_predictions(predictions, target, valid, indices, stations)
        self.assertEqual(result['case_order'], list(metrics.CASE_NAMES))
        self.assertEqual(result['station_ids'], stations.tolist())
        self.assertEqual(result['mask_policy'], 'caller_supplied_original_mask_unchanged')
        for key in predictions:
            np.testing.assert_array_equal(predictions[key], before[0][key])
        for current, old in zip((target, valid, indices, stations), before[1:]):
            np.testing.assert_array_equal(current, old)
        for bad in ({key: value for key, value in predictions.items() if key != 'FULL'},
            dict(predictions, EXTRA=target), dict(predictions, UNKNOWN=predictions['FULL'])):
            with self.assertRaisesRegex(ValueError, 'twelve'):
                metrics.summarize_predictions(bad, target, valid, indices, stations)
        bad = dict(predictions, FULL=np.zeros((4, 4, 3)))
        with self.assertRaisesRegex(ValueError, 'axes'):
            metrics.summarize_predictions(bad, target, valid, indices, stations)
        with self.assertRaises(ValueError):
            metrics.summarize_predictions(predictions, target, valid, indices, stations, case_order=['FULL']*12)

    def test_utility_sign_full_minus_intervention_response_and_units(self):
        predictions, target, valid, indices, stations = fixture()
        result = metrics.summarize_predictions(predictions, target, valid, indices, stations)
        full = result['cases']['FULL']['groups']['all_nodes']
        self.assertEqual(full['mae_pooled'], 2.)
        self.assertEqual(full['rmse_pooled'], 2.)
        self.assertEqual(full['mae_macro'], 2.)
        self.assertAlmostEqual(full['mape_pooled'], float((2./target.astype(np.float64)).mean()))
        positive = result['comparisons']['DIF_FORECAST_READOUT_OFF']['groups']['all_nodes']
        negative = result['comparisons']['INH_FORECAST_READOUT_OFF']['groups']['all_nodes']
        self.assertEqual(positive['mae_gain_pooled'], 1.)
        self.assertEqual(negative['mae_gain_pooled'], -1.)
        self.assertEqual(positive['prediction_signed_mean'], -1.)
        self.assertEqual(positive['prediction_down_fraction'], 1.)
        self.assertEqual(positive['prediction_response_fraction'], 1.)
        self.assertEqual(result['utility_ranking'][0], 'DIF_FORECAST_READOUT_OFF')
        self.assertEqual(result['utility_ranking'][-1], 'INH_FORECAST_READOUT_OFF')
        unchanged = result['comparisons']['NORM_ONLY']['groups']['all_nodes']
        self.assertEqual(unchanged['cancellation_fraction'], 0.)
        self.assertEqual(unchanged['prediction_response_fraction'], 0.)

    def test_pooled_and_macro_distinct_and_mask_does_not_depend_on_prediction(self):
        predictions, target, valid, indices, stations = fixture(rows=1, horizons=2, nodes=3)
        valid[0, 1, 1:] = False
        for case in predictions:
            predictions[case][:, 0] = target[:, 0]+1.
            predictions[case][:, 1] = target[:, 1]+3.
            predictions[case][~valid] += 1e6
        result = metrics.summarize_predictions(predictions, target, valid, indices, stations)
        full = result['cases']['FULL']['groups']['all_nodes']
        self.assertEqual(full['valid_count'], 4)
        self.assertEqual(full['mae_pooled'], 1.5)
        self.assertEqual(full['mae_macro'], 2.)
        self.assertAlmostEqual(full['rmse_pooled'], np.sqrt(3.))
        self.assertEqual(full['rmse_macro'], 2.)
        self.assertEqual(full['valid_count_per_horizon'], [3, 1])

    def test_positive_negative_cancellation_and_response_threshold(self):
        predictions, target, valid, indices, stations = fixture(rows=1, horizons=1, nodes=4)
        predictions['DIF_FORECAST_READOUT_OFF'] = target+np.array([[[3., 3., 1., 1.]]])
        predictions['ICSF_ONLY'] = predictions['FULL']+.0001
        result = metrics.summarize_predictions(predictions, target, valid, indices, stations, response_tol=.001)
        row = result['comparisons']['DIF_FORECAST_READOUT_OFF']['groups']['all_nodes']
        self.assertEqual(row['mae_gain_pooled'], 0.)
        self.assertEqual(row['gain_positive'], .5)
        self.assertEqual(row['gain_negative'], .5)
        self.assertEqual(row['cancellation_fraction'], 1.)
        self.assertEqual(row['prediction_signed_mean'], 0.)
        self.assertEqual(row['prediction_up_fraction'], .5)
        self.assertEqual(row['prediction_down_fraction'], .5)
        tiny = result['comparisons']['ICSF_ONLY']['groups']['all_nodes']
        self.assertGreater(tiny['prediction_absolute_mean'], 0.)
        self.assertEqual(tiny['prediction_response_fraction'], 0.)

    def test_random_float64_oracle_all_cases_horizons_and_pairing(self):
        rng = np.random.default_rng(20261015)
        target = rng.lognormal(3., 1.3, (9, 12, 7)).astype(np.float32)
        valid = rng.random(target.shape) > .2
        valid[0, :, 0] = True
        predictions = {case: (target+rng.normal(0, 3, target.shape)).astype(np.float32) for case in metrics.CASE_NAMES}
        result = metrics.summarize_predictions(predictions, target, valid, np.arange(9), np.arange(7))
        y = target.astype(np.float64)
        full_error = np.abs(predictions['FULL'].astype(np.float64)-y)
        for case, prediction in predictions.items():
            error = np.abs(prediction.astype(np.float64)-y)
            actual = result['cases'][case]['groups']['all_nodes']
            self.assertAlmostEqual(actual['mae_pooled'], float(error[valid].mean()), places=12)
            self.assertAlmostEqual(actual['rmse_pooled'], float(np.sqrt(np.square(error[valid]).mean())), places=12)
            self.assertAlmostEqual(actual['mape_pooled'], float((error/y)[valid].mean()), places=12)
            self.assertAlmostEqual(actual['mae_macro'], float(np.mean([error[:, h][valid[:, h]].mean() for h in range(12)])), places=12)
            if case != 'FULL':
                gain = (error-full_error)[valid]
                comparison = result['comparisons'][case]['groups']['all_nodes']
                self.assertAlmostEqual(comparison['mae_gain_pooled'], float(gain.mean()), places=12)
                self.assertAlmostEqual(comparison['gain_positive']-comparison['gain_negative'], comparison['mae_gain_pooled'], places=12)
                self.assertAlmostEqual(comparison['prediction_signed_mean'], float((predictions['FULL'].astype(np.float64)-prediction)[valid].mean()), places=12)

    def test_full_fixed_hard_complete_windows_ties_and_no_candidate_reselection(self):
        predictions, target, valid, indices, stations = fixture()
        indices[:] = [20, 10, 30, 40]
        errors = np.array([5., 5., 999., 1.])[:, None, None]
        predictions['FULL'] = target+errors
        valid[2, 0] = False
        predictions['DIF_FORECAST_READOUT_OFF'] = target+np.array([1., 1., 999., 1000.])[:, None, None]
        result = metrics.summarize_predictions(predictions, target, valid, indices, stations, hard_fraction=.5)
        selected = result['fixed_hard_windows']
        self.assertEqual(selected['complete_windows'], 3)
        self.assertEqual(selected['sample_indices'], [10, 20])
        self.assertEqual(result['cases']['FULL']['fixed_hard_window_mean'], 5.)
        self.assertEqual(result['cases']['DIF_FORECAST_READOUT_OFF']['fixed_hard_window_mean'], 1.)
        self.assertEqual(result['comparisons']['DIF_FORECAST_READOUT_OFF']['fixed_hard_window_gain'], -4.)
        self.assertEqual(result['comparisons']['DIF_FORECAST_READOUT_OFF']['groups']['fixed_FULL_hard']['mae_gain_pooled'], -4.)

    def test_flow_road_global_contributions_and_empty_groups(self):
        predictions, target, valid, indices, stations = fixture(rows=2, horizons=2, nodes=4)
        target[:] = np.array([1., 10., 60., 250.])
        predictions = {case: target+2. for case in predictions}
        predictions['DIF_FORECAST_READOUT_OFF'] = target+3.
        roads = {'first': [0, 1], 'second': [2, 3], 'empty': []}
        result = metrics.summarize_predictions(predictions, target, valid, indices, stations, roads)
        groups = result['comparisons']['DIF_FORECAST_READOUT_OFF']['groups']
        flows = [row for key, row in groups.items() if key.startswith('flow:')]
        self.assertEqual(sum(row['valid_count'] for row in flows), groups['all_nodes']['valid_count'])
        self.assertEqual(sum(row['global_mae_contribution'] or 0. for row in flows), 1.)
        self.assertEqual(groups['road:first']['global_mae_contribution']+groups['road:second']['global_mae_contribution'], 1.)
        empty = groups['road:empty']
        self.assertEqual(empty['valid_count'], 0)
        for key in ('mae_gain_pooled', 'mae_gain_macro', 'gain_positive', 'gain_negative', 'prediction_absolute_max',
            'cancellation_fraction', 'global_mae_contribution'):
            self.assertIsNone(empty[key])
        case = result['cases']['FULL']['groups']['road:empty']
        self.assertIsNone(case['mae_pooled'])
        self.assertIsNone(case['rmse_macro'])
        labels = np.array(['first', 'first', 'second', 'second'])
        alternate = metrics.summarize_predictions(predictions, target, valid, indices, stations, labels)
        self.assertEqual(alternate['comparisons']['DIF_FORECAST_READOUT_OFF']['groups']['road:first'], groups['road:first'])

    def test_missing_horizon_macro_and_no_complete_hard_windows(self):
        predictions, target, valid, indices, stations = fixture()
        valid[:, 1] = False
        result = metrics.summarize_predictions(predictions, target, valid, indices, stations)
        self.assertIsNone(result['cases']['FULL']['groups']['all_nodes']['mae_macro'])
        self.assertIsNone(result['comparisons']['DIF_FORECAST_READOUT_OFF']['groups']['all_nodes']['mae_gain_macro'])
        self.assertEqual(result['fixed_hard_windows']['sample_indices'], [])
        self.assertIsNone(result['comparisons']['DIF_FORECAST_READOUT_OFF']['fixed_hard_window_gain'])

    def test_undefined_inputs_fail_closed(self):
        predictions, target, valid, indices, stations = fixture()
        for bad_valid in (valid.astype(np.float32), np.zeros_like(valid)):
            with self.assertRaises(ValueError):
                metrics.summarize_predictions(predictions, target, bad_valid, indices, stations)
        for bad_target in (np.full_like(target, -1.), np.zeros_like(target), np.full_like(target, np.nan)):
            with self.assertRaises(ValueError):
                metrics.summarize_predictions(predictions, bad_target, valid, indices, stations)
        with self.assertRaises(ValueError):
            metrics.summarize_predictions(predictions, target, valid, np.zeros_like(indices), stations)
        with self.assertRaises(ValueError):
            metrics.summarize_predictions(predictions, target, valid, indices, np.zeros_like(stations))
        for roads in ({'bad': [4]}, {'bad': [1, 1]}, {'bad': [.5]}):
            with self.assertRaises(ValueError):
                metrics.summarize_predictions(predictions, target, valid, indices, stations, roads)

    def test_timing_raw_records_warmup_scope_and_missing_cases(self):
        timings = {'FULL': [dict(wall_seconds=999., warmup=True),
            dict(wall_seconds=2., cuda_seconds=1., peak_allocated_bytes=100, incremental_peak_bytes=20, batch_size=64,
                pass_index=0, batch_index=0, case_order_index=0, warmup=False),
            dict(wall_seconds=4., cuda_seconds=3., peak_allocated_bytes=150, incremental_peak_bytes=30, batch_size=5,
                pass_index=0, batch_index=1, case_order_index=0, warmup=False)]}
        before = copy.deepcopy(timings)
        result = metrics.summarize_timings(timings)
        row = result['cases']['FULL']
        self.assertEqual(row['recorded_calls'], 3)
        self.assertEqual(row['measured_calls'], 2)
        self.assertEqual(row['wall_seconds_mean'], 3.)
        self.assertEqual(row['wall_seconds_total'], 6.)
        self.assertEqual(row['cuda_seconds_mean'], 2.)
        self.assertEqual(row['peak_allocated_bytes_max'], 150)
        self.assertEqual(row['incremental_peak_bytes_max'], 30)
        self.assertEqual(result['cases']['NORM_ONLY']['status'], 'unmeasured')
        self.assertFalse(result['training_time_measured'])
        self.assertEqual(before, timings)
        alternate = metrics.summarize_timings({'FULL': {'records': [dict(seconds=.1)]}})
        self.assertEqual(alternate['cases']['FULL']['wall_seconds_mean'], .1)
        for bad in ({'EXTRA': []}, {'FULL': [dict(wall_seconds=-1)]}, {'FULL': [dict(wall_seconds=np.inf)]},
            {'FULL': [dict(wall_seconds=1, warmup=1)]}, {'FULL': [dict(wall_seconds=1, incremental_peak_bytes=-1)]}):
            with self.assertRaises(ValueError):
                metrics.summarize_timings(bad)

    def test_two_complete_validation_passes_batch_sizes_and_sample_weighting(self):
        records = []
        for pass_index in range(2):
            for batch_index, size in enumerate([48]*17+[21]):
                records.append(dict(wall_seconds=size*.001, batch_size=size,
                    pass_index=pass_index, batch_index=batch_index, warmup=False))
        result = metrics.summarize_timings({'FULL': records}, expected_validation_samples=837, expected_passes=2)
        row = result['cases']['FULL']
        self.assertEqual(row['measured_calls'], 36)
        self.assertEqual(row['measured_samples'], 837*2)
        self.assertEqual(row['batch_sizes']['48']['measured_calls'], 34)
        self.assertEqual(row['batch_sizes']['21']['measured_calls'], 2)
        self.assertAlmostEqual(row['batch_sizes']['48']['wall_seconds_mean'], .048)
        self.assertAlmostEqual(row['batch_sizes']['21']['wall_seconds_mean'], .021)
        self.assertEqual([p['measured_samples'] for p in row['validation_passes']], [837, 837])
        self.assertTrue(row['validation_sample_counts_verified'])
        self.assertEqual(row['complete_validation_passes'], 2)
        self.assertAlmostEqual(row['validation_seconds_per_pass_mean'], .837)
        self.assertAlmostEqual(row['ms_per_sample'], 1.)
        self.assertAlmostEqual(row['measured_ms_per_sample'], 1.)

    def test_missing_pass_partial_samples_or_unknown_metadata_do_not_claim_validation_time(self):
        records = [dict(wall_seconds=.048, batch_size=48, pass_index=0, batch_index=i) for i in range(17)]
        complete = records+[dict(wall_seconds=.021, batch_size=21, pass_index=0, batch_index=17)]
        for values in (records, complete, [dict(wall_seconds=.5)]):
            result = metrics.summarize_timings({'FULL': values}, expected_validation_samples=837, expected_passes=2)
            row = result['cases']['FULL']
            self.assertFalse(row['validation_sample_counts_verified'])
            self.assertIsNone(row['validation_seconds_per_pass_mean'])
            self.assertIsNone(row['ms_per_sample'])
            self.assertEqual(row['measured_calls'], len(values))
        old = metrics.summarize_timings({'FULL': [dict(wall_seconds=.5)]})['cases']['FULL']
        self.assertEqual(old['wall_seconds_mean'], .5)
        self.assertIsNone(old['measured_samples'])
        self.assertIsNone(old['validation_seconds_per_pass_mean'])
        self.assertIsNone(old['ms_per_sample'])
        with self.assertRaisesRegex(ValueError, 'Repeated timing batch_index'):
            metrics.summarize_timings({'FULL': complete+[complete[-1]]}, expected_validation_samples=837, expected_passes=1)

    def test_report_json_csv_units_metadata_and_no_overwrite(self):
        predictions, target, valid, indices, stations = fixture()
        result = metrics.summarize_predictions(predictions, target, valid, indices, stations, {'road': [0, 1, 2, 3]})
        metadata = dict(engineering_only=True, primary_baseline='original_IGSTGNN', registered_parameters=1234,
            inactive_local_bn_parameters=12, replay={'status': 'PASS'}, checkpoint_identity={'sha256': 'fixture'},
            evaluation_mask={'source': 'original_engine'})
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temporary:
            output = Path(temporary)
            returned = metrics.write_report(output, result, metadata, {'FULL': [dict(wall_seconds=.01, warmup=False)]})
            self.assertTrue(all((output/name).is_file() for name in returned['files']))
            restored = json.loads((output/'analysis.json').read_text(encoding='utf-8'))
            self.assertEqual(restored, result)
            self.assertEqual(json.loads((output/'metadata.json').read_text(encoding='utf-8')), metadata)
            report = (output/'report.md').read_text(encoding='utf-8')
            for phrase in ('干预状态误差 − FULL误差', '正数表示完整路径更好', '不代表真正删除模块后的速度',
                '没有训练耗时结论', '不视为独立事故重复', '注册量不等于活跃计算量'):
                self.assertIn(phrase, report)
            self.assertIn(f"{100*result['cases']['FULL']['groups']['all_nodes']['mape_pooled']:.9f}", report)
            with (output/'cases.csv').open(encoding='utf-8-sig', newline='') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 12)
            self.assertAlmostEqual(float(rows[0]['mape_pooled']), result['cases']['FULL']['groups']['all_nodes']['mape_pooled'])
            with (output/'timings.csv').open(encoding='utf-8-sig', newline='') as stream:
                timing_rows = list(csv.DictReader(stream))
            self.assertNotIn('batch_sizes', timing_rows[0])
            self.assertNotIn('validation_passes', timing_rows[0])
            self.assertIn('validation_seconds_per_pass_mean', timing_rows[0])
            self.assertIn('ms_per_sample', timing_rows[0])
            with self.assertRaisesRegex(ValueError, 'refusing overwrite'):
                metrics.write_report(output, result, metadata)


if __name__ == '__main__':
    unittest.main()
