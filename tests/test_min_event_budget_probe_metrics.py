"""Hand calculations for model gains, interventions and masked denominators."""
import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.chronological.min_event_budget_probe_metrics import CASES, summarize, write_report


REPO = Path(__file__).resolve().parents[1]


def fixture(rows=2, nodes=2):
    target = np.full((rows, 12, nodes), 10., dtype=np.float32)
    valid = np.ones_like(target, dtype=np.bool_)
    predictions = {name: target.copy()+2 for name in CASES}
    regions = {'all_nodes': np.ones(nodes, dtype=np.bool_)}
    indices = np.arange(rows, dtype=np.int64)
    return predictions, target, valid, regions, indices


def analyze(values):
    return summarize(*values)


class EventBudgetProbeMetricsTests(unittest.TestCase):
    def test_seven_cases_gain_sign_scope_cancellation_and_prediction_response(self):
        predictions, target, valid, regions, indices = fixture()
        predictions['L_ON'] = target+1
        predictions['L_OFF'] = target+3
        predictions['S_ON'] = target+np.array([0., 3.], dtype=np.float32)[None, None]
        predictions['S_OFF'] = target+np.array([3., 1.], dtype=np.float32)[None, None]
        result = summarize(predictions, target, valid, regions, indices)
        metric = result['comparisons']['S_vs_A']['metrics']['all_nodes']
        self.assertAlmostEqual(metric['mae_gain_macro'], .5)
        self.assertAlmostEqual(metric['gain_positive'], 1.)
        self.assertAlmostEqual(metric['gain_negative'], .5)
        self.assertAlmostEqual(metric['cancellation_fraction'], 2/3)
        self.assertAlmostEqual(metric['prediction_change']['absolute_mean_horizon_macro'], 1.5)
        self.assertEqual(metric['prediction_change']['absolute_max'], 2.)
        self.assertAlmostEqual(metric['prediction_change']['response_fraction_above']['1.0'], .5)
        self.assertEqual(result['comparisons']['L_vs_A']['comparison_scope'], 'independently_trained_best_validation')
        dependency = result['comparisons']['L_report_dependency']
        self.assertEqual((dependency['control'], dependency['candidate']), ('L_OFF', 'L_ON'))
        self.assertEqual(dependency['comparison_scope'], 'same_checkpoint_dependency')
        self.assertAlmostEqual(dependency['metrics']['all_nodes']['mae_gain_macro'], 2.)
        self.assertEqual(result['comparisons']['S_semantic_UNK_dependency']['comparison_scope'],
                         'same_checkpoint_UNK_sensitivity_potential_OOD')
        self.assertEqual(result['new_inference_calls'], 0)

    def test_rmse_macro_is_distinct_from_pooled_H3_H6_H12_are_individual_steps(self):
        predictions, target, valid, regions, indices = fixture(1, 1)
        predictions['A_ON'] = target+np.array([1.]*6+[3.]*6, dtype=np.float32)[None, :, None]
        result = summarize(predictions, target, valid, regions, indices)
        metric = result['cases']['A_ON']['metrics']['all_nodes']
        self.assertAlmostEqual(metric['rmse_macro'], 2.)
        self.assertAlmostEqual(metric['rmse_pooled'], np.sqrt(5.))
        self.assertEqual(metric['selected_horizons']['H3']['mae'], 1.)
        self.assertEqual(metric['selected_horizons']['H6']['mae'], 1.)
        self.assertEqual(metric['selected_horizons']['H12']['mae'], 3.)
        self.assertAlmostEqual(metric['mape_macro'], .2)

    def test_nonuniform_valid_macro_global_contributions_and_explicit_partition(self):
        predictions, target, valid, regions, indices = fixture(1, 2)
        predictions['A_ON'] = target+3
        predictions['L_ON'] = target+np.array([1., 5.], dtype=np.float32)[None, None]
        predictions['L_ON'][:, 1::2, 0] = target[:, 1::2, 0]+2
        valid[:, 1::2, 1] = False
        target[:, 1::2, 1] = np.nan
        for prediction in predictions.values():
            prediction[:, 1::2, 1] = np.nan
        regions.update(direct_report_nodes=np.array([True, False]), potential_only=np.array([False, False]),
                       remaining_nodes=np.array([False, True]), candidate_potential=np.array([True, False]))
        result = summarize(predictions, target, valid, regions, indices)
        metrics = result['comparisons']['L_vs_A']['metrics']
        self.assertAlmostEqual(metrics['all_nodes']['mae_gain_macro'], .5)
        self.assertAlmostEqual(metrics['all_nodes']['mae_gain_pooled'], 1/3)
        direct = metrics['direct_report_nodes']
        self.assertAlmostEqual(direct['mae_gain_macro'], 1.5)
        self.assertAlmostEqual(direct['global_valid_weight_fraction'], .75)
        self.assertAlmostEqual(direct['global_gain_contribution'], 1.)
        self.assertNotAlmostEqual(direct['global_gain_contribution'],
                                 direct['mae_gain_macro']*direct['global_valid_weight_fraction'])
        self.assertEqual(metrics['remaining_nodes']['status'], 'partial_horizons')
        self.assertIsNone(metrics['remaining_nodes']['mae_gain_macro'])
        self.assertAlmostEqual(metrics['remaining_nodes']['global_gain_contribution'], -.5)
        self.assertAlmostEqual(direct['global_gain_contribution']+metrics['remaining_nodes']['global_gain_contribution'],
                               metrics['all_nodes']['mae_gain_macro'])
        self.assertTrue(result['partitions']['additive'])

    def test_bias_partition_uses_control_residual_and_global_contributions_add(self):
        predictions, target, valid, regions, indices = fixture(1, 3)
        predictions['A_ON'] = target+np.array([2., -2., 0.], dtype=np.float32)[None, None]
        predictions['L_ON'] = target+np.array([1., -3., 1.], dtype=np.float32)[None, None]
        result = summarize(predictions, target, valid, regions, indices)
        comparison = result['comparisons']['L_vs_A']
        bias = comparison['bias_metrics']
        self.assertAlmostEqual(bias['over']['global_gain_contribution'], 1/3)
        self.assertAlmostEqual(bias['under']['global_gain_contribution'], -1/3)
        self.assertAlmostEqual(bias['exact']['global_gain_contribution'], -1/3)
        self.assertAlmostEqual(sum(group['global_gain_contribution'] for group in bias.values()),
                               comparison['metrics']['all_nodes']['mae_gain_macro'])
        self.assertTrue(result['interpretation']['bias_uses_future_labels_only_in_analysis'])

    def test_fixed_A_hard_set_is_not_each_case_own_tail(self):
        predictions, target, valid, regions, indices = fixture(10, 1)
        predictions['A_ON'] = target+np.arange(10, dtype=np.float32)[:, None, None]
        predictions['L_ON'] = target+np.arange(9, -1, -1, dtype=np.float32)[:, None, None]
        result = summarize(predictions, target, valid, regions, indices)
        fixed = result['fixed_hard_windows']
        self.assertEqual(fixed['selected_count'], 1)
        self.assertEqual(fixed['sample_indices'], [9])
        self.assertEqual(fixed['cases']['A_ON']['window_mean_mae'], 9.)
        self.assertEqual(fixed['cases']['L_ON']['window_mean_mae'], 0.)
        self.assertEqual(fixed['cases']['L_ON']['A_minus_case_window_mean_gain'], 9.)
        self.assertEqual(result['cases']['L_ON']['metrics']['all_nodes']['window_mae']['own_worst_5_percent_mean'], 9.)
        self.assertEqual(result['comparisons']['L_vs_A']['fixed_hard_windows']['window_mean_gain'], 9.)

    def test_hard_set_ceil_stable_ties_and_incomplete_window_exclusion(self):
        predictions, target, valid, regions, indices = fixture(11, 1)
        indices = indices[::-1]
        result = summarize(predictions, target, valid, regions, indices)
        self.assertEqual(result['fixed_hard_windows']['selected_count'], 2)
        self.assertEqual(result['fixed_hard_windows']['sample_indices'], [0, 1])
        valid[0, 0] = False
        target[0, 0] = np.nan
        for prediction in predictions.values():
            prediction[0, 0] = np.nan
        result = summarize(predictions, target, valid, regions, indices)
        fixed = result['fixed_hard_windows']
        self.assertEqual(fixed['complete_candidate_windows'], 10)
        self.assertEqual(fixed['selected_count'], 1)
        self.assertNotIn(10, fixed['sample_indices'])

    def test_fixed_window_equal_weight_and_point_weighted_metrics_have_distinct_denominators(self):
        predictions, target, valid, regions, indices = fixture(11, 2)
        predictions['A_ON'] = target.copy()
        predictions['A_ON'][0] += 10.
        predictions['A_ON'][1] += 9.
        valid[1, :, 1] = False
        target[1, :, 1] = np.nan
        result = summarize(predictions, target, valid, regions, indices)
        fixed = result['fixed_hard_windows']
        self.assertEqual(fixed['sample_indices'], [0, 1])
        self.assertEqual(fixed['cases']['A_ON']['window_mean_mae'], 9.5)
        self.assertAlmostEqual(fixed['cases']['A_ON']['metrics']['mae_macro'], 29/3)

    def test_empty_region_and_zero_gross_are_unavailable_or_null_not_false_zero_gain(self):
        values = fixture()
        values[3]['empty'] = np.zeros(2, dtype=np.bool_)
        result = analyze(values)
        metric = result['cases']['A_ON']['metrics']['empty']
        self.assertEqual(metric['status'], 'unavailable')
        self.assertIsNone(metric['mae_macro'])
        self.assertIsNone(metric['rmse_pooled'])
        self.assertIsNone(metric['window_mae']['p95'])
        empty = result['comparisons']['S_vs_A']['metrics']['empty']
        self.assertIsNone(empty['mae_gain_macro'])
        self.assertIsNone(empty['global_gain_contribution'])
        all_nodes = result['comparisons']['S_vs_A']['metrics']['all_nodes']
        self.assertEqual(all_nodes['mae_gain_macro'], 0.)
        self.assertIsNone(all_nodes['cancellation_fraction'])

    def test_zero_valid_targets_participate_in_MAE_RMSE_but_not_MAPE(self):
        predictions, target, valid, regions, indices = fixture()
        target.fill(0.)
        for prediction in predictions.values():
            prediction.fill(2.)
        result = summarize(predictions, target, valid, regions, indices)
        metric = result['cases']['A_ON']['metrics']['all_nodes']
        self.assertEqual(metric['valid_count'], target.size)
        self.assertEqual(metric['mae_macro'], 2.)
        self.assertEqual(metric['rmse_macro'], 2.)
        self.assertIsNone(metric['mape_macro'])
        self.assertIsNone(metric['mape_pooled'])
        self.assertEqual(metric['positive_target_count_per_horizon'], [0]*12)

    def test_overlapping_regions_not_additive_road_masks_and_invalid_partition_rejected(self):
        predictions, target, valid, regions, indices = fixture()
        regions.update(report_associated=np.array([True, False]), candidate_potential=np.array([True, True]))
        result = summarize(predictions, target, valid, regions, indices,
                           roads={'SR4-W': np.array([False, True], dtype=np.bool_)})
        self.assertIn('road:SR4-W', result['cases']['A_ON']['metrics'])
        self.assertFalse(result['partitions']['additive'])
        regions.update(direct_report_nodes=np.array([True, False]), potential_only=np.array([True, False]),
                       remaining_nodes=np.array([False, True]))
        with self.assertRaisesRegex(ValueError, 'disjoint'):
            summarize(predictions, target, valid, regions, indices)

    def test_reject_bad_axes_masks_cases_indices_and_nonfinite_valid_values(self):
        for fault, message in (('cases', 'seven'), ('axes', 'axes'), ('mask', 'bool'),
                               ('indices', 'unique'), ('prediction_nan', 'Non-finite'),
                               ('target_inf', 'finite'), ('negative', 'nonnegative'), ('region', 'bool')):
            with self.subTest(fault=fault):
                predictions, target, valid, regions, indices = fixture()
                if fault == 'cases':
                    predictions.pop('L_OFF')
                elif fault == 'axes':
                    predictions['A_ON'] = np.zeros((12, 2, 2))
                elif fault == 'mask':
                    valid = valid.astype(np.int64)
                elif fault == 'indices':
                    indices[:] = 0
                elif fault == 'prediction_nan':
                    predictions['S_ON'][0, 0, 0] = np.nan
                elif fault == 'target_inf':
                    target[0, 0, 0] = np.inf
                elif fault == 'negative':
                    target[0, 0, 0] = -1
                else:
                    regions['bad'] = np.ones(2, dtype=np.int64)
                with self.assertRaisesRegex(ValueError, message):
                    summarize(predictions, target, valid, regions, indices)

    def test_invalid_nan_is_ignored_inputs_preserved_and_serialization_is_finite(self):
        predictions, target, valid, regions, indices = fixture()
        valid[0, 0, 0] = False
        target[0, 0, 0] = np.nan
        for prediction in predictions.values():
            prediction[0, 0, 0] = np.nan
        snapshots = copy.deepcopy((predictions, target, valid, regions, indices))
        result = summarize(predictions, target, valid, regions, indices)
        json.dumps(result, allow_nan=False)
        for name in CASES:
            np.testing.assert_array_equal(predictions[name], snapshots[0][name])
        np.testing.assert_array_equal(target, snapshots[1])
        np.testing.assert_array_equal(valid, snapshots[2])
        np.testing.assert_array_equal(indices, snapshots[4])

    def test_report_and_CSV_have_readable_scopes_metrics_and_all_horizons(self):
        result = analyze(fixture())
        with tempfile.TemporaryDirectory(dir=REPO) as directory:
            paths = write_report(directory, result)
            root = Path(directory)
            self.assertTrue(all((root/path).is_file() for path in paths.values()))
            report = (root/'report.md').read_text(encoding='utf-8')
            self.assertIn('重新训练', report)
            self.assertIn('固定困难窗口', report)
            self.assertIn('UNK', report)
            self.assertIn('比例', report)
            with (root/'global.csv').open(encoding='utf-8-sig', newline='') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 14)
            a = next(row for row in rows if row['case'] == 'A_ON')
            self.assertAlmostEqual(float(a['mape_macro_percent']), 20.)
            with (root/'horizon.csv').open(encoding='utf-8-sig', newline='') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 14*12)
            self.assertEqual({int(row['horizon']) for row in rows}, set(range(1, 13)))


if __name__ == '__main__':
    unittest.main()
