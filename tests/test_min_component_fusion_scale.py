"""Release-sized synthetic F2 reductions without model or traffic inputs."""
import unittest

import numpy as np

from experiments.chronological import evaluate_min_component_fusion as evaluation
from tests.test_min_forecast_fusion_summation import _fixture


class ComponentFusionScaleTests(unittest.TestCase):
    def test_release_axes_fractional_metrics_reconcile_and_missing_macro_stays_unavailable(self):
        count = 338686
        original, target, valid, regions, indices, stations, _ = _fixture(837, 496, count)
        shape = target.shape
        predictions = {name: original[name] for name in evaluation.CASES if name in original}
        predictions['CONTROL_ON'] = np.broadcast_to(np.array(11.0002), shape)
        predictions['EVENT_CONTENT_OFF'] = np.broadcast_to(np.array(11.00007), shape)
        result = evaluation.summarize(predictions, target, valid, regions, indices, stations)
        self.assertEqual(result['status'], 'MIN_COMPONENT_FUSION_ANALYSIS_PASS')
        self.assertEqual(set(result['cases']), set(evaluation.CASES))
        self.assertEqual(set(result['comparisons']), set(evaluation.COMPARISONS))
        for name, row in result['cases'].items():
            metric = row['metrics']['all_nodes']
            with self.subTest(case=name):
                for field in ('mae_macro', 'rmse_macro', 'mape_macro'):
                    self.assertIsNone(metric[field])
                self.assertTrue(all(value is None for value in metric['per_horizon_mae'][1:]))
        for name, row in result['comparisons'].items():
            metric = row['metrics']['all_nodes']
            with self.subTest(comparison=name):
                self.assertEqual(metric['valid_count'], count)
                self.assertEqual(metric['valid_count_per_horizon'], [count]+[0]*11)
                for field in ('mae_gain_macro', 'rmse_gain_macro', 'mape_gain_macro'):
                    self.assertIsNone(metric[field])
                for family in (['direct_report_nodes', 'potential_only', 'remaining_nodes'],):
                    self.assertEqual(sum(row['metrics'][key]['valid_count'] for key in family), count)
                self.assertEqual(sum(value['valid_count'] for key, value in row['strata'].items()
                                     if key.startswith('flow:')), count)
                self.assertEqual(sum(value['valid_count'] for key, value in row['strata'].items()
                                     if key.startswith('bias:')), count)
        event = result['cases']['EVENT_ON']['metrics']['all_nodes']
        error = abs(11.0001-10.)
        self.assertAlmostEqual(event['mae_pooled'], error, places=13)
        self.assertAlmostEqual(event['rmse_pooled'], error, places=13)
        self.assertAlmostEqual(event['mape_pooled'], error/10., places=14)
        self.assertEqual(result['fixed_hard_windows']['sample_indices'], [])


if __name__ == '__main__':
    unittest.main()
