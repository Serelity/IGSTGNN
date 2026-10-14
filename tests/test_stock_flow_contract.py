import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.utils.stock_flow_contract import audit_graph_contract, fit_training_flow_scale


class StockFlowContractTests(unittest.TestCase):
    def test_actual_graph_does_not_upgrade_candidate_coverage_to_physical(self):
        root = Path(__file__).resolve().parents[1]/'experiments/chronological/internal_spatial_graph_is11'
        a = audit_graph_contract(root/'graph.json', root/'source_summary.json')
        self.assertEqual(a['candidate_covered_nodes'], 476)
        self.assertEqual(a['jointly_certified_nodes'], 0)
        self.assertFalse(a['strict_physical_ready'])
        self.assertTrue(a['represented_open_boundaries_learned_not_zero'])
        self.assertTrue(a['unrepresented_model_sources_zero_by_model_definition'])
        self.assertFalse(a['absence_of_real_unrepresented_sources_certified'])
        self.assertEqual(len(a['open_storage_entry_completions']), 20)
        self.assertEqual(len(a['open_storage_exit_completions']), 20)
        self.assertFalse(a['work_conserving_classical_queue'])
        self.assertEqual(a['historical_multichannel_semantics']['source_order'], ['flow', 'occupancy', 'speed'])
        with self.assertRaisesRegex(ValueError, 'independent.*certificate'):
            audit_graph_contract(root/'graph.json', root/'source_summary.json', strict_physical=True)

    def test_duplicate_or_wrong_station_axis_is_rejected(self):
        root = Path(__file__).resolve().parents[1]/'experiments/chronological/internal_spatial_graph_is11'
        with self.assertRaisesRegex(ValueError, 'station axis'):
            audit_graph_contract(root/'graph.json', root/'source_summary.json', [1, 2])
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as folder:
            graph = json.loads((root/'graph.json').read_text())
            graph['edges'][1] = dict(graph['edges'][0], index=1)
            path = Path(folder)/'graph.json'
            path.write_text(json.dumps(graph))
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                audit_graph_contract(path, root/'source_summary.json')

    def test_training_scale_is_fixed_positive_and_targets_not_an_argument(self):
        x = np.zeros((2, 12, 3, 3), dtype=np.float32)
        x[..., 0, 0], x[..., 1, 0] = 2., 6.
        scale, p = fit_training_flow_scale(x, 0., 1.)
        np.testing.assert_array_equal(scale, [2., 6., 4.])
        self.assertEqual(p['zero_only_fallback_nodes'], [2])
        self.assertFalse(p['future_targets_used'])
        self.assertEqual(p['stations'][2]['positive_observation_status'], 'no_positive_training_observations')
        self.assertIsNone(p['stations'][2]['positive_max'])
        self.assertIsNone(p['stations'][2]['positive_p95'])
        self.assertEqual(p['expanded_nodes'], [])
        json.dumps(p, allow_nan=False)
        changed_clocks = x.copy()
        changed_clocks[..., 1:] = .5
        other, q = fit_training_flow_scale(changed_clocks, 0., 1.)
        np.testing.assert_array_equal(scale, other)
        self.assertEqual(p['scale_sha256'], q['scale_sha256'])
        self.assertEqual(p['historical_flow_sha256'], q['historical_flow_sha256'])
        zero, _ = fit_training_flow_scale(np.zeros_like(x), 0., 1.)
        np.testing.assert_array_equal(zero, np.ones(3, dtype=np.float32))

    def test_low_median_spike_is_covered_without_incident_condition_factor(self):
        x = np.ones((1, 12, 1), dtype=np.float32)
        x[0, -1, 0] = 80.
        scale, p = fit_training_flow_scale(x, 0., 1.)
        np.testing.assert_array_equal(scale, [25.])
        self.assertEqual(p['stations'][0]['raw_positive_median'], 1.)
        self.assertEqual(p['stations'][0]['positive_max'], 80.)
        self.assertGreaterEqual(4*float(scale[0]), 1.25*80.)
        self.assertEqual(p['expanded_nodes'], [0])
        self.assertEqual(p['base_service_ceiling_audit_before']['above_ceiling_history_points'], 1)
        self.assertEqual(p['base_service_ceiling_audit_after']['above_ceiling_history_points'], 0)
        self.assertFalse(p['conditional_D_factor_required_to_cover_trainX_peaks'])
        self.assertFalse(p['physical_capacity_estimate'])
        self.assertIn('numeric_envelope', p['envelope_scope'])
        json.dumps(p, allow_nan=False)

    def test_peak_envelope_parameters_are_validated_and_float32_rounds_up(self):
        x = np.ones((1, 12, 1), dtype=np.float32)
        x[0, -1, 0] = 10.1
        scale, p = fit_training_flow_scale(x, 0., 1., peak_margin=1.25, max_rate_factor=4.)
        self.assertGreaterEqual(4*float(scale[0]), 1.25*float(x.max()))
        self.assertEqual(p['peak_margin'], 1.25)
        for options in ({'peak_margin': True}, {'peak_margin': .9}, {'peak_margin': float('nan')},
                        {'max_rate_factor': False}, {'max_rate_factor': 0.}, {'max_rate_factor': float('inf')}):
            with self.subTest(options=options):
                with self.assertRaises(ValueError):
                    fit_training_flow_scale(x, 0., 1., **options)

    def test_invalid_training_history_or_scaler_rejected(self):
        x = np.zeros((2, 12, 3, 3), dtype=np.float32)
        for value, mean, std in ((x, 0., 0.), (x[:, :3], 0., 1.), (x, float('nan'), 1.)):
            with self.assertRaises(ValueError):
                fit_training_flow_scale(value, mean, std)


if __name__ == '__main__':
    unittest.main()
