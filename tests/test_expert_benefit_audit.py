"""Tests for v6b branch oracles and the report-time-safe shallow router."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.chronological.audit_expert_benefit import (
    aggregate_advantage, fit_weighted_ridge, gate_decision, load_protocol,
    location_category, oracle_summary, predicted_gate_matrix, routed_error,
)


REPO = Path(__file__).resolve().parents[1]
PROTOCOL = REPO / 'experiments/chronological/expert_benefit_audit_v6b.json'


class ProtocolTests(unittest.TestCase):
    def test_protocol_freezes_oracle_router_and_no_test_boundary(self):
        protocol = load_protocol(PROTOCOL)
        self.assertEqual(protocol['oracle']['primary_level_for_development_tier'],
                         'event_node')
        self.assertEqual(protocol['router']['ridge_alpha'], 10.)
        self.assertEqual(protocol['router']['activation_threshold'], 0.)
        self.assertEqual(protocol['development_gate']['global_population'],
                         'full_positive_validation')
        self.assertTrue(protocol['information_boundary']['test_split_prohibited'])
        self.assertFalse(protocol['v5c_stratification_input'][
            'high_impact_label_is_router_input'])
        self.assertEqual(
            protocol['sensor_metadata']['sha256'],
            '682f3cdf75e643f0b37356ab69cbabb27389be5089f41d3b2cbc4bede3332094')

    def test_protocol_rejects_post_result_threshold_weakening(self):
        protocol = json.loads(PROTOCOL.read_text(encoding='utf-8'))
        protocol['development_gate']['maximum_global_harm_fraction_of_A_mae'] = .01
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'changed.json'
            path.write_text(json.dumps(protocol), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'gate changed'):
                load_protocol(path)


class OracleTests(unittest.TestCase):
    @staticmethod
    def arrays():
        on = np.full((2, 12, 3, 1), 10., dtype=np.float32)
        off = on.copy()
        # Event 0: on wins at node 0, off wins at node 1. Event-node oracle can use both.
        off[0, :6, 0, 0] = 12.
        off[0, :6, 1, 0] = 5.
        # Event 1: on wins at both nodes.
        off[1, :6, :2, 0] = 11.
        valid = np.ones_like(on, dtype=bool)
        candidate = np.array([[True, True, False], [True, True, False]])
        return {'absolute_error_on': on, 'absolute_error_off': off,
                'valid': valid, 'candidate_mask': candidate,
                'positive_sample_index': np.array([1, 2])}

    def test_event_node_oracle_never_worse_and_protects_late_horizons(self):
        arrays = self.arrays()
        result = oracle_summary(arrays)
        self.assertGreater(result['event_node']['all']['routed_improvement_vs_on'], 0)
        self.assertEqual(result['event_node']['candidate_h7_h12'][
            'routed_improvement_vs_on'], 0)
        self.assertGreaterEqual(result['cell']['all']['routed_improvement_vs_on'],
                                result['event_node']['all']['routed_improvement_vs_on'])

    def test_routing_changes_only_candidate_h1_h6(self):
        arrays = self.arrays()
        advantage = arrays['absolute_error_off'] - arrays['absolute_error_on']
        node_advantage = aggregate_advantage(
            advantage, arrays['valid'], arrays['candidate_mask'], 'event_node')
        gate = node_advantage >= 0
        routed = routed_error(
            arrays['absolute_error_on'], arrays['absolute_error_off'], arrays['valid'],
            arrays['candidate_mask'], gate, 'event_node')
        np.testing.assert_array_equal(routed[:, 6:], arrays['absolute_error_on'][:, 6:])
        np.testing.assert_array_equal(routed[:, :, 2], arrays['absolute_error_on'][:, :, 2])


class LocationCategoryTests(unittest.TestCase):
    def test_category_is_derived_from_unique_nonzero_distance_support(self):
        category = location_category(
            np.array([True, True, False]),
            np.array(['4', '4', '24']), np.array(['E', 'E', 'W']))
        self.assertEqual(category, ('4', 'E'))

    def test_mixed_location_support_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'multiple freeway/direction'):
            location_category(
                np.array([True, True]),
                np.array(['4', '24']), np.array(['E', 'W']))


class RidgeTests(unittest.TestCase):
    def test_weighted_ridge_learns_direction_without_validation_tuning(self):
        x = np.array([[-2.], [-1.], [1.], [2.]])
        y = np.array([-4., -2., 2., 4.])
        model = fit_weighted_ridge(x, y, np.ones(4), alpha=.01)
        prediction = model.predict(np.array([[-3.], [3.]]))
        self.assertLess(prediction[0], 0)
        self.assertGreater(prediction[1], 0)

    def test_equal_event_node_weighting_changes_fit(self):
        x = np.array([[0.], [0.], [0.], [1.]])
        y = np.array([0., 0., 0., 10.])
        equal_rows = fit_weighted_ridge(x, y, np.ones(4), alpha=10.)
        equal_events = fit_weighted_ridge(
            x, y, np.array([1 / 3, 1 / 3, 1 / 3, 1.]), alpha=10.)
        self.assertNotEqual(equal_rows.target_mean, equal_events.target_mean)
        self.assertAlmostEqual(equal_rows.target_mean, 2.5)
        self.assertAlmostEqual(equal_events.target_mean, 5.)

    def test_event_node_gate_preserves_noncontiguous_sample_order(self):
        metadata = [
            ({'sample': 10}, 0), ({'sample': 10}, 2),
            ({'sample': 42}, 1), ({'sample': 42}, 2),
        ]
        gate = predicted_gate_matrix(
            np.array([1., -1., -2., 2.]), metadata, samples=2, nodes=3,
            level='event_node', threshold=0.)
        np.testing.assert_array_equal(
            gate, np.array([[True, True, False], [True, False, True]]))


class DecisionTests(unittest.TestCase):
    def test_development_gate_uses_full_positive_for_global_checks(self):
        protocol = load_protocol(PROTOCOL)
        full = {
            'all': {'mae_on': 10.},
            'candidate_h7_h12': {'routed_improvement_vs_on': 0.},
            'noncandidate': {'routed_improvement_vs_on': 0.},
            'routing': {'switch_off_fraction': .5},
            'uncertainty': {
                'all_improvement_vs_on': {'ci_low': -.005},
                'candidate_h1_h6_improvement_vs_on': {'ci_low': .1},
            },
        }
        matched = {'high_impact_candidate_h1_h6': {'improvement_vs_on': .2}}
        control = {
            'candidate_h1_h6': {'mae_off': 10.},
            'uncertainty': {'candidate_h1_h6_harm_vs_off': {'ci_high': .01}},
        }
        validation = {'event_node': {
            'incident_full': full, 'incident': matched,
            'primary_control': control, 'secondary_control': control,
        }}
        oracle = {'val': {'incident_full': {'event_node': {
            'all': {'relative_improvement_vs_on_percent': .5},
        }}}}
        decision = gate_decision(validation, oracle, protocol)
        self.assertTrue(decision['router_gate_passed'])
        self.assertEqual(decision['recommendation'],
                         'SMALL_BASELINE_ANCHORED_BRANCH_ROUTER_STUDY_ALLOWED')


if __name__ == '__main__':
    unittest.main()
