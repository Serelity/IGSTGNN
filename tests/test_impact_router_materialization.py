"""Tests for frozen v8c hierarchical router materialization."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.chronological.materialize_impact_router import (
    load_protocol, materialize_cohort, validate_route_arrays,
)


REPO = Path(__file__).resolve().parents[1]
PROTOCOL = (REPO / 'experiments/chronological' /
            'impact_router_materialization_v8c.json')


class ConstantModel:
    def __init__(self, value):
        self.value = value

    def predict(self, features):
        return np.full(len(features), self.value, dtype=np.float64)


class FakeSource:
    def sample(self, cohort, position):
        return {
            'sample': 10 + position,
            'freeway': '4', 'direction': 'E',
            'timestamp': '2023-01-01T12:00:00',
            'report_age': 5.,
            'candidate': np.array([True, False, True]),
            'distances': np.array([[1., 2., 3.], [0., 0., 0.], [3., 2., 1.]]),
            'history': np.array([
                [1., 2., 3., 4., 1.], [2., 3., 4., 5., 1.],
                [3., 4., 5., 6., 1.]]),
        }


class ProtocolTests(unittest.TestCase):
    def test_protocol_freezes_validation_and_expert_boundaries(self):
        protocol = load_protocol(PROTOCOL)
        self.assertTrue(protocol['information_boundary']['validation_labels_prohibited'])
        self.assertTrue(protocol['information_boundary']['expert_training_prohibited'])
        self.assertEqual(
            protocol['routing']['hierarchical_route'],
            'event_route_and_node_route_and_candidate_mask')

    def test_protocol_rejects_validation_target_access(self):
        protocol = json.loads(PROTOCOL.read_text(encoding='utf-8'))
        protocol['materialization']['validation_future_targets_read'] = True
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'changed.json'
            path.write_text(json.dumps(protocol), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'materialization design changed'):
                load_protocol(path)


class RouteTests(unittest.TestCase):
    def test_hierarchical_route_is_hard_supported(self):
        arrays = materialize_cohort(
            FakeSource(), 'incident_full', 2, 3,
            {'freeway': ['4'], 'direction': ['E']},
            ConstantModel(.8), .5, ConstantModel(.7), .5)
        np.testing.assert_array_equal(arrays['event_route'], [True, True])
        np.testing.assert_array_equal(
            arrays['hierarchical_route'],
            [[True, False, True], [True, False, True]])
        self.assertTrue(np.all(arrays['node_score'][:, 1] == 0))

    def test_event_gate_protects_all_nodes(self):
        arrays = materialize_cohort(
            FakeSource(), 'incident_full', 1, 3,
            {'freeway': ['4'], 'direction': ['E']},
            ConstantModel(.2), .5, ConstantModel(.9), .5)
        self.assertFalse(arrays['hierarchical_route'].any())
        self.assertTrue(arrays['node_route'][0, [0, 2]].all())

    def test_validation_rejects_route_outside_candidate(self):
        arrays = materialize_cohort(
            FakeSource(), 'incident_full', 1, 3,
            {'freeway': ['4'], 'direction': ['E']},
            ConstantModel(.8), .5, ConstantModel(.7), .5)
        arrays['node_route'][0, 1] = True
        with self.assertRaisesRegex(ValueError, 'hard support'):
            validate_route_arrays(arrays, 1, 3)


if __name__ == '__main__':
    unittest.main()
