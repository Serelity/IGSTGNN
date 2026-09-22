"""Tests for the train-only v8b node-localization audit."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.chronological.audit_incident_impact_node_localization import (
    add_route_summary, load_protocol, node_excess,
)


REPO = Path(__file__).resolve().parents[1]
PROTOCOL = (REPO / 'experiments/chronological' /
            'incident_impact_node_localization_v8b.json')


class ProtocolTests(unittest.TestCase):
    def test_protocol_freezes_node_target_and_expert_boundary(self):
        protocol = load_protocol(PROTOCOL)
        self.assertEqual(protocol['target']['early_Y_slice'], [14, 20])
        self.assertTrue(protocol['information_boundary']['neural_node_localizer_prohibited'])
        self.assertEqual(protocol['estimator']['route_unit'], 'candidate_node_expert_activation')

    def test_protocol_rejects_post_result_threshold_change(self):
        protocol = json.loads(PROTOCOL.read_text(encoding='utf-8'))
        protocol['development_gate']['maximum_primary_control_node_route_fraction'] = .5
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'changed.json'
            path.write_text(json.dumps(protocol), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'development gate changed'):
                load_protocol(path)


class TargetTests(unittest.TestCase):
    def test_node_excess_uses_baseline_adjusted_early_divergence(self):
        positive = np.zeros((26, 2), dtype=np.float64)
        primary = np.zeros_like(positive)
        secondary = np.zeros_like(positive)
        positive[14:20, 0] = 10.
        primary[14:20, 0] = 2.
        secondary[14:20, 0] = 2.
        positive[14:20, 1] = 3.
        primary[14:20, 1] = 1.
        secondary[14:20, 1] = 1.
        nodes, excess = node_excess(
            positive, primary, secondary, np.array([True, False]), 1.)
        np.testing.assert_array_equal(nodes, [0])
        self.assertAlmostEqual(excess[0], 8.)

    def test_node_excess_rejects_empty_candidate_support(self):
        zeros = np.zeros((26, 2), dtype=np.float64)
        with self.assertRaisesRegex(ValueError, 'support is empty'):
            node_excess(zeros, zeros, zeros, np.array([False, False]), 1.)


class SummaryTests(unittest.TestCase):
    def test_route_summary_initializes_every_cohort(self):
        summaries = {'incident': {'roc_auc': .7}}
        for cohort in ('incident', 'primary_control', 'secondary_control'):
            add_route_summary(
                summaries, cohort, np.array([.2, .8]), np.array([False, True]))
        self.assertEqual(set(summaries), {
            'incident', 'primary_control', 'secondary_control'})
        self.assertEqual(summaries['incident']['roc_auc'], .7)
        self.assertEqual(summaries['primary_control']['nodes'], 2)
        self.assertEqual(summaries['secondary_control']['route_fraction'], .5)


if __name__ == '__main__':
    unittest.main()
