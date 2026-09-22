"""Tests for the train-only v8a observable impact-identification audit."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.chronological.audit_incident_impact_identification import (
    load_protocol, rolling_folds, roc_auc, top_quartile_lift,
)


REPO = Path(__file__).resolve().parents[1]
PROTOCOL = (REPO / 'experiments/chronological' /
            'incident_impact_identification_v8a.json')


class ProtocolTests(unittest.TestCase):
    def test_protocol_freezes_report_time_boundary_and_route_gate(self):
        protocol = load_protocol(PROTOCOL)
        self.assertTrue(protocol['features']['future_outcome_features_forbidden'])
        self.assertTrue(protocol['information_boundary']['expert_training_prohibited'])
        self.assertEqual(protocol['estimator']['route_threshold'], 'fit_incident_score_q75')

    def test_protocol_rejects_post_result_gate_change(self):
        protocol = json.loads(PROTOCOL.read_text(encoding='utf-8'))
        protocol['development_gate']['maximum_primary_control_route_fraction'] = .5
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'changed.json'
            path.write_text(json.dumps(protocol), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'development gate changed'):
                load_protocol(path)


def repeated_weeks(counts):
    values = []
    for number, count in enumerate(counts, start=1):
        values.extend([f'2023-W{number:02d}'] * count)
    return np.asarray(values)


class RollingOriginTests(unittest.TestCase):
    def test_final_seventeen_weeks_are_audited_once(self):
        protocol = load_protocol(PROTOCOL)
        weeks = repeated_weeks([89] * 17 + [91] + [89] * 5 + [91] +
                                [103] * 6 + [68] * 4 + [66])
        folds = rolling_folds(weeks, protocol)
        self.assertEqual([len(fold['audit_indices']) for fold in folds], [536, 618, 338])
        self.assertIsInstance(folds[0]['fit_weeks'], list)
        self.assertIsInstance(folds[0]['audit_weeks'], list)
        for fold in folds:
            self.assertLess(max(fold['fit_weeks']), min(fold['audit_weeks']))
            self.assertEqual(np.intersect1d(
                fold['fit_indices'], fold['audit_indices']).size, 0)


class MetricTests(unittest.TestCase):
    def test_roc_auc_handles_ties(self):
        self.assertAlmostEqual(
            roc_auc([0, 1, 0, 1], [.2, .8, .2, .4]), 1.)

    def test_top_quartile_lift_detects_concentration(self):
        self.assertAlmostEqual(
            top_quartile_lift([1, 1, 0, 0], [.9, .8, .2, .1]), 2.)

    def test_metrics_reject_single_class_auc(self):
        with self.assertRaisesRegex(ValueError, 'both classes'):
            roc_auc([0, 0], [.1, .2])


if __name__ == '__main__':
    unittest.main()
