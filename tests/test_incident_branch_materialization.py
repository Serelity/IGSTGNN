"""Tests for frozen-A incident-branch counterfactual materialization."""

from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from experiments.chronological.materialize_incident_branch import (
    COHORTS, MatchedCounterfactualDataset, forecast_clock, load_protocol,
)


REPO = Path(__file__).resolve().parents[1]
PROTOCOL = (REPO / 'experiments/chronological' /
            'incident_branch_materialize_v6a.json')


class ProtocolTests(unittest.TestCase):
    def test_cli_exposes_bounded_engineering_check(self):
        result = subprocess.run(
            [sys.executable, 'experiments/chronological/materialize_incident_branch.py',
             '--help'], cwd=REPO, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--check', result.stdout)

    def test_protocol_freezes_checkpoint_cohort_and_information_boundary(self):
        protocol = load_protocol(PROTOCOL)
        self.assertEqual(protocol['expected_common_triples'], {'train': 3106, 'val': 618})
        self.assertEqual(protocol['expected_positive_samples'], {'train': 3604, 'val': 917})
        self.assertEqual(protocol['checkpoint']['best_epoch'], 99)
        self.assertEqual(protocol['checkpoint']['best_validation_mae'],
                         22.686666155323852)
        self.assertEqual(protocol['checkpoint']['best_model_sha256'],
                         'b0c712ad9c00007417ccc6ea6268f373852d04063efba2d15d3f49f5497b8e13')
        self.assertTrue(protocol['information_boundary']['test_split_prohibited'])
        self.assertTrue(protocol['information_boundary']['gradient_computation_prohibited'])
        self.assertEqual(protocol['output_schema']['cohorts'], list(COHORTS))
        self.assertEqual(
            protocol['candidate_mask_compatibility']['expected_frozen_only_pairs']['train'],
            [{
                'positive_sample_index': 1542,
                'station_id': 402510,
                'postmile_delta_miles': 10.0,
                'reason': 'positive_10_mile_boundary_has_zero_normalized_similarity',
            }])

    def test_protocol_rejects_control_clock_drift(self):
        protocol = json.loads(PROTOCOL.read_text(encoding='utf-8'))
        protocol['counterfactual_modes']['control_pseudo_event']['forecast_clock'] = \
            'from positive incident'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'changed.json'
            path.write_text(json.dumps(protocol), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'construction changed'):
                load_protocol(path)


class CounterfactualConstructionTests(unittest.TestCase):
    @staticmethod
    def validation_fixture(frozen_mask, distances):
        dataset = MatchedCounterfactualDataset.__new__(MatchedCounterfactualDataset)
        dataset.split = 'train'
        dataset.station_ids = np.array([10, 20])
        positive_row = {
            'sample_index': '7', 'incident_id': 'incident-7', 'split': 'train',
            't0': '2023-09-18T13:25:00',
        }
        primary_row = {
            'positive_sample_index': '7', 'control_index': '0',
            'incident_id': 'incident-7', 'split': 'train',
            'positive_t0': '2023-09-18T13:25:00',
        }
        secondary_row = dict(primary_row)
        dataset.positive = type('Positive', (), {
            'rows': [positive_row],
            'context': {'distances': np.asarray(distances, dtype=np.float32)[None]},
        })()
        dataset.positive_positions = {7: 0}
        dataset.primary_rows = [primary_row]
        dataset.primary_by_sample = {7: primary_row}
        dataset.secondary_rows = [secondary_row]
        dataset.primary_masks = np.asarray([frozen_mask], dtype=bool)
        dataset.secondary_masks = np.asarray([frozen_mask], dtype=bool)
        return dataset

    def test_forecast_clock_uses_candidate_timestamp(self):
        tod, dow = forecast_clock('2023-09-18T13:25:00')
        self.assertEqual(int(tod), 161)
        self.assertEqual(int(dow), 1)

    def test_control_uses_control_flow_clock_and_positive_event_context(self):
        dataset = MatchedCounterfactualDataset.__new__(MatchedCounterfactualDataset)
        dataset.split = 'train'
        dataset.cohort = 'primary_control'
        dataset.scaler = {
            'mean': 0., 'std': 1., 'node_fill_mean': [0., 0.],
        }
        dataset.station_ids = np.array([10, 20])
        positive_row = {
            'sample_index': '7', 'incident_id': 'incident-7', 'split': 'train',
            't0': '2023-09-18T13:25:00', 'x_start': '2023-09-18T12:20:00',
        }
        primary_row = {
            'positive_sample_index': '7', 'control_index': '0',
            'candidate_t0': '2023-09-25T08:10:00',
            'x_start': '2023-09-25T07:05:00',
        }
        secondary_row = {'positive_sample_index': '7'}
        distances = np.array([[1., 2., 3.], [0., 0., 0.]], dtype=np.float32)
        dataset.positive = type('Positive', (), {
            'rows': [positive_row],
            'flow': np.full((1, 26, 2), 5., dtype=np.float32),
            'context': {
                'report_age_minutes': np.array([4.], dtype=np.float32),
                'distances': distances[None],
            },
        })()
        dataset.positive_positions = {7: 0}
        dataset.primary_rows = [primary_row]
        dataset.primary_by_sample = {7: primary_row}
        dataset.secondary_rows = [secondary_row]
        dataset.primary_flow = np.full((1, 26, 2), 9., dtype=np.float32)
        dataset.secondary_flow = np.full((1, 26, 2), 11., dtype=np.float32)
        item = dataset[0]
        self.assertTrue(np.all(item['x'][..., 0] == 9.))
        self.assertTrue(np.all(item['y_flow'] == 9.))
        self.assertEqual(float(item['incident']['report_age_minutes']), 4.)
        np.testing.assert_array_equal(item['incident']['distances'], distances)
        expected_tod, expected_dow = forecast_clock(primary_row['candidate_t0'])
        self.assertEqual(item['incident']['forecast_tod'], expected_tod)
        self.assertEqual(item['incident']['forecast_dow'], expected_dow)
        np.testing.assert_array_equal(item['candidate_mask'], [True, False])
        first_time = datetime.fromisoformat(primary_row['x_start'])
        self.assertAlmostEqual(item['x'][0, 0, 1],
                               (first_time.hour * 12 + first_time.minute // 5) / 288)

    def test_known_frozen_only_boundary_is_accepted_but_not_a_model_candidate(self):
        dataset = self.validation_fixture(
            [True, True], [[1., 0., 0.], [0., 0., 0.]])
        dataset._validate_rows([{'positive_sample_index': 7, 'station_id': 20}])
        self.assertEqual(dataset.candidate_mask_compatibility, {
            'model_connected_outside_frozen_node_references': 0,
            'frozen_only_node_references': 1,
            'frozen_only_pairs': [{'positive_sample_index': 7, 'station_id': 20}],
        })

    def test_model_candidate_outside_frozen_mask_is_rejected(self):
        dataset = self.validation_fixture(
            [True, False], [[1., 0., 0.], [0., 1., 0.]])
        with self.assertRaisesRegex(ValueError, 'outside the frozen affected mask'):
            dataset._validate_rows([])


if __name__ == '__main__':
    unittest.main()
