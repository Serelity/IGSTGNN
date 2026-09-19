"""Tests for traffic-X scoring and unique matched-control assignment."""

from datetime import datetime
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.chronological.audit_matched_controls import sha256
from experiments.chronological.score_matched_controls import (
    ASSIGNMENT_FIELDS, SCORE_FIELDS, balance_summary, candidate_history,
    load_protocol, maximum_preference_matching, score_histories, verify_inputs,
)


PROTOCOL = (Path(__file__).resolve().parents[1] /
            'experiments/chronological/matched_nonincident_x_v2.json')


def edge(sample, candidate, distance, eligible=True):
    return {
        'split': 'train', 'sample_index': sample, 'candidate_t0': candidate,
        'x_distance': distance, 'missing_pattern_mismatch_fraction': 0.0,
        'absolute_day_distance': 7, 'eligible': eligible,
    }


class ProtocolTests(unittest.TestCase):
    def test_frozen_protocol_prohibits_leakage_inputs(self):
        protocol = load_protocol(PROTOCOL)
        self.assertTrue({'forecast_Y', 'test_split', 'incident_description', 'incident_type'} <=
                        set(protocol['x_similarity']['prohibited_inputs']))
        self.assertFalse(protocol['main_training_ready'])
        self.assertFalse(protocol['assignment']['global_cost_optimality_claimed'])

    def test_protocol_missing_any_leakage_prohibition_is_rejected(self):
        protocol = json.loads(PROTOCOL.read_text(encoding='utf-8'))
        protocol['x_similarity']['prohibited_inputs'].remove('incident_type')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bad.json'
            path.write_text(json.dumps(protocol), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'prohibit Y, test, description, and type'):
                load_protocol(path)

    def test_protocol_cannot_change_history_spacing(self):
        protocol = json.loads(PROTOCOL.read_text(encoding='utf-8'))
        protocol['x_similarity']['step_minutes'] = 10
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bad.json'
            path.write_text(json.dumps(protocol), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'five-minute spacing'):
                load_protocol(path)

    def test_input_fingerprint_drift_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, audit = root / 'data', root / 'audit'
            data.mkdir()
            audit.mkdir()
            sensors = root / 'sensors.csv'
            for name in ('summary.json', 'scaler.json', 'station_ids.npy'):
                (data / name).write_bytes(name.encode())
            sensors.write_text('sensors', encoding='utf-8')
            (audit / 'candidate_pairs.csv').write_text('pairs', encoding='utf-8')
            (audit / 'sample_audit.csv').write_text('samples', encoding='utf-8')
            audit_summary = {
                'status': 'MATCHED_NONINCIDENT_CANDIDATE_AUDIT_PASS',
                'protocol_id': 'audit-v1', 'forecast_Y_used_for_matching': False,
                'test_split_read': False,
            }
            (audit / 'summary.json').write_text(json.dumps(audit_summary), encoding='utf-8')
            protocol = {
                'data_inputs': {
                    'summary_sha256': sha256(data / 'summary.json'),
                    'scaler_sha256': sha256(data / 'scaler.json'),
                    'station_ids_sha256': sha256(data / 'station_ids.npy'),
                    'sensors_sha256': sha256(sensors),
                },
                'candidate_audit': {
                    'protocol_id': 'audit-v1',
                    'summary_sha256': sha256(audit / 'summary.json'),
                    'candidate_pairs_sha256': sha256(audit / 'candidate_pairs.csv'),
                    'sample_audit_sha256': sha256(audit / 'sample_audit.csv'),
                },
            }
            verify_inputs(data, audit, sensors, protocol)
            (audit / 'candidate_pairs.csv').write_text('changed', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'fingerprints differ'):
                verify_inputs(data, audit, sensors, protocol)


class ScoringTests(unittest.TestCase):
    def test_negative_and_nonfinite_values_do_not_enter_distance(self):
        positive = np.arange(12, dtype=np.float64)[:, None]
        control = positive + 2
        control[0, 0] = -1
        result = score_histories(positive, control, train_std=2, minimum_overlap=0.9)
        self.assertEqual(result['pairwise_valid_count'], 11)
        self.assertAlmostEqual(result['pairwise_valid_overlap_fraction'], 11 / 12)
        self.assertAlmostEqual(result['missing_pattern_mismatch_fraction'], 1 / 12)
        self.assertAlmostEqual(result['x_distance'], 1.0)
        self.assertTrue(result['eligible'])

        control[1, 0] = np.nan
        result = score_histories(positive, control, train_std=2, minimum_overlap=0.9)
        self.assertFalse(result['eligible'])

    def test_candidate_history_crosses_month_without_shifting_slots(self):
        january = np.zeros((2, 31 * 288), dtype=np.float32)
        february = np.zeros((2, 28 * 288), dtype=np.float32)
        january[1] = np.arange(31 * 288)
        february[1] = 10000 + np.arange(28 * 288)
        protocol = load_protocol(PROTOCOL)
        values = candidate_history(
            datetime(2023, 2, 1, 0, 30), np.asarray([1]),
            {1: january, 2: february}, protocol)[:, 0]
        self.assertEqual(values[:7].tolist(), list(range(31 * 288 - 7, 31 * 288)))
        self.assertEqual(values[7:].tolist(), list(range(10000, 10005)))

    def test_all_nonfinite_balance_is_json_safe_and_not_observed(self):
        row = {
            'positive_history_mean': float('nan'), 'control_history_mean': float('nan'),
            'positive_last_step_mean': float('nan'), 'control_last_step_mean': float('nan'),
            'positive_late3_minus_early3': float('nan'),
            'control_late3_minus_early3': float('nan'),
        }
        result = balance_summary([row])
        self.assertTrue(all(item['standardized_mean_difference'] is None
                            for item in result.values()))
        json.dumps(result, allow_nan=False)


class AssignmentTests(unittest.TestCase):
    def test_augmenting_path_recovers_maximum_cardinality(self):
        edges = [
            edge(1, 'A', 0.1), edge(1, 'B', 0.2),
            edge(2, 'A', 0.1), edge(2, 'C', 0.2),
            edge(3, 'A', 0.1), edge(3, 'C', 0.2),
        ]
        matched = maximum_preference_matching(edges)
        self.assertEqual(len(matched), 3)
        self.assertEqual(len({row['candidate_t0'] for row in matched.values()}), 3)

    def test_matching_is_deterministic_and_ignores_ineligible_edges(self):
        edges = [
            edge(2, 'A', 0.2), edge(1, 'A', 0.1),
            edge(1, 'B', 0.3), edge(2, 'B', 0.1), edge(3, 'C', 0.0, False),
        ]
        forward = maximum_preference_matching(edges)
        reverse = maximum_preference_matching(list(reversed(edges)))
        selected = lambda result: {
            key: row['candidate_t0'] for key, row in result.items()
        }
        self.assertEqual(selected(forward), selected(reverse))
        self.assertNotIn(('train', 3), forward)

    def test_assignment_csv_contract_contains_every_score_field_once(self):
        self.assertEqual(ASSIGNMENT_FIELDS, SCORE_FIELDS + ['candidate_preference_rank'])
        self.assertEqual(len(ASSIGNMENT_FIELDS), len(set(ASSIGNMENT_FIELDS)))


if __name__ == '__main__':
    unittest.main()
