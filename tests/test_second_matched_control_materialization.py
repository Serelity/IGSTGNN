"""Tests for frozen second-control materialization and triple alignment."""

from copy import deepcopy
from pathlib import Path
import json
import tempfile
import unittest

from experiments.chronological.materialize_second_matched_controls import (
    combined_overlap_diagnostics, load_protocol, normalize_assignments,
    verify_primary_alignment,
)


PROTOCOL = (Path(__file__).resolve().parents[1] / 'experiments/chronological' /
            'second_matched_control_materialize_v5b.json')


def positive(sample, x_start):
    return {
        'sample_index': str(sample), 'incident_id': f'i-{sample}',
        't0': f'2023-03-{sample:02d}T12:00:00', 'x_start': x_start,
    }


def control(sample, candidate, x_start):
    return {
        'positive_sample_index': str(sample), 'incident_id': f'i-{sample}',
        'positive_t0': f'2023-03-{sample:02d}T12:00:00',
        'candidate_t0': candidate, 'x_start': x_start, 'freeway': '4',
        'direction': 'E', 'affected_node_count': '1',
    }


class ProtocolTests(unittest.TestCase):
    def test_protocol_freezes_assignment_before_y_and_prohibits_v4_results(self):
        protocol = load_protocol(PROTOCOL)
        boundary = protocol['information_boundary']
        self.assertTrue(boundary['secondary_assignment_frozen_before_materialization'])
        self.assertFalse(boundary['future_Y_may_change_assignment'])
        self.assertTrue(boundary['test_split_prohibited'])
        self.assertTrue(boundary['v4_outcome_results_prohibited'])
        self.assertFalse(protocol['main_training_ready'])

    def test_protocol_rejects_y_dependent_reassignment(self):
        protocol = json.loads(PROTOCOL.read_text(encoding='utf-8'))
        protocol['information_boundary']['future_Y_may_change_assignment'] = True
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bad.json'
            path.write_text(json.dumps(protocol), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Future Y'):
                load_protocol(path)


class AlignmentTests(unittest.TestCase):
    def setUp(self):
        self.rows = {
            'train': [control(1, '2023-03-22T12:00:00', '2023-03-22T10:55:00')],
            'val': [],
        }
        self.primary = {
            'train': [control(1, '2023-03-15T12:00:00', '2023-03-15T10:55:00')],
            'val': [],
        }
        self.positive = {
            'train': [positive(1, '2023-03-01T10:55:00')], 'val': [],
        }

    def test_normalization_preserves_secondary_rank(self):
        rows = normalize_assignments([{'original_candidate_preference_rank': '3'}])
        self.assertEqual(rows[0]['candidate_preference_rank'], '3')

    def test_primary_and_secondary_centers_are_disjoint(self):
        self.rows['train'][0]['freeway'] = 4
        self.rows['train'][0]['affected_node_count'] = 1
        result = verify_primary_alignment(self.rows, self.primary, self.positive)
        self.assertEqual(result['train']['common_triples'], 1)
        self.assertEqual(result['train']['primary_secondary_candidate_t0_overlap'], 0)

    def test_identity_mismatch_is_rejected(self):
        rows = deepcopy(self.rows)
        rows['train'][0]['incident_id'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'identities differ'):
            verify_primary_alignment(rows, self.primary, self.positive)

    def test_duplicate_secondary_center_is_rejected(self):
        rows = deepcopy(self.rows)
        duplicate = deepcopy(rows['train'][0])
        duplicate['positive_sample_index'] = '2'
        duplicate['incident_id'] = 'i-2'
        duplicate['positive_t0'] = '2023-03-02T12:00:00'
        rows['train'].append(duplicate)
        primary = deepcopy(self.primary)
        primary['train'].append(control(
            2, '2023-03-16T12:00:00', '2023-03-16T10:55:00'))
        positive_rows = deepcopy(self.positive)
        positive_rows['train'].append(positive(2, '2023-03-02T10:55:00'))
        with self.assertRaisesRegex(ValueError, 'reuse'):
            verify_primary_alignment(rows, primary, positive_rows)

    def test_combined_overlap_reports_strict_triple_subset(self):
        protocol = load_protocol(PROTOCOL)
        second = deepcopy(self.rows)
        second['train'].append(control(
            2, '2023-03-22T13:00:00', '2023-03-22T11:55:00'))
        primary = deepcopy(self.primary)
        primary['train'].append(control(
            2, '2023-03-15T13:00:00', '2023-03-15T11:55:00'))
        positive_rows = deepcopy(self.positive)
        positive_rows['train'].append(positive(2, '2023-03-01T11:55:00'))
        result = combined_overlap_diagnostics(second, primary, positive_rows, protocol)
        train = result['splits']['train']
        self.assertEqual(train['triples_with_internal_source_slot_overlap'], 0)
        self.assertEqual(train['triples_sharing_source_slots_with_other_triples'], 2)
        self.assertEqual(train['strict_three_window_nonoverlap_triples'], 1)
        self.assertEqual(result['cross_split_combined_source_slot_overlap'], 0)

    def test_internal_overlap_is_excluded_from_strict_subset(self):
        protocol = load_protocol(PROTOCOL)
        second = deepcopy(self.rows)
        second['train'][0]['x_start'] = self.primary['train'][0]['x_start']
        result = combined_overlap_diagnostics(
            second, self.primary, self.positive, protocol)['splits']['train']
        self.assertEqual(result['triples_with_internal_source_slot_overlap'], 1)
        self.assertEqual(result['strict_three_window_nonoverlap_triples'], 0)


if __name__ == '__main__':
    unittest.main()
