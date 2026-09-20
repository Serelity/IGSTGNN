"""Tests for post-assignment matched-control materialization."""

from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.chronological.materialize_matched_controls import (
    build_control_rows, build_month_plan, control_window_slots, load_protocol,
    overlap_diagnostics, raw_quality, split_for_slot, verify_assignment_scores,
)
from experiments.chronological.score_matched_controls import score_histories


PROTOCOL = (Path(__file__).resolve().parents[1] /
            'experiments/chronological/matched_nonincident_materialize_v3.json')


def assignment(split, sample, positive_t0, candidate_t0, road=4, direction='E'):
    positive = np.arange(12, dtype=np.float64)[:, None]
    control = positive + 1
    scores = score_histories(positive, control, train_std=2, minimum_overlap=0.9)
    return {
        'split': split, 'sample_index': str(sample), 'incident_id': f'i-{sample}',
        'incident_t0': positive_t0, 'candidate_t0': candidate_t0,
        'freeway': str(road), 'direction': direction, 'incident_postmile': '10',
        'affected_node_count': '1', 'candidate_preference_rank': '1',
        **{key: str(value) for key, value in scores.items()},
    }


def manifest(split, sample, t0):
    return {
        'split': split, 'sample_index': str(sample), 'incident_id': f'i-{sample}',
        't0': t0, 'source_version': '8',
    }


class ProtocolTests(unittest.TestCase):
    def test_protocol_freezes_assignment_before_y_and_prohibits_test(self):
        protocol = load_protocol(PROTOCOL)
        boundary = protocol['selection_boundary']
        self.assertTrue(boundary['assignment_frozen_before_materialization'])
        self.assertFalse(boundary['future_Y_may_change_assignment'])
        self.assertTrue(boundary['test_split_prohibited'])
        self.assertFalse(protocol['main_training_ready'])

    def test_protocol_allowing_y_to_change_assignment_is_rejected(self):
        protocol = json.loads(PROTOCOL.read_text(encoding='utf-8'))
        protocol['selection_boundary']['future_Y_may_change_assignment'] = True
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bad.json'
            path.write_text(json.dumps(protocol), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Future Y'):
                load_protocol(path)


class WindowTests(unittest.TestCase):
    def setUp(self):
        self.protocol = load_protocol(PROTOCOL)

    def test_full_window_retains_x_gap_and_y_slot_alignment(self):
        t0 = datetime(2023, 2, 1, 0, 30)
        slots = control_window_slots(t0, self.protocol)
        self.assertEqual(len(slots), 26)
        self.assertEqual(slots[0], datetime(2023, 1, 31, 23, 25))
        self.assertEqual(slots[11], datetime(2023, 2, 1, 0, 20))
        self.assertEqual(slots[12], datetime(2023, 2, 1, 0, 25))
        self.assertEqual(slots[13], t0)
        self.assertEqual(slots[14], datetime(2023, 2, 1, 0, 35))
        self.assertEqual(slots[25], datetime(2023, 2, 1, 1, 30))

    def test_split_boundary_window_is_rejected_by_row_builder(self):
        protocol = deepcopy(self.protocol)
        protocol['assignment_input']['expected_assigned_samples'] = {'train': 1, 'val': 0}
        positive_t0 = '2023-08-31T23:30:00'
        assignments = [assignment('train', 1, positive_t0, positive_t0)]
        manifests = {'train': [manifest('train', 1, positive_t0)], 'val': []}
        axes = (np.asarray([4]), np.asarray(['E']), np.asarray([10.0]))
        with self.assertRaisesRegex(ValueError, 'crosses its frozen chronological split'):
            build_control_rows(assignments, manifests, axes, protocol)

    def test_month_plan_preserves_cross_month_slots(self):
        rows = {'train': [], 'val': [{'control_index': 0, 'candidate_t0':
                                      '2023-10-01T00:30:00'}]}
        plan = build_month_plan(rows, self.protocol)
        self.assertEqual(len(plan[9]['val']['window_positions']), 7)
        self.assertEqual(len(plan[10]['val']['window_positions']), 19)
        self.assertEqual(split_for_slot(datetime(2023, 11, 1), self.protocol), None)


class ConstructionTests(unittest.TestCase):
    def setUp(self):
        self.protocol = load_protocol(PROTOCOL)

    def test_rows_and_masks_follow_positive_identity_and_sensor_scope(self):
        protocol = deepcopy(self.protocol)
        protocol['assignment_input']['expected_assigned_samples'] = {'train': 1, 'val': 1}
        assignments = [
            assignment('train', 1, '2023-03-01T12:00:00', '2023-03-08T12:00:00'),
            assignment('val', 2, '2023-09-05T12:00:00', '2023-09-12T12:00:00'),
        ]
        manifests = {
            'train': [manifest('train', 1, '2023-03-01T12:00:00')],
            'val': [manifest('val', 2, '2023-09-05T12:00:00')],
        }
        axes = (np.asarray([4, 4, 24]), np.asarray(['E', 'E', 'W']),
                np.asarray([10.0, 25.0, 10.0]))
        rows, masks = build_control_rows(assignments, manifests, axes, protocol)
        self.assertEqual(rows['train'][0]['positive_sample_index'], 1)
        self.assertEqual(rows['train'][0]['y_start'], '2023-03-08T12:05:00')
        self.assertEqual(masks['train'].tolist(), [[True, False, False]])
        self.assertEqual(masks['val'].sum(), 1)

    def test_duplicate_candidate_timestamp_is_rejected(self):
        protocol = deepcopy(self.protocol)
        protocol['assignment_input']['expected_assigned_samples'] = {'train': 2, 'val': 0}
        candidate = '2023-03-08T12:00:00'
        assignments = [
            assignment('train', 1, '2023-03-01T12:00:00', candidate),
            assignment('train', 2, '2023-03-02T12:00:00', candidate),
        ]
        manifests = {'train': [manifest('train', 1, '2023-03-01T12:00:00'),
                               manifest('train', 2, '2023-03-02T12:00:00')], 'val': []}
        axes = (np.asarray([4]), np.asarray(['E']), np.asarray([10.0]))
        with self.assertRaisesRegex(ValueError, 'reuses a positive or candidate'):
            build_control_rows(assignments, manifests, axes, protocol)

    def test_overlap_diagnostics_report_shared_slots_without_cross_split_leakage(self):
        rows = {
            'train': [
                {'candidate_t0': '2023-03-08T12:00:00'},
                {'candidate_t0': '2023-03-08T13:00:00'},
            ],
            'val': [{'candidate_t0': '2023-09-08T12:00:00'}],
        }
        result = overlap_diagnostics(rows, self.protocol)
        self.assertEqual(result['maximum_candidate_t0_reuse'], 1)
        self.assertEqual(result['cross_split_source_slot_overlap'], 0)
        self.assertEqual(result['splits']['train']['windows_sharing_at_least_one_source_slot'], 2)
        self.assertGreater(result['splits']['train']['source_slots_used_more_than_once'], 0)

    def test_assignment_scores_are_reproduced_from_materialized_x(self):
        assignments = [
            assignment('train', 1, '2023-03-01T12:00:00', '2023-03-08T12:00:00'),
            assignment('val', 2, '2023-09-01T12:00:00', '2023-09-08T12:00:00'),
        ]
        positive = np.zeros((1, 26, 1), dtype=np.float32)
        positive[0, :12, 0] = np.arange(12)
        control = positive.copy()
        control[0, :12, 0] += 1
        rows = {
            'train': [{'split': 'train', 'positive_sample_index': 1, 'control_index': 0}],
            'val': [{'split': 'val', 'positive_sample_index': 2, 'control_index': 0}],
        }
        masks = {'train': np.ones((1, 1), bool), 'val': np.ones((1, 1), bool)}
        result, passed, maximum = verify_assignment_scores(
            assignments, rows, masks, {'train': control, 'val': control},
            {'train': positive, 'val': positive}, {'train': {1: 0}, 'val': {2: 0}},
            train_std=2, protocol=self.protocol)
        self.assertTrue(passed)
        self.assertEqual(maximum, 0.0)
        self.assertTrue(result['train']['all_reproduced'])

    def test_raw_quality_preserves_negative_and_nonfinite_diagnostics(self):
        array = np.ones((1, 26, 2), dtype=np.float32)
        array[0, 0, 0] = -1
        array[0, 14, 0] = np.nan
        quality = raw_quality(array, np.asarray([[True, False]]))
        self.assertEqual(quality['X']['full_graph_negative'], 1)
        self.assertEqual(quality['Y']['full_graph_nonfinite'], 1)
        self.assertLess(quality['Y']['affected_valid_fraction'], 1.0)


if __name__ == '__main__':
    unittest.main()
