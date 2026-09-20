"""Tests for the fixed-primary, X-only second-control assignment."""

import json
from pathlib import Path
import tempfile
import unittest

from experiments.chronological.assign_second_matched_controls import (
    assign_secondary, eligible_value, load_protocol,
)


PROTOCOL = (Path(__file__).resolve().parents[1] /
            'experiments/chronological/second_matched_control_v5a.json')


def edge(sample, candidate, distance, eligible=True, split='train'):
    return {
        'split': split, 'sample_index': str(sample), 'incident_id': f'i-{sample}',
        'incident_t0': f'2023-01-{sample:02d}T12:00:00',
        'candidate_t0': candidate, 'freeway': '4', 'direction': 'E',
        'incident_postmile': '10', 'affected_node_count': '1',
        'absolute_day_distance': '7', 'pairwise_valid_count': '12',
        'pairwise_total_count': '12', 'pairwise_valid_overlap_fraction': '1',
        'missing_pattern_mismatch_fraction': '0', 'x_distance': str(distance),
        'eligible': eligible, 'positive_history_mean': '10',
        'control_history_mean': '10', 'positive_last_step_mean': '10',
        'control_last_step_mean': '10', 'positive_late3_minus_early3': '0',
        'control_late3_minus_early3': '0',
    }


class ProtocolTests(unittest.TestCase):
    def test_protocol_prohibits_outcomes_and_freezes_primary(self):
        protocol = load_protocol(PROTOCOL)
        boundary = protocol['selection_boundary']
        self.assertTrue(boundary['primary_assignment_immutable'])
        self.assertTrue(boundary['forecast_Y_prohibited'])
        self.assertTrue(boundary['outcome_audit_results_prohibited'])
        self.assertFalse(protocol['main_training_ready'])

    def test_protocol_rejects_primary_candidate_reuse(self):
        protocol = json.loads(PROTOCOL.read_text(encoding='utf-8'))
        protocol['selection_boundary']['all_primary_candidate_centers_excluded'] = False
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bad.json'
            path.write_text(json.dumps(protocol), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'boundary'):
                load_protocol(path)


class AssignmentTests(unittest.TestCase):
    def test_csv_boolean_parser_is_strict(self):
        self.assertTrue(eligible_value('True'))
        self.assertFalse(eligible_value('false'))
        with self.assertRaisesRegex(ValueError, 'eligible'):
            eligible_value('yes')

    def test_all_primary_centers_are_excluded_for_every_positive(self):
        primary = [
            edge(1, '2023-01-08T12:00:00', .1),
            edge(2, '2023-01-15T12:00:00', .1),
        ]
        candidates = [
            *primary,
            edge(1, '2023-01-15T12:00:00', .2),
            edge(1, '2023-01-22T12:00:00', .3),
            edge(2, '2023-01-08T12:00:00', .2),
            edge(2, '2023-01-29T12:00:00', .3),
        ]
        secondary, unmatched, _ = assign_secondary(primary, candidates)
        self.assertEqual({row['candidate_t0'] for row in secondary}, {
            '2023-01-22T12:00:00', '2023-01-29T12:00:00'})
        self.assertFalse(unmatched)

    def test_augmenting_path_maximizes_secondary_cardinality(self):
        primary = [
            edge(1, '2023-01-01T12:00:00', .1),
            edge(2, '2023-01-08T12:00:00', .1),
        ]
        shared = '2023-01-22T12:00:00'
        unique = '2023-01-29T12:00:00'
        candidates = [
            *primary, edge(1, shared, .1), edge(1, unique, .2),
            edge(2, shared, .1),
        ]
        secondary, unmatched, _ = assign_secondary(primary, candidates)
        self.assertEqual(len(secondary), 2)
        self.assertEqual(len({row['candidate_t0'] for row in secondary}), 2)
        self.assertFalse(unmatched)

    def test_ineligible_edges_never_become_secondary_controls(self):
        primary = [edge(1, '2023-01-01T12:00:00', .1)]
        candidates = [*primary, edge(1, '2023-01-08T12:00:00', .2, eligible='False')]
        secondary, unmatched, counts = assign_secondary(primary, candidates)
        self.assertFalse(secondary)
        self.assertEqual(counts[('train', 1)], 0)
        self.assertEqual(unmatched[0]['unmatched_reason'], 'no_unused_eligible_candidate')

    def test_capacity_conflict_is_reported_separately(self):
        primary = [
            edge(1, '2023-01-01T12:00:00', .1),
            edge(2, '2023-01-08T12:00:00', .1),
        ]
        shared = '2023-01-22T12:00:00'
        candidates = [*primary, edge(1, shared, .2), edge(2, shared, .2)]
        secondary, unmatched, _ = assign_secondary(primary, candidates)
        self.assertEqual(len(secondary), 1)
        self.assertEqual(unmatched[0]['unmatched_reason'],
                         'secondary_candidate_capacity_conflict')

    def test_original_preference_rank_includes_the_primary_candidate(self):
        primary = [edge(1, '2023-01-01T12:00:00', .1)]
        candidates = [
            *primary, edge(1, '2023-01-08T12:00:00', .2),
            edge(1, '2023-01-15T12:00:00', .3),
        ]
        secondary, _, _ = assign_secondary(primary, candidates)
        self.assertEqual(secondary[0]['original_candidate_preference_rank'], 2)


if __name__ == '__main__':
    unittest.main()
