"""Tests for the frozen incident-versus-routine placebo audit."""

from copy import deepcopy
from pathlib import Path
import json
import tempfile
import unittest

import numpy as np

from experiments.chronological.audit_matched_placebo import (
    compute_divergences, load_protocol, placebo_gate, select_nonoverlap_triples,
)


PROTOCOL = (Path(__file__).resolve().parents[1] / 'experiments/chronological' /
            'matched_placebo_audit_v5c.json')


class ProtocolTests(unittest.TestCase):
    def test_protocol_freezes_late_gate_and_prohibits_training_and_test(self):
        protocol = load_protocol(PROTOCOL)
        self.assertEqual(protocol['placebo_gate']['primary_horizon'], 'late_H7_H12')
        self.assertTrue(protocol['information_boundary']['test_split_prohibited'])
        self.assertTrue(protocol['information_boundary']['model_training_prohibited'])
        self.assertFalse(protocol['main_training_ready'])

    def test_protocol_rejects_changed_primary_horizon(self):
        protocol = json.loads(PROTOCOL.read_text(encoding='utf-8'))
        protocol['placebo_gate']['primary_horizon'] = 'early_H1_H6'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bad.json'
            path.write_text(json.dumps(protocol), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'gate changed'):
                load_protocol(path)

    def test_protocol_rejects_outcome_based_nonoverlap_selection(self):
        protocol = json.loads(PROTOCOL.read_text(encoding='utf-8'))
        protocol['nonoverlap_sensitivity']['outcomes_used_for_selection'] = True
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bad.json'
            path.write_text(json.dumps(protocol), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'sensitivity changed'):
                load_protocol(path)


class EstimandTests(unittest.TestCase):
    def test_symmetric_incident_and_routine_divergence_formula(self):
        incident = np.zeros((26, 2))
        primary = np.ones((26, 2))
        secondary = np.full((26, 2), 3.0)
        result = compute_divergences(incident, primary, secondary, slice(9, 12))
        self.assertTrue(np.all(result['incident_divergence'] == 2.0))
        self.assertTrue(np.all(result['routine_divergence'] == 2.0))
        self.assertTrue(np.all(result['excess'] == 0.0))

    def test_each_divergence_subtracts_its_own_baseline(self):
        incident = np.zeros((26, 1))
        primary = np.ones((26, 1))
        secondary = np.full((26, 1), 3.0)
        incident[20:] = 10
        result = compute_divergences(incident, primary, secondary, slice(9, 12))
        self.assertTrue(np.all(result['routine_change'] == 0))
        self.assertTrue(np.all(result['incident_change'][:20] == 0))
        self.assertTrue(np.all(result['excess'][20:] == 6.0))

    def test_nonfinite_values_are_rejected(self):
        values = np.zeros((26, 1))
        invalid = values.copy()
        invalid[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, 'finite'):
            compute_divergences(invalid, values, values, slice(9, 12))


class NonoverlapTests(unittest.TestCase):
    @staticmethod
    def row(index, sample, start):
        return {
            'control_index': str(index), 'positive_sample_index': str(sample),
            'positive_t0': f'2023-03-{sample:02d}T12:00:00', 'x_start': start,
        }

    def test_one_exclusion_set_covers_all_three_window_sides(self):
        protocol = load_protocol(PROTOCOL)
        second = [
            self.row(0, 1, '2023-03-22T10:55:00'),
            self.row(1, 2, '2023-03-22T11:55:00'),
        ]
        primary = {
            1: self.row(10, 1, '2023-03-15T10:55:00'),
            2: self.row(11, 2, '2023-03-15T11:55:00'),
        }
        positive = {
            1: {'x_start': '2023-03-01T10:55:00'},
            2: {'x_start': '2023-03-01T11:55:00'},
        }
        selected = select_nonoverlap_triples(second, primary, positive, protocol['window'])
        self.assertEqual(selected, {0})

    def test_internal_window_overlap_is_rejected(self):
        protocol = load_protocol(PROTOCOL)
        second = [self.row(0, 1, '2023-03-15T10:55:00')]
        primary = {1: self.row(10, 1, '2023-03-15T10:55:00')}
        positive = {1: {'x_start': '2023-03-01T10:55:00'}}
        self.assertFalse(select_nonoverlap_triples(
            second, primary, positive, protocol['window']))


class GateTests(unittest.TestCase):
    @staticmethod
    def summaries(main_effect=.06, main_low=.01, strict_effect=.03,
                  strict_count_train=350, strict_count_val=80):
        result = {}
        for split, count in (('train', strict_count_train), ('val', strict_count_val)):
            result[split] = {
                'all_common_triples': {
                    'late_excess_in_train_std': main_effect,
                    'late_excess_ci_low_raw': main_low,
                },
                'strict_nonoverlap': {
                    'late_excess_in_train_std': strict_effect,
                    'late_excess_divergence_raw': strict_effect,
                    'triples': count,
                },
            }
        return result

    def test_gate_passes_only_positive_main_and_sensitivity_evidence(self):
        checks, ready = placebo_gate(self.summaries(), load_protocol(PROTOCOL))
        self.assertTrue(ready)
        self.assertTrue(all(all(value.values()) for value in checks.values()))

    def test_gate_rejects_main_interval_crossing_zero(self):
        _, ready = placebo_gate(
            self.summaries(main_low=-.01), load_protocol(PROTOCOL))
        self.assertFalse(ready)

    def test_gate_rejects_small_strict_subset(self):
        _, ready = placebo_gate(
            self.summaries(strict_count_val=79), load_protocol(PROTOCOL))
        self.assertFalse(ready)


if __name__ == '__main__':
    unittest.main()
