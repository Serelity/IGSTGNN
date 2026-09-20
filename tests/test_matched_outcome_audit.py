"""Tests for the frozen matched incident/control outcome audit."""

from datetime import datetime
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.chronological.audit_matched_outcomes import (
    cluster_bootstrap_mean, phenomenon_gate, load_protocol, phase_for_step,
    select_nonoverlap, source_slots,
)


PROTOCOL = (Path(__file__).resolve().parents[1] /
            'experiments/chronological/matched_outcome_audit_v4.json')


class ProtocolTests(unittest.TestCase):
    def test_protocol_freezes_observational_boundary_and_prohibits_test(self):
        protocol = load_protocol(PROTOCOL)
        self.assertTrue(protocol['information_boundary']['test_split_prohibited'])
        self.assertTrue(protocol['information_boundary']['outcomes_may_not_change_matching'])
        self.assertFalse(protocol['estimand']['causal_effect_claimed'])
        self.assertFalse(protocol['main_training_ready'])

    def test_protocol_rejects_changed_late_horizon(self):
        protocol = json.loads(PROTOCOL.read_text(encoding='utf-8'))
        protocol['window']['late_Y_slice'] = [19, 26]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bad.json'
            path.write_text(json.dumps(protocol), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'window'):
                load_protocol(path)

    def test_protocol_rejects_outcome_based_sensitivity_selection(self):
        protocol = json.loads(PROTOCOL.read_text(encoding='utf-8'))
        protocol['nonoverlap_sensitivity']['outcomes_used_for_selection'] = True
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bad.json'
            path.write_text(json.dumps(protocol), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'non-overlap'):
                load_protocol(path)

    def test_descriptive_outputs_cannot_change_the_primary_gate(self):
        protocol = json.loads(PROTOCOL.read_text(encoding='utf-8'))
        protocol['descriptive_outputs'][
            'descriptive_results_may_not_change_the_primary_gate'] = False
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bad.json'
            path.write_text(json.dumps(protocol), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'descriptive'):
                load_protocol(path)


class WindowTests(unittest.TestCase):
    def setUp(self):
        self.protocol = load_protocol(PROTOCOL)

    def test_source_slots_and_phases_reproduce_frozen_layout(self):
        slots = source_slots('2023-02-01T00:00:00', self.protocol['window'])
        self.assertEqual(len(slots), 26)
        self.assertEqual(slots[-1], datetime(2023, 2, 1, 2, 5))
        self.assertEqual(phase_for_step(11, self.protocol['window']), 'X')
        self.assertEqual(phase_for_step(12, self.protocol['window']), 'excluded_latency')
        self.assertEqual(phase_for_step(14, self.protocol['window']), 'Y')

    def test_nonoverlap_uses_one_slot_set_across_positive_and_control_sides(self):
        positives = {
            1: {'x_start': '2023-01-01T00:00:00'},
            # This positive window overlaps pair 1's control window.
            2: {'x_start': '2023-01-22T00:05:00'},
            3: {'x_start': '2023-01-15T00:00:00'},
        }
        rows = [
            {'control_index': '0', 'positive_sample_index': '1',
             'positive_t0': '2023-01-01T01:05:00', 'x_start': '2023-01-22T00:00:00'},
            {'control_index': '1', 'positive_sample_index': '2',
             'positive_t0': '2023-01-08T01:05:00', 'x_start': '2023-02-05T00:00:00'},
            {'control_index': '2', 'positive_sample_index': '3',
             'positive_t0': '2023-01-15T01:05:00', 'x_start': '2023-01-29T00:00:00'},
        ]
        selected = select_nonoverlap(rows, positives, self.protocol['window'])
        self.assertEqual(selected, {0, 2})


class BootstrapTests(unittest.TestCase):
    def test_cluster_bootstrap_is_deterministic_and_contains_constant_mean(self):
        values = np.asarray([1., 1., 3., 3.])
        clusters = np.asarray(['a', 'a', 'b', 'b'])
        first = cluster_bootstrap_mean(values, clusters, 1000, .95, 7)
        second = cluster_bootstrap_mean(values, clusters, 1000, .95, 7)
        self.assertEqual(first, second)
        self.assertLessEqual(first[0], 2.)
        self.assertGreaterEqual(first[1], 2.)

    def test_cluster_bootstrap_supports_missing_distance_strata(self):
        values = np.asarray([[1., np.nan], [1., 2.], [3., 4.], [3., np.nan]])
        low, high = cluster_bootstrap_mean(
            values, np.asarray(['a', 'a', 'b', 'b']), 1000, .95, 9)
        self.assertTrue(np.isfinite(low).all())
        self.assertTrue(np.isfinite(high).all())


def population(effect, low, high, pairs=1000):
    return {
        'pairs': pairs,
        'late_y_change_contrast_raw': effect * 100,
        'late_y_change_contrast_ci_low_raw': low * 100,
        'late_y_change_contrast_ci_high_raw': high * 100,
        'late_y_change_contrast_in_train_std': effect,
    }


class GateTests(unittest.TestCase):
    def setUp(self):
        self.protocol = load_protocol(PROTOCOL)

    def test_gate_passes_only_consistent_detected_main_and_sensitivity_signal(self):
        summaries = {
            'train': {'all_matched': population(-.10, -.15, -.05),
                      'nonoverlap': population(-.06, -.12, .01, 650)},
            'val': {'all_matched': population(-.08, -.13, -.02),
                    'nonoverlap': population(-.04, -.10, .03, 150)},
        }
        checks, ready = phenomenon_gate(summaries, self.protocol)
        self.assertTrue(ready)
        self.assertTrue(checks['main_direction_consistent_across_splits'])

    def test_gate_rejects_shuffled_direction_even_with_large_magnitudes(self):
        summaries = {
            'train': {'all_matched': population(-.10, -.15, -.05),
                      'nonoverlap': population(-.06, -.12, .01, 650)},
            'val': {'all_matched': population(.10, .05, .15),
                    'nonoverlap': population(.06, -.01, .12, 150)},
        }
        _, ready = phenomenon_gate(summaries, self.protocol)
        self.assertFalse(ready)

    def test_gate_rejects_ci_crossing_zero(self):
        summaries = {
            'train': {'all_matched': population(-.10, -.15, .01),
                      'nonoverlap': population(-.06, -.12, .01, 650)},
            'val': {'all_matched': population(-.08, -.13, -.02),
                    'nonoverlap': population(-.04, -.10, .03, 150)},
        }
        checks, ready = phenomenon_gate(summaries, self.protocol)
        self.assertFalse(ready)
        self.assertFalse(checks['main_block_ci_excludes_zero']['train'])


if __name__ == '__main__':
    unittest.main()
