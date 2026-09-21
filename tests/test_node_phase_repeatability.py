"""Tests for the train-only v7c node-phase repeatability audit."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.chronological.audit_node_phase_repeatability import (
    correction_from_lookup, decision, fit_lookup, load_protocol,
    rolling_origin_folds, weighted_median,
)


REPO = Path(__file__).resolve().parents[1]
PROTOCOL = (REPO / 'experiments/chronological' /
            'node_phase_repeatability_v7c.json')


class ProtocolTests(unittest.TestCase):
    def test_protocol_freezes_train_only_rolling_origin_boundary(self):
        protocol = load_protocol(PROTOCOL)
        self.assertEqual(protocol['rolling_origin']['initial_fit_week_count'], 18)
        self.assertEqual(protocol['rolling_origin']['audit_week_block_sizes'], [6, 6, 5])
        self.assertEqual(protocol['primary_family'], 'zero_anchored_mean')
        self.assertEqual(protocol['primary_level'], 'node_phase')
        self.assertTrue(protocol['information_boundary'][
            'validation_residual_arrays_prohibited'])
        self.assertTrue(protocol['information_boundary'][
            'control_future_Y_prohibited_from_lookup_fit'])

    def test_protocol_rejects_post_result_fold_change(self):
        protocol = json.loads(PROTOCOL.read_text(encoding='utf-8'))
        protocol['rolling_origin']['initial_fit_week_count'] = 17
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'changed.json'
            path.write_text(json.dumps(protocol), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'rolling-origin design changed'):
                load_protocol(path)


def repeated_weeks(counts):
    values = []
    for number, count in enumerate(counts, start=1):
        values.extend([f'2023-W{number:02d}'] * count)
    return np.asarray(values)


class RollingOriginTests(unittest.TestCase):
    def test_final_seventeen_weeks_are_audited_once_with_past_only_fit(self):
        protocol = load_protocol(PROTOCOL)
        full_early = [98] * 17 + [91]
        common_early = [89] * 17 + [101]
        full = repeated_weeks(full_early + [101] * 5 + [97] +
                              [121] * 5 + [126] + [103] * 4 + [102])
        common = repeated_weeks(common_early + [89] * 5 + [91] +
                                [103] * 6 + [68] * 4 + [66])
        self.assertEqual(len(full), 3604)
        self.assertEqual(len(common), 3106)
        folds = rolling_origin_folds(full, common, protocol)
        self.assertEqual([len(fold['full_audit_indices']) for fold in folds],
                         [602, 731, 514])
        self.assertEqual([len(fold['common_audit_indices']) for fold in folds],
                         [536, 618, 338])
        for fold in folds:
            self.assertLess(max(fold['fit_weeks']), min(fold['audit_weeks']))
            self.assertEqual(
                np.intersect1d(fold['common_fit_indices'],
                               fold['common_audit_indices']).size, 0)


def incident_arrays():
    residual = np.zeros((2, 6, 2, 1), dtype=np.float32)
    residual[0, :, 0, 0] = 6.
    residual[1, :, 1, 0] = 3.
    valid = np.ones_like(residual, dtype=bool)
    candidate = np.array([[True, False], [False, True]])
    return {
        'signed_residual': residual, 'valid': valid,
        'candidate_mask': candidate,
        'positive_sample_index': np.array([10, 20], dtype=np.int64),
    }


class LookupTests(unittest.TestCase):
    def test_weighted_median_respects_weights(self):
        self.assertEqual(weighted_median([1., 10., 20.], [1., 5., 1.]), 10.)

    def test_zero_anchored_mean_uses_equal_event_and_cohort_weight(self):
        incident = incident_arrays()
        controls = [
            {'candidate_mask': incident['candidate_mask'].copy(),
             'positive_sample_index': incident['positive_sample_index'].copy()}
            for _ in range(2)
        ]
        median, median_known, _ = fit_lookup(
            incident, controls, np.array([0, 1]), 'node',
            'incident_median', nodes=2)
        anchored, anchored_known, summary = fit_lookup(
            incident, controls, np.array([0, 1]), 'node',
            'zero_anchored_mean', nodes=2)
        np.testing.assert_allclose(median, [5.97, 3.])
        np.testing.assert_array_equal(median_known, [True, True])
        np.testing.assert_allclose(anchored, [2., 1.])
        np.testing.assert_array_equal(anchored_known, [True, True])
        self.assertEqual(summary['fit_control_cohorts'], 2)

    def test_node_phase_correction_has_shape_safe_candidate_support(self):
        arrays = {
            'signed_residual': np.zeros((1, 6, 3, 1), dtype=np.float32),
            'valid': np.ones((1, 6, 3, 1), dtype=bool),
            'candidate_mask': np.array([[True, True, False]]),
        }
        lookup = np.array([[1., 2.], [3., 4.], [5., 6.]])
        known = np.ones_like(lookup, dtype=bool)
        correction, covered, total = correction_from_lookup(
            arrays, lookup, known, 'node_phase')
        np.testing.assert_array_equal(correction[0, :3, 0, 0], 1.)
        np.testing.assert_array_equal(correction[0, 3:, 0, 0], 2.)
        np.testing.assert_array_equal(correction[0, :3, 1, 0], 3.)
        np.testing.assert_array_equal(correction[0, 3:, 1, 0], 4.)
        np.testing.assert_array_equal(correction[0, :, 2, 0], 0.)
        self.assertEqual(covered, 12)
        self.assertEqual(total, 12)

    def test_node_correction_has_shape_safe_candidate_support(self):
        arrays = {
            'signed_residual': np.zeros((1, 6, 3, 1), dtype=np.float32),
            'valid': np.ones((1, 6, 3, 1), dtype=bool),
            'candidate_mask': np.array([[True, True, False]]),
        }
        lookup = np.array([1., 3., 5.])
        known = np.ones_like(lookup, dtype=bool)
        correction, covered, total = correction_from_lookup(
            arrays, lookup, known, 'node')
        np.testing.assert_array_equal(correction[0, :, 0, 0], 1.)
        np.testing.assert_array_equal(correction[0, :, 1, 0], 3.)
        np.testing.assert_array_equal(correction[0, :, 2, 0], 0.)
        self.assertEqual(covered, 12)
        self.assertEqual(total, 12)


def result_population(improvement=.1, candidate_improvement=.2,
                      candidate_ci_low=.1, harm_ci_high=.01):
    return {
        'all': {'mae_A': 10., 'improvement_vs_A': improvement},
        'candidate_h1_h6': {
            'mae_A': 10., 'improvement_vs_A': candidate_improvement,
        },
        'correction': {'nonzero_fraction': .5},
        'protected_h7_h12_exact_A': True,
        'protected_noncandidate_exact_A': True,
        'uncertainty': {
            'all_improvement_vs_A': {'ci_low': 0.},
            'candidate_h1_h6_improvement_vs_A': {
                'ci_low': candidate_ci_low,
            },
            'candidate_h1_h6_harm_vs_A': {'ci_high': harm_ci_high},
        },
    }


class DecisionTests(unittest.TestCase):
    def test_primary_gate_authorizes_only_node_phase_development(self):
        protocol = load_protocol(PROTOCOL)
        primary = {
            cohort: result_population() for cohort in
            ('incident_full', 'incident', 'primary_control', 'secondary_control')
        }
        diagnostic = {'incident_full': result_population()}
        results = {
            'zero_anchored_mean': {'node_phase': primary},
            'incident_median': {'node_phase': diagnostic},
        }
        outcome = decision(results, protocol)
        self.assertTrue(outcome['repeatability_gate_passed'])
        self.assertEqual(
            outcome['recommendation'],
            'NODE_PHASE_EMBEDDED_EXPERT_DEVELOPMENT_ALLOWED')

    def test_incident_repeatability_does_not_override_failed_zero_anchor(self):
        protocol = load_protocol(PROTOCOL)
        primary = {
            cohort: result_population() for cohort in
            ('incident_full', 'incident', 'primary_control', 'secondary_control')
        }
        primary['incident_full'] = result_population(
            improvement=-.01, candidate_ci_low=-.1)
        diagnostic = {'incident_full': result_population()}
        results = {
            'zero_anchored_mean': {'node_phase': primary},
            'incident_median': {'node_phase': diagnostic},
        }
        outcome = decision(results, protocol)
        self.assertFalse(outcome['repeatability_gate_passed'])
        self.assertTrue(outcome['incident_only_node_phase_repeatability_detected'])
        self.assertEqual(
            outcome['recommendation'],
            'NODE_PHASE_REPEATS_BUT_ZERO_ANCHOR_NOT_VALIDATED')

    def test_absent_repeatability_stops_node_phase_direction(self):
        protocol = load_protocol(PROTOCOL)
        primary = {
            cohort: result_population() for cohort in
            ('incident_full', 'incident', 'primary_control', 'secondary_control')
        }
        primary['incident_full'] = result_population(
            improvement=-.01, candidate_ci_low=-.1)
        diagnostic = {'incident_full': result_population(
            improvement=-.02, candidate_ci_low=-.2)}
        results = {
            'zero_anchored_mean': {'node_phase': primary},
            'incident_median': {'node_phase': diagnostic},
        }
        outcome = decision(results, protocol)
        self.assertFalse(outcome['repeatability_gate_passed'])
        self.assertFalse(outcome['incident_only_node_phase_repeatability_detected'])
        self.assertEqual(
            outcome['recommendation'],
            'STOP_NODE_PHASE_REPEATABILITY_NOT_ESTABLISHED')


if __name__ == '__main__':
    unittest.main()
