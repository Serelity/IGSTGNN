"""Tests for the hard-supported v7b signed-residual feasibility probe."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.chronological.audit_signed_residual_probe import (
    LEVELS, correction_from_predictions, correction_metrics, load_protocol,
    evaluation_target_for, make_probe_rows, oracle_correction, oracle_summary,
    residual_feature_names, residual_feature_vector, target_for,
)


REPO = Path(__file__).resolve().parents[1]
PROTOCOL = (REPO / 'experiments/chronological' /
            'signed_residual_probe_v7b.json')


class ProtocolTests(unittest.TestCase):
    def test_protocol_freezes_support_probe_and_information_boundary(self):
        protocol = load_protocol(PROTOCOL)
        self.assertEqual(protocol['oracle']['primary_family_for_development_tier'],
                         'event_node_phase')
        self.assertEqual(protocol['probe']['control_target'],
                         'exact_zero_without_reading_control_future_Y')
        self.assertEqual(
            protocol['probe']['control_row_support'],
            'candidate_nodes_and_predeclared_phases_without_valid_or_future_Y')
        self.assertTrue(protocol['support']['correction_exactly_zero_outside_support'])
        self.assertTrue(protocol['information_boundary'][
            'validation_prediction_support_independent_of_future_Y'])
        self.assertTrue(protocol['information_boundary']['test_split_prohibited'])

    def test_protocol_rejects_post_result_gate_weakening(self):
        protocol = json.loads(PROTOCOL.read_text(encoding='utf-8'))
        protocol['development_gate'][
            'maximum_each_control_H1_H6_harm_fraction_of_A_mae'] = .05
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'changed.json'
            path.write_text(json.dumps(protocol), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'gate changed'):
                load_protocol(path)


def residual_arrays():
    residual = np.zeros((2, 6, 4, 1), dtype=np.float32)
    residual[0, :, 0, 0] = [1., 2., 3., 10., 11., 12.]
    residual[0, :, 2, 0] = [3., 4., 5., 20., 21., 22.]
    residual[1, :, 1, 0] = [-4., -3., -2., 5., 6., 7.]
    valid = np.ones_like(residual, dtype=bool)
    candidate = np.array([
        [True, False, True, False],
        [False, True, False, False],
    ])
    support_sum = np.where(
        candidate[:, None, :, None], np.abs(residual), 0.).sum(axis=(1, 2, 3))
    return {
        'signed_residual': residual,
        'baseline_prediction': np.full_like(residual, 20.),
        'valid': valid,
        'candidate_mask': candidate,
        'positive_sample_index': np.array([10, 42]),
        'baseline_all_absolute_sum': support_sum.astype(np.float64) + 100.,
        'baseline_all_valid_count': np.full(2, 48, dtype=np.int64),
    }


class OracleTests(unittest.TestCase):
    def test_event_oracle_uses_all_candidate_horizons_without_axis_swap(self):
        arrays = residual_arrays()
        correction = oracle_correction(arrays, 'event')
        expected = np.median(np.concatenate([
            arrays['signed_residual'][0, :, 0, 0],
            arrays['signed_residual'][0, :, 2, 0],
        ]))
        np.testing.assert_array_equal(correction[0, :, 0, 0], expected)
        np.testing.assert_array_equal(correction[0, :, 2, 0], expected)
        np.testing.assert_array_equal(correction[0, :, 1, 0], 0.)

    def test_phase_oracle_never_worsens_and_is_hard_supported(self):
        arrays = residual_arrays()
        correction = oracle_correction(arrays, 'event_node_phase')
        self.assertTrue(np.all(correction[:, :, 3, 0] == 0))
        result = oracle_summary(arrays)['event_node_phase']
        self.assertGreaterEqual(result['all']['improvement_vs_A'], 0.)
        self.assertGreater(result['candidate_h1_h6']['improvement_vs_A'], 0.)
        self.assertTrue(result['protected_h7_h12_exact_A'])
        self.assertTrue(result['protected_noncandidate_exact_A'])

    def test_global_metric_reconstructs_frozen_A_outside_local_support(self):
        arrays = residual_arrays()
        correction = np.zeros_like(arrays['signed_residual'])
        result = correction_metrics(arrays, correction)
        expected = arrays['baseline_all_absolute_sum'].sum() / \
            arrays['baseline_all_valid_count'].sum()
        self.assertAlmostEqual(result['all']['mae_A'], expected)
        self.assertEqual(result['all']['mae_A'], result['all']['mae_corrected'])


class _NoFutureArrays(dict):
    def __getitem__(self, key):
        if key in ('signed_residual', 'valid'):
            raise AssertionError('Control-row construction read future Y')
        return super().__getitem__(key)


class _Source:
    def __init__(self, samples):
        self.samples = samples

    def sample(self, cohort, position):
        return self.samples[position]


def report_time_sample(index, candidate):
    nodes = len(candidate)
    return {
        'sample': index,
        'incident_id': f'incident-{index}',
        'timestamp': '2023-09-18T13:25:00',
        'positive_t0': '2023-09-18T13:25:00',
        'iso_week': '2023-W38',
        'freeway': '4', 'direction': 'E', 'report_age': 5.,
        'distances': np.arange(nodes * 3, dtype=np.float64).reshape(nodes, 3),
        'candidate': np.asarray(candidate, dtype=bool),
        'history': np.ones((nodes, 5), dtype=np.float64),
    }


class ProbeRowTests(unittest.TestCase):
    def test_control_zero_targets_do_not_read_residual_or_validity(self):
        candidate = np.array([[True, False, True]])
        arrays = _NoFutureArrays({
            'positive_sample_index': np.array([7]),
            'candidate_mask': candidate,
            'baseline_prediction': np.ones((1, 6, 3, 1), dtype=np.float32),
        })
        source = _Source([report_time_sample(7, candidate[0])])
        features, targets, weights, metadata = make_probe_rows(
            source, 'primary_control', arrays,
            {'freeway': ['4'], 'direction': ['E']},
            'event_node_phase', 'zero')
        self.assertEqual(features.shape[0], 4)
        np.testing.assert_array_equal(targets, 0.)
        self.assertAlmostEqual(weights.sum(), 1.)
        self.assertEqual(len(metadata), 4)

    def test_zero_target_rows_are_defined_without_future_arrays(self):
        rows = target_for(None, None, np.array([1, 3]),
                          'event_node_phase', False)
        self.assertEqual(rows, [
            (1, 0, 0.), (1, 1, 0.), (3, 0, 0.), (3, 1, 0.),
        ])

    def test_evaluation_rows_do_not_shrink_when_future_targets_are_missing(self):
        residual = np.ones((6, 3, 1))
        valid = np.ones_like(residual, dtype=bool)
        valid[:, 2, 0] = False
        rows = evaluation_target_for(
            residual, valid, np.array([0, 2]), 'event_node_phase')
        self.assertEqual(len(rows), 4)
        self.assertTrue(np.isfinite(rows[0][2]))
        self.assertTrue(np.isfinite(rows[1][2]))
        self.assertTrue(np.isnan(rows[2][2]))
        self.assertTrue(np.isnan(rows[3][2]))

    def test_feature_names_match_vectors_for_every_level(self):
        sample = report_time_sample(7, [True, False, True])
        baseline = np.ones((6, 3, 1), dtype=np.float32)
        categories = {'freeway': ['4'], 'direction': ['E']}
        for level in LEVELS:
            node = None if level == 'event' else 0
            phase = 0 if level == 'event_node_phase' else None
            vector = residual_feature_vector(
                sample, baseline, categories, node=node, phase=phase)
            self.assertEqual(len(vector), len(residual_feature_names(categories, level)))

    def test_event_prediction_is_written_only_to_candidate_nodes(self):
        arrays = residual_arrays()
        sample = report_time_sample(10, arrays['candidate_mask'][0])
        correction, clipped = correction_from_predictions(
            arrays, np.array([9.]), [(0, None, None, sample)], clip=2.)
        self.assertEqual(float(clipped[0]), 2.)
        np.testing.assert_array_equal(correction[0, :, 0, 0], 2.)
        np.testing.assert_array_equal(correction[0, :, 2, 0], 2.)
        np.testing.assert_array_equal(correction[0, :, 1, 0], 0.)


if __name__ == '__main__':
    unittest.main()
