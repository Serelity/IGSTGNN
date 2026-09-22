"""Tests for the minimal v9a hard-routed residual expert."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.chronological.audit_protected_residual_expert import (
    gate_decision, load_control_training_inputs, load_protocol,
    make_expert_rows, protected_metrics,
)


REPO = Path(__file__).resolve().parents[1]
PROTOCOL = (REPO / 'experiments/chronological' /
            'protected_residual_expert_v9a.json')


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
        'sample': index, 'incident_id': f'incident-{index}',
        'timestamp': '2023-09-18T13:25:00',
        'positive_t0': '2023-09-18T13:25:00', 'iso_week': '2023-W38',
        'freeway': '4', 'direction': 'E', 'report_age': 5.,
        'distances': np.arange(nodes * 3, dtype=np.float64).reshape(nodes, 3),
        'candidate': np.asarray(candidate, dtype=bool),
        'history': np.ones((nodes, 5), dtype=np.float64),
    }


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
        'valid': valid, 'candidate_mask': candidate,
        'positive_sample_index': np.array([10, 42]),
        'baseline_all_absolute_sum': support_sum.astype(np.float64) + 100.,
        'baseline_all_valid_count': np.full(2, 48, dtype=np.int64),
    }


def routes(arrays):
    candidate = arrays['candidate_mask'].copy()
    hierarchical = np.zeros_like(candidate)
    hierarchical[0, 0] = True
    hierarchical[1, 1] = True
    return {
        'positive_sample_index': arrays['positive_sample_index'].copy(),
        'candidate_mask': candidate,
        'hierarchical_route': hierarchical,
    }


class ProtocolTests(unittest.TestCase):
    def test_protocol_freezes_router_and_hard_support(self):
        protocol = load_protocol(PROTOCOL)
        self.assertTrue(protocol['information_boundary'][
            'v8c_router_and_thresholds_frozen'])
        self.assertTrue(protocol['support'][
            'correction_exactly_zero_outside_hierarchical_route'])
        self.assertEqual(protocol['expert']['training_cohorts'], [
            'incident_full', 'primary_control', 'secondary_control'])

    def test_protocol_rejects_post_result_harm_weakening(self):
        protocol = json.loads(PROTOCOL.read_text(encoding='utf-8'))
        protocol['development_gate'][
            'maximum_each_control_routed_harm_fraction_of_A_mae'] = .05
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'changed.json'
            path.write_text(json.dumps(protocol), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'development gate changed'):
                load_protocol(path)


class RowTests(unittest.TestCase):
    def test_control_loader_does_not_deserialize_future_outcomes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'train_primary_control_signed_residuals.npz'
            np.savez_compressed(
                path,
                signed_residual=np.asarray([object()], dtype=object),
                baseline_prediction=np.ones((2, 6, 4, 1), dtype=np.float32),
                valid=np.asarray([object()], dtype=object),
                candidate_mask=np.ones((2, 4), dtype=bool),
                positive_sample_index=np.asarray([10, 42], dtype=np.int64),
                baseline_all_absolute_sum=np.asarray([object()], dtype=object),
                baseline_all_valid_count=np.asarray([object()], dtype=object),
            )
            arrays = load_control_training_inputs(
                directory, 'train', 'primary_control', 2, 4)
        self.assertEqual(set(arrays), {
            'baseline_prediction', 'candidate_mask', 'positive_sample_index'})

    def test_control_rows_use_only_frozen_route_and_zero_target(self):
        base = residual_arrays()
        protected = _NoFutureArrays({
            'positive_sample_index': base['positive_sample_index'],
            'candidate_mask': base['candidate_mask'],
            'baseline_prediction': base['baseline_prediction'],
        })
        source = _Source([
            report_time_sample(10, base['candidate_mask'][0]),
            report_time_sample(42, base['candidate_mask'][1]),
        ])
        features, targets, weights, metadata, routed_events = make_expert_rows(
            source, 'primary_control', protected, routes(base),
            {'freeway': ['4'], 'direction': ['E']}, 'zero')
        self.assertEqual(routed_events, 2)
        self.assertEqual(len(features), 4)
        np.testing.assert_array_equal(targets, 0.)
        self.assertAlmostEqual(weights.sum(), 2.)
        self.assertEqual({row[1] for row in metadata}, {0, 1})

    def test_evaluation_rows_do_not_depend_on_future_validity(self):
        arrays = residual_arrays()
        arrays['valid'][0, 3:6, 0, 0] = False
        source = _Source([
            report_time_sample(10, arrays['candidate_mask'][0]),
            report_time_sample(42, arrays['candidate_mask'][1]),
        ])
        features, targets, weights, metadata, routed_events = make_expert_rows(
            source, 'incident_full', arrays, routes(arrays),
            {'freeway': ['4'], 'direction': ['E']}, 'evaluation')
        self.assertEqual(routed_events, 2)
        self.assertEqual(len(features), 4)
        self.assertEqual(len(metadata), 4)
        self.assertEqual(len(weights), 4)
        invalid_rows = [
            target for target, row in zip(targets, metadata)
            if row[0] == 0 and row[1] == 0 and row[2] == 1
        ]
        self.assertEqual(len(invalid_rows), 1)
        self.assertTrue(np.isnan(invalid_rows[0]))


class MetricTests(unittest.TestCase):
    def test_correction_is_exact_A_outside_frozen_route(self):
        arrays = residual_arrays()
        current_routes = routes(arrays)
        correction = np.zeros_like(arrays['signed_residual'])
        correction[0, :, 0, 0] = 2.
        correction[1, :, 1, 0] = -2.
        result = protected_metrics(arrays, current_routes, correction)
        self.assertTrue(result['protected_outside_hierarchical_route_exact_A'])
        self.assertGreater(result['routed_candidate_h1_h6']['valid_cells'], 0)

    def test_correction_outside_route_is_rejected(self):
        arrays = residual_arrays()
        correction = np.zeros_like(arrays['signed_residual'])
        correction[0, :, 2, 0] = 1.
        with self.assertRaisesRegex(ValueError, 'escaped'):
            protected_metrics(arrays, routes(arrays), correction)


class DecisionTests(unittest.TestCase):
    def test_gate_requires_forecasting_gain_and_control_safety(self):
        protocol = load_protocol(PROTOCOL)

        def result(improvement, low, harm_high=0.):
            return {
                'all': {'mae_A': 20., 'improvement_vs_A': improvement},
                'candidate_h1_h6': {'mae_A': 15.},
                'routed_candidate_h1_h6': {
                    'mae_A': 15., 'improvement_vs_A': improvement,
                    'nonzero_correction_fraction': .5},
                'uncertainty': {
                    'all_improvement_vs_A': {'ci_low': low},
                    'routed_improvement_vs_A': {'ci_low': low},
                    'routed_harm_vs_A': {'ci_high': harm_high}},
                'protected_outside_hierarchical_route_exact_A': True,
                'protected_h7_h12_exact_A': True,
                'protected_noncandidate_exact_A': True,
            }

        validation = {
            'incident_full': result(.2, .1),
            'incident': result(.1, .01),
            'primary_control': result(0., 0., .01),
            'secondary_control': result(0., 0., .01),
        }
        self.assertTrue(gate_decision(
            validation, protocol)['protected_residual_expert_gate_passed'])
        validation['primary_control']['uncertainty'][
            'routed_harm_vs_A']['ci_high'] = 1.
        self.assertFalse(gate_decision(
            validation, protocol)['protected_residual_expert_gate_passed'])


if __name__ == '__main__':
    unittest.main()
