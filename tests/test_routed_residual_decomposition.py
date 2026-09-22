"""Tests for the v9b cross-fitted routed-residual decomposition."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.chronological.audit_routed_residual_decomposition import (
    correction_for_subset, diagnostics_summary, fit_experts, gate_decision,
    load_protocol, make_rows, policy_predictions, weighted_auc,
    weighted_correlation,
)
from experiments.chronological.audit_expert_benefit import RidgeModel


REPO = Path(__file__).resolve().parents[1]
PROTOCOL = (REPO / 'experiments/chronological' /
            'routed_residual_decomposition_v9b.json')


class _NoFutureArrays(dict):
    def __getitem__(self, key):
        if key in ('signed_residual', 'valid'):
            raise AssertionError('Control expert fit read future outcomes')
        return super().__getitem__(key)


class _Source:
    def __init__(self, samples):
        self.samples = samples

    def sample(self, cohort, position):
        return self.samples[position]


def sample(index, candidate, week):
    nodes = len(candidate)
    return {
        'sample': index, 'incident_id': f'incident-{index}',
        'timestamp': '2023-09-18T13:25:00',
        'positive_t0': '2023-09-18T13:25:00', 'iso_week': week,
        'freeway': '4', 'direction': 'E', 'report_age': 5.,
        'distances': np.arange(nodes * 3, dtype=np.float64).reshape(nodes, 3),
        'candidate': np.asarray(candidate, dtype=bool),
        'history': np.ones((nodes, 5), dtype=np.float64),
    }


def arrays():
    residual = np.zeros((3, 6, 3, 1), dtype=np.float32)
    residual[0, :, 0, 0] = [1., 2., 3., 4., 5., 6.]
    residual[1, :, 1, 0] = [-1., -2., -3., -4., -5., -6.]
    residual[2, :, 2, 0] = 10.
    return {
        'signed_residual': residual,
        'baseline_prediction': np.full_like(residual, 20.),
        'valid': np.ones_like(residual, dtype=bool),
        'candidate_mask': np.eye(3, dtype=bool),
        'positive_sample_index': np.asarray([10, 20, 30], dtype=np.int64),
    }


def routes(current):
    candidate = current['candidate_mask'].copy()
    return {
        'positive_sample_index': current['positive_sample_index'].copy(),
        'candidate_mask': candidate,
        'hierarchical_route': candidate.copy(),
    }


def source_for(current):
    return _Source([
        sample(10, current['candidate_mask'][0], '2023-W20'),
        sample(20, current['candidate_mask'][1], '2023-W21'),
        sample(30, current['candidate_mask'][2], '2023-W22'),
    ])


class ProtocolTests(unittest.TestCase):
    def test_protocol_freezes_cross_fitted_router_and_no_v9a_override(self):
        protocol = load_protocol(PROTOCOL)
        self.assertFalse(protocol['router_design']['full_train_v8c_routes_used'])
        self.assertTrue(protocol['information_boundary'][
            'router_and_expert_fit_on_past_weeks_only'])
        self.assertTrue(protocol['information_boundary']['v9a_gate_not_overridden'])
        self.assertEqual(
            protocol['diagnostic_gate']['primary_policy'],
            'two_part_abstained')

    def test_protocol_rejects_post_result_auc_weakening(self):
        protocol = json.loads(PROTOCOL.read_text(encoding='utf-8'))
        protocol['diagnostic_gate']['require_direction_auc_ci_lower_above'] = .5
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'changed.json'
            path.write_text(json.dumps(protocol), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'diagnostic gate changed'):
                load_protocol(path)


class MetricTests(unittest.TestCase):
    def test_weighted_auc_handles_ties_and_weights(self):
        labels = np.asarray([0, 1, 0, 1])
        scores = np.asarray([0., 1., 1., 1.])
        weights = np.asarray([1., 1., 2., 1.])
        self.assertAlmostEqual(weighted_auc(labels, scores, weights), 2. / 3.)

    def test_weighted_correlation_returns_zero_for_constant_prediction(self):
        self.assertEqual(weighted_correlation(
            [1., 2., 3.], [5., 5., 5.], [1., 1., 1.]), 0.)

    def test_diagnostics_bootstrap_is_stratified_across_all_folds(self):
        targets, weeks, folds = [], [], []
        for fold in range(1, 4):
            for week in range(2):
                targets.extend([-2., -1., 1., 2.])
                weeks.extend([f'f{fold}-w{week}'] * 4)
                folds.extend([fold] * 4)
        targets = np.asarray(targets)
        protocol = load_protocol(PROTOCOL)
        protocol['uncertainty'] = {
            'method': 'audit_iso_week_cluster_bootstrap',
            'draws': 100, 'confidence_level': .95, 'seed': 2025}
        result = diagnostics_summary(
            targets, targets, targets, np.abs(targets),
            np.ones(len(targets)), np.asarray(weeks), np.asarray(folds),
            np.ones(len(targets), dtype=bool), protocol)
        self.assertEqual(result['direction_auc'], 1.)
        self.assertEqual(result['magnitude_correlation'], 1.)
        self.assertEqual(len(result['folds']), 3)


class RowTests(unittest.TestCase):
    def test_control_fit_uses_only_selected_rows_and_zero_targets(self):
        current = arrays()
        protected = _NoFutureArrays({
            'baseline_prediction': current['baseline_prediction'],
            'candidate_mask': current['candidate_mask'],
            'positive_sample_index': current['positive_sample_index'],
        })
        features, targets, weights, metadata, routed = make_rows(
            source_for(current), 'primary_control', protected, routes(current),
            {'freeway': ['4'], 'direction': ['E']}, np.asarray([0, 2]), 'zero')
        self.assertEqual(routed, 2)
        self.assertEqual(len(features), 4)
        np.testing.assert_array_equal(targets, 0.)
        self.assertEqual({row[0] for row in metadata}, {0, 2})
        self.assertAlmostEqual(weights.sum(), 2.)

    def test_evaluation_support_does_not_shrink_on_missing_target(self):
        current = arrays()
        current['valid'][0, 3:6, 0, 0] = False
        _, targets, _, metadata, _ = make_rows(
            source_for(current), 'incident_full', current, routes(current),
            {'freeway': ['4'], 'direction': ['E']}, np.asarray([0]),
            'evaluation')
        self.assertEqual(len(metadata), 2)
        self.assertTrue(np.isnan(targets[1]))

    def test_subset_correction_cannot_use_nonselected_event(self):
        current = arrays()
        metadata = [(2, 2, 0, source_for(current).sample('incident_full', 2))]
        correction = correction_for_subset(
            current, np.asarray([2]), np.asarray([3.]), metadata)
        np.testing.assert_array_equal(correction[0, :3, 2, 0], 3.)
        np.testing.assert_array_equal(correction[0, 3:, 2, 0], 0.)


class PolicyTests(unittest.TestCase):
    @staticmethod
    def model(target_mean):
        return RidgeModel(
            mean=np.zeros(1), scale=np.ones(1), target_mean=target_mean,
            coefficient=np.zeros(1), alpha=10.)

    def test_two_part_abstention_is_fit_threshold_controlled(self):
        models = {
            'signed': self.model(2.),
            'direction': self.model(.4),
            'magnitude': self.model(3.),
        }
        policies, direction, magnitude, confident = policy_predictions(
            models, np.ones((2, 1)), np.asarray([5., -5.]), 10., .5)
        np.testing.assert_array_equal(direction, .4)
        np.testing.assert_array_equal(magnitude, 3.)
        np.testing.assert_array_equal(confident, False)
        np.testing.assert_array_equal(policies['two_part_all'], 3.)
        np.testing.assert_array_equal(policies['two_part_abstained'], 0.)
        np.testing.assert_array_equal(policies['oracle_direction'], [3., -3.])

    def test_three_equal_weight_cohorts_fit_all_decomposition_models(self):
        current = arrays()
        control = _NoFutureArrays({
            'baseline_prediction': current['baseline_prediction'],
            'candidate_mask': current['candidate_mask'],
            'positive_sample_index': current['positive_sample_index'],
        })
        models, clip, abstention, summary = fit_experts(
            source_for(current), current,
            {'primary_control': control, 'secondary_control': control},
            {cohort: routes(current) for cohort in
             ('incident_full', 'primary_control', 'secondary_control')},
            {'freeway': ['4'], 'direction': ['E']},
            np.asarray([0, 1, 2]), np.asarray([0, 1, 2]),
            load_protocol(PROTOCOL))
        self.assertEqual(set(models), {'signed', 'direction', 'magnitude'})
        self.assertGreater(clip, 0.)
        self.assertGreaterEqual(abstention, 0.)
        for cohort in ('incident_full', 'primary_control', 'secondary_control'):
            self.assertAlmostEqual(summary[cohort]['normalized_weight_sum'], 1.)


def result(improvement=.1, low=.01, harm_high=.01):
    return {
        'all': {'mae_A': 20., 'improvement_vs_A': improvement},
        'routed_candidate_h1_h6': {
            'mae_A': 15., 'improvement_vs_A': improvement},
        'uncertainty': {
            'all_improvement_vs_A': {'ci_low': low},
            'routed_improvement_vs_A': {'ci_low': low},
            'routed_harm_vs_A': {'ci_high': harm_high}},
        'protected_outside_hierarchical_route_exact_A': True,
        'protected_h7_h12_exact_A': True,
        'protected_noncandidate_exact_A': True,
    }


class DecisionTests(unittest.TestCase):
    def test_gate_supports_only_new_preregistration(self):
        protocol = load_protocol(PROTOCOL)
        primary = {
            cohort: result() for cohort in
            ('incident_full', 'incident', 'primary_control', 'secondary_control')
        }
        results = {'two_part_abstained': primary}
        diagnostics = {
            'direction_auc_ci_low': .6,
            'magnitude_correlation_ci_low': .1,
            'abstained_row_coverage': .25,
            'folds': [
                {'direction_auc': .6, 'magnitude_correlation': .1}
                for _ in range(3)],
        }
        outcome = gate_decision(results, diagnostics, protocol)
        self.assertTrue(outcome['diagnostic_gate_passed'])
        self.assertFalse(outcome['v9a_gate_overridden'])
        self.assertEqual(
            outcome['recommendation'],
            'NEW_NONLINEAR_EXPERT_PREREGISTRATION_SUPPORTED')

    def test_direction_failure_stops_new_hypothesis(self):
        protocol = load_protocol(PROTOCOL)
        primary = {
            cohort: result() for cohort in
            ('incident_full', 'incident', 'primary_control', 'secondary_control')
        }
        results = {'two_part_abstained': primary}
        diagnostics = {
            'direction_auc_ci_low': .5,
            'magnitude_correlation_ci_low': .1,
            'abstained_row_coverage': .25,
            'folds': [
                {'direction_auc': .6, 'magnitude_correlation': .1}
                for _ in range(3)],
        }
        outcome = gate_decision(results, diagnostics, protocol)
        self.assertFalse(outcome['diagnostic_gate_passed'])


if __name__ == '__main__':
    unittest.main()
