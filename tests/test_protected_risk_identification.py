"""Tests for the v10a protected residual-risk identification audit."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.chronological.audit_protected_risk_identification import (
    event_equal_weights, gate_decision, load_fold_models, load_protocol,
    risk_metrics, weighted_alert_lift, weighted_quantile,
)


REPO = Path(__file__).resolve().parents[1]
PROTOCOL = (REPO / 'experiments/chronological' /
            'protected_risk_identification_v10a.json')


class ProtocolTests(unittest.TestCase):
    def test_protocol_preserves_A_and_reuses_frozen_v9b_models(self):
        protocol = load_protocol(PROTOCOL)
        self.assertEqual(
            protocol['support']['point_forecast'],
            'frozen_A_unchanged_everywhere')
        self.assertTrue(protocol['information_boundary'][
            'v9b_fold_models_and_routes_not_refit'])
        self.assertTrue(protocol['information_boundary'][
            'prediction_interval_claim_prohibited'])
        self.assertTrue(protocol['information_boundary'][
            'mae_improvement_claim_prohibited'])
        self.assertTrue(protocol['information_boundary'][
            'independent_confirmation_claim_prohibited'])

    def test_protocol_rejects_post_result_auc_weakening(self):
        protocol = json.loads(PROTOCOL.read_text(encoding='utf-8'))
        protocol['development_gate'][
            'require_incident_full_auc_ci_lower_above'] = .5
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'changed.json'
            path.write_text(json.dumps(protocol), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'development gate changed'):
                load_protocol(path)


class MetricTests(unittest.TestCase):
    @staticmethod
    def ridge_arrays(prefix, features):
        return {
            f'{prefix}_mean': np.zeros(features),
            f'{prefix}_scale': np.ones(features),
            f'{prefix}_target_mean': np.asarray(1.),
            f'{prefix}_coefficient': np.zeros(features),
            f'{prefix}_ridge_alpha': np.asarray(10.),
        }

    def test_frozen_fold_models_reload_without_refitting(self):
        protocol = load_protocol(PROTOCOL)
        arrays = {
            'feature_names': np.asarray(['expert-a', 'expert-b']),
            'event_router_feature_names': np.asarray(['event']),
            'node_router_feature_names': np.asarray(['node-a', 'node-b']),
        }
        for fold in range(1, 4):
            prefix = f'fold_{fold}'
            arrays.update(self.ridge_arrays(f'{prefix}_event_router', 1))
            arrays.update(self.ridge_arrays(f'{prefix}_node_router', 2))
            arrays.update(self.ridge_arrays(f'{prefix}_magnitude', 2))
            arrays[f'{prefix}_event_route_threshold'] = np.asarray(.2)
            arrays[f'{prefix}_node_route_threshold'] = np.asarray(.3)
            arrays[f'{prefix}_signed_clip_abs'] = np.asarray(20.)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'models.npz'
            np.savez_compressed(path, **arrays)
            artifacts, schemas = load_fold_models(path, protocol)
        self.assertEqual(len(artifacts), 3)
        self.assertEqual(schemas['feature_names'], ['expert-a', 'expert-b'])
        np.testing.assert_array_equal(
            artifacts[0]['magnitude_model'].predict(np.ones((1, 2))), [1.])

    def test_weighted_quantile_respects_event_weights(self):
        self.assertEqual(weighted_quantile(
            [1., 2., 100.], [1., .1, .1], .75), 1.)

    def test_weighted_alert_lift_uses_only_alerted_rows(self):
        lift = weighted_alert_lift(
            [0, 0, 1, 1], [False, False, True, True], np.ones(4))
        self.assertEqual(lift, 2.)

    def test_evaluable_rows_are_renormalized_per_event(self):
        metadata = [
            (10, 1, 0, {}),
            (20, 2, 0, {}),
            (20, 2, 1, {}),
        ]
        np.testing.assert_allclose(
            event_equal_weights(metadata), [1., .5, .5])

    def test_risk_metrics_bootstrap_is_fold_stratified(self):
        rows = []
        for fold in range(1, 4):
            for week in range(2):
                for index, target in enumerate((1., 2., 5., 6.)):
                    rows.append({
                        'fold': fold,
                        'positive_sample_index': fold * 100 + week * 10 + index,
                        'week': f'f{fold}-w{week}',
                        'target_magnitude': target,
                        'risk_score': target,
                        'high_risk_label': target >= 5.,
                        'risk_alert': target >= 5.,
                        'weight': 1.,
                    })
                rows.append({
                    'fold': fold,
                    'positive_sample_index': fold * 100 + week * 10 + 9,
                    'week': f'f{fold}-w{week}',
                    'target_magnitude': '',
                    'risk_score': 0.,
                    'high_risk_label': '',
                    'risk_alert': False,
                    'weight': 1.,
                })
        protocol = load_protocol(PROTOCOL)
        protocol['uncertainty'] = {
            'method': 'audit_iso_week_cluster_bootstrap_stratified_by_fold',
            'draws': 100, 'confidence_level': .95, 'seed': 2025}
        result = risk_metrics(rows, protocol, 2025)
        self.assertEqual(result['roc_auc'], 1.)
        self.assertEqual(result['alert_lift'], 2.)
        self.assertEqual(result['magnitude_correlation'], 1.)
        self.assertEqual(result['evaluable_rows'], 24)
        self.assertEqual(result['rows'], 30)
        self.assertAlmostEqual(result['alert_fraction'], .4)
        self.assertEqual(len(result['folds']), 3)


def metric(auc=.7, auc_low=.6, lift_low=1.3, correlation_low=.1,
           alert_fraction=.25):
    return {
        'roc_auc': auc, 'roc_auc_ci_low': auc_low,
        'alert_lift_ci_low': lift_low,
        'magnitude_correlation_ci_low': correlation_low,
        'alert_fraction': alert_fraction,
        'folds': [{'roc_auc': .6} for _ in range(3)],
    }


class DecisionTests(unittest.TestCase):
    def test_pass_authorizes_only_interval_calibration_audit(self):
        protocol = load_protocol(PROTOCOL)
        results = {
            'incident_full': metric(), 'incident': metric(),
            'primary_control': metric(alert_fraction=.2),
            'secondary_control': metric(alert_fraction=.2),
        }
        decision = gate_decision(results, True, True, protocol)
        self.assertTrue(decision[
            'protected_risk_identification_gate_passed'])
        self.assertEqual(
            decision['recommendation'],
            'PROTECTED_INTERVAL_CALIBRATION_AUDIT_ALLOWED')

    def test_point_forecast_change_always_fails_gate(self):
        protocol = load_protocol(PROTOCOL)
        results = {
            'incident_full': metric(), 'incident': metric(),
            'primary_control': metric(alert_fraction=.2),
            'secondary_control': metric(alert_fraction=.2),
        }
        decision = gate_decision(results, True, False, protocol)
        self.assertFalse(decision[
            'protected_risk_identification_gate_passed'])
        self.assertFalse(decision['checks']['point_forecast_exactly_A'])

    def test_unstable_fold_or_routine_overalert_fails_gate(self):
        protocol = load_protocol(PROTOCOL)
        full = metric()
        full['folds'][1]['roc_auc'] = .49
        results = {
            'incident_full': full, 'incident': metric(),
            'primary_control': metric(alert_fraction=.36),
            'secondary_control': metric(alert_fraction=.2),
        }
        decision = gate_decision(results, True, True, protocol)
        self.assertFalse(decision[
            'protected_risk_identification_gate_passed'])
        self.assertFalse(decision['checks']['each_fold_incident_full_auc'])
        self.assertFalse(decision['checks']['primary_control_alert_fraction'])


if __name__ == '__main__':
    unittest.main()
