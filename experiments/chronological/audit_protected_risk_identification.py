"""Audit frozen-A residual-risk identification without changing point forecasts."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.chronological.audit_expert_benefit import (
    RidgeModel, feature_names,
)
from experiments.chronological.audit_incident_impact_identification import (
    read_labels as read_event_router_labels,
)
from experiments.chronological.audit_incident_impact_node_localization import (
    build_data as build_node_router_data,
)
from experiments.chronological.audit_matched_controls import sha256, write_csv
from experiments.chronological.audit_matched_placebo import (
    load_protocol as load_v5c_protocol,
    verify_inputs as verify_v5c_inputs,
)
from experiments.chronological.audit_node_phase_repeatability import (
    manifest_weeks, rolling_origin_folds,
    verify_inputs as verify_v7a_train_inputs,
)
from experiments.chronological.audit_protected_residual_expert import (
    load_control_training_inputs,
)
from experiments.chronological.audit_routed_residual_decomposition import (
    COHORTS, load_protocol as load_v9b_protocol, make_rows,
    stratified_week_bootstrap, weighted_auc, weighted_correlation,
)
from experiments.chronological.audit_signed_residual_probe import (
    load_residuals, residual_feature_names,
)
from experiments.chronological.materialize_impact_router import materialize_cohort


def load_protocol(path):
    protocol = json.loads(Path(path).read_text(encoding='utf-8'))
    if (protocol.get('protocol_id') !=
            'contra_v8_protected_risk_identification_v10a' or
            protocol.get('scope') !=
            'train_only_rolling_origin_frozen_point_forecast_routed_risk_identification' or
            protocol.get('main_training_ready') is not False):
        raise ValueError('v10a protocol identity or scope changed')
    if [protocol.get(key) for key in ('source_year', 'source_version',
                                      'expected_sensor_count')] != [2023, 8, 496]:
        raise ValueError('v10a source identity changed')
    if (protocol.get('expected_common_train_samples') != 3106 or
            protocol.get('expected_full_positive_train_samples') != 3604):
        raise ValueError('v10a cohort sizes changed')
    expected_v9b = {
        'protocol_id': 'contra_v8_routed_residual_decomposition_v9b',
        'protocol_sha256':
        '7bc06e97d9df10a029133fe8cefb85d3c0b27065f37e499cbbf84daf98b81d43',
        'status': 'ROUTED_RESIDUAL_DECOMPOSITION_AUDIT_COMPLETE',
        'observed_recommendation':
        'ROUTED_RESIDUAL_DIRECTION_OR_MAGNITUDE_NOT_ESTABLISHED',
        'observed_direction_auc': 0.4833916754156446,
        'observed_magnitude_correlation': 0.19080560698706533,
        'observed_magnitude_correlation_ci_low': 0.11347316362646023,
        'role':
        'frozen_cross_fitted_models_and_motivation_not_point_correction_authorization',
    }
    if protocol.get('v9b_input') != expected_v9b:
        raise ValueError('v10a v9b context changed')
    expected_rolling = {
        'time_source': 'positive_incident_t0_iso_week',
        'expected_unique_train_iso_weeks': 35,
        'initial_fit_week_count': 18,
        'audit_fold_count': 3,
        'audit_week_block_sizes': [6, 6, 5],
        'fit_rule':
        'reuse_v9b_models_fit_on_all_weeks_strictly_before_each_audit_block',
        'audit_rule': 'each_of_the_final_17_train_iso_weeks_exactly_once',
        'expected_full_positive_audit_counts': [602, 731, 514],
        'expected_common_audit_counts': [536, 618, 338],
    }
    if protocol.get('rolling_origin') != expected_rolling:
        raise ValueError('v10a rolling-origin design changed')
    if protocol.get('support') != {
            'route_source': 'frozen_v9b_fold_specific_hierarchical_route',
            'active_horizons_zero_based_half_open': [0, 6],
            'phase_slices_zero_based_half_open': [[0, 3], [3, 6]],
            'risk_row': 'routed_event_node_phase',
            'point_forecast': 'frozen_A_unchanged_everywhere',
            'prediction_correction': 'exact_zero_everywhere'}:
        raise ValueError('v10a protected support changed')
    if protocol.get('risk_definition') != {
            'score': 'frozen_v9b_nonnegative_clipped_magnitude_prediction',
            'outcome':
            'absolute_median_signed_residual_per_routed_node_phase',
            'high_risk_label_threshold':
            'fold_fit_incident_outcome_weighted_q75',
            'alert_threshold': 'fold_fit_incident_score_weighted_q75',
            'row_weighting': 'equal_routed_event_weight',
            'primary_population': 'incident_full',
            'common_incident_population': 'incident',
            'routine_negative_controls':
            ['primary_control', 'secondary_control']}:
        raise ValueError('v10a risk definition changed')
    if protocol.get('uncertainty') != {
            'method': 'audit_iso_week_cluster_bootstrap_stratified_by_fold',
            'draws': 2000, 'confidence_level': 0.95, 'seed': 2025}:
        raise ValueError('v10a uncertainty changed')
    expected_gate = {
        'require_incident_full_auc_ci_lower_above': 0.55,
        'require_incident_full_alert_lift_ci_lower_above': 1.2,
        'require_incident_full_magnitude_correlation_ci_lower_above': 0.0,
        'require_each_fold_incident_full_auc_above': 0.5,
        'require_common_incident_auc_above': 0.5,
        'minimum_incident_full_alert_fraction': 0.05,
        'maximum_incident_full_alert_fraction': 0.5,
        'maximum_primary_control_alert_fraction': 0.35,
        'maximum_secondary_control_alert_fraction': 0.35,
        'require_all_audit_weeks_covered_once': True,
        'require_point_forecast_exactly_A': True,
    }
    if protocol.get('development_gate') != expected_gate:
        raise ValueError('v10a development gate changed')
    required = (
        'train_residual_arrays_only', 'validation_residual_arrays_prohibited',
        'test_split_prohibited', 'v9b_fold_models_and_routes_not_refit',
        'audit_weeks_prohibited_from_fold_thresholds',
        'future_weeks_prohibited_from_each_fold_threshold',
        'control_future_Y_prohibited_from_any_model_fit',
        'control_future_Y_read_only_after_models_are_frozen_for_evaluation',
        'point_prediction_modification_prohibited',
        'prediction_interval_claim_prohibited', 'mae_improvement_claim_prohibited',
        'independent_confirmation_claim_prohibited',
        'folds_score_target_thresholds_and_gate_may_not_change_after_result',
        'test_access_requires_a_separate_final_protocol',
    )
    if not all(protocol.get('information_boundary', {}).get(key) is True
               for key in required):
        raise ValueError('v10a information boundary changed')
    return protocol


def _close(observed, expected):
    return np.isclose(float(observed), float(expected), rtol=0., atol=1e-12)


def verify_v9b(v9b_dir, protocol):
    v9b_dir = Path(v9b_dir)
    summary_path = v9b_dir / 'summary.json'
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    expected = protocol['v9b_input']
    diagnostics = summary.get('diagnostics', {})
    gate = summary.get('diagnostic_gate', {})
    if (summary.get('status') != expected['status'] or
            summary.get('protocol_id') != expected['protocol_id'] or
            summary.get('protocol_sha256') != expected['protocol_sha256'] or
            summary.get('validation_residual_arrays_read') is not False or
            summary.get('test_split_read') is not False or
            summary.get('full_train_v8c_routes_used') is not False or
            summary.get('router_and_expert_cross_fitted') is not True or
            summary.get('control_future_Y_used_for_expert_fit') is not False or
            summary.get('v9a_gate_overridden') is not False or
            gate.get('diagnostic_gate_passed') is not False or
            gate.get('recommendation') != expected['observed_recommendation'] or
            not _close(diagnostics.get('direction_auc'),
                       expected['observed_direction_auc']) or
            not _close(diagnostics.get('magnitude_correlation'),
                       expected['observed_magnitude_correlation']) or
            not _close(diagnostics.get('magnitude_correlation_ci_low'),
                       expected['observed_magnitude_correlation_ci_low'])):
        raise ValueError('v9b result identity or boundary changed')
    v9b_protocol = Path(__file__).with_name(
        'routed_residual_decomposition_v9b.json')
    v9b_code = Path(__file__).with_name(
        'audit_routed_residual_decomposition.py')
    if (sha256(v9b_protocol) != expected['protocol_sha256'] or
            summary.get('inputs', {}).get('protocol_sha256') !=
            expected['protocol_sha256'] or
            summary.get('inputs', {}).get('code_sha256') != sha256(v9b_code)):
        raise ValueError('v9b frozen protocol or implementation differs')
    expected_outputs = {
        'fold_models.npz', 'incident_full_decomposition_rows.csv',
        'deployable_policy_event_metrics.csv'}
    if set(summary.get('outputs', {})) != expected_outputs:
        raise ValueError('v9b output inventory changed')
    for name, metadata in summary['outputs'].items():
        path = v9b_dir / name
        if (sha256(path) != metadata.get('sha256') or
                path.stat().st_size != metadata.get('bytes')):
            raise ValueError(f'v9b output differs: {name}')
    return summary, v9b_dir / 'fold_models.npz'


def _scalar(stored, name):
    value = np.asarray(stored[name])
    if value.size != 1 or not np.isfinite(value).all():
        raise ValueError(f'Invalid scalar in v9b models: {name}')
    return float(value.reshape(-1)[0])


def load_ridge(stored, prefix, expected_features):
    required = [
        f'{prefix}_mean', f'{prefix}_scale', f'{prefix}_target_mean',
        f'{prefix}_coefficient', f'{prefix}_ridge_alpha']
    if any(name not in stored.files for name in required):
        raise ValueError(f'Missing v9b ridge arrays: {prefix}')
    mean = np.asarray(stored[f'{prefix}_mean'], dtype=np.float64)
    scale = np.asarray(stored[f'{prefix}_scale'], dtype=np.float64)
    coefficient = np.asarray(
        stored[f'{prefix}_coefficient'], dtype=np.float64)
    if (mean.shape != (expected_features,) or scale.shape != mean.shape or
            coefficient.shape != mean.shape or not np.isfinite(mean).all() or
            not np.isfinite(scale).all() or not np.isfinite(coefficient).all() or
            np.any(scale <= 0)):
        raise ValueError(f'Invalid v9b ridge model: {prefix}')
    return RidgeModel(
        mean, scale, _scalar(stored, f'{prefix}_target_mean'), coefficient,
        _scalar(stored, f'{prefix}_ridge_alpha'))


def load_fold_models(path, protocol):
    artifacts = []
    with np.load(path, allow_pickle=False) as stored:
        schema_keys = (
            'feature_names', 'event_router_feature_names',
            'node_router_feature_names')
        if any(key not in stored.files for key in schema_keys):
            raise ValueError('v9b model feature schemas are missing')
        schemas = {key: stored[key].tolist() for key in schema_keys}
        if any(not values or not all(isinstance(value, str) for value in values)
               for values in schemas.values()):
            raise ValueError('v9b model feature schemas are invalid')
        for fold in range(1, protocol['rolling_origin']['audit_fold_count'] + 1):
            prefix = f'fold_{fold}'
            artifacts.append({
                'fold': fold,
                'event_model': load_ridge(
                    stored, f'{prefix}_event_router',
                    len(schemas['event_router_feature_names'])),
                'node_model': load_ridge(
                    stored, f'{prefix}_node_router',
                    len(schemas['node_router_feature_names'])),
                'magnitude_model': load_ridge(
                    stored, f'{prefix}_magnitude',
                    len(schemas['feature_names'])),
                'event_route_threshold': _scalar(
                    stored, f'{prefix}_event_route_threshold'),
                'node_route_threshold': _scalar(
                    stored, f'{prefix}_node_route_threshold'),
                'clip_abs': _scalar(stored, f'{prefix}_signed_clip_abs'),
            })
    return artifacts, schemas


def weighted_quantile(values, weights, quantile):
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if (values.ndim != 1 or weights.shape != values.shape or not len(values) or
            not np.isfinite(values).all() or not np.isfinite(weights).all() or
            np.any(weights <= 0) or not 0 <= quantile <= 1):
        raise ValueError('Invalid weighted quantile input')
    order = np.argsort(values, kind='stable')
    cumulative = np.cumsum(weights[order])
    position = float(quantile) * weights.sum()
    index = min(int(np.searchsorted(cumulative, position, side='left')),
                len(order) - 1)
    return float(values[order[index]])


def weighted_alert_lift(labels, alerts, weights):
    labels = np.asarray(labels, dtype=np.int64)
    alerts = np.asarray(alerts, dtype=bool)
    weights = np.asarray(weights, dtype=np.float64)
    if (labels.ndim != 1 or alerts.shape != labels.shape or
            weights.shape != labels.shape or set(labels.tolist()) - {0, 1} or
            not np.isfinite(weights).all() or np.any(weights <= 0)):
        raise ValueError('Invalid weighted alert-lift input')
    prevalence = float(np.average(labels, weights=weights))
    if prevalence <= 0 or not alerts.any():
        raise ValueError('Alert lift requires positives and active alerts')
    selected = float(np.average(labels[alerts], weights=weights[alerts]))
    return selected / prevalence


def event_equal_weights(metadata):
    if not metadata:
        raise ValueError('Event-equal weights require evaluation rows')
    counts = {}
    for position, _, _, _ in metadata:
        counts[int(position)] = counts.get(int(position), 0) + 1
    weights = np.asarray(
        [1. / counts[int(position)] for position, _, _, _ in metadata],
        dtype=np.float64)
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError('Invalid event-equal evaluation weights')
    return weights


def risk_metrics(rows, protocol, seed):
    score_all = np.asarray([row['risk_score'] for row in rows], dtype=np.float64)
    alert_all = np.asarray([row['risk_alert'] for row in rows], dtype=bool)
    weight_all = np.asarray([row['weight'] for row in rows], dtype=np.float64)
    target_all = np.asarray([
        np.nan if row['target_magnitude'] == '' else row['target_magnitude']
        for row in rows], dtype=np.float64)
    weeks_all = np.asarray([row['week'] for row in rows])
    folds_all = np.asarray([row['fold'] for row in rows], dtype=np.int64)
    finite = np.isfinite(target_all)
    if not finite.any():
        raise ValueError('Risk metrics have no evaluable targets')
    target = target_all[finite]
    score = score_all[finite]
    alert = alert_all[finite]
    weeks = weeks_all[finite]
    folds = folds_all[finite]
    evaluable_rows = [row for row, keep in zip(rows, finite) if keep]
    weight = np.empty(len(evaluable_rows), dtype=np.float64)
    counts = {}
    for row in evaluable_rows:
        key = (row['fold'], row['positive_sample_index'])
        counts[key] = counts.get(key, 0) + 1
    for index, row in enumerate(evaluable_rows):
        weight[index] = 1. / counts[(row['fold'], row['positive_sample_index'])]
    label = np.asarray(
        [row['high_risk_label'] for row in evaluable_rows], dtype=np.int64)
    auc = weighted_auc(label, score, weight)
    correlation = weighted_correlation(target, score, weight)
    lift = weighted_alert_lift(label, alert, weight)
    auc_ci = stratified_week_bootstrap(
        label, score, weight, weeks, folds, weighted_auc, protocol, seed)
    correlation_ci = stratified_week_bootstrap(
        target, score, weight, weeks, folds, weighted_correlation,
        protocol, seed + 1)
    lift_ci = stratified_week_bootstrap(
        label, alert.astype(np.float64), weight, weeks, folds,
        lambda current_label, current_alert, current_weight:
        weighted_alert_lift(current_label, current_alert > .5, current_weight),
        protocol, seed + 2)
    fold_results = []
    for fold in np.unique(folds_all):
        selected = folds == fold
        selected_all = folds_all == fold
        fold_results.append({
            'fold': int(fold), 'rows': int(selected.sum()),
            'roc_auc': weighted_auc(
                label[selected], score[selected], weight[selected]),
            'magnitude_correlation': weighted_correlation(
                target[selected], score[selected], weight[selected]),
            'positive_fraction': float(np.average(
                label[selected], weights=weight[selected])),
            'alert_fraction': float(np.average(
                alert_all[selected_all], weights=weight_all[selected_all])),
            'alert_lift': weighted_alert_lift(
                label[selected], alert[selected], weight[selected]),
        })
    return {
        'rows': int(len(rows)), 'evaluable_rows': int(finite.sum()),
        'events': int(len({(row['fold'], row['positive_sample_index'])
                           for row in rows})),
        'positive_fraction': float(np.average(label, weights=weight)),
        'alert_fraction': float(np.average(alert_all, weights=weight_all)),
        'roc_auc': auc, 'roc_auc_ci_low': auc_ci[0],
        'roc_auc_ci_high': auc_ci[1],
        'alert_lift': lift, 'alert_lift_ci_low': lift_ci[0],
        'alert_lift_ci_high': lift_ci[1],
        'magnitude_correlation': correlation,
        'magnitude_correlation_ci_low': correlation_ci[0],
        'magnitude_correlation_ci_high': correlation_ci[1],
        'folds': fold_results,
    }


def gate_decision(results, audit_weeks_covered_once, point_forecast_exactly_A,
                  protocol):
    gate = protocol['development_gate']
    full = results['incident_full']
    checks = {
        'incident_full_auc_ci_lower':
            full['roc_auc_ci_low'] >
            gate['require_incident_full_auc_ci_lower_above'],
        'incident_full_alert_lift_ci_lower':
            full['alert_lift_ci_low'] >
            gate['require_incident_full_alert_lift_ci_lower_above'],
        'incident_full_magnitude_correlation_ci_lower':
            full['magnitude_correlation_ci_low'] >
            gate['require_incident_full_magnitude_correlation_ci_lower_above'],
        'each_fold_incident_full_auc': all(
            fold['roc_auc'] > gate['require_each_fold_incident_full_auc_above']
            for fold in full['folds']),
        'common_incident_auc':
            results['incident']['roc_auc'] >
            gate['require_common_incident_auc_above'],
        'incident_full_alert_fraction_minimum':
            full['alert_fraction'] >=
            gate['minimum_incident_full_alert_fraction'],
        'incident_full_alert_fraction_maximum':
            full['alert_fraction'] <=
            gate['maximum_incident_full_alert_fraction'],
        'primary_control_alert_fraction':
            results['primary_control']['alert_fraction'] <=
            gate['maximum_primary_control_alert_fraction'],
        'secondary_control_alert_fraction':
            results['secondary_control']['alert_fraction'] <=
            gate['maximum_secondary_control_alert_fraction'],
        'all_audit_weeks_covered_once': bool(audit_weeks_covered_once),
        'point_forecast_exactly_A': bool(point_forecast_exactly_A),
    }
    passed = all(checks.values())
    return {
        'checks': checks,
        'protected_risk_identification_gate_passed': passed,
        'recommendation': (
            'PROTECTED_INTERVAL_CALIBRATION_AUDIT_ALLOWED' if passed else
            'STOP_ROUTED_RISK_IDENTIFICATION_NOT_ESTABLISHED'),
    }


def audit(data_dir, primary_dir, secondary_dir, residual_dir, placebo_dir,
          sensors_path, v9b_dir, protocol_path, output):
    output = Path(output)
    partial = output.with_name(output.name + '.partial')
    if output.exists() or partial.exists():
        raise FileExistsError('v10a output or partial output exists; use a new directory')
    protocol = load_protocol(protocol_path)
    v9b_summary, model_path = verify_v9b(v9b_dir, protocol)
    artifacts, model_schemas = load_fold_models(model_path, protocol)
    v9b_protocol_path = Path(__file__).with_name(
        'routed_residual_decomposition_v9b.json')
    v9b_protocol = load_v9b_protocol(v9b_protocol_path)
    v7a_summary, package_hashes, residual_input_hashes = \
        verify_v7a_train_inputs(
            data_dir, primary_dir, secondary_dir, residual_dir, v9b_protocol)
    v5c_path = Path(__file__).with_name('matched_placebo_audit_v5c.json')
    v5c = load_v5c_protocol(v5c_path)
    context = v9b_protocol['v5c_context']
    if (sha256(Path(placebo_dir) / 'summary.json') !=
            context['summary_sha256'] or
            sha256(Path(placebo_dir) / 'triple_metrics.csv') !=
            context['triple_metrics_sha256'] or
            sha256(sensors_path) != context['sensor_metadata_sha256']):
        raise ValueError('v10a placebo or sensor context differs from v9b')
    matched_input_hashes = verify_v5c_inputs(
        data_dir, primary_dir, secondary_dir, v5c)
    print(json.dumps({'stage': 'inputs_verified'}), flush=True)

    nodes = protocol['expected_sensor_count']
    common_count = protocol['expected_common_train_samples']
    full_count = protocol['expected_full_positive_train_samples']
    full_arrays = load_residuals(
        residual_dir, 'train', 'incident_full', full_count, nodes)
    common_identity = load_control_training_inputs(
        residual_dir, 'train', 'primary_control', common_count, nodes)
    full_weeks, common_weeks = manifest_weeks(
        data_dir, secondary_dir, full_arrays['positive_sample_index'],
        common_identity['positive_sample_index'])
    folds = rolling_origin_folds(full_weeks, common_weeks, v9b_protocol)
    samples, label_weeks, _ = read_event_router_labels(
        placebo_dir, common_count,
        protocol['rolling_origin']['expected_unique_train_iso_weeks'])
    scaler = json.loads((Path(data_dir) / 'scaler.json').read_text(encoding='utf-8'))
    source, categories, _, _, _, _, source_weeks = build_node_router_data(
        data_dir, primary_dir, secondary_dir, sensors_path, samples, nodes,
        float(scaler['std']))
    if (not np.array_equal(samples, common_identity['positive_sample_index']) or
            not np.array_equal(label_weeks, common_weeks) or
            not np.array_equal(source_weeks, common_weeks)):
        raise ValueError('v10a source identities or weeks differ')
    expected_schemas = {
        'feature_names':
            residual_feature_names(categories, 'event_node_phase'),
        'event_router_feature_names': feature_names(categories, 'event'),
        'node_router_feature_names': feature_names(categories, 'event_node'),
    }
    if model_schemas != expected_schemas:
        raise ValueError('v9b model feature schemas differ from frozen inputs')
    print(json.dumps({
        'stage': 'frozen_models_loaded', 'folds': len(artifacts),
        'v9b_models_refit': False,
    }), flush=True)

    # Future control outcomes are loaded only after every frozen model is present.
    evaluation_arrays = {
        'incident_full': full_arrays,
        'incident': load_residuals(
            residual_dir, 'train', 'incident', common_count, nodes),
        'primary_control': load_residuals(
            residual_dir, 'train', 'primary_control', common_count, nodes),
        'secondary_control': load_residuals(
            residual_dir, 'train', 'secondary_control', common_count, nodes),
    }
    accumulated = {cohort: [] for cohort in COHORTS}
    fold_summaries = []
    csv_rows = []
    for artifact, fold in zip(artifacts, folds):
        if artifact['fold'] != fold['fold']:
            raise ValueError('v10a fold models and rolling folds differ')
        full_fit_indices = np.flatnonzero(
            np.isin(full_weeks, fold['fit_weeks']))
        if (not len(full_fit_indices) or
                np.intersect1d(
                    full_fit_indices, fold['full_audit_indices']).size):
            raise ValueError('v10a full-positive fit/audit separation changed')
        routes = {
            cohort: materialize_cohort(
                source, cohort,
                full_count if cohort == 'incident_full' else common_count,
                nodes, categories, artifact['event_model'],
                artifact['event_route_threshold'], artifact['node_model'],
                artifact['node_route_threshold'])
            for cohort in COHORTS
        }
        fit_features, fit_targets, fit_weights, _, _ = make_rows(
            source, 'incident_full', full_arrays, routes['incident_full'],
            categories, full_fit_indices, 'residual')
        fit_scores = np.clip(
            artifact['magnitude_model'].predict(fit_features),
            0., artifact['clip_abs'])
        target_threshold = weighted_quantile(
            np.abs(fit_targets), fit_weights, .75)
        score_threshold = weighted_quantile(fit_scores, fit_weights, .75)
        current_summary = {
            'fold': fold['fold'], 'fit_weeks': list(fold['fit_weeks']),
            'audit_weeks': list(fold['audit_weeks']),
            'fit_full_positive_events': int(len(full_fit_indices)),
            'audit_full_positive_events': int(len(fold['full_audit_indices'])),
            'audit_common_events': int(len(fold['common_audit_indices'])),
            'fit_risk_target_weighted_q75': target_threshold,
            'fit_risk_score_weighted_q75': score_threshold,
            'cohorts': {},
        }
        for cohort in COHORTS:
            positions = (fold['full_audit_indices'] if cohort == 'incident_full'
                         else fold['common_audit_indices'])
            features, targets, _, metadata, routed_events = make_rows(
                source, cohort, evaluation_arrays[cohort], routes[cohort],
                categories, positions, 'evaluation')
            scores = np.clip(
                artifact['magnitude_model'].predict(features),
                0., artifact['clip_abs'])
            finite = np.isfinite(targets)
            current_summary['cohorts'][cohort] = {
                'eligible_events': int(len(positions)),
                'routed_events': int(routed_events),
                'rows': int(len(targets)),
                'evaluable_rows': int(finite.sum()),
            }
            prediction_weights = event_equal_weights(metadata)
            for target, score, weight, row in zip(
                    targets, scores, prediction_weights, metadata):
                _, node, phase, sample = row
                evaluable = bool(np.isfinite(target))
                record = {
                    'cohort': cohort, 'fold': fold['fold'],
                    'positive_sample_index': int(sample['sample']),
                    'week': sample['iso_week'], 'node_index': int(node),
                    'phase': int(phase),
                    'target_magnitude': float(abs(target)) if evaluable else '',
                    'risk_score': float(score),
                    'high_risk_label': (
                        bool(abs(target) >= target_threshold) if evaluable else ''),
                    'risk_alert': bool(score >= score_threshold),
                    'weight': float(weight),
                }
                accumulated[cohort].append(record)
                csv_rows.append(record)
        fold_summaries.append(current_summary)
        print(json.dumps({
            'stage': 'fold_evaluation_complete', 'fold': fold['fold'],
            'fit_target_q75': target_threshold,
            'fit_score_q75': score_threshold,
        }), flush=True)

    results = {
        cohort: risk_metrics(
            rows, protocol,
            int(protocol['uncertainty']['seed']) + number * 100)
        for number, (cohort, rows) in enumerate(accumulated.items())
    }
    gate = gate_decision(results, True, True, protocol)
    partial.mkdir(parents=True, exist_ok=False)
    rows_path = partial / 'risk_rows.csv'
    write_csv(rows_path, csv_rows, [
        'cohort', 'fold', 'positive_sample_index', 'week', 'node_index',
        'phase', 'target_magnitude', 'risk_score', 'high_risk_label',
        'risk_alert', 'weight'])
    summary = {
        'status': 'PROTECTED_RISK_IDENTIFICATION_AUDIT_COMPLETE',
        'scope': protocol['scope'], 'protocol_id': protocol['protocol_id'],
        'protocol_sha256': sha256(protocol_path), 'main_training_ready': False,
        'train_residual_arrays_only': True,
        'validation_residual_arrays_read': False, 'test_split_read': False,
        'v9b_fold_models_refit': False,
        'control_future_Y_used_for_model_fit': False,
        'control_future_Y_read_for_evaluation': True,
        'point_forecast_modified': False,
        'point_forecast_exactly_A': True,
        'prediction_interval_constructed': False,
        'mae_improvement_evaluated': False,
        'independent_confirmation': False,
        'folds': fold_summaries, 'results': results,
        'development_gate': gate,
        'inputs': {
            'v9b_summary_sha256': sha256(Path(v9b_dir) / 'summary.json'),
            'v9b_fold_models_sha256': sha256(model_path),
            'v9b_protocol_sha256': v9b_summary['protocol_sha256'],
            'v7a_summary_sha256': sha256(Path(residual_dir) / 'summary.json'),
            'v7a_protocol_sha256': v7a_summary['protocol_sha256'],
            'positive_package': package_hashes,
            'residual_training_inputs': residual_input_hashes,
            'matched_inputs': {
                'positive': matched_input_hashes[0],
                'primary': matched_input_hashes[1],
                'secondary': matched_input_hashes[2],
            },
            'v5c_summary_sha256': sha256(Path(placebo_dir) / 'summary.json'),
            'v5c_triple_metrics_sha256':
                sha256(Path(placebo_dir) / 'triple_metrics.csv'),
            'protocol_sha256': sha256(protocol_path),
            'code_sha256': sha256(__file__),
        },
        'outputs': {
            rows_path.name: {
                'sha256': sha256(rows_path), 'bytes': rows_path.stat().st_size,
                'rows': len(csv_rows),
            }
        },
        'environment': {
            'python_version': sys.version, 'numpy_version': np.__version__,
            'threads_requested': {
                name: os.environ.get(name) for name in
                ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')},
            'git_head': subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
        },
        'interpretation': protocol['interpretation'],
    }
    (partial / 'summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
        encoding='utf-8')
    partial.rename(output)
    print(json.dumps({
        'status': summary['status'], 'development_gate': gate,
        'results': results,
    }, ensure_ascii=False, indent=2), flush=True)
    print(f'Saved v10a protected-risk audit: {output / "summary.json"}',
          flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--primary-control-dir', type=Path, required=True)
    parser.add_argument('--secondary-control-dir', type=Path, required=True)
    parser.add_argument('--residual-dir', type=Path, required=True)
    parser.add_argument('--placebo-dir', type=Path, required=True)
    parser.add_argument('--sensors', type=Path, required=True)
    parser.add_argument('--v9b-dir', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, default=Path(__file__).with_name(
        'protected_risk_identification_v10a.json'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    audit(
        args.data_dir, args.primary_control_dir, args.secondary_control_dir,
        args.residual_dir, args.placebo_dir, args.sensors, args.v9b_dir,
        args.protocol, args.output)


if __name__ == '__main__':
    main()
