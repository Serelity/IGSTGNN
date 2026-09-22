"""Cross-fit routers and decompose routed residual direction and magnitude."""

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
    feature_names, fit_weighted_ridge,
)
from experiments.chronological.audit_incident_impact_identification import (
    build_features as build_event_router_features,
    fit_fold as fit_event_router,
    read_labels as read_event_router_labels,
)
from experiments.chronological.audit_incident_impact_node_localization import (
    build_data as build_node_router_data,
    fit_node_model as fit_node_router,
)
from experiments.chronological.audit_matched_controls import sha256, write_csv
from experiments.chronological.audit_matched_placebo import (
    load_protocol as load_v5c_protocol,
    verify_inputs as verify_v5c_inputs,
)
from experiments.chronological.audit_node_phase_repeatability import (
    concatenate_arrays, manifest_weeks, rolling_origin_folds, subset_arrays,
    verify_inputs as verify_v7a_train_inputs,
)
from experiments.chronological.audit_protected_residual_expert import (
    load_control_training_inputs, protected_metrics, uncertainty,
)
from experiments.chronological.audit_signed_residual_probe import (
    PHASES, evaluation_target_for, load_residuals, residual_feature_names,
    residual_feature_vector, target_for,
)
from experiments.chronological.materialize_impact_router import materialize_cohort


COHORTS = ('incident_full', 'incident', 'primary_control', 'secondary_control')
TRAIN_COHORTS = ('incident_full', 'primary_control', 'secondary_control')
DEPLOYABLE_POLICIES = ('direct_signed', 'two_part_all', 'two_part_abstained')
ORACLE_POLICIES = ('oracle_direction', 'oracle_magnitude')


def load_protocol(path):
    protocol = json.loads(Path(path).read_text(encoding='utf-8'))
    if (protocol.get('protocol_id') !=
            'contra_v8_routed_residual_decomposition_v9b' or
            protocol.get('scope') !=
            'train_only_rolling_origin_cross_fitted_router_residual_decomposition' or
            protocol.get('main_training_ready') is not False):
        raise ValueError('v9b protocol identity or scope changed')
    if [protocol.get(key) for key in ('source_year', 'source_version',
                                      'expected_sensor_count')] != [2023, 8, 496]:
        raise ValueError('v9b source identity changed')
    if (protocol.get('expected_common_train_samples') != 3106 or
            protocol.get('expected_full_positive_train_samples') != 3604):
        raise ValueError('v9b cohort sizes changed')
    if protocol.get('v7a_input') != {
            'protocol_id': 'contra_v8_signed_residual_materialize_v7a',
            'protocol_sha256':
            'c3d547679b9c6c97ad29720578ddc46a2460d7e9ba336114c59aba429ac03541',
            'status': 'SIGNED_RESIDUAL_MATERIALIZATION_COMPLETE'}:
        raise ValueError('v9b v7a input changed')
    if protocol.get('router_design') != {
            'event_protocol_sha256':
            '1462d7b2a4d51f947ede97f772d2ae002f008698f1f3d4325f7cadeea16576e6',
            'node_protocol_sha256':
            '318a4701c2dc3754656d5653af9f37a841c5969899e2df4f2bee93d11de08c0c',
            'fit': 'refit_v8a_event_and_v8b_node_router_on_each_fold_past_weeks',
            'route': 'event_route_and_node_route_and_candidate_mask',
            'full_train_v8c_routes_used': False}:
        raise ValueError('v9b cross-fitted router design changed')
    if protocol.get('v5c_context') != {
            'protocol_id': 'contra_v8_matched_placebo_audit_v5c',
            'summary_sha256':
            'c443581334f1f0aebece82c910a38c78b1dc29a22565d081f6a583298875571a',
            'triple_metrics_sha256':
            '192b44e82223a9f943f3b47a5141ea1d19819b7cd0d0c75b0c234d372dcb5bc5',
            'sensor_metadata_sha256':
            '682f3cdf75e643f0b37356ab69cbabb27389be5089f41d3b2cbc4bede3332094'}:
        raise ValueError('v9b placebo and sensor context changed')
    if protocol.get('v9a_context') != {
            'protocol_id': 'contra_v8_protected_residual_expert_v9a',
            'protocol_sha256':
            'bec1f27bf5d45376ad8becab429d31e4a63d493926172aa1be87c0df240186b3',
            'observed_recommendation':
            'STOP_SHALLOW_PROTECTED_RESIDUAL_EXPERT_NOT_VALIDATED',
            'observed_full_positive_routed_gain_raw_mae': 0.07415722674564051,
            'observed_full_positive_routed_gain_ci_low': -0.05854695574359848,
            'role': 'motivation_only_not_an_input'}:
        raise ValueError('v9b v9a context changed')
    if protocol.get('rolling_origin') != {
            'time_source': 'positive_incident_t0_iso_week',
            'expected_unique_train_iso_weeks': 35,
            'initial_fit_week_count': 18, 'audit_fold_count': 3,
            'audit_week_block_sizes': [6, 6, 5],
            'fit_rule': 'all_train_iso_weeks_strictly_before_each_audit_block',
            'audit_rule': 'each_of_the_final_17_train_iso_weeks_exactly_once',
            'expected_full_positive_audit_counts': [602, 731, 514],
            'expected_common_audit_counts': [536, 618, 338]}:
        raise ValueError('v9b rolling-origin design changed')
    if protocol.get('support') != {
            'route_source': 'fold_past_only_cross_fitted_v8a_and_v8b_router',
            'active_horizons_zero_based_half_open': [0, 6],
            'phase_slices_zero_based_half_open': [[0, 3], [3, 6]],
            'protected_horizons_zero_based_half_open': [6, 12],
            'protected_noncandidate_nodes': True,
            'protected_unrouted_candidate_nodes': True,
            'correction_exactly_zero_outside_hierarchical_route': True}:
        raise ValueError('v9b support changed')
    if protocol.get('expert') != {
            'features':
            'v9a_report_time_event_node_phase_plus_frozen_A_prediction',
            'ridge_alpha': 10.0,
            'training_cohorts': list(TRAIN_COHORTS),
            'row_weighting': 'equal_cohort_and_equal_routed_event_weight',
            'signed_target':
            'incident_median_signed_residual_and_control_exact_zero',
            'direction_target':
            'incident_sign_of_median_signed_residual_and_control_exact_zero',
            'magnitude_target':
            'incident_absolute_median_signed_residual_and_control_exact_zero',
            'signed_clip': 'fit_incident_target_abs_q99',
            'magnitude_clip': 'fit_incident_target_abs_q99',
            'direction_score_clip': [-1.0, 1.0],
            'abstention_threshold':
            'fit_incident_absolute_direction_score_q75'}:
        raise ValueError('v9b expert decomposition changed')
    if protocol.get('policies') != {
            'direct_signed': 'clipped_signed_ridge_prediction',
            'two_part_all':
            'sign_of_direction_score_times_nonnegative_magnitude_prediction',
            'two_part_abstained':
            'two_part_all_only_above_fit_direction_confidence_q75',
            'oracle_direction':
            'future_direction_times_predicted_magnitude_diagnostic_only',
            'oracle_magnitude':
            'predicted_direction_times_future_magnitude_diagnostic_only'}:
        raise ValueError('v9b policies changed')
    if protocol.get('uncertainty') != {
            'method': 'audit_iso_week_cluster_bootstrap', 'draws': 2000,
            'confidence_level': 0.95, 'seed': 2025}:
        raise ValueError('v9b uncertainty changed')
    if protocol.get('diagnostic_gate') != {
            'primary_policy': 'two_part_abstained',
            'require_direction_auc_ci_lower_above': 0.55,
            'require_magnitude_correlation_ci_lower_above': 0.0,
            'require_each_fold_direction_auc_above': 0.5,
            'require_each_fold_magnitude_correlation_above': 0.0,
            'require_full_positive_global_point_improvement': True,
            'maximum_full_positive_global_harm_fraction_of_A_mae': 0.001,
            'require_full_positive_routed_improvement_ci_lower_above_zero': True,
            'require_common_incident_routed_point_improvement': True,
            'maximum_each_control_routed_harm_fraction_of_A_mae': 0.005,
            'minimum_abstained_row_coverage': 0.05,
            'require_exact_A_outside_hierarchical_route': True}:
        raise ValueError('v9b diagnostic gate changed')
    required = (
        'train_residual_arrays_only', 'validation_residual_arrays_prohibited',
        'test_split_prohibited', 'router_and_expert_fit_on_past_weeks_only',
        'audit_weeks_prohibited_from_router_and_expert_fit',
        'future_weeks_prohibited_from_each_fold_fit',
        'control_future_Y_prohibited_from_expert_fit',
        'oracle_policies_prohibited_from_performance_claims',
        'v9a_gate_not_overridden', 'neural_expert_training_prohibited',
        'folds_features_policies_thresholds_and_gate_may_not_change_after_result')
    if not all(protocol.get('information_boundary', {}).get(key) is True
               for key in required):
        raise ValueError('v9b information boundary changed')
    return protocol


def weighted_auc(labels, scores, weights):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if (labels.ndim != 1 or scores.shape != labels.shape or
            weights.shape != labels.shape or set(labels.tolist()) - {0, 1} or
            not np.isfinite(scores).all() or not np.isfinite(weights).all() or
            np.any(weights <= 0)):
        raise ValueError('Invalid weighted AUC input')
    positive_weight = float(weights[labels == 1].sum())
    negative_weight = float(weights[labels == 0].sum())
    if positive_weight <= 0 or negative_weight <= 0:
        raise ValueError('Weighted AUC requires both direction classes')
    order = np.argsort(scores, kind='stable')
    numerator = negative_before = 0.0
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and scores[order[end]] == scores[order[start]]:
            end += 1
        group = order[start:end]
        current_positive = float(weights[group][labels[group] == 1].sum())
        current_negative = float(weights[group][labels[group] == 0].sum())
        numerator += current_positive * (negative_before + .5 * current_negative)
        negative_before += current_negative
        start = end
    return float(numerator / (positive_weight * negative_weight))


def weighted_correlation(left, right, weights):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if (left.ndim != 1 or right.shape != left.shape or weights.shape != left.shape or
            not np.isfinite(left).all() or not np.isfinite(right).all() or
            not np.isfinite(weights).all() or np.any(weights <= 0)):
        raise ValueError('Invalid weighted correlation input')
    normalized = weights / weights.sum()
    left_centered = left - np.sum(left * normalized)
    right_centered = right - np.sum(right * normalized)
    denominator = np.sqrt(
        np.sum(np.square(left_centered) * normalized) *
        np.sum(np.square(right_centered) * normalized))
    if denominator <= 1e-12:
        return 0.0
    return float(np.sum(left_centered * right_centered * normalized) / denominator)


def stratified_week_bootstrap(values, scores, weights, weeks, folds, metric,
                              protocol, seed):
    values = np.asarray(values)
    scores = np.asarray(scores, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    weeks = np.asarray(weeks)
    folds = np.asarray(folds, dtype=np.int64)
    if not (len(values) == len(scores) == len(weights) == len(weeks) == len(folds)):
        raise ValueError('Diagnostic bootstrap inputs are not aligned')
    strata = []
    for fold in np.unique(folds):
        fold_rows = np.flatnonzero(folds == fold)
        fold_weeks = np.unique(weeks[fold_rows])
        if len(fold_weeks) < 2:
            raise ValueError('Each diagnostic fold needs multiple audit weeks')
        strata.append([
            np.flatnonzero((folds == fold) & (weeks == week))
            for week in fold_weeks
        ])
    rng = np.random.default_rng(seed)
    estimates = []
    draws = int(protocol['uncertainty']['draws'])
    for _ in range(draws):
        selected = []
        for clusters in strata:
            chosen = rng.integers(0, len(clusters), size=len(clusters))
            selected.extend(clusters[index] for index in chosen)
        indices = np.concatenate(selected)
        try:
            estimates.append(metric(
                values[indices], scores[indices], weights[indices]))
        except ValueError:
            continue
    if len(estimates) < max(100, draws // 2):
        raise ValueError('Too few valid diagnostic bootstrap draws')
    alpha = (1 - float(protocol['uncertainty']['confidence_level'])) / 2
    return (float(np.quantile(estimates, alpha)),
            float(np.quantile(estimates, 1 - alpha)))


def make_rows(source, cohort, arrays, routes, categories, positions, target_mode):
    if target_mode not in ('residual', 'zero', 'evaluation'):
        raise ValueError('Unknown v9b target mode')
    if (not np.array_equal(
            arrays['positive_sample_index'], routes['positive_sample_index']) or
            not np.array_equal(arrays['candidate_mask'], routes['candidate_mask'])):
        raise ValueError('v9b residual and route identities differ')
    features, targets, weights, metadata = [], [], [], []
    routed_events = 0
    for position in np.asarray(positions, dtype=np.int64):
        sample = source.sample(cohort, int(position))
        if (sample['sample'] != int(arrays['positive_sample_index'][position]) or
                not np.array_equal(
                    sample['candidate'], arrays['candidate_mask'][position])):
            raise ValueError('v9b feature and residual identities differ')
        nodes = np.flatnonzero(routes['hierarchical_route'][position])
        if not len(nodes):
            continue
        routed_events += 1
        if target_mode == 'zero':
            rows = target_for(None, None, nodes, 'event_node_phase', False)
        elif target_mode == 'evaluation':
            rows = evaluation_target_for(
                arrays['signed_residual'][position], arrays['valid'][position],
                nodes, 'event_node_phase')
        else:
            rows = target_for(
                arrays['signed_residual'][position], arrays['valid'][position],
                nodes, 'event_node_phase', True)
        if not rows:
            continue
        event_weight = 1. / len(rows)
        for node, phase, target in rows:
            features.append(residual_feature_vector(
                sample, arrays['baseline_prediction'][position], categories,
                node=node, phase=phase))
            targets.append(target)
            weights.append(event_weight)
            metadata.append((int(position), node, phase, sample))
    if not features:
        raise ValueError(f'No v9b rows for {cohort} {target_mode}')
    return (np.asarray(features, dtype=np.float64),
            np.asarray(targets, dtype=np.float64),
            np.asarray(weights, dtype=np.float64), metadata, routed_events)


def fit_experts(source, full_arrays, control_arrays, routes, categories,
                full_positions, common_positions, protocol):
    cohort_rows, summary = {}, {}
    for cohort in TRAIN_COHORTS:
        arrays = full_arrays if cohort == 'incident_full' else control_arrays[cohort]
        positions = full_positions if cohort == 'incident_full' else common_positions
        target_mode = 'residual' if cohort == 'incident_full' else 'zero'
        rows = make_rows(
            source, cohort, arrays, routes[cohort], categories, positions,
            target_mode)
        features, targets, weights, _, routed_events = rows
        weights = weights / weights.sum()
        cohort_rows[cohort] = (features, targets, weights)
        summary[cohort] = {
            'eligible_events': int(len(positions)),
            'routed_events': int(routed_events), 'rows': int(len(features)),
            'normalized_weight_sum': float(weights.sum()),
        }
    incident_features, incident_targets, _ = cohort_rows['incident_full']
    features = np.concatenate([cohort_rows[key][0] for key in TRAIN_COHORTS])
    weights = np.concatenate([cohort_rows[key][2] for key in TRAIN_COHORTS])
    signed_targets = np.concatenate([cohort_rows[key][1] for key in TRAIN_COHORTS])
    direction_targets = np.concatenate([
        np.sign(cohort_rows[key][1]) if key == 'incident_full'
        else cohort_rows[key][1] for key in TRAIN_COHORTS])
    magnitude_targets = np.concatenate([
        np.abs(cohort_rows[key][1]) if key == 'incident_full'
        else cohort_rows[key][1] for key in TRAIN_COHORTS])
    alpha = float(protocol['expert']['ridge_alpha'])
    models = {
        'signed': fit_weighted_ridge(features, signed_targets, weights, alpha),
        'direction': fit_weighted_ridge(
            features, direction_targets, weights, alpha),
        'magnitude': fit_weighted_ridge(
            features, magnitude_targets, weights, alpha),
    }
    clip = float(np.quantile(np.abs(incident_targets), .99))
    direction_fit = np.clip(
        models['direction'].predict(incident_features), -1., 1.)
    abstention = float(np.quantile(np.abs(direction_fit), .75))
    if (not np.isfinite(clip) or clip <= 0 or not np.isfinite(abstention) or
            abstention < 0):
        raise ValueError('Invalid v9b fit-only clip or abstention threshold')
    summary['clip_abs'] = clip
    summary['direction_confidence_q75'] = abstention
    return models, clip, abstention, summary


def policy_predictions(models, features, targets, clip, abstention):
    direct = np.clip(models['signed'].predict(features), -clip, clip)
    direction = np.clip(models['direction'].predict(features), -1., 1.)
    magnitude = np.clip(models['magnitude'].predict(features), 0., clip)
    two_part = np.sign(direction) * magnitude
    confident = np.abs(direction) >= abstention
    safe_target = np.where(np.isfinite(targets), targets, 0.)
    policies = {
        'direct_signed': direct,
        'two_part_all': two_part,
        'two_part_abstained': np.where(confident, two_part, 0.),
        'oracle_direction': np.sign(safe_target) * magnitude,
        'oracle_magnitude': np.sign(direction) * np.abs(safe_target),
    }
    return policies, direction, magnitude, confident


def correction_for_subset(arrays, positions, predictions, metadata):
    positions = np.asarray(positions, dtype=np.int64)
    correction = np.zeros(
        (len(positions), *arrays['signed_residual'].shape[1:]),
        dtype=np.float64)
    local = {int(position): index for index, position in enumerate(positions)}
    if len(predictions) != len(metadata):
        raise ValueError('v9b predictions and metadata differ')
    for value, (position, node, phase, _) in zip(predictions, metadata):
        if position not in local or node is None or phase is None:
            raise ValueError('v9b correction metadata escaped its audit subset')
        first, last = PHASES[phase]
        correction[local[position], first:last, node, 0] = value
    if not np.isfinite(correction).all():
        raise ValueError('v9b correction contains non-finite values')
    return correction


def model_arrays(prefix, model):
    return {
        f'{prefix}_mean': model.mean,
        f'{prefix}_scale': model.scale,
        f'{prefix}_target_mean': np.asarray(model.target_mean),
        f'{prefix}_coefficient': model.coefficient,
        f'{prefix}_ridge_alpha': np.asarray(model.alpha),
    }


def diagnostics_summary(targets, direct, direction, magnitude, weights, weeks,
                        folds, abstained_active, protocol):
    targets = np.asarray(targets, dtype=np.float64)
    finite = np.isfinite(targets)
    nonzero = finite & (targets != 0)
    if not nonzero.any():
        raise ValueError('v9b has no nonzero incident targets')
    labels = (targets[nonzero] > 0).astype(np.int64)
    direction_auc = weighted_auc(
        labels, direction[nonzero], weights[nonzero])
    direction_ci = stratified_week_bootstrap(
        labels, direction[nonzero], weights[nonzero], weeks[nonzero],
        folds[nonzero], weighted_auc, protocol,
        int(protocol['uncertainty']['seed']))
    magnitude_correlation = weighted_correlation(
        np.abs(targets[finite]), magnitude[finite], weights[finite])
    magnitude_ci = stratified_week_bootstrap(
        np.abs(targets[finite]), magnitude[finite], weights[finite],
        weeks[finite], folds[finite], weighted_correlation, protocol,
        int(protocol['uncertainty']['seed']) + 1)
    signed_correlation = weighted_correlation(
        targets[finite], direct[finite], weights[finite])
    direction_accuracy = float(np.average(
        np.sign(direction[nonzero]) == np.sign(targets[nonzero]),
        weights=weights[nonzero]))
    coverage = float(np.average(
        abstained_active[finite], weights=weights[finite]))
    fold_results = []
    for fold in np.unique(folds):
        selected = finite & (folds == fold)
        selected_nonzero = nonzero & (folds == fold)
        fold_results.append({
            'fold': int(fold),
            'rows': int(selected.sum()),
            'direction_auc': weighted_auc(
                (targets[selected_nonzero] > 0).astype(np.int64),
                direction[selected_nonzero], weights[selected_nonzero]),
            'magnitude_correlation': weighted_correlation(
                np.abs(targets[selected]), magnitude[selected], weights[selected]),
            'signed_correlation': weighted_correlation(
                targets[selected], direct[selected], weights[selected]),
            'abstained_row_coverage': float(np.average(
                abstained_active[selected], weights=weights[selected])),
        })
    return {
        'rows': int(finite.sum()),
        'direction_auc': direction_auc,
        'direction_auc_ci_low': direction_ci[0],
        'direction_auc_ci_high': direction_ci[1],
        'direction_accuracy': direction_accuracy,
        'magnitude_correlation': magnitude_correlation,
        'magnitude_correlation_ci_low': magnitude_ci[0],
        'magnitude_correlation_ci_high': magnitude_ci[1],
        'signed_correlation': signed_correlation,
        'abstained_row_coverage': coverage,
        'folds': fold_results,
    }


def gate_decision(results, diagnostics, protocol):
    gate = protocol['diagnostic_gate']
    primary = results[gate['primary_policy']]
    full, common = primary['incident_full'], primary['incident']
    margin = (gate['maximum_full_positive_global_harm_fraction_of_A_mae'] *
              full['all']['mae_A'])
    checks = {
        'direction_auc_ci_lower':
            diagnostics['direction_auc_ci_low'] >
            gate['require_direction_auc_ci_lower_above'],
        'magnitude_correlation_ci_lower':
            diagnostics['magnitude_correlation_ci_low'] >
            gate['require_magnitude_correlation_ci_lower_above'],
        'each_fold_direction_auc': all(
            fold['direction_auc'] > gate['require_each_fold_direction_auc_above']
            for fold in diagnostics['folds']),
        'each_fold_magnitude_correlation': all(
            fold['magnitude_correlation'] >
            gate['require_each_fold_magnitude_correlation_above']
            for fold in diagnostics['folds']),
        'full_positive_global_point_improvement':
            full['all']['improvement_vs_A'] > 0,
        'full_positive_global_noninferiority':
            full['uncertainty']['all_improvement_vs_A']['ci_low'] >= -margin,
        'full_positive_routed_improvement':
            full['uncertainty']['routed_improvement_vs_A']['ci_low'] > 0,
        'common_incident_routed_point_improvement':
            common['routed_candidate_h1_h6']['improvement_vs_A'] > 0,
        'minimum_abstained_row_coverage':
            diagnostics['abstained_row_coverage'] >=
            gate['minimum_abstained_row_coverage'],
        'protected_outside_hierarchical_route_exact_A':
            full['protected_outside_hierarchical_route_exact_A'],
        'protected_h7_h12_exact_A': full['protected_h7_h12_exact_A'],
        'protected_noncandidate_exact_A': full['protected_noncandidate_exact_A'],
    }
    fraction = gate['maximum_each_control_routed_harm_fraction_of_A_mae']
    for cohort in ('primary_control', 'secondary_control'):
        current = primary[cohort]
        checks[f'{cohort}_routed_harm_bound'] = (
            current['uncertainty']['routed_harm_vs_A']['ci_high'] <=
            fraction * current['routed_candidate_h1_h6']['mae_A'])
    passed = all(checks.values())
    return {
        'checks': checks, 'diagnostic_gate_passed': passed,
        'v9a_gate_overridden': False,
        'maximum_global_harm_raw_mae': margin,
        'recommendation': (
            'NEW_NONLINEAR_EXPERT_PREREGISTRATION_SUPPORTED' if passed else
            'ROUTED_RESIDUAL_DIRECTION_OR_MAGNITUDE_NOT_ESTABLISHED'),
    }


def audit(data_dir, primary_dir, secondary_dir, residual_dir, placebo_dir,
          sensors_path, protocol_path, output):
    output = Path(output)
    partial = output.with_name(output.name + '.partial')
    if output.exists() or partial.exists():
        raise FileExistsError('v9b output or partial output exists; use a new directory')
    protocol = load_protocol(protocol_path)
    v7a_summary, package_hashes, residual_input_hashes = verify_v7a_train_inputs(
        data_dir, primary_dir, secondary_dir, residual_dir, protocol)
    v5c_path = Path(__file__).with_name('matched_placebo_audit_v5c.json')
    v8a_path = Path(__file__).with_name('incident_impact_identification_v8a.json')
    v8b_path = Path(__file__).with_name(
        'incident_impact_node_localization_v8b.json')
    v5c = load_v5c_protocol(v5c_path)
    v5c_context = protocol['v5c_context']
    if (v5c['protocol_id'] != v5c_context['protocol_id'] or
            sha256(v8a_path) != protocol['router_design']['event_protocol_sha256'] or
            sha256(v8b_path) != protocol['router_design']['node_protocol_sha256'] or
            sha256(Path(placebo_dir) / 'summary.json') !=
            v5c_context['summary_sha256'] or
            sha256(Path(placebo_dir) / 'triple_metrics.csv') !=
            v5c_context['triple_metrics_sha256'] or
            sha256(sensors_path) != v5c_context['sensor_metadata_sha256']):
        raise ValueError('v9b frozen router or placebo context differs')
    matched_input_hashes = verify_v5c_inputs(
        data_dir, primary_dir, secondary_dir, v5c)
    print(json.dumps({'stage': 'inputs_verified'}), flush=True)

    nodes = protocol['expected_sensor_count']
    common_count = protocol['expected_common_train_samples']
    full_count = protocol['expected_full_positive_train_samples']
    full_arrays = load_residuals(
        residual_dir, 'train', 'incident_full', full_count, nodes)
    control_inputs = {
        cohort: load_control_training_inputs(
            residual_dir, 'train', cohort, common_count, nodes)
        for cohort in ('primary_control', 'secondary_control')
    }
    common_identity = load_control_training_inputs(
        residual_dir, 'train', 'primary_control', common_count, nodes)
    full_weeks, common_weeks = manifest_weeks(
        data_dir, secondary_dir, full_arrays['positive_sample_index'],
        common_identity['positive_sample_index'])
    folds = rolling_origin_folds(full_weeks, common_weeks, protocol)

    samples, label_weeks, event_excess = read_event_router_labels(
        placebo_dir, common_count,
        protocol['rolling_origin']['expected_unique_train_iso_weeks'])
    if not np.array_equal(samples, common_identity['positive_sample_index']):
        raise ValueError('v9b router labels and residual identities differ')
    event_features, categories, event_feature_weeks = build_event_router_features(
        data_dir, primary_dir, secondary_dir, sensors_path, samples, nodes)
    scaler = json.loads((Path(data_dir) / 'scaler.json').read_text(encoding='utf-8'))
    (source, node_categories, node_features, _, node_excess, node_event_indices,
     node_feature_weeks) = build_node_router_data(
         data_dir, primary_dir, secondary_dir, sensors_path, samples, nodes,
         float(scaler['std']))
    if (categories != node_categories or
            not np.array_equal(label_weeks, common_weeks) or
            not np.array_equal(event_feature_weeks, common_weeks) or
            not np.array_equal(node_feature_weeks, common_weeks)):
        raise ValueError('v9b router feature, label, and residual weeks differ')
    event_names = feature_names(categories, 'event')
    node_names = feature_names(categories, 'event_node')
    print(json.dumps({
        'stage': 'router_features_ready',
        'common_events': common_count,
        'node_rows': int(len(node_excess)),
    }), flush=True)

    fold_artifacts, stored_models, fold_summaries = [], {}, []
    for fold in folds:
        fit_common = fold['common_fit_indices']
        fit_full = np.flatnonzero(np.isin(full_weeks, fold['fit_weeks']))
        event_label_threshold = float(np.quantile(event_excess[fit_common], .75))
        event_labels = (event_excess >= event_label_threshold).astype(np.int64)
        event_model, event_route_threshold = fit_event_router(
            event_features, event_labels, fit_common, 10.0)
        fit_node_rows = np.flatnonzero(np.isin(node_event_indices, fit_common))
        node_label_threshold = float(np.quantile(node_excess[fit_node_rows], .75))
        node_labels = (node_excess >= node_label_threshold).astype(np.int64)
        node_model, node_route_threshold = fit_node_router(
            node_features, node_labels, node_event_indices, fit_common, 10.0)
        if (len(event_model.coefficient) != len(event_names) or
                len(node_model.coefficient) != len(node_names)):
            raise ValueError('v9b router feature names and coefficients differ')
        routes = {
            cohort: materialize_cohort(
                source, cohort, full_count if cohort == 'incident_full' else common_count,
                nodes, categories, event_model, event_route_threshold,
                node_model, node_route_threshold)
            for cohort in COHORTS
        }
        models, clip, abstention, expert_summary = fit_experts(
            source, full_arrays, control_inputs, routes, categories,
            fit_full, fit_common, protocol)
        expert_names = residual_feature_names(categories, 'event_node_phase')
        if any(len(model.coefficient) != len(expert_names)
               for model in models.values()):
            raise ValueError('v9b expert feature names and coefficients differ')
        fold_artifacts.append({
            'fold': fold, 'routes': routes, 'models': models,
            'clip': clip, 'abstention': abstention,
        })
        prefix = f'fold_{fold["fold"]}'
        stored_models.update(model_arrays(f'{prefix}_event_router', event_model))
        stored_models.update(model_arrays(f'{prefix}_node_router', node_model))
        for name, model in models.items():
            stored_models.update(model_arrays(f'{prefix}_{name}', model))
        stored_models[f'{prefix}_signed_clip_abs'] = np.asarray(clip)
        stored_models[f'{prefix}_direction_confidence_q75'] = np.asarray(abstention)
        stored_models[f'{prefix}_event_label_threshold'] = np.asarray(
            event_label_threshold)
        stored_models[f'{prefix}_event_route_threshold'] = np.asarray(
            event_route_threshold)
        stored_models[f'{prefix}_node_label_threshold'] = np.asarray(
            node_label_threshold)
        stored_models[f'{prefix}_node_route_threshold'] = np.asarray(
            node_route_threshold)
        fold_summaries.append({
            'fold': fold['fold'], 'fit_weeks': list(fold['fit_weeks']),
            'audit_weeks': list(fold['audit_weeks']),
            'fit_common_events': int(len(fit_common)),
            'fit_full_positive_events': int(len(fit_full)),
            'audit_common_events': int(len(fold['common_audit_indices'])),
            'audit_full_positive_events': int(len(fold['full_audit_indices'])),
            'event_label_threshold': event_label_threshold,
            'event_route_threshold': event_route_threshold,
            'node_label_threshold': node_label_threshold,
            'node_route_threshold': node_route_threshold,
            'audit_routes': {
                cohort: {
                    'event_route_fraction': float(
                        routes[cohort]['event_route'][
                            fold['full_audit_indices'] if cohort == 'incident_full'
                            else fold['common_audit_indices']].mean()),
                    'hierarchical_routed_nodes': int(
                        routes[cohort]['hierarchical_route'][
                            fold['full_audit_indices'] if cohort == 'incident_full'
                            else fold['common_audit_indices']].sum()),
                }
                for cohort in COHORTS
            },
            'expert': expert_summary,
        })
        print(json.dumps({
            'stage': 'fold_fit_complete', 'fold': fold['fold'],
            'fit_common_events': int(len(fit_common)),
            'fit_full_positive_events': int(len(fit_full)),
            'event_route_threshold': event_route_threshold,
            'node_route_threshold': node_route_threshold,
            'expert_clip_abs': clip,
            'direction_confidence_q75': abstention,
        }), flush=True)

    evaluation_arrays = {
        'incident_full': full_arrays,
        'incident': load_residuals(
            residual_dir, 'train', 'incident', common_count, nodes),
        'primary_control': load_residuals(
            residual_dir, 'train', 'primary_control', common_count, nodes),
        'secondary_control': load_residuals(
            residual_dir, 'train', 'secondary_control', common_count, nodes),
    }
    accumulated = {
        policy: {cohort: {'arrays': [], 'routes': [], 'corrections': [],
                          'weeks': [], 'folds': []}
                 for cohort in COHORTS}
        for policy in DEPLOYABLE_POLICIES
    }
    oracle_accumulated = {
        policy: {cohort: {'arrays': [], 'routes': [], 'corrections': [],
                          'weeks': [], 'folds': []}
                 for cohort in ('incident_full', 'incident')}
        for policy in ORACLE_POLICIES
    }
    diagnostic_rows = []
    decomposition_csv_rows = []
    for artifact in fold_artifacts:
        fold = artifact['fold']
        for cohort in COHORTS:
            arrays = evaluation_arrays[cohort]
            positions = (fold['full_audit_indices'] if cohort == 'incident_full'
                         else fold['common_audit_indices'])
            features, targets, weights, metadata, _ = make_rows(
                source, cohort, arrays, artifact['routes'][cohort], categories,
                positions, 'evaluation')
            predictions, direction, magnitude, confident = policy_predictions(
                artifact['models'], features, targets, artifact['clip'],
                artifact['abstention'])
            weeks = full_weeks[positions] if cohort == 'incident_full' \
                else common_weeks[positions]
            part = subset_arrays(arrays, positions)
            route_part = subset_arrays(artifact['routes'][cohort], positions)
            for policy in DEPLOYABLE_POLICIES:
                current = accumulated[policy][cohort]
                current['arrays'].append(part)
                current['routes'].append(route_part)
                current['corrections'].append(correction_for_subset(
                    arrays, positions, predictions[policy], metadata))
                current['weeks'].append(weeks)
                current['folds'].append(np.full(
                    len(positions), fold['fold'], dtype=np.int64))
            if cohort in ('incident_full', 'incident'):
                for policy in ORACLE_POLICIES:
                    current = oracle_accumulated[policy][cohort]
                    current['arrays'].append(part)
                    current['routes'].append(route_part)
                    current['corrections'].append(correction_for_subset(
                        arrays, positions, predictions[policy], metadata))
                    current['weeks'].append(weeks)
                    current['folds'].append(np.full(
                        len(positions), fold['fold'], dtype=np.int64))
            if cohort == 'incident_full':
                for (row, target, weight, direct, score, size, keep, two_part,
                     abstained) in zip(
                        metadata, targets, weights, predictions['direct_signed'],
                        direction, magnitude, confident, predictions['two_part_all'],
                        predictions['two_part_abstained']):
                    _, node, phase, sample = row
                    diagnostic_rows.append({
                        'target': target, 'direct': direct, 'direction': score,
                        'magnitude': size, 'weight': weight,
                        'week': sample['iso_week'], 'fold': fold['fold'],
                        'abstained_active': abs(abstained) > 1e-6,
                    })
                    decomposition_csv_rows.append({
                        'fold': fold['fold'],
                        'positive_sample_index': sample['sample'],
                        'positive_iso_week': sample['iso_week'],
                        'node_index': node, 'phase': phase,
                        'target_correction': target,
                        'direct_signed_prediction': direct,
                        'direction_score': score,
                        'magnitude_prediction': size,
                        'direction_confident': bool(keep),
                        'two_part_prediction': two_part,
                    })
        print(json.dumps({
            'stage': 'fold_evaluation_complete', 'fold': fold['fold'],
            'audit_full_positive_events': int(len(fold['full_audit_indices'])),
            'audit_common_events': int(len(fold['common_audit_indices'])),
        }), flush=True)

    results, event_rows = {}, []
    for policy_number, policy in enumerate(DEPLOYABLE_POLICIES):
        results[policy] = {}
        for cohort_number, cohort in enumerate(COHORTS):
            current = accumulated[policy][cohort]
            arrays = concatenate_arrays(current['arrays'])
            routes = concatenate_arrays(current['routes'])
            correction = np.concatenate(current['corrections'])
            weeks = np.concatenate(current['weeks'])
            fold_numbers = np.concatenate(current['folds'])
            metrics = protected_metrics(arrays, routes, correction)
            metrics['uncertainty'] = uncertainty(
                metrics, weeks, protocol,
                int(protocol['uncertainty']['seed']) +
                policy_number * 100 + cohort_number * 10)
            per_event = metrics.pop('_per_event')
            for position, sample in enumerate(arrays['positive_sample_index']):
                event_rows.append({
                    'policy': policy, 'cohort': cohort,
                    'fold': int(fold_numbers[position]),
                    'positive_sample_index': int(sample),
                    'positive_iso_week': weeks[position],
                    'all_improvement_vs_A':
                        float(per_event['all_improvement_vs_A'][position]),
                    'routed_improvement_vs_A': (
                        float(per_event['routed_improvement_vs_A'][position])
                        if np.isfinite(per_event['routed_improvement_vs_A'][position])
                        else ''),
                })
            results[policy][cohort] = metrics
    oracle_results = {}
    for policy in ORACLE_POLICIES:
        oracle_results[policy] = {}
        for cohort in ('incident_full', 'incident'):
            current = oracle_accumulated[policy][cohort]
            metrics = protected_metrics(
                concatenate_arrays(current['arrays']),
                concatenate_arrays(current['routes']),
                np.concatenate(current['corrections']))
            metrics.pop('_per_event')
            oracle_results[policy][cohort] = metrics

    diagnostic = diagnostics_summary(
        np.asarray([row['target'] for row in diagnostic_rows]),
        np.asarray([row['direct'] for row in diagnostic_rows]),
        np.asarray([row['direction'] for row in diagnostic_rows]),
        np.asarray([row['magnitude'] for row in diagnostic_rows]),
        np.asarray([row['weight'] for row in diagnostic_rows]),
        np.asarray([row['week'] for row in diagnostic_rows]),
        np.asarray([row['fold'] for row in diagnostic_rows]),
        np.asarray([row['abstained_active'] for row in diagnostic_rows]),
        protocol)
    gate = gate_decision(results, diagnostic, protocol)

    partial.mkdir(parents=True, exist_ok=False)
    model_path = partial / 'fold_models.npz'
    with model_path.open('wb') as stream:
        np.savez_compressed(
            stream,
            feature_names=np.asarray(expert_names),
            event_router_feature_names=np.asarray(event_names),
            node_router_feature_names=np.asarray(node_names),
            **stored_models)
    decomposition_path = partial / 'incident_full_decomposition_rows.csv'
    write_csv(decomposition_path, decomposition_csv_rows, [
        'fold', 'positive_sample_index', 'positive_iso_week', 'node_index',
        'phase', 'target_correction', 'direct_signed_prediction',
        'direction_score', 'magnitude_prediction', 'direction_confident',
        'two_part_prediction'])
    event_path = partial / 'deployable_policy_event_metrics.csv'
    write_csv(event_path, event_rows, [
        'policy', 'cohort', 'fold', 'positive_sample_index',
        'positive_iso_week', 'all_improvement_vs_A',
        'routed_improvement_vs_A'])
    outputs = [model_path, decomposition_path, event_path]
    summary = {
        'status': 'ROUTED_RESIDUAL_DECOMPOSITION_AUDIT_COMPLETE',
        'scope': protocol['scope'], 'protocol_id': protocol['protocol_id'],
        'protocol_sha256': sha256(protocol_path), 'main_training_ready': False,
        'train_residual_arrays_only': True,
        'validation_residual_arrays_read': False, 'test_split_read': False,
        'full_train_v8c_routes_used': False,
        'router_and_expert_cross_fitted': True,
        'control_future_Y_used_for_expert_fit': False,
        'oracle_reported_as_model_performance': False,
        'v9a_gate_overridden': False, 'neural_expert_trained': False,
        'folds': fold_summaries, 'diagnostics': diagnostic,
        'deployable_results': results, 'oracle_diagnostics': oracle_results,
        'diagnostic_gate': gate, 'support': protocol['support'],
        'inputs': {
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
            path.name: {'sha256': sha256(path), 'bytes': path.stat().st_size}
            for path in outputs},
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
        'status': summary['status'], 'diagnostics': diagnostic,
        'diagnostic_gate': gate,
        'primary_result': results['two_part_abstained']['incident_full'],
    }, ensure_ascii=False, indent=2), flush=True)
    print(f'Saved v9b routed residual decomposition: {output / "summary.json"}',
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
    parser.add_argument('--protocol', type=Path, default=Path(__file__).with_name(
        'routed_residual_decomposition_v9b.json'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    audit(
        args.data_dir, args.primary_control_dir, args.secondary_control_dir,
        args.residual_dir, args.placebo_dir, args.sensors, args.protocol,
        args.output)


if __name__ == '__main__':
    main()
