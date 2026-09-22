"""Audit prospective localization of high-impact incident nodes from report-time features."""

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.chronological.audit_expert_benefit import (
    FeatureSource, category_schema, feature_vector, fit_weighted_ridge,
)
from experiments.chronological.audit_incident_impact_identification import (
    bootstrap_metric, roc_auc, rolling_folds, top_quartile_lift,
)
from experiments.chronological.audit_matched_controls import (
    read_csv, sha256, write_csv,
)
from experiments.chronological.audit_matched_placebo import (
    load_protocol as load_v5c_protocol, verify_inputs as verify_v5c_inputs,
)


COHORTS = ('incident', 'primary_control', 'secondary_control')
SLICES = {'baseline': slice(9, 12), 'early': slice(14, 20)}


def load_protocol(path):
    protocol = json.loads(Path(path).read_text(encoding='utf-8'))
    if (protocol.get('protocol_id') !=
            'contra_v8_observable_incident_impact_node_localization_v8b' or
            protocol.get('scope') !=
            'train_only_rolling_origin_observable_incident_impact_node_localization' or
            protocol.get('main_training_ready') is not False):
        raise ValueError('v8b protocol identity or scope changed')
    if [protocol.get(key) for key in ('source_year', 'source_version',
                                      'expected_sensor_count')] != [2023, 8, 496]:
        raise ValueError('v8b is frozen to the 2023 source-v8 Contra496 package')
    if protocol.get('expected_common_train_samples') != 3106 or \
            protocol.get('expected_full_positive_train_samples') != 3604:
        raise ValueError('v8b training cohorts changed')
    if protocol.get('v8a_context') != {
            'protocol_id': 'contra_v8_observable_incident_impact_identification_v8a',
            'protocol_sha256': '1462d7b2a4d51f947ede97f772d2ae002f008698f1f3d4325f7cadeea16576e6',
            'observed_recommendation': 'OBSERVABLE_IMPACT_GATE_DEVELOPMENT_ALLOWED',
            'role': 'development_authorization_only_not_an_input'}:
        raise ValueError('v8b v8a context changed')
    if protocol.get('v5c_context') != {
            'protocol_id': 'contra_v8_matched_placebo_audit_v5c',
            'summary_sha256': 'c443581334f1f0aebece82c910a38c78b1dc29a22565d081f6a583298875571a',
            'triple_metrics_sha256': '192b44e82223a9f943f3b47a5141ea1d19819b7cd0d0c75b0c234d372dcb5bc5',
            'role': 'freeze_common_sample_identity_and_time_order'}:
        raise ValueError('v8b v5c context changed')
    if protocol.get('sensor_metadata') != {
            'sha256': '682f3cdf75e643f0b37356ab69cbabb27389be5089f41d3b2cbc4bede3332094',
            'role': 'derive_report_freeway_direction_from_nonzero_distance_support'}:
        raise ValueError('v8b sensor metadata changed')
    rolling = protocol.get('rolling_origin', {})
    if rolling != {
            'time_source': 'positive_incident_t0_iso_week',
            'expected_unique_train_iso_weeks': 35,
            'initial_fit_week_count': 18,
            'audit_fold_count': 3,
            'audit_week_block_sizes': [6, 6, 5],
            'fit_rule': 'all_common_train_weeks_strictly_before_each_audit_block',
            'audit_rule': 'each_of_the_final_17_common_train_weeks_exactly_once',
            'expected_common_audit_counts': [536, 618, 338],
            'expected_common_fit_minimums': [1, 1, 1]}:
        raise ValueError('v8b rolling-origin design changed')
    target = protocol.get('target', {})
    if target != {
            'family': 'node_early_excess_divergence',
            'formula': 'mean_early(incident_control_divergence_baseline_adjusted - routine_control_divergence_baseline_adjusted)',
            'early_Y_slice': [14, 20], 'baseline_slice': [9, 12],
            'scale': 'frozen_train_scaler_std',
            'incident_label': 'node_excess_at_or_above_fit_incident_candidate_node_q75',
            'control_label': 'exact_zero',
            'candidate_support': 'positive_report_location_distance_nonzero',
            'future_outcome_target_used_only_in_fit_and_audit_labels': True}:
        raise ValueError('v8b target design changed')
    features = protocol.get('features', {})
    if (features.get('family') != 'report_time_safe_event_node_features' or
            features.get('future_outcome_features_forbidden') is not True or
            features.get('frozen_A_prediction_features_forbidden') is not True or
            features.get('post_event_fields_forbidden') is not True):
        raise ValueError('v8b feature boundary changed')
    estimator = protocol.get('estimator', {})
    if estimator != {
            'family': 'weighted_ridge_binary_node_score', 'ridge_alpha': 10.0,
            'training_cohorts': list(COHORTS),
            'row_weighting': 'equal_cohort_and_equal_event_weight_within_candidate_nodes',
            'score_clip': [0.0, 1.0],
            'route_threshold': 'fit_incident_node_score_q75',
            'route_unit': 'candidate_node_expert_activation'}:
        raise ValueError('v8b estimator design changed')
    if protocol.get('uncertainty') != {
            'method': 'audit_iso_week_cluster_bootstrap', 'draws': 2000,
            'confidence_level': 0.95, 'seed': 2025}:
        raise ValueError('v8b uncertainty design changed')
    if protocol.get('development_gate') != {
            'require_incident_node_auc_ci_lower_above': 0.55,
            'require_incident_node_top_quartile_lift_ci_lower_above': 1.2,
            'require_incident_node_route_fraction_at_least': 0.05,
            'maximum_primary_control_node_route_fraction': 0.35,
            'maximum_secondary_control_node_route_fraction': 0.35,
            'require_all_audit_weeks_covered_once': True}:
        raise ValueError('v8b development gate changed')
    required = (
        'train_common_triples_only', 'validation_residual_arrays_prohibited',
        'test_split_prohibited', 'future_outcome_labels_prohibited_from_features',
        'audit_weeks_prohibited_from_fit', 'future_weeks_prohibited_from_each_fold_fit',
        'thresholds_and_gate_may_not_change_after_result',
        'expert_training_prohibited', 'neural_node_localizer_prohibited')
    if not all(protocol.get('information_boundary', {}).get(key) is True
               for key in required):
        raise ValueError('v8b information boundary changed')
    return protocol


def node_excess(positive, primary, secondary, candidate, scaler_std):
    positive = np.asarray(positive, dtype=np.float64)
    primary = np.asarray(primary, dtype=np.float64)
    secondary = np.asarray(secondary, dtype=np.float64)
    if positive.shape != primary.shape or positive.shape != secondary.shape:
        raise ValueError('Matched flow shapes differ')
    if positive.ndim != 2 or candidate.shape != (positive.shape[1],):
        raise ValueError('Invalid matched flow or candidate shape')
    finite = (np.isfinite(positive).all(axis=0) &
              np.isfinite(primary).all(axis=0) &
              np.isfinite(secondary).all(axis=0))
    support = np.asarray(candidate, dtype=bool) & finite
    if not support.any() or scaler_std <= 0:
        raise ValueError('Node impact support is empty or scaler is invalid')
    incident_divergence = .5 * (
        np.abs(positive - primary) + np.abs(positive - secondary))
    routine_divergence = np.abs(primary - secondary)
    incident_change = incident_divergence - incident_divergence[SLICES['baseline']].mean(axis=0)
    routine_change = routine_divergence - routine_divergence[SLICES['baseline']].mean(axis=0)
    excess = (incident_change[SLICES['early']].mean(axis=0) -
              routine_change[SLICES['early']].mean(axis=0)) / float(scaler_std)
    return np.flatnonzero(support), excess[support]


def read_identity(placebo_dir, expected_samples, expected_weeks):
    rows = [row for row in read_csv(Path(placebo_dir) / 'triple_metrics.csv')
            if row['split'] == 'train']
    if len(rows) != expected_samples:
        raise ValueError('v5c train triple count changed')
    samples = np.asarray([int(row['positive_sample_index']) for row in rows], dtype=np.int64)
    weeks = np.asarray([row['positive_iso_week'] for row in rows])
    if len(set(samples.tolist())) != expected_samples or len(set(weeks.tolist())) != expected_weeks:
        raise ValueError('v5c sample identity or week support changed')
    return samples, weeks


def build_data(data_dir, primary_dir, secondary_dir, sensors_path,
               samples, expected_nodes, scaler_std):
    source = FeatureSource(
        data_dir, primary_dir, secondary_dir, sensors_path, 'train',
        len(samples), 3604, expected_nodes)
    categories = category_schema(source)
    features, nodes, excess_values, event_indices, weeks = {
        cohort: [] for cohort in COHORTS}, [], [], [], []
    for position, sample_id in enumerate(samples):
        positive_sample = source.sample('incident', position)
        if positive_sample['sample'] != int(sample_id):
            raise ValueError('Positive feature identity differs from v5c')
        candidate = positive_sample['candidate']
        node_ids, current_excess = node_excess(
            source.positive_flow[source.positive_positions[int(sample_id)]],
            source.primary_flow[int(source.primary_by_sample[int(sample_id)]['control_index'])],
            source.secondary_flow[position], candidate, scaler_std)
        if not len(node_ids):
            raise ValueError('Common incident has no valid candidate nodes')
        weeks.append(positive_sample['iso_week'])
        nodes.extend(node_ids.tolist())
        excess_values.extend(current_excess.tolist())
        event_indices.extend([position] * len(node_ids))
        for cohort in COHORTS:
            sample = source.sample(cohort, position)
            if sample['sample'] != int(sample_id):
                raise ValueError(f'{cohort} feature identity differs from v5c')
            features[cohort].extend(
                feature_vector(sample, categories, int(node)) for node in node_ids)
    features = {cohort: np.asarray(values, dtype=np.float64)
                for cohort, values in features.items()}
    return (source, categories, features, np.asarray(nodes, dtype=np.int64),
            np.asarray(excess_values, dtype=np.float64),
            np.asarray(event_indices, dtype=np.int64), np.asarray(weeks))


def metric_summary(labels, scores, clusters, protocol, seed):
    auc = roc_auc(labels, scores)
    lift = top_quartile_lift(labels, scores)
    auc_ci = bootstrap_metric(
        labels, scores, clusters, roc_auc,
        protocol['uncertainty']['draws'], protocol['uncertainty']['confidence_level'], seed)
    lift_ci = bootstrap_metric(
        labels, scores, clusters, top_quartile_lift,
        protocol['uncertainty']['draws'], protocol['uncertainty']['confidence_level'], seed + 1)
    return {
        'nodes': int(len(labels)), 'positive_nodes': int(labels.sum()),
        'positive_fraction': float(labels.mean()), 'roc_auc': auc,
        'roc_auc_ci_low': auc_ci[0], 'roc_auc_ci_high': auc_ci[1],
        'top_quartile_lift': lift,
        'top_quartile_lift_ci_low': lift_ci[0],
        'top_quartile_lift_ci_high': lift_ci[1],
    }


def fit_node_model(features, labels, event_indices, fit_events, alpha):
    training_features, training_targets, training_weights = [], [], []
    for cohort in COHORTS:
        rows = np.flatnonzero(np.isin(event_indices, fit_events))
        current = features[cohort][rows]
        target = labels[rows] if cohort == 'incident' else np.zeros(len(rows))
        counts = np.bincount(event_indices[rows], minlength=int(event_indices.max()) + 1)
        weights = 1. / counts[event_indices[rows]]
        training_features.append(current)
        training_targets.append(target)
        training_weights.append(weights / len(fit_events))
    model = fit_weighted_ridge(
        np.concatenate(training_features), np.concatenate(training_targets),
        np.concatenate(training_weights), float(alpha))
    incident_rows = np.flatnonzero(np.isin(event_indices, fit_events))
    fit_scores = np.clip(model.predict(features['incident'][incident_rows]), 0., 1.)
    return model, float(np.quantile(fit_scores, .75))


def decision(results, protocol, weeks_covered):
    gate = protocol['development_gate']
    incident = results['incident']
    checks = {
        'incident_node_auc_ci_lower':
            incident['roc_auc_ci_low'] > gate['require_incident_node_auc_ci_lower_above'],
        'incident_node_top_quartile_lift_ci_lower':
            incident['top_quartile_lift_ci_low'] >
            gate['require_incident_node_top_quartile_lift_ci_lower_above'],
        'incident_node_route_fraction':
            results['incident']['route_fraction'] >=
            gate['require_incident_node_route_fraction_at_least'],
        'primary_control_node_route_fraction':
            results['primary_control']['route_fraction'] <=
            gate['maximum_primary_control_node_route_fraction'],
        'secondary_control_node_route_fraction':
            results['secondary_control']['route_fraction'] <=
            gate['maximum_secondary_control_node_route_fraction'],
        'all_audit_weeks_covered_once': weeks_covered,
    }
    passed = all(checks.values())
    return {
        'checks': checks, 'impact_node_localization_gate_passed': passed,
        'recommendation': (
            'PROTECTED_NODE_EXPERT_DEVELOPMENT_ALLOWED' if passed else
            'STOP_OBSERVABLE_NODE_LOCALIZATION_NOT_ESTABLISHED'),
    }


def audit(data_dir, primary_dir, secondary_dir, placebo_dir, sensors_path,
          protocol_path, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError('v8b output exists; preserve it and use a new directory')
    protocol = load_protocol(protocol_path)
    v5c_path = REPO / 'experiments/chronological/matched_placebo_audit_v5c.json'
    v5c = load_v5c_protocol(v5c_path)
    if (v5c['protocol_id'] != protocol['v5c_context']['protocol_id'] or
            sha256(Path(placebo_dir) / 'summary.json') !=
            protocol['v5c_context']['summary_sha256'] or
            sha256(Path(placebo_dir) / 'triple_metrics.csv') !=
            protocol['v5c_context']['triple_metrics_sha256']):
        raise ValueError('v5c inputs differ from frozen v8b context')
    input_hashes = verify_v5c_inputs(data_dir, primary_dir, secondary_dir, v5c)
    if sha256(sensors_path) != protocol['sensor_metadata']['sha256']:
        raise ValueError('Sensor metadata differs from frozen v8b context')
    samples, weeks = read_identity(
        placebo_dir, protocol['expected_common_train_samples'],
        protocol['rolling_origin']['expected_unique_train_iso_weeks'])
    scaler = json.loads((Path(data_dir) / 'scaler.json').read_text(encoding='utf-8'))
    source, categories, features, node_ids, excess, event_indices, feature_weeks = build_data(
        data_dir, primary_dir, secondary_dir, sensors_path, samples,
        protocol['expected_sensor_count'], float(scaler['std']))
    if not np.array_equal(feature_weeks, weeks):
        raise ValueError('v5c label weeks and feature weeks differ')
    folds = rolling_folds(weeks, protocol)
    all_scores = {cohort: [] for cohort in COHORTS}
    all_routes = {cohort: [] for cohort in COHORTS}
    all_labels, all_clusters = [], []
    fold_summaries, event_rows = [], []
    for fold in folds:
        fit_events, audit_events = fold['fit_indices'], fold['audit_indices']
        fit_rows = np.flatnonzero(np.isin(event_indices, fit_events))
        threshold = float(np.quantile(excess[fit_rows], .75))
        labels = (excess >= threshold).astype(np.int64)
        model, route_threshold = fit_node_model(
            features, labels, event_indices, fit_events,
            protocol['estimator']['ridge_alpha'])
        audit_rows = np.flatnonzero(np.isin(event_indices, audit_events))
        fold_summary = {
            'fold': fold['fold'], 'fit_weeks': list(fold['fit_weeks']),
            'audit_weeks': list(fold['audit_weeks']),
            'fit_events': int(len(fit_events)), 'audit_events': int(len(audit_events)),
            'fit_node_label_threshold': threshold,
            'fit_route_threshold': route_threshold, 'cohorts': {},
        }
        incident_scores = np.clip(model.predict(features['incident'][audit_rows]), 0., 1.)
        incident_labels = labels[audit_rows]
        fold_summary['cohorts']['incident'] = metric_summary(
            incident_labels, incident_scores, weeks[event_indices[audit_rows]],
            protocol, int(protocol['uncertainty']['seed']) + fold['fold'] * 10)
        for cohort in COHORTS:
            scores = np.clip(model.predict(features[cohort][audit_rows]), 0., 1.)
            routes = scores >= route_threshold
            all_scores[cohort].append(scores)
            all_routes[cohort].append(routes)
            fold_summary['cohorts'][cohort]['nodes'] = int(len(scores))
            fold_summary['cohorts'][cohort]['route_fraction'] = float(routes.mean())
            fold_summary['cohorts'][cohort]['mean_score'] = float(scores.mean())
            for position, row in enumerate(audit_rows):
                event_rows.append({
                    'fold': fold['fold'], 'cohort': cohort,
                    'positive_sample_index': int(samples[event_indices[row]]),
                    'positive_iso_week': weeks[event_indices[row]],
                    'node_index': int(node_ids[row]), 'score': float(scores[position]),
                    'route': bool(routes[position]),
                    'incident_label': int(labels[row]) if cohort == 'incident' else 0,
                })
        all_labels.extend(incident_labels.tolist())
        all_clusters.extend(weeks[event_indices[audit_rows]].tolist())
        fold_summaries.append(fold_summary)
    results = {}
    for cohort in COHORTS:
        scores = np.concatenate(all_scores[cohort])
        routes = np.concatenate(all_routes[cohort])
        results[cohort] = {
            'nodes': int(len(scores)), 'route_fraction': float(routes.mean()),
            'mean_score': float(scores.mean()),
        }
    incident_scores = np.concatenate(all_scores['incident'])
    incident_labels = np.asarray(all_labels, dtype=np.int64)
    clusters = np.asarray(all_clusters)
    results['incident'].update(metric_summary(
        incident_labels, incident_scores, clusters, protocol,
        int(protocol['uncertainty']['seed'])))
    gate = decision(
        results, protocol,
        len(set(clusters.tolist())) == sum(protocol['rolling_origin']['audit_week_block_sizes']))
    output.mkdir(parents=True, exist_ok=False)
    event_path = output / 'audit_node_scores.csv'
    write_csv(event_path, event_rows, [
        'fold', 'cohort', 'positive_sample_index', 'positive_iso_week',
        'node_index', 'score', 'route', 'incident_label'])
    summary = {
        'status': 'OBSERVABLE_IMPACT_NODE_LOCALIZATION_AUDIT_COMPLETE',
        'scope': protocol['scope'], 'protocol_id': protocol['protocol_id'],
        'protocol_sha256': sha256(protocol_path), 'main_training_ready': False,
        'train_common_triples_only': True, 'validation_residual_arrays_read': False,
        'test_split_read': False, 'expert_trained': False,
        'neural_node_localizer_trained': False, 'features_are_report_time_safe': True,
        'categories': categories, 'folds': fold_summaries,
        'results': results, 'development_gate': gate,
        'inputs': {
            'v8a_protocol_sha256': protocol['v8a_context']['protocol_sha256'],
            'v5c_summary_sha256': protocol['v5c_context']['summary_sha256'],
            'v5c_triple_metrics_sha256': protocol['v5c_context']['triple_metrics_sha256'],
            'positive_inputs': input_hashes[0], 'primary_inputs': input_hashes[1],
            'secondary_inputs': input_hashes[2],
            'protocol_sha256': sha256(protocol_path), 'code_sha256': sha256(__file__),
        },
        'environment': {
            'python_version': sys.version, 'numpy_version': np.__version__,
            'git_head': subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
        },
        'interpretation': protocol['interpretation'],
    }
    summary['outputs'] = {
        event_path.name: {'sha256': sha256(event_path), 'bytes': event_path.stat().st_size}}
    (output / 'summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
        encoding='utf-8')
    print(json.dumps({'status': summary['status'], 'development_gate': gate,
                      'incident_result': results['incident']},
                     ensure_ascii=False, indent=2), flush=True)
    print(f'Saved v8b observable-impact node audit: {output / "summary.json"}', flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--primary-control-dir', type=Path, required=True)
    parser.add_argument('--secondary-control-dir', type=Path, required=True)
    parser.add_argument('--placebo-dir', type=Path, required=True)
    parser.add_argument('--sensors', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, default=Path(__file__).with_name(
        'incident_impact_node_localization_v8b.json'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    audit(args.data_dir, args.primary_control_dir, args.secondary_control_dir,
          args.placebo_dir, args.sensors, args.protocol, args.output)


if __name__ == '__main__':
    main()
