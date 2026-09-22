"""Audit prospective identification of high-impact incidents from report-time features."""

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
from experiments.chronological.audit_matched_controls import (
    read_csv, sha256, write_csv,
)
from experiments.chronological.audit_matched_placebo import (
    load_protocol as load_v5c_protocol, verify_inputs as verify_v5c_inputs,
)


COHORTS = ('incident', 'primary_control', 'secondary_control')


def load_protocol(path):
    protocol = json.loads(Path(path).read_text(encoding='utf-8'))
    if (protocol.get('protocol_id') !=
            'contra_v8_observable_incident_impact_identification_v8a' or
            protocol.get('scope') !=
            'train_only_rolling_origin_observable_high_impact_incident_identification' or
            protocol.get('main_training_ready') is not False):
        raise ValueError('v8a protocol identity or scope changed')
    if [protocol.get(key) for key in ('source_year', 'source_version',
                                      'expected_sensor_count')] != [2023, 8, 496]:
        raise ValueError('v8a is frozen to the 2023 source-v8 Contra496 package')
    if (protocol.get('expected_common_train_samples') != 3106 or
            protocol.get('expected_full_positive_train_samples') != 3604):
        raise ValueError('v8a common training cohort changed')
    context = protocol.get('v5c_context', {})
    if context != {
            'protocol_id': 'contra_v8_matched_placebo_audit_v5c',
            'summary_sha256': 'c443581334f1f0aebece82c910a38c78b1dc29a22565d081f6a583298875571a',
            'triple_metrics_sha256': '192b44e82223a9f943f3b47a5141ea1d19819b7cd0d0c75b0c234d372dcb5bc5',
            'label': 'early_excess_in_train_std',
            'label_definition': 'v5c early baseline-adjusted incident-minus-routine excess divided by frozen train scaler std',
            'role': 'train_only_outcome_label_not_available_at_inference'}:
        raise ValueError('v8a v5c label context changed')
    if protocol.get('sensor_metadata') != {
            'sha256': '682f3cdf75e643f0b37356ab69cbabb27389be5089f41d3b2cbc4bede3332094',
            'role': 'derive_report_freeway_direction_from_nonzero_distance_support'}:
        raise ValueError('v8a sensor metadata changed')
    rolling = protocol.get('rolling_origin', {})
    if rolling != {
            'time_source': 'positive_incident_t0_iso_week',
            'expected_unique_train_iso_weeks': 35,
            'initial_fit_week_count': 18,
            'audit_fold_count': 3,
            'audit_week_block_sizes': [6, 6, 5],
            'fit_rule': 'all_common_train_weeks_strictly_before_each_audit_block',
            'audit_rule': 'each_of_the_final_17_common_train_weeks_exactly_once',
            'expected_common_fit_minimums': [1, 1, 1],
            'expected_common_audit_counts': [536, 618, 338]}:
        raise ValueError('v8a rolling-origin design changed')
    features = protocol.get('features', {})
    if (features.get('family') != 'report_time_safe_event_features' or
            features.get('future_outcome_features_forbidden') is not True or
            features.get('frozen_A_prediction_features_forbidden') is not True or
            features.get('post_event_fields_forbidden') is not True):
        raise ValueError('v8a feature boundary changed')
    estimator = protocol.get('estimator', {})
    if estimator != {
            'family': 'weighted_ridge_binary_score', 'ridge_alpha': 10.0,
            'training_cohorts': list(COHORTS),
            'row_weighting': 'equal_cohort_and_equal_event_weight',
            'incident_target': 'early_excess_at_or_above_fit_incident_q75',
            'control_target': 'exact_zero',
            'label_threshold': 'fit_incident_early_excess_in_train_std_q75',
            'score_clip': [0.0, 1.0],
            'route_threshold': 'fit_incident_score_q75',
            'route_unit': 'event_level_expert_activation'}:
        raise ValueError('v8a estimator design changed')
    uncertainty = protocol.get('uncertainty', {})
    if uncertainty != {
            'method': 'audit_iso_week_cluster_bootstrap', 'draws': 2000,
            'confidence_level': 0.95, 'seed': 2025}:
        raise ValueError('v8a uncertainty design changed')
    gate = protocol.get('development_gate', {})
    if gate != {
            'require_incident_auc_ci_lower_above': 0.55,
            'require_incident_top_quartile_lift_ci_lower_above': 1.2,
            'require_incident_route_fraction_at_least': 0.05,
            'maximum_primary_control_route_fraction': 0.35,
            'maximum_secondary_control_route_fraction': 0.35,
            'require_all_audit_weeks_covered_once': True}:
        raise ValueError('v8a development gate changed')
    boundary = protocol.get('information_boundary', {})
    required = (
        'train_common_triples_only', 'validation_residual_arrays_prohibited',
        'test_split_prohibited', 'future_outcome_labels_prohibited_from_features',
        'audit_weeks_prohibited_from_fit', 'future_weeks_prohibited_from_each_fold_fit',
        'thresholds_and_gate_may_not_change_after_result',
        'expert_training_prohibited', 'node_localizer_training_prohibited')
    if not all(boundary.get(key) is True for key in required):
        raise ValueError('v8a information boundary changed')
    return protocol


def iso_week(value):
    current = datetime.fromisoformat(value).isocalendar()
    return f'{current.year:04d}-W{current.week:02d}'


def rolling_folds(weeks, protocol):
    specification = protocol['rolling_origin']
    unique = sorted(set(weeks))
    if len(unique) != specification['expected_unique_train_iso_weeks']:
        raise ValueError('Unique common-triple week count changed')
    initial = specification['initial_fit_week_count']
    sizes = specification['audit_week_block_sizes']
    if initial + sum(sizes) != len(unique):
        raise ValueError('Rolling-origin week partition is incomplete')
    folds, position, audit_weeks = [], initial, []
    for number, size in enumerate(sizes, start=1):
        current_fit = unique[:position]
        current_audit = unique[position:position + size]
        fit = np.flatnonzero(np.isin(weeks, current_fit))
        audit = np.flatnonzero(np.isin(weeks, current_audit))
        if (len(fit) < specification['expected_common_fit_minimums'][number - 1] or
                not len(audit) or np.intersect1d(fit, audit).size):
            raise ValueError('Rolling-origin fit/audit separation changed')
        folds.append({
            'fold': number, 'fit_weeks': current_fit,
            'audit_weeks': current_audit, 'fit_indices': fit,
            'audit_indices': audit,
        })
        audit_weeks.extend(current_audit)
        position += size
    if sorted(audit_weeks) != unique[initial:]:
        raise ValueError('Audit weeks are not covered exactly once')
    return folds


def roc_auc(labels, scores):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if (labels.ndim != 1 or scores.shape != labels.shape or
            not np.isfinite(scores).all() or set(labels.tolist()) - {0, 1} or
            labels.sum() == 0 or labels.sum() == len(labels)):
        raise ValueError('ROC AUC requires finite binary scores with both classes')
    order = np.argsort(scores, kind='stable')
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and scores[order[end]] == scores[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end + 1) / 2
        start = end
    positives = labels == 1
    negatives = labels == 0
    return float((ranks[positives].sum() - labels.sum() * (labels.sum() + 1) / 2) /
                 (labels.sum() * negatives.sum()))


def top_quartile_lift(labels, scores):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.ndim != 1 or scores.shape != labels.shape or not len(labels):
        raise ValueError('Invalid top-quartile input')
    prevalence = float(labels.mean())
    if prevalence <= 0:
        return 0.0
    count = max(1, int(math.ceil(len(labels) * .25)))
    selected = np.argsort(-scores, kind='stable')[:count]
    return float(labels[selected].mean() / prevalence)


def bootstrap_metric(labels, scores, clusters, metric, draws, confidence, seed):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    clusters = np.asarray(clusters)
    unique = np.unique(clusters)
    if len(unique) < 2 or len(labels) != len(scores) or len(labels) != len(clusters):
        raise ValueError('Cluster bootstrap inputs are not aligned')
    by_cluster = [np.flatnonzero(clusters == value) for value in unique]
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(int(draws)):
        selected = rng.integers(0, len(unique), size=len(unique))
        indices = np.concatenate([by_cluster[index] for index in selected])
        try:
            values.append(metric(labels[indices], scores[indices]))
        except ValueError:
            continue
    if len(values) < max(100, int(draws) // 2):
        raise ValueError('Too few valid bootstrap draws')
    alpha = (1 - float(confidence)) / 2
    return float(np.quantile(values, alpha)), float(np.quantile(values, 1 - alpha))


def score_metrics(labels, scores, clusters, protocol, seed):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    auc = roc_auc(labels, scores)
    lift = top_quartile_lift(labels, scores)
    auc_ci = bootstrap_metric(
        labels, scores, clusters, roc_auc,
        protocol['uncertainty']['draws'], protocol['uncertainty']['confidence_level'], seed)
    lift_ci = bootstrap_metric(
        labels, scores, clusters, top_quartile_lift,
        protocol['uncertainty']['draws'], protocol['uncertainty']['confidence_level'], seed + 1)
    return {
        'events': int(len(labels)), 'positive_events': int(labels.sum()),
        'positive_fraction': float(labels.mean()), 'roc_auc': auc,
        'roc_auc_ci_low': auc_ci[0], 'roc_auc_ci_high': auc_ci[1],
        'top_quartile_lift': lift,
        'top_quartile_lift_ci_low': lift_ci[0],
        'top_quartile_lift_ci_high': lift_ci[1],
    }


def fit_fold(features, labels, fit_indices, model_alpha):
    training_features, training_targets, training_weights = [], [], []
    for cohort_number, cohort in enumerate(COHORTS):
        current_features = features[cohort][fit_indices]
        current_labels = labels[fit_indices] if cohort == 'incident' else np.zeros(len(fit_indices))
        training_features.append(current_features)
        training_targets.append(current_labels)
        training_weights.append(np.full(len(fit_indices), 1. / len(fit_indices)))
    model = fit_weighted_ridge(
        np.concatenate(training_features), np.concatenate(training_targets),
        np.concatenate(training_weights), float(model_alpha))
    incident_fit_scores = np.clip(model.predict(features['incident'][fit_indices]), 0., 1.)
    return model, float(np.quantile(incident_fit_scores, .75))


def read_labels(placebo_dir, expected_samples, expected_weeks):
    rows = read_csv(Path(placebo_dir) / 'triple_metrics.csv')
    rows = [row for row in rows if row['split'] == 'train']
    if len(rows) != expected_samples:
        raise ValueError('v5c train triple count changed')
    values, samples, weeks = [], [], []
    for row in rows:
        samples.append(int(row['positive_sample_index']))
        weeks.append(row['positive_iso_week'])
        values.append(float(row['early_excess_in_train_std']))
    if len(set(samples)) != expected_samples or len(set(weeks)) != expected_weeks:
        raise ValueError('v5c train label identities or weeks changed')
    return np.asarray(samples, dtype=np.int64), np.asarray(weeks), np.asarray(values, dtype=np.float64)


def build_features(data_dir, primary_dir, secondary_dir, sensors_path, samples, nodes):
    source = FeatureSource(
        data_dir, primary_dir, secondary_dir, sensors_path, 'train',
        len(samples), 3604, nodes)
    categories = category_schema(source)
    features = {cohort: [] for cohort in COHORTS}
    identity = {cohort: [] for cohort in COHORTS}
    weeks = []
    for position in range(len(samples)):
        for cohort in COHORTS:
            sample = source.sample(cohort, position)
            features[cohort].append(feature_vector(sample, categories))
            identity[cohort].append(sample['sample'])
            if cohort == 'incident':
                weeks.append(sample['iso_week'])
    for cohort in COHORTS:
        features[cohort] = np.asarray(features[cohort], dtype=np.float64)
        if not np.array_equal(identity[cohort], samples):
            raise ValueError(f'{cohort} feature identity differs from v5c labels')
    return features, categories, np.asarray(weeks)


def decision(primary, protocol, weeks):
    gate = protocol['development_gate']
    checks = {
        'incident_auc_ci_lower':
            primary['incident']['roc_auc_ci_low'] > gate['require_incident_auc_ci_lower_above'],
        'incident_top_quartile_lift_ci_lower':
            primary['incident']['top_quartile_lift_ci_low'] >
            gate['require_incident_top_quartile_lift_ci_lower_above'],
        'incident_route_fraction':
            primary['incident']['route_fraction'] >=
            gate['require_incident_route_fraction_at_least'],
        'primary_control_route_fraction':
            primary['primary_control']['route_fraction'] <=
            gate['maximum_primary_control_route_fraction'],
        'secondary_control_route_fraction':
            primary['secondary_control']['route_fraction'] <=
            gate['maximum_secondary_control_route_fraction'],
        'all_audit_weeks_covered_once': weeks,
    }
    passed = all(checks.values())
    return {
        'checks': checks, 'impact_identification_gate_passed': passed,
        'recommendation': (
            'OBSERVABLE_IMPACT_GATE_DEVELOPMENT_ALLOWED' if passed else
            'STOP_OBSERVABLE_IMPACT_IDENTIFICATION_NOT_ESTABLISHED'),
    }


def audit(data_dir, primary_dir, secondary_dir, placebo_dir, sensors_path,
          protocol_path, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError('v8a output exists; preserve it and use a new directory')
    protocol = load_protocol(protocol_path)
    v5c_path = REPO / 'experiments/chronological/matched_placebo_audit_v5c.json'
    v5c = load_v5c_protocol(v5c_path)
    if (v5c['protocol_id'] != protocol['v5c_context']['protocol_id'] or
            sha256(Path(placebo_dir) / 'summary.json') !=
            protocol['v5c_context']['summary_sha256'] or
            sha256(Path(placebo_dir) / 'triple_metrics.csv') !=
            protocol['v5c_context']['triple_metrics_sha256']):
        raise ValueError('v5c inputs differ from frozen v8a context')
    if sha256(sensors_path) != protocol['sensor_metadata']['sha256']:
        raise ValueError('Sensor metadata differs from frozen v8a context')
    input_hashes = verify_v5c_inputs(data_dir, primary_dir, secondary_dir, v5c)
    samples, weeks, excess = read_labels(
        placebo_dir, protocol['expected_common_train_samples'],
        protocol['rolling_origin']['expected_unique_train_iso_weeks'])
    features, categories, feature_weeks = build_features(
        data_dir, primary_dir, secondary_dir, sensors_path, samples,
        protocol['expected_sensor_count'])
    if not np.array_equal(feature_weeks, weeks):
        raise ValueError('v5c label weeks and feature weeks differ')
    folds = rolling_folds(weeks, protocol)
    labels_by_fold = []
    audit_rows, fold_summaries = [], []
    all_scores = {cohort: [] for cohort in COHORTS}
    all_labels, all_clusters = [], []
    route_flags = {cohort: [] for cohort in COHORTS}
    for fold in folds:
        fit_indices, audit_indices = fold['fit_indices'], fold['audit_indices']
        threshold = float(np.quantile(excess[fit_indices], .75))
        labels = (excess >= threshold).astype(np.int64)
        model, route_threshold = fit_fold(
            features, labels, fit_indices, protocol['estimator']['ridge_alpha'])
        fold_metrics = {'fold': fold['fold'], 'fit_weeks': list(fold['fit_weeks']),
                        'audit_weeks': list(fold['audit_weeks']),
                        'fit_events': int(len(fit_indices)),
                        'audit_events': int(len(audit_indices)),
                        'fit_label_threshold': threshold,
                        'fit_route_threshold': route_threshold, 'cohorts': {}}
        incident_scores = np.clip(model.predict(features['incident'][audit_indices]), 0., 1.)
        incident_labels = labels[audit_indices]
        fold_metrics['cohorts']['incident'] = score_metrics(
            incident_labels, incident_scores, weeks[audit_indices], protocol,
            int(protocol['uncertainty']['seed']) + fold['fold'] * 10)
        for cohort_number, cohort in enumerate(COHORTS):
            scores = np.clip(model.predict(features[cohort][audit_indices]), 0., 1.)
            flags = scores >= route_threshold
            route_flags[cohort].append(flags)
            all_scores[cohort].append(scores)
            fold_metrics['cohorts'][cohort]['route_fraction'] = float(flags.mean())
            fold_metrics['cohorts'][cohort]['mean_score'] = float(scores.mean())
            for position, index in enumerate(audit_indices):
                audit_rows.append({
                    'fold': fold['fold'], 'cohort': cohort,
                    'positive_sample_index': int(samples[index]),
                    'positive_iso_week': weeks[index],
                    'score': float(scores[position]),
                    'route': bool(flags[position]),
                    'incident_label': int(labels[index]) if cohort == 'incident' else 0,
                })
        all_labels.extend(incident_labels.tolist())
        all_clusters.extend(weeks[audit_indices].tolist())
        fold_summaries.append(fold_metrics)
    combined_primary = {}
    for cohort in COHORTS:
        scores = np.concatenate(all_scores[cohort])
        flags = np.concatenate(route_flags[cohort])
        combined_primary[cohort] = {
            'events': int(len(scores)), 'route_fraction': float(flags.mean()),
            'mean_score': float(scores.mean()),
        }
    incident_scores = np.concatenate(all_scores['incident'])
    incident_labels = np.asarray(all_labels, dtype=np.int64)
    incident_clusters = np.asarray(all_clusters)
    combined_primary['incident'].update(score_metrics(
        incident_labels, incident_scores, incident_clusters, protocol,
        int(protocol['uncertainty']['seed'])))
    gate = decision(
        combined_primary,
        protocol,
        len(set(all_clusters)) == sum(protocol['rolling_origin']['audit_week_block_sizes']))
    output.mkdir(parents=True, exist_ok=False)
    event_path = output / 'audit_event_scores.csv'
    write_csv(event_path, audit_rows, [
        'fold', 'cohort', 'positive_sample_index', 'positive_iso_week',
        'score', 'route', 'incident_label'])
    summary = {
        'status': 'OBSERVABLE_IMPACT_IDENTIFICATION_AUDIT_COMPLETE',
        'scope': protocol['scope'], 'protocol_id': protocol['protocol_id'],
        'protocol_sha256': sha256(protocol_path), 'main_training_ready': False,
        'train_common_triples_only': True, 'validation_residual_arrays_read': False,
        'test_split_read': False, 'expert_trained': False,
        'node_localizer_trained': False, 'features_are_report_time_safe': True,
        'categories': categories, 'folds': fold_summaries,
        'results': combined_primary, 'development_gate': gate,
        'inputs': {
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
                      'incident_result': combined_primary['incident']},
                     ensure_ascii=False, indent=2), flush=True)
    print(f'Saved v8a observable-impact audit: {output / "summary.json"}', flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--primary-control-dir', type=Path, required=True)
    parser.add_argument('--secondary-control-dir', type=Path, required=True)
    parser.add_argument('--placebo-dir', type=Path, required=True)
    parser.add_argument('--sensors', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, default=Path(__file__).with_name(
        'incident_impact_identification_v8a.json'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    audit(args.data_dir, args.primary_control_dir, args.secondary_control_dir,
          args.placebo_dir, args.sensors, args.protocol, args.output)


if __name__ == '__main__':
    main()
