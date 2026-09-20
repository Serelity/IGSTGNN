"""Audit constrained signed-residual oracles and train-only shallow probes."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.chronological.audit_expert_benefit import (
    FeatureSource, category_schema, cluster_bootstrap_mean, feature_names,
    feature_vector, fit_weighted_ridge, load_high_impact, verify_package,
)
from experiments.chronological.audit_matched_controls import read_csv, sha256, write_csv


SPLITS = ('train', 'val')
COHORTS = ('incident', 'primary_control', 'secondary_control')
LEVELS = ('event', 'event_node', 'event_node_phase')
PHASES = ((0, 3), (3, 6))


def load_protocol(path):
    protocol = json.loads(Path(path).read_text(encoding='utf-8'))
    if (protocol.get('protocol_id') != 'contra_v8_signed_residual_probe_v7b' or
            protocol.get('scope') !=
            'baseline_anchored_local_signed_residual_feasibility' or
            protocol.get('main_training_ready') is not False):
        raise ValueError('v7b protocol identity or scope changed')
    if [protocol.get(key) for key in ('source_year', 'source_version',
                                      'expected_sensor_count')] != [2023, 8, 496]:
        raise ValueError('v7b is frozen to the 2023 source-v8 Contra496 package')
    if protocol.get('expected_common_triples') != {'train': 3106, 'val': 618}:
        raise ValueError('v7b common-triple cohort changed')
    if protocol.get('expected_positive_samples') != {'train': 3604, 'val': 917}:
        raise ValueError('v7b full-positive cohort changed')
    v7a = protocol.get('v7a_input', {})
    if (v7a.get('protocol_id') != 'contra_v8_signed_residual_materialize_v7a' or
            len(v7a.get('protocol_sha256', '')) != 64 or
            v7a.get('status') != 'SIGNED_RESIDUAL_MATERIALIZATION_COMPLETE'):
        raise ValueError('v7b v7a-input identity changed')
    sensors = protocol.get('sensor_metadata', {})
    if sensors != {
            'sha256': '682f3cdf75e643f0b37356ab69cbabb27389be5089f41d3b2cbc4bede3332094',
            'role': 'derive_report_freeway_direction_from_nonzero_distance_support',
            'required_unique_categories': [
                '4-E', '4-W', '24-E', '24-W', '242-N', '242-S']}:
        raise ValueError('v7b sensor metadata changed')
    estimand = protocol.get('estimand', {})
    if (estimand.get('baseline') != 'frozen_A_incident_on' or
            estimand.get('signed_residual') !=
            'target_minus_frozen_A_prediction' or
            estimand.get('corrected_prediction') !=
            'frozen_A_prediction_plus_probe_correction' or
            estimand.get('causal_effect_claimed') is not False):
        raise ValueError('v7b estimand changed')
    support = protocol.get('support', {})
    if (support.get('active_horizons_zero_based_half_open') != [0, 6] or
            support.get('protected_horizons_zero_based_half_open') != [6, 12] or
            support.get('protected_noncandidate_nodes') is not True or
            support.get('correction_exactly_zero_outside_support') is not True):
        raise ValueError('v7b support changed')
    oracle = protocol.get('oracle', {})
    if (oracle.get('families') != list(LEVELS) or
            oracle.get('phase_slices_zero_based_half_open') != [[0, 3], [3, 6]] or
            oracle.get('loss_optimum') != 'median_signed_residual' or
            oracle.get('primary_family_for_development_tier') !=
            'event_node_phase' or
            oracle.get('primary_population_for_development_tier') !=
            'full_positive_validation' or
            oracle.get('all_node_relative_improvement_percent') != {
                'stop_below': 0.3, 'small_probe_below': 1.0,
                'formal_study_at_or_above': 1.0} or
            oracle.get('reported_as_model_performance') is not False or
            oracle.get('not_an_upper_bound_on_arbitrary_neural_residual') is not True):
        raise ValueError('v7b oracle design or thresholds changed')
    probe = protocol.get('probe', {})
    if (probe.get('family') != 'weighted_ridge_regression' or
            probe.get('levels') != list(LEVELS) or
            probe.get('primary_level') != 'event_node_phase' or
            float(probe.get('ridge_alpha', 0)) != 10.0 or
            probe.get('training_cohorts') != list(COHORTS) or
            probe.get('incident_target') !=
            'median_signed_residual_on_level_support' or
            probe.get('control_target') !=
            'exact_zero_without_reading_control_future_Y' or
            probe.get('control_row_support') !=
            'candidate_nodes_and_predeclared_phases_without_valid_or_future_Y' or
            probe.get('validation_prediction_support') !=
            'candidate_nodes_and_predeclared_phases_independent_of_future_Y' or
            probe.get('row_weighting') != 'equal_event_cohort_weight' or
            probe.get('prediction_clip') !=
            'symmetric_train_incident_target_abs_q99_per_level'):
        raise ValueError('v7b probe design changed')
    uncertainty = protocol.get('uncertainty', {})
    if (uncertainty.get('method') != 'positive_incident_iso_week_cluster_bootstrap' or
            int(uncertainty.get('draws', 0)) != 2000 or
            float(uncertainty.get('confidence_level', 0)) != 0.95 or
            int(uncertainty.get('seed', 0)) != 2025):
        raise ValueError('v7b uncertainty design changed')
    expected_gate = {
        'primary_probe': 'event_node_phase',
        'global_population': 'full_positive_validation',
        'high_impact_population': 'common_matched_validation',
        'require_positive_global_point_improvement': True,
        'maximum_global_harm_fraction_of_A_mae': 0.001,
        'require_candidate_H1_H6_improvement_ci_lower_above_zero': True,
        'require_high_impact_candidate_H1_H6_point_improvement': True,
        'maximum_each_control_H1_H6_harm_fraction_of_A_mae': 0.005,
        'require_nonzero_correction_fraction': 0.05,
        'require_exact_A_equality_outside_support': True,
    }
    if protocol.get('development_gate') != expected_gate:
        raise ValueError('v7b prospective development gate changed')
    boundary = protocol.get('information_boundary', {})
    required = (
        'train_and_validation_only', 'test_split_prohibited',
        'frozen_A_not_updated', 'neural_expert_training_prohibited',
        'v7a_cohort_may_not_change',
        'validation_may_not_tune_features_alpha_clip_or_gate',
        'control_future_Y_prohibited_from_probe_fit',
        'validation_prediction_support_independent_of_future_Y',
        'v5c_label_used_only_for_evaluation_stratification',
    )
    if not all(boundary.get(key) is True for key in required):
        raise ValueError('v7b information boundary changed')
    return protocol


def verify_inputs(data_dir, primary_dir, secondary_dir, materialized_dir,
                  placebo_dir, sensors_path, protocol):
    materialized_dir, placebo_dir = Path(materialized_dir), Path(placebo_dir)
    summary_path = materialized_dir / 'summary.json'
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    expected = protocol['v7a_input']
    if (summary.get('status') != expected['status'] or
            summary.get('protocol_id') != expected['protocol_id'] or
            summary.get('protocol_sha256') != expected['protocol_sha256'] or
            summary.get('test_split_read') is not False or
            summary.get('model_training_performed') is not False):
        raise ValueError('v7a materialization identity or boundary changed')
    expected_outputs = {
        f'{split}_{cohort}_signed_residuals.npz'
        for split in SPLITS for cohort in (*COHORTS, 'incident_full')
    }
    if set(summary.get('outputs', {})) != expected_outputs:
        raise ValueError('v7a materialization output set is incomplete')
    for filename, metadata in summary['outputs'].items():
        if sha256(materialized_dir / filename) != metadata['sha256']:
            raise ValueError(f'v7a output checksum mismatch: {filename}')
    package = verify_package(data_dir)
    if package != summary['inputs']['positive_package']:
        raise ValueError('Positive package differs from v7a materialization')
    for directory, key in ((primary_dir, 'primary_controls'),
                           (secondary_dir, 'secondary_controls')):
        actual = {
            name: sha256(Path(directory) / name)
            for name in summary['inputs'][key]
        }
        if actual != summary['inputs'][key]:
            raise ValueError(f'{key} differ from v7a materialization')
    sensors_hash = sha256(sensors_path)
    context = json.loads(
        (Path(data_dir) / 'context_manifest.json').read_text(encoding='utf-8'))
    if (sensors_hash != protocol['sensor_metadata']['sha256'] or
            sensors_hash not in context.get('sources', {}).values()):
        raise ValueError('Sensor metadata differs from the report-location context source')
    v5c = protocol['v5c_stratification_input']
    if (sha256(placebo_dir / 'summary.json') != v5c['summary_sha256'] or
            sha256(placebo_dir / 'triple_metrics.csv') !=
            v5c['triple_metrics_sha256']):
        raise ValueError('v5c stratification artifacts differ from the v7b protocol')
    placebo_summary = json.loads(
        (placebo_dir / 'summary.json').read_text(encoding='utf-8'))
    if (placebo_summary.get('protocol_id') != v5c['protocol_id'] or
            placebo_summary.get('test_split_read') is not False):
        raise ValueError('v5c identity or test boundary changed')
    return summary


def load_residuals(materialized_dir, split, cohort, expected_samples, expected_nodes):
    path = Path(materialized_dir) / f'{split}_{cohort}_signed_residuals.npz'
    with np.load(path, allow_pickle=False) as stored:
        required = {
            'signed_residual', 'baseline_prediction', 'valid', 'candidate_mask',
            'positive_sample_index', 'baseline_all_absolute_sum',
            'baseline_all_valid_count',
        }
        if set(stored.files) != required:
            raise ValueError(f'Unexpected v7a array schema: {path.name}')
        arrays = {key: stored[key].copy() for key in stored.files}
    shape = (expected_samples, 6, expected_nodes, 1)
    if (arrays['signed_residual'].shape != shape or
            arrays['baseline_prediction'].shape != shape or
            arrays['valid'].shape != shape or
            arrays['candidate_mask'].shape != (expected_samples, expected_nodes) or
            arrays['positive_sample_index'].shape != (expected_samples,) or
            arrays['baseline_all_absolute_sum'].shape != (expected_samples,) or
            arrays['baseline_all_valid_count'].shape != (expected_samples,) or
            arrays['signed_residual'].dtype != np.float32 or
            arrays['baseline_prediction'].dtype != np.float32 or
            arrays['valid'].dtype != np.bool_ or
            arrays['candidate_mask'].dtype != np.bool_ or
            arrays['baseline_all_absolute_sum'].dtype != np.float64 or
            arrays['baseline_all_valid_count'].dtype != np.int64 or
            not np.isfinite(arrays['signed_residual']).all() or
            not np.isfinite(arrays['baseline_prediction']).all() or
            not np.isfinite(arrays['baseline_all_absolute_sum']).all() or
            np.any(arrays['baseline_all_valid_count'] <= 0)):
        raise ValueError(f'Invalid v7a arrays: {path.name}')
    return arrays


def geometric_support(arrays):
    return np.broadcast_to(
        arrays['candidate_mask'][:, None, :, None], arrays['valid'].shape)


def active_support(arrays):
    return geometric_support(arrays) & arrays['valid']


def correction_metrics(arrays, correction):
    residual = arrays['signed_residual'].astype(np.float64)
    correction = np.asarray(correction, dtype=np.float64)
    if correction.shape != residual.shape or not np.isfinite(correction).all():
        raise ValueError('Correction and residual arrays differ')
    geometry = geometric_support(arrays)
    if np.any(correction[~geometry] != 0):
        raise ValueError('Residual correction escaped candidate H1-H6 support')
    active = geometry & arrays['valid']
    baseline = np.abs(residual)
    corrected = np.abs(residual - correction)
    baseline_support_sum = np.where(active, baseline, 0.).sum(axis=(1, 2, 3))
    corrected_support_sum = np.where(active, corrected, 0.).sum(axis=(1, 2, 3))
    support_count = active.sum(axis=(1, 2, 3))
    if np.any(support_count <= 0):
        raise ValueError('At least one residual event has empty active support')
    all_count = arrays['baseline_all_valid_count'].astype(np.int64)
    baseline_all_sum = arrays['baseline_all_absolute_sum'].astype(np.float64)
    if (np.any(all_count < support_count) or np.any(baseline_all_sum < 0) or
            np.any(baseline_support_sum > baseline_all_sum + 1e-8)):
        raise ValueError('Stored frozen-A totals are inconsistent with local support')
    corrected_all_sum = baseline_all_sum - baseline_support_sum + corrected_support_sum
    baseline_all_mae = float(baseline_all_sum.sum() / all_count.sum())
    corrected_all_mae = float(corrected_all_sum.sum() / all_count.sum())
    baseline_support_mae = float(baseline_support_sum.sum() / support_count.sum())
    corrected_support_mae = float(corrected_support_sum.sum() / support_count.sum())
    nonzero = np.abs(correction[active]) > 1e-6
    return {
        'all': {
            'valid_cells': int(all_count.sum()),
            'mae_A': baseline_all_mae,
            'mae_corrected': corrected_all_mae,
            'improvement_vs_A': baseline_all_mae - corrected_all_mae,
            'relative_improvement_vs_A_percent':
                100 * (baseline_all_mae - corrected_all_mae) / baseline_all_mae,
        },
        'candidate_h1_h6': {
            'valid_cells': int(support_count.sum()),
            'mae_A': baseline_support_mae,
            'mae_corrected': corrected_support_mae,
            'improvement_vs_A': baseline_support_mae - corrected_support_mae,
            'relative_improvement_vs_A_percent':
                100 * (baseline_support_mae - corrected_support_mae) /
                baseline_support_mae,
        },
        'correction': {
            'valid_support_cells': int(active.sum()),
            'nonzero_fraction': float(nonzero.mean()),
            'abs_mean': float(np.abs(correction[active]).mean()),
            'abs_max': float(np.abs(correction[active]).max()),
        },
        'protected_h7_h12_exact_A': True,
        'protected_noncandidate_exact_A': True,
        '_per_event': {
            'all_improvement_vs_A':
                (baseline_support_sum - corrected_support_sum) / all_count,
            'candidate_h1_h6_improvement_vs_A':
                (baseline_support_sum - corrected_support_sum) / support_count,
        },
    }


def oracle_correction(arrays, level):
    residual = arrays['signed_residual'].astype(np.float64)
    valid = arrays['valid']
    candidate = arrays['candidate_mask']
    correction = np.zeros_like(residual, dtype=np.float64)
    for event in range(len(residual)):
        nodes = np.flatnonzero(candidate[event])
        if level == 'event':
            event_valid = valid[event][:, nodes, :]
            values = residual[event][:, nodes, :][event_valid]
            if values.size:
                event_correction = correction[event]
                event_correction[:, nodes, :] = np.median(values)
            continue
        for node in nodes:
            if level == 'event_node':
                selected = valid[event, :, node, 0]
                if selected.any():
                    correction[event, :, node, 0] = np.median(
                        residual[event, selected, node, 0])
            elif level == 'event_node_phase':
                for first, last in PHASES:
                    selected = valid[event, first:last, node, 0]
                    if selected.any():
                        correction[event, first:last, node, 0] = np.median(
                            residual[event, first:last, node, 0][selected])
            else:
                raise ValueError('Unknown residual oracle family')
    return correction


def oracle_summary(arrays):
    result = {}
    for level in LEVELS:
        metrics = correction_metrics(arrays, oracle_correction(arrays, level))
        if (metrics['all']['improvement_vs_A'] < -1e-10 or
                metrics['candidate_h1_h6']['improvement_vs_A'] < -1e-10):
            raise ValueError('Median signed-residual oracle worsened frozen A')
        metrics.pop('_per_event')
        result[level] = metrics
    return result


def residual_feature_vector(sample, baseline, categories, node=None, phase=None):
    values = feature_vector(sample, categories, node).tolist()
    candidate = sample['candidate']
    if node is None:
        series = np.asarray(baseline[:, candidate, 0], dtype=np.float64).mean(axis=1)
    else:
        series = np.asarray(baseline[:, node, 0], dtype=np.float64)
    values.extend(series.tolist())
    values.extend([
        float(series.mean()), float(series.std()),
        float(series[-3:].mean() - series[:3].mean()),
    ])
    if phase is not None:
        first, last = PHASES[phase]
        current = series[first:last]
        values.extend([
            float(phase == 0), float(phase == 1), float(current.mean()),
            float(current.std()), float(current[-1] - current[0]),
        ])
    return np.asarray(values, dtype=np.float64)


def residual_feature_names(categories, level):
    base_level = 'event' if level == 'event' else 'event_node'
    names = feature_names(categories, base_level)
    prefix = 'candidate_A_prediction' if level == 'event' else 'node_A_prediction'
    names.extend([f'{prefix}_h{horizon}' for horizon in range(1, 7)])
    names.extend([f'{prefix}_mean', f'{prefix}_std', f'{prefix}_late_minus_early'])
    if level == 'event_node_phase':
        names.extend([
            'phase=H1-H3', 'phase=H4-H6', 'phase_A_prediction_mean',
            'phase_A_prediction_std', 'phase_A_prediction_last_minus_first',
        ])
    return names


def target_for(residual, valid, nodes, level, use_residual_target):
    rows = []
    if not use_residual_target:
        if level == 'event':
            return [(None, None, 0.0)] if len(nodes) else rows
        if level == 'event_node':
            return [(int(node), None, 0.0) for node in nodes]
        if level == 'event_node_phase':
            return [(int(node), phase, 0.0)
                    for node in nodes for phase in range(len(PHASES))]
        raise ValueError('Unknown residual probe level')
    if residual is None or valid is None:
        raise ValueError('Incident residual target requires future outcomes')
    if level == 'event':
        selected = valid[:, nodes, 0]
        if not selected.any():
            return rows
        target = float(np.median(residual[:, nodes, 0][selected]))
        return [(None, None, target)]
    for node in nodes:
        if level == 'event_node':
            selected = valid[:, node, 0]
            if selected.any():
                target = float(np.median(residual[:, node, 0][selected]))
                rows.append((int(node), None, target))
        elif level == 'event_node_phase':
            for phase, (first, last) in enumerate(PHASES):
                selected = valid[first:last, node, 0]
                if selected.any():
                    target = float(np.median(
                        residual[first:last, node, 0][selected]))
                    rows.append((int(node), phase, target))
        else:
            raise ValueError('Unknown residual probe level')
    return rows


def evaluation_target_for(residual, valid, nodes, level):
    """Create fixed prediction rows; NaN marks rows without an evaluable target."""
    if level == 'event':
        selected = valid[:, nodes, 0]
        target = float(np.median(residual[:, nodes, 0][selected])) \
            if selected.any() else np.nan
        return [(None, None, target)] if len(nodes) else []
    rows = []
    for node in nodes:
        if level == 'event_node':
            selected = valid[:, node, 0]
            target = float(np.median(residual[:, node, 0][selected])) \
                if selected.any() else np.nan
            rows.append((int(node), None, target))
        elif level == 'event_node_phase':
            for phase, (first, last) in enumerate(PHASES):
                selected = valid[first:last, node, 0]
                target = float(np.median(
                    residual[first:last, node, 0][selected])) \
                    if selected.any() else np.nan
                rows.append((int(node), phase, target))
        else:
            raise ValueError('Unknown residual probe level')
    return rows


def make_probe_rows(source, cohort, arrays, categories, level, target_mode):
    if target_mode not in ('residual', 'zero', 'evaluation'):
        raise ValueError('Unknown residual-probe target mode')
    features, targets, weights, metadata = [], [], [], []
    for position in range(len(arrays['positive_sample_index'])):
        sample = source.sample(cohort, position)
        if (sample['sample'] != int(arrays['positive_sample_index'][position]) or
                not np.array_equal(sample['candidate'], arrays['candidate_mask'][position])):
            raise ValueError('Feature rows and v7a residuals have different identities')
        nodes = np.flatnonzero(sample['candidate'])
        if target_mode == 'evaluation':
            rows = evaluation_target_for(
                arrays['signed_residual'][position], arrays['valid'][position],
                nodes, level)
        else:
            use_residual_target = target_mode == 'residual'
            rows = target_for(
                arrays['signed_residual'][position] if use_residual_target else None,
                arrays['valid'][position] if use_residual_target else None,
                nodes, level, use_residual_target)
        if not rows:
            raise ValueError('Residual-probe event has no valid training rows')
        event_weight = 1.0 / len(rows)
        for node, phase, target in rows:
            features.append(residual_feature_vector(
                sample, arrays['baseline_prediction'][position], categories,
                node=node, phase=phase))
            targets.append(target)
            weights.append(event_weight)
            metadata.append((position, node, phase, sample))
    return (np.asarray(features), np.asarray(targets), np.asarray(weights), metadata)


def correction_from_predictions(arrays, predictions, metadata, clip):
    correction = np.zeros_like(arrays['signed_residual'], dtype=np.float64)
    predictions = np.clip(np.asarray(predictions, dtype=np.float64), -clip, clip)
    if len(predictions) != len(metadata):
        raise ValueError('Residual predictions and metadata differ')
    for value, (position, node, phase, sample) in zip(predictions, metadata):
        if node is None:
            event_correction = correction[position]
            event_correction[:, sample['candidate'], 0] = value
        elif phase is None:
            correction[position, :, node, 0] = value
        else:
            first, last = PHASES[phase]
            correction[position, first:last, node, 0] = value
    return correction, predictions


def uncertainty(metrics, clusters, protocol, seed):
    event = metrics['_per_event']
    values = np.column_stack([
        event['all_improvement_vs_A'],
        event['candidate_h1_h6_improvement_vs_A'],
        -event['candidate_h1_h6_improvement_vs_A'],
    ])
    low, high = cluster_bootstrap_mean(
        values, clusters, int(protocol['uncertainty']['draws']),
        float(protocol['uncertainty']['confidence_level']), seed)
    keys = (
        'all_improvement_vs_A', 'candidate_h1_h6_improvement_vs_A',
        'candidate_h1_h6_harm_vs_A',
    )
    return {
        key: {'mean': float(values[:, index].mean()),
              'ci_low': float(low[index]), 'ci_high': float(high[index])}
        for index, key in enumerate(keys)
    }


def high_impact_metrics(arrays, correction, selected_events):
    active = active_support(arrays) & selected_events[:, None, None, None]
    count = int(active.sum())
    if count < 1:
        raise ValueError('High-impact residual population is empty')
    residual = arrays['signed_residual'].astype(np.float64)
    baseline = float(np.abs(residual)[active].mean())
    corrected = float(np.abs(residual - correction)[active].mean())
    return {
        'events': int(selected_events.sum()), 'valid_cells': count,
        'mae_A': baseline, 'mae_corrected': corrected,
        'improvement_vs_A': baseline - corrected,
    }


def write_probe_rows(path, split, cohort, level, targets, predictions, metadata,
                     high_impact):
    fields = [
        'split', 'cohort', 'level', 'positive_sample_index', 'incident_id',
        'positive_t0', 'positive_iso_week', 'node_index', 'phase',
        'target_correction', 'predicted_correction', 'high_impact_v5c',
    ]
    rows = []
    for target, prediction, (_, node, phase, sample) in zip(
            targets, predictions, metadata):
        rows.append({
            'split': split, 'cohort': cohort, 'level': level,
            'positive_sample_index': sample['sample'],
            'incident_id': sample['incident_id'],
            'positive_t0': sample['positive_t0'],
            'positive_iso_week': sample['iso_week'],
            'node_index': '' if node is None else node,
            'phase': '' if phase is None else phase,
            'target_correction': float(target) if np.isfinite(target) else '',
            'predicted_correction': float(prediction),
            'high_impact_v5c': high_impact.get(
                (split, sample['sample']), ''),
        })
    write_csv(path, rows, fields)


def evaluate_probe(model, clip, level, source, cohort, arrays, categories,
                   protocol, high_impact, output, seed):
    features, targets, _, metadata = make_probe_rows(
        source, cohort, arrays, categories, level, 'evaluation')
    correction, predictions = correction_from_predictions(
        arrays, model.predict(features), metadata, clip)
    metrics = correction_metrics(arrays, correction)
    clusters = np.asarray([
        source.sample(cohort, position)['iso_week']
        for position in range(len(arrays['positive_sample_index']))
    ])
    metrics['uncertainty'] = uncertainty(metrics, clusters, protocol, seed)
    labeled = np.asarray([
        (source.split, int(sample)) in high_impact
        for sample in arrays['positive_sample_index']
    ])
    selected = np.asarray([
        high_impact.get((source.split, int(sample)), False)
        for sample in arrays['positive_sample_index']
    ])
    if selected.any():
        high = high_impact_metrics(arrays, correction, selected)
        high['labeled_events'] = int(labeled.sum())
        metrics['high_impact_candidate_h1_h6'] = high
    evaluable = np.isfinite(targets)
    if not evaluable.any():
        raise ValueError('Residual-probe validation population has no evaluable targets')
    metrics['probe'] = {
        'rows': int(len(predictions)), 'clip_abs': float(clip),
        'evaluable_target_rows': int(evaluable.sum()),
        'target_mean': float(targets[evaluable].mean()),
        'prediction_mean': float(predictions.mean()),
        'prediction_target_correlation': (
            float(np.corrcoef(predictions[evaluable], targets[evaluable])[0, 1])
            if (np.std(predictions[evaluable]) > 0 and
                np.std(targets[evaluable]) > 0) else None),
    }
    metrics.pop('_per_event')
    write_probe_rows(
        output, source.split, cohort, level, targets, predictions, metadata,
        high_impact)
    return metrics


def gate_decision(validation, oracle, protocol):
    gate = protocol['development_gate']
    primary = validation[gate['primary_probe']]
    full = primary['incident_full']
    matched = primary['incident']
    margin = gate['maximum_global_harm_fraction_of_A_mae'] * full['all']['mae_A']
    checks = {
        'positive_validation_global_point_improvement':
            full['all']['improvement_vs_A'] > 0,
        'validation_global_noninferiority':
            full['uncertainty']['all_improvement_vs_A']['ci_low'] >= -margin,
        'validation_candidate_h1_h6_improvement':
            full['uncertainty']['candidate_h1_h6_improvement_vs_A']['ci_low'] > 0,
        'validation_high_impact_point_improvement':
            matched['high_impact_candidate_h1_h6']['improvement_vs_A'] > 0,
        'nonzero_correction_fraction':
            full['correction']['nonzero_fraction'] >=
            gate['require_nonzero_correction_fraction'],
        'protected_h7_h12_exact_A': full['protected_h7_h12_exact_A'],
        'protected_noncandidate_exact_A': full['protected_noncandidate_exact_A'],
    }
    control_margin = gate['maximum_each_control_H1_H6_harm_fraction_of_A_mae']
    for cohort in ('primary_control', 'secondary_control'):
        current = primary[cohort]
        checks[f'{cohort}_routine_harm_bound'] = (
            current['uncertainty']['candidate_h1_h6_harm_vs_A']['ci_high'] <=
            control_margin * current['candidate_h1_h6']['mae_A'])
    probe_pass = all(checks.values())
    oracle_improvement = oracle['val']['incident_full']['event_node_phase']['all'][
        'relative_improvement_vs_A_percent']
    thresholds = protocol['oracle']['all_node_relative_improvement_percent']
    if oracle_improvement < thresholds['stop_below']:
        recommendation = 'STOP_CONSTRAINED_RESIDUAL_ORACLE_BELOW_0_3_PERCENT'
    elif not probe_pass:
        recommendation = 'SIGNED_RESIDUAL_PROBE_NOT_VALIDATED'
    elif oracle_improvement < thresholds['small_probe_below']:
        recommendation = 'SMALL_BASELINE_ANCHORED_RESIDUAL_STUDY_ALLOWED'
    else:
        recommendation = 'NEURAL_RESIDUAL_EXPERT_STUDY_ALLOWED'
    return {
        'checks': checks, 'probe_gate_passed': probe_pass,
        'primary_oracle_all_node_relative_improvement_percent': oracle_improvement,
        'maximum_global_harm_raw_mae': margin,
        'recommendation': recommendation,
    }


def audit(data_dir, primary_dir, secondary_dir, materialized_dir, placebo_dir,
          sensors_path, protocol_path, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError('v7b output exists; preserve it and use a new directory')
    protocol = load_protocol(protocol_path)
    v7a_summary = verify_inputs(
        data_dir, primary_dir, secondary_dir, materialized_dir, placebo_dir,
        sensors_path, protocol)
    high_values, high_threshold = load_high_impact(placebo_dir)
    high_impact = {key: value >= high_threshold for key, value in high_values.items()}
    sources = {
        split: FeatureSource(
            data_dir, primary_dir, secondary_dir, sensors_path, split,
            protocol['expected_common_triples'][split],
            protocol['expected_positive_samples'][split],
            protocol['expected_sensor_count'])
        for split in SPLITS
    }
    required_categories = {
        tuple(value.split('-', 1))
        for value in protocol['sensor_metadata']['required_unique_categories']
    }
    if any(source.location_categories != required_categories
           for source in sources.values()):
        raise ValueError('v7b report-location category support changed')
    categories = category_schema(sources['train'])
    for split in SPLITS:
        samples = {int(row['positive_sample_index']) for row in sources[split].secondary_rows}
        if samples != {sample for current_split, sample in high_values
                       if current_split == split}:
            raise ValueError('v5c stratification and v7a cohorts differ')
    output.mkdir(parents=True, exist_ok=False)
    oracle = {split: {} for split in SPLITS}
    for split in SPLITS:
        count = protocol['expected_common_triples'][split]
        for cohort in (*COHORTS, 'incident_full'):
            expected_count = (protocol['expected_positive_samples'][split]
                              if cohort == 'incident_full' else count)
            arrays = load_residuals(
                materialized_dir, split, cohort, expected_count,
                protocol['expected_sensor_count'])
            oracle[split][cohort] = oracle_summary(arrays)
    validation = {level: {} for level in LEVELS}
    model_arrays, probe_files = {}, []
    for level_number, level in enumerate(LEVELS):
        train_features, train_targets, train_weights = [], [], []
        incident_targets = None
        for cohort in COHORTS:
            arrays = load_residuals(
                materialized_dir, 'train', cohort,
                protocol['expected_common_triples']['train'],
                protocol['expected_sensor_count'])
            target_mode = 'residual' if cohort == 'incident' else 'zero'
            features, targets, weights, _ = make_probe_rows(
                sources['train'], cohort, arrays, categories, level, target_mode)
            if cohort == 'incident':
                incident_targets = targets
            train_features.append(features)
            train_targets.append(targets)
            train_weights.append(weights)
        clip = float(np.quantile(np.abs(incident_targets), .99))
        if not np.isfinite(clip) or clip <= 0:
            raise ValueError('Train-only residual correction clip is invalid')
        model = fit_weighted_ridge(
            np.concatenate(train_features), np.concatenate(train_targets),
            np.concatenate(train_weights), float(protocol['probe']['ridge_alpha']))
        names = residual_feature_names(categories, level)
        if len(names) != len(model.coefficient):
            raise ValueError('Residual feature names and coefficients differ')
        model_arrays.update({
            f'{level}_feature_names': np.asarray(names),
            f'{level}_feature_mean': model.mean,
            f'{level}_feature_scale': model.scale,
            f'{level}_coefficient': model.coefficient,
            f'{level}_target_mean': np.asarray(model.target_mean),
            f'{level}_ridge_alpha': np.asarray(model.alpha),
            f'{level}_prediction_clip_abs': np.asarray(clip),
        })
        for cohort_number, cohort in enumerate(('incident_full', *COHORTS)):
            expected_count = (protocol['expected_positive_samples']['val']
                              if cohort == 'incident_full' else
                              protocol['expected_common_triples']['val'])
            arrays = load_residuals(
                materialized_dir, 'val', cohort, expected_count,
                protocol['expected_sensor_count'])
            path = output / f'val_{cohort}_{level}_probe.csv'
            validation[level][cohort] = evaluate_probe(
                model, clip, level, sources['val'], cohort, arrays, categories,
                protocol, high_impact, path,
                int(protocol['uncertainty']['seed']) +
                level_number * 10 + cohort_number)
            probe_files.append(path)
    model_path = output / 'probe_models.npz'
    with model_path.open('wb') as stream:
        np.savez_compressed(stream, **model_arrays)
    decision = gate_decision(validation, oracle, protocol)
    summary = {
        'status': 'SIGNED_RESIDUAL_PROBE_COMPLETE', 'scope': protocol['scope'],
        'protocol_id': protocol['protocol_id'],
        'protocol_sha256': sha256(protocol_path), 'main_training_ready': False,
        'test_split_read': False, 'frozen_A_updated': False,
        'neural_expert_trained': False, 'shallow_probe_fit_on_train_only': True,
        'control_future_Y_used_for_probe_fit': False,
        'validation_used_to_tune_probe': False,
        'v5c_label_used_as_probe_input': False,
        'high_impact_train_q75_raw': high_threshold,
        'categorical_schema_fit_on_train': categories,
        'oracle_reported_as_model_performance': False,
        'oracle': oracle, 'validation_probe': validation,
        'development_gate': decision,
        'estimand': protocol['estimand'], 'support': protocol['support'],
        'uncertainty': protocol['uncertainty'],
        'limitations': [
            'Oracle corrections use forecast outcomes and are unattainable diagnostics within only the declared constant/two-phase families.',
            'The primary oracle is not an upper bound on an arbitrary neural residual expert.',
            'The shallow probe is an additive diagnostic and not a trained graph expert.',
            'Matched controls are observational routine placebos, not causal counterfactual outcomes.',
            'Passing gates would authorize only further train/validation development, never test access.',
        ],
        'inputs': {
            'v7a_summary_sha256': sha256(Path(materialized_dir) / 'summary.json'),
            'v7a_protocol_sha256': v7a_summary['protocol_sha256'],
            'v5c_summary_sha256': sha256(Path(placebo_dir) / 'summary.json'),
            'v5c_triple_metrics_sha256': sha256(
                Path(placebo_dir) / 'triple_metrics.csv'),
            'sensors_sha256': sha256(sensors_path),
            'protocol_sha256': sha256(protocol_path),
            'code_sha256': sha256(__file__),
        },
    }
    outputs = [model_path, *probe_files]
    summary['outputs'] = {
        path.name: {'sha256': sha256(path), 'bytes': path.stat().st_size}
        for path in outputs
    }
    summary_path = output / 'summary.json'
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
        encoding='utf-8')
    print(json.dumps({
        'status': summary['status'], 'development_gate': decision,
        'high_impact_train_q75_raw': high_threshold,
    }, indent=2, ensure_ascii=False), flush=True)
    print(f'Saved v7b signed residual probe: {summary_path}', flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--primary-control-dir', type=Path, required=True)
    parser.add_argument('--secondary-control-dir', type=Path, required=True)
    parser.add_argument('--materialized-dir', type=Path, required=True)
    parser.add_argument('--placebo-dir', type=Path, required=True)
    parser.add_argument('--sensors', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, default=Path(__file__).with_name(
        'signed_residual_probe_v7b.json'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    audit(args.data_dir, args.primary_control_dir, args.secondary_control_dir,
          args.materialized_dir, args.placebo_dir, args.sensors, args.protocol,
          args.output)


if __name__ == '__main__':
    main()
