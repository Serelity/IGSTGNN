"""Train and audit the minimal v9a hard-routed residual expert."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.chronological.audit_expert_benefit import (
    FeatureSource, category_schema, cluster_bootstrap_mean, fit_weighted_ridge,
)
from experiments.chronological.audit_matched_controls import sha256, write_csv
from experiments.chronological.audit_signed_residual_probe import (
    PHASES, correction_from_predictions, correction_metrics,
    evaluation_target_for,
    load_protocol as load_v7b_protocol, load_residuals, oracle_correction,
    residual_feature_names, residual_feature_vector, target_for,
    verify_inputs as verify_v7a_inputs,
)


SPLITS = ('train', 'val')
COHORTS = ('incident_full', 'incident', 'primary_control', 'secondary_control')
TRAIN_COHORTS = ('incident_full', 'primary_control', 'secondary_control')
ROUTE_FIELDS = {
    'positive_sample_index', 'candidate_mask', 'event_score', 'event_route',
    'node_score', 'node_route', 'hierarchical_route',
}
RESIDUAL_FIELDS = {
    'signed_residual', 'baseline_prediction', 'valid', 'candidate_mask',
    'positive_sample_index', 'baseline_all_absolute_sum',
    'baseline_all_valid_count',
}


def load_protocol(path):
    protocol = json.loads(Path(path).read_text(encoding='utf-8'))
    if (protocol.get('protocol_id') != 'contra_v8_protected_residual_expert_v9a' or
            protocol.get('scope') !=
            'frozen_hierarchical_router_shallow_residual_expert_validation' or
            protocol.get('main_training_ready') is not False):
        raise ValueError('v9a protocol identity or scope changed')
    if [protocol.get(key) for key in ('source_year', 'source_version',
                                      'expected_sensor_count')] != [2023, 8, 496]:
        raise ValueError('v9a source identity changed')
    if protocol.get('expected_common_samples') != {'train': 3106, 'val': 618} or \
            protocol.get('expected_positive_samples') != {'train': 3604, 'val': 917}:
        raise ValueError('v9a cohort sizes changed')
    if protocol.get('v7a_input') != {
            'protocol_id': 'contra_v8_signed_residual_materialize_v7a',
            'protocol_sha256': 'c3d547679b9c6c97ad29720578ddc46a2460d7e9ba336114c59aba429ac03541',
            'status': 'SIGNED_RESIDUAL_MATERIALIZATION_COMPLETE'}:
        raise ValueError('v9a v7a input changed')
    if protocol.get('v8c_input') != {
            'protocol_id': 'contra_v8_hierarchical_impact_router_materialization_v8c',
            'protocol_sha256': '6817e3636ac0886c1dcbab323a22ef9137ff856354716ec723a0d31007a1f114',
            'status': 'HIERARCHICAL_IMPACT_ROUTER_MATERIALIZATION_COMPLETE',
            'protected_expert_development_ready': True}:
        raise ValueError('v9a v8c input changed')
    if protocol.get('estimand') != {
            'baseline': 'frozen_A_incident_on',
            'signed_residual': 'target_minus_frozen_A_prediction',
            'corrected_prediction':
                'frozen_A_prediction_plus_hard_routed_expert_correction',
            'positive_improvement':
                'absolute_error_A_minus_absolute_error_corrected',
            'causal_effect_claimed': False}:
        raise ValueError('v9a estimand changed')
    if protocol.get('support') != {
            'route_source': 'frozen_v8c_hierarchical_route',
            'active_horizons_zero_based_half_open': [0, 6],
            'phase_slices_zero_based_half_open': [[0, 3], [3, 6]],
            'protected_horizons_zero_based_half_open': [6, 12],
            'protected_noncandidate_nodes': True,
            'protected_unrouted_candidate_nodes': True,
            'correction_exactly_zero_outside_hierarchical_route': True}:
        raise ValueError('v9a hard support changed')
    if protocol.get('expert') != {
            'family': 'weighted_ridge_event_node_phase_residual',
            'ridge_alpha': 10.0,
            'training_cohorts': list(TRAIN_COHORTS),
            'incident_target': 'median_signed_residual_per_routed_node_phase',
            'control_target': 'exact_zero_without_reading_control_future_Y',
            'row_weighting': 'equal_cohort_and_equal_routed_event_weight',
            'prediction_clip': 'symmetric_train_incident_target_abs_q99',
            'features':
                'v7b_report_time_event_node_phase_plus_frozen_A_prediction',
            'route_threshold_tuning': 'prohibited'}:
        raise ValueError('v9a expert design changed')
    if protocol.get('oracle') != {
            'family': 'route_conditioned_event_node_phase_median',
            'uses_same_event_future_outcomes': True,
            'reported_as_model_performance': False}:
        raise ValueError('v9a oracle changed')
    if protocol.get('uncertainty') != {
            'method': 'positive_incident_iso_week_cluster_bootstrap',
            'draws': 2000, 'confidence_level': 0.95, 'seed': 2025}:
        raise ValueError('v9a uncertainty changed')
    if protocol.get('development_gate') != {
            'require_full_positive_global_point_improvement': True,
            'maximum_full_positive_global_harm_fraction_of_A_mae': 0.001,
            'require_full_positive_routed_improvement_ci_lower_above_zero': True,
            'require_common_incident_routed_point_improvement': True,
            'maximum_each_control_routed_harm_fraction_of_A_mae': 0.005,
            'minimum_nonzero_correction_fraction_on_routed_support': 0.05,
            'require_exact_A_outside_hierarchical_route': True}:
        raise ValueError('v9a development gate changed')
    required = (
        'train_and_validation_only', 'test_split_prohibited',
        'frozen_A_not_updated', 'v8c_router_and_thresholds_frozen',
        'validation_routes_frozen_before_expert_fit',
        'validation_may_not_tune_features_alpha_clip_or_gate',
        'control_future_Y_prohibited_from_expert_fit',
        'validation_future_Y_used_only_for_evaluation',
        'neural_expert_training_prohibited')
    if not all(protocol.get('information_boundary', {}).get(key) is True
               for key in required):
        raise ValueError('v9a information boundary changed')
    return protocol


def verify_v8c(router_dir, protocol):
    router_dir = Path(router_dir)
    summary_path = router_dir / 'summary.json'
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    expected = protocol['v8c_input']
    if (summary.get('status') != expected['status'] or
            summary.get('protocol_id') != expected['protocol_id'] or
            summary.get('protocol_sha256') != expected['protocol_sha256'] or
            summary.get('protected_expert_development_ready') is not True or
            summary.get('router_fit_on_common_train_only') is not True or
            summary.get('validation_labels_read') is not False or
            summary.get('validation_residual_arrays_read') is not False or
            summary.get('validation_future_targets_read') is not False or
            summary.get('test_split_read') is not False or
            summary.get('expert_training_performed') is not False):
        raise ValueError('v8c router materialization identity or boundary changed')
    expected_outputs = {'router_models.npz'} | {
        f'{split}_{cohort}_routes.npz'
        for split in SPLITS for cohort in COHORTS
    }
    if set(summary.get('outputs', {})) != expected_outputs:
        raise ValueError('v8c router output set changed')
    for filename, metadata in summary['outputs'].items():
        if sha256(router_dir / filename) != metadata['sha256']:
            raise ValueError(f'v8c router checksum mismatch: {filename}')
    return summary


def load_routes(router_dir, split, cohort, expected_samples, expected_nodes):
    path = Path(router_dir) / f'{split}_{cohort}_routes.npz'
    with np.load(path, allow_pickle=False) as stored:
        if set(stored.files) != ROUTE_FIELDS:
            raise ValueError(f'Unexpected v8c route schema: {path.name}')
        routes = {key: stored[key].copy() for key in stored.files}
    sample_shape = (expected_samples,)
    node_shape = (expected_samples, expected_nodes)
    if (routes['positive_sample_index'].shape != sample_shape or
            routes['event_score'].shape != sample_shape or
            routes['event_route'].shape != sample_shape or
            any(routes[key].shape != node_shape for key in (
                'candidate_mask', 'node_score', 'node_route',
                'hierarchical_route')) or
            routes['positive_sample_index'].dtype != np.int64 or
            routes['event_score'].dtype != np.float32 or
            routes['node_score'].dtype != np.float32 or
            any(routes[key].dtype != np.bool_ for key in (
                'candidate_mask', 'event_route', 'node_route',
                'hierarchical_route'))):
        raise ValueError(f'Invalid v8c route arrays: {path.name}')
    expected = (routes['event_route'][:, None] & routes['node_route'] &
                routes['candidate_mask'])
    if (not np.array_equal(routes['hierarchical_route'], expected) or
            np.any(routes['node_route'] & ~routes['candidate_mask']) or
            not np.isfinite(routes['event_score']).all() or
            not np.isfinite(routes['node_score']).all()):
        raise ValueError(f'Invalid v8c hard support: {path.name}')
    return routes


def verify_alignment(arrays, routes):
    if (not np.array_equal(
            arrays['positive_sample_index'], routes['positive_sample_index']) or
            not np.array_equal(arrays['candidate_mask'], routes['candidate_mask'])):
        raise ValueError('v7a residuals and v8c routes are not aligned')


def load_control_training_inputs(materialized_dir, split, cohort,
                                 expected_samples, expected_nodes):
    if split != 'train' or cohort not in ('primary_control', 'secondary_control'):
        raise ValueError('Control-only loader is restricted to training controls')
    path = Path(materialized_dir) / f'{split}_{cohort}_signed_residuals.npz'
    with np.load(path, allow_pickle=False) as stored:
        if set(stored.files) != RESIDUAL_FIELDS:
            raise ValueError(f'Unexpected v7a array schema: {path.name}')
        arrays = {
            key: stored[key].copy()
            for key in ('baseline_prediction', 'candidate_mask',
                        'positive_sample_index')
        }
    shape = (expected_samples, 6, expected_nodes, 1)
    if (arrays['baseline_prediction'].shape != shape or
            arrays['candidate_mask'].shape != (expected_samples, expected_nodes) or
            arrays['positive_sample_index'].shape != (expected_samples,) or
            arrays['baseline_prediction'].dtype != np.float32 or
            arrays['candidate_mask'].dtype != np.bool_ or
            arrays['positive_sample_index'].dtype != np.int64 or
            not np.isfinite(arrays['baseline_prediction']).all()):
        raise ValueError(f'Invalid control training inputs: {path.name}')
    return arrays


def make_expert_rows(source, cohort, arrays, routes, categories, target_mode):
    if target_mode not in ('residual', 'zero', 'evaluation'):
        raise ValueError('Unknown protected-expert target mode')
    verify_alignment(arrays, routes)
    features, targets, weights, metadata = [], [], [], []
    routed_events = 0
    for position in range(len(arrays['positive_sample_index'])):
        sample = source.sample(cohort, position)
        if (sample['sample'] != int(arrays['positive_sample_index'][position]) or
                not np.array_equal(sample['candidate'], arrays['candidate_mask'][position])):
            raise ValueError('Protected-expert features and residuals are not aligned')
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
            metadata.append((position, node, phase, sample))
    if not features or routed_events < 1:
        raise ValueError(f'Protected-expert cohort has no routed rows: {cohort}')
    return (np.asarray(features, dtype=np.float64),
            np.asarray(targets, dtype=np.float64),
            np.asarray(weights, dtype=np.float64), metadata, routed_events)


def protected_metrics(arrays, routes, correction):
    verify_alignment(arrays, routes)
    correction = np.asarray(correction, dtype=np.float64)
    residual = arrays['signed_residual'].astype(np.float64)
    if correction.shape != residual.shape or not np.isfinite(correction).all():
        raise ValueError('Protected correction and residual shapes differ')
    route_geometry = np.broadcast_to(
        routes['hierarchical_route'][:, None, :, None], residual.shape)
    if np.any(correction[~route_geometry] != 0):
        raise ValueError('Expert correction escaped the frozen hierarchical route')
    metrics = correction_metrics(arrays, correction)
    active = route_geometry & arrays['valid']
    counts = active.sum(axis=(1, 2, 3))
    selected = counts > 0
    if not selected.any():
        raise ValueError('Validation cohort has no valid routed support')
    baseline_sum = np.where(active, np.abs(residual), 0.).sum(axis=(1, 2, 3))
    corrected_sum = np.where(
        active, np.abs(residual - correction), 0.).sum(axis=(1, 2, 3))
    total_count = int(counts.sum())
    baseline_mae = float(baseline_sum.sum() / total_count)
    corrected_mae = float(corrected_sum.sum() / total_count)
    nonzero = np.abs(correction[active]) > 1e-6
    metrics['routed_candidate_h1_h6'] = {
        'routed_events': int(selected.sum()), 'valid_cells': total_count,
        'mae_A': baseline_mae, 'mae_corrected': corrected_mae,
        'improvement_vs_A': baseline_mae - corrected_mae,
        'relative_improvement_vs_A_percent':
            100 * (baseline_mae - corrected_mae) / baseline_mae,
        'nonzero_correction_fraction': float(nonzero.mean()),
    }
    metrics['protected_outside_hierarchical_route_exact_A'] = True
    metrics['_per_event']['routed_improvement_vs_A'] = np.divide(
        baseline_sum - corrected_sum, counts,
        out=np.full(len(counts), np.nan, dtype=np.float64), where=selected)
    return metrics


def route_conditioned_oracle(arrays, routes):
    correction = oracle_correction(arrays, 'event_node_phase')
    support = np.broadcast_to(
        routes['hierarchical_route'][:, None, :, None], correction.shape)
    correction = np.where(support, correction, 0.)
    metrics = protected_metrics(arrays, routes, correction)
    metrics.pop('_per_event')
    return metrics


def uncertainty(metrics, weeks, protocol, seed):
    weeks = np.asarray(weeks)
    per_event = metrics['_per_event']
    global_values = np.asarray(per_event['all_improvement_vs_A'], dtype=np.float64)
    routed_values = np.asarray(per_event['routed_improvement_vs_A'], dtype=np.float64)
    selected = np.isfinite(routed_values)
    if len(global_values) != len(weeks) or not selected.any():
        raise ValueError('Protected-expert uncertainty rows are not aligned')
    draws = int(protocol['uncertainty']['draws'])
    confidence = float(protocol['uncertainty']['confidence_level'])
    global_low, global_high = cluster_bootstrap_mean(
        global_values, weeks, draws, confidence, seed)
    routed_matrix = np.column_stack([
        routed_values[selected], -routed_values[selected]])
    routed_low, routed_high = cluster_bootstrap_mean(
        routed_matrix, weeks[selected], draws, confidence, seed + 1)
    return {
        'all_improvement_vs_A': {
            'mean': float(global_values.mean()),
            'ci_low': float(global_low[0]), 'ci_high': float(global_high[0])},
        'routed_improvement_vs_A': {
            'mean': float(routed_values[selected].mean()),
            'ci_low': float(routed_low[0]), 'ci_high': float(routed_high[0])},
        'routed_harm_vs_A': {
            'mean': float(-routed_values[selected].mean()),
            'ci_low': float(routed_low[1]), 'ci_high': float(routed_high[1])},
    }


def write_expert_rows(path, cohort, targets, predictions, metadata):
    rows = []
    for target, prediction, (_, node, phase, sample) in zip(
            targets, predictions, metadata):
        rows.append({
            'cohort': cohort, 'positive_sample_index': sample['sample'],
            'positive_iso_week': sample['iso_week'], 'node_index': node,
            'phase': phase, 'target_correction': target,
            'predicted_correction': prediction,
        })
    write_csv(path, rows, [
        'cohort', 'positive_sample_index', 'positive_iso_week', 'node_index',
        'phase', 'target_correction', 'predicted_correction'])


def evaluate(model, clip, source, cohort, arrays, routes, categories,
             protocol, output, seed):
    features, targets, _, metadata, routed_events = make_expert_rows(
        source, cohort, arrays, routes, categories, 'evaluation')
    correction, predictions = correction_from_predictions(
        arrays, model.predict(features), metadata, clip)
    metrics = protected_metrics(arrays, routes, correction)
    weeks = np.asarray([
        source.sample(cohort, position)['iso_week']
        for position in range(len(arrays['positive_sample_index']))])
    metrics['uncertainty'] = uncertainty(metrics, weeks, protocol, seed)
    evaluable = np.isfinite(targets)
    metrics['expert'] = {
        'rows': int(len(predictions)), 'routed_events': routed_events,
        'evaluable_target_rows': int(evaluable.sum()), 'clip_abs': float(clip),
        'target_mean': float(targets[evaluable].mean()),
        'prediction_mean': float(predictions.mean()),
        'prediction_abs_mean': float(np.abs(predictions).mean()),
        'prediction_target_correlation': (
            float(np.corrcoef(predictions[evaluable], targets[evaluable])[0, 1])
            if (evaluable.any() and np.std(predictions[evaluable]) > 0 and
                np.std(targets[evaluable]) > 0) else None),
    }
    metrics.pop('_per_event')
    write_expert_rows(output, cohort, targets, predictions, metadata)
    return metrics


def gate_decision(validation, protocol):
    gate = protocol['development_gate']
    full = validation['incident_full']
    common = validation['incident']
    margin = (gate['maximum_full_positive_global_harm_fraction_of_A_mae'] *
              full['all']['mae_A'])
    checks = {
        'full_positive_global_point_improvement':
            full['all']['improvement_vs_A'] > 0,
        'full_positive_global_noninferiority':
            full['uncertainty']['all_improvement_vs_A']['ci_low'] >= -margin,
        'full_positive_routed_improvement':
            full['uncertainty']['routed_improvement_vs_A']['ci_low'] > 0,
        'common_incident_routed_point_improvement':
            common['routed_candidate_h1_h6']['improvement_vs_A'] > 0,
        'nonzero_correction_fraction':
            full['routed_candidate_h1_h6']['nonzero_correction_fraction'] >=
            gate['minimum_nonzero_correction_fraction_on_routed_support'],
        'protected_outside_hierarchical_route_exact_A':
            full['protected_outside_hierarchical_route_exact_A'],
        'protected_h7_h12_exact_A': full['protected_h7_h12_exact_A'],
        'protected_noncandidate_exact_A': full['protected_noncandidate_exact_A'],
    }
    control_fraction = gate['maximum_each_control_routed_harm_fraction_of_A_mae']
    for cohort in ('primary_control', 'secondary_control'):
        current = validation[cohort]
        checks[f'{cohort}_routed_harm_bound'] = (
            current['uncertainty']['routed_harm_vs_A']['ci_high'] <=
            control_fraction * current['routed_candidate_h1_h6']['mae_A'])
    passed = all(checks.values())
    return {
        'checks': checks, 'protected_residual_expert_gate_passed': passed,
        'maximum_global_harm_raw_mae': margin,
        'recommendation': (
            'PROTECTED_NEURAL_RESIDUAL_EXPERT_DEVELOPMENT_ALLOWED' if passed else
            'STOP_SHALLOW_PROTECTED_RESIDUAL_EXPERT_NOT_VALIDATED'),
    }


def audit(data_dir, primary_dir, secondary_dir, residual_dir, router_dir,
          placebo_dir, sensors_path, protocol_path, output):
    output = Path(output)
    partial = output.with_name(output.name + '.partial')
    if output.exists() or partial.exists():
        raise FileExistsError('v9a output or partial output exists; use a new directory')
    protocol = load_protocol(protocol_path)
    v7b_path = Path(__file__).with_name('signed_residual_probe_v7b.json')
    v7b = load_v7b_protocol(v7b_path)
    v7a_summary = verify_v7a_inputs(
        data_dir, primary_dir, secondary_dir, residual_dir, placebo_dir,
        sensors_path, v7b)
    if (v7a_summary['protocol_id'] != protocol['v7a_input']['protocol_id'] or
            v7a_summary['protocol_sha256'] !=
            protocol['v7a_input']['protocol_sha256'] or
            v7a_summary['status'] != protocol['v7a_input']['status']):
        raise ValueError('v7a residual materialization differs from v9a')
    v8c_summary = verify_v8c(router_dir, protocol)
    sources = {
        split: FeatureSource(
            data_dir, primary_dir, secondary_dir, sensors_path, split,
            protocol['expected_common_samples'][split],
            protocol['expected_positive_samples'][split],
            protocol['expected_sensor_count'])
        for split in SPLITS
    }
    categories = category_schema(sources['train'])
    train_features, train_targets, train_weights = [], [], []
    incident_targets = None
    training_summary = {}
    for cohort in TRAIN_COHORTS:
        count = (protocol['expected_positive_samples']['train']
                 if cohort == 'incident_full' else
                 protocol['expected_common_samples']['train'])
        if cohort == 'incident_full':
            arrays = load_residuals(
                residual_dir, 'train', cohort, count,
                protocol['expected_sensor_count'])
        else:
            arrays = load_control_training_inputs(
                residual_dir, 'train', cohort, count,
                protocol['expected_sensor_count'])
        routes = load_routes(
            router_dir, 'train', cohort, count,
            protocol['expected_sensor_count'])
        target_mode = 'residual' if cohort == 'incident_full' else 'zero'
        features, targets, weights, _, routed_events = make_expert_rows(
            sources['train'], cohort, arrays, routes, categories, target_mode)
        if cohort == 'incident_full':
            incident_targets = targets
        weights = weights / weights.sum()
        train_features.append(features)
        train_targets.append(targets)
        train_weights.append(weights)
        training_summary[cohort] = {
            'events': count, 'routed_events': routed_events,
            'rows': int(len(features)), 'weight_sum': float(weights.sum()),
        }
    clip = float(np.quantile(np.abs(incident_targets), .99))
    if not np.isfinite(clip) or clip <= 0:
        raise ValueError('v9a train-only prediction clip is invalid')
    model = fit_weighted_ridge(
        np.concatenate(train_features), np.concatenate(train_targets),
        np.concatenate(train_weights), float(protocol['expert']['ridge_alpha']))
    names = residual_feature_names(categories, 'event_node_phase')
    if len(names) != len(model.coefficient):
        raise ValueError('v9a feature names and coefficients differ')
    del train_features, train_targets, train_weights

    partial.mkdir(parents=True, exist_ok=False)
    model_path = partial / 'protected_expert_model.npz'
    with model_path.open('wb') as stream:
        np.savez_compressed(
            stream, feature_names=np.asarray(names), feature_mean=model.mean,
            feature_scale=model.scale, coefficient=model.coefficient,
            target_mean=np.asarray(model.target_mean),
            ridge_alpha=np.asarray(model.alpha),
            prediction_clip_abs=np.asarray(clip))
    validation, oracle, row_files = {}, {}, []
    for cohort_number, cohort in enumerate(COHORTS):
        count = (protocol['expected_positive_samples']['val']
                 if cohort == 'incident_full' else
                 protocol['expected_common_samples']['val'])
        arrays = load_residuals(
            residual_dir, 'val', cohort, count,
            protocol['expected_sensor_count'])
        routes = load_routes(
            router_dir, 'val', cohort, count,
            protocol['expected_sensor_count'])
        row_path = partial / f'val_{cohort}_expert_rows.csv'
        validation[cohort] = evaluate(
            model, clip, sources['val'], cohort, arrays, routes, categories,
            protocol, row_path,
            int(protocol['uncertainty']['seed']) + cohort_number * 10)
        oracle[cohort] = route_conditioned_oracle(arrays, routes)
        row_files.append(row_path)
        print(json.dumps({
            'cohort': cohort,
            'global_improvement': validation[cohort]['all']['improvement_vs_A'],
            'routed_improvement': validation[cohort][
                'routed_candidate_h1_h6']['improvement_vs_A'],
        }), flush=True)
    gate = gate_decision(validation, protocol)
    outputs = [model_path, *row_files]
    summary = {
        'status': 'PROTECTED_RESIDUAL_EXPERT_AUDIT_COMPLETE',
        'scope': protocol['scope'], 'protocol_id': protocol['protocol_id'],
        'protocol_sha256': sha256(protocol_path), 'main_training_ready': False,
        'test_split_read': False, 'frozen_A_updated': False,
        'v8c_router_updated': False, 'validation_used_to_tune_expert': False,
        'control_future_Y_used_for_expert_fit': False,
        'neural_expert_trained': False, 'shallow_expert_fit_on_train_only': True,
        'training': training_summary,
        'fit': {'prediction_clip_abs': clip, 'features': len(names)},
        'validation': validation, 'route_conditioned_oracle': oracle,
        'development_gate': gate,
        'estimand': protocol['estimand'], 'support': protocol['support'],
        'inputs': {
            'v7a_summary_sha256': sha256(Path(residual_dir) / 'summary.json'),
            'v8c_summary_sha256': sha256(Path(router_dir) / 'summary.json'),
            'v8c_git_head': v8c_summary['environment']['git_head'],
            'protocol_sha256': sha256(protocol_path), 'code_sha256': sha256(__file__),
        },
        'outputs': {
            path.name: {'sha256': sha256(path), 'bytes': path.stat().st_size}
            for path in outputs},
        'environment': {
            'python_version': sys.version, 'numpy_version': np.__version__,
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
        'full_positive': validation['incident_full'],
    }, ensure_ascii=False, indent=2), flush=True)
    print(f'Saved v9a protected residual expert: {output / "summary.json"}', flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--primary-control-dir', type=Path, required=True)
    parser.add_argument('--secondary-control-dir', type=Path, required=True)
    parser.add_argument('--residual-dir', type=Path, required=True)
    parser.add_argument('--router-dir', type=Path, required=True)
    parser.add_argument('--placebo-dir', type=Path, required=True)
    parser.add_argument('--sensors', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, default=Path(__file__).with_name(
        'protected_residual_expert_v9a.json'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    audit(
        args.data_dir, args.primary_control_dir, args.secondary_control_dir,
        args.residual_dir, args.router_dir, args.placebo_dir, args.sensors,
        args.protocol, args.output)


if __name__ == '__main__':
    main()
