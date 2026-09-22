"""Fit the authorized v8a/v8b routers and materialize frozen v8c routes."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.chronological.audit_expert_benefit import (
    FeatureSource, category_schema, feature_names, feature_vector,
)
from experiments.chronological.audit_incident_impact_identification import (
    build_features, fit_fold, load_protocol as load_v8a_protocol, read_labels,
)
from experiments.chronological.audit_incident_impact_node_localization import (
    build_data, fit_node_model, load_protocol as load_v8b_protocol,
)
from experiments.chronological.audit_matched_controls import sha256
from experiments.chronological.audit_matched_placebo import (
    load_protocol as load_v5c_protocol, verify_inputs as verify_v5c_inputs,
)


COHORTS = ('incident_full', 'incident', 'primary_control', 'secondary_control')
COMMON_COHORTS = COHORTS[1:]
ROUTE_FIELDS = {
    'positive_sample_index', 'candidate_mask', 'event_score', 'event_route',
    'node_score', 'node_route', 'hierarchical_route',
}


def load_protocol(path):
    protocol = json.loads(Path(path).read_text(encoding='utf-8'))
    if (protocol.get('protocol_id') !=
            'contra_v8_hierarchical_impact_router_materialization_v8c' or
            protocol.get('scope') !=
            'full_train_fit_report_time_safe_hierarchical_router_materialization' or
            protocol.get('main_training_ready') is not False):
        raise ValueError('v8c protocol identity or scope changed')
    if [protocol.get(key) for key in ('source_year', 'source_version',
                                      'expected_sensor_count')] != [2023, 8, 496]:
        raise ValueError('v8c source package identity changed')
    if protocol.get('expected_common_samples') != {'train': 3106, 'val': 618} or \
            protocol.get('expected_positive_samples') != {'train': 3604, 'val': 917}:
        raise ValueError('v8c cohort sizes changed')
    if protocol.get('authorization') != {
            'event': {
                'protocol_id': 'contra_v8_observable_incident_impact_identification_v8a',
                'protocol_sha256': '1462d7b2a4d51f947ede97f772d2ae002f008698f1f3d4325f7cadeea16576e6',
                'status': 'OBSERVABLE_IMPACT_IDENTIFICATION_AUDIT_COMPLETE',
                'recommendation': 'OBSERVABLE_IMPACT_GATE_DEVELOPMENT_ALLOWED'},
            'node': {
                'protocol_id': 'contra_v8_observable_incident_impact_node_localization_v8b',
                'protocol_sha256': '318a4701c2dc3754656d5653af9f37a841c5969899e2df4f2bee93d11de08c0c',
                'status': 'OBSERVABLE_IMPACT_NODE_LOCALIZATION_AUDIT_COMPLETE',
                'recommendation': 'PROTECTED_NODE_EXPERT_DEVELOPMENT_ALLOWED'}}:
        raise ValueError('v8c development authorization changed')
    if protocol.get('v5c_context') != {
            'protocol_id': 'contra_v8_matched_placebo_audit_v5c',
            'summary_sha256': 'c443581334f1f0aebece82c910a38c78b1dc29a22565d081f6a583298875571a',
            'triple_metrics_sha256': '192b44e82223a9f943f3b47a5141ea1d19819b7cd0d0c75b0c234d372dcb5bc5'}:
        raise ValueError('v8c v5c context changed')
    if protocol.get('sensor_metadata_sha256') != \
            '682f3cdf75e643f0b37356ab69cbabb27389be5089f41d3b2cbc4bede3332094':
        raise ValueError('v8c sensor metadata changed')
    if protocol.get('fit') != {
            'population': 'all_common_train_triples',
            'cohorts': list(COMMON_COHORTS),
            'event_router': {
                'features': 'v8a_report_time_safe_event_features',
                'label': 'fit_incident_early_excess_q75',
                'control_target': 'exact_zero', 'ridge_alpha': 10.0,
                'route_threshold': 'full_fit_incident_score_q75'},
            'node_router': {
                'features': 'v8b_report_time_safe_event_node_features',
                'label': 'fit_incident_candidate_node_early_excess_q75',
                'control_target': 'exact_zero', 'ridge_alpha': 10.0,
                'route_threshold': 'full_fit_incident_candidate_node_score_q75'}}:
        raise ValueError('v8c full-training fit changed')
    if protocol.get('routing') != {
            'event_route': 'event_score_at_or_above_frozen_threshold',
            'node_route': 'candidate_node_score_at_or_above_frozen_threshold',
            'hierarchical_route': 'event_route_and_node_route_and_candidate_mask',
            'active_horizons_zero_based_half_open': [0, 6],
            'protected_horizons_zero_based_half_open': [6, 12],
            'noncandidate_nodes_protected': True}:
        raise ValueError('v8c hierarchical routing changed')
    if protocol.get('materialization') != {
            'splits': ['train', 'val'], 'cohorts': list(COHORTS),
            'validation_history_slice': [0, 12],
            'validation_future_targets_read': False,
            'atomic_output_publication': True}:
        raise ValueError('v8c materialization design changed')
    required = (
        'router_fit_on_common_train_only', 'validation_features_are_pre_t0_only',
        'validation_labels_prohibited', 'validation_residual_arrays_prohibited',
        'test_split_prohibited', 'future_outcome_features_prohibited',
        'frozen_A_predictions_prohibited', 'expert_training_prohibited',
        'thresholds_frozen_before_validation_materialization')
    if not all(protocol.get('information_boundary', {}).get(key) is True
               for key in required):
        raise ValueError('v8c information boundary changed')
    return protocol


def verify_authorization(directory, expected):
    directory = Path(directory)
    path = directory / 'summary.json'
    summary = json.loads(path.read_text(encoding='utf-8'))
    if (summary.get('protocol_id') != expected['protocol_id'] or
            summary.get('protocol_sha256') != expected['protocol_sha256'] or
            summary.get('status') != expected['status'] or
            summary.get('test_split_read') is not False or
            summary.get('validation_residual_arrays_read') is not False or
            summary.get('development_gate', {}).get('recommendation') !=
            expected['recommendation']):
        raise ValueError(f"Authorization audit differs: {expected['protocol_id']}")
    passed_key = ('impact_identification_gate_passed' if
                  expected['protocol_id'].endswith('_v8a') else
                  'impact_node_localization_gate_passed')
    if summary['development_gate'].get(passed_key) is not True:
        raise ValueError(f"Authorization gate did not pass: {expected['protocol_id']}")
    for filename, metadata in summary.get('outputs', {}).items():
        if sha256(directory / filename) != metadata.get('sha256'):
            raise ValueError(f'Authorization output checksum mismatch: {filename}')
    return {'summary_sha256': sha256(path), 'git_head': summary['environment']['git_head']}


def model_arrays(prefix, model, names, label_threshold, route_threshold):
    if len(names) != len(model.coefficient):
        raise ValueError(f'{prefix} feature names and coefficients differ')
    return {
        f'{prefix}_feature_names': np.asarray(names),
        f'{prefix}_feature_mean': np.asarray(model.mean, dtype=np.float64),
        f'{prefix}_feature_scale': np.asarray(model.scale, dtype=np.float64),
        f'{prefix}_coefficient': np.asarray(model.coefficient, dtype=np.float64),
        f'{prefix}_target_mean': np.asarray(model.target_mean, dtype=np.float64),
        f'{prefix}_ridge_alpha': np.asarray(model.alpha, dtype=np.float64),
        f'{prefix}_label_threshold': np.asarray(label_threshold, dtype=np.float64),
        f'{prefix}_route_threshold': np.asarray(route_threshold, dtype=np.float64),
    }


def validate_route_arrays(arrays, expected_samples, expected_nodes):
    if set(arrays) != ROUTE_FIELDS:
        raise ValueError('v8c route array schema changed')
    sample_shape = (expected_samples,)
    node_shape = (expected_samples, expected_nodes)
    if (arrays['positive_sample_index'].shape != sample_shape or
            arrays['event_score'].shape != sample_shape or
            arrays['event_route'].shape != sample_shape or
            any(arrays[key].shape != node_shape for key in (
                'candidate_mask', 'node_score', 'node_route', 'hierarchical_route'))):
        raise ValueError('v8c route array shapes changed')
    if (arrays['positive_sample_index'].dtype != np.int64 or
            arrays['event_score'].dtype != np.float32 or
            arrays['node_score'].dtype != np.float32 or
            any(arrays[key].dtype != np.bool_ for key in (
                'candidate_mask', 'event_route', 'node_route', 'hierarchical_route'))):
        raise ValueError('v8c route array dtypes changed')
    candidate = arrays['candidate_mask']
    expected_hierarchical = (
        arrays['event_route'][:, None] & arrays['node_route'] & candidate)
    if (len(set(arrays['positive_sample_index'].tolist())) != expected_samples or
            not candidate.any(axis=1).all() or
            np.any(arrays['node_route'] & ~candidate) or
            np.any(arrays['node_score'][~candidate] != 0) or
            not np.array_equal(arrays['hierarchical_route'], expected_hierarchical) or
            not np.isfinite(arrays['event_score']).all() or
            not np.isfinite(arrays['node_score']).all()):
        raise ValueError('v8c route hard support or identity changed')


def materialize_cohort(source, cohort, expected_samples, expected_nodes, categories,
                       event_model, event_threshold, node_model, node_threshold):
    sample_ids = np.empty(expected_samples, dtype=np.int64)
    candidates = np.zeros((expected_samples, expected_nodes), dtype=bool)
    event_scores = np.empty(expected_samples, dtype=np.float32)
    node_scores = np.zeros((expected_samples, expected_nodes), dtype=np.float32)
    for position in range(expected_samples):
        sample = source.sample(cohort, position)
        if (sample['freeway'] not in categories['freeway'] or
                sample['direction'] not in categories['direction']):
            raise ValueError('Materialized route has an unseen location category')
        candidate = np.asarray(sample['candidate'], dtype=bool)
        nodes = np.flatnonzero(candidate)
        sample_ids[position] = int(sample['sample'])
        candidates[position] = candidate
        event_feature = feature_vector(sample, categories)[None]
        event_scores[position] = np.clip(event_model.predict(event_feature)[0], 0., 1.)
        node_features = np.stack([
            feature_vector(sample, categories, int(node)) for node in nodes])
        node_scores[position, nodes] = np.clip(
            node_model.predict(node_features), 0., 1.).astype(np.float32)
    event_routes = event_scores >= float(event_threshold)
    node_routes = candidates & (node_scores >= float(node_threshold))
    arrays = {
        'positive_sample_index': sample_ids,
        'candidate_mask': candidates,
        'event_score': event_scores,
        'event_route': event_routes,
        'node_score': node_scores,
        'node_route': node_routes,
        'hierarchical_route': event_routes[:, None] & node_routes & candidates,
    }
    validate_route_arrays(arrays, expected_samples, expected_nodes)
    return arrays


def route_summary(arrays):
    candidate = arrays['candidate_mask']
    routed = arrays['hierarchical_route']
    return {
        'events': int(len(arrays['event_route'])),
        'candidate_nodes': int(candidate.sum()),
        'event_route_fraction': float(arrays['event_route'].mean()),
        'node_route_fraction_on_candidates': float(arrays['node_route'][candidate].mean()),
        'hierarchical_route_fraction_on_candidates': float(routed[candidate].mean()),
        'hierarchical_routed_nodes': int(routed.sum()),
    }


def materialize(data_dir, primary_dir, secondary_dir, placebo_dir, sensors_path,
                event_audit_dir, node_audit_dir, protocol_path, output):
    output = Path(output)
    partial = output.with_name(output.name + '.partial')
    if output.exists() or partial.exists():
        raise FileExistsError('v8c output or partial output exists; use a new directory')
    protocol = load_protocol(protocol_path)
    v8a_path = Path(__file__).with_name('incident_impact_identification_v8a.json')
    v8b_path = Path(__file__).with_name('incident_impact_node_localization_v8b.json')
    v8a = load_v8a_protocol(v8a_path)
    v8b = load_v8b_protocol(v8b_path)
    if (sha256(v8a_path) != protocol['authorization']['event']['protocol_sha256'] or
            sha256(v8b_path) != protocol['authorization']['node']['protocol_sha256'] or
            v8a['protocol_id'] != protocol['authorization']['event']['protocol_id'] or
            v8b['protocol_id'] != protocol['authorization']['node']['protocol_id']):
        raise ValueError('v8c authorization protocols differ')
    authorizations = {
        'event': verify_authorization(
            event_audit_dir, protocol['authorization']['event']),
        'node': verify_authorization(
            node_audit_dir, protocol['authorization']['node']),
    }
    v5c_path = Path(__file__).with_name('matched_placebo_audit_v5c.json')
    v5c = load_v5c_protocol(v5c_path)
    if (v5c['protocol_id'] != protocol['v5c_context']['protocol_id'] or
            sha256(Path(placebo_dir) / 'summary.json') !=
            protocol['v5c_context']['summary_sha256'] or
            sha256(Path(placebo_dir) / 'triple_metrics.csv') !=
            protocol['v5c_context']['triple_metrics_sha256']):
        raise ValueError('v5c inputs differ from frozen v8c context')
    if sha256(sensors_path) != protocol['sensor_metadata_sha256']:
        raise ValueError('Sensor metadata differs from frozen v8c context')
    input_hashes = verify_v5c_inputs(data_dir, primary_dir, secondary_dir, v5c)

    common_train = protocol['expected_common_samples']['train']
    nodes = protocol['expected_sensor_count']
    samples, _, event_excess = read_labels(placebo_dir, common_train, 35)
    event_features, categories, _ = build_features(
        data_dir, primary_dir, secondary_dir, sensors_path, samples, nodes)
    fit_events = np.arange(common_train, dtype=np.int64)
    event_label_threshold = float(np.quantile(event_excess, .75))
    event_labels = (event_excess >= event_label_threshold).astype(np.int64)
    event_model, event_route_threshold = fit_fold(
        event_features, event_labels, fit_events,
        protocol['fit']['event_router']['ridge_alpha'])

    scaler = json.loads((Path(data_dir) / 'scaler.json').read_text(encoding='utf-8'))
    (_, node_categories, node_features, _, node_excess_values,
     event_indices, _) = build_data(
         data_dir, primary_dir, secondary_dir, sensors_path, samples,
         nodes, float(scaler['std']))
    if node_categories != categories:
        raise ValueError('Event and node categorical schemas differ')
    node_label_threshold = float(np.quantile(node_excess_values, .75))
    node_labels = (node_excess_values >= node_label_threshold).astype(np.int64)
    node_model, node_route_threshold = fit_node_model(
        node_features, node_labels, event_indices, fit_events,
        protocol['fit']['node_router']['ridge_alpha'])

    models = {}
    models.update(model_arrays(
        'event', event_model, feature_names(categories, 'event'),
        event_label_threshold, event_route_threshold))
    models.update(model_arrays(
        'node', node_model, feature_names(categories, 'event_node'),
        node_label_threshold, node_route_threshold))
    partial.mkdir(parents=True, exist_ok=False)
    model_path = partial / 'router_models.npz'
    with model_path.open('wb') as stream:
        np.savez_compressed(stream, **models)

    results, output_metadata = {}, {}
    for split in protocol['materialization']['splits']:
        source = FeatureSource(
            data_dir, primary_dir, secondary_dir, sensors_path, split,
            protocol['expected_common_samples'][split],
            protocol['expected_positive_samples'][split], nodes)
        results[split] = {}
        for cohort in COHORTS:
            count = (protocol['expected_positive_samples'][split]
                     if cohort == 'incident_full' else
                     protocol['expected_common_samples'][split])
            arrays = materialize_cohort(
                source, cohort, count, nodes, categories,
                event_model, event_route_threshold, node_model, node_route_threshold)
            path = partial / f'{split}_{cohort}_routes.npz'
            with path.open('wb') as stream:
                np.savez_compressed(stream, **arrays)
            results[split][cohort] = route_summary(arrays)
            output_metadata[path.name] = {
                'sha256': sha256(path), 'bytes': path.stat().st_size,
                'samples': count,
            }
    output_metadata[model_path.name] = {
        'sha256': sha256(model_path), 'bytes': model_path.stat().st_size}
    summary = {
        'status': 'HIERARCHICAL_IMPACT_ROUTER_MATERIALIZATION_COMPLETE',
        'scope': protocol['scope'], 'protocol_id': protocol['protocol_id'],
        'protocol_sha256': sha256(protocol_path), 'main_training_ready': False,
        'protected_expert_development_ready': True,
        'router_fit_on_common_train_only': True,
        'validation_features_are_pre_t0_only': True,
        'validation_labels_read': False, 'validation_residual_arrays_read': False,
        'validation_future_targets_read': False, 'test_split_read': False,
        'expert_training_performed': False, 'frozen_A_predictions_read': False,
        'categories': categories,
        'fit': {
            'common_train_events': common_train,
            'event_label_threshold': event_label_threshold,
            'event_route_threshold': event_route_threshold,
            'event_features': len(event_model.coefficient),
            'node_rows': int(len(node_excess_values)),
            'node_label_threshold': node_label_threshold,
            'node_route_threshold': node_route_threshold,
            'node_features': len(node_model.coefficient),
        },
        'routing': protocol['routing'], 'results': results,
        'authorizations': authorizations,
        'inputs': {
            'v5c_summary_sha256': protocol['v5c_context']['summary_sha256'],
            'v5c_triple_metrics_sha256': protocol['v5c_context']['triple_metrics_sha256'],
            'positive_inputs': input_hashes[0], 'primary_inputs': input_hashes[1],
            'secondary_inputs': input_hashes[2],
            'protocol_sha256': sha256(protocol_path), 'code_sha256': sha256(__file__),
        },
        'outputs': output_metadata,
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
        'status': summary['status'], 'fit': summary['fit'],
        'validation_incident_full': results['val']['incident_full'],
    }, ensure_ascii=False, indent=2), flush=True)
    print(f'Saved v8c hierarchical routes: {output / "summary.json"}', flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--primary-control-dir', type=Path, required=True)
    parser.add_argument('--secondary-control-dir', type=Path, required=True)
    parser.add_argument('--placebo-dir', type=Path, required=True)
    parser.add_argument('--sensors', type=Path, required=True)
    parser.add_argument('--event-audit-dir', type=Path, required=True)
    parser.add_argument('--node-audit-dir', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, default=Path(__file__).with_name(
        'impact_router_materialization_v8c.json'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    materialize(
        args.data_dir, args.primary_control_dir, args.secondary_control_dir,
        args.placebo_dir, args.sensors, args.event_audit_dir,
        args.node_audit_dir, args.protocol, args.output)


if __name__ == '__main__':
    main()
