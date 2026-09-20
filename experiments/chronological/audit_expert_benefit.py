"""Audit frozen-A branch oracles and report-time-safe shallow routing."""

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.chronological.audit_matched_controls import (
    parse_direction, parse_freeway, read_csv, sha256, write_csv,
)


SPLITS = ('train', 'val')
COHORTS = ('incident', 'primary_control', 'secondary_control')
LEVELS = ('event', 'event_node')


def load_protocol(path):
    protocol = json.loads(Path(path).read_text(encoding='utf-8'))
    if (protocol.get('protocol_id') != 'contra_v8_expert_benefit_audit_v6b' or
            protocol.get('scope') !=
            'baseline_anchored_incident_branch_oracle_and_router_audit' or
            protocol.get('main_training_ready') is not False):
        raise ValueError('v6b protocol identity or scope changed')
    if [protocol.get(key) for key in ('source_year', 'source_version',
                                      'expected_sensor_count')] != [2023, 8, 496]:
        raise ValueError('v6b is frozen to the 2023 source-v8 Contra496 package')
    if protocol.get('expected_common_triples') != {'train': 3106, 'val': 618}:
        raise ValueError('v6b common-triple cohort changed')
    if protocol.get('expected_positive_samples') != {'train': 3604, 'val': 917}:
        raise ValueError('v6b full positive cohort changed')
    sensor_metadata = protocol.get('sensor_metadata', {})
    if sensor_metadata != {
            'sha256': '682f3cdf75e643f0b37356ab69cbabb27389be5089f41d3b2cbc4bede3332094',
            'role': 'derive_report_freeway_direction_from_nonzero_distance_support',
            'required_unique_categories': [
                '4-E', '4-W', '24-E', '24-W', '242-N', '242-S'],
            'matched_manifest_agreement_required': True}:
        raise ValueError('v6b sensor-metadata identity or role changed')
    estimand = protocol.get('estimand', {})
    if (estimand.get('activation_advantage') !=
            'absolute_error_off_minus_absolute_error_on' or
            estimand.get('published_A_anchor') != 'incident_on' or
            estimand.get('causal_effect_claimed') is not False):
        raise ValueError('v6b estimand changed')
    support = protocol.get('support', {})
    if (support.get('active_horizons_zero_based_half_open') != [0, 6] or
            support.get('protected_horizons_zero_based_half_open') != [6, 12] or
            support.get('protected_noncandidate_nodes') is not True):
        raise ValueError('v6b intervention support changed')
    oracle = protocol.get('oracle', {})
    if (oracle.get('levels') != ['event', 'event_node', 'cell'] or
            oracle.get('primary_level_for_development_tier') != 'event_node' or
            oracle.get('primary_population_for_development_tier') !=
            'full_positive_validation' or
            oracle.get('all_node_relative_improvement_percent') != {
                'stop_below': 0.3, 'small_router_below': 1.0,
                'formal_study_at_or_above': 1.0} or
            oracle.get('reported_as_model_performance') is not False):
        raise ValueError('v6b oracle design or thresholds changed')
    router = protocol.get('router', {})
    if (router.get('family') != 'weighted_ridge_regression' or
            router.get('levels') != list(LEVELS) or
            router.get('primary_level') != 'event_node' or
            float(router.get('ridge_alpha', 0)) != 10.0 or
            float(router.get('activation_threshold', 1)) != 0.0 or
            router.get('training_cohorts') != list(COHORTS) or
            router.get('row_weighting') != 'equal_event_cohort_weight' or
            'forecast_Y' not in router.get('forbidden_inputs', [])):
        raise ValueError('v6b shallow-router design changed')
    uncertainty = protocol.get('uncertainty', {})
    if (uncertainty.get('method') != 'positive_incident_iso_week_cluster_bootstrap' or
            int(uncertainty.get('draws', 0)) < 1000 or
            float(uncertainty.get('confidence_level', 0)) != 0.95 or
            int(uncertainty.get('seed', 0)) != 2025):
        raise ValueError('v6b uncertainty design changed')
    gate = protocol.get('development_gate', {})
    expected_gate = {
        'primary_router': 'event_node',
        'global_population': 'full_positive_validation',
        'high_impact_population': 'common_matched_validation',
        'maximum_global_harm_fraction_of_A_mae': 0.001,
        'require_candidate_H1_H6_improvement_ci_lower_above_zero': True,
        'require_high_impact_candidate_H1_H6_point_improvement': True,
        'maximum_each_control_H1_H6_harm_fraction_of_off_mae': 0.005,
        'minimum_switch_off_fraction': 0.05,
        'maximum_switch_off_fraction': 0.95,
        'require_exact_A_equality_outside_support': True,
    }
    if gate != expected_gate:
        raise ValueError('v6b prospective development gate changed')
    boundary = protocol.get('information_boundary', {})
    if not all(boundary.get(key) is True for key in (
            'train_and_validation_only', 'test_split_prohibited', 'frozen_A_not_updated',
            'neural_expert_training_prohibited', 'v6a_cohort_may_not_change',
            'validation_may_not_tune_features_alpha_or_threshold',
            'v5c_label_used_only_for_evaluation_stratification')):
        raise ValueError('v6b information boundary changed')
    return protocol


def verify_package(directory):
    directory = Path(directory)
    result = {}
    for name, key in (('summary.json', 'files'), ('context_manifest.json', 'outputs')):
        metadata = json.loads((directory / name).read_text(encoding='utf-8'))
        for filename, expected in metadata[key].items():
            if sha256(directory / filename) != expected:
                raise ValueError(f'Positive package checksum mismatch: {filename}')
        result[name] = sha256(directory / name)
    return result


def verify_inputs(data_dir, primary_dir, secondary_dir, materialized_dir,
                  placebo_dir, sensors_path, protocol):
    materialized_dir, placebo_dir = Path(materialized_dir), Path(placebo_dir)
    summary = json.loads((materialized_dir / 'summary.json').read_text(encoding='utf-8'))
    expected_v6a = protocol['v6a_input']
    if (summary.get('status') != expected_v6a['status'] or
            summary.get('protocol_id') != expected_v6a['protocol_id'] or
            summary.get('protocol_sha256') != expected_v6a['protocol_sha256'] or
            summary.get('test_split_read') is not False or
            summary.get('model_training_performed') is not False):
        raise ValueError('v6a materialization identity or boundary changed')
    expected_outputs = {
        f'{split}_{cohort}_branch_errors.npz'
        for split in SPLITS for cohort in (*COHORTS, 'incident_full')
    }
    if set(summary.get('outputs', {})) != expected_outputs:
        raise ValueError('v6a materialization output set is incomplete')
    for filename, metadata in summary['outputs'].items():
        path = materialized_dir / filename
        if sha256(path) != metadata['sha256']:
            raise ValueError(f'v6a output checksum mismatch: {filename}')
    package = verify_package(data_dir)
    if package != summary['inputs']['positive_package']:
        raise ValueError('Positive package differs from v6a materialization')
    sensors_hash = sha256(sensors_path)
    context_manifest = json.loads(
        (Path(data_dir) / 'context_manifest.json').read_text(encoding='utf-8'))
    if (sensors_hash != protocol['sensor_metadata']['sha256'] or
            sensors_hash not in context_manifest.get('sources', {}).values()):
        raise ValueError('Sensor metadata differs from the report-location context source')
    for directory, key in ((primary_dir, 'primary_controls'),
                           (secondary_dir, 'secondary_controls')):
        actual = {name: sha256(Path(directory) / name)
                  for name in summary['inputs'][key]}
        if actual != summary['inputs'][key]:
            raise ValueError(f'{key} differ from v6a materialization')
    v5c = protocol['v5c_stratification_input']
    if (sha256(placebo_dir / 'summary.json') != v5c['summary_sha256'] or
            sha256(placebo_dir / 'triple_metrics.csv') !=
            v5c['triple_metrics_sha256']):
        raise ValueError('v5c stratification artifacts differ from the v6b protocol')
    placebo_summary = json.loads(
        (placebo_dir / 'summary.json').read_text(encoding='utf-8'))
    if (placebo_summary.get('protocol_id') != v5c['protocol_id'] or
            placebo_summary.get('test_split_read') is not False):
        raise ValueError('v5c identity or test boundary changed')
    return summary


def iso_week(timestamp):
    value = datetime.fromisoformat(timestamp).isocalendar()
    return f'{value.year}-W{value.week:02d}'


def clock_features(timestamp):
    value = datetime.fromisoformat(timestamp)
    tod = value.hour * 12 + value.minute // 5
    dow = (value.weekday() + 1) % 7
    return tod, dow


def history_features(raw, scaler):
    history = np.asarray(raw[:12], dtype=np.float64)
    valid = np.isfinite(history) & (history >= 0)
    fill = np.asarray(scaler['node_fill_mean'], dtype=np.float64)
    values = (np.where(valid, history, fill) - float(scaler['mean'])) / float(
        scaler['std'])
    return np.column_stack([
        values.mean(axis=0), values[-1],
        values[-3:].mean(axis=0) - values[:3].mean(axis=0),
        values.std(axis=0), valid.mean(axis=0),
    ])


def ordered_sensor_categories(path, station_ids):
    rows = read_csv(path)
    by_id = {int(row['station_id']): row for row in rows}
    if len(by_id) != len(station_ids) or set(by_id) != set(map(int, station_ids)):
        raise ValueError('Sensor metadata and station axis differ')
    freeways, directions = [], []
    for station in station_ids:
        row = by_id[int(station)]
        freeways.append(str(parse_freeway(row['Fwy'])))
        directions.append(parse_direction(row['Direction']))
    return np.asarray(freeways), np.asarray(directions)


def location_category(candidate, freeways, directions):
    candidate = np.asarray(candidate)
    freeways, directions = np.asarray(freeways), np.asarray(directions)
    if (candidate.dtype != np.bool_ or candidate.ndim != 1 or
            freeways.shape != candidate.shape or directions.shape != candidate.shape or
            not candidate.any()):
        raise ValueError('Invalid candidate support for location-category derivation')
    categories = {
        (str(freeway), str(direction))
        for freeway, direction in zip(freeways[candidate], directions[candidate])
    }
    if len(categories) != 1:
        raise ValueError('Nonzero distance support spans multiple freeway/direction categories')
    return next(iter(categories))


class FeatureSource:
    """Report-time-safe features in the exact v6a common-triple order."""

    def __init__(self, data_dir, primary_dir, secondary_dir, sensors_path, split,
                 expected_count, expected_positive_count, expected_nodes):
        self.split = split
        self.data_dir = Path(data_dir)
        self.scaler = json.loads((self.data_dir / 'scaler.json').read_text(encoding='utf-8'))
        self.station_ids = np.load(
            self.data_dir / 'station_ids.npy', allow_pickle=False)
        self.sensor_freeways, self.sensor_directions = ordered_sensor_categories(
            sensors_path, self.station_ids)
        self.positive_rows = read_csv(self.data_dir / f'{split}_manifest.csv')
        self.positive_positions = {
            int(row['sample_index']): index for index, row in enumerate(self.positive_rows)
        }
        self.positive_flow = np.load(
            self.data_dir / f'{split}_flow.npy', mmap_mode='r', allow_pickle=False)
        with np.load(self.data_dir / f'{split}_context.npz', allow_pickle=False) as stored:
            self.context = {key: stored[key].copy() for key in stored.files}
        self.location_categories = {
            location_category(
                np.any(distances != 0, axis=-1),
                self.sensor_freeways, self.sensor_directions)
            for distances in self.context['distances']
        }
        self.primary_rows = read_csv(Path(primary_dir) / f'{split}_control_manifest.csv')
        self.primary_by_sample = {
            int(row['positive_sample_index']): row for row in self.primary_rows
        }
        self.primary_flow = np.load(
            Path(primary_dir) / f'{split}_control_flow.npy', mmap_mode='r',
            allow_pickle=False)
        self.secondary_rows = read_csv(
            Path(secondary_dir) / f'{split}_second_control_manifest.csv')
        self.secondary_flow = np.load(
            Path(secondary_dir) / f'{split}_second_control_flow.npy', mmap_mode='r',
            allow_pickle=False)
        if (len(self.secondary_rows) != expected_count or
                len(self.positive_rows) != expected_positive_count or
                len(self.station_ids) != expected_nodes or
                self.positive_flow.shape[1:] != (26, expected_nodes) or
                self.primary_flow.shape[1:] != (26, expected_nodes) or
                self.secondary_flow.shape != (expected_count, 26, expected_nodes)):
            raise ValueError('Feature source differs from the v6b protocol')

    def sample(self, cohort, position):
        if cohort == 'incident_full':
            positive_position = position
            positive = self.positive_rows[positive_position]
            sample = int(positive['sample_index'])
            raw = self.positive_flow[positive_position]
            timestamp = positive['t0']
            incident_id = positive['incident_id']
            manifest_category = None
        else:
            secondary = self.secondary_rows[position]
            sample = int(secondary['positive_sample_index'])
            positive_position = self.positive_positions[sample]
            positive = self.positive_rows[positive_position]
            primary = self.primary_by_sample[sample]
            incident_id = secondary['incident_id']
            manifest_category = (str(secondary['freeway']), secondary['direction'])
            if cohort == 'incident':
                raw = self.positive_flow[positive_position]
                timestamp = positive['t0']
            elif cohort == 'primary_control':
                raw = self.primary_flow[int(primary['control_index'])]
                timestamp = primary['candidate_t0']
            elif cohort == 'secondary_control':
                raw = self.secondary_flow[position]
                timestamp = secondary['candidate_t0']
            else:
                raise ValueError('Unknown cohort')
        distances = np.asarray(self.context['distances'][positive_position], dtype=np.float64)
        candidate = np.any(distances != 0, axis=-1)
        if not candidate.any():
            raise ValueError('Common triple has no candidate nodes')
        freeway, direction = location_category(
            candidate, self.sensor_freeways, self.sensor_directions)
        if manifest_category is not None and (freeway, direction) != manifest_category:
            raise ValueError('Derived report location and matched manifest category differ')
        return {
            'sample': sample, 'incident_id': incident_id,
            'timestamp': timestamp, 'positive_t0': positive['t0'],
            'iso_week': iso_week(positive['t0']),
            'freeway': freeway, 'direction': direction,
            'report_age': float(self.context['report_age_minutes'][positive_position]),
            'distances': distances, 'candidate': candidate,
            'history': history_features(raw, self.scaler),
        }


def category_schema(source):
    roads = sorted({str(row['freeway']) for row in source.secondary_rows})
    directions = sorted({row['direction'] for row in source.secondary_rows})
    if not roads or not directions:
        raise ValueError('Training categorical schema is empty')
    return {'freeway': roads, 'direction': directions}


def _one_hot(value, categories):
    return [float(value == category) for category in categories]


def feature_vector(sample, categories, node=None):
    candidate = sample['candidate']
    distances = sample['distances'][candidate]
    history = sample['history'][candidate]
    tod, dow = clock_features(sample['timestamp'])
    values = [
        sample['report_age'], math.sin(2 * math.pi * tod / 288),
        math.cos(2 * math.pi * tod / 288), math.sin(2 * math.pi * dow / 7),
        math.cos(2 * math.pi * dow / 7), float(candidate.sum()),
        *distances.mean(axis=0), *distances.std(axis=0),
        *history.mean(axis=0), *history.std(axis=0),
        *_one_hot(sample['freeway'], categories['freeway']),
        *_one_hot(sample['direction'], categories['direction']),
    ]
    if node is not None:
        values.extend(sample['distances'][node])
        values.extend(sample['history'][node])
    return np.asarray(values, dtype=np.float64)


def feature_names(categories, level):
    names = [
        'report_age_minutes', 'forecast_tod_sin', 'forecast_tod_cos',
        'forecast_dow_sin', 'forecast_dow_cos', 'candidate_count',
        *[f'distance_{index}_candidate_mean' for index in range(3)],
        *[f'distance_{index}_candidate_std' for index in range(3)],
        *[f'history_{name}_candidate_mean' for name in
          ('mean', 'last', 'trend', 'dispersion', 'valid_fraction')],
        *[f'history_{name}_candidate_std' for name in
          ('mean', 'last', 'trend', 'dispersion', 'valid_fraction')],
        *[f'freeway={value}' for value in categories['freeway']],
        *[f'direction={value}' for value in categories['direction']],
    ]
    if level == 'event_node':
        names.extend([f'node_distance_{index}' for index in range(3)])
        names.extend([f'node_history_{name}' for name in
                      ('mean', 'last', 'trend', 'dispersion', 'valid_fraction')])
    return names


def load_errors(materialized_dir, split, cohort, expected_samples, expected_nodes):
    path = Path(materialized_dir) / f'{split}_{cohort}_branch_errors.npz'
    with np.load(path, allow_pickle=False) as stored:
        required = {'absolute_error_on', 'absolute_error_off', 'valid',
                    'candidate_mask', 'positive_sample_index'}
        if set(stored.files) != required:
            raise ValueError(f'Unexpected v6a array schema: {path.name}')
        arrays = {key: stored[key].copy() for key in stored.files}
    shape = (expected_samples, 12, expected_nodes, 1)
    if (arrays['absolute_error_on'].shape != shape or
            arrays['absolute_error_off'].shape != shape or
            arrays['valid'].shape != shape or
            arrays['candidate_mask'].shape != (expected_samples, expected_nodes) or
            arrays['positive_sample_index'].shape != (expected_samples,) or
            arrays['absolute_error_on'].dtype != np.float32 or
            arrays['absolute_error_off'].dtype != np.float32 or
            arrays['valid'].dtype != np.bool_ or
            arrays['candidate_mask'].dtype != np.bool_ or
            not np.isfinite(arrays['absolute_error_on']).all() or
            not np.isfinite(arrays['absolute_error_off']).all()):
        raise ValueError(f'Invalid v6a arrays: {path.name}')
    return arrays


def active_support(valid, candidate):
    support = np.zeros_like(valid, dtype=bool)
    support[:, :6, :, :] = candidate[:, None, :, None]
    return support & valid


def aggregate_advantage(advantage, valid, candidate, level):
    support = active_support(valid, candidate)
    selected = np.where(support, advantage, 0.)
    if level == 'event':
        counts = support.sum(axis=(1, 2, 3))
        return selected.sum(axis=(1, 2, 3)) / counts
    if level == 'event_node':
        counts = support.sum(axis=(1, 3))
        return np.divide(selected.sum(axis=(1, 3)), counts,
                         out=np.zeros_like(counts, dtype=np.float64), where=counts > 0)
    if level == 'cell':
        return advantage
    raise ValueError('Unknown aggregation level')


def routed_error(error_on, error_off, valid, candidate, gate_on, level):
    support = active_support(valid, candidate)
    if level == 'event':
        gate = gate_on[:, None, None, None]
    elif level == 'event_node':
        gate = gate_on[:, None, :, None]
    elif level == 'cell':
        gate = gate_on
    else:
        raise ValueError('Unknown routing level')
    choose_off = support & ~gate
    return np.where(choose_off, error_off, error_on)


def population_mae(error, selected):
    count = int(selected.sum())
    if count < 1:
        raise ValueError('Metric population is empty')
    return float(error[selected].astype(np.float64).mean())


def metric_populations(valid, candidate):
    candidate_cells = candidate[:, None, :, None]
    early = np.zeros_like(valid, dtype=bool)
    early[:, :6] = True
    return {
        'all': valid,
        'candidate_h1_h6': valid & candidate_cells & early,
        'candidate_h7_h12': valid & candidate_cells & ~early,
        'noncandidate': valid & ~candidate_cells,
    }


def comparison_summary(error_on, error_off, routed, valid, candidate):
    result = {}
    for name, selected in metric_populations(valid, candidate).items():
        on = population_mae(error_on, selected)
        off = population_mae(error_off, selected)
        route = population_mae(routed, selected)
        result[name] = {
            'valid_cells': int(selected.sum()), 'mae_on': on, 'mae_off': off,
            'mae_routed': route, 'routed_improvement_vs_on': on - route,
            'routed_harm_vs_off': route - off,
            'relative_improvement_vs_on_percent':
                100 * (on - route) / on if on else None,
        }
    return result


def oracle_summary(arrays):
    on, off, valid, candidate = (arrays[key] for key in (
        'absolute_error_on', 'absolute_error_off', 'valid', 'candidate_mask'))
    advantage = off.astype(np.float64) - on.astype(np.float64)
    result = {}
    for level in ('event', 'event_node', 'cell'):
        aggregate = aggregate_advantage(advantage, valid, candidate, level)
        gate_on = aggregate >= 0
        routed = routed_error(on, off, valid, candidate, gate_on, level)
        result[level] = comparison_summary(on, off, routed, valid, candidate)
    return result


class RidgeModel:
    def __init__(self, mean, scale, target_mean, coefficient, alpha):
        self.mean, self.scale = mean, scale
        self.target_mean, self.coefficient, self.alpha = target_mean, coefficient, alpha

    def predict(self, features):
        return self.target_mean + ((features - self.mean) / self.scale).dot(
            self.coefficient)


def fit_weighted_ridge(features, targets, weights, alpha):
    features = np.asarray(features, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if (features.ndim != 2 or targets.shape != (len(features),) or
            weights.shape != targets.shape or not np.isfinite(features).all() or
            not np.isfinite(targets).all() or not np.isfinite(weights).all() or
            np.any(weights <= 0) or alpha <= 0):
        raise ValueError('Invalid weighted-ridge input')
    normalized = weights / weights.sum()
    mean = np.sum(features * normalized[:, None], axis=0)
    scale = np.sqrt(np.sum(np.square(features - mean) * normalized[:, None], axis=0))
    scale = np.where(scale > 1e-12, scale, 1.)
    target_mean = float(np.sum(targets * normalized))
    standardized = (features - mean) / scale
    root_weight = np.sqrt(weights)
    design = standardized * root_weight[:, None]
    response = (targets - target_mean) * root_weight
    coefficient = np.linalg.solve(
        design.T.dot(design) + alpha * np.eye(features.shape[1]),
        design.T.dot(response))
    return RidgeModel(mean, scale, target_mean, coefficient, alpha)


def make_router_rows(source, cohort, arrays, categories, level):
    advantage = arrays['absolute_error_off'].astype(np.float64) - \
        arrays['absolute_error_on'].astype(np.float64)
    target = aggregate_advantage(
        advantage, arrays['valid'], arrays['candidate_mask'], level)
    features, targets, weights, metadata = [], [], [], []
    expected_samples = arrays['absolute_error_on'].shape[0]
    for position in range(expected_samples):
        sample = source.sample(cohort, position)
        if (sample['sample'] != int(arrays['positive_sample_index'][position]) or
                not np.array_equal(sample['candidate'], arrays['candidate_mask'][position])):
            raise ValueError('Feature rows and v6a errors have different identities')
        if level == 'event':
            features.append(feature_vector(sample, categories))
            targets.append(float(target[position]))
            weights.append(1.)
            metadata.append((sample, None))
        else:
            nodes = np.flatnonzero(sample['candidate'])
            for node in nodes:
                features.append(feature_vector(sample, categories, int(node)))
                targets.append(float(target[position, node]))
                weights.append(1. / len(nodes))
                metadata.append((sample, int(node)))
    return (np.asarray(features), np.asarray(targets), np.asarray(weights), metadata)


def predicted_gate_matrix(predictions, metadata, samples, nodes, level, threshold):
    if level == 'event':
        if len(predictions) != samples:
            raise ValueError('Event predictions do not match sample count')
        return predictions >= threshold
    gate = np.ones((samples, nodes), dtype=bool)
    positions = {}
    for row, (_, node) in enumerate(metadata):
        sample = int(metadata[row][0]['sample'])
        positions.setdefault(sample, len(positions))
        gate[positions[sample], node] = predictions[row] >= threshold
    if len(positions) != samples:
        raise ValueError('Event-node predictions do not cover all samples')
    return gate


def cluster_bootstrap_mean(values, clusters, draws, confidence, seed):
    values = np.asarray(values, dtype=np.float64)
    clusters = np.asarray(clusters)
    if values.ndim == 1:
        values = values[:, None]
    unique = np.unique(clusters)
    if len(values) != len(clusters) or len(unique) < 2:
        raise ValueError('Cluster bootstrap requires aligned rows and multiple clusters')
    rng = np.random.default_rng(seed)
    by_cluster = [np.flatnonzero(clusters == cluster) for cluster in unique]
    estimates = np.empty((draws, values.shape[1]), dtype=np.float64)
    for draw in range(draws):
        selected = rng.integers(0, len(unique), size=len(unique))
        indices = np.concatenate([by_cluster[index] for index in selected])
        estimates[draw] = values[indices].mean(axis=0)
    alpha = (1 - confidence) / 2
    return (np.quantile(estimates, alpha, axis=0),
            np.quantile(estimates, 1 - alpha, axis=0))


def per_event_difference(left, right, selected):
    difference = np.where(selected, left.astype(np.float64) - right.astype(np.float64), 0.)
    counts = selected.sum(axis=(1, 2, 3))
    return difference.sum(axis=(1, 2, 3)) / counts


def uncertainty_summary(on, off, routed, valid, candidate, clusters, protocol, seed):
    populations = metric_populations(valid, candidate)
    values = np.column_stack([
        per_event_difference(on, routed, populations['all']),
        per_event_difference(on, routed, populations['candidate_h1_h6']),
        per_event_difference(routed, off, populations['candidate_h1_h6']),
    ])
    low, high = cluster_bootstrap_mean(
        values, clusters, int(protocol['uncertainty']['draws']),
        float(protocol['uncertainty']['confidence_level']), seed)
    keys = ('all_improvement_vs_on', 'candidate_h1_h6_improvement_vs_on',
            'candidate_h1_h6_harm_vs_off')
    return {
        key: {'mean': float(values[:, index].mean()), 'ci_low': float(low[index]),
              'ci_high': float(high[index])}
        for index, key in enumerate(keys)
    }


def load_high_impact(placebo_dir):
    rows = read_csv(Path(placebo_dir) / 'triple_metrics.csv')
    mapping = {(row['split'], int(row['positive_sample_index'])):
               float(row['early_excess_divergence']) for row in rows}
    train = np.asarray([value for (split, _), value in mapping.items() if split == 'train'])
    if len(train) != 3106:
        raise ValueError('v5c train cohort differs from v6b')
    threshold = float(np.quantile(train, .75))
    return mapping, threshold


def write_router_rows(path, split, cohort, level, targets, predictions, metadata,
                      high_impact, threshold):
    fields = ['split', 'cohort', 'level', 'positive_sample_index', 'incident_id',
              'positive_t0', 'positive_iso_week', 'node_index',
              'target_activation_advantage', 'predicted_activation_advantage',
              'choose_incident_on', 'high_impact_v5c']
    rows = []
    for target, prediction, (sample, node) in zip(targets, predictions, metadata):
        rows.append({
            'split': split, 'cohort': cohort, 'level': level,
            'positive_sample_index': sample['sample'],
            'incident_id': sample['incident_id'], 'positive_t0': sample['positive_t0'],
            'positive_iso_week': sample['iso_week'],
            'node_index': '' if node is None else node,
            'target_activation_advantage': float(target),
            'predicted_activation_advantage': float(prediction),
            'choose_incident_on': bool(prediction >= threshold),
            'high_impact_v5c': (
                '' if (split, sample['sample']) not in high_impact else
                bool(high_impact[(split, sample['sample'])])),
        })
    write_csv(path, rows, fields)
    return rows


def evaluate_router(model, level, source, cohort, arrays, categories, protocol,
                    high_impact, output, seed):
    features, targets, _, metadata = make_router_rows(
        source, cohort, arrays, categories, level)
    predictions = model.predict(features)
    threshold = float(protocol['router']['activation_threshold'])
    sample_count = arrays['absolute_error_on'].shape[0]
    gate_on = predicted_gate_matrix(
        predictions, metadata, sample_count, len(source.station_ids),
        level, threshold)
    on, off, valid, candidate = (arrays[key] for key in (
        'absolute_error_on', 'absolute_error_off', 'valid', 'candidate_mask'))
    routed = routed_error(on, off, valid, candidate, gate_on, level)
    clusters = np.asarray([source.sample(cohort, index)['iso_week']
                           for index in range(sample_count)])
    metrics = comparison_summary(on, off, routed, valid, candidate)
    metrics['uncertainty'] = uncertainty_summary(
        on, off, routed, valid, candidate, clusters, protocol, seed)
    support = active_support(valid, candidate)
    labeled = np.asarray([
        (source.split, int(sample)) in high_impact
        for sample in arrays['positive_sample_index']])
    high_mask = np.asarray([
        high_impact.get((source.split, int(sample)), False)
        for sample in arrays['positive_sample_index']])
    if high_mask.any():
        high_cells = support & high_mask[:, None, None, None]
        metrics['high_impact_candidate_h1_h6'] = {
            'labeled_events': int(labeled.sum()), 'events': int(high_mask.sum()),
            'valid_cells': int(high_cells.sum()),
            'mae_on': population_mae(on, high_cells),
            'mae_routed': population_mae(routed, high_cells),
            'improvement_vs_on': population_mae(on, high_cells) -
                                 population_mae(routed, high_cells),
        }
    candidate_gate = gate_on if level == 'event' else gate_on[candidate]
    metrics['routing'] = {
        'rows': int(len(predictions)),
        'choose_on_fraction': float(candidate_gate.mean()),
        'switch_off_fraction': float((~candidate_gate).mean()),
        'target_mean': float(targets.mean()),
        'prediction_mean': float(predictions.mean()),
        'prediction_target_correlation': (float(np.corrcoef(predictions, targets)[0, 1])
                                          if np.std(predictions) > 0 and
                                          np.std(targets) > 0 else None),
    }
    write_router_rows(
        output, source.split, cohort, level, targets, predictions, metadata,
        high_impact, threshold)
    return metrics


def gate_decision(validation, oracle, protocol):
    gate = protocol['development_gate']
    primary = validation['event_node']
    incident_full = primary['incident_full']
    incident_matched = primary['incident']
    all_metrics = incident_full['all']
    uncertainty = incident_full['uncertainty']
    maximum_global_harm = (gate['maximum_global_harm_fraction_of_A_mae'] *
                           all_metrics['mae_on'])
    checks = {
        'validation_global_noninferiority':
            uncertainty['all_improvement_vs_on']['ci_low'] >= -maximum_global_harm,
        'validation_candidate_h1_h6_improvement':
            uncertainty['candidate_h1_h6_improvement_vs_on']['ci_low'] > 0,
        'validation_high_impact_point_improvement':
            incident_matched['high_impact_candidate_h1_h6']['improvement_vs_on'] > 0,
        'selective_switch_fraction':
            gate['minimum_switch_off_fraction'] <=
            incident_full['routing']['switch_off_fraction'] <=
            gate['maximum_switch_off_fraction'],
        'protected_h7_h12_exact_A':
            incident_full['candidate_h7_h12']['routed_improvement_vs_on'] == 0,
        'protected_noncandidate_exact_A':
            incident_full['noncandidate']['routed_improvement_vs_on'] == 0,
    }
    control_margin = gate['maximum_each_control_H1_H6_harm_fraction_of_off_mae']
    for cohort in ('primary_control', 'secondary_control'):
        current = primary[cohort]
        checks[f'{cohort}_routine_harm_bound'] = (
            current['uncertainty']['candidate_h1_h6_harm_vs_off']['ci_high'] <=
            control_margin * current['candidate_h1_h6']['mae_off'])
    router_pass = all(checks.values())
    oracle_improvement = oracle['val']['incident_full']['event_node']['all'][
        'relative_improvement_vs_on_percent']
    thresholds = protocol['oracle']['all_node_relative_improvement_percent']
    if oracle_improvement < thresholds['stop_below']:
        recommendation = 'STOP_BRANCH_ROUTER_ORACLE_BELOW_0_3_PERCENT'
    elif not router_pass:
        recommendation = 'BRANCH_ROUTING_NOT_VALIDATED'
    elif oracle_improvement < thresholds['small_router_below']:
        recommendation = 'SMALL_BASELINE_ANCHORED_BRANCH_ROUTER_STUDY_ALLOWED'
    else:
        recommendation = 'FORMAL_SELECTIVE_BRANCH_ROUTING_STUDY_ALLOWED'
    return {
        'checks': checks, 'router_gate_passed': router_pass,
        'primary_oracle_all_node_relative_improvement_percent': oracle_improvement,
        'maximum_global_harm_raw_mae': maximum_global_harm,
        'recommendation': recommendation,
    }


def audit(data_dir, primary_dir, secondary_dir, materialized_dir, placebo_dir,
          sensors_path, protocol_path, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError('v6b output exists; preserve it and use a new directory')
    protocol = load_protocol(protocol_path)
    v6a_summary = verify_inputs(
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
    categories = category_schema(sources['train'])
    required_categories = {
        tuple(value.split('-', 1))
        for value in protocol['sensor_metadata']['required_unique_categories']
    }
    for split, source in sources.items():
        if source.location_categories != required_categories:
            raise ValueError(f'{split} report-location category support changed')
    for split in SPLITS:
        samples = {int(row['positive_sample_index']) for row in sources[split].secondary_rows}
        if samples != {sample for current_split, sample in high_values if current_split == split}:
            raise ValueError('v5c stratification and v6a cohorts differ')
    output.mkdir(parents=True, exist_ok=False)
    oracle = {split: {} for split in SPLITS}
    for split in SPLITS:
        count = protocol['expected_common_triples'][split]
        for cohort in (*COHORTS, 'incident_full'):
            expected_count = (protocol['expected_positive_samples'][split]
                              if cohort == 'incident_full' else count)
            arrays = load_errors(
                materialized_dir, split, cohort, expected_count,
                protocol['expected_sensor_count'])
            oracle[split][cohort] = oracle_summary(arrays)
    validation, router_files = {level: {} for level in LEVELS}, []
    model_arrays = {}
    for level_number, level in enumerate(LEVELS):
        train_features, train_targets, train_weights = [], [], []
        for cohort in COHORTS:
            arrays = load_errors(
                materialized_dir, 'train', cohort,
                protocol['expected_common_triples']['train'],
                protocol['expected_sensor_count'])
            features, targets, weights, _ = make_router_rows(
                sources['train'], cohort, arrays, categories, level)
            train_features.append(features)
            train_targets.append(targets)
            train_weights.append(weights)
        model = fit_weighted_ridge(
            np.concatenate(train_features), np.concatenate(train_targets),
            np.concatenate(train_weights), float(protocol['router']['ridge_alpha']))
        names = feature_names(categories, level)
        if len(names) != len(model.coefficient):
            raise ValueError('Router feature names and coefficients differ')
        model_arrays.update({
            f'{level}_feature_names': np.asarray(names),
            f'{level}_feature_mean': model.mean,
            f'{level}_feature_scale': model.scale,
            f'{level}_coefficient': model.coefficient,
            f'{level}_target_mean': np.asarray(model.target_mean),
            f'{level}_ridge_alpha': np.asarray(model.alpha),
        })
        for cohort_number, cohort in enumerate(('incident_full', *COHORTS)):
            expected_count = (protocol['expected_positive_samples']['val']
                              if cohort == 'incident_full' else
                              protocol['expected_common_triples']['val'])
            arrays = load_errors(
                materialized_dir, 'val', cohort,
                expected_count,
                protocol['expected_sensor_count'])
            path = output / f'val_{cohort}_{level}_router.csv'
            validation[level][cohort] = evaluate_router(
                model, level, sources['val'], cohort, arrays, categories, protocol,
                high_impact, path,
                int(protocol['uncertainty']['seed']) + level_number * 10 + cohort_number)
            router_files.append(path)
    model_path = output / 'router_models.npz'
    with model_path.open('wb') as stream:
        np.savez_compressed(stream, **model_arrays)
    decision = gate_decision(validation, oracle, protocol)
    summary = {
        'status': 'EXPERT_BENEFIT_AUDIT_COMPLETE', 'scope': protocol['scope'],
        'protocol_id': protocol['protocol_id'], 'protocol_sha256': sha256(protocol_path),
        'main_training_ready': False, 'test_split_read': False,
        'frozen_A_updated': False, 'neural_expert_trained': False,
        'shallow_router_fit_on_train_only': True,
        'validation_used_to_tune_router': False,
        'v5c_label_used_as_router_input': False,
        'high_impact_train_q75_raw': high_threshold,
        'categorical_schema_fit_on_train': categories,
        'report_location_category_source': protocol['sensor_metadata']['role'],
        'oracle_reported_as_model_performance': False,
        'oracle': oracle, 'validation_router': validation,
        'development_gate': decision,
        'estimand': protocol['estimand'], 'support': protocol['support'],
        'uncertainty': protocol['uncertainty'],
        'limitations': [
            'The oracle uses forecast outcomes and is only an unattainable diagnostic ceiling.',
            'The shallow router selects between incident-on and incident-off inference from one checkpoint; it is not a newly trained residual expert.',
            'Its oracle is not an upper bound on an arbitrary future residual expert.',
            'Matched controls are observational routine placebos, not causal counterfactual outcomes.',
            'Passing development gates would authorize only train/validation branch-router development, never test access.',
        ],
        'inputs': {
            'v6a_summary_sha256': sha256(Path(materialized_dir) / 'summary.json'),
            'v6a_protocol_sha256': v6a_summary['protocol_sha256'],
            'v5c_summary_sha256': sha256(Path(placebo_dir) / 'summary.json'),
            'v5c_triple_metrics_sha256': sha256(
                Path(placebo_dir) / 'triple_metrics.csv'),
            'sensors_sha256': sha256(sensors_path),
            'protocol_sha256': sha256(protocol_path), 'code_sha256': sha256(__file__),
        },
    }
    outputs = [model_path, *router_files]
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
    print(f'Saved v6b expert-benefit audit: {summary_path}', flush=True)
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
        'expert_benefit_audit_v6b.json'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    audit(args.data_dir, args.primary_control_dir, args.secondary_control_dir,
          args.materialized_dir, args.placebo_dir, args.sensors, args.protocol,
          args.output)


if __name__ == '__main__':
    main()
