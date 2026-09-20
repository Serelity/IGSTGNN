"""Audit incident divergence against two-control routine placebo divergence."""

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.chronological.audit_matched_controls import read_csv, sha256, write_csv
from experiments.chronological.audit_matched_outcomes import (
    cluster_bootstrap_mean, iso_week, phase_for_step, source_slots,
)
from experiments.chronological.score_matched_controls import SPLITS


TRIPLE_FIELDS = [
    'split', 'secondary_control_index', 'primary_control_index',
    'positive_sample_index', 'incident_id', 'positive_t0',
    'primary_candidate_t0', 'secondary_candidate_t0', 'freeway', 'direction',
    'affected_node_count', 'positive_iso_week', 'strict_nonoverlap_selected',
    'baseline_incident_divergence', 'baseline_routine_divergence',
    'early_incident_change', 'early_routine_change', 'early_excess_divergence',
    'early_excess_in_train_std', 'late_incident_change', 'late_routine_change',
    'late_excess_divergence', 'late_excess_in_train_std',
]

TRAJECTORY_FIELDS = [
    'split', 'population', 'step_index', 'relative_minutes', 'phase', 'triples',
    'incident_mean', 'primary_control_mean', 'secondary_control_mean',
    'incident_divergence', 'incident_divergence_ci_low',
    'incident_divergence_ci_high', 'routine_divergence',
    'routine_divergence_ci_low', 'routine_divergence_ci_high',
    'baseline_adjusted_incident_divergence', 'incident_change_ci_low',
    'incident_change_ci_high', 'baseline_adjusted_routine_divergence',
    'routine_change_ci_low', 'routine_change_ci_high', 'excess_divergence',
    'excess_divergence_ci_low', 'excess_divergence_ci_high',
]

ROAD_FIELDS = [
    'split', 'freeway', 'direction', 'triples', 'positive_iso_week_blocks',
    'early_incident_change_raw', 'early_routine_change_raw',
    'early_excess_divergence_raw', 'early_excess_ci_low_raw',
    'early_excess_ci_high_raw', 'early_excess_in_train_std',
    'late_incident_change_raw', 'late_routine_change_raw',
    'late_excess_divergence_raw', 'late_excess_ci_low_raw',
    'late_excess_ci_high_raw', 'late_excess_in_train_std',
]


def load_protocol(path):
    protocol = json.loads(Path(path).read_text(encoding='utf-8'))
    if protocol.get('scope') != 'matched_incident_vs_routine_placebo_audit':
        raise ValueError('Protocol scope must remain incident-versus-routine placebo audit')
    if protocol.get('main_training_ready') is not False:
        raise ValueError('The placebo audit cannot pre-authorize main training')
    if (int(protocol.get('source_year', 0)) != 2023 or
            int(protocol.get('source_version', 0)) != 8 or
            int(protocol.get('expected_sensor_count', 0)) != 496):
        raise ValueError('v5c is frozen to the 2023 source-v8 Contra496 package')
    expected_window = {
        'steps': 26, 'step_minutes': 5, 'first_relative_minute': -65,
        'X_slice': [0, 12], 'excluded_latency_slice': [12, 14],
        'Y_slice': [14, 26], 'baseline_slice': [9, 12],
        'early_Y_slice': [14, 20], 'late_Y_slice': [20, 26],
    }
    if any(protocol['window'].get(key) != value
           for key, value in expected_window.items()):
        raise ValueError('v5c window, baseline, or horizon definition changed')
    boundary = protocol['information_boundary']
    required = [
        'both_control_assignments_frozen_before_outcome_audit',
        'outcomes_may_not_change_matching_or_common_cohort',
        'train_and_validation_only', 'test_split_prohibited',
        'model_training_prohibited', 'model_predictions_prohibited',
    ]
    if not all(boundary.get(key) is True for key in required):
        raise ValueError('v5c information boundary changed')
    estimand = protocol['estimand']
    if (estimand.get('incident_divergence') !=
            '0.5*(mean_nodes(abs(I-C1))+mean_nodes(abs(I-C2)))' or
            estimand.get('routine_divergence') != 'mean_nodes(abs(C1-C2))' or
            estimand.get('baseline_adjustment') !=
            'subtract_each_divergence_series_own_mean_over_Hminus3_to_Hminus1' or
            estimand.get('primary_quantity') !=
            'baseline_adjusted_incident_divergence_minus_baseline_adjusted_routine_divergence' or
            estimand.get('event_equal_weighting') is not True or
            estimand.get('causal_effect_claimed') is not False):
        raise ValueError('v5c estimand changed')
    uncertainty = protocol['uncertainty']
    if (uncertainty.get('method') != 'positive_incident_iso_week_cluster_bootstrap' or
            int(uncertainty.get('draws', 0)) < 1000 or
            not 0 < float(uncertainty.get('confidence_level', 0)) < 1):
        raise ValueError('v5c uncertainty protocol changed or is underpowered')
    sensitivity = protocol['nonoverlap_sensitivity']
    if (sensitivity.get('selection_order') !=
            'positive_t0_then_positive_sample_index' or
            sensitivity.get(
                'all_incident_primary_secondary_source_slots_share_one_exclusion_set') is not True or
            sensitivity.get('outcomes_used_for_selection') is not False or
            sensitivity.get('minimum_selected_triples') != {'train': 350, 'val': 80}):
        raise ValueError('v5c strict non-overlap sensitivity changed')
    descriptive = protocol['descriptive_outputs']
    if not all(descriptive.get(key) is True for key in (
            'early_Y_block_interval_reported', 'road_direction_stratification_reported',
            'descriptive_results_may_not_change_the_primary_gate')):
        raise ValueError('v5c descriptive reporting boundary changed')
    gate = protocol['placebo_gate']
    if (gate.get('primary_horizon') != 'late_H7_H12' or
            float(gate.get('minimum_late_excess_in_train_std', 0)) != 0.05 or
            float(gate.get('minimum_nonoverlap_late_excess_in_train_std', 0)) != 0.025 or
            gate.get('require_main_block_ci_lower_above_zero_in_both_splits') is not True or
            gate.get('require_nonoverlap_positive_in_both_splits') is not True):
        raise ValueError('v5c placebo gate changed')
    return protocol


def _hash_group(directory, specification, excluded):
    directory = Path(directory)
    return {
        name: sha256(directory / name) for name in specification if name not in excluded
    }


def verify_inputs(data_dir, primary_dir, secondary_dir, protocol):
    positive_hashes = _hash_group(
        data_dir, protocol['positive_inputs'], set())
    if positive_hashes != protocol['positive_inputs']:
        raise ValueError('Positive package fingerprints differ from v5c protocol')
    primary_hashes = _hash_group(
        primary_dir, protocol['primary_control_inputs'], {'protocol_id', 'expected_shapes'})
    expected_primary = {
        key: value for key, value in protocol['primary_control_inputs'].items()
        if key not in ('protocol_id', 'expected_shapes')
    }
    if primary_hashes != expected_primary:
        raise ValueError('Primary-control fingerprints differ from v5c protocol')
    secondary_hashes = _hash_group(
        secondary_dir, protocol['secondary_control_inputs'], {'protocol_id', 'expected_shapes'})
    expected_secondary = {
        key: value for key, value in protocol['secondary_control_inputs'].items()
        if key not in ('protocol_id', 'expected_shapes')
    }
    if secondary_hashes != expected_secondary:
        raise ValueError('Secondary-control fingerprints differ from v5c protocol')
    positive_summary = json.loads(
        (Path(data_dir) / 'summary.json').read_text(encoding='utf-8'))
    primary_summary = json.loads(
        (Path(primary_dir) / 'summary.json').read_text(encoding='utf-8'))
    secondary_summary = json.loads(
        (Path(secondary_dir) / 'summary.json').read_text(encoding='utf-8'))
    if (positive_summary.get('source_version') != 8 or
            positive_summary.get('test_flow_built') is not False or
            positive_summary.get('split_counts', {}).get('test') is not None):
        raise ValueError('Positive package violates the development-only boundary')
    if (primary_summary.get('status') != 'MATCHED_NONINCIDENT_MATERIALIZATION_PASS' or
            primary_summary.get('protocol_id') !=
            protocol['primary_control_inputs']['protocol_id'] or
            primary_summary.get('future_Y_used_to_rank_or_replace_controls') is not False or
            primary_summary.get('test_split_read') is not False):
        raise ValueError('Primary-control materialization boundary differs')
    if (secondary_summary.get('status') !=
            'SECOND_MATCHED_CONTROL_MATERIALIZATION_PASS' or
            secondary_summary.get('protocol_id') !=
            protocol['secondary_control_inputs']['protocol_id'] or
            secondary_summary.get(
                'future_Y_used_to_rank_replace_or_remove_controls') is not False or
            secondary_summary.get('test_split_read') is not False or
            secondary_summary.get('outcome_audit_results_read') is not False):
        raise ValueError('Secondary-control materialization boundary differs')
    return positive_hashes, primary_hashes, secondary_hashes


def select_nonoverlap_triples(secondary_rows, primary_by_sample, positive_by_sample, window):
    """Greedily select triples with no shared source slot within or between triples."""
    selected, used = set(), set()
    for row in sorted(secondary_rows, key=lambda item: (
            item['positive_t0'], int(item['positive_sample_index']))):
        sample = int(row['positive_sample_index'])
        if sample not in primary_by_sample or sample not in positive_by_sample:
            raise ValueError('Second control references an unknown common-triple sample')
        three = [
            set(source_slots(positive_by_sample[sample]['x_start'], window)),
            set(source_slots(primary_by_sample[sample]['x_start'], window)),
            set(source_slots(row['x_start'], window)),
        ]
        if any(three[left] & three[right]
               for left, right in ((0, 1), (0, 2), (1, 2))):
            continue
        combined = set().union(*three)
        if combined & used:
            continue
        selected.add(int(row['control_index']))
        used.update(combined)
    return selected


def compute_divergences(incident, primary, secondary, baseline):
    incident = np.asarray(incident, dtype=np.float64)
    primary = np.asarray(primary, dtype=np.float64)
    secondary = np.asarray(secondary, dtype=np.float64)
    if (incident.shape != primary.shape or incident.shape != secondary.shape or
            incident.ndim != 2 or incident.shape[0] != 26 or not incident.shape[1]):
        raise ValueError('Three affected-node windows must have one shared non-empty shape')
    if (not np.isfinite(incident).all() or not np.isfinite(primary).all() or
            not np.isfinite(secondary).all() or (incident < 0).any() or
            (primary < 0).any() or (secondary < 0).any()):
        raise ValueError('Triple affected-node values must be finite and nonnegative')
    incident_divergence = 0.5 * (
        np.abs(incident - primary).mean(axis=1) +
        np.abs(incident - secondary).mean(axis=1))
    routine_divergence = np.abs(primary - secondary).mean(axis=1)
    incident_change = incident_divergence - incident_divergence[baseline].mean()
    routine_change = routine_divergence - routine_divergence[baseline].mean()
    return {
        'incident': incident.mean(axis=1), 'primary': primary.mean(axis=1),
        'secondary': secondary.mean(axis=1),
        'incident_divergence': incident_divergence,
        'routine_divergence': routine_divergence,
        'incident_change': incident_change, 'routine_change': routine_change,
        'excess': incident_change - routine_change,
    }


def validate_and_extract_split(data_dir, primary_dir, secondary_dir, split, protocol):
    data_dir, primary_dir, secondary_dir = map(Path, (data_dir, primary_dir, secondary_dir))
    station_ids = np.load(data_dir / 'station_ids.npy', allow_pickle=False)
    positive_rows = read_csv(data_dir / f'{split}_manifest.csv')
    positive_by_sample = {int(row['sample_index']): row for row in positive_rows}
    positive_positions = {
        int(row['sample_index']): index for index, row in enumerate(positive_rows)
    }
    primary_rows = read_csv(primary_dir / f'{split}_control_manifest.csv')
    primary_by_sample = {int(row['positive_sample_index']): row for row in primary_rows}
    secondary_rows = read_csv(
        secondary_dir / f'{split}_second_control_manifest.csv')
    if (len(positive_by_sample) != len(positive_rows) or
            len(primary_by_sample) != len(primary_rows)):
        raise ValueError(f'{split} positive or primary manifest contains duplicate samples')

    positive = np.load(data_dir / f'{split}_flow.npy', mmap_mode='r', allow_pickle=False)
    primary = np.load(primary_dir / f'{split}_control_flow.npy', mmap_mode='r',
                      allow_pickle=False)
    primary_masks = np.load(primary_dir / f'{split}_affected_mask.npy', allow_pickle=False)
    secondary = np.load(
        secondary_dir / f'{split}_second_control_flow.npy', mmap_mode='r',
        allow_pickle=False)
    secondary_masks = np.load(
        secondary_dir / f'{split}_second_affected_mask.npy', allow_pickle=False)
    primary_shape = tuple(protocol['primary_control_inputs']['expected_shapes'][split])
    secondary_shape = tuple(protocol['secondary_control_inputs']['expected_shapes'][split])
    if (primary.shape != primary_shape or
            primary_masks.shape != (primary_shape[0], primary_shape[2]) or
            secondary.shape != secondary_shape or
            secondary_masks.shape != (secondary_shape[0], secondary_shape[2]) or
            positive.shape != (len(positive_rows), 26, primary_shape[2]) or
            station_ids.shape != (primary_shape[2],) or
            primary_masks.dtype != np.bool_ or secondary_masks.dtype != np.bool_):
        raise ValueError(f'{split} array shapes or mask dtypes differ from v5c protocol')
    if len(secondary_rows) != secondary_shape[0]:
        raise ValueError(f'{split} secondary manifest count differs from v5c protocol')

    strict = select_nonoverlap_triples(
        secondary_rows, primary_by_sample, positive_by_sample, protocol['window'])
    baseline = slice(*protocol['window']['baseline_slice'])
    early = slice(*protocol['window']['early_Y_slice'])
    late = slice(*protocol['window']['late_Y_slice'])
    scaler = json.loads((data_dir / 'scaler.json').read_text(encoding='utf-8'))
    train_std = float(scaler['std'])
    if not np.isfinite(train_std) or train_std <= 0:
        raise ValueError('Training standard deviation must be finite and positive')
    pair_rows, collected = [], {
        key: [] for key in (
            'incident', 'primary', 'secondary', 'incident_divergence',
            'routine_divergence', 'incident_change', 'routine_change', 'excess')
    }
    for position, row in enumerate(secondary_rows):
        secondary_index = int(row['control_index'])
        sample = int(row['positive_sample_index'])
        if (secondary_index != position or sample not in primary_by_sample or
                sample not in positive_by_sample):
            raise ValueError(f'{split} secondary order or primary identity is inconsistent')
        primary_row, positive_row = primary_by_sample[sample], positive_by_sample[sample]
        primary_index = int(primary_row['control_index'])
        if (row['split'] != split or primary_row['split'] != split or
                positive_row['split'] != split or
                len({row['incident_id'], primary_row['incident_id'],
                     positive_row['incident_id']}) != 1 or
                len({row['positive_t0'], primary_row['positive_t0'],
                     positive_row['t0']}) != 1 or
                int(row['freeway']) != int(primary_row['freeway']) or
                row['direction'] != primary_row['direction']):
            raise ValueError(f'{split} triple manifest identities differ')
        for manifest in (positive_row, primary_row, row):
            if source_slots(manifest['x_start'], protocol['window'])[-1].isoformat() != \
                    manifest['y_end']:
                raise ValueError(f'{split} triple timestamp layout differs')
        primary_mask = primary_masks[primary_index]
        secondary_mask = secondary_masks[secondary_index]
        if (not np.array_equal(primary_mask, secondary_mask) or
                int(primary_mask.sum()) != int(row['affected_node_count']) or
                int(primary_mask.sum()) != int(primary_row['affected_node_count']) or
                not primary_mask.any()):
            raise ValueError(f'{split} triple affected masks differ')
        values = compute_divergences(
            positive[positive_positions[sample]][:, primary_mask],
            primary[primary_index][:, primary_mask],
            secondary[secondary_index][:, primary_mask], baseline)
        for key in collected:
            collected[key].append(values[key])
        pair_rows.append({
            'split': split, 'secondary_control_index': secondary_index,
            'primary_control_index': primary_index, 'positive_sample_index': sample,
            'incident_id': row['incident_id'], 'positive_t0': row['positive_t0'],
            'primary_candidate_t0': primary_row['candidate_t0'],
            'secondary_candidate_t0': row['candidate_t0'],
            'freeway': int(row['freeway']), 'direction': row['direction'],
            'affected_node_count': int(primary_mask.sum()),
            'positive_iso_week': iso_week(row['positive_t0']),
            'strict_nonoverlap_selected': secondary_index in strict,
            'baseline_incident_divergence': float(
                values['incident_divergence'][baseline].mean()),
            'baseline_routine_divergence': float(
                values['routine_divergence'][baseline].mean()),
            'early_incident_change': float(values['incident_change'][early].mean()),
            'early_routine_change': float(values['routine_change'][early].mean()),
            'early_excess_divergence': float(values['excess'][early].mean()),
            'early_excess_in_train_std': float(values['excess'][early].mean() / train_std),
            'late_incident_change': float(values['incident_change'][late].mean()),
            'late_routine_change': float(values['routine_change'][late].mean()),
            'late_excess_divergence': float(values['excess'][late].mean()),
            'late_excess_in_train_std': float(values['excess'][late].mean() / train_std),
        })
    return pair_rows, {key: np.asarray(value) for key, value in collected.items()}, strict


def population_summary(pair_rows, arrays, selected, protocol, train_std, seed):
    indices = np.asarray([index for index, row in enumerate(pair_rows)
                          if selected(row)], dtype=np.int64)
    if not len(indices):
        raise ValueError('Placebo-audit population is empty')
    early = slice(*protocol['window']['early_Y_slice'])
    late = slice(*protocol['window']['late_Y_slice'])
    clusters = np.asarray([pair_rows[index]['positive_iso_week'] for index in indices])
    components = []
    for horizon in (early, late):
        components.extend([
            arrays['incident_change'][indices, horizon].mean(axis=1),
            arrays['routine_change'][indices, horizon].mean(axis=1),
            arrays['excess'][indices, horizon].mean(axis=1),
        ])
    values = np.column_stack(components)
    low, high = cluster_bootstrap_mean(
        values, clusters, protocol['uncertainty']['draws'],
        protocol['uncertainty']['confidence_level'], seed)
    means = values.mean(axis=0)
    return {
        'triples': int(len(indices)),
        'positive_iso_week_blocks': int(len(set(clusters.tolist()))),
        'early_incident_change_raw': float(means[0]),
        'early_incident_change_ci_low_raw': float(low[0]),
        'early_incident_change_ci_high_raw': float(high[0]),
        'early_routine_change_raw': float(means[1]),
        'early_routine_change_ci_low_raw': float(low[1]),
        'early_routine_change_ci_high_raw': float(high[1]),
        'early_excess_divergence_raw': float(means[2]),
        'early_excess_ci_low_raw': float(low[2]),
        'early_excess_ci_high_raw': float(high[2]),
        'early_excess_in_train_std': float(means[2] / train_std),
        'late_incident_change_raw': float(means[3]),
        'late_incident_change_ci_low_raw': float(low[3]),
        'late_incident_change_ci_high_raw': float(high[3]),
        'late_routine_change_raw': float(means[4]),
        'late_routine_change_ci_low_raw': float(low[4]),
        'late_routine_change_ci_high_raw': float(high[4]),
        'late_excess_divergence_raw': float(means[5]),
        'late_excess_ci_low_raw': float(low[5]),
        'late_excess_ci_high_raw': float(high[5]),
        'late_excess_in_train_std': float(means[5] / train_std),
        'late_excess_ci_low_in_train_std': float(low[5] / train_std),
        'late_excess_ci_high_in_train_std': float(high[5] / train_std),
        'late_excess_positive_fraction': float((values[:, 5] > 0).mean()),
    }, indices, clusters


def trajectory_rows(split, population, arrays, indices, clusters, protocol, seed):
    selected = {key: value[indices] for key, value in arrays.items()}
    bootstrap_keys = [
        'incident_divergence', 'routine_divergence', 'incident_change',
        'routine_change', 'excess',
    ]
    matrix = np.concatenate([selected[key] for key in bootstrap_keys], axis=1)
    low, high = cluster_bootstrap_mean(
        matrix, clusters, protocol['uncertainty']['draws'],
        protocol['uncertainty']['confidence_level'], seed)
    steps = int(protocol['window']['steps'])
    result = []
    for step in range(steps):
        intervals = {
            key: (float(low[offset * steps + step]),
                  float(high[offset * steps + step]))
            for offset, key in enumerate(bootstrap_keys)
        }
        result.append({
            'split': split, 'population': population, 'step_index': step,
            'relative_minutes': int(protocol['window']['first_relative_minute']) +
                                step * int(protocol['window']['step_minutes']),
            'phase': phase_for_step(step, protocol['window']), 'triples': len(indices),
            'incident_mean': float(selected['incident'][:, step].mean()),
            'primary_control_mean': float(selected['primary'][:, step].mean()),
            'secondary_control_mean': float(selected['secondary'][:, step].mean()),
            'incident_divergence': float(
                selected['incident_divergence'][:, step].mean()),
            'incident_divergence_ci_low': intervals['incident_divergence'][0],
            'incident_divergence_ci_high': intervals['incident_divergence'][1],
            'routine_divergence': float(selected['routine_divergence'][:, step].mean()),
            'routine_divergence_ci_low': intervals['routine_divergence'][0],
            'routine_divergence_ci_high': intervals['routine_divergence'][1],
            'baseline_adjusted_incident_divergence': float(
                selected['incident_change'][:, step].mean()),
            'incident_change_ci_low': intervals['incident_change'][0],
            'incident_change_ci_high': intervals['incident_change'][1],
            'baseline_adjusted_routine_divergence': float(
                selected['routine_change'][:, step].mean()),
            'routine_change_ci_low': intervals['routine_change'][0],
            'routine_change_ci_high': intervals['routine_change'][1],
            'excess_divergence': float(selected['excess'][:, step].mean()),
            'excess_divergence_ci_low': intervals['excess'][0],
            'excess_divergence_ci_high': intervals['excess'][1],
        })
    return result


def road_direction_rows(split, pair_rows, arrays, protocol, train_std, seed):
    groups = sorted({(int(row['freeway']), row['direction']) for row in pair_rows})
    result = []
    for offset, (freeway, direction) in enumerate(groups):
        summary, _, _ = population_summary(
            pair_rows, arrays,
            lambda row, f=freeway, d=direction:
                int(row['freeway']) == f and row['direction'] == d,
            protocol, train_std, seed + offset)
        result.append({
            'split': split, 'freeway': freeway, 'direction': direction,
            **{field: summary[field] for field in ROAD_FIELDS
               if field not in ('split', 'freeway', 'direction')},
        })
    return result


def placebo_gate(summaries, protocol):
    gate = protocol['placebo_gate']
    main = {split: summaries[split]['all_common_triples'] for split in SPLITS}
    sensitivity = {split: summaries[split]['strict_nonoverlap'] for split in SPLITS}
    minimum = float(gate['minimum_late_excess_in_train_std'])
    sensitivity_minimum = float(gate['minimum_nonoverlap_late_excess_in_train_std'])
    checks = {
        'minimum_main_late_excess': {
            split: main[split]['late_excess_in_train_std'] >= minimum for split in SPLITS
        },
        'main_block_ci_lower_above_zero': {
            split: main[split]['late_excess_ci_low_raw'] > 0 for split in SPLITS
        },
        'minimum_nonoverlap_late_excess': {
            split: sensitivity[split]['late_excess_in_train_std'] >= sensitivity_minimum
            for split in SPLITS
        },
        'nonoverlap_late_excess_positive': {
            split: sensitivity[split]['late_excess_divergence_raw'] > 0 for split in SPLITS
        },
        'minimum_nonoverlap_triple_count': {
            split: sensitivity[split]['triples'] >= int(
                protocol['nonoverlap_sensitivity']['minimum_selected_triples'][split])
            for split in SPLITS
        },
    }
    ready = all(all(values.values()) for values in checks.values())
    return checks, ready


def audit(data_dir, primary_dir, secondary_dir, protocol_path, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError('Placebo-audit output exists; preserve it and use a new directory')
    protocol = load_protocol(protocol_path)
    input_hashes = verify_inputs(data_dir, primary_dir, secondary_dir, protocol)
    scaler = json.loads((Path(data_dir) / 'scaler.json').read_text(encoding='utf-8'))
    train_std = float(scaler['std'])
    pair_output, trajectory_output, road_output = [], [], []
    summaries, strict_counts = {}, {}
    seed = int(protocol['uncertainty']['seed'])
    for split_number, split in enumerate(SPLITS):
        pair_rows, arrays, strict = validate_and_extract_split(
            data_dir, primary_dir, secondary_dir, split, protocol)
        all_summary, all_indices, all_clusters = population_summary(
            pair_rows, arrays, lambda row: True, protocol, train_std,
            seed + split_number)
        strict_summary, strict_indices, strict_clusters = population_summary(
            pair_rows, arrays, lambda row: bool(row['strict_nonoverlap_selected']),
            protocol, train_std, seed + 10 + split_number)
        summaries[split] = {
            'all_common_triples': all_summary, 'strict_nonoverlap': strict_summary,
        }
        strict_counts[split] = len(strict)
        pair_output.extend(pair_rows)
        trajectory_output.extend(trajectory_rows(
            split, 'all_common_triples', arrays, all_indices, all_clusters,
            protocol, seed + split_number))
        trajectory_output.extend(trajectory_rows(
            split, 'strict_nonoverlap', arrays, strict_indices, strict_clusters,
            protocol, seed + 10 + split_number))
        road_output.extend(road_direction_rows(
            split, pair_rows, arrays, protocol, train_std,
            seed + 20 + split_number * 10))
    checks, ready = placebo_gate(summaries, protocol)
    output.mkdir(parents=True)
    names = ['triple_metrics.csv', 'trajectory.csv', 'road_direction.csv']
    write_csv(output / names[0], pair_output, TRIPLE_FIELDS)
    write_csv(output / names[1], trajectory_output, TRAJECTORY_FIELDS)
    write_csv(output / names[2], road_output, ROAD_FIELDS)
    summary = {
        'status': 'MATCHED_PLACEBO_AUDIT_COMPLETE', 'scope': protocol['scope'],
        'protocol_id': protocol['protocol_id'], 'protocol_sha256': sha256(protocol_path),
        'main_training_ready': False,
        'heterogeneity_screening_ready': ready,
        'traffic_X_read': True, 'forecast_Y_read_after_both_assignments': True,
        'forecast_Y_used_to_change_matching_or_common_cohort': False,
        'test_split_read': False, 'model_training_performed': False,
        'model_predictions_read': False, 'event_equal_weighting': True,
        'estimand': protocol['estimand'], 'uncertainty': protocol['uncertainty'],
        'split_results': summaries, 'strict_nonoverlap_selected_triples': strict_counts,
        'placebo_gate': checks, 'terminology': protocol['terminology'],
        'limitations': [
            'The excess-divergence contrast is observational, not a causal treatment effect.',
            'The common-triple cohort excludes primary pairs without a second eligible control.',
            ('Main intervals cluster by positive incident ISO week; control-side dependence is '
             'also examined in the strict three-window no-shared-slot subset.'),
            'The strict validation subset and validation SR4 strata are small.',
            'Recorded-incident-free status is local to the positive affected-node set.',
        ],
        'inputs': {
            **{f'positive_{key}': value for key, value in input_hashes[0].items()},
            **{f'primary_{key}': value for key, value in input_hashes[1].items()},
            **{f'secondary_{key}': value for key, value in input_hashes[2].items()},
            'protocol_sha256': sha256(protocol_path), 'code_sha256': sha256(__file__),
        },
    }
    summary['outputs'] = {
        name: {'sha256': sha256(output / name), 'bytes': (output / name).stat().st_size}
        for name in names
    }
    (output / 'summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
        encoding='utf-8')
    print(json.dumps({key: value for key, value in summary.items()
                      if key not in ('inputs', 'outputs')},
                     ensure_ascii=False, indent=2, allow_nan=False), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--primary-control-dir', type=Path, required=True)
    parser.add_argument('--secondary-control-dir', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, default=Path(__file__).with_name(
        'matched_placebo_audit_v5c.json'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    audit(args.data_dir, args.primary_control_dir, args.secondary_control_dir,
          args.protocol, args.output)


if __name__ == '__main__':
    main()
