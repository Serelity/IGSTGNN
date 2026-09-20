"""Audit post-assignment incident/control outcome contrasts before paired training."""

import argparse
from datetime import datetime, timedelta
import json
from pathlib import Path

import numpy as np

from experiments.chronological.audit_matched_controls import read_csv, sha256, write_csv
from experiments.chronological.score_matched_controls import SPLITS, ordered_sensors


PAIR_FIELDS = [
    'split', 'control_index', 'positive_sample_index', 'incident_id',
    'positive_t0', 'candidate_t0', 'freeway', 'direction', 'x_distance',
    'affected_node_count', 'positive_iso_week', 'nonoverlap_selected',
    'pre_incident_mean', 'pre_control_mean', 'pre_signed_difference',
    'pre_pair_mae', 'early_y_incident_mean', 'early_y_control_mean',
    'early_y_signed_difference', 'early_y_pair_mae',
    'early_y_change_contrast', 'late_y_incident_mean',
    'late_y_control_mean', 'late_y_signed_difference', 'late_y_pair_mae',
    'late_y_change_contrast', 'late_y_change_in_train_std',
]

TRAJECTORY_FIELDS = [
    'split', 'population', 'step_index', 'relative_minutes', 'phase',
    'pairs', 'affected_node_values', 'incident_mean', 'control_mean',
    'signed_difference', 'signed_difference_ci_low',
    'signed_difference_ci_high', 'mean_pair_absolute_difference',
    'mean_pair_absolute_difference_ci_low',
    'mean_pair_absolute_difference_ci_high',
    'baseline_adjusted_change_contrast', 'change_contrast_ci_low',
    'change_contrast_ci_high',
]

BAND_FIELDS = [
    'split', 'distance_band', 'step_index', 'relative_minutes', 'phase',
    'pairs_with_nodes', 'node_values', 'incident_mean', 'control_mean',
    'signed_difference', 'mean_pair_absolute_difference',
    'baseline_adjusted_change_contrast',
]

ROAD_FIELDS = [
    'split', 'freeway', 'direction', 'pairs', 'positive_iso_week_blocks',
    'pre_pair_mae_raw', 'early_y_change_contrast_raw',
    'early_y_change_contrast_ci_low_raw', 'early_y_change_contrast_ci_high_raw',
    'early_y_change_contrast_in_train_std', 'late_y_change_contrast_raw',
    'late_y_change_contrast_ci_low_raw', 'late_y_change_contrast_ci_high_raw',
    'late_y_change_contrast_in_train_std',
]


def load_protocol(path):
    protocol = json.loads(Path(path).read_text(encoding='utf-8'))
    if protocol.get('scope') != 'matched_observational_outcome_audit':
        raise ValueError('Protocol scope must remain matched observational outcome audit')
    if protocol.get('main_training_ready') is not False:
        raise ValueError('The outcome audit cannot pre-authorize main training')
    if (int(protocol.get('source_year', 0)) != 2023 or
            int(protocol.get('source_version', 0)) != 8 or
            int(protocol.get('expected_sensor_count', 0)) != 496):
        raise ValueError('v4 is frozen to the 2023 source-v8 Contra496 package')
    window = protocol['window']
    expected = {
        'steps': 26, 'step_minutes': 5, 'first_relative_minute': -65,
        'X_slice': [0, 12], 'excluded_latency_slice': [12, 14],
        'Y_slice': [14, 26], 'baseline_slice': [9, 12],
        'early_Y_slice': [14, 20], 'late_Y_slice': [20, 26],
    }
    if any(window.get(key) != value for key, value in expected.items()):
        raise ValueError('v4 window, baseline, or horizon definition changed')
    boundary = protocol['information_boundary']
    required = [
        'assignment_frozen_before_outcome_audit', 'outcomes_may_not_change_matching',
        'train_and_validation_only', 'test_split_prohibited',
        'model_training_prohibited', 'model_predictions_prohibited',
    ]
    if not all(boundary.get(key) is True for key in required):
        raise ValueError('v4 information boundary changed')
    estimand = protocol['estimand']
    if (estimand.get('event_equal_weighting') is not True or
            estimand.get('causal_effect_claimed') is not False):
        raise ValueError('v4 must remain event-weighted and observational')
    uncertainty = protocol['uncertainty']
    if (uncertainty.get('method') != 'positive_incident_iso_week_cluster_bootstrap' or
            int(uncertainty.get('draws', 0)) < 1000 or
            not 0 < float(uncertainty.get('confidence_level', 0)) < 1):
        raise ValueError('v4 uncertainty protocol changed or is underpowered')
    sensitivity = protocol['nonoverlap_sensitivity']
    if (sensitivity.get('selection_order') !=
            'positive_t0_then_positive_sample_index' or
            sensitivity.get('all_positive_and_control_source_slots_share_one_exclusion_set')
            is not True or sensitivity.get('outcomes_used_for_selection') is not False):
        raise ValueError('v4 non-overlap selection boundary changed')
    bands = protocol['distance_bands_miles']
    if [(item['name'], item['lower'], item['upper'], item['include_lower'])
            for item in bands] != [
                ('d00_03', 0.0, 3.0, True),
                ('d03_06', 3.0, 6.0, False),
                ('d06_10', 6.0, 10.0, False)]:
        raise ValueError('v4 distance bands changed')
    descriptive = protocol['descriptive_outputs']
    if not all(descriptive.get(key) is True for key in (
            'early_Y_block_interval_reported',
            'road_direction_stratification_reported',
            'descriptive_results_may_not_change_the_primary_gate')):
        raise ValueError('v4 descriptive reporting boundary changed')
    gate = protocol['phenomenon_gate']
    if (float(gate['minimum_absolute_late_change_in_train_std']) <= 0 or
            float(gate['minimum_nonoverlap_absolute_late_change_in_train_std']) <= 0):
        raise ValueError('v4 phenomenon thresholds must be positive')
    return protocol


def verify_inputs(data_dir, control_dir, sensors_path, protocol):
    data_dir, control_dir = Path(data_dir), Path(control_dir)
    positive_paths = {
        'summary.json': data_dir / 'summary.json',
        'scaler.json': data_dir / 'scaler.json',
        'station_ids.npy': data_dir / 'station_ids.npy',
        'train_flow.npy': data_dir / 'train_flow.npy',
        'val_flow.npy': data_dir / 'val_flow.npy',
        'train_manifest.csv': data_dir / 'train_manifest.csv',
        'val_manifest.csv': data_dir / 'val_manifest.csv',
        'sensors.csv': Path(sensors_path),
    }
    positive_hashes = {name: sha256(path) for name, path in positive_paths.items()}
    if positive_hashes != protocol['positive_inputs']:
        raise ValueError('Positive package or sensor fingerprints differ from v4 protocol')
    control_paths = {
        name: control_dir / name for name in protocol['control_inputs']
        if name not in ('protocol_id', 'expected_shapes')
    }
    control_hashes = {name: sha256(path) for name, path in control_paths.items()}
    expected_control = {name: value for name, value in protocol['control_inputs'].items()
                        if name not in ('protocol_id', 'expected_shapes')}
    if control_hashes != expected_control:
        raise ValueError('Materialized-control fingerprints differ from v4 protocol')
    positive_summary = json.loads((data_dir / 'summary.json').read_text(encoding='utf-8'))
    control_summary = json.loads((control_dir / 'summary.json').read_text(encoding='utf-8'))
    if (positive_summary.get('source_version') != 8 or
            positive_summary.get('test_flow_built') is not False or
            positive_summary.get('split_counts', {}).get('test') is not None):
        raise ValueError('Positive package violates the frozen development-only boundary')
    if (control_summary.get('status') != 'MATCHED_NONINCIDENT_MATERIALIZATION_PASS' or
            control_summary.get('protocol_id') != protocol['control_inputs']['protocol_id'] or
            control_summary.get('test_split_read') is not False or
            control_summary.get('future_Y_used_to_rank_or_replace_controls') is not False):
        raise ValueError('Control materialization status or information boundary differs')
    return positive_hashes, control_hashes


def phase_for_step(step, window):
    if window['X_slice'][0] <= step < window['X_slice'][1]:
        return 'X'
    if window['excluded_latency_slice'][0] <= step < window['excluded_latency_slice'][1]:
        return 'excluded_latency'
    if window['Y_slice'][0] <= step < window['Y_slice'][1]:
        return 'Y'
    raise ValueError('Step lies outside the frozen 26-slot window')


def source_slots(x_start, window):
    start = datetime.fromisoformat(x_start)
    if start.tzinfo is not None or start.minute % 5 or start.second or start.microsecond:
        raise ValueError('Window start must be a naive five-minute nominal timestamp')
    return tuple(start + timedelta(minutes=int(window['step_minutes']) * step)
                 for step in range(int(window['steps'])))


def select_nonoverlap(control_rows, positive_by_sample, window):
    """Greedy, outcome-blind subset with no source slot reused on either side."""
    ordered = sorted(control_rows, key=lambda row: (
        row['positive_t0'], int(row['positive_sample_index'])))
    used, selected = set(), set()
    for row in ordered:
        sample = int(row['positive_sample_index'])
        if sample not in positive_by_sample:
            raise ValueError('Control references an unknown positive sample')
        slots = set(source_slots(positive_by_sample[sample]['x_start'], window))
        slots.update(source_slots(row['x_start'], window))
        if slots & used:
            continue
        selected.add(int(row['control_index']))
        used.update(slots)
    return selected


def iso_week(value):
    year, week, _ = datetime.fromisoformat(value).isocalendar()
    return f'{year}-W{week:02d}'


def cluster_bootstrap_mean(values, clusters, draws, confidence, seed):
    """Cluster bootstrap an event-equal mean; columns may contain missing strata."""
    values = np.asarray(values, dtype=np.float64)
    one_dimensional = values.ndim == 1
    if one_dimensional:
        values = values[:, None]
    clusters = np.asarray(clusters)
    if values.ndim != 2 or len(values) != len(clusters) or not len(values):
        raise ValueError('Bootstrap values and cluster labels must be non-empty and aligned')
    labels = sorted(set(clusters.tolist()))
    if len(labels) < 2:
        raise ValueError('Cluster bootstrap requires at least two time blocks')
    sums, counts = [], []
    for label in labels:
        block = values[clusters == label]
        valid = np.isfinite(block)
        sums.append(np.where(valid, block, 0).sum(axis=0))
        counts.append(valid.sum(axis=0))
    sums, counts = np.asarray(sums), np.asarray(counts)
    rng = np.random.default_rng(int(seed))
    weights = rng.multinomial(len(labels), np.full(len(labels), 1 / len(labels)),
                              size=int(draws))
    totals = weights @ sums
    denominators = weights @ counts
    estimates = np.divide(totals, denominators, out=np.full_like(totals, np.nan),
                          where=denominators > 0)
    alpha = (1 - float(confidence)) / 2
    low = np.nanquantile(estimates, alpha, axis=0)
    high = np.nanquantile(estimates, 1 - alpha, axis=0)
    if one_dimensional:
        return float(low[0]), float(high[0])
    return low, high


def validate_and_extract_split(data_dir, control_dir, sensors_path, split, protocol):
    station_ids = np.load(Path(data_dir) / 'station_ids.npy', allow_pickle=False)
    expected_shape = tuple(protocol['control_inputs']['expected_shapes'][split])
    positive_rows = read_csv(Path(data_dir) / f'{split}_manifest.csv')
    positive_by_sample = {int(row['sample_index']): row for row in positive_rows}
    if len(positive_by_sample) != len(positive_rows):
        raise ValueError(f'{split} positive manifest contains duplicate sample indices')
    positive_positions = {int(row['sample_index']): index
                          for index, row in enumerate(positive_rows)}
    control_rows = read_csv(Path(control_dir) / f'{split}_control_manifest.csv')
    positive = np.load(Path(data_dir) / f'{split}_flow.npy', mmap_mode='r', allow_pickle=False)
    control = np.load(Path(control_dir) / f'{split}_control_flow.npy', mmap_mode='r',
                      allow_pickle=False)
    masks = np.load(Path(control_dir) / f'{split}_affected_mask.npy', allow_pickle=False)
    if (control.shape != expected_shape or masks.shape != (expected_shape[0], expected_shape[2]) or
            masks.dtype != np.bool_ or positive.shape !=
            (len(positive_rows), expected_shape[1], expected_shape[2]) or
            station_ids.shape != (expected_shape[2],)):
        raise ValueError(f'{split} array shapes or affected-mask dtype differ from v4 protocol')
    roads, directions, postmiles = ordered_sensors(sensors_path, station_ids)
    nonoverlap = select_nonoverlap(control_rows, positive_by_sample, protocol['window'])
    event_incident, event_control, event_signed, event_absolute = [], [], [], []
    pair_rows, band_values = [], {item['name']: [] for item in protocol['distance_bands_miles']}
    window = protocol['window']
    baseline = slice(*window['baseline_slice'])
    for position, row in enumerate(control_rows):
        index = int(row['control_index'])
        sample = int(row['positive_sample_index'])
        if index != position or sample not in positive_positions:
            raise ValueError(f'{split} control order or positive identity is inconsistent')
        positive_row = positive_by_sample[sample]
        if (row['split'] != split or positive_row['split'] != split or
                row['incident_id'] != positive_row['incident_id'] or
                row['positive_t0'] != positive_row['t0']):
            raise ValueError(f'{split} paired manifest identities differ')
        if (source_slots(positive_row['x_start'], window)[-1].isoformat() !=
                positive_row['y_end'] or
                source_slots(row['x_start'], window)[-1].isoformat() != row['y_end']):
            raise ValueError(f'{split} paired window timestamps differ from the 26-slot layout')
        mask = masks[index]
        if int(mask.sum()) != int(row['affected_node_count']) or not mask.any():
            raise ValueError(f'{split} affected mask and manifest disagree')
        incident_values = np.asarray(
            positive[positive_positions[sample]][:, mask], dtype=np.float64)
        control_values = np.asarray(control[index][:, mask], dtype=np.float64)
        if (not np.isfinite(incident_values).all() or not np.isfinite(control_values).all() or
                (incident_values < 0).any() or (control_values < 0).any()):
            raise ValueError(f'{split} paired affected-node values must be finite and nonnegative')
        incident_mean = incident_values.mean(axis=1)
        control_mean = control_values.mean(axis=1)
        signed = incident_mean - control_mean
        absolute = np.abs(incident_values - control_values).mean(axis=1)
        change = signed - signed[baseline].mean()
        event_incident.append(incident_mean)
        event_control.append(control_mean)
        event_signed.append(signed)
        event_absolute.append(absolute)
        x_slice = slice(*window['X_slice'])
        early = slice(*window['early_Y_slice'])
        late = slice(*window['late_Y_slice'])
        pair_rows.append({
            'split': split, 'control_index': index, 'positive_sample_index': sample,
            'incident_id': row['incident_id'], 'positive_t0': row['positive_t0'],
            'candidate_t0': row['candidate_t0'], 'freeway': row['freeway'],
            'direction': row['direction'], 'x_distance': float(row['x_distance']),
            'affected_node_count': int(mask.sum()),
            'positive_iso_week': iso_week(row['positive_t0']),
            'nonoverlap_selected': index in nonoverlap,
            'pre_incident_mean': float(incident_mean[x_slice].mean()),
            'pre_control_mean': float(control_mean[x_slice].mean()),
            'pre_signed_difference': float(signed[x_slice].mean()),
            'pre_pair_mae': float(absolute[x_slice].mean()),
            'early_y_incident_mean': float(incident_mean[early].mean()),
            'early_y_control_mean': float(control_mean[early].mean()),
            'early_y_signed_difference': float(signed[early].mean()),
            'early_y_pair_mae': float(absolute[early].mean()),
            'early_y_change_contrast': float(change[early].mean()),
            'late_y_incident_mean': float(incident_mean[late].mean()),
            'late_y_control_mean': float(control_mean[late].mean()),
            'late_y_signed_difference': float(signed[late].mean()),
            'late_y_pair_mae': float(absolute[late].mean()),
            'late_y_change_contrast': float(change[late].mean()),
        })
        road, direction, event_pm = (int(row['freeway']), row['direction'],
                                     float(row['incident_postmile']))
        same_road = (roads == road) & (directions == direction)
        distance = np.abs(postmiles - event_pm)
        covered = np.zeros(len(mask), dtype=bool)
        for band in protocol['distance_bands_miles']:
            lower = distance >= float(band['lower']) if band['include_lower'] else (
                distance > float(band['lower']))
            band_mask = same_road & lower & (distance <= float(band['upper']))
            covered |= band_mask
            if not band_mask.any():
                band_values[band['name']].append(None)
                continue
            band_incident = np.asarray(
                positive[positive_positions[sample]][:, band_mask], dtype=np.float64)
            band_control = np.asarray(control[index][:, band_mask], dtype=np.float64)
            band_signed = band_incident.mean(axis=1) - band_control.mean(axis=1)
            band_values[band['name']].append({
                'incident': band_incident.mean(axis=1),
                'control': band_control.mean(axis=1),
                'signed': band_signed,
                'absolute': np.abs(band_incident - band_control).mean(axis=1),
                'change': band_signed - band_signed[baseline].mean(),
                'nodes': int(band_mask.sum()),
            })
        if not np.array_equal(covered, mask):
            raise ValueError(f'{split} distance bands do not exactly partition the affected mask')
    scaler = json.loads((Path(data_dir) / 'scaler.json').read_text(encoding='utf-8'))
    train_std = float(scaler['std'])
    for row in pair_rows:
        row['late_y_change_in_train_std'] = row['late_y_change_contrast'] / train_std
    arrays = {
        'incident': np.asarray(event_incident), 'control': np.asarray(event_control),
        'signed': np.asarray(event_signed), 'absolute': np.asarray(event_absolute),
    }
    baseline_mean = arrays['signed'][:, baseline].mean(axis=1, keepdims=True)
    arrays['change'] = arrays['signed'] - baseline_mean
    return pair_rows, arrays, band_values, nonoverlap


def population_summary(pair_rows, arrays, selected, protocol, train_std, seed):
    indices = np.asarray([index for index, row in enumerate(pair_rows)
                          if selected(row)], dtype=np.int64)
    if not len(indices):
        raise ValueError('Outcome population is empty')
    window = protocol['window']
    x_slice = slice(*window['X_slice'])
    early = slice(*window['early_Y_slice'])
    late = slice(*window['late_Y_slice'])
    clusters = np.asarray([pair_rows[index]['positive_iso_week'] for index in indices])
    late_change = arrays['change'][indices, late].mean(axis=1)
    early_change = arrays['change'][indices, early].mean(axis=1)
    interval_low, interval_high = cluster_bootstrap_mean(
        np.column_stack([early_change, late_change]), clusters,
        protocol['uncertainty']['draws'],
        protocol['uncertainty']['confidence_level'], seed)
    early_low, low = map(float, interval_low)
    early_high, high = map(float, interval_high)
    pre_mae = arrays['absolute'][indices, x_slice].mean(axis=1)
    early_mae = arrays['absolute'][indices, early].mean(axis=1)
    late_mae = arrays['absolute'][indices, late].mean(axis=1)
    if not np.isfinite(train_std) or train_std <= 0:
        raise ValueError('Training standard deviation must be finite and positive')
    return {
        'pairs': int(len(indices)), 'positive_iso_week_blocks': int(len(set(clusters.tolist()))),
        'affected_node_count_q0_q25_q50_q75_q100': np.quantile(
            [pair_rows[index]['affected_node_count'] for index in indices],
            [0, .25, .5, .75, 1], method='nearest').astype(int).tolist(),
        'pre_pair_mae_raw': float(pre_mae.mean()),
        'early_y_pair_mae_raw': float(early_mae.mean()),
        'late_y_pair_mae_raw': float(late_mae.mean()),
        'early_to_pre_pair_mae_ratio': float(early_mae.mean() / pre_mae.mean()),
        'late_to_pre_pair_mae_ratio': float(late_mae.mean() / pre_mae.mean()),
        'early_y_change_contrast_raw': float(early_change.mean()),
        'early_y_change_contrast_ci_low_raw': early_low,
        'early_y_change_contrast_ci_high_raw': early_high,
        'early_y_change_contrast_in_train_std': float(early_change.mean() / train_std),
        'late_y_change_contrast_raw': float(late_change.mean()),
        'late_y_change_contrast_ci_low_raw': low,
        'late_y_change_contrast_ci_high_raw': high,
        'late_y_change_contrast_in_train_std': float(late_change.mean() / train_std),
        'late_y_change_ci_low_in_train_std': float(low / train_std),
        'late_y_change_ci_high_in_train_std': float(high / train_std),
        'late_y_change_positive_fraction': float((late_change > 0).mean()),
    }, indices, clusters


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


def trajectory_rows(split, population, pair_rows, arrays, indices, clusters, protocol, seed):
    selected = {name: values[indices] for name, values in arrays.items()}
    matrix = np.concatenate([
        selected['signed'], selected['absolute'], selected['change']], axis=1)
    low, high = cluster_bootstrap_mean(
        matrix, clusters, protocol['uncertainty']['draws'],
        protocol['uncertainty']['confidence_level'], seed)
    steps = int(protocol['window']['steps'])
    result = []
    node_values = sum(pair_rows[index]['affected_node_count'] for index in indices)
    for step in range(steps):
        result.append({
            'split': split, 'population': population, 'step_index': step,
            'relative_minutes': int(protocol['window']['first_relative_minute']) +
                                step * int(protocol['window']['step_minutes']),
            'phase': phase_for_step(step, protocol['window']),
            'pairs': len(indices), 'affected_node_values': node_values,
            'incident_mean': float(selected['incident'][:, step].mean()),
            'control_mean': float(selected['control'][:, step].mean()),
            'signed_difference': float(selected['signed'][:, step].mean()),
            'signed_difference_ci_low': float(low[step]),
            'signed_difference_ci_high': float(high[step]),
            'mean_pair_absolute_difference': float(selected['absolute'][:, step].mean()),
            'mean_pair_absolute_difference_ci_low': float(low[steps + step]),
            'mean_pair_absolute_difference_ci_high': float(high[steps + step]),
            'baseline_adjusted_change_contrast': float(selected['change'][:, step].mean()),
            'change_contrast_ci_low': float(low[2 * steps + step]),
            'change_contrast_ci_high': float(high[2 * steps + step]),
        })
    return result


def distance_band_rows(split, band_values, protocol):
    result = []
    for name, events in band_values.items():
        present = [event for event in events if event is not None]
        if not present:
            continue
        for step in range(int(protocol['window']['steps'])):
            result.append({
                'split': split, 'distance_band': name, 'step_index': step,
                'relative_minutes': int(protocol['window']['first_relative_minute']) +
                                    step * int(protocol['window']['step_minutes']),
                'phase': phase_for_step(step, protocol['window']),
                'pairs_with_nodes': len(present),
                'node_values': sum(event['nodes'] for event in present),
                'incident_mean': float(np.mean([event['incident'][step] for event in present])),
                'control_mean': float(np.mean([event['control'][step] for event in present])),
                'signed_difference': float(np.mean([event['signed'][step] for event in present])),
                'mean_pair_absolute_difference': float(np.mean(
                    [event['absolute'][step] for event in present])),
                'baseline_adjusted_change_contrast': float(np.mean(
                    [event['change'][step] for event in present])),
            })
    return result


def sign(value):
    return 1 if value > 0 else (-1 if value < 0 else 0)


def phenomenon_gate(summaries, protocol):
    gate = protocol['phenomenon_gate']
    main = {split: summaries[split]['all_matched'] for split in SPLITS}
    sensitivity = {split: summaries[split]['nonoverlap'] for split in SPLITS}
    minimum = float(gate['minimum_absolute_late_change_in_train_std'])
    minimum_nonoverlap = float(
        gate['minimum_nonoverlap_absolute_late_change_in_train_std'])
    magnitude = {split: abs(main[split]['late_y_change_contrast_in_train_std']) >= minimum
                 for split in SPLITS}
    ci = {split: (main[split]['late_y_change_contrast_ci_low_raw'] > 0 or
                  main[split]['late_y_change_contrast_ci_high_raw'] < 0)
          for split in SPLITS}
    main_direction = (sign(main['train']['late_y_change_contrast_raw']) != 0 and
                      sign(main['train']['late_y_change_contrast_raw']) ==
                      sign(main['val']['late_y_change_contrast_raw']))
    nonoverlap_direction = {
        split: (sign(sensitivity[split]['late_y_change_contrast_raw']) ==
                sign(main[split]['late_y_change_contrast_raw']) and
                abs(sensitivity[split]['late_y_change_contrast_in_train_std']) >=
                minimum_nonoverlap)
        for split in SPLITS
    }
    minimum_count = {
        split: sensitivity[split]['pairs'] >= int(
            protocol['nonoverlap_sensitivity']['minimum_selected_pairs'][split])
        for split in SPLITS
    }
    checks = {
        'minimum_main_effect_magnitude': magnitude,
        'main_block_ci_excludes_zero': ci,
        'main_direction_consistent_across_splits': main_direction,
        'nonoverlap_direction_and_magnitude_consistent': nonoverlap_direction,
        'minimum_nonoverlap_pair_count': minimum_count,
    }
    ready = (all(magnitude.values()) and all(ci.values()) and main_direction and
             all(nonoverlap_direction.values()) and all(minimum_count.values()))
    return checks, ready


def audit(data_dir, control_dir, sensors_path, protocol_path, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError('Outcome-audit output exists; preserve it and use a new directory')
    protocol = load_protocol(protocol_path)
    positive_hashes, control_hashes = verify_inputs(
        data_dir, control_dir, sensors_path, protocol)
    pair_output, trajectory_output, band_output, road_output = [], [], [], []
    summaries, nonoverlap_counts = {}, {}
    base_seed = int(protocol['uncertainty']['seed'])
    scaler = json.loads((Path(data_dir) / 'scaler.json').read_text(encoding='utf-8'))
    train_std = float(scaler['std'])
    for split_number, split in enumerate(SPLITS):
        pair_rows, arrays, bands, nonoverlap = validate_and_extract_split(
            data_dir, control_dir, sensors_path, split, protocol)
        all_summary, all_indices, all_clusters = population_summary(
            pair_rows, arrays, lambda row: True, protocol, train_std,
            base_seed + split_number)
        nonoverlap_summary, sensitivity_indices, sensitivity_clusters = population_summary(
            pair_rows, arrays, lambda row: bool(row['nonoverlap_selected']),
            protocol, train_std, base_seed + 10 + split_number)
        summaries[split] = {
            'all_matched': all_summary, 'nonoverlap': nonoverlap_summary,
        }
        nonoverlap_counts[split] = len(nonoverlap)
        pair_output.extend(pair_rows)
        trajectory_output.extend(trajectory_rows(
            split, 'all_matched', pair_rows, arrays, all_indices, all_clusters,
            protocol, base_seed + split_number))
        trajectory_output.extend(trajectory_rows(
            split, 'nonoverlap', pair_rows, arrays, sensitivity_indices,
            sensitivity_clusters, protocol, base_seed + 10 + split_number))
        band_output.extend(distance_band_rows(split, bands, protocol))
        road_output.extend(road_direction_rows(
            split, pair_rows, arrays, protocol, train_std,
            base_seed + 20 + split_number * 10))
    checks, benchmark_ready = phenomenon_gate(summaries, protocol)
    output.mkdir(parents=True)
    names = ['pair_metrics.csv', 'trajectory.csv', 'distance_bands.csv',
             'road_direction.csv']
    write_csv(output / names[0], pair_output, PAIR_FIELDS)
    write_csv(output / names[1], trajectory_output, TRAJECTORY_FIELDS)
    write_csv(output / names[2], band_output, BAND_FIELDS)
    write_csv(output / names[3], road_output, ROAD_FIELDS)
    positive_counts = {
        split: len(read_csv(Path(data_dir) / f'{split}_manifest.csv')) for split in SPLITS
    }
    matched_coverage = {
        split: summaries[split]['all_matched']['pairs'] / positive_counts[split]
        for split in SPLITS
    }
    summary = {
        'status': 'MATCHED_OUTCOME_AUDIT_COMPLETE',
        'scope': protocol['scope'], 'protocol_id': protocol['protocol_id'],
        'protocol_sha256': sha256(protocol_path),
        'main_training_ready': False,
        'paired_benchmark_ready': benchmark_ready,
        'traffic_X_read': True, 'forecast_Y_read_after_assignment': True,
        'forecast_Y_used_to_change_matching': False,
        'test_split_read': False, 'model_predictions_read': False,
        'event_equal_weighting': True,
        'uncertainty': protocol['uncertainty'],
        'split_results': summaries,
        'development_positive_samples': positive_counts,
        'matched_population_coverage': matched_coverage,
        'nonoverlap_selected_pairs': nonoverlap_counts,
        'phenomenon_gate': checks,
        'terminology': protocol['terminology'],
        'limitations': [
            'The matched change contrast is observational and is not a causal treatment effect.',
            'Recorded-incident-free status is local to the positive affected-node set.',
            ('The main interval clusters by positive incident ISO week; control-side dependence '
             'is addressed by a strict no-shared-source-slot sensitivity subset, not claimed '
             'absent.'),
            ('Validation SR4-E matching coverage is limited; later benchmarks must retain '
             'road-direction coverage reporting.'),
            'Distance bands are descriptive and do not encode upstream/downstream traffic physics.',
        ],
        'inputs': {
            **{f'positive_{key}': value for key, value in positive_hashes.items()},
            **{f'control_{key}': value for key, value in control_hashes.items()},
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
    parser.add_argument('--control-dir', type=Path, required=True)
    parser.add_argument('--sensors', type=Path, required=True)
    parser.add_argument('--protocol', type=Path,
                        default=Path(__file__).with_name('matched_outcome_audit_v4.json'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    audit(args.data_dir, args.control_dir, args.sensors, args.protocol, args.output)


if __name__ == '__main__':
    main()
