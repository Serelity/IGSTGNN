"""Materialize frozen matched controls after selection, without using Y to reselect them."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.chronological.audit_matched_controls import read_csv, sha256, write_csv
from experiments.chronological.score_matched_controls import (
    SPLITS, VerifiedMonthCache, ordered_sensors, score_histories, source_slot,
)


MANIFEST_FIELDS = [
    'control_index', 'split', 'positive_sample_index', 'incident_id', 'positive_t0',
    'candidate_t0', 'source_version', 'x_start', 'x_end',
    'excluded_latency_start', 'excluded_latency_end', 'y_start', 'y_end',
    'support_start', 'support_end_exclusive', 'freeway', 'direction',
    'incident_postmile', 'affected_node_count', 'x_distance',
    'candidate_preference_rank',
]
FLOAT_SCORE_FIELDS = [
    'pairwise_valid_overlap_fraction', 'missing_pattern_mismatch_fraction', 'x_distance',
    'positive_history_mean', 'control_history_mean',
    'positive_last_step_mean', 'control_last_step_mean',
    'positive_late3_minus_early3', 'control_late3_minus_early3',
]


def load_protocol(path):
    protocol = json.loads(Path(path).read_text(encoding='utf-8'))
    if protocol.get('scope') != 'post_assignment_full_window_materialization':
        raise ValueError('Protocol scope must remain post-assignment materialization')
    if protocol.get('main_training_ready') is not False:
        raise ValueError('Materialization cannot declare main training readiness')
    if (int(protocol.get('source_year', 0)) != 2023 or
            int(protocol.get('source_version', 0)) != 8 or
            int(protocol.get('expected_sensor_count', 0)) != 496):
        raise ValueError('v3 is frozen to the 2023 source-v8 Contra496 package')
    window = protocol['window']
    if (int(window['start_minutes_from_candidate_t0']) != -65 or
            int(window['steps']) != 26 or int(window['step_minutes']) != 5 or
            window['X_slice'] != [0, 12] or
            window['excluded_latency_slice'] != [12, 14] or
            window['Y_slice'] != [14, 26] or window['dtype'] != 'float32' or
            window.get('raw_missing_preserved') is not True):
        raise ValueError('v3 full-window layout or raw-value policy changed')
    boundary = protocol['selection_boundary']
    required_true = [
        'assignment_frozen_before_materialization',
        'future_Y_may_be_materialized_after_assignment',
        'test_split_prohibited', 'incident_description_prohibited',
        'incident_type_prohibited',
    ]
    if not all(boundary.get(key) is True for key in required_true):
        raise ValueError('v3 selection and prohibited-input boundary changed')
    if boundary.get('future_Y_may_change_assignment') is not False:
        raise ValueError('Future Y must never alter the frozen assignment')
    expected_months = {'train': list(range(1, 9)), 'val': [9, 10]}
    if protocol.get('split_months') != expected_months:
        raise ValueError('v3 chronological split months changed')
    if float(protocol['assignment_reproduction']['maximum_absolute_float_difference']) < 0:
        raise ValueError('Score-reproduction tolerance must be nonnegative')
    return protocol


def verify_inputs(data_dir, assignment_dir, sensors_path, protocol):
    data_dir, assignment_dir = Path(data_dir), Path(assignment_dir)
    data_actual = {
        'summary_sha256': sha256(data_dir / 'summary.json'),
        'scaler_sha256': sha256(data_dir / 'scaler.json'),
        'station_ids_sha256': sha256(data_dir / 'station_ids.npy'),
        'raw_node_indices_sha256': sha256(data_dir / 'raw_node_indices.npy'),
        'sensors_sha256': sha256(sensors_path),
    }
    if data_actual != protocol['data_inputs']:
        raise ValueError('Data input fingerprints differ from the v3 protocol')
    assignment_actual = {
        'summary_sha256': sha256(assignment_dir / 'summary.json'),
        'assignments_sha256': sha256(assignment_dir / 'assignments.csv'),
        'edge_scores_sha256': sha256(assignment_dir / 'edge_scores.csv'),
        'unmatched_sha256': sha256(assignment_dir / 'unmatched.csv'),
    }
    expected = protocol['assignment_input']
    if any(assignment_actual[key] != expected[key] for key in assignment_actual):
        raise ValueError('Frozen assignment fingerprints differ from the v3 protocol')
    summary = json.loads((assignment_dir / 'summary.json').read_text(encoding='utf-8'))
    if (summary.get('status') != 'MATCHED_NONINCIDENT_X_ASSIGNMENT_PASS' or
            summary.get('protocol_id') != expected['protocol_id'] or
            summary.get('forecast_Y_used_for_scoring_or_assignment') is not False or
            summary.get('test_split_read') is not False or
            summary.get('maximum_candidate_reuse') != 1 or
            not all(summary.get('acceptance', {}).get('coverage', {}).values()) or
            not all(summary.get('acceptance', {}).get('median_overlap', {}).values()) or
            summary.get('acceptance', {}).get('feature_balance') is not True or
            summary.get('acceptance', {}).get('candidate_reuse') is not True):
        raise ValueError('Frozen assignment status or information boundary differs')
    if summary.get('outputs') != {
            'edge_scores.csv': assignment_actual['edge_scores_sha256'],
            'assignments.csv': assignment_actual['assignments_sha256'],
            'unmatched.csv': assignment_actual['unmatched_sha256']}:
        raise ValueError('Assignment summary output fingerprints are internally inconsistent')
    return data_actual, assignment_actual


def control_window_slots(candidate_t0, protocol):
    if (candidate_t0.tzinfo is not None or candidate_t0.year != int(protocol['source_year']) or
            candidate_t0.minute % 5 or candidate_t0.second or candidate_t0.microsecond):
        raise ValueError('Candidate t0 must be a naive 2023 five-minute nominal timestamp')
    window = protocol['window']
    start = candidate_t0 + timedelta(minutes=int(window['start_minutes_from_candidate_t0']))
    return [start + timedelta(minutes=int(window['step_minutes']) * step)
            for step in range(int(window['steps']))]


def split_for_slot(slot, protocol):
    for split in SPLITS:
        if slot.year == int(protocol['source_year']) and slot.month in protocol['split_months'][split]:
            return split
    return None


def build_control_rows(assignments, manifests, sensor_axes, protocol):
    sensor_roads, sensor_directions, sensor_postmiles = sensor_axes
    expected_counts = protocol['assignment_input']['expected_assigned_samples']
    positive_lookup = {
        split: {int(row['sample_index']): row for row in manifests[split]} for split in SPLITS
    }
    result, masks = {}, {}
    seen_positive, seen_candidate = set(), set()
    for split in SPLITS:
        selected = sorted((row for row in assignments if row['split'] == split),
                          key=lambda row: int(row['sample_index']))
        if len(selected) != int(expected_counts[split]):
            raise ValueError(f'{split} assignment count differs from the v3 protocol')
        rows = []
        split_masks = np.zeros((len(selected), len(sensor_roads)), dtype=bool)
        for control_index, assignment in enumerate(selected):
            sample = int(assignment['sample_index'])
            positive_key, candidate_key = (split, sample), assignment['candidate_t0']
            if positive_key in seen_positive or candidate_key in seen_candidate:
                raise ValueError('Frozen assignment reuses a positive or candidate timestamp')
            seen_positive.add(positive_key)
            seen_candidate.add(candidate_key)
            if sample not in positive_lookup[split]:
                raise ValueError('Assignment references an unknown positive sample')
            positive = positive_lookup[split][sample]
            if (assignment['incident_id'] != positive['incident_id'] or
                    assignment['incident_t0'] != positive['t0'] or
                    int(positive['source_version']) != int(protocol['source_version'])):
                raise ValueError('Assignment and positive manifest identities differ')
            candidate = datetime.fromisoformat(assignment['candidate_t0'])
            slots = control_window_slots(candidate, protocol)
            if any(split_for_slot(slot, protocol) != split for slot in slots):
                raise ValueError('Control window crosses its frozen chronological split')
            road, direction, postmile = (int(assignment['freeway']), assignment['direction'],
                                         float(assignment['incident_postmile']))
            nodes = ((sensor_roads == road) & (sensor_directions == direction) &
                     (np.abs(sensor_postmiles - postmile) <= 10.0))
            if int(nodes.sum()) != int(assignment['affected_node_count']):
                raise ValueError('Materialized affected-node mask differs from the assignment')
            split_masks[control_index] = nodes
            rows.append({
                'control_index': control_index, 'split': split,
                'positive_sample_index': sample, 'incident_id': assignment['incident_id'],
                'positive_t0': assignment['incident_t0'],
                'candidate_t0': assignment['candidate_t0'],
                'source_version': protocol['source_version'],
                'x_start': slots[0].isoformat(), 'x_end': slots[11].isoformat(),
                'excluded_latency_start': slots[12].isoformat(),
                'excluded_latency_end': slots[13].isoformat(),
                'y_start': slots[14].isoformat(), 'y_end': slots[25].isoformat(),
                'support_start': (candidate - timedelta(minutes=70)).isoformat(),
                'support_end_exclusive': (candidate + timedelta(minutes=65)).isoformat(),
                'freeway': road, 'direction': direction, 'incident_postmile': postmile,
                'affected_node_count': int(nodes.sum()),
                'x_distance': float(assignment['x_distance']),
                'candidate_preference_rank': int(assignment['candidate_preference_rank']),
            })
        result[split], masks[split] = rows, split_masks
    return result, masks


def build_month_plan(rows, protocol):
    plan = defaultdict(lambda: defaultdict(lambda: {
        'control_positions': [], 'window_positions': [], 'month_slots': [],
    }))
    for split in SPLITS:
        for row in rows[split]:
            slots = control_window_slots(datetime.fromisoformat(row['candidate_t0']), protocol)
            for step, slot in enumerate(slots):
                if split_for_slot(slot, protocol) != split:
                    raise ValueError('Month plan contains a forbidden source slot')
                item = plan[slot.month][split]
                item['control_positions'].append(int(row['control_index']))
                item['window_positions'].append(step)
                item['month_slots'].append(source_slot(slot))
    for by_split in plan.values():
        for item in by_split.values():
            for key in item:
                item[key] = np.asarray(item[key], dtype=np.int64)
    return plan


def overlap_diagnostics(rows, protocol):
    result, raw_slots = {}, {}
    maximum_candidate_reuse = 0
    for split in SPLITS:
        window_slots = []
        slot_counts = Counter()
        candidate_counts = Counter(row['candidate_t0'] for row in rows[split])
        maximum_candidate_reuse = max(
            maximum_candidate_reuse, max(candidate_counts.values(), default=0))
        for row in rows[split]:
            slots = control_window_slots(datetime.fromisoformat(row['candidate_t0']), protocol)
            keys = [slot.isoformat() for slot in slots]
            window_slots.append(keys)
            slot_counts.update(keys)
        raw_slots[split] = set(slot_counts)
        shared_windows = sum(any(slot_counts[key] > 1 for key in keys) for keys in window_slots)
        result[split] = {
            'control_windows': len(window_slots),
            'total_window_slot_references': sum(slot_counts.values()),
            'unique_source_slots': len(slot_counts),
            'source_slots_used_more_than_once': sum(value > 1 for value in slot_counts.values()),
            'maximum_source_slot_reuse': max(slot_counts.values(), default=0),
            'windows_sharing_at_least_one_source_slot': shared_windows,
            'windows_sharing_fraction': shared_windows / len(window_slots) if window_slots else 0.0,
        }
    cross_split = len(raw_slots['train'] & raw_slots['val'])
    return {
        'splits': result, 'maximum_candidate_t0_reuse': maximum_candidate_reuse,
        'cross_split_source_slot_overlap': cross_split,
    }


def verify_assignment_scores(assignments, rows, masks, control_arrays, positive_arrays,
                             positive_positions, train_std, protocol):
    by_key = {(row['split'], int(row['sample_index'])): row for row in assignments}
    tolerance = float(protocol['assignment_reproduction']['maximum_absolute_float_difference'])
    minimum_overlap = float(
        protocol['assignment_reproduction']['minimum_pairwise_valid_overlap_fraction'])
    result = {}
    all_reproduced = True
    global_maximum = 0.0
    for split in SPLITS:
        discrete_mismatches = 0
        nonfinite_float_mismatches = 0
        maximum_difference = 0.0
        for row in rows[split]:
            key = (split, int(row['positive_sample_index']))
            assignment = by_key[key]
            nodes = masks[split][int(row['control_index'])]
            positive = positive_arrays[split][positive_positions[split][key[1]], :12][:, nodes]
            control = control_arrays[split][int(row['control_index']), :12][:, nodes]
            reproduced = score_histories(positive, control, train_std, minimum_overlap)
            if (int(assignment['pairwise_valid_count']) != reproduced['pairwise_valid_count'] or
                    int(assignment['pairwise_total_count']) != reproduced['pairwise_total_count'] or
                    assignment['eligible'].lower() != str(reproduced['eligible']).lower()):
                discrete_mismatches += 1
            for field in FLOAT_SCORE_FIELDS:
                expected, actual = float(assignment[field]), float(reproduced[field])
                if np.isnan(expected) and np.isnan(actual):
                    difference = 0.0
                elif not np.isfinite(expected) or not np.isfinite(actual):
                    if expected != actual:
                        nonfinite_float_mismatches += 1
                    difference = 0.0
                else:
                    difference = abs(expected - actual)
                maximum_difference = max(maximum_difference, difference)
        reproduced_split = (discrete_mismatches == 0 and nonfinite_float_mismatches == 0 and
                            maximum_difference <= tolerance)
        all_reproduced &= reproduced_split
        global_maximum = max(global_maximum, maximum_difference)
        result[split] = {
            'assignments_checked': len(rows[split]),
            'discrete_mismatch_rows': discrete_mismatches,
            'nonfinite_float_mismatches': nonfinite_float_mismatches,
            'maximum_absolute_float_difference': maximum_difference,
            'all_reproduced': reproduced_split,
        }
    return result, all_reproduced, global_maximum


def raw_quality(array, mask):
    scopes = {'X': slice(0, 12), 'excluded_latency': slice(12, 14), 'Y': slice(14, 26)}
    result = {}
    for name, steps in scopes.items():
        total = finite = negative = nonfinite = 0
        affected_total = affected_valid = 0
        for first in range(0, len(array), 128):
            last = min(first + 128, len(array))
            block = np.asarray(array[first:last, steps], dtype=np.float32)
            valid = np.isfinite(block) & (block >= 0)
            total += block.size
            finite += int(valid.sum())
            negative += int((np.isfinite(block) & (block < 0)).sum())
            nonfinite += int((~np.isfinite(block)).sum())
            expanded_mask = mask[first:last, None, :]
            affected_total += int(expanded_mask.sum()) * block.shape[1]
            affected_valid += int((valid & expanded_mask).sum())
        result[name] = {
            'full_graph_total': total, 'full_graph_valid_nonnegative': finite,
            'full_graph_negative': negative, 'full_graph_nonfinite': nonfinite,
            'affected_total': affected_total, 'affected_valid_nonnegative': affected_valid,
            'affected_valid_fraction': affected_valid / affected_total if affected_total else None,
        }
    return result


def materialize(data_dir, assignment_dir, sensors_path, protocol_path, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError('Materialization output exists; preserve it and use a new directory')
    protocol = load_protocol(protocol_path)
    input_hashes, assignment_hashes = verify_inputs(
        data_dir, assignment_dir, sensors_path, protocol)
    data_dir, assignment_dir = Path(data_dir), Path(assignment_dir)
    station_ids = np.load(data_dir / 'station_ids.npy', allow_pickle=False)
    if station_ids.shape != (int(protocol['expected_sensor_count']),):
        raise ValueError('Station axis differs from the v3 protocol')
    sensor_axes = ordered_sensors(sensors_path, station_ids)
    scaler = json.loads((data_dir / 'scaler.json').read_text(encoding='utf-8'))
    if scaler['station_ids'] != station_ids.tolist():
        raise ValueError('Scaler and station axes differ')
    train_std = float(scaler['std'])
    assignments = read_csv(assignment_dir / 'assignments.csv')
    manifests, positive_arrays, positive_positions = {}, {}, {}
    for split in SPLITS:
        manifests[split] = read_csv(data_dir / f'{split}_manifest.csv')
        positive_positions[split] = {
            int(row['sample_index']): index for index, row in enumerate(manifests[split])
        }
        positive_arrays[split] = np.load(
            data_dir / f'{split}_flow.npy', mmap_mode='r', allow_pickle=False)
        expected_shape = (len(manifests[split]), 26, len(station_ids))
        if positive_arrays[split].shape != expected_shape:
            raise ValueError(f'{split} positive flow shape differs')
    rows, masks = build_control_rows(assignments, manifests, sensor_axes, protocol)
    plan = build_month_plan(rows, protocol)
    overlap = overlap_diagnostics(rows, protocol)

    output.mkdir(parents=True)
    partial_paths = {split: output / f'{split}_control_flow.npy.partial' for split in SPLITS}
    arrays = {
        split: np.lib.format.open_memmap(
            partial_paths[split], mode='w+', dtype=np.float32,
            shape=(len(rows[split]), 26, len(station_ids)))
        for split in SPLITS
    }
    source = VerifiedMonthCache(data_dir)
    month_hashes = {}
    for month in sorted(plan):
        month_flow, month_hash = source.load(month)
        if month_hash != protocol['source_month_sha256'][str(month)]:
            raise ValueError(f'Source month {month} fingerprint differs from the v3 protocol')
        month_hashes[str(month)] = month_hash
        for split, item in plan[month].items():
            arrays[split][item['control_positions'], item['window_positions'], :] = (
                month_flow[:, item['month_slots']].T)
            arrays[split].flush()
        del month_flow

    reproduction, all_reproduced, maximum_score_difference = verify_assignment_scores(
        assignments, rows, masks, arrays, positive_arrays, positive_positions,
        train_std, protocol)
    quality = {split: raw_quality(arrays[split], masks[split]) for split in SPLITS}
    expected_shapes = {
        split: [len(rows[split]), 26, len(station_ids)] for split in SPLITS
    }
    shapes_correct = all(list(arrays[split].shape) == expected_shapes[split] for split in SPLITS)
    acceptance = {
        'assignment_scores_reproduced': all_reproduced,
        'exact_output_shapes': shapes_correct,
        'candidate_t0_reuse': overlap['maximum_candidate_t0_reuse'] <=
                              int(protocol['acceptance']['maximum_candidate_t0_reuse']),
        'cross_split_source_slot_overlap': overlap['cross_split_source_slot_overlap'] <=
                                           int(protocol['acceptance'][
                                               'maximum_cross_split_source_slot_overlap']),
        'all_source_month_fingerprints': month_hashes == protocol['source_month_sha256'],
    }
    passed = all(acceptance.values())
    for array in arrays.values():
        array.flush()
        array._mmap.close()
    arrays.clear()

    output_files = []
    for split in SPLITS:
        final_array = output / f'{split}_control_flow.npy'
        partial_paths[split].replace(final_array)
        output_files.append(final_array.name)
        mask_name = f'{split}_affected_mask.npy'
        np.save(output / mask_name, masks[split], allow_pickle=False)
        output_files.append(mask_name)
        manifest_name = f'{split}_control_manifest.csv'
        write_csv(output / manifest_name, rows[split], MANIFEST_FIELDS)
        output_files.append(manifest_name)

    cache_digest = hashlib.sha256()
    for record in source.records:
        cache_digest.update(json.dumps(record, separators=(',', ':')).encode())
    summary = {
        'status': 'MATCHED_NONINCIDENT_MATERIALIZATION_PASS' if passed else
                  'MATCHED_NONINCIDENT_MATERIALIZATION_REJECTED',
        'scope': protocol['scope'], 'main_training_ready': False,
        'protocol_id': protocol['protocol_id'], 'protocol_sha256': sha256(protocol_path),
        'assignment_frozen_before_future_Y_read': True,
        'future_Y_materialized_after_assignment': True,
        'future_Y_used_to_rank_or_replace_controls': False,
        'test_split_read': False, 'incident_text_read': False,
        'array_shapes': expected_shapes, 'dtype': 'float32',
        'raw_missing_preserved': True, 'assignment_reproduction': reproduction,
        'maximum_assignment_score_difference': maximum_score_difference,
        'overlap_diagnostics': overlap, 'raw_quality': quality,
        'acceptance': acceptance,
        'source_cache': {
            'verified_row_reads': len(source.records),
            'verified_row_bytes': sum(record[3] for record in source.records),
            'ordered_record_digest': cache_digest.hexdigest(),
            'month_manifest_sha256': month_hashes,
        },
        'control_semantics': protocol['control_semantics'],
        'limitations': [
            'Recorded-incident-free status is local to each positive affected-node set, not the full graph.',
            'Remote or unreported incidents may remain in a materialized control window.',
            'Within-split control windows may share source slots; overlap is reported, not hidden.',
            'Materialized controls remain matched observational comparisons, not causal counterfactuals.',
        ],
        'inputs': {
            **input_hashes,
            **{f'assignment_{key}': value for key, value in assignment_hashes.items()},
            'protocol_sha256': sha256(protocol_path), 'code_sha256': sha256(__file__),
        },
    }
    summary['outputs'] = {
        name: {'sha256': sha256(output / name), 'bytes': (output / name).stat().st_size}
        for name in output_files
    }
    (output / 'summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
        encoding='utf-8')
    print(json.dumps({key: value for key, value in summary.items()
                      if key not in ('inputs', 'outputs')}, ensure_ascii=False, indent=2), flush=True)
    if not passed:
        raise ValueError('Matched-control materialization failed the frozen acceptance gates')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--assignment-dir', type=Path, required=True)
    parser.add_argument('--sensors', type=Path, required=True)
    parser.add_argument('--protocol', type=Path,
                        default=Path(__file__).with_name(
                            'matched_nonincident_materialize_v3.json'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    materialize(args.data_dir, args.assignment_dir, args.sensors, args.protocol, args.output)


if __name__ == '__main__':
    main()
