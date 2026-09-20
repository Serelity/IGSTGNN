"""Materialize a frozen second routine control without reopening its X-only assignment."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.chronological.audit_matched_controls import read_csv, sha256, write_csv
from experiments.chronological.materialize_matched_controls import (
    MANIFEST_FIELDS, build_control_rows, build_month_plan, control_window_slots,
    overlap_diagnostics, raw_quality, verify_assignment_scores,
)
from experiments.chronological.score_matched_controls import (
    SPLITS, VerifiedMonthCache, ordered_sensors,
)


def load_protocol(path):
    protocol = json.loads(Path(path).read_text(encoding='utf-8'))
    if protocol.get('scope') != 'fixed_secondary_assignment_full_window_materialization':
        raise ValueError('Protocol scope must remain fixed-secondary materialization')
    if protocol.get('main_training_ready') is not False:
        raise ValueError('Second-control materialization cannot authorize main training')
    if (int(protocol.get('source_year', 0)) != 2023 or
            int(protocol.get('source_version', 0)) != 8 or
            int(protocol.get('expected_sensor_count', 0)) != 496):
        raise ValueError('v5b is frozen to the 2023 source-v8 Contra496 package')
    window = protocol['window']
    if (int(window['start_minutes_from_candidate_t0']) != -65 or
            int(window['steps']) != 26 or int(window['step_minutes']) != 5 or
            window['X_slice'] != [0, 12] or
            window['excluded_latency_slice'] != [12, 14] or
            window['Y_slice'] != [14, 26] or window['dtype'] != 'float32' or
            window.get('raw_missing_preserved') is not True):
        raise ValueError('v5b full-window layout or raw-value policy changed')
    if protocol.get('split_months') != {'train': list(range(1, 9)), 'val': [9, 10]}:
        raise ValueError('v5b chronological split months changed')
    boundary = protocol['information_boundary']
    required = [
        'secondary_assignment_frozen_before_materialization',
        'primary_control_immutable', 'future_Y_may_be_materialized_after_assignment',
        'test_split_prohibited', 'incident_description_prohibited',
        'incident_type_prohibited', 'v4_outcome_results_prohibited',
    ]
    if not all(boundary.get(key) is True for key in required):
        raise ValueError('v5b information boundary changed')
    if boundary.get('future_Y_may_change_assignment') is not False:
        raise ValueError('Future Y must never alter the frozen secondary assignment')
    reproduction = protocol['assignment_reproduction']
    if (float(reproduction['minimum_pairwise_valid_overlap_fraction']) != 0.9 or
            float(reproduction['maximum_absolute_float_difference']) < 0):
        raise ValueError('v5b assignment-reproduction rule changed')
    if protocol['secondary_assignment_input'].get('expected_assigned_samples') != {
            'train': 3106, 'val': 618}:
        raise ValueError('v5b expected second-control counts changed')
    acceptance = protocol['acceptance']
    if (int(acceptance['maximum_secondary_candidate_t0_reuse']) != 1 or
            int(acceptance['maximum_primary_secondary_candidate_t0_overlap']) != 0 or
            int(acceptance['maximum_cross_split_source_slot_overlap']) != 0 or
            acceptance.get('require_all_assignment_scores_reproduced') is not True or
            acceptance.get('require_exact_output_shapes') is not True):
        raise ValueError('v5b acceptance gates changed')
    return protocol


def verify_inputs(data_dir, secondary_dir, primary_dir, sensors_path, protocol):
    data_dir, secondary_dir, primary_dir = map(Path, (data_dir, secondary_dir, primary_dir))
    data_actual = {
        'summary_sha256': sha256(data_dir / 'summary.json'),
        'scaler_sha256': sha256(data_dir / 'scaler.json'),
        'station_ids_sha256': sha256(data_dir / 'station_ids.npy'),
        'raw_node_indices_sha256': sha256(data_dir / 'raw_node_indices.npy'),
        'sensors_sha256': sha256(sensors_path),
    }
    if data_actual != protocol['data_inputs']:
        raise ValueError('Data input fingerprints differ from the v5b protocol')
    secondary_actual = {
        'summary_sha256': sha256(secondary_dir / 'summary.json'),
        'secondary_assignments_sha256': sha256(
            secondary_dir / 'secondary_assignments.csv'),
        'unmatched_primary_pairs_sha256': sha256(
            secondary_dir / 'unmatched_primary_pairs.csv'),
    }
    expected_secondary = {
        key: value for key, value in protocol['secondary_assignment_input'].items()
        if key not in ('protocol_id', 'expected_assigned_samples')
    }
    if secondary_actual != expected_secondary:
        raise ValueError('Frozen secondary-assignment fingerprints differ from v5b protocol')
    secondary_summary = json.loads(
        (secondary_dir / 'summary.json').read_text(encoding='utf-8'))
    if (secondary_summary.get('status') != 'SECOND_MATCHED_CONTROL_ASSIGNMENT_PASS' or
            secondary_summary.get('protocol_id') !=
            protocol['secondary_assignment_input']['protocol_id'] or
            secondary_summary.get('primary_assignment_immutable') is not True or
            secondary_summary.get('forecast_Y_read') is not False or
            secondary_summary.get('test_split_read') is not False or
            secondary_summary.get('outcome_audit_results_read') is not False or
            not all(secondary_summary.get('acceptance', {}).get('coverage', {}).values()) or
            secondary_summary.get('acceptance', {}).get('road_direction_coverage') is not True or
            secondary_summary.get('acceptance', {}).get('feature_balance') is not True or
            secondary_summary.get('acceptance', {}).get('candidate_center_reuse') is not True):
        raise ValueError('Secondary assignment status or information boundary differs')
    if secondary_summary.get('outputs') != {
            'secondary_assignments.csv':
            secondary_actual['secondary_assignments_sha256'],
            'unmatched_primary_pairs.csv':
            secondary_actual['unmatched_primary_pairs_sha256']}:
        raise ValueError('Secondary-assignment output fingerprints are inconsistent')

    primary_actual = {
        'summary_sha256': sha256(primary_dir / 'summary.json'),
        'train_control_manifest_sha256': sha256(
            primary_dir / 'train_control_manifest.csv'),
        'val_control_manifest_sha256': sha256(primary_dir / 'val_control_manifest.csv'),
    }
    expected_primary = {
        key: value for key, value in protocol['primary_control_input'].items()
        if key != 'protocol_id'
    }
    if primary_actual != expected_primary:
        raise ValueError('Frozen primary-control fingerprints differ from v5b protocol')
    primary_summary = json.loads((primary_dir / 'summary.json').read_text(encoding='utf-8'))
    if (primary_summary.get('status') != 'MATCHED_NONINCIDENT_MATERIALIZATION_PASS' or
            primary_summary.get('protocol_id') !=
            protocol['primary_control_input']['protocol_id'] or
            primary_summary.get('future_Y_used_to_rank_or_replace_controls') is not False or
            primary_summary.get('test_split_read') is not False):
        raise ValueError('Primary-control status or information boundary differs')
    return data_actual, secondary_actual, primary_actual


def normalize_assignments(rows):
    normalized = []
    for row in rows:
        if 'candidate_preference_rank' in row:
            raise ValueError('Secondary assignment unexpectedly uses primary rank field')
        if 'original_candidate_preference_rank' not in row:
            raise ValueError('Secondary assignment lacks its frozen preference rank')
        normalized.append({
            **row,
            'candidate_preference_rank': row['original_candidate_preference_rank'],
        })
    return normalized


def verify_primary_alignment(rows, primary_rows, positive_rows):
    """Verify that every second control completes one immutable I/C1/C2 triple."""
    diagnostics = {}
    for split in SPLITS:
        primary = {int(row['positive_sample_index']): row for row in primary_rows[split]}
        positive = {int(row['sample_index']): row for row in positive_rows[split]}
        if len(primary) != len(primary_rows[split]) or len(positive) != len(positive_rows[split]):
            raise ValueError(f'{split} primary or positive manifest contains duplicate samples')
        primary_centers = {row['candidate_t0'] for row in primary_rows[split]}
        second_centers = set()
        for row in rows[split]:
            sample = int(row['positive_sample_index'])
            if sample not in primary or sample not in positive:
                raise ValueError('Second control references a non-common positive sample')
            first, incident = primary[sample], positive[sample]
            if (row['incident_id'] != first['incident_id'] or
                    row['incident_id'] != incident['incident_id'] or
                    row['positive_t0'] != first['positive_t0'] or
                    row['positive_t0'] != incident['t0'] or
                    int(row['freeway']) != int(first['freeway']) or
                    row['direction'] != first['direction'] or
                    int(row['affected_node_count']) !=
                    int(first['affected_node_count'])):
                raise ValueError('Incident, primary-control, and second-control identities differ')
            if row['candidate_t0'] in second_centers:
                raise ValueError('Second controls reuse a candidate center')
            second_centers.add(row['candidate_t0'])
        overlap = primary_centers & second_centers
        diagnostics[split] = {
            'common_triples': len(rows[split]),
            'primary_candidate_centers': len(primary_centers),
            'secondary_candidate_centers': len(second_centers),
            'primary_secondary_candidate_t0_overlap': len(overlap),
        }
    return diagnostics


def combined_overlap_diagnostics(rows, primary_rows, positive_rows, protocol):
    """Report source-slot dependence within and across the three matched windows."""
    result, all_split_slots = {}, {}
    for split in SPLITS:
        primary = {int(row['positive_sample_index']): row for row in primary_rows[split]}
        positive = {int(row['sample_index']): row for row in positive_rows[split]}
        global_counts, used = Counter(), set()
        internal_overlap = 0
        c1_c2_overlap = 0
        incident_c2_overlap = 0
        strict_nonoverlap = 0
        triple_sets = []
        for row in sorted(rows[split], key=lambda item: (
                item['positive_t0'], int(item['positive_sample_index']))):
            sample = int(row['positive_sample_index'])
            windows = {
                'incident': set(control_window_slots(
                    _parse_start_as_center(positive[sample]['x_start'], protocol), protocol)),
                'primary': set(control_window_slots(
                    _parse_start_as_center(primary[sample]['x_start'], protocol), protocol)),
                'secondary': set(control_window_slots(
                    _parse_start_as_center(row['x_start'], protocol), protocol)),
            }
            if windows['primary'] & windows['secondary']:
                c1_c2_overlap += 1
            if windows['incident'] & windows['secondary']:
                incident_c2_overlap += 1
            has_internal_overlap = bool(
                (windows['incident'] & windows['primary']) or
                (windows['incident'] & windows['secondary']) or
                (windows['primary'] & windows['secondary']))
            if has_internal_overlap:
                internal_overlap += 1
            combined = set().union(*windows.values())
            triple_sets.append(combined)
            global_counts.update(combined)
            if not has_internal_overlap and not combined & used:
                strict_nonoverlap += 1
                used.update(combined)
        all_split_slots[split] = set(global_counts)
        overlapping_triples = sum(
            any(global_counts[slot] > 1 for slot in slots) for slots in triple_sets)
        result[split] = {
            'common_triples': len(rows[split]),
            'triples_with_internal_source_slot_overlap': internal_overlap,
            'primary_secondary_windows_with_source_slot_overlap': c1_c2_overlap,
            'incident_secondary_windows_with_source_slot_overlap': incident_c2_overlap,
            'combined_unique_source_slots': len(global_counts),
            'combined_source_slots_used_more_than_once': sum(
                count > 1 for count in global_counts.values()),
            'combined_maximum_source_slot_reuse': max(global_counts.values(), default=0),
            'triples_sharing_source_slots_with_other_triples': overlapping_triples,
            'strict_three_window_nonoverlap_triples': strict_nonoverlap,
            'strict_three_window_nonoverlap_fraction':
                strict_nonoverlap / len(rows[split]) if rows[split] else 0.0,
        }
    return {
        'splits': result,
        'cross_split_combined_source_slot_overlap': len(
            all_split_slots['train'] & all_split_slots['val']),
    }


def _parse_start_as_center(x_start, protocol):
    from datetime import datetime, timedelta
    return datetime.fromisoformat(x_start) - timedelta(
        minutes=int(protocol['window']['start_minutes_from_candidate_t0']))


def materialize(data_dir, secondary_dir, primary_dir, sensors_path, protocol_path, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError('Second-control output exists; preserve it and use a new directory')
    protocol = load_protocol(protocol_path)
    input_hashes = verify_inputs(
        data_dir, secondary_dir, primary_dir, sensors_path, protocol)
    data_dir, secondary_dir, primary_dir = map(Path, (data_dir, secondary_dir, primary_dir))
    station_ids = np.load(data_dir / 'station_ids.npy', allow_pickle=False)
    if station_ids.shape != (int(protocol['expected_sensor_count']),):
        raise ValueError('Station axis differs from the v5b protocol')
    sensor_axes = ordered_sensors(sensors_path, station_ids)
    scaler = json.loads((data_dir / 'scaler.json').read_text(encoding='utf-8'))
    if scaler['station_ids'] != station_ids.tolist():
        raise ValueError('Scaler and station axes differ')

    assignments = normalize_assignments(read_csv(
        secondary_dir / 'secondary_assignments.csv'))
    manifests, positive_arrays, positive_positions, primary_rows = {}, {}, {}, {}
    for split in SPLITS:
        manifests[split] = read_csv(data_dir / f'{split}_manifest.csv')
        primary_rows[split] = read_csv(primary_dir / f'{split}_control_manifest.csv')
        positive_positions[split] = {
            int(row['sample_index']): index for index, row in enumerate(manifests[split])
        }
        positive_arrays[split] = np.load(
            data_dir / f'{split}_flow.npy', mmap_mode='r', allow_pickle=False)
        expected_shape = (len(manifests[split]), 26, len(station_ids))
        if positive_arrays[split].shape != expected_shape:
            raise ValueError(f'{split} positive flow shape differs')

    construction_protocol = {
        **protocol, 'assignment_input': protocol['secondary_assignment_input']
    }
    rows, masks = build_control_rows(
        assignments, manifests, sensor_axes, construction_protocol)
    identity = verify_primary_alignment(rows, primary_rows, manifests)
    plan = build_month_plan(rows, protocol)
    secondary_overlap = overlap_diagnostics(rows, protocol)
    combined_overlap = combined_overlap_diagnostics(rows, primary_rows, manifests, protocol)

    output.mkdir(parents=True)
    partial_paths = {
        split: output / f'{split}_second_control_flow.npy.partial' for split in SPLITS
    }
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
            raise ValueError(f'Source month {month} fingerprint differs from v5b protocol')
        month_hashes[str(month)] = month_hash
        for split, item in plan[month].items():
            arrays[split][item['control_positions'], item['window_positions'], :] = (
                month_flow[:, item['month_slots']].T)
            arrays[split].flush()
        del month_flow

    reproduction, all_reproduced, maximum_score_difference = verify_assignment_scores(
        assignments, rows, masks, arrays, positive_arrays, positive_positions,
        float(scaler['std']), protocol)
    quality = {split: raw_quality(arrays[split], masks[split]) for split in SPLITS}
    expected_shapes = {
        split: [len(rows[split]), 26, len(station_ids)] for split in SPLITS
    }
    primary_secondary_overlap = sum(
        identity[split]['primary_secondary_candidate_t0_overlap'] for split in SPLITS)
    acceptance = {
        'assignment_scores_reproduced': all_reproduced,
        'exact_output_shapes': all(
            list(arrays[split].shape) == expected_shapes[split] for split in SPLITS),
        'secondary_candidate_t0_reuse':
            secondary_overlap['maximum_candidate_t0_reuse'] <= int(
                protocol['acceptance']['maximum_secondary_candidate_t0_reuse']),
        'primary_secondary_candidate_t0_overlap': primary_secondary_overlap <= int(
            protocol['acceptance']['maximum_primary_secondary_candidate_t0_overlap']),
        'cross_split_source_slot_overlap':
            combined_overlap['cross_split_combined_source_slot_overlap'] <= int(
                protocol['acceptance']['maximum_cross_split_source_slot_overlap']),
        'all_source_month_fingerprints': month_hashes == protocol['source_month_sha256'],
    }
    passed = all(acceptance.values())
    for array in arrays.values():
        array.flush()
        array._mmap.close()
    arrays.clear()

    output_files = []
    for split in SPLITS:
        final_array = output / f'{split}_second_control_flow.npy'
        partial_paths[split].replace(final_array)
        output_files.append(final_array.name)
        mask_name = f'{split}_second_affected_mask.npy'
        np.save(output / mask_name, masks[split], allow_pickle=False)
        output_files.append(mask_name)
        manifest_name = f'{split}_second_control_manifest.csv'
        write_csv(output / manifest_name, rows[split], MANIFEST_FIELDS)
        output_files.append(manifest_name)

    cache_digest = hashlib.sha256()
    for record in source.records:
        cache_digest.update(json.dumps(record, separators=(',', ':')).encode())
    summary = {
        'status': ('SECOND_MATCHED_CONTROL_MATERIALIZATION_PASS' if passed else
                   'SECOND_MATCHED_CONTROL_MATERIALIZATION_REJECTED'),
        'scope': protocol['scope'], 'protocol_id': protocol['protocol_id'],
        'protocol_sha256': sha256(protocol_path), 'main_training_ready': False,
        'secondary_assignment_frozen_before_future_Y_read': True,
        'primary_control_immutable': True,
        'future_Y_materialized_after_assignment': True,
        'future_Y_used_to_rank_replace_or_remove_controls': False,
        'test_split_read': False, 'incident_text_read': False,
        'outcome_audit_results_read': False,
        'array_shapes': expected_shapes, 'dtype': 'float32',
        'raw_missing_preserved': True, 'triple_identity_diagnostics': identity,
        'assignment_reproduction': reproduction,
        'maximum_assignment_score_difference': maximum_score_difference,
        'secondary_overlap_diagnostics': secondary_overlap,
        'combined_three_window_overlap_diagnostics': combined_overlap,
        'raw_quality': quality, 'acceptance': acceptance,
        'source_cache': {
            'verified_row_reads': len(source.records),
            'verified_row_bytes': sum(record[3] for record in source.records),
            'ordered_record_digest': cache_digest.hexdigest(),
            'month_manifest_sha256': month_hashes,
        },
        'control_semantics': protocol['control_semantics'],
        'limitations': [
            'The second-control cohort is a subset of the frozen primary matched population.',
            'Recorded-incident-free status is local to each positive affected-node set.',
            'Remote or unreported incidents may remain in a materialized control window.',
            'Source-slot overlap within a split is reported and requires blocked uncertainty.',
            'The three windows are matched observational comparisons, not counterfactuals.',
        ],
        'inputs': {
            **input_hashes[0],
            **{f'secondary_{key}': value for key, value in input_hashes[1].items()},
            **{f'primary_{key}': value for key, value in input_hashes[2].items()},
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
                      if key not in ('inputs', 'outputs')},
                     ensure_ascii=False, indent=2, allow_nan=False), flush=True)
    if not passed:
        raise ValueError('Second-control materialization failed the frozen acceptance gates')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--secondary-assignment-dir', type=Path, required=True)
    parser.add_argument('--primary-control-dir', type=Path, required=True)
    parser.add_argument('--sensors', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, default=Path(__file__).with_name(
        'second_matched_control_materialize_v5b.json'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    materialize(args.data_dir, args.secondary_assignment_dir, args.primary_control_dir,
                args.sensors, args.protocol, args.output)


if __name__ == '__main__':
    main()
