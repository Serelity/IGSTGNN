"""Audit joint input coverage over the complete network, using training X only.

Computational input presence and evidence for physical equations are reported
separately. This audit never declares a physical model ready from numeric coverage.
"""

import argparse
from collections import Counter
from datetime import datetime
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).absolute().parents[2]
sys.path.insert(0, str(REPO))
from experiments.chronological.prepare_incident_corridors import load_inputs, verified
from experiments.chronological.prepare_context import spatial_features
from experiments.chronological.prepare_incident_physics_evidence import (
    candidate_inventory, check_source_metadata,
)
from src.utils.incident_corridor import (
    CHANNELS, read_json, require, sha256, unique_index, write_json, write_rows,
)


def joint_history_coverage(history, labels):
    require(history.ndim == 4 and history.shape[1] == 12 and history.shape[3] == 3
            and history.shape[:2] == labels.shape, 'Unexpected history/time axes')
    flat = history.reshape(-1, history.shape[2], 3)
    times, first, inverse = np.unique(labels.reshape(-1), return_index=True, return_inverse=True)
    values = flat[first]
    for start in range(0, len(flat), 256):
        require(np.array_equal(flat[start:start + 256], values[inverse[start:start + 256]],
                               equal_nan=True), 'Conflicting values at repeated station/time')
    usable = np.isfinite(values) & (values >= 0)
    # Intersection at the SAME station and nominal time, not a marginal average.
    joint = usable.all(axis=2)
    return times, values, usable.sum(axis=0), joint.sum(axis=0)


def inspect_network(history, labels, station_ids, published, raw, distances, adjacency):
    require(adjacency.shape == (len(station_ids), len(station_ids))
            and np.isfinite(adjacency).all() and (adjacency >= 0).all(), 'Invalid graph')
    require(distances.shape == (history.shape[0], len(station_ids), 3)
            and np.isfinite(distances).all(), 'Report/node axes mismatch')
    by_id = unique_index(published)
    require(set(by_id) == set(map(int, station_ids)), 'Station metadata identity mismatch')
    ordered = [by_id[int(s)] for s in station_ids]
    check_source_metadata(ordered, raw)
    times, values, channel_counts, joint_counts = joint_history_coverage(history, labels)
    support = np.any(distances != 0, axis=-1)
    graph_supported = (adjacency > 0).any(0) | (adjacency > 0).any(1)
    stations = []
    for i, row in enumerate(ordered):
        static = (bool(row['Fwy'].strip()) and row['Direction'] in ('N', 'S', 'E', 'W')
                  and all(np.isfinite(float(row[k])) for k in ('Abs PM', 'Lat', 'Lng')))
        stations.append({
            'station_id': int(station_ids[i]), 'package_node_index': i,
            'road': row['Fwy'], 'direction': row['Direction'],
            'metadata_fields_present': bool(static), 'has_learning_graph_neighbor': bool(graph_supported[i]),
            **{c + '_usable_unique_history_slots': int(channel_counts[i, k]) for k, c in enumerate(CHANNELS)},
            'all_three_usable_same_slot_count': int(joint_counts[i]),
            'unique_history_slots': len(times),
            'common_input_presence': bool(static and graph_supported[i] and joint_counts[i] > 0),
            'all_history_slots_jointly_usable': bool(static and graph_supported[i] and joint_counts[i] == len(times)),
            'report_supported_train_windows': int(support[:, i].sum()),
            'physical_units_certified': False, 'physical_connections_certified': False,
        })
    pairs = candidate_inventory(ordered, raw, values[..., 0], support)
    groups = {}
    for i, row in enumerate(ordered):
        groups.setdefault((row['Fwy'], row['Direction']), []).append(i)
    roads = []
    for (road, direction), indices in sorted(groups.items()):
        local_pairs = [r for r in pairs if r['road'] == road and r['direction'] == direction]
        roads.append({
            'road': road, 'direction': direction, 'stations': len(indices),
            'common_input_presence_stations': sum(stations[i]['common_input_presence'] for i in indices),
            'all_history_slots_jointly_usable_stations': sum(stations[i]['all_history_slots_jointly_usable'] for i in indices),
            'joint_usable_station_time_fraction': float(joint_counts[indices].sum() / (len(times) * len(indices))),
            'report_supported_train_windows': int(support[:, indices].any(1).sum()),
            'stations_with_any_report_support': int(support[:, indices].any(0).sum()),
            'candidate_postmile_pairs': len(local_pairs),
            'metadata_prefilter_pass_pairs': sum(r['metadata_prefilter_pass'] for r in local_pairs),
            'pairs_with_known_nonmainline': sum(r['known_nonmainline_count'] > 0 for r in local_pairs),
            'physical_ready_pair_count_established': False,
        })
    summary = {
        'stations': len(station_ids), 'road_direction_groups': len(roads), 'train_windows': len(history),
        'unique_history_labels': len(times),
        'numeric_channel_usable_cells': dict(zip(CHANNELS, map(int, channel_counts.sum(0)))),
        'per_channel_station_time_denominator': int(len(times) * len(station_ids)),
        'joint_three_channel_usable_station_time_cells': int(joint_counts.sum()),
        'common_input_presence_stations': sum(r['common_input_presence'] for r in stations),
        'all_history_slots_jointly_usable_stations': sum(r['all_history_slots_jointly_usable'] for r in stations),
        'stations_with_any_report_support': int(support.any(0).sum()),
        'station_report_support_fraction': float(support.any(0).mean()),
        'road_groups_with_any_report_support': sum(r['report_supported_train_windows'] > 0 for r in roads),
        'learning_graph_nonzero_entries': int(np.count_nonzero(adjacency)),
        'learning_graph_is_physical_connection_evidence': False,
        'candidate_postmile_pairs': len(pairs),
        'metadata_prefilter_pass_pairs': sum(r['metadata_prefilter_pass'] for r in pairs),
        'pairs_with_known_nonmainline': sum(r['known_nonmainline_count'] > 0 for r in pairs),
        'road_groups_without_a_candidate_pair': sum(r['candidate_postmile_pairs'] == 0 for r in roads),
        'physical_equation_joint_coverage': None,
        'physical_equation_joint_coverage_status': 'NOT_ESTABLISHED_BY_EXISTING_EVIDENCE',
        'semantic_or_physical_readiness_inferred_from_numeric_coverage': False,
    }
    return summary, stations, roads, pairs


def trace_training_reports(records, rows, ordered_sensors, distances):
    """Replay only training identity metadata; no descriptions/types become inputs."""
    index = {int(r['sample_index']): r for r in records}
    require(len(index) == len(records), 'Duplicate identity records')
    require(distances.shape == (len(rows), len(ordered_sensors), 3), 'Replay axes mismatch')
    counts = Counter()
    for i, row in enumerate(rows):
        record = index[int(row['sample_index'])]
        require(record['status'] == 'unique_metadata_candidate' and len(record['candidates']) == 1,
                'Unresolved training identity')
        event = record['candidates'][0]
        require(event['incident_id'] == row['incident_id'] and
                datetime.fromisoformat(event['report_time']) == datetime.fromisoformat(row['report_time']),
                'Training manifest identity mismatch')
        require(np.array_equal(spatial_features(event, ordered_sensors), distances[i]),
                'Training spatial context replay mismatch')
        counts[str(event['freeway']) + '-' + event['direction']] += 1
    return {'training_rows_replayed': len(rows), 'train_context_replay_exact': True,
            'train_event_road_counts': dict(sorted(counts.items())),
            'scope': 'selected_training_identity_metadata_only',
            'interpretation': 'Association scope already exists in frozen training identities; raw event coverage remains unestablished'}


def audit(data_dir, history_dir, sensors, metadata_dir, identity, output, event_sidecar=None):
    output = Path(output)
    partial = output.with_name(output.name + '.partial')
    if output.exists() or partial.exists():
        raise FileExistsError('Use a new audit output directory')
    config = read_json(identity)
    (history, ids, rows, report, labels, published, raw,
     semantics, fingerprints, source) = load_inputs(
        Path(data_dir), Path(history_dir), Path(sensors), Path(metadata_dir), config)
    # Reuse only the frozen identity; the 40-station corridor selection is NOT applied.
    context = read_json(Path(data_dir) / 'context_manifest.json')
    fingerprints['data/adjacency.npy'] = verified(Path(data_dir) / 'adjacency.npy',
                                                 context['outputs']['adjacency.npy'])
    adjacency = np.load(Path(data_dir) / 'adjacency.npy', allow_pickle=False)
    result, stations, roads, pairs = inspect_network(history, labels, ids, published, raw,
                                                    report['distances'], adjacency)
    if event_sidecar is not None:
        expected = [v for k, v in context['sources'].items()
                    if k.replace('\\', '/').endswith('/event_identity_sidecar.json')]
        require(len(expected) == 1, 'Missing or ambiguous frozen identity source')
        fingerprints['event_sidecar'] = verified(event_sidecar, expected[0])
        ordered = unique_index(published)
        result['training_report_provenance'] = trace_training_reports(
            read_json(event_sidecar), rows, [ordered[int(s)] for s in ids], report['distances'])
    fingerprints.update(identity=sha256(identity), audit=sha256(Path(__file__)),
                        input_loader=sha256(REPO / 'experiments/chronological/prepare_incident_corridors.py'),
                        pair_inventory=sha256(REPO / 'experiments/chronological/prepare_incident_physics_evidence.py'))
    result.update(
        status='NETWORK_COMMON_INPUT_COVERAGE_AUDIT_COMPLETE',
        architecture_status='PHYSICAL_ARCHITECTURE_NOT_YET_FROZEN',
        scope='all_published_stations_train_history_only',
        majority_threshold=None, majority_threshold_status='NOT_USER_SPECIFIED_NOT_INFERRED',
        main_training_ready=False, model_training_performed=False,
        raw_flow_gap_or_Y_file_opened=False, validation_array_opened=False, test_array_opened=False,
        incident_semantic_fields_used=False, identity_sidecar_opened=event_sidecar is not None,
        selection_40_station_filter_applied=False,
        channel_semantics=semantics, inputs_sha256=fingerprints, metadata_source=source,
        value_mask_meaning='finite_and_nonnegative_not_certified_sensor_observations',
        report_support_meaning='original_conditional_report_location_association_not_true_impact',
        requirements_policy='Mandatory inputs must be jointly supported on most of the network; minority-only evidence is auxiliary',
    )
    partial.mkdir(parents=True)
    write_rows(partial / 'station_joint_coverage.csv', stations)
    write_rows(partial / 'road_joint_coverage.csv', roads)
    write_rows(partial / 'candidate_pair_evidence.csv', pairs)
    result['outputs'] = {p.name: sha256(p) for p in sorted(partial.iterdir())}
    write_json(partial / 'summary.json', result)
    partial.rename(output)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--history-dir', type=Path, required=True)
    parser.add_argument('--sensors', type=Path, required=True)
    parser.add_argument('--metadata-dir', type=Path, default=Path(__file__).parent / 'physics_metadata')
    parser.add_argument('--identity', type=Path, default=Path(__file__).parent / 'incident_corridor_selection_v1.json')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--event-sidecar', type=Path, help='Optional frozen identity file for training-only report replay')
    args = parser.parse_args(argv)
    result = audit(args.data_dir, args.history_dir, args.sensors, args.metadata_dir,
                   args.identity, args.output_dir, args.event_sidecar)
    import json
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ('inputs_sha256', 'outputs', 'metadata_source')}, indent=2))


if __name__ == '__main__':
    main()
