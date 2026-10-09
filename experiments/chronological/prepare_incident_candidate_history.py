"""Build a separate all-node candidate train-X pack from the local v8 cache."""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timedelta
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).absolute().parents[2]
sys.path.insert(0, str(REPO))
from experiments.chronological.audit_incident_expansion import (
    cache_month, nearest_candidate, road_key, sensor_groups, training_window,
)
from experiments.chronological.prepare_incident_corridors import verified
from experiments.chronological.prepare_context import spatial_features
from src.utils.incident_corridor import read_json, read_rows, require, sha256, write_json, write_rows
from src.utils.incident_candidate_history import CandidateHistory, SCHEMA, station_diagnostics


def select_events(candidates, raw_rows, sensors, rules, baseline):
    """Selection uses identity and location only, never traffic values."""
    chosen, excluded = {}, []
    for c in candidates:
        reason = None
        if c['identity_conflict'] != 'False':
            reason = 'previous_identity_conflict'
        elif float(c['nearest_sensor_km']) > rules['maximum_nearest_station_km']:
            reason = 'outside_fixed_1km_candidate_radius'
        elif float(c['source_postmile_difference']) > rules['maximum_source_postmile_difference']:
            reason = 'source_postmile_conflict'
        if reason:
            excluded.append({'source_row_index': c['source_row_index'], 'incident_id': c['incident_id'], 'reason': reason})
        else:
            require(c['incident_id'] not in chosen, 'Duplicate ID in audited candidate table')
            chosen[c['incident_id']] = c
    groups, signatures, conflicts, events = sensor_groups(sensors), {}, set(), []
    duplicate_rows = 0
    for source_index, r in enumerate(raw_rows):
        event_id = r['incident_id'].strip()
        if event_id not in chosen:
            continue
        signature = tuple(r[k] for k in ('dt', 'Fwy', 'Freeway_direction', 'Abs PM', 'Latitude', 'Longitude'))
        if event_id in signatures:
            duplicate_rows += 1
            if signatures[event_id] != signature:
                conflicts.add(event_id)
        else:
            signatures[event_id] = signature
        c = chosen[event_id]
        if source_index != int(c['source_row_index']):
            continue
        report = datetime.strptime(r['dt'], '%m/%d/%Y %H:%M:%S')
        t0 = training_window(report)
        match = nearest_candidate(r, groups)
        require(t0 is not None and t0.isoformat() == c['t0'] and report.isoformat() == c['report_time'], 'Candidate clock replay mismatch')
        require(match is not None and match['road'] == c['road'] and match['station_id'] == int(c['station_id'])
                and abs(match['nearest_sensor_km'] - float(c['nearest_sensor_km'])) < 1e-9
                and match['postmile_agrees_within_10_source_units'], 'Candidate location replay mismatch')
        road, direction = road_key(r['Fwy'], r['Freeway_direction'].upper())
        events.append({'source_row_index': source_index, 'incident_id': event_id, 'report_time': report.isoformat(),
                       't0': t0.isoformat(), 'x_start': (t0 - timedelta(minutes=65)).isoformat(),
                       'x_end': (t0 - timedelta(minutes=10)).isoformat(), 'road': match['road'],
                       'freeway': road, 'direction': direction, 'postmile': float(r['Abs PM']),
                       'latitude': float(r['Latitude']), 'longitude': float(r['Longitude']),
                       'nearest_station_id': match['station_id'], 'nearest_station_km': match['nearest_sensor_km'],
                       'baseline_sample_index': baseline.get(event_id, {}).get('sample_index', ''),
                       'split': 'train', 'source_version': 8, 'mapping_status': 'CANDIDATE_NOT_TOPOLOGY_CERTIFIED'})
    require(len(events) == len(chosen), 'Audited source row not recovered')
    for e in events:
        if e['incident_id'] in conflicts:
            excluded.append({'source_row_index': e['source_row_index'], 'incident_id': e['incident_id'], 'reason': 'conflicting_source_id_metadata'})
    events = sorted([e for e in events if e['incident_id'] not in conflicts],
                    key=lambda e: (e['t0'], e['report_time'], e['incident_id'], e['source_row_index']))
    require(bool(events), 'No candidate events remain')
    for i, event in enumerate(events):
        event['candidate_index'] = i
    return events, excluded, duplicate_rows


def history_plan(events):
    labels = []
    for event in events:
        issued, cutoff = datetime.fromisoformat(event['report_time']), datetime.fromisoformat(event['t0'])
        require(training_window(issued) == cutoff, 'Event outside frozen train-time rules')
        slots = [cutoff - timedelta(minutes=65) + timedelta(minutes=5*k) for k in range(12)]
        require(event['x_start'] == slots[0].isoformat() and event['x_end'] == slots[-1].isoformat(), 'X offset mismatch')
        labels.append(slots)
    unique, inverse = np.unique(np.asarray(labels, dtype='datetime64[m]'), return_inverse=True)
    return unique, inverse.reshape(len(events), 12).astype(np.int32)


def extract_node_x(blob, month_slots, selected):
    selected = np.asarray(selected)
    require(selected.ndim == 1 and len(selected) > 0 and np.issubdtype(selected.dtype, np.integer)
            and selected.min() >= 0 and selected.max() < month_slots
            and (np.diff(selected) > 0).all(), 'Invalid unique X slots')
    source = np.memmap(blob, dtype='<f4', mode='r', shape=(month_slots, 3))
    try:
        return np.asarray(source[selected]).copy()
    finally:
        source._mmap.close()


def road_order(sensors):
    roads = []
    for key, group in sorted(sensor_groups(sensors).items()):
        ties = defaultdict(list)
        for s in group:
            ties[float(s['Abs PM'])].append(int(s['station_id']))
        ordered = sorted(ties, reverse=key[1] in ('S', 'W'))
        roads.append({'road': group[0]['Fwy'], 'direction': key[1],
                      'travel_postmile_order_assumption': 'descending' if key[1] in ('S', 'W') else 'ascending',
                      'ordered_groups': [{'source_postmile': p, 'station_ids': sorted(ties[p])} for p in ordered],
                      'coincident_groups': sum(len(v) > 1 for v in ties.values()),
                      'between_group_order_candidates': max(0, len(ordered)-1),
                      'direct_connections_certified': False, 'physical_lengths_defined': False})
    return roads


def prepare(data_dir, audit_dir, raw_events, sensors_path, protocol_path, output):
    partial = output.with_name(output.name + '.partial')
    require(not output.exists() and not partial.exists(), 'Use a fresh output directory')
    protocol = read_json(protocol_path)
    require(protocol['schema'] == SCHEMA, 'Wrong candidate protocol')
    verified(data_dir / 'summary.json', protocol['data_summary_sha256'])
    verified(raw_events, protocol['raw_event_sha256'])
    verified(sensors_path, protocol['sensors_sha256'])
    candidate_path = audit_dir / 'candidate_events.csv'
    verified(candidate_path, protocol['candidate_table_sha256'])
    summary, sensors = read_json(data_dir / 'summary.json'), read_rows(sensors_path)
    for filename in ('station_ids.npy', 'raw_node_indices.npy', 'train_manifest.csv'):
        verified(data_dir / filename, summary['files'][filename])
    ids = np.load(data_dir / 'station_ids.npy', allow_pickle=False)
    axes = np.load(data_dir / 'raw_node_indices.npy', allow_pickle=False)
    require(np.array_equal(ids, [int(s['station_id']) for s in sensors]) and axes.shape == ids.shape, 'Station axes mismatch')
    baseline = {r['incident_id']: r for r in read_rows(data_dir / 'train_manifest.csv')}
    with raw_events.open(encoding='utf-8-sig', newline='') as stream:
        events, excluded, duplicate_rows = select_events(read_rows(candidate_path), csv.DictReader(stream, delimiter='\t'),
                                                        sensors, protocol['selection'], baseline)
    labels, index = history_plan(events)
    partial.mkdir(parents=True)
    write_json(partial / 'protocol.json', protocol)
    write_rows(partial / 'train_events.csv', events)
    if excluded:
        write_rows(partial / 'excluded_candidates.csv', excluded)
    np.save(partial / 'history_labels.npy', labels, allow_pickle=False)
    np.save(partial / 'history_index.npy', index, allow_pickle=False)
    np.save(partial / 'station_ids.npy', ids, allow_pickle=False)
    shape = (len(labels), len(ids), 3)
    values = np.lib.format.open_memmap(partial / 'history_values.npy', mode='w+', dtype='float32', shape=shape)
    usable = np.lib.format.open_memmap(partial / 'history_usable.npy', mode='w+', dtype='bool', shape=shape)
    month_ids = labels.astype('datetime64[M]')
    cache_checks, written = [], np.zeros(len(labels), dtype=bool)
    for month in protocol['time']['training_months']:
        key = np.datetime64(f'2023-{month:02d}', 'M')
        positions = np.flatnonzero(month_ids == key)
        if not len(positions):
            continue
        checked = cache_month(data_dir, month, axes, summary['files'][f'source_month_{month:02d}.json'])
        cache_checks.append(checked)
        meta = read_json(data_dir / f'source_month_{month:02d}.json')
        selected = ((labels[positions] - key.astype('datetime64[m]')).astype(np.int64) // 5)
        for row in meta['rows']:
            node = row['published_node_index']
            x = extract_node_x(data_dir / 'row_cache/blobs' / (row['sha256'] + '.bin'), checked['nominal_slots_per_station'], selected)
            values[positions, node] = x
            usable[positions, node] = np.isfinite(x) & (x >= 0)
        require(not written[positions].any(), 'Repeated month assignment')
        written[positions] = True
        print(f'Extracted training X: month={month}, unique_labels={len(positions)}', flush=True)
    require(written.all(), 'Unwritten history labels')
    values.flush()
    usable.flush()
    stations = []
    for i, sensor in enumerate(sensors):
        diagnostic = station_diagnostics(np.asarray(values[:, i]), labels, protocol['diagnostics'])
        stations.append({'station_id': int(ids[i]), 'road': sensor['Fwy'], 'direction': sensor['Direction'], **diagnostic})
    report = np.asarray([spatial_features(e, sensors) for e in events], dtype=np.float32)
    support = np.any(report != 0, axis=-1)
    ages = [(datetime.fromisoformat(e['t0']) - datetime.fromisoformat(e['report_time'])).total_seconds()/60 for e in events]
    np.savez_compressed(partial / 'train_report.npz', distances=report, report_age_minutes=np.asarray(ages, dtype=np.float32),
                        station_ids=ids, candidate_indices=np.arange(len(events), dtype=np.int64))
    roads = []
    for key, group in sorted(sensor_groups(sensors).items()):
        nodes = [i for i, s in enumerate(sensors) if s['Fwy'] == group[0]['Fwy'] and s['Direction'] == key[1]]
        road_events = [e for e in events if e['road'] == group[0]['Fwy'] and e['direction'] == key[1]]
        fractions = [stations[i]['joint_usable_slots']/len(labels) for i in nodes]
        roads.append({'road': group[0]['Fwy'], 'direction': key[1], 'stations': len(nodes), 'events': len(road_events),
                      'new_events': sum(e['baseline_sample_index'] == '' for e in road_events),
                      'stations_with_report_support': int(support[:, nodes].any(0).sum()),
                      'minimum_joint_numeric_fraction': min(fractions), 'mean_joint_numeric_fraction': float(np.mean(fractions)),
                      'stations_with_ratio_diagnostic': sum(stations[i]['alpha_early_source_units'] is not None for i in nodes),
                      'physics_certified': False})
    # Per-event quality over its complete X, without flattening repeated histories for population statistics.
    qualities = []
    for i, event in enumerate(events):
        mask = np.asarray(usable[index[i]])
        qualities.append({'candidate_index': i, 'incident_id': event['incident_id'],
                          'joint_numeric_fraction': float(mask.all(-1).mean()),
                          'direct_report_support_stations': int(support[i].sum())})
    write_rows(partial / 'station_observation_diagnostics.csv', stations)
    write_rows(partial / 'road_coverage.csv', roads)
    write_rows(partial / 'event_quality.csv', qualities)
    write_json(partial / 'road_order_candidates.json', road_order(sensors))
    result = {'schema': SCHEMA, 'status': 'CANDIDATE_TRAIN_X_PACK_COMPLETE', 'physics_ready': False,
              'model_training_ready': False, 'samples': len(events), 'stations': len(ids), 'road_direction_groups': len(roads),
              'logical_history_shape': [len(events), 12, len(ids), 3], 'stored_history_shape': list(shape),
              'unique_history_labels': len(labels), 'repeated_event_history_slots': len(events)*12-len(labels),
              'distinct_cutoffs': len({e['t0'] for e in events}), 'extra_events_sharing_cutoff': len(events)-len({e['t0'] for e in events}),
              'source_duplicate_id_rows': duplicate_rows, 'excluded_candidate_count': len(excluded),
              'excluded_reasons': dict(Counter(e['reason'] for e in excluded)),
              'baseline_events_retained': sum(e['baseline_sample_index'] != '' for e in events),
              'new_events': sum(e['baseline_sample_index'] == '' for e in events),
              'baseline_events_not_in_candidate_pack': len(baseline)-sum(e['baseline_sample_index'] != '' for e in events),
              'stations_with_report_support': int(support.any(0).sum()),
              'joint_numeric_cells': sum(s['joint_usable_slots'] for s in stations),
              'joint_numeric_denominator': len(labels)*len(ids), 'roads': roads, 'cache_verification': cache_checks,
              'relative_references_scope': 'unique_candidate_train_X_station_channel_p95_not_capacity',
              'observation_relation_scope': 'descriptive_early_fit_late_check_within_training_only',
              'traffic_values_decoded': 'candidate_X_union_only', 'gap_or_Y_used': False,
              'validation_test_traffic_opened': False, 'incident_duration_type_description_used': False,
              'imputation_performed': False, 'normalization_applied_to_arrays': False,
              'baseline_modified': False, 'online_semantics_certified': False,
              'inputs_sha256': {'protocol': sha256(protocol_path), 'candidate_table': sha256(candidate_path),
                               'raw_events': protocol['raw_event_sha256'], 'sensors': protocol['sensors_sha256'],
                               'data_summary': protocol['data_summary_sha256'], 'builder': sha256(Path(__file__)),
                               'reader_and_diagnostics': sha256(REPO / 'src/utils/incident_candidate_history.py')}}
    values._mmap.close()
    usable._mmap.close()
    result['outputs_sha256'] = {p.name: sha256(p) for p in sorted(partial.iterdir())}
    write_json(partial / 'summary.json', result)
    reader = CandidateHistory(partial)
    require(reader.window(0)['history_source_units'].shape == (12, len(ids), 3), 'Consumer read-back mismatch')
    reader.close()
    partial.rename(output)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('data-dir', 'audit-dir', 'raw-events', 'sensors', 'output-dir'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--protocol', type=Path, default=Path(__file__).with_name('incident_candidate_history_v1.json'))
    a = p.parse_args()
    result = prepare(a.data_dir, a.audit_dir, a.raw_events, a.sensors, a.protocol, a.output_dir)
    import json
    print(json.dumps({k: result[k] for k in ('status', 'samples', 'stations', 'logical_history_shape', 'stored_history_shape',
                                          'stations_with_report_support', 'joint_numeric_cells', 'joint_numeric_denominator', 'physics_ready')}, indent=2))


if __name__ == '__main__':
    main()
