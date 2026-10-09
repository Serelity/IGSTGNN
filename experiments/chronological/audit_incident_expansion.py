"""Metadata-only incident coverage and byte-verified training-month cache audit.

Candidate matching is NOT an incident impact label or an approved new dataset.
Traffic payloads are hashed, never decoded; validation/test months are not opened.
"""
import argparse
import calendar
from collections import Counter, defaultdict
import csv
from datetime import datetime, timedelta
from pathlib import Path
import re
import sys

import numpy as np

REPO = Path(__file__).absolute().parents[2]
sys.path.insert(0, str(REPO))
from src.utils.incident_corridor import read_json, read_rows, require, sha256, write_json, write_rows
from experiments.chronological.prepare_incident_corridors import verified

RAW_SHA256 = 'fd186c2b3334c332d2e70c7f8b2fd35b6bf990da2c4aa5c77517bf6361d97d05'
RADII_KM = (1., 3., 5.)  # Sensitivity bands, not certified mapping thresholds.


def training_window(report):
    """Same next-grid, support-boundary and DST-day rules as frozen chronology."""
    t0 = report.replace(minute=report.minute // 5 * 5, second=0, microsecond=0) + timedelta(minutes=5)
    lo, hi = t0 - timedelta(minutes=70), t0 + timedelta(minutes=65)
    if report.tzinfo is not None or lo < datetime(2023, 1, 1) or hi > datetime(2023, 9, 1):
        return None
    if lo < datetime(2023, 3, 13) and hi > datetime(2023, 3, 12):
        return None
    return t0


def road_key(name, direction):
    match = re.fullmatch(r'(?:I|SR|US)?(\d+)(?:\.0)?(?:-[NSEW])?', name.strip())
    require(match is not None and direction in 'NSEW' and len(direction) == 1, 'Invalid road identity')
    suffix = re.search(r'-([NSEW])$', name.strip())
    require(suffix is None or suffix.group(1) == direction, 'Conflicting direction')
    return int(match.group(1)), direction


def sensor_groups(sensors):
    groups = defaultdict(list)
    for s in sensors:
        groups[road_key(s['Fwy'], s['Direction'])].append(s)
    return dict(groups)


def nearest_candidate(row, groups):
    """Same numbered road/direction, geodesic distance, then postmile agreement."""
    try:
        key = road_key(row['Fwy'], row['Freeway_direction'].upper())
        lat, lon, pm = [float(row[k]) for k in ('Latitude', 'Longitude', 'Abs PM')]
        if not np.isfinite([lat, lon, pm]).all() or not (-90 <= lat <= 90 and -180 <= lon <= 180):
            return None
        group = groups.get(key)
        if not group:
            return None
        xy = np.radians([[float(s['Lat']), float(s['Lng'])] for s in group])
        center = np.radians([lat, lon])
        delta = xy - center
        a = np.sin(delta[:, 0] / 2)**2 + np.cos(xy[:, 0]) * np.cos(center[0]) * np.sin(delta[:, 1] / 2)**2
        distances = 2 * 6371.0088 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
        i = int(np.argmin(distances))
        station = group[i]
        return {'road': station['Fwy'], 'direction': key[1],
                'station_id': int(station['station_id']), 'nearest_sensor_km': float(distances[i]),
                'source_postmile_difference': abs(pm - float(station['Abs PM'])),
                'postmile_agrees_within_10_source_units': abs(pm - float(station['Abs PM'])) <= 10}
    except (ValueError, KeyError):
        return None


def cache_month(data_dir, month, axes, expected_hash):
    require(month in range(1, 9), 'Training months only')
    path = data_dir / f'source_month_{month:02d}.json'
    verified(path, expected_hash)
    meta = read_json(path)
    slots = calendar.monthrange(2023, month)[1] * 288
    layout = meta['layout']
    require(meta['month'] == month and not meta.get('failures') and
            layout['shape'] == [16972, slots, 3] and layout['dtype'] == '<f4', 'Invalid source layout')
    rows = sorted(meta['rows'], key=lambda r: r['published_node_index'])
    require([r['published_node_index'] for r in rows] == list(range(len(axes))), 'Incomplete node axis')
    for i, row in enumerate(rows):
        identity = row['identity']
        start = layout['data_offset'] + int(axes[i]) * slots * 12
        require(identity['dataset'] == 'gpxlcj/xtraffic' and identity['version'] == 8 and
                identity['year'] == 2023 and identity['month'] == month and
                identity['file_name'] == f'year_2023/year_2023/2023_p{month:02d}.npy' and
                identity['raw_node_index'] == int(axes[i]) and identity['start'] == start and
                identity['end'] == start + slots * 12 - 1 and row['bytes'] == slots * 12 and
                identity['header_sha256'] == meta['header']['sha256'], 'Cached row identity mismatch')
        blob = data_dir / 'row_cache' / 'blobs' / (row['sha256'] + '.bin')
        require(blob.stat().st_size == row['bytes'], 'Cached row length mismatch')
        verified(blob, row['sha256'])
    return {'month': month, 'station_rows_verified': len(rows), 'nominal_slots_per_station': slots,
            'source_row_bytes_verified': sum(r['bytes'] for r in rows), 'manifest_sha256': expected_hash}


def audit(data_dir, sensors_path, sidecar_path, raw_path, output):
    require(not output.exists() and not output.with_name(output.name + '.partial').exists(), 'Use a new output directory')
    identity = read_json(REPO / 'experiments/chronological/incident_corridor_selection_v1.json')
    verified(data_dir / 'summary.json', identity['data_summary_sha256'])
    verified(data_dir / 'context_manifest.json', identity['context_manifest_sha256'])
    summary, context = read_json(data_dir / 'summary.json'), read_json(data_dir / 'context_manifest.json')
    sources = {k.replace('\\', '/').split('/')[-1]: v for k, v in context['sources'].items()}
    verified(sensors_path, sources['sensors.csv'])
    verified(sidecar_path, sources['event_identity_sidecar.json'])
    verified(raw_path, RAW_SHA256)
    for name in ('raw_node_indices.npy', 'station_ids.npy', 'train_manifest.csv'):
        verified(data_dir / name, summary['files'][name])
    sensors = read_rows(sensors_path)
    ids = np.load(data_dir / 'station_ids.npy', allow_pickle=False)
    axes = np.load(data_dir / 'raw_node_indices.npy', allow_pickle=False)
    require(np.array_equal(ids, [int(s['station_id']) for s in sensors]) and axes.shape == ids.shape,
            'Station axes mismatch')
    months = [cache_month(data_dir, m, axes, summary['files'][f'source_month_{m:02d}.json']) for m in range(1, 9)]
    released = read_json(sidecar_path)
    released_counts = Counter((str(r.get('spatial_roads')), r['status']) for r in released)
    existing = {r['incident_id'] for r in read_rows(data_dir / 'train_manifest.csv')}
    groups, counts, candidates, seen, conflicts = sensor_groups(sensors), Counter(), [], {}, set()
    with raw_path.open(encoding='utf-8-sig', newline='') as stream:
        for source_index, row in enumerate(csv.DictReader(stream, delimiter='\t')):
            counts['source_rows'] += 1
            try:
                report = datetime.strptime(row['dt'], '%m/%d/%Y %H:%M:%S')
                t0 = training_window(report)
            except ValueError:
                counts['invalid_report_time_rows'] += 1
                continue
            if t0 is None:
                continue
            counts['training_time_eligible_rows'] += 1
            match = nearest_candidate(row, groups)
            if match is None or match['nearest_sensor_km'] > max(RADII_KM):
                continue
            event_id = row['incident_id'].strip()
            if not event_id:
                counts['candidate_missing_id_rows'] += 1
                continue
            signature = tuple(row[k] for k in ('dt', 'Fwy', 'Freeway_direction', 'Abs PM', 'Latitude', 'Longitude'))
            if event_id in seen:
                counts['candidate_duplicate_id_rows'] += 1
                if seen[event_id] != signature:
                    conflicts.add(event_id)
                continue
            seen[event_id] = signature
            candidates.append({'source_row_index': source_index, 'incident_id': event_id,
                               'report_time': report.isoformat(), 't0': t0.isoformat(), **match,
                               'already_in_frozen_training': event_id in existing})
    for row in candidates:
        row['identity_conflict'] = row['incident_id'] in conflicts
    roads = []
    for key, group in sorted(groups.items()):
        for radius in RADII_KM:
            chosen = [r for r in candidates if r['road'] == group[0]['Fwy'] and
                      r['direction'] == key[1] and r['nearest_sensor_km'] <= radius and
                      r['postmile_agrees_within_10_source_units'] and not r['identity_conflict']]
            roads.append({'road': group[0]['Fwy'], 'direction': key[1], 'station_count': len(group),
                          'radius_km': radius, 'candidate_distinct_events': len(chosen),
                          'not_in_frozen_training': sum(not r['already_in_frozen_training'] for r in chosen),
                          'nearest_station_count': len({r['station_id'] for r in chosen}),
                          'months_with_candidates': len({r['report_time'][:7] for r in chosen}),
                          'full_network_training_cache_available': all(m['station_rows_verified'] == len(ids) for m in months),
                          'mapping_and_numeric_quality_certified': False})
    result = {'status': 'INCIDENT_EXPANSION_FEASIBILITY_AUDITED_NOT_A_NEW_DATASET',
              'source_counts': dict(counts), 'candidate_conflicting_ids': len(conflicts),
              'published_metadata_rows': len(released),
              'published_spatial_status_counts': [{'spatial_roads': k[0], 'status': k[1], 'count': v}
                                                  for k, v in sorted(released_counts.items())],
              'training_cache': months, 'roads': roads,
              'policy': {'geodesic_radii_km_sensitivity_only': list(RADII_KM),
                         'source_postmile_tolerance': 10, 'training_months': list(range(1, 9)),
                         'scope': 'offline_conditional_metadata_candidates',
                         'traffic_values_decoded': False, 'validation_test_traffic_months_opened': False,
                         'identity_conflict_scope': 'training_time_candidates_within_5km',
                         'duration_type_description_used': False, 'frozen_data_modified': False,
                         'training_performed': False, 'new_windows_materialized': False,
                         'earth_radius_km_convention': 6371.0088},
              'inputs_sha256': {'data_summary': sha256(data_dir / 'summary.json'),
                               'sensors': sha256(sensors_path), 'identity_sidecar': sha256(sidecar_path),
                               'raw_events': sha256(raw_path), 'auditor': sha256(Path(__file__))}}
    partial = output.with_name(output.name + '.partial')
    partial.mkdir(parents=True)
    write_rows(partial / 'candidate_events.csv', candidates)
    write_rows(partial / 'road_expansion_coverage.csv', roads)
    result['outputs_sha256'] = {p.name: sha256(p) for p in partial.iterdir()}
    write_json(partial / 'summary.json', result)
    partial.rename(output)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('data-dir', 'sensors', 'event-sidecar', 'raw-events', 'output-dir'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    r = audit(a.data_dir, a.sensors, a.event_sidecar, a.raw_events, a.output_dir)
    import json
    print(json.dumps({k: r[k] for k in ('status', 'source_counts', 'candidate_conflicting_ids', 'published_spatial_status_counts', 'roads')}, indent=2))


if __name__ == '__main__':
    main()
