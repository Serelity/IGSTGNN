"""Prepare train-X-only physical evidence; never certify or fit a bottleneck.

Full source metadata detects known ramp/connector contamination hidden by the
mainline-only model package. Historical LargeST lanes are a cross-check only.
"""

import argparse
from collections import Counter
import csv
from datetime import datetime, timedelta
import gzip
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.chronological.smoke import sha256, verify_package
from src.utils.chronological import ChronologicalDataset, read_rows
from src.utils.traffic_physics_contract import check_manifest_timing, contract_template

PAPER = 'https://proceedings.neurips.cc/paper_files/paper/2025/file/7813e19a86fd73d40f7e811ab15f6d5f-Paper-Datasets_and_Benchmarks_Track.pdf'


def unique_history(flow, rows):
    """Deduplicate only X by nominal time; check all repeated station values."""
    if flow.ndim != 3 or flow.shape[:2] != (len(rows), 26) or not rows:
        raise ValueError('Expected nonempty raw train flow [samples,26,nodes]')
    times = []
    for row in rows:
        start = datetime.fromisoformat(row['x_start'])
        if row.get('split') != 'train' or int(row.get('source_version', 0)) != 8:
            raise ValueError('Only v8 train rows may define calibration support')
        if start.tzinfo is not None or start.second or start.microsecond or start.minute % 5:
            raise ValueError('Expected naive five-minute source labels')
        slots = [start + timedelta(minutes=5 * k) for k in range(12)]
        if any(t.year != 2023 or not 1 <= t.month <= 8 for t in slots):
            raise ValueError('X calibration support must stay in January-August 2023')
        if datetime.fromisoformat(row['x_end']) != slots[-1]:
            raise ValueError('History end does not match twelve five-minute labels')
        times.extend(slots)
    labels, first, inverse = np.unique(np.asarray(times, dtype='datetime64[m]'),
                                       return_index=True, return_inverse=True)
    history = np.asarray(flow[:, :12]).reshape(-1, flow.shape[2])
    values = history[first].copy()
    # Bounded comparison avoids a second full expanded temporary matrix.
    for start in range(0, len(history), 1024):
        if not np.array_equal(history[start:start + 1024], values[inverse[start:start + 1024]], equal_nan=True):
            raise ValueError('Repeated nominal station/time has inconsistent values')
    return labels, values


def index_metadata(rows, id_key):
    result = {int(r[id_key]): r for r in rows}
    if len(result) != len(rows):
        raise ValueError('Duplicate station IDs in metadata')
    return result


def check_source_metadata(published, raw):
    by_id = index_metadata(raw, 'station_id')
    for r in published:
        source = by_id.get(int(r['station_id']))
        if source is None:
            raise ValueError('Published station missing from full source metadata')
        if any(r[a] != source[b] for a, b in (
                ('Fwy', 'Fwy Name'), ('Direction', 'Direction'), ('Type', 'Type'), ('County', 'County'))):
            raise ValueError('Published road/direction/type/county differs from source metadata')
        for field in ('Abs PM', 'Lat', 'Lng'):
            a, b = float(r[field]), float(source[field])
            if not np.isfinite(a) or not np.isfinite(b) or abs(a - b) > 1e-6:
                raise ValueError('Published coordinates/postmile differ from source metadata')
        if r['Type'] != 'Mainline':
            raise ValueError('This evidence stage expects the mainline-only package')


def historical_lane_audit(published, lane_rows):
    old = index_metadata(lane_rows, 'ID')
    result = []
    for r in published:
        prior = old.get(int(r['station_id']))
        match = bool(prior and prior['Fwy'] == r['Fwy'] and prior['Direction'] == r['Direction']
                     and prior['Type'] == r['Type']
                     and all(abs(float(prior[k]) - float(r[k])) <= 1e-4 for k in ('Lat', 'Lng')))
        lanes = int(prior['Lanes']) if prior else None
        if prior and lanes < 1:
            raise ValueError('Invalid historical lane count')
        result.append({'station_id': int(r['station_id']), 'historical_id_match': prior is not None,
                       'historical_road_coordinate_match': match, 'historical_lanes': lanes,
                       'lane_count_2023_certified': False})
    return result


def candidate_inventory(published, raw, values, support):
    if values.ndim != 2 or support.ndim != 2 or values.shape[1] != len(published) or support.shape[1] != len(published):
        raise ValueError('Station axes differ in candidate inputs')
    groups = {}
    for i, r in enumerate(published):
        groups.setdefault((r['Fwy'], r['Direction']), []).append((i, r))
    result = []
    for (road, direction), group in sorted(groups.items()):
        ordered = sorted(group, key=lambda x: (float(x[1]['Abs PM']), int(x[1]['station_id'])))
        full = [r for r in raw if r['Fwy Name'] == road and r['Direction'] == direction]
        if any(not np.isfinite(float(r['Abs PM'])) for r in full):
            raise ValueError('Invalid full-source postmile')
        for (i, left), (j, right) in zip(ordered, ordered[1:]):
            lo, hi = float(left['Abs PM']), float(right['Abs PM'])
            nonmain = [r for r in full if r['Type'] != 'Mainline' and lo - 1e-6 <= float(r['Abs PM']) <= hi + 1e-6]
            coincident = [r for r in full if r['Type'] == 'Mainline'
                          and int(r['station_id']) not in (int(left['station_id']), int(right['station_id']))
                          and (abs(float(r['Abs PM']) - lo) <= 1e-6 or abs(float(r['Abs PM']) - hi) <= 1e-6)]
            valid = np.isfinite(values[:, [i, j]]) & (values[:, [i, j]] >= 0)
            flags = []
            if hi - lo <= 1e-6:
                flags.append('same_postmile')
            if nonmain:
                flags.append('known_ramp_or_connector_within_closed_postmile_interval')
            if coincident:
                flags.append('additional_mainline_at_boundary_postmile')
            result.append({'road': road, 'direction': direction,
                           'low_postmile_station_id': int(left['station_id']),
                           'high_postmile_station_id': int(right['station_id']),
                           'low_postmile': lo, 'high_postmile': hi, 'postmile_difference': hi - lo,
                           'known_nonmainline_count': len(nonmain),
                           'known_nonmainline_ids': '|'.join(str(r['station_id']) for r in nonmain),
                           'known_nonmainline_types': '|'.join(sorted(set(r['Type'] for r in nonmain))),
                           'additional_boundary_mainline_count': len(coincident),
                           'metadata_flags': '|'.join(flags),
                           'metadata_prefilter_pass': not flags,
                           'both_valid_unique_train_x_fraction': float(valid.all(1).mean()),
                           'both_report_supported_train_windows': int((support[:, i] & support[:, j]).sum()),
                           'physical_direction_certified': False, 'closed_boundary_certified': False,
                           'arrival_delay_certified': False, 'capacity_calibrated': False})
    return result


def station_profiles(values, ids, support):
    result = []
    for i, station in enumerate(ids):
        valid = values[:, i][np.isfinite(values[:, i]) & (values[:, i] >= 0)]
        quantiles = np.quantile(valid, [.5, .9, .95, .99, 1]).tolist() if len(valid) else [None] * 5
        result.append({'station_id': int(station), 'valid_unique_train_x_count': len(valid),
                       'zero_count': int((valid == 0).sum()),
                       'integer_fraction': float(np.mean(valid == np.round(valid))) if len(valid) else None,
                       **dict(zip(('q50_source_units', 'q90_source_units', 'q95_source_units', 'q99_source_units', 'max_source_units'), quantiles)),
                       'report_supported_train_windows': int(support[:, i].sum()),
                       'these_quantiles_are_capacity_estimates': False})
    return result


def compare_pems_records(path, labels, values, ids):
    """Compare a caller-supplied official Station 5-Minute export on train X.

    File format: timestamp, station, district, freeway, direction, lane type,
    length, samples, percent observed, total flow, occupancy, speed, ...
    Content matching does not authenticate where the supplied file came from.
    """
    positions = {int(x): i for i, x in enumerate(ids)}
    times = {int(t): i for i, t in enumerate(labels.astype('int64'))}
    seen, matched_times, matched_stations = {}, set(), set()
    equal_count, equal_hour = 0, 0
    open_file = gzip.open if path.suffix == '.gz' else open
    with open_file(path, 'rt', encoding='utf-8-sig', newline='') as stream:
        for row in csv.reader(stream):
            if not row:
                continue
            if row[0].strip().lower() == 'timestamp':
                continue
            if len(row) < 12:
                raise ValueError('Expected a Station 5-Minute source row with at least 12 fields')
            station = int(row[1])
            if station not in positions:
                continue
            stamp = np.datetime64(datetime.strptime(row[0], '%m/%d/%Y %H:%M:%S'), 's')
            # Reject second offsets instead of silently flooring to a nominal slot.
            if int(stamp.astype('int64')) % 300:
                raise ValueError('Source record is not aligned to five-minute labels')
            minute = int(stamp.astype('datetime64[m]').astype('int64'))
            if minute not in times:
                continue
            if row[5].strip() != 'ML' or int(row[2]) != 4:
                raise ValueError('Matched source station is not District 4 mainline')
            if not row[9].strip():
                continue
            source = float(row[9])
            i, j = times[minute], positions[station]
            actual = float(values[i, j])
            if not np.isfinite(source) or source < 0 or not np.isfinite(actual) or actual < 0:
                continue
            key = (i, j)
            if key in seen:
                if seen[key] != source:
                    raise ValueError('Conflicting duplicate PeMS records')
                continue
            seen[key] = source
            matched_times.add(minute)
            matched_stations.add(station)
            equal_count += abs(actual - source) <= 1e-4
            equal_hour += abs(actual - 12 * source) <= 1e-4
    count = len(seen)
    enough = count >= 100 and len(matched_times) >= 2 and len(matched_stations) >= 2
    candidates = [unit for unit, n in (('vehicles_per_5min', equal_count), ('vehicles_per_hour', equal_hour))
                  if enough and n == count]
    resolved = candidates[0] if len(candidates) == 1 else None
    return {'status': 'SOURCE_UNIT_MATCH' if resolved else 'SOURCE_UNIT_NOT_RESOLVED',
            'matched_valid_train_x_cells': count, 'distinct_stations': len(matched_stations),
            'distinct_time_labels': len(matched_times), 'equal_as_5min_count': int(equal_count),
            'equal_as_hourly_rate': int(equal_hour), 'inferred_array_flow_unit': resolved,
            'official_file_provenance_independently_certified': False,
            'source_sha256': sha256(path), 'calibration_or_training_enabled': False}


def write_csv(path, rows):
    if not rows:
        raise ValueError('Cannot export an empty evidence table')
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prepare(data_dir, published_path, raw_path, lane_path, output_dir, pems_path=None):
    if output_dir.exists():
        raise FileExistsError('Use a fresh evidence directory')
    package = verify_package(data_dir)
    train = ChronologicalDataset(data_dir, 'train')
    check_manifest_timing(train.rows, 'start')
    published = read_rows(published_path)
    if not published or any(r['County'] != 'Contra Costa' or r['District'] != '4' for r in published):
        raise ValueError('This evidence protocol is restricted to District 4 Contra Costa')
    if [int(r['station_id']) for r in published] != train.station_ids.tolist():
        raise ValueError('Published sensor order differs from training node axis')
    with raw_path.open(encoding='utf-8-sig', newline='') as stream:
        raw = list(csv.DictReader(stream, delimiter='\t'))
    check_source_metadata(published, raw)
    labels, values = unique_history(train.flow, train.rows)
    support = np.any(train.context['distances'] != 0, axis=-1)
    profiles = station_profiles(values, train.station_ids, support)
    pairs = candidate_inventory(published, raw, values, support)
    lanes = historical_lane_audit(published, read_rows(lane_path)) if lane_path else None
    source_match = compare_pems_records(pems_path, labels, values, train.station_ids) if pems_path else None
    counties = {r['County'] for r in published}
    county_raw = [r for r in raw if r['County'] in counties]
    source_records = {str(p): sha256(p) for p in (published_path, raw_path, Path(__file__))}
    if lane_path:
        source_records[str(lane_path)] = sha256(lane_path)
    if pems_path:
        source_records[str(pems_path)] = sha256(pems_path)
    template = contract_template(train.station_ids, package)
    # Publisher declaration narrows one question; this is not source-clock certification.
    template['interval_label'] = 'start'
    template['evidence']['aggregation'] = PAPER + ' section 4.1.1, printed p6; publisher-declared interval starts; timezone/DST/latency not certified'
    output_dir.mkdir(parents=True)
    write_csv(output_dir / 'unique_train_x_station_profiles.csv', profiles)
    write_csv(output_dir / 'bottleneck_candidate_inventory.csv', pairs)
    if lanes:
        write_csv(output_dir / 'historical_lane_crosscheck.csv', lanes)
    spec = {'status': 'SOURCE_RECORD_REQUIRED', 'purpose': 'Verify v8 array conversion against official Station 5-Minute total-flow records',
            'year': 2023, 'district': 4, 'required_columns': ['Timestamp', 'Station', 'Total Flow', '% Observed', 'Lane Type'],
            'match_keys': ['station_id', 'interval_start_label'],
            'allowed_values': 'Only unique January-August training X; no validation, gap or Y calibration',
            'capacity_scope': 'A source-unit match does not calibrate bottleneck capacity'}
    valid = values[np.isfinite(values) & (values >= 0)]
    report = {'status': 'PHYSICS_EVIDENCE_PREPARATION_COMPLETE', 'training_started': False, 'main_training_ready': False,
              'readiness': 'PHYSICAL_CONTRACT_REQUIRED', 'physical_contract_exported': False,
              'package_sha256': package, 'source_sha256': source_records,
              'train_windows': len(train), 'stations': len(train.station_ids),
              'unique_train_x_nominal_slots': len(labels), 'train_window_x_slots': len(train) * 12,
              'duplicate_x_slots_removed': len(train) * 12 - len(labels),
              'profile_scope': 'unique_train_X_only_not_capacity_calibration',
              'validation_gap_and_target_values_used_for_statistics': False,
              'source_interval_label': 'start', 'time_evidence_level': 'publisher_declared_not_independent_clock_certification',
              'time_bounds_relative_to_T_minutes': {'last_X_end': -5, 'first_Y_start': 5, 'last_Y_end': 65},
              'flow_unit_status': 'UNRESOLVED_V8_CONVERSION_CHAIN',
              'optional_source_record_comparison': source_match,
              'observed_integer_fraction': float(np.mean(valid == np.round(valid))),
              'published_types': dict(Counter(r['Type'] for r in published)),
              'full_source_county_types': dict(Counter(r['Type'] for r in county_raw)),
              'candidate_pairs': len(pairs), 'metadata_prefilter_pass_pairs': sum(p['metadata_prefilter_pass'] for p in pairs),
              'pairs_with_known_nonmainline': sum(p['known_nonmainline_count'] > 0 for p in pairs),
              'certified_bottlenecks': 0, 'capacity_calibration_performed': False,
              'historical_lane_id_matches': sum(r['historical_id_match'] for r in lanes) if lanes else None,
              'historical_lane_road_coordinate_matches': sum(r['historical_road_coordinate_match'] for r in lanes) if lanes else None,
              'lane_count_2023_certified': False,
              'remaining_evidence': ['v8_flow_units_and_lane_aggregation', 'bottleneck_boundaries_and_arrival_delay', 'capacity_and_queue_scale_calibration']}
    for name, payload in [('contract_draft_UNRESOLVED.json', template), ('source_record_request.json', spec), ('summary.json', report)]:
        (output_dir / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--published-sensors', type=Path, required=True)
    parser.add_argument('--raw-sensors', type=Path, required=True)
    parser.add_argument('--historical-lanes', type=Path)
    parser.add_argument('--pems-5min', type=Path, help='Optional official raw CSV/txt or gzip export; compare train X only')
    parser.add_argument('--output-dir', type=Path, required=True)
    a = parser.parse_args()
    print(json.dumps(prepare(a.data_dir, a.published_sensors, a.raw_sensors, a.historical_lanes, a.output_dir, a.pems_5min), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
