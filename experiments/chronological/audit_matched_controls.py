"""Audit locally incident-free matched controls without reading traffic X or Y."""

import argparse
import bisect
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
from datetime import date, datetime, timedelta
import hashlib
import json
import math
from pathlib import Path
import re


REQUIRED_SPLITS = ('train', 'val')
SAMPLE_FIELDS = [
    'split', 'sample_index', 'incident_id', 'incident_t0', 'freeway', 'direction',
    'postmile', 'affected_node_count', 'calendar_candidate_count',
    'clean_candidate_count', 'nearest_clean_day_distance',
]
PAIR_FIELDS = [
    'split', 'sample_index', 'incident_id', 'incident_t0', 'candidate_t0',
    'absolute_day_distance', 'freeway', 'direction', 'incident_postmile',
    'affected_node_count',
]


@dataclass(frozen=True)
class Incident:
    report_time: datetime
    postmile: float
    post_report_minutes: float
    incident_id: str


@dataclass
class IncidentIndex:
    events: dict
    times: dict
    max_post_minutes: dict
    quality: dict


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path, delimiter=','):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream, delimiter=delimiter))


def parse_freeway(value):
    match = re.search(r'\d+', str(value))
    if match is None:
        raise ValueError(f'No numeric freeway in {value!r}')
    return int(match.group())


def parse_direction(value):
    direction = str(value).strip().upper()
    if direction not in {'N', 'S', 'E', 'W'}:
        raise ValueError(f'Invalid freeway direction {value!r}')
    return direction


def duration_minutes(value, fallback, minimum):
    try:
        parsed = float(str(value).strip())
    except ValueError:
        parsed = fallback
    if not math.isfinite(parsed) or parsed < 0:
        parsed = fallback
    return max(parsed, minimum)


def load_protocol(path):
    protocol = json.loads(Path(path).read_text(encoding='utf-8'))
    if protocol.get('scope') != 'metadata_only_candidate_audit':
        raise ValueError('Protocol scope must remain metadata-only candidate audit')
    if protocol.get('main_training_ready') is not False:
        raise ValueError('Candidate audit cannot declare main training readiness')
    if int(protocol.get('expected_sensor_count', 0)) <= 0:
        raise ValueError('Protocol must declare a positive expected sensor count')
    matching = protocol['candidate_matching']
    if not matching.get('exact_weekday') or not matching.get('exact_five_minute_slot'):
        raise ValueError('v1 requires exact weekday and five-minute-slot matching')
    if not matching.get('exclude_same_t0'):
        raise ValueError('v1 must exclude the incident window itself')
    if int(matching['search_radius_days']) <= 0:
        raise ValueError('Search radius must be positive')
    window = protocol['window']
    if int(window['support_start_minutes_from_t0']) >= 0:
        raise ValueError('Window support must begin before t0')
    if int(window['support_end_exclusive_minutes_from_t0']) <= 0:
        raise ValueError('Window support must end after t0')
    if set(protocol['splits']) != set(REQUIRED_SPLITS):
        raise ValueError('Only the frozen train and validation development splits are allowed')
    for split in REQUIRED_SPLITS:
        bounds = protocol['splits'][split]
        if datetime.fromisoformat(bounds['start']) >= datetime.fromisoformat(bounds['end_exclusive']):
            raise ValueError(f'Invalid {split} boundaries')
    if 'forecast_Y' not in protocol.get('prohibited_matching_inputs', []):
        raise ValueError('Protocol must explicitly prohibit matching on forecast Y')
    if protocol['incident_blackout'].get('duration_unit_assumption') != 'minutes_not_independently_certified':
        raise ValueError('Duration unit uncertainty must remain explicit')
    return protocol


def load_sensors(path):
    result = defaultdict(list)
    station_ids = set()
    for row in read_csv(path):
        station_id = int(row['station_id'])
        if station_id in station_ids:
            raise ValueError(f'Duplicate station_id {station_id}')
        station_ids.add(station_id)
        road = (parse_freeway(row['Fwy']), parse_direction(row['Direction']))
        postmile = float(row['Abs PM'])
        if not math.isfinite(postmile):
            raise ValueError(f'Non-finite sensor postmile for {station_id}')
        result[road].append(postmile)
    if not result:
        raise ValueError('No valid sensors')
    for road in result:
        result[road].sort()
    return dict(result), len(station_ids)


def load_incidents(path, sensors, blackout, postmile_radius, source_year):
    fallback = float(blackout['duration_missing_or_invalid_minutes'])
    minimum = float(blackout['minimum_post_report_minutes'])
    quality = Counter()
    events = defaultdict(list)
    for row in read_csv(path, delimiter='\t'):
        quality['source_rows'] += 1
        try:
            report = datetime.strptime(row['dt'], '%m/%d/%Y %H:%M:%S')
            road = (parse_freeway(row['Fwy']), parse_direction(row['Freeway_direction']))
            postmile = float(row['Abs PM'])
            if not math.isfinite(postmile) or not row['incident_id']:
                raise ValueError('Missing incident identity or postmile')
        except (KeyError, ValueError):
            quality['invalid_required_fields'] += 1
            continue
        if report.year != source_year:
            quality['outside_source_year'] += 1
            continue
        if road not in sensors:
            quality['outside_sensor_road_keys'] += 1
            continue
        if not any(abs(postmile - sensor_pm) <= postmile_radius for sensor_pm in sensors[road]):
            quality['outside_all_sensor_postmile_exposure'] += 1
            continue
        raw_duration = str(row.get('duration', '')).strip()
        try:
            parsed_duration = float(raw_duration)
        except ValueError:
            parsed_duration = None
        if parsed_duration is None or not math.isfinite(parsed_duration):
            quality['indexed_duration_missing_or_nonfinite'] += 1
        elif parsed_duration < 0:
            quality['indexed_duration_negative'] += 1
        elif parsed_duration < minimum:
            quality['indexed_duration_raised_to_minimum'] += 1
        post = duration_minutes(raw_duration, fallback, minimum)
        events[road].append(Incident(report, postmile, post, row['incident_id']))
        quality['indexed_within_sensor_exposure'] += 1
    times, maxima = {}, {}
    for road, values in events.items():
        values.sort(key=lambda event: event.report_time)
        times[road] = [event.report_time for event in values]
        maxima[road] = max(event.post_report_minutes for event in values)
    return IncidentIndex(dict(events), times, maxima, dict(quality))


def validate_manifests(data_dir, source_version):
    rows = {}
    sample_indices = set()
    for split in REQUIRED_SPLITS:
        values = read_csv(Path(data_dir) / f'{split}_manifest.csv')
        if not values:
            raise ValueError(f'{split} manifest is empty')
        for row in values:
            sample = int(row['sample_index'])
            t0 = datetime.fromisoformat(row['t0'])
            if row['split'] != split or int(row['source_version']) != source_version:
                raise ValueError(f'{split} manifest mixes split or source version')
            if t0.second or t0.microsecond or t0.minute % 5:
                raise ValueError('Manifest t0 must use the nominal five-minute grid')
            if sample in sample_indices:
                raise ValueError(f'Duplicate sample_index {sample}')
            sample_indices.add(sample)
        rows[split] = values
    return rows


def candidate_times(origin, split_start, split_end, protocol):
    matching = protocol['candidate_matching']
    radius = int(matching['search_radius_days'])
    window = protocol['window']
    support_before = timedelta(minutes=-int(window['support_start_minutes_from_t0']))
    support_after = timedelta(minutes=int(window['support_end_exclusive_minutes_from_t0']))
    excluded = {date.fromisoformat(value) for value in matching['excluded_nominal_dates']}
    candidates = []
    for offset in range(-radius, radius + 1, 7):
        if offset == 0:
            continue
        candidate = origin + timedelta(days=offset)
        if candidate.date() in excluded:
            continue
        if candidate - support_before < split_start or candidate + support_after > split_end:
            continue
        if candidate.weekday() != origin.weekday() or candidate.time() != origin.time():
            raise AssertionError('Calendar shift violated exact matching')
        candidates.append(candidate)
    return candidates


def locally_clean(candidate, road, affected_sensor_postmiles, incident_index, protocol):
    if not affected_sensor_postmiles:
        raise ValueError('Positive incident has no affected sensor')
    events = incident_index.events.get(road, [])
    if not events:
        return True
    window = protocol['window']
    blackout = protocol['incident_blackout']
    radius = float(protocol['spatial_exposure']['postmile_radius'])
    support_start = candidate + timedelta(minutes=int(window['support_start_minutes_from_t0']))
    support_end = candidate + timedelta(minutes=int(window['support_end_exclusive_minutes_from_t0']))
    pre = float(blackout['pre_report_minutes'])
    earliest_report = support_start - timedelta(minutes=incident_index.max_post_minutes[road])
    latest_report = support_end + timedelta(minutes=pre)
    times = incident_index.times[road]
    first = bisect.bisect_left(times, earliest_report)
    last = bisect.bisect_right(times, latest_report)
    for event in events[first:last]:
        if not any(abs(event.postmile - sensor_pm) <= radius for sensor_pm in affected_sensor_postmiles):
            continue
        active_start = event.report_time - timedelta(minutes=pre)
        active_end = event.report_time + timedelta(minutes=event.post_report_minutes)
        if active_start < support_end and active_end > support_start:
            return False
    return True


def five_number(values):
    ordered = sorted(values)
    if not ordered:
        raise ValueError('Cannot summarize an empty population')
    indices = [0, len(ordered) // 4, len(ordered) // 2, 3 * len(ordered) // 4, len(ordered) - 1]
    return [ordered[index] for index in indices]


def write_csv(path, rows, fieldnames):
    with Path(path).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def audit(data_dir, identity_sidecar, incidents_path, sensors_path, protocol_path, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError('Audit output already exists; preserve it and use a new directory')
    protocol = load_protocol(protocol_path)
    sensors, sensor_count = load_sensors(sensors_path)
    if sensor_count != int(protocol['expected_sensor_count']):
        raise ValueError('Sensor count differs from the frozen protocol')
    radius = float(protocol['spatial_exposure']['postmile_radius'])
    incident_index = load_incidents(
        incidents_path, sensors, protocol['incident_blackout'], radius,
        int(protocol['source_year']))
    manifests = validate_manifests(data_dir, int(protocol['source_version']))
    identity_values = json.loads(Path(identity_sidecar).read_text(encoding='utf-8'))
    identity = {int(row['sample_index']): row for row in identity_values}
    if len(identity) != len(identity_values):
        raise ValueError('Identity sidecar contains duplicate sample indices')

    sample_rows, pair_rows = [], []
    candidate_degree = Counter()
    split_counts = {}
    for split in REQUIRED_SPLITS:
        bounds = protocol['splits'][split]
        split_start = datetime.fromisoformat(bounds['start'])
        split_end = datetime.fromisoformat(bounds['end_exclusive'])
        counts = []
        for row in manifests[split]:
            sample = int(row['sample_index'])
            record = identity.get(sample)
            if record is None or record.get('status') != 'unique_metadata_candidate' or len(record.get('candidates', [])) != 1:
                raise ValueError(f'Sample {sample} lacks one unique identity candidate')
            event = record['candidates'][0]
            if str(event['incident_id']) != str(row['incident_id']):
                raise ValueError(f'Manifest and sidecar incident identity differ for {sample}')
            if datetime.fromisoformat(event['report_time']) != datetime.fromisoformat(row['report_time']):
                raise ValueError(f'Manifest and sidecar report time differ for {sample}')
            road = (int(event['freeway']), parse_direction(event['direction']))
            event_pm = float(event['postmile'])
            affected = [value for value in sensors.get(road, []) if abs(value - event_pm) <= radius]
            if not affected:
                raise ValueError(f'Incident {sample} has no affected sensor under the frozen spatial rule')
            origin = datetime.fromisoformat(row['t0'])
            window = protocol['window']
            positive_start = origin + timedelta(minutes=int(window['support_start_minutes_from_t0']))
            positive_end = origin + timedelta(minutes=int(window['support_end_exclusive_minutes_from_t0']))
            if positive_start < split_start or positive_end > split_end:
                raise ValueError(f'Positive sample {sample} crosses its frozen split boundary')
            calendar = candidate_times(origin, split_start, split_end, protocol)
            clean = [candidate for candidate in calendar
                     if locally_clean(candidate, road, affected, incident_index, protocol)]
            counts.append(len(clean))
            nearest = min((abs((candidate - origin).days) for candidate in clean), default=None)
            sample_rows.append({
                'split': split, 'sample_index': sample, 'incident_id': row['incident_id'],
                'incident_t0': origin.isoformat(), 'freeway': road[0], 'direction': road[1],
                'postmile': event_pm, 'affected_node_count': len(affected),
                'calendar_candidate_count': len(calendar), 'clean_candidate_count': len(clean),
                'nearest_clean_day_distance': '' if nearest is None else nearest,
            })
            for candidate in clean:
                key = (split, candidate.isoformat())
                candidate_degree[key] += 1
                pair_rows.append({
                    'split': split, 'sample_index': sample, 'incident_id': row['incident_id'],
                    'incident_t0': origin.isoformat(), 'candidate_t0': candidate.isoformat(),
                    'absolute_day_distance': abs((candidate - origin).days),
                    'freeway': road[0], 'direction': road[1], 'incident_postmile': event_pm,
                    'affected_node_count': len(affected),
                })
        coverage = sum(value > 0 for value in counts) / len(counts)
        roads = defaultdict(list)
        affected_counts = []
        for sample_row in sample_rows:
            if sample_row['split'] == split:
                roads[f"{sample_row['freeway']}-{sample_row['direction']}"].append(
                    int(sample_row['clean_candidate_count']))
                affected_counts.append(int(sample_row['affected_node_count']))
        split_counts[split] = {
            'samples': len(counts), 'samples_with_candidates': sum(value > 0 for value in counts),
            'samples_without_candidates': sum(value == 0 for value in counts),
            'coverage': coverage, 'candidate_pairs': sum(counts),
            'candidate_count_q0_q25_q50_q75_q100': five_number(counts),
            'mean_candidate_count': sum(counts) / len(counts),
            'affected_node_count_q0_q25_q50_q75_q100': five_number(affected_counts),
            'road_coverage': {
                road: {'samples': len(values),
                       'samples_without_candidates': sum(value == 0 for value in values),
                       'coverage': sum(value > 0 for value in values) / len(values),
                       'median_candidates': five_number(values)[2]}
                for road, values in sorted(roads.items())
            },
        }

    acceptance = {}
    for split in REQUIRED_SPLITS:
        result = split_counts[split]
        wanted_coverage = float(protocol['acceptance']['minimum_coverage'][split])
        wanted_median = int(protocol['acceptance']['minimum_median_candidates'][split])
        acceptance[split] = {
            'coverage_pass': result['coverage'] >= wanted_coverage,
            'median_candidates_pass': result['candidate_count_q0_q25_q50_q75_q100'][2] >= wanted_median,
        }
    passed = all(all(checks.values()) for checks in acceptance.values())
    pair_rows.sort(key=lambda row: (row['split'], row['sample_index'], row['absolute_day_distance'], row['candidate_t0']))
    sample_rows.sort(key=lambda row: (row['split'], row['sample_index']))
    degree_values = list(candidate_degree.values()) or [0]

    output.mkdir(parents=True)
    sample_name, pairs_name = 'sample_audit.csv', 'candidate_pairs.csv'
    write_csv(output / sample_name, sample_rows, SAMPLE_FIELDS)
    write_csv(output / pairs_name, pair_rows, PAIR_FIELDS)
    inputs = [Path(data_dir) / f'{split}_manifest.csv' for split in REQUIRED_SPLITS]
    inputs += [Path(identity_sidecar), Path(incidents_path), Path(sensors_path), Path(protocol_path), Path(__file__)]
    summary = {
        'status': 'MATCHED_NONINCIDENT_CANDIDATE_AUDIT_PASS' if passed else 'MATCHED_NONINCIDENT_CANDIDATE_AUDIT_REJECTED',
        'scope': protocol['scope'], 'main_training_ready': False,
        'protocol_id': protocol['protocol_id'], 'protocol_sha256': sha256(protocol_path),
        'traffic_arrays_read': False, 'forecast_Y_used_for_matching': False,
        'test_split_read': False, 'sensor_count': sensor_count,
        'incident_source_quality': incident_index.quality,
        'splits': split_counts, 'acceptance': acceptance,
        'candidate_t0_usage': {
            'unique_split_candidate_t0': len(candidate_degree),
            'positive_samples_per_candidate_t0_q0_q25_q50_q75_q100': five_number(degree_values),
            'maximum_positive_samples_per_candidate_t0': max(degree_values),
            'note': 'Candidate membership only; no final control assignment or reuse has occurred.'
        },
        'limitations': [
            'Incident logs may omit real incidents.',
            'Source timestamps use publisher nominal calendar without independently certified timezone or DST semantics.',
            'Final duration is used only for offline exclusion and is not a prediction-time feature.',
            'Duration is interpreted as minutes for exclusion but its source unit is not independently certified.',
            'Candidates are not causal counterfactuals and have not yet been ranked by traffic-X similarity.',
        ],
        'inputs': {str(path): sha256(path) for path in inputs},
    }
    summary['outputs'] = {name: sha256(output / name) for name in (sample_name, pairs_name)}
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: value for key, value in summary.items() if key not in ('inputs', 'outputs')},
                     ensure_ascii=False, indent=2), flush=True)
    if not passed:
        raise ValueError('Candidate audit failed the frozen coverage or median acceptance gate')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--identity-sidecar', type=Path, required=True)
    parser.add_argument('--incidents', type=Path, required=True)
    parser.add_argument('--sensors', type=Path, required=True)
    parser.add_argument('--protocol', type=Path,
                        default=Path(__file__).with_name('matched_nonincident_v1.json'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    audit(args.data_dir, args.identity_sidecar, args.incidents, args.sensors, args.protocol, args.output)


if __name__ == '__main__':
    main()
