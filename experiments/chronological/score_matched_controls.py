"""Score matched controls using history X only and assign them without reuse."""

import argparse
import calendar
from collections import Counter, defaultdict
import csv
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from experiments.chronological.audit_matched_controls import (
    five_number, parse_direction, parse_freeway, read_csv, sha256, write_csv,
)


SPLITS = ('train', 'val')
SCORE_FIELDS = [
    'split', 'sample_index', 'incident_id', 'incident_t0', 'candidate_t0',
    'freeway', 'direction', 'incident_postmile', 'affected_node_count',
    'absolute_day_distance', 'pairwise_valid_count', 'pairwise_total_count',
    'pairwise_valid_overlap_fraction', 'missing_pattern_mismatch_fraction',
    'x_distance', 'eligible', 'positive_history_mean', 'control_history_mean',
    'positive_last_step_mean', 'control_last_step_mean',
    'positive_late3_minus_early3', 'control_late3_minus_early3',
]
ASSIGNMENT_FIELDS = SCORE_FIELDS + ['candidate_preference_rank']


def load_protocol(path):
    protocol = json.loads(Path(path).read_text(encoding='utf-8'))
    if protocol.get('scope') != 'offline_x_only_control_assignment':
        raise ValueError('Protocol scope must remain X-only control assignment')
    if protocol.get('main_training_ready') is not False:
        raise ValueError('Control assignment cannot declare main training readiness')
    if int(protocol.get('source_year', 0)) != 2023 or int(protocol.get('source_version', 0)) != 8:
        raise ValueError('v2 is frozen to the 2023 source-v8 development package')
    if int(protocol.get('expected_sensor_count', 0)) != 496:
        raise ValueError('v2 is frozen to the 496-node Contra station axis')
    similarity = protocol['x_similarity']
    if similarity['positive_X_slice'] != [0, 12] or int(similarity['steps']) != 12:
        raise ValueError('v2 requires the frozen 12-step history X')
    if int(similarity['candidate_X_start_minutes_from_t0']) != -65:
        raise ValueError('Candidate X must align with the positive history interval')
    if int(similarity['step_minutes']) != 5:
        raise ValueError('Candidate X must retain five-minute spacing')
    if (similarity.get('node_scope') != 'positive_incident_affected_nodes' or
            similarity.get('valid_value') != 'finite_and_nonnegative' or
            similarity.get('ranking_distance') !=
            'mean_absolute_raw_flow_difference_divided_by_train_global_std'):
        raise ValueError('v2 X scope, validity rule, or distance definition changed')
    prohibited = set(similarity.get('prohibited_inputs', []))
    required_prohibitions = {
        'forecast_Y', 'test_split', 'incident_description', 'incident_type',
    }
    if not required_prohibitions.issubset(prohibited):
        raise ValueError('Protocol must prohibit Y, test, description, and type inputs')
    if protocol['assignment'].get('candidate_capacity') != 1:
        raise ValueError('v2 requires no candidate-time reuse')
    if protocol['assignment'].get('global_cost_optimality_claimed') is not False:
        raise ValueError('Preference matching must not claim global cost optimality')
    overlap = float(similarity['minimum_pairwise_valid_overlap_fraction'])
    if not 0 < overlap <= 1:
        raise ValueError('Pairwise overlap threshold must be in (0, 1]')
    return protocol


def valid_flow(values):
    values = np.asarray(values)
    return np.isfinite(values) & (values >= 0)


def history_features(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != 12:
        raise ValueError('History features require [12, affected_nodes]')
    valid = valid_flow(values)

    def mean(block, mask):
        return float(block[mask].mean()) if mask.any() else float('nan')

    overall = mean(values, valid)
    last = mean(values[-1], valid[-1])
    early = mean(values[:3], valid[:3])
    late = mean(values[-3:], valid[-3:])
    trend = late - early if np.isfinite(early) and np.isfinite(late) else float('nan')
    return {'history_mean': overall, 'last_step_mean': last,
            'late3_minus_early3': trend}


def score_histories(positive, control, train_std, minimum_overlap):
    positive = np.asarray(positive, dtype=np.float64)
    control = np.asarray(control, dtype=np.float64)
    if positive.shape != control.shape or positive.ndim != 2 or positive.shape[0] != 12:
        raise ValueError('Positive and control histories must share [12, affected_nodes] shape')
    if not np.isfinite(train_std) or train_std <= 0:
        raise ValueError('Training scale must be positive and finite')
    positive_valid, control_valid = valid_flow(positive), valid_flow(control)
    overlap = positive_valid & control_valid
    total = overlap.size
    overlap_fraction = float(overlap.sum() / total)
    mismatch = float(np.logical_xor(positive_valid, control_valid).sum() / total)
    distance = (float(np.abs(positive[overlap] - control[overlap]).mean() / train_std)
                if overlap.any() else float('inf'))
    result = {
        'pairwise_valid_count': int(overlap.sum()), 'pairwise_total_count': int(total),
        'pairwise_valid_overlap_fraction': overlap_fraction,
        'missing_pattern_mismatch_fraction': mismatch, 'x_distance': distance,
        'eligible': bool(overlap_fraction >= minimum_overlap and np.isfinite(distance)),
    }
    for prefix, values in (('positive', positive), ('control', control)):
        result.update({f'{prefix}_{key}': value for key, value in history_features(values).items()})
    return result


def edge_preference(edge):
    return (float(edge['x_distance']), float(edge['missing_pattern_mismatch_fraction']),
            int(edge['absolute_day_distance']), edge['candidate_t0'])


def maximum_preference_matching(edges):
    """Maximum-cardinality matching; preferences order augmenting-path exploration."""
    by_left = defaultdict(list)
    for edge in edges:
        if edge['eligible']:
            by_left[(edge['split'], int(edge['sample_index']))].append(edge)
    for values in by_left.values():
        values.sort(key=edge_preference)
    matched_left, matched_right = {}, {}
    previous_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(previous_limit, len(by_left) * 2 + 100))

    def augment(left, seen_left, seen_right):
        if left in seen_left:
            return False
        seen_left.add(left)
        for edge in by_left[left]:
            right = (edge['split'], edge['candidate_t0'])
            if right in seen_right:
                continue
            seen_right.add(right)
            owner = matched_right.get(right)
            if owner is None or augment(owner, seen_left, seen_right):
                matched_right[right] = left
                matched_left[left] = edge
                return True
        return False

    try:
        order = sorted(by_left, key=lambda left: (len(by_left[left]), edge_preference(by_left[left][0]), left))
        for left in order:
            augment(left, set(), set())
    finally:
        sys.setrecursionlimit(previous_limit)
    if len(matched_left) != len(matched_right):
        raise AssertionError('Left/right matching cardinalities differ')
    return matched_left


def source_slot(value):
    return (value.day - 1) * 288 + value.hour * 12 + value.minute // 5


def candidate_history(t0, node_indices, month_flows, protocol):
    similarity = protocol['x_similarity']
    if t0.minute % 5 or t0.second or t0.microsecond or t0.tzinfo is not None:
        raise ValueError('Candidate t0 must be a naive five-minute nominal timestamp')
    start = t0 + timedelta(minutes=int(similarity['candidate_X_start_minutes_from_t0']))
    step_minutes = int(similarity['step_minutes'])
    result = np.empty((int(similarity['steps']), len(node_indices)), dtype=np.float32)
    for step in range(result.shape[0]):
        point = start + timedelta(minutes=step_minutes * step)
        if point.year != int(protocol['source_year']) or point.month not in month_flows:
            raise ValueError('Candidate X requests an unavailable source month')
        result[step] = month_flows[point.month][node_indices, source_slot(point)]
    return result


class VerifiedMonthCache:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.blobs = self.data_dir / 'row_cache/blobs'
        self.axes = np.load(self.data_dir / 'raw_node_indices.npy', allow_pickle=False)
        self.records = []

    def load(self, month):
        manifest_path = self.data_dir / f'source_month_{month:02d}.json'
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        layout = manifest['layout']
        expected_slots = calendar.monthrange(2023, month)[1] * 288
        if (manifest['month'] != month or layout['shape'] != [16972, expected_slots, 3] or
                layout['dtype'] != '<f4' or manifest.get('failures')):
            raise ValueError(f'Invalid completed source-month manifest {month}')
        rows = sorted(manifest['rows'], key=lambda row: row['published_node_index'])
        if [row['published_node_index'] for row in rows] != list(range(len(self.axes))):
            raise ValueError(f'Month {month} does not contain the complete published node axis')
        flow = np.empty((len(rows), expected_slots), dtype=np.float32)
        for index, row in enumerate(rows):
            if int(row['identity']['month']) != month or int(row['identity']['raw_node_index']) != int(self.axes[index]):
                raise ValueError('Source-month row identity differs from the frozen node axis')
            blob = self.blobs / f"{row['sha256']}.bin"
            payload = blob.read_bytes()
            if len(payload) != int(row['bytes']) or hashlib.sha256(payload).hexdigest() != row['sha256']:
                raise ValueError('Cached source row checksum or length differs')
            values = np.frombuffer(payload, dtype='<f4')
            if values.size != expected_slots * 3:
                raise ValueError('Cached source row shape differs')
            flow[index] = values.reshape(expected_slots, 3)[:, 0]
            self.records.append((month, index, row['sha256'], len(payload)))
        return flow, sha256(manifest_path)


def ordered_sensors(path, station_ids):
    by_id = {int(row['station_id']): row for row in read_csv(path)}
    if len(by_id) != len(station_ids) or set(by_id) != set(map(int, station_ids)):
        raise ValueError('Sensor metadata and station axis differ')
    roads, directions, postmiles = [], [], []
    for station in station_ids:
        row = by_id[int(station)]
        roads.append(parse_freeway(row['Fwy']))
        directions.append(parse_direction(row['Direction']))
        postmiles.append(float(row['Abs PM']))
    return np.asarray(roads), np.asarray(directions), np.asarray(postmiles)


def standardized_mean_difference(left, right):
    left, right = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    valid = np.isfinite(left) & np.isfinite(right)
    left, right = left[valid], right[valid]
    if not len(left):
        return None, 0
    pooled = np.sqrt((left.var() + right.var()) / 2)
    difference = float(left.mean() - right.mean())
    if pooled == 0:
        return (0.0 if difference == 0 else None), len(left)
    return float(difference / pooled), len(left)


def verify_inputs(data_dir, candidate_audit, sensors_path, protocol):
    data_dir, candidate_audit = Path(data_dir), Path(candidate_audit)
    expected = protocol['data_inputs']
    actual = {
        'summary_sha256': sha256(data_dir / 'summary.json'),
        'scaler_sha256': sha256(data_dir / 'scaler.json'),
        'station_ids_sha256': sha256(data_dir / 'station_ids.npy'),
        'sensors_sha256': sha256(sensors_path),
    }
    if actual != expected:
        raise ValueError('Data input fingerprints differ from the v2 protocol')
    audit_expected = protocol['candidate_audit']
    audit_summary = json.loads((candidate_audit / 'summary.json').read_text(encoding='utf-8'))
    audit_actual = {
        'summary_sha256': sha256(candidate_audit / 'summary.json'),
        'candidate_pairs_sha256': sha256(candidate_audit / 'candidate_pairs.csv'),
        'sample_audit_sha256': sha256(candidate_audit / 'sample_audit.csv'),
    }
    if any(audit_actual[key] != audit_expected[key] for key in audit_actual):
        raise ValueError('Candidate audit fingerprints differ from the v2 protocol')
    if (audit_summary['status'] != 'MATCHED_NONINCIDENT_CANDIDATE_AUDIT_PASS' or
            audit_summary['protocol_id'] != audit_expected['protocol_id'] or
            audit_summary.get('forecast_Y_used_for_matching') is not False or
            audit_summary.get('test_split_read') is not False):
        raise ValueError('Candidate audit status or information boundary differs')
    return actual, audit_actual


def balance_summary(assignments):
    feature_pairs = {
        'affected_history_mean': ('positive_history_mean', 'control_history_mean'),
        'affected_last_step_mean': ('positive_last_step_mean', 'control_last_step_mean'),
        'affected_late3_minus_early3': (
            'positive_late3_minus_early3', 'control_late3_minus_early3'),
    }
    result = {}
    for name, (positive_key, control_key) in feature_pairs.items():
        positive = np.asarray([float(row[positive_key]) for row in assignments], dtype=np.float64)
        control = np.asarray([float(row[control_key]) for row in assignments], dtype=np.float64)
        smd, count = standardized_mean_difference(positive, control)
        paired = np.isfinite(positive) & np.isfinite(control)
        result[name] = {
            'paired_finite_count': count,
            'positive_mean': float(positive[np.isfinite(positive)].mean())
            if np.isfinite(positive).any() else None,
            'control_mean': float(control[np.isfinite(control)].mean())
            if np.isfinite(control).any() else None,
            'standardized_mean_difference': smd,
            'paired_mean_absolute_difference': float(np.abs(
                positive[paired] - control[paired]).mean()) if paired.any() else None,
        }
    return result


def five_number_or_none(values):
    return five_number(values) if values else [None, None, None, None, None]


def score_and_assign(data_dir, candidate_audit, sensors_path, protocol_path, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError('Assignment output exists; preserve it and use a new directory')
    protocol = load_protocol(protocol_path)
    input_hashes, audit_hashes = verify_inputs(data_dir, candidate_audit, sensors_path, protocol)
    data_dir, candidate_audit = Path(data_dir), Path(candidate_audit)
    station_ids = np.load(data_dir / 'station_ids.npy', allow_pickle=False)
    if station_ids.shape != (int(protocol['expected_sensor_count']),):
        raise ValueError('Station axis differs from the v2 protocol')
    sensor_roads, sensor_directions, sensor_postmiles = ordered_sensors(sensors_path, station_ids)
    scaler = json.loads((data_dir / 'scaler.json').read_text(encoding='utf-8'))
    train_std = float(scaler['std'])
    if scaler['station_ids'] != station_ids.tolist():
        raise ValueError('Scaler and station axes differ')
    manifests, positive_flows, positions = {}, {}, {}
    for split in SPLITS:
        manifests[split] = read_csv(data_dir / f'{split}_manifest.csv')
        positive_flows[split] = np.load(data_dir / f'{split}_flow.npy', mmap_mode='r', allow_pickle=False)
        positions[split] = {int(row['sample_index']): index
                            for index, row in enumerate(manifests[split])}
        if positive_flows[split].shape != (len(manifests[split]), 26, len(station_ids)):
            raise ValueError(f'{split} positive flow shape differs')
    samples = read_csv(candidate_audit / 'sample_audit.csv')
    candidate_rows = read_csv(candidate_audit / 'candidate_pairs.csv')
    if len(samples) != sum(len(manifests[split]) for split in SPLITS):
        raise ValueError('Candidate sample audit population differs from data manifests')
    by_month = defaultdict(list)
    for row in candidate_rows:
        candidate = datetime.fromisoformat(row['candidate_t0'])
        if row['split'] not in SPLITS or candidate.year != int(protocol['source_year']):
            raise ValueError('Candidate pair leaves the frozen development population')
        by_month[candidate.month].append(row)

    minimum_overlap = float(protocol['x_similarity']['minimum_pairwise_valid_overlap_fraction'])
    source = VerifiedMonthCache(data_dir)
    scored = []
    previous_month, previous_flow = None, None
    month_manifest_hashes = {}
    for month in sorted(by_month):
        current_flow, month_hash = source.load(month)
        month_manifest_hashes[str(month)] = month_hash
        month_flows = {month: current_flow}
        needs_previous = any(
            (datetime.fromisoformat(row['candidate_t0']) - timedelta(minutes=65)).month != month
            for row in by_month[month])
        if needs_previous:
            wanted = month - 1
            if previous_month == wanted:
                month_flows[wanted] = previous_flow
            else:
                extra, extra_hash = source.load(wanted)
                month_flows[wanted] = extra
                month_manifest_hashes[str(wanted)] = extra_hash
        for row in by_month[month]:
            split, sample = row['split'], int(row['sample_index'])
            if sample not in positions[split]:
                raise ValueError('Candidate edge references an unknown positive sample')
            road, direction, event_pm = (int(row['freeway']), parse_direction(row['direction']),
                                         float(row['incident_postmile']))
            nodes = np.flatnonzero((sensor_roads == road) & (sensor_directions == direction) &
                                   (np.abs(sensor_postmiles - event_pm) <= 10.0))
            if len(nodes) != int(row['affected_node_count']):
                raise ValueError('Candidate affected-node count differs from sensor metadata')
            positive = positive_flows[split][positions[split][sample], :12][:, nodes]
            control = candidate_history(
                datetime.fromisoformat(row['candidate_t0']), nodes, month_flows, protocol)
            metrics = score_histories(positive, control, train_std, minimum_overlap)
            scored.append({**row, **metrics})
        previous_month, previous_flow = month, current_flow

    scored.sort(key=lambda row: (row['split'], int(row['sample_index']), edge_preference(row)))
    matched = maximum_preference_matching(scored)
    assignments = []
    preference = defaultdict(list)
    for row in scored:
        if row['eligible']:
            preference[(row['split'], int(row['sample_index']))].append(row)
    for values in preference.values():
        values.sort(key=edge_preference)
    for left, row in sorted(matched.items()):
        rank = preference[left].index(row) + 1
        assignments.append({**row, 'candidate_preference_rank': rank})
    used = Counter((row['split'], row['candidate_t0']) for row in assignments)
    max_reuse = max(used.values(), default=0)
    if max_reuse > int(protocol['assignment']['candidate_capacity']):
        raise AssertionError('Candidate capacity was violated')

    sample_keys = {(row['split'], int(row['sample_index'])): row for row in samples}
    assigned_keys = set(matched)
    eligible_keys = set(preference)
    unmatched = []
    for key, row in sorted(sample_keys.items()):
        if key in assigned_keys:
            continue
        if int(row['clean_candidate_count']) == 0:
            reason = 'no_metadata_candidate'
        elif key not in eligible_keys:
            reason = 'no_candidate_passed_x_overlap'
        else:
            reason = 'candidate_capacity_conflict'
        unmatched.append({**row, 'unmatched_reason': reason})

    split_summary = {}
    maximum_smd = None
    all_balance_features_observed = True
    for split in SPLITS:
        population = [row for row in samples if row['split'] == split]
        selected = [row for row in assignments if row['split'] == split]
        split_unmatched = [row for row in unmatched if row['split'] == split]
        distances = [float(row['x_distance']) for row in selected]
        overlaps = [float(row['pairwise_valid_overlap_fraction']) for row in selected]
        mismatches = [float(row['missing_pattern_mismatch_fraction']) for row in selected]
        balance = balance_summary(selected)
        finite_smd = [abs(item['standardized_mean_difference']) for item in balance.values()
                      if item['standardized_mean_difference'] is not None]
        all_balance_features_observed &= len(finite_smd) == len(balance)
        if finite_smd:
            split_maximum_smd = max(finite_smd)
            maximum_smd = (split_maximum_smd if maximum_smd is None else
                           max(maximum_smd, split_maximum_smd))
        roads = defaultdict(lambda: [0, 0])
        for row in population:
            roads[f"{row['freeway']}-{row['direction']}"][0] += 1
        for row in selected:
            roads[f"{row['freeway']}-{row['direction']}"][1] += 1
        split_summary[split] = {
            'positive_samples': len(population), 'assigned_samples': len(selected),
            'assignment_coverage': len(selected) / len(population),
            'unmatched_reasons': dict(Counter(row['unmatched_reason'] for row in split_unmatched)),
            'x_distance_q0_q25_q50_q75_q100': five_number_or_none(distances),
            'valid_overlap_q0_q25_q50_q75_q100': five_number_or_none(overlaps),
            'missing_mismatch_q0_q25_q50_q75_q100': five_number_or_none(mismatches),
            'preference_rank_q0_q25_q50_q75_q100': five_number_or_none(
                [int(row['candidate_preference_rank']) for row in selected]),
            'balance': balance,
            'road_assignment_coverage': {
                road: {'positive_samples': values[0], 'assigned_samples': values[1],
                       'coverage': values[1] / values[0]}
                for road, values in sorted(roads.items())
            },
        }

    acceptance = {
        'coverage': {split: split_summary[split]['assignment_coverage'] >=
                     float(protocol['acceptance']['minimum_assignment_coverage'][split])
                     for split in SPLITS},
        'median_overlap': {
            split: (split_summary[split]['valid_overlap_q0_q25_q50_q75_q100'][2] is not None and
                    split_summary[split]['valid_overlap_q0_q25_q50_q75_q100'][2] >=
                    float(protocol['acceptance'][
                        'minimum_median_pairwise_valid_overlap_fraction']))
            for split in SPLITS
        },
        'feature_balance': (
            all_balance_features_observed and maximum_smd is not None and
            maximum_smd <= float(protocol['acceptance']['maximum_absolute_feature_smd'])),
        'candidate_reuse': max_reuse <= int(protocol['acceptance']['maximum_candidate_reuse']),
    }
    passed = (all(acceptance['coverage'].values()) and all(acceptance['median_overlap'].values()) and
              acceptance['feature_balance'] and acceptance['candidate_reuse'])
    output.mkdir(parents=True)
    score_name, assignment_name, unmatched_name = 'edge_scores.csv', 'assignments.csv', 'unmatched.csv'
    write_csv(output / score_name, scored, SCORE_FIELDS)
    write_csv(output / assignment_name, assignments, ASSIGNMENT_FIELDS)
    unmatched_fields = list(samples[0]) + ['unmatched_reason']
    write_csv(output / unmatched_name, unmatched, unmatched_fields)
    cache_digest = hashlib.sha256()
    for record in source.records:
        cache_digest.update(json.dumps(record, separators=(',', ':')).encode())
    summary = {
        'status': 'MATCHED_NONINCIDENT_X_ASSIGNMENT_PASS' if passed else
                  'MATCHED_NONINCIDENT_X_ASSIGNMENT_REJECTED',
        'scope': protocol['scope'], 'main_training_ready': False,
        'protocol_id': protocol['protocol_id'], 'protocol_sha256': sha256(protocol_path),
        'traffic_X_read': True, 'forecast_Y_used_for_scoring_or_assignment': False,
        'test_split_read': False, 'candidate_capacity': 1, 'maximum_candidate_reuse': max_reuse,
        'global_cost_optimality_claimed': False, 'splits': split_summary,
        'maximum_absolute_feature_smd': maximum_smd, 'acceptance': acceptance,
        'source_cache': {
            'verified_row_reads': len(source.records),
            'verified_row_bytes': sum(record[3] for record in source.records),
            'ordered_record_digest': cache_digest.hexdigest(),
            'month_manifest_sha256': month_manifest_hashes,
        },
        'limitations': [
            'Matching uses recorded incident exclusions and cannot rule out unreported incidents.',
            'X matching improves observed-history balance but does not identify a causal counterfactual.',
            'Maximum-cardinality preference matching does not claim globally minimum total X distance.',
            'Unmatched positives remain a separately reported population and are not silently discarded.',
        ],
        'inputs': {**input_hashes, **{f'audit_{key}': value for key, value in audit_hashes.items()},
                   'protocol_sha256': sha256(protocol_path), 'code_sha256': sha256(__file__)},
    }
    summary['outputs'] = {name: sha256(output / name)
                          for name in (score_name, assignment_name, unmatched_name)}
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2,
                                                    allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({key: value for key, value in summary.items()
                      if key not in ('inputs', 'outputs')}, ensure_ascii=False, indent=2), flush=True)
    if not passed:
        raise ValueError('X-only assignment failed the frozen acceptance gates')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--candidate-audit', type=Path, required=True)
    parser.add_argument('--sensors', type=Path, required=True)
    parser.add_argument('--protocol', type=Path,
                        default=Path(__file__).with_name('matched_nonincident_x_v2.json'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    score_and_assign(args.data_dir, args.candidate_audit, args.sensors, args.protocol, args.output)


if __name__ == '__main__':
    main()
