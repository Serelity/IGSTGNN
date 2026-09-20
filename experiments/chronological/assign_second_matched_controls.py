"""Assign a second X-only control without changing or reusing frozen primary controls."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import json
from pathlib import Path

from experiments.chronological.audit_matched_controls import (
    five_number, read_csv, sha256, write_csv,
)
from experiments.chronological.score_matched_controls import (
    SCORE_FIELDS, SPLITS, balance_summary, edge_preference,
    maximum_preference_matching,
)


SECONDARY_FIELDS = SCORE_FIELDS + ['original_candidate_preference_rank']
UNMATCHED_FIELDS = [
    'split', 'sample_index', 'incident_id', 'incident_t0',
    'primary_candidate_t0', 'remaining_eligible_candidate_count',
    'unmatched_reason',
]


def load_protocol(path):
    protocol = json.loads(Path(path).read_text(encoding='utf-8'))
    if protocol.get('scope') != 'fixed_primary_x_only_secondary_assignment':
        raise ValueError('Protocol scope must remain fixed-primary X-only assignment')
    if protocol.get('main_training_ready') is not False:
        raise ValueError('Secondary assignment cannot authorize main training')
    if (int(protocol.get('source_year', 0)) != 2023 or
            int(protocol.get('source_version', 0)) != 8 or
            int(protocol.get('expected_sensor_count', 0)) != 496):
        raise ValueError('v5a is frozen to the 2023 source-v8 Contra496 package')
    boundary = protocol['selection_boundary']
    required = [
        'primary_assignment_immutable',
        'secondary_selected_only_from_frozen_v2_edge_scores',
        'all_primary_candidate_centers_excluded', 'forecast_Y_prohibited',
        'test_split_prohibited', 'incident_description_prohibited',
        'incident_type_prohibited', 'outcome_audit_results_prohibited',
    ]
    if not all(boundary.get(key) is True for key in required):
        raise ValueError('v5a information boundary changed')
    assignment = protocol['secondary_assignment']
    if (assignment.get('eligible_v2_edges_only') is not True or
            int(assignment.get('candidate_capacity_across_primary_and_secondary', 0)) != 1 or
            assignment.get('algorithm') !=
            'maximum_cardinality_preference_augmenting_path' or
            assignment.get('global_cost_optimality_claimed') is not False or
            assignment.get('preference_order') != [
                'x_distance', 'missing_pattern_mismatch_fraction',
                'absolute_day_distance', 'candidate_t0']):
        raise ValueError('v5a assignment algorithm or capacity changed')
    acceptance = protocol['acceptance']
    if (not 0 < float(acceptance['minimum_secondary_coverage_of_primary']['train']) <= 1 or
            not 0 < float(acceptance['minimum_secondary_coverage_of_primary']['val']) <= 1 or
            not 0 < float(acceptance['minimum_road_direction_coverage']) <= 1 or
            not 0 < float(acceptance['maximum_absolute_feature_smd']) <= 1 or
            int(acceptance['maximum_candidate_center_reuse_across_both_controls']) != 1):
        raise ValueError('v5a acceptance thresholds are invalid')
    return protocol


def verify_inputs(primary_dir, protocol):
    primary_dir = Path(primary_dir)
    actual = {
        'summary_sha256': sha256(primary_dir / 'summary.json'),
        'assignments_sha256': sha256(primary_dir / 'assignments.csv'),
        'edge_scores_sha256': sha256(primary_dir / 'edge_scores.csv'),
        'unmatched_sha256': sha256(primary_dir / 'unmatched.csv'),
    }
    expected = protocol['primary_assignment_input']
    if any(actual[key] != expected[key] for key in actual):
        raise ValueError('Frozen v2 assignment fingerprints differ from v5a protocol')
    summary = json.loads((primary_dir / 'summary.json').read_text(encoding='utf-8'))
    if (summary.get('status') != 'MATCHED_NONINCIDENT_X_ASSIGNMENT_PASS' or
            summary.get('protocol_id') != expected['protocol_id'] or
            summary.get('forecast_Y_used_for_scoring_or_assignment') is not False or
            summary.get('test_split_read') is not False or
            summary.get('maximum_candidate_reuse') != 1):
        raise ValueError('Frozen v2 status or information boundary differs')
    if summary.get('outputs') != {
            'edge_scores.csv': actual['edge_scores_sha256'],
            'assignments.csv': actual['assignments_sha256'],
            'unmatched.csv': actual['unmatched_sha256']}:
        raise ValueError('Frozen v2 summary output fingerprints are inconsistent')
    return actual


def eligible_value(value):
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text not in ('true', 'false'):
        raise ValueError(f'Invalid eligible value {value!r}')
    return text == 'true'


def assign_secondary(primary_rows, edge_rows):
    primary = {}
    used_primary = set()
    for row in primary_rows:
        key = (row['split'], int(row['sample_index']))
        right = (row['split'], row['candidate_t0'])
        if row['split'] not in SPLITS or key in primary or right in used_primary:
            raise ValueError('Primary assignments contain a duplicate or invalid identity')
        primary[key] = row
        used_primary.add(right)
    preferences, available = defaultdict(list), []
    edge_identities = set()
    for row in edge_rows:
        split = row['split']
        key = (split, int(row['sample_index']))
        right = (split, row['candidate_t0'])
        identity = (key, right)
        if split not in SPLITS or identity in edge_identities:
            raise ValueError('Frozen edge scores contain a duplicate or invalid edge')
        edge_identities.add(identity)
        if not eligible_value(row['eligible']):
            continue
        preferences[key].append(row)
        if key in primary and right not in used_primary:
            available.append({**row, 'eligible': True})
    for rows in preferences.values():
        rows.sort(key=edge_preference)
    matched = maximum_preference_matching(available)
    secondary = []
    for key, row in sorted(matched.items()):
        rank = next(
            index + 1 for index, candidate in enumerate(preferences[key])
            if candidate['candidate_t0'] == row['candidate_t0'])
        secondary.append({**row, 'original_candidate_preference_rank': rank})
    secondary_keys = {(row['split'], int(row['sample_index'])) for row in secondary}
    available_counts = Counter(
        (row['split'], int(row['sample_index'])) for row in available)
    unmatched = []
    for key, row in sorted(primary.items()):
        if key in secondary_keys:
            continue
        count = available_counts[key]
        reason = ('no_unused_eligible_candidate' if count == 0 else
                  'secondary_candidate_capacity_conflict')
        unmatched.append({
            'split': row['split'], 'sample_index': int(row['sample_index']),
            'incident_id': row['incident_id'], 'incident_t0': row['incident_t0'],
            'primary_candidate_t0': row['candidate_t0'],
            'remaining_eligible_candidate_count': count,
            'unmatched_reason': reason,
        })
    return secondary, unmatched, available_counts


def summarize(primary_rows, secondary_rows, unmatched_rows, available_counts, protocol):
    primary_by_key = {
        (row['split'], int(row['sample_index'])): row for row in primary_rows
    }
    all_centers = Counter(
        (row['split'], row['candidate_t0'])
        for row in [*primary_rows, *secondary_rows])
    maximum_reuse = max(all_centers.values(), default=0)
    split_results = {}
    maximum_smd = None
    minimum_road_coverage = None
    for split in SPLITS:
        primary = [row for row in primary_rows if row['split'] == split]
        secondary = [row for row in secondary_rows if row['split'] == split]
        unmatched = [row for row in unmatched_rows if row['split'] == split]
        balance = balance_summary(secondary)
        smds = [abs(item['standardized_mean_difference']) for item in balance.values()
                if item['standardized_mean_difference'] is not None]
        if len(smds) != len(balance):
            raise ValueError(f'{split} secondary balance has an unavailable feature')
        split_maximum_smd = max(smds)
        maximum_smd = (split_maximum_smd if maximum_smd is None else
                       max(maximum_smd, split_maximum_smd))
        roads = defaultdict(lambda: [0, 0])
        for row in primary:
            roads[f"{row['freeway']}-{row['direction']}"][0] += 1
        for row in secondary:
            roads[f"{row['freeway']}-{row['direction']}"][1] += 1
        road_summary = {
            key: {
                'primary_pairs': values[0], 'secondary_pairs': values[1],
                'coverage': values[1] / values[0],
            }
            for key, values in sorted(roads.items())
        }
        split_minimum_road = min(item['coverage'] for item in road_summary.values())
        minimum_road_coverage = (split_minimum_road if minimum_road_coverage is None else
                                 min(minimum_road_coverage, split_minimum_road))
        counts = [available_counts[(split, int(row['sample_index']))] for row in primary]
        ranks = [int(row['original_candidate_preference_rank']) for row in secondary]
        distances = [float(row['x_distance']) for row in secondary]
        first_second_days = [abs((
            datetime.fromisoformat(row['candidate_t0']) -
            datetime.fromisoformat(
                primary_by_key[(split, int(row['sample_index']))]['candidate_t0'])
        ).days) for row in secondary]
        split_results[split] = {
            'primary_pairs': len(primary), 'secondary_pairs': len(secondary),
            'coverage_of_primary': len(secondary) / len(primary),
            'unmatched_primary_pairs': len(unmatched),
            'unmatched_reasons': dict(Counter(
                row['unmatched_reason'] for row in unmatched)),
            'remaining_candidate_count_q0_q25_q50_q75_q100': five_number(counts),
            'original_preference_rank_q0_q25_q50_q75_q100': five_number(ranks),
            'x_distance_q0_q25_q50_q75_q100': five_number(distances),
            'first_second_absolute_day_distance_q0_q25_q50_q75_q100':
                five_number(first_second_days),
            'balance': balance, 'road_direction_coverage': road_summary,
        }
    acceptance = {
        'coverage': {
            split: split_results[split]['coverage_of_primary'] >= float(
                protocol['acceptance']['minimum_secondary_coverage_of_primary'][split])
            for split in SPLITS
        },
        'road_direction_coverage': minimum_road_coverage >= float(
            protocol['acceptance']['minimum_road_direction_coverage']),
        'feature_balance': maximum_smd <= float(
            protocol['acceptance']['maximum_absolute_feature_smd']),
        'candidate_center_reuse': maximum_reuse <= int(
            protocol['acceptance']['maximum_candidate_center_reuse_across_both_controls']),
    }
    return {
        'splits': split_results, 'maximum_absolute_feature_smd': maximum_smd,
        'minimum_road_direction_coverage': minimum_road_coverage,
        'maximum_candidate_center_reuse_across_both_controls': maximum_reuse,
        'acceptance': acceptance,
    }


def assign(primary_dir, protocol_path, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError('Secondary-assignment output exists; use a new directory')
    protocol = load_protocol(protocol_path)
    input_hashes = verify_inputs(primary_dir, protocol)
    primary_dir = Path(primary_dir)
    primary_rows = read_csv(primary_dir / 'assignments.csv')
    edge_rows = read_csv(primary_dir / 'edge_scores.csv')
    expected_counts = protocol['primary_assignment_input']['expected_primary_pairs']
    if any(sum(row['split'] == split for row in primary_rows) != int(expected_counts[split])
           for split in SPLITS):
        raise ValueError('Primary assignment counts differ from v5a protocol')
    secondary, unmatched, available_counts = assign_secondary(primary_rows, edge_rows)
    diagnostics = summarize(
        primary_rows, secondary, unmatched, available_counts, protocol)
    acceptance = diagnostics['acceptance']
    passed = (all(acceptance['coverage'].values()) and
              acceptance['road_direction_coverage'] and
              acceptance['feature_balance'] and acceptance['candidate_center_reuse'])
    output.mkdir(parents=True)
    assignment_name = 'secondary_assignments.csv'
    unmatched_name = 'unmatched_primary_pairs.csv'
    write_csv(output / assignment_name, secondary, SECONDARY_FIELDS)
    write_csv(output / unmatched_name, unmatched, UNMATCHED_FIELDS)
    summary = {
        'status': ('SECOND_MATCHED_CONTROL_ASSIGNMENT_PASS' if passed else
                   'SECOND_MATCHED_CONTROL_ASSIGNMENT_REJECTED'),
        'scope': protocol['scope'], 'protocol_id': protocol['protocol_id'],
        'protocol_sha256': sha256(protocol_path), 'main_training_ready': False,
        'primary_assignment_immutable': True,
        'primary_assignment_rows_changed': 0,
        'traffic_arrays_read': False, 'forecast_Y_read': False,
        'test_split_read': False, 'incident_text_read': False,
        'outcome_audit_results_read': False,
        'global_cost_optimality_claimed': False,
        **diagnostics,
        'terminology': protocol['terminology'],
        'limitations': [
            'The secondary cohort is a subset of the frozen primary matched population.',
            ('Fixing all primary controls preserves the prior audit but may reduce secondary '
             'coverage relative to a joint two-control rematch.'),
            ('Low road-direction coverage, especially validation freeway 4, limits '
             'population-wide placebo claims.'),
            'Both controls remain matched observational comparisons, not counterfactuals.',
        ],
        'inputs': {
            **input_hashes, 'protocol_sha256': sha256(protocol_path),
            'code_sha256': sha256(__file__),
        },
    }
    summary['outputs'] = {
        assignment_name: sha256(output / assignment_name),
        unmatched_name: sha256(output / unmatched_name),
    }
    (output / 'summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
        encoding='utf-8')
    print(json.dumps({key: value for key, value in summary.items()
                      if key not in ('inputs', 'outputs')},
                     ensure_ascii=False, indent=2, allow_nan=False), flush=True)
    if not passed:
        raise ValueError('Secondary assignment failed the frozen acceptance gates')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--primary-assignment-dir', type=Path, required=True)
    parser.add_argument('--protocol', type=Path,
                        default=Path(__file__).with_name(
                            'second_matched_control_v5a.json'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assign(args.primary_assignment_dir, args.protocol, args.output)


if __name__ == '__main__':
    main()
