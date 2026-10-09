"""H1 pre-development audit: package identity, X-only profiles and road inventory.

No inferred units, fitted capacities, certified road directions or training.
Output templates remain unresolved until physical evidence is supplied.
"""

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.chronological.smoke import sha256, verify_package
from src.utils.chronological import ChronologicalDataset, read_rows
from src.utils.traffic_physics_contract import check_manifest_timing, contract_template, validate_contract


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def history_profiles(flow, station_ids):
    """Descriptive window-weighted statistics; not parameter/scaler fitting."""
    rows = []
    for i, station in enumerate(station_ids):
        values = np.asarray(flow[:, :12, i]).reshape(-1)
        observed = values[np.isfinite(values) & (values >= 0)]
        q = np.quantile(observed, [.05, .5, .95, .99]) if len(observed) else [None] * 4
        rows.append({'station_id': int(station), 'history_cells': len(values),
                     'valid_cells': len(observed), 'zero_cells': int((observed == 0).sum()),
                     **{name: float(x) if x is not None else None for name, x in zip(('q05', 'q50', 'q95', 'q99'), q)}})
    return rows


def write_csv(path, rows, fields):
    with Path(path).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sensor_inventory(sensors_path, station_ids):
    if sensors_path is None or not sensors_path.is_file():
        return [], [], ['sensor_metadata_missing']
    rows = read_rows(sensors_path)
    by_id = {int(row['station_id']): row for row in rows}
    if len(by_id) != len(rows) or any(int(x) not in by_id for x in station_ids):
        raise ValueError('Sensor table has duplicate IDs or misses package stations')
    inventory, groups = [], {}
    for station in station_ids:
        row = by_id[int(station)]
        pm = float(row['Abs PM'])
        if not np.isfinite(pm):
            raise ValueError('Invalid sensor postmile')
        record = {'station_id': int(station), 'road': row['Fwy'], 'direction': row['Direction'],
                  'postmile': pm, 'source_length_unconfirmed_unit': row.get('Length', ''),
                  'source_type': row.get('Type', ''), 'design_speed_not_measurement': row.get('Design Speed Limit', '')}
        inventory.append(record)
        groups.setdefault((row['Fwy'], row['Direction']), []).append(record)
    pairs = []
    for (road, direction), group in sorted(groups.items()):
        ordered = sorted(group, key=lambda x: (x['postmile'], x['station_id']))
        for left, right in zip(ordered, ordered[1:]):
            pairs.append({'road': road, 'direction': direction,
                          'low_postmile_station_id': left['station_id'], 'high_postmile_station_id': right['station_id'],
                          'postmile_difference': right['postmile'] - left['postmile'],
                          'physical_direction_certified': False, 'direct_connection_certified': False})
    return inventory, pairs, []


def audit(data_dir, output_dir, sensors=None, contract_path=None):
    if output_dir.exists():
        raise FileExistsError('Use a new audit output directory')
    fingerprints = verify_package(data_dir)
    train, val = ChronologicalDataset(data_dir, 'train'), ChronologicalDataset(data_dir, 'val')
    if not np.array_equal(train.station_ids, val.station_ids):
        raise ValueError('Train and validation node order differs')
    timing = check_manifest_timing(train.rows + val.rows)
    inventory, pairs, missing = sensor_inventory(sensors, train.station_ids)
    context_meta = json.loads((data_dir / 'context_manifest.json').read_text(encoding='utf-8'))
    if sensors is not None and sensors.is_file():
        expected = [v for k, v in context_meta.get('sources', {}).items() if Path(k).name == 'sensors.csv']
        if not expected or any(sha256(sensors) != x for x in expected):
            raise ValueError('Sensor metadata is not the source used by this context package')
    profiles = history_profiles(train.flow, train.station_ids)
    template = contract_template(train.station_ids, fingerprints)
    readiness = 'PHYSICAL_CONTRACT_REQUIRED'
    missing += ['flow_units_and_lane_aggregation', 'source_interval_label',
                'bottleneck_arrival_departure_observation_mapping', 'capacity_and_queue_scale_calibration']
    checked = None
    if contract_path is not None:
        checked = validate_contract(json.loads(contract_path.read_text(encoding='utf-8')),
                                    train.station_ids, fingerprints)
        if not inventory:
            raise ValueError('A declared contract also requires the original sensor inventory')
        source = {x['station_id']: x for x in inventory}
        for item in checked['bottlenecks']:
            a, d = source[item['arrival_station_id']], source[item['departure_station_id']]
            if (a['road'], a['direction']) != (d['road'], d['direction']):
                raise ValueError('Bottleneck boundary stations belong to different roads/directions')
        timing = check_manifest_timing(train.rows + val.rows, checked['interval_label'])
        readiness, missing = 'DECLARED_CONTRACT_VALIDATED_FOR_ENGINEERING_CHECK', []
    output_dir.mkdir(parents=True)
    write_json(output_dir / 'contract_template.json', template)
    if checked is not None:
        write_json(output_dir / 'declared_contract.json', checked)
    write_csv(output_dir / 'train_history_profiles.csv', profiles, list(profiles[0]))
    if inventory:
        write_csv(output_dir / 'sensor_inventory.csv', inventory, list(inventory[0]))
    write_csv(output_dir / 'candidate_postmile_neighbors.csv', pairs,
              ['road', 'direction', 'low_postmile_station_id', 'high_postmile_station_id',
               'postmile_difference', 'physical_direction_certified', 'direct_connection_certified'])
    report = {'status': 'PHYSICS_INPUT_AUDIT_COMPLETE', 'readiness': readiness,
              'training_started': False, 'main_training_ready': False, 'ctm_real_data_enabled': False,
              'train_samples': len(train), 'val_samples': len(val), 'stations': len(train.station_ids),
              'package_sha256': fingerprints, 'timing': timing,
              'training_input_channels': ['flow_standardized', 'time_of_day', 'day_of_week'],
              'profile_scope': 'train_X_only_window_weighted_not_capacity_calibration',
              'validation_and_gap_values_used_for_statistics': False,
              'test_data_loaded': False, 'candidate_neighbor_pairs': len(pairs),
              'missing_physical_evidence': missing,
              'sources': {'audit': sha256(Path(__file__)),
                          'sensors': sha256(sensors) if sensors is not None and sensors.is_file() else None,
                          'contract': sha256(contract_path) if contract_path else None}}
    write_json(output_dir / 'summary.json', report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--sensors', type=Path)
    parser.add_argument('--contract', type=Path)
    args = parser.parse_args(argv)
    result = audit(args.data_dir, args.output_dir, args.sensors, args.contract)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
