"""Build report_location_v1 fields without importing final event descriptions/types."""

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.chronological import read_rows


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def spatial_features(event, sensors):
    pm = np.asarray([float(s['Abs PM']) for s in sensors])
    freeway = np.asarray([int(re.search(r'\d+', s['Fwy']).group()) for s in sensors])
    directions = np.asarray([s['Direction'].upper() for s in sensors])
    event_pm = float(event['postmile'])
    if not np.isfinite(pm).all() or not np.isfinite(event_pm):
        raise ValueError('Non-finite postmile')
    delta = pm - event_pm
    support = ((freeway == int(event['freeway'])) & (directions == event['direction'])
               & (np.abs(delta) <= 10))
    floor = np.exp(-50)
    score = np.where(support, (np.exp(-delta ** 2 / 2) - floor) / (1 - floor), 0)
    return np.stack([np.zeros_like(pm), score, support & (delta < 0)], axis=-1).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--identity-sidecar', type=Path, required=True)
    parser.add_argument('--sensors', type=Path, required=True)
    parser.add_argument('--adjacency', type=Path, required=True)
    args = parser.parse_args()
    output = args.data_dir
    destinations = [output / f'{s}_context.npz' for s in ('train', 'val')]
    destinations += [output / 'context_manifest.json', output / 'adjacency.npy']
    if any(p.exists() for p in destinations):
        raise FileExistsError('Research context already exists; refusing to overwrite')
    stations = np.load(output / 'station_ids.npy', allow_pickle=False)
    sensor_rows = read_rows(args.sensors)
    sensor_by_id = {int(row['station_id']): row for row in sensor_rows}
    sensors = [sensor_by_id[int(s)] for s in stations]
    original_ids = [int(row['station_id']) for row in sensor_rows]
    indices = [original_ids.index(int(s)) for s in stations]
    adj = np.load(args.adjacency, allow_pickle=False)
    if adj.shape != (len(original_ids), len(original_ids)) or not np.isfinite(adj).all() or (adj < 0).any():
        raise ValueError('Invalid adjacency or sensor axis')
    identity = {int(r['sample_index']): r for r in json.loads(args.identity_sidecar.read_text())}
    summaries = {}
    for split in ('train', 'val'):
        rows = read_rows(output / f'{split}_manifest.csv')
        contexts, age, tod, dow = [], [], [], []
        for row in rows:
            record = identity[int(row['sample_index'])]
            if record['status'] != 'unique_metadata_candidate' or len(record['candidates']) != 1:
                raise ValueError('Only unique metadata candidates can produce an event context')
            event = record['candidates'][0]
            report, t0 = datetime.fromisoformat(row['report_time']), datetime.fromisoformat(row['t0'])
            if event['incident_id'] != row['incident_id'] or datetime.fromisoformat(event['report_time']) != report:
                raise ValueError('Manifest and candidate event disagree')
            elapsed = (t0 - report).total_seconds() / 60
            if not 0 < elapsed <= 5:
                raise ValueError('Invalid next-grid report age')
            contexts.append(spatial_features(event, sensors))
            age.append(elapsed)
            tod.append(t0.hour * 12 + t0.minute // 5)
            dow.append((t0.weekday() + 1) % 7)
        contexts = np.asarray(contexts, dtype=np.float32)
        np.savez_compressed(output / f'{split}_context.npz', distances=contexts,
                            report_age_minutes=np.asarray(age, dtype=np.float32),
                            forecast_tod=np.asarray(tod, dtype=np.int64),
                            forecast_dow=np.asarray(dow, dtype=np.int64),
                            sample_indices=np.asarray([int(r['sample_index']) for r in rows]),
                            station_ids=stations)
        connected = np.any(contexts != 0, axis=-1).sum(axis=-1)
        summaries[split] = {'samples': len(rows), 'min_connected_nodes': int(connected.min()),
                            'mean_connected_nodes': float(connected.mean()),
                            'zero_connected_samples': int(np.count_nonzero(connected == 0))}
    np.save(output / 'adjacency.npy', adj[np.ix_(indices, indices)].astype(np.float32))
    report = {'schema': 'report_location_v1', 'scope': 'conditional_development',
              'assumptions': ['Recorded freeway/direction/postmile known at first report',
                              'Publisher-declared nominal calendar; no certified timezone/DST'],
              'source_snapshot_is_initial_report_certified': False,
              'features': ['report_age_minutes', 'forecast_tod', 'forecast_dow', 'distances'],
              'distance_columns': ['constant_zero', 'same_road_direction_PM_similarity',
                                   'same_support_event_PM_greater_than_sensor_PM'],
              'main_training_ready': False, 'splits': summaries,
              'sources': {str(p): sha256(p) for p in
                          [args.identity_sidecar, args.sensors, args.adjacency, Path(__file__)]},
              'outputs': {p.name: sha256(p) for p in destinations if p.name != 'context_manifest.json'}}
    (output / 'context_manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report['splits'], indent=2))


if __name__ == '__main__':
    main()
