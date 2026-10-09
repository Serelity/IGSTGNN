"""Read-only follow-up for the 55 count-matched candidate paper accidents.

Checks published clock boundaries, not the unseen author model matrix. Thresholds
are diagnostic flags; this script neither repairs values nor selects training data.
"""
import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import numpy as np

import audit_chattanooga as audit


def token(value):
    return hashlib.sha256(value.encode()).hexdigest()[:20]


def checks(a):
    return {
        'volume_gt_1000': a[..., 1] > 1000,
        'volume_gt_1million': a[..., 1] > 1e6,
        'occupancy_gt_100': a[..., 2] > 100,
        'volume_nonfinite': ~np.isfinite(a[..., 1]),
    }


def run(root):
    for name, expected in audit.ARCHIVES.items():
        if audit.digest((root / name).read_bytes()) != expected:
            raise ValueError('Unexpected archive checksum')
    with ZipFile(root / 'metaData.zip') as z:
        names = {PurePosixPath(i.filename).name: i.filename for i in audit.preflight(z)}
        header, body = audit.table(z.read(names['SensorTopology.csv']))
        topo = {r['Name']: r for r in (dict(zip(header, row)) for row in body)}
        aliases = {audit.sensor_alias(r): r['Name'] for r in topo.values()}
        if len(aliases) != len(topo) or len(topo) != len(body):
            raise ValueError('Nonunique sensor keys')
        geometry = json.loads(z.read(names['SensorZones.geojson']))
        lanes = defaultdict(set)
        for feature in geometry['features']:
            p = feature['properties']
            lanes[p['RDS_Sensor']].add(int(p['NBR_LANES']))
    records = []
    with ZipFile(root / 'annotatedData.zip') as z:
        for info in sorted(audit.preflight(z), key=lambda i: i.filename):
            name = info.filename
            if '/bestData/accident/' not in name or not name.endswith('.csv'):
                continue
            header, rows = audit.table(z.read(name))
            if header != audit.EXPECTED or not rows:
                raise ValueError('Unexpected CSV')
            if rows[0][1] not in ('00I75N', '00I75S', '00I24E', '00I24W') or rows[0][3] not in (
                'Fatal', 'Suspected Minor Injury', 'Suspected Serious Injury'
            ):
                continue
            m = re.fullmatch(r'(\d{4}-\d{2}-\d{2})-(\d{2})(\d{2})-(.+)\.csv', PurePosixPath(name).name)
            if not m:
                raise ValueError('Unexpected filename')
            date, hh, mm, center = m.groups()
            if center not in aliases:
                raise ValueError('Unmapped center')
            clock = int(hh) * 3600 + int(mm) * 60
            relative = audit.relative_times([r[7] for r in rows], clock)
            if not np.array_equal(relative, np.arange(-900, 931, 30)):
                raise ValueError('Unexpected time grid')
            raw = np.array([r[10:] for r in rows], dtype=str).reshape(-1, 11, 3)
            values = np.array([float(v) if v.strip() else np.nan for v in raw.ravel()]).reshape(raw.shape)
            records.append(dict(road=rows[0][1][:-1], window_token=token(name), date=date,
                                time=relative + datetime.strptime(date, '%Y-%m-%d').toordinal() * 86400 + clock,
                                relative=relative, values=values, raw=raw,
                                sensors=audit.neighbourhood(aliases[center], topo)))
    counts = Counter(r['road'] for r in records)
    if counts != {'00I75': 24, '00I24': 31}:
        raise ValueError('Candidate selection no longer matches expected counts')

    sensitivity = []
    for road in ('00I75', '00I24', 'combined'):
        group = [r for r in records if road == 'combined' or r['road'] == road]
        for radius in (0, 1, 5):
            for end in range(0, 421, 30):
                totals = Counter()
                for r in group:
                    a = r['values'][(r['relative'] >= -240) & (r['relative'] <= end), 5-radius:6+radius]
                    for flag, mask in checks(a).items():
                        totals[flag + '_windows'] += int(mask.any())
                        totals[flag + '_cells'] += int(mask.sum())
                    totals['finite_volume_cells'] += int(np.isfinite(a[..., 1]).sum())
                    totals['finite_occupancy_cells'] += int(np.isfinite(a[..., 2]).sum())
                sensitivity.append(dict(road=road, windows=len(group), radius=radius,
                                        from_seconds=-240, to_seconds=end, **totals))

    # Deduplicate mapped physical sensor/time observations before localization.
    unique = {}
    unmapped_finite, repeated, conflicts = 0, 0, 0
    bad_window_dates = set()
    for r in records:
        used = np.flatnonzero((r['relative'] >= -240) & (r['relative'] <= 420))
        bad = checks(r['values'][used])
        if (bad['volume_gt_1000'] | bad['occupancy_gt_100']).any():
            bad_window_dates.add(r['date'])
        for i in used:
            for j, sensor in enumerate(r['sensors']):
                value = r['values'][i, j]
                if sensor is None:
                    unmapped_finite += int(np.isfinite(value).sum())
                    continue
                key = sensor, int(r['time'][i])
                if key in unique:
                    repeated += 1
                    old = unique[key]['value']
                    conflicts += int(not np.array_equal(value, old, equal_nan=True))
                else:
                    unique[key] = dict(value=value, raw=r['raw'][i, j], window=r['window_token'],
                                       offset=int(r['relative'][i]), hop=j-5)
    if conflicts:
        raise ValueError('Conflicting repeated sensor/time values')
    by_sensor = defaultdict(Counter)
    examples = {}
    for (sensor, time), row in unique.items():
        stats = by_sensor[sensor]
        stats['sensor_timestamps'] += 1
        value = row['value']
        for kind, mask in checks(value).items():
            stats[kind] += int(mask)
        for j, kind in ((1, 'volume'), (2, 'occupancy')):
            if np.isfinite(value[j]) and (kind not in examples or value[j] > examples[kind]['numeric_value']):
                # Public numeric extreme only, no event date, name, or record body.
                examples[kind] = dict(numeric_value=float(value[j]), raw_numeric_token=str(row['raw'][j]),
                                      sensor_token=token(sensor), sensor_time_token=token(f'{sensor}|{time}'),
                                      window_token=row['window'], relative_seconds=row['offset'], hop=row['hop'])
    stations = []
    for sensor, counts in by_sensor.items():
        lane_counts = sorted(lanes.get(sensor, set()))
        stations.append(dict(sensor_token=token(sensor), geometry_lane_counts=lane_counts,
                             geometry_status='unmatched' if not lane_counts else ('unique' if len(lane_counts) == 1 else 'multiple'),
                             **counts))
    stations.sort(key=lambda r: (-r['volume_gt_1000'], r['sensor_token']))
    return {
        'status': 'DIAGNOSTIC_ONLY_NO_REPAIR_OR_TRAINING',
        'archives_sha256': audit.ARCHIVES,
        'script_sha256': audit.digest(Path(__file__).read_bytes()),
        'selection': 'bestData accident; I75/I24; Fatal or Suspected Minor/Serious Injury; counts only match paper',
        'excluded': 'All controls, property-damage accidents, other roads; no certification of author matrix identity',
        'window_convention': 'Inclusive published CSV clock labels from -240 seconds through n=0..420 seconds; bin start/end semantics unverified',
        'thresholds': 'volume>1000 and >1million are diagnostic flags, not calibrated capacity; occupancy>100 violates declared percent mean range',
        'count_matched_candidates': dict(Counter(r['road'] for r in records)),
        'sensitivity': sensitivity,
        'localization': {
            'scope': 'Candidate 55; -240..420 seconds; all 11 hops; topology-implied IDs; wall-clock time',
            'unique_mapped_sensor_timestamps': len(unique),
            'repeated_sensor_timestamps': repeated,
            'repeated_value_conflicts': conflicts,
            'finite_values_on_unmapped_hops': unmapped_finite,
            'sampled_sensors': len(stations),
            'volume_flagged_sensors': sum(s['volume_gt_1000'] > 0 for s in stations),
            'occupancy_flagged_sensors': sum(s['occupancy_gt_100'] > 0 for s in stations),
            'candidate_window_dates': len({r['date'] for r in records}),
            'candidate_window_dates_with_flags': len(bad_window_dates),
            'sensor_profiles': stations,
            'extremes': examples,
        },
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Refuse to overwrite diagnostic output')
    result = run(args.data_dir)
    with args.output.open('x', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps({k: v for k, v in result['localization'].items() if k not in ('sensor_profiles', 'extremes')}))
