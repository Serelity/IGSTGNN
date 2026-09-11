"""Build train/val-only Contra496 v8 raw windows, without training or filling."""
import argparse
import calendar
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import resource
import time

import numpy as np

from .source_v8 import RowCache, V8Reader, save_json, sha256

PROJECT = Path(__file__).resolve().parents[3]
PROTOCOL = PROJECT / '论文学习/时间切分协议_20260911'
DEFAULT_OUTPUT = PROJECT / '论文学习/研究开发_20260911/data_v8_contra'


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def select_development_rows(rows):
    return {split: [row for row in rows if row['split'] == split] for split in ('train', 'val')}


def window_slots(row):
    start = datetime.fromisoformat(row['x_start'])
    if start.tzinfo is not None or start.year != 2023 or start.minute % 5 or start.second or start.microsecond:
        raise ValueError('Expected naive 2023 five-minute nominal labels')
    if int(row['source_version']) != 8:
        raise ValueError('Frozen source version must be 8')
    slots = [start + timedelta(minutes=5 * i) for i in range(26)]
    if any(datetime.fromisoformat(row[key]) != slots[index]
           for key, index in [('x_end', 11), ('y_start', 14), ('y_end', 25)]):
        raise ValueError('Frozen X/Y/latency gap differs from 0:12 and 14:26')
    return slots


def month_plan(rows, split):
    result = {}
    for index, row in enumerate(rows):
        if row['split'] != split or split not in ('train', 'val'):
            raise ValueError('Only matching development split rows are permitted')
        for step, slot in enumerate(window_slots(row)):
            if slot.year != 2023 or slot.month not in range(1, 11):
                raise ValueError('A requested window enters forbidden test months')
            expected_split = 'train' if slot.month <= 8 else 'val'
            if expected_split != split:
                raise ValueError('A requested window crosses the frozen split boundary')
            month = result.setdefault(slot.month, {
                'sample_positions': [], 'window_positions': [], 'month_slots': [],
                'train_x_union': np.zeros(calendar.monthrange(2023, slot.month)[1] * 288, bool)})
            source_slot = (slot.day - 1) * 288 + slot.hour * 12 + slot.minute // 5
            month['sample_positions'].append(index)
            month['window_positions'].append(step)
            month['month_slots'].append(source_slot)
            if split == 'train' and step < 12:
                month['train_x_union'][source_slot] = True
    for month in result.values():
        for key in ('sample_positions', 'window_positions', 'month_slots'):
            month[key] = np.asarray(month[key], dtype=np.int64)
    return result


class TrainMoments:
    def __init__(self, station_ids):
        self.stations = np.asarray(station_ids, dtype=np.int64)
        self.count = np.zeros(len(station_ids), np.int64)
        self.total = np.zeros(len(station_ids), np.float64)
        self.squares = np.zeros(len(station_ids), np.float64)
        self.unique_slots = 0
        self.zero_count = 0

    def update(self, month_flow, train_x_union):
        if not train_x_union.any():
            return
        values = np.asarray(month_flow[:, train_x_union], dtype=np.float64)
        valid = np.isfinite(values) & (values >= 0)
        self.unique_slots += int(train_x_union.sum())
        self.zero_count += int(np.count_nonzero(values == 0))
        self.count += valid.sum(axis=1)
        values[~valid] = 0
        self.total += values.sum(axis=1)
        self.squares += np.square(values).sum(axis=1)

    def finish(self, train_rows):
        count = int(self.count.sum())
        if not count:
            raise ValueError('No valid training X observations; no validation fallback')
        mean = float(self.total.sum() / count)
        variance = float(self.squares.sum() / count - mean * mean)
        if not np.isfinite(variance) or variance <= 0:
            raise ValueError('Training standard deviation is not positive')
        return {'status': 'conditional_development', 'source_version': 8,
                'fit_scope': 'train_X_0:12_unique_station_nominal_slot_finite_nonnegative',
                'mean': mean, 'std': float(np.sqrt(variance)),
                'station_ids': self.stations.tolist(),
                'node_fill_mean': [float(total / n) if n else None
                                   for total, n in zip(self.total, self.count)],
                'valid_unique_input_count_per_station': self.count.tolist(),
                'all_missing_station_ids': self.stations[self.count == 0].tolist(),
                'unique_training_nominal_slots': self.unique_slots,
                'unique_training_input_keys': self.unique_slots * len(self.stations),
                'valid_training_input_count': count, 'zero_training_input_count': self.zero_count,
                'fitted_sample_indices': [int(row['sample_index']) for row in train_rows],
                'main_training_ready': False}


def prepare(output):
    manifest_path = PROTOCOL / 'manifest/manifest.csv'
    summary_path = PROTOCOL / 'manifest/summary.json'
    frozen = json.loads(summary_path.read_text(encoding='utf-8'))
    if sha256(manifest_path) != frozen['files']['manifest.csv']:
        raise ValueError('Frozen manifest checksum changed')
    if sha256(PROTOCOL / '数据协议.md') != frozen['protocol_sha256']:
        raise ValueError('Frozen protocol checksum changed')
    all_rows = read_csv(manifest_path)
    counts = {split: sum(row['split'] == split for row in all_rows) for split in ('train', 'val', 'test')}
    if counts != {'train': 3604, 'val': 917, 'test': 894} or counts != frozen['split_counts']:
        raise ValueError('Frozen split counts changed')
    rows = select_development_rows(all_rows)
    plans = {split: month_plan(values, split) for split, values in rows.items()}
    sensors_path = PROJECT / 'data/xtraffic/Contra_Costa/sensors.csv'
    order_path = PROJECT.parent / 'data/raw/traffident_v8/node_order.npy'
    sensors = read_csv(sensors_path)
    stations = np.array([int(row['station_id']) for row in sensors], np.int64)
    order = np.load(order_path, allow_pickle=False)
    if len(stations) != 496 or len(np.unique(stations)) != 496 or order.shape != (16972,) or len(np.unique(order)) != 16972:
        raise ValueError('Expected unique 496 published and 16972 raw stations')
    by_id = {int(station): index for index, station in enumerate(order)}
    axes = np.asarray([by_id[int(station)] for station in stations], np.int64)
    if not np.array_equal(axes, [int(row['order']) for row in sensors]):
        raise ValueError('Sensor order does not match source node_order')
    inputs = {str(path): sha256(path) for path in [manifest_path, summary_path, sensors_path, order_path,
                                                  PROTOCOL / '数据协议.md']}
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / 'build_inputs.json'
    if state_path.exists() and json.loads(state_path.read_text()) != inputs:
        raise ValueError('Existing build input fingerprints differ; preserve prior artifacts')
    save_json(state_path, inputs)
    for split, values in rows.items():
        target = output / f'{split}_manifest.csv'
        if target.exists():
            if read_csv(target) != values:
                raise ValueError('Existing split manifest differs')
        else:
            with target.open('w', encoding='utf-8', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(values[0]))
                writer.writeheader()
                writer.writerows(values)
    for name, array in [('station_ids', stations), ('raw_node_indices', axes)]:
        target = output / f'{name}.npy'
        if target.exists():
            if not np.array_equal(np.load(target, allow_pickle=False), array):
                raise ValueError('Existing node axis differs')
        else:
            np.save(target, array, allow_pickle=False)
    return rows, plans, stations, axes, inputs


def fetch_month(reader, cache, month, axes, workers, output):
    remote = reader.open_month(month, cache)
    flow = np.empty((len(axes), remote['layout']['shape'][1]), np.float32)
    records, failures = [], []
    began = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(reader.read_node, remote, int(axis), cache): i for i, axis in enumerate(axes)}
        for number, future in enumerate(as_completed(futures), 1):
            i = futures[future]
            try:
                payload, record = future.result()
                flow[i] = np.frombuffer(payload, dtype='<f4').reshape(-1, 3)[:, 0]
                records.append({**record, 'published_node_index': i})
            except Exception as error:
                failures.append({'month': month, 'published_node_index': i,
                                 'raw_node_index': int(axes[i]), 'error_type': type(error).__name__,
                                 'note': 'No NaN substitution; cache permits retry on next invocation'})
            if number == 1 or number % 100 == 0 or number == len(axes):
                print(f'month={month:02d} rows={number}/{len(axes)} failures={len(failures)} '
                      f'elapsed={time.perf_counter() - began:.1f}s', flush=True)
    record = {'month': month, 'layout': remote['layout'], 'header': remote['header_record'],
              'rows': sorted(records, key=lambda r: r['published_node_index']), 'failures': failures,
              'elapsed_seconds': time.perf_counter() - began}
    save_json(output / f'source_month_{month:02d}.json', record)
    if failures:
        raise RuntimeError(f'Month {month} has failed source rows; see saved failure records')
    return flow, record


def compare_pilot(output, rows):
    pilot_path = PROTOCOL / 'pilot/pilot_raw.npz'
    pilot_summary = json.loads((PROTOCOL / 'pilot/summary.json').read_text(encoding='utf-8'))
    if sha256(pilot_path) != pilot_summary['files']['pilot_raw.npz']:
        raise ValueError('Existing pilot fingerprint changed')
    arrays = {split: np.load(output / f'{split}_flow.npy', mmap_mode='r', allow_pickle=False)
              for split in ('train', 'val')}
    location = {int(row['sample_index']): (split, i) for split, values in rows.items()
                for i, row in enumerate(values)}
    stations = np.load(output / 'station_ids.npy', allow_pickle=False)
    with np.load(pilot_path, allow_pickle=False) as pilot:
        nodes = pilot['node_indices']
        if not np.array_equal(stations[nodes], pilot['station_ids']):
            raise ValueError('Pilot station mapping differs')
        expected = pilot['data'][..., 0]
        actual = np.stack([arrays[location[int(sample)][0]][location[int(sample)][1]][:, nodes]
                           for sample in pilot['sample_indices']])
        equal = (actual == expected) | (np.isnan(actual) & np.isnan(expected))
        result = {'pilot_sha256': sha256(pilot_path), 'shape': list(actual.shape),
                  'compared_flow_points': int(equal.size), 'mismatched_points': int((~equal).sum()),
                  'source_missing_points': int(np.isnan(expected).sum()), 'all_equal': bool(equal.all())}
    save_json(output / 'pilot_comparison.json', result)
    if not result['all_equal']:
        raise ValueError('Existing 12-sample seven-station raw pilot differs')
    return result


def build(output, workers=10, preflight=False):
    began = time.perf_counter()
    rows, plans, stations, axes, inputs = prepare(output)
    reader, cache = V8Reader(), RowCache(output / 'row_cache')
    if preflight:
        flow, record = fetch_month(reader, cache, 1, axes[:3], min(workers, 3), output)
        result = {'status': 'preflight_pass', 'shape': list(flow.shape),
                  'source_version': 8, 'months_requested': [1], 'rows_requested': 3,
                  'range_protocol': '206_exact_content_range_total_and_body_length',
                  'reader_stats': reader.stats(), 'main_training_ready': False}
        save_json(output / 'preflight.json', result)
        print(json.dumps(result, indent=2), flush=True)
        return result
    if not (output / 'preflight.json').exists():
        raise ValueError('Run --preflight first')
    if any((output / name).exists() for name in ('train_flow.npy', 'val_flow.npy', 'scaler.json', 'summary.json')):
        raise FileExistsError('Preserve completed data files; use a separate output directory')
    arrays = {split: np.lib.format.open_memmap(output / f'{split}_flow.npy.partial', mode='w+',
                                              dtype=np.float32, shape=(len(values), 26, len(stations)))
              for split, values in rows.items()}
    moments = TrainMoments(stations)
    month_records = []
    months = sorted({month for plan in plans.values() for month in plan})
    for month in months:
        flow, record = fetch_month(reader, cache, month, axes, workers, output)
        month_records.append(record)
        for split, plan in plans.items():
            if month not in plan:
                continue
            item = plan[month]
            # Bound temporary gather memory independently of the window count.
            for first in range(0, len(item['month_slots']), 1024):
                block = slice(first, first + 1024)
                arrays[split][item['sample_positions'][block], item['window_positions'][block], :] = flow[:, item['month_slots'][block]].T
            moments.update(flow, item['train_x_union'])
            arrays[split].flush()
        del flow
    scaler = moments.finish(rows['train'])
    save_json(output / 'scaler.json', scaler)
    if scaler['all_missing_station_ids']:
        raise ValueError('Stations without valid train X; scaler.json records diagnostics, pipeline not accepted')
    for split in arrays:
        arrays[split].flush()
    del arrays
    for split in rows:
        (output / f'{split}_flow.npy.partial').replace(output / f'{split}_flow.npy')
    pilot = compare_pilot(output, rows)
    filenames = ['train_flow.npy', 'val_flow.npy', 'train_manifest.csv', 'val_manifest.csv',
                 'station_ids.npy', 'raw_node_indices.npy', 'scaler.json', 'build_inputs.json',
                 'pilot_comparison.json', 'preflight.json']
    filenames += [f'source_month_{month:02d}.json' for month in months]
    records = [r for month in month_records for r in month['rows']]
    summary = {'status': 'conditional_development', 'build_complete': True, 'main_training_ready': False,
               'online_semantics_certified': False, 'source_version': 8,
               'source_dataset': 'gpxlcj/xtraffic', 'split_counts': {s: len(v) for s, v in rows.items()},
               'test_manifest_count': 894, 'test_flow_built': False, 'months_requested': months,
               'station_count': len(stations), 'dtype': 'float32',
               'shapes': {s: [len(v), 26, len(stations)] for s, v in rows.items()},
               'X_slice': [0, 12], 'excluded_latency_slice': [12, 14], 'Y_slice': [14, 26],
               'raw_missing_preserved': True, 'row_requests': len(records),
               'cache_hit_rows': sum(r['cache_hit'] for r in records),
               'selected_source_row_bytes': sum(r['bytes'] for r in records),
               'reader_stats': reader.stats(), 'pilot_comparison': pilot,
               'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
               'elapsed_seconds': time.perf_counter() - began,
               'created_utc': datetime.now(timezone.utc).isoformat(), 'inputs': inputs,
               'code_sha256': {p.name: sha256(p) for p in [Path(__file__), Path(__file__).with_name('source_v8.py')]},
               'files': {name: sha256(output / name) for name in filenames},
               'artifact_bytes': {name: (output / name).stat().st_size for name in filenames},
               'all_missing_station_ids': scaler['all_missing_station_ids']}
    if summary['peak_rss_bytes'] >= 2 * 1024 ** 3:
        raise ValueError('Build exceeded the 2 GiB peak RSS acceptance gate')
    save_json(output / 'summary.json', summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in ('inputs', 'files', 'artifact_bytes')},
                     indent=2, ensure_ascii=False), flush=True)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--workers', type=int, default=10, choices=range(1, 13))
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    build(args.output, args.workers, args.preflight)
