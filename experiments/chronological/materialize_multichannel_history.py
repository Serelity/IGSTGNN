"""Materialize cache-only report-time multichannel history for the v11a data gate."""

import argparse
import calendar
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import resource
import time

import numpy as np


CHANNELS = ('flow', 'occupancy', 'speed')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + '\n',
        encoding='utf-8')
    temporary.replace(path)


def read_rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def load_protocol(path):
    protocol = json.loads(Path(path).read_text(encoding='utf-8'))
    if (protocol.get('protocol_id') !=
            'contra_v8_multichannel_history_materialize_v11a' or
            protocol.get('scope') !=
            'cache_only_report_time_multichannel_history_materialization' or
            protocol.get('status') !=
            'prospectively_frozen_before_materialization'):
        raise ValueError('Unexpected v11a protocol identity')
    expected_data = {
        'summary_sha256':
        '7611ac36a0e8ac5fdfb11a9d09487d3505f0fae24b0da05af7cbdcd24f113dd7',
        'source_version': 8, 'source_year': 2023,
        'expected_split_samples': {'train': 3604, 'val': 917},
        'expected_nodes': 496,
        'expected_months': list(range(1, 11)),
        'expected_source_channels': 3,
    }
    if protocol.get('data_input') != expected_data:
        raise ValueError('v11a data input changed')
    expected_semantics = {
        'source_order': list(CHANNELS),
        'evidence_status':
        'inferred_from_publication_and_numeric_ranges_not_declared_by_source_matrix_metadata',
        'flow_units': 'source_units',
        'occupancy_units': 'unconfirmed_source_units',
        'speed_units': 'unconfirmed_source_units',
    }
    if protocol.get('channel_semantics') != expected_semantics:
        raise ValueError('v11a channel semantics changed')
    expected_materialization = {
        'splits': ['train', 'val'], 'history_steps': 12, 'step_minutes': 5,
        'output_dtype': 'float32',
        'output_axis_order': ['sample', 'history_step', 'node', 'channel'],
        'preserve_raw_nonfinite_and_negative_values': True,
        'existing_flow_channel_exact_anchor': True,
        'fit_statistics_on_unique_train_history_slots_only': True,
        'valid_for_statistics': 'finite_and_nonnegative_with_zero_retained',
        'cache_only': True, 'network_access_prohibited': True,
        'future_gap_and_Y_values_prohibited': True,
    }
    if protocol.get('materialization') != expected_materialization:
        raise ValueError('v11a materialization design changed')
    required_boundary = (
        'development_train_and_validation_manifests_only',
        'report_time_history_only', 'validation_targets_prohibited',
        'validation_residual_arrays_prohibited',
        'test_manifest_and_test_traffic_prohibited',
        'incident_type_description_duration_prohibited',
        'model_training_prohibited', 'performance_evaluation_prohibited',
        'v10a_gate_not_overridden')
    if not all(protocol.get('information_boundary', {}).get(key) is True
               for key in required_boundary):
        raise ValueError('v11a information boundary changed')
    expected_acceptance = {
        'all_declared_source_months_hash_verified': True,
        'all_cached_source_rows_complete_and_hash_verified': True,
        'every_history_cell_materialized_exactly_once': True,
        'flow_channel_exactly_matches_existing_history': True,
        'flow_train_statistics_reproduce_existing_scaler': True,
        'each_channel_has_positive_train_standard_deviation': True,
        'no_channel_has_an_all_missing_train_station': True,
        'no_test_or_future_target_read': True,
    }
    if protocol.get('acceptance') != expected_acceptance:
        raise ValueError('v11a acceptance gate changed')
    expected_authorization = {
        'on_pass':
        'ALLOW_SEPARATE_PROSPECTIVE_MULTICHANNEL_INCREMENTAL_INFORMATION_AUDIT',
        'on_fail': 'STOP_MULTICHANNEL_INFORMATION_BRANCH',
        'engineering_check_on_pass': 'NO_SCIENTIFIC_AUTHORIZATION',
        'does_not_authorize_v10b': True,
        'does_not_authorize_neural_expert': True,
        'does_not_authorize_test_access': True,
    }
    if protocol.get('authorization') != expected_authorization:
        raise ValueError('v11a authorization changed')
    return protocol


def history_month_plan(rows, split):
    if split not in ('train', 'val'):
        raise ValueError('Only development train and validation are permitted')
    result = {}
    for sample_position, row in enumerate(rows):
        if row.get('split') != split or int(row.get('source_version', -1)) != 8:
            raise ValueError('Manifest row has an unexpected split or source version')
        start = datetime.fromisoformat(row['x_start'])
        end = datetime.fromisoformat(row['x_end'])
        if (start.tzinfo is not None or start.year != 2023 or start.minute % 5 or
                start.second or start.microsecond or
                end != start + timedelta(minutes=55)):
            raise ValueError('History timestamps differ from the frozen 12-step grid')
        for step in range(12):
            point = start + timedelta(minutes=5 * step)
            if point.year != 2023 or point.month not in range(1, 11):
                raise ValueError('History requests a forbidden source month')
            expected_split = 'train' if point.month <= 8 else 'val'
            if expected_split != split:
                raise ValueError('History crosses the frozen split boundary')
            month = result.setdefault(point.month, {
                'sample_positions': [], 'history_positions': [],
                'month_slots': [],
                'train_unique_slots': np.zeros(
                    calendar.monthrange(2023, point.month)[1] * 288,
                    dtype=bool),
            })
            source_slot = ((point.day - 1) * 288 + point.hour * 12 +
                           point.minute // 5)
            month['sample_positions'].append(sample_position)
            month['history_positions'].append(step)
            month['month_slots'].append(source_slot)
            if split == 'train':
                month['train_unique_slots'][source_slot] = True
    for month in result.values():
        for key in ('sample_positions', 'history_positions', 'month_slots'):
            month[key] = np.asarray(month[key], dtype=np.int64)
    return result


class ChannelMoments:
    def __init__(self, station_ids):
        self.station_ids = np.asarray(station_ids, dtype=np.int64)
        shape = (len(self.station_ids), len(CHANNELS))
        self.count = np.zeros(shape, dtype=np.int64)
        self.total = np.zeros(shape, dtype=np.float64)
        self.squares = np.zeros(shape, dtype=np.float64)
        self.zero_count = np.zeros(len(CHANNELS), dtype=np.int64)
        self.negative_count = np.zeros(len(CHANNELS), dtype=np.int64)
        self.nonfinite_count = np.zeros(len(CHANNELS), dtype=np.int64)
        self.unique_slots = 0

    def update(self, month_values, train_unique_slots):
        values = np.asarray(month_values)
        slots = np.asarray(train_unique_slots)
        if (values.ndim != 3 or values.shape[0] != len(self.station_ids) or
                values.shape[2] != len(CHANNELS) or
                slots.shape != (values.shape[1],)):
            raise ValueError('Channel moments received an unexpected source shape')
        if not slots.any():
            return
        selected = values[:, slots, :].astype(np.float64)
        finite = np.isfinite(selected)
        valid = finite & (selected >= 0)
        self.unique_slots += int(slots.sum())
        self.zero_count += np.count_nonzero(selected == 0, axis=(0, 1))
        self.negative_count += np.count_nonzero(finite & (selected < 0), axis=(0, 1))
        self.nonfinite_count += np.count_nonzero(~finite, axis=(0, 1))
        self.count += valid.sum(axis=1)
        selected[~valid] = 0
        self.total += selected.sum(axis=1)
        self.squares += np.square(selected).sum(axis=1)

    def finish(self, sample_indices):
        global_count = self.count.sum(axis=0)
        if np.any(global_count == 0):
            raise ValueError('At least one channel has no valid training history')
        mean = self.total.sum(axis=0) / global_count
        variance = self.squares.sum(axis=0) / global_count - np.square(mean)
        if not np.isfinite(variance).all() or np.any(variance <= 0):
            raise ValueError('At least one channel lacks positive training variance')
        node_fill = np.divide(
            self.total, self.count,
            out=np.full_like(self.total, np.nan), where=self.count > 0)
        return {
            'status': 'MULTICHANNEL_TRAIN_STATISTICS_COMPLETE',
            'fit_scope':
            'unique_train_X_12_steps_station_channel_finite_nonnegative',
            'channel_order': list(CHANNELS),
            'mean': mean.tolist(), 'std': np.sqrt(variance).tolist(),
            'valid_count': global_count.tolist(),
            'zero_count': self.zero_count.tolist(),
            'negative_count': self.negative_count.tolist(),
            'nonfinite_count': self.nonfinite_count.tolist(),
            'station_ids': self.station_ids.tolist(),
            'node_valid_count': self.count.tolist(),
            'node_fill_mean': node_fill.tolist(),
            'all_missing_station_ids_by_channel': {
                CHANNELS[channel]: self.station_ids[
                    self.count[:, channel] == 0].tolist()
                for channel in range(len(CHANNELS))},
            'unique_training_nominal_slots': self.unique_slots,
            'fitted_sample_indices': [int(value) for value in sample_indices],
        }


class VerifiedMultichannelMonthCache:
    def __init__(self, data_dir, axes, expected_files):
        self.data_dir = Path(data_dir)
        self.axes = np.asarray(axes, dtype=np.int64)
        self.expected_files = dict(expected_files)
        self.blobs = self.data_dir / 'row_cache' / 'blobs'
        self.months = []
        self.row_records = 0
        self.unique_blobs = set()

    def load(self, month):
        manifest_path = self.data_dir / f'source_month_{month:02d}.json'
        expected_hash = self.expected_files.get(manifest_path.name)
        if expected_hash is None or sha256(manifest_path) != expected_hash:
            raise ValueError(f'Source-month manifest differs: {month}')
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        slots = calendar.monthrange(2023, month)[1] * 288
        layout = manifest.get('layout', {})
        if (manifest.get('month') != month or
                layout.get('shape') != [16972, slots, len(CHANNELS)] or
                layout.get('dtype') != '<f4' or manifest.get('failures')):
            raise ValueError(f'Invalid completed source-month manifest: {month}')
        header_hash = manifest.get('header', {}).get('sha256')
        rows = sorted(manifest.get('rows', []),
                      key=lambda row: row['published_node_index'])
        if ([row.get('published_node_index') for row in rows] !=
                list(range(len(self.axes)))):
            raise ValueError(f'Source month lacks the complete node axis: {month}')
        result = np.empty((len(self.axes), slots, len(CHANNELS)), dtype=np.float32)
        expected_bytes = slots * len(CHANNELS) * np.dtype('<f4').itemsize
        for index, row in enumerate(rows):
            identity = row.get('identity', {})
            if (identity.get('dataset') != 'gpxlcj/xtraffic' or
                    int(identity.get('version', -1)) != 8 or
                    int(identity.get('year', -1)) != 2023 or
                    int(identity.get('month', -1)) != month or
                    int(identity.get('raw_node_index', -1)) != int(self.axes[index]) or
                    identity.get('header_sha256') != header_hash or
                    int(row.get('bytes', -1)) != expected_bytes or
                    int(identity.get('end', -1)) - int(identity.get('start', 0)) + 1 !=
                    expected_bytes):
                raise ValueError('Cached source-row identity differs')
            digest = row.get('sha256')
            blob = self.blobs / f'{digest}.bin'
            payload = blob.read_bytes()
            if (len(payload) != expected_bytes or
                    hashlib.sha256(payload).hexdigest() != digest):
                raise ValueError('Cached source-row checksum or length differs')
            result[index] = np.frombuffer(payload, dtype='<f4').reshape(
                slots, len(CHANNELS))
            self.row_records += 1
            self.unique_blobs.add(digest)
        self.months.append(month)
        return result


def verify_data_package(data_dir, protocol):
    data_dir = Path(data_dir)
    summary_path = data_dir / 'summary.json'
    if sha256(summary_path) != protocol['data_input']['summary_sha256']:
        raise ValueError('Chronological package summary differs from v11a')
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    expected = protocol['data_input']
    if (summary.get('status') != 'conditional_development' or
            summary.get('build_complete') is not True or
            summary.get('source_version') != expected['source_version'] or
            summary.get('split_counts') != expected['expected_split_samples'] or
            summary.get('station_count') != expected['expected_nodes'] or
            summary.get('months_requested') != expected['expected_months'] or
            summary.get('X_slice') != [0, 12] or
            summary.get('test_flow_built') is not False):
        raise ValueError('Chronological package boundary differs from v11a')
    required = [
        'train_flow.npy', 'val_flow.npy', 'train_manifest.csv', 'val_manifest.csv',
        'station_ids.npy', 'raw_node_indices.npy', 'scaler.json']
    required += [f'source_month_{month:02d}.json'
                 for month in expected['expected_months']]
    files = summary.get('files', {})
    if any(name not in files or sha256(data_dir / name) != files[name]
           for name in required):
        raise ValueError('Chronological package input fingerprint differs')
    station_ids = np.load(data_dir / 'station_ids.npy', allow_pickle=False)
    axes = np.load(data_dir / 'raw_node_indices.npy', allow_pickle=False)
    if (station_ids.shape != (expected['expected_nodes'],) or
            axes.shape != station_ids.shape or
            len(np.unique(station_ids)) != len(station_ids) or
            len(np.unique(axes)) != len(axes)):
        raise ValueError('Chronological station axis differs')
    return summary, station_ids, axes


def compare_raw(left, right):
    left, right = np.asarray(left), np.asarray(right)
    if left.shape != right.shape:
        raise ValueError('Raw comparison shapes differ')
    equal = (left == right) | (np.isnan(left) & np.isnan(right))
    return int(np.count_nonzero(~equal)), int(equal.size)


def split_channel_statistics(values):
    result = {}
    for channel, name in enumerate(CHANNELS):
        current = np.asarray(values[..., channel])
        finite = np.isfinite(current)
        valid = finite & (current >= 0)
        result[name] = {
            'cells': int(current.size), 'finite_cells': int(finite.sum()),
            'valid_nonnegative_cells': int(valid.sum()),
            'negative_cells': int(np.count_nonzero(finite & (current < 0))),
            'nonfinite_cells': int(np.count_nonzero(~finite)),
            'zero_cells': int(np.count_nonzero(current == 0)),
            'minimum_valid': float(current[valid].min()) if valid.any() else None,
            'maximum_valid': float(current[valid].max()) if valid.any() else None,
        }
    return result


def flow_scaler_matches(multichannel, existing):
    counts = np.asarray(multichannel['node_valid_count'], dtype=np.int64)[:, 0]
    fills = np.asarray(multichannel['node_fill_mean'], dtype=np.float64)[:, 0]
    expected_fills = np.asarray(existing['node_fill_mean'], dtype=np.float64)
    return bool(
        multichannel['unique_training_nominal_slots'] ==
        existing['unique_training_nominal_slots'] and
        multichannel['valid_count'][0] == existing['valid_training_input_count'] and
        multichannel['zero_count'][0] == existing['zero_training_input_count'] and
        np.array_equal(counts, existing['valid_unique_input_count_per_station']) and
        np.allclose(fills, expected_fills, rtol=0., atol=1e-12,
                    equal_nan=True) and
        np.isclose(multichannel['mean'][0], existing['mean'], rtol=0., atol=1e-12) and
        np.isclose(multichannel['std'][0], existing['std'], rtol=0., atol=1e-12))


def materialize(data_dir, protocol_path, output, check=False):
    began = time.perf_counter()
    data_dir, protocol_path, output = map(Path, (data_dir, protocol_path, output))
    protocol = load_protocol(protocol_path)
    partial = output.with_name(output.name + '.partial')
    if output.exists() or partial.exists():
        raise FileExistsError('v11a output or partial output exists; use a new directory')
    summary, station_ids, axes = verify_data_package(data_dir, protocol)
    rows = {split: read_rows(data_dir / f'{split}_manifest.csv')
            for split in protocol['materialization']['splits']}
    expected_counts = protocol['data_input']['expected_split_samples']
    if {split: len(values) for split, values in rows.items()} != expected_counts:
        raise ValueError('Development manifest counts differ')
    if check:
        rows = {split: values[:2] for split, values in rows.items()}
    plans = {split: history_month_plan(values, split)
             for split, values in rows.items()}
    partial.mkdir(parents=True)
    arrays, coverage = {}, {}
    for split, values in rows.items():
        arrays[split] = np.lib.format.open_memmap(
            partial / f'{split}_history.npy', mode='w+', dtype=np.float32,
            shape=(len(values), 12, len(station_ids), len(CHANNELS)))
        coverage[split] = np.zeros((len(values), 12), dtype=np.int8)
    moments = ChannelMoments(station_ids)
    cache = VerifiedMultichannelMonthCache(data_dir, axes, summary['files'])
    requested_months = sorted({month for plan in plans.values() for month in plan})
    for month in requested_months:
        source = cache.load(month)
        train_plan = plans['train'].get(month)
        if train_plan is not None:
            moments.update(source, train_plan['train_unique_slots'])
        for split in rows:
            plan = plans[split].get(month)
            if plan is None:
                continue
            sample = plan['sample_positions']
            history = plan['history_positions']
            slots = plan['month_slots']
            arrays[split][sample, history] = source[:, slots, :].transpose(1, 0, 2)
            coverage[split][sample, history] += 1
            arrays[split].flush()
        print(json.dumps({
            'stage': 'source_month_materialized', 'month': month,
            'cached_rows_verified': cache.row_records}), flush=True)
    complete = all(np.all(value == 1) for value in coverage.values())
    if not complete:
        raise ValueError('Every report-time history cell must be materialized once')
    sample_indices = [int(row['sample_index']) for row in rows['train']]
    train_scaler = moments.finish(sample_indices)
    save_json(partial / 'train_multichannel_scaler.json', train_scaler)
    flow_mismatches, flow_cells, split_stats = {}, {}, {}
    for split, values in arrays.items():
        existing = np.load(data_dir / f'{split}_flow.npy', mmap_mode='r',
                           allow_pickle=False)[:len(rows[split]), :12]
        mismatches, cells = compare_raw(values[..., 0], existing)
        flow_mismatches[split], flow_cells[split] = mismatches, cells
        split_stats[split] = split_channel_statistics(values)
        values.flush()
    del arrays
    existing_scaler = json.loads((data_dir / 'scaler.json').read_text())
    scaler_match = False if check else flow_scaler_matches(
        train_scaler, existing_scaler)
    all_months = cache.months == protocol['data_input']['expected_months']
    expected_rows = (len(station_ids) * len(cache.months))
    checks = {
        'all_declared_source_months_hash_verified': all_months,
        'all_cached_source_rows_complete_and_hash_verified':
        cache.row_records == expected_rows,
        'every_history_cell_materialized_exactly_once': complete,
        'flow_channel_exactly_matches_existing_history':
        all(value == 0 for value in flow_mismatches.values()),
        'flow_train_statistics_reproduce_existing_scaler': scaler_match,
        'each_channel_has_positive_train_standard_deviation':
        all(value > 0 for value in train_scaler['std']),
        'no_channel_has_an_all_missing_train_station':
        all(not values for values in
            train_scaler['all_missing_station_ids_by_channel'].values()),
        'no_test_or_future_target_read': True,
    }
    gate_passed = (not check) and all(checks.values())
    engineering_checks = {
        'all_requested_source_months_hash_verified':
        cache.months == requested_months,
        'all_requested_cached_source_rows_complete_and_hash_verified':
        cache.row_records == expected_rows,
        'every_requested_history_cell_materialized_exactly_once': complete,
        'flow_channel_exactly_matches_existing_history':
        all(value == 0 for value in flow_mismatches.values()),
        'each_channel_has_positive_sample_standard_deviation':
        all(value > 0 for value in train_scaler['std']),
        'no_channel_has_an_all_missing_sample_station':
        all(not values for values in
            train_scaler['all_missing_station_ids_by_channel'].values()),
        'no_test_or_future_target_read': True,
    }
    engineering_passed = check and all(engineering_checks.values())
    successful = gate_passed or engineering_passed
    output_names = [f'{split}_history.npy' for split in rows]
    output_names.append('train_multichannel_scaler.json')
    report = {
        'status': ('MULTICHANNEL_HISTORY_ENGINEERING_CHECK_PASS' if engineering_passed
                   else 'MULTICHANNEL_HISTORY_MATERIALIZATION_COMPLETE' if gate_passed
                   else 'MULTICHANNEL_HISTORY_MATERIALIZATION_FAILED'),
        'scope': protocol['scope'], 'protocol_id': protocol['protocol_id'],
        'protocol_sha256': sha256(protocol_path),
        'engineering_check': bool(check),
        'main_training_ready': False, 'model_training_performed': False,
        'performance_evaluation_performed': False,
        'network_access_performed': False, 'future_Y_values_read': False,
        'validation_targets_read': False,
        'validation_residual_arrays_read': False, 'test_split_read': False,
        'incident_semantic_fields_read': False,
        'channel_semantics': protocol['channel_semantics'],
        'source_cache': {
            'months_verified': cache.months,
            'cached_rows_verified': cache.row_records,
            'unique_blob_hashes_verified': len(cache.unique_blobs),
        },
        'splits': {
            split: {
                'samples': len(rows[split]),
                'shape': [len(rows[split]), 12, len(station_ids), len(CHANNELS)],
                'flow_anchor_cells': flow_cells[split],
                'flow_anchor_mismatches': flow_mismatches[split],
                'channel_statistics': split_stats[split],
            } for split in rows},
        'acceptance': {
            'checks': checks, 'gate_evaluated': not check,
            'gate_passed': gate_passed,
            'engineering_checks': engineering_checks,
            'engineering_check_passed': engineering_passed,
        },
        'authorization': {
            'recommendation': (
                protocol['authorization']['engineering_check_on_pass'] if engineering_passed
                else protocol['authorization']['on_pass'] if gate_passed
                else protocol['authorization']['on_fail']),
            'v10b_authorized': False, 'neural_expert_authorized': False,
            'test_access_authorized': False,
        },
        'inputs': {
            'data_summary_sha256': sha256(data_dir / 'summary.json'),
            'protocol_sha256': sha256(protocol_path),
            'code_sha256': sha256(Path(__file__)),
        },
        'outputs': {
            name: {'sha256': sha256(partial / name),
                   'bytes': (partial / name).stat().st_size}
            for name in output_names},
        'environment': {
            'numpy_version': np.__version__,
            'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            'elapsed_seconds': time.perf_counter() - began,
            'created_utc': datetime.now(timezone.utc).isoformat(),
        },
        'interpretation':
        'Report-time data availability only; not model performance or incident effect evidence',
    }
    save_json(partial / 'summary.json', report)
    if not successful:
        raise RuntimeError('v11a multichannel materialization failed its frozen data gate')
    partial.replace(output)
    print(json.dumps({
        'status': report['status'], 'samples': {k: len(v) for k, v in rows.items()},
        'months_verified': cache.months, 'flow_anchor_mismatches': flow_mismatches,
        'channel_train_std': train_scaler['std'],
    }, indent=2), flush=True)
    print(f'Saved v11a multichannel history: {output / "summary.json"}', flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, default=Path(__file__).with_name(
        'multichannel_history_materialize_v11a.json'))
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    materialize(args.data_dir, args.protocol, args.output, args.check)


if __name__ == '__main__':
    main()
