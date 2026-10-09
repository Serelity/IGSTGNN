"""Read-only, full-file audit of the two published Chattanooga ZIP archives.

No fitting, imputation, extraction, network access, or training. Numeric rows are
never exported. Overlap means shared sensor/time, not causal or label validity.
"""
import argparse
import csv
import hashlib
import io
import json
import math
import platform
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import numpy as np

ARCHIVES = {
    'annotatedData.zip': '79989452d1c74a05d9c93454d73cfc57643fe9c2e642e9e3826b46f0d73b83e4',
    'metaData.zip': '954cbe4fa01a6be2ded9f335d415966cc2de3b828424286164a12fc7645ac2d3',
}
KINDS = ('speed', 'volume', 'occupancy')
HOPS = list(range(-5, 6))
BASE = ['incident at sensor (i)', 'road', 'mile', 'type', 'date',
        'incident_time', 'incident_hour', 'data_time', 'weather', 'light']
EXPECTED = BASE + [f'{k} (i{h:+d})' if h else f'{k} (i)' for h in HOPS for k in KINDS]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def regular_member(info):
    p = PurePosixPath(info.filename)
    return not info.is_dir() and '__MACOSX' not in p.parts and not p.name.startswith('.')


def preflight(archive):
    infos = archive.infolist()
    if len(infos) > 100000 or sum(i.file_size for i in infos) > 1024 ** 3:
        raise ValueError('Archive scope exceeds audited bounds')
    if len({i.filename for i in infos}) != len(infos):
        raise ValueError('Duplicate archive member names')
    for i in infos:
        p = PurePosixPath(i.filename)
        if p.is_absolute() or '..' in p.parts or '\\' in i.filename or ':' in i.filename:
            raise ValueError('Unsafe archive member path')
        if i.flag_bits & 1 or (i.external_attr >> 16) & 0o170000 == 0o120000:
            raise ValueError('Encrypted or linked archive member')
        if i.file_size > 16 * 1024 ** 2 or i.file_size / max(i.compress_size, 1) > 500:
            raise ValueError('Archive member exceeds audited bounds')
    return [i for i in infos if regular_member(i)]


def table(data):
    rows = list(csv.reader(io.StringIO(data.decode('utf-8-sig')), strict=True))
    if not rows or len(rows) > 10000 or len(rows[0]) > 100:
        raise ValueError('CSV exceeds bounds or is empty')
    if any(len(r) != len(rows[0]) for r in rows):
        raise ValueError('Nonrectangular CSV')
    return rows[0], rows[1:]


def clock_seconds(value):
    parts = [int(s) for s in value.split(':')]
    if len(parts) not in (2, 3):
        raise ValueError('Unexpected clock format')
    hh, mm = parts[:2]
    ss = parts[2] if len(parts) == 3 else 0
    if not (0 <= hh < 24 and 0 <= mm < 60 and 0 <= ss < 60):
        raise ValueError('Invalid clock')
    return hh * 3600 + mm * 60 + ss


def relative_times(values, report_seconds):
    # Event windows are < 24 h. Match each wall-clock value to the nearest day.
    # This handles midnight but does not disambiguate daylight-saving folds.
    return np.array([(clock_seconds(s) - report_seconds + 43200) % 86400 - 43200
                     for s in values], dtype=np.int64)


def neighbourhood(center, topo):
    result = {0: center if center in topo else None}
    for sign, field in [(-1, 'Previous'), (1, 'Next')]:
        cur = center
        for hop in range(1, 6):
            cur = topo[cur][field] if cur in topo else None
            if cur not in topo:
                cur = None
            result[sign * hop] = cur
    return [result[h] for h in HOPS]


def sensor_alias(row):
    """Window filenames use Road + Heading + unpadded Mile, unlike Name."""
    return row['Road'] + row['Heading'] + f"{float(row['Mile']):.1f}"


def distribution(values):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return {'n': 0}
    q = np.quantile(x, [0, .01, .25, .5, .75, .99, 1])
    median = float(q[3])
    lo, hi = q[2] - 1.5 * (q[4] - q[2]), q[4] + 1.5 * (q[4] - q[2])
    trimmed = x[(x >= q[1]) & (x <= q[-2])]
    return {'n': int(len(x)), 'min_p01_p25_median_p75_p99_max': q.tolist(),
            'mean': float(x.mean()), 'std_population': float(x.std()),
            'mad': float(np.median(np.abs(x - median))),
            'outside_1_5_iqr_fraction': float(np.mean((x < lo) | (x > hi))),
            'mean_within_p01_p99': float(trimmed.mean())}


class Profile:
    def __init__(self):
        self.windows = 0
        self.rows = 0
        self.finite = np.zeros((11, 3), dtype=np.int64)
        self.empty = self.finite.copy()
        self.negative = self.finite.copy()
        self.zero = self.finite.copy()
        self.structural_absent_cells = 0
        self.missing_on_mapped_cells = 0
        self.sample = [[] for _ in KINDS]
        self.sums = np.zeros(3)
        self.squares = np.zeros(3)
        self.extrema = np.array([[np.inf, -np.inf]] * 3)
        self.flags = Counter()

    def add(self, a, empty, ids, flags):
        self.windows += 1
        self.rows += len(a)
        valid = np.isfinite(a)
        self.finite += valid.sum(axis=0)
        self.empty += empty.sum(axis=0)
        self.negative += (a < 0).sum(axis=0)
        self.zero += (a == 0).sum(axis=0)
        mapped = np.array([s is not None for s in ids])
        self.structural_absent_cells += int((~valid[:, ~mapped, :]).sum())
        self.missing_on_mapped_cells += int((~valid[:, mapped, :]).sum())
        clean = np.where(valid, a, 0).astype(np.float64)
        self.sums += clean.sum(axis=(0, 1))
        self.squares += (clean ** 2).sum(axis=(0, 1))
        for k in range(3):
            # Deterministic systematic sample for robust distributions only.
            self.sample[k].extend(a[:, :, k].ravel()[::19].tolist())
            values = a[:, :, k][valid[:, :, k]]
            if len(values):
                self.extrema[k, 0] = min(self.extrema[k, 0], values.min())
                self.extrema[k, 1] = max(self.extrema[k, 1], values.max())
        self.flags.update(flags)

    def result(self):
        out = {'windows': self.windows, 'rows': self.rows, 'flags': dict(self.flags),
               'missing_on_unmapped_hops_cells': self.structural_absent_cells,
               'missing_on_mapped_hops_cells': self.missing_on_mapped_cells,
               'hop_order': HOPS, 'metrics': {}}
        for k, name in enumerate(KINDS):
            n = int(self.finite[:, k].sum())
            mean = float(self.sums[k] / n) if n else None
            out['metrics'][name] = {
                'finite': n, 'total': self.rows * 11,
                'nonfinite': self.rows * 11 - n, 'empty': int(self.empty[:, k].sum()),
                'negative': int(self.negative[:, k].sum()), 'zero': int(self.zero[:, k].sum()),
                'nonfinite_by_hop': (self.rows - self.finite[:, k]).tolist(),
                'exact_mean': mean,
                'exact_min_max': self.extrema[k].tolist() if n else None,
                'exact_std_population': math.sqrt(max(0, self.squares[k] / n - mean ** 2)) if n else None,
                'systematic_every_19th_per_window_sample': distribution(self.sample[k]),
            }
        return out


class Union:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, i):
        while i != self.parent[i]:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def join(self, a, b):
        self.parent[self.find(a)] = self.find(b)


def overlap_audit(records):
    """Exact shared timestamps within topology-implied sensor identities."""
    intervals = defaultdict(list)
    for i, r in enumerate(records):
        for j, sensor in enumerate(r['_ids']):
            if sensor is not None and len(r['_times']):
                intervals[sensor].append((int(r['_times'].min()), int(r['_times'].max()), i, j))
    pairs, different_labels, both_best = set(), set(), set()
    overlap_windows, cross_calendar = set(), set()
    uf = Union(len(records))
    shared, disagree, finite_comparisons = 0, 0, 0
    for spans in intervals.values():
        active = []
        for start, end, i, hop in sorted(spans):
            active = [s for s in active if s[1] >= start]
            for _, _, j, otherhop in active:
                if i == j:
                    continue
                _, a, b = np.intersect1d(records[i]['_times'], records[j]['_times'], return_indices=True)
                if not len(a):
                    continue
                x, y = records[i]['_array'][a, hop], records[j]['_array'][b, otherhop]
                valid = np.isfinite(x) & np.isfinite(y)
                finite_comparisons += int(valid.sum())
                disagree += int((valid & (np.abs(x - y) > 1e-6)).sum())
                shared += len(a)
                key = tuple(sorted((i, j)))
                pairs.add(key)
                if len(pairs) > 2000000:
                    raise ValueError('Overlap pair bound exceeded')
                overlap_windows.update(key)
                uf.join(i, j)
                if records[i]['label'] != records[j]['label']:
                    different_labels.add(key)
                if records[i]['best'] and records[j]['best']:
                    both_best.add(key)
                if records[i]['candidate_calendar_split'] != records[j]['candidate_calendar_split']:
                    cross_calendar.add(key)
            active.append((start, end, i, hop))
    components = Counter(uf.find(i) for i in range(len(records)))
    return {'window_pairs_with_shared_sensor_timestamp': len(pairs),
            'windows_with_any_overlap': len(overlap_windows),
            'pairs_with_different_window_labels': len(different_labels),
            'pairs_with_both_in_best': len(both_best),
            'shared_sensor_timestamps_summed_over_pairs': shared,
            'finite_value_comparisons': finite_comparisons,
            'finite_value_disagreements_absolute_gt_1e_6': disagree,
            'overlap_connected_components': len(components),
            'largest_component_windows': max(components.values(), default=0),
            'component_size_histogram': dict(Counter(components.values())),
            'pairs_crossing_candidate_calendar_split': len(cross_calendar),
            'scope': 'allData only; topology-implied sensor identities; naive local clocks; no DST fold resolution'}


def main():
    script_sha256 = digest(Path(__file__).read_bytes())
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    args = ap.parse_args()
    root = args.data_dir.resolve(strict=True)
    out = args.output_dir.resolve()
    if out.exists():
        raise ValueError('Use a new output directory; prior audit is preserved')
    sources = {}
    for name, expected in ARCHIVES.items():
        path = root / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 512 * 1024 ** 2:
            raise ValueError('Invalid source file')
        actual = digest(path.read_bytes())
        if actual != expected:
            raise ValueError('Source differs from verified publisher download')
        sources[name] = {'bytes': path.stat().st_size, 'sha256': actual}
    with ZipFile(root / 'metaData.zip') as z:
        metadata = {PurePosixPath(i.filename).name: i.filename for i in preflight(z)}
        header, body = table(z.read(metadata['SensorTopology.csv']))
        topo_rows = [dict(zip(header, r)) for r in body]
        topo = {r['Name']: r for r in topo_rows}
        if len(topo) != len(topo_rows):
            raise ValueError('Duplicated topology keys')
        aliases = {sensor_alias(r): r['Name'] for r in topo_rows}
        if len(aliases) != len(topo):
            raise ValueError('Ambiguous Road/Heading/Mile aliases')
        ah, ab = table(z.read(metadata['Accidents.csv']))
        accidents = [dict(zip(ah, r)) for r in ab]
    topo_flags = Counter()
    lengths, mile_residual = [], []
    for name, r in topo.items():
        for side, reverse, dist in [('Previous', 'Next', 'PrevDist'), ('Next', 'Previous', 'NextDist')]:
            neighbour = r[side]
            if neighbour not in topo:
                topo_flags[side + '_not_in_topology'] += 1
                topo_flags[side + ('_empty_endpoint' if not neighbour.strip() else '_dangling_nonempty_reference')] += 1
                continue
            n = topo[neighbour]
            topo_flags[side + '_nonreciprocal'] += n[reverse] != name
            topo_flags[side + '_road_or_heading_change'] += (r['Road'], r['Heading']) != (n['Road'], n['Heading'])
            d = float(r[dist])
            lengths.append(d)
            topo_flags['nonpositive_directed_distance'] += d <= 0
            mile_residual.append(abs(d - abs(float(r['Mile']) - float(n['Mile']))))
        ids = [s for s in neighbourhood(name, topo) if s is not None]
        topo_flags['neighbourhood_repeats_sensor'] += len(ids) != len(set(ids))
    metadata_report = {'accident_records': len(accidents), 'topology_rows': len(topo),
                       'topology_flags': dict(topo_flags), 'directed_distance_raw': distribution(lengths),
                       'distance_minus_absolute_mile_difference': distribution(mile_residual),
                       'accident_field_empty_counts': {h: sum(not r[h].strip() for r in accidents) for h in ah},
                       'corrected_report_clock_changes': sum(clock_seconds(r['Time_orig']) != clock_seconds(r['Time_fixed']) for r in accidents),
                       'report_clock_adjustment_seconds': dict(Counter((clock_seconds(r['Time_fixed']) - clock_seconds(r['Time_orig'])) % 86400 for r in accidents))}
    profiles = {g: Profile() for g in ['all_accident', 'all_control', 'best_accident', 'best_control']}
    records, hashes, traffic_hashes, families = [], defaultdict(list), defaultdict(list), defaultdict(list)
    dates_delta = Counter()
    parent_keys = defaultdict(list)
    subset_stats = Counter()
    dimensions, offsets, steps, schema_counts, flags_total = Counter(), Counter(), Counter(), Counter(), Counter()
    nonnumeric = Counter()
    with ZipFile(root / 'annotatedData.zip') as z:
        files = [i.filename for i in preflight(z) if i.filename.endswith('.csv')]
        all_names = sorted(n for n in files if '/allData/' in n)
        best_names = {n.replace('/bestData/', '/allData/'): n for n in files if '/bestData/' in n}
        subset_stats['best_not_in_all'] = len(set(best_names) - set(all_names))
        for number, name in enumerate(all_names):
            data = z.read(name)
            best = name in best_names
            if best:
                equal = data == z.read(best_names[name])
                subset_stats['byte_identical_best_to_all' if equal else 'different_best_to_all'] += 1
                if not equal:
                    raise ValueError('bestData differs; separate profiling needed')
            header, rows = table(data)
            schema_counts[digest(json.dumps(header).encode())] += 1
            if header != EXPECTED:
                raise ValueError('Unexpected traffic schema')
            if not rows:
                raise ValueError('Empty event window')
            fn = PurePosixPath(name).name
            match = re.fullmatch(r'(\d{4}-\d{2}-\d{2})-(\d{2})(\d{2})-(.+)\.csv', fn)
            if not match:
                raise ValueError('Unexpected event filename')
            date, hh, mm, center = match.groups()
            date_value = datetime.strptime(date, '%Y-%m-%d')
            report_clock = int(hh) * 3600 + int(mm) * 60
            rel = relative_times([r[7] for r in rows], report_clock)
            times = rel + date_value.toordinal() * 86400 + report_clock
            label = int('/accident/' in name)
            ids = neighbourhood(aliases.get(center), topo)
            raw = np.array([r[10:] for r in rows], dtype=object).reshape(-1, 11, 3)
            empty = raw == ''
            a = np.empty(raw.shape, dtype=np.float64)
            for index, value in np.ndenumerate(raw):
                try:
                    a[index] = float(value) if value.strip() else np.nan
                except ValueError:
                    nonnumeric[digest(value.encode())[:12]] += 1
                    a[index] = np.nan
            flags = Counter()
            flags['windows_with_nonfinite'] = int(not np.isfinite(a).all())
            flags['windows_non_30s_steps'] = int(not np.all(np.diff(rel) == 30))
            flags['windows_repeated_times'] = int(len(np.unique(times)) != len(times))
            flags['windows_nonmonotonic'] = int(np.any(np.diff(rel) <= 0))
            flags['windows_cross_midnight'] = int(np.any((report_clock + rel < 0) | (report_clock + rel >= 86400)))
            flags['center_not_in_topology'] = int(center not in aliases)
            flags['filename_date_differs_from_row'] = int(any(r[4] != date for r in rows))
            flags['filename_report_time_differs_from_row'] = int(any(clock_seconds(r[5]) != report_clock for r in rows))
            flags['label_differs_from_folder'] = int(any(float(r[0]) != label for r in rows))
            flags['filename_road_mile_differs_from_row'] = int(any(r[1] + f'{float(r[2]):.1f}' != center for r in rows))
            flags['speed_missing_volume_zero_cells'] = int((~np.isfinite(a[:, :, 0]) & (a[:, :, 1] == 0)).sum())
            flags['speed_missing_volume_positive_cells'] = int((~np.isfinite(a[:, :, 0]) & (a[:, :, 1] > 0)).sum())
            flags['all_three_missing_cells'] = int((~np.isfinite(a)).all(axis=2).sum())
            flags['occupancy_gt_100_cells'] = int((a[:, :, 2] > 100).sum())
            flags['volume_gt_1000_cells'] = int((a[:, :, 1] > 1000).sum())
            flags['volume_gt_1million_cells'] = int((a[:, :, 1] > 1e6).sum())
            flags['windows_occupancy_gt_100'] = int(np.any(a[:, :, 2] > 100))
            flags['windows_volume_gt_1000'] = int(np.any(a[:, :, 1] > 1000))
            flags['speed_gt_120_mph_cells'] = int((a[:, :, 0] > 120).sum())
            flags['noninteger_volume_cells'] = int((np.isfinite(a[:, :, 1]) & (np.abs(a[:, :, 1] - np.round(a[:, :, 1])) > 1e-6)).sum())
            flags['mapped_neighbour_duplicate'] = int(len([s for s in ids if s]) != len({s for s in ids if s}))
            flags_total.update(flags)
            dimensions[len(rows)] += 1
            offsets[(int(rel.min()), int(rel.max()))] += 1
            steps.update(int(x) for x in np.diff(rel))
            group = 'accident' if label else 'control'
            profiles['all_' + group].add(a, empty, ids, flags)
            if best:
                profiles['best_' + group].add(a, empty, ids, flags)
            window_hash = digest(name.encode())[:20]
            family = digest(f'{center}|{date_value.weekday()}|{report_clock}'.encode())[:20]
            rowdate = datetime.strptime(rows[0][4], '%Y-%m-%d')
            dates_delta[(date_value - rowdate).days] += 1
            parent_key = digest(f'{center}|{rows[0][4]}|{report_clock}'.encode())[:20]
            parent_keys[parent_key].append(number)
            calendar = 'train_before_march' if date < '2021-03-01' else ('validation_march' if date < '2021-04-01' else 'test_april')
            record = {'window_token': window_hash, 'label': label, 'best': best,
                      'month': date[:7], 'candidate_calendar_split': calendar,
                      'family_token': family, 'rows': len(rows), 'min_offset_seconds': int(rel.min()),
                      'parent_event_token_from_row_date': parent_key,
                      'max_offset_seconds': int(rel.max()), 'history_rows_before_report': int((rel < 0).sum()),
                      'positive_offset_rows': int((rel > 0).sum()), 'mapped_sensor_count': sum(s is not None for s in ids),
                      'finite_fraction': float(np.isfinite(a).mean()),
                      'center_volume_finite_fraction': float(np.isfinite(a[:, 5, 1]).mean()),
                      'all_11_volume_finite': bool(np.isfinite(a[:, :, 1]).all()),
                      'all_11_triplets_finite': bool(np.isfinite(a).all()),
                      **dict(flags), '_ids': ids, '_times': times, '_array': a}
            volume_ok = np.isfinite(a[:, :, 1]) & (a[:, :, 1] >= 0) & (a[:, :, 1] <= 1000)
            observation_ok = volume_ok & np.isfinite(a[:, :, 2]) & (a[:, :, 2] >= 0) & (a[:, :, 2] <= 100)
            observation_ok &= (np.isfinite(a[:, :, 0]) & (a[:, :, 0] >= 0) & (a[:, :, 0] <= 120)) | (a[:, :, 1] == 0)
            for minutes in [5, 10, 15]:
                used = ((rel >= -900) & (rel < 0)) | ((rel > 0) & (rel <= minutes * 60))
                for tag, valid in [('volume_screen', volume_ok), ('three_metric_screen', observation_ok)]:
                    record[f'center_{tag}_{minutes}min'] = bool(valid[used, 5].all())
                    record[f'central3_{tag}_{minutes}min'] = bool(valid[used, 4:7].all())
            record['dst_ambiguous_local_hour'] = bool(date == '2020-11-01' and np.any((report_clock + rel >= 3600) & (report_clock + rel < 7200)))
            record['dst_nonexistent_local_hour'] = bool(date == '2021-03-14' and np.any((report_clock + rel >= 7200) & (report_clock + rel < 10800)))
            records.append(record)
            hashes[digest(data)].append(number)
            traffic_hashes[digest(json.dumps([[r[7]] + r[10:] for r in rows], separators=(',', ':')).encode())].append(number)
            families[family].append(number)
            if (number + 1) % 3000 == 0:
                print(json.dumps({'scanned_allData_windows': number + 1, 'total': len(all_names)}), flush=True)
    print('Auditing shared sensor/time observations', flush=True)
    if flags_total['center_not_in_topology']:
        raise ValueError('Unmapped centers: fix identifier mapping before interpreting overlap')
    overlap = overlap_audit(records)
    groups = {g: [r for r in records if (not g.startswith('best') or r['best']) and r['label'] == int(g.endswith('accident'))] for g in profiles}
    coverage = {}
    for g, selected in groups.items():
        coverage[g] = {'all_11_volume_complete_windows': sum(r['all_11_volume_finite'] for r in selected),
                       'all_11_triplets_complete_windows': sum(r['all_11_triplets_finite'] for r in selected),
                       'center_volume_complete_windows': sum(r['center_volume_finite_fraction'] == 1 for r in selected),
                       'month_counts': dict(Counter(r['month'] for r in selected)),
                       'candidate_calendar_counts': dict(Counter(r['candidate_calendar_split'] for r in selected)),
                       'history_rows_distribution': dict(Counter(r['history_rows_before_report'] for r in selected)),
                       'positive_offset_rows_distribution': dict(Counter(r['positive_offset_rows'] for r in selected))}
        coverage[g]['screened_window_counts_not_cleaned_data'] = {k: sum(r[k] for r in selected)
            for k in records[0] if '_screen_' in k}
        coverage[g]['screened_calendar_15min_central3_triplets'] = dict(Counter(r['candidate_calendar_split'] for r in selected if r['central3_three_metric_screen_15min']))
        coverage[g]['dst_ambiguous_windows'] = sum(r['dst_ambiguous_local_hour'] for r in selected)
        coverage[g]['dst_nonexistent_windows'] = sum(r['dst_nonexistent_local_hour'] for r in selected)
    report = {'status': 'DATA_AUDIT_COMPLETE_NOT_TRAINING_READY', 'sources': sources,
              'runtime': {'python': platform.python_version(), 'numpy': np.__version__, 'script_sha256': script_sha256},
              'scope': {'allData_scanned': len(records), 'bestData_compared': len(best_names),
                        'traffic_truncated': False, 'imputation': False, 'training': False,
                        'distribution_sampling': 'only robust quantiles/MAD/IQR sample every 19th value per window/hop-flattened metric; counts/mean/SD exact'},
              'metadata': metadata_report, 'subset': dict(subset_stats),
              'window_rows_histogram': dict(dimensions), 'window_offsets_histogram': {f'{a}:{b}': c for (a, b), c in offsets.items()},
              'time_steps_histogram_seconds': dict(steps), 'schema_hash_counts': dict(schema_counts),
              'nonnumeric_token_hash_counts': dict(nonnumeric), 'flags': dict(flags_total),
              'filename_date_minus_row_date_days_histogram': dict(dates_delta),
              'parent_link_from_row_date': {'parent_groups': len(parent_keys),
                    'control_windows_with_exactly_one_published_accident_parent': sum(sum(not records[i]['label'] for i in v) for v in parent_keys.values() if sum(records[i]['label'] for i in v) == 1),
                    'control_windows_without_published_accident_parent': sum(len(v) for v in parent_keys.values() if not any(records[i]['label'] for i in v)),
                    'groups_with_multiple_accident_parents': sum(sum(records[i]['label'] for i in v) > 1 for v in parent_keys.values())},
              'profiles': {g: p.result() for g, p in profiles.items()}, 'coverage': coverage,
              'duplicates': {'byte_identical_groups': sum(len(v) > 1 for v in hashes.values()),
                             'traffic_and_clock_identical_groups': sum(len(v) > 1 for v in traffic_hashes.values()),
                             'traffic_and_clock_identical_windows': sum(len(v) for v in traffic_hashes.values() if len(v) > 1),
                             'note': 'Traffic hash omits calendar date; repeated sequences alone do not prove shared observation.'},
              'matching_families': {'definition': 'center sensor + weekday + report clock; inferred, no published parent event ID',
                                   'families': len(families),
                                   'with_multiple_accident_windows': sum(sum(records[i]['label'] for i in v) > 1 for v in families.values()),
                                   'control_only_families': sum(not any(records[i]['label'] for i in v) for v in families.values()),
                                   'spanning_candidate_calendar_splits': sum(len({records[i]['candidate_calendar_split'] for i in v}) > 1 for v in families.values())},
              'overlap': overlap,
              'limitations': ['No incident onset truth or first-report field availability certification.',
                              'Screen thresholds volume<=1000 vehicles/30s and speed<=120mph are exploratory plausibility screens, not certified limits; no rows removed.',
                              'Clock labels have no timezone/fold metadata; nearest-day midnight unwrapping is explicit.',
                              'Topology does not certify ramp-free boundaries, capacity, lane aggregation, or queues.',
                              'Calendar split is a coverage diagnostic, not a finalized training protocol.',
                              'No causal, forecasting, or physical-module benefit was evaluated.']}
    out.mkdir(parents=True)
    (out / 'audit_summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    public_records = [{k: v for k, v in r.items() if not k.startswith('_')} for r in records]
    with (out / 'window_audit.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(public_records[0]))
        writer.writeheader()
        writer.writerows(public_records)
    print(json.dumps({'status': report['status'], 'windows': len(records), 'best_windows': len(best_names), 'overlap': overlap}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
