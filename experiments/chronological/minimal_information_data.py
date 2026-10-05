"""Train-only raw package access and a new fit-X-only scaler for v13b."""

import csv
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
COHORTS = ('incident_full', 'incident', 'primary_control', 'secondary_control')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def read_rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def verify_sources(data_dir, primary_dir, secondary_dir, protocol):
    """Hash only declared train files; never open an old scaler/checkpoint/val/test."""
    hashes = {}
    baseline_path = REPO / 'experiments/chronological/incident_branch_materialize_v6a.json'
    require(sha256(baseline_path) == protocol['data_contract_sha256'], 'Data contract changed')
    baseline = json.loads(baseline_path.read_text())

    def checked(group, directory, name, expected):
        actual = sha256(Path(directory) / name)
        require(actual == expected, f'Input fingerprint mismatch: {group}/{name}')
        hashes[f'{group}/{name}'] = actual

    for name, key in (('summary.json', 'summary_sha256'), ('context_manifest.json', 'context_manifest_sha256')):
        checked('positive', data_dir, name, baseline['positive_package'][key])
    meta = json.loads((Path(data_dir) / 'summary.json').read_text())
    context = json.loads((Path(data_dir) / 'context_manifest.json').read_text())
    require(meta['build_complete'] and not meta['test_flow_built'] and context['schema'] == 'report_location_v1',
            'Incomplete or incompatible development package')
    for name in ('train_flow.npy', 'train_manifest.csv', 'station_ids.npy'):
        checked('positive', data_dir, name, meta['files'][name])
    for name in ('train_context.npz', 'adjacency.npy'):
        checked('positive', data_dir, name, context['outputs'][name])
    for group, directory, spec in (('primary', primary_dir, baseline['primary_control_inputs']),
                                   ('secondary', secondary_dir, baseline['secondary_control_inputs'])):
        for name, expected in spec.items():
            if name == 'summary.json' or name.startswith('train_'):
                checked(group, directory, name, expected)
    return hashes


def stamps(row, control=False):
    require(row['split'] == 'train' and int(row['source_version']) == 8, 'Only v8 train rows allowed')
    t = datetime.fromisoformat(row['candidate_t0' if control else 't0'])
    start, end = (datetime.fromisoformat(row[k]) for k in ('support_start', 'support_end_exclusive'))
    require(all(v.tzinfo is None for v in (t, start, end)) and not t.second and not t.microsecond
            and t.minute % 5 == 0, 'Invalid nominal clock')
    for key, minutes in (('x_start', -65), ('x_end', -10), ('y_start', 5), ('y_end', 60),
                         ('support_start', -70), ('support_end_exclusive', 65)):
        require(datetime.fromisoformat(row[key]) == t + timedelta(minutes=minutes), 'Incomplete or inconsistent temporal support')
    return t, start, end


def inside(row, bounds, control=False):
    t, first, last = stamps(row, control)
    start, end = map(datetime.fromisoformat, bounds)
    return start <= first < last <= end and start <= t < end


class DevelopmentData:
    def __init__(self, directory, primary, secondary, protocol):
        directory, primary, secondary = map(Path, (directory, primary, secondary))
        self.protocol = protocol
        self.rows = read_rows(directory / 'train_manifest.csv')
        self.primary_rows = read_rows(primary / 'train_control_manifest.csv')
        self.secondary_rows = read_rows(secondary / 'train_second_control_manifest.csv')
        self.stations = np.load(directory / 'station_ids.npy', allow_pickle=False)
        self.adjacency = np.load(directory / 'adjacency.npy', allow_pickle=False)
        self.flow = np.load(directory / 'train_flow.npy', mmap_mode='r', allow_pickle=False)
        self.primary_flow = np.load(primary / 'train_control_flow.npy', mmap_mode='r', allow_pickle=False)
        self.secondary_flow = np.load(secondary / 'train_second_control_flow.npy', mmap_mode='r', allow_pickle=False)
        with np.load(directory / 'train_context.npz', allow_pickle=False) as data:
            self.context = {key: data[key] for key in ('sample_indices', 'station_ids', 'distances',
                                                     'report_age_minutes', 'forecast_tod', 'forecast_dow')}
        n = len(self.stations)
        require(n == protocol['nodes'] and len(np.unique(self.stations)) == n, 'Station count/order mismatch')
        require(self.adjacency.shape == (n, n) and np.isfinite(self.adjacency).all()
                and (self.adjacency >= 0).all(), 'Invalid adjacency')
        for rows, flow, expected in ((self.rows, self.flow, protocol['source_counts']['positive']),
                (self.primary_rows, self.primary_flow, protocol['source_counts']['primary']),
                (self.secondary_rows, self.secondary_flow, protocol['source_counts']['secondary'])):
            require(len(rows) == expected and flow.shape == (expected, 26, n), 'Source shape mismatch')
        ids = [int(row['sample_index']) for row in self.rows]
        self.positions = {sample: i for i, sample in enumerate(ids)}
        self.primary_by_id = {int(row['positive_sample_index']): i for i, row in enumerate(self.primary_rows)}
        require(len(self.positions) == len(ids) and len(self.primary_by_id) == len(self.primary_rows), 'Duplicate source identity')
        require(np.array_equal(self.context['sample_indices'], ids)
                and np.array_equal(self.context['station_ids'], self.stations), 'Context identity/order mismatch')
        require(self.context['distances'].shape == (len(ids), n, 3)
                and np.isfinite(self.context['distances']).all()
                and np.all(self.context['distances'][..., 0] == 0), 'Distances/D0 outside frozen scope')
        for key in ('report_age_minutes', 'forecast_tod', 'forecast_dow'):
            require(self.context[key].shape == (len(ids),) and np.isfinite(self.context[key]).all(), 'Invalid context vector')
        for i, row in enumerate(self.rows):
            t, _, _ = stamps(row)
            age = (t - datetime.fromisoformat(row['report_time'])).total_seconds() / 60
            require(0 < age <= 5 and abs(self.context['report_age_minutes'][i] - age) < 1e-5,
                    'Report age does not match clock')
            require(self.context['forecast_tod'][i] == t.hour * 12 + t.minute // 5 and
                    self.context['forecast_dow'][i] == (t.weekday() + 1) % 7, 'Report/calendar mismatch')
        masks1 = np.load(primary / 'train_affected_mask.npy', allow_pickle=False)
        masks2 = np.load(secondary / 'train_second_affected_mask.npy', allow_pickle=False)
        require(masks1.shape == (len(self.primary_rows), n) and masks2.shape == (len(self.secondary_rows), n)
                and masks1.dtype == np.bool_ and masks2.dtype == np.bool_, 'Control mask shape/dtype mismatch')
        for i, row in enumerate(self.primary_rows):
            stamps(row, True)
            require(int(row['control_index']) == i, 'Primary order mismatch')
        seen, boundary = set(), []
        for i, row in enumerate(self.secondary_rows):
            stamps(row, True)
            sample = int(row['positive_sample_index'])
            require(sample not in seen and int(row['control_index']) == i and sample in self.positions
                    and sample in self.primary_by_id, 'Matched identity/order mismatch')
            seen.add(sample)
            j, k = self.positions[sample], self.primary_by_id[sample]
            p, c = self.rows[j], self.primary_rows[k]
            require(len({p['incident_id'], c['incident_id'], row['incident_id']}) == 1 and
                    len({p['t0'], c['positive_t0'], row['positive_t0']}) == 1 and
                    np.array_equal(masks1[k], masks2[i]), 'Matched identity or masks disagree')
            connected = np.any(self.context['distances'][j] != 0, -1)
            require(not (connected & ~masks1[k]).any(), 'Model support outside control mask')
            boundary.extend([sample, int(self.stations[node])] for node in np.flatnonzero(masks1[k] & ~connected))
        require(boundary == protocol['frozen_only_candidate_pairs'], 'Candidate boundary changed')
        periods = list(protocol['periods'].values())
        require(all(datetime.fromisoformat(a[1]) < datetime.fromisoformat(b[0]) for a, b in zip(periods, periods[1:])),
                'Periods overlap or lack embargo')
        self.plan = {}
        for phase, bounds in protocol['periods'].items():
            positive = [i for i, row in enumerate(self.rows) if inside(row, bounds)]
            matched = [i for i, row in enumerate(self.secondary_rows)
                if self.positions[int(row['positive_sample_index'])] in positive
                and inside(row, bounds, True)
                and inside(self.primary_rows[self.primary_by_id[int(row['positive_sample_index'])]], bounds, True)]
            self.plan[phase] = {'incident_full': positive}
            if phase != 'fit':
                self.plan[phase].update({cohort: matched for cohort in COHORTS[1:]})
            require({k: len(v) for k, v in self.plan[phase].items()} == protocol['expected_phase_samples'][phase],
                    f'Time-only eligibility changed: {phase}')
        self.audit_unlocked = False
        self.scaler = None

    def build_scaler(self):
        """Each nominal X slot contributes once; no target or later-period values."""
        unique = {}
        for index in self.plan['fit']['incident_full']:
            first = datetime.fromisoformat(self.rows[index]['x_start'])
            for h, values in enumerate(self.flow[index, :12]):
                slot = (first + timedelta(minutes=5 * h)).isoformat()
                if slot in unique:
                    require(np.array_equal(unique[slot], values, equal_nan=True), 'Conflicting duplicate fit X slot')
                else:
                    unique[slot] = np.array(values, dtype=np.float64)
        values = np.stack([unique[k] for k in sorted(unique)])
        valid = np.isfinite(values) & (values >= 0)
        counts = valid.sum(0)
        require((counts > 0).all(), 'No valid fit X for at least one station; no later-data fallback')
        observed = values[valid]
        mean, std = float(observed.mean()), float(observed.std())
        require(np.isfinite(mean) and np.isfinite(std) and std > 0, 'Invalid fit scaler')
        self.scaler = {'scope': 'v13b_unique_fit_X_only', 'period': self.protocol['periods']['fit'],
            'sample_ids': [int(self.rows[i]['sample_index']) for i in self.plan['fit']['incident_full']],
            'station_ids': self.stations.tolist(), 'unique_slots': len(unique),
            'slot_ids_sha256': json_hash(sorted(unique)), 'valid_counts_per_station': counts.tolist(),
            'true_zero_count': int((observed == 0).sum()), 'mean': mean, 'std': std,
            'node_fill_mean': (np.where(valid, values, 0).sum(0) / counts).tolist(),
            'old_scaler_used': False, 'fit_Y_used': False, 'later_numeric_values_used': False}
        return self.scaler

    def reference(self, cohort, index):
        if cohort == 'incident_full':
            return index, index, self.rows[index], self.flow
        row = self.secondary_rows[index]
        sample = int(row['positive_sample_index'])
        positive = self.positions[sample]
        if cohort == 'incident':
            return positive, positive, self.rows[positive], self.flow
        if cohort == 'primary_control':
            primary = self.primary_by_id[sample]
            return positive, primary, self.primary_rows[primary], self.primary_flow
        if cohort == 'secondary_control':
            return positive, index, row, self.secondary_flow
        raise ValueError('Unknown cohort')

    def batch(self, phase, cohort, indices, arm, targets=True):
        require(self.scaler is not None and arm in ('M0', 'M1', 'M2'), 'Missing scaler or unknown arm')
        allowed = self.plan.get(phase, {}).get(cohort, [])
        require(len(indices) > 0 and len(set(indices)) == len(indices) and all(type(i) is int and i in allowed for i in indices),
                'Batch indices outside declared phase/cohort')
        require(phase != 'audit' or self.audit_unlocked, 'Audit locked until all endpoints freeze')
        parts = []
        for index in indices:
            p, source, row, flow = self.reference(cohort, index)
            history = np.asarray(flow[source, :12], dtype=np.float32)
            valid = np.isfinite(history) & (history >= 0)
            x = np.empty((*history.shape, 3), dtype=np.float32)
            x[..., 0] = (np.where(valid, history, self.scaler['node_fill_mean']) - self.scaler['mean']) / self.scaler['std']
            start = datetime.fromisoformat(row['x_start'])
            for h in range(12):
                stamp = start + timedelta(minutes=5 * h)
                x[h, :, 1] = (stamp.hour * 12 + stamp.minute // 5) / 288
                x[h, :, 2] = ((stamp.weekday() + 1) % 7) / 7
            t = datetime.fromisoformat(row['t0' if cohort in ('incident', 'incident_full') else 'candidate_t0'])
            item = {'x': x, 'clock': np.array([t.hour * 12 + t.minute // 5, (t.weekday() + 1) % 7], dtype=np.int64),
                    'sample_id': np.int64(self.rows[p]['sample_index']), 'source_index': np.int64(source)}
            if arm != 'M0':
                item['distances'] = self.context['distances'][p].astype(np.float32)
            if arm == 'M2':
                item['age'] = np.float32(self.context['report_age_minutes'][p])
            if targets:
                target = np.asarray(flow[source, 14:26], dtype=np.float32)
                y_valid = np.isfinite(target) & (target >= 0)
                item.update(y=np.where(y_valid, target, 0), valid=y_valid,
                            candidate=np.any(self.context['distances'][p] != 0, -1))
            parts.append(item)
        return {key: np.stack([item[key] for item in parts]) for key in parts[0]}
