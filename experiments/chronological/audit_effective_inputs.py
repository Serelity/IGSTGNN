"""v13a: fit-only input inventory and label-free singleton-ICSF equivalence."""

import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
PROTOCOL = Path(__file__).with_name('effective_input_audit_v13a.json')
PROTOCOL_SHA256 = '71aa29cde966505b4359b406cbc9f9db6826c95dc6bd9d52f08f4043a043d0ef'
CONTEXT_FIELDS = ('report_age_minutes', 'forecast_tod', 'forecast_dow', 'distances')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    temporary = Path(str(path) + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_protocol(path=PROTOCOL):
    if sha256(path) != PROTOCOL_SHA256:
        raise ValueError('Frozen v13a protocol fingerprint changed')
    return json.loads(Path(path).read_text())


def verify_inputs(directory, protocol, checkpoint=None):
    """Only train files are hashed; whole-file bytes include unparsed future slots."""
    hashes = {}

    def checked(path, expected):
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f'Input fingerprint mismatch: {path}')
        hashes[str(Path(path).resolve())] = actual

    baseline_path = PROTOCOL.with_name(protocol['baseline_protocol'])
    checked(baseline_path, protocol['baseline_protocol_sha256'])
    baseline = json.loads(baseline_path.read_text())
    for name, expected in protocol['model_source_sha256'].items():
        checked(REPO / name, expected)
    directory = Path(directory)
    for name, key in [('summary.json', 'summary_sha256'), ('context_manifest.json', 'context_manifest_sha256')]:
        checked(directory / name, baseline['positive_package'][key])
    summary = json.loads((directory / 'summary.json').read_text())
    context = json.loads((directory / 'context_manifest.json').read_text())
    if not summary['build_complete'] or context['schema'] != 'report_location_v1':
        raise ValueError('Incomplete or incompatible input package')
    for name in ('train_flow.npy', 'train_manifest.csv', 'station_ids.npy', 'scaler.json'):
        checked(directory / name, summary['files'][name])
    for name in ('train_context.npz', 'adjacency.npy'):
        checked(directory / name, context['outputs'][name])
    if checkpoint is not None:
        checked(checkpoint, baseline['checkpoint']['best_model_sha256'])
    return hashes


class FitInputs:
    """Expose X only: never call the training dataset's Y-producing __getitem__."""

    def __init__(self, directory, protocol):
        self.directory = Path(directory)
        with (self.directory / 'train_manifest.csv').open(encoding='utf-8-sig', newline='') as stream:
            self.rows = list(csv.DictReader(stream))
        self.stations = np.load(self.directory / 'station_ids.npy', allow_pickle=False)
        self.flow = np.load(self.directory / 'train_flow.npy', mmap_mode='r', allow_pickle=False)
        with np.load(self.directory / 'train_context.npz', allow_pickle=False) as stored:
            self.context = {name: stored[name] for name in ('sample_indices', 'station_ids', *CONTEXT_FIELDS)}
        self.scaler = json.loads((self.directory / 'scaler.json').read_text())
        ids = [int(row['sample_index']) for row in self.rows]
        n, nodes = len(ids), len(self.stations)
        if (n != protocol['expected_train_samples'] or nodes != protocol['expected_nodes']
                or len(set(ids)) != n or len(np.unique(self.stations)) != nodes
                or self.flow.shape != (n, 26, nodes)):
            raise ValueError('Unexpected source count, duplicate identity or flow shape')
        if (not np.array_equal(self.context['sample_indices'], ids)
                or not np.array_equal(self.context['station_ids'], self.stations)
                or self.context['distances'].shape != (n, nodes, 3)):
            raise ValueError('Context sample/station order or spatial shape differs')
        if any(self.context[name].shape != (n,) for name in CONTEXT_FIELDS[:-1]):
            raise ValueError('Expected one report context per sample')
        if (self.scaler.get('source_version') != 8
                or self.scaler.get('fit_scope') != 'train_X_0:12_unique_station_nominal_slot_finite_nonnegative'
                or self.scaler['fitted_sample_indices'] != ids
                or not np.array_equal(self.scaler['station_ids'], self.stations)):
            raise ValueError('Old-A scaler provenance or station order differs')
        fill = np.asarray(self.scaler['node_fill_mean'])
        if (fill.shape != (nodes,) or not np.isfinite(fill).all()
                or not np.isfinite([self.scaler['mean'], self.scaler['std']]).all()
                or self.scaler['std'] <= 0):
            raise ValueError('Invalid original scaler')
        start, end = map(datetime.fromisoformat, protocol['fit_period'])
        self.indices, self.eligibility = [], []
        for i, row in enumerate(self.rows):
            if row['split'] != 'train' or int(row['source_version']) != 8:
                raise ValueError('Only v8 train manifest rows are allowed')
            stamps = {name: datetime.fromisoformat(row[name]) for name in (
                'report_time', 't0', 'x_start', 'x_end', 'y_start', 'y_end', 'support_start', 'support_end_exclusive')}
            if any(value.tzinfo is not None for value in stamps.values()):
                raise ValueError('Expected nominal naive calendar, not a timezone conversion')
            t0 = stamps['t0']
            expected = {'x_start': -65, 'x_end': -10, 'y_start': 5, 'y_end': 60}
            if (t0.second or t0.microsecond or t0.minute % 5
                    or any(stamps[key] != t0 + timedelta(minutes=value) for key, value in expected.items())
                    or not stamps['support_start'] <= stamps['x_start'] < stamps['y_end'] < stamps['support_end_exclusive']):
                raise ValueError('Manifest input/target clock or complete support is inconsistent')
            retained = start <= stamps['support_start'] < stamps['support_end_exclusive'] <= end and start <= t0 < end
            if retained:
                self.indices.append(i)
                age = (t0 - stamps['report_time']).total_seconds() / 60
                if (not 0 < age <= 5 or not np.isfinite(self.context['report_age_minutes'][i])
                        or abs(float(self.context['report_age_minutes'][i]) - age) > 1e-5
                        or self.context['forecast_tod'][i] != t0.hour * 12 + t0.minute // 5
                        or self.context['forecast_dow'][i] != (t0.weekday() + 1) % 7):
                    raise ValueError('Fit report age/forecast clock disagrees with manifest')
                if not np.isfinite(self.context['distances'][i]).all():
                    raise ValueError('Nonfinite fit distances')
            self.eligibility.append({**row, 'included_in_fit': retained,
                                     'reason': 'eligible' if retained else 'outside_complete_fit_support'})
        if len(self.indices) != protocol['expected_fit_samples']:
            raise ValueError('Fit eligibility count differs from the frozen time-only plan')
        self.fit_index_set = frozenset(self.indices)

    def history(self, indices):
        # The slice is applied at the source: no target/gap values are materialized.
        if not len(indices) or any(not isinstance(i, (int, np.integer)) or isinstance(i, (bool, np.bool_))
                                   or i not in self.fit_index_set for i in indices):
            raise ValueError('History access requires eligible fit indices')
        return np.asarray(self.flow[np.asarray(indices), :12, :])

    def model_batch(self, indices, device):
        import torch
        history = self.history(indices)
        valid = np.isfinite(history) & (history >= 0)
        x = np.empty((*history.shape, 3), dtype=np.float32)
        x[..., 0] = (np.where(valid, history, self.scaler['node_fill_mean'])
                     - self.scaler['mean']) / self.scaler['std']
        for local, index in enumerate(indices):
            start = datetime.fromisoformat(self.rows[index]['x_start'])
            for step in range(12):
                stamp = start + timedelta(minutes=step * 5)
                x[local, step, :, 1] = (stamp.hour * 12 + stamp.minute // 5) / 288
                x[local, step, :, 2] = ((stamp.weekday() + 1) % 7) / 7
        incident = {name: torch.as_tensor(self.context[name][indices], device=device) for name in CONTEXT_FIELDS}
        for name in ('forecast_tod', 'forecast_dow'):
            incident[name] = incident[name].long()
        return torch.as_tensor(x, device=device), incident


class NumericInventory:
    def __init__(self, cap=4096):
        self.cap = cap
        self.count = self.valid = self.zeros = self.nonfinite = self.negative = 0
        self.total = self.squared = 0.
        self.minimum, self.maximum = float('inf'), float('-inf')
        self.values = set()
        self.capped = False

    def update(self, values):
        values = np.asarray(values)
        finite = np.isfinite(values)
        valid = finite & (values >= 0)
        self.count += values.size
        self.nonfinite += int((~finite).sum())
        self.negative += int((finite & (values < 0)).sum())
        selected = values[valid].astype(np.float64)
        self.valid += selected.size
        self.zeros += int((selected == 0).sum())
        if selected.size:
            self.minimum = min(self.minimum, float(selected.min()))
            self.maximum = max(self.maximum, float(selected.max()))
            self.total += float(selected.sum())
            self.squared += float(np.square(selected).sum())
            if not self.capped:
                self.values.update(np.unique(selected).tolist())
                if len(self.values) > self.cap:
                    self.values.clear()
                    self.capped = True

    def result(self):
        mean = self.total / self.valid if self.valid else None
        return {'count': int(self.count), 'valid_count': int(self.valid),
                'nonfinite_count': self.nonfinite, 'negative_count': self.negative,
                'invalid_fraction': 1 - self.valid / self.count if self.count else None,
                'zero_count': self.zeros, 'zero_fraction_valid': self.zeros / self.valid if self.valid else None,
                'min': self.minimum if self.valid else None, 'max': self.maximum if self.valid else None,
                'mean': mean, 'std': max(0., self.squared / self.valid - mean * mean) ** .5 if self.valid else None,
                'constant_on_valid': self.minimum == self.maximum if self.valid else None,
                'unique_count': len(self.values) if not self.capped else None,
                'unique_count_lower_bound': self.cap + 1 if self.capped else len(self.values)}


def inventory(data, indices, protocol):
    statistics = {name: NumericInventory(protocol['unique_value_cap']) for name in (
        'history_flow', 'history_tod', 'history_dow', 'report_age_minutes', 'forecast_tod', 'forecast_dow',
        'D0', 'D1', 'D2', 'D1_supported', 'D2_supported', 'candidate_nodes_per_window')}
    node_valid = np.zeros(len(data.stations), dtype=np.int64)
    node_zero = node_valid.copy()
    for offset in range(0, len(indices), protocol['batch_size']):
        batch = indices[offset:offset + protocol['batch_size']]
        history = data.history(batch)
        valid = np.isfinite(history) & (history >= 0)
        statistics['history_flow'].update(history)
        node_valid += valid.sum(axis=(0, 1))
        node_zero += ((history == 0) & valid).sum(axis=(0, 1))
        distance = data.context['distances'][batch]
        support = np.abs(distance).sum(-1) > 0
        statistics['candidate_nodes_per_window'].update(support.sum(-1))
        for channel in range(3):
            statistics[f'D{channel}'].update(distance[..., channel])
        for channel in (1, 2):
            statistics[f'D{channel}_supported'].update(distance[..., channel][support])
        for name in CONTEXT_FIELDS[:-1]:
            statistics[name].update(data.context[name][batch])
        for index in batch:
            start = datetime.fromisoformat(data.rows[index]['x_start'])
            stamps = [start + timedelta(minutes=5 * h) for h in range(12)]
            statistics['history_tod'].update([s.hour * 12 + s.minute // 5 for s in stamps])
            statistics['history_dow'].update([(s.weekday() + 1) % 7 for s in stamps])
    values = {name: item.result() for name, item in statistics.items()}
    meanings = {
        'history_flow': ('backbone; history-derived adapter features', 'Observed under nominal-calendar/latency assumptions', 'Window-exposure weighted raw X; overlapping slots count again; not a new scaler'),
        'history_tod': ('backbone shared clock embedding', 'Derived from X timestamps', 'Integer 5-minute bins; not repeated over the node axis in this inventory'),
        'history_dow': ('backbone shared clock embedding', 'Derived from X timestamps', 'Sunday=0; not repeated over the node axis'),
        'report_age_minutes': ('report encoder -> V and K; some adapters', 'Nominal report timestamp; no first-report snapshot', 'T-report_time; variation is not predictive gain'),
        'forecast_tod': ('report encoder -> V and K', 'Known forecast timestamp', 'Deterministic from the last X label time plus 10 minutes; not independent incident semantics'),
        'forecast_dow': ('report encoder -> V and K', 'Known forecast timestamp', 'Same calendar as history with possible day rollover'),
        'D0': ('ICSF support; TIID context; some adapters', 'Constructed by report_location_v1', 'Deliberate constant-zero channel'),
        'D1': ('ICSF support; TIID numeric context; some adapters', 'Location-at-first-report is conditional', 'Proximity numeric value cannot modulate singleton ICSF attention'),
        'D2': ('ICSF support; TIID numeric context; some adapters', 'Location-at-first-report is conditional', 'Mile ordering, not independently certified physical upstream/downstream'),
        'D1_supported': ('diagnostic subset of D1', 'Same as D1', 'Avoid dilution by disconnected zero values'),
        'D2_supported': ('diagnostic subset of D2', 'Same as D2', 'Avoid dilution by disconnected zero values'),
        'candidate_nodes_per_window': ('ICSF/TIID mask; evaluation support', 'Derived from nonzero D', 'Not an independent feature; pure no-report model must not receive it'),
    }
    fields = [{'field': name, 'consumers': meanings[name][0], 'availability': meanings[name][1],
               'interpretation': meanings[name][2], **value} for name, value in values.items()]
    for name, reason in (
        ('static_sensor_attributes', 'Disabled in current chronological A; no active contribution'),
        ('incident_type_description_duration', 'Excluded by report_location_v1; not represented by V'),
        ('speed_occupancy', 'Not in this flow input package; no new channel effectiveness claim'),
        ('ICSF_q_and_fusion_weights', 'Singleton attention; bypass tested separately; K remains for TIID'),
    ):
        fields.append({'field': name, 'consumers': 'not active in this information path',
                       'availability': 'not measured as an input here', 'interpretation': reason})
    total_per_node = len(indices) * 12
    nodes = [{'station_id': int(station), 'input_cells': total_per_node,
              'valid_cells': int(node_valid[i]), 'invalid_cells': int(total_per_node - node_valid[i]),
              'true_zero_cells': int(node_zero[i]),
              'invalid_fraction': float(1 - node_valid[i] / total_per_node)}
             for i, station in enumerate(data.stations)]
    return fields, nodes, values


def state_hash(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(f'{name}:{value.dtype}:{tuple(value.shape)}'.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def equivalence(data, indices, protocol, checkpoint, device_name, progress):
    import torch
    from experiments.chronological.smoke import make_model, set_seed
    from experiments.chronological.train import configure_determinism
    from src.models.single_incident_icsf import compare_single_incident
    device = torch.device(device_name)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; activate igstgnn on the allocated GPU node')
    configure_determinism(device)
    torch.set_num_threads(max(1, int(os.environ.get('OMP_NUM_THREADS', '3'))))
    set_seed(protocol['seed'])
    model = make_model(data.directory, len(data.stations), device, 'fixed')
    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
    model.eval().requires_grad_(False)
    before = state_hash(model)
    results = {}
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    for offset in range(0, len(indices), protocol['batch_size']):
        batch = indices[offset:offset + protocol['batch_size']]
        x, incident = data.model_batch(batch, device)
        observed = compare_single_incident(model, x, incident,
            atol=protocol['equivalence']['atol_standardized'], rtol=protocol['equivalence']['rtol'])
        for name, value in observed.items():
            aggregate = results.setdefault(name, {'cells': 0, 'abs_sum': 0., 'abs_max': 0., 'exactly_equal': True})
            aggregate['cells'] += value['cells']
            aggregate['abs_sum'] += value['abs_sum']
            aggregate['abs_max'] = max(aggregate['abs_max'], value['abs_max'])
            aggregate['exactly_equal'] &= value['exactly_equal']
        if offset // protocol['batch_size'] % 5 == 0 or offset + len(batch) == len(indices):
            progress('equivalence', completed=offset + len(batch), total=len(indices))
    after = state_hash(model)
    if before != after or any(p.grad is not None for p in model.parameters()):
        raise ValueError('Frozen model state changed or gradients were computed')
    return {'status': 'EQUIVALENCE_PASS', 'weight_identity': 'frozen_A' if checkpoint is not None else 'random_engineering_only',
            'device': str(device), 'device_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU',
            'torch_version': torch.__version__, 'sample_count': len(indices), 'tensors': results,
            'state_sha256_before': before, 'state_sha256_after': after,
            'bypassed_parameter_count': sum(p.numel() for block in (model.icsf_module.q_proj, model.icsf_module.icsf_fusion_mlp) for p in block.parameters()),
            'bypassed_parameters_removed_from_checkpoint': False,
            'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
            'optimizer_steps': 0, 'predictive_gain_measured': False}


def run(data_dir, output, checkpoint=None, device='cuda:0', check=False,
        inventory_only=False, random_init_check=False):
    if random_init_check and (not check or inventory_only or checkpoint is not None):
        raise ValueError('Random initialization is allowed only for a check without checkpoint')
    if not inventory_only and not random_init_check and checkpoint is None:
        raise ValueError('Frozen A checkpoint required for equivalence')
    if inventory_only and checkpoint is not None:
        raise ValueError('Inventory-only does not use a checkpoint')
    output = Path(output)
    partial = Path(str(output) + '.partial')
    if any(p.exists() or p.is_symlink() for p in (output, partial)):
        raise FileExistsError('Preserve existing output/partial; use a new run name')
    protocol = load_protocol()
    partial.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()

    def progress(stage, **fields):
        entry = {'stage': stage, 'elapsed_seconds': round(time.monotonic() - started, 3), **fields}
        write_json(partial / 'progress.json', entry)
        print(json.dumps(entry), flush=True)

    try:
        progress('verifying_train_input_identity')
        hashes = verify_inputs(data_dir, protocol, checkpoint)
        data = FitInputs(data_dir, protocol)
        indices = data.indices[:protocol['check_samples']] if check else data.indices
        progress('inventory', eligible_fit=len(data.indices), measured=len(indices))
        fields, nodes, statistics = inventory(data, indices, protocol)
        write_csv(partial / 'field_inventory.csv', fields)
        write_csv(partial / 'node_input_quality.csv', nodes)
        write_csv(partial / 'eligibility.csv', data.eligibility)
        write_json(partial / 'measured_sample_ids.json', [int(data.rows[i]['sample_index']) for i in indices])
        comparison = None if inventory_only else equivalence(data, indices, protocol, checkpoint, device, progress)
        if check:
            status = 'ENGINEERING_CHECK_PASS_RANDOM_WEIGHTS' if random_init_check else 'ENGINEERING_CHECK_PASS'
        else:
            status = 'FIT_INPUT_INVENTORY_COMPLETE' if inventory_only else 'FIT_INPUT_AND_EQUIVALENCE_COMPLETE'
        commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO, capture_output=True, text=True, check=False)
        sources = [Path(__file__), PROTOCOL, REPO / 'src/models/single_incident_icsf.py',
                   REPO / 'experiments/chronological/smoke.py', REPO / 'experiments/chronological/train.py']
        summary = {'status': status, 'scientific_status': protocol['decision'],
                   'protocol_id': protocol['protocol_id'], 'protocol_sha256': PROTOCOL_SHA256,
                   'host': socket.gethostname(), 'pid': os.getpid(), 'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
                   'python': sys.executable, 'numpy_version': np.__version__, 'commit': commit.stdout.strip(),
                   'finished_utc': datetime.now(timezone.utc).isoformat(), 'elapsed_seconds': time.monotonic() - started,
                   'engineering_check': check, 'eligible_fit_samples': len(data.indices), 'measured_samples': len(indices),
                   'nodes': len(data.stations), 'fit_period': protocol['fit_period'], 'statistics': statistics,
                   'equivalence': comparison, 'information_boundary': protocol['information_boundary'],
                   'input_sha256': hashes,
                   'source_sha256': {str(p.relative_to(REPO)): sha256(p) for p in sources},
                   'artifacts': {p.name: sha256(p) for p in partial.iterdir() if p.name != 'progress.json'}}
        write_json(partial / 'summary.json', summary)
        partial.rename(output)
        report(summary)
        return summary
    except Exception as exc:
        write_json(partial / 'failure.json', {'status': 'FAILED', 'exception': repr(exc), 'traceback': traceback.format_exc()})
        raise


def report(summary):
    print('Status:', summary['status'])
    print('Scientific status:', summary['scientific_status'])
    print('Fit samples eligible/measured:', summary['eligible_fit_samples'], '/', summary['measured_samples'])
    for name in ('history_flow', 'report_age_minutes', 'D0', 'candidate_nodes_per_window'):
        print(name + ':', json.dumps(summary['statistics'][name], ensure_ascii=False))
    result = summary['equivalence']
    print('Equivalence:', json.dumps(result, ensure_ascii=False) if result is not None else 'NOT_RUN_INVENTORY_ONLY')
    print('Y values/loss: not evaluated. Whole train file hashes include unparsed Y bytes. No validation/test files.')
    print('No new model training, speedup measurement or predictive-gain claim.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    execute = commands.add_parser('run')
    execute.add_argument('--data-dir', type=Path, required=True)
    execute.add_argument('--output', type=Path, required=True)
    execute.add_argument('--checkpoint', type=Path)
    execute.add_argument('--device', default='cuda:0')
    execute.add_argument('--check', action='store_true')
    execute.add_argument('--inventory-only', action='store_true')
    execute.add_argument('--random-init-check', action='store_true')
    show = commands.add_parser('report')
    show.add_argument('summary', type=Path)
    args = parser.parse_args()
    if args.action == 'report':
        report(json.loads(args.summary.read_text()))
    else:
        run(args.data_dir, args.output, args.checkpoint, args.device, args.check, args.inventory_only, args.random_init_check)


if __name__ == '__main__':
    main()
