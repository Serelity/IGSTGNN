"""v12g: NumPy-only diagnosis of saved v12f selection and temporal composition."""

import argparse
import csv
from datetime import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import socket
import sys
import time
import traceback

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.chronological import audit_architecture_regions as regions_audit
from experiments.chronological.audit_matched_controls import sha256
from experiments.chronological.state_interaction_trajectory import analyze_history, COHORTS, REGIONS

PROTOCOL = Path(__file__).with_name('state_interaction_transfer_v12g.json')
PROTOCOL_SHA256 = 'cef7b6b53d78bd11b5af547585a9758015af350e09c93feccb2cc84a582af683'
SOURCE_PROTOCOL = PROTOCOL.with_name('incident_state_interaction_v12f.json')
INHERITED_PROTOCOL = PROTOCOL.with_name('incident_strength_gate_v12c.json')
ARMS = ('strength', 'state_vector', 'interaction_vector')
GROUPS = ('incident_full_common', 'incident_full_complement')


def load_protocol():
    if sha256(PROTOCOL) != PROTOCOL_SHA256:
        raise ValueError('Frozen v12g protocol changed')
    return json.loads(PROTOCOL.read_text(encoding='utf-8'))


def artifact_hash(payload):
    return hashlib.sha256(payload).hexdigest()


def close(observed, expected, label, rtol=1e-10, atol=1e-9):
    if not np.allclose(observed, expected, rtol=rtol, atol=atol):
        raise ValueError(f'{label} disagrees')


class Source:
    def __init__(self, root, protocol):
        self.root = Path(root).resolve()
        self.protocol = protocol
        payload = (self.root / 'summary.json').read_bytes()
        self.hashes = {'summary.json': artifact_hash(payload)}
        self.summary = json.loads(payload)
        for path, key in ((SOURCE_PROTOCOL, 'source_protocol_sha256'),
                          (INHERITED_PROTOCOL, 'inherited_protocol_sha256')):
            if sha256(path) != protocol[key]:
                raise ValueError('Local source/inherited protocol changed')
        self.inherited = json.loads(INHERITED_PROTOCOL.read_text())
        frozen = json.loads(SOURCE_PROTOCOL.read_text())
        s = self.summary
        expected = {'status': 'INCIDENT_STATE_INTERACTION_COMPARISON_COMPLETE',
                    'protocol_sha256': protocol['source_protocol_sha256'],
                    'engineering_check': False, 'model_training_performed': True,
                    'validation_arrays_read': False, 'test_split_read': False,
                    'independent_confirmation': False, 'vector_paired_initialization_exact': True}
        if any(s.get(k) != v or type(s.get(k)) is not type(v) for k, v in expected.items()):
            raise ValueError('Require a complete frozen full-budget v12f source')
        if (s['frozen_protocol'] != frozen or s['inherited_protocol'] != self.inherited
                or s['effective_training'] != self.inherited['training']
                or s['phase_samples'] != protocol['source_phase_samples']
                or set(s['runs']) != {str(seed) for seed in protocol['seeds']}):
            raise ValueError('Source protocol, sample budget or seeds changed')
        if s['code_sha256'] != protocol['source_code_sha256']:
            raise ValueError('Source implementation code identity changed')
        for detail in s['runs'].values():
            if set(detail) != set(ARMS):
                raise ValueError('Source arm set changed')
        self.identity = self.read_json('run_identity.json')
        for key in ('protocol_sha256', 'engineering_check', 'inputs', 'code_sha256'):
            if self.identity[key] != s[key]:
                raise ValueError(f'Source run identity disagrees with summary: {key}')
        self.plan = self.read_json('eligibility.json')
        if set(self.plan) != set(self.inherited['periods']):
            raise ValueError('Source period set changed')
        for phase, p in self.plan.items():
            if (p['bounds'] != self.inherited['periods'][phase]
                    or p['indices'] != self.identity['indices'][phase]
                    or {c: len(v) for c, v in p['indices'].items()} != s['phase_samples'][phase]
                    or len(p['positive_ids']) != len(p['indices']['incident_full'])
                    or len(set(p['positive_ids'])) != len(p['positive_ids'])
                    or len(set(p['matched_ids'])) != len(p['matched_ids'])
                    or not set(p['matched_ids']).issubset(p['positive_ids'])):
                raise ValueError(f'Source eligibility or ordering changed: {phase}')
            if phase != 'fit' and any(len(p['matched_ids']) != len(p['indices'][c]) for c in COHORTS[1:]):
                raise ValueError('Matched source budget disagrees')

    def read(self, name):
        path = (self.root / name).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError('Artifact path escapes source')
        payload = path.read_bytes()
        digest = artifact_hash(payload)
        if self.summary['outputs'].get(name) != digest:
            raise ValueError(f'Source artifact hash mismatch: {name}')
        self.hashes[name] = digest
        return payload

    def read_json(self, name):
        return json.loads(self.read(name))


def read_manifest(data_dir, source):
    path = Path(data_dir) / 'train_manifest.csv'
    expected = {value for name, value in source.summary['inputs'].items()
                if Path(name).name == 'train_manifest.csv'}
    if len(expected) != 1 or sha256(path) not in expected:
        raise ValueError('Train manifest fingerprint disagrees with v12f provenance')
    source.hashes[str(path.resolve())] = sha256(path)
    with path.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    ids = [int(row['sample_index']) for row in rows]
    if len(set(ids)) != len(ids) or any(row['split'] != 'train' for row in rows):
        raise ValueError('Duplicate IDs or non-train manifest rows')
    by_id = {sample: row for sample, row in zip(ids, rows)}
    membership = []
    for phase, p in source.plan.items():
        start, end = map(datetime.fromisoformat, p['bounds'])
        if [ids[i] for i in p['indices']['incident_full']] != p['positive_ids']:
            raise ValueError('Full-positive source positions disagree with manifest')
        common = set(p['matched_ids']) if phase != 'fit' else set()
        for sample in p['positive_ids']:
            row = by_id[sample]
            t0 = datetime.fromisoformat(row['t0'])
            support_start = datetime.fromisoformat(row['support_start'])
            support_end = datetime.fromisoformat(row['support_end_exclusive'])
            if not (start <= support_start < support_end <= end and start <= t0 < end):
                raise ValueError('Manifest sample violates original temporal support eligibility')
            iso = t0.isocalendar()
            membership.append({'phase': phase, 'sample_id': sample, 't0': row['t0'],
                               'positive_week': f'{iso.year}-W{iso.week:02d}',
                               'group': 'fit_positive' if phase == 'fit' else
                                        'phase_common' if sample in common else 'phase_complement'})
    return by_id, membership


def read_record(source, name, expected_ids):
    with np.load(io.BytesIO(source.read(name)), allow_pickle=False) as arrays:
        names = arrays['regions'].tolist()
        if len(set(names)) != len(names) or not set(REGIONS).issubset(names):
            raise ValueError('Saved region axis changed')
        columns = [names.index(r) for r in REGIONS]
        result = {key: arrays[key].copy() for key in ('ids', 'source_indices', 'candidate_mask')}
        result.update({key: arrays[key][:, columns].copy() for key in ('errors', 'counts', 'prediction_counts')})
    n = len(expected_ids)
    ids, indices, mask = result['ids'], result['source_indices'], result['candidate_mask']
    errors, counts, predicted = (result[k] for k in ('errors', 'counts', 'prediction_counts'))
    if (ids.dtype.kind not in 'iu' or indices.dtype.kind not in 'iu' or mask.dtype.kind != 'b'
            or ids.shape != (n,) or indices.shape != (n,) or np.any(indices < 0)
            or mask.shape != (n, source.protocol['node_count'])
            or not np.array_equal(ids, expected_ids) or len(set(ids.tolist())) != n):
        raise ValueError(f'Saved sample identities/order/support changed: {name}')
    if (errors.shape != (n, len(REGIONS)) or counts.shape != errors.shape or predicted.shape != errors.shape
            or errors.dtype.kind != 'f' or counts.dtype.kind not in 'iu' or predicted.dtype.kind not in 'iu'
            or not np.isfinite(errors).all() or np.any(errors < 0) or np.any(counts < 0)
            or np.any(predicted < counts) or np.any(np.where(counts == 0, errors, 0.) != 0)):
        raise ValueError(f'Invalid saved error sums or valid counts: {name}')
    if source.protocol['horizon_count'] != 12:
        raise ValueError('Protected early/late partition requires 12 horizons')
    expected_predicted = np.stack((np.full(n, 12 * mask.shape[1]),
                                  mask.sum(1) * 6, mask.sum(1) * 6,
                                  (~mask).sum(1) * 12), axis=-1)
    if not np.array_equal(predicted, expected_predicted):
        raise ValueError('Geometric counts disagree with candidate support')
    for values in (counts, predicted):
        if not np.array_equal(values[:, 0], values[:, 1:].sum(1)):
            raise ValueError('Saved regional counts do not partition all cells')
    close(errors[:, 0], errors[:, 1:].sum(1), 'Regional error partition', atol=1e-7)
    result['regions'] = list(REGIONS)
    return result


def metrics(record):
    cells, errors = record['counts'].sum(0), record['errors'].sum(0)
    return {'samples': len(record['ids']),
            'mae': {r: float(errors[i] / cells[i]) if cells[i] else None for i, r in enumerate(REGIONS)},
            'valid_cell_fraction': {r: float(cells[i] / cells[0]) if cells[0] else 0. for i, r in enumerate(REGIONS)}}


def reconcile_metrics(record, declared, label):
    actual = metrics(record)
    if actual['samples'] != declared['samples']:
        raise ValueError(f'{label} sample budget disagrees')
    for region in REGIONS:
        a, b = actual['mae'][region], declared['mae'][region]
        if (a is None) != (b is None):
            raise ValueError(f'{label} undefined metric disagrees')
        if a is not None:
            close(a, b, f'{label}/{region}')


def aligned(a, b):
    for key in ('ids', 'source_indices', 'candidate_mask', 'counts', 'prediction_counts'):
        if not np.array_equal(a[key], b[key]):
            raise ValueError(f'Saved evaluation alignment mismatch: {key}')


def restrict(record, positions):
    return {key: (value[positions] if isinstance(value, np.ndarray) else value)
            for key, value in record.items()}


def common_replay(full, common, tolerance):
    positions = {int(sample): i for i, sample in enumerate(full['ids'])}
    try:
        selected = restrict(full, [positions[int(sample)] for sample in common['ids']])
    except KeyError as error:
        raise ValueError('Matched positive ID is outside full-positive period') from error
    aligned(selected, common)
    close(selected['errors'], common['errors'], 'Full/common saved positive replay', **tolerance)
    delta = selected['errors'] - common['errors']
    cells = common['counts'].sum(0)
    return {'maximum_absolute_sample_region_error_sum_difference': float(np.abs(delta).max()) if delta.size else 0.,
            'pooled_mae_difference_full_minus_matched': {
                r: float(delta[:, i].sum() / cells[i]) if cells[i] else None
                for i, r in enumerate(REGIONS)}}


def comparisons():
    result = {arm + '_vs_A': ['A', arm] for arm in ARMS}
    result.update({arm + '_vs_strength': ['strength', arm] for arm in ARMS[1:]})
    result['interaction_vector_vs_state_vector'] = ['state_vector', 'interaction_vector']
    return result


def audit_records(reference, learned, tolerance):
    records, replay = {}, {}
    for cohort in COHORTS:
        ordered = [reference[cohort]] + [learned[arm][cohort] for arm in ARMS]
        for other in ordered[1:]:
            aligned(ordered[0], other)
        records[cohort] = {**{k: ordered[0][k] for k in ('ids', 'source_indices', 'candidate_mask',
                                  'counts', 'prediction_counts', 'regions')},
                           'errors': np.stack([r['errors'] for r in ordered], axis=1)}
    for cohort in COHORTS[2:]:
        for key in ('ids', 'candidate_mask', 'prediction_counts'):
            if not np.array_equal(records[cohort][key], records['incident'][key]):
                raise ValueError('Matched control identity/geometric support changed')
    replay['A'] = common_replay(reference['incident_full'], reference['incident'], tolerance)
    for arm in ARMS:
        replay[arm] = common_replay(learned[arm]['incident_full'], learned[arm]['incident'], tolerance)
        replay[arm]['gain_difference_due_to_replay_raw_mae'] = {
            r: (replay['A']['pooled_mae_difference_full_minus_matched'][r]
                - replay[arm]['pooled_mae_difference_full_minus_matched'][r])
            if replay['A']['pooled_mae_difference_full_minus_matched'][r] is not None else None for r in REGIONS}
    common_ids = set(records['incident']['ids'].tolist())
    mask = np.asarray([int(sample) in common_ids for sample in records['incident_full']['ids']])
    records[GROUPS[0]] = restrict(records['incident_full'], mask)
    records[GROUPS[1]] = restrict(records['incident_full'], ~mask)
    for key in ('errors', 'counts', 'prediction_counts'):
        close(records['incident_full'][key].sum(0),
              sum(records[group][key].sum(0) for group in GROUPS), 'Common/complement sufficient-statistic partition', atol=1e-7)
    return records, replay


def partition_accounting(records):
    full = records['incident_full']
    total = int(full['counts'][:, 0].sum())
    paths = ['A', *ARMS]
    result = {}
    for comparison, (left, right) in comparisons().items():
        li, ri = paths.index(left), paths.index(right)
        full_gain = float((full['errors'][:, li, 0] - full['errors'][:, ri, 0]).sum() / total)
        groups = {}
        for group in GROUPS:
            r = records[group]
            gain_sums = (r['errors'][:, li] - r['errors'][:, ri]).sum(0)
            contributions = {region: float(gain_sums[i] / total) for i, region in enumerate(REGIONS)}
            close(contributions['all'], sum(contributions[r] for r in REGIONS[1:]), 'Group regional contributions')
            groups[group] = {'samples': len(r['ids']), 'valid_cell_share_of_full_global': float(r['counts'][:, 0].sum() / total),
                             'contribution_to_full_global_gain': contributions}
        reconstructed = sum(value['contribution_to_full_global_gain']['all'] for value in groups.values())
        close(full_gain, reconstructed, 'Group global gain contributions')
        result[comparison] = {'full_global_gain_raw_mae': full_gain,
                              'reconstructed_full_global_gain_raw_mae': reconstructed, 'groups': groups}
    return result


def weekly_rows(records, times, seed):
    weeks, positions = regions_audit.week_grid(times)
    paths = ['A', *ARMS]
    totals, cells, samples, window_mae_sums, evaluable_windows = {}, {}, {}, {}, {}
    for cohort, record in records.items():
        index = np.asarray([positions[int(s)] for s in record['ids']], dtype=int)
        totals[cohort] = np.zeros((len(weeks), len(paths), len(REGIONS)))
        cells[cohort] = np.zeros((len(weeks), len(REGIONS)), dtype=np.int64)
        samples[cohort] = np.bincount(index, minlength=len(weeks))
        window_mae_sums[cohort] = np.zeros_like(totals[cohort])
        evaluable_windows[cohort] = np.zeros_like(cells[cohort])
        np.add.at(totals[cohort], index, record['errors'])
        np.add.at(cells[cohort], index, record['counts'])
        window_mae = regions_audit.ratio(record['errors'], record['counts'][:, None, :])
        np.add.at(window_mae_sums[cohort], index, np.nan_to_num(window_mae, nan=0.))
        np.add.at(evaluable_windows[cohort], index, record['counts'] > 0)
    rows = []
    for cohort in records:
        for w, week in enumerate(weeks):
            for i, region in enumerate(REGIONS):
                n = int(cells[cohort][w, i])
                nw = int(evaluable_windows[cohort][w, i])
                a = float(totals[cohort][w, 0, i] / n) if n else None
                window_a = float(window_mae_sums[cohort][w, 0, i] / nw) if nw else None
                full_n = int(cells['incident_full'][w, 0])
                for p, arm in enumerate(ARMS, 1):
                    gain_sum = float(totals[cohort][w, 0, i] - totals[cohort][w, p, i])
                    window_b = float(window_mae_sums[cohort][w, p, i] / nw) if nw else None
                    rows.append({'seed': seed, 'cohort': cohort, 'positive_week': week, 'region': region,
                        'arm': arm, 'forecast_windows': int(samples[cohort][w]), 'valid_cells': n,
                        'mae_A': a, 'mae': float(totals[cohort][w, p, i] / n) if n else None,
                        'gain_vs_A_raw_mae': gain_sum / n if n else None,
                        'evaluable_forecast_windows': nw, 'equal_forecast_window_mae_A': window_a,
                        'equal_forecast_window_mae': window_b,
                        'equal_forecast_window_gain_vs_A_raw_mae': window_a - window_b if nw else None,
                        'full_global_valid_cells_this_week': full_n,
                        'contribution_to_full_global_gain': gain_sum / full_n if full_n and cohort in ('incident_full', *GROUPS) else None})
    for w in range(len(weeks)):
        if not np.array_equal(cells['incident_full'][w], sum(cells[g][w] for g in GROUPS)):
            raise ValueError('Weekly group valid-count partition disagrees')
        for p in range(len(paths)):
            close(totals['incident_full'][w, p], sum(totals[g][w, p] for g in GROUPS), 'Weekly group error partition', atol=1e-7)
    return rows


def write_csv(path, rows):
    if not rows:
        raise ValueError('Cannot publish an empty diagnostic CSV')
    with Path(path).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(source_dir, data_dir, output):
    protocol = load_protocol()
    output, source_root = Path(output).resolve(), Path(source_dir).resolve()
    partial = output.with_name(output.name + '.partial')
    if output.is_relative_to(source_root) or partial.is_relative_to(source_root):
        raise ValueError('Write diagnostic output outside the read-only v12f source')
    if output.exists() or output.is_symlink() or partial.exists() or partial.is_symlink():
        raise FileExistsError('Preserve output/partial; choose a new diagnostic run')
    partial.mkdir(parents=True)
    started = time.monotonic()
    def progress(stage, **fields):
        event = {'stage': stage, 'elapsed_seconds': time.monotonic() - started, **fields}
        regions_audit.write_json(partial / 'progress.json', event)
        print(json.dumps(event), flush=True)
    try:
        progress('verifying_saved_source')
        source = Source(source_root, protocol)
        manifest, membership = read_manifest(data_dir, source)
        baseline, reference = {}, {}
        for phase in ('selection', 'audit'):
            reference[phase] = {}
            for cohort in COHORTS:
                expected_ids = source.plan[phase]['positive_ids' if cohort == 'incident_full' else 'matched_ids']
                record = read_record(source, f'{phase}_A_{cohort}.npz', expected_ids)
                # Matched plans index secondary rows; saved indices name each cohort's own source.
                if cohort in ('incident_full', 'secondary_control'):
                    expected_indices = source.plan[phase]['indices'][cohort]
                elif cohort == 'incident':
                    positions = {sample: i for i, sample in enumerate(manifest)}
                    expected_indices = [positions[sample] for sample in expected_ids]
                else:
                    expected_indices = None
                if expected_indices is not None and not np.array_equal(record['source_indices'], expected_indices):
                    raise ValueError(f'{phase}/{cohort} source indices disagree with original source mapping')
                reference[phase][cohort] = record
                if phase == 'selection':
                    baseline[cohort] = metrics(record)
                else:
                    reconcile_metrics(record, source.summary['baseline_audit'][cohort], f'baseline audit/{cohort}')
            common_replay(reference[phase]['incident_full'], reference[phase]['incident'], protocol['matched_replay_error_tolerance'])
        times = {s: manifest[s]['t0'] for s in source.plan['audit']['positive_ids']}
        probe_indices = source.identity['probe_indices']
        fit = source.plan['fit']['indices']['incident_full']
        expected_probe = [fit[i] for i in np.linspace(0, len(fit) - 1,
            min(len(fit), source.summary['frozen_protocol']['representation_probe']['samples']), dtype=int)]
        if probe_indices != expected_probe:
            raise ValueError('Fit probe no longer follows frozen manifest-only selection')
        # Probe order is the original fit-index order, not numerical sample-ID order.
        ordered_manifest_ids = list(manifest)
        probe_ids = [ordered_manifest_ids[i] for i in probe_indices]
        expected_steps = (len(fit) + source.inherited['training']['batch_size'] - 1) // source.inherited['training']['batch_size']
        trajectories, analyses, accounting, probes, replays = {}, {}, {}, {}, {}
        epoch_rows, comparison_rows, week_rows = [], [], []
        initial_probe = None
        for seed in protocol['seeds']:
            learned = {}
            trajectories[str(seed)], probes[str(seed)] = {}, {}
            for arm in ARMS:
                name = f'{arm}_s{seed}'
                detail = source.summary['runs'][str(seed)][arm]
                trajectory, rows = analyze_history(source.read_json(f'{name}/history.json'), detail,
                                                   baseline, source.inherited, expected_steps)
                trajectories[str(seed)][arm] = trajectory
                epoch_rows.extend({'seed': seed, 'arm': arm, **row} for row in rows)
                learned[arm] = {}
                for cohort in COHORTS:
                    record = read_record(source, f'{name}/audit_{cohort}.npz',
                                         reference['audit'][cohort]['ids'].tolist())
                    reconcile_metrics(record, detail['audit'][cohort], f'{name} audit/{cohort}')
                    learned[arm][cohort] = record
                snapshots = {}
                probe_reference = None
                for label in ('initial', 'last', 'selected'):
                    record = read_record(source, f'{name}/representation_{label}.npz', probe_ids)
                    declared = detail['representation_diagnostics'][label]
                    reconcile_metrics(record, declared, f'{name} fit probe/{label}')
                    if declared['sample_ids'] != probe_ids or not np.array_equal(record['source_indices'], probe_indices):
                        raise ValueError('Fit probe sample identity disagrees')
                    if label == 'initial':
                        probe_reference = record
                        if initial_probe is None:
                            initial_probe = record
                        aligned(initial_probe, record)
                        close(initial_probe['errors'], record['errors'], 'Initial probes across arms/seeds', rtol=0, atol=0)
                    aligned(probe_reference, record)
                    m = metrics(record)
                    snapshots[label] = {**m, 'gain_vs_initial_probe_raw_mae': {
                        r: metrics(probe_reference)['mae'][r] - m['mae'][r] if m['mae'][r] is not None else None
                        for r in REGIONS}, 'candidate_representation': declared['candidate_representation']}
                probes[str(seed)][arm] = snapshots
                progress('saved_arm_verified', seed=seed, arm=arm, selected_epoch=trajectory['selected_epoch'],
                         eligible_epochs=trajectory['eligible_epochs'])
            records, replay = audit_records(reference['audit'], learned, protocol['matched_replay_error_tolerance'])
            analyses[str(seed)], rows, arrays = regions_audit.analyze(records, times, {
                'paths': ['A', *ARMS], 'bootstrap': source.inherited['bootstrap'], 'comparisons': comparisons()})
            comparison_rows.extend({'seed': seed, **row} for row in rows)
            week_rows.extend(weekly_rows(records, times, seed))
            accounting[str(seed)] = partition_accounting(records)
            replays[str(seed)] = replay
            np.savez_compressed(partial / f'audit_weekly_s{seed}.npz', **arrays)
            progress('audit_seed_analyzed', seed=seed)
        write_csv(partial / 'selection_epochs.csv', epoch_rows)
        write_csv(partial / 'audit_regions.csv', comparison_rows)
        write_csv(partial / 'audit_weekly.csv', week_rows)
        write_csv(partial / 'group_membership.csv', membership)
        summary = {'status': 'STATE_INTERACTION_TRANSFER_AUDIT_COMPLETE',
            'protocol_id': protocol['protocol_id'], 'protocol_sha256': PROTOCOL_SHA256,
            'frozen_protocol': protocol, 'source_directory': str(source_root),
            'source_protocol_sha256': source.summary['protocol_sha256'],
            'source_git_head': source.summary['environment']['git_head'],
            **protocol['information_boundary'], 'main_training_ready': False,
            'recommendation': protocol['decision'], 'interpretation': protocol['interpretation'],
            'baseline_selection': baseline, 'selection_trajectories': trajectories,
            'audit': analyses, 'partition_accounting': accounting, 'matched_positive_replay': replays,
            'fit_probe': probes, 'full_fit_evaluation_performed': False,
            'phase_samples': source.summary['phase_samples'],
            'inputs': source.hashes, 'code_sha256': {str(path.relative_to(REPO)): sha256(path) for path in (
                Path(__file__), PROTOCOL, Path(__file__).with_name('state_interaction_trajectory.py'),
                Path(regions_audit.__file__), Path(__file__).with_name('audit_matched_controls.py'))},
            'environment': {'host': socket.gethostname(), 'python': sys.version, 'numpy': np.__version__,
                            'slurm_job_id': os.environ.get('SLURM_JOB_ID')},
            'outputs': {p.name: sha256(p) for p in partial.iterdir() if p.is_file() and p.name != 'progress.json'}}
        regions_audit.write_json(partial / 'summary.json', summary)
        if output.exists() or output.is_symlink():
            raise FileExistsError('Final output appeared; preserve partial')
        partial.rename(output)
    except BaseException as error:
        regions_audit.write_json(partial / 'failure.json', {'error': str(error), 'traceback': traceback.format_exc()})
        raise
    report(summary)
    print('Saved v12g transfer audit:', output / 'summary.json', flush=True)
    return summary


def report(summary):
    print('status:', summary['status'])
    print('protocol_sha256:', summary['protocol_sha256'])
    print('Saved-model development diagnostic; pointwise intervals; no independent confirmation.')
    for seed, arms in summary['selection_trajectories'].items():
        for arm, trajectory in arms.items():
            print(f'\n[{arm} seed={seed}] selected_epoch={trajectory["selected_epoch"]} '
                  f'eligible={trajectory["eligible_epochs"]}/{trajectory["epochs"]}')
            print('rejected_constraints=', {r['epoch']: r['failed_constraints'] for r in trajectory['rejected_epochs']})
            for cohort in ('incident_full', 'incident'):
                effects = trajectory['cohorts'][cohort]
                print('selected_selection', cohort, {r: effects['selected_selection'][r]['gain_vs_A_raw'] for r in REGIONS})
            for cohort in ('incident_full', *GROUPS, 'primary_control', 'secondary_control'):
                result = summary['audit'][seed]['results'][cohort]
                early = result['regions']['candidate_h1_h6']['comparisons'][arm + '_vs_A']
                global_ = result['regions']['all']['comparisons'][arm + '_vs_A']
                ci = early['intervals']['week']['pooled']
                print('audit', cohort, 'windows=', result['samples'], 'global_gain=', global_['gain_raw_mae'],
                      'candidate_early_gain=', early['gain_raw_mae'], 'early_week_CI=', [ci['ci_low'], ci['ci_high']])
            print('fit_probe_selected_gain=', summary['fit_probe'][seed][arm]['selected']['gain_vs_initial_probe_raw_mae'])
    print('Weekly details: audit_weekly.csv; group/region accounting and paired control contrasts: summary.json.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    execute = sub.add_parser('run')
    for name in ('source-dir', 'data-dir', 'output'):
        execute.add_argument('--' + name, type=Path, required=True)
    sub.add_parser('report').add_argument('summary', type=Path)
    args = parser.parse_args()
    if args.action == 'report':
        if not args.summary.is_file():
            print('INCOMPLETE: no final summary; inspect .job/run.log and preserve .partial.')
        else:
            report(json.loads(args.summary.read_text()))
    else:
        run(args.source_dir, args.data_dir, args.output)


if __name__ == '__main__':
    main()
