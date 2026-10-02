"""v12j: observable fit support and fixed-adapter regional gains, NumPy only."""

import argparse
import csv
from datetime import datetime, timedelta
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
from experiments.chronological import audit_state_interaction_transfer as transfer
from experiments.chronological.audit_architecture_regions import write_json
from experiments.chronological.audit_matched_controls import sha256, read_csv
from experiments.chronological.state_interaction_fit import PATHS, REGIONS
from experiments.chronological.state_interaction_support import analyze_support, FEATURE_NAMES

PROTOCOL = Path(__file__).with_name('state_interaction_support_v12j.json')
PROTOCOL_SHA256 = '02040548974590f1b3addc2347561839c33aae6cdee2c6468a089f8dd7891b29'
FIT_PROTOCOL = PROTOCOL.with_name('state_interaction_fit_v12i.json')


def load_protocol():
    if sha256(PROTOCOL) != PROTOCOL_SHA256:
        raise ValueError('Frozen v12j protocol changed')
    return json.loads(PROTOCOL.read_text(encoding='utf-8'))


class FitSource:
    def __init__(self, root, source, protocol):
        self.root = Path(root).resolve()
        self.protocol = source.protocol
        payload = (self.root / 'summary.json').read_bytes()
        self.hashes = {'summary.json': hashlib.sha256(payload).hexdigest()}
        self.summary = json.loads(payload)
        if sha256(FIT_PROTOCOL) != protocol['fit_protocol_sha256']:
            raise ValueError('Local frozen v12i protocol changed')
        frozen = json.loads(FIT_PROTOCOL.read_text(encoding='utf-8'))
        s = self.summary
        expected = {'status': 'STATE_INTERACTION_FULL_FIT_AUDIT_COMPLETE',
                    'protocol_id': frozen['protocol_id'],
                    'protocol_sha256': protocol['fit_protocol_sha256'],
                    'source_protocol_sha256': source.summary['protocol_sha256'],
                    'source_git_head': protocol['source_v12f_git_head'],
                    'main_training_ready': False, 'recommendation': frozen['decision'],
                    **frozen['information_boundary']}
        if any(s.get(key) != value or type(s.get(key)) is not type(value)
               for key, value in expected.items()):
            raise ValueError('Require a complete frozen v12i fixed-model source')
        if (s['frozen_protocol'] != frozen or s['phase_samples'] != source.summary['phase_samples']
                or s['environment']['git_head'] != protocol['fit_git_head']
                or source.summary['environment']['git_head'] != protocol['source_v12f_git_head']):
            raise ValueError('Fit source protocol, budget or implementation changed')
        for name, digest in protocol['fit_producer_code_sha256'].items():
            if s['code_sha256'].get(name) != digest:
                raise ValueError('Fit producer code identity changed')
        for name in frozen['runtime_source_identity_files']:
            if s['code_sha256'].get(name) != source.summary['code_sha256'][name]:
                raise ValueError('Fit inference runtime code identity changed')
        seeds = {str(seed) for seed in protocol['seeds']}
        if set(s['results']) != seeds or set(s['selected_models']) != seeds:
            raise ValueError('Fit source seed set changed')
        backbones = set()
        for seed in seeds:
            if set(s['selected_models'][seed]) != set(protocol['source_arms']):
                raise ValueError('Fit source arm set changed')
            for arm, metadata in s['selected_models'][seed].items():
                original = source.summary['runs'][seed][arm]
                if (type(metadata['selected_epoch']) is not int
                        or not 0 <= metadata['selected_epoch'] <= source.inherited['training']['epochs']
                        or metadata['selected_epoch'] != original['selected_epoch']
                        or type(metadata['adapter_parameters']) is not int
                        or metadata['adapter_parameters'] != original['trainable_parameters']
                        or metadata['adapter_state_sha256'] != original['representation_diagnostics']['selected']['adapter_state_sha256']
                        or metadata['model_state_unchanged'] is not True
                        or metadata['full_fit_samples'] != len(source.plan['fit']['positive_ids'])):
                    raise ValueError('Fit selected model metadata disagrees with original source')
                for field in ('adapter_state_sha256', 'backbone_state_sha256'):
                    digest = metadata[field]
                    if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
                        raise ValueError('Fit selected tensor state hash is invalid')
                backbones.add(metadata['backbone_state_sha256'])
                fallback = metadata['epoch_zero_full_fit_error_sums_exact_A']
                if (metadata['selected_epoch'] == 0 and fallback is not True
                        or metadata['selected_epoch'] != 0 and fallback is not None):
                    raise ValueError('Fit identity-fallback metadata disagrees')
        if len(backbones) != 1:
            raise ValueError('Fit source must retain the same frozen backbone across models')
        self.verify_source_hashes(source)

    def verify_source_hashes(self, source):
        for name, digest in source.hashes.items():
            if self.summary['inputs'].get(name) != digest:
                raise ValueError(f'v12i/v12f consumed source chain mismatch: {name}')

    def read(self, name):
        path = (self.root / name).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError('Fit artifact path escapes read-only source')
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if self.summary['outputs'].get(name) != digest:
            raise ValueError(f'Fit artifact hash mismatch: {name}')
        self.hashes[name] = digest
        return payload


def history_features(history, candidate, age):
    """History is already sliced to X; no target or scaler enters these summaries."""
    history, candidate = np.asarray(history), np.asarray(candidate)
    if (history.ndim != 2 or history.shape[0] != 12 or history.dtype.kind != 'f'
            or candidate.dtype.kind != 'b' or candidate.shape != (history.shape[1],)
            or not np.isfinite(age) or not 0 < age <= 5):
        raise ValueError('Invalid X-only history/candidate/report-age geometry')
    n = int(candidate.sum())
    values = history[:, candidate].astype(np.float64)
    valid = np.isfinite(values) & (values >= 0)
    observed = np.where(valid, values, 0.)
    counts = valid.sum(0)
    means = np.divide(observed.sum(0), counts, out=np.zeros(n), where=counts > 0)
    variance = np.divide(np.where(valid, (observed - means) ** 2, 0.).sum(0), counts,
                         out=np.zeros(n), where=counts > 0)
    halves = []
    for first, last in ((0, 6), (6, 12)):
        cells = valid[first:last].sum(0)
        halves.append((np.divide(observed[first:last].sum(0), cells,
                        out=np.zeros(n), where=cells > 0), cells > 0))
    paired = halves[0][1] & halves[1][1]
    return {
        'history_mean': float(observed.sum() / valid.sum()) if valid.any() else np.nan,
        'history_trend': float((halves[1][0] - halves[0][0])[paired].mean()) if paired.any() else np.nan,
        'history_volatility': float(np.sqrt(variance[counts >= 2]).mean()) if (counts >= 2).any() else np.nan,
        'history_missing_fraction': float(1 - valid.sum() / (12 * n)) if n else np.nan,
        'report_age_minutes': float(age), 'candidate_node_count': float(n),
    }


def read_features(data_dir, source, records):
    if set(records) != {'fit', 'audit'}:
        raise ValueError('Only fit/audit X history may enter state extraction')
    data_dir = Path(data_dir)
    hashes = {}
    for name in ('train_manifest.csv', 'train_flow.npy', 'station_ids.npy', 'train_context.npz'):
        path = data_dir / name
        declared = {digest for key, digest in source.summary['inputs'].items() if Path(key).name == name}
        digest = sha256(path)
        if len(declared) != 1 or digest not in declared:
            raise ValueError(f'Original train input fingerprint differs: {name}')
        hashes[str(path.resolve())] = digest
    rows = read_csv(data_dir / 'train_manifest.csv')
    station_ids = np.load(data_dir / 'station_ids.npy', allow_pickle=False)
    flow = np.load(data_dir / 'train_flow.npy', mmap_mode='r', allow_pickle=False)
    with np.load(data_dir / 'train_context.npz', allow_pickle=False) as stored:
        # Only report-time fields; no other context channels enter the diagnostic.
        context = {name: stored[name].copy() for name in
                   ('sample_indices', 'station_ids', 'distances', 'report_age_minutes', 'forecast_tod', 'forecast_dow')}
    ids = [int(row['sample_index']) for row in rows]
    n, nodes = len(rows), source.protocol['node_count']
    if (station_ids.ndim != 1 or station_ids.dtype.kind not in 'iu'
            or len(station_ids) != nodes or len(np.unique(station_ids)) != nodes
            or len(set(ids)) != n or flow.shape != (n, 26, nodes) or flow.dtype.kind != 'f'
            or not np.array_equal(context['sample_indices'], ids)
            or not np.array_equal(context['station_ids'], station_ids)
            or context['distances'].shape != (n, nodes, 3)
            or not np.isfinite(context['distances']).all()
            or any(context[name].shape != (n,) or not np.isfinite(context[name]).all()
                   for name in ('report_age_minutes', 'forecast_tod', 'forecast_dow'))
            or any(row['split'] != 'train' or int(row['source_version']) != 8 for row in rows)):
        raise ValueError('Original train history/report context axes disagree')
    features, times = {}, {}
    for phase, record in records.items():
        output = {name: [] for name in FEATURE_NAMES}
        times[phase] = {}
        for position, sample in enumerate(record['ids']):
            index = int(record['source_indices'][position])
            if not 0 <= index < n or ids[index] != sample:
                raise ValueError('History source index disagrees with saved forecast identity')
            row = rows[index]
            t0, report, x_start = [datetime.fromisoformat(row[name]) for name in ('t0', 'report_time', 'x_start')]
            elapsed = (t0 - report).total_seconds() / 60
            if (not 0 < elapsed <= 5
                    or abs(float(context['report_age_minutes'][index]) - elapsed) > 1e-5
                    or not x_start + timedelta(minutes=55) < report
                    or context['forecast_tod'][index] != t0.hour * 12 + t0.minute // 5
                    or context['forecast_dow'][index] != (t0.weekday() + 1) % 7):
                raise ValueError('Report context clock or history availability changed')
            candidate = np.any(context['distances'][index] != 0, axis=-1)
            if not np.array_equal(candidate, record['candidate_mask'][position]):
                raise ValueError('Saved candidate mask differs from original report context')
            values = history_features(flow[index, :12, :], candidate, elapsed)
            for name, value in values.items():
                output[name].append(value)
            times[phase][int(sample)] = row['t0']
        features[phase] = {name: np.asarray(values, dtype=np.float64) for name, values in output.items()}
    return features, times, hashes


def stack(reference, learned):
    ordered = [reference] + [learned[arm] for arm in transfer.ARMS]
    for record in ordered[1:]:
        transfer.aligned(reference, record)
    return {**reference, 'errors': np.stack([record['errors'] for record in ordered], axis=1)}


def reconcile(record, declared, label):
    for r, region in enumerate(REGIONS):
        saved = declared['regions'][region]
        if (saved['valid_cells'] != int(record['counts'][:, r].sum())
                or saved['prediction_cells'] != int(record['prediction_counts'][:, r].sum())
                or saved['evaluable_samples'] != int((record['counts'][:, r] > 0).sum())):
            raise ValueError(f'{label} saved support disagrees')
        denominator = record['counts'][:, r].sum()
        for p, path in enumerate(PATHS):
            value = float(record['errors'][:, p, r].sum() / denominator) if denominator else None
            expected = saved['mae'][path]
            if (value is None) != (expected is None):
                raise ValueError(f'{label} empty region availability disagrees')
            if value is not None:
                transfer.close(value, expected, f'{label}/{region}/{path}')


def write_csv(path, rows):
    if not rows:
        raise ValueError('Cannot publish an empty diagnostic table')
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(fit_dir, source_dir, data_dir, output):
    protocol = load_protocol()
    requested = Path(output)
    partial = requested.with_name(requested.name + '.partial')
    if any(path.exists() or path.is_symlink() for path in (requested, partial)):
        raise FileExistsError('Preserve output/partial; choose a new diagnostic run')
    inputs = [Path(path).resolve() for path in (fit_dir, source_dir, data_dir)]
    for path in (Path(fit_dir), Path(source_dir)):
        if path.is_symlink() or path.resolve().name.endswith('.partial'):
            raise ValueError('Require completed final sources, not symlinks or partial')
    output = requested.resolve()
    partial = output.with_name(output.name + '.partial')
    if any(output.is_relative_to(root) or partial.is_relative_to(root) for root in inputs):
        raise ValueError('Write diagnostic outside all read-only input directories')
    partial.mkdir(parents=True)
    started = time.monotonic()
    def progress(stage, **fields):
        event = {'stage': stage, 'elapsed_seconds': time.monotonic() - started, **fields}
        write_json(partial / 'progress.json', event)
        print(json.dumps(event), flush=True)
    try:
        progress('verifying_v12i_v12f_source_chain')
        source = transfer.Source(inputs[1], transfer.load_protocol())
        fit = FitSource(inputs[0], source, protocol)
        transfer.read_manifest(inputs[2], source)
        references = {
            'fit': transfer.read_record(fit, 'fit_A.npz', source.plan['fit']['positive_ids']),
            'audit': transfer.read_record(source, 'audit_A_incident_full.npz', source.plan['audit']['positive_ids']),
        }
        for phase, record in references.items():
            if not np.array_equal(record['source_indices'], source.plan[phase]['indices']['incident_full']):
                raise ValueError('Saved phase source positions changed')
        transfer.reconcile_metrics(references['audit'], source.summary['baseline_audit']['incident_full'], 'A audit')
        features, times, data_hashes = read_features(inputs[2], source, references)
        for name, digest in data_hashes.items():
            declared = {value for key, value in fit.summary['inputs'].items() if Path(key).name == Path(name).name}
            if declared != {digest}:
                raise ValueError('v12i history input provenance differs from v12f')
        progress('report_time_history_features_verified',
                 phase_samples={phase: len(record['ids']) for phase, record in references.items()})
        results, state_rows, gain_rows, weekly_rows, composition_rows = {}, [], [], [], []
        memberships = None
        for seed in protocol['seeds']:
            learned = {'fit': {}, 'audit': {}}
            for arm in protocol['source_arms']:
                learned['fit'][arm] = transfer.read_record(fit, f'fit_{arm}_s{seed}.npz', source.plan['fit']['positive_ids'])
                learned['audit'][arm] = transfer.read_record(source, f'{arm}_s{seed}/audit_incident_full.npz', source.plan['audit']['positive_ids'])
                transfer.reconcile_metrics(learned['audit'][arm],
                    source.summary['runs'][str(seed)][arm]['audit']['incident_full'], f'{arm}/s{seed} audit')
                if fit.summary['selected_models'][str(seed)][arm]['selected_epoch'] == 0:
                    transfer.close(learned['fit'][arm]['errors'], references['fit']['errors'],
                                   'Epoch-zero full-fit identity', rtol=0., atol=0.)
                    transfer.close(learned['audit'][arm]['errors'], references['audit']['errors'],
                                   'Epoch-zero audit identity', rtol=0., atol=0.)
            records = {phase: stack(reference, learned[phase]) for phase, reference in references.items()}
            for phase, record in records.items():
                reconcile(record, fit.summary['results'][str(seed)]['phases'][phase], f's{seed}/{phase}')
            analysis, states, gains, weeks, composition = analyze_support(
                records, features, times, source.inherited['bootstrap'], protocol['state_spec'])
            if memberships is None:
                memberships = states
                state_rows = states
            elif states != memberships:
                raise ValueError('State membership changed across seeds')
            results[str(seed)] = analysis
            gain_rows.extend({'seed': seed, **row} for row in gains)
            weekly_rows.extend({'seed': seed, **row} for row in weeks)
            composition_rows.extend({'seed': seed, **row} for row in composition)
            progress('observable_support_statistics_complete', seed=seed)
        fit.verify_source_hashes(source)
        for name, rows in (('state_membership.csv', state_rows), ('conditional_gains.csv', gain_rows),
                           ('weekly_gains.csv', weekly_rows), ('composition_accounting.csv', composition_rows)):
            write_csv(partial / name, rows)
        code_files = [Path(__file__), PROTOCOL, Path(__file__).with_name('state_interaction_support.py'),
                      Path(__file__).with_name('run_state_interaction_support_audit.sh'),
                      Path(transfer.__file__), transfer.PROTOCOL, transfer.SOURCE_PROTOCOL,
                      transfer.INHERITED_PROTOCOL, FIT_PROTOCOL,
                      Path(__file__).with_name('state_interaction_fit.py'),
                      Path(__file__).with_name('audit_architecture_regions.py'),
                      Path(__file__).with_name('audit_matched_controls.py')]
        summary = {'status': 'STATE_INTERACTION_OBSERVABLE_SUPPORT_AUDIT_COMPLETE',
            'protocol_id': protocol['protocol_id'], 'protocol_sha256': PROTOCOL_SHA256,
            'frozen_protocol': protocol, **protocol['information_boundary'],
            'main_training_ready': False, 'recommendation': protocol['decision'],
            'interpretation': protocol['interpretation'], 'results': results,
            'selected_models': fit.summary['selected_models'],
            'selection_conditional_statistics_status': 'UNAVAILABLE_SAVED_AGGREGATES_ONLY',
            'fit_directory': str(inputs[0]), 'source_directory': str(inputs[1]),
            'source_phase_samples': source.summary['phase_samples'],
            'inputs': {'v12i': fit.hashes, 'v12f': source.hashes, 'report_time_inputs': data_hashes},
            'code_sha256': {str(path.relative_to(REPO)): sha256(path) for path in code_files},
            'environment': {'host': socket.gethostname(), 'python': sys.version, 'numpy': np.__version__,
                            'device': 'cpu', 'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
                            'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()},
            'outputs': {path.name: sha256(path) for path in partial.iterdir()
                        if path.is_file() and path.name != 'progress.json'}}
        write_json(partial / 'summary.json', summary)
        if output.exists() or output.is_symlink():
            raise FileExistsError('Final output appeared; preserve partial')
        partial.rename(output)
    except BaseException as error:
        write_json(partial / 'failure.json', {'error': str(error), 'traceback': traceback.format_exc()})
        raise
    report(summary)
    print('Saved v12j observable-support audit:', output / 'summary.json', flush=True)
    return summary


def report(summary):
    print('status:', summary['status'])
    print('protocol_sha256:', summary['protocol_sha256'])
    print('FIT-ONLY OBSERVABLE RULES; SAVED MODELS; NO NEW TRAINING, SELECTION OR ROUTER')
    first = next(iter(summary['results'].values()))
    print('fit_only_marginal_bins:', first['marginal_bins'])
    print('fit_only_joint_cells:', first['joint_cells'])
    for seed, analysis in summary['results'].items():
        print(f'\n[seed={seed}]')
        print('saved_selected_epochs:', {arm: summary['selected_models'][seed][arm]['selected_epoch']
              for arm in ('state_vector', 'interaction_vector')})
        for phase, result in analysis['phases'].items():
            for group in result['groups']:
                if group['grouping'] != 'support':
                    continue
                region = group['regions']['candidate_h1_h6']
                for comparison, effect in region['comparisons'].items():
                    ci = effect['intervals']['four_week_block']['pooled']
                    print(phase, group['group'], comparison,
                          'windows=', group['forecast_windows'], 'early_cells=', region['valid_cells'],
                          'early_cell_share=', region['phase_region_valid_cell_fraction'],
                          'gain=', effect['gain_raw_mae'],
                          'equal_window_gain=', effect['equal_forecast_window_gain_raw_mae'],
                          'block_CI=', [ci['ci_low'], ci['ci_high']], 'CI_status=', ci['status'])
        for row in analysis['composition_accounting']:
            if row['region'] == 'candidate_h1_h6':
                fields = ('status', 'shared_supported_cells', 'fit_common_gain_raw_mae',
                          'audit_common_gain_raw_mae', 'audit_fit_mass_standardized_gain_raw_mae',
                          'composition_component_raw_mae', 'within_cell_component_raw_mae',
                          'fit_shared_supported_fraction_of_all_region_mass',
                          'audit_shared_supported_fraction_of_all_region_mass')
                print('early_composition', row['comparison'], row['estimand'],
                      {key: row[key] for key in fields})
    print('Fit is in-sample; audit is reused development. Coarse support and composition algebra are not causal diagnoses or deployment safety.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    execute = commands.add_parser('run')
    for name in ('fit-dir', 'source-dir', 'data-dir', 'output'):
        execute.add_argument('--' + name, type=Path, required=True)
    commands.add_parser('report').add_argument('summary', type=Path)
    args = parser.parse_args()
    if args.action == 'report':
        if args.summary.is_file():
            report(json.loads(args.summary.read_text(encoding='utf-8')))
        else:
            print('INCOMPLETE: no final summary; inspect .job/run.log and preserve .partial.')
    else:
        run(args.fit_dir, args.source_dir, args.data_dir, args.output)


if __name__ == '__main__':
    main()
