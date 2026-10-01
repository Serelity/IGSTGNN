"""v12h: composition contrasts and temporal sensitivity from saved v12g statistics."""

import argparse
import csv
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
from experiments.chronological.audit_architecture_regions import write_json
from experiments.chronological.audit_matched_controls import sha256
from experiments.chronological.state_interaction_stability import analyze_weekly, COHORTS, GROUPS, REGIONS

PROTOCOL = Path(__file__).with_name('state_interaction_stability_v12h.json')
PROTOCOL_SHA256 = 'cb396d4384fdd0b3eb81920bdfd66ecef827abc93bd25dde092b7ccf1259c5a2'
SOURCE_PROTOCOL = PROTOCOL.with_name('state_interaction_transfer_v12g.json')


def load_protocol():
    if sha256(PROTOCOL) != PROTOCOL_SHA256:
        raise ValueError('Frozen v12h protocol changed')
    return json.loads(PROTOCOL.read_text(encoding='utf-8'))


def close(observed, expected, label):
    if not np.allclose(observed, expected, rtol=1e-10, atol=1e-9):
        raise ValueError(f'{label} disagrees')


class Source:
    def __init__(self, root, protocol):
        self.root = Path(root).resolve()
        self.protocol = protocol
        payload = (self.root / 'summary.json').read_bytes()
        self.hashes = {'summary.json': hashlib.sha256(payload).hexdigest()}
        self.summary = json.loads(payload)
        if sha256(SOURCE_PROTOCOL) != protocol['source_protocol_sha256']:
            raise ValueError('Local frozen v12g protocol changed')
        frozen = json.loads(SOURCE_PROTOCOL.read_text(encoding='utf-8'))
        s = self.summary
        expected = {
            'status': 'STATE_INTERACTION_TRANSFER_AUDIT_COMPLETE',
            'protocol_id': frozen['protocol_id'], 'protocol_sha256': protocol['source_protocol_sha256'],
            'source_protocol_sha256': protocol['source_v12f_protocol_sha256'],
            'source_git_head': protocol['source_v12f_git_head'],
            'main_training_ready': False, 'full_fit_evaluation_performed': False,
            'recommendation': frozen['decision'], **frozen['information_boundary'],
        }
        if any(s.get(k) != value or type(s.get(k)) is not type(value) for k, value in expected.items()):
            raise ValueError('Require a complete frozen v12g saved-only development source')
        if (s['frozen_protocol'] != frozen or s['phase_samples'] != protocol['source_phase_samples']
                or s['code_sha256'] != protocol['source_code_sha256']):
            raise ValueError('Source protocol, budget or recorded implementation changed')
        seeds = {str(seed) for seed in protocol['seeds']}
        for field in ('audit', 'partition_accounting', 'selection_trajectories'):
            if set(s[field]) != seeds:
                raise ValueError(f'Source seed set changed: {field}')
        for seed in seeds:
            analysis = s['audit'][seed]
            if analysis['weeks'] != protocol['weeks'] or set(analysis['results']) != set(COHORTS):
                raise ValueError('Source calendar or cohort axes changed')
            for cohort, result in analysis['results'].items():
                if (type(result['samples']) is not int or result['samples'] != protocol['cohort_samples'][cohort]
                        or set(result['regions']) != set(REGIONS)):
                    raise ValueError('Source cohort budget or region set changed')
                for region in result['regions'].values():
                    if set(region['comparisons']) != set(protocol['comparisons']):
                        raise ValueError('Source comparison set changed')
            if set(s['partition_accounting'][seed]) != set(protocol['comparisons']):
                raise ValueError('Source accounting comparison set changed')
            trajectories = s['selection_trajectories'][seed]
            if set(trajectories) != set(protocol['arms']):
                raise ValueError('Source arm set changed')
            for trajectory in trajectories.values():
                if (type(trajectory['selected_epoch']) is not int or not 0 <= trajectory['selected_epoch'] <= 12
                        or trajectory['epochs'] != 12):
                    raise ValueError('Source selection metadata changed')

    def read_arrays(self, seed):
        name = f'audit_weekly_s{seed}.npz'
        path = (self.root / name).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError('Saved artifact path escapes read-only source')
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if self.summary['outputs'].get(name) != digest:
            raise ValueError(f'Source artifact hash mismatch: {name}')
        self.hashes[name] = digest
        with np.load(io.BytesIO(payload), allow_pickle=False) as arrays:
            result = {name: arrays[name].copy() for name in arrays.files}
        if result['weeks'].tolist() != self.protocol['weeks']:
            raise ValueError('Saved calendar axis differs from frozen source period')
        return result


def reconcile_accounting(arrays, analysis, source_analysis, declared):
    total = arrays['incident_full_valid_counts'][:, 0].sum()
    for comparison, observed in analysis['full_positive_contributions'].items():
        saved = declared[comparison]
        close(observed['full_global_gain_raw_mae'], saved['full_global_gain_raw_mae'], 'Source full-global gain')
        close(observed['full_global_gain_raw_mae'], saved['reconstructed_full_global_gain_raw_mae'], 'Source reconstructed gain')
        if set(saved['groups']) != set(GROUPS):
            raise ValueError('Source accounting group set changed')
        for group in GROUPS:
            group_cells = arrays[group + '_valid_counts'][:, 0].sum()
            item = saved['groups'][group]
            if item['samples'] != source_analysis['results'][group]['samples']:
                raise ValueError('Source group accounting sample count disagrees')
            close(group_cells / total, item['valid_cell_share_of_full_global'], 'Source group cell share')
            for region in REGIONS:
                close(observed['contribution_to_full_global_gain'][group][region],
                      item['contribution_to_full_global_gain'][region], 'Source group/region global contribution')


def write_csv(path, rows):
    if not rows:
        raise ValueError('Cannot publish an empty diagnostic table')
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(source_dir, output):
    protocol = load_protocol()
    requested = Path(output)
    requested_partial = requested.with_name(requested.name + '.partial')
    if any(path.exists() or path.is_symlink() for path in (requested, requested_partial)):
        raise FileExistsError('Preserve output/partial; choose a new diagnostic run')
    output, source_root = requested.resolve(), Path(source_dir).resolve()
    partial = output.with_name(output.name + '.partial')
    if output.is_relative_to(source_root) or partial.is_relative_to(source_root):
        raise ValueError('Write diagnostic output outside the read-only v12g source')
    partial.mkdir(parents=True)
    started = time.monotonic()
    def progress(stage, **fields):
        event = {'stage': stage, 'elapsed_seconds': time.monotonic() - started, **fields}
        write_json(partial / 'progress.json', event)
        print(json.dumps(event), flush=True)
    try:
        progress('verifying_saved_v12g_source')
        source = Source(source_root, protocol)
        analyses, selection = {}, {}
        contrast_rows, temporal_rows, contribution_rows = [], [], []
        original_weights, original_support = None, None
        for seed in protocol['seeds']:
            arrays = source.read_arrays(seed)
            weights = {key: arrays[key] for key in ('bootstrap_week_weights', 'bootstrap_four_week_block_weights')}
            if original_weights is None:
                original_weights = weights
            elif any(not np.array_equal(value, original_weights[key]) for key, value in weights.items()):
                raise ValueError('Source bootstrap matrices differ across seeds')
            support = {f'{cohort}_{field}': arrays[f'{cohort}_{field}']
                       for cohort in COHORTS for field in ('valid_counts', 'evaluable_events')}
            if original_support is None:
                original_support = support
            elif any(not np.array_equal(value, original_support[key]) for key, value in support.items()):
                raise ValueError('Source evaluation support differs across seeds')
            analysis, contrasts, weeks, contributions = analyze_weekly(
                arrays, source.summary['audit'][str(seed)], protocol['comparisons'], protocol['bootstrap'])
            reconcile_accounting(arrays, analysis, source.summary['audit'][str(seed)],
                                 source.summary['partition_accounting'][str(seed)])
            analyses[str(seed)] = analysis
            contrast_rows.extend({'seed': seed, **row} for row in contrasts)
            temporal_rows.extend({'seed': seed, **row} for row in weeks)
            contribution_rows.extend({'seed': seed, **row} for row in contributions)
            selection[str(seed)] = {arm: {
                'selected_epoch': t['selected_epoch'], 'eligible_epochs': t['eligible_epochs'],
                'selected_selection': {cohort: t['cohorts'][cohort]['selected_selection']
                                       for cohort in ('incident_full', 'incident')},
            } for arm, t in source.summary['selection_trajectories'][str(seed)].items()}
            progress('saved_seed_reconciled', seed=seed)
        write_csv(partial / 'composition_contrasts.csv', contrast_rows)
        write_csv(partial / 'temporal_stability.csv', temporal_rows)
        write_csv(partial / 'global_contributions.csv', contribution_rows)
        code_files = (Path(__file__), PROTOCOL, Path(__file__).with_name('state_interaction_stability.py'),
                      Path(__file__).with_name('audit_architecture_regions.py'),
                      Path(__file__).with_name('audit_matched_controls.py'))
        summary = {
            'status': 'STATE_INTERACTION_STABILITY_AUDIT_COMPLETE',
            'protocol_id': protocol['protocol_id'], 'protocol_sha256': PROTOCOL_SHA256,
            'frozen_protocol': protocol, **protocol['information_boundary'], 'main_training_ready': False,
            'recommendation': protocol['decision'], 'interpretation': protocol['interpretation'],
            'source_directory': str(source_root), 'source_protocol_sha256': source.summary['protocol_sha256'],
            'source_v12f_git_head': source.summary['source_git_head'],
            'audit': analyses, 'selection_context': selection,
            'cohort_samples': protocol['cohort_samples'], 'inputs': source.hashes,
            'absolute_error_sums_rechecked': False, 'original_traffic_input_hashes_rechecked': False,
            'code_sha256': {str(path.relative_to(REPO)): sha256(path) for path in code_files},
            'environment': {'host': socket.gethostname(), 'python': sys.version, 'numpy': np.__version__,
                            'slurm_job_id': os.environ.get('SLURM_JOB_ID')},
            'outputs': {p.name: sha256(p) for p in partial.iterdir() if p.is_file() and p.name != 'progress.json'},
        }
        write_json(partial / 'summary.json', summary)
        if output.exists() or output.is_symlink():
            raise FileExistsError('Final output appeared; preserve partial')
        partial.rename(output)
    except BaseException as error:
        write_json(partial / 'failure.json', {'error': str(error), 'traceback': traceback.format_exc()})
        raise
    report(summary)
    print('Saved v12h stability audit:', output / 'summary.json', flush=True)
    return summary


def report(summary):
    print('status:', summary['status'])
    print('protocol_sha256:', summary['protocol_sha256'])
    print('Saved-model development diagnostic; calendar-paired pointwise intervals; no new selection.')
    for seed, analysis in summary['audit'].items():
        for arm in summary['frozen_protocol']['arms']:
            comparison = arm + '_vs_A'
            context = summary['selection_context'][seed][arm]
            print(f'\n[{arm} seed={seed}] saved_selected_epoch={context["selected_epoch"]}')
            print('selection_early_gain=', context['selected_selection']['incident_full']['candidate_h1_h6']['gain_vs_A_raw'])
            for group in ('incident_full', *GROUPS):
                effects = analysis['cohort_stability'][group]['regions']['candidate_h1_h6']['comparisons'][comparison]
                for name, effect in effects.items():
                    ci = effect['intervals']['four_week_block']
                    loo = effect['leave_one_week_out']
                    print('early', group, name, 'gain=', effect['point_gain_raw_mae'],
                          'block_CI=', [ci['ci_low'], ci['ci_high']], 'CI_status=', ci['status'],
                          'week_signs=', effect['weekly_sign_counts'],
                          'LOO_range=', [loo['minimum_gain_raw_mae'], loo['maximum_gain_raw_mae']])
            for name, effect in analysis['common_minus_complement']['candidate_h1_h6'][comparison].items():
                print('common_minus_complement', name, 'gain=', effect['point_gain_raw_mae'],
                      'week_CI=', effect['intervals']['week'],
                      'block_CI=', effect['intervals']['four_week_block'])
            accounting = analysis['full_positive_contributions'][comparison]
            print('full_global_gain=', accounting['full_global_gain_raw_mae'])
            print('global_contributions=', accounting['contribution_to_full_global_gain'])
    print('LOO is descriptive; group identity is observational, not a deployable router. All seeds retained.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    execute = sub.add_parser('run')
    for name in ('source-dir', 'output'):
        execute.add_argument('--' + name, type=Path, required=True)
    sub.add_parser('report').add_argument('summary', type=Path)
    args = parser.parse_args()
    if args.action == 'report':
        if not args.summary.is_file():
            print('INCOMPLETE: no final summary; inspect .job/run.log and preserve .partial.')
        else:
            report(json.loads(args.summary.read_text(encoding='utf-8')))
    else:
        run(args.source_dir, args.output)


if __name__ == '__main__':
    main()
