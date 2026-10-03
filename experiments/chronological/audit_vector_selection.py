"""v12l: explain frozen v12k selection decisions using saved JSON only."""

import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.chronological import vector_selection_audit as analysis

PROTOCOL = Path(__file__).with_name('vector_selection_audit_v12l.json')
PROTOCOL_SHA256 = '52a1a87fc79f71077ba35af6ddc670ba25602c74437ba722a944e302be5e4df7'
SOURCE_PROTOCOL = PROTOCOL.with_name('vector_objective_alignment_v12k.json')
INHERITED_PROTOCOL = PROTOCOL.with_name('incident_strength_gate_v12c.json')
require, identical = analysis.require, analysis.identical


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def valid_hash(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def decode(payload):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, f'Duplicate JSON key: {key}')
            result[key] = value
        return result
    def invalid(value):
        raise ValueError(f'Nonfinite JSON number: {value}')
    return json.loads(payload, object_pairs_hook=pairs, parse_constant=invalid)


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def ordinary_path(path):
    path = Path(os.path.abspath(path))
    require(not any(p.is_symlink() for p in (path, *path.parents)), f'Symlinked path is not allowed: {path}')
    return path


def load_protocol():
    require(sha256(PROTOCOL) == PROTOCOL_SHA256, 'Frozen v12l protocol changed')
    protocol = decode(PROTOCOL.read_bytes())
    require(sha256(SOURCE_PROTOCOL) == protocol['source_protocol_sha256'], 'Frozen v12k protocol changed')
    require(sha256(INHERITED_PROTOCOL) == protocol['inherited_protocol_sha256'], 'Frozen v12c protocol changed')
    return protocol


def validate_metrics(value, expected_samples, spec):
    require(set(value) == set(expected_samples), 'Selection cohort set changed')
    for cohort, samples in expected_samples.items():
        item = value[cohort]
        require(type(item['samples']) is int and item['samples'] == samples, 'Selection sample count changed')
        for region in spec['protected_regions']:
            require(region in item['mae'], f'Missing selection metric: {cohort}/{region}')
        for metric in item['mae'].values():
            require(metric is None or analysis.number(metric), 'Selection MAE must be nonnegative finite or null')


class Source:
    """Read an explicit JSON allowlist, verifying the producer's recorded hashes."""

    def __init__(self, root, protocol):
        self.root = ordinary_path(root)
        require(self.root.is_dir() and not self.root.name.endswith('.partial'), 'Require a completed source directory')
        self.hashes = {}
        self.summary = self.read('summary.json', manifest=False)
        s = self.summary
        frozen = decode(SOURCE_PROTOCOL.read_bytes())
        inherited = decode(INHERITED_PROTOCOL.read_bytes())
        expected = {'status': 'VECTOR_OBJECTIVE_ALIGNMENT_COMPARISON_COMPLETE',
            'protocol_id': frozen['protocol_id'], 'protocol_sha256': protocol['source_protocol_sha256'],
            'engineering_check': False, 'main_training_ready': False,
            'all_selectors_frozen_before_audit_evaluation': True, 'paired_initialization_exact': True,
            'model_training_performed': True, 'training_scope': 'icsf_vector_adapter_only',
            'recommendation': frozen['decision'], **frozen['information_boundary']}
        require(all(k in s and identical(s[k], v) for k, v in expected.items()), 'Require a complete frozen full v12k source')
        require(identical(s['frozen_protocol'], frozen) and identical(s['inherited_protocol'], inherited),
                'Source frozen/inherited protocol mismatch')
        require(s['phase_samples'] == frozen['expected_phase_samples'], 'Source phase sample budget changed')
        training = copy.deepcopy(inherited['training'])
        training['objective'] = 'v12k paired global/candidate_early objectives; unchanged vector energy penalty'
        require(identical(s['effective_training'], training), 'Source training budget changed')
        for name, digest in protocol['producer_code_sha256'].items():
            require(s['code_sha256'].get(name) == digest, f'Producer code mismatch: {name}')
        for path, digest in ((SOURCE_PROTOCOL, protocol['source_protocol_sha256']),
                             (INHERITED_PROTOCOL, protocol['inherited_protocol_sha256'])):
            require(s['code_sha256'].get(str(path.relative_to(REPO))) == digest, 'Producer protocol hash mismatch')
        identity = self.read('run_identity.json')
        expected_identity = {'protocol_sha256': s['protocol_sha256'], 'engineering_check': False,
            'code_sha256': s['code_sha256'], 'inputs': s['inputs'], 'effective_training': training,
            'seeds': protocol['seeds']}
        require(all(identical(identity.get(k), v) for k, v in expected_identity.items()), 'Source run identity mismatch')
        require(set(identity['indices']) == set(s['phase_samples']), 'Run identity phase set changed')
        for phase, cohorts in s['phase_samples'].items():
            require(set(identity['indices'][phase]) == set(cohorts), 'Run identity cohort set changed')
            for cohort, count in cohorts.items():
                ids = identity['indices'][phase][cohort]
                require(isinstance(ids, list) and len(ids) == count and len(set(ids)) == count
                        and all(type(i) is int and i >= 0 for i in ids), 'Source indices invalid')
        self.frozen_endpoints = self.read('selected_endpoints_frozen.json')
        self.spec = inherited['selection']
        self.baseline = s['baseline']['selection']
        self.expected_samples = s['phase_samples']['selection']
        validate_metrics(self.baseline, self.expected_samples, self.spec)
        require(set(s['runs']) == {str(seed) for seed in protocol['seeds']}, 'Source seed set changed')
        self.fits = {}
        checkpoint_keys, backbones = set(), set()
        steps = (s['phase_samples']['fit']['incident_full'] + training['batch_size'] - 1) // training['batch_size']
        for seed in protocol['seeds']:
            names = {f'{a}__loss_{l}_s{seed}': (a, l) for a in protocol['arms'] for l in protocol['losses']}
            require(set(s['runs'][str(seed)]) == set(names), 'Source architecture/loss set changed')
            initials = set()
            for name, (arm, loss) in names.items():
                detail = self.read(f'{name}/fit_summary.json')
                reported = copy.deepcopy(s['runs'][str(seed)][name])
                require(set(reported['selectors']) == set(analysis.SELECTORS), 'Source final selector set changed')
                for selected in reported['selectors'].values():
                    require(identical(selected['phase_metrics']['selection'], selected['selection_metrics']),
                            'Selected selection metrics disagree with final replay')
                    # Never consult fit/audit outcomes when explaining selection.
                    del selected['phase_metrics']
                require(identical(detail, reported), 'Fit summary disagrees with main summary')
                for key, value in {'arm': arm, 'loss': loss, 'seed': seed, 'epochs': protocol['epochs'],
                        'optimizer_steps': steps * protocol['epochs'], 'trainable_parameters': 4288,
                        'initial_prediction_exactly_A': True, 'backbone_state_unchanged': True}.items():
                    require(identical(detail.get(key), value), f'Fit metadata mismatch: {name}/{key}')
                for key in ('initial_adapter_sha256', 'backbone_state_sha256'):
                    require(valid_hash(detail[key]), f'Invalid state hash: {name}/{key}')
                initials.add(detail['initial_adapter_sha256'])
                backbones.add(detail['backbone_state_sha256'])
                history = self.read(f'{name}/history.json')
                require(isinstance(history, list) and len(history) == protocol['epochs'], 'Incomplete epoch history')
                for row in history:
                    validate_metrics(row['selection'], self.expected_samples, self.spec)
                    require(type(row['training']['optimizer_steps']) is int and row['training']['optimizer_steps'] == steps
                            and row['training']['loss_region'] == loss, 'History training budget/loss mismatch')
                    require(valid_hash(row['adapter_state_sha256']), 'Invalid historical state hash')
                for selector, choice in detail['selectors'].items():
                    ep = choice['selected_epoch']
                    require(type(ep) is int and 0 <= ep <= protocol['epochs'], 'Invalid selected epoch')
                    expected_state = history[ep - 1]['adapter_state_sha256'] if ep else detail['initial_adapter_sha256']
                    require(choice['adapter_state_sha256'] == expected_state, 'Selected state reference disagrees with history')
                    key = f'{name}/selected_{selector}.pt'
                    checkpoint_keys.add(key)
                    digest = choice['checkpoint_sha256']
                    require(valid_hash(digest) and self.frozen_endpoints.get(key) == digest
                            and s['outputs'].get(key) == digest, 'Frozen checkpoint hash references disagree')
                self.fits[name] = (detail, history)
            require(len(initials) == 1, 'Paired initialization references differ')
        require(len(backbones) == 1, 'Frozen backbone references differ')
        require(set(self.frozen_endpoints) == checkpoint_keys, 'Frozen endpoint set changed')

    def read(self, name, manifest=True):
        require(name.endswith('.json') and not Path(name).is_absolute() and '..' not in Path(name).parts,
                'Only relative JSON artifact paths are allowed')
        path = ordinary_path(self.root / name)
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if manifest:
            require(self.summary['outputs'].get(name) == digest, f'Source artifact hash mismatch: {name}')
        self.hashes[name] = digest
        return decode(payload)

    def recheck(self):
        for name, digest in self.hashes.items():
            require(sha256(ordinary_path(self.root / name)) == digest, f'Source changed during audit: {name}')


def write_csv(path, rows):
    require(bool(rows), f'Empty required table: {path.name}')
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                             for k, v in row.items()})


def run(source_dir, output):
    protocol = load_protocol()
    source_dir, output = ordinary_path(source_dir), ordinary_path(output)
    partial = ordinary_path(output.with_name(output.name + '.partial'))
    require(not any(p.is_relative_to(source_dir) or source_dir.is_relative_to(p) for p in (output, partial)),
            'Output must be separate from the read-only source')
    for path in (output, partial):
        if path.exists():
            raise FileExistsError(f'Preserve existing output and use a new run name: {path}')
    partial.mkdir(parents=True)
    started = time.monotonic()
    paths = [Path(__file__), Path(analysis.__file__), PROTOCOL, SOURCE_PROTOCOL, INHERITED_PROTOCOL,
             PROTOCOL.with_name('run_vector_selection_audit.sh')]
    code_hashes = {str(p.relative_to(REPO)): sha256(p) for p in paths}

    def progress(stage, **fields):
        item = {'stage': stage, 'elapsed_seconds': time.monotonic() - started, **fields}
        write_json(partial / 'progress.json', item)
        print(json.dumps(item), flush=True)

    try:
        progress('verifying_saved_json')
        source = Source(source_dir, protocol)
        progress('source_verified', fits=len(source.fits), consumed_json_files=len(source.hashes))
        fits, epochs, checks, decisions = {}, [], [], []
        selected_rows = []
        for name, (detail, history) in source.fits.items():
            result = analysis.audit_history(history, source.baseline, source.spec)
            tag = {k: detail[k] for k in ('arm', 'loss', 'seed')}
            tag['fit'] = name
            for selector, replay in result['replayed_selectors'].items():
                original = detail['selectors'][selector]
                require(all(identical(original[k], v) for k, v in replay.items()),
                        f'Final selector replay mismatch: {name}/{selector}')
                selected_rows.append({**tag, 'selector': selector, 'selected_epoch': replay['selected_epoch'],
                    'fallback_to_A': replay['selected_epoch'] == 0,
                    'global_gain': analysis.gain(source.baseline['incident_full']['mae']['all'],
                                                replay['selection_metrics']['incident_full']['mae']['all']),
                    'early_gain': analysis.gain(source.baseline['incident_full']['mae'][analysis.EARLY],
                                               replay['selection_metrics']['incident_full']['mae'][analysis.EARLY])})
            fits[name] = {**tag, 'selected_epochs': {s: r['selected_epoch'] for s, r in result['replayed_selectors'].items()},
                'accounting': analysis.accounting(result['epochs'], result['checks'], result['decisions'], source.spec),
                'trajectory': result['epochs']}
            epochs.extend({**tag, **row} for row in result['epochs'])
            checks.extend({**tag, **row} for row in result['checks'])
            decisions.extend({**tag, **row} for row in result['decisions'])
            progress('fit_replayed', fit=name, selected_epochs=fits[name]['selected_epochs'])
        joint_rejected = [e for e in epochs if e['joint_positive_protection_rejected']]
        for epoch in joint_rejected:
            epoch['blockers'] = [r for r in checks if r['fit'] == epoch['fit'] and r['epoch'] == epoch['epoch'] and not r['passed']]
        total = analysis.accounting(epochs, checks, decisions, source.spec)
        total.update(fits=len(fits), selected_endpoints=len(selected_rows),
                     fallback_endpoints=sum(r['fallback_to_A'] for r in selected_rows))
        tables = {'epoch_metrics.csv': checks, 'selector_decisions.csv': decisions,
                  'selected_endpoints.csv': selected_rows}
        for name, rows in tables.items():
            write_csv(partial / name, rows)
        write_json(partial / 'joint_positive_rejections.json', joint_rejected)
        write_json(partial / 'cofailures.json', total['cofailures'])
        source.recheck()
        require(code_hashes == {str(p.relative_to(REPO)): sha256(p) for p in paths}, 'Audit code changed while running')
        summary = {'status': 'VECTOR_SELECTION_FAILURE_AUDIT_COMPLETE',
            'protocol_id': protocol['protocol_id'], 'protocol_sha256': PROTOCOL_SHA256, 'frozen_protocol': protocol,
            **protocol['information_boundary'], 'main_training_ready': False,
            'recommendation': protocol['decision'], 'interpretation': protocol['interpretation'],
            'source_dir': str(source_dir), 'source_protocol_sha256': source.summary['protocol_sha256'],
            'source_git_head': source.summary['environment']['git_head'],
            'source_json_sha256': source.hashes, 'code_sha256': code_hashes,
            'all_stored_decisions_replayed_exactly': True, 'all_final_selection_references_match': True,
            'protection_rule': source.spec, 'totals': total, 'fits': fits,
            'joint_positive_rejections': joint_rejected,
            'environment': {'host': socket.gethostname(), 'python': sys.version, 'device': 'cpu',
                'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
                'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()},
            'elapsed_seconds': time.monotonic() - started,
            'outputs': {name: sha256(partial / name) for name in [*tables, 'joint_positive_rejections.json', 'cofailures.json']}}
        write_json(partial / 'summary.json', summary)
        require(not output.exists() and not output.is_symlink(), 'Final output appeared; preserve partial')
        partial.rename(output)
    except BaseException as error:
        write_json(partial / 'failure.json', {'error': str(error), 'traceback': traceback.format_exc()})
        raise
    report(summary)
    print('Saved v12l selection audit:', output / 'summary.json', flush=True)
    return summary


def report(summary):
    print('status:', summary['status'])
    print('protocol_sha256:', summary['protocol_sha256'])
    print('source_git_head:', summary['source_git_head'])
    print('Exact original-selector replay:', summary['all_stored_decisions_replayed_exactly'])
    t = summary['totals']
    print('fits/epochs/selector_decisions/endpoints/fallbacks:',
          *(t[k] for k in ('fits', 'trajectory_epochs', 'selector_decisions', 'selected_endpoints', 'fallback_endpoints')))
    print('joint_positive_epochs:', t['joint_positive_epochs'])
    print('joint_positive_protection_rejected_epochs:', t['joint_positive_protection_rejected_epochs'])
    for name, result in summary['fits'].items():
        print(f'\n[{name}] selected_epochs={result["selected_epochs"]}')
        a = result['accounting']
        print('global_gain/early_gain/both_gain/protection_pass_epochs:', a['raw_global_improvement_epochs'],
              a['raw_early_improvement_epochs'], a['joint_positive_epochs'], a['trajectory_epochs'] - a['protection_rejected_epochs'])
        for selector, counts in a['selector_reason_counts'].items():
            print(selector, json.dumps(counts))
    print('\nALL PROTECTION CHECKS: trajectory epochs counted once, failures overlap')
    for key, counts in t['protection_failure_counts'].items():
        print(key, json.dumps(counts))
    print('\nOVERLAPPING COHORT/REGION UNIONS: do not sum these counts')
    for key, counts in t['overlapping_group_unions'].items():
        print(key, json.dumps(counts))
    print('\nNONZERO PROTECTION CO-FAILURES: all epochs / joint-positive epochs')
    for pair in t['cofailures']:
        if pair['epochs']:
            print(pair['first'], '+', pair['second'], pair['epochs'], '/', pair['joint_positive_epochs'])
    print('\nEVERY JOINT-POSITIVE REJECTED EPOCH: selection only, raw MAE units')
    for item in summary['joint_positive_rejections']:
        print(item['fit'], 'epoch=', item['epoch'], 'global_gain=', item['global_gain'], 'early_gain=', item['early_gain'])
        for blocker in item['blockers']:
            print(' ', blocker['check'], 'harm=', blocker['harm_raw_mae'],
                  'excess_over_limit=', blocker['excess_over_decision_limit'], 'valid=', blocker['valid'])
    print('\nNO NEW TRAINING, EVALUATION OR MODEL SELECTION. Checkpoint/array files were not read.')
    print('Interpretation:', summary['interpretation'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    run_parser = sub.add_parser('run')
    run_parser.add_argument('--source-dir', type=Path, required=True)
    run_parser.add_argument('--output', type=Path, required=True)
    sub.add_parser('report').add_argument('summary', type=Path)
    args = parser.parse_args()
    if args.action == 'report':
        if not args.summary.is_file():
            parser.exit(1, 'INCOMPLETE: inspect status/log and preserve partial; retry with a new run name.\n')
        report(decode(args.summary.read_bytes()))
    else:
        run(args.source_dir, args.output)


if __name__ == '__main__':
    main()
