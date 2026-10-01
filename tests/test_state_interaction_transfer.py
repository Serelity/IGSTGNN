"""Saved-only v12g provenance, composition accounting and temporal diagnostics."""

from contextlib import redirect_stdout
import copy
import csv
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experiments.chronological import audit_state_interaction_transfer as e


def fixture(root):
    source, data = root / 'source', root / 'data'
    source.mkdir()
    data.mkdir()
    p = copy.deepcopy(e.load_protocol())
    p['node_count'] = 4
    p['source_phase_samples'] = {
        'fit': {'incident_full': 2},
        'selection': dict(zip(e.COHORTS, [2, 1, 1, 1])),
        'audit': dict(zip(e.COHORTS, [4, 2, 2, 2]))}
    inherited = json.loads(e.INHERITED_PROTOCOL.read_text())
    days = ['2023-01-10', '2023-01-17', '2023-05-16', '2023-05-23',
            '2023-07-04', '2023-07-11', '2023-07-18', '2023-07-25']
    ids = [11, 73, 22, 98, 15, 123, 4, 56]
    rows = [{'sample_index': s, 'incident_id': 900 + i // 2, 'split': 'train',
             't0': day + 'T12:00:00', 'support_start': day + 'T11:00:00',
             'support_end_exclusive': day + 'T14:00:00'} for i, (s, day) in enumerate(zip(ids, days))]
    with (data / 'train_manifest.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plan, cohort_positions = {}, {}
    matched_positive_positions = [2, 4, 6]
    primary_source_positions = {2: 40, 4: 30, 6: 20}
    for phase, indices, matched in (('fit', [0, 1], []), ('selection', [2, 3], [2]),
                                    ('audit', [4, 5, 6, 7], [4, 6])):
        groups = {'incident_full': indices}
        cohort_positions[phase] = {'incident_full': indices}
        if phase != 'fit':
            groups.update({c: [matched_positive_positions.index(i) for i in matched] for c in e.COHORTS[1:]})
            cohort_positions[phase].update({c: matched for c in e.COHORTS[1:]})
        plan[phase] = {'bounds': inherited['periods'][phase], 'indices': groups,
                       'positive_ids': [ids[i] for i in indices], 'matched_ids': [ids[i] for i in matched]}
    inputs = {str(data / 'train_manifest.csv'): e.sha256(data / 'train_manifest.csv')}
    identity = {'protocol_sha256': p['source_protocol_sha256'], 'engineering_check': False,
                'inputs': inputs, 'code_sha256': p['source_code_sha256'],
                'indices': {phase: value['indices'] for phase, value in plan.items()},
                'probe_indices': [0, 1], 'checkpoint_sha256': 'a' * 64}
    def save_json(name, value):
        (source / name).parent.mkdir(parents=True, exist_ok=True)
        (source / name).write_text(json.dumps(value))
    save_json('run_identity.json', identity)
    save_json('eligibility.json', plan)
    def record(indices, rate=None, control=False, cohort='incident_full'):
        mask = np.tile([True, True, False, False], (len(indices), 1))
        predicted = np.tile([48, 12, 12, 24], (len(indices), 1)).astype(np.int64)
        counts = predicted.copy()
        if control:
            counts[:, 1] -= 1
            counts[:, 0] -= 1
        rates = np.full((len(indices), 3), 10.) if rate is None else np.asarray(rate)
        errors = np.zeros_like(counts, dtype=float)
        errors[:, 1:] = rates * counts[:, 1:]
        errors[:, 0] = errors[:, 1:].sum(1)
        sources = ([primary_source_positions[i] for i in indices] if cohort == 'primary_control' else
                   [matched_positive_positions.index(i) for i in indices] if cohort == 'secondary_control' else indices)
        return {'ids': np.asarray([ids[i] for i in indices]), 'source_indices': np.asarray(sources),
                'candidate_mask': mask, 'errors': errors, 'counts': counts,
                'prediction_counts': predicted, 'regions': list(e.REGIONS)}
    def save_record(name, r):
        (source / name).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(source / name, **r)
    baseline, reference = {}, {}
    for phase in ('selection', 'audit'):
        reference[phase] = {}
        for cohort in e.COHORTS:
            r = record(cohort_positions[phase][cohort], control=cohort in e.COHORTS[2:], cohort=cohort)
            reference[phase][cohort] = r
            save_record(f'{phase}_A_{cohort}.npz', r)
            if phase == 'selection':
                baseline[cohort] = e.metrics(r)
    summary = {'status': 'INCIDENT_STATE_INTERACTION_COMPARISON_COMPLETE',
        'protocol_sha256': p['source_protocol_sha256'], 'engineering_check': False,
        'model_training_performed': True, 'validation_arrays_read': False,
        'test_split_read': False, 'independent_confirmation': False,
        'vector_paired_initialization_exact': True,
        'frozen_protocol': json.loads(e.SOURCE_PROTOCOL.read_text()), 'inherited_protocol': inherited,
        'effective_training': inherited['training'], 'phase_samples': p['source_phase_samples'],
        'inputs': inputs, 'code_sha256': p['source_code_sha256'],
        'baseline_audit': {c: e.metrics(r) for c, r in reference['audit'].items()},
        'environment': {'git_head': 'synthetic-fixture'}, 'runs': {}}
    for seed in p['seeds']:
        summary['runs'][str(seed)] = {}
        for arm in e.ARMS:
            directory = f'{arm}_s{seed}'
            learned = arm != 'strength'
            current = copy.deepcopy(baseline)
            if learned:
                for cohort in e.COHORTS:
                    r = record(cohort_positions['selection'][cohort],
                               rate=np.tile([10.003, 10., 9.997], (len(cohort_positions['selection'][cohort]), 1)),
                               control=cohort in e.COHORTS[2:], cohort=cohort)
                    current[cohort] = e.metrics(r)
            history = [{'epoch': epoch,
                'training': {'mae_standardized': .1, 'maximum_gradient_norm': .01,
                             'gate_parameters_changed': True, 'optimizer_steps': 1},
                'selection': copy.deepcopy(current), 'eligible': True,
                'protection_checks': {f'{c}/{r}': True for c in e.COHORTS for r in e.REGIONS},
                'best_epoch': 1 if learned else 0} for epoch in range(1, 13)]
            save_json(f'{directory}/history.json', history)
            audit = {}
            for cohort in e.COHORTS:
                indices = cohort_positions['audit'][cohort]
                rates = []
                for index in indices:
                    rates.append([10.003 if index in [4, 6] else 10.020, 10., 9.998]
                                 if cohort in e.COHORTS[:2] else [9.99, 10.004, 9.999])
                r = record(indices, rates if learned else None, control=cohort in e.COHORTS[2:], cohort=cohort)
                save_record(f'{directory}/audit_{cohort}.npz', r)
                audit[cohort] = e.metrics(r)
            probes = {}
            for label in ('initial', 'last', 'selected'):
                r = record([0, 1], np.tile([9.9, 9.95, 9.999], (2, 1))
                           if label != 'initial' and learned else None)
                save_record(f'{directory}/representation_{label}.npz', r)
                probes[label] = {**e.metrics(r), 'sample_ids': ids[:2], 'candidate_representation': {}}
            summary['runs'][str(seed)][arm] = {
                'selected_epoch': 1 if learned else 0, 'optimizer_steps': 12,
                'initial_prediction_exactly_A': True, 'backbone_state_unchanged': True,
                'selection_metrics': current, 'audit': audit, 'representation_diagnostics': probes}
    def publish():
        summary['outputs'] = {str(path.relative_to(source)): e.sha256(path)
                              for path in source.rglob('*') if path.is_file() and path.name != 'summary.json'}
        save_json('summary.json', summary)
    publish()
    return p, source, data, summary, publish


class TransferTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.protocol, self.source, self.data, self.summary, self.publish = fixture(self.root)

    def run_audit(self, output):
        with patch.object(e, 'load_protocol', return_value=self.protocol), redirect_stdout(io.StringIO()):
            return e.run(self.source, self.data, output)

    def test_frozen_protocol_and_numpy_only_import(self):
        self.assertEqual(e.load_protocol()['arms'], list(e.ARMS))
        code = '''import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(('torch', 'scipy', 'pandas')):
            raise AssertionError('Forbidden heavy dependency: '+fullname)
sys.meta_path.insert(0, Block())
from experiments.chronological import audit_state_interaction_transfer
'''
        result = subprocess.run([sys.executable, '-c', code], cwd=e.REPO, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_complete_read_only_workflow_and_exact_group_accounting(self):
        before = {str(p): e.sha256(p) for p in self.source.rglob('*') if p.is_file()}
        output = self.root / 'output'
        summary = self.run_audit(output)
        self.assertEqual(summary['status'], 'STATE_INTERACTION_TRANSFER_AUDIT_COMPLETE')
        self.assertFalse(summary['model_loaded'])
        self.assertFalse(summary['full_fit_evaluation_performed'])
        self.assertEqual(before, {str(p): e.sha256(p) for p in self.source.rglob('*') if p.is_file()})
        for seed in map(str, self.protocol['seeds']):
            analysis = summary['audit'][seed]
            self.assertEqual(len(analysis['weeks']), 4)
            self.assertEqual(analysis['results']['incident_full_common']['samples'], 2)
            self.assertEqual(analysis['results']['incident_full_complement']['samples'], 2)
            effect = analysis['results']['incident_full_complement']['regions']['candidate_h1_h6']['comparisons']['state_vector_vs_A']
            self.assertAlmostEqual(effect['gain_raw_mae'], -.020)
            whole = summary['partition_accounting'][seed]['state_vector_vs_A']
            self.assertAlmostEqual(whole['full_global_gain_raw_mae'], whole['reconstructed_full_global_gain_raw_mae'])
            self.assertAlmostEqual(sum(g['contribution_to_full_global_gain']['all'] for g in whole['groups'].values()),
                                   whole['full_global_gain_raw_mae'])
            self.assertEqual(summary['selection_trajectories'][seed]['state_vector']['selected_epoch'], 1)
            self.assertAlmostEqual(summary['fit_probe'][seed]['state_vector']['selected']['gain_vs_initial_probe_raw_mae']['candidate_h1_h6'], .1)
        for filename, digest in summary['outputs'].items():
            self.assertEqual(e.sha256(output / filename), digest)
        with (output / 'audit_weekly.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        empty = [r for r in rows if r['cohort'] == 'incident_full_common' and r['positive_week'] == '2023-W28']
        self.assertTrue(empty)
        self.assertTrue(all(r['valid_cells'] == '0' and r['gain_vs_A_raw_mae'] == '' for r in empty))
        self.assertTrue(all(r['evaluable_forecast_windows'] == '0' and
                            r['equal_forecast_window_gain_vs_A_raw_mae'] == '' for r in empty))
        json.dumps(summary, allow_nan=False)
        with self.assertRaises(FileExistsError):
            self.run_audit(output)
        with self.assertRaisesRegex(ValueError, 'outside'):
            self.run_audit(self.source / 'nested')

    def test_source_hash_protocol_budget_and_manifest_changes_are_rejected(self):
        (self.source / 'strength_s2025/history.json').write_text('[]')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            self.run_audit(self.root / 'bad_hash')
        self.publish()
        with self.assertRaisesRegex(ValueError, 'every configured epoch'):
            self.run_audit(self.root / 'bad_history')
        self.summary['engineering_check'] = True
        self.publish()
        with self.assertRaisesRegex(ValueError, 'full-budget'):
            self.run_audit(self.root / 'engineering')
        self.summary['engineering_check'] = False
        self.publish()
        with (self.data / 'train_manifest.csv').open('a') as stream:
            stream.write('\n')
        with self.assertRaisesRegex(ValueError, 'fingerprint'):
            self.run_audit(self.root / 'bad_manifest')

    def test_array_identity_partitions_and_selected_metrics_are_validated(self):
        path = self.source / 'state_vector_s2025/audit_incident_full.npz'
        with np.load(path, allow_pickle=False) as stored:
            record = {k: stored[k].copy() for k in stored.files}
        original = copy.deepcopy(record)
        record['ids'] = record['ids'][::-1]
        np.savez_compressed(path, **record)
        self.publish()
        with self.assertRaisesRegex(ValueError, 'identities'):
            self.run_audit(self.root / 'wrong_order')
        record = copy.deepcopy(original)
        record['errors'][0, 0] += 1
        np.savez_compressed(path, **record)
        self.publish()
        with self.assertRaisesRegex(ValueError, 'partition'):
            self.run_audit(self.root / 'wrong_partition')
        np.savez_compressed(path, **original)
        self.summary['runs']['2025']['state_vector']['audit']['incident_full']['mae']['all'] += .1
        self.publish()
        with self.assertRaisesRegex(ValueError, 'audit/incident_full/all'):
            self.run_audit(self.root / 'wrong_metric')

    def test_complete_historical_code_identity_is_required(self):
        original = copy.deepcopy(self.summary['code_sha256'])
        for index, changed in enumerate((
                {**original, 'src/utils/chronological.py': '0' * 64},
                {k: v for k, v in original.items() if k != 'src/models/igstgnn.py'},
                {**original, 'src/models/new_unfrozen_model.py': '0' * 64})):
            self.summary['code_sha256'] = changed
            self.publish()
            with self.assertRaisesRegex(ValueError, 'code identity'):
                self.run_audit(self.root / f'wrong_code_{index}')

    def test_saved_source_positions_use_cohort_specific_coordinates(self):
        source = e.Source(self.source, self.protocol)
        actual = {cohort: e.read_record(source, f'audit_A_{cohort}.npz', source.plan['audit']['matched_ids'])
                  ['source_indices'].tolist() for cohort in e.COHORTS[1:]}
        self.assertEqual(actual['incident'], [4, 6])
        self.assertEqual(actual['primary_control'], [30, 20])
        self.assertEqual(actual['secondary_control'], [1, 2])
        self.assertEqual(source.plan['audit']['indices']['incident'], [1, 2])
        self.run_audit(self.root / 'different_source_coordinates')
        path = self.source / 'audit_A_secondary_control.npz'
        with np.load(path, allow_pickle=False) as stored:
            record = {k: stored[k].copy() for k in stored.files}
        record['source_indices'][0] += 1
        np.savez_compressed(path, **record)
        self.publish()
        with self.assertRaisesRegex(ValueError, 'secondary_control source indices'):
            self.run_audit(self.root / 'wrong_control_position')

    def test_weekly_cell_and_equal_window_weighting_are_distinct(self):
        source = e.Source(self.source, self.protocol)
        full = e.read_record(source, 'audit_A_incident_full.npz', source.plan['audit']['positive_ids'])
        full['counts'][1] //= 2
        full['errors'][1] /= 2
        full['counts'][3] = 0
        full['errors'][3] = 0
        common = e.restrict(full, [0, 2])
        reference = {'incident_full': full, 'incident': common,
                     'primary_control': copy.deepcopy(common), 'secondary_control': copy.deepcopy(common)}
        learned = {arm: copy.deepcopy(reference) for arm in e.ARMS}
        modified = learned['state_vector']['incident_full']
        modified['errors'][:2, 1] -= modified['counts'][:2, 1] * [1., 3.]
        modified['errors'][:, 0] = modified['errors'][:, 1:].sum(1)
        learned['state_vector']['incident'] = e.restrict(modified, [0, 2])
        records, _ = e.audit_records(reference, learned, self.protocol['matched_replay_error_tolerance'])
        days = [4, 4, 18, 25]
        times = {int(s): f'2023-07-{day:02d}T12:00:00' for s, day in zip(full['ids'], days)}
        rows = e.weekly_rows(records, times, 2025)
        early = [r for r in rows if r['cohort'] == 'incident_full' and
                 r['region'] == 'candidate_h1_h6' and r['arm'] == 'state_vector']
        first = next(r for r in early if r['positive_week'] == '2023-W27')
        self.assertEqual(first['evaluable_forecast_windows'], 2)
        self.assertAlmostEqual(first['gain_vs_A_raw_mae'], 5 / 3)
        self.assertAlmostEqual(first['equal_forecast_window_gain_vs_A_raw_mae'], 2.)
        last = next(r for r in early if r['positive_week'] == '2023-W30')
        self.assertEqual(last['forecast_windows'], 1)
        self.assertEqual(last['evaluable_forecast_windows'], 0)
        self.assertIsNone(last['equal_forecast_window_gain_vs_A_raw_mae'])

    def test_sparse_empty_groups_keep_full_calendar_and_explicit_undefined_intervals(self):
        source = e.Source(self.source, self.protocol)
        full = e.read_record(source, 'audit_A_incident_full.npz', source.plan['audit']['positive_ids'])
        common = e.restrict(full, [0])
        control = copy.deepcopy(common)
        reference = {'incident_full': full, 'incident': common,
                     'primary_control': control, 'secondary_control': control}
        learned = {arm: copy.deepcopy(reference) for arm in e.ARMS}
        records, _ = e.audit_records(reference, learned, self.protocol['matched_replay_error_tolerance'])
        records['incident_full_complement'] = e.restrict(records['incident_full_complement'], [])
        times = {int(s): f'2023-07-{4 + 7*i:02d}T12:00:00' for i, s in enumerate(full['ids'])}
        analysis, _, arrays = e.regions_audit.analyze(records, times, {
            'paths': ['A', *e.ARMS], 'comparisons': e.comparisons(),
            'bootstrap': source.inherited['bootstrap']})
        self.assertEqual(len(analysis['weeks']), 4)
        item = analysis['results']['incident_full_complement']['regions']['all']['comparisons']['state_vector_vs_A']
        self.assertIsNone(item['gain_raw_mae'])
        self.assertEqual(item['intervals']['week']['pooled']['status'], 'INSUFFICIENT_VALID_DRAWS')
        self.assertEqual(arrays['incident_full_complement_valid_counts'].sum(), 0)


if __name__ == '__main__':
    unittest.main()
