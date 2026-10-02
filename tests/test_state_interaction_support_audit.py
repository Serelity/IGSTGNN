"""X-only observable features, exact saved-source lineage and atomic v12j output."""

from contextlib import redirect_stdout
import copy
import csv
from datetime import datetime, timedelta
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experiments.chronological import audit_state_interaction_support as e
from experiments.chronological.state_interaction_fit import analyze_phases, COMPARISONS
from test_state_interaction_transfer import fixture as source_fixture


def fixture(root):
    reader, source, data, original, publish_original = source_fixture(root)
    protocol = copy.deepcopy(e.load_protocol())
    rows = e.read_csv(data / 'train_manifest.csv')
    rows[1] = {key: value.replace('2023-01-17', '2023-02-07') if isinstance(value, str) else value
               for key, value in rows[1].items()}
    for row in rows:
        t0 = datetime.fromisoformat(row['t0'])
        row.update(source_version=8, report_time=(t0 - timedelta(minutes=3)).isoformat(),
                   x_start=(t0 - timedelta(minutes=60)).isoformat())
    with (data / 'train_manifest.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    station_ids = np.array([401, 402, 403, 404])
    flow = np.tile(np.arange(26, dtype=float)[None, :, None] + 100., (len(rows), 1, 4))
    flow[:, 14:] = np.nan  # Raw future is irrelevant; saved forecast errors supply outcomes.
    np.save(data / 'train_flow.npy', flow)
    np.save(data / 'station_ids.npy', station_ids)
    distances = np.zeros((len(rows), 4, 3))
    distances[:, :2, 0] = 1.
    np.savez_compressed(data / 'train_context.npz', sample_indices=[int(r['sample_index']) for r in rows],
        station_ids=station_ids, distances=distances, report_age_minutes=np.full(len(rows), 3.),
        forecast_tod=np.full(len(rows), 144), forecast_dow=[(datetime.fromisoformat(r['t0']).weekday()+1)%7 for r in rows])
    for name in ('train_manifest.csv', 'train_flow.npy', 'station_ids.npy', 'train_context.npz'):
        original['inputs'][str(data / name)] = e.sha256(data / name)
    identity = json.loads((source / 'run_identity.json').read_text())
    identity['inputs'] = original['inputs']
    (source / 'run_identity.json').write_text(json.dumps(identity))
    original['environment']['git_head'] = protocol['source_v12f_git_head']
    for seed, arms in original['runs'].items():
        for arm, detail in arms.items():
            detail['trainable_parameters'] = 1633 if arm == 'strength' else 4288
            detail['representation_diagnostics']['selected']['adapter_state_sha256'] = 'a' * 64
    publish_original()
    fit = root / 'fit'
    fit.mkdir()
    with np.load(source / 'strength_s2025/representation_initial.npz') as stored:
        a_fit = {name: stored[name].copy() for name in stored.files}
    a_fit['regions'] = a_fit['regions'].tolist()
    np.savez_compressed(fit / 'fit_A.npz', **a_fit)
    src = e.transfer.Source(source, reader)
    a_audit = e.transfer.read_record(src, 'audit_A_incident_full.npz', src.plan['audit']['positive_ids'])
    times = {phase: {sample: rows[i]['t0'] for sample, i in zip(item['positive_ids'], item['indices']['incident_full'])}
             for phase, item in src.plan.items() if phase in ('fit', 'audit')}
    results, models = {}, {}
    for seed in protocol['seeds']:
        learned = {'fit': {}, 'audit': {}}
        models[str(seed)] = {}
        for arm in protocol['source_arms']:
            record = copy.deepcopy(a_fit)
            if arm != 'strength':
                record['errors'][:, 1] -= record['counts'][:, 1] * .1
                record['errors'][:, 0] = record['errors'][:, 1:].sum(1)
            np.savez_compressed(fit / f'fit_{arm}_s{seed}.npz', **record)
            learned['fit'][arm] = record
            learned['audit'][arm] = e.transfer.read_record(src, f'{arm}_s{seed}/audit_incident_full.npz', src.plan['audit']['positive_ids'])
            detail = original['runs'][str(seed)][arm]
            models[str(seed)][arm] = {'selected_epoch': detail['selected_epoch'],
                'adapter_state_sha256': 'a' * 64, 'backbone_state_sha256': 'b' * 64,
                'adapter_parameters': detail['trainable_parameters'],
                'model_state_unchanged': True, 'full_fit_samples': 2,
                'epoch_zero_full_fit_error_sums_exact_A': True if detail['selected_epoch'] == 0 else None}
        records = {'fit': e.stack(a_fit, learned['fit']), 'audit': e.stack(a_audit, learned['audit'])}
        results[str(seed)] = analyze_phases(records, COMPARISONS, src.inherited['bootstrap'], times)[0]
    frozen = json.loads(e.FIT_PROTOCOL.read_text())
    summary = {'status': 'STATE_INTERACTION_FULL_FIT_AUDIT_COMPLETE', 'protocol_id': frozen['protocol_id'],
        'protocol_sha256': protocol['fit_protocol_sha256'], 'frozen_protocol': frozen,
        **frozen['information_boundary'], 'main_training_ready': False, 'recommendation': frozen['decision'],
        'source_protocol_sha256': original['protocol_sha256'], 'source_git_head': protocol['source_v12f_git_head'],
        'phase_samples': original['phase_samples'], 'environment': {'git_head': protocol['fit_git_head']},
        'code_sha256': {**original['code_sha256'], **protocol['fit_producer_code_sha256']},
        'selected_models': models, 'results': results,
        'inputs': {**original['inputs'], **src.hashes}}
    # v12i also consumed manifest; its hash uses the resolved identity path.
    summary['inputs'][str((data / 'train_manifest.csv').resolve())] = e.sha256(data / 'train_manifest.csv')
    def publish_fit():
        summary['outputs'] = {p.name: e.sha256(p) for p in fit.iterdir() if p.name != 'summary.json'}
        (fit / 'summary.json').write_text(json.dumps(summary, allow_nan=False))
    publish_fit()
    (data / 'val_flow.npy').write_bytes(b'Forbidden old validation sentinel')
    (data / 'test_flow.npy').write_bytes(b'Forbidden test sentinel')
    return protocol, reader, source, data, fit, original, summary, publish_original, publish_fit


class SupportAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.protocol, self.reader, self.source, self.data, self.fit, self.original,
         self.summary, self.publish_original, self.publish_fit) = fixture(self.root)

    def run_audit(self, name='output'):
        with patch.object(e, 'load_protocol', return_value=self.protocol), \
                patch.object(e.transfer, 'load_protocol', return_value=self.reader), redirect_stdout(io.StringIO()):
            return e.run(self.fit, self.source, self.data, self.root / name)

    def test_complete_saved_only_workflow_and_source_preservation(self):
        before = {str(p): e.sha256(p) for p in self.root.rglob('*') if p.is_file()}
        result = self.run_audit()
        self.assertEqual(result['status'], 'STATE_INTERACTION_OBSERVABLE_SUPPORT_AUDIT_COMPLETE')
        self.assertEqual(before, {str(p): e.sha256(p) for p in self.root.rglob('*') if str(p) in before})
        self.assertEqual(set(result['results']), {'2025', '2026', '2027'})
        for field in ('model_loaded', 'checkpoint_loaded', 'model_training_performed',
                      'new_model_selection_performed', 'validation_arrays_read', 'test_split_read',
                      'raw_train_future_values_accessed', 'raw_selection_history_values_read',
                      'automatic_development_gate', 'deployable_router_constructed'):
            self.assertFalse(result[field])
        output = self.root / 'output'
        self.assertFalse((self.root / 'output.partial').exists())
        self.assertEqual(set(result['outputs']), {'state_membership.csv', 'conditional_gains.csv',
                         'weekly_gains.csv', 'composition_accounting.csv'})
        for name, digest in result['outputs'].items():
            self.assertEqual(e.sha256(output / name), digest)
        self.assertEqual(result['selection_conditional_statistics_status'], 'UNAVAILABLE_SAVED_AGGREGATES_ONLY')
        json.dumps(result, allow_nan=False)
        capture = io.StringIO()
        with redirect_stdout(capture):
            e.report(result)
        self.assertIn('low_joint_support state_vector_vs_A', capture.getvalue())
        self.assertIn('early_composition', capture.getvalue())
        self.assertIn('NO NEW TRAINING', capture.getvalue())

    def test_numpy_only_import_and_protocol_freeze(self):
        self.assertEqual(e.load_protocol()['features'], list(e.FEATURE_NAMES))
        code = '''import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(('torch', 'scipy', 'pandas')):
            raise AssertionError('Forbidden dependency: '+fullname)
sys.meta_path.insert(0, Block())
from experiments.chronological import audit_state_interaction_support
'''
        run = subprocess.run([sys.executable, '-c', code], cwd=e.REPO, capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)

    def test_history_values_zero_missing_and_within_node_volatility(self):
        history = np.tile(np.arange(12., dtype=float)[:, None], (1, 3))
        history[:, 1] = 1000.  # Different station means do not inflate within-node volatility.
        history[:, 2] = np.nan
        got = e.history_features(history, np.array([True, True, False]), 2.)
        self.assertAlmostEqual(got['history_mean'], 502.75)
        self.assertAlmostEqual(got['history_trend'], 3.)
        self.assertAlmostEqual(got['history_volatility'], np.std(np.arange(12.))/2)
        self.assertEqual(got['history_missing_fraction'], 0.)
        history[:6, 0] = -1.
        got = e.history_features(history, np.array([True, False, False]), 2.)
        self.assertTrue(np.isnan(got['history_trend']))
        self.assertEqual(got['history_missing_fraction'], .5)
        got = e.history_features(history, np.array([False, False, False]), 2.)
        self.assertTrue(np.isnan(got['history_mean']))
        self.assertEqual(got['candidate_node_count'], 0.)

    def test_feature_extraction_indexes_only_x_and_no_selection(self):
        source = e.transfer.Source(self.source, self.reader)
        fit = e.FitSource(self.fit, source, self.protocol)
        records = {'fit': e.transfer.read_record(fit, 'fit_A.npz', source.plan['fit']['positive_ids']),
                   'audit': e.transfer.read_record(source, 'audit_A_incident_full.npz', source.plan['audit']['positive_ids'])}
        original_load = np.load
        seen = []
        class XOnly:
            shape, dtype = (8, 26, 4), np.dtype('float64')
            def __getitem__(self, index):
                i, slots, nodes = index
                if i not in (0, 1, 4, 5, 6, 7) or slots != slice(None, 12) or nodes != slice(None):
                    raise AssertionError('Future or selection history access')
                seen.append(i)
                return np.full((12, 4), 12., dtype=float)
        def guarded(path, **kw):
            if Path(path).name == 'train_flow.npy':
                self.assertEqual(kw['mmap_mode'], 'r')
                return XOnly()
            return original_load(path, **kw)
        with patch.object(e.np, 'load', side_effect=guarded):
            features, times, _ = e.read_features(self.data, source, records)
        self.assertEqual(seen, [0, 1, 4, 5, 6, 7])
        self.assertEqual(features['fit']['history_mean'].tolist(), [12., 12.])

    def test_source_lineage_hash_and_code_fail_atomically(self):
        self.summary['inputs']['summary.json'] = '0' * 64
        self.publish_fit()
        with self.assertRaisesRegex(ValueError, 'source chain'):
            self.run_audit('bad_lineage')
        self.assertFalse((self.root / 'bad_lineage').exists())
        self.assertTrue((self.root / 'bad_lineage.partial/failure.json').is_file())
        self.summary['inputs']['summary.json'] = e.sha256(self.source / 'summary.json')
        self.summary['code_sha256'] = {**self.summary['code_sha256'],
            'experiments/chronological/state_interaction_fit.py': '0' * 64}
        self.publish_fit()
        with self.assertRaisesRegex(ValueError, 'code identity'):
            self.run_audit('bad_code')

    def test_input_and_array_corruption_rejected(self):
        original_bytes = (self.data / 'train_flow.npy').read_bytes()
        with (self.data / 'train_flow.npy').open('ab') as stream:
            stream.write(b'corruption')
        with self.assertRaisesRegex(ValueError, 'fingerprint'):
            self.run_audit('bad_input')
        (self.data / 'train_flow.npy').write_bytes(original_bytes)
        # A wrong saved array still fails its own artifact identity before any features.
        (self.fit / 'fit_A.npz').write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'Fit artifact hash mismatch'):
            self.run_audit('bad_array')

    def test_republished_mask_index_count_and_metric_corruption_is_rejected(self):
        path = self.fit / 'fit_state_vector_s2025.npz'
        with np.load(path, allow_pickle=False) as stored:
            original = {key: stored[key].copy() for key in stored.files}
        for index, field in enumerate(('candidate_mask', 'source_indices', 'counts', 'errors')):
            record = copy.deepcopy(original)
            if field == 'candidate_mask':
                record[field][:, [0, 2]] = record[field][:, [2, 0]]
            elif field == 'source_indices':
                record[field] = record[field][::-1]
            elif field == 'counts':
                record[field][0, 1] -= 1
                record[field][0, 0] -= 1
            else:
                record[field][:, 1] += .1
                record[field][:, 0] += .1
            np.savez_compressed(path, **record)
            self.publish_fit()
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'alignment|disagrees'):
                self.run_audit('bad_republished_' + str(index))
        np.savez_compressed(path, **original)
        self.publish_fit()

    def test_selected_backbone_epoch_and_fallback_metadata_are_checked(self):
        metadata = self.summary['selected_models']['2025']['strength']
        original = copy.deepcopy(metadata)
        for index, (name, value) in enumerate((('backbone_state_sha256', 'c' * 64),
                ('selected_epoch', 13), ('epoch_zero_full_fit_error_sums_exact_A', False))):
            self.summary['selected_models']['2025']['strength'] = {**original, name: value}
            self.publish_fit()
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'backbone|metadata'):
                self.run_audit('bad_metadata_' + str(index))

    def test_existing_nested_partial_and_symlink_guards(self):
        self.run_audit()
        with self.assertRaises(FileExistsError):
            self.run_audit()
        with patch.object(e, 'load_protocol', return_value=self.protocol):
            for parent in (self.fit, self.source, self.data):
                with self.assertRaisesRegex(ValueError, 'outside'):
                    e.run(self.fit, self.source, self.data, parent / 'nested')
            alias = self.root / 'alias'
            alias.symlink_to(self.fit, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, 'completed'):
                e.run(alias, self.source, self.data, self.root / 'bad_alias')
            dangling = self.root / 'dangling'
            dangling.symlink_to(self.root / 'absent')
            with self.assertRaises(FileExistsError):
                e.run(self.fit, self.source, self.data, dangling)
        capture = io.StringIO()
        with redirect_stdout(capture), patch.object(sys, 'argv', ['audit', 'report', str(self.root / 'absent.json')]):
            e.main()
        self.assertIn('INCOMPLETE', capture.getvalue())


if __name__ == '__main__':
    unittest.main()
