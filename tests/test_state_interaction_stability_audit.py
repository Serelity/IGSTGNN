"""v12h saved-source integrity, NumPy-only workflow and atomic publication."""

from contextlib import redirect_stdout
import copy
from datetime import date, timedelta
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experiments.chronological import audit_state_interaction_stability as e
from experiments.chronological import audit_state_interaction_transfer as transfer
from experiments.chronological import audit_architecture_regions as regions


def fixture(root):
    source = root / 'source'
    source.mkdir()
    protocol = copy.deepcopy(e.load_protocol())
    protocol['cohort_samples'] = {c: 12 if c == 'incident_full' else 6 for c in e.COHORTS}
    protocol['source_phase_samples'] = {
        'fit': {'incident_full': 2}, 'selection': dict(zip(e.COHORTS[:4], [2, 1, 1, 1])),
        'audit': dict(zip(e.COHORTS[:4], [12, 6, 6, 6]))}
    frozen = json.loads(e.SOURCE_PROTOCOL.read_text())
    summary = {'status': 'STATE_INTERACTION_TRANSFER_AUDIT_COMPLETE',
        'protocol_id': frozen['protocol_id'], 'protocol_sha256': protocol['source_protocol_sha256'],
        'source_protocol_sha256': protocol['source_v12f_protocol_sha256'],
        'source_git_head': protocol['source_v12f_git_head'], 'main_training_ready': False,
        'full_fit_evaluation_performed': False, 'recommendation': frozen['decision'],
        **frozen['information_boundary'], 'frozen_protocol': frozen,
        'phase_samples': protocol['source_phase_samples'], 'code_sha256': protocol['source_code_sha256'],
        'audit': {}, 'partition_accounting': {}, 'selection_trajectories': {}}
    ids = np.arange(101, 113)
    counts = np.asarray([[10, 2, 2, 6]]) * (1 + np.arange(12)[:, None] % 2)
    common_indices = np.arange(0, 12, 2)
    common = np.arange(12) % 2 == 0
    times = {int(sample): (date(2023, 7, 4) + timedelta(weeks=i % 9)).isoformat() + 'T12:00:00'
             for i, sample in enumerate(ids)}
    for seed in protocol['seeds']:
        gains = np.zeros((12, 4, 3))
        gains[:, 1] = .002 if seed == 2027 else 0.
        gains[:, 2] = np.asarray([np.where(common, -.001, -.025),
                                 np.full(12, .006), np.full(12, .010)]).T
        gains[:, 3] = np.asarray([np.where(common, .001, -.030),
                                 np.full(12, .007), np.full(12, .012)]).T
        errors = np.zeros((12, 4, 4))
        errors[:, :, 1:] = (10. - gains) * counts[:, None, 1:]
        errors[:, :, 0] = errors[:, :, 1:].sum(2)
        full = {'ids': ids, 'counts': counts, 'errors': errors, 'regions': list(e.REGIONS)}
        common_record = {key: value[common_indices] if isinstance(value, np.ndarray) else value
                         for key, value in full.items()}
        complement = {key: value[~common] if isinstance(value, np.ndarray) else value
                      for key, value in full.items()}
        records = {'incident_full': full, 'incident': copy.deepcopy(common_record),
                   'primary_control': copy.deepcopy(common_record),
                   'secondary_control': copy.deepcopy(common_record),
                   'incident_full_common': common_record, 'incident_full_complement': complement}
        for cohort in ('primary_control', 'secondary_control'):
            control = records[cohort]
            new_counts = control['counts'].copy()
            new_counts[:, 1] -= 1
            new_counts[:, 0] -= 1
            control['errors'][:, :, 1:] *= new_counts[:, None, 1:] / control['counts'][:, None, 1:]
            control['errors'][:, :, 0] = control['errors'][:, :, 1:].sum(2)
            control['counts'] = new_counts
        analysis, _, arrays = regions.analyze(records, times, {
            'paths': ['A', *protocol['arms']], 'comparisons': protocol['comparisons'],
            'bootstrap': protocol['bootstrap']})
        np.savez_compressed(source / f'audit_weekly_s{seed}.npz', **arrays)
        summary['audit'][str(seed)] = analysis
        summary['partition_accounting'][str(seed)] = transfer.partition_accounting(records)
        summary['selection_trajectories'][str(seed)] = {arm: {
            'epochs': 12, 'selected_epoch': 0 if arm == 'strength' and seed != 2027 else 1,
            'eligible_epochs': 12, 'cohorts': {cohort: {'selected_selection': {
                r: {'gain_vs_A_raw': 0. if arm == 'strength' else -.001} for r in e.REGIONS}}
                for cohort in ('incident_full', 'incident')},
        } for arm in protocol['arms']}
    (source / 'train_flow.npy').write_bytes(b'Unconsumed raw-data sentinel')
    def publish():
        summary['outputs'] = {path.name: e.sha256(path) for path in source.glob('*.npz')}
        (source / 'summary.json').write_text(json.dumps(summary, allow_nan=False))
    publish()
    return protocol, source, summary, publish


class StabilityAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.protocol, self.source, self.summary, self.publish = fixture(self.root)

    def run_audit(self, output):
        with patch.object(e, 'load_protocol', return_value=self.protocol), redirect_stdout(io.StringIO()):
            return e.run(self.source, output)

    def test_complete_read_only_saved_only_publication_and_report(self):
        before = {p.name: e.sha256(p) for p in self.source.iterdir()}
        output = self.root / 'output'
        summary = self.run_audit(output)
        self.assertEqual(summary['status'], 'STATE_INTERACTION_STABILITY_AUDIT_COMPLETE')
        self.assertEqual(set(summary['inputs']), {'summary.json', *[f'audit_weekly_s{s}.npz' for s in self.protocol['seeds']]})
        self.assertEqual(before, {p.name: e.sha256(p) for p in self.source.iterdir()})
        self.assertFalse(summary['model_loaded'])
        self.assertFalse(summary['manifests_read'])
        self.assertFalse(summary['absolute_error_sums_rechecked'])
        self.assertFalse(summary['original_traffic_input_hashes_rechecked'])
        self.assertFalse(output.with_name('output.partial').exists())
        for name, digest in summary['outputs'].items():
            self.assertEqual(e.sha256(output / name), digest)
        result = summary['audit']['2025']['common_minus_complement']['candidate_h1_h6']['state_vector_vs_A']
        self.assertAlmostEqual(result['pooled']['point_gain_raw_mae'], .024)
        capture = io.StringIO()
        with redirect_stdout(capture):
            e.report(summary)
        self.assertIn('common_minus_complement', capture.getvalue())
        self.assertIn('no new selection', capture.getvalue())
        json.dumps(summary, allow_nan=False)

    def test_status_budget_boundary_and_code_changes_are_rejected(self):
        mutations = (
            ('status', 'INCOMPLETE'), ('model_training_performed', True),
            ('independent_confirmation', True), ('full_fit_evaluation_performed', True),
            ('code_sha256', {}), ('phase_samples', {}),
            ('source_git_head', 'wrong-source-commit'),
        )
        original = copy.deepcopy(self.summary)
        for i, (key, value) in enumerate(mutations):
            with self.subTest(key=key):
                self.summary.clear()
                self.summary.update(copy.deepcopy(original))
                self.summary[key] = value
                self.publish()
                with self.assertRaises(ValueError):
                    self.run_audit(self.root / f'bad_metadata_{i}')

    def test_consumed_hash_failure_is_atomic_and_preserves_source(self):
        path = self.source / 'audit_weekly_s2025.npz'
        path.write_bytes(b'corrupted saved statistics')
        output = self.root / 'bad_hash'
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            self.run_audit(output)
        self.assertFalse(output.exists())
        self.assertTrue((self.root / 'bad_hash.partial/failure.json').is_file())
        self.assertEqual(path.read_bytes(), b'corrupted saved statistics')

    def test_summary_interval_and_accounting_corruption_are_rejected(self):
        effect = self.summary['audit']['2025']['results']['incident_full']['regions']['candidate_h1_h6']['comparisons']['state_vector_vs_A']
        effect['intervals']['week']['pooled']['valid_draws'] -= 1
        self.publish()
        with self.assertRaisesRegex(ValueError, 'valid_draws'):
            self.run_audit(self.root / 'bad_interval')
        effect['intervals']['week']['pooled']['valid_draws'] += 1
        item = self.summary['partition_accounting']['2025']['state_vector_vs_A']['groups']['incident_full_complement']
        item['contribution_to_full_global_gain']['candidate_h1_h6'] += .1
        self.publish()
        with self.assertRaisesRegex(ValueError, 'global contribution'):
            self.run_audit(self.root / 'bad_accounting')

    def test_calendar_and_cross_seed_saved_weights_cannot_change(self):
        path = self.source / 'audit_weekly_s2026.npz'
        with np.load(path, allow_pickle=False) as stored:
            original = {key: stored[key].copy() for key in stored.files}
        arrays = copy.deepcopy(original)
        arrays['weeks'] = arrays['weeks'][::-1]
        np.savez_compressed(path, **arrays)
        self.publish()
        with self.assertRaisesRegex(ValueError, 'calendar axis'):
            self.run_audit(self.root / 'bad_axis')
        arrays = copy.deepcopy(original)
        arrays['bootstrap_week_weights'] = arrays['bootstrap_week_weights'][::-1]
        np.savez_compressed(path, **arrays)
        self.publish()
        with self.assertRaisesRegex(ValueError, 'across seeds'):
            self.run_audit(self.root / 'bad_pairing')

    def test_no_overwrite_dangling_links_or_source_nested_output(self):
        with self.assertRaisesRegex(ValueError, 'outside'):
            self.run_audit(self.source / 'nested')
        for name in ('existing', 'existing.partial'):
            (self.root / name).mkdir()
        with self.assertRaises(FileExistsError):
            self.run_audit(self.root / 'existing')
        link = self.root / 'dangling'
        link.symlink_to(self.root / 'missing-target')
        with self.assertRaises(FileExistsError):
            self.run_audit(link)
        self.assertTrue(link.is_symlink())
        self.assertFalse((self.root / 'missing-target').exists())

    def test_fixed_evaluation_support_cannot_change_across_seeds(self):
        path = self.source / 'audit_weekly_s2026.npz'
        with np.load(path, allow_pickle=False) as stored:
            original = {key: stored[key].copy() for key in stored.files}
        for field in ('valid_counts', 'evaluable_events'):
            with self.subTest(field=field):
                arrays = copy.deepcopy(original)
                arrays[f'primary_control_{field}'][0, 0] += 1
                np.savez_compressed(path, **arrays)
                self.publish()
                with self.assertRaisesRegex(ValueError, 'support differs across seeds'):
                    self.run_audit(self.root / f'changed_{field}')

    def test_numpy_only_import_and_missing_report(self):
        code = '''import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(('torch', 'pandas', 'scipy')):
            raise AssertionError('Forbidden heavy dependency: '+fullname)
sys.meta_path.insert(0, Block())
from experiments.chronological import audit_state_interaction_stability
'''
        result = subprocess.run([sys.executable, '-c', code], cwd=e.REPO, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        result = subprocess.run([sys.executable, str(Path(e.__file__)), 'report', str(self.root / 'absent.json')],
                                cwd=e.REPO, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('INCOMPLETE', result.stdout)


if __name__ == '__main__':
    unittest.main()
