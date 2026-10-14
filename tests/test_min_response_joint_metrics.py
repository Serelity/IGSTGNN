"""Committed joint reports must preserve original/current OFF distinctions."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from experiments.chronological.min_response_metrics import summarize_response_pair
from experiments.chronological.min_response_joint_metrics import ARMS, SCHEMA, STAGE, write_diagnostics


class JointResponseMetricsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1])
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def write_json(self, path, payload):
        Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False), encoding='utf-8')

    def read_json(self, path):
        return json.loads(Path(path).read_text(encoding='utf-8'))

    def digest(self, path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    def arm(self, name='state', shift=.5):
        path = self.root/name
        path.mkdir()
        target = np.tile(np.array([2., 300.])[None, None], (2, 12, 1))
        original = target+1.
        current = target+.7
        on = target+shift
        common = dict(target=target, valid=np.ones_like(target, dtype=bool),
            sample_indices=np.array([3909, 3910]), fixed_hard_indices=np.array([0]),
            history_low_mask=np.array([[True, False], [False, True]]))
        np.savez_compressed(path/'baseline_predictions.npz', prediction=original, **common)
        np.savez_compressed(path/'best_predictions.npz', on=on, off=original,
            original_off=original, current_off=current, **common)
        def pair(a, b):
            return summarize_response_pair(a, b, target, common['valid'], common['fixed_hard_indices'],
                                           common['history_low_mask'])
        zero = pair(original, original)
        zero['current_off'] = pair(original, original)
        selected = pair(on, original)
        selected['current_off'] = pair(on, current)
        history = [dict(epoch=0, validation=zero),
                   dict(epoch=1, validation=selected, training={'train_order_sha256': 'same_order'})]
        identity = dict(schema=SCHEMA, training_stage=STAGE, arm=name,
                        source_state_sha256='immutable_source', source_MIN_training_updates=0)
        checkpoint = dict(schema=SCHEMA, identity=identity, completed_epoch=1, history=history,
            best_epoch=1, best_metric=selected['on']['mae_pooled'], source_state_sha256='immutable_source',
            baseline_sha256=self.digest(path/'baseline_predictions.npz'))
        torch.save(checkpoint, path/'last.pt')
        summary = dict(schema=SCHEMA, training_stage=STAGE, arm=name, status='JOINT_COMPLETE',
            source_MIN_training_updates=0, completed_epoch=1, best_epoch=1,
            best_mae=checkpoint['best_metric'], source_state_sha256='immutable_source',
            source_reference_preserved=True, current_OFF_exactly_matches_original=False,
            trainable_parameters=100, engineering_only=False, test_evaluated=False,
            baseline_predictions_sha256=self.digest(path/'baseline_predictions.npz'),
            best_predictions_sha256=self.digest(path/'best_predictions.npz'))
        self.write_json(path/'summary.json', summary)
        return path

    def change_archive(self, path, **changes):
        with np.load(path/'best_predictions.npz', allow_pickle=False) as old:
            archive = {key: old[key].copy() for key in old.files}
        archive.update(changes)
        np.savez_compressed(path/'best_predictions.npz', **archive)
        summary = self.read_json(path/'summary.json')
        summary['best_predictions_sha256'] = self.digest(path/'best_predictions.npz')
        self.write_json(path/'summary.json', summary)

    def test_saved_report_separates_original_reference_gain_from_path_sensitivity(self):
        self.arm('state', .6)
        self.arm('location', .5)
        report = write_diagnostics(self.root)
        self.assertEqual(report['status'], 'COMPLETE')
        self.assertEqual(report['source_MIN_training_updates'], 0)
        row = report['runs']['location']
        self.assertAlmostEqual(row['original_reference_pair']['mae_gain_pooled'], .5)
        self.assertAlmostEqual(row['current_off_pair']['mae_gain_pooled'], .2)
        self.assertFalse(row['current_OFF_exactly_matches_original'])
        self.assertAlmostEqual(report['selected_best_location_vs_state']['pair']['mae_gain_pooled'], .1)
        self.assertFalse(report['test_evaluated'])
        self.assertTrue((self.root/'report.md').is_file())
        self.assertEqual(set(report['runs']), set(ARMS))
        json.dumps(report, allow_nan=False)

    def test_rejects_new_MIN_training_arm_and_source_updates(self):
        with self.assertRaises(ValueError):
            write_diagnostics(self.root, arms=('minimal',))
        path = self.arm()
        summary = self.read_json(path/'summary.json')
        summary['source_MIN_training_updates'] = 1
        self.write_json(path/'summary.json', summary)
        with self.assertRaisesRegex(ValueError, 'without a source update'):
            write_diagnostics(self.root, arms=('state',))

    def test_changed_archive_hash_is_rejected_before_reporting(self):
        path = self.arm()
        with (path/'best_predictions.npz').open('ab') as stream:
            stream.write(b'uncommitted_tail')
        with self.assertRaisesRegex(ValueError, 'archive hash changed'):
            write_diagnostics(self.root, arms=('state',))

    def test_rehashed_validity_or_history_mask_drift_is_rejected(self):
        for key in ('valid', 'history_low_mask'):
            with self.subTest(key=key):
                # Separate arms keep both integrity attacks independent.
                arm = 'state' if key == 'valid' else 'location'
                path = self.arm(arm)
                with np.load(path/'best_predictions.npz', allow_pickle=False) as old:
                    changed = old[key].copy()
                changed.flat[0] = ~changed.flat[0]
                self.change_archive(path, **{key: changed})
                with self.assertRaisesRegex(ValueError, 'targets, masks or sample axes changed'):
                    write_diagnostics(self.root, arms=(arm,))

    def test_current_off_cannot_replace_original_reference_even_after_rehash(self):
        path = self.arm()
        with np.load(path/'best_predictions.npz', allow_pickle=False) as old:
            current = old['current_off'].copy()
        self.change_archive(path, original_off=current, off=current)
        with self.assertRaisesRegex(ValueError, 'Original MIN predictions were replaced'):
            write_diagnostics(self.root, arms=('state',))

    def test_stale_summary_is_pending_not_a_completed_candidate(self):
        path = self.arm()
        summary = self.read_json(path/'summary.json')
        summary['completed_epoch'] = 0
        self.write_json(path/'summary.json', summary)
        report = write_diagnostics(self.root, arms=('state',))
        self.assertEqual(report['status'], 'PARTIAL')
        self.assertNotIn('state', report['runs'])
        self.assertEqual(report['pending']['state'], 'summary_needs_finalization_after_checkpoint_commit')

    def test_inconsistent_selection_or_archived_gain_record_is_rejected(self):
        path = self.arm()
        saved = torch.load(path/'last.pt', map_location='cpu', weights_only=False)
        saved['best_metric'] = -1.
        torch.save(saved, path/'last.pt')
        with self.assertRaisesRegex(ValueError, 'selection certificate'):
            write_diagnostics(self.root, arms=('state',))
        saved['best_metric'] = saved['history'][1]['validation']['on']['mae_pooled']
        saved['history'][1]['validation']['current_off']['paired']['mae_gain_pooled'] += 1.
        torch.save(saved, path/'last.pt')
        with self.assertRaisesRegex(ValueError, 'archived gains differ'):
            write_diagnostics(self.root, arms=('state',))

    def test_equal_budget_candidates_must_use_identical_training_order(self):
        self.arm('state')
        path = self.arm('location')
        saved = torch.load(path/'last.pt', map_location='cpu', weights_only=False)
        saved['history'][1]['training']['train_order_sha256'] = 'different_order'
        torch.save(saved, path/'last.pt')
        with self.assertRaisesRegex(ValueError, 'training sample order changed'):
            write_diagnostics(self.root)


if __name__ == '__main__':
    unittest.main()
