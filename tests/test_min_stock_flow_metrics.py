"""The new decoder's condition OFF must never masquerade as source MIN."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from experiments.chronological.min_response_metrics import summarize_response_pair
from experiments.chronological.min_stock_flow_metrics import SCHEMA, STAGE, write_diagnostics


class StockFlowMetricsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1])
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def fixture(self, arm, shift):
        path = self.root/arm
        path.mkdir()
        target = np.tile(np.array([2., 300.])[None, None], (2, 12, 1))
        original, current, on = target+1., target+.7, target+shift
        common = dict(target=target, valid=np.ones_like(target, dtype=bool),
            sample_indices=np.array([3909, 3910]), fixed_hard_indices=np.array([0]),
            history_low_mask=np.array([[True, False], [False, True]]))
        np.savez_compressed(path/'baseline_predictions.npz', prediction=original, **common)
        np.savez_compressed(path/'best_predictions.npz', on=on, current_off=current,
            original_off=original, off=original, **common)
        pair = lambda a, b: summarize_response_pair(a, b, target, common['valid'],
            common['fixed_hard_indices'], common['history_low_mask'])
        zero = pair(target+2., original)
        zero['current_off'] = pair(target+2., target+2.)
        best = pair(on, original)
        best['current_off'] = pair(on, current)
        history = [dict(epoch=0, validation=zero), dict(epoch=1, validation=best,
                   training={'train_order_sha256': 'same'})]
        identity = dict(schema=SCHEMA, training_stage=STAGE, arm=arm,
                        source_MIN_training_updates=0, source_state_sha256='old_MIN')
        digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
        saved = dict(schema=SCHEMA, identity=identity, completed_epoch=1, best_epoch=1,
            best_metric=best['on']['mae_pooled'], history=history, source_state_sha256='old_MIN',
            baseline_sha256=digest(path/'baseline_predictions.npz'))
        torch.save(saved, path/'last.pt')
        summary = dict(schema=SCHEMA, training_stage=STAGE, arm=arm, status='PASS',
            source_MIN_training_updates=0, source_reference_preserved=True,
            source_state_sha256='old_MIN', completed_epoch=1, best_epoch=1,
            best_mae=saved['best_metric'], trainable_parameters=100,
            engineering_only=False, test_evaluated=False,
            baseline_predictions_sha256=digest(path/'baseline_predictions.npz'),
            best_predictions_sha256=digest(path/'best_predictions.npz'))
        (path/'summary.json').write_text(json.dumps(summary), encoding='utf-8')
        return path

    def test_source_condition_off_and_equal_budget_are_separate(self):
        self.fixture('state', .6)
        self.fixture('location', .5)
        report = write_diagnostics(self.root)
        self.assertEqual(report['status'], 'COMPLETE')
        row = report['runs']['location']
        self.assertAlmostEqual(row['original_reference_pair']['mae_gain_pooled'], .5)
        self.assertAlmostEqual(row['current_off_pair']['mae_gain_pooled'], .2)
        self.assertIn('same_stock_flow_decoder', row['current_off_pair']['reference_scope'])
        self.assertFalse(row['exact_source_prediction_initialization'])
        self.assertAlmostEqual(report['selected_best_location_vs_state']['pair']['mae_gain_pooled'], .1)
        self.assertEqual(report['source_MIN_training_updates'], 0)
        self.assertFalse(report['test_evaluated'])
        self.assertIn('条件 OFF', (self.root/'report.md').read_text(encoding='utf-8'))

    def test_current_off_cannot_replace_source_even_if_rehashed(self):
        path = self.fixture('state', .6)
        with np.load(path/'best_predictions.npz') as f:
            archive = {k: f[k].copy() for k in f.files}
        archive['original_off'] = archive['current_off'].copy()
        np.savez_compressed(path/'best_predictions.npz', **archive)
        summary = json.loads((path/'summary.json').read_text())
        summary['best_predictions_sha256'] = hashlib.sha256((path/'best_predictions.npz').read_bytes()).hexdigest()
        (path/'summary.json').write_text(json.dumps(summary))
        with self.assertRaisesRegex(ValueError, 'reference was replaced'):
            write_diagnostics(self.root, ('state',))

    def test_stale_summary_is_pending_and_source_training_rejected(self):
        path = self.fixture('state', .6)
        summary = json.loads((path/'summary.json').read_text())
        summary['completed_epoch'] = 0
        (path/'summary.json').write_text(json.dumps(summary))
        self.assertEqual(write_diagnostics(self.root, ('state',))['status'], 'PARTIAL')
        summary['source_MIN_training_updates'] = 1
        (path/'summary.json').write_text(json.dumps(summary))
        with self.assertRaisesRegex(ValueError, 'Source MIN updates'):
            write_diagnostics(self.root, ('state',))
        with self.assertRaises(ValueError):
            write_diagnostics(self.root, ('minimal',))

    def test_wrong_training_order_blocks_equal_budget_attribution(self):
        self.fixture('state', .6)
        path = self.fixture('location', .5)
        saved = torch.load(path/'last.pt', weights_only=False)
        saved['history'][1]['training']['train_order_sha256'] = 'different'
        torch.save(saved, path/'last.pt')
        with self.assertRaisesRegex(ValueError, 'training order differs'):
            write_diagnostics(self.root)


if __name__ == '__main__':
    unittest.main()
