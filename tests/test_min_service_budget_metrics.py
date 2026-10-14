import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from experiments.chronological.min_service_budget_metrics import EXPERIMENT_SCHEMA, summarize_risks, write_diagnostics


class ServiceBudgetMetricsTests(unittest.TestCase):
    def fixture(self):
        y = np.tile(np.array([0., 2., 10., 35., 80., 150., 300.])[None, None], (2, 12, 1))
        off = y+np.array([1., 1., -1., 1., -1., 1., -1.])
        on = off+.5
        valid = np.ones_like(y, dtype=bool)
        return on, off, y, valid

    def test_signed_gains_flow_partitions_and_low_flow_cost(self):
        on, off, y, valid = self.fixture()
        r = summarize_risks(on, off, y, valid)['groups']
        self.assertAlmostEqual(r['over']['gain']['mae'], -.5)
        self.assertAlmostEqual(r['under']['gain']['mae'], .5)
        self.assertEqual(r['all_nodes']['prediction_change']['up_fraction'], 1.)
        self.assertEqual(r['all_nodes']['positive_target_count'], 144)
        self.assertEqual(r['all_nodes']['valid_count'], 168)
        self.assertEqual(r['flow_below_100']['valid_count'], 120)
        self.assertLess(r['flow_below_100']['gain']['mae'], 0)

    def test_invalid_excluded_points_are_not_reintroduced_and_empty_group_is_null(self):
        on, off, y, valid = self.fixture()
        valid[:, :, :2] = False
        on[:, :, :2] = np.nan
        r = summarize_risks(on, off, y, valid)['groups']
        self.assertEqual(r['all_nodes']['valid_count'], 120)
        self.assertEqual(r['[0,5)']['status'], 'unavailable')
        self.assertIsNone(r['[0,5)']['gain']['mae'])
        self.assertEqual(r['exact']['status'], 'unavailable')
        json.dumps(r, allow_nan=False)

    def test_rejects_wrong_axes_mask_and_nonfinite_valid(self):
        on, off, y, valid = self.fixture()
        for values in ((on[:, :11], off[:, :11], y[:, :11], valid[:, :11]),
                       (on, off, y, valid.astype(np.int8))):
            with self.assertRaises(ValueError):
                summarize_risks(*values)
        on[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            summarize_risks(on, off, y, valid)

    def write_arm(self, root, arm, epoch, shift=.5):
        _, off, y, valid = self.fixture()
        on = off+shift
        path = root/arm
        path.mkdir()
        common = dict(target=y, valid=valid, sample_indices=np.array([3909, 3910]), fixed_hard_indices=np.array([0]))
        np.savez_compressed(path/'baseline_predictions.npz', prediction=off, **common)
        np.savez_compressed(path/'best_predictions.npz', on=on, off=off, **common)
        summary = dict(schema=EXPERIMENT_SCHEMA, service_mode=arm,
                       status='MIN_SERVICE_BUDGET_COMPLETE', best_adapter_epoch=epoch, completed_epoch=epoch,
                       best_mae=float(np.abs(on-y).mean()), trainable_parameters=100, test_evaluated=False)
        summary.update({name+'_sha256': hashlib.sha256((path/(name+'.npz')).read_bytes()).hexdigest()
                        for name in ('baseline_predictions', 'best_predictions')})
        (path/'summary.json').write_text(json.dumps(summary), encoding='utf-8')
        torch.save(dict(schema='fixture_checkpoint_container', completed_epoch=epoch, best_epoch=epoch,
            best_metric=summary['best_mae'], identity=dict(schema=EXPERIMENT_SCHEMA, arm=arm,
            service_mode=arm, checkpoint_schema_reused='fixture_checkpoint_container'),
            history=[dict(epoch=i, validation=dict(on=dict(mae_pooled=summary['best_mae'] if i == epoch else summary['best_mae']+1)))
                     for i in range(epoch+1)]), path/'last.pt')
        return path

    def test_writer_checks_baseline_and_defers_unequal_budget(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory)
            self.write_arm(root, 'location', 2)
            self.write_arm(root, 'state', 1, shift=.25)
            result = write_diagnostics(root)
            self.assertIsNone(result['selected_best_location_vs_state'])
            self.assertEqual(result['comparison_deferred'], 'unequal_completed_training_budgets')
            self.assertTrue((root/'report.md').is_file())

    def test_equal_budget_reports_selected_pair_and_preserves_sources(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory)
            self.write_arm(root, 'location', 1)
            self.write_arm(root, 'state', 1, shift=.25)
            protected = {str(p): p.read_bytes() for p in root.rglob('*') if p.is_file()}
            result = write_diagnostics(root)
            self.assertEqual(result['status'], 'COMPLETE')
            pair = result['selected_best_location_vs_state']['pair']
            _, off, y, _ = self.fixture()
            expected = float((np.abs(off+.25-y)-np.abs(off+.5-y)).mean())
            self.assertAlmostEqual(pair['mae_gain_pooled'], expected)
            self.assertEqual(protected, {p: Path(p).read_bytes() for p in protected})

    def test_archive_summary_mismatch_and_changed_sample_ids_are_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory)
            path = self.write_arm(root, 'location', 1)
            summary = json.loads((path/'summary.json').read_text())
            summary['best_mae'] += 1
            (path/'summary.json').write_text(json.dumps(summary))
            result = write_diagnostics(root)
            self.assertIn('location', result['pending'])
            summary['best_mae'] -= 1
            (path/'summary.json').write_text(json.dumps(summary))
            self.write_arm(root, 'state', 1)
            with np.load(root/'state/baseline_predictions.npz') as saved:
                baseline = dict(saved)
            baseline['sample_indices'] += 1
            np.savez_compressed(root/'state/baseline_predictions.npz', **baseline)
            with self.assertRaisesRegex(ValueError, 'archive hash'):
                write_diagnostics(root)

    def test_checkpoint_wrong_identity_or_selection_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory)
            path = self.write_arm(root, 'location', 1)
            saved = torch.load(path/'last.pt', weights_only=False)
            saved['best_epoch'] = 0
            torch.save(saved, path/'last.pt')
            with self.assertRaisesRegex(ValueError, 'selection certificate'):
                write_diagnostics(root)

    def test_interruption_before_new_summary_finalization_is_partial_not_foreign_run(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory)
            path = self.write_arm(root, 'location', 1)
            summary = json.loads((path/'summary.json').read_text())
            del summary['schema'], summary['service_mode']
            (path/'summary.json').write_text(json.dumps(summary))
            result = write_diagnostics(root)
            self.assertEqual(result['runs'], {})
            self.assertEqual(result['pending']['location'], 'service_summary_not_finalized_after_fit')

    def test_engineering_scope_and_window_count_are_visible(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory)
            path = self.write_arm(root, 'location', 1)
            summary = json.loads((path/'summary.json').read_text())
            summary['engineering_only'] = True
            (path/'summary.json').write_text(json.dumps(summary))
            result = write_diagnostics(root)
            self.assertTrue(result['engineering_only'])
            self.assertEqual(result['runs']['location']['validation_rows'], 2)
            self.assertIn('工程检查数据', (root/'report.md').read_text(encoding='utf-8'))


if __name__ == '__main__':
    unittest.main()
