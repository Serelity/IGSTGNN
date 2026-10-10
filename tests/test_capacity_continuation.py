"""Continuation audits must reject identity drift and preserve metric meaning."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.chronological import continue_incident_capacity as continuation
from experiments.chronological import train_incident_capacity as training
from src.utils.incident_corridor import read_json, write_json
from test_acdg import incident_batch, make_model
import test_capacity_training as training_tests
from test_incident_capacity_fusion import fixture


class CapacityContinuationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scratch = training.REPO/'experiments/chronological_runs'
        cls.scratch.mkdir(exist_ok=True)
        cls.temporary = tempfile.TemporaryDirectory(dir=cls.scratch)
        cls.root = Path(cls.temporary.name)
        cls.protocol = dict(read_json(continuation.PROTOCOL), train_samples=4, val_samples=4, stations=3, batch_size=2)
        cls.identity = dict(schema='capacity_training_m42_v1', check=True, batch_size=2, seed=11)
        cls.regions = dict(all_nodes=np.ones(3, bool), common_structure=np.array([True, True, False]),
                           added_structure=np.array([False, False, True]), candidate_structure=np.ones(3, bool),
                           outside_candidate=np.zeros(3, bool), candidate_boundary=np.array([True, False, True]),
                           **{'road_road-W': np.ones(3, bool)})
        base, cap_inputs = fixture()
        backbone = make_model()
        batch = dict(x=torch.rand(2, 12, 3, 3), incident=incident_batch(), capacity_inputs=cap_inputs)
        inputs = training_tests.CapacityTrainingTests().fake_inputs(batch)
        for arm in training.ARMS:
            model = training.build_arm(backbone, base.state_dict(), base.graph, base.outgoing_weights, arm, 11)
            training.train_arm(model, inputs, arm, cls.root, dict(cls.identity, arm=arm), cls.protocol,
                               11, 2, 2, 'cpu', True, False)
        cls.expected = dict(target=continuation.array_digest(np.ones((4, 12, 3, 1), np.float32)*2),
                            valid=continuation.array_digest(np.ones((4, 12, 3, 1), bool)),
                            sample_indices=continuation.array_digest(np.arange(4)))

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def audit(self, arm='P1', root=None, expected=None):
        return continuation.audit_arm((root or self.root)/arm, arm, self.identity, self.protocol,
                                      self.regions, expected or self.expected, 1.)

    def test_real_six_arm_readback_and_same_epoch_comparisons(self):
        runs = {arm: self.audit(arm) for arm in training.ARMS}
        report = continuation.comparisons(runs)
        for run in runs.values():
            self.assertEqual(run['global_updates'], 4)
            self.assertEqual(run['completed_epoch'], 2)
            self.assertIsNone(run['best_prediction_regions']['outside_candidate'])
            self.assertEqual(run['best_prediction_regions']['all_nodes']['valid_count'], 144)
        pair = report['same_epoch']['P1_vs_F'][-1]
        self.assertEqual(pair['epoch'], 2)
        self.assertAlmostEqual(pair['regions']['all_nodes'],
            runs['F']['epochs'][-1]['validation_mae_macro']-runs['P1']['epochs'][-1]['validation_mae_macro'])
        best = report['best_selected']['P1_vs_F']
        self.assertEqual(best['candidate_best_epoch'], runs['P1']['best_epoch'])
        self.assertEqual(best['control_best_epoch'], runs['F']['best_epoch'])
        # Selection and matched-epoch tables must remain distinct if optima differ.
        runs['P1']['best_epoch'] = 1
        changed = continuation.comparisons(runs)
        self.assertEqual(changed['best_selected']['P1_vs_F']['candidate_best_epoch'], 1)
        self.assertEqual(changed['same_epoch'], report['same_epoch'])

    def test_rejects_changed_validation_target_mask_or_order(self):
        for key in self.expected:
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'target/mask/order'):
                self.audit(expected=dict(self.expected, **{key: 'changed'}))

    def test_rejects_summary_checksum_and_checkpoint_order_corruption(self):
        import shutil
        with tempfile.TemporaryDirectory(dir=self.scratch) as tmp:
            root = Path(tmp)
            shutil.copytree(self.root/'P1', root/'P1')
            path = root/'P1/summary.json'
            summary = read_json(path)
            write_json(path, dict(summary, prediction_sha256='wrong'))
            with self.assertRaisesRegex(ValueError, 'Prediction file checksum'):
                self.audit(root=root)
            write_json(path, summary)
            checkpoint_path = root/'P1/last_checkpoint.pt'
            saved = torch.load(checkpoint_path, weights_only=False)
            saved['history'][0]['train_order_sha256'] = 'wrong'
            torch.save(saved, checkpoint_path)
            # A matched corrupted summary must still fail independently computed ordering.
            summary['history'] = saved['history']
            write_json(path, summary)
            with self.assertRaisesRegex(ValueError, 'Training order/update'):
                self.audit(root=root)

    def test_summary_lag_accepts_exact_prefix_and_uses_checkpoint_best(self):
        import shutil
        with tempfile.TemporaryDirectory(dir=self.scratch) as tmp:
            root = Path(tmp)
            shutil.copytree(self.root/'P1', root/'P1')
            path = root/'P1/summary.json'
            summary = read_json(path)
            summary['history'] = summary['history'][:1]
            summary['completed_epoch'] = 1
            summary['global_updates'] = 2
            summary['best_epoch'] = 1
            summary['best_metric'] = summary['history'][0]['validation_mae_macro']
            write_json(path, summary)
            audited = self.audit(root=root)
            self.assertTrue(audited['summary_lagged'])
            self.assertEqual(audited['completed_epoch'], 2)

    def test_metrics_preserve_horizon_pooling_empty_and_complement(self):
        error = np.array([[2., 18., 8.], [8., 4., 4.]])
        count = np.array([[1, 3, 2], [2, 1, 1]])
        metric = continuation.error_metrics(error, count, np.array([True, True, False]))
        self.assertEqual(metric['mae_macro'], (20/4+12/3)/2)
        self.assertEqual(metric['valid_count'], 7)
        self.assertEqual(continuation.error_metrics(error, count, np.array([False, False, True]))['mae_macro'], 4.)
        self.assertIsNone(continuation.error_metrics(error, count, np.zeros(3, bool)))
        count[0, 2] = 0
        self.assertIsNone(continuation.error_metrics(error, count, np.array([False, False, True])))

    def test_diagnostic_tail_weight_and_loss_units(self):
        row = dict(epoch=1, train_mae_macro=2., validation_mae_macro=3., seconds=1.,
                   losses=dict(main=2., auxiliary=4., prefix=6., curve=8.), branch_gradient_max_l1={},
                   diagnostics=[dict(capacity_limited_fraction=.2, state_min=.1, state_max=.8),
                                dict(capacity_limited_fraction=.8, state_min=.05, state_max=.9)])
        result = continuation.epoch_diagnostic(row, self.protocol, 10., 3, 2)
        self.assertAlmostEqual(result['capacity_limited_fraction'], .4)
        self.assertEqual(result['state_min'], .05)
        self.assertEqual(result['weighted_extra_loss_raw_units_batch_mean'],
                         dict(future_auxiliary=2., prefix_auxiliary=3., curve=.08))
        self.assertAlmostEqual(result['extra_to_main_loss_ratio'], 2.54)
        self.assertIn('first_two_batches', result['main_loss_gradient_scope'])

    def test_discovery_excludes_checks_and_refuses_multiple_full_runs(self):
        with tempfile.TemporaryDirectory(dir=self.scratch) as tmp:
            root = Path(tmp)
            for name, check in (('check', True), ('full1', False), ('full2', False)):
                directory = root/('contra_training_m42_'+name)
                directory.mkdir()
                write_json(directory/'identity.json', dict(self.identity, check=check))
                if name == 'full1':
                    self.assertEqual(continuation.discover_run(root), directory.resolve())
            with self.assertRaisesRegex(ValueError, '--run-dir'):
                continuation.discover_run(root)

    def test_command_keeps_identity_and_restores_optimizer(self):
        directories = {key: Path(key) for key in ('data', 'history', 'network', 'reports')}
        identity = dict(self.identity, device='cuda:0', ramp_exchanges=False)
        command = continuation.trainer_command(self.root, identity, directories, Path('sensors'), 5)
        self.assertIn('--resume', command)
        self.assertIn('--without-ramp-exchanges', command)
        self.assertEqual(command[command.index('--epochs')+1], '5')
        self.assertEqual(command[command.index('--seed')+1], '11')
        self.assertNotIn('--check', command)
        for option in ('--network-dir', '--report-bundle', '--history-dir'):
            self.assertIn(option, command)

    def test_frozen_source_set_rejects_drift_without_touching_model_code(self):
        from src.utils.incident_corridor import sha256
        identity = dict(self.identity, protocol_sha256=sha256(continuation.PROTOCOL),
                        source_sha256={name: sha256(training.REPO/name) for name in training.SOURCE_FILES})
        continuation.verify_sources(identity)
        identity['source_sha256']['src/models/igstgnn.py'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'Frozen training source changed'):
            continuation.verify_sources(identity)

    def test_failed_audit_blocks_training_and_releases_own_lock(self):
        with tempfile.TemporaryDirectory(dir=self.scratch) as tmp:
            root = Path(tmp)
            write_json(root/'identity.json', self.identity)
            with patch.object(continuation, 'verify_sources'), patch.object(continuation, 'verify_inputs'), \
                    patch.object(continuation, 'audit_run', side_effect=ValueError('broken checkpoint')), \
                    patch.object(continuation.subprocess, 'Popen') as child:
                with self.assertRaisesRegex(ValueError, 'broken checkpoint'):
                    continuation.main(['--run-dir', str(root), '--history-dir', str(root), '--audit-only'])
                child.assert_not_called()
            self.assertFalse((root/'capacity_continuation.lock').exists())

    def test_audit_only_preserves_report_and_never_starts_trainer(self):
        with tempfile.TemporaryDirectory(dir=self.scratch) as tmp:
            root = Path(tmp)
            write_json(root/'identity.json', self.identity)
            report = dict(status='fixture_audit_pass', test_accessed=False)
            with patch.object(continuation, 'verify_sources'), patch.object(continuation, 'verify_inputs'), \
                    patch.object(continuation, 'audit_run', return_value=report), \
                    patch.object(continuation, 'print_table'), patch.object(continuation.subprocess, 'Popen') as child:
                continuation.main(['--run-dir', str(root), '--history-dir', str(root), '--audit-only'])
                child.assert_not_called()
            reports = list(root.glob('continuation_*/before.json'))
            self.assertEqual(len(reports), 1)
            self.assertEqual(read_json(reports[0]), report)


if __name__ == '__main__':
    unittest.main()
