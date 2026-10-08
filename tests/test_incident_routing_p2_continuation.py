"""Check checkpoint gates and continuation orchestration without GPU training."""

import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from experiments.chronological import continue_incident_routing_p2 as continuation


class P2ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.repo = Path(self.tmp.name)
        self.root = self.repo / 'pair'
        self.protocol = self.repo / 'protocol.json'
        self.protocol.write_text(json.dumps({'patience': 20, 'max_epochs': 100}))
        for name in continuation.SOURCE_NAMES:
            path = self.repo / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('frozen fixture source')
        self.summaries = {}
        for variant in continuation.VARIANTS:
            directory = self.root / variant
            directory.mkdir(parents=True)
            identity = {
                'variant': variant, 'check': False, 'seed': 2025,
                'batch_size': 48, 'max_epochs': 100, 'device': 'cuda:0',
                'protocol_sha256': continuation.digest(self.protocol),
                'package_sha256': {'summary.json': 'raw', 'context_manifest.json': 'context'},
                'common_initialization_sha256': 'paired',
                'source_sha256': {name: continuation.digest(self.repo / name)
                                  for name in continuation.SOURCE_NAMES},
                'torch_version': 'fixture', 'numpy_version': 'fixture',
            }
            metric = 30. if variant == 'fixed' else 29.
            summary = {
                'variant': variant, 'identity': identity,
                'status': continuation.PAUSED, 'selection_metric': 'all_nodes.mae_macro',
                'completed_epoch': 1, 'best_epoch': 1, 'best_metric': metric,
                'global_updates': 76, 'initial_max_abs_difference_from_A': 0.,
                'parameters': 443645 if variant == 'fixed' else 465730,
                'history': self.history(1), 'runtime_epochs': [{'epoch': 1, 'seconds': 60.}],
            }
            self.summaries[variant] = summary
            (directory / 'summary.json').write_text(json.dumps(summary))
            checkpoint = {
                'format_version': 1, 'identity': copy.deepcopy(identity),
                'completed_epoch': 1, 'global_updates': 76, 'wait': 0,
                'best_epoch': 1, 'best_metric': metric, 'history': self.history(1),
            }
            (directory / 'last_checkpoint.pt').write_text(json.dumps(checkpoint))
        self.fake_torch = types.SimpleNamespace(
            __version__='fixture',
            load=lambda path, **kwargs: json.loads(Path(path).read_text()),
            cuda=types.SimpleNamespace(is_available=lambda: True,
                                       get_device_name=lambda _: 'Tesla V100 fixture'))

    @staticmethod
    def history(epochs):
        return [{'epoch': epoch, 'train_order_sha256': f'order-{epoch}'}
                for epoch in range(1, epochs + 1)]

    def checkpoint(self, variant='fixed'):
        return json.loads((self.root / variant / 'last_checkpoint.pt').read_text())

    def finish(self, variant):
        directory = self.root / variant
        s = json.loads((directory / 'summary.json').read_text())
        s.update(status=continuation.COMPLETE, completed_epoch=21, global_updates=21 * 76,
                 stop_reason='early_stopping', history=self.history(21),
                 runtime_epochs=[{'epoch': i, 'seconds': 60.} for i in range(1, 22)])
        metrics = {'mae_macro': s['best_metric'], 'rmse_macro': 40., 'mape_macro': .2,
                   'per_horizon_mae': [s['best_metric']] * 12}
        s['best_validation_metrics'] = {'all_nodes': metrics, 'associated_nodes': metrics}
        (directory / 'best_model.pt').write_bytes(b'best model fixture')
        (directory / 'best_validation_predictions.npz').write_bytes(b'prediction fixture')
        s['best_validation_predictions_sha256'] = continuation.digest(
            directory / 'best_validation_predictions.npz')
        (directory / 'summary.json').write_text(json.dumps(s))
        checkpoint = self.checkpoint(variant)
        checkpoint.update(completed_epoch=21, global_updates=21 * 76,
                          wait=20, history=self.history(21))
        (directory / 'last_checkpoint.pt').write_text(json.dumps(checkpoint))

    def run_main(self, train=None):
        def complete_command(command, **kwargs):
            self.assertIn('--resume', command)
            self.assertNotIn('--check', command)
            self.assertNotIn('--stop-after-epoch', command)
            self.assertTrue(kwargs['check'])
            variant = command[command.index('--variant') + 1]
            self.assertEqual(Path(command[command.index('--output-dir') + 1]), self.root / variant)
            self.finish(variant)
        with patch.object(continuation, 'REPO', self.repo), \
                patch.object(continuation, 'PROTOCOL', self.protocol), \
                patch.dict('sys.modules', {'torch': self.fake_torch,
                                          'numpy': types.SimpleNamespace(__version__='fixture')}), \
                patch.object(continuation.subprocess, 'run', side_effect=train or complete_command) as runner, \
                redirect_stdout(io.StringIO()):
            continuation.main(['--run-dir', str(self.root), '--data-dir', str(self.repo / 'data')])
            return runner.call_args_list

    def test_resumes_both_existing_runs_and_exports_comparison(self):
        calls = self.run_main()
        self.assertEqual(len(calls), 2)
        self.assertIn('fixed', calls[0].args[0])
        self.assertIn('acdg', calls[1].args[0])
        report = json.loads((self.root / 'completion_report.json').read_text())
        self.assertEqual(report['status'], 'P2_PAIRED_SCREENING_COMPLETE')
        self.assertEqual(report['selection_mae_gain_fixed_minus_acdg'], 1.)
        self.assertAlmostEqual(report['selection_relative_gain_percent'], 100 / 30)
        self.assertEqual(report['runs']['acdg']['metrics']['associated_nodes']['h7_h12_mae_macro'], 29.)

    def test_completed_baseline_is_skipped(self):
        self.finish('fixed')
        before = (self.root / 'fixed/last_checkpoint.pt').read_bytes()
        calls = self.run_main()
        self.assertEqual(len(calls), 1)
        self.assertIn('acdg', calls[0].args[0])
        self.assertEqual(before, (self.root / 'fixed/last_checkpoint.pt').read_bytes())

    def test_complete_pair_is_reported_without_retraining_or_gpu(self):
        self.finish('fixed')
        self.finish('acdg')
        self.fake_torch.cuda.is_available = lambda: False
        self.assertEqual(self.run_main(), [])

    def test_failed_training_does_not_start_second_arm(self):
        calls = []
        def fail(command, **kwargs):
            calls.append(command)
            raise continuation.subprocess.CalledProcessError(23, command)
        with self.assertRaises(continuation.subprocess.CalledProcessError):
            self.run_main(train=fail)
        self.assertEqual(len(calls), 1)
        self.assertFalse((self.root / 'completion_report.json').exists())

    def test_changed_sources_and_mismatched_pair_are_rejected(self):
        self.summaries['acdg']['identity']['common_initialization_sha256'] = 'different'
        with self.assertRaisesRegex(ValueError, 'Paired identity mismatch'):
            continuation.validate_pair(self.summaries, self.repo, self.protocol)
        self.summaries['acdg']['identity']['common_initialization_sha256'] = 'paired'
        (self.repo / 'src/models/igstgnn.py').write_text('changed model')
        with self.assertRaisesRegex(ValueError, 'Training source changed'):
            continuation.validate_pair(self.summaries, self.repo, self.protocol)

    def test_stale_summary_accepts_newer_nonterminal_checkpoint(self):
        checkpoint = self.checkpoint()
        checkpoint.update(completed_epoch=3, global_updates=228, history=self.history(3), wait=2)
        self.assertEqual(continuation.checkpoint_action(
            self.summaries['fixed'], checkpoint, self.root / 'fixed'), 'resume')

    def test_terminal_checkpoint_without_export_must_not_train_extra_epochs(self):
        for epoch, wait in ((100, 1), (21, 20)):
            checkpoint = self.checkpoint()
            checkpoint.update(completed_epoch=epoch, global_updates=epoch * 76,
                              history=self.history(epoch), wait=wait)
            with self.assertRaisesRegex(ValueError, 'export is incomplete'):
                continuation.checkpoint_action(self.summaries['fixed'], checkpoint, self.root / 'fixed')

    def test_bad_checkpoint_identity_is_rejected(self):
        checkpoint = self.checkpoint()
        checkpoint['identity']['seed'] = 2026
        with self.assertRaisesRegex(ValueError, 'identity mismatch'):
            continuation.checkpoint_action(self.summaries['fixed'], checkpoint, self.root / 'fixed')

    def test_completed_prediction_tampering_is_rejected(self):
        self.finish('fixed')
        summary = json.loads((self.root / 'fixed/summary.json').read_text())
        (self.root / 'fixed/best_validation_predictions.npz').write_bytes(b'different')
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
            continuation.checkpoint_action(summary, self.checkpoint(), self.root / 'fixed')


if __name__ == '__main__':
    unittest.main()
