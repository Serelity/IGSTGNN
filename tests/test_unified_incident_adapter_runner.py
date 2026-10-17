"""Matched budget and exact resume contracts for all six v0.2 arms."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.chronological import train_unified_incident_adapter as runner
from tests import test_original_isp_runner as fixtures


class UnifiedRunnerTests(unittest.TestCase):
    def test_parser_six_arms_and_fixed_architecture(self):
        args = runner.parse_args(['--run-dir', 'test-run'])
        self.assertEqual(args.arms, list(runner.ARMS))
        self.assertEqual(len(args.arms), 6)
        self.assertEqual(args.epochs, 5)
        self.assertFalse(args.resume)
        self.assertEqual(runner.ADAPTER_CONFIG, dict(state_dim=16, graph_rank=8, max_graph_mix=.25, max_state_delta=.1))
        source = Path(runner.__file__).read_text(encoding='utf-8')
        self.assertNotIn("evaluate('test')", source)
        self.assertNotIn('test_loader', source)
        self.assertNotIn('source-run', source)

    def test_parser_invalid_subset_and_budget(self):
        for tail in (['--only', 'TIME', 'TIME'], ['--arms', 'ORIGINAL', '--only', 'BOTH'],
                     ['--epochs', '0'], ['--epochs', '101'], ['--threads', '0']):
            with self.subTest(tail=tail), patch('sys.stderr'), self.assertRaises(SystemExit):
                runner.parse_args(['--run-dir', 'test-run', *tail])

    def test_initialization_checks_do_not_confuse_core_with_complete_original(self):
        init = dict(original_initial_tensor_sha256='original', traffic_tensor_sha256='core',
                    post_construction_rng_sha256='rng', adapter_tensor_sha256=None)
        values = dict(ORIGINAL=init, BOTH=dict(init, adapter_tensor_sha256='adapter'),
                      GRAPH=dict(init, adapter_tensor_sha256='adapter'))
        runner.verify_initializations(values)
        for key in init:
            bad = deepcopy(values)
            bad['BOTH'][key] = 'different'
            with self.subTest(key=key), self.assertRaises(ValueError):
                runner.verify_initializations(bad)

    def test_nonoriginal_arm_resume_keeps_rng_order_optimizer_and_scheduler(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            whole, split = Path(tmp)/'whole', Path(tmp)/'split'
            whole.mkdir()
            split.mkdir()
            a = fixtures.Engine()
            identity = fixtures.identity_for(a, 'BOTH')
            identity['arms'] = list(runner.ARMS)
            with patch('builtins.print'):
                completed = runner.run_epochs(a, whole, identity, 'BOTH', 4)
                runner.run_epochs(fixtures.Engine(), split, identity, 'BOTH', 2)
                checkpoint = runner.load_checkpoint(split/'last.pt')
                runner.validate_saved(checkpoint, identity, 'BOTH', split, 2)
                resumed = runner.run_epochs(fixtures.Engine(99), split, identity, 'BOTH', 4, checkpoint)
            self.assertEqual(fixtures.comparable_history(completed), fixtures.comparable_history(resumed))
            for key in ('model_state', 'optimizer_state', 'scheduler_state', 'rng_state', 'train_order'):
                self.assertEqual(runner.tensor_tree_hash(completed[key]), runner.tensor_tree_hash(resumed[key]))

    @staticmethod
    def state(arm, metrics):
        history = [dict(epoch=i, validation_mae=v, validation_rmse=v+1,
                        validation_mape_fraction=v*.01) for i, v in enumerate(metrics, 1)]
        best = min(history, key=lambda r: r['validation_mae'])
        return dict(arm=arm, history=history, completed_epoch=len(history), best_epoch=best['epoch'],
                    best_validation_mae=best['validation_mae'], best_checkpoint_file='best.pt',
                    best_checkpoint_sha256='test', model_tensor_sha256='test',
                    global_updates=2*len(history), initialization={})

    def test_summary_uses_common_prefix_and_full_original_baseline(self):
        states = {arm: self.state(arm, [4., 3.]) for arm in runner.ARMS}
        states['ORIGINAL'] = self.state('ORIGINAL', [4., 3., .5])
        states['BOTH'] = self.state('BOTH', [3., 2.])
        identity = dict(arms=list(runner.ARMS), engineering_only=True, data={})
        summary = runner.paired_summary(states, identity, 3)
        self.assertEqual(summary['common_completed_epochs'], 2)
        self.assertFalse(summary['same_completed_training_budget'])
        self.assertEqual(summary['comparisons']['BOTH_vs_ORIGINAL']['independent_best_at_common_budget']['mae_gain'], 1.)
        self.assertEqual(len(summary['comparisons']), 7)
        self.assertTrue(summary['primary_baseline_present'])
        self.assertEqual(summary['status'], 'UNIFIED_ADAPTER_TRAINING_PARTIAL')
        self.assertFalse(summary['test_evaluated'])

    def test_missing_arms_do_not_get_comparability_claim(self):
        identity = dict(arms=list(runner.ARMS), engineering_only=True, data={})
        result = runner.paired_summary({'BOTH': self.state('BOTH', [2.])}, identity, 1)
        self.assertEqual(result['comparisons'], {})
        self.assertIsNone(result['common_completed_epochs'])
        self.assertFalse(result['primary_baseline_present'])
        self.assertFalse(result['same_completed_training_budget'])


if __name__ == '__main__':
    unittest.main()
