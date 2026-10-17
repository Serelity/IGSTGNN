"""GW2 common-budget comparisons and inherited exact resume protocol."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.chronological import train_original_graph_message as runner
from tests.test_original_isp_runner import Engine, identity_for, comparable_history


class GraphMessageRunnerTests(unittest.TestCase):
    def identity(self, engine):
        identity = identity_for(engine)
        identity.update(schema=runner.SCHEMA, design=runner.DESIGN, arms=list(runner.ARMS))
        identity['initializations'] = {arm: identity['initializations']['ORIGINAL'] for arm in runner.ARMS}
        return identity

    def test_parser_pilot_and_total_budget(self):
        args = runner.parse_args(['--run-dir', 'unused'])
        self.assertEqual(args.epochs, 5)
        self.assertIsNone(args.only)
        for extra in (['--epochs', '0'], ['--epochs', '101'], ['--only', 'STATIC', 'STATIC']):
            with self.assertRaises(SystemExit):
                runner.parse_args(['--run-dir', 'unused', *extra])

    def test_exact_resume_and_identity_failure(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            full, split = Path(temp)/'full', Path(temp)/'split'
            full.mkdir(); split.mkdir()
            a = Engine()
            identity = self.identity(a)
            with patch('builtins.print'):
                expected = runner.run_epochs(a, full, identity, 'ORIGINAL', 4)
                runner.run_epochs(Engine(), split, identity, 'ORIGINAL', 2)
                saved = runner.load_checkpoint(split/'last.pt')
                runner.validate_saved(saved, identity, 'ORIGINAL', split, 2)
                actual = runner.run_epochs(Engine(19), split, identity, 'ORIGINAL', 4, saved)
            self.assertEqual(comparable_history(expected), comparable_history(actual))
            for key in ('model_state', 'optimizer_state', 'scheduler_state', 'rng_state', 'train_order'):
                self.assertEqual(runner.tensor_tree_hash(expected[key]), runner.tensor_tree_hash(actual[key]))
            for field, bad in (('identity_sha256', 'wrong'), ('global_updates', 100), ('arm', 'OTHER')):
                corrupted = deepcopy(saved); corrupted[field] = bad
                with self.assertRaises(ValueError):
                    runner.validate_saved(corrupted, identity, 'ORIGINAL', split, 2)

    def test_summary_compares_equal_prefix_including_negative_results(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            root = Path(temp); a = Engine(); identity = self.identity(a)
            with patch('builtins.print'):
                state = runner.run_epochs(a, root, identity, 'ORIGINAL', 2)
            states = {arm: deepcopy(state) for arm in runner.ARMS}
            for arm, value in states.items():
                value['arm'] = arm
            states['POST_MESSAGE']['history'][0]['validation_mae'] += 100
            states['POST_MESSAGE']['history'][1]['validation_mae'] += 100
            extra = dict(states['POST_MESSAGE']['history'][-1], epoch=3, validation_mae=0.)
            states['POST_MESSAGE']['history'].append(extra)
            states['POST_MESSAGE']['completed_epoch'] = 3
            summary = runner.paired_summary(states, identity, 3)
            self.assertEqual(summary['common_completed_epochs'], 2)
            self.assertFalse(summary['same_completed_training_budget'])
            self.assertEqual(summary['status'], 'GW2_TRAINING_PARTIAL')
            gain = summary['comparisons']['POST_MESSAGE_vs_ORIGINAL']['independent_best_at_common_budget']['mae_gain']
            self.assertAlmostEqual(gain, -100)
            self.assertNotIn(3, summary['comparisons']['POST_MESSAGE_vs_ORIGINAL']['best_epochs'].values())


if __name__ == '__main__':
    unittest.main()
