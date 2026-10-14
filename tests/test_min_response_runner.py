"""Stage-one replay, immutable backbone and committed resume lifecycle."""
import copy
from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import subprocess
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from experiments.chronological import train_min_response as training
from experiments.chronological import train_min_event_budget as original
from experiments.chronological import train_b3_r_minimal_paper as paper
from experiments.chronological import train_internal_spatial as legacy
from experiments.chronological import probe_min_event_budget as probe
from experiments.chronological.smoke import set_seed
from src.models.incident_spatial_residual import tensor_state_hash
from tests import test_min_frozen_budget_runner as source_fixture
from tests.test_min_event_budget_runner import tiny_minimal


class ResponseRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source_fixture.FrozenBudgetRunnerTests.setUpClass()
        cls.fixture = source_fixture.FrozenBudgetRunnerTests
        cls.root = cls.fixture.root
        cls.recipe = legacy.read_json(training.RECIPE)

    @classmethod
    def tearDownClass(cls):
        cls.fixture.tearDownClass()

    def base(self):
        with patch.object(paper, 'build_model', side_effect=tiny_minimal):
            base, _ = original.build_arm(self.fixture.args, 'A', 4, self.fixture.graph, 'cpu')
        base.load_state_dict(self.fixture.saved['best_model_state'], strict=True)
        return base.eval()

    def models(self, arm='location'):
        base = self.base()
        set_seed(2025)
        model, certificate = training.build_response(base, arm, self.recipe)
        identity = dict(schema=training.SCHEMA, training_stage=self.recipe['training_stage'],
            arm=arm, response_mode=arm, recipe=copy.deepcopy(self.recipe), engineering_check=True, seed=2025,
            frozen_state_sha256=certificate['frozen_state_sha256'],
            response_initial_sha256=certificate['response_initial_sha256'])
        return model, base, self.fixture.data(), identity

    def fit(self, model, base, data, identity, output, epoch, **kwargs):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return training.fit(model, base, data, self.fixture.saved, identity, output, epoch, self.recipe, **kwargs)

    def output(self, suffix=''):
        return self.root/(self._testMethodName+suffix)

    def load(self, output):
        return torch.load(output/'last.pt', map_location='cpu', weights_only=False)

    def equal_nested(self, a, b):
        self.fixture.equal_nested(self, a, b)

    def test_cli_absolute_epochs_and_stage_no_test_or_batch_override(self):
        args = training.parse_args(['--source-run', str(self.fixture.source)])
        self.assertEqual(args.epochs, 1)
        self.assertEqual(args.arms, ['location', 'state'])
        self.assertEqual(self.recipe['training_stage'], 'frozen_diagnostic_stage1')
        self.assertEqual(training.parse_args(['--source-run', str(self.fixture.source), '--arms', 'state', 'location']).arms,
                         ['location', 'state'])
        for command in (['--epochs', '0'], ['--epochs', '11'], ['--arms', 'state', 'state'],
                        ['--arms', 'state', '--worker', 'location']):
            with self.subTest(command=command), self.assertRaises(ValueError):
                training.parse_args(['--source-run', str(self.fixture.source), *command])
        for command in (['--batch-size', '128'], ['--evaluate-test']):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                training.parse_args(['--source-run', str(self.fixture.source), *command])

    def test_both_initial_designs_equal_source_and_common_weights(self):
        args = training.parse_args(['--source-run', str(self.fixture.source)])
        result = training.preflight(args, self.fixture.source_identity, self.fixture.saved,
                                    self.fixture.data(), self.base(), self.recipe)
        self.assertEqual(result['status'], 'MIN_RESPONSE_INITIALIZATION_PAIR_PASS')
        self.assertTrue(result['all_initial_ON_OFF_exactly_match_MIN'])
        self.assertTrue(result['common_initialization_exact'])
        self.assertEqual(result['batch_partition'], [2, 1])
        self.assertEqual(len({c['common_initial_sha256'] for c in result['arms'].values()}), 1)

    def test_failed_initial_replay_blocks_optimizer_and_both_arms(self):
        args = training.parse_args(['--source-run', str(self.fixture.source)])
        evaluate = training.evaluate_pair
        def corrupt_state(model, *values, **kwargs):
            report, prediction = evaluate(model, *values, **kwargs)
            return report, prediction+1. if model.response.mode == 'state' else prediction
        with patch.object(training, 'evaluate_pair', side_effect=corrupt_state), \
                patch.object(training, 'optimizer_for', side_effect=AssertionError('Optimizer before replay')):
            with self.assertRaisesRegex(ValueError, 'Initial response ON differs'):
                training.preflight(args, self.fixture.source_identity, self.fixture.saved,
                                   self.fixture.data(), self.base(), self.recipe)

    def test_training_changes_only_response_and_preserves_source_and_archives(self):
        model, base, data, identity = self.models()
        before = tensor_state_hash(model.backbone.state_dict())
        response_before = tensor_state_hash(model.response.state_dict())
        paths = {str(p.relative_to(self.fixture.source)): p for p in self.fixture.source.rglob('*') if p.is_file()}
        identity['source_artifacts_sha256'] = probe.snapshot_files(paths)
        with patch.object(data, 'test_loader', side_effect=AssertionError('Unexpected test evaluation')):
            result = self.fit(model, base, data, identity, self.output(), 1, protected_paths=paths)
        self.assertEqual(tensor_state_hash(model.backbone.state_dict()), before)
        self.assertNotEqual(tensor_state_hash(model.response.state_dict()), response_before)
        self.assertEqual(probe.snapshot_files(paths), identity['source_artifacts_sha256'])
        self.assertTrue(result['off_exactly_matches_source'])
        self.assertFalse(result['final_joint_model'])
        self.assertFalse(result['test_evaluated'])
        saved = self.load(self.output())
        self.assertNotIn('model_state', saved)
        self.assertEqual(saved['history'][1]['training']['full_supervised_horizons'], 12)
        self.assertGreater(saved['history'][1]['training']['batches_with_nonzero_response_gradient'], 0)
        self.assertGreater(saved['history'][1]['training']['batches_with_nonzero_condition_gradient'], 0)
        with np.load(self.output()/'best_predictions.npz', allow_pickle=False) as best, \
                np.load(self.output()/'baseline_predictions.npz', allow_pickle=False) as baseline:
            self.assertEqual(best['history_low_mask'].shape, (3, 4))
            np.testing.assert_array_equal(best['off'], baseline['prediction'])
            np.testing.assert_array_equal(best['valid'], baseline['valid'])

    def test_two_epochs_equal_split_resume_optimizer_rng_order_and_response(self):
        model, base, data, identity = self.models('state')
        self.fit(model, base, data, identity, self.output('_whole'), 2)
        expected = self.load(self.output('_whole'))
        model, base, data, identity = self.models('state')
        self.fit(model, base, data, identity, self.output('_split'), 1)
        model, base, data, identity = self.models('state')
        summary = self.fit(model, base, data, identity, self.output('_split'), 2)
        actual = self.load(self.output('_split'))
        for key in ('response_state', 'best_response_state', 'optimizer_state', 'train_order', 'numpy_rng',
                    'torch_rng', 'cuda_rng', 'completed_epoch', 'best_epoch', 'best_metric'):
            with self.subTest(key=key):
                self.equal_nested(expected[key], actual[key])
        for left, right in zip(expected['history'], actual['history']):
            self.equal_nested({k: v for k, v in left.items() if k != 'seconds'},
                              {k: v for k, v in right.items() if k != 'seconds'})
        self.assertEqual(summary['optimizer_updates_this_invocation'], 2)
        model, base, data, identity = self.models('state')
        self.assertEqual(self.fit(model, base, data, identity, self.output('_split'), 2)['optimizer_updates_this_invocation'], 0)

    def test_epoch_zero_remains_candidate_when_updates_are_zero(self):
        model, base, data, identity = self.models()
        recipe = dict(self.recipe, learning_rate=0., weight_decay=0.)
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            result = training.fit(model, base, data, self.fixture.saved, identity, self.output(), 2, recipe)
        self.assertEqual(result['best_epoch'], 0)
        self.assertEqual(result['baseline_minus_best_mae'], 0.)

    def test_resume_rejects_legacy_identity_and_tampered_selection(self):
        model, base, data, identity = self.models()
        self.fit(model, base, data, identity, self.output(), 1)
        saved = self.load(self.output())
        for field, value in [('schema', 'old_frozen_schema'), ('best_metric', -1.)]:
            bad = copy.deepcopy(saved)
            bad[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                training.validate_checkpoint(bad, identity, model, saved['baseline_sha256'])

    def test_atomic_checkpoint_interruption_preserves_prior_epoch(self):
        model, base, data, identity = self.models()
        self.fit(model, base, data, identity, self.output(), 1)
        before = (self.output()/'last.pt').read_bytes()
        model, base, data, identity = self.models()
        real_save = training.save_checkpoint
        def interrupted(path, payload):
            if payload['completed_epoch'] == 2:
                raise OSError('simulated checkpoint write interruption')
            return real_save(path, payload)
        with patch.object(training, 'save_checkpoint', side_effect=interrupted), self.assertRaisesRegex(OSError, 'interruption'):
            self.fit(model, base, data, identity, self.output(), 2)
        self.assertEqual((self.output()/'last.pt').read_bytes(), before)
        model, base, data, identity = self.models()
        self.assertEqual(self.fit(model, base, data, identity, self.output(), 2)['completed_epoch'], 2)

    def test_output_directory_and_child_command_guards(self):
        args = training.parse_args(['--source-run', str(self.fixture.source)])
        for directory in (self.fixture.source, self.fixture.source/'nested', self.fixture.source.parent):
            args.run_dir = directory
            with self.subTest(directory=directory), self.assertRaises(ValueError):
                training.require_independent_directory(args)
        old = self.output('_old')
        old.mkdir()
        legacy.write_json(old/'identity.json', {'schema': 'old'})
        for directory in (old, old/'empty_child'):
            args.run_dir = directory
            with self.assertRaises(ValueError):
                training.require_independent_directory(args)
        args.run_dir, args.device = self.output(), 'cpu'
        command = training.child_command(args, 'state')
        self.assertTrue(Path(command[2]).is_absolute())
        self.assertEqual(command[command.index('--epochs')+1], '1')
        self.assertEqual(command[command.index('--worker')+1], 'state')

    def test_child_termination_retains_exit_code_diagnostic(self):
        args = training.parse_args(['--source-run', str(self.fixture.source), '--run-dir', str(self.output())])
        args.device = 'cpu'
        session = self.output('_session')
        session.mkdir()
        process = MagicMock()
        process.stdout.__iter__.side_effect = KeyboardInterrupt()
        process.poll.return_value = None
        process.wait.side_effect = [subprocess.TimeoutExpired('worker', 10), -9]
        process.returncode = -9
        with patch.object(training.subprocess, 'Popen', return_value=process), self.assertRaises(KeyboardInterrupt):
            training.run_worker(args, 'state', session)
        process.terminate.assert_called_once()
        process.kill.assert_called_once()
        self.assertEqual(legacy.read_json(session/'state_process_result.json'),
                         {'exit_code': -9, 'launcher_interrupted': True})


if __name__ == '__main__':
    unittest.main()
