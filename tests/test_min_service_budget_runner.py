"""Service-only paired initialization, immutable source and exact resume."""
import copy
from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import subprocess
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from experiments.chronological import train_min_service_budget as training
from experiments.chronological import train_min_frozen_budget as frozen
from experiments.chronological import train_min_event_budget as original
from experiments.chronological import train_b3_r_minimal_paper as paper
from experiments.chronological import train_internal_spatial as legacy
from experiments.chronological import probe_min_event_budget as probe
from experiments.chronological.smoke import set_seed
from src.models.incident_spatial_residual import tensor_state_hash
from src.models.incident_spatial_frozen_budget import frozen_state_hash
from tests import test_min_frozen_budget_runner as source_fixture
from tests.test_min_event_budget_runner import tiny_minimal


class ServiceBudgetRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Reuse the source *fixture constructor*, without inheriting its tests.
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
        model, certificate = training.build_service(base, arm, self.recipe)
        identity = dict(schema=training.SCHEMA, arm=arm, service_mode=arm,
                        checkpoint_schema_reused=frozen.SCHEMA, recipe=copy.deepcopy(self.recipe),
                        engineering_check=True, seed=2025, adapter_certificate=certificate,
                        frozen_state_sha256=certificate['frozen_state_sha256'],
                        adapter_initial_sha256=certificate['adapter_initial_sha256'])
        return model, base, self.fixture.data(), identity

    def fit(self, model, base, data, identity, output, epochs):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return frozen.fit(model, base, data, self.fixture.saved, identity, output, epochs, self.recipe)

    def output(self, suffix=''):
        return self.root/(self._testMethodName+suffix)

    def load(self, path):
        return torch.load(path/'last.pt', map_location='cpu', weights_only=False)

    def equal_nested(self, left, right):
        self.fixture.equal_nested(self, left, right)

    def test_cli_absolute_epochs_arms_and_no_batch_or_test_override(self):
        args = training.parse_args(['--source-run', str(self.fixture.source)])
        self.assertEqual(args.epochs, 1)
        self.assertEqual(args.arms, ['location', 'state'])
        self.assertEqual(training.parse_args(['--source-run', str(self.fixture.source),
            '--arms', 'state', 'location', '--epochs', '10']).arms, ['location', 'state'])
        for command in (['--epochs', '0'], ['--epochs', '11'], ['--arms', 'state', 'state'],
                        ['--arms', 'state', '--worker', 'location']):
            with self.subTest(command=command), self.assertRaises(ValueError):
                training.parse_args(['--source-run', str(self.fixture.source), *command])
        for command in (['--batch-size', '128'], ['--evaluate-test']):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                training.parse_args(['--source-run', str(self.fixture.source), *command])

    def test_both_initial_designs_equal_source_before_any_training(self):
        args = training.parse_args(['--source-run', str(self.fixture.source)])
        result = training.preflight(args, self.fixture.source_identity, self.fixture.saved,
                                    self.fixture.data(), self.base(), self.recipe)
        self.assertEqual(result['status'], 'MIN_SERVICE_LIMIT_INITIALIZATION_PAIR_PASS')
        self.assertTrue(result['all_initial_ON_OFF_exactly_match_MIN'])
        self.assertFalse(result['shared_adapter_initialization'])
        self.assertEqual(result['batch_partition'], [2, 1])
        self.assertEqual(set(result['arms']), {'location', 'state'})
        self.assertEqual(len({c['frozen_state_sha256'] for c in result['arms'].values()}), 1)
        self.assertNotEqual(result['arms']['location']['adapter_initial_sha256'],
                            result['arms']['state']['adapter_initial_sha256'])

    def test_failed_second_initial_replay_blocks_the_whole_pair(self):
        args = training.parse_args(['--source-run', str(self.fixture.source)])
        evaluate = frozen.evaluate_pair

        def bad_state(model, *args, **kwargs):
            report, pred = evaluate(model, *args, **kwargs)
            if model.event_budget_adapter.conditioning_mode == 'state':
                pred = pred+1.
            return report, pred

        with patch.object(frozen, 'evaluate_pair', side_effect=bad_state), \
                patch.object(frozen, 'optimizer_for', side_effect=AssertionError('Optimizer before pair replay')):
            with self.assertRaisesRegex(ValueError, 'Initial service ON differs'):
                training.preflight(args, self.fixture.source_identity, self.fixture.saved,
                                   self.fixture.data(), self.base(), self.recipe)

    def test_state_and_location_preserve_all_backbone_tensors_and_full_horizon_training(self):
        source_before = tensor_state_hash(self.fixture.saved['best_model_state'])
        for arm in training.ARMS:
            with self.subTest(arm=arm):
                model, base, data, identity = self.models(arm)
                frozen_before = frozen_state_hash(model)
                adapter_before = tensor_state_hash(model.event_budget_adapter.state_dict())
                with patch.object(data, 'test_loader', side_effect=AssertionError('Unexpected test evaluation')):
                    summary = self.fit(model, base, data, identity, self.output('_'+arm), 1)
                self.assertEqual(frozen_state_hash(model), frozen_before)
                self.assertNotEqual(tensor_state_hash(model.event_budget_adapter.state_dict()), adapter_before)
                self.assertTrue(summary['off_exactly_matches_source'])
                self.assertTrue(summary['selection_includes_epoch_zero'])
                self.assertFalse(summary['test_evaluated'])
                saved = self.load(self.output('_'+arm))
                self.assertEqual(saved['schema'], frozen.SCHEMA)
                self.assertEqual(saved['identity']['schema'], training.SCHEMA)
                self.assertEqual(saved['history'][1]['training']['full_supervised_horizons'], 12)
                self.assertGreater(saved['history'][1]['training']['batches_with_nonzero_adapter_gradient'], 0)
        self.assertEqual(tensor_state_hash(self.fixture.saved['best_model_state']), source_before)

    def test_state_resume_equals_uninterrupted_optimizer_rng_order_and_adapter(self):
        model, base, data, identity = self.models('state')
        self.fit(model, base, data, identity, self.output('_whole'), 2)
        expected = self.load(self.output('_whole'))
        model, base, data, identity = self.models('state')
        self.fit(model, base, data, identity, self.output('_split'), 1)
        model, base, data, identity = self.models('state')
        result = self.fit(model, base, data, identity, self.output('_split'), 2)
        actual = self.load(self.output('_split'))
        for key in ('adapter_state', 'best_adapter_state', 'optimizer_state', 'train_order',
                    'numpy_rng', 'torch_rng', 'cuda_rng', 'best_epoch', 'best_metric', 'completed_epoch'):
            with self.subTest(key=key):
                self.equal_nested(expected[key], actual[key])
        self.assertEqual(result['optimizer_updates_this_invocation'], 2)
        model, base, data, identity = self.models('state')
        self.assertEqual(self.fit(model, base, data, identity, self.output('_split'), 2)
                         ['optimizer_updates_this_invocation'], 0)

    def test_new_identity_rejects_legacy_or_other_arm_resume(self):
        model, base, data, identity = self.models()
        self.fit(model, base, data, identity, self.output(), 1)
        saved = self.load(self.output())
        for change in ('legacy', 'other_arm'):
            bad = copy.deepcopy(saved)
            if change == 'legacy':
                bad['identity']['schema'] = frozen.SCHEMA
            else:
                bad['identity']['service_mode'] = 'state'
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, 'identity changed'):
                frozen.validate_checkpoint(bad, identity, model, saved['baseline_sha256'])

    def test_partial_pair_compares_only_common_epoch_and_rejects_order_drift(self):
        root = self.output()
        root.mkdir()
        args = training.parse_args(['--source-run', str(self.fixture.source), '--run-dir', str(root), '--epochs', '2'])
        identity = dict(schema=training.SCHEMA, arms=list(training.ARMS), initialization={'arms': {arm: {
            'frozen_state_sha256': 'f'*64, 'adapter_initial_sha256': arm} for arm in training.ARMS}})

        def history(values):
            return [dict(epoch=e, validation={'on': {'mae_pooled': value}},
                         **({'training': {'train_order_sha256': 'same_'+str(e)}} if e else {}))
                    for e, value in enumerate(values)]

        saves = {}
        for arm, values in [('location', [10., 8., 5.]), ('state', [10., 7.])]:
            (root/arm).mkdir()
            (root/arm/'last.pt').touch()
            saves[arm] = dict(schema=frozen.SCHEMA, identity=training.arm_identity(identity, arm),
                history=history(values), completed_epoch=len(values)-1, best_epoch=len(values)-1, best_metric=min(values))

        with patch.object(training.torch, 'load', side_effect=lambda path, **kw: saves[path.parent.name]):
            result = training.comparison(args, identity)
            self.assertEqual(result['status'], 'MIN_SERVICE_LIMIT_PAIRED_STAGE_PARTIAL')
            self.assertEqual(result['common_epoch'], 1)
            self.assertEqual(result['comparisons']['location_vs_state']['common_epoch_mae_gain'], -1.)
            self.assertEqual(result['comparisons']['location_vs_state']['common_prefix_best_mae_gain'], -1.)
            saves['state']['history'][1]['training']['train_order_sha256'] = 'changed'
            with self.assertRaisesRegex(ValueError, 'sample orders differ'):
                training.comparison(args, identity)

    def test_output_rejects_source_ancestors_and_old_experiment_descendants(self):
        args = training.parse_args(['--source-run', str(self.fixture.source)])
        for output in (self.fixture.source, self.fixture.source/'nested', self.fixture.source.parent):
            args.run_dir = output
            with self.subTest(output=output), self.assertRaises(ValueError):
                training.require_independent_directory(args)
        old = self.output('_old')
        old.mkdir()
        legacy.write_json(old/'identity.json', {'schema': frozen.SCHEMA})
        for output in (old, old/'empty_child'):
            args.run_dir = output
            with self.subTest(output=output), self.assertRaises(ValueError):
                training.require_independent_directory(args)

    def test_child_keeps_termination_and_exit_diagnostic_on_interrupt(self):
        args = training.parse_args(['--source-run', str(self.fixture.source), '--run-dir', str(self.output())])
        args.device = 'cpu'
        session = self.output('_session')
        session.mkdir()
        process = MagicMock()
        process.stdout.__iter__.side_effect = KeyboardInterrupt()
        process.poll.return_value = None
        process.wait.side_effect = [subprocess.TimeoutExpired('worker', 10), -9]
        process.returncode = -9
        with patch.object(training.subprocess, 'Popen', return_value=process):
            with self.assertRaises(KeyboardInterrupt):
                training.run_worker(args, 'state', session)
        process.terminate.assert_called_once()
        process.kill.assert_called_once()
        process.stdout.close.assert_called_once()
        self.assertEqual(legacy.read_json(session/'state_process_result.json'),
                         {'exit_code': -9, 'launcher_interrupted': True})


if __name__ == '__main__':
    unittest.main()
