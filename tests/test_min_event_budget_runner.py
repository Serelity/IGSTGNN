"""Paired initialization, published engine forwarding and safe resume contracts."""
import copy
from contextlib import redirect_stderr, redirect_stdout
import io
import logging
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from experiments.chronological import train_min_event_budget as training
from experiments.chronological import train_b3_r_minimal_paper as paper
from experiments.chronological import train_internal_spatial as legacy
from experiments.chronological.smoke import set_seed
from src.models.incident_spatial import configure_spatial_mode
from src.models.incident_spatial_minimal import from_residual
from src.models.incident_spatial_residual import from_canonical_b3
from src.utils.igstgnn_published_events import EventPublishedData
from tests.test_architecture_mechanisms import tiny_model
from tests.test_incident_spatial import graph
from tests.test_min_event_budget_data import event_fixture


def tiny_minimal(args, nodes, candidate_graph, device):
    """Actual five-layer MIN and exchange; only width and station count shrink."""
    if nodes != 4:
        raise AssertionError('Expected the four-station fixture')
    set_seed(args.seed)
    canonical = configure_spatial_mode(tiny_model().requires_grad_(True),
                                       'state_restricted', copy.deepcopy(candidate_graph))
    residual, _ = from_canonical_b3(canonical)
    model, certificate = from_residual(residual, clock_offset_slots=0)
    return model.to(device), certificate


def args_for(root, *, epochs=1, check=True):
    return SimpleNamespace(run_dir=root, data_dir=root, graph_dir=training.HERE/'internal_spatial_graph_is11',
                           protocol=training.PROTOCOL, arms=list(training.ARMS), only=None, worker=None,
                           batch_size=None, device='cpu', seed=2025, threads=1, epochs=epochs,
                           check=check, evaluate_test=False)


class EventBudgetRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        cls.logger = logging.getLogger('event_budget_runner_test')
        cls.logger.addHandler(logging.NullHandler())

    def equal_nested(self, left, right):
        if torch.is_tensor(left):
            torch.testing.assert_close(left, right, atol=0, rtol=0)
        elif isinstance(left, np.ndarray):
            np.testing.assert_array_equal(left, right)
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.equal_nested(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(type(left), type(right))
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.equal_nested(a, b)
        else:
            self.assertEqual(left, right)

    def data(self, root, protocol):
        return EventPublishedData(root, protocol, self.logger, check=True)

    def test_cli_absolute_epochs_batch_defaults_and_worker_guard_before_data_access(self):
        args = training.parse_args([])
        self.assertEqual(args.arms, ['A', 'L', 'S'])
        self.assertEqual(args.epochs, 10)
        self.assertEqual(training.load_protocol(args)['batch_size'], 64)
        self.assertEqual(training.parse_args(['--check']).epochs, 1)
        self.assertEqual(training.parse_args(['--arms', 'S', 'A']).arms, ['A', 'S'])
        for command in (['--arms', 'A', 'A'], ['--arms', 'A', '--only', 'S'],
                        ['--only', 'L', 'L'], ['--batch-size', '0'],
                        ['--check', '--epochs', '2'], ['--check', '--evaluate-test']):
            with self.subTest(command=command), self.assertRaises(ValueError):
                training.parse_args(command)
        with tempfile.TemporaryDirectory(dir=training.REPO) as temp:
            with patch.dict(os.environ, {training.WORKER_ENV: ''}):
                with patch.object(training, 'prepare', side_effect=AssertionError('Data loaded before worker guard')):
                    with self.assertRaisesRegex(ValueError, 'Internal worker'):
                        training.main(['--run-dir', temp, '--device', 'cpu', '--worker', 'L'])
            root = Path(temp)
            legacy.write_json(root/'identity.json', {'schema': paper.SCHEMA})
            with patch.object(training, 'prepare', side_effect=AssertionError('Legacy experiment accessed data')):
                with self.assertRaisesRegex(ValueError, 'old experiment'):
                    training.main(['--run-dir', temp, '--device', 'cpu'])

    def test_frozen_protocol_and_identity_allow_only_and_epochs_but_reject_batch_drift(self):
        args = training.parse_args([])
        original = training.PROTOCOL.read_bytes()
        protocol = training.load_protocol(args)
        self.assertEqual(training.load_protocol(training.parse_args(['--batch-size', '32'])),
                         dict(protocol, batch_size=32))
        self.assertEqual(original, training.PROTOCOL.read_bytes())
        with tempfile.TemporaryDirectory(dir=training.REPO) as temp:
            root = Path(temp)
            args = args_for(root, check=False)
            certificate = {'fixture': 'shared initialization'}
            first = training.make_identity(args, protocol, {'data': 'fixed'}, {'graph': 'fixed'}, certificate)
            args.only, args.epochs, args.worker = ['S'], 70, 'S'
            second = training.make_identity(args, protocol, {'data': 'fixed'}, {'graph': 'fixed'}, certificate)
            self.assertEqual(first, second)
            self.assertEqual(first['schema'], training.SCHEMA)
            self.assertEqual(first['batch_size'], 64)
            self.assertNotIn('gpu', first['runtime'])
            self.assertEqual(first['excluded_event_fields'],
                             ['Incident Time', 'durations', 'Holiday', 'incident_position'])
            training.verify_identity(args, first, certificate, allow_new=True)
            snapshot = {name: (root/name).read_bytes() for name in ('identity.json', 'initialization.json')}
            changed = training.make_identity(args, dict(protocol, batch_size=32),
                                             {'data': 'fixed'}, {'graph': 'fixed'}, certificate)
            with self.assertRaisesRegex(ValueError, 'identity changed'):
                training.verify_identity(args, changed, certificate, allow_new=False)
            with self.assertRaisesRegex(ValueError, 'certificate changed'):
                training.verify_identity(args, first, {'fixture': 'changed'}, allow_new=False)
            self.assertEqual(snapshot, {name: (root/name).read_bytes() for name in snapshot})
            changed_path = root/'changed_protocol.json'
            legacy.write_json(changed_path, dict(protocol, learning_rate=.003))
            args.protocol = changed_path
            with self.assertRaisesRegex(ValueError, 'protocol changed'):
                training.load_protocol(args)
        self.assertTrue(paper.source_fingerprints().items() <= training.source_fingerprints().items())

    def test_actual_preflight_pairs_common_MIN_and_adapter_parameters_and_preserves_RNG(self):
        with tempfile.TemporaryDirectory(dir=training.REPO) as temp:
            root = Path(temp)
            _, protocol = event_fixture(root)
            args, data, g = args_for(root), self.data(root, protocol), graph(4)
            with patch.object(paper, 'build_model', side_effect=tiny_minimal):
                initialization = training.preflight(args, data, g)
                states = []
                for arm in training.ARMS:
                    model, _ = training.build_arm(args, arm, 4, g, 'cpu')
                    states.append((np.random.get_state(), torch.get_rng_state().clone()))
                for state in states[1:]:
                    self.equal_nested(states[0], state)
            self.assertEqual(initialization['status'], 'MIN_I_INITIALIZATION_PAIR_PASS')
            self.assertTrue(initialization['common_parameters_exact'])
            self.assertEqual(initialization['zero_initialized_prediction_abs_max'], 0.)
            self.assertEqual(initialization['report_off_prediction_abs_max'], 0.)
            arms = initialization['arms']
            self.assertEqual(len({row['common_state_sha256'] for row in arms.values()}), 1)
            self.assertLess(arms['A']['parameters'], arms['L']['parameters'])
            self.assertLess(arms['L']['parameters'], arms['S']['parameters'])
            self.assertEqual(arms['L']['conversion']['adapter_A_initial_sha256'],
                             arms['S']['conversion']['adapter_A_initial_sha256'])

    def test_engine_keeps_event_types_ignores_future_targets_and_A_ignores_event_payload(self):
        with tempfile.TemporaryDirectory(dir=training.REPO) as temp:
            root = Path(temp)
            _, protocol = event_fixture(root)
            args, data, g = args_for(root), self.data(root, protocol), graph(4)
            batch = next(data.loaders['train_loader'].get_iterator())
            with patch.object(paper, 'build_model', side_effect=tiny_minimal):
                for arm in training.ARMS:
                    with self.subTest(arm=arm):
                        model, _ = training.build_arm(args, arm, 4, g, 'cpu')
                        engine = training.make_engine(args, arm, protocol, data, model, self.logger)
                        x, y, event, sensor = engine._prepare_batch(batch)
                        if arm == 'A':
                            self.assertIsNone(event)
                        else:
                            self.assertEqual(event['type'].dtype, torch.long)
                            self.assertEqual(event['description'].dtype, torch.long)
                            self.assertEqual(event['relation'].dtype, torch.float32)
                            self.assertEqual(event['type_valid'].dtype, torch.bool)
                            self.assertEqual(event['report_present'].dtype, torch.bool)
                        model.eval()
                        with torch.inference_mode():
                            actual = engine._predict(x, y, event, sensor)
                            changed_target = engine._predict(x, y+10000., event, sensor)
                        torch.testing.assert_close(actual, changed_target, atol=0, rtol=0)
                        with patch.object(data, 'test_loader', side_effect=AssertionError('Implicit test read')):
                            engine.evaluate('val')
                        self.assertFalse(engine.collect_diagnostics)
                        with self.assertRaisesRegex(ValueError, 'explicit'):
                            engine.evaluate('test')
                engine = training.make_engine(args, 'A', protocol, data, model, self.logger)
                malformed = dict(batch, event_data={'anything': object()})
                self.assertIsNone(engine._prepare_batch(malformed)[2])

    def test_each_arm_resume_is_exact_and_three_arms_keep_same_training_order(self):
        with tempfile.TemporaryDirectory(dir=training.REPO) as temp, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            root = Path(temp)
            _, protocol = event_fixture(root)
            g, orders = graph(4), {}
            with patch.object(paper, 'build_model', side_effect=tiny_minimal):
                for arm in training.ARMS:
                    identity = {'fixture': 'restart contract', 'arm': arm}
                    checkpoints = []
                    for name, stages in [('continuous', [2]), ('resumed', [1, 2])]:
                        run = root/(arm+'_'+name)
                        run.mkdir()
                        args = args_for(run)
                        args.data_dir = root
                        for epoch in stages:
                            args.epochs = epoch
                            data = self.data(root, protocol)
                            with patch.object(data, 'test_loader', side_effect=AssertionError('Implicit test evaluation')):
                                summary = training.train_arm(args, arm, protocol, data, identity, g, self.logger)
                        path = run/arm/'last.pt'
                        before = path.read_bytes()
                        summary = training.train_arm(args, arm, protocol, self.data(root, protocol), identity, g, self.logger)
                        self.assertEqual(before, path.read_bytes())
                        self.assertEqual(summary['optimizer_updates_this_invocation'], 0)
                        checkpoints.append(torch.load(path, map_location='cpu', weights_only=False))
                    a, b = checkpoints
                    for key in ('model_state', 'best_model_state', 'optimizer_state', 'scheduler_state',
                                'train_order', 'numpy_rng', 'torch_rng', 'cuda_rng', 'global_updates',
                                'curriculum_length', 'best_metric', 'wait', 'completed_epoch'):
                        self.equal_nested(a[key], b[key])
                    for left, right in zip(a['history'], b['history']):
                        self.equal_nested({k: v for k, v in left.items() if k != 'seconds'},
                                          {k: v for k, v in right.items() if k != 'seconds'})
                    self.assertEqual(a['global_updates'], 4)
                    self.assertEqual(a['history'][0]['curriculum_length'], 1)
                    self.assertEqual(a['history'][1]['curriculum_length'], 2)
                    orders[arm] = [row['train_order_sha256'] for row in a['history']]
            self.assertEqual(orders['A'], orders['L'])
            self.assertEqual(orders['L'], orders['S'])

    def test_early_stop_is_explicit_and_resume_does_not_silently_restart(self):
        with tempfile.TemporaryDirectory(dir=training.REPO) as temp, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            root = Path(temp)
            _, protocol = event_fixture(root)
            protocol = dict(protocol, patience=1)
            run = root/'run'
            run.mkdir()
            args = args_for(run, epochs=5, check=False)
            args.data_dir = root
            identity, g = {'fixture': 'early stopping'}, graph(4)
            with patch.object(paper, 'build_model', side_effect=tiny_minimal):
                with patch.object(training.EventBudgetEngine, 'evaluate', return_value=(10., .1, 20.)):
                    summary = training.train_arm(args, 'L', protocol, self.data(root, protocol), identity, g, self.logger)
                self.assertEqual(summary['status'], 'EARLY_STOPPED')
                self.assertEqual(summary['completed_epoch'], 2)
                self.assertEqual(summary['best_epoch'], 1)
                self.assertEqual(summary['wait'], 1)
                path = run/'L'/'last.pt'
                before = path.read_bytes()
                with patch.object(training.EventBudgetEngine, 'train_batch', side_effect=AssertionError('Early stop restarted')):
                    again = training.train_arm(args, 'L', protocol, self.data(root, protocol), identity, g, self.logger)
                self.assertEqual(before, path.read_bytes())
                self.assertEqual(again['status'], 'EARLY_STOPPED')
                self.assertEqual(again['optimizer_updates_this_invocation'], 0)

    def test_sidecar_repair_backs_up_damage_rejects_conflicts_and_preserves_checkpoint(self):
        with tempfile.TemporaryDirectory(dir=training.REPO) as temp, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            root = Path(temp)
            _, protocol = event_fixture(root)
            run = root/'run'
            run.mkdir()
            args = args_for(run)
            args.data_dir = root
            identity, g = {'fixture': 'sidecar recovery'}, graph(4)
            with patch.object(paper, 'build_model', side_effect=tiny_minimal):
                training.train_arm(args, 'S', protocol, self.data(root, protocol), identity, g, self.logger)
            path, arm_root = run/'S'/'last.pt', run/'S'
            before = path.read_bytes()
            saved = torch.load(path, map_location='cpu', weights_only=False)
            for name in ('history.json', 'summary.json'):
                (arm_root/name).write_text('{interrupted', encoding='utf-8')
            training.repair_sidecars(arm_root, saved, args, protocol)
            self.assertEqual(legacy.read_json(arm_root/'history.json'), saved['history'])
            self.assertEqual(len(list(arm_root.glob('*.interrupted_*'))), 2)
            self.assertEqual(before, path.read_bytes())
            legacy.write_json(arm_root/'history.json', [{'epoch': 99}])
            with self.assertRaisesRegex(ValueError, 'history conflicts'):
                training.repair_sidecars(arm_root, saved, args, protocol)
            self.assertEqual(legacy.read_json(arm_root/'history.json'), [{'epoch': 99}])
            with self.assertRaisesRegex(ValueError, 'identity/arm mismatch'):
                training.validate_saved(saved, {'different': True}, 'S')
            with self.assertRaisesRegex(ValueError, 'identity/arm mismatch'):
                training.validate_saved(saved, identity, 'L')
            legacy.write_json(arm_root/'history.json', saved['history'])
            original_summary = training.summary_for(saved, args, protocol)
            for field in ('best_epoch', 'best_metric', 'global_updates', 'curriculum_length'):
                with self.subTest(field=field):
                    conflicting = dict(original_summary, **{field: original_summary[field]+1})
                    legacy.write_json(arm_root/'summary.json', conflicting)
                    summary_bytes = (arm_root/'summary.json').read_bytes()
                    with self.assertRaisesRegex(ValueError, 'summary counters conflict'):
                        training.repair_sidecars(arm_root, saved, args, protocol)
                    self.assertEqual(summary_bytes, (arm_root/'summary.json').read_bytes())
                    self.assertEqual(before, path.read_bytes())

    def test_comparison_uses_common_prefix_checks_order_and_labels_attribution(self):
        with tempfile.TemporaryDirectory(dir=training.REPO) as temp, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            root = Path(temp)
            _, protocol = event_fixture(root)
            run = root/'run'
            run.mkdir()
            args = args_for(run)
            args.data_dir = root
            identity, g = {'fixture': 'comparison', 'protocol': protocol}, graph(4)
            with patch.object(paper, 'build_model', side_effect=tiny_minimal):
                for arm in training.ARMS:
                    training.train_arm(args, arm, protocol, self.data(root, protocol), identity, g, self.logger)
            report = training.comparison(args, identity)
            self.assertEqual(report['status'], 'MIN_I_PAIRED_STAGE_COMPLETE')
            self.assertEqual(report['common_epoch'], 1)
            values = {arm: legacy.read_json(run/arm/'history.json')[0]['validation_mae'] for arm in training.ARMS}
            self.assertAlmostEqual(report['comparisons']['S_vs_L']['common_epoch_mae_gain'], values['L']-values['S'])
            self.assertIn('additional_embedding_parameters', report['interpretation']['S_vs_L'])
            self.assertFalse(report['predictive_gain_claim'])
            path = run/'L'/'last.pt'
            saved = torch.load(path, map_location='cpu', weights_only=False)
            saved['history'][0]['train_order_sha256'] = 'different-order'
            torch.save(saved, path)
            with self.assertRaisesRegex(ValueError, 'sample order differs'):
                training.comparison(args, identity)

    def test_explicit_test_uses_saved_best_without_training_and_noop_preserves_report(self):
        with tempfile.TemporaryDirectory(dir=training.REPO) as temp, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            root = Path(temp)
            _, protocol = event_fixture(root)
            run = root/'run'
            run.mkdir()
            args = args_for(run)
            args.data_dir = root
            identity, g = {'fixture': 'explicit test'}, graph(4)
            with patch.object(paper, 'build_model', side_effect=tiny_minimal):
                training.train_arm(args, 'S', protocol, self.data(root, protocol), identity, g, self.logger)
                path = run/'S'/'last.pt'
                before = path.read_bytes()
                args.evaluate_test = True
                with patch.object(training.EventBudgetEngine, 'train_batch', side_effect=AssertionError('Test updated model')):
                    result = training.train_arm(args, 'S', protocol, self.data(root, protocol), identity, g, self.logger)
                self.assertTrue(result['test_evaluated'])
                self.assertEqual(result['test_metrics']['samples'], 3)
                self.assertEqual(len(result['test_metrics']['by_horizon']), 12)
                self.assertEqual(result['test_metrics']['best_epoch'], result['best_epoch'])
                self.assertEqual(before, path.read_bytes())
                args.evaluate_test = False
                again = training.train_arm(args, 'S', protocol, self.data(root, protocol), identity, g, self.logger)
                self.assertTrue(again['test_evaluated'])
                self.assertEqual(result['test_metrics'], again['test_metrics'])
                self.assertEqual(before, path.read_bytes())

    def test_interrupted_launcher_terminates_or_kills_child_and_records_exit_before_unlock(self):
        with tempfile.TemporaryDirectory(dir=training.REPO) as temp, redirect_stdout(io.StringIO()):
            root = Path(temp)
            args = args_for(root)
            for require_kill in (False, True):
                with self.subTest(require_kill=require_kill):
                    session = root/('kill' if require_kill else 'terminate')
                    session.mkdir()
                    process = MagicMock()
                    process.returncode = None
                    process.poll.return_value = None

                    def lines():
                        yield 'last complete progress line\n'
                        raise KeyboardInterrupt()

                    def wait(timeout=None):
                        if require_kill and not process.kill.called:
                            raise training.subprocess.TimeoutExpired('worker', timeout)
                        process.returncode = -9 if process.kill.called else -15
                        return process.returncode

                    process.stdout.__iter__.side_effect = lines
                    process.wait.side_effect = wait
                    with patch.object(training.subprocess, 'Popen', return_value=process):
                        with self.assertRaises(KeyboardInterrupt):
                            training.run_worker(args, 'S', session)
                    process.terminate.assert_called_once_with()
                    self.assertEqual(process.kill.call_count, int(require_kill))
                    process.stdout.close.assert_called_once_with()
                    result = legacy.read_json(session/'S_process_result.json')
                    self.assertTrue(result['launcher_interrupted'])
                    self.assertEqual(result['exit_code'], -9 if require_kill else -15)
                    self.assertIn('last complete progress line', (session/'S.log').read_text(encoding='utf-8'))

    def test_worker_own_lock_blocks_surviving_child_overlap_before_loading_data(self):
        with tempfile.TemporaryDirectory(dir=training.REPO) as temp:
            root = Path(temp)
            legacy.write_json(root/'identity.json', {'schema': training.SCHEMA})
            arm_root = root/'L'
            arm_root.mkdir()
            with legacy.run_lock(arm_root), patch.dict(os.environ, {training.WORKER_ENV: '1'}):
                with patch.object(training, 'prepare', side_effect=AssertionError('Overlapping worker loaded data')):
                    with self.assertRaisesRegex(ValueError, 'Another launcher owns'):
                        training.main(['--run-dir', str(root), '--device', 'cpu', '--worker', 'L'])


if __name__ == '__main__':
    unittest.main()
