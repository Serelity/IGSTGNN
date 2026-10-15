"""Replacement-operator stage zero, paired budgets and checkpoint-authority resume."""
import copy
from contextlib import redirect_stderr, redirect_stdout
import io
import json
import logging
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.chronological import train_min_conditional_propagation as training
from experiments.chronological import evaluate_min_conditional_propagation as evaluation
from experiments.chronological import train_internal_spatial as legacy
from experiments.chronological.smoke import sha256
from tests import test_min_forecast_fusion_runner as source_fixture
from src.utils.igstgnn_published_events import EventPublishedData


def args_for(root, raw, source, epochs=1):
    return SimpleNamespace(run_dir=root, data_dir=raw, source_run=source, reference_run=root.parent/'missing_reference',
        graph_dir=training.HERE/'internal_spatial_graph_is11', protocol=training.PROTOCOL,
        arms=list(training.ARMS), only=None, worker=None, batch_size=None, device='cpu', threads=1,
        seed=2025, check=True, epochs=epochs, receiver_chunk_size=2, checkpoint_messages=True,
        no_checkpoint_messages=False)


class ConditionalPropagationRunnerTests(unittest.TestCase):
    equal_nested = source_fixture.ForecastFusionRunnerTests.equal_nested

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        cls.logger = logging.getLogger('conditional_propagation_runner_tests')
        cls.logger.addHandler(logging.NullHandler())

    def setUp(self):
        source_fixture.ForecastFusionRunnerTests.setUp(self)
        fixture = self.protocol
        self.protocol = dict(legacy.read_json(training.PROTOCOL), **{k: fixture[k] for k in
            ('total_samples', 'train_samples', 'val_samples', 'test_samples', 'stations', 'batch_size')})
        self.data = lambda: EventPublishedData(self.raw, self.protocol, self.logger, check=True)
        self.adjacency = np.load(self.raw/'adj_matrix.npy', allow_pickle=False)

    def setup_run(self, name):
        root = self.root/name
        root.mkdir()
        args = args_for(root, self.raw, self.source_root)
        source = training.load_source(args)
        initialization = training.preflight(args, self.data(), self.g, source, self.adjacency)
        identity = training.make_identity(args, self.protocol, source['identity']['package'], {'fixture': True}, initialization, source)
        return args, source, identity

    def quiet_train(self, args, arm, identity, source):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return training.train_arm(args, arm, self.protocol, self.data(), identity, self.g, self.logger, source, self.adjacency)

    def test_cli_default_one_epoch_and_resource_resume_identity(self):
        args = training.parse_args([])
        self.assertEqual(args.epochs, 1)
        self.assertEqual(args.arms, ['HISTORY', 'M_ONLY', 'EVENT'])
        self.assertEqual(args.receiver_chunk_size, 4)
        self.assertTrue(args.checkpoint_messages)
        self.assertEqual(training.load_protocol(args)['learning_rate'], 1e-4)
        for command in (['--epochs', '0'], ['--check', '--epochs', '2'], ['--receiver-chunk-size', '0'],
            ['--arms', 'HISTORY', '--only', 'EVENT'], ['--only', 'EVENT', 'EVENT'], ['--batch-size', '0'],
            ['--source-run', str(self.source_root), '--run-dir', str(self.source_root)]):
            with self.subTest(command=command), self.assertRaises(ValueError):
                training.parse_args(command)
        with self.assertRaisesRegex(ValueError, 'Internal worker'):
            training.parse_args(['--worker', 'EVENT', '--run-dir', str(self.root/'new')])
        self.assertNotIn('--evaluate-after', training.child_command(args, 'EVENT'))
        args, source, identity = self.setup_run('identity')
        training.verify_identity(args, identity, identity['initialization'], allow_new=True)
        args.epochs, args.only = 10, ['EVENT']
        self.equal_nested(identity, training.make_identity(args, self.protocol, identity['package'], identity['graph'], identity['initialization'], source))
        args.receiver_chunk_size += 1
        changed = training.make_identity(args, self.protocol, identity['package'], identity['graph'], identity['initialization'], source)
        with self.assertRaisesRegex(ValueError, 'identity changed'):
            training.verify_identity(args, changed, changed['initialization'])

    def test_candidate_own_stage_zero_and_equal_registered_initialization(self):
        args, source, identity = self.setup_run('stagezero')
        before = training.source_snapshot(args)
        initial = identity['initialization']
        self.assertTrue(initial['registered_state_and_predictions_equal'])
        self.assertFalse(initial['old_MIN_stage_zero_equivalence_claim'])
        self.assertEqual(initial['source_validation_forward_calls'], 0)
        self.assertEqual(len({r['initial_state_sha256'] for r in initial['arms'].values()}), 1)
        with patch.object(training.PropagationEngine, 'evaluate', side_effect=[(100., .2, 120.), (101., .3, 121.)]):
            result = self.quiet_train(args, 'EVENT', identity, source)
        self.assertEqual(result['best_epoch'], 0)
        self.assertEqual(result['best_metric'], 100.)
        saved = torch.load(args.run_dir/'EVENT/last.pt', map_location='cpu', weights_only=False)
        self.assertNotEqual(saved['best_metric'], identity['source']['best_metrics'][0])
        initial_model, _ = training.build_arm(args, 'EVENT', 4, self.g, 'cpu', source, self.adjacency)
        self.equal_nested(saved['best_model_state'], initial_model.state_dict())
        self.assertEqual(before, training.source_snapshot(args))

    def test_optimizer_all_joint_and_history_does_not_convert_reports(self):
        args, source, _ = self.setup_run('optimizer')
        for arm in training.ARMS:
            model, _ = training.build_arm(args, arm, 4, self.g, 'cpu', source, self.adjacency)
            engine = training.make_engine(args, arm, self.protocol, self.data(), model, self.logger)
            self.assertEqual(len(engine._optimizer.param_groups), 1)
            self.assertEqual(engine._optimizer.param_groups[0]['lr'], 1e-4)
            self.assertFalse(engine._optimizer.state)
            parameters = engine._optimizer.param_groups[0]['params']
            self.assertEqual({id(p) for p in model.parameters()}, {id(p) for p in parameters})
            self.assertTrue(all(p.requires_grad for p in parameters))
            batch = next(self.data().loaders['train_loader'].get_iterator())
            batch['event_data'] = object()
            if arm == 'HISTORY':
                self.assertIsNone(engine._prepare_batch(batch)[2])
            else:
                with self.assertRaises(AttributeError):
                    engine._prepare_batch(batch)
            for _ in range(20):
                engine._update_curriculum_length()
                self.assertEqual(engine._cl_len, 12)

    def test_exact_resume_noop_and_common_training_order(self):
        orders = {}
        for arm in training.ARMS:
            checkpoints = []
            for name, targets in (('continuous', [2]), ('resume', [1, 2])):
                args, source, identity = self.setup_run(arm+'_'+name)
                for target in targets:
                    args.epochs = target
                    self.quiet_train(args, arm, identity, source)
                path = args.run_dir/arm/'last.pt'
                checkpoints.append(torch.load(path, map_location='cpu', weights_only=False))
                before = path.read_bytes()
                with patch.object(training, 'build_arm', side_effect=AssertionError('no-op must not construct model')):
                    result = self.quiet_train(args, arm, identity, source)
                self.assertEqual(result['optimizer_updates_this_invocation'], 0)
                self.assertEqual(before, path.read_bytes())
            left, right = checkpoints
            for key in ('model_state', 'best_model_state', 'optimizer_state', 'scheduler_state', 'best_metric',
                'best_epoch', 'global_updates', 'train_order', 'numpy_rng', 'torch_rng', 'cuda_rng', 'stage_zero'):
                self.equal_nested(left[key], right[key])
            for a, b in zip(left['history'], right['history']):
                self.equal_nested({k: v for k, v in a.items() if k != 'seconds'}, {k: v for k, v in b.items() if k != 'seconds'})
            orders[arm] = [r['train_order_sha256'] for r in left['history']]
        self.assertEqual(orders['HISTORY'], orders['M_ONLY'])
        self.assertEqual(orders['HISTORY'], orders['EVENT'])

    def test_interrupted_epoch_and_corrupt_checkpoint_guards(self):
        args, source, identity = self.setup_run('interruption')
        self.quiet_train(args, 'HISTORY', identity, source)
        path = args.run_dir/'HISTORY/last.pt'
        original = path.read_bytes()
        args.epochs = 2
        with patch.object(training.PropagationEngine, 'evaluate', side_effect=RuntimeError('interrupted')):
            with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                self.quiet_train(args, 'HISTORY', identity, source)
        self.assertEqual(path.read_bytes(), original)
        saved = torch.load(path, map_location='cpu', weights_only=False)
        bad = copy.deepcopy(saved)
        bad['global_updates'] += 1
        with self.assertRaisesRegex(ValueError, 'update budget'):
            training.validate_saved(bad, identity, 'HISTORY')
        bad = copy.deepcopy(saved)
        bad['stage_zero']['initial_state_sha256'] = 'bad'
        with self.assertRaisesRegex(ValueError, 'Stage-zero state'):
            training.validate_saved(bad, identity, 'HISTORY')

    def test_intervention_routes_input_preservation_and_macro_metrics(self):
        args, source, _ = self.setup_run('interventions')
        batch = next(self.data().loaders['val_loader'].get_iterator())
        digest = evaluation.input_digest(batch)
        for arm, case in zip(training.ARMS, evaluation.CASES[:3]):
            model, _ = training.build_arm(args, arm, 4, self.g, 'cpu', source, self.adjacency)
            model.eval()
            cases = (case,)+evaluation.CASES[3:] if arm == 'EVENT' else (case,)
            for current in cases:
                prediction, statistics = evaluation.forward_case(model, batch, self.data().scaler, current)
                self.assertEqual(prediction.shape, (len(batch['x_data']), 12, 4))
                self.assertEqual(statistics['forward_calls'], 1)
            self.assertEqual(digest, evaluation.input_digest(batch))
        target = np.ones((2, 12, 4), np.float32)
        valid = np.ones_like(target, bool)
        valid[0, 0, 0] = False
        better, worse = target+1, target+3
        regions = {'all_nodes': np.ones((2, 4), bool)}
        result = evaluation.summarize({'HISTORY_ON': worse, 'M_ONLY_ON': worse, 'EVENT_ON': better,
            'EVENT_REPORT_OFF': worse}, target, valid, regions, np.arange(2))
        self.assertEqual(result['cases']['EVENT_ON']['mae_pooled'], 1.)
        self.assertEqual(result['comparisons']['EVENT_vs_HISTORY']['groups']['all_nodes']['mae_gain_macro'], 2.)
        self.assertIsNone(result['fixed_hard_sample_indices'])

    def test_source_archive_reuse_never_falls_back_to_inference(self):
        args, source, identity = self.setup_run('sourcearchive')
        data = self.data()
        batches, axes, mask = evaluation.validation_axes(data)
        partitions = [len(b['x_data']) for b in batches]
        actual, certificate = evaluation.load_source_archive(args, identity, axes, partitions, mask)
        self.assertIsNone(actual)
        self.assertEqual(certificate['new_inference_calls'], 0)
        args.reference_run.mkdir()
        with patch.object(training, 'build_core', side_effect=AssertionError('Source inference forbidden')):
            prediction = axes['target']+1
            np.savez_compressed(args.reference_run/'source_reference.npz', prediction=prediction, **axes)
            from src.utils.metrics import compute_all_metrics
            values = list(compute_all_metrics(torch.from_numpy(prediction), torch.from_numpy(axes['target']), torch.tensor(mask)))
            identity['source']['best_metrics'] = values
            certificate = dict(status='SOURCE_MIN_A_BEST_FULL_ON_REPLAY_PASS', source_best_epoch=identity['source']['best_epoch'],
                source_best_model_sha256=identity['source']['best_model_sha256'], engineering_only=True,
                archive_sha256=sha256(args.reference_run/'source_reference.npz'), batch_partition=partitions, mask_value=mask, actual=values)
            training.write_json(args.reference_run/'source_reference.json', certificate)
            actual, result = evaluation.load_source_archive(args, identity, axes, partitions, mask)
            np.testing.assert_array_equal(actual, prediction)
            self.assertFalse(result['new_checkpoint_replay'])
            broken = dict(axes, station_ids=axes['station_ids'][::-1])
            with self.assertRaisesRegex(ValueError, 'axes/targets differ'):
                evaluation.load_source_archive(args, identity, broken, partitions, mask)

    def test_empty_invalid_and_nonpositive_valid_targets_rejected(self):
        target = np.ones((2, 12, 4), np.float32)
        valid = np.ones_like(target, bool)
        for value, mask in ((target, np.zeros_like(valid)), (target[:0], valid[:0]),
            (np.full_like(target, -2.), valid), (np.zeros_like(target), valid)):
            with self.subTest(shape=value.shape, value=value.flat[0] if value.size else None), self.assertRaises(ValueError):
                evaluation.metrics(value, value, mask)
        empty = SimpleNamespace(loaders={'val_loader': SimpleNamespace(get_iterator=lambda: iter(()))})
        with self.assertRaisesRegex(ValueError, 'nonempty batches'):
            evaluation.validation_axes(empty)

    def test_full_saved_checkpoint_evaluation_and_preservation(self):
        args, source, identity = self.setup_run('full_eval')
        training.verify_identity(args, identity, identity['initialization'], allow_new=True)
        for arm in training.ARMS:
            self.quiet_train(args, arm, identity, source)
        before = {str(p): sha256(p) for p in args.run_dir.rglob('*') if p.is_file()}
        request = SimpleNamespace(run_dir=args.run_dir, output_dir=args.run_dir/'evaluation', device='cpu', threads=1, check=True)
        with patch.object(training, 'prepare', return_value=(self.protocol, self.data(), self.g, self.adjacency,
            identity['initialization'], identity, source)), patch.object(training.legacy, 'load_graph',
            return_value=(self.g, {'fixture_road': [0, 1, 2, 3]}, {'fixture': True})), \
            redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            result = evaluation.evaluate_run(request)
        self.assertEqual(result['status'], 'MIN_CONDITIONAL_PROPAGATION_EVALUATION_PASS')
        self.assertTrue(result['on_exactly_restored'])
        self.assertEqual(result['new_inference_calls']['SOURCE_MIN'], 0)
        self.assertNotIn('SOURCE_MIN', result['cases'])
        self.assertEqual(before, {name: sha256(Path(name)) for name in before})
        for name in ('report.md', 'analysis.json', 'paired_predictions.npz', 'mechanisms.json', 'replay.json', 'preservation.json'):
            self.assertTrue((request.output_dir/name).is_file(), name)
        with np.load(request.output_dir/'paired_predictions.npz', allow_pickle=False) as archive:
            self.assertEqual(archive['EVENT_ON'].shape, archive['target'].shape)
            minimum = float(archive['target'].min())
            np.testing.assert_array_equal(archive['valid'], archive['target'] != (minimum if minimum < 1 else 0.))
        mechanism = legacy.read_json(request.output_dir/'mechanisms.json')
        self.assertTrue(mechanism['no_legacy_latent_exchange_audit'])


if __name__ == '__main__':
    unittest.main()
