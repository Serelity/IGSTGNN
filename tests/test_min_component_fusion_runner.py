"""F2 controls, reuse certificates, checkpoint-authority resume and isolation."""
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

from experiments.chronological import train_min_component_fusion as training
from experiments.chronological import train_min_forecast_fusion as previous
from experiments.chronological import train_internal_spatial as legacy
from tests import test_min_forecast_fusion_runner as source_fixture
from src.utils.igstgnn_published_events import EventPublishedData


def args_for(root, raw, source, reference, epochs=1):
    return SimpleNamespace(run_dir=root, data_dir=raw, source_run=source, reference_run=reference,
        graph_dir=training.HERE/'internal_spatial_graph_is11', protocol=training.PROTOCOL,
        arms=list(training.ARMS), only=None, worker=None, batch_size=None, device='cpu',
        threads=1, seed=2025, check=True, epochs=epochs, evaluate_after=False)


class ComponentFusionRunnerTests(unittest.TestCase):
    equal_nested = source_fixture.ForecastFusionRunnerTests.equal_nested

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        cls.logger = logging.getLogger('component_fusion_runner_tests')
        cls.logger.addHandler(logging.NullHandler())

    def setUp(self):
        source_fixture.ForecastFusionRunnerTests.setUp(self)
        fixture = self.protocol
        self.protocol = dict(legacy.read_json(training.PROTOCOL), **{k: fixture[k] for k in
            ('total_samples', 'train_samples', 'val_samples', 'test_samples', 'stations', 'batch_size')})
        self.data = lambda: EventPublishedData(self.raw, self.protocol, self.logger, check=True)

    def setup_run(self, name):
        root = self.root/name
        root.mkdir()
        args = args_for(root, self.raw, self.source_root, self.root/'missing_reference')
        source = training.load_source(args)
        with redirect_stdout(io.StringIO()):
            training.create_source_reference(args, self.data(), self.g, source)
        init = training.preflight(args, self.data(), self.g, source)
        identity = training.make_identity(args, self.protocol, source['identity']['package'], {'fixture': True}, init, source)
        return args, source, identity

    def test_cli_all_three_groups_only_absolute_epochs_and_immutable_source_paths(self):
        args = training.parse_args([])
        self.assertEqual(args.arms, ['CONTROL', 'STATE', 'EVENT'])
        self.assertEqual(args.epochs, 5)
        self.assertEqual(training.load_protocol(args)['fusion_spec']['width'], 16)
        self.assertEqual(training.load_protocol(args)['learning_rate'], 1e-4)
        for extra in (['--only', 'CONTROL', 'CONTROL'], ['--arms', 'CONTROL', '--only', 'EVENT'],
                      ['--epochs', '0'], ['--check', '--epochs', '2'], ['--batch-size', '0'],
                      ['--run-dir', str(self.source_root), '--source-run', str(self.source_root)],
                      ['--run-dir', str(self.root/'ref'), '--reference-run', str(self.root/'ref')]):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                training.parse_args(extra)
        with self.assertRaisesRegex(ValueError, 'Internal worker'):
            training.parse_args(['--run-dir', str(self.root/'new'), '--worker', 'CONTROL'])
        self.assertNotIn('--evaluate-after', training.child_command(args, 'CONTROL'))

    def test_initial_prediction_all_equal_and_control_has_no_router(self):
        args, source, identity = self.setup_run('initial')
        baseline = source['saved']['best_model_state']
        models = {arm: training.build_arm(args, arm, 4, self.g, 'cpu', source)[0].eval() for arm in training.ARMS}
        self.assertEqual(models['CONTROL'].state_dict().keys(), baseline.keys())
        self.equal_nested(models['STATE'].state_dict(), models['EVENT'].state_dict())
        batch = next(self.data().loaders['train_loader'].get_iterator())
        with torch.inference_mode():
            predicted = [m(torch.from_numpy(batch['x_data']), incident_data=training.event_tensors(batch['event_data'], 'cpu'))
                         for m in models.values()]
        self.assertTrue(all(torch.equal(predicted[0], value) for value in predicted))
        init = identity['initialization']
        self.assertEqual(init['initialization_source_forward_calls'], 1)
        self.assertEqual(init['full_source_validation_calls'], 0)
        self.assertFalse(identity['same_registered_parameter_layout'])
        self.assertTrue(identity['same_fused_registered_parameter_layout'])

    def test_control_one_optimizer_group_fused_two_and_non_event_ignores_report(self):
        args, source, _ = self.setup_run('optimizer')
        for arm in training.ARMS:
            model, _ = training.build_arm(args, arm, 4, self.g, 'cpu', source)
            engine = training.make_engine(args, arm, self.protocol, self.data(), model, self.logger)
            groups = engine._optimizer.param_groups
            self.assertEqual(len(groups), 1 if arm == 'CONTROL' else 2)
            self.assertEqual(groups[0]['lr'], 1e-4)
            self.assertFalse(engine._optimizer.state)
            if arm != 'CONTROL':
                self.assertEqual(groups[1]['lr'], 1e-3)
            grouped = [id(p) for g in groups for p in g['params']]
            self.assertEqual(len(grouped), len(set(grouped)))
            self.assertEqual(set(grouped), {id(p) for p in model.parameters()})
            batch = next(self.data().loaders['train_loader'].get_iterator())
            batch['event_data'] = object()
            if arm == 'EVENT':
                with self.assertRaises(AttributeError):
                    engine._prepare_batch(batch)
            else:
                self.assertIsNone(engine._prepare_batch(batch)[2])
            for _ in range(50):
                engine._update_curriculum_length()
                self.assertEqual(engine._cl_len, 12)

    def test_all_three_exact_resume_and_shared_training_order(self):
        orders = {}
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            for arm in training.ARMS:
                checkpoints = []
                for name, epochs in (('continuous', [2]), ('resumed', [1, 2])):
                    args, source, identity = self.setup_run(arm+'_'+name)
                    for epoch in epochs:
                        args.epochs = epoch
                        training.train_arm(args, arm, self.protocol, self.data(), identity, self.g, self.logger, source)
                    path = args.run_dir/arm/'last.pt'
                    checkpoints.append(torch.load(path, map_location='cpu', weights_only=False))
                    before = path.read_bytes()
                    result = training.train_arm(args, arm, self.protocol, self.data(), identity, self.g, self.logger, source)
                    self.assertEqual(result['optimizer_updates_this_invocation'], 0)
                    self.assertEqual(path.read_bytes(), before)
                left, right = checkpoints
                for key in ('model_state', 'best_model_state', 'optimizer_state', 'scheduler_state', 'best_metric',
                            'best_epoch', 'global_updates', 'curriculum_length', 'train_order', 'numpy_rng', 'torch_rng', 'cuda_rng'):
                    self.equal_nested(left[key], right[key])
                for a, b in zip(left['history'], right['history']):
                    self.equal_nested({k: v for k, v in a.items() if k != 'seconds'}, {k: v for k, v in b.items() if k != 'seconds'})
                orders[arm] = [r['train_order_sha256'] for r in left['history']]
        self.assertEqual(orders['CONTROL'], orders['STATE'])
        self.assertEqual(orders['STATE'], orders['EVENT'])

    def test_interrupted_epoch_does_not_commit_partial_state_or_sidecars(self):
        args, source, identity = self.setup_run('interrupted')
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            training.train_arm(args, 'CONTROL', self.protocol, self.data(), identity, self.g, self.logger, source)
        files = {n: (args.run_dir/'CONTROL'/n).read_bytes() for n in ('last.pt', 'history.json', 'summary.json')}
        args.epochs = 2
        with patch.object(training.ComponentEngine, 'evaluate', side_effect=RuntimeError('interrupted before epoch commit')), \
             redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                training.train_arm(args, 'CONTROL', self.protocol, self.data(), identity, self.g, self.logger, source)
        # Summary may be repaired to its authoritative committed checkpoint;
        # no second epoch appears and the atomic checkpoint itself is unchanged.
        self.assertEqual((args.run_dir/'CONTROL/last.pt').read_bytes(), files['last.pt'])
        self.assertEqual(legacy.read_json(args.run_dir/'CONTROL/history.json')[-1]['epoch'], 1)
        self.assertEqual(legacy.read_json(args.run_dir/'CONTROL/summary.json')['completed_epoch'], 1)

    def test_stage_zero_selection_and_checkpoint_semantics_guards(self):
        args, source, identity = self.setup_run('zero')
        with patch.object(training.ComponentEngine, 'evaluate', return_value=(1e6, .1, 2e6)), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            result = training.train_arm(args, 'CONTROL', self.protocol, self.data(), identity, self.g, self.logger, source)
        self.assertEqual(result['best_epoch'], 0)
        saved = torch.load(args.run_dir/'CONTROL/last.pt', map_location='cpu', weights_only=False)
        self.equal_nested(saved['best_model_state'], source['saved']['best_model_state'])
        corrupt = copy.deepcopy(saved)
        corrupt['global_updates'] += 1
        corrupt['history'][-1]['global_updates'] += 1
        with self.assertRaisesRegex(ValueError, 'update budget'):
            training.validate_saved(corrupt, identity, 'CONTROL')
        corrupt = copy.deepcopy(saved)
        corrupt['model_state']['forecast_fusion.fake'] = torch.zeros(1)
        corrupt['best_model_state']['forecast_fusion.fake'] = torch.zeros(1)
        with self.assertRaisesRegex(ValueError, 'semantics'):
            training.validate_saved(corrupt, identity, 'CONTROL')

    def test_identity_epoch_only_resume_and_input_changes_refused(self):
        args, source, identity = self.setup_run('identity')
        training.verify_identity(args, identity, identity['initialization'], allow_new=True)
        args.epochs, args.only = 9, ['EVENT']
        same = training.make_identity(args, self.protocol, identity['package'], identity['graph'], identity['initialization'], source)
        self.assertEqual(same, identity)
        self.assertNotIn('gpu', identity['runtime'])
        for key, value in (('batch_size', 99), ('source_sha256', {}), ('runtime', dict(identity['runtime'], torch='changed'))):
            with self.assertRaisesRegex(ValueError, 'identity changed'):
                training.verify_identity(args, dict(identity, **{key: value}), identity['initialization'], allow_new=False)

    def make_old_reference(self):
        args, source, _ = self.setup_run('old_reference')
        initialization = {'fixture': 'previous_F1'}
        old_args = SimpleNamespace(**vars(args))
        old_args.arms, old_args.protocol = ['STATE', 'EVENT'], previous.PROTOCOL
        identity = previous.make_identity(old_args, self.protocol, source['identity']['package'], {'fixture': True}, initialization, source)
        legacy.write_json(args.run_dir/'identity.json', identity)
        return args.run_dir

    def test_certified_reference_reuse_has_zero_source_forwards_and_checks_axes(self):
        reference = self.make_old_reference()
        destination = self.root/'reused'
        destination.mkdir()
        args = args_for(destination, self.raw, self.source_root, reference)
        source = training.load_source(args)
        before = {p.name: p.read_bytes() for p in reference.iterdir() if p.is_file()}
        with patch.object(training.shared, 'create_source_reference', side_effect=AssertionError('repeated source inference')), \
             patch.object(training, 'build_core', side_effect=AssertionError('repeated source inference')), \
             redirect_stdout(io.StringIO()):
            cert = training.create_source_reference(args, self.data(), self.g, source)
        self.assertTrue(cert['reused_archive'])
        self.assertEqual(cert['source_full_validation_calls'], 0)
        self.assertEqual((destination/'source_reference.npz').read_bytes(), (reference/'source_reference.npz').read_bytes())
        self.assertEqual(before, {p.name: p.read_bytes() for p in reference.iterdir() if p.is_file()})
        altered_data = self.data()
        altered_data.station_ids = altered_data.station_ids[::-1]
        with self.assertRaisesRegex(ValueError, 'station axes'):
            training.create_source_reference(args, altered_data, self.g, source)

    def test_incomplete_or_corrupt_reference_never_silently_runs_source(self):
        args = args_for(self.root/'new', self.raw, self.source_root, self.root/'broken_reference')
        args.run_dir.mkdir()
        args.reference_run.mkdir()
        (args.reference_run/'identity.json').write_text('{}')
        with patch.object(training.shared, 'create_source_reference', side_effect=AssertionError('must not infer')):
            with self.assertRaisesRegex(ValueError, 'Incomplete reference'):
                training.create_source_reference(args, self.data(), self.g, training.load_source(args))

    def test_parent_resume_only_uses_selected_worker_and_no_initialization_forwards(self):
        args, source, identity = self.setup_run('parent')
        training.verify_identity(args, identity, identity['initialization'], allow_new=True)
        with patch.object(training, 'prepare', return_value=(self.protocol, self.data(), self.g, identity['initialization'], identity, source)) as prepare, \
             patch.object(training, 'create_source_reference', side_effect=AssertionError('source inference repeated')), \
             patch.object(training, 'run_worker', return_value=0) as worker, \
             patch.object(training, 'comparison', return_value={'status': 'fixture'}), redirect_stdout(io.StringIO()):
            training.main(['--run-dir', str(args.run_dir), '--source-run', str(self.source_root), '--reference-run', str(args.reference_run),
                           '--data-dir', str(self.raw), '--device', 'cpu', '--threads', '1', '--check', '--only', 'EVENT'])
        self.assertEqual(worker.call_count, 1)
        self.assertEqual(worker.call_args.args[1], 'EVENT')
        self.assertTrue(prepare.call_args.kwargs['skip_preflight'])


if __name__ == '__main__':
    unittest.main()
