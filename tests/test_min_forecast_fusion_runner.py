"""MIN A best warmstarts, source protection and exact single-pass stage resume."""
import copy
from contextlib import redirect_stderr, redirect_stdout
import io
import logging
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.chronological import train_min_forecast_fusion as training
from experiments.chronological import train_min_event_budget as source_training
from experiments.chronological import train_internal_spatial as legacy
from experiments.chronological.smoke import sha256
from src.models.incident_spatial_residual import tensor_state_hash
from src.utils.igstgnn_published_events import EventPublishedData
from tests.test_min_event_budget_data import event_fixture
from tests.test_min_event_budget_runner import tiny_minimal
from tests.test_incident_spatial import graph


def args_for(root, raw, source, *, epochs=1):
    return SimpleNamespace(run_dir=root, data_dir=raw, source_run=source,
        graph_dir=training.HERE/'internal_spatial_graph_is11', protocol=training.PROTOCOL,
        arms=list(training.ARMS), only=None, worker=None, batch_size=None,
        device='cpu', threads=1, seed=2025, check=True, epochs=epochs, evaluate_after=False)


class ForecastFusionRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        cls.logger = logging.getLogger('forecast_fusion_runner_tests')
        cls.logger.addHandler(logging.NullHandler())

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=training.REPO)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.raw, self.source_root = self.root/'raw', self.root/'source'
        self.raw.mkdir()
        self.source_root.mkdir()
        (self.source_root/'A').mkdir()
        (self.source_root/'active.lock').write_bytes(b'0')
        _, fixture_protocol = event_fixture(self.raw)
        self.protocol = dict(legacy.read_json(training.PROTOCOL), **{key: fixture_protocol[key]
            for key in ('total_samples', 'train_samples', 'val_samples', 'test_samples', 'stations', 'batch_size')})
        self.g = graph(4)
        self.builder = patch.object(training.paper, 'build_model', side_effect=tiny_minimal)
        self.builder.start()
        self.addCleanup(self.builder.stop)
        setup = args_for(self.root/'unused', self.raw, self.source_root)
        setup.protocol = source_training.PROTOCOL
        core, _ = tiny_minimal(setup, 4, self.g, 'cpu')
        core.eval()
        self.data = lambda: EventPublishedData(self.raw, self.protocol, self.logger, check=True)
        arrays, native = training.validation_predictions(core, self.data(), torch.device('cpu'))
        source_identity = dict(schema=source_training.SCHEMA, arm_modes={'A': 'minimal'}, engineering_check=True,
            seed=2025, batch_size=2, runtime=dict(torch=str(torch.__version__), cuda=torch.version.cuda, device_type='cpu'),
            source_sha256=source_training.source_fingerprints(), protocol_sha256=sha256(source_training.PROTOCOL),
            protocol=dict(legacy.read_json(source_training.PROTOCOL), train_samples=14, val_samples=3, stations=4, batch_size=2),
            graph_inputs_sha256={name: sha256(setup.graph_dir/name) for name in ('graph.json', 'stations.csv', 'source_summary.json')},
            package={name: sha256(self.raw/name) for name in self.protocol['data_sha256']}, initialization={'fixture': 'source'})
        source_state = copy.deepcopy(core.state_dict())
        last_state = copy.deepcopy(source_state)
        last_state['embedding.weight'] += .1
        history = [dict(epoch=1, validation_mae=native['actual'][0], validation_mape=native['actual'][1],
            validation_rmse=native['actual'][2], best_epoch=1, best_metric=native['actual'][0], global_updates=2),
            dict(epoch=2, validation_mae=native['actual'][0]+1., validation_mape=native['actual'][1],
            validation_rmse=native['actual'][2], best_epoch=1, best_metric=native['actual'][0], global_updates=4)]
        saved = dict(identity=source_identity, arm='A', completed_epoch=2, history=history,
            model_state=last_state, best_model_state=source_state, best_metric=native['actual'][0], global_updates=4,
            optimizer_state={'must_not_restore': True}, scheduler_state={'must_not_restore': True})
        torch.save(saved, self.source_root/'A/last.pt')
        legacy.write_json(self.source_root/'identity.json', source_identity)
        legacy.write_json(self.source_root/'initialization.json', source_identity['initialization'])
        legacy.write_json(self.source_root/'data_audit.json', self.data().audit)
        legacy.write_json(self.source_root/'A/history.json', history)
        legacy.write_json(self.source_root/'A/summary.json', dict(best_epoch=1, best_metric=saved['best_metric'], completed_epoch=2))

    def setup_run(self, name, epochs=1):
        root = self.root/name
        root.mkdir()
        args = args_for(root, self.raw, self.source_root, epochs=epochs)
        source = training.load_source(args)
        with redirect_stdout(io.StringIO()):
            training.create_source_reference(args, self.data(), self.g, source)
        certificate = training.preflight(args, self.data(), self.g, source)
        identity = training.make_identity(args, self.protocol, source['identity']['package'], {'fixture': True}, certificate, source)
        return args, source, identity

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

    def test_cli_defaults_full_horizon_protocol_and_old_run_guards(self):
        args = training.parse_args([])
        self.assertEqual(args.arms, ['STATE', 'EVENT'])
        self.assertEqual(args.epochs, 5)
        self.assertEqual(args.source_run, training.DEFAULT_SOURCE)
        protocol = training.load_protocol(args)
        self.assertEqual(protocol['learning_rate'], 1e-4)
        self.assertEqual(protocol['fusion_learning_rate'], 1e-3)
        self.assertEqual(protocol['warm_epoch'], 0)
        self.assertEqual(protocol['batch_size'], 64)
        for command in (['--source-run', str(self.source_root), '--run-dir', str(self.source_root)],
                        ['--arms', 'STATE', '--only', 'EVENT'], ['--check', '--epochs', '2'], ['--batch-size', '0']):
            with self.subTest(command=command), self.assertRaises(ValueError):
                training.parse_args(command)
        with self.assertRaisesRegex(ValueError, 'Internal worker'):
            training.parse_args(['--worker', 'EVENT', '--run-dir', str(self.root/'new')])
        self.assertNotIn('--evaluate-after', training.child_command(args_for(self.root/'new', self.raw, self.source_root), 'EVENT'))

    def test_active_source_ON_preflight_uses_best_weights_and_preserves_original_bytes(self):
        before = training.source_snapshot(args_for(self.root/'new', self.raw, self.source_root))
        args, source, identity = self.setup_run('preflight')
        self.assertEqual(source['selected']['epoch'], 1)
        self.assertEqual(source['saved']['completed_epoch'], 2)
        models = [training.build_arm(args, arm, 4, self.g, 'cpu', source)[0].eval() for arm in training.ARMS]
        self.equal_nested(models[0].state_dict(), models[1].state_dict())
        self.assertTrue(all(p.requires_grad for model in models for p in model.parameters()))
        self.assertTrue(all(not name.startswith('core.') for name in models[0].state_dict()))
        batch = next(self.data().loaders['train_loader'].get_iterator())
        x, events = torch.from_numpy(batch['x_data']), training.event_tensors(batch['event_data'], 'cpu')
        with torch.inference_mode():
            on = models[0](x, incident_data=events)
            reference = training.build_core(args, 4, self.g, 'cpu', source).eval()(x)
        self.assertTrue(torch.equal(on, reference))
        self.assertTrue(torch.equal(models[0].embedding.weight, source['saved']['best_model_state']['embedding.weight']))
        self.assertFalse(torch.equal(models[0].embedding.weight, source['saved']['model_state']['embedding.weight']))
        self.assertTrue(identity['initialization']['same_parameter_layout_and_initialization'])
        self.assertFalse(identity['source_optimizer_restored'])
        self.assertFalse(identity['source_exact_resume'])
        self.assertNotIn('gpu', identity['runtime'])
        self.assertEqual(before, training.source_snapshot(args))

    def test_bad_source_full_replay_blocks_reference_before_any_training(self):
        args = args_for(self.root/'bad_replay', self.raw, self.source_root)
        args.run_dir.mkdir()
        source = training.load_source(args)
        source['selected']['validation_mae'] += 1.
        with self.assertRaisesRegex(ValueError, 'full ON replay failed'):
            training.create_source_reference(args, self.data(), self.g, source)
        self.assertFalse((args.run_dir/'source_reference.npz').exists())
        self.assertFalse(any(args.run_dir.glob('*/last.pt')))

    def test_fresh_optimizer_groups_joint_training_and_full_horizon_loss(self):
        args, source, identity = self.setup_run('optimizer')
        model, _ = training.build_arm(args, 'EVENT', 4, self.g, 'cpu', source)
        engine = training.make_engine(args, 'EVENT', self.protocol, self.data(), model, self.logger)
        self.assertEqual(engine._optimizer.state, {})
        self.assertEqual(engine._optimizer.param_groups[0]['lr'], 1e-4)
        self.assertEqual(engine._optimizer.param_groups[1]['lr'], 1e-3)
        all_ids = {id(p) for p in model.parameters()}
        grouped = [id(p) for group in engine._optimizer.param_groups for p in group['params']]
        self.assertEqual(set(grouped), all_ids)
        self.assertEqual(len(grouped), len(all_ids))
        self.assertEqual(set(map(id, model.forecast_fusion.parameters())),
                         set(map(id, engine._optimizer.param_groups[1]['params'])))
        for _ in range(100):
            engine._update_curriculum_length()
            self.assertEqual(engine._cl_len, 12)
        batch = next(self.data().loaders['train_loader'].get_iterator())
        state, _ = training.build_arm(args, 'STATE', 4, self.g, 'cpu', source)
        state_engine = training.make_engine(args, 'STATE', self.protocol, self.data(), state, self.logger)
        # A STATE training forward must not even convert malformed event data.
        batch['event_data'] = object()
        self.assertIsNone(state_engine._prepare_batch(batch)[2])
        with self.assertRaises(AttributeError):
            engine._prepare_batch(batch)

    def test_continuous_and_resumed_stages_match_exactly_for_both_arms(self):
        before = training.source_snapshot(args_for(self.root/'unused', self.raw, self.source_root))
        orders = {}
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            for arm in training.ARMS:
                saved_states = []
                for suffix, stages in (('continuous', [2]), ('resumed', [1, 2])):
                    args, source, identity = self.setup_run(arm+'_'+suffix)
                    for epoch in stages:
                        args.epochs = epoch
                        training.train_arm(args, arm, self.protocol, self.data(), identity, self.g, self.logger, source)
                    checkpoint = args.run_dir/arm/'last.pt'
                    saved_states.append(torch.load(checkpoint, map_location='cpu', weights_only=False))
                    original = checkpoint.read_bytes()
                    no_op = training.train_arm(args, arm, self.protocol, self.data(), identity, self.g, self.logger, source)
                    self.assertEqual(no_op['optimizer_updates_this_invocation'], 0)
                    self.assertEqual(original, checkpoint.read_bytes())
                a, b = saved_states
                for key in ('model_state', 'best_model_state', 'optimizer_state', 'scheduler_state', 'best_metric', 'best_epoch',
                            'global_updates', 'curriculum_length', 'train_order', 'numpy_rng', 'torch_rng', 'cuda_rng'):
                    self.equal_nested(a[key], b[key])
                for left, right in zip(a['history'], b['history']):
                    self.equal_nested({key: value for key, value in left.items() if key != 'seconds'},
                                      {key: value for key, value in right.items() if key != 'seconds'})
                orders[arm] = [row['train_order_sha256'] for row in a['history']]
                self.assertEqual(a['global_updates'], 4)
                self.assertEqual(a['curriculum_length'], 12)
        self.assertEqual(orders['STATE'], orders['EVENT'])
        self.assertEqual(before, training.source_snapshot(args_for(self.root/'unused', self.raw, self.source_root)))

    def test_epoch_zero_remains_eligible_and_checkpoint_best_is_native_source(self):
        args, source, identity = self.setup_run('epoch_zero')
        with patch.object(training.FusionEngine, 'evaluate', return_value=(1e6, .1, 2e6)), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            summary = training.train_arm(args, 'EVENT', self.protocol, self.data(), identity, self.g, self.logger, source)
        self.assertEqual(summary['best_epoch'], 0)
        self.assertTrue(summary['best_is_source_epoch_zero'])
        saved = torch.load(args.run_dir/'EVENT/last.pt', map_location='cpu', weights_only=False)
        for key, value in source['saved']['best_model_state'].items():
            self.equal_nested(saved['best_model_state'][key], value)
        self.assertFalse(all(torch.equal(saved['model_state'][name], saved['best_model_state'][name])
                             for name in saved['model_state']))
        training.validate_saved(saved, identity, 'EVENT')
        corrupt = copy.deepcopy(saved)
        corrupt['best_metric'] -= 1.
        corrupt['history'][-1]['best_metric'] = corrupt['best_metric']
        with self.assertRaisesRegex(ValueError, 'eligible source-zero'):
            training.validate_saved(corrupt, identity, 'EVENT')
        corrupt = copy.deepcopy(saved)
        key = next(k for k in corrupt['model_state'] if k.startswith('forecast_fusion.'))
        corrupt['model_state'][key].view(-1)[0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'Invalid checkpoint tensor'):
            training.validate_saved(corrupt, identity, 'EVENT')

    def test_new_identity_rejects_batch_source_or_runtime_drift_and_accepts_epoch_only_resume(self):
        args, source, identity = self.setup_run('identity')
        legacy.write_json(args.run_dir/'identity.json', identity)
        legacy.write_json(args.run_dir/'initialization.json', identity['initialization'])
        args.epochs, args.only = 9, ['EVENT']
        same = training.make_identity(args, self.protocol, identity['package'], identity['graph'], identity['initialization'], source)
        self.assertEqual(same, identity)
        training.verify_identity(args, same, identity['initialization'], allow_new=False)
        for field, value in (('batch_size', 99), ('runtime', dict(identity['runtime'], torch='changed'))):
            changed = dict(identity, **{field: value})
            with self.assertRaisesRegex(ValueError, 'identity changed'):
                training.verify_identity(args, changed, identity['initialization'], allow_new=False)
        (args.run_dir/'source_reference.npz').write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'reference artifacts changed'):
            training.verify_source_reference(args, identity)

    def test_parent_resume_reuses_source_cache_and_only_runs_requested_worker(self):
        args, source, identity = self.setup_run('parent_cache')
        legacy.write_json(args.run_dir/'identity.json', identity)
        legacy.write_json(args.run_dir/'initialization.json', identity['initialization'])
        with patch.object(training, 'prepare', return_value=(self.protocol, self.data(), self.g, identity['initialization'], identity, source)), \
             patch.object(training, 'create_source_reference', side_effect=AssertionError('Repeated source full validation replay')), \
             patch.object(training, 'run_worker', return_value=0) as worker, \
             patch.object(training, 'comparison', return_value={'status': 'fixture'}), \
             redirect_stdout(io.StringIO()):
            training.main(['--run-dir', str(args.run_dir), '--source-run', str(self.source_root), '--data-dir', str(self.raw),
                           '--device', 'cpu', '--threads', '1', '--check', '--only', 'EVENT'])
        self.assertEqual(worker.call_count, 1)
        self.assertEqual(worker.call_args.args[1], 'EVENT')

    def test_comparison_uses_actual_stage_zero_reference_and_separates_complete_from_equal_budget(self):
        args, source, identity = self.setup_run('comparison')
        baseline = legacy.read_json(args.run_dir/'source_reference.json')['actual'][0]
        identity['source']['best_metrics'][0] = baseline+1e-6
        for arm in training.ARMS:
            (args.run_dir/arm).mkdir()
            (args.run_dir/arm/'last.pt').touch()
        histories = {'STATE': [dict(validation_mae=baseline+2e-6, train_order_sha256='a'*64)],
                     'EVENT': [dict(validation_mae=baseline-5e-7, train_order_sha256='a'*64),
                                 dict(validation_mae=baseline+1e-6, train_order_sha256='b'*64)]}
        def load(path, **kwargs):
            arm = Path(path).parent.name
            return dict(arm=arm, history=histories[arm])
        def summary(saved, args, protocol):
            count = len(saved['history'])
            return dict(completed_epoch=count, global_updates=2*count, early_stopped=True)
        with patch.object(training.torch, 'load', side_effect=load), patch.object(training, 'validate_saved'), \
             patch.object(training, 'summary_for', side_effect=summary):
            result = training.comparison(args, identity)
        self.assertIn('COMPLETE', result['status'])
        self.assertFalse(result['same_completed_training_budget'])
        self.assertIn('unequal_completed_stage_budget', result['independent_best_scope'])
        self.assertAlmostEqual(result['comparisons']['EVENT_vs_STATE']['common_prefix_best_mae_gain'], 5e-7, places=10)

    def test_source_cache_reader_validates_hash_mask_axes_and_no_model_calls(self):
        args, source, identity = self.setup_run('cache_reader')
        with patch.object(training, 'build_core', side_effect=AssertionError('Cache reader must not load a model')):
            certificate, arrays = training.load_source_reference(args, identity)
        self.assertEqual(certificate['source_best_epoch'], 1)
        self.assertEqual(arrays['target'].shape, (3, 12, 4))
        self.assertEqual(arrays['valid'].dtype, np.bool_)
        self.assertEqual(certificate['status'], 'SOURCE_MIN_A_BEST_FULL_ON_REPLAY_PASS')
        arrays['valid'] = arrays['valid'].astype(np.float32)
        np.savez_compressed(args.run_dir/'source_reference.npz', **arrays)
        # Updating a sidecar hash cannot make a non-Boolean validity mask legal.
        certificate['archive_sha256'] = sha256(args.run_dir/'source_reference.npz')
        legacy.write_json(args.run_dir/'source_reference.json', certificate)
        identity['source_reference_files_sha256'] = {name: sha256(args.run_dir/name)
            for name in ('source_reference.json', 'source_reference.npz')}
        with self.assertRaisesRegex(ValueError, 'prediction/mask/axis'):
            training.load_source_reference(args, identity)

    def test_source_batch_mismatch_stops_prepare_before_preflight(self):
        args = args_for(self.root/'bad_batch', self.raw, self.source_root)
        source = training.load_source(args)
        source['identity']['batch_size'] = 99
        with patch.object(training, 'load_protocol', return_value=self.protocol), \
             patch.object(training.paper, 'verify_release', return_value=source['identity']['package']), \
             patch.object(training.legacy, 'load_graph', return_value=(self.g, None, {'fixture': True})), \
             patch.object(training, 'load_source', return_value=source), \
             patch.object(training, 'preflight', side_effect=AssertionError('Batch mismatch must stop first')):
            with self.assertRaisesRegex(ValueError, 'source original'):
                training.prepare(args, self.logger)

    def test_source_lock_is_read_only_and_active_source_is_rejected(self):
        args = args_for(self.root/'lock_check', self.raw, self.source_root)
        original = (self.source_root/'active.lock').read_bytes()
        with training.readonly_source_lock(self.source_root):
            with self.assertRaisesRegex(ValueError, 'Source MIN A run is active'):
                with training.readonly_source_lock(self.source_root):
                    pass
        self.assertEqual((self.source_root/'active.lock').read_bytes(), original)
        (self.source_root/'active.lock').unlink()
        with self.assertRaisesRegex(ValueError, 'existing lock'):
            with training.readonly_source_lock(self.source_root):
                pass
        self.assertFalse((self.source_root/'active.lock').exists())


if __name__ == '__main__':
    unittest.main()
