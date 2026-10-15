"""Paired warmstarts, projected updates, exact stage resume and replay gates."""
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

from experiments.chronological import train_min_event_horizon as training
from experiments.chronological import evaluate_min_event_horizon as evaluation
from experiments.chronological import train_min_early_event as early
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


class HorizonRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        cls.logger = logging.getLogger('horizon_runner_tests')
        cls.logger.addHandler(logging.NullHandler())

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=training.REPO)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.raw, self.source_root = self.root/'raw', self.root/'source'
        self.raw.mkdir()
        self.source_root.mkdir()
        (self.source_root/'EVENT').mkdir()
        (self.source_root/'active.lock').write_bytes(b'0')
        _, fixture_protocol = event_fixture(self.raw)
        self.protocol = dict(legacy.read_json(training.PROTOCOL), **{key: fixture_protocol[key]
            for key in ('total_samples', 'train_samples', 'val_samples', 'test_samples', 'stations', 'batch_size')})
        self.g = graph(4)
        self.builder = patch.object(training.paper, 'build_model', side_effect=tiny_minimal)
        self.builder.start()
        self.addCleanup(self.builder.stop)
        setup = args_for(self.root/'unused', self.raw, self.source_root)
        setup.protocol = early.PROTOCOL
        core, _ = early.build_arm(setup, 'EVENT', 4, self.g, 'cpu')
        with torch.no_grad():
            core.early_event_condition.condition[-1].weight.fill_(.025)
            core.early_event_condition.condition[-1].bias.fill_(.015)
        core.eval()
        self.data = lambda: EventPublishedData(self.raw, self.protocol, self.logger, check=True)
        arrays, native = training.validation_predictions(core, self.data(), torch.device('cpu'))
        source_identity = dict(schema=early.SCHEMA, arm_modes={'EVENT': 'event'}, engineering_check=True,
            seed=2025, runtime=dict(torch=str(torch.__version__), cuda=torch.version.cuda, device_type='cpu'),
            source_sha256=early.source_fingerprints(), protocol_sha256=sha256(early.PROTOCOL),
            protocol=dict(legacy.read_json(early.PROTOCOL), train_samples=14, val_samples=3, stations=4, batch_size=2),
            graph_inputs_sha256={name: sha256(setup.graph_dir/name) for name in ('graph.json', 'stations.csv', 'source_summary.json')},
            package={name: sha256(self.raw/name) for name in self.protocol['data_sha256']}, initialization={'fixture': 'source'})
        source_state = copy.deepcopy(core.state_dict())
        last_state = copy.deepcopy(source_state)
        last_state['embedding.weight'] += .1
        history = [dict(epoch=1, validation_mae=native['actual'][0], validation_mape=native['actual'][1],
            validation_rmse=native['actual'][2], best_epoch=1, best_metric=native['actual'][0], global_updates=2),
            dict(epoch=2, validation_mae=native['actual'][0]+1., validation_mape=native['actual'][1],
            validation_rmse=native['actual'][2], best_epoch=1, best_metric=native['actual'][0], global_updates=4)]
        saved = dict(identity=source_identity, arm='EVENT', completed_epoch=2, history=history,
            model_state=last_state, best_model_state=source_state, best_metric=native['actual'][0], global_updates=4,
            optimizer_state={'must_not_restore': True}, scheduler_state={'must_not_restore': True})
        torch.save(saved, self.source_root/'EVENT/last.pt')
        legacy.write_json(self.source_root/'identity.json', source_identity)
        legacy.write_json(self.source_root/'initialization.json', source_identity['initialization'])
        legacy.write_json(self.source_root/'data_audit.json', self.data().audit)
        legacy.write_json(self.source_root/'EVENT/history.json', history)
        legacy.write_json(self.source_root/'EVENT/summary.json', dict(best_epoch=1, best_metric=saved['best_metric'], completed_epoch=2))

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
        self.assertEqual(args.arms, ['CONTROL', 'HORIZON'])
        self.assertEqual(args.epochs, 5)
        self.assertEqual(args.source_run, training.DEFAULT_SOURCE)
        protocol = training.load_protocol(args)
        self.assertEqual(protocol['learning_rate'], 1e-4)
        self.assertEqual(protocol['horizon_learning_rate'], 1e-3)
        self.assertEqual(protocol['warm_epoch'], 0)
        self.assertEqual(protocol['batch_size'], 64)
        for command in (['--source-run', str(self.source_root), '--run-dir', str(self.source_root)],
                        ['--arms', 'CONTROL', '--only', 'HORIZON'], ['--check', '--epochs', '2'], ['--batch-size', '0']):
            with self.subTest(command=command), self.assertRaises(ValueError):
                training.parse_args(command)
        with self.assertRaisesRegex(ValueError, 'Internal worker'):
            training.parse_args(['--worker', 'HORIZON', '--run-dir', str(self.root/'new')])
        self.assertNotIn('--evaluate-after', training.child_command(args_for(self.root/'new', self.raw, self.source_root), 'HORIZON'))

    def test_active_source_ON_preflight_uses_best_weights_and_preserves_original_bytes(self):
        before = training.source_snapshot(args_for(self.root/'new', self.raw, self.source_root))
        args, source, identity = self.setup_run('preflight')
        self.assertEqual(source['selected']['epoch'], 1)
        self.assertEqual(source['saved']['completed_epoch'], 2)
        models = [training.build_arm(args, arm, 4, self.g, 'cpu', source)[0].eval() for arm in training.ARMS]
        self.equal_nested(models[0].state_dict(), models[1].state_dict())
        self.assertFalse(models[0].horizon_offset.requires_grad)
        self.assertTrue(models[1].horizon_offset.requires_grad)
        batch = next(self.data().loaders['train_loader'].get_iterator())
        x, events = torch.from_numpy(batch['x_data']), training.event_tensors(batch['event_data'], 'cpu')
        with torch.inference_mode():
            on, off = models[0](x, incident_data=events), models[0](x, incident_data=events, report_enabled=False)
        self.assertFalse(torch.equal(on, off), 'Source must contain a nonzero trained incident condition')
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

    def test_optimizer_projection_has_no_offset_decay_and_curriculum_is_always_twelve(self):
        args, source, identity = self.setup_run('projection')
        model, _ = training.build_arm(args, 'HORIZON', 4, self.g, 'cpu', source)
        engine = training.make_engine(args, 'HORIZON', self.protocol, self.data(), model, self.logger)
        self.assertEqual(engine._optimizer.state, {})
        self.assertEqual(engine._optimizer.param_groups[1]['weight_decay'], 0.)
        self.assertEqual(engine._optimizer.param_groups[1]['lr'], 1e-3)
        engine._optimizer.param_groups[1]['lr'] = 100.
        model.horizon_offset.grad = torch.ones(12)
        engine._optimizer.step()
        self.assertTrue(torch.equal(model.horizon_offset, torch.full((12,), -1.)))
        for _ in range(100):
            engine._update_curriculum_length()
            self.assertEqual(engine._cl_len, 12)
        self.assertTrue(all(value.requires_grad for value in model.core.parameters()))

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
        self.assertEqual(orders['CONTROL'], orders['HORIZON'])
        self.assertEqual(before, training.source_snapshot(args_for(self.root/'unused', self.raw, self.source_root)))

    def test_epoch_zero_remains_eligible_and_best_and_last_gates_are_distinct(self):
        args, source, identity = self.setup_run('epoch_zero')
        with patch.object(training.HorizonEngine, 'evaluate', return_value=(1e6, .1, 2e6)), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            summary = training.train_arm(args, 'HORIZON', self.protocol, self.data(), identity, self.g, self.logger, source)
        self.assertEqual(summary['best_epoch'], 0)
        self.assertTrue(summary['best_is_source_epoch_zero'])
        self.assertEqual(summary['best_horizon_weights'], [1.]*12)
        saved = torch.load(args.run_dir/'HORIZON/last.pt', map_location='cpu', weights_only=False)
        self.assertEqual(evaluation.selected_metrics(saved, legacy.read_json(args.run_dir/'source_reference.json')),
                         legacy.read_json(args.run_dir/'source_reference.json')['actual'])
        training.validate_saved(saved, identity, 'HORIZON')
        for bad in (torch.ones(12), torch.full((12,), -2.), torch.zeros(11), torch.full((12,), float('nan'))):
            corrupt = copy.deepcopy(saved)
            corrupt['model_state']['horizon_offset'] = bad
            with self.assertRaisesRegex(ValueError, 'horizon offsets'):
                training.validate_saved(corrupt, identity, 'HORIZON')

    def test_new_identity_rejects_batch_source_or_runtime_drift_and_accepts_epoch_only_resume(self):
        args, source, identity = self.setup_run('identity')
        legacy.write_json(args.run_dir/'identity.json', identity)
        legacy.write_json(args.run_dir/'initialization.json', identity['initialization'])
        args.epochs, args.only = 9, ['HORIZON']
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
                           '--device', 'cpu', '--threads', '1', '--check', '--only', 'HORIZON'])
        self.assertEqual(worker.call_count, 1)
        self.assertEqual(worker.call_args.args[1], 'HORIZON')

    def test_comparison_uses_actual_stage_zero_reference_and_separates_complete_from_equal_budget(self):
        args, source, identity = self.setup_run('comparison')
        baseline = legacy.read_json(args.run_dir/'source_reference.json')['actual'][0]
        identity['source']['best_metrics'][0] = baseline+1e-6
        for arm in training.ARMS:
            (args.run_dir/arm).mkdir()
            (args.run_dir/arm/'last.pt').touch()
        histories = {'CONTROL': [dict(validation_mae=baseline+2e-6, train_order_sha256='a'*64)],
                     'HORIZON': [dict(validation_mae=baseline-5e-7, train_order_sha256='a'*64),
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
        self.assertAlmostEqual(result['comparisons']['HORIZON_vs_CONTROL']['common_prefix_best_mae_gain'], 5e-7, places=10)

    def test_best_replay_gates_all_evaluation_interventions_and_restores_inputs(self):
        args, source, identity = self.setup_run('evaluation')
        baseline = legacy.read_json(args.run_dir/'source_reference.json')
        with np.load(args.run_dir/'source_reference.npz', allow_pickle=False) as archive:
            baseline_arrays = {name: archive[name] for name in archive.files}
        saved, models = {}, {}
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            for arm in training.ARMS:
                training.train_arm(args, arm, self.protocol, self.data(), identity, self.g, self.logger, source)
                saved[arm] = torch.load(args.run_dir/arm/'last.pt', map_location='cpu', weights_only=False)
                models[arm], _ = training.build_arm(args, arm, 4, self.g, 'cpu', source)
                models[arm].load_state_dict(saved[arm]['best_model_state'])
                models[arm].eval()
        paths = {'source/'+name: args.source_run/name for name in training.SOURCE_FILES}
        result = evaluation.run_probe(models, self.data(), saved, self.g, {'west': [0, 1]},
            args.run_dir/'evaluation_good', baseline, baseline_arrays, paths)
        self.assertEqual(result['status'], 'MIN_EVENT_HORIZON_EVALUATION_PASS')
        self.assertEqual(set(result['cases']), set(evaluation.CASES))
        self.assertEqual(result['optimizer_updates'], 0)
        self.assertTrue(result['on_exactly_restored'])
        self.assertTrue(result['engineering_only'])
        report = (args.run_dir/'evaluation_good'/'report.md').read_text(encoding='utf-8')
        self.assertIn('CONTROL_ON', report)
        self.assertIn('工程检查：True；验证窗口：3', report)
        self.assertIn('不构成正式预测效果', report)
        self.assertAlmostEqual(result['all_nodes']['SOURCE_EVENT_ON']['mae_pooled'], baseline['actual'][0], places=5)
        mechanisms = legacy.read_json(args.run_dir/'evaluation_good'/'mechanisms.json')['cases']
        self.assertEqual(mechanisms['CONTROL_ON']['response_measured_calls'], 0)
        self.assertNotIn('raw_delta_abs_sum', mechanisms['CONTROL_ON'])
        bad_saved = copy.deepcopy(saved)
        if bad_saved['HORIZON']['best_epoch'] == 0:
            bad_baseline = dict(baseline, actual=[baseline['actual'][0]+1., *baseline['actual'][1:]])
        else:
            bad_baseline = baseline
            bad_saved['HORIZON']['history'][bad_saved['HORIZON']['best_epoch']-1]['validation_mae'] += 1.
            bad_saved['HORIZON']['best_metric'] += 1.
        seen, original = [], evaluation.forward_case
        def record(*values, **kwargs):
            seen.append(values[3])
            return original(*values, **kwargs)
        with patch.object(evaluation, 'forward_case', side_effect=record):
            with self.assertRaisesRegex(ValueError, 'replay failed'):
                evaluation.run_probe(models, self.data(), bad_saved, self.g, {}, args.run_dir/'evaluation_bad',
                                     bad_baseline, baseline_arrays, paths)
        self.assertTrue(all(case.endswith('_ON') for case in seen))
        self.assertFalse((args.run_dir/'evaluation_bad'/'summary.json').exists())


if __name__ == '__main__':
    unittest.main()
