"""Support math, streaming statistics, original masks and read-only contracts."""
from contextlib import contextmanager, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.chronological.original_edge_delay import EdgeDelay, install_delay, delay_mode
from experiments.chronological.original_edge_delay_diagnostics import (
    LagTelemetry, existing_run_read_lock, independent_output, parameter_changes, summarize_cases)
from experiments.chronological import evaluate_original_edge_delay as diagnostic
from experiments.chronological.original_igstgnn_checkpoint_compat import counter_weights_only_compat
from experiments.chronological.original_igstgnn_runtime import tensor_tree_hash, run_lock
from experiments.chronological.original_igstgnn_source import get_original_modules
from tests import test_original_edge_delay as edge_tests


class EdgeDelayDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.modules = get_original_modules()

    def temp(self):
        value = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(value.cleanup)
        return Path(value.name)

    def model(self):
        fixture = edge_tests.EdgeDelayTests('test_invalid_graphs_rejected')
        fixture.modules = self.modules
        self.addCleanup(fixture.doCleanups)
        return fixture.model()

    def test_boundary_mass_matches_explicit_paths_and_clock(self):
        graph = torch.tensor([[.25, .5, 0.], [.2, 0., .8], [.7, .1, 0.]], dtype=torch.float64)
        estimator = EdgeDelay(graph, 4, 2)
        delay = torch.linspace(.1, 1.7, len(estimator.source), dtype=torch.float64)[None]
        telemetry = LagTelemetry(SimpleNamespace(layers=[]))
        telemetry._support(2, 0, 3, 5, 12, estimator, delay)
        from experiments.chronological.original_edge_delay import lag_kernels
        dense = torch.zeros(1, 3, 3, dtype=torch.float64)
        dense[:, estimator.receiver, estimator.source] = delay
        kernels = lag_kernels(graph, dense, 2).numpy()[0]
        for row in telemetry.finish()['boundary_support']:
            t, order = row['summary_index'], row['order']
            expected = 0.
            for i in range(3):
                for j in range(3):
                    if order == 1:
                        if i != j:
                            expected += sum(kernels[a, i, j] for a in range(3) if a <= t)
                    else:
                        for k in range(3):
                            if i != k:
                                expected += sum(kernels[a, i, j]*kernels[b, j, k]
                                    for a in range(3) for b in range(3) if a+b <= t)
            self.assertAlmostEqual(row['available_mass_sum'], expected, places=12)
            self.assertEqual(row['window_end_input_index'], 9+t)

    def test_zero_path_mass_is_unavailable_not_zero_loss(self):
        estimator = EdgeDelay(torch.zeros(3, 3), 4)
        telemetry = LagTelemetry(SimpleNamespace(layers=[]))
        telemetry._support(0, 0, 3, 4, 12, estimator, torch.zeros(2, 0))
        for row in telemetry.finish()['boundary_support']:
            self.assertIsNone(row['missing_weight_fraction'])
            self.assertIsNone(row['available_fraction_receiver_mean'])
            self.assertEqual(row['zero_reference_positions'], 6)

    def test_streamed_lag_stats_quantile_intervals_and_shuffle_changes(self):
        graph = torch.ones(3, 3)-torch.eye(3)
        estimator = EdgeDelay(graph, 4)
        estimator.mode = 'SHUFFLE'
        values = np.linspace(.1, 1.8, 18).reshape(3, 6).astype(np.float32)
        telemetry = LagTelemetry(SimpleNamespace(layers=[]))
        for chunk in (values[:2], values[2:]):
            telemetry.pending.append((0, 0, 3, 5, estimator, torch.as_tensor(chunk)))
            telemetry.consume(12)
        row = telemetry.finish()['lags'][0]
        self.assertAlmostEqual(row['mean'], float(values.astype(np.float64).mean()))
        self.assertAlmostEqual(row['sample_std_per_edge_mean'], float(values.astype(np.float64).std(0).mean()))
        self.assertEqual(row['shuffle_changed_positions'], 18)
        for q, bounds in row['quantile_intervals'].items():
            value = np.sort(values.ravel())[int(np.ceil(float(q)*18))-1]
            self.assertLessEqual(bounds[0], value)
            self.assertGreaterEqual(bounds[1], value)

    def test_constant_lag_has_zero_cross_sample_variance(self):
        estimator = EdgeDelay(torch.ones(3,3)-torch.eye(3),4)
        estimator.mode = 'FIXED'
        telemetry = LagTelemetry(SimpleNamespace(layers=[]))
        for batch in (2,1):
            telemetry.pending.append((0,0,3,5,estimator,torch.full((batch,6),.23840584)))
            telemetry.consume(12)
        row = telemetry.finish()['lags'][0]
        self.assertEqual(row['sample_std_per_edge_mean'],0.)
        self.assertEqual(row['edge_std_per_sample_mean'],0.)

    def test_native_forward_is_unchanged_and_hooks_are_removed(self):
        model = install_delay(self.model(), 'DYNAMIC').eval()
        x = torch.rand(2, 12, 4, 3)*.2
        with torch.no_grad():
            expected = model(x)
        telemetry = LagTelemetry(model)
        before = tensor_tree_hash(model.state_dict())
        with torch.no_grad():
            actual = model(x)
            telemetry.consume(12)
        telemetry.close()
        self.assertTrue(torch.equal(expected, actual))
        self.assertEqual(before, tensor_tree_hash(model.state_dict()))
        self.assertTrue(all(not l.dif_layer._forward_hooks for l in model.layers))
        self.assertEqual(sorted({r['summary_length'] for r in telemetry.finish()['boundary_support']}), [2, 4, 6, 8, 10])

    def test_no_align_does_not_report_stale_lags(self):
        model = install_delay(self.model(), 'DYNAMIC').eval()
        x = torch.rand(2, 12, 4, 3)*.2
        with torch.no_grad():
            model(x)
        telemetry = LagTelemetry(model)
        with delay_mode(model, 'NO_ALIGN'), torch.no_grad():
            model(x)
            telemetry.consume(12)
        telemetry.close()
        self.assertEqual(telemetry.finish()['lags'], [])

    def test_original_mask_fixed_hard_gain_sign_and_empty_group(self):
        target = np.full((3, 12, 2), 10., np.float32)
        target[0, 0, 0] = 0
        cases = {'SOURCE': target+2, 'DYNAMIC_LAST': target+1, 'DYNAMIC_LAST_NO_ALIGN': target+3}
        result = summarize_cases(cases, target, [10, 11, 12], [5, 7], {'r': [0]}, [12],
            [('SOURCE', 'DYNAMIC_LAST'), ('DYNAMIC_LAST_NO_ALIGN', 'DYNAMIC_LAST')])
        comparison = result['comparisons']['DYNAMIC_LAST_vs_SOURCE']
        self.assertEqual(result['valid_points'], 71)
        self.assertEqual(result['fixed_hard_ids'], [12])
        self.assertEqual(comparison['groups']['all_nodes']['mae_gain_pooled'], 1.)
        self.assertEqual(comparison['groups']['all_nodes']['prediction_signed_mean'], -1.)
        self.assertIsNone(comparison['groups']['flow:[200,inf)']['mae_gain_pooled'])
        self.assertEqual(comparison['groups']['bias:over']['count'], 71)

    def test_unpaired_predictions_and_hard_ids_are_rejected(self):
        y = np.ones((2, 12, 2), np.float32)*10
        for cases, hard in (({'SOURCE': y[:, :2]}, [1]), ({'SOURCE': y}, [9])):
            with self.assertRaises(ValueError):
                summarize_cases(cases, y, [1, 2], [7, 8], {}, hard, [])

    def test_parameter_changes_map_wrapped_backbone_and_buffers(self):
        source = {'layers.0.dif_layer.weight': torch.tensor([1., 2.]), 'counter': torch.tensor(1)}
        state = {'layers.0.dif_layer.original.weight': torch.tensor([4., 6.]), 'counter': torch.tensor(1),
            'layers.0.dif_layer.delays.0.edge_logits': torch.tensor([20.])}
        result = parameter_changes(state, source)
        self.assertEqual(result['backbone_change_l2'], 5.)
        self.assertEqual(result['backbone_change_abs_max'], 4.)
        extra = {'layers.0.dif_layer.delays.0.edge_logits': torch.tensor([18.])}
        result = parameter_changes(state, source, extra)
        self.assertEqual(result['new_parameter_change_l2'], 2.)
        self.assertEqual(result['new_parameter_positions'], 1)
        state['counter'] = torch.tensor(3)
        result = parameter_changes(state, source, extra, {'layers.0.dif_layer.weight'})
        self.assertEqual(result['backbone_parameter_change_l2'], 5.)
        self.assertEqual(result['backbone_buffer_change_l2'], 2.)

    def test_output_overlap_and_existing_output_rejected(self):
        root = self.temp()
        for output in (root, root/'child', root.parent):
            with self.assertRaises(ValueError):
                independent_output(output, [root])

    def test_read_lock_does_not_write_and_conflicts_with_writer(self):
        root = self.temp()
        (root/'active.lock').write_bytes(b'0')
        before = diagnostic.snapshot(root)
        with run_lock(root):
            with self.assertRaises(ValueError):
                with existing_run_read_lock(root):
                    pass
        self.assertEqual(before, diagnostic.snapshot(root))
        with existing_run_read_lock(root):
            self.assertEqual(before, diagnostic.snapshot(root))
        self.assertEqual(before, diagnostic.snapshot(root))

    def test_read_lock_never_creates_missing_lock(self):
        root = self.temp()
        with self.assertRaises(FileNotFoundError):
            with existing_run_read_lock(root):
                pass
        self.assertEqual(list(root.iterdir()), [])

    def test_metric_replay_checks_all_three_metrics(self):
        expected = dict(epoch=1, validation_mae=10., validation_rmse=12., validation_mape_fraction=.2)
        actual = dict(mae=10., rmse=12., mape_fraction=.2)
        self.assertIn('PASS', diagnostic.metric_replay(actual, expected)['status'])
        for key in actual:
            with self.assertRaises(ValueError):
                diagnostic.metric_replay(dict(actual, **{key: actual[key]+1}), expected)

    def test_best_and_last_are_separate_comparison_targets(self):
        cases = dict(SOURCE=None)
        for arm in diagnostic.pilot.ARMS:
            for stage in ('BEST', 'LAST'):
                cases[arm+'_'+stage] = None
                if arm != 'CONTROL':
                    for mode in (('NO_ALIGN',) if arm == 'FIXED' else ('NO_ALIGN', 'MASS_ONLY', 'SHUFFLE')):
                        cases[arm+'_'+stage+'_'+mode] = None
        pairs = diagnostic.make_pairs(cases)
        self.assertIn(('DYNAMIC_LAST_NO_ALIGN', 'DYNAMIC_LAST'), pairs)
        self.assertIn(('DYNAMIC_BEST', 'DYNAMIC_LAST'), pairs)
        self.assertNotIn(('DYNAMIC_BEST_NO_ALIGN', 'DYNAMIC_LAST'), pairs)

    def test_evaluation_detects_rng_mutation(self):
        class RandomModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList()
                self.weight = torch.nn.Parameter(torch.ones(()))
            def forward(self, x, **kwargs):
                return x*self.weight+torch.rand_like(x)
        x = torch.ones(1, 12, 2, 1)
        data = SimpleNamespace(loaders={'val_loader':SimpleNamespace(num_batch=1,get_iterator=lambda: iter([0]))},
            scaler=SimpleNamespace(inverse_transform=lambda x:x))
        with patch.object(diagnostic, 'device_batch', return_value=(x,x*10,None,None)), redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, 'RNG'):
                diagnostic.evaluate_case(RandomModel(), data, self.modules, 'cpu', 'fixture', {})

    def test_unrelated_pilot_is_rejected_without_creating_output(self):
        root = self.temp()
        (root/'identity.json').write_text(json.dumps({'schema':'unrelated'}))
        output = root.parent/(root.name+'_output')
        with self.assertRaisesRegex(ValueError, 'existing original'):
            diagnostic.main(['--pilot-run',str(root),'--output-dir',str(output)])
        self.assertFalse(output.exists())

    def paired_fixture(self):
        root = self.temp()
        source, pilot_run = root/'source', root/'pilot'
        source.mkdir(); pilot_run.mkdir()
        for folder in (source, pilot_run):
            (folder/'active.lock').write_bytes(b'0')
        from experiments.chronological.original_igstgnn_runtime import configure_runtime
        runtime = configure_runtime('cpu', 1, 2025)
        provenance = dict(path=str(source), identity_sha256='fixture', checkpoint_sha256='fixture',
            source_fingerprint=self.modules.source_fingerprint, source_files={})
        val = dict(selected_indices=[10, 11], count=2, partition=[2], batch_size=2)
        identity = dict(schema=diagnostic.pilot.SCHEMA, source=provenance, engineering_only=True,
            training_config=dict(seed=2025, device='cpu', warm_epoch=30, cl_epoch=3), runtime=runtime,
            architecture=dict(max_delay=2, score_dim=8),
            data=dict(splits={'train':dict(count=2, partition=[2]), 'val':val}),
            execution_source={p:diagnostic.sha256(diagnostic.REPO/p) for p in diagnostic.pilot.EXECUTION_FILES})
        identity['identity_sha256'] = diagnostic.pilot.digest(identity)
        diagnostic.write_json(pilot_run/'identity.json', identity)
        y = np.full((2, 12, 4), 10., np.float32)
        p = np.ones_like(y)
        diagnostic.pilot.save_predictions(pilot_run/'source_predictions.npz', p, y, [10,11], np.arange(4))
        diagnostic.write_json(pilot_run/'analysis.json', dict(fixed_hard_ids=[10]))
        states, completed = {}, {}
        for arm in diagnostic.pilot.ARMS:
            folder = pilot_run/arm; folder.mkdir()
            torch.save({'weight':torch.ones(())}, folder/'best.pt')
            diagnostic.pilot.save_predictions(folder/'best.npz', p, y, [10,11], np.arange(4))
            history = [dict(epoch=e, global_updates=e, validation_mae=9.+2*e,
                validation_rmse=9.+2*e, validation_mape_fraction=.9+.2*e) for e in (0,1)]
            state = dict(arm=arm, completed_epoch=1, best_epoch=0, best_validation_mae=9., history=history,
                global_updates=1, identity_sha256=identity['identity_sha256'], train_order=torch.arange(2),
                scheduler_state={'last_epoch':1}, curriculum_length=12, optimizer_state={}, rng_state={},
                model_state={'weight':-torch.ones(())}, best_checkpoint_file='best.pt', best_prediction_file='best.npz',
                best_checkpoint_sha256=diagnostic.sha256(folder/'best.pt'), best_prediction_sha256=diagnostic.sha256(folder/'best.npz'))
            torch.save(state, folder/'last.pt')
            diagnostic.write_json(folder/'history.json', history)
            diagnostic.write_json(folder/'summary.json', {k:v for k,v in state.items() if k not in
                ('model_state','optimizer_state','scheduler_state','train_order','rng_state')})
            states[arm]=state
            completed[arm]=dict(epochs=1,best_epoch=0,updates=1)
        metrics = diagnostic.pilot.metric_summary(p,y,np.ones_like(y,dtype=bool))
        summary = dict(schema=diagnostic.pilot.SCHEMA, engineering_only=True, same_completed_budget=True,
            status='EDGE_DELAY_ENGINEERING_PASS', completed=completed, metrics={a:metrics for a in diagnostic.pilot.ARMS})
        diagnostic.write_json(pilot_run/'summary.json', summary)
        return root, pilot_run, identity, provenance

    def test_pilot_validates_best_zero_last_one_and_rejects_wrong_selection(self):
        _, pilot_run, _, _ = self.paired_fixture()
        _, _, states = diagnostic.load_pilot(pilot_run, True)
        self.assertEqual(states['DYNAMIC']['best_epoch'],0)
        self.assertEqual(float(states['DYNAMIC']['model_state']['weight']),-1.)
        state = states['DYNAMIC']
        state['history'][1]['validation_mae']=8.
        torch.save(state, pilot_run/'DYNAMIC/last.pt')
        with self.assertRaisesRegex(ValueError,'best selection'):
            diagnostic.load_pilot(pilot_run, True)

    def test_formal_subset_and_partial_pilot_are_blocked(self):
        _, pilot_run, _, _ = self.paired_fixture()
        with self.assertRaisesRegex(ValueError, 'subset conversion'):
            diagnostic.load_pilot(pilot_run, False)
        summary=diagnostic.read_json(pilot_run/'summary.json')
        summary['status']='EDGE_DELAY_PILOT_RUNNING'
        diagnostic.write_json(pilot_run/'summary.json',summary)
        with self.assertRaisesRegex(ValueError, 'running/partial'):
            diagnostic.load_pilot(pilot_run,True)

    def test_best_zero_last_one_end_to_end_is_read_only(self):
        root, pilot_run, identity, provenance = self.paired_fixture()
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.layers=torch.nn.ModuleList(); self.weight=torch.nn.Parameter(torch.ones(())); self.mode=None
            def forward(self,x,**kwargs):
                return x*self.weight
        @contextmanager
        def mode(model,value):
            old=model.mode; model.mode=value
            try: yield
            finally: model.mode=old
        x=torch.ones(2,12,4,1)
        data=SimpleNamespace(identity=dict(identity['data'],splits={'val':identity['data']['splits']['val']}),
            station_ids=np.arange(4),roads={'r':[0,1]},
            loaders={'val_loader':SimpleNamespace(num_batch=1,get_iterator=lambda:iter([0]))},
            scaler=SimpleNamespace(inverse_transform=lambda x:x))
        before=diagnostic.snapshot(pilot_run)
        with patch.object(diagnostic,'source_from_run',return_value=({'weight':torch.ones(())},{},9.,provenance)), \
                patch.object(diagnostic,'load_data',return_value=data), \
                patch.object(diagnostic,'build_model',side_effect=lambda *a:Model()), \
                patch.object(diagnostic,'install_delay',side_effect=lambda model,*a,**k:model), \
                patch.object(diagnostic,'delay_mode',side_effect=mode), \
                patch.object(torch.optim,'Adam',side_effect=AssertionError('No optimizer allowed')), \
                patch.object(diagnostic,'device_batch',return_value=(x,x*10,None,None)), redirect_stdout(io.StringIO()):
            summary=diagnostic.main(['--pilot-run',str(pilot_run),'--output-dir',str(root/'output'),'--check'])
        self.assertEqual(summary['optimizer_updates'],0)
        self.assertEqual(summary['metrics']['DYNAMIC_BEST']['mae_pooled'],9.)
        self.assertEqual(summary['metrics']['DYNAMIC_LAST']['mae_pooled'],11.)
        self.assertEqual(summary['new_inference_calls']['DYNAMIC_LAST'],1)
        self.assertEqual(before,diagnostic.snapshot(pilot_run))
        replay=diagnostic.read_json(root/'output/replay.json')
        self.assertTrue(replay['DYNAMIC_LAST']['on_exactly_restored'])
        self.assertEqual(replay['DYNAMIC_LAST']['expected']['epoch'],1)
        with np.load(root/'output/paired_predictions.npz') as saved:
            self.assertFalse(np.array_equal(saved['DYNAMIC_BEST'],saved['DYNAMIC_LAST']))


if __name__ == '__main__':
    unittest.main()
