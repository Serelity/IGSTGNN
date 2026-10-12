"""Factorial interventions must preserve weights, inputs and both other paths."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.chronological import probe_b3_r_modules as probe
from src.models.incident_spatial_residual import tensor_state_hash
from src.models.incident_spatial_strength import from_residual
from tests.test_b3_r import controlled_forecast
from tests.test_b3_r_path_probe import ProbeDataset, hook_counts, saved_windows


class Dataset(ProbeDataset):
    station_ids = np.arange(4)


class FactorialProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_each_switch_matches_direct_forward_and_keeps_spatial_context(self):
        with controlled_forecast() as (model, x, report):
            with torch.no_grad():
                model.report_spatial_state.D[-1].bias.fill_(.8)
            before, hooks = tensor_state_hash(model.state_dict()), hook_counts(model)
            batch = {'x': x, 'incident': report}
            input_hash = probe.previous.tree_hash(batch)
            contexts, predictions = {}, {}
            for case, (tiid, d) in probe.CASES.items():
                with torch.inference_mode():
                    independent = model(x, incident_data=report, spatial_report_enabled=d,
                                        tiid_report_enabled=tiid, return_spatial_details=True)
                prediction, contexts[case] = probe.forward(model, batch, {'mean': 0., 'std': 1.}, case)
                predictions[case] = prediction
                np.testing.assert_array_equal(prediction, independent['prediction'].numpy())
            _, restored = probe.forward(model, batch, {'mean': 0., 'std': 1.}, probe.ON)
            probe.check_contexts(contexts, restored)
            self.assertGreater(np.abs(predictions[probe.ON]-predictions['TIID_OFF_D_ON']).max(), 0.)
            self.assertGreater(np.abs(predictions[probe.ON]-predictions['TIID_ON_D_OFF']).max(), 0.)
            self.assertEqual(before, tensor_state_hash(model.state_dict()))
            self.assertEqual(hooks, hook_counts(model))
            self.assertEqual(input_hash, probe.previous.tree_hash(batch))
            self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_half_and_gate_models_are_supported_and_initial_d_zero_is_exact(self):
        with controlled_forecast() as (parent, x, report):
            for mode in ('fixed_half', 'identity_gate'):
                model, _ = from_residual(parent, mode)
                model.eval()
                probe.validate_model(model)
                batch = {'x': x, 'incident': report}
                for tiid in (True, False):
                    first = 'TIID_'+('ON' if tiid else 'OFF')+'_D_'
                    on, _ = probe.forward(model, batch, {'mean': 0., 'std': 1.}, first+'ON')
                    off, _ = probe.forward(model, batch, {'mean': 0., 'std': 1.}, first+'OFF')
                    np.testing.assert_array_equal(on, off)

    def test_forward_exception_removes_only_probe_hooks(self):
        with controlled_forecast() as (model, x, report):
            hooks, before = hook_counts(model), tensor_state_hash(model.state_dict())
            with patch.object(model.tiid_module, 'forward', side_effect=RuntimeError('fixture TIID failure')):
                with self.assertRaisesRegex(RuntimeError, 'fixture TIID failure'):
                    probe.forward(model, {'x': x, 'incident': report}, {'mean': 0., 'std': 1.}, probe.ON)
            self.assertEqual(hooks, hook_counts(model))
            self.assertEqual(before, tensor_state_hash(model.state_dict()))
            self.assertTrue(model.report_spatial_state.report_path_enabled)

    def test_rejects_training_gradients_disabled_training_path_and_missing_exchange(self):
        with controlled_forecast() as (model, x, report):
            model.train()
            with self.assertRaisesRegex(ValueError, 'eval mode'):
                probe.validate_model(model)
            model.eval()
            parameter = next(model.parameters())
            parameter.grad = torch.zeros_like(parameter)
            with self.assertRaisesRegex(ValueError, 'gradients'):
                probe.validate_model(model)
            parameter.grad = None
            model.report_spatial_state.report_path_enabled = False
            with self.assertRaisesRegex(ValueError, 'NO_D'):
                probe.validate_model(model)
            model.report_spatial_state.report_path_enabled = True
            del model.layers[4].dif_layer.localized_st_conv.incident_exchange
            with self.assertRaisesRegex(ValueError, 'coverage'):
                probe.validate_model(model)

    def test_failed_saved_replay_blocks_every_off_call(self):
        with controlled_forecast() as (model, x, report), tempfile.TemporaryDirectory() as directory:
            dataset = Dataset(x, report)
            indices = [row['sample_index'] for row in dataset.rows]
            saved, mae = saved_windows(model, dataset, indices, 3)
            saved['absolute'] += 100.
            calls, original = [], probe.forward
            def observe(*args):
                calls.append(args[3])
                return original(*args)
            with patch.object(probe, 'forward', side_effect=observe):
                with self.assertRaisesRegex(ValueError, 'OFF blocked'):
                    probe.run_probe(model, dataset, indices, 3, torch.device('cpu'), model.internal_spatial_graph,
                                    {'fixture': [0, 1]}, saved, mae, Path(directory))
            self.assertEqual(calls, [probe.ON, probe.ON])
            self.assertFalse((Path(directory)/'paired_predictions.npz').exists())

    def test_failed_final_on_blocks_analysis_and_prediction_artifacts(self):
        with controlled_forecast() as (model, x, report), tempfile.TemporaryDirectory() as directory:
            dataset = Dataset(x, report)
            indices = [row['sample_index'] for row in dataset.rows]
            saved, mae = saved_windows(model, dataset, indices, 3)
            original, hooks, full_on_calls = probe.forward, hook_counts(model), 0
            def corrupt(*args):
                nonlocal full_on_calls
                prediction, context = original(*args)
                if args[3] == probe.ON:
                    full_on_calls += 1
                    if full_on_calls == 4:  # two replay batches, paired ON, restored ON
                        prediction += 1.
                return prediction, context
            with patch.object(probe, 'forward', side_effect=corrupt):
                with self.assertRaisesRegex(ValueError, 'Final ON prediction restoration failed'):
                    probe.run_probe(model, dataset, indices, 3, torch.device('cpu'), model.internal_spatial_graph,
                                    {'fixture': [0, 1]}, saved, mae, Path(directory))
            self.assertEqual(hooks, hook_counts(model))
            for name in ('analysis.json', 'paired_predictions.npz', 'summary.json'):
                self.assertFalse((Path(directory)/name).exists())

    def test_complete_probe_preserves_partition_and_writes_all_four_predictions(self):
        with controlled_forecast() as (model, x, report), tempfile.TemporaryDirectory() as directory:
            with torch.no_grad():
                model.report_spatial_state.D[-1].bias.fill_(.8)
            dataset = Dataset(x, report)
            indices = [row['sample_index'] for row in dataset.rows]
            saved, mae = saved_windows(model, dataset, indices, 3)
            before, hooks, calls = tensor_state_hash(model.state_dict()), hook_counts(model), []
            original = probe.forward
            def observe(*args):
                calls.append((len(args[1]['x']), args[3]))
                return original(*args)
            output = Path(directory)
            with patch.object(probe, 'forward', side_effect=observe):
                result = probe.run_probe(model, dataset, indices, 3, torch.device('cpu'), model.internal_spatial_graph,
                                         {'fixture': [0, 1]}, saved, mae, output)
            expected = [(3, probe.ON), (1, probe.ON)]
            for size in (3, 1):
                expected.extend((size, case) for case in list(probe.CASES)+[probe.ON])
            self.assertEqual(calls, expected)
            self.assertEqual(result['batch_partition'], [3, 1])
            self.assertEqual(result['inference_calls'], 12)
            self.assertTrue(result['inference_only'] and result['final_on_exactly_restored'])
            self.assertEqual(result['training_updates'], 0)
            self.assertFalse(result['test_read'])
            self.assertEqual(before, tensor_state_hash(model.state_dict()))
            self.assertEqual(hooks, hook_counts(model))
            with np.load(output/'paired_predictions.npz', allow_pickle=False) as archive:
                for case in probe.CASES:
                    self.assertEqual(archive[case].shape, (4, 12, 4, 1))
                np.testing.assert_array_equal(archive['sample_indices'], indices)
                np.testing.assert_array_equal(archive['station_ids'], dataset.station_ids)
                self.assertTrue(archive['valid'].dtype == bool)
            self.assertAlmostEqual(result['analysis']['cases'][probe.ON]['metrics']['all_nodes']['mae_macro'], mae, places=12)

    def test_no_report_samples_have_identical_four_states_and_tiid_is_local(self):
        with controlled_forecast() as (model, x, report):
            with torch.no_grad():
                model.report_spatial_state.D[-1].bias.fill_(.8)
            report['distances'][1].zero_()
            batch = {'x': x, 'incident': report}
            predictions = {case: probe.forward(model, batch, {'mean': 0., 'std': 1.}, case)[0] for case in probe.CASES}
            associated = (report['distances'].abs().sum(-1) > 0).numpy()
            probe.check_prediction_support(predictions, associated)
            np.testing.assert_array_equal(predictions[probe.ON][1], predictions['TIID_OFF_D_OFF'][1])
            altered = {name: value.copy() for name, value in predictions.items()}
            altered['TIID_OFF_D_ON'][0, 0, 3, 0] += 1.
            with self.assertRaisesRegex(ValueError, 'outside direct report support'):
                probe.check_prediction_support(altered, associated)

    def test_best_selection_supports_gate_and_rejects_different_identity_or_history(self):
        identity = {'fixture': 'G1.1'}
        history = [dict(epoch=i+1, validation_mae_macro=value, best_epoch=1 if i == 0 else 2)
                   for i, value in enumerate((5., 2., 3.))]
        checkpoint = dict(identity=identity, arm='B3R_GATE', history=history, completed_epoch=3,
                          best_metric=2., best_model_state={'fixture': torch.ones(1)})
        summary = dict(arm='B3R_GATE', completed_epoch=3, best_epoch=2, best_metric=2.)
        self.assertEqual(probe.select_best(checkpoint, identity, 'B3R_GATE', history, summary), (2, 2.))
        with self.assertRaisesRegex(ValueError, 'identity/arm'):
            probe.select_best(checkpoint, identity, 'B3R', history, summary)
        with self.assertRaisesRegex(ValueError, 'selection'):
            probe.select_best(dict(checkpoint, best_model_state=None), identity, 'B3R_GATE', history, summary)


if __name__ == '__main__':
    unittest.main()
