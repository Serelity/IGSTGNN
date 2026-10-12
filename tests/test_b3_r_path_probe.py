"""Inference-only checks for the B3-R capacity-to-forecast path probe."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.chronological import probe_b3_r_path as probe
from src.models import incident_spatial as spatial
from src.models.incident_spatial_residual import tensor_state_hash
from tests.test_b3_r import controlled_forecast


def batch_for(x, report):
    return {'x': x, 'incident': report}


def exchange_modules(model):
    return [layer.dif_layer.localized_st_conv.incident_exchange for layer in model.layers]


def hook_counts(model):
    return [(len(module._forward_pre_hooks), len(module._forward_hooks))
            for module in model.modules()]


class ProbeDataset:
    def __init__(self, x, report):
        self.scaler = {'mean': 10., 'std': 2.}
        self.rows = [dict(sample_index=10+i, t0=f'2020-01-01T12:{i*5:02d}:00') for i in range(4)]
        self.samples = [dict(x=x[i % 2].clone(),
                             incident={name: value[i % 2].clone() for name, value in report.items()},
                             y_flow=torch.full((12, 4, 1), 9.+i),
                             y_valid=torch.ones((12, 4, 1), dtype=torch.bool)) for i in range(4)]

    def __getitem__(self, index):
        return self.samples[index]


def saved_windows(model, dataset, indices, batch_size):
    predictions, targets, valid_parts = [], [], []
    for batch in probe.batches(dataset, indices, batch_size):
        prediction, _ = probe.forward(model, batch, dataset.scaler, True)
        predictions.append(prediction)
        targets.append(batch['y_flow'].numpy())
        valid_parts.append(batch['y_valid'].numpy())
    prediction, target, valid = [np.concatenate(parts) for parts in (predictions, targets, valid_parts)]
    absolute = np.where(valid, np.abs(prediction.astype(np.float64)-target), 0.).sum((2, 3))
    counts = valid.sum((2, 3))
    times = {row['sample_index']: row['t0'] for row in dataset.rows}
    saved = dict(absolute=absolute, counts=counts, sample_indices=np.array(indices),
                 t0=np.array([times[index] for index in indices]))
    return saved, float((absolute.sum(0)/counts.sum(0)).mean())


class ExchangeTraceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_recording_preserves_prediction_state_inputs_and_exact_coverage(self):
        with controlled_forecast() as (model, x, report):
            before = tensor_state_hash(model.state_dict())
            original_x = x.clone()
            original_report = {name: value.clone() for name, value in report.items()}
            batch = batch_for(x, report)
            expected, _ = probe.forward(model, batch, {'mean': 10., 'std': 2.}, True)
            hooks = hook_counts(model)
            original_step = spatial.exchange_step
            with probe.ExchangeTrace(model) as trace:
                actual, _ = probe.forward(model, batch, {'mean': 10., 'std': 2.}, True)
            np.testing.assert_array_equal(actual, expected)
            self.assertEqual(before, tensor_state_hash(model.state_dict()))
            torch.testing.assert_close(x, original_x, atol=0, rtol=0)
            for name, value in original_report.items():
                torch.testing.assert_close(report[name], value, atol=0, rtol=0)
            self.assertIs(spatial.exchange_step, original_step)
            self.assertEqual(hook_counts(model), hooks)
            self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
            for field, expected_count in [('proposal', 40), ('flux', 40), ('readout', 20)]:
                keys = [key for key in trace.tensors if key[-1] == field]
                self.assertEqual(len(keys), expected_count, field)
                self.assertEqual({key[0] for key in keys}, set(range(5)))
                self.assertEqual({key[1] for key in keys}, set(range(4)))
                expected_steps = {-1} if field == 'readout' else {0, 1}
                self.assertEqual({key[2] for key in keys}, expected_steps)
            self.assertTrue(all(isinstance(value, np.ndarray) and np.isfinite(value).all()
                                for value in trace.tensors.values()))

    def test_zero_initialized_d_has_exact_on_off_and_restoration_identity(self):
        with controlled_forecast() as (model, x, report):
            batch = batch_for(x, report)
            with probe.ExchangeTrace(model) as on_trace:
                on, on_context = probe.forward(model, batch, {'mean': 0., 'std': 1.}, True)
            with probe.ExchangeTrace(model) as off_trace:
                off, off_context = probe.forward(model, batch, {'mean': 0., 'std': 1.}, False)
            np.testing.assert_array_equal(on, off)
            self.assertIsNone(off_context['report_logit_delta'])
            self.assertTrue(on_context['report_path_executed'])
            self.assertFalse(off_context['report_path_executed'])
            self.assertEqual(set(on_trace.tensors), set(off_trace.tensors))
            for key in on_trace.tensors:
                np.testing.assert_array_equal(on_trace.tensors[key], off_trace.tensors[key])
            with probe.ExchangeTrace(model, reference=on_trace.tensors, restoration=True):
                restored, _ = probe.forward(model, batch, {'mean': 0., 'std': 1.}, True)
            np.testing.assert_array_equal(on, restored)

    def test_nonzero_d_changes_actual_flux_readout_and_forecast_without_disabling_tiid(self):
        with controlled_forecast() as (model, x, report):
            with torch.no_grad():
                model.report_spatial_state.D[-1].bias.fill_(1.)
            batch = batch_for(x, report)
            tiid_calls = []
            handle = model.tiid_module.register_forward_hook(lambda *args: tiid_calls.append(True))
            try:
                with probe.ExchangeTrace(model) as on_trace:
                    on, on_context = probe.forward(model, batch, {'mean': 0., 'std': 1.}, True)
                with patch.object(model.report_spatial_state.D, 'forward',
                                  side_effect=AssertionError('D must not execute in the OFF pass')):
                    with probe.ExchangeTrace(model) as off_trace:
                        off, off_context = probe.forward(model, batch, {'mean': 0., 'std': 1.}, False)
            finally:
                handle.remove()
            self.assertEqual(len(tiid_calls), 2)
            self.assertGreater(float(np.max(np.abs(on-off))), 0.)
            for field in ('flux', 'readout'):
                response = sum(float(np.abs(value-off_trace.tensors[key]).sum())
                               for key, value in on_trace.tensors.items() if key[-1] == field)
                self.assertGreater(response, 0., field)
            for key in ('base_logit', 'base_node_capacity', 'base_edge_capacity', 'boundary'):
                torch.testing.assert_close(on_context[key], off_context[key], atol=0, rtol=0)

    def test_nonbottleneck_report_changes_node_capacity_without_edge_or_forecast_response(self):
        with controlled_forecast() as (model, x, report):
            with torch.no_grad():
                model.report_spatial_state.D[-1].bias.fill_(1.)
            report['distances'].zero_()
            report['distances'][:, 1, 1] = .8  # Destination has C=.88; source remains the C=.12 bottleneck.
            batch = batch_for(x, report)
            with probe.ExchangeTrace(model) as on_trace:
                on, on_context = probe.forward(model, batch, {'mean': 0., 'std': 1.}, True)
            with probe.ExchangeTrace(model) as off_trace:
                off, off_context = probe.forward(model, batch, {'mean': 0., 'std': 1.}, False)
            self.assertGreater(float((on_context['node_capacity']-off_context['node_capacity']).abs().max()), 0.)
            torch.testing.assert_close(on_context['capacity'], off_context['capacity'], atol=0, rtol=0)
            np.testing.assert_array_equal(on, off)
            for key in on_trace.tensors:
                np.testing.assert_array_equal(on_trace.tensors[key], off_trace.tensors[key])

    def test_slack_capacity_changes_do_not_imply_flux_gain_under_shared_sending_budget(self):
        with controlled_forecast() as (model, x, report):
            with torch.no_grad():
                model.report_spatial_state.D[-1].bias.fill_(1.)
            report['distances'][:, 1, 1] = .8
            def high_capacity(module, args, output):
                return torch.full_like(output, 2.)  # C~.88/.95; proposals~.88 exceed the x=.5 sending budget.
            handle = model.report_spatial_state.P.register_forward_hook(high_capacity)
            try:
                batch = batch_for(x, report)
                with probe.ExchangeTrace(model) as on_trace:
                    on, on_context = probe.forward(model, batch, {'mean': 0., 'std': 1.}, True)
                with probe.ExchangeTrace(model) as off_trace:
                    off, off_context = probe.forward(model, batch, {'mean': 0., 'std': 1.}, False)
            finally:
                handle.remove()
            self.assertGreater(float((on_context['capacity']-off_context['capacity']).abs().max()), 0.)
            self.assertGreater(sum(float(row[:, 12].sum()) for row in on_context['records']), 0.)
            np.testing.assert_array_equal(on, off)
            for key in on_trace.tensors:
                np.testing.assert_array_equal(on_trace.tensors[key], off_trace.tensors[key])

    def test_exception_removes_every_hook_and_restores_global_exchange_step(self):
        with controlled_forecast() as (model, x, report):
            original_step = spatial.exchange_step
            hooks = hook_counts(model)
            proposal = exchange_modules(model)[2].proposal
            with patch.object(proposal, 'forward', side_effect=RuntimeError('fixture proposal failure')):
                with self.assertRaisesRegex(RuntimeError, 'fixture proposal failure'):
                    with probe.ExchangeTrace(model):
                        probe.forward(model, batch_for(x, report), {'mean': 0., 'std': 1.}, True)
            self.assertIs(spatial.exchange_step, original_step)
            self.assertEqual(hook_counts(model), hooks)
            # A later ordinary forward must remain usable after the failed instrumented pass.
            prediction, _ = probe.forward(model, batch_for(x, report), {'mean': 0., 'std': 1.}, True)
            self.assertTrue(np.isfinite(prediction).all())

    def test_rejects_training_mode_and_missing_forward_coverage(self):
        with controlled_forecast() as (model, x, report):
            model.train()
            with self.assertRaises(ValueError):
                with probe.ExchangeTrace(model):
                    probe.forward(model, batch_for(x, report), {'mean': 0., 'std': 1.}, True)
            model.eval()
            with self.assertRaises(ValueError):
                with probe.ExchangeTrace(model):
                    pass

    def test_restoration_detects_changed_exchange_values_and_cleans_up(self):
        with controlled_forecast() as (model, x, report):
            batch = batch_for(x, report)
            with probe.ExchangeTrace(model) as initial:
                probe.forward(model, batch, {'mean': 0., 'std': 1.}, True)
            # A valid intervention is not a valid final ON restoration.
            with torch.no_grad():
                model.report_spatial_state.D[-1].bias.fill_(1.)
            original_step = spatial.exchange_step
            hooks = hook_counts(model)
            with self.assertRaises(ValueError):
                with probe.ExchangeTrace(model, reference=initial.tensors, restoration=True):
                    probe.forward(model, batch, {'mean': 0., 'std': 1.}, True)
            self.assertIs(spatial.exchange_step, original_step)
            self.assertEqual(hook_counts(model), hooks)


class MetricAndReplayTests(unittest.TestCase):
    def test_best_selection_uses_first_minimum_and_rejects_inconsistent_saved_evidence(self):
        identity = {'fixture': 'best_selection'}
        history = [dict(epoch=epoch, validation_mae_macro=mae, best_epoch=best)
                   for epoch, mae, best in [(1, 3., 1), (2, 1., 2), (3, 1., 2), (4, 2., 2)]]
        checkpoint = dict(identity=identity, arm='B3R', history=history, completed_epoch=4,
                          best_metric=1., best_model_state={'epoch': torch.tensor(2)},
                          model_state={'epoch': torch.tensor(4)})
        summary = dict(arm='B3R', completed_epoch=4, best_epoch=2, best_metric=1.)
        self.assertEqual(probe.select_best(checkpoint, identity, history, summary), (2, 1.))
        self.assertEqual(int(checkpoint['model_state']['epoch']), 4)
        self.assertEqual(int(checkpoint['best_model_state']['epoch']), 2)
        with self.assertRaises(ValueError):
            probe.select_best(checkpoint, identity, history, dict(summary, best_epoch=3))
        with self.assertRaises(ValueError):
            probe.select_best(dict(checkpoint, best_model_state=None), identity, history, summary)
        with self.assertRaises(ValueError):
            probe.select_best(dict(checkpoint, best_metric=2.), identity, history, summary)
        altered = [dict(row) for row in history]
        altered[-1]['best_epoch'] = 3
        with self.assertRaises(ValueError):
            probe.select_best(dict(checkpoint, history=altered), identity, altered, summary)
        with self.assertRaises(ValueError):
            probe.select_best(checkpoint, {'fixture': 'changed_identity'}, history, summary)

    def test_change_totals_keeps_absolute_effect_when_signed_changes_cancel(self):
        collector = probe.ChangeTotals()
        collector.add('fixture', np.zeros(2), np.array([1., -1.]))
        collector.add('fixture', np.zeros(1), np.array([4.]))
        row = collector.result()['fixture']
        self.assertEqual(row['count'], 3)
        self.assertEqual(row['absolute_mean'], 2.)
        self.assertAlmostEqual(row['signed_mean_off_minus_on'], 4./3)
        self.assertEqual(row['absolute_max'], 4.)
        self.assertEqual(row['response_fraction_above']['0.1'], 1.)
        collector.add('empty', np.zeros(2), np.ones(2), np.zeros(2, dtype=bool))
        self.assertEqual(collector.result()['empty']['status'], 'unavailable')
        self.assertIsNone(collector.result()['empty']['absolute_mean'])

    def test_paired_metric_uses_off_minus_on_and_horizon_macro_axes(self):
        on = np.zeros((2, 2, 2, 1), dtype=np.float32)
        off = on.copy()
        off[:, 0] = 2.
        off[:, 1] = 4.
        valid = np.ones_like(on, dtype=bool)
        valid[:, 0] = False
        valid[0, 0, 0] = True
        result = probe.paired_metrics(on, off, np.zeros_like(on), valid, np.ones((2, 2), bool))
        self.assertAlmostEqual(result['on_mae_macro'], 0.)
        self.assertAlmostEqual(result['off_mae_macro'], 3.)
        self.assertAlmostEqual(result['off_minus_on_mae'], 3.)
        inverse = probe.paired_metrics(off, on, np.zeros_like(on), valid, np.ones((2, 2), bool))
        self.assertAlmostEqual(inverse['off_minus_on_mae'], -3.)
        self.assertEqual(list(result['counts']), [1, 4])

    def test_empty_region_is_unavailable_and_wrong_axes_cannot_broadcast(self):
        prediction = np.zeros((2, 2, 2, 1), dtype=np.float32)
        valid = np.ones_like(prediction, dtype=bool)
        result = probe.paired_metrics(prediction, prediction, prediction, valid, np.zeros((2, 2), bool))
        self.assertEqual(result['status'], 'unavailable')
        self.assertIsNone(result['off_minus_on_mae'])
        with self.assertRaises(ValueError):
            probe.paired_metrics(prediction, prediction, prediction[:, :, :, 0], valid, np.ones((2, 2), bool))
        with self.assertRaises(ValueError):
            probe.paired_metrics(prediction, prediction, prediction, valid, np.ones((2, 1), bool))

    def test_saved_window_replay_checks_counts_order_times_and_errors(self):
        prediction = np.arange(8, dtype=np.float32).reshape(2, 2, 2, 1)
        target = np.zeros_like(prediction)
        valid = np.ones_like(prediction, dtype=bool)
        indices = np.array([8, 3], dtype=np.int64)
        t0 = np.array(['2020-01-01T12:00:00', '2020-01-01T12:05:00'])
        saved = dict(absolute=np.abs(prediction.astype(np.float64)-target).sum((2, 3)),
                     counts=valid.sum((2, 3)), sample_indices=indices.copy(), t0=t0.copy())
        expected = float((saved['absolute'].sum(0)/saved['counts'].sum(0)).mean())
        probe.verify_saved_windows(prediction, target, valid, indices, t0, saved, expected)
        changes = {'absolute': saved['absolute']+10., 'counts': saved['counts']+1,
                   'sample_indices': indices[::-1], 't0': t0[::-1]}
        for key, value in changes.items():
            with self.subTest(field=key):
                altered = dict(saved, **{key: value})
                with self.assertRaises(ValueError):
                    probe.verify_saved_windows(prediction, target, valid, indices, t0, altered, expected)
        with self.assertRaises(ValueError):
            probe.verify_saved_windows(prediction, target, valid, indices, t0, saved, expected+1.)


class FullProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_initial_failed_replay_blocks_every_off_call(self):
        with controlled_forecast() as (model, x, report), tempfile.TemporaryDirectory() as directory:
            dataset = ProbeDataset(x, report)
            indices = [row['sample_index'] for row in dataset.rows]
            saved, mae = saved_windows(model, dataset, indices, 3)
            saved['absolute'] += 100.
            calls = []
            original_forward = probe.forward
            def observe(*args, **kwargs):
                calls.append(args[3])
                return original_forward(*args, **kwargs)
            with patch.object(probe, 'forward', side_effect=observe):
                with self.assertRaisesRegex(ValueError, 'OFF blocked'):
                    probe.run_probe(model, dataset, indices, 3, torch.device('cpu'),
                                    model.internal_spatial_graph, {'fixture': [0, 1]}, saved, mae, Path(directory))
            self.assertEqual(calls, [True, True])
            self.assertFalse((Path(directory)/'paired_predictions.npz').exists())
            self.assertFalse((Path(directory)/'exchange_changes.json').exists())

    def test_final_on_prediction_failure_blocks_paired_artifacts_and_restores_hooks(self):
        with controlled_forecast() as (model, x, report), tempfile.TemporaryDirectory() as directory:
            dataset = ProbeDataset(x, report)
            indices = [row['sample_index'] for row in dataset.rows]
            saved, mae = saved_windows(model, dataset, indices, 3)
            original_step, hooks = spatial.exchange_step, hook_counts(model)
            original_forward = probe.forward
            def corrupt_restoration(*args, **kwargs):
                prediction, context = original_forward(*args, **kwargs)
                trace = args[4] if len(args) > 4 else kwargs.get('trace')
                if trace is not None and trace.restoration:
                    prediction += 1.
                return prediction, context
            with patch.object(probe, 'forward', side_effect=corrupt_restoration):
                with self.assertRaisesRegex(ValueError, 'Final ON prediction restoration failed'):
                    probe.run_probe(model, dataset, indices, 3, torch.device('cpu'),
                                    model.internal_spatial_graph, {'fixture': [0, 1]}, saved, mae, Path(directory))
            self.assertIs(spatial.exchange_step, original_step)
            self.assertEqual(hook_counts(model), hooks)
            self.assertFalse((Path(directory)/'paired_predictions.npz').exists())

    def test_complete_probe_preserves_original_tail_batch_and_writes_paired_evidence(self):
        with controlled_forecast() as (model, x, report), tempfile.TemporaryDirectory() as directory:
            with torch.no_grad():
                model.report_spatial_state.D[-1].bias.fill_(1.)
            dataset = ProbeDataset(x, report)
            indices = [row['sample_index'] for row in dataset.rows]
            saved, mae = saved_windows(model, dataset, indices, 3)
            original_step, hooks = spatial.exchange_step, hook_counts(model)
            before = tensor_state_hash(model.state_dict())
            calls = []
            original_forward = probe.forward
            def observe(*args, **kwargs):
                calls.append((len(args[1]['x']), args[3]))
                return original_forward(*args, **kwargs)
            output = Path(directory)
            with patch.object(probe, 'forward', side_effect=observe):
                result = probe.run_probe(model, dataset, indices, 3, torch.device('cpu'),
                                         model.internal_spatial_graph, {'fixture': [0, 1]}, saved, mae, output)
            self.assertEqual(calls, [(3, True), (1, True), (3, True), (3, False), (3, True),
                                     (1, True), (1, False), (1, True)])
            self.assertEqual(result['batch_partition'], [3, 1])
            self.assertTrue(result['inference_only'] and result['final_on_exactly_restored'])
            self.assertTrue(result['tiid_report_enabled'])
            self.assertFalse(result['test_read'])
            self.assertEqual(before, tensor_state_hash(model.state_dict()))
            self.assertIs(spatial.exchange_step, original_step)
            self.assertEqual(hook_counts(model), hooks)
            self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
            for filename in ('replay.json', 'exchange_changes.json', 'capacity_changes.json',
                             'on_mechanisms.json', 'off_mechanisms.json', 'paired_predictions.npz'):
                self.assertTrue((output/filename).is_file(), filename)
            with np.load(output/'paired_predictions.npz', allow_pickle=False) as archive:
                self.assertEqual(archive['on'].shape, (4, 12, 4, 1))
                np.testing.assert_array_equal(archive['sample_indices'], indices)
                self.assertTrue(bool(archive['final_on_exactly_restored']))
                self.assertGreater(float(np.max(np.abs(archive['on']-archive['off']))), 0.)
            metrics = result['metrics']
            gain = metrics['all_nodes']['off_mae_macro']-metrics['all_nodes']['on_mae_macro']
            self.assertAlmostEqual(metrics['all_nodes']['off_minus_on_mae'], gain)
            self.assertEqual(metrics['no_report_samples']['status'], 'unavailable')


if __name__ == '__main__':
    unittest.main()
