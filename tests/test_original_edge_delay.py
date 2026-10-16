"""Lag arithmetic, original-path preservation, meaningful gradients and resume."""
from contextlib import redirect_stdout
import io
import json
import logging
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.chronological.original_edge_delay import (
    EdgeDelay, HistoricalDelayDif, apply_kernels, delay_mode, historical_forward,
    install_delay, lag_kernels, road_messages)
from experiments.chronological.original_igstgnn_source import get_original_modules, verify_original_source
from experiments.chronological.original_igstgnn_runtime import CONFIG, tensor_tree_hash
from experiments.chronological import train_original_edge_delay as pilot


class EdgeDelayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.modules = get_original_modules()

    def temp(self):
        value = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(value.cleanup)
        return Path(value.name)

    def model(self):
        directory = self.temp()
        for name in ('type_mapping.json', 'desc_mapping.json'):
            (directory/name).write_text(json.dumps({'a': 0, 'b': 1}))
        graph = torch.tensor([[0., .3, .7, 0.], [0., 0., 1., 0.], [.5, 0., 0., .5], [0., 1., 0., 0.]])
        model = self.modules.model.IGSTGNN(dict(num_feat=1, num_hidden=8, node_hidden=4, time_emb_dim=4,
            tpd=288, k_s=2, k_t=3, layer=5, gap=3, dropout=.1, dataset='fixture',
            data_path=str(directory), use_sensor_info=False, adjs=[graph, graph.T]),
            node_num=4, input_dim=3, output_dim=1, seq_len=12, horizon=12)
        return model

    def test_fractional_sampling_and_boundary_mass(self):
        graph = torch.tensor([[0., 1.], [0., 0.]])
        delays = torch.full((1, 2, 2), .5)
        kernels = lag_kernels(graph, delays, 2)
        x = torch.zeros(1, 4, 2, 1)
        x[0, :, 1, 0] = torch.tensor([2., 4., 6., 8.])
        y = apply_kernels(x, kernels)
        torch.testing.assert_close(y[0, :, 0, 0], torch.tensor([1., 3., 5., 7.]))
        torch.testing.assert_close(kernels.sum(1), graph.unsqueeze(0))

    def test_same_source_different_receivers_read_different_times(self):
        graph = torch.zeros(3, 3)
        graph[1, 0] = graph[2, 0] = 1
        delays = torch.zeros(1, 3, 3)
        delays[0, 1, 0], delays[0, 2, 0] = 1, 2
        x = torch.zeros(1, 5, 3, 1)
        x[0, :, 0, 0] = torch.arange(5.)
        result = road_messages(x, lag_kernels(graph, delays, 2), 1)[0]
        self.assertEqual(float(result[0, 4, 1, 0]), 3.)
        self.assertEqual(float(result[0, 4, 2, 0]), 2.)

    def test_two_hop_delays_accumulate(self):
        graph = torch.zeros(3, 3)
        graph[1, 0] = graph[2, 1] = 1
        delays = torch.ones(1, 3, 3)
        x = torch.zeros(1, 6, 3, 1)
        x[0, :, 0, 0] = torch.arange(6.) + 10
        first, second = road_messages(x, lag_kernels(graph, delays, 2))
        self.assertEqual(float(first[0, 4, 1, 0]), 13.)
        self.assertEqual(float(second[0, 4, 2, 0]), 12.)
        self.assertEqual(float(second[0, 1, 2, 0]), 0.)

    def test_remove_two_hop_return_path_at_its_accumulated_time(self):
        graph = torch.tensor([[0., 1.], [1., 0.]])
        x = torch.randn(2, 6, 2, 3)
        delays = torch.ones(2, 2, 2)
        _, second = road_messages(x, lag_kernels(graph, delays, 2))
        torch.testing.assert_close(second, torch.zeros_like(second))

    def test_fractional_return_path_matches_explicit_edge_path_reference(self):
        torch.manual_seed(4)
        graph = torch.rand(3, 3, dtype=torch.float64)
        delay = torch.rand(2, 3, 3, dtype=torch.float64) * 2
        x = torch.randn(2, 5, 3, 2, dtype=torch.float64)
        kernels = lag_kernels(graph, delay, 2)
        actual = road_messages(x, kernels)[1]
        expected = torch.zeros_like(x)
        for batch in range(2):
            for t in range(5):
                for receiver in range(3):
                    for middle in range(3):
                        for source in range(3):
                            if receiver == source:
                                continue
                            for a in range(3):
                                for b in range(3):
                                    if t >= a+b:
                                        expected[batch, t, receiver] += kernels[batch, a, receiver, middle] * kernels[batch, b, middle, source] * x[batch, t-a-b, source]
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)

    def test_zero_delay_matches_original_spatial_orders_including_diagonal_rules(self):
        graph = torch.tensor([[.2, .8, 0.], [.3, 0., .7], [.6, .4, 0.]])
        x = torch.randn(2, 5, 3, 4)
        actual = road_messages(x, lag_kernels(graph, torch.zeros(2, 3, 3), 2))
        for order, result in enumerate(actual, 1):
            power = torch.linalg.matrix_power(graph, order)
            power = power * (1-torch.eye(3))
            expected = torch.einsum('nm,btmc->btnc', power, x)
            torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)

    def test_sampler_never_reads_future(self):
        graph = torch.ones(3, 3)
        kernels = lag_kernels(graph, torch.full((1, 3, 3), .4), 2)
        x = torch.randn(1, 6, 3, 2)
        y = x.clone()
        y[:, 4:] += 100
        for left, right in zip(road_messages(x, kernels), road_messages(y, kernels)):
            torch.testing.assert_close(left[:, :4], right[:, :4])

    def test_mass_only_control_preserves_boundary_mass_and_changes_time_read(self):
        torch.manual_seed(3)
        graph = torch.rand(3, 3)
        kernels = lag_kernels(graph, torch.rand(2, 3, 3)*2, 2)
        constant = torch.randn(2, 1, 3, 2).expand(-1, 6, -1, -1)
        for aligned, mass in zip(road_messages(constant, kernels), road_messages(constant, kernels, aligned=False)):
            torch.testing.assert_close(aligned, mass, atol=1e-6, rtol=1e-6)
        changing = constant * torch.arange(1., 7.)[None, :, None, None]
        self.assertFalse(torch.allclose(road_messages(changing, kernels)[1], road_messages(changing, kernels, aligned=False)[1]))

    def test_delay_gradient_matches_finite_difference(self):
        graph = torch.tensor([[0., 1.], [0., 0.]], dtype=torch.float64)
        delay = torch.full((1, 2, 2), .37, dtype=torch.float64, requires_grad=True)
        x = torch.randn(1, 5, 2, 2, dtype=torch.float64)
        def loss(value):
            return sum(result.square().sum() for result in road_messages(x, lag_kernels(graph, value, 2)))
        loss(delay).backward()
        epsilon = 1e-6
        plus, minus = delay.detach().clone(), delay.detach().clone()
        plus[0, 0, 1] += epsilon
        minus[0, 0, 1] -= epsilon
        numeric = (loss(plus)-loss(minus))/(2*epsilon)
        torch.testing.assert_close(delay.grad[0, 0, 1], numeric, atol=1e-7, rtol=1e-7)

    def test_dynamic_correction_learns_then_depends_on_state(self):
        estimator = EdgeDelay(torch.ones(3, 3)-torch.eye(3), 4)
        x = torch.randn(2, 6, 3, 4)
        estimator.mode = 'FIXED'
        initial = estimator.delays(x)
        estimator.mode = 'DYNAMIC'
        torch.testing.assert_close(estimator.delays(x), initial)
        estimator.delays(x).square().sum().backward()
        self.assertGreater(float(estimator.score.weight.grad.abs().sum()), 0.)
        self.assertGreater(float(estimator.edge_logits.grad.abs().sum()), 0.)
        with torch.no_grad():
            estimator.score.weight.add_(.2)
        self.assertFalse(torch.equal(estimator.delays(x), estimator.delays(x+3)))
        estimator.mode = 'FIXED'
        torch.testing.assert_close(estimator.delays(x), estimator.delays(x+3))

    def test_invalid_graphs_rejected(self):
        for graph in (torch.ones(2, 3), -torch.ones(2, 2), torch.full((2, 2), float('nan'))):
            with self.assertRaises(ValueError):
                EdgeDelay(graph, 4)

    def test_no_align_full_model_exact_and_original_source_unchanged(self):
        model = self.model().eval()
        x = torch.rand(2, 12, 4, 3) * .2
        with torch.no_grad():
            expected = model(x)
        before = torch.get_rng_state().clone()
        install_delay(model, 'NO_ALIGN')
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        with torch.no_grad():
            self.assertTrue(torch.equal(model(x), expected))
        self.assertEqual(len(verify_original_source()['files']), 9)

    def test_historical_zero_delay_preserves_every_original_bank(self):
        model = self.model().eval()
        dif = model.layers[0].dif_layer
        history = torch.randn(2, 12, 4, 8)
        graph = torch.rand(4, 4)
        dynamic = [g.unsqueeze(0).expand(2, -1, -1) for g in dif.localized_st_conv.get_graph([graph, graph.T])]
        static = [graph]
        wrapped = HistoricalDelayDif(dif).eval()
        for estimator in wrapped.delays:
            estimator.delays = lambda history, e=estimator: history.new_zeros((len(history), len(e.source)))
        with torch.no_grad():
            expected = dif.localized_st_conv(history, dynamic, static)
            actual = historical_forward(dif.localized_st_conv, history, dynamic, static, wrapped.delays)
        torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-6)

    def test_forecast_uses_native_convolution_and_delay_probe_restores_mode(self):
        model = install_delay(self.model(), 'DYNAMIC').eval()
        calls = [0] * 5
        hooks = []
        for number, layer in enumerate(model.layers):
            def hook(module, args, output, number=number):
                calls[number] += 1
            hooks.append(layer.dif_layer.original.localized_st_conv.register_forward_hook(hook))
        self.addCleanup(lambda: [hook.remove() for hook in hooks])
        x = torch.rand(2, 12, 4, 3) * .2
        with torch.no_grad():
            expected = model(x)
            self.assertEqual(calls, [3]*5)  # historical replaced; three future calls native
            before = tensor_tree_hash(model.state_dict())
            try:
                with delay_mode(model, 'SHUFFLE'):
                    raise RuntimeError('fixture')
            except RuntimeError:
                pass
            self.assertTrue(all(layer.dif_layer.mode == 'DYNAMIC' for layer in model.layers))
            self.assertEqual(before, tensor_tree_hash(model.state_dict()))
            self.assertTrue(torch.equal(model(x), expected))

    def test_complete_model_backward_reaches_delay_and_backbone(self):
        model = install_delay(self.model(), 'DYNAMIC').train()
        x = torch.rand(2, 12, 4, 3) * .2
        loss = model(x).square().mean()
        loss.backward()
        for layer in model.layers:
            self.assertTrue(torch.isfinite(layer.dif_layer.delays[0].edge_logits.grad).all())
            self.assertGreater(float(layer.dif_layer.delays[0].edge_logits.grad.abs().sum()), 0.)
            self.assertGreater(float(layer.dif_layer.delays[0].score.weight.grad.abs().sum()), 0.)
        self.assertGreater(float(model.embedding.weight.grad.abs().sum()), 0.)

    def test_analysis_sign_fixed_hard_and_target_mask(self):
        target = np.ones((20, 12, 4), np.float32)*10
        source = target + np.arange(20)[:, None, None]
        analysis = pilot.compare_predictions(dict(SOURCE=source, CONTROL=target+1, DYNAMIC=target+.5),
            target, np.arange(20), {'road': [0, 1]})
        self.assertEqual(analysis['fixed_hard_ids'], [19, 18])
        self.assertEqual(analysis['comparisons']['DYNAMIC_vs_CONTROL']['all_nodes']['mae_gain'], .5)
        self.assertAlmostEqual(analysis['metrics']['DYNAMIC']['mape_fraction'], .05)

    def test_resume_epoch_boundary_matches_uninterrupted_training(self):
        from tests.test_original_igstgnn_runner import FakeEngine
        class Engine(FakeEngine):
            def __init__(self, **kwargs):
                super().__init__()
                self.model = kwargs['model']
                self._optimizer, self._lr_scheduler = kwargs['optimizer'], kwargs['scheduler']
                self._device = 'cpu'
                self._dataloader = kwargs['dataloader']
            def train_batch(self):
                result = super().train_batch()
                self._cl_len = 12
                return result
        modules = SimpleNamespace(engine=SimpleNamespace(IGSTGNN_Engine=Engine), metrics=self.modules.metrics)
        data = SimpleNamespace(config=dict(CONFIG, device='cpu'),
            loaders={'train_loader': SimpleNamespace(samples=[0, 1, 2, 3], num_batch=4)}, scaler=None,
            identity={'splits': {'val': {'selected_indices': [0, 1]}}}, station_ids=np.arange(4))
        identity = dict(identity_sha256='fixture', training_config=CONFIG)
        def validation(model, *args):
            with torch.no_grad():
                value = float((model(torch.ones(1, 1))-1).abs().mean())
            target = np.ones((2, 12, 4), np.float32)*10
            return dict(mae=value, rmse=value+.1, mape_fraction=.1), target+value, target, [dict(forward_seconds=.1)]
        def run(directory, epochs, resume=False):
            torch.manual_seed(41)
            np.random.seed(42)
            model = torch.nn.Linear(1, 1)
            data.loaders['train_loader'].samples = [0, 1, 2, 3]
            return pilot.run_arm('CONTROL', model, data, modules, identity, directory, epochs, resume)
        a, b = self.temp(), self.temp()
        from experiments.chronological.original_igstgnn_checkpoint_compat import counter_weights_only_compat
        with patch.object(pilot, 'validate', side_effect=validation), redirect_stdout(io.StringIO()), counter_weights_only_compat():
            left = run(a, 3)
            run(b, 1)
            right = run(b, 3, True)
        for field in ('model_state', 'optimizer_state', 'scheduler_state', 'rng_state', 'train_order', 'global_updates', 'best_epoch'):
            self.assertEqual(tensor_tree_hash(left[field]), tensor_tree_hash(right[field]), field)

    def test_no_source_directory_overlap_or_repeated_arms(self):
        for arguments in (['--source-run', 'a', '--run-dir', 'a/sub'],
                ['--source-run', 'a', '--run-dir', 'b', '--arms', 'CONTROL', 'CONTROL']):
            with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
                pilot.parse_args(arguments)

    def test_resume_rejects_unrelated_directory_without_writing(self):
        directory = self.temp()
        (directory/'summary.json').write_text('unrelated experiment')
        with self.assertRaisesRegex(ValueError, 'not an edge-delay pilot'):
            pilot.main(['--source-run', 'unused-source', '--run-dir', str(directory), '--resume', '--check'])
        self.assertEqual((directory/'summary.json').read_text(), 'unrelated experiment')
        self.assertFalse((directory/'active.lock').exists())


if __name__ == '__main__':
    unittest.main()
