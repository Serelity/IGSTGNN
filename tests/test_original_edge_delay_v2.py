"""V2 path oracle, exact initialization/RNG, staged gradients and audit guards."""
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

from experiments.chronological import original_edge_delay_v2 as v2
from experiments.chronological import train_original_edge_delay_v2 as runner
from experiments.chronological.original_edge_delay import lag_kernels, road_messages
from experiments.chronological.original_edge_delay_gradient_audit import audit_gradients, decay_only_adam
from experiments.chronological.original_igstgnn_source import get_original_modules, verify_original_source
from experiments.chronological.original_igstgnn_runtime import CONFIG, rng_fingerprint, tensor_tree_hash


class V2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.modules = get_original_modules()

    def temp(self):
        directory = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(directory.cleanup)
        return Path(directory.name)

    def model(self):
        directory = self.temp()
        for name in ('type_mapping.json', 'desc_mapping.json'):
            (directory / name).write_text(json.dumps({'a': 0, 'b': 1}))
        graph = torch.tensor([[0., .3, .7, 0.], [0., 0., 1., 0.], [.5, 0., 0., .5], [0., 1., 0., 0.]])
        return self.modules.model.IGSTGNN(dict(num_feat=1, num_hidden=8, node_hidden=4, time_emb_dim=4,
            tpd=288, k_s=2, k_t=3, layer=5, gap=3, dropout=.1, dataset='fixture',
            data_path=str(directory), use_sensor_info=False, adjs=[graph, graph.T]),
            node_num=4, input_dim=3, output_dim=1, seq_len=12, horizon=12)

    def test_path_fallback_matches_explicit_fractional_one_and_two_hop_oracle(self):
        torch.manual_seed(31)
        graph = torch.rand(3, 3, dtype=torch.float64)
        delay = torch.rand(2, 3, 3, dtype=torch.float64) * 2
        kernels = lag_kernels(graph, delay, 2)
        x = torch.randn(2, 4, 3, 2, dtype=torch.float64)
        powers = [graph, graph @ graph]
        native = [torch.einsum('nm,btmc->btnc', a * (1 - torch.eye(3)), x) for a in powers]
        actual = [a + r for a, r in zip(native, v2.temporal_residuals(x, kernels))]
        expected = [torch.zeros_like(x), torch.zeros_like(x)]
        for batch in range(2):
            for time in range(4):
                for receiver in range(3):
                    for source in range(3):
                        if receiver == source:
                            continue
                        for a in range(3):
                            expected[0][batch, time, receiver] += kernels[batch, a, receiver, source] * x[batch, time - a if time >= a else time, source]
                            for middle in range(3):
                                for b in range(3):
                                    coefficient = kernels[batch, a, receiver, middle] * kernels[batch, b, middle, source]
                                    expected[1][batch, time, receiver] += coefficient * x[batch, time - a - b if time >= a + b else time, source]
        for observed, oracle in zip(actual, expected):
            torch.testing.assert_close(observed, oracle, atol=1e-12, rtol=1e-12)

    def test_constant_sequences_keep_full_native_mass_at_every_boundary(self):
        graph = torch.rand(4, 4, dtype=torch.float64)
        x = torch.rand(2, 1, 4, 3, dtype=torch.float64).expand(-1, 2, -1, -1)
        kernels = lag_kernels(graph, torch.rand(2, 4, 4, dtype=torch.float64) * 2, 2)
        for residual in v2.temporal_residuals(x, kernels):
            torch.testing.assert_close(residual, torch.zeros_like(residual), atol=2e-15, rtol=0)

    def test_missing_path_uses_current_value_not_first_history(self):
        graph = torch.zeros(3, 3)
        graph[1, 0] = graph[2, 1] = 1
        x = torch.arange(5.).view(1, 5, 1, 1).expand(-1, -1, 3, -1) + 10
        kernels = lag_kernels(graph, torch.ones(1, 3, 3) * 2, 2)
        residual = v2.temporal_residuals(x, kernels)[1]
        fallback = torch.einsum('nm,btmc->btnc', graph @ graph, x) + residual
        self.assertEqual(float(fallback[0, 2, 2, 0]), 12.)
        self.assertEqual(float(fallback[0, 4, 2, 0]), 10.)
        self.assertEqual(float(road_messages(x, kernels)[1][0, 2, 2, 0]), 0.)

    def test_fixed_kernel_does_not_read_future_values(self):
        graph = torch.rand(3, 3)
        kernels = lag_kernels(graph, torch.rand(2, 3, 3) * 2, 2)
        x = torch.randn(2, 6, 3, 2)
        later = x.clone()
        later[:, 4:] += 100
        for a, b in zip(v2.temporal_residuals(x, kernels), v2.temporal_residuals(later, kernels)):
            torch.testing.assert_close(a[:, :4], b[:, :4])

    def test_fallback_delay_gradient_finite_difference(self):
        graph = torch.rand(3, 3, dtype=torch.float64)
        delay = torch.full((1, 3, 3), .37, dtype=torch.float64, requires_grad=True)
        x = torch.randn(1, 5, 3, 2, dtype=torch.float64)
        def loss(d):
            return sum(r.square().sum() for r in v2.temporal_residuals(x, lag_kernels(graph, d, 2)))
        loss(delay).backward()
        plus, minus = delay.detach().clone(), delay.detach().clone()
        plus[0, 0, 1] += 1e-6
        minus[0, 0, 1] -= 1e-6
        torch.testing.assert_close(delay.grad[0, 0, 1], (loss(plus) - loss(minus)) / 2e-6, atol=1e-7, rtol=1e-7)

    def test_only_first_layer_wrapped_and_install_preserves_rng(self):
        model = self.model()
        rng = rng_fingerprint()
        v2.install_delay(model)
        self.assertEqual(rng, rng_fingerprint())
        self.assertIsInstance(model.layers[0].dif_layer, v2.FirstLayerDelayDif)
        self.assertTrue(all(not hasattr(layer.dif_layer, 'original') for layer in model.layers[1:]))
        self.assertEqual(len(verify_original_source()['files']), 9)
        with self.assertRaises(ValueError):
            v2.install_delay(model)

    def test_initial_complete_model_exact_in_eval_and_training_with_same_dropout_rng(self):
        import copy
        torch.manual_seed(12)
        model = self.model()
        aligned = v2.install_delay(copy.deepcopy(model))
        x = torch.rand(2, 12, 4, 3) * .2
        for training in (False, True):
            model.train(training)
            aligned.train(training)
            torch.manual_seed(14)
            with torch.no_grad():
                expected = model(x)
            expected_rng = torch.get_rng_state().clone()
            torch.manual_seed(14)
            with torch.no_grad():
                actual = aligned(x)
            self.assertTrue(torch.equal(expected, actual))
            self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))

    def test_zero_scale_backbone_gradient_matches_native(self):
        import copy
        torch.manual_seed(7)
        original = self.model().train()
        aligned = v2.install_delay(copy.deepcopy(original)).train()
        x = torch.rand(2, 12, 4, 3) * .2
        torch.manual_seed(9)
        original(x).square().mean().backward()
        torch.manual_seed(9)
        aligned(x).square().mean().backward()
        other = dict(aligned.named_parameters())
        for name, parameter in original.named_parameters():
            key = name.replace('layers.0.dif_layer.', 'layers.0.dif_layer.original.')
            if parameter.grad is None:
                self.assertIsNone(other[key].grad)
            else:
                torch.testing.assert_close(parameter.grad, other[key].grad, atol=1e-6, rtol=1e-6)

    def test_scale_learns_first_then_delay_and_state_score_receive_task_gradients(self):
        torch.manual_seed(22)
        model = v2.install_delay(self.model()).eval()
        x = torch.rand(2, 12, 4, 3) * .2
        model(x).square().mean().backward()
        wrapper = model.layers[0].dif_layer
        self.assertGreater(float(wrapper.alignment_scale.grad.abs().sum()), 0)
        for estimator in wrapper.delays:
            self.assertEqual(float(estimator.edge_deviation.grad.abs().sum()), 0)
        model.zero_grad(set_to_none=True)
        with torch.no_grad():
            wrapper.alignment_scale.fill_(.1)
        model(x).square().mean().backward()
        for estimator in wrapper.delays:
            self.assertGreater(float(estimator.edge_deviation.grad.abs().sum()), 0)
            self.assertGreater(float(estimator.score.weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.embedding.weight.grad.abs().sum()), 0)

    def test_future_native_convolution_calls_and_probe_input_lags_are_preserved(self):
        model = v2.install_delay(self.model()).eval()
        wrapper = model.layers[0].dif_layer
        with torch.no_grad():
            wrapper.alignment_scale.fill_(.1)
            wrapper.delays[0].score.weight.fill_(.1)
        calls = [0] * 5
        handles = []
        for number, layer in enumerate(model.layers):
            conv = wrapper.original.localized_st_conv if number == 0 else layer.dif_layer.localized_st_conv
            handles.append(conv.register_forward_hook(lambda m, a, o, n=number: calls.__setitem__(n, calls[n] + 1)))
        self.addCleanup(lambda: [h.remove() for h in handles])
        x = torch.rand(2, 12, 4, 3) * .2
        with torch.no_grad():
            expected = model(x)
            lags = [e.last_delay.clone() for e in wrapper.delays]
            self.assertEqual(calls, [3, 4, 4, 4, 4])
            with v2.delay_mode(model, 'ZERO_BOUNDARY'):
                altered = model(x)
                self.assertTrue(all(torch.equal(a, b.last_delay) for a, b in zip(lags, wrapper.delays)))
            self.assertFalse(torch.equal(expected, altered))
            with v2.delay_mode(model, 'SHUFFLE'):
                model(x)
                self.assertTrue(all(torch.equal(a.roll(1, -1), b.last_delay) for a, b in zip(lags, wrapper.delays)))
            self.assertTrue(torch.equal(expected, model(x)))

    def test_probe_exception_restores_mode_and_state(self):
        model = v2.install_delay(self.model(), 'FIXED')
        before = tensor_tree_hash(model.state_dict())
        with self.assertRaises(RuntimeError):
            with v2.delay_mode(model, 'SHUFFLE'):
                raise RuntimeError('fixture')
        self.assertEqual(model.layers[0].dif_layer.mode, 'FIXED')
        self.assertEqual(before, tensor_tree_hash(model.state_dict()))

    def test_static_and_dynamic_delays_depend_on_correct_inputs(self):
        estimator = v2.ResidualEdgeDelay(torch.ones(3, 3), 4)
        x = torch.randn(2, 6, 3, 4)
        self.assertNotIn('base_logit', dict(estimator.named_parameters()))
        self.assertEqual(float(estimator.base_logit), -2.)
        with torch.no_grad():
            estimator.score.weight.fill_(.2)
        self.assertFalse(torch.equal(estimator.delays(x), estimator.delays(x + 2)))
        estimator.mode = 'FIXED'
        self.assertTrue(torch.equal(estimator.delays(x), estimator.delays(x + 2)))

    def test_decay_only_control_exposes_v1_bias_drift_and_keeps_v2_reference(self):
        old = torch.nn.Parameter(torch.tensor(-2., dtype=torch.float64))
        estimator = v2.ResidualEdgeDelay(torch.ones(2, 2), 4).double()
        opt = torch.optim.Adam([old, estimator.edge_deviation], lr=.002, weight_decay=1e-5, eps=1e-8)
        for _ in range(82):
            old.grad = torch.zeros_like(old)
            estimator.edge_deviation.grad = torch.zeros_like(estimator.edge_deviation)
            opt.step()
        self.assertAlmostEqual(float(old), decay_only_adam()['final_logit'], places=12)
        self.assertAlmostEqual(decay_only_adam()['final_delay'], .27451909582815076, places=12)
        self.assertTrue(torch.equal(estimator.edge_deviation, torch.zeros_like(estimator.edge_deviation)))
        self.assertEqual(float(estimator.base_logit), -2.)

    def test_audit_restores_buffers_modes_rng_and_existing_gradients_without_optimizer(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 1)
                self.bn = torch.nn.BatchNorm1d(1)
                self.dropout = torch.nn.Dropout(.2)
            def forward(self, x, y, **kwargs):
                values = self.linear(x)
                return self.dropout(self.bn(values.flatten(0, 2))).view_as(y)
        model = Model().eval()
        model.dropout.train()
        for p in model.parameters():
            p.grad = torch.ones_like(p) * .3
        before, rng = tensor_tree_hash(model.state_dict()), rng_fingerprint()
        grads = tensor_tree_hash([p.grad for p in model.parameters()])
        modes = [m.training for m in model.modules()]
        data = SimpleNamespace(scaler=SimpleNamespace(inverse_transform=lambda x: x))
        modules = SimpleNamespace(metrics=SimpleNamespace(masked_mae=lambda p, y, mask: (p-y).abs().mean()))
        x, y = torch.randn(2, 12, 3, 3), torch.ones(2, 12, 3, 1) * 3
        # Snapshot RNG after generating the fixture inputs.
        rng = rng_fingerprint()
        with patch('experiments.chronological.original_edge_delay_gradient_audit.device_batch', return_value=(x, y, None, None)), patch.object(torch.optim, 'Adam', side_effect=AssertionError('optimizer prohibited')):
            result = audit_gradients(model, [None, None], data, modules,
                weight_decay=1e-5, clip_norm=5, device='cpu', phase='fixture')
        self.assertEqual(result['optimizer_updates'], 0)
        self.assertEqual(result['backward_calls'], 2)
        self.assertEqual(before, tensor_tree_hash(model.state_dict()))
        self.assertEqual(rng, rng_fingerprint())
        self.assertEqual(grads, tensor_tree_hash([p.grad for p in model.parameters()]))
        self.assertEqual(modes, [m.training for m in model.modules()])

    def test_audit_failure_also_restores_state(self):
        model = torch.nn.Linear(1, 1)
        before, rng = tensor_tree_hash(model.state_dict()), rng_fingerprint()
        with patch('experiments.chronological.original_edge_delay_gradient_audit.device_batch', side_effect=RuntimeError('fixture')):
            with self.assertRaises(RuntimeError):
                audit_gradients(model, [None], None, None, weight_decay=1e-5, clip_norm=5, device='cpu', phase='fixture')
        self.assertEqual(before, tensor_tree_hash(model.state_dict()))
        self.assertEqual(rng, rng_fingerprint())

    def test_output_overlap_and_legacy_resume_rejected_before_writes(self):
        with self.assertRaises(SystemExit):
            runner.parse_args(['--source-run', 'a', '--run-dir', 'a/sub'])
        directory = self.temp()
        (directory / 'identity.json').write_text(json.dumps({'schema': 'old'}))
        with self.assertRaisesRegex(ValueError, 'not a v2'):
            runner.main(['--source-run', 'unused', '--run-dir', str(directory), '--resume'])
        self.assertFalse((directory / 'active.lock').exists())

    def test_telemetry_records_first_layer_raw_boundary_support(self):
        model = v2.install_delay(self.model()).eval()
        observer = runner.telemetry_for(model)
        try:
            with torch.no_grad():
                model(torch.rand(2, 12, 4, 3) * .2)
            observer.consume(12)
            result = observer.finish()
        finally:
            observer.close()
        self.assertEqual({r['layer'] for r in result['lags']}, {0})
        self.assertEqual(len(result['lags']), 2)
        self.assertTrue(any(r['missing_weight_fraction'] > 0 for r in result['boundary_support']))

    def test_best_last_pairs_do_not_mix_checkpoint_conditions(self):
        cases = dict(SOURCE=None)
        for arm in runner.ARMS:
            for stage in ('BEST', 'LAST'):
                cases[arm + '_' + stage] = None
                if arm != 'CONTROL':
                    modes = ('NO_ALIGN', 'ZERO_BOUNDARY') if arm == 'FIXED' else ('NO_ALIGN', 'ZERO_BOUNDARY', 'SHUFFLE')
                    cases.update({arm + '_' + stage + '_' + mode: None for mode in modes})
        pairs = runner.pairs_for(cases)
        self.assertIn(('DYNAMIC_LAST_ZERO_BOUNDARY', 'DYNAMIC_LAST'), pairs)
        self.assertIn(('FIXED_BEST_NO_ALIGN', 'FIXED_BEST'), pairs)
        self.assertNotIn(('DYNAMIC_BEST_ZERO_BOUNDARY', 'DYNAMIC_LAST'), pairs)

    def test_evaluation_detects_rng_mutation_and_always_removes_hooks(self):
        model = v2.install_delay(self.model()).eval()
        wrapper = model.layers[0].dif_layer
        native = model.forward
        def random_forward(*args, **kwargs):
            result = native(*args, **kwargs)
            return result + torch.rand_like(result)
        model.forward = random_forward
        x, y = torch.rand(2, 12, 4, 3) * .2, torch.ones(2, 12, 4, 1) * 10
        data = SimpleNamespace(loaders={'val_loader': SimpleNamespace(num_batch=1, get_iterator=lambda: iter([None]))},
            scaler=SimpleNamespace(inverse_transform=lambda x: x))
        count = len(wrapper._forward_hooks)
        with patch.object(runner, 'device_batch', return_value=(x, y, None, None)), redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, 'RNG'):
                runner.evaluate(model, data, self.modules, 'cpu', 'fixture', {})
        self.assertEqual(len(wrapper._forward_hooks), count)


if __name__ == '__main__':
    unittest.main()
