"""Independent coefficient and real five-layer inference intervention checks."""
import copy
import unittest
from unittest.mock import patch

import torch

from src.models.incident_spatial_forecast_fusion import from_minimal, fuse_components
from src.models.incident_spatial_fusion_relative import relative_fusion, transform_delta
from src.models.incident_spatial_residual import tensor_state_hash
from tests.test_min_event_budget_model import event_inputs, minimal
from tests.test_min_forecast_fusion_model import activate


def independent_relative(delta, mode, bound):
    """Oracle uses explicit ten-component arithmetic, not the implementation."""
    values = delta.to(torch.float64)
    common = sum(values[..., k:k+1] for k in range(10))/10
    centered = torch.cat([values[..., k:k+1]-common for k in range(10)], -1)
    if mode == 'bounded':
        maximum = torch.stack([centered[..., k].abs() for k in range(10)], -1).max(-1, keepdim=True).values
        factor = torch.where(maximum > bound, bound/maximum.clamp_min(1e-30), torch.ones_like(maximum))
        centered = centered*factor
    return centered.to(delta.dtype)


class RelativeForecastFusionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(937)
        self.model, _ = from_minimal(minimal(), mode='event', width=12)
        self.model.eval()
        self.x, self.incident = event_inputs()

    def test_hand_centering_and_shared_bounding_preserve_axes_and_relative_ratios(self):
        for dtype in (torch.float32, torch.float64):
            raw = torch.full((2, 12, 4, 10), -.1, dtype=dtype)
            raw[..., 0] = .1
            raw[:, :, 0] = torch.linspace(-.02, .02, 10, dtype=dtype)
            for mode in ('exact', 'bounded'):
                with self.subTest(dtype=dtype, mode=mode):
                    original = raw.clone()
                    out, stats = transform_delta(raw, mode)
                    expected = independent_relative(raw, mode, .1)
                    # An explicit left-to-right oracle may differ from a
                    # parallel float64 reduction by the last bit.
                    torch.testing.assert_close(out, expected, atol=1e-16, rtol=1e-15)
                    torch.testing.assert_close(raw, original, atol=0, rtol=0)
                    self.assertEqual(out.dtype, dtype)
                    self.assertEqual(out.device, raw.device)
                    self.assertEqual(out.shape, raw.shape)
                    self.assertLessEqual(float(out.double().sum(-1).abs().max()), 64*torch.finfo(dtype).eps*.1)
                    self.assertEqual(stats['positions'], 96)
                    self.assertEqual(stats['component_positions'], 960)
                    self.assertEqual(stats['uncapped_exceeds_bound_positions'], 72)
                    self.assertAlmostEqual(stats['uncapped_relative_abs_max'], .18, places=7)
                    if mode == 'exact':
                        self.assertGreater(stats['output_delta_abs_max'], .1)
                        self.assertEqual(stats['bounded_rescaled_positions'], 0)
                        self.assertEqual(stats['bounded_scale_min'], 1.)
                    else:
                        self.assertLessEqual(stats['output_delta_abs_max'], .1+1e-8)
                        self.assertEqual(stats['bounded_rescaled_positions'], 72)
                        self.assertAlmostEqual(stats['bounded_scale_min'], 5/9, places=7)
                        self.assertAlmostEqual(float(out[0, 0, 1, 0]), .1, places=7)
                        self.assertAlmostEqual(float(out[0, 0, 1, 1]), -1/90, places=7)

    def test_constant_coefficients_and_zero_initialization_are_exact_fixed_sum(self):
        constant = torch.full((2, 12, 4, 10), .037, dtype=torch.float32)
        with torch.inference_mode():
            native = self.model(self.x, incident_data=self.incident, fusion_enabled=False)
            for mode in ('exact', 'bounded'):
                output, stats = transform_delta(constant, mode)
                self.assertTrue(torch.equal(output, torch.zeros_like(output)))
                self.assertEqual(stats['output_delta_component_sum_abs_max'], 0.)
                with relative_fusion(self.model, mode) as audit:
                    actual = self.model(self.x, incident_data=self.incident)
                self.assertTrue(torch.equal(actual, native))
                self.assertEqual(audit['hook_calls'], 1)
                self.assertEqual(audit['output_delta_abs_max'], 0.)

    def test_actual_five_layer_outputs_match_independent_coefficient_oracle_once(self):
        activate(self.model)
        before = tensor_state_hash(self.model.state_dict())
        for mode in ('exact', 'bounded'):
            oracle = copy.deepcopy(self.model)
            original_forward = oracle.forecast_fusion.forward
            def manual_forward(raw, relation, semantic):
                return independent_relative(original_forward(raw, relation, semantic), mode,
                                            oracle.forecast_fusion.max_delta)
            with torch.inference_mode():
                with patch.object(oracle.forecast_fusion, 'forward', side_effect=manual_forward):
                    expected = oracle(self.x, incident_data=self.incident, return_spatial_details=True)
                with patch.object(self.model, '_prepare_inputs', wraps=self.model._prepare_inputs) as prepare:
                    with relative_fusion(self.model, mode) as audit:
                        actual = self.model(self.x, incident_data=self.incident, return_spatial_details=True)
                self.assertEqual(prepare.call_count, 1)
                self.assertEqual(audit['hook_calls'], 1)
                torch.testing.assert_close(actual['prediction'], expected['prediction'], atol=0, rtol=0)
                torch.testing.assert_close(actual['fusion']['weights'], expected['fusion']['weights'], atol=0, rtol=0)
                self.assertEqual(actual['prediction'].shape, (2, 12, 4, 1))
                self.assertEqual(actual['prediction'].dtype, torch.float32)
                self.assertEqual(len(actual['spatial']['records']), 20)
                for key in ('capacity', 'boundary', 'base_logit', 'base_node_capacity'):
                    torch.testing.assert_close(actual['spatial'][key], expected['spatial'][key], atol=0, rtol=0)
            self.assertEqual(tensor_state_hash(self.model.state_dict()), before)
            self.assertFalse(self.model.forecast_fusion._forward_hooks)

    def test_centered_feature_fusion_is_not_subtraction_of_final_predictions(self):
        # Use the actual nonlinear native head, with a known ReLU crossing.
        with torch.no_grad():
            self.model.out_fc_1.weight.zero_()
            self.model.out_fc_1.bias.zero_()
            self.model.out_fc_1.weight[0, 0] = 1.
            self.model.out_fc_1.bias[0] = -1.05
            self.model.out_fc_2.weight.zero_()
            self.model.out_fc_2.bias.zero_()
            self.model.out_fc_2.weight[0, 0] = 1.
        hidden = self.model.out_fc_1.in_features
        components = [torch.zeros(1, 4, 1, hidden) for _ in range(10)]
        components[0][..., 0] = 1.
        raw = torch.zeros(1, 12, 1, 10)
        raw[..., 0] = .1
        uniform = raw.mean(-1, keepdim=True).expand_as(raw)
        centered = independent_relative(raw, 'exact', .1)
        def head(delta):
            feature, _ = fuse_components(components[:5], components[5:], delta, horizon=12, gap=3)
            return self.model.out_fc_2(torch.relu(self.model.out_fc_1(torch.relu(feature))))[..., 0]
        with torch.inference_mode():
            original, common, relative, fixed = (head(value) for value in (raw, uniform, centered, torch.zeros_like(raw)))
        self.assertFalse(torch.allclose(relative, original-common+fixed, atol=1e-6, rtol=0))
        torch.testing.assert_close(relative, torch.full_like(relative, .04), atol=1e-6, rtol=0)

    def test_context_preserves_parameters_gradients_modes_inputs_rng_and_original_ON(self):
        activate(self.model)
        gradients = {}
        for index, (name, parameter) in enumerate(self.model.named_parameters()):
            if index % 11 == 0:
                parameter.grad = torch.full_like(parameter, .125)
            gradients[name] = None if parameter.grad is None else parameter.grad.clone()
        tensor_hash = tensor_state_hash(self.model.state_dict())
        modes = {name: module.training for name, module in self.model.named_modules()}
        flags = {name: parameter.requires_grad for name, parameter in self.model.named_parameters()}
        x_before, incidents_before = self.x.clone(), copy.deepcopy(self.incident)
        with torch.inference_mode():
            on = self.model(self.x, incident_data=self.incident)
            rng = torch.get_rng_state().clone()
            for mode in ('exact', 'bounded'):
                with relative_fusion(self.model, mode):
                    self.model(self.x, incident_data=self.incident)
                self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            restored = self.model(self.x, incident_data=self.incident)
        torch.testing.assert_close(restored, on, atol=0, rtol=0)
        torch.testing.assert_close(self.x, x_before, atol=0, rtol=0)
        for name in self.incident:
            torch.testing.assert_close(self.incident[name], incidents_before[name], atol=0, rtol=0)
        self.assertEqual(tensor_state_hash(self.model.state_dict()), tensor_hash)
        self.assertEqual({name: module.training for name, module in self.model.named_modules()}, modes)
        self.assertEqual({name: value.requires_grad for name, value in self.model.named_parameters()}, flags)
        for name, value in self.model.named_parameters():
            if gradients[name] is None:
                self.assertIsNone(value.grad)
            else:
                torch.testing.assert_close(value.grad, gradients[name], atol=0, rtol=0)

    def test_exception_removes_hook_and_preserves_original_exception_and_ON(self):
        activate(self.model)
        with torch.inference_mode():
            baseline = self.model(self.x, incident_data=self.incident)
            with self.assertRaisesRegex(RuntimeError, 'original fixture failure'):
                with relative_fusion(self.model):
                    raise RuntimeError('original fixture failure')
            self.assertFalse(self.model.forecast_fusion._forward_hooks)
            with self.assertRaisesRegex(RuntimeError, 'failure after coefficient transformation'):
                with relative_fusion(self.model):
                    with patch.object(self.model.out_fc_1, 'forward', side_effect=RuntimeError('failure after coefficient transformation')):
                        self.model(self.x, incident_data=self.incident)
            self.assertFalse(self.model.forecast_fusion._forward_hooks)
            restored = self.model(self.x, incident_data=self.incident)
        torch.testing.assert_close(restored, baseline, atol=0, rtol=0)

    def test_incomplete_excess_and_nested_hook_coverage_are_rejected_and_cleaned(self):
        with self.assertRaisesRegex(ValueError, 'coverage'):
            with relative_fusion(self.model):
                pass
        self.assertFalse(self.model.forecast_fusion._forward_hooks)
        with torch.inference_mode():
            with self.assertRaisesRegex(ValueError, 'coverage'):
                with relative_fusion(self.model, expected_calls=1):
                    self.model(self.x, incident_data=self.incident)
                    self.model(self.x, incident_data=self.incident)
            self.assertFalse(self.model.forecast_fusion._forward_hooks)
            with relative_fusion(self.model, expected_calls=2) as audit:
                self.model(self.x, incident_data=self.incident)
                self.model(self.x, incident_data=self.incident)
            self.assertEqual(audit['hook_calls'], 2)
            self.assertEqual(audit['positions'], 2*2*12*4)
            with self.assertRaisesRegex(ValueError, 'pre-existing'):
                with relative_fusion(self.model):
                    with relative_fusion(self.model):
                        pass
            self.assertFalse(self.model.forecast_fusion._forward_hooks)

    def test_eval_inference_original_architecture_and_existing_hooks_are_required(self):
        with self.assertRaisesRegex(ValueError, 'evaluation mode'):
            with relative_fusion(self.model.train()):
                pass
        self.model.eval()
        with self.assertRaisesRegex(ValueError, 'inference/no_grad'):
            with relative_fusion(self.model):
                self.model(self.x, incident_data=self.incident)
        self.assertFalse(self.model.forecast_fusion._forward_hooks)
        state, _ = from_minimal(minimal(), mode='state')
        for invalid in (minimal(), state):
            with self.assertRaisesRegex(ValueError, 'EVENT MIN-F1'):
                with relative_fusion(invalid):
                    pass
        for kind in ('forward', 'pre'):
            handle = (self.model.forecast_fusion.register_forward_hook(lambda module, args, output: output)
                      if kind == 'forward' else self.model.forecast_fusion.register_forward_pre_hook(lambda module, args: args))
            try:
                with self.assertRaisesRegex(ValueError, 'pre-existing'):
                    with relative_fusion(self.model):
                        pass
            finally:
                handle.remove()
        with patch.object(self.model.forecast_fusion, 'forward', wraps=self.model.forecast_fusion.forward):
            with self.assertRaisesRegex(ValueError, 'original coefficient forward'):
                with relative_fusion(self.model):
                    pass

    def test_unknown_mode_invalid_call_count_delta_and_bound_fail_closed(self):
        raw = torch.zeros(2, 12, 4, 10)
        for mode in ('unknown', None):
            with self.assertRaisesRegex(ValueError, 'Unknown'):
                transform_delta(raw, mode)
            with self.assertRaisesRegex(ValueError, 'Unknown'):
                with relative_fusion(self.model, mode):
                    pass
        for count in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                with relative_fusion(self.model, expected_calls=count):
                    pass
        invalid_delta = (raw[:, :11], raw[..., :9], raw[:0], raw[:, :, :0], raw.long(),
                         raw.half(), raw.clone().fill_(float('nan')), raw.clone().fill_(float('inf')),
                         raw.clone().fill_(.1001), raw.clone().requires_grad_(True))
        for invalid in invalid_delta:
            with self.subTest(shape=invalid.shape, dtype=invalid.dtype):
                with self.assertRaises(ValueError):
                    transform_delta(invalid, 'bounded')
        for bound in (0., -1., 1., float('nan'), float('inf'), True):
            with self.assertRaises(ValueError):
                transform_delta(raw, 'bounded', bound)

    def test_context_uses_registered_nondefault_bound_and_checks_mode_change(self):
        self.model.forecast_fusion.max_delta = .04
        with torch.no_grad():
            bias = self.model.forecast_fusion.condition[-1].bias
            bias.fill_(-10.)
            bias[0] = 10.
        with torch.inference_mode():
            with relative_fusion(self.model, 'bounded') as audit:
                actual = self.model(self.x, incident_data=self.incident, return_spatial_details=True)
            self.assertEqual(audit['max_delta'], .04)
            self.assertEqual(audit['bounded_rescaled_positions'], 2*12*4)
            self.assertLessEqual(audit['output_delta_abs_max'], .04+1e-8)
            self.assertLessEqual(float((actual['fusion']['weights']-1).abs().max()), .04+1e-7)
            with self.assertRaisesRegex(ValueError, 'evaluation mode changed'):
                with relative_fusion(self.model):
                    self.model.forecast_fusion.train()
                    self.model(self.x, incident_data=self.incident)
        self.assertFalse(self.model.forecast_fusion._forward_hooks)


if __name__ == '__main__':
    unittest.main()
