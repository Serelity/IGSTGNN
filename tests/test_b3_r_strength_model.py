"""G1.1 identity initialization and signed-strength checks on real IGSTGNN."""

import math
import unittest
from unittest.mock import patch

import torch

from experiments.chronological.b3_r_diagnostics import CapacityDiagnostics
from src.models.incident_spatial_attenuation import from_residual as old_attenuation
from src.models.incident_spatial_residual import EPSILON, from_canonical_b3, tensor_state_hash
from src.models.incident_spatial_strength import MAPPING_VERSION, ReportCapacityStrength, from_residual
from src.utils.chronological import masked_flow_mae
from tests.test_architecture_mechanisms import inputs
from tests.test_b3_r import canonical, controlled_forecast
from tests.test_incident_spatial import graph


def evaluate(model, x, report, enabled=True):
    return model(x, incident_data=report, spatial_report_enabled=enabled, return_spatial_details=True)


class StrengthModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_nonzero_d_identity_gate_preserves_original_full_prediction_and_capacity_exactly(self):
        with controlled_forecast() as (parent, x, report):
            with torch.no_grad():
                parent.report_spatial_state.D[-1].weight.fill_(.02)
                parent.report_spatial_state.D[-1].bias.copy_(torch.tensor([.8, -.6, 1.2, -1.]))
            candidate, certificate = from_residual(parent, 'identity_gate')
            gate = candidate.report_spatial_state.gate
            self.assertEqual(certificate['initial_alpha'], 1.)
            self.assertEqual(gate[0].in_features, parent.report_spatial_state.D[0].in_features)
            self.assertEqual(gate[0].out_features, 16)
            self.assertEqual(gate[-1].out_features, 1)
            self.assertEqual(float(gate[-1].weight.detach().abs().sum()+gate[-1].bias.detach().abs().sum()), 0.)
            self.assertGreater(float(gate[0].weight.detach().abs().sum()), 0.)
            with torch.inference_mode():
                original, gated = evaluate(parent, x, report), evaluate(candidate, x, report)
            torch.testing.assert_close(gated['prediction'], original['prediction'], atol=0, rtol=0)
            for field in ('base_logit', 'report_logit_delta', 'effective_logit_delta', 'updated_logit',
                          'node_capacity', 'capacity', 'base_edge_capacity', 'boundary'):
                torch.testing.assert_close(gated['spatial'][field], original['spatial'][field], atol=0, rtol=0)
            torch.testing.assert_close(gated['spatial']['d_strength'], torch.ones(2, 4, 1), atol=0, rtol=0)
            diagnostics = CapacityDiagnostics()
            diagnostics.update(gated['spatial'])
            self.assertEqual(len(diagnostics.result()['rows']), 40)

    def test_zero_d_task_gradient_matches_original_and_gate_gradient_is_zero(self):
        with controlled_forecast() as (parent, x, report):
            candidate, _ = from_residual(parent, 'identity_gate')
            predictions = []
            for model in (parent, candidate):
                model.zero_grad(set_to_none=True)
                prediction = model(x, incident_data=report)
                predictions.append(prediction.detach())
                loss = masked_flow_mae(prediction, prediction.detach()+1., torch.ones_like(prediction, dtype=torch.bool))
                loss.backward()
            torch.testing.assert_close(predictions[0], predictions[1], atol=0, rtol=0)
            left, right = parent.report_spatial_state.D, candidate.report_spatial_state.D
            for name, parameter in left.named_parameters():
                torch.testing.assert_close(parameter.grad, dict(right.named_parameters())[name].grad, atol=0, rtol=0)
            self.assertGreater(float(left[-1].weight.grad.abs().sum()+left[-1].bias.grad.abs().sum()), 0.)
            self.assertEqual(sum(float(parameter.grad.abs().sum())
                                 for parameter in candidate.report_spatial_state.gate.parameters()), 0.)

    def test_suppressing_and_amplifying_signed_d_match_scalar_logit_and_capacity_oracle(self):
        with controlled_forecast() as (parent, x, report):
            with torch.no_grad():
                parent.report_spatial_state.D[-1].bias.copy_(torch.tensor([.8, -.6, 1.2, -1.]))
            candidate, _ = from_residual(parent, 'identity_gate')
            for alpha in (.5, 1.5):
                with self.subTest(alpha=alpha):
                    with torch.no_grad():
                        candidate.report_spatial_state.gate[-1].bias.fill_(math.log(alpha/(2-alpha)))
                    with torch.inference_mode():
                        context = evaluate(candidate, x, report)['spatial']
                    torch.testing.assert_close(context['d_strength'], torch.full((2, 4, 1), alpha), atol=2e-7, rtol=0)
                    expected_delta = context['report_mask']*context['report_logit_delta']*context['d_strength']
                    expected_logit = context['base_logit']+expected_delta
                    torch.testing.assert_close(context['effective_logit_delta'], expected_delta, atol=0, rtol=0)
                    torch.testing.assert_close(context['updated_logit'], expected_logit, atol=0, rtol=0)
                    torch.testing.assert_close(context['node_capacity'],
                                               EPSILON+(1-EPSILON)*expected_logit.sigmoid(), atol=0, rtol=0)
                    raw = context['raw_effective_logit_delta'][:, 0]
                    actual = context['effective_logit_delta'][:, 0]
                    self.assertTrue((raw[:, [0, 2]] > 0).all())
                    self.assertTrue((raw[:, [1, 3]] < 0).all())
                    self.assertTrue((actual.sign() == raw.sign()).all())
                    self.assertTrue((actual.abs() < raw.abs()).all() if alpha < 1 else (actual.abs() > raw.abs()).all())

    def test_fixed_half_reproduces_frozen_g10_with_nonzero_parent_d(self):
        with controlled_forecast() as (parent, x, report):
            with torch.no_grad():
                parent.report_spatial_state.D[-1].weight.fill_(.02)
                parent.report_spatial_state.D[-1].bias.copy_(torch.tensor([.8, -.6, 1.2, -1.]))
            old, _ = old_attenuation(parent, 'fixed_half')
            half, certificate = from_residual(parent, 'fixed_half')
            self.assertEqual(certificate['initial_alpha'], .5)
            self.assertEqual(certificate['added_parameters'], 0)
            self.assertIsNone(half.report_spatial_state.gate)
            self.assertEqual(tensor_state_hash(old.state_dict()), tensor_state_hash(half.state_dict()))
            with torch.inference_mode():
                a, b = evaluate(old, x, report), evaluate(half, x, report)
            torch.testing.assert_close(a['prediction'], b['prediction'], atol=0, rtol=0)
            for field in ('d_strength', 'raw_effective_logit_delta', 'effective_logit_delta', 'capacity'):
                torch.testing.assert_close(a['spatial'][field], b['spatial'][field], atol=0, rtol=0)

    def test_shared_state_rng_parameter_count_and_source_independence_are_certified(self):
        parent, _ = from_canonical_b3(canonical())
        parent.eval()
        source = parent.state_dict()
        parent_hash = tensor_state_hash(source)
        for mode in ('fixed_half', 'identity_gate'):
            with self.subTest(mode=mode):
                rng = torch.get_rng_state().clone()
                candidate, certificate = from_residual(parent, mode)
                torch.testing.assert_close(torch.get_rng_state(), rng, atol=0, rtol=0)
                self.assertEqual(certificate['mapping_version'], MAPPING_VERSION)
                self.assertEqual(certificate['parent_state_sha256'], parent_hash)
                self.assertEqual(certificate['retained_shared_state_sha256'], parent_hash)
                self.assertEqual(tensor_state_hash(parent.state_dict()), parent_hash)
                self.assertFalse(any(module.training for module in candidate.modules()))
                self.assertIsNot(candidate.tiid_module, parent.tiid_module)
                self.assertIsNot(candidate.internal_spatial_graph, parent.internal_spatial_graph)
                copied = candidate.state_dict()
                for name, value in source.items():
                    torch.testing.assert_close(copied[name], value, atol=0, rtol=0)
                    if value.numel():
                        self.assertNotEqual(copied[name].data_ptr(), value.data_ptr(), name)
                count = sum(parameter.numel() for parameter in candidate.parameters())
                base_count = sum(parameter.numel() for parameter in parent.parameters())
                expected_added = ((parent.report_spatial_state.D[0].in_features+1)*16+17
                                  if mode == 'identity_gate' else 0)
                self.assertEqual(certificate['parameter_count'], count)
                self.assertEqual(certificate['parent_parameter_count'], base_count)
                self.assertEqual(certificate['added_parameters'], expected_added)
                self.assertEqual(count-base_count, expected_added)
                self.assertTrue(all(parameter.requires_grad for parameter in candidate.parameters()))
                with torch.no_grad():
                    candidate.report_spatial_state.P[0].weight.add_(1.)
                self.assertEqual(tensor_state_hash(parent.state_dict()), parent_hash)

    def test_conversion_under_inference_produces_ordinary_trainable_parameters_and_identity_prediction(self):
        with controlled_forecast() as (parent, x, report):
            with torch.no_grad():
                parent.report_spatial_state.D[-1].bias.fill_(1.)
            with torch.inference_mode():
                candidate, _ = from_residual(parent, 'identity_gate')
                self.assertFalse(any(parameter.is_inference() for parameter in candidate.parameters()))
                reference = parent(x, incident_data=report)
                actual = candidate(x, incident_data=report)
            torch.testing.assert_close(reference, actual, atol=0, rtol=0)
            prediction = candidate(x, incident_data=report)
            prediction.sum().backward()
            self.assertTrue(any(parameter.grad is not None for parameter in candidate.parameters()))

    def test_off_skips_d_and_gate_while_preserving_p_boundary_and_original_tiid(self):
        with controlled_forecast() as (parent, x, report):
            with torch.no_grad():
                parent.report_spatial_state.D[-1].bias.fill_(1.)
            candidate, _ = from_residual(parent, 'identity_gate')
            calls = []
            handle = candidate.tiid_module.register_forward_hook(lambda *args: calls.append(True))
            try:
                with torch.inference_mode():
                    on = evaluate(candidate, x, report)
                    with (patch.object(candidate.report_spatial_state.D, 'forward', side_effect=AssertionError('D executed OFF')),
                          patch.object(candidate.report_spatial_state.gate, 'forward', side_effect=AssertionError('G executed OFF'))):
                        off = evaluate(candidate, x, report, False)
                    original_off = evaluate(parent, x, report, False)
                    no_tiid = candidate(x, incident_data=report, spatial_report_enabled=False, tiid_report_enabled=False)
            finally:
                handle.remove()
            self.assertEqual(len(calls), 2)
            torch.testing.assert_close(off['prediction'], original_off['prediction'], atol=0, rtol=0)
            self.assertGreater(float((off['prediction']-no_tiid).abs().max()), 0.)
            self.assertIsNone(off['spatial']['report_logit_delta'])
            self.assertIsNone(off['spatial']['d_strength'])
            self.assertFalse(off['spatial']['report_path_executed'])
            for field in ('raw_effective_logit_delta', 'effective_logit_delta'):
                self.assertEqual(float(off['spatial'][field].abs().sum()), 0.)
            torch.testing.assert_close(off['spatial']['node_capacity'], off['spatial']['base_node_capacity'], atol=0, rtol=0)
            for field in ('base_logit', 'base_node_capacity', 'base_edge_capacity', 'boundary', 'report_mask'):
                torch.testing.assert_close(on['spatial'][field], off['spatial'][field], atol=0, rtol=0)

    def test_unsupported_nodes_are_exact_p_and_capacity_remains_bounded_at_gate_extremes(self):
        with controlled_forecast() as (parent, x, report):
            with torch.no_grad():
                parent.report_spatial_state.D[-1].bias.copy_(torch.tensor([3., -3., 5., -5.]))
            candidate, _ = from_residual(parent, 'identity_gate')
            for bias in (-100., 100.):
                with self.subTest(bias=bias):
                    with torch.no_grad():
                        candidate.report_spatial_state.gate[-1].bias.fill_(bias)
                    with torch.inference_mode():
                        context = evaluate(candidate, x, report)['spatial']
                    unsupported = ~context['report_mask'].bool().expand_as(context['base_logit'])
                    torch.testing.assert_close(context['node_capacity'][unsupported],
                                               context['base_node_capacity'][unsupported], atol=0, rtol=0)
                    self.assertEqual(float(context['effective_logit_delta'][unsupported].abs().sum()), 0.)
                    self.assertTrue(((context['d_strength'] >= 0.) & (context['d_strength'] <= 2.)).all())
                    self.assertTrue(((context['node_capacity'] >= EPSILON) & (context['node_capacity'] <= 1.)).all())
                    if bias < 0:
                        torch.testing.assert_close(context['node_capacity'], context['base_node_capacity'], atol=0, rtol=0)

    def test_complete_inputs_are_shared_with_d_and_future_target_fields_are_not_consumed(self):
        with controlled_forecast() as (parent, x, report):
            with torch.no_grad():
                parent.report_spatial_state.D[-1].bias.fill_(1.)
            candidate, _ = from_residual(parent, 'identity_gate')
            with torch.no_grad():
                candidate.report_spatial_state.gate[-1].weight.fill_(.01)
            d_inputs, gate_inputs = [], []
            handles = [candidate.report_spatial_state.D.register_forward_pre_hook(
                lambda module, args: d_inputs.append(args[0].detach().clone())),
                candidate.report_spatial_state.gate.register_forward_pre_hook(
                    lambda module, args: gate_inputs.append(args[0].detach().clone()))]
            try:
                with torch.inference_mode():
                    reference = evaluate(candidate, x, report)
                    supplied = dict(report, y_flow=torch.full((2, 12, 4, 1), 1e9),
                                    future_traffic=torch.full_like(x, -1e9),
                                    incident_resolution_time=torch.full((2,), 1e9))
                    altered = evaluate(candidate, x, supplied)
            finally:
                for handle in handles:
                    handle.remove()
            for a, b in zip(d_inputs, gate_inputs):
                torch.testing.assert_close(a, b, atol=0, rtol=0)
            torch.testing.assert_close(d_inputs[0], d_inputs[1], atol=0, rtol=0)
            torch.testing.assert_close(reference['prediction'], altered['prediction'], atol=0, rtol=0)

    def test_real_forecast_mae_trains_gate_last_layer_then_hidden_layer_with_nonzero_d(self):
        with controlled_forecast() as (parent, x, report):
            with torch.no_grad():
                parent.report_spatial_state.D[-1].bias.fill_(1.)
            candidate, _ = from_residual(parent, 'identity_gate')
            gate = candidate.report_spatial_state.gate
            optimizer = torch.optim.SGD(gate.parameters(), lr=.1, weight_decay=0.)
            gradients = []
            for step in range(2):
                candidate.zero_grad(set_to_none=True)
                prediction = candidate(x, incident_data=report)
                masked_flow_mae(prediction, prediction.detach()+1., torch.ones_like(prediction, dtype=torch.bool)).backward()
                gradients.append((float(gate[0].weight.grad.abs().sum()+gate[0].bias.grad.abs().sum()),
                                  float(gate[-1].weight.grad.abs().sum()+gate[-1].bias.grad.abs().sum())))
                self.assertGreater(gradients[-1][1], 0.)
                self.assertTrue(all(torch.isfinite(parameter.grad).all() for parameter in gate.parameters()))
                optimizer.step()
            self.assertEqual(gradients[0][0], 0.)
            self.assertGreater(gradients[1][0], 0.)

    def test_open_edge_mapping_and_boundary_heads_remain_original(self):
        parent, _ = from_canonical_b3(canonical())
        parent.eval()
        with torch.no_grad():
            parent.report_spatial_state.D[-1].bias.copy_(torch.tensor([.8, -.6, 1.2, -1.]))
        candidate, _ = from_residual(parent, 'identity_gate')
        _, report = inputs()
        generator = torch.Generator().manual_seed(123)
        history = torch.rand((2, 12, 4, 8), generator=generator)
        tod, dow, embedding = [torch.rand(shape, generator=generator) for shape in ((2, 4), (2, 4), (2, 8))]
        g = graph(4, ((-1, 0), (0, 1), (1, -1), (-1, 3), (3, -1)))
        with torch.inference_mode():
            original = parent.report_spatial_state(history, tod, dow, embedding, report, g)
            context = candidate.report_spatial_state(history, tod, dow, embedding, report, g)
        capacity = context['node_capacity']
        expected = torch.stack((capacity[:, 0], torch.minimum(capacity[:, 0], capacity[:, 1]),
                                capacity[:, 1], capacity[:, 3], capacity[:, 3]), dim=1)
        torch.testing.assert_close(context['capacity'], expected, atol=0, rtol=0)
        torch.testing.assert_close(context['boundary'], original['boundary'], atol=0, rtol=0)
        torch.testing.assert_close(context['boundary'][:, 1], torch.zeros_like(context['boundary'][:, 1]), atol=0, rtol=0)

    def test_unknown_mode_nonresidual_input_and_repeated_conversion_are_rejected(self):
        parent, _ = from_canonical_b3(canonical())
        with self.assertRaises(ValueError):
            from_residual(parent, 'state_gate')
        with self.assertRaises(ValueError):
            from_residual(canonical(), 'identity_gate')
        candidate, _ = from_residual(parent, 'identity_gate')
        self.assertIsInstance(candidate.report_spatial_state, ReportCapacityStrength)
        with self.assertRaises(ValueError):
            from_residual(candidate, 'identity_gate')


if __name__ == '__main__':
    unittest.main()
