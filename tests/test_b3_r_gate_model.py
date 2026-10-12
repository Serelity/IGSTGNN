"""Real-backbone checks for fixed and causal scalar D-logit attenuation."""

import unittest
from unittest.mock import patch

import torch

from experiments.chronological.b3_r_diagnostics import CapacityDiagnostics
from src.models.incident_spatial_attenuation import ReportCapacityAttenuation, from_residual
from src.models.incident_spatial_residual import EPSILON, from_canonical_b3, tensor_state_hash
from src.utils.chronological import masked_flow_mae
from tests.test_b3_r import canonical, controlled_forecast
from tests.test_architecture_mechanisms import inputs
from tests.test_incident_spatial import graph


def result(model, x, report, enabled=True):
    return model(x, incident_data=report, spatial_report_enabled=enabled, return_spatial_details=True)


class AttenuationModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_fixed_half_scales_signed_logit_delta_before_nonlinear_capacity(self):
        with controlled_forecast() as (parent, x, report):
            with torch.no_grad():
                parent.report_spatial_state.D[-1].bias.copy_(torch.tensor([.8, -.6, 1.2, -1.]))
            half, _ = from_residual(parent, 'fixed_half')
            with torch.inference_mode():
                output = result(half, x, report)
            context = output['spatial']
            self.assertIsNone(half.report_spatial_state.gate)
            self.assertEqual(context['d_strength'].shape, (2, 4, 1))
            torch.testing.assert_close(context['d_strength'], torch.full((2, 4, 1), .5), atol=0, rtol=0)
            raw = context['report_mask']*context['report_logit_delta']
            expected_logit = context['base_logit']+.5*raw
            expected_node = EPSILON+(1-EPSILON)*expected_logit.sigmoid()
            torch.testing.assert_close(context['raw_effective_logit_delta'], raw, atol=0, rtol=0)
            torch.testing.assert_close(context['effective_logit_delta'], .5*raw, atol=0, rtol=0)
            torch.testing.assert_close(context['updated_logit'], expected_logit, atol=0, rtol=0)
            torch.testing.assert_close(context['node_capacity'], expected_node, atol=0, rtol=0)
            self.assertTrue((context['effective_logit_delta'][:, 0, [0, 2]] > 0).all())
            self.assertTrue((context['effective_logit_delta'][:, 0, [1, 3]] < 0).all())

    def test_initial_state_gate_equals_fixed_half_for_nonzero_trained_d_and_full_forecast(self):
        with controlled_forecast() as (parent, x, report):
            with torch.no_grad():
                parent.report_spatial_state.D[-1].weight.fill_(.02)
                parent.report_spatial_state.D[-1].bias.copy_(torch.tensor([.8, -.6, 1.2, -1.]))
            half, _ = from_residual(parent, 'fixed_half')
            gated, _ = from_residual(parent, 'state_gate')
            gate = gated.report_spatial_state.gate
            self.assertEqual(gate[0].in_features, parent.report_spatial_state.D[0].in_features)
            self.assertEqual(gate[0].out_features, 16)
            self.assertEqual(gate[-1].out_features, 1)
            self.assertEqual(float((gate[-1].weight.abs().sum()+gate[-1].bias.abs().sum()).detach()), 0.)
            self.assertGreater(float(gate[0].weight.detach().abs().sum()), 0.)
            with torch.inference_mode():
                a, b = result(half, x, report), result(gated, x, report)
            torch.testing.assert_close(a['prediction'], b['prediction'], atol=0, rtol=0)
            for name in ('d_strength', 'base_logit', 'report_logit_delta', 'raw_effective_logit_delta',
                         'effective_logit_delta', 'node_capacity', 'capacity', 'boundary'):
                torch.testing.assert_close(a['spatial'][name], b['spatial'][name], atol=0, rtol=0)
            diagnostics = CapacityDiagnostics()
            diagnostics.update(b['spatial'])
            self.assertEqual(len(diagnostics.result()['rows']), 40)

    def test_conversion_preserves_all_parent_parameters_buffers_rng_and_independence(self):
        parent, _ = from_canonical_b3(canonical())
        parent.eval()
        parent_state = parent.state_dict()
        before = tensor_state_hash(parent_state)
        for mode in ('fixed_half', 'state_gate'):
            with self.subTest(mode=mode):
                rng = torch.get_rng_state().clone()
                candidate, certificate = from_residual(parent, mode)
                torch.testing.assert_close(torch.get_rng_state(), rng, atol=0, rtol=0)
                self.assertEqual(before, tensor_state_hash(parent.state_dict()))
                self.assertEqual(certificate['parent_state_sha256'], before)
                self.assertEqual(certificate['retained_shared_state_sha256'], before)
                self.assertEqual(certificate['mode'], mode)
                self.assertEqual(certificate['initial_alpha'], .5)
                self.assertFalse(candidate.training)
                self.assertFalse(any(module.training for module in candidate.modules()))
                self.assertIsNot(parent.tiid_module, candidate.tiid_module)
                self.assertIsNot(parent.internal_spatial_graph, candidate.internal_spatial_graph)
                copied = candidate.state_dict()
                self.assertTrue(parent_state.keys() <= copied.keys())
                for name, value in parent_state.items():
                    torch.testing.assert_close(copied[name], value, atol=0, rtol=0)
                    if value.numel():
                        self.assertNotEqual(copied[name].data_ptr(), value.data_ptr(), name)
                count = sum(parameter.numel() for parameter in candidate.parameters())
                parent_count = sum(parameter.numel() for parameter in parent.parameters())
                self.assertEqual(certificate['parameter_count'], count)
                self.assertEqual(certificate['parent_parameter_count'], parent_count)
                self.assertTrue(all(parameter.requires_grad for parameter in candidate.parameters()))
                expected_added = ((parent.report_spatial_state.D[0].in_features+1)*16+17
                                  if mode == 'state_gate' else 0)
                self.assertEqual(certificate['added_parameters'], expected_added)
                with torch.no_grad():
                    candidate.report_spatial_state.P[0].weight.add_(1.)
                self.assertEqual(before, tensor_state_hash(parent.state_dict()))

    def test_zero_d_initialization_preserves_original_forecast_and_keeps_tiid_active(self):
        parent, _ = from_canonical_b3(canonical())
        parent.eval()
        x, report = inputs()
        with torch.inference_mode():
            reference = parent(x, incident_data=report)
            for mode in ('fixed_half', 'state_gate'):
                candidate, _ = from_residual(parent, mode)
                self.assertFalse(any(parameter.is_inference() for parameter in candidate.parameters()))
                actual = candidate(x, incident_data=report)
                off = candidate(x, incident_data=report, spatial_report_enabled=False)
                no_tiid = candidate(x, incident_data=report, spatial_report_enabled=False, tiid_report_enabled=False)
                torch.testing.assert_close(actual, reference, atol=0, rtol=0)
                torch.testing.assert_close(off, reference, atol=0, rtol=0)
                self.assertGreater(float((off-no_tiid).abs().max()), 0.)

    def test_off_skips_d_and_gate_preserves_p_boundary_and_original_tiid(self):
        with controlled_forecast() as (parent, x, report):
            with torch.no_grad():
                parent.report_spatial_state.D[-1].bias.fill_(1.)
            candidate, _ = from_residual(parent, 'state_gate')
            calls = []
            hook = candidate.tiid_module.register_forward_hook(lambda *args: calls.append(True))
            try:
                with torch.inference_mode():
                    on = result(candidate, x, report)
                    with (patch.object(candidate.report_spatial_state.D, 'forward', side_effect=AssertionError('D executed OFF')),
                          patch.object(candidate.report_spatial_state.gate, 'forward', side_effect=AssertionError('Gate executed OFF'))):
                        off = result(candidate, x, report, False)
            finally:
                hook.remove()
            self.assertEqual(len(calls), 2)
            self.assertIsNone(off['spatial']['d_strength'])
            self.assertIsNone(off['spatial']['report_logit_delta'])
            self.assertFalse(off['spatial']['report_path_executed'])
            self.assertEqual(float(off['spatial']['raw_effective_logit_delta'].abs().sum()), 0.)
            self.assertEqual(float(off['spatial']['effective_logit_delta'].abs().sum()), 0.)
            torch.testing.assert_close(off['spatial']['node_capacity'], off['spatial']['base_node_capacity'], atol=0, rtol=0)
            for name in ('base_logit', 'base_node_capacity', 'base_edge_capacity', 'boundary', 'report_mask'):
                torch.testing.assert_close(on['spatial'][name], off['spatial'][name], atol=0, rtol=0)
            self.assertGreater(float((on['prediction']-off['prediction']).abs().max()), 0.)

    def test_unsupported_nodes_remain_exactly_p_even_with_nonzero_d_and_gate_biases(self):
        with controlled_forecast() as (parent, x, report):
            with torch.no_grad():
                parent.report_spatial_state.D[-1].bias.fill_(3.)
            candidate, _ = from_residual(parent, 'state_gate')
            with torch.no_grad():
                candidate.report_spatial_state.gate[-1].bias.fill_(2.)
            with torch.inference_mode():
                context = result(candidate, x, report)['spatial']
            unsupported = ~context['report_mask'].bool().expand_as(context['base_logit'])
            torch.testing.assert_close(context['raw_effective_logit_delta'][unsupported],
                                       torch.zeros_like(context['base_logit'][unsupported]), atol=0, rtol=0)
            torch.testing.assert_close(context['node_capacity'][unsupported],
                                       context['base_node_capacity'][unsupported], atol=0, rtol=0)
            self.assertTrue(((context['d_strength'] >= 0.) & (context['d_strength'] <= 1.)).all())
            self.assertTrue(((context['node_capacity'] >= EPSILON) & (context['node_capacity'] <= 1.)).all())
            self.assertTrue((context['effective_logit_delta'].abs() <= context['raw_effective_logit_delta'].abs()).all())

    def test_gate_and_d_observe_identical_available_evidence_and_ignore_future_target_fields(self):
        with controlled_forecast() as (parent, x, report):
            candidate, _ = from_residual(parent, 'state_gate')
            d_inputs, gate_inputs = [], []
            handles = [candidate.report_spatial_state.D.register_forward_pre_hook(
                lambda module, args: d_inputs.append(args[0].detach().clone())),
                candidate.report_spatial_state.gate.register_forward_pre_hook(
                    lambda module, args: gate_inputs.append(args[0].detach().clone()))]
            try:
                with torch.inference_mode():
                    reference = result(candidate, x, report)
                    supplied = dict(report, y_flow=torch.full((2, 12, 4, 1), 1e9),
                                    future_traffic=torch.full_like(x, -1e9),
                                    incident_resolution_time=torch.full((2,), 1e9))
                    altered = result(candidate, x, supplied)
            finally:
                for handle in handles:
                    handle.remove()
            torch.testing.assert_close(d_inputs[0], gate_inputs[0], atol=0, rtol=0)
            torch.testing.assert_close(d_inputs[1], gate_inputs[1], atol=0, rtol=0)
            torch.testing.assert_close(d_inputs[0], d_inputs[1], atol=0, rtol=0)
            torch.testing.assert_close(reference['prediction'], altered['prediction'], atol=0, rtol=0)

    def test_real_forecast_mae_drives_gate_final_then_hidden_gradients_when_d_is_nonzero(self):
        with controlled_forecast() as (parent, x, report):
            with torch.no_grad():
                parent.report_spatial_state.D[-1].bias.fill_(1.)
            candidate, _ = from_residual(parent, 'state_gate')
            gate = candidate.report_spatial_state.gate
            optimizer = torch.optim.SGD(gate.parameters(), lr=.1, weight_decay=0.)
            gradients = []
            for step in range(2):
                candidate.zero_grad(set_to_none=True)
                prediction = candidate(x, incident_data=report)
                loss = masked_flow_mae(prediction, prediction.detach()+1., torch.ones_like(prediction, dtype=torch.bool))
                loss.backward()
                gradients.append((float(gate[0].weight.grad.abs().sum()+gate[0].bias.grad.abs().sum()),
                                  float(gate[-1].weight.grad.abs().sum()+gate[-1].bias.grad.abs().sum())))
                self.assertGreater(gradients[-1][1], 0.)
                self.assertTrue(all(torch.isfinite(parameter.grad).all() for parameter in gate.parameters()))
                optimizer.step()
            self.assertEqual(gradients[0][0], 0.)
            self.assertGreater(gradients[1][0], 0.)

    def test_zero_d_has_zero_gate_gradient_but_d_last_can_still_learn_from_forecast(self):
        with controlled_forecast() as (parent, x, report):
            candidate, _ = from_residual(parent, 'state_gate')
            prediction = candidate(x, incident_data=report)
            loss = masked_flow_mae(prediction, prediction.detach()+1., torch.ones_like(prediction, dtype=torch.bool))
            loss.backward()
            self.assertEqual(sum(float(parameter.grad.abs().sum()) for parameter in candidate.report_spatial_state.gate.parameters()), 0.)
            d = candidate.report_spatial_state.D[-1]
            self.assertGreater(float(d.weight.grad.abs().sum()+d.bias.grad.abs().sum()), 0.)

    def test_open_edge_mapping_and_boundary_heads_are_preserved(self):
        parent, _ = from_canonical_b3(canonical())
        parent.eval()
        with torch.no_grad():
            parent.report_spatial_state.D[-1].bias.copy_(torch.tensor([.8, -.6, 1.2, -1.]))
        candidate, _ = from_residual(parent, 'state_gate')
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

    def test_rejects_unknown_mode_canonical_b3_and_repeated_conversion(self):
        parent, _ = from_canonical_b3(canonical())
        with self.assertRaises(ValueError):
            from_residual(parent, 'unknown')
        with self.assertRaises(ValueError):
            from_residual(canonical(), 'fixed_half')
        gated, _ = from_residual(parent, 'state_gate')
        self.assertIsInstance(gated.report_spatial_state, ReportCapacityAttenuation)
        with self.assertRaises(ValueError):
            from_residual(gated, 'fixed_half')


if __name__ == '__main__':
    unittest.main()
