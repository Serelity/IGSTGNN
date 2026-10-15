"""Single-backbone, exact identity and real-horizon forecast fusion checks."""
import copy
import unittest
from unittest.mock import patch

import torch

from src.models.incident_spatial_forecast_fusion import (
    COMPONENT_ORDER, MODES, ForecastFusionIGSTGNN, from_minimal, fuse_components,
)
from src.models.incident_spatial_residual import tensor_state_hash
from tests.test_min_event_budget_model import event_inputs, minimal


def activate(model):
    with torch.no_grad():
        scorer = model.forecast_fusion.condition
        generator = torch.Generator().manual_seed(55)
        scorer[0].weight.copy_(.04*torch.randn(scorer[0].weight.shape, generator=generator))
        scorer[0].bias.fill_(.3)
        scorer[-1].weight.copy_(.07*torch.randn(scorer[-1].weight.shape, generator=generator))
        scorer[-1].bias.copy_(torch.linspace(-.3, .3, 10))


class ForecastFusionModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_retained_independent_state_identical_mode_initialization_and_rng(self):
        parent = minimal().requires_grad_(False)
        before, rng = tensor_state_hash(parent.state_dict()), torch.get_rng_state().clone()
        candidates = [from_minimal(parent, mode) for mode in MODES]
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(before, tensor_state_hash(parent.state_dict()))
        self.assertEqual(candidates[0][1]['fusion_initial_sha256'], candidates[1][1]['fusion_initial_sha256'])
        for model, certificate in candidates:
            self.assertIsInstance(model, ForecastFusionIGSTGNN)
            self.assertFalse(hasattr(model, 'core'))
            self.assertTrue(certificate['all_parameters_trainable'])
            self.assertEqual(certificate['component_order'], list(COMPONENT_ORDER))
            self.assertEqual(certificate['added_parameters'], sum(p.numel() for p in model.forecast_fusion.parameters()))
            for name, value in parent.state_dict().items():
                self.assertTrue(torch.equal(value, model.state_dict()[name]))
                self.assertNotEqual(value.data_ptr(), model.state_dict()[name].data_ptr())
            self.assertTrue(all(not value.is_inference() for value in model.parameters()))
            self.assertTrue(all(name.startswith('forecast_fusion.') for name in certificate['added_state_keys']))

    def test_zero_initialization_exact_MIN_once_and_all_original_spatial_records_preserved(self):
        parent, (x, incident) = minimal(), event_inputs()
        with torch.inference_mode():
            baseline = parent(x, return_spatial_details=True)
            for mode in MODES:
                model, _ = from_minimal(parent, mode)
                with patch.object(model.embedding, 'forward', wraps=model.embedding.forward) as embed:
                    with patch.object(model._graph_constructor.__self__, '_graph_constructor', wraps=model._graph_constructor) as graph:
                        with patch.object(model.layers[0], 'forward', wraps=model.layers[0].forward) as layer:
                            actual = model(x, incident_data=incident, return_spatial_details=True)
                self.assertTrue(torch.equal(actual['prediction'], baseline['prediction']))
                self.assertEqual(embed.call_count, 1)
                self.assertEqual(graph.call_count, 1)
                self.assertEqual(layer.call_count, 1)
                self.assertEqual(actual['prediction'].shape, (2, 12, 4, 1))
                self.assertTrue(torch.equal(actual['fusion']['weights'], torch.ones(2, 12, 4, 10)))
                self.assertEqual(len(actual['spatial']['records']), 20)
                for left, right in zip(actual['spatial']['records'], baseline['spatial']['records']):
                    self.assertTrue(torch.equal(left, right))
                for key in ('capacity', 'boundary', 'base_logit', 'base_node_capacity'):
                    self.assertTrue(torch.equal(actual['spatial'][key], baseline['spatial'][key]))

    def test_gap_three_expansion_component_order_native_head_channels_and_gradient(self):
        parent, (x, incident) = minimal(), event_inputs()
        model, _ = from_minimal(parent, 'event')
        activate(model)
        dif, inh = [], []
        handles = []
        for layer in model.layers:
            def capture(module, args, output):
                dif.append(output[1])
                inh.append(output[2])
            handles.append(layer.register_forward_hook(capture))
        try:
            with torch.inference_mode():
                actual = model(x, incident_data=incident, return_spatial_details=True)
        finally:
            for handle in handles:
                handle.remove()
        delta = actual['fusion']['weights']-1
        # Independently explicit expansion and original head/channel selection.
        expected_steps = (sum(dif)+sum(inh)).repeat_interleave(3, dim=1)
        adjustment = torch.zeros_like(expected_steps)
        for index, component in enumerate((*dif, *inh)):
            adjustment += delta[..., index:index+1]*component.repeat_interleave(3, dim=1)
        expected_head = model.out_fc_2(torch.relu(model.out_fc_1(torch.relu(expected_steps+adjustment))))
        expected = torch.stack([expected_head[:, h, :, h % 3] for h in range(12)], 1)[..., None]
        # weights=1+delta is rounded separately; allow that diagnostic roundoff.
        torch.testing.assert_close(actual['prediction'], expected, atol=3e-7, rtol=1e-6)
        self.assertGreater(float((actual['prediction'][:, 0]-actual['prediction'][:, 3]).abs().max()), 0.)
        # A scalar hand calculation independently fixes [block,within-block] order.
        components = [torch.arange(4, dtype=torch.double).reshape(1, 4, 1, 1)+k for k in range(10)]
        weights = torch.arange(120, dtype=torch.double).reshape(1, 12, 1, 10)/1200
        weights.requires_grad_(True)
        value, _ = fuse_components(components[:5], components[5:], weights, horizon=12, gap=3)
        hand = torch.tensor([sum((h//3+k)*(1+h*10/1200+k/1200) for k in range(10))
                             for h in range(12)], dtype=torch.double).reshape(1, 12, 1, 1)
        torch.testing.assert_close(value, hand, atol=1e-13, rtol=1e-13)
        value.sum().backward()
        for h in range(12):
            torch.testing.assert_close(weights.grad[0, h, 0], torch.arange(10, dtype=torch.double)+h//3)

    def test_zero_initialization_training_dropout_BN_and_caller_rng_match_native_MIN(self):
        parent, (x, incident) = minimal().train(), event_inputs()
        model, _ = from_minimal(parent, 'event')
        torch.manual_seed(71)
        expected = parent(x)
        expected_rng = torch.get_rng_state().clone()
        torch.manual_seed(71)
        actual = model(x, incident_data=incident)
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
        for name, value in parent.state_dict().items():
            with self.subTest(state=name):
                self.assertTrue(torch.equal(value, model.state_dict()[name]))

    def test_state_reads_no_incident_and_EVENT_support_only_changes_supported_forecasts(self):
        parent, (x, incident) = minimal(), event_inputs()
        state, _ = from_minimal(parent, 'state')
        event, _ = from_minimal(parent, 'event')
        activate(state)
        event.load_state_dict(state.state_dict())
        incident['relation'].zero_()
        incident['relation'][:, 0, 1] = .7
        with torch.inference_mode():
            a = state(x, incident_data='never read', return_spatial_details=True)
            b = state(x, incident_data=incident)
            off = event(x, report_enabled=False, incident_data='never read', return_spatial_details=True)
            on = event(x, incident_data=incident, return_spatial_details=True)
        self.assertTrue(torch.equal(a['prediction'], b))
        self.assertTrue(torch.equal(a['prediction'], off['prediction']))
        self.assertFalse(bool(a['fusion']['support'].any()))
        self.assertTrue(torch.equal(on['fusion']['weights'][:, :, 1:], off['fusion']['weights'][:, :, 1:]))
        self.assertTrue(torch.equal(on['prediction'][:, :, 1:], off['prediction'][:, :, 1:]))
        self.assertGreater(float((on['prediction'][:, :, 0]-off['prediction'][:, :, 0]).abs().max()), 0.)
        for key in ('capacity', 'boundary', 'base_logit', 'base_node_capacity'):
            self.assertTrue(torch.equal(on['spatial'][key], off['spatial'][key]))

    def test_missing_report_or_support_matches_current_state_fusion_not_old_MIN(self):
        parent, (x, report) = minimal(), event_inputs()
        model, _ = from_minimal(parent, 'event')
        activate(model)
        with torch.inference_mode():
            original = parent(x)
            expected = model(x, report_enabled=False)
            self.assertGreater(float((expected-original).abs().max()), 0.)
        variants = (None, {**report, 'report_present': torch.zeros(2, dtype=torch.bool)},
                    {**report, 'relation_available': torch.zeros(2, dtype=torch.bool)},
                    {**report, 'relation': torch.zeros_like(report['relation'])})
        with patch.object(model.forecast_fusion.semantic_encoder, 'forward', side_effect=AssertionError('null semantics encoded')):
            with torch.inference_mode():
                for incident in variants:
                    result = model(x, incident_data=incident, return_spatial_details=True)
                    self.assertTrue(torch.equal(result['prediction'], expected))
                    self.assertFalse(result['fusion']['event_context_used'])
                    self.assertFalse(bool(result['fusion']['support'].any()))

    def test_uniform_weights_exact_same_checkpoint_average_and_fusion_OFF_restores_MIN(self):
        parent, (x, incident) = minimal(), event_inputs()
        model, _ = from_minimal(parent, 'event')
        activate(model)
        state_before = tensor_state_hash(model.state_dict())
        with torch.inference_mode():
            on = model(x, incident_data=incident, return_spatial_details=True)
            uniform = model(x, incident_data=incident, uniform_weights=True, return_spatial_details=True)
            off = model(x, incident_data='ignored', fusion_enabled=False, return_spatial_details=True)
            final = model(x, incident_data=incident)
        torch.testing.assert_close(uniform['fusion']['weights'], on['fusion']['weights'].mean(-1, keepdim=True).expand_as(on['fusion']['weights']), atol=1e-7, rtol=1e-7)
        self.assertGreater(float((uniform['prediction']-on['prediction']).abs().max()), 0.)
        self.assertTrue(torch.equal(final, on['prediction']))
        with torch.inference_mode():
            self.assertTrue(torch.equal(off['prediction'], parent(x)))
        self.assertEqual(float(off['fusion']['statistics']['fusion_calls']), 0.)
        self.assertEqual(float(off['fusion']['weighted_hidden_change_abs_max']), 0.)
        self.assertEqual(tensor_state_hash(model.state_dict()), state_before)

    def test_initial_last_projection_gradient_then_state_horizon_semantic_backbone_gradients(self):
        model, _ = from_minimal(minimal(), 'event')
        x, incident = event_inputs()
        with torch.no_grad():
            model.forecast_fusion.condition[0].bias.fill_(.8)
        output = model(x, incident_data=incident)
        (output-output.detach()-1).abs().mean().backward()
        condition = model.forecast_fusion.condition
        self.assertGreater(float(condition[-1].weight.grad.abs().sum()), 0.)
        self.assertEqual(float(condition[0].weight.grad.abs().sum()), 0.)
        self.assertGreater(float(model.embedding.weight.grad.abs().sum()), 0.)
        activate(model)
        model.zero_grad(set_to_none=True)
        output = model(x, incident_data=incident)
        output.square().mean().backward()
        for name, parameter in (
            ('state', condition[0].weight), ('horizon', model.forecast_fusion.horizon_embedding.weight),
            ('semantic_type', model.forecast_fusion.semantic_encoder.type_embedding.weight),
            ('semantic_description', model.forecast_fusion.semantic_encoder.description_embedding.weight),
            ('head', model.out_fc_2.weight), ('embedding', model.embedding.weight)):
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertGreater(float(parameter.grad.abs().sum()), 0.)
                self.assertTrue(bool(torch.isfinite(parameter.grad).all()))
        state, _ = from_minimal(minimal(), 'state')
        activate(state)
        state(x, incident_data='ignored').square().mean().backward()
        self.assertGreater(float(state.forecast_fusion.condition[0].weight.grad.abs().sum()), 0.)
        self.assertTrue(all(p.grad is None for p in state.forecast_fusion.semantic_encoder.parameters()))

    def test_report_null_semantics_and_explicit_UNK_are_distinct_supported_contexts(self):
        model, _ = from_minimal(minimal(), 'event')
        activate(model)
        x, incident = event_inputs()
        invalid = {**incident, 'type_valid': torch.zeros(2, dtype=torch.bool),
                   'description_valid': torch.zeros(2, dtype=torch.bool)}
        with torch.inference_mode():
            null = model(x, incident_data={**incident, 'type': 'ignored'}, semantic_enabled=False,
                         return_spatial_details=True)
            unknown = model(x, incident_data=invalid, return_spatial_details=True)
        self.assertFalse(null['fusion']['semantic_context_used'])
        self.assertTrue(unknown['fusion']['semantic_context_used'])
        self.assertGreater(float((null['prediction']-unknown['prediction']).abs().max()), 0.)
        self.assertTrue(torch.equal(null['fusion']['support'], unknown['fusion']['support']))

    def test_unavailable_row_semantic_values_cannot_affect_active_row_or_raise_category_error(self):
        model, _ = from_minimal(minimal(), 'event')
        activate(model)
        x, incident = event_inputs()
        incident['report_present'][1] = False
        changed = copy.deepcopy(incident)
        changed['type'][1] = 999
        changed['description'][1] = -999
        with torch.inference_mode():
            expected = model(x, incident_data=incident)
            actual = model(x, incident_data=changed)
        self.assertTrue(torch.equal(actual, expected))

    def test_bounds_diagnostics_validation_and_serialization(self):
        parent, (x, incident) = minimal(), event_inputs()
        model, _ = from_minimal(parent, 'event')
        activate(model)
        with torch.inference_mode():
            result = model(x, incident_data=incident, return_spatial_details=True)
        fusion = result['fusion']
        self.assertTrue(bool(((fusion['weights'] >= .9) & (fusion['weights'] <= 1.1)).all()))
        for field in ('weight_delta_by_horizon_component', 'weight_delta_abs_mean_by_horizon_component',
                      'component_increment_abs_mean_by_horizon_component'):
            self.assertEqual(fusion[field].shape, (12, 10))
            self.assertFalse(fusion[field].requires_grad)
        self.assertEqual(fusion['weighted_hidden_change_abs_mean_by_horizon'].shape, (12,))
        self.assertTrue(all(value.ndim == 0 and not value.requires_grad for value in fusion['statistics'].values()))
        for kwargs in ({'mode': 'bad'}, {'width': 0}, {'width': True}, {'max_delta': 0},
                       {'max_delta': float('nan')}, {'max_delta': True}, {'type_count': 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                from_minimal(parent, **kwargs)
        with self.assertRaisesRegex(ValueError, 'unadapted'):
            from_minimal(model)
        with self.assertRaisesRegex(ValueError, 'bool'):
            model(x, fusion_enabled=1)
        for replacement, message in (({'relation': incident['relation'].double()}, 'Float32'),
                                     ({'report_present': torch.ones(2)}, 'Bool'),
                                     ({'type': torch.tensor([6, 0])}, 'category')):
            with self.subTest(replacement=replacement), self.assertRaisesRegex(ValueError, message):
                model(x, incident_data={**incident, **replacement})
        fresh, _ = from_minimal(parent, 'event')
        fresh.load_state_dict(model.state_dict(), strict=True)
        with torch.inference_mode():
            self.assertTrue(torch.equal(fresh(x, incident_data=incident), result['prediction']))


if __name__ == '__main__':
    unittest.main()
