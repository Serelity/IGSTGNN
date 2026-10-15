"""Engineering contracts for the F2 interaction prototype; no quality claim."""
import unittest
from unittest.mock import patch

import torch

from src.models.incident_spatial_component_fusion import from_minimal
from src.models.incident_spatial_interaction_fusion import (
    InteractionFusionIGSTGNN, _center_added_scores, centered_bounded_delta,
    from_component_fusion,
)
from src.models.incident_spatial_residual import tensor_state_hash
from tests.test_min_component_fusion_model import activate
from tests.test_min_event_budget_model import event_inputs, minimal


def parent_f2(mode='event'):
    parent, _ = from_minimal(minimal(), mode)
    activate(parent)
    return parent


def activate_interaction(model):
    with torch.no_grad():
        generator = torch.Generator().manual_seed(101)
        for name in ('history_readout', 'report_readout'):
            weight = getattr(model.forecast_interaction, name).weight
            weight.copy_(.4*torch.randn(weight.shape, generator=generator))


class InteractionFusionModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_conversion_retains_independent_mature_F2_source_and_rng(self):
        parent = parent_f2().requires_grad_(False)
        before, rng = tensor_state_hash(parent.state_dict()), torch.get_rng_state().clone()
        with torch.inference_mode():
            model, certificate = from_component_fusion(parent)
        self.assertIsInstance(model, InteractionFusionIGSTGNN)
        self.assertEqual(before, tensor_state_hash(parent.state_dict()))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(certificate['retained_state_exact_verified'])
        self.assertTrue(certificate['all_parameters_trainable'])
        self.assertTrue(certificate['legacy_report_scoring_retained'])
        self.assertTrue(certificate['prototype'])
        self.assertEqual(certificate['added_parameters'],
                         sum(p.numel() for p in model.forecast_interaction.parameters()))
        self.assertTrue(all(name.startswith('forecast_interaction.') for name in certificate['added_state_keys']))
        self.assertTrue(all(not p.is_inference() for p in model.parameters()))
        self.assertTrue(all(not p.requires_grad for p in parent.parameters()))
        for name, value in parent.state_dict().items():
            self.assertTrue(torch.equal(value, model.state_dict()[name]), name)
            self.assertNotEqual(value.data_ptr(), model.state_dict()[name].data_ptr(), name)
        with torch.no_grad():
            model.embedding.weight.add_(1.)
        self.assertEqual(before, tensor_state_hash(parent.state_dict()))

    def test_zero_new_readouts_exact_active_parent_F2_and_legacy_switches(self):
        x, incident = event_inputs()
        for mode in ('state', 'event'):
            parent = parent_f2(mode)
            model, _ = from_component_fusion(parent)
            for switches in ({}, {'content_enabled': False}, {'semantic_enabled': False},
                             {'report_enabled': False}, {'uniform_weights': True},
                             {'spatial_report_enabled': False, 'tiid_report_enabled': False}):
                with self.subTest(mode=mode, switches=switches), torch.inference_mode():
                    expected = parent(x, incident_data=incident, return_spatial_details=True, **switches)
                    actual = model(x, incident_data=incident, return_spatial_details=True, **switches)
                self.assertTrue(torch.equal(actual['prediction'], expected['prediction']))
                self.assertTrue(torch.equal(actual['fusion']['weights'], expected['fusion']['weights']))
                self.assertEqual(len(actual['spatial']['records']), len(expected['spatial']['records']))
                for left, right in zip(actual['spatial']['records'], expected['spatial']['records']):
                    self.assertTrue(torch.equal(left, right))
                for key in ('capacity', 'boundary', 'base_logit', 'base_node_capacity'):
                    self.assertTrue(torch.equal(actual['spatial'][key], expected['spatial'][key]))

    def test_both_new_OFF_exact_parent_even_after_new_readouts_active(self):
        parent, (x, incident) = parent_f2(), event_inputs()
        model, _ = from_component_fusion(parent)
        activate_interaction(model)
        with torch.inference_mode():
            expected = parent(x, incident_data=incident, return_spatial_details=True)
            with patch.object(model.forecast_interaction, 'forward', side_effect=AssertionError('new OFF ran')):
                with patch.object(model.out_fc_1, 'forward', wraps=model.out_fc_1.forward) as head:
                    actual = model(x, incident_data=incident, history_interaction_enabled=False,
                                   report_interaction_enabled=False, return_spatial_details=True)
        self.assertEqual(head.call_count, 1)
        self.assertTrue(torch.equal(actual['prediction'], expected['prediction']))
        self.assertTrue(torch.equal(actual['fusion']['weights'], expected['fusion']['weights']))

    def test_fusion_OFF_exact_current_MIN_and_no_new_module_or_extra_head(self):
        parent, (x, incident) = parent_f2(), event_inputs()
        model, _ = from_component_fusion(parent)
        activate_interaction(model)
        with torch.inference_mode():
            expected = parent(x, fusion_enabled=False)
            with patch.object(model.forecast_interaction, 'forward', side_effect=AssertionError('fusion OFF ran')):
                with patch.object(model.out_fc_1, 'forward', wraps=model.out_fc_1.forward) as head:
                    actual = model(x, incident_data='not read', fusion_enabled=False, return_spatial_details=True)
        self.assertEqual(head.call_count, 1)
        self.assertTrue(torch.equal(expected, actual['prediction']))
        self.assertTrue(torch.equal(actual['fusion']['weights'], torch.ones(2, 12, 4, 10)))

    def test_single_backbone_one_extra_complete_head_and_one_bounded_transform(self):
        parent, (x, incident) = parent_f2(), event_inputs()
        model, _ = from_component_fusion(parent)
        calls = []
        original = model.forecast_interaction.forward

        def capture(*args, **kwargs):
            calls.append(kwargs)
            return original(*args, **kwargs)

        with torch.inference_mode():
            expected_fixed_sum = parent(x, fusion_enabled=False)
            with patch.object(model.embedding, 'forward', wraps=model.embedding.forward) as embedding:
                with patch.object(model.layers[0], 'forward', wraps=model.layers[0].forward) as layer:
                    with patch.object(model.out_fc_1, 'forward', wraps=model.out_fc_1.forward) as head:
                        with patch.object(model.forecast_fusion, 'forward', side_effect=AssertionError('F2 delta called')):
                            with patch.object(model.forecast_interaction, 'forward', side_effect=capture):
                                with patch('src.models.incident_spatial_interaction_fusion.centered_bounded_delta',
                                           wraps=centered_bounded_delta) as bound:
                                    model(x, incident_data=incident)
        self.assertEqual(embedding.call_count, 1)
        self.assertEqual(layer.call_count, 1)
        self.assertEqual(head.call_count, 2)
        self.assertEqual(bound.call_count, 1)
        self.assertEqual(len(calls), 1)
        self.assertTrue(torch.equal(calls[0]['historical_flow'], x[..., :1]))
        self.assertTrue(torch.equal(calls[0]['fixed_sum_prediction'], expected_fixed_sum))
        self.assertEqual(calls[0]['fixed_sum_prediction'].shape, (2, 12, 4, 1))
        self.assertEqual(calls[0]['component_encodings'].shape, (2, 4, 4, 10, 8))
        self.assertEqual(calls[0]['horizon_features'].shape, (12, 4))
        self.assertEqual(calls[0]['report_features'].shape, (2, 4, 19))

    def test_constant_new_scores_strict_identity_with_mature_nonuniform_F2_scores(self):
        parent, (x, incident) = parent_f2(), event_inputs()
        model, _ = from_component_fusion(parent)
        with torch.inference_mode():
            expected = parent(x, incident_data=incident, return_spatial_details=True)
            self.assertGreater(float((expected['fusion']['weights']-1).abs().max()), 0.)
            for first, second in ((.1, -.7), (5., 6.), (-.3, .2)):
                supplied = dict(history_scores=x.new_full((2, 12, 4, 10), first),
                                report_scores=x.new_full((2, 12, 4, 10), second), diagnostics=None)
                with patch.object(model.forecast_interaction, 'forward', return_value=supplied):
                    actual = model(x, incident_data=incident, return_spatial_details=True)
                self.assertTrue(torch.equal(actual['prediction'], expected['prediction']))
                self.assertTrue(torch.equal(actual['fusion']['weights'], expected['fusion']['weights']))
                for name in ('centered_history', 'centered_report'):
                    self.assertEqual(float(actual['fusion']['statistics'][name+'_interaction_score_abs_max']), 0.)
                self.assertTrue(torch.equal(_center_added_scores(supplied['history_scores']),
                                            torch.zeros_like(supplied['history_scores'])))

    def test_report_increment_zero_outside_support_and_new_OFF_retains_legacy_report(self):
        parent, (x, incident) = parent_f2(), event_inputs()
        model, _ = from_component_fusion(parent)
        activate_interaction(model)
        incident['relation'].zero_()
        incident['relation'][:, 0, 1] = .7
        captured = []
        handle = model.forecast_interaction.register_forward_hook(
            lambda module, args, output: captured.append(output['report_scores'].detach()))
        try:
            with torch.inference_mode():
                on = model(x, incident_data=incident, return_spatial_details=True)
                off = model(x, incident_data=incident, report_interaction_enabled=False, return_spatial_details=True)
                all_new_off = model(x, incident_data=incident, history_interaction_enabled=False,
                                    report_interaction_enabled=False)
                legacy = parent(x, incident_data=incident)
                legacy_report_off = parent(x, incident_data='not read', report_enabled=False)
        finally:
            handle.remove()
        self.assertTrue(torch.equal(captured[0][:, :, 1:], torch.zeros_like(captured[0][:, :, 1:])))
        self.assertGreater(float(captured[0][:, :, 0].abs().max()), 0.)
        self.assertTrue(torch.equal(on['prediction'][:, :, 1:], off['prediction'][:, :, 1:]))
        self.assertTrue(torch.equal(on['fusion']['weights'][:, :, 1:], off['fusion']['weights'][:, :, 1:]))
        self.assertGreater(float((on['fusion']['weights'][:, :, 0]-off['fusion']['weights'][:, :, 0]).abs().max()), 0.)
        self.assertTrue(torch.equal(all_new_off, legacy))
        self.assertGreater(float((legacy-legacy_report_off).abs().max()), 0.)
        self.assertTrue(off['fusion']['event_context_used'])
        self.assertFalse(off['fusion']['interaction']['report_enabled'])

    def test_report_OFF_disables_all_reports_and_STATE_never_reads_incident(self):
        x, incident = event_inputs()
        for mode in ('state', 'event'):
            model, _ = from_component_fusion(parent_f2(mode))
            activate_interaction(model)
            kwargs = {} if mode == 'state' else {'report_enabled': False}
            with torch.inference_mode():
                with patch.object(model.forecast_fusion.semantic_encoder, 'forward',
                                  side_effect=AssertionError('disabled semantics read')):
                    expected = model(x, incident_data=None, **kwargs)
                    actual = model(x, incident_data='not read', return_spatial_details=True, **kwargs)
            self.assertTrue(torch.equal(expected, actual['prediction']))
            self.assertFalse(actual['fusion']['event_context_used'])
            self.assertFalse(actual['fusion']['interaction']['report_enabled'])
            self.assertEqual(float(actual['fusion']['statistics']['report_interaction_score_abs_max']), 0.)

    def test_first_update_opens_differentiable_fixedsum_content_and_backbone_paths(self):
        model, _ = from_component_fusion(parent_f2())
        x, incident = event_inputs()
        optimizer = torch.optim.SGD(model.parameters(), lr=.02)
        prediction = model(x, incident_data=incident)
        (prediction-prediction.detach()-1).abs().mean().backward()
        interaction = model.forecast_interaction
        for name in ('history_readout', 'report_readout'):
            gradient = getattr(interaction, name).weight.grad
            self.assertIsNotNone(gradient)
            self.assertGreater(float(gradient.abs().sum()), 0.)
        optimizer.step()
        model.zero_grad(set_to_none=True)
        captured = []
        original = interaction.forward

        def capture(*args, **kwargs):
            for name in ('fixed_sum_prediction', 'component_encodings'):
                self.assertTrue(kwargs[name].requires_grad)
                kwargs[name].retain_grad()
            captured.append(kwargs)
            return original(*args, **kwargs)

        with patch.object(interaction, 'forward', side_effect=capture):
            prediction = model(x, incident_data=incident)
        (prediction-prediction.detach()-1).abs().mean().backward()
        for name in ('fixed_sum_prediction', 'component_encodings'):
            gradient = captured[0][name].grad
            self.assertIsNotNone(gradient, name)
            self.assertGreater(float(gradient.abs().sum()), 0., name)
        for parameter in (model.embedding.weight, model.out_fc_2.weight,
                          interaction.history_component.weight, interaction.report_projection.weight):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()))
            self.assertGreater(float(parameter.grad.abs().sum()), 0.)

    def test_train_mode_rng_and_retained_BN_state_match_active_parent_at_zero_start(self):
        parent, (x, incident) = parent_f2().train(), event_inputs()
        model, _ = from_component_fusion(parent)
        torch.manual_seed(81)
        expected = parent(x, incident_data=incident)
        expected_rng = torch.get_rng_state().clone()
        torch.manual_seed(81)
        actual = model(x, incident_data=incident)
        self.assertTrue(torch.equal(expected, actual))
        self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
        for name, value in parent.state_dict().items():
            self.assertTrue(torch.equal(value, model.state_dict()[name]), name)

    def test_diagnostics_serialization_ignored_labels_and_contracts(self):
        parent, (x, incident) = parent_f2(), event_inputs()
        model, _ = from_component_fusion(parent)
        activate_interaction(model)
        before = tensor_state_hash(model.state_dict())
        with torch.inference_mode():
            actual = model(x, label='never read', sensor_data='never read', incident_data=incident,
                           return_spatial_details=True)
            self.assertTrue(torch.equal(actual['prediction'], model(x, incident_data=incident)))
        self.assertEqual(before, tensor_state_hash(model.state_dict()))
        fusion = actual['fusion']
        self.assertTrue(fusion['prototype'])
        self.assertTrue(bool(((fusion['weights'] >= .9) & (fusion['weights'] <= 1.1)).all()))
        self.assertTrue(all(value.ndim == 0 and not value.requires_grad for value in fusion['statistics'].values()))
        for name in ('history', 'report', 'centered_history', 'centered_report'):
            value = fusion['interaction'][name+'_score_abs_mean_by_horizon_component']
            self.assertEqual(value.shape, (12, 10))
            self.assertFalse(value.requires_grad)
        fresh, _ = from_component_fusion(parent)
        fresh.load_state_dict(model.state_dict(), strict=True)
        with torch.inference_mode():
            self.assertTrue(torch.equal(fresh(x, incident_data=incident), actual['prediction']))
        for name in ('history_interaction_enabled', 'report_interaction_enabled', 'report_enabled',
                     'content_enabled', 'return_spatial_details', 'uniform_weights'):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'bool'):
                model(x, **{name: 1})
        with self.assertRaisesRegex(ValueError, 'external'):
            model(x, forecast_delta=torch.zeros(1))
        for unsupported in (minimal(), model, object()):
            with self.subTest(parent=type(unsupported)), self.assertRaisesRegex(ValueError, 'ComponentFusionIGSTGNN'):
                from_component_fusion(unsupported)


if __name__ == '__main__':
    unittest.main()
