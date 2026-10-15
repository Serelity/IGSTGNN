"""Component-aware relative fusion contracts; no predictive-quality claims."""
import copy
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from src.models.incident_spatial_component_fusion import (
    COMPONENT_ORDER, MODES, ComponentFusion, ComponentFusionIGSTGNN,
    centered_bounded_delta, from_minimal,
)
from src.models.incident_spatial_residual import tensor_state_hash
from tests.test_min_event_budget_model import event_inputs, minimal


def activate(model):
    with torch.no_grad():
        generator = torch.Generator().manual_seed(55)
        scorer = model.forecast_fusion.condition
        scorer[0].weight.copy_(.15*torch.randn(scorer[0].weight.shape, generator=generator))
        scorer[0].bias.fill_(.3)
        scorer[-1].weight.copy_(.3*torch.randn(scorer[-1].weight.shape, generator=generator))
        scorer[-1].bias.fill_(.07)


class ComponentFusionModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_independent_retained_MIN_state_same_modes_initialization_and_rng(self):
        parent = minimal().requires_grad_(False)
        before, rng = tensor_state_hash(parent.state_dict()), torch.get_rng_state().clone()
        candidates = [from_minimal(parent, mode) for mode in MODES]
        self.assertEqual(before, tensor_state_hash(parent.state_dict()))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(candidates[0][1]['fusion_initial_sha256'], candidates[1][1]['fusion_initial_sha256'])
        for model, certificate in candidates:
            self.assertIsInstance(model, ComponentFusionIGSTGNN)
            self.assertTrue(certificate['all_parameters_trainable'])
            self.assertFalse(certificate['full_hidden_component_stack'])
            self.assertFalse(certificate['component_gradients_detached'])
            self.assertEqual(certificate['added_parameters'], sum(p.numel() for p in model.forecast_fusion.parameters()))
            self.assertTrue(all(name.startswith('forecast_fusion.') for name in certificate['added_state_keys']))
            self.assertTrue(all(not value.is_inference() for value in model.parameters()))
            for name, value in parent.state_dict().items():
                self.assertTrue(torch.equal(value, model.state_dict()[name]))
                self.assertNotEqual(value.data_ptr(), model.state_dict()[name].data_ptr())

    def test_zero_initialization_exact_MIN_single_backbone_and_backcast_records(self):
        parent, (x, incident) = minimal(), event_inputs()
        with torch.inference_mode():
            expected = parent(x, return_spatial_details=True)
            for mode in MODES:
                model, _ = from_minimal(parent, mode)
                with patch.object(model.embedding, 'forward', wraps=model.embedding.forward) as embedding:
                    with patch.object(model.layers[0], 'forward', wraps=model.layers[0].forward) as layer:
                        actual = model(x, incident_data=incident, return_spatial_details=True)
                self.assertEqual(embedding.call_count, 1)
                self.assertEqual(layer.call_count, 1)
                self.assertTrue(torch.equal(actual['prediction'], expected['prediction']))
                self.assertTrue(torch.equal(actual['fusion']['weights'], torch.ones(2, 12, 4, 10)))
                self.assertEqual(len(actual['spatial']['records']), 20)
                for left, right in zip(actual['spatial']['records'], expected['spatial']['records']):
                    self.assertTrue(torch.equal(left, right))
                for key in ('capacity', 'boundary', 'base_logit', 'base_node_capacity'):
                    self.assertTrue(torch.equal(actual['spatial'][key], expected['spatial'][key]))

    def test_zero_start_train_mode_rng_dropout_and_BN_match_MIN(self):
        parent, (x, incident) = minimal().train(), event_inputs()
        model, _ = from_minimal(parent, 'event')
        torch.manual_seed(71)
        expected = parent(x)
        expected_rng = torch.get_rng_state().clone()
        torch.manual_seed(71)
        actual = model(x, incident_data=incident)
        self.assertTrue(torch.equal(expected, actual))
        self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
        for name, value in parent.state_dict().items():
            self.assertTrue(torch.equal(value, model.state_dict()[name]), name)

    def test_active_router_reads_components_at_identical_history_and_content_OFF_erases_only_content(self):
        model, _ = from_minimal(minimal(), 'event')
        activate(model)
        router = model.forecast_fusion
        generator = torch.Generator().manual_seed(47)
        raw = torch.randn(2, 12, 4, model._hidden_dim, generator=generator)
        values = [torch.randn(2, 4, 4, model.out_fc_1.in_features, generator=generator) for _ in range(10)]
        changed = list(values)
        changed[3] = 4*values[3]+3
        relation, semantic = torch.randn(2, 4, 3, generator=generator), torch.randn(2, 4, 16, generator=generator)
        with torch.inference_mode():
            on, _ = router(raw, relation, semantic, values[:5], values[5:])
            alternative, _ = router(raw, relation, semantic, changed[:5], changed[5:])
            off, details = router(raw, relation, semantic, values[:5], values[5:], content_enabled=False, collect=True)
            other_off, _ = router(raw, relation, semantic, changed[:5], changed[5:], content_enabled=False)
        self.assertGreater(float((on-alternative).abs().max()), 1e-5)
        self.assertTrue(torch.equal(off, other_off))
        self.assertGreater(float((on-off).abs().max()), 1e-5)
        self.assertEqual(float(details['router_statistics']['component_encoding_abs_sum']), 0.)
        self.assertEqual(float(details['router_statistics']['component_log_rms_sum']), 0.)
        self.assertGreater(float(off.abs().max()), 0.)  # IDs/history/report remain active.

    def test_centered_bound_extreme_logits_preserves_relative_direction_and_zero_start(self):
        scores = torch.tensor([[-100., 100., 100., 100., 100., 100., 100., 100., 100., 100.],
                               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]], dtype=torch.double,
                              requires_grad=True)
        delta, raw, common, scale = centered_bounded_delta(scores, .1)
        self.assertLessEqual(float(delta.detach().abs().max()), .1)
        torch.testing.assert_close(delta.sum(-1), torch.zeros(2, dtype=torch.double), atol=1e-15, rtol=0)
        self.assertGreater(float(scale[0].detach()), 1.)
        self.assertEqual(float(scale[1].detach()), 1.)
        torch.testing.assert_close(delta*scale, raw-common, atol=1e-15, rtol=0)
        self.assertTrue(torch.equal(delta[1], torch.zeros(10, dtype=torch.double)))
        (delta*torch.arange(10, dtype=torch.double)).sum().backward()
        self.assertTrue(bool(torch.isfinite(scores.grad).all()))
        self.assertGreater(float(scores.grad[1].abs().sum()), 0.)

    def test_shared_split_affine_matches_independent_full_feature_scorer(self):
        model, _ = from_minimal(minimal(), 'event')
        activate(model)
        router = model.forecast_fusion
        generator = torch.Generator().manual_seed(33)
        raw = torch.randn(2, 12, 4, model._hidden_dim, generator=generator)
        components = [torch.randn(2, 4, 4, model.out_fc_1.in_features, generator=generator) for _ in range(10)]
        relation, semantic = torch.randn(2, 4, 3, generator=generator), torch.randn(2, 4, 16, generator=generator)
        with torch.inference_mode():
            actual, _ = router(raw, relation, semantic, components[:5], components[5:])
            encodings = [F.silu(router.component_projection(router.component_norm(value))) for value in components]
            mean = sum(encodings)/10
            summary = torch.cat((raw[:, -1], raw.mean(1), raw[:, -1]-raw[:, 0]), -1)
            state = F.silu(router.state_projection(router.state_norm(summary)))[:, None].expand(-1, 12, -1, -1)
            horizon = router.horizon_embedding.weight[None, :, None].expand(2, -1, 4, -1)
            scores = []
            for i, value in enumerate(components):
                rms = torch.log1p(torch.linalg.vector_norm(value, dim=-1)/(model.out_fc_1.in_features**.5))[..., None]
                features = torch.cat((encodings[i].repeat_interleave(3, 1),
                    (encodings[i]-mean).repeat_interleave(3, 1), rms.repeat_interleave(3, 1), state, horizon,
                    relation[:, None].expand(-1, 12, -1, -1), semantic[:, None].expand(-1, 12, -1, -1),
                    router.component_embedding.weight[i].view(1, 1, 1, -1).expand(2, 12, 4, -1)), -1)
                scores.append(router.condition(features))
            expected = centered_bounded_delta(torch.cat(scores, -1), .1)[0]
        torch.testing.assert_close(actual, expected, atol=2e-8, rtol=2e-6)

    def test_joint_gradients_router_and_original_backbone_without_detach(self):
        model, _ = from_minimal(minimal(), 'event')
        x, incident = event_inputs()
        model(x, incident_data=incident).square().mean().backward()
        self.assertGreater(float(model.forecast_fusion.condition[-1].weight.grad.abs().sum()), 0.)
        self.assertEqual(float(model.forecast_fusion.component_projection.weight.grad.abs().sum()), 0.)
        self.assertGreater(float(model.embedding.weight.grad.abs().sum()), 0.)
        activate(model)
        model.zero_grad(set_to_none=True)
        model(x, incident_data=incident).square().mean().backward()
        router = model.forecast_fusion
        for name, parameter in (
            ('content', router.component_projection.weight), ('content_norm', router.component_norm.weight),
            ('history', router.state_projection.weight), ('horizon', router.horizon_embedding.weight),
            ('id', router.component_embedding.weight), ('scorer', router.condition[0].weight),
            ('type', router.semantic_encoder.type_embedding.weight),
            ('description', router.semantic_encoder.description_embedding.weight),
            ('head', model.out_fc_2.weight), ('embedding', model.embedding.weight)):
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(bool(torch.isfinite(parameter.grad).all()))
                self.assertGreater(float(parameter.grad.abs().sum()), 0.)
        # Direct input gradients certify the content reader is differentiable,
        # rather than merely ordinary gradients through fixed component sums.
        raw = torch.randn(2, 12, 4, model._hidden_dim)
        components = [torch.randn(2, 4, 4, model.out_fc_1.in_features, requires_grad=True) for _ in range(10)]
        delta, _ = router(raw, torch.zeros(2, 4, 3), torch.zeros(2, 4, 16), components[:5], components[5:])
        (delta*torch.arange(10)).sum().backward()
        self.assertTrue(all(value.grad is not None and float(value.grad.abs().sum()) > 0 for value in components))

    def test_report_OFF_exact_outside_support_state_ignores_report_and_backcasts_unchanged(self):
        parent, (x, incident) = minimal(), event_inputs()
        state, _ = from_minimal(parent, 'state')
        event, _ = from_minimal(parent, 'event')
        activate(state)
        event.load_state_dict(state.state_dict())
        incident['relation'].zero_()
        incident['relation'][:, 0, 1] = .7
        with torch.inference_mode():
            a = state(x, incident_data='not read')
            b = event(x, report_enabled=False, incident_data='not read', return_spatial_details=True)
            c = event(x, incident_data=incident, return_spatial_details=True)
        self.assertTrue(torch.equal(a, b['prediction']))
        self.assertTrue(torch.equal(b['prediction'][:, :, 1:], c['prediction'][:, :, 1:]))
        self.assertTrue(torch.equal(b['fusion']['weights'][:, :, 1:], c['fusion']['weights'][:, :, 1:]))
        self.assertGreater(float((b['prediction'][:, :, 0]-c['prediction'][:, :, 0]).abs().max()), 0.)
        for left, right in zip(b['spatial']['records'], c['spatial']['records']):
            self.assertTrue(torch.equal(left, right))
        for key in ('capacity', 'boundary', 'base_logit', 'base_node_capacity'):
            self.assertTrue(torch.equal(b['spatial'][key], c['spatial'][key]))

    def test_OFF_content_semantics_switches_restore_exact_ON_and_state_is_preserved(self):
        parent, (x, incident) = minimal(), event_inputs()
        model, _ = from_minimal(parent, 'event')
        activate(model)
        before = tensor_state_hash(model.state_dict())
        with torch.inference_mode():
            on = model(x, incident_data=incident, return_spatial_details=True)
            content_off = model(x, incident_data=incident, content_enabled=False, return_spatial_details=True)
            semantic_off = model(x, incident_data=incident, semantic_enabled=False, return_spatial_details=True)
            off = model(x, incident_data='not read', fusion_enabled=False, return_spatial_details=True)
            final = model(x, incident_data=incident)
            expected = parent(x)
        self.assertTrue(torch.equal(off['prediction'], expected))
        self.assertTrue(torch.equal(final, on['prediction']))
        self.assertGreater(float((content_off['prediction']-on['prediction']).abs().max()), 0.)
        self.assertGreater(float((semantic_off['prediction']-on['prediction']).abs().max()), 0.)
        self.assertFalse(content_off['fusion']['component_content_enabled'])
        self.assertTrue(content_off['fusion']['event_context_used'])
        self.assertFalse(semantic_off['fusion']['semantic_context_used'])
        self.assertEqual(before, tensor_state_hash(model.state_dict()))

    def test_diagnostics_bounds_serialization_and_invalid_contracts(self):
        parent, (x, incident) = minimal(), event_inputs()
        model, _ = from_minimal(parent, 'event')
        activate(model)
        with torch.inference_mode():
            result = model(x, incident_data=incident, return_spatial_details=True)
        fusion = result['fusion']
        self.assertEqual(fusion['component_order'], list(COMPONENT_ORDER))
        self.assertTrue(bool(((fusion['weights'] >= .9) & (fusion['weights'] <= 1.1)).all()))
        self.assertLess(float((fusion['weights']-1).sum(-1).abs().max()), 5e-7)
        for key in ('component_encoding_abs_mean_by_component', 'component_contrast_abs_mean_by_component',
                    'component_log_rms_mean_by_component'):
            self.assertEqual(fusion[key].shape, (10,))
            self.assertFalse(fusion[key].requires_grad)
        self.assertTrue(all(value.ndim == 0 and not value.requires_grad for value in fusion['statistics'].values()))
        fresh, _ = from_minimal(parent, 'event')
        fresh.load_state_dict(model.state_dict(), strict=True)
        with torch.inference_mode():
            self.assertTrue(torch.equal(fresh(x, incident_data=incident), result['prediction']))
        for kwargs in ({'mode': 'bad'}, {'width': 0}, {'component_dim': True}, {'context_dim': -1},
                       {'max_delta': 0}, {'max_delta': float('nan')}, {'type_count': 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                from_minimal(parent, **kwargs)
        with self.assertRaisesRegex(ValueError, 'unadapted'):
            from_minimal(model)
        with self.assertRaisesRegex(ValueError, 'bool'):
            model(x, content_enabled=1)
        with self.assertRaisesRegex(ValueError, 'external'):
            model(x, forecast_delta=torch.zeros(1))
        with self.assertRaisesRegex(ValueError, 'Float32'):
            model(x, incident_data={**incident, 'relation': incident['relation'].double()})


if __name__ == '__main__':
    unittest.main()
