"""Service-only report routing, frozen MIN identities and task gradients."""
import copy
import unittest
from unittest.mock import patch

import torch

from src.models.incident_spatial import exchange_step
from src.models.incident_spatial_event_budget import SharedEventBudgetAdapter
from src.models.incident_spatial_frozen_budget import (
    ADAPTER_PREFIX, assert_frozen, set_frozen_train_mode, trainable_named_parameters,
)
from src.models.incident_spatial_minimal import from_residual
from src.models.incident_spatial_residual import tensor_state_hash
from src.models.incident_spatial_service_budget import (
    DIAGNOSTIC_SCOPE, PROPOSAL_BUDGET_SOURCE, ServiceBudgetExchange,
    SharedServiceBudgetAdapter, adapter_parameter_count, build_frozen_service,
    matched_state_width,
)
from tests.test_b3_r import controlled_forecast
from tests.test_incident_spatial import graph
from tests.test_min_event_budget_model import event_inputs, minimal, nonzero_adapter


class ServiceBudgetModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_initial_on_off_exact_source_rng_and_parameter_certificates(self):
        base, (x, report) = minimal(), event_inputs()
        digest, rng = tensor_state_hash(base.state_dict()), torch.get_rng_state().clone()
        reference = copy.deepcopy(base).requires_grad_(False).eval()
        for mode in ('location', 'state'):
            model, certificate = build_frozen_service(base, mode)
            with torch.inference_mode():
                expected = reference(x)
                actual = model(x, incident_data=report, return_spatial_details=True)
                off = model(x, incident_data=report, report_enabled=False)
            torch.testing.assert_close(actual['prediction'], expected, atol=0, rtol=0)
            torch.testing.assert_close(off, expected, atol=0, rtol=0)
            self.assertEqual(certificate['frozen_state_sha256'], digest)
            self.assertEqual(certificate['service_mode'], mode)
            self.assertEqual(certificate['conditioning_mode'], mode)
            self.assertEqual(certificate['adapter_compatibility_mode'], 'location')
            self.assertEqual(certificate['trainable_parameters'], sum(p.numel() for _, p in trainable_named_parameters(model)))
            self.assertEqual(certificate['proposal_budget_source'], PROPOSAL_BUDGET_SOURCE)
            self.assertEqual(actual['spatial']['event_budget_diagnostic_scope'], DIAGNOSTIC_SCOPE)
            self.assertEqual(actual['spatial']['service_conditioning_mode'], mode)
            self.assertTrue(assert_frozen(model, digest))
            self.assertTrue(all(type(layer.dif_layer.localized_st_conv.incident_exchange) is ServiceBudgetExchange
                                for layer in model.layers))
        torch.testing.assert_close(rng, torch.get_rng_state(), atol=0, rtol=0)
        self.assertEqual(digest, tensor_state_hash(base.state_dict()))

    def test_compact_location_preserves_old_active_weights_and_function_without_zero_semantics(self):
        torch.manual_seed(47)
        old = SharedEventBudgetAdapter(8, 4, 'location', 6, 29)
        with torch.no_grad():
            old.A[-1].weight.normal_()
            old.A[-1].bias.normal_()
        compact = SharedServiceBudgetAdapter(8, 4, 'location', 16)
        self.assertEqual(compact.initialize_from_legacy_location(old), 16)
        torch.testing.assert_close(compact.A[0].weight, old.A[0].weight[:, :-16], atol=0, rtol=0)
        torch.testing.assert_close(compact.A[0].bias, old.A[0].bias, atol=0, rtol=0)
        torch.testing.assert_close(compact.A[-1].weight, old.A[-1].weight, atol=0, rtol=0)
        torch.testing.assert_close(compact.A[-1].bias, old.A[-1].bias, atol=0, rtol=0)
        h, z, c, d = torch.randn(2, 3, 8), torch.rand(2, 3, 4), torch.rand(2, 2, 4), torch.rand(2, 3, 3)
        g = graph(3, ((0, 1), (1, 2)))
        # Removing zero columns can alter summation kernels at ordinary roundoff.
        torch.testing.assert_close(compact(h, z, c, d, None, g), old(h, z, c, d, torch.zeros(2, 16), g), atol=5e-7, rtol=1e-6)
        self.assertEqual(sum(p.numel() for p in old.parameters())-sum(p.numel() for p in compact.parameters()), 16*16)

    def test_default_width_matches_active_parameter_count_and_shares_common_state_weights(self):
        self.assertEqual(matched_state_width(32, 4), 17)
        self.assertEqual(adapter_parameter_count(82, 16, 4), 1396)
        self.assertEqual(adapter_parameter_count(76, 17, 4), 1381)
        base = minimal()
        location, left = build_frozen_service(base, 'location')
        state, right = build_frozen_service(base, 'state')
        a, b = location.event_budget_adapter, state.event_budget_adapter
        rows = min(a.hidden_width, b.hidden_width)
        torch.testing.assert_close(a.A[0].weight[:rows, :b.input_dimension], b.A[0].weight[:rows], atol=0, rtol=0)
        torch.testing.assert_close(a.A[0].bias[:rows], b.A[0].bias[:rows], atol=0, rtol=0)
        self.assertEqual(a.input_dimension-b.input_dimension, 6)
        self.assertNotEqual(left['adapter_initial_sha256'], right['adapter_initial_sha256'])
        self.assertFalse(right['location_values_used'])
        self.assertTrue(right['support_mask_equal_across_modes'])

    def test_nonzero_off_missing_and_no_supported_internal_edges_equal_frozen_MIN(self):
        base, (x, report) = minimal(), event_inputs()
        reference = copy.deepcopy(base).requires_grad_(False).eval()
        with torch.inference_mode():
            expected = reference(x)
        for mode in ('location', 'state'):
            model, certificate = build_frozen_service(base, mode)
            nonzero_adapter(model)
            with patch.object(model.event_budget_adapter, 'forward', side_effect=AssertionError('Adapter should be skipped')):
                with torch.inference_mode():
                    for kwargs in ({'incident_data': {'malformed': True}, 'report_enabled': False},
                                   {'incident_data': None},
                                   {'incident_data': {**report, 'relation': torch.zeros_like(report['relation'])}},
                                   {'incident_data': {**report, 'relation_available': torch.tensor([False, False])}},
                                   {'incident_data': {**report, 'report_present': torch.tensor([False, False])}}):
                        torch.testing.assert_close(model(x, **kwargs), expected, atol=0, rtol=0)
            self.assertTrue(assert_frozen(model, certificate['frozen_state_sha256']))

    def test_state_control_never_reads_numeric_relation_values_at_fixed_support(self):
        with controlled_forecast() as (residual, _, _):
            model, _ = build_frozen_service(from_residual(residual)[0].eval(), 'state')
            nonzero_adapter(model)
            x, report = event_inputs()
            changed = {**report, 'relation': report['relation']*7.+(report['relation'] != 0).float()*.3}
            with torch.inference_mode():
                real = model(x, incident_data=report, return_spatial_details=True)
                altered = model(x, incident_data=changed, return_spatial_details=True)
            torch.testing.assert_close(real['prediction'], altered['prediction'], atol=0, rtol=0)
            torch.testing.assert_close(real['spatial']['event_edge_support'], altered['spatial']['event_edge_support'], atol=0, rtol=0)
            adapter = model.event_budget_adapter
            g = graph(3, ((0, 1), (1, 2)))
            h, z, c = torch.randn(2, 3, adapter.hidden), torch.rand(2, 3, 4), torch.rand(2, 2, 4)
            # Direct adapter can run with absent relation and semantic values.
            direct = adapter(h, z, c, None, None, g)
            torch.testing.assert_close(direct, adapter(h, z, c, object(), object(), g), atol=0, rtol=0)

    def test_location_values_can_change_service_limit_and_complete_prediction_at_fixed_support(self):
        with controlled_forecast() as (residual, _, _):
            model, _ = build_frozen_service(from_residual(residual)[0].eval(), 'location')
            adapter = model.event_budget_adapter
            with torch.no_grad():
                adapter.A[0].weight.zero_()
                adapter.A[0].bias.fill_(.2)
                # Source relation column one, after H_src/H_dst/z_src/z_dst/C0.
                adapter.A[0].weight[:, 2*adapter.hidden+3*adapter.latent+1] = .3
                adapter.A[-1].weight.fill_(.05)
            x, report = event_inputs()
            changed = {**report, 'relation': report['relation']*1.5}
            with torch.inference_mode():
                real = model(x, incident_data=report, return_spatial_details=True)
                altered = model(x, incident_data=changed, return_spatial_details=True)
            torch.testing.assert_close(real['spatial']['event_edge_support'], altered['spatial']['event_edge_support'], atol=0, rtol=0)
            self.assertGreater(float((real['prediction']-altered['prediction']).abs().max()), 0.)
            self.assertNotEqual(float(real['spatial']['event_budget_records'][0]['capacity_change_abs_sum']),
                                float(altered['spatial']['event_budget_records'][0]['capacity_change_abs_sum']))

    def test_full_forecast_loss_reaches_adapter_while_frozen_parameters_and_buffers_stay_identical(self):
        for mode in ('location', 'state'):
            with self.subTest(mode=mode), controlled_forecast() as (residual, _, _):
                base = from_residual(residual)[0].eval()
                model, certificate = build_frozen_service(base, mode)
                adapter = model.event_budget_adapter
                with torch.no_grad():
                    adapter.A[0].weight.fill_(.005)
                    adapter.A[0].bias.fill_(1.)
                optimizer = torch.optim.SGD([p for _, p in trainable_named_parameters(model)], lr=.005)
                x, report = event_inputs()
                for step in range(2):
                    set_frozen_train_mode(model)
                    optimizer.zero_grad(set_to_none=True)
                    prediction = model(x, incident_data=report)
                    (prediction-prediction.detach()-1).abs().mean().backward()
                    self.assertGreater(float(adapter.A[-1].weight.grad.abs().sum()), 0.)
                    self.assertEqual(float(adapter.A[0].weight.grad.abs().sum()) == 0., step == 0)
                    self.assertTrue(all(p.grad is None for name, p in model.named_parameters() if not name.startswith(ADAPTER_PREFIX)))
                    optimizer.step()
                    self.assertTrue(assert_frozen(model, certificate['frozen_state_sha256']))
                with torch.inference_mode():
                    on = model(x, incident_data=report)
                    off = model(x, incident_data=report, report_enabled=False)
                self.assertGreater(float((on-off).abs().max()), 0.)

    def test_same_state_proposal_shared_between_changed_and_original_service_limit(self):
        with controlled_forecast() as (residual, _, _):
            model, _ = build_frozen_service(from_residual(residual)[0].eval())
            nonzero_adapter(model)
            seen = []
            def capture(state, proposal, capacity, g, **kwargs):
                record = exchange_step(state, proposal, capacity, g, **kwargs)
                seen.append((state.detach().clone(), proposal.detach().clone(), capacity.detach().clone()))
                return record
            x, report = event_inputs()
            model.eval()
            with patch('src.models.incident_spatial_service_budget.exchange_step', side_effect=capture), torch.inference_mode():
                actual = model(x, incident_data=report, return_spatial_details=True)
            self.assertEqual(len(seen), 80)
            self.assertTrue(any(float((seen[j][2]-seen[j+1][2]).abs().max()) > 0 for j in range(0, len(seen), 2)))
            for j in range(0, len(seen), 2):
                torch.testing.assert_close(seen[j][0], seen[j+1][0], atol=0, rtol=0)
                torch.testing.assert_close(seen[j][1], seen[j+1][1], atol=0, rtol=0)
            self.assertTrue(all(float(row['proposal_change_abs_max']) == 0. for row in actual['spatial']['event_budget_records']))

    def test_nonbinding_service_change_has_exact_zero_flux_and_prediction_effect(self):
        with controlled_forecast() as (residual, _, _):
            base = from_residual(residual)[0].eval()
            # Proposals depend on C0, but are safely below both service limits.
            for layer in base.layers:
                exchange = layer.dif_layer.localized_st_conv.incident_exchange
                with torch.no_grad():
                    exchange.proposal[0].weight.fill_(.005)
                    exchange.proposal[0].bias.fill_(1.)
                    exchange.proposal[-1].weight.fill_(.001)
                    exchange.proposal[-1].bias.fill_(-8.)
            model, _ = build_frozen_service(base)
            nonzero_adapter(model)
            x, report = event_inputs()
            model.eval()
            with torch.inference_mode():
                on = model(x, incident_data=report, return_spatial_details=True)
                off = model(x, incident_data=report, report_enabled=False)
            rows = on['spatial']['event_budget_records']
            self.assertGreater(max(float(row['capacity_change_abs_max']) for row in rows), 0.)
            self.assertEqual(max(float(row['flux_change_abs_max']) for row in rows), 0.)
            torch.testing.assert_close(on['prediction'], off, atol=0, rtol=0)

    def test_binding_service_change_can_change_internal_flux_and_preserves_boundary_budgets(self):
        base = minimal(((-1, 0), (0, 1), (1, 2), (2, 3), (3, -1)))
        model, certificate = build_frozen_service(base)
        nonzero_adapter(model)
        x, report = event_inputs()
        report['relation'].zero_()
        report['relation'][:, 0, 1] = .8
        seen = []
        def capture(state, proposal, capacity, g, **kwargs):
            record = exchange_step(state, proposal, capacity, g, **kwargs)
            seen.append((capacity.detach().clone(), record))
            return record
        model.eval()
        with patch('src.models.incident_spatial_service_budget.exchange_step', side_effect=capture), torch.inference_mode():
            actual = model(x, incident_data=report, return_spatial_details=True)
        context = actual['spatial']
        for capacity, record in seen:
            windows = capacity.shape[0]//2
            original = context['capacity'][:, None].expand(-1, windows, -1, -1).reshape_as(capacity)
            torch.testing.assert_close(capacity[:, [0, 2, 3, 4]], original[:, [0, 2, 3, 4]], atol=0, rtol=0)
            self.assertTrue(bool(((capacity >= .001) & (capacity <= 1)).all()))
            self.assertLess(float(record['balance'].abs().max()), 1e-5)
            self.assertLess(float((record['flux']-capacity).clamp_min(0).max()), 1e-5)
            self.assertTrue(bool(((record['state'] >= -1e-5) & (record['state'] <= 1+1e-5)).all()))
        self.assertGreater(max(float(row['flux_change_abs_max']) for row in context['event_budget_records']), 0.)
        self.assertTrue(assert_frozen(model, certificate['frozen_state_sha256']))

    def test_invalid_modes_and_widths_are_rejected(self):
        base = minimal()
        with self.assertRaisesRegex(ValueError, 'Unknown'):
            build_frozen_service(base, 'semantic')
        with self.assertRaisesRegex(ValueError, 'positive integer'):
            build_frozen_service(base, hidden_width=0)
        with self.assertRaisesRegex(ValueError, 'positive integer'):
            build_frozen_service(base, hidden_width=True)


if __name__ == '__main__':
    unittest.main()
