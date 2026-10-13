"""Event budget identities, routing and trainability; no quality claims."""
import copy
import unittest
from unittest.mock import patch

import torch

from src.models.incident_spatial import configure_spatial_mode, exchange_step
from src.models.incident_spatial_event_budget import (
    DIAGNOSTIC_SCOPE, EventBudgetExchange, from_minimal, signed_budget_update,
)
from src.models.incident_spatial_minimal import from_residual as prune
from src.models.incident_spatial_residual import from_canonical_b3, tensor_state_hash
from tests.test_architecture_mechanisms import tiny_model
from tests.test_b3_r import controlled_forecast
from tests.test_b3_r_minimal_model import chronological_inputs
from tests.test_incident_spatial import graph


def minimal(pairs=((0, 1),)):
    canonical = configure_spatial_mode(tiny_model().requires_grad_(True), 'state_restricted', graph(4, pairs))
    return prune(from_canonical_b3(canonical)[0])[0].eval()


def event_inputs():
    x, legacy = chronological_inputs()
    relation = legacy['distances'].clone().float()
    return x, dict(type=torch.tensor([1, 2]), description=torch.tensor([2, 3]),
        type_valid=torch.tensor([True, True]), description_valid=torch.tensor([True, True]),
        relation=relation, relation_available=torch.tensor([True, True]), report_present=torch.tensor([True, True]))


def nonzero_adapter(model):
    with torch.no_grad():
        model.event_budget_adapter.A[0].weight.fill_(.005)
        model.event_budget_adapter.A[0].bias.fill_(1.)
        model.event_budget_adapter.A[-1].weight.fill_(.02)
        model.event_budget_adapter.A[-1].bias.copy_(torch.tensor([-.2, .1, -.1, .2]))


class EventBudgetModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_signed_interpolation_hand_values_bounds_and_nonzero_zero_start_gradient(self):
        base = torch.tensor([[[.2, .6, .001, 1.]]], dtype=torch.double)
        change = torch.tensor([[[-.5, .5, -1., 1.]]], dtype=torch.double, requires_grad=True)
        actual = signed_budget_update(base, change, torch.ones(1, 1, 1, dtype=torch.bool))
        torch.testing.assert_close(actual, torch.tensor([[[.1005, .8, .001, 1.]]], dtype=torch.double))
        self.assertTrue(bool(((actual >= .001) & (actual <= 1)).all()))
        zero = torch.zeros_like(base, requires_grad=True)
        identity = signed_budget_update(base, zero, torch.ones(1, 1, 1, dtype=torch.bool))
        torch.testing.assert_close(identity, base, atol=0, rtol=0)
        identity.sum().backward()
        self.assertGreater(float(zero.grad[..., :3].sum()), 0.)
        torch.testing.assert_close(signed_budget_update(base, change, torch.zeros(1, 1, 1, dtype=torch.bool)),
                                   base, atol=0, rtol=0)

    def test_retained_exact_independent_state_rng_shared_A_and_unique_adapter(self):
        parent = minimal()
        before, rng = tensor_state_hash(parent.state_dict()), torch.get_rng_state().clone()
        location, left = from_minimal(parent, 'location')
        semantic, right = from_minimal(parent, 'semantic')
        torch.testing.assert_close(rng, torch.get_rng_state(), atol=0, rtol=0)
        self.assertEqual(tensor_state_hash(parent.state_dict()), before)
        self.assertEqual(left['adapter_A_initial_sha256'], right['adapter_A_initial_sha256'])
        for candidate, certificate in ((location, left), (semantic, right)):
            self.assertTrue(certificate['retained_state_exact_verified'])
            self.assertEqual(certificate['added_parameters'], sum(p.numel() for p in candidate.parameters())
                             -sum(p.numel() for p in parent.parameters()))
            for name, value in parent.state_dict().items():
                torch.testing.assert_close(candidate.state_dict()[name], value, atol=0, rtol=0)
                self.assertNotEqual(candidate.state_dict()[name].data_ptr(), value.data_ptr())
            for layer in candidate.layers:
                exchange = layer.dif_layer.localized_st_conv.incident_exchange
                self.assertIsInstance(exchange, EventBudgetExchange)
                self.assertFalse(any(name.startswith('event_budget_adapter.') for name in exchange.state_dict()))
            owners = [name for name, _ in candidate.named_modules() if name == 'event_budget_adapter']
            self.assertEqual(owners, ['event_budget_adapter'])
            self.assertTrue(all(not value.is_inference() for value in candidate.parameters()))
        self.assertFalse(hasattr(location.event_budget_adapter, 'semantic_encoder'))

    def test_zero_initialized_complete_prediction_and_original_collect_equal_MIN(self):
        parent, (x, report) = minimal(), event_inputs()
        for mode in ('location', 'semantic'):
            model, _ = from_minimal(parent, mode)
            with torch.inference_mode():
                baseline = parent(x, return_spatial_details=True)
                actual = model(x, incident_data=report, return_spatial_details=True)
            torch.testing.assert_close(actual['prediction'], baseline['prediction'], atol=0, rtol=0)
            for field in ('capacity', 'boundary', 'base_node_capacity', 'base_logit'):
                torch.testing.assert_close(actual['spatial'][field], baseline['spatial'][field], atol=0, rtol=0)
            self.assertEqual(len(actual['spatial']['records']), 20)
            for left, right in zip(actual['spatial']['records'], baseline['spatial']['records']):
                torch.testing.assert_close(left, right, atol=0, rtol=0)
            records = actual['spatial']['event_budget_records']
            self.assertEqual(len(records), 40)
            self.assertEqual(actual['spatial']['event_budget_diagnostic_scope'], DIAGNOSTIC_SCOPE)
            for row in records:
                self.assertEqual(float(row['capacity_change_abs_max']), 0.)
                self.assertEqual(float(row['flux_change_abs_max']), 0.)
                self.assertTrue(all(not value.requires_grad for value in row.values() if isinstance(value, torch.Tensor)))

    def test_nonzero_report_off_missing_and_unavailable_return_same_weight_MIN_without_adapter(self):
        parent, (x, report) = minimal(), event_inputs()
        model, _ = from_minimal(parent, 'semantic')
        nonzero_adapter(model)
        with torch.inference_mode():
            baseline = parent(x)
            on = model(x, incident_data=report)
        self.assertGreater(float((on-baseline).abs().max()), 0.)
        with patch.object(model.event_budget_adapter, 'forward', side_effect=AssertionError('Adapter ran while OFF')):
            with torch.inference_mode():
                for kwargs in ({'incident_data': {'malformed': True}, 'report_enabled': False},
                               {'incident_data': None},
                               {'incident_data': {**report, 'report_present': torch.tensor([False, False])}},
                               {'incident_data': {**report, 'relation_available': torch.tensor([False, False])}},
                               {'incident_data': {**report, 'relation': torch.zeros_like(report['relation'])}}):
                    actual = model(x, **kwargs)
                    torch.testing.assert_close(actual, baseline, atol=0, rtol=0)

    def test_semantic_off_preserves_relation_support_and_uses_UNK_not_whole_adapter_off(self):
        with controlled_forecast() as (residual, _, _):
            parent = prune(residual)[0]
            model, _ = from_minimal(parent, 'semantic')
            nonzero_adapter(model)
            encoder = model.event_budget_adapter.semantic_encoder
            with torch.no_grad():
                encoder.type_embedding.weight.fill_(.1)
                encoder.type_embedding.weight[-1].fill_(.9)
                encoder.description_embedding.weight.fill_(.2)
                encoder.description_embedding.weight[-1].fill_(.8)
                encoder.projection[0].weight.fill_(.1)
                encoder.projection[0].bias.fill_(.2)
            x, report = event_inputs()
            with torch.inference_mode():
                on = model(x, incident_data=report, return_spatial_details=True)
                sem_off = model(x, incident_data=report, semantic_enabled=False, return_spatial_details=True)
                invalid_ignored = model(x, incident_data={**report, 'type': 'not read', 'description': None}, semantic_enabled=False)
                all_off = model(x, incident_data=report, report_enabled=False)
            for name in ('event_relation', 'event_edge_support', 'report_mask'):
                torch.testing.assert_close(on['spatial'][name], sem_off['spatial'][name], atol=0, rtol=0)
            torch.testing.assert_close(sem_off['prediction'], invalid_ignored, atol=0, rtol=0)
            self.assertGreater(float((sem_off['prediction']-all_off).abs().max()), 0.)
            self.assertGreater(float((sem_off['prediction']-on['prediction']).abs().max()), 0.)
            self.assertFalse(sem_off['spatial']['event_semantic_enabled'])
            self.assertTrue(sem_off['spatial']['report_path_executed'])

    def test_boundaries_and_unsupported_edges_unchanged_budgets_and_internal_balance(self):
        parent = minimal(((-1, 0), (0, 1), (1, 2), (2, 3), (3, -1)))
        model, _ = from_minimal(parent, 'location')
        nonzero_adapter(model)
        x, report = event_inputs()
        report['relation'].zero_()
        report['relation'][:, 0, 1] = .8
        seen = []
        original = exchange_step
        def capture(state, proposal, capacity, graph_value, **kwargs):
            record = original(state, proposal, capacity, graph_value, **kwargs)
            seen.append((capacity.detach().clone(), record, state.detach().clone()))
            return record
        with patch('src.models.incident_spatial_event_budget.exchange_step', side_effect=capture):
            with torch.inference_mode():
                actual = model(x, incident_data=report, return_spatial_details=True)
        context = actual['spatial']
        self.assertTrue(bool(context['event_edge_support'][:, 1].all()))
        self.assertFalse(bool(context['event_edge_support'][:, [0, 2, 3, 4]].any()))
        for capacity, record, state in seen:
            windows = capacity.shape[0]//2
            base = context['capacity'][:, None].expand(-1, windows, -1, -1).reshape_as(capacity)
            torch.testing.assert_close(capacity[:, [0, 2, 3, 4]], base[:, [0, 2, 3, 4]], atol=0, rtol=0)
            self.assertTrue(bool(((capacity >= .001) & (capacity <= 1)).all()))
            self.assertTrue(bool(((record['state'] >= -1e-5) & (record['state'] <= 1+1e-5)).all()))
            self.assertLess(float(record['balance'].abs().max()), 1e-5)
            self.assertLess(float((record['flux']-capacity).clamp_min(0).max()), 1e-5)
        self.assertGreater(max(float(row['capacity_change_abs_max']) for row in context['event_budget_records']), 0.)

    def test_full_task_gradient_reaches_final_projection_then_hidden_and_semantics(self):
        with controlled_forecast() as (residual, _, _):
            model, _ = from_minimal(prune(residual)[0], 'semantic')
            with torch.no_grad():
                model.event_budget_adapter.A[0].weight.fill_(.005)
                model.event_budget_adapter.A[0].bias.fill_(1.)
                model.event_budget_adapter.semantic_encoder.projection[0].bias.fill_(1.)
            x, report = event_inputs()
            optimizer = torch.optim.SGD(model.event_budget_adapter.parameters(), lr=.005)
            output = model(x, incident_data=report)
            (output-output.detach()-1).abs().mean().backward()
            adapter = model.event_budget_adapter
            self.assertGreater(float(adapter.A[-1].weight.grad.abs().sum()), 0.)
            self.assertEqual(float(adapter.A[0].weight.grad.abs().sum()), 0.)
            self.assertEqual(float(adapter.semantic_encoder.type_embedding.weight.grad.abs().sum()), 0.)
            optimizer.step()
            model.zero_grad(set_to_none=True)
            result = model(x, incident_data=report, return_spatial_details=True)
            (result['prediction']-result['prediction'].detach()-1).abs().mean().backward()
            self.assertGreater(float(adapter.A[0].weight.grad.abs().sum()), 0.)
            self.assertGreater(float(adapter.semantic_encoder.type_embedding.weight.grad.abs().sum()), 0.)
            self.assertTrue(all(value.grad is None or bool(torch.isfinite(value.grad).all()) for value in model.parameters()))
            self.assertGreater(sum(float(row['flux_change_abs_sum']) for row in result['spatial']['event_budget_records']), 0.)

    def test_current_state_recomputed_each_substep_from_same_base_not_accumulated(self):
        with controlled_forecast() as (residual, _, _):
            model, _ = from_minimal(prune(residual)[0], 'location')
            adapter = model.event_budget_adapter
            with torch.no_grad():
                adapter.A[0].weight.zero_()
                adapter.A[0].weight[:, 2*adapter.hidden] = .5  # source z, not fixed edge hidden
                adapter.A[0].bias.fill_(1.)
                adapter.A[-1].weight.fill_(.03)
            captured = []
            def observe(module, args, output):
                captured.append((args[1].detach().clone(), args[2].detach().clone(), output.detach().clone()))
            handle = adapter.register_forward_hook(observe)
            try:
                with torch.inference_mode():
                    x, report = event_inputs()
                    model(x, incident_data=report)
            finally:
                handle.remove()
            self.assertEqual(len(captured), 40)
            state0, base0, update0 = captured[0]
            state1, base1, update1 = captured[1]
            torch.testing.assert_close(base0, base1, atol=0, rtol=0)
            self.assertGreater(float((state0-state1).abs().max()), 0.)
            self.assertGreater(float((update0-update1).abs().max()), 0.)

    def test_nonbinding_constant_proposal_has_zero_flux_response_despite_budget_change(self):
        with controlled_forecast() as (residual, _, _):
            parent = prune(residual)[0]
            for layer in parent.layers:
                with torch.no_grad():
                    layer.dif_layer.localized_st_conv.incident_exchange.proposal[-1].bias.fill_(-8.)
            model, _ = from_minimal(parent, 'location')
            nonzero_adapter(model)
            x, report = event_inputs()
            with torch.inference_mode():
                actual = model(x, incident_data=report, return_spatial_details=True)
                baseline = parent(x)
            rows = actual['spatial']['event_budget_records']
            self.assertGreater(max(float(row['capacity_change_abs_max']) for row in rows), 0.)
            self.assertEqual(max(float(row['flux_change_abs_max']) for row in rows), 0.)
            torch.testing.assert_close(actual['prediction'], baseline, atol=0, rtol=0)

    def test_location_ignores_semantics_missing_categories_UNK_and_invalid_contract_rejected(self):
        x, report = event_inputs()
        parent = minimal()
        location, _ = from_minimal(parent, 'location')
        semantic, _ = from_minimal(parent, 'semantic')
        with torch.inference_mode():
            expected = location(x, incident_data=report)
            actual = location(x, incident_data={key: value for key, value in report.items()
                if key not in ('type', 'description', 'type_valid', 'description_valid')})
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            missing = {**report, 'type': torch.tensor([-999, -1]), 'type_valid': torch.tensor([False, False]),
                       'description': torch.tensor([999, -1]), 'description_valid': torch.tensor([False, False])}
            result = semantic(x, incident_data=missing)
            self.assertTrue(bool(torch.isfinite(result).all()))
        for replacement, message in (({'type': torch.tensor([6, 0])}, 'category'),
            ({'description': torch.tensor([29, 0])}, 'category'),
            ({'type': torch.tensor([1., 2.])}, 'Long'),
            ({'type_valid': torch.tensor([1, 1])}, 'Bool'),
            ({'relation': report['relation'].double()}, 'Float32'),
            ({'relation_available': torch.ones(2)}, 'Bool'),
            ({'relation': torch.full_like(report['relation'], float('nan'))}, 'finite')):
            with self.subTest(replacement=replacement), self.assertRaisesRegex(ValueError, message):
                semantic(x, incident_data={**report, **replacement})
        with self.assertRaisesRegex(ValueError, 'bool'):
            semantic(x, incident_data=report, report_enabled=0)
        with self.assertRaisesRegex(ValueError, 'Unknown'):
            from_minimal(parent, 'unsupported')
        with self.assertRaisesRegex(ValueError, 'unadapted'):
            from_minimal(semantic, 'semantic')

    def test_on_off_on_exception_cleanup_strict_reload_and_no_cross_forward_cache(self):
        model, _ = from_minimal(minimal(), 'semantic')
        nonzero_adapter(model)
        x, report = event_inputs()
        before = tensor_state_hash(model.state_dict())
        with torch.inference_mode():
            first = model(x, incident_data=report, return_spatial_details=True)
            model(x, incident_data=report, report_enabled=False)
            with self.assertRaisesRegex(ValueError, 'category'):
                model(x, incident_data={**report, 'type': torch.tensor([99, 1])})
            restored = model(x, incident_data=report, return_spatial_details=True)
        torch.testing.assert_close(first['prediction'], restored['prediction'], atol=0, rtol=0)
        self.assertIsNot(first['spatial'], restored['spatial'])
        self.assertIsNot(first['spatial']['event_budget_records'], restored['spatial']['event_budget_records'])
        self.assertEqual(len(first['spatial']['event_budget_records']), 40)
        self.assertEqual(before, tensor_state_hash(model.state_dict()))
        reloaded, _ = from_minimal(minimal(), 'semantic')
        reloaded.load_state_dict(model.state_dict(), strict=True)
        with torch.inference_mode():
            torch.testing.assert_close(reloaded(x, incident_data=report), first['prediction'], atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
