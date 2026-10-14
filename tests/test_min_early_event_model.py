"""Early report condition identities and data-flow checks; no quality claims."""
import copy
import unittest
from unittest.mock import patch

import torch

from src.models.incident_spatial_early_event import MODES, from_minimal
from src.models.incident_spatial_residual import tensor_state_hash
from tests.test_min_event_budget_model import event_inputs, minimal


def activate(model):
    with torch.no_grad():
        condition = model.early_event_condition.condition
        condition[0].weight.fill_(.02)
        condition[0].bias.fill_(.5)
        condition[-1].weight.fill_(.03)
        condition[-1].bias.fill_(.2)


class EarlyEventModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_identical_registered_initial_state_all_modes_independent_copy_and_rng(self):
        parent = minimal().requires_grad_(False)
        digest, rng = tensor_state_hash(parent.state_dict()), torch.get_rng_state().clone()
        candidates = [from_minimal(parent, mode) for mode in MODES]
        torch.testing.assert_close(torch.get_rng_state(), rng, atol=0, rtol=0)
        self.assertEqual(tensor_state_hash(parent.state_dict()), digest)
        self.assertEqual(len({cert['condition_initial_sha256'] for _, cert in candidates}), 1)
        self.assertEqual(len({cert['parameter_count'] for _, cert in candidates}), 1)
        for model, cert in candidates:
            self.assertTrue(cert['retained_state_exact_verified'])
            self.assertTrue(cert['all_parameters_trainable'])
            self.assertEqual(cert['added_parameters'], sum(p.numel() for p in model.early_event_condition.parameters()))
            for name, value in parent.state_dict().items():
                torch.testing.assert_close(model.state_dict()[name], value, atol=0, rtol=0)
                self.assertNotEqual(model.state_dict()[name].data_ptr(), value.data_ptr())
            self.assertFalse(any(p.is_inference() for p in model.parameters()))

    def test_zero_initialized_all_predictions_equal_MIN_and_original_exchange_calls_preserved(self):
        parent, (x, report) = minimal(), event_inputs()
        with torch.inference_mode():
            baseline = parent(x, return_spatial_details=True)
            for mode in MODES:
                model, _ = from_minimal(parent, mode)
                actual = model(x, incident_data=report, return_spatial_details=True)
                torch.testing.assert_close(actual['prediction'], baseline['prediction'], atol=0, rtol=0)
                self.assertEqual(len(actual['spatial']['records']), 20)
                self.assertEqual(float(actual['spatial']['early_event_diagnostics']['injection_calls']), 1.)
                for left, right in zip(actual['spatial']['records'], baseline['spatial']['records']):
                    torch.testing.assert_close(left, right, atol=0, rtol=0)

    def test_nonzero_OFF_and_missing_support_exact_MIN(self):
        parent, (x, report) = minimal(), event_inputs()
        model, _ = from_minimal(parent, 'event')
        activate(model)
        with torch.inference_mode():
            baseline = parent(x)
            self.assertGreater(float((model(x, incident_data=report)-baseline).abs().max()), 0.)
        with patch.object(model.early_event_condition, 'forward', side_effect=AssertionError('OFF condition executed')):
            with torch.inference_mode():
                for kwargs in ({'report_enabled': False, 'incident_data': 'ignored'}, {'incident_data': None},
                    {'incident_data': {**report, 'report_present': torch.zeros(2, dtype=torch.bool)}},
                    {'incident_data': {**report, 'relation_available': torch.zeros(2, dtype=torch.bool)}},
                    {'incident_data': {**report, 'relation': torch.zeros_like(report['relation'])}}):
                    actual = model(x, return_spatial_details=True, **kwargs)
                    torch.testing.assert_close(actual['prediction'], baseline, atol=0, rtol=0)
                    self.assertFalse(bool(actual['spatial']['early_event_support'].any()))
                    self.assertEqual(float(actual['spatial']['early_event_diagnostics']['injection_calls']), 0.)

    def test_local_mask_bound_single_latest_slot_and_unchanged_prior_and_graph_inputs(self):
        parent, (x, report) = minimal(), event_inputs()
        model, _ = from_minimal(parent, 'location')
        activate(model)
        report['relation'].zero_()
        report['relation'][:, 0, 1] = .8
        seen = {}
        def capture(name):
            def observe(module, args, output):
                seen[name] = args[0].detach().clone()
            return observe
        handles = [model.layers[0].register_forward_hook(capture('conditioned')),
                   parent.layers[0].register_forward_hook(capture('original'))]
        try:
            with patch.object(parent, '_graph_constructor', wraps=parent._graph_constructor) as original_graph:
                with patch.object(model, '_graph_constructor', wraps=model._graph_constructor) as condition_graph:
                    with torch.inference_mode():
                        baseline = parent(x, return_spatial_details=True)
                        actual = model(x, incident_data=report, return_spatial_details=True)
                    torch.testing.assert_close(original_graph.call_args.kwargs['history_data'],
                                               condition_graph.call_args.kwargs['history_data'], atol=0, rtol=0)
        finally:
            for handle in handles:
                handle.remove()
        context = actual['spatial']
        torch.testing.assert_close(seen['original'][:, :-1], seen['conditioned'][:, :-1], atol=0, rtol=0)
        torch.testing.assert_close(seen['original'][:, -1, 1:], seen['conditioned'][:, -1, 1:], atol=0, rtol=0)
        self.assertGreater(float(context['early_event_direct_change'][:, 0].abs().max()), 0.)
        self.assertEqual(float(context['early_event_direct_change'][:, 1:].abs().max()), 0.)
        self.assertLessEqual(float(context['early_event_gamma'].abs().max()), .25)
        for name in ('capacity', 'boundary', 'base_logit', 'base_node_capacity'):
            torch.testing.assert_close(context[name], baseline['spatial'][name], atol=0, rtol=0)
        self.assertTrue(all(not value.requires_grad for value in context['early_event_diagnostics'].values()))

    def test_state_ignores_all_report_values_and_location_semantic_OFF_equals_event_UNK(self):
        parent, (x, report) = minimal(), event_inputs()
        state, _ = from_minimal(parent, 'state')
        location, _ = from_minimal(parent, 'location')
        event, _ = from_minimal(parent, 'event')
        for model in (state, location, event):
            activate(model)
        with torch.inference_mode():
            expected = state(x, incident_data=report, return_spatial_details=True)
            actual = state(x, incident_data='never read')
            torch.testing.assert_close(actual, expected['prediction'], atol=0, rtol=0)
            self.assertTrue(bool(expected['spatial']['early_event_support'].all()))
            a = location(x, incident_data={key: value for key, value in report.items() if key.startswith('relation') or key == 'report_present'})
            b = event(x, incident_data={**report, 'type': 'ignored'}, semantic_enabled=False)
            torch.testing.assert_close(a, b, atol=0, rtol=0)
            full = event(x, incident_data=report)
            self.assertGreater(float((full-b).abs().max()), 0.)

    def test_task_gradient_updates_condition_and_backbone_jointly(self):
        model, _ = from_minimal(minimal(), 'event')
        x, report = event_inputs()
        with torch.no_grad():
            model.early_event_condition.condition[0].bias.fill_(1.)
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        condition = model.early_event_condition.condition
        backbone_before = model.out_fc_2.weight.detach().clone()
        output = model(x, incident_data=report)
        (output-output.detach()-1).abs().mean().backward()
        self.assertGreater(float(condition[-1].weight.grad.abs().sum()), 0.)
        self.assertEqual(float(condition[0].weight.grad.abs().sum()), 0.)
        self.assertGreater(float(model.embedding.weight.grad.abs().sum()), 0.)
        optimizer.step()
        self.assertFalse(torch.equal(model.out_fc_2.weight, backbone_before))
        model.zero_grad(set_to_none=True)
        output = model(x, incident_data=report)
        (output-output.detach()-1).abs().mean().backward()
        self.assertGreater(float(condition[0].weight.grad.abs().sum()), 0.)
        self.assertGreater(float(model.early_event_condition.semantic_encoder.type_embedding.weight.grad.abs().sum()), 0.)
        self.assertTrue(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()))
        same_weight_MIN = minimal()
        same_weight_MIN.load_state_dict({name: value for name, value in model.state_dict().items()
                                        if not name.startswith('early_event_condition.')}, strict=True)
        with torch.inference_mode():
            torch.testing.assert_close(model(x, report_enabled=False), same_weight_MIN(x), atol=0, rtol=0)

    def test_invalid_contracts_missing_category_UNK_and_lifecycle_restore(self):
        model, _ = from_minimal(minimal(), 'event')
        activate(model)
        x, report = event_inputs()
        before = tensor_state_hash(model.state_dict())
        with torch.inference_mode():
            first = model(x, incident_data=report, return_spatial_details=True)
            model(x, report_enabled=False)
            missing = {**report, 'type': torch.tensor([-99, -99]), 'type_valid': torch.zeros(2, dtype=torch.bool),
                       'description': torch.tensor([99, 99]), 'description_valid': torch.zeros(2, dtype=torch.bool)}
            self.assertTrue(bool(torch.isfinite(model(x, incident_data=missing)).all()))
        for replacement, message in (({'type': torch.tensor([6, 0])}, 'category'),
            ({'description': torch.tensor([29, 0])}, 'category'),
            ({'type_valid': torch.ones(2)}, 'Bool'), ({'relation': report['relation'].double()}, 'Float32'),
            ({'relation_available': torch.ones(2)}, 'Bool'),
            ({'relation': torch.full_like(report['relation'], float('nan'))}, 'finite')):
            with self.subTest(replacement=replacement), self.assertRaisesRegex(ValueError, message):
                model(x, incident_data={**report, **replacement})
        with torch.inference_mode():
            final = model(x, incident_data=report, return_spatial_details=True)
        torch.testing.assert_close(first['prediction'], final['prediction'], atol=0, rtol=0)
        self.assertIsNot(first['spatial'], final['spatial'])
        self.assertEqual(tensor_state_hash(model.state_dict()), before)
        fresh, _ = from_minimal(minimal(), 'event')
        fresh.load_state_dict(model.state_dict(), strict=True)
        with torch.inference_mode():
            torch.testing.assert_close(fresh(x, incident_data=report), first['prediction'], atol=0, rtol=0)
        for kwargs in ({'mode': 'unknown'}, {'width': 0}, {'width': True}, {'max_scale': 1},
                       {'max_scale': float('nan')}, {'max_scale': True}):
            with self.assertRaises(ValueError):
                from_minimal(minimal(), **kwargs)
        with self.assertRaisesRegex(ValueError, 'unadapted'):
            from_minimal(model)
        with self.assertRaisesRegex(ValueError, 'bool'):
            model(x, report_enabled=0)


if __name__ == '__main__':
    unittest.main()
