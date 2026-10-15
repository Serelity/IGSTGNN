"""Integration contracts for the new DIF operator; no forecasting quality claim."""
import copy
import io
import unittest
from unittest.mock import patch

import torch

from src.models.incident_spatial_conditional_propagation import from_minimal
from src.models.incident_spatial_residual import tensor_state_hash
from tests.test_min_event_budget_model import minimal, event_inputs


def adjacency(pairs):
    value = torch.zeros(4, 4)
    for i, j in pairs:
        value[i, j] = value[j, i] = 1.
    return value


def candidate(mode='event', pairs=((0, 1), (2, 3)), checkpoint=False):
    return from_minimal(minimal(pairs), mode, adjacency(pairs),
                        receiver_chunk_size=2, checkpoint_messages=checkpoint)


def activate(model):
    with torch.no_grad():
        for layer in model.layers:
            kernel = layer.dif_layer.localized_st_conv
            kernel.event_logit.weight.fill_(.03)
            kernel.event_value.weight.fill_(.03)
            kernel.event_value.bias.copy_(torch.linspace(-.01, .02, model._hidden_dim))


class ConditionalPropagationModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(87)

    def test_conversion_owns_source_copy_and_keeps_only_compatible_tensors(self):
        parent = minimal()
        source_hash = tensor_state_hash(parent.state_dict())
        rng = torch.get_rng_state().clone()
        model, certificate = from_minimal(parent, 'event', adjacency(((0, 1),)))
        self.assertEqual(source_hash, tensor_state_hash(parent.state_dict()))
        torch.testing.assert_close(rng, torch.get_rng_state(), atol=0, rtol=0)
        self.assertTrue(certificate['retained_state_exact_verified'])
        self.assertFalse(certificate['original_MIN_prediction_equivalence'])
        self.assertTrue(certificate['candidate_own_stage_zero_required'])
        self.assertEqual(certificate['hidden_dim'], 8)
        self.assertFalse(certificate['formal_contract_width'])
        self.assertTrue(all(p.requires_grad and not p.is_inference() for p in model.parameters()))
        self.assertFalse(hasattr(model, 'report_spatial_state'))
        self.assertFalse(hasattr(model, 'internal_spatial_graph'))
        self.assertFalse(any('incident_exchange' in n or 'fc_list_updt' in n for n in model.state_dict()))
        for name in certificate['retained_state_keys']:
            torch.testing.assert_close(parent.state_dict()[name], model.state_dict()[name], atol=0, rtol=0)
            self.assertNotEqual(parent.state_dict()[name].data_ptr(), model.state_dict()[name].data_ptr())

    def test_three_arms_have_identical_initial_tensors_and_own_stage_zero(self):
        parent, (x, report) = minimal(), event_inputs()
        models = [from_minimal(parent, mode, adjacency(((0, 1),)))[0]
                  for mode in ('history', 'mask_only', 'event')]
        self.assertEqual(len(set(tensor_state_hash(m.state_dict()) for m in models)), 1)
        with torch.inference_mode():
            predictions = [m(x, incident_data=report) for m in models]
            old = parent(x)
        for p in predictions[1:]:
            torch.testing.assert_close(p, predictions[0], atol=0, rtol=0)
        self.assertEqual(predictions[0].shape, (2, 12, 4, 1))
        self.assertGreater(float((predictions[0]-old).abs().max()), 1e-6)

    def test_report_fallbacks_skip_semantics_and_equal_current_history(self):
        model, _ = candidate()
        activate(model)
        history = copy.deepcopy(model)
        history._conditional_propagation_mode = 'history'
        x, report = event_inputs()
        unavailable = dict(report, relation_available=torch.zeros(2, dtype=torch.bool),
                           type='unread', description='unread')
        zero = dict(report, relation=torch.zeros_like(report['relation']), type='unread', description='unread')
        with torch.inference_mode(), patch.object(model.report_condition.semantic_encoder, 'forward',
                                                 side_effect=AssertionError('semantics read')):
            expected = history(x, incident_data='unread')
            for value, switches in ((None, {}), ('unread', {'report_enabled': False}),
                                    ('unread', {'spatial_report_enabled': False}),
                                    ('unread', {'logit_enabled': False, 'value_enabled': False}),
                                    (unavailable, {}), (zero, {})):
                actual = model(x, incident_data=value, **switches)
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_mask_only_preserves_support_without_reading_details_or_categories(self):
        model, _ = candidate('mask_only')
        activate(model)
        x, report = event_inputs()
        changed = dict(report, relation=report['relation']*7, type='unread', description='unread')
        with torch.inference_mode(), patch.object(model.report_condition.semantic_encoder, 'forward',
                                                 side_effect=AssertionError('semantics read')):
            a = model(x, incident_data=report, return_spatial_details=True)
            b = model(x, incident_data=changed, return_spatial_details=True)
        torch.testing.assert_close(a['prediction'], b['prediction'], atol=0, rtol=0)
        self.assertEqual(int(a['propagation']['support'].sum()), 4)
        self.assertEqual(int(a['propagation']['relation'].count_nonzero()), 0)
        self.assertFalse(a['propagation']['semantic_context_used'])

    def test_fixed_support_content_response_and_disconnected_component_isolation(self):
        model, _ = candidate()
        activate(model)
        x, report = event_inputs()
        report = dict(report, relation=torch.zeros_like(report['relation']))
        report['relation'][:, 0, :] = .25
        changed = dict(report, relation=report['relation']*3)
        with torch.inference_mode():
            off = model(x, incident_data=report, report_enabled=False)
            on = model(x, incident_data=report)
            other = model(x, incident_data=changed)
        self.assertGreater(float((on[:, :, :2]-off[:, :, :2]).abs().max()), 1e-7)
        self.assertGreater(float((on-other).abs().max()), 1e-7)
        torch.testing.assert_close(on[:, :, 2:], off[:, :, 2:], atol=0, rtol=0)
        torch.testing.assert_close(other[:, :, 2:], off[:, :, 2:], atol=0, rtol=0)

    def test_time_anchors_generated_blocks_and_single_graph_encoder_head(self):
        model, _ = candidate()
        x, report = event_inputs()
        with torch.inference_mode(), patch.object(model, '_graph_constructor', wraps=model._graph_constructor) as construct, \
                patch.object(model.embedding, 'forward', wraps=model.embedding.forward) as embed, \
                patch.object(model, '_decode_forecast', wraps=model._decode_forecast) as decode:
            output = model(x, incident_data=report, return_spatial_details=True)
        self.assertEqual((construct.call_count, embed.call_count, decode.call_count), (1, 1, 1))
        details = output['propagation']
        self.assertEqual(details['kernel_calls'], 20)
        self.assertEqual(details['history_lengths'], [12, 10, 8, 6, 4, 2])
        self.assertEqual(details['history_anchors_by_layer'], [list(range(i, 12)) for i in range(0, 12, 2)])
        self.assertEqual(len(details['forecast_windows']), 15)
        for row in details['forecast_windows']:
            block = row['target_block']
            self.assertEqual(row['token_kind'], [0]*(3-block)+[1]*block)
            self.assertEqual(row['token_block'], [-1]*(3-block)+list(range(block)))
            self.assertEqual(row['history_anchors'], list(range(9+block, 12))+[-1]*block)
        self.assertEqual(float(details['statistics']['condition_encoder_calls']), 1.)
        self.assertTrue(all(not v.requires_grad and v.numel() == 1 for v in details['statistics'].values()))
        self.assertTrue(all(layer.dif_layer.localized_st_conv.last_details is None for layer in model.layers))

    def test_zero_start_readouts_then_condition_factors_receive_training_gradients(self):
        model, _ = candidate(checkpoint=True)
        model.train()
        # Keep the semantic ReLU away from its all-negative boundary.
        with torch.no_grad():
            model.report_condition.semantic_encoder.projection[0].bias.fill_(1.)
        x, report = event_inputs()
        optimizer = torch.optim.SGD(model.parameters(), lr=.05)
        kernel = model.layers[0].dif_layer.localized_st_conv
        model(x, incident_data=report).square().mean().backward()
        self.assertGreater(float(kernel.event_value.weight.grad.abs().sum()), 0.)
        self.assertEqual(float(model.report_condition.condition[0].weight.grad.abs().sum()), 0.)
        self.assertGreater(float(model.embedding.weight.grad.abs().sum()), 0.)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        model(x, incident_data=report).square().mean().backward()
        self.assertGreater(float(model.report_condition.condition[0].weight.grad.abs().sum()), 0.)
        self.assertGreater(float(model.report_condition.semantic_encoder.type_embedding.weight.grad.abs().sum()), 0.)
        self.assertTrue(all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None))

    def test_serialization_keeps_full_prediction_and_diagnostics_nonpersistent(self):
        model, _ = candidate()
        activate(model)
        x, report = event_inputs()
        before = tensor_state_hash(model.state_dict())
        with torch.inference_mode():
            expected = model(x, incident_data=report, return_spatial_details=True)['prediction']
        self.assertEqual(before, tensor_state_hash(model.state_dict()))
        stream = io.BytesIO()
        torch.save(model.state_dict(), stream)
        stream.seek(0)
        restored, _ = candidate()
        restored.load_state_dict(torch.load(stream, weights_only=True))
        with torch.inference_mode():
            torch.testing.assert_close(restored(x, incident_data=report), expected, atol=0, rtol=0)

    def test_labels_sensors_and_semantic_off_are_not_future_inputs(self):
        model, _ = candidate()
        activate(model)
        x, report = event_inputs()
        invalid = dict(report, type='unread', description='unread', type_valid='unread', description_valid='unread')
        with torch.inference_mode():
            expected = model(x, incident_data=invalid, semantic_enabled=False)
            actual = model(x, label=torch.full((2, 12, 4, 1), float('nan')), sensor_data='unread',
                           incident_data=invalid, semantic_enabled=False)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        with self.assertRaisesRegex(ValueError, 'external residual'):
            model(x, forecast_delta=torch.zeros_like(expected))

    def test_invalid_active_inputs_and_clock_fail_closed(self):
        model, _ = candidate()
        x, report = event_inputs()
        for kwargs in ({'report_enabled': 1}, {'incident_data': 'invalid'},
                       {'incident_data': dict(report, relation=report['relation'].double())}):
            with self.assertRaises(ValueError):
                model(x, **kwargs)
        invalid = x.clone()
        invalid[:, 0, :, 1] = float('nan')
        with self.assertRaises(ValueError):
            model(invalid)


if __name__ == '__main__':
    unittest.main()
