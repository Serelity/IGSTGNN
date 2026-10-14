"""Single-path output, compatible transfer and condition routing contracts."""
import json
import unittest
from unittest.mock import patch

import torch

from src.models import incident_stock_flow as stock_flow
from src.models.incident_spatial_residual import tensor_state_hash
from tests.test_min_event_budget_model import event_inputs, minimal


def build(mode='location', pairs=((0, 1),), **recipe):
    args = dict(scaler_mean=3., scaler_std=2.)
    args.update(recipe)
    return stock_flow.build_stock_flow(minimal(pairs), mode, torch.full((4,), 2.), args)


def capacity_limited(model):
    """A controlled demand-above-service regime, not a forecast-quality claim."""
    with torch.no_grad():
        model.backbone.out_fc_2.weight.zero_()
        model.backbone.out_fc_2.bias.fill_(10.)
        model.decoder.service_prior[-1].weight.zero_()
        model.decoder.service_prior[-1].bias.fill_(-4.)
        model.decoder.initial_stock[-1].weight.zero_()
        model.decoder.initial_stock[-1].bias.zero_()


class IncidentStockFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_prunes_every_old_exchange_preserves_source_and_compatible_tensors(self):
        parent = minimal()
        before, rng = tensor_state_hash(parent.state_dict()), torch.get_rng_state().clone()
        model, certificate = stock_flow.build_stock_flow(parent, 'location', torch.full((4,), 2.),
            dict(scaler_mean=3., scaler_std=2.))
        self.assertFalse(hasattr(model.backbone, 'report_spatial_state'))
        self.assertFalse(hasattr(model.backbone, 'internal_spatial_graph'))
        self.assertFalse(any('incident_exchange' in name for name, _ in model.named_modules()))
        self.assertFalse(hasattr(model, 'response'))
        self.assertFalse(hasattr(model.decoder, 'readout'))
        for name, value in model.backbone.state_dict().items():
            torch.testing.assert_close(value, parent.state_dict()[name], rtol=0, atol=0)
            self.assertNotEqual(value.data_ptr(), parent.state_dict()[name].data_ptr())
        self.assertEqual(tensor_state_hash(parent.state_dict()), before)
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        self.assertTrue(certificate['compatible_warm_start'])
        self.assertFalse(certificate['exact_source_prediction_initialization'])
        self.assertEqual(certificate['physical_paths'], 1)
        self.assertEqual(certificate['trajectories'], 1)
        self.assertTrue(all(p.requires_grad for p in model.parameters()))
        json.dumps(certificate)

    def test_common_initialization_and_unmodified_ON_OFF_exact(self):
        x, report = event_inputs()
        location, left = build('location')
        state, right = build('state')
        self.assertEqual(left['common_initial_sha256'], right['common_initial_sha256'])
        self.assertEqual(left['decoder_parameters']-right['decoder_parameters'], 24)
        self.assertFalse(hasattr(state.decoder, 'relation_projection'))
        for model in (location, state):
            model.eval()
            with torch.inference_mode():
                on = model(x, report)
                off = model(x, report, report_enabled=False)
                missing = model(x)
            torch.testing.assert_close(on, off, rtol=0, atol=0)
            torch.testing.assert_close(missing, off, rtol=0, atol=0)
            self.assertEqual(on.shape, (2, 12, 4, 1))

    def test_dead_registration_cleanup_preserves_statistical_forecast_exactly(self):
        parent = minimal()
        reference = __import__('copy').deepcopy(parent)
        reference.__class__ = stock_flow.StatisticalMIN
        del reference.report_spatial_state
        del reference.internal_spatial_graph
        for layer in reference.layers:
            del layer.dif_layer.localized_st_conv.incident_exchange
        candidate, _ = stock_flow.prune_statistical_MIN(parent)
        x, _ = event_inputs()
        candidate.eval()
        reference.eval()
        with torch.no_grad():
            torch.testing.assert_close(candidate(x), reference(x), rtol=0, atol=0)
        self.assertFalse(any('.localized_st_conv.bn.' in name or '.time_slot_embedding.' in name
                             for name in candidate.state_dict()))
        self.assertFalse(hasattr(candidate.layers[-1].inh_layer, 'backcast_fc'))

    def test_missing_sides_receive_explicit_open_boundaries(self):
        model, certificate = build(pairs=((0, 1),))
        pairs = set(map(tuple, model.decoder.edge_index.t().tolist()))
        self.assertEqual(pairs, {(0, 1), (-1, 0), (1, -1), (-1, 2), (2, -1), (-1, 3), (3, -1)})
        self.assertEqual(certificate['added_boundary_count'], 6)
        self.assertFalse(certificate['physical_certified'])
        self.assertIn('not_certified_cells', certificate['graph_scope'])

    def test_final_prediction_is_integrated_actual_outflow_without_neural_bypass(self):
        model, _ = build()
        x, report = event_inputs()
        model.eval()
        native_step, records = stock_flow.step, []
        def collect(*args, **kwargs):
            record = native_step(*args, **kwargs)
            records.append(record['outflow'])
            return record
        with patch.object(stock_flow, 'step', side_effect=collect):
            result = model(x, report, return_details=True)
        expected = torch.stack([sum(records[2*i:2*i+2])*.5 for i in range(12)], 1)
        torch.testing.assert_close(result['flow_prediction'], expected, rtol=0, atol=0)
        torch.testing.assert_close(result['prediction'], (expected-3.)/2., rtol=0, atol=0)
        self.assertEqual(len(records), 24)
        self.assertTrue(bool((result['flow_prediction'] >= 0).all()))
        self.assertEqual(result['stock_flow']['output_scope'], 'sum_actual_node_outflow_dt_no_native_prediction_residual')
        self.assertIn('not_source_MIN', result['stock_flow']['off_reference'])
        self.assertEqual(len(result['stock_flow']['by_slot']), 12)
        json.dumps(result['stock_flow'])

    def test_relation_affects_service_only_not_arrivals_demand_or_initial_stock(self):
        model, _ = build(pairs=())
        capacity_limited(model)
        with torch.no_grad():
            model.decoder.condition_output.weight.fill_(.3)
            model.decoder.condition_output.bias.fill_(.2)
            model.decoder.relation_projection.weight.fill_(.5)
        x, report = event_inputs()
        model.eval()
        native_step = stock_flow.step
        traces = []
        def collect(*args):
            traces.append(tuple(value.detach().clone() if isinstance(value, torch.Tensor) else value
                                for value in args))
            return native_step(*args)
        with patch.object(stock_flow, 'step', side_effect=collect):
            first = model(x, report)
        left, traces = traces, []
        with patch.object(stock_flow, 'step', side_effect=collect):
            second = model(x, dict(report, relation=report['relation']*3.))
        torch.testing.assert_close(left[0][0], traces[0][0], rtol=0, atol=0)
        for a, b in zip(left, traces):
            torch.testing.assert_close(a[3], b[3], rtol=0, atol=0)  # edge demand including entry arrivals
            torch.testing.assert_close(a[1], b[1], rtol=0, atol=0)  # storage
        self.assertGreater(float((left[0][2]-traces[0][2]).abs().max()), 0.)
        self.assertGreater(float((first-second).detach().abs().max()), 0.)

    def test_state_ignores_numeric_D_and_unavailable_report_equals_OFF(self):
        model, _ = build('state', pairs=())
        capacity_limited(model)
        with torch.no_grad():
            model.decoder.condition_output.bias.fill_(.4)
        x, report = event_inputs()
        model.eval()
        with torch.inference_mode():
            actual = model(x, report)
            changed = model(x, dict(report, relation=report['relation']*9.))
            unavailable = model(x, dict(report, report_present=torch.zeros(2, dtype=torch.bool)))
            off = model(x, {'invalid': True}, report_enabled=False)
        torch.testing.assert_close(actual, changed, rtol=0, atol=0)
        torch.testing.assert_close(unavailable, off, rtol=0, atol=0)
        self.assertGreater(float((actual-off).abs().max()), 0.)

    def test_condition_gradient_when_service_binds_and_stock_is_persistent(self):
        model, _ = build(pairs=())
        capacity_limited(model)
        x, report = event_inputs()
        calls = []
        handle = model.decoder.initial_stock.register_forward_hook(lambda *args: calls.append(1))
        try:
            result = model(x, report, return_details=True)
            result['prediction'].sum().backward()
        finally:
            handle.remove()
        self.assertEqual(calls, [1])
        self.assertGreater(float(model.decoder.condition_output.weight.grad.abs().sum()), 0.)
        self.assertGreater(result['stock_flow']['service_active_count'], 0.)
        self.assertLess(result['stock_flow']['node_balance_abs_max'], 1e-5)
        self.assertLessEqual(result['stock_flow']['capacity_violation_max'], 1e-6)

    def test_statistics_and_forecast_head_have_task_gradients_in_demand_limited_regime(self):
        model, _ = build(pairs=())
        x, report = event_inputs()
        model.eval()
        with torch.no_grad():
            model.backbone.out_fc_2.bias.fill_(-1.)
            model.decoder.service_prior[-1].bias.fill_(4.)
        model(x, report).square().mean().backward()
        self.assertGreater(float(model.backbone.out_fc_2.bias.grad.abs().sum()), 0.)
        self.assertGreater(float(model.backbone.embedding.weight.grad.abs().sum()), 0.)
        self.assertEqual([name for name, value in model.named_parameters() if value.grad is None], [])

    def test_failed_backbone_call_removes_capture_hook_and_invalid_contract_rejected(self):
        model, _ = build()
        x, _ = event_inputs()
        with patch.object(model.backbone, 'forward', side_effect=RuntimeError('synthetic')):
            with self.assertRaisesRegex(RuntimeError, 'synthetic'):
                model(x)
        self.assertEqual(len(model.backbone.out_fc_1._forward_pre_hooks), 0)
        with self.assertRaisesRegex(ValueError, 'Strict'):
            build(strict_physics=True)
        with self.assertRaisesRegex(ValueError, 'Strict'):
            build(strict_physical=True)
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            build(steps=3, dt=.5)
        with self.assertRaisesRegex(ValueError, 'positive integers'):
            build(steps=True)
        with self.assertRaisesRegex(ValueError, 'scaler'):
            stock_flow.build_stock_flow(minimal(), flow_scale=torch.ones(4))


if __name__ == '__main__':
    unittest.main()
