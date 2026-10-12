"""Behavioral contracts for pruning incident paths, not scientific outcomes."""
import unittest

import torch

from src.models.incident_spatial_minimal import (
    MinimalIGSTGNN, cutoff_clock_from_history, from_residual,
)
from src.models.incident_spatial_residual import tensor_state_hash
from src.models.incident_spatial_strength import from_residual as with_strength
from tests.test_architecture_mechanisms import inputs, tiny_model
from tests.test_b3_r import canonical
from src.models.incident_spatial_residual import from_canonical_b3


def chronological_inputs():
    x, incident = inputs()
    # Cutoffs cross midnight, including the Saturday/Sunday week rollover.
    for row, (last_slot, last_day) in enumerate(((287, 6), (286, 2))):
        for step in range(12):
            absolute = last_slot-11+step
            x[row, step, :, 1] = (absolute % 288)/288
            x[row, step, :, 2] = ((last_day+absolute//288) % 7)/7
    tod, dow = cutoff_clock_from_history(x)
    incident['forecast_tod'], incident['forecast_dow'] = tod, dow
    return x, incident


class MinimalModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_clock_recovery_all_daily_slots_and_week_rollover(self):
        x = torch.zeros(288*7, 12, 2, 3)
        slots = torch.arange(288).repeat(7)
        days = torch.arange(7).repeat_interleave(288)
        x[:, -1, :, 1] = (slots.float()/288)[:, None]
        x[:, -1, :, 2] = (days.float()/7)[:, None]
        tod, dow = cutoff_clock_from_history(x)
        torch.testing.assert_close(tod, (slots+2) % 288, atol=0, rtol=0)
        torch.testing.assert_close(dow, (days+(slots+2)//288) % 7, atol=0, rtol=0)
        self.assertEqual(int(dow[-1]), 0)

    def test_exact_state_subset_and_parent_rng_preservation(self):
        parent, _ = from_canonical_b3(canonical())
        before = tensor_state_hash(parent.state_dict())
        rng = torch.get_rng_state().clone()
        result, certificate = from_residual(parent)
        self.assertIsInstance(result, MinimalIGSTGNN)
        self.assertEqual(before, tensor_state_hash(parent.state_dict()))
        torch.testing.assert_close(rng, torch.get_rng_state(), atol=0, rtol=0)
        self.assertFalse(hasattr(result, 'tiid_module'))
        self.assertFalse(hasattr(result.icsf_module, 'report_encoder'))
        self.assertFalse(hasattr(result.icsf_module, 'k_proj'))
        self.assertFalse(hasattr(result.report_spatial_state, 'D'))
        self.assertFalse(hasattr(result.report_spatial_state, 'gate'))
        self.assertEqual(certificate['parameter_count'], sum(p.numel() for p in result.parameters()))
        self.assertEqual(certificate['removed_parameter_count'], sum(certificate['removed_parameter_groups'].values()))
        for name, tensor in result.state_dict().items():
            torch.testing.assert_close(tensor, parent.state_dict()[name], atol=0, rtol=0)
            self.assertNotEqual(tensor.data_ptr(), parent.state_dict()[name].data_ptr())
        self.assertTrue(certificate['all_parameters_trainable'])
        self.assertFalse(certificate['report_inputs_consumed'])

    def test_full_predictions_capacities_and_exchange_equal_both_off(self):
        parent, _ = from_canonical_b3(canonical())
        parent.eval()
        with torch.no_grad():
            parent.report_spatial_state.D[-1].bias.fill_(1.3)
        result, _ = from_residual(parent)
        result.eval()
        x, incident = chronological_inputs()
        with torch.inference_mode():
            reference = parent(x, incident_data=incident, spatial_report_enabled=False,
                               tiid_report_enabled=False, return_spatial_details=True)
            actual = result(x, return_spatial_details=True)
        torch.testing.assert_close(actual['prediction'], reference['prediction'], atol=0, rtol=0)
        for field in ('capacity', 'base_edge_capacity', 'boundary', 'base_node_capacity',
                      'node_capacity', 'base_logit', 'updated_logit', 'effective_logit_delta'):
            torch.testing.assert_close(actual['spatial'][field], reference['spatial'][field], atol=0, rtol=0)
        for left, right in zip(actual['spatial']['records'], reference['spatial']['records']):
            torch.testing.assert_close(left, right, atol=0, rtol=0)
        self.assertEqual(len(actual['spatial']['records']), 20)
        self.assertFalse(bool(actual['spatial']['report_mask'].any()))
        self.assertIsNone(actual['spatial']['report_logit_delta'])

    def test_predictions_do_not_consume_report_or_label_and_flags_cannot_reactivate(self):
        parent, _ = from_canonical_b3(canonical())
        result, _ = from_residual(parent.eval())
        x, report = chronological_inputs()
        with torch.inference_mode():
            reference = result(x)
            altered = {'forecast_tod': torch.tensor([-999., float('nan')]),
                       'forecast_dow': None, 'distances': 'intentionally malformed report',
                       'report_age_minutes': torch.tensor([1000., -100.])}
            for supplied in (report, altered):
                for tiid, spatial in ((True, True), (False, False), (True, False), (False, True)):
                    actual = result(x, label=torch.full((2, 12, 4, 1), float('nan')),
                                    incident_data=supplied, sensor_data={'ignored': True},
                                    tiid_report_enabled=tiid, spatial_report_enabled=spatial)
                    torch.testing.assert_close(actual, reference, atol=0, rtol=0)

    def test_gradient_through_retained_prior_and_exchange_and_strict_reload(self):
        parent, _ = from_canonical_b3(canonical())
        parent.eval()
        result, _ = from_residual(parent)
        x, incident = chronological_inputs()
        # The complete real P network remains active; no output-replacing hook
        # can manufacture a gradient or bypass its learned weights.
        optimizer = torch.optim.SGD(result.parameters(), lr=.1)
        before = result.report_spatial_state.P[-1].weight.detach().clone()
        output = result(x)
        (output-output.detach()-1).abs().mean().backward()
        for parameter in result.report_spatial_state.P.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertGreater(float(parameter.grad.abs().sum()), 0.)
        exchange = result.layers[0].dif_layer.localized_st_conv.incident_exchange
        self.assertIsNotNone(exchange.readout.weight.grad)
        self.assertGreater(float(exchange.readout.weight.grad.abs().sum()), 0.)
        self.assertTrue(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in result.parameters()))
        optimizer.step()
        self.assertFalse(torch.equal(before, result.report_spatial_state.P[-1].weight))
        copied, _ = from_residual(parent)
        copied.load_state_dict(result.state_dict(), strict=True)
        copied.eval()
        # Retaining the newly optimized shared weights in the full parent is
        # still exactly equivalent to disabling its two report paths.
        missing = parent.load_state_dict(result.state_dict(), strict=False)
        self.assertFalse(missing.unexpected_keys)
        self.assertTrue(missing.missing_keys)
        with torch.inference_mode():
            torch.testing.assert_close(copied(x), result(x), atol=0, rtol=0)
            torch.testing.assert_close(result(x), parent(x, incident_data=incident,
                spatial_report_enabled=False, tiid_report_enabled=False), atol=0, rtol=0)

    def test_optional_strength_gate_is_physically_pruned_and_inference_copy_trainable(self):
        parent, _ = from_canonical_b3(canonical())
        gated, _ = with_strength(parent, 'identity_gate')
        with torch.inference_mode():
            result, certificate = from_residual(gated)
        self.assertIn('report_spatial_state.gate', certificate['removed_parameter_groups'])
        self.assertFalse(any('gate.' in name for name in result.report_spatial_state.state_dict()))
        self.assertTrue(all(not parameter.is_inference() for parameter in result.parameters()))
        x, incident = chronological_inputs()
        gated.eval()
        result.eval()
        with torch.inference_mode():
            torch.testing.assert_close(result(x), gated(x, incident_data=incident,
                spatial_report_enabled=False, tiid_report_enabled=False), atol=0, rtol=0)

    def test_invalid_parent_history_clocks_and_external_residual_fail(self):
        with self.assertRaisesRegex(ValueError, 'architecture'):
            from_residual(tiny_model())
        result, _ = from_residual(from_canonical_b3(canonical())[0])
        x, _ = chronological_inputs()
        malformed = x.clone()
        malformed[0, -1, 1, 1] += .1
        with self.assertRaisesRegex(ValueError, 'across nodes'):
            result(malformed)
        malformed = x.clone()
        malformed[0, -1, :, 1] = .1234
        with self.assertRaisesRegex(ValueError, 'nominal'):
            result(malformed)
        with self.assertRaisesRegex(ValueError, 'contract'):
            result(x[:, :11])
        with self.assertRaisesRegex(ValueError, 'bool'):
            result(x, tiid_report_enabled=0)
        with self.assertRaisesRegex(ValueError, 'external'):
            result(x, forecast_delta=torch.zeros(2, 12, 4, 256))


if __name__ == '__main__':
    unittest.main()
