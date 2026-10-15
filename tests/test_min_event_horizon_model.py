"""Exact shared-core response, gradients, dropout and BN recomputation checks."""
import copy
import unittest
from unittest.mock import patch

import torch
from torch import nn

from src.models.incident_spatial_early_event import EarlyEventIGSTGNN, from_minimal
from src.models.incident_spatial_event_horizon import EventHorizonIGSTGNN
from src.models.incident_spatial_residual import tensor_state_hash
from tests.test_min_event_budget_model import event_inputs, minimal
from tests.test_min_early_event_model import activate


class ToyEvent(EarlyEventIGSTGNN):
    """Fast stochastic core to expose buffer/RNG and both-path gradients."""
    def __init__(self):
        nn.Module.__init__(self)
        self._early_event_mode = 'event'
        self.bn = nn.BatchNorm1d(3)
        self.dropout = nn.Dropout(.3)
        self.linear = nn.Linear(3, 2)
        self.response = nn.Parameter(torch.tensor([.2, -.3, .5]))

    def forward(self, x, label=None, *, report_enabled=True, return_spatial_details=False, **kwargs):
        z = x+self.response if report_enabled else x
        y = self.linear(self.dropout(self.bn(z)))[:, None, :, None].expand(-1, 12, -1, -1)
        return dict(prediction=y, spatial={}) if return_spatial_details else y


class EventHorizonModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def toy(self, checkpoint=True):
        torch.manual_seed(13)
        core = ToyEvent()
        return EventHorizonIGSTGNN(core, checkpoint_branches=checkpoint)

    def test_initial_native_ON_identity_OFF_and_single_intermediate_horizon(self):
        core, _ = from_minimal(minimal(), 'event')
        activate(core)
        core.eval()
        x, report = event_inputs()
        model = EventHorizonIGSTGNN(copy.deepcopy(core)).eval()
        before = tensor_state_hash(core.state_dict())
        with torch.inference_mode():
            on = core(x, incident_data=report)
            off = core(x, report_enabled=False)
            self.assertTrue(torch.equal(model(x, incident_data=report), on))
            self.assertTrue(torch.equal(model(x, incident_data=report, report_enabled=False), off))
            model.horizon_offset[3] = -.4
            result = model(x, incident_data=report, return_spatial_details=True)
            expected = on.clone()
            expected[:, 3] = on[:, 3]-.4*(on[:, 3]-off[:, 3])
            torch.testing.assert_close(result['prediction'], expected, atol=0, rtol=0)
            self.assertEqual(result['horizon']['paired_core_calls'], 2)
            self.assertEqual(result['horizon']['weights'].tolist(), [1., 1., 1., .6000000238418579]+[1.]*8)
            raw = model(x, incident_data=report, horizon_enabled=False)
            self.assertTrue(torch.equal(raw, on))
        self.assertEqual(tensor_state_hash(core.state_dict()), before)

    def test_gate_projection_bounds_and_no_amplification_or_reversal(self):
        model = self.toy().eval()
        x = torch.randn(5, 3)
        with torch.no_grad():
            model.horizon_offset.copy_(torch.linspace(-2, 1, 12))
            model.project_horizon_weights_()
            result = model(x, return_spatial_details=True)
        delta, applied = result['horizon']['raw_delta'], result['horizon']['applied_delta']
        self.assertTrue(bool((model.horizon_weights() >= 0).all() & (model.horizon_weights() <= 1).all()))
        self.assertTrue(bool((applied.abs() <= delta.abs()+1e-7).all()))
        self.assertTrue(bool((applied*delta >= 0).all()))
        # This mathematical endpoint is allowed normal float32 rounding.
        torch.testing.assert_close(result['prediction'][:, 0], result['horizon']['raw_off'][:, 0], atol=1e-7, rtol=1e-6)

    def test_checkpoint_matches_full_paired_gradients_buffers_and_rng(self):
        for seed in (0, 27):
            with self.subTest(seed=seed):
                model = self.toy(True).train()
                direct = self.toy(False).train()
                with torch.no_grad():
                    model.horizon_offset.fill_(-.35)
                    direct.load_state_dict(model.state_dict())
                x = torch.arange(15, dtype=torch.float32).reshape(5, 3)/7
                torch.manual_seed(seed)
                actual = model(x)
                actual.square().mean().backward()
                actual_rng = torch.get_rng_state().clone()
                torch.manual_seed(seed)
                expected = direct(x)
                expected.square().mean().backward()
                self.assertTrue(torch.equal(actual, expected))
                self.assertTrue(torch.equal(actual_rng, torch.get_rng_state()))
                self.assertEqual(tensor_state_hash(model.state_dict()), tensor_state_hash(direct.state_dict()))
                self.assertEqual(int(model.core.bn.num_batches_tracked), 1)
                for (name, parameter), (other, reference) in zip(model.named_parameters(), direct.named_parameters()):
                    self.assertEqual(name, other)
                    torch.testing.assert_close(parameter.grad, reference.grad, atol=1e-7, rtol=1e-6)

    def test_zero_gate_gradient_includes_current_OFF_core_not_detached(self):
        model = self.toy().eval()
        with torch.no_grad():
            model.horizon_offset.fill_(-1.)
        x = torch.randn(5, 3)
        model(x).square().mean().backward()
        self.assertIsNotNone(model.core.linear.weight.grad)
        self.assertGreater(float(model.core.linear.weight.grad.abs().sum()), 0.)
        self.assertIsNotNone(model.horizon_offset.grad)
        self.assertGreater(float(model.horizon_offset.grad.abs().sum()), 0.)
        # OFF does not read the event-specific response parameter.
        torch.testing.assert_close(model.core.response.grad, torch.zeros_like(model.core.response), atol=0, rtol=0)

    def test_actual_MIN_checkpoint_and_full_graph_have_matching_joint_gradients(self):
        core, _ = from_minimal(minimal(), 'event')
        activate(core)
        model = EventHorizonIGSTGNN(copy.deepcopy(core), checkpoint_branches=True).train()
        direct = EventHorizonIGSTGNN(copy.deepcopy(core), checkpoint_branches=False).train()
        with torch.no_grad():
            model.horizon_offset.copy_(torch.linspace(-.8, -.1, 12))
            direct.horizon_offset.copy_(model.horizon_offset)
        x, report = event_inputs()
        torch.manual_seed(199)
        actual = model(x, incident_data=report)
        actual.square().mean().backward()
        rng = torch.get_rng_state().clone()
        torch.manual_seed(199)
        expected = direct(x, incident_data=report)
        expected.square().mean().backward()
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(tensor_state_hash(model.state_dict()), tensor_state_hash(direct.state_dict()))
        for (name, left), (_, right) in zip(model.named_parameters(), direct.named_parameters()):
            with self.subTest(parameter=name):
                torch.testing.assert_close(left.grad, right.grad, atol=1e-7, rtol=1e-6)

    def test_no_report_zero_response_even_with_training_dropout(self):
        core, _ = from_minimal(minimal(), 'event')
        activate(core)
        model = EventHorizonIGSTGNN(core).train()
        x, report = event_inputs()
        report['report_present'].zero_()
        with torch.no_grad():
            model.horizon_offset.fill_(-.5)
        model(x, incident_data=report).mean().backward()
        torch.testing.assert_close(model.horizon_offset.grad, torch.zeros(12), atol=0, rtol=0)

    def test_training_gate_one_matches_native_core_update_and_rng(self):
        model = self.toy(True).train()
        native = copy.deepcopy(model.core).train()
        x = torch.randn(5, 3)
        torch.manual_seed(71)
        actual = model(x)
        actual.square().mean().backward()
        rng = torch.get_rng_state().clone()
        torch.manual_seed(71)
        expected = native(x)
        expected.square().mean().backward()
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(tensor_state_hash(model.core.state_dict()), tensor_state_hash(native.state_dict()))
        for left, right in zip(model.core.parameters(), native.parameters()):
            torch.testing.assert_close(left.grad, right.grad, atol=0, rtol=0)

    def test_control_is_one_pass_and_freezes_only_new_offsets(self):
        core, _ = from_minimal(minimal(), 'event')
        activate(core)
        model = EventHorizonIGSTGNN(core, 'control').eval()
        x, report = event_inputs()
        with patch.object(core, 'forward', wraps=core.forward) as forward:
            with torch.no_grad():
                result = model(x, incident_data=report, return_spatial_details=True)
            self.assertEqual(forward.call_count, 1)
        self.assertFalse(result['horizon']['response_measured'])
        self.assertIsNone(result['horizon']['raw_delta'])
        self.assertFalse(model.horizon_offset.requires_grad)
        self.assertTrue(all(p.requires_grad for p in core.parameters()))
        self.assertEqual(model.param_num(), sum(p.numel() for p in core.parameters())+12)

    def test_auxiliary_exception_restores_BN_and_rng_and_input(self):
        model = self.toy(False).train()
        native = copy.deepcopy(model.core).train()
        x = torch.randn(5, 3)
        untouched = x.clone()
        torch.manual_seed(44)
        native(x)
        expected_rng = torch.get_rng_state().clone()
        original = model.core.forward
        def failing(*args, **kwargs):
            output = original(*args, **kwargs)
            if not kwargs['report_enabled']:
                raise RuntimeError('auxiliary failed')
            return output
        torch.manual_seed(44)
        with patch.object(model.core, 'forward', side_effect=failing), self.assertRaisesRegex(RuntimeError, 'auxiliary failed'):
            model(x)
        self.assertEqual(tensor_state_hash(model.core.state_dict()), tensor_state_hash(native.state_dict()))
        self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
        self.assertTrue(torch.equal(x, untouched))


if __name__ == '__main__':
    unittest.main()
