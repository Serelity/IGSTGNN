"""Server-only GW11 contracts using a small native-shaped source double.

These checks establish placement, nesting and gradients, not forecast gains.
Do not run Python, imports, syntax checks or tests on the local workstation.
"""
import copy
from math import pi
import unittest
from unittest.mock import patch

import torch
from torch import nn

from experiments.state_gate_gw11 import model as gw11


class FakeGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.fully_connected_layer_1 = nn.Linear(4, 4)
        self.activation = nn.ReLU()
        self.fully_connected_layer_2 = nn.Linear(4, 1)
        self.logit_shapes, self.history_lengths, self.used_gates = [], [], []

    def forward(self, history, original_history, emb_u, emb_d):
        batch, full, nodes, _ = original_history.shape
        first = emb_u.mean(-1)[None, None, :].expand(batch, full, nodes)
        second = emb_d.mean(-1)[None, None, :].expand(batch, full, nodes)
        context = torch.stack((first, second, original_history[..., 1], original_history[..., 2]), -1)
        logits = self.fully_connected_layer_2(self.activation(self.fully_connected_layer_1(context)))
        gate = torch.sigmoid(logits)[:, -history.shape[1]:]
        self.logit_shapes.append(tuple(logits.shape))
        self.history_lengths.append(history.shape[1])
        self.used_gates.append(gate.detach().clone())
        return history*gate


class FakeLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.estimation_gate = FakeGate()


class FakeOriginal(nn.Module):
    def __init__(self, nodes):
        super().__init__()
        self._model_args = dict(num_feat=1)
        self.node_emb_u = nn.Parameter(torch.arange(nodes*3, dtype=torch.float32).reshape(nodes, 3)/10)
        self.node_emb_d = nn.Parameter(torch.arange(nodes*2, dtype=torch.float32).reshape(nodes, 2)/7)
        self.layers = nn.ModuleList(FakeLayer() for _ in range(5))


class FakeSource(nn.Module):
    def __init__(self, nodes=3):
        super().__init__()
        self.arm, self.stage = 'SUM_CONTROL', 'joint'
        self.node_num, self.input_dim, self.output_dim = nodes, 3, 1
        self.seq_len, self.horizon = 12, 12
        self.original = FakeOriginal(nodes)
        self.head = nn.Linear(1, 1)
        with torch.no_grad():
            self.head.weight.fill_(1.)
            self.head.bias.fill_(.7)
        self.calls, self.fail_at, self.skip_layer = 0, None, None
        self.stochastic = False
        self.received, self.cache_stale = None, False

    @property
    def cache_is_clear(self):
        return not self.cache_stale

    def assert_cache_clear(self):
        if self.cache_stale:
            raise AssertionError('fake source cache is stale')
        return True

    def start_diagnostics(self):
        self.diag_start = self.calls

    def diagnostics(self):
        return dict(complete_forwards=self.calls-getattr(self, 'diag_start', self.calls))

    def forward(self, history_data, label=None, incident_data=None, sensor_data=None):
        self.calls += 1
        self.received = (incident_data, sensor_data)
        history = history_data[..., :1]
        for index, layer in enumerate(self.original.layers):
            if index == self.fail_at:
                raise RuntimeError('native synthetic failure')
            if index != self.skip_layer:
                history = layer.estimation_gate(history, history_data,
                    self.original.node_emb_u, self.original.node_emb_d)
            if index < 4:
                history = history[:, 2:]
        result = self.head(history.mean(dim=1, keepdim=True)).expand(-1, 12, -1, -1)
        if self.stochastic and self.training:
            result = result+torch.rand_like(result)*.01
        return result


class CoordinateAdapter(nn.Module):
    def forward(self, features):
        return features[..., 6:7]


class StateGateGW11Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def source(self):
        return FakeSource()

    def wrapper(self, source, arm):
        with patch.object(gw11, 'BranchReadoutIGSTGNN', FakeSource):
            return gw11.StateGateIGSTGNN(source, arm)

    @staticmethod
    def batch(nodes=3):
        history = torch.ones(2, 12, nodes, 3)
        history[..., 0] = torch.arange(12, dtype=torch.float32)[None, :, None]/3+1.
        history[..., 1] = .25
        history[..., 2] = 2/7
        return history

    @staticmethod
    def assert_no_native_hooks(model):
        for layer in model.original.layers:
            if layer.estimation_gate.fully_connected_layer_2._forward_hooks:
                raise AssertionError('Native fc2 retained a GW11 hook')
        model.assert_cache_clear()

    def test_registered_arms_and_source_contract(self):
        self.assertEqual(gw11.ARMS, ('SUM_CONTINUE', 'STATIC_GATE', 'STATE_READOUT', 'STATE_GATE'))
        with self.assertRaises(ValueError): gw11.StateGateIGSTGNN(nn.Linear(1, 1), 'STATE_GATE')
        for changed in ('arm', 'stage', 'head', 'horizon'):
            source = self.source()
            setattr(source, changed, {'arm': 'BRANCH_READOUT', 'stage': 'probe', 'head': None, 'horizon': 6}[changed])
            with self.subTest(changed=changed), self.assertRaises(ValueError): self.wrapper(source, 'STATE_GATE')
        source = self.source(); source.head.weight.requires_grad_(False)
        with self.assertRaises(ValueError): self.wrapper(source, 'STATE_GATE')
        with self.assertRaises(ValueError): self.wrapper(self.source(), 'OTHER')
        with patch.object(gw11, 'BranchReadoutIGSTGNN', FakeSource), self.assertRaises(ValueError):
            gw11.StateGateIGSTGNN(self.source(), 'STATE_GATE', hidden=8)

    def test_constructor_preserves_source_tensor_flags_modes_rng_and_cross_arm_initialization(self):
        source = self.source().train()
        source.original.layers[0].eval()
        original = {name: value.clone() for name, value in source.state_dict().items()}
        modes = {name: module.training for name, module in source.named_modules()}
        flags = {name: parameter.requires_grad for name, parameter in source.named_parameters()}
        rng = torch.get_rng_state().clone()
        variants = [self.wrapper(copy.deepcopy(source), arm) for arm in gw11.ARMS]
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        for model in variants:
            self.assertEqual(modes, {name: module.training for name, module in model.source.named_modules()})
            self.assertEqual(flags, {name: parameter.requires_grad for name, parameter in model.source.named_parameters()})
            for name, value in model.source.state_dict().items(): self.assertTrue(torch.equal(value, original[name]))
            self.assert_no_native_hooks(model)
        first = variants[1].adapters.state_dict()
        for model in variants[1:]:
            self.assertEqual(model.trainable_parameter_counts()['added'], 725)
            self.assertEqual(len(model.adapters), 5)
            self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))
            for name, value in model.adapters.state_dict().items(): self.assertTrue(torch.equal(value, first[name]))
            for adapter in model.adapters:
                self.assertEqual(sum(parameter.numel() for parameter in adapter.parameters()), 145)
                self.assertTrue(torch.equal(adapter.output.weight, torch.zeros_like(adapter.output.weight)))
                self.assertTrue(torch.equal(adapter.output.bias, torch.zeros_like(adapter.output.bias)))
        self.assertEqual(variants[0].trainable_parameter_counts()['added'], 0)

    def test_stage_zero_recovers_trained_sum_in_train_eval_grad_and_no_grad(self):
        source, history = self.source(), self.batch()
        source.stochastic = True
        for arm in gw11.ARMS:
            for training in (False, True):
                for grad_enabled in (False, True):
                    with self.subTest(arm=arm, training=training, grad_enabled=grad_enabled):
                        expected_source = copy.deepcopy(source).train(training)
                        model = self.wrapper(copy.deepcopy(source), arm).train(training)
                        with torch.set_grad_enabled(grad_enabled):
                            before = torch.get_rng_state().clone()
                            expected = expected_source(history)
                            after = torch.get_rng_state().clone()
                            torch.set_rng_state(before)
                            actual = model(history, label=torch.zeros_like(expected))
                        self.assertTrue(torch.equal(expected, actual))
                        self.assertTrue(torch.equal(torch.get_rng_state(), after))
                        self.assertEqual(model.source.calls, 1)
                        self.assert_no_native_hooks(model)

    def test_seven_state_features_and_static_feature_replacement(self):
        history = self.batch()
        state = self.wrapper(self.source(), 'STATE_GATE').state_features(history)
        self.assertEqual(tuple(state.shape), (2, 12, 3, 7))
        torch.testing.assert_close(state[:, :, :, 0], history[:, -1, :, 0][:, None].expand(2, 12, 3))
        expected_change = torch.log1p((history[:, 1:, :, 0]-history[:, :-1, :, 0]).abs().mean(1))
        torch.testing.assert_close(state[:, :, :, 1], expected_change[:, None].expand(2, 12, 3))
        torch.testing.assert_close(state[..., 2], torch.ones(2, 12, 3))
        torch.testing.assert_close(state[..., 3], torch.zeros(2, 12, 3), atol=1e-6, rtol=0)
        torch.testing.assert_close(state[..., 4], torch.full((2, 12, 3), torch.sin(torch.tensor(4*pi/7)).item()))
        torch.testing.assert_close(state[..., 5], torch.full((2, 12, 3), torch.cos(torch.tensor(4*pi/7)).item()))
        torch.testing.assert_close(state[0, :, 0, 6], torch.arange(12)/11)
        static_model = self.wrapper(self.source(), 'STATIC_GATE')
        static = static_model.state_features(history)
        torch.testing.assert_close(static[..., 0], static_model.original.node_emb_u.mean(-1)[None, None].expand(2, 12, 3))
        torch.testing.assert_close(static[..., 1], static_model.original.node_emb_d.mean(-1)[None, None].expand(2, 12, 3))
        self.assertTrue(torch.equal(static[..., 2:], state[..., 2:]))
        changed = history.clone(); changed[..., 0] *= 11
        self.assertTrue(torch.equal(static_model.state_features(changed), static))
        self.assertFalse(torch.equal(self.wrapper(self.source(), 'STATE_GATE').state_features(changed), state))

    def test_gate_delta_is_before_native_sigmoid_and_every_native_fc2_executes_once(self):
        source = self.source()
        for layer in source.original.layers:
            with torch.no_grad():
                layer.estimation_gate.fully_connected_layer_2.weight.zero_()
                layer.estimation_gate.fully_connected_layer_2.bias.zero_()
        model = self.wrapper(source, 'STATE_GATE')
        for adapter in model.adapters:
            with torch.no_grad(): adapter.output.bias.fill_(2.)
        history = torch.ones(2, 12, 3, 3)
        model.start_diagnostics()
        actual = model(history)
        expected = torch.sigmoid(torch.tensor(2.)).pow(5)+.7
        torch.testing.assert_close(actual, torch.full_like(actual, expected.item()))
        self.assertEqual(model.source.calls, 1)
        for index, layer in enumerate(model.original.layers):
            gate = layer.estimation_gate
            self.assertEqual(gate.logit_shapes, [(2, 12, 3, 1)])
            self.assertEqual(gate.history_lengths, [12-2*index])
        diagnostics = model.diagnostics()
        self.assertEqual(diagnostics['gate_hook_calls'], [1]*5)
        self.assertEqual(diagnostics['adapter_calls'], [1]*5)
        self.assertEqual(diagnostics['source_forward_calls'], 1)
        self.assertEqual(diagnostics['complete_forwards'], 1)
        for record in diagnostics['layer_logit_delta']: self.assertEqual(record['abs_max'], 2.)
        self.assert_no_native_hooks(model)

    def test_coordinate_keeps_full_twelve_axis_then_native_layer_crops_suffix(self):
        source = self.source()
        for layer in source.original.layers:
            with torch.no_grad():
                layer.estimation_gate.fully_connected_layer_2.weight.zero_()
                layer.estimation_gate.fully_connected_layer_2.bias.zero_()
        model = self.wrapper(source, 'STATE_GATE')
        model.adapters = nn.ModuleList(CoordinateAdapter() for _ in range(5))
        history = torch.ones(2, 12, 3, 3)
        model(history)
        for index, layer in enumerate(model.original.layers):
            expected = torch.sigmoid(torch.arange(12, dtype=history.dtype)/11)[2*index:]
            torch.testing.assert_close(layer.estimation_gate.used_gates[-1][0, :, 0, 0], expected)
        self.assert_no_native_hooks(model)

    def test_readout_uses_same_state_features_after_complete_source_and_mean_of_five(self):
        source, history = self.source(), self.batch()
        expected = copy.deepcopy(source)(history)
        model = self.wrapper(source, 'STATE_READOUT')
        for index, adapter in enumerate(model.adapters):
            with torch.no_grad(): adapter.output.bias.fill_(index+1.)
        incident, sensors = {'incident': 4}, torch.ones(1)
        model.start_diagnostics()
        actual = model(history, incident_data=incident, sensor_data=sensors)
        torch.testing.assert_close(actual, expected+3.)
        self.assertIs(model.source.received[0], incident)
        self.assertIs(model.source.received[1], sensors)
        diagnostics = model.diagnostics()
        self.assertEqual(diagnostics['gate_hook_calls'], [0]*5)
        self.assertEqual(diagnostics['adapter_calls'], [1]*5)
        self.assertEqual(diagnostics['readout_delta']['abs_max'], 3.)
        self.assertEqual(model.source.calls, 1)
        self.assert_no_native_hooks(model)

    def test_joint_gradients_reach_source_and_all_adapter_outputs_then_inputs(self):
        history = self.batch()
        for arm in gw11.ADAPTER_ARMS:
            model = self.wrapper(self.source(), arm).train()
            model(history).sum().backward()
            self.assertIsNotNone(model.source.head.bias.grad)
            self.assertGreater(float(model.source.head.bias.grad.abs().sum()), 0.)
            for adapter in model.adapters:
                self.assertIsNotNone(adapter.output.bias.grad)
                self.assertGreater(float(adapter.output.bias.grad.abs().sum()), 0.)
                with torch.no_grad(): adapter.output.weight.fill_(.1)
            model.zero_grad(set_to_none=True)
            model(history).square().sum().backward()
            for adapter in model.adapters:
                self.assertIsNotNone(adapter.input.weight.grad)
                self.assertGreater(float(adapter.input.weight.grad.abs().sum()), 0.)
            self.assert_no_native_hooks(model)

    def test_native_exception_missing_layer_and_reentrant_call_cleanup(self):
        history = self.batch()
        model = self.wrapper(self.source(), 'STATE_GATE')
        model.source.fail_at = 2
        model.start_diagnostics()
        with self.assertRaisesRegex(RuntimeError, 'native synthetic failure'): model(history)
        self.assertEqual(model.diagnostics()['failed_forwards'], 1)
        self.assert_no_native_hooks(model)
        model.source.fail_at = None; model.source.skip_layer = 2
        with self.assertRaisesRegex(RuntimeError, 'native layer order'): model(history)
        self.assert_no_native_hooks(model)
        model.source.skip_layer = None
        def recurse(module, inputs):
            model(history)
        handle = model.source.register_forward_pre_hook(recurse)
        try:
            with self.assertRaisesRegex(RuntimeError, 'Reentrant'): model(history)
        finally:
            handle.remove()
        self.assert_no_native_hooks(model)
        model(history)
        self.assert_no_native_hooks(model)

    def test_invalid_feature_shapes_stale_cache_and_nested_source_cache(self):
        model, history = self.wrapper(self.source(), 'STATE_GATE'), self.batch()
        for invalid in (history[:, :11], history[..., :2], history.double(), history[:, :, :2]):
            with self.assertRaises(ValueError): model(invalid)
            self.assert_no_native_hooks(model)
        model._cache['stale'] = history
        with self.assertRaisesRegex(RuntimeError, 'Stale'): model(history)
        self.assert_no_native_hooks(model)
        model.source.cache_stale = True
        with self.assertRaises(AssertionError): model.assert_cache_clear()
        model.source.cache_stale = False
        self.assert_no_native_hooks(model)

    def test_strict_state_round_trip_has_source_and_five_adapters(self):
        source = self.source()
        for arm in gw11.ARMS:
            model = self.wrapper(copy.deepcopy(source), arm)
            restored = self.wrapper(copy.deepcopy(source), arm)
            restored.load_state_dict(model.state_dict(), strict=True)
            self.assertTrue(all(name.startswith(('source.', 'adapters.')) for name in model.state_dict()))
            for name, value in restored.state_dict().items(): self.assertTrue(torch.equal(value, model.state_dict()[name]))


if __name__ == '__main__': unittest.main()
