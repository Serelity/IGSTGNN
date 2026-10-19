"""GW15 server-only placement, nesting, gradient and cleanup contracts.

Do not execute Python imports, syntax checks or tests on the local workstation.
These small source doubles establish implementation contracts, not usefulness.
"""
import copy
import unittest

import torch
from torch import nn

from experiments.trajectory_encoder_gw15 import model as gw15


class FakeOriginal(nn.Module):
    def __init__(self, hidden=4):
        super().__init__()
        self.embedding = nn.Linear(1, hidden)
        self.graph = nn.Linear(hidden, hidden)
        self.forecast = nn.Linear(hidden, 1)
        self.seen_embedding, self.seen_incident_input = None, None

    def forward(self, history, incident_data=None, duplicate=False):
        encoded = self.embedding(history[..., :1])
        self.seen_embedding = encoded.detach().clone()
        if duplicate:
            self.embedding(history[..., :1])
        self.seen_incident_input = encoded.detach().clone()
        if incident_data is not None:
            encoded = encoded+incident_data
        # The added signal travels through incident fusion, graph mixing and
        # the existing forecast mapping before reaching the complete source.
        mixed = torch.tanh(encoded+self.graph(encoded.mean(2, keepdim=True)))
        return self.forecast(mixed)


class FakeSource(nn.Module):
    def __init__(self, hidden=4):
        super().__init__()
        self.original = FakeOriginal(hidden)
        self.head = nn.Linear(1, 1)
        self.fail, self.duplicate = False, False

    def forward(self, history, incident_data=None):
        if self.fail:
            raise RuntimeError('synthetic source failure')
        return self.head(self.original(history, incident_data, self.duplicate))


class TrajectoryEncoderGW15Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    @staticmethod
    def history(changes=None, batch=2, nodes=3):
        if changes is None:
            changes = [2., -1., 3., -2., 1., 4., -3., 2., -1., 0., 1.]
        traffic = torch.cat((torch.tensor([10.]), 10.+torch.tensor(changes).cumsum(0)))
        result = torch.zeros(batch, 12, nodes, 3)
        result[..., 0] = traffic[None, :, None]
        result[..., 1], result[..., 2] = .25, 2/7
        return result

    def encoder(self, arm='ORDERED_DELTA', hidden=4):
        return gw15.TrajectoryEncoder(arm, hidden)

    def assert_clear(self, source):
        self.assertFalse(source.original.embedding._forward_hooks)
        self.assertFalse(hasattr(source, gw15._GUARD))
        self.assertFalse(hasattr(source.original.embedding, gw15._GUARD))

    def test_registered_arms_and_fixed_width(self):
        self.assertEqual(gw15.ARMS, ('SORTED_DELTA', 'ORDERED_DELTA'))
        self.assertEqual((gw15.HISTORY, gw15.INPUTS, gw15.HIDDEN, gw15.RESIDUAL_SCALE), (12, 11, 16, .1))
        for kwargs in (dict(arm='OTHER', source_hidden=4), dict(arm='ORDERED_DELTA', source_hidden=0),
                dict(arm='ORDERED_DELTA', source_hidden=True),
                dict(arm='ORDERED_DELTA', source_hidden=4, hidden=8)):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                gw15.TrajectoryEncoder(**kwargs)

    def test_sorted_control_preserves_change_multiset_and_discards_order(self):
        first = self.history()
        changes = first[0, 1:, 0, 0]-first[0, :-1, 0, 0]
        second = self.history(changes.flip(0).tolist())
        self.assertTrue(torch.equal(gw15.history_delta(first, 'SORTED_DELTA'),
            gw15.history_delta(second, 'SORTED_DELTA')))
        self.assertFalse(torch.equal(gw15.history_delta(first, 'ORDERED_DELTA'),
            gw15.history_delta(second, 'ORDERED_DELTA')))
        ordered = gw15.history_delta(first, 'ORDERED_DELTA')
        self.assertTrue(torch.equal(ordered.sort(-1).values, gw15.history_delta(first, 'SORTED_DELTA')))
        self.assertTrue(torch.equal(first[:, -1, :, 0], second[:, -1, :, 0]))

    def test_normalization_is_nodewise_and_detaches_history(self):
        history = self.history().requires_grad_()
        features = gw15.history_delta(history, 'ORDERED_DELTA')
        change = (history[:, 1:, :, 0]-history[:, :-1, :, 0]).permute(0, 2, 1)
        expected = change/(1+change.abs().mean(-1, keepdim=True))
        self.assertEqual(tuple(features.shape), (2, 3, 11))
        self.assertTrue(torch.equal(features, expected))
        self.assertFalse(features.requires_grad)
        self.assertIsNone(features.grad_fn)
        self.assertLess(float(features.abs().max()), 11.)
        altered = history.detach().clone()
        altered[:, :, 1, 0] *= 3.
        changed = gw15.history_delta(altered, 'ORDERED_DELTA')
        self.assertTrue(torch.equal(changed[:, 0], features[:, 0]))
        self.assertFalse(torch.equal(changed[:, 1], features[:, 1]))

    def test_no_history_level_or_clock_input_and_no_history_mutation(self):
        history, original = self.history(), self.history()
        changed = history.clone()
        changed[..., 0] += 100.
        changed[..., 1], changed[..., 2] = .9, .1
        for arm in gw15.ARMS:
            self.assertTrue(torch.equal(gw15.history_delta(history, arm), gw15.history_delta(changed, arm)))
        self.assertTrue(torch.equal(history, original))
        constant = torch.ones_like(history)
        self.assertTrue(torch.equal(gw15.history_delta(constant, 'ORDERED_DELTA'), torch.zeros(2, 3, 11)))

    def test_history_contract_rejects_invalid_axes_types_nonfinite_and_future_arguments(self):
        valid = self.history()
        bad = [None, valid.long(), valid[:, :-1], valid[..., :2], valid[:, :, :0], valid[:0]]
        for index in (0, 1, 2):
            item = valid.clone(); item[0, 0, 0, index] = float('nan'); bad.append(item)
        item = valid.clone(); item[0, 0, 0, 0] = float('inf'); bad.append(item)
        for item in bad:
            with self.subTest(item_type=type(item)), self.assertRaises(ValueError):
                gw15.history_delta(item, 'ORDERED_DELTA')
        with self.assertRaises(ValueError): gw15.history_delta(valid, 'OTHER')
        with self.assertRaises(TypeError): gw15.history_delta(valid, 'ORDERED_DELTA', label=valid)

    def test_pair_initialization_is_identical_preserves_rng_and_owns_only_adapter(self):
        torch.manual_seed(2025)
        rng = torch.get_rng_state().clone()
        ordered, sorted_encoder = self.encoder(), self.encoder('SORTED_DELTA')
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        for name, tensor in ordered.state_dict().items():
            self.assertTrue(torch.equal(tensor, sorted_encoder.state_dict()[name]))
        self.assertEqual(set(ordered.state_dict()), {'input.weight', 'input.bias', 'output.weight'})
        self.assertEqual(ordered.param_num(), 16*12+16*12*4)
        self.assertEqual(self.encoder(hidden=32).param_num(), 6336)
        self.assertTrue(torch.equal(ordered.output.weight, torch.zeros_like(ordered.output.weight)))
        self.assertFalse(any(isinstance(module, FakeSource) for module in ordered.modules()))

    def test_output_axes_zero_nesting_and_bound_after_nonzero_weights(self):
        encoder, features = self.encoder(), gw15.history_delta(self.history(), 'ORDERED_DELTA')
        initial = encoder(features)
        self.assertEqual(tuple(initial.shape), (2, 12, 3, 4))
        self.assertTrue(torch.equal(initial, torch.zeros_like(initial)))
        with torch.no_grad(): encoder.output.weight.fill_(50.)
        actual = encoder(features)
        self.assertTrue(torch.isfinite(actual).all())
        self.assertTrue((actual.abs() <= gw15.RESIDUAL_SCALE).all())

    def test_encoder_feature_shape_dtype_and_finite_contract(self):
        encoder, valid = self.encoder(), torch.zeros(2, 3, 11)
        bad = [valid.double(), valid[:, :, :10], valid[:, :0], valid[:0], valid.unsqueeze(0), None]
        item = valid.clone(); item[0, 0, 0] = float('nan'); bad.append(item)
        for item in bad:
            with self.subTest(item_type=type(item)), self.assertRaises(ValueError): encoder(item)

    def test_zero_context_recovers_complete_original_forecast_exactly(self):
        source, history = FakeSource().eval(), self.history()
        report = torch.full((2, 12, 3, 4), .2)
        original_state = {name: value.clone() for name, value in source.state_dict().items()}
        for arm in gw15.ARMS:
            expected = source(history, incident_data=report)
            encoder = self.encoder(arm)
            with gw15.encoding_context(source, encoder, history):
                actual = source(history, incident_data=report)
            self.assertTrue(torch.equal(actual, expected))
            self.assert_clear(source)
        for name, value in source.state_dict().items(): self.assertTrue(torch.equal(value, original_state[name]))

    def test_residual_enters_embedding_before_incident_and_native_graph(self):
        source, encoder, history = FakeSource().eval(), self.encoder(), self.history()
        with torch.no_grad(): encoder.output.weight.fill_(.03)
        expected_native = source.original.embedding(history[..., :1])
        residual = encoder(gw15.history_delta(history, encoder.arm))
        report = torch.full_like(expected_native, .2)
        expected_mixed = torch.tanh(expected_native+residual+report+
            source.original.graph((expected_native+residual+report).mean(2, keepdim=True)))
        expected = source.head(source.original.forecast(expected_mixed))
        with gw15.encoding_context(source, encoder, history):
            actual = source(history, incident_data=report)
        self.assertTrue(torch.equal(source.original.seen_incident_input, expected_native+residual))
        self.assertTrue(torch.equal(actual, expected))
        self.assert_clear(source)

    def test_adapter_only_autograd_learns_through_source_without_source_grad_or_update(self):
        torch.manual_seed(19)
        source, encoder, history = FakeSource().eval(), self.encoder(), self.history()/10
        before = copy.deepcopy(source.state_dict())
        parameter_flags = [parameter.requires_grad for parameter in source.parameters()]
        params = tuple(encoder.parameters())
        with gw15.encoding_context(source, encoder, history):
            prediction = source(history)
            loss = prediction.square().sum()+prediction.sum()
        gradients = torch.autograd.grad(loss, params)
        self.assertGreater(float(gradients[-1].abs().sum()), 0.)
        self.assertTrue(all(parameter.grad is None for parameter in source.parameters()))
        with torch.no_grad():
            for parameter, gradient in zip(params, gradients): parameter.add_(gradient, alpha=-.1)
        with gw15.encoding_context(source, encoder, history):
            second = source(history)
            second_loss = second.square().sum()+second.sum()
        gradients = torch.autograd.grad(second_loss, params)
        self.assertGreater(float(gradients[0].abs().sum()), 0.)
        self.assertFalse(torch.equal(second, prediction))
        self.assertEqual(parameter_flags, [parameter.requires_grad for parameter in source.parameters()])
        for name, value in source.state_dict().items(): self.assertTrue(torch.equal(value, before[name]))
        self.assert_clear(source)

    def test_context_preserves_modes_and_original_autograd_path(self):
        source, encoder, history = FakeSource().train(), self.encoder(), self.history().requires_grad_()
        source.original.graph.eval()
        modes = {name: module.training for name, module in source.named_modules()}
        with gw15.encoding_context(source, encoder, history):
            prediction = source(history)
        history_gradient, = torch.autograd.grad(prediction.sum(), (history,))
        self.assertGreater(float(history_gradient[..., 0].abs().sum()), 0.)
        self.assertEqual(modes, {name: module.training for name, module in source.named_modules()})
        self.assert_clear(source)

    def test_source_exception_removes_hook_and_guard_and_preserves_original_exception(self):
        source, encoder, history = FakeSource(), self.encoder(), self.history()
        source.fail = True
        with self.assertRaisesRegex(RuntimeError, 'synthetic source failure'):
            with gw15.encoding_context(source, encoder, history): source(history)
        self.assert_clear(source)
        source.fail = False
        with gw15.encoding_context(source, encoder, history): source(history)
        self.assert_clear(source)

    def test_missing_and_duplicate_embedding_execution_fail_and_cleanup(self):
        source, encoder, history = FakeSource(), self.encoder(), self.history()
        with self.assertRaisesRegex(RuntimeError, 'exactly once'):
            with gw15.encoding_context(source, encoder, history): pass
        self.assert_clear(source)
        source.duplicate = True
        with self.assertRaisesRegex(RuntimeError, 'exactly once'):
            with gw15.encoding_context(source, encoder, history): source(history)
        self.assert_clear(source)

    def test_reentrancy_and_alias_source_context_are_rejected_without_damaging_outer(self):
        source, encoder, history = FakeSource(), self.encoder(), self.history()
        alias = FakeSource(); alias.original = source.original
        with gw15.encoding_context(source, encoder, history):
            with self.assertRaisesRegex(RuntimeError, 'reentrant'):
                with gw15.encoding_context(source, encoder, history): pass
            with self.assertRaisesRegex(RuntimeError, 'active embedding'):
                with gw15.encoding_context(alias, encoder, history): pass
            source(history)
        self.assert_clear(source)
        self.assertFalse(hasattr(alias, gw15._GUARD))

    def test_invalid_context_and_hook_input_contracts_cleanup(self):
        history, encoder = self.history(), self.encoder()
        with self.assertRaises(ValueError):
            with gw15.encoding_context(nn.Linear(1, 4), encoder, history): pass
        source = FakeSource(hidden=5)
        with self.assertRaises(ValueError):
            with gw15.encoding_context(source, encoder, history): pass
        self.assert_clear(source)
        source = FakeSource()
        with self.assertRaises(ValueError):
            with gw15.encoding_context(source, encoder, history.double()): pass
        self.assert_clear(source)
        with self.assertRaisesRegex(ValueError, 'input/output contract'):
            with gw15.encoding_context(source, encoder, history): source(history[:, :-1])
        self.assert_clear(source)

    def test_failed_feature_computation_removes_guards(self):
        source, encoder, history = FakeSource(), self.encoder(), self.history()
        history[0, 0, 0, 0] = float('nan')
        with self.assertRaises(ValueError):
            with gw15.encoding_context(source, encoder, history): pass
        self.assert_clear(source)

    def test_context_computes_encoder_once_diagnostics_are_plain_values(self):
        source, encoder, history = FakeSource(), self.encoder(), self.history()
        encoder.start_diagnostics()
        with gw15.encoding_context(source, encoder, history): source(history)
        record = encoder.diagnostics()
        self.assertEqual((record['complete_forwards'], record['adapter_calls'], record['embedding_hook_calls']), (1, 1, 1))
        self.assertEqual((record['delta_rms'], record['delta_mean'], record['delta_abs_max']), (0., 0., 0.))
        self.assertTrue(all(not torch.is_tensor(value) for value in record.values()))
        self.assertIsNone(encoder._diagnostics)
        self.assert_clear(source)


if __name__ == '__main__':
    unittest.main()
