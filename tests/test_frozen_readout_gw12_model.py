"""Server-only small-tensor contracts; these tests do not establish model gains.

Do not execute Python, imports, syntax checks or these tests on the workstation.
"""
from math import pi
import unittest

import torch
from torch import nn

from experiments.frozen_readout_gw12 import model as gw12


class FrozenReadoutGW12ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    @staticmethod
    def batch(nodes=3):
        history = torch.ones(2, 12, nodes, 3)
        history[..., 0] = torch.arange(12, dtype=torch.float32)[None, :, None]/3+1
        history[..., 1] = .25
        history[..., 2] = 2/7
        emb_u = torch.arange(nodes*3, dtype=torch.float32).reshape(nodes, 3)/10
        emb_d = torch.arange(nodes*2, dtype=torch.float32).reshape(nodes, 2)/7
        return history, emb_u, emb_d

    def features(self, arm='STATE_READOUT'):
        return gw12.origin_features(*self.batch(), arm)

    def test_only_two_registered_arms_and_fixed_head_width(self):
        self.assertEqual(gw12.ARMS, ('STATIC_READOUT', 'STATE_READOUT'))
        for arm in ('SUM_CONTINUE', 'STATE_GATE', '', None):
            with self.subTest(arm=arm), self.assertRaises(ValueError):
                gw12.FrozenReadoutHead(arm)
        with self.assertRaises(ValueError): gw12.FrozenReadoutHead('STATE_READOUT', hidden=8)

    def test_exact_parameter_count_and_source_is_not_registered(self):
        for arm in gw12.ARMS:
            head = gw12.FrozenReadoutHead(arm)
            self.assertEqual(head.param_num(), 725)
            self.assertEqual(len(head.adapters), 5)
            self.assertEqual(sum(p.numel() for p in head.parameters() if p.requires_grad), 725)
            self.assertEqual(set(head._modules), {'adapters'})
            self.assertFalse(hasattr(head, 'source'))
            for adapter in head.adapters:
                self.assertEqual(sum(p.numel() for p in adapter.parameters()), 145)
                self.assertIsInstance(adapter.activation, nn.Tanh)

    def test_zero_output_and_exact_source_recovery_in_train_and_eval(self):
        features = self.features()
        source = torch.linspace(-10, 10, features[..., :1].numel()).reshape(*features.shape[:3], 1)
        for arm in gw12.ARMS:
            for mode in (False, True):
                head = gw12.FrozenReadoutHead(arm).train(mode)
                delta = head(features)
                self.assertTrue(torch.equal(delta, torch.zeros_like(delta)))
                self.assertTrue(torch.equal(source+delta, source))
                for adapter in head.adapters:
                    self.assertTrue(torch.equal(adapter.output.weight, torch.zeros_like(adapter.output.weight)))
                    self.assertTrue(torch.equal(adapter.output.bias, torch.zeros_like(adapter.output.bias)))

    def test_paired_initialization_and_cpu_rng_are_identical(self):
        before = torch.get_rng_state().clone()
        first = gw12.FrozenReadoutHead('STATIC_READOUT')
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        second = gw12.FrozenReadoutHead('STATE_READOUT')
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        for name, value in first.state_dict().items():
            self.assertTrue(torch.equal(value, second.state_dict()[name]))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA RNG contract requires server CUDA')
    def test_constructor_preserves_initialized_cuda_rng(self):
        cuda_before = [state.clone() for state in torch.cuda.get_rng_state_all()]
        cpu_before = torch.get_rng_state().clone()
        gw12.FrozenReadoutHead('STATIC_READOUT')
        gw12.FrozenReadoutHead('STATE_READOUT')
        self.assertTrue(torch.equal(cpu_before, torch.get_rng_state()))
        for before, after in zip(cuda_before, torch.cuda.get_rng_state_all()):
            self.assertTrue(torch.equal(before, after))

    def test_state_columns_and_unchanged_source_normalization(self):
        history, emb_u, emb_d = self.batch()
        features = gw12.origin_features(history, emb_u, emb_d, 'STATE_READOUT')
        self.assertEqual(tuple(features.shape), (2, 12, 3, 7))
        torch.testing.assert_close(features[..., 0], history[:, -1, :, 0][:, None].expand(2, 12, 3))
        expected = torch.log1p((history[:, 1:, :, 0]-history[:, :-1, :, 0]).abs().mean(1))
        torch.testing.assert_close(features[..., 1], expected[:, None].expand(2, 12, 3))
        torch.testing.assert_close(features[..., 2], torch.ones(2, 12, 3))
        torch.testing.assert_close(features[..., 3], torch.zeros(2, 12, 3), atol=1e-6, rtol=0)
        torch.testing.assert_close(features[..., 4], torch.full((2, 12, 3), torch.sin(torch.tensor(4*pi/7)).item()))
        torch.testing.assert_close(features[..., 5], torch.full((2, 12, 3), torch.cos(torch.tensor(4*pi/7)).item()))
        torch.testing.assert_close(features[0, :, 0, 6], torch.arange(12)/11)

    def test_static_columns_and_traffic_invariance(self):
        history, emb_u, emb_d = self.batch()
        before = gw12.origin_features(history, emb_u, emb_d, 'STATIC_READOUT')
        torch.testing.assert_close(before[..., 0], emb_u.mean(-1)[None, None].expand(2, 12, 3))
        torch.testing.assert_close(before[..., 1], emb_d.mean(-1)[None, None].expand(2, 12, 3))
        altered = history.clone(); altered[..., 0] *= 17
        self.assertTrue(torch.equal(before, gw12.origin_features(altered, emb_u, emb_d, 'STATIC_READOUT')))
        self.assertFalse(torch.equal(self.features(), gw12.origin_features(altered, emb_u, emb_d, 'STATE_READOUT')))
        self.assertTrue(torch.equal(before[..., 2:], self.features()[..., 2:]))

    def test_state_uses_change_even_if_last_flow_is_unchanged(self):
        history, emb_u, emb_d = self.batch()
        changed = history.clone(); changed[:, :-1, :, 0] *= -5
        before = gw12.origin_features(history, emb_u, emb_d, 'STATE_READOUT')
        after = gw12.origin_features(changed, emb_u, emb_d, 'STATE_READOUT')
        self.assertTrue(torch.equal(before[..., 0], after[..., 0]))
        self.assertFalse(torch.equal(before[..., 1], after[..., 1]))
        self.assertTrue(torch.equal(before[..., 2:], after[..., 2:]))

    def test_clocks_use_only_last_observed_position(self):
        history, emb_u, emb_d = self.batch()
        changed = history.clone(); changed[:, :-1, :, 1:] += 6
        for arm in gw12.ARMS:
            before = gw12.origin_features(history, emb_u, emb_d, arm)
            self.assertTrue(torch.equal(before, gw12.origin_features(changed, emb_u, emb_d, arm)))
            moved = history.clone(); moved[:, -1, :, 1] += .125
            self.assertFalse(torch.equal(before[..., 2:4], gw12.origin_features(moved, emb_u, emb_d, arm)[..., 2:4]))

    def test_features_and_forward_prevent_history_embedding_and_feature_gradients(self):
        for arm in gw12.ARMS:
            history, emb_u, emb_d = (value.requires_grad_() for value in self.batch())
            features = gw12.origin_features(history, emb_u, emb_d, arm)
            self.assertFalse(features.requires_grad)
            self.assertIsNone(features.grad_fn)
            supplied = features.clone().requires_grad_()
            head = gw12.FrozenReadoutHead(arm)
            for adapter in head.adapters:
                with torch.no_grad(): adapter.output.weight.fill_(.1)
            head(supplied).sum().backward()
            for value in (history, emb_u, emb_d, supplied): self.assertIsNone(value.grad)
            self.assertTrue(all(parameter.grad is not None for parameter in head.parameters()))

    def test_head_learns_without_holding_or_modifying_source_tensors(self):
        history, emb_u, emb_d = self.batch()
        snapshots = [value.clone() for value in (history, emb_u, emb_d)]
        head = gw12.FrozenReadoutHead('STATE_READOUT')
        features = gw12.origin_features(history, emb_u, emb_d, head.arm)
        optimizer = torch.optim.SGD(head.parameters(), lr=.01)
        before = head(features).detach().clone()
        for _ in range(2):
            optimizer.zero_grad()
            (head(features)-.5).square().mean().backward()
            optimizer.step()
        self.assertFalse(torch.equal(before, head(features)))
        self.assertLess(float((head(features)-.5).square().mean()), .25)
        self.assertTrue(any(adapter.input.weight.grad.abs().sum() > 0 for adapter in head.adapters))
        for current, snapshot in zip((history, emb_u, emb_d), snapshots):
            self.assertTrue(torch.equal(current, snapshot))

    def test_average_of_five_adapters_and_output_shape(self):
        head = gw12.FrozenReadoutHead('STATE_READOUT')
        for index, adapter in enumerate(head.adapters):
            with torch.no_grad(): adapter.output.bias.fill_(index+1)
        actual = head(self.features())
        self.assertEqual(tuple(actual.shape), (2, 12, 3, 1))
        self.assertTrue(torch.equal(actual, torch.full_like(actual, 3)))

    def test_diagnostics_aggregate_mean_rms_calls_and_reset(self):
        head = gw12.FrozenReadoutHead('STATE_READOUT')
        for adapter in head.adapters:
            with torch.no_grad(): adapter.output.bias.fill_(-2)
        head.start_diagnostics()
        head(self.features()); head(self.features()[:1])
        actual = head.diagnostics()
        self.assertEqual(actual['adapter_calls'], [2]*5)
        self.assertEqual(actual['complete_forwards'], 2)
        self.assertEqual(actual['failed_forwards'], 0)
        self.assertEqual(actual['readout_delta']['elements'], 3*12*3)
        self.assertEqual(actual['delta_mean'], -2)
        self.assertEqual(actual['delta_rms'], 2)
        self.assertEqual(actual['units'], 'normalized_forecast')
        self.assertIsNone(head.diagnostics()['delta_rms'])

    def test_invalid_history_embeddings_and_feature_axes_are_rejected(self):
        history, emb_u, emb_d = self.batch()
        for bad in (history[:, :11], history[:, :, :0], history[..., :2], history.long(), history[:0]):
            with self.subTest(shape=tuple(bad.shape)), self.assertRaises(ValueError):
                gw12.origin_features(bad, emb_u, emb_d, 'STATE_READOUT')
        for bad in (emb_u[:2], emb_u.double(), emb_u[:, :0], emb_u[None]):
            with self.subTest(shape=tuple(bad.shape)), self.assertRaises(ValueError):
                gw12.origin_features(history, bad, emb_d, 'STATIC_READOUT')
        head = gw12.FrozenReadoutHead('STATE_READOUT')
        for bad in (self.features()[:, :11], self.features()[..., :6], self.features().double(), self.features()[:0]):
            with self.subTest(shape=tuple(bad.shape)), self.assertRaises(ValueError): head(bad)

    def test_nonfinite_inputs_fail_and_diagnostics_record_failure(self):
        history, emb_u, emb_d = self.batch()
        history[0, 0, 0, 0] = float('nan')
        with self.assertRaises(ValueError): gw12.origin_features(history, emb_u, emb_d, 'STATE_READOUT')
        features = self.features(); features[0, 0, 0, 0] = float('inf')
        head = gw12.FrozenReadoutHead('STATE_READOUT'); head.start_diagnostics()
        with self.assertRaises(ValueError): head(features)
        diagnostics = head.diagnostics()
        self.assertEqual(diagnostics['failed_forwards'], 1)
        self.assertEqual(diagnostics['adapter_calls'], [0]*5)


if __name__ == '__main__':
    unittest.main()
