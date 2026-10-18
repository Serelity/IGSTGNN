"""Server-run recovery, history provenance, gradient and cache-lifetime checks."""
from copy import deepcopy
import unittest

import torch
from torch import nn

from experiments.chronological import train_original_history_anchor as train
from experiments.chronological.original_graph_message import GraphMessageIGSTGNN
from experiments.chronological.original_receiver_residual import (
    LogitResidual, ReceiverResidualIGSTGNN, split_state)
from experiments.chronological.original_history_anchor import (
    ARMS, HistoryLogitResidual, HistoryAnchorIGSTGNN)
from experiments.chronological.verify_original_receiver_residual import FixtureOriginal as OldFixtureOriginal


class FixtureOriginal(OldFixtureOriginal):
    def __init__(self):
        super().__init__('cpu')
        self.seq_len = self.horizon = 12
        self.readout = nn.Linear(4, 3)

    def forward(self, history_data, incident_data=None, sensor_data=None):
        conv = self.layers[0].dif_layer.localized_st_conv
        dynamic = conv.native.get_graph([self.graph3, self.graph4])
        static = [self.graph5]
        history = conv(history_data, dynamic, static)
        values = [history[:, -1:]]
        # Match the authenticated native DifForecast provenance exactly.
        for _ in range(3):
            recent = values[-3:]
            if len(recent) < 3:
                recent = [history_data[:, -(3-len(recent)):], *recent]
            values.append(conv(torch.cat(recent, 1), dynamic, static))
        blocks = self.readout(torch.cat(values, 1))
        return blocks.permute(0, 1, 3, 2).reshape(len(blocks), 12, self.node_num, 1)


def fixture():
    torch.manual_seed(1847)
    anchor = GraphMessageIGSTGNN(FixtureOriginal(), 'POST_MESSAGE').eval()
    with torch.no_grad():
        anchor.selector.router.static_logits.uniform_(-.3, .3)
        anchor.selector.router.score_head.weight.normal_(0., .2)
    return anchor, torch.randn(2, 12, 5, 4), torch.randn(2, 12, 5, 1)


class HistoryAnchorTests(unittest.TestCase):
    def setUp(self):
        train.base.configure_runtime('cpu', 1, 2025)

    def test_initial_recovery_capacity_rng_and_frozen_eval(self):
        anchor, x, target = fixture()
        with torch.no_grad():
            expected = anchor(x)
        hashes = []
        for arm in ARMS:
            torch.manual_seed(93)
            before = train.base.tensor_tree_hash(train.base.rng_state())
            model = HistoryAnchorIGSTGNN(deepcopy(anchor), arm)
            self.assertEqual(before, train.base.tensor_tree_hash(train.base.rng_state()))
            row = train.initialization(model)
            hashes.append((row['frozen_tensor_sha256'], row['added_tensor_sha256'], row['post_construction_rng_sha256']))
            self.assertEqual(row['trainable_parameters'], 159)
            for training in (True, False):
                model.train(training)
                before = train.base.tensor_tree_hash(train.base.rng_state())
                with torch.no_grad():
                    prediction = model(x, label=target*100.)
                self.assertTrue(torch.allclose(prediction, expected, atol=2e-6, rtol=2e-6))
                self.assertEqual(before, train.base.tensor_tree_hash(train.base.rng_state()))
                self.assertFalse(any(module.training for module in model.selector.native.modules()))
                self.assertFalse(model.selector.residual._active)
                self.assertEqual(model.selector.residual._anchors, {})
            self.assertEqual(train.base.tensor_tree_hash(split_state(model.state_dict())[0]), train.base.tensor_tree_hash(anchor.state_dict()))
        self.assertEqual(len(set(hashes)), 1)

    def test_free_intervention_recovers_trained_GW4_and_base_recovers_GW2(self):
        anchor, x, _ = fixture()
        old = ReceiverResidualIGSTGNN(deepcopy(anchor), 'RECEIVER_RESIDUAL').eval()
        with torch.no_grad():
            old.selector.residual.key.weight.normal_(0., .3)
            old.selector.residual.bias.uniform_(-.2, .2)
            expected, base = old(x), anchor(x)
        for arm in ARMS:
            model = HistoryAnchorIGSTGNN(deepcopy(anchor), arm)
            model.load_state_dict(old.state_dict(), strict=True)
            with torch.no_grad(), model.intervention('FREE'):
                free = model(x)
            with torch.no_grad(), model.intervention('BASE'):
                actual_base = model(x)
            self.assertTrue(torch.allclose(free, expected, atol=2e-6, rtol=2e-6))
            self.assertTrue(torch.allclose(actual_base, base, atol=2e-6, rtol=2e-6))
            self.assertIsNone(model.selector.override_mode)
            self.assertEqual(model.selector.residual._anchors, {})

    def test_final_anchor_is_invariant_to_current_evidence_and_retains_history_gradient(self):
        residual = LogitResidual(4, 3, 5, 'RECEIVER_RESIDUAL')
        with torch.no_grad():
            residual.key.weight.normal_(0., .2)
        policy = HistoryLogitResidual(residual, 'HISTORY_ANCHORED')
        message = torch.randn(2, 4, 5, 3, 4, requires_grad=True)
        receiver = torch.randn(2, 4, 5, 4, requires_grad=True)
        policy.begin_forward(); policy.step = 0
        history, _, _ = policy(message, receiver, 0)
        policy.step = 3
        future_message = torch.randn(2, 1, 5, 3, 4, requires_grad=True)
        future_receiver = torch.randn(2, 1, 5, 4, requires_grad=True)
        last, _, _ = policy(future_message, future_receiver, 0)
        changed, _, _ = policy(future_message*7., future_receiver*-3., 0)
        self.assertTrue(torch.equal(last, history[:, -1:]))
        self.assertTrue(torch.equal(changed, last))
        policy.end_forward()
        # Backward AFTER cache cleanup must still reach the history evidence.
        last.square().mean().backward()
        self.assertGreater(float(message.grad.abs().sum()), 0.)
        self.assertGreater(float(receiver.grad.abs().sum()), 0.)
        self.assertEqual(float(message.grad[:, :-1].abs().sum()), 0.)
        self.assertIsNone(future_message.grad)
        self.assertIsNone(future_receiver.grad)

    def test_fixed_history_and_shrink_have_distinct_final_limits(self):
        for arm in ('HISTORY_FIXED', 'PHASE_SHRINK', 'BIAS_SHRINK'):
            residual = LogitResidual(4, 3, 5, 'RECEIVER_RESIDUAL')
            with torch.no_grad():
                residual.key.weight.normal_(0., .3)
                residual.bias.uniform_(-.2, .2)
            policy = HistoryLogitResidual(residual, arm)
            policy.begin_forward(); policy.step = 0
            history, _, _ = policy(torch.randn(2, 4, 5, 3, 4), torch.randn(2, 4, 5, 4), 0)
            policy.step = 3
            last, _, _ = policy(torch.randn(2, 1, 5, 3, 4), torch.randn(2, 1, 5, 4), 0)
            expected = (history[:, -1:] if arm == 'HISTORY_FIXED' else
                residual.bias[0].expand_as(last) if arm == 'BIAS_SHRINK' else torch.zeros_like(last))
            self.assertTrue(torch.equal(last, expected))
            policy.end_forward()

    def test_cache_clears_across_batch_sizes_failures_and_delayed_backward(self):
        anchor, x, target = fixture()
        model = HistoryAnchorIGSTGNN(anchor, 'HISTORY_ANCHORED').train()
        with torch.no_grad():
            model.selector.residual.key.weight.normal_(0., .2)
        first = model(x)
        second = model(x[:1]*2.)
        fresh = deepcopy(model)
        expected = fresh(x[:1]*2.)
        self.assertTrue(torch.allclose(second, expected, atol=2e-6, rtol=2e-6))
        (first-target).square().mean().backward()
        self.assertGreater(float(model.selector.residual.key.weight.grad.abs().sum()), 0.)
        self.assertEqual(model.selector.residual._anchors, {})
        with self.assertRaises(ValueError):
            model(x[:, :-1])
        self.assertFalse(model.selector.residual._active)
        self.assertEqual(model.selector.residual._anchors, {})
        with self.assertRaises(RuntimeError):
            with model.intervention('FREE'):
                with model.intervention('ANCHOR_ROLL'):
                    raise RuntimeError('intentional probe restoration')
        self.assertIsNone(model.selector.override_mode)
        self.assertFalse(any('anchor' in name or 'cache' in name for name in model.state_dict()))

    def test_bias_preserving_control_does_not_shrink_a_constant_bias(self):
        for arm in ARMS:
            residual = LogitResidual(4, 3, 5, 'RECEIVER_RESIDUAL')
            with torch.no_grad():
                residual.bias[0].copy_(torch.tensor([.1, -.2, .3]))
            policy = HistoryLogitResidual(residual, arm)
            policy.begin_forward(); policy.step = 0
            policy(torch.randn(2, 4, 5, 3, 4), torch.randn(2, 4, 5, 4), 0)
            for step in (1, 2, 3):
                policy.step = step
                actual, _, _ = policy(torch.randn(2, 1, 5, 3, 4), torch.randn(2, 1, 5, 4), 0)
                bias = residual.bias[0].expand_as(actual)
                expected = bias*(1.-step/3.) if arm == 'PHASE_SHRINK' else bias
                self.assertTrue(torch.allclose(actual, expected, atol=1e-7, rtol=1e-6))
            policy.end_forward()

    def test_history_key_gradient_matches_finite_difference(self):
        anchor, x, target = fixture()
        model = HistoryAnchorIGSTGNN(anchor, 'HISTORY_ANCHORED').double()
        model.selector.native.pre_defined_graph = [graph.double() for graph in model.selector.native.pre_defined_graph]
        x, target = x.double(), target.double()
        loss = (model(x)-target).square().mean()
        loss.backward()
        parameter = model.selector.residual.key.weight
        index = int(parameter.grad.abs().argmax())
        analytic = float(parameter.grad.view(-1)[index])
        epsilon = 1e-5
        with torch.no_grad():
            parameter.view(-1)[index] += epsilon
            plus = float((model(x)-target).square().mean())
            parameter.view(-1)[index] -= 2*epsilon
            minus = float((model(x)-target).square().mean())
            parameter.view(-1)[index] += epsilon
        numeric = (plus-minus)/(2*epsilon)
        self.assertGreater(abs(analytic), 0.)
        self.assertLessEqual(abs(analytic-numeric), 1e-7+1e-3*abs(analytic))

    def test_diagnostics_distinguish_call_provenance_from_real_horizons(self):
        anchor, x, _ = fixture()
        model = HistoryAnchorIGSTGNN(anchor, 'HISTORY_ANCHORED')
        with torch.no_grad():
            model.selector.residual.key.weight.normal_(0., .3)
            model.start_diagnostics()
            model(x)
        diagnostics = model.diagnostics()
        self.assertEqual(set(diagnostics['policy_by_phase']), {'history', 'forecast_1', 'forecast_2', 'forecast_3'})
        for step, phase in enumerate(('history', 'forecast_1', 'forecast_2', 'forecast_3')):
            for row in diagnostics['policy_by_phase'][phase].values():
                self.assertAlmostEqual(row['generated_input_fraction'], step/3.)
                self.assertAlmostEqual(row['scheduled_lambda'], 1.-step/3.)
                if step == 3:
                    self.assertEqual(row['effective_departure_from_history_abs_mean'], 0.)


if __name__ == '__main__':
    unittest.main()
