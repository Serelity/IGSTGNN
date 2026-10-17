"""GW3 contracts; run on the server, not as local development checks."""
import copy
import itertools
import unittest

import torch

from experiments.chronological.original_receiver_match import (
    ARMS, MatchingRouter, ReceiverMatchConv, build_variant, split_state)
from experiments.chronological.original_igstgnn_runtime import tensor_tree_hash, rng_fingerprint
from experiments.chronological.original_igstgnn_source import get_original_modules
from tests import test_original_graph_window as fixtures


class ReceiverMatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.modules = get_original_modules()

    model = fixtures.GraphWindowTests.model
    batch = staticmethod(fixtures.GraphWindowTests.batch)

    def test_initial_recovery_native_tensors_rng_inputs_and_both_modes(self):
        original = self.model()
        x, incident = self.batch()
        before = tensor_tree_hash((x, incident))
        for arm, training in itertools.product(ARMS, (False, True)):
            with self.subTest(arm=arm, training=training):
                native = copy.deepcopy(original).train(training)
                candidate = build_variant(copy.deepcopy(original), arm).train(training)
                torch.manual_seed(83)
                expected = native(x, incident_data=incident)
                rng = rng_fingerprint()
                torch.manual_seed(83)
                actual = candidate(x, incident_data=incident)
                torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
                self.assertEqual(rng, rng_fingerprint())
                self.assertEqual(tensor_tree_hash(native.state_dict()), tensor_tree_hash(split_state(candidate)[0]))
                self.assertEqual(before, tensor_tree_hash((x, incident)))

    def test_matching_capacity_initialization_and_only_first_dif_changes(self):
        native = self.model()
        hashes, counts = [], []
        rng = rng_fingerprint()
        for arm in ARMS:
            model = build_variant(copy.deepcopy(native), arm)
            hashes.append(tensor_tree_hash(split_state(model)[1]))
            self.assertEqual(rng, rng_fingerprint())
            self.assertIsInstance(model.selector, ReceiverMatchConv)
            self.assertTrue(all(not isinstance(layer.dif_layer.localized_st_conv, ReceiverMatchConv)
                for layer in model.original.layers[1:]))
            self.assertEqual(model.param_num(), sum(p.numel() for p in model.parameters()))
            counts.append(sum(p.numel() for p in model.selector.router.parameters() if p.requires_grad))
        self.assertEqual(len(set(hashes)), 1)
        self.assertEqual(counts[0], 0)
        self.assertEqual(counts[1], counts[2])
        router = MatchingRouter(32, 3, 5, hidden=16)
        self.assertEqual(sum(p.numel() for p in router.parameters()), 1055)
        self.assertIsNone(router.key.bias)

    def test_query_changes_receiver_selection_with_messages_held_fixed(self):
        router = MatchingRouter(2, 3, 1, hidden=2)
        with torch.no_grad():
            router.query.weight.copy_(torch.eye(2))
            router.query.bias.zero_()
            router.key.weight.copy_(torch.eye(2))
        message = torch.tensor([[[[[1., 0.], [0., 1.], [0., 0.]]]]])
        a = torch.tensor([1., 0.]).expand_as(message)
        b = torch.tensor([0., 1.]).expand_as(message)
        self.assertFalse(torch.allclose(router(a, message, 0, 'RECEIVER_MATCH'), router(b, message, 0, 'RECEIVER_MATCH')))
        self.assertTrue(torch.equal(router(a, message, 0, 'MESSAGE_SELF'), router(b, message, 0, 'MESSAGE_SELF')))
        self.assertTrue(torch.equal(router(a, message, 0, 'QUERY_SELF'), router(a, message, 0, 'MESSAGE_SELF')))

    def test_dot_product_reference_softmax_axis_static_and_uniform_limits(self):
        router = MatchingRouter(4, 3, 2, hidden=3).double()
        local, message = (torch.randn(2, 5, 4, 3, 4, dtype=torch.float64) for _ in range(2))
        self.assertTrue(torch.equal(router(local, message, 1, 'RECEIVER_MATCH'), torch.ones_like(message[..., 0])))
        with torch.no_grad():
            router.key.weight.normal_(std=.2)
            router.static_logits.normal_(std=.2)
        score = (router.query(local.mean(-2)).unsqueeze(-2)*router.key(message)).sum(-1)/(3.**.5)
        expected = 3*torch.softmax(score+router.static_logits[1], -1)
        actual = router(local, message, 1, 'RECEIVER_MATCH')
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual.sum(-1), torch.full_like(actual[..., 0], 3.))
        self.assertTrue(bool((actual >= 0).all() and (actual <= 3).all()))
        self.assertTrue(torch.equal(router(local, message, 1, 'STATIC'), router(local*20, message+9, 1, 'STATIC')))
        self.assertTrue(torch.equal(router(local, message, 1, 'FIXED_UNIFORM'), torch.ones_like(actual)))

    def test_native_mae_learns_key_then_query_and_fixed_router_stays_inactive(self):
        x, incident = self.batch()
        target = torch.linspace(1., 200., 96).reshape(2, 12, 4, 1)
        for arm in ARMS:
            model = build_variant(self.model(), arm).eval()
            router = model.selector.router
            loss = self.modules.metrics.masked_mae(model(x, incident_data=incident)*30+50, target, target.new_tensor(0.))
            loss.backward()
            if arm == 'FIXED_UNIFORM':
                self.assertTrue(all(p.grad is None for p in router.parameters()))
                continue
            self.assertGreater(float(router.key.weight.grad.abs().sum()), 0.)
            self.assertEqual(float(router.query.weight.grad.abs().sum()), 0.)
            optimizer = torch.optim.Adam(model.parameters(), lr=.002)
            optimizer.step()
            model.zero_grad(set_to_none=True)
            self.modules.metrics.masked_mae(model(x, incident_data=incident)*30+50, target, target.new_tensor(0.)).backward()
            self.assertGreater(float(router.query.weight.grad.abs().sum()), 0.)
            self.assertTrue(all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None))

    def test_uniform_control_has_same_native_gradients_as_uniform_intervention(self):
        native = self.model()
        a, b = (build_variant(copy.deepcopy(native), arm).eval() for arm in ('FIXED_UNIFORM', 'RECEIVER_MATCH'))
        x, incident = self.batch()
        a(x, incident_data=incident).square().mean().backward()
        with b.intervention('UNIFORM'):
            b(x, incident_data=incident).square().mean().backward()
        for (name, p), (other, q) in zip(a.original.named_parameters(), b.original.named_parameters()):
            self.assertEqual(name, other)
            if '.router.' not in name:
                if p.grad is None:
                    self.assertIsNone(q.grad)
                else:
                    torch.testing.assert_close(p.grad, q.grad)

    def test_no_label_input_intervention_restoration_and_phase_diagnostics(self):
        model = build_variant(self.model(), 'RECEIVER_MATCH').eval()
        with torch.no_grad():
            model.selector.router.key.weight.normal_(std=.1)
        x, incident = self.batch()
        a = model(x, torch.zeros(2, 12, 4, 1), incident_data=incident)
        self.assertTrue(torch.equal(a, model(x, torch.ones(2, 12, 4, 1)*1e6, incident_data=incident)))
        with self.assertRaisesRegex(RuntimeError, 'probe'):
            with model.intervention('QUERY_SELF'):
                model(x, incident_data=incident)
                raise RuntimeError('probe')
        self.assertIsNone(model.selector.override_mode)
        self.assertTrue(torch.equal(a, model(x, incident_data=incident)))
        model.start_diagnostics()
        model(x, incident_data=incident)
        stats = model.diagnostics()
        self.assertEqual(set(stats['by_phase']), {'history', 'forecast_1', 'forecast_2', 'forecast_3'})
        self.assertLess(stats['coefficient_mass_error_abs_max'], 1e-5)
        self.assertEqual(model.diagnostics()['calls'], 0)

    def test_invalid_mode_axes_and_double_wrapping_are_rejected(self):
        with self.assertRaises(ValueError):
            build_variant(self.model(), 'ORIGINAL')
        with self.assertRaises(ValueError):
            build_variant(build_variant(self.model(), 'RECEIVER_MATCH').original, 'MESSAGE_SELF')
        router = MatchingRouter(4, 3, 2)
        with self.assertRaises(ValueError):
            router(torch.ones(2, 3, 4), torch.ones(2, 3, 4), 0, 'RECEIVER_MATCH')


if __name__ == '__main__':
    unittest.main()
