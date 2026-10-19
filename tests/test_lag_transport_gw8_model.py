"""Server-only GW8 contracts; synthetic checks are not predictive evidence.

Run on the server: python -m unittest tests.test_lag_transport_gw8_model -v
Do not run model imports, syntax checks, or these tests on the local workstation.
"""
import copy
import unittest
from unittest.mock import patch

import torch

from experiments.chronological.original_graph_message import GraphMessageIGSTGNN
from experiments.chronological.original_receiver_residual import ReceiverResidualIGSTGNN
from experiments.chronological.original_igstgnn_runtime import rng_fingerprint, tensor_tree_hash
from experiments.chronological.original_igstgnn_source import get_original_modules
from experiments.graph_memory_gw6.model import GraphMemoryIGSTGNN
from experiments.lag_transport_gw8.model import (
    ARMS, LagTransportIGSTGNN, RawLagKernel, build_transport_prior)
from tests import test_original_graph_window as fixtures


class LagTransportGW8Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.modules = get_original_modules()

    native_model = fixtures.GraphWindowTests.model
    batch = staticmethod(fixtures.GraphWindowTests.batch)

    def anchor(self):
        gw4 = ReceiverResidualIGSTGNN(
            GraphMessageIGSTGNN(self.native_model(), 'POST_MESSAGE'), 'RECEIVER_RESIDUAL')
        with torch.no_grad():
            gw4.selector.router.static_logits.uniform_(-.4, .4)
            gw4.selector.router.score_head.weight.normal_(std=.1)
            gw4.selector.residual.key.weight.normal_(std=.15)
            gw4.selector.residual.bias.normal_(std=.1)
        return GraphMemoryIGSTGNN(gw4, 'RECURRENT_CONTROL', stage='joint').eval()

    @staticmethod
    def repeated_support(graph, kernel=3):
        """Independent native [receiver, slot*sender] support construction."""
        if graph.ndim == 2:
            return graph[:, None, :].expand(-1, kernel, -1).reshape(graph.shape[0], -1)
        return graph[:, :, None, :].expand(-1, -1, kernel, -1).reshape(
            graph.shape[0], graph.shape[1], -1)

    def test_registered_arms_and_construction_preserve_anchor_tensors_rng_and_reload(self):
        self.assertEqual(ARMS, ('R_CONTINUE', 'R_LOCAL_KERNEL',
            'R_SEPARABLE_KERNEL', 'R_JOINT_KERNEL'))
        anchor = self.anchor()
        before, rng = tensor_tree_hash(anchor.state_dict()), rng_fingerprint()
        for arm in ARMS:
            model = LagTransportIGSTGNN(copy.deepcopy(anchor), arm)
            self.assertEqual(before, tensor_tree_hash(model.anchor.state_dict()))
            self.assertEqual(list(anchor.state_dict()), list(model.anchor.state_dict()))
            self.assertEqual(rng, rng_fingerprint())
            self.assertEqual(model.param_num(), sum(p.numel() for p in model.parameters()))
            restored = LagTransportIGSTGNN(copy.deepcopy(anchor), arm)
            restored.load_state_dict(model.state_dict(), strict=True)
            self.assertEqual(tensor_tree_hash(model.state_dict()),
                tensor_tree_hash(restored.state_dict()))
            model.assert_cache_clear()

    def test_three_kernel_controls_have_identical_active_capacity_and_initial_state(self):
        anchor = self.anchor()
        variants = [LagTransportIGSTGNN(copy.deepcopy(anchor), arm) for arm in ARMS[1:]]
        self.assertIsNone(LagTransportIGSTGNN(copy.deepcopy(anchor), ARMS[0]).kernel)
        reference = variants[0]
        reference_names = [(n, tuple(p.shape), p.requires_grad)
            for n, p in reference.kernel.named_parameters()]
        for model in variants[1:]:
            self.assertEqual(reference.trainable_parameter_counts(), model.trainable_parameter_counts())
            self.assertEqual(reference_names, [(n, tuple(p.shape), p.requires_grad)
                for n, p in model.kernel.named_parameters()])
            self.assertEqual(tensor_tree_hash(reference.kernel.state_dict()),
                tensor_tree_hash(model.kernel.state_dict()))
        self.assertTrue(all(p.requires_grad for model in variants
            for p in model.kernel.parameters()))

    def test_prior_uses_receiver_rows_first_orders_equal_sources_and_reports_topk_loss(self):
        first = torch.tensor([[0., 6., 3., 1.], [1., 99., 6., 3.],
            [0., -4., 9., 0.], [0., 0., 0., 0.]], dtype=torch.float64)
        second = torch.tensor([[7., 1., 2., 7.], [0., 0., 0., 0.],
            [4., 1., 12., 5.], [0., 0., 0., 0.]], dtype=torch.float64)
        ignored = torch.arange(16., dtype=torch.float64).reshape(4, 4)*1000.
        supports = [self.repeated_support(first), self.repeated_support(ignored),
            self.repeated_support(second).expand(2, -1, -1), self.repeated_support(-ignored)]
        indices, prior, stats = build_transport_prior(supports, 2, 4, 3, 2, 2, 2)
        self.assertEqual(tuple(indices.shape), (2, 4, 2))
        self.assertEqual(indices.dtype, torch.long)
        self.assertTrue(bool(torch.isfinite(prior).all()))
        self.assertTrue(bool((prior >= 0).all()))
        dense = torch.zeros(2, 4, 4, dtype=torch.float64).scatter(-1, indices, prior)
        expected = torch.tensor([[0., 7./15., 0., 8./15.],
            [0., 0., 2./3., 1./3.], [4./9., 0., 0., 5./9.],
            [0., 0., 0., 0.]], dtype=torch.float64)
        torch.testing.assert_close(dense, expected.expand(2, -1, -1), atol=1e-14, rtol=1e-14)
        torch.testing.assert_close(stats['retained_mass'],
            torch.tensor([[.75, .9, .9, 0.]], dtype=torch.float64).expand(2, -1),
            atol=1e-14, rtol=1e-14)
        self.assertTrue(torch.equal(stats['nonempty_row_mask'],
            torch.tensor([[True, True, True, False]]).expand(2, -1)))
        self.assertEqual(int(stats['zero_neighbor_rows']), 2)
        # A source contributes a row prior, not its absolute unnormalized weight.
        rescaled = list(supports)
        rescaled[0] = rescaled[0]*1000.
        other_ids, other_prior, _ = build_transport_prior(rescaled, 2, 4, 3, 2, 2, 2)
        self.assertTrue(torch.equal(indices, other_ids))
        torch.testing.assert_close(prior, other_prior, atol=1e-14, rtol=1e-14)

    def test_prior_is_differentiable_for_retained_edges_and_masks_self_and_empty_rows(self):
        graph = torch.tensor([[13., 1., 2., 4.], [1., 7., 3., 2.],
            [3., 4., 19., 1.], [0., 0., 0., 0.]], dtype=torch.float64, requires_grad=True)
        ids, mass, stats = build_transport_prior([self.repeated_support(graph)], 1, 4, 3, 1, 1, 2)
        self.assertTrue(mass.requires_grad)
        self.assertTrue(stats['retained_mass'].requires_grad)
        (mass*torch.tensor([1., 3.], dtype=torch.float64)).sum().backward()
        self.assertGreater(float(graph.grad.abs().sum()), 0.)
        self.assertTrue(torch.equal(graph.grad.diag(), torch.zeros(4, dtype=torch.float64)))
        self.assertTrue(torch.equal(graph.grad[-1], torch.zeros(4, dtype=torch.float64)))
        self.assertFalse(bool(((ids == torch.arange(4)[None, :, None]) & (mass > 0)).any()))

    def test_prior_rejects_invalid_dimensions_counts_and_nonfinite_first_order(self):
        graph = self.repeated_support(torch.ones(3, 3, dtype=torch.float64))
        for supports, b, n, k, s, o, e in (
                ([], 1, 3, 3, 1, 1, 2), ([graph], 1, 3, 3, 1, 2, 2),
                ([graph[:, :-1]], 1, 3, 3, 1, 1, 2), ([graph], 1, 3, 3, 1, 1, 0)):
            with self.assertRaises(ValueError):
                build_transport_prior(supports, b, n, k, s, o, e)
        for invalid in (float('nan'), float('inf')):
            corrupted = graph.clone()
            corrupted[0, 1] = invalid
            with self.assertRaises(ValueError):
                build_transport_prior([corrupted], 1, 3, 3, 1, 1, 2)
        corrupted = graph.clone()
        corrupted[0, 4] += .3
        with self.assertRaises(ValueError):
            build_transport_prior([corrupted], 1, 3, 3, 1, 1, 2)

    @staticmethod
    def small_attention_case():
        coordinates = torch.arange(36., dtype=torch.float64).reshape(1, 2, 3, 3, 2)
        window = torch.sin(coordinates*.73)+torch.cos(coordinates*.19)
        indices = torch.tensor([[[1, 2], [2, 0], [0, 1]]], dtype=torch.long)
        mass = torch.tensor([[[.7, .3], [.4, .6], [0., 0.]]], dtype=torch.float64)
        return window, (indices, mass)

    def test_joint_attention_and_readout_match_independent_sender_slot_reference(self):
        window, prior = self.small_attention_case()
        kernel = RawLagKernel(2, 3, 'R_JOINT_KERNEL', hidden=2, node_chunk=2).double()
        with torch.no_grad():
            for module in (kernel.query, kernel.key, kernel.value, kernel.output):
                module.weight.copy_(torch.eye(2, dtype=torch.float64))
            kernel.lag_bias.copy_(torch.tensor([.2, -.1, .3], dtype=torch.float64))
        expected = torch.zeros(1, 2, 3, 2, 3, dtype=torch.float64)
        expected_output = torch.zeros(1, 2, 3, 2, dtype=torch.float64)
        ids, mass = prior
        for time in range(2):
            for receiver in range(2):  # Third receiver has no positive edge.
                scores = []
                for edge in range(2):
                    sender = int(ids[0, receiver, edge])
                    for slot in range(3):
                        dot = (window[0, time, receiver, -1]*window[0, time, sender, slot]).sum()
                        scores.append(dot/(2.**.5)+kernel.lag_bias[slot]
                            +mass[0, receiver, edge].log())
                expected[0, time, receiver] = torch.softmax(torch.stack(scores), 0).reshape(2, 3)
                for edge in range(2):
                    for slot in range(3):
                        expected_output[0, time, receiver] += (expected[0, time, receiver, edge, slot]
                            *window[0, time, int(ids[0, receiver, edge]), slot])
        actual = kernel.attention_weights(window, prior)
        torch.testing.assert_close(actual, expected, atol=1e-14, rtol=1e-14)
        torch.testing.assert_close(kernel(window, prior), expected_output, atol=1e-14, rtol=1e-14)
        self.assertTrue(torch.equal(actual[:, :, 2], torch.zeros_like(actual[:, :, 2])))

    def test_separable_removes_only_coupling_and_preserves_both_joint_marginals(self):
        window, prior = self.small_attention_case()
        joint = RawLagKernel(2, 3, 'R_JOINT_KERNEL', hidden=2).double()
        with torch.no_grad():
            joint.query.weight.copy_(torch.eye(2, dtype=torch.float64))
            joint.key.weight.copy_(torch.eye(2, dtype=torch.float64))
            joint.output.weight.copy_(torch.eye(2, dtype=torch.float64))
        separable = copy.deepcopy(joint)
        separable.arm = 'R_SEPARABLE_KERNEL'
        p = joint.attention_weights(window, prior)
        q = separable.attention_weights(window, prior)
        independent = p.sum(-1, keepdim=True)*p.sum(-2, keepdim=True)
        torch.testing.assert_close(q, independent, atol=1e-14, rtol=1e-14)
        torch.testing.assert_close(q.sum(-1), p.sum(-1), atol=1e-14, rtol=1e-14)
        torch.testing.assert_close(q.sum(-2), p.sum(-2), atol=1e-14, rtol=1e-14)
        self.assertGreater(float((p-q).abs().max()), 1e-5)
        self.assertTrue(torch.equal(q[:, :, 2], torch.zeros_like(q[:, :, 2])))

    def test_local_kernel_has_only_own_slots_and_never_reads_other_nodes(self):
        window, _ = self.small_attention_case()
        kernel = RawLagKernel(2, 3, 'R_LOCAL_KERNEL', hidden=3).double()
        with torch.no_grad():
            kernel.output.weight.normal_(std=.2)
        probability = kernel.attention_weights(window)
        self.assertEqual(tuple(probability.shape), (1, 2, 3, 1, 3))
        torch.testing.assert_close(probability.sum((-2, -1)),
            torch.ones(1, 2, 3, dtype=torch.float64), atol=1e-14, rtol=1e-14)
        changed = window.clone()
        changed[:, :, 1:] += 7.
        before, after = kernel(window), kernel(changed)
        self.assertTrue(torch.equal(before[:, :, :1], after[:, :, :1]))
        self.assertGreater(float((before[:, :, 1:]-after[:, :, 1:]).abs().sum()), 0.)

    def test_chunking_preserves_outputs_input_graph_and_parameter_gradients(self):
        torch.manual_seed(41)
        window = torch.randn(2, 4, 5, 3, 3, dtype=torch.float64)
        graph = torch.rand(2, 5, 5, dtype=torch.float64)
        ids, mass, _ = build_transport_prior([self.repeated_support(graph)], 2, 5, 3, 1, 1, 3)
        for arm in ARMS[1:]:
            first = RawLagKernel(3, 3, arm, hidden=4, node_chunk=2).double()
            with torch.no_grad():
                first.output.weight.normal_(std=.1)
            second = copy.deepcopy(first)
            second.node_chunk = 100
            xa, xb = window.clone().requires_grad_(), window.clone().requires_grad_()
            ma, mb = mass.detach().clone().requires_grad_(), mass.detach().clone().requires_grad_()
            ya = first(xa, None if arm == 'R_LOCAL_KERNEL' else (ids, ma))
            yb = second(xb, None if arm == 'R_LOCAL_KERNEL' else (ids, mb))
            torch.testing.assert_close(ya, yb, atol=1e-12, rtol=1e-12)
            ya.square().sum().backward()
            yb.square().sum().backward()
            torch.testing.assert_close(xa.grad, xb.grad, atol=1e-11, rtol=1e-11)
            if arm != 'R_LOCAL_KERNEL':
                self.assertIsNotNone(ma.grad)
                torch.testing.assert_close(ma.grad, mb.grad, atol=1e-11, rtol=1e-11)
            for pa, pb in zip(first.parameters(), second.parameters()):
                self.assertIsNotNone(pa.grad)
                torch.testing.assert_close(pa.grad, pb.grad, atol=1e-10, rtol=1e-10)

    def test_empty_graph_rows_remain_exact_zero_after_output_weights_become_nonzero(self):
        window, _ = self.small_attention_case()
        for arm in ('R_SEPARABLE_KERNEL', 'R_JOINT_KERNEL'):
            kernel = RawLagKernel(2, 3, arm, hidden=3).double()
            with torch.no_grad():
                kernel.output.weight.normal_()
            ids = torch.tensor([[[1, 2], [2, 0], [0, 1]]], dtype=torch.long)
            zero = torch.zeros(1, 3, 2, dtype=torch.float64)
            actual = kernel(window, (ids, zero))
            self.assertTrue(torch.equal(actual, torch.zeros_like(actual)))
            self.assertTrue(bool(torch.isfinite(actual).all()))
            actual.sum().backward()
            self.assertTrue(all(bool(torch.isfinite(p.grad).all())
                for p in kernel.parameters() if p.grad is not None))
        singleton = window[:, :, :1]
        ids, mass, _ = build_transport_prior([self.repeated_support(
            torch.ones(1, 1, dtype=torch.float64))], 1, 1, 3, 1, 1, 8)
        self.assertEqual(tuple(ids.shape), (1, 1, 0))
        self.assertTrue(torch.equal(kernel(singleton, (ids, mass)),
            torch.zeros(1, 2, 1, 2, dtype=torch.float64)))

    def test_raw_kernel_rejects_invalid_history_and_invalid_sender_prior(self):
        window, prior = self.small_attention_case()
        kernel = RawLagKernel(2, 3, 'R_JOINT_KERNEL', hidden=2).double()
        ids, mass = prior
        with self.assertRaises(ValueError):
            kernel(window)
        for invalid_ids, invalid_mass in ((ids+3, mass), (ids.to(torch.float64), mass),
                (ids, -mass), (ids, mass[..., :1])):
            with self.assertRaises(ValueError):
                kernel(window, (invalid_ids, invalid_mass))
        bad_window = window.clone()
        bad_window[0, 0, 0, 0, 0] = float('nan')
        for malformed in (window.squeeze(0), window[..., :1], bad_window):
            with self.assertRaises(ValueError):
                kernel(malformed, prior)

    def test_zero_output_recovers_r_exactly_in_train_eval_with_and_without_autograd(self):
        anchor = self.anchor()
        x, report = self.batch()
        inputs_before = tensor_tree_hash((x, report))
        for arm in ARMS:
            for training in (False, True):
                reference = copy.deepcopy(anchor).train(training)
                model = LagTransportIGSTGNN(copy.deepcopy(anchor), arm).train(training)
                self.assertEqual({n: p.requires_grad for n, p in reference.named_parameters()},
                    {n: p.requires_grad for n, p in model.anchor.named_parameters()})
                self.assertEqual({n: m.training for n, m in reference.named_modules()},
                    {n: m.training for n, m in model.anchor.named_modules()})
                for grad_enabled in (False, True):
                    with self.subTest(arm=arm, training=training, grad_enabled=grad_enabled):
                        with torch.set_grad_enabled(grad_enabled):
                            torch.manual_seed(379)
                            expected = reference(x, incident_data=report)
                            expected_rng = rng_fingerprint()
                            torch.manual_seed(379)
                            actual = model(x, incident_data=report)
                        self.assertTrue(torch.equal(actual, expected))
                        self.assertEqual(expected_rng, rng_fingerprint())
                        model.assert_cache_clear()
        self.assertEqual(inputs_before, tensor_tree_hash((x, report)))

    def test_construction_preserves_training_mode_for_every_nested_native_module(self):
        for training in (False, True):
            anchor = self.anchor().train(training)
            modes = {name: module.training for name, module in anchor.named_modules()}
            for arm in ARMS:
                model = LagTransportIGSTGNN(copy.deepcopy(anchor), arm)
                self.assertEqual(model.training, training)
                self.assertEqual(modes,
                    {name: module.training for name, module in model.anchor.named_modules()})

    def test_true_pre_linear_slots_are_read_on_every_first_dif_call(self):
        model = LagTransportIGSTGNN(self.anchor(), 'R_JOINT_KERNEL').eval()
        x, report = self.batch()
        raw, selected, transformed = [], [], []
        conv = model.anchor.anchor.selector
        handles = [
            conv.register_forward_pre_hook(lambda module, inputs: raw.append(inputs[0])),
            model.kernel.register_forward_pre_hook(lambda module, inputs: selected.append(inputs[0])),
            conv.native.fc_list_updt.register_forward_hook(
                lambda module, inputs, output: transformed.append(conv.native.activation(output))),
        ]
        try:
            model(x, incident_data=report)
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(len(raw), 4)  # One history call plus three native forecast calls.
        self.assertEqual(len(selected), 4)
        self.assertEqual(len(transformed), 4)
        self.assertEqual([v.shape[1] for v in selected], [10, 1, 1, 1])
        for before_fc, observed, mixed in zip(raw, selected, transformed):
            expected = before_fc.unfold(1, 3, 1).permute(0, 1, 2, 4, 3)
            self.assertTrue(torch.equal(observed, expected))
            self.assertTrue(observed.requires_grad)
            self.assertEqual(observed.shape, mixed.reshape_as(observed).shape)
            self.assertFalse(torch.equal(observed, mixed.reshape_as(observed)))
        model.assert_cache_clear()

    def test_nonzero_kernel_is_added_before_the_existing_native_dropout(self):
        model = LagTransportIGSTGNN(self.anchor(), 'R_JOINT_KERNEL').eval()
        with torch.no_grad():
            model.kernel.output.weight.normal_(std=.1)
        gcn, delta, dropout_input = [], [], []
        native = model.anchor.anchor.selector.native
        handles = [
            native.gcn_updt.register_forward_hook(lambda module, inputs, output: gcn.append(output)),
            model.kernel.register_forward_hook(lambda module, inputs, output: delta.append(output)),
            native.dropout.register_forward_pre_hook(lambda module, inputs: dropout_input.append(inputs[0])),
        ]
        x, report = self.batch()
        try:
            model(x, incident_data=report)
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual((len(gcn), len(delta), len(dropout_input)), (4, 4, 4))
        self.assertGreater(sum(float(v.abs().sum()) for v in delta), 0.)
        for ordinary, correction, actual in zip(gcn, delta, dropout_input):
            torch.testing.assert_close(actual, ordinary+correction, atol=0., rtol=0.)
        model.assert_cache_clear()

    def test_diagnostics_observe_one_graph_construction_and_all_four_native_calls(self):
        model = LagTransportIGSTGNN(self.anchor(), 'R_JOINT_KERNEL').eval()
        x, report = self.batch()
        graph_calls = []
        handle = model.original.dynamic_graph_constructor.register_forward_pre_hook(
            lambda module, inputs: graph_calls.append(True))
        model.start_diagnostics()
        try:
            result = model(x, incident_data=report)
        finally:
            handle.remove()
        stats = model.diagnostics()
        self.assertEqual(tuple(result.shape), (2, 12, 4, 1))
        self.assertEqual(len(graph_calls), 1)
        self.assertEqual(stats['complete_forwards'], 1)
        self.assertEqual(stats['failed_forwards'], 0)
        self.assertEqual(stats['transport_calls'], 4)
        self.assertEqual(stats['history_calls'], 1)
        self.assertEqual(stats['forecast_calls'], 3)
        self.assertEqual(stats['hidden_update_abs_max'], 0.)
        self.assertTrue(stats['cache_clear'])
        model.assert_cache_clear()

    def test_native_forecasts_remain_trainable_and_new_head_then_interior_receive_gradients(self):
        x, report = self.batch()
        target = torch.linspace(.1, 1., 12)[None, :, None, None]
        for arm in ARMS:
            with self.subTest(arm=arm):
                model = LagTransportIGSTGNN(self.anchor(), arm).eval()
                self.assertTrue(all(p.requires_grad for p in model.anchor.parameters()))
                (model(x, incident_data=report)-target).square().mean().backward()
                for layer in model.original.layers:
                    branch = list(layer.dif_layer.forecast_branch.parameters())
                    self.assertTrue(all(p.requires_grad for p in branch))
                    self.assertGreater(sum(float(p.grad.abs().sum()) for p in branch
                        if p.grad is not None), 0.)
                if model.kernel is not None:
                    self.assertGreater(float(model.kernel.output.weight.grad.abs().sum()), 0.)
                    with torch.no_grad():
                        model.kernel.output.weight.normal_(std=.1)
                    model.zero_grad(set_to_none=True)
                    (model(x, incident_data=report)-target).square().mean().backward()
                    interior = [p for name, p in model.kernel.named_parameters()
                        if not name.startswith('output.') and p.grad is not None]
                    self.assertTrue(interior)
                    self.assertGreater(sum(float(p.grad.abs().sum()) for p in interior), 0.)
                self.assertTrue(all(bool(torch.isfinite(p.grad).all())
                    for p in model.parameters() if p.grad is not None))
                model.assert_cache_clear()

    def test_labels_never_affect_nonzero_kernel_prediction_and_inputs_are_unchanged(self):
        x, report = self.batch()
        target = torch.randn(2, 12, 4, 1)
        before = tensor_tree_hash((x, report, target))
        for arm in ARMS[1:]:
            model = LagTransportIGSTGNN(self.anchor(), arm).eval()
            with torch.no_grad():
                model.kernel.output.weight.normal_(std=.1)
            first = model(x, label=target, incident_data=report)
            second = model(x, label=target*1e6, incident_data=report)
            self.assertTrue(torch.equal(first, second))
            self.assertEqual(before, tensor_tree_hash((x, report, target)))
            model.assert_cache_clear()

    def test_errors_clear_temporary_hooks_and_caches_and_next_forward_is_fresh(self):
        model = LagTransportIGSTGNN(self.anchor(), 'R_JOINT_KERNEL').eval()
        x, report = self.batch()
        native = model.anchor.anchor.selector.native
        watched = (model.anchor.anchor.selector, native.gcn_updt, model.original.embedding)
        initial = [(len(m._forward_pre_hooks), len(m._forward_hooks)) for m in watched]
        for module in (model.kernel, native.gcn_updt, native.dropout, model.original.out_fc_2):
            with patch.object(module, 'forward', side_effect=RuntimeError('deliberate exception')):
                with self.assertRaisesRegex(RuntimeError, 'deliberate exception'):
                    model(x, incident_data=report)
            model.assert_cache_clear()
            self.assertEqual(initial,
                [(len(m._forward_pre_hooks), len(m._forward_hooks)) for m in watched])
        expected = model(x, incident_data=report)
        model(x*2., incident_data=report)
        self.assertTrue(torch.equal(expected, model(x, incident_data=report)))
        model.assert_cache_clear()

    def test_reentrant_forward_is_rejected_without_destroying_enclosing_forward(self):
        model = LagTransportIGSTGNN(self.anchor(), 'R_JOINT_KERNEL').eval()
        x, report = self.batch()
        observed = []
        def try_nested(module, inputs):
            if observed:
                return
            observed.append(True)
            with self.assertRaisesRegex(RuntimeError, '[Rr]eentrant'):
                model(x, incident_data=report)
        handle = model.original.embedding.register_forward_pre_hook(try_nested)
        try:
            result = model(x, incident_data=report)
        finally:
            handle.remove()
        self.assertEqual(tuple(result.shape), (2, 12, 4, 1))
        self.assertEqual(observed, [True])
        model.assert_cache_clear()


if __name__ == '__main__':
    unittest.main()
