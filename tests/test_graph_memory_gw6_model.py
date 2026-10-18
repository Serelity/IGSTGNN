"""Server-only GW6 scientific/engineering contracts, using synthetic inputs.

Run ``python -m unittest tests.test_graph_memory_gw6_model -v`` on the server.
These checks do not train on the release or provide predictive evidence.
"""
import copy
import unittest
from unittest.mock import patch

import torch

from experiments.chronological.original_graph_message import GraphMessageIGSTGNN
from experiments.chronological.original_receiver_residual import ReceiverResidualIGSTGNN
from experiments.chronological.original_igstgnn_runtime import rng_fingerprint, tensor_tree_hash
from experiments.chronological.original_igstgnn_source import get_original_modules
from experiments.graph_memory_gw6.model import (
    ARMS, DEFAULT_ARMS, GraphMemoryDecoder, GraphMemoryIGSTGNN, history_graph_memory)
from tests import test_original_graph_window as fixtures


class GraphMemoryGW6Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.modules = get_original_modules()

    native_model = fixtures.GraphWindowTests.model
    batch = staticmethod(fixtures.GraphWindowTests.batch)

    def anchor(self):
        model = ReceiverResidualIGSTGNN(
            GraphMessageIGSTGNN(self.native_model(), 'POST_MESSAGE'), 'RECEIVER_RESIDUAL')
        # Exercise learned weights instead of relying on a uniform/zero case.
        with torch.no_grad():
            model.selector.router.static_logits.uniform_(-.4, .4)
            model.selector.router.score_head.weight.normal_(std=.1)
            model.selector.residual.key.weight.normal_(std=.15)
            model.selector.residual.bias.normal_(std=.1)
        return model.eval()

    def assert_exact_prediction(self, actual, expected, context):
        """Keep bitwise regression strict and make server failures actionable."""
        self.assertEqual(actual.shape, expected.shape, context)
        self.assertEqual(actual.dtype, expected.dtype, context)
        self.assertEqual(actual.device, expected.device, context)
        if torch.equal(actual, expected):
            return
        actual, expected = actual.detach(), expected.detach()
        different = actual != expected
        first = tuple(int(i) for i in different.nonzero()[0])
        finite = torch.isfinite(actual) & torch.isfinite(expected)
        error = (actual.double()-expected.double()).abs()
        self.fail(f'{context}: bitwise prediction mismatch; torch={torch.__version__}, '
            f'device={actual.device}, dtype={actual.dtype}, '
            f'different={int(different.sum())}/{actual.numel()}, '
            f'finite_pairs={int(finite.sum())}/{actual.numel()}, '
            f'max_abs={float(error[finite].max()) if bool(finite.any()) else None}, '
            f'first_index={first}, actual={float(actual[first])}, expected={float(expected[first])}')

    def assert_same_anchor_execution_state(self, reference, candidate):
        # no_grad() alone does not equalize Tensor.requires_grad. PyTorch may
        # select different high-dimensional Linear/matmul kernels from those
        # flags even with autograd disabled (ATen should_fold).
        self.assertEqual({n: p.requires_grad for n, p in reference.named_parameters()},
            {n: p.requires_grad for n, p in candidate.named_parameters()})
        self.assertEqual({n: m.training for n, m in reference.named_modules()},
            {n: m.training for n, m in candidate.named_modules()})

    def test_construction_preserves_all_anchor_tensors_names_and_rng(self):
        anchor = self.anchor()
        before, rng = tensor_tree_hash(anchor.state_dict()), rng_fingerprint()
        for arm in ARMS:
            model = GraphMemoryIGSTGNN(copy.deepcopy(anchor), arm)
            self.assertEqual(before, tensor_tree_hash(model.anchor.state_dict()))
            self.assertEqual(rng, rng_fingerprint())
            self.assertEqual(list(anchor.state_dict()), list(model.anchor.state_dict()))
            self.assertEqual(model.param_num(), sum(p.numel() for p in model.parameters()))
            restored = GraphMemoryIGSTGNN(copy.deepcopy(anchor), arm)
            restored.load_state_dict(model.state_dict(), strict=True)
            self.assertEqual(tensor_tree_hash(restored.state_dict()), tensor_tree_hash(model.state_dict()))

    def test_recurrent_control_is_exact_original_forward_and_joint_training(self):
        # Compare independent models under the joint control's all-trainable
        # flags, rather than the GW4 residual-only training flags.
        anchor = self.anchor().requires_grad_(True).eval()
        x, report = self.batch()
        model = GraphMemoryIGSTGNN(copy.deepcopy(anchor), 'RECURRENT_CONTROL').eval()
        self.assert_same_anchor_execution_state(anchor, model.anchor)
        with torch.no_grad():
            expected = anchor(x, incident_data=report)
            actual = model(x, incident_data=report)
        self.assert_exact_prediction(actual, expected, 'RECURRENT_CONTROL eval')
        # The native engine's train() must activate old backbone gradients and
        # dropout for both the recurrent joint control and direct joint arms.
        for arm in DEFAULT_ARMS:
            model = GraphMemoryIGSTGNN(copy.deepcopy(anchor), arm).train()
            self.assertTrue(model.original.embedding.training)
            self.assertTrue(model.original.layers[0].dif_layer.localized_st_conv.native.dropout.training)
            self.assertTrue(model.original.embedding.weight.requires_grad)
            for layer in model.original.layers:
                self.assertEqual(any(p.requires_grad for p in layer.dif_layer.forecast_branch.parameters()),
                    arm == 'RECURRENT_CONTROL')

    def test_frozen_probe_zero_output_recovers_gw4_exactly_and_learns(self):
        # The probe freezes the residual too; match that execution state in
        # its independent reference without changing any reference tensors.
        anchor = self.anchor().requires_grad_(False).eval()
        x, report = self.batch()
        with torch.no_grad():
            expected = anchor(x, incident_data=report)
        for arm in DEFAULT_ARMS[1:]:
            model = GraphMemoryIGSTGNN(copy.deepcopy(anchor), arm, stage='probe').train()
            self.assert_same_anchor_execution_state(anchor, model.anchor)
            anchor_hash = tensor_tree_hash(model.anchor.state_dict())
            rng = rng_fingerprint()
            actual = model(x, incident_data=report)
            self.assert_exact_prediction(actual, expected, f'{arm} frozen probe')
            self.assertEqual(rng, rng_fingerprint())
            self.assertTrue(all(not p.requires_grad for p in model.anchor.parameters()))
            self.assertFalse(any(m.training for m in model.anchor.modules()))
            model.assert_cache_clear()
            self.assertNotIn('forward', model.anchor.selector.__dict__)
            optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=.002)
            target = expected.detach()+torch.linspace(.1, 1., 12)[None, :, None, None]
            (actual-target).square().mean().backward()
            self.assertGreater(float(model.decoders[0].output.weight.grad.abs().sum()), 0.)
            self.assertTrue(all(p.grad is None for p in model.anchor.parameters()))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            (model(x, incident_data=report)-target).square().mean().backward()
            self.assertGreater(float(model.decoders[0].value.weight.grad.abs().sum()), 0.)
            if arm != 'POOLED_MEMORY':
                self.assertGreater(float(model.decoders[0].key.weight.grad.abs().sum()), 0.)
            self.assertEqual(anchor_hash, tensor_tree_hash(model.anchor.state_dict()))

    def test_history_memory_matches_receiver_and_native_conv_and_source_order_reference(self):
        anchor = self.anchor()
        original = anchor.base.original
        # Use non-symmetric supports and heterogeneous windows; a transpose or
        # accidentally merged order/source axis cannot pass this comparison.
        x = torch.arange(1., 1+2*12*4*8).reshape(2, 12, 4, 8)/100.
        convs = (anchor.selector, original.layers[1].dif_layer.localized_st_conv)
        for conv in convs:
            native = conv.native if hasattr(conv, 'native') else conv
            dynamic = [g[None].expand(2, -1, -1) for g in native.get_graph([
                torch.tensor([[0., .2, .8, 0.], [.4, 0., .6, 0.], [.1, .6, 0., .3], [.8, .2, 0., 0.]]),
                torch.tensor([[0., .4, .1, .5], [.6, 0., .4, 0.], [.3, .7, 0., 0.], [.1, .2, .7, 0.]])])]
            static = [torch.softmax(torch.arange(16.).reshape(4, 4)/5., -1)]
            if hasattr(conv, 'call_index'):
                conv.call_index = 0
            expected = conv(x, dynamic, static)
            if hasattr(conv, 'call_index'):
                conv.call_index = 0
            actual, memory, receiver = history_graph_memory(conv, x, dynamic, static)
            if hasattr(conv, 'weights'):
                self.assert_exact_prediction(actual, expected, 'GW4 historical memory capture')
            else:
                torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
            self.assertEqual(tuple(memory.shape), (2, 4, 5, 2, 10, 8))
            windows = x.unfold(1, native.k_t, 1).permute(0, 1, 2, 4, 3)
            transformed = native.activation(native.fc_list_updt(windows.reshape(2, 10, 4, 24))).reshape(2, 10, 4, 3, 8)
            torch.testing.assert_close(receiver, transformed.mean(-2)[:, -1])
            supports = list(native.pre_defined_graph)+dynamic+native.get_graph(static)
            for source in range(5):
                messages = []
                for order in range(2):
                    graph = supports[source*2+order].reshape(*supports[source*2+order].shape[:-1], 3, 4)
                    message = (torch.einsum('nkm,btmkc->btnkc', graph, transformed)
                               if graph.ndim == 3 else torch.einsum('bnkm,btmkc->btnkc', graph, transformed))
                    messages.append(message)
                weights = (conv.weights(messages[0], transformed.mean(-2), source)
                           if hasattr(conv, 'weights') else torch.ones_like(messages[0][..., 0]))
                for order, message in enumerate(messages):
                    reference = (message*weights[..., None]).sum(-2).permute(0, 2, 1, 3)
                    torch.testing.assert_close(memory[:, :, source, order], reference, atol=2e-5, rtol=2e-5)

    def test_joint_uses_one_history_propagation_per_layer_no_dif_recurrence(self):
        anchor = self.anchor()
        x, report = self.batch()
        model = GraphMemoryIGSTGNN(anchor, 'JOINT_MEMORY').eval()
        originals = [layer.dif_layer.forecast_branch for layer in model.original.layers]
        # A forecast_branch call would prove the old DIF recurrence survived.
        with patch.object(originals[0], 'forward', side_effect=AssertionError('obsolete DIF recurrence')):
            model.start_diagnostics()
            prediction = model(x, incident_data=report)
        self.assertEqual(tuple(prediction.shape), (2, 12, 4, 1))
        self.assertTrue(bool(torch.isfinite(prediction).all()))
        stats = model.diagnostics()
        self.assertEqual(set(stats['layers']), {'0', '1', '2', '3', '4'})
        self.assertEqual([stats['layers'][str(i)]['memory_shape'][4] for i in range(5)], [10, 8, 6, 4, 2])
        self.assertTrue(all(row['history_calls'] == 1 for row in stats['layers'].values()))
        self.assertEqual(set(stats['anchor']['by_phase']), {'history'})
        self.assertTrue(stats['cache_clear'])

    def test_actual_twelve_leads_original_gap_selection_and_native_tiid_expansion(self):
        channels = torch.arange(2*12*4*3.).reshape(2, 12, 4, 3)
        result = GraphMemoryIGSTGNN.select_gap_channels(channels, 3)
        for h in range(12):
            self.assertTrue(torch.equal(result[:, h, :, 0], channels[:, h, :, h % 3]))
        # Repeating native block features and selecting the old within-block
        # channel is exactly the original block-major flattening convention.
        native = torch.randn(2, 4, 4, 3)
        expanded = native.repeat_interleave(3, dim=1)
        expected = native.transpose(1, 2).contiguous().view(2, 4, 12).transpose(1, 2).unsqueeze(-1)
        self.assertTrue(torch.equal(GraphMemoryIGSTGNN.select_gap_channels(expanded, 3), expected))
        anchor = self.anchor()
        model = GraphMemoryIGSTGNN(anchor, 'JOINT_MEMORY').eval()
        x, report = self.batch()
        captured = []
        handle = model.original.tiid_module.register_forward_pre_hook(lambda module, inputs: captured.append(inputs[0].shape[1]))
        try:
            model(x, incident_data=report)
        finally:
            handle.remove()
        self.assertEqual(captured, [4])

    def test_chunked_attention_matches_unchunked_outputs_and_gradients(self):
        torch.manual_seed(31)
        memory = torch.randn(2, 5, 3, 2, 4, 6, dtype=torch.float64)
        receiver, context = (torch.randn(2, 5, 6, dtype=torch.float64) for _ in range(2))
        for arm in ARMS[1:]:
            a = GraphMemoryDecoder(6, 3, 2, 4, 12, 7, arm, hidden=5, node_chunk=2).double()
            b = copy.deepcopy(a)
            b.node_chunk = 100
            ma, mb = memory.clone().requires_grad_(), memory.clone().requires_grad_()
            ya, yb = a(ma, receiver, context), b(mb, receiver, context)
            torch.testing.assert_close(ya, yb, atol=1e-12, rtol=1e-12)
            ya.square().sum().backward()
            yb.square().sum().backward()
            torch.testing.assert_close(ma.grad, mb.grad, atol=1e-11, rtol=1e-11)
            for pa, pb in zip(a.parameters(), b.parameters()):
                if pa.grad is None:
                    self.assertIsNone(pb.grad)
                else:
                    torch.testing.assert_close(pa.grad, pb.grad, atol=1e-10, rtol=1e-10)

    def test_joint_nonseparability_factorized_odds_and_information_ablation(self):
        torch.manual_seed(51)
        memory = torch.randn(2, 3, 3, 2, 4, 6, dtype=torch.float64)
        receiver, context = (torch.randn(2, 3, 6, dtype=torch.float64) for _ in range(2))
        for arm in ('JOINT_MEMORY', 'FACTORIZED_MEMORY'):
            decoder = GraphMemoryDecoder(6, 3, 2, 4, 12, 7, arm, hidden=5).double()
            weights = decoder.attention_weights(memory, receiver, context)
            torch.testing.assert_close(weights.sum((-3, -2, -1)), torch.ones(2, 3, 12, dtype=torch.float64))
            logp = weights.log()
            odds = logp[..., 0, :, 0]+logp[..., 1, :, 1]-logp[..., 0, :, 1]-logp[..., 1, :, 0]
            if arm == 'FACTORIZED_MEMORY':
                self.assertLess(float(odds.abs().max()), 1e-12)
            else:
                self.assertGreater(float(odds.abs().max()), 1e-4)
                self.assertFalse(torch.allclose(weights[:, :, 0], weights[:, :, -1]))
                self.assertFalse(torch.allclose(weights, decoder.attention_weights(memory, receiver+2., context)))
        decoder = GraphMemoryDecoder(6, 3, 2, 4, 12, 7, 'LEAD_ONLY', hidden=5).double()
        self.assertTrue(torch.equal(decoder.attention_weights(memory, receiver, context),
            decoder.attention_weights(memory, receiver+99., context-99.)))
        decoder = GraphMemoryDecoder(6, 3, 2, 4, 12, 7, 'NO_LEAD', hidden=5).double()
        weights = decoder.attention_weights(memory, receiver, context)
        self.assertTrue(torch.equal(weights[:, :, 0], weights[:, :, -1]))
        output = decoder(memory, receiver, context)
        self.assertFalse(torch.allclose(output[:, 0], output[:, -1]))
        output.square().sum().backward()
        self.assertTrue(decoder.lead.requires_grad)
        self.assertGreater(float(decoder.lead.grad.abs().sum()), 0.)

    def test_training_gradient_full_dif_graph_and_no_target_inputs_or_input_mutation(self):
        anchor = self.anchor()
        x, report = self.batch()
        target = torch.linspace(.1, 2., 96).reshape(2, 12, 4, 1)
        before = tensor_tree_hash((x, report, target))
        for arm in DEFAULT_ARMS[1:]:
            model = GraphMemoryIGSTGNN(copy.deepcopy(anchor), arm).eval()
            first = model(x, label=target, incident_data=report)
            second = model(x, label=target*1e6, incident_data=report)
            self.assertTrue(torch.equal(first, second))
            (first-target).square().mean().backward()
            self.assertGreater(float(model.original.embedding.weight.grad.abs().sum()), 0.)
            for index, layer in enumerate(model.original.layers):
                conv = layer.dif_layer.localized_st_conv
                native = conv.native if hasattr(conv, 'native') else conv
                self.assertGreater(float(native.fc_list_updt.weight.grad.abs().sum()), 0.)
                self.assertGreater(float(model.decoders[index].value.weight.grad.abs().sum()), 0.)
                self.assertTrue(all(p.grad is None for p in layer.dif_layer.forecast_branch.parameters()))
            self.assertTrue(all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None))
            model.assert_cache_clear()
        self.assertEqual(before, tensor_tree_hash((x, report, target)))

    def test_cache_and_probe_method_restore_after_exception_and_new_batch(self):
        x, report = self.batch()
        for stage in ('joint', 'probe'):
            model = GraphMemoryIGSTGNN(self.anchor(), 'JOINT_MEMORY', stage=stage).eval()
            with patch.object(model.decoders[0], 'forward', side_effect=RuntimeError('deliberate decoder exception')):
                with self.assertRaisesRegex(RuntimeError, 'deliberate'):
                    model(x, incident_data=report)
            model.assert_cache_clear()
            self.assertNotIn('forward', model.anchor.selector.__dict__)
            self.assertFalse(model.original.icsf_module._forward_hooks)
            first = model(x, incident_data=report)
            model(x*2., incident_data=report)
            self.assertTrue(torch.equal(first, model(x, incident_data=report)))
            model.assert_cache_clear()


if __name__ == '__main__':
    unittest.main()
