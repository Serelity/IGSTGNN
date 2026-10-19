"""Server-only GW7 contracts; these are synthetic checks, not training evidence.

Run on the server: python -m unittest tests.test_graph_interaction_gw7_model -v
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
from experiments.graph_interaction_gw7.model import (
    ARMS, InteractionMemoryIGSTGNN, ResidualMemory, memory_banks)
from tests import test_original_graph_window as fixtures


class GraphInteractionGW7Tests(unittest.TestCase):
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

    def test_projection_matches_independently_hand_computed_interaction(self):
        memory = torch.tensor([1., 3., 2., 8., 4., 6., 10., 14.], dtype=torch.float64
            ).reshape(1, 1, 2, 2, 2, 1)
        stable, changing = memory_banks(memory, True)
        expected_stable = torch.tensor([1., -1., -1., 1.], dtype=torch.float64
            ).reshape(1, 1, 2, 2, 1, 1)
        expected_changing = torch.tensor([.25, -.25, -.25, .25, -.25, .25, .25, -.25],
            dtype=torch.float64).reshape_as(memory)
        self.assertTrue(torch.equal(stable, expected_stable))
        self.assertTrue(torch.equal(changing, expected_changing))
        ordinary_a, ordinary_b = memory_banks(memory)
        self.assertTrue(torch.equal(ordinary_a.expand_as(memory)+ordinary_b, memory))
        direct = memory-memory.mean(2, keepdim=True)-memory.mean(3, keepdim=True)
        direct = direct+memory.mean((2, 3), keepdim=True)
        torch.testing.assert_close(stable+changing, direct, atol=0., rtol=0.)

    def test_projection_removes_additive_source_and_order_effects_at_each_window(self):
        source_window = torch.tensor([1., 4., 8., -2., 6., 3.], dtype=torch.float64
            ).reshape(1, 1, 3, 1, 2, 1)
        order_window = torch.tensor([2., 5., -4., 9.], dtype=torch.float64
            ).reshape(1, 1, 1, 2, 2, 1)
        stable, changing = memory_banks(source_window+order_window, True)
        torch.testing.assert_close(stable, torch.zeros_like(stable), atol=1e-14, rtol=0.)
        torch.testing.assert_close(changing, torch.zeros_like(changing), atol=1e-14, rtol=0.)
        for value, axis in ((stable, 2), (stable, 3), (changing, 2), (changing, 3), (changing, 4)):
            torch.testing.assert_close(value.mean(axis), torch.zeros_like(value.mean(axis)), atol=1e-14, rtol=0.)

    def test_zero_banks_cannot_create_output_from_key_positions_or_context(self):
        for arm in ('R_MEMORY', 'R_INTERACTION_MEMORY'):
            decoder = ResidualMemory(3, 2, 2, 4, 12, 1, arm, hidden=5).double()
            with torch.no_grad():
                decoder.output.weight.normal_()
            result = decoder(torch.zeros(2, 3, 2, 2, 4, 3, dtype=torch.float64),
                torch.randn(2, 3, 3, dtype=torch.float64), torch.randn(2, 3, 3, dtype=torch.float64))
            self.assertTrue(torch.equal(result, torch.zeros_like(result)))

    def test_interaction_projection_does_not_impose_output_mean_conservation(self):
        decoder = ResidualMemory(1, 2, 2, 2, 2, 1, 'R_INTERACTION_MEMORY', hidden=1).double()
        with torch.no_grad():
            decoder.context.weight.zero_()
            decoder.context.bias.fill_(1.)
            decoder.lead.fill_(1.)
            decoder.output.weight.fill_(1.)
            for reader in decoder.readers:
                reader.key.weight.fill_(1.)
                reader.value.weight.fill_(1.)
                reader.source_embedding.zero_()
                reader.order_embedding.zero_()
                if reader.window_embedding is not None:
                    reader.window_embedding.zero_()
        memory = torch.tensor([1., -1., -1., 1.], dtype=torch.float64
            ).reshape(1, 1, 2, 2, 1, 1).expand(1, 1, 2, 2, 2, 1)
        result = decoder(memory, torch.zeros(1, 1, 1, dtype=torch.float64))
        self.assertGreater(float(result.mean()), .5)

    def test_construction_preserves_anchor_tensors_names_rng_and_strict_reload(self):
        anchor = self.anchor()
        before, rng = tensor_tree_hash(anchor.state_dict()), rng_fingerprint()
        for arm in ARMS:
            model = InteractionMemoryIGSTGNN(copy.deepcopy(anchor), arm)
            self.assertEqual(before, tensor_tree_hash(model.anchor.state_dict()))
            self.assertEqual(list(anchor.state_dict()), list(model.anchor.state_dict()))
            self.assertEqual(rng, rng_fingerprint())
            self.assertEqual(model.param_num(), sum(p.numel() for p in model.parameters()))
            restored = InteractionMemoryIGSTGNN(copy.deepcopy(anchor), arm)
            restored.load_state_dict(model.state_dict(), strict=True)
            self.assertEqual(tensor_tree_hash(model.state_dict()), tensor_tree_hash(restored.state_dict()))

    def test_memory_and_interaction_have_identical_active_capacity_and_initial_state(self):
        anchor = self.anchor()
        ordinary = InteractionMemoryIGSTGNN(copy.deepcopy(anchor), 'R_MEMORY')
        interaction = InteractionMemoryIGSTGNN(copy.deepcopy(anchor), 'R_INTERACTION_MEMORY')
        self.assertEqual(ordinary.trainable_parameter_counts(), interaction.trainable_parameter_counts())
        self.assertEqual(tensor_tree_hash(ordinary.residual.state_dict()),
            tensor_tree_hash(interaction.residual.state_dict()))
        self.assertIsNone(ordinary.residual.readers[0].window_embedding)
        self.assertIsNotNone(ordinary.residual.readers[1].window_embedding)
        self.assertNotIn('readers.0.window_embedding', ordinary.residual.state_dict())
        mlp = InteractionMemoryIGSTGNN(copy.deepcopy(anchor), 'R_MLP')
        self.assertTrue(all(p.requires_grad for p in mlp.residual.parameters()))
        counts = mlp.trainable_parameter_counts()
        self.assertIsInstance(counts['mlp_width'], int)
        # Width is the closest integer active capacity, up to the one-width step.
        step = 2*mlp.channels+3*mlp.hidden+1
        self.assertLessEqual(abs(counts['added']-ordinary.trainable_parameter_counts()['added']), step/2)

    def test_zero_output_recovers_recurrent_prediction_train_eval_and_rng_exactly(self):
        anchor = self.anchor()
        x, report = self.batch()
        for arm in ARMS:
            for training in (False, True):
                with self.subTest(arm=arm, training=training):
                    reference = copy.deepcopy(anchor).train(training)
                    model = InteractionMemoryIGSTGNN(copy.deepcopy(anchor), arm).train(training)
                    self.assertEqual({n: p.requires_grad for n, p in reference.named_parameters()},
                        {n: p.requires_grad for n, p in model.anchor.named_parameters()})
                    self.assertEqual({n: m.training for n, m in reference.named_modules()},
                        {n: m.training for n, m in model.anchor.named_modules()})
                    for grad_enabled in (False, True):
                        with torch.set_grad_enabled(grad_enabled):
                            torch.manual_seed(379)
                            expected = reference(x, incident_data=report)
                            expected_rng = rng_fingerprint()
                            torch.manual_seed(379)
                            actual = model(x, incident_data=report)
                        self.assertTrue(torch.equal(actual, expected),
                            f'{arm}, training={training}, grad={grad_enabled}')
                        self.assertEqual(expected_rng, rng_fingerprint())
                        model.assert_cache_clear()

    def test_captured_memory_uses_existing_native_parts_without_recomputing_graph(self):
        model = InteractionMemoryIGSTGNN(self.anchor(), 'R_MEMORY').eval()
        x, report = self.batch()
        observed, read = [], []
        native = model.anchor.anchor.selector.native
        def native_hook(module, inputs):
            if not observed:
                observed.append(inputs[0])
        def residual_hook(module, inputs):
            read.append(inputs[:2])
        h1 = native.gcn_updt.register_forward_pre_hook(native_hook)
        h2 = model.residual.register_forward_pre_hook(residual_hook)
        try:
            model(x, incident_data=report)
        finally:
            h1.remove()
            h2.remove()
        parts = observed[0].reshape(2, 10, 4, 11, 8)
        memory, receiver = read[0]
        self.assertTrue(memory.requires_grad)
        self.assertTrue(receiver.requires_grad)
        self.assertTrue(torch.equal(receiver, parts[:, -1, :, 0]))
        for source in range(5):
            for order in range(2):
                self.assertTrue(torch.equal(memory[:, :, source, order],
                    parts[:, :, :, 1+source*2+order].permute(0, 2, 1, 3)))

    def test_labels_do_not_affect_nonzero_correction_and_inputs_are_not_mutated(self):
        model = InteractionMemoryIGSTGNN(self.anchor(), 'R_INTERACTION_MEMORY').eval()
        with torch.no_grad():
            model.residual.output.weight.normal_(std=.1)
        x, report = self.batch()
        target = torch.randn(2, 12, 4, 1)
        before = tensor_tree_hash((x, report, target))
        first = model(x, label=target, incident_data=report)
        second = model(x, label=target*1e6, incident_data=report)
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(before, tensor_tree_hash((x, report, target)))

    def test_component_capture_is_one_forward_preserves_autograd_and_clears_cache(self):
        model = InteractionMemoryIGSTGNN(self.anchor(), 'R_INTERACTION_MEMORY').eval()
        x, report = self.batch()
        model.start_diagnostics()
        components = model.capture_components(x, incident_data=report)
        self.assertEqual(set(components), {'base', 'correction', 'total'})
        self.assertTrue(all(v.requires_grad for v in components.values()))
        self.assertTrue(torch.equal(components['total'], components['base']+components['correction']))
        self.assertEqual(tuple(components['total'].shape), (2, 12, 4, 1))
        components['total'].square().mean().backward()
        model.assert_cache_clear()
        stats = model.diagnostics()
        self.assertEqual(stats['history_captures'], 1)
        self.assertEqual(stats['complete_forwards'], 1)
        self.assertEqual(set(stats['components']), {'base', 'correction', 'total'})
        self.assertTrue(stats['cache_clear'])

    def test_joint_training_keeps_all_native_forecast_gradients_and_learns_new_head(self):
        x, report = self.batch()
        target = torch.linspace(.1, 1., 12)[None, :, None, None]
        for arm in ARMS:
            model = InteractionMemoryIGSTGNN(self.anchor(), arm).eval()
            self.assertTrue(all(p.requires_grad for p in model.anchor.parameters()))
            result = model(x, incident_data=report)
            (result-target).square().mean().backward()
            for layer in model.original.layers:
                parameters = list(layer.dif_layer.forecast_branch.parameters())
                self.assertTrue(all(p.requires_grad for p in parameters))
                self.assertGreater(sum(float(p.grad.abs().sum()) for p in parameters if p.grad is not None), 0.)
            if arm != 'R_CONTINUE':
                self.assertGreater(float(model.residual.output.weight.grad.abs().sum()), 0.)
                # At zero W_out, interior gradients can be zero. Enable the
                # trained output path before checking reader/query gradients.
                with torch.no_grad():
                    model.residual.output.weight.normal_(std=.1)
                model.zero_grad(set_to_none=True)
                (model(x, incident_data=report)-target).square().mean().backward()
                self.assertGreater(float(model.residual.context.weight.grad.abs().sum()), 0.)
                if arm != 'R_MLP':
                    self.assertGreater(float(model.residual.readers[0].value.weight.grad.abs().sum()), 0.)
                    self.assertGreater(float(model.residual.readers[1].key.weight.grad.abs().sum()), 0.)
            self.assertTrue(all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None))
            model.assert_cache_clear()

    def test_chunked_and_unchunked_readout_outputs_and_gradients_match(self):
        torch.manual_seed(43)
        memory = torch.randn(2, 5, 3, 2, 4, 3, dtype=torch.float64)
        receiver, context = (torch.randn(2, 5, 3, dtype=torch.float64) for _ in range(2))
        for arm in ARMS[1:]:
            a = ResidualMemory(3, 3, 2, 4, 12, 1, arm, hidden=5, node_chunk=2).double()
            with torch.no_grad():
                a.output.weight.normal_(std=.1)
            b = copy.deepcopy(a)
            b.node_chunk = 100
            ma, mb = memory.clone().requires_grad_(), memory.clone().requires_grad_()
            ya, yb = a(ma, receiver, context), b(mb, receiver, context)
            torch.testing.assert_close(ya, yb, atol=1e-12, rtol=1e-12)
            ya.square().sum().backward()
            yb.square().sum().backward()
            torch.testing.assert_close(ma.grad, mb.grad, atol=1e-11, rtol=1e-11)
            for pa, pb in zip(a.parameters(), b.parameters()):
                torch.testing.assert_close(pa.grad, pb.grad, atol=1e-10, rtol=1e-10)

    def test_hooks_cache_and_forward_state_clear_after_decoder_and_anchor_exceptions(self):
        model = InteractionMemoryIGSTGNN(self.anchor(), 'R_MEMORY').eval()
        x, report = self.batch()
        for module in (model.residual, model.original.out_fc_2):
            with patch.object(module, 'forward', side_effect=RuntimeError('deliberate exception')):
                with self.assertRaisesRegex(RuntimeError, 'deliberate'):
                    model(x, incident_data=report)
            model.assert_cache_clear()
            self.assertFalse(model.anchor.anchor.selector.native.gcn_updt._forward_pre_hooks)
            self.assertFalse(model.original.icsf_module._forward_hooks)
        expected = model(x, incident_data=report)
        model(x*2., incident_data=report)
        self.assertTrue(torch.equal(expected, model(x, incident_data=report)))

    def test_reentrant_forward_is_rejected_without_destroying_outer_state(self):
        model = InteractionMemoryIGSTGNN(self.anchor(), 'R_MEMORY').eval()
        x, report = self.batch()
        seen = []
        def try_nested(module, inputs):
            if seen:
                return
            seen.append(True)
            with self.assertRaisesRegex(RuntimeError, 'Reentrant'):
                model(x, incident_data=report)
        handle = model.original.embedding.register_forward_pre_hook(try_nested)
        try:
            result = model(x, incident_data=report)
        finally:
            handle.remove()
        self.assertEqual(tuple(result.shape), (2, 12, 4, 1))
        model.assert_cache_clear()


if __name__ == '__main__':
    unittest.main()
