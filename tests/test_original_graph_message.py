"""GW2 scientific contracts: graph evidence, matched controls and recovery."""
import copy
import itertools
import unittest

import torch

from experiments.chronological.original_graph_message import (
    ARMS, GraphMessageConv, MessageRouter, build_variant, split_state)
from experiments.chronological.original_graph_window import GraphWindowConv
from experiments.chronological.original_igstgnn_runtime import tensor_tree_hash, rng_fingerprint
from experiments.chronological.original_igstgnn_source import get_original_modules, verify_original_source
from tests import test_original_graph_window as gw1_fixtures


class GraphMessageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.modules = get_original_modules()

    model = gw1_fixtures.GraphWindowTests.model
    batch = staticmethod(gw1_fixtures.GraphWindowTests.batch)

    def test_frozen_original_source_preserved(self):
        verify_original_source()

    def test_recovery_train_eval_rng_native_state_and_inputs(self):
        original = self.model()
        x, report = self.batch()
        before = tensor_tree_hash((x, report))
        for arm, training in itertools.product(ARMS[1:], (False, True)):
            with self.subTest(arm=arm, training=training):
                native = copy.deepcopy(original).train(training)
                candidate = build_variant(copy.deepcopy(original), arm).train(training)
                torch.manual_seed(309)
                expected = native(x, incident_data=report)
                rng = rng_fingerprint()
                torch.manual_seed(309)
                actual = candidate(x, incident_data=report)
                torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
                self.assertEqual(rng, rng_fingerprint())
                self.assertEqual(tensor_tree_hash(native.state_dict()), tensor_tree_hash(split_state(candidate)[0]))
                self.assertEqual(before, tensor_tree_hash((x, report)))

    def test_candidate_initial_capacity_and_rng_match_only_first_conv_changes(self):
        native = self.model()
        rng = rng_fingerprint()
        added = []
        for arm in ARMS[1:]:
            model = build_variant(copy.deepcopy(native), arm)
            added.append(tensor_tree_hash(split_state(model)[1]))
            self.assertEqual(rng, rng_fingerprint())
            self.assertIsInstance(model.selector, GraphMessageConv)
            self.assertTrue(all(not isinstance(layer.dif_layer.localized_st_conv, GraphMessageConv)
                for layer in model.original.layers[1:]))
            self.assertEqual(model.param_num(), sum(p.numel() for p in model.parameters()))
        self.assertEqual(len(set(added)), 1)
        self.assertIs(build_variant(native, 'ORIGINAL'), native)

    def test_slot_messages_match_explicit_receiver_lag_sender_with_gradients(self):
        # Deliberately lag-dependent, nonsymmetric supports; both batch layouts.
        for batched in (False, True):
            x = torch.arange(1., 1+2*3*2*3*2).reshape(2, 3, 2, 3, 2).double().requires_grad_()
            graph = (torch.arange(1., 13).reshape(2, 6)/13).double()
            if batched:
                graph = torch.stack((graph, graph.flip(-1)*.7))
            graph.requires_grad_()
            result = GraphMessageConv.slot_messages(graph, x)
            expanded = graph if batched else graph[None].expand(2, -1, -1)
            expected = torch.stack([torch.stack([sum(
                expanded[:, receiver, slot*2+sender, None, None]*x[:, :, sender, slot]
                for sender in range(2)) for slot in range(3)], 2) for receiver in range(2)], 2)
            torch.testing.assert_close(result, expected)
            a = torch.autograd.grad(result.square().sum(), (x, graph), retain_graph=True)
            b = torch.autograd.grad(expected.square().sum(), (x, graph))
            for p, q in zip(a, b):
                torch.testing.assert_close(p, q)

    def test_nonuniform_static_commutes_with_native_gw1_sender_selection(self):
        native = self.model().layers[0].dif_layer.localized_st_conv
        before, after = GraphWindowConv(copy.deepcopy(native), 'STATIC'), GraphMessageConv(copy.deepcopy(native), 'STATIC')
        with torch.no_grad():
            before.router.static_logits.normal_(std=.5)
            after.router.static_logits.copy_(before.router.static_logits)
        x = torch.rand(2, 5, 4, 8)
        dynamic = [g[None].expand(2, -1, -1) for g in native.pre_defined_graph]
        static = [torch.eye(4)]
        before.eval(); after.eval()
        torch.testing.assert_close(after(x, dynamic, static), before(x, dynamic, static), atol=2e-5, rtol=2e-5)

    def test_packed_native_kernel_matches_general_kernel_and_upstream_graph_gradients(self):
        for batched in (False, True):
            torch.manual_seed(7)
            matrix = torch.randn((2, 4, 4) if batched else (4, 4), dtype=torch.float64, requires_grad=True)
            x = torch.randn(2, 5, 4, 3, 2, dtype=torch.float64, requires_grad=True)
            # Exactly the frozen native repeat operation; validate gradient
            # against the underlying adjacency, not independently free copies.
            graph = matrix.unsqueeze(-2).expand(*matrix.shape[:-1], 3, 4).reshape(*matrix.shape[:-1], 12)
            a = GraphMessageConv.slot_messages(graph, x)
            b = GraphMessageConv.slot_messages(graph, x, native_repeated=True)
            torch.testing.assert_close(a, b)
            ga = torch.autograd.grad(a.square().sum(), (matrix, x), retain_graph=True)
            gb = torch.autograd.grad(b.square().sum(), (matrix, x))
            for p, q in zip(ga, gb):
                torch.testing.assert_close(p, q)

    def test_receiver_weights_and_graph_orders_match_explicit_reference(self):
        graph = torch.tensor([[0., .2], [.8, 0.]])
        for pre, dy, sta in itertools.product((False, True), repeat=3):
            if not (pre or dy or sta):
                continue
            native = self.modules.model.STLocalizedConv(2, pre_defined_graph=[graph, graph.T],
                k_t=3, k_s=2, use_pre=pre, dy_graph=dy, sta_graph=sta, dropout=0.)
            conv = GraphMessageConv(native, 'POST_MESSAGE')
            with torch.no_grad():
                native.fc_list_updt.weight.copy_(torch.eye(6))
                conv.router.score_head.weight.normal_(std=.2)
                conv.router.static_logits.normal_(std=.2)
            x = torch.arange(1., 1+2*5*2*2).reshape(2, 5, 2, 2)/20
            dynamic = [g[None].expand(2, -1, -1) for g in native.get_graph([graph, graph.T])]
            static = [graph]
            actual = conv(x, dynamic, static)
            window = x.unfold(1, 3, 1).permute(0, 1, 2, 4, 3)
            parts = [window.mean(-2)]
            supports = conv._supports(dynamic, static)
            for source in range(len(conv.source_labels)):
                banks = supports[source*2:source*2+2]
                first = conv.slot_messages(banks[0], window)
                weights = conv.router(first, source, 'POST_MESSAGE')
                for bank in banks:
                    expanded = bank[None].expand(2, -1, -1) if bank.ndim == 2 else bank
                    values = []
                    for receiver in range(2):
                        values.append(sum(expanded[:, receiver, slot*2+sender, None, None]
                            *window[:, :, sender, slot]*weights[:, :, receiver, slot, None]
                            for slot, sender in itertools.product(range(3), range(2))))
                    parts.append(torch.stack(values, 2))
            expected = native.gcn_updt(torch.cat(parts, -1))
            torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)

    def test_message_scores_respond_to_neighbor_evidence_local_control_does_not(self):
        native = self.model().layers[0].dif_layer.localized_st_conv
        local = GraphMessageConv(copy.deepcopy(native), 'POST_LOCAL').eval()
        message = GraphMessageConv(copy.deepcopy(native), 'POST_MESSAGE').eval()
        with torch.no_grad():
            local.router.score_head.weight.normal_(std=.5)
            message.load_state_dict(local.state_dict())
        x = torch.rand(2, 5, 4, 8)
        changed = x.clone(); changed[:, 1, 1] += 10
        dynamic = [g[None].expand(2, -1, -1) for g in native.pre_defined_graph]
        static = [torch.eye(4)]
        records = {}
        for name, conv in (('local', local), ('message', message)):
            rows = []
            handle = conv.router.register_forward_hook(lambda _, args, out: rows.append(out.detach().clone()))
            try:
                conv(x, dynamic, static); conv(changed, dynamic, static)
            finally:
                handle.remove()
            g = len(conv.source_labels)
            records[name] = (rows[0][:, :, 0], rows[g][:, :, 0])
        self.assertTrue(torch.equal(*records['local']))
        self.assertFalse(torch.allclose(*records['message']))

    def test_router_coefficients_static_control_and_zero_head_encoder_gradient(self):
        router = MessageRouter(8, 3, 5)
        x = torch.randn(2, 4, 3, 3, 8)
        self.assertTrue(torch.equal(router(x, 0, 'POST_MESSAGE'), torch.ones_like(x[..., 0])))
        with torch.no_grad():
            router.static_logits.normal_()
            router.score_head.weight.normal_(std=.3)
        static = router(x, 0, 'STATIC')
        self.assertTrue(torch.equal(static, router(x*10, 0, 'STATIC')))
        actual = router(x, 0, 'POST_MESSAGE')
        self.assertFalse(torch.equal(actual, router(x*10, 0, 'POST_MESSAGE')))
        self.assertTrue(bool((actual >= 0).all() and (actual <= 3).all()))
        torch.testing.assert_close(actual.sum(-1), torch.full_like(actual[..., 0], 3.))

    def test_native_mae_learns_active_heads_then_encoder(self):
        x, report = self.batch()
        target = torch.linspace(1, 200, 96).reshape(2, 12, 4, 1)
        for arm in ARMS[1:]:
            model = build_variant(self.model(), arm).eval()
            router = model.selector.router
            loss = self.modules.metrics.masked_mae(model(x, incident_data=report)*30+50, target, target.new_tensor(0.))
            loss.backward()
            head = router.static_logits if arm == 'STATIC' else router.score_head.weight
            self.assertGreater(float(head.grad.abs().sum()), 0.)
            for p in model.parameters():
                if p.grad is not None:
                    self.assertTrue(bool(torch.isfinite(p.grad).all()))
            if arm != 'STATIC':
                self.assertEqual(float(router.encoder.weight.grad.abs().sum()), 0.)
                # A real objective update, not a synthetic nonzero head.
                optimizer = torch.optim.Adam(model.parameters(), lr=.002)
                optimizer.step()
                model.zero_grad(set_to_none=True)
                self.modules.metrics.masked_mae(model(x, incident_data=report)*30+50, target, target.new_tensor(0.)).backward()
                self.assertGreater(float(router.encoder.weight.grad.abs().sum()), 0.)

    def test_initial_native_gradients_match_with_reduction_tolerance(self):
        native = self.model().eval()
        model = build_variant(copy.deepcopy(native), 'POST_MESSAGE').eval()
        x, report = self.batch()
        for m in (native, model):
            m(x, incident_data=report).square().mean().backward()
        prefix = 'layers.0.dif_layer.localized_st_conv.'
        grads = {name.replace(prefix+'native.', prefix): p.grad for name, p in model.original.named_parameters() if '.router.' not in name}
        for name, p in native.named_parameters():
            if p.grad is None:
                self.assertIsNone(grads[name])
            else:
                torch.testing.assert_close(p.grad, grads[name], atol=3e-5, rtol=3e-5, msg=name)

    def test_label_invariance_and_intervention_restoration(self):
        model = build_variant(self.model(), 'POST_MESSAGE').eval()
        x, report = self.batch()
        with torch.no_grad():
            model.selector.router.score_head.weight.normal_(std=.2)
        a = model(x, torch.zeros(2, 12, 4, 1), incident_data=report)
        b = model(x, torch.ones(2, 12, 4, 1)*1e6, incident_data=report)
        self.assertTrue(torch.equal(a, b))
        with self.assertRaisesRegex(RuntimeError, 'probe'):
            with model.intervention('UNIFORM'):
                model(x, incident_data=report)
                raise RuntimeError('probe')
        self.assertIsNone(model.selector.override_mode)
        self.assertTrue(torch.equal(a, model(x, incident_data=report)))

    def test_diagnostics_remove_fixed_node_and_window_differences_and_keep_phase(self):
        conv = GraphMessageConv(self.model().layers[0].dif_layer.localized_st_conv, 'POST_MESSAGE')
        # Strong node/window differences, but every sample has identical routes.
        w = 3*torch.softmax(torch.randn(1, 4, 4, 3).double(), -1)
        conv.start_diagnostics()
        for source in range(5):
            conv._accumulate(w.expand(2, -1, -1, -1), source, 'history')
            conv._accumulate(w, source, 'history')
            conv._accumulate(w[:, :1], source, 'forecast_1')
        stats = conv.finish_diagnostics()
        self.assertGreater(max(stats['weight_std'][0]), .01)
        row = stats['by_phase']['history']['PREDEFINED_0']
        self.assertEqual(row['samples'], 3)
        self.assertLess(max(row['across_sample_std_rms']), 3e-8)
        self.assertEqual(stats['by_phase']['forecast_1']['PREDEFINED_0']['samples'], 1)

    def test_model_diagnostics_tail_partition_and_call_phase(self):
        model = build_variant(self.model(), 'POST_MESSAGE').eval()
        x, report = self.batch()
        with torch.no_grad():
            model.selector.router.score_head.weight.normal_(std=.2)
            model.start_diagnostics(); model(x, incident_data=report)
            whole = model.diagnostics()
            model.start_diagnostics()
            for i in range(2):
                model(x[i:i+1], incident_data={key: value[i:i+1] for key, value in report.items()})
            split = model.diagnostics()
        self.assertEqual(set(whole['by_phase']), {'history', 'forecast_1', 'forecast_2', 'forecast_3'})
        torch.testing.assert_close(torch.tensor(whole['weight_mean']), torch.tensor(split['weight_mean']))
        for phase in whole['by_phase']:
            a, b = (s['by_phase'][phase]['PREDEFINED_0'] for s in (whole, split))
            torch.testing.assert_close(torch.tensor(a['across_sample_std_rms']), torch.tensor(b['across_sample_std_rms']), atol=1e-7, rtol=1e-4)
        self.assertEqual(model.diagnostics()['calls'], 0)

    def test_invalid_axes_and_wrapped_model_are_rejected(self):
        with self.assertRaises(ValueError):
            GraphMessageConv.slot_messages(torch.ones(2, 6), torch.ones(2, 3, 3, 3, 2))
        with self.assertRaises(ValueError):
            build_variant(self.model(), 'UNKNOWN')
        with self.assertRaisesRegex(ValueError, 'unwrapped'):
            build_variant(build_variant(self.model(), 'POST_MESSAGE').original, 'STATIC')


if __name__ == '__main__':
    unittest.main()
