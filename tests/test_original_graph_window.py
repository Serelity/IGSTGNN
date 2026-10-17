"""GW1 scientific contracts: native recovery, graph axes, controls and learning."""
import copy
import itertools
import json
from pathlib import Path
import tempfile
import unittest

import torch

from experiments.chronological.original_graph_window import (
    ARMS, GraphWindowConv, WindowRouter, build_variant, split_state)
from experiments.chronological.original_igstgnn_runtime import tensor_tree_hash, rng_fingerprint
from experiments.chronological.original_igstgnn_source import get_original_modules, verify_original_source


class GraphWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.modules = get_original_modules()

    def model(self):
        tmp = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(tmp.cleanup)
        for name in ('desc_mapping.json', 'type_mapping.json'):
            (Path(tmp.name)/name).write_text(json.dumps({'a': 0, 'b': 1}))
        graph = torch.tensor([[0., .3, .7, 0.], [.2, 0., .8, 0.], [.4, .3, 0., .3], [0., .4, .6, 0.]])
        torch.manual_seed(19)
        return self.modules.model.IGSTGNN(dict(num_feat=1, num_hidden=8, node_hidden=4,
            time_emb_dim=4, tpd=288, k_s=2, k_t=3, layer=5, gap=3, dropout=.1,
            dataset='fixture', data_path=tmp.name, use_sensor_info=False,
            adjs=[graph, graph.T]), node_num=4, input_dim=3, output_dim=1, seq_len=12, horizon=12)

    @staticmethod
    def batch():
        torch.manual_seed(11)
        x = torch.rand(2, 12, 4, 3)*.2
        report = dict(incident=torch.tensor([[0., 1., 1., 0.], [0., 0., 0., 1.]]),
            position=torch.tensor([0, 1]), distances=torch.zeros(2, 4, 3))
        report['distances'][:, :2, 0] = 1
        return x, report

    def test_frozen_original_source_preserved(self):
        verify_original_source()

    def test_all_candidates_recover_native_train_eval_and_dropout_rng_exactly(self):
        native = self.model()
        x, report = self.batch()
        before = tensor_tree_hash((x, report))
        for arm, training in itertools.product(ARMS[1:], (False, True)):
            with self.subTest(arm=arm, training=training):
                a, b = copy.deepcopy(native).train(training), build_variant(copy.deepcopy(native), arm).train(training)
                torch.manual_seed(309)
                expected = a(x, incident_data=report)
                expected_rng = rng_fingerprint()
                torch.manual_seed(309)
                actual = b(x, incident_data=report)
                self.assertTrue(torch.equal(expected, actual))
                self.assertEqual(expected_rng, rng_fingerprint())
                self.assertEqual(tensor_tree_hash(a.state_dict()), tensor_tree_hash(split_state(b)[0]))
                self.assertEqual(tensor_tree_hash((x, report)), before)

    def test_construction_preserves_rng_and_candidate_initial_tensors(self):
        native = self.model()
        before = rng_fingerprint()
        added = []
        for arm in ARMS[1:]:
            model = build_variant(copy.deepcopy(native), arm)
            added.append(tensor_tree_hash(split_state(model)[1]))
            self.assertEqual(before, rng_fingerprint())
            self.assertEqual(model.selector.source_labels,
                ('PREDEFINED_0', 'PREDEFINED_1', 'DYNAMIC_0', 'DYNAMIC_1', 'ADAPTIVE_0'))
        self.assertEqual(len(set(added)), 1)

    def test_only_first_conv_is_replaced_no_other_path(self):
        model = build_variant(self.model(), 'GRAPH_CONTENT')
        self.assertEqual(model.horizon, 12)
        self.assertEqual(model.node_num, 4)
        self.assertEqual(model.param_num(), sum(p.numel() for p in model.parameters()))
        self.assertIsInstance(model.original.layers[0].dif_layer.localized_st_conv, GraphWindowConv)
        self.assertTrue(all(not isinstance(layer.dif_layer.localized_st_conv, GraphWindowConv)
                            for layer in model.original.layers[1:]))
        self.assertIs(build_variant(self.model(), 'ORIGINAL').__class__, self.modules.model.IGSTGNN)

    def test_controls_obey_information_restrictions_and_mass(self):
        window = torch.randn(2, 4, 3, 3, 8)
        router = WindowRouter(8, 3, 5, 'GRAPH_CONTENT')
        with torch.no_grad():
            router.static_logits.normal_()
            router.shared_head.weight.normal_()
            router.interaction_head.weight.normal_()
        static = router(window, 'STATIC')
        changed = router(window+1, 'STATIC')
        self.assertTrue(torch.equal(static, changed))
        shared = router(window, 'SHARED_DYNAMIC')
        self.assertTrue(torch.equal(shared[..., 0, :], shared[..., 4, :]))
        self.assertFalse(torch.equal(shared, router(window+1, 'SHARED_DYNAMIC')))
        joint = router(window)
        self.assertFalse(torch.equal(joint[..., 0, :], joint[..., 4, :]))
        self.assertFalse(torch.equal(joint, router(window+1)))
        self.assertTrue(torch.all(joint >= -1e-6))
        self.assertTrue(torch.all(joint <= 3+1e-6))
        torch.testing.assert_close(joint.sum(-1), torch.full_like(joint.sum(-1), 3))

    def test_nonuniform_selector_matches_explicit_receiver_lag_source_reference(self):
        # Non-symmetric graphs and nonconstant lag signals expose transposes and
        # node/lag flattening mistakes that zero-initialization cannot detect.
        graphs = [torch.tensor([[0., .2], [.8, 0.]]), torch.tensor([[0., .7], [.3, 0.]])]
        for use_pre, dy, sta in itertools.product((False, True), repeat=3):
            if not (use_pre or dy or sta):
                continue
            with self.subTest(flags=(use_pre, dy, sta)):
                native = self.modules.model.STLocalizedConv(2, pre_defined_graph=graphs,
                    k_t=3, k_s=2, use_pre=use_pre, dy_graph=dy, sta_graph=sta, dropout=0.)
                conv = GraphWindowConv(native, 'STATIC')
                with torch.no_grad():
                    native.fc_list_updt.weight.copy_(torch.eye(6))
                    conv.router.static_logits.copy_(torch.arange(conv.router.sources*3).reshape(-1, 3)*.2
                        * torch.arange(1, conv.router.sources+1)[:, None])
                x = torch.arange(1., 1+2*5*2*2).reshape(2, 5, 2, 2)
                dynamic = [g.unsqueeze(0).expand(2, -1, -1) for g in native.get_graph(graphs)]
                static = [graphs[0]]
                actual = conv(x, dynamic, static)
                windows = x.unfold(1, 3, 1).permute(0, 1, 2, 4, 3)
                weights = conv.router(windows)
                expected_parts = [windows.mean(-2)]
                for bank_index, graph in enumerate(conv._supports(dynamic, static)):
                    bank = graph.unsqueeze(0).expand(2, -1, -1) if graph.ndim == 2 else graph
                    source = bank_index//2
                    message = torch.zeros_like(expected_parts[0])
                    for receiver, lag, sender in itertools.product(range(2), range(3), range(2)):
                        message[:, :, receiver] += (bank[:, receiver, lag*2+sender, None, None]
                            * windows[:, :, sender, lag] * weights[:, :, sender, source, lag, None])
                    expected_parts.append(message)
                expected = native.gcn_updt(torch.cat(expected_parts, -1))
                torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-6)

    def test_label_cannot_affect_output_and_inputs_not_mutated(self):
        model = build_variant(self.model(), 'GRAPH_CONTENT').eval()
        x, report = self.batch()
        a = model(x, torch.zeros(2, 12, 4, 1), incident_data=report)
        b = model(x, torch.ones(2, 12, 4, 1)*10000, incident_data=report)
        self.assertTrue(torch.equal(a, b))

    def test_native_masked_mae_reaches_active_heads_then_encoder(self):
        x, report = self.batch()
        target = torch.linspace(1, 200, 96).reshape(2, 12, 4, 1)
        for arm in ARMS[1:]:
            model = build_variant(self.model(), arm).eval()
            loss = self.modules.metrics.masked_mae(model(x, incident_data=report)*30+50, target, target.new_tensor(0.))
            loss.backward()
            router = model.selector.router
            head = router.static_logits if arm == 'STATIC' else router.shared_head.weight
            self.assertIsNotNone(head.grad)
            self.assertGreater(float(head.grad.abs().sum()), 0)
            if arm != 'STATIC':
                self.assertEqual(float(router.encoder.weight.grad.abs().sum()), 0.)
                with torch.no_grad():
                    router.shared_head.weight.normal_(std=.03)
                    router.interaction_head.weight.normal_(std=.03)
                model.zero_grad(set_to_none=True)
                self.modules.metrics.masked_mae(model(x, incident_data=report)*30+50, target, target.new_tensor(0.)).backward()
                self.assertGreater(float(router.encoder.weight.grad.abs().sum()), 0.)
                if arm == 'GRAPH_CONTENT':
                    self.assertGreater(float(router.interaction_head.weight.grad.abs().sum()), 0.)

    def test_initial_native_parameter_gradients_match_with_float_tolerance(self):
        native = self.model().eval()
        candidate = build_variant(copy.deepcopy(native), 'GRAPH_CONTENT').eval()
        x, report = self.batch()
        target = torch.linspace(1, 200, 96).reshape(2, 12, 4, 1)
        for model in (native, candidate):
            self.modules.metrics.masked_mae(model(x, incident_data=report)*30+50,
                target, target.new_tensor(0.)).backward()
        prefix = 'layers.0.dif_layer.localized_st_conv.'
        grads = {key.replace(prefix+'native.', prefix): p.grad
            for key, p in candidate.original.named_parameters() if '.router.' not in key}
        for key, parameter in native.named_parameters():
            if parameter.grad is None:
                self.assertIsNone(grads[key])
            else:
                torch.testing.assert_close(parameter.grad, grads[key], atol=2e-5, rtol=2e-5, msg=key)

    def test_uniform_intervention_restores_after_exception(self):
        model = build_variant(self.model(), 'GRAPH_CONTENT').eval()
        x, report = self.batch()
        with torch.no_grad():
            model.selector.router.static_logits.normal_(std=.3)
        before = model(x, incident_data=report)
        with self.assertRaisesRegex(RuntimeError, 'probe'):
            with model.intervention('UNIFORM'):
                model(x, incident_data=report)
                raise RuntimeError('probe')
        self.assertIsNone(model.selector.override_mode)
        self.assertTrue(torch.equal(before, model(x, incident_data=report)))
        with self.assertRaises(ValueError):
            with model.intervention('UNKNOWN'):
                pass

    def test_diagnostics_weight_tail_batches_by_positions(self):
        native = self.model().layers[0].dif_layer.localized_st_conv
        conv = GraphWindowConv(native, 'GRAPH_CONTENT').eval()
        with torch.no_grad():
            conv.router.shared_head.weight.normal_(std=.1)
        x = torch.randn(3, 5, 4, 8)
        graphs = [g.unsqueeze(0).expand(3, -1, -1) for g in native.pre_defined_graph]
        static = [torch.eye(4)]
        conv.start_diagnostics(); conv(x, graphs, static)
        whole = conv.finish_diagnostics()
        conv.start_diagnostics()
        for sl in (slice(0, 2), slice(2, 3)):
            conv(x[sl], [g[sl] for g in graphs], static)
        split = conv.finish_diagnostics()
        self.assertEqual(whole['positions'], split['positions'])
        torch.testing.assert_close(torch.tensor(whole['weight_mean']), torch.tensor(split['weight_mean']))
        self.assertEqual(conv.finish_diagnostics()['calls'], 0)

    def test_invalid_source_count_and_shape_are_rejected(self):
        native = self.model().layers[0].dif_layer.localized_st_conv
        conv = GraphWindowConv(native, 'STATIC')
        x = torch.rand(2, 5, 4, 8)
        with self.assertRaisesRegex(ValueError, 'Dynamic graph count'):
            conv(x, [], [torch.eye(4)])
        with self.assertRaisesRegex(ValueError, 'input must'):
            conv(x[:, :2], [], [])
        with self.assertRaises(ValueError):
            build_variant(self.model(), 'UNKNOWN')
        with self.assertRaisesRegex(ValueError, 'twice'):
            build_variant(build_variant(self.model(), 'STATIC').original, 'STATIC')


if __name__ == '__main__':
    unittest.main()
