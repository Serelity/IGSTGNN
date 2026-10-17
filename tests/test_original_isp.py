"""ISP contracts: original recovery, causal inputs, graph direction and learning."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch

from experiments.chronological.original_isp import ISPIGSTGNN, IncidentState, supported_reweight
from experiments.chronological.original_igstgnn_runtime import tensor_tree_hash, rng_fingerprint
from experiments.chronological.original_igstgnn_source import get_original_modules, verify_original_source


class ISPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.modules = get_original_modules()

    def model(self):
        directory = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(directory.cleanup)
        for name in ('desc_mapping.json', 'type_mapping.json'):
            (Path(directory.name) / name).write_text(json.dumps({'a': 0, 'b': 1}))
        graph = torch.tensor([[0., .3, .7, 0.], [.2, 0., .8, 0.], [.4, .3, 0., .3], [0., .4, .6, 0.]])
        torch.manual_seed(19)
        return self.modules.model.IGSTGNN(dict(num_feat=1, num_hidden=8, node_hidden=4,
            time_emb_dim=4, tpd=288, k_s=2, k_t=3, layer=5, gap=3, dropout=.1,
            dataset='fixture', data_path=directory.name, use_sensor_info=False,
            adjs=[graph, graph.T]), node_num=4, input_dim=3, output_dim=1, seq_len=12, horizon=12)

    @staticmethod
    def batch():
        torch.manual_seed(11)
        x = torch.rand(2, 12, 4, 3) * .2
        report = dict(incident=torch.tensor([[0., 1., 1., 0.], [0., 0., 0., 1.]]),
                      position=torch.tensor([0, 1]), distances=torch.zeros(2, 4, 3))
        report['distances'][:, :2, 0] = 1
        return x, report

    @staticmethod
    def activate(model):
        with torch.no_grad():
            torch.manual_seed(23)
            model.state.output.weight.normal_(0, .2)
            for layer in list(model.graph.query_node) + list(model.graph.query_state):
                layer.weight.normal_(0, .3)

    def test_zero_scores_exact_and_differentiable(self):
        base = torch.tensor([[[.1, .3, .6], [.3, .2, .5], [.2, .7, .1]]], dtype=torch.float64)
        roads = ~torch.eye(3, dtype=torch.bool)
        scores = torch.zeros_like(base, requires_grad=True)
        actual = supported_reweight(base, scores, roads, torch.tensor([.2]), torch.ones(1, 3, dtype=torch.bool))
        self.assertTrue(torch.equal(base, actual))
        actual.square().sum().backward()
        self.assertGreater(float(scores.grad.abs().sum()), 0)

    def test_graph_mass_nonnegative_offroad_and_inactive_rows(self):
        torch.manual_seed(10)
        base = torch.softmax(torch.randn(2, 4, 4, dtype=torch.float64), -1)
        roads = torch.tensor([[0, 1, 1, 0], [1, 0, 1, 0], [1, 1, 0, 1], [0, 0, 0, 0]], dtype=torch.bool)
        active = torch.tensor([[True, False, True, True], [True, True, False, True]])
        actual = supported_reweight(base, torch.randn_like(base) * 100, roads,
                                    torch.tensor([1., .25]), active)
        self.assertTrue(torch.isfinite(actual).all())
        self.assertGreaterEqual(float(actual.min()), 0)
        torch.testing.assert_close(actual.sum(-1), base.sum(-1), atol=1e-14, rtol=1e-14)
        self.assertTrue(torch.equal(actual[:, ~roads], base[:, ~roads]))
        self.assertTrue(torch.equal(actual[~active], base[~active]))
        self.assertTrue(torch.equal(actual[:, 3], base[:, 3]))

    def test_graph_gradient_finite_difference(self):
        base = torch.tensor([[[0., .3, .7], [.3, 0., .7], [.4, .6, 0.]]], dtype=torch.float64)
        score = torch.randn_like(base, requires_grad=True)
        def objective(value):
            result = supported_reweight(base, value, base[0] > 0, torch.tensor([.25]),
                                        torch.ones(1, 3, dtype=torch.bool))
            return result.square().sum()
        objective(score).backward()
        plus, minus = score.detach().clone(), score.detach().clone()
        plus[0, 0, 1] += 1e-6
        minus[0, 0, 1] -= 1e-6
        torch.testing.assert_close(score.grad[0, 0, 1], (objective(plus) - objective(minus)) / 2e-6,
                                   atol=1e-8, rtol=1e-6)

    def test_receiver_source_direction(self):
        graph = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
        sources = torch.tensor([[True, False, False]])
        reached = sources | (sources.float() @ graph.T > 0)
        self.assertEqual(reached.tolist(), [[True, True, False]])
        values = torch.tensor([[[7.], [0.], [0.]]])
        self.assertEqual(torch.bmm(graph.unsqueeze(0), values).squeeze().tolist(), [0., 7., 0.])

    def test_constructor_preserves_rng_and_original_tensors(self):
        original = self.model()
        before, rng = tensor_tree_hash(original.state_dict()), rng_fingerprint()
        wrapper = ISPIGSTGNN(original)
        self.assertEqual(before, tensor_tree_hash(wrapper.original.state_dict()))
        self.assertEqual(rng, rng_fingerprint())
        self.assertEqual(len(verify_original_source()['files']), 9)

    def test_modes_have_identical_initial_parameters(self):
        original = self.model()
        state = ISPIGSTGNN(copy.deepcopy(original), mode='STATE')
        propagation = ISPIGSTGNN(copy.deepcopy(original), mode='STATE_PROPAGATION')
        self.assertEqual(tensor_tree_hash(state.state_dict()), tensor_tree_hash(propagation.state_dict()))

    def test_initial_complete_forward_and_dropout_rng_exact(self):
        original = self.model()
        candidates = [ISPIGSTGNN(copy.deepcopy(original), mode=mode) for mode in ('STATE', 'STATE_PROPAGATION')]
        x, report = self.batch()
        for training in (False, True):
            original.train(training)
            torch.manual_seed(35)
            with torch.no_grad():
                expected = original(x, incident_data=report)
            rng = torch.get_rng_state()
            for candidate in candidates:
                candidate.train(training)
                torch.manual_seed(35)
                with torch.no_grad():
                    actual = candidate(x, incident_data=report)
                self.assertTrue(torch.equal(actual, expected), (training, candidate.mode, (actual - expected).abs().max()))
                self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_initial_backbone_gradients_close(self):
        original = self.model().eval()
        wrapper = ISPIGSTGNN(copy.deepcopy(original)).eval()
        x, report = self.batch()
        original(x, incident_data=report).square().mean().backward()
        wrapper(x, incident_data=report).square().mean().backward()
        for (name, first), (other, second) in zip(original.named_parameters(), wrapper.original.named_parameters()):
            self.assertEqual(name, other)
            self.assertEqual(first.grad is None, second.grad is None, name)
            if first.grad is not None:
                torch.testing.assert_close(first.grad, second.grad, atol=2e-6, rtol=2e-5, msg=name)

    def test_no_report_preserves_native_even_after_learning(self):
        original = self.model().eval()
        wrapper = ISPIGSTGNN(copy.deepcopy(original)).eval()
        self.activate(wrapper)
        x, _ = self.batch()
        self.assertTrue(torch.equal(wrapper(x), original(x)))

    def test_zero_support_preserves_native_normalization_after_learning(self):
        original = self.model().eval()
        wrapper = ISPIGSTGNN(copy.deepcopy(original)).eval()
        self.activate(wrapper)
        x, report = self.batch()
        report['distances'].zero_()
        self.assertTrue(torch.equal(wrapper(x, incident_data=report), original(x, incident_data=report)))

    def test_batch_reordering_and_fresh_state(self):
        wrapper = ISPIGSTGNN(self.model()).eval()
        self.activate(wrapper)
        x, report = self.batch()
        with torch.no_grad():
            first = wrapper(x, incident_data=report)
            reordered = wrapper(x.flip(0), incident_data={k: v.flip(0) for k, v in report.items()})
            wrapper(x + .05, incident_data=report)
            again = wrapper(x, incident_data=report)
        torch.testing.assert_close(reordered.flip(0), first, atol=1e-6, rtol=1e-6)
        self.assertTrue(torch.equal(first, again))

    def test_label_ignored_and_inputs_unchanged(self):
        wrapper = ISPIGSTGNN(self.model()).eval()
        self.activate(wrapper)
        x, report = self.batch()
        before = tensor_tree_hash((x, report))
        with torch.no_grad():
            a = wrapper(x, torch.zeros(2, 12, 4, 1), incident_data=report)
            b = wrapper(x, torch.full((2, 12, 4, 1), 999.), incident_data=report)
        self.assertTrue(torch.equal(a, b))
        self.assertEqual(before, tensor_tree_hash((x, report)))

    def test_historical_state_encoding_does_not_read_report(self):
        state = IncidentState(8, 4, 16, 4, .1).eval()
        x = torch.randn(2, 12, 3, 8)
        support = torch.ones(2, 3, dtype=torch.bool)
        nodes_a, states_a, _, _ = state(x, torch.zeros(2, 8), support)
        nodes_b, states_b, _, _ = state(x, torch.ones(2, 8), support)
        self.assertTrue(torch.equal(nodes_a, nodes_b))
        self.assertFalse(torch.equal(states_a, states_b))

    def test_new_report_off_recovers_native_and_context_restores(self):
        wrapper = ISPIGSTGNN(self.model()).eval()
        self.activate(wrapper)
        x, report = self.batch()
        with torch.no_grad():
            expected = wrapper.original(x, incident_data=report)
            on = wrapper(x, incident_data=report)
            with wrapper.intervention(new_report=False):
                off = wrapper(x, incident_data=report)
            again = wrapper(x, incident_data=report)
        self.assertTrue(torch.equal(expected, off))
        self.assertFalse(torch.equal(on, off))
        self.assertTrue(torch.equal(on, again))
        with self.assertRaises(RuntimeError):
            with wrapper.intervention(state=False, propagation=False):
                raise RuntimeError('test')
        self.assertTrue(wrapper._state_enabled and wrapper._graph_enabled)

    def test_two_paths_off_recovers_native_after_learning(self):
        wrapper = ISPIGSTGNN(self.model()).eval()
        self.activate(wrapper)
        x, report = self.batch()
        with torch.no_grad(), wrapper.intervention(state=False, propagation=False):
            self.assertTrue(torch.equal(wrapper(x, incident_data=report), wrapper.original(x, incident_data=report)))

    def test_new_paths_learn_from_zero_and_state_gradients_wake(self):
        wrapper = ISPIGSTGNN(self.model()).eval()
        x, report = self.batch()
        optimizer = torch.optim.Adam(wrapper.parameters(), lr=.002, weight_decay=0)
        wrapper(x, incident_data=report).square().mean().backward()
        self.assertGreater(float(wrapper.state.output.weight.grad.abs().sum()), 0)
        query_grad = sum(float(layer.weight.grad.abs().sum()) for layer in wrapper.graph.query_node)
        self.assertGreater(query_grad, 0)
        optimizer.step()
        optimizer.zero_grad()
        wrapper(x, incident_data=report).square().mean().backward()
        self.assertGreater(float(wrapper.state.transition.weight_hh.grad.abs().sum()), 0)
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in wrapper.parameters() if p.grad is not None))

    def test_graph_changes_future_calls_only(self):
        wrapper = ISPIGSTGNN(self.model()).eval()
        self.activate(wrapper)
        x, report = self.batch()
        captured = []
        def capture(module, args):
            captured.append([value.detach().clone() for value in args[1]])
        handles = [layer.dif_layer.localized_st_conv.register_forward_pre_hook(capture) for layer in wrapper.original.layers]
        try:
            with torch.no_grad(), wrapper.intervention(propagation=False):
                wrapper(x, incident_data=report)
            native = captured[:]
            captured.clear()
            with torch.no_grad():
                wrapper(x, incident_data=report)
            self.assertEqual(len(native), 20)
            # Learned future messages use compact graph residuals; only the
            # five historical calls execute the unchanged full conv forward.
            self.assertEqual(len(captured), 5)
            for k in range(0, 20, 4):
                self.assertTrue(all(torch.equal(a, b) for a, b in zip(native[k], captured[k // 4])))
            with torch.no_grad(), wrapper.intervention(state=False, propagation=False):
                expected = wrapper(x, incident_data=report)
            with torch.no_grad(), wrapper.intervention(state=False):
                changed = wrapper(x, incident_data=report)
            self.assertFalse(torch.equal(expected, changed))
        finally:
            for handle in handles:
                handle.remove()

    def test_diagnostics_are_json_safe_and_honest(self):
        wrapper = ISPIGSTGNN(self.model()).eval()
        x, report = self.batch()
        with torch.no_grad():
            wrapper(x, incident_data=report)
        result = wrapper.diagnostics()
        json.dumps(result, allow_nan=False)
        self.assertFalse(result['physics_certified'])
        self.assertFalse(result['persistent_cross_sample_state'])
        self.assertEqual(result['forecast_blocks'], 4)
        self.assertEqual(result['state_delta_abs_max'], 0)

    def test_compact_future_convolution_matches_expanded_reference(self):
        wrapper = ISPIGSTGNN(self.model()).eval()
        self.activate(wrapper)
        x, report = self.batch()
        native = wrapper.original
        traffic, u, d, tod, dow = native._prepare_inputs(x)
        traffic = native.embedding(traffic)
        enhanced, _ = native.icsf_module(traffic, report, None, tod[:, -1, 0], dow[:, -1, 0])
        static, dynamic, bases = wrapper._original_graphs(enhanced, u, d, tod, dow)
        nodes, states, intensity, _ = wrapper.state(traffic,
            native.icsf_module.embed_incident_features(report, tod[:, -1, 0], dow[:, -1, 0]),
            report['distances'].abs().sum(-1) > 0)
        corrected = wrapper.graph(nodes, states[:, 1], bases, wrapper.road_support,
            wrapper.max_graph_mix * intensity[:, 1], torch.ones(2, 4, dtype=torch.bool))
        constructor = native.dynamic_graph_constructor
        ordered = constructor.multi_order(corrected)
        compact = [g for direction in ordered for g in direction]
        expanded = constructor.st_localization(ordered)
        convolution = native.layers[0].dif_layer.localized_st_conv
        window = enhanced[:, -3:]
        actual = wrapper._future_convolution(convolution, window, dynamic, static, compact)
        expected = convolution(window, expanded, static)
        torch.testing.assert_close(actual, expected, atol=3e-7, rtol=3e-6)
        self.assertTrue(all(g.shape == (2, 4, 4) for g in compact))

    def test_invalid_configuration(self):
        original = self.model()
        for kwargs in (dict(mode='UNKNOWN'), dict(max_graph_mix=2), dict(state_dim=0)):
            with self.assertRaises(ValueError):
                ISPIGSTGNN(original, **kwargs)


if __name__ == '__main__':
    unittest.main()
