"""Counterexamples and contracts for the v03 incident-state adapter."""
import copy
import json
import unittest

import torch

from experiments.chronological.original_igstgnn_runtime import tensor_tree_hash, rng_fingerprint
from experiments.chronological.original_igstgnn_source import get_original_modules
from experiments.chronological.unified_incident_adapter_v03 import (
    ARMS, DEFAULT_ARMS, IncidentGraph, OrderedTrafficSummary, ReportEncoder,
    TrafficCore, UnifiedIncidentIGSTGNN, arm_spec, build_variant, common_traffic_state)
from tests import test_original_isp as fixtures


class UnifiedAdapterV03Tests(unittest.TestCase):
    model = fixtures.ISPTests.model
    batch = staticmethod(fixtures.ISPTests.batch)

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.modules = get_original_modules()

    @staticmethod
    def activate(model):
        with torch.no_grad():
            torch.manual_seed(7)
            model.adapter.output.weight.normal_(0, .2)
            for layer in model.adapter.graph.query:
                layer.weight.normal_(0, .3)
            for layer in model.adapter.graph.strength:
                layer.weight.normal_(0, .2)

    def test_default_arms_are_information_and_direct_controls(self):
        self.assertEqual(DEFAULT_ARMS, ('ORIGINAL', 'TRAFFIC_CORE', 'H_ONLY', 'R_ONLY', 'BOTH', 'DIRECT'))
        self.assertEqual(arm_spec('H_ONLY')['information'], 'H')
        self.assertEqual(arm_spec('DIRECT')['state_mode'], 'direct')
        with self.assertRaises(ValueError):
            arm_spec('invented')

    def test_all_arms_same_traffic_and_adapter_initialization_preserves_rng(self):
        original = self.model()
        before = rng_fingerprint()
        variants = [build_variant(copy.deepcopy(original), arm) for arm in ARMS]
        self.assertEqual(before, rng_fingerprint())
        self.assertEqual(len({tensor_tree_hash(common_traffic_state(m)) for m in variants}), 1)
        self.assertEqual(len({tensor_tree_hash(m.adapter.state_dict()) for m in variants if hasattr(m, 'adapter')}), 1)

    def test_constant_history_has_zero_change_before_cutoff_normalization(self):
        candidate = UnifiedIncidentIGSTGNN(self.model())
        x, _ = self.batch()
        x[:, :, :, 0] = .25
        h, raw, *_ = candidate.core.prepare_with_summary(x)
        c = raw.shape[-1]
        statistics = candidate.adapter.summary.statistics(raw)
        self.assertTrue(torch.equal(statistics[..., c:2*c], torch.zeros_like(statistics[..., c:2*c])))
        self.assertTrue(torch.equal(statistics[..., 2*c:], torch.zeros_like(statistics[..., 2*c:])))
        self.assertGreater(float((h[:, -1]-h[:, 0]).detach().abs().max()), .1)

    def test_ordered_summary_distinguishes_histories_with_equal_old_statistics(self):
        summary = OrderedTrafficSummary(1, 4, 12)
        a = torch.zeros(1, 12, 1, 1)
        a[:, 3] = 1
        b = a.clone()
        b[:, 3], b[:, 8] = 0, 1
        self.assertTrue(torch.equal(summary.statistics(a), summary.statistics(b)))
        self.assertFalse(torch.equal(summary(a), summary(b)))

    def test_zero_adapter_exact_core_and_original_recovery_with_dropout(self):
        original = self.model()
        x, report = self.batch()
        for training in (False, True):
            for arm in ('TIME', 'GRAPH', 'BOTH', 'H_ONLY', 'R_ONLY', 'DIRECT', 'SEPARATE', 'STACKED'):
                candidate = build_variant(copy.deepcopy(original), arm).train(training)
                reference = copy.deepcopy(original).train(training) if arm == 'STACKED' else TrafficCore(copy.deepcopy(original)).train(training)
                torch.manual_seed(17)
                expected = reference(x, incident_data=report)
                expected_rng = torch.get_rng_state()
                torch.manual_seed(17)
                actual = candidate(x, incident_data=report)
                self.assertTrue(torch.equal(actual, expected), (training, arm, (actual-expected).abs().max()))
                self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))

    def test_initial_traffic_gradients_are_preserved(self):
        original = self.model()
        core = TrafficCore(copy.deepcopy(original)).eval()
        candidate = build_variant(copy.deepcopy(original), 'BOTH').eval()
        x, report = self.batch()
        core(x).square().mean().backward()
        candidate(x, incident_data=report).square().mean().backward()
        for (name, left), (other, right) in zip(core.named_parameters(), candidate.core.named_parameters()):
            self.assertEqual(name, other)
            self.assertEqual(left.grad is None, right.grad is None)
            if left.grad is not None:
                torch.testing.assert_close(left.grad, right.grad, atol=2e-6, rtol=2e-5, msg=name)

    def test_relation_changes_neighbor_choice_when_traffic_is_uniform(self):
        roads = torch.tensor([[[0, 1, 1], [1, 0, 0], [1, 0, 0]]]*2, dtype=torch.bool)
        graph = IncidentGraph(2, 1, roads)
        nodes, state = torch.ones(1, 3, 2), torch.ones(1, 2)
        relation = torch.zeros(1, 3, 3)
        relation[:, 1, 0], relation[:, 2, 0] = -1., 1.
        with torch.no_grad():
            for query, key in zip(graph.query, graph.key):
                query.weight.zero_()
                query.weight[0, 0] = 1.
                key.weight.zero_()
                key.weight[0, 4] = 1.
        base = roads[0].float().unsqueeze(0) * .5
        values = graph(nodes, state, relation, [base, base], torch.ones(1, 3, dtype=torch.bool), torch.ones(1, 3))
        self.assertGreater(float(values[0][0, 0, 2].detach()), float(values[0][0, 0, 1].detach()))
        reversed_relation = relation.flip(1)
        other = graph(nodes, state, reversed_relation, [base, base], torch.ones(1, 3, dtype=torch.bool), torch.ones(1, 3))
        self.assertFalse(torch.equal(values[0], other[0]))

    def test_single_neighbor_strength_changes_and_offroad_remains_exact(self):
        roads = torch.tensor([[[0, 1, 0], [0, 0, 1], [0, 0, 0]]]*2, dtype=torch.bool)
        graph = IncidentGraph(2, 2, roads)
        base = torch.tensor([[[.1, .8, .1], [.1, .1, .8], [.2, .3, .5]]])
        nodes, state = torch.ones(1, 3, 2), torch.ones(1, 2)
        args = (nodes, state, torch.zeros(1, 3, 3), [base, base],
                torch.tensor([[True, False, True]]), torch.ones(1, 3))
        initial = graph(*args)
        self.assertTrue(torch.equal(initial[0], base))
        with torch.no_grad():
            for layer in graph.strength:
                layer.bias.fill_(-2)
        result = graph(*args)[0]
        self.assertGreaterEqual(float(result.detach().min()), 0)
        self.assertLess(float(result[0, 0, 1].detach()), float(base[0, 0, 1]) * .2)
        self.assertTrue(torch.equal(result[:, ~roads[0]], base[:, ~roads[0]]))
        self.assertTrue(torch.equal(result[:, 1], base[:, 1]))

    def test_graph_zero_initialization_keeps_live_query_and_strength_gradients(self):
        roads = ~torch.eye(3, dtype=torch.bool)[None].expand(2, -1, -1)
        graph = IncidentGraph(2, 2, roads)
        base = torch.softmax(torch.randn(2, 3, 3), -1)
        result = graph(torch.randn(2, 3, 2), torch.randn(2, 2), torch.randn(2, 3, 3), [base, base],
                       torch.ones(2, 3, dtype=torch.bool), torch.ones(2, 3))
        self.assertTrue(torch.equal(result[0], base))
        result[0].square().sum().backward()
        self.assertGreater(float(graph.query[0].weight.grad.abs().sum()), 0)
        self.assertGreater(float(graph.strength[0].weight.grad.abs().sum()), 0)

    def test_no_report_unavailable_zero_support_and_nonfinite_placeholder_fallback(self):
        candidate = build_variant(self.model(), 'BOTH').eval()
        self.activate(candidate)
        x, report = self.batch()
        unavailable = copy.deepcopy(report)
        unavailable.update(available=torch.zeros(2, dtype=torch.bool),
                           relation_available=torch.zeros(2, dtype=torch.bool))
        unavailable['distances'].fill_(float('nan'))
        unavailable['incident'].fill_(float('nan'))
        empty = dict(report, distances=torch.zeros_like(report['distances']))
        with torch.no_grad():
            expected = candidate.core(x)
            for absent in (None, unavailable, empty):
                self.assertTrue(torch.equal(candidate(x, incident_data=absent), expected))
            mixed = dict(report, available=torch.tensor([True, False]), relation_available=torch.ones(2, dtype=torch.bool))
            mixed['distances'] = mixed['distances'].clone()
            mixed['distances'][1] = float('nan')
            actual = candidate(x, incident_data=mixed)
            self.assertTrue(torch.equal(actual[1], expected[1]))
            self.assertFalse(torch.equal(actual[0], expected[0]))

    def test_available_nonfinite_relations_rejected_and_fractional_categories_unknown(self):
        original = self.model()
        encoder = ReportEncoder(original.icsf_module)
        result = encoder.lookup(encoder.desc_embedding, torch.tensor([float('nan'), .5, 0.]))
        self.assertTrue(torch.equal(result[0], encoder.desc_embedding.weight[-1]))
        self.assertTrue(torch.equal(result[1], encoder.desc_embedding.weight[-1]))
        self.assertTrue(torch.equal(result[2], encoder.desc_embedding.weight[0]))
        candidate = build_variant(original, 'BOTH')
        x, report = self.batch()
        report['distances'][0, 0, 0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            candidate(x, incident_data=report)

    def test_h_control_never_uses_report_support_categories_or_availability(self):
        candidate = build_variant(self.model(), 'H_ONLY').eval()
        self.activate(candidate)
        x, report = self.batch()
        changed = copy.deepcopy(report)
        changed['distances'].fill_(float('nan'))
        changed['incident'][:, :3] = float('nan')
        changed['position'] = torch.ones(2) * float('nan')
        changed['available'] = torch.zeros(2, dtype=torch.bool)
        with torch.no_grad():
            a = candidate(x, incident_data=report)
            b = candidate(x, incident_data=changed)
        self.assertTrue(torch.equal(a, b))
        self.assertEqual(candidate.diagnostics()['supported_positions'], 8)

    def test_r_control_ignores_incident_semantics_keeps_relation_and_holiday(self):
        candidate = build_variant(self.model(), 'R_ONLY').eval()
        self.activate(candidate)
        x, report = self.batch()
        changed = copy.deepcopy(report)
        changed['incident'][:, 1:3] = 99
        changed['position'] = torch.ones(2) * 99
        with torch.no_grad():
            a = candidate(x, incident_data=report)
            b = candidate(x, incident_data=changed)
        self.assertTrue(torch.equal(a, b))

    def test_time_is_output_space_bounded_and_graph_mode_skips_time_decoder(self):
        candidate = build_variant(self.model(), 'TIME').eval()
        self.activate(candidate)
        x, report = self.batch()
        with torch.no_grad():
            baseline = candidate.core(x)
            actual = candidate(x, incident_data=report)
        self.assertLessEqual(float((actual-baseline).abs().max()), .1000001)
        self.assertTrue(torch.equal(actual[:, :, 2:], baseline[:, :, 2:]))
        self.assertTrue((actual[:, :3] != baseline[:, :3]).any())
        self.assertTrue((actual[:, 9:] != baseline[:, 9:]).any())
        candidate = build_variant(self.model(), 'GRAPH')
        called = []
        handle = candidate.adapter.output.register_forward_hook(lambda *args: called.append(True))
        try:
            candidate(x, incident_data=report)
        finally:
            handle.remove()
        self.assertEqual(called, [])

    def test_output_correction_gradient_survives_dead_native_relu(self):
        candidate = build_variant(self.model(), 'TIME').eval()
        with torch.no_grad():
            candidate.core.out_fc_1.weight.zero_()
            candidate.core.out_fc_1.bias.fill_(-100)
        x, report = self.batch()
        prediction = candidate(x, incident_data=report)
        (prediction-torch.ones_like(prediction)).abs().mean().backward()
        self.assertGreater(float(candidate.adapter.output.weight.grad.abs().sum()), 0)
        self.assertEqual(float(candidate.core.out_fc_1.weight.grad.abs().sum()), 0)

    def test_task_mae_heads_learn_then_state_parameters_learn(self):
        x, report = self.batch()
        for arm in ('TIME', 'GRAPH', 'BOTH', 'DIRECT'):
            candidate = build_variant(self.model(), arm).eval()
            optimizer = torch.optim.Adam(candidate.parameters(), lr=.002)
            target = torch.ones(2, 12, 4, 1) * .4
            (candidate(x, incident_data=report)-target).abs().mean().backward()
            if arm != 'GRAPH':
                self.assertGreater(float(candidate.adapter.output.weight.grad.abs().sum()), 0)
            if arm != 'TIME':
                self.assertGreater(float(candidate.adapter.graph.query[0].weight.grad.abs().sum()), 0)
            optimizer.step()
            optimizer.zero_grad()
            (candidate(x, incident_data=report)-target).abs().mean().backward()
            parameter = candidate.adapter.direct[0].weight if arm == 'DIRECT' else candidate.adapter.transition.weight_hh
            self.assertGreater(float(parameter.grad.abs().sum()), 0)
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in candidate.parameters() if p.grad is not None))

    def test_graph_first_block_native_and_diagnostics_json(self):
        candidate = build_variant(self.model(), 'BOTH').eval()
        x, report = self.batch()
        with torch.no_grad():
            h, raw, u, d, tod, dow = candidate.core.prepare_with_summary(x)
            _, _, bases = candidate.core.graphs(h, u, d, tod, dow)
            values = candidate.adapter(raw, report, None, tod, dow)
            schedule = candidate.adapter.graph_schedule(candidate.core, values, bases)
            candidate(x, incident_data=report)
        self.assertIsNone(schedule[0])
        self.assertEqual(len(schedule), 4)
        json.dumps(candidate.batch_diagnostics(), allow_nan=False)
        self.assertFalse(candidate.diagnostics()['graph_row_mass_preserved'])

    def test_label_independence_and_stacked_disable_is_unambiguous(self):
        candidate = build_variant(self.model(), 'BOTH').eval()
        self.activate(candidate)
        x, report = self.batch()
        before = tensor_tree_hash((x, report))
        with torch.no_grad():
            a = candidate(x, label=torch.zeros(2, 12, 4, 1), incident_data=report)
            b = candidate(x, label=torch.ones(2, 12, 4, 1)*999, incident_data=report)
            with candidate.intervention(report=False):
                self.assertTrue(torch.equal(candidate(x, incident_data=report), candidate.core(x)))
        self.assertTrue(torch.equal(a, b))
        self.assertEqual(before, tensor_tree_hash((x, report)))
        stacked = build_variant(self.model(), 'STACKED').eval()
        with self.assertRaisesRegex(ValueError, 'ambiguous'):
            with stacked.intervention(report=False):
                pass
        with torch.no_grad(), stacked.intervention(all_reports=True):
            self.assertTrue(torch.equal(stacked(x, incident_data=report), stacked.original(x, incident_data=None)))
        with self.assertRaisesRegex(ValueError, 'complete available'):
            stacked(x, incident_data=dict(report, available=torch.zeros(2, dtype=torch.bool)))


if __name__ == '__main__':
    unittest.main()
