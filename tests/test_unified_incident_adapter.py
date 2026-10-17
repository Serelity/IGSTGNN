"""Contracts for the report-free core and the one-entry incident adapter."""
import copy
import json
import unittest

import torch

from experiments.chronological.unified_incident_adapter import (
    ARMS, ReportEncoder, TrafficCore, UnifiedIncidentIGSTGNN, build_variant, common_traffic_state)
from experiments.chronological.original_igstgnn_runtime import tensor_tree_hash, rng_fingerprint
from experiments.chronological.original_igstgnn_source import get_original_modules, verify_original_source
from tests import test_original_isp as isp_fixtures


class UnifiedAdapterTests(unittest.TestCase):
    model = isp_fixtures.ISPTests.model
    batch = staticmethod(isp_fixtures.ISPTests.batch)

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.modules = get_original_modules()

    @staticmethod
    def activate(candidate):
        torch.manual_seed(23)
        with torch.no_grad():
            candidate.adapter.output_state.weight.normal_(0, .2)
            candidate.adapter.output_node.weight.normal_(0, .2)
            for layer in list(candidate.adapter.graph.query_node) + list(candidate.adapter.graph.query_state):
                layer.weight.normal_(0, .3)

    def test_core_has_no_legacy_report_parameters_or_modules(self):
        core = TrafficCore(self.model())
        self.assertFalse(hasattr(core, 'icsf_module'))
        self.assertFalse(hasattr(core, 'tiid_module'))
        self.assertFalse(any('incident' in n or 'icsf' in n or 'tiid' in n for n, _ in core.named_parameters()))
        self.assertFalse(core.diagnostics()['report_read_by_traffic_core'])

    def test_core_always_normalizes_latest_slot_and_ignores_reports(self):
        core = TrafficCore(self.model()).eval()
        x, report = self.batch()
        encoded, *_ = core.prepare(x)
        raw = core.embedding(core._prepare(core, x)[0])
        self.assertTrue(torch.equal(encoded[:, :-1], raw[:, :-1]))
        self.assertTrue(torch.equal(encoded[:, -1], core.latest_norm(raw[:, -1])))
        with torch.no_grad():
            self.assertTrue(torch.equal(core(x), core(x, incident_data=report)))

    def test_constructor_preserves_rng_and_shared_initializations(self):
        original = self.model()
        original_hash, rng = tensor_tree_hash(original.state_dict()), rng_fingerprint()
        variants = {arm: build_variant(copy.deepcopy(original), arm) for arm in ARMS}
        self.assertEqual(rng, rng_fingerprint())
        self.assertEqual(original_hash, tensor_tree_hash(original.state_dict()))
        self.assertEqual(len({tensor_tree_hash(common_traffic_state(m)) for m in variants.values()}), 1)
        self.assertEqual(len({tensor_tree_hash(m.adapter.state_dict()) for m in variants.values()
                              if hasattr(m, 'adapter')}), 1)
        self.assertEqual(len(verify_original_source()['files']), 9)

    def test_zero_adapter_exact_core_recovery_and_dropout_rng(self):
        original = self.model()
        core = TrafficCore(copy.deepcopy(original))
        candidates = [UnifiedIncidentIGSTGNN(copy.deepcopy(original), mode=m) for m in ('TIME', 'GRAPH', 'BOTH')]
        x, report = self.batch()
        for training in (False, True):
            core.train(training)
            torch.manual_seed(35)
            with torch.no_grad():
                expected = core(x, incident_data=report)
            rng = torch.get_rng_state()
            for candidate in candidates:
                candidate.train(training)
                torch.manual_seed(35)
                with torch.no_grad():
                    actual = candidate(x, incident_data=report)
                self.assertTrue(torch.equal(expected, actual), (training, candidate.mode, (expected-actual).abs().max()))
                self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_stacked_zero_adapter_recovers_full_original(self):
        original = self.model()
        candidate = UnifiedIncidentIGSTGNN(copy.deepcopy(original), mode='STACKED')
        x, report = self.batch()
        for training in (False, True):
            original.train(training)
            candidate.train(training)
            for input_report in (None, report):
                torch.manual_seed(35)
                with torch.no_grad():
                    expected = original(x, incident_data=input_report)
                rng = torch.get_rng_state()
                torch.manual_seed(35)
                with torch.no_grad():
                    actual = candidate(x, incident_data=input_report)
                self.assertTrue(torch.equal(expected, actual))
                self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_initial_core_gradients_are_preserved(self):
        original = self.model()
        core = TrafficCore(copy.deepcopy(original)).eval()
        candidate = UnifiedIncidentIGSTGNN(copy.deepcopy(original)).eval()
        x, report = self.batch()
        core(x).square().mean().backward()
        candidate(x, incident_data=report).square().mean().backward()
        for (name, p), (other, q) in zip(core.named_parameters(), candidate.core.named_parameters()):
            self.assertEqual(name, other)
            self.assertEqual(p.grad is None, q.grad is None)
            if p.grad is not None:
                torch.testing.assert_close(p.grad, q.grad, atol=2e-6, rtol=2e-5, msg=name)

    def test_sensor_context_and_known_report_encoding_match_native(self):
        original = self.model()
        original.icsf_module = self.modules.model.IncidentContextSpatialFusion(8, 4, True, 2, 2)
        candidate = UnifiedIncidentIGSTGNN(copy.deepcopy(original)).eval()
        x, report = self.batch()
        sensors = {name: torch.zeros(2, 4) for name in ('sensor_type', 'surface', 'roadway_use', 'road_width')}
        sensors['speed_limit'] = torch.ones(2, 4)*50
        h, _, _, tod, dow = candidate.core.prepare(x)
        with torch.no_grad():
            a = candidate.adapter.encoder(report, tod[:, -1, 0], dow[:, -1, 0])
            b = original.icsf_module.embed_incident_features(report, tod[:, -1, 0], dow[:, -1, 0])
            self.assertTrue(torch.equal(a, b))
            self.assertEqual(candidate(x, incident_data=report, sensor_data=sensors).shape, (2, 12, 4, 1))
            self.assertEqual(candidate.adapter.encoder.sensors(sensors, h[:, -1]).shape, (2, 4, 8))
        self.assertFalse(any(isinstance(m, torch.nn.Dropout) for m in candidate.adapter.modules()))

    def test_unique_report_entry_and_no_legacy_calls(self):
        original = self.model()
        x, report = self.batch()
        for mode in ('TIME', 'GRAPH', 'BOTH'):
            counts = dict(encoder=0, legacy=0)
            candidate = UnifiedIncidentIGSTGNN(copy.deepcopy(original), mode=mode).eval()
            self.assertEqual(sum(isinstance(m, ReportEncoder) for m in candidate.modules()), 1)
            def record(module, args, output):
                counts['encoder'] += 1
            handle = candidate.adapter.encoder.register_forward_hook(record)
            try:
                with torch.no_grad():
                    candidate(x, incident_data=report)
            finally:
                handle.remove()
            self.assertEqual(counts['encoder'], 1)
            self.assertFalse(any(type(m).__name__ in ('IncidentContextSpatialFusion', 'TemporalIncidentImpactDecay')
                                 for m in candidate.modules()))

    def test_absent_unavailable_and_zero_support_fall_back_after_learning(self):
        candidate = UnifiedIncidentIGSTGNN(self.model()).eval()
        self.activate(candidate)
        x, report = self.batch()
        empty = copy.deepcopy(report)
        empty['distances'].zero_()
        unavailable = dict(report, available=torch.zeros(2, dtype=torch.bool))
        with torch.no_grad():
            expected = candidate.core(x)
            for value in (None, empty, unavailable):
                actual = candidate(x, incident_data=value)
                self.assertTrue(torch.equal(actual, expected))
            mixed = dict(report, available=torch.tensor([True, False]))
            actual = candidate(x, incident_data=mixed)
            self.assertTrue(torch.equal(actual[1], expected[1]))
            self.assertFalse(torch.equal(actual[0], expected[0]))

    def test_label_independence_no_mutation_no_cross_batch_cache(self):
        candidate = UnifiedIncidentIGSTGNN(self.model()).eval()
        self.activate(candidate)
        x, report = self.batch()
        before = tensor_tree_hash((x, report))
        with torch.no_grad():
            a = candidate(x, torch.zeros(2, 12, 4, 1), incident_data=report)
            b = candidate(x, torch.ones(2, 12, 4, 1)*999, incident_data=report)
            reordered = candidate(x.flip(0), incident_data={k: v.flip(0) for k, v in report.items()})
            candidate(x+.01, incident_data=report)
            again = candidate(x, incident_data=report)
        self.assertEqual(before, tensor_tree_hash((x, report)))
        self.assertTrue(torch.equal(a, b))
        self.assertTrue(torch.equal(a, again))
        torch.testing.assert_close(a, reordered.flip(0), atol=1e-6, rtol=1e-6)

    def test_report_intervention_recovers_core_and_nested_restore(self):
        candidate = UnifiedIncidentIGSTGNN(self.model()).eval()
        self.activate(candidate)
        x, report = self.batch()
        with torch.no_grad():
            on = candidate(x, incident_data=report)
            with candidate.intervention(report=False):
                off = candidate(x, incident_data=report)
                with candidate.intervention():
                    self.assertTrue(torch.equal(candidate(x, incident_data=report), off))
            self.assertTrue(torch.equal(off, candidate.core(x)))
            self.assertTrue(torch.equal(on, candidate(x, incident_data=report)))
            self.assertFalse(torch.equal(on, off))

    def test_known_categories_preserved_unknown_not_mapped_to_zero(self):
        original = self.model()
        encoder = ReportEncoder(original.icsf_module)
        source = original.icsf_module.desc_embedding
        self.assertTrue(torch.equal(source.weight, encoder.desc_embedding.weight[:-1]))
        actual = encoder.lookup(encoder.desc_embedding, torch.tensor([-1, 999]))
        self.assertTrue(torch.equal(actual[0], encoder.desc_embedding.weight[-1]))
        self.assertTrue(torch.equal(actual[0], actual[1]))
        self.assertFalse(torch.equal(actual[0], encoder.desc_embedding.weight[0]))

    def test_first_step_heads_learn_then_state_transition_learns(self):
        x, report = self.batch()
        for mode in ('TIME', 'GRAPH', 'BOTH'):
            candidate = UnifiedIncidentIGSTGNN(self.model(), mode=mode).eval()
            optimizer = torch.optim.Adam(candidate.parameters(), lr=.002, weight_decay=0)
            candidate(x, incident_data=report).square().mean().backward()
            adapter = candidate.adapter
            if mode != 'GRAPH':
                self.assertGreater(float(adapter.output_state.weight.grad.abs().sum()), 0)
                self.assertGreater(float(adapter.output_node.weight.grad.abs().sum()), 0)
            else:
                self.assertIsNone(adapter.output_state.weight.grad)
            if mode != 'TIME':
                self.assertGreater(sum(float(p.weight.grad.abs().sum()) for p in adapter.graph.query_node), 0)
            else:
                self.assertIsNone(adapter.graph.query_node[0].weight.grad)
            optimizer.step()
            optimizer.zero_grad()
            candidate(x, incident_data=report).square().mean().backward()
            self.assertGreater(float(adapter.transition.weight_hh.grad.abs().sum()), 0)
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in candidate.parameters() if p.grad is not None))

    def test_graph_schedule_compact_bounded_and_historical_call_unchanged(self):
        candidate = UnifiedIncidentIGSTGNN(self.model()).eval()
        self.activate(candidate)
        x, report = self.batch()
        with torch.no_grad():
            h, u, d, tod, dow = candidate.core.prepare(x)
            _, _, bases = candidate.core.graphs(h, u, d, tod, dow)
            values = candidate.adapter(h, report, None, tod, dow)
            schedule = candidate.adapter.graph_schedule(candidate.core, values, bases)
        self.assertIsNone(schedule[0])
        self.assertEqual(len(schedule), 4)
        self.assertEqual(values[-1].shape, (2, 4, 4, 256))
        self.assertLessEqual(float(values[-1].abs().max()), .1)
        self.assertTrue(all(g.shape == (2, 4, 4) for block in schedule[1:] for g in block))
        captured = []
        def record(module, args):
            captured.append([g.detach().clone() for g in args[1]])
        handles = [layer.dif_layer.localized_st_conv.register_forward_pre_hook(record) for layer in candidate.core.layers]
        try:
            with torch.no_grad(), candidate.intervention(graph=False):
                candidate(x, incident_data=report)
            native = captured[:]
            captured.clear()
            with torch.no_grad():
                candidate(x, incident_data=report)
            self.assertEqual(len(native), 20)
            self.assertEqual(len(captured), 5)
            for i in range(5):
                self.assertTrue(all(torch.equal(a, b) for a, b in zip(native[4*i], captured[i])))
        finally:
            for handle in handles:
                handle.remove()

    def test_diagnostics_and_invalid_configuration(self):
        candidate = UnifiedIncidentIGSTGNN(self.model()).eval()
        x, report = self.batch()
        with torch.no_grad():
            candidate(x, incident_data=report)
        data = candidate.diagnostics()
        json.dumps(data, allow_nan=False)
        self.assertEqual(data['report_entry_count'], 1)
        self.assertEqual(data['forecast_blocks'], 4)
        self.assertFalse(data['native_ICSF_TIID_retained'])
        self.assertEqual(data['zero_adapter_reference'], 'report_free_traffic_core')
        self.assertEqual(data['delta_abs_max'], 0)
        with self.assertRaises(ValueError):
            candidate(x, incident_data=dict(report, available=torch.tensor([1, .5])))
        for config in (dict(mode='bad'), dict(max_graph_mix=2), dict(state_dim=0)):
            with self.assertRaises(ValueError):
                UnifiedIncidentIGSTGNN(self.model(), **config)


if __name__ == '__main__':
    unittest.main()
