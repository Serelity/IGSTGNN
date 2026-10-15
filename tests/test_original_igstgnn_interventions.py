"""CPU contracts for fixed-weight interventions of the released model."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from experiments.chronological.original_igstgnn_interventions import (
    CASE_NAMES, intervention, run_case, run_case_with_audit,
)
from experiments.chronological.original_igstgnn_source import get_original_modules


def original_module():
    return get_original_modules().model


def clone_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: clone_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(clone_tree(item) for item in value)
    return value


@torch.inference_mode()
def manual_path(model, x, incident, history_kind, use_tiid):
    history, node_u, node_d, tod, dow = model._prepare_inputs(x)
    embedded = model.embedding(history)
    enhanced, context = model.icsf_module(embedded, incident, None, tod[:, -1, 0], dow[:, -1, 0])
    normalized = embedded.clone()
    normalized[:, -1] = model.icsf_module.output_norm(embedded[:, -1])
    history = {'enhanced': enhanced, 'normalized': normalized, 'raw': embedded}[history_kind]
    static, dynamic = model._graph_constructor(node_embedding_u=node_u, node_embedding_d=node_d,
                                              history_data=history, time_in_day_feat=tod,
                                              day_in_week_feat=dow)
    dif, inh = [], []
    for layer in model.layers:
        history, part_dif, part_inh = layer(history, dynamic, static, node_u, node_d, tod, dow)
        dif.append(part_dif)
        inh.append(part_inh)
    hidden = sum(dif)+sum(inh)
    if use_tiid:
        hidden = model.tiid_module(hidden, context)
    forecast = model.out_fc_2(torch.relu(model.out_fc_1(torch.relu(hidden))))
    forecast = forecast.transpose(1, 2).contiguous().view(forecast.shape[0], forecast.shape[2], -1)
    return forecast.transpose(1, 2).unsqueeze(-1)


class OriginalInterventionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.original = original_module()
        cls.data = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        for name, size in (('type_mapping.json', 6), ('desc_mapping.json', 29)):
            (Path(cls.data.name)/name).write_text(json.dumps({str(i): i for i in range(size)}))

    @classmethod
    def tearDownClass(cls):
        cls.data.cleanup()

    def setUp(self):
        torch.manual_seed(51)
        adjacency = torch.tensor([[0., 1., 0., 0.], [0., 0., 1., 0.],
                                  [0., 0., 0., 1.], [1., 0., 0., 0.]])
        args = dict(num_feat=1, num_hidden=8, node_hidden=4, time_emb_dim=4,
                    tpd=288, k_s=2, k_t=3, layer=5, gap=3, dropout=.1,
                    dataset='fixture', data_path=self.data.name, use_sensor_info=False,
                    adjs=[adjacency, adjacency.T])
        self.model = self.original.IGSTGNN(args, node_num=4, input_dim=3,
                                          output_dim=1, seq_len=12, horizon=12).eval()
        self.x = torch.randn(2, 12, 4, 3)
        self.x[..., 1] = .4
        self.x[..., 2] = 2/7
        self.incident = {
            'incident': torch.tensor([[0., 2., 1., 0.], [0., 4., 3., 1.]]),
            'position': torch.tensor([2, 8]),
            'distances': torch.tensor([[[.2, .8, 1.], [.5, .1, 0.], [0., 0., 0.], [0., 0., 0.]],
                                       [[0., 0., 0.], [0., 0., 0.], [.8, .2, 1.], [.1, .7, 0.]]]),
        }

    def assertTreeEqual(self, left, right):
        if isinstance(left, torch.Tensor):
            self.assertTrue(torch.equal(left, right))
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for name in left:
                self.assertTreeEqual(left[name], right[name])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assertTreeEqual(a, b)
        else:
            self.assertEqual(left, right)

    def hook_inventory(self):
        return {name: (tuple(module._forward_hooks), tuple(module._forward_pre_hooks))
                for name, module in self.model.named_modules()}

    def test_full_and_raw_match_native_four_block_model(self):
        with torch.inference_mode():
            native = self.model(self.x, incident_data=self.incident)
            raw = self.model(self.x, incident_data=None)
        self.assertTrue(torch.equal(run_case(self.model, self.x, self.incident, None, 'FULL'), native))
        self.assertTrue(torch.equal(run_case(self.model, self.x, self.incident, None, 'RAW_ALL_REPORT_OFF'), raw))
        self.assertTrue(torch.equal(run_case(self.model, self.x, None, None, 'RAW_ALL_REPORT_OFF'), raw))

    def test_all_cases_shapes_coverage_state_inputs_rng_and_hook_restoration(self):
        before = clone_tree(self.model.state_dict())
        inputs = clone_tree((self.x, self.incident))
        graph_buffers = [clone_tree(layer.dif_layer.localized_st_conv.pre_defined_graph)
                         for layer in self.model.layers]
        rng = torch.get_rng_state().clone()
        hooks = self.hook_inventory()
        for case in CASE_NAMES:
            prediction, audit = run_case_with_audit(self.model, self.x, self.incident, None, case)
            self.assertEqual(prediction.shape, (2, 12, 4, 1), case)
            self.assertTrue(bool(torch.isfinite(prediction).all()), case)
            self.assertTrue(audit['execution_coverage_passed'], case)
            self.assertEqual(audit['counts']['localized_conv'], 20)
            self.assertEqual(audit['counts']['gru_cell'], 45)
            self.assertEqual(audit['counts']['attention'], 20)
            json.dumps(audit, allow_nan=False)
            self.assertEqual(self.hook_inventory(), hooks, case)
            self.assertTreeEqual(self.model.state_dict(), before)
            self.assertTreeEqual((self.x, self.incident), inputs)
            self.assertTreeEqual([layer.dif_layer.localized_st_conv.pre_defined_graph
                                  for layer in self.model.layers], graph_buffers)
            self.assertTrue(torch.equal(torch.get_rng_state(), rng))
            self.assertTrue(all(p.grad is None for p in self.model.parameters()))

    def test_incident_norm_factorial_matches_independent_original_decode(self):
        cases = {'FULL': ('enhanced', True), 'ICSF_ONLY': ('enhanced', False),
                 'NORM_ONLY': ('normalized', False), 'TIID_ONLY_ON_NORM': ('normalized', True),
                 'RAW_ALL_REPORT_OFF': ('raw', False)}
        for case, (history, tiid) in cases.items():
            expected = manual_path(self.model, self.x, self.incident, history, tiid)
            actual = run_case(self.model, self.x, self.incident, None, case)
            self.assertTrue(torch.equal(actual, expected), case)

    def test_zero_support_retains_layernorm_and_tiid_is_zero(self):
        incident = dict(self.incident, distances=torch.zeros_like(self.incident['distances']))
        norm = run_case(self.model, self.x, incident, None, 'NORM_ONLY')
        for case in ('FULL', 'ICSF_ONLY', 'TIID_ONLY_ON_NORM'):
            self.assertTrue(torch.equal(run_case(self.model, self.x, incident, None, case), norm), case)
        raw = run_case(self.model, self.x, incident, None, 'RAW_ALL_REPORT_OFF')
        self.assertFalse(torch.equal(raw, norm))

    def test_tiid_effect_remains_zero_outside_native_support(self):
        full = run_case(self.model, self.x, self.incident, None, 'FULL')
        off = run_case(self.model, self.x, self.incident, None, 'ICSF_ONLY')
        outside = ~(self.incident['distances'].abs().sum(-1) > 0)
        outside = outside[:, None, :, None].expand_as(full)
        self.assertTrue(torch.equal(full[outside], off[outside]))

    def collect_residuals_and_tiid_input(self, case):
        records = {'layer_history': [], 'inh_inputs': [], 'dif_forecasts': [], 'inh_forecasts': []}
        handles = []
        for layer in self.model.layers:
            handles.append(layer.register_forward_pre_hook(lambda _m, args: records['layer_history'].append(args[0].clone())))
            handles.append(layer.inh_layer.register_forward_pre_hook(lambda _m, args: records['inh_inputs'].append(args[0].clone())))
            handles.append(layer.dif_layer.register_forward_hook(lambda _m, _a, out: records['dif_forecasts'].append(out[1].clone())))
            handles.append(layer.inh_layer.register_forward_hook(lambda _m, _a, out: records['inh_forecasts'].append(out[1].clone())))
        handles.append(self.model.tiid_module.register_forward_pre_hook(lambda _m, args: records.update(tiid_input=args[0].clone())))
        try:
            run_case(self.model, self.x, self.incident, None, case)
        finally:
            for handle in handles:
                handle.remove()
        return records

    def test_forecast_off_preserves_all_residual_histories_and_masks_only_readout(self):
        baseline = self.collect_residuals_and_tiid_input('FULL')
        for case, remaining in (('DIF_FORECAST_READOUT_OFF', 'inh_forecasts'),
                                ('INH_FORECAST_READOUT_OFF', 'dif_forecasts')):
            candidate = self.collect_residuals_and_tiid_input(case)
            self.assertTreeEqual(candidate['layer_history'], baseline['layer_history'])
            self.assertTreeEqual(candidate['inh_inputs'], baseline['inh_inputs'])
            self.assertTrue(torch.equal(candidate['tiid_input'], sum(baseline[remaining])))

    def test_graph_family_masks_preserve_width_and_cover_generated_calls(self):
        expected_slices = {'PRE_BANKS_ZERO': (8, 40), 'DYN_BANKS_ZERO': (40, 72),
                           'STA_BANKS_ZERO': (72, 88)}
        for case, (first, last) in expected_slices.items():
            observed = []
            handles = [layer.dif_layer.localized_st_conv.gcn_updt.register_forward_hook(
                lambda _m, args, _out: observed.append(args[0].clone())) for layer in self.model.layers]
            try:
                _, audit = run_case_with_audit(self.model, self.x, self.incident, None, case)
            finally:
                for handle in handles:
                    handle.remove()
            self.assertEqual(len(observed), 20)
            self.assertEqual(audit['counts']['graph_bank_inputs_masked'], 20)
            for features in observed:
                self.assertEqual(features.shape[-1], 88)
                self.assertTrue(torch.equal(features[..., first:last], torch.zeros_like(features[..., first:last])))
            self.assertTrue(any(bool(features[..., :8].abs().sum() > 0) for features in observed))

    def test_gru_identity_reaches_history_and_forecast_cells(self):
        observed = []
        handles = []
        try:
            with intervention(self.model, 'GRU_TRANSITION_IDENTITY') as audit:
                for layer in self.model.layers:
                    handles.append(layer.inh_layer.rnn_layer.gru_cell.register_forward_hook(
                        lambda _m, args, out: observed.append(torch.equal(args[0], out))))
                with torch.inference_mode():
                    self.model(self.x, incident_data=self.incident)
            self.assertEqual(len(observed), 45)
            self.assertTrue(all(observed))
            self.assertEqual(audit['counts']['gru_outputs_replaced_by_input'], 45)
        finally:
            for handle in handles:
                handle.remove()

    def test_attention_identity_reaches_history_and_forecast_queries(self):
        observed = []
        handles = []
        try:
            with intervention(self.model, 'ATTENTION_QUERY_IDENTITY') as audit:
                for layer in self.model.layers:
                    handles.append(layer.inh_layer.transformer_layer.multi_head_self_attention.register_forward_hook(
                        lambda _m, args, out: observed.append(torch.equal(args[0], out[0]) and out[1] is None)))
                with torch.inference_mode():
                    self.model(self.x, incident_data=self.incident)
            self.assertEqual(len(observed), 20)
            self.assertTrue(all(observed))
            self.assertEqual(audit['counts']['attention_outputs_replaced_by_query'], 20)
        finally:
            for handle in handles:
                handle.remove()

    def test_native_tiid_context_is_preserved_when_icsf_history_is_replaced(self):
        observed = []
        handle = self.model.tiid_module.register_forward_pre_hook(
            lambda _m, args: observed.append(args[1].clone()))
        try:
            run_case(self.model, self.x, self.incident, None, 'FULL')
            run_case(self.model, self.x, self.incident, None, 'TIID_ONLY_ON_NORM')
        finally:
            handle.remove()
        self.assertTrue(torch.equal(observed[0], observed[1]))

    def test_hooks_restore_after_forward_failure_for_every_case(self):
        before = self.hook_inventory()
        for case in CASE_NAMES:
            with patch.object(self.model.out_fc_2, 'forward', side_effect=RuntimeError('synthetic head failure')):
                with self.assertRaisesRegex(RuntimeError, 'synthetic head failure'):
                    run_case(self.model, self.x, self.incident, None, case)
            self.assertEqual(self.hook_inventory(), before, case)

    def test_context_manager_enforces_inference_and_restores_mode(self):
        previous = torch.is_inference_mode_enabled()
        with intervention(self.model, 'FULL'):
            self.assertTrue(torch.is_inference_mode_enabled())
            prediction = self.model(self.x, incident_data=self.incident)
            self.assertTrue(prediction.is_inference())
        self.assertEqual(torch.is_inference_mode_enabled(), previous)

    def test_hooks_restore_after_setup_failure_and_preserve_existing_hooks(self):
        observed = []
        handle = self.model.embedding.register_forward_hook(lambda _m, _a, _o: observed.append(1))
        before = self.hook_inventory()
        conv = self.model.layers[-1].dif_layer.localized_st_conv
        previous = conv.gcn_updt.in_features
        try:
            conv.gcn_updt.in_features += 1
            with self.assertRaisesRegex(ValueError, 'graph-bank axis'):
                run_case(self.model, self.x, self.incident, None, 'PRE_BANKS_ZERO')
            self.assertEqual(self.hook_inventory(), before)
        finally:
            conv.gcn_updt.in_features = previous
        run_case(self.model, self.x, self.incident, None, 'FULL')
        self.assertEqual(observed, [1])
        self.assertEqual(self.hook_inventory(), before)
        handle.remove()

    def test_invalid_mode_gradients_case_report_and_modified_model_are_rejected(self):
        self.model.layers[0].train()
        with self.assertRaisesRegex(ValueError, 'eval mode'):
            run_case(self.model, self.x, self.incident, None, 'FULL')
        self.model.eval()
        parameter = next(self.model.parameters())
        parameter.grad = torch.ones_like(parameter)
        with self.assertRaisesRegex(ValueError, 'absent parameter gradients'):
            run_case(self.model, self.x, self.incident, None, 'FULL')
        parameter.grad = None
        with self.assertRaisesRegex(ValueError, 'Unknown'):
            run_case(self.model, self.x, self.incident, None, 'not_a_case')
        with self.assertRaisesRegex(ValueError, 'native report'):
            run_case(self.model, self.x, None, None, 'FULL')
        self.model._decode_forecast = lambda x: x
        with self.assertRaisesRegex(ValueError, 'isolated released model'):
            run_case(self.model, self.x, self.incident, None, 'FULL')


if __name__ == '__main__':
    unittest.main()
