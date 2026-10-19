"""Server-only numerical checks for the GW16 structural/dataflow contract."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.position_state_gw16.geometry import RegionGeometry, pool_contrast, lift_contrast
from experiments.position_state_gw16.model import build_model, PositionStateExchange, GraphGRUUpdate, parameter_count


def fixture(directory=None):
    n, d = 6, 32
    config = dict(nodes=n, history=12, horizon=12, hidden=d, update_hidden=3*d, spatial_layers=2, dropout=0.,
        exchange_rank=3, step_initial=.1, direct_head_width=None, free_transition_width=None,
        capacity_tolerance=.02, coupling='both', event_ablation='none')
    graph = np.eye(n, dtype=np.float32)*.5+np.roll(np.eye(n, dtype=np.float32), 1, axis=1)*.5
    data = SimpleNamespace(assignment=np.array([0, 0, 0, 1, 1, 1]),
        graphs=(graph, graph.T.copy(), np.array([[.75, .25], [.25, .75]], dtype=np.float32)),
        sensors=dict(sensor_type=np.zeros(n, dtype=np.int64), surface=np.zeros(n, dtype=np.int64),
            roadway_use=np.zeros(n, dtype=np.int64), road_width=np.ones(n)*3., speed_limit=np.ones(n)*90.),
        sensor_sizes=[1, 1, 1], mean=100., std=20., position_features=np.arange(n*6, dtype=np.float32).reshape(n, 6)/30.,
        position_categories=np.array([[1, 1], [1, 1], [1, 1], [2, 2], [2, 2], [2, 2]]),
        position_sizes=[3, 3], native_adjacency=graph, directory=Path(directory) if directory else Path('.'))
    history = torch.randn(2, 12, n, 3)
    history[..., 1] = torch.arange(12).view(1, 12, 1)/288
    history[..., 2] = 2/7
    event = dict(type=torch.tensor([0, 5]), description=torch.tensor([0, 28]),
        type_valid=torch.ones(2, dtype=torch.bool), description_valid=torch.ones(2, dtype=torch.bool),
        relation=torch.randn(2, n, 3), relation_available=torch.ones(2, dtype=torch.bool),
        report_present=torch.ones(2, dtype=torch.bool))
    return data, config, history, event


class PositionStateModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2025)
        self.data, self.config, self.history, self.event = fixture()

    def model(self, arm='STATE_EXCHANGE'):
        return build_model(self.data, self.config, arm, 'cpu').eval()

    def test_geometry_recovers_hidden_and_local_zero_mean(self):
        geometry = RegionGeometry(self.data.assignment)
        value = torch.randn(2, 12, 6, 8, dtype=torch.float64)
        coarse, local = geometry.pool(value), geometry.center(value)
        torch.testing.assert_close(geometry.lift(coarse)+local, value, atol=1e-14, rtol=1e-14)
        torch.testing.assert_close(geometry.pool(local), torch.zeros_like(coarse), atol=1e-14, rtol=0)

    def test_constant_weights_cannot_transfer_contrast(self):
        geometry = RegionGeometry(self.data.assignment)
        local = geometry.center(torch.randn(2, 6, 8))
        coarse = torch.randn(2, 2, 8)
        weights = torch.full((2, 6, 1), .7)
        self.assertLess(pool_contrast(geometry, local, weights).abs().max().item(), 1e-6)
        self.assertLess(lift_contrast(geometry, coarse, weights).abs().max().item(), 1e-6)

    def test_all_mechanism_outputs_and_additive_coordinates(self):
        for arm in ('DIRECT', 'FREE_RECURRENT', 'ORIGIN_EXCHANGE', 'STATE_EXCHANGE'):
            with self.subTest(arm=arm):
                model = self.model(arm)
                output = model(self.history, self.event)
                self.assertEqual(output['prediction'].shape, (2, 12, 6, 1))
                self.assertEqual(output['alpha'].shape, (2, 12, 2, 1))
                self.assertEqual(output['state_rms'].shape, (12, 2))
                for value in output.values():
                    self.assertTrue(torch.isfinite(value).all())
                expected = output['reference']+model.std*(model.geometry.lift(output['alpha'])+output['beta'])
                torch.testing.assert_close(output['prediction'], expected)
                torch.testing.assert_close(model.geometry.pool(output['beta']),
                    torch.zeros_like(output['alpha']), atol=1e-6, rtol=0)
                torch.testing.assert_close(model.geometry.pool(output['prediction']), output['regional'])

    def test_shared_modules_have_identical_initial_tensors(self):
        models = []
        for arm in ('DIRECT', 'FREE_RECURRENT', 'ORIGIN_EXCHANGE', 'STATE_EXCHANGE'):
            torch.manual_seed(333)
            models.append(self.model(arm))
        for name in ('position', 'encoder', 'incident', 'origin_condition'):
            expected = getattr(models[0], name).state_dict()
            for model in models[1:]:
                actual = getattr(model, name).state_dict()
                self.assertEqual(set(actual), set(expected))
                self.assertTrue(all(torch.equal(actual[key], expected[key]) for key in expected))

    def test_origin_and_state_identical_initialization_and_first_update(self):
        torch.manual_seed(41)
        origin = self.model('ORIGIN_EXCHANGE')
        torch.manual_seed(41)
        current = self.model('STATE_EXCHANGE')
        self.assertEqual(set(origin.state_dict()), set(current.state_dict()))
        self.assertTrue(all(torch.equal(value, current.state_dict()[key])
            for key, value in origin.state_dict().items()))
        old, new = origin(self.history, self.event), current(self.history, self.event)
        torch.testing.assert_close(old['prediction'][:, 0], new['prediction'][:, 0], atol=0, rtol=0)
        self.assertGreater((old['prediction'][:, 1:]-new['prediction'][:, 1:]).abs().max().item(), 1e-7)

    def test_reference_is_exact_persistence_and_readout_has_no_shortcut(self):
        model = self.model()
        with torch.no_grad():
            model.head.coarse_readout.weight.zero_()
            model.head.local_readout.weight.zero_()
        output = model(self.history, self.event)
        expected = (self.history[:, -1:, :, :1]*20+100).expand(2, 12, 6, 1)
        self.assertTrue(torch.equal(output['reference'], expected))
        self.assertTrue(torch.equal(output['prediction'], expected))
        self.assertIsNone(model.head.coarse_readout.bias)
        self.assertIsNone(model.head.local_readout.bias)

    def test_no_report_clears_event_but_traffic_continues(self):
        model = self.model()
        event = dict(self.event, report_present=torch.zeros(2, dtype=torch.bool))
        changed = dict(event, type=self.event['type'].flip(0), description=self.event['description'].flip(0),
                       relation=torch.randn_like(self.event['relation'])*10)
        one, two = model(self.history, event), model(self.history, changed)
        self.assertTrue(torch.equal(one['prediction'], two['prediction']))
        self.assertFalse(one['available'].any())
        self.assertFalse(one['support'].any())
        self.assertGreater((one['prediction']-one['reference']).abs().max().item(), 1e-6)

    def test_explicit_no_report_ablation_clears_all_incident_paths(self):
        self.config['event_ablation'] = 'no_report'
        model = self.model()
        one = model(self.history, self.event)
        absent = dict(self.event, report_present=torch.zeros(2, dtype=torch.bool))
        two = model(self.history, absent)
        self.assertTrue(torch.equal(one['prediction'], two['prediction']))
        self.assertFalse(one['available'].any())

    def test_missing_category_is_not_class_zero(self):
        model = self.model()
        missing = dict(self.event, type_valid=torch.zeros(2, dtype=torch.bool),
            description_valid=torch.zeros(2, dtype=torch.bool))
        zero = dict(self.event, type=torch.zeros(2, dtype=torch.long), description=torch.zeros(2, dtype=torch.long))
        first = model.incident(missing)[0]
        second = model.incident(zero)[0]
        self.assertFalse(torch.equal(first, second))
        self.assertEqual(model.incident.type_embedding.num_embeddings, 7)
        self.assertEqual(model.incident.description_embedding.num_embeddings, 30)

    def test_exchange_weights_center_and_respond_to_both_inputs(self):
        geometry = RegionGeometry(self.data.assignment)
        exchange = PositionStateExchange(8, 3)
        position, condition = torch.randn(6, 8), torch.randn(2, 6, 8)
        coarse, local = torch.randn(2, 6, 8), torch.randn(2, 6, 8)
        base = exchange(position, condition, coarse, local, geometry)
        torch.testing.assert_close(geometry.pool(base), torch.zeros(2, 2, 2), atol=1e-6, rtol=0)
        altered_position = exchange(position.flip(0), condition, coarse, local, geometry)
        altered_state = exchange(position, condition, coarse+torch.randn_like(coarse), local, geometry)
        self.assertGreater((base-altered_position).abs().max().item(), 1e-6)
        self.assertGreater((base-altered_state).abs().max().item(), 1e-6)

    def test_shared_encoder_and_exchange_receive_gradients(self):
        model = self.model()
        loss = (model(self.history, self.event)['prediction']-123.).square().mean()
        loss.backward()
        for module in (model.encoder, model.position, model.incident, model.head.exchange):
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters()))

    def test_active_head_counts_match_and_config_is_unchanged(self):
        before = dict(self.config)
        for arm in ('DIRECT', 'FREE_RECURRENT'):
            model = self.model(arm)
            self.assertEqual(parameter_count(model.head), model.capacity['actual_head_parameters'])
            self.assertLessEqual(model.capacity['match_fraction'], .02)
        self.assertEqual(self.config, before)

    def test_free_gate_equations_match_torch_grucell(self):
        d = 8
        update = GraphGRUUpdate(d, 5, 0.).eval()
        cell = torch.nn.GRUCell(1, d)
        input_bias = torch.randn(3*d)
        with torch.no_grad():
            update.gates[0].weight.zero_()
            update.gates[0].bias.zero_()
            update.gates[-1].weight.zero_()
            update.gates[-1].bias.copy_(input_bias)
            cell.weight_ih.zero_()
            cell.bias_ih.copy_(input_bias)
            cell.weight_hh.copy_(update.hidden_gates.weight)
            cell.bias_hh.copy_(update.hidden_gates.bias)
        state, other, condition = (torch.randn(2, 6, d) for _ in range(3))
        graph = torch.eye(6)
        actual = update(state, other, condition, graph, graph)
        expected = cell(torch.zeros(12, 1), state.reshape(12, d)).reshape(2, 6, d)
        torch.testing.assert_close(actual, expected)

    def test_only_joint_stage_is_supported(self):
        model = self.model()
        model.set_stage('joint')
        with self.assertRaises(ValueError):
            model.set_stage('reference')

    def test_unknown_arm_and_wrong_history_rejected(self):
        with self.assertRaises(ValueError):
            self.model('UNKNOWN')
        with self.assertRaises(ValueError):
            self.model()(self.history[:, :-1], self.event)

    def test_native_unknown_rows_and_absent_report_boundary(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'type_mapping.json').write_text(json.dumps({str(i):i for i in range(6)}))
            (root/'desc_mapping.json').write_text(json.dumps({str(i):i for i in range(29)}))
            data, config, history, event = fixture(root)
            model = build_model(data, config, 'NATIVE_POS', 'cpu').eval()
            self.assertEqual(model.native.icsf_module.incident_type_embedding.num_embeddings, 7)
            self.assertEqual(model.native.icsf_module.desc_embedding.num_embeddings, 30)
            absent = dict(event, report_present=torch.zeros(2, dtype=torch.bool))
            changed = dict(absent, relation=torch.randn_like(event['relation'])*17,
                type=event['type'].flip(0), description=event['description'].flip(0))
            with torch.no_grad(), patch.object(model.native, '_graph_constructor',
                    wraps=model.native._graph_constructor) as graph_call:
                one = model(history, absent)
                entered = graph_call.call_args.kwargs['history_data']
                expected_entered = model.native.embedding(history[..., :1])+model.position()[None, None]
                self.assertTrue(torch.equal(entered, expected_entered))
                two = model(history, changed)
            self.assertTrue(torch.equal(one['prediction'], two['prediction']))
            self.assertFalse(one['available'].any())
            self.assertEqual(one['prediction'].shape, (2, 12, 6, 1))
            self.assertTrue(torch.isfinite(one['prediction']).all())
            payload, _, _ = model._event(dict(event, type_valid=torch.zeros(2, dtype=torch.bool),
                description_valid=torch.zeros(2, dtype=torch.bool), Holiday=torch.ones(2), incident_position=torch.ones(2)))
            self.assertTrue((payload['incident'][:, 1] == 29).all())
            self.assertTrue((payload['incident'][:, 2] == 6).all())
            self.assertTrue((payload['incident'][:, 3] == 0).all())
            self.assertTrue((payload['position'] == 0).all())


if __name__ == '__main__':
    unittest.main()
