"""M4.2 local independence, prefix information boundary and paired updates."""
import copy
from datetime import datetime
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.chronological import train_incident_capacity as training
from src.models.incident_capacity_exchange import LocalCapacityRecurrence, edge_capacities
from src.utils.capacity_training_inputs import CapacityDevelopmentInputs, CutoffReports, prefix_inputs
from src.utils.incident_corridor import read_json
from test_acdg import make_model, incident_batch
from test_incident_capacity_fusion import fixture

torch.set_num_threads(3)


class CapacityTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.protocol = read_json(training.REPO/'experiments/chronological/incident_capacity_training_m42.json')

    def test_local_future_nodes_are_independent_and_inflow_is_not_deleted(self):
        branch, inputs = fixture()
        graph = branch.graph
        rates = torch.ones(3, 4)
        local = LocalCapacityRecurrence(graph, rates, rates)
        times = torch.arange(9)*1.25
        coefficients = torch.full((1, 4, 4), .5)
        capacity = edge_capacities(coefficients, times[:-1], 65., graph, rates, branch.outgoing_weights)
        boundary = torch.zeros_like(capacity)
        boundary[:, :, 0] = .4
        demand = torch.zeros_like(capacity)
        demand[:, :, 1:3] = .3
        initial = torch.full((1, 3, 4), .4, requires_grad=True)
        result = local(initial, capacity, boundary, times, branch.outgoing_weights, local_demand=demand)
        gradient = torch.autograd.grad(result['states'][0, -1, 1].sum(), initial)[0]
        self.assertEqual(float(gradient[:, [0, 2]].abs().sum()), 0)
        self.assertGreater(float(gradient[:, 1].abs().sum()), 0)
        self.assertGreater(float(result['inflow'][:, :, 1].detach().sum()), 0)
        self.assertNotIn('balance_residual', result)
        changed = initial.detach().clone()
        changed[:, 0] = .95
        replay = local(changed, capacity, boundary, times, branch.outgoing_weights, local_demand=demand)
        torch.testing.assert_close(replay['states'][:, :, 1:], result['states'][:, :, 1:], atol=0, rtol=0)

    def test_local_bounds_on_random_merge_fork_and_open_graph(self):
        from src.models.incident_capacity_exchange import ExchangeGraph
        graph = ExchangeGraph(4, torch.tensor([[-1, 0, 1, 1, 2, 3], [0, 1, 2, 3, 3, -1]]),
                              torch.ones(4, dtype=torch.bool), torch.tensor([True, False, False, True]),
                              evidence_scope='synthetic_only')
        rates, weights = torch.ones(4, 4)*2, torch.tensor([0., 1., .3, .7, 1., 1.])
        times = torch.arange(53)*1.25
        c = edge_capacities(torch.rand(2, 6, 4)*.95, times[:-1], 65., graph, rates, weights)
        boundary, demand = torch.zeros_like(c), torch.zeros_like(c)
        boundary[:, :, 0] = 2
        demand[:, :, 1:5] = torch.rand_like(demand[:, :, 1:5])*2
        result = LocalCapacityRecurrence(graph, rates, rates)(torch.rand(2, 4, 4), c, boundary,
                            times, weights, local_demand=demand)
        self.assertGreaterEqual(float(result['states'].min()), -1e-6)
        self.assertLessEqual(float(result['states'].max()), 1+1e-6)
        with self.assertRaises(ValueError):
            LocalCapacityRecurrence(graph, rates, rates)(torch.rand(2, 4, 4), c, boundary,
                times, weights, local_demand=torch.ones_like(demand))

    def test_local_branch_reuses_common_parameters_and_report_only_capacity_entry(self):
        base, inputs = fixture()
        local = training.IncidentCapacityBranch(copy.deepcopy(base.graph), base.outgoing_weights, mode='local')
        local.load_state_dict(base.state_dict(), strict=True)
        self.assertEqual(sum(p.numel() for p in base.parameters()), sum(p.numel() for p in local.parameters()))
        on, off = local(**inputs), local(**inputs, incident_enabled=False)
        for key in ('initial', 'boundary_demand', 'boundary_coefficients', 'hidden', 'history_coefficients'):
            torch.testing.assert_close(on[key], off[key], atol=0, rtol=0)
        self.assertGreater(float((on['coefficients']-off['coefficients']).abs().max().detach()), 0)

    def test_prefix_padding_and_held_out_values_never_enter_encoder(self):
        branch, inputs = fixture()
        inputs.pop('reports')
        prefix = prefix_inputs(inputs)
        changed = copy.deepcopy(inputs)
        changed['history'][:, 8:] = 1e10
        changed['valid'][:, 8:] = False
        other = prefix_inputs(changed)
        for key in prefix:
            torch.testing.assert_close(prefix[key], other[key], atol=0, rtol=0)
        self.assertFalse(bool(prefix['valid'][:, :4].any()))
        torch.testing.assert_close(prefix['history'][:, 4:], inputs['history'][:, :8])
        prepared = branch.prepare(**prefix)
        short = branch.rollout_prepared(prepared, torch.tensor([[5., 10.], [10., 15.]]))
        full = branch(**prefix)
        torch.testing.assert_close(short['features'], full['features'][:, :2], atol=0, rtol=0)
        self.assertEqual(float(short['rollout']['times'][-1]), 15)

    def report_fixture(self):
        rows = [dict(incident_id=str(i), report_time='2023-01-01T'+stamp,
                     edge_index=1, confidence=.7, distance_km=.2) for i, stamp in enumerate(
                         ['06:39:00', '06:40:00', '07:35:00', '07:40:00', '07:50:00', '08:00:00', '08:01:00'])]
        with patch('src.utils.capacity_training_inputs.associate_reports', side_effect=lambda r, *_: copy.deepcopy(r)):
            return CutoffReports(rows, {'edges': [1, 2, 3, 4]}, [])

    def test_prefix_report_collection_recomputed_at_earlier_cutoff(self):
        reports = self.report_fixture()
        earlier = reports.batch([datetime(2023, 1, 1, 7, 40)])
        self.assertEqual(earlier['ages'].tolist(), [[60., 5., 0.]])
        self.assertEqual(earlier['weights'].shape, (1, 3, 4))
        current = reports.batch([datetime(2023, 1, 1, 8)])
        self.assertEqual(current['ages'].tolist(), [[25., 20., 10., 0.]])
        off = reports.batch([datetime(2023, 1, 1, 8)], enabled=False)
        self.assertEqual(off['weights'].shape, (1, 0, 4))

    def test_input_adapter_never_reads_target_file_and_prefix_removes_native_trigger(self):
        inputs = CapacityDevelopmentInputs.__new__(CapacityDevelopmentInputs)
        values = np.ones((2, 12, 3, 3), np.float32)
        rows = [dict(t0='2023-01-01T08:00:00', x_start='2023-01-01T06:55:00')]*2
        adapter = SimpleNamespace(values=values, events=rows, trigger={k: v.numpy() for k, v in incident_batch().items()},
            scaler={'mean': 0., 'std': 1., 'node_fill_mean': [1., 1., 1.]}, references=np.ones((3, 3), np.float32))
        inputs.adapters, inputs.events, inputs.reports = {'train': adapter}, {'train': rows}, self.report_fixture()
        class Forbidden:
            def __getitem__(self, _):
                raise AssertionError('Target file entered input construction')
        inputs.datasets = {'train': SimpleNamespace(flow=Forbidden())}
        normal = inputs.batch('train', [0, 1])
        self.assertIn('incident', normal)
        prefix = inputs.batch('train', [0, 1], prefix=True)
        self.assertEqual(set(prefix), {'capacity_inputs'})
        self.assertEqual(prefix['capacity_inputs']['reports']['ages'][0].tolist(), [60., 5., 0.])
        with self.assertRaises(ValueError):
            inputs.batch('val', [0], prefix=True)

    def test_auxiliary_mask_and_loss_scale_have_independent_expected_value(self):
        prediction = torch.tensor([[[[1.], [99.], [3.]]]])
        target = torch.tensor([[[[14.], [20.], [18.]]]])
        valid = torch.ones_like(target, dtype=torch.bool)
        value = training.masked_auxiliary(prediction, target, valid, torch.tensor([True, False, True]),
                                           {'mean': 10., 'std': 2.})
        self.assertEqual(float(value), 1.)

    def test_six_arms_equal_initial_prediction_then_train_with_finite_gradients(self):
        backbone = make_model().eval()
        base, inputs = fixture()
        with torch.no_grad():
            base.coefficients[-1].bias.fill_(.8)
        state = base.state_dict()
        batch = dict(x=torch.rand(2, 12, 3, 3), incident=incident_batch(), capacity_inputs=inputs)
        target, valid = torch.ones(2, 12, 3, 1)*2, torch.ones(2, 12, 3, 1, dtype=torch.bool)
        prefix_target, prefix_valid = target[:, :2], valid[:, :2]
        expected = training.forward_arm(backbone, batch, 'F', {'mean': 0., 'std': 1.})
        for arm in training.ARMS:
            with self.subTest(arm=arm):
                model = training.build_arm(backbone, state, base.graph, base.outgoing_weights, arm, 11).eval()
                actual = training.forward_arm(model, batch, arm, {'mean': 0., 'std': 1.})
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                optimizer = torch.optim.Adam(model.parameters(), lr=.002)
                _, terms, grads, _ = training.train_update(model, optimizer, batch, target, valid,
                    {'capacity_inputs': inputs}, prefix_target, prefix_valid, arm, {'mean': 0., 'std': 1.}, self.protocol)
                self.assertTrue(all(np.isfinite(v) for v in terms.values()))
                if arm != 'F':
                    self.assertGreater(grads['branch.projection.weight'], 0)
                    self.assertGreater(grads['branch.coefficients.2.weight'], 0)

    def fake_inputs(self, batch):
        def get_batch(split, indices, device, **kwargs):
            b = copy.deepcopy(batch)
            if not kwargs.get('new_reports'):
                b['capacity_inputs']['reports'] = {k: v[:, :0] for k, v in b['capacity_inputs']['reports'].items()}
            return {'capacity_inputs': b['capacity_inputs']} if kwargs.get('prefix') else b
        def targets(split, indices, device, **kwargs):
            h = 2 if kwargs.get('prefix') else 12
            return torch.ones(2, h, 3, 1)*2, torch.ones(2, h, 3, 1, dtype=torch.bool)
        return SimpleNamespace(structure_mask=np.ones(3, bool), common_mask=np.array([True, True, False]),
            road_groups={'road-W': np.ones(3, bool)}, network=SimpleNamespace(structure={'boundary_nodes': [True, False, True]}),
            events={s: [dict(sample_index=i) for i in range(4)] for s in ('train', 'val')},
            original=SimpleNamespace(scaler={'mean': 0., 'std': 1.}), batch=get_batch, targets=targets)

    def test_saved_resume_matches_uninterrupted_next_epoch_and_rejects_identity_change(self):
        base, cap_inputs = fixture()
        backbone = make_model()
        model = training.build_arm(backbone, base.state_dict(), base.graph, base.outgoing_weights, 'P1', 11)
        batch = dict(x=torch.rand(2, 12, 3, 3), incident=incident_batch(), capacity_inputs=cap_inputs)
        inputs = self.fake_inputs(batch)
        scratch = training.REPO/'experiments/chronological_runs'
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as tmp:
            a, b = Path(tmp)/'full', Path(tmp)/'resumed'
            a.mkdir()
            b.mkdir()
            first, second = copy.deepcopy(model), copy.deepcopy(model)
            args = (inputs, 'P1')
            training.train_arm(first, *args, a, {'seed': 11}, self.protocol, 11, 2, 2, 'cpu', True, False)
            training.train_arm(second, *args, b, {'seed': 11}, self.protocol, 11, 1, 2, 'cpu', True, False)
            training.train_arm(second, *args, b, {'seed': 11}, self.protocol, 11, 2, 2, 'cpu', True, True)
            for k, value in first.state_dict().items():
                torch.testing.assert_close(value, second.state_dict()[k], atol=0, rtol=0)
            with self.assertRaises(ValueError):
                training.train_arm(second, *args, b, {'seed': 22}, self.protocol, 11, 3, 2, 'cpu', True, True)


if __name__ == '__main__':
    unittest.main()
