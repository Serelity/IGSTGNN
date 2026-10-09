"""Physical invariants, units, intervention locality and native integration."""

import copy
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.chronological.audit_incident_physics import history_profiles
from experiments.chronological.check_incident_physics import augmented, synthetic_fixture
from src.models.incident_queue import IncidentQueueBranch, QueueAugmentedIGSTGNN
from src.models.traffic_physics import point_queue_step, rollout_point_queue, ctm_corridor_step
from src.utils.traffic_physics_contract import (
    contract_template, validate_contract, build_physical_inputs, queue_config, check_manifest_timing,
)

torch.set_num_threads(3)


def example_contract():
    c = contract_template([10, 20, 30], {'summary.json': 'fixture', 'context_manifest.json': 'fixture'})
    c.update(status='declared_for_conditional_screening', flow_unit='vehicles_per_5min',
             lane_basis='all_lanes_total', interval_label='end',
             evidence={key: 'SYNTHETIC TEST EVIDENCE ONLY' for key in c['evidence']},
             bottlenecks=[{'id': 'synthetic', 'arrival_station_id': 10, 'departure_station_id': 20,
                           'readout_station_id': 20, 'capacity_veh_per_hour': 600., 'queue_scale_vehicles': 120.,
                           'boundary_evidence': 'SYNTHETIC TEST BOUNDARIES ONLY'}])
    return c


class PointQueueTests(unittest.TestCase):
    def test_clearance_and_conservation_match_analytic_solution(self):
        initial = torch.tensor([[120.]], dtype=torch.double)
        arrival = torch.full((1, 60, 1), 2200., dtype=torch.double)
        capacity = torch.full_like(arrival, 2400.)
        states, departure = rollout_point_queue(initial, arrival, capacity, 1 / 60)
        self.assertGreater(states[0, 35, 0], 0)
        self.assertLess(float(states[0, 36, 0]), 1e-10)
        torch.testing.assert_close(states[:, 1:] - states[:, :-1], (arrival - departure) / 60, atol=1e-12, rtol=0)
        self.assertTrue((states >= 0).all())
        self.assertTrue((departure <= capacity).all())

    def test_low_demand_capacity_changes_have_no_effect_and_zero_is_valid(self):
        q = torch.zeros(2, dtype=torch.double)
        arrival = torch.tensor([0., 100.], dtype=torch.double)
        a = point_queue_step(q, arrival, torch.tensor([500., 500.], dtype=torch.double), .1)
        b = point_queue_step(q, arrival, torch.tensor([200., 200.], dtype=torch.double), .1)
        for left, right in zip(a, b):
            torch.testing.assert_close(left, right)
        torch.testing.assert_close(a[1], arrival)

    def test_queue_can_clear_before_capacity_fully_recovers(self):
        rates = torch.full((1, 3, 1), 100., dtype=torch.double)
        capacities = torch.tensor([[[200.], [300.], [600.]]], dtype=torch.double)
        states, _ = rollout_point_queue(torch.tensor([[1.]], dtype=torch.double), rates, capacities, 1 / 60)
        self.assertEqual(float(states[0, 1, 0]), 0.)
        self.assertLess(float(capacities[0, 0, 0]), float(capacities[0, -1, 0]))

    def test_queue_gradients_and_bad_rates(self):
        values = [torch.tensor([30., 50.], dtype=torch.double, requires_grad=True),
                  torch.tensor([100., 120.], dtype=torch.double, requires_grad=True),
                  torch.tensor([150., 180.], dtype=torch.double, requires_grad=True)]
        self.assertTrue(torch.autograd.gradcheck(lambda *x: point_queue_step(*x, .1), tuple(values)))
        for bad in (-1., float('nan')):
            with self.assertRaises(ValueError):
                point_queue_step(torch.tensor([0.]), torch.tensor([bad]), torch.tensor([100.]), .1)
        with self.assertRaises(ValueError):
            point_queue_step(*[x.detach() for x in values], 0)


class CTMTests(unittest.TestCase):
    def inputs(self):
        return [torch.tensor([[20., 70., 100.]], dtype=torch.double),
                torch.full((1, 3), 1800., dtype=torch.double),
                torch.tensor([1., 1.5, 2.], dtype=torch.double),
                torch.full((3,), 60., dtype=torch.double), torch.full((3,), 20., dtype=torch.double),
                torch.full((3,), 150., dtype=torch.double),
                torch.tensor([900.], dtype=torch.double), torch.tensor([600.], dtype=torch.double)]

    def test_shared_fluxes_preserve_total_vehicle_count_and_bounds(self):
        values = self.inputs()
        density, flux = ctm_corridor_step(*values, .005)
        expected = torch.tensor([[900., 1200., 1000., 600.]], dtype=torch.double)
        torch.testing.assert_close(flux, expected)
        torch.testing.assert_close(((density - values[0]) * values[2]).sum(1), .005 * (flux[:, 0] - flux[:, -1]))
        self.assertTrue((density >= 0).all() and (density <= values[5]).all())

    def test_blocked_downstream_receiving_causes_upstream_accumulation(self):
        values = self.inputs()
        values[1][:, -1] = 0.
        updated, flux = ctm_corridor_step(*values, .005)
        self.assertEqual(float(flux[0, 2]), 0)
        self.assertGreater(float(updated[0, 1]), float(values[0][0, 1]))

    def test_cfl_capacity_envelope_and_storage_are_enforced(self):
        with self.assertRaisesRegex(ValueError, 'CFL'):
            ctm_corridor_step(*self.inputs(), 5 / 60)
        values = self.inputs()
        values[1].fill_(3000)
        with self.assertRaisesRegex(ValueError, 'envelope'):
            ctm_corridor_step(*values, .005)
        values = self.inputs()
        values[0][0, 0] = 160.
        with self.assertRaisesRegex(ValueError, 'storage'):
            ctm_corridor_step(*values, .005)


class PhysicsContractTests(unittest.TestCase):
    def test_unresolved_units_and_changed_node_identity_block_real_use(self):
        good = example_contract()
        validate_contract(good, [10, 20, 30], good['package_sha256'])
        for key, value in [('flow_unit', None), ('lane_basis', 'per_lane'),
                           ('status', 'unresolved'), ('interval_label', 'unknown')]:
            bad = copy.deepcopy(good)
            bad[key] = value
            with self.assertRaises(ValueError):
                validate_contract(bad, [10, 20, 30], good['package_sha256'])
        with self.assertRaisesRegex(ValueError, 'station order'):
            validate_contract(good, [20, 10, 30], good['package_sha256'])
        with self.assertRaisesRegex(ValueError, 'fingerprints'):
            validate_contract(good, [10, 20, 30], {})

    def test_readout_and_evidence_are_explicit(self):
        c = example_contract()
        cfg = queue_config(c, 'queue')
        np.testing.assert_array_equal(cfg['readout_weights'], [[0, 1, 0]])
        c['bottlenecks'][0]['readout_station_id'] = 30
        with self.assertRaisesRegex(ValueError, 'readout'):
            validate_contract(c, [10, 20, 30], c['package_sha256'])
        c = example_contract()
        c['evidence']['flow_units'] = None
        with self.assertRaisesRegex(ValueError, 'evidence'):
            validate_contract(c, [10, 20, 30], c['package_sha256'])

    def test_mapping_converts_counts_masks_missing_and_rejects_future(self):
        c = example_contract()
        raw = torch.full((2, 12, 3), 50.)
        mask = torch.ones_like(raw, dtype=torch.bool)
        raw[0, 0, 0], mask[0, 0, 0] = float('nan'), False
        incident = synthetic_fixture(torch.device('cpu'))[1]['incident']
        result = build_physical_inputs(raw, mask, incident, c)
        self.assertEqual(float(result['history_rates'][1, 0, 0, 0]), 600)
        self.assertEqual(float(result['history_rates'][0, 0, 0, 0]), 0)
        self.assertFalse(result['history_valid'][0, 0, 0, 0])
        with self.assertRaisesRegex(ValueError, 'only history'):
            build_physical_inputs(torch.ones(2, 26, 3), torch.ones(2, 26, 3, dtype=torch.bool), incident, c)

    def test_profile_cannot_use_gap_or_forecast_values(self):
        a = np.ones((3, 26, 2), dtype=np.float32)
        a[0, 0, 0] = 0.
        before = history_profiles(a, [10, 20])
        a[:, 12:, :] = float('nan')
        self.assertEqual(before, history_profiles(a, [10, 20]))
        self.assertEqual(before[0]['zero_cells'], 1)

    def test_timing_preserves_gap_and_does_not_certify_online_semantics(self):
        row = dict(t0='2023-01-03T15:25:00', x_start='2023-01-03T14:20:00',
                   x_end='2023-01-03T15:15:00', y_start='2023-01-03T15:30:00', y_end='2023-01-03T16:25:00')
        result = check_manifest_timing([row], 'end')
        self.assertEqual(result['unobserved_gap_minutes_if_common_5min_bins'], 10)
        self.assertFalse(result['online_semantics_certified'])
        row['x_end'] = row['t0']
        with self.assertRaises(ValueError):
            check_manifest_timing([row])


class QueueBranchTests(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cpu')
        self.reference, self.batch, self.physical, self.config, _ = synthetic_fixture(self.device)

    def test_both_arms_preserve_common_initial_state_and_exact_prediction(self):
        native = self.reference.eval()
        with torch.no_grad():
            expected = native(self.batch['x'], incident_data=self.batch['incident'])
            models = [augmented(native, self.config, mode, self.device).eval() for mode in ('queue', 'recurrent')]
            for model in models:
                actual = model(self.batch['x'], incident_data=self.batch['incident'], physical_inputs=self.physical)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            for name, value in models[0].queue_branch.state_dict().items():
                torch.testing.assert_close(value, models[1].queue_branch.state_dict()[name], atol=0, rtol=0)
        for key, value in (('incident_schema', 'legacy'), ('incident_routing', 'acdg'), ('time_response', 'shared')):
            args = {**self.reference._model_args, key: value}
            with self.assertRaisesRegex(ValueError, 'complete fixed'):
                QueueAugmentedIGSTGNN(args, self.config, node_num=3, input_dim=3, output_dim=1, seq_len=12, horizon=12)

    def test_event_only_changes_capacity_not_initial_queue_or_arrival(self):
        branch = IncidentQueueBranch(**self.config)
        _, first = branch(self.physical)
        other = copy.deepcopy(self.physical)
        other['event_features'] += 2
        _, second = branch(other)
        torch.testing.assert_close(first['initial_queue'], second['initial_queue'], rtol=0, atol=0)
        torch.testing.assert_close(first['arrival'], second['arrival'], rtol=0, atol=0)
        self.assertGreater(float((first['capacity'] - second['capacity']).abs().max()), 0)

    def test_gap_states_are_rolled_out_and_readout_is_local_even_after_learning(self):
        branch = IncidentQueueBranch(**self.config)
        with torch.no_grad():
            branch.projection.weight.fill_(.1)
        delta, trace = branch(self.physical)
        self.assertEqual(trace['queue'].shape[1], 71)
        expected = torch.stack([trace['queue'][:, 11:16].mean(1) / branch.queue_scale,
                                trace['departure'][:, 10:15].mean(1) / branch.base_capacity,
                                trace['capacity'][:, 10:15].mean(1) / branch.base_capacity,
                                trace['arrival'][:, 10:15].mean(1) / branch.base_capacity], -1)
        torch.testing.assert_close(trace['features'][:, 0], expected)
        self.assertGreater(float(delta[:, :, 1].abs().sum()), 0)
        self.assertEqual(float(delta[:, :, [0, 2]].abs().sum()), 0)
        off = copy.deepcopy(self.physical)
        off['report_support'].zero_()
        self.assertEqual(float(branch(off)[0].abs().sum()), 0)

    def test_missing_placeholders_cannot_change_state(self):
        branch = IncidentQueueBranch(**self.config)
        inputs = copy.deepcopy(self.physical)
        inputs['history_valid'][:, 0, :, 0] = False
        original = branch(inputs)[1]['arrival']
        inputs['history_rates'][:, 0, :, 0] = 1e9
        torch.testing.assert_close(branch(inputs)[1]['arrival'], original, atol=0, rtol=0)

    def test_failed_backbone_forward_removes_temporary_context(self):
        model = augmented(self.reference, self.config, 'queue', self.device)
        with patch.object(model, '_graph_constructor', side_effect=RuntimeError('injected')):
            with self.assertRaisesRegex(RuntimeError, 'injected'):
                model(self.batch['x'], incident_data=self.batch['incident'], physical_inputs=self.physical)
        self.assertFalse(hasattr(model, '_queue_delta'))
        actual = model(self.batch['x'], incident_data=self.batch['incident'], physical_inputs=self.physical)
        self.assertEqual(actual.shape, (2, 12, 3, 1))

    def test_label_is_not_used_as_a_physics_input(self):
        model = augmented(self.reference, self.config, 'queue', self.device).eval()
        with torch.no_grad():
            model.queue_branch.projection.weight.fill_(.1)
            a = model(self.batch['x'], label=torch.zeros(2, 12, 3, 1), incident_data=self.batch['incident'], physical_inputs=self.physical)
            b = model(self.batch['x'], label=torch.full((2, 12, 3, 1), float('nan')), incident_data=self.batch['incident'], physical_inputs=self.physical)
        torch.testing.assert_close(a, b, atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
