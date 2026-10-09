"""Behavioral contracts for relative state, input clocks and matched control."""
import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.models.incident_relative_capacity import SharedIncidentState, supply_limited_flow
from src.utils.incident_candidate_history import SCHEMA
from src.utils.incident_corridor import sha256, write_json, write_rows
from src.utils.relative_capacity_inputs import CapacityHistoryInputs

REPO = Path(__file__).resolve().parents[1]
torch.set_num_threads(3)


def example_inputs(nodes=4, batch=2, dtype=torch.float32, device='cpu'):
    generator = torch.Generator().manual_seed(17)
    history = .2 + torch.rand(batch, 12, nodes, 3, generator=generator)
    report = torch.zeros(batch, nodes, 3)
    report[:, ::2, 1] = .8
    report[:, ::2, 2] = 1
    return {'history': history.to(device=device, dtype=dtype),
            'valid': torch.ones_like(history, dtype=torch.bool, device=device),
            'references': torch.ones(nodes, 3, dtype=dtype, device=device),
            'report_features': report.to(device=device, dtype=dtype),
            'report_age_minutes': torch.full((batch,), 3., dtype=dtype, device=device),
            'elapsed_minutes': torch.arange(0, 65, 5, dtype=dtype, device=device)}


class RelativeCapacityTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)

    def test_capacity_recovery_low_demand_and_downstream_limit(self):
        model = SharedIncidentState().double()
        # Controlled head: a=0.5, tau=30 min; test against an analytic trajectory.
        with torch.no_grad():
            model.condition_head[-1].weight.zero_()
            model.condition_head[-1].bias.copy_(torch.tensor([2., np.log(25 / 150)], dtype=torch.double))
        inputs = example_inputs(dtype=torch.double)
        state = model(**inputs)['state'][:, :, ::2]
        expected = 1 - .5 * torch.exp(-inputs['elapsed_minutes'] / 30)
        torch.testing.assert_close(state[0, :, 0], expected)
        self.assertTrue((torch.diff(state, dim=1) > 0).all())
        base = torch.ones_like(state)
        low = torch.full_like(state, .2)
        torch.testing.assert_close(supply_limited_flow(low, base, state, base), low)
        high = torch.full_like(state, 2.)
        torch.testing.assert_close(supply_limited_flow(high, base, state, high), state)
        blocked = torch.full_like(state, .1)
        torch.testing.assert_close(supply_limited_flow(high, base, state, blocked), blocked)
        torch.testing.assert_close(supply_limited_flow(high, base, state, torch.zeros_like(state)), torch.zeros_like(state))

    def test_zero_amplitude_and_bounds_are_exactly_supported(self):
        model, inputs = SharedIncidentState(), example_inputs()
        with torch.no_grad():
            model.condition_head[-1].weight.zero_()
            model.condition_head[-1].bias[0] = -5
        self.assertTrue((model(**inputs)['state'] == 1).all())
        with torch.no_grad():
            model.condition_head[-1].bias[0] = 20
        state = model(**inputs)['state']
        self.assertTrue((state >= .05 - 1e-7).all() and (state <= 1).all())

    def test_intervention_locality_restoration_and_history_is_report_independent(self):
        for mode in ('capacity', 'ordinary'):
            model, inputs = SharedIncidentState(mode=mode), example_inputs()
            before = copy.deepcopy(model.state_dict())
            on = model(**inputs)
            off = model(**inputs, incident_enabled=False)
            self.assertTrue((off['state'] == 1).all())
            self.assertTrue((on['state'][:, :, 1::2] == 1).all())
            torch.testing.assert_close(on['history_state'], off['history_state'], rtol=0, atol=0)
            altered = dict(inputs, report_age_minutes=inputs['report_age_minutes'] + 50)
            torch.testing.assert_close(model(**altered)['history_state'], on['history_state'], rtol=0, atol=0)
            torch.testing.assert_close(model(**inputs)['state'], on['state'], rtol=0, atol=0)
            for k, value in before.items():
                torch.testing.assert_close(value, model.state_dict()[k], rtol=0, atol=0)

    def test_masked_values_do_not_become_observations_and_all_missing_is_neutral(self):
        model, inputs = SharedIncidentState(), example_inputs()
        inputs['valid'][:, :, 0] = False
        inputs['valid'][:, :5, 2, 1] = False
        x = inputs['history'].clone()
        x[~inputs['valid']] = float('nan')
        left = model(**dict(inputs, history=x))
        x[~inputs['valid']] = -999.
        right = model(**dict(inputs, history=x))
        torch.testing.assert_close(left['state'], right['state'], atol=0, rtol=0)
        self.assertTrue((left['state'][:, :, 0] == 1).all())
        self.assertFalse(left['history_available'][:, 0].any())
        self.assertTrue(left['history_available'][:, 2].all())

    def test_shared_parameters_permutation_batch_and_unit_scale_invariance(self):
        model, inputs = SharedIncidentState().double(), example_inputs(dtype=torch.double)
        reference = model(**inputs)['state']
        order = torch.tensor([2, 0, 3, 1])
        permuted = dict(inputs, history=inputs['history'][:, :, order], valid=inputs['valid'][:, :, order],
                        references=inputs['references'][order], report_features=inputs['report_features'][:, order])
        torch.testing.assert_close(model(**permuted)['state'], reference[:, :, order], atol=1e-12, rtol=0)
        scales = torch.tensor([100., .01, 1.609344], dtype=torch.double)
        scaled = dict(inputs, history=inputs['history'] * scales, references=inputs['references'] * scales)
        torch.testing.assert_close(model(**scaled)['state'], reference, atol=1e-12, rtol=0)
        single = {k: v[:1] if k in ('history', 'valid', 'report_features', 'report_age_minutes') else v for k, v in inputs.items()}
        torch.testing.assert_close(model(**single)['state'], reference[:1], atol=1e-12, rtol=0)
        self.assertEqual(model(**example_inputs(nodes=496, batch=1, dtype=torch.double))['state'].shape, (1, 13, 496))
        self.assertEqual(model(**example_inputs(nodes=1, batch=1, dtype=torch.double))['state'].shape, (1, 13, 1))

    def test_control_matches_parameters_and_common_initialization_without_capacity_bounds(self):
        capacity = SharedIncidentState()
        ordinary = SharedIncidentState(mode='ordinary')
        ordinary.load_state_dict(capacity.state_dict())
        self.assertEqual(sum(p.numel() for p in capacity.parameters()), sum(p.numel() for p in ordinary.parameters()))
        inputs = example_inputs()
        torch.testing.assert_close(capacity(**inputs)['history_state'], ordinary(**inputs)['history_state'], rtol=0, atol=0)
        with torch.no_grad():
            ordinary.condition_head[-1].weight.zero_()
            ordinary.condition_head[-1].bias.fill_(2)
        result = ordinary(**inputs)
        self.assertNotIn('capacity_loss', result)
        self.assertTrue((result['state'][:, :, ::2] > 1).all())

    def test_gradients_reach_history_and_condition_but_not_unsupported_reports(self):
        for mode in ('capacity', 'ordinary'):
            model, inputs = SharedIncidentState(mode=mode).double(), example_inputs(dtype=torch.double)
            inputs['history'].requires_grad_()
            inputs['report_features'].requires_grad_()
            loss = (model(**inputs)['state'] - .7).square().mean()
            loss.backward()
            for name, p in model.named_parameters():
                self.assertIsNotNone(p.grad, name)
                self.assertTrue(torch.isfinite(p.grad).all(), name)
                self.assertGreater(float(p.grad.abs().sum()), 0, name)
            self.assertGreater(float(inputs['history'].grad.abs().sum()), 0)
            self.assertTrue((inputs['report_features'].grad[:, 1::2] == 0).all())

    def test_gradcheck_supply_and_capacity_inside_smooth_regime(self):
        model = SharedIncidentState(hidden=3).double()
        inputs = example_inputs(nodes=1, batch=1, dtype=torch.double)
        x = inputs['history'].requires_grad_()
        self.assertTrue(torch.autograd.gradcheck(lambda value: model(**dict(inputs, history=value))['state'], (x,)))
        inputs2 = tuple(torch.tensor([v], dtype=torch.double, requires_grad=True) for v in (2., 1., .7, 3.))
        self.assertTrue(torch.autograd.gradcheck(supply_limited_flow, inputs2))

    def test_optimizer_resume_matches_uninterrupted_next_update(self):
        for mode in ('capacity', 'ordinary'):
            model, inputs = SharedIncidentState(hidden=7, mode=mode), example_inputs()
            optimizer = torch.optim.Adam(model.parameters(), lr=.003)
            def step(m, o):
                o.zero_grad(set_to_none=True)
                (m(**inputs)['state'] - .8).square().mean().backward()
                o.step()
            step(model, optimizer)
            memory = io.BytesIO()
            torch.save({'config': model.configuration(), 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, memory)
            memory.seek(0)
            saved = torch.load(memory, weights_only=True)
            restored = SharedIncidentState(**saved['config'])
            restored.load_state_dict(saved['model'])
            resumed = torch.optim.Adam(restored.parameters(), lr=.003)
            resumed.load_state_dict(saved['optimizer'])
            step(model, optimizer)
            step(restored, resumed)
            for k, value in model.state_dict().items():
                torch.testing.assert_close(value, restored.state_dict()[k], atol=0, rtol=0)

    def test_invalid_inputs_and_future_conditions_fail_closed(self):
        model, inputs = SharedIncidentState(), example_inputs()
        mutations = [dict(references=torch.zeros(4, 3)), dict(report_age_minutes=torch.tensor([-1., 1.])),
                     dict(elapsed_minutes=torch.tensor([0., 0.])), dict(elapsed_minutes=torch.tensor([-1., 5.])),
                     dict(valid=inputs['valid'].float()), dict(history=torch.full_like(inputs['history'], float('nan'))),
                     dict(report_features=torch.ones_like(inputs['report_features'])),
                     dict(history=inputs['history'][:, :11]), dict(elapsed_minutes=torch.tensor([]))]
        for change in mutations:
            with self.subTest(keys=list(change)), self.assertRaises(ValueError):
                model(**dict(inputs, **change))
        with self.assertRaises(TypeError):
            model(**inputs, future_y=torch.ones(1))
        with self.assertRaises(ValueError):
            supply_limited_flow(*[torch.tensor([x]) for x in (1., 1., 1.1, 1.)])
        with self.assertRaises(ValueError):
            supply_limited_flow(*[torch.tensor([x]) for x in (-1., 1., 1., 1.)])

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA not available on this host')
    def test_cuda_state_matches_cpu_and_has_gradients(self):
        cpu = SharedIncidentState()
        gpu = copy.deepcopy(cpu).cuda()
        expected = cpu(**example_inputs())['state']
        actual = gpu(**example_inputs(device='cuda'))['state']
        torch.testing.assert_close(actual.cpu(), expected, atol=1e-5, rtol=1e-5)
        actual.sum().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in gpu.parameters()))


class CapacityInputTests(unittest.TestCase):
    def setUp(self):
        # Windows tempfile outside the workspace may be unreadable under sandbox.
        self.tmp = tempfile.TemporaryDirectory(prefix='capacity_test_', dir=REPO)
        self.root = Path(self.tmp.name).resolve()
        self.assertEqual(self.root.parent, REPO)
        np.save(self.root / 'history_values.npy', np.ones((12, 2, 3), dtype=np.float32))
        np.save(self.root / 'history_usable.npy', np.ones((12, 2, 3), dtype=bool))
        np.save(self.root / 'history_labels.npy', np.datetime64('2023-02-01T11:00') + np.arange(0, 60, 5).astype('timedelta64[m]'))
        np.save(self.root / 'history_index.npy', np.arange(12).reshape(1, 12))
        np.save(self.root / 'station_ids.npy', [10, 20])
        write_rows(self.root / 'train_events.csv', [dict(t0='2023-02-01T12:05:00', report_time='2023-02-01T12:03:00', split='train')])
        write_rows(self.root / 'station_observation_diagnostics.csv',
                   [dict(station_id=n, flow_reference=2, occupancy_reference=3, speed_reference=4) for n in (10, 20)])
        np.savez(self.root / 'train_report.npz', station_ids=[10, 20], candidate_indices=[0],
                 distances=np.array([[[0, 1, 0], [0, 0, 0]]], dtype=np.float32), report_age_minutes=np.array([2.], dtype=np.float32))
        self.manifest()

    def tearDown(self):
        self.tmp.cleanup()

    def manifest(self):
        write_json(self.root / 'summary.json', dict(schema=SCHEMA, status='CANDIDATE_TRAIN_X_PACK_COMPLETE',
                   relative_references_scope='unique_candidate_train_X_station_channel_p95_not_capacity',
                   outputs_sha256={p.name: sha256(p) for p in self.root.iterdir() if p.name != 'summary.json'}))

    def test_roundtrip_reads_only_history_and_metadata(self):
        original = np.load
        opened = []
        def guarded(path, *args, **kwargs):
            opened.append(Path(path).name)
            self.assertNotIn('target', str(path))
            self.assertNotIn('validation', str(path))
            return original(path, *args, **kwargs)
        with patch('numpy.load', side_effect=guarded):
            adapter = CapacityHistoryInputs(self.root)
            try:
                inputs = adapter.batch([0], [0., 5.])
                self.assertEqual(SharedIncidentState()(**inputs)['state'].shape, (1, 2, 2))
                torch.testing.assert_close(inputs['references'], torch.tensor([[2., 3., 4.], [2., 3., 4.]]))
                with self.assertRaises(ValueError):
                    adapter.batch([-1], [0.])
            finally:
                adapter.close()
        self.assertEqual(set(opened), {'history_values.npy', 'history_usable.npy', 'history_labels.npy',
                                      'history_index.npy', 'station_ids.npy', 'train_report.npz'})

    def test_checksum_tampering_is_rejected(self):
        np.save(self.root / 'history_values.npy', np.zeros((12, 2, 3), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, 'checksum'):
            CapacityHistoryInputs(self.root)

    def test_shifted_history_and_station_axes_rejected_even_with_new_hashes(self):
        labels = np.load(self.root / 'history_labels.npy')
        np.save(self.root / 'history_labels.npy', labels + np.timedelta64(5, 'm'))
        self.manifest()
        with self.assertRaisesRegex(ValueError, 'clock'):
            CapacityHistoryInputs(self.root)
        np.save(self.root / 'history_labels.npy', labels)
        np.save(self.root / 'station_ids.npy', [20, 10])
        self.manifest()
        with self.assertRaisesRegex(ValueError, 'station axis'):
            CapacityHistoryInputs(self.root)


if __name__ == '__main__':
    unittest.main()
