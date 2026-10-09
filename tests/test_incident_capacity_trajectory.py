"""M2.2 curve behavior, matched controls and compatibility with M2.1."""
import copy
import io
import unittest

import torch

from experiments.chronological.check_incident_relative_capacity import synthetic_inputs
from src.models.incident_relative_capacity import SharedIncidentState, supply_limited_flow
from src.models.incident_capacity_trajectory import TrajectoryIncidentState, capacity_curve, pulse_kernel

torch.set_num_threads(3)


def fixture(dtype=torch.float64):
    return {key: value.to(dtype) if value.is_floating_point() else value
            for key, value in synthetic_inputs('cpu').items()}


class CapacityTrajectoryTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(226)

    def curve(self, kind='pulse', amplitude=.2, strength=.6, time=None):
        args = [torch.tensor([[v]], dtype=torch.double) for v in (amplitude, 30., strength, 30.)]
        if time is None:
            time = torch.tensor([0., 5., 15., 30., 60., 120., 600.], dtype=torch.double)
        return capacity_curve(*args, time, kind)[0, :, 0]

    def test_pulse_worsens_then_recovers_and_mixture_always_recovers(self):
        pulse = self.curve()
        mixture = self.curve('mixture')
        self.assertEqual(float(pulse[0]), .2)
        self.assertGreater(float(pulse[3]), float(pulse[0]))
        self.assertLess(float(pulse[-1]), 1e-6)
        self.assertTrue((torch.diff(mixture) <= 0).all())
        self.assertTrue((pulse >= 0).all() and (pulse <= .95).all())

    def test_delayed_impact_zero_disturbance_and_exact_exponential_nesting(self):
        delayed = self.curve(amplitude=0)
        self.assertEqual(float(delayed[0]), 0)
        self.assertGreater(float(delayed[3]), 0)
        self.assertTrue((self.curve(amplitude=0, strength=0) == 0).all())
        times = torch.tensor([0., 5., 15., 30., 60., 120., 600.], dtype=torch.double)
        base = .2 * torch.exp(-times / 30)
        for kind in ('pulse', 'mixture'):
            torch.testing.assert_close(self.curve(kind, strength=0), base, atol=0, rtol=0)

    def test_kernel_peak_grid_independence_and_continuous_slope_bound(self):
        times = torch.linspace(0, 180, 721, dtype=torch.double, requires_grad=True)
        args = [torch.tensor([[v]], dtype=torch.double) for v in (.2, 30., .6, 30.)]
        curve = capacity_curve(*args, times, 'pulse')[0, :, 0]
        derivative, = torch.autograd.grad(curve.sum(), times)
        bound = .2/30 + .95*.6*torch.exp(torch.tensor(1., dtype=torch.double))/30
        self.assertTrue((derivative.abs() <= bound + 1e-12).all())
        torch.testing.assert_close(capacity_curve(*args, times.detach()[::20], 'pulse')[0, :, 0], curve.detach()[::20], rtol=0, atol=0)
        kernel = pulse_kernel(times.detach(), torch.tensor(30., dtype=torch.double))
        self.assertEqual(float(times[kernel.argmax()]), 30.)
        self.assertEqual(float(kernel.max()), 1.)
        # The total loss peak differs from the isolated kernel's peak.
        self.assertLess(float(times[curve.argmax()]), 30.)

    def test_low_demand_and_downstream_supply_override_capacity_shape(self):
        retention = 1 - self.curve()
        base = torch.ones_like(retention)
        low = torch.full_like(base, .02)
        high = torch.full_like(base, 2.)
        supply = torch.full_like(base, .01)
        torch.testing.assert_close(supply_limited_flow(low, base, retention, high), low)
        torch.testing.assert_close(supply_limited_flow(high, base, retention, high), retention)
        torch.testing.assert_close(supply_limited_flow(high, base, retention, supply), supply)

    def test_dense_extreme_parameters_stay_bounded_and_long_time_is_finite(self):
        grid = torch.cartesian_prod(torch.tensor([0., .2, .95]), torch.tensor([5., 180.]),
                                    torch.tensor([0., .5, 1.]), torch.tensor([15., 90.])).double()
        args = [grid[:, i:i+1] for i in range(4)]
        time = torch.cat([torch.linspace(0, 360, 721, dtype=torch.double), torch.tensor([1e100], dtype=torch.double)])
        for kind in ('pulse', 'mixture'):
            loss = capacity_curve(*args, time, kind)
            self.assertTrue(torch.isfinite(loss).all())
            self.assertTrue((loss >= -1e-15).all() and (loss <= .95 + 1e-15).all())
            self.assertTrue((loss[:, -1] == 0).all())
            if kind == 'mixture':
                self.assertTrue((torch.diff(loss, dim=1) <= 1e-15).all())

    def test_all_four_parameters_have_finite_gradients_and_numerical_gradcheck(self):
        inputs = fixture()
        for kind in ('pulse', 'mixture', 'ordinary'):
            model = TrajectoryIncidentState(hidden=5, trajectory=kind).double()
            result = model(**inputs)
            (result['state'] - .5).square().mean().backward()
            head = model.condition_head[-1]
            self.assertTrue(torch.isfinite(head.weight.grad).all())
            self.assertTrue((head.weight.grad.abs().sum(-1) > 0).all(), kind)
            self.assertTrue((head.bias.grad.abs() > 0).all(), kind)
            self.assertGreater(float(model.history_encoder.weight_ih_l0.grad.abs().sum()), 0)
        args = tuple(torch.tensor([[v]], dtype=torch.double, requires_grad=True) for v in (.2, 30., .6, 50.))
        time = torch.tensor([0., 5., 20., 60.], dtype=torch.double)
        for kind in ('pulse', 'mixture'):
            self.assertTrue(torch.autograd.gradcheck(lambda *p: capacity_curve(*p, time, kind), args))

    def test_common_weights_and_zero_strength_reproduce_legacy_exactly(self):
        legacy, inputs = SharedIncidentState().double(), fixture()
        expanded = TrajectoryIncidentState().double()
        expanded.initialize_common_from(legacy)
        with torch.no_grad():
            expanded.condition_head[-1].weight[2].zero_()
            expanded.condition_head[-1].bias[2] = -.5  # exact p=0, not an epsilon
        state = legacy(**inputs)['state']
        for kind in ('pulse', 'mixture'):
            current = TrajectoryIncidentState(trajectory=kind).double()
            current.load_state_dict(expanded.state_dict())
            torch.testing.assert_close(current(**inputs)['state'], state, rtol=0, atol=0)
        self.assertEqual(legacy.configuration(), {'hidden': 16, 'mode': 'capacity'})
        with self.assertRaises(ValueError):
            expanded.initialize_common_from(SharedIncidentState(mode='ordinary'))

    def test_condition_intervention_missing_history_and_report_locality(self):
        inputs = fixture()
        inputs['valid'][:, :, 0] = False
        inputs['history'][:, :, 0] = float('nan')
        for kind in ('pulse', 'mixture', 'ordinary'):
            model = TrajectoryIncidentState(trajectory=kind).double()
            before = copy.deepcopy(model.state_dict())
            on = model(**inputs)
            off = model(**inputs, incident_enabled=False)
            self.assertTrue((off['state'] == 1).all())
            self.assertTrue((on['state'][:, :, [0, 1, 3]] == 1).all())
            torch.testing.assert_close(on['history_state'], off['history_state'], atol=0, rtol=0)
            torch.testing.assert_close(model(**inputs)['state'], on['state'], atol=0, rtol=0)
            for k, value in before.items():
                torch.testing.assert_close(value, model.state_dict()[k], rtol=0, atol=0)

    def test_parameter_budget_configuration_and_optimizer_continuation(self):
        models = [TrajectoryIncidentState(hidden=7, trajectory=k) for k in ('pulse', 'mixture', 'ordinary')]
        self.assertEqual(len({sum(p.numel() for p in m.parameters()) for m in models}), 1)
        inputs = fixture(torch.float32)
        for model in models:
            opt = torch.optim.Adam(model.parameters(), lr=.002)
            def step(m, o):
                o.zero_grad(set_to_none=True)
                (m(**inputs)['state'] - .5).square().mean().backward()
                o.step()
            step(model, opt)
            buffer = io.BytesIO()
            torch.save({'config': model.configuration(), 'model': model.state_dict(), 'optimizer': opt.state_dict()}, buffer)
            buffer.seek(0)
            checkpoint = torch.load(buffer, weights_only=True)
            restored = TrajectoryIncidentState(**checkpoint['config'])
            restored.load_state_dict(checkpoint['model'])
            resumed = torch.optim.Adam(restored.parameters(), lr=.002)
            resumed.load_state_dict(checkpoint['optimizer'])
            step(model, opt)
            step(restored, resumed)
            for k, value in model.state_dict().items():
                torch.testing.assert_close(value, restored.state_dict()[k], rtol=0, atol=0)

    def test_invalid_time_parameters_and_unknown_modes_rejected(self):
        args = [torch.tensor([[v]], dtype=torch.double) for v in (.2, 30., .6, 30.)]
        for index, bad in ((0, 1.), (1, 0.), (2, -1.), (3, 1.), (0, float('nan'))):
            altered = [a.clone() for a in args]
            altered[index].fill_(bad)
            with self.assertRaises(ValueError):
                capacity_curve(*altered, torch.tensor([0., 5.], dtype=torch.double), 'pulse')
        for time in ([-1., 0.], [0., 0.], []):
            with self.assertRaises(ValueError):
                capacity_curve(*args, torch.tensor(time, dtype=torch.double), 'pulse')
        with self.assertRaises(ValueError):
            TrajectoryIncidentState(trajectory='unknown')

    def test_inactive_shape_and_equal_time_scales_are_not_identifiable(self):
        # Record these structural degeneracies rather than claiming every
        # finite gradient establishes unique physical parameter identification.
        times = torch.arange(0., 65., 5., dtype=torch.double)
        a, tau = [torch.tensor([[v]], dtype=torch.double) for v in (.2, 30.)]
        zero, one = torch.zeros_like(a), torch.ones_like(a)
        torch.testing.assert_close(capacity_curve(a, tau, zero, tau, times, 'mixture'),
                                   capacity_curve(a, tau, one, tau, times, 'mixture'), rtol=0, atol=0)
        torch.testing.assert_close(capacity_curve(a, tau, zero, tau, times, 'pulse'),
                                   capacity_curve(a, tau, zero, tau*2, times, 'pulse'), rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_shapes_match_cpu_with_nonzero_new_parameter_gradients(self):
        for kind in ('pulse', 'mixture', 'ordinary'):
            cpu = TrajectoryIncidentState(trajectory=kind)
            gpu = copy.deepcopy(cpu).cuda()
            actual = gpu(**synthetic_inputs('cuda'))['state']
            expected = cpu(**synthetic_inputs('cpu'))['state']
            torch.testing.assert_close(actual.cpu(), expected, atol=1e-5, rtol=1e-5)
            (actual - .5).square().mean().backward()
            self.assertTrue((gpu.condition_head[-1].weight.grad.abs().sum(-1) > 0).all())


if __name__ == '__main__':
    unittest.main()
