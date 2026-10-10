"""Independent budget, timing, intervention and gradient checks for V2 M3.0."""
import copy
import io
import math
import unittest

import torch

from src.models.incident_capacity_exchange import (
    ExchangeGraph, CapacityLimitedExchange, OrdinaryDirectedRecurrence,
    bernstein_loss, condition_coefficients, edge_capacities, target_window_features)

torch.set_num_threads(3)


def graph(nodes, edges, *, mask=None, boundaries=(), device='cpu'):
    active = torch.ones(nodes, dtype=torch.bool, device=device) if mask is None else torch.tensor(mask, dtype=torch.bool, device=device)
    boundary = torch.zeros(nodes, dtype=torch.bool, device=device)
    boundary[list(boundaries)] = True
    index = torch.tensor(edges, dtype=torch.long, device=device).reshape(-1, 2).T.contiguous()
    return ExchangeGraph(nodes, index, active, boundary, evidence_scope='synthetic_only')


def chain_fixture(nodes=4, channels=1, steps=52, device='cpu', dtype=torch.double):
    edges = [(-1, 0)] + [(i, i+1) for i in range(nodes-1)] + [(nodes-1, -1)]
    topology = graph(nodes, edges, boundaries=(0, nodes-1), device=device)
    alpha = torch.ones(nodes, channels, dtype=dtype, device=device)
    weights = torch.ones(nodes+1, dtype=dtype, device=device)
    weights[0] = 0
    times = torch.arange(steps+1, dtype=dtype, device=device)*1.25
    capacity = torch.ones(1, steps, nodes+1, channels, dtype=dtype, device=device)
    capacity[:, :, 0] = 0
    boundary = torch.zeros_like(capacity)
    initial = torch.full((1, nodes, channels), .5, dtype=dtype, device=device)
    return topology, alpha, weights, times, capacity, boundary, initial


class CapacityExchangeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3010)

    def test_single_chain_matches_independent_minimum_and_shared_update(self):
        topology, alpha, weights, times, cap, boundary, initial = chain_fixture(nodes=3, steps=1)
        initial[0, :, 0] = torch.tensor([.8, .2, .1])
        cap[0, 0, 1, 0] = .3
        boundary[0, 0, 0, 0] = .4
        result = CapacityLimitedExchange(topology, alpha, alpha)(initial, cap, boundary, times, weights)
        expected_flux = torch.tensor([.2, .3, .2, .1], dtype=torch.double)
        torch.testing.assert_close(result['edge_flux'][0, 0, :, 0], expected_flux)
        expected = initial[0, :, 0] + .25*(expected_flux[:-1]-expected_flux[1:])
        torch.testing.assert_close(result['states'][0, 1, :, 0], expected)
        self.assertLess(float(result['balance_residual'].abs().max()), 1e-15)

    def test_merge_shares_one_receiving_budget(self):
        topology = graph(3, [(0, 2), (1, 2)])
        initial = torch.tensor([[[.9], [.9], [.8]]], dtype=torch.double)
        rate = torch.ones(3, 1, dtype=torch.double)
        cap = torch.ones(1, 1, 2, 1, dtype=torch.double)
        result = CapacityLimitedExchange(topology, rate, rate)(
            initial, cap, torch.zeros_like(cap), torch.tensor([0., 1.25], dtype=torch.double), torch.ones(2, dtype=torch.double))
        torch.testing.assert_close(result['edge_flux'].flatten(), torch.tensor([.1, .1], dtype=torch.double))
        self.assertAlmostEqual(float(result['inflow'][0, 0, 2]), .2)
        self.assertTrue(result['receiving_limited'].all())

    def test_fork_and_exit_share_sending_budget_without_automatic_rerouting(self):
        topology = graph(3, [(0, 1), (0, 2), (0, -1)], boundaries=(0,))
        rate = torch.full((3, 1), 2., dtype=torch.double)
        initial = torch.tensor([[[.9], [0.], [0.]]], dtype=torch.double)
        weights = torch.tensor([.2, .3, .5], dtype=torch.double)
        cap = (2*weights)[None, None, :, None]
        args = (initial, cap, torch.zeros_like(cap), torch.tensor([0., 1.25], dtype=torch.double), weights)
        model = CapacityLimitedExchange(topology, rate, rate)
        on = model(*args)
        torch.testing.assert_close(on['edge_flux'].flatten(), 1.8*weights)
        blocked = cap.clone()
        blocked[:, :, 0] = 0
        off = model(initial, blocked, args[2], args[3], weights)
        torch.testing.assert_close(off['edge_flux'][..., 1:, :], on['edge_flux'][..., 1:, :])
        self.assertLess(float(off['outflow'][0, 0, 0]), float(on['outflow'][0, 0, 0]))

    def test_external_inlet_competes_with_internal_inflow(self):
        topology = graph(2, [(0, 1), (-1, 1)], boundaries=(1,))
        rate = torch.ones(2, 1, dtype=torch.double)
        initial = torch.tensor([[[.7], [.8]]], dtype=torch.double)
        cap = torch.tensor([[[[1.], [0.]]]], dtype=torch.double)
        boundary = torch.tensor([[[[0.], [2.]]]], dtype=torch.double)
        result = CapacityLimitedExchange(topology, rate, rate)(
            initial, cap, boundary, torch.tensor([0., 1.25], dtype=torch.double), torch.tensor([1., 0.], dtype=torch.double))
        torch.testing.assert_close(result['edge_flux'].flatten(), torch.tensor([.7, 2.], dtype=torch.double)*(.2/2.7))
        self.assertAlmostEqual(float(result['inflow'][0, 0, 1]), .2)

    def test_random_multichannel_graph_stays_bounded_and_balanced_without_clipping(self):
        topology = graph(5, [(-1, 0), (0, 1), (0, 2), (1, 3), (2, 3), (-1, 3), (3, 4), (4, -1)], boundaries=(0, 3, 4))
        rate = .05 + 1.95*torch.rand(5, 4, dtype=torch.double)
        receive = .05 + 1.95*torch.rand_like(rate)
        weights = torch.tensor([0., .4, .6, 1., 1., 0., 1., 1.], dtype=torch.double)
        times = torch.arange(201, dtype=torch.double)*1.25
        src = topology.edge_index[0]
        base = weights[:, None]*rate[src.clamp_min(0)]
        cap = torch.rand(3, 200, 8, 4, dtype=torch.double)*base[None, None]
        boundary = torch.zeros_like(cap)
        boundary[:, :, src < 0] = 2*torch.rand(3, 200, 2, 4, dtype=torch.double)
        initial = torch.rand(3, 5, 4, dtype=torch.double)
        result = CapacityLimitedExchange(topology, rate, receive)(initial, cap, boundary, times, weights)
        self.assertTrue((result['states'] >= 0).all() and (result['states'] <= 1).all())
        self.assertLess(float(result['balance_residual'].abs().max()), 2e-15)
        self.assertTrue((result['outflow'] <= rate*result['states'][:, :-1]+1e-14).all())
        self.assertTrue((result['inflow'] <= receive*(1-result['states'][:, :-1])+1e-14).all())

    def test_zero_bids_have_finite_outputs_and_backward(self):
        topology, rate, weights, times, cap, boundary, initial = chain_fixture(steps=4)
        initial.zero_().requires_grad_(True)
        result = CapacityLimitedExchange(topology, rate, rate)(initial, cap, boundary, times, weights)
        self.assertTrue((result['receiving_scale'] == 1).all())
        result['states'].sum().backward()
        self.assertTrue(torch.isfinite(initial.grad).all())
        self.assertTrue((result['edge_flux'] == 0).all())

    def test_bottleneck_propagates_upstream_outside_direct_report_support_and_recovers(self):
        topology, rate, weights, times, cap, boundary, initial = chain_fixture(steps=80)
        boundary[:, :16, 0] = .5
        model = CapacityLimitedExchange(topology, rate, rate)
        base = model(initial, cap, boundary, times, weights)
        reduced = cap.clone()
        reduced[:, :16, 2] = .05  # Only interface 1->2 directly changed.
        affected = model(initial, reduced, boundary, times, weights)
        self.assertGreater(float(affected['states'][0, 12, 1, 0]), float(base['states'][0, 12, 1, 0]))
        self.assertGreater(float(affected['states'][0, 12, 0, 0]), float(base['states'][0, 12, 0, 0]))
        self.assertLess(float(affected['states'][0, -1].sum()), float(affected['states'][0, 16].sum())*.05)
        self.assertTrue(torch.equal(model(initial, cap, boundary, times, weights)['states'], base['states']))

    def test_low_demand_can_make_capacity_change_exactly_invisible(self):
        topology, rate, weights, times, cap, boundary, initial = chain_fixture(steps=20)
        initial.fill_(.01)
        reduced = cap.clone()
        reduced[:, :, 2] = .5
        model = CapacityLimitedExchange(topology, rate, rate)
        self.assertTrue(torch.equal(model(initial, cap, boundary, times, weights)['states'],
                                    model(initial, reduced, boundary, times, weights)['states']))

    def test_receiving_bottleneck_can_mask_gradient_even_when_capacity_limits_the_bid(self):
        topology = graph(2, [(0, 1)])
        rate = torch.ones(2, 1, dtype=torch.double)
        initial = torch.tensor([[[.8], [.9]]], dtype=torch.double)
        cap = torch.full((1, 1, 1, 1), .2, dtype=torch.double, requires_grad=True)
        result = CapacityLimitedExchange(topology, rate, rate)(
            initial, cap, torch.zeros_like(cap), torch.tensor([0., 1.25], dtype=torch.double), torch.ones(1, dtype=torch.double))
        self.assertTrue(result['capacity_limited'].all() and result['receiving_limited'].all())
        self.assertAlmostEqual(float(result['edge_flux'].detach().sum()), .1)
        result['edge_flux'].sum().backward()
        torch.testing.assert_close(cap.grad, torch.zeros_like(cap), atol=1e-15, rtol=0)

    def test_edge_order_and_node_relabeling_do_not_change_solution(self):
        topology, rate, weights, times, cap, boundary, initial = chain_fixture()
        boundary[:, :, 0] = .3
        cap[:, :, 2] = .15
        expected = CapacityLimitedExchange(topology, rate, rate)(initial, cap, boundary, times, weights)
        order = torch.tensor([3, 0, 4, 1, 2])
        old_to_new = torch.tensor([2, 0, 3, 1])
        relabeled = topology.edge_index.clone()
        inside = relabeled >= 0
        relabeled[inside] = old_to_new[relabeled[inside]]
        new = ExchangeGraph(4, relabeled[:, order], topology.operator_mask,
                            topology.boundary_nodes[torch.argsort(old_to_new)], evidence_scope='synthetic_only')
        actual = CapacityLimitedExchange(new, rate, rate)(initial[:, torch.argsort(old_to_new)], cap[:, :, order],
                                                         boundary[:, :, order], times, weights[order])
        torch.testing.assert_close(actual['states'][:, :, old_to_new], expected['states'], atol=1e-14, rtol=0)
        torch.testing.assert_close(actual['edge_flux'][:, :, torch.argsort(order)], expected['edge_flux'], atol=1e-14, rtol=0)

    def test_condition_switch_preserves_history_and_does_not_mask_propagation_nodes(self):
        history = torch.full((2, 5, 4), .1, dtype=torch.double)
        report = torch.full_like(history, .8)
        support = torch.zeros(2, 5, dtype=torch.double)
        support[:, 2] = .7
        on = condition_coefficients(history, report, support)
        off = condition_coefficients(history, report, support, incident_enabled=False)
        self.assertTrue(torch.equal(off, history))
        self.assertTrue(torch.equal(on[:, [0, 1, 3, 4]], history[:, [0, 1, 3, 4]]))
        self.assertGreater(float(on[:, 2].mean()), float(history[:, 2].mean()))
        self.assertTrue(torch.equal(condition_coefficients(history, report, support), on))

    def test_bernstein_is_bounded_zero_capable_endpoint_correct_and_grid_independent(self):
        coefficients = torch.tensor([[[.2, .9, .6, .1]]], dtype=torch.double)
        times = torch.tensor([0., 5., 15., 65.], dtype=torch.double)
        loss = bernstein_loss(coefficients, times, 65.)
        self.assertEqual(float(loss[0, 0, 0]), .2)
        self.assertEqual(float(loss[0, -1, 0]), .1)
        self.assertTrue(((loss >= 0) & (loss <= .95)).all())
        self.assertTrue((bernstein_loss(torch.zeros_like(coefficients), times, 65.) == 0).all())
        torch.testing.assert_close(bernstein_loss(coefficients, times[::2], 65.), loss[:, ::2], atol=0, rtol=0)
        monotone = torch.tensor([[[.9, .7, .3, .1]]], dtype=torch.double)
        self.assertTrue((torch.diff(bernstein_loss(monotone, torch.arange(66, dtype=torch.double), 65.), dim=1) <= 0).all())

    def test_four_coefficients_receive_flow_gradient_and_match_finite_differences(self):
        topology, rate, weights, times, cap, boundary, initial = chain_fixture(nodes=2, steps=4)
        initial[0, :, 0] = torch.tensor([.8, .1])
        coefficients = torch.full((1, 3, 4), .75, dtype=torch.double, requires_grad=True)
        model = CapacityLimitedExchange(topology, rate, rate)
        def response(c):
            capacities = edge_capacities(c, times[:-1], float(times[-1]), topology, rate, weights)
            return model(initial, capacities, boundary, times, weights)['edge_flux'][:, :, 1]
        response(coefficients).sum().backward()
        self.assertTrue((coefficients.grad[0, 1].abs() > 0).all())
        self.assertTrue(torch.autograd.gradcheck(response, (coefficients.detach().requires_grad_(True),), fast_mode=True))

    def test_explicit_target_windows_preserve_original_clock_and_integrated_balance(self):
        topology, rate, weights, times, cap, boundary, initial = chain_fixture(channels=4)
        result = CapacityLimitedExchange(topology, rate, rate)(initial, cap, boundary, times, weights)
        starts = torch.arange(5., 65., 5., dtype=torch.double)
        windows = torch.stack((starts, starts+5), -1)
        features = target_window_features(result, windows)
        self.assertEqual(features.shape, (1, 12, 4, 16))
        self.assertTrue(torch.equal(features[:, 0, :, :4], result['states'][:, 8]))  # first Y ends at +10
        self.assertTrue(torch.equal(features[:, -1, :, :4], result['states'][:, 52]))  # last Y ends at +65
        torch.testing.assert_close(features[..., 12:], features[..., 4:8]-features[..., 8:12], atol=2e-16, rtol=1e-12)
        with self.assertRaises(ValueError):
            target_window_features(result, torch.tensor([[5.1, 10.]], dtype=torch.double))

    def test_ineligible_nodes_are_preserved_and_fusion_features_are_zero(self):
        topology = graph(4, [(0, 1), (1, 2)], mask=[True, True, True, False])
        rate = torch.ones(4, 1, dtype=torch.double)
        initial = torch.full((1, 4, 1), .4, dtype=torch.double)
        cap = torch.ones(1, 4, 2, 1, dtype=torch.double)
        times, weights = torch.arange(5, dtype=torch.double)*1.25, torch.ones(2, dtype=torch.double)
        for cls in (CapacityLimitedExchange, OrdinaryDirectedRecurrence):
            model = cls(copy.deepcopy(topology), rate, rate).double()
            result = model(initial, cap, torch.zeros_like(cap), times, weights)
            self.assertTrue((result['states'][:, :, 3] == .4).all())
            self.assertTrue((target_window_features(result, torch.tensor([[0., 5.]], dtype=torch.double))[:, :, 3] == 0).all())

    def test_ordinary_control_uses_same_capacity_only_interface_and_is_not_called_conservative(self):
        topology, rate, weights, times, cap, boundary, initial = chain_fixture(channels=4)
        model = OrdinaryDirectedRecurrence(topology, rate, rate).double()
        result = model(initial, cap, boundary, times, weights)
        self.assertTrue((result['states'] >= 0).all() and (result['states'] <= 1).all())
        self.assertNotIn('balance_residual', result)
        self.assertIn('NOT_physical', result['kind'])
        loss = target_window_features(result, torch.tensor([[5., 10.]], dtype=torch.double)).square().sum()
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
                            for p in model.parameters()))

    def test_checkpoint_and_optimizer_resume_reproduce_next_update(self):
        topology, rate, weights, times, cap, boundary, initial = chain_fixture(channels=2, steps=4)
        model = OrdinaryDirectedRecurrence(topology, rate, rate).double()
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        def step(m, opt):
            opt.zero_grad(set_to_none=True)
            m(initial, cap, boundary, times, weights)['states'][:, -1].square().mean().backward()
            opt.step()
        step(model, optimizer)
        buffer = io.BytesIO()
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict()), buffer)
        buffer.seek(0)
        saved = torch.load(buffer, weights_only=True)
        restored = OrdinaryDirectedRecurrence(copy.deepcopy(topology), rate, rate).double()
        restored.load_state_dict(saved['model'])
        resumed = torch.optim.Adam(restored.parameters(), lr=.001)
        resumed.load_state_dict(saved['optimizer'])
        step(model, optimizer)
        step(restored, resumed)
        self.assertTrue(all(torch.equal(v, restored.state_dict()[k]) for k, v in model.state_dict().items()))

    def test_substep_refinement_approaches_independent_one_cell_exact_solution(self):
        topology = graph(1, [(0, -1)], boundaries=(0,))
        rate, weights = torch.ones(1, 1, dtype=torch.double), torch.ones(1, dtype=torch.double)
        initial = torch.tensor([[[.4]]], dtype=torch.double)
        errors = []
        for substeps in (4, 8, 16):
            times = torch.arange(substeps+1, dtype=torch.double)*5/substeps
            cap = torch.ones(1, substeps, 1, 1, dtype=torch.double)
            result = CapacityLimitedExchange(topology, rate, rate)(initial, cap, torch.zeros_like(cap), times, weights)
            errors.append(abs(float(result['states'][0, -1, 0, 0])-.4*math.exp(-1)))
        self.assertGreater(errors[0], errors[1])
        self.assertGreater(errors[1], errors[2])

    def test_graph_weight_step_and_boundary_contract_violations_fail_closed(self):
        invalid = [([(0, 0)], ()), ([(0, 1), (0, 1)], ()), ([(-1, -1)], ()),
                   ([(0, 3)], ()), ([(-1, 0)], ())]
        for edges, boundaries in invalid:
            with self.subTest(edges=edges), self.assertRaises(ValueError):
                graph(2, edges, boundaries=boundaries)
        with self.assertRaises(ValueError):
            graph(2, [(0, 1)], mask=[True, False])
        topology, rate, weights, times, cap, boundary, initial = chain_fixture(steps=1)
        model = CapacityLimitedExchange(topology, rate*2, rate*2)
        with self.assertRaises(ValueError):
            model(initial, cap, boundary, torch.tensor([0., 5.], dtype=torch.double), weights)
        with self.assertRaises(ValueError):
            model(initial, cap, boundary, times, weights*.5)
        bad = boundary.clone()
        bad[:, :, 1] = .1
        with self.assertRaises(ValueError):
            model(initial, cap, bad, times, weights)
        with self.assertRaises(ValueError):
            condition_coefficients(torch.zeros(1, 2, 4), torch.zeros(1, 2, 4), torch.full((1, 2), 1.1))

    def test_empty_graph_keeps_all_nodes_and_has_zero_exchange(self):
        topology = graph(2, [], mask=[False, False])
        rate = torch.ones(2, 1, dtype=torch.double)
        initial = torch.tensor([[[.2], [.7]]], dtype=torch.double)
        cap = torch.empty(1, 1, 0, 1, dtype=torch.double)
        result = CapacityLimitedExchange(topology, rate, rate)(initial, cap, cap, torch.tensor([0., 1.25], dtype=torch.double),
                                                               torch.empty(0, dtype=torch.double))
        self.assertTrue(torch.equal(result['states'][:, 0], result['states'][:, 1]))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_matches_cpu_and_supports_deterministic_backward(self):
        outputs = []
        previous = torch.are_deterministic_algorithms_enabled()
        try:
            torch.use_deterministic_algorithms(True)
            for device in ('cpu', 'cuda'):
                topology, rate, weights, times, cap, boundary, initial = chain_fixture(channels=4, steps=12, device=device, dtype=torch.float32)
                cap[:, :, 2] = .15
                cap.requires_grad_(True)
                result = CapacityLimitedExchange(topology, rate, rate)(initial, cap, boundary, times, weights)
                result['edge_flux'][:, :, 2].sum().backward()
                self.assertTrue(torch.isfinite(cap.grad).all() and cap.grad.abs().sum() > 0)
                outputs.append(result['states'].detach().cpu())
            torch.testing.assert_close(outputs[0], outputs[1], atol=2e-6, rtol=2e-6)
        finally:
            torch.use_deterministic_algorithms(previous)

if __name__ == '__main__':
    unittest.main()
