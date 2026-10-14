"""Analytic open-ledger, coupled constraints and capacity mechanism tests."""
import unittest

import torch

from src.models.open_stock_flow import SCHEMA, step


def node(values, *, dtype=torch.float64, requires_grad=False):
    return torch.tensor(values, dtype=dtype).reshape(1, -1, 1).requires_grad_(requires_grad)


def edges(pairs):
    return torch.tensor(pairs, dtype=torch.long).reshape(-1, 2).T.contiguous()


def run(stock, limit, service, demand, pairs, backlog=None, dt=1.):
    demand = node(demand)
    return step(node(stock), node(limit), node(service), demand,
                torch.zeros_like(demand) if backlog is None else node(backlog), edges(pairs), dt)


class OpenStockFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def assert_no_violation(self, record, atol=1e-12):
        for name, value in record['stats'].items():
            if name.endswith('violation_max') or name.endswith('balance_abs_max'):
                self.assertLessEqual(float(value), atol, name)
                self.assertFalse(value.requires_grad)

    def test_one_internal_edge_has_one_shared_flux_and_exact_ledger(self):
        r = run([6, 0], [10, 10], [4, 10], [8], [(0, 1)])
        self.assertEqual(r['schema'], SCHEMA)
        torch.testing.assert_close(r['flux'], node([4]))
        torch.testing.assert_close(r['stock'], node([2, 4]))
        torch.testing.assert_close(r['inflow'], node([0, 4]))
        torch.testing.assert_close(r['outflow'], node([4, 0]))
        self.assertIs(r['stock'], r['stock_next'])
        self.assert_no_violation(r)

    def test_capacity_matched_state_perturbation_changes_final_flow_when_binding(self):
        high = run([6, 0], [10, 10], [4, 10], [8], [(0, 1)])
        low = run([6, 0], [10, 10], [2, 10], [8], [(0, 1)])
        torch.testing.assert_close(high['flux']-low['flux'], node([2]))
        torch.testing.assert_close(low['stock']-high['stock'], node([2, -2]))
        self.assertEqual(float(low['stats']['service_active_count']), 1)

    def test_low_demand_correctly_has_no_capacity_response(self):
        high = run([6, 0], [10, 10], [4, 10], [1], [(0, 1)])
        low = run([6, 0], [10, 10], [2, 10], [1], [(0, 1)])
        torch.testing.assert_close(high['flux'], low['flux'], atol=0, rtol=0)
        self.assertEqual(float(low['stats']['service_active_count']), 0)

    def test_merge_all_sources_share_same_old_destination_room(self):
        r = run([3, 4, 8], [10, 10, 10], [10, 10, 10], [3, 4], [(0, 2), (1, 2)])
        torch.testing.assert_close(r['flux'], node([6/7, 8/7]))
        torch.testing.assert_close(r['stock'], node([15/7, 20/7, 10]))
        self.assertEqual(float(r['stats']['receiving_limited_count']), 1)
        self.assert_no_violation(r)

    def test_diverge_source_service_shared_and_receiving_can_reduce_final_activity(self):
        r = run([8, 0, 9], [10, 10, 10], [6, 10, 10], [8, 4], [(0, 1), (0, 2)])
        torch.testing.assert_close(r['source_allocated_edge_rate'], node([4, 2]))
        torch.testing.assert_close(r['flux'], node([4, 1]))
        torch.testing.assert_close(r['stock'], node([3, 4, 10]))
        self.assertEqual(float(r['stats']['service_limited_count']), 1)
        self.assertEqual(float(r['stats']['service_active_count']), 0)
        self.assert_no_violation(r)

    def test_stock_bound_has_priority_when_it_is_tighter_than_service(self):
        r = run([2, 0], [10, 10], [10, 10], [8], [(0, 1)], dt=.5)
        torch.testing.assert_close(r['flux'], node([4]))
        torch.testing.assert_close(r['stock'], node([0, 2]))
        self.assertEqual(float(r['stats']['stock_limited_count']), 1)
        self.assert_no_violation(r)

    def test_entry_and_exit_use_old_stock_and_blocked_external_demand_is_retained(self):
        r = run([4], [5], [3], [2, 5], [(-1, 0), (0, -1)], backlog=[3, 0])
        torch.testing.assert_close(r['flux'], node([1, 3]))
        torch.testing.assert_close(r['stock'], node([2]))
        torch.testing.assert_close(r['external_backlog'], node([4, 0]))
        torch.testing.assert_close(r['outflow'], node([3]))
        self.assert_no_violation(r)
        later = step(r['stock'], node([5]), node([3]), node([0, 0]),
                     r['external_backlog'], edges([(-1, 0), (0, -1)]), 1.)
        torch.testing.assert_close(later['flux'], node([3, 0]))
        torch.testing.assert_close(later['external_backlog'], node([1, 0]))
        self.assert_no_violation(later)

    def test_internal_and_external_entries_share_destination_room(self):
        r = run([5, 8], [10, 10], [10, 10], [3, 5], [(0, 1), (-1, 1)], backlog=[0, 2])
        torch.testing.assert_close(r['flux'], node([.6, 1.4]))
        torch.testing.assert_close(r['stock'], node([4.4, 10]))
        torch.testing.assert_close(r['external_backlog'], node([0, 5.6]))
        self.assert_no_violation(r)

    def test_empty_edges_and_isolated_nodes_have_zero_outflow_without_fallback(self):
        r = run([2, 3], [5, 5], [4, 4], [], [])
        torch.testing.assert_close(r['stock'], node([2, 3]), atol=0, rtol=0)
        torch.testing.assert_close(r['outflow'], node([0, 0]), atol=0, rtol=0)
        self.assertEqual(r['flux'].shape, (1, 0, 1))
        self.assert_no_violation(r)

    def test_service_gradient_matches_finite_difference_and_nonbinding_is_zero(self):
        n, k, d, graph = node([6, 0]), node([10, 10]), node([8]), edges([(0, 1)])
        c = node([2, 10], requires_grad=True)
        r = step(n, k, c, d, torch.zeros_like(d), graph, 1.)
        r['outflow'].sum().backward()
        torch.testing.assert_close(c.grad, node([1, 0]))
        eps = 1e-5
        plus = step(n, k, node([2+eps, 10]), d, torch.zeros_like(d), graph, 1.)['outflow'].sum()
        minus = step(n, k, node([2-eps, 10]), d, torch.zeros_like(d), graph, 1.)['outflow'].sum()
        self.assertAlmostEqual(float((plus-minus)/(2*eps)), float(c.grad.sum()), places=9)
        c2 = node([2, 10], requires_grad=True)
        step(n, k, c2, node([.5]), node([0]), graph, 1.)['outflow'].sum().backward()
        torch.testing.assert_close(c2.grad, node([0, 0]), atol=0, rtol=0)

    def test_receiving_limit_and_backlog_keep_autograd(self):
        n = node([8], requires_grad=True)
        k = node([10], requires_grad=True)
        d = node([5], requires_grad=True)
        backlog = node([3], requires_grad=True)
        r = step(n, k, node([10]), d, backlog, edges([(-1, 0)]), 1.)
        self.assertTrue(r['external_backlog'].requires_grad)
        r['flux'].sum().backward(retain_graph=True)
        torch.testing.assert_close(n.grad, node([-1]))
        torch.testing.assert_close(k.grad, node([1]))
        torch.testing.assert_close(d.grad, node([0]), atol=1e-15, rtol=0)
        torch.testing.assert_close(backlog.grad, node([0]), atol=1e-15, rtol=0)

    def test_subnormal_softplus_demands_and_zero_limits_have_finite_gradients(self):
        # Real 496-station engineering weights can request subnormal rates.
        # Unused limit/request divisions previously produced inf and NaN grads.
        from torch.nn import functional as F
        for service, destination_stock in ((2., 0.), (0., 0.), (2., 1.)):
            with self.subTest(service=service, destination_stock=destination_stock):
                n = node([1., destination_stock], dtype=torch.float32, requires_grad=True)
                k = node([2., 1.], dtype=torch.float32, requires_grad=True)
                c = node([service, 2.], dtype=torch.float32, requires_grad=True)
                logits = node([-100.], dtype=torch.float32, requires_grad=True)
                demand = F.softplus(logits)
                self.assertGreater(float(demand.detach()), 0.)
                self.assertLess(float(demand.detach()), torch.finfo(torch.float32).tiny)
                r = step(n, k, c, demand, torch.zeros_like(demand), edges([(0, 1)]), .5)
                loss = r['outflow'].sum()+r['stock'].square().sum()
                loss.backward()
                for value in (n, k, c, logits):
                    self.assertIsNotNone(value.grad)
                    self.assertTrue(bool(torch.isfinite(value.grad).all()))
                self.assert_no_violation(r, atol=1e-6)

    def test_subnormal_external_arrivals_at_full_storage_keep_finite_gradients(self):
        from torch.nn import functional as F
        n = node([1.], dtype=torch.float32, requires_grad=True)
        k = node([1.], dtype=torch.float32, requires_grad=True)
        c = node([0.], dtype=torch.float32, requires_grad=True)
        logits = node([-100.], dtype=torch.float32, requires_grad=True)
        backlog = node([0.], dtype=torch.float32, requires_grad=True)
        r = step(n, k, c, F.softplus(logits), backlog, edges([(-1, 0)]), .5)
        (r['inflow'].sum()+r['external_backlog'].sum()+r['stock'].sum()).backward()
        for value in (n, k, c, logits, backlog):
            self.assertIsNotNone(value.grad)
            self.assertTrue(bool(torch.isfinite(value.grad).all()))
        torch.testing.assert_close(r['flux'], node([0.], dtype=torch.float32), atol=0, rtol=0)
        self.assertGreater(float(r['external_backlog'].detach().sum()), 0.)

    def test_edge_reordering_preserves_all_node_results_and_reorders_flux(self):
        graph = edges([(-1, 0), (0, 1), (0, 2), (1, 2), (2, -1)])
        n, k, c = node([5, 6, 2]), node([10, 10, 10]), node([3, 4, 5])
        d, b = node([3, 4, 6, 2, 8]), node([2, 0, 0, 0, 0])
        a = step(n, k, c, d, b, graph, .25)
        perm = torch.tensor([4, 2, 0, 3, 1])
        z = step(n, k, c, d[:, perm], b[:, perm], graph[:, perm], .25)
        for name in ('stock', 'inflow', 'outflow'):
            torch.testing.assert_close(a[name], z[name], atol=1e-14, rtol=1e-14)
        torch.testing.assert_close(a['flux'][:, perm], z['flux'])
        self.assert_no_violation(a)
        self.assert_no_violation(z)

    def test_closed_network_multiple_steps_preserves_total_mass_and_bounds(self):
        torch.manual_seed(104)
        graph = edges([(0, 1), (0, 2), (1, 2), (2, 3), (3, 0), (3, 1)])
        n, k, c = torch.rand(3, 4, 1, dtype=torch.float64)*10, torch.full((3, 4, 1), 10., dtype=torch.float64), torch.ones(3, 4, 1, dtype=torch.float64)*6
        original_mass = n.sum(1)
        for _ in range(100):
            d = torch.rand(3, graph.shape[1], 1, dtype=torch.float64)*12
            r = step(n, k, c, d, torch.zeros_like(d), graph, .5)
            n = r['stock']
            self.assertTrue(bool(((n >= 0) & (n <= k)).all()))
            self.assert_no_violation(r)
        torch.testing.assert_close(n.sum(1), original_mass, atol=1e-12, rtol=1e-12)

    def test_float32_open_long_run_keeps_strict_input_bounds_without_stock_clipping(self):
        # This seed used to fail on a subsequent call from a negative backlog
        # ulp, or stock just above K, after an exactly saturated allocation.
        torch.manual_seed(1)
        n = torch.rand(16, 20, 1)*10
        k, c = torch.full_like(n, 10.), torch.rand_like(n)*30
        graph = torch.randint(0, 20, (2, 80))
        graph[0, :20], graph[1, 20:40] = -1, -1
        backlog = torch.zeros(16, 80, 1)
        for _ in range(200):
            d = torch.rand_like(backlog)*30
            r = step(n, k, c, d, backlog, graph, .37)
            # Stored stock is the raw ledger update, not a repaired value.
            torch.testing.assert_close(r['stock'], n+.37*(r['inflow']-r['outflow']), atol=0, rtol=0)
            n, backlog = r['stock'], r['external_backlog']
            self.assertTrue(bool(((n >= 0) & (n <= k)).all()))
            self.assertTrue(bool((backlog >= 0).all()))
            self.assertLessEqual(float(r['stats']['sending_violation_max']), 1e-5)
            self.assertLessEqual(float(r['stats']['receiving_violation_max']), 1e-5)
        self.assertGreater(float(r['stats']['allocation_relative_margin']), 0)

    def test_input_tensors_are_preserved_and_batch_samples_do_not_interact(self):
        n, k, c = node([3, 2]).repeat(2, 1, 1), node([10, 10]).repeat(2, 1, 1), node([2, 3]).repeat(2, 1, 1)
        d, backlog = node([5, 4]).repeat(2, 1, 1), node([0, 2]).repeat(2, 1, 1)
        graph = edges([(0, 1), (-1, 0)])
        original = [v.clone() for v in (n, k, c, d, backlog, graph)]
        whole = step(n, k, c, d, backlog, graph, .5)
        for actual, expected in zip((n, k, c, d, backlog, graph), original):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        single = step(n[:1], k[:1], c[:1], d[:1], backlog[:1], graph, .5)
        for key in ('stock', 'flux', 'external_backlog'):
            torch.testing.assert_close(whole[key][:1], single[key], atol=0, rtol=0)

    def test_rejects_bad_axes_dtype_endpoints_backlog_and_values(self):
        inputs = [node([2, 1]), node([5, 5]), node([3, 3]), node([2]), node([0]), edges([(0, 1)]), 1.]
        changes = [(0, node([-1, 1])), (1, node([1, 5])), (2, node([float('nan'), 3])),
                   (3, node([float('inf')])), (4, node([1])), (5, edges([(-1, -1)])),
                   (5, edges([(0, 2)])), (5, edges([(-2, 0)])), (5, edges([(0, 1)]).float()),
                   (6, 0.), (6, -1.), (6, float('inf')), (6, True),
                   (2, node([3, 3], dtype=torch.float32)), (3, node([2, 3])),
                   (0, torch.ones(2, 2, dtype=torch.float64))]
        for index, replacement in changes:
            with self.subTest(index=index, replacement=str(replacement)):
                args = list(inputs)
                args[index] = replacement
                with self.assertRaises(ValueError):
                    step(*args)


if __name__ == '__main__':
    unittest.main()
