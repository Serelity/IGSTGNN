import copy
import unittest

import torch

from src.models.incident_capacity_exchange import ExchangeGraph
from src.models.incident_spatial import configure_spatial_mode, exchange_step, SpatialFeatureExchange
from tests.test_acdg import make_model, incident_batch


def graph(nodes=3, pairs=((-1, 0), (0, 1), (1, 2), (2, -1))):
    edges = torch.tensor(pairs, dtype=torch.long).reshape(-1, 2).T
    return ExchangeGraph(nodes, edges, torch.ones(nodes, dtype=torch.bool),
                         torch.ones(nodes, dtype=torch.bool), evidence_scope='synthetic_only')


class IncidentSpatialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_equal_in_out_changes_throughput_with_zero_displacement(self):
        g = graph()
        x = torch.full((1, 3, 1), .5, dtype=torch.double)
        q = torch.full((1, 4, 1), .45, dtype=torch.double)
        records = [exchange_step(x, q, torch.full_like(q, c), g, restricted=True) for c in (.4, .2)]
        for record in records:
            torch.testing.assert_close(record['state'], x, atol=1e-15, rtol=0)
            torch.testing.assert_close(record['balance'], torch.zeros(1, 1, dtype=torch.double))
        torch.testing.assert_close(records[0]['inflow'], 2*records[1]['inflow'])
        torch.testing.assert_close(records[0]['outflow'], 2*records[1]['outflow'])

    def test_split_shares_sending_and_merge_shares_receiving_independent_hand_solution(self):
        # 0 -> 1, 0 -> external share .2; 0 -> 1 and external -> 1 share .1.
        g = graph(2, ((0, 1), (0, -1), (-1, 1)))
        x = torch.tensor([[[.2], [.9]]], dtype=torch.double)
        q = torch.full((1, 3, 1), .8, dtype=torch.double)
        record = exchange_step(x, q, torch.ones_like(q), g, restricted=True, eta=1.)
        expected = torch.tensor([[[.1/9], [.1], [.8/9]]], dtype=torch.double)
        torch.testing.assert_close(record['flux'], expected)
        torch.testing.assert_close(record['state'], torch.tensor([[[.2-.1-.1/9], [1.]]], dtype=torch.double))
        torch.testing.assert_close(record['balance'], torch.zeros(1, 1, dtype=torch.double), atol=1e-15, rtol=0)

    def test_closed_zero_empty_isolated_and_reordered_graphs(self):
        for g in (graph(3, ()), graph(3, ((0, 1), (1, 2), (2, 0)))):
            for value in (0., .4, 1.):
                x = torch.full((2, 3, 2), value, dtype=torch.double)
                q = torch.full((2, g.edges, 2), .7, dtype=torch.double)
                r = exchange_step(x, q, torch.ones_like(q), g, restricted=True)
                torch.testing.assert_close(r['balance'], torch.zeros(2, 2, dtype=torch.double), atol=1e-14, rtol=0)
                self.assertTrue(((r['state'] >= 0) & (r['state'] <= 1)).all())
        g = graph()
        x = torch.rand(2, 3, 2, dtype=torch.double)
        q, c = torch.rand(2, 4, 2, dtype=torch.double), torch.rand(2, 4, 2, dtype=torch.double)
        order = torch.tensor([2, 0, 3, 1])
        reordered = graph(3, g.edge_index[:, order].T.tolist())
        a = exchange_step(x, q, c, g, restricted=True)
        b = exchange_step(x, q[:, order], c[:, order], reordered, restricted=True)
        torch.testing.assert_close(a['state'], b['state'])
        node_order = torch.tensor([2, 0, 1])
        inverse = node_order.argsort()
        edges = torch.where(g.edge_index >= 0, inverse[g.edge_index.clamp_min(0)], -1)
        reordered = graph(3, edges.T.tolist())
        b = exchange_step(x[:, node_order], q, c, reordered, restricted=True)
        torch.testing.assert_close(a['state'][:, node_order], b['state'])
        empty = SpatialFeatureExchange(8, restricted=True)
        context = dict(graph=graph(3, ()), capacity=torch.empty(2, 0, 4), boundary=torch.empty(2, 0, 4))
        torch.testing.assert_close(empty(torch.rand(2, 3, 3, 8), context), torch.zeros(2, 3, 3, 8))

    def test_b2_clipping_and_flux_gradient_are_separate(self):
        g = graph(2, ((0, 1), (0, -1)))
        q = torch.tensor([[[.9], [.9]]], requires_grad=True)
        r = exchange_step(torch.tensor([[[.1], [.9]]]), q, torch.ones_like(q), g,
                          restricted=False, eta=1.)
        self.assertGreater(float(r['clip'].detach().abs().sum()), 0)
        state_grad = torch.autograd.grad(r['state'].sum(), q, retain_graph=True)[0]
        flux_grad = torch.autograd.grad(r['inflow'].sum()+r['outflow'].sum(), q)[0]
        torch.testing.assert_close(state_grad, torch.zeros_like(q))
        self.assertTrue((flux_grad > 0).all())

    def test_native_b1_identity_and_b2_b3_parameter_pairing(self):
        torch.manual_seed(12)
        native = make_model().eval()
        gated = configure_spatial_mode(copy.deepcopy(native), 'strength').eval()
        x, report = torch.rand(2, 12, 3, 3), incident_batch()
        with torch.inference_mode():
            torch.testing.assert_close(native(x, incident_data=report), gated(x, incident_data=report), atol=0, rtol=0)
        self.assertTrue(all(p.requires_grad for p in gated.parameters()))
        for k, v in native.state_dict().items():
            torch.testing.assert_close(gated.state_dict()[k], v, atol=0, rtol=0)
        torch.manual_seed(99)
        ordinary = configure_spatial_mode(copy.deepcopy(native), 'state_ordinary', graph())
        torch.manual_seed(99)
        restricted = configure_spatial_mode(copy.deepcopy(native), 'state_restricted', graph())
        self.assertEqual(set(dict(ordinary.named_parameters())), set(dict(restricted.named_parameters())))
        for k, v in ordinary.state_dict().items():
            torch.testing.assert_close(restricted.state_dict()[k], v, atol=0, rtol=0)
        self.assertFalse(hasattr(restricted.icsf_module, 'q_proj'))

    def test_report_access_and_all_spatial_calls_and_main_loss_gradients(self):
        torch.manual_seed(15)
        model = configure_spatial_mode(make_model(), 'state_restricted', graph()).eval()
        x, report = torch.rand(2, 12, 3, 3), incident_batch()
        on = model(x, incident_data=report, return_spatial_details=True)
        off = model(x, incident_data=report, spatial_report_enabled=False, return_spatial_details=True)
        self.assertEqual(len(on['spatial']['records']), 2*4)  # two fixture layers, four calls each
        torch.testing.assert_close(on['spatial']['prior'], off['spatial']['prior'], atol=0, rtol=0)
        torch.testing.assert_close(on['spatial']['boundary'], off['spatial']['boundary'], atol=0, rtol=0)
        torch.testing.assert_close(off['spatial']['posterior'], off['spatial']['prior'], atol=0, rtol=0)
        mask = on['spatial']['report_mask'].expand_as(on['spatial']['prior']).bool()
        torch.testing.assert_close(on['spatial']['posterior'][~mask], on['spatial']['prior'][~mask], atol=0, rtol=0)
        on['prediction'].square().mean().backward()
        for group in ('prior', 'proposal', 'adoption'):
            grads = [p.grad for p in getattr(model.report_spatial_state, group).parameters()]
            self.assertTrue(all(v is not None and torch.isfinite(v).all() for v in grads))
            self.assertGreater(sum(float(v.abs().sum()) for v in grads), 0.)
        all_off = model(x, incident_data=report, spatial_report_enabled=False, tiid_report_enabled=False)
        self.assertGreater(float((off['prediction']-all_off).abs().max()), 0.)
        with self.assertRaisesRegex(ValueError, 'external forecast'):
            model(x, incident_data=report, forecast_delta=torch.zeros(2, 12, 3, 256))
        with self.assertRaisesRegex(ValueError, 'cutoff clocks'):
            model(x)
        again = model(x, incident_data=report)
        torch.testing.assert_close(on['prediction'], again, atol=0, rtol=0)
        self.assertFalse(any('context' in k for k in model.__dict__))

    def test_fixed_proposal_capacity_gradient_and_readout_detects_throughput(self):
        g = graph()
        x, q = torch.full((1, 3, 1), .5), torch.full((1, 4, 1), .45)
        c = torch.full_like(q, .2, requires_grad=True)
        r = exchange_step(x, q, c, g, restricted=True)
        self.assertGreater(float(torch.autograd.grad(r['inflow'].sum(), c)[0].abs().sum()), 0.)
        module = SpatialFeatureExchange(8, restricted=True)
        self.assertEqual(module.readout.in_features, 12)
        self.assertIsNone(module.readout.bias)
        self.assertGreater(float(module.readout.weight[:, 4:].detach().abs().sum()), 0.)

    def test_invalid_configuration_leaves_native_unchanged(self):
        model = make_model()
        with self.assertRaises(ValueError):
            configure_spatial_mode(model, 'state_restricted')
        self.assertFalse(hasattr(model, '_internal_spatial_mode'))

    def test_main_prediction_report_gradient_when_only_hard_capacity_path_is_available(self):
        torch.manual_seed(17)
        restricted = configure_spatial_mode(make_model(), 'state_restricted', graph(3, ((0, 1), (1, 2)))).eval()
        ordinary = copy.deepcopy(restricted)
        for model in (restricted, ordinary):
            for layer in model.layers:
                exchange = layer.dif_layer.localized_st_conv.incident_exchange
                exchange.restricted = model is restricted
                with torch.no_grad():
                    exchange.encode.weight.zero_()
                    exchange.encode.bias.zero_()
                    for p in exchange.proposal.parameters():
                        p.zero_()
                    exchange.proposal[-1].bias.fill_(3.)  # fixed high q, independent of C or hidden state
        x, report = torch.rand(2, 12, 3, 3), incident_batch()
        report['distances'].fill_(.2)
        for model in (restricted, ordinary):
            on = model(x, incident_data=report, tiid_report_enabled=False)
            off = model(x, incident_data=report, tiid_report_enabled=False, spatial_report_enabled=False)
            if model is ordinary:
                torch.testing.assert_close(on, off, atol=0, rtol=0)
            else:
                self.assertGreater(float((on-off).detach().abs().max()), 0.)
                on.square().mean().backward()
                gradient = sum(float(p.grad.abs().sum()) for p in model.report_spatial_state.proposal.parameters())
                self.assertGreater(gradient, 0.)

    def test_invalid_latent_values_rejected(self):
        g = graph()
        for bad in (float('nan'), -.1):
            with self.assertRaisesRegex(ValueError, 'Invalid latent'):
                exchange_step(torch.full((1, 3, 1), .5), torch.full((1, 4, 1), bad),
                              torch.ones(1, 4, 1), g, restricted=True)


if __name__ == '__main__':
    unittest.main()
