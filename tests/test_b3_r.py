from contextlib import contextmanager
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.chronological import train_b3_r as training
from experiments.chronological import train_internal_spatial as legacy
from experiments.chronological.audit_b3_r_development import audit_development
from experiments.chronological.b3_r_diagnostics import CapacityDiagnostics
from experiments.chronological.train import state_sha256
from src.models.incident_spatial import configure_spatial_mode, exchange_step
from src.models.incident_spatial_residual import (
    from_canonical_b3, node_capacity_to_edges, retained_state, tensor_state_hash,
)
from src.utils.chronological import masked_flow_mae
from tests.test_architecture_mechanisms import inputs, tiny_model
from tests.test_incident_spatial import graph
from tests.test_internal_spatial_training import Dataset


def canonical():
    return configure_spatial_mode(tiny_model().requires_grad_(True), 'state_restricted', graph(4, ((0, 1),)))


@contextmanager
def controlled_forecast():
    """Full IGSTGNN, one report bottleneck, slack budgets, active D ReLU.

    A P-output fixture fixes the endpoint ordering; D, both exchange substeps,
    throughput readout and the full forecast/loss are the actual model path.
    """
    model, _ = from_canonical_b3(canonical())
    model.eval()
    with torch.no_grad():
        model.report_spatial_state.D[0].weight.zero_()
        model.report_spatial_state.D[0].bias.fill_(1.)
        for layer in model.layers:
            exchange = layer.dif_layer.localized_st_conv.incident_exchange
            exchange.encode.weight.zero_()
            exchange.encode.bias.zero_()  # x=.5, C at source approximately .12
            for parameter in exchange.proposal.parameters():
                parameter.zero_()
            exchange.proposal[-1].bias.fill_(2.)  # q=.88; no Q(C) shortcut
            exchange.readout.weight.zero_()
            exchange.readout.weight[:, 4:].fill_(.1)  # read throughput, not just net displacement
    def capacity_fixture(module, args, output):
        return output.new_tensor([-2., 2., 2., 2.])[None, :, None].expand_as(output)
    handle = model.report_spatial_state.P.register_forward_hook(capacity_fixture)
    x, report = inputs()
    report['distances'].zero_()
    report['distances'][:, 0, 1] = .8
    try:
        yield model, x, report
    finally:
        handle.remove()


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_open_internal_empty_capacity_maps_and_gradient_bottleneck(self):
        g = graph(3)
        c = torch.tensor([[[.2], [.7], [.4]]], requires_grad=True)
        actual = node_capacity_to_edges(c, g)
        torch.testing.assert_close(actual, torch.tensor([[[.2], [.2], [.4], [.4]]]))
        actual.sum().backward()
        torch.testing.assert_close(c.grad, torch.tensor([[[2.], [0.], [2.]]]))
        self.assertEqual(node_capacity_to_edges(c, graph(3, ())).shape, (1, 0, 1))
        with self.assertRaisesRegex(ValueError, 'axes'):
            node_capacity_to_edges(c[:, :2], g)

    def test_initial_prediction_mapping_common_parameters_and_buffers(self):
        original = canonical().eval()
        before = tensor_state_hash(original.state_dict())
        model, certificate = from_canonical_b3(original)
        x, report = inputs()
        with torch.inference_mode():
            on = model(x, incident_data=report, return_spatial_details=True)
            off = model(x, incident_data=report, spatial_report_enabled=False, return_spatial_details=True)
            reference = original(x, incident_data=report, spatial_report_enabled=False)
        torch.testing.assert_close(on['prediction'], off['prediction'], atol=0, rtol=0)
        torch.testing.assert_close(on['prediction'], reference, atol=2e-6, rtol=1e-5)
        self.assertEqual(before, tensor_state_hash(original.state_dict()))
        self.assertEqual(tensor_state_hash(retained_state(original)), certificate['retained_state_sha256'])
        self.assertIn('internal_spatial_graph.edge_index', certificate['retained_state_specification'])
        self.assertEqual(sum(p.numel() for p in original.parameters())-certificate['parameter_count'],
                         sum(p.numel() for p in original.report_spatial_state.adoption.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.parameters()))
        self.assertFalse(any('.adoption.' in name or 'report_spatial_state.proposal.' in name for name, _ in model.named_parameters()))
        for k, v in retained_state(original).items():
            torch.testing.assert_close(retained_state(model)[k], v, atol=0, rtol=0)

    def test_nonzero_delta_mask_and_spatial_off_skip_d_but_preserve_tiid(self):
        model, _ = from_canonical_b3(canonical())
        model.eval()
        with torch.no_grad():
            model.report_spatial_state.D[-1].bias.fill_(.7)
        x, report = inputs()
        on = model(x, incident_data=report, return_spatial_details=True)
        off = model(x, incident_data=report, spatial_report_enabled=False, return_spatial_details=True)
        context = on['spatial']
        self.assertIsNone(off['spatial']['report_logit_delta'])
        mask = context['report_mask'].expand_as(context['node_capacity']).bool()
        torch.testing.assert_close(context['node_capacity'][~mask], context['base_node_capacity'][~mask], atol=0, rtol=0)
        self.assertGreater(float((context['node_capacity']-context['base_node_capacity']).detach()[mask].min()), 0.)
        self.assertTrue(((context['node_capacity'] >= .001) & (context['node_capacity'] <= 1)).all())
        torch.testing.assert_close(off['spatial']['node_capacity'], off['spatial']['base_node_capacity'], atol=0, rtol=0)
        with patch.object(model.report_spatial_state.D, 'forward', side_effect=AssertionError('D executed while OFF')):
            model(x, incident_data=report, spatial_report_enabled=False)
        forced_off, _ = from_canonical_b3(canonical(), report_path_enabled=False)
        with patch.object(forced_off.report_spatial_state.D, 'forward', side_effect=AssertionError('OFF training ran D')):
            prediction = forced_off(x, incident_data=report)
        no_tiid = forced_off(x, incident_data=report, tiid_report_enabled=False)
        self.assertGreater(float((prediction-no_tiid).detach().abs().max()), 0.)

    def test_full_forecast_mae_starts_d_then_hidden_task_gradients(self):
        with controlled_forecast() as (model, x, report):
            optimizer = torch.optim.SGD(model.report_spatial_state.D.parameters(), lr=.1, weight_decay=0.)
            gradients = []
            for step in range(2):
                optimizer.zero_grad(set_to_none=True)
                result = model(x, incident_data=report, return_spatial_details=True)
                prediction = result['prediction']
                loss = masked_flow_mae(prediction, prediction.detach()+1., torch.ones_like(prediction, dtype=torch.bool))
                loss.backward()
                d = model.report_spatial_state.D
                gradients.append(dict(hidden=float(d[0].weight.grad.abs().sum()+d[0].bias.grad.abs().sum()),
                                      last=float(d[-1].weight.grad.abs().sum()+d[-1].bias.grad.abs().sum())))
                self.assertGreater(gradients[-1]['last'], 0.)
                for record in result['spatial']['records']:
                    self.assertEqual(float(record[:, 12:14].sum()), 0.)  # sending/receiving slack
                optimizer.step()
            self.assertEqual(gradients[0]['hidden'], 0.)
            self.assertGreater(gradients[1]['hidden'], 0.)
            on = model(x, incident_data=report)
            off = model(x, incident_data=report, spatial_report_enabled=False)
            self.assertGreater(float((on-off).detach().abs().max()), 0.)
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.report_spatial_state.D.parameters()))

    def test_expected_zero_response_under_saturated_budget_and_nonbottleneck_endpoint(self):
        # These remain meaningful zero gradients; cap hits alone do not prove a useful signal.
        c = torch.tensor([[[.4]]], requires_grad=True)
        record = exchange_step(torch.tensor([[[.1], [.5]]]), torch.tensor([[[.9]]]), c,
                               graph(2, ((0, 1),)), restricted=True)
        torch.testing.assert_close(record['flux'], torch.tensor([[[.1]]]))
        self.assertEqual(float(torch.autograd.grad(record['flux'].sum(), c)[0].abs().sum()), 0.)
        nodes = torch.tensor([[[.3], [.8]]], requires_grad=True)
        edge = node_capacity_to_edges(nodes, graph(2, ((0, 1),)))
        torch.testing.assert_close(torch.autograd.grad(edge.sum(), nodes)[0], torch.tensor([[[1.], [0.]]]))

    def test_diagnostics_once_per_forward_exact_coverage_and_off_not_computed(self):
        model, _ = from_canonical_b3(canonical())
        model.eval()
        x, report = inputs()
        with torch.inference_mode():
            result = model(x, incident_data=report, return_spatial_details=True)
            off = model(x, incident_data=report, spatial_report_enabled=False, return_spatial_details=True)
        collector = CapacityDiagnostics()
        collector.update(result['spatial'])
        collector.update(result['spatial'])
        report_out = collector.result()
        self.assertEqual(len(report_out['rows']), 40)
        self.assertEqual(report_out['capacity_evaluations'], 2)
        self.assertEqual(report_out['capacity_statistics']['node_capacity:all_nodes']['count'], 2*2*4*4)
        self.assertEqual(report_out['capacity_statistics']['node_capacity:report_nodes']['count'], 2*2*2*4)
        self.assertFalse(any(isinstance(v, torch.Tensor) for v in collector.__dict__.values()))
        off_collector = CapacityDiagnostics()
        off_collector.update(off['spatial'])
        self.assertEqual(off_collector.result()['raw_delta_status'], 'not_computed')
        self.assertFalse(any('report_logit_delta:' in key for key in off_collector.result()['capacity_statistics']))
        result['spatial']['records'].pop()
        with self.assertRaisesRegex(ValueError, 'coverage'):
            CapacityDiagnostics().update(result['spatial'])

    def test_no_report_region_is_unavailable_and_bottlenecks_separate_open_edges(self):
        model, _ = from_canonical_b3(canonical())
        model.internal_spatial_graph = graph(4, ((-1, 0), (0, 1), (1, -1)))
        model.eval()
        x, report = inputs()
        report['distances'].zero_()
        with torch.inference_mode():
            context = model(x, incident_data=report, return_spatial_details=True)['spatial']
        collector = CapacityDiagnostics()
        collector.update(context)
        result = collector.result()
        self.assertEqual(result['capacity_statistics']['node_capacity:report_nodes']['status'], 'unavailable')
        self.assertIsNone(result['capacity_statistics']['node_capacity:report_nodes']['mean'])
        for kind in ('internal', 'inlet', 'exit'):
            self.assertEqual(result['bottlenecks'][kind]['positions'], 2*4)
            self.assertEqual(result['bottlenecks'][kind]['report_endpoint_positions'], 0)


class RunnerTests(unittest.TestCase):
    def test_isolated_loop_leaves_original_globals_intact_and_off_is_distinct_identity(self):
        original = (legacy.ARMS, legacy.build_arm, legacy.Diagnostics)
        loop = training.training_loop(['B3R'])
        self.assertEqual(original, (legacy.ARMS, legacy.build_arm, legacy.Diagnostics))
        self.assertEqual(list(loop.ARMS), ['B3R'])
        self.assertIs(loop.worker.__globals__['Diagnostics'], loop.Diagnostics)
        protocol = legacy.read_json(training.HERE/'internal_spatial_b3_r_v11.json')
        args = SimpleNamespace(protocol=training.HERE/'internal_spatial_b3_r_v11.json',
                               graph_dir=training.HERE/'internal_spatial_graph_is11', seed=2025, check=False,
                               threads=3, device='cpu', arms=['B3R'])
        on = training.make_identity(args, protocol, {}, {}, {'fixture': 'same weights'})
        args.arms = ['B3R_OFF']
        off = training.make_identity(args, protocol, {}, {}, {'fixture': 'same weights'})
        self.assertNotEqual(on, off)
        self.assertNotIn('gpu', on['runtime'])
        self.assertEqual(on['source_sha256']['experiments/chronological/train_internal_spatial.py'],
                         sha_for_legacy())
        # Numeric replay errors are tolerance-checked rather than hashed across hosts.
        initial = {'mapping': 'same exact parameters', 'canonical_spatial_off_prediction_abs_max': 1e-7}
        changed_rounding = {**initial, 'canonical_spatial_off_prediction_abs_max': 2e-7}
        self.assertEqual(training.initialization_identity(initial), training.initialization_identity(changed_rounding))
        self.assertNotEqual(training.initialization_identity(initial), training.initialization_identity({**initial, 'mapping': 'different'}))
        training.validate_architecture_protocol(protocol)
        for changed in ({'batch_size': 24}, {'readout': 'other'}, {'latent_channels': 8}):
            with self.assertRaisesRegex(ValueError, 'protocol'):
                training.validate_architecture_protocol({**protocol, **changed})

    def test_new_checkpoint_resume_matches_continuous_and_rejects_cross_mode_before_mutation(self):
        loop = training.training_loop(['B3R'])
        protocol = legacy.read_json(training.HERE/'internal_spatial_b3_r_v11.json')
        dataset, g = Dataset(), graph(4, ((0, 1),))
        identity = dict(fixture='B3R resume', batch_size=2, arm_modes={'B3R': 'on'})
        with tempfile.TemporaryDirectory(dir=training.REPO) as directory:
            roots = [Path(directory)/name for name in ('continuous', 'resumed')]
            for root in roots:
                root.mkdir()
            args = SimpleNamespace(worker='B3R', run_dir=roots[0], device='cpu', seed=2025, check=True, epochs=2)
            def candidate(*args):
                return from_canonical_b3(canonical())[0]
            with patch.object(loop, 'build_arm', side_effect=candidate):
                loop.worker(args, identity, protocol, dataset, dataset, g, {'fixture': list(range(4))})
                args.run_dir, args.epochs = roots[1], 1
                loop.worker(args, identity, protocol, dataset, dataset, g, {'fixture': list(range(4))})
                args.epochs = 2
                loop.worker(args, identity, protocol, dataset, dataset, g, {'fixture': list(range(4))})
                snapshot = (roots[1]/'B3R/last.pt').read_bytes()
                loop.worker(args, identity, protocol, dataset, dataset, g, {'fixture': list(range(4))})
                self.assertEqual(snapshot, (roots[1]/'B3R/last.pt').read_bytes())
            a, b = [torch.load(root/'B3R/last.pt', weights_only=False) for root in roots]
            self.assertEqual(a['global_updates'], b['global_updates'])
            self.assertEqual(a['scheduler_state'], b['scheduler_state'])
            self.assertEqual(a['best_metric'], b['best_metric'])
            for name in a['model_state']:
                torch.testing.assert_close(a['model_state'][name], b['model_state'][name], atol=0, rtol=0)
            for key, state in a['optimizer_state']['state'].items():
                for name, value in state.items():
                    torch.testing.assert_close(value, b['optimizer_state']['state'][key][name], atol=0, rtol=0)
            model = candidate()
            optimizer = loop.optimizer_for(model, protocol)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [1], gamma=.5)
            before = state_sha256(model.state_dict())
            with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                loop.restore(roots[1]/'B3R/last.pt', {**identity, 'arm_modes': {'B3R': 'off'}}, 'B3R', model, optimizer, scheduler)
            self.assertEqual(before, state_sha256(model.state_dict()))

    def test_audit_separates_missing_evidence_from_observed_nominal_conflict(self):
        def dataset(split, start):
            return SimpleNamespace(station_ids=np.arange(4), rows=[dict(split=split, sample_index='0', incident_id=split,
                report_time=start+'01:03:00', x_start=start+'00:00:00', x_end=start+'00:55:00',
                t0=start+'01:05:00', y_start=start+'01:10:00', y_end=start+'02:05:00')])
        train, val = dataset('train', '2023-01-01T'), dataset('val', '2023-09-01T')
        audit = audit_development(train, val, {'source_snapshot_is_initial_report_certified': False})
        self.assertFalse(audit['blocking_contract_conflict'])
        self.assertFalse(audit['online_time_audit_pass'])
        self.assertEqual(audit['checks']['report_time_semantics']['status'], 'missing')
        conflict = audit_development(train, train, {})
        self.assertTrue(conflict['blocking_contract_conflict'])


def sha_for_legacy():
    from experiments.chronological.smoke import sha256
    return sha256(Path(legacy.__file__))


if __name__ == '__main__':
    unittest.main()
