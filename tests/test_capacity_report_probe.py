"""P1 checkpoint intervention: scope, restoration, error sign and artifacts."""
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.chronological import probe_incident_capacity_reports as probe
from experiments.chronological import train_incident_capacity as training
from src.models.incident_capacity_fusion import CapacityAugmentedIGSTGNN
from src.utils.incident_corridor import read_json
from test_acdg import incident_batch, make_model
from test_incident_capacity_fusion import fixture


class CapacityReportProbeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(451)
        torch.set_num_threads(3)
        branch, capacity = fixture()
        with torch.no_grad():
            branch.projection.weight.normal_(0, .03)
            branch.coefficients[0].weight.fill_(.01)
            branch.coefficients[0].bias.zero_()
            branch.coefficients[2].weight.fill_(.1)
            branch.coefficients[2].bias.fill_(.65)
        # One supported and one unsupported row; matched report age is known.
        capacity['reports']['weights'][1] = 0
        self.model = CapacityAugmentedIGSTGNN(make_model(), branch).eval()
        self.batch = dict(x=torch.rand(2, 12, 3, 3), incident=incident_batch(), capacity_inputs=capacity)
        self.scaler = {'mean': 10., 'std': 2.}
        graph = branch.graph
        structure = dict(station_ids=[11, 22, 33], boundary_nodes=graph.boundary_nodes.tolist(),
            edges=[dict(source=int(a), destination=int(b)) for a, b in graph.edge_index.T])
        self.events = [dict(sample_index=100+i, incident_id=str(i),
                            t0='2023-01-01T08:0'+str(i//2)+':00') for i in range(4)]

        def get_batch(split, selected, device, *, new_reports=True):
            self.assertEqual(split, 'val')
            self.assertTrue(new_reports)
            ix = [int(i) % 2 for i in selected]
            def select(value):
                return value[ix].clone().to(device)
            cap = self.batch['capacity_inputs']
            return dict(x=select(self.batch['x']), incident={k: select(v) for k, v in self.batch['incident'].items()},
                capacity_inputs=dict(history=select(cap['history']), valid=select(cap['valid']),
                    references=cap['references'].to(device), labels=cap['labels'].to(device),
                    reports={k: select(v) for k, v in cap['reports'].items()}))

        def targets(split, selected, device):
            self.assertEqual(split, 'val')
            target = torch.full((len(selected), 12, 3, 1), 2., device=device)
            return target, torch.ones_like(target, dtype=torch.bool)
        self.inputs = SimpleNamespace(batch=get_batch, targets=targets, events={'val': self.events},
            structure_mask=np.ones(3, bool), common_mask=np.array([True, True, False]),
            road_groups={'4': np.ones(3, bool)}, network=SimpleNamespace(structure=structure))
        self.indices = list(range(4))
        self.metrics, self.arrays = training.evaluate(self.model, self.inputs, 'P1', self.indices, 2, 'cpu', self.scaler)
        self.regions = dict(all_nodes=np.ones(3, bool), common_structure=np.array([True, True, False]),
                            added_structure=np.array([False, False, True]), outside_candidate=np.zeros(3, bool))
        self.scratch = training.REPO/'experiments/chronological_runs'
        self.scratch.mkdir(exist_ok=True)

    def run_probe(self, output, arrays=None, metrics=None, batch_size=2):
        return probe.run_probe(self.model, self.inputs, self.indices, 2, batch_size, 'cpu', self.scaler,
                               self.arrays if arrays is None else arrays,
                               self.metrics if metrics is None else metrics, self.regions, output)

    def test_off_equals_independent_report_removal_and_restores_every_tensor(self):
        before = training.state_digest(self.model.state_dict())
        native = probe.tensor_tree_digest(self.batch['incident'])
        with torch.inference_mode():
            on = probe.forward(self.model, self.batch, self.scaler, True)
            off = probe.forward(self.model, self.batch, self.scaler, False)
            removed = copy.deepcopy(self.batch)
            removed['capacity_inputs']['reports'] = {k: v[:, :0] for k, v in
                                                       removed['capacity_inputs']['reports'].items()}
            independent = probe.forward(self.model, removed, self.scaler, True)
            restored = probe.forward(self.model, self.batch, self.scaler, True)
            probe.verify_intervention(on, off, restored, self.model.branch.graph)
        for name in ('prediction', 'coefficients', 'capacity', 'forecast_delta', 'boundary_demand', 'initial'):
            torch.testing.assert_close(off[name], independent[name], atol=0, rtol=0)
        self.assertGreater(float((on['coefficients'][0]-off['coefficients'][0]).abs().max()), 0)
        torch.testing.assert_close(on['prediction'][1], off['prediction'][1], atol=0, rtol=0)
        self.assertEqual(before, training.state_digest(self.model.state_dict()))
        self.assertEqual(native, probe.tensor_tree_digest(self.batch['incident']))

    def test_complete_probe_writes_paired_artifacts_and_keeps_state_and_gradients(self):
        before = training.state_digest(self.model.state_dict())
        with tempfile.TemporaryDirectory(dir=self.scratch) as tmp:
            root = Path(tmp)
            with patch('torch.optim.Adam', side_effect=AssertionError('Optimizer must not be constructed')):
                report = self.run_probe(root, batch_size=1)
            self.assertEqual(report['status'], 'P1_REPORT_CAPACITY_PROBE_PASS')
            self.assertEqual(report['coverage']['rows_with_matched_reports'], 2)
            self.assertEqual(report['coverage']['matched_report_row_occurrences'], 2)
            self.assertEqual(report['distinct_cutoffs'], 2)
            self.assertIsNone(report['metrics']['outside_candidate'])
            self.assertEqual(report['metrics']['no_report_supported_samples']['prediction_abs_change_max'], 0)
            self.assertFalse(report['test_accessed'])
            self.assertEqual(report['optimizer_updates'], 0)
            with np.load(root/'paired_predictions.npz', allow_pickle=False) as paired:
                self.assertEqual(paired['on_prediction'].shape, (4, 12, 3, 1))
                self.assertEqual(paired['sample_indices'].tolist(), [100, 101, 102, 103])
                self.assertEqual(paired['target_windows_minutes'][0].tolist(), [5., 10.])
                self.assertEqual(paired['target_windows_minutes'][-1].tolist(), [60., 65.])
                self.assertEqual(paired['unique_cutoff_weights'].tolist(), [.5]*4)
                expected = probe.paired_metric(paired['on_prediction'], paired['off_prediction'],
                    paired['target'], paired['valid'], np.ones(3, bool))
                self.assertEqual(report['metrics']['all_nodes'], expected)
            with np.load(root/'pathway_curves.npz', allow_pickle=False) as curves:
                self.assertEqual(curves['capacity_on_mean'].shape, (52, 4))
                self.assertEqual(curves['states_on_mean'].shape, (53, 3))
                np.testing.assert_allclose(curves['capacity_off_minus_on_mean'],
                                           curves['capacity_off_mean']-curves['capacity_on_mean'], atol=1e-7)
            self.assertTrue((root/'samples.csv').is_file())
            self.assertTrue(read_json(root/'replay.json')['matching'])
        self.assertEqual(before, training.state_digest(self.model.state_dict()))
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))

    def test_failed_saved_on_replay_blocks_every_off_call(self):
        arrays = dict(self.arrays, prediction=self.arrays['prediction']+1)
        with tempfile.TemporaryDirectory(dir=self.scratch) as tmp, patch.object(probe, 'forward') as forward:
            with self.assertRaisesRegex(ValueError, 'OFF pass blocked'):
                self.run_probe(Path(tmp), arrays=arrays)
            forward.assert_not_called()
            self.assertFalse((Path(tmp)/'paired_predictions.npz').exists())

    def test_failed_saved_metric_replay_blocks_every_off_call(self):
        metrics = copy.deepcopy(self.metrics)
        metrics['all_nodes']['mae_macro'] += 1
        with tempfile.TemporaryDirectory(dir=self.scratch) as tmp, patch.object(probe, 'forward') as forward:
            with self.assertRaisesRegex(ValueError, 'metric replay failed'):
                self.run_probe(Path(tmp), metrics=metrics)
            forward.assert_not_called()

    def test_wrong_target_validity_or_order_is_rejected_before_off(self):
        for key in ('target', 'valid', 'sample_indices'):
            arrays = copy.deepcopy(self.arrays)
            arrays[key].flat[0] = not arrays[key].flat[0] if key == 'valid' else arrays[key].flat[0]+1
            with self.subTest(key=key), tempfile.TemporaryDirectory(dir=self.scratch) as tmp:
                with patch.object(probe, 'forward') as forward:
                    with self.assertRaisesRegex(ValueError, 'target/mask/order'):
                        self.run_probe(Path(tmp), arrays=arrays)
                    forward.assert_not_called()

    def test_failed_restoration_blocks_output_and_exception_preserves_state(self):
        before = training.state_digest(self.model.state_dict())
        original = probe.forward
        calls = []
        def corrupt(model, batch, scaler, enabled):
            result = original(model, batch, scaler, enabled)
            calls.append(enabled)
            if len(calls) == 3:
                result['prediction'] = result['prediction']+1
            return result
        with tempfile.TemporaryDirectory(dir=self.scratch) as tmp, patch.object(probe, 'forward', side_effect=corrupt):
            with self.assertRaisesRegex(ValueError, 'restoration failed'):
                self.run_probe(Path(tmp))
            self.assertEqual(calls, [True, False, True])
            self.assertFalse((Path(tmp)/'paired_predictions.npz').exists())
        self.assertEqual(before, training.state_digest(self.model.state_dict()))
        with tempfile.TemporaryDirectory(dir=self.scratch) as tmp, patch.object(probe, 'forward', side_effect=RuntimeError('stop')):
            with self.assertRaisesRegex(RuntimeError, 'stop'):
                self.run_probe(Path(tmp))
        self.assertEqual(before, training.state_digest(self.model.state_dict()))

    def test_rejects_training_mode_and_gradient_enabled_forward(self):
        with self.assertRaisesRegex(ValueError, 'without gradients'):
            probe.forward(self.model, self.batch, self.scaler, True)
        self.model.train()
        with tempfile.TemporaryDirectory(dir=self.scratch) as tmp:
            with self.assertRaisesRegex(ValueError, 'eval mode'):
                self.run_probe(Path(tmp))

    def test_metrics_sign_horizon_pooling_empty_regions_and_cutoff_weights(self):
        on = np.array([[[[1.], [3.]], [[2.], [99.]]], [[[5.], [99.]], [[4.], [8.]]]])
        target = np.zeros_like(on)
        off = on+2
        valid = on != 99
        mask = np.ones(2, bool)
        result = probe.paired_metric(on, off, target, valid, mask)
        self.assertEqual(result['on_mae_macro'], (9/3+14/3)/2)
        self.assertEqual(result['off_minus_on_mae'], 2)
        self.assertEqual(result['per_horizon_valid_count'], [3, 3])
        weighted = probe.paired_metric(on, off, target, valid, mask, np.array([.5, 1.]))
        self.assertAlmostEqual(weighted['on_mae_macro'], ((.5*4+5)/2+(.5*2+12)/2.5)/2)
        self.assertIsNone(probe.paired_metric(on, off, target, valid, np.zeros(2, bool)))
        invalid_horizon = valid.copy()
        invalid_horizon[:, 0] = False
        self.assertIsNone(probe.paired_metric(on, off, target, invalid_horizon, mask))
        with self.assertRaises(ValueError):
            probe.paired_metric(on, off, target, valid, np.ones((2, 2, 2), bool))

    def test_regions_use_report_inputs_and_weak_connectivity_without_external_bridges(self):
        edges = np.array([[-1, 0, 1, 2, -1, 3, 4], [0, 1, 2, -1, 3, 4, -1]])
        association = np.zeros((2, 7))
        association[0, 1] = 1
        masks = probe.input_regions(association, edges, 6)
        self.assertEqual(masks['report_associated_nodes'][0].tolist(), [True, True, False, False, False, False])
        self.assertEqual(masks['potential_propagation_only_nodes'][0].tolist(), [False, False, True, False, False, False])
        self.assertFalse(masks['report_supported_samples'][1].any())
        association[0, 0] = .5
        with self.assertRaisesRegex(ValueError, 'internal edges'):
            probe.input_regions(association, edges, 6)

    def test_streaming_curves_match_independent_full_tensor_averages(self):
        with torch.inference_mode():
            on = probe.forward(self.model, self.batch, self.scaler, True)
            off = probe.forward(self.model, self.batch, self.scaler, False)
            total = probe.PathwayTotals()
            total.update(on, off)
            summary, curves = total.result(2)
            expected = (off['capacity']-on['capacity']).abs().double().mean((0, 3)).numpy()
            np.testing.assert_allclose(curves['capacity_abs_change_mean'], expected, atol=0, rtol=0)
            self.assertEqual(summary['capacity']['positions'], on['capacity'].numel())
            support = (on['report_association'] > 0)[:, None, :, None].expand_as(on['capacity'])
            self.assertAlmostEqual(summary['direct_report_edges']['capacity']['abs_change_mean'],
                float((off['capacity']-on['capacity']).abs()[support].double().mean()))

    def test_empty_report_set_is_exact_noop_and_direct_statistics_unavailable(self):
        cap = self.batch['capacity_inputs']
        cap['reports'] = {k: v[:, :0] for k, v in cap['reports'].items()}
        with torch.inference_mode():
            on = probe.forward(self.model, self.batch, self.scaler, True)
            off = probe.forward(self.model, self.batch, self.scaler, False)
            probe.verify_intervention(on, off, on, self.model.branch.graph)
            torch.testing.assert_close(on['prediction'], off['prediction'], atol=0, rtol=0)
            count, matched, age = probe.report_support(self.batch)
            self.assertEqual(count.tolist(), [0, 0])
            self.assertTrue(np.isinf(age).all())
            total = probe.PathwayTotals()
            total.update(on, off)
            summary, _ = total.result(2)
            self.assertIsNone(summary['direct_report_edges']['capacity']['abs_change_mean'])

    def test_snapshot_keys_are_portable_and_changes_are_detected(self):
        with tempfile.TemporaryDirectory(dir=self.scratch) as tmp:
            root = Path(tmp)
            (root/'P1').mkdir()
            (root/'identity.json').write_text('{}', encoding='utf-8')
            (root/'P1/best_model.pt').write_bytes(b'first')
            before = probe.snapshot_run(root)
            self.assertIn('P1/best_model.pt', before)
            (root/'P1/best_model.pt').write_bytes(b'second')
            self.assertNotEqual(before, probe.snapshot_run(root))


if __name__ == '__main__':
    unittest.main()
