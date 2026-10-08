"""Real CPU model tests for inference intervention and saved-output replay gates."""

from contextlib import redirect_stdout
import copy
import io
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.chronological import probe_incident_routing_p2 as probe
from src.models.igstgnn import IGSTGNN


def tiny_model(routing='acdg', layers=2):
    model = IGSTGNN(
        model_args=dict(num_feat=1, num_hidden=8, node_hidden=4, time_emb_dim=4,
                        layer=layers, k_s=1, k_t=2, tpd=288, dropout=.1, gap=3,
                        sigma_t=1., lambda_incident=1., adjs=[torch.eye(3), torch.eye(3)],
                        incident_schema='report_location_v1', time_response='fixed',
                        incident_routing=routing),
        node_num=3, input_dim=3, output_dim=1, seq_len=12, horizon=12,
        dataset='probe_test', use_sensor_info=False).eval()
    if routing == 'acdg':
        with torch.no_grad():
            for layer in model.layers:
                layer.estimation_gate.incident_condition.mlp[-1].bias.fill_(.8)
    return model


class TinyValidation:
    def __init__(self):
        self.rows = [{'sample_index': 10}, {'sample_index': 20}]
        self.station_ids = np.array([100, 200, 300])
        self.scaler = {'mean': 0., 'std': 1.}
        self.history = torch.rand(2, 12, 3, 3)
        self.incident = {
            'report_age_minutes': torch.tensor([2., 3.]),
            'forecast_tod': torch.tensor([10, 20], dtype=torch.long),
            'forecast_dow': torch.tensor([1, 2], dtype=torch.long),
            'distances': torch.tensor([[[1., .2, 0.], [0., 0., 0.], [.4, .1, 1.]],
                                       [[0., 0., 0.], [.3, .1, 0.], [.2, .2, 0.]]]),
        }

    def __getitem__(self, i):
        return {'x': self.history[i], 'y_flow': torch.zeros(12, 3, 1),
                'y_valid': torch.ones(12, 3, 1, dtype=torch.bool),
                'incident': {key: value[i] for key, value in self.incident.items()}}


class GateInterventionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(123)
        self.model = tiny_model()
        self.data = TinyValidation()

    def forward(self, model=None):
        with torch.inference_mode():
            return (model or self.model)(self.data.history, incident_data=self.data.incident)

    def test_strided_base_gate_statistics_use_the_same_layout_and_keep_zero_changes_exact(self):
        for batch_size in (1, 2, 3):
            for steps in (12, 11, 9, 8, 7):
                with self.subTest(batch_size=batch_size, steps=steps):
                    full = torch.linspace(-5., 5., batch_size * 12 * 3).reshape(batch_size, 12, 3, 1)
                    base = full[:, -steps:]
                    before = full.clone()
                    applied = torch.zeros(base.shape)
                    applied[:, :, 0] = .8
                    with patch.object(probe.torch, 'sigmoid', wraps=torch.sigmoid) as sigmoid:
                        native, actual = probe.diagnostic_gate_values(base, applied)
                    native_logits, actual_logits = [call.args[0] for call in sigmoid.call_args_list]
                    self.assertTrue(native_logits.is_contiguous())
                    self.assertEqual(native_logits.stride(), actual_logits.stride())
                    self.assertEqual(native_logits.storage_offset(), actual_logits.storage_offset())
                    torch.testing.assert_close(native[applied == 0], actual[applied == 0], rtol=0, atol=0)
                    self.assertTrue(torch.all(actual[applied != 0] > native[applied != 0]).item())
                    native, actual = probe.diagnostic_gate_values(base, torch.zeros_like(applied))
                    torch.testing.assert_close(native, actual, rtol=0, atol=0)
                    torch.testing.assert_close(full, before, rtol=0, atol=0)

    def test_on_preserves_original_and_off_matches_native_at_identical_common_weights(self):
        before = probe.train.state_sha256(self.model.state_dict())
        original = self.forward()
        with probe.GateIntervention(self.model, 'on') as intervention:
            on = self.forward()
            rows = intervention.results(2, 1)
        torch.testing.assert_close(on, original, rtol=0, atol=0)
        supported = [r for r in rows if r['region'] == 'associated_nodes']
        self.assertTrue(all(r['gate_change_abs_mean'] > 0 for r in supported))
        with probe.GateIntervention(self.model, 'off') as intervention:
            off = self.forward()
            off_rows = intervention.results(2, 1)
        native = tiny_model('none')
        # This native-shaped model receives ACDG's common weights, not separately trained weights.
        native.load_state_dict({key: self.model.state_dict()[key] for key in native.state_dict()}, strict=True)
        torch.testing.assert_close(off, self.forward(native), rtol=0, atol=0)
        self.assertGreater(float((on - off).abs().max()), 0.)
        self.assertTrue(all(r['applied_delta_abs_max'] == 0 for r in off_rows))
        self.assertTrue(all(r['gate_change_abs_max'] == 0 for r in off_rows))
        self.assertEqual(before, probe.train.state_sha256(self.model.state_dict()))
        torch.testing.assert_close(self.forward(), original, rtol=0, atol=0)

    def test_unsupported_condition_stays_zero_with_nonzero_learned_biases(self):
        with probe.GateIntervention(self.model, 'on') as intervention:
            self.forward()
            rows = intervention.results(2, 1)
        for row in rows:
            if row['region'] == 'nonassociated_nodes':
                self.assertGreater(row['gate_positions'], 0)
                self.assertEqual(row['proposed_delta_abs_max'], 0.)
                self.assertEqual(row['gate_change_abs_max'], 0.)

    def test_exception_removes_all_probe_hooks_and_restores_forward(self):
        original = self.forward()
        with self.assertRaisesRegex(RuntimeError, 'intentional'):
            with probe.GateIntervention(self.model, 'off'):
                self.forward()
                raise RuntimeError('intentional test failure')
        for layer in self.model.layers:
            self.assertFalse(layer.estimation_gate.incident_condition._forward_hooks)
            self.assertFalse(layer.estimation_gate.fully_connected_layer_2._forward_hooks)
        torch.testing.assert_close(self.forward(), original, rtol=0, atol=0)

    def test_rejects_training_mode_and_incomplete_hook_coverage(self):
        with self.assertRaisesRegex(ValueError, 'eval'):
            probe.GateIntervention(self.model.train(), 'on')
        self.model.eval()
        with probe.GateIntervention(self.model, 'on') as intervention:
            self.forward()
            with self.assertRaisesRegex(ValueError, 'validation batches'):
                intervention.results(3, 2)

    def test_original_evaluator_runs_inference_only_and_preserves_all_state(self):
        before = probe.train.state_sha256(self.model.state_dict())
        with patch.object(torch.optim, 'Adam', side_effect=AssertionError('Optimizer forbidden')):
            on, rows, _ = probe.infer(self.model, self.data, 2, torch.device('cpu'), 'on')
            off, _, _ = probe.infer(self.model, self.data, 2, torch.device('cpu'), 'off')
        probe.paired_predictions(on, off)
        self.assertEqual(on['prediction'].shape, (2, 12, 3, 1))
        self.assertEqual(on['sample_indices'].tolist(), [10, 20])
        self.assertEqual(len(rows), 4)
        self.assertEqual(before, probe.train.state_sha256(self.model.state_dict()))
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))

    def test_zero_initialized_branch_has_exactly_identical_on_off_predictions(self):
        with torch.no_grad():
            for layer in self.model.layers:
                layer.estimation_gate.incident_condition.mlp[-1].bias.zero_()
        on, _, _ = probe.infer(self.model, self.data, 2, torch.device('cpu'), 'on')
        off, _, _ = probe.infer(self.model, self.data, 2, torch.device('cpu'), 'off')
        np.testing.assert_array_equal(on['prediction'], off['prediction'])

    def test_failed_on_replay_blocks_off_pass(self):
        saved, _, _ = probe.infer(self.model, self.data, 2, torch.device('cpu'), 'on')
        changed = copy.deepcopy(saved)
        changed['prediction'] += 1.
        with patch.object(probe, 'infer', return_value=(changed, [], 1.)) as mocked, \
                patch.object(probe.train, 'atomic_json') as writer, redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, 'OFF was not run'):
                probe.inference_pair(self.model, self.data, 2, torch.device('cpu'), saved, Path('unused-probe-output'))
        self.assertEqual(mocked.call_count, 1)
        self.assertEqual(mocked.call_args.args[-1], 'on')
        self.assertFalse(writer.call_args.args[1]['passed'])

    def test_metric_sign_and_axes_for_same_checkpoint_intervention(self):
        on, _, _ = probe.infer(self.model, self.data, 2, torch.device('cpu'), 'on')
        fixed, off = copy.deepcopy(on), copy.deepcopy(on)
        fixed['prediction'].fill(1.5)
        on['prediction'].fill(2.)
        off['prediction'].fill(1.)
        report, rows = probe.comparisons(fixed, on, off)
        self.assertEqual(report['all_nodes']['off_minus_on_mae'], -1.)
        self.assertEqual(report['all_nodes']['off_minus_fixed_mae'], -.5)
        self.assertEqual(len(rows), 36)
        off['sample_indices'] = off['sample_indices'][::-1].copy()
        with self.assertRaisesRegex(ValueError, 'sample_indices'):
            probe.comparisons(fixed, on, off)


if __name__ == '__main__':
    unittest.main()
