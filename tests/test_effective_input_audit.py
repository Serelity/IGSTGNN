"""Input isolation, provenance, and real-model equivalence for v13a."""

import contextlib
import copy
import csv
from datetime import datetime, timedelta
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.chronological import audit_effective_inputs as audit
from src.models.igstgnn import IGSTGNN
from src.models.single_incident_icsf import SingleIncidentICSF, compare_single_incident
from src.utils.chronological import encode_window


def tiny_model():
    torch.manual_seed(13025)
    torch.set_num_threads(1)
    return IGSTGNN(
        model_args=dict(num_feat=1, num_hidden=8, node_hidden=4, time_emb_dim=4,
                        layer=5, k_s=2, k_t=3, tpd=288, dropout=.1, gap=3,
                        sigma_t=1., lambda_incident=1., adjs=[torch.ones(4, 4) / 4] * 2,
                        incident_schema='report_location_v1', time_response='fixed'),
        node_num=4, input_dim=3, output_dim=1, seq_len=12, horizon=12,
        dataset='fixture', use_sensor_info=False).eval().requires_grad_(False)


def inputs():
    rng = torch.Generator().manual_seed(13)
    return torch.rand(2, 12, 4, 3, generator=rng), {
        'report_age_minutes': torch.tensor([1., 5.]),
        'forecast_tod': torch.tensor([0, 287]), 'forecast_dow': torch.tensor([0, 6]),
        'distances': torch.tensor([[[0., .8, 0.], [0., .3, 1.], [0., 0., 0.], [0., 0., 0.]]] * 2),
    }


def fixture(directory):
    protocol = copy.deepcopy(audit.load_protocol())
    protocol.update(expected_train_samples=3, expected_nodes=4, expected_fit_samples=2, batch_size=1)
    rows = []
    times = [datetime(2023, 2, 5, 0, 5), datetime(2023, 2, 6, 12), datetime(2023, 5, 10, 12)]
    for i, t0 in enumerate(times):
        stamps = {'report_time': t0 - timedelta(minutes=i + 1), 't0': t0,
                  'x_start': t0 - timedelta(minutes=65), 'x_end': t0 - timedelta(minutes=10),
                  'y_start': t0 + timedelta(minutes=5), 'y_end': t0 + timedelta(minutes=60),
                  'support_start': t0 - timedelta(minutes=70), 'support_end_exclusive': t0 + timedelta(minutes=65)}
        rows.append({'sample_index': str(90 + i), 'incident_id': str(400 + i), 'split': 'train',
                     'source_version': '8', **{k: v.isoformat() for k, v in stamps.items()}})
    audit.write_csv(directory / 'train_manifest.csv', rows)
    flow = np.full((3, 26, 4), 12., dtype=np.float32)
    flow[0, 0, 0], flow[0, 1, 0], flow[0, 2, 1] = 0., np.nan, -1.
    flow[:, 12:] = 99123.
    flow[2, :12] = 88777.  # Excluded later window must never enter inventory.
    np.save(directory / 'train_flow.npy', flow)
    np.save(directory / 'station_ids.npy', [10, 20, 30, 40])
    np.save(directory / 'adjacency.npy', np.eye(4))
    context = {'sample_indices': np.array([90, 91, 92]), 'station_ids': np.array([10, 20, 30, 40]),
               'report_age_minutes': np.array([1., 2., 3.], dtype=np.float32),
               'forecast_tod': np.array([t.hour * 12 + t.minute // 5 for t in times]),
               'forecast_dow': np.array([(t.weekday() + 1) % 7 for t in times]),
               'distances': np.array([[[0., .8, 0.], [0., .3, 1.], [0., 0., 0.], [0., 0., 0.]]] * 3, dtype=np.float32)}
    np.savez(directory / 'train_context.npz', **context)
    audit.write_json(directory / 'scaler.json', {
        'mean': 10., 'std': 2., 'node_fill_mean': [10., 11., 12., 13.], 'station_ids': [10, 20, 30, 40],
        'fitted_sample_indices': [90, 91, 92], 'source_version': 8,
        'fit_scope': 'train_X_0:12_unique_station_nominal_slot_finite_nonnegative'})
    summary = {'build_complete': True, 'files': {name: audit.sha256(directory / name) for name in (
        'train_flow.npy', 'train_manifest.csv', 'station_ids.npy', 'scaler.json')}}
    context_meta = {'schema': 'report_location_v1', 'outputs': {name: audit.sha256(directory / name) for name in (
        'train_context.npz', 'adjacency.npy')}}
    audit.write_json(directory / 'summary.json', summary)
    audit.write_json(directory / 'context_manifest.json', context_meta)
    return protocol, rows, context


class InputTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        self.protocol, self.rows, self.context = fixture(self.directory)

    def test_inventory_excludes_targets_gap_and_later_inputs_and_counts_real_zero(self):
        data = audit.FitInputs(self.directory, self.protocol)
        fields, nodes, stats = audit.inventory(data, data.indices, self.protocol)
        self.assertEqual(data.indices, [0, 1])
        self.assertEqual(stats['history_flow']['count'], 96)
        self.assertEqual(stats['history_flow']['valid_count'], 94)
        self.assertEqual(stats['history_flow']['zero_count'], 1)
        self.assertEqual(stats['history_flow']['max'], 12.)
        self.assertEqual(sum(n['invalid_cells'] for n in nodes), 2)
        changed = np.load(self.directory / 'train_flow.npy')
        changed[:, 12:] = np.nan
        changed[2, :12] = -np.inf
        del data
        np.save(self.directory / 'train_flow.npy', changed)
        altered = audit.FitInputs(self.directory, self.protocol)
        self.assertEqual((fields, nodes, stats), audit.inventory(altered, altered.indices, self.protocol))

    def test_history_encoding_matches_original_scaler_and_day_rollover(self):
        data = audit.FitInputs(self.directory, self.protocol)
        x, incident = data.model_batch([0, 1], 'cpu')
        for local, index in enumerate([0, 1]):
            original = encode_window(data.flow[index], datetime.fromisoformat(data.rows[index]['x_start']), data.scaler)[0]
            np.testing.assert_array_equal(x[local].numpy(), original)
        self.assertEqual(incident['forecast_dow'].tolist(), [0, 1])
        self.assertEqual(float(x[0, -1, 0, 2]), np.float32(6 / 7))
        self.assertEqual(float(x[0, 0, 0, 0]), -5.)  # A real zero was not filled.
        self.assertEqual(float(x[0, 1, 0, 0]), 0.)  # Missing input uses frozen fill.

    def test_complete_support_not_just_t0_controls_eligibility(self):
        row = self.rows[1]
        row['support_end_exclusive'] = '2023-04-24T00:05:00'
        audit.write_csv(self.directory / 'train_manifest.csv', self.rows)
        with self.assertRaisesRegex(ValueError, 'Fit eligibility'):
            audit.FitInputs(self.directory, self.protocol)

    def test_input_view_cannot_expose_a_later_window_even_if_called_directly(self):
        data = audit.FitInputs(self.directory, self.protocol)
        for indices in ([2], [-1], [True], []):
            with self.assertRaisesRegex(ValueError, 'eligible fit indices'):
                data.history(indices)
        with self.assertRaisesRegex(ValueError, 'eligible fit indices'):
            data.model_batch([2], 'cpu')

    def test_order_clock_duplicate_and_scaler_corruption_rejected(self):
        cases = ['order', 'clock', 'duplicate', 'scaler', 'nonfinite_age', 'spatial_shape']
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                base = Path(directory)
                protocol, rows, context = fixture(base)
                if case == 'order': context['station_ids'] = context['station_ids'][::-1]
                if case == 'clock': context['forecast_tod'][0] += 1
                if case == 'nonfinite_age': context['report_age_minutes'][0] = np.nan
                if case == 'spatial_shape': context['distances'] = context['distances'][:, :, None, :]
                if case == 'duplicate':
                    rows[1]['sample_index'] = rows[0]['sample_index']
                    audit.write_csv(base / 'train_manifest.csv', rows)
                if case == 'scaler':
                    scaler = json.loads((base / 'scaler.json').read_text())
                    scaler['fitted_sample_indices'] = [90, 91]
                    audit.write_json(base / 'scaler.json', scaler)
                np.savez(base / 'train_context.npz', **context)
                with self.assertRaises(ValueError): audit.FitInputs(base, protocol)

    def test_fingerprints_and_train_only_file_allowlist(self):
        baseline = {'positive_package': {'summary_sha256': audit.sha256(self.directory / 'summary.json'),
                                        'context_manifest_sha256': audit.sha256(self.directory / 'context_manifest.json')},
                    'checkpoint': {'best_model_sha256': 'unused'}}
        baseline_path = self.directory / self.protocol['baseline_protocol']
        audit.write_json(baseline_path, baseline)
        self.protocol['baseline_protocol_sha256'] = audit.sha256(baseline_path)
        self.protocol['model_source_sha256'] = {}
        # Fixture intentionally contains no val/test files; real verification must succeed without them.
        with patch.object(audit, 'PROTOCOL', self.directory / 'fixture_protocol.json'):
            actual = audit.verify_inputs(self.directory, self.protocol)
            self.assertTrue(all('val_' not in path and 'test_' not in path for path in actual))
            with (self.directory / 'train_manifest.csv').open('a') as stream: stream.write('\n')
            with self.assertRaisesRegex(ValueError, 'fingerprint mismatch'):
                audit.verify_inputs(self.directory, self.protocol)

    def test_frozen_protocol_cannot_be_changed(self):
        path = self.directory / 'changed.json'
        audit.write_json(path, {**audit.load_protocol(), 'expected_fit_samples': 1})
        with self.assertRaisesRegex(ValueError, 'fingerprint changed'): audit.load_protocol(path)

    def test_cardinality_cap_and_all_missing_are_not_reported_as_constant_zero(self):
        stats = audit.NumericInventory(cap=2)
        stats.update([0, 1, 2, 3, np.nan, -1, np.inf])
        result = stats.result()
        self.assertIsNone(result['unique_count'])
        self.assertEqual(result['unique_count_lower_bound'], 3)
        missing = audit.NumericInventory()
        missing.update([np.nan, -1])
        self.assertIsNone(missing.result()['constant_on_valid'])
        self.assertIsNone(missing.result()['min'])

    def test_atomic_inventory_output_and_failure_preservation(self):
        out = self.directory / 'result'
        with patch.object(audit, 'load_protocol', return_value=self.protocol), \
                patch.object(audit, 'verify_inputs', return_value={}), contextlib.redirect_stdout(io.StringIO()):
            result = audit.run(self.directory, out, inventory_only=True)
            self.assertEqual(result['status'], 'FIT_INPUT_INVENTORY_COMPLETE')
            self.assertIsNone(result['equivalence'])
            self.assertFalse(Path(str(out) + '.partial').exists())
            for name, expected in result['artifacts'].items(): self.assertEqual(audit.sha256(out / name), expected)
            with self.assertRaises(FileExistsError): audit.run(self.directory, out, inventory_only=True)
            with patch.object(audit, 'inventory', side_effect=ValueError('deliberate failure')):
                failed = self.directory / 'failed'
                with self.assertRaises(ValueError): audit.run(self.directory, failed, inventory_only=True)
                self.assertFalse(failed.exists())
                self.assertIn('deliberate failure', (Path(str(failed) + '.partial') / 'failure.json').read_text())

    def test_random_weights_never_permitted_as_formal_equivalence(self):
        with self.assertRaisesRegex(ValueError, 'Random initialization'):
            audit.run(self.directory, self.directory / 'unsafe', random_init_check=True)
        with self.assertRaisesRegex(ValueError, 'checkpoint required'):
            audit.run(self.directory, self.directory / 'unsafe')


class EquivalenceTests(unittest.TestCase):
    def test_real_backbone_matches_with_no_parameter_change_and_bypasses_q_mlp(self):
        model = tiny_model()
        before = audit.state_hash(model)
        original = model.icsf_module
        with patch.object(original.q_proj, 'forward', wraps=original.q_proj.forward) as q, \
                patch.object(original.icsf_fusion_mlp, 'forward', wraps=original.icsf_fusion_mlp.forward) as fusion:
            result = compare_single_incident(model, *inputs())
        self.assertEqual(q.call_count, 1)
        self.assertEqual(fusion.call_count, 1)
        self.assertTrue(all(item['exactly_equal'] for item in result.values()))
        self.assertGreater(result['forecast']['cells'], 0)
        self.assertIs(model.icsf_module, original)
        self.assertEqual(before, audit.state_hash(model))
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_zero_mixed_and_all_support_and_distance_variation(self):
        model = tiny_model()
        x, context = inputs()
        for kind in ('zero', 'all', 'tiny_proximity', 'mixed'):
            changed = {k: v.clone() for k, v in context.items()}
            if kind == 'zero': changed['distances'].zero_()
            if kind == 'all': changed['distances'][..., 1] = .7
            if kind == 'tiny_proximity': changed['distances'][..., 1] *= 1e-12
            result = compare_single_incident(model, x, changed)
            self.assertTrue(all(item['exactly_equal'] for item in result.values()), kind)
        base = model.icsf_module
        history = torch.randn(2, 12, 4, 8)
        context['distances'].zero_()
        with torch.inference_mode():
            enhanced, _ = SingleIncidentICSF(base).eval()(history, context, None, torch.zeros(2, 4), torch.zeros(2, 4))
            torch.testing.assert_close(enhanced[:, -1], base.output_norm(history[:, -1]), atol=0, rtol=0)
            self.assertFalse(torch.equal(enhanced[:, -1], history[:, -1]))

    def test_wrong_simplification_is_rejected_and_original_restored(self):
        model = tiny_model()
        original = model.icsf_module
        forward = SingleIncidentICSF.forward

        def broken(module, *args, **kwargs):
            enhanced, context = forward(module, *args, **kwargs)
            enhanced = enhanced + .1
            return enhanced, context

        with patch.object(SingleIncidentICSF, 'forward', broken):
            with self.assertRaisesRegex(ValueError, 'equivalence failed'):
                compare_single_incident(model, *inputs())
        self.assertIs(model.icsf_module, original)
        self.assertEqual(len(original._forward_hooks), 0)

    def test_nonfinite_and_out_of_scope_paths_fail_closed(self):
        model = tiny_model()
        x, context = inputs()
        x[0, 0, 0, 0] = float('nan')
        with self.assertRaises(ValueError): compare_single_incident(model, x, context)
        model.train()
        with self.assertRaisesRegex(ValueError, 'eval-mode'): compare_single_incident(model, *inputs())
        model.eval()
        model._time_response_mode = 'shared'
        with self.assertRaises(ValueError): compare_single_incident(model, *inputs())
        wrapper = SingleIncidentICSF(model.icsf_module).eval()
        with self.assertRaisesRegex(ValueError, 'inference-only'):
            wrapper(torch.zeros(2, 12, 4, 8), context)
        model.icsf_module.use_sensor_info = True
        with self.assertRaisesRegex(ValueError, 'static sensor'): SingleIncidentICSF(model.icsf_module)


if __name__ == '__main__':
    unittest.main()
