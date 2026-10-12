"""Published-data fidelity and restart checks; no accuracy claims."""
import csv
import io
import logging
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout, redirect_stderr

import numpy as np
import torch

from experiments.chronological import train_b3_r_minimal as entry
from experiments.chronological import train_b3_r_minimal_paper as training
from experiments.chronological import train_internal_spatial as legacy
from src.models.incident_spatial_minimal import cutoff_clock_from_history, from_residual
from src.models.incident_spatial_residual import from_canonical_b3
from src.utils.dataloader import IncidentDataLoader
from src.utils.igstgnn_published import PublishedData
from tests.test_b3_r import canonical
from tests.test_incident_spatial import graph


def fixture(directory):
    samples = []
    for i in range(20):
        x, y = np.zeros((12, 4, 3), np.float32), np.zeros((12, 4, 3), np.float32)
        x[..., 0] = (np.arange(48).reshape(12, 4)+i)/50
        y[..., 0] = x[..., 0]+.2
        x[..., 1] = np.arange(10, 22)[:, None]/288
        y[..., 1] = np.arange(22, 34)[:, None]/288
        x[..., 2] = y[..., 2] = 2/7
        samples.append(dict(x_data=x, y_data=y, incident_features={'unused': i}))
    np.save(directory/'incident_all.npy', np.array(samples, dtype=object))
    np.savez(directory/'incident_stats.npz', mean=np.float32(100), std=np.float32(20), normalized=True)
    np.save(directory/'adj_matrix.npy', np.eye(4, dtype=np.float32))
    with (directory/'sensors.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['station_id'])
        writer.writeheader()
        writer.writerows({'station_id': i+100} for i in range(4))
    protocol = dict(legacy.read_json(training.PROTOCOL), total_samples=20, train_samples=14,
                    val_samples=3, test_samples=3, stations=4, batch_size=2, warm_epoch=1, cl_epoch=1)
    return samples, protocol


class PaperMinimalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        cls.logger = logging.getLogger('paper_min_test')
        cls.logger.addHandler(logging.NullHandler())

    def nested_equal(self, left, right):
        if torch.is_tensor(left):
            torch.testing.assert_close(left, right, atol=0, rtol=0)
        elif isinstance(left, np.ndarray):
            np.testing.assert_array_equal(left, right)
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.nested_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.nested_equal(a, b)
        else:
            self.assertEqual(left, right)

    def test_batches_are_exactly_author_loader_and_scaler_is_not_refitted(self):
        with tempfile.TemporaryDirectory(dir=training.REPO) as temp:
            root = Path(temp)
            samples, protocol = fixture(root)
            data = PublishedData(root, protocol, self.logger)
            self.assertEqual(data.station_ids, [100, 101, 102, 103])
            original = IncidentDataLoader(samples[:14], False, 2, self.logger, input_dim=3)
            for actual, expected in zip(data.loaders['train_loader'].get_iterator(), original.get_iterator()):
                self.nested_equal(actual, expected)
            x, y = next(data.loaders['val_loader'].get_iterator())
            np.testing.assert_array_equal(x[0], samples[14]['x_data'])
            np.testing.assert_array_equal(y[0], samples[14]['y_data'][..., :1])
            self.assertEqual(float(data.scaler.mean), 100)
            self.assertEqual(float(data.scaler.std), 20)
            self.assertTrue(data.audit['test_payload_deserialized'])
            self.assertFalse(data.audit['test_evaluated'])
            np.testing.assert_array_equal(next(data.test_loader(2, self.logger).get_iterator())[0][0], samples[17]['x_data'])
            # Validation traffic changes must not alter the supplied statistics.
            samples[14]['x_data'][..., 0] = 999
            np.save(root/'incident_all.npy', np.array(samples, dtype=object))
            changed = PublishedData(root, protocol, self.logger)
            self.assertEqual(float(changed.scaler.mean), 100)
            self.assertEqual(float(changed.scaler.std), 20)

    def test_release_hash_gate_rejects_changed_source_and_chronological_path(self):
        with tempfile.TemporaryDirectory(dir=training.REPO) as temp:
            root = Path(temp)
            fixture(root)
            expected = {'data_sha256': {'incident_all.npy': training.sha256(root/'incident_all.npy')}}
            training.verify_release(root, expected)
            (root/'incident_all.npy').write_bytes(b'changed source')
            with self.assertRaisesRegex(ValueError, 'fingerprint mismatch'):
                training.verify_release(root, expected)
            with self.assertRaisesRegex(ValueError, 'file missing'):
                training.verify_release(root/'v8', expected)

    def test_entry_defaults_to_paper_and_old_identity_is_rejected_before_loading(self):
        with patch.object(training, 'main', return_value='paper') as call:
            self.assertEqual(entry.main(['--epochs', '1']), 'paper')
            call.assert_called_once_with(['--epochs', '1'])
        with tempfile.TemporaryDirectory(dir=training.REPO) as temp:
            root = Path(temp)
            legacy.write_json(root/'identity.json', {'schema': 'b3r_minimal_run_v1'})
            with patch.object(training, 'verify_release', side_effect=AssertionError('Data accessed')):
                with self.assertRaisesRegex(ValueError, 'chronological checkpoints'):
                    training.main(['--run-dir', str(root), '--device', 'cpu'])

    def test_batch_override_preserves_frozen_protocol_and_resume_guard(self):
        before = training.PROTOCOL.read_bytes()
        default = training.load_protocol(training.parse_args([]))
        larger = training.load_protocol(training.parse_args(['--batch-size', '64']))
        self.assertEqual(default['batch_size'], 48)
        self.assertEqual(larger, dict(default, batch_size=64))
        self.assertEqual(training.PROTOCOL.read_bytes(), before)
        for value in ('0', '-1'):
            with self.assertRaisesRegex(ValueError, 'positive integer'):
                training.parse_args(['--batch-size', value])
        with tempfile.TemporaryDirectory(dir=training.REPO) as temp:
            root = Path(temp)
            _, protocol = fixture(root)
            larger_fixture = dict(protocol, batch_size=4)
            data = PublishedData(root, larger_fixture, self.logger)
            self.assertEqual(data.loaders['train_loader'].bs, 4)
            self.assertEqual(data.loaders['val_loader'].bs, 4)
            self.assertEqual(data.loaders['train_loader'].num_batch, 4)
            check = PublishedData(root, larger_fixture, self.logger, check=True)
            self.assertEqual(check.loaders['train_loader'].bs, 2)
            old_identity = {'schema': training.SCHEMA, 'protocol': default, 'batch_size': 48}
            new_identity = dict(old_identity, protocol=larger, batch_size=64)
            legacy.write_json(root/'identity.json', old_identity)
            with self.assertRaisesRegex(ValueError, 'Run identity changed'):
                training.verify_identity(root, new_identity)

    def test_published_clock_has_no_gap_and_pruning_matches_reference_at_same_clock(self):
        with tempfile.TemporaryDirectory(dir=training.REPO) as temp:
            samples, _ = fixture(Path(temp))
        x = torch.from_numpy(np.stack([s['x_data'] for s in samples[:2]]))
        tod, dow = cutoff_clock_from_history(x, offset_slots=0)
        self.assertEqual(tod.tolist(), [21, 21])
        self.assertEqual(dow.tolist(), [2, 2])
        parent, _ = from_canonical_b3(canonical())
        parent.eval()
        model, certificate = from_residual(parent, clock_offset_slots=0)
        model.eval()
        report = dict(forecast_tod=tod, forecast_dow=dow, report_age_minutes=torch.ones(2),
                      distances=torch.zeros(2, 4, 3))
        with torch.inference_mode():
            expected = parent(x, incident_data=report, spatial_report_enabled=False, tiid_report_enabled=False)
            actual = model(x)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        self.assertEqual(certificate['cutoff_clock_policy'], 'last_published_history_slot')
        self.assertFalse(hasattr(model, 'tiid_module'))
        self.assertFalse(hasattr(model.report_spatial_state, 'D'))
        x[:, -1, :, 1], x[:, -1, :, 2] = 287/288, 6/7
        tod, dow = cutoff_clock_from_history(x, offset_slots=0)
        self.assertEqual((tod.tolist(), dow.tolist()), ([287, 287], [6, 6]))

    def test_original_curriculum_and_exact_epoch_boundary_resume(self):
        with tempfile.TemporaryDirectory(dir=training.REPO) as temp, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            root = Path(temp)
            _, protocol = fixture(root)
            identity = {'schema': training.SCHEMA, 'fixture': True}
            saved_results = []
            for name, stages in [('continuous', [2]), ('resumed', [1, 2])]:
                run = root/name
                run.mkdir()
                for epochs in stages:
                    args = SimpleNamespace(run_dir=run, data_dir=root, epochs=epochs, check=True,
                                           device='cpu', seed=2025, threads=1, evaluate_test=False)
                    data = PublishedData(root, protocol, self.logger, check=True)
                    training.train(args, protocol, data, identity, graph(4, ((0, 1),)), self.logger)
                saved = torch.load(run/training.ARM/'last.pt', map_location='cpu', weights_only=False)
                saved_results.append(saved)
                self.assertEqual(saved['global_updates'], 4)
                self.assertEqual(saved['history'][0]['curriculum_length'], 1)
                self.assertEqual(saved['history'][1]['curriculum_length'], 2)
                before = (run/training.ARM/'last.pt').read_bytes()
                training.train(args, protocol, data, identity, graph(4, ((0, 1),)), self.logger)
                self.assertEqual(before, (run/training.ARM/'last.pt').read_bytes())
            a, b = saved_results
            for key in ('model_state', 'best_model_state', 'optimizer_state', 'scheduler_state', 'train_order',
                        'numpy_rng', 'torch_rng', 'best_metric', 'wait', 'global_updates', 'curriculum_length'):
                self.nested_equal(a[key], b[key])
            for x, y in zip(a['history'], b['history']):
                self.nested_equal({k: v for k, v in x.items() if k != 'seconds'},
                                  {k: v for k, v in y.items() if k != 'seconds'})
            bad = dict(identity, fixture=False)
            engine = training.make_engine(args, protocol, data, training.build_model(args, 4, graph(4, ((0, 1),)), 'cpu')[0], self.logger)
            with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                training.restore(b, bad, engine, data)


if __name__ == '__main__':
    unittest.main()
