"""Server-only checks for FULL-only original validation; not executed locally."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.original_baseline_gw16 import evaluate as entry
from experiments.chronological.original_igstgnn_runtime import sha256
from experiments.chronological.original_igstgnn_source import get_original_modules


def arrays(engineering=True):
    count = 4 if engineering else 837
    target = np.full((count, 12, 496, 1), 10., dtype=np.float32)
    return dict(prediction=target+1, target=target, sample_ids=np.arange(3909, 3909+count, dtype=np.int64),
                station_ids=np.arange(496, dtype=np.int64))


class OriginalBaselineGW16EvaluateTests(unittest.TestCase):
    def test_native_summary_reuses_global_null_for_every_horizon(self):
        saved = arrays()
        saved['target'][0, 0, 0, 0] = -1
        saved['prediction'][0, 0, 0, 0] = 1000
        saved['target'][:, 1] = 2
        modules = get_original_modules()
        result = entry.native_summary(saved['prediction'], saved['target'], modules)
        self.assertEqual(result['mask_value'], -1)
        expected = modules.metrics.compute_all_metrics(
            torch.as_tensor(saved['prediction'][:, 1, :, 0]),
            torch.as_tensor(saved['target'][:, 1, :, 0]), torch.tensor(-1.))
        self.assertEqual(result['per_horizon'][1]['mae'], expected[0])
        self.assertEqual(result['per_horizon'][1]['mape_percent'], 100*expected[1])
        self.assertAlmostEqual(result['horizon_macro']['mae'],
                               np.mean([row['mae'] for row in result['per_horizon']]))

    def test_native_macro_rmse_is_mean_of_horizon_rmse(self):
        saved = arrays()
        saved['prediction'][:, 0] += 5
        result = entry.native_summary(saved['prediction'], saved['target'])
        self.assertNotAlmostEqual(result['pooled']['rmse'], result['horizon_macro']['rmse'])

    def test_explicit_validation_axis_and_float32_required(self):
        saved = arrays()
        entry.validate_arrays(saved, True)
        saved['sample_ids'][0] = 0
        with self.assertRaises(ValueError):
            entry.validate_arrays(saved, True)
        saved = arrays()
        saved['prediction'] = saved['prediction'].astype(np.float64)
        with self.assertRaises(ValueError):
            entry.validate_arrays(saved, True)

    def test_engineering_cannot_have_formal_axes(self):
        with self.assertRaises(ValueError):
            entry.validate_arrays(arrays(), False)

    def test_unsafe_output_paths_rejected(self):
        with TemporaryDirectory() as name:
            root = Path(name)
            data = root/'data'
            source = root/'source'
            data.mkdir()
            source.mkdir()
            for path in (data/'eval', source/'eval', source):
                with self.assertRaises(ValueError):
                    entry.protect_output(path, data, source)
            other = root/'other'
            other.mkdir()
            (other/'identity.json').write_text('{}', encoding='utf-8')
            with self.assertRaises(ValueError):
                entry.protect_output(other/'eval', data, source)
            entry.protect_output(root/'fresh', data, source)

    def test_historical_five_epoch_archive_never_fully_trained(self):
        archive = SimpleNamespace(config=dict(max_epochs=5, patience=20), log_text='',
            metadata=dict(completed_epochs=5, best_epoch=5,
                          training_rows=[dict(epoch=h, validation_mae=20-h) for h in range(1, 6)]))
        self.assertFalse(entry.archive_reference(archive)['training_complete'])
        self.assertEqual(entry.archive_reference(archive)['reference_level'], 'historical_reference')

    def test_original_early_stop_or_full_budget_completion_evidence(self):
        rows = [dict(epoch=h, validation_mae=10 if h == 1 else 11) for h in range(1, 22)]
        archive = SimpleNamespace(config=dict(max_epochs=100, patience=20),
            log_text='Early stop at epoch 21, loss = 10.000000',
            metadata=dict(completed_epochs=21, best_epoch=1, training_rows=rows))
        self.assertTrue(entry.archive_reference(archive)['training_complete'])
        archive.log_text = ''
        self.assertFalse(entry.archive_reference(archive)['training_complete'])
        archive.metadata['completed_epochs'] = 100
        archive.metadata['training_rows'] = [dict(epoch=h, validation_mae=10 if h == 1 else 11) for h in range(1, 101)]
        self.assertTrue(entry.archive_reference(archive)['training_complete'])

    def test_cli_requires_one_exact_source_and_paired_training_log(self):
        with self.assertRaises(SystemExit):
            entry.parse_args(['--output-dir', 'out', '--checkpoint', 'weights.pt'])
        args = entry.parse_args(['--output-dir', 'out', '--checkpoint', 'weights.pt', '--training-log', 'record.log'])
        self.assertEqual(args.checkpoint, Path('weights.pt'))
        self.assertFalse(args.check)

    def test_read_archive_rejects_payload_tampering_before_metrics(self):
        with TemporaryDirectory() as name:
            root = Path(name)
            np.savez_compressed(root/'predictions.npz', **arrays())
            metadata = dict(schema=entry.SCHEMA, native_replay=dict(passed=True), test_evaluated=False,
                            archive_sha256=sha256(root/'predictions.npz'), engineering_only=True)
            (root/'metadata.json').write_text(json.dumps(metadata), encoding='utf-8')
            with (root/'predictions.npz').open('ab') as stream:
                stream.write(b'tampered')
            with self.assertRaisesRegex(ValueError, 'fingerprint'):
                entry.read_archive(root)

    def test_read_archive_roundtrip_and_metric_tampering(self):
        with TemporaryDirectory() as name:
            root = Path(name)
            saved = arrays()
            native = entry.native_summary(saved['prediction'], saved['target'])
            modules = get_original_modules()
            np.savez_compressed(root/'predictions.npz', **saved)
            metadata = dict(schema=entry.SCHEMA, native_replay=dict(passed=True), test_evaluated=False,
                archive_sha256=sha256(root/'predictions.npz'), engineering_only=True, source={},
                data_identity=dict(hashes=entry.RELEASE, splits=dict(val=dict(selected_indices=saved['sample_ids'].tolist()))),
                execution_source={}, data_dir=str(root), source_fingerprint=modules.source_fingerprint,
                registered_parameters=455220, station_ids=saved['station_ids'].tolist(),
                native_metrics=native['pooled'], native_per_horizon=native['per_horizon'],
                native_macro_average=native['horizon_macro'], runtime=dict(threads=torch.get_num_threads()),
                gw16_common_metrics=entry.numpy_metrics(saved['prediction'], saved['target'], entry.COMMON_CONFIG),
                expected_validation_mae=native['pooled']['mae'])
            (root/'metadata.json').write_text(json.dumps(metadata), encoding='utf-8')
            with patch.object(entry, 'recheck_files', return_value=modules):
                loaded, _ = entry.read_archive(root)
                np.testing.assert_array_equal(loaded['prediction'], saved['prediction'])
                metadata['native_macro_average']['mae'] += 1
                (root/'metadata.json').write_text(json.dumps(metadata), encoding='utf-8')
                with self.assertRaisesRegex(ValueError, 'cached metrics changed'):
                    entry.read_archive(root)

    def test_native_replay_mismatch_not_relaxed(self):
        with self.assertRaisesRegex(ValueError, 'replay MAE mismatch'):
            entry.verify_replay(dict(mae=14.), 13.)


if __name__ == '__main__':
    unittest.main()
