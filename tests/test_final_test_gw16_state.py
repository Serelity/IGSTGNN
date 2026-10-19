"""Server-only checks of frozen test identity and prediction-cache boundaries."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.chronological.original_igstgnn_runtime import sha256, write_json
from experiments.final_test_gw16.state import (
    cache_read, cache_write, freeze_output, payload_hash, snapshot_files,
    validate_arrays, verify_files)


class FinalTestStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        target = np.zeros((839, 12, 496, 1), dtype=np.float32)
        cls.arrays = dict(prediction=target, target=target,
                          sample_ids=np.arange(4746, 5585, dtype=np.int64),
                          station_ids=np.arange(496, dtype=np.int64))

    def test_payload_hash_is_stable_and_refuses_nonfinite_json(self):
        self.assertEqual(payload_hash(dict(a=1, b=[2, 3])), payload_hash(dict(b=[2, 3], a=1)))
        self.assertNotEqual(payload_hash(dict(epoch=99)), payload_hash(dict(epoch=100)))
        with self.assertRaises(ValueError):
            payload_hash(dict(loss=float('nan')))

    def test_snapshot_resolves_paths_and_detects_change_or_missing_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'weights.pt'
            path.write_bytes(b'first frozen weights')
            snapshot = snapshot_files([path])
            self.assertEqual(set(snapshot), {str(path.resolve())})
            verify_files(snapshot)
            path.write_bytes(b'changed weights')
            with self.assertRaisesRegex(ValueError, 'changed'):
                verify_files(snapshot)
            path.unlink()
            with self.assertRaisesRegex(ValueError, 'missing'):
                verify_files(snapshot)
            with self.assertRaisesRegex(ValueError, 'not a file'):
                snapshot_files([path])

    def test_freeze_is_idempotent_and_survives_completed_cache_children(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)/'final'
            payload = dict(selection='validation_best', epoch=99, split='test')
            expected = freeze_output(directory, payload)
            before = (directory/'freeze.json').read_bytes()
            (directory/'STATE_EXCHANGE').mkdir()
            self.assertEqual(freeze_output(directory, payload), expected)
            self.assertEqual((directory/'freeze.json').read_bytes(), before)
            self.assertEqual(expected, payload_hash(payload))

    def test_freeze_accepts_an_existing_empty_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(freeze_output(folder, dict(epoch=99)), payload_hash(dict(epoch=99)))

    def test_freeze_refuses_changed_selection_and_tampered_record(self):
        with tempfile.TemporaryDirectory() as folder:
            payload = dict(epoch=99)
            freeze_output(folder, payload)
            with self.assertRaisesRegex(ValueError, 'freeze differs'):
                freeze_output(folder, dict(epoch=100))
            path = Path(folder)/'freeze.json'
            saved = json.loads(path.read_text())
            saved['freeze_id'] = 'forged'
            write_json(path, saved)
            with self.assertRaisesRegex(ValueError, 'freeze differs'):
                freeze_output(folder, payload)

    def test_freeze_refuses_uncommitted_residuals(self):
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder)/'partial.npz').write_bytes(b'partial')
            with self.assertRaisesRegex(ValueError, 'residual'):
                freeze_output(folder, dict(epoch=99))

    def test_valid_cache_roundtrip_preserves_caller_metadata_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)/'STATE_EXCHANGE'
            written = cache_write(directory, 'fixed', 'STATE_EXCHANGE', self.arrays,
                                  dict(checkpoint_epoch=99, metrics=dict(mae=0.)))
            arrays, metadata = cache_read(directory, 'fixed', 'STATE_EXCHANGE')
            self.assertEqual(metadata, written)
            self.assertEqual(metadata['checkpoint_epoch'], 99)
            self.assertEqual(metadata['metrics'], dict(mae=0.))
            for key in self.arrays:
                self.assertTrue(np.array_equal(arrays[key], self.arrays[key]))
            with self.assertRaisesRegex(ValueError, 'already exists'):
                cache_write(directory, 'fixed', 'STATE_EXCHANGE', self.arrays, {})

    def test_cache_rejects_changed_freeze_case_or_scope(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)/'cache'
            cache_write(directory, 'fixed', 'STATE_EXCHANGE', self.arrays, {})
            for freeze, case in [('different', 'STATE_EXCHANGE'), ('fixed', 'ORIGINAL_IGSTGNN')]:
                with self.subTest(freeze=freeze, case=case), self.assertRaisesRegex(ValueError, 'scope differs'):
                    cache_read(directory, freeze, case)
            path = directory/'metadata.json'
            original = json.loads(path.read_text())
            for key, value in [('split', 'val'), ('test_evaluated', False), ('optimizer_updates', 1),
                               ('optimizer_updates', False), ('test_evaluated', 1), ('schema', 'old')]:
                changed = dict(original)
                changed[key] = value
                write_json(path, changed)
                with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, 'scope differs'):
                    cache_read(directory, 'fixed', 'STATE_EXCHANGE')

    def test_cache_rejects_changed_archive_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)/'cache'
            cache_write(directory, 'fixed', 'STATE_EXCHANGE', self.arrays, {})
            with (directory/'predictions.npz').open('ab') as stream:
                stream.write(b'tamper')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                cache_read(directory, 'fixed', 'STATE_EXCHANGE')

    def test_cache_rejects_wrong_arrays_even_if_checksum_is_updated(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)/'cache'
            cache_write(directory, 'fixed', 'STATE_EXCHANGE', self.arrays, {})
            arrays = dict(self.arrays)
            arrays['sample_ids'] = arrays['sample_ids'][::-1]
            np.savez_compressed(directory/'predictions.npz', **arrays)
            path = directory/'metadata.json'
            metadata = json.loads(path.read_text())
            metadata['archive_sha256'] = sha256(directory/'predictions.npz')
            write_json(path, metadata)
            with self.assertRaisesRegex(ValueError, 'sample_ids'):
                cache_read(directory, 'fixed', 'STATE_EXCHANGE')

    def test_cache_refuses_incomplete_existing_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)/'cache'
            directory.mkdir()
            with self.assertRaisesRegex(ValueError, 'incomplete'):
                cache_read(directory, 'fixed', 'STATE_EXCHANGE')
            with self.assertRaisesRegex(ValueError, 'already exists'):
                cache_write(directory, 'fixed', 'STATE_EXCHANGE', self.arrays, {})

    def test_cache_write_cannot_override_fixed_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            for key, value in [('split', 'val'), ('test_evaluated', False), ('optimizer_updates', 1),
                               ('optimizer_updates', False), ('freeze_id', 'other')]:
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'override fixed'):
                    cache_write(Path(folder)/key, 'fixed', 'STATE_EXCHANGE', self.arrays, {key:value})

    def test_validate_arrays_rejects_fields_shapes_dtypes_and_nonfinite(self):
        invalid = []
        arrays = dict(self.arrays, extra=np.asarray(1))
        invalid.append(arrays)
        arrays = dict(self.arrays)
        arrays['prediction'] = self.arrays['prediction'][:4]
        invalid.append(arrays)
        arrays = dict(self.arrays)
        arrays['prediction'] = np.broadcast_to(np.asarray(0., dtype=np.float64), (839, 12, 496, 1))
        invalid.append(arrays)
        arrays = dict(self.arrays)
        arrays['target'] = np.broadcast_to(np.asarray(np.nan, dtype=np.float32), (839, 12, 496, 1))
        invalid.append(arrays)
        arrays = dict(self.arrays)
        arrays['sample_ids'] = np.arange(4746, 5585, dtype=np.int32)
        invalid.append(arrays)
        arrays = dict(self.arrays)
        arrays['station_ids'] = np.arange(496, dtype=np.int32)
        invalid.append(arrays)
        arrays = dict(self.arrays)
        arrays['station_ids'] = np.zeros(496, dtype=np.int64)
        invalid.append(arrays)
        for number, arrays in enumerate(invalid):
            with self.subTest(number=number), self.assertRaises(ValueError):
                validate_arrays(arrays)


if __name__ == '__main__':
    unittest.main()
