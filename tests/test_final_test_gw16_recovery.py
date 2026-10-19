"""Server-only regression tests for the immutable GW16 cache layout recovery.

These tests never open research data or run a forecasting model. Synthetic
callbacks make the recovery branch deterministic across PyTorch versions; one
test also replays the actual frozen author metric functions on varied arrays.
"""
import copy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch

from experiments.final_test_gw16_recovery import recover
from experiments.final_test_gw16.state import cache_read, cache_write, freeze_output
from experiments.original_baseline_gw16.evaluate import native_summary


def native_layout(value):
    """Independent construction of the frozen model's B,N,H storage order."""
    return value.transpose(0, 2, 1, 3).copy(order='C').transpose(0, 2, 1, 3)


def layout_summary(prediction, target):
    """A stable metric-shaped callback that distinguishes C and native storage."""
    native = prediction.strides[1] == prediction.itemsize
    value = 1. if native else 2.
    row = dict(mae=value, rmse=value+1., mape_percent=value+2.)
    return dict(pooled=dict(mae=value, rmse=value+1., mape_fraction=(value+2.)/100.,
                            mask_value=0.),
                per_horizon=[dict(lead=lead, **row) for lead in range(1, 13)],
                horizon_macro=dict(row), mask_value=0.)


class FinalTestGW16RecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        index = np.arange(839*12*496, dtype=np.float32).reshape(839, 12, 496, 1)
        cls.target = np.asarray(80.+(index % 139)*.31, dtype=np.float32)
        cls.prediction = np.asarray(cls.target+20.*np.sin(index*.01)+(index % 17)*.001,
                                    dtype=np.float32)
        cls.samples = np.arange(4746, 5585, dtype=np.int64)
        cls.stations = np.arange(496, dtype=np.int64)

    def arrays(self):
        return dict(prediction=self.prediction.copy(order='C'), target=self.target,
                    sample_ids=self.samples, station_ids=self.stations)

    def assert_audit(self, audit, restored):
        self.assertIs(audit['layout_restored'], restored)
        self.assertIs(audit['prediction_values_bitwise_unchanged'], True)
        self.assertIs(audit['strict_metrics_reproduced'], True)

    def assert_bits_equal(self, first, second):
        self.assertEqual(first.dtype, second.dtype)
        self.assertEqual(first.shape, second.shape)
        np.testing.assert_array_equal(first.view(np.uint32), second.view(np.uint32))

    def test_restore_preserves_every_float32_bit_and_native_stride_order(self):
        original = self.prediction.copy(order='C')
        original[0, 0, 0, 0] = np.float32(-0.)
        original[0, 0, 1, 0] = np.float32(0.)
        restored = recover.restore_native_layout(original)
        self.assert_bits_equal(original, restored)
        self.assertEqual(restored.strides, (12*496*4, 4, 12*4, 4))
        self.assertFalse(restored.flags.c_contiguous)
        again = recover.restore_native_layout(restored)
        self.assert_bits_equal(restored, again)
        self.assertEqual(again.strides, restored.strides)

    def test_already_matching_c_cache_keeps_existing_array_objects(self):
        arrays = self.arrays()
        metadata = dict(author_metrics=layout_summary(arrays['prediction'], arrays['target']))
        before = copy.deepcopy(metadata)
        repaired, audit = recover.repair_cached_arrays(
            arrays, metadata, 'ORIGINAL_IGSTGNN', layout_summary)
        for key in arrays:
            self.assertIs(repaired[key], arrays[key])
        self.assertEqual(metadata, before)
        self.assert_audit(audit, False)

    def test_original_cache_uses_only_fixed_native_layout_and_retains_axes(self):
        arrays = self.arrays()
        expected = native_layout(arrays['prediction'])
        metadata = dict(author_metrics=layout_summary(expected, arrays['target']))
        calls = []

        def summary(prediction, target):
            calls.append(prediction.strides)
            return layout_summary(prediction, target)

        repaired, audit = recover.repair_cached_arrays(
            arrays, metadata, 'ORIGINAL_IGSTGNN', summary)
        self.assertEqual(calls, [arrays['prediction'].strides, expected.strides])
        self.assertEqual(repaired['prediction'].strides, expected.strides)
        self.assert_bits_equal(repaired['prediction'], arrays['prediction'])
        for key in ('target', 'sample_ids', 'station_ids'):
            self.assertIs(repaired[key], arrays[key])
        self.assert_audit(audit, True)

    def test_original_still_rejects_a_metric_change_after_layout_recovery(self):
        arrays = self.arrays()
        metadata = dict(author_metrics=layout_summary(native_layout(arrays['prediction']), arrays['target']))
        metadata['author_metrics']['pooled']['mae'] = 12345.
        before = copy.deepcopy(metadata)
        with self.assertRaisesRegex(ValueError, r'(pooled|mae|author_metrics)'):
            recover.repair_cached_arrays(arrays, metadata, 'ORIGINAL_IGSTGNN', layout_summary)
        self.assertEqual(metadata, before)
        self.assertTrue(arrays['prediction'].flags.c_contiguous)

    def test_state_exchange_cannot_use_the_original_layout_repair(self):
        arrays = self.arrays()
        metadata = dict(author_metrics=layout_summary(native_layout(arrays['prediction']), arrays['target']))
        calls = []

        def summary(prediction, target):
            calls.append(prediction.strides)
            return layout_summary(prediction, target)

        with self.assertRaises(ValueError):
            recover.repair_cached_arrays(arrays, metadata, 'STATE_EXCHANGE', summary)
        self.assertEqual(calls, [arrays['prediction'].strides])
        self.assertTrue(arrays['prediction'].flags.c_contiguous)

    def test_matching_state_exchange_cache_does_not_change_layout_or_values(self):
        arrays = self.arrays()
        metadata = dict(author_metrics=layout_summary(arrays['prediction'], arrays['target']))
        repaired, audit = recover.repair_cached_arrays(
            arrays, metadata, 'STATE_EXCHANGE', layout_summary)
        self.assertIs(repaired['prediction'], arrays['prediction'])
        self.assertEqual(repaired['prediction'].strides, arrays['prediction'].strides)
        self.assert_audit(audit, False)

    def test_recovery_leaves_frozen_record_metadata_and_npz_bytes_unchanged(self):
        arrays = self.arrays()
        arrays['prediction'] = native_layout(arrays['prediction'])
        metadata = dict(author_metrics=layout_summary(arrays['prediction'], arrays['target']))
        with TemporaryDirectory() as temporary:
            root = Path(temporary)/'frozen_test'
            freeze_id = freeze_output(root, dict(selection='validation_best', epoch=99))
            directory = root/'ORIGINAL_IGSTGNN'
            cache_write(directory, freeze_id, 'ORIGINAL_IGSTGNN', arrays, metadata)
            paths = [root/'freeze.json', directory/'metadata.json', directory/'predictions.npz']
            before = {path:hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
            saved_arrays, saved_metadata = cache_read(directory, freeze_id, 'ORIGINAL_IGSTGNN')
            recorded = json.loads((directory/'metadata.json').read_text(encoding='utf-8'))
            self.assertTrue(saved_arrays['prediction'].flags.c_contiguous)
            repaired, audit = recover.repair_cached_arrays(
                saved_arrays, saved_metadata, 'ORIGINAL_IGSTGNN', layout_summary)
            self.assert_audit(audit, True)
            self.assert_bits_equal(arrays['prediction'], repaired['prediction'])
            self.assertEqual(saved_metadata, recorded)
            self.assertEqual(before, {path:hashlib.sha256(path.read_bytes()).hexdigest() for path in paths})

    def test_actual_frozen_native_metrics_reproduce_after_npz_layout_roundtrip(self):
        previous_threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, previous_threads)
        torch.set_num_threads(3)
        arrays = self.arrays()
        arrays['prediction'] = native_layout(arrays['prediction'])
        expected = native_summary(arrays['prediction'], arrays['target'])
        with TemporaryDirectory() as temporary:
            path = Path(temporary)/'prediction.npz'
            np.savez_compressed(path, **arrays)
            with np.load(path, allow_pickle=False) as saved:
                loaded = {key:saved[key] for key in saved.files}
            restored = recover.restore_native_layout(loaded['prediction'])
            self.assert_bits_equal(restored, arrays['prediction'])
            self.assertEqual(native_summary(restored, loaded['target']), expected)
            # Some CPU versions reduce both layouts identically. Recovery must
            # accept either the already exact cache or the one fixed layout.
            repaired, audit = recover.repair_cached_arrays(
                loaded, dict(author_metrics=expected), 'ORIGINAL_IGSTGNN', native_summary)
            self.assertEqual(native_summary(repaired['prediction'], repaired['target']), expected)
            self.assertIs(audit['strict_metrics_reproduced'], True)
            self.assertIs(audit['prediction_values_bitwise_unchanged'], True)


if __name__ == '__main__':
    unittest.main()
