"""Server-only validation archive and serialization regression tests."""
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experiments.observed_state_gw17.common import load_config, sha256, write_json
from experiments.observed_state_gw17.evaluate import (ARCHIVE_SCHEMA, canonical_arrays,
    read_archive, render_report, require_reference_axes, validation_replay)
from experiments.observed_state_gw17.objective import archive_diagnostics, metric_contract


class GW17ArchiveTests(unittest.TestCase):
    def guards(self):
        stack = ExitStack()
        stack.enter_context(patch('experiments.observed_state_gw17.evaluate.validate_identity'))
        stack.enter_context(patch('experiments.observed_state_gw17.evaluate.source_hashes', return_value={}))
        stack.enter_context(patch('experiments.observed_state_gw17.evaluate.verify_reference'))
        return stack

    def fixture(self, directory):
        config = load_config()
        config.update(nodes=2, horizon=2, history=3)
        target = np.array([[[[100.], [400.]], [[120.], [420.]]]], dtype=np.float32)
        prediction = target+1
        arrays = canonical_arrays(dict(prediction=prediction, target=target,
            reference=np.full_like(target, 100), regional=prediction.mean(axis=2, keepdims=True),
            alpha=np.zeros((1, 2, 1, 1), np.float32), beta=np.zeros_like(target),
            support=np.array([[True, False]]), available=np.array([True]),
            station_ids=np.array([10, 20], np.int64), sample_ids=np.array([3909], np.int64),
            assignment=np.array([0, 0], np.int64), state_rms_batch=np.zeros((1, 2, 2), np.float32),
            history_prediction=prediction.copy(), history_target=target.copy(),
            history_valid=np.array([[[[True], [True]], [[False], [True]]]])))
        identity = dict(identity_sha256='fixture', config=config, engineering_only=False,
            execution_source={}, gw16_reference=dict(directory='old'),
            data=dict(loader_batching=dict(val_loader=dict(samples=1, batches=1)),
                station_ids=[10, 20], partition=dict(assignment=[0, 0])))
        np.savez_compressed(directory/'predictions.npz', **arrays)
        metadata = dict(schema=ARCHIVE_SCHEMA, identity=identity, identity_sha256='fixture',
            metric_contract=metric_contract(config), engineering_only=False,
            split='val', test_evaluated=False, arm='INNOVATION_STATE',
            gw16_reference=identity['gw16_reference'], archive_sha256=sha256(directory/'predictions.npz'),
            validation_replay=dict(passed=True), diagnostics=archive_diagnostics(arrays, config))
        write_json(directory/'metadata.json', metadata)
        return arrays, metadata

    def test_all_fields_are_canonical_before_and_after_npz_roundtrip(self):
        # A transpose creates the same class of non-C model output that caused
        # the previous native FP32 cache mismatch. Freeze before any metric.
        source = np.arange(2*3*4, dtype=np.float32).reshape(2, 3, 4).transpose(0, 2, 1)
        self.assertFalse(source.flags.c_contiguous)
        canonical = canonical_arrays(dict(prediction=source))
        self.assertTrue(canonical['prediction'].flags.c_contiguous)
        self.assertEqual(canonical['prediction'].dtype, source.dtype)
        np.testing.assert_array_equal(canonical['prediction'], source)
        self.assertFalse(np.shares_memory(canonical['prediction'], source))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'roundtrip.npz'
            np.savez_compressed(path, **canonical)
            with np.load(path, allow_pickle=False) as payload:
                restored = payload['prediction']
            self.assertTrue(restored.flags.c_contiguous)
            np.testing.assert_array_equal(restored, canonical['prediction'])

    def test_valid_archive_recomputes_future_and_masked_history(self):
        with tempfile.TemporaryDirectory() as temporary:
            arrays, metadata = self.fixture(Path(temporary))
            with self.guards():
                restored, record = read_archive(temporary)
            self.assertEqual(record['diagnostics'], metadata['diagnostics'])
            self.assertEqual(record['diagnostics']['history_preupdate']['valid_count'], 3)
            for key in arrays:
                np.testing.assert_array_equal(restored[key], arrays[key])

    def test_checksum_change_is_rejected_on_every_read(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.fixture(directory)
            with self.guards():
                read_archive(directory)
                with (directory/'predictions.npz').open('ab') as stream:
                    stream.write(b'changed')
                with self.assertRaisesRegex(ValueError, 'checksum'):
                    read_archive(directory)

    def test_edited_history_metrics_rejected_even_when_future_unchanged(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            _, metadata = self.fixture(directory)
            metadata['diagnostics']['history_preupdate']['mae'] += 1
            write_json(directory/'metadata.json', metadata)
            with self.guards(), self.assertRaisesRegex(ValueError, 'diagnostics'):
                read_archive(directory)

    def test_rehashed_station_order_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            arrays, metadata = self.fixture(directory)
            arrays['station_ids'] = arrays['station_ids'][::-1].copy()
            np.savez_compressed(directory/'predictions.npz', **arrays)
            metadata['archive_sha256'] = sha256(directory/'predictions.npz')
            write_json(directory/'metadata.json', metadata)
            with self.guards(), self.assertRaisesRegex(ValueError, 'axis'):
                read_archive(directory)

    def test_mask_dtype_must_not_silently_become_float(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            arrays, metadata = self.fixture(directory)
            arrays['history_valid'] = arrays['history_valid'].astype(np.float32)
            np.savez_compressed(directory/'predictions.npz', **arrays)
            metadata['archive_sha256'] = sha256(directory/'predictions.npz')
            write_json(directory/'metadata.json', metadata)
            with self.guards(), self.assertRaisesRegex(ValueError, 'boolean'):
                read_archive(directory)

    def test_float_precision_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            arrays, metadata = self.fixture(directory)
            arrays['history_prediction'] = arrays['history_prediction'].astype(np.float64)
            np.savez_compressed(directory/'predictions.npz', **arrays)
            metadata['archive_sha256'] = sha256(directory/'predictions.npz')
            write_json(directory/'metadata.json', metadata)
            with self.guards(), self.assertRaisesRegex(ValueError, 'float32'):
                read_archive(directory)

    def test_test_scope_is_rejected_even_for_valid_archive(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            _, metadata = self.fixture(directory)
            metadata.update(split='test', test_evaluated=True)
            write_json(directory/'metadata.json', metadata)
            with self.guards(), self.assertRaisesRegex(ValueError, 'validation-only'):
                read_archive(directory)

    def test_source_drift_is_checked_before_cached_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            self.fixture(Path(temporary))
            with self.guards(), patch('experiments.observed_state_gw17.evaluate.source_hashes',
                    return_value={'changed':'bytes'}):
                with self.assertRaisesRegex(ValueError, 'execution source'):
                    read_archive(temporary)

    def test_reference_requires_exact_target_dtype_values_and_order(self):
        left = {key:np.arange(3, dtype=np.int64) for key in
            ('target', 'sample_ids', 'station_ids', 'assignment', 'support', 'available')}
        require_reference_axes(left, deepcopy(left))
        for key in left:
            with self.subTest(key=key):
                right = deepcopy(left)
                right[key] = right[key][::-1]
                with self.assertRaisesRegex(ValueError, key):
                    require_reference_axes(left, right)
        right = deepcopy(left)
        right['target'] = right['target'].astype(np.float32)
        with self.assertRaisesRegex(ValueError, 'target'):
            require_reference_axes(left, right)

    def test_replay_checks_history_before_update_and_its_count(self):
        diagnostics = dict(overall=dict(mae=2., rmse=3., mape_percent=4., valid_count=9,
            mape_valid_count=7), region_mae=1., reference=dict(mae=5.),
            history_preupdate=dict(mae=6., valid_count=11))
        expected = dict(diagnostics['overall'], region_mae=1., reference_mae=5.,
            history_preupdate_mae=6., history_valid_count=11)
        self.assertTrue(validation_replay(diagnostics, expected)['passed'])
        expected['history_preupdate_mae'] += .01
        self.assertFalse(validation_replay(diagnostics, expected)['passed'])
        expected['history_valid_count'] = 10
        with self.assertRaisesRegex(ValueError, 'historical target count'):
            validation_replay(diagnostics, expected)

    def test_report_labels_historical_reference_and_engineering_scope(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            _, metadata = self.fixture(directory)
            metadata['checkpoint_role'] = dict(completed_training_epochs=100, epoch=99,
                registered_parameters=10000)
            reference = deepcopy(metadata)
            del reference['diagnostics']['history_preupdate']
            render_report(directory, {'INNOVATION_STATE':metadata}, 'best', reference)
            text = (directory/'report.md').read_text(encoding='utf-8')
            self.assertIn('| GW16_REFERENCE |', text)
            self.assertIn('不是本轮等预算训练对照', text)
            self.assertIn('不自动打开或反复使用该测试', text)
            render_report(directory, {'INNOVATION_STATE':metadata}, 'best', reference,
                engineering_only=True)
            text = (directory/'report.md').read_text(encoding='utf-8')
            self.assertNotIn('| GW16_REFERENCE |', text)
            self.assertIn('工程小样本仅核对自身流程', text)


if __name__ == '__main__':
    unittest.main()
