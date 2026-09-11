"""Offline safety and statistics boundaries for the conditional v8 builder."""
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from experiments.chronological.build_data import (
    TrainMoments, month_plan, select_development_rows, window_slots,
)
from experiments.chronological.source_v8 import (
    RangeProtocolError, RowCache, V8Reader, parse_header, validate_range_headers,
)


def row(split='train', start='2023-01-31T23:00:00'):
    from datetime import datetime, timedelta
    dt = datetime.fromisoformat(start)
    return {'split': split, 'source_version': '8', 'sample_index': '1',
            'x_start': start, 'x_end': (dt + timedelta(minutes=55)).isoformat(),
            'y_start': (dt + timedelta(minutes=70)).isoformat(),
            'y_end': (dt + timedelta(minutes=125)).isoformat()}


class WindowTests(unittest.TestCase):
    def test_cross_month_preserves_all_26_slots_and_gap(self):
        slots = window_slots(row())
        self.assertEqual(slots[11].isoformat(), '2023-01-31T23:55:00')
        self.assertEqual(slots[12].isoformat(), '2023-02-01T00:00:00')
        self.assertEqual(slots[14].isoformat(), '2023-02-01T00:10:00')
        self.assertEqual(slots[25].isoformat(), '2023-02-01T01:05:00')
        plan = month_plan([row(), row()], 'train')
        self.assertEqual(int(plan[1]['train_x_union'].sum()), 12)
        self.assertEqual(int(plan[2]['train_x_union'].sum()), 0)
        self.assertEqual(plan[2]['month_slots'][:14].tolist(), list(range(14)))

    def test_test_manifest_does_not_become_a_download_request(self):
        rows = select_development_rows([row(), row('test', '2023-12-01T00:00:00')])
        self.assertEqual(set(month_plan(rows['train'], 'train')), {1, 2})
        self.assertEqual(rows['val'], [])
        with self.assertRaises(ValueError):
            month_plan([row('val', '2023-10-31T23:00:00')], 'val')

    def test_rejects_misaligned_or_changed_latency_slices(self):
        changed = row()
        changed['y_start'] = '2023-02-01T00:05:00'
        with self.assertRaises(ValueError):
            window_slots(changed)
        with self.assertRaises(ValueError):
            window_slots(row(start='2023-01-01T01:01:00'))


class StatisticsTests(unittest.TestCase):
    def test_only_unique_train_x_valid_values_fit_scaler(self):
        # Duplicate event rows must not double-weight [0, 2, 4]. Invalid and
        # future values, including the latency gap and Y, never enter fit.
        repeated = row(start='2023-01-01T00:00:00')
        mask = month_plan([repeated, repeated], 'train')[1]['train_x_union']
        data = np.full((2, 8928), np.nan, dtype=np.float32)
        data[0, :4] = [0, 2, 4, -1]
        data[0, 4] = np.inf
        data[:, 12:26] = 1e8
        data[1, 0] = -3
        stats = TrainMoments([10, 20])
        stats.update(data, mask)
        result = stats.finish([repeated])
        self.assertEqual(result['mean'], 2.0)
        self.assertAlmostEqual(result['std'], np.sqrt(8 / 3))
        self.assertEqual(result['valid_unique_input_count_per_station'], [3, 0])
        self.assertEqual(result['node_fill_mean'], [2.0, None])
        self.assertEqual(result['all_missing_station_ids'], [20])
        self.assertEqual(result['zero_training_input_count'], 1)
        self.assertEqual(result['unique_training_input_keys'], 24)

    def test_validation_month_mask_is_empty_and_cannot_alter_fit(self):
        mask = month_plan([row('val', '2023-09-01T01:00:00')], 'val')[9]['train_x_union']
        stats = TrainMoments([10])
        stats.update(np.array([[0, 2, 4]], dtype=np.float32), np.ones(3, bool))
        before = stats.finish([])
        stats.update(np.full((1, 8640), 1e9, dtype=np.float32), mask)
        self.assertEqual(stats.finish([]), before)


class SourceTests(unittest.TestCase):
    def test_http_failure_retries_without_consuming_full_body_or_leaking_url(self):
        class Response:
            status_code = 200
            headers = {'Content-Length': '1000000000'}
            raw = io.BytesIO(b'not-a-valid-range')

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        response = Response()
        reader = V8Reader()
        with patch('experiments.chronological.source_v8.requests.get', return_value=response) as external, \
                patch('experiments.chronological.source_v8.time.sleep'):
            with self.assertRaises(RuntimeError) as caught:
                reader.read_range('https://example.invalid/private-token', 0, 9, 100)
        self.assertNotIn('private-token', str(caught.exception))
        self.assertEqual(response.raw.tell(), 0)
        self.assertEqual(external.call_count, 4)
        self.assertEqual(reader.stats()['range_failures'], 4)

    def test_truncated_range_body_is_retried_then_rejected(self):
        class Response:
            status_code = 206
            headers = {'Content-Range': 'bytes 0-9/100', 'Content-Length': '10'}

            def __enter__(self):
                self.raw = io.BytesIO(b'12345')
                return self

            def __exit__(self, *args):
                return False

        reader = V8Reader()
        with patch('experiments.chronological.source_v8.requests.get', return_value=Response()), \
                patch('experiments.chronological.source_v8.time.sleep'):
            with self.assertRaises(RuntimeError):
                reader.read_range('unused', 0, 9, 100)
        self.assertEqual(reader.stats()['range_bytes_received'], 20)
        self.assertEqual(reader.stats().get('range_successes', 0), 0)

    def test_rejects_full_body_wrong_range_total_or_length_before_read(self):
        good = {'Content-Range': 'bytes 10-19/100', 'Content-Length': '10'}
        validate_range_headers(206, good, 10, 19, 100)
        for status, headers in [(200, good), (206, {**good, 'Content-Range': 'bytes 0-9/100'}),
                                (206, {**good, 'Content-Range': 'bytes 10-19/101'}),
                                (206, {**good, 'Content-Length': '11'})]:
            with self.subTest(status=status, headers=headers), self.assertRaises(RangeProtocolError):
                validate_range_headers(status, headers, 10, 19, 100)

    def test_cache_reuses_verified_content_and_detects_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = RowCache(Path(directory))
            identity = {'version': 8, 'month': 1, 'raw_node_index': 27, 'start': 10, 'end': 19}
            payload = b'0123456789'
            with patch.object(V8Reader, 'read_range', return_value=payload) as external:
                result, record = cache.read(identity, lambda: V8Reader().read_range('unused', 10, 19))
                self.assertEqual(result, payload)
                self.assertEqual(external.call_count, 1)
                again, cached = cache.read(identity, lambda: self.fail('valid cache must avoid network'))
                self.assertEqual(again, payload)
                self.assertTrue(cached['cache_hit'])
            blob = Path(directory) / 'blobs' / (record['sha256'] + '.bin')
            blob.write_bytes(b'corrupted!')
            with self.assertRaisesRegex(ValueError, 'cache'):
                cache.read(identity, lambda: self.fail('corruption must fail closed'))

    def test_network_failure_is_not_cached_as_missing_flow(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = RowCache(Path(directory))
            identity = {'version': 8, 'month': 1, 'raw_node_index': 27, 'start': 0, 'end': 3}
            def fail():
                raise OSError('network failed')
            with self.assertRaises(OSError):
                cache.read(identity, fail)
            self.assertEqual(list((Path(directory) / 'requests').glob('*.json')), [])

    def test_header_rejects_wrong_axis_dtype_or_month_length(self):
        stream = io.BytesIO()
        np.lib.format.write_array_header_1_0(stream, {
            'descr': '<f4', 'fortran_order': False, 'shape': (16972, 8928, 3)})
        result = parse_header(stream.getvalue(), 1)
        self.assertEqual(result['data_offset'], 128)
        self.assertEqual(result['total_bytes'], 1818312320)
        with self.assertRaises(ValueError):
            parse_header(stream.getvalue(), 2)
        reader = V8Reader()
        with self.assertRaises(ValueError):
            reader.open_month(11, RowCache(Path(tempfile.gettempdir()) / 'not-created-v8-cache'))


if __name__ == '__main__':
    unittest.main()
