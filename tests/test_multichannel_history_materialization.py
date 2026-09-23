"""Tests for cache-only v11a report-time multichannel materialization."""

import calendar
from copy import deepcopy
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.chronological.materialize_multichannel_history import (
    ChannelMoments, VerifiedMultichannelMonthCache, compare_raw,
    history_month_plan, load_protocol,
)


REPO = Path(__file__).resolve().parents[1]
PROTOCOL = REPO / 'experiments/chronological/multichannel_history_materialize_v11a.json'


def manifest_row(split='train', start='2023-01-31T23:00:00'):
    value = datetime.fromisoformat(start)
    return {
        'split': split, 'source_version': '8', 'sample_index': '7',
        'x_start': start,
        'x_end': (value + timedelta(minutes=55)).isoformat(),
    }


class ProtocolTests(unittest.TestCase):
    def test_protocol_freezes_information_boundary_and_authorization(self):
        protocol = load_protocol(PROTOCOL)
        self.assertTrue(protocol['information_boundary']['test_manifest_and_test_traffic_prohibited'])
        self.assertTrue(protocol['information_boundary']['validation_targets_prohibited'])
        self.assertTrue(protocol['authorization']['does_not_authorize_v10b'])
        self.assertEqual(protocol['authorization']['engineering_check_on_pass'],
                         'NO_SCIENTIFIC_AUTHORIZATION')
        self.assertTrue(protocol['materialization']['future_gap_and_Y_values_prohibited'])

    def test_protocol_rejects_semantic_or_gate_drift(self):
        protocol = json.loads(PROTOCOL.read_text())
        for mutation in ('semantics', 'gate'):
            changed = deepcopy(protocol)
            if mutation == 'semantics':
                changed['channel_semantics']['speed_units'] = 'mph'
            else:
                changed['acceptance']['flow_channel_exactly_matches_existing_history'] = False
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'protocol.json'
                path.write_text(json.dumps(changed))
                with self.assertRaises(ValueError):
                    load_protocol(path)


class HistoryPlanTests(unittest.TestCase):
    def test_plan_materializes_only_twelve_history_steps_across_months(self):
        plan = history_month_plan([
            manifest_row(start='2023-01-31T23:30:00')], 'train')
        self.assertEqual(plan[1]['history_positions'].tolist(), list(range(6)))
        self.assertEqual(plan[2]['history_positions'].tolist(), list(range(6, 12)))
        self.assertEqual(int(plan[1]['train_unique_slots'].sum()), 6)
        self.assertEqual(int(plan[2]['train_unique_slots'].sum()), 6)

    def test_plan_rejects_test_and_split_crossing(self):
        with self.assertRaises(ValueError):
            history_month_plan([manifest_row('test')], 'test')
        with self.assertRaisesRegex(ValueError, 'split boundary'):
            history_month_plan([
                manifest_row('train', '2023-08-31T23:30:00')], 'train')


class StatisticsTests(unittest.TestCase):
    def test_statistics_use_only_unique_nonnegative_train_values(self):
        values = np.full((2, 6, 3), 1e8, dtype=np.float32)
        values[:, :3] = [
            [[0, 10, 20], [2, 12, 22], [4, 14, 24]],
            [[-1, 20, 30], [np.nan, 22, 32], [8, 24, 34]],
        ]
        moments = ChannelMoments([10, 20])
        moments.update(values, np.array([1, 1, 1, 0, 0, 0], dtype=bool))
        result = moments.finish([7])
        self.assertEqual(result['unique_training_nominal_slots'], 3)
        self.assertEqual(result['valid_count'], [4, 6, 6])
        self.assertEqual(result['negative_count'], [1, 0, 0])
        self.assertEqual(result['nonfinite_count'], [1, 0, 0])
        self.assertEqual(result['zero_count'], [1, 0, 0])
        self.assertAlmostEqual(result['mean'][0], 3.5)

    def test_raw_comparison_preserves_nan_but_not_other_differences(self):
        left = np.array([1, np.nan, np.inf, -1], dtype=np.float32)
        right = np.array([1, np.nan, np.inf, 0], dtype=np.float32)
        self.assertEqual(compare_raw(left, right), (1, 4))


class CacheTests(unittest.TestCase):
    def make_cache(self, root):
        month, nodes = 1, 2
        slots = calendar.monthrange(2023, month)[1] * 288
        blob_dir = root / 'row_cache/blobs'
        blob_dir.mkdir(parents=True)
        rows = []
        expected = []
        header_hash = 'a' * 64
        for index in range(nodes):
            values = (np.arange(slots * 3, dtype=np.float32).reshape(slots, 3) + index)
            payload = values.astype('<f4').tobytes()
            digest = hashlib.sha256(payload).hexdigest()
            (blob_dir / f'{digest}.bin').write_bytes(payload)
            rows.append({
                'published_node_index': index, 'bytes': len(payload), 'sha256': digest,
                'identity': {
                    'dataset': 'gpxlcj/xtraffic', 'version': 8, 'year': 2023,
                    'month': month, 'raw_node_index': 10 + index,
                    'start': index * len(payload),
                    'end': (index + 1) * len(payload) - 1,
                    'header_sha256': header_hash,
                },
            })
            expected.append(values)
        manifest = {
            'month': month,
            'layout': {'shape': [16972, slots, 3], 'dtype': '<f4'},
            'header': {'sha256': header_hash}, 'rows': rows, 'failures': [],
        }
        path = root / 'source_month_01.json'
        path.write_text(json.dumps(manifest))
        return path, np.asarray(expected)

    def test_cache_loads_all_channels_and_rejects_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, expected = self.make_cache(root)
            cache = VerifiedMultichannelMonthCache(
                root, np.array([10, 11]), {path.name: hashlib.sha256(path.read_bytes()).hexdigest()})
            actual = cache.load(1)
            np.testing.assert_array_equal(actual, expected)
            self.assertEqual(cache.row_records, 2)
            first = next((root / 'row_cache/blobs').glob('*.bin'))
            first.write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                VerifiedMultichannelMonthCache(
                    root, np.array([10, 11]),
                    {path.name: hashlib.sha256(path.read_bytes()).hexdigest()}).load(1)


if __name__ == '__main__':
    unittest.main()
