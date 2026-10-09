"""Reject unsupported event matches and preserve the frozen train-time boundary."""
from datetime import datetime
from pathlib import Path
import tempfile
import unittest

from experiments.chronological.audit_incident_expansion import (
    cache_month, nearest_candidate, road_key, sensor_groups, training_window,
)
from src.utils.incident_corridor import sha256, write_json


class IncidentExpansionTests(unittest.TestCase):
    def test_road_parser_handles_interstates_and_state_routes_without_prefix_loss(self):
        self.assertEqual(road_key('I680-N', 'N'), (680, 'N'))
        self.assertEqual(road_key('SR24-W', 'W'), (24, 'W'))
        self.assertEqual(road_key('80.0', 'E'), (80, 'E'))
        with self.assertRaises(ValueError):
            road_key('I80-E', 'W')

    def test_train_window_preserves_gap_support_split_and_dst_exclusion(self):
        self.assertEqual(training_window(datetime(2023, 2, 1, 12, 5)), datetime(2023, 2, 1, 12, 10))
        for date in (datetime(2023, 1, 1), datetime(2023, 8, 31, 23, 10),
                     datetime(2023, 3, 12, 8), datetime(2023, 3, 11, 23, 30),
                     datetime(2023, 9, 1), datetime(2023, 11, 1)):
            self.assertIsNone(training_window(date))

    def test_geo_matching_respects_road_and_direction_and_flags_pm_conflict(self):
        sensors = [dict(station_id=1, Fwy='I80-E', Direction='E', Lat=38, Lng=-122, **{'Abs PM': 10}),
                   dict(station_id=2, Fwy='I80-W', Direction='W', Lat=38, Lng=-122, **{'Abs PM': 10})]
        groups = sensor_groups(sensors)
        event = dict(Fwy='80.0', Freeway_direction='E', Latitude=38, Longitude=-122, **{'Abs PM': 10})
        match = nearest_candidate(event, groups)
        self.assertEqual(match['station_id'], 1)
        self.assertEqual(match['nearest_sensor_km'], 0)
        self.assertTrue(match['postmile_agrees_within_10_source_units'])
        event['Abs PM'] = 100
        self.assertFalse(nearest_candidate(event, groups)['postmile_agrees_within_10_source_units'])
        event['Fwy'] = '680.0'
        self.assertIsNone(nearest_candidate(event, groups))
        event['Fwy'], event['Latitude'] = '80.0', 'nan'
        self.assertIsNone(nearest_candidate(event, groups))

    def test_cache_verification_detects_tampering_without_decoding_traffic(self):
        workspace = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix='incident-audit-test-', dir=workspace) as temp:
            root = Path(temp)
            self.assertEqual(root.resolve().parent, workspace)
            folder = root / 'row_cache/blobs'
            folder.mkdir(parents=True)
            payload = b'\0' * (8928 * 12)
            pending = folder / 'pending.bin'
            pending.write_bytes(payload)
            digest = sha256(pending)
            blob = folder / (digest + '.bin')
            pending.rename(blob)
            row = dict(published_node_index=0, bytes=len(payload), sha256=digest,
                       identity=dict(dataset='gpxlcj/xtraffic', version=8, year=2023, month=1,
                                     file_name='year_2023/year_2023/2023_p01.npy', raw_node_index=2,
                                     start=128+2*len(payload), end=128+3*len(payload)-1,
                                     header_sha256='header'))
            path = root / 'source_month_01.json'
            write_json(path, dict(month=1, layout=dict(shape=[16972, 8928, 3], dtype='<f4', data_offset=128),
                                 header=dict(sha256='header'), rows=[row]))
            result = cache_month(root, 1, [2], sha256(path))
            self.assertEqual(result['station_rows_verified'], 1)
            blob.write_bytes(b'1' + payload[1:])
            with self.assertRaisesRegex(ValueError, 'checksum'):
                cache_month(root, 1, [2], sha256(path))
            with self.assertRaisesRegex(ValueError, 'Training months only'):
                cache_month(root, 9, [2], sha256(path))


if __name__ == '__main__':
    unittest.main()
