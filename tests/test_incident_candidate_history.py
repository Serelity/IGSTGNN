"""Candidate selection, time leakage, deduplication and observation diagnostics."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experiments.chronological.prepare_incident_candidate_history import (
    extract_node_x, history_plan, road_order, select_events,
)
from src.utils.incident_candidate_history import CandidateHistory, SCHEMA, station_diagnostics
from src.utils.incident_corridor import read_json, sha256, write_json, write_rows

REPO = Path(__file__).resolve().parents[1]
PROTOCOL = read_json(REPO / 'experiments/chronological/incident_candidate_history_v1.json')


class CandidateHistoryTests(unittest.TestCase):
    def sample(self):
        sensors = [dict(station_id=4, Fwy='I80-E', Direction='E', Lat=38., Lng=-122., **{'Abs PM': 10.})]
        raw = dict(incident_id='event1', dt='02/01/2023 12:01:00', Fwy='80.0', Freeway_direction='E',
                   Latitude='38', Longitude='-122', **{'Abs PM': '10'})
        candidate = dict(source_row_index='0', incident_id='event1', report_time='2023-02-01T12:01:00',
                         t0='2023-02-01T12:05:00', identity_conflict='False', nearest_sensor_km='0',
                         source_postmile_difference='0', road='I80-E', station_id='4')
        return sensors, raw, candidate

    def test_same_history_different_events_stay_separate_and_future_is_rejected(self):
        sensors, raw, candidate = self.sample()
        events, _, _ = select_events([candidate], [raw], sensors, PROTOCOL['selection'], {})
        second = dict(events[0], incident_id='event2')
        labels, index = history_plan(events + [second])
        self.assertEqual(len(labels), 12)
        np.testing.assert_array_equal(index[0], index[1])
        self.assertEqual(str(labels[-1]), '2023-02-01T11:55')
        bad = dict(second, t0='2023-09-01T00:05:00', report_time='2023-09-01T00:01:00')
        with self.assertRaisesRegex(ValueError, 'train-time'):
            history_plan([bad])

    def test_selection_replays_identity_and_excludes_conflicting_ids(self):
        sensors, raw, c = self.sample()
        rows = [raw, dict(raw, incident_id='event2'), dict(raw, Latitude='37')]
        cs = [c, dict(c, incident_id='event2', source_row_index='1')]
        events, excluded, duplicates = select_events(cs, rows, sensors, PROTOCOL['selection'], {})
        self.assertEqual([e['incident_id'] for e in events], ['event2'])
        self.assertEqual(excluded[0]['reason'], 'conflicting_source_id_metadata')
        self.assertEqual(duplicates, 1)
        c['t0'] = '2023-02-01T12:10:00'
        with self.assertRaisesRegex(ValueError, 'clock replay'):
            select_events([c], [raw], sensors, PROTOCOL['selection'], {})

    def test_radius_fixed_without_traffic_and_identical_duplicate_is_retained_once(self):
        sensors, raw, c = self.sample()
        outside = dict(c, incident_id='far', source_row_index='2', nearest_sensor_km='1.01')
        events, excluded, duplicates = select_events([c, outside], [raw, dict(raw)], sensors, PROTOCOL['selection'], {})
        self.assertEqual(len(events), 1)
        self.assertEqual(duplicates, 1)
        self.assertEqual(excluded[0]['reason'], 'outside_fixed_1km_candidate_radius')

    def test_only_selected_X_slots_are_indexed_from_source(self):
        with tempfile.TemporaryDirectory(prefix='candidate-test-', dir=REPO) as tmp:
            root = Path(tmp)
            self.assertEqual(root.resolve().parent, REPO)
            data = np.arange(78, dtype='<f4').reshape(26, 3)
            data[12:] = np.nan
            path = root / 'row.bin'
            data.tofile(path)
            original = np.memmap
            calls = []

            class Guard:
                def __init__(self, *args, **kwargs):
                    self.array = original(*args, **kwargs)
                    self._mmap = self.array._mmap

                def __getitem__(self, slots):
                    if np.any(slots >= 12):
                        raise AssertionError('Future value indexed')
                    calls.append(slots.tolist())
                    return self.array[slots]

            with patch('experiments.chronological.prepare_incident_candidate_history.np.memmap', Guard):
                result = extract_node_x(path, 26, np.arange(12))
            np.testing.assert_array_equal(result, data[:12])
            self.assertEqual(calls, [list(range(12))])
            with self.assertRaises(ValueError):
                extract_node_x(path, 26, [0, 0])

    def observations(self):
        labels = np.array(['2023-02-01T00:00', '2023-02-01T00:05', '2023-02-01T00:10',
                           '2023-06-01T00:00', '2023-06-01T00:05', '2023-06-01T00:10'], dtype='datetime64[m]')
        o = np.array([1, 2, 3, 1, 2, 3.])
        v = np.array([3, 2, 1, 3, 2, 1.])
        q = o*v*2
        q[3:] *= 2
        rules = dict(PROTOCOL['diagnostics'], minimum_positive_slots_per_block=2)
        return np.stack([q, o, v], axis=-1), labels, rules

    def test_early_fit_is_not_refit_using_late_training_block(self):
        x, labels, rules = self.observations()
        result = station_diagnostics(x, labels, rules)
        self.assertEqual(result['alpha_early_source_units'], 2)
        self.assertEqual(result['late_to_early_ratio_median'], 2)
        self.assertAlmostEqual(result['late_symmetric_relative_error_median'], 2/3)
        x[3:, 0] *= 10
        self.assertEqual(station_diagnostics(x, labels, rules)['alpha_early_source_units'], 2)

    def test_relative_diagnostics_are_invariant_to_positive_unit_scaling(self):
        x, labels, rules = self.observations()
        a = station_diagnostics(x, labels, rules)
        b = station_diagnostics(x * [12, 100, 1.609344], labels, rules)
        for key in ('late_to_early_ratio_median', 'late_symmetric_relative_error_median', 'occupancy_speed_rank_correlation'):
            self.assertAlmostEqual(a[key], b[key])
        self.assertFalse(b['physics_certified'])

    def test_zero_missing_and_constant_observations_are_not_physical_certification(self):
        x, labels, rules = self.observations()
        x[:] = 0
        x[0, 0] = np.nan
        result = station_diagnostics(x, labels, rules)
        self.assertEqual(result['joint_usable_slots'], 5)
        self.assertEqual(result['positive_relation_slots'], 0)
        self.assertIsNone(result['alpha_early_source_units'])
        self.assertIsNone(result['occupancy_speed_rank_correlation'])
        self.assertIsNone(result['speed_reference'])

    def test_coincident_stations_are_grouped_without_zero_length_connections(self):
        sensors, _, _ = self.sample()
        sensors += [dict(sensors[0], station_id=5), dict(sensors[0], station_id=6, **{'Abs PM': 11.})]
        result = road_order(sensors)[0]
        self.assertEqual(result['coincident_groups'], 1)
        self.assertEqual(result['between_group_order_candidates'], 1)
        self.assertEqual(result['ordered_groups'][0]['station_ids'], [4, 5])
        self.assertFalse(result['direct_connections_certified'])

    def test_reader_roundtrip_and_checksum_rejection(self):
        with tempfile.TemporaryDirectory(prefix='candidate-reader-test-', dir=REPO) as tmp:
            root = Path(tmp)
            self.assertEqual(root.resolve().parent, REPO)
            x = np.arange(72, dtype=np.float32).reshape(12, 2, 3)
            for name, array in {'history_values': x, 'history_usable': np.ones_like(x, dtype=bool),
                                'history_index': np.arange(12, dtype=np.int32)[None],
                                'history_labels': np.arange(12).astype('datetime64[m]'),
                                'station_ids': np.array([1, 2])}.items():
                np.save(root / (name + '.npy'), array, allow_pickle=False)
            write_rows(root / 'train_events.csv', [{'incident_id': 'one'}])
            summary = {'schema': SCHEMA, 'status': 'CANDIDATE_TRAIN_X_PACK_COMPLETE',
                       'outputs_sha256': {p.name: sha256(p) for p in root.iterdir()}}
            write_json(root / 'summary.json', summary)
            reader = CandidateHistory(root)
            try:
                np.testing.assert_array_equal(reader.window(0)['history_source_units'], x)
            finally:
                reader.close()
            (root / 'train_events.csv').write_text('modified', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                CandidateHistory(root)


if __name__ == '__main__':
    unittest.main()
