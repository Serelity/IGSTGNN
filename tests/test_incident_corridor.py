"""Check information boundaries, axis identity and incomplete physical evidence."""

import copy
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experiments.chronological.prepare_incident_corridors import prepare
from src.utils.incident_corridor import (
    CHANNELS, SCHEMA, CorridorHistory, history_diagnostics, read_json,
    select_corridors, sha256, validate_train_clock, write_json, write_rows,
)


REPO = Path(__file__).absolute().parents[1]


class CorridorTests(unittest.TestCase):
    def setUp(self):
        scratch = REPO / 'experiments/chronological_runs'
        scratch.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix='corridor_test_', dir=scratch)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data, self.multi, self.meta = [self.root / name for name in ('data', 'multi', 'meta')]
        for p in (self.data, self.multi, self.meta):
            p.mkdir()
        self.sensors = self.root / 'sensors.csv'
        self.selection = self.root / 'selection.json'
        self.ids = np.asarray([30, 10, 20, 40], dtype=np.int64)
        self.public = [dict(station_id=sid, Fwy='SR4-W' if sid != 40 else 'SR4-E',
                            Direction='W' if sid != 40 else 'E', County='Contra Costa',
                            Type='Mainline', **{'Abs PM': pm, 'Lat': 38., 'Lng': -122.})
                       for sid, pm in ((10, 0.), (20, 1.), (30, 2.), (40, 1.))]
        self.raw = [{('Fwy Name' if k == 'Fwy' else k): v for k, v in row.items()}
                    for row in self.public]
        ramp = dict(self.raw[0], station_id=90, Type='On Ramp')
        ramp['Abs PM'] = 1.5
        self.raw.append(ramp)
        write_rows(self.sensors, self.public)
        # The metadata subset uses TSV; do not reuse CSV serialization.
        import csv
        with (self.meta / 'source_sensor_subset.tsv').open('w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(self.raw[0]), delimiter='\t')
            writer.writeheader()
            writer.writerows(self.raw)
        write_json(self.meta / 'manifest.json', {
            'schema_version': 1, 'published_sensors_sha256': sha256(self.sensors),
            'files': {'source_sensor_subset.tsv': {
                'sha256': sha256(self.meta / 'source_sensor_subset.tsv'), 'license': 'fixture'}}})
        self.rows = []
        for i in range(2):
            start = datetime(2023, 1, 3, 8) + timedelta(minutes=i * 5)
            cutoff = start + timedelta(minutes=65)
            self.rows.append({'sample_index': str(100 + i), 'incident_id': 'same-incident',
                              'source_version': '8', 'split': 'train', 'x_start': start.isoformat(),
                              'x_end': (start + timedelta(minutes=55)).isoformat(),
                              't0': cutoff.isoformat(),
                              'report_time': (cutoff - timedelta(minutes=3)).isoformat()})
        write_rows(self.data / 'train_manifest.csv', self.rows)
        np.save(self.data / 'station_ids.npy', self.ids)
        source = np.arange(13 * 4 * 3, dtype=np.float32).reshape(13, 4, 3)
        source[0, 0, 1], source[0, 0, 2] = np.nan, -1
        self.history = np.stack([source[:12], source[1:13]])
        np.save(self.multi / 'train_history.npy', self.history)
        self.report = {'sample_indices': np.array([100, 101]), 'station_ids': self.ids,
                       'report_age_minutes': np.array([3., 3.], dtype=np.float32),
                       'forecast_tod': np.array([109, 110]), 'forecast_dow': np.array([2, 2]),
                       'distances': np.zeros((2, 4, 3), dtype=np.float32)}
        self.report['distances'][:, :3, 1] = .5
        np.savez(self.data / 'train_context.npz', **self.report)
        write_json(self.multi / 'train_multichannel_scaler.json', {
            'fit_scope': 'unique_train_X_12_steps_station_channel_finite_nonnegative',
            'channel_order': CHANNELS, 'station_ids': self.ids.tolist(),
            'fitted_sample_indices': [100, 101]})
        self.config = {'schema': SCHEMA,
                       'selection_scope': 'fixed_anchors_source_postmile_no_outcome_selection',
                       'corridors': [{'id': 'west', 'road': 'SR4-W', 'direction': 'W',
                                      'anchor_station_ids': [10, 20],
                                      'candidate_travel_postmile_order': 'descending',
                                      'extend_each_end_source_postmile': 2.}]}
        self.refresh_provenance()

    def refresh_provenance(self):
        write_json(self.data / 'summary.json', {
            'status': 'conditional_development', 'source_version': 8, 'build_complete': True,
            'X_slice': [0, 12], 'station_count': 4, 'split_counts': {'train': 2},
            'files': {name: sha256(self.data / name) for name in
                      ('station_ids.npy', 'train_manifest.csv')}})
        write_json(self.data / 'context_manifest.json', {
            'schema': 'report_location_v1', 'scope': 'conditional_development',
            'sources': {'sensors.csv': sha256(self.sensors)},
            'outputs': {'train_context.npz': sha256(self.data / 'train_context.npz')}})
        write_json(self.multi / 'summary.json', {
            'status': 'MULTICHANNEL_HISTORY_MATERIALIZATION_COMPLETE',
            'protocol_id': 'contra_v8_multichannel_history_materialize_v11a',
            'engineering_check': False, 'acceptance': {'gate_passed': True},
            'inputs': {'data_summary_sha256': sha256(self.data / 'summary.json')},
            'channel_semantics': {'source_order': CHANNELS, 'flow_units': 'source_units'},
            'splits': {'train': {'shape': [2, 12, 4, 3], 'flow_anchor_mismatches': 0}},
            'outputs': {name: {'sha256': sha256(self.multi / name)} for name in
                        ('train_history.npy', 'train_multichannel_scaler.json')}})
        self.config['data_summary_sha256'] = sha256(self.data / 'summary.json')
        self.config['context_manifest_sha256'] = sha256(self.data / 'context_manifest.json')
        write_json(self.selection, self.config)

    def build(self, name='out'):
        return prepare(self.data, self.multi, self.sensors, self.meta,
                       self.selection, self.root / name)

    def test_pack_roundtrip_preserves_axes_raw_missing_zero_and_report_identity(self):
        result = self.build()
        package = CorridorHistory(self.root / 'out')
        item = package.window(0, 'west')
        self.assertEqual(item['station_ids'].tolist(), [30, 20, 10])
        self.assertEqual(item['history_source_units'].shape, (12, 3, 3))
        np.testing.assert_equal(item['history_source_units'], self.history[0][:, [0, 2, 1]])
        self.assertTrue(item['value_usable'][0, 0, 0])  # true zero is retained
        self.assertFalse(item['value_usable'][0, 0, 1])
        self.assertFalse(item['value_usable'][0, 0, 2])
        self.assertTrue(np.isnan(item['history_source_units'][0, 0, 1]))
        self.assertEqual(item['history_source_units'][0, 0, 2], -1)
        self.assertEqual(item['sample_index'], 100)
        self.assertEqual(result['history_diagnostics']['unique_train_history_labels'], 13)
        self.assertEqual(result['corridors'][0]['report_supported_train_windows'], 2)
        self.assertEqual(result['corridors'][0]['report_supported_distinct_incident_ids'], 1)
        self.assertFalse(item['physics_ready'])

    def test_does_not_open_raw_flow_validation_or_test_files(self):
        opened = []
        original = Path.open
        original_load = np.load
        def guarded(path, *args, **kwargs):
            opened.append(path.name)
            if path.name.startswith(('val_', 'test_')) or path.name.endswith('_flow.npy'):
                raise AssertionError('Forbidden data file opened: ' + str(path))
            return original(path, *args, **kwargs)
        def guarded_load(path, *args, **kwargs):
            name = Path(path).name
            if name.startswith(('val_', 'test_')) or name.endswith('_flow.npy'):
                raise AssertionError('Forbidden array opened: ' + str(path))
            return original_load(path, *args, **kwargs)
        with patch.object(Path, 'open', guarded), patch.object(np, 'load', guarded_load):
            self.build()
        self.assertNotIn('train_flow.npy', opened)
        # None of those files exists in the fixture: no dependency on future targets.
        self.assertFalse((self.data / 'val_flow.npy').exists())

    def test_selection_is_independent_of_csv_order_and_marks_ramps(self):
        corridors, union, _ = select_corridors(self.config, self.ids, self.public[::-1], self.raw)
        self.assertEqual(union.tolist(), [0, 1, 2])
        c = corridors[0]
        self.assertEqual(c['station_ids'], [30, 20, 10])
        self.assertEqual(c['candidate_segments'][0]['known_nonmainline'][0]['station_id'], 90)
        for segment in c['candidate_segments']:
            self.assertFalse(segment['direct_connection_certified'])
            self.assertIsNone(segment['physical_length_km'])
        config = copy.deepcopy(self.config)
        config['corridors'][0]['candidate_travel_postmile_order'] = 'ascending'
        self.assertEqual(select_corridors(config, self.ids, self.public, self.raw)[0][0]['station_ids'],
                         [10, 20, 30])

    def test_coincident_and_hidden_mainline_are_flagged_not_certified(self):
        public, raw = copy.deepcopy(self.public), copy.deepcopy(self.raw)
        for row in public + raw:
            if int(row['station_id']) == 30:
                row['Abs PM'] = 1.
        extra = dict(raw[0], station_id=91)
        extra['Abs PM'] = .5
        raw.append(extra)
        c = select_corridors(self.config, self.ids, public, raw)[0][0]
        self.assertIn('coincident_postmile', c['candidate_segments'][0]['metadata_flags'])
        self.assertIn(91, c['candidate_segments'][1]['additional_source_mainline_ids'])

    def test_checksum_corruption_and_existing_output_are_rejected(self):
        self.build()
        with self.assertRaises(FileExistsError):
            self.build()
        with (self.multi / 'train_history.npy').open('ab') as f:
            f.write(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            self.build('corrupt')
        self.assertFalse((self.root / 'corrupt').exists())

    def test_reordered_report_and_multichannel_provenance_are_rejected(self):
        self.report['sample_indices'] = self.report['sample_indices'][::-1]
        np.savez(self.data / 'train_context.npz', **self.report)
        self.refresh_provenance()
        with self.assertRaisesRegex(ValueError, 'sample order'):
            self.build()
        self.report['sample_indices'] = self.report['sample_indices'][::-1]
        np.savez(self.data / 'train_context.npz', **self.report)
        scaler = read_json(self.multi / 'train_multichannel_scaler.json')
        scaler['station_ids'] = scaler['station_ids'][::-1]
        write_json(self.multi / 'train_multichannel_scaler.json', scaler)
        self.refresh_provenance()
        with self.assertRaisesRegex(ValueError, 'station/sample order'):
            self.build()

    def test_history_clock_rejects_nontraining_future_and_duplicate_samples(self):
        for mutation in ('split', 'future', 'duplicate'):
            rows = copy.deepcopy(self.rows)
            if mutation == 'split':
                rows[0]['split'] = 'val'
            elif mutation == 'future':
                rows[0]['x_start'] = rows[0]['t0']
            else:
                rows[1]['sample_index'] = rows[0]['sample_index']
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_train_clock(rows, self.report)

    def test_overlap_conflicts_and_extra_time_slots_are_rejected(self):
        self.history[1, 0, 0, 0] += 1
        np.save(self.multi / 'train_history.npy', self.history)
        self.refresh_provenance()
        with self.assertRaisesRegex(ValueError, 'conflicting'):
            self.build()
        labels = validate_train_clock(self.rows, self.report)
        with self.assertRaisesRegex(ValueError, 'axes'):
            history_diagnostics(np.zeros((2, 26, 4, 3)), labels)

    def test_bad_metadata_anchor_and_duplicate_ids_are_rejected(self):
        config = copy.deepcopy(self.config)
        config['corridors'][0]['anchor_station_ids'] = [10, 40]
        with self.assertRaisesRegex(ValueError, 'Anchor road'):
            select_corridors(config, self.ids, self.public, self.raw)
        with self.assertRaisesRegex(ValueError, 'Duplicate metadata'):
            select_corridors(self.config, self.ids, self.public + self.public[:1], self.raw)
        raw = copy.deepcopy(self.raw)
        raw[0]['Lat'] = 39
        with self.assertRaisesRegex(ValueError, 'coordinate'):
            select_corridors(self.config, self.ids, self.public, raw)

    def test_partial_failure_never_publishes_completed_package(self):
        with patch('experiments.chronological.prepare_incident_corridors.np.save',
                   side_effect=OSError('simulated disk failure')):
            with self.assertRaises(OSError):
                self.build()
        self.assertFalse((self.root / 'out').exists())
        self.assertFalse((self.root / 'out.partial/summary.json').exists())
        with self.assertRaises(FileExistsError):
            self.build()

    def test_reader_rejects_tampering_and_invalid_window_indices(self):
        self.build()
        package = CorridorHistory(self.root / 'out')
        for position, corridor in ((-1, 'west'), (2, 'west'), (0, 'missing')):
            with self.assertRaises(ValueError):
                package.window(position, corridor)
        path = self.root / 'out/corridors.json'
        path.write_text('[]', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            CorridorHistory(self.root / 'out')


if __name__ == '__main__':
    unittest.main()
