"""Server-only final-test adapter boundary checks; never executed locally."""
from contextlib import ExitStack
import copy
import csv
import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from experiments.final_test_gw16 import data as entry
from experiments.chronological.original_igstgnn_source import get_original_modules
from src.utils.dataloader import IncidentDataLoader, StandardScaler
from src.utils.igstgnn_published_events import EventPublishedLoader


class Stats(dict):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def fixture(directory):
    stations = list(range(1000, 1496))
    with (directory/'sensors.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['station_id'])
        writer.writeheader()
        writer.writerows([dict(station_id=value) for value in stations])
    x = np.zeros((12, 496, 3), dtype=np.float32)
    y = np.ones_like(x)
    sample = dict(x_data=x, y_data=y, incident_features=dict(Type=0, Description=0),
                  incident_position=0, incident_distances=np.ones((496, 3), dtype=np.float32))
    samples = [dict(sample) for _ in range(839)]
    audit = dict(counts=entry.COUNTS.copy(), split_policy='author_order_70_15_15',
        scaler_refitted=False, category_counts=dict(type=6, description=29),
        unknown_tokens=dict(type=6, description=29),
        event_input_schema='published_type_description_relation_v1', mean=100., std=20.)
    logger = logging.getLogger(__name__)
    published = SimpleNamespace(_test_samples=samples, audit=audit, station_ids=stations)
    published.test_loader = lambda batch, log: EventPublishedLoader(published._test_samples, batch, log)
    gw16 = SimpleNamespace(directory=directory, identity=dict(hashes=entry.RELEASE.copy(),
        engineering_only=False, saved_training_token='unchanged'),
        published=published, station_ids=stations, mean=100., std=20.,
        loaders=dict(train_loader=object(), val_loader=object()),
        position_features=np.zeros((496, 6), dtype=np.float32), assignment=np.zeros(496, dtype=np.int64))
    config = dict(dataset='Contra_Costa', seq_len=12, horizon=12, input_dim=3,
        output_dim=1, incident=True, use_sensor_info=True, bs=48)
    def sensors(path, nodes, args, log):
        args.sensor_type_size = 1
        args.surface_size = 1
        args.roadway_use_size = 1
        return dict(sensor_type=np.zeros(nodes, dtype=np.int64), surface=np.zeros(nodes, dtype=np.int64),
                    roadway_use=np.zeros(nodes, dtype=np.int64), road_width=np.ones(nodes), speed_limit=np.ones(nodes))
    modules = SimpleNamespace(loader=SimpleNamespace(_load_sensor_info=sensors,
        StandardScaler=StandardScaler, IncidentDataLoader=IncidentDataLoader), source_fingerprint='frozen_source')
    return config, gw16, modules, logger


class FinalTestGW16DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.config, self.gw16, self.modules, self.logger = fixture(self.directory)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.hash_mock = self.stack.enter_context(patch.object(entry, 'sha256',
            side_effect=lambda path: entry.RELEASE[Path(path).name]))
        self.stats_mock = self.stack.enter_context(patch.object(entry.np, 'load',
            return_value=Stats(mean=np.asarray(100.), std=np.asarray(20.), normalized=np.asarray(True))))

    def load(self):
        return entry.load_test_data(self.directory, self.config, self.gw16, self.modules, self.logger)

    def test_exact_test_axes_same_raw_order_and_no_reloading_all_samples(self):
        result = self.load()
        np.testing.assert_array_equal(result.sample_ids, np.arange(4746, 5585, dtype=np.int64))
        self.assertEqual(result.sample_ids.dtype, np.int64)
        self.assertEqual(result.station_ids.shape, (496,))
        self.assertEqual(result.identity['batch_partition'], [48]*17+[23])
        self.assertEqual(result.identity['bounds_half_open'], [4746, 5585])
        self.assertEqual(result.native_loader.size, 839)
        self.assertEqual(result.candidate_loader.size, 839)
        for native, candidate, raw in zip(result.native_loader.samples,
                result.candidate_loader.original_samples, self.gw16.published._test_samples):
            self.assertIs(native, raw)
            self.assertIs(candidate, raw)
        self.assertEqual(self.stats_mock.call_count, 1)
        self.assertEqual(Path(self.stats_mock.call_args.args[0]).name, 'incident_stats.npz')
        self.assertEqual(self.stats_mock.call_args.kwargs, dict(allow_pickle=False))
        self.assertEqual(self.hash_mock.call_count, 6)

    def test_frozen_native_parser_and_scaler_preserve_config_and_training_identity(self):
        original_config = copy.deepcopy(self.config)
        original_identity = copy.deepcopy(self.gw16.identity)
        old_train, old_val = self.gw16.loaders.values()
        result = self.load()
        self.assertEqual(self.config, original_config)
        self.assertEqual(self.gw16.identity, original_identity)
        self.assertIs(self.gw16.loaders['train_loader'], old_train)
        self.assertIs(self.gw16.loaders['val_loader'], old_val)
        self.assertEqual(result.native_data.config['sensor_type_size'], 1)
        self.assertEqual(result.native_data.config['node_num'], 496)
        self.assertEqual(float(result.native_data.scaler.mean), 100.)
        self.assertEqual(float(result.native_data.scaler.std), 20.)
        self.assertFalse(result.identity['scaler_refitted'])
        self.assertFalse(result.identity['test_evaluated'])
        self.assertFalse(result.identity['absolute_calendar_order_independently_verified'])

    def test_both_parsers_produce_identical_history_and_target_arrays(self):
        result = self.load()
        native = next(result.native_loader.get_iterator())
        candidate = next(result.candidate_loader.get_iterator())
        np.testing.assert_array_equal(native['x_data'], candidate['x_data'])
        np.testing.assert_array_equal(native['y_data'], candidate['y_data'])
        self.assertEqual(native['x_data'].shape, (48, 12, 496, 3))
        self.assertEqual(native['y_data'].shape, (48, 12, 496, 1))

    def test_hash_checked_frozen_original_modules_accept_build_materials(self):
        self.modules = get_original_modules()
        result = self.load()
        self.assertIsInstance(result.native_loader, self.modules.loader.IncidentDataLoader)
        self.assertIsInstance(result.native_data.scaler, self.modules.loader.StandardScaler)
        self.assertEqual(result.native_data.config['sensor_type_size'], 1)
        native = next(result.native_loader.get_iterator())
        candidate = next(result.candidate_loader.get_iterator())
        np.testing.assert_array_equal(native['x_data'], candidate['x_data'])
        np.testing.assert_array_equal(native['y_data'], candidate['y_data'])
        self.assertEqual(result.identity['native_source_fingerprint'], self.modules.source_fingerprint)

    def test_modified_release_hash_rejected_before_test_validation(self):
        self.hash_mock.side_effect = lambda path: 'changed'
        with self.assertRaisesRegex(ValueError, 'fingerprint mismatch'):
            self.load()

    def test_engineering_or_foreign_gw16_identity_rejected(self):
        self.gw16.identity['engineering_only'] = True
        with self.assertRaisesRegex(ValueError, 'Engineering'):
            self.load()
        self.gw16.identity['engineering_only'] = False
        self.gw16.identity['hashes']['sensors.csv'] = 'other'
        with self.assertRaisesRegex(ValueError, 'six-file'):
            self.load()

    def test_wrong_sample_split_count_rejected(self):
        self.gw16.published._test_samples.pop()
        with self.assertRaisesRegex(ValueError, '839'):
            self.load()

    def test_wrong_audited_counts_rejected(self):
        self.gw16.published.audit['counts']['test_samples'] = 837
        with self.assertRaisesRegex(ValueError, 'split counts'):
            self.load()

    def test_changed_station_axis_rejected(self):
        self.gw16.station_ids = list(reversed(self.gw16.station_ids))
        with self.assertRaisesRegex(ValueError, 'station axis'):
            self.load()

    def test_changed_scaler_or_event_vocabulary_rejected(self):
        self.gw16.std = 21.
        with self.assertRaisesRegex(ValueError, 'scaler differs'):
            self.load()
        self.gw16.std = 20.
        self.gw16.published.audit['category_counts']['type'] = 7
        with self.assertRaisesRegex(ValueError, 'event schema'):
            self.load()

    def test_every_test_window_is_validated_not_only_first_record(self):
        self.gw16.published._test_samples[-1]['y_data'] = np.ones((12, 495, 3), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, 'test y_data: 5584'):
            self.load()

    def test_clock_mismatch_and_missing_native_payload_rejected(self):
        clocks = self.gw16.published._test_samples[-1]['x_data'].copy()
        clocks[0, 1, 1] = .5
        self.gw16.published._test_samples[-1]['x_data'] = clocks
        with self.assertRaisesRegex(ValueError, 'clocks disagree'):
            self.load()
        self.gw16.published._test_samples[-1]['x_data'] = np.zeros((12, 496, 3), dtype=np.float32)
        del self.gw16.published._test_samples[-1]['incident_position']
        with self.assertRaisesRegex(ValueError, 'incident payload'):
            self.load()

    def test_original_noncanonical_shape_or_batch_rejected(self):
        self.config['bs'] = 32
        with self.assertRaisesRegex(ValueError, 'configuration'):
            self.load()

    def test_candidate_cannot_reorder_test_slice(self):
        self.gw16.published.test_loader = lambda batch, log: EventPublishedLoader(
            list(reversed(self.gw16.published._test_samples)), batch, log)
        with self.assertRaisesRegex(ValueError, 'sample order'):
            self.load()


if __name__ == '__main__':
    unittest.main()
