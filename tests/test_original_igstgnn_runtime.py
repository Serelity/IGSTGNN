"""Original train/validation boundaries on small, authenticated test fixtures."""
import json
import logging
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.chronological import original_igstgnn_runtime as runtime
from experiments.chronological.original_igstgnn_source import get_original_modules


class OriginalRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.modules = get_original_modules()
        cls.logger = logging.getLogger('original_runtime_tests')
        cls.logger.addHandler(logging.NullHandler())

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        fields = 'station_id,Sensor Type,Type,Surface,Roadway Use,Road Width,Lane Width,Design Speed Limit,Fwy,Direction\n'
        rows = [f'{10000+i},sensor{i%3},Mainline,surface{i%4},use{i%7},12 ft,3.6 m,65 mph,I680,S' for i in range(496)]
        (self.directory/'sensors.csv').write_text(fields+'\n'.join(rows)+'\n', encoding='utf-8')
        (self.directory/'desc_mapping.json').write_text(json.dumps({str(i): i for i in range(29)}))
        (self.directory/'type_mapping.json').write_text(json.dumps({str(i): i for i in range(6)}))
        np.save(self.directory/'adj_matrix.npy', np.eye(496, dtype=np.float32))
        np.savez(self.directory/'incident_stats.npz', mean=np.float32(10), std=np.float32(2), normalized=True)
        x = np.zeros((12, 496, 3), dtype=np.float32)
        x[..., 0] = 1
        y = x.copy()
        self.sample = dict(x_data=x, y_data=y, incident_features={'Incident Time': 3,
            'Description': 1, 'Type': 1, 'Holiday': 0}, incident_position=0,
            incident_distances=np.zeros((496, 3), dtype=np.float32))
        # Many references to one sample keep the fixture small. Test entries
        # lack target fields, proving load_data does not examine test targets.
        self.samples = [self.sample]*4746 + [{'never_evaluate_test': True}]*839
        np.save(self.directory/'incident_all.npy', np.asarray(self.samples, dtype=object))
        self.release = {name: runtime.sha256(self.directory/name) for name in runtime.RELEASE}
        self.identity_patch = patch.object(runtime, 'RELEASE', self.release)
        self.identity_patch.start()
        self.addCleanup(self.identity_patch.stop)

    def load(self, *, check=True, training=False):
        return runtime.load_data(self.directory, runtime.CONFIG, self.modules, self.logger,
                                 check=check, training=training)

    def test_fixed_release_hash_is_required_even_in_check(self):
        # The fixture has its own declared hashes; the production constants are
        # never bypassed in runtime.load_data.
        self.assertEqual(self.load().identity['hashes'], self.release)
        with (self.directory/'sensors.csv').open('a', encoding='utf-8') as stream:
            stream.write('\n')
        for check in (True, False):
            with self.assertRaisesRegex(ValueError, 'fingerprint mismatch: sensors.csv'):
                self.load(check=check)

    def test_ignores_untrusted_split_files_and_never_builds_test(self):
        for name in ('incident_train.npy', 'incident_val.npy', 'incident_test.npy'):
            (self.directory/name).write_bytes(b'not an authenticated release')
        data = self.load(training=True)
        self.assertEqual(set(data.loaders), {'train_loader', 'val_loader'})
        self.assertEqual([data.loaders[k].size for k in data.loaders], [4, 4])
        self.assertFalse(hasattr(data, 'test_loader'))
        self.assertTrue(data.identity['test_payload_deserialized'])
        self.assertFalse(data.identity['test_file_opened'])
        self.assertFalse(data.identity['test_evaluated'])
        self.assertEqual(data.identity['splits']['val']['selected_indices'], [3909, 3910, 3911, 3912])
        self.assertEqual(data.identity['splits']['train']['partition'], [2, 2])
        self.assertNotIn('incident_test.npy', data.identity['hashes'])

    def test_full_validation_preserves_original_batch48_tail(self):
        data = self.load(check=False)
        self.assertEqual(set(data.loaders), {'val_loader'})
        self.assertEqual(data.loaders['val_loader'].size, 837)
        self.assertEqual(data.loaders['val_loader'].bs, 48)
        self.assertEqual(data.identity['splits']['val']['partition'], [48]*17+[21])
        self.assertEqual(data.identity['splits']['val']['selected_indices'], list(range(3909, 4746)))
        self.assertEqual(data.roads, {'I680-S': list(range(496))})

    def test_original_sensor_parser_device_batch_and_count455220(self):
        data = self.load()
        self.assertEqual((data.config['sensor_type_size'], data.config['surface_size'],
                          data.config['roadway_use_size']), (3, 4, 7))
        self.assertTrue(np.array_equal(data.sensors['road_width'], np.zeros(496)))
        self.assertTrue(np.array_equal(data.sensors['speed_limit'], np.full(496, 50)))
        model = runtime.build_model(data, self.modules, torch.device('cpu'))
        self.assertEqual(model.param_num(), 455220)
        self.assertIs(type(model), self.modules.model.IGSTGNN)
        self.assertEqual(model.node_num, 496)
        self.assertTrue(all(p.dtype == torch.float32 for p in model.parameters()))
        self.assertTrue(all(a.dtype == torch.float32 for a in model._model_args['adjs']))
        batch = next(data.loaders['val_loader'].get_iterator())
        native = object.__new__(self.modules.engine.IGSTGNN_Engine)
        native._device, native._incident = torch.device('cpu'), True
        actual, expected = runtime.device_batch(batch, 'cpu'), native._prepare_batch(batch)
        self.assertEqual(runtime.tensor_tree_hash(actual), runtime.tensor_tree_hash(expected))
        self.assertEqual(actual[0].shape, (2, 12, 496, 3))
        self.assertEqual(actual[1].shape, (2, 12, 496, 1))
        # Model construction only: no real/synthetic forward or optimizer.

    def test_native_mask_matches_original_engine_rule(self):
        native = object.__new__(self.modules.engine.IGSTGNN_Engine)
        for target in (torch.tensor([0., 2., 4.]), torch.tensor([2., 3., 4.]),
                       torch.tensor([.25, .5, 1.])):
            prediction = target + torch.tensor([3., 1., 2.])
            actual = runtime.native_metrics(prediction, target, self.modules)
            null = native._mask_value(target)
            self.assertEqual(actual['mask_value'], float(null))
            for key, function in (('mae', 'masked_mae'), ('rmse', 'masked_rmse'), ('mape_fraction', 'masked_mape')):
                self.assertEqual(actual[key], float(getattr(self.modules.metrics, function)(prediction, target, null)))

    def test_bad_count_and_axes_rejected_after_fingerprint_validation(self):
        cases = [self.samples[:-1],
                 [dict(self.sample, x_data=np.zeros((12, 495, 3), dtype=np.float32))]*5585,
                 [dict(self.sample, y_data=np.zeros((12, 496, 3), dtype=np.float64))]*5585]
        for values in cases:
            np.save(self.directory/'incident_all.npy', np.asarray(values, dtype=object))
            self.release['incident_all.npy'] = runtime.sha256(self.directory/'incident_all.npy')
            with self.assertRaises(ValueError):
                self.load()

    def test_default_data_dir_repository_and_adjacent_layouts(self):
        outer = self.directory/'project'
        repo = outer/'code'
        repo.mkdir(parents=True)
        adjacent = outer/'data/xtraffic/Contra_Costa'
        adjacent.mkdir(parents=True)
        with patch.object(runtime, 'REPO', repo):
            self.assertEqual(runtime.default_data_dir(), adjacent)
            internal = repo/'data/xtraffic/Contra_Costa'
            internal.mkdir(parents=True)
            self.assertEqual(runtime.default_data_dir(), internal)


if __name__ == '__main__':
    unittest.main()
