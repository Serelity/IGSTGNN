"""Frozen original identity, literal-only archive metadata and train/val inputs."""
import io
import json
import logging
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import torch

from experiments.chronological import original_igstgnn_source as source


def original_config():
    return dict(device='cpu', dataset='Contra_Costa', model_name='igstgnn', seed=2025,
                bs=2, seq_len=12, horizon=12, input_dim=3, output_dim=1, mode='train',
                max_epochs=100, patience=20, incident=True, use_sensor_info=True,
                num_feat=1, num_hidden=32, node_hidden=12, time_emb_dim=12,
                layer=5, k_t=3, k_s=2, gap=3, cl_epoch=3, warm_epoch=30,
                tpd=288, lrate=.002, wdecay=1e-5, dropout=.1, clip_grad_value=5,
                icsf_dim=64, module_name='igstgnn', lambda_incident=1., sigma_t=1.,
                run_time='20261015_120000')


def create_data(directory):
    directory.mkdir()
    directory.joinpath('sensors.csv').write_text(
        'station_id,Sensor Type,Type,Surface,Roadway Use,Road Width,Lane Width,Design Speed Limit\n'
        '101,Loop,Mainline,A,X,12 ft,3.6 m,65 mph\n'
        '102,Loop,Mainline,A,X,12 ft,3.6 m,65 mph\n'
        '103,Radar,Mainline,B,Y,12 ft,3.6 m,65 mph\n'
        '104,Radar,Mainline,B,Y,12 ft,3.6 m,65 mph\n', encoding='utf-8')
    directory.joinpath('desc_mapping.json').write_text(json.dumps({'UNK': 0, 'ACC': 1}))
    directory.joinpath('type_mapping.json').write_text(json.dumps({'UNK': 0, 'ACC': 1}))
    np.save(directory / 'adj_matrix.npy', np.eye(4, dtype=np.float32))
    np.savez(directory / 'incident_stats.npz', mean=np.float32(10), std=np.float32(2), normalized=True)
    samples = []
    for index in range(17):
        x = np.zeros((12, 4, 3), dtype=np.float32)
        x[..., 0] = index
        samples.append(dict(x_data=x, y_data=x.copy(), incident_features={'Incident Time': 3,
                            'Description': 1, 'Type': 1, 'Holiday': 0},
                            incident_position=0, incident_distances=np.zeros((4, 3), dtype=np.float32)))
    # If test targets are inspected this fails: the test entries deliberately
    # do not implement the model-ready sample contract.
    samples.extend([{'test_payload_not_to_be_evaluated': True}] * 3)
    np.save(directory / 'incident_all.npy', np.asarray(samples, dtype=object))
    return samples


def create_log(config, parameter_count):
    namespace = 'Namespace(' + ', '.join(f'{key}={value!r}' for key, value in config.items()) + ')'
    return '\n'.join([
        '2026-10-15 - ' + namespace,
        'train: samples=14, x=(12, 4, 3), y=(12, 4, 3)',
        'val: samples=3, x=(12, 4, 3), y=(12, 4, 3)',
        f'The number of parameters: {parameter_count}',
        'Epoch: 001, Train Loss: 20.0, Valid Loss: 15.0000, Valid RMSE: 25.0000, Valid MAPE: 0.2000',
        'Val loss decrease from inf to 15.0000',
        'Epoch: 002, Train Loss: 16.0, Valid Loss: 12.1235, Valid RMSE: 22.0000, Valid MAPE: 0.1800',
        'Val loss decrease from 15.0000 to 12.1235',
        'Epoch: 003, Train Loss: 14.0, Valid Loss: 13.0000, Valid RMSE: 23.0000, Valid MAPE: 0.1900',
        'Early stop at epoch 3, loss = 12.123456',
    ])


class OriginalSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.modules = source.get_original_modules()
        cls.logger = logging.getLogger('original_source_test')
        cls.logger.addHandler(logging.NullHandler())

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.directory = self.root / 'data'
        self.samples = create_data(self.directory)
        self.config = original_config()
        self.args = SimpleNamespace(**self.config, node_num=4)
        self.sensor_info = self.modules.loader._load_sensor_info(str(self.directory), 4, self.args, self.logger)
        args = vars(self.args).copy()
        args['adjs'] = [torch.eye(4), torch.eye(4)]
        self.model = self.modules.model.IGSTGNN(model_args=args, node_num=4, input_dim=3,
                     output_dim=1, seq_len=12, horizon=12, data_path=str(self.directory),
                     dataset='Contra_Costa', use_sensor_info=True)
        self.state = self.model.state_dict()
        self.log = create_log(self.config, self.model.param_num())
        weight = io.BytesIO()
        torch.save(self.state, weight)
        self.weight_bytes = weight.getvalue()
        self.zip_path = self.root / 'original.zip'
        with zipfile.ZipFile(self.zip_path, 'w') as zipped:
            # Historical ZIP members have a leading slash. Reading is safe:
            # the loader never extracts archive members to the filesystem.
            zipped.writestr('/run/final_model_s2025.pt', self.weight_bytes)
            zipped.writestr('/run/record_igstgnn_s2025_20261015.log', self.log)

    def test_frozen_imports_do_not_rebind_current_src(self):
        current = {key: value for key, value in sys.modules.items() if key.startswith('src.')}
        modules = source.get_original_modules()
        self.assertIs(modules.model.IGSTGNN.__bases__[0], modules.base_model.BaseModel)
        self.assertIs(modules.engine.IGSTGNN_Engine.__bases__[0], modules.base_engine.BaseEngine)
        self.assertEqual(current, {key: value for key, value in sys.modules.items() if key.startswith('src.')})
        self.assertNotIn('implementation', vars(modules.model.IGSTGNN))
        self.assertEqual(modules.source_fingerprint, source.MANIFEST_SHA256)

    def test_source_tampering_rejected_after_cached_import(self):
        other = self.root / 'vendor'
        shutil.copytree(source.VENDOR, other)
        with patch.object(source, 'VENDOR', other):
            self.assertIs(source.get_original_modules(), self.modules)
            with (other / 'model.py').open('ab') as stream:
                stream.write(b'\n# changed\n')
            with self.assertRaisesRegex(ValueError, 'model.py'):
                source.get_original_modules()

    def test_namespace_is_literal_only_and_unique(self):
        self.assertEqual(source.parse_config(self.log), self.config)
        for text in (self.log + '\nNamespace(x=1)',
                     self.log.replace("device='cpu'", "device=__import__('os').getcwd()"),
                     self.log.replace("device='cpu'", "device='cpu', device='cpu'"),
                     self.log.replace("device='cpu'", "device='cpu', **{}"),
                     self.log.replace("device='cpu'", "device=[]"),
                     self.log.replace("device='cpu'", "implementation='paper_aligned_v1', device='cpu'")):
            with self.subTest(text=text[:100]), self.assertRaises(ValueError):
                source.parse_config(text)

    def test_best_save_epoch_and_literal_precision(self):
        parsed = source.parse_training_log(self.log)
        self.assertEqual(parsed['best_epoch'], 2)
        self.assertEqual(parsed['best_validation_mae_literal'], '12.123456')
        self.assertEqual(parsed['best_validation_mae_precision_digits'], 6)
        self.assertEqual(parsed['best_epoch_metrics']['validation_mae_literal'], '12.1235')
        self.assertEqual(parsed['counts'], {'train': 14, 'val': 3, 'node_num': 4})
        for bad in (self.log.replace('loss = 12.123456', 'loss = 10.0'),
                    self.log.replace('Early stop at epoch 3', 'Early stop at epoch 2'),
                    self.log.replace('Epoch: 002', 'Epoch: 001'),
                    self.log.replace('Val loss decrease from 15.0000 to 12.1235', 'not a save')):
            with self.assertRaises(ValueError):
                source.parse_training_log(bad)

    def test_archive_reads_without_extraction_and_strict_load(self):
        before = sorted(str(path) for path in self.root.rglob('*'))
        loaded = source.load_archive(self.zip_path, model=self.model)
        self.assertEqual(before, sorted(str(path) for path in self.root.rglob('*')))
        self.assertEqual(loaded.best_epoch, 2)
        self.assertTrue(loaded.provenance['weights_only'])
        self.assertTrue(loaded.provenance['full_strict_model_load_checked'])
        self.assertFalse(loaded.provenance['zip_extracted'])
        self.assertTrue(all(torch.equal(value, loaded.state_dict[key]) for key, value in self.state.items()))
        checkpoint, log = self.root/'final_model.pt', self.root/'train.log'
        checkpoint.write_bytes(self.weight_bytes)
        log.write_text(self.log, encoding='utf-8')
        pair = source.load_archive(checkpoint=checkpoint, train_log=log)
        self.assertEqual(loaded.provenance['checkpoint_sha256'], pair.provenance['checkpoint_sha256'])
        with self.assertRaises(ValueError):
            source.load_archive(self.zip_path, checkpoint=checkpoint, train_log=log)

    def test_ambiguous_zip_and_modified_checkpoint_rejected(self):
        with zipfile.ZipFile(self.zip_path, 'a') as zipped:
            zipped.writestr('/other/final_model_s2026.pt', self.weight_bytes)
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            source.load_archive(self.zip_path)
        checkpoint, log = self.root/'bad.pt', self.root/'train.log'
        log.write_text(self.log, encoding='utf-8')
        for state in ({'state_dict': self.state},
                      dict(self.state, **{'tiid_module.context_mlp.0.weight': torch.zeros(1)}),
                      {k: v for k, v in self.state.items() if k != 'icsf_module.distance_encoder.0.weight'},
                      dict(self.state, **{'node_emb_u': torch.zeros(5, 12)}),
                      dict(self.state, **{'embedding.weight': torch.full((32, 1), float('nan'))})):
            torch.save(state, checkpoint)
            with self.assertRaises(ValueError):
                source.load_archive(checkpoint=checkpoint, train_log=log)
        state = {k: v for k, v in self.state.items() if k != 'out_fc_1.bias'}
        torch.save(state, checkpoint)
        with self.assertRaises(RuntimeError):
            source.load_archive(checkpoint=checkpoint, train_log=log, model=self.model)

    def test_archive_seed_mismatch_rejected(self):
        other = self.root/'other.zip'
        with zipfile.ZipFile(other, 'w') as zipped:
            zipped.writestr('/run/final_model_s2026.pt', self.weight_bytes)
            zipped.writestr('/run/record_igstgnn_s2025_20261015.log', self.log)
        with self.assertRaisesRegex(ValueError, 'member seed'):
            source.load_archive(other)

    def test_original_sensor_parser_and_batch_semantics(self):
        self.assertEqual(self.args.sensor_type_size, 2)
        self.assertTrue(np.array_equal(self.sensor_info['road_width'], np.zeros(4)))
        self.assertTrue(np.array_equal(self.sensor_info['speed_limit'], np.full(4, 50)))
        loader = self.modules.loader.IncidentDataLoader(self.samples[14:17], True, 2,
                    self.logger, input_dim=3, sensor_info=self.sensor_info)
        batches = list(loader.get_iterator())
        self.assertEqual([len(batch['x_data']) for batch in batches], [2, 1])
        self.assertEqual(batches[0]['incident_features'].tolist(), [[3., 1., 1., 0.]] * 2)
        engine = object.__new__(self.modules.engine.IGSTGNN_Engine)
        engine._device, engine._incident = torch.device('cpu'), True
        X, Y, incident, sensors = engine._prepare_batch(batches[0])
        self.assertEqual(X.shape, (2, 12, 4, 3))
        self.assertEqual(Y.shape, (2, 12, 4, 1))
        self.assertEqual(incident['position'].dtype, torch.int64)
        self.assertEqual(sensors['road_width'].shape, (2, 4))


if __name__ == '__main__':
    unittest.main()
