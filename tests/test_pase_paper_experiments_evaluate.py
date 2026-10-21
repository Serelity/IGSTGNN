"""Server-only contract tests; no paper accuracy is inferred from toy arrays."""
import copy
import logging
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.original_baseline_gw16.evaluate import native_summary
from experiments.pase_paper_experiments import evaluate as evaluation
from experiments.pase_paper_experiments.protocol import CORE_ARMS, configuration
from experiments.position_state_gw16.geometry import RegionGeometry


class PaperEvaluationTests(unittest.TestCase):
    def formal_guard_fixture(self):
        identity = dict(engineering_only=False, config={'max_epochs':100},
                        comparison_arms=list(CORE_ARMS))
        roles = {arm:dict(completed_training_epochs=100, full_protocol_reached=True,
            selection='best', selected_on='validation_pooled_mae',
            best_includes_stage_zero=True) for arm in CORE_ARMS}
        return identity, roles

    def test_test_requires_all_nine_complete_even_when_reporting_one(self):
        identity, roles = self.formal_guard_fixture()
        evaluation.test_guard(identity, roles, ['FULL'])
        for alteration in ('missing', 'epoch', 'last', 'engineering', 'selection'):
            candidate, checkpoints = copy.deepcopy(identity), copy.deepcopy(roles)
            if alteration == 'missing':
                checkpoints.pop(CORE_ARMS[-1])
            elif alteration == 'epoch':
                checkpoints[CORE_ARMS[-1]]['completed_training_epochs'] = 99
            elif alteration == 'last':
                checkpoints['FULL']['selection'] = 'last'
            elif alteration == 'engineering':
                candidate['engineering_only'] = True
            else:
                checkpoints['FULL']['selected_on'] = 'test_mae'
            with self.subTest(alteration=alteration), self.assertRaises(ValueError):
                evaluation.test_guard(candidate, checkpoints, ['FULL'])

    def test_native_horizon_mean_rmse_is_not_pooled_rmse(self):
        y = np.full((2, 12, 3, 1), 20., dtype=np.float32)
        p = y+np.arange(1, 13, dtype=np.float32)[None, :, None, None]
        result = native_summary(p, y)
        self.assertAlmostEqual(result['horizon_macro']['mae'], 6.5)
        self.assertAlmostEqual(result['horizon_macro']['rmse'], 6.5)
        self.assertGreater(abs(result['pooled']['rmse']-6.5), .5)
        self.assertEqual([row['lead'] for row in result['per_horizon']], list(range(1, 13)))

    def test_validation_replay_rejects_metric_or_count_changes(self):
        expected = dict(mae=12., rmse=20., valid_count=17, mape_valid_count=15,
                        region_mae=None, region_valid_count=0)
        self.assertTrue(evaluation.replay_validation(expected, expected)['passed'])
        for key, value in (('mae', 12.01), ('valid_count', 18), ('region_mae', 0.)):
            altered = dict(expected, **{key:value})
            with self.subTest(key=key), self.assertRaises(ValueError):
                evaluation.replay_validation(altered, expected)

    def cache_fixture(self):
        config = configuration()
        y = np.full((2, 12, 3, 1), 20., dtype=np.float32)
        arrays = dict(prediction=y+2, target=y,
                      sample_ids=np.asarray([3909, 3910], dtype=np.int64),
                      station_ids=np.asarray([11, 12, 13], dtype=np.int64))
        role = dict(model_tensor_sha256='weights', epoch=3,
                    completed_training_epochs=100, selection='best')
        key = 's2025__FULL'
        frozen = dict(frozen_sha256='frozen', records={key:dict(role=role)})
        metadata = dict(schema=evaluation.SCHEMA, frozen_sha256='frozen', record_key=key,
            checkpoint_role=role, config=config, metric_contract=evaluation.METRIC_CONTRACT,
            native_metrics=native_summary(arrays['prediction'], arrays['target']),
            common_mask_pooled=evaluation.numpy_metrics(arrays['prediction'], arrays['target'], config),
            target_tensor_sha256=evaluation.tensor_tree_hash(arrays['target']),
            sample_ids_sha256=evaluation.tensor_tree_hash(arrays['sample_ids']),
            station_ids_sha256=evaluation.tensor_tree_hash(arrays['station_ids']),
            audit=dict(model_tensor_sha256='weights', registered_tensors_unchanged=True,
                rng_unchanged=True, raw_inputs_unchanged=True,
                future_target_passed_to_forward=False, optimizer_updates=0))
        return arrays, metadata, frozen, key

    def test_completed_cache_reuses_without_model_or_collect(self):
        arrays, metadata, frozen, key = self.cache_fixture()
        for save in (False, True):
            with self.subTest(save=save), tempfile.TemporaryDirectory() as tmp:
                destination = Path(tmp)/key
                evaluation.commit_cache(destination, arrays, metadata, save, frozen, key)
                with (patch.object(evaluation, 'build_model', side_effect=AssertionError('new model')),
                      patch.object(evaluation, 'collect', side_effect=AssertionError('new forward'))):
                    cached = evaluation.read_cache(destination, frozen, key)
                self.assertEqual(cached['native_metrics'], metadata['native_metrics'])
                self.assertEqual((destination/'predictions.npz').exists(), save)

    def test_cache_rejects_changed_selection_metadata_and_prediction_bytes(self):
        arrays, metadata, frozen, key = self.cache_fixture()
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp)/key
            evaluation.commit_cache(destination, arrays, metadata, True, frozen, key)
            changed = dict(frozen, frozen_sha256='another-best')
            with self.assertRaisesRegex(ValueError, 'another checkpoint'):
                evaluation.read_cache(destination, changed, key)
            saved = evaluation.read_json(destination/'metadata.json')
            saved['checkpoint_role']['epoch'] = 4
            evaluation.write_json(destination/'metadata.json', saved)
            with self.assertRaisesRegex(ValueError, 'metadata checksum'):
                evaluation.read_cache(destination, frozen, key)
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp)/key
            evaluation.commit_cache(destination, arrays, metadata, True, frozen, key)
            with (destination/'predictions.npz').open('ab') as stream:
                stream.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'Prediction cache checksum'):
                evaluation.read_cache(destination, frozen, key)

    def test_interrupted_write_is_not_a_completed_cache(self):
        arrays, metadata, frozen, key = self.cache_fixture()
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp)/key
            with patch.object(evaluation, 'write_json', side_effect=OSError('interrupted')):
                with self.assertRaises(OSError):
                    evaluation.commit_cache(destination, arrays, metadata, False, frozen, key)
            self.assertFalse(destination.exists())
            evaluation.commit_cache(destination, arrays, metadata, False, frozen, key)
            self.assertEqual(evaluation.read_cache(destination, frozen, key)['record_key'], key)

    def test_outputs_do_not_intersect_raw_data_or_seed_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, data = Path(tmp)/'matrix', Path(tmp)/'raw'
            root.mkdir()
            data.mkdir()
            for output in (root, Path(tmp), data, data/'evaluation'):
                with self.subTest(output=output), self.assertRaises(ValueError):
                    evaluation.safe_output(output, root, data)
            seed = root/'s2025'
            seed.mkdir()
            (seed/'identity.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'seed training run'):
                evaluation.safe_output(seed/'evaluation', root, data)
            self.assertEqual(evaluation.safe_output(root/'evaluation_val', root, data),
                             (root/'evaluation_val').resolve())

    def test_cli_defaults_to_validation_and_rejects_engineering_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            options = ['--run-root', str(Path(tmp)/'matrix'), '--data-dir', str(Path(tmp)/'raw')]
            args = evaluation.parse_args(options)
            self.assertEqual(args.split, 'val')
            self.assertEqual(args.arms, list(CORE_ARMS))
            with self.assertRaises(ValueError):
                evaluation.parse_args(options+['--engineering', '--split', 'test'])

    def test_seed_aggregation_uses_all_registered_values_and_sample_std(self):
        records = [dict(arm='FULL', seed=seed, parameters={'registered':73835},
            native_metrics={'horizon_macro':dict(mae=value, rmse=2*value, mape_percent=3*value)})
            for seed, value in ((2025, 10.), (2026, 12.), (2027, 14.))]
        result = evaluation.aggregate(records, ['FULL'])['FULL']['metrics']['mae']
        self.assertEqual(result['mean'], 12.)
        self.assertEqual(result['sample_std'], 2.)
        self.assertEqual(result['seed_values'], {'2025':10., '2026':12., '2027':14.})

    def test_collection_only_passes_history_and_event_and_respects_engineering_prefix(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.paper_arm = self.arm = 'FULL'
                self.geometry = RegionGeometry([0, 0, 1])
                self.calls = 0

            def forward(self, history, event):
                self.calls += 1
                prediction = history[..., :1]
                return dict(prediction=prediction, reference=prediction)

        class Loader:
            size, num_batch = 4, 2

            def get_iterator(self):
                for _ in range(2):
                    yield dict(x_data=np.full((2, 12, 3, 3), 20., dtype=np.float32),
                               y_data=np.full((2, 12, 3, 1), 21., dtype=np.float32), event_data={})

        data = SimpleNamespace(station_ids=[11, 12, 13], mean=0., std=1.)
        config = dict(configuration(), nodes=3)
        model = Model()
        with patch.object(evaluation, 'validation_loader', return_value=(Loader(), 3909)):
            arrays, metrics, audit = evaluation.collect(model, data, config, 'val', 'cpu',
                logging.getLogger('toy'), max_batches=1)
        self.assertEqual(model.calls, 1)
        self.assertEqual(arrays['sample_ids'].tolist(), [3909, 3910])
        self.assertEqual(metrics['mae'], 1.)
        self.assertEqual(audit['forward_batches'], 1)
        self.assertFalse(audit['future_target_passed_to_forward'])
        with patch.object(evaluation, 'validation_loader', return_value=(Loader(), 3909)):
            arrays, _, audit = evaluation.collect(model, data, config, 'test', 'cpu',
                logging.getLogger('toy'))
        self.assertEqual(len(arrays['target']), 4)
        class ShortLoader(Loader):
            def get_iterator(self):
                yield next(super().get_iterator())
        with (patch.object(evaluation, 'validation_loader', return_value=(ShortLoader(), 3909)),
              self.assertRaisesRegex(ValueError, 'declared number of batches')):
            evaluation.collect(model, data, config, 'test', 'cpu', logging.getLogger('toy'))


if __name__ == '__main__':
    unittest.main()
