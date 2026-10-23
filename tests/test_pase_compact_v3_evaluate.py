"""Server/remote CI contracts for compact PASE evaluation; toy arrays are not results."""
import copy
from contextlib import redirect_stderr
import io
import logging
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.original_baseline_gw16.evaluate import native_summary
from experiments.pase_compact_v3 import evaluate as evaluation
from experiments.pase_compact_v3.protocol import CORE_ARMS
from experiments.position_state_gw16.geometry import RegionGeometry


def toy_config():
    return dict(batch_size=2, horizon=12, nodes=3, target_minimum=.001,
                mape_minimum=1., minimum_region_coverage=.8, region_loss_weight=.2)


class CompactEvaluationTests(unittest.TestCase):
    def formal_guard_fixture(self, arms=('FULL',)):
        identity = dict(engineering_only=False, config={'max_epochs':100},
                        comparison_arms=list(CORE_ARMS))
        roles = {arm:dict(completed_training_epochs=100, full_protocol_reached=True,
            selection='best', selected_on='validation_pooled_mae',
            best_includes_stage_zero=True) for arm in arms}
        return identity, roles

    def test_completed_full_can_evaluate_test_without_optional_ablations(self):
        identity, roles = self.formal_guard_fixture()
        evaluation.test_guard(identity, roles, ['FULL'])
        self.assertEqual(set(roles), {'FULL'})
        self.assertGreater(len(identity['comparison_arms']), len(roles))

    def test_every_requested_arm_must_have_a_complete_validation_best(self):
        identity, roles = self.formal_guard_fixture(('FULL', 'UP_ONLY'))
        evaluation.test_guard(identity, roles, ['FULL', 'UP_ONLY'])
        for alteration in ('missing', 'epoch', 'protocol', 'last', 'selection', 'stage_zero'):
            candidate = copy.deepcopy(roles)
            if alteration == 'missing':
                candidate.pop('UP_ONLY')
            elif alteration == 'epoch':
                candidate['UP_ONLY']['completed_training_epochs'] = 99
            elif alteration == 'protocol':
                candidate['UP_ONLY']['full_protocol_reached'] = False
            elif alteration == 'last':
                candidate['UP_ONLY']['selection'] = 'last'
            elif alteration == 'selection':
                candidate['UP_ONLY']['selected_on'] = 'test_mae'
            else:
                candidate['UP_ONLY']['best_includes_stage_zero'] = False
            with self.subTest(alteration=alteration), self.assertRaises(ValueError):
                evaluation.test_guard(identity, candidate, ['FULL', 'UP_ONLY'])

    def test_engineering_and_changed_budget_cannot_evaluate_test(self):
        identity, roles = self.formal_guard_fixture()
        for change in ('engineering_flag', 'engineering_identity', 'budget', 'arm_registry'):
            candidate = copy.deepcopy(identity)
            if change == 'engineering_identity':
                candidate['engineering_only'] = True
            elif change == 'budget':
                candidate['config']['max_epochs'] = 101
            elif change == 'arm_registry':
                candidate['comparison_arms'] = ['FULL']
            with self.subTest(change=change), self.assertRaises(ValueError):
                evaluation.test_guard(candidate, roles, ['FULL'], change == 'engineering_flag')

    def test_native_horizon_mean_rmse_is_not_pooled_rmse(self):
        target = np.full((2, 12, 3, 1), 20., dtype=np.float32)
        prediction = target+np.arange(1, 13, dtype=np.float32)[None, :, None, None]
        observed = native_summary(prediction, target)
        self.assertAlmostEqual(observed['horizon_macro']['mae'], 6.5)
        self.assertAlmostEqual(observed['horizon_macro']['rmse'], 6.5)
        self.assertGreater(abs(observed['pooled']['rmse']-6.5), .5)
        self.assertEqual([row['lead'] for row in observed['per_horizon']], list(range(1, 13)))

    def test_cli_defaults_to_full_single_2025_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            options = ['--run-root', str(Path(tmp)/'matrix'), '--data-dir', str(Path(tmp)/'raw')]
            args = evaluation.parse_args(options)
            self.assertEqual(args.split, 'val')
            self.assertEqual(args.seeds, [2025])
            self.assertEqual(args.arms, ['FULL'])
            with self.assertRaises(ValueError):
                evaluation.parse_args(options+['--engineering', '--split', 'test'])
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                evaluation.parse_args(options+['--seeds', '2026'])
            with self.assertRaises(ValueError):
                evaluation.parse_args(options+['--arms', 'FULL', 'FULL'])
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                evaluation.parse_args(options+['--arms', 'NO_POSITION_INPUT'])
            self.assertEqual(tuple(CORE_ARMS), ('FULL', 'UP_ONLY', 'DOWN_ONLY'))
            imported = evaluation.parse_args(options+['--conversion-evaluation', str(Path(tmp)/'replay')])
            self.assertEqual(imported.conversion_evaluation, Path(tmp)/'replay')
            for scope in (['--split', 'test'], ['--engineering'], ['--arms', 'UP_ONLY']):
                with self.subTest(scope=scope), self.assertRaises(ValueError):
                    evaluation.parse_args(options+['--conversion-evaluation', str(Path(tmp)/'replay')]+scope)

    def test_promoted_full_uses_authenticated_selection_metrics_not_native_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)/'promoted'
            identity = dict(dataset='Contra_Costa', external_full_reference=dict(
                directory=str(directory), metadata_sha256='metadata-bits', model_sha256='model-bits'))
            weights = {'layer.weight':torch.ones(2, 3)}
            role = dict(selection='best', epoch=99, completed_training_epochs=100,
                selected_on='validation_pooled_mae', best_includes_stage_zero=True)
            selected = {'mae':12.47443, 'rmse':21.6}
            metadata = dict(selection_metrics=selected, metadata_sha256='metadata-bits',
                            native_metrics={'horizon_macro':{'mae':12.474597}})
            checksum = lambda path: 'metadata-bits' if Path(path).name == 'metadata.json' else 'model-bits'
            with (patch.object(evaluation, 'sha256', side_effect=checksum),
                  patch.object(evaluation, 'load_promoted', return_value=(weights, role, metadata)) as load,
                  patch.object(evaluation, 'load_state', side_effect=AssertionError('training LAST read'))):
                observed, returned_role, metrics = evaluation.selected_best(Path(tmp)/'run', identity, 'FULL')
            self.assertIs(observed, weights)
            self.assertEqual(metrics, selected)
            self.assertTrue(returned_role['full_protocol_reached'])
            self.assertEqual(returned_role['epoch'], 99)
            load.assert_called_once_with(directory, expected_dataset='Contra_Costa')

    def source_reference(self, arrays, config):
        return dict(validation_axes={name: evaluation.tensor_tree_hash(arrays[field])
            for name, field in (('target_tensor_sha256', 'target'),
                ('sample_ids_sha256', 'sample_ids'), ('station_ids_sha256', 'station_ids'))},
            native_metrics=native_summary(arrays['prediction'], arrays['target']),
            selection_metrics=evaluation.numpy_metrics(arrays['prediction'], arrays['target'], config))

    def test_source_validation_reference_authenticates_conversion_contract(self):
        arrays, _, _, _ = self.cache_fixture()
        reference = self.source_reference(arrays, toy_config())
        metadata = dict(reference, conversion_validation_required=True,
            source_metrics_are_reference_only=True, copied_prediction_cache=False)
        identity = dict(dataset='Contra_Costa', external_full_reference=dict(
            directory='promoted', metadata_sha256='meta', model_sha256='model'))
        checksum = lambda path: 'meta' if Path(path).name == 'metadata.json' else 'model'
        with (patch.object(evaluation, 'sha256', side_effect=checksum),
              patch.object(evaluation, 'load_promoted', return_value=({}, {}, metadata))):
            observed = evaluation.converted_source_validation(identity, 'FULL')
            self.assertEqual(observed, reference)
            self.assertIsNone(evaluation.converted_source_validation(identity, 'UP_ONLY'))
            for key, value in (('conversion_validation_required', False),
                               ('source_metrics_are_reference_only', False),
                               ('copied_prediction_cache', True)):
                with self.subTest(key=key), patch.object(evaluation, 'load_promoted',
                    return_value=({}, {}, dict(metadata, **{key:value}))), self.assertRaises(ValueError):
                    evaluation.converted_source_validation(identity, 'FULL')

    def test_converted_forward_uses_actual_compact_arrays_and_fixed_tolerance(self):
        arrays, metadata, _, _ = self.cache_fixture()
        reference = self.source_reference(arrays, toy_config())
        # Source metrics are a fixed reference, never the compact model's report.
        reference['selection_metrics']['mae'] += .0005
        reference['native_metrics']['horizon_macro']['mae'] += .0005
        replay = evaluation.validate_converted_forward(arrays,
            metadata['common_mask_pooled'], metadata['native_metrics'], reference)
        self.assertTrue(replay['passed'])
        self.assertTrue(replay['actual_compact_model_forward'])
        self.assertFalse(replay['source_prediction_cache_reused'])
        self.assertEqual(replay['tolerance'], 1e-3)
        self.assertEqual(replay['new_training_updates'], 0)
        self.assertFalse(replay['new_checkpoint_selection'])
        self.assertNotEqual(reference['native_metrics'], metadata['native_metrics'])
        for key in reference['validation_axes']:
            candidate = copy.deepcopy(reference)
            candidate['validation_axes'][key] = 'changed'
            with self.subTest(axis=key), self.assertRaisesRegex(ValueError, 'axis differs'):
                evaluation.validate_converted_forward(arrays, metadata['common_mask_pooled'],
                    metadata['native_metrics'], candidate)
        for part in ('selection_metrics', 'native_metrics'):
            candidate = copy.deepcopy(reference)
            target = candidate[part] if part == 'selection_metrics' else candidate[part]['horizon_macro']
            target['mae'] += .01
            with self.subTest(part=part), self.assertRaises(ValueError):
                evaluation.validate_converted_forward(arrays, metadata['common_mask_pooled'],
                    metadata['native_metrics'], candidate)
        candidate = copy.deepcopy(reference)
        candidate['native_metrics']['per_horizon'][5]['rmse'] += .01
        with self.assertRaisesRegex(ValueError, 'native metrics'):
            evaluation.validate_converted_forward(arrays, metadata['common_mask_pooled'],
                metadata['native_metrics'], candidate)

    def test_promoted_reference_tamper_is_rejected_before_loading(self):
        identity = dict(dataset='Contra_Costa', external_full_reference=dict(
            directory='promoted', metadata_sha256='expected-meta', model_sha256='expected-model'))
        with (patch.object(evaluation, 'sha256', return_value='changed'),
              patch.object(evaluation, 'load_promoted') as load,
              self.assertRaisesRegex(ValueError, 'Promoted FULL reference changed')):
            evaluation.selected_best(Path('run'), identity, 'FULL')
        load.assert_not_called()

    def test_validation_replay_rejects_metrics_counts_and_missing_values(self):
        expected = dict(mae=12., rmse=20., valid_count=17, mape_valid_count=15,
                        region_mae=None, region_valid_count=0)
        self.assertTrue(evaluation.replay_validation(expected, expected)['passed'])
        for key, value in (('mae', 12.01), ('valid_count', 18), ('region_mae', 0.)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                evaluation.replay_validation(dict(expected, **{key:value}), expected)

    def cache_fixture(self):
        config = toy_config()
        target = np.full((2, 12, 3, 1), 20., dtype=np.float32)
        arrays = dict(prediction=target+2, target=target,
                      sample_ids=np.asarray([3909, 3910], dtype=np.int64),
                      station_ids=np.asarray([11, 12, 13], dtype=np.int64))
        role = dict(model_tensor_sha256='weights', epoch=100,
                    completed_training_epochs=100, selection='best')
        key = 's2025__FULL'
        frozen = dict(frozen_sha256='frozen', split='val', records={key:dict(role=copy.deepcopy(role))})
        metadata = dict(schema=evaluation.SCHEMA, frozen_sha256='frozen', record_key=key,
            checkpoint_role=role, config=config, split='val', metric_contract=evaluation.METRIC_CONTRACT,
            native_metrics=native_summary(arrays['prediction'], arrays['target']),
            common_mask_pooled=evaluation.numpy_metrics(arrays['prediction'], arrays['target'], config),
            target_tensor_sha256=evaluation.tensor_tree_hash(arrays['target']),
            sample_ids_sha256=evaluation.tensor_tree_hash(arrays['sample_ids']),
            station_ids_sha256=evaluation.tensor_tree_hash(arrays['station_ids']),
            audit=dict(model_tensor_sha256='weights', registered_tensors_unchanged=True,
                rng_unchanged=True, raw_inputs_unchanged=True,
                future_target_passed_to_forward=False, optimizer_updates=0))
        return arrays, metadata, frozen, key

    def test_complete_cache_read_needs_no_model_construction_or_forward(self):
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

    def test_compact_full_cache_requires_real_replay_even_without_saved_arrays(self):
        arrays, metadata, frozen, key = self.cache_fixture()
        source = self.source_reference(arrays, toy_config())
        frozen['records'][key]['source_validation'] = source
        metadata.update(split='val', converted_model_replayed=True,
            conversion_validation_replay=evaluation.validate_converted_forward(
                arrays, metadata['common_mask_pooled'], metadata['native_metrics'], source))
        for save in (False, True):
            with self.subTest(save=save), tempfile.TemporaryDirectory() as tmp:
                destination = Path(tmp)/key
                evaluation.commit_cache(destination, arrays, metadata, save, frozen, key)
                self.assertTrue(evaluation.read_cache(destination, frozen, key)['converted_model_replayed'])
                saved = evaluation.read_json(destination/'metadata.json')
                saved['converted_model_replayed'] = False
                saved['payload_sha256'] = evaluation.cache_digest(saved)
                evaluation.write_json(destination/'metadata.json', saved)
                with self.assertRaisesRegex(ValueError, 'actual compact-model replay'):
                    evaluation.read_cache(destination, frozen, key)

    def test_report_uses_actual_compact_forward_metrics_and_preserves_source_epoch(self):
        arrays, metadata, frozen, key = self.cache_fixture()
        metadata.update(arm='FULL', seed=2025, converted_model_replayed=True,
            parameters=dict(registered=73357, structurally_active=73357, coefficient_registered=1586))
        metadata['checkpoint_role']['epoch'] = 99
        frozen.update(arms=['FULL'], seeds=[2025], dataset='Contra_Costa', split='val',
                      engineering_only=False)
        with tempfile.TemporaryDirectory() as tmp:
            evaluation.render_report(Path(tmp), [metadata], frozen)
            report = (Path(tmp)/'report.md').read_text(encoding='utf-8')
            self.assertIn('PASE-Net (compact MLP)', report)
            self.assertIn('| 2025 | 99 | 100 | 2.000000 |', report)
            self.assertIn('73357', report)
            self.assertIn('1e-3', report)
            self.assertNotIn('原CONCAT_MLP权重', report)

    def conversion_import_fixture(self, tmp, save_predictions=False):
        arrays, metadata, source, key = self.cache_fixture()
        reference = self.source_reference(arrays, toy_config())
        source['records'][key].update(identity_sha256='training-identity',
            expected_validation=reference['selection_metrics'], source_validation=reference)
        source.update(schema=evaluation.SCHEMA, seeds=[2025], arms=['FULL'], dataset='Contra_Costa',
            split='val', engineering_only=False, runtime={'2025':{'device':'cpu'}},
            metric_contract=evaluation.METRIC_CONTRACT, evaluation_source={'probe.py':'checksum'})
        source['frozen_sha256'] = evaluation.digest_json(
            {name:value for name, value in source.items() if name != 'frozen_sha256'})
        metadata.update(arm='FULL', seed=2025, identity_sha256='training-identity',
            frozen_sha256=source['frozen_sha256'], converted_model_replayed=True,
            conversion_validation_replay=evaluation.validate_converted_forward(
                arrays, metadata['common_mask_pooled'], metadata['native_metrics'], reference))
        replay = Path(tmp)/'compact_full_replay'
        replay.mkdir()
        evaluation.write_json(replay/'frozen.json', source)
        evaluation.commit_cache(replay/key, arrays, metadata, save_predictions, source, key)
        target = copy.deepcopy(source)
        target['arms'] = ['FULL', 'UP_ONLY', 'DOWN_ONLY']
        target['frozen_sha256'] = evaluation.digest_json(
            {name:value for name, value in target.items() if name != 'frozen_sha256'})
        return replay, target, key, arrays

    def test_certified_compact_full_can_populate_three_arm_report_without_new_forward(self):
        for saved_arrays in (False, True):
            with self.subTest(saved_arrays=saved_arrays), tempfile.TemporaryDirectory() as tmp:
                replay, target, key, arrays = self.conversion_import_fixture(tmp, saved_arrays)
                destination = Path(tmp)/'final_report'/key
                with (patch.object(evaluation, 'evaluation_sources', return_value=target['evaluation_source']),
                      patch.object(evaluation, 'build_model', side_effect=AssertionError('new model')),
                      patch.object(evaluation, 'collect', side_effect=AssertionError('new forward')),
                      patch.object(evaluation, 'load_data', side_effect=AssertionError('new data read'))):
                    imported = evaluation.import_conversion_cache(replay, destination, target, key, saved_arrays)
                self.assertEqual(imported['native_metrics'], native_summary(arrays['prediction'], arrays['target']))
                self.assertTrue(imported['converted_model_replayed'])
                self.assertEqual(imported['imported_from']['new_forward_batches'], 0)
                self.assertFalse(imported['imported_from']['original_v2_prediction_cache'])
                self.assertEqual(imported['frozen_sha256'], target['frozen_sha256'])
                self.assertEqual(imported['imported_from']['metadata_sha256'],
                                 evaluation.sha256(replay/key/'metadata.json'))
                self.assertEqual(evaluation.read_cache(destination, target, key), imported)
                self.assertEqual((destination/'predictions.npz').exists(), saved_arrays)

    def test_conversion_cache_import_rejects_changed_runtime_role_identity_or_v2_schema(self):
        for change in ('runtime', 'role', 'identity', 'source', 'axes', 'schema'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                replay, target, key, _ = self.conversion_import_fixture(tmp)
                old = evaluation.read_json(replay/'frozen.json')
                if change == 'runtime':
                    old['runtime']['2025']['device'] = 'changed-device'
                elif change == 'role':
                    old['records'][key]['role']['epoch'] = 97
                elif change == 'identity':
                    old['records'][key]['identity_sha256'] = 'different-training-run'
                elif change == 'source':
                    old['evaluation_source']['probe.py'] = 'changed-source'
                elif change == 'axes':
                    old['records'][key]['source_validation']['validation_axes']['sample_ids_sha256'] = 'reordered'
                else:
                    old['schema'] = 'pase_mlp_v2_best_evaluation_v1'
                old['frozen_sha256'] = evaluation.digest_json(
                    {name:value for name, value in old.items() if name != 'frozen_sha256'})
                evaluation.write_json(replay/'frozen.json', old)
                destination = Path(tmp)/'final_report'/key
                with (patch.object(evaluation, 'evaluation_sources', return_value=target['evaluation_source']),
                      self.assertRaises(ValueError)):
                    evaluation.import_conversion_cache(replay, destination, target, key)
                self.assertFalse(destination.exists())

    def test_prediction_import_requires_source_arrays_and_live_source_dependency_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            replay, target, key, _ = self.conversion_import_fixture(tmp)
            destination = Path(tmp)/'final_report'/key
            with (patch.object(evaluation, 'evaluation_sources', return_value=target['evaluation_source']),
                  self.assertRaisesRegex(ValueError, 'no prediction arrays')):
                evaluation.import_conversion_cache(replay, destination, target, key, True)
            self.assertFalse(destination.exists())
            with (patch.object(evaluation, 'evaluation_sources', return_value={'probe.py':'changed'}),
                  self.assertRaisesRegex(ValueError, 'numerical source changed')):
                evaluation.import_conversion_cache(replay, destination, target, key)
            self.assertFalse(destination.exists())

    def test_cache_rejects_frozen_selection_metadata_and_prediction_byte_changes(self):
        arrays, metadata, frozen, key = self.cache_fixture()
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp)/key
            evaluation.commit_cache(destination, arrays, metadata, True, frozen, key)
            with self.assertRaisesRegex(ValueError, 'another checkpoint'):
                evaluation.read_cache(destination, dict(frozen, frozen_sha256='changed-best'), key)
            saved = evaluation.read_json(destination/'metadata.json')
            saved['checkpoint_role']['epoch'] = 99
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

    def test_cache_rejects_changed_target_or_axes_even_after_file_checksum_refresh(self):
        arrays, metadata, frozen, key = self.cache_fixture()
        for field in ('target', 'sample_ids', 'station_ids'):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                destination = Path(tmp)/key
                evaluation.commit_cache(destination, arrays, metadata, True, frozen, key)
                changed = {name:value.copy() for name, value in arrays.items()}
                changed[field].flat[0] += 1
                np.savez_compressed(destination/'predictions.npz', **changed)
                saved = evaluation.read_json(destination/'metadata.json')
                saved['prediction_sha256'] = evaluation.sha256(destination/'predictions.npz')
                saved['payload_sha256'] = evaluation.cache_digest(saved)
                evaluation.write_json(destination/'metadata.json', saved)
                with self.assertRaisesRegex(ValueError, 'axes or target differ'):
                    evaluation.read_cache(destination, frozen, key)

    def test_interrupted_write_is_not_a_complete_cache_and_does_not_block_retry(self):
        arrays, metadata, frozen, key = self.cache_fixture()
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp)/key
            with patch.object(evaluation, 'write_json', side_effect=OSError('interrupted')):
                with self.assertRaises(OSError):
                    evaluation.commit_cache(destination, arrays, metadata, False, frozen, key)
            self.assertFalse(destination.exists())
            evaluation.commit_cache(destination, arrays, metadata, False, frozen, key)
            self.assertEqual(evaluation.read_cache(destination, frozen, key)['record_key'], key)

    def test_output_cannot_intersect_data_or_seed_training(self):
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

    def test_collect_respects_each_dataset_split_offset_and_only_passes_history_event(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.paper_arm = self.arm = 'FULL'
                self.geometry = RegionGeometry([0, 0, 1])
                self.calls = 0

            def forward(self, history, event):
                self.calls += 1
                if history.shape != (2, 12, 3, 3) or set(event) != {'probe'}:
                    raise AssertionError('Unexpected history/event input or future labels')
                if not bool((history == 20.).all() and (event['probe'] == 17).all()):
                    raise AssertionError('History or event changed')
                prediction = history[..., :1]*2.+10.
                return dict(prediction=prediction, reference=prediction)

        class Loader:
            size, num_batch = 4, 2

            def get_iterator(self):
                for _ in range(2):
                    yield dict(x_data=np.full((2, 12, 3, 3), 20., dtype=np.float32),
                               y_data=np.full((2, 12, 3, 1), 21., dtype=np.float32),
                               event_data={'probe':np.full((2,), 17, dtype=np.int64)})

        offsets = {'Contra_Costa':(3909, 4746), 'Alameda':(10280, 12483),
                   'Orange':(13090, 15895)}
        for dataset, (validation_offset, test_offset) in offsets.items():
            data = SimpleNamespace(identity={'dataset':dataset}, station_ids=[11, 12, 13], mean=10., std=2.)
            for split, offset in (('val', validation_offset), ('test', test_offset)):
                model = Model()
                with self.subTest(dataset=dataset, split=split), patch.object(
                        evaluation, 'validation_loader', return_value=(Loader(), offset)) as loader:
                    arrays, metrics, audit = evaluation.collect(model, data, toy_config(), split, 'cpu',
                                                               logging.getLogger('toy-mlp'))
                loader.assert_called_once_with(data, split, 2, logging.getLogger('toy-mlp'))
                self.assertEqual(model.calls, 2)
                self.assertEqual(arrays['sample_ids'].tolist(), list(range(offset, offset+4)))
                self.assertEqual(arrays['station_ids'].tolist(), [11, 12, 13])
                self.assertEqual(metrics['mae'], 2.)
                self.assertTrue(np.all(arrays['target'] == 52.))
                self.assertFalse(audit['future_target_passed_to_forward'])
                self.assertEqual(audit['optimizer_updates'], 0)
                self.assertTrue(audit['rng_unchanged'])
                self.assertEqual(audit['forward_batches'], 2)

    def test_collect_engineering_prefix_and_incomplete_loader_are_explicit(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.paper_arm = self.arm = 'FULL'
                self.geometry = RegionGeometry([0, 0, 1])
                self.calls = 0

            def forward(self, history, event):
                self.calls += 1
                return dict(prediction=history[..., :1], reference=history[..., :1])

        class Loader:
            size, num_batch = 4, 2

            def get_iterator(self):
                for _ in range(2):
                    yield dict(x_data=np.full((2, 12, 3, 3), 20., dtype=np.float32),
                               y_data=np.full((2, 12, 3, 1), 21., dtype=np.float32), event_data={})

        class ShortLoader(Loader):
            def get_iterator(self):
                yield next(super().get_iterator())

        data = SimpleNamespace(station_ids=[11, 12, 13], mean=0., std=1.)
        model = Model()
        with patch.object(evaluation, 'validation_loader', return_value=(Loader(), 3909)):
            arrays, _, audit = evaluation.collect(model, data, toy_config(), 'val', 'cpu',
                                                  logging.getLogger('toy-mlp'), max_batches=1)
        self.assertEqual(model.calls, 1)
        self.assertEqual(arrays['sample_ids'].tolist(), [3909, 3910])
        self.assertEqual(audit['forward_batches'], 1)
        with (patch.object(evaluation, 'validation_loader', return_value=(ShortLoader(), 3909)),
              self.assertRaisesRegex(ValueError, 'declared number of batches')):
            evaluation.collect(model, data, toy_config(), 'val', 'cpu', logging.getLogger('toy-mlp'))


if __name__ == '__main__':
    unittest.main()
