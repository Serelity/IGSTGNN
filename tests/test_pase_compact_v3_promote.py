"""Compact export authenticates v2 provenance and never invents new predictions."""
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from experiments.pase_compact_v3 import promote
from experiments.pase_mlp_v2 import train as source_train
from experiments.position_state_gw16.common import (
    REPO, atomic_save, digest_json, read_json, runtime, sha256, tensor_tree_hash, write_json)
from experiments.state_exchange_multidata.common import load_config
from tests.test_pase_paper_experiments_training import TinyModel, tiny_accounting


class CompactV3PromotionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.run, self.evaluation, self.output = (
            self.root/'v2_run', self.root/'v2_eval', self.root/'compact')
        self.run.mkdir()
        self.evaluation.mkdir()
        self.config = load_config('Contra_Costa')
        self.source_name = 'experiments/pase_mlp_v2/state.py'
        sources = {self.source_name: sha256(REPO/self.source_name)}
        batching = dict(train_loader=dict(samples=4, batches=2, batch_size=48),
                        val_loader=dict(samples=837, batches=18, batch_size=48))
        data = SimpleNamespace(identity={'loader_batching': batching},
                               position_identity={}, partition={})
        with patch.object(source_train, 'build_model', side_effect=lambda *args: TinyModel()), \
             patch.object(source_train, 'parameter_accounting', side_effect=tiny_accounting), \
             patch.object(source_train, 'source_hashes', return_value=sources):
            self.identity = source_train.make_identity(data, self.config, 2025)
        write_json(self.run/'identity.json', self.identity)
        self.runtime = runtime('cpu', 1, 2025)
        write_json(self.run/'runtime.json', dict(self.runtime, conda_default_env='igstgnn'))
        self.weights = {
            'encoder.fixture': torch.tensor([[1., -2.]], dtype=torch.float32),
            promote.FIRST_WEIGHT: torch.arange(16*128, dtype=torch.float32).reshape(16, 128),
            'head.exchange.network.0.bias': torch.arange(16, dtype=torch.float32),
            'head.exchange.network.2.weight': torch.arange(32, dtype=torch.float32).reshape(2, 16),
            'head.exchange.network.2.bias': torch.tensor([.2, -.1], dtype=torch.float32)}
        self.selection = dict(mae=12.459936, rmse=21.6, mape_percent=17.222980,
            signed_bias=0., region_mae=5.2, reference_mae=20., region_valid_count=100)
        directory = self.run/promote.SOURCE_ARM
        directory.mkdir()
        best = directory/'best_epoch_099_fixture.pt'
        atomic_save(best, dict(model_state=self.weights))
        atomic_save(directory/'last.pt', dict(history=[{'epoch': value} for value in range(101)]))
        self.role = dict(selection='best', epoch=99, completed_training_epochs=100,
            checkpoint_file=best.name, checkpoint_sha256=sha256(best),
            model_tensor_sha256=tensor_tree_hash(self.weights), best_includes_stage_zero=True,
            selected_on='validation_pooled_mae', full_protocol_reached=True)
        self.key = 's2025__'+promote.SOURCE_ARM
        self.frozen = dict(schema=promote.source_evaluate.SCHEMA, dataset='Contra_Costa',
            split='val', seeds=[2025], arms=[promote.SOURCE_ARM], engineering_only=False,
            metric_contract=promote.source_evaluate.METRIC_CONTRACT, runtime={'2025': self.runtime},
            evaluation_source=sources, records={self.key: dict(seed=2025,
                arm=promote.SOURCE_ARM, role=self.role, expected_validation=self.selection,
                identity_sha256=self.identity['identity_sha256'])})
        self.write_frozen()
        native = dict(horizon_macro=dict(mae=12.459936, rmse=21.494183, mape_percent=17.222980),
            per_horizon=[dict(lead=value, mae=12.459936, rmse=21.494183, mape_percent=17.222980)
                         for value in range(1, 13)])
        self.parameters = dict(registered=73869, trainable=73869, structurally_active=73357,
            coefficient_registered=2098, coefficient_active=1586, inactive_registered=512,
            parameter_bytes=73869*4, buffer_bytes=20,
            active_rule='MLP_shared_hidden_layer_partial_output_rows_and_zero_input_columns',
            one_way_unused_coefficient_rows_retained=False,
            all_registered_parameters_included_when_frozen=True)
        self.cached = dict(schema=promote.source_evaluate.SCHEMA,
            frozen_sha256=self.frozen['frozen_sha256'], record_key=self.key,
            checkpoint_role=self.role, metric_contract=promote.source_evaluate.METRIC_CONTRACT,
            identity_sha256=self.identity['identity_sha256'], config=self.config, split='val',
            test_evaluated=False, selection_metric_replay={'passed': True}, native_metrics=native,
            common_mask_pooled={name: value for name, value in self.selection.items()
                if name not in ('region_mae', 'reference_mae', 'region_valid_count')},
            parameters=self.parameters,
            audit=dict(samples=837, model_tensor_sha256=self.role['model_tensor_sha256'],
                registered_tensors_unchanged=True, rng_unchanged=True, raw_inputs_unchanged=True,
                future_target_passed_to_forward=False, optimizer_updates=0),
            target_tensor_sha256='target-axis', sample_ids_sha256='sample-axis',
            station_ids_sha256='station-axis', prediction_file=None, prediction_sha256=None)
        self.write_cache()
        selected = patch.object(promote.source_evaluate, 'selected_best',
            return_value=(self.weights, self.role, self.selection))
        self.selection_mock = selected.start()
        self.addCleanup(selected.stop)

    def write_frozen(self):
        self.frozen['frozen_sha256'] = digest_json({key: value for key, value in self.frozen.items()
                                                  if key != 'frozen_sha256'})
        write_json(self.evaluation/'frozen.json', self.frozen)

    def write_cache(self, key=None, cached=None):
        key, cached = key or self.key, self.cached if cached is None else cached
        directory = self.evaluation/key
        directory.mkdir(exist_ok=True)
        cached['payload_sha256'] = promote.source_evaluate.cache_digest(cached)
        write_json(directory/'metadata.json', cached)

    def run_promotion(self):
        return promote.promote(self.run, self.evaluation, self.output)

    def test_exports_sliced_weights_preserves_retained_bits_without_forward_or_training(self):
        before = {str(path.relative_to(self.root)): sha256(path)
            for directory in (self.run, self.evaluation) for path in directory.rglob('*')
            if path.is_file()}
        with patch.object(promote.source_evaluate, 'collect', side_effect=AssertionError('new forward')):
            metadata = self.run_promotion()
        weights, role, loaded = promote.load_promoted(self.output, 'Contra_Costa')
        for name, original in self.weights.items():
            expected = original[:, 32:] if name == promote.FIRST_WEIGHT else original
            self.assertTrue(torch.equal(weights[name], expected))
        self.assertEqual(tuple(weights[promote.FIRST_WEIGHT].shape), (16, 96))
        self.assertEqual(metadata, loaded)
        self.assertEqual(role['role_kind'], 'external_compact_inference')
        self.assertEqual(role['source_arm'], 'NO_POSITION_INPUT')
        self.assertEqual(role['epoch'], 99)
        self.assertEqual(role['model_tensor_sha256'], tensor_tree_hash(weights))
        self.assertEqual(role['source_model_tensor_sha256'], tensor_tree_hash(self.weights))
        self.assertNotEqual(role['model_tensor_sha256'], role['source_model_tensor_sha256'])
        self.assertEqual(role['metadata_sha256'], sha256(self.output/'metadata.json'))
        self.assertEqual(metadata['new_model_forwards'], 0)
        self.assertEqual(metadata['new_optimizer_updates'], 0)
        self.assertTrue(metadata['conversion_validation_required'])
        self.assertTrue(metadata['source_metrics_are_reference_only'])
        self.assertFalse(metadata['copied_prediction_cache'])
        self.assertFalse(metadata['conversion']['bitwise_prediction_equivalence_claimed'])
        saved = promote.load_state(self.output/'model.pt')
        self.assertNotIn('optimizer_state', saved)
        self.assertFalse((self.output/'last.pt').exists())
        self.assertFalse((self.output/'identity.json').exists())
        self.assertFalse((self.output/'predictions.npz').exists())
        for name, checksum in before.items():
            self.assertEqual(sha256(self.root/name), checksum)
        self.selection_mock.assert_called_once_with(self.run, self.identity, 'NO_POSITION_INPUT')

    def test_accounting_deletes_real_columns_and_keeps_source_metrics_separate(self):
        metadata = self.run_promotion()
        counts = metadata['parameters']
        self.assertEqual(counts['registered'], 73357)
        self.assertEqual(counts['trainable'], 73357)
        self.assertEqual(counts['coefficient_registered'], 1586)
        self.assertEqual(counts['coefficient_active'], 1586)
        self.assertEqual(counts['structurally_active'], 73357)
        self.assertEqual(counts['inactive_registered'], 0)
        self.assertEqual(counts['parameter_bytes'], 73357*4)
        self.assertEqual(metadata['cached_metadata']['parameters']['registered'], 73869)
        self.assertEqual(metadata['selection_metrics'], self.selection)

    def test_compact_accounting_matches_the_actual_model_including_all_field_names(self):
        from experiments.pase_compact_v3 import model as compact_model
        from experiments.pase_mlp_v2 import model as v2_model
        from tests.test_position_state_gw16_model import fixture
        data, config, _, _ = fixture()
        config.update(exchange_rank=8, horizon=3)
        torch.manual_seed(2025)
        source = v2_model.build_model(data, config, 'NO_POSITION_INPUT', 'cpu')
        converted = compact_model.compact_state_dict(source.state_dict(), config['hidden'])
        torch.manual_seed(2025)
        actual = compact_model.build_model(data, config, 'FULL', 'cpu')
        actual.load_state_dict(converted, strict=True)
        certificate = promote.compact_parameters(v2_model.parameter_accounting(source),
                                                 config['hidden'], converted)
        self.assertEqual(certificate, compact_model.parameter_accounting(actual))

    def test_repeat_promotion_reuses_immutable_export(self):
        first = self.run_promotion()
        checksum = sha256(self.output/'model.pt')
        second = self.run_promotion()
        self.assertEqual(first, second)
        self.assertEqual(sha256(self.output/'model.pt'), checksum)
        self.assertEqual(promote.reference_identity(self.output)['source_arm'], 'NO_POSITION_INPUT')

    def test_wrong_dataset_and_corrupt_artifact_rejected(self):
        self.run_promotion()
        with self.assertRaisesRegex(ValueError, 'dataset differs'):
            promote.load_promoted(self.output, 'Orange')
        (self.output/'model.pt').write_bytes(b'broken')
        with self.assertRaisesRegex(ValueError, 'checksum changed'):
            promote.load_promoted(self.output)

    def test_incomplete_budget_or_test_source_rejected(self):
        role = dict(self.role, completed_training_epochs=99, full_protocol_reached=False)
        self.selection_mock.return_value = (self.weights, role, self.selection)
        with self.assertRaisesRegex(ValueError, '100-epoch budget'):
            self.run_promotion()
        self.selection_mock.return_value = (self.weights, self.role, self.selection)
        self.frozen['split'] = 'test'
        self.write_frozen()
        with self.assertRaisesRegex(ValueError, 'validation evaluation'):
            self.run_promotion()
        self.assertFalse(self.output.exists())

    def test_best_epoch_is_authenticated_not_hardcoded(self):
        self.role['epoch'] = 97
        self.write_frozen()
        self.cached['frozen_sha256'] = self.frozen['frozen_sha256']
        self.write_cache()
        metadata = self.run_promotion()
        self.assertEqual(metadata['source_role']['epoch'], 97)

    def test_cache_tampering_and_missing_pair_rejected(self):
        self.cached['native_metrics']['horizon_macro']['mae'] = 999.
        write_json(self.evaluation/self.key/'metadata.json', self.cached)
        with self.assertRaisesRegex(ValueError, 'checksum differs'):
            self.run_promotion()
        self.write_cache()
        self.frozen['records']['s2025__FULL'] = dict(self.frozen['records'][self.key])
        self.write_frozen()
        self.cached['frozen_sha256'] = self.frozen['frozen_sha256']
        self.write_cache()
        with self.assertRaisesRegex(ValueError, 'paired evaluation is incomplete'):
            self.run_promotion()

    def test_source_parameter_certificate_cannot_claim_removed_weights_active(self):
        self.cached['parameters']['inactive_registered'] = 0
        self.write_cache()
        with self.assertRaisesRegex(ValueError, 'parameter certificate'):
            self.run_promotion()

    def test_source_axes_and_runtime_remain_fixed(self):
        self.cached['audit']['samples'] = 836
        self.write_cache()
        with self.assertRaisesRegex(ValueError, 'sample axis'):
            self.run_promotion()
        self.cached['audit']['samples'] = 837
        self.write_cache()
        altered = dict(self.runtime, threads=4)
        write_json(self.run/'runtime.json', altered)
        with self.assertRaisesRegex(ValueError, 'runtime/device/threads'):
            self.run_promotion()

    def test_repaired_metadata_cannot_change_source_metrics_or_conversion_shape(self):
        self.run_promotion()
        original = read_json(self.output/'metadata.json')
        for field in ('native_metrics', 'conversion'):
            metadata = copy.deepcopy(original)
            if field == 'native_metrics':
                metadata[field]['horizon_macro']['mae'] = 999.
                pattern = 'source certificate differs'
            else:
                metadata[field]['compact_shape'] = [16, 128]
                pattern = 'conversion or parameter accounting differs'
            metadata['metadata_sha256'] = promote.metadata_digest(metadata)
            write_json(self.output/'metadata.json', metadata)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, pattern):
                promote.load_promoted(self.output)

    def test_source_directories_cannot_be_overwritten(self):
        for output in (self.run, self.run/'promoted', self.evaluation, self.root):
            with self.subTest(output=output), self.assertRaisesRegex(ValueError, 'must not intersect'):
                promote.promote(self.run, self.evaluation, output)


if __name__ == '__main__':
    unittest.main()
