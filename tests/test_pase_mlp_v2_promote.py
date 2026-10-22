"""Promotion uses legacy certificates without new model forwards or optimizers."""
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from experiments.pase_mlp_v2 import promote
from experiments.pase_paper_experiments import train as legacy_train
from experiments.pase_paper_experiments import state as legacy_state
from experiments.position_state_gw16.common import (
    REPO, atomic_save, digest_json, read_json, runtime, sha256, tensor_tree_hash, write_json)
from experiments.state_exchange_multidata.common import load_config
from tests.test_pase_paper_experiments_training import TinyModel, tiny_accounting


class MLPV2PromotionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.run, self.evaluation, self.output = (self.root/'old_run', self.root/'old_eval', self.root/'v2')
        self.run.mkdir()
        self.evaluation.mkdir()
        self.config = load_config('Contra_Costa')
        self.source_name = 'experiments/pase_paper_experiments/state.py'
        sources = {self.source_name: sha256(REPO/self.source_name)}
        batching = dict(train_loader=dict(samples=4, batches=2, batch_size=48),
                        val_loader=dict(samples=837, batches=18, batch_size=48))
        data = SimpleNamespace(identity={'loader_batching': batching},
            position_identity={}, partition={})
        with patch.object(legacy_train, 'build_model', side_effect=lambda *a: TinyModel()), \
             patch.object(legacy_train, 'parameter_accounting', side_effect=tiny_accounting), \
             patch.object(legacy_train, 'source_hashes', return_value=sources):
            self.identity = legacy_train.make_identity(data, self.config, 2025)
        write_json(self.run/'identity.json', self.identity)
        self.runtime = runtime('cpu', 1, 2025)
        # Training extras need not be present in evaluation runtime.
        write_json(self.run/'runtime.json', dict(self.runtime, conda_default_env='igstgnn'))
        self.weights = {'head.weight': torch.tensor([[1., -2.]], dtype=torch.float32)}
        self.selection = dict(mae=12.4, rmse=21.6, mape_percent=17.1, signed_bias=0.,
                              region_mae=5.2, reference_mae=20., region_valid_count=100)
        arm_dir = self.run/'CONCAT_MLP'
        arm_dir.mkdir()
        best = arm_dir/'best_epoch_100_fixture.pt'
        atomic_save(best, dict(model_state=self.weights))
        atomic_save(arm_dir/'last.pt', dict(history=[{'epoch': value} for value in range(101)]))
        self.role = dict(selection='best', epoch=100, completed_training_epochs=100,
            checkpoint_file=best.name, checkpoint_sha256=sha256(best),
            model_tensor_sha256=tensor_tree_hash(self.weights), best_includes_stage_zero=True,
            selected_on='validation_pooled_mae', full_protocol_reached=True)
        self.key = 's2025__CONCAT_MLP'
        self.frozen = dict(schema=promote.legacy_evaluate.SCHEMA, split='val', seeds=[2025],
            arms=['CONCAT_MLP'], engineering_only=False,
            metric_contract=promote.legacy_evaluate.METRIC_CONTRACT, runtime={'2025': self.runtime},
            evaluation_source=sources,
            records={self.key: dict(seed=2025, arm='CONCAT_MLP', role=self.role,
                expected_validation=self.selection, identity_sha256=self.identity['identity_sha256'])})
        self.write_frozen()
        native = dict(horizon_macro=dict(mae=12.4, rmse=21.5, mape_percent=17.1),
            per_horizon=[dict(lead=value, mae=12.4, rmse=21.5, mape_percent=17.1)
                         for value in range(1, 13)])
        self.cached = dict(schema=promote.legacy_evaluate.SCHEMA,
            frozen_sha256=self.frozen['frozen_sha256'], record_key=self.key,
            checkpoint_role=self.role, metric_contract=promote.legacy_evaluate.METRIC_CONTRACT,
            identity_sha256=self.identity['identity_sha256'], config=self.config, split='val',
            test_evaluated=False, selection_metric_replay={'passed': True}, native_metrics=native,
            common_mask_pooled={name:value for name, value in self.selection.items()
                                if name not in ('region_mae', 'reference_mae', 'region_valid_count')},
            parameters={'registered': 73869},
            audit=dict(samples=837, model_tensor_sha256=self.role['model_tensor_sha256'],
                registered_tensors_unchanged=True, rng_unchanged=True, raw_inputs_unchanged=True,
                future_target_passed_to_forward=False, optimizer_updates=0),
            target_tensor_sha256='target-axis', sample_ids_sha256='sample-axis',
            station_ids_sha256='station-axis', prediction_file=None, prediction_sha256=None)
        self.write_cache()
        self.selection_patch = patch.object(promote.legacy_evaluate, 'selected_best',
            return_value=(self.weights, self.role, self.selection))
        self.selection_mock = self.selection_patch.start()
        self.addCleanup(self.selection_patch.stop)

    def write_frozen(self):
        self.frozen['frozen_sha256'] = digest_json({key: value for key, value in self.frozen.items()
                                                   if key != 'frozen_sha256'})
        write_json(self.evaluation/'frozen.json', self.frozen)

    def write_cache(self):
        directory = self.evaluation/self.key
        directory.mkdir(exist_ok=True)
        self.cached['payload_sha256'] = promote.legacy_evaluate.cache_digest(self.cached)
        write_json(directory/'metadata.json', self.cached)

    def run_promotion(self):
        return promote.promote(self.run, self.evaluation, self.output)

    def test_exports_exact_tensor_bits_and_never_creates_resume_state(self):
        before = {str(path.relative_to(self.root)): sha256(path)
                  for directory in (self.run, self.evaluation) for path in directory.rglob('*')
                  if path.is_file()}
        metadata = self.run_promotion()
        weights, role, loaded = promote.load_promoted(self.output, 'Contra_Costa')
        self.assertTrue(torch.equal(weights['head.weight'], self.weights['head.weight']))
        self.assertEqual(metadata, loaded)
        self.assertEqual(role['role_kind'], 'external_promoted_inference')
        self.assertEqual(metadata['new_model_forwards'], 0)
        self.assertEqual(metadata['new_optimizer_updates'], 0)
        saved = promote.load_state(self.output/'model.pt')
        self.assertNotIn('optimizer_state', saved)
        self.assertFalse((self.output/'last.pt').exists())
        self.assertFalse((self.output/'identity.json').exists())
        for name, checksum in before.items():
            self.assertEqual(sha256(self.root/name), checksum)
        self.selection_mock.assert_called_once()

    def test_repeat_promotion_reuses_immutable_export(self):
        first = self.run_promotion()
        checksum = sha256(self.output/'model.pt')
        second = self.run_promotion()
        self.assertEqual(first, second)
        self.assertEqual(sha256(self.output/'model.pt'), checksum)

    def test_corrupt_export_and_wrong_dataset_rejected(self):
        self.run_promotion()
        with self.assertRaisesRegex(ValueError, 'dataset differs'):
            promote.load_promoted(self.output, 'Orange')
        (self.output/'model.pt').write_bytes(b'broken')
        with self.assertRaisesRegex(ValueError, 'checksum changed'):
            promote.load_promoted(self.output)

    def test_incomplete_legacy_training_cannot_be_promoted(self):
        role = dict(self.role, completed_training_epochs=99, full_protocol_reached=False)
        self.selection_mock.return_value = (self.weights, role, self.selection)
        with self.assertRaisesRegex(ValueError, '100-epoch budget'):
            self.run_promotion()
        self.assertFalse(self.output.exists())

    def test_test_split_and_tampered_cache_rejected(self):
        self.frozen['split'] = 'test'
        self.write_frozen()
        with self.assertRaisesRegex(ValueError, 'validation evaluation'):
            self.run_promotion()
        self.frozen['split'] = 'val'
        self.write_frozen()
        self.cached['native_metrics']['horizon_macro']['mae'] = 999.
        # Do not repair its cryptographic metadata payload.
        write_json(self.evaluation/self.key/'metadata.json', self.cached)
        with self.assertRaisesRegex(ValueError, 'checksum differs'):
            self.run_promotion()

    def test_recorded_sources_checked_individually_and_cannot_escape_repo(self):
        promote.verify_recorded_sources({self.source_name: sha256(REPO/self.source_name)})
        with self.assertRaisesRegex(ValueError, 'source changed'):
            promote.verify_recorded_sources({self.source_name: '0'*64})
        with self.assertRaisesRegex(ValueError, 'escapes'):
            promote.verify_recorded_sources({'../outside.py': '0'*64})

    def test_source_directories_cannot_be_overwritten(self):
        for output in (self.run, self.run/'promoted', self.evaluation, self.root):
            with self.subTest(output=output), self.assertRaisesRegex(ValueError, 'must not intersect'):
                promote.promote(self.run, self.evaluation, output)


if __name__ == '__main__':
    unittest.main()
