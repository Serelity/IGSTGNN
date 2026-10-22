"""Server/remote CPU tests: resume equivalence and explicit external FULL roles."""
import copy
import logging
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from experiments.pase_mlp_v2 import state, train
from experiments.pase_mlp_v2.protocol import ARMS, configuration
from experiments.state_exchange_multidata.common import DATASETS, load_config
from tests.test_pase_paper_experiments_training import (
    TinyLoader, TinyModel, tiny_accounting, tiny_phase)


class MLPV2TrainingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config = configuration('FULL', dataset='Contra_Costa')
        self.config.update(max_epochs=3, lr_milestones=[])
        batches = {key: {'batches': 2, 'batch_size': 2, 'samples': 4}
                   for key in ('train_loader', 'val_loader')}
        self.data = SimpleNamespace(loaders={'train_loader': TinyLoader(), 'val_loader': TinyLoader()},
            identity={'loader_batching': batches}, partition={'assignment': [0, 0]},
            position_identity={'fixture': 'position'})
        for item in (patch.object(train, 'build_model', side_effect=lambda *a: TinyModel()),
                     patch.object(train, 'parameter_accounting', side_effect=tiny_accounting),
                     patch.object(train, 'source_hashes', return_value={'fixture': 'fixed'}),
                     patch.object(train, 'run_phase', side_effect=tiny_phase)):
            item.start()
            self.addCleanup(item.stop)
        self.identity = train.make_identity(self.data, self.config, 2025, True, 2, 2)
        self.logger = logging.getLogger('pase-mlp-v2-test')

    def args(self, name):
        directory = self.root/name
        directory.mkdir()
        return SimpleNamespace(run_dir=directory, seed=2025, device='cpu')

    def run_arm(self, args, arm='FULL'):
        train.train_arm(args, self.data, self.config, self.identity, arm, self.logger)
        return state.load_state(args.run_dir/arm/'last.pt')

    def test_configuration_unchanged_for_each_dataset_and_seed_fixed(self):
        for dataset in DATASETS:
            config = configuration('FULL', dataset=dataset)
            self.assertEqual(config, load_config(dataset))
            state.validate_config(config)
            with self.assertRaises(ValueError):
                state.validate_config(dict(config, max_epochs=101))
        base = ['--run-dir', str(self.root/'run'), '--data-dir', str(self.root/'data')]
        self.assertEqual(train.parse_args(base).arms, ['FULL'])
        with self.assertRaises(SystemExit):
            train.parse_args([*base, '--seed', '2026'])
        with self.assertRaises(SystemExit):
            train.parse_args([*base, '--epochs', '150'])

    def test_registered_common_initializations_for_all_new_arms(self):
        state.validate_identity(self.identity)
        self.assertEqual(self.identity['comparison_arms'], list(ARMS))
        self.assertIsNone(self.identity['external_full_reference'])
        for name in ('encoder', 'incident', 'position', 'origin_condition'):
            self.assertEqual(len({value[name+'_tensor_sha256']
                for value in self.identity['initializations'].values()}), 1)

    def test_resume_preserves_optimizer_scheduler_rng_and_orders(self):
        whole = self.run_arm(self.args('whole'))
        resumed_args = self.args('resumed')
        original = train.collect_summary

        def interrupt(run, identity):
            result = original(run, identity)
            if result['arms']['FULL']['completed_epoch'] == 1:
                raise RuntimeError('committed-one')
            return result

        with patch.object(train, 'collect_summary', side_effect=interrupt):
            with self.assertRaisesRegex(RuntimeError, 'committed-one'):
                self.run_arm(resumed_args)
        resumed = self.run_arm(resumed_args)
        for key in ('model_state', 'optimizer_state', 'scheduler_state', 'rng_state', 'train_order'):
            self.assertEqual(state.tensor_tree_hash(whole[key]), state.tensor_tree_hash(resumed[key]))
        self.assertEqual(resumed['global_updates'], 6)

    def test_uncommitted_epoch_is_replayed_and_best_may_be_zero(self):
        args = self.args('partial')
        calls = [0]

        def interrupted(model, loader, data, config, device, optimizer=None):
            metrics, timing = tiny_phase(model, loader, data, config, device, optimizer)
            if optimizer is None:
                calls[0] += 1
                if calls[0] == 2:
                    raise RuntimeError('before-commit')
            return metrics, timing

        with patch.object(train, 'run_phase', side_effect=interrupted):
            with self.assertRaisesRegex(RuntimeError, 'before-commit'):
                self.run_arm(args)
        saved = state.load_state(args.run_dir/'FULL'/'last.pt')
        self.assertEqual((saved['completed_epoch'], saved['best_epoch']), (0, 0))
        resumed = self.run_arm(args)
        whole = self.run_arm(self.args('replay-whole'))
        self.assertEqual(state.tensor_tree_hash(resumed['model_state']),
                         state.tensor_tree_hash(whole['model_state']))

    def test_old_schema_and_corrupt_payload_cannot_resume_as_v2(self):
        args = self.args('corrupt')
        saved = self.run_arm(args)
        altered = copy.deepcopy(saved)
        altered['schema'] = 'pase_paper_experiments_training_v1'
        with self.assertRaisesRegex(ValueError, 'identity/arm'):
            state.validate_saved(altered, self.identity, 'FULL', args.run_dir/'FULL')
        altered = copy.deepcopy(saved)
        altered['train_order'] = altered['train_order'][::-1]
        with self.assertRaisesRegex(ValueError, 'payload changed'):
            state.validate_saved(altered, self.identity, 'FULL', args.run_dir/'FULL')

    def test_external_full_is_summary_role_not_new_training_checkpoint(self):
        args = self.args('external')
        reference = dict(directory=str(self.root/'artifact'), role_kind='external_promoted_inference',
            source_arm='CONCAT_MLP', completed_training_epochs=100)
        identity = dict(self.identity, external_full_reference=reference)
        role = dict(completed_training_epochs=100, epoch=100)
        metadata = dict(selection_metrics={'mae': 12.4}, native_metrics={}, parameters={})
        with patch.object(train, 'load_promoted', return_value=({}, role, metadata)), \
             patch.object(train, 'reference_identity', return_value=reference), \
             patch.object(train, 'build_model') as build, \
             patch.object(train, 'run_phase') as phase:
            train.train_arm(args, self.data, self.config, identity, 'FULL', self.logger)
            build.assert_not_called()
            phase.assert_not_called()
        self.assertFalse((args.run_dir/'FULL').exists())
        summary = state.read_json(args.run_dir/'summary.json')
        self.assertTrue(summary['arms']['FULL']['no_new_training_updates'])
        self.assertFalse(summary['arms']['FULL']['contributes_to_equal_from_scratch_comparison'])

    def test_external_reference_rejects_new_full_checkpoint(self):
        args = self.args('collision')
        (args.run_dir/'FULL').mkdir()
        identity = dict(self.identity, external_full_reference={'directory': 'artifact'})
        with self.assertRaisesRegex(ValueError, 'must not coexist'):
            train.train_arm(args, self.data, self.config, identity, 'FULL', self.logger)

    def test_default_loader_remains_streaming_and_bounded(self):
        loader = train.LimitedLoader(TinyLoader(), 1)
        self.assertEqual(list(loader.get_iterator()), [0])
        loader.set_order(np.array([3, 2, 1, 0]))
        self.assertEqual(loader.order.tolist(), [3, 2, 1, 0])


if __name__ == '__main__':
    unittest.main()
