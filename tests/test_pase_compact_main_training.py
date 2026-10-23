"""CPU/CI tests of FULL-only scratch training, immutable CC and exact resume."""
import copy
import logging
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.pase_compact_main import state, train, evaluate, model
from experiments.pase_compact_main.protocol import configuration
from experiments.pase_compact_v3 import model as old_model, state as old_state
from experiments.state_exchange_multidata.common import load_config
from tests.test_pase_paper_experiments_training import (
    TinyLoader, TinyModel, tiny_accounting, tiny_phase,
)


class MainTrainingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config = configuration(dataset='Alameda')
        self.config.update(max_epochs=3, lr_milestones=[])
        batches = {key: dict(batches=2, batch_size=2, samples=4)
                   for key in ('train_loader', 'val_loader')}
        self.data = SimpleNamespace(loaders=dict(train_loader=TinyLoader(), val_loader=TinyLoader()),
            identity={'loader_batching': batches}, partition={'assignment': [0, 0]},
            position_identity={'fixture': 'position'})
        for item in (patch.object(train, 'build_model', side_effect=lambda *a: TinyModel()),
                     patch.object(train, 'parameter_accounting', side_effect=tiny_accounting),
                     patch.object(train, 'source_hashes', return_value={'fixture': 'fixed'}),
                     patch.object(train, 'run_phase', side_effect=tiny_phase)):
            item.start()
            self.addCleanup(item.stop)
        self.identity = train.make_identity(self.data, self.config, 2025, True, 2, 2)

    def run_full(self, name):
        directory = self.root/name
        directory.mkdir(exist_ok=True)
        args = SimpleNamespace(run_dir=directory, seed=2025, device='cpu')
        train.train_arm(args, self.data, self.config, self.identity, 'FULL', logging.getLogger('test'))
        return state.load_state(directory/'FULL'/'last.pt')

    def test_model_and_per_dataset_recipe_are_unchanged(self):
        self.assertIs(model.build_model, old_model.build_model)
        for dataset, nodes in (('Alameda', 521), ('Orange', 990)):
            config = configuration(dataset=dataset)
            self.assertEqual(config, load_config(dataset))
            self.assertEqual(config['nodes'], nodes)
            state.validate_config(config)
        with self.assertRaises(ValueError):
            configuration(dataset='Contra_Costa')

    def test_formal_cli_only_scratch_full_new_datasets_and_fixed_budget(self):
        base = ['--run-dir', str(self.root/'run'), '--data-dir', str(self.root/'data')]
        for dataset in ('Alameda', 'Orange'):
            args = train.parse_args(base+['--dataset', dataset])
            self.assertEqual(args.arms, ['FULL'])
            self.assertIsNone(args.full_reference)
        with self.assertRaises(SystemExit):
            train.parse_args(base+['--dataset', 'Contra_Costa'])
        with self.assertRaises(ValueError):
            train.parse_args(base+['--dataset', 'Orange', '--arms', 'UP_ONLY'])
        with self.assertRaises(SystemExit):
            train.parse_args(base+['--dataset', 'Orange', '--epochs', '150'])

    def test_resume_restores_model_optimizer_scheduler_rng_and_order(self):
        whole = self.run_full('whole')
        original = train.collect_summary
        def interrupt(run, identity):
            result = original(run, identity)
            if result['arms']['FULL']['completed_epoch'] == 1:
                raise RuntimeError('committed-one')
            return result
        with patch.object(train, 'collect_summary', side_effect=interrupt):
            with self.assertRaisesRegex(RuntimeError, 'committed-one'):
                self.run_full('resumed')
        resumed = self.run_full('resumed')
        for key in ('model_state', 'optimizer_state', 'scheduler_state', 'rng_state', 'train_order'):
            self.assertEqual(state.tensor_tree_hash(whole[key]), state.tensor_tree_hash(resumed[key]))
        with patch.object(train, 'run_phase', side_effect=AssertionError('completed training rerun')):
            self.run_full('resumed')

    def test_old_identities_remain_separate_and_source_hashes_do_not_change(self):
        state.validate_identity(self.identity)
        with self.assertRaises(ValueError):
            old_state.validate_identity(self.identity)
        with self.assertRaises(ValueError):
            state.validate_identity(dict(self.identity, schema=old_state.SCHEMA))
        self.assertFalse(any('/pase_compact_main/' in key for key in old_state.source_hashes()))
        self.assertTrue(any('/pase_compact_v3/' in key for key in state.source_hashes()))

    def test_test_guard_requires_completed_validation_best(self):
        identity = dict(engineering_only=False, config={'max_epochs':100}, comparison_arms=['FULL'])
        role = dict(completed_training_epochs=100, full_protocol_reached=True,
                    selection='best', selected_on='validation_pooled_mae', best_includes_stage_zero=True)
        evaluate.test_guard(identity, {'FULL':role}, ['FULL'])
        for change in ({'completed_training_epochs':99}, {'selected_on':'test_mae'}, {'selection':'last'}):
            with self.assertRaises(ValueError):
                evaluate.test_guard(identity, {'FULL':dict(role, **change)}, ['FULL'])
        with self.assertRaises(ValueError):
            evaluate.test_guard(dict(identity, engineering_only=True), {'FULL':role}, ['FULL'], True)

    def test_selected_best_loads_committed_history_and_rejects_tamper(self):
        self.run_full('selection')
        directory = self.root/'selection'
        weights, role, expected = evaluate.selected_best(directory, self.identity, 'FULL')
        self.assertEqual(role['completed_training_epochs'], 3)
        self.assertEqual(state.tensor_tree_hash(weights), role['model_tensor_sha256'])
        last_path = directory/'FULL'/'last.pt'
        saved = state.load_state(last_path)
        saved['best_epoch'] += 1
        state.atomic_save(last_path, saved)
        with self.assertRaises(ValueError):
            evaluate.selected_best(directory, self.identity, 'FULL')


if __name__ == '__main__':
    unittest.main()
