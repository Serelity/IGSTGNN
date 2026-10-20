"""Server-side tests for original-model adaptation and resumable transactions."""
from copy import deepcopy
from collections import Counter
import logging
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.chronological.original_igstgnn_runtime import tensor_tree_hash
from experiments.state_exchange_multidata import baseline as baseline


class FakeEngine:
    def __init__(self):
        self.model = torch.nn.Linear(1, 1)
        self._optimizer = torch.optim.Adam(self.model.parameters(), lr=.002, eps=1e-8, weight_decay=1e-5)
        self._lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self._optimizer,
            milestones=baseline.MILESTONES, gamma=.5)
        self._iter_cnt = self._cl_len = 0
        self._patience = 20
        self._device = 'cpu'
        self._logger = logging.getLogger('fake-original-test')
        self._dataloader = {'train_loader': SimpleNamespace(samples=[0, 1, 2, 3], num_batch=2)}
        self.evaluated = []

    def train_batch(self):
        self.model.train()
        self._dataloader['train_loader'].shuffle()
        for _ in range(2):
            self._optimizer.zero_grad()
            value = torch.rand(1, 1)+np.random.uniform()+random.random()
            self.model(value).square().mean().backward()
            self._optimizer.step()
            self._iter_cnt += 1
            self._cl_len = 12
        return 1., .1, 2.

    def evaluate(self, split):
        self.evaluated.append(split)
        return [3., 2., 4.][self._iter_cnt//2-1], .2, 4.


def identity():
    return dict(schema=baseline.SCHEMA, identity_sha256='synthetic', registered_parameters=2,
        engineering_only=True, source_fingerprint='frozen', training_config=baseline.original_config('Alameda'),
        data={'splits': {'train': {'count': 4, 'partition': [2, 2]}}})


def seed():
    random.seed(2025)
    np.random.seed(2025)
    torch.manual_seed(2025)


class BaselineTests(unittest.TestCase):
    def test_original_recipe_is_shared_and_locked(self):
        for dataset in ('Alameda', 'Orange', 'Contra_Costa'):
            config = baseline.original_config(dataset)
            self.assertEqual(config['dataset'], dataset)
            self.assertEqual(config['bs'], 48)
            self.assertEqual(config['max_epochs'], 100)
            self.assertEqual(config['patience'], 20)
            self.assertEqual(config['warm_epoch'], 30)
            self.assertEqual(config['cl_epoch'], 3)
            self.assertEqual(config['icsf_dim'], 64)
        self.assertEqual(baseline.MILESTONES, [1, 38, 46, 54, 62, 70, 80])

    def test_engineering_batch_two_is_explicit(self):
        self.assertEqual(baseline.original_config('Orange', check=True)['bs'], 2)

    def test_checkpoint_contains_plain_milestones_and_reads_with_restricted_torch_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            seed()
            baseline.run_epochs(FakeEngine(), tmp, identity(), 1)
            with patch.object(torch.serialization, 'add_safe_globals', create=True,
                              side_effect=AssertionError('No global allowlist changes')):
                saved = torch.load(Path(tmp)/'last.pt', map_location='cpu', weights_only=True)
            self.assertIs(type(saved['scheduler_state']['milestones']), dict)
            self.assertEqual(saved['scheduler_state']['milestones'], dict(Counter(baseline.MILESTONES)))
            baseline.validate_saved(saved, identity(), tmp, 2)

    def test_restored_scheduler_rebuilds_counter_and_preserves_later_milestones(self):
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=.002)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
            milestones=baseline.MILESTONES, gamma=.5)
        for _ in range(39):
            optimizer.zero_grad()
            model(torch.ones(1, 1)).sum().backward()
            optimizer.step()
            scheduler.step()
        saved_optimizer = deepcopy(optimizer.state_dict())
        saved_scheduler = baseline.scheduler_snapshot(scheduler)
        resumed_optimizer = torch.optim.Adam(model.parameters(), lr=.002)
        resumed_scheduler = torch.optim.lr_scheduler.MultiStepLR(resumed_optimizer,
            milestones=baseline.MILESTONES, gamma=.5)
        resumed_optimizer.load_state_dict(saved_optimizer)
        baseline.restore_scheduler(resumed_scheduler, saved_scheduler)
        self.assertIsInstance(resumed_scheduler.milestones, Counter)
        # MultiStepLR's closed-form path also requires Counter.elements().
        self.assertEqual(resumed_scheduler._get_closed_form_lr(), scheduler._get_closed_form_lr())
        self.assertEqual(baseline.scheduler_snapshot(resumed_scheduler), saved_scheduler)
        for _ in range(39, 82):
            for active_optimizer, active_scheduler in (
                (optimizer, scheduler), (resumed_optimizer, resumed_scheduler)):
                active_optimizer.zero_grad()
                model(torch.ones(1, 1)).sum().backward()
                active_optimizer.step()
                active_scheduler.step()
            self.assertEqual(resumed_optimizer.param_groups[0]['lr'], optimizer.param_groups[0]['lr'])
        self.assertEqual(baseline.scheduler_snapshot(resumed_scheduler), baseline.scheduler_snapshot(scheduler))

    def test_modified_milestones_are_rejected_even_with_resigned_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            saved = self._saved(tmp)
            saved['scheduler_state']['milestones'][38] = 2
            saved['state_payload_sha256'] = baseline.state_digest(saved)
            with self.assertRaisesRegex(ValueError, 'milestones'):
                baseline.validate_saved(saved, identity(), tmp, 2)

    def test_unknown_dataset_rejected(self):
        with self.assertRaises(ValueError):
            baseline.original_config('not-a-release')

    def test_cli_budget_and_seed_are_total_and_fixed(self):
        base = ['--dataset', 'Alameda', '--data-dir', 'data', '--run-dir', 'run']
        self.assertEqual(baseline.parse_args(base+['--epochs', '20']).epochs, 20)
        for suffix in (['--epochs', '101'], ['--check', '--epochs', '4'], ['--seed', '8']):
            with self.assertRaises(SystemExit):
                baseline.parse_args(base+suffix)

    def test_generic_constructor_forwards_registry_dimensions(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            np.save(directory/'adj_matrix.npy', np.eye(3, dtype=np.float32))
            seen = {}
            class Model:
                def __init__(self, **kwargs):
                    seen.update(kwargs)
                def to(self, device):
                    return self
            modules = SimpleNamespace(graph=SimpleNamespace(normalize_adj_mx=lambda value, _: [value]),
                                      model=SimpleNamespace(IGSTGNN=Model))
            data = SimpleNamespace(directory=directory, config=dict(node_num=3, dataset='Alameda'))
            baseline.build_original_model(data, modules, 'cpu')
            self.assertEqual(seen['node_num'], 3)
            self.assertEqual(seen['dataset'], 'Alameda')
            self.assertEqual(seen['model_args']['node_num'], 3)
            self.assertEqual(seen['seq_len'], 12)
            self.assertTrue(seen['use_sensor_info'])

    def test_test_loader_requires_explicit_request(self):
        seen = []
        sample = {'x_data': np.zeros((12, 3, 3), dtype=np.float32),
                  'y_data': np.zeros((12, 3, 3), dtype=np.float32)}
        def loader(samples, incident, batch, logger, **kwargs):
            seen.append((samples, incident, batch, kwargs))
            return 'explicit-test-loader'
        modules = SimpleNamespace(loader=SimpleNamespace(IncidentDataLoader=loader))
        data = SimpleNamespace(test_samples=(sample,), config={'node_num': 3}, sensors={})
        self.assertEqual(baseline.original_loader(data, modules, 'test', 48, None), 'explicit-test-loader')
        self.assertEqual(len(seen), 1)
        self.assertIs(seen[0][0], data.test_samples)

    def test_loader_rejects_unrecognized_split(self):
        with self.assertRaisesRegex(ValueError, 'split'):
            baseline.original_loader(None, None, 'unknown', 48, None)

    def _data_fixture(self, directory, training):
        (directory/'sensors.csv').write_text('station_id,Fwy,Direction\n1,1,N\n2,1,N\n3,1,N\n',
                                             encoding='utf-8')
        sample = {'x_data': np.zeros((12, 3, 3), dtype=np.float32),
                  'y_data': np.zeros((12, 3, 3), dtype=np.float32)}
        class Poison:
            def __getitem__(self, key):
                raise AssertionError('test sample was inspected')
        class Stats:
            def __enter__(self):
                return dict(mean=10., std=2., normalized=True)
            def __exit__(self, *args):
                pass
        modules = SimpleNamespace(loader=SimpleNamespace(
            _load_sensor_info=lambda *args: {}, StandardScaler=lambda mean, std: (mean, std),
            IncidentDataLoader=lambda samples, *args, **kwargs: SimpleNamespace(samples=samples)))
        registered = dict(nodes=3, total_samples=8, train_samples=4, val_samples=2, test_samples=2,
                          hashes={'incident_all.npy': 'release'})
        with patch.object(baseline, 'spec', return_value=registered), \
                patch.object(baseline, 'sha256', return_value='release'), \
                patch.object(baseline.np, 'load', return_value=Stats()), \
                patch.object(baseline, 'load_array', return_value=[sample]*6+[Poison(), Poison()]):
            return baseline.load_original_data(directory, 'Alameda', baseline.original_config('Alameda'),
                                                modules, None, training=training), modules

    def test_validation_loading_holds_but_never_inspects_test_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            data, modules = self._data_fixture(Path(tmp), False)
            self.assertEqual(list(data.loaders), ['val_loader'])
            self.assertEqual(data.identity['splits']['val']['selected_indices'], [4, 5])
            self.assertEqual(len(data.test_samples), 2)
            self.assertFalse(data.identity['test_loader_constructed'])
            with self.assertRaisesRegex(AssertionError, 'test sample'):
                baseline.original_loader(data, modules, 'test', 48, None)

    def test_training_loading_builds_only_train_and_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            data, _ = self._data_fixture(Path(tmp), True)
            self.assertEqual(set(data.loaders), {'train_loader', 'val_loader'})
            self.assertEqual(data.identity['splits']['train']['selected_indices'], [0, 1, 2, 3])
            self.assertEqual(data.identity['normalization'], dict(mean=10., std=2., refitted=False))

    def test_split_shape_and_dtype_are_authenticated(self):
        sample = {'x_data': np.zeros((12, 3, 3), dtype=np.float32),
                  'y_data': np.zeros((12, 3, 3), dtype=np.float32)}
        baseline._validate_samples([sample], 3, 'val')
        for wrong in (np.zeros((12, 4, 3), dtype=np.float32), np.zeros((12, 3, 3), dtype=np.float64)):
            with self.assertRaisesRegex(ValueError, 'published'):
                baseline._validate_samples([dict(sample, y_data=wrong)], 3, 'val')

    def test_full_budget_or_original_patience_marks_complete(self):
        ident = identity()
        ident['engineering_only'] = False
        self.assertFalse(baseline._complete({'completed_epoch': 40, 'wait': 19}, ident))
        self.assertTrue(baseline._complete({'completed_epoch': 40, 'wait': 20}, ident))
        self.assertTrue(baseline._complete({'completed_epoch': 100, 'wait': 0}, ident))

    def test_engineering_budget_is_three_but_original_formal_recipe_stays_100(self):
        ident = identity()
        self.assertEqual(ident['training_config']['max_epochs'], 100)
        self.assertFalse(baseline._complete({'completed_epoch': 2, 'wait': 0}, ident))
        self.assertTrue(baseline._complete({'completed_epoch': 3, 'wait': 0}, ident))

    def test_pre_update_boundary_is_resumable_and_does_not_select_stage_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            seed()
            engine = FakeEngine()
            baseline.run_epochs(engine, tmp, identity(), 0)
            saved = torch.load(Path(tmp)/'last.pt', map_location='cpu', weights_only=True)
            baseline.validate_saved(saved, identity(), tmp, 2)
            self.assertEqual(saved['completed_epoch'], 0)
            self.assertEqual(saved['history'], [])
            self.assertIsNone(saved['best_epoch'])
            self.assertIsNone(saved['best_checkpoint_file'])
            self.assertEqual(saved['optimizer_state']['state'], {})
            self.assertEqual(engine.evaluated, [])
            with patch.object(baseline, '_read_identity', return_value=identity()):
                with self.assertRaisesRegex(ValueError, 'before the first trained epoch'):
                    baseline.selected_checkpoint(tmp)

    def test_first_epoch_interruption_replays_initial_rng_weights_and_sample_order(self):
        with tempfile.TemporaryDirectory() as full_dir, tempfile.TemporaryDirectory() as resumed_dir:
            seed()
            full = baseline.run_epochs(FakeEngine(), full_dir, identity(), 3)
            seed()
            interrupted = FakeEngine()
            with patch.object(interrupted, 'evaluate', side_effect=RuntimeError('interrupted first epoch')):
                with self.assertRaisesRegex(RuntimeError, 'first epoch'):
                    baseline.run_epochs(interrupted, resumed_dir, identity(), 3)
            initial = torch.load(Path(resumed_dir)/'last.pt', map_location='cpu', weights_only=True)
            self.assertEqual(initial['completed_epoch'], 0)
            baseline.validate_saved(initial, identity(), resumed_dir, 2)
            replay = baseline.run_epochs(FakeEngine(), resumed_dir, identity(), 3, initial)
            for key in ('model_state', 'optimizer_state', 'scheduler_state', 'rng_state', 'train_order'):
                self.assertEqual(tensor_tree_hash(full[key]), tensor_tree_hash(replay[key]))

    def test_continuation_matches_uninterrupted_optimizer_rng_and_order(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            seed()
            full_engine = FakeEngine()
            full = baseline.run_epochs(full_engine, first, identity(), 3)
            seed()
            initial_engine = FakeEngine()
            baseline.run_epochs(initial_engine, second, identity(), 1)
            saved = torch.load(Path(second)/'last.pt', map_location='cpu', weights_only=True)
            baseline.validate_saved(saved, identity(), second, 2)
            resumed_engine = FakeEngine()
            resumed = baseline.run_epochs(resumed_engine, second, identity(), 3, saved)
            for key in ('model_state', 'optimizer_state', 'scheduler_state', 'rng_state', 'train_order'):
                self.assertEqual(tensor_tree_hash(full[key]), tensor_tree_hash(resumed[key]))
            self.assertEqual([row['train_order_sha256'] for row in full['history']],
                             [row['train_order_sha256'] for row in resumed['history']])
            self.assertEqual(full_engine.evaluated, ['val', 'val', 'val'])
            self.assertEqual(resumed_engine.evaluated, ['val', 'val'])

    def _saved(self, directory):
        seed()
        baseline.run_epochs(FakeEngine(), directory, identity(), 3)
        return torch.load(Path(directory)/'last.pt', map_location='cpu', weights_only=True)

    def test_saved_best_tracks_validation_only_and_excludes_stage_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            saved = self._saved(tmp)
            baseline.validate_saved(saved, identity(), tmp, 2)
            summary = baseline.summary_for(saved, identity())
            self.assertEqual(summary['best_epoch'], 2)
            self.assertEqual(summary['best_validation_mae'], 2.)
            self.assertEqual(saved['wait'], 1)
            self.assertFalse(summary['best_selection_includes_stage_zero'])
            self.assertFalse(summary['test_evaluated'])

    def test_resume_rejects_contradictory_order_selection_curriculum_and_lr(self):
        with tempfile.TemporaryDirectory() as tmp:
            saved = self._saved(tmp)
            changes = [dict(train_order=[0, 0, 2, 3]), dict(best_epoch=1), dict(curriculum_length=1),
                       dict(global_updates=5), dict(registered_parameters=455220)]
            for change in changes:
                modified = dict(saved, **change)
                modified['state_payload_sha256'] = baseline.state_digest(modified)
                with self.assertRaises(ValueError):
                    baseline.validate_saved(modified, identity(), tmp, 2)
            changed = deepcopy(saved)
            changed['optimizer_state']['param_groups'][0]['lr'] = .123
            changed['state_payload_sha256'] = baseline.state_digest(changed)
            with self.assertRaisesRegex(ValueError, 'learning rate'):
                baseline.validate_saved(changed, identity(), tmp, 2)

    def test_model_optimizer_and_rng_tampering_are_detected_by_payload_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            saved = self._saved(tmp)
            model = deepcopy(saved)
            model['model_state']['weight'][0, 0] += 1
            optimizer = deepcopy(saved)
            next(iter(optimizer['optimizer_state']['state'].values()))['exp_avg'].add_(1)
            rng = deepcopy(saved)
            rng['rng_state']['torch'][0] ^= 1
            for changed in (model, optimizer, rng):
                with self.assertRaisesRegex(ValueError, 'payload changed'):
                    baseline.validate_saved(changed, identity(), tmp, 2)

    def test_tampered_best_file_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            saved = self._saved(tmp)
            (Path(tmp)/saved['best_checkpoint_file']).write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'fingerprint'):
                baseline.validate_saved(saved, identity(), tmp, 2)

    def test_selected_checkpoint_returns_original_role_and_expected_native_mae(self):
        with tempfile.TemporaryDirectory() as tmp:
            saved = self._saved(tmp)
            with patch.object(baseline, '_read_identity', return_value=identity()):
                weights, role, expected, checkpoint = baseline.selected_checkpoint(tmp)
            self.assertEqual(role['case'], 'ORIGINAL_IGSTGNN')
            self.assertEqual(role['selected_epoch'], 2)
            self.assertTrue(role['training_complete'])
            self.assertEqual(role['training_budget_epochs'], 3)
            self.assertTrue(role['engineering_only'])
            self.assertEqual(expected, 2.)
            self.assertEqual(checkpoint.name, saved['best_checkpoint_file'])
            self.assertEqual(tensor_tree_hash(weights), role['model_state_sha256'])


if __name__ == '__main__':
    unittest.main()
