"""Server-only transaction tests: tiny Adam task, no released traffic or test split."""
import copy
import json
import logging
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from experiments.pase_paper_experiments import train, state
from experiments.pase_paper_experiments.protocol import CORE_ARMS, configuration


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        for name in ('encoder', 'incident', 'position', 'origin_condition'):
            setattr(self, name, nn.Linear(2, 2))
        self.head = nn.Linear(2, 1)

    def set_stage(self, stage):
        if stage != 'joint':
            raise ValueError('Only joint training exists')


class TinyLoader:
    size, num_batch = 4, 2

    def __init__(self):
        self.order = np.arange(self.size)

    def set_order(self, order):
        self.order = np.asarray(order).copy()

    def get_iterator(self):
        for value in range(self.num_batch):
            yield value


def tiny_accounting(model):
    count = sum(value.numel() for value in model.parameters())
    return {'registered': count, 'trainable': count, 'structurally_active': count}


def tiny_phase(model, loader, data, config, device, optimizer=None):
    if optimizer is not None:
        for _ in loader.get_iterator():
            optimizer.zero_grad(set_to_none=True)
            r = torch.rand(())+float(np.random.rand())+random.random()
            order = float(np.dot(loader.order, np.arange(loader.size)))/100.
            loss = sum(value.square().mean()+(r+order)*value.mean() for value in model.parameters())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['clip_grad_norm'])
            optimizer.step()
    mae = sum(float(value.detach().square().sum()) for value in model.parameters())
    return {'mae': mae, 'signed_bias': 0., 'region_mae': mae}, {'seconds': 0.}


class PaperTrainingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config = configuration('FULL')
        self.config.update(max_epochs=2, lr_milestones=[])
        batches = {key: {'batches': 2, 'batch_size': 2, 'samples': 4}
                   for key in ('train_loader', 'val_loader')}
        self.data = SimpleNamespace(
            loaders={'train_loader': TinyLoader(), 'val_loader': TinyLoader()},
            identity={'loader_batching': batches}, partition={'assignment': [0, 0]},
            position_identity={'fixture': 'position'})
        self.patches = [patch.object(train, 'build_model', side_effect=lambda *args: TinyModel()),
                        patch.object(train, 'parameter_accounting', side_effect=tiny_accounting),
                        patch.object(train, 'source_hashes', return_value={'fixture': 'fixed'}),
                        patch.object(train, 'run_phase', side_effect=tiny_phase)]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)
        self.identity = train.make_identity(self.data, self.config, 2025, True, 2, 2)
        self.logger = logging.getLogger('pase-paper-training-tests')

    def args(self, name):
        path = self.root/name
        path.mkdir(exist_ok=True)
        return SimpleNamespace(run_dir=path, seed=2025, device='cpu')

    def run_arm(self, args, arm='FULL'):
        train.train_arm(args, self.data, self.config, self.identity, arm, self.logger)
        return state.load_state(args.run_dir/arm/'last.pt')

    def test_formal_budget_and_seeds_are_fixed(self):
        formal = configuration('FULL')
        self.assertEqual(formal['max_epochs'], 100)
        state.validate_config(formal)
        for key, value in (('max_epochs', 101), ('learning_rate', .003),
                           ('region_loss_weight', 0.), ('batch_size', 24)):
            altered = dict(formal, **{key: value})
            with self.subTest(key=key), self.assertRaises(ValueError):
                state.validate_config(altered)
        with self.assertRaises(SystemExit):
            train.parse_args(['--run-dir', str(self.root/'r'), '--data-dir', str(self.root/'d'),
                              '--epochs', '101'])
        with self.assertRaises(SystemExit):
            train.parse_args(['--run-dir', str(self.root/'r'), '--data-dir', str(self.root/'d'),
                              '--seed', '99'])

    def test_common_initial_tensors_registered_for_all_nine_arms(self):
        state.validate_identity(self.identity)
        self.assertEqual(self.identity['comparison_arms'], list(CORE_ARMS))
        for module in ('encoder', 'incident', 'position', 'origin_condition'):
            values = {row[module+'_tensor_sha256']
                      for row in self.identity['initializations'].values()}
            self.assertEqual(len(values), 1)
        self.assertFalse(self.identity['historical_trained_weights_reused'])

    def test_resume_preserves_weights_adam_scheduler_all_rng_and_order(self):
        full = self.run_arm(self.args('full'))
        args = self.args('resumed')
        original_summary = train.collect_summary

        def interrupted(run, identity):
            result = original_summary(run, identity)
            if result['arms']['FULL']['completed_epoch'] == 1:
                raise RuntimeError('stop after committed epoch one')
            return result

        with patch.object(train, 'collect_summary', side_effect=interrupted):
            with self.assertRaisesRegex(RuntimeError, 'committed epoch one'):
                self.run_arm(args)
        partial = state.load_state(args.run_dir/'FULL'/'last.pt')
        self.assertEqual(partial['completed_epoch'], 1)
        resumed = self.run_arm(args)
        for key in ('model_state', 'optimizer_state', 'scheduler_state', 'rng_state', 'train_order'):
            with self.subTest(key=key):
                self.assertEqual(state.tensor_tree_hash(full[key]), state.tensor_tree_hash(resumed[key]))
        self.assertEqual(resumed['global_updates'], 4)
        self.assertEqual([row['validation'] for row in full['history']],
                         [row['validation'] for row in resumed['history']])
        self.assertIsNone(resumed['loader_generator_state'])

    def test_kill_during_uncommitted_epoch_replays_from_stage_zero(self):
        full = self.run_arm(self.args('uninterrupted'))
        args = self.args('killed')
        calls = [0]

        def killed_phase(model, loader, data, config, device, optimizer=None):
            if optimizer is None:
                calls[0] += 1
                if calls[0] == 2:
                    raise RuntimeError('kill after optimizer updates before epoch commit')
            return tiny_phase(model, loader, data, config, device, optimizer)

        with patch.object(train, 'run_phase', side_effect=killed_phase):
            with self.assertRaisesRegex(RuntimeError, 'before epoch commit'):
                self.run_arm(args)
        committed = state.load_state(args.run_dir/'FULL'/'last.pt')
        self.assertEqual((committed['completed_epoch'], committed['global_updates']), (0, 0))
        self.assertEqual(committed['optimizer_state']['state'], {})
        resumed = self.run_arm(args)
        for key in ('model_state', 'optimizer_state', 'scheduler_state', 'rng_state', 'train_order'):
            self.assertEqual(state.tensor_tree_hash(full[key]), state.tensor_tree_hash(resumed[key]))

    def test_best_includes_zero_even_if_all_trained_epochs_worsen(self):
        count = [0]

        def worsening(model, loader, data, config, device, optimizer=None):
            metrics, timing = tiny_phase(model, loader, data, config, device, optimizer)
            if optimizer is None:
                metrics['mae'] = float(count[0])
                count[0] += 1
            return metrics, timing

        with patch.object(train, 'run_phase', side_effect=worsening):
            result = self.run_arm(self.args('worse'))
        self.assertEqual(result['best_epoch'], 0)
        self.assertEqual(result['completed_epoch'], 2)

    def test_jsonl_recovered_from_committed_state_without_duplicates(self):
        args = self.args('jsonl')
        saved = self.run_arm(args)
        path = args.run_dir/'FULL'/'history.jsonl'
        path.write_text('interrupted derived file\n', encoding='utf-8')
        self.run_arm(args)
        rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        self.assertEqual(rows, saved['history'])
        self.assertEqual([row['epoch'] for row in rows], [0, 1, 2])

    def test_payload_scheduler_order_and_best_hash_are_checked(self):
        args = self.args('tamper')
        original = self.run_arm(args)
        changed = copy.deepcopy(original)
        changed['optimizer_state']['param_groups'][0]['lr'] = 1.
        with self.assertRaisesRegex(ValueError, 'payload changed'):
            state.validate_saved(changed, self.identity, 'FULL', args.run_dir/'FULL')
        changed = copy.deepcopy(original)
        changed['scheduler_state']['last_epoch'] += 1
        changed['state_payload_sha256'] = state.state_digest(changed)
        with self.assertRaisesRegex(ValueError, 'scheduler'):
            state.validate_saved(changed, self.identity, 'FULL', args.run_dir/'FULL')
        changed = copy.deepcopy(original)
        changed['train_order'] = changed['train_order'][::-1]
        changed['state_payload_sha256'] = state.state_digest(changed)
        with self.assertRaisesRegex(ValueError, 'loader order'):
            state.validate_saved(changed, self.identity, 'FULL', args.run_dir/'FULL')
        (args.run_dir/'FULL'/original['best_checkpoint_file']).write_bytes(b'corrupted')
        with self.assertRaisesRegex(ValueError, 'BEST checkpoint'):
            state.validate_saved(original, self.identity, 'FULL', args.run_dir/'FULL')

    def test_permutations_match_across_arms_and_engineering_summary_is_not_formal(self):
        args = self.args('two_arms')
        first = self.run_arm(args, 'FULL')
        second = self.run_arm(args, 'CONCAT_MLP')
        self.assertEqual([row['train_order_sha256'] for row in first['history']],
                         [row['train_order_sha256'] for row in second['history']])
        summary = state.read_json(args.run_dir/'summary.json')
        self.assertEqual(summary['status'], 'PASE_PAPER_ENGINEERING_PARTIAL')
        self.assertFalse(summary['test_evaluated'])
        self.assertTrue(summary['engineering_only'])

    def test_engineering_loader_is_lazy_and_bounded(self):
        original = TinyLoader()
        limited = train.LimitedLoader(original, 1)
        self.assertEqual(limited.num_batch, 1)
        self.assertEqual(list(limited.get_iterator()), [0])
        limited.set_order([3, 2, 1, 0])
        self.assertEqual(limited.order.tolist(), [3, 2, 1, 0])

    def test_identity_and_output_protection(self):
        changed = copy.deepcopy(self.identity)
        changed['seed'] = 2026
        with self.assertRaisesRegex(ValueError, 'identity'):
            state.validate_identity(changed)
        state.write_json(self.root/'identity.json', {'schema': state.SCHEMA})
        with self.assertRaisesRegex(ValueError, 'intersects another experiment'):
            state.protected_output(self.root/'nested', self.root/'data')
        with self.assertRaisesRegex(ValueError, 'outside the dataset'):
            state.protected_output(self.root/'data'/'run', self.root/'data')

    def test_partial_initialization_fails_before_loading_data(self):
        args = SimpleNamespace(run_dir=self.root, data_dir=self.root/'data', engineering=True,
            engineering_epochs=2, engineering_max_train_batches=2, engineering_max_val_batches=2,
            resume=True, seed=2025, device='cpu', threads=1, arms=['FULL'])
        with patch.object(train, 'parse_args', return_value=args), patch.object(train, 'load_data') as loading:
            with self.assertRaisesRegex(ValueError, 'No committed identity'):
                train.main([])
            loading.assert_not_called()


if __name__ == '__main__':
    unittest.main()
