"""Server-only multi-dataset GW16 transaction and resume tests.

These tests use a tiny synthetic optimizer task; they do not open the released
traffic array, run a research experiment, or claim numerical validation locally.
"""
import copy
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

from experiments.state_exchange_multidata import ARMS
from experiments.state_exchange_multidata import train
from experiments.state_exchange_multidata.common import (
    LOCKED_CONFIG, digest_json, load_config, load_state, protected_output,
    read_json, sha256, tensor_tree_hash, validate_config, write_json)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(2, 2)
        self.incident = nn.Linear(2, 2)
        self.position = nn.Linear(2, 2)
        self.origin_condition = nn.Linear(2, 2)
        self.head = nn.Linear(2, 1)
        self.stage = 'joint'

    def set_stage(self, stage):
        if stage != 'joint':
            raise ValueError('Only joint training exists')
        self.stage = stage


class TinyLoader:
    size, num_batch = 4, 2

    def __init__(self):
        self.order = np.arange(self.size)

    def set_order(self, order):
        self.order = np.asarray(order).copy()


def tiny_builder(data, config, arm, device):
    return TinyModel().to(device)


def tiny_phase(model, loader, data, config, device, optimizer=None):
    """Consume all three host RNGs and exercise persistent Adam moments."""
    if optimizer is not None:
        model.train()
        for batch in range(loader.num_batch):
            optimizer.zero_grad(set_to_none=True)
            r = torch.rand(())+float(np.random.rand())+random.random()
            order_factor = float(np.dot(loader.order, np.arange(loader.size)))/100.
            loss = sum((value.square().mean()+(r+order_factor)*value.mean())
                       for value in model.parameters())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['clip_grad_norm'])
            optimizer.step()
    else:
        model.eval()
    mae = sum(float(value.detach().square().sum()) for value in model.parameters())
    return dict(mae=mae, region_mae=mae, signed_bias=0., batches=loader.num_batch), dict(seconds=0.)


class TrainingTransactionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = copy.deepcopy(LOCKED_CONFIG)
        self.config.update(max_epochs=3, lr_milestones=[])
        self.data = SimpleNamespace(
            loaders={'train_loader':TinyLoader(), 'val_loader':TinyLoader()},
            identity={'published':{'used_counts':{'train_loader':4, 'val_loader':4}},
                      'loader_batching':{'train_loader':{'batches':2, 'batch_size':2, 'samples':4}}},
            partition={'assignment':[0, 0]}, position_identity={'fixture':'positions'})
        self.patches = [patch.object(train, 'build_model', side_effect=tiny_builder),
                        patch.object(train, 'source_hashes', return_value={'fixture.py':'fixed'}),
                        patch.object(train, 'run_phase', side_effect=tiny_phase)]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)
        self.identity = train.make_identity(self.data, self.config, 2025, True)
        self.logger = logging.getLogger('gw16-training-tests')

    def args(self, name, epochs):
        directory = self.root/name
        directory.mkdir(exist_ok=True)
        return SimpleNamespace(run_dir=directory, epochs=epochs, seed=2025, device='cpu')

    def run_arm(self, args, arm='STATE_EXCHANGE'):
        train.train_arm(args, self.data, self.config, self.identity, arm, self.logger)
        return load_state(args.run_dir/arm/'last.pt')

    def resign(self, state):
        state['state_payload_sha256'] = train.state_digest(state)
        return state

    def test_protocol_has_one_joint_budget_and_locked_config(self):
        validate_config(self.config, engineering=True)
        validate_config(copy.deepcopy(LOCKED_CONFIG))
        for key, value in (('learning_rate', .003), ('event_ablation', 'no_report'),
                           ('max_epochs', 101), ('region_loss_weight', .0), ('early_stopping', True)):
            altered = copy.deepcopy(LOCKED_CONFIG)
            altered[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_config(altered)
        self.assertEqual({train.phase_for(epoch, self.config) for epoch in range(4)}, {'joint'})

    def test_external_config_cannot_change_any_protocol_field(self):
        path = self.root/'override.json'
        write_json(path, {'learning_rate':.004})
        with self.assertRaisesRegex(ValueError, 'locked'):
            load_config('Contra_Costa', path)
        write_json(path, {'learning_rate':.002})
        self.assertEqual(load_config('Contra_Costa', path), load_config('Contra_Costa'))

    def test_selected_architecture_is_single_arm_and_reproducibly_initialized(self):
        train.validate_identity(self.identity)
        self.assertEqual(ARMS, ('STATE_EXCHANGE',))
        again = train.make_identity(self.data, self.config, 2025, True)
        self.assertEqual(again['initializations'], self.identity['initializations'])
        for name in ('encoder', 'incident', 'position', 'origin_condition'):
            self.assertIsNotNone(self.identity['initializations']['STATE_EXCHANGE'][name+'_tensor_sha256'])

    def test_stage_zero_is_committed_before_first_optimizer_update(self):
        args = self.args('interrupt_before_training', 3)

        def interrupted(model, loader, data, config, device, optimizer=None):
            if optimizer is not None:
                raise RuntimeError('synthetic pre-update interruption')
            return tiny_phase(model, loader, data, config, device)

        with patch.object(train, 'run_phase', side_effect=interrupted):
            with self.assertRaisesRegex(RuntimeError, 'pre-update'):
                self.run_arm(args)
        state = load_state(args.run_dir/'STATE_EXCHANGE'/'last.pt')
        train.validate_saved(state, self.identity, 'STATE_EXCHANGE',
                             args.run_dir/'STATE_EXCHANGE', 2)
        self.assertEqual((state['completed_epoch'], state['global_updates'], state['best_epoch']), (0, 0, 0))
        self.assertEqual(len(state['history']), 1)
        self.assertIsNone(state['history'][0]['training'])
        self.assertEqual(state['optimizer_state']['state'], {})
        self.assertEqual(state['scheduler_state']['last_epoch'], 0)
        best = load_state(args.run_dir/'STATE_EXCHANGE'/state['best_checkpoint_file'])
        self.assertEqual(best['epoch'], 0)
        self.assertEqual(best['validation'], state['history'][0]['validation'])
        self.assertEqual(self.run_arm(args)['completed_epoch'], 3)

    def test_uninterrupted_equals_resume_in_weights_adam_scheduler_rng_and_order(self):
        uninterrupted = self.run_arm(self.args('full', 3))
        partial_args = self.args('resumed', 1)
        partial = self.run_arm(partial_args)
        self.assertEqual(partial['completed_epoch'], 1)
        partial_args.epochs = 3
        resumed = self.run_arm(partial_args)
        for key in ('model_state', 'optimizer_state', 'scheduler_state', 'rng_state', 'train_order'):
            with self.subTest(field=key):
                self.assertEqual(tensor_tree_hash(uninterrupted[key]), tensor_tree_hash(resumed[key]))
        self.assertEqual(resumed['global_updates'], 6)
        self.assertEqual([row['epoch'] for row in resumed['history']], [0, 1, 2, 3])
        self.assertEqual([row['validation'] for row in uninterrupted['history']],
                         [row['validation'] for row in resumed['history']])

    def test_same_order_across_runs_independent_of_rng_consumption(self):
        first = self.run_arm(self.args('first', 3), 'STATE_EXCHANGE')
        random.seed(44)
        np.random.seed(55)
        torch.manual_seed(66)
        second = self.run_arm(self.args('second', 3), 'STATE_EXCHANGE')
        self.assertEqual([row['train_order_sha256'] for row in first['history']],
                         [row['train_order_sha256'] for row in second['history']])
        self.assertEqual(first['train_order'], second['train_order'])

    def test_interrupted_uncommitted_epoch_replays_from_stage_zero(self):
        full = self.run_arm(self.args('full_after_interruption', 3))
        args = self.args('interrupted_after_updates', 3)
        validation_calls = [0]

        def interrupted(model, loader, data, config, device, optimizer=None):
            if optimizer is None:
                validation_calls[0] += 1
                if validation_calls[0] == 2:
                    raise RuntimeError('synthetic interruption after optimizer updates')
            return tiny_phase(model, loader, data, config, device, optimizer)

        with patch.object(train, 'run_phase', side_effect=interrupted):
            with self.assertRaisesRegex(RuntimeError, 'after optimizer updates'):
                self.run_arm(args)
        self.assertEqual(load_state(args.run_dir/'STATE_EXCHANGE'/'last.pt')['completed_epoch'], 0)
        replay = self.run_arm(args)
        for key in ('model_state', 'optimizer_state', 'scheduler_state', 'rng_state', 'train_order'):
            with self.subTest(field=key):
                self.assertEqual(tensor_tree_hash(full[key]), tensor_tree_hash(replay[key]))

    def test_stage_zero_can_remain_best_after_worse_trained_epochs(self):
        args = self.args('stage_zero_best', 3)
        evaluation_calls = [0]

        def worsening(model, loader, data, config, device, optimizer=None):
            result, timing = tiny_phase(model, loader, data, config, device, optimizer)
            if optimizer is None:
                result['mae'] = float(evaluation_calls[0])
                evaluation_calls[0] += 1
            return result, timing

        with patch.object(train, 'run_phase', side_effect=worsening):
            state = self.run_arm(args)
        self.assertEqual(state['best_epoch'], 0)
        self.assertEqual(state['completed_epoch'], 3)
        self.assertEqual(state['best_metrics']['mae'], 0.)

    def test_payload_corruption_is_detected(self):
        args = self.args('corruption', 1)
        state = self.run_arm(args)
        state['optimizer_state']['param_groups'][0]['lr'] = .05
        with self.assertRaisesRegex(ValueError, 'payload changed'):
            train.validate_saved(state, self.identity, 'STATE_EXCHANGE', args.run_dir/'STATE_EXCHANGE', 2)

    def test_scheduler_and_order_semantics_checked_beyond_payload_hash(self):
        args = self.args('semantics', 1)
        original = self.run_arm(args)
        scheduler = copy.deepcopy(original)
        scheduler['scheduler_state']['last_epoch'] += 1
        with self.assertRaisesRegex(ValueError, 'scheduler'):
            train.validate_saved(self.resign(scheduler), self.identity, 'STATE_EXCHANGE',
                                 args.run_dir/'STATE_EXCHANGE', 2)
        order = copy.deepcopy(original)
        order['train_order'] = order['train_order'][::-1]
        with self.assertRaisesRegex(ValueError, 'sample order'):
            train.validate_saved(self.resign(order), self.identity, 'STATE_EXCHANGE',
                                 args.run_dir/'STATE_EXCHANGE', 2)

    def test_best_file_corruption_is_detected(self):
        args = self.args('best_corruption', 1)
        state = self.run_arm(args)
        (args.run_dir/'STATE_EXCHANGE'/state['best_checkpoint_file']).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'BEST checkpoint'):
            train.validate_saved(state, self.identity, 'STATE_EXCHANGE', args.run_dir/'STATE_EXCHANGE', 2)

    def test_identity_tampering_is_detected(self):
        identity = copy.deepcopy(self.identity)
        identity['seed'] += 1
        with self.assertRaisesRegex(ValueError, 'identity'):
            train.validate_identity(identity)

    def test_schedule_does_not_restart_at_resume_milestone(self):
        model = TinyModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=.002, eps=1e-8, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 70, 90], gamma=.5)
        used = []
        for epoch in range(1, 42):
            used.append(optimizer.param_groups[0]['lr'])
            optimizer.zero_grad(set_to_none=True)
            sum(value.sum() for value in model.parameters()).backward()
            optimizer.step()
            scheduler.step()
            if epoch == 40:
                saved_optimizer = copy.deepcopy(optimizer.state_dict())
                saved_scheduler = copy.deepcopy(scheduler.state_dict())
                optimizer = torch.optim.Adam(model.parameters(), lr=.002)
                scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 70, 90], gamma=.5)
                optimizer.load_state_dict(saved_optimizer)
                scheduler.load_state_dict(saved_scheduler)
        self.assertEqual(used[39], .002)
        self.assertEqual(used[40], .001)
        self.assertEqual(scheduler.last_epoch, 41)

    def test_runtime_drift_is_rejected(self):
        previous = dict(torch='recorded', numpy='recorded', threads=3, dtype='float32',
            cuda=None, device_name='CPU', python_device='cpu', deterministic_algorithms=True,
            tf32=False, cudnn_benchmark=False, python='recorded', seed=2025,
            scipy='recorded', cudnn=None, cublas_workspace_config=':4096:8', default_dtype='torch.float32')
        train.validate_runtime(previous, copy.deepcopy(previous))
        current = dict(previous, threads=4)
        with self.assertRaisesRegex(ValueError, 'runtime'):
            train.validate_runtime(previous, current)

    def test_archived_protocol_position_and_metric_contract_are_authenticated(self):
        expected = {'config.json':self.config, 'partition.json':self.data.partition,
            'position.json':self.data.position_identity, 'metric_contract.json':self.identity['metric_contract']}
        for name, value in expected.items():
            write_json(self.root/name, value)
        train.validate_artifacts(self.root, self.identity, self.data)
        write_json(self.root/'position.json', {'fixture':'changed'})
        with self.assertRaisesRegex(ValueError, 'position.json'):
            train.validate_artifacts(self.root, self.identity, self.data)

    def test_resume_without_identity_fails_before_loading_data(self):
        args = SimpleNamespace(run_dir=self.root, data_dir=self.root/'data', config=None,
            engineering=True, epochs=1, resume=True, device='cpu', threads=1,
            seed=2025, arms=list(ARMS), dataset='Contra_Costa')
        with patch.object(train, 'parse_args', return_value=args), patch.object(train, 'load_data') as loading:
            with self.assertRaisesRegex(ValueError, 'initialization was interrupted'):
                train.main([])
            loading.assert_not_called()

    def test_output_cannot_nest_under_any_run_identity(self):
        write_json(self.root/'identity.json', {'schema':self.identity['schema']})
        with self.assertRaisesRegex(ValueError, 'intersects another experiment'):
            protected_output(self.root/'nested', self.root/'dataset')

    def test_evaluation_child_is_allowed_only_for_gw16_parent(self):
        output = self.root/'eval_best_val'
        write_json(self.root/'identity.json', {'schema':self.identity['schema']})
        self.assertEqual(protected_output(output, self.root/'dataset', allow_run_children=True), output.resolve())
        write_json(self.root/'identity.json', {'schema':'legacy_experiment'})
        with self.assertRaisesRegex(ValueError, 'intersects another experiment'):
            protected_output(output, self.root/'dataset', allow_run_children=True)

    def test_summary_distinguishes_engineering_from_confirmation(self):
        args = self.args('summary', 1)
        self.run_arm(args)
        summary = read_json(args.run_dir/'summary.json')
        self.assertTrue(summary['engineering_only'])
        self.assertFalse(summary['test_evaluated'])
        self.assertFalse(summary['independent_confirmation'])
        self.assertEqual(summary['status'], 'STATE_EXCHANGE_MULTIDATA_PARTIAL')
        self.assertEqual(summary['best_selection'], 'validation_pooled_mae_includes_stage_zero')


if __name__ == '__main__':
    unittest.main()
