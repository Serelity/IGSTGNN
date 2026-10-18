"""Run on the server: GW6 budget, joint/probe and exact-resume contracts.

These are deliberately tiny synthetic optimizations. They validate the new
runner, not the predictive value or resource fit of the real traffic model.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

from experiments.graph_memory_gw6 import train
from tests.test_original_isp_runner import Engine as PreviousEngine, comparable_history


class ToyMemory(nn.Module):
    """Small stochastic anchor with the state namespace used by GW6."""
    def __init__(self, stage='joint', arm='JOINT_MEMORY'):
        super().__init__()
        self.arm, self.stage = arm, stage
        self.anchor = nn.Sequential(nn.Linear(1, 3), nn.Dropout(.2), nn.Linear(3, 1))
        self.readout = nn.Linear(1, 1, bias=False)
        if stage == 'probe':
            self.anchor.requires_grad_(False)
            self.anchor.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.stage == 'probe':
            self.anchor.eval()
        return self

    def forward(self, x):
        return self.anchor(x)+self.readout(x)

    def anchor_state(self):
        return self.anchor.state_dict()

    def param_num(self):
        return sum(p.numel() for p in self.parameters())


class Engine(PreviousEngine):
    def __init__(self, seed=2025, stage='joint'):
        super().__init__(seed)
        self.model = ToyMemory(stage)
        active = [p for p in self.model.parameters() if p.requires_grad]
        self._optimizer = torch.optim.Adam(active, lr=.01)
        self._lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self._optimizer, milestones=[1, 3], gamma=.5)
        self._warm_step, self._cl_step = 202, 6
        train.base.configure_runtime('cpu', 1, seed)


def identity_for(engine):
    arms = [arm for arm in train.DEFAULT_ARMS
            if engine.model.stage != 'probe' or arm != 'RECURRENT_CONTROL']
    identity = dict(schema=train.SCHEMA, design=train.DESIGN, stage=engine.model.stage,
        arms=arms, engineering_only=True,
        initializations={arm:train.initialization(engine.model) for arm in arms},
        training_config=dict(warm_epoch=101, cl_epoch=3),
        data=dict(splits=dict(train=dict(count=4))), source_reference={},
        model_options=dict(hidden=32, node_chunk=32), curriculum='full')
    identity['identity_sha256'] = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return identity


def state_for(arm, errors):
    history = [dict(epoch=index, validation_mae=value, validation_rmse=value+1,
        validation_mape_fraction=value*.01) for index,value in enumerate(errors, 1)]
    best = min(history, key=lambda row:row['validation_mae'])
    return dict(arm=arm, history=history, completed_epoch=len(history),
        best_epoch=best['epoch'], best_validation_mae=best['validation_mae'],
        best_checkpoint_file='fixture.pt', best_checkpoint_sha256='fixture',
        model_tensor_sha256='fixture', global_updates=2*len(history), initialization={})


class GraphMemoryWorkflowTests(unittest.TestCase):
    def test_parser_defaults_and_rejects_mismatched_budgets_or_arms(self):
        argv = ['--gw5-run', 'old', '--run-dir', 'new']
        args = train.parse_args(argv)
        self.assertEqual((args.stage,args.curriculum,args.epochs), ('joint','full',5))
        self.assertEqual(args.arms, list(train.DEFAULT_ARMS))
        invalid = (['--epochs','0'], ['--epochs','101'], ['--hidden','0'],
            ['--node-chunk','0'], ['--threads','0'],
            ['--arms','JOINT_MEMORY','JOINT_MEMORY'],
            ['--only','JOINT_MEMORY','JOINT_MEMORY'],
            ['--arms','JOINT_MEMORY','--only','RECURRENT_CONTROL'], ['--stage','probe'])
        for extra in invalid:
            with self.subTest(extra=extra), self.assertRaises(SystemExit):
                train.parse_args(argv+extra)
        probe = train.parse_args(argv+['--stage','probe','--arms','JOINT_MEMORY'])
        self.assertEqual(probe.arms, ['JOINT_MEMORY'])

    def test_exact_epoch_resume_preserves_optimizer_rng_order_and_full_curriculum(self):
        with tempfile.TemporaryDirectory() as directory:
            whole, split = Path(directory)/'whole', Path(directory)/'split'
            whole.mkdir(); split.mkdir()
            identity = identity_for(Engine())
            with patch('builtins.print'):
                expected = train.base.run_epochs(Engine(), whole, identity, 'JOINT_MEMORY', 4)
                train.base.run_epochs(Engine(), split, identity, 'JOINT_MEMORY', 2)
                saved = train.base.load_checkpoint(split/'last.pt')
                train.validate_saved(saved, identity, 'JOINT_MEMORY', split, 2)
                resumed = train.base.run_epochs(Engine(19), split, identity, 'JOINT_MEMORY', 4, saved)
                train.validate_saved(resumed, identity, 'JOINT_MEMORY', split, 2)
            self.assertEqual(comparable_history(expected), comparable_history(resumed))
            for key in ('model_state','optimizer_state','scheduler_state','rng_state','train_order'):
                self.assertEqual(train.base.tensor_tree_hash(expected[key]),
                    train.base.tensor_tree_hash(resumed[key]), key)
            self.assertTrue(all(row['curriculum_length'] == 12 for row in resumed['history']))
            self.assertEqual(resumed['global_updates'], 8)

    def test_first_epoch_interruption_replays_from_saved_epoch_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            whole, killed = Path(directory)/'whole', Path(directory)/'killed'
            whole.mkdir(); killed.mkdir()
            identity = identity_for(Engine())
            with patch('builtins.print'):
                expected = train.base.run_epochs(Engine(), whole, identity, 'JOINT_MEMORY', 1)
                interrupted = Engine(); interrupted.fail_at_update = 1
                with self.assertRaises(KeyboardInterrupt):
                    train.base.run_epochs(interrupted, killed, identity, 'JOINT_MEMORY', 1)
                saved = train.base.load_checkpoint(killed/'last.pt')
                self.assertEqual(saved['completed_epoch'], 0)
                train.validate_saved(saved, identity, 'JOINT_MEMORY', killed, 2)
                resumed = train.base.run_epochs(Engine(99), killed, identity, 'JOINT_MEMORY', 1, saved)
            self.assertEqual(comparable_history(expected), comparable_history(resumed))
            self.assertEqual(expected['model_tensor_sha256'], resumed['model_tensor_sha256'])

    def test_probe_rejects_changed_anchor_even_with_recomputed_model_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            engine = Engine(stage='probe'); identity = identity_for(engine)
            with patch('builtins.print'):
                state = train.base.run_epochs(engine, root, identity, 'JOINT_MEMORY', 2)
            train.validate_saved(state, identity, 'JOINT_MEMORY', root, 2)
            train.check_probe_frozen(engine.model, identity)
            self.assertTrue(all(p.grad is None for p in engine.model.anchor.parameters()))
            altered = deepcopy(state)
            name = next(name for name in altered['model_state'] if name.startswith('anchor.'))
            altered['model_state'][name].view(-1)[0] += .1
            altered['model_tensor_sha256'] = train.base.tensor_tree_hash(altered['model_state'])
            with self.assertRaisesRegex(ValueError, 'probe anchor'):
                train.validate_saved(altered, identity, 'JOINT_MEMORY', root, 2)

    def test_saved_identity_and_metadata_tampering_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); identity = identity_for(Engine())
            with patch('builtins.print'):
                state = train.base.run_epochs(Engine(), root, identity, 'JOINT_MEMORY', 1)
            broken_identity = deepcopy(identity); broken_identity['curriculum'] = 'native'
            with self.assertRaises(ValueError):
                train.validate_saved(state, broken_identity, 'JOINT_MEMORY', root, 2)
            for key,value in (('identity_sha256','other'),('global_updates',3),('curriculum_length',1)):
                broken = deepcopy(state); broken[key] = value
                with self.subTest(key=key), self.assertRaises(ValueError):
                    train.validate_saved(broken, identity, 'JOINT_MEMORY', root, 2)

    def test_summary_uses_common_new_epoch_prefix_and_keeps_negative_gain(self):
        identity = identity_for(Engine())
        states = {arm:state_for(arm, [2.,1.]) for arm in train.DEFAULT_ARMS}
        states['RECURRENT_CONTROL'] = state_for('RECURRENT_CONTROL',[2.,1.,.01])
        states['JOINT_MEMORY'] = state_for('JOINT_MEMORY',[3.,2.])
        partial = train.paired_summary(states, identity, 3)
        comparison = partial['comparisons']['JOINT_MEMORY_vs_RECURRENT_CONTROL']
        self.assertEqual(comparison['common_new_epochs'], 2)
        self.assertEqual(comparison['independent_best_at_common_budget']['mae_gain'], -1.)
        self.assertEqual(comparison['common_last_epoch']['mae_gain'], -1.)
        self.assertFalse(partial['same_completed_new_training_budget'])
        self.assertEqual(partial['status'], 'GW6_TRAINING_PARTIAL')
        self.assertFalse(partial['historical_GW4_GW5_are_equal_budget_controls'])
        self.assertFalse(partial['predictive_accuracy_pass'])
        self.assertFalse(partial['test_evaluated'])
        states['RECURRENT_CONTROL'] = state_for('RECURRENT_CONTROL',[2.,1.])
        complete = train.paired_summary(states, identity, 2)
        self.assertEqual(complete['status'], 'GW6_TRAINING_COMPLETE')
        self.assertTrue(complete['same_completed_new_training_budget'])

    def test_initialization_counts_frozen_and_active_parameters_separately(self):
        joint, probe = (Engine(stage=stage).model for stage in ('joint','probe'))
        a,b = train.initialization(joint),train.initialization(probe)
        self.assertEqual(a['source_tensor_sha256'],b['source_tensor_sha256'])
        self.assertEqual(a['registered_parameters'],b['registered_parameters'])
        self.assertEqual(a['frozen_parameters'],0)
        self.assertGreater(b['frozen_parameters'],0)
        self.assertEqual(b['trainable_parameters'],sum(p.numel() for p in probe.readout.parameters()))


if __name__ == '__main__':
    unittest.main()
