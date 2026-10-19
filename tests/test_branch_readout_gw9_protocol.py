"""Server-only GW9 budget, resume, checkpoint identity and paired screen tests.

Synthetic optimizations exercise the runner; they are not forecasting evidence.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from experiments.branch_readout_gw9 import train, evaluate, reference
from tests.test_original_isp_runner import Engine as PreviousEngine, comparable_history


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Sequential(nn.Linear(1, 3), nn.Dropout(.2), nn.Linear(3, 1))
        self.kernel = nn.Linear(1, 1, bias=False)
        nn.init.zeros_(self.kernel.weight)

    def forward(self, x):
        return self.anchor(x)+self.kernel(x)

    def trainable_parameter_counts(self):
        return dict(total=sum(p.numel() for p in self.parameters()))

    def diagnostics(self):
        return {}


class Engine(PreviousEngine):
    def __init__(self, seed=2025, constant_validation=None):
        super().__init__(seed)
        self.model = ToyModel()
        self._optimizer = torch.optim.Adam(self.model.parameters(), lr=.01)
        self._lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self._optimizer, milestones=[1, 3], gamma=.5)
        self._warm_step, self._cl_step = 202, 6
        self.constant_validation = constant_validation
        train.base.configure_runtime('cpu', 1, seed)

    def evaluate(self, mode):
        values = super().evaluate(mode)
        return values if self.constant_validation is None else (self.constant_validation, .1, .5)


def fixed_prediction(engine):
    # Deterministic finite model-domain reference; no random-number consumption.
    prediction = torch.ones(2, 12, 1, 1)*next(engine.model.anchor.parameters()).detach().mean()
    return 'fixed-input', prediction


def identity_for(engine):
    value = dict(schema=train.SCHEMA, design=train.DESIGN, arms=list(train.ARMS),
        engineering_only=False,
        initializations={arm:train.initialization(engine.model) for arm in train.ARMS},
        data=dict(splits=dict(train=dict(count=4))),
        training_config=dict(warm_epoch=101, cl_epoch=3), source_reference={},
        model_options=dict(hidden=16))
    value['identity_sha256'] = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    return value


def screen_inputs():
    cases = {arm:dict(mae_pooled=value)
        for arm,value in zip(train.ARMS, (10., 10.01, 9.9))}
    groups = {arm:{name:dict(mae=value)
        for name in ('high_value_ge314', 'long_horizon_7_12')}
        for arm,value in zip(train.ARMS, (20., 21., 19.5))}
    return dict(cases=cases, groups=groups)


class BranchReadoutProtocolTests(unittest.TestCase):
    def test_parser_locks_budget_capacity_and_unique_arms(self):
        argv = ['--gw8-run', 'old', '--probe-dir', 'probe', '--run-dir', 'new']
        args = train.parse_args(argv)
        self.assertEqual((args.epochs, args.hidden), (5, 16))
        for extra in (['--epochs','20'], ['--epochs','0'], ['--hidden','32'],
            ['--only','BRANCH_READOUT','BRANCH_READOUT']):
            with self.subTest(extra=extra), self.assertRaises(SystemExit):
                train.parse_args(argv+extra)
        with self.assertRaises(SystemExit):
            train.parse_args(['--run-dir','new'])

    def test_stage_zero_is_best_on_ties(self):
        with tempfile.TemporaryDirectory() as directory, patch('builtins.print'), \
                patch.object(train,'fixed_prediction',fixed_prediction):
            engine = Engine(constant_validation=1.)
            identity = identity_for(engine)
            state = train.run_epochs(engine, Path(directory), identity, 'R_CONTINUE', 2)
            train.validate_saved(state, identity, 'R_CONTINUE', directory, 2)
            self.assertEqual(state['best_epoch'], 0)
            self.assertEqual(state['wait'], 2)
            self.assertFalse(train.arm_summary(state)['exceeds_stage_zero'])
            self.assertEqual(train.arm_summary(state)['best_metrics']['mae'], 1.)

    def test_exact_resume_preserves_optimizer_rng_order_and_prediction_reference(self):
        with tempfile.TemporaryDirectory() as directory, patch('builtins.print'), \
                patch.object(train,'fixed_prediction',fixed_prediction):
            whole, split = Path(directory)/'whole', Path(directory)/'split'
            whole.mkdir(); split.mkdir()
            identity = identity_for(Engine())
            expected = train.run_epochs(Engine(), whole, identity, 'R_CONTINUE', 4)
            train.run_epochs(Engine(), split, identity, 'R_CONTINUE', 2)
            saved = train.base.load_checkpoint(split/'last.pt')
            train.validate_saved(saved, identity, 'R_CONTINUE', split, 2)
            resumed = train.run_epochs(Engine(97), split, identity, 'R_CONTINUE', 4, saved)
            train.validate_saved(resumed, identity, 'R_CONTINUE', split, 2)
            self.assertEqual(comparable_history(expected), comparable_history(resumed))
            for key in ('model_state','optimizer_state','scheduler_state','rng_state',
                    'train_order','fixed_reference_prediction'):
                self.assertEqual(train.base.tensor_tree_hash(expected[key]),
                    train.base.tensor_tree_hash(resumed[key]), key)
            self.assertEqual(resumed['global_updates'], 8)
            self.assertTrue(all(row['curriculum_length']==12 for row in resumed['history']))

    def test_killed_first_epoch_replays_from_stage_zero(self):
        with tempfile.TemporaryDirectory() as directory, patch('builtins.print'), \
                patch.object(train,'fixed_prediction',fixed_prediction):
            whole, killed = Path(directory)/'whole', Path(directory)/'killed'
            whole.mkdir(); killed.mkdir()
            identity = identity_for(Engine())
            expected = train.run_epochs(Engine(), whole, identity, 'R_CONTINUE', 1)
            interrupted = Engine(); interrupted.fail_at_update = 1
            with self.assertRaises(KeyboardInterrupt):
                train.run_epochs(interrupted, killed, identity, 'R_CONTINUE', 1)
            saved = train.base.load_checkpoint(killed/'last.pt')
            self.assertEqual(saved['completed_epoch'], 0)
            train.validate_saved(saved, identity, 'R_CONTINUE', killed, 2)
            resumed = train.run_epochs(Engine(99), killed, identity, 'R_CONTINUE', 1, saved)
            self.assertEqual(comparable_history(expected), comparable_history(resumed))
            self.assertEqual(expected['model_tensor_sha256'], resumed['model_tensor_sha256'])

    def test_saved_state_rejects_identity_budget_order_selection_and_reference_tampering(self):
        with tempfile.TemporaryDirectory() as directory, patch('builtins.print'), \
                patch.object(train,'fixed_prediction',fixed_prediction):
            engine = Engine(); identity = identity_for(engine)
            state = train.run_epochs(engine, Path(directory), identity, 'R_CONTINUE', 1)
            for mutation in (lambda s:s.update(global_updates=3),
                lambda s:s.update(identity_sha256='bad'),
                lambda s:s.update(train_order=[0,0,1,2]),
                lambda s:s.update(best_epoch=4),
                lambda s:s['fixed_reference_prediction'].fill_(float('nan')),
                lambda s:s['model_state']['kernel.weight'].add_(1.),
                lambda s:s.update(stage_zero_model_tensor_sha256='bad')):
                bad = deepcopy(state); mutation(bad)
                with self.assertRaises(ValueError):
                    train.validate_saved(bad, identity, 'R_CONTINUE', directory, 2)

    def test_loop_refuses_extension_or_negative_budget(self):
        for epoch in (-1, 6, 20):
            with self.assertRaises(ValueError):
                train.run_epochs(None, Path('unused'), {}, 'R_CONTINUE', epoch)

    def test_stage_zero_must_replay_certified_source(self):
        with tempfile.TemporaryDirectory() as directory, patch('builtins.print'), \
                patch.object(train,'fixed_prediction',fixed_prediction):
            engine = Engine(); identity = identity_for(engine)
            identity['source_reference'] = dict(source_validation_metrics=
                dict(mae=999.,rmse=999.,mape_fraction=999.))
            with self.assertRaisesRegex(ValueError,'certified GW8 recurrent BEST'):
                train.run_epochs(engine, Path(directory), identity, 'R_CONTINUE', 0)

    def test_screen_pass_is_only_development_confirmation_candidate(self):
        best = screen_inputs(); last = deepcopy(best)
        last['cases']['BRANCH_READOUT']['mae_pooled'] = 9.95
        result = evaluate.screening(best,last)
        self.assertTrue(result['candidate_for_confirmation'])
        self.assertFalse(result['test_evaluated'])
        self.assertIn('not_statistical_significance',result['interpretation'])
        self.assertTrue(result['screening_failure_does_not_authorize_budget_extension'])

    def test_screen_rejects_each_control_and_late_or_regime_regression(self):
        mutations = (
            lambda b,l:b['cases']['BRANCH_READOUT'].update(mae_pooled=9.99),
            lambda b,l:b['cases']['SUM_CONTROL'].update(mae_pooled=9.91),
            lambda b,l:l['cases']['BRANCH_READOUT'].update(mae_pooled=10.2),
            lambda b,l:b['groups']['BRANCH_READOUT']['high_value_ge314'].update(mae=20.1),
            lambda b,l:b['groups']['BRANCH_READOUT']['long_horizon_7_12'].update(mae=None))
        for mutate in mutations:
            best = screen_inputs(); last = deepcopy(best); mutate(best,last)
            self.assertFalse(evaluate.screening(best,last)['candidate_for_confirmation'])

    def test_groups_keep_original_mask_horizon_and_threshold(self):
        y = np.ones((2,12,2))*314.; prediction = y+1.
        valid = np.ones_like(y,dtype=bool)
        y[0,0,0] = 313.; valid[1,11,1] = False; prediction[1,11,1] = 9999.
        groups = evaluate.group_metrics(prediction,y,valid)
        self.assertEqual(groups['high_value_ge314']['valid_count'],46)
        self.assertEqual(groups['high_value_ge314']['mae'],1.)
        self.assertEqual(groups['long_horizon_7_12']['valid_count'],23)
        self.assertEqual(groups['long_horizon_7_12']['mae'],1.)

    def test_epoch_zero_replay_uses_stage_zero_not_last_history(self):
        state = dict(stage_zero=dict(validation_mae=1.,validation_rmse=2.,validation_mape_fraction=.1),
            history=[dict(validation_mae=4.,validation_rmse=5.,validation_mape_fraction=.4)])
        with patch.object(evaluate.metrics,'native_metrics',return_value=
                dict(mae=1.,rmse=2.,mape_fraction=.1)):
            result = evaluate.replay_metrics(None,None,state,0,None)
        self.assertTrue(result['passed'])
        self.assertEqual(result['expected']['mae'],1.)

    def test_paired_axes_reject_node_order_dtype_and_mask_changes(self):
        arrays = {key:np.arange(4,dtype=np.int64) for key in evaluate.AXIS_KEYS}
        evaluate.require_matching_axes(arrays,deepcopy(arrays))
        for key,value in (('node_ids',np.arange(4,dtype=np.int64)[::-1]),
                ('y',np.arange(4,dtype=np.float32)), ('valid',np.ones(4,dtype=np.int64))):
            wrong = deepcopy(arrays); wrong[key] = value
            with self.assertRaisesRegex(ValueError,key):
                evaluate.require_matching_axes(arrays,wrong)

    def test_source_identity_is_locked_and_rejected_before_source_files_are_loaded(self):
        wrong = dict(schema=reference.gw8.SCHEMA, design=reference.gw8.DESIGN,
            identity_sha256='different-run',arms=list(reference.gw8.ARMS),
            engineering_only=False,best_selection_includes_stage_zero=True)
        with patch.object(reference,'read_json',return_value=wrong), \
                patch.object(reference,'verify_identity_digest'):
            with self.assertRaisesRegex(ValueError,'existing certified GW8'):
                reference.certify(Path('other-run'))


if __name__ == '__main__':
    unittest.main()
