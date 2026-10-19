"""Server-only tests for the synchronous, frozen GW9 probe protocol.

Synthetic optimizations verify interruption recovery and budget boundaries;
they do not provide forecasting evidence. Never execute these locally.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from experiments.branch_readout_gw9 import probe
from experiments.chronological.original_igstgnn_558ab01 import metrics as native_metrics
from tests.test_original_isp_runner import Loader


class ToyRunner:
    """Paired optimizers with stochastic updates and an intentional kill point."""
    def __init__(self, seed=2025, constant_validation=None):
        probe.base.configure_runtime('cpu', 1, seed)
        template = nn.Sequential(nn.Linear(1, 3), nn.Dropout(.2), nn.Linear(3, 1, bias=False))
        nn.init.zeros_(template[-1].weight)
        self.heads = nn.ModuleDict({arm:deepcopy(template) for arm in probe.ARMS})
        self.optimizers = {arm:torch.optim.Adam(self.heads[arm].parameters(), lr=.01)
            for arm in probe.ARMS}
        self.schedulers = {arm:torch.optim.lr_scheduler.MultiStepLR(self.optimizers[arm],
            milestones=[1, 3], gamma=.5) for arm in probe.ARMS}
        self.loader = Loader()
        self.backbone_tensor_sha256 = 'unchanged-frozen-backbone'
        self.constant_validation = constant_validation
        self.fail_before = None
        self.steps = {arm:0 for arm in probe.ARMS}
        self.frozen_checks = 0
        probe.base.configure_runtime('cpu', 1, seed)

    def check_backbone(self):
        self.frozen_checks += 1
        return True

    def train_epoch(self, epoch):
        self.heads.train()
        self.loader.shuffle()
        losses = {arm:[] for arm in probe.ARMS}
        for batch_index, x in enumerate(self.loader.get_iterator()):
            x = x+random.random()*.01+float(np.random.random())*.01
            for index, arm in enumerate(probe.ARMS):
                if self.fail_before == (epoch, batch_index, index):
                    raise KeyboardInterrupt('simulated kill between paired optimizer steps')
                optimizer = self.optimizers[arm]
                optimizer.zero_grad(set_to_none=True)
                prediction = .5*x+self.heads[arm](x)
                loss = (prediction-.25*x).abs().mean()
                loss.backward()
                optimizer.step()
                self.steps[arm] += 1
                losses[arm].append(float(loss.detach()))
        values = {arm:dict(mae=float(np.mean(rows)), mape_fraction=.1, rmse=.5)
            for arm, rows in losses.items()}
        return values, dict(seconds=0., peak_allocated_bytes=None)

    def evaluate(self):
        self.heads.eval()
        x = torch.arange(1., 5.).reshape(-1, 1)
        truth = .25*x
        with torch.no_grad():
            predictions = {'FROZEN_R':.5*x}
            predictions.update({arm:.5*x+self.heads[arm](x) for arm in probe.ARMS})
        records = {}
        for case, prediction in predictions.items():
            error = (prediction-truth).double()
            mae = float(error.abs().mean())
            rmse = float(error.square().mean().sqrt())
            mape = float((error.abs()/truth.double()).mean())
            if self.constant_validation is not None:
                mae, rmse, mape = self.constant_validation, .5, .1
            records[case] = dict(mae_pooled=mae, rmse_pooled=rmse, mape_pooled=mape,
                signed_bias=float(error.mean()), window_p95=float(error.abs().max()),
                groups={group:dict(mae=mae, valid_count=4)
                    for group in ('high_value_ge314', 'long_horizon_7_12')},
                native=dict(mae=mae, rmse=rmse, mape_fraction=mape))
        return records, {}


def identity_for(runner):
    initializations = {arm:dict(
        head_tensor_sha256=probe.base.tensor_tree_hash(runner.heads[arm].state_dict()),
        registered_parameters=sum(p.numel() for p in runner.heads[arm].parameters()),
        trainable_parameters=sum(p.numel() for p in runner.heads[arm].parameters()))
        for arm in probe.ARMS}
    value = dict(schema=probe.SCHEMA, design=probe.DESIGN, arms=list(probe.ARMS),
        initializations=initializations, engineering_only=False,
        source_reference=dict(source_initialization=dict(
            model_tensor_sha256=runner.backbone_tensor_sha256)),
        backbone_tensor_sha256=runner.backbone_tensor_sha256,
        data=dict(splits=dict(train=dict(count=4))), fixed_screen_epochs=probe.PROBE_EPOCHS)
    value['identity_sha256'] = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    return value


def comparable_history(state):
    # Wall-clock timing and immutable checkpoint UUID names may differ.
    return [{key:row[key] for key in ('epoch','validation','train_metrics')}
        for row in state['history']]


class SyntheticLoader:
    def __init__(self):
        self.samples = [0,1,2]
        self.num_batch = 2
        self.shuffle = lambda:None

    def get_iterator(self):
        for start in (0,2):
            indices = self.samples[start:start+2]
            scales = torch.tensor([1.,2.,10.])[indices]
            x = scales[:,None,None,None].expand(-1,12,2,2).clone()
            x[...,1] *= 2.
            truth = 1.+x[...,:1]*.25
            if 0 in indices:
                truth[indices.index(0),0,0,0] = 0.
            batch = len(indices)
            yield dict(x_data=x,y_data=truth,
                incident_features=torch.zeros(batch,1,1),incident_position=torch.zeros(batch,1),
                incident_distances=torch.ones(batch,2,1),sensor_data={})


class FrozenExtractor(nn.Module):
    """Eval BN/dropout state makes accidental train mode or duplicate calls visible."""
    stage, head, channels, gap = 'probe',None,2,1

    def __init__(self):
        super().__init__()
        self.anchor = nn.Sequential(nn.BatchNorm1d(2),nn.Dropout(.5))
        self.anchor.requires_grad_(False)
        self.eval()
        self.calls = 0

    def capture_features(self,x,incident_data=None,sensor_data=None):
        self.calls += 1
        features = self.anchor(x.reshape(-1,2)).reshape_as(x)
        return dict(base=features[...,:1].detach(),S=features.detach(),
            Q=features.flip(-1).detach())

    def assert_cache_clear(self):
        return True


class BranchReadoutProbeTests(unittest.TestCase):
    def test_fixed_probe_budget_and_required_source(self):
        args = probe.parse_args(['--gw8-run','old','--run-dir','new'])
        self.assertEqual((args.epochs,args.hidden), (3,16))
        for extra in (['--epochs','4'], ['--epochs','0'], ['--hidden','32']):
            with self.subTest(extra=extra), self.assertRaises(SystemExit):
                probe.parse_args(['--gw8-run','old','--run-dir','new',*extra])
        for target in (-1,4,20):
            with self.assertRaises(ValueError):
                probe.run_epochs(None,Path('unused'),{},target)

    def test_stage_zero_wins_ties_for_both_heads(self):
        with tempfile.TemporaryDirectory() as directory, patch('builtins.print'):
            runner = ToyRunner(constant_validation=1.)
            identity = identity_for(runner)
            state = probe.run_epochs(runner,Path(directory),identity,2)
            probe.validate_saved(state,identity,Path(directory),2)
            self.assertEqual(state['completed_epoch'],2)
            self.assertEqual(state['global_updates'],4)
            self.assertEqual(state['head_updates'],{arm:4 for arm in probe.ARMS})
            for arm in probe.ARMS:
                self.assertEqual(state['best'][arm]['epoch'],0)
                self.assertEqual(state['best'][arm]['mae_pooled'],1.)

    def test_resume_restores_both_heads_optimizers_rng_order_and_schedulers(self):
        with tempfile.TemporaryDirectory() as directory, patch('builtins.print'):
            whole, split = Path(directory)/'whole',Path(directory)/'split'
            whole.mkdir(); split.mkdir()
            identity = identity_for(ToyRunner())
            expected = probe.run_epochs(ToyRunner(),whole,identity,3)
            probe.run_epochs(ToyRunner(),split,identity,1)
            saved = probe.base.load_checkpoint(split/'last.pt')
            probe.validate_saved(saved,identity,split,2)
            resumed = probe.run_epochs(ToyRunner(seed=91),split,identity,3,saved)
            probe.validate_saved(resumed,identity,split,2)
            self.assertEqual(comparable_history(expected),comparable_history(resumed))
            for key in ('heads_state','optimizers_state','schedulers_state',
                    'rng_state','train_order','head_updates'):
                self.assertEqual(probe.base.tensor_tree_hash(expected[key]),
                    probe.base.tensor_tree_hash(resumed[key]),key)

    def test_kill_between_head_steps_replays_both_from_last_atomic_epoch(self):
        with tempfile.TemporaryDirectory() as directory, patch('builtins.print'):
            whole, killed = Path(directory)/'whole',Path(directory)/'killed'
            whole.mkdir(); killed.mkdir()
            identity = identity_for(ToyRunner())
            expected = probe.run_epochs(ToyRunner(),whole,identity,2)
            interrupted = ToyRunner()
            interrupted.fail_before = (1,0,1)
            with self.assertRaises(KeyboardInterrupt):
                probe.run_epochs(interrupted,killed,identity,2)
            self.assertEqual(interrupted.steps[probe.ARMS[0]],1)
            self.assertEqual(interrupted.steps[probe.ARMS[1]],0)
            saved = probe.base.load_checkpoint(killed/'last.pt')
            self.assertEqual(saved['completed_epoch'],0)
            self.assertEqual(saved['head_updates'],{arm:0 for arm in probe.ARMS})
            resumed = probe.run_epochs(ToyRunner(seed=44),killed,identity,2,saved)
            self.assertEqual(comparable_history(expected),comparable_history(resumed))
            for key in ('heads_state','optimizers_state','schedulers_state','rng_state','train_order'):
                self.assertEqual(probe.base.tensor_tree_hash(expected[key]),
                    probe.base.tensor_tree_hash(resumed[key]),key)

    def test_resume_rejects_budget_identity_head_order_scheduler_and_selection_changes(self):
        with tempfile.TemporaryDirectory() as directory, patch('builtins.print'):
            runner = ToyRunner()
            identity = identity_for(runner)
            state = probe.run_epochs(runner,Path(directory),identity,1)
            arm = probe.ARMS[0]
            mutations = (
                lambda s:s.update(identity_sha256='wrong'),
                lambda s:s.update(global_updates=3),
                lambda s:s['head_updates'].update({arm:3}),
                lambda s:s.update(train_order=[0,0,1,2]),
                lambda s:s['schedulers_state'][arm].update(last_epoch=0),
                lambda s:s['heads_state'][arm]['2.weight'].add_(1.),
                lambda s:s['best'][arm].update(epoch=3),
                lambda s:s['history'][0]['validation'][arm]['native'].update(rmse=float('nan')),
                lambda s:s['history'][0]['validation']['FROZEN_R'].update(signed_bias=100.),
                lambda s:s.update(backbone_tensor_sha256='changed'))
            for mutation in mutations:
                changed = deepcopy(state)
                mutation(changed)
                with self.assertRaises(ValueError):
                    probe.validate_saved(changed,identity,Path(directory),2)

    def test_committed_heads_are_immutable_snapshots_of_live_parameters(self):
        with tempfile.TemporaryDirectory() as directory, patch('builtins.print'):
            runner = ToyRunner()
            identity = identity_for(runner)
            state = probe.run_epochs(runner,Path(directory),identity,1)
            before = probe.base.tensor_tree_hash(state['heads_state'])
            with torch.no_grad():
                for head in runner.heads.values():
                    head[-1].weight.add_(10.)
            self.assertEqual(before,probe.base.tensor_tree_hash(state['heads_state']))
            saved = probe.base.load_checkpoint(Path(directory)/'last.pt')
            self.assertEqual(before,probe.base.tensor_tree_hash(saved['heads_state']))

    def test_production_streams_one_frozen_forward_per_batch_and_pools_validation(self):
        probe.base.configure_runtime('cpu',1,2025)
        extractor = FrozenExtractor()
        loader = SyntheticLoader()
        truth_rows,base_rows = [],[]
        with torch.no_grad():
            for raw in loader.get_iterator():
                truth_rows.append(raw['y_data'].squeeze(-1).numpy())
                base_rows.append(extractor.capture_features(raw['x_data'])['base'].squeeze(-1).numpy())
        truth,baseline_prediction = np.concatenate(truth_rows),np.concatenate(base_rows)
        baseline = dict(y=truth,valid=truth!=0,
            direct_support=np.ones((3,2),dtype=bool),pred__FROZEN_R=baseline_prediction)
        data = SimpleNamespace(loaders=dict(train_loader=loader,val_loader=SyntheticLoader()),
            config=dict(lrate=.01,wdecay=0.,clip_grad_value=1.),
            scaler=SimpleNamespace(inverse_transform=lambda value:value))
        modules = SimpleNamespace(metrics=native_metrics)
        runner = probe.ProbeEngine(extractor,data,modules,'cpu',baseline)
        frozen = probe.base.tensor_tree_hash(extractor.anchor.state_dict())
        extractor.calls = 0
        with patch('builtins.print'):
            train_metrics,timing = runner.train_epoch(1)
            self.assertEqual(extractor.calls,loader.num_batch)
            records,predictions = runner.evaluate()
        self.assertEqual(extractor.calls,2*loader.num_batch)
        self.assertTrue(timing['shared_forward_between_heads'])
        self.assertEqual(frozen,probe.base.tensor_tree_hash(extractor.anchor.state_dict()))
        self.assertFalse(any(module.training for module in extractor.anchor.modules()))
        self.assertFalse(any(p.grad is not None for p in extractor.anchor.parameters()))
        expected = probe.metrics.native_metrics(baseline_prediction,truth,modules)
        self.assertEqual(records['FROZEN_R']['native'],expected)
        error = (baseline_prediction-truth)[truth!=0].astype(np.float64)
        self.assertAlmostEqual(records['FROZEN_R']['mae_pooled'],float(np.abs(error).mean()))
        batch_means = [float(np.abs((baseline_prediction[part]-truth[part])[
            truth[part]!=0].astype(np.float64)).mean()) for part in (slice(0,2),slice(2,3))]
        self.assertNotAlmostEqual(records['FROZEN_R']['mae_pooled'],float(np.mean(batch_means)))
        for arm in probe.ARMS:
            self.assertEqual(train_metrics[arm]['updates'],loader.num_batch)
            self.assertTrue(bool(runner.heads[arm].output.weight.detach().abs().sum()>0))
            self.assertEqual(predictions[arm].shape,truth.shape)

    def test_probe_screen_pass_only_permits_joint_stage_and_never_confirms_accuracy(self):
        records,_ = ToyRunner(constant_validation=10.).evaluate()
        records['BRANCH_READOUT']['mae_pooled'] = 9.9
        result = probe.probe_screening(records,deepcopy(records))
        self.assertTrue(result['eligible_for_joint_screen'])
        self.assertFalse(result['candidate_for_confirmation'])
        self.assertFalse(result['frozen_R_is_equal_budget_control'])
        self.assertFalse(result['joint_training_started'])
        self.assertFalse(result['test_evaluated'])

    def test_probe_screen_rejects_control_late_or_regime_regression(self):
        mutations = (
            lambda b,l:b['BRANCH_READOUT'].update(mae_pooled=9.99),
            lambda b,l:b['SUM_CONTROL'].update(mae_pooled=9.91),
            lambda b,l:l['BRANCH_READOUT'].update(mae_pooled=10.2),
            lambda b,l:b['BRANCH_READOUT']['groups']['high_value_ge314'].update(mae=10.1),
            lambda b,l:b['BRANCH_READOUT']['groups']['long_horizon_7_12'].update(mae=None))
        for mutate in mutations:
            best,_ = ToyRunner(constant_validation=10.).evaluate()
            best['BRANCH_READOUT']['mae_pooled'] = 9.9
            last = deepcopy(best)
            mutate(best,last)
            self.assertFalse(probe.probe_screening(best,last)['eligible_for_joint_screen'])


if __name__ == '__main__':
    unittest.main()
