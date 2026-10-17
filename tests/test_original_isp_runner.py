"""Resume, matched-budget and checkpoint boundaries for ISP-Lite runners."""
from copy import deepcopy
import logging
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.chronological import train_original_isp as runner


class Loader:
    def __init__(self):
        self.samples = [1., 2., 3., 4.]
        self.num_batch = 2

    def get_iterator(self):
        for start in (0, 2):
            yield torch.tensor(self.samples[start:start+2]).reshape(-1, 1)


class Engine:
    """Small stochastic optimizer with the frozen engine's curriculum rules."""
    def __init__(self, seed=2025):
        runner.configure_runtime('cpu', 1, seed)
        self.model = torch.nn.Sequential(torch.nn.Linear(1, 3), torch.nn.Dropout(.2), torch.nn.Linear(3, 1))
        self._optimizer = torch.optim.Adam(self.model.parameters(), lr=.01)
        self._lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self._optimizer, milestones=[1, 3], gamma=.5)
        self._device = torch.device('cpu')
        self._dataloader = dict(train_loader=Loader(), val_loader=Loader())
        self._iter_cnt, self._cl_len = 0, 0
        self._logger = logging.getLogger('test-isp')
        self._warm_step, self._cl_step = 4, 2
        self._patience = 0  # Runner must ignore patience to finish common budget.
        self.fail_at_update = None

    def train_batch(self):
        self.model.train()
        self._dataloader['train_loader'].shuffle()
        losses = []
        for x in self._dataloader['train_loader'].get_iterator():
            if self.fail_at_update == self._iter_cnt:
                raise KeyboardInterrupt('simulated job kill')
            self._optimizer.zero_grad()
            x = x + random.random()*.01 + float(np.random.random())*.01
            loss = (self.model(x)-x*.5).square().mean()
            loss.backward()
            self._optimizer.step()
            self._iter_cnt += 1
            if self._iter_cnt < self._warm_step:
                self._cl_len = 12
            elif self._iter_cnt == self._warm_step:
                self._cl_len = 1
            elif (self._iter_cnt-self._warm_step) % self._cl_step == 0:
                self._cl_len = min(12, self._cl_len+1)
            losses.append(float(loss.detach()))
        mae = float(np.mean(losses))
        return mae, mae*.1, mae**.5

    def evaluate(self, mode):
        if mode != 'val':
            raise AssertionError('Test loader must never be used')
        self.model.eval()
        with torch.no_grad():
            values = [(self.model(x)-x*.5).abs() for x in self._dataloader['val_loader'].get_iterator()]
        mae = float(torch.cat(values).mean())
        return mae, mae*.1, mae+.5


def identity_for(engine, arm='ORIGINAL'):
    init = dict(original_tensor_sha256=runner.tensor_tree_hash(engine.model.state_dict()), added_tensor_sha256='empty')
    return dict(identity_sha256='test-identity', arms=['ORIGINAL', 'STATE', 'STATE_PROPAGATION'],
        initializations={arm: init}, engineering_only=True,
        training_config=dict(warm_epoch=2, cl_epoch=1),
        data=dict(splits=dict(train=dict(count=4))))


def comparable_history(state):
    keys = ('epoch', 'train_mae', 'train_mape_fraction', 'train_rmse', 'validation_mae',
            'validation_rmse', 'validation_mape_fraction', 'learning_rate', 'global_updates',
            'curriculum_length', 'train_order_sha256')
    return [{k: row[k] for k in keys} for row in state['history']]


class ISPRunnerTests(unittest.TestCase):
    def test_parser_default_matched_pilot(self):
        args = runner.parse_args(['--run-dir', 'test-run'])
        self.assertEqual(args.epochs, 5)
        self.assertEqual(args.arms, list(runner.ARMS))
        self.assertFalse(args.resume)

    def test_parser_only_requires_registered_unique_subset(self):
        for argv in (['--only', 'STATE', 'STATE'], ['--arms', 'ORIGINAL', '--only', 'STATE'],
                     ['--arms', 'STATE', 'STATE'], ['--epochs', '0'], ['--epochs', '101']):
            with self.subTest(argv=argv), self.assertRaises(SystemExit):
                runner.parse_args(['--run-dir', 'test-run', *argv])

    def test_resume_exact_optimizer_rng_order_curriculum_and_scheduler(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            whole, split = Path(tmp)/'whole', Path(tmp)/'split'
            whole.mkdir(); split.mkdir()
            a = Engine()
            identity = identity_for(a)
            with patch('builtins.print'):
                completed = runner.run_epochs(a, whole, identity, 'ORIGINAL', 4)
                b = Engine()
                runner.run_epochs(b, split, identity, 'ORIGINAL', 2)
                checkpoint = runner.load_checkpoint(split/'last.pt')
                runner.validate_saved(checkpoint, identity, 'ORIGINAL', split, 2)
                c = Engine(seed=99)
                resumed = runner.run_epochs(c, split, identity, 'ORIGINAL', 4, checkpoint)
            self.assertEqual(comparable_history(completed), comparable_history(resumed))
            for key in ('model_state', 'optimizer_state', 'scheduler_state', 'rng_state', 'train_order'):
                self.assertEqual(runner.tensor_tree_hash(completed[key]), runner.tensor_tree_hash(resumed[key]), key)
            self.assertEqual(resumed['completed_epoch'], 4)
            self.assertEqual(resumed['curriculum_length'], 3)
            self.assertEqual(resumed['global_updates'], 8)

    def test_killed_first_epoch_resumes_epoch_zero(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            whole, killed = Path(tmp)/'whole', Path(tmp)/'killed'
            whole.mkdir(); killed.mkdir()
            a = Engine()
            identity = identity_for(a)
            with patch('builtins.print'):
                expected = runner.run_epochs(a, whole, identity, 'ORIGINAL', 1)
                b = Engine()
                b.fail_at_update = 1
                with self.assertRaises(KeyboardInterrupt):
                    runner.run_epochs(b, killed, identity, 'ORIGINAL', 1)
                checkpoint = runner.load_checkpoint(killed/'last.pt')
                self.assertEqual(checkpoint['completed_epoch'], 0)
                runner.validate_saved(checkpoint, identity, 'ORIGINAL', killed, 2)
                recovered = runner.run_epochs(Engine(99), killed, identity, 'ORIGINAL', 1, checkpoint)
            self.assertEqual(comparable_history(expected), comparable_history(recovered))
            self.assertEqual(expected['model_tensor_sha256'], recovered['model_tensor_sha256'])

    def test_resume_rejects_identity_budget_selection_and_tensor_tampering(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            run = Path(tmp)
            engine = Engine()
            identity = identity_for(engine)
            with patch('builtins.print'):
                runner.run_epochs(engine, run, identity, 'ORIGINAL', 2)
            state = runner.load_checkpoint(run/'last.pt')
            runner.validate_saved(state, identity, 'ORIGINAL', run, 2)
            mutations = [lambda s: s.update(identity_sha256='wrong'),
                         lambda s: s.update(arm='STATE'),
                         lambda s: s.update(global_updates=5),
                         lambda s: s.update(train_order=[0, 0, 1, 2]),
                         lambda s: s.update(curriculum_length=12),
                         lambda s: s.update(best_epoch=0),
                         lambda s: s['scheduler_state'].update(last_epoch=0),
                         lambda s: s['model_state']['0.weight'].add_(1)]
            for mutation in mutations:
                bad = deepcopy(state)
                mutation(bad)
                with self.assertRaises(ValueError):
                    runner.validate_saved(bad, identity, 'ORIGINAL', run, 2)
            best = run/state['best_checkpoint_file']
            best.write_bytes(best.read_bytes()+b'altered')
            with self.assertRaises(ValueError):
                runner.validate_saved(state, identity, 'ORIGINAL', run, 2)

    def test_same_budget_summary_uses_common_prefix(self):
        def state(arm, metrics):
            history = [dict(epoch=i, validation_mae=value, validation_rmse=value+1,
                validation_mape_fraction=value*.01) for i, value in enumerate(metrics, 1)]
            best = min(history, key=lambda r: r['validation_mae'])
            return dict(arm=arm, history=history, completed_epoch=len(history), best_epoch=best['epoch'],
                best_validation_mae=best['validation_mae'], best_checkpoint_file='best.pt',
                best_checkpoint_sha256='test', model_tensor_sha256='test', global_updates=2*len(history),
                initialization={})
        states = {'ORIGINAL': state('ORIGINAL', [4., 3., 1.]),
                  'STATE': state('STATE', [3., 2.]),
                  'STATE_PROPAGATION': state('STATE_PROPAGATION', [2., 1.5])}
        identity = dict(arms=list(runner.ARMS), engineering_only=True, data={})
        summary = runner.paired_summary(states, identity, 3)
        self.assertFalse(summary['same_completed_training_budget'])
        self.assertEqual(summary['common_completed_epochs'], 2)
        comparison = summary['comparisons']['STATE_vs_ORIGINAL']
        self.assertEqual(comparison['independent_best_at_common_budget']['mae_gain'], 1.)
        self.assertEqual(comparison['best_epochs']['ORIGINAL'], 2)
        self.assertEqual(summary['status'], 'ORIGINAL_ISP_TRAINING_PARTIAL')
        self.assertFalse(summary['test_evaluated'])

    def test_missing_arm_has_no_comparability_claim(self):
        summary = runner.paired_summary({}, dict(arms=list(runner.ARMS), engineering_only=True, data={}), 5)
        self.assertFalse(summary['same_completed_training_budget'])
        self.assertIsNone(summary['common_completed_epochs'])
        self.assertEqual(summary['comparisons'], {})
        self.assertFalse(summary['primary_baseline_present'])

    def test_unregistered_original_is_not_claimed_as_in_run_baseline(self):
        summary = runner.paired_summary({}, dict(arms=['STATE'], engineering_only=True, data={}), 5)
        self.assertEqual(summary['primary_baseline'], 'original_IGSTGNN_558ab01_not_present_in_this_run')
        self.assertFalse(summary['primary_baseline_present'])

    def test_safe_load_never_unrestricted_pickle(self):
        with patch.object(torch, 'load', return_value={'ok': True}) as load:
            self.assertEqual(runner.load_checkpoint('checkpoint.pt'), {'ok': True})
        self.assertTrue(load.call_args.kwargs['weights_only'])

    def test_validation_progress_restores_iterator_on_failure(self):
        loader = Loader()
        original = loader.get_iterator
        with patch('builtins.print'):
            with self.assertRaises(RuntimeError):
                with runner.batch_progress(loader, 'STATE', 'validation', 1):
                    next(loader.get_iterator())
                    raise RuntimeError('failure')
        self.assertEqual(loader.get_iterator, original)

    def test_fixed_recipe_and_test_free_entry(self):
        source = Path(runner.__file__).read_text(encoding='utf-8')
        self.assertNotIn("evaluate('test')", source)
        self.assertNotIn('test_loader', source)
        self.assertNotIn('source-run', source)
        self.assertEqual(runner.ISP_CONFIG, dict(state_dim=16, graph_rank=8, max_graph_mix=.25, max_state_delta=.1))
        self.assertEqual(runner.MILESTONES, [1, 38, 46, 54, 62, 70, 80])


if __name__ == '__main__':
    unittest.main()
