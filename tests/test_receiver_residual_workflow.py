"""Run on the server: GW4 interruption, anchor, archive and reuse boundaries."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.chronological import train_original_receiver_residual as train
from experiments.chronological import evaluate_original_receiver_residual as evaluation
from experiments.chronological import receiver_residual_reference as reference
from experiments.chronological.original_receiver_residual import ReceiverResidualIGSTGNN, split_state
from experiments.chronological.verify_original_receiver_residual import fixture
from tests.test_original_isp_runner import Engine as PreviousFixtureEngine, comparable_history
from tests.test_original_graph_window_evaluation import fixture as archive_fixture


class Engine(PreviousFixtureEngine):
    def __init__(self, seed=2025):
        super().__init__(seed)
        anchor, self.x, self.target = fixture('cpu')
        self.model = ReceiverResidualIGSTGNN(anchor, 'RECEIVER_RESIDUAL')
        self._optimizer = torch.optim.Adam([p for p in self.model.parameters() if p.requires_grad], lr=.002)
        self._lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self._optimizer, milestones=[1, 3], gamma=.5)
        train.base.configure_runtime('cpu', 1, seed)

    def pair(self, ids):
        scale = ids.reshape(-1, 1, 1, 1)/4.
        return self.x[:1]*scale, self.target[:1]*scale

    def train_batch(self):
        self.model.train()
        self._dataloader['train_loader'].shuffle()
        values = []
        for ids in self._dataloader['train_loader'].get_iterator():
            if self.fail_at_update == self._iter_cnt:
                raise KeyboardInterrupt('simulated interrupted GW4 epoch')
            x, y = self.pair(ids)
            self._optimizer.zero_grad(set_to_none=True)
            loss = (self.model(x)-y-.01*torch.randn_like(y)).square().mean()
            loss.backward()
            self._optimizer.step()
            self._iter_cnt += 1
            if self._iter_cnt < self._warm_step:
                self._cl_len = 12
            elif self._iter_cnt == self._warm_step:
                self._cl_len = 1
            elif (self._iter_cnt-self._warm_step) % self._cl_step == 0:
                self._cl_len = min(12, self._cl_len+1)
            values.append(float(loss.detach()))
        value = float(np.mean(values))
        return value, value*.1, value**.5

    def evaluate(self, mode):
        if mode != 'val':
            raise AssertionError('No test evaluation')
        self.model.eval()
        self.model.start_diagnostics()
        with torch.no_grad():
            values = [(self.model(x)-y).abs() for ids in self._dataloader['val_loader'].get_iterator()
                for x, y in [self.pair(ids)]]
        value = float(torch.cat(values).mean())
        return value, value*.1, value+.5


def identity_for(engine):
    frozen, _ = split_state(engine.model.state_dict())
    return dict(schema=train.SCHEMA, design=train.DESIGN, arms=list(train.ARMS), identity_sha256='gw4-fixture',
        engineering_only=False, initializations={arm:train.initialization(engine.model) for arm in train.ARMS},
        training_config=dict(warm_epoch=2, cl_epoch=1), data=dict(splits=dict(train=dict(count=4))),
        reference=dict(source_model_tensor_sha256=train.base.tensor_tree_hash(frozen),
            source_best_metrics=dict(mae=.01, rmse=.01, mape_fraction=.01)))


def cached_fixture():
    old = archive_fixture()
    archive = {key:value for key, value in old.items() if not key.startswith('pred__')}
    archive['schema'] = np.array(evaluation.SCHEMA)
    for index, case in enumerate(evaluation.CASES):
        archive['pred__'+case] = archive['y']+index+1.
    return archive


class ReceiverResidualWorkflowTests(unittest.TestCase):
    def test_adaptation_budget_and_unique_controls(self):
        arguments = ['--run-dir', 'new', '--gw2-run', 'old']
        self.assertEqual(train.parse_args(arguments).epochs, 5)
        for extra in (['--epochs', '0'], ['--epochs', '6'],
                      ['--only', 'MESSAGE_RESIDUAL', 'MESSAGE_RESIDUAL']):
            with self.assertRaises(SystemExit):
                train.parse_args(arguments+extra)

    def test_whole_epoch_resume_matches_continuous_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            whole, split = root/'whole', root/'split'
            whole.mkdir(); split.mkdir()
            identity = identity_for(Engine())
            with patch('builtins.print'):
                full = train.base.run_epochs(Engine(), whole, identity, 'RECEIVER_RESIDUAL', 4)
                train.base.run_epochs(Engine(), split, identity, 'RECEIVER_RESIDUAL', 2)
                saved = train.base.load_checkpoint(split/'last.pt')
                train.validate_saved(saved, identity, 'RECEIVER_RESIDUAL', split, 2)
                resumed = train.base.run_epochs(Engine(19), split, identity, 'RECEIVER_RESIDUAL', 4, saved)
                train.validate_saved(resumed, identity, 'RECEIVER_RESIDUAL', split, 2)
            self.assertEqual(comparable_history(full), comparable_history(resumed))
            for key in ('model_state', 'optimizer_state', 'scheduler_state', 'rng_state', 'train_order'):
                self.assertEqual(train.base.tensor_tree_hash(full[key]), train.base.tensor_tree_hash(resumed[key]), key)

    def test_interrupted_first_epoch_recovers_from_epoch_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            whole, killed = root/'whole', root/'killed'
            whole.mkdir(); killed.mkdir()
            identity = identity_for(Engine())
            with patch('builtins.print'):
                expected = train.base.run_epochs(Engine(), whole, identity, 'RECEIVER_RESIDUAL', 1)
                engine = Engine(); engine.fail_at_update = 1
                with self.assertRaises(KeyboardInterrupt):
                    train.base.run_epochs(engine, killed, identity, 'RECEIVER_RESIDUAL', 1)
                saved = train.base.load_checkpoint(killed/'last.pt')
                self.assertEqual(saved['completed_epoch'], 0)
                train.validate_saved(saved, identity, 'RECEIVER_RESIDUAL', killed, 2)
                recovered = train.base.run_epochs(Engine(99), killed, identity, 'RECEIVER_RESIDUAL', 1, saved)
            self.assertEqual(comparable_history(expected), comparable_history(recovered))
            self.assertEqual(expected['model_tensor_sha256'], recovered['model_tensor_sha256'])

    def test_checkpoint_validation_rejects_frozen_tensor_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            identity = identity_for(Engine())
            with patch('builtins.print'):
                state = train.base.run_epochs(Engine(), path, identity, 'RECEIVER_RESIDUAL', 1)
            broken = deepcopy(state)
            name = next(name for name in broken['model_state'] if '.residual.' not in name)
            broken['model_state'][name].view(-1)[0] += .01
            broken['model_tensor_sha256'] = train.base.tensor_tree_hash(broken['model_state'])
            with self.assertRaisesRegex(ValueError, 'frozen anchor'):
                train.validate_saved(broken, identity, 'RECEIVER_RESIDUAL', path, 2)

    def test_negative_adaptation_results_are_not_hidden_by_stage_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            identity = identity_for(Engine())
            with patch('builtins.print'):
                state = train.base.run_epochs(Engine(), Path(directory), identity, 'RECEIVER_RESIDUAL', 2)
            states = {arm:deepcopy(state) for arm in train.ARMS}
            for arm, saved in states.items():
                saved['arm'] = arm
            for row in states['RECEIVER_RESIDUAL']['history']:
                row['validation_mae'] += 1.
            summary = train.paired_summary(states, identity, 2)
            self.assertEqual(summary['status'], 'GW4_ADAPTATION_COMPLETE')
            self.assertLess(summary['comparisons']['RECEIVER_RESIDUAL_vs_GW2_BASE']['best_adaptation_gain']['mae_gain'], 0.)
            self.assertAlmostEqual(summary['comparisons']['RECEIVER_RESIDUAL_vs_MESSAGE_RESIDUAL']
                ['independent_best_at_common_budget']['mae_gain'], -1.)
            self.assertFalse(summary['stage_zero_in_best_selection'])
            self.assertEqual(summary['baseline_new_inference_calls'], 0)

    def test_source_cache_selection_always_uses_anchor_best(self):
        saved = dict(gw2_reference={'archives':{'best':'best', 'last':'last'}})
        with patch.object(reference, 'load_gw2_cached', return_value=({}, {})) as load:
            reference.load_cached(saved)
        load.assert_called_once_with(saved['gw2_reference'], 'best')

    def test_best_last_identical_tensor_cache_never_builds_a_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root/'eval_best'; output.mkdir()
            archive = cached_fixture()
            path = output/'paired_predictions.npz'
            np.savez(path, **archive)
            metadata = dict(schema='gw4_receiver_residual_producer_v1', status='GW4_PREDICTION_PASS',
                run_identity_sha256='run', reference={}, archive_sha256=evaluation.metrics.sha256(path),
                checkpoint_roles={'RECEIVER_RESIDUAL':dict(selection='best', model_tensor_sha256='same')},
                validation_replay={'RECEIVER_RESIDUAL':dict(passed=True)},
                producer_source={name:evaluation.metrics.sha256(evaluation.REPO/'experiments/chronological'/name)
                    for name in evaluation.PRODUCERS})
            (output/'metadata.json').write_text(json.dumps(metadata), encoding='utf-8')
            identity = dict(identity_sha256='run', reference={})
            role = dict(selection='last', model_tensor_sha256='same')
            with patch.object(train, 'model_for_arm', side_effect=AssertionError('No repeated model forward')):
                self.assertIsNotNone(evaluation.existing_cache(root, identity, 'RECEIVER_RESIDUAL', role, archive))
            changed = deepcopy(archive)
            changed['pred__GW2_BASE'][0, 0, 0] += 1.
            with self.assertRaises(ValueError):
                evaluation.existing_cache(root, identity, 'RECEIVER_RESIDUAL', role, changed)

    def test_report_keeps_shared_masks_and_separate_hard_sets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = cached_fixture()
            result = evaluation.report(archive, root/'report', dict(engineering_only=True))
            self.assertEqual(result['status'], 'GW4_EVALUATION_COMPLETE')
            self.assertEqual(len(result['cases']), 8)
            self.assertLess(result['comparisons']['RECEIVER_RESIDUAL_vs_GW2_BASE']['mae_gain_pooled'], 0.)
            self.assertIn('GW2_fixed_hard_ids', result)
            self.assertIn('BASE 直接读取认证历史缓存', (root/'report'/'report.md').read_text(encoding='utf-8'))
            invalid = deepcopy(archive); invalid['valid'][0, 0, 0] = True
            with self.assertRaises(ValueError):
                evaluation.validate_archive(invalid)


if __name__ == '__main__':
    unittest.main()
