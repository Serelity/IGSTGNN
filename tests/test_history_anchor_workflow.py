"""Server-run GW5 budget, epoch recovery, source and cache contracts."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.chronological import train_original_history_anchor as train
from experiments.chronological import evaluate_original_history_anchor as evaluation
from experiments.chronological import history_anchor_reference as reference
from experiments.chronological.original_history_anchor import HistoryAnchorIGSTGNN
from experiments.chronological.original_receiver_residual import split_state
from tests.test_receiver_residual_workflow import Engine as PreviousEngine
from tests.test_original_isp_runner import comparable_history
from tests.test_original_graph_window_evaluation import fixture as old_archive_fixture
from tests.test_history_anchor import fixture


class Engine(PreviousEngine):
    def __init__(self, seed=2025):
        super().__init__(seed)
        anchor, self.x, self.target = fixture()
        self.model = HistoryAnchorIGSTGNN(anchor, 'HISTORY_ANCHORED')
        self._optimizer = torch.optim.Adam([p for p in self.model.parameters() if p.requires_grad],
            lr=.002, weight_decay=1e-5)
        self._lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self._optimizer, milestones=[1, 3], gamma=.5)
        train.base.configure_runtime('cpu', 1, seed)


def identity_for(engine):
    frozen, _ = split_state(engine.model.state_dict())
    return dict(schema=train.SCHEMA, design=train.DESIGN, arms=list(train.ARMS), identity_sha256='gw5-fixture',
        engineering_only=False, initializations={arm:train.initialization(engine.model) for arm in train.ARMS},
        training_config=dict(warm_epoch=2, cl_epoch=1), data=dict(splits=dict(train=dict(count=4))),
        reference=dict(source_model_tensor_sha256=train.base.tensor_tree_hash(frozen),
            source_best_metrics=dict(mae=.01, rmse=.01, mape_fraction=.01)),
        gw4_reference=dict(completed_adaptation_epochs=5))


def archive_fixture():
    old = old_archive_fixture()
    archive = {key:value for key, value in old.items() if not key.startswith('pred__')}
    archive['schema'] = np.array(evaluation.SCHEMA)
    for index, case in enumerate(evaluation.CASES):
        archive['pred__'+case] = archive['y']+index+1.
    return archive


class HistoryAnchorWorkflowTests(unittest.TestCase):
    def test_parser_requires_GW4_and_total_bounded_unique_budget(self):
        arguments = ['--run-dir', 'new', '--gw4-run', 'old']
        self.assertEqual(train.parse_args(arguments).epochs, 5)
        for extra in (['--epochs', '0'], ['--epochs', '6'], ['--only', 'HISTORY_FIXED', 'HISTORY_FIXED']):
            with self.assertRaises(SystemExit):
                train.parse_args(arguments+extra)

    def test_full_epoch_resume_matches_optimizer_rng_order_and_curriculum(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            whole, split = root/'whole', root/'split'
            whole.mkdir(); split.mkdir()
            identity = identity_for(Engine())
            with patch('builtins.print'):
                full = train.base.run_epochs(Engine(), whole, identity, 'HISTORY_ANCHORED', 4)
                train.base.run_epochs(Engine(), split, identity, 'HISTORY_ANCHORED', 2)
                saved = train.base.load_checkpoint(split/'last.pt')
                train.validate_saved(saved, identity, 'HISTORY_ANCHORED', split, 2)
                resumed = train.base.run_epochs(Engine(19), split, identity, 'HISTORY_ANCHORED', 4, saved)
                train.validate_saved(resumed, identity, 'HISTORY_ANCHORED', split, 2)
            self.assertEqual(comparable_history(full), comparable_history(resumed))
            for key in ('model_state', 'optimizer_state', 'scheduler_state', 'rng_state', 'train_order'):
                self.assertEqual(train.base.tensor_tree_hash(full[key]), train.base.tensor_tree_hash(resumed[key]), key)

    def test_interrupted_first_epoch_replays_from_zero_with_no_cached_history(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            whole, killed = root/'whole', root/'killed'
            whole.mkdir(); killed.mkdir()
            identity = identity_for(Engine())
            with patch('builtins.print'):
                expected = train.base.run_epochs(Engine(), whole, identity, 'HISTORY_ANCHORED', 1)
                engine = Engine(); engine.fail_at_update = 1
                with self.assertRaises(KeyboardInterrupt):
                    train.base.run_epochs(engine, killed, identity, 'HISTORY_ANCHORED', 1)
                self.assertFalse(engine.model.selector.residual._active)
                self.assertEqual(engine.model.selector.residual._anchors, {})
                saved = train.base.load_checkpoint(killed/'last.pt')
                self.assertEqual(saved['completed_epoch'], 0)
                train.validate_saved(saved, identity, 'HISTORY_ANCHORED', killed, 2)
                resumed = train.base.run_epochs(Engine(99), killed, identity, 'HISTORY_ANCHORED', 1, saved)
            self.assertEqual(comparable_history(expected), comparable_history(resumed))
            self.assertEqual(expected['model_tensor_sha256'], resumed['model_tensor_sha256'])

    def test_frozen_tensor_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            identity = identity_for(Engine())
            with patch('builtins.print'):
                state = train.base.run_epochs(Engine(), root, identity, 'HISTORY_ANCHORED', 1)
            bad = deepcopy(state)
            key = next(key for key in bad['model_state'] if '.residual.' not in key)
            bad['model_state'][key].view(-1)[0] += .1
            bad['model_tensor_sha256'] = train.base.tensor_tree_hash(bad['model_state'])
            with self.assertRaisesRegex(ValueError, 'frozen anchor'):
                train.validate_saved(bad, identity, 'HISTORY_ANCHORED', root, 2)

    def test_short_new_budget_is_not_paired_to_old_five_epoch_control(self):
        with tempfile.TemporaryDirectory() as directory:
            identity = identity_for(Engine())
            with patch('builtins.print'):
                state = train.base.run_epochs(Engine(), Path(directory), identity, 'HISTORY_ANCHORED', 2)
            states = {arm:deepcopy(state) for arm in train.ARMS}
            for arm, saved in states.items():
                saved['arm'] = arm
            old = {arm:dict(completed_epoch=5, history=state['history']*3) for arm in train.gw4.ARMS}
            summary = train.paired_summary(states, identity, 2, old)
            self.assertFalse(summary['GW4_control_budget_matched'])
            self.assertNotIn('HISTORY_ANCHORED_vs_RECEIVER_RESIDUAL', summary['comparisons'])
            self.assertIn('HISTORY_ANCHORED_vs_HISTORY_FIXED', summary['comparisons'])

    def test_GW4_certificate_rejects_role_budget_producer_and_replay_changes(self):
        identity = dict(identity_sha256='old', reference={'source':'GW2'})
        states = {arm:dict(best_epoch=3, completed_epoch=5, best_checkpoint_sha256=arm+'best',
            _last_file_sha256=arm+'last', _best_model_sha256=arm+'weights', _last_model_sha256=arm+'lastweights')
            for arm in train.gw4.ARMS}
        metadata = dict(schema='gw4_receiver_residual_producer_v1', status='GW4_PREDICTION_PASS',
            run_identity_sha256='old', reference=identity['reference'], common_completed_adaptation_epochs=5,
            engineering_only=False, old_parameters_frozen=True,
            checkpoint_roles={arm:dict(selection='best', epoch=3, checkpoint_sha256=arm+'best',
                model_tensor_sha256=arm+'weights') for arm in states},
            validation_replay={arm:dict(passed=True) for arm in states},
            producer_source={name:train.base.sha256(train.REPO/'experiments/chronological'/name)
                for name in reference.evaluation.PRODUCERS})
        reference.validate_certificate(metadata, identity, states, 'best', 5)
        changes = (lambda m:m.update(common_completed_adaptation_epochs=4),
            lambda m:m['checkpoint_roles']['RECEIVER_RESIDUAL'].update(selection='last'),
            lambda m:m['validation_replay']['MESSAGE_RESIDUAL'].update(passed=False),
            lambda m:m['producer_source'].update({'evaluate_original_receiver_residual.py':'bad'}))
        for change in changes:
            broken = deepcopy(metadata); change(broken)
            with self.assertRaises(ValueError):
                reference.validate_certificate(broken, identity, states, 'best', 5)

    def test_cache_reuses_only_identical_policy_without_repeating_model_forward(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root/'eval_best'; output.mkdir()
            archive = archive_fixture()
            path = output/'paired_predictions.npz'; np.savez(path, **archive)
            metadata = dict(schema='gw5_history_anchor_producer_v1', status='GW5_PREDICTION_PASS',
                run_identity_sha256='run', gw4_reference={}, archive_sha256=evaluation.metrics.sha256(path),
                checkpoint_roles={'HISTORY_ANCHORED':dict(policy='HISTORY_ANCHORED', model_tensor_sha256='same')},
                validation_replay={'HISTORY_ANCHORED':dict(passed=True)},
                producer_source={name:evaluation.metrics.sha256(evaluation.REPO/'experiments/chronological'/name)
                    for name in evaluation.PRODUCERS})
            metadata_path = output/'metadata.json'
            metadata_path.write_text(json.dumps(metadata), encoding='utf-8')
            identity = dict(identity_sha256='run', gw4_reference={})
            role = dict(policy='HISTORY_ANCHORED', model_tensor_sha256='same')
            with patch.object(train, 'model_for_arm', side_effect=AssertionError('No repeated forward')):
                self.assertIsNotNone(evaluation.existing_cache(root, identity, 'HISTORY_ANCHORED', role, archive))
                changed_control = deepcopy(archive)
                changed_control['pred__RECEIVER_RESIDUAL'] += .5
                self.assertIsNotNone(evaluation.existing_cache(root, identity, 'HISTORY_ANCHORED', role, changed_control))
            metadata['checkpoint_roles']['HISTORY_ANCHORED']['policy'] = 'HISTORY_FIXED'
            metadata_path.write_text(json.dumps(metadata), encoding='utf-8')
            self.assertIsNone(evaluation.existing_cache(root, identity, 'HISTORY_ANCHORED', role, archive))

    def test_report_retains_negative_gains_and_separate_hard_sets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = archive_fixture()
            result = evaluation.report(archive, root/'report', dict(engineering_only=True))
            self.assertEqual(result['status'], 'GW5_EVALUATION_COMPLETE')
            self.assertEqual(len(result['cases']), 14)
            self.assertLess(result['comparisons']['HISTORY_ANCHORED_vs_RECEIVER_RESIDUAL']['mae_gain_pooled'], 0.)
            self.assertEqual(set(result['additional_fixed_hard_ids']), {'GW2_BASE', 'RECEIVER_RESIDUAL'})
            self.assertIn('不等于旧 GW4', (root/'report'/'report.md').read_text(encoding='utf-8'))
            invalid = deepcopy(archive); invalid['valid'][0, 0, 0] = True
            with self.assertRaises(ValueError):
                evaluation.validate_archive(invalid)


if __name__ == '__main__':
    unittest.main()
