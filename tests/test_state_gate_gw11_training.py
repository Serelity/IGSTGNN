"""Server-only GW11 continuation bookkeeping and source-integrity tests.

No author arrays or native model forward are needed. Execute on the server;
the local workstation must not run imports, syntax checks or these tests.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from torch import nn

from experiments.state_gate_gw11 import state, train
from experiments.state_gate_gw11.protocol import ARMS, EPOCHS, GROUPS


SEED = 2025
TRAIN_COUNT = 3909
BATCHES = 82


def metric_fixture(mae=10.):
    return dict(valid_count=120, mae=float(mae), rmse=float(mae)+1.,
        mape_fraction=.2, signed_bias=-.1,
        groups={name: dict(valid_count=30, mae=float(mae)+2., signed_bias=-.2)
            for name in GROUPS}, window_p95=float(mae)+3.)


def identity_fixture():
    initial = {}
    for arm in ARMS:
        added = 0 if arm == 'SUM_CONTINUE' else 725
        initial[arm] = dict(source_tensor_sha256='a'*64,
            added_tensor_sha256=('b' if added else 'c')*64,
            registered_parameters=10000+added, trainable_parameters=10000+added,
            added_trainable_parameters=added)
    result = dict(schema=state.SCHEMA, arms=list(ARMS), candidate='STATE_GATE',
        new_epochs=EPOCHS, seed=SEED, train_count=TRAIN_COUNT, train_batches=BATCHES,
        initializations=initial, full_horizon=12, extension_allowed=False,
        same_source='certified_GW9_SUM_CONTROL_BEST_including_trained_SUM_head')
    result['identity_sha256'] = state.identity_digest(result)
    return result


def saved_fixture(identity, arm='STATE_GATE', completed=3, scores=None, best_epoch=None):
    # This checkpoint fixture independently spells out the locked experiment
    # order/RNG contract, rather than invoking the helper under validation.
    scores = [9., 9., 9.5] if scores is None else list(scores)
    rows = []
    for epoch in range(1, completed+1):
        order = np.random.RandomState(SEED+epoch).permutation(TRAIN_COUNT)
        rows.append(dict(epoch=epoch, global_updates=epoch*BATCHES,
            validation=metric_fixture(scores[epoch-1]),
            train_order_sha256=hashlib.sha256(order.astype('<i8').tobytes()).hexdigest(),
            dropout_epoch_seed=SEED+10000*epoch,
            mean_batch_train_mae=12.-epoch))
    if best_epoch is None:
        best_epoch = 0 if completed == 0 else 1
    best_mae = 10. if best_epoch == 0 else scores[best_epoch-1]
    return dict(schema=state.SCHEMA, arm=arm,
        identity_sha256=identity['identity_sha256'], initialization=deepcopy(identity['initializations'][arm]),
        completed_epoch=completed, global_updates=completed*BATCHES,
        stage_zero=metric_fixture(10.), history=rows,
        best=dict(epoch=best_epoch, mae=float(best_mae), checkpoint_file='best_fixture.pt',
            checkpoint_sha256='d'*64, model_tensor_sha256='e'*64),
        full_horizon=12, model_state={}, optimizer_state={}, scheduler_state={'last_epoch': completed})


class TinyCompleteSum(nn.Module):
    """A body and already fitted SUM head, without a forward implementation."""
    def __init__(self):
        super().__init__()
        self.body = nn.Parameter(torch.tensor([2., 3.]))
        self.head = nn.Linear(2, 1)


class TinyJointWrapper(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.source = source
        self.adapters = nn.ParameterList([nn.Parameter(torch.zeros(725))])


class GW11TrainingStateTests(unittest.TestCase):
    def test_fixed_orders_are_reproducible_permutations_without_global_rng_consumption(self):
        previous = np.random.get_state()
        try:
            np.random.seed(1911)
            before = np.random.get_state()
            first = state.epoch_order(SEED, 1, 3909)
            second = state.epoch_order(SEED, 2, 3909)
            again = state.epoch_order(SEED, 1, 3909)
            after = np.random.get_state()
            np.testing.assert_array_equal(first, again)
            np.testing.assert_array_equal(np.sort(first), np.arange(3909))
            self.assertFalse(np.array_equal(first, second))
            self.assertEqual(before[0], after[0])
            np.testing.assert_array_equal(before[1], after[1])
            self.assertEqual(before[2:], after[2:])
            # A small known NumPy RandomState(0) permutation anchors the rule.
            np.testing.assert_array_equal(state.epoch_order(-1, 1, 5), [2, 0, 1, 3, 4])
            self.assertEqual([state.epoch_seed(SEED, epoch) for epoch in (1, 2, 3)],
                [12025, 22025, 32025])
        finally:
            np.random.set_state(previous)

    def test_partial_paired_resume_skips_committed_arms_and_replays_only_uncommitted_work(self):
        identity = identity_fixture()
        states = {arm: saved_fixture(identity, arm, completed=(1 if arm in ARMS[:2] else 0))
            for arm in ARMS}
        for arm, saved in states.items(): state.validate_progress(saved, identity, arm, BATCHES)
        pending = state.next_work(states, 1)
        self.assertEqual(pending, ['STATE_READOUT', 'STATE_GATE'])
        self.assertNotIn('SUM_CONTINUE', pending)
        self.assertNotIn('STATIC_GATE', pending)
        # A killed partial READOUT epoch never committed a new saved epoch.
        self.assertEqual(state.next_work(deepcopy(states), 1), pending)
        for arm in pending: states[arm] = saved_fixture(identity, arm, completed=1)
        self.assertEqual(state.next_work(states, 1), [])
        self.assertEqual(state.next_work(states, 2), list(ARMS))

    def test_all_three_completed_epochs_have_no_more_work_and_cannot_extend(self):
        identity = identity_fixture()
        states = {arm: saved_fixture(identity, arm, completed=3) for arm in ARMS}
        for epoch in (1, 2, 3): self.assertEqual(state.next_work(states, epoch), [])
        with self.assertRaises(ValueError): state.next_work(states, 4)
        for epoch in (0, 4, True):
            with self.subTest(epoch=epoch), self.assertRaises(ValueError): state.epoch_order(SEED, epoch, 17)
            with self.subTest(epoch=epoch), self.assertRaises(ValueError): state.epoch_seed(SEED, epoch)
        with self.assertRaises(ValueError): state.epoch_order(SEED, 1, 0)

    def test_stage_zero_best_and_first_ties_are_retained(self):
        identity = identity_fixture()
        tied = saved_fixture(identity, scores=[10., 10., 10.], best_epoch=0)
        self.assertEqual(state.validate_progress(tied, identity, 'STATE_GATE', BATCHES), 3)
        later_tie = deepcopy(tied); later_tie['best'].update(epoch=3, mae=10.)
        with self.assertRaisesRegex(ValueError, 'first minimum'):
            state.validate_progress(later_tie, identity, 'STATE_GATE', BATCHES)
        improved = saved_fixture(identity, scores=[9., 9., 9.5], best_epoch=1)
        self.assertEqual(state.validate_progress(improved, identity, 'STATE_GATE', BATCHES), 3)
        improved['best']['epoch'] = 2
        with self.assertRaisesRegex(ValueError, 'first minimum'):
            state.validate_progress(improved, identity, 'STATE_GATE', BATCHES)

    def test_identity_arm_initialization_and_budget_corruption_are_rejected(self):
        identity = identity_fixture(); saved = saved_fixture(identity)
        self.assertEqual(state.validate_progress(saved, identity, 'STATE_GATE', BATCHES), 3)
        mutations = {
            'identity': lambda item: item.update(identity_sha256='0'*64),
            'arm': lambda item: item.update(arm='STATIC_GATE'),
            'initial_source': lambda item: item['initialization'].update(source_tensor_sha256='0'*64),
            'initial_added_capacity': lambda item: item['initialization'].update(added_trainable_parameters=726),
            'updates': lambda item: item.update(global_updates=7),
            'updates_float': lambda item: item.update(global_updates=246.),
            'history_updates': lambda item: item['history'][1].update(global_updates=5),
            'order_hash': lambda item: item['history'][1].update(train_order_sha256='0'*64),
            'dropout_seed': lambda item: item['history'][1].update(dropout_epoch_seed=SEED),
            'duplicate_epoch': lambda item: item['history'][1].update(epoch=1),
            'missing_epoch': lambda item: item['history'].pop(1),
            'invalid_score': lambda item: item['history'][1]['validation'].update(mae=float('nan')),
            'extended_budget': lambda item: item.update(completed_epoch=4),
        }
        for name, mutation in mutations.items():
            changed = deepcopy(saved); mutation(changed)
            with self.subTest(name=name), self.assertRaises(ValueError):
                state.validate_progress(changed, identity, 'STATE_GATE', BATCHES)
        changed_identity = deepcopy(identity); changed_identity['seed'] = 2026
        with self.assertRaises(ValueError): state.validate_progress(saved, changed_identity, 'STATE_GATE', BATCHES)
        with self.assertRaises(ValueError): state.validate_progress(saved, identity, 'UNREGISTERED', BATCHES)

    def test_recomputed_identity_cannot_change_fixed_budget_arms_or_batch_partition(self):
        identity = identity_fixture()
        for key, value in (('new_epochs', 4), ('arms', list(reversed(ARMS)))):
            changed = deepcopy(identity); changed[key] = value
            changed['identity_sha256'] = state.identity_digest(changed)
            saved = saved_fixture(changed)
            with self.subTest(key=key), self.assertRaises(ValueError):
                state.validate_progress(saved, changed, 'STATE_GATE', BATCHES)
        saved = saved_fixture(identity)
        with self.assertRaises(ValueError): state.validate_progress(saved, identity, 'STATE_GATE', BATCHES+1)

    def test_resume_requires_all_arms_and_prior_epochs(self):
        identity = identity_fixture()
        states = {arm: saved_fixture(identity, arm, completed=0) for arm in ARMS}
        with self.assertRaisesRegex(ValueError, 'preceding epoch'): state.next_work(states, 2)
        del states['STATE_GATE']
        with self.assertRaisesRegex(ValueError, 'All registered'): state.next_work(states, 1)

    def test_incompatible_arm_checkpoints_cannot_be_combined_into_a_paired_resume(self):
        identity = identity_fixture()
        for progress in ((3,0,0,0),(0,1,0,0),(2,1,0,0)):
            states = {arm:saved_fixture(identity,arm,completed=done)
                for arm,done in zip(ARMS,progress)}
            with self.subTest(progress=progress), self.assertRaisesRegex(ValueError,'paired epoch order'):
                state.next_work(states,1)


class GW11TrainingSourceTests(unittest.TestCase):
    def test_factory_keeps_joint_source_and_strictly_loads_trained_sum_head(self):
        source = TinyCompleteSum()
        fitted_head = dict(weight=torch.tensor([[2., -3.]]), bias=torch.tensor([7.]))
        data, modules, gw4, recurrent = object(), object(), object(), object()
        wrapper = MagicMock(); wrapper.to.return_value = wrapper
        with patch.object(train.gw9, 'model_for_arm', return_value=source) as source_factory, \
                patch.object(train, 'StateGateIGSTGNN', return_value=wrapper) as wrapper_factory:
            actual = train.model_for_arm(data, modules, 'cpu', 'STATE_GATE', gw4, recurrent, fitted_head)
        source_factory.assert_called_once_with(data, modules, 'cpu', 'SUM_CONTROL', gw4, recurrent,
            hidden=16, stage='joint')
        wrapper_factory.assert_called_once_with(source, 'STATE_GATE', hidden=16)
        wrapper.to.assert_called_once_with('cpu')
        self.assertIs(actual, wrapper)
        for name, value in source.head.state_dict().items(): torch.testing.assert_close(value, fitted_head[name])
        self.assertTrue(all(parameter.requires_grad for parameter in source.parameters()))

    def test_factory_cannot_silently_use_an_absent_or_partial_sum_head(self):
        for bad_head in (None, {'weight': torch.ones(1, 2)}):
            with self.subTest(head=bad_head), \
                    patch.object(train.gw9, 'model_for_arm', return_value=TinyCompleteSum()), \
                    patch.object(train, 'StateGateIGSTGNN') as wrapper_factory:
                with self.assertRaises((TypeError, RuntimeError)):
                    train.model_for_arm(object(), object(), 'cpu', 'STATE_GATE', object(), object(), bad_head)
                wrapper_factory.assert_not_called()

    def test_initialization_hash_includes_the_complete_source_head_and_exact_added_capacity(self):
        model = TinyJointWrapper(TinyCompleteSum())
        baseline = train.initialization(model)
        self.assertEqual(baseline['added_trainable_parameters'], 725)
        self.assertEqual(baseline['registered_parameters'], baseline['trainable_parameters'])
        changed_head = deepcopy(model)
        with torch.no_grad(): changed_head.source.head.bias.add_(1.)
        changed = train.initialization(changed_head)
        self.assertNotEqual(changed['source_tensor_sha256'], baseline['source_tensor_sha256'])
        self.assertEqual(changed['added_tensor_sha256'], baseline['added_tensor_sha256'])
        changed_adapter = deepcopy(model)
        with torch.no_grad(): changed_adapter.adapters[0].add_(1.)
        changed = train.initialization(changed_adapter)
        self.assertEqual(changed['source_tensor_sha256'], baseline['source_tensor_sha256'])
        self.assertNotEqual(changed['added_tensor_sha256'], baseline['added_tensor_sha256'])
        replicas = [train.initialization(deepcopy(model)) for _ in ARMS[1:]]
        self.assertEqual(len({row['source_tensor_sha256'] for row in replicas}), 1)
        self.assertEqual(len({row['added_tensor_sha256'] for row in replicas}), 1)

    def test_committed_cpu_tree_owns_optimizer_rng_and_nested_metadata(self):
        tensor = torch.tensor([1., 2.], requires_grad=True)
        array = np.array([3, 4])
        live = dict(optimizer=dict(state={0: dict(exp_avg=tensor)}),
            rng=(array, torch.tensor([5, 6])), metadata=dict(tags=['original']))
        snapshot = train.cpu_tree(live)
        with torch.no_grad(): tensor.add_(100.)
        array[:] = 0
        live['rng'][1].zero_()
        live['metadata']['tags'].append('later')
        torch.testing.assert_close(snapshot['optimizer']['state'][0]['exp_avg'], torch.tensor([1., 2.]))
        self.assertFalse(snapshot['optimizer']['state'][0]['exp_avg'].requires_grad)
        self.assertEqual(snapshot['optimizer']['state'][0]['exp_avg'].device.type, 'cpu')
        np.testing.assert_array_equal(snapshot['rng'][0], [3, 4])
        torch.testing.assert_close(snapshot['rng'][1], torch.tensor([5, 6]))
        self.assertEqual(snapshot['metadata']['tags'], ['original'])

    def test_record_keeps_empty_guard_groups_unavailable_and_excludes_empty_windows(self):
        truth = np.full((2, 12, 3), 20., dtype=np.float32)
        prediction = truth+2.; prediction[1] += 1000.
        valid = np.ones(truth.shape, dtype=np.bool_); valid[1] = False
        row = train.record(prediction, truth, valid)
        self.assertEqual(row['valid_count'], 36)
        self.assertEqual(row['mae'], 2.)
        self.assertEqual(row['window_p95'], 2.)
        for name in ('high_value_ge314', 'flow_50_200'):
            self.assertEqual(row['groups'][name]['valid_count'], 0)
            self.assertIsNone(row['groups'][name]['mae'])
            self.assertIsNone(row['groups'][name]['signed_bias'])
        self.assertEqual(row['groups']['long_horizon_7_12']['valid_count'], 18)
        self.assertTrue(train.same_record(row, deepcopy(row)))
        changed = deepcopy(row); changed['groups']['high_value_ge314']['mae'] = 0.
        self.assertFalse(train.same_record(row, changed))
        changed = deepcopy(row); changed['groups']['long_horizon_7_12']['mae'] += .1
        self.assertFalse(train.same_record(row, changed))

    def test_parser_requires_sources_and_only_fixed_budget_and_resume_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            required = []
            for name in ('probe-dir', 'cache-dir', 'gw10-dir', 'audit-dir', 'data-dir', 'run-dir'):
                required.extend(('--'+name, str(Path(directory)/name)))
            args = train.parse_args(required)
            self.assertEqual(args.epochs, 3)
            self.assertEqual(args.seed, 2025)
            self.assertFalse(args.resume or args.evaluate_only)
            self.assertTrue(args.probe_dir.is_absolute())
            for invalid in (['--epochs', '4'], ['--seed', '2026'], ['--threads', '0'],
                    ['--resume', '--evaluate-only'], ['--candidate', 'STATIC_GATE']):
                with self.subTest(invalid=invalid), self.assertRaises(SystemExit):
                    train.parse_args(required+invalid)
            with self.assertRaises(SystemExit): train.parse_args(required[2:])
            with self.assertRaises(SystemExit): train.parse_args([])


class GW11TrainingStartMarkerTests(unittest.TestCase):
    def checkpoints(self, directory, identity, progress=(0,0,0,0), missing=()):
        states = {}
        for arm,done in zip(ARMS,progress):
            saved = saved_fixture(identity,arm,completed=done)
            states[arm] = saved
            child = Path(directory)/arm; child.mkdir(exist_ok=True)
            if arm not in missing:
                torch.save(saved,child/'last.pt')
        return states

    def test_partial_stage_zero_without_any_training_evidence_can_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            identity = identity_fixture()
            self.checkpoints(directory,identity,missing=ARMS[2:])
            result = train.validate_resume_checkpoint_set(directory,identity)
            self.assertFalse(result['training_evidence_detected'])
            self.assertEqual(result['missing_arms'],list(ARMS[2:]))
            self.assertFalse((Path(directory)/'training_started.json').exists())

    def test_marker_is_bound_persisted_and_immutable_before_first_update(self):
        with tempfile.TemporaryDirectory() as directory:
            identity = identity_fixture(); states = self.checkpoints(directory,identity)
            marker = train.ensure_training_started(directory,identity,states)
            path = Path(directory)/'training_started.json'; original = path.read_bytes()
            self.assertEqual(marker['schema'],train.TRAINING_MARKER_SCHEMA)
            self.assertEqual(marker['training_schema'],state.SCHEMA)
            self.assertEqual(marker['identity_sha256'],identity['identity_sha256'])
            self.assertTrue(marker['written_before_first_optimizer_update'])
            self.assertEqual(train.ensure_training_started(directory,identity,states),marker)
            self.assertEqual(path.read_bytes(),original)
            self.assertTrue(train.validate_resume_checkpoint_set(directory,identity)['marker_present'])

    def test_marker_rejects_a_missing_checkpoint_even_if_all_survivors_are_epoch_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            identity = identity_fixture(); states = self.checkpoints(directory,identity)
            train.ensure_training_started(directory,identity,states)
            (Path(directory)/ARMS[0]/'last.pt').unlink()
            with self.assertRaisesRegex(ValueError,'missing an arm checkpoint'):
                train.validate_resume_checkpoint_set(directory,identity)

    def test_marker_rejects_loss_of_all_four_last_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            identity = identity_fixture(); states = self.checkpoints(directory,identity)
            train.ensure_training_started(directory,identity,states)
            for arm in ARMS: (Path(directory)/arm/'last.pt').unlink()
            with self.assertRaisesRegex(ValueError,'missing an arm checkpoint'):
                train.validate_resume_checkpoint_set(directory,identity)

    def test_summary_or_history_protects_a_lost_only_trained_checkpoint_without_marker(self):
        for ledger in ('summary','history'):
            with self.subTest(ledger=ledger),tempfile.TemporaryDirectory() as directory:
                identity = identity_fixture()
                states = self.checkpoints(directory,identity,progress=(1,0,0,0),missing=(ARMS[0],))
                if ledger=='summary':
                    train.base.write_json(Path(directory)/'summary.json',train.summary(states,identity))
                else:
                    train.base.write_json(Path(directory)/ARMS[0]/'history.json',states[ARMS[0]]['history'])
                with self.assertRaisesRegex(ValueError,'missing an arm checkpoint'):
                    train.validate_resume_checkpoint_set(directory,identity)

    def test_positive_surviving_checkpoint_still_requires_all_four_states(self):
        with tempfile.TemporaryDirectory() as directory:
            identity = identity_fixture()
            self.checkpoints(directory,identity,progress=(1,0,0,0),missing=(ARMS[-1],))
            with self.assertRaisesRegex(ValueError,'missing an arm checkpoint'):
                train.validate_resume_checkpoint_set(directory,identity)

    def test_tampered_marker_is_rejected_even_when_all_checkpoints_exist(self):
        mutations = dict(identity_sha256='0'*64,schema='wrong-schema',
            training_schema='wrong-run',registered_arms=list(reversed(ARMS)),
            written_before_first_optimizer_update=False,unexpected_extra=True)
        for key,value in mutations.items():
            with self.subTest(key=key),tempfile.TemporaryDirectory() as directory:
                identity = identity_fixture(); states = self.checkpoints(directory,identity)
                marker = train.ensure_training_started(directory,identity,states)
                marker[key] = value
                (Path(directory)/'training_started.json').write_text(json.dumps(marker),encoding='utf-8')
                with self.assertRaisesRegex(ValueError,'marker differs'):
                    train.validate_resume_checkpoint_set(directory,identity)
                with self.assertRaisesRegex(ValueError,'marker differs'):
                    train.ensure_training_started(directory,identity,states)

    def test_new_preupdate_marker_cannot_be_fabricated_for_an_already_trained_run(self):
        with tempfile.TemporaryDirectory() as directory:
            identity = identity_fixture(); states = self.checkpoints(directory,identity,progress=(1,0,0,0))
            with self.assertRaisesRegex(ValueError,'original training-start marker'):
                train.ensure_training_started(directory,identity,states)
            self.assertFalse((Path(directory)/'training_started.json').exists())

    def test_initialization_cannot_mark_started_until_all_stage_zero_saves_exist(self):
        with tempfile.TemporaryDirectory() as directory:
            identity = identity_fixture(); states = self.checkpoints(directory,identity,missing=(ARMS[-1],))
            with self.assertRaisesRegex(ValueError,'All four stage-zero checkpoints'):
                train.ensure_training_started(directory,identity,states)
            self.assertFalse((Path(directory)/'training_started.json').exists())


if __name__ == '__main__': unittest.main()
