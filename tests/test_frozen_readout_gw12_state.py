"""Server-only tests for pure GW12 atomic paired checkpoint bookkeeping.

Do not execute Python, imports, syntax checks or tests on the workstation.
No Torch, native model, cached author arrays or GPU is needed by these tests.
"""
from copy import deepcopy
import hashlib
import unittest

import numpy as np

from experiments.frozen_readout_gw12 import state
from experiments.frozen_readout_gw12.protocol import (
    ARMS, CANDIDATE, CASES, EPOCHS, GROUPS, REFERENCE, SEED,
    TRAIN_BATCHES, TRAIN_COUNT)


def record(mae=10.):
    return dict(mae=float(mae),rmse=15.,mape_fraction=.2,signed_bias=-.1,
        window_p95=18.,valid_count=100,groups={name:dict(
            valid_count=20,mae=20.,signed_bias=-3.) for name in GROUPS})


def identity_fixture():
    result = dict(schema=state.SCHEMA,arms=list(ARMS),candidate=CANDIDATE,
        new_epochs=EPOCHS,train_batches=TRAIN_BATCHES,train_count=TRAIN_COUNT,
        seed=SEED,source_tensor_sha256='a'*64,initializations={arm:dict(
            head_tensor_sha256='b'*64,added_trainable_parameters=725) for arm in ARMS},
        full_horizon=12,extension_allowed=False)
    result['identity_sha256'] = state.identity_digest(result)
    return result


def saved_fixture(identity,completed=3,scores=None):
    scores = {arm:[9.,9.,9.5] for arm in ARMS} if scores is None else scores
    stage_zero = {arm:record() for arm in CASES}
    history = []
    for epoch in range(1,completed+1):
        # Independently spell out the paired order/RNG contract under test.
        order = np.random.RandomState(SEED+epoch).permutation(TRAIN_COUNT)
        validation = {REFERENCE:record()}
        validation.update({arm:record(scores[arm][epoch-1]) for arm in ARMS})
        history.append(dict(epoch=epoch,global_updates=epoch*TRAIN_BATCHES,
            head_updates=dict.fromkeys(ARMS,epoch*TRAIN_BATCHES),validation=validation,
            train_order_sha256=hashlib.sha256(order.astype('<i8').tobytes()).hexdigest(),
            dropout_epoch_seed=SEED+10000*epoch,train_metrics={arm:dict(
                mean_batch_train_mae=12.-epoch,lr_used=.002,lr_next=.001) for arm in ARMS}))
    best = {}
    for arm in ARMS:
        rows = [(0,stage_zero[arm])]+[(row['epoch'],row['validation'][arm]) for row in history]
        epoch,metric = min(rows,key=lambda pair:pair[1]['mae'])
        best[arm] = dict(epoch=epoch,mae=metric['mae'],checkpoint_file='best_'+arm+'.pt',
            checkpoint_sha256='c'*64,head_tensor_sha256='d'*64)
    return dict(schema=state.SCHEMA,identity_sha256=identity['identity_sha256'],
        completed_epoch=completed,global_updates=completed*TRAIN_BATCHES,
        head_updates=dict.fromkeys(ARMS,completed*TRAIN_BATCHES),history=history,
        initializations=deepcopy(identity['initializations']),source_tensor_sha256='a'*64,
        stage_zero_validation=stage_zero,best=best)


class FrozenReadoutStateTests(unittest.TestCase):
    def test_all_valid_committed_epochs_include_zero(self):
        identity = identity_fixture()
        for done in range(4):
            with self.subTest(completed=done):
                self.assertEqual(state.validate_progress(
                    saved_fixture(identity,done),identity,TRAIN_BATCHES),done)

    def test_identity_digest_ignores_only_its_own_field_and_rejects_nan(self):
        identity = identity_fixture()
        changed = deepcopy(identity); changed['identity_sha256'] = '0'*64
        self.assertEqual(state.identity_digest(changed),identity['identity_sha256'])
        changed['seed'] = SEED+1
        self.assertNotEqual(state.identity_digest(changed),identity['identity_sha256'])
        with self.assertRaises(ValueError): state.identity_digest(dict(value=float('nan')))

    def test_order_is_an_explicit_reproducible_permutation_without_global_rng_use(self):
        previous = np.random.get_state()
        try:
            np.random.seed(1911); before = np.random.get_state()
            first = state.epoch_order(SEED,1,TRAIN_COUNT)
            np.testing.assert_array_equal(first,state.epoch_order(SEED,1,TRAIN_COUNT))
            np.testing.assert_array_equal(np.sort(first),np.arange(TRAIN_COUNT))
            self.assertFalse(np.array_equal(first,state.epoch_order(SEED,2,TRAIN_COUNT)))
            after = np.random.get_state()
            self.assertEqual(before[0],after[0]); np.testing.assert_array_equal(before[1],after[1])
            self.assertEqual(before[2:],after[2:])
            np.testing.assert_array_equal(state.epoch_order(-1,1,5),[2,0,1,3,4])
            self.assertEqual([state.epoch_seed(SEED,e) for e in (1,2,3)],[12025,22025,32025])
        finally:
            np.random.set_state(previous)

    def test_order_and_seed_helpers_reject_extended_epochs_and_invalid_types(self):
        for epoch in (0,4,True,1.):
            with self.subTest(epoch=epoch),self.assertRaises(ValueError): state.epoch_order(SEED,epoch,5)
            with self.subTest(epoch=epoch),self.assertRaises(ValueError): state.epoch_seed(SEED,epoch)
        for count in (0,-1,True,5.):
            with self.subTest(count=count),self.assertRaises(ValueError): state.epoch_order(SEED,1,count)
        for seed in (True,2025.):
            with self.subTest(seed=seed),self.assertRaises(ValueError): state.epoch_seed(seed,1)

    def test_stage_zero_tie_uses_first_minimum(self):
        identity = identity_fixture()
        saved = saved_fixture(identity,scores=dict.fromkeys(ARMS,[10.,10.,10.]))
        self.assertEqual(state.validate_progress(saved,identity,TRAIN_BATCHES),3)
        self.assertTrue(all(saved['best'][arm]['epoch']==0 for arm in ARMS))
        saved['best'][CANDIDATE]['epoch'] = 3
        with self.assertRaisesRegex(ValueError,'first minimum'):
            state.validate_progress(saved,identity,TRAIN_BATCHES)

    def test_later_tied_best_is_rejected_and_heads_select_independently(self):
        identity = identity_fixture()
        saved = saved_fixture(identity,scores={'STATIC_READOUT':[9.,9.,9.5],
            'STATE_READOUT':[9.5,9.,9.]})
        self.assertEqual(state.validate_progress(saved,identity,TRAIN_BATCHES),3)
        self.assertEqual(saved['best']['STATIC_READOUT']['epoch'],1)
        self.assertEqual(saved['best']['STATE_READOUT']['epoch'],2)
        saved['best']['STATE_READOUT']['epoch'] = 3
        with self.assertRaisesRegex(ValueError,'first minimum'):
            state.validate_progress(saved,identity,TRAIN_BATCHES)

    def test_same_atomic_epoch_requires_equal_update_counts_for_both_heads(self):
        identity = identity_fixture(); saved = saved_fixture(identity)
        changes = (lambda s:s['head_updates'].update(STATIC_READOUT=164),
            lambda s:s['head_updates'].update(POSTHOC=246),
            lambda s:s['head_updates'].pop('STATE_READOUT'),
            lambda s:s['history'][1]['head_updates'].update(STATE_READOUT=82),
            lambda s:s.update(global_updates=492),
            lambda s:s['history'][1].update(global_updates=82),
            lambda s:s.update(global_updates=246.))
        for index,change in enumerate(changes):
            bad = deepcopy(saved); change(bad)
            with self.subTest(change=index),self.assertRaises(ValueError):
                state.validate_progress(bad,identity,TRAIN_BATCHES)

    def test_history_cannot_skip_duplicate_extend_or_use_boolean_epochs(self):
        identity = identity_fixture(); saved = saved_fixture(identity)
        changes = (lambda s:s.update(completed_epoch=4),lambda s:s.update(completed_epoch=True),
            lambda s:s['history'].pop(1),lambda s:s['history'][1].update(epoch=1),
            lambda s:s['history'][0].update(epoch=True),lambda s:s.update(history=None))
        for index,change in enumerate(changes):
            bad = deepcopy(saved); change(bad)
            with self.subTest(change=index),self.assertRaises(ValueError):
                state.validate_progress(bad,identity,TRAIN_BATCHES)

    def test_rehashed_identity_cannot_extend_or_change_locked_design(self):
        mutations = dict(new_epochs=4,train_count=3908,train_batches=81,seed=2026,
            full_horizon=6,extension_allowed=True,arms=list(reversed(ARMS)),candidate='STATIC_READOUT')
        for key,value in mutations.items():
            identity = identity_fixture(); identity[key] = value
            identity['identity_sha256'] = state.identity_digest(identity)
            saved = saved_fixture(identity)
            with self.subTest(field=key),self.assertRaises(ValueError):
                state.validate_progress(saved,identity,TRAIN_BATCHES)

    def test_identity_and_source_or_initialization_binding_cannot_change(self):
        identity = identity_fixture(); saved = saved_fixture(identity)
        changes = (lambda s:s.update(identity_sha256='0'*64),
            lambda s:s.update(schema='gw11_same_source_sum_joint_training_v1'),
            lambda s:s.update(source_tensor_sha256='0'*64),
            lambda s:s['initializations'][CANDIDATE].update(added_trainable_parameters=726))
        for index,change in enumerate(changes):
            bad = deepcopy(saved); change(bad)
            with self.subTest(change=index),self.assertRaises(ValueError):
                state.validate_progress(bad,identity,TRAIN_BATCHES)
        changed = deepcopy(identity); changed['seed'] += 1
        with self.assertRaises(ValueError): state.validate_progress(saved,changed,TRAIN_BATCHES)

    def test_shared_input_order_and_epoch_rng_are_locked(self):
        identity = identity_fixture(); saved = saved_fixture(identity)
        for field,value in (('train_order_sha256','0'*64),('dropout_epoch_seed',2025),
                ('dropout_epoch_seed',22025.)):
            bad = deepcopy(saved); bad['history'][1][field] = value
            with self.subTest(field=field,value=value),self.assertRaises(ValueError):
                state.validate_progress(bad,identity,TRAIN_BATCHES)

    def test_frozen_reference_metrics_must_remain_unchanged_each_epoch(self):
        identity = identity_fixture(); saved = saved_fixture(identity)
        for field in ('mae','rmse','signed_bias','window_p95'):
            bad = deepcopy(saved); bad['history'][1]['validation'][REFERENCE][field] += .01
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,'Frozen source'):
                state.validate_progress(bad,identity,TRAIN_BATCHES)

    def test_all_heads_have_exact_stage_zero_reference_metrics(self):
        identity = identity_fixture(); saved = saved_fixture(identity,completed=0)
        saved['stage_zero_validation'][CANDIDATE]['signed_bias'] += .01
        with self.assertRaisesRegex(ValueError,'zero output'):
            state.validate_progress(saved,identity,TRAIN_BATCHES)

    def test_validation_requires_finite_complete_metrics_and_stable_support(self):
        identity = identity_fixture(); saved = saved_fixture(identity)
        changes = (lambda s:s['history'][0]['validation'][CANDIDATE].update(mae=float('nan')),
            lambda s:s['history'][0]['validation'].pop('STATIC_READOUT'),
            lambda s:s['history'][0]['validation'][CANDIDATE].pop('window_p95'),
            lambda s:[s['history'][0]['validation'][a].update(valid_count=101) for a in CASES])
        for index,change in enumerate(changes):
            bad = deepcopy(saved); change(bad)
            with self.subTest(change=index),self.assertRaises(ValueError):
                state.validate_progress(bad,identity,TRAIN_BATCHES)

    def test_best_artifacts_require_local_basename_and_valid_digests(self):
        identity = identity_fixture(); saved = saved_fixture(identity)
        for field,value in (('checkpoint_file','../best.pt'),('checkpoint_file','C:\\best.pt'),
                ('checkpoint_file','best.json'),('checkpoint_sha256','z'*64),
                ('head_tensor_sha256','a'*63),('epoch',1.),('mae',8.)):
            bad = deepcopy(saved); bad['best'][CANDIDATE][field] = value
            with self.subTest(field=field,value=value),self.assertRaises(ValueError):
                state.validate_progress(bad,identity,TRAIN_BATCHES)

    def test_wrong_batch_partition_and_missing_head_best_are_rejected(self):
        identity = identity_fixture(); saved = saved_fixture(identity)
        for batches in (81,82.,True):
            with self.subTest(batches=batches),self.assertRaises(ValueError):
                state.validate_progress(saved,identity,batches)
        del saved['best']['STATIC_READOUT']
        with self.assertRaises(ValueError): state.validate_progress(saved,identity,TRAIN_BATCHES)


if __name__=='__main__':
    unittest.main()
