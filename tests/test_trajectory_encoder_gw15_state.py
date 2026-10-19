"""Server-only GW15 paired checkpoint tests using synthetic records.

Do not run Python, imports, syntax checks or tests on the workstation.
No Torch, model, cached author arrays or GPU is needed by these tests.
"""
from copy import deepcopy
import hashlib
import unittest

import numpy as np

from experiments.trajectory_encoder_gw15 import state
from experiments.trajectory_encoder_gw15.protocol import (
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
            head_tensor_sha256='b'*64,registered_parameters=6336,
            trainable_parameters=6336) for arm in ARMS},
        full_horizon=12,extension_allowed=False)
    result['identity_sha256'] = state.identity_digest(result)
    return result


def saved_fixture(identity,completed=3,scores=None):
    scores = {arm:[9.,9.,9.5] for arm in ARMS} if scores is None else scores
    stage_zero = {arm:record() for arm in CASES}
    history = []
    for epoch in range(1,completed+1):
        order = np.random.RandomState(SEED+epoch).permutation(TRAIN_COUNT)
        validation = {REFERENCE:record()}
        validation.update({arm:record(scores[arm][epoch-1]) for arm in ARMS})
        history.append(dict(epoch=epoch,global_updates=epoch*TRAIN_BATCHES,
            head_updates=dict.fromkeys(ARMS,epoch*TRAIN_BATCHES),validation=validation,
            train_order_sha256=hashlib.sha256(order.astype('<i8').tobytes()).hexdigest(),
            dropout_epoch_seed=SEED+10000*epoch))
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


class TrajectoryEncoderStateTests(unittest.TestCase):
    def test_all_committed_epochs_and_first_minimum_include_stage_zero(self):
        identity = identity_fixture()
        for done in range(4):
            with self.subTest(completed=done):
                self.assertEqual(state.validate_progress(
                    saved_fixture(identity,done),identity,TRAIN_BATCHES),done)
        saved = saved_fixture(identity,scores=dict.fromkeys(ARMS,[10.,10.,10.]))
        self.assertEqual(state.validate_progress(saved,identity,TRAIN_BATCHES),3)
        self.assertTrue(all(saved['best'][arm]['epoch']==0 for arm in ARMS))
        saved['best'][CANDIDATE]['epoch'] = 3
        with self.assertRaisesRegex(ValueError,'first minimum'):
            state.validate_progress(saved,identity,TRAIN_BATCHES)

    def test_paired_adapters_select_best_independently(self):
        identity = identity_fixture()
        saved = saved_fixture(identity,scores={'SORTED_DELTA':[9.,9.,9.5],
            'ORDERED_DELTA':[9.5,9.,9.]})
        self.assertEqual(state.validate_progress(saved,identity,TRAIN_BATCHES),3)
        self.assertEqual(saved['best']['SORTED_DELTA']['epoch'],1)
        self.assertEqual(saved['best']['ORDERED_DELTA']['epoch'],2)
        saved['best']['ORDERED_DELTA']['epoch'] = 3
        with self.assertRaisesRegex(ValueError,'first minimum'):
            state.validate_progress(saved,identity,TRAIN_BATCHES)

    def test_atomic_epoch_rejects_unequal_or_incomplete_updates_and_history(self):
        identity = identity_fixture(); saved = saved_fixture(identity)
        changes = (lambda s:s['head_updates'].update(SORTED_DELTA=164),
            lambda s:s['head_updates'].update(POSTHOC=246),
            lambda s:s['history'][1]['head_updates'].update(ORDERED_DELTA=82),
            lambda s:s.update(global_updates=492),
            lambda s:s.update(completed_epoch=True),
            lambda s:s.update(completed_epoch=4),
            lambda s:s['history'].pop(1),
            lambda s:s['history'][1].update(epoch=1))
        for index,change in enumerate(changes):
            bad = deepcopy(saved); change(bad)
            with self.subTest(change=index),self.assertRaises(ValueError):
                state.validate_progress(bad,identity,TRAIN_BATCHES)

    def test_rehashed_identity_cannot_change_locked_design(self):
        mutations = dict(new_epochs=4,train_count=3908,train_batches=81,seed=2026,
            full_horizon=6,extension_allowed=True,arms=list(reversed(ARMS)),candidate='SORTED_DELTA')
        for key,value in mutations.items():
            identity = identity_fixture(); identity[key] = value
            identity['identity_sha256'] = state.identity_digest(identity)
            saved = saved_fixture(identity)
            with self.subTest(field=key),self.assertRaises(ValueError):
                state.validate_progress(saved,identity,TRAIN_BATCHES)

    def test_source_identity_and_initialization_are_bound_to_checkpoint(self):
        identity = identity_fixture(); saved = saved_fixture(identity)
        changes = (lambda s:s.update(identity_sha256='0'*64),
            lambda s:s.update(schema='gw12_frozen_sum_readout_probe_v1'),
            lambda s:s.update(source_tensor_sha256='0'*64),
            lambda s:s['initializations'][CANDIDATE].update(trainable_parameters=1))
        for index,change in enumerate(changes):
            bad = deepcopy(saved); change(bad)
            with self.subTest(change=index),self.assertRaises(ValueError):
                state.validate_progress(bad,identity,TRAIN_BATCHES)

    def test_equal_capacity_and_identical_initial_tensors_are_required(self):
        changes = (dict(trainable_parameters=6335),
            dict(registered_parameters=6335,trainable_parameters=6335),
            dict(head_tensor_sha256='c'*64),dict(head_tensor_sha256='z'*64),
            dict(registered_parameters=True,trainable_parameters=True))
        for change in changes:
            identity = identity_fixture(); identity['initializations'][CANDIDATE].update(change)
            identity['identity_sha256'] = state.identity_digest(identity)
            with self.subTest(change=change),self.assertRaises(ValueError):
                state.validate_progress(saved_fixture(identity),identity,TRAIN_BATCHES)

    def test_shared_order_frozen_reference_and_validation_support_cannot_drift(self):
        identity = identity_fixture(); saved = saved_fixture(identity)
        changes = (lambda s:s['history'][1].update(train_order_sha256='0'*64),
            lambda s:s['history'][1].update(dropout_epoch_seed=2025),
            lambda s:s['history'][1]['validation'][REFERENCE].update(mae=9.),
            lambda s:s['history'][0]['validation'][CANDIDATE].update(mae=float('nan')),
            lambda s:s['history'][0]['validation'].pop('SORTED_DELTA'),
            lambda s:[s['history'][0]['validation'][a].update(valid_count=101) for a in CASES],
            lambda s:s['stage_zero_validation'][CANDIDATE].update(signed_bias=.1))
        for index,change in enumerate(changes):
            bad = deepcopy(saved); change(bad)
            with self.subTest(change=index),self.assertRaises(ValueError):
                state.validate_progress(bad,identity,TRAIN_BATCHES)

    def test_best_requires_local_artifacts_valid_digests_and_exact_epoch(self):
        identity = identity_fixture(); saved = saved_fixture(identity)
        for field,value in (('checkpoint_file','../best.pt'),('checkpoint_file','C:\\best.pt'),
                ('checkpoint_file','best.json'),('checkpoint_sha256','z'*64),
                ('head_tensor_sha256','a'*63),('epoch',1.),('mae',8.)):
            bad = deepcopy(saved); bad['best'][CANDIDATE][field] = value
            with self.subTest(field=field,value=value),self.assertRaises(ValueError):
                state.validate_progress(bad,identity,TRAIN_BATCHES)
        for batches in (81,82.,True):
            with self.subTest(batches=batches),self.assertRaises(ValueError):
                state.validate_progress(saved,identity,batches)

    def test_digest_and_order_helpers_are_reproducible_without_global_rng_use(self):
        identity = identity_fixture(); changed = deepcopy(identity)
        changed['identity_sha256'] = '0'*64
        self.assertEqual(state.identity_digest(changed),identity['identity_sha256'])
        changed['seed'] += 1
        self.assertNotEqual(state.identity_digest(changed),identity['identity_sha256'])
        with self.assertRaises(ValueError): state.identity_digest(dict(value=float('nan')))
        previous = np.random.get_state()
        try:
            np.random.seed(1911); before = np.random.get_state()
            first = state.epoch_order(SEED,1,TRAIN_COUNT)
            np.testing.assert_array_equal(first,state.epoch_order(SEED,1,TRAIN_COUNT))
            np.testing.assert_array_equal(np.sort(first),np.arange(TRAIN_COUNT))
            after = np.random.get_state()
            self.assertEqual(before[0],after[0]); np.testing.assert_array_equal(before[1],after[1])
            self.assertEqual(before[2:],after[2:])
            self.assertEqual([state.epoch_seed(SEED,e) for e in (1,2,3)],[12025,22025,32025])
        finally:
            np.random.set_state(previous)
        for epoch in (0,4,True,1.):
            with self.subTest(epoch=epoch),self.assertRaises(ValueError): state.epoch_order(SEED,epoch,5)
            with self.subTest(epoch=epoch),self.assertRaises(ValueError): state.epoch_seed(SEED,epoch)


if __name__=='__main__':
    unittest.main()
