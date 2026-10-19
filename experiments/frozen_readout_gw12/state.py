"""Pure bookkeeping for an atomic, paired three-epoch frozen-source probe."""
import hashlib
import json

import numpy as np

from experiments.frozen_readout_gw12.protocol import (
    ARMS, CANDIDATE, CASES, EPOCHS, GROUPS, REFERENCE, SEED, TRAIN_BATCHES,
    TRAIN_COUNT, assert_same_reference, require, validate_records)

SCHEMA = 'gw12_frozen_sum_readout_probe_v1'


def identity_digest(identity):
    return hashlib.sha256(json.dumps({key:value for key,value in identity.items()
        if key!='identity_sha256'},sort_keys=True,allow_nan=False).encode()).hexdigest()


def epoch_order(seed, epoch, count):
    require(type(seed) is int and type(epoch) is int and 1<=epoch<=EPOCHS
        and type(count) is int and count>0, 'Invalid fixed epoch order request')
    return np.random.RandomState(seed+epoch).permutation(count)


def epoch_seed(seed, epoch):
    require(type(seed) is int and type(epoch) is int and 1<=epoch<=EPOCHS,
        'Invalid fixed epoch seed request')
    return seed+10000*epoch


def _sha256(value):
    return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)


def _updates(value, expected, context):
    require(type(value) is int and value==expected, 'Saved update budget differs: '+context)


def _head_updates(value, expected, context):
    require(isinstance(value,dict) and set(value)==set(ARMS),
        'Saved head update ledger differs: '+context)
    for arm in ARMS:
        _updates(value[arm],expected,context+'/'+arm)


def _checkpoint_best(best, epoch, mae, arm):
    require(isinstance(best,dict) and type(best.get('epoch')) is int
        and best['epoch']==epoch and best.get('mae')==mae,
        'BEST must select the first minimum including stage zero: '+arm)
    filename = best.get('checkpoint_file')
    require(isinstance(filename,str) and filename.endswith('.pt') and len(filename)>3
        and '/' not in filename and '\\' not in filename and ':' not in filename,
        'BEST checkpoint must be a local PT basename: '+arm)
    require(_sha256(best.get('checkpoint_sha256')) and _sha256(best.get('head_tensor_sha256')),
        'BEST checkpoint/tensor digest is invalid: '+arm)


def validate_progress(saved, identity, batches):
    """Validate one complete paired checkpoint without Torch or file access.

    A checkpoint commits both heads at the same completed epoch. Interrupted
    work inside the next epoch is replayed from this checkpoint for both heads.
    ``global_updates`` counts paired input batches, not the sum of two optimizer
    step counts; ``head_updates`` records the individual optimizer budgets.
    Tensor, optimizer, scheduler, RNG and file content digests are runner checks.
    """
    require(isinstance(identity,dict) and identity_digest(identity)==identity.get('identity_sha256')
        and identity.get('schema')==SCHEMA, 'GW12 run identity changed')
    require(identity.get('arms')==list(ARMS) and identity.get('candidate')==CANDIDATE
        and type(identity.get('new_epochs')) is int and identity['new_epochs']==EPOCHS
        and type(identity.get('train_count')) is int and identity['train_count']==TRAIN_COUNT
        and type(identity.get('train_batches')) is int and identity['train_batches']==TRAIN_BATCHES
        and type(identity.get('seed')) is int and identity['seed']==SEED
        and type(identity.get('full_horizon')) is int and identity['full_horizon']==12
        and identity.get('extension_allowed') is False,
        'Saved fixed experiment budget, order or registered heads differ')
    require(type(batches) is int and batches==TRAIN_BATCHES,
        'Saved epoch batch partition differs')
    initializations = identity.get('initializations')
    require(isinstance(initializations,dict) and set(initializations)==set(ARMS),
        'Both head initializations are required')
    require(_sha256(identity.get('source_tensor_sha256')),
        'Frozen source fingerprint is invalid')
    require(isinstance(saved,dict) and saved.get('schema')==SCHEMA
        and saved.get('identity_sha256')==identity['identity_sha256'],
        'Checkpoint identity differs')
    require(saved.get('initializations')==initializations,
        'Saved readout initialization differs')
    require(saved.get('source_tensor_sha256')==identity['source_tensor_sha256'],
        'Frozen source tensor fingerprint changed')
    done = saved.get('completed_epoch')
    history = saved.get('history')
    require(type(done) is int and 0<=done<=EPOCHS and isinstance(history,list)
        and len(history)==done and all(isinstance(row,dict) for row in history)
        and all(type(row.get('epoch')) is int for row in history)
        and [row['epoch'] for row in history]==list(range(1,done+1)),
        'Saved paired epoch ledger differs')
    _updates(saved.get('global_updates'),done*batches,'checkpoint')
    _head_updates(saved.get('head_updates'),done*batches,'checkpoint')
    stage_zero = saved.get('stage_zero_validation')
    validate_records(stage_zero,'stage zero')
    for arm in ARMS:
        assert_same_reference(stage_zero[REFERENCE],stage_zero[arm],'zero output/'+arm)
    all_records = [(0,stage_zero)]
    for row in history:
        epoch = row['epoch']
        _updates(row.get('global_updates'),epoch*batches,'epoch '+str(epoch))
        _head_updates(row.get('head_updates'),epoch*batches,'epoch '+str(epoch))
        expected = epoch_order(identity['seed'],epoch,identity['train_count'])
        require(row.get('train_order_sha256')==hashlib.sha256(expected.astype('<i8').tobytes()).hexdigest()
            and type(row.get('dropout_epoch_seed')) is int
            and row['dropout_epoch_seed']==epoch_seed(identity['seed'],epoch),
            'Saved paired input order or epoch RNG differs')
        records = row.get('validation')
        validate_records(records,'epoch '+str(epoch))
        assert_same_reference(stage_zero[REFERENCE],records[REFERENCE],'epoch '+str(epoch))
        for arm in CASES:
            require(records[arm]['valid_count']==stage_zero[arm]['valid_count']
                and all(records[arm]['groups'][name]['valid_count']==
                    stage_zero[arm]['groups'][name]['valid_count'] for name in GROUPS),
                'Validation support changed across epochs: '+arm)
        all_records.append((epoch,records))
    best = saved.get('best')
    require(isinstance(best,dict) and set(best)==set(ARMS),
        'Both head BEST selections are required')
    for arm in ARMS:
        epoch,records = min(all_records,key=lambda pair:pair[1][arm]['mae'])
        _checkpoint_best(best[arm],epoch,records[arm]['mae'],arm)
    return done
