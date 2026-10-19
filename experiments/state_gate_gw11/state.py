"""Pure bookkeeping for a fixed three-epoch paired continuation experiment."""
import hashlib
import json

import numpy as np

from experiments.state_gate_gw11.protocol import ARMS, EPOCHS

SCHEMA = 'gw11_same_source_sum_joint_training_v1'


def require(condition, message):
    if not condition: raise ValueError(message)


def identity_digest(identity):
    return hashlib.sha256(json.dumps({key:value for key,value in identity.items()
        if key!='identity_sha256'},sort_keys=True,allow_nan=False).encode()).hexdigest()


def epoch_order(seed, epoch, count):
    require(type(epoch) is int and 1<=epoch<=EPOCHS and count>0,'Invalid fixed epoch order request')
    return np.random.RandomState(seed+epoch).permutation(count)


def epoch_seed(seed, epoch):
    require(type(epoch) is int and 1<=epoch<=EPOCHS,'Invalid fixed epoch seed request')
    return seed+10000*epoch


def validate_progress(saved, identity, arm, batches):
    require(identity_digest(identity)==identity.get('identity_sha256')
        and identity.get('schema')==SCHEMA,'GW11 run identity changed')
    require(arm in ARMS and saved.get('schema')==SCHEMA and saved.get('arm')==arm
        and saved.get('identity_sha256')==identity['identity_sha256'],'Checkpoint identity/arm differs')
    require(type(batches) is int and batches>0 and batches==identity.get('train_batches'),
        'Saved epoch batch partition differs')
    require(identity.get('new_epochs')==EPOCHS and identity.get('arms')==list(ARMS),
        'Saved fixed experiment budget or registered arms differ')
    done = saved.get('completed_epoch'); history = saved.get('history',[])
    require(type(done) is int and 0<=done<=EPOCHS and len(history)==done
        and [row.get('epoch') for row in history]==list(range(1,done+1)),
        'Saved epoch ledger differs')
    require(type(saved.get('global_updates')) is int and saved['global_updates']==done*batches
        and all(type(row.get('global_updates')) is int
            and row['global_updates']==row['epoch']*batches for row in history),
        'Saved update budget differs')
    require(saved.get('initialization')==identity['initializations'][arm],
        'Saved model initialization differs')
    rows = [(0,saved['stage_zero'])]+[(row['epoch'],row['validation']) for row in history]
    require(all(np.isfinite(row['mae']) and row['mae']>=0 for _,row in rows),
        'Saved validation score is invalid')
    best_epoch,best_row = min(rows,key=lambda pair:pair[1]['mae'])
    require(saved['best']['epoch']==best_epoch and saved['best']['mae']==best_row['mae'],
        'BEST must select the first minimum including stage zero')
    for row in history:
        expected = epoch_order(identity['seed'],row['epoch'],identity['train_count'])
        require(row.get('train_order_sha256')==hashlib.sha256(expected.astype('<i8').tobytes()).hexdigest()
            and row.get('dropout_epoch_seed')==epoch_seed(identity['seed'],row['epoch']),
            'Saved paired input order or epoch RNG differs')
    return done


def next_work(states, epoch):
    """Committed arms are skipped after a partially completed paired epoch."""
    require(type(epoch) is int and 1<=epoch<=EPOCHS,'Cannot extend the fixed epoch budget')
    require(set(states)==set(ARMS),'All registered arm states are required')
    completed = [states[arm]['completed_epoch'] for arm in ARMS]
    require(all(type(value) is int and 0<=value<=EPOCHS for value in completed),
        'Invalid completed epoch')
    require(max(completed)-min(completed)<=1 and completed==sorted(completed,reverse=True),
        'Saved arms cannot arise from the prescribed paired epoch order')
    result = []
    for arm in ARMS:
        done = states[arm]['completed_epoch']
        require(type(done) is int and 0<=done<=EPOCHS,'Invalid completed epoch')
        if done>=epoch: continue
        require(done==epoch-1,'A preceding epoch is missing')
        result.append(arm)
    return result
