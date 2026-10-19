"""GW9 fixed five-epoch continuation with stage-zero BEST and atomic resume.

No test loader, architecture search, budget extension or source-run mutation.
All three arms inherit fresh certified GW8 recurrent BEST tensors and a newly
reset Adam. Probe-trained head weights are deliberately never imported here.
"""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
import uuid

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO))
import numpy as np
import torch

from experiments.graph_memory_gw6 import train as gw6
from experiments.lag_transport_gw8 import train as gw8
from experiments.lag_transport_gw8.model import LagTransportIGSTGNN
from experiments.branch_readout_gw9.model import ARMS, DESIGN, BranchReadoutIGSTGNN
from experiments.branch_readout_gw9.reference import (
    certify, recertify, protected_paths, read_json, require, verify_identity_digest)
from experiments.chronological import train_original_isp as epoch_tools
from experiments.chronological.original_igstgnn_runtime import device_batch

base = gw6.base
SCHEMA = 'original_igstgnn_branch_readout_gw9_joint_training_v1'
SCREEN_EPOCHS = 5
EXECUTION_FILES = tuple(dict.fromkeys((*gw8.EXECUTION_FILES,
    'experiments/lag_transport_gw8/evaluate.py',
    'experiments/chronological/evaluate_original_graph_message.py',
    'experiments/chronological/evaluate_unified_adapter_v03.py',
    'experiments/branch_readout_gw9/__init__.py',
    'experiments/branch_readout_gw9/model.py',
    'experiments/branch_readout_gw9/reference.py',
    'experiments/branch_readout_gw9/train.py',
    'experiments/branch_readout_gw9/evaluate.py',
    'experiments/branch_readout_gw9/replay.py',
    'experiments/branch_readout_gw9/probe.py')))


def model_for_arm(data, modules, device, arm, gw4_weights, recurrent_weights,
                  hidden=16, stage='joint'):
    require(bool(recurrent_weights) and all(key.startswith('anchor.') for key in recurrent_weights),
        'Expected the entire certified GW8 recurrent wrapper state')
    inner = gw6.model_for_arm(data,modules,device,'RECURRENT_CONTROL',gw4_weights,stage='joint')
    inner.load_state_dict({key[len('anchor.'):]:value for key,value in recurrent_weights.items()},strict=True)
    anchor = LagTransportIGSTGNN(inner,'R_CONTINUE',hidden=16,edges=8,node_chunk=32).to(device)
    anchor.load_state_dict(recurrent_weights,strict=True)
    return BranchReadoutIGSTGNN(anchor,arm,stage=stage,hidden=hidden).to(device)


def joint_probe_gate(probe_dir, reference):
    # Local import: probe uses the shared model constructor, while this module
    # must remain independently importable for protocol tests.
    from experiments.branch_readout_gw9.probe import validate_gate
    return validate_gate(Path(probe_dir).resolve(), reference)


def initialization(model):
    added = {name:value for name,value in model.state_dict().items() if not name.startswith('anchor.')}
    return dict(source_tensor_sha256=base.tensor_tree_hash(model.anchor.state_dict()),
        added_tensor_sha256=base.tensor_tree_hash(added),
        registered_parameters=sum(p.numel() for p in model.parameters()),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        frozen_parameters=sum(p.numel() for p in model.parameters() if not p.requires_grad),
        added_trainable_parameters=sum(p.numel() for name,p in model.named_parameters()
            if not name.startswith('anchor.') and p.requires_grad),
        architecture_parameter_counts=model.trainable_parameter_counts())


def metric_row(epoch, values):
    mae,mape,rmse = map(float,values)
    require(bool(np.isfinite([mae,mape,rmse]).all()), 'Nonfinite validation metrics')
    return dict(epoch=epoch,validation_mae=mae,validation_rmse=rmse,
        validation_mape_fraction=mape,test_evaluated=False)


def selection_rows(state):
    return [state['stage_zero'], *state['history']]


def validate_saved(state,identity,arm,directory,batches):
    verify_identity_digest(identity)
    require(identity.get('schema') == SCHEMA and identity.get('design') == DESIGN
        and identity.get('arms') == list(ARMS) and arm in ARMS, 'Expected a GW9 training identity')
    require(state.get('identity_sha256') == identity['identity_sha256'] and state.get('arm') == arm,
        'Saved GW9 identity or arm differs')
    completed,history = state.get('completed_epoch'),state.get('history',[])
    require(type(completed) is int and 0 <= completed <= SCREEN_EPOCHS
        and len(history) == completed and [r.get('epoch') for r in history] == list(range(1,completed+1)),
        'Saved GW9 epoch history differs')
    require(state['global_updates'] == completed*batches
        and all(r['global_updates'] == r['epoch']*batches for r in history), 'Saved GW9 budget differs')
    require(state['initialization'] == identity['initializations'][arm], 'Saved initialization differs')
    require(state['stage_zero'].get('epoch') == 0 and all(np.isfinite([
        r['validation_mae'],r['validation_rmse'],r['validation_mape_fraction']]).all()
        for r in selection_rows(state)), 'Invalid stage-zero or validation metrics')
    order = np.asarray(state['train_order'])
    count = identity['data']['splits']['train']['count']
    require(order.shape == (count,) and np.array_equal(np.sort(order),np.arange(count)),
        'Saved training permutation differs')
    require(state['curriculum_length'] == (12 if completed else 0)
        and state['scheduler_state']['last_epoch'] == completed, 'Saved curriculum/scheduler differs')
    require(base.tensor_tree_hash(state['model_state']) == state['model_tensor_sha256'],
        'Saved model tensor digest differs')
    best = min(selection_rows(state),key=lambda row:row['validation_mae'])
    name = state['best_checkpoint_file']
    require(best['epoch'] == state['best_epoch']
        and best['validation_mae'] == state['best_validation_mae']
        and state['wait'] == completed-state['best_epoch']
        and isinstance(name,str) and Path(name).name == name
        and base.sha256(Path(directory)/name) == state['best_checkpoint_sha256'],
        'Saved BEST selection differs (stage zero must participate)')
    require(state.get('fixed_batch_sha256') and state.get('fixed_reference_prediction') is not None,
        'Saved fixed-input prediction reference missing')
    prediction = state['fixed_reference_prediction']
    require(prediction.ndim == 4 and prediction.shape[1] == 12 and prediction.shape[-1] == 1
        and bool(torch.isfinite(prediction).all()), 'Invalid stage-zero prediction tensor')
    require(state.get('stage_zero_model_tensor_sha256') and state.get('stage_zero_checkpoint_file'),
        'Missing immutable stage-zero weights')
    initial_file=state['stage_zero_checkpoint_file']
    require(Path(initial_file).name==initial_file
        and base.sha256(Path(directory)/initial_file)==state['stage_zero_checkpoint_sha256'],
        'Stage-zero file differs')
    initial_weights=base.load_checkpoint(Path(directory)/initial_file)
    require(base.tensor_tree_hash(initial_weights)==state['stage_zero_model_tensor_sha256']
        and base.tensor_tree_hash({key[len('anchor.'):]:value for key,value in initial_weights.items()
            if key.startswith('anchor.')})==state['initialization']['source_tensor_sha256']
        and base.tensor_tree_hash({key:value for key,value in initial_weights.items()
            if not key.startswith('anchor.')})==state['initialization']['added_tensor_sha256'],
        'Stage-zero checkpoint differs from certified recurrent source')


def make_engine(model,data,modules,device,directory,logger,seed):
    class Engine(modules.engine.IGSTGNN_Engine):
        def evaluate(self,mode):
            require(mode == 'val', 'GW9 evaluates validation only')
            self.model.start_diagnostics()
            return super().evaluate(mode)
    config,batches = data.config,data.loaders['train_loader'].num_batch
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
        lr=config['lrate'],weight_decay=config['wdecay'],eps=1e-8)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=base.MILESTONES,gamma=.5)
    return Engine(device=torch.device(device),model=model,dataloader=data.loaders,scaler=data.scaler,
        sampler=None,loss_fn=modules.metrics.masked_mae,lrate=config['lrate'],optimizer=optimizer,
        scheduler=scheduler,clip_grad_value=config['clip_grad_value'],max_epochs=SCREEN_EPOCHS,
        patience=config['patience'],log_dir=str(directory),logger=logger,seed=seed,
        cl_step=config['cl_epoch']*batches,warm_step=config['warm_epoch']*batches,horizon=12,incident=True)


def fixed_prediction(engine):
    loader = engine._dataloader['val_loader']
    batch = device_batch(next(loader.get_iterator()), engine._device)
    x, _, incident, sensors = batch
    rng, mode = base.rng_state(), engine.model.training
    try:
        engine.model.eval()
        with torch.no_grad():
            prediction = engine.model(x, incident_data=incident, sensor_data=sensors)
            require(prediction.ndim == 4 and prediction.shape[1] == 12 and prediction.shape[-1] == 1
                and bool(torch.isfinite(prediction).all()), 'Invalid fixed-input prediction')
            prediction = prediction.detach().cpu().clone()
        engine.model.assert_cache_clear()
        return base.tensor_tree_hash(batch), prediction
    finally:
        epoch_tools.restore_rng(rng)
        engine.model.train(mode)


def prediction_record(value, reference):
    x = value.double()
    delta = x-reference.double()
    return dict(domain='normalized_model_prediction_before_single_inverse_transform',
        mean=float(x.mean()), rms=float(x.square().mean().sqrt()),
        negative_fraction=float((x<0).double().mean()), change_mean=float(delta.mean()),
        change_rms=float(delta.square().mean().sqrt()),
        caveat='prediction_change_includes_native_and_small_readout_updates')


def arm_summary(state):
    result = base.arm_summary(state,SCREEN_EPOCHS,False)
    best = min(selection_rows(state),key=lambda row:row['validation_mae'])
    result.update(best_metrics=epoch_tools.metric_record(best),stage_zero=state['stage_zero'],
        best_selection_includes_stage_zero=True,optimizer_reset_at_start=True,
        exceeds_stage_zero=state['best_validation_mae'] < state['stage_zero']['validation_mae'])
    return result


def run_epochs(engine,directory,identity,arm,target_epoch=SCREEN_EPOCHS,state=None):
    require(type(target_epoch) is int and 0 <= target_epoch <= SCREEN_EPOCHS,
        'GW9 screening is capped at five new epochs')
    directory=Path(directory)
    get_order=epoch_tools.tracked_shuffle(engine._dataloader['train_loader'],
        None if state is None else state['train_order'])
    if state is None:
        (mae,mape,rmse),timing = epoch_tools.timed_phase(engine,lambda:engine.evaluate('val'))
        stage_zero=metric_row(0,(mae,mape,rmse)); stage_zero['validation_timing']=timing
        stage_zero['mechanisms']=engine.model.diagnostics()
        expected=identity.get('source_reference',{}).get('source_validation_metrics')
        if expected is not None:
            require(all(np.isclose(stage_zero['validation_'+key],value,atol=1e-5,rtol=1e-6)
                for key,value in expected.items()),'Stage-zero output differs from certified GW8 recurrent BEST validation')
        batch_sha,prediction=fixed_prediction(engine)
        name='best_epoch_000_'+uuid.uuid4().hex[:12]+'.pt'
        epoch_tools.atomic_save(directory/name,engine.model.state_dict())
        state=dict(arm=arm,completed_epoch=0,best_epoch=0,best_validation_mae=mae,wait=0,
            best_checkpoint_file=name,best_checkpoint_sha256=base.sha256(directory/name),
            global_updates=0,history=[],stage_zero=stage_zero,
            initialization=identity['initializations'][arm],fixed_batch_sha256=batch_sha,
            fixed_reference_prediction=prediction,stage_zero_checkpoint_file=name,
            stage_zero_checkpoint_sha256=base.sha256(directory/name),
            stage_zero_model_tensor_sha256=base.tensor_tree_hash(engine.model.state_dict()))
        stage_zero['fixed_input_prediction']=prediction_record(prediction,prediction)
        # Fixed prediction capture can collect diagnostics if enabled;
        # clear once before training so no synchronization enters train_batch.
        engine.model.diagnostics()
        epoch_tools.commit_state(engine,state,directory,identity,get_order)
    else:
        engine.model.load_state_dict(state['model_state'],strict=True)
        engine._optimizer.load_state_dict(state['optimizer_state'])
        engine._lr_scheduler.load_state_dict(state['scheduler_state'])
        engine._iter_cnt,state_cl=state['global_updates'],state['curriculum_length']
        engine._cl_len=state_cl
        epoch_tools.restore_rng(state['rng_state'])
    for epoch in range(state['completed_epoch']+1,target_epoch+1):
        print(json.dumps(dict(arm=arm,phase='training',epoch=epoch,target_epochs=SCREEN_EPOCHS)),flush=True)
        with epoch_tools.batch_progress(engine._dataloader['train_loader'],arm,'training',epoch):
            (train_mae,train_mape,train_rmse),train_timing=epoch_tools.timed_phase(engine,engine.train_batch)
        with epoch_tools.batch_progress(engine._dataloader['val_loader'],arm,'validation',epoch):
            values,val_timing=epoch_tools.timed_phase(engine,lambda:engine.evaluate('val'))
        row=metric_row(epoch,values)
        row.update(train_mae=train_mae,train_mape_fraction=train_mape,train_rmse=train_rmse,
            train_timing=train_timing,validation_timing=val_timing,
            train_seconds=train_timing['seconds'],validation_seconds=val_timing['seconds'],
            learning_rate=engine._lr_scheduler.get_last_lr()[0],global_updates=engine._iter_cnt,
            curriculum_length=engine._cl_len,
            train_order_sha256=hashlib.sha256(np.asarray(get_order(),dtype=np.int64).tobytes()).hexdigest(),
            mechanisms=engine.model.diagnostics())
        engine._lr_scheduler.step()
        batch_sha,prediction=fixed_prediction(engine)
        require(batch_sha == state['fixed_batch_sha256'], 'Fixed validation input changed')
        row['fixed_input_prediction']=prediction_record(prediction,state['fixed_reference_prediction'])
        state['history'].append(row)
        if row['validation_mae'] < state['best_validation_mae']:
            name='best_epoch_{:03d}_{}.pt'.format(epoch,uuid.uuid4().hex[:12])
            epoch_tools.atomic_save(directory/name,engine.model.state_dict())
            state.update(best_epoch=epoch,best_validation_mae=row['validation_mae'],wait=0,
                best_checkpoint_file=name,best_checkpoint_sha256=base.sha256(directory/name))
        else:
            state['wait']+=1
        state['completed_epoch']=epoch
        epoch_tools.commit_state(engine,state,directory,identity,get_order)
        base.write_json(directory/'summary.json',arm_summary(state))
        print(json.dumps(dict(arm=arm,**row)),flush=True)
    return state


def paired_summary(states,identity):
    complete=set(states)==set(ARMS) and all(s['completed_epoch']==SCREEN_EPOCHS for s in states.values())
    return dict(schema=SCHEMA,status='GW9_TRAINING_COMPLETE' if complete else 'GW9_TRAINING_PARTIAL',
        requested_new_epochs=SCREEN_EPOCHS,same_completed_new_training_budget=complete,
        arms={arm:arm_summary(state) for arm,state in states.items()},source_reference=identity['source_reference'],
        model_options=identity['model_options'],registered_arms=list(ARMS),
        best_selection_includes_stage_zero=True,optimizer_reset_at_start=True,
        optimizer_reset_on_resume=False,early_stopping_enabled=False,test_evaluated=False,
        predictive_accuracy_pass=False,interpretation='single_seed_development_screen_not_independent_confirmation')


def parse_args(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gw8-run',type=Path,required=True)
    parser.add_argument('--gw8-best-eval',type=Path)
    parser.add_argument('--gw8-last-eval',type=Path)
    parser.add_argument('--data-dir',type=Path,default=base.default_data_dir())
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--probe-dir',type=Path,required=True,
        help='Completed authenticated three-epoch probe with all locked checks passing')
    parser.add_argument('--epochs',type=int,choices=(SCREEN_EPOCHS,),default=SCREEN_EPOCHS,
        help='Predeclared fixed total new budget; cannot extend this screening run')
    parser.add_argument('--hidden',type=int,choices=(16,),default=16)
    parser.add_argument('--seed',type=int,default=2025)
    parser.add_argument('--threads',type=int,default=3)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--only',nargs='+',choices=ARMS)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args(argv)
    if min(args.hidden,args.threads)<1 or (args.only and len(args.only)!=len(set(args.only))):
        parser.error('Positive resource options and a unique --only subset are required')
    for key in ('gw8_run','data_dir','run_dir','probe_dir'):
        setattr(args,key,getattr(args,key).resolve())
    return args


def main(argv=None):
    args=parse_args(argv)
    require(args.run_dir.exists()==args.resume,'Use a fresh GW9 run, or --resume the same run')
    identity_path=args.run_dir/'identity.json'
    if args.resume and identity_path.is_file():
        saved=read_json(identity_path)
        require(saved.get('schema')==SCHEMA,'Cannot resume another experiment')
        reference=saved['source_reference']
        require(reference['source_run']==str(args.gw8_run),'Cannot replace GW8 source')
        old_identity,gw4_weights,recurrent_weights=recertify(reference)
    else:
        if args.resume:
            require(all(p.name in ('active.lock','record.log') or
                (p.is_file() and p.name.startswith('failure_') and p.suffix=='.json')
                for p in args.run_dir.iterdir()),'Unrecognized files in identity-less resume directory')
        reference,old_identity,gw4_weights,recurrent_weights=certify(args.gw8_run,
            {key:getattr(args,'gw8_'+key+'_eval') for key in ('best','last')})
    probe_gate=joint_probe_gate(args.probe_dir,reference)
    for protected in (*protected_paths(reference),args.probe_dir):
        require(args.run_dir!=protected and protected not in args.run_dir.parents,
            'GW9 output must be outside every immutable source run')
    args.run_dir.mkdir(parents=True,exist_ok=args.resume)
    with base.run_lock(args.run_dir):
        logger=logging.getLogger('gw9-'+str(args.run_dir)); logger.setLevel(logging.INFO)
        handler=logging.FileHandler(args.run_dir/'record.log',encoding='utf-8'); logger.addHandler(handler)
        try:
            runtime=base.configure_runtime(args.device,args.threads,args.seed)
            modules=base.get_original_modules()
            require(modules.source_fingerprint==old_identity['source_fingerprint'],'Native source differs')
            config=dict(base.CONFIG,device=args.device,seed=args.seed,warm_epoch=101)
            data=base.load_data(args.data_dir,config,modules,logger,check=False,training=True)
            require(data.identity==old_identity['data'],'GW9 data/split identity differs')
            samples=tuple(data.loaders['train_loader'].samples)
            options=dict(hidden=args.hidden)
            initializations={}
            for arm in ARMS:
                base.configure_runtime(args.device,args.threads,args.seed)
                model=model_for_arm(data,modules,args.device,arm,gw4_weights,recurrent_weights,**options)
                initializations[arm]=initialization(model)
                require(initializations[arm]['source_tensor_sha256']==reference['source_initialization']['model_tensor_sha256'],
                    'Construction changed GW8 recurrent BEST source: '+arm)
                del model
            require(len({initializations[arm]['trainable_parameters'] for arm in ARMS[1:]}) == 1
                and len({initializations[arm]['added_tensor_sha256'] for arm in ARMS[1:]}) == 1
                and all(initializations[arm]['added_trainable_parameters']==8256 for arm in ARMS[1:]),
                'GW9 active capacities or initial readout tensors differ')
            identity=dict(schema=SCHEMA,design=DESIGN,arms=list(ARMS),model_options=options,
                training_config=config,data=data.identity,runtime=runtime,source_fingerprint=modules.source_fingerprint,
                source_reference=reference,probe_gate=probe_gate,initializations=initializations,
                execution_source={name:base.sha256(REPO/name) for name in EXECUTION_FILES},
                initial_train_order_sha256=hashlib.sha256(np.arange(len(samples),dtype=np.int64).tobytes()).hexdigest(),
                scheduler=dict(milestones=base.MILESTONES,gamma=.5),fixed_screen_epochs=SCREEN_EPOCHS,
                stage='joint',probe_weights_loaded=False,heads_reset_to_zero_output=True,
                common_post_construction_training_rng='configure_runtime(seed)_after_model_and_engine_construction',
                best_selection_includes_stage_zero=True,optimizer_reset_at_start=True,
                engineering_only=False,test_evaluated=False)
            identity['identity_sha256']=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
            budget=dict(target_completed_epochs=SCREEN_EPOCHS,total_not_additional=True,maximum=SCREEN_EPOCHS,
                extension_allowed=False,test_evaluated=False)
            if identity_path.is_file():
                require(read_json(identity_path)==identity,'GW9 identity changed')
                require(read_json(args.run_dir/'source_reference.json')==reference,'GW9 source certificate changed')
                require(read_json(args.run_dir/'budget.json')==budget,'GW9 fixed budget changed')
            else:
                base.write_json(args.run_dir/'source_reference.json',reference)
                base.write_json(args.run_dir/'budget.json',budget)
                base.write_json(identity_path,identity)
            states={}; batches=data.loaders['train_loader'].num_batch
            for arm in ARMS:
                path=args.run_dir/arm/'last.pt'
                if path.is_file():
                    states[arm]=base.load_checkpoint(path)
                    validate_saved(states[arm],identity,arm,path.parent,batches)
            for arm in args.only or ARMS:
                directory=args.run_dir/arm; directory.mkdir(exist_ok=True)
                state=states.get(arm)
                if state and state['completed_epoch']==SCREEN_EPOCHS:
                    print(json.dumps(dict(arm=arm,status='ALREADY_COMPLETE')),flush=True); continue
                base.configure_runtime(args.device,args.threads,args.seed)
                data.loaders['train_loader'].samples=list(samples)
                model=model_for_arm(data,modules,args.device,arm,gw4_weights,recurrent_weights,**options)
                require(initialization(model)==initializations[arm],'Actual arm initialization changed')
                engine=make_engine(model,data,modules,args.device,directory,logger,args.seed)
                # New modules consume different constructor RNG. Reset the actual
                # training/dropout/shuffle stream after construction in EVERY arm.
                base.configure_runtime(args.device,args.threads,args.seed)
                states[arm]=run_epochs(engine,directory,identity,arm,state=state)
                validate_saved(states[arm],identity,arm,directory,batches)
                base.write_json(args.run_dir/'summary.json',paired_summary(states,identity))
                del model,engine; gc.collect()
                if torch.device(args.device).type=='cuda': torch.cuda.empty_cache()
            recertify(reference)
            require(joint_probe_gate(args.probe_dir,reference)==probe_gate,
                'Frozen probe certificate changed during joint training')
            for name in EXECUTION_FILES:
                require(identity['execution_source'][name]==base.sha256(REPO/name),
                    'GW9 execution source changed during training: '+name)
            summary=paired_summary(states,identity); base.write_json(args.run_dir/'summary.json',summary)
            print(json.dumps(summary),flush=True)
            return summary
        except BaseException as error:
            base.write_json(args.run_dir/('failure_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')+'.json'),
                dict(status='GW9_TRAINING_FAILED',error_type=type(error).__name__,error=str(error),
                    resume_semantics='last_atomic_epoch_incomplete_epoch_replayed',test_evaluated=False))
            raise
        finally:
            logger.removeHandler(handler); handler.close()


if __name__=='__main__': main()
