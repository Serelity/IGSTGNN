"""Fixed four-arm joint continuation from certified GW9 SUM BEST.

Run the CPU mask audit first. All source weights, including the trained SUM
head, remain present and trainable. Partial epochs are replayed from the last
atomic arm checkpoint; committed arms in a paired epoch are skipped on resume.
"""
import argparse
import copy
from datetime import datetime, timezone
import gc
import hashlib
import json
import logging
import os
from pathlib import Path
from time import perf_counter
import uuid

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch

from experiments.branch_readout_gw9 import train as gw9
from experiments.branch_readout_gw9 import probe as gw9_probe
from experiments.branch_readout_gw9.reference import recertify, read_json
from experiments.branch_readout_gw9.readout_auth import assert_unchanged as assert_source_unchanged
from experiments.chronological.graph_message_reuse import RUNTIME_KEYS
from experiments.chronological.original_igstgnn_runtime import device_batch
from experiments.conditional_residual_gw10.median_probe import load_model, predict_corrections
from experiments.conditional_residual_gw10.screen import metrics
from experiments.state_gate_gw11.model import StateGateIGSTGNN
from experiments.state_gate_gw11.protocol import ARMS, CANDIDATE, EPOCHS, GROUPS, screen
from experiments.state_gate_gw11.reference import (
    load_inputs, require_output_outside_sources, assert_unchanged as assert_reference_unchanged)
from experiments.state_gate_gw11.mask_audit import validate_audit
from experiments.state_gate_gw11.state import (
    SCHEMA, epoch_order, epoch_seed, identity_digest, next_work, require, validate_progress)

base, epoch_tools, REPO = gw9.base,gw9.epoch_tools,gw9.REPO
TRAINING_MARKER_SCHEMA = 'gw11_training_started_marker_v1'
EXECUTION_FILES = tuple(dict.fromkeys((*gw9.EXECUTION_FILES,
    'experiments/state_gate_gw11/__init__.py','experiments/state_gate_gw11/model.py',
    'experiments/state_gate_gw11/protocol.py','experiments/state_gate_gw11/state.py',
    'experiments/state_gate_gw11/reference.py','experiments/state_gate_gw11/mask_audit.py',
    'experiments/state_gate_gw11/train.py','experiments/conditional_residual_gw10/run_probe.py',
    'experiments/conditional_residual_gw10/export_cache.py',
    'experiments/conditional_residual_gw10/median_probe.py',
    'experiments/conditional_residual_gw10/screen.py',
    'experiments/branch_readout_gw9/readout_auth.py')))


def source_hashes():
    return {name:base.sha256(REPO/name) for name in EXECUTION_FILES}


def checkpoint_child(directory, name):
    require(isinstance(name,str) and Path(name).name==name and name.endswith('.pt'),
        'Checkpoint path must be a local PT basename')
    return Path(directory)/name


def record(prediction, truth, valid):
    result = metrics(prediction,truth,valid)
    masks = dict(high_value_ge314=truth>=314,
        long_horizon_7_12=np.arange(12)[None,:,None]>=6,
        flow_50_200=(truth>=50)&(truth<200))
    result['groups'] = {}
    for name in GROUPS:
        selected = valid & np.broadcast_to(masks[name],valid.shape)
        if selected.any():
            result['groups'][name] = metrics(prediction,truth,selected)
        else:
            result['groups'][name] = dict(valid_count=0,mae=None,signed_bias=None)
    counts = valid.sum(axis=(1,2)); complete = counts>0
    scores = np.divide((np.abs(prediction.astype(np.float64)-truth)*valid).sum(axis=(1,2)),
        counts,out=np.zeros(len(counts),dtype=np.float64),where=complete)
    result['window_p95'] = float(np.quantile(scores[complete],.95)) if complete.any() else None
    return result


def same_record(actual, expected):
    def equal(left, right):
        if left is None or right is None:
            return left is None and right is None
        return bool(np.isclose(left,right,atol=1e-10,rtol=1e-10))
    fields = ('mae','rmse','mape_fraction','signed_bias')
    return (actual['valid_count']==expected['valid_count']
        and all(equal(actual[key],expected[key]) for key in fields)
        and equal(actual.get('window_p95'),expected.get('window_p95'))
        and all(actual['groups'][group]['valid_count']==expected['groups'][group]['valid_count']
            and set(actual['groups'][group])==set(expected['groups'][group])
            and all(equal(value,expected['groups'][group][key])
                for key,value in actual['groups'][group].items() if key!='valid_count')
            for group in GROUPS))


def cpu_tree(value):
    """Detach owned snapshots so no committed state aliases a live optimizer."""
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value,dict):
        return {key:cpu_tree(child) for key,child in value.items()}
    if isinstance(value,list):
        return [cpu_tree(child) for child in value]
    if isinstance(value,tuple):
        return tuple(cpu_tree(child) for child in value)
    if isinstance(value,np.ndarray):
        return value.copy()
    return copy.deepcopy(value)


def weights_copy(model):
    return cpu_tree(model.state_dict())


def initialization(model):
    added = {name:value for name,value in model.state_dict().items() if not name.startswith('source.')}
    return dict(source_tensor_sha256=base.tensor_tree_hash(model.source.state_dict()),
        added_tensor_sha256=base.tensor_tree_hash(added),
        registered_parameters=sum(value.numel() for value in model.parameters()),
        trainable_parameters=sum(value.numel() for value in model.parameters() if value.requires_grad),
        added_trainable_parameters=sum(value.numel() for name,value in model.named_parameters()
            if not name.startswith('source.') and value.requires_grad))


def model_for_arm(data, modules, device, arm, gw4, recurrent, head):
    source = gw9.model_for_arm(data,modules,device,'SUM_CONTROL',gw4,recurrent,hidden=16,stage='joint')
    source.head.load_state_dict(head,strict=True)
    return StateGateIGSTGNN(source,arm,hidden=16).to(device)


def make_optimizer(model, config):
    optimizer = torch.optim.Adam(model.parameters(),lr=config['lrate'],weight_decay=config['wdecay'],eps=1e-8)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=base.MILESTONES,gamma=.5)
    return optimizer,scheduler


def evaluate(model, data, device, baseline, arm, strict_source=False):
    model.eval(); model.start_diagnostics(); prediction,truth,support = [],[],[]
    if torch.device(device).type=='cuda': torch.cuda.synchronize(device)
    start = perf_counter()
    with torch.no_grad():
        for index,raw in enumerate(data.loaders['val_loader'].get_iterator(),1):
            x,y,incident,sensors = device_batch(raw,device)
            values = model(x,incident_data=incident,sensor_data=sensors)
            prediction.append(data.scaler.inverse_transform(values).squeeze(-1).cpu().numpy())
            truth.append(data.scaler.inverse_transform(y).squeeze(-1).cpu().numpy())
            support.append(incident['distances'].abs().sum(-1).cpu().numpy()>0)
            if index==1 or index==data.loaders['val_loader'].num_batch:
                print(json.dumps(dict(arm=arm,phase='validation',batch=index,
                    total=data.loaders['val_loader'].num_batch)),flush=True)
    prediction,truth,support = map(np.concatenate,(prediction,truth,support))
    require(truth.dtype==baseline['y'].dtype and np.array_equal(truth,baseline['y'])
        and np.array_equal(support,baseline['direct_support']),'GW11 validation axes changed')
    require(prediction.dtype==np.float32 and prediction.shape==truth.shape
        and np.isfinite(prediction).all(),'Nonfinite GW11 validation prediction')
    if strict_source:
        require(prediction.dtype==baseline['pred__SUM_CONTROL'].dtype
            and np.array_equal(prediction,baseline['pred__SUM_CONTROL']),
            'GW11 stage zero differs from certified GW9 SUM BEST; exact replay is required')
    model.assert_cache_clear()
    if torch.device(device).type=='cuda': torch.cuda.synchronize(device)
    timing = dict(seconds=perf_counter()-start,batches=data.loaders['val_loader'].num_batch,
        complete_model_forwards=data.loaders['val_loader'].num_batch,
        scope='complete_validation_loader_H2D_forward_inverse_and_axis_checks')
    return record(prediction,truth,baseline['valid']),prediction,timing,model.diagnostics()


def train_epoch(model,optimizer,data,modules,device,arm,epoch,train_cache,seed,original_samples):
    order = epoch_order(seed,epoch,len(original_samples))
    data.loaders['train_loader'].samples = [original_samples[int(index)] for index in order]
    # Reset stochastic training execution identically for each arm and epoch.
    # This also makes replay of an uncommitted partial epoch deterministic.
    base.configure_runtime(device,torch.get_num_threads(),epoch_seed(seed,epoch))
    model.train(); model.start_diagnostics(); losses = []; position = 0; equivalent_null_batches = 0
    null = float(train_cache['y'].min()) if train_cache['y'].min()<1 else 0.
    if torch.device(device).type=='cuda':
        torch.cuda.synchronize(device); torch.cuda.reset_peak_memory_stats(device)
    start = perf_counter()
    try:
        for batch,raw in enumerate(data.loaders['train_loader'].get_iterator(),1):
            x,y,incident,sensors = device_batch(raw,device)
            target = data.scaler.inverse_transform(y)
            observed = target.squeeze(-1).detach().cpu().numpy()
            ids = order[position:position+len(observed)]
            require(observed.dtype==train_cache['y'].dtype
                and np.array_equal(observed,train_cache['y'][ids]),'Training target/order differs from certified cache')
            batch_null = float(target.min()) if target.min()<1 else 0.
            require(np.array_equal(observed!=batch_null,train_cache['valid'][ids]),
                'Actual training batch mask differs from the audited global rule; stop without changing masks')
            equivalent_null_batches += int(batch_null!=null)
            position += len(observed); optimizer.zero_grad(set_to_none=True)
            output = model(x,incident_data=incident,sensor_data=sensors)
            prediction = data.scaler.inverse_transform(output)
            require(bool(torch.isfinite(prediction).all()),'Nonfinite training prediction')
            loss = modules.metrics.masked_mae(prediction,target,target.new_tensor(batch_null))
            require(bool(torch.isfinite(loss)),'Nonfinite training loss')
            loss.backward()
            require(all(value.grad is None or bool(torch.isfinite(value.grad).all())
                for value in model.parameters()),'Nonfinite model gradients')
            if data.config['clip_grad_value']:
                torch.nn.utils.clip_grad_norm_(model.parameters(),data.config['clip_grad_value'])
            optimizer.step(); losses.append(float(loss.detach()))
            if batch==1 or batch%10==0 or batch==data.loaders['train_loader'].num_batch:
                print(json.dumps(dict(arm=arm,epoch=epoch,phase='training',batch=batch,
                    total=data.loaders['train_loader'].num_batch)),flush=True)
        require(position==len(original_samples) and len(losses)==data.loaders['train_loader'].num_batch,
            'Incomplete full-horizon training epoch')
        model.assert_cache_clear()
        if torch.device(device).type=='cuda': torch.cuda.synchronize(device)
        timing = dict(seconds=perf_counter()-start,batches=len(losses),
            complete_model_forwards=len(losses),full_model_backwards=len(losses),
            peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device))
                if torch.device(device).type=='cuda' else None,
            scope='full_model_training_with_target_and_mask_integrity_checks')
        return dict(mean_batch_train_mae=float(np.mean(losses)),mask_difference_points=0,
            different_native_null_with_identical_mask_batches=equivalent_null_batches,
            train_order_sha256=hashlib.sha256(order.astype('<i8').tobytes()).hexdigest(),
            dropout_epoch_seed=epoch_seed(seed,epoch),timing=timing,mechanisms=model.diagnostics())
    finally:
        data.loaders['train_loader'].samples = list(original_samples)


def validate_saved(saved, identity, arm, directory, batches):
    validate_progress(saved,identity,arm,batches)
    require(base.tensor_tree_hash(saved['model_state'])==saved['model_tensor_sha256'],
        'Saved model tensor hash differs')
    for field in ('optimizer_state','scheduler_state','rng_state'):
        require(base.tensor_tree_hash(saved[field])==saved.get(field+'_sha256'),
            'Saved optimizer/scheduler/RNG snapshot hash differs: '+field)
    require(saved['scheduler_state']['last_epoch']==saved['completed_epoch'],
        'Saved scheduler epoch differs')
    best = saved['best']; path = checkpoint_child(directory,best['checkpoint_file'])
    require(base.sha256(path)==best['checkpoint_sha256'],'Saved BEST bytes changed')
    weights = base.load_checkpoint(path)
    require(base.tensor_tree_hash(weights)==best['model_tensor_sha256'],'Saved BEST tensors changed')
    require(saved.get('full_horizon')==12,'Saved training horizon differs')


def commit_model(model, optimizer, scheduler, saved, directory, identity):
    saved.update(schema=SCHEMA,identity_sha256=identity['identity_sha256'],
        model_state=weights_copy(model),optimizer_state=cpu_tree(optimizer.state_dict()),
        scheduler_state=cpu_tree(scheduler.state_dict()),
        rng_state=cpu_tree(epoch_tools.rng_state()),full_horizon=12)
    saved['model_tensor_sha256'] = base.tensor_tree_hash(saved['model_state'])
    for field in ('optimizer_state','scheduler_state','rng_state'):
        saved[field+'_sha256'] = base.tensor_tree_hash(saved[field])
    epoch_tools.atomic_save(Path(directory)/'last.pt',saved)
    base.write_json(Path(directory)/'history.json',saved['history'])


def best_snapshot(model, directory, epoch, row):
    name = 'best_epoch_{:03d}_{}.pt'.format(epoch,uuid.uuid4().hex[:12])
    weights = weights_copy(model); epoch_tools.atomic_save(Path(directory)/name,weights)
    return dict(epoch=epoch,mae=row['mae'],checkpoint_file=name,
        checkpoint_sha256=base.sha256(Path(directory)/name),model_tensor_sha256=base.tensor_tree_hash(weights))


def training_marker(identity):
    require(identity.get('schema')==SCHEMA
        and identity_digest(identity)==identity.get('identity_sha256'),
        'Training marker requires the unchanged run identity')
    return dict(schema=TRAINING_MARKER_SCHEMA,training_schema=SCHEMA,
        identity_sha256=identity['identity_sha256'],registered_arms=list(ARMS),
        fixed_new_epochs=EPOCHS,all_stage_zero_checkpoints_committed=True,
        written_before_first_optimizer_update=True)


def validate_resume_checkpoint_set(directory, identity):
    """Do not recreate an arm after any persistent evidence of training.

    A missing checkpoint is recoverable only during partial initialization.
    The marker also protects the interval before the first committed epoch,
    when every surviving checkpoint can still truthfully report epoch zero.
    """
    directory = Path(directory)
    marker = directory/'training_started.json'
    marker_exists = marker.exists()
    if marker_exists:
        require(read_json(marker)==training_marker(identity),
            'Training-start marker differs from the bound experiment identity')
    checkpoints = {arm:directory/arm/'last.pt' for arm in ARMS}
    missing = [arm for arm,path in checkpoints.items() if not path.is_file()]
    positive_checkpoint = False
    for arm,path in checkpoints.items():
        if not path.is_file(): continue
        saved = base.load_checkpoint(path)
        require(saved.get('schema')==SCHEMA and saved.get('arm')==arm
            and saved.get('identity_sha256')==identity['identity_sha256'],
            'Resume checkpoint identity/arm differs: '+arm)
        done = saved.get('completed_epoch')
        require(type(done) is int and 0<=done<=EPOCHS,'Invalid resume checkpoint epoch: '+arm)
        positive_checkpoint = positive_checkpoint or done>0
        del saved
    ledger_evidence = False
    summary_path = directory/'summary.json'
    if summary_path.exists():
        ledger = read_json(summary_path)
        require(ledger.get('schema')==SCHEMA
            and ledger.get('identity_sha256')==identity['identity_sha256'],
            'Resume summary identity differs')
        for name in ('completed_epochs','updates'):
            values = ledger.get(name)
            require(isinstance(values,dict) and set(values)==set(ARMS)
                and all(type(value) is int and value>=0 for value in values.values()),
                'Resume summary has an invalid progress ledger: '+name)
            ledger_evidence = ledger_evidence or any(value>0 for value in values.values())
    for arm in ARMS:
        history_path = directory/arm/'history.json'
        if history_path.exists():
            history = read_json(history_path)
            require(isinstance(history,list),'Resume arm history must be a list: '+arm)
            ledger_evidence = ledger_evidence or bool(history)
    evidence = marker_exists or positive_checkpoint or ledger_evidence
    require(not evidence or not missing,
        'A previously started paired run is missing an arm checkpoint; preserve the run and recover: '
        +','.join(missing))
    return dict(training_evidence_detected=evidence,marker_present=marker_exists,missing_arms=missing)


def ensure_training_started(directory, identity, states):
    """Persist the immutable marker after all stage-zero saves, before updates."""
    directory = Path(directory); marker = directory/'training_started.json'
    expected = training_marker(identity)
    require(set(states)==set(ARMS)
        and all((directory/arm/'last.pt').is_file() for arm in ARMS),
        'All four stage-zero checkpoints must exist before training starts')
    if marker.exists():
        require(read_json(marker)==expected,
            'Training-start marker differs from the bound experiment identity')
        return expected
    # Never fabricate an allegedly pre-update marker after committed training.
    require(all(saved.get('completed_epoch')==0 and saved.get('global_updates')==0
        and saved.get('schema')==SCHEMA and saved.get('arm')==arm
        and saved.get('identity_sha256')==identity['identity_sha256']
        for arm,saved in states.items()),
        'A trained run is missing its original training-start marker; recover the marker')
    temporary = directory/('.training_started.'+uuid.uuid4().hex+'.tmp')
    try:
        with temporary.open('x',encoding='utf-8',newline='\n') as stream:
            stream.write(json.dumps(expected,sort_keys=True,indent=2,allow_nan=False)+'\n')
            stream.flush(); os.fsync(stream.fileno())
        require(not marker.exists(),'Training-start marker appeared concurrently')
        temporary.replace(marker)
        if os.name!='nt':
            descriptor = os.open(directory,os.O_RDONLY|getattr(os,'O_DIRECTORY',0))
            try: os.fsync(descriptor)
            finally: os.close(descriptor)
    finally:
        if temporary.exists(): temporary.unlink()
    require(read_json(marker)==expected,'Training-start marker failed its persisted replay')
    return expected


def summary(states, identity):
    return dict(schema=SCHEMA,status='GW11_TRAINING_COMPLETE' if all(
        value['completed_epoch']==EPOCHS for value in states.values()) else 'GW11_TRAINING_PARTIAL',
        identity_sha256=identity['identity_sha256'],fixed_new_epochs=EPOCHS,
        registered_arms=list(ARMS),completed_epochs={arm:value['completed_epoch'] for arm,value in states.items()},
        updates={arm:value['global_updates'] for arm,value in states.items()},
        best={arm:value['best'] for arm,value in states.items()},
        best_selection_includes_stage_zero=True,optimizer_reset_at_start=True,
        optimizer_reset_on_resume=False,
        rng_restore_on_resume=True,epoch_rng_reset_prescribed=True,
        same_source='certified_GW9_SUM_CONTROL_BEST_including_trained_SUM_head',
        test_evaluated=False,extension_allowed=False)


def final_report(run_dir, identity, states, construct, data, device, baseline, cache, gw10ref):
    require(all(value['completed_epoch']==EPOCHS for value in states.values()),'Complete all four fixed budgets first')
    output = run_dir/('evaluation_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'_'+uuid.uuid4().hex[:8])
    output.mkdir(); all_records = {}; provenance = {}
    old_tables = load_model(Path(gw10ref['directory'])/'model.json')
    deltas = predict_corrections(old_tables,cache['val']['features'],horizon=12)
    historical_predictions = {'FROZEN_SUM':baseline['pred__SUM_CONTROL'],
        **{arm:baseline['pred__SUM_CONTROL'].astype(np.float64)+deltas[arm] for arm in ('ORIGIN','OWN4')}}
    historical_records = {arm:record(value,baseline['y'],baseline['valid']) for arm,value in historical_predictions.items()}
    del deltas,old_tables
    for selection in ('best','last'):
        child = output/selection; child.mkdir(); predictions = {}; rows = {}; roles = {}; timings = {}
        for arm in ARMS:
            directory = run_dir/arm; saved = states[arm]
            validate_saved(saved,identity,arm,directory,data.loaders['train_loader'].num_batch)
            if selection=='best':
                role = saved['best']; path = checkpoint_child(directory,role['checkpoint_file'])
                weights = base.load_checkpoint(path); expected = saved['stage_zero'] if role['epoch']==0 else saved['history'][role['epoch']-1]['validation']
            else:
                path = directory/'last.pt'; weights = saved['model_state']; expected = saved['history'][-1]['validation']
                role = dict(epoch=EPOCHS,mae=expected['mae'],model_tensor_sha256=saved['model_tensor_sha256'],
                    checkpoint_file='last.pt',checkpoint_sha256=base.sha256(path))
            model = construct(arm); model.load_state_dict(weights,strict=True)
            row,prediction,timing,diagnostics = evaluate(model,data,device,baseline,arm,
                strict_source=role['epoch']==0)
            require(same_record(row,expected),'Selected BEST/LAST prediction failed metric replay: '+arm)
            rows[arm],predictions[arm],roles[arm] = row,prediction,dict(role,selection=selection)
            timings[arm] = dict(timing=timing,mechanisms=diagnostics)
            del model,weights; gc.collect()
            if torch.device(device).type=='cuda': torch.cuda.empty_cache()
        archive = {key:baseline[key] for key in ('y','valid','sample_ids','node_ids','road_labels','direct_support')}
        archive.update(schema=np.array('gw11_state_gate_prediction_archive_v1'),axis_order=np.array('sample,horizon,node'),
            **{'pred__'+arm:value for arm,value in {**historical_predictions,**predictions}.items()})
        np.savez_compressed(child/'paired_predictions.npz',**archive)
        metadata = dict(schema='gw11_state_gate_prediction_producer_v1',checkpoint_selection=selection,
            run_identity_sha256=identity['identity_sha256'],checkpoint_roles=roles,records=rows,
            historical_records=historical_records,benchmark_scope=timings,
            archive_sha256=base.sha256(child/'paired_predictions.npz'),test_evaluated=False,
            optimizer_updates_this_evaluation=0,common_completed_new_epochs=EPOCHS)
        base.write_json(child/'metadata.json',metadata); base.write_json(child/'metrics.json',{**historical_records,**rows})
        all_records[selection] = rows
        provenance[selection] = dict(directory=str(child),metadata_sha256=base.sha256(child/'metadata.json'),
            archive_sha256=metadata['archive_sha256'])
    decision = screen(all_records['best'],all_records['last'],{'OWN4':historical_records['OWN4']})
    decision.update(run_identity_sha256=identity['identity_sha256'],evaluations=provenance)
    base.write_json(output/'screening.json',decision)
    lines = ['# GW11 own-state internal gate development screen','',
        'Four arms inherit identical certified GW9 SUM BEST, including its trained SUM head. Three new epochs per arm.',
        'BEST includes stage zero. Lookup references are frozen historical controls, not equal-update training arms.',
        'The CPU mask audit passed; every actual new training batch was checked again. No test evaluation.','',
        '| Selection | Case | MAE | RMSE | MAPE % | Signed bias |','|---|---|---:|---:|---:|---:|']
    for selection in ('best','last'):
        for arm,row in {**historical_records,**all_records[selection]}.items():
            lines.append('| {} | {} | {:.9f} | {:.9f} | {:.9f} | {:.9f} |'.format(selection,arm,
                row['mae'],row['rmse'],100*row['mape_fraction'],row['signed_bias']))
    lines += ['','| Fixed check | Passed |','|---|---|']
    lines += ['| '+name+' | '+str(value)+' |' for name,value in decision['checks'].items()]
    lines += ['', 'Passing is a single-seed development screen, not statistical or independent confirmation.',
        'Failure does not authorize extra epochs, candidate changes or criterion changes.',
        'High-flow bias is descriptive; truth-defined high-flow groups are not inference inputs.','']
    (output/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    return output,decision


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('probe-dir','cache-dir','gw10-dir','audit-dir','data-dir','run-dir'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--device',default='cuda:0'); parser.add_argument('--threads',type=int,default=3)
    parser.add_argument('--seed',type=int,choices=(2025,),default=2025)
    parser.add_argument('--epochs',type=int,choices=(EPOCHS,),default=EPOCHS)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--resume',action='store_true'); mode.add_argument('--evaluate-only',action='store_true')
    args = parser.parse_args(argv)
    if args.threads<1: parser.error('threads must be positive')
    for name in ('probe_dir','cache_dir','gw10_dir','audit_dir','data_dir','run_dir'):
        setattr(args,name,getattr(args,name).resolve())
    return args


def main(argv=None):
    args = parse_args(argv); existing = args.resume or args.evaluate_only
    baseline,certificate,cache,cache_meta,cache_digests,gw10ref = load_inputs(args.probe_dir,args.cache_dir,args.gw10_dir)
    gate = validate_audit(args.audit_dir,certificate,cache_digests,gw10ref)
    # Check existing resume locations before run_lock can create active.lock.
    # The fresh-output helper also checks absence, so it cannot cover resume.
    protected = {args.probe_dir,args.cache_dir,args.gw10_dir,args.audit_dir,args.data_dir,
        *(Path(path).resolve() for path in certificate['protected_directories'])}
    for source in protected:
        require(args.run_dir!=source and source not in args.run_dir.parents
            and args.run_dir not in source.parents,'GW11 run output overlaps an immutable source: '+str(source))
    if not existing:
        require_output_outside_sources(args.run_dir,args.probe_dir,certificate,args.cache_dir,args.gw10_dir,args.data_dir)
        args.run_dir.mkdir(parents=True)
    else:
        require(args.run_dir.is_dir() and (args.run_dir/'identity.json').is_file(),
            'Resume/evaluate-only requires a run identity; keep partial initialization and use a fresh run if missing')
    with base.run_lock(args.run_dir):
        try:
            old_identity,gw4,recurrent = recertify(certificate['source_reference'])
            probe_identity = read_json(args.probe_dir/'identity.json')
            runtime = base.configure_runtime(args.device,args.threads,args.seed)
            for key in RUNTIME_KEYS:
                require(runtime.get(key)==old_identity['runtime'].get(key)==probe_identity['runtime'].get(key),
                    'Source/GW11 runtime differs: '+key)
            modules = base.get_original_modules()
            require(modules.source_fingerprint==old_identity['source_fingerprint'],'Original source differs')
            config = dict(base.CONFIG,device=args.device,seed=args.seed,warm_epoch=101)
            data = base.load_data(args.data_dir,config,modules,logging.getLogger('gw11'),check=False,training=True)
            require(data.identity==certificate['data_identity']==old_identity['data'],'GW11 data identity differs')
            require(np.array_equal(data.station_ids,baseline['node_ids']),'GW11 node axis differs')
            old_state = base.load_checkpoint(args.probe_dir/'last.pt')
            gw9_probe.validate_saved(old_state,probe_identity,args.probe_dir,data.loaders['train_loader'].num_batch)
            head,head_role = gw9_probe.selected_role(old_state,args.probe_dir,'SUM_CONTROL','best')
            require(head_role==certificate['head_roles']['best']['SUM_CONTROL'],'SUM BEST role differs')
            del old_state; gc.collect()
            def construct(arm):
                base.configure_runtime(args.device,args.threads,args.seed)
                return model_for_arm(data,modules,args.device,arm,gw4,recurrent,head)
            initial = {}
            for arm in ARMS:
                model = construct(arm); initial[arm] = initialization(model)
                require(base.tensor_tree_hash(model.source.anchor.state_dict())==certificate['source_reference'][
                    'source_initialization']['model_tensor_sha256']
                    and base.tensor_tree_hash(model.source.head.state_dict())==head_role['head_tensor_sha256'],
                    'Constructed source differs from certified complete SUM')
                require(initial[arm]['registered_parameters']==initial[arm]['trainable_parameters'],
                    'A joint arm contains unexpectedly frozen parameters')
                del model; gc.collect()
                if torch.device(args.device).type=='cuda': torch.cuda.empty_cache()
            require(len({row['source_tensor_sha256'] for row in initial.values()})==1
                and initial['SUM_CONTINUE']['added_trainable_parameters']==0
                and all(initial[arm]['added_trainable_parameters']==725 for arm in ARMS if arm!='SUM_CONTINUE')
                and len({initial[arm]['added_tensor_sha256'] for arm in ARMS if arm!='SUM_CONTINUE'})==1,
                'Initial source, added capacity or paired adapter tensors differ')
            identity = dict(schema=SCHEMA,arms=list(ARMS),candidate=CANDIDATE,new_epochs=EPOCHS,
                seed=args.seed,train_count=3909,train_batches=data.loaders['train_loader'].num_batch,
                source_certificate=certificate,cache_metadata_sha256=base.sha256(args.cache_dir/'cache.json'),
                gw10_reference=gw10ref,mask_gate=gate,data=data.identity,config=config,runtime=runtime,
                initializations=initial,execution_source=source_hashes(),full_horizon=12,
                order_rule='RandomState(seed+epoch)_shared_between_arms',
                epoch_rng_rule='seed+10000*epoch_reset_before_each_arm_epoch',
                optimizer_reset_at_start=True,best_selection_includes_stage_zero=True,
                test_evaluated=False,extension_allowed=False)
            identity['identity_sha256'] = identity_digest(identity)
            if existing:
                require(read_json(args.run_dir/'identity.json')==identity,'Cannot resume with changed protocol/source/runtime')
            else:
                base.write_json(args.run_dir/'identity.json',identity)
            original_samples = tuple(data.loaders['train_loader'].samples); states = {}
            if existing:
                validate_resume_checkpoint_set(args.run_dir,identity)
            for arm in ARMS:
                directory = args.run_dir/arm; directory.mkdir(exist_ok=True)
                if (directory/'last.pt').is_file():
                    require(existing,'Fresh run unexpectedly contains an arm checkpoint')
                    states[arm] = base.load_checkpoint(directory/'last.pt')
                    validate_saved(states[arm],identity,arm,directory,data.loaders['train_loader'].num_batch)
                    continue
                require(not args.evaluate_only,'All arm states are required for evaluate-only')
                model = construct(arm); optimizer,scheduler = make_optimizer(model,config)
                row,_,timing,diagnostics = evaluate(model,data,args.device,baseline,arm,strict_source=True)
                saved = dict(arm=arm,completed_epoch=0,global_updates=0,history=[],stage_zero=row,
                    stage_zero_timing=timing,stage_zero_mechanisms=diagnostics,initialization=initial[arm],
                    best=best_snapshot(model,directory,0,row))
                commit_model(model,optimizer,scheduler,saved,directory,identity); states[arm] = saved
                del model,optimizer,scheduler; gc.collect()
                if torch.device(args.device).type=='cuda': torch.cuda.empty_cache()
            if not args.evaluate_only:
                ensure_training_started(args.run_dir,identity,states)
                for epoch in range(1,EPOCHS+1):
                    for arm in next_work(states,epoch):
                        directory = args.run_dir/arm; saved = states[arm]
                        validate_saved(saved,identity,arm,directory,data.loaders['train_loader'].num_batch)
                        model = construct(arm); model.load_state_dict(saved['model_state'],strict=True)
                        optimizer,scheduler = make_optimizer(model,config)
                        optimizer.load_state_dict(saved['optimizer_state']); scheduler.load_state_dict(saved['scheduler_state'])
                        epoch_tools.restore_rng(saved['rng_state'])
                        trained = train_epoch(model,optimizer,data,modules,args.device,arm,epoch,
                            cache['train'],args.seed,original_samples)
                        row,_,timing,diagnostics = evaluate(model,data,args.device,baseline,arm)
                        scheduler.step()
                        saved['history'].append(dict(epoch=epoch,validation=row,validation_timing=timing,
                            validation_mechanisms=diagnostics,global_updates=epoch*data.loaders['train_loader'].num_batch,
                            learning_rate=float(optimizer.param_groups[0]['lr']),**trained))
                        saved.update(completed_epoch=epoch,global_updates=epoch*data.loaders['train_loader'].num_batch)
                        if row['mae']<saved['best']['mae']: saved['best'] = best_snapshot(model,directory,epoch,row)
                        commit_model(model,optimizer,scheduler,saved,directory,identity)
                        base.write_json(args.run_dir/'summary.json',summary(states,identity))
                        print(json.dumps(dict(status='GW11_ARM_EPOCH_COMMITTED',arm=arm,epoch=epoch,
                            validation=row,updates=saved['global_updates'])),flush=True)
                        del model,optimizer,scheduler; gc.collect()
                        if torch.device(args.device).type=='cuda': torch.cuda.empty_cache()
            result_dir,decision = final_report(args.run_dir,identity,states,construct,data,args.device,baseline,cache,gw10ref)
            require(identity['execution_source']==source_hashes(),'Execution source changed during GW11')
            assert_source_unchanged(certificate); assert_reference_unchanged(gw10ref)
            validate_audit(args.audit_dir,certificate,cache_digests,gw10ref)
            base.write_json(args.run_dir/'summary.json',summary(states,identity))
            base.write_json(args.run_dir/'screening.json',decision)
            base.write_json(args.run_dir/'result_pointer.json',dict(directory=str(result_dir),
                report_sha256=base.sha256(result_dir/'report.md'),screening_sha256=base.sha256(result_dir/'screening.json')))
            (args.run_dir/'report.md').write_text((result_dir/'report.md').read_text(encoding='utf-8'),encoding='utf-8')
            print(json.dumps(dict(status='GW11_EVALUATION_COMPLETE',output_dir=str(result_dir),test_evaluated=False)),flush=True)
            return decision
        except BaseException as error:
            base.write_json(args.run_dir/('failure_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')+'.json'),
                dict(status='GW11_FAILED',error_type=type(error).__name__,error=str(error),
                    resume='restore_last_committed_arm_epoch_replay_incomplete_epoch_no_budget_extension',test_evaluated=False))
            raise


if __name__=='__main__': main()
