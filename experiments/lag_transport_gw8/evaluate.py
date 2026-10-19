"""Authenticated paired GW8 validation; BEST/LAST and predeclared screen.

True-value groups are evaluation only. No test evaluation or optimizer steps.
"""
import argparse
import gc
import json
import logging
from pathlib import Path
import sys

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO))
import numpy as np
import torch

from experiments.lag_transport_gw8 import train
from experiments.lag_transport_gw8.reference import (
    recertify, protected_paths, read_json, require, verify_identity_digest)
from experiments.chronological import evaluate_original_graph_message as metrics
from experiments.chronological.graph_message_reuse import RUNTIME_KEYS
from experiments.graph_memory_gw6 import evaluate as history_eval

SCHEMA='original_igstgnn_lag_transport_gw8_archive_v1'
PRODUCER_SCHEMA='gw8_lag_transport_producer_v1'
HISTORICAL=('ORIGINAL','GW2_BASE','RECEIVER_RESIDUAL')
CASES=(*HISTORICAL,*train.ARMS)
AXIS_KEYS=('y','valid','sample_ids','node_ids','road_labels','direct_support')
PRODUCERS=('experiments/lag_transport_gw8/evaluate.py',
    'experiments/lag_transport_gw8/reference.py',
    'experiments/graph_memory_gw6/evaluate.py',
    'experiments/graph_memory_gw6/reference.py',
    'experiments/chronological/evaluate_original_graph_message.py',
    'experiments/chronological/evaluate_unified_adapter_v03.py')


def producer_source():
    return {name:metrics.sha256(REPO/name) for name in PRODUCERS}


def validate_archive(archive):
    require(np.array_equal(np.asarray(archive.get('registered_arms')),np.asarray(train.ARMS)),
        'Exactly the four predeclared GW8 arms are required')
    return metrics.validate_archive(archive,CASES,SCHEMA)


def require_matching_axes(left,right):
    for key in AXIS_KEYS:
        require(left[key].shape==right[key].shape and left[key].dtype==right[key].dtype
            and np.array_equal(left[key],right[key]),'BEST/LAST axes differ: '+key)


def replay_metrics(prediction,truth,state,epoch,modules):
    actual=metrics.native_metrics(prediction,truth,modules)
    row=state['stage_zero'] if epoch==0 else state['history'][epoch-1]
    expected={key:row['validation_'+key] for key in ('mae','rmse','mape_fraction')}
    passed=all(np.isclose(actual[key],value,atol=1e-5,rtol=1e-6) for key,value in expected.items())
    return dict(actual=actual,expected=expected,passed=bool(passed),atol=1e-5,rtol=1e-6)


def selection(directory,state,checkpoint,arm):
    if checkpoint=='best':
        path=directory/state['best_checkpoint_file']
        require(metrics.sha256(path)==state['best_checkpoint_sha256'],'GW8 BEST digest differs')
        weights,epoch=train.base.load_checkpoint(path),state['best_epoch']
    else:
        path,weights,epoch=directory/'last.pt',state['model_state'],state['completed_epoch']
    digest=metrics.sha256(path)
    role=dict(selection=checkpoint,epoch=epoch,policy=arm,checkpoint_sha256=digest,
        model_tensor_sha256=train.base.tensor_tree_hash(weights),historical=False)
    require(metrics.sha256(path)==digest,'Selected checkpoint changed while reading')
    return weights,role


def certified_historical_replay(reference,checkpoint,values,metadata):
    """Recover GW7's omitted historical replay from its certified GW6 cache.

    GW7 copied historical predictions and roles but emitted validation_replay
    only for its new arms. Do not repair the immutable GW7 metadata or infer a
    passing replay: require the actual ancestor record and identical arrays.
    """
    ancestor=reference['gw7_source_reference']
    saved=ancestor['archives'][checkpoint]; directory=Path(saved['directory'])
    metadata_path=directory/'metadata.json'; archive_path=directory/'paired_predictions.npz'
    require(metrics.sha256(metadata_path)==saved['metadata_sha256'],
        'Historical GW6 metadata changed')
    source=read_json(metadata_path)
    require(source.get('schema')==history_eval.PRODUCER_SCHEMA
        and source.get('status')=='GW6_PREDICTION_PASS'
        and source.get('run_identity_sha256')==ancestor['source_identity_sha256']
        and source.get('source_reference')==ancestor['gw6_source_reference']
        and source.get('checkpoint_selection')==checkpoint
        and source.get('producer_source')==history_eval.producer_source()
        and source.get('engineering_only') is False
        and source.get('test_evaluated') is False,
        'Historical GW6 evaluation certificate differs')
    require(metrics.sha256(archive_path)==saved['archive_sha256']
        and source.get('archive_sha256')==saved['archive_sha256'],
        'Historical GW6 archive changed')
    with np.load(archive_path,allow_pickle=False) as payload:
        require(str(np.asarray(payload['schema']).item())==history_eval.SCHEMA
            and str(np.asarray(payload['axis_order']).item())=='sample,horizon,node',
            'Historical GW6 archive schema or axes differ')
        ancestor_values={key:payload[key]
            for key in (*AXIS_KEYS,*('pred__'+name for name in HISTORICAL))}
    require_matching_axes(values,ancestor_values)
    provenance=dict(directory=str(directory.resolve()),
        metadata_sha256=saved['metadata_sha256'],archive_sha256=saved['archive_sha256'],
        checkpoint_selection=checkpoint,source_run_identity_sha256=ancestor['source_identity_sha256'])
    replays={}
    for name in HISTORICAL:
        key='pred__'+name; current,previous=values[key],ancestor_values[key]
        require(current.shape==previous.shape and current.dtype==previous.dtype
            and np.array_equal(current,previous),
            'Historical GW7/GW6 predictions differ: '+name)
        role=source.get('checkpoint_roles',{}).get(name)
        require(isinstance(role,dict) and metadata.get('checkpoint_roles',{}).get(name)
            ==dict(role,historical=True,reused=True),
            'Historical GW7/GW6 checkpoint role differs: '+name)
        replay=source.get('validation_replay',{}).get(name)
        require(isinstance(replay,dict) and replay.get('passed') is True,
            'Missing or failed certified historical GW6 validation replay: '+name)
        existing=metadata.get('validation_replay',{}).get(name)
        require(existing is None or (isinstance(existing,dict) and existing.get('passed') is True),
            'Conflicting historical GW7 validation replay: '+name)
        replays[name]=dict(replay,mode='certified_historical_cache_no_new_forward',
            equal_new_training_budget=False,certification_source=dict(provenance))
    require(metrics.sha256(metadata_path)==saved['metadata_sha256'],
        'Historical GW6 metadata changed during read')
    require(metrics.sha256(archive_path)==saved['archive_sha256'],
        'Historical GW6 archive changed during read')
    return replays,provenance


def load_historical(reference,checkpoint):
    saved=reference['archives'][checkpoint]; directory=Path(saved['directory'])
    require(metrics.sha256(directory/'metadata.json')==saved['metadata_sha256'],'GW7 metadata changed')
    metadata=read_json(directory/'metadata.json'); path=directory/'paired_predictions.npz'
    require(metadata.get('checkpoint_selection')==checkpoint
        and metadata.get('archive_sha256')==saved['archive_sha256'],
        'GW7 historical cache selection or archive certificate differs')
    require(metrics.sha256(path)==saved['archive_sha256'],'GW7 cached predictions changed')
    with np.load(path,allow_pickle=False) as payload:
        values={key:payload[key] for key in (*AXIS_KEYS,*('pred__'+name for name in HISTORICAL))}
    require(metrics.sha256(path)==saved['archive_sha256'],'GW7 cache changed during read')
    replays,provenance=certified_historical_replay(reference,checkpoint,values,metadata)
    # The copy is an in-memory view for GW8 only. Source files stay immutable.
    metadata=dict(metadata,validation_replay={**metadata.get('validation_replay',{}),**replays},
        historical_replay_provenance=provenance)
    require(metrics.sha256(directory/'metadata.json')==saved['metadata_sha256'],
        'GW7 metadata changed during historical replay certification')
    require(metrics.sha256(path)==saved['archive_sha256'],
        'GW7 cache changed during historical replay certification')
    return values,metadata


def produce(args,output):
    identity_file_sha256=metrics.sha256(args.run_dir/'identity.json')
    identity=read_json(args.run_dir/'identity.json'); verify_identity_digest(identity)
    require(identity.get('schema')==train.SCHEMA and identity.get('design')==train.DESIGN
        and identity.get('arms')==list(train.ARMS),'GW8 training identity required')
    for name in train.EXECUTION_FILES:
        require(identity['execution_source'].get(name)==metrics.sha256(REPO/name),'GW8 training source changed: '+name)
    initial_producer_source=producer_source()
    reference=identity['source_reference']
    old_identity,gw4_weights,recurrent_weights=recertify(reference)
    require(identity['data']==old_identity['data'],'GW8 source data identity differs')
    require(read_json(args.run_dir/'budget.json')['target_completed_epochs']==train.SCREEN_EPOCHS,
        'GW8 fixed screening budget differs')
    states,last_sha={},{}
    batches=len(identity['data']['splits']['train']['partition'])
    for arm in train.ARMS:
        directory=args.run_dir/arm; last_sha[arm]=metrics.sha256(directory/'last.pt')
        state=train.base.load_checkpoint(directory/'last.pt')
        train.validate_saved(state,identity,arm,directory,batches)
        require(state['completed_epoch']==train.SCREEN_EPOCHS,'Finish every GW8 arm at five epochs')
        require(metrics.sha256(directory/'last.pt')==last_sha[arm],'GW8 LAST changed during read')
        states[arm]=state
    device=args.device or identity['runtime']['python_device']
    runtime=metrics.configure_runtime(device,identity['runtime']['threads'],identity['training_config']['seed'])
    for key in RUNTIME_KEYS:
        require(runtime.get(key)==identity['runtime'].get(key),'GW8 evaluation runtime differs: '+key)
    modules=metrics.get_original_modules()
    require(modules.source_fingerprint==identity['source_fingerprint'],'Native source differs')
    data=metrics.load_data(args.data_dir,identity['training_config'],modules,logging.getLogger('gw8-eval'),
        check=False,training=True)
    require(data.identity==identity['data'],'Actual evaluation data differs')
    archive,historical_metadata=load_historical(reference,args.checkpoint)
    require(np.array_equal(archive['sample_ids'],identity['data']['splits']['val']['selected_indices'])
        and np.array_equal(archive['node_ids'],data.station_ids),'Historical sample/node axes differ')
    roads=np.full(len(data.station_ids),'unknown',dtype='<U40')
    for road,indices in data.roads.items(): roads[indices]=road
    require(np.array_equal(archive['road_labels'],roads),'Road axis differs')
    archive.update(schema=np.array(SCHEMA),axis_order=np.array('sample,horizon,node'),
        registered_arms=np.asarray(train.ARMS,dtype=str))
    replay,roles,mechanisms,benchmarks={},{},{},{}
    for name in HISTORICAL:
        require(historical_metadata['validation_replay'][name]['passed'] is True,
            'Missing historical validation replay')
        roles[name]=dict(historical_metadata['checkpoint_roles'][name],historical=True,reused=True)
        replay[name]=dict(historical_metadata['validation_replay'][name])
    for arm,state in states.items():
        weights,roles[arm]=selection(args.run_dir/arm,state,args.checkpoint,arm)
        model=train.model_for_arm(data,modules,device,arm,gw4_weights,recurrent_weights,**identity['model_options'])
        model.load_state_dict(weights,strict=True); model.eval(); model.start_diagnostics()
        predictions,truths,supports=[],[],[]
        with torch.no_grad():
            for index,raw in enumerate(data.loaders['val_loader'].get_iterator(),1):
                x,y,incident,sensors=metrics.device_batch(raw,device)
                prediction=model(x,incident_data=incident,sensor_data=sensors)
                model.assert_cache_clear()
                predictions.append(data.scaler.inverse_transform(prediction).squeeze(-1).cpu().numpy())
                truths.append(data.scaler.inverse_transform(y).squeeze(-1).cpu().numpy())
                supports.append(incident['distances'].abs().sum(-1).cpu().numpy()>0)
                if index==1 or index%10==0 or index==data.loaders['val_loader'].num_batch:
                    print(json.dumps(dict(case=arm,batch=index,total=data.loaders['val_loader'].num_batch)),flush=True)
        prediction,truth,support=map(np.concatenate,(predictions,truths,supports))
        require(np.array_equal(truth,archive['y']) and np.array_equal(support,archive['direct_support']),
            'Actual shared target/support axes differ')
        archive['pred__'+arm]=prediction
        replay[arm]=replay_metrics(prediction,truth,state,roles[arm]['epoch'],modules)
        require(replay[arm]['passed'],'GW8 native metric replay failed: '+arm)
        mechanisms[arm]=model.diagnostics()
        # Separately timed full forwards, never mislabeled training duration.
        benchmarks[arm]=metrics.benchmark(model,metrics.device_batch(
            next(data.loaders['val_loader'].get_iterator()),device),device)
        model.assert_cache_clear()
        del model,weights; gc.collect()
        if torch.device(device).type=='cuda': torch.cuda.empty_cache()
    validate_archive(archive)
    for arm in train.ARMS:
        require(metrics.sha256(args.run_dir/arm/'last.pt')==last_sha[arm],'GW8 LAST changed during evaluation')
        selected=args.run_dir/arm/(states[arm]['best_checkpoint_file'] if args.checkpoint=='best' else 'last.pt')
        require(metrics.sha256(selected)==roles[arm]['checkpoint_sha256'],'GW8 selected weights changed')
    recertify(reference)
    require(initial_producer_source==producer_source(),'Evaluation producer source changed during inference')
    for name in train.EXECUTION_FILES:
        require(identity['execution_source'].get(name)==metrics.sha256(REPO/name),'GW8 source changed during inference')
    require(metrics.sha256(args.run_dir/'identity.json')==identity_file_sha256,
        'GW8 identity changed during inference')
    metadata=dict(schema=PRODUCER_SCHEMA,status='GW8_PREDICTION_PASS',engineering_only=False,
        run_identity_sha256=identity['identity_sha256'],source_reference=reference,
        checkpoint_selection=args.checkpoint,checkpoint_roles=roles,registered_arms=list(train.ARMS),
        common_completed_new_epochs=train.SCREEN_EPOCHS,best_selection_includes_stage_zero=True,
        validation_replay=replay,mechanism_summary=mechanisms,benchmarks=benchmarks,
        initializations=identity['initializations'],runtime=runtime,producer_source=producer_source(),
        historical_replay_provenance=historical_metadata['historical_replay_provenance'],
        historical_new_inference_calls=0,historical_equal_new_budget_controls=False,
        optimizer_updates=0,test_evaluated=False)
    path=output/'paired_predictions.npz'; np.savez_compressed(path,**archive)
    metadata['archive_sha256']=metrics.sha256(path)
    metrics.write_json(output/'metadata.json',metadata)
    return archive,metadata


def group_metrics(prediction,y,valid):
    result={}
    for name,mask in (('high_value_ge314',valid & (y>=314)),
        ('long_horizon_7_12',valid & (np.arange(12)[None,:,None]>=6))):
        count=int(mask.sum())
        result[name]=dict(valid_count=count,mae=float(np.abs(
            prediction[mask].astype(np.float64)-y[mask].astype(np.float64)).mean()) if count else None)
    return result


def report(archive,output,metadata):
    y,valid,predictions=validate_archive(archive)
    cases={}; groups={}; horizons=[]; comparisons={}
    for arm,prediction in predictions.items():
        row=metrics.point_metrics(prediction,y,valid)
        error=prediction[valid].astype(np.float64)-y[valid].astype(np.float64)
        row['signed_bias']=float(error.mean())
        scores,complete=metrics.window_scores(prediction,y,valid)
        row['window_p95']=float(np.quantile(scores[complete],.95)) if complete.any() else None
        row['historical_reference']=arm in HISTORICAL
        cases[arm]=row; groups[arm]=group_metrics(prediction,y,valid)
        for horizon in row['horizons']:
            h=horizon['step']-1; mask=valid[:,h]
            horizon['signed_bias']=float((prediction[:,h][mask].astype(np.float64)-y[:,h][mask]).mean()) if mask.any() else None
            horizons.append(dict(case=arm,**horizon))
    hard={name:metrics.fixed_hard_indices(predictions[name],y,valid,archive['sample_ids'])
        for name in ('GW2_BASE','RECEIVER_RESIDUAL')}
    for control in train.ARMS[:-1]:
        candidate='R_JOINT_KERNEL'; key=candidate+'_vs_'+control
        pair=metrics.paired_metrics(predictions[control],predictions[candidate],y,valid)
        left,_=metrics.window_scores(predictions[control],y,valid)
        right,_=metrics.window_scores(predictions[candidate],y,valid)
        pair.update({name+'_fixed_hard_gain':float((left[index]-right[index]).mean()) if len(index) else None
            for name,index in hard.items()})
        comparisons[key]=pair
    result=dict(schema='gw8_lag_transport_evaluation_v1',status='GW8_EVALUATION_COMPLETE',
        validation_rows=len(y),valid_points=int(valid.sum()),cases=cases,groups=groups,comparisons=comparisons,
        fixed_hard_ids={key:archive['sample_ids'][index].tolist() for key,index in hard.items()},
        producer_metadata=metadata,engineering_only=metadata.get('engineering_only',True),
        mask_shared=True,test_evaluated=False,optimizer_updates=0,
        interpretation='descriptive_development_validation_not_independent_confirmation')
    metrics.write_json(output/'analysis.json',result)
    metrics._write_csv(output/'cases.csv',[dict(case=key,**{k:v for k,v in row.items() if k!='horizons'}) for key,row in cases.items()])
    metrics._write_csv(output/'horizons.csv',horizons)
    metrics._write_csv(output/'groups.csv',[dict(case=arm,group=group,**row) for arm,items in groups.items() for group,row in items.items()])
    metrics._write_csv(output/'comparisons.csv',[dict(comparison=key,**{k:v for k,v in row.items()
        if k not in ('horizons','directions')}) for key,row in comparisons.items()])
    fmt=lambda v:'NA' if v is None else f'{v:.9f}'
    lines=['# GW8 paired development validation','',
        'All four GW8 arms start from identical certified GW7 R_CONTINUE BEST tensors and receive five new epochs. BEST includes stage zero.',
        'Historical rows reuse immutable caches and are not new-budget controls. No test evaluation.','',
        '| Case | MAE | RMSE | MAPE % | Window P95 | Signed bias |',
        '|---|---:|---:|---:|---:|---:|']
    for arm,row in cases.items():
        lines.append('| '+arm+' | '+' | '.join(fmt(v) for v in (row['mae_pooled'],row['rmse_pooled'],
            100*row['mape_pooled'],row['window_p95'],row['signed_bias']))+' |')
    lines+=['','High-value >=314 and horizon 7-12 are evaluation-only groups; see groups.csv.',
        'Fixed historical hard windows are reported for both GW2 and GW4; they do not change with the candidate.',
        'Run the explicit BEST/LAST screening command to apply the locked criteria. Passing is not statistical confirmation.',
        'Synchronized complete-forward timing uses two warmups and three measured forwards on the same first batch; see metadata benchmarks.',
        'The development validation set has repeatedly informed design. Independent test evaluation requires a frozen method.','']
    (output/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    return result


def screening(best,last):
    candidate='R_JOINT_KERNEL'
    b,l=best['cases'],last['cases']
    value=b[candidate]['mae_pooled']
    checks={
        'BEST_relative_gain_vs_R_at_least_0_005':(b['R_CONTINUE']['mae_pooled']-value)/b['R_CONTINUE']['mae_pooled']>=.005,
        'BEST_relative_gain_vs_SEPARABLE_at_least_0_005':(b['R_SEPARABLE_KERNEL']['mae_pooled']-value)/b['R_SEPARABLE_KERNEL']['mae_pooled']>=.005,
        'BEST_beats_LOCAL':value<b['R_LOCAL_KERNEL']['mae_pooled'],
        'LAST_own_degradation_at_most_0_01':(l[candidate]['mae_pooled']-value)/value<=.01,
    }
    for group in ('high_value_ge314','long_horizon_7_12'):
        a=best['groups'][candidate][group]['mae']; r=best['groups']['R_CONTINUE'][group]['mae']
        checks['BEST_'+group+'_no_worse_than_R']=a is not None and r is not None and a<=r
    return dict(status='GW8_SCREENING_COMPLETE',checks=checks,
        candidate_for_confirmation=all(checks.values()),
        LAST_candidate_MAE=l[candidate]['mae_pooled'],
        LAST_relative_gains={arm:(l[arm]['mae_pooled']-l[candidate]['mae_pooled'])/l[arm]['mae_pooled']
            for arm in train.ARMS[:-1]},
        high_value_threshold=314,long_horizons=[7,8,9,10,11,12],
        interpretation='predeclared_single_seed_development_screen_not_statistical_significance',
        screening_failure_does_not_authorize_budget_extension=True,test_evaluated=False)


def authenticated_analysis(directory,identity,selection_role,run):
    directory=Path(directory).resolve(); metadata=read_json(directory/'metadata.json')
    metadata_sha256=metrics.sha256(directory/'metadata.json')
    require(metadata.get('schema')==PRODUCER_SCHEMA and metadata.get('status')=='GW8_PREDICTION_PASS'
        and metadata.get('checkpoint_selection')==selection_role
        and metadata.get('run_identity_sha256')==identity['identity_sha256']
        and metadata.get('source_reference')==identity['source_reference']
        and metadata.get('producer_source')==producer_source()
        and metadata.get('engineering_only') is False
        and metadata.get('common_completed_new_epochs')==train.SCREEN_EPOCHS,
        'Screening requires matching independently authenticated BEST/LAST evaluations')
    require(metrics.sha256(directory/'paired_predictions.npz')==metadata['archive_sha256'],'Screening archive changed')
    batches=len(identity['data']['splits']['train']['partition'])
    for arm in train.ARMS:
        state=train.base.load_checkpoint(run/arm/'last.pt')
        train.validate_saved(state,identity,arm,run/arm,batches)
        require(state['completed_epoch']==train.SCREEN_EPOCHS,'Screening requires every arm at five epochs')
        weights,role=selection(run/arm,state,selection_role,arm)
        require(metadata.get('checkpoint_roles',{}).get(arm)==role
            and metadata.get('validation_replay',{}).get(arm,{}).get('passed') is True,
            'Screening checkpoint role or native replay differs: '+arm)
        del state,weights
    # Recompute report inputs from the certified archive: an edited analysis.json
    # must never convert a failed screen into a pass.
    with np.load(directory/'paired_predictions.npz',allow_pickle=False) as payload:
        archive=dict(payload)
    y,valid,predictions=validate_archive(archive)
    cases={arm:metrics.point_metrics(predictions[arm],y,valid) for arm in train.ARMS}
    groups={arm:group_metrics(predictions[arm],y,valid) for arm in train.ARMS}
    require(metrics.sha256(directory/'paired_predictions.npz')==metadata['archive_sha256'],'Screening archive changed while reading')
    require(metrics.sha256(directory/'metadata.json')==metadata_sha256,'Screening metadata changed while reading')
    axes={key:archive[key] for key in AXIS_KEYS}
    return dict(cases=cases,groups=groups),metadata,axes


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--checkpoint',choices=('best','last'),default='best')
    parser.add_argument('--data-dir',type=Path,default=metrics.default_data_dir())
    parser.add_argument('--device')
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--screen',action='store_true')
    parser.add_argument('--best-eval',type=Path)
    parser.add_argument('--last-eval',type=Path)
    args=parser.parse_args(argv); args.run_dir=args.run_dir.resolve(); args.output_dir=args.output_dir.resolve()
    identity=read_json(args.run_dir/'identity.json'); verify_identity_digest(identity)
    require(identity.get('schema')==train.SCHEMA,'Expected GW8 identity')
    for protected in protected_paths(identity['source_reference']):
        require(args.output_dir!=protected and protected not in args.output_dir.parents,'Do not write into immutable historical runs')
    require(args.output_dir.parent==args.run_dir,'Evaluation output must be a new direct child of the GW8 run')
    require(not args.output_dir.exists(),'Use a fresh GW8 output directory')
    args.output_dir.mkdir()
    try:
        if args.screen:
            require(args.best_eval is not None and args.last_eval is not None,'Supply both --best-eval and --last-eval')
            recertify(identity['source_reference'])
            for name in train.EXECUTION_FILES:
                require(identity['execution_source'].get(name)==metrics.sha256(REPO/name),'GW8 source changed: '+name)
            best,bm,baxes=authenticated_analysis(args.best_eval,identity,'best',args.run_dir)
            last,lm,laxes=authenticated_analysis(args.last_eval,identity,'last',args.run_dir)
            require_matching_axes(baxes,laxes)
            result=screening(best,last)
            result.update(run_identity_sha256=identity['identity_sha256'],
                best_metadata_sha256=metrics.sha256(args.best_eval/'metadata.json'),
                last_metadata_sha256=metrics.sha256(args.last_eval/'metadata.json'))
            metrics.write_json(args.output_dir/'screening.json',result)
            (args.output_dir/'report.md').write_text('# GW8 development screening\n\n```json\n'+
                json.dumps(result,ensure_ascii=False,indent=2)+'\n```\n',encoding='utf-8')
        else:
            archive,metadata=produce(args,args.output_dir)
            result=report(archive,args.output_dir,metadata)
        print(json.dumps(dict(status=result['status'],output_dir=str(args.output_dir))),flush=True)
        return result
    except BaseException as error:
        metrics.write_json(args.output_dir/'failure.json',dict(status='GW8_EVALUATION_FAILED',
            error_type=type(error).__name__,error=str(error),optimizer_updates=0,test_evaluated=False))
        raise


if __name__=='__main__': main()
