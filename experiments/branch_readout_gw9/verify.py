"""Server-only real batch48 recovery, gradient and frozen-probe preflight.

No optimizer step is taken. Temporary BatchNorm updates from joint train-mode
checks are discarded before the measured backward. Source runs stay read-only.
"""
import argparse
from copy import deepcopy
import gc
import json
import logging
from pathlib import Path
import sys
from time import perf_counter

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO))
import torch

from experiments.branch_readout_gw9 import train
from experiments.branch_readout_gw9.reference import certify,recertify,protected_paths,require
from experiments.chronological.original_igstgnn_runtime import device_batch


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gw8-run',type=Path,required=True)
    parser.add_argument('--gw8-best-eval',type=Path)
    parser.add_argument('--gw8-last-eval',type=Path)
    parser.add_argument('--data-dir',type=Path,default=train.base.default_data_dir())
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--batch-size',type=int,choices=(48,),default=48)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--threads',type=int,default=3)
    parser.add_argument('--seed',type=int,default=2025)
    parser.add_argument('--hidden',type=int,choices=(16,),default=16)
    args=parser.parse_args(argv); args.output=args.output.resolve()
    require(args.threads>0,'Positive thread count required')
    require(not args.output.exists(),'Use a fresh GW9 verification filename')
    reference,old_identity,gw4_weights,recurrent_weights=certify(args.gw8_run,
        {role:getattr(args,'gw8_'+role+'_eval') for role in ('best','last')})
    for protected in protected_paths(reference):
        require(args.output!=protected and protected not in args.output.parents,'Do not write into source runs')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    evidence=dict(schema='gw9_real_batch_verification_v1',status='GW9_VERIFICATION_RUNNING',
        engineering_only=True,predictive_accuracy_evidence=False,real_data_optimizer_updates=0,
        test_evaluated=False,smaller_batch_fallback=False,source_reference=reference,
        source_checkpoint_writes=0,temporary_training_mode_buffer_updates_discarded=True,
        execution_source={name:train.base.sha256(REPO/name) for name in
            (*train.EXECUTION_FILES,'experiments/branch_readout_gw9/verify.py')})
    try:
        evidence['runtime']=train.base.configure_runtime(args.device,args.threads,args.seed)
        modules=train.base.get_original_modules()
        require(modules.source_fingerprint==old_identity['source_fingerprint'],'Native source differs')
        config=dict(train.base.CONFIG,bs=48,device=args.device,seed=args.seed,warm_epoch=101)
        data=train.base.load_data(args.data_dir,config,modules,logging.getLogger('gw9-check'),check=False,training=True)
        require(data.identity==old_identity['data'],'GW9 real-batch data identity differs')
        batch=device_batch(next(data.loaders['train_loader'].get_iterator()),args.device)
        x,y,incident,sensors=batch
        require(len(x)==48,'Full batch48 required; no smaller fallback')
        input_sha=train.base.tensor_tree_hash(batch)
        cuda=torch.device(args.device).type=='cuda'; evidence['joint_arms']={}
        for arm in train.ARMS:
            train.base.configure_runtime(args.device,args.threads,args.seed)
            model=train.model_for_arm(data,modules,args.device,arm,gw4_weights,recurrent_weights,
                hidden=args.hidden,stage='joint')
            initial=train.initialization(model)
            require(initial['source_tensor_sha256']==reference['source_initialization']['model_tensor_sha256'],
                'GW8 recurrent BEST initialization changed')
            expected_model=deepcopy(model.anchor); exact={}
            for mode in (False,True):
                expected_model.train(mode); model.train(mode)
                rng=train.base.rng_state()
                with torch.no_grad(): expected=expected_model(x,incident_data=incident,sensor_data=sensors)
                expected_rng=train.base.tensor_tree_hash(train.base.rng_state())
                train.epoch_tools.restore_rng(rng)
                with torch.no_grad(): actual=model(x,incident_data=incident,sensor_data=sensors)
                require(torch.equal(expected,actual),'Zero readout does not recover native R in mode '+str(mode))
                require(train.base.tensor_tree_hash(train.base.rng_state())==expected_rng,
                    'Readout changed native forward RNG consumption in mode '+str(mode))
                exact['train' if mode else 'eval']=True
                model.assert_cache_clear(); expected_model.assert_cache_clear()
            del expected_model,expected,actual
            model.anchor.load_state_dict(recurrent_weights,strict=True)
            require(train.base.tensor_tree_hash(model.anchor.state_dict())==
                reference['source_initialization']['model_tensor_sha256'],
                'Preflight did not restore the certified anchor before backward')
            model.train(); model.zero_grad(set_to_none=True); model.start_diagnostics()
            if cuda:
                torch.cuda.synchronize(args.device); torch.cuda.reset_peak_memory_stats(args.device)
            started=perf_counter()
            prediction=data.scaler.inverse_transform(model(x,incident_data=incident,sensor_data=sensors))
            target=data.scaler.inverse_transform(y)
            require(prediction.shape==target.shape and prediction.shape[1]==12,'Actual horizon/output shape differs')
            null=target.min() if target.min()<1 else target.new_tensor(0.)
            loss=modules.metrics.masked_mae(prediction,target,null)
            require(bool(torch.isfinite(loss)),'Nonfinite task loss'); loss.backward()
            if cuda: torch.cuda.synchronize(args.device)
            seconds=perf_counter()-started
            gradients={}
            for name,p in model.named_parameters():
                require(p.requires_grad,'Joint stage unexpectedly froze parameters: '+name)
                require(p.grad is None or bool(torch.isfinite(p.grad).all()),'Nonfinite task gradient: '+name)
                gradients[name]=None if p.grad is None else float(p.grad.detach().double().square().sum())
            forecast={}
            for index,layer in enumerate(model.original.layers):
                parameters=list(layer.dif_layer.forecast_branch.parameters())
                forecast[str(index)]=dict(squared_norm=sum(float(p.grad.detach().double().square().sum())
                    for p in parameters if p.grad is not None),
                    parameters_with_gradient=sum(p.grad is not None for p in parameters))
            require(bool(forecast) and all(row['squared_norm']>0 for row in forecast.values()),
                'Task loss must reach every native DIF forecast branch')
            if arm!='R_CONTINUE':
                require(initial['added_trainable_parameters']==8256,
                    'Expected exactly 8256 trainable readout parameters')
                require(any(value is not None and value>0 for name,value in gradients.items()
                    if not name.startswith('anchor.')),'Task loss did not reach the zero readout')
            model.assert_cache_clear()
            evidence['joint_arms'][arm]=dict(batch_size=48,horizon=12,initialization=initial,
                exact_zero_recovery=exact,loss=float(loss.detach()),gradients=gradients,
                diagnostics=model.diagnostics(),native_dif_forecast_gradient_by_layer=forecast,
                forward_backward_seconds=seconds,
                timing_scope='complete_forward_inverse_native_MAE_backward_before_gradient_statistics',
                peak_allocated_bytes=int(torch.cuda.max_memory_allocated(args.device)) if cuda else None,
                peak_reserved_bytes=int(torch.cuda.max_memory_reserved(args.device)) if cuda else None,
                optimizer_updates=0,forward_rng_consumption_matches_anchor=True)
            print(json.dumps(dict(arm=arm,status='GW9_NATIVE_BACKWARD_COMPLETE',loss=float(loss.detach()))),flush=True)
            del model,prediction,target,loss; gc.collect()
            if cuda: torch.cuda.empty_cache()
        counts=evidence['joint_arms']
        require(len({counts[arm]['initialization']['trainable_parameters'] for arm in train.ARMS[1:]})==1
            and len({counts[arm]['initialization']['added_tensor_sha256'] for arm in train.ARMS[1:]})==1,
            'Active readout capacity or initial tensors differ')
        evidence['probe_arms']={}
        for arm in train.ARMS[1:]:
            train.base.configure_runtime(args.device,args.threads,args.seed)
            model=train.model_for_arm(data,modules,args.device,arm,gw4_weights,recurrent_weights,
                hidden=args.hidden,stage='probe')
            before=train.base.tensor_tree_hash(model.anchor.state_dict())
            model.train(); model.zero_grad(set_to_none=True)
            require(not model.anchor.training and all(not p.requires_grad for p in model.anchor.parameters()),
                'Probe must keep its entire anchor frozen in evaluation mode')
            prediction=data.scaler.inverse_transform(model(x,incident_data=incident,sensor_data=sensors))
            target=data.scaler.inverse_transform(y)
            null=target.min() if target.min()<1 else target.new_tensor(0.)
            loss=modules.metrics.masked_mae(prediction,target,null)
            require(bool(torch.isfinite(loss)),'Nonfinite frozen-probe task loss'); loss.backward()
            require(all(p.grad is None for p in model.anchor.parameters()),'Probe leaked gradients into anchor')
            require(any(p.grad is not None and bool(torch.isfinite(p.grad).all())
                and bool((p.grad!=0).any()) for p in model.head.parameters()),'Probe head receives no task gradient')
            require(train.base.tensor_tree_hash(model.anchor.state_dict())==before==
                reference['source_initialization']['model_tensor_sha256'],
                'Frozen probe mutated source tensors or BatchNorm buffers')
            model.assert_cache_clear()
            evidence['probe_arms'][arm]=dict(initialization=train.initialization(model),
                anchor_gradients_absent=True,anchor_buffers_unchanged=True,head_task_gradient=True,
                anchor_training=False,optimizer_updates=0,loss=float(loss.detach()))
            del model,prediction,target,loss; gc.collect()
            if cuda: torch.cuda.empty_cache()
        require(train.base.tensor_tree_hash(batch)==input_sha,'Preflight mutated inputs')
        recertify(reference)
        for name,digest in evidence['execution_source'].items():
            require(train.base.sha256(REPO/name)==digest,'Preflight source changed: '+name)
        evidence.update(status='GW9_VERIFICATION_PASS',full_native_backward_checked=True)
        train.base.write_json(args.output,evidence)
        print(json.dumps(dict(status=evidence['status'],output=str(args.output))),flush=True)
        return evidence
    except BaseException as error:
        evidence.update(status='GW9_VERIFICATION_FAILED',error_type=type(error).__name__,error=str(error))
        train.base.write_json(args.output,evidence); raise


if __name__=='__main__': main()
