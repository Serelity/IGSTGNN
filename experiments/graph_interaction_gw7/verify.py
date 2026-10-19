"""Server-only real-batch GW7 zero-recovery and gradient/resource preflight."""
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

from experiments.graph_interaction_gw7 import train
from experiments.graph_interaction_gw7.reference import certify,recertify,protected_paths,require
from experiments.chronological.original_igstgnn_runtime import device_batch


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gw6-run',type=Path,required=True)
    parser.add_argument('--gw6-best-eval',type=Path)
    parser.add_argument('--gw6-last-eval',type=Path)
    parser.add_argument('--data-dir',type=Path,default=train.base.default_data_dir())
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--batch-size',type=int,choices=(48,),default=48)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--threads',type=int,default=3)
    parser.add_argument('--seed',type=int,default=2025)
    parser.add_argument('--hidden',type=int,default=32)
    parser.add_argument('--node-chunk',type=int,default=32)
    args=parser.parse_args(argv); args.output=args.output.resolve()
    require(min(args.threads,args.hidden,args.node_chunk)>0,'Positive resource options required')
    require(not args.output.exists(),'Use a fresh GW7 verification filename')
    reference,old_identity,gw4_weights,recurrent_weights=certify(args.gw6_run,
        {role:getattr(args,'gw6_'+role+'_eval') for role in ('best','last')})
    for protected in protected_paths(reference):
        require(args.output!=protected and protected not in args.output.parents,'Do not write into source runs')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    evidence=dict(schema='gw7_real_batch_verification_v1',status='GW7_VERIFICATION_RUNNING',
        engineering_only=True,predictive_accuracy_evidence=False,real_data_optimizer_updates=0,
        test_evaluated=False,smaller_batch_fallback=False,source_reference=reference,
        execution_source={name:train.base.sha256(REPO/name) for name in
            (*train.EXECUTION_FILES,'experiments/graph_interaction_gw7/verify.py')})
    try:
        evidence['runtime']=train.base.configure_runtime(args.device,args.threads,args.seed)
        modules=train.base.get_original_modules()
        require(modules.source_fingerprint==old_identity['source_fingerprint'],'Native source differs')
        config=dict(train.base.CONFIG,bs=48,device=args.device,seed=args.seed,warm_epoch=101)
        data=train.base.load_data(args.data_dir,config,modules,logging.getLogger('gw7-check'),check=False,training=True)
        require(data.identity==old_identity['data'],'GW7 real-batch data identity differs')
        batch=device_batch(next(data.loaders['train_loader'].get_iterator()),args.device)
        x,y,incident,sensors=batch
        require(len(x)==48,'Full batch48 required; no smaller fallback')
        input_sha=train.base.tensor_tree_hash(batch)
        cuda=torch.device(args.device).type=='cuda'; evidence['arms']={}
        for arm in train.ARMS:
            train.base.configure_runtime(args.device,args.threads,args.seed)
            model=train.model_for_arm(data,modules,args.device,arm,gw4_weights,recurrent_weights,
                hidden=args.hidden,node_chunk=args.node_chunk)
            initial=train.initialization(model)
            require(initial['source_tensor_sha256']==reference['source_initialization']['model_tensor_sha256'],
                'R20 initialization changed')
            expected_model=deepcopy(model.anchor)
            # Align requires_grad, execution mode, precision and RNG for each
            # exact recovery assertion, including stochastic native dropout.
            exact={}
            for mode in (False,True):
                expected_model.train(mode); model.train(mode)
                rng=train.base.rng_state()
                with torch.no_grad(): expected=expected_model(x,incident_data=incident,sensor_data=sensors)
                train.epoch_tools.restore_rng(rng)
                with torch.no_grad(): actual=model(x,incident_data=incident,sensor_data=sensors)
                require(torch.equal(expected,actual),'Zero readout does not recover original in mode '+str(mode))
                exact['train' if mode else 'eval']=True
                model.assert_cache_clear(); expected_model.assert_cache_clear()
            del expected_model
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
            forward_backward_seconds=perf_counter()-started
            gradients={}
            for name,p in model.named_parameters():
                require(p.requires_grad,'GW7 unexpectedly froze native/added parameters: '+name)
                require(p.grad is None or bool(torch.isfinite(p.grad).all()),'Nonfinite task gradient: '+name)
                gradients[name]=None if p.grad is None else float(p.grad.detach().double().square().sum())
            native_forecast=[(name,value) for name,value in gradients.items()
                if name.startswith('anchor.') and '.dif_layer.forecast_branch.' in name]
            forecast_by_layer={}
            for index,layer in enumerate(model.original.layers):
                parameters=list(layer.dif_layer.forecast_branch.parameters())
                norm=sum(float(p.grad.detach().double().square().sum()) for p in parameters if p.grad is not None)
                forecast_by_layer[str(index)]=dict(squared_norm=norm,
                    parameters_with_gradient=sum(p.grad is not None for p in parameters))
            require(bool(native_forecast) and bool(forecast_by_layer)
                and all(row['squared_norm']>0 for row in forecast_by_layer.values()),
                'Task loss must reach every native DIF forecast branch')
            if arm!='R_CONTINUE':
                require((gradients.get('residual.output.weight') or 0)>0,'Task loss did not reach zero output projection')
            model.assert_cache_clear()
            evidence['arms'][arm]=dict(batch_size=48,horizon=12,initialization=initial,exact_zero_recovery=exact,
                loss=float(loss.detach()),gradients=gradients,diagnostics=model.diagnostics(),
                native_dif_forecast_receives_task_gradient=True,forward_backward_seconds=forward_backward_seconds,
                native_dif_forecast_gradient_by_layer=forecast_by_layer,
                timing_scope='complete_forward_inverse_native_MAE_backward_before_gradient_statistics',
                peak_allocated_bytes=int(torch.cuda.max_memory_allocated(args.device)) if cuda else None,
                peak_reserved_bytes=int(torch.cuda.max_memory_reserved(args.device)) if cuda else None,
                optimizer_updates=0,zero_initialized_output_may_block_reader_gradient_at_step_zero=True)
            print(json.dumps(dict(arm=arm,status='GW7_NATIVE_BACKWARD_COMPLETE',loss=float(loss.detach()))),flush=True)
            del model,prediction,target,loss; gc.collect()
            if cuda: torch.cuda.empty_cache()
        counts=evidence['arms']
        require(counts['R_MEMORY']['initialization']['trainable_parameters']==
            counts['R_INTERACTION_MEMORY']['initialization']['trainable_parameters'],
            'Memory active parameter counts differ')
        require(train.base.tensor_tree_hash(batch)==input_sha,'Preflight mutated inputs')
        recertify(reference)
        evidence.update(status='GW7_VERIFICATION_PASS',full_native_backward_checked=True)
        train.base.write_json(args.output,evidence)
        print(json.dumps(dict(status=evidence['status'],output=str(args.output))),flush=True)
        return evidence
    except BaseException as error:
        evidence.update(status='GW7_VERIFICATION_FAILED',error_type=type(error).__name__,error=str(error))
        train.base.write_json(args.output,evidence); raise


if __name__=='__main__': main()
