"""Server-only real-batch GW6 gradient and resource verification.

Synthetic unit tests precede this command. This verifies a requested real
batch without optimizer updates and never silently reduces batch size.
"""
import argparse
from copy import deepcopy
import gc
import json
import logging
from pathlib import Path
import sys
from time import perf_counter

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import torch

from experiments.graph_memory_gw6 import train
from experiments.graph_memory_gw6.model import ARMS, DEFAULT_ARMS
from experiments.graph_memory_gw6.reference import certify, recertify, require
from experiments.chronological.original_igstgnn_runtime import device_batch


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gw5-run', type=Path, required=True)
    parser.add_argument('--gw5-best-eval', type=Path)
    parser.add_argument('--gw5-last-eval', type=Path)
    parser.add_argument('--data-dir', type=Path, default=train.base.default_data_dir())
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=48)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--threads', type=int, default=3)
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--stage', choices=('joint','probe'), default='joint')
    parser.add_argument('--arms', nargs='+', choices=ARMS, default=list(DEFAULT_ARMS))
    parser.add_argument('--hidden', type=int, default=32)
    parser.add_argument('--node-chunk', type=int, default=32)
    parser.add_argument('--curriculum', choices=('full','native'), default='full')
    args = parser.parse_args(argv)
    if not 1 <= args.batch_size <= 3909 or min(args.threads,args.hidden,args.node_chunk) < 1:
        parser.error('Invalid batch size or nonpositive resource option')
    if len(args.arms) != len(set(args.arms)):
        parser.error('arms must be unique')
    if args.stage == 'probe' and 'RECURRENT_CONTROL' in args.arms:
        parser.error('For probe, explicitly omit RECURRENT_CONTROL from --arms')
    if args.output.exists():
        parser.error('Use a fresh verification output filename')
    return args


def main(argv=None):
    args = parse_args(argv)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    evidence = dict(schema='graph_memory_gw6_verification_v1', status='GW6_VERIFICATION_RUNNING',
        engineering_only=True, predictive_accuracy_evidence=False, stage=args.stage,
        real_data_optimizer_updates=0, test_evaluated=False, smaller_batch_fallback=False,
        execution_source={name:train.base.sha256(REPO/name) for name in
            (*train.EXECUTION_FILES,'experiments/graph_memory_gw6/verify.py')})
    try:
        runtime = train.base.configure_runtime(args.device,args.threads,args.seed)
        evidence['runtime'] = runtime
        reference, old_identity, weights, _ = certify(args.gw5_run,
            {key:getattr(args,'gw5_'+key+'_eval') for key in ('best','last')})
        evidence['source_reference'] = reference
        modules = train.base.get_original_modules()
        require(modules.source_fingerprint == old_identity['source_fingerprint'], 'Original source changed')
        config = dict(train.base.CONFIG, bs=args.batch_size,device=args.device,seed=args.seed)
        if args.curriculum == 'full':
            config['warm_epoch'] = 101
        data = train.base.load_data(args.data_dir,config,modules,logging.getLogger('gw6-check'),
                                   check=False,training=True)
        # Explicit engineering batch overrides may change only batching, never
        # the release, split rows, normalization or sensor semantics.
        comparable = deepcopy(data.identity)
        for split in comparable['splits']:
            for key in ('batch_size','partition'):
                comparable['splits'][split][key] = old_identity['data']['splits'][split][key]
        require(comparable == old_identity['data'], 'GW6 verification data identity differs from GW5')
        evidence['reference_batch_size'] = old_identity['data']['splits']['train']['batch_size']
        evidence['requested_batch_matches_training'] = (
            args.batch_size == evidence['reference_batch_size'])
        batch = device_batch(next(data.loaders['train_loader'].get_iterator()),args.device)
        x,y,incident,sensors = batch
        require(len(x) == args.batch_size, 'Requested batch unavailable; no automatic smaller batch')
        before_input = train.base.tensor_tree_hash(batch)
        expected_source_hash = train.base.tensor_tree_hash(weights)
        cuda = torch.device(args.device).type == 'cuda'
        evidence['arms'] = {}
        for arm in args.arms:
            train.base.configure_runtime(args.device,args.threads,args.seed)
            model = train.model_for_arm(data,modules,args.device,arm,weights,args.stage,
                hidden=args.hidden,node_chunk=args.node_chunk)
            initial = train.initialization(model)
            require(initial['source_tensor_sha256'] == expected_source_hash,'Initial anchor differs: '+arm)
            model.train()
            model.zero_grad(set_to_none=True)
            model.start_diagnostics()
            if cuda:
                torch.cuda.synchronize(args.device)
                torch.cuda.reset_peak_memory_stats(args.device)
            start = perf_counter()
            prediction = data.scaler.inverse_transform(model(x,incident_data=incident,sensor_data=sensors))
            target = data.scaler.inverse_transform(y)
            require(prediction.shape == target.shape,'GW6 target axis or output shape differs')
            null = target.min() if target.min() < 1 else target.new_tensor(0.)
            loss = modules.metrics.masked_mae(prediction,target,null)
            require(bool(torch.isfinite(loss)),'Nonfinite native MAE')
            loss.backward()
            if cuda:
                torch.cuda.synchronize(args.device)
            seconds = perf_counter()-start
            gradients = {}; active_nonzero = 0
            for name,p in model.named_parameters():
                require(p.grad is None or bool(torch.isfinite(p.grad).all()),'Nonfinite gradient: '+name)
                if not p.requires_grad:
                    require(p.grad is None,'Frozen parameter received gradient: '+name)
                squared = float(p.grad.detach().double().square().sum()) if p.grad is not None else None
                active_nonzero += int(p.requires_grad and squared is not None and squared > 0.)
                gradients[name] = dict(active=p.requires_grad,squared_norm=squared)
            require(active_nonzero > 0,'Task MAE reached no active parameter')
            if args.stage == 'probe':
                require(train.base.tensor_tree_hash(train.anchor_state(model)) == expected_source_hash,
                        'Probe changed frozen anchor parameters or buffers')
            if hasattr(model,'assert_cache_clear'):
                model.assert_cache_clear()
            diagnostics = model.diagnostics()
            evidence['arms'][arm] = dict(batch_size=len(x),nodes=x.shape[2],horizon=prediction.shape[1],
                loss=float(loss.detach()),initialization=initial,active_parameters_with_nonzero_gradient=active_nonzero,
                gradients=gradients,diagnostics=diagnostics,forward_backward_seconds=seconds,
                peak_allocated_bytes=int(torch.cuda.max_memory_allocated(args.device)) if cuda else None,
                peak_reserved_bytes=int(torch.cuda.max_memory_reserved(args.device)) if cuda else None,
                new_candidate_forward_calls=1,optimizer_updates=0,
                timing_scope='full_candidate_forward_inverse_MAE_backward_including_diagnostics',
                zero_initialized_readout_may_block_upstream_decoder_gradients_at_step_zero=True)
            print(json.dumps(dict(arm=arm,status='GW6_NATIVE_BACKWARD_COMPLETE',loss=float(loss.detach()))),flush=True)
            del model,prediction,target,loss,gradients
            gc.collect()
            if cuda:
                torch.cuda.empty_cache()
        require(train.base.tensor_tree_hash(batch) == before_input,'Verification mutated input data')
        recertify(reference)
        evidence.update(status='GW6_VERIFICATION_PASS',full_native_backward_checked=True)
        train.base.write_json(args.output,evidence)
        print(json.dumps(dict(status=evidence['status'],output=str(args.output.resolve()))),flush=True)
        return evidence
    except BaseException as error:
        evidence.update(status='GW6_VERIFICATION_FAILED',error_type=type(error).__name__,error=str(error))
        train.base.write_json(args.output,evidence)
        raise


if __name__ == '__main__':
    main()
