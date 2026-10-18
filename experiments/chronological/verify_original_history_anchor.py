"""Server-only native GW5 batch verification; zero real-data optimizer updates.

Run tests.test_history_anchor and tests.test_history_anchor_workflow first for
synthetic math, recovery, cache lifetime and interrupted-epoch checks.
"""
import argparse
import gc
import json
import logging
from pathlib import Path
import sys
from time import perf_counter

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import torch

from experiments.chronological import train_original_history_anchor as train
from experiments.chronological.history_anchor_reference import certify, recertify, require
from experiments.chronological.original_igstgnn_runtime import device_batch
from experiments.chronological.original_receiver_residual import split_state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gw4-run', type=Path, required=True)
    parser.add_argument('--gw4-best-eval', type=Path)
    parser.add_argument('--gw4-last-eval', type=Path)
    parser.add_argument('--data-dir', type=Path, default=train.base.default_data_dir())
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=48)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--threads', type=int, default=3)
    args = parser.parse_args(argv)
    if not 1 <= args.batch_size <= 3909:
        parser.error('batch-size must be in 1..3909')
    if args.output.exists():
        parser.error('Preserve verification evidence; use a fresh output filename')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    evidence = dict(schema='gw5_history_anchor_verification_v1', status='GW5_VERIFICATION_RUNNING',
        engineering_only=True, design=train.DESIGN, real_data_optimizer_updates=0,
        baseline_new_training_updates=0, baseline_new_inference_calls=0, test_evaluated=False,
        smaller_batch_fallback=False, execution_source={name:train.base.sha256(REPO/name)
            for name in (*train.EXECUTION_FILES, 'experiments/chronological/verify_original_history_anchor.py')})
    try:
        evidence['runtime'] = train.base.configure_runtime(args.device, args.threads)
        reference, old_identity, weights, _ = certify(args.gw4_run,
            {key:getattr(args, 'gw4_'+key+'_eval') for key in ('best', 'last')})
        evidence['gw4_reference'] = reference
        modules = train.base.get_original_modules()
        require(modules.source_fingerprint == old_identity['source_fingerprint'], 'Native source differs')
        config = dict(old_identity['training_config'], device=args.device, bs=args.batch_size)
        data = train.base.load_data(args.data_dir, config, modules, logging.getLogger('gw5-check'),
            check=False, training=True)
        batch = device_batch(next(data.loaders['train_loader'].get_iterator()), args.device)
        x, y, incident, sensors = batch
        require(len(x) == args.batch_size, 'Requested batch unavailable; no smaller fallback')
        input_hash, cuda = train.base.tensor_tree_hash(batch), torch.device(args.device).type == 'cuda'
        evidence['backward'] = {}
        for arm in train.ARMS:
            train.base.configure_runtime(args.device, args.threads, old_identity['training_config']['seed'])
            model = train.model_for_arm(data, modules, args.device, arm, weights)
            initial = train.initialization(model)
            for key in ('frozen_tensor_sha256', 'added_tensor_sha256', 'post_construction_rng_sha256',
                        'registered_parameters', 'trainable_parameters'):
                require(initial[key] == old_identity['initializations']['RECEIVER_RESIDUAL'][key],
                        'GW5/GW4 engineering initialization differs: '+key)
            model.train()
            model.zero_grad(set_to_none=True)
            model.start_diagnostics()
            if cuda:
                torch.cuda.synchronize(args.device)
                torch.cuda.reset_peak_memory_stats(args.device)
            started = perf_counter()
            prediction = data.scaler.inverse_transform(model(x, incident_data=incident, sensor_data=sensors))
            target = data.scaler.inverse_transform(y)
            null = target.min() if target.min() < 1 else target.new_tensor(0.)
            loss = modules.metrics.masked_mae(prediction, target, null)
            loss.backward()
            if cuda:
                torch.cuda.synchronize(args.device)
            seconds = perf_counter()-started
            require(bool(torch.isfinite(loss)), 'Nonfinite native GW5 objective')
            gradients = {}
            for name, parameter in model.named_parameters():
                if '.residual.' not in name:
                    require(not parameter.requires_grad and parameter.grad is None, 'Frozen parameter received a gradient')
                    continue
                require(parameter.grad is None or bool(torch.isfinite(parameter.grad).all()), 'Nonfinite GW5 gradient')
                gradients[name.rsplit('.residual.', 1)[1]] = dict(active=parameter.requires_grad,
                    squared_norm=float(parameter.grad.double().square().sum()) if parameter.grad is not None else None)
            require(gradients['key.weight']['squared_norm'] is not None and gradients['key.weight']['squared_norm'] > 0.,
                    'Native MAE did not reach the new key projection')
            require(not model.selector.residual._anchors and not model.selector.residual._active,
                    'History cache persisted after a native forward')
            require(train.base.tensor_tree_hash(split_state(model.state_dict())[0]) ==
                old_identity['reference']['source_model_tensor_sha256'], 'GW5 verification altered its GW2 anchor')
            mechanisms = model.diagnostics()
            require(set(mechanisms['policy_by_phase']) == {'history', 'forecast_1', 'forecast_2', 'forecast_3'},
                    'Native policy did not observe all four call phases')
            evidence['backward'][arm] = dict(batch_size=len(x), nodes=x.shape[2], loss=float(loss.detach()),
                initialization=initial, gradients=gradients, mechanisms=mechanisms,
                forward_backward_seconds=seconds,
                peak_allocated_bytes=int(torch.cuda.max_memory_allocated(args.device)) if cuda else None,
                timing_scope='new_candidate_forward_inverse_loss_backward_with_mechanism_diagnostics',
                real_data_optimizer_updates=0, new_candidate_forward_calls=1, baseline_forward_calls=0)
            print(json.dumps(dict(arm=arm, phase='native_backward_complete', loss=float(loss.detach()))), flush=True)
            del model, prediction, loss
            gc.collect()
            if cuda:
                torch.cuda.empty_cache()
        require(input_hash == train.base.tensor_tree_hash(batch), 'GW5 verification mutated input')
        recertify(reference)
        evidence.update(status='GW5_VERIFICATION_PASS', full_native_backward_checked=True,
            predictive_accuracy_evidence=False)
        train.base.write_json(args.output, evidence)
        print(json.dumps(dict(status=evidence['status'], full_native_backward_checked=True, output=str(args.output))), flush=True)
        return evidence
    except BaseException as error:
        evidence.update(status='GW5_VERIFICATION_FAILED', error_type=type(error).__name__, error=str(error))
        train.base.write_json(args.output, evidence)
        raise


if __name__ == '__main__':
    main()
