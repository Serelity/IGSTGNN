"""GW2 engineering recovery and native-MAE gradients; zero optimizer updates.

--check: four real release rows, batch two, train/eval native recovery for all
arms, then real-MAE backward. --stress: one exact requested batch for all four
arms, measured separately. OOM never triggers a smaller fallback.
"""
import argparse
import copy
import gc
import json
import logging
import os
from pathlib import Path
import sys
from time import perf_counter

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.chronological.original_igstgnn_source import get_original_modules
from experiments.chronological.original_igstgnn_runtime import (
    CONFIG, build_model, configure_runtime, default_data_dir, device_batch,
    load_data, rng_fingerprint, sha256, tensor_tree_hash, write_json)
from experiments.chronological.original_graph_message import ARMS, DESIGN, build_variant, split_state


def synchronize(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


def recovery(original, batch):
    x, _, report, sensors = batch
    results = {}
    before = tensor_tree_hash(batch)
    for training in (False, True):
        native = copy.deepcopy(original).train(training)
        torch.manual_seed(440+int(training))
        with torch.no_grad():
            reference = native(x, incident_data=report, sensor_data=sensors)
        reference_rng = rng_fingerprint()
        rows = {}
        for arm in ARMS[1:]:
            model = build_variant(copy.deepcopy(original), arm).train(training)
            torch.manual_seed(440+int(training))
            with torch.no_grad():
                prediction = model(x, incident_data=report, sensor_data=sensors)
            rows[arm] = dict(prediction_exact=bool(torch.equal(reference, prediction)),
                prediction_close=bool(torch.allclose(reference, prediction, atol=2e-5, rtol=2e-5)),
                abs_max=float((reference-prediction).abs().max()),
                atol=2e-5, rtol=2e-5,
                rng_exact=reference_rng == rng_fingerprint(),
                native_state_exact=tensor_tree_hash(native.state_dict()) == tensor_tree_hash(split_state(model)[0]))
        results['train' if training else 'eval'] = rows
    if tensor_tree_hash(batch) != before or not all(
            row['prediction_close'] and row['rng_exact'] and row['native_state_exact']
            for phase in results.values() for row in phase.values()):
        raise ValueError('GW2 native recovery/RNG/input contract failed')
    return results


def backward(model, batch, data, modules, device):
    x, y, report, sensors = batch
    model.train(); model.zero_grad(set_to_none=True)
    input_hash = tensor_tree_hash(batch)
    cuda = torch.device(device).type == 'cuda'
    synchronize(device)
    baseline = torch.cuda.memory_allocated(device) if cuda else None
    if cuda:
        torch.cuda.reset_peak_memory_stats(device)
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record()
    start = perf_counter()
    prediction = data.scaler.inverse_transform(model(x, incident_data=report, sensor_data=sensors))
    target = data.scaler.inverse_transform(y)
    null = target.min() if target.min() < 1 else target.new_tensor(0.)
    loss = modules.metrics.masked_mae(prediction, target, null)
    loss.backward()
    if cuda:
        end.record()
    synchronize(device)
    seconds = perf_counter()-start
    groups = {}
    for name, parameter in model.named_parameters():
        group = 'native'
        if '.router.' in name:
            group = name.split('.router.', 1)[1].split('.')[0]
        row = groups.setdefault(group, dict(parameters=0, with_grad=0, squared_norm=0., finite=True))
        row['parameters'] += parameter.numel()
        if parameter.grad is not None:
            grad = parameter.grad.detach()
            row['with_grad'] += parameter.numel()
            row['finite'] &= bool(torch.isfinite(grad).all())
            row['squared_norm'] += float(grad.double().square().sum())
    if (not bool(torch.isfinite(loss)) or not all(row['finite'] for row in groups.values())
            or tensor_tree_hash(batch) != input_hash):
        raise ValueError('Nonfinite GW2 objective/gradients or mutated input')
    return dict(batch_size=len(x), nodes=x.shape[2], loss=float(loss.detach()),
        objective='native_inverse_transform_masked_mae_all_12_leads',
        forward_backward_seconds=seconds, cuda_milliseconds=begin.elapsed_time(end) if cuda else None,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if cuda else None,
        incremental_peak_allocated_bytes=torch.cuda.max_memory_allocated(device)-baseline if cuda else None,
        peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if cuda else None,
        groups=groups, optimizer_updates=0, labels_passed_to_model=False, inputs_preserved=True,
        timing_scope='train_forward_inverse_loss_backward_excludes_H2D_hash_gradient_summary',
        caveat='zero_heads_intentionally_give_zero_initial_encoder_gradient')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--check', action='store_true')
    mode.add_argument('--stress', action='store_true')
    p.add_argument('--batch-size', type=int, default=48)
    p.add_argument('--data-dir', type=Path, default=default_data_dir())
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--threads', type=int, default=3)
    p.add_argument('--arms', nargs='+', choices=ARMS, default=['POST_LOCAL', 'POST_MESSAGE'],
        help='Backward/stress only these arms; defaults to the two new models')
    args = p.parse_args(argv)
    if args.batch_size < 1 or args.batch_size > 3909:
        p.error('batch size must be in 1..3909')
    if len(set(args.arms)) != len(args.arms):
        p.error('arms must be unique')
    if args.output.exists():
        raise ValueError('Verification output already exists')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    evidence = dict(schema='original_igstgnn_graph_message_verification_v1',
        status='GW2_VERIFICATION_RUNNING', engineering_only=True, optimizer_updates=0,
        test_evaluated=False, mode='stress' if args.stress else 'check',
        requested_batch_size=args.batch_size if args.stress else 2, design=DESIGN, backward_arms=args.arms)
    try:
        evidence['runtime'] = configure_runtime(args.device, args.threads)
        modules = get_original_modules()
        config = dict(CONFIG, device=args.device, bs=args.batch_size if args.stress else 2)
        data = load_data(args.data_dir, config, modules, logging.getLogger('gw2-check'),
            check=not args.stress, training=True)
        evidence['data'] = data.identity
        batch = device_batch(next(data.loaders['train_loader'].get_iterator()), args.device)
        if len(batch[0]) != evidence['requested_batch_size']:
            raise ValueError('Actual batch differs; no smaller fallback permitted')
        native = build_model(data, modules, args.device)
        evidence['registered_native_parameters'] = sum(p.numel() for p in native.parameters())
        evidence['source_fingerprint'] = modules.source_fingerprint
        evidence['execution_source'] = {name: sha256(REPO/'experiments/chronological'/name) for name in (
            'original_graph_message.py', 'verify_original_graph_message.py', 'original_igstgnn_runtime.py')}
        if not args.stress:
            evidence['recovery'] = recovery(native, batch)
        # Store CPU native weights once; measure one GPU model at a time.
        initial = copy.deepcopy(native.cpu().state_dict())
        del native; gc.collect()
        if torch.device(args.device).type == 'cuda':
            torch.cuda.empty_cache()
        evidence['backward'] = {}
        for arm in args.arms:
            configure_runtime(args.device, args.threads)
            model = build_variant(build_model(data, modules, args.device), arm)
            if arm == 'ORIGINAL':
                model.load_state_dict(initial)
            # All arms freshly constructed with the same seed and native data.
            if tensor_tree_hash(split_state(model)[0]) != tensor_tree_hash(initial):
                raise ValueError('Verification initialization differs')
            torch.manual_seed(1440)
            result = backward(model, batch, data, modules, args.device)
            result['registered_parameters'] = sum(p.numel() for p in model.parameters())
            if arm != 'ORIGINAL':
                groups = result['groups']
                active = 'static_logits' if arm == 'STATIC' else 'score_head'
                if groups[active]['squared_norm'] <= 0:
                    raise ValueError('Native masked MAE did not reach an active router head')
            evidence['backward'][arm] = result
            print(json.dumps(dict(arm=arm, phase='backward_complete', **result)), flush=True)
            del model
            gc.collect()
            if torch.device(args.device).type == 'cuda':
                torch.cuda.empty_cache()
        evidence['status'] = 'GW2_VERIFICATION_PASS'
        write_json(args.output, evidence)
        return evidence
    except BaseException as error:
        evidence.update(status='GW2_VERIFICATION_FAILED', error_type=type(error).__name__, error=str(error),
                        smaller_batch_fallback=False)
        write_json(args.output, evidence)
        raise


if __name__ == '__main__':
    main()
