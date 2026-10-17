"""Server-only GW3 recovery and native-MAE backward; zero optimizer updates."""
import argparse
import copy
import gc
import json
import logging
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.chronological import train_original_receiver_match as train
from experiments.chronological import verify_original_graph_message as timing
from experiments.chronological.original_receiver_match import ARMS, DESIGN, build_variant, split_state
import torch


def recovery(original, batch):
    x, _, incident, sensors = batch
    result = {}
    before = train.base.tensor_tree_hash(batch)
    for training in (False, True):
        native = copy.deepcopy(original).train(training)
        torch.manual_seed(440+int(training))
        with torch.no_grad():
            expected = native(x, incident_data=incident, sensor_data=sensors)
        rng = timing.rng_fingerprint()
        rows = {}
        for arm in ARMS:
            candidate = build_variant(copy.deepcopy(original), arm).train(training)
            torch.manual_seed(440+int(training))
            with torch.no_grad():
                actual = candidate(x, incident_data=incident, sensor_data=sensors)
            rows[arm] = dict(prediction_close=bool(torch.allclose(actual, expected, atol=2e-5, rtol=2e-5)),
                abs_max=float((actual-expected).abs().max()), atol=2e-5, rtol=2e-5,
                rng_exact=rng == timing.rng_fingerprint(),
                native_state_exact=train.base.tensor_tree_hash(native.state_dict()) == train.base.tensor_tree_hash(split_state(candidate)[0]))
            del candidate
        result['train' if training else 'eval'] = rows
        del native
    timing_result = all(row['prediction_close'] and row['rng_exact'] and row['native_state_exact']
        for phase in result.values() for row in phase.values())
    if not timing_result or before != train.base.tensor_tree_hash(batch):
        raise ValueError('GW3 initial recovery/RNG/native state/input contract failed')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--check', action='store_true')
    mode.add_argument('--stress', action='store_true')
    parser.add_argument('--batch-size', type=int, default=48)
    parser.add_argument('--data-dir', type=Path, default=train.base.default_data_dir())
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--threads', type=int, default=3)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if not 1 <= args.batch_size <= 3909:
        parser.error('batch-size must be in 1..3909')
    if args.output.exists():
        raise ValueError('Verification output exists; use a timestamped fresh filename')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    evidence = dict(schema='gw3_receiver_match_verification_v1', status='GW3_VERIFICATION_RUNNING',
        engineering_only=True, design=DESIGN, optimizer_updates=0, test_evaluated=False,
        requested_batch_size=args.batch_size if args.stress else 2)
    try:
        evidence['runtime'] = train.base.configure_runtime(args.device, args.threads)
        modules = train.base.get_original_modules()
        config = dict(train.base.CONFIG, device=args.device, bs=args.batch_size if args.stress else 2)
        data = train.base.load_data(args.data_dir, config, modules, logging.getLogger('gw3-verify'),
            check=not args.stress, training=True)
        batch = timing.device_batch(next(data.loaders['train_loader'].get_iterator()), args.device)
        if len(batch[0]) != evidence['requested_batch_size']:
            raise ValueError('Requested batch not supplied; no smaller fallback')
        evidence['data'] = data.identity
        evidence['source_fingerprint'] = modules.source_fingerprint
        evidence['execution_source'] = {name: train.base.sha256(REPO/name) for name in train.EXECUTION_FILES}
        native = train.base.build_model(data, modules, args.device)
        if not args.stress:
            evidence['recovery'] = recovery(native, batch)
        initial = copy.deepcopy(native.cpu().state_dict())
        del native
        gc.collect()
        if torch.device(args.device).type == 'cuda':
            torch.cuda.empty_cache()
        evidence['backward'] = {}
        for arm in ARMS:
            train.base.configure_runtime(args.device, args.threads)
            model = train.model_for_arm(data, modules, args.device, arm)
            if train.base.tensor_tree_hash(split_state(model)[0]) != train.base.tensor_tree_hash(initial):
                raise ValueError('GW3 verification native initialization differs')
            result = timing.backward(model, batch, data, modules, args.device)
            result['registered_parameters'] = sum(p.numel() for p in model.parameters())
            result['trainable_parameters'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
            result['caveat'] = 'zero_key_gives_zero_initial_query_gradient; first_key_gradient_must_be_nonzero'
            if arm == 'FIXED_UNIFORM':
                if any(v['with_grad'] for k, v in result['groups'].items() if k != 'native'):
                    raise ValueError('Fixed uniform control unexpectedly trained router parameters')
            elif result['groups'].get('key', {}).get('squared_norm', 0.) <= 0:
                raise ValueError('Native MAE did not reach the GW3 key projection')
            evidence['backward'][arm] = result
            print(json.dumps(dict(arm=arm, phase='backward_complete', **result)), flush=True)
            del model
            gc.collect()
            if torch.device(args.device).type == 'cuda':
                torch.cuda.empty_cache()
        evidence['status'] = 'GW3_VERIFICATION_PASS'
        train.base.write_json(args.output, evidence)
        return evidence
    except BaseException as error:
        evidence.update(status='GW3_VERIFICATION_FAILED', error_type=type(error).__name__,
            error=str(error), smaller_batch_fallback=False)
        train.base.write_json(args.output, evidence)
        raise


if __name__ == '__main__':
    main()
