"""Original IGSTGNN validation-only module screening from one best checkpoint."""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import hashlib
import logging
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.chronological.original_igstgnn_source import get_original_modules, load_archive
from experiments.chronological.original_igstgnn_interventions import CASE_NAMES, CASE_DESCRIPTIONS, run_case_with_audit
from experiments.chronological.original_igstgnn_ablation_metrics import summarize_predictions, write_report
from experiments.chronological.original_igstgnn_module_timing import profile_full_modules, write_module_timing
from experiments.chronological.train_original_igstgnn_baseline import validate_saved, summary_for
from experiments.chronological.original_igstgnn_runtime import (
    build_model, configure_runtime, default_data_dir, device_batch, load_data,
    native_metrics, rng_fingerprint, run_lock, sha256, tensor_tree_hash, write_json)

SCHEMA = 'original_igstgnn_fixed_checkpoint_module_screen_v1'
DEFAULT_ARCHIVE = REPO.parent/'复现结果/Contra_Costa_igstgnn_s2025_20260908_193127_20260909090248.zip'


def source_from_run(path, check=False):
    path = Path(path).resolve()
    snapshot = {str(path/name): sha256(path/name) for name in ('identity.json', 'summary.json', 'last.pt')}
    identity = json.loads((path/'identity.json').read_text(encoding='utf-8'))
    summary = json.loads((path/'summary.json').read_text(encoding='utf-8'))
    if identity.get('schema') != 'original_igstgnn_baseline_training_v1' or summary.get('schema') != identity['schema']:
        raise ValueError('Expected an original baseline training run')
    unsigned_identity = {key: value for key, value in identity.items() if key != 'identity_sha256'}
    if hashlib.sha256(json.dumps(unsigned_identity, sort_keys=True).encode()).hexdigest() != identity['identity_sha256']:
        raise ValueError('Original training identity digest differs')
    if identity['engineering_only'] and not check:
        raise ValueError('Engineering baseline cannot become a formal screening source')
    name = summary['best_checkpoint_file']
    if Path(name).name != name:
        raise ValueError('Invalid best checkpoint relative path')
    checkpoint = path/name
    if sha256(checkpoint) != summary['best_checkpoint_sha256']:
        raise ValueError('Best checkpoint fingerprint differs')
    last = torch.load(path/'last.pt', map_location='cpu', weights_only=True)
    validate_saved(last, identity, path, len(identity['data']['splits']['train']['partition']))
    if summary != summary_for(last, identity):
        raise ValueError('Source summary differs from validated committed training state')
    if (last['identity_sha256'] != identity['identity_sha256'] or last['best_checkpoint_file'] != name
            or last['best_checkpoint_sha256'] != summary['best_checkpoint_sha256']
            or last['best_validation_mae'] != summary['best_validation_mae']
            or last['completed_epoch'] != summary['completed_epoch']
            or last['best_epoch'] != summary['best_epoch']
            or last['global_updates'] != summary['global_updates']
            or last['registered_parameters'] != summary['registered_parameters']
            or summary['training_config'] != identity['training_config']
            or summary['source_fingerprint'] != identity['source_fingerprint']):
        raise ValueError('Best selection summary disagrees with committed training state')
    history = last['history']
    best = min(history, key=lambda row: row['validation_mae'])
    if (len(history) != last['completed_epoch'] or best['epoch'] != last['best_epoch']
            or best['validation_mae'] != last['best_validation_mae']):
        raise ValueError('Best selection disagrees with saved validation history')
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    provenance = dict(kind='original_baseline_run', path=str(path), checkpoint_path=str(checkpoint),
        checkpoint_sha256=summary['best_checkpoint_sha256'], identity_sha256=identity['identity_sha256'],
        input_data_identity=identity['data'], source_fingerprint=identity['source_fingerprint'],
        completed_epoch=summary['completed_epoch'], best_epoch=summary['best_epoch'],
        training_complete=summary['status'] == 'ORIGINAL_IGSTGNN_BASELINE_COMPLETE',
        source_runtime=identity['runtime'], engineering_only=identity['engineering_only'],
        strict=True, weights_only=True, source_files=dict(snapshot, **{str(checkpoint): sha256(checkpoint)}))
    if any(sha256(file) != expected for file, expected in snapshot.items()):
        raise ValueError('Source run changed while selecting the best checkpoint')
    return state, identity['training_config'], summary['best_validation_mae'], provenance


def verify_replay(actual, expected, *, engineering_only=False, atol=1e-4, rtol=1e-6):
    result = dict(expected_validation_mae=expected, actual=actual, metric_atol=atol, metric_rtol=rtol,
                  original_metric_float32=True, independent_historical_execution_signature=False)
    if engineering_only:
        result.update(status='ENGINEERING_REPLAY_ONLY', full_metric_gate_applied=False)
    else:
        if not np.isfinite(actual['mae']) or not np.isclose(actual['mae'], expected, atol=atol, rtol=rtol):
            raise ValueError(f'Original FULL replay MAE mismatch: actual={actual["mae"]}, expected={expected}; interventions blocked')
        result.update(status='ORIGINAL_FULL_VALIDATION_REPLAY_PASS', full_metric_gate_applied=True)
    return result


def timed_case(model, batch, case, device):
    x, _, incident, sensors = batch
    is_cuda = torch.device(device).type == 'cuda'
    if is_cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        baseline = torch.cuda.memory_allocated(device)
        start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record(torch.cuda.current_stream(device))
    started = time.perf_counter()
    output, audit = run_case_with_audit(model, x, incident, sensors, case)
    if is_cuda:
        stop.record(torch.cuda.current_stream(device))
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter()-started
    timing = dict(wall_seconds=elapsed, batch_size=len(x), warmup=False,
        cuda_seconds=start.elapsed_time(stop)/1000 if is_cuda else None,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if is_cuda else None,
        incremental_peak_bytes=torch.cuda.max_memory_allocated(device)-baseline if is_cuda else None)
    return output, audit, timing


def screen(model, loader, scaler, modules, indices, device, output, *, expected_mae,
           engineering_only=False, passes=2, warmup=1, calls=None, profile_repetitions=0):
    """Complete FULL replay gates all subsequent OFF interventions."""
    output = Path(output)
    calls = {case: 0 for case in CASE_NAMES} if calls is None else calls
    before = tensor_tree_hash(model.state_dict())
    rng_before = rng_fingerprint()
    targets, full = [], []
    partition, first_batch = [], None
    for number, raw in enumerate(loader.get_iterator(), 1):
        batch = device_batch(raw, device)
        if first_batch is None:
            first_batch = batch
        x, y, incident, sensors = batch
        input_before = tensor_tree_hash(batch)
        calls['FULL'] += 1
        prediction, audit = run_case_with_audit(model, x, incident, sensors, 'FULL')
        if prediction.shape != y.shape or not bool(torch.isfinite(prediction).all()):
            raise ValueError('Original FULL output axes/values invalid')
        full.append(scaler.inverse_transform(prediction).squeeze(-1).cpu().numpy().copy())
        targets.append(scaler.inverse_transform(y).squeeze(-1).cpu().numpy().copy())
        partition.append(len(x))
        if tensor_tree_hash(batch) != input_before:
            raise ValueError('FULL replay changed inputs')
        print(json.dumps(dict(phase='full_replay', completed_batches=number, total_batches=loader.num_batch)), flush=True)
    target, baseline = np.concatenate(targets), np.concatenate(full)
    if len(target) != len(indices) or tensor_tree_hash(model.state_dict()) != before or rng_fingerprint() != rng_before:
        raise ValueError('FULL replay changed model or sample axis')
    replay = verify_replay(native_metrics(baseline, target, modules), expected_mae, engineering_only=engineering_only)
    replay.update(batch_partition=partition, model_state_sha256=before, checkpoint_strict_load=True)
    write_json(output/'replay.json', replay)
    print(json.dumps(dict(phase='replay_passed', **replay)), flush=True)
    for repetition in range(warmup):
        for case in CASE_NAMES:
            calls[case] += 1
            _, _ = run_case_with_audit(model, first_batch[0], first_batch[2], first_batch[3], case)
        print(json.dumps(dict(phase='warmup_completed', repetition=repetition+1)), flush=True)
    if profile_repetitions:
        profile_input = tensor_tree_hash(first_batch)
        calls['FULL'] += profile_repetitions
        profiled, profile = profile_full_modules(model, first_batch[0], first_batch[2], first_batch[3],
                                                device, repetitions=profile_repetitions)
        profiled = scaler.inverse_transform(profiled).squeeze(-1).cpu().numpy()
        if (not np.array_equal(profiled, baseline[:partition[0]])
                or tensor_tree_hash(first_batch) != profile_input
                or tensor_tree_hash(model.state_dict()) != before or rng_fingerprint() != rng_before):
            raise ValueError('Passive module timing changed native FULL or its state')
        profile.update(batch_size=partition[0], sample_indices=list(map(int, indices[:partition[0]])),
                       native_prediction_exact=True, model_inputs_rng_preserved=True)
        write_json(output/'module_profile_raw.json', profile)
        del profiled
        print(json.dumps(dict(phase='module_profile_completed', repetitions=profile_repetitions)), flush=True)
    del first_batch, prediction, full, targets, raw, batch
    predictions = {case: np.empty_like(baseline) for case in CASE_NAMES}
    timings = {case: [] for case in CASE_NAMES}
    certificates = {case: [] for case in CASE_NAMES}
    for pass_index in range(passes):
        position = 0
        for batch_index, raw in enumerate(loader.get_iterator()):
            batch = device_batch(raw, device)
            x, y, incident, sensors = batch
            count = len(x)
            input_before = tensor_tree_hash(batch)
            # Rotate deterministic case order to reduce a persistent first-case
            # initialization/thermal ordering effect. No input/weight RNG draws.
            shift = (batch_index+pass_index) % len(CASE_NAMES)
            order = CASE_NAMES[shift:]+CASE_NAMES[:shift]
            for order_index, case in enumerate(order):
                calls[case] += 1
                prediction, audit, timing = timed_case(model, batch, case, device)
                value = scaler.inverse_transform(prediction).squeeze(-1).cpu().numpy().copy()
                if value.shape != target[position:position+count].shape or not np.isfinite(value).all():
                    raise ValueError('Intervention output axes/values invalid: '+case)
                if pass_index == 0:
                    predictions[case][position:position+count] = value
                elif not np.array_equal(predictions[case][position:position+count], value):
                    raise ValueError('Repeated intervention prediction changed: '+case)
                if case == 'FULL' and not np.array_equal(value, baseline[position:position+count]):
                    raise ValueError('Paired FULL changed after successful replay')
                timing.update(pass_index=pass_index, batch_index=batch_index, case_order_index=order_index)
                timings[case].append(timing)
                if pass_index == 0:
                    certificates[case].append(audit)
                print(json.dumps(dict(phase='screening', pass_index=pass_index+1, batch=batch_index+1,
                    total_batches=loader.num_batch, case=case, seconds=timing['wall_seconds'])), flush=True)
                del prediction, value
            calls['FULL'] += 1
            restored, _ = run_case_with_audit(model, x, incident, sensors, 'FULL')
            restored = scaler.inverse_transform(restored).squeeze(-1).cpu().numpy()
            if not np.array_equal(restored, baseline[position:position+count]):
                raise ValueError('Final FULL restoration differs')
            if (tensor_tree_hash(batch) != input_before or tensor_tree_hash(model.state_dict()) != before
                    or rng_fingerprint() != rng_before):
                raise ValueError('Screening changed input or model state')
            if any(p.grad is not None for p in model.parameters()):
                raise ValueError('Screening created parameter gradients')
            position += count
            del raw, batch, x, y, incident, sensors, restored
        if position != len(target):
            raise ValueError('Incomplete validation screening')
    valid = target != replay['actual']['mask_value']
    np.savez_compressed(output/'paired_predictions.npz', **predictions, target=target, valid=valid,
        sample_indices=np.asarray(indices, dtype=np.int64))
    write_json(output/'interventions.json', dict(descriptions=CASE_DESCRIPTIONS, calls=certificates,
        model_and_inputs_preserved=True, rng_preserved=True, original_on_exactly_restored=True))
    return predictions, target, valid, replay, timings


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--archive', type=Path)
    group.add_argument('--checkpoint', type=Path)
    group.add_argument('--source-run', type=Path)
    parser.add_argument('--training-log', type=Path)
    parser.add_argument('--data-dir', type=Path, default=default_data_dir())
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--threads', type=int, default=3)
    parser.add_argument('--passes', type=int, default=2, help='Repeated full validation timing passes; same predictions required')
    parser.add_argument('--warmup', type=int, default=1, help='Untimed first-batch calls of every state')
    parser.add_argument('--profile-repetitions', type=int, default=2,
                        help='Passive FULL module timings on the first original batch; zero disables')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args(argv)
    if args.passes < 1 or args.warmup < 0 or args.profile_repetitions < 0:
        parser.error('passes must be positive; warmup and profile-repetitions nonnegative')
    if bool(args.checkpoint) != bool(args.training_log):
        parser.error('--checkpoint requires --training-log, and vice versa')
    if not any((args.archive, args.checkpoint, args.source_run)):
        if DEFAULT_ARCHIVE.is_file():
            args.archive = DEFAULT_ARCHIVE
        else:
            parser.error('Provide --source-run, --archive, or --checkpoint with --training-log')
    if args.output_dir is None:
        root = args.source_run or REPO/'experiments/chronological_runs'
        args.output_dir = root/('original_module_screen_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f'))
    args.output_dir = args.output_dir.resolve()
    return args


def main(argv=None):
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    calls = {case: 0 for case in CASE_NAMES}
    locks = ExitStack()
    try:
        modules = get_original_modules()
        if args.source_run:
            if not (args.source_run/'active.lock').is_file():
                raise ValueError('Original source run is missing its training lock')
            locks.enter_context(run_lock(args.source_run))
            state, config, expected, source = source_from_run(args.source_run, check=args.check)
        else:
            archive = load_archive(args.archive, checkpoint=args.checkpoint, train_log=args.training_log)
            state, config, expected = archive.state_dict, archive.config, archive.best_validation_mae
            source = dict(archive.provenance, training_metadata=archive.metadata)
            source['source_files'] = {source['path']: sha256(source['path'])}
            if source['kind'] == 'checkpoint_and_log':
                source['source_files'][source['log_path']] = sha256(source['log_path'])
        runtime = configure_runtime(args.device, args.threads, config['seed'])
        execution = {str(path.relative_to(REPO)): sha256(path) for path in (
            Path(__file__), Path(__file__).with_name('train_original_igstgnn_baseline.py'),
            *Path(__file__).parent.glob('original_igstgnn_*.py'))}
        logger = logging.getLogger('original-igstgnn-screen')
        logger.setLevel(logging.INFO)
        logging.basicConfig(level=logging.INFO)
        data = load_data(args.data_dir, config, modules, logger, check=args.check)
        if args.source_run:
            if source['source_fingerprint'] != modules.source_fingerprint:
                raise ValueError('Training and screening original source differ')
            old_data = source['input_data_identity']
            if (old_data['hashes'] != data.identity['hashes']
                    or old_data['normalization'] != data.identity['normalization']):
                raise ValueError('Training and screening data fingerprints differ')
            if not args.check and old_data['splits']['val'] != data.identity['splits']['val']:
                raise ValueError('Training and screening validation partition differ')
        model = build_model(data, modules, args.device)
        model.load_state_dict(state, strict=True)
        model.eval()
        del state
        parameters = sum(p.numel() for p in model.parameters())
        if parameters != 455220:
            raise ValueError('Original full model registered parameter count differs')
        predictions, target, valid, replay, timings = screen(model, data.loaders['val_loader'], data.scaler,
            modules, data.identity['splits']['val']['selected_indices'], args.device, args.output_dir,
            expected_mae=expected, engineering_only=args.check, passes=args.passes, warmup=args.warmup, calls=calls,
            profile_repetitions=args.profile_repetitions)
        # Include both axes in the paired numerical archive.
        with np.load(args.output_dir/'paired_predictions.npz', allow_pickle=False) as saved:
            paired = {name: saved[name] for name in saved.files}
        road_axis = ['']*len(data.station_ids)
        for road, nodes in data.roads.items():
            for node in nodes:
                road_axis[node] = road
        np.savez_compressed(args.output_dir/'paired_predictions.npz', **paired, station_ids=data.station_ids,
                            road_names=np.asarray(road_axis, dtype=str))
        # Recheck immutable source inputs before publishing a successful report.
        for name, expected_hash in data.identity['hashes'].items():
            if sha256(data.directory/name) != expected_hash:
                raise ValueError('Source data changed during screening: '+name)
        get_original_modules()
        if any(sha256(REPO/path) != expected_hash for path, expected_hash in execution.items()):
            raise ValueError('Screening execution source changed while running')
        for path, expected_hash in source['source_files'].items():
            if sha256(path) != expected_hash:
                raise ValueError('Source training artifact changed during screening: '+path)
        metadata = dict(schema=SCHEMA, primary_baseline='original_IGSTGNN_558ab01_reference',
            source=source, source_fingerprint=modules.source_fingerprint, execution_source=execution,
            data=data.identity, runtime=runtime,
            engineering_only=args.check, replay=replay, registered_parameters=parameters,
            inactive_local_bn_parameters=320, new_inference_calls=calls, optimizer_updates=0, test_evaluated=False,
            timing_scope='synchronized_complete_model_forward_with_screening_hooks_excludes_data_H2D_inverse_transform_statistics_hashing',
            timing_passes=args.passes, warmup_calls_per_case=args.warmup,
            inference_only_no_training_speedup_claim=True, retrained_ablation=False)
        analysis = summarize_predictions(predictions, target, valid,
            np.asarray(data.identity['splits']['val']['selected_indices']), data.station_ids, data.roads,
            case_order=CASE_NAMES)
        write_report(args.output_dir, analysis, metadata, timings)
        if args.profile_repetitions:
            section = write_module_timing(args.output_dir,
                json.loads((args.output_dir/'module_profile_raw.json').read_text(encoding='utf-8')))
            with (args.output_dir/'report.md').open('a', encoding='utf-8') as stream:
                stream.write('\n'+section)
        summary = dict(schema=SCHEMA, status='ORIGINAL_IGSTGNN_MODULE_SCREEN_PASS',
            engineering_only=args.check, validation_rows=len(target), expected_validation_mae=expected,
            actual_full_native_metrics=replay['actual'], new_inference_calls=calls, optimizer_updates=0,
            test_evaluated=False, source_artifacts_preserved=True, output_dir=str(args.output_dir))
        write_json(args.output_dir/'summary.json', summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return summary
    except BaseException as error:
        write_json(args.output_dir/'failure.json', dict(schema=SCHEMA, status='ORIGINAL_IGSTGNN_MODULE_SCREEN_FAILED',
            error_type=type(error).__name__, error=str(error), new_inference_calls=calls,
            optimizer_updates=0, test_evaluated=False))
        raise
    finally:
        locks.close()


if __name__ == '__main__':
    main()
