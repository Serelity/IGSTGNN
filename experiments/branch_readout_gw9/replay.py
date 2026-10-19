"""Read-only diagnostics for a GW8 cache / frozen GW9 forward disagreement.

This tool never trains, changes a tolerance, writes a historical run, or grants
permission to proceed to joint training. A partial replay is labelled partial;
even a complete exact replay is engineering evidence, not an accuracy screen.
"""
import argparse
from datetime import datetime, timezone
import gc
import itertools
import json
import logging
import os
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch

from experiments.branch_readout_gw9 import train
from experiments.branch_readout_gw9.reference import certify, protected_paths, recertify, require
from experiments.chronological import evaluate_original_graph_message as metrics
from experiments.chronological.graph_message_reuse import RUNTIME_KEYS
from experiments.chronological.original_igstgnn_runtime import device_batch

SCHEMA = 'gw9_frozen_forward_replay_diagnostic_v1'
AXIS_KEYS = ('y', 'valid', 'sample_ids', 'node_ids', 'road_labels', 'direct_support')
CASES = ('GW8_SOURCE_DIAGNOSTICS_ON', 'GW8_SOURCE_DIAGNOSTICS_OFF',
    'GW8_REQUIRES_GRAD_FALSE', 'GW9_PROBE_CAPTURE', 'GW9_PROBE_CONTROL')


def _finite_scalar(value):
    return float(value) if np.isfinite(value) else None


def compare_arrays(actual, expected):
    """Exact comparison plus finite float64 differences; never applies allclose."""
    actual, expected = np.asarray(actual), np.asarray(expected)
    result = dict(actual_shape=list(actual.shape), expected_shape=list(expected.shape),
        actual_dtype=str(actual.dtype), expected_dtype=str(expected.dtype),
        nonfinite_counts=dict(actual=int(np.count_nonzero(~np.isfinite(actual))),
            expected=int(np.count_nonzero(~np.isfinite(expected)))),
        exact_equal=False, mismatch_count=None, max_abs=None, mean_abs=None,
        rms=None, first_mismatch=None)
    if actual.shape != expected.shape:
        return result
    result['exact_equal'] = bool(actual.dtype == expected.dtype
        and result['nonfinite_counts'] == dict(actual=0, expected=0)
        and np.array_equal(actual, expected))
    mismatch = actual != expected
    result['mismatch_count'] = int(np.count_nonzero(mismatch))
    finite = np.isfinite(actual) & np.isfinite(expected)
    if np.any(finite):
        delta = actual[finite].astype(np.float64)-expected[finite].astype(np.float64)
        result.update(max_abs=float(np.max(np.abs(delta))),
            mean_abs=float(np.mean(np.abs(delta))), rms=float(np.sqrt(np.mean(delta*delta))))
    if np.any(mismatch):
        index = tuple(int(i) for i in np.unravel_index(int(np.argmax(mismatch)), actual.shape))
        result['first_mismatch'] = dict(index=list(index), actual=_finite_scalar(actual[index]),
            expected=_finite_scalar(expected[index]))
    return result


def diagnostic_scope(selected_batches, total_batches):
    require(type(selected_batches) is int and type(total_batches) is int
        and 1 <= selected_batches <= total_batches, 'Invalid diagnostic replay batch scope')
    return dict(selected_validation_batches=selected_batches, total_validation_batches=total_batches,
        full_validation_replayed=selected_batches == total_batches,
        partial_validation_replay=selected_batches != total_batches,
        engineering_only=True, predictive_accuracy_evidence=False,
        screening_performed=False, candidate_for_confirmation=False,
        joint_eligibility_changed=False, optimizer_updates=0, new_training_updates=0,
        source_checkpoint_writes=0, test_evaluated=False)


def _cache(reference):
    saved = reference['archives']['best']; directory = Path(saved['directory'])
    require(train.base.sha256(directory/'metadata.json') == saved['metadata_sha256']
        and train.base.sha256(directory/'paired_predictions.npz') == saved['archive_sha256'],
        'Certified GW8 BEST archive changed before diagnostic replay')
    with np.load(directory/'paired_predictions.npz', allow_pickle=False) as archive:
        baseline = {name: archive[name].copy() for name in AXIS_KEYS}
        baseline['pred__R_CONTINUE'] = archive['pred__R_CONTINUE'].copy()
    metadata = json.loads((directory/'metadata.json').read_text(encoding='utf-8'))
    require(train.base.sha256(directory/'metadata.json') == saved['metadata_sha256']
        and train.base.sha256(directory/'paired_predictions.npz') == saved['archive_sha256'],
        'Certified GW8 BEST archive changed while reading')
    return baseline, metadata


def _runtime_details(runtime, source_runtime, cache_runtime, device):
    value = dict(configured=runtime, source_training_runtime=source_runtime,
        source_cache_runtime=cache_runtime, cudnn_version=torch.backends.cudnn.version(),
        torch_cuda_version=torch.version.cuda,
        current_device_name=runtime.get('device_name'),
        device_name_matches_source_training=runtime.get('device_name') == source_runtime.get('device_name'),
        device_name_matches_source_cache=runtime.get('device_name') == cache_runtime.get('device_name'),
        note='Runtime identity keys do not certify identical GPU hardware or driver kernels.')
    if torch.device(device).type == 'cuda':
        properties = torch.cuda.get_device_properties(device)
        value['cuda_properties'] = dict(name=properties.name, major=properties.major,
            minor=properties.minor, total_memory=int(properties.total_memory),
            multi_processor_count=properties.multi_processor_count)
    return value


def _collect(model, data, modules, device, count, *, capture=False, diagnostics=False):
    """One model, one pass over a validation prefix; no optimizer or backward."""
    model.eval()
    expected_hash = train.base.tensor_tree_hash(model.state_dict())
    flags = tuple(p.requires_grad for p in model.parameters())
    modes = tuple(module.training for module in model.modules())
    predictions, truths, supports, input_hashes = [], [], [], []
    if diagnostics:
        model.start_diagnostics()
    with torch.no_grad():
        for batch_index, raw in enumerate(itertools.islice(
                data.loaders['val_loader'].get_iterator(), count), 1):
            batch = device_batch(raw, device); x, y, incident, sensors = batch
            before = train.base.tensor_tree_hash(batch)
            if capture:
                prediction = model.capture_features(x, incident_data=incident, sensor_data=sensors)['base']
            else:
                prediction = model(x, incident_data=incident, sensor_data=sensors)
            require(train.base.tensor_tree_hash(batch) == before, 'Diagnostic forward mutated inputs')
            input_hashes.append(before)
            predictions.append(data.scaler.inverse_transform(prediction).squeeze(-1).cpu().numpy())
            truths.append(data.scaler.inverse_transform(y).squeeze(-1).cpu().numpy())
            supports.append(incident['distances'].abs().sum(-1).cpu().numpy() > 0)
            model.assert_cache_clear()
            print(json.dumps(dict(phase='GW9_REPLAY_DIAGNOSTIC', batch=batch_index,
                selected_batches=count, capture=capture, diagnostics=diagnostics)), flush=True)
    observed = model.diagnostics() if diagnostics else None
    require(len(predictions) == count, 'Diagnostic validation prefix is incomplete')
    require(train.base.tensor_tree_hash(model.state_dict()) == expected_hash,
        'Diagnostic forward changed model parameters or buffers')
    require(tuple(p.requires_grad for p in model.parameters()) == flags
        and tuple(module.training for module in model.modules()) == modes,
        'Diagnostic forward did not restore parameter flags or module modes')
    return dict(prediction=np.concatenate(predictions), truth=np.concatenate(truths),
        support=np.concatenate(supports), input_hashes=input_hashes,
        diagnostics=observed, model_tensor_sha256=expected_hash,
        state_unchanged=True, parameter_flags_restored=True, module_modes_restored=True)


def _native_delta(actual, expected):
    return {name: float(actual[name]-expected[name])
        for name in ('mae', 'rmse', 'mape_fraction')}


def _annotate_first(comparison, sample_ids, node_ids):
    first = comparison['first_mismatch']
    if first is not None and len(first['index']) == 3:
        sample, horizon, node = first['index']
        first.update(sample_id=int(sample_ids[sample]), horizon=horizon+1, node_id=int(node_ids[node]))
    return comparison


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gw8-run', type=Path, required=True)
    parser.add_argument('--gw8-best-eval', type=Path)
    parser.add_argument('--gw8-last-eval', type=Path)
    parser.add_argument('--data-dir', type=Path, default=train.base.default_data_dir())
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--threads', type=int, default=3)
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--hidden', type=int, choices=(16,), default=16)
    parser.add_argument('--batches', type=int, default=1,
        help='Number of leading validation batches; 0 replays the entire validation set.')
    args = parser.parse_args(argv)
    if args.batches < 0: parser.error('batches must be nonnegative')
    if args.threads < 1: parser.error('threads must be positive')
    for name in ('gw8_run', 'data_dir', 'output_dir'):
        setattr(args, name, getattr(args, name).resolve())
    return args


def main(argv=None):
    args = parse_args(argv)
    require(not args.output_dir.exists(), 'Use a fresh independent diagnostic output directory')
    reference, old_identity, gw4_weights, recurrent_weights = certify(args.gw8_run,
        {role: getattr(args, 'gw8_'+role+'_eval') for role in ('best', 'last')})
    for protected in protected_paths(reference):
        require(args.output_dir != protected and protected not in args.output_dir.parents,
            'Diagnostic output must be outside immutable historical runs')
    args.output_dir.mkdir(parents=True)
    files = tuple(dict.fromkeys((*train.EXECUTION_FILES,
        'experiments/branch_readout_gw9/replay.py')))
    execution_source = {name: train.base.sha256(train.REPO/name) for name in files}
    evidence = dict(schema=SCHEMA, status='GW9_REPLAY_DIAGNOSTIC_RUNNING',
        created_utc=datetime.now(timezone.utc).isoformat(), source_reference=reference,
        execution_source=execution_source, engineering_only=True, optimizer_updates=0,
        new_training_updates=0, screening_performed=False, joint_eligibility_changed=False,
        source_checkpoint_writes=0, test_evaluated=False)
    try:
        runtime = train.base.configure_runtime(args.device, args.threads, args.seed)
        for name in RUNTIME_KEYS:
            require(runtime.get(name) == old_identity['runtime'].get(name),
                'Diagnostic/source runtime differs: '+name)
        modules = train.base.get_original_modules()
        require(modules.source_fingerprint == old_identity['source_fingerprint'], 'Native source differs')
        data = train.base.load_data(args.data_dir, old_identity['training_config'], modules,
            logging.getLogger('gw9-replay'), check=False, training=True)
        require(data.identity == old_identity['data'], 'Diagnostic/source data identity differs')
        baseline, cached_metadata = _cache(reference)
        require(np.array_equal(baseline['sample_ids'], data.identity['splits']['val']['selected_indices'])
            and np.array_equal(baseline['node_ids'], data.station_ids), 'Diagnostic sample/node axes differ')
        roads = np.full(len(data.station_ids), 'unknown', dtype='<U40')
        for road, indices in data.roads.items(): roads[indices] = road
        require(np.array_equal(baseline['road_labels'], roads), 'Diagnostic road axes differ')
        total = data.loaders['val_loader'].num_batch
        count = total if args.batches == 0 else args.batches
        evidence.update(diagnostic_scope(count, total))
        evidence['runtime'] = _runtime_details(runtime, old_identity['runtime'],
            cached_metadata.get('runtime', {}), args.device)
        size = sum(data.identity['splits']['val']['partition'][:count])
        cached_prediction = baseline['pred__R_CONTINUE'][:size]
        truth = baseline['y'][:size]
        evidence['validation_rows'] = size
        evidence['source_cache_native_metrics_for_selected_rows'] = metrics.native_metrics(
            cached_prediction, truth, modules)
        # One ordinary GW8 wrapper is used for the source and flag-isolation cases.
        raw = train.model_for_arm(data, modules, args.device, 'R_CONTINUE', gw4_weights,
            recurrent_weights, hidden=args.hidden, stage='joint').anchor
        raw.eval()
        require(train.base.tensor_tree_hash(raw.state_dict()) ==
            reference['source_initialization']['model_tensor_sha256'], 'Raw GW8 source tensors differ')
        common_rng = train.base.rng_state(); results = {}; first_inputs = None
        raw_source = None; raw_source_native = None
        for name in CASES:
            if name.startswith('GW8_'):
                model = raw
                model.requires_grad_(name != 'GW8_REQUIRES_GRAD_FALSE')
                capture, diagnostics = False, name == 'GW8_SOURCE_DIAGNOSTICS_ON'
            else:
                # A fresh frozen wrapper prevents state leakage between diagnostic cases.
                model = train.model_for_arm(data, modules, args.device, 'R_CONTINUE', gw4_weights,
                    recurrent_weights, hidden=args.hidden, stage='probe')
                require(train.base.tensor_tree_hash(model.anchor.state_dict()) ==
                    reference['source_initialization']['model_tensor_sha256'], 'GW9 source tensors differ')
                capture, diagnostics = name == 'GW9_PROBE_CAPTURE', False
            train.epoch_tools.restore_rng(common_rng)
            initial_rng = train.base.tensor_tree_hash(train.base.rng_state())
            collected = _collect(model, data, modules, args.device, count,
                capture=capture, diagnostics=diagnostics)
            prediction = collected.pop('prediction')
            actual_truth, support = collected.pop('truth'), collected.pop('support')
            require(np.array_equal(actual_truth, truth)
                and actual_truth.dtype == truth.dtype
                and np.array_equal(support, baseline['direct_support'][:size]),
                'Diagnostic validation truth/support differs from certified cache')
            if first_inputs is None: first_inputs = collected['input_hashes']
            require(collected['input_hashes'] == first_inputs, 'Diagnostic cases used different input batches')
            require(bool(np.isfinite(prediction).all()), 'Diagnostic prediction contains nonfinite values')
            native = metrics.native_metrics(prediction, truth, modules)
            if raw_source is None:
                raw_source, raw_source_native = prediction.copy(), dict(native)
            collected.update(native_metrics=native,
                versus_cache=_annotate_first(compare_arrays(prediction, cached_prediction),
                    baseline['sample_ids'][:size], baseline['node_ids']),
                versus_raw_source=_annotate_first(compare_arrays(prediction, raw_source),
                    baseline['sample_ids'][:size], baseline['node_ids']),
                native_delta_versus_cache=_native_delta(native,
                    evidence['source_cache_native_metrics_for_selected_rows']),
                native_delta_versus_raw_source=_native_delta(native, raw_source_native),
                rng_unchanged=initial_rng == train.base.tensor_tree_hash(train.base.rng_state()),
                frozen_parameters_after_forward=all(not p.requires_grad for p in model.parameters()),
                capture_features=capture, source_style_diagnostics=diagnostics)
            require(collected['rng_unchanged'], 'Evaluation-mode diagnostic consumed RNG state')
            results[name] = collected
            evidence['cases'] = results
            train.base.write_json(args.output_dir/'diagnostics.json', evidence)
            if name.startswith('GW9_'):
                del model; gc.collect()
                if torch.device(args.device).type == 'cuda': torch.cuda.empty_cache()
        del raw; gc.collect()
        recertify(reference)
        require(all(train.base.sha256(train.REPO/name) == digest for name, digest in execution_source.items()),
            'Diagnostic execution source changed during replay')
        evidence.update(status='GW9_REPLAY_DIAGNOSTIC_COMPLETE',
            all_cases_exact_with_cache=all(row['versus_cache']['exact_equal'] for row in results.values()),
            all_cases_exact_with_raw_source=all(row['versus_raw_source']['exact_equal'] for row in results.values()),
            diagnosis_requires_review=True,
            tolerance_changed=False, historical_cache_replaced=False)
        lines = ['# GW9 frozen-forward replay diagnostics', '',
            'Engineering diagnostics only. No optimizer steps, test evaluation, screening or eligibility changes.',
            'Validation replay: {} batches / {}, {} rows ({}).'.format(count, total, size,
                'full' if count == total else 'partial'), '',
            '| Case | Exact cache | Exact raw source | Cache max abs | Cache mean abs | MAE |',
            '|---|---|---|---:|---:|---:|']
        for name, row in results.items():
            difference = row['versus_cache']
            lines.append('| {} | {} | {} | {} | {} | {:.9f} |'.format(name,
                difference['exact_equal'], row['versus_raw_source']['exact_equal'],
                difference['max_abs'], difference['mean_abs'], row['native_metrics']['mae']))
        lines.extend(['', 'No tolerance is relaxed and no historical prediction cache is rewritten.',
            'Partial replay cannot certify all validation batches. Full replay is also not an accuracy screen.',
            'See diagnostics.json for first mismatch sample/horizon/node, metrics and hardware identity.', ''])
        (args.output_dir/'report.md').write_text('\n'.join(lines), encoding='utf-8')
        train.base.write_json(args.output_dir/'diagnostics.json', evidence)
        print(json.dumps(dict(status=evidence['status'], output_dir=str(args.output_dir),
            full_validation_replayed=evidence['full_validation_replayed'],
            all_cases_exact_with_cache=evidence['all_cases_exact_with_cache'], optimizer_updates=0)), flush=True)
        return evidence
    except BaseException as error:
        evidence.update(status='GW9_REPLAY_DIAGNOSTIC_FAILED', error_type=type(error).__name__, error=str(error))
        train.base.write_json(args.output_dir/'diagnostics.json', evidence)
        raise


if __name__ == '__main__': main()
