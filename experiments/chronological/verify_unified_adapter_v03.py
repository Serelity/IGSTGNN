"""Engineering-only v0.3 recovery, intervention and real-MAE gradient evidence.

--check (also the default) uses four train/validation rows and batch two.
--stress honors --batch-size exactly and runs one 496-node training batch each
for ORIGINAL and BOTH. Neither mode updates parameters, evaluates test data,
certifies model accuracy, or substitutes a smaller batch after an OOM.
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
from experiments.chronological.original_igstgnn_source import get_original_modules, verify_original_source
from experiments.chronological.original_igstgnn_runtime import (
    CONFIG, build_model, configure_runtime, default_data_dir, device_batch,
    load_data, rng_fingerprint, sha256, tensor_tree_hash, write_json)
from experiments.chronological.unified_adapter_data_v03 import (
    audit_data, device_batch_v03, make_loaders)
from experiments.chronological.unified_incident_adapter_v03 import (
    DEFAULT_ARMS, build_variant, common_traffic_state)


SCHEMA = 'unified_adapter_engineering_verification_v03'
EXECUTION_FILES = (
    'experiments/chronological/verify_unified_adapter_v03.py',
    'experiments/chronological/unified_incident_adapter_v03.py',
    'experiments/chronological/unified_adapter_data_v03.py',
    'experiments/chronological/unified_incident_adapter.py',
    'experiments/chronological/original_igstgnn_runtime.py',
    'experiments/chronological/original_igstgnn_source.py')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _synchronize(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


def gradient_groups(model):
    """Real objective gradients, with zero/absent gradients explicitly retained."""
    groups = {}
    for name, parameter in model.named_parameters():
        if not name.startswith('adapter.'):
            group = 'traffic_core'
        else:
            component = name.split('.')[1]
            group = ('report_encoder' if component == 'encoder' else
                'traffic_summary' if component in ('summary', 'pool') else
                'graph_head' if component == 'graph' else
                'time_head' if component in ('output', 'time_interaction') else
                'adoption_uninterpreted_local_gate' if component == 'local_gate' else 'state')
        row = groups.setdefault(group, dict(parameters=0, parameters_with_grad=0,
            parameters_with_nonzero_grad=0, gradient_elements=0, squared_norm=0., finite=True))
        row['parameters'] += 1
        if parameter.grad is not None:
            grad = parameter.grad.detach()
            row['parameters_with_grad'] += 1
            row['gradient_elements'] += grad.numel()
            finite = bool(torch.isfinite(grad).all())
            row['finite'] &= finite
            if finite:
                square = float(grad.double().square().sum())
                row['squared_norm'] += square
                row['parameters_with_nonzero_grad'] += int(square > 0)
    for row in groups.values():
        row['l2_norm'] = row.pop('squared_norm')**.5 if row['finite'] else None
    return groups


def real_mae_backward(model, batch, scaler, metrics, device, *, curriculum_length=12):
    """One native-scale masked-MAE forward/backward, zero optimizer updates."""
    x, target, report, sensors = batch
    require(x.shape[0] == target.shape[0] and 1 <= curriculum_length <= target.shape[1],
        'Invalid real-MAE batch or curriculum')
    model.train()
    model.zero_grad(set_to_none=True)
    inputs_before = tensor_tree_hash((x, target, report, sensors))
    _synchronize(device)
    cuda = torch.device(device).type == 'cuda'
    baseline = torch.cuda.memory_allocated(device) if cuda else None
    if cuda:
        torch.cuda.reset_peak_memory_stats(device)
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record()
    start = perf_counter()
    prediction = model(x, incident_data=report, sensor_data=sensors)
    require(bool(torch.isfinite(prediction).all()), 'Nonfinite model prediction')
    predicted_original = scaler.inverse_transform(prediction)
    target_original = scaler.inverse_transform(target)
    # The native engine chooses its null value before slicing curriculum leads.
    null = target_original.min() if target_original.min() < 1 else target_original.new_tensor(0.)
    loss = metrics.masked_mae(predicted_original[:, :curriculum_length],
        target_original[:, :curriculum_length], null)
    require(bool(torch.isfinite(loss)), 'Nonfinite real masked MAE')
    loss.backward()
    if cuda:
        end.record()
    _synchronize(device)
    wall = perf_counter()-start
    peak_allocated = torch.cuda.max_memory_allocated(device) if cuda else None
    peak_reserved = torch.cuda.max_memory_reserved(device) if cuda else None
    groups = gradient_groups(model)
    require(all(row['finite'] for row in groups.values()), 'Nonfinite real-MAE gradient')
    require(tensor_tree_hash((x, target, report, sensors)) == inputs_before, 'Forward/backward mutated input batch')
    result = dict(actual_batch_size=x.shape[0], nodes=x.shape[2], horizon=target.shape[1],
        curriculum_length=curriculum_length, objective='native_inverse_transform_masked_mae',
        true_targets_used_in_objective=True, true_targets_passed_to_model=False,
        mask_value=float(null), loss=float(loss.detach()), optimizer_updates=0,
        forward_calls=1, backward_calls=1, wall_seconds=wall,
        cuda_milliseconds=begin.elapsed_time(end) if cuda else None,
        baseline_allocated_bytes=baseline,
        peak_allocated_bytes=peak_allocated,
        incremental_peak_allocated_bytes=peak_allocated-baseline if cuda else None,
        peak_reserved_bytes=peak_reserved,
        groups=groups, inputs_preserved=True,
        timing_scope='one_train_forward_native_inverse_loss_backward_excludes_data_H2D_hash_and_gradient_summary',
        initialization_caveat='zero_output_heads_can_intentionally_block_upstream_state_gradient_on_first_backward')
    if hasattr(model, 'batch_diagnostics'):
        result['mechanisms'] = model.batch_diagnostics()
    return result


def zero_recovery(source, batch, arms=DEFAULT_ARMS):
    """Check train/eval predictions, dropout RNG and core tensors independently."""
    x, _, report, sensors = batch
    source_hash, inputs_hash = tensor_tree_hash(source.state_dict()), tensor_tree_hash(batch)
    results = {}
    for training in (False, True):
        reference = build_variant(copy.deepcopy(source), 'TRAFFIC_CORE').train(training)
        _seed(2200+int(training))
        with torch.no_grad():
            expected = reference(x, incident_data=report, sensor_data=sensors)
        expected_rng, expected_core = rng_fingerprint(), tensor_tree_hash(common_traffic_state(reference))
        for arm in arms:
            if arm == 'ORIGINAL':
                continue
            before = rng_fingerprint()
            candidate = build_variant(copy.deepcopy(source), arm).train(training)
            constructor_rng = before == rng_fingerprint()
            require(constructor_rng, arm+' constructor consumes global RNG')
            _seed(2200+int(training))
            with torch.no_grad():
                actual = candidate(x, incident_data=report, sensor_data=sensors)
            exact = torch.equal(actual, expected)
            rng_equal = rng_fingerprint() == expected_rng
            core_equal = tensor_tree_hash(common_traffic_state(candidate)) == expected_core
            require(exact and rng_equal and core_equal, arm+' zero-init core recovery failed')
            results.setdefault(arm, {})['train' if training else 'eval'] = dict(
                prediction_exact=exact, dropout_rng_exact=rng_equal,
                constructor_rng_exact=constructor_rng, traffic_parameters_and_buffers_exact=core_equal,
                prediction_abs_max=float((actual-expected).abs().amax()))
            del candidate, actual
        del reference, expected
    require(tensor_tree_hash(source.state_dict()) == source_hash, 'Zero recovery changed source model')
    require(tensor_tree_hash(batch) == inputs_hash, 'Zero recovery changed inputs')
    return results


def activated_interventions(source, batch, *, scaler=None, metrics=None, device=None):
    """Nonzero random engineering perturbation, never a learned-model claim."""
    x, _, report, sensors = batch
    model = build_variant(copy.deepcopy(source), 'BOTH').eval()
    with torch.random.fork_rng(devices=[x.device.index] if x.is_cuda else []):
        _seed(913)
        with torch.no_grad():
            model.adapter.output.weight.normal_(0, .03)
            for layer in list(model.adapter.graph.query)+list(model.adapter.graph.strength):
                layer.weight.normal_(0, .03)
    with torch.no_grad():
        core = model.core(x)
        on = model(x, incident_data=report, sensor_data=sensors)
        with model.intervention(time=False):
            time_off = model(x, incident_data=report, sensor_data=sensors)
        with model.intervention(graph=False):
            graph_off = model(x, incident_data=report, sensor_data=sensors)
        with model.intervention(time=False, graph=False):
            both_off = model(x, incident_data=report, sensor_data=sensors)
        with model.intervention(report=False):
            report_off = model(x, incident_data=report, sensor_data=sensors)
        absent = dict(report, available=torch.zeros(x.shape[0], dtype=torch.bool, device=x.device),
            distances=torch.full_like(report['distances'], float('nan')))
        absent_prediction = model(x, incident_data=absent, sensor_data=sensors)
        target_a = model(x, torch.zeros_like(batch[1]), incident_data=report, sensor_data=sensors)
        target_b = model(x, torch.ones_like(batch[1])*999, incident_data=report, sensor_data=sensors)
    result = dict(engineering_parameter_perturbation_not_training=True, optimizer_updates=0,
        time_response_abs_max=float((on-time_off).abs().amax()),
        graph_response_abs_max=float((on-graph_off).abs().amax()),
        both_heads_off_recovers_core_exact=torch.equal(both_off, core),
        report_off_recovers_core_exact=torch.equal(report_off, core),
        absent_nan_report_recovers_core_exact=torch.equal(absent_prediction, core),
        target_independence_exact=torch.equal(target_a, target_b),
        first_graph_unmodified_block_exact=torch.equal(time_off[:, :3], core[:, :3]))
    for key in ('both_heads_off_recovers_core_exact', 'report_off_recovers_core_exact',
            'absent_nan_report_recovers_core_exact', 'target_independence_exact', 'first_graph_unmodified_block_exact'):
        require(result[key], 'Activated intervention failed: '+key)
    require(result['time_response_abs_max'] > 0 and result['graph_response_abs_max'] > 0,
        'Activated heads did not change any predictions in this batch')
    if scaler is not None or metrics is not None:
        require(scaler is not None and metrics is not None and device is not None,
            'Supply scaler, metrics and device together for perturbed real-MAE evidence')
        gradients = real_mae_backward(model, batch, scaler, metrics, device)
        for name in ('report_encoder', 'traffic_summary', 'state', 'graph_head', 'time_head',
                'adoption_uninterpreted_local_gate'):
            require(gradients['groups'][name]['l2_norm'] > 0,
                'Perturbed real MAE did not reach '+name)
        result['perturbed_real_mae_backward'] = gradients
        result['perturbation_caveat'] = 'manual_nontrained_head_perturbation_proves_path_connection_not_learning_or_accuracy'
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, default=default_data_dir())
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=48)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--stress', action='store_true')
    modes.add_argument('--check', action='store_true')
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.threads < 1:
        parser.error('batch-size and threads must be positive')
    args.output, args.data_dir = args.output.resolve(), args.data_dir.resolve()
    return args


def is_out_of_memory(error):
    return isinstance(error, torch.cuda.OutOfMemoryError) or (
        isinstance(error, RuntimeError) and 'out of memory' in str(error).lower())


def main(argv=None):
    args = parse_args(argv)
    require(not args.output.exists(), 'Use a fresh evidence output path')
    evidence = dict(schema=SCHEMA, status='UNIFIED_ADAPTER_V03_ENGINEERING_RUNNING',
        engineering_only=True, mode='stress' if args.stress else 'check',
        requested_batch_size=args.batch_size, actual_batch_size=None,
        requested_batch_fallback_permitted=False, optimizer_updates=0,
        test_evaluated=False, accuracy_comparison_performed=False,
        initialization={}, real_mae_backward={})
    try:
        evidence['runtime'] = configure_runtime(args.device, args.threads, 2025)
        evidence['source_before'] = verify_original_source()
        evidence['execution_source'] = {name: sha256(REPO/name) for name in EXECUTION_FILES}
        modules = get_original_modules()
        config = dict(CONFIG, bs=args.batch_size if args.stress else 2)
        data = load_data(args.data_dir, config, modules, logging.getLogger('adapter-v03-check'),
            check=not args.stress, training=True)
        evidence['data'] = data.identity
        evidence['data_audit'] = audit_data(data)
        candidates = make_loaders(data, modules, logging.getLogger('adapter-v03-check'))
        original_raw = next(data.loaders['train_loader'].get_iterator())
        raw = next(candidates['train_loader'].get_iterator())
        candidate_batch = device_batch_v03(raw, args.device)
        actual = candidate_batch[0].shape[0]
        evidence['actual_batch_size'] = actual
        evidence['batch_policy'] = 'exact_requested_single_train_batch' if args.stress else 'explicit_smoke_four_rows_batch_two'
        require(actual == (args.batch_size if args.stress else 2), 'Actual batch differs from engineering mode contract')
        require(candidate_batch[0].shape[2] == 496, 'Published verification requires the real 496-node axis')
        evidence['support'] = dict(available_reports=int(candidate_batch[2]['available'].sum()),
            relation_available=int(candidate_batch[2]['relation_available'].sum()),
            supported_node_positions=int((candidate_batch[2]['distances'].abs().sum(-1) > 0).sum()))
        if not args.stress:
            source = build_model(data, modules, args.device)
            evidence['initialization'] = zero_recovery(source, candidate_batch)
            evidence['activated_interventions'] = activated_interventions(source, candidate_batch,
                scaler=data.scaler, metrics=modules.metrics, device=args.device)
            del source
        for arm in ('ORIGINAL', 'BOTH'):
            evidence['phase'] = arm+'_real_mae_backward'
            configure_runtime(args.device, args.threads, 2025)
            source = build_model(data, modules, args.device)
            model = build_variant(source, arm)
            batch = device_batch(original_raw, args.device) if arm == 'ORIGINAL' else candidate_batch
            evidence['real_mae_backward'][arm] = real_mae_backward(model, batch, data.scaler,
                modules.metrics, args.device)
            if arm == 'BOTH':
                groups = evidence['real_mae_backward'][arm]['groups']
                require(groups['time_head']['l2_norm'] > 0 and groups['graph_head']['l2_norm'] > 0,
                    'Real MAE must reach both output paths at zero initialization')
            del model, source, batch
            gc.collect()
            if torch.device(args.device).type == 'cuda':
                torch.cuda.empty_cache()
        evidence['source_after'] = verify_original_source()
        require(evidence['source_after'] == evidence['source_before'], 'Frozen source changed')
        evidence['phase'] = 'complete'
        evidence['status'] = 'UNIFIED_ADAPTER_V03_ENGINEERING_PASS'
        return evidence
    except BaseException as error:
        evidence.update(status='UNIFIED_ADAPTER_V03_ENGINEERING_OOM' if is_out_of_memory(error)
            else 'UNIFIED_ADAPTER_V03_ENGINEERING_FAILED', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, evidence)
        print(json.dumps(dict(status=evidence['status'], output=str(args.output),
            actual_batch_size=evidence['actual_batch_size'], engineering_only=True, optimizer_updates=0)), flush=True)


if __name__ == '__main__':
    main()
