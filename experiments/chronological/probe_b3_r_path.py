"""Best-checkpoint D-path sensitivity: verified ON, then paired ON/OFF/ON.

This standalone inference tool never changes the frozen training implementation.
OFF removes the entire history/report-conditioned D path; original TIID stays ON.
Latent exchange values are not traffic counts or identified road capacities.
"""
import argparse
from datetime import datetime
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import uuid

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.chronological import train_b3_r as training
from experiments.chronological import train_internal_spatial as legacy
from experiments.chronological.b3_r_diagnostics import CapacityDiagnostics
from experiments.chronological.smoke import device_batch, sha256, verify_package
from experiments.chronological.train import batches, validate_protocol
from src.models import incident_spatial as spatial
from src.models.incident_spatial import require, SpatialFeatureExchange
from src.models.incident_spatial_residual import CONTEXT_SCHEMA, ReportCapacityResidual, tensor_state_hash
from src.utils.chronological import ChronologicalDataset

SCHEMA = 'b3r_d_path_probe_v1'
REPLAY_ATOL, REPLAY_RTOL = 1e-4, 1e-6
THRESHOLDS = (1e-7, 1e-5, 1e-3, 1e-2, .1)
_TRACE_ACTIVE = False


def tree_hash(value):
    digest = hashlib.sha256()
    def visit(item):
        if isinstance(item, dict):
            for name in sorted(item):
                digest.update(name.encode())
                visit(item[name])
        elif isinstance(item, (list, tuple)):
            for entry in item:
                visit(entry)
        elif isinstance(item, torch.Tensor):
            array = item.detach().cpu().contiguous().numpy()
            digest.update(str((array.dtype.str, array.shape)).encode())
            digest.update(array.tobytes())
        else:
            digest.update(repr(item).encode())
    visit(value)
    return digest.hexdigest()


class ChangeTotals:
    """Weighted element statistics, retaining no full-dataset exchange tensors."""
    def __init__(self):
        self.rows = {}

    def add(self, name, on, off, mask=None):
        on, off = np.asarray(on), np.asarray(off)
        require(on.shape == off.shape, 'Paired tensor axes differ: '+name)
        if mask is not None:
            chosen = np.broadcast_to(mask, on.shape)
            on, off = on[chosen], off[chosen]
        delta = off.astype(np.float64)-on.astype(np.float64)
        require(np.isfinite(delta).all(), 'Non-finite paired tensor: '+name)
        row = self.rows.setdefault(name, dict(count=0, absolute_sum=0., signed_sum=0., absolute_max=0.,
                                            above={str(t): 0 for t in THRESHOLDS}))
        absolute = np.abs(delta)
        row['count'] += delta.size
        row['absolute_sum'] += float(absolute.sum())
        row['signed_sum'] += float(delta.sum())
        if delta.size:
            row['absolute_max'] = max(row['absolute_max'], float(absolute.max()))
        for t in THRESHOLDS:
            row['above'][str(t)] += int((absolute > t).sum())

    def result(self):
        return {name: dict(status='available' if row['count'] else 'unavailable',
                           **dict(row, absolute_max=row['absolute_max'] if row['count'] else None),
                           absolute_mean=row['absolute_sum']/row['count'] if row['count'] else None,
                           signed_mean_off_minus_on=row['signed_sum']/row['count'] if row['count'] else None,
                           response_fraction_above={t: n/row['count'] if row['count'] else None
                                                    for t, n in row['above'].items()})
                for name, row in sorted(self.rows.items())}


def exchange_summary(changes):
    totals = ChangeTotals()
    for name, row in changes.rows.items():
        if not name.endswith('/all'):
            continue
        layer, _, _, field, _ = name.split('/')
        for key in (field, layer+'/'+field):
            old = totals.rows.setdefault(key, dict(count=0, absolute_sum=0., signed_sum=0., absolute_max=0.,
                                                   above={str(t): 0 for t in THRESHOLDS}))
            for statistic in ('count', 'absolute_sum', 'signed_sum'):
                old[statistic] += row[statistic]
            old['absolute_max'] = max(old['absolute_max'], row['absolute_max'])
            for threshold, count in row['above'].items():
                old['above'][threshold] += count
    return totals.result()


class ExchangeTrace:
    """Observe the original function and readout, restoring all hooks on exit.

    Record at most one ON batch on CPU; stream OFF against it, then verify final
    ON exactly. No exchange formula is reimplemented and outputs are untouched.
    """
    def __init__(self, model, reference=None, differences=None, restoration=False):
        self.model, self.reference = model, reference
        self.differences, self.restoration = differences, restoration
        self.tensors, self.handles, self.seen = {}, [], set()
        self.calls, self.frame, self.bytes = [0]*5, None, 0
        self.d_calls = 0

    def __enter__(self):
        global _TRACE_ACTIVE
        require(not _TRACE_ACTIVE, 'Nested exchange instrumentation is forbidden')
        require(not any(m.training for m in self.model.modules()), 'Probe requires eval mode')
        require(type(self.model.report_spatial_state) is ReportCapacityResidual
                and self.model.report_spatial_state.report_path_enabled, 'Expected B3R ON architecture')
        expected = {f'layers.{i}.dif_layer.localized_st_conv.incident_exchange' for i in range(5)}
        found = {name for name, m in self.model.named_modules() if isinstance(m, SpatialFeatureExchange)}
        require(found == expected, 'Incomplete or misplaced five-layer exchange coverage')
        self.original = spatial.exchange_step
        _TRACE_ACTIVE = True
        try:
            for layer in range(5):
                module = self.model.get_submodule(f'layers.{layer}.dif_layer.localized_st_conv.incident_exchange')
                require(module.steps == 2 and module.restricted and module.eta == .5, 'Wrong exchange architecture')
                self.handles.append(module.register_forward_pre_hook(self.pre(layer)))
                self.handles.append(module.register_forward_hook(self.post))
                self.handles.append(module.readout.register_forward_hook(self.readout))
            self.handles.append(self.model.report_spatial_state.D.register_forward_hook(self.count_d))
            spatial.exchange_step = self.step
        except BaseException:
            self.cleanup()
            raise
        return self

    def count_d(self, module, args, output):
        self.d_calls += 1

    def pre(self, layer):
        def hook(module, args):
            require(self.frame is None and not torch.is_grad_enabled(), 'Nested exchange or autograd enabled')
            hidden, context = args
            require(context['schema'] == CONTEXT_SCHEMA, 'Unsupported context schema')
            call = self.calls[layer]
            self.calls[layer] += 1
            self.frame = dict(layer=layer, call=call, step=0, context=context, b=hidden.shape[0], w=hidden.shape[1])
        return hook

    def post(self, module, args, output):
        require(self.frame is not None and self.frame['step'] == 2, 'Incomplete exchange substeps')
        self.frame = None

    def observe(self, field, tensor, step):
        f = self.frame
        key = (f['layer'], f['call'], step, field)
        require(key not in self.seen, 'Repeated trace key')
        self.seen.add(key)
        array = tensor.detach().cpu().numpy().copy()
        require(np.isfinite(array).all(), 'Non-finite exchange trace')
        if self.reference is None:
            self.bytes += array.nbytes
            require(self.bytes <= 2*1024**3, 'One-batch CPU trace exceeds 2 GiB safety bound; batch must remain fixed')
            self.tensors[key] = array
            return
        require(key in self.reference and self.reference[key].shape == array.shape, 'Trace axes or coverage changed')
        on = self.reference[key]
        if self.restoration:
            require(np.array_equal(on, array), 'Final ON exchange restoration failed: '+str(key))
            return
        require(self.differences is not None, 'Missing paired accumulator')
        name = f'layer_{key[0]}/call_{key[1]}/step_{step}/{field}'
        groups = {'all': None}
        context, graph = f['context'], f['context']['graph']
        associated = context['report_mask'].detach().cpu().numpy().astype(bool)
        if field == 'readout':
            groups.update(operator=graph.operator_mask.cpu().numpy()[None, None, :, None],
                          report_associated=associated[:, None])
        else:
            src, dst = graph.edge_index.cpu().numpy()
            edge_report = associated[:, np.maximum(src, 0)] | associated[:, np.maximum(dst, 0)]
            edge_report[:, src < 0] = associated[:, dst[src < 0]]
            edge_report[:, dst < 0] = associated[:, src[dst < 0]]
            groups.update(internal=((src >= 0) & (dst >= 0))[None, :, None],
                          inlet=(src < 0)[None, :, None], exit=(dst < 0)[None, :, None],
                          report_endpoint=np.broadcast_to(edge_report[:, None],
                              (f['b'], f['w'], graph.edges, 1)).reshape(f['b']*f['w'], graph.edges, 1))
        for group, mask in groups.items():
            self.differences.add(name+'/'+group, on, array, mask)

    def step(self, state, proposal, capacity, graph, **kwargs):
        require(self.frame is not None and graph is self.frame['context']['graph'], 'Unowned exchange function call')
        step = self.frame['step']
        result = self.original(state, proposal, capacity, graph, **kwargs)
        self.observe('proposal', proposal, step)
        self.observe('flux', result['flux'], step)
        self.frame['step'] += 1
        return result

    def readout(self, module, args, output):
        require(self.frame is not None and self.frame['step'] == 2, 'Misplaced exchange readout')
        f = self.frame
        self.observe('readout', output.reshape(f['b'], f['w'], *output.shape[1:]), -1)

    def cleanup(self):
        global _TRACE_ACTIVE
        spatial.exchange_step = self.original
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        _TRACE_ACTIVE = False

    def __exit__(self, exc_type, exc, tb):
        self.cleanup()
        if exc_type is None:
            require(self.frame is None and self.calls == [4]*5 and len(self.seen) == 100,
                    'Incomplete 20 calls / 40 substeps / 20 readouts')
            if self.reference is not None:
                require(self.seen == self.reference.keys(), 'Trace key coverage changed')
        return False


def forward(model, batch, scaler, enabled, trace=None):
    require(not any(m.training for m in model.modules()), 'Probe requires eval mode')
    require(all(p.grad is None for p in model.parameters()), 'Probe found existing parameter gradients')
    with torch.inference_mode():
        result = model(batch['x'], incident_data=batch['incident'], spatial_report_enabled=enabled,
                       tiid_report_enabled=True, return_spatial_details=True)
        prediction = result['prediction']*scaler['std']+scaler['mean']
    context = result['spatial']
    require(context['report_path_executed'] is enabled, 'D-path switch did not take effect')
    if trace is not None:
        require(trace.d_calls == int(enabled), 'D execution coverage mismatch')
    require(bool(torch.isfinite(prediction).all()), 'Non-finite prediction')
    return prediction.detach().cpu().numpy().copy(), context


def paired_metrics(on, off, target, valid, mask):
    require(on.shape == off.shape == target.shape == valid.shape and on.ndim == 4 and on.shape[-1] == 1,
            'Prediction/target/valid axes mismatch')
    require(valid.dtype == bool and mask.dtype == bool and mask.shape == (on.shape[0], on.shape[2]), 'Region axes mismatch')
    chosen = valid & mask[:, None, :, None]
    counts = chosen.sum((0, 2, 3))
    on_error = np.where(chosen, np.abs(on.astype(np.float64)-target), 0.).sum((0, 2, 3))
    off_error = np.where(chosen, np.abs(off.astype(np.float64)-target), 0.).sum((0, 2, 3))
    on_h = [float(e/n) if n else None for e, n in zip(on_error, counts)]
    off_h = [float(e/n) if n else None for e, n in zip(off_error, counts)]
    gains = [b-a if a is not None else None for a, b in zip(on_h, off_h)]
    available = bool((counts > 0).all())
    changes = ChangeTotals()
    changes.add('prediction', on, off, chosen)
    return dict(status='available' if available else 'unavailable', counts=counts.tolist(),
                on_mae_by_horizon=on_h, off_mae_by_horizon=off_h, off_minus_on_by_horizon=gains,
                on_mae_macro=float(np.mean(on_h)) if available else None,
                off_mae_macro=float(np.mean(off_h)) if available else None,
                off_minus_on_mae=float(np.mean(gains)) if available else None,
                prediction_change=changes.result()['prediction'])


def verify_saved_windows(prediction, target, valid, sample_indices, t0, saved, expected_mae):
    require(prediction.shape == target.shape == valid.shape and prediction.ndim == 4, 'Replay axes mismatch')
    require(np.array_equal(sample_indices, saved['sample_indices']) and np.array_equal(t0, saved['t0']),
            'Saved replay sample/time order differs')
    counts = valid.sum((2, 3))
    absolute = np.where(valid, np.abs(prediction.astype(np.float64)-target), 0.).sum((2, 3))
    require(absolute.shape == saved['absolute'].shape and np.array_equal(counts, saved['counts']),
            'Saved replay error axes/counts differ')
    require(np.isfinite(saved['absolute']).all() and (saved['absolute'] >= 0).all(), 'Invalid saved window errors')
    # Compare normalized window errors so tolerance does not grow with station count.
    observed = np.divide(absolute, counts, out=np.zeros_like(absolute), where=counts > 0)
    previous = np.divide(saved['absolute'], counts, out=np.zeros_like(absolute), where=counts > 0)
    require(np.allclose(observed, previous, atol=REPLAY_ATOL, rtol=REPLAY_RTOL),
            'ON replay differs from saved window errors; OFF blocked')
    horizon_counts = counts.sum(0)
    require((horizon_counts > 0).all(), 'Missing replay horizon targets')
    mae = float(np.mean(absolute.sum(0)/horizon_counts))
    saved_mae = float(np.mean(saved['absolute'].sum(0)/horizon_counts))
    require(abs(saved_mae-expected_mae) <= 1e-10, 'Saved windows disagree with best metric')
    require(abs(mae-expected_mae) <= REPLAY_ATOL, 'ON replay MAE failed; OFF blocked')
    return dict(status='BEST_ON_WINDOW_REPLAY_PASS', on_mae_macro=mae, saved_mae_macro=expected_mae,
                window_mae_abs_difference_max=float(np.abs(observed-previous).max()),
                atol=REPLAY_ATOL, rtol=REPLAY_RTOL,
                evidence='saved_window_horizon_absolute_errors_counts_order_not_saved_elementwise_predictions')


def capacity_changes(on, off, changes):
    for name in ('base_logit', 'base_node_capacity', 'base_edge_capacity', 'boundary', 'report_mask'):
        require(torch.equal(on[name], off[name]), 'OFF changed report-independent context: '+name)
    require(off['report_logit_delta'] is None and not bool(off['effective_logit_delta'].any()), 'OFF did not skip D')
    require(torch.equal(off['node_capacity'], off['base_node_capacity']), 'OFF node capacity is not P')
    graph, mask = on['graph'], on['report_mask'].cpu().numpy().astype(bool)
    src, dst = graph.edge_index.cpu().numpy()
    edge_mask = mask[:, np.maximum(src, 0)] | mask[:, np.maximum(dst, 0)]
    edge_mask[:, src < 0] = mask[:, dst[src < 0]]
    edge_mask[:, dst < 0] = mask[:, src[dst < 0]]
    for field, groups in (('node_capacity', {'all': None, 'report_associated': mask}),
                          ('capacity', {'all': None, 'report_endpoint': edge_mask,
                                        'internal': ((src >= 0) & (dst >= 0))[None, :, None],
                                        'inlet': (src < 0)[None, :, None], 'exit': (dst < 0)[None, :, None]})):
        for group, chosen in groups.items():
            changes.add(field+'/'+group, on[field].cpu().numpy(), off[field].cpu().numpy(), chosen)


def run_probe(model, dataset, indices, batch_size, device, graph, roads, saved_windows, expected_mae, output):
    """Four validation passes: complete uninstrumented replay, then batched O/O/O."""
    require(not any(m.training for m in model.modules()), 'Probe requires eval mode')
    require(list(indices) == [int(r['sample_index']) for r in dataset.rows[:len(indices)]],
            'Probe must preserve original validation manifest order')
    state_before = tensor_state_hash(model.state_dict())
    switch_before = model.report_spatial_state.report_path_enabled
    on_parts, targets, valid_parts, associated_parts = [], [], [], []
    for raw in batches(dataset, indices, batch_size):
        batch = device_batch(raw, device)
        before = tree_hash(batch)
        prediction, _ = forward(model, batch, dataset.scaler, True)
        require(before == tree_hash(batch), 'Replay changed inputs')
        on_parts.append(prediction)
        targets.append(batch['y_flow'].cpu().numpy().copy())
        valid_parts.append(batch['y_valid'].cpu().numpy().copy())
        associated_parts.append((batch['incident']['distances'].abs().sum(-1) > 0).cpu().numpy())
        print(f'Verified ON replay: {sum(len(part) for part in on_parts)}/{len(indices)} rows', flush=True)
    on, target, valid = np.concatenate(on_parts), np.concatenate(targets), np.concatenate(valid_parts)
    associated = np.concatenate(associated_parts)
    del on_parts, targets, valid_parts, associated_parts, batch, raw
    t0 = np.array([r['t0'] for r in dataset.rows[:len(indices)]])
    replay = verify_saved_windows(on, target, valid, np.array(indices), t0, saved_windows, expected_mae)
    require(tensor_state_hash(model.state_dict()) == state_before, 'Replay changed model state; OFF blocked')
    legacy.write_json(output/'replay.json', replay)
    print('Best ON replay passed; permitting entire D-path OFF (TIID remains ON)', flush=True)
    differences, capacity = ChangeTotals(), ChangeTotals()
    on_diagnostics, off_diagnostics = CapacityDiagnostics(), CapacityDiagnostics()
    off, position, trace_peak = np.empty_like(on), 0, 0
    partition = []
    for raw in batches(dataset, indices, batch_size):
        batch = device_batch(raw, device)
        before = tree_hash(batch)
        with ExchangeTrace(model) as observed:
            paired_on, on_context = forward(model, batch, dataset.scaler, True, observed)
        end = position+len(paired_on)
        require(np.array_equal(paired_on, on[position:end]), 'Instrumented ON differs from verified ON replay')
        trace_peak = max(trace_peak, observed.bytes)
        with ExchangeTrace(model, observed.tensors, differences) as disabled:
            paired_off, off_context = forward(model, batch, dataset.scaler, False, disabled)
        capacity_changes(on_context, off_context, capacity)
        on_diagnostics.update(on_context)
        off_diagnostics.update(off_context)
        with ExchangeTrace(model, observed.tensors, restoration=True) as restored:
            final_on, restored_context = forward(model, batch, dataset.scaler, True, restored)
        require(np.array_equal(paired_on, final_on), 'Final ON prediction restoration failed')
        for field in ('base_logit', 'node_capacity', 'capacity', 'effective_logit_delta', 'boundary'):
            require(torch.equal(on_context[field], restored_context[field]), 'Final ON context restoration failed')
        require(before == tree_hash(batch), 'Paired inference changed inputs')
        off[position:end] = paired_off
        partition.append(end-position)
        position = end
        print(f'Paired ON/OFF/ON: {position}/{len(indices)} rows', flush=True)
        del observed, disabled, restored, batch, raw, on_context, off_context, restored_context, paired_on, paired_off, final_on
        gc.collect()
    require(position == len(indices), 'Incomplete paired validation')
    require(tensor_state_hash(model.state_dict()) == state_before
            and model.report_spatial_state.report_path_enabled is switch_before
            and all(p.grad is None for p in model.parameters()), 'Inference changed model state/switch/gradients')
    # Use the trainer's Torch operation; avoid starting a second BLAS runtime.
    with torch.inference_mode():
        potential = (torch.from_numpy(associated).float() @ legacy.component_reach(graph).cpu() > 0).numpy()
    operator = graph.operator_mask.cpu().numpy()
    masks = dict(all_nodes=np.ones_like(associated), report_associated=associated,
                 candidate_potential=potential, potential_only=potential & ~associated,
                 direct_outside_operator=associated & ~operator[None],
                 outside_operator=np.broadcast_to(~operator, associated.shape),
                 no_report_samples=np.broadcast_to(~associated.any(1)[:, None], associated.shape))
    for road, nodes in roads.items():
        mask = np.zeros_like(associated)
        mask[:, nodes] = True
        masks['road:'+road] = mask
    metrics = {name: paired_metrics(on, off, target, valid, mask) for name, mask in masks.items()}
    np.savez_compressed(output/'paired_predictions.npz', on=on, off=off, target=target, valid=valid,
                        sample_indices=indices, t0=t0, report_associated=associated,
                        candidate_potential=potential, operator_mask=operator, final_on_exactly_restored=True)
    legacy.write_json(output/'exchange_changes.json', differences.result())
    exchange_totals = exchange_summary(differences)
    legacy.write_json(output/'exchange_summary.json', exchange_totals)
    legacy.write_json(output/'capacity_changes.json', capacity.result())
    legacy.write_json(output/'on_mechanisms.json', on_diagnostics.result())
    legacy.write_json(output/'off_mechanisms.json', off_diagnostics.result())
    compact = {field: exchange_totals[field]['absolute_mean'] for field in ('proposal', 'flux', 'readout')}
    compact.update(node_capacity_report=capacity.result()['node_capacity/report_associated']['absolute_mean'],
                   edge_capacity_report=capacity.result()['capacity/report_endpoint']['absolute_mean'])
    return dict(replay=replay, metrics=metrics, chain_mean_absolute_change=compact,
                batch_partition=partition, peak_one_batch_trace_bytes=trace_peak,
                model_state_sha256=state_before, final_on_exactly_restored=True, inputs_preserved=True,
                inference_only=True, tiid_report_enabled=True, test_read=False,
                trace_coverage_per_batch=dict(layers=5, calls_per_layer=4, substeps_per_call=2, readouts=20),
                metric_sign='MAE_OFF_minus_MAE_ON_positive_means_D_helpful_at_this_checkpoint',
                group_scope='candidate_potential_is_operator_supported_component_envelope; excludes_direct_reports_outside_operator; groups_overlap_not_additive',
                exchange_units='dimensionless_latent_values; readout_in_hidden_feature_units',
                prediction_units='original_flow_target_units')


def select_best(checkpoint, identity, history, summary):
    require(checkpoint.get('identity') == identity and checkpoint.get('arm') == 'B3R', 'Checkpoint identity/arm mismatch')
    require(checkpoint['history'] == history and len(history) == checkpoint['completed_epoch'], 'Checkpoint/history mismatch')
    require(history and [r['epoch'] for r in history] == list(range(1, len(history)+1)), 'Incomplete epoch history')
    metrics = [r['validation_mae_macro'] for r in history]
    require(np.isfinite(metrics).all(), 'Non-finite selection metric')
    best = int(np.argmin(metrics))+1
    require(history[-1]['best_epoch'] == best and checkpoint['best_metric'] == metrics[best-1]
            and checkpoint['best_model_state'] is not None, 'Best checkpoint selection mismatch')
    require(summary['arm'] == 'B3R' and summary['completed_epoch'] == len(history)
            and summary['best_epoch'] == best and summary['best_metric'] == metrics[best-1], 'Summary/checkpoint mismatch')
    return best, metrics[best-1]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--data-dir', type=Path, default=legacy.default_data())
    parser.add_argument('--graph-dir', type=Path, default=training.HERE/'internal_spatial_graph_is11')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--threads', type=int, default=3)
    args = parser.parse_args(argv)
    require(args.threads >= 1, 'Threads must be positive')
    for name in ('run_dir', 'data_dir', 'graph_dir'):
        setattr(args, name, getattr(args, name).resolve())
    return args


def main(argv=None):
    args = parse_args(argv)
    require(args.run_dir.is_dir(), 'Run directory does not exist')
    output = args.run_dir/('b3r_path_probe_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6])
    output.mkdir()
    print('Probe directory: '+str(output), flush=True)
    try:
        with legacy.run_lock(args.run_dir):
            report = execute(args, output)
        legacy.write_json(output/'summary.json', report)
        print(json.dumps(dict(status=report['status'], best_epoch=report['best_epoch'],
                             all_nodes=report['metrics']['all_nodes'], chain_mean_absolute_change=report['chain_mean_absolute_change'],
                             output=str(output/'summary.json')), indent=2), flush=True)
    except BaseException as error:
        legacy.write_json(output/'failure.json', dict(status='B3R_PATH_PROBE_FAILED', error_type=type(error).__name__,
                          message=str(error), completed=False, test_read=False))
        raise


def execute(args, output):
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    require(device.type in ('cpu', 'cuda') and (device.type != 'cuda' or torch.cuda.is_available()), 'Requested device unavailable')
    identity = legacy.read_json(args.run_dir/'identity.json')
    require(identity.get('schema') == 'b3r_run_v1' and identity.get('tiid_report_enabled') is True
            and identity.get('arm_modes', {}).get('B3R') == 'pd_report_on' and identity.get('test_read') is False,
            'Expected original B3R development run identity')
    require(training.source_fingerprints() == identity['source_sha256'], 'Frozen training source fingerprints changed')
    protocol_path = training.HERE/'internal_spatial_b3_r_v11.json'
    require(sha256(protocol_path) == identity['protocol_sha256'], 'Frozen protocol fingerprint changed')
    protocol = legacy.read_json(protocol_path)
    training.validate_architecture_protocol(protocol)
    for filename, key in (('summary.json', 'files'), ('context_manifest.json', 'outputs')):
        names = legacy.read_json(args.data_dir/filename)[key]
        require(all(Path(name).name == name and not name.lower().startswith('test') for name in names),
                'Manifest attempts to read a non-development path')
    hashes = verify_package(args.data_dir)
    train, val = ChronologicalDataset(args.data_dir, 'train'), ChronologicalDataset(args.data_dir, 'val')
    validate_protocol(protocol, hashes, len(train), len(val), len(train.station_ids))
    graph, roads, graph_info = legacy.load_graph(args.graph_dir, train.station_ids, protocol['graph_sha256'])
    original_args = SimpleNamespace(**vars(args), seed=identity['seed'], check=identity['engineering_check'],
                                    arms=list(identity['arm_modes']), protocol=protocol_path)
    initialization = training.preflight(original_args, train, graph)
    require(training.make_identity(original_args, protocol, hashes, graph_info, initialization) == identity,
            'Reconstructed run identity differs (runtime, data, graph, threads or initialization)')
    require(training.initialization_identity(legacy.read_json(args.run_dir/'initialization.json'))
            == training.initialization_identity(initialization), 'Reconstructed initialization differs')
    del train
    paths = [args.run_dir/name for name in ('identity.json', 'initialization.json', 'B3R/last.pt', 'B3R/history.json', 'B3R/summary.json')]
    before = {str(path.relative_to(args.run_dir)): sha256(path) for path in paths}
    checkpoint = torch.load(args.run_dir/'B3R/last.pt', map_location='cpu', weights_only=False)
    history = legacy.read_json(args.run_dir/'B3R/history.json')
    best, expected_mae = select_best(checkpoint, identity, history, legacy.read_json(args.run_dir/'B3R/summary.json'))
    windows_path = args.run_dir/'B3R'/f'validation_windows_epoch_{best:03d}.npz'
    paths.append(windows_path)
    before[str(windows_path.relative_to(args.run_dir))] = sha256(windows_path)
    with np.load(windows_path, allow_pickle=False) as archive:
        saved_windows = {key: archive[key].copy() for key in ('sample_indices', 't0', 'absolute', 'counts')}
    model = training.build_arm(original_args, 'B3R', len(val.station_ids), graph, device).eval()
    model.load_state_dict(checkpoint['best_model_state'], strict=True)
    best_hash = tensor_state_hash(checkpoint['best_model_state'])
    completed = checkpoint['completed_epoch']
    del checkpoint, history
    gc.collect()
    environment = legacy.environment(device)
    legacy.write_json(output/'provenance.json', dict(schema=SCHEMA, best_epoch=best, completed_epoch=completed,
                      weights='B3R/last.pt:best_model_state', best_model_state_sha256=best_hash,
                      saved_files_sha256=before, source_sha256=identity['source_sha256'],
                      probe_source_sha256=sha256(Path(__file__)), environment=environment,
                      original_environment=legacy.read_json(args.run_dir/'initial_environment.json'),
                      identity=identity, test_read=False))
    indices = [int(r['sample_index']) for r in val.rows[:4 if identity['engineering_check'] else len(val)]]
    print(f'Verifying best epoch {best} ON validation ({len(indices)} rows, original batch {identity["batch_size"]})', flush=True)
    result = run_probe(model, val, indices, identity['batch_size'], device, graph, roads, saved_windows, expected_mae, output)
    after = {str(path.relative_to(args.run_dir)): sha256(path) for path in paths}
    graph_after = {name: sha256(args.graph_dir/name) for name in identity['graph_inputs_sha256']}
    source_preserved = training.source_fingerprints() == identity['source_sha256'] and sha256(protocol_path) == identity['protocol_sha256']
    legacy.write_json(output/'preservation.json', dict(before=before, after=after, source_preserved=source_preserved,
                      graph_before=identity['graph_inputs_sha256'], graph_after=graph_after))
    require(after == before and source_preserved and graph_after == identity['graph_inputs_sha256']
            and verify_package(args.data_dir) == hashes,
            'Source/input/training artifact changed during inference; completion blocked')
    require(tensor_state_hash(model.state_dict()) == best_hash, 'Loaded best model state changed')
    return dict(status='B3R_PATH_PROBE_PASS', schema=SCHEMA, best_epoch=best, completed_training_epoch=completed,
                scope='engineering_only' if identity['engineering_check'] else 'conditional_offline_checkpoint_sensitivity',
                attribution='entire_D_path_including_history_Q_capacity_and_budget_reallocation; original_TIID_ON',
                predictive_gain_claim=False, accident_only_attribution=False, off_retraining_claim=False,
                source_artifacts_preserved=True, **result)


if __name__ == '__main__':
    main()
