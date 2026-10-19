"""Server-only, no-training replay of certified GW6 JOINT BEST/LAST layers.

Numerical helpers import only NumPy. Model imports occur inside ``main``.
Inputs are read-only; a completed report is written to a fresh run child.
"""
import argparse
from datetime import datetime, timezone
import gc
import json
import logging
import os
from pathlib import Path
import sys

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import numpy as np

from experiments.graph_memory_gw6 import diagnose_drift as drift


ARM = 'JOINT_MEMORY'
SCHEMA = 'gw6_layer_diagnostic_v1'
LIMITATIONS = [
    'Only the first one or two unchanged validation batches are replayed. Their summaries are not full-validation model scores or evidence that drift begins at a particular epoch.',
    'BEST18 and LAST20 are endpoint checkpoints; no epoch19 checkpoint trajectory is inferred.',
    'History-amplitude bins use only origin-available input numerical features on this fixed small subset, never future truth. These boundaries are descriptive and are not trained gates.',
    'Component time indices refer to their own native axes: direct/fused have 12 positions, INH/TIID have four blocks, backcasts have layer-specific historical lengths.',
    'Activation RMS and mean changes do not establish a causal source of prediction deterioration. Statistics differences are not elementwise difference RMS.',
    'Cross-checkpoint heads can break coadaptation and create out-of-distribution representations. The 2x2 results are conditional sensitivity checks, not causal attribution or deployable models.',
    'DistanceFunction BN is evaluated using saved running buffers. No recalibration, train mode, optimizer, gradient update or checkpoint write occurs.',
    'The authenticated author array includes test payloads when deserialized; only the original train/validation loaders are constructed, and only validation batches are forwarded. No test loader or test evaluation is performed.',
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def compare_arrays(actual, expected, atol=2e-5, rtol=1e-6):
    """Reject axis/nonfinite errors; report an explicit numeric replay tolerance."""
    actual, expected = np.asarray(actual), np.asarray(expected)
    require(actual.shape == expected.shape and actual.size > 0, 'Replay arrays must have the same nonempty shape')
    require(np.isfinite(actual).all() and np.isfinite(expected).all(), 'Replay contains nonfinite values')
    error = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    return dict(passed=bool(np.all(error <= atol + rtol * np.abs(expected))),
        max_absolute_error=float(error.max()), mean_absolute_error=float(error.mean()),
        atol=float(atol), rtol=float(rtol), shape=list(actual.shape))


def historical_amplitude(x, mean, std, num_feat=1):
    """[B,T,N,F] -> [B,N], inverse-scaled numerical history only."""
    x = np.asarray(x)
    require(x.ndim == 4 and min(x.shape) > 0 and 1 <= num_feat <= x.shape[-1],
        'Expected nonempty [batch,time,node,feature] history')
    require(np.isfinite(x).all() and np.isfinite([mean, std]).all() and std > 0,
        'Finite history and a positive saved normalization scale required')
    return np.abs(x[..., :num_feat].astype(np.float64) * std + mean).mean(axis=(1, 3))


def _moments(value):
    value = np.asarray(value, dtype=np.float64).reshape(-1)
    require(value.size > 0 and np.isfinite(value).all(), 'Finite nonempty activation required')
    return dict(count=int(value.size), total=float(value.sum()),
        square_total=float(np.dot(value, value)), negative_count=int(np.count_nonzero(value < 0)))


class ActivationStatistics:
    """Streaming scalar moments; full component activation arrays are discarded."""
    def __init__(self):
        self.sums = {}
        self.shapes = {}

    def add(self, role, component, value, labels, n_groups):
        value, labels = np.asarray(value), np.asarray(labels)
        require(value.ndim == 4 and min(value.shape) > 0, 'Component must be [batch,time,node,channel]')
        require(labels.shape == (value.shape[0], value.shape[2]) and labels.dtype.kind in 'iu',
            'History-amplitude labels must have the identical batch/node axes')
        require(n_groups > 0 and labels.min() >= 0 and labels.max() < n_groups, 'Invalid history group labels')
        shape_key = role + '/' + component
        shape = list(value.shape[1:])
        require(self.shapes.get(shape_key, shape) == shape, 'Component axes changed between fixed batches')
        self.shapes[shape_key] = shape

        def add_group(axis, group, selected):
            if selected.size == 0:
                return
            key = (role, component, axis, int(group))
            row = self.sums.setdefault(key, dict(count=0, total=0., square_total=0., negative_count=0))
            for name, amount in _moments(selected).items():
                row[name] += amount

        add_group('all', -1, value)
        for index in range(value.shape[1]):
            add_group('component_time_index', index, value[:, index])
        # Transpose only; advanced indexing copies one group at a time.
        node_first = value.transpose(0, 2, 1, 3)
        for index in range(n_groups):
            add_group('historical_amplitude', index, node_first[labels == index])

    def rows(self):
        rows = []
        for (role, component, axis, group), sums in sorted(self.sums.items()):
            count = sums['count']
            rows.append(dict(role=role, component=component, axis=axis, group=group,
                count=count, mean=sums['total'] / count,
                rms=float(np.sqrt(max(0., sums['square_total'] / count))),
                negative_fraction=sums['negative_count'] / count))
        return rows


def statistics_changes(rows):
    paired = {}
    for row in rows:
        key = (row['component'], row['axis'], row['group'])
        require(row['role'] in ('best', 'last'), 'Unknown checkpoint role in statistics')
        require(row['role'] not in paired.setdefault(key, {}), 'Duplicate component statistics')
        paired[key][row['role']] = row
    result = []
    for (component, axis, group), roles in sorted(paired.items()):
        require(set(roles) == {'best', 'last'}, 'A component is missing one endpoint')
        a, b = roles['best'], roles['last']
        require(a['count'] == b['count'], 'Endpoint activation exposure differs')
        result.append(dict(component=component, axis=axis, group=group, count=a['count'],
            **{role + '_' + name: roles[role][name] for role in ('best', 'last')
               for name in ('mean', 'rms', 'negative_fraction')},
            **{name + '_change': b[name] - a[name] for name in ('mean', 'rms', 'negative_fraction')}))
    return result


def select_gap_numpy(channels, gap):
    channels = np.asarray(channels)
    require(channels.ndim == 4 and gap > 0 and channels.shape[-1] == gap,
        'Expected [batch,horizon,node,gap] channel outputs')
    index = np.arange(channels.shape[1]) % gap
    return np.take_along_axis(channels, index[None, :, None, None], axis=-1)


def head_predictions_numpy(latent, head, gap):
    """Tiny independent reference for server numerical tests, not real replay."""
    latent = np.asarray(latent)
    first = np.maximum(latent, 0) @ np.asarray(head['w1']).T + np.asarray(head['b1'])
    channels = np.maximum(first, 0) @ np.asarray(head['w2']).T + np.asarray(head['b2'])
    return select_gap_numpy(channels, gap)


def _tensor_record(value, labels, n_groups, role, component, statistics):
    statistics.add(role, component, value.detach().cpu().numpy(), labels, n_groups)


class LayerCapture:
    """Temporary observers. No forward replacement, parameters or buffers edited."""
    def __init__(self, model, labels, n_groups, role, statistics, torch):
        self.model, self.labels, self.n_groups = model, labels, n_groups
        self.role, self.statistics, self.torch = role, statistics, torch
        self.handles, self.calls, self.checks = [], {}, {}
        self.direct_sum = self.inherent_sum = self.tiid = self.fc1 = None
        self.latent = self.head_prediction = None
        self.bn_calls = 0

    def record(self, name, value):
        self.calls[name] = self.calls.get(name, 0) + 1
        require(self.calls[name] == 1, 'Unexpected repeated component call: ' + name)
        _tensor_record(value, self.labels, self.n_groups, self.role, name, self.statistics)

    def check(self, name, actual, expected):
        result = compare_arrays(actual.detach().cpu().numpy(), expected.detach().cpu().numpy())
        self.checks[name] = result
        require(result['passed'], 'Hook reconstruction mismatch: ' + name)

    def __enter__(self):
        def direct(index):
            def hook(module, inputs, output):
                self.record('direct_dif_' + str(index), output)
                value = output.detach()
                self.direct_sum = value if self.direct_sum is None else self.direct_sum + value
            return hook

        def inherent(index):
            def hook(module, inputs, output):
                value = output[1].detach()
                self.record('inherent_' + str(index), value)
                self.inherent_sum = value if self.inherent_sum is None else self.inherent_sum + value
            return hook

        def backcast(index):
            return lambda module, inputs, output: self.record('dif_backcast_' + str(index), output)

        def tiid(module, inputs, output):
            require(self.inherent_sum is not None, 'TIID executed before INH capture')
            self.check('tiid_input_equals_sum_inherent', inputs[0], self.inherent_sum)
            self.record('inherent_sum_before_tiid', inputs[0])
            self.record('inherent_sum_after_tiid', output)
            self.record('tiid_increment', output - inputs[0])
            self.tiid = output.detach()

        def fc1_input(module, inputs):
            require(self.direct_sum is not None and self.tiid is not None,
                'Missing direct or TIID component before output head')
            fused = self.direct_sum + self.tiid.repeat_interleave(self.model.gap, dim=1)
            self.check('fc1_input_equals_relu_reconstructed_fusion', inputs[0], fused.relu())
            self.record('direct_dif_sum', self.direct_sum)
            self.record('fused_before_relu', fused)
            self.record('fc1_input_after_relu', inputs[0])
            self.latent = fused.detach().cpu()
            self.direct_sum = self.inherent_sum = self.tiid = None

        def fc1_output(module, inputs, output):
            self.record('fc1_output_before_relu', output)
            self.record('fc1_output_after_relu', output.relu())
            self.fc1 = output.detach()

        def fc2_input(module, inputs):
            require(self.fc1 is not None, 'Missing first head output')
            self.check('fc2_input_equals_relu_fc1_output', inputs[0], self.fc1.relu())
            self.fc1 = None

        def fc2_output(module, inputs, output):
            self.record('fc2_all_gap_channels', output)
            self.head_prediction = self.model.select_gap_channels(output, self.model.gap).detach().cpu()

        def bn(module, inputs, output):
            self.bn_calls += 1

        try:
            original = self.model.original
            require(len(original.layers) == len(self.model.decoders) == 5 and self.model.gap == 3,
                'Expected the certified five-layer/direct12/gap3 GW6 architecture')
            for index, (layer, decoder) in enumerate(zip(original.layers, self.model.decoders)):
                self.handles.append(decoder.register_forward_hook(direct(index)))
                self.handles.append(layer.inh_layer.register_forward_hook(inherent(index)))
                self.handles.append(layer.dif_layer.backcast_branch.register_forward_hook(backcast(index)))
            self.handles.append(original.tiid_module.register_forward_hook(tiid))
            self.handles.append(original.out_fc_1.register_forward_pre_hook(fc1_input))
            self.handles.append(original.out_fc_1.register_forward_hook(fc1_output))
            self.handles.append(original.out_fc_2.register_forward_pre_hook(fc2_input))
            self.handles.append(original.out_fc_2.register_forward_hook(fc2_output))
            self.handles.append(original.dynamic_graph_constructor.distance_function.bn.register_forward_hook(bn))
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, exc_type, exc, traceback):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.model.clear_cache()
        self.direct_sum = self.inherent_sum = self.tiid = self.fc1 = None


def _head_on_latent(model, latent, device, torch):
    # Functional calls avoid invoking component hooks during crossed-head checks.
    original = model.original
    value = latent.to(device)
    first = torch.nn.functional.linear(value.relu(), original.out_fc_1.weight, original.out_fc_1.bias)
    channels = torch.nn.functional.linear(first.relu(), original.out_fc_2.weight, original.out_fc_2.bias)
    return model.select_gap_channels(channels, model.gap).detach().cpu()


def _bn_buffers(model):
    bn = model.original.dynamic_graph_constructor.distance_function.bn
    require(bn.training is False and bn.track_running_stats is True, 'DistanceFunction BN must use saved eval buffers')
    return {name: value.detach().cpu().clone() for name, value in bn.named_buffers()}


def _archive_slice(path, identity, rows, expected_truth, station_ids):
    with np.load(path, allow_pickle=False) as payload:
        axes = drift.load_axes(payload, identity)
        ids = np.asarray(identity['data']['splits']['val']['selected_indices'])
        require(np.array_equal(axes['sample_ids'], ids), 'Archive validation sample axis differs')
        require(np.array_equal(axes['node_ids'], station_ids), 'Archive station axis differs')
        require(np.array_equal(axes['y'][:rows], expected_truth), 'Actual fixed-batch truth differs from certified archive')
        prediction = payload['pred__' + ARM]
        require(prediction.shape == axes['y'].shape and prediction.dtype.kind == 'f'
            and np.isfinite(prediction).all(), 'Invalid JOINT archive prediction')
        return prediction[:rows].copy(), axes['valid'][:rows].copy(), axes['direct_support'][:rows].copy()


def _checkpoint_authentication(run, identity, saved, best_meta, last_meta, train, evaluate):
    records, selected = {}, {}
    train_batches = len(identity['data']['splits']['train']['partition'])
    for arm in identity['arms']:
        directory = run / arm
        path = directory / 'last.pt'
        digest = drift.sha256(path)
        state = train.base.load_checkpoint(path)
        train.validate_saved(state, identity, arm, directory, train_batches)
        require(state['completed_epoch'] == 20 and drift.sha256(path) == digest,
            'Incomplete or changing current GW6 checkpoint: ' + arm)
        require(train.base.tensor_tree_hash(state['model_state']) == saved['arms'][arm]['last_model_tensor_sha256'],
            'Current checkpoint tensors differ from training summary: ' + arm)
        records[str(path)] = digest
        if arm == ARM:
            for role, metadata in (('best', best_meta), ('last', last_meta)):
                weights, certificate = evaluate.selection(directory, state, role, arm)
                expected = metadata['checkpoint_roles'][arm]
                for key in ('selection', 'epoch', 'policy', 'checkpoint_sha256', 'model_tensor_sha256', 'historical'):
                    require(certificate[key] == expected[key], 'Selected JOINT role differs: ' + role + '/' + key)
                selected[role] = weights
                selected_path = directory / (state['best_checkpoint_file'] if role == 'best' else 'last.pt')
                records[str(selected_path)] = certificate['checkpoint_sha256']
        del state
        gc.collect()
    require(best_meta['checkpoint_roles'][ARM]['epoch'] == 18
        and last_meta['checkpoint_roles'][ARM]['epoch'] == 20,
        'This bounded diagnosis requires the observed JOINT BEST18/LAST20 endpoints')
    return selected, records


def _report(summary, changes, crossed):
    lines = ['# GW6 JOINT 第18/20轮分支诊断', '',
        '固定验证输入上的无训练诊断。差值方向为 LAST20 − BEST18；所有激活均为模型内部数值域。',
        f'固定窗口：{summary["validation_rows"]}；原生batch：{summary["batch_size"]}；重建及归档回放全部通过。', '',
        '| 组件 | BEST均值 | LAST均值 | 均值变化 | BEST RMS | LAST RMS | 负值比例变化 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for row in changes:
        if row['axis'] == 'all':
            lines.append('| ' + row['component'] + ' | ' + ' | '.join(f'{row[key]:.6f}' for key in (
                'best_mean', 'last_mean', 'mean_change', 'best_rms', 'last_rms', 'negative_fraction_change')) + ' |')
    lines += ['', '2×2输出头敏感性（原目标数值域，只含固定小批次）：', '',
        '| latent来源 | head来源 | MAE | signed bias | 相对原BEST的预测均值变化 |',
        '|---|---|---:|---:|---:|']
    for row in crossed:
        if row['history_group'] == -1:
            lines.append(f'| {row["latent_role"]} | {row["head_role"]} | {row["mae"]:.6f} | {row["bias"]:.6f} | {row["change_from_native_best"]:.6f} |')
    lines += ['', '这些混合检查点未训练，可能破坏共同适应；该表不能用于直接分摊因果贡献或宣布某个分支为根因。', '',
        '历史幅值=起点历史主数值特征按保存的训练均值/标准差逆变换后，跨历史窗与数值特征的平均绝对值。分组边界仅由所选固定输入产生：' +
        json.dumps(summary['historical_amplitude_bounds'], ensure_ascii=False) + '。', '',
        'DistanceFunction BN：实际调用均已记录，保存的running buffers在前向前后逐元素一致；本次未重新估计统计量。', '',
        '完整激活统计和LAST−BEST统计量变化见components.csv、component_changes.csv；分组/时距明细在同表axis列；头部交叉见head_cross.csv。', '',
        '限制：']
    lines.extend('- ' + item for item in LIMITATIONS)
    return '\n'.join(lines) + '\n'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--threads', type=int, default=3)
    parser.add_argument('--batches', type=int, choices=(1, 2), default=2)
    parser.add_argument('--batch-size', type=int, default=48)
    args = parser.parse_args(argv)
    run, output = args.run_dir.resolve(), args.output_dir.resolve()
    require(output.parent == run and not output.exists(), 'Use a fresh direct child of the GW6 run')
    script_hash = drift.sha256(Path(__file__))
    identity, saved, run_records = drift.verify_run(run, 20)
    best_dir, best_meta, best_record = drift.select_eval(run, None, 'best', identity, saved, 20)
    last_dir, last_meta, last_record = drift.select_eval(run, None, 'last', identity, saved, 20)
    require(args.batch_size == identity['training_config']['bs'] == 48,
        'Batch size must equal the certified native batch48; no loader reconfiguration')
    require(args.threads == identity['runtime']['threads'], 'Thread count must equal certified GW6 runtime')

    import torch
    from experiments.graph_memory_gw6 import train, evaluate
    from experiments.graph_memory_gw6.reference import recertify
    from experiments.chronological.graph_message_reuse import RUNTIME_KEYS

    execution = {name: drift.sha256(REPO / name) for name in train.EXECUTION_FILES}
    require(execution == identity['execution_source'], 'Certified GW6 execution source changed')
    producer_source = evaluate.producer_source()
    helper_source = {'experiments/graph_memory_gw6/diagnose_drift.py': drift.sha256(Path(drift.__file__))}
    diagnostic_sources = dict(execution)
    diagnostic_sources.update(producer_source)
    diagnostic_sources.update(helper_source)
    diagnostic_sources['experiments/graph_memory_gw6/diagnose_layers.py'] = script_hash
    require(best_meta.get('producer_source') == producer_source
        and last_meta.get('producer_source') == producer_source, 'Certified evaluation producer source changed')
    historical_identity, source_weights, _ = recertify(identity['source_reference'])
    require(identity['data'] == historical_identity['data']
        and identity['source_fingerprint'] == historical_identity['source_fingerprint'],
        'Historical data or original source certificate differs')
    modules = evaluate.metrics.get_original_modules()
    require(modules.source_fingerprint == identity['source_fingerprint'], 'Original source fingerprint changed')
    runtime = evaluate.metrics.configure_runtime(args.device, args.threads, identity['training_config']['seed'])
    for key in RUNTIME_KEYS:
        require(runtime.get(key) == identity['runtime'].get(key), 'Certified runtime differs: ' + key)
    data_dir = args.data_dir or evaluate.metrics.default_data_dir()
    data = evaluate.metrics.load_data(data_dir, identity['training_config'], modules,
        logging.getLogger('gw6-layer-diagnostic'), check=False, training=True)
    require(data.identity == identity['data'], 'Actual native data/partition/normalization identity differs')
    weights, checkpoint_records = _checkpoint_authentication(run, identity, saved, best_meta, last_meta, train, evaluate)
    fixed = []
    iterator = data.loaders['val_loader'].get_iterator()
    for _ in range(args.batches):
        fixed.append(next(iterator))
    sizes = [len(raw['x_data']) for raw in fixed]
    require(all(size == args.batch_size for size in sizes), 'Expected complete original batch48 inputs')
    fixed_hash = train.base.tensor_tree_hash(fixed)
    normalization = identity['data']['normalization']
    amplitudes = [historical_amplitude(raw['x_data'], normalization['mean'], normalization['std'],
        identity['training_config']['num_feat']) for raw in fixed]
    labels, bounds = drift.target_bins(np.concatenate(amplitudes).reshape(-1))
    group_labels, start = [], 0
    for amplitude in amplitudes:
        stop = start + amplitude.size
        group_labels.append(labels[start:stop].reshape(amplitude.shape))
        start = stop
    n_groups = len(bounds) - 1
    truth = np.concatenate([data.scaler.inverse_transform(torch.as_tensor(raw['y_data'])).squeeze(-1).numpy()
        for raw in fixed])
    rows = len(truth)
    archive, masks, supports = {}, {}, {}
    for role, directory in (('best', best_dir), ('last', last_dir)):
        archive[role], masks[role], supports[role] = _archive_slice(directory / 'paired_predictions.npz',
            identity, rows, truth, data.station_ids)
    require(np.array_equal(masks['best'], masks['last']) and np.array_equal(supports['best'], supports['last']),
        'BEST/LAST archive masks or supports differ')

    models, bn_before, model_before = {}, {}, {}
    for role in ('best', 'last'):
        model = train.model_for_arm(data, modules, args.device, ARM, source_weights,
            stage='joint', **identity['model_options'])
        model.load_state_dict(weights[role], strict=True)
        model.eval()
        models[role] = model
        bn_before[role] = _bn_buffers(model)
        model_before[role] = train.base.tensor_tree_hash(model.state_dict())
    del weights, source_weights
    statistics, reconstruction, replay, crossed_values, bn_calls = ActivationStatistics(), [], [], {}, {'best': 0, 'last': 0}
    start = 0
    with torch.no_grad():
        for batch_index, (raw, grouping) in enumerate(zip(fixed, group_labels), 1):
            x, y_unused, incident, sensors = evaluate.metrics.device_batch(raw, args.device)
            del y_unused
            stop = start + len(x)
            actual_support = incident['distances'].abs().sum(-1).cpu().numpy() > 0
            require(np.array_equal(actual_support, supports['best'][start:stop]), 'Fixed-batch incident support differs')
            latents, native = {}, {}
            for role, model in models.items():
                with LayerCapture(model, grouping, n_groups, role, statistics, torch) as capture:
                    prediction = model(x, incident_data=incident, sensor_data=sensors)
                    model.assert_cache_clear()
                    require(capture.latent is not None and capture.bn_calls > 0, 'Required head/BN hooks were not executed')
                    capture.record('final_prediction', prediction)
                    capture.check('head_gap_select_equals_actual_prediction', prediction.cpu(), capture.head_prediction)
                    latents[role] = capture.latent
                    native[role] = data.scaler.inverse_transform(prediction).squeeze(-1).cpu().numpy()
                    check = compare_arrays(native[role], archive[role][start:stop], atol=2e-4, rtol=2e-6)
                    require(check['passed'], 'Fixed-batch prediction differs from certified ' + role.upper() + ' archive')
                    replay.append(dict(role=role, batch=batch_index, **check))
                    reconstruction.append(dict(role=role, batch=batch_index, checks=capture.checks,
                        component_calls=capture.calls, distance_function_bn_calls=capture.bn_calls))
                    bn_calls[role] += capture.bn_calls
                del prediction, capture
            for latent_role in ('best', 'last'):
                for head_role, model in models.items():
                    prediction = _head_on_latent(model, latents[latent_role], args.device, torch)
                    physical = data.scaler.inverse_transform(prediction).squeeze(-1).numpy()
                    if latent_role == head_role:
                        check = compare_arrays(physical, native[head_role], atol=2e-4, rtol=2e-6)
                        require(check['passed'], 'Own-head recomputation differs from original forward')
                        reconstruction.append(dict(role=head_role, batch=batch_index,
                            checks={'functional_head_equals_actual_prediction': check}))
                    crossed_values.setdefault((latent_role, head_role), []).append(physical)
                    del prediction, physical
            del latents, native, x, incident, sensors
            start = stop
            print(json.dumps(dict(status='GW6_LAYER_DIAGNOSTIC_BATCH_COMPLETE', batch=batch_index,
                total=args.batches)), flush=True)

    bn_records = {}
    for role, model in models.items():
        after = _bn_buffers(model)
        require(set(after) == set(bn_before[role]) and all(torch.equal(after[name], bn_before[role][name])
            for name in after), 'DistanceFunction BN buffers changed during eval replay')
        bn_records[role] = dict(module='anchor.base.original.dynamic_graph_constructor.distance_function.bn',
            actual_calls=bn_calls[role], buffers_unchanged=True,
            before_sha256=train.base.tensor_tree_hash(bn_before[role]),
            after_sha256=train.base.tensor_tree_hash(after),
            saved_buffers={name: value.numpy().tolist() for name, value in bn_before[role].items()})
        require(train.base.tensor_tree_hash(model.state_dict()) == model_before[role],
            'A model parameter or buffer changed during no-training replay')
        require(model.training is False and not any(module.training for module in model.modules()),
            'A model module left eval mode')
    del models
    gc.collect()
    all_groups = np.concatenate(group_labels)
    mask = masks['best']
    crossed = []
    for (latent_role, head_role), predictions in crossed_values.items():
        prediction = np.concatenate(predictions)
        for index in [-1, *range(n_groups)]:
            selected = mask if index == -1 else mask & (all_groups[:, None, :] == index)
            if not selected.any():
                continue
            error = prediction[selected].astype(np.float64) - truth[selected]
            delta = prediction[selected].astype(np.float64) - archive['best'][selected]
            crossed.append(dict(latent_role=latent_role, head_role=head_role, history_group=index,
                count=int(selected.sum()), mae=float(np.abs(error).mean()),
                bias=float(error.mean()), rms_error=float(np.sqrt(np.mean(error * error))),
                change_from_native_best=float(delta.mean()), change_rms=float(np.sqrt(np.mean(delta * delta)))))

    require(train.base.tensor_tree_hash(fixed) == fixed_hash, 'Fixed validation batch input changed')
    for record in run_records.values():
        require(drift.sha256(record['path']) == record['sha256'], 'Authenticated run document changed')
    for path, digest in checkpoint_records.items():
        require(drift.sha256(path) == digest, 'An authenticated checkpoint file changed')
    for record in (best_record, last_record):
        require(drift.sha256(record['archive_path']) == record['archive_sha256'], 'Prediction archive changed')
        for name in ('metadata', 'analysis'):
            require(drift.sha256(Path(record['directory']) / (name + '.json')) == record[name + '_sha256'],
                'Producer document changed')
    require(execution == {name: drift.sha256(REPO / name) for name in train.EXECUTION_FILES},
        'GW6 execution source changed during replay')
    require(producer_source == evaluate.producer_source()
        and helper_source == {'experiments/graph_memory_gw6/diagnose_drift.py': drift.sha256(Path(drift.__file__))},
        'Producer or diagnostic helper source changed during replay')
    require(drift.sha256(Path(__file__)) == script_hash, 'Diagnostic source changed during replay')
    recertify(identity['source_reference'])
    components = statistics.rows()
    changes = statistics_changes(components)
    result = dict(schema=SCHEMA, status='GW6_LAYER_DIAGNOSTIC_COMPLETE',
        source_run=str(run), source_run_identity_sha256=identity['identity_sha256'],
        arm=ARM, checkpoint_roles={role: metadata['checkpoint_roles'][ARM]
            for role, metadata in (('best', best_meta), ('last', last_meta))},
        source_budget_epochs=20, validation_rows=rows, batch_size=args.batch_size,
        producer_source=diagnostic_sources,
        component_change_definition='LAST_endpoint_statistic_minus_BEST_endpoint_statistic_not_elementwise_change_RMS',
        batches=args.batches, sample_ids=identity['data']['splits']['val']['selected_indices'][:rows],
        historical_amplitude_bounds=bounds.tolist(), history_amplitude_uses_future_truth=False,
        fixed_input_sha256=fixed_hash, fixed_input_unchanged=True, runtime=runtime,
        reconstruction_pass=True, archive_prediction_replay_pass=True,
        reconstruction=reconstruction, archive_prediction_replay=replay,
        distance_function_bn=bn_records, component_shapes_excluding_batch=statistics.shapes,
        model_tensor_sha256_before=model_before, model_tensors_unchanged=True,
        optimizer_updates=0, checkpoint_writes=0, bn_recalibration=False,
        test_evaluated=False, model_forward_calls=2 * args.batches,
        functional_head_calls=4 * args.batches, limitations=LIMITATIONS)
    manifest = dict(schema='gw6_layer_diagnostic_input_v1',
        created_utc=datetime.now(timezone.utc).isoformat(),
        source_run=str(run), source_run_identity_sha256=identity['identity_sha256'],
        run_documents=run_records, checkpoint_files=checkpoint_records,
        best=best_record, last=last_record, execution_source=execution,
        producer_source=diagnostic_sources, diagnostic_helper_source=helper_source,
        diagnostic_script_sha256=script_hash,
        actual_data_directory=str(data.directory), data_identity=data.identity,
        fixed_input_sha256=fixed_hash, fixed_input_unchanged=True,
        historical_evidence_recertified_before_after=True,
        archive_loaded_prediction_arms=[ARM], full_validation_replayed=False,
        test_evaluated=False, optimizer_updates=0)
    output.mkdir()
    drift.write_json(output / 'input_manifest.json', manifest)
    result['input_manifest_sha256'] = drift.sha256(output / 'input_manifest.json')
    report = _report(result, changes, crossed)
    drift.write_json(output / 'summary.json', result)
    drift.write_json(output / 'metadata.json', result)
    drift.write_csv(output / 'components.csv', components)
    drift.write_csv(output / 'component_changes.csv', changes)
    drift.write_csv(output / 'head_cross.csv', crossed)
    (output / 'report.md').write_text(report, encoding='utf-8')
    print(report)
    print(json.dumps(dict(status=result['status'], output_dir=str(output)), ensure_ascii=False), flush=True)
    return result


if __name__ == '__main__':
    main()
