"""Passive inclusive module timings of native released IGSTGNN inference.

This is separate from fixed-weight intervention latency: intervention hooks
discard outputs after modules execute, so their differences cannot measure
the time saved by deleting a module.  Timing hooks add profiling overhead.
"""
import csv
import io
import json
from pathlib import Path
import time

import torch


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _device_for(model, x, device):
    target = torch.device(device)
    _require(target.type in ('cpu', 'cuda'), 'Module timing supports CPU or CUDA only')
    _require(target.type == x.device.type, 'Timing device must match X device')
    if target.type == 'cpu':
        target = torch.device('cpu')
    else:
        # Resolve unindexed cuda against the actual input, avoiding an implicit
        # switch to device zero on a multi-GPU host.
        target = torch.device('cuda', x.device.index if target.index is None else target.index)
    _require(x.device == target, 'Timing device must match X device')
    _require(all(parameter.device == target for parameter in model.parameters()),
             'Model parameters must share the timing device')
    _require(all(buffer.device == target for buffer in model.buffers()),
             'Model buffers must share the timing device')
    return target


def profile_full_modules(model, x, incident, sensors, device, repetitions=2, *,
                         selected_paths=None):
    """Return ``(last_native_prediction, JSON-compatible_timing_report)``.

    Paths are exact ``named_modules`` keys; the root path is ``''``.  The
    default covers every named module, including modules never called.  Times
    are inclusive, so parent and child totals overlap and cannot be summed.
    CUDA pre/post events use the target device's current stream; synchronize
    once before the first measured forward and once after each complete
    forward, never after individual modules.
    No warmup, data loading, training or backward pass is performed here.
    """
    _require(isinstance(x, torch.Tensor) and x.ndim == 4, 'Expected native X[B,T,N,F]')
    _require(isinstance(repetitions, int) and not isinstance(repetitions, bool) and repetitions > 0,
             'repetitions must be a positive integer')
    _require(not any(module.training for module in model.modules()),
             'Module timing requires every module in eval mode')
    _require(all(parameter.grad is None for parameter in model.parameters()),
             'Module timing requires absent parameter gradients')
    _require(not hasattr(model, '_decode_forecast') and not hasattr(model, 'report_spatial_state'),
             'Profile the isolated released IGSTGNN, not a later model variant')
    _require(incident is not None, 'Native FULL profiling requires a supplied original report')
    target = _device_for(model, x, device)
    named = dict(model.named_modules())
    if selected_paths is None:
        paths = tuple(named)
    else:
        _require(isinstance(selected_paths, (tuple, list)) and all(isinstance(path, str) for path in selected_paths),
                 'selected_paths must be a list/tuple of exact module paths')
        paths = tuple(selected_paths)
        _require(len(paths) == len(set(paths)), 'selected_paths contains duplicates')
        _require(all(path in named for path in paths), 'Unknown selected module path')
    use_cuda = target.type == 'cuda'
    records = [{path: [] for path in paths} for _ in range(repetitions)]
    starts = {path: [] for path in paths}
    handles = []
    current_repetition = 0
    forward_seconds, forward_wall_seconds = [], []
    stream = torch.cuda.current_stream(device=target) if use_cuda else None

    def before(path):
        def record_start(_module, _args):
            if use_cuda:
                stamp = torch.cuda.Event(enable_timing=True)
                stamp.record(stream)
            else:
                stamp = time.perf_counter_ns()
            starts[path].append(stamp)
        return record_start

    def after(path):
        def record_end(_module, _args, _output):
            if use_cuda:
                stamp = torch.cuda.Event(enable_timing=True)
                stamp.record(stream)
            else:
                stamp = time.perf_counter_ns()
            first = starts[path].pop()
            records[current_repetition][path].append((first, stamp))
        return record_end

    try:
        for path in paths:
            handles.append(named[path].register_forward_pre_hook(before(path)))
            handles.append(named[path].register_forward_hook(after(path)))
        with torch.inference_mode():
            if use_cuda:
                # Exclude outstanding warmup/earlier work from the first
                # measured host wall interval, without per-module syncs.
                stream.synchronize()
            for current_repetition in range(repetitions):
                wall_start = time.perf_counter_ns()
                if use_cuda:
                    first = torch.cuda.Event(enable_timing=True)
                    last = torch.cuda.Event(enable_timing=True)
                    first.record(stream)
                prediction = model(x, incident_data=incident, sensor_data=sensors)
                if use_cuda:
                    last.record(stream)
                    stream.synchronize()
                    elapsed = first.elapsed_time(last)/1000.
                else:
                    elapsed = (time.perf_counter_ns()-wall_start)/1e9
                wall_elapsed = (time.perf_counter_ns()-wall_start)/1e9
                forward_seconds.append(float(elapsed))
                forward_wall_seconds.append(float(wall_elapsed))
                _require(all(not stack for stack in starts.values()), 'Unmatched module timing hooks')
    finally:
        for handle in reversed(handles):
            handle.remove()

    modules = []
    for path in paths:
        counts, totals = [], []
        for repetition in records:
            calls = repetition[path]
            counts.append(len(calls))
            if use_cuda:
                totals.append(sum(first.elapsed_time(last)/1000. for first, last in calls))
            else:
                totals.append(sum((last-first)/1e9 for first, last in calls))
        total_calls = sum(counts)
        total_seconds = float(sum(totals))
        modules.append({
            'path': path, 'module_type': type(named[path]).__name__,
            'parent_path': path.rpartition('.')[0] if path else None,
            'depth': path.count('.')+1 if path else 0,
            'calls': total_calls, 'calls_per_repetition': counts,
            'inclusive_seconds_total': total_seconds,
            'inclusive_seconds_mean_per_call': total_seconds/total_calls if total_calls else None,
            'inclusive_seconds_mean_per_repetition': total_seconds/repetitions,
            'inclusive_seconds_per_repetition': [float(value) for value in totals],
        })
    report = {
        'schema': 'original_igstgnn_module_timing_v1',
        'case': 'FULL', 'device': str(target), 'repetitions': repetitions,
        'batch_size': int(x.shape[0]), 'input_shape': list(x.shape),
        'clock': 'cuda_events_target_device_current_stream' if use_cuda else 'perf_counter_ns',
        'cuda_stream': int(stream.cuda_stream) if use_cuda else None,
        'cuda_synchronizations': repetitions+1 if use_cuda else 0,
        'cuda_synchronizations_before_first_forward': int(use_cuda),
        'cuda_synchronizations_after_complete_forward': repetitions if use_cuda else 0,
        'synchronization_scope': 'once_before_first_and_after_each_complete_forward' if use_cuda else 'none',
        'scope': 'passive_native_FULL_forward_only_no_loader_no_training_no_backward',
        'profiling_hook_overhead_included': True,
        'parent_child_inclusive_times_overlap': True,
        'module_times_must_not_be_summed': True,
        'zero_calls_does_not_prove_parameters_unused': True,
        'unnamed_functional_operations_are_in_parent_times_only': True,
        'forward_seconds': forward_seconds,
        'forward_wall_seconds': forward_wall_seconds,
        'forward_seconds_mean': sum(forward_seconds)/repetitions,
        'forward_wall_seconds_mean': sum(forward_wall_seconds)/repetitions,
        'selected_paths': list(paths), 'modules': modules,
    }
    return prediction, report


def write_module_timing(output_dir, report):
    """Write fresh JSON/CSV and return a Markdown section for the caller.

    Existing ``report.md`` is deliberately left untouched.  The runner can
    append the returned section when assembling its complete report.  Refuse
    to replace either timing artifact, including a prior partial result.
    """
    _require(isinstance(report, dict) and report.get('schema') == 'original_igstgnn_module_timing_v1',
             'Expected an original IGSTGNN module timing report')
    _require(isinstance(report.get('modules'), list), 'Timing report must contain module rows')
    serialized = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n'
    columns = (
        'path', 'module_type', 'parent_path', 'depth', 'calls',
        'inclusive_seconds_total', 'inclusive_seconds_mean_per_call',
        'inclusive_seconds_mean_per_repetition', 'calls_per_repetition',
        'inclusive_seconds_per_repetition',
    )
    csv_text = io.StringIO(newline='')
    writer = csv.DictWriter(csv_text, fieldnames=columns)
    writer.writeheader()
    for module in report['modules']:
        row = {name: module[name] for name in columns}
        for name in ('calls_per_repetition', 'inclusive_seconds_per_repetition'):
            row[name] = json.dumps(row[name], allow_nan=False)
        writer.writerow(row)
    directory = Path(output_dir)
    json_path = directory/'module_timings.json'
    csv_path = directory/'module_timings.csv'
    if json_path.exists() or csv_path.exists():
        raise FileExistsError('Module timing artifacts already exist; refusing to replace them')
    directory.mkdir(parents=True, exist_ok=True)
    with json_path.open('x', encoding='utf-8', newline='') as handle:
        handle.write(serialized)
    with csv_path.open('x', encoding='utf-8', newline='') as handle:
        handle.write(csv_text.getvalue())

    key_types = {
        'IGSTGNN', 'DecoupleLayer', 'DifBlock', 'InhBlock',
        'IncidentContextSpatialFusion', 'TemporalIncidentImpactDecay',
        'DynamicGraphConstructor', 'EstimationGate', 'GRUCell',
        'MultiheadAttention', 'STLocalizedConv',
    }
    selected = [module for module in report['modules']
                if module['module_type'] in key_types
                or module['path'] in ('embedding', 'out_fc_1', 'out_fc_2')]
    lines = [
        '## 各模块耗时：原模型 FULL 独立被动测量', '',
        '测量保持原模型完整计算，独立于十二状态的输出干预。'
        '模块时间为 inclusive，父模块包含子模块，表内各行不能相加。', '',
        '设备：{}；重复：{}；批量：{}；时钟：{}。'.format(
            report['device'], report['repetitions'], report.get('batch_size', '未记录'), report['clock']), '',
        '完整前向均值：{:.3f} ms；主机墙钟均值：{:.3f} ms。'.format(
            report['forward_seconds_mean']*1000., report['forward_wall_seconds_mean']*1000.), '',
        '调用次数为全部重复累计；单次均值按实际调用数计算。'
        '结果含 profiling hooks 开销，不代表训练或反向速度。'
        '未调用模块完整保存在 CSV/JSON；零调用不证明其参数没有通过函数式运算参与计算。', '',
        '| 模块路径 | 实际类名 | 累计调用 | inclusive 累计 ms | 单次均值 ms |',
        '|---|---|---:|---:|---:|',
    ]
    for module in selected:
        path = module['path'] or '<root>'
        path = path.replace('|', '\\|').replace('\n', ' ')
        kind = module['module_type'].replace('|', '\\|').replace('\n', ' ')
        mean = module['inclusive_seconds_mean_per_call']
        lines.append('| `{}` | {} | {} | {:.3f} | {} |'.format(
            path, kind, module['calls'], module['inclusive_seconds_total']*1000.,
            '{:.3f}'.format(mean*1000.) if mean is not None else '不可用'))
    lines.extend(('', '明细：module_timings.json、module_timings.csv。', ''))
    return '\n'.join(lines)
