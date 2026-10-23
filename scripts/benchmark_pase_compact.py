"""Complete FP32 inference benchmark; isolated CUDA process per model/batch.

Only the first validation windows are used, never test targets or accuracy
selection. No optimizer, checkpoint writes, prediction cache or partial forward.
"""
import argparse
import csv
import gc
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                   allow_nan=False)+'\n', encoding='utf-8')


def timing_summary(samples_ms, batch_size):
    if not samples_ms or any(not math.isfinite(x) or x <= 0 for x in samples_ms):
        raise ValueError('Positive complete-forward timings required')
    ordered = sorted(samples_ms)
    index = (len(ordered)-1)*.95
    lo = int(index)
    p95 = ordered[lo]+(ordered[min(lo+1, len(ordered)-1)]-ordered[lo])*(index-lo)
    mean = statistics.mean(samples_ms)
    return dict(mean_ms=mean, median_ms=statistics.median(samples_ms), p95_ms=p95,
                windows_per_second=batch_size*1000/mean, measured_forwards=len(samples_ms))


def paired_check(rows):
    for size in (1, 48):
        pair = [r for r in rows if r['batch_size'] == size]
        if {r['model'] for r in pair} != {'IGSTGNN', 'PASE-Net'} or len(pair) != 2:
            raise ValueError('Expected one result per model and batch size')
        for field in ('history_sha256', 'station_ids_sha256', 'sample_ids',
                      'runtime', 'warmup', 'repeats', 'rounds', 'output_shape'):
            if pair[0][field] != pair[1][field]:
                raise ValueError('Paired benchmark mismatch: '+field)


def slice_batch(value, size):
    if isinstance(value, dict):
        return {key: slice_batch(item, size) for key, item in value.items()}
    selected = value[:size]
    # Original sensor fields are CPU torch.Tensor; histories/events are NumPy.
    return selected.clone().contiguous() if hasattr(selected, 'clone') else selected.copy()


def prepare(args, torch, logger):
    from experiments.chronological import original_igstgnn_runtime as original
    from experiments.chronological.original_igstgnn_source import get_original_modules, load_archive
    from experiments.original_baseline_gw16.evaluate import archive_reference
    from experiments.position_state_gw16.common import read_json, require, sha256, tensor_tree_hash
    if args.worker == 'IGSTGNN':
        archive = load_archive(checkpoint=args.original_checkpoint, train_log=args.original_log)
        role = archive_reference(archive)
        require(role['training_complete'], 'Original archived run is incomplete')
        modules = get_original_modules()
        config = dict(archive.config, bs=48)
        data = original.load_data(args.data_dir, config, modules, logger)
        raw = slice_batch(next(data.loaders['val_loader'].get_iterator()), args.batch_size)
        model = original.build_model(data, modules, args.device)
        model.load_state_dict(archive.state_dict, strict=True)
        x, target, events, sensors = original.device_batch(raw, args.device)
        del target
        # Include inverse scaling so both models return raw-flow predictions.
        scaler = data.scaler
        def forward():
            return scaler.inverse_transform(model(x, label=None, incident_data=events, sensor_data=sensors))
        source = dict(checkpoint_sha256=sha256(args.original_checkpoint),
                      log_sha256=sha256(args.original_log), checkpoint_role=role,
                      source_fingerprint=modules.source_fingerprint)
        expected_parameters = 455220
    else:
        from experiments.pase_compact_v3.evaluate import selected_best
        from experiments.pase_compact_v3.state import validate_identity, source_hashes
        from experiments.pase_compact_v3.model import build_model
        from experiments.state_exchange_multidata.data import load_data
        identity = read_json(args.pase_run/'identity.json')
        validate_identity(identity)
        require(identity['dataset'] == 'Contra_Costa' and not identity['engineering_only'],
                'Use completed Contra Costa compact run')
        require(identity['execution_source'] == source_hashes(), 'Numerical source changed')
        weights, role, _ = selected_best(args.pase_run, identity, 'FULL')
        require(role['completed_training_epochs'] == 100 and role['selection'] == 'best',
                'Completed validation BEST required')
        data = load_data(args.data_dir, identity['config'], logger)
        require(data.identity == identity['data'], 'Compact data identity changed')
        raw = slice_batch(next(data.loaders['val_loader'].get_iterator()), args.batch_size)
        model = build_model(data, identity['config'], 'FULL', args.device)
        model.load_state_dict(weights, strict=True)
        x = torch.as_tensor(raw['x_data'], dtype=torch.float32, device=args.device)
        events = {k: torch.as_tensor(v, device=args.device) for k, v in raw['event_data'].items()}
        def forward():
            return model(x, events)['prediction']
        source = dict(identity_sha256=identity['identity_sha256'], checkpoint_role=role,
                      execution_source=identity['execution_source'])
        expected_parameters = 73357
    model.eval()
    require(sum(p.numel() for p in model.parameters()) == expected_parameters,
            'Unexpected model parameter count')
    axes = dict(history_sha256=tensor_tree_hash(raw['x_data']),
                station_ids_sha256=tensor_tree_hash(__import__('numpy').asarray(data.station_ids, dtype='int64')),
                sample_ids=list(range(3909, 3909+args.batch_size)))
    return model, forward, source, axes


def worker(args):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import logging
    import torch
    from experiments.chronological.original_igstgnn_runtime import configure_runtime, tensor_tree_hash
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    if not args.device.startswith('cuda') or not torch.cuda.is_available():
        raise ValueError('CUDA GPU required for paper efficiency measurement')
    runtime = configure_runtime(args.device, args.threads, 2025)
    runtime.update(python=sys.version.split()[0], cudnn=torch.backends.cudnn.version(),
                   total_gpu_bytes=torch.cuda.get_device_properties(args.device).total_memory)
    model, forward, source, axes = prepare(args, torch, logging.getLogger('efficiency'))
    state_before = tensor_tree_hash(model.state_dict())
    gc.collect()
    timings = []
    peaks = []
    shape = None
    with torch.no_grad():
        for round_index in range(args.rounds):
            for _ in range(args.warmup):
                prediction = forward()
                shape = list(prediction.shape)
                if shape != [args.batch_size, 12, 496, 1] or not torch.isfinite(prediction).all():
                    raise ValueError('Invalid complete 12-step output')
                del prediction
            torch.cuda.synchronize(args.device)
            for _ in range(args.repeats):
                start = time.perf_counter()
                prediction = forward()
                torch.cuda.synchronize(args.device)
                timings.append((time.perf_counter()-start)*1000)
                del prediction
            # Separate memory pass after clearing cached allocator blocks.
            # No second model or targets are resident on GPU.
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize(args.device)
            baseline = torch.cuda.memory_allocated(args.device)
            torch.cuda.reset_peak_memory_stats(args.device)
            prediction = forward()
            torch.cuda.synchronize(args.device)
            peaks.append(dict(baseline_allocated_bytes=baseline,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(args.device),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(args.device)))
            del prediction
            print(f'{args.worker} batch={args.batch_size} round={round_index+1}/{args.rounds} complete', flush=True)
    if tensor_tree_hash(model.state_dict()) != state_before:
        raise ValueError('Forward modified model weights or buffers')
    result = dict(model=args.worker, batch_size=args.batch_size,
        parameters=sum(p.numel() for p in model.parameters()), runtime=runtime,
        warmup=args.warmup, repeats=args.repeats, rounds=args.rounds,
        output_shape=shape, source=source, **axes, **timing_summary(timings, args.batch_size),
        peak_allocated_mib=max(r['peak_allocated_bytes'] for r in peaks)/2**20,
        peak_reserved_mib=max(r['peak_reserved_bytes'] for r in peaks)/2**20,
        additional_forward_mib=max(r['peak_allocated_bytes']-r['baseline_allocated_bytes'] for r in peaks)/2**20,
        memory_rounds=peaks, raw_latency_ms=timings,
        protocol='eval_no_grad_FP32_no_compile_no_autocast_complete_12_step_raw_flow',
        timing='synchronized_wall_time_model_forward_including_inverse_scaling_excluding_IO_H2D',
        memory='PyTorch_allocator_peak_including_model_buffers_inputs_output_not_whole_GPU_usage',
        subset='first_validation_windows_same_order', test_evaluated=False,
        training_updates=0, source_script_sha256=__import__('hashlib').sha256(Path(__file__).read_bytes()).hexdigest())
    save_json(args.output_dir/(args.worker.replace('-', '_')+f'_b{args.batch_size}.json'), result)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--code-dir', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--pase-run', type=Path, required=True)
    p.add_argument('--original-checkpoint', type=Path, required=True)
    p.add_argument('--original-log', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--threads', type=int, default=3)
    p.add_argument('--warmup', type=int, default=20)
    p.add_argument('--repeats', type=int, default=50)
    p.add_argument('--rounds', type=int, default=3)
    p.add_argument('--worker', choices=['IGSTGNN', 'PASE-Net'], help=argparse.SUPPRESS)
    p.add_argument('--batch-size', type=int, choices=[1, 48], help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    if min(args.threads, args.warmup, args.repeats, args.rounds) < 1:
        p.error('Timing counts and threads must be positive')
    sys.path.insert(0, str(args.code_dir.resolve()))
    if args.worker:
        worker(args)
        return
    for source in (args.pase_run/'identity.json', args.original_checkpoint, args.original_log,
                   args.code_dir/'experiments/pase_compact_v3/model.py'):
        if not source.is_file():
            raise FileNotFoundError(source)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    fields = ['code_dir', 'data_dir', 'pase_run', 'original_checkpoint', 'original_log',
              'output_dir', 'device', 'threads', 'warmup', 'repeats', 'rounds']
    base = [sys.executable, str(Path(__file__).resolve())]
    for field in fields:
        base.extend(['--'+field.replace('_', '-'), str(getattr(args, field))])
    # Independent processes release all model, data and CUDA allocator state.
    rows = []
    for size in (1, 48):
        for model in ('IGSTGNN', 'PASE-Net'):
            subprocess.run(base+['--worker', model, '--batch-size', str(size)], check=True,
                           cwd=args.code_dir)
            rows.append(json.loads((args.output_dir/(model.replace('-', '_')+f'_b{size}.json')).read_text()))
    paired_check(rows)
    save_json(args.output_dir/'summary.json', dict(status='PASE_EFFICIENCY_COMPLETE',
        paired_inputs_verified=True, test_evaluated=False, results=rows))
    columns = ['model', 'batch_size', 'parameters', 'mean_ms', 'median_ms', 'p95_ms',
               'windows_per_second', 'peak_allocated_mib', 'additional_forward_mib']
    with (args.output_dir/'results.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# IGSTGNN 与当前 PASE-Net：完整前向效率', '',
        'Contra Costa；同一组验证窗口；FP32；完整12步预测；每个模型/批量独立进程。',
        '每轮预热20次、测量50次，共3轮（实际配置见JSON）。不含数据读取、权重加载和CPU到GPU传输。',
        '峰值显存为PyTorch实际分配峰值，包含模型、缓冲区、输入及前向输出，不是整卡nvidia-smi占用。', '',
        '| 模型 | 批量 | 参数量 | 平均延迟(ms/批) | 中位延迟(ms/批) | P95(ms/批) | 吞吐量(窗口/s) | 峰值显存(MiB) |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    lines[3] = f'每轮预热{args.warmup}次、测量{args.repeats}次，共{args.rounds}轮。不含数据读取、权重加载和CPU到GPU传输。'
    for r in rows:
        lines.append('| '+ ' | '.join([r['model'], str(r['batch_size']), str(r['parameters'])]+
            [f'{r[k]:.3f}' for k in ['mean_ms', 'median_ms', 'p95_ms', 'windows_per_second', 'peak_allocated_mib']])+' |')
    lines.extend(['', '吞吐量为批量/平均延迟；本入口不统计FLOPs。原始耗时、环境和权重来源见各模型JSON。',
                  '本结果仅覆盖上述固定输入，不代表全验证集延迟分布。请在独占GPU上运行。', ''])
    (args.output_dir/'report.md').write_text('\n'.join(lines), encoding='utf-8')
    print('PASE_EFFICIENCY_COMPLETE: '+str(args.output_dir/'report.md'), flush=True)


if __name__ == '__main__':
    main()
