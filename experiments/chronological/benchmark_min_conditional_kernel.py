"""Synchronized synthetic CUDA probe; no traffic tensors, training or test split.

Compare the current implementation with an explicit local Git revision. The
default measures one receiver chunk of one kernel, not an epoch or full model.
"""
import argparse
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import types

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.models.igstgnn import STLocalizedConv
from src.models.incident_conditional_propagation import ConditionalMessagePropagation


def load_reference(ref):
    # Read only a repository-local Git object; never fetch or execute a URL.
    path = 'src/models/incident_conditional_propagation.py'
    source = subprocess.check_output(['git', 'show', ref+':'+path], cwd=REPO)
    module = types.ModuleType('conditional_kernel_benchmark_reference')
    exec(compile(source, ref+':'+path, 'exec'), module.__dict__)
    return module.ConditionalMessagePropagation, hashlib.sha256(source).hexdigest()


def measure(call, repetitions):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    started = time.perf_counter()
    start.record()
    for _ in range(repetitions):
        output = call()
    stop.record()
    torch.cuda.synchronize()
    return output, dict(wall_seconds_per_call=(time.perf_counter()-started)/repetitions,
        cuda_seconds_per_call=start.elapsed_time(stop)/1000/repetitions,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        incremental_peak_bytes=torch.cuda.max_memory_allocated()-baseline)


def prepared(module, x, dynamic, static):
    with torch.no_grad():
        metadata = module._metadata(x, None, None)
        windows = x.unfold(1, 3, 1).permute(0, 1, 2, 4, 3)
        values = torch.stack([layer(windows[..., i, :]) for i, layer in enumerate(module.source_value)], -2)
        queries = torch.tanh(module.query(windows[..., -1, :]))
        keys = torch.tanh(module.key(windows)+module.position(metadata)[None, :, None])
        times = torch.tanh(module.time_factor(metadata))
        weights = module._graph_weights(dynamic, static, len(x), x)
    return values, queries, keys, times, None, None, None, weights


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adjacency', type=Path, required=True)
    parser.add_argument('--compare-ref', default='d36a7c7')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--receiver-chunk-size', type=int, default=4)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--repetitions', type=int, default=2)
    parser.add_argument('--full-kernel', action='store_true')
    parser.add_argument('--collect-diagnostics', action='store_true', help='Match stage-zero diagnostics collection')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if not torch.cuda.is_available():
        raise ValueError('CUDA required for this resource-only probe')
    if min(args.batch_size, args.receiver_chunk_size, args.repetitions) < 1 or args.warmup < 0:
        raise ValueError('Invalid benchmark shape/repetitions')
    if args.output.exists():
        raise ValueError('Use a fresh benchmark output file')
    torch.set_num_threads(3)
    torch.manual_seed(2025)
    torch.cuda.manual_seed_all(2025)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    adjacency = torch.from_numpy(np.load(args.adjacency, allow_pickle=False)).float()
    nodes, hidden = len(adjacency), 32
    if adjacency.shape != (nodes, nodes):
        raise ValueError('Adjacency must be square')
    forward = adjacency/adjacency.sum(-1, keepdim=True).clamp_min(1)
    reverse = adjacency.T/adjacency.T.sum(-1, keepdim=True).clamp_min(1)
    old_conv = STLocalizedConv(hidden, pre_defined_graph=[forward, reverse], k_s=2, k_t=3,
        dropout=0., use_pre=True, dy_graph=True, sta_graph=True)
    old_class, reference_hash = load_reference(args.compare_ref)
    current = ConditionalMessagePropagation(old_conv, adjacency, receiver_chunk_size=args.receiver_chunk_size,
                                            checkpoint_messages=True)
    state = current.state_dict()
    reference = old_class(old_conv, adjacency, receiver_chunk_size=args.receiver_chunk_size,
                          checkpoint_messages=True)
    reference.load_state_dict(state, strict=True)
    # Both implementations see exactly the same report-free synthetic tensors.
    x = torch.randn(args.batch_size, 12, nodes, hidden, device='cuda')
    base = torch.rand(args.batch_size, nodes, nodes, device='cuda').softmax(-1)
    base_reverse = base.transpose(-1, -2).contiguous()
    localize = lambda g: g.unsqueeze(-2).expand(*g.shape[:-1], 3, nodes).reshape(*g.shape[:-1], 3*nodes)
    dynamic = [localize(g) for g in (base, base@base, base_reverse, base_reverse@base_reverse)]
    static = [torch.rand(nodes, nodes, device='cuda').softmax(-1)]
    result = dict(schema='min_conditional_kernel_resource_probe_v1', torch=str(torch.__version__),
        cuda=torch.version.cuda, device=torch.cuda.get_device_name(0), deterministic_algorithms=True,
        dtype='float32', batch_size=args.batch_size, nodes=nodes, history_slots=12, hidden_dim=hidden,
        receiver_chunk_size=args.receiver_chunk_size, warmup_calls=args.warmup,
        repetitions=args.repetitions, scope='one_full_kernel' if args.full_kernel else 'first_receiver_chunk_only',
        collect_diagnostics=args.collect_diagnostics,
        traffic_XY_read=False, optimizer_updates=0, test_evaluated=False,
        reference_ref=args.compare_ref, reference_kernel_sha256=reference_hash,
        current_kernel_sha256=hashlib.sha256((REPO/'src/models/incident_conditional_propagation.py').read_bytes()).hexdigest())
    print(json.dumps(dict(status='PROBE_START', **result)), flush=True)
    outputs = {}
    with torch.no_grad():
        for name, model in (('reference', reference), ('current', current)):
            model.cuda().eval()
            tensors = None if args.full_kernel else prepared(model, x, dynamic, static)
            stop = min(nodes, args.receiver_chunk_size)
            if args.full_kernel:
                call = lambda: model(x, dynamic, static, collect=args.collect_diagnostics)
            else:
                options = {'collect': args.collect_diagnostics} if 'collect' in inspect.signature(model._chunk).parameters else {}
                call = lambda: model._chunk(*tensors, 0, stop, 0, model._edge_offsets_cpu[stop], False, False, **options)[0]
            print(name+': warmup_start', flush=True)
            for _ in range(args.warmup):
                call()
            print(name+': timed_start', flush=True)
            output, timing = measure(call, args.repetitions)
            outputs[name] = output.detach().cpu().clone()
            result[name] = timing
            print(json.dumps(dict(implementation=name, **timing)), flush=True)
            del output, call, tensors
            model.cpu()
            gc.collect()
            torch.cuda.empty_cache()
    difference = (outputs['reference']-outputs['current']).abs()
    torch.testing.assert_close(outputs['reference'], outputs['current'], atol=1e-5, rtol=1e-5)
    result.update(status='MIN_CONDITIONAL_KERNEL_RESOURCE_PROBE_PASS',
        output_abs_max=float(difference.max()), output_abs_mean=float(difference.mean()),
        wall_speedup=result['reference']['wall_seconds_per_call']/result['current']['wall_seconds_per_call'],
        full_model_or_epoch_speedup_claim=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
