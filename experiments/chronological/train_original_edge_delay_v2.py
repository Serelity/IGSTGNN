"""Matched warmstart CONTROL/FIXED/DYNAMIC with first-layer delay residuals.

One fixed recipe, original-equivalent initialization, best AND last probes,
read-only task/decay gradient audits, immutable old source and run artifacts.
"""
import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import numpy as np
import torch

from experiments.chronological import original_edge_delay_v2 as architecture
from experiments.chronological import train_original_edge_delay as legacy
from experiments.chronological.original_edge_delay_gradient_audit import audit_gradients, decay_only_adam
from experiments.chronological.original_edge_delay_diagnostics import (
    LagTelemetry, existing_run_read_lock, summarize_cases, write_csv)
from experiments.chronological.evaluate_original_edge_delay import (
    load_pilot, metric_replay, prediction_archive, snapshot)
from experiments.chronological.original_igstgnn_checkpoint_compat import counter_weights_only_compat
from experiments.chronological.original_igstgnn_source import get_original_modules
from experiments.chronological.original_igstgnn_runtime import (
    build_model, configure_runtime, default_data_dir, load_data, rng_fingerprint,
    run_lock, sha256, tensor_tree_hash, write_json, device_batch)
from experiments.chronological.evaluate_original_igstgnn_modules import source_from_run, verify_replay

SCHEMA = 'original_igstgnn_first_layer_boundary_delay_v2'
ARMS = legacy.ARMS
EXECUTION_FILES = tuple(dict.fromkeys(legacy.EXECUTION_FILES + (
    'experiments/chronological/original_edge_delay_v2.py',
    'experiments/chronological/original_edge_delay_gradient_audit.py',
    'experiments/chronological/train_original_edge_delay_v2.py',
    'experiments/chronological/original_edge_delay_diagnostics.py',
    'experiments/chronological/evaluate_original_edge_delay.py',
    'experiments/chronological/original_igstgnn_ablation_metrics.py',
    'experiments/chronological/run_original_edge_delay_v2.sh')))


def telemetry_for(model):
    telemetry = LagTelemetry(model)
    wrapper = model.layers[0].dif_layer
    if isinstance(wrapper, architecture.FirstLayerDelayDif):
        def capture(module, args, kwargs, output):
            if module.mode == 'NO_ALIGN':
                return
            history = kwargs['gated_history_data'] if 'gated_history_data' in kwargs else args[1]
            for graph, estimator in enumerate(module.delays):
                telemetry.pending.append((0, graph, module.original.localized_st_conv.k_t,
                    history.shape[1], estimator, estimator.last_delay))
        telemetry.handles.append(wrapper.register_forward_hook(capture, with_kwargs=True))
    return telemetry


@torch.no_grad()
def evaluate(model, data, modules, device, name, calls):
    model.eval()
    before, rng_before = tensor_tree_hash(model.state_dict()), rng_fingerprint()
    # Keep expensive support reductions outside complete-forward timings.
    telemetry = telemetry_for(model)
    predictions, targets, timings = [], [], []
    try:
        for number, raw in enumerate(data.loaders['val_loader'].get_iterator(), 1):
            x, y, incident, sensors = device_batch(raw, device)
            legacy.sync(device)
            cuda = torch.device(device).type == 'cuda'
            baseline = torch.cuda.memory_allocated(device) if cuda else None
            if cuda:
                torch.cuda.reset_peak_memory_stats(device)
            start = time.perf_counter()
            calls[name] = calls.get(name, 0) + 1
            prediction = model(x, incident_data=incident, sensor_data=sensors)
            legacy.sync(device)
            elapsed = time.perf_counter() - start
            peak = torch.cuda.max_memory_allocated(device) if cuda else None
            if prediction.shape != y.shape or not torch.isfinite(prediction).all():
                raise ValueError('Invalid prediction: ' + name)
            start = time.perf_counter()
            telemetry.consume(x.shape[1])
            legacy.sync(device)
            timings.append(dict(batch=number, batch_size=len(x), forward_seconds=elapsed,
                diagnostic_seconds=time.perf_counter() - start, peak_allocated_bytes=peak,
                incremental_peak_bytes=peak - baseline if cuda else None,
                module_diagnostics=architecture.diagnostics(model)))
            predictions.append(data.scaler.inverse_transform(prediction).squeeze(-1).cpu().numpy())
            targets.append(data.scaler.inverse_transform(y).squeeze(-1).cpu().numpy())
            print(json.dumps(dict(phase=name, batch=number,
                total_batches=data.loaders['val_loader'].num_batch, forward_seconds=elapsed)), flush=True)
    finally:
        telemetry.close()
    if before != tensor_tree_hash(model.state_dict()) or rng_before != rng_fingerprint():
        raise ValueError('Inference changed tensors/RNG: ' + name)
    prediction, target = np.concatenate(predictions), np.concatenate(targets)
    diagnostics = telemetry.finish()
    diagnostics['boundary_interpretation'] = (
        'raw missing delayed path mass before fallback; v2 default replaces it with native current values; '
        'ZERO_BOUNDARY drops it; signed residual scale is not a physical mixing probability')
    return dict(prediction=prediction, target=target, native=legacy.native_metrics(prediction, target, modules),
                timing=timings, telemetry=diagnostics, model_hash=before)


def audit_legacy(directory, data, modules, device, batches, output, check, source):
    with existing_run_read_lock(directory):
        before = snapshot(directory)
        identity, summary, states = load_pilot(directory, check)
        # Audits must use the same source, complete training partition and recipe.
        if identity['data'] != data.identity:
            raise ValueError('Legacy audit training/validation data identity differs')
        if identity['source']['checkpoint_sha256'] != source['checkpoint_sha256']:
            raise ValueError('Legacy audit original source checkpoint differs')
        for key, value in identity['training_config'].items():
            if key != 'device' and data.config.get(key) != value:
                raise ValueError('Legacy audit training recipe differs: ' + key)
        from experiments.chronological.original_edge_delay import install_delay
        for arm in ARMS:
            configure_runtime(device, identity['runtime']['threads'], data.config['seed'])
            model = build_model(data, modules, device)
            if arm != 'CONTROL':
                install_delay(model, arm, **identity['architecture'], seed=data.config['seed'])
            model.load_state_dict(states[arm]['model_state'], strict=True)
            result = audit_gradients(model, batches, data, modules, device=device,
                weight_decay=data.config['wdecay'], clip_norm=data.config['clip_grad_value'], phase='LEGACY_' + arm + '_LAST')
            write_json(output / ('gradient_LEGACY_' + arm + '_LAST.json'), result)
            del model
        if before != snapshot(directory):
            raise ValueError('Legacy run changed during gradient audit')
        return dict(pilot_run=str(directory), identity_sha256=identity['identity_sha256'],
            completed=summary['completed'], source_files_unchanged=True,
            backward_calls=len(batches) * len(ARMS), optimizer_updates=0)


def pairs_for(cases):
    pairs = [('SOURCE', name) for name in cases if name != 'SOURCE']
    for stage in ('BEST', 'LAST'):
        pairs += [('CONTROL_' + stage, arm + '_' + stage) for arm in ('FIXED', 'DYNAMIC')]
        pairs.append(('FIXED_' + stage, 'DYNAMIC_' + stage))
        for arm in ('FIXED', 'DYNAMIC'):
            for mode in (('NO_ALIGN', 'ZERO_BOUNDARY') if arm == 'FIXED' else ('NO_ALIGN', 'ZERO_BOUNDARY', 'SHUFFLE')):
                pairs.append((arm + '_' + stage + '_' + mode, arm + '_' + stage))
    pairs += [(arm + '_BEST', arm + '_LAST') for arm in ARMS]
    return list(dict.fromkeys(pairs))


def write_report(directory, analysis, summary, timings, telemetry):
    write_json(directory / 'analysis.json', analysis)
    write_json(directory / 'timings.json', timings)
    write_json(directory / 'mechanisms.json', telemetry)
    write_csv(directory / 'cases.csv', [dict(case=k, **{f: v for f, v in m.items() if f != 'horizons'})
        for k, m in analysis['metrics'].items()])
    rows = [dict(comparison=k, group=g, **{f: v for f, v in m.items() if f != 'horizons'})
        for k, c in analysis['comparisons'].items() for g, m in c['groups'].items()]
    for filename, prefix in (('comparisons', 'all_nodes'), ('flow', 'flow:'), ('roads', 'road:'), ('bias', 'bias:')):
        write_csv(directory / (filename + '.csv'), [r for r in rows if r['group'].startswith(prefix)])
    write_csv(directory / 'horizons.csv', [dict(case=k, **h) for k, m in analysis['metrics'].items() for h in m['horizons']])
    write_csv(directory / 'boundary_support.csv', [dict(case=k, **r) for k, v in telemetry.items() for r in v['boundary_support']])
    gradient_rows = [dict(phase=value['phase'], **row)
        for path in sorted(directory.glob('gradient_*.json'))
        for value in [json.loads(path.read_text(encoding='utf-8'))] for row in value['groups']]
    write_csv(directory / 'gradient_summary.csv', gradient_rows)
    lines = ['# 原 IGSTGNN：第一层边界回退时延模块 v2', '',
        f"工程检查：{summary['engineering_only']}；验证窗口：{summary['validation_rows']}；等完成预算：{summary['same_completed_budget']}。", '',
        '原发布版 IGSTGNN 为主 baseline；CONTROL 是相同成熟权重、相同额外预算的纯主干微调。',
        '三组初始化均精确回放源预测。只改第一层历史道路 bank；未来 gap=3 递推、其余四层和其他分支沿用原计算。',
        '缺失延迟路径回退到原同刻值；固定 -2 偏置是 buffer，L2 只约束零初值的边偏差。',
        '零初值有符号残差尺度是等价初始化保护；并不表示传播概率或物理容量。第一步尺度可获梯度，时延梯度随后开启。', '',
        '| 状态 | MAE pooled | MAE macro | RMSE pooled | MAPE (%) |', '|---|---:|---:|---:|---:|']
    for name, m in analysis['metrics'].items():
        lines.append(f"| {name} | {m['mae_pooled']:.9f} | {m['mae_macro']:.9f} | {m['rmse_pooled']:.9f} | {100*m['mape_pooled']:.9f} |")
    lines += ['', '| 候选 vs 对照 | MAE收益 pooled | MAE收益 macro | MAPE收益百分点 | 固定困难窗口收益 |', '|---|---:|---:|---:|---:|']
    fmt = lambda value: '不可用' if value is None else f'{value:.9f}'
    for name, value in analysis['comparisons'].items():
        group = value['groups']['all_nodes']
        lines.append(f"| {name} | {fmt(group['mae_gain_pooled'])} | {fmt(group['mae_gain_macro'])} | {fmt(100*value['mape_gain_fraction'])} | {fmt(value['fixed_hard_window_gain'])} |")
    lines += ['', '梯度审计（原引擎全局裁剪后的任务梯度，与 Adam 随后加上的 L2 项分开）：', '',
        '| 阶段 | 训练批 | 参数组 | task L2 | active decay L2 | decay/task |', '|---|---:|---|---:|---:|---:|']
    for row in gradient_rows:
        if row['group'] in ('alignment_scale', 'legacy_edge_logits', 'edge_deviation'):
            lines.append(f"| {row['phase']} | {row['batch']} | {row['group']} | {row['clipped_task_l2']:.9g} | {row['active_decay_l2']:.9g} | {fmt(row['decay_to_task_l2'])} |")
    lines += ['', '| 同机完整前向 | 新测调用 | 新测完整验证总秒 |', '|---|---:|---:|']
    for name in ('SOURCE', 'CONTROL_LAST', 'FIXED_LAST', 'DYNAMIC_LAST'):
        values = timings[name]['batches']
        lines.append(f"| {name} | {len(values)} | {sum(v['forward_seconds'] for v in values):.6f} |")
    lines += ['', 'BEST 与 LAST 分别加载、回放和探针；BEST 用于选模，LAST 检查训练是否退化。',
        'NO_ALIGN、ZERO_BOUNDARY、SHUFFLE 为同检查点敏感性，不能当作删除后重训效果。',
        '本版仅第一层读取时延；三种探针的第一层输入相同，NO_ALIGN 外的延迟张量只在 SHUFFLE 显式重排。',
        'boundary_support.csv 是回退前缺失路径系数；默认计算已回退，不能将该表标成实际信号损失。',
        '真值流量和偏差只用于事后分层；固定困难 ID 由源预测一次确定。',
        '梯度审计使用训练批，只有 backward，没有 optimizer step；审计后恢复模型、梯度、模式和 RNG。',
        '梯度范数比例不等于历史更新的因果占比。全网计时不含 H2D/诊断；训练墙钟含加载及原引擎统计。',
        '没有重新开始原模型的从头训练或精确恢复其最佳轮优化器。开发验证已用于设计，窗口可能重叠。',
        '不评价测试集，不宣称已超过 baseline 或识别真实传播时间。']
    (directory / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--audit-pilot-run', type=Path, help='Optional immutable v1 pilot: audit LAST gradients before v2 training')
    parser.add_argument('--data-dir', type=Path, default=default_data_dir())
    parser.add_argument('--epochs', type=int, default=1, help='Total extra epochs per arm, 1..5; --resume targets total budget')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--threads', type=int, default=3)
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--audit-batches', type=int, default=2, help='First fixed training batches; no shuffle/updates in audit')
    parser.add_argument('--check', action='store_true', help='4 train + 4 val rows, batch 2; engineering only')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    if not 1 <= args.epochs <= 5 or not 1 <= args.audit_batches <= 8 or args.threads < 1:
        parser.error('Invalid epoch/audit/thread budget')
    args.run_dir, args.source_run, args.data_dir = (p.resolve() for p in (args.run_dir, args.source_run, args.data_dir))
    if args.audit_pilot_run:
        args.audit_pilot_run = args.audit_pilot_run.resolve()
    for root in (args.source_run, args.data_dir, args.audit_pilot_run):
        if root and (args.run_dir == root or root in args.run_dir.parents or args.run_dir in root.parents):
            parser.error('Output must be outside source, data and legacy pilot')
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.run_dir.exists() != args.resume:
        raise ValueError('Use a fresh v2 directory, or --resume its existing identity')
    if args.resume and json.loads((args.run_dir / 'identity.json').read_text(encoding='utf-8')).get('schema') != SCHEMA:
        raise ValueError('Resume target is not a v2 run')
    calls = {}
    with counter_weights_only_compat(), existing_run_read_lock(args.source_run):
        source_before = snapshot(args.source_run)
        runtime = configure_runtime(args.device, args.threads, args.seed)
        modules = get_original_modules()
        source_state, source_config, expected, source = source_from_run(args.source_run, check=args.check)
        if source['source_fingerprint'] != modules.source_fingerprint:
            raise ValueError('Expected frozen original IGSTGNN reference')
        config = dict(source_config, seed=args.seed, device=args.device)
        data = load_data(args.data_dir, config, modules, logging.getLogger('edge-delay-v2'), check=args.check, training=True)
        identity = dict(schema=SCHEMA, source=source, training_config=config, data=data.identity,
            architecture=dict(max_delay=2, score_dim=8), execution_source={p: sha256(REPO / p) for p in EXECUTION_FILES},
            runtime={k: v for k, v in runtime.items() if k != 'device_name'}, engineering_only=args.check,
            arm_order=list(ARMS), audit_batches=args.audit_batches,
            audit_pilot_run=str(args.audit_pilot_run) if args.audit_pilot_run else None,
            source_optimizer_resumed=False, stage_zero_selected=True, early_stopping=False,
            optimizer_scheduler='native Adam/decay/clipping/milestones reset equally', test_evaluated=False)
        identity['identity_sha256'] = legacy.digest(identity)
        if args.resume and json.loads((args.run_dir / 'identity.json').read_text(encoding='utf-8')) != identity:
            raise ValueError('Resume identity differs; original v1 runs cannot be resumed by v2')
        args.run_dir.mkdir(parents=True, exist_ok=args.resume)
        with run_lock(args.run_dir):
            try:
                write_json(args.run_dir / 'identity.json', identity)
                write_json(args.run_dir / 'runtime.json', runtime)
                write_json(args.run_dir / 'summary.json', dict(schema=SCHEMA, status='V2_RUNNING', test_evaluated=False))
                train_samples = tuple(data.loaders['train_loader'].samples)
                batches = []
                for raw in data.loaders['train_loader'].get_iterator():
                    batches.append(raw)
                    if len(batches) == args.audit_batches:
                        break
                if len(batches) != args.audit_batches:
                    raise ValueError('Requested more audit batches than training partition')
                audit_info = audit_legacy(args.audit_pilot_run, data, modules, args.device, batches,
                    args.run_dir, args.check, source) if args.audit_pilot_run else None
                write_json(args.run_dir / 'decay_counterfactual.json', dict(
                    legacy=decay_only_adam(), zero_deviation=decay_only_adam(parameter=0.),
                    caveat='pure scalar counterfactual, not a reconstructed historical update'))
                cases, timings, telemetry, replay, completed, metadata, resources = {}, {}, {}, {}, {}, {}, {}
                target, cache = None, {}
                def run(model, name, mode=None, force=False):
                    nonlocal target
                    with architecture.delay_mode(model, mode) if mode else nullcontext():
                        key = (tensor_tree_hash(model.state_dict()), mode)
                        calls.setdefault(name, 0)
                        if key in cache and not force:
                            origin, result = cache[key]
                            print(json.dumps(dict(phase=name, reused_from=origin)), flush=True)
                        else:
                            origin, result = None, evaluate(model, data, modules, args.device, name, calls)
                            if not force:
                                cache[key] = (name, result)
                        if target is None:
                            target = result['target']
                        if not np.array_equal(target, result['target']):
                            raise ValueError('Unpaired targets: ' + name)
                        if not force:
                            cases[name] = result['prediction']
                            telemetry[name] = result['telemetry']
                            timings[name] = dict(reused_from=origin, batches=result['timing'] if origin is None else [])
                            metadata[name] = dict(model_hash=result['model_hash'], mode=mode, reused_from=origin)
                        return result
                model = build_model(data, modules, args.device)
                model.load_state_dict(source_state, strict=True)
                original = run(model, 'SOURCE')
                replay['SOURCE'] = verify_replay(original['native'], expected, engineering_only=args.check)
                hard_ids = legacy.compare_predictions({'SOURCE': cases['SOURCE']}, target,
                    data.identity['splits']['val']['selected_indices'], data.roads)['fixed_hard_ids']
                del model
                for arm in ARMS:
                    configure_runtime(args.device, args.threads, args.seed)
                    data.loaders['train_loader'].samples = list(train_samples)
                    model = build_model(data, modules, args.device)
                    model.load_state_dict(source_state, strict=True)
                    if arm != 'CONTROL':
                        architecture.install_delay(model, arm, seed=args.seed, **identity['architecture'])
                    write_json(args.run_dir / (arm + '_architecture.json'), architecture.description(model))
                    initial = evaluate(model, data, modules, args.device, arm + '_INITIAL', calls)
                    if not np.array_equal(initial['prediction'], cases['SOURCE']):
                        raise ValueError('Initialization is not exactly original-equivalent: ' + arm)
                    replay[arm + '_INITIAL'] = dict(archived_source_exact=True)
                    audit = audit_gradients(model, batches, data, modules, device=args.device,
                        weight_decay=config['wdecay'], clip_norm=config['clip_grad_value'], phase=arm + '_INITIAL')
                    write_json(args.run_dir / ('gradient_' + arm + '_INITIAL.json'), audit)
                    def count_training_validation(module, inputs):
                        if not module.training:
                            key = arm + '_TRAINING_VALIDATION'
                            calls[key] = calls.get(key, 0) + 1
                    handle = model.register_forward_pre_hook(count_training_validation)
                    cuda = torch.device(args.device).type == 'cuda'
                    if cuda:
                        torch.cuda.reset_peak_memory_stats(args.device)
                    try:
                        state = legacy.run_arm(arm, model, data, modules, identity, args.run_dir / arm, args.epochs, args.resume)
                    finally:
                        handle.remove()
                    resources[arm] = dict(scope='run_arm including training, validation, checkpoint reload; excludes initial gradient audit',
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(args.device) if cuda else None,
                        training_seconds_by_epoch={str(r['epoch']): r['train_seconds_including_loading_and_metrics'] for r in state['history']})
                    completed[arm] = dict(epochs=state['completed_epoch'], best_epoch=state['best_epoch'], updates=state['global_updates'])
                    for stage in ('BEST', 'LAST'):
                        weights = torch.load(args.run_dir / arm / state['best_checkpoint_file'], map_location='cpu', weights_only=True) if stage == 'BEST' else state['model_state']
                        model.load_state_dict(weights, strict=True)
                        name = arm + '_' + stage
                        result = run(model, name, None if arm == 'CONTROL' else arm)
                        epoch = state['best_epoch'] if stage == 'BEST' else state['completed_epoch']
                        replay[name] = metric_replay(result['native'], state['history'][epoch])
                        metadata[name].update(epoch=epoch, selection=stage)
                        if stage == 'BEST':
                            archive, truth = prediction_archive(args.run_dir / arm / state['best_prediction_file'],
                                data.identity['splits']['val']['selected_indices'], data.station_ids)
                            if not np.array_equal(archive, result['prediction']) or not np.array_equal(truth, target):
                                raise ValueError('BEST archived predictions differ: ' + arm)
                        else:
                            audit = audit_gradients(model, batches, data, modules, device=args.device,
                                weight_decay=config['wdecay'], clip_norm=config['clip_grad_value'], phase=name)
                            write_json(args.run_dir / ('gradient_' + name + '.json'), audit)
                        if arm != 'CONTROL':
                            for mode in (('NO_ALIGN', 'ZERO_BOUNDARY') if arm == 'FIXED' else ('NO_ALIGN', 'ZERO_BOUNDARY', 'SHUFFLE')):
                                run(model, name + '_' + mode, mode)
                            restored = run(model, name + '_RESTORED', arm, force=True)
                            if not np.array_equal(restored['prediction'], cases[name]):
                                raise ValueError('ON not exactly restored: ' + name)
                            replay[name]['on_exactly_restored'] = True
                    del model, state
                analysis = summarize_cases(cases, target, data.identity['splits']['val']['selected_indices'],
                    data.station_ids, data.roads, hard_ids, pairs_for(cases))
                if snapshot(args.source_run) != source_before:
                    raise ValueError('Source run changed')
                summary = dict(schema=SCHEMA, status='FIRST_LAYER_BOUNDARY_DELAY_V2_PASS', engineering_only=args.check,
                    same_completed_budget=len({v['epochs'] for v in completed.values()}) == 1, completed=completed,
                    validation_rows=len(target), valid_points=analysis['valid_points'], cases=list(cases),
                    new_inference_calls=calls, total_new_inference_calls=sum(calls.values()),
                    audit_backward_calls=len(batches) * 6 + (audit_info['backward_calls'] if audit_info else 0),
                    optimizer_updates={k: v['updates'] for k, v in completed.items()}, test_evaluated=False,
                    original_initialization_exact=True, on_exactly_restored=True, source_artifacts_preserved=True,
                    legacy_gradient_audit=audit_info, replay_passed=True, metrics=analysis['metrics'])
                with (args.run_dir / 'paired_predictions.npz').open('wb') as stream:
                    np.savez_compressed(stream, **cases, target=target, sample_indices=np.asarray(data.identity['splits']['val']['selected_indices']), station_ids=data.station_ids)
                write_json(args.run_dir / 'replay.json', replay)
                write_json(args.run_dir / 'case_metadata.json', metadata)
                write_json(args.run_dir / 'resources.json', resources)
                write_json(args.run_dir / 'preservation.json', dict(source_before=source_before,
                    source_after=snapshot(args.source_run), unchanged=True))
                write_report(args.run_dir, analysis, summary, timings, telemetry)
                write_json(args.run_dir / 'summary.json', summary)
                print(json.dumps({k: v for k, v in summary.items() if k != 'metrics'}, ensure_ascii=False), flush=True)
                print('Output directory: ' + str(args.run_dir), flush=True)
                return summary
            except BaseException as error:
                failure = dict(schema=SCHEMA, status='FIRST_LAYER_BOUNDARY_DELAY_V2_FAILED',
                    error_type=type(error).__name__, error=str(error), new_inference_calls=calls, test_evaluated=False)
                write_json(args.run_dir / 'summary.json', failure)
                write_json(args.run_dir / ('failure_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f') + '.json'), failure)
                raise


if __name__ == '__main__':
    main()
