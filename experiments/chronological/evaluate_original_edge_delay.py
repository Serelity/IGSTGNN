"""Validation-only best/last diagnosis of an existing original edge-delay pilot."""
import argparse
from contextlib import ExitStack, nullcontext
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import sys
import time
import uuid

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import numpy as np
import torch

from experiments.chronological.original_edge_delay import delay_mode, install_delay
from experiments.chronological.original_edge_delay_diagnostics import (
    LagTelemetry, existing_run_read_lock, independent_output, parameter_changes, summarize_cases, write_csv)
from experiments.chronological.original_igstgnn_checkpoint_compat import counter_weights_only_compat
from experiments.chronological.original_igstgnn_source import get_original_modules
from experiments.chronological.original_igstgnn_runtime import (
    build_model, configure_runtime, default_data_dir, device_batch, load_data,
    native_metrics, rng_fingerprint, sha256, tensor_tree_hash, write_json)
from experiments.chronological.evaluate_original_igstgnn_modules import source_from_run, verify_replay
from experiments.chronological import train_original_edge_delay as pilot

SCHEMA = 'original_edge_delay_best_last_diagnostic_v1'
EXECUTION_FILES = ('experiments/chronological/evaluate_original_edge_delay.py',
    'experiments/chronological/original_edge_delay_diagnostics.py',
    'experiments/chronological/original_igstgnn_ablation_metrics.py',
    'experiments/chronological/run_original_edge_delay_diagnostic.sh')


def snapshot(directory):
    result = {}
    for path in sorted(Path(directory).rglob('*')):
        # Windows byte-range locks deny reads through a second handle even
        # for this process. The pre-existing coordination file is opened rb
        # only; all scientific artifacts are hashed under its read lock.
        if path == Path(directory)/'active.lock':
            continue
        if path.is_symlink():
            raise ValueError('Input run contains a symlink: '+str(path))
        if path.is_file():
            result[str(path)] = sha256(path)
    return result


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load_pilot(directory, check):
    identity = read_json(directory/'identity.json')
    if identity.get('schema') != pilot.SCHEMA:
        raise ValueError('Expected an existing original edge-delay pilot')
    unsigned = {k: v for k, v in identity.items() if k != 'identity_sha256'}
    if pilot.digest(unsigned) != identity.get('identity_sha256'):
        raise ValueError('Pilot identity digest differs')
    if bool(identity['engineering_only']) != bool(check):
        raise ValueError('Use --check only for an already engineering-only pilot; no subset conversion')
    for name in pilot.EXECUTION_FILES:
        if identity['execution_source'].get(name) != sha256(REPO/name):
            raise ValueError('Historical execution source differs: '+name)
    summary = read_json(directory/'summary.json')
    if (summary.get('schema') != pilot.SCHEMA or not summary.get('same_completed_budget')
            or summary.get('status') not in ('EDGE_DELAY_ENGINEERING_PASS', 'EDGE_DELAY_PAIRED_PILOT_PASS')
            or bool(summary.get('engineering_only')) != bool(check)):
        raise ValueError('Expected a complete paired pilot; do not evaluate a running/partial run')
    states = {}
    training = identity['data']['splits']['train']
    for arm in pilot.ARMS:
        folder = directory/arm
        state = torch.load(folder/'last.pt', map_location='cpu', weights_only=True)
        pilot.validate_resume(state, identity, arm, len(training['partition']), training['count'], folder)
        filtered = {k: v for k, v in state.items() if k not in
            ('model_state', 'optimizer_state', 'scheduler_state', 'train_order', 'rng_state')}
        if filtered != read_json(folder/'summary.json') or state['history'] != read_json(folder/'history.json'):
            raise ValueError('Arm summary/history differs from committed last: '+arm)
        completed = dict(epochs=state['completed_epoch'], best_epoch=state['best_epoch'], updates=state['global_updates'])
        if summary['completed'].get(arm) != completed:
            raise ValueError('Root summary differs from committed arm: '+arm)
        states[arm] = state
    if len({s['completed_epoch'] for s in states.values()}) != 1:
        raise ValueError('Unequal completed arm budgets')
    return identity, summary, states


def prediction_archive(path, indices, stations):
    with np.load(path, allow_pickle=False) as values:
        if (not np.array_equal(values['sample_indices'], indices)
                or not np.array_equal(values['station_ids'], stations)):
            raise ValueError('Archived prediction axes differ: '+str(path))
        return values['prediction'].copy(), values['target'].copy()


def metric_replay(actual, expected):
    for key, log_key in (('mae', 'validation_mae'), ('rmse', 'validation_rmse'),
            ('mape_fraction', 'validation_mape_fraction')):
        if not np.isclose(actual[key], expected[log_key], atol=1e-4, rtol=1e-6):
            raise ValueError('Committed epoch metric replay mismatch: '+key)
    return dict(status='COMMITTED_EPOCH_METRIC_REPLAY_PASS', actual=actual,
        expected={k: expected[k] for k in ('epoch', 'validation_mae', 'validation_rmse', 'validation_mape_fraction')})


@torch.no_grad()
def evaluate_case(model, data, modules, device, name, calls):
    model.eval()
    before, rng_before = tensor_tree_hash(model.state_dict()), rng_fingerprint()
    telemetry = LagTelemetry(model)
    predictions, targets, timings = [], [], []
    try:
        for number, raw in enumerate(data.loaders['val_loader'].get_iterator(), 1):
            x, y, incident, sensors = device_batch(raw, device)
            pilot.sync(device)
            is_cuda = torch.device(device).type == 'cuda'
            baseline = torch.cuda.memory_allocated(device) if is_cuda else None
            if is_cuda:
                torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            calls[name] = calls.get(name, 0)+1
            prediction = model(x, incident_data=incident, sensor_data=sensors)
            pilot.sync(device)
            elapsed = time.perf_counter()-started
            peak = torch.cuda.max_memory_allocated(device) if is_cuda else None
            if prediction.shape != y.shape or not torch.isfinite(prediction).all():
                raise ValueError('Prediction axes/nonfinite values: '+name)
            started = time.perf_counter()
            telemetry.consume(x.shape[1])
            pilot.sync(device)
            diagnostic_seconds = time.perf_counter()-started
            timings.append(dict(batch=number, batch_size=len(x), forward_seconds=elapsed,
                diagnostic_seconds=diagnostic_seconds, peak_allocated_bytes=peak,
                incremental_peak_bytes=peak-baseline if is_cuda else None))
            predictions.append(data.scaler.inverse_transform(prediction).squeeze(-1).cpu().numpy())
            targets.append(data.scaler.inverse_transform(y).squeeze(-1).cpu().numpy())
            print(json.dumps(dict(phase=name, batch=number,
                total_batches=data.loaders['val_loader'].num_batch,
                forward_seconds=elapsed, diagnostic_seconds=diagnostic_seconds)), flush=True)
    finally:
        telemetry.close()
    if before != tensor_tree_hash(model.state_dict()) or rng_before != rng_fingerprint():
        raise ValueError('Read-only inference changed model tensors or RNG: '+name)
    prediction, target = np.concatenate(predictions), np.concatenate(targets)
    return dict(prediction=prediction, target=target, native=native_metrics(prediction, target, modules),
        timings=timings, telemetry=telemetry.finish(), model_hash=before, rng_preserved=True)


def make_pairs(cases):
    pairs = [('SOURCE', name) for name in cases if name != 'SOURCE']
    for stage in ('BEST', 'LAST'):
        for candidate in ('FIXED', 'DYNAMIC'):
            pairs.append(('CONTROL_'+stage, candidate+'_'+stage))
        pairs.append(('FIXED_'+stage, 'DYNAMIC_'+stage))
        for mode in ('NO_ALIGN', 'MASS_ONLY', 'SHUFFLE'):
            pairs.append(('DYNAMIC_'+stage+'_'+mode, 'DYNAMIC_'+stage))
        pairs.append(('FIXED_'+stage+'_NO_ALIGN', 'FIXED_'+stage))
    for arm in pilot.ARMS:
        pairs.append((arm+'_BEST', arm+'_LAST'))
    return list(dict.fromkeys(pairs))


def write_outputs(output, analysis, summary, native, telemetry, timings, changes):
    write_json(output/'analysis.json', analysis)
    write_json(output/'lag_statistics.json', telemetry)
    write_json(output/'timings.json', timings)
    write_json(output/'parameter_changes.json', changes)
    write_csv(output/'cases.csv', [dict(case=k, **{f: v for f, v in m.items() if f != 'horizons'}) for k, m in analysis['metrics'].items()])
    write_csv(output/'horizons.csv', [dict(case=k, **h) for k, m in analysis['metrics'].items() for h in m['horizons']])
    rows = [dict(comparison=k, group=g, **{f: v for f, v in m.items() if f != 'horizons'})
        for k, c in analysis['comparisons'].items() for g, m in c['groups'].items()]
    for name, prefix in (('comparisons', 'all_nodes'), ('flow', 'flow:'), ('roads', 'road:'),
            ('current_bias', 'bias:'), ('fixed_hard', 'fixed_hard')):
        write_csv(output/(name+'.csv'), [r for r in rows if r['group'].startswith(prefix)])
    write_csv(output/'group_horizons.csv', [dict(comparison=k, group=g, **h)
        for k, c in analysis['comparisons'].items() for g, m in c['groups'].items() for h in m['horizons']])
    write_csv(output/'boundary_support.csv', [dict(case=k, **r) for k, v in telemetry.items() for r in v['boundary_support']])
    lines = ['# 原IGSTGNN边时延：best/last只读诊断', '',
        f"工程检查：{summary['engineering_only']}；验证窗口：{summary['validation_rows']}。无训练更新，无测试集评价。", '',
        'best用于选模，last用于诊断训练；两者分别回放。指标是开发验证集上的探索结果。', '',
        'NO_ALIGN恢复原道路算子；MASS_ONLY保留相同边界可用系数质量，但值取同刻；SHUFFLE错配边时延。',
        '这些是同检查点推理干预，可能分布外，不等于删模块重训或事故因果证据。', '',
        '| 状态 | MAE pooled | MAE macro | RMSE pooled | MAPE (%) |', '|---|---:|---:|---:|---:|']
    for name, m in analysis['metrics'].items():
        lines.append(f"| {name} | {m['mae_pooled']:.9f} | {m['mae_macro']:.9f} | {m['rmse_pooled']:.9f} | {100*m['mape_pooled']:.9f} |")
    lines += ['', '| 比较：候选 vs 对照 | MAE收益 pooled | MAE收益 macro | MAPE收益百分点 | 固定困难窗口收益 |', '|---|---:|---:|---:|---:|']
    fmt = lambda v: '不可用' if v is None else f'{v:.9f}'
    for name, c in analysis['comparisons'].items():
        m = c['groups']['all_nodes']
        lines.append(f"| {name} | {fmt(m['mae_gain_pooled'])} | {fmt(m['mae_gain_macro'])} | {fmt(100*c['mape_gain_fraction'])} | {fmt(c['fixed_hard_window_gain'])} |")
    lines += ['', '## 历史支持与时延', '',
        '支持统计以实际时延在float64中重算一跳/两跳道路系数质量，按原规则去除最终对角；不是隐特征强度或真实容量。',
        'window_end_input_index是本输入窗口中的0起算历史位置；两个插值槽不是两个额外观测。',
        '分位数为2048个固定宽度区间的包络，不是精确分位数；零参考路径单列，不视为零损害。', '',
        '| 状态 | 层 | 方向 | 时延均值 | 跨边std均值 | 跨样本std均值 | 错配改变位置数 |', '|---|---:|---:|---:|---:|---:|---:|']
    for name, v in telemetry.items():
        for r in v['lags']:
            lines.append(f"| {name} | {r['layer']} | {r['graph']} | {fmt(r['mean'])} | {fmt(r['edge_std_per_sample_mean'])} | {fmt(r['sample_std_per_edge_mean'])} | {r['shuffle_changed_positions']} |")
    lines += ['', '## 回放、成本与解释边界', '',
        '原best预测逐点核对，last原生MAE/RMSE/MAPE核对日志；DYNAMIC干预后重新ON，检查预测、权重与RNG恢复。',
        '完整前向CUDA同步wall不含H2D/逆变换/统计；hook仅保存引用，归约在计时后执行。首批可含冷启动，不作公平加速结论。',
        'timings.json另列diagnostic_seconds，边界诊断成本不能算成模型前向成本；未新增反向或优化器更新。',
        '旧训练墙钟仅复用原history，包含加载/统计；未测本轮反向速度。',
        'SOURCE困难ID原样复用。真值流量/偏差仅用于事后诊断，偏差由DYNAMIC_LAST_NO_ALIGN统一定义。',
        '窗口/预测点可能重叠，不当成独立事故重复；保留负结果，不自动选择新架构。',
        '只构建验证loader；原发布对象数组会反序列化完整payload，不能声称从未读取测试payload。', '',
        f"完成状态：{summary['status']}；新增模型前向：{summary['total_new_inference_calls']}；优化更新：0。",
        '制品：analysis.json、cases/comparisons/horizons/flow/roads/current_bias/fixed_hard/group_horizons.csv、boundary_support.csv、lag_statistics.json、paired_predictions.npz、replay/preservation/provenance/timings/parameter_changes.json。']
    (output/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot-run', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--data-dir', type=Path, default=default_data_dir())
    parser.add_argument('--device', default=None)
    parser.add_argument('--threads', type=int, default=None)
    parser.add_argument('--check', action='store_true', help='Only for an existing engineering pilot; never subsets a formal pilot')
    args = parser.parse_args(argv)
    if args.threads is not None and args.threads < 1:
        parser.error('--threads must be positive')
    args.pilot_run, args.data_dir = args.pilot_run.resolve(), args.data_dir.resolve()
    if args.output_dir is None:
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
        args.output_dir = args.pilot_run.parent/(args.pilot_run.name+'_diagnostic_'+stamp+'_'+uuid.uuid4().hex[:6])
    args.output_dir = args.output_dir.resolve()
    return args


def main(argv=None):
    args = parse_args(argv)
    identity_header = read_json(args.pilot_run/'identity.json')
    if identity_header.get('schema') != pilot.SCHEMA:
        raise ValueError('Expected an existing original edge-delay pilot')
    source_dir = Path(identity_header['source']['path']).resolve()
    output = independent_output(args.output_dir, (args.pilot_run, source_dir, args.data_dir))
    calls, active_phase = {}, 'preflight'
    with ExitStack() as stack:
        stack.enter_context(counter_weights_only_compat())
        for root in (args.pilot_run, source_dir):
            stack.enter_context(existing_run_read_lock(root))
        before = dict(pilot=snapshot(args.pilot_run), source=snapshot(source_dir))
        identity, original_summary, states = load_pilot(args.pilot_run, args.check)
        device = args.device or identity['runtime']['python_device']
        runtime = configure_runtime(device, args.threads or identity['runtime']['threads'], identity['training_config']['seed'])
        if {k: runtime[k] for k in identity['runtime']} != identity['runtime']:
            raise ValueError('Diagnostic runtime differs from pilot; use its original device/versions/threads')
        modules = get_original_modules()
        source_state, source_config, expected, source = source_from_run(source_dir, check=args.check)
        for key in ('identity_sha256', 'checkpoint_sha256', 'source_fingerprint', 'source_files'):
            if source[key] != identity['source'][key]:
                raise ValueError('Original source identity/artifacts differ: '+key)
        if modules.source_fingerprint != source['source_fingerprint']:
            raise ValueError('Frozen original model reference differs')
        config = dict(identity['training_config'], device=device)
        data = load_data(args.data_dir, config, modules, logging.getLogger('edge-delay-diagnostic'), check=args.check, training=False)
        expected_data = dict(identity['data'], splits={'val': identity['data']['splits']['val']})
        if data.identity != expected_data:
            raise ValueError('Validation data/scaler/batch identity differs')
        output.mkdir(parents=True, exist_ok=False)
        try:
            write_json(output/'summary.json', dict(schema=SCHEMA, status='DIAGNOSTIC_RUNNING', optimizer_updates=0, test_evaluated=False))
            indices, stations = data.identity['splits']['val']['selected_indices'], data.station_ids
            source_archive, target = prediction_archive(args.pilot_run/'source_predictions.npz', indices, stations)
            hard_ids = read_json(args.pilot_run/'analysis.json')['fixed_hard_ids']
            selected = pilot.compare_predictions({'SOURCE': source_archive}, target, indices, data.roads)['fixed_hard_ids']
            if hard_ids != selected:
                raise ValueError('Original fixed hard IDs are not consistent with SOURCE archive')
            cases, native, telemetry, timings, changes, replay, cache, case_metadata = {}, {}, {}, {}, {}, {}, {}, {}

            def run(model, name, mode, force=False):
                nonlocal active_phase
                active_phase = name
                model.eval()
                with (delay_mode(model, mode) if mode is not None else nullcontext()):
                    key = (tensor_tree_hash(model.state_dict()), mode)
                    calls.setdefault(name, 0)
                    if key in cache and not force:
                        origin, result = cache[key]
                        print(json.dumps(dict(phase=name, reused_from=origin, new_model_calls=0)), flush=True)
                    else:
                        origin = None
                        result = evaluate_case(model, data, modules, device, name, calls)
                        if not force:
                            cache[key] = (name, result)
                    if not np.array_equal(result['target'], target):
                        raise ValueError('Validation targets changed: '+name)
                    if not force:
                        cases[name], native[name] = result['prediction'], result['native']
                        telemetry[name], timings[name] = result['telemetry'], dict(reused_from=origin, batches=result['timings'] if origin is None else [])
                        case_metadata[name] = dict(model_state_sha256=result['model_hash'], mode=mode,
                            reused_from=origin, rng_preserved=result['rng_preserved'])
                    return result

            model = build_model(data, modules, device)
            model.load_state_dict(source_state, strict=True)
            result = run(model, 'SOURCE', None)
            replay['SOURCE'] = verify_replay(result['native'], expected, engineering_only=args.check)
            if not np.array_equal(result['prediction'], source_archive):
                raise ValueError('SOURCE archive prediction is not exactly replayed')
            del model
            for arm in pilot.ARMS:
                state = states[arm]
                model = build_model(data, modules, device)
                model.load_state_dict(source_state, strict=True)
                backbone_parameter_names = set(dict(model.named_parameters()))
                if arm != 'CONTROL':
                    install_delay(model, arm, **identity['architecture'], seed=identity['training_config']['seed'])
                initial_extra = {k:v.detach().cpu().clone() for k,v in model.named_parameters() if '.dif_layer.delays.' in k}
                for stage in ('BEST', 'LAST'):
                    weights = (torch.load(args.pilot_run/arm/state['best_checkpoint_file'], map_location='cpu', weights_only=True)
                        if stage == 'BEST' else state['model_state'])
                    model.load_state_dict(weights, strict=True)
                    name = arm+'_'+stage
                    changes[name] = parameter_changes(weights, source_state, initial_extra, backbone_parameter_names)
                    result = run(model, name, None if arm == 'CONTROL' else arm)
                    epoch = state['best_epoch'] if stage == 'BEST' else state['completed_epoch']
                    replay[name] = metric_replay(result['native'], state['history'][epoch])
                    case_metadata[name].update(arm=arm, selection=stage, epoch=epoch,
                        checkpoint_file=state['best_checkpoint_file'] if stage == 'BEST' else 'last.pt')
                    if stage == 'BEST':
                        archived, saved_target = prediction_archive(args.pilot_run/arm/state['best_prediction_file'], indices, stations)
                        if not np.array_equal(saved_target, target) or not np.array_equal(archived, result['prediction']):
                            raise ValueError('BEST archived prediction mismatch: '+arm)
                        replay[name]['archived_prediction_exact'] = True
                    if arm != 'CONTROL':
                        modes = ('NO_ALIGN',) if arm == 'FIXED' else ('NO_ALIGN', 'MASS_ONLY', 'SHUFFLE')
                        for mode in modes:
                            run(model, name+'_'+mode, mode)
                        if arm == 'DYNAMIC':
                            restored = run(model, name+'_RESTORED', arm, force=True)
                            if not np.array_equal(restored['prediction'], cases[name]):
                                raise ValueError('DYNAMIC ON prediction was not exactly restored')
                            replay[name]['on_exactly_restored'] = True
                            timings[name+'_RESTORED'] = dict(reused_from=None, batches=restored['timings'])
                del model
            analysis = summarize_cases(cases, target, indices, stations, data.roads, hard_ids, make_pairs(cases))
            for arm in pilot.ARMS:
                for old, new in (('mae_pooled', 'mae_pooled'), ('mae_macro', 'mae_macro'),
                        ('rmse_pooled', 'rmse_pooled'), ('mape_fraction', 'mape_pooled')):
                    if not np.isclose(original_summary['metrics'][arm][old], analysis['metrics'][arm+'_BEST'][new], atol=1e-10, rtol=1e-12):
                        raise ValueError('Original root BEST metric differs: '+arm+' '+old)
            after = dict(pilot=snapshot(args.pilot_run), source=snapshot(source_dir))
            if after != before:
                raise ValueError('Input source/pilot artifacts changed during diagnosis')
            summary = dict(schema=SCHEMA, status='EDGE_DELAY_BEST_LAST_DIAGNOSTIC_PASS', engineering_only=args.check,
                completed=original_summary['completed'], validation_rows=len(target), valid_points=analysis['valid_points'],
                cases=list(cases), new_inference_calls=calls, total_new_inference_calls=sum(calls.values()),
                optimizer_updates=0, test_evaluated=False, source_artifacts_preserved=True,
                replay_passed=True, on_exactly_restored=True, metrics=analysis['metrics'])
            write_json(output/'replay.json', replay)
            write_json(output/'preservation.json', dict(before=before, after=after, all_input_files_unchanged=True,
                input_locks='existing read locks; no input files created/written',
                coordination_lock_contents_excluded_from_hash_snapshot=True))
            write_json(output/'provenance.json', dict(schema=SCHEMA, pilot_run=str(args.pilot_run),
                pilot_identity_sha256=identity['identity_sha256'], runtime=runtime, data=data.identity,
                source=source, historical_execution_source=identity['execution_source'],
                diagnostic_execution_source={p: sha256(REPO/p) for p in EXECUTION_FILES},
                checkpoint_cases=case_metadata, training_history_reused={a:s['history'] for a,s in states.items()},
                new_backward_calls=0, optimizer_updates=0, test_evaluated=False))
            with (output/'paired_predictions.npz').open('wb') as stream:
                np.savez_compressed(stream, **cases, target=target, valid_mask=target != analysis['mask_value'],
                    sample_indices=np.asarray(indices), station_ids=stations)
            write_outputs(output, analysis, summary, native, telemetry, timings, changes)
            write_json(output/'summary.json', summary)
            print(json.dumps({k:v for k,v in summary.items() if k != 'metrics'}, ensure_ascii=False), flush=True)
            print('Output directory: '+str(output), flush=True)
            return summary
        except BaseException as error:
            failure = dict(schema=SCHEMA, status='EDGE_DELAY_BEST_LAST_DIAGNOSTIC_FAILED', phase=active_phase,
                error_type=type(error).__name__, error=str(error), new_inference_calls=calls,
                optimizer_updates=0, test_evaluated=False, input_files_unchanged=(before == dict(
                    pilot=snapshot(args.pilot_run), source=snapshot(source_dir))))
            write_json(output/'failure.json', failure)
            write_json(output/'summary.json', failure)
            raise


if __name__ == '__main__':
    main()
