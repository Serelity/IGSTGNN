"""Paired warmstart pilot: original CONTROL, static FIXED, state DYNAMIC lag.

Uses the original best checkpoint, published train/validation batches and
native engine. No test loader; no early stop; all requested arms receive the
same completed extra-epoch budget. Stage zero participates in selection.
"""
import argparse
from datetime import datetime, timezone
import hashlib
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

from experiments.chronological.original_edge_delay import description, delay_diagnostics, delay_mode, install_delay
from experiments.chronological.original_igstgnn_checkpoint_compat import counter_weights_only_compat
from experiments.chronological.original_igstgnn_source import get_original_modules
from experiments.chronological.original_igstgnn_runtime import (
    build_model, configure_runtime, default_data_dir, device_batch, load_data,
    native_metrics, run_lock, sha256, tensor_tree_hash, write_json)
from experiments.chronological.evaluate_original_igstgnn_modules import source_from_run, verify_replay
from experiments.chronological.train_original_igstgnn_baseline import (
    MILESTONES, atomic_save, restore_rng, rng_state, tracked_shuffle)


SCHEMA = 'original_igstgnn_edge_delay_pilot_v1'
ARMS = ('CONTROL', 'FIXED', 'DYNAMIC')
EXECUTION_FILES = (
    'experiments/chronological/original_edge_delay.py',
    'experiments/chronological/train_original_edge_delay.py',
    'experiments/chronological/original_igstgnn_runtime.py',
    'experiments/chronological/train_original_igstgnn_baseline.py',
    'experiments/chronological/evaluate_original_igstgnn_modules.py',
    'experiments/chronological/original_igstgnn_checkpoint_compat.py',
    'experiments/chronological/original_igstgnn_source.py')


def sync(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def save_predictions(path, prediction, target, indices, stations):
    temporary = path.with_suffix('.npz.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, prediction=prediction, target=target,
            sample_indices=np.asarray(indices), station_ids=stations)
    temporary.replace(path)


@torch.no_grad()
def validate(model, data, modules, device, phase):
    model.eval()
    predictions, targets, timings = [], [], []
    for number, raw in enumerate(data.loaders['val_loader'].get_iterator(), 1):
        x, y, incident, sensors = device_batch(raw, device)
        sync(device)
        if torch.device(device).type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        prediction = model(x, incident_data=incident, sensor_data=sensors)
        sync(device)
        elapsed = time.perf_counter() - started
        if prediction.shape != y.shape or not torch.isfinite(prediction).all():
            raise ValueError('Invalid prediction axes or nonfinite values')
        timings.append(dict(batch=number, batch_size=len(x), forward_seconds=elapsed,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if torch.device(device).type == 'cuda' else None,
            delay_diagnostics=delay_diagnostics(model)))
        predictions.append(data.scaler.inverse_transform(prediction).squeeze(-1).cpu().numpy())
        targets.append(data.scaler.inverse_transform(y).squeeze(-1).cpu().numpy())
        print(json.dumps(dict(phase=phase, batch=number,
            total_batches=data.loaders['val_loader'].num_batch, forward_seconds=elapsed)), flush=True)
    prediction, target = np.concatenate(predictions), np.concatenate(targets)
    return native_metrics(prediction, target, modules), prediction, target, timings


def metric_summary(prediction, target, mask):
    error = prediction.astype(np.float64) - target.astype(np.float64)
    count = mask.sum(axis=(0, 2))
    absolute = np.where(mask, np.abs(error), 0).sum(axis=(0, 2))
    squared = np.where(mask, error * error, 0).sum(axis=(0, 2))
    percentage = np.divide(np.abs(error), target, out=np.zeros_like(error), where=mask)
    mae = np.divide(absolute, count, out=np.zeros_like(absolute), where=count > 0)
    rmse = np.sqrt(np.divide(squared, count, out=np.zeros_like(squared), where=count > 0))
    return dict(valid_points=int(count.sum()), mae_pooled=float(absolute.sum() / count.sum()),
        mae_macro=float(mae.mean()) if (count > 0).all() else None,
        rmse_pooled=float(np.sqrt(squared.sum() / count.sum())),
        mape_fraction=float(percentage.sum() / count.sum()),
        horizons=[dict(step=i+1, mae=float(mae[i]) if count[i] else None,
            rmse=float(rmse[i]) if count[i] else None, count=int(count[i])) for i in range(len(count))])


def compare_predictions(cases, target, indices, roads):
    if 'SOURCE' not in cases or not np.isfinite(target).all():
        raise ValueError('Expected finite paired arrays and SOURCE predictions')
    null = float(np.min(target)) if np.min(target) < 1 else 0.
    mask = target != null
    if not mask.any() or (target[mask] <= 0).any():
        raise ValueError('Original MAPE requires positive valid targets')
    source_error = np.abs(cases['SOURCE'].astype(np.float64) - target)
    counts = mask.sum(2)
    complete = (counts > 0).all(1)
    scores = np.divide(np.where(mask, source_error, 0).sum(2), counts,
        out=np.zeros_like(counts, dtype=np.float64), where=counts > 0).mean(1)
    eligible = np.flatnonzero(complete)
    hard = eligible[np.lexsort((np.asarray(indices)[eligible], -scores[eligible]))[:int(np.ceil(len(eligible)*.1))]]
    groups = {'all_nodes': mask, 'fixed_hard': mask & np.isin(np.arange(len(target)), hard)[:, None, None]}
    for low, high in zip((0, 5, 20, 50, 100, 200), (5, 20, 50, 100, 200, float('inf'))):
        groups[f'flow:[{low},{high})'] = mask & (target >= low) & (target < high)
    for road, members in roads.items():
        groups['road:' + road] = mask & np.isin(np.arange(target.shape[2]), members)[None, None, :]
    metrics = {}
    for name, prediction in cases.items():
        if prediction.shape != target.shape or not np.isfinite(prediction).all():
            raise ValueError('Unpaired prediction: ' + name)
        metrics[name] = metric_summary(prediction, target, mask)
    comparisons = {}
    pairs = [('SOURCE', name) for name in cases if name != 'SOURCE']
    pairs += [(a, b) for a, b in [('CONTROL', 'FIXED'), ('CONTROL', 'DYNAMIC'),
        ('FIXED', 'DYNAMIC'), ('DYNAMIC_NO_ALIGN', 'DYNAMIC'), ('DYNAMIC_SHUFFLE', 'DYNAMIC'),
        ('DYNAMIC_MASS_ONLY', 'DYNAMIC')]
        if a in cases and b in cases]
    for reference, candidate in pairs:
        gain = np.abs(cases[reference].astype(np.float64) - target) - np.abs(cases[candidate].astype(np.float64) - target)
        values = {}
        for name, group in groups.items():
            count = int(group.sum())
            values[name] = dict(count=count, mae_gain=float(gain[group].mean()) if count else None,
                global_mae_contribution=float(np.where(group, gain, 0).sum() / mask.sum()))
        values['horizons'] = [float(gain[:, i][mask[:, i]].mean()) if mask[:, i].any() else None for i in range(target.shape[1])]
        comparisons[candidate + '_vs_' + reference] = values
    return dict(metrics=metrics, comparisons=comparisons, fixed_hard_ids=np.asarray(indices)[hard].tolist(),
        mask_value=null, hard_selection='SOURCE top 10 percent complete-window macro MAE; fixed across candidates',
        gain_sign='reference absolute error minus candidate absolute error', mape_unit='fraction')


def validate_resume(state, identity, arm, batches, samples, directory):
    if state.get('identity_sha256') != identity['identity_sha256'] or state.get('arm') != arm:
        raise ValueError('Resume identity/arm differs')
    epoch, history = state['completed_epoch'], state['history']
    if (type(epoch) is not int or epoch < 0 or len(history) != epoch + 1
            or [row['epoch'] for row in history] != list(range(epoch + 1))
            or state['global_updates'] != epoch * batches
            or [row['global_updates'] for row in history] != [i * batches for i in range(epoch + 1)]
            or not all(np.isfinite(row['validation_mae']) for row in history)):
        raise ValueError('Resume budget/history invalid')
    best = min(history, key=lambda row: row['validation_mae'])
    if best['epoch'] != state['best_epoch'] or best['validation_mae'] != state['best_validation_mae']:
        raise ValueError('Resume best selection invalid')
    order = np.asarray(state['train_order'])
    if order.shape != (samples,) or not np.array_equal(np.sort(order), np.arange(samples)):
        raise ValueError('Resume train order invalid')
    for kind in ('checkpoint', 'prediction'):
        name = state['best_' + kind + '_file']
        if Path(name).name != name or sha256(directory/name) != state['best_' + kind + '_sha256']:
            raise ValueError('Resume best artifact fingerprint differs')
    if state['scheduler_state']['last_epoch'] != epoch:
        raise ValueError('Resume scheduler epoch invalid')
    warm = identity['training_config']['warm_epoch'] * batches
    step = identity['training_config']['cl_epoch'] * batches
    updates = state['global_updates']
    length = 0 if updates == 0 else (12 if updates < warm else min(12, 1+(updates-warm)//step))
    if state['curriculum_length'] != length:
        raise ValueError('Resume curriculum invalid')


def run_arm(arm, model, data, modules, identity, directory, epochs, resume=False):
    directory.mkdir(parents=True, exist_ok=True)
    config = data.config
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lrate'], weight_decay=config['wdecay'], eps=1e-8)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=MILESTONES, gamma=.5)
    batches = data.loaders['train_loader'].num_batch
    engine = modules.engine.IGSTGNN_Engine(device=torch.device(config['device']), model=model,
        dataloader=data.loaders, scaler=data.scaler, sampler=None, loss_fn=modules.metrics.masked_mae,
        lrate=config['lrate'], optimizer=optimizer, scheduler=scheduler,
        clip_grad_value=config['clip_grad_value'], max_epochs=100, patience=100,
        log_dir=str(directory), logger=logging.getLogger('edge-delay'), seed=config['seed'],
        cl_step=config['cl_epoch']*batches, warm_step=config['warm_epoch']*batches, horizon=12, incident=True)
    state = None
    if (directory/'last.pt').exists():
        if not resume:
            raise ValueError('Existing arm requires --resume')
        state = torch.load(directory/'last.pt', map_location='cpu', weights_only=True)
        validate_resume(state, identity, arm, batches, len(data.loaders['train_loader'].samples), directory)
        if state['completed_epoch'] > epochs:
            raise ValueError('Resume target epochs cannot decrease the completed budget')
        model.load_state_dict(state['model_state'], strict=True)
        optimizer.load_state_dict(state['optimizer_state'])
        scheduler.load_state_dict(state['scheduler_state'])
        engine._iter_cnt, engine._cl_len = state['global_updates'], state['curriculum_length']
    get_order = tracked_shuffle(data.loaders['train_loader'], None if state is None else state['train_order'])
    if state is not None:
        restore_rng(state['rng_state'])
    else:
        state = dict(arm=arm, completed_epoch=-1, best_epoch=None, best_validation_mae=float('inf'), history=[])
    for epoch in range(state['completed_epoch']+1, epochs+1):
        train_seconds, train_metrics = 0., None
        if epoch:
            print(json.dumps(dict(phase='training', arm=arm, epoch=epoch, total_batches=batches)), flush=True)
            sync(config['device'])
            started = time.perf_counter()
            train_metrics = list(map(float, engine.train_batch()))
            sync(config['device'])
            train_seconds = time.perf_counter() - started
        metrics, prediction, target, timings = validate(model, data, modules, config['device'], f'{arm}_epoch_{epoch}')
        if train_metrics is not None and not np.isfinite(train_metrics).all():
            raise ValueError('Nonfinite training metrics')
        row = dict(epoch=epoch, validation_mae=metrics['mae'], validation_rmse=metrics['rmse'],
            validation_mape_fraction=metrics['mape_fraction'], train_metrics_mae_mape_rmse=train_metrics,
            train_seconds_including_loading_and_metrics=train_seconds,
            validation_forward_seconds=sum(t['forward_seconds'] for t in timings),
            global_updates=engine._iter_cnt, curriculum_length=engine._cl_len,
            learning_rate=optimizer.param_groups[0]['lr'])
        if epoch:
            scheduler.step()
        state['history'].append(row)
        if metrics['mae'] < state['best_validation_mae']:
            stem = f'best_epoch_{epoch:03d}_{uuid.uuid4().hex[:10]}'
            atomic_save(directory/(stem+'.pt'), model.state_dict())
            save_predictions(directory/(stem+'.npz'), prediction, target,
                data.identity['splits']['val']['selected_indices'], data.station_ids)
            state.update(best_epoch=epoch, best_validation_mae=metrics['mae'],
                best_checkpoint_file=stem+'.pt', best_checkpoint_sha256=sha256(directory/(stem+'.pt')),
                best_prediction_file=stem+'.npz', best_prediction_sha256=sha256(directory/(stem+'.npz')))
        state.update(completed_epoch=epoch, global_updates=engine._iter_cnt, curriculum_length=engine._cl_len,
            identity_sha256=identity['identity_sha256'], model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict(), scheduler_state=scheduler.state_dict(),
            train_order=get_order(), rng_state=rng_state())
        atomic_save(directory/'last.pt', state)
        write_json(directory/f'validation_epoch_{epoch:03d}.json', timings)
        write_json(directory/'history.json', state['history'])
        write_json(directory/'summary.json', {key: value for key, value in state.items() if key not in
            ('model_state', 'optimizer_state', 'scheduler_state', 'train_order', 'rng_state')})
        print(json.dumps(dict(arm=arm, **row)), flush=True)
    # Reload committed state so loading the best model cannot mutate tensor
    # references in the returned last-epoch state_dict.
    state = torch.load(directory/'last.pt', map_location='cpu', weights_only=True)
    write_json(directory/'history.json', state['history'])
    write_json(directory/'summary.json', {key: value for key, value in state.items() if key not in
        ('model_state', 'optimizer_state', 'scheduler_state', 'train_order', 'rng_state')})
    # Best selected checkpoint, not the last epoch, supplies all OFF probes.
    model.load_state_dict(torch.load(directory/state['best_checkpoint_file'], map_location=config['device'], weights_only=True), strict=True)
    model.eval()
    return state


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, default=default_data_dir())
    parser.add_argument('--epochs', type=int, default=5, help='Total extra epochs per arm; stage zero included in selection')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--threads', type=int, default=3)
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--max-delay', type=int, default=2, help='Maximum one-edge lag in original observation steps')
    parser.add_argument('--score-dim', type=int, default=8)
    parser.add_argument('--arms', nargs='+', choices=ARMS, default=list(ARMS))
    parser.add_argument('--check', action='store_true', help='Four train/val rows, batch 2; engineering only')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    if not 1 <= args.epochs <= 100 or args.max_delay < 1 or args.score_dim < 1 or len(set(args.arms)) != len(args.arms):
        parser.error('Invalid epoch/delay/scorer/arm setting')
    args.run_dir, args.source_run, args.data_dir = (p.resolve() for p in (args.run_dir, args.source_run, args.data_dir))
    if args.run_dir == args.source_run or args.source_run in args.run_dir.parents or args.run_dir in args.source_run.parents:
        parser.error('Use a separate output directory, outside the source run')
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.run_dir.exists() != args.resume:
        raise ValueError('Use a fresh run directory or --resume an existing pilot')
    if args.resume:
        # Reject unrelated directories before creating a lock or writing any
        # status/failure artifact into them.
        identity_path = args.run_dir/'identity.json'
        if not identity_path.is_file() or json.loads(identity_path.read_text(encoding='utf-8')).get('schema') != SCHEMA:
            raise ValueError('Resume directory is not an edge-delay pilot')
    args.run_dir.mkdir(parents=True, exist_ok=args.resume)
    with run_lock(args.run_dir), counter_weights_only_compat():
        try:
            runtime = configure_runtime(args.device, args.threads, args.seed)
            modules = get_original_modules()
            source_state, source_config, expected, provenance = source_from_run(args.source_run, check=args.check)
            if provenance['source_fingerprint'] != modules.source_fingerprint:
                raise ValueError('Source is not the frozen original reference')
            config = dict(source_config, seed=args.seed, device=args.device)
            data = load_data(args.data_dir, config, modules, logging.getLogger('edge-delay'), check=args.check, training=True)
            # build_model uses data.config; preserve the selected source config.
            data.config['device'] = args.device
            identity = dict(schema=SCHEMA, source=provenance, training_config=config, data=data.identity,
                architecture=dict(max_delay=args.max_delay, score_dim=args.score_dim),
                execution_source={p: sha256(REPO/p) for p in EXECUTION_FILES},
                runtime={k: v for k, v in runtime.items() if k != 'device_name'}, engineering_only=args.check,
                arm_order=list(ARMS), stage_zero_selected=True, early_stopping=False,
                source_optimizer_resumed=False, optimizer_scheduler='reset in every arm; original Adam and milestones',
                curriculum='original engine, reset identically in all arms; first 5 epochs train all 12 horizons',
                test_evaluated=False)
            identity['identity_sha256'] = digest(identity)
            if args.resume:
                if json.loads((args.run_dir/'identity.json').read_text(encoding='utf-8')) != identity:
                    raise ValueError('Pilot identity differs; use a fresh directory')
            else:
                write_json(args.run_dir/'identity.json', identity)
                write_json(args.run_dir/'runtime.json', runtime)
            write_json(args.run_dir/'summary.json', dict(schema=SCHEMA, status='EDGE_DELAY_PILOT_RUNNING',
                target_extra_epochs=args.epochs, requested_arms=args.arms, engineering_only=args.check,
                test_evaluated=False))
            model = build_model(data, modules, args.device)
            model.load_state_dict(source_state, strict=True)
            metrics, source_prediction, target, timing = validate(model, data, modules, args.device, 'SOURCE_replay')
            write_json(args.run_dir/'source_replay.json', dict(verify_replay(metrics, expected, engineering_only=args.check), timings=timing))
            save_predictions(args.run_dir/'source_predictions.npz', source_prediction, target,
                data.identity['splits']['val']['selected_indices'], data.station_ids)
            del model
            train_samples = tuple(data.loaders['train_loader'].samples)
            arm_states = {}
            for arm in ARMS:
                if arm not in args.arms:
                    continue
                configure_runtime(args.device, args.threads, args.seed)
                data.loaders['train_loader'].samples = list(train_samples)
                model = build_model(data, modules, args.device)
                model.load_state_dict(source_state, strict=True)
                if arm != 'CONTROL':
                    install_delay(model, arm, args.max_delay, args.score_dim, args.seed)
                write_json(args.run_dir/(arm+'_architecture.json'), description(model))
                state = run_arm(arm, model, data, modules, identity, args.run_dir/arm, args.epochs, args.resume)
                arm_states[arm] = state['completed_epoch']
                if arm == 'DYNAMIC':
                    before = tensor_tree_hash(model.state_dict())
                    for mode in ('NO_ALIGN', 'SHUFFLE', 'MASS_ONLY'):
                        with delay_mode(model, mode):
                            _, prediction, probe_target, probe_time = validate(model, data, modules, args.device, 'DYNAMIC_'+mode)
                        if not np.array_equal(probe_target, target) or tensor_tree_hash(model.state_dict()) != before:
                            raise ValueError('Delay probe changed model or target axis')
                        save_predictions(args.run_dir/('DYNAMIC_'+mode+'.npz'), prediction, target,
                            data.identity['splits']['val']['selected_indices'], data.station_ids)
                        write_json(args.run_dir/('DYNAMIC_'+mode+'_timing.json'), probe_time)
                    _, restored, _, _ = validate(model, data, modules, args.device, 'DYNAMIC_restored')
                    with np.load(args.run_dir/arm/state['best_prediction_file']) as saved:
                        if not np.array_equal(restored, saved['prediction']):
                            raise ValueError('DYNAMIC best predictions were not exactly restored')
                del model, state
            cases, completed = {'SOURCE': source_prediction}, {}
            for arm in ARMS:
                path = args.run_dir/arm/'last.pt'
                if not path.exists():
                    continue
                saved = torch.load(path, map_location='cpu', weights_only=True)
                validate_resume(saved, identity, arm, data.loaders['train_loader'].num_batch, len(train_samples), path.parent)
                with np.load(path.parent/saved['best_prediction_file']) as values:
                    if (not np.array_equal(values['target'], target) or not np.array_equal(values['station_ids'], data.station_ids)
                            or not np.array_equal(values['sample_indices'], data.identity['splits']['val']['selected_indices'])):
                        raise ValueError('Arm prediction target/station axes differ')
                    cases[arm] = values['prediction'].copy()
                completed[arm] = dict(epochs=saved['completed_epoch'], best_epoch=saved['best_epoch'], updates=saved['global_updates'])
                if arm == 'DYNAMIC' and arm in arm_states:
                    for mode in ('NO_ALIGN', 'SHUFFLE', 'MASS_ONLY'):
                        with np.load(args.run_dir/('DYNAMIC_'+mode+'.npz')) as values:
                            cases['DYNAMIC_'+mode] = values['prediction'].copy()
            for file, expected_hash in provenance['source_files'].items():
                if sha256(file) != expected_hash:
                    raise ValueError('Source run changed during pilot')
            paired = set(completed) == set(ARMS) and len({v['epochs'] for v in completed.values()}) == 1
            analysis = compare_predictions(cases, target, data.identity['splits']['val']['selected_indices'], data.roads)
            summary = dict(schema=SCHEMA, status='EDGE_DELAY_ENGINEERING_PASS' if args.check else
                ('EDGE_DELAY_PAIRED_PILOT_PASS' if paired else 'EDGE_DELAY_PARTIAL'), engineering_only=args.check,
                completed=completed, same_completed_budget=paired, source_artifacts_preserved=True,
                validation_rows=len(target), requested_extra_epochs=args.epochs,
                test_evaluated=False, optimizer_updates={k: v['updates'] for k, v in completed.items()},
                metrics=analysis['metrics'])
            write_json(args.run_dir/'analysis.json', analysis)
            write_json(args.run_dir/'summary.json', summary)
            lines = ['# 原版 IGSTGNN：道路边时延首轮配对实验', '',
                '验证集探索性结果；不是独立测试集结论。原模型额外微调 CONTROL 是主要同预算对照。', '',
                'FIXED 为训练得到的固定边时延；DYNAMIC 为边常数加状态修正。NO_ALIGN/SHUFFLE 为同一 DYNAMIC 最佳检查点推理干预。', '',
                'MASS_ONLY 保留相同延迟造成的边界可用消息质量，但所有值读取当前摘要；用于区分时间对齐与边界衰减。', '',
                '仅历史 DIF 道路 bank 改动；原三步投影、自身/动态图/静态图、INH、未来 gap=3 递推与事故模块保留。', '',
                '时延作用于完整窗口摘要的结束时刻；边界丢弃无历史的消息质量。窗口内时延固定，多跳沿路径累计。', '',
                '零时延仅在数学上恢复原道路算子；NO_ALIGN 直接调用原代码。训练初始时延非零，阶段0参加选模。', '',
                '方向为原邻接及其转置的计算方向，不认证物理方向；学习时延不等于真实旅行或冲击波传播时间。', '',
                f'工程检查：{args.check}；等完成预算：{paired}；阶段：{completed}', '',
                '| 状态 | MAE pooled | MAE macro | RMSE pooled | MAPE (%) |', '|---|---:|---:|---:|---:|']
            for name, value in analysis['metrics'].items():
                lines.append(f"| {name} | {value['mae_pooled']:.9f} | {value['mae_macro']:.9f} | {value['rmse_pooled']:.9f} | {100*value['mape_fraction']:.9f} |")
            lines += ['', '| 比较（候选 vs 对照） | MAE收益 | 固定困难集收益 |', '|---|---:|---:|']
            for name, value in analysis['comparisons'].items():
                lines.append(f"| {name} | {value['all_nodes']['mae_gain']:.9f} | {value['fixed_hard']['mae_gain']:.9f} |")
            lines += ['', '逐时距、流量、道路与固定困难ID见 analysis.json。训练包含数据加载/统计的墙钟见各 arm/history.json；',
                '同步完整前向计时（不含H2D、逆变换、CPU统计）见 source_replay.json 与探针 timing.json。首批包含冷启动，不作公平加速结论。',
                '相同训练预算不意味着同检查点消融是删模块重训效果；历史验证已用于设计，样本窗口可能重叠。',
                '本轮为原最佳检查点的配对额外微调，不是从头同预算训练或原训练优化器的精确续训。']
            (args.run_dir/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
            print(json.dumps(dict(summary, metrics={name: {key: value for key, value in metric.items()
                if key != 'horizons'} for name, metric in summary['metrics'].items()}), ensure_ascii=False), flush=True)
            return summary
        except BaseException as error:
            write_json(args.run_dir/'summary.json', dict(schema=SCHEMA, status='EDGE_DELAY_PILOT_FAILED',
                error_type=type(error).__name__, error=str(error), engineering_only=args.check, test_evaluated=False))
            write_json(args.run_dir/('failure_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')+'.json'),
                dict(status='EDGE_DELAY_PILOT_FAILED', error_type=type(error).__name__, error=str(error), test_evaluated=False))
            raise


if __name__ == '__main__':
    main()
