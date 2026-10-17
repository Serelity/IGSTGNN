"""GW2 validation replay, paired metrics and cost report; never evaluates test.

--run-dir performs inference from a completed common four-arm budget.
--archive only reanalyzes saved predictions without models or inference.
Interventions keep the jointly trained native backbone, so are not retrained
deletions or independent training contributions.
"""
import argparse
from contextlib import nullcontext
import gc
import json
import logging
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.chronological.original_graph_message import ARMS, DESIGN
from experiments.chronological.original_igstgnn_runtime import (
    configure_runtime, default_data_dir, device_batch, load_data, native_metrics, sha256, write_json)
from experiments.chronological.original_igstgnn_source import get_original_modules
from experiments.chronological.train_original_graph_message import (
    SCHEMA as TRAIN_SCHEMA, EXECUTION_FILES, PAIRS, load_checkpoint, model_for_arm, validate_saved)
# Pure numerical routines are reused, not v0.3 archive identities/arm mappings.
from experiments.chronological.evaluate_unified_adapter_v03 import (
    point_metrics, paired_metrics, fixed_hard_indices, window_scores, _write_csv)

SCHEMA = 'original_igstgnn_graph_message_archive_v1'
CASES = (*ARMS, 'STATIC_UNIFORM', 'POST_LOCAL_UNIFORM',
         'POST_MESSAGE_UNIFORM', 'POST_MESSAGE_STATIC_ONLY')
COMPARISONS = tuple((b+'_vs_'+a, a, b) for a, b in PAIRS) + (
    ('static_selector_dependency', 'STATIC_UNIFORM', 'STATIC'),
    ('local_selector_dependency', 'POST_LOCAL_UNIFORM', 'POST_LOCAL'),
    ('same_checkpoint_selector_dependency', 'POST_MESSAGE_UNIFORM', 'POST_MESSAGE'),
    ('same_checkpoint_content_dependency', 'POST_MESSAGE_STATIC_ONLY', 'POST_MESSAGE'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_archive(archive):
    require(str(np.asarray(archive.get('schema')).item()) == SCHEMA, 'Not a GW2 prediction archive')
    require(str(np.asarray(archive.get('axis_order')).item()) == 'sample,horizon,node', 'Explicit GW2 axes required')
    y, valid = np.asarray(archive['y']), np.asarray(archive['valid'])
    require(y.ndim == 3 and y.shape[1] == 12 and y.dtype.kind == 'f' and min(y.shape) > 0,
            'Expected floating [sample,12,node] targets')
    require(valid.shape == y.shape and valid.dtype == np.bool_ and valid.any(), 'Shared Boolean target mask required')
    require(np.isfinite(y).all() and (y[valid] > 0).all(), 'Finite targets and positive valid MAPE targets required')
    null = float(y.min()) if y.min() < 1 else 0.
    require(np.array_equal(valid, y != null), 'Mask differs from the original global-null rule')
    for key, count in (('sample_ids', y.shape[0]), ('node_ids', y.shape[2])):
        ids = np.asarray(archive[key])
        require(ids.shape == (count,) and ids.dtype.kind in 'iu' and len(np.unique(ids)) == count, 'Invalid unique '+key)
    require(np.asarray(archive['road_labels']).shape == (y.shape[2],)
            and np.asarray(archive['road_labels']).dtype.kind in 'US', 'Invalid road axis')
    require(np.asarray(archive['direct_support']).shape == (y.shape[0], y.shape[2])
            and np.asarray(archive['direct_support']).dtype == np.bool_, 'Invalid direct-support axis')
    predictions = {key[6:]: np.asarray(value) for key, value in archive.items() if key.startswith('pred__')}
    require(set(predictions) == set(CASES), 'Exactly the predefined eight GW2 states are required')
    for name, p in predictions.items():
        require(p.shape == y.shape and p.dtype.kind == 'f' and np.isfinite(p).all(), 'Invalid prediction axes/values: '+name)
    return y, valid, predictions


def report(archive, output, metadata):
    y, valid, predictions = validate_archive(archive)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    require(not (output/'analysis.json').exists() and not (output/'report.md').exists(), 'Existing analysis is preserved')
    hard = fixed_hard_indices(predictions['ORIGINAL'], y, valid, archive['sample_ids'])
    global_count = int(valid.sum())
    hard_mask = np.zeros(y.shape[0], bool); hard_mask[hard] = True
    groups = {'all_nodes': np.ones(y.shape, bool), 'fixed_hard_windows': np.broadcast_to(hard_mask[:, None, None], y.shape),
        'direct_report_nodes': np.broadcast_to(archive['direct_support'][:, None], y.shape),
        'outside_direct_support': np.broadcast_to(~archive['direct_support'][:, None], y.shape)}
    edges = (0., 5., 20., 50., 100., 200., float('inf'))
    for lo, hi in zip(edges[:-1], edges[1:]):
        groups[f'flow:[{lo:g},{hi:g})'] = (y >= lo) & (y < hi)
    for name in np.unique(archive['road_labels']):
        groups['road:'+str(name)] = np.broadcast_to((archive['road_labels'] == name)[None, None], y.shape)
    cases, horizons, comparisons, groups_table = {}, [], {}, []
    for name, prediction in predictions.items():
        metrics = point_metrics(prediction, y, valid)
        score, complete = window_scores(prediction, y, valid)
        metrics['window_p95'] = float(np.quantile(score[complete], .95)) if complete.any() else None
        cases[name] = metrics
        horizons += [dict(case=name, **row) for row in metrics['horizons']]
    for name, control, candidate in COMPARISONS:
        pair = paired_metrics(predictions[control], predictions[candidate], y, valid)
        scores_a, complete = window_scores(predictions[control], y, valid)
        scores_b, _ = window_scores(predictions[candidate], y, valid)
        pair['fixed_hard_window_equal_weight_gain'] = float((scores_a[hard]-scores_b[hard]).mean()) if len(hard) else None
        comparisons[name] = pair
        for group, mask in groups.items():
            result = paired_metrics(predictions[control], predictions[candidate], y, valid & mask, global_count)
            groups_table.append(dict(comparison=name, group=group,
                **{key: result[key] for key in ('valid_count', 'mae_gain_pooled', 'mae_gain_macro',
                    'rmse_gain_pooled', 'mape_gain', 'global_mae_contribution')}))
    analysis = dict(schema='original_igstgnn_graph_message_evaluation_v1', status='GW2_EVALUATION_PASS',
        validation_rows=len(y), engineering_only=bool(metadata.get('engineering_only', True)),
        cases=cases, comparisons=comparisons, fixed_hard_ids=np.asarray(archive['sample_ids'])[hard].tolist(),
        producer_metadata=metadata, optimizer_updates=0, test_evaluated=False,
        inference_calls_this_analysis=0, mask_shared=True,
        caveat='overlapping_validation_rows_not_independent_incidents_and_used_for_design_selection')
    write_json(output/'analysis.json', analysis)
    _write_csv(output/'cases.csv', [dict(case=name, **{k: v for k, v in row.items() if k != 'horizons'}) for name, row in cases.items()])
    _write_csv(output/'horizons.csv', horizons)
    _write_csv(output/'groups.csv', groups_table)
    _write_csv(output/'comparisons.csv', [dict(comparison=name, **{k: v for k, v in row.items()
        if k not in ('horizons', 'directions')}) for name, row in comparisons.items()])
    lines = ['# GW2：图消息感知窗口聚合', '',
        f"工程检查：{analysis['engineering_only']}；验证窗口：{len(y)}。", '',
        '训练设计要求四组使用相同原模型初始化、损失、优化器、课程与完成预算；实际身份与回放证据见 producer_metadata。',
        '主比较为 POST_MESSAGE vs ORIGINAL；收益=对照误差−候选误差。验证结果不等于原论文测试集复现。', '',
        '必须同时比较 POST_MESSAGE vs STATIC 和 POST_MESSAGE vs POST_LOCAL；两个动态组共享评分器容量与接收端选择位置。', '',
        '| 状态 | MAE pooled | MAE macro | RMSE pooled | MAPE % | 窗口 P95 |',
        '|---|---:|---:|---:|---:|---:|']
    fmt = lambda value: '不可用' if value is None else f'{value:.9f}'
    for name, row in cases.items():
        lines.append('| '+name+' | '+' | '.join(fmt(row[key]) if key != 'mape_pooled' else fmt(100*row[key])
            for key in ('mae_pooled', 'mae_macro', 'rmse_pooled', 'mape_pooled', 'window_p95'))+' |')
    lines += ['', '| 比较 | MAE pooled 收益 | RMSE 收益 | MAPE 收益百分点 | 固定困难窗口收益 |', '|---|---:|---:|---:|---:|']
    for name, row in comparisons.items():
        lines.append('| '+name+' | '+' | '.join(fmt(value) for value in (row['mae_gain_pooled'], row['rmse_gain_pooled'],
            100*row['mape_gain'], row['fixed_hard_window_equal_weight_gain']))+' |')
    lines += ['', '## 推理成本', '',
        '相同首批量、先热身两次后测量三次；CUDA 同步完整前向，不含 H2D、反归一化、统计与哈希。',
        '只描述该设备首批量的推理，不能当作训练速度或单样本延迟。训练阶段总耗时见训练 summary.json。', '',
        '| 训练组 | 批量 | 前向均值 ms | 峰值 allocated bytes |', '|---|---:|---:|---:|']
    for arm, cost in metadata.get('benchmarks', {}).items():
        lines.append(f"| {arm} | {cost['batch_size']} | {fmt(cost['mean_milliseconds'])} | {cost['peak_allocated_bytes']} |")
    lines += ['', '## 解释边界', '',
        '各 UNIFORM 状态保留其各自联合训练后的主干；STATIC_ONLY 关闭 POST_MESSAGE 动态评分，不替代独立重训组。',
        'STATIC_ONLY 仅保留显式 static_logits；不保留内容评分的训练均值，不是纯跨样本变化贡献。',
        'POST_LOCAL 读取原变换后的接收节点槽位；POST_MESSAGE 读取一阶图消息，二者都读取槽位值与窗口内中心化差异。',
        '所有内容继承原 ICSF、状态/时钟门控；没有独立新增事故或未来标签输入。',
        '机制统计按历史/自回归调用阶段分开，并固定节点和局部窗口位置统计跨样本变化；方差仍不等于预测效用。',
        '系数和固定为 K 不保证输出均值、能量或车辆守恒。先传播再按槽位汇聚改变归约顺序，零初始化仅承诺容差恢复。',
        '槽位已经过原线性层混合，权重不代表物理传播时延。高阶图共享同一来源的选择，self 路径不加权。',
        '固定困难集由本次 ORIGINAL 最差 10% 完整窗口选定，不随候选更换；跨历史筛查困难集并不自动相同。',
        '逐道路、流量及事故支持分组见 groups.csv；全网贡献分母固定，重叠分组不可相加。',
        '五轮仅作结构试跑；尚需相同完整预算、多种子与冻结方案后的独立测试验证。',
        '原发布数组包含测试载荷，加载时可能反序列化；本流程没有构造或评价测试批次。', '']
    if not metadata.get('validation_replay'):
        lines += ['本次仅重新分析预测归档，未独立核验生产者训练身份、预算或检查点回放。', '']
    (output/'report.md').write_text('\n'.join(lines), encoding='utf-8')
    return analysis


def benchmark(model, batch, device):
    x, _, incident, sensors = batch
    cuda = torch.device(device).type == 'cuda'
    records = []
    with torch.no_grad():
        for _ in range(2):
            model(x, incident_data=incident, sensor_data=sensors)
        if cuda:
            torch.cuda.synchronize(device); torch.cuda.reset_peak_memory_stats(device)
        for _ in range(3):
            if cuda:
                torch.cuda.synchronize(device)
            start = perf_counter()
            model(x, incident_data=incident, sensor_data=sensors)
            if cuda:
                torch.cuda.synchronize(device)
            records.append(1000*(perf_counter()-start))
    return dict(batch_size=len(x), warmup_calls=2, measured_calls=3, milliseconds=records,
        mean_milliseconds=float(np.mean(records)),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if cuda else None,
        scope='synchronized_complete_forward_same_first_batch_excludes_H2D_inverse_stats_hash')


def replay_metrics(prediction, truth, state, epoch, modules):
    actual = native_metrics(prediction, truth, modules)
    expected = {key: state['history'][epoch-1]['validation_'+key] for key in ('mae', 'rmse', 'mape_fraction')}
    passed = all(np.isclose(actual[key], value, atol=1e-5, rtol=1e-6) for key, value in expected.items())
    return dict(actual=actual, expected=expected, passed=bool(passed), atol=1e-5, rtol=1e-6)


def produce(args, output):
    identity = json.loads((args.run_dir/'identity.json').read_text(encoding='utf-8'))
    require(identity.get('schema') == TRAIN_SCHEMA and identity.get('design') == DESIGN
            and identity.get('arms') == list(ARMS), 'GW2 training identity required')
    for name in EXECUTION_FILES:
        require(identity['execution_source'].get(name) == sha256(REPO/name), 'Training source changed: '+name)
    modules = get_original_modules()
    require(identity['source_fingerprint'] == modules.source_fingerprint, 'Native source identity differs')
    device = args.device or identity['runtime']['python_device']
    runtime = configure_runtime(device, identity['runtime']['threads'], identity['training_config']['seed'])
    data = load_data(args.data_dir, identity['training_config'], modules, logging.getLogger('gw2-eval'),
        check=identity['engineering_only'], training=True)
    require(data.identity == identity['data'], 'Validation data identity differs from training')
    states = {arm: load_checkpoint(args.run_dir/arm/'last.pt') for arm in ARMS}
    completed = {state['completed_epoch'] for state in states.values()}
    budget = json.loads((args.run_dir/'budget.json').read_text())['target_completed_epochs']
    require(len(completed) == 1 and min(completed) >= budget, 'Finish all four arms to a common completed budget')
    for arm, state in states.items():
        validate_saved(state, identity, arm, args.run_dir/arm, data.loaders['train_loader'].num_batch)
    loader = data.loaders['val_loader']
    road_labels = np.full(len(data.station_ids), 'unknown', dtype='<U40')
    for road, indices in data.roads.items():
        road_labels[indices] = road
    archive = dict(schema=np.array(SCHEMA), axis_order=np.array('sample,horizon,node'),
        sample_ids=np.asarray(identity['data']['splits']['val']['selected_indices'], dtype=np.int64),
        node_ids=data.station_ids, road_labels=road_labels)
    replay, costs, roles, mechanisms = {}, {}, {}, {}
    for arm, state in states.items():
        model = model_for_arm(data, modules, device, arm)
        if args.checkpoint == 'best':
            path = args.run_dir/arm/state['best_checkpoint_file']
            require(sha256(path) == state['best_checkpoint_sha256'], 'Best file changed')
            weights, epoch = load_checkpoint(path), state['best_epoch']
        else:
            path, weights, epoch = args.run_dir/arm/'last.pt', state['model_state'], state['completed_epoch']
        model.load_state_dict(weights, strict=True); model.eval()
        roles[arm] = dict(selection=args.checkpoint, epoch=epoch, checkpoint_sha256=sha256(path))
        costs[arm] = benchmark(model, device_batch(next(loader.get_iterator()), device), device)
        conditions = [(arm, None)]
        if arm in ('STATIC', 'POST_LOCAL'):
            conditions += [(arm+'_UNIFORM', 'UNIFORM')]
        if arm == 'POST_MESSAGE':
            conditions += [('POST_MESSAGE_UNIFORM', 'UNIFORM'), ('POST_MESSAGE_STATIC_ONLY', 'STATIC')]
        for case, setting in conditions:
            predictions, truths, supports = [], [], []
            if hasattr(model, 'start_diagnostics'):
                model.start_diagnostics()
            context = model.intervention(setting) if setting else nullcontext()
            with torch.no_grad(), context:
                for index, raw in enumerate(loader.get_iterator(), 1):
                    x, y, report_data, sensors = device_batch(raw, device)
                    p = model(x, incident_data=report_data, sensor_data=sensors)
                    predictions.append(data.scaler.inverse_transform(p).squeeze(-1).cpu().numpy())
                    truths.append(data.scaler.inverse_transform(y).squeeze(-1).cpu().numpy())
                    supports.append(report_data['distances'].abs().sum(-1).cpu().numpy() > 0)
                    if index == 1 or index % 10 == 0 or index == loader.num_batch:
                        print(json.dumps(dict(case=case, batch=index, total=loader.num_batch)), flush=True)
            prediction, truth, support = np.concatenate(predictions), np.concatenate(truths), np.concatenate(supports)
            if 'y' in archive:
                require(np.array_equal(archive['y'], truth) and np.array_equal(archive['direct_support'], support),
                    'Case target/support/order differs')
            else:
                archive.update(y=truth, direct_support=support)
            archive['pred__'+case] = prediction
            mechanisms[case] = model.diagnostics() if hasattr(model, 'diagnostics') else {}
            if setting is None:
                replay[arm] = replay_metrics(prediction, truth, state, epoch, modules)
        del model, weights
        gc.collect()
        if torch.device(device).type == 'cuda':
            torch.cuda.empty_cache()
    null = float(archive['y'].min()) if archive['y'].min() < 1 else 0.
    archive['valid'] = archive['y'] != null
    validate_archive(archive)
    require(all(row['passed'] for row in replay.values()), 'Saved native validation replay failed')
    metadata = dict(schema='original_igstgnn_graph_message_producer_v1', status='GW2_PREDICTION_PASS',
        engineering_only=identity['engineering_only'], run_identity_sha256=identity['identity_sha256'],
        common_completed_epochs=min(completed), checkpoint_roles=roles, runtime=runtime,
        validation_replay=replay, benchmarks=costs, source_fingerprint=modules.source_fingerprint,
        test_evaluated=False, optimizer_updates=0, design=DESIGN,
        producer_source={name: sha256(REPO/'experiments/chronological'/name) for name in (
            'evaluate_original_graph_message.py', 'evaluate_unified_adapter_v03.py')})
    path = output/'paired_predictions.npz'
    np.savez_compressed(path, **archive)
    metadata['archive_sha256'] = sha256(path)
    write_json(output/'metadata.json', metadata)
    write_json(output/'mechanisms.json', mechanisms)
    return archive, metadata


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--run-dir', type=Path)
    source.add_argument('--archive', type=Path)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--checkpoint', choices=('best', 'last'), default='best')
    p.add_argument('--data-dir', type=Path, default=default_data_dir())
    p.add_argument('--device')
    args = p.parse_args(argv)
    require(not args.output_dir.exists(), 'Use a fresh evaluation output directory')
    args.output_dir.mkdir(parents=True)
    try:
        if args.archive:
            before = sha256(args.archive)
            with np.load(args.archive, allow_pickle=False) as saved:
                archive = dict(saved)
            metadata = dict(engineering_only=True, source_archive_sha256=before,
                producer_identity_independently_verified=False, new_inference_calls=0, test_evaluated=False)
            result = report(archive, args.output_dir, metadata)
            require(sha256(args.archive) == before, 'Source archive changed')
        else:
            archive, metadata = produce(args, args.output_dir)
            result = report(archive, args.output_dir, metadata)
        print(json.dumps(dict(status=result['status'], validation_rows=result['validation_rows'],
            engineering_only=result['engineering_only'], output_dir=str(args.output_dir.resolve()))), flush=True)
        return result
    except BaseException as error:
        write_json(args.output_dir/'failure.json', dict(status='GW2_EVALUATION_FAILED',
            error_type=type(error).__name__, error=str(error), optimizer_updates=0, test_evaluated=False))
        raise


if __name__ == '__main__':
    main()
