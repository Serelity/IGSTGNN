"""GW4 paired validation; BASE is the cached GW2 BEST, never a new forward.

Only new residual checkpoints/interventions are evaluated. Identical model
tensors reuse a completed GW4 prediction cache across BEST/LAST wrappers.
"""
import argparse
from contextlib import nullcontext
import gc
import json
import logging
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import numpy as np
import torch

from experiments.chronological import train_original_receiver_residual as train
from experiments.chronological import evaluate_original_graph_message as metrics
from experiments.chronological.graph_message_reuse import RUNTIME_KEYS
from experiments.chronological.receiver_residual_reference import (
    compatible, load_cached, read_json, recertify, require, verify_identity_digest)


SCHEMA = 'original_igstgnn_receiver_residual_archive_v1'
CASES = ('ORIGINAL', 'GW2_BASE', *train.ARMS, 'MESSAGE_RESIDUAL_STATIC_ONLY',
    'RECEIVER_RESIDUAL_STATIC_ONLY', 'RECEIVER_RESIDUAL_QUERY_SELF')
COMPARISONS = (
    *((arm+'_vs_GW2_BASE', 'GW2_BASE', arm) for arm in train.ARMS),
    *((b+'_vs_'+a, a, b) for a, b in train.PAIRS),
    ('message_dynamic_dependency', 'MESSAGE_RESIDUAL_STATIC_ONLY', 'MESSAGE_RESIDUAL'),
    ('receiver_dynamic_dependency', 'RECEIVER_RESIDUAL_STATIC_ONLY', 'RECEIVER_RESIDUAL'),
    ('receiver_query_dependency', 'RECEIVER_RESIDUAL_QUERY_SELF', 'RECEIVER_RESIDUAL'))
PRODUCERS = ('evaluate_original_receiver_residual.py', 'receiver_residual_reference.py',
    'evaluate_original_graph_message.py', 'evaluate_unified_adapter_v03.py')


def conditions(arm):
    result = [(arm, None)]
    if arm != 'STATIC_RESIDUAL':
        result.append((arm+'_STATIC_ONLY', 'STATIC_ONLY'))
    if arm == 'RECEIVER_RESIDUAL':
        result.append((arm+'_QUERY_SELF', 'QUERY_SELF'))
    return result


def validate_archive(archive):
    return metrics.validate_archive(archive, CASES, SCHEMA)


def selection(directory, state, checkpoint):
    if checkpoint == 'best':
        path = directory/state['best_checkpoint_file']
        require(metrics.sha256(path) == state['best_checkpoint_sha256'], 'GW4 BEST digest differs')
        weights, epoch = train.base.load_checkpoint(path), state['best_epoch']
    else:
        path, weights, epoch = directory/'last.pt', state['model_state'], state['completed_epoch']
    return weights, dict(selection=checkpoint, epoch=epoch, checkpoint_sha256=metrics.sha256(path),
        model_tensor_sha256=train.base.tensor_tree_hash(weights))


def existing_cache(run, identity, arm, role, shared):
    for path in sorted(run.glob('*/metadata.json'), reverse=True):
        meta = read_json(path)
        if meta.get('schema') != 'gw4_receiver_residual_producer_v1' or meta.get('status') != 'GW4_PREDICTION_PASS':
            continue
        if meta.get('run_identity_sha256') != identity['identity_sha256'] or meta.get('reference') != identity['reference']:
            continue
        previous = meta.get('checkpoint_roles', {}).get(arm, {})
        if previous.get('model_tensor_sha256') != role['model_tensor_sha256']:
            continue
        if meta.get('validation_replay', {}).get(arm, {}).get('passed') is not True:
            continue
        for name in PRODUCERS:
            require(meta.get('producer_source', {}).get(name) == metrics.sha256(REPO/'experiments/chronological'/name),
                    'GW4 cached producer changed: '+name)
        archive_path = path.parent/'paired_predictions.npz'
        require(metrics.sha256(archive_path) == meta.get('archive_sha256'), 'GW4 cached archive digest differs')
        with np.load(archive_path, allow_pickle=False) as payload:
            archive = dict(payload)
        validate_archive(archive)
        for key in ('y', 'valid', 'sample_ids', 'node_ids', 'road_labels', 'direct_support', 'pred__GW2_BASE', 'pred__ORIGINAL'):
            require(np.array_equal(archive[key], shared[key]), 'GW4 cached axes/anchor differ: '+key)
        require(metrics.sha256(archive_path) == meta['archive_sha256'], 'GW4 cache changed while loading')
        return archive, meta, path.parent
    return None


def produce(args, output):
    identity = read_json(args.run_dir/'identity.json')
    require(identity.get('schema') == train.SCHEMA and identity.get('design') == train.DESIGN,
            'GW4 adaptation identity required')
    verify_identity_digest(identity)
    for name in train.EXECUTION_FILES:
        require(identity['execution_source'].get(name) == metrics.sha256(REPO/name), 'GW4 execution source changed: '+name)
    reference = read_json(args.run_dir/'reference.json')
    require(reference == identity['reference'], 'GW4 reference file differs from its identity')
    source_identity, source_weights = recertify(reference)
    compatible(source_identity, identity)
    epochs = read_json(args.run_dir/'budget.json')['target_completed_epochs']
    states = {}
    for arm in train.ARMS:
        directory = args.run_dir/arm
        before = metrics.sha256(directory/'last.pt')
        state = train.base.load_checkpoint(directory/'last.pt')
        train.validate_saved(state, identity, arm, directory, len(identity['data']['splits']['train']['partition']))
        require(state['completed_epoch'] == epochs, 'Complete all GW4 arms at the common adaptation budget')
        require(metrics.sha256(directory/'last.pt') == before, 'GW4 checkpoint changed while loading')
        states[arm] = state
    modules = metrics.get_original_modules()
    require(modules.source_fingerprint == identity['source_fingerprint'], 'Frozen author source differs')
    device = args.device or identity['runtime']['python_device']
    runtime = metrics.configure_runtime(device, identity['runtime']['threads'], identity['training_config']['seed'])
    for key in RUNTIME_KEYS:
        require(runtime.get(key) == identity['runtime'].get(key), 'GW4 evaluation runtime differs: '+key)
    data = metrics.load_data(args.data_dir, identity['training_config'], modules,
        logging.getLogger('gw4-eval'), check=False, training=True)
    require(data.identity == identity['data'], 'Actual GW4 evaluation data differs')
    cached, source_meta = load_cached(reference)
    require(np.array_equal(cached['sample_ids'], identity['data']['splits']['val']['selected_indices']),
            'GW2 validation sample order differs')
    require(np.array_equal(cached['node_ids'], data.station_ids), 'GW2 node order differs')
    roads = np.full(len(data.station_ids), 'unknown', dtype='<U40')
    for road, indices in data.roads.items():
        roads[indices] = road
    require(np.array_equal(cached['road_labels'], roads), 'GW2 road labels differ')
    archive = {key: cached[key] for key in ('y', 'valid', 'sample_ids', 'node_ids', 'road_labels', 'direct_support')}
    archive.update(schema=np.array(SCHEMA), axis_order=np.array('sample,horizon,node'),
        pred__ORIGINAL=cached['pred__ORIGINAL'], pred__GW2_BASE=cached['pred__POST_MESSAGE'])
    actual = metrics.native_metrics(archive['pred__GW2_BASE'], archive['y'], modules)
    expected = reference['source_best_metrics']
    anchor_passed = all(np.isclose(actual[key], expected[key], atol=1e-5, rtol=1e-6) for key in expected)
    require(anchor_passed, 'Cached GW2 BEST metrics differ from source history')
    replay = dict(GW2_BASE=dict(actual=actual, expected=expected, passed=bool(anchor_passed),
        mode='cached_GW2_BEST_no_new_forward', new_inference_calls=0),
        ORIGINAL=dict(source_meta['validation_replay']['ORIGINAL'], mode='certified_historical_cache'))
    roles = dict(GW2_BASE=dict(selection='best', epoch=reference['source_best_epoch'],
        checkpoint_sha256=reference['source_checkpoint_sha256'],
        model_tensor_sha256=reference['source_model_tensor_sha256'], reused=True),
        ORIGINAL=dict(source_meta['checkpoint_roles']['ORIGINAL'], reused=True))
    mechanisms, calls = {}, dict(ORIGINAL=0, GW2_BASE=0)
    for arm, state in states.items():
        weights, role = selection(args.run_dir/arm, state, args.checkpoint)
        roles[arm] = role
        prior = existing_cache(args.run_dir, identity, arm, role, archive)
        if prior:
            previous, previous_meta, directory = prior
            for case, _ in conditions(arm):
                archive['pred__'+case] = previous['pred__'+case]
                mechanisms[case] = dict(reused=True, source_directory=str(directory))
            replay[arm] = dict(metrics.replay_metrics(archive['pred__'+arm], archive['y'], state, role['epoch'], modules),
                mode='cached_identical_GW4_model_tensors_no_new_forward')
            require(replay[arm]['passed'], 'GW4 cached validation metrics differ')
            roles[arm]['prediction_source_role'] = previous_meta['checkpoint_roles'][arm]
            calls[arm] = 0
            del weights
            continue
        model = train.model_for_arm(data, modules, device, arm, source_weights)
        model.load_state_dict(weights, strict=True)
        train.check_frozen(model.state_dict(), identity)
        model.eval()
        calls[arm] = 0
        for case, setting in conditions(arm):
            predictions, truths, supports = [], [], []
            model.start_diagnostics()
            context = model.intervention(setting) if setting else nullcontext()
            with torch.no_grad(), context:
                for index, raw in enumerate(data.loaders['val_loader'].get_iterator(), 1):
                    x, y, incident, sensors = metrics.device_batch(raw, device)
                    prediction = model(x, incident_data=incident, sensor_data=sensors)
                    calls[arm] += 1
                    predictions.append(data.scaler.inverse_transform(prediction).squeeze(-1).cpu().numpy())
                    truths.append(data.scaler.inverse_transform(y).squeeze(-1).cpu().numpy())
                    supports.append(incident['distances'].abs().sum(-1).cpu().numpy() > 0)
                    if index == 1 or index % 10 == 0 or index == data.loaders['val_loader'].num_batch:
                        print(dict(case=case, batch=index, total=data.loaders['val_loader'].num_batch), flush=True)
            prediction, truth, support = map(np.concatenate, (predictions, truths, supports))
            require(np.array_equal(truth, archive['y']) and np.array_equal(support, archive['direct_support']),
                    'Actual GW4 labels/support differ from the cached reference')
            archive['pred__'+case] = prediction
            mechanisms[case] = model.diagnostics()
            if setting is None:
                replay[arm] = dict(metrics.replay_metrics(prediction, truth, state, role['epoch'], modules), mode='new_GW4_forward')
                require(replay[arm]['passed'], 'GW4 native validation replay failed: '+arm)
        train.check_frozen(model.state_dict(), identity)
        del model, weights
        gc.collect()
        if torch.device(device).type == 'cuda':
            torch.cuda.empty_cache()
    validate_archive(archive)
    recertify(reference)
    metadata = dict(schema='gw4_receiver_residual_producer_v1', status='GW4_PREDICTION_PASS',
        engineering_only=False, run_identity_sha256=identity['identity_sha256'],
        common_completed_adaptation_epochs=epochs, reference=reference,
        checkpoint_roles=roles, validation_replay=replay, runtime=runtime,
        new_inference_calls=calls, baseline_new_inference_calls=0, baseline_new_training_updates=0,
        optimizer_updates=0, old_parameters_frozen=True, test_evaluated=False,
        inference_benchmark_performed=False,
        producer_source={name: metrics.sha256(REPO/'experiments/chronological'/name) for name in PRODUCERS})
    path = output/'paired_predictions.npz'
    np.savez_compressed(path, **archive)
    metadata['archive_sha256'] = metrics.sha256(path)
    metrics.write_json(output/'metadata.json', metadata)
    metrics.write_json(output/'mechanisms.json', mechanisms)
    return archive, metadata


def report(archive, output, metadata):
    analysis = metrics.report(archive, output, metadata, CASES, COMPARISONS, SCHEMA)
    y, valid, predictions = validate_archive(archive)
    hard = metrics.fixed_hard_indices(predictions['GW2_BASE'], y, valid, archive['sample_ids'])
    extra = []
    for name, control, candidate in COMPARISONS:
        a, _ = metrics.window_scores(predictions[control], y, valid)
        b, _ = metrics.window_scores(predictions[candidate], y, valid)
        gain = float((a[hard]-b[hard]).mean()) if len(hard) else None
        analysis['comparisons'][name]['GW2_fixed_hard_window_gain'] = gain
        extra.append(dict(comparison=name, GW2_fixed_hard_window_gain=gain,
            mean_prediction_change=float((predictions[candidate]-predictions[control])[valid].mean())))
    analysis.update(schema='original_igstgnn_receiver_residual_evaluation_v1', status='GW4_EVALUATION_COMPLETE',
        GW2_fixed_hard_ids=archive['sample_ids'][hard].tolist(),
        budget_caveat='GW2_source_budget_plus_extra_residual_adaptation_not_equal_from_scratch_budget')
    metrics.write_json(output/'analysis.json', analysis)
    metrics._write_csv(output/'gw2_hard_windows_and_shift.csv', extra)
    lines = ['# GW4：冻结 GW2 的接收条件消息修正', '',
        f"验证窗口：{len(y)}；仅归档重分析：{bool(metadata.get('engineering_only', True))}。", '',
        'GW2_BASE 始终复用源 POST_MESSAGE 的 BEST 预测，即使本次选择适配 LAST；没有新增旧基线训练或推理。',
        'ORIGINAL 是历史参考；三个新组均付出额外适配预算，不能将对旧模型的增益解释为等预算从头训练优势。',
        '主结构比较 RECEIVER_RESIDUAL vs MESSAGE_RESIDUAL；二者活跃容量相同。STATIC_RESIDUAL 只有来源×槽位偏置活跃。',
        '旧参数冻结且旧模块保持 eval；仍允许梯度穿过冻结的后续算子。适配 BEST 不包含第零步锚点。', '',
        '| 状态 | MAE pooled | RMSE pooled | MAPE % | 窗口 P95 |', '|---|---:|---:|---:|---:|']
    fmt = lambda value: '不可用' if value is None else f'{value:.9f}'
    for name, row in analysis['cases'].items():
        lines.append('| '+name+' | '+' | '.join(fmt(value) for value in (
            row['mae_pooled'], row['rmse_pooled'], 100*row['mape_pooled'], row['window_p95']))+' |')
    lines += ['', '| 比较 | MAE 收益 | RMSE 收益 | MAPE 收益百分点 | GW2 固定困难窗口收益 |', '|---|---:|---:|---:|---:|']
    for name, row in analysis['comparisons'].items():
        lines.append('| '+name+' | '+' | '.join(fmt(value) for value in (row['mae_gain_pooled'],
            row['rmse_gain_pooled'], 100*row['mape_gain'], row['GW2_fixed_hard_window_gain']))+' |')
    lines += ['', '收益=对照误差−候选误差。gw2_hard_windows_and_shift.csv 另列预测平均变化；不要用该变化在验证集上拟合补偿。',
        'analysis.json 的原 fixed_hard_ids / groups.csv 固定困难组沿用历史 ORIGINAL；GW2_fixed_hard_ids 与上表使用固定 GW2_BASE 最差 10% 窗口。',
        'STATIC_ONLY 只关闭新增点积，保留已训练新增偏置及完整旧 GW2 评分；QUERY_SELF 只把新增接收查询替换成消息均值。',
        '这些同检查点干预可能分布外，不替代独立训练对照。没有执行 BASE 干预前向，BASE 直接读取认证历史缓存。',
        '权重、熵、查询/键范数用于诊断，不证明预测效用、物理时延或因果传播；潜在槽位已经跨原时间窗口混合。',
        '5 轮为适配筛查；后续需固定方案后做共同完整预算、配对多种子和独立确认。真实未来流量分组仅为事后诊断。',
        '没有新增推理计时；训练/验证完整阶段的时间和峰值显存保存在训练 summary.json。重叠窗口不是独立事故重复。',
        '未创建或评价测试 loader；原发布对象数组加载时可能反序列化包含测试的载荷。', '']
    if metadata.get('engineering_only', True):
        lines += ['本次仅重分析给定归档，未独立重新认证训练身份、源缓存和检查点回放。', '']
    (output/'report.md').write_text('\n'.join(lines), encoding='utf-8')
    return analysis


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--run-dir', type=Path)
    source.add_argument('--archive', type=Path)
    parser.add_argument('--data-dir', type=Path, default=metrics.default_data_dir())
    parser.add_argument('--checkpoint', choices=('best', 'last'), default='best')
    parser.add_argument('--device')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    args.output_dir = args.output_dir.resolve()
    if args.run_dir:
        args.run_dir = args.run_dir.resolve()
    require(not args.output_dir.exists(), 'Use a fresh GW4 evaluation directory')
    args.output_dir.mkdir(parents=True)
    try:
        if args.archive:
            before = metrics.sha256(args.archive)
            with np.load(args.archive, allow_pickle=False) as payload:
                archive = dict(payload)
            metadata = dict(engineering_only=True, source_archive_sha256=before,
                new_inference_calls=0, optimizer_updates=0, test_evaluated=False)
            result = report(archive, args.output_dir, metadata)
            require(metrics.sha256(args.archive) == before, 'Source archive changed during reanalysis')
        else:
            archive, metadata = produce(args, args.output_dir)
            result = report(archive, args.output_dir, metadata)
        print(json.dumps(dict(status=result['status'], validation_rows=result['validation_rows'],
            output_dir=str(args.output_dir))), flush=True)
        return result
    except BaseException as error:
        metrics.write_json(args.output_dir/'failure.json', dict(status='GW4_EVALUATION_FAILED',
            error_type=type(error).__name__, error=str(error), optimizer_updates=0, test_evaluated=False))
        raise


if __name__ == '__main__':
    main()
