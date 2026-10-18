"""GW5 paired validation: certified GW4/GW2 controls are never rerun."""
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

from experiments.chronological import train_original_history_anchor as train
from experiments.chronological import evaluate_original_graph_message as metrics
from experiments.chronological.graph_message_reuse import RUNTIME_KEYS
from experiments.chronological.history_anchor_reference import (
    compatible, load_cached, read_json, recertify, require, verify_identity_digest)


SCHEMA = 'original_igstgnn_history_anchor_archive_v1'
REUSED_CASES = ('ORIGINAL', 'GW2_BASE', *train.gw4.ARMS)
CASES = (*REUSED_CASES, *train.ARMS, *(arm+'_FREE' for arm in train.ARMS), 'HISTORY_ANCHORED_ANCHOR_ROLL')
COMPARISONS = (
    *((arm+'_vs_RECEIVER_RESIDUAL', 'RECEIVER_RESIDUAL', arm) for arm in train.ARMS),
    *((arm+'_vs_GW2_BASE', 'GW2_BASE', arm) for arm in train.ARMS),
    *((b+'_vs_'+a, a, b) for a, b in train.PAIRS),
    ('HISTORY_ANCHORED_vs_STATIC_RESIDUAL', 'STATIC_RESIDUAL', 'HISTORY_ANCHORED'),
    ('HISTORY_ANCHORED_vs_MESSAGE_RESIDUAL', 'MESSAGE_RESIDUAL', 'HISTORY_ANCHORED'),
    *((arm+'_policy_dependency', arm+'_FREE', arm) for arm in train.ARMS),
    ('history_anchor_node_identity_sensitivity', 'HISTORY_ANCHORED_ANCHOR_ROLL', 'HISTORY_ANCHORED'))
PRODUCERS = ('evaluate_original_history_anchor.py', 'history_anchor_reference.py',
    'evaluate_original_graph_message.py', 'evaluate_unified_adapter_v03.py')


def conditions(arm):
    return [(arm, None), (arm+'_FREE', 'FREE')] + (
        [('HISTORY_ANCHORED_ANCHOR_ROLL', 'ANCHOR_ROLL')] if arm == 'HISTORY_ANCHORED' else [])


def validate_archive(archive):
    return metrics.validate_archive(archive, CASES, SCHEMA)


def selection(directory, state, checkpoint, arm):
    if checkpoint == 'best':
        path = directory/state['best_checkpoint_file']
        require(metrics.sha256(path) == state['best_checkpoint_sha256'], 'GW5 BEST digest differs')
        weights, epoch = train.base.load_checkpoint(path), state['best_epoch']
    else:
        path, weights, epoch = directory/'last.pt', state['model_state'], state['completed_epoch']
    return weights, dict(selection=checkpoint, epoch=epoch, policy=arm,
        checkpoint_sha256=metrics.sha256(path), model_tensor_sha256=train.base.tensor_tree_hash(weights))


def existing_cache(run, identity, arm, role, shared):
    for path in sorted(run.glob('*/metadata.json'), reverse=True):
        meta = read_json(path)
        if meta.get('schema') != 'gw5_history_anchor_producer_v1' or meta.get('status') != 'GW5_PREDICTION_PASS':
            continue
        if meta.get('run_identity_sha256') != identity['identity_sha256'] or meta.get('gw4_reference') != identity['gw4_reference']:
            continue
        previous = meta.get('checkpoint_roles', {}).get(arm, {})
        if previous.get('policy') != arm or previous.get('model_tensor_sha256') != role['model_tensor_sha256']:
            continue
        if meta.get('validation_replay', {}).get(arm, {}).get('passed') is not True:
            continue
        for name in PRODUCERS:
            require(meta.get('producer_source', {}).get(name) == metrics.sha256(REPO/'experiments/chronological'/name),
                    'GW5 cache producer changed: '+name)
        archive_path = path.parent/'paired_predictions.npz'
        require(metrics.sha256(archive_path) == meta.get('archive_sha256'), 'GW5 cached archive digest differs')
        with np.load(archive_path, allow_pickle=False) as payload:
            archive = dict(payload)
        validate_archive(archive)
        # A GW4 BEST/LAST control can change while this new model stays
        # identical. Its cached prediction remains valid on the same axes and
        # frozen GW2 anchor; copy only the new-model cases below.
        keys = ('y', 'valid', 'sample_ids', 'node_ids', 'road_labels', 'direct_support',
                'pred__ORIGINAL', 'pred__GW2_BASE')
        require(all(np.array_equal(archive[key], shared[key]) for key in keys), 'GW5 cached axes or controls differ')
        require(metrics.sha256(archive_path) == meta['archive_sha256'], 'GW5 cache changed during read')
        return archive, meta, path.parent
    return None


def produce(args, output):
    identity = read_json(args.run_dir/'identity.json')
    verify_identity_digest(identity)
    require(identity.get('schema') == train.SCHEMA and identity.get('design') == train.DESIGN, 'GW5 identity required')
    for name in train.EXECUTION_FILES:
        require(identity['execution_source'].get(name) == metrics.sha256(REPO/name), 'GW5 execution source changed: '+name)
    reference = read_json(args.run_dir/'gw4_reference.json')
    require(reference == identity['gw4_reference'], 'GW5 reference differs from its identity')
    old_identity, source_weights, old_states = recertify(reference)
    compatible(old_identity, identity)
    epochs = read_json(args.run_dir/'budget.json')['target_completed_epochs']
    require(epochs == reference['completed_adaptation_epochs'] == 5, 'Pair GW5 and GW4 at their common five-epoch adaptation budget')
    states = {}
    for arm in train.ARMS:
        directory = args.run_dir/arm
        digest = metrics.sha256(directory/'last.pt')
        state = train.base.load_checkpoint(directory/'last.pt')
        train.validate_saved(state, identity, arm, directory, len(identity['data']['splits']['train']['partition']))
        require(state['completed_epoch'] == epochs, 'Complete each GW5 policy at the common budget')
        require(metrics.sha256(directory/'last.pt') == digest, 'GW5 checkpoint changed during read')
        states[arm] = state
    modules = metrics.get_original_modules()
    require(modules.source_fingerprint == identity['source_fingerprint'], 'Frozen native source differs')
    device = args.device or identity['runtime']['python_device']
    runtime = metrics.configure_runtime(device, identity['runtime']['threads'], identity['training_config']['seed'])
    for key in RUNTIME_KEYS:
        require(runtime.get(key) == identity['runtime'].get(key), 'GW5 evaluation runtime differs: '+key)
    data = metrics.load_data(args.data_dir, identity['training_config'], modules,
        logging.getLogger('gw5-eval'), check=False, training=True)
    require(data.identity == identity['data'], 'Actual GW5 evaluation data differs')
    cached, source_meta = load_cached(reference, args.checkpoint)
    require(np.array_equal(cached['sample_ids'], identity['data']['splits']['val']['selected_indices'])
        and np.array_equal(cached['node_ids'], data.station_ids), 'GW4 cached sample/node order differs')
    roads = np.full(len(data.station_ids), 'unknown', dtype='<U40')
    for road, indices in data.roads.items():
        roads[indices] = road
    require(np.array_equal(cached['road_labels'], roads), 'GW4 cached road labels differ')
    archive = {key:cached[key] for key in ('y', 'valid', 'sample_ids', 'node_ids', 'road_labels', 'direct_support')}
    archive.update(schema=np.array(SCHEMA), axis_order=np.array('sample,horizon,node'))
    replay, roles = {}, {}
    for case in REUSED_CASES:
        archive['pred__'+case] = cached['pred__'+case]
        roles[case] = dict(source_meta['checkpoint_roles'][case], reused=True)
        if case in old_states:
            state = old_states[case]
            epoch = state['best_epoch'] if args.checkpoint == 'best' else state['completed_epoch']
            replay[case] = dict(metrics.replay_metrics(archive['pred__'+case], archive['y'], state, epoch, modules),
                mode='certified_GW4_cache_no_new_forward')
        else:
            replay[case] = dict(source_meta['validation_replay'][case], mode='certified_historical_cache_no_new_forward')
        require(replay[case]['passed'], 'Cached control metrics differ: '+case)
    mechanisms = {case:dict(reused=True, source_directory=reference['archives'][args.checkpoint]['directory']) for case in REUSED_CASES}
    calls = {case:0 for case in REUSED_CASES}
    for arm, state in states.items():
        weights, role = selection(args.run_dir/arm, state, args.checkpoint, arm)
        roles[arm] = role
        prior = existing_cache(args.run_dir, identity, arm, role, archive)
        if prior:
            previous, previous_meta, directory = prior
            for case, _ in conditions(arm):
                archive['pred__'+case] = previous['pred__'+case]
                mechanisms[case] = dict(reused=True, source_directory=str(directory))
            replay[arm] = dict(metrics.replay_metrics(archive['pred__'+arm], archive['y'], state, role['epoch'], modules),
                mode='cached_identical_GW5_tensors_AND_policy')
            require(replay[arm]['passed'], 'GW5 cached replay differs')
            roles[arm]['prediction_source_role'] = previous_meta['checkpoint_roles'][arm]
            calls[arm] = 0
            del weights
            continue
        model = train.model_for_arm(data, modules, device, arm, source_weights)
        model.load_state_dict(weights, strict=True)
        train.gw4.check_frozen(model.state_dict(), identity)
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
                    'Actual GW5 labels/support differ from cached controls')
            archive['pred__'+case] = prediction
            mechanisms[case] = model.diagnostics()
            if setting is None:
                replay[arm] = dict(metrics.replay_metrics(prediction, truth, state, role['epoch'], modules), mode='new_GW5_forward')
                require(replay[arm]['passed'], 'GW5 native metric replay failed: '+arm)
        require(not model.selector.residual._anchors and not model.selector.residual._active, 'History cache leaked after evaluation')
        train.gw4.check_frozen(model.state_dict(), identity)
        del model, weights
        gc.collect()
        if torch.device(device).type == 'cuda':
            torch.cuda.empty_cache()
    validate_archive(archive)
    recertify(reference)
    metadata = dict(schema='gw5_history_anchor_producer_v1', status='GW5_PREDICTION_PASS',
        engineering_only=False, run_identity_sha256=identity['identity_sha256'], reference=identity['reference'],
        gw4_reference=reference, common_completed_adaptation_epochs=epochs, runtime=runtime,
        checkpoint_roles=roles, validation_replay=replay, new_inference_calls=calls,
        baseline_new_inference_calls=0, baseline_new_training_updates=0, optimizer_updates=0,
        test_evaluated=False, producer_source={name:metrics.sha256(REPO/'experiments/chronological'/name) for name in PRODUCERS})
    path = output/'paired_predictions.npz'
    np.savez_compressed(path, **archive)
    metadata['archive_sha256'] = metrics.sha256(path)
    metrics.write_json(output/'metadata.json', metadata)
    metrics.write_json(output/'mechanisms.json', mechanisms)
    return archive, metadata


def report(archive, output, metadata):
    analysis = metrics.report(archive, output, metadata, CASES, COMPARISONS, SCHEMA)
    y, valid, predictions = validate_archive(archive)
    extra, hard_ids = [], {}
    for baseline in ('GW2_BASE', 'RECEIVER_RESIDUAL'):
        hard = metrics.fixed_hard_indices(predictions[baseline], y, valid, archive['sample_ids'])
        hard_ids[baseline] = archive['sample_ids'][hard].tolist()
        for name, control, candidate in COMPARISONS:
            left, _ = metrics.window_scores(predictions[control], y, valid)
            right, _ = metrics.window_scores(predictions[candidate], y, valid)
            gain = float((left[hard]-right[hard]).mean()) if len(hard) else None
            analysis['comparisons'][name][baseline+'_fixed_hard_gain'] = gain
            extra.append(dict(hard_set=baseline, comparison=name, fixed_hard_window_gain=gain,
                mean_prediction_change=float((predictions[candidate]-predictions[control])[valid].mean())))
    analysis.update(schema='original_igstgnn_history_anchor_evaluation_v1', status='GW5_EVALUATION_COMPLETE',
        additional_fixed_hard_ids=hard_ids, budget_caveat='new_policies_and_GW4_controls_match_adaptation_not_total_from_scratch')
    comparisons = analysis['comparisons']
    screening = dict(
        beats_GW4_receiver_MAE=comparisons['HISTORY_ANCHORED_vs_RECEIVER_RESIDUAL']['mae_gain_pooled'] > 0.,
        beats_all_three_policy_controls_MAE=all(comparisons['HISTORY_ANCHORED_vs_'+arm]['mae_gain_pooled'] > 0.
            for arm in ('HISTORY_FIXED', 'PHASE_SHRINK', 'BIAS_SHRINK')),
        does_not_harm_GW2_fixed_hard_windows=comparisons['HISTORY_ANCHORED_vs_GW2_BASE']['GW2_BASE_fixed_hard_gain'] is not None
            and comparisons['HISTORY_ANCHORED_vs_GW2_BASE']['GW2_BASE_fixed_hard_gain'] >= 0.,
        P95_no_worse_than_GW4=analysis['cases']['HISTORY_ANCHORED']['window_p95'] is not None
            and analysis['cases']['RECEIVER_RESIDUAL']['window_p95'] is not None
            and analysis['cases']['HISTORY_ANCHORED']['window_p95'] <= analysis['cases']['RECEIVER_RESIDUAL']['window_p95'])
    screening['candidate_for_confirmation'] = all(screening.values())
    screening['interpretation'] = 'descriptive_single_seed_development_screen_not_statistical_confirmation'
    analysis['history_anchor_screening'] = screening
    metrics.write_json(output/'analysis.json', analysis)
    metrics._write_csv(output/'fixed_hard_and_shift.csv', extra)
    fmt = lambda value:'不可用' if value is None else f'{value:.9f}'
    lines = ['# GW5：历史锚定的递推消息修正', '',
        f"验证窗口：{len(y)}；仅归档重分析：{bool(metadata.get('engineering_only', True))}。", '',
        '四个新组与旧 GW4 RECEIVER 都从同一 GW2 起点和相同零残差开始，各五轮适配；没有从已训练 GW4 继续追加五轮。',
        'HISTORY_FIXED 保留最后历史窗口残差；HISTORY_ANCHORED 按 2/3、1/3、0 限制后续修正偏离；PHASE_SHRINK 仅缩小当前修正。',
        'BIAS_SHRINK 保留新增静态偏置，只缩小当前点积，排除历史锚定仅因保住偏置而获益的解释。',
        '旧 ORIGINAL、GW2、GW4 三组直接复用认证缓存；新组不增加可训练参数，旧模块保持 eval。', '',
        '| 状态 | MAE pooled | RMSE pooled | MAPE % | 窗口 P95 |', '|---|---:|---:|---:|---:|']
    for name, row in analysis['cases'].items():
        lines.append('| '+name+' | '+' | '.join(fmt(value) for value in (
            row['mae_pooled'], row['rmse_pooled'], 100*row['mape_pooled'], row['window_p95']))+' |')
    lines += ['', '| 比较 | MAE收益 | RMSE收益 | MAPE收益百分点 | GW2困难窗口收益 | GW4接收组困难窗口收益 |',
        '|---|---:|---:|---:|---:|---:|']
    for name, row in analysis['comparisons'].items():
        lines.append('| '+name+' | '+' | '.join(fmt(value) for value in (row['mae_gain_pooled'],
            row['rmse_gain_pooled'], 100*row['mape_gain'], row['GW2_BASE_fixed_hard_gain'],
            row['RECEIVER_RESIDUAL_fixed_hard_gain']))+' |')
    lines += ['', '收益=对照误差−候选误差。主方案须同时超过原 GW4、固定历史及两种缩放对照，并检查困难窗口与 P95。',
        'FREE 仅在同一新检查点取消策略，保留其已训练参数；不等于旧 GW4，也不替代独立训练。ANCHOR_ROLL 循环错配节点历史锚点，可能分布外。',
        'lambda=0 在历史锚定组保留历史修正，不恢复 GW2；所有新增修正为零或全程 BASE 才回到 GW2。评估 BASE 用历史缓存，不另跑模型。',
        '机制中的调用编号与输入来源比例不是最终预测时距、传播时延或置信度；最终12时距指标见 horizons.csv。',
        'history 缓存只在单次完整前向存在并保留梯度，完成或异常后清空；不能跨打乱窗口延续状态。',
        'groups.csv 原固定困难组沿用 ORIGINAL；fixed_hard_and_shift.csv 和本表另固定 GW2/GW4 两套困难集，集合分别标识。',
        '源GW2只完成五轮；本轮是同一锚点、单种子和反复使用的开发验证集上的适配筛查，不证明收敛或泛化。',
        '没有构造或评价测试 loader；原发布对象数组加载时可能反序列化包含测试的载荷。',
        '没有新增推理计时，训练阶段成本见训练 summary.json。', '']
    lines += ['## 预先声明的开发筛查', '',
        '进入确认阶段：'+str(screening['candidate_for_confirmation'])+'。只反映当前开发集四项描述性条件，不是显著性检验或独立确认。',
        '四项条件：MAE超过原GW4；MAE超过三个策略对照；GW2固定困难窗口无净损害；P95不差于原GW4。', '']
    if metadata.get('engineering_only', True):
        lines += ['本次仅重分析给定归档，没有独立重新认证训练身份和生产者。', '']
    (output/'report.md').write_text('\n'.join(lines), encoding='utf-8')
    return analysis


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--run-dir', type=Path)
    source.add_argument('--archive', type=Path)
    parser.add_argument('--checkpoint', choices=('best', 'last'), default='best')
    parser.add_argument('--data-dir', type=Path, default=metrics.default_data_dir())
    parser.add_argument('--device')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    args.output_dir = args.output_dir.resolve()
    if args.run_dir:
        args.run_dir = args.run_dir.resolve()
    require(not args.output_dir.exists(), 'Use a fresh GW5 evaluation directory')
    args.output_dir.mkdir(parents=True)
    try:
        if args.archive:
            before = metrics.sha256(args.archive)
            with np.load(args.archive, allow_pickle=False) as payload:
                archive = dict(payload)
            metadata = dict(engineering_only=True, source_archive_sha256=before, optimizer_updates=0,
                new_inference_calls=0, test_evaluated=False)
            result = report(archive, args.output_dir, metadata)
            require(metrics.sha256(args.archive) == before, 'Source archive changed')
        else:
            archive, metadata = produce(args, args.output_dir)
            result = report(archive, args.output_dir, metadata)
        print(json.dumps(dict(status=result['status'], validation_rows=result['validation_rows'], output_dir=str(args.output_dir))), flush=True)
        return result
    except BaseException as error:
        metrics.write_json(args.output_dir/'failure.json', dict(status='GW5_EVALUATION_FAILED',
            error_type=type(error).__name__, error=str(error), test_evaluated=False))
        raise


if __name__ == '__main__':
    main()
