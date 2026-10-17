"""GW3 paired validation; old experiments and identical-model caches are reused."""
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

from experiments.chronological import train_original_receiver_match as train
from experiments.chronological import evaluate_original_graph_message as metrics
from experiments.chronological.receiver_match_reference import (
    REUSED_ARMS, certify, load_cached, read_json, require, verify_identity_digest)


SCHEMA = 'original_igstgnn_receiver_match_archive_v1'
CASES = (*REUSED_ARMS, *train.ARMS, 'MESSAGE_SELF_UNIFORM', 'MESSAGE_SELF_STATIC_ONLY',
    'RECEIVER_MATCH_UNIFORM', 'RECEIVER_MATCH_STATIC_ONLY', 'RECEIVER_MATCH_QUERY_SELF')
COMPARISONS = tuple((b+'_vs_'+a, a, b) for a, b in train.PAIRS) + (
    ('message_selector_dependency', 'MESSAGE_SELF_UNIFORM', 'MESSAGE_SELF'),
    ('message_content_dependency', 'MESSAGE_SELF_STATIC_ONLY', 'MESSAGE_SELF'),
    ('receiver_selector_dependency', 'RECEIVER_MATCH_UNIFORM', 'RECEIVER_MATCH'),
    ('receiver_content_dependency', 'RECEIVER_MATCH_STATIC_ONLY', 'RECEIVER_MATCH'),
    ('receiver_query_dependency', 'RECEIVER_MATCH_QUERY_SELF', 'RECEIVER_MATCH'))
PRODUCERS = ('evaluate_original_receiver_match.py', 'receiver_match_reference.py',
    'evaluate_original_graph_message.py', 'evaluate_unified_adapter_v03.py')


def conditions(arm):
    result = [(arm, None)]
    if arm != 'FIXED_UNIFORM':
        result += [(arm+'_UNIFORM', 'UNIFORM'), (arm+'_STATIC_ONLY', 'STATIC')]
    if arm == 'RECEIVER_MATCH':
        result.append((arm+'_QUERY_SELF', 'QUERY_SELF'))
    return result


def validate_archive(archive):
    return metrics.validate_archive(archive, CASES, SCHEMA)


def reference_predictions(reference, selection, identity, states, modules):
    cached, metadata = load_cached(reference, selection)
    require(np.array_equal(cached['sample_ids'], identity['data']['splits']['val']['selected_indices']),
            'Reference validation order differs')
    archive = {key: cached[key] for key in ('y', 'valid', 'sample_ids', 'node_ids', 'road_labels', 'direct_support')}
    archive.update(schema=np.array(SCHEMA), axis_order=np.array('sample,horizon,node'))
    replay = {}
    for arm in REUSED_ARMS:
        archive['pred__'+arm] = cached['pred__'+arm]
        state = states[arm]
        epoch = state['best_epoch'] if selection == 'best' else state['completed_epoch']
        replay[arm] = dict(metrics.replay_metrics(cached['pred__'+arm], cached['y'], state, epoch, modules),
            mode='cached_metrics_not_new_baseline_forward', new_inference_calls=0)
        require(replay[arm]['passed'], 'Reference metrics differ from certified history: '+arm)
    return archive, metadata, replay


def model_selection(directory, state, selection):
    if selection == 'best':
        path = directory/state['best_checkpoint_file']
        require(metrics.sha256(path) == state['best_checkpoint_sha256'], 'GW3 best checkpoint digest differs')
        weights, epoch = train.base.load_checkpoint(path), state['best_epoch']
    else:
        path, weights, epoch = directory/'last.pt', state['model_state'], state['completed_epoch']
    return weights, dict(selection=selection, epoch=epoch, checkpoint_sha256=metrics.sha256(path),
                        model_tensor_sha256=train.base.tensor_tree_hash(weights))


def existing_cache(run, identity, arm, role, reference):
    """Reuse a model's predictions across best/last file wrappers by tensor hash."""
    for path in sorted(run.glob('*/metadata.json'), reverse=True):
        meta = read_json(path)
        if meta.get('status') != 'GW3_PREDICTION_PASS' or meta.get('schema') != 'gw3_receiver_match_producer_v1':
            continue
        if meta.get('run_identity_sha256') != identity['identity_sha256']:
            continue
        previous = meta.get('checkpoint_roles', {}).get(arm, {})
        if previous.get('model_tensor_sha256') != role['model_tensor_sha256']:
            continue
        if meta.get('validation_replay', {}).get(arm, {}).get('passed') is not True:
            continue
        for name in PRODUCERS:
            require(meta.get('producer_source', {}).get(name) == metrics.sha256(REPO/'experiments/chronological'/name),
                    'GW3 cached producer changed: '+name)
        archive_path = path.parent/'paired_predictions.npz'
        require(metrics.sha256(archive_path) == meta.get('archive_sha256'), 'GW3 cached archive digest differs')
        with np.load(archive_path, allow_pickle=False) as payload:
            cached = dict(payload)
        validate_archive(cached)
        require(all(np.array_equal(cached[key], reference[key]) for key in (
            'y', 'valid', 'sample_ids', 'node_ids', 'road_labels', 'direct_support')), 'GW3 cache axes/mask differ')
        require(metrics.sha256(archive_path) == meta['archive_sha256'], 'GW3 cache changed during read')
        return cached, meta, path.parent
    return None


def produce(args, output):
    identity = read_json(args.run_dir/'identity.json')
    require(identity.get('schema') == train.SCHEMA and identity.get('design') == train.DESIGN,
            'GW3 training identity required')
    verify_identity_digest(identity)
    require(not identity['engineering_only'], 'Full paired report requires the full validation/reference run')
    for name in train.EXECUTION_FILES:
        require(identity['execution_source'].get(name) == metrics.sha256(REPO/name), 'GW3 execution source changed: '+name)
    epochs = read_json(args.run_dir/'budget.json')['target_completed_epochs']
    saved_reference = read_json(args.run_dir/('reference_epoch_%03d.json' % epochs))
    evaluations = {key: saved_reference['archives'][key]['directory'] for key in ('best', 'last')}
    reference, _, old_states = certify(saved_reference['source_run'], epochs, target=identity, evaluations=evaluations)
    require(reference == saved_reference, 'Historical reference changed')
    states = {}
    for arm in train.ARMS:
        directory = args.run_dir/arm
        before = metrics.sha256(directory/'last.pt')
        state = train.base.load_checkpoint(directory/'last.pt')
        train.validate_saved(state, identity, arm, directory, len(identity['data']['splits']['train']['partition']))
        require(state['completed_epoch'] == epochs, 'Complete each new arm at the common budget')
        require(metrics.sha256(directory/'last.pt') == before, 'GW3 last checkpoint changed during read')
        states[arm] = state
    modules = metrics.get_original_modules()
    require(modules.source_fingerprint == identity['source_fingerprint'], 'Frozen native source differs')
    device = args.device or identity['runtime']['python_device']
    runtime = metrics.configure_runtime(device, identity['runtime']['threads'], identity['training_config']['seed'])
    # Training and native replay must use the recorded runtime/device config.
    from experiments.chronological.graph_message_reuse import RUNTIME_KEYS
    for key in RUNTIME_KEYS:
        require(runtime.get(key) == identity['runtime'].get(key), 'Evaluation runtime differs: '+key)
    data = metrics.load_data(args.data_dir, identity['training_config'], modules,
        logging.getLogger('gw3-eval'), check=False, training=True)
    require(data.identity == identity['data'], 'Actual evaluation data differs')
    archive, source_meta, replay = reference_predictions(reference, args.checkpoint, identity, old_states, modules)
    require(np.array_equal(archive['node_ids'], data.station_ids), 'Reference node order differs')
    roads = np.full(len(data.station_ids), 'unknown', dtype='<U40')
    for road, indices in data.roads.items():
        roads[indices] = road
    require(np.array_equal(archive['road_labels'], roads), 'Reference road axis differs')
    costs = {arm: dict(source_meta['benchmarks'][arm], measurement_origin='prior_session_reference') for arm in REUSED_ARMS}
    roles = {arm: dict(source_meta['checkpoint_roles'][arm], reused=True) for arm in REUSED_ARMS}
    mechanisms = {arm: dict(reused=True, source_directory=reference['archives'][args.checkpoint]['directory']) for arm in REUSED_ARMS}
    calls = {arm: 0 for arm in REUSED_ARMS}
    loader = data.loaders['val_loader']
    for arm, state in states.items():
        weights, role = model_selection(args.run_dir/arm, state, args.checkpoint)
        roles[arm] = role
        cached = existing_cache(args.run_dir, identity, arm, role, archive)
        if cached:
            prior, prior_meta, prior_directory = cached
            for case, _ in conditions(arm):
                archive['pred__'+case] = prior['pred__'+case]
                mechanisms[case] = dict(reused=True, source_directory=str(prior_directory))
            replay[arm] = dict(metrics.replay_metrics(archive['pred__'+arm], archive['y'], state, role['epoch'], modules),
                mode='cached_identical_model_tensors_no_new_forward')
            require(replay[arm]['passed'], 'GW3 cached metrics differ')
            costs[arm] = dict(prior_meta['benchmarks'][arm], measurement_origin='GW3_cached_prior_session')
            roles[arm]['prediction_source_role'] = prior_meta['checkpoint_roles'][arm]
            calls[arm] = 0
            del weights
            continue
        model = train.model_for_arm(data, modules, device, arm)
        model.load_state_dict(weights, strict=True)
        model.eval()
        costs[arm] = dict(metrics.benchmark(model, metrics.device_batch(next(loader.get_iterator()), device), device),
            measurement_origin='GW3_current_session', measurement_runtime=runtime)
        calls[arm] = 5
        for case, setting in conditions(arm):
            predictions, truths, supports = [], [], []
            model.start_diagnostics()
            context = model.intervention(setting) if setting else nullcontext()
            with torch.no_grad(), context:
                for index, raw in enumerate(loader.get_iterator(), 1):
                    x, y, incident, sensors = metrics.device_batch(raw, device)
                    prediction = model(x, incident_data=incident, sensor_data=sensors)
                    calls[arm] += 1
                    predictions.append(data.scaler.inverse_transform(prediction).squeeze(-1).cpu().numpy())
                    truths.append(data.scaler.inverse_transform(y).squeeze(-1).cpu().numpy())
                    supports.append(incident['distances'].abs().sum(-1).cpu().numpy() > 0)
                    if index == 1 or index % 10 == 0 or index == loader.num_batch:
                        print(dict(case=case, batch=index, total=loader.num_batch), flush=True)
            prediction, truth, support = map(np.concatenate, (predictions, truths, supports))
            require(np.array_equal(truth, archive['y']) and np.array_equal(support, archive['direct_support']),
                    'GW3 actual targets/support differ from cached references')
            archive['pred__'+case] = prediction
            mechanisms[case] = model.diagnostics()
            if setting is None:
                replay[arm] = dict(metrics.replay_metrics(prediction, truth, state, role['epoch'], modules), mode='new_forward')
                require(replay[arm]['passed'], 'GW3 native validation replay failed: '+arm)
        del model, weights
        gc.collect()
        if torch.device(device).type == 'cuda':
            torch.cuda.empty_cache()
    validate_archive(archive)
    fresh, _, _ = certify(reference['source_run'], epochs, target=identity, evaluations=evaluations)
    require(fresh == reference, 'Reference artifacts changed during evaluation')
    metadata = dict(schema='gw3_receiver_match_producer_v1', status='GW3_PREDICTION_PASS',
        engineering_only=False, run_identity_sha256=identity['identity_sha256'],
        common_completed_epochs=epochs, checkpoint_roles=roles, validation_replay=replay,
        benchmarks=costs, runtime=runtime, reference=reference, new_inference_calls=calls,
        baseline_new_inference_calls=0, optimizer_updates=0, test_evaluated=False,
        producer_source={name: metrics.sha256(REPO/'experiments/chronological'/name) for name in PRODUCERS})
    path = output/'paired_predictions.npz'
    np.savez_compressed(path, **archive)
    metadata['archive_sha256'] = metrics.sha256(path)
    metrics.write_json(output/'metadata.json', metadata)
    metrics.write_json(output/'mechanisms.json', mechanisms)
    return archive, metadata


def report(archive, output, metadata):
    # Reuse only the established numerical summaries/masks/group definitions.
    analysis = metrics.report(archive, output, metadata, CASES, COMPARISONS, SCHEMA)
    analysis.update(schema='original_igstgnn_receiver_match_evaluation_v1', status='GW3_EVALUATION_PASS')
    metrics.write_json(output/'analysis.json', analysis)
    lines = ['# GW3：接收状态—图消息匹配', '',
        f"工程检查：{analysis['engineering_only']}；验证窗口：{analysis['validation_rows']}。", '',
        'ORIGINAL、STATIC、POST_MESSAGE 使用已认证历史归档，本轮新增训练/推理均为零。',
        'STATIC 为 GW1 历史内核参考；POST_MESSAGE 为 GW2 历史评分器，均不是 GW3 同容量重训组。',
        'FIXED_UNIFORM 从训练开始固定均匀汇聚；MESSAGE_SELF 与 RECEIVER_MATCH 使用相同匹配参数与初始化，只有查询输入不同。',
        '主比较 RECEIVER_MATCH vs ORIGINAL；必须同时看 FIXED_UNIFORM 与 MESSAGE_SELF 两个结构对照。',
        '收益=对照误差−候选误差，正值更好；五轮验证不证明收敛、多种子或独立测试优势。', '',
        '| 状态 | MAE pooled | RMSE pooled | MAPE % | 窗口 P95 |', '|---|---:|---:|---:|---:|']
    fmt = lambda x: '不可用' if x is None else f'{x:.9f}'
    for name, row in analysis['cases'].items():
        lines.append('| '+name+' | '+' | '.join(fmt(v) for v in (
            row['mae_pooled'], row['rmse_pooled'], 100*row['mape_pooled'], row['window_p95']))+' |')
    lines += ['', '| 比较 | MAE收益 | RMSE收益 | MAPE收益百分点 | 固定困难窗口收益 |', '|---|---:|---:|---:|---:|']
    for name, row in analysis['comparisons'].items():
        lines.append('| '+name+' | '+' | '.join(fmt(v) for v in (
            row['mae_gain_pooled'], row['rmse_gain_pooled'], 100*row['mape_gain'], row['fixed_hard_window_equal_weight_gain']))+' |')
    lines += ['', '## 成本与机制边界', '',
        '| 训练组 | 批量 | 前向均值 ms | 峰值 allocated bytes | 来源 |', '|---|---:|---:|---:|---|']
    for arm, cost in metadata.get('benchmarks', {}).items():
        lines.append(f"| {arm} | {cost['batch_size']} | {fmt(cost['mean_milliseconds'])} | {cost['peak_allocated_bytes']} | {cost['measurement_origin']} |")
    lines += ['', '历史参考不重新计时，不以跨会话数字宣称加速。当前计时为首批量完整前向，热身2次、测量3次，CUDA同步，不含H2D及统计，不代表训练速度。',
        'UNIFORM/STATIC_ONLY/QUERY_SELF 保留各自联合训练后的主干；不能替代独立训练组。QUERY_SELF 改变查询证据，可能分布外。',
        'STATIC_ONLY 仅保留显式图槽位先验，不保留动态评分的训练均值；不能当作纯跨样本变化贡献。',
        '点积匹配不等于可信度或因果；可能偏好相似状态而漏掉邻居领先变化。',
        '槽位经过原线性层混合，不是物理时延；一阶权重用于原有高阶图，自身路径不加权。',
        '权重和K不保证预测均值、能量或车辆守恒；零键初始化只承诺浮点容差恢复。',
        '固定困难窗口沿用该归档ORIGINAL最差10%；真值流量分组只作事后诊断，窗口可能重叠。',
        '没有新增事故信息输入、传播阶数或测试集评价。原发布数组可能反序列化测试载荷，未构造测试批次。',
        '数值文件：analysis.json、cases.csv、comparisons.csv、horizons.csv、groups.csv；来源：metadata.json、mechanisms.json、paired_predictions.npz。', '']
    (output/'report.md').write_text('\n'.join(lines), encoding='utf-8')
    return analysis


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, default=metrics.default_data_dir())
    parser.add_argument('--checkpoint', choices=('best', 'last'), default='best')
    parser.add_argument('--device')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    args.run_dir, args.output_dir = args.run_dir.resolve(), args.output_dir.resolve()
    require(not args.output_dir.exists(), 'Use a fresh output directory; existing evaluations are preserved')
    # Acquire before creating output, so a running trainer is not disturbed.
    with train.base.run_lock(args.run_dir):
        args.output_dir.mkdir(parents=True)
        try:
            archive, metadata = produce(args, args.output_dir)
            analysis = report(archive, args.output_dir, metadata)
            print(json.dumps(dict(status=analysis['status'], output_dir=str(args.output_dir),
                new_inference_calls=metadata['new_inference_calls'])), flush=True)
            return analysis
        except BaseException as error:
            metrics.write_json(args.output_dir/'failure.json', dict(status='GW3_EVALUATION_FAILED',
                error_type=type(error).__name__, error=str(error), optimizer_updates=0, test_evaluated=False))
            raise


if __name__ == '__main__':
    main()
