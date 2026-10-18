"""GW6 paired validation of newly trained graph-memory architecture arms.

All primary comparisons use newly trained arms with the same GW4 BEST start,
stage, training budget, seed and curriculum. GW4/GW5 cached predictions are
historical references only. No test loader is constructed or evaluated.
"""
import argparse
import gc
import json
import logging
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import numpy as np
import torch

from experiments.graph_memory_gw6 import train
from experiments.graph_memory_gw6.reference import (
    gw5, load_cached, read_json, recertify, require, verify_identity_digest)
from experiments.chronological import evaluate_original_graph_message as metrics
from experiments.chronological.graph_message_reuse import RUNTIME_KEYS


SCHEMA = 'original_igstgnn_graph_memory_gw6_archive_v1'
PRODUCER_SCHEMA = 'gw6_graph_memory_producer_v1'
HISTORICAL_CASES = ('ORIGINAL', 'GW2_BASE', *train.gw4.ARMS, *gw5.ARMS)
CASES = (*HISTORICAL_CASES, *train.DEFAULT_ARMS)
PRODUCERS = ('experiments/graph_memory_gw6/evaluate.py',
    'experiments/graph_memory_gw6/reference.py',
    'experiments/chronological/evaluate_original_graph_message.py',
    'experiments/chronological/evaluate_unified_adapter_v03.py')
AXIS_KEYS = ('y', 'valid', 'sample_ids', 'node_ids', 'road_labels', 'direct_support')


def registered_arms(archive):
    values = np.asarray(archive.get('registered_arms'))
    require(values.ndim == 1 and len(values) > 0 and values.dtype.kind == 'U',
        'Archive requires an explicit Unicode registered_arms axis')
    arms = tuple(values.tolist())
    require(len(arms) == len(set(arms)) and set(arms) <= set(train.ARMS),
        'Archive has duplicate or unknown GW6 arms')
    return arms


def archive_stage(archive):
    value = np.asarray(archive.get('stage'))
    require(value.ndim == 0 and value.dtype.kind == 'U' and value.item() in ('joint', 'probe'),
        'Archive requires the explicit joint/probe training stage')
    return str(value.item())


def validate_archive(archive):
    arms = registered_arms(archive)
    stage = archive_stage(archive)
    require(stage != 'probe' or 'RECURRENT_CONTROL' not in arms,
        'Frozen probe cannot contain a newly trained recurrent control')
    return metrics.validate_archive(archive, (*HISTORICAL_CASES, *arms), SCHEMA)


def comparison_specs(arms):
    if 'JOINT_MEMORY' not in arms:
        return ()
    return tuple(('JOINT_MEMORY_vs_'+arm, arm, 'JOINT_MEMORY')
        for arm in arms if arm != 'JOINT_MEMORY') + (
        ('JOINT_MEMORY_vs_historical_RECEIVER_RESIDUAL', 'RECEIVER_RESIDUAL', 'JOINT_MEMORY'),
        ('JOINT_MEMORY_vs_historical_HISTORY_ANCHORED', 'HISTORY_ANCHORED', 'JOINT_MEMORY'))


def producer_source():
    return {name:metrics.sha256(REPO/name) for name in PRODUCERS}


def selection(directory, state, checkpoint, arm):
    if checkpoint == 'best':
        path = directory/state['best_checkpoint_file']
        require(metrics.sha256(path) == state['best_checkpoint_sha256'], 'GW6 BEST digest differs: '+arm)
        weights, epoch = train.base.load_checkpoint(path), state['best_epoch']
    else:
        path, weights, epoch = directory/'last.pt', state['model_state'], state['completed_epoch']
    digest = metrics.sha256(path)
    role = dict(selection=checkpoint, epoch=epoch, policy=arm,
        checkpoint_sha256=digest, model_tensor_sha256=train.base.tensor_tree_hash(weights),
        historical=False)
    require(metrics.sha256(path) == digest, 'GW6 checkpoint changed during selection: '+arm)
    return weights, role


def existing_cache(run, identity, arm, role, shared):
    """Reuse only the same arm's identical tensors under the same run identity.

    BEST and LAST may select identical tensors. The checkpoint role is still
    recorded separately, and validation metrics are replayed for the new role.
    A different arm with identical tensors is never treated as interchangeable.
    """
    for path in sorted(run.glob('*/metadata.json'), reverse=True):
        metadata = read_json(path)
        if metadata.get('schema') != PRODUCER_SCHEMA or metadata.get('status') != 'GW6_PREDICTION_PASS':
            continue
        if metadata.get('run_identity_sha256') != identity['identity_sha256']:
            continue
        if (metadata.get('source_reference') != identity['source_reference'] or
            metadata.get('stage') != identity['stage'] or
            metadata.get('registered_arms') != identity['arms']):
            continue
        previous = metadata.get('checkpoint_roles', {}).get(arm, {})
        if previous.get('policy') != arm or previous.get('model_tensor_sha256') != role['model_tensor_sha256']:
            continue
        if metadata.get('validation_replay', {}).get(arm, {}).get('passed') is not True:
            continue
        require(metadata.get('producer_source') == producer_source(), 'GW6 cached producer source changed')
        archive_path = path.parent/'paired_predictions.npz'
        require(metrics.sha256(archive_path) == metadata.get('archive_sha256'), 'GW6 cached archive digest differs')
        with np.load(archive_path, allow_pickle=False) as payload:
            archive = dict(payload)
        validate_archive(archive)
        require(list(registered_arms(archive)) == identity['arms'] and archive_stage(archive) == identity['stage'],
            'GW6 cache registered architecture or stage differs')
        keys = (*AXIS_KEYS, 'pred__ORIGINAL', 'pred__GW2_BASE')
        require(all(np.array_equal(archive[key], shared[key]) for key in keys), 'GW6 cached axes or immutable controls differ')
        require(metrics.sha256(archive_path) == metadata['archive_sha256'], 'GW6 cache changed during read')
        return archive, metadata, path.parent
    return None


def produce(args, output):
    identity = read_json(args.run_dir/'identity.json')
    verify_identity_digest(identity)
    require(identity.get('schema') == train.SCHEMA and identity.get('design') == train.DESIGN
        and identity.get('engineering_only') is False, 'Completed GW6 training identity required')
    arms = identity['arms']
    require(bool(arms) and len(arms) == len(set(arms)) and set(arms) <= set(train.ARMS),
        'GW6 registered arms differ')
    for name in train.EXECUTION_FILES:
        require(identity['execution_source'].get(name) == metrics.sha256(REPO/name), 'GW6 training source changed: '+name)
    reference = read_json(args.run_dir/'source_reference.json')
    require(reference == identity['source_reference'], 'GW6 source certificate differs from its identity')
    historical_identity, source_weights, _ = recertify(reference)
    require(identity['data'] == historical_identity['data']
        and identity['source_fingerprint'] == historical_identity['source_fingerprint'],
        'GW6 data or native source differ from certified historical evidence')
    for protected in train.protected_run_paths(reference):
        require(output != protected and protected not in output.parents,
            'GW6 evaluation must not write inside a historical source run')
    epochs = read_json(args.run_dir/'budget.json')['target_completed_epochs']
    require(isinstance(epochs, int) and 1 <= epochs <= 100, 'Invalid registered GW6 new-epoch budget')
    states, last_digests = {}, {}
    batches = len(identity['data']['splits']['train']['partition'])
    for arm in arms:
        directory = args.run_dir/arm
        last_digests[arm] = metrics.sha256(directory/'last.pt')
        state = train.base.load_checkpoint(directory/'last.pt')
        train.validate_saved(state, identity, arm, directory, batches)
        require(state['completed_epoch'] == epochs, 'Complete every registered GW6 arm at the same new budget')
        require(metrics.sha256(directory/'last.pt') == last_digests[arm], 'GW6 checkpoint changed during read')
        states[arm] = state
    modules = metrics.get_original_modules()
    require(modules.source_fingerprint == identity['source_fingerprint'], 'Native source fingerprint differs')
    device = args.device or identity['runtime']['python_device']
    runtime = metrics.configure_runtime(device, identity['runtime']['threads'], identity['training_config']['seed'])
    for key in RUNTIME_KEYS:
        require(runtime.get(key) == identity['runtime'].get(key), 'GW6 evaluation runtime differs: '+key)
    data = metrics.load_data(args.data_dir, identity['training_config'], modules,
        logging.getLogger('gw6-eval'), check=False, training=True)
    require(data.identity == identity['data'], 'Actual GW6 evaluation data differs')
    cached, historical_metadata = load_cached(reference, args.checkpoint)
    require(np.array_equal(cached['sample_ids'], identity['data']['splits']['val']['selected_indices'])
        and np.array_equal(cached['node_ids'], data.station_ids), 'Cached historical sample/node order differs')
    roads = np.full(len(data.station_ids), 'unknown', dtype='<U40')
    for road, indices in data.roads.items():
        roads[indices] = road
    require(np.array_equal(cached['road_labels'], roads), 'Cached historical road labels differ')
    archive = {key:cached[key] for key in AXIS_KEYS}
    archive.update(schema=np.array(SCHEMA), axis_order=np.array('sample,horizon,node'),
        registered_arms=np.asarray(arms, dtype=str), stage=np.array(identity['stage']))
    replay, roles, mechanisms, calls = {}, {}, {}, {}
    for case in HISTORICAL_CASES:
        archive['pred__'+case] = cached['pred__'+case]
        require(case in historical_metadata['checkpoint_roles']
            and historical_metadata['validation_replay'].get(case, {}).get('passed') is True,
            'Historical role or native replay missing: '+case)
        roles[case] = dict(historical_metadata['checkpoint_roles'][case], historical=True, reused=True)
        replay[case] = dict(historical_metadata['validation_replay'][case],
            mode='certified_historical_cache_no_new_forward', equal_new_training_budget=False)
        mechanisms[case] = dict(historical=True, reused=True,
            source_directory=reference['archives'][args.checkpoint]['directory'])
        calls[case] = 0
    for arm, state in states.items():
        weights, role = selection(args.run_dir/arm, state, args.checkpoint, arm)
        roles[arm] = role
        prior = existing_cache(args.run_dir, identity, arm, role, archive)
        if prior:
            previous, previous_metadata, directory = prior
            archive['pred__'+arm] = previous['pred__'+arm]
            mechanisms[arm] = dict(reused=True, source_directory=str(directory),
                prior_diagnostics=previous_metadata.get('mechanism_summary', {}).get(arm))
            replay[arm] = dict(metrics.replay_metrics(archive['pred__'+arm], archive['y'],
                state, role['epoch'], modules), mode='cached_identical_GW6_tensors_AND_architecture')
            require(replay[arm]['passed'], 'GW6 cached native replay differs: '+arm)
            roles[arm]['prediction_source_role'] = previous_metadata['checkpoint_roles'][arm]
            calls[arm] = 0
            del weights
            continue
        model = train.model_for_arm(data, modules, device, arm, source_weights,
            stage=identity['stage'], **identity['model_options'])
        model.load_state_dict(weights, strict=True)
        train.check_probe_frozen(model, identity)
        model.eval()
        model.start_diagnostics()
        predictions, truths, supports = [], [], []
        calls[arm] = 0
        with torch.no_grad():
            for index, raw in enumerate(data.loaders['val_loader'].get_iterator(), 1):
                x, y, incident, sensors = metrics.device_batch(raw, device)
                prediction = model(x, incident_data=incident, sensor_data=sensors)
                model.assert_cache_clear()
                calls[arm] += 1
                predictions.append(data.scaler.inverse_transform(prediction).squeeze(-1).cpu().numpy())
                truths.append(data.scaler.inverse_transform(y).squeeze(-1).cpu().numpy())
                supports.append(incident['distances'].abs().sum(-1).cpu().numpy() > 0)
                if index == 1 or index % 10 == 0 or index == data.loaders['val_loader'].num_batch:
                    print(json.dumps(dict(case=arm, batch=index,
                        total=data.loaders['val_loader'].num_batch)), flush=True)
        prediction, truth, support = map(np.concatenate, (predictions, truths, supports))
        require(np.array_equal(truth, archive['y']) and np.array_equal(support, archive['direct_support']),
            'Actual GW6 targets/support differ from shared historical axes')
        archive['pred__'+arm] = prediction
        mechanisms[arm] = model.diagnostics()
        replay[arm] = dict(metrics.replay_metrics(prediction, truth, state, role['epoch'], modules),
            mode='new_GW6_forward')
        require(replay[arm]['passed'], 'GW6 native metric replay failed: '+arm)
        model.assert_cache_clear()
        train.check_probe_frozen(model, identity)
        del model, weights
        gc.collect()
        if torch.device(device).type == 'cuda':
            torch.cuda.empty_cache()
    validate_archive(archive)
    for arm in arms:
        require(metrics.sha256(args.run_dir/arm/'last.pt') == last_digests[arm],
            'GW6 last checkpoint changed during evaluation: '+arm)
        selected = args.run_dir/arm/(states[arm]['best_checkpoint_file'] if args.checkpoint == 'best' else 'last.pt')
        require(metrics.sha256(selected) == roles[arm]['checkpoint_sha256'],
            'GW6 selected checkpoint changed during evaluation: '+arm)
    recertify(reference)
    metadata = dict(schema=PRODUCER_SCHEMA, status='GW6_PREDICTION_PASS',
        engineering_only=False, run_identity_sha256=identity['identity_sha256'], source_reference=reference,
        stage=identity['stage'], registered_arms=arms, model_options=identity['model_options'],
        curriculum=identity['curriculum'], training_config=identity['training_config'],
        initializations=identity['initializations'], common_completed_new_epochs=epochs,
        checkpoint_selection=args.checkpoint, runtime=runtime, checkpoint_roles=roles,
        validation_replay=replay, new_inference_calls=calls, mechanism_summary=mechanisms,
        historical_new_inference_calls=0, historical_new_training_updates=0,
        historical_GW4_GW5_are_equal_budget_controls=False, optimizer_updates=0,
        test_evaluated=False, producer_source=producer_source())
    path = output/'paired_predictions.npz'
    np.savez_compressed(path, **archive)
    metadata['archive_sha256'] = metrics.sha256(path)
    metrics.write_json(output/'metadata.json', metadata)
    metrics.write_json(output/'mechanisms.json', mechanisms)
    return archive, metadata


def report(archive, output, metadata):
    y, valid, predictions = validate_archive(archive)
    arms, stage = registered_arms(archive), archive_stage(archive)
    if not metadata.get('engineering_only', True):
        require(metadata.get('registered_arms') == list(arms) and metadata.get('stage') == stage,
            'Report producer stage/arm registration differs from archive')
    comparisons = comparison_specs(arms)
    cases = (*HISTORICAL_CASES, *arms)
    analysis = metrics.report(archive, output, metadata, cases, comparisons, SCHEMA)
    extra, cancellation, hard_ids = [], [], {}
    for baseline in ('GW2_BASE', 'RECEIVER_RESIDUAL'):
        hard = metrics.fixed_hard_indices(predictions[baseline], y, valid, archive['sample_ids'])
        hard_ids[baseline] = archive['sample_ids'][hard].tolist()
        for name, control, candidate in comparisons:
            left, _ = metrics.window_scores(predictions[control], y, valid)
            right, _ = metrics.window_scores(predictions[candidate], y, valid)
            gain = float((left[hard]-right[hard]).mean()) if len(hard) else None
            analysis['comparisons'][name][baseline+'_fixed_hard_gain'] = gain
            extra.append(dict(hard_set=baseline, comparison=name, fixed_hard_window_gain=gain,
                comparison_kind='historical_reference' if control in HISTORICAL_CASES else 'matched_new_training'))
    for name, prediction in predictions.items():
        error = prediction[valid].astype(np.float64)-y[valid].astype(np.float64)
        row = analysis['cases'][name]
        row['signed_bias'] = float(error.mean())
        row['historical_reference'] = name in HISTORICAL_CASES
        for horizon in row['horizons']:
            h = horizon['step']-1
            mask = valid[:, h]
            horizon['signed_bias'] = float((prediction[:, h][mask].astype(np.float64)-
                y[:, h][mask].astype(np.float64)).mean()) if mask.any() else None
    for name, control, candidate in comparisons:
        row = analysis['comparisons'][name]
        row['comparison_kind'] = 'historical_reference' if control in HISTORICAL_CASES else 'matched_new_training'
        row['control_signed_bias'] = analysis['cases'][control]['signed_bias']
        row['candidate_signed_bias'] = analysis['cases'][candidate]['signed_bias']
        cancellation.append(dict(comparison=name, **{key:row[key] for key in (
            'comparison_kind', 'mae_gain_pooled', 'positive_gain', 'negative_gain',
            'cancellation_fraction', 'prediction_change_mean', 'prediction_change_abs_mean',
            'control_signed_bias', 'candidate_signed_bias')}))
    primary_names = ['JOINT_MEMORY_vs_'+arm for arm in train.DEFAULT_ARMS if arm != 'JOINT_MEMORY']
    primary_available = all(name in analysis['comparisons'] for name in primary_names)
    screening = dict(all_default_primary_controls_present=primary_available,
        jointly_trained_primary_controls=stage == 'joint',
        beats_all_primary_controls_MAE=(all(analysis['comparisons'][name]['mae_gain_pooled'] > 0.
            for name in primary_names) if primary_available else None),
        interpretation='descriptive_single_seed_development_screen_not_statistical_confirmation')
    if 'JOINT_MEMORY_vs_RECURRENT_CONTROL' in analysis['comparisons']:
        pair = analysis['comparisons']['JOINT_MEMORY_vs_RECURRENT_CONTROL']
        screening['no_loss_on_GW2_fixed_hard'] = pair['GW2_BASE_fixed_hard_gain'] is not None and pair['GW2_BASE_fixed_hard_gain'] >= 0.
        screening['no_loss_on_GW4_fixed_hard'] = pair['RECEIVER_RESIDUAL_fixed_hard_gain'] is not None and pair['RECEIVER_RESIDUAL_fixed_hard_gain'] >= 0.
        screening['P95_no_worse_than_new_recurrent_control'] = (
            analysis['cases']['JOINT_MEMORY']['window_p95'] is not None and
            analysis['cases']['RECURRENT_CONTROL']['window_p95'] is not None and
            analysis['cases']['JOINT_MEMORY']['window_p95'] <= analysis['cases']['RECURRENT_CONTROL']['window_p95'])
    screening['producer_independently_certified'] = not metadata.get('engineering_only', True)
    screening['candidate_for_confirmation'] = (screening['producer_independently_certified'] and
        primary_available and stage == 'joint' and
        screening['beats_all_primary_controls_MAE'] is True and
        screening.get('no_loss_on_GW2_fixed_hard') is True and
        screening.get('no_loss_on_GW4_fixed_hard') is True and
        screening.get('P95_no_worse_than_new_recurrent_control') is True)
    analysis.update(schema='original_igstgnn_graph_memory_gw6_evaluation_v1', status='GW6_EVALUATION_COMPLETE',
        stage=stage, registered_arms=list(arms), historical_cases=list(HISTORICAL_CASES),
        additional_fixed_hard_ids=hard_ids, graph_memory_screening=screening,
        historical_GW4_GW5_are_equal_budget_controls=False,
        budget_caveat='new_arms_share_GW4_BEST_start_and_completed_new_budget_not_total_from_scratch')
    output = Path(output)
    metrics.write_json(output/'analysis.json', analysis)
    metrics._write_csv(output/'fixed_hard_and_shift.csv', extra)
    metrics._write_csv(output/'cancellation_and_bias.csv', cancellation)
    metrics._write_csv(output/'cases.csv', [dict(case=name, **{key:value for key,value in row.items()
        if key != 'horizons'}) for name,row in analysis['cases'].items()])
    metrics._write_csv(output/'horizons.csv', [dict(case=name, **horizon)
        for name,row in analysis['cases'].items() for horizon in row['horizons']])
    metrics._write_csv(output/'comparisons.csv', [dict(comparison=name, **{key:value for key,value in row.items()
        if key not in ('horizons', 'directions')}) for name,row in analysis['comparisons'].items()])
    fmt = lambda value:'不可用' if value is None else f'{value:.9f}'
    lines = ['# GW6：结构化历史图消息记忆与直接多时距解码', '',
        f'阶段：{stage}；验证窗口：{len(y)}；仅归档重分析：{analysis["engineering_only"]}。', '',
        '主比较只在本轮独立训练的新结构臂之间进行；各臂继承同一认证 GW4 RECEIVER_RESIDUAL BEST 权重，并使用相同新增轮数、数据、种子、损失和课程。',
        'RECURRENT_CONTROL 继续训练原递推架构；POOLED_MEMORY 合并历史消息再读出；ADDITIVE_MEMORY 使用普通加法 query 的联合 attention；JOINT_MEMORY 使用接收状态与时距嵌入交互的 query 联合读取结构化历史记忆。',
        '可选 FACTORIZED_MEMORY 才显式分解来源和历史窗口的读取概率；联合 attention 的轴为来源×传播阶数×历史窗口。',
        '旧 GW4/GW5 行来自认证缓存，本轮训练和推理调用为零，仅是历史参考，不属于新增预算匹配对照。',
        'joint 联合更新继承主干和新增结构；probe 冻结主干，只筛查新增读出，不支持联合训练主结论。', '',
        '| 状态 | 角色 | MAE pooled | RMSE pooled | MAPE % | 窗口 P95 | 有符号偏差 |',
        '|---|---|---:|---:|---:|---:|---:|']
    for name,row in analysis['cases'].items():
        role = '历史参考' if name in HISTORICAL_CASES else '本轮结构对照'
        lines.append('| '+name+' | '+role+' | '+' | '.join(fmt(value) for value in (
            row['mae_pooled'],row['rmse_pooled'],100*row['mape_pooled'],row['window_p95'],row['signed_bias']))+' |')
    lines += ['', '| 比较 | 类型 | MAE收益 | GW2固定困难窗口收益 | GW4固定困难窗口收益 | 正收益 | 负收益幅度 | 抵消比例 |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for name,row in analysis['comparisons'].items():
        lines.append('| '+name+' | '+row['comparison_kind']+' | '+' | '.join(fmt(row[key]) for key in (
            'mae_gain_pooled','GW2_BASE_fixed_hard_gain','RECEIVER_RESIDUAL_fixed_hard_gain',
            'positive_gain','negative_gain','cancellation_fraction'))+' |')
    lines += ['', '收益=对照误差−候选误差，负收益如实保留；正负幅度和抵消比例均以共享有效点统计。signed_bias=预测−真实值的均值。',
        '十二个预测时距保留于 horizons.csv；P95 按完整十二时距窗口的等权 MAE 计算。',
        'GW2/GW4 困难窗口分别由认证历史 GW2_BASE/RECEIVER_RESIDUAL 误差预先固定，不能按候选重新选择；groups.csv 的旧固定困难组由 ORIGINAL 定义。',
        '直接多时距解码属于架构改变，不是原模型的零增量残差：不声称零初始化就精确恢复原模型。递推对照独立继续训练。',
        '不同架构的实际可训练参数量见 metadata.json.initializations；容量不必相等，不能把等轮数解读为等 FLOPs。训练成本见 run summary.json。',
        '本次未追加推理计时。相同验证集已反复用于设计筛查；单种子和短预算结果不证明收敛、普适优越或独立泛化。',
        '没有构造或评价测试 loader；原发布对象数组加载时可能反序列化包含测试的载荷。',
        '预声明开发筛查进入确认阶段：'+str(screening['candidate_for_confirmation'])+'。该标记不是统计显著性结论。', '']
    if metadata.get('engineering_only', True):
        lines += ['本次仅重分析给定归档，未独立核验模型、训练身份与生产者。', '']
    (output/'report.md').write_text('\n'.join(lines), encoding='utf-8')
    return analysis


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--run-dir', type=Path)
    source.add_argument('--archive', type=Path)
    parser.add_argument('--checkpoint', choices=('best','last'), default='best')
    parser.add_argument('--data-dir', type=Path, default=metrics.default_data_dir())
    parser.add_argument('--device')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    args.output_dir = args.output_dir.resolve()
    if args.run_dir:
        args.run_dir = args.run_dir.resolve()
        # Authenticate protected source paths before creating any output.
        identity = read_json(args.run_dir/'identity.json')
        for protected in train.protected_run_paths(identity.get('source_reference', {})):
            require(args.output_dir != protected and protected not in args.output_dir.parents,
                'Evaluation output must be outside every historical source run')
    require(not args.output_dir.exists(), 'Use a fresh GW6 evaluation directory')
    args.output_dir.mkdir(parents=True)
    try:
        if args.archive:
            before = metrics.sha256(args.archive)
            with np.load(args.archive, allow_pickle=False) as payload:
                archive = dict(payload)
            metadata = dict(engineering_only=True, source_archive_sha256=before,
                producer_identity_independently_verified=False, optimizer_updates=0,
                new_inference_calls=0, test_evaluated=False)
            result = report(archive, args.output_dir, metadata)
            require(metrics.sha256(args.archive) == before, 'Source archive changed')
        else:
            archive, metadata = produce(args, args.output_dir)
            result = report(archive, args.output_dir, metadata)
        print(json.dumps(dict(status=result['status'], validation_rows=result['validation_rows'],
            output_dir=str(args.output_dir))), flush=True)
        return result
    except BaseException as error:
        metrics.write_json(args.output_dir/'failure.json', dict(status='GW6_EVALUATION_FAILED',
            error_type=type(error).__name__, error=str(error), optimizer_updates=0, test_evaluated=False))
        raise


if __name__ == '__main__':
    main()
