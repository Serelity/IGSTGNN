"""Read-only paired best-checkpoint evaluation of MIN forecast feature fusion.

SOURCE_MIN is an immutable, previously verified prediction archive. Both current
ON replays must pass before any intervention. No test set or training is used.
"""
import argparse
from contextlib import ExitStack
import csv
from datetime import datetime
import gc
import json
import logging
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import uuid

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.chronological import train_min_forecast_fusion as training
from experiments.chronological import evaluate_min_early_event as early_evaluation
from experiments.chronological import min_early_event_metrics as old_metrics
from experiments.chronological import probe_min_event_budget as probe
from experiments.chronological.smoke import sha256
from src.models.incident_spatial import require
from src.models.incident_spatial_residual import tensor_state_hash
from src.utils.igstgnn_published_events import EVENT_KEYS, encode_event
from src.utils.metrics import compute_all_metrics

SCHEMA = 'min_forecast_fusion_evaluation_v1'
CASES = ('STATE_ON', 'EVENT_ON', 'EVENT_REPORT_OFF', 'EVENT_FUSION_OFF',
         'EVENT_UNIFORM', 'EVENT_SEMANTIC_UNK', 'EVENT_LOCATION_SHUFFLE', 'SOURCE_MIN')
COMPONENT_ORDER = tuple('DIF'+str(i) for i in range(5))+tuple('INH'+str(i) for i in range(5))
# The legacy metric engine genuinely selects its fixed hard set and bias labels
# from internal STATE_ON. Map that reference to SOURCE_MIN, not the new STATE.
ALIASES = {'STATE_ON': 'SOURCE_MIN', 'FUSION_STATE_ON': 'STATE_ON', 'EVENT_ON': 'EVENT_ON',
    'EVENT_OFF': 'EVENT_REPORT_OFF', 'EVENT_FUSION_OFF': 'EVENT_FUSION_OFF',
    'EVENT_UNIFORM': 'EVENT_UNIFORM', 'EVENT_SEMANTIC_UNK': 'EVENT_SEMANTIC_UNK',
    'EVENT_LOCATION_SHUFFLE': 'EVENT_LOCATION_SHUFFLE'}
COMPARISONS = {
    'EVENT_vs_STATE': ('FUSION_STATE_ON', 'EVENT_ON', 'paired_warmstart_independent_best_validation'),
    'report_content_dependency': ('EVENT_OFF', 'EVENT_ON', 'same_checkpoint_report_content_OFF_state_fusion_preserved'),
    'fusion_dependency': ('EVENT_FUSION_OFF', 'EVENT_ON', 'same_checkpoint_fusion_OFF_current_MIN_not_SOURCE_MIN'),
    'component_differentiation': ('EVENT_UNIFORM', 'EVENT_ON', 'same_checkpoint_component_mean_weights_not_independently_trained_calibration'),
    'semantic_UNK_sensitivity': ('EVENT_SEMANTIC_UNK', 'EVENT_ON', 'same_checkpoint_semantic_UNK_potential_OOD'),
    'location_shuffle_sensitivity': ('EVENT_LOCATION_SHUFFLE', 'EVENT_ON', 'same_checkpoint_full_station_shuffle_potential_OOD'),
    'EVENT_vs_SOURCE': ('STATE_ON', 'EVENT_ON', 'source_best_vs_warmstart_candidate_including_extra_training'),
    'STATE_vs_SOURCE': ('STATE_ON', 'FUSION_STATE_ON', 'source_best_vs_warmstart_candidate_including_extra_training'),
}


def selected_metrics(saved, baseline):
    if saved['best_epoch'] == 0:
        return baseline['actual']
    row = saved['history'][saved['best_epoch']-1]
    require(row['validation_mae'] == saved['best_metric'], 'Selected stage checkpoint metric changed')
    return [row[key] for key in ('validation_mae', 'validation_mape', 'validation_rmse')]


def semantic_unknown(events):
    """Use genuine registered UNK tokens; preserve relation, support and values."""
    result = {key: np.asarray(value).copy() for key, value in events.items()}
    for key in ('type_valid', 'description_valid'):
        result[key].fill(False)
    return result


def forward_case(model, batch, scaler, case, events_override=None):
    require(case in CASES[:-1], 'Unknown inference case')
    before = probe.input_digest(batch, events_override)
    try:
        device = next(model.parameters()).device
        x = torch.as_tensor(batch['x_data'], device=device, dtype=torch.float32)
        y = torch.as_tensor(batch['y_data'], device=device, dtype=torch.float32)
        events = batch['event_data'] if events_override is None else events_override
        if case == 'EVENT_SEMANTIC_UNK':
            events = semantic_unknown(events)
        event = training.event_tensors(events, device)
        with torch.inference_mode():
            result = model(x, incident_data=event, report_enabled=case != 'EVENT_REPORT_OFF',
                semantic_enabled=True, fusion_enabled=case != 'EVENT_FUSION_OFF',
                uniform_weights=case == 'EVENT_UNIFORM', return_spatial_details=True)
            prediction = scaler.inverse_transform(result['prediction']).squeeze(-1).cpu().numpy().copy()
            target = scaler.inverse_transform(y).squeeze(-1).cpu().numpy().copy()
        require(np.isfinite(prediction).all() and np.isfinite(target).all(), 'Nonfinite fusion validation output')
        fusion = result['fusion']
        weights = fusion['weights'].detach().cpu().numpy().astype(np.float64)
        require(weights.shape == prediction.shape+(10,) and np.isfinite(weights).all(), 'Fusion weight axes differ')
        require(tuple(fusion['component_order']) == COMPONENT_ORDER, 'Fusion component order differs')
        support = fusion['support'].detach().cpu().numpy()
        require(support.dtype == np.bool_ and support.shape == (len(prediction), prediction.shape[2], 1),
                'Fusion support axes differ')
        mechanisms = early_evaluation.small_mechanisms(result['spatial'])
        mechanisms.update(fusion_enabled_calls=int(fusion['enabled']),
            fusion_event_context_calls=int(fusion['event_context_used']),
            fusion_supported_node_positions=int(support.sum()),
            fusion_node_positions=int(support.shape[0]*support.shape[1]))
        for name, value in fusion.get('statistics', {}).items():
            if torch.is_tensor(value) and value.numel() == 1:
                mechanisms['fusion_'+name] = float(value.detach().cpu())
        means = {name: fusion[name].detach().cpu().numpy().astype(np.float64) for name in (
            'component_increment_abs_mean_by_horizon_component', 'weighted_hidden_change_abs_mean_by_horizon')}
        require(means['component_increment_abs_mean_by_horizon_component'].shape == (12, 10)
                and means['weighted_hidden_change_abs_mean_by_horizon'].shape == (12,)
                and all(np.isfinite(value).all() for value in means.values()), 'Fusion feature diagnostic axes differ')
        return dict(prediction=prediction, target=target, mechanisms=mechanisms, weights=weights, fusion_means=means)
    finally:
        require(probe.input_digest(batch, events_override) == before, 'Fusion evaluation changed input arrays')


def update_weights(total, weights, means=None):
    """Equal sample-node weights at each horizon, independent of target validity."""
    count = weights.shape[0]*weights.shape[2]
    for name, value in (('sum', weights.sum((0, 2))), ('abs_delta_sum', np.abs(weights-1).sum((0, 2)))):
        total[name] = total.get(name, np.zeros((12, 10), np.float64))+value
    total['min'] = np.minimum(total.get('min', np.full((12, 10), np.inf)), weights.min((0, 2)))
    total['max'] = np.maximum(total.get('max', np.full((12, 10), -np.inf)), weights.max((0, 2)))
    total['count'] = total.get('count', 0)+count
    for name, value in (means or {}).items():
        total[name+'_sum'] = total.get(name+'_sum', np.zeros_like(value))+value*count


def weight_results(totals):
    return {case: dict(mean=(row['sum']/row['count']).tolist(),
        absolute_delta_from_one_mean=(row['abs_delta_sum']/row['count']).tolist(),
        minimum=row['min'].tolist(), maximum=row['max'].tolist(), count_per_horizon_component=row['count'],
        **{name.removesuffix('_sum'): (value/row['count']).tolist() for name, value in row.items()
           if name.endswith('_sum') and name not in ('abs_delta_sum',)})
        for case, row in totals.items()}


def summarize(predictions, target, valid, regions, indices):
    require(set(predictions) == set(CASES), 'Expected all eight fusion prediction cases')
    aliases = {name: predictions[actual] for name, actual in ALIASES.items()}
    result = old_metrics.summarize(aliases, target, valid, regions, indices, comparisons=COMPARISONS)
    result['schema'], result['status'] = 'min_forecast_fusion_metrics_v1', 'MIN_FORECAST_FUSION_ANALYSIS_PASS'
    result['cases'] = {ALIASES[name]: row for name, row in result['cases'].items()}
    fixed = result['fixed_hard_windows']
    fixed['selector'] = 'SOURCE_MIN'
    fixed['cases'] = {ALIASES[name]: row for name, row in fixed['cases'].items()}
    for row in result['comparisons'].values():
        row['control'], row['candidate'] = ALIASES[row['control']], ALIASES[row['candidate']]
    result['bias_reference'] = 'SOURCE_MIN_for_all_comparisons'
    result['forecast_scope'] = 'one_fusion_module_no_repeated_forecast_component_ablation'
    return result


def write_report(output, result, comparable, weights):
    fmt = lambda value: '不可用' if value is None else f'{value:.9f}'
    lines = ['# MIN 预测分量融合：配对暖启动评估', '',
        '工程检查：'+str(comparable['engineering_only'])+'；验证窗口：'+str(comparable['validation_rows'])+'。', '',
        ('本报告仅验证少量工程窗口上的训练、回放、干预和恢复流程，不构成正式预测效果结论。'
         if comparable['engineering_only'] else '完整验证集上的探索性评估；该验证集已用于模型设计和选模。'), '',
        'STATE 与 EVENT 从同一成熟 MIN 最佳权重暖启动，新增一个预测特征融合模块，主干与融合器联合训练。STATE 不读取事故输入；EVENT 额外使用已发布关系和类别。',
        '两组注册参数相同不等于有效参与训练的语义参数或报告支持范围相同。源 MIN 未重新训练；阶段0源模型参与最佳选模。', '',
        '| 状态 | MAE pooled | MAE macro | RMSE pooled | RMSE macro | MAPE (%) | 窗口 P95 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for case in CASES:
        m = result['cases'][case]['metrics']['all_nodes']
        lines.append('| '+' | '.join([case]+[fmt(m[key]) for key in ('mae_pooled', 'mae_macro', 'rmse_pooled', 'rmse_macro')]
            +[fmt(None if m['mape_pooled'] is None else 100*m['mape_pooled']), fmt(m['window_mae']['p95'])])+' |')
    lines += ['', '| 比较 | MAE收益 macro | 正收益 | 负收益 | 抵消比例 | 固定困难窗口收益 |', '|---|---:|---:|---:|---:|---:|']
    for name, row in result['comparisons'].items():
        m = row['metrics']['all_nodes']
        lines.append('| '+' | '.join([name]+[fmt(m[key]) for key in ('mae_gain_macro', 'gain_positive', 'gain_negative', 'cancellation_fraction')]
            +[fmt(row['fixed_hard_window_gain'])])+' |')
    primary = result['comparisons']['EVENT_vs_STATE']
    for title, family in (('道路与事故关联区域', 'metrics'), ('真实流量与源 MIN 偏差', 'strata')):
        lines += ['', '## '+title, '', '| 分组 | EVENT相对STATE的MAE收益 | 全网MAE贡献 |', '|---|---:|---:|']
        for name, m in primary[family].items():
            lines.append('| '+name+' | '+fmt(m['mae_gain_macro'])+' | '+fmt(m['global_gain_contribution'])+' |')
    lines += ['', '## 逐预测时距', '', '| 步 | SOURCE_MIN | STATE | EVENT | EVENT相对STATE收益 |', '|---|---:|---:|---:|---:|']
    for horizon in range(12):
        lines.append('| '+str(horizon+1)+' | '+' | '.join(fmt(result['cases'][case]['metrics']['all_nodes']['per_horizon_mae'][horizon])
            for case in ('SOURCE_MIN', 'STATE_ON', 'EVENT_ON'))+' | '+fmt(primary['metrics']['all_nodes']['per_horizon_mae_gain'][horizon])+' |')
    lines += ['', '## 融合权重与解释边界', '',
        'weights.csv 按状态、十二时距和十个部件汇总全部样本×节点的权重均值、范围及偏离1的程度；权重不等于重要性或预测误差贡献。',
        'EVENT_REPORT_OFF 只关闭报告内容，保留状态融合；EVENT_FUSION_OFF 关闭整个融合，保留当前训练后的 MIN，其权重不等于源 MIN。',
        '本版事故只条件化最终节点融合，图、十个预测分量及主干均不读取报告，未新增事故空间传播；同检查点报告内容效应在原支持外应为零。',
        'EVENT_UNIFORM 将同一检查点十个部件的权重取均值，只检查部件差异性，不等于独立训练的通用校准对照。',
        'SEMANTIC_UNK 使用注册的未知类别向量，并保留关系与支持；全站位置重排移动零行与报告支持，两种干预均可能产生分布外输入。',
        'SOURCE_MIN 只复用源完整回放归档，新增模型调用为0；不把归档一致性检查写成新的源模型回放。',
        '固定困难窗口和高低估分组均真正由 SOURCE_MIN 定义，所有状态共享；真值分层仅用于事后诊断，不能作为在线门控。',
        '源模型与新候选之间包含新增微调，不能把差异单独归因为结构；组间内容对照也不等于真实事故因果。',
        '完成阶段轮次：'+str(comparable['completed_epochs'])+'；最佳阶段轮次：'+str(comparable['best_stage_epochs'])+'；完成训练预算相同：'+str(comparable['same_completed_training_budget'])+'。',
        '窗口可能重叠，预测点不是独立事故重复；事故关联范围不是已认证事故影响范围。',
        '本次未训练、未评价测试集、未重复十二组 forecast 读出消融。MAPE 在本报告为百分数，在 JSON/CSV 为比例。',
        '制品：analysis.json、comparability.json、paired_predictions.npz、weights.csv、mechanisms.json、replay.json、preservation.json。', '']
    (output/'report.md').write_text('\n'.join(lines), encoding='utf-8')
    with (output/'weights.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['case', 'horizon', 'component', 'mean', 'absolute_delta_from_one_mean', 'minimum', 'maximum', 'sample_node_count'])
        for case, row in weights.items():
            for h in range(12):
                for k, component in enumerate(COMPONENT_ORDER):
                    writer.writerow([case, h+1, component]+[row[key][h][k] for key in ('mean', 'absolute_delta_from_one_mean', 'minimum', 'maximum')]
                        +[row['count_per_horizon_component']])
    with (output/'horizons.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['case', 'horizon', 'mae', 'rmse', 'mape'])
        for case, row in result['cases'].items():
            m = row['metrics']['all_nodes']
            for h in range(12):
                writer.writerow([case, h+1]+[m[key][h] for key in ('per_horizon_mae', 'per_horizon_rmse', 'per_horizon_mape')])
    for filename, family in (('regions.csv', 'metrics'), ('strata.csv', 'strata')):
        with (output/filename).open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['comparison', 'group', 'valid_count', 'mae_gain_macro', 'global_gain_contribution', 'gain_positive', 'gain_negative'])
            for name, row in result['comparisons'].items():
                for group, m in row[family].items():
                    writer.writerow([name, group]+[m[key] for key in ('valid_count', 'mae_gain_macro', 'global_gain_contribution', 'gain_positive', 'gain_negative')])


def run_probe(models, data, saved, graph, roads, output, baseline, baseline_arrays, protected_paths):
    output = Path(output)
    require(not output.exists(), 'Use a fresh fusion evaluation directory')
    output.mkdir(parents=True)
    before = probe.snapshot_files(protected_paths)
    predictions, replay, mechanisms, preservation, calls, weight_totals = {}, {}, {}, {}, {}, {}
    try:
        require(models.keys() == saved.keys() == {'STATE', 'EVENT'}, 'Both STATE and EVENT best models are required')
        histories = {arm: value['history'] for arm, value in saved.items()}
        require(all(value['completed_epoch'] == len(histories[arm]) and len(histories[arm]) > 0 for arm, value in saved.items()),
                'Invalid completed training budgets')
        common = min(len(value) for value in histories.values())
        for epoch in range(common):
            hashes = [histories[arm][epoch].get('train_order_sha256') for arm in histories]
            require(all(isinstance(value, str) and len(value) == 64 for value in hashes) and len(set(hashes)) == 1,
                    'Paired common-prefix sample order differs')
        comparable = dict(completed_epochs={arm: value['completed_epoch'] for arm, value in saved.items()},
            engineering_only=baseline['engineering_only'], validation_rows=baseline['validation_rows'],
            global_updates={arm: value['global_updates'] for arm, value in saved.items()},
            best_stage_epochs={arm: value['best_epoch'] for arm, value in saved.items()}, common_stage_epoch=common,
            source_best_epoch=baseline['source_best_epoch'],
            same_completed_training_budget=len({value['completed_epoch'] for value in saved.values()}) == 1
                and len({value['global_updates'] for value in saved.values()}) == 1,
            source_exact_resume=False, optimizer_and_scheduler_reset_at_stage_start=True, existing_MIN_retrained=False,
            registered_parameter_caveat='equal_registered_size_does_not_imply_equal_active_semantic_parameters_or_support')
        target, valid, indices = [np.asarray(baseline_arrays[key]) for key in ('target', 'valid', 'sample_indices')]
        require(target.ndim == 3 and target.shape[1:] == (12, len(data.station_ids)) and target.dtype.kind == 'f', 'Source target axes differ')
        require(len(target) == data.loaders['val_loader'].size and np.array_equal(baseline_arrays['station_ids'], data.station_ids),
                'Saved source-reference sample/station axes differ')
        expected_indices = data.audit['counts']['train_samples']+np.arange(len(target), dtype=np.int64)
        require(np.array_equal(indices, expected_indices), 'Saved source-reference original sample indices differ')
        require(valid.dtype == np.bool_ and valid.shape == target.shape and np.array_equal(valid, target != baseline['mask_value']),
                'Source reference original mask differs')
        require(baseline_arrays['prediction'].shape == target.shape and np.isfinite(baseline_arrays['prediction']).all(), 'Source prediction axes differ')
        cached = compute_all_metrics(torch.from_numpy(baseline_arrays['prediction']), torch.from_numpy(target), torch.tensor(baseline['mask_value']))
        require(np.allclose(cached, baseline['actual'], atol=1e-5, rtol=1e-6), 'Cached source metrics differ')
        rows = [encode_event(sample, len(data.station_ids)) for sample in data.loaders['val_loader'].original_samples]
        events = {key: np.asarray([row[key] for row in rows], dtype=np.float32 if key == 'relation'
            else np.int64 if key in ('type', 'description') else bool) for key in EVENT_KEYS}
        shuffled, donors, shuffle_audit = early_evaluation.shuffled_locations(events, indices, 20261015)
        regions = probe.regions_for(events, graph, roads)
        model_hashes = {arm: tensor_state_hash(model.state_dict()) for arm, model in models.items()}
        with ExitStack() as stack:
            for arm, model in models.items():
                require(model_hashes[arm] == tensor_state_hash(saved[arm]['best_model_state']), 'Model is not saved selected best')
                stack.enter_context(probe.preserve_model(model, preservation, arm))
            for arm, model in models.items():
                case, position, partitions = arm+'_ON', 0, []
                predictions[case] = np.empty_like(target)
                for batch in data.loaders['val_loader'].get_iterator():
                    row = forward_case(model, batch, data.scaler, case)
                    end = position+len(row['prediction'])
                    require(end <= len(target) and np.array_equal(row['target'], target[position:end]), 'Targets differ from immutable source reference')
                    predictions[case][position:end] = row['prediction']
                    probe.merge_mechanisms(mechanisms.setdefault(case, {}), row['mechanisms'])
                    update_weights(weight_totals.setdefault(case, {}), row['weights'], row['fusion_means'])
                    calls[case] = calls.get(case, 0)+1
                    partitions.append(end-position)
                    position = end
                require(position == len(target) and partitions == baseline['batch_partition'], 'Original validation batch partition changed')
                actual = compute_all_metrics(torch.from_numpy(predictions[case]), torch.from_numpy(target), torch.tensor(baseline['mask_value']))
                expected = selected_metrics(saved[arm], baseline)
                require(np.allclose(actual, expected, atol=1e-5, rtol=1e-6), arm+' selected best ON replay failed')
                replay[arm] = dict(status='BEST_ON_REPLAY_PASS', best_stage_epoch=saved[arm]['best_epoch'], actual=actual,
                    expected=expected, mask_value=baseline['mask_value'], batch_partition=partitions, atol=1e-5, rtol=1e-6)
            # Both complete ON replays passed; only now permit interventions.
            position, model = 0, models['EVENT']
            for batch in data.loaders['val_loader'].get_iterator():
                end = position+len(batch['x_data'])
                for case in CASES[2:-1]:
                    changed = {key: value[position:end] for key, value in shuffled.items()} if case.endswith('SHUFFLE') else None
                    row = forward_case(model, batch, data.scaler, case, changed)
                    require(np.array_equal(row['target'], target[position:end]), 'Intervention target axes differ')
                    predictions.setdefault(case, np.empty_like(target))[position:end] = row['prediction']
                    probe.merge_mechanisms(mechanisms.setdefault(case, {}), row['mechanisms'])
                    update_weights(weight_totals.setdefault(case, {}), row['weights'], row['fusion_means'])
                    calls[case] = calls.get(case, 0)+1
                    restored = forward_case(model, batch, data.scaler, 'EVENT_ON')
                    calls['EVENT_ON'] += 1
                    require(np.array_equal(restored['prediction'], predictions['EVENT_ON'][position:end]), 'EVENT ON restoration failed')
                position = end
            require(position == len(target), 'Incomplete fusion interventions')
        require(all(tensor_state_hash(model.state_dict()) == model_hashes[arm] for arm, model in models.items()), 'Fusion model changed during evaluation')
        predictions['SOURCE_MIN'] = baseline_arrays['prediction']
        result = summarize(predictions, target, valid, regions, indices)
        result['training_comparability'] = comparable
        result['comparisons']['EVENT_vs_STATE']['scope'] = ('paired_warmstart_equal_completed_stage_budget_independent_best'
            if comparable['same_completed_training_budget'] else 'unequal_completed_stage_budget_exploratory_independent_best')
        weights = weight_results(weight_totals)
        region_keys = {name: 'region_'+str(i) for i, name in enumerate(regions)}
        np.savez_compressed(output/'paired_predictions.npz', **predictions, target=target, valid=valid,
            sample_indices=indices, station_ids=np.asarray(data.station_ids, np.int64), location_donor_station_indices=donors,
            **{region_keys[name]: mask for name, mask in regions.items()})
        training.legacy.write_json(output/'region_definitions.json', dict(masks=region_keys,
            direct='original_author_relation_support_not_certified_incident_effect',
            potential='weak_component_envelope_not_certified_physical_propagation'))
        for name, value in (('analysis.json', result), ('comparability.json', comparable), ('replay.json', replay),
            ('mechanisms.json', dict(cases=mechanisms, component_order=COMPONENT_ORDER, fusion_weights=weights,
                weight_scope='all_original_sample_node_positions_not_importance', source_weights='unmeasured_cached_predictions_only',
                location_shuffle=shuffle_audit, semantic_unknown='registered_UNK_tokens_validity_flags_false_relation_support_preserved'))):
            training.legacy.write_json(output/name, value)
        write_report(output, result, comparable, weights)
        files = probe.require_preserved(before, protected_paths)
        training.legacy.write_json(output/'preservation.json', dict(files=files, models=preservation))
        calls['SOURCE_MIN'] = 0
        summary = dict(schema=SCHEMA, status='MIN_FORECAST_FUSION_EVALUATION_PASS', validation_rows=len(target),
            engineering_only=baseline['engineering_only'], best_stage_epochs=comparable['best_stage_epochs'],
            source_best_epoch=baseline['source_best_epoch'], cases=list(predictions),
            all_nodes={name: row['metrics']['all_nodes'] for name, row in result['cases'].items()},
            comparisons={name: row['metrics']['all_nodes'] for name, row in result['comparisons'].items()},
            replay_passed=True, on_exactly_restored=True, training_comparability=comparable,
            source_artifacts_preserved=True, new_inference_calls=calls, optimizer_updates=0, test_evaluated=False,
            source_scope='conditional_offline_validation_warmstart_checkpoint_sensitivity', output_dir=str(output))
        training.legacy.write_json(output/'summary.json', summary)
        return summary
    except BaseException as error:
        training.legacy.write_json(output/'failure.json', dict(status='MIN_FORECAST_FUSION_EVALUATION_FAILED',
            error_type=type(error).__name__, error=str(error), new_inference_calls=calls, optimizer_updates=0, test_evaluated=False))
        after = probe.snapshot_files(protected_paths)
        training.legacy.write_json(output/'preservation.json', dict(source_artifacts_preserved=before == after,
            before=before, after=after, models=preservation, evaluation_completed=False))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--source-run', type=Path, default=training.DEFAULT_SOURCE)
    parser.add_argument('--data-dir', type=Path, default=training.paper.default_data())
    parser.add_argument('--graph-dir', type=Path, default=training.HERE/'internal_spatial_graph_is11')
    parser.add_argument('--protocol', type=Path, default=training.PROTOCOL)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--threads', type=int)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args(argv)
    for name in ('run_dir', 'source_run', 'data_dir', 'graph_dir', 'protocol'):
        setattr(args, name, getattr(args, name).resolve())
    identity = training.legacy.read_json(args.run_dir/'identity.json')
    require(identity.get('schema') == training.SCHEMA, 'Expected a forecast fusion warmstart run')
    require(args.check is identity['engineering_check'], 'Engineering marker differs from training run')
    args.threads = identity['threads'] if args.threads is None else args.threads
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    setup = SimpleNamespace(**dict(vars(args), arms=list(identity['arm_modes']), batch_size=identity['protocol']['batch_size'],
        seed=identity['seed'], epochs=1, only=None, worker=None, evaluate_after=False))
    logger = logging.getLogger('min_forecast_fusion_evaluation')
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler(sys.stdout))
    with training.legacy.run_lock(args.run_dir), training.readonly_source_lock(args.source_run):
        protocol, data, graph, initialization, current, source = training.prepare(setup, logger)
        require(current == identity, 'Fusion training identity changed')
        baseline, baseline_arrays = training.load_source_reference(setup, identity)
        _, roads, _ = training.legacy.load_graph(args.graph_dir, data.station_ids, protocol['graph_sha256'])
        saved, models = {}, {}
        for arm in ('STATE', 'EVENT'):
            path = args.run_dir/arm/'last.pt'
            require(path.is_file(), 'Both completed STATE and EVENT arms are required')
            saved[arm] = torch.load(path, map_location='cpu', weights_only=False)
            training.validate_saved(saved[arm], identity, arm)
            require(training.legacy.read_json(args.run_dir/arm/'history.json') == saved[arm]['history'], 'History sidecar differs from checkpoint')
            companion = training.legacy.read_json(args.run_dir/arm/'summary.json')
            require(all(companion.get(key) == value for key, value in dict(arm=arm, mode=training.ARMS[arm],
                completed_epoch=saved[arm]['completed_epoch'], best_epoch=saved[arm]['best_epoch'], best_metric=saved[arm]['best_metric']).items()),
                'Summary sidecar differs from checkpoint')
            models[arm], _ = training.build_arm(setup, arm, len(data.station_ids), graph, torch.device(args.device), source)
            models[arm].load_state_dict(saved[arm]['best_model_state'], strict=True)
            models[arm].eval()
        paths = {'run/'+name: args.run_dir/name for name in ('identity.json', 'initialization.json', 'source_reference.json', 'source_reference.npz')}
        paths.update({'source/'+name: args.source_run/name for name in training.SOURCE_FILES})
        for arm in saved:
            paths.update({arm+'/'+name: args.run_dir/arm/name for name in ('last.pt', 'history.json', 'summary.json')})
        for key, directory in (('source_sha256', REPO), ('package', args.data_dir), ('graph_inputs_sha256', args.graph_dir)):
            paths.update({key+'/'+name: directory/name for name in identity[key]})
        paths['evaluation_source'], paths['protocol'] = Path(__file__).resolve(), args.protocol
        args.output_dir = args.output_dir.resolve() if args.output_dir else args.run_dir/('forecast_fusion_evaluation_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6])
        require(args.output_dir != args.run_dir and args.output_dir not in args.run_dir.parents
            and args.output_dir != args.source_run and args.source_run not in args.output_dir.parents,
            'Evaluation output must be a fresh directory outside the original source run')
        require(all(args.output_dir != args.run_dir/arm and args.run_dir/arm not in args.output_dir.parents for arm in training.ARMS),
            'Evaluation output must not be inside a checkpoint directory')
        print('Evaluation directory: '+str(args.output_dir), flush=True)
        result = run_probe(models, data, saved, graph, roads, args.output_dir, baseline, baseline_arrays, paths)
        training.legacy.write_json(args.output_dir/'provenance.json', dict(training_identity=identity,
            evaluation_source_sha256={str(Path(__file__).relative_to(REPO)): sha256(__file__)},
            source_reference_reused=True, existing_MIN_retrained=False, test_evaluated=False))
        print(json.dumps({key: result[key] for key in ('status', 'validation_rows', 'best_stage_epochs', 'optimizer_updates', 'output_dir')}, indent=2, allow_nan=False), flush=True)
    gc.collect()
    return result


if __name__ == '__main__':
    main()
