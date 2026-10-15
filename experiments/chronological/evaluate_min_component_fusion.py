"""F2 best-checkpoint validation: three trained controls and five interventions.

Only saved current models run. SOURCE_MIN predictions are reused unchanged;
there is no training, test-loader iteration, or initialization preflight here.
"""
import argparse
from contextlib import ExitStack
import csv
from datetime import datetime
import gc
import json
import logging
import math
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
from experiments.chronological import train_min_component_fusion as training
from experiments.chronological import evaluate_min_forecast_fusion as f1
from experiments.chronological import evaluate_min_early_event as early
from experiments.chronological import probe_min_event_budget as probe
from experiments.chronological import min_early_event_gain_metrics as gm
from experiments.chronological.min_forecast_fusion_gain_metrics import _sum_plan, _pairwise_bincount
from experiments.chronological.min_event_budget_probe_metrics import window_scores, tail_summary
from experiments.chronological.smoke import sha256
from src.models.incident_spatial import require
from src.models.incident_spatial_residual import tensor_state_hash
from src.utils.igstgnn_published_events import EVENT_KEYS, encode_event
from src.utils.metrics import compute_all_metrics

SCHEMA = 'min_component_content_fusion_evaluation_v1'
CASES = ('CONTROL_ON', 'STATE_ON', 'EVENT_ON', 'EVENT_REPORT_OFF', 'EVENT_FUSION_OFF',
         'EVENT_CONTENT_OFF', 'EVENT_SEMANTIC_UNK', 'EVENT_LOCATION_SHUFFLE', 'SOURCE_MIN')
ON_CASES = CASES[:3]
INTERVENTIONS = CASES[3:-1]
COMPONENT_ORDER = f1.COMPONENT_ORDER
COMPARISONS = {
    'STATE_vs_CONTROL': ('CONTROL_ON', 'STATE_ON', 'same_budget_new_fusion_and_training_joint_effect_not_content_only'),
    'EVENT_vs_STATE': ('STATE_ON', 'EVENT_ON', 'same_budget_report_conditioning_independent_best_models'),
    'EVENT_vs_CONTROL': ('CONTROL_ON', 'EVENT_ON', 'same_budget_complete_candidate_effect_not_incident_causality'),
    'fusion_dependency': ('EVENT_FUSION_OFF', 'EVENT_ON', 'same_checkpoint_entire_fusion_dependency'),
    'report_content_dependency': ('EVENT_REPORT_OFF', 'EVENT_ON', 'same_checkpoint_report_context_dependency'),
    'component_content_dependency': ('EVENT_CONTENT_OFF', 'EVENT_ON', 'same_checkpoint_component_content_OFF_potential_OOD_not_retrained_deletion'),
    'semantic_UNK_sensitivity': ('EVENT_SEMANTIC_UNK', 'EVENT_ON', 'same_checkpoint_UNK_potential_OOD'),
    'location_shuffle_sensitivity': ('EVENT_LOCATION_SHUFFLE', 'EVENT_ON', 'same_checkpoint_full_station_shuffle_potential_OOD'),
    'CONTROL_vs_SOURCE': ('SOURCE_MIN', 'CONTROL_ON', 'pure_MIN_additional_finetuning_not_new_MIN_training'),
    'EVENT_vs_SOURCE': ('SOURCE_MIN', 'EVENT_ON', 'source_to_candidate_includes_extra_training'),
}
CODE_SOURCES = ('experiments/chronological/evaluate_min_component_fusion.py',
    'experiments/chronological/evaluate_min_forecast_fusion.py',
    'experiments/chronological/evaluate_min_early_event.py',
    'experiments/chronological/min_early_event_gain_metrics.py',
    'experiments/chronological/min_forecast_fusion_gain_metrics.py',
    'experiments/chronological/min_event_budget_probe_metrics.py',
    'experiments/chronological/probe_min_event_budget.py', 'src/utils/metrics.py')


def forward_case(model, batch, scaler, case, events_override=None):
    require(case in CASES[:-1], 'Unknown F2 inference case')
    expected_mode = {'CONTROL_ON': None, 'STATE_ON': 'state'}.get(case, 'event')
    require(getattr(model, '_forecast_fusion_mode', None) == expected_mode, 'Inference case belongs to another model arm')
    before = probe.input_digest(batch, events_override)
    try:
        device = next(model.parameters()).device
        x = torch.as_tensor(batch['x_data'], device=device, dtype=torch.float32)
        y = torch.as_tensor(batch['y_data'], device=device, dtype=torch.float32)
        event_data = batch['event_data'] if events_override is None else events_override
        if case == 'EVENT_SEMANTIC_UNK':
            event_data = f1.semantic_unknown(event_data)
        with torch.inference_mode():
            if case == 'CONTROL_ON':
                result = model(x, return_spatial_details=True)
            else:
                event = None if case == 'STATE_ON' else training.event_tensors(event_data, device)
                result = model(x, incident_data=event,
                    report_enabled=case != 'EVENT_REPORT_OFF', fusion_enabled=case != 'EVENT_FUSION_OFF',
                    content_enabled=case != 'EVENT_CONTENT_OFF', semantic_enabled=True, return_spatial_details=True)
            prediction = scaler.inverse_transform(result['prediction']).squeeze(-1).cpu().numpy().copy()
            target = scaler.inverse_transform(y).squeeze(-1).cpu().numpy().copy()
        require(prediction.shape == target.shape and np.isfinite(prediction).all() and np.isfinite(target).all(), 'Nonfinite or mismatched F2 output')
        mechanisms = early.small_mechanisms(result['spatial'])
        if case == 'CONTROL_ON':
            return dict(prediction=prediction, target=target, mechanisms=mechanisms, weights=None, fusion_means={})
        fusion = result['fusion']
        require(fusion['schema'] == 'MIN_component_content_relative_fusion_v1'
            and tuple(fusion['component_order']) == COMPONENT_ORDER
            and fusion['coefficient_policy'] == 'centered_relative_uniformly_bounded', 'Wrong F2 fusion details')
        weights = fusion['weights'].detach().cpu().numpy().astype(np.float64)
        require(weights.shape == prediction.shape+(10,) and np.isfinite(weights).all(), 'F2 coefficient axes differ')
        require(np.max(np.abs((weights-1).sum(-1))) <= 2e-6
            and np.max(np.abs(weights-1)) <= fusion['max_delta']+2e-7, 'F2 coefficient constraints failed')
        support = fusion['support'].detach().cpu().numpy()
        require(support.dtype == np.bool_ and support.shape == (len(prediction), prediction.shape[2], 1), 'F2 support axes differ')
        require(fusion['enabled'] is (case != 'EVENT_FUSION_OFF')
            and fusion['component_content_enabled'] is (case not in ('EVENT_FUSION_OFF', 'EVENT_CONTENT_OFF')),
            'Wrong F2 intervention executed')
        if case == 'EVENT_REPORT_OFF':
            require(not support.any() and not fusion['event_context_used'], 'Report OFF left report context enabled')
        mechanisms.update(fusion_enabled_calls=int(fusion['enabled']),
            fusion_event_context_calls=int(fusion['event_context_used']),
            fusion_component_content_calls=int(fusion['component_content_enabled']),
            fusion_supported_node_positions=int(support.sum()), fusion_node_positions=int(support.shape[0]*support.shape[1]))
        for name, value in fusion.get('statistics', {}).items():
            if torch.is_tensor(value) and value.numel() == 1:
                mechanisms['fusion_'+name] = float(value.detach().cpu())
        means = {name: fusion[name].detach().cpu().numpy().astype(np.float64) for name in (
            'component_increment_abs_mean_by_horizon_component', 'weighted_hidden_change_abs_mean_by_horizon')}
        for name in ('component_encoding_abs_mean_by_component','component_contrast_abs_mean_by_component',
                     'component_log_rms_mean_by_component'):
            if name in fusion:
                means[name]=fusion[name].detach().cpu().numpy().astype(np.float64)
                require(means[name].shape==(10,),'F2 compact component diagnostic axes differ')
        require(means['component_increment_abs_mean_by_horizon_component'].shape == (12, 10)
            and means['weighted_hidden_change_abs_mean_by_horizon'].shape == (12,)
            and all(np.isfinite(value).all() for value in means.values()), 'F2 feature diagnostic axes differ')
        return dict(prediction=prediction, target=target, mechanisms=mechanisms, weights=weights, fusion_means=means)
    finally:
        require(probe.input_digest(batch, events_override) == before, 'F2 evaluation changed input arrays')


def merge_mechanisms(total, row):
    for name, value in row.items():
        require(np.isfinite(value), 'Nonfinite F2 mechanism statistic')
        if name.endswith('_min'):
            total[name] = min(total.get(name, value), value)
        elif name.endswith('_max'):
            total[name] = max(total.get(name, value), value)
        else:
            total[name] = total.get(name, 0)+value


def summarize(predictions, target, valid, regions, indices, station_ids):
    """Arbitrary-case metrics with SOURCE hard set and current-backbone bias.

    Uses immutable NumPy contracts/summary helpers with locally defined cases;
    no monkeypatching, duplicate dummy predictions, or model calls are needed.
    """
    require(set(predictions) == set(CASES), 'Expected all nine F2 prediction cases')
    aliases = {name: predictions[actual] for name, actual in (
        ('STATE_ON', 'EVENT_FUSION_OFF'), ('EVENT_ON', 'EVENT_ON'), ('EVENT_OFF', 'EVENT_REPORT_OFF'),
        ('EVENT_SEMANTIC_UNK', 'EVENT_SEMANTIC_UNK'), ('EVENT_LOCATION_SHUFFLE', 'EVENT_LOCATION_SHUFFLE'))}
    _, target, valid, masks, indices, stations, _, edges, response_tol = gm._validate(
        aliases, target, valid, regions, indices, station_ids, np.array([], np.int64), gm.FLOW_EDGES, .001)
    arrays = {case: np.asarray(predictions[case]) for case in CASES}
    require(all(a.shape == target.shape and a.dtype.kind in 'fiu' and np.isfinite(a[valid]).all() for a in arrays.values()), 'F2 prediction axes differ')
    rows, horizons, _ = target.shape
    counts = valid.sum((0, 2)).astype(np.int64)
    positive_counts = (valid & (target > 0)).sum((0, 2)).astype(np.int64)
    window_counts = valid.sum(2).astype(np.int64)
    window_errors = {case: np.zeros((rows, horizons)) for case in CASES}
    for h in range(horizons):
        active = valid[:, h]
        row_ids = np.broadcast_to(np.arange(rows)[:, None], active.shape)[active]
        plan = _sum_plan(row_ids, rows)
        for case in CASES:
            window_errors[case][:, h] = _pairwise_bincount(np.abs(arrays[case][:, h][active].astype(np.float64)-target[:, h][active]), plan)
    scores = {case: window_scores(error, window_counts) for case, error in window_errors.items()}
    eligible = np.flatnonzero(np.isfinite(scores['SOURCE_MIN']))
    ordering = np.lexsort((indices[eligible], -scores['SOURCE_MIN'][eligible]))
    chosen = eligible[ordering[:max(1, math.ceil(.1*len(eligible)))]] if len(eligible) else np.array([], np.int64)
    fixed_rows = np.zeros(rows, bool)
    fixed_rows[chosen] = True
    names, definitions = list(masks), {name: dict(family='road' if name.startswith('road:') else 'region') for name in masks}
    flow_names = ['flow:'+gm._label(lo, edges[i+1] if i+1 < len(edges) else None) for i, lo in enumerate(edges)]
    bias_names = ['bias:'+bias for bias in gm.BIASES]
    for family, members in (('flow', flow_names), ('current_MIN_bias', bias_names), ('fixed_SOURCE_MIN_hard', ['fixed_hard_windows'])):
        for name in members:
            require(name not in definitions, 'Reserved F2 group conflicts with region')
            names.append(name)
            definitions[name] = dict(family=family)
    accumulators = {key: {field: np.zeros((len(names), horizons), np.int64 if field in gm.COUNT_FIELDS else np.float64)
        for field in (*gm.SUM_FIELDS, *gm.COUNT_FIELDS, 'prediction_max')} for key in COMPARISONS}
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        for h in range(horizons):
            active = valid[:, h]
            y = target[:, h][active].astype(np.float64)
            values = {case: array[:, h][active].astype(np.float64) for case, array in arrays.items()}
            errors = {case: np.abs(value-y) for case, value in values.items()}
            percentages = {case: np.divide(error, y, out=np.zeros_like(y), where=y > 0) for case, error in errors.items()}
            flow_ids = np.searchsorted(edges, y, side='right')-1
            bias = values['EVENT_FUSION_OFF']-y
            selections = [mask[active] for mask in masks.values()]
            selections += [flow_ids == i for i in range(len(edges))]+[bias > 0, bias < 0, bias == 0]
            selections += [np.broadcast_to(fixed_rows[:, None], active.shape)[active]]
            for key, (left, right, _) in COMPARISONS.items():
                change, gain = values[right]-values[left], errors[left]-errors[right]
                sums = dict(control_absolute=errors[left], candidate_absolute=errors[right],
                    control_squared=np.square(errors[left]), candidate_squared=np.square(errors[right]),
                    control_percentage=percentages[left], candidate_percentage=percentages[right],
                    gain_net=gain, gain_positive=np.maximum(gain, 0), gain_negative=np.maximum(-gain, 0),
                    prediction_signed=change, prediction_absolute=np.abs(change))
                fields = dict(count=np.ones(len(y), bool), positive_target_count=y > 0,
                    prediction_up=change > 0, prediction_down=change < 0, prediction_unchanged=change == 0,
                    prediction_response=np.abs(change) > response_tol, gain_improved=gain > 0,
                    gain_worsened=gain < 0, gain_equal=gain == 0)
                acc = accumulators[key]
                for group, mask in enumerate(selections):
                    plan = _sum_plan(np.zeros(np.count_nonzero(mask), np.int64), 1)
                    for field, value in sums.items():
                        acc[field][group, h] = _pairwise_bincount(value[mask], plan)[0]
                    for field, value in fields.items():
                        acc[field][group, h] = np.count_nonzero(value[mask])
                    acc['prediction_max'][group, h] = np.max(np.abs(change[mask]), initial=0.)
    comparisons, cases = {}, {}
    for key, (left, right, scope) in COMPARISONS.items():
        groups = {name: gm._summary(accumulators[key], i, counts, positive_counts) for i, name in enumerate(names)}
        for case, side in ((left, 'control'), (right, 'candidate')):
            if case not in cases:
                metrics = {name: row[side] for name, row in groups.items() if name in masks}
                metrics['all_nodes']['window_mae'] = tail_summary(scores[case])
                cases[case] = dict(metrics=metrics)
        comparisons[key] = dict(control=left, candidate=right, scope=scope,
            metrics={name: row for name, row in groups.items() if name in masks},
            strata={name: row for name, row in groups.items() if name not in masks},
            fixed_hard_window_gain=float((scores[left][chosen]-scores[right][chosen]).mean()) if len(chosen) else None)
        acc, global_id = accumulators[key], names.index('all_nodes')
        for family in (gm.PARTITION, flow_names, bias_names):
            positions = [names.index(name) for name in family]
            for field in (*gm.SUM_FIELDS, *gm.COUNT_FIELDS):
                total, expected = acc[field][positions].sum(0), acc[field][global_id]
                # Signed cancellation can make the net tiny while the summed
                # magnitudes are large. Scale by the arithmetic inputs, as in
                # the immutable F1 analyzer; never relax count conservation.
                scale = acc['control_absolute'][global_id]+acc['candidate_absolute'][global_id]+1.
                if field == 'prediction_signed':
                    scale = acc['prediction_absolute'][global_id]+1.
                elif field not in ('gain_net',):
                    scale = np.maximum(np.abs(total), np.abs(expected))+1.
                tolerance = 1e-10+128*np.finfo(np.float64).eps*scale
                require(np.array_equal(total, expected) if field in gm.COUNT_FIELDS else np.all(np.abs(total-expected) <= tolerance),
                    'F2 family conservation failed: '+key+' '+str(family)+' '+field)
    require(set(cases) == set(CASES), 'A case has no metric comparison')
    fixed = dict(selector='SOURCE_MIN', selection_origin='deterministic_top10_percent_same_immutable_SOURCE_MIN_predictions',
        fraction=.1, sample_indices=indices[chosen].tolist(), selected_count=len(chosen),
        tie_break='ascending_original_sample_id', cases={case: dict(window_mean_mae=float(scores[case][chosen].mean()) if len(chosen) else None) for case in CASES})
    return dict(schema=SCHEMA, status='MIN_COMPONENT_FUSION_ANALYSIS_PASS', cases=cases, comparisons=comparisons,
        fixed_hard_windows=fixed, validation_rows=rows, horizons=horizons, stations=len(stations), group_definitions=definitions,
        gain_sign='control_error_minus_candidate_error', bias_reference='EVENT_FUSION_OFF_for_all_comparisons',
        flow_edges=edges.tolist(), mape_unit='fraction', response_tolerance=response_tol,
        primary_baseline='original_IGSTGNN_not_present_in_this_archive',
        scope='exploratory_validation_same_checkpoint_dependency_not_incident_causality')


def write_report(output, result, comparable, weights):
    fmt = lambda value: '不可用' if value is None else f'{value:.9f}'
    lines = ['# MIN-F2 分量内容感知融合：三组配对评估', '',
        '原 IGSTGNN 是论文主 baseline；本归档不包含原 IGSTGNN，不能据此证明超过它。', '',
        '工程检查：'+str(comparable['engineering_only'])+'；验证窗口：'+str(result['validation_rows'])+'。',
        ('少量工程窗口只检验流程，不构成预测效果结论。' if comparable['engineering_only'] else '完整开发验证集上的探索性诊断；验证集已用于设计和选模。'), '',
        'CONTROL 为同一成熟 MIN 最佳权重的纯主干额外微调；STATE 和 EVENT 从相同权重开始，联合训练主干及新增融合器。源 MIN 没有从头重训。',
        'F2 同时改变路由输入、共享评分器及零和系数参数化；STATE_vs_CONTROL 衡量这些变化与联合训练的整体差异，不能单独归因为分量内容感知。', '',
        '| 状态 | MAE pooled | MAE macro | RMSE pooled | RMSE macro | MAPE (%) | 窗口 P95 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    if not comparable['same_completed_training_budget']:
        lines[7:7]=['三组完成训练预算不相等；各比较的实际预算匹配状态见 analysis.json，预算不同的模型差异只能探索性描述。','']
    for case in CASES:
        m = result['cases'][case]['metrics']['all_nodes']
        lines.append('| '+' | '.join([case]+[fmt(m[key]) for key in ('mae_pooled','mae_macro','rmse_pooled','rmse_macro')]
            +[fmt(None if m['mape_pooled'] is None else m['mape_pooled']*100), fmt(m['window_mae']['p95'])])+' |')
    lines += ['', '| 比较 | MAE收益 macro | RMSE收益 macro | MAPE收益（百分点） | 正收益 macro | 负收益 macro | 抵消比例 | 固定困难窗口收益 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name, row in result['comparisons'].items():
        m = row['metrics']['all_nodes']
        lines.append('| '+' | '.join([name, fmt(m['mae_gain_macro']),fmt(m['rmse_gain_macro']),
            fmt(None if m['mape_gain_macro'] is None else m['mape_gain_macro']*100)]+[fmt(m[k]) for k in ('gain_positive_macro','gain_negative_macro','cancellation_macro')]
            +[fmt(row['fixed_hard_window_gain'])])+' |')
    for title, family in (('道路与事故关联区域', 'metrics'), ('真实流量与当前 EVENT 主干偏差', 'strata')):
        lines += ['', '## '+title, '', '| 比较 | 分组 | MAE收益 pooled | 全网MAE贡献 pooled |', '|---|---|---:|---:|']
        for name in ('STATE_vs_CONTROL','EVENT_vs_STATE','fusion_dependency','report_content_dependency','component_content_dependency'):
            for group, row in result['comparisons'][name][family].items():
                lines.append('| '+name+' | '+group+' | '+fmt(row['mae_gain_pooled'])+' | '+fmt(row['global_mae_contribution_pooled'])+' |')
    lines += ['', '## 逐预测时距', '', '| 步 | SOURCE_MIN | CONTROL | STATE | EVENT | 融合收益 | 内容读取依赖 | 报告依赖 |', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for h in range(12):
        lines.append('| '+str(h+1)+' | '+' | '.join([fmt(result['cases'][case]['metrics']['all_nodes']['per_horizon_mae'][h])
            for case in ('SOURCE_MIN','CONTROL_ON','STATE_ON','EVENT_ON')]+[fmt(result['comparisons'][name]['metrics']['all_nodes']['per_horizon_mae_gain'][h])
            for name in ('fusion_dependency','component_content_dependency','report_content_dependency')])+' |')
    lines += ['', '## 解释边界与来源', '',
        '收益为对照误差减候选误差，负消融完整保留。MAPE 在本报告为百分数，在 JSON/CSV 为比例。',
        'EVENT_CONTENT_OFF 仅在同检查点关闭路由器对分量内容的读取，保留各分量、原主干及状态/时距/报告上下文；可能分布外，不证明删掉后重新训练仍退化。',
        'EVENT_REPORT_OFF 只关闭报告；EVENT_FUSION_OFF 保留当前联合训练后的主干固定求和，既不是 CONTROL，也不是 SOURCE_MIN。',
        '事故只条件化最终节点融合，图与主干不读取报告；未新增事故空间传播。报告效应在原支持外须严格为零。',
        'SEMANTIC_UNK 使用注册未知向量；全站位置重排移动零行及支持，均为敏感性干预，可能产生分布外输入。',
        '固定困难窗口由相同 SOURCE_MIN 归档预测按原规则确定，不随新候选更换；偏差分组统一由当前 EVENT_FUSION_OFF 定义。',
        '真值流量和高低估为事后诊断，不作为在线输入。区域是关系支持/弱连通候选包络，不是认证事故影响范围。',
        'weights.csv 使用全部样本×节点等权，权重不等于重要性、误差贡献或物理容量。无量纲受限交换不等于已识别车辆通行能力。',
        '完成阶段轮次：'+str(comparable['completed_epochs'])+'；最佳阶段：'+str(comparable['best_stage_epochs'])+'；相同完成预算：'+str(comparable['same_completed_training_budget'])+'。',
        'SOURCE_MIN 完全复用归档，新增源模型调用为0。阶段0参加选模；本评估没有初始化前向、训练、测试集评价或重复十二组读出消融。',
        '窗口可重叠，预测点不是独立事故重复。模型间差异、同检查点路径依赖及真实事故因果不可互换。', '',
        '制品：analysis.json、comparability.json、paired_predictions.npz、weights.csv、mechanisms.json、replay.json、preservation.json、provenance.json。', '']
    (output/'report.md').write_text('\n'.join(lines), encoding='utf-8')
    with (output/'weights.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['case','horizon','component','mean','absolute_delta_from_one_mean','minimum','maximum','sample_node_count'])
        for case,row in weights.items():
            for h in range(12):
                for k,component in enumerate(COMPONENT_ORDER):
                    writer.writerow([case,h+1,component]+[row[key][h][k] for key in ('mean','absolute_delta_from_one_mean','minimum','maximum')]+[row['count_per_horizon_component']])
    with (output/'horizons.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['case','horizon','mae','rmse','mape'])
        for case,row in result['cases'].items():
            for h in range(12):
                writer.writerow([case,h+1]+[row['metrics']['all_nodes'][key][h] for key in ('per_horizon_mae','per_horizon_rmse','per_horizon_mape')])
    for filename,family in (('regions.csv','metrics'),('strata.csv','strata')):
        with (output/filename).open('w',encoding='utf-8-sig',newline='') as stream:
            writer=csv.writer(stream);writer.writerow(['comparison','group','valid_count','mae_gain_pooled','mae_gain_macro','rmse_gain_pooled','mape_gain_pooled','global_mae_contribution_pooled','gain_positive_pooled','gain_negative_pooled','cancellation_pooled'])
            for name,row in result['comparisons'].items():
                for group,m in row[family].items():
                    writer.writerow([name,group]+[m[key] for key in ('valid_count','mae_gain_pooled','mae_gain_macro','rmse_gain_pooled','mape_gain_pooled','global_mae_contribution_pooled','gain_positive_pooled','gain_negative_pooled','cancellation_pooled')])


def run_probe(models, data, saved, graph, roads, output, baseline, baseline_arrays, protected_paths):
    output=Path(output)
    require(not output.exists(), 'Use a fresh F2 evaluation directory')
    output.mkdir(parents=True)
    before=probe.snapshot_files(protected_paths)
    predictions,replay,mechanisms,preservation,calls,weight_totals={},{},{},{},{},{}
    try:
        require(models.keys()==saved.keys()==set(training.ARMS), 'CONTROL, STATE and EVENT best models are required')
        histories={arm:value['history'] for arm,value in saved.items()}
        require(all(value['completed_epoch']==len(histories[arm]) and len(histories[arm])>0 for arm,value in saved.items()), 'Invalid completed F2 budgets')
        common=min(map(len,histories.values()))
        for epoch in range(common):
            hashes=[history[epoch].get('train_order_sha256') for history in histories.values()]
            require(all(isinstance(v,str) and len(v)==64 for v in hashes) and len(set(hashes))==1, 'F2 common-prefix sample order differs')
        comparable=dict(completed_epochs={a:s['completed_epoch'] for a,s in saved.items()},
            engineering_only=baseline['engineering_only'],validation_rows=baseline['validation_rows'],
            global_updates={a:s['global_updates'] for a,s in saved.items()},best_stage_epochs={a:s['best_epoch'] for a,s in saved.items()},
            common_stage_epoch=common,source_best_epoch=baseline['source_best_epoch'],
            same_completed_training_budget=len({s['completed_epoch'] for s in saved.values()})==1 and len({s['global_updates'] for s in saved.values()})==1,
            whole_model_scope='equal_completed_budget_independent_best_validation' if len({s['completed_epoch'] for s in saved.values()})==1 and len({s['global_updates'] for s in saved.values()})==1 else 'unequal_completed_budget_exploratory',
            source_exact_resume=False,optimizer_and_scheduler_reset_at_stage_start=True,existing_MIN_retrained=False,
            CONTROL_scope='pure_MIN_additional_finetuning_control_actual_budget_recorded',primary_baseline='original_IGSTGNN_not_in_archive')
        target,valid,indices=[np.asarray(baseline_arrays[k]) for k in ('target','valid','sample_indices')]
        require(target.ndim==3 and target.shape[1:]==(12,len(data.station_ids)) and target.dtype.kind=='f', 'Source target axes differ')
        require(len(target)==data.loaders['val_loader'].size==baseline['validation_rows'] and np.array_equal(baseline_arrays['station_ids'],data.station_ids), 'Source station/sample axes differ')
        require(np.array_equal(indices,data.audit['counts']['train_samples']+np.arange(len(target),dtype=np.int64)), 'Source original sample IDs differ')
        require(valid.dtype==np.bool_ and valid.shape==target.shape and np.array_equal(valid,target!=baseline['mask_value']), 'Source original mask differs')
        require(baseline_arrays['prediction'].shape==target.shape and np.isfinite(baseline_arrays['prediction']).all(), 'Source prediction axes differ')
        cached=compute_all_metrics(torch.from_numpy(baseline_arrays['prediction']),torch.from_numpy(target),torch.tensor(baseline['mask_value']))
        require(np.allclose(cached,baseline['actual'],atol=1e-5,rtol=1e-6),'Cached source metrics differ')
        rows=[encode_event(sample,len(data.station_ids)) for sample in data.loaders['val_loader'].original_samples]
        events={key:np.asarray([row[key] for row in rows],dtype=np.float32 if key=='relation' else np.int64 if key in ('type','description') else bool) for key in EVENT_KEYS}
        shuffled,donors,shuffle_audit=early.shuffled_locations(events,indices,20261015)
        regions=probe.regions_for(events,graph,roads)
        model_hashes={arm:tensor_state_hash(model.state_dict()) for arm,model in models.items()}
        def collect(case,row,position,end):
            require(np.array_equal(row['target'],target[position:end]),'Current targets differ from immutable source reference')
            predictions.setdefault(case,np.empty_like(target))[position:end]=row['prediction']
            merge_mechanisms(mechanisms.setdefault(case,{}),row['mechanisms'])
            if row['weights'] is not None:
                f1.update_weights(weight_totals.setdefault(case,{}),row['weights'],row['fusion_means'])
        with ExitStack() as stack:
            for arm,model in models.items():
                require(model_hashes[arm]==tensor_state_hash(saved[arm]['best_model_state']),'Model is not saved selected best')
                stack.enter_context(probe.preserve_model(model,preservation,arm))
            for arm,model in models.items():
                case,position,partitions=arm+'_ON',0,[]
                for batch in data.loaders['val_loader'].get_iterator():
                    calls[case]=calls.get(case,0)+1
                    row=forward_case(model,batch,data.scaler,case)
                    end=position+len(row['prediction'])
                    require(end<=len(target),'Too many replay samples')
                    collect(case,row,position,end);partitions.append(end-position);position=end
                require(position==len(target) and partitions==baseline['batch_partition'],'Original validation batch partition changed')
                actual=compute_all_metrics(torch.from_numpy(predictions[case]),torch.from_numpy(target),torch.tensor(baseline['mask_value']))
                expected=f1.selected_metrics(saved[arm],baseline)
                require(np.allclose(actual,expected,atol=1e-5,rtol=1e-6),arm+' selected best ON replay failed')
                replay[arm]=dict(status='BEST_ON_REPLAY_PASS',best_stage_epoch=saved[arm]['best_epoch'],actual=actual,expected=expected,
                    mask_value=baseline['mask_value'],batch_partition=partitions,atol=1e-5,rtol=1e-6,model_state_sha256=model_hashes[arm])
            # No OFF/content/UNK/shuffle call is permitted before all three full replays.
            position,model=0,models['EVENT']
            for batch in data.loaders['val_loader'].get_iterator():
                end=position+len(batch['x_data'])
                for case in INTERVENTIONS:
                    override={key:value[position:end] for key,value in shuffled.items()} if case=='EVENT_LOCATION_SHUFFLE' else None
                    calls[case]=calls.get(case,0)+1
                    row=forward_case(model,batch,data.scaler,case,override)
                    collect(case,row,position,end)
                    calls['EVENT_ON']+=1
                    restored=forward_case(model,batch,data.scaler,'EVENT_ON')
                    require(np.array_equal(restored['prediction'],predictions['EVENT_ON'][position:end]),'EVENT ON restoration failed')
                position=end
            require(position==len(target),'Incomplete F2 intervention pass')
        require(all(tensor_state_hash(m.state_dict())==model_hashes[a] for a,m in models.items()),'F2 model changed during evaluation')
        predictions['SOURCE_MIN']=baseline_arrays['prediction']
        supported=np.broadcast_to(regions['direct_report_nodes'][:,None,:],target.shape)
        require(np.array_equal(predictions['EVENT_ON'][~supported],predictions['EVENT_REPORT_OFF'][~supported]),'Report content changed unsupported predictions')
        result=summarize(predictions,target,valid,regions,indices,data.station_ids)
        result['training_comparability']=comparable
        for name in ('STATE_vs_CONTROL','EVENT_vs_STATE','EVENT_vs_CONTROL'):
            row=result['comparisons'][name]
            left,right=row['control'].removesuffix('_ON'),row['candidate'].removesuffix('_ON')
            matched=saved[left]['completed_epoch']==saved[right]['completed_epoch'] and saved[left]['global_updates']==saved[right]['global_updates']
            row['same_completed_training_budget']=matched
            if not matched:
                row['scope']='unequal_completed_budget_exploratory_independent_best_not_matched_training_control'
        weights=f1.weight_results(weight_totals)
        region_keys={name:'region_'+str(i) for i,name in enumerate(regions)}
        np.savez_compressed(output/'paired_predictions.npz',**predictions,target=target,valid=valid,sample_indices=indices,
            station_ids=np.asarray(data.station_ids,np.int64),location_donor_station_indices=donors,
            fixed_hard_sample_indices=np.asarray(result['fixed_hard_windows']['sample_indices'],np.int64),
            **{region_keys[name]:mask for name,mask in regions.items()})
        training.legacy.write_json(output/'region_definitions.json',dict(masks=region_keys,
            direct='original_author_relation_support_not_certified_incident_effect',potential='weak_component_envelope_not_certified_physical_propagation'))
        for name,value in (('analysis.json',result),('comparability.json',comparable),('replay.json',replay),
            ('mechanisms.json',dict(cases=mechanisms,component_order=COMPONENT_ORDER,fusion_weights=weights,
                aggregation='sum_except_suffix_max_and_min',weight_scope='all_original_sample_node_positions_not_importance',
                source_weights='unmeasured_cached_predictions_only',CONTROL_weights='native_fixed_sum_no_router',location_shuffle=shuffle_audit,
                semantic_unknown='registered_UNK_tokens_relation_support_preserved'))):
            training.legacy.write_json(output/name,value)
        write_report(output,result,comparable,weights)
        files=probe.require_preserved(before,protected_paths)
        training.legacy.write_json(output/'preservation.json',dict(files=files,models=preservation,source_artifacts_preserved=True))
        calls['SOURCE_MIN']=0
        summary=dict(schema=SCHEMA,status='MIN_COMPONENT_FUSION_EVALUATION_PASS',validation_rows=len(target),
            engineering_only=baseline['engineering_only'],best_stage_epochs=comparable['best_stage_epochs'],source_best_epoch=baseline['source_best_epoch'],
            cases=list(CASES),all_nodes={name:row['metrics']['all_nodes'] for name,row in result['cases'].items()},
            comparisons={name:row['metrics']['all_nodes'] for name,row in result['comparisons'].items()},
            replay_passed=True,on_exactly_restored=True,training_comparability=comparable,source_artifacts_preserved=True,
            new_inference_calls=calls,total_new_inference_calls=sum(calls.values()),optimizer_updates=0,test_evaluated=False,
            initialization_forward_calls=0,primary_baseline='original_IGSTGNN_not_present_in_archive',output_dir=str(output))
        training.legacy.write_json(output/'summary.json',summary)
        return summary
    except BaseException as error:
        training.legacy.write_json(output/'failure.json',dict(status='MIN_COMPONENT_FUSION_EVALUATION_FAILED',error_type=type(error).__name__,
            error=str(error),new_inference_calls=calls,optimizer_updates=0,test_evaluated=False))
        after=probe.snapshot_files(protected_paths)
        training.legacy.write_json(output/'preservation.json',dict(source_artifacts_preserved=before==after,before=before,after=after,
            models=preservation,evaluation_completed=False))
        raise


def parse_args(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    for name,default in (('source-run',training.DEFAULT_SOURCE),('reference-run',training.DEFAULT_REFERENCE),
        ('data-dir',training.paper.default_data()),('graph-dir',training.HERE/'internal_spatial_graph_is11'),('protocol',training.PROTOCOL)):
        parser.add_argument('--'+name,type=Path,default=default)
    parser.add_argument('--seed',type=int)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--threads',type=int)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args(argv)
    for name in ('run_dir','source_run','reference_run','data_dir','graph_dir','protocol'):
        setattr(args,name,getattr(args,name).resolve())
    return args


def main(argv=None):
    args=parse_args(argv)
    identity=training.legacy.read_json(args.run_dir/'identity.json')
    require(identity.get('schema')==training.SCHEMA,'Expected an F2 component-fusion run')
    require(args.check is identity['engineering_check'],'Engineering marker differs')
    args.threads=identity['threads'] if args.threads is None else args.threads
    args.seed=identity['seed'] if args.seed is None else args.seed
    require(args.threads==identity['threads'] and args.seed==identity['seed'],'Replay threads/seed differ from training')
    torch.set_num_threads(args.threads);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=torch.backends.cudnn.allow_tf32=False
    setup=SimpleNamespace(**dict(vars(args),arms=list(identity['arm_modes']),batch_size=identity['protocol']['batch_size'],
        epochs=1,only=None,worker=None,evaluate_after=False))
    logger=logging.getLogger('min_component_fusion_evaluation')
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler(sys.stdout))
    with training.legacy.run_lock(args.run_dir),training.readonly_source_lock(args.source_run):
        # Read preparation contracts directly; never call inference preflight.
        protocol=training.load_protocol(setup)
        hashes=training.paper.verify_release(args.data_dir,protocol)
        data=training.EventPublishedData(args.data_dir,protocol,logger,check=args.check)
        graph,roads,graph_info=training.legacy.load_graph(args.graph_dir,data.station_ids,protocol['graph_sha256'])
        source=training.load_source(setup,hashes)
        initialization=training.legacy.read_json(args.run_dir/'initialization.json')
        current=training.make_identity(setup,protocol,hashes,graph_info,initialization,source)
        current['train_batches']=data.loaders['train_loader'].num_batch
        require(current==identity,'F2 training identity changed; no inference or training was run')
        require(data.loaders['val_loader'].bs==data.loaders['train_loader'].bs==source['identity']['batch_size'],'Replay original batch size changed')
        baseline,baseline_arrays=training.load_source_reference(setup,identity)
        saved,models={},{}
        for arm in training.ARMS:
            require((args.run_dir/arm/'last.pt').is_file(),'All three completed F2 arms are required')
            saved[arm]=torch.load(args.run_dir/arm/'last.pt',map_location='cpu',weights_only=False)
            training.validate_saved(saved[arm],identity,arm)
            require(training.legacy.read_json(args.run_dir/arm/'history.json')==saved[arm]['history'],'History sidecar differs from checkpoint')
            companion=training.legacy.read_json(args.run_dir/arm/'summary.json')
            require(companion.get('arm')==arm and companion.get('mode')==training.ARMS[arm]
                and all(companion.get(k)==saved[arm][k] for k in ('completed_epoch','best_epoch','best_metric')),'Summary sidecar differs from checkpoint')
            models[arm],_=training.build_arm(setup,arm,len(data.station_ids),graph,torch.device(args.device),source)
            models[arm].load_state_dict(saved[arm]['best_model_state'],strict=True);models[arm].eval()
        paths={'run/'+name:args.run_dir/name for name in ('identity.json','initialization.json','source_reference.json','source_reference.npz')}
        paths.update({'source/'+name:args.source_run/name for name in training.SOURCE_FILES})
        for arm in saved:
            paths.update({arm+'/'+name:args.run_dir/arm/name for name in ('last.pt','history.json','summary.json')})
        for key,directory in (('source_sha256',REPO),('package',args.data_dir),('graph_inputs_sha256',args.graph_dir)):
            paths.update({key+'/'+name:directory/name for name in identity[key]})
        paths.update({'evaluation_code/'+name:REPO/name for name in CODE_SOURCES});paths['protocol']=args.protocol
        args.output_dir=args.output_dir.resolve() if args.output_dir else args.run_dir/('component_fusion_evaluation_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6])
        require(args.output_dir!=args.run_dir and args.output_dir not in args.run_dir.parents
            and all(args.output_dir!=path and path not in args.output_dir.parents and args.output_dir not in path.parents for path in (args.source_run,args.reference_run)),
            'Evaluation output must be fresh and outside original source/reference runs')
        require(all(args.output_dir!=args.run_dir/arm and args.run_dir/arm not in args.output_dir.parents for arm in training.ARMS),'Output must not be inside checkpoint directory')
        print('Evaluation directory: '+str(args.output_dir),flush=True)
        result=run_probe(models,data,saved,graph,roads,args.output_dir,baseline,baseline_arrays,paths)
        training.legacy.write_json(args.output_dir/'provenance.json',dict(training_identity=identity,
            evaluation_code_sha256={name:sha256(REPO/name) for name in CODE_SOURCES},source_reference_reused=True,
            no_source_model_forward=True,preflight_rerun=False,existing_MIN_retrained=False,test_payload_deserialized=True,
            no_test_loader_iteration=True,test_evaluated=False))
        print(json.dumps({k:result[k] for k in ('status','validation_rows','best_stage_epochs','optimizer_updates','total_new_inference_calls','output_dir')},indent=2,allow_nan=False),flush=True)
    gc.collect()
    return result


if __name__=='__main__':
    main()
