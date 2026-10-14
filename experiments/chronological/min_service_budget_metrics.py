"""Read-only risk summaries for service-interface adapters on frozen MIN.

Flow bins and OFF error signs are retrospective diagnostics, never model input.
The writer reads committed best archives and checks a common frozen baseline;
it does not load a model, select a new checkpoint, or evaluate the test set.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.chronological.min_frozen_budget_metrics import summarize_pair


SCHEMA = 'min_service_budget_risks_v1'
EXPERIMENT_SCHEMA = 'frozen_MIN_service_limit_pair_v1'
FLOW_EDGES = (0., 5., 20., 50., 100., 200.)
ARMS = ('location', 'state')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def summarize_risks(on, off, target, valid):
    on, off, target, valid = map(np.asarray, (on, off, target, valid))
    require(target.ndim == 3 and target.shape[1] == 12 and target.shape[2] > 0,
            'Expected [sample,12,station] targets')
    require(valid.dtype == np.bool_ and valid.shape == target.shape,
            'Expected shared explicit boolean validity mask')
    for value in (on, off, target):
        require(value.shape == target.shape and value.dtype.kind in 'fiu'
                and np.isfinite(value[valid]).all(), 'Invalid valid prediction or target')
    require((target[valid] >= 0).all(), 'Negative valid target')
    names = ['all_nodes', 'flow_below_100', 'over', 'under', 'exact']
    bins = [(lower, FLOW_EDGES[i+1] if i+1 < len(FLOW_EDGES) else None)
            for i, lower in enumerate(FLOW_EDGES)]
    labels = [f'[{low:g},{high:g})' if high is not None else f'[{low:g},inf)'
              for low, high in bins]
    names += labels
    stats = {name: dict(count=0, percentage_count=0, on_abs=0., off_abs=0.,
        on_sq=0., off_sq=0., on_percentage=0., off_percentage=0.,
        up=0, down=0, unchanged=0, signed_change=0., absolute_change=0.) for name in names}
    for h in range(12):
        mask = valid[:, h]
        y = target[:, h][mask].astype(np.float64)
        a, b = on[:, h][mask].astype(np.float64), off[:, h][mask].astype(np.float64)
        selections = dict(all_nodes=np.ones(y.size, dtype=bool), flow_below_100=y < 100,
                          over=b > y, under=b < y, exact=b == y)
        selections.update({name: (y >= low) & ((y < high) if high is not None else True)
                           for name, (low, high) in zip(labels, bins)})
        for name, selected in selections.items():
            row = stats[name]
            aa, bb, yy = a[selected], b[selected], y[selected]
            ea, eb, change = np.abs(aa-yy), np.abs(bb-yy), aa-bb
            positive = yy > 0
            row['count'] += int(selected.sum())
            row['percentage_count'] += int(positive.sum())
            row['on_abs'] += float(ea.sum())
            row['off_abs'] += float(eb.sum())
            row['on_sq'] += float(np.square(ea).sum())
            row['off_sq'] += float(np.square(eb).sum())
            row['on_percentage'] += float((ea[positive]/yy[positive]).sum())
            row['off_percentage'] += float((eb[positive]/yy[positive]).sum())
            row['up'] += int((change > 0).sum())
            row['down'] += int((change < 0).sum())
            row['unchanged'] += int((change == 0).sum())
            row['signed_change'] += float(change.sum())
            row['absolute_change'] += float(np.abs(change).sum())
    output = {}
    for name, row in stats.items():
        n, p = row['count'], row['percentage_count']
        mean = lambda value: float(value/n) if n else None
        metrics = {}
        for mode in ('on', 'off'):
            metrics[mode] = dict(mae=mean(row[mode+'_abs']),
                rmse=float(np.sqrt(row[mode+'_sq']/n)) if n else None,
                mape=float(row[mode+'_percentage']/p) if p else None)
        output[name] = dict(status='available' if n else 'unavailable', valid_count=n,
            positive_target_count=p, on=metrics['on'], off=metrics['off'],
            gain={key: metrics['off'][key]-metrics['on'][key]
                  if metrics['on'][key] is not None else None for key in ('mae', 'rmse', 'mape')},
            prediction_change=dict(signed_mean_on_minus_off=mean(row['signed_change']),
                absolute_mean=mean(row['absolute_change']), up_fraction=mean(row['up']),
                down_fraction=mean(row['down']), unchanged_fraction=mean(row['unchanged'])))
    n = output['all_nodes']['valid_count']
    require(sum(output[name]['valid_count'] for name in labels) == n
            and sum(output[name]['valid_count'] for name in ('over', 'under', 'exact')) == n,
            'Risk partitions do not cover all valid predictions')
    return dict(schema=SCHEMA, groups=output, flow_edges=list(FLOW_EDGES),
        mape_unit='fraction', gain_sign='OFF_error_minus_ON_error',
        group_scope='retrospective_target_flow_and_frozen_MIN_bias_not_inference_inputs')


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _save(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name+'.partial')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def _number(value, percent=False):
    return '不可用' if value is None else f'{value*(100 if percent else 1):.9f}'


def write_diagnostics(run_dir, arms=ARMS):
    """Check committed sidecars and summarize each arm's already selected best."""
    root = Path(run_dir)
    require(arms and len(set(arms)) == len(arms) and set(arms) <= set(ARMS), 'Unknown or duplicate service arm')
    payload = dict(schema=SCHEMA, status='PARTIAL', test_evaluated=False,
        predictive_gain_claim=False, source='saved_selected_best_validation_archives', runs={}, pending={},
        selected_best_location_vs_state=None,
        interpretation='location_vs_state_is_relation_conditioning_plus_small_architecture_difference; '
                       'both_keep_original_report_support; not_support_location_ablation_or_causal_effect')
    archives, reference = {}, None
    for arm in arms:
        folder = root/arm
        files = [folder/name for name in ('summary.json', 'baseline_predictions.npz', 'best_predictions.npz', 'last.pt')]
        if not all(p.is_file() for p in files):
            continue
        summary = _json(files[0])
        require('best_adapter_epoch' in summary and 'completed_epoch' in summary
                and not summary.get('test_evaluated', False), 'Missing committed service summary')
        require(not summary.get('status', '').endswith('FAILED'), 'Failed arm cannot be ranked')
        import torch
        checkpoint = torch.load(files[3], map_location='cpu', weights_only=False)
        identity = checkpoint.get('identity', {})
        require(identity.get('schema') == EXPERIMENT_SCHEMA and identity.get('arm') == arm
                and identity.get('service_mode') == arm
                and checkpoint.get('schema') == identity.get('checkpoint_schema_reused'),
                'Selected checkpoint belongs to another experiment or arm')
        # A worker can be interrupted after the reusable fit writes its legacy
        # summary but before the service wrapper finalizes the new sidecar.
        if summary.get('schema') != EXPERIMENT_SCHEMA and 'service_mode' not in summary:
            payload['pending'][arm] = 'service_summary_not_finalized_after_fit'
            continue
        require(summary.get('schema') == EXPERIMENT_SCHEMA and summary.get('service_mode') == arm,
                'Summary belongs to another experiment or arm')
        history, epoch = checkpoint['history'], checkpoint['completed_epoch']
        require([row['epoch'] for row in history] == list(range(epoch+1)), 'Invalid committed checkpoint history')
        chosen = min(range(len(history)), key=lambda i: history[i]['validation']['on']['mae_pooled'])
        require(checkpoint['best_epoch'] == chosen
                and checkpoint['best_metric'] == history[chosen]['validation']['on']['mae_pooled'],
                'Best selection certificate disagrees with committed history')
        if (summary['completed_epoch'] != epoch or summary['best_adapter_epoch'] != chosen
                or summary['best_mae'] != checkpoint['best_metric']):
            payload['pending'][arm] = 'selected_sidecars_need_finalization_after_checkpoint_commit'
            continue
        for key, path in (('baseline_predictions_sha256', files[1]), ('best_predictions_sha256', files[2])):
            require(summary.get(key) == hashlib.sha256(path.read_bytes()).hexdigest(),
                    'Selected prediction archive hash differs: '+arm+'/'+key)
        with np.load(files[1], allow_pickle=False) as saved:
            baseline = {key: saved[key].copy() for key in ('prediction', 'target', 'valid', 'sample_indices', 'fixed_hard_indices')}
        with np.load(files[2], allow_pickle=False) as saved:
            archive = {key: saved[key].copy() for key in ('on', 'off', 'target', 'valid', 'sample_indices', 'fixed_hard_indices')}
        for key in ('target', 'valid', 'sample_indices', 'fixed_hard_indices'):
            require(np.array_equal(archive[key], baseline[key]), 'Best/baseline archive differs: '+arm+'/'+key)
        require(np.array_equal(archive['off'], baseline['prediction']), 'Selected OFF differs from source baseline')
        require(archive['sample_indices'].ndim == 1 and len(archive['sample_indices']) == len(archive['on'])
                and np.unique(archive['sample_indices']).size == len(archive['on']), 'Invalid sample IDs')
        if reference is None:
            reference = baseline
        else:
            require(all(np.array_equal(reference[key], baseline[key]) for key in reference),
                    'Service arms do not share identical source predictions, targets, masks and hard windows')
        pair = summarize_pair(archive['on'], archive['off'], archive['target'], archive['valid'], archive['fixed_hard_indices'])
        require(np.isclose(pair['on']['mae_pooled'], summary['best_mae'], rtol=1e-12, atol=1e-12),
                'Selected archive MAE differs from committed best summary')
        payload['runs'][arm] = dict(best_adapter_epoch=summary['best_adapter_epoch'],
            completed_epoch=summary['completed_epoch'], trainable_parameters=summary['trainable_parameters'],
            engineering_only=bool(summary.get('engineering_only', False)), validation_rows=len(archive['on']),
            pair=pair, risks=summarize_risks(archive['on'], archive['off'], archive['target'], archive['valid']))
        archives[arm] = archive
    if len(payload['runs']) == len(arms):
        payload['status'] = 'COMPLETE'
    scopes = {run['engineering_only'] for run in payload['runs'].values()}
    require(len(scopes) <= 1, 'Engineering and formal validation arms cannot be mixed')
    payload['engineering_only'] = next(iter(scopes)) if scopes else None
    if set(archives) == set(ARMS):
        loc, state = payload['runs']['location'], payload['runs']['state']
        if loc['completed_epoch'] == state['completed_epoch']:
            archive = archives['location']
            payload['selected_best_location_vs_state'] = dict(
                completed_training_budget=loc['completed_epoch'],
                location_best_epoch=loc['best_adapter_epoch'], state_best_epoch=state['best_adapter_epoch'],
                scope='separately_selected_best_at_equal_completed_budget_exploratory_validation',
                pair=summarize_pair(archive['on'], archives['state']['on'], archive['target'], archive['valid'], archive['fixed_hard_indices']))
        else:
            payload['comparison_deferred'] = 'unequal_completed_training_budgets'
    _save(root/'risk_diagnostics.json', payload)
    scope_line = ('范围：工程检查数据，仅验证实现和续跑，不解释预测收益。' if payload['engineering_only'] else
                  '范围：作者发布验证集上的条件离线探索。' if payload['engineering_only'] is False else
                  '范围：尚无可汇总的完整预测制品。')
    lines = ['# 冻结 MIN：事故服务预算接口对照', '', scope_line, '',
        'location 读取事故关系细节；state 仅读交通状态。两组共享原事故支持范围、冻结 MIN 和原验证批次。', '',
        '交换提议只使用基线预算；新增预算只进入最终交换限制。预算无认证交通单位。', '',
        '以下是各组已完成预算内选出的最佳验证检查点，包括第 0 轮。正收益为 MIN 误差减候选误差；MAPE 为百分数。', '',
        '| 组 | 验证窗口 | 完成轮次 | 最佳轮次 | 训练参数 | MAE | RMSE | MAPE % | MAE收益 | 预测上升占比 % |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for arm, run in payload['runs'].items():
        row = run['risks']['groups']['all_nodes']
        lines.append('| '+arm+' | '+str(run['validation_rows'])+' | '+str(run['completed_epoch'])+' | '+str(run['best_adapter_epoch'])+' | '+str(run['trainable_parameters'])+' | '+
            ' | '.join((_number(row['on']['mae']), _number(row['on']['rmse']), _number(row['on']['mape'], True),
                        _number(row['gain']['mae']), _number(row['prediction_change']['up_fraction'], True)))+' |')
    lines += ['', '## 低流量与原 MIN 高估/低估', '',
        '分组由真实目标和固定 MIN 预测在事后定义，不能作为实际预测门控。空组记为不可用。', '',
        '| 组 | 区间/偏差 | 有效点数 | MIN MAE | 候选 MAE | MAE收益 | MAPE收益（百分点） |',
        '|---|---|---:|---:|---:|---:|---:|']
    for arm, run in payload['runs'].items():
        for name, row in run['risks']['groups'].items():
            if name == 'all_nodes':
                continue
            lines.append('| '+arm+' | '+name+' | '+str(row['valid_count'])+' | '+
                ' | '.join((_number(row['off']['mae']), _number(row['on']['mae']), _number(row['gain']['mae']),
                            _number(row['gain']['mape'], True)))+' |')
    comparison = payload['selected_best_location_vs_state']
    lines += ['', '## 关系细节增量与解释', '']
    if comparison is not None:
        lines.append('相同已完成训练预算下，location 最佳验证 MAE 相对 state 最佳验证 MAE 的收益：'+
                     _number(comparison['pair']['mae_gain_pooled'])+'。')
    else:
        lines.append('两组尚未同时完成相同预算，暂不比较各自最佳检查点的排名。')
    lines += ['', 'location 与 state 的比较仍包含小幅有效参数量/隐藏宽度差异，不能单独解释为纯事故内容效应。', '',
        'state 仍通过原支持掩码使用粗粒度事故位置，不是无事故信息模型。正确/置乱关系对照仍可在选定模型后检验细节依赖。', '',
        '固定困难窗口结果、逐预测步误差、修正方向和空组状态见 risk_diagnostics.json；两组训练曲线的共同轮次比较见 comparison.json。', '',
        '仅条件离线验证集探索；没有重新推理、重新选择最佳轮次或评价测试集。窗口可能重叠，不视为独立事故重复。', '']
    (root/'report.md').write_text('\n'.join(lines), encoding='utf-8')
    return payload
