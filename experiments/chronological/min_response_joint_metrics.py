"""Reports for two jointly adapted candidates reusing an existing MIN.

Original MIN is immutable; current OFF is an independently updated backbone.
Neither comparison substitutes for an independently budget-matched MIN-only
experiment, which is deliberately not run again. Location/state is the matched
candidate comparison. Reports never evaluate the test set or invoke a model.
"""
from pathlib import Path

import numpy as np
import torch

from experiments.chronological.min_response_metrics import (
    _hash, _number, _read, _write, require, summarize_response_pair,
)


SCHEMA = 'MIN_conditional_response_joint_pair_v1'
STAGE = 'joint_finetune_stage2'
ARMS = ('state', 'location')
KEYS = ('target', 'valid', 'sample_indices', 'fixed_hard_indices', 'history_low_mask')


def write_diagnostics(run_dir, arms=ARMS):
    """Validate committed checkpoint, selection and archive identities first."""
    root = Path(run_dir)
    require(arms and len(set(arms)) == len(arms) and set(arms) <= set(ARMS),
            'Only state/location candidates are permitted; existing MIN is reused')
    payload = dict(schema=SCHEMA, training_stage=STAGE, status='PARTIAL', runs={}, pending={},
        source_MIN_training_updates=0, source='committed_best_validation_archives',
        predictive_gain_claim=False, test_evaluated=False,
        interpretation='candidate_plus_extra_finetuning_vs_existing_MIN_not_pure_module_gain',
        selected_best_location_vs_state=None)
    reference, source_hash, archives, histories = None, None, {}, {}
    for arm in arms:
        folder = root/arm
        names = ('summary.json', 'last.pt', 'baseline_predictions.npz', 'best_predictions.npz')
        paths = {name: folder/name for name in names}
        if not all(path.is_file() for path in paths.values()):
            payload['pending'][arm] = 'committed_artifacts_not_yet_complete'
            continue
        summary = _read(paths['summary.json'])
        checkpoint = torch.load(paths['last.pt'], map_location='cpu', weights_only=False)
        identity = checkpoint.get('identity', {})
        require(checkpoint.get('schema') == SCHEMA and identity.get('schema') == SCHEMA
                and identity.get('training_stage') == STAGE and identity.get('arm') == arm,
                'Checkpoint belongs to a different joint stage or arm')
        require(summary.get('schema') == SCHEMA and summary.get('training_stage') == STAGE
                and summary.get('arm') == arm and not summary.get('test_evaluated', False),
                'Invalid joint response summary identity')
        require(summary.get('source_MIN_training_updates') == 0
                and identity.get('source_MIN_training_updates') == 0,
                'Joint candidate run must reuse existing MIN without a source update')
        require(not summary.get('status', '').endswith('FAILED'), 'Failed joint arm cannot be ranked')
        epoch, history = checkpoint['completed_epoch'], checkpoint['history']
        require(type(epoch) is int and epoch >= 0 and len(history) == epoch+1
                and [row['epoch'] for row in history] == list(range(epoch+1)),
                'Invalid committed joint response history')
        chosen = min(range(len(history)), key=lambda e: history[e]['validation']['on']['mae_pooled'])
        require(checkpoint['best_epoch'] == chosen
                and checkpoint['best_metric'] == history[chosen]['validation']['on']['mae_pooled'],
                'Joint best selection certificate disagrees with history')
        if (summary['completed_epoch'] != epoch or summary['best_epoch'] != chosen
                or summary['best_mae'] != checkpoint['best_metric']):
            payload['pending'][arm] = 'summary_needs_finalization_after_checkpoint_commit'
            continue
        require(checkpoint['source_state_sha256'] == identity['source_state_sha256']
                == summary['source_state_sha256'], 'Joint source model certificate changed')
        require(bool(summary.get('source_reference_preserved')),
                'The original MIN reference must be preserved')
        require(checkpoint['baseline_sha256'] == _hash(paths['baseline_predictions.npz']),
                'Joint checkpoint baseline archive hash changed')
        for field, name in (('baseline_predictions_sha256', 'baseline_predictions.npz'),
                            ('best_predictions_sha256', 'best_predictions.npz')):
            require(summary.get(field) == _hash(paths[name]), 'Joint prediction archive hash changed: '+arm+'/'+name)
        with np.load(paths['baseline_predictions.npz'], allow_pickle=False) as saved:
            baseline = {key: saved[key].copy() for key in ('prediction',)+KEYS}
        with np.load(paths['best_predictions.npz'], allow_pickle=False) as saved:
            archive = {key: saved[key].copy() for key in ('on', 'current_off', 'original_off', 'off')+KEYS}
        require(all(np.array_equal(archive[key], baseline[key]) for key in KEYS),
                'Joint prediction targets, masks or sample axes changed')
        require(np.array_equal(archive['original_off'], baseline['prediction'])
                and np.array_equal(archive['off'], baseline['prediction']),
                'Original MIN predictions were replaced by an updated OFF backbone')
        require(archive['current_off'].shape == baseline['prediction'].shape,
                'Joint current OFF axes changed')
        indices = archive['sample_indices']
        require(indices.ndim == 1 and indices.dtype.kind in 'iu' and len(indices) == len(archive['on'])
                and np.unique(indices).size == len(indices), 'Invalid global joint sample indices')
        if reference is None:
            reference, source_hash = baseline, identity['source_state_sha256']
        else:
            require(source_hash == identity['source_state_sha256']
                    and all(np.array_equal(reference[key], baseline[key]) for key in reference),
                    'Joint candidates do not share the same original MIN, data and masks')
        def pair(other):
            return summarize_response_pair(archive['on'], other, archive['target'], archive['valid'],
                                           archive['fixed_hard_indices'], archive['history_low_mask'])
        original, current = pair(archive['original_off']), pair(archive['current_off'])
        original['reference_scope'] = 'immutable_original_MIN_not_budget_matched_new_training'
        current['reference_scope'] = 'current_joint_checkpoint_OFF_path_sensitivity_only'
        require(np.isclose(original['on']['mae_pooled'], summary['best_mae'], rtol=1e-12, atol=1e-12),
                'Joint selected prediction archive MAE differs from committed selection')
        expected = history[chosen]['validation']
        require(original['paired'] == expected['paired']
                and current['paired'] == expected['current_off']['paired'],
                'Joint archived gains differ from selected validation record')
        current_equal = bool(np.array_equal(archive['current_off'], archive['original_off']))
        require(current_equal == summary['current_OFF_exactly_matches_original'],
                'Joint OFF reference identity flag disagrees with archives')
        payload['runs'][arm] = dict(completed_epoch=epoch, best_epoch=chosen,
            trainable_parameters=summary['trainable_parameters'], validation_rows=len(indices),
            engineering_only=bool(summary['engineering_only']),
            original_reference_pair=original, current_off_pair=current,
            current_OFF_exactly_matches_original=current_equal)
        archives[arm], histories[arm] = archive, history
    scopes = {row['engineering_only'] for row in payload['runs'].values()}
    require(len(scopes) <= 1, 'Cannot mix engineering and formal joint response runs')
    payload['engineering_only'] = next(iter(scopes)) if scopes else None
    if len(payload['runs']) == len(arms):
        payload['status'] = 'COMPLETE'
    if set(archives) == set(ARMS):
        a, b = payload['runs']['location'], payload['runs']['state']
        if a['completed_epoch'] == b['completed_epoch']:
            for epoch in range(1, a['completed_epoch']+1):
                require(histories['location'][epoch]['training']['train_order_sha256']
                        == histories['state'][epoch]['training']['train_order_sha256'],
                        'Joint candidate training sample order changed')
            loc = archives['location']
            comparison = summarize_response_pair(loc['on'], archives['state']['on'], loc['target'], loc['valid'],
                                                 loc['fixed_hard_indices'], loc['history_low_mask'])
            comparison['reference_scope'] = 'state_candidate_at_equal_extra_training_budget'
            payload['selected_best_location_vs_state'] = dict(completed_training_budget=a['completed_epoch'],
                location_best_epoch=a['best_epoch'], state_best_epoch=b['best_epoch'],
                scope='separately_selected_best_at_equal_extra_budget_exploratory_validation', pair=comparison)
        else:
            payload['comparison_deferred'] = 'unequal_extra_training_budgets'
    _write(root/'risk_diagnostics.json', payload)
    _write_report(root, payload)
    return payload


def _write_report(root, payload):
    scope = ('工程检查，不用于评价预测收益。' if payload['engineering_only'] else
             '作者发布验证集上的条件离线探索。' if payload['engineering_only'] is False else '制品尚未完整。')
    lines = ['# MIN-R1：联合适应与已有 MIN 复用', '', '范围：'+scope, '',
        '只训练 state/location 两个新候选；已有源 MIN 不更新、不新增 MIN-only 训练。', '',
        '原 MIN 是固定旧参考；current OFF 是该联合检查点的主干，可能已改变。',
        'ON 相对原 MIN 的收益包含新结构与额外微调；ON/OFF 是路径敏感性。',
        '同预算 location/state 比较关系细节增量，两组都保留原支持 M，不是有事故/无事故对照。', '',
        '| 组 | 完成轮次 | 最佳轮次 | 参数 | ON MAE | 原 MIN MAE | current OFF MAE | 相对原 MIN 收益 | 路径收益 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for arm, row in payload['runs'].items():
        original, current = row['original_reference_pair'], row['current_off_pair']
        lines.append('| '+' | '.join((arm, str(row['completed_epoch']), str(row['best_epoch']),
            str(row['trainable_parameters']), _number(original['on']['mae_pooled']),
            _number(original['off']['mae_pooled']), _number(current['off']['mae_pooled']),
            _number(original['paired']['mae_gain_pooled']), _number(current['paired']['mae_gain_pooled'])))+' |')
    comparison = payload['selected_best_location_vs_state']
    if comparison:
        pair = comparison['pair']
        lines += ['', '相同新增训练预算下，location 相对 state 的验证 MAE 收益：'+
            _number(pair['paired']['mae_gain_pooled'])+'；正值表示 location 更好。',
            '两组分别按验证 MAE 选择最佳轮次，零轮也参与选模。']
    for arm, row in payload['runs'].items():
        pair = row['original_reference_pair']
        lines += ['', '## '+arm+'：条件风险（相对原 MIN）', '',
            '| 分组 | 点数 | ON MAE | 原 MIN MAE | MAE收益 | MAPE收益（百分点） |',
            '|---|---:|---:|---:|---:|---:|']
        for name, group in pair['risks']['groups'].items():
            lines.append('| '+' | '.join((name, str(group['valid_count']), _number(group['on']['mae']),
                _number(group['off']['mae']), _number(group['gain']['mae']), _number(group['gain']['mape'], True)))+' |')
        low = pair['history_low_group']
        all_nodes = pair['risks']['groups']['all_nodes']
        lines += ['', 'ON pooled RMSE：'+_number(pair['on']['rmse_pooled'])+'；MAPE：'+
            _number(pair['on']['mape_pooled'], True)+'%。',
            '上调点占比：'+_number(all_nodes['prediction_change']['up_fraction'], True)+'%；预测变化仍须结合误差判断。',
            '输入历史低流量组平均 MAE 收益：'+_number(low.get('gain', {}).get('mae_gain_pooled'))+'。',
            '源 MIN 固定困难窗口 MAE 收益：'+_number(pair['fixed_hard_windows'].get('mae_gain'))+'。', '',
            '| 目标槽 | ON MAE | 原 MIN MAE | MAE收益 |', '|---|---:|---:|---:|']
        for slot, (on, off, gain) in enumerate(zip(pair['on']['per_horizon_mae'], pair['off']['per_horizon_mae'],
                pair['paired']['per_horizon_mae_gain']), 1):
            lines.append('| '+' | '.join((str(slot), _number(on), _number(off), _number(gain)))+' |')
    lines += ['', '历史低流量组由输入 X 的十二槽均值定义。真实目标流量桶与原 MIN 高估/低估是事后分组，不能进入推理门控。',
        '风险惩罚只约束输入历史组的平均超额 MAE，不保证每点、每路或 MAPE 不退化。',
        '潜状态和预算未认证为密度/容量，局部平衡不等于最终流量预测守恒。目标槽不自动转换为已认证的未来分钟数。',
        '验证集已经反复用于研发；本结果不构成事故因果、独立泛化或超过原论文的证明。没有评价测试集。',
        '数值明细：risk_diagnostics.json；各组 history.json、summary.json、baseline_predictions.npz 和 best_predictions.npz。', '']
    path = root/'report.md'
    temporary = path.with_name(path.name+'.partial')
    temporary.write_text('\n'.join(lines), encoding='utf-8')
    temporary.replace(path)
