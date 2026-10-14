"""Read-only validation reports for the single output-domain R2 decoder.

Source MIN, same-decoder condition OFF and independently trained state/location
are three different references. None is silently substituted for another.
"""
from pathlib import Path

import numpy as np
import torch

from experiments.chronological.min_response_metrics import (
    _hash, _number, _read, _write, require, summarize_response_pair,
)


SCHEMA = 'MIN_single_stock_flow_joint_pair_r2'
STAGE = 'single_stock_flow_joint'
ARMS = ('state', 'location')
KEYS = ('target', 'valid', 'sample_indices', 'fixed_hard_indices', 'history_low_mask')


def write_diagnostics(run_dir, arms=ARMS):
    """Validate committed artifacts, then report without loading any model."""
    root = Path(run_dir)
    require(arms and len(set(arms)) == len(arms) and set(arms) <= set(ARMS), 'Unknown/duplicate R2 arms')
    payload = dict(schema=SCHEMA, training_stage=STAGE, status='PARTIAL', runs={}, pending={},
        source_MIN_training_updates=0, source='committed_best_validation_archives',
        predictive_gain_claim=False, test_evaluated=False,
        interpretation='new_decoder_plus_joint_optimization_vs_immutable_MIN_not_pure_incident_gain',
        selected_best_location_vs_state=None)
    reference, source_hash, archives, histories = None, None, {}, {}
    for arm in arms:
        folder = root/arm
        paths = {name: folder/name for name in
                 ('summary.json', 'last.pt', 'baseline_predictions.npz', 'best_predictions.npz')}
        if not all(path.is_file() for path in paths.values()):
            payload['pending'][arm] = 'committed_artifacts_not_yet_complete'
            continue
        summary = _read(paths['summary.json'])
        saved = torch.load(paths['last.pt'], map_location='cpu', weights_only=False)
        identity = saved.get('identity', {})
        require(saved.get('schema') == identity.get('schema') == summary.get('schema') == SCHEMA
                and identity.get('training_stage') == summary.get('training_stage') == STAGE
                and identity.get('arm') == summary.get('arm') == arm, 'R2 artifact stage/arm mismatch')
        require(identity.get('source_MIN_training_updates') == summary.get('source_MIN_training_updates') == 0
                and not summary.get('test_evaluated', False), 'Source MIN updates or test use violate R2 protocol')
        require(not summary.get('status', '').endswith('FAILED'), 'Failed R2 arm cannot be ranked')
        epoch, history = saved['completed_epoch'], saved['history']
        require(type(epoch) is int and epoch >= 0 and len(history) == epoch+1
                and [r['epoch'] for r in history] == list(range(epoch+1)), 'Invalid R2 committed history')
        chosen = min(range(len(history)), key=lambda e: history[e]['validation']['on']['mae_pooled'])
        require(saved['best_epoch'] == chosen
                and saved['best_metric'] == history[chosen]['validation']['on']['mae_pooled'],
                'R2 best selection differs from committed history')
        if (summary['completed_epoch'] != epoch or summary['best_epoch'] != chosen
                or summary['best_mae'] != saved['best_metric']):
            payload['pending'][arm] = 'summary_needs_finalization_after_checkpoint_commit'
            continue
        require(saved['source_state_sha256'] == identity['source_state_sha256'] == summary['source_state_sha256']
                and summary.get('source_reference_preserved') is True, 'Source MIN certificate changed')
        require(saved['baseline_sha256'] == _hash(paths['baseline_predictions.npz']), 'R2 source reference archive changed')
        for field, name in (('baseline_predictions_sha256', 'baseline_predictions.npz'),
                            ('best_predictions_sha256', 'best_predictions.npz')):
            require(summary.get(field) == _hash(paths[name]), 'R2 prediction archive changed: '+arm+'/'+name)
        with np.load(paths['baseline_predictions.npz'], allow_pickle=False) as f:
            baseline = {k: f[k].copy() for k in ('prediction',)+KEYS}
        with np.load(paths['best_predictions.npz'], allow_pickle=False) as f:
            archive = {k: f[k].copy() for k in ('on', 'current_off', 'original_off', 'off')+KEYS}
        require(all(np.array_equal(archive[k], baseline[k]) for k in KEYS), 'R2 sample/target/mask axes changed')
        require(np.array_equal(archive['original_off'], baseline['prediction'])
                and np.array_equal(archive['off'], baseline['prediction']), 'Original MIN reference was replaced')
        require(archive['on'].shape == archive['current_off'].shape == baseline['prediction'].shape,
                'Candidate decoder prediction axes differ')
        indices = archive['sample_indices']
        require(indices.ndim == 1 and indices.dtype.kind in 'iu' and len(indices) == len(archive['on'])
                and np.unique(indices).size == len(indices), 'Invalid released sample indices')
        if reference is None:
            reference, source_hash = baseline, identity['source_state_sha256']
        else:
            require(source_hash == identity['source_state_sha256']
                    and all(np.array_equal(reference[k], baseline[k]) for k in reference),
                    'R2 arms have different original MIN/data/masks')
        def pair(other):
            return summarize_response_pair(archive['on'], other, archive['target'], archive['valid'],
                                           archive['fixed_hard_indices'], archive['history_low_mask'])
        original, current = pair(archive['original_off']), pair(archive['current_off'])
        original['reference_scope'] = 'immutable_existing_MIN_not_new_budget_matched_MIN_training'
        current['reference_scope'] = 'same_stock_flow_decoder_condition_disabled_checkpoint_sensitivity'
        require(np.isclose(original['on']['mae_pooled'], summary['best_mae'], rtol=1e-12, atol=1e-12),
                'R2 selected prediction MAE differs from committed best')
        expected = history[chosen]['validation']
        require(original['paired'] == expected['paired']
                and current['paired'] == expected['current_off']['paired'], 'R2 selected paired gains changed')
        payload['runs'][arm] = dict(completed_epoch=epoch, best_epoch=chosen,
            trainable_parameters=summary['trainable_parameters'], validation_rows=len(indices),
            engineering_only=bool(summary['engineering_only']), original_reference_pair=original,
            current_off_pair=current, exact_source_prediction_initialization=False,
            current_OFF_exactly_matches_original=bool(np.array_equal(archive['current_off'], archive['original_off'])),
            mechanisms=expected.get('mechanisms', {}))
        archives[arm], histories[arm] = archive, history
    scopes = {r['engineering_only'] for r in payload['runs'].values()}
    require(len(scopes) <= 1, 'Engineering and formal R2 arms cannot be combined')
    payload['engineering_only'] = next(iter(scopes)) if scopes else None
    if len(payload['runs']) == len(arms):
        payload['status'] = 'COMPLETE'
    if set(archives) == set(ARMS):
        a, b = payload['runs']['location'], payload['runs']['state']
        if a['completed_epoch'] == b['completed_epoch']:
            for e in range(1, a['completed_epoch']+1):
                require(histories['location'][e]['training']['train_order_sha256']
                        == histories['state'][e]['training']['train_order_sha256'], 'R2 paired training order differs')
            loc = archives['location']
            comparison = summarize_response_pair(loc['on'], archives['state']['on'], loc['target'], loc['valid'],
                loc['fixed_hard_indices'], loc['history_low_mask'])
            payload['selected_best_location_vs_state'] = dict(completed_training_budget=a['completed_epoch'],
                location_best_epoch=a['best_epoch'], state_best_epoch=b['best_epoch'], pair=comparison,
                scope='separately_selected_best_at_equal_extra_budget_exploratory_validation_same_M_support')
        else:
            payload['comparison_deferred'] = 'unequal_completed_extra_training_budgets'
    _write(root/'risk_diagnostics.json', payload)
    _write_report(root, payload)
    return payload


def _write_report(root, payload):
    scope = ('工程检查，不评价预测收益。' if payload['engineering_only'] else
             '作者发布验证集上的条件离线探索。' if payload['engineering_only'] is False else '尚无完整制品。')
    lines = ['# MIN-R2：单一开放存量—流量解码器', '', '范围：'+scope, '',
        '仅联合训练两个新候选；已有 MIN 只读，不新增 MIN-only 训练。',
        '迁移兼容编码权重后删除旧潜物理路径；初始化不再等于原 MIN。',
        '最终预测由求解器流出量直接产生，没有物理层后的自由预测残差。', '',
        '原 MIN 是固定旧参考；current OFF 仍运行新解码器，只关闭条件服务调整。',
        'ON 对原 MIN 包含结构替换及额外训练；ON/OFF 仅为同检查点条件路径敏感性。',
        'state/location 都保留支持 M，location 额外读取 D 数值，不是完整事故有无对照。', '',
        '| 候选 | 完成轮次 | 最佳轮次 | 参数 | ON MAE | 原 MIN MAE | 条件 OFF MAE | 对原 MIN 收益 | 条件路径收益 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for arm, row in payload['runs'].items():
        old, current = row['original_reference_pair'], row['current_off_pair']
        lines.append('| '+' | '.join((arm, str(row['completed_epoch']), str(row['best_epoch']),
            str(row['trainable_parameters']), _number(old['on']['mae_pooled']), _number(old['off']['mae_pooled']),
            _number(current['off']['mae_pooled']), _number(old['paired']['mae_gain_pooled']),
            _number(current['paired']['mae_gain_pooled'])))+' |')
    comparison = payload['selected_best_location_vs_state']
    if comparison:
        lines += ['', '同新增预算、分别选模的 location 对 state MAE 收益：'+
                  _number(comparison['pair']['paired']['mae_gain_pooled'])+'。正值表示 location 更好。']
    for arm, row in payload['runs'].items():
        pair = row['original_reference_pair']
        lines += ['', '## '+arm+'：误差分布与风险', '',
            '| 分组 | 点数 | ON MAE | 原 MIN MAE | MAE收益 | MAPE收益（百分点） |',
            '|---|---:|---:|---:|---:|---:|']
        for name, group in pair['risks']['groups'].items():
            lines.append('| '+' | '.join((name, str(group['valid_count']), _number(group['on']['mae']),
                _number(group['off']['mae']), _number(group['gain']['mae']), _number(group['gain']['mape'], True)))+' |')
        lines += ['', 'ON pooled RMSE：'+_number(pair['on']['rmse_pooled'])+'；MAPE：'+
            _number(pair['on']['mape_pooled'], True)+'%。',
            '源 MIN 固定困难窗口 MAE 收益：'+_number(pair['fixed_hard_windows'].get('mae_gain'))+'。', '',
            '| 目标槽 | ON MAE | 原 MIN MAE | MAE收益 |', '|---|---:|---:|---:|']
        for slot, (on, off, gain) in enumerate(zip(pair['on']['per_horizon_mae'], pair['off']['per_horizon_mae'],
                pair['paired']['per_horizon_mae_gain']), 1):
            lines.append('| '+' | '.join((str(slot), _number(on), _number(off), _number(gain)))+' |')
    lines += ['', '## 物理解释与下一步判据', '',
        '核对 mechanisms 中节点与开放边界账本误差、服务/发送/接收约束违规、外部未进入需求积压及服务主导比例。',
        '非主导区容量扰动无响应是正确现象；仅改变参数并不代表最终流量有效改变。',
        '图是候选连接，当前联合物理认证为0/496。存量、储存上限、固定分流和站点流出观测映射是显式模型假设。',
        '这里证明的是模型内部约束；没有认证真实道路容量、排队长度、拥堵回溢或事故因果。',
        '真实流量桶和高估/低估使用真值事后划分，不能进入推理输入。验证集已反复用于研发，未评价测试集。',
        '数值：risk_diagnostics.json；各组 best_predictions.npz、history.json；初始化与图证据见运行目录审计。', '']
    path = root/'report.md'
    temporary = path.with_name(path.name+'.partial')
    temporary.write_text('\n'.join(lines), encoding='utf-8')
    temporary.replace(path)
