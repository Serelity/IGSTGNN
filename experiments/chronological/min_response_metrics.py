"""Paired training risk and saved validation reports for the MIN response pilot.

The protected training group uses *observed history* flow, never future truth or
future error signs. Target-flow and MIN-bias groups remain retrospective reports.
All predictions passed to archive metrics use the author's original flow scale;
that scale is not a newly certified physical unit.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from experiments.chronological.min_frozen_budget_metrics import summarize_pair
from experiments.chronological.min_service_budget_metrics import summarize_risks


SCHEMA = 'min_response_risks_v1'
EXPERIMENT_SCHEMA = 'frozen_MIN_conditional_response_pair_v1'
ARMS = ('location', 'state')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def history_low_mask(history_normalized, scaler_mean, scaler_std, threshold=100.):
    """Return Bool[B,N] from the released twelve-slot observed flow mean.

    Released values and preprocessing are retained; this does not invent a new
    observation-validity channel or infer incident-free conditions.
    """
    require(isinstance(history_normalized, torch.Tensor)
            and history_normalized.ndim == 4 and history_normalized.shape[1] == 12
            and history_normalized.shape[-1] == 3
            and history_normalized.is_floating_point()
            and bool(torch.isfinite(history_normalized).all()),
            'Expected finite Float[B,12,N,3] observed history')
    mean = torch.as_tensor(scaler_mean, dtype=history_normalized.dtype, device=history_normalized.device)
    std = torch.as_tensor(scaler_std, dtype=history_normalized.dtype, device=history_normalized.device)
    require(mean.numel() == std.numel() == 1 and bool(torch.isfinite(mean).all())
            and bool(torch.isfinite(std).all()) and bool((std > 0).all()), 'Invalid flow scaler')
    require(np.isfinite(threshold) and threshold > 0., 'Invalid low-history threshold')
    flow = history_normalized[..., 0]*std+mean
    average = flow.mean(dim=1)
    return (average >= 0.) & (average < threshold)


def paired_training_loss(on_normalized, off_normalized, target_normalized, history_normalized,
                         scaler_mean, scaler_std, *, low_history_threshold=100., risk_weight=.25):
    """Full author-scale MAE plus paired low-observed-flow excess-risk penalty.

    The main mask follows the published training convention: mask the finite
    target minimum when it is below one, otherwise mask zero. This differs from
    future-target binning, which is reserved for evaluation. OFF is detached.
    """
    require(np.isfinite(risk_weight) and risk_weight >= 0., 'Risk weight must be finite and nonnegative')
    require(isinstance(on_normalized, torch.Tensor) and on_normalized.ndim == 4
            and on_normalized.shape[1] == 12 and on_normalized.shape[-1] == 1
            and on_normalized.is_floating_point(), 'Expected Float[B,12,N,1] predictions')
    for value in (off_normalized, target_normalized):
        require(isinstance(value, torch.Tensor) and value.shape == on_normalized.shape
                and value.dtype == on_normalized.dtype and value.device == on_normalized.device,
                'Paired training shape, dtype or device mismatch')
    require(isinstance(history_normalized, torch.Tensor) and history_normalized.ndim == 4,
            'Expected Float[B,12,N,3] observed history')
    require(history_normalized.device == on_normalized.device
            and history_normalized.dtype == on_normalized.dtype
            and history_normalized.shape[0] == on_normalized.shape[0]
            and history_normalized.shape[2] == on_normalized.shape[2], 'History axes do not match targets')
    low = history_low_mask(history_normalized, scaler_mean, scaler_std, low_history_threshold)
    mean = torch.as_tensor(scaler_mean, dtype=on_normalized.dtype, device=on_normalized.device)
    std = torch.as_tensor(scaler_std, dtype=on_normalized.dtype, device=on_normalized.device)
    on = on_normalized*std+mean
    off = off_normalized.detach()*std+mean
    target = target_normalized.detach()*std+mean
    finite = torch.isfinite(target)
    null = target[finite].min() if bool(finite.any()) else target.new_zeros(())
    null = null if bool(null < 1.) else target.new_zeros(())
    valid = finite & (target != null)
    require(bool(torch.isfinite(on[valid]).all()) and bool(torch.isfinite(off[valid]).all()),
            'Non-finite valid paired prediction')
    clean_target = torch.where(valid, target, torch.zeros_like(target))
    on_error = (torch.where(valid, on, torch.zeros_like(on))-clean_target).abs()
    off_error = (torch.where(valid, off, torch.zeros_like(off))-clean_target).abs()
    count = valid.sum()
    mae_on, mae_off = on_error.sum()/count.clamp_min(1), off_error.sum()/count.clamp_min(1)
    group = valid & low[:, None, :, None]
    group_count = group.sum()
    low_on = torch.where(group, on_error, torch.zeros_like(on_error)).sum()/group_count.clamp_min(1)
    low_off = torch.where(group, off_error, torch.zeros_like(off_error)).sum()/group_count.clamp_min(1)
    excess = low_on-low_off
    penalty = torch.relu(excess)*risk_weight
    loss = mae_on+penalty
    scalar = lambda value: float(value.detach().cpu())
    available = bool(group_count > 0)
    stats = dict(loss=scalar(loss), mae_on=scalar(mae_on), mae_off=scalar(mae_off),
        low_history_mae_on=scalar(low_on) if available else None,
        low_history_mae_off=scalar(low_off) if available else None,
        low_history_excess_mae=scalar(excess) if available else None,
        risk_penalty=scalar(penalty), low_history_valid_count=int(group_count), valid_count=int(count),
        history_low_threshold=float(low_history_threshold), risk_weight=float(risk_weight),
        group_scope='observed_twelve_slot_flow_mean_not_future_target',
        target_mask_scope='published_finite_target_min_below_one_otherwise_zero')
    return loss, stats


def summarize_response_pair(on, off, target, valid, fixed_hard_indices, history_low_mask=None):
    """Extend existing paired metrics without changing their selection/masks."""
    result = summarize_pair(on, off, target, valid, fixed_hard_indices)
    result['scope'] = 'saved_validation_paired_predictions_with_explicit_reference'
    result['response_schema'] = SCHEMA
    result['risks'] = summarize_risks(on, off, target, valid)
    if history_low_mask is None:
        result['history_low_group'] = dict(status='unavailable', reason='observed_history_mask_not_supplied')
    else:
        low = np.asarray(history_low_mask)
        require(low.dtype == np.bool_ and low.shape == (target.shape[0], target.shape[2]),
                'Expected Bool[sample,station] observed-history group mask')
        mask = valid & low[:, None, :]
        subset = summarize_pair(on, off, target, mask, np.empty(0, dtype=np.int64))
        result['history_low_group'] = dict(status=subset['on']['status'],
            valid_count=subset['on']['valid_count'], on=subset['on'], off=subset['off'],
            gain=subset['paired'], group_scope='observed_twelve_slot_flow_mean_not_future_target')
    return result


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name+'.partial')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def _number(value, percent=False):
    return '不可用' if value is None else f'{value*(100 if percent else 1):.9f}'


def write_diagnostics(run_dir, arms=ARMS):
    """Verify committed best checkpoints/archives before reporting a pair.

    A stale summary after an interrupted checkpoint commit is pending, never a
    completed arm eligible for ranking. No model or test inference occurs here.
    """
    root = Path(run_dir)
    require(arms and len(set(arms)) == len(arms) and set(arms) <= set(ARMS), 'Unknown/duplicate response arms')
    payload = dict(schema=SCHEMA, status='PARTIAL', runs={}, pending={},
        predictive_gain_claim=False, test_evaluated=False,
        source='committed_best_validation_archives', selected_best_location_vs_state=None,
        interpretation='same_support_location_vs_state_not_support_ablation_or_incident_causality')
    reference, archives = None, {}
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
        require(checkpoint.get('schema') == EXPERIMENT_SCHEMA
                and identity.get('schema') == EXPERIMENT_SCHEMA and identity.get('arm') == arm
                and identity.get('training_stage') == 'frozen_diagnostic_stage1',
                'Checkpoint belongs to another response experiment or arm')
        require(summary.get('schema') == EXPERIMENT_SCHEMA and summary.get('arm') == arm
                and summary.get('training_stage') == 'frozen_diagnostic_stage1'
                and not summary.get('test_evaluated', False), 'Invalid response summary identity')
        require(not summary.get('status', '').endswith('FAILED'), 'Failed response arm cannot be ranked')
        history, epoch = checkpoint['history'], checkpoint['completed_epoch']
        require(type(epoch) is int and len(history) == epoch+1
                and [row['epoch'] for row in history] == list(range(epoch+1)), 'Invalid committed response history')
        chosen = min(range(len(history)), key=lambda i: history[i]['validation']['on']['mae_pooled'])
        require(checkpoint['best_epoch'] == chosen
                and checkpoint['best_metric'] == history[chosen]['validation']['on']['mae_pooled'],
                'Response selection certificate disagrees with history')
        if (summary['completed_epoch'] != epoch or summary['best_epoch'] != chosen
                or summary['best_mae'] != checkpoint['best_metric']):
            payload['pending'][arm] = 'summary_needs_finalization_after_checkpoint_commit'
            continue
        for field, name in (('baseline_predictions_sha256', 'baseline_predictions.npz'),
                            ('best_predictions_sha256', 'best_predictions.npz')):
            require(summary.get(field) == _hash(paths[name]), 'Response prediction archive hash changed: '+arm+'/'+name)
        keys = ('target', 'valid', 'sample_indices', 'fixed_hard_indices', 'history_low_mask')
        with np.load(paths['baseline_predictions.npz'], allow_pickle=False) as saved:
            baseline = {key: saved[key].copy() for key in ('prediction',)+keys}
        with np.load(paths['best_predictions.npz'], allow_pickle=False) as saved:
            archive = {key: saved[key].copy() for key in ('on', 'off')+keys}
        require(all(np.array_equal(archive[key], baseline[key]) for key in keys),
                'Response best and baseline axes/masks differ')
        require(np.array_equal(archive['off'], baseline['prediction']), 'Response OFF differs from baseline archive')
        indices = archive['sample_indices']
        require(indices.ndim == 1 and indices.dtype.kind in 'iu' and len(indices) == len(archive['on'])
                and np.unique(indices).size == len(indices), 'Invalid global response sample indices')
        if reference is None:
            reference = baseline
        else:
            require(all(np.array_equal(reference[key], baseline[key]) for key in reference),
                    'Response arms do not share identical baseline, targets, masks and hard set')
        pair = summarize_response_pair(archive['on'], archive['off'], archive['target'], archive['valid'],
                                       archive['fixed_hard_indices'], archive['history_low_mask'])
        require(np.isclose(pair['on']['mae_pooled'], summary['best_mae'], rtol=1e-12, atol=1e-12),
                'Response best archive MAE differs from committed selection')
        payload['runs'][arm] = dict(completed_epoch=epoch, best_epoch=chosen,
            trainable_parameters=summary['trainable_parameters'],
            engineering_only=bool(summary.get('engineering_only', False)),
            validation_rows=len(indices), pair=pair)
        archives[arm] = archive
    scopes = {row['engineering_only'] for row in payload['runs'].values()}
    require(len(scopes) <= 1, 'Engineering and formal response arms cannot be mixed')
    payload['engineering_only'] = next(iter(scopes)) if scopes else None
    if len(payload['runs']) == len(arms):
        payload['status'] = 'COMPLETE'
    if set(archives) == set(ARMS):
        location, state = payload['runs']['location'], payload['runs']['state']
        if location['completed_epoch'] == state['completed_epoch']:
            a = archives['location']
            payload['selected_best_location_vs_state'] = dict(
                completed_training_budget=location['completed_epoch'],
                location_best_epoch=location['best_epoch'], state_best_epoch=state['best_epoch'],
                scope='separately_selected_best_at_equal_completed_budget_exploratory_validation',
                pair=summarize_pair(a['on'], archives['state']['on'], a['target'], a['valid'], a['fixed_hard_indices']))
        else:
            payload['comparison_deferred'] = 'unequal_completed_training_budgets'
    _write(root/'risk_diagnostics.json', payload)
    lines = ['# MIN-R1：事故条件持续响应模块', '',
        '范围：'+('工程检查，仅验证实现和续跑。' if payload['engineering_only'] else
                 '作者发布验证集上的条件离线探索。' if payload['engineering_only'] is False else '制品尚未完整。'), '',
        'location 使用低秩关系交互；state 不读取关系数值。两组都保留原事故支持范围，并共同训练新增小模块。', '',
        'MIN 固定；零更新和 OFF 回到原 MIN。参考轨迹采用 C0，状态组不会被关系参考差分抵消为零。', '',
        '训练保护组由输入历史十二槽流量均值定义；下面的真实流量桶与高估/低估是事后诊断，不是推理输入。', '',
        '正收益表示 MIN 误差减候选误差；MAPE 显示为百分数。最佳验证轮次包括零轮，未评价测试集。', '',
        '| 组 | 完成轮次 | 最佳轮次 | 可训练参数 | MAE | RMSE pooled | MAPE % | MAE收益 | 上调占比 % |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for arm, run in payload['runs'].items():
        pair = run['pair']
        risk = pair['risks']['groups']['all_nodes']
        lines.append('| '+' | '.join((arm, str(run['completed_epoch']), str(run['best_epoch']),
            str(run['trainable_parameters']), _number(pair['on']['mae_pooled']),
            _number(pair['on']['rmse_pooled']), _number(pair['on']['mape_pooled'], True),
            _number(pair['paired']['mae_gain_pooled']),
            _number(risk['prediction_change']['up_fraction'], True)))+' |')
        lines += ['', '## '+arm+'：条件风险', '',
            '| 分组 | 有效点数 | ON MAE | MIN MAE | MAE收益 | MAPE收益（百分点） |',
            '|---|---:|---:|---:|---:|---:|']
        for name, row in pair['risks']['groups'].items():
            lines.append('| '+' | '.join((name, str(row['valid_count']), _number(row['on']['mae']),
                _number(row['off']['mae']), _number(row['gain']['mae']), _number(row['gain']['mape'], True)))+' |')
        low = pair['history_low_group']
        lines += ['', '输入历史低流量组 MAE 收益：'+_number(
            low.get('gain', {}).get('mae_gain_pooled'))+'；有效点数：'+str(low.get('valid_count', 0))+'。',
            '固定困难窗口 MAE 收益：'+_number(pair['fixed_hard_windows'].get('mae_gain'))+'。', '',
            '| 目标槽 | ON MAE | MIN MAE | MAE收益 |', '|---|---:|---:|---:|']
        for slot, (a, b, gain) in enumerate(zip(pair['on']['per_horizon_mae'], pair['off']['per_horizon_mae'],
                                               pair['paired']['per_horizon_mae_gain']), 1):
            lines.append('| '+' | '.join((str(slot), _number(a), _number(b), _number(gain)))+' |')
    lines += ['', '预算与潜状态没有认证物理单位，目标槽不自动解释为已认证的未来分钟数。',
        '预测上移本身不代表有害；需要结合误差与条件风险。局部平衡不等于输出流量守恒。',
        '两组使用相同支持范围，不能据此否定事故位置价值。反复使用验证集的结果属于探索性诊断。',
        '数值明细：risk_diagnostics.json；各组 history.json、summary.json 与 best_predictions.npz。', '']
    temporary = root/'report.md.partial'
    temporary.write_text('\n'.join(lines), encoding='utf-8')
    temporary.replace(root/'report.md')
    return payload
