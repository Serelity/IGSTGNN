"""Descriptive content contrasts with every stratum fixed by the original MIN.

The true location and one support-preserving shuffled location are compared at
the same weights. Target flow and MIN error signs are post-hoc diagnostics,
never prediction-time inputs. MAPE values and contributions use fraction units.
"""
import csv
import json
from pathlib import Path

import numpy as np

try:
    from .min_frozen_budget_metrics import summarize_pair
    from .min_frozen_gain_metrics import summarize_strata
except ImportError:
    from min_frozen_budget_metrics import summarize_pair
    from min_frozen_gain_metrics import summarize_strata


SCHEMA = 'min_frozen_location_content_probe_metrics_v1'
FAMILIES = ('flow', 'bias', 'regions', 'cross')
COUNT_FIELDS = ('count', 'valid_count', 'valid_count_per_horizon',
                'positive_target_count', 'positive_target_count_per_horizon',
                'zero_target_count', 'valid_fraction_pooled',
                'positive_target_fraction_pooled')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _difference(left, right):
    require((left is None) == (right is None), 'Metric availability differs across matched cases')
    return None if left is None else float(left-right)


def _vector_difference(left, right):
    require(len(left) == len(right), 'Horizon dimensions differ across matched cases')
    return [_difference(a, b) for a, b in zip(left, right)]


def _content_row(real, shuffled):
    for field in COUNT_FIELDS:
        require(real[field] == shuffled[field], 'Fixed MIN stratum differs: '+field)
    require(real['off'] == shuffled['off'], 'Fixed MIN baseline metrics differ')
    gain = {key: _difference(shuffled['on'][key], real['on'][key])
            for key in ('mae_pooled', 'rmse_pooled', 'mape_pooled')}
    row = {key: real[key] for key in COUNT_FIELDS}
    row.update(status=real['status'], MIN=real['off'], real=real['on'], shuffled=shuffled['on'],
               gain=gain, mae_gain_pooled=gain['mae_pooled'],
               rmse_gain_pooled=gain['rmse_pooled'], mape_gain_pooled=gain['mape_pooled'],
               per_horizon_mae_gain=_vector_difference(shuffled['on']['per_horizon_mae'], real['on']['per_horizon_mae']),
               per_horizon_mape_gain=_vector_difference(shuffled['on']['per_horizon_mape'], real['on']['per_horizon_mape']))
    # Both branch contrasts use exactly the same MIN grouping and denominators.
    # Their net contributions may be subtracted; positive/negative point gains
    # must instead come from the actual real-versus-shuffled pair.
    for metric in ('mae', 'mape'):
        row['global_'+metric+'_content_contribution'] = _difference(
            real['global_'+metric+'_gain_contribution'], shuffled['global_'+metric+'_gain_contribution'])
        row['per_horizon_global_'+metric+'_content_contribution'] = _vector_difference(
            real['per_horizon_global_'+metric+'_gain_contribution'],
            shuffled['per_horizon_global_'+metric+'_gain_contribution'])
    return row


def _reconcile(content):
    global_row, output = content['global'], {}
    for family in FAMILIES:
        rows = list(content[family].values())
        for field in ('valid_count', 'positive_target_count'):
            require(sum(row[field] for row in rows) == global_row[field],
                    'Content stratum counts do not reconcile: '+family+' '+field)
        for field in ('valid_count_per_horizon', 'positive_target_count_per_horizon'):
            require(np.array_equal(np.sum([row[field] for row in rows], axis=0), global_row[field]),
                    'Content stratum horizon counts do not reconcile: '+family+' '+field)
        entry = dict(status='PASS', valid_count_sum=sum(row['valid_count'] for row in rows))
        for metric in ('mae', 'mape'):
            field = 'global_'+metric+'_content_contribution'
            wanted = global_row[field]
            actual = sum(row[field] for row in rows) if wanted is not None else None
            require(wanted is None or np.isclose(actual, wanted, rtol=1.e-10, atol=1.e-12),
                    'Content contribution does not reconcile: '+family+' '+metric)
            entry[metric+'_content_contribution_sum'] = actual
            entry[metric+'_content_residual'] = _difference(actual, wanted)
            field = 'per_horizon_global_'+metric+'_content_contribution'
            residuals = []
            for h, wanted in enumerate(global_row[field]):
                actual = sum(row[field][h] for row in rows) if wanted is not None else None
                require(wanted is None or np.isclose(actual, wanted, rtol=1.e-10, atol=1.e-12),
                        'Content horizon contribution does not reconcile: '+family+' '+metric)
                residuals.append(_difference(actual, wanted))
            entry['per_horizon_'+metric+'_content_residual'] = residuals
        output[family] = entry
    return dict(status='PASS', families=output,
                empty_group_contribution='zero_when_global_denominator_exists_otherwise_unavailable')


def summarize_location_probe(real, shuffled, baseline, target, valid, regions, fixed_hard_indices):
    """Return three paired contrasts and fixed-MIN content strata.

    Real-content gain is shuffled error minus real error. Both stratified branch
    contrasts are computed against MIN, so bias groups do not change when the
    shuffled prediction crosses the target. No significance test is performed.
    """
    real_strata = summarize_strata(real, baseline, target, valid, regions)
    shuffle_strata = summarize_strata(shuffled, baseline, target, valid, regions)
    pairs = dict(real_vs_MIN=summarize_pair(real, baseline, target, valid, fixed_hard_indices),
                 shuffle_vs_MIN=summarize_pair(shuffled, baseline, target, valid, fixed_hard_indices),
                 real_vs_shuffle=summarize_pair(real, shuffled, target, valid, fixed_hard_indices))
    content = dict(global_=_content_row(real_strata['global'], shuffle_strata['global']))
    content['global'] = content.pop('global_')
    for family in FAMILIES:
        require(real_strata[family].keys() == shuffle_strata[family].keys(), 'Fixed stratum keys differ')
        content[family] = {name: _content_row(row, shuffle_strata[family][name])
                           for name, row in real_strata[family].items()}
    for metric in ('mae', 'rmse', 'mape'):
        actual, wanted = content['global'][metric+'_gain_pooled'], pairs['real_vs_shuffle']['paired'][metric+'_gain_pooled']
        require(actual is None and wanted is None or actual is not None and wanted is not None and
                np.isclose(actual, wanted, rtol=1.e-10, atol=1.e-12), 'Content pair disagrees with fixed strata: '+metric)
    return dict(schema=SCHEMA, pairs=pairs, real_strata=real_strata, shuffle_strata=shuffle_strata,
                content_strata=content, reconciliation=_reconcile(content),
                flow_bin_definitions=real_strata['flow_bin_definitions'], mape_unit='fraction',
                fixed_bias_reference='MIN_prediction_minus_observed_target',
                signed_content_gain='shuffled_error_minus_real_error_positive_favors_correct_location',
                grouping_scope='posthoc_target_flow_fixed_MIN_error_sign_and_original_report_regions',
                scope='single_fixed_shuffle_conditional_offline_validation_checkpoint_sensitivity',
                partial_positive_negative_decomposition='not_derived_by_subtracting_branch_decompositions',
                limitations=['single_shuffle_is_not_an_independent_accident_repeat',
                             'support_preserving_shuffle_does_not_identify_support_effect',
                             'shuffled_content_may_be_out_of_distribution',
                             'no_accident_causality_or_measured_capacity_identification'])


def _number(value, percent=False):
    return '不可用' if value is None else f'{100*value if percent else value:.9f}'


def _rows(analysis):
    content = analysis['content_strata']
    yield 'global', 'all_nodes', content['global']
    for family in FAMILIES:
        for name, row in content[family].items():
            yield family, name, row


def _write_csv(root, analysis):
    fields = ['family', 'group', 'status', 'valid_count', 'positive_target_count', 'valid_fraction_pooled']
    fields += [case+'_'+metric+'_pooled' for case in ('MIN', 'real', 'shuffled') for metric in ('mae', 'rmse', 'mape')]
    fields += ['mae_gain_pooled', 'rmse_gain_pooled', 'mape_gain_pooled',
               'global_mae_content_contribution', 'global_mape_content_contribution']
    with (root/'content_strata.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for family, name, row in _rows(analysis):
            result = {key: row[key] for key in fields if key in row}
            result.update(family=family, group=name)
            result.update({case+'_'+metric+'_pooled': row[case][metric+'_pooled']
                           for case in ('MIN', 'real', 'shuffled') for metric in ('mae', 'rmse', 'mape')})
            writer.writerow(result)
    fields = ['family', 'group', 'horizon', 'valid_count', 'positive_target_count']
    fields += [case+'_'+metric for case in ('MIN', 'real', 'shuffled') for metric in ('mae', 'mape')]
    fields += ['mae_content_gain', 'mape_content_gain', 'global_mae_content_contribution', 'global_mape_content_contribution']
    with (root/'horizons.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for family, name, row in _rows(analysis):
            for h, count in enumerate(row['valid_count_per_horizon']):
                result = dict(family=family, group=name, horizon=h+1, valid_count=count,
                              positive_target_count=row['positive_target_count_per_horizon'][h])
                result.update({case+'_'+metric: row[case]['per_horizon_'+metric][h]
                               for case in ('MIN', 'real', 'shuffled') for metric in ('mae', 'mape')})
                for metric in ('mae', 'mape'):
                    result[metric+'_content_gain'] = row['per_horizon_'+metric+'_gain'][h]
                    result['global_'+metric+'_content_contribution'] = row['per_horizon_global_'+metric+'_content_contribution'][h]
                writer.writerow(result)


def write_report(output, analysis, metadata):
    """Write UTF-8 Markdown and fraction-unit CSV tables, returning the report path."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    audit = metadata.get('shuffle_audit', {})
    audit = audit if isinstance(audit, dict) else {}
    display_metadata = dict(metadata)
    if display_metadata.get('validation_rows') is None and 'rows' in audit:
        display_metadata['validation_rows'] = audit['rows']
    if audit:
        # The full per-window donor audit is a separate JSON artifact; retain
        # its coverage totals here without hundreds of repeated report rows.
        display_metadata['shuffle_audit'] = {key: value for key, value in audit.items() if key != 'per_sample'}
        display_metadata['shuffle_audit_per_sample_artifact'] = 'shuffle_audit.json'
    lines = ['# 冻结 MIN：事故位置内容对照', '',
             '固定已训练 MIN 与预算适配器，比较正确位置、一次固定置乱位置及关闭适配器的 MIN。', '',
             '这是条件离线验证集上的探索性检查点敏感性分析。没有训练更新，没有评价测试集；源文件保护结果见本次运行元数据。', '',
             '原 MIN 最佳轮次：'+str(metadata.get('source_best_epoch'))+'；适配器最佳轮次：'+str(metadata.get('best_adapter_epoch'))+
             '；验证窗口：'+str(display_metadata.get('validation_rows'))+'；固定置乱 seed：'+str(metadata.get('shuffle_seed'))+'。', '',
             '## 三种预测状态与配对收益', '',
             '正分支收益为 MIN 误差减该状态误差；正内容收益为置乱误差减正确位置误差。MAPE 表中为百分数，收益为百分点，JSON/CSV 均为比例。', '',
             '| 状态 | MAE pooled | RMSE pooled | MAPE % | 相对 MIN MAE收益 | 固定困难集MAE |',
             '|---|---:|---:|---:|---:|---:|']
    real_pair, shuffle_pair, content_pair = (analysis['pairs'][key] for key in ('real_vs_MIN', 'shuffle_vs_MIN', 'real_vs_shuffle'))
    for name, case, gain, hard in (
            ('MIN / 适配器关闭', real_pair['off'], 0. if real_pair['off']['mae_pooled'] is not None else None, real_pair['fixed_hard_windows']['off_mae']),
            ('正确位置', real_pair['on'], real_pair['mae_gain_pooled'], real_pair['fixed_hard_windows']['on_mae']),
            ('置乱位置', shuffle_pair['on'], shuffle_pair['mae_gain_pooled'], shuffle_pair['fixed_hard_windows']['on_mae'])):
        lines.append('| '+' | '.join((name, _number(case['mae_pooled']), _number(case['rmse_pooled']),
                     _number(case['mape_pooled'], True), _number(gain), _number(hard)))+' |')
    paired = content_pair['paired']
    lines += ['', '正确位置相对置乱：MAE 收益 '+_number(paired['mae_gain_pooled'])+
              '；RMSE 收益 '+_number(paired['rmse_gain_pooled'])+'；MAPE 收益（百分点） '+_number(paired['mape_gain_pooled'], True)+
              '；固定困难窗口收益 '+_number(content_pair['fixed_hard_mae_gain'])+'。', '',
              '内容配对逐点正收益 '+_number(paired['gain_positive_pooled'])+'；负收益 '+_number(paired['gain_negative_pooled'])+
              '；抵消比例 '+_number(paired['cancellation_fraction_pooled'], True)+'%。这些来自真实逐点内容对照，不是两个分支收益分解相减。', '',
              '正确位置与置乱位置预测绝对变化均值 '+_number(paired['prediction_change']['absolute_mean_pooled'])+
              '；置乱减正确预测的有符号均值 '+_number(paired['prediction_change']['signed_mean_off_minus_on'])+'。', '',
              '## 有效扰动覆盖', '',
              '| 验证窗口 | 可置乱窗口 | 实际改变窗口 | 支持站点向量 | 移动向量 | 数值实际改变向量 |',
              '|---:|---:|---:|---:|---:|---:|',
              '| '+' | '.join(str(audit.get(key, '未提供')) for key in ('rows', 'eligible_rows', 'effective_changed_rows',
                  'supported_station_count', 'moved_station_count', 'changed_vector_count'))+' |', '',
              ('本次没有数值实际改变的关联向量，不能判定关联细节是否有作用；预测不变不能据此解释为模型不需要位置内容。'
               if audit.get('effective_changed_rows') == 0 or audit.get('changed_vector_count') == 0 else
               '向量索引移动不必意味着数值改变；重复关系向量、仅一个支持站点或无支持站点会限制有效扰动覆盖。没有改动的窗口不能提供内容识别力。'), '',
              '## 按固定 MIN 分组的位置内容收益', '',
              '流量桶由真实目标固定划分，高估／低估由原 MIN − 真实目标固定划分；置乱预测不能改变组别。这些真值可见的事后分组不能用作预测门控。', '',
              '区域始终来自原事故关系编码；候选区域仅为内部图弱连通分量，不能视为真实事故传播范围，其余区域也不是已认证无事故区域。', '']
    for family, title in (('global', '全网'), ('flow', '真实流量区间'), ('bias', '原 MIN 高估／低估'), ('regions', '原事故关联区域'), ('cross', '流量 × 原 MIN 偏差 × 原事故区域')):
        lines += ['### '+title, '',
                  '| 分组 | 点数 | 点数占比 % | MIN MAE | 正确 MAE | 置乱 MAE | 内容 MAE收益 | 内容 RMSE收益 | 内容 MAPE收益（百分点） | 全网MAE贡献 | 全网MAPE贡献（百分点） |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        groups = {'all_nodes': analysis['content_strata']['global']} if family == 'global' else analysis['content_strata'][family]
        for name, row in groups.items():
            if family == 'cross' and not row['valid_count']:
                continue
            lines.append('| '+' | '.join((name.replace('|', ' / '), str(row['valid_count']), _number(row['valid_fraction_pooled'], True),
                         _number(row['MIN']['mae_pooled']), _number(row['real']['mae_pooled']), _number(row['shuffled']['mae_pooled']),
                         _number(row['mae_gain_pooled']), _number(row['rmse_gain_pooled']), _number(row['mape_gain_pooled'], True),
                         _number(row['global_mae_content_contribution']), _number(row['global_mape_content_contribution'], True)))+' |')
        lines.append('')
    lines += ['## 一致性与解释边界', '',
              '各分组共享原有效掩码，MAPE 另排除零目标；流量、偏差、区域和交叉组的点数及全网 MAE/MAPE 贡献分别核对守恒。RMSE 不做线性贡献分解。空组局部统计不可用，存在全网分母时其贡献为零。', '',
              '保留事故支持站点与支持范围的置乱，仅检验该固定模型对范围内部位置细节的敏感性，不能识别支持范围本身的贡献。置乱输入可能偏离训练分布；支持内局部向量互换不能保证全局道路关联的联合分布真实。一次固定置乱不能视为独立事故重复，不据此计算事故因果显著性。', '',
              '窗口与预测点可能重叠；正确关联优于置乱也不等于事故因果效应或真实道路容量识别。这里只比较无量纲隐空间预算适配器，不自动建议部署、扩大训练或按真值分组设计门控。', '',
              '## 运行元数据', '', '原 MIN 最佳轮次、适配器最佳轮次、固定置乱 seed、有效扰动覆盖及源文件保护记录如下；覆盖不足会限制本对照的解释范围。', '',
              '```json', json.dumps(display_metadata, ensure_ascii=False, indent=2, allow_nan=False), '```', '',
              '数值明细：analysis.json；全部分组（含空组）：content_strata.csv；逐预测步：horizons.csv。CSV 的 MAPE 与贡献单位均为比例。', '']
    path = output/'report.md'
    path.write_text('\n'.join(lines), encoding='utf-8')
    _write_csv(output, analysis)
    return path
