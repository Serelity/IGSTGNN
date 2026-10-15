"""Pure NumPy metrics for exact and bounded relative F1 coefficients.

The four F/U/R/E predictions form a checkpoint sensitivity factorial, not a
training experiment.  Every group uses the saved validity and the same current
FUSION_OFF residual labels. No model, checkpoint or traffic loader is imported.
"""
from pathlib import Path

import numpy as np

from experiments.chronological.min_early_event_gain_metrics import (
    COUNT_FIELDS, FLOW_EDGES, PARTITION, SUM_FIELDS, _case_metrics, _divide,
    _macro, _mean, _summary, _values, flat_rows, require,
)
from experiments.chronological.min_early_event_direction_metrics import DIRECTIONS, direction_codes
from experiments.chronological.min_forecast_fusion_gain_metrics import (
    BASE_CASES as ARCHIVED_CASES, _accumulate, _groups, _pairwise_bincount,
    _sum_plan, _validate as _validate_archived,
)


SCHEMA = 'min_forecast_fusion_relative_metrics_v1'
BASE_CASES = (*ARCHIVED_CASES, 'EVENT_RELATIVE_EXACT', 'EVENT_RELATIVE_BOUNDED')
COMPARISONS = {
    'EXACT_vs_FUSION_OFF': ('EVENT_FUSION_OFF', 'EVENT_RELATIVE_EXACT'),
    'EXACT_vs_ON': ('EVENT_ON', 'EVENT_RELATIVE_EXACT'),
    'EXACT_vs_UNIFORM': ('EVENT_UNIFORM', 'EVENT_RELATIVE_EXACT'),
    'BOUNDED_vs_FUSION_OFF': ('EVENT_FUSION_OFF', 'EVENT_RELATIVE_BOUNDED'),
    'BOUNDED_vs_ON': ('EVENT_ON', 'EVENT_RELATIVE_BOUNDED'),
    'BOUNDED_vs_EXACT': ('EVENT_RELATIVE_EXACT', 'EVENT_RELATIVE_BOUNDED'),
}
DIRECTION_COMPARISONS = ('EXACT_vs_FUSION_OFF', 'EXACT_vs_ON',
                         'BOUNDED_vs_FUSION_OFF', 'BOUNDED_vs_ON')


def _selections(active, masks, y, values, edges, offsets, partition_ids, fixed_rows):
    entire = np.ones(len(y), bool)
    flow = np.searchsorted(edges, y, side='right')-1
    current = np.where(values['EVENT_FUSION_OFF'] > y, 0,
                       np.where(values['EVENT_FUSION_OFF'] < y, 1, 2))
    source = np.where(values['SOURCE_MIN'] > y, 0,
                      np.where(values['SOURCE_MIN'] < y, 1, 2))
    selections = []
    for i, mask in enumerate(masks.values()):
        selected = mask[active]
        selections.append((i, 1, selected, np.zeros(int(selected.sum()), np.int64)))
    selections.extend(((offsets['flow'], len(edges), entire, flow),
        (offsets['bias'], 3, entire, current), (offsets['source_bias'], 3, entire, source),
        (offsets['cross'], len(edges)*9, entire, (flow*3+current)*3+partition_ids[active])))
    selected = np.broadcast_to(fixed_rows[:, None], active.shape)[active]
    selections.append((offsets['fixed'], 1, selected, np.zeros(int(selected.sum()), np.int64)))
    return selections, entire


def _sum_fields(acc, h, selections, fields):
    for offset, size, selected, labels in selections:
        plan = _sum_plan(labels, size)
        for name, value in fields.items():
            acc[name][offset:offset+size, h] = _pairwise_bincount(value[selected], plan)


def _families(names, offsets):
    return dict(report_partition=[names.index(name) for name in PARTITION],
        flow=list(range(offsets['flow'], offsets['bias'])),
        current_MIN_bias=list(range(offsets['bias'], offsets['source_bias'])),
        source_MIN_bias=list(range(offsets['source_bias'], offsets['cross'])),
        cross=list(range(offsets['cross'], offsets['fixed'])))


def _check_partition(values, families, global_index, scale, context):
    eps = np.finfo(np.float64).eps
    result = {}
    for family, ids in families.items():
        residual = values[ids].sum(0)-values[global_index]
        limit = 1e-10+128*eps*scale
        require((np.abs(residual) <= limit).all(),
            'Relative F1 family conservation failed: '+context+' '+family
            +' residual='+str(residual.tolist())+' limit='+str(limit.tolist()))
        result[family] = dict(status='PASS', maximum_absolute_sum_residual=float(np.abs(residual).max()),
                              sum_tolerance_per_horizon=np.asarray(limit).tolist())
    return result


def summarize(predictions, target, valid, regions, sample_indices, station_ids,
              fixed_hard_sample_ids, response_tol=.001):
    """Summarize ten states and six paired relative-coefficient comparisons.

    Both new candidates retain report inputs. Their comparisons are fusion
    coefficient sensitivities, and must not be called accident-content gains.
    """
    arrays, target, valid, masks, samples, stations, hard_ids, edges, response_tol = _validate_archived(
        predictions, target, valid, regions, sample_indices, station_ids, fixed_hard_sample_ids,
        FLOW_EDGES, response_tol)
    for case in BASE_CASES[len(ARCHIVED_CASES):]:
        require(case in predictions, 'Both exact and bounded relative predictions are required')
        value = np.asarray(predictions[case])
        require(value.shape == target.shape and value.dtype.kind in 'fiu' and np.isfinite(value[valid]).all(),
                'New relative predictions must share numeric axes and finite valid positions')
        arrays[case] = value
    names, definitions, offsets = _groups(masks, edges)
    g, horizons = len(names), 12
    accumulators = {key: {field: np.zeros((g, horizons), np.int64 if field in COUNT_FIELDS else np.float64)
        for field in (*SUM_FIELDS, *COUNT_FIELDS, 'prediction_max')} for key in COMPARISONS}
    case_sums = {case: {field: np.zeros((g, horizons)) for field in ('absolute', 'squared', 'percentage')}
                 for case in BASE_CASES}
    interaction_sums = {field: np.zeros((g, horizons)) for field in ('net', 'positive', 'negative', 'percentage')}
    direction_counts = {key: np.zeros((g, horizons, 8), np.int64) for key in DIRECTION_COMPARISONS}
    direction_sums = {key: np.zeros((g, horizons, 8)) for key in DIRECTION_COMPARISONS}
    partition_ids = np.zeros(target.shape[::2], np.int8)
    for i, name in enumerate(PARTITION):
        partition_ids[masks[name]] = i
    fixed_rows = np.isin(samples, hard_ids)
    window_counts = np.zeros((len(samples), horizons), np.int64)
    window_errors = {case: np.zeros((len(samples), horizons)) for case in BASE_CASES}
    max_pointwise_identity_residual = 0.
    eps = np.finfo(np.float64).eps
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        try:
            for h in range(horizons):
                active = valid[:, h]
                window_counts[:, h] = active.sum(1)
                y = target[:, h][active].astype(np.float64)
                if not y.size:
                    continue
                values = {case: value[:, h][active].astype(np.float64) for case, value in arrays.items()}
                errors = {case: np.abs(value-y) for case, value in values.items()}
                percentage = {case: np.divide(error, y, out=np.zeros_like(y), where=y > 0.)
                              for case, error in errors.items()}
                selections, entire = _selections(active, masks, y, values, edges, offsets, partition_ids, fixed_rows)
                row_ids = np.broadcast_to(np.arange(len(samples))[:, None], active.shape)[active]
                row_plan = _sum_plan(row_ids, len(samples))
                for case in BASE_CASES:
                    _sum_fields(case_sums[case], h, selections, dict(absolute=errors[case],
                        squared=np.square(errors[case]), percentage=percentage[case]))
                    window_errors[case][:, h] = _pairwise_bincount(errors[case], row_plan)
                f, u, r, e = (errors[case] for case in
                    ('EVENT_FUSION_OFF', 'EVENT_UNIFORM', 'EVENT_RELATIVE_EXACT', 'EVENT_ON'))
                interaction = (u+r)-(e+f)
                full_gain, common_gain, relative_gain = f-e, f-u, f-r
                residual = full_gain-common_gain-relative_gain-interaction
                scale = f+u+r+e+1.
                max_pointwise_identity_residual = max(max_pointwise_identity_residual, float(np.abs(residual).max()))
                require((np.abs(residual) <= 1e-12+32*eps*scale).all(), 'Exact relative factorial identity failed')
                fp, up, rp, ep = (percentage[case] for case in
                    ('EVENT_FUSION_OFF', 'EVENT_UNIFORM', 'EVENT_RELATIVE_EXACT', 'EVENT_ON'))
                _sum_fields(interaction_sums, h, selections, dict(net=interaction,
                    positive=np.maximum(interaction, 0.), negative=np.maximum(-interaction, 0.),
                    percentage=(up+rp)-(ep+fp)))
                for key, (left, right) in COMPARISONS.items():
                    gain, delta = errors[left]-errors[right], values[right]-values[left]
                    sums = dict(control_absolute=errors[left], candidate_absolute=errors[right],
                        control_squared=np.square(errors[left]), candidate_squared=np.square(errors[right]),
                        control_percentage=percentage[left], candidate_percentage=percentage[right], gain_net=gain,
                        gain_positive=np.maximum(gain, 0.), gain_negative=np.maximum(-gain, 0.),
                        prediction_signed=delta, prediction_absolute=np.abs(delta))
                    counts = dict(count=entire, positive_target_count=y > 0., prediction_up=delta > 0.,
                        prediction_down=delta < 0., prediction_unchanged=delta == 0.,
                        prediction_response=np.abs(delta) > response_tol, gain_improved=gain > 0.,
                        gain_worsened=gain < 0., gain_equal=gain == 0.)
                    _accumulate(accumulators[key], h, selections, sums, counts, np.abs(delta))
                    if key in DIRECTION_COMPARISONS:
                        directions = direction_codes(values[left], values[right], y)
                        for offset, size, selected, labels in selections:
                            bins = labels*8+directions[selected]
                            direction_counts[key][offset:offset+size, h] = np.bincount(bins, minlength=size*8).reshape(size, 8)
                            direction_sums[key][offset:offset+size, h] = _pairwise_bincount(
                                gain[selected], _sum_plan(bins, size*8)).reshape(size, 8)
        except FloatingPointError as error:
            raise ValueError('Finite archived values overflowed float64 relative F1 arithmetic') from error
    require(all(np.isfinite(value).all() for acc in accumulators.values() for value in acc.values())
        and all(np.isfinite(value).all() for acc in case_sums.values() for value in acc.values())
        and all(np.isfinite(value).all() for value in (*interaction_sums.values(), *direction_sums.values())),
        'Nonfinite relative F1 accumulation')
    global_index = names.index('all_nodes')
    reference = accumulators['EXACT_vs_FUSION_OFF']
    counts, positive_counts = reference['count'], reference['positive_target_count']
    global_counts, global_positive = counts[global_index], positive_counts[global_index]
    families = _families(names, offsets)
    comparisons = {key: dict(control=left, candidate=right,
        scope='same_checkpoint_relative_coefficient_sensitivity_report_remains_ON_not_incident_gain',
        groups={name: _summary(accumulators[key], i, global_counts, global_positive) for i, name in enumerate(names)})
        for key, (left, right) in COMPARISONS.items()}
    case_groups = {case: {name: dict(
        status='available' if (counts[i] > 0).all() else 'partial_horizons' if counts[i].sum() else 'unavailable',
        valid_count=int(counts[i].sum()), valid_count_per_horizon=counts[i].tolist(),
        positive_target_count=int(positive_counts[i].sum()),
        **_case_metrics(acc['absolute'][i], acc['squared'][i], acc['percentage'][i], counts[i], positive_counts[i]))
        for i, name in enumerate(names)} for case, acc in case_sums.items()}
    reconciliation = dict(counts={}, comparisons={}, cases={}, interaction={})
    for family, ids in families.items():
        require(np.array_equal(counts[ids].sum(0), global_counts)
            and np.array_equal(positive_counts[ids].sum(0), global_positive), 'Relative F1 count conservation failed: '+family)
        reconciliation['counts'][family] = 'PASS'
    for key, acc in accumulators.items():
        require(np.array_equal(acc['count'], counts)
            and np.array_equal(acc['positive_target_count'], positive_counts),
            'Relative F1 comparisons must retain identical saved validity: '+key)
        reconciliation['comparisons'][key] = dict(
            mae=_check_partition(acc['gain_net'], families, global_index,
                acc['control_absolute'][global_index]+acc['candidate_absolute'][global_index]+1., key+' MAE'),
            mape=_check_partition(acc['control_percentage']-acc['candidate_percentage'], families, global_index,
                acc['control_percentage'][global_index]+acc['candidate_percentage'][global_index]+1., key+' MAPE'))
    for case, acc in case_sums.items():
        reconciliation['cases'][case] = {field: _check_partition(value, families, global_index,
            value[global_index]+1., case+' '+field) for field, value in acc.items()}
    interaction_scale = sum(case_sums[case]['absolute'][global_index] for case in
        ('EVENT_FUSION_OFF', 'EVENT_UNIFORM', 'EVENT_RELATIVE_EXACT', 'EVENT_ON'))+1.
    reconciliation['interaction'] = _check_partition(interaction_sums['net'], families, global_index,
                                                    interaction_scale, 'factorial interaction')
    interaction_groups = {}
    max_grouped_identity_residual = 0.
    for i, name in enumerate(names):
        acc = {case: case_sums[case]['absolute'][i] for case in
            ('EVENT_FUSION_OFF', 'EVENT_UNIFORM', 'EVENT_RELATIVE_EXACT', 'EVENT_ON')}
        f, u, r, e = (acc[case] for case in
            ('EVENT_FUSION_OFF', 'EVENT_UNIFORM', 'EVENT_RELATIVE_EXACT', 'EVENT_ON'))
        net = interaction_sums['net'][i]
        residual = (f-e)-(f-u)-(f-r)-net
        max_grouped_identity_residual = max(max_grouped_identity_residual, float(np.abs(residual).max()))
        require((np.abs(residual) <= 1e-10+128*eps*(f+u+r+e+1.)).all(), 'Grouped exact relative factorial identity failed')
        interaction_groups[name] = dict(
            status=case_groups['EVENT_ON'][name]['status'], valid_count=int(counts[i].sum()),
            valid_count_per_horizon=counts[i].tolist(),
            interaction_mae_pooled=_mean(net.sum(), counts[i].sum()),
            interaction_mae_macro=_macro(_divide(net, counts[i]), counts[i]),
            interaction_mape_pooled=_mean(interaction_sums['percentage'][i].sum(), positive_counts[i].sum()),
            interaction_mape_macro=_macro(_divide(interaction_sums['percentage'][i], positive_counts[i]), positive_counts[i]),
            positive_interaction_pooled=_mean(interaction_sums['positive'][i].sum(), counts[i].sum()),
            negative_interaction_pooled=_mean(interaction_sums['negative'][i].sum(), counts[i].sum()),
            global_mae_contribution_pooled=_mean(net.sum(), global_counts.sum()),
            global_mae_contribution_macro=_macro(_divide(net, global_counts), global_counts),
            per_horizon_interaction_mae=_values(_divide(net, counts[i]), counts[i]),
            per_horizon_global_mae_contribution=_values(_divide(net, global_counts), global_counts))
    directions_result = {}
    for key in DIRECTION_COMPARISONS:
        groups = {}
        for i, name in enumerate(names):
            net = direction_sums[key][i]
            require(np.array_equal(direction_counts[key][i].sum(1), counts[i]), 'Relative direction count conservation failed')
            scale = accumulators[key]['control_absolute'][i]+accumulators[key]['candidate_absolute'][i]+1.
            require((np.abs(net.sum(1)-accumulators[key]['gain_net'][i]) <= 1e-10+128*eps*scale).all(),
                    'Relative direction gain conservation failed')
            groups[name] = dict(valid_count=int(counts[i].sum()), directions={label: dict(
                valid_count=int(direction_counts[key][i, :, j].sum()),
                valid_count_per_horizon=direction_counts[key][i, :, j].tolist(),
                within_group_fraction=_mean(direction_counts[key][i, :, j].sum(), counts[i].sum()),
                gain_pooled=_mean(net[:, j].sum(), direction_counts[key][i, :, j].sum()),
                global_mae_contribution_pooled=_mean(net[:, j].sum(), global_counts.sum()),
                global_mae_contribution_macro=_macro(_divide(net[:, j], global_counts), global_counts),
                per_horizon_mae_gain=_values(_divide(net[:, j], direction_counts[key][i, :, j]), direction_counts[key][i, :, j]))
                for j, label in enumerate(DIRECTIONS)})
        left, right = COMPARISONS[key]
        directions_result[key] = dict(control=left, candidate=right, direction_baseline=left, groups=groups)
    complete = fixed_rows & (window_counts > 0).all(1)
    window_values = {case: _divide(error, window_counts).mean(1) for case, error in window_errors.items()}
    fixed = dict(selector='saved_SOURCE_MIN_fixed_sample_ids_no_reselection', sample_indices=hard_ids.tolist(),
        selected_count=len(hard_ids), complete_window_count=int(complete.sum()), comparisons={})
    for key, (left, right) in COMPARISONS.items():
        gain = window_values[left][complete]-window_values[right][complete]
        fixed['comparisons'][key] = dict(window_equal_weight_mean_gain=float(gain.mean()) if gain.size else None,
            window_improved_fraction=float((gain > 0).mean()) if gain.size else None,
            window_worsened_fraction=float((gain < 0).mean()) if gain.size else None,
            point_metrics=comparisons[key]['groups']['fixed_hard_windows'])
    wi = (window_values['EVENT_UNIFORM']+window_values['EVENT_RELATIVE_EXACT']
          -window_values['EVENT_ON']-window_values['EVENT_FUSION_OFF'])[complete]
    fixed['factorial_interaction_window_equal_weight'] = float(wi.mean()) if wi.size else None
    return dict(schema=SCHEMA, status='MIN_FORECAST_FUSION_RELATIVE_METRICS_PASS',
        cases={case: groups['all_nodes'] for case, groups in case_groups.items()}, case_groups=case_groups,
        comparisons=comparisons, group_definitions=definitions, reconciliation=reconciliation,
        directions=directions_result, fixed_hard_windows=fixed,
        factorial=dict(schema='exact_relative_checkpoint_two_factor_error_interaction_v1',
            states=dict(F='EVENT_FUSION_OFF', U='EVENT_UNIFORM', R='EVENT_RELATIVE_EXACT', E='EVENT_ON'),
            formula='error(U)+error(R)-error(E)-error(F)',
            error_measure='pointwise_absolute_error_RMSE_has_no_such_linear_decomposition',
            positive_interpretation='joint_error_reduction_exceeds_sum_of_each_individual_reduction',
            identity='F_minus_E=(F_minus_U)+(F_minus_R)+interaction',
            identity_status='POINTWISE_AND_GROUPED_FACTORIAL_IDENTITY_PASS',
            max_pointwise_identity_residual=max_pointwise_identity_residual,
            max_grouped_identity_residual=max_grouped_identity_residual,
            groups=interaction_groups, bounded_excluded=True,
            scope='same_checkpoint_descriptive_interaction_not_training_or_incident_causality'),
        axis=dict(samples=len(samples), horizons=12, stations=len(stations),
            sample_indices=samples.astype(int).tolist(), station_ids=stations.astype(int).tolist()),
        definitions=dict(primary_paper_baseline='original_IGSTGNN_not_in_this_ten_case_archive',
            SOURCE_MIN_role='intermediate_structural_control_not_original_IGSTGNN',
            gain='control_absolute_error_minus_candidate_absolute_error',
            current_MIN_bias='EVENT_FUSION_OFF_minus_target_fixed_for_all_comparisons',
            source_MIN_bias='SOURCE_MIN_minus_target_separately_labelled',
            candidate_report_status='ON_for_exact_and_bounded_not_an_incident_content_ablation',
            relative_exact='delta_minus_its_component_mean_without_rescaling',
            relative_bounded='relative_exact_rescaled_if_needed_to_preserve_the_original_increment_bound',
            finite_precision_note='exact_means_no_amplitude_rescaling_not_infinite_precision_cached_uniform_and_new_centering_may_round_differently',
            zero_sum_coefficients_do_not_imply_zero_output_shift=True,
            pooled='saved_valid_points_equal_weight', macro='12_horizon_means_only_when_all_nonempty',
            global_contribution='common_all_nodes_denominator_pooled_or_per_horizon_macro',
            mape_unit='fraction', mape_zero_policy='saved_validity_preserved_positive_targets_only_for_MAPE',
            grouping_uses_targets_posthoc_only=True, direction_uses_each_comparison_own_control=True,
            prediction_change='candidate_minus_control', response_tol=response_tol,
            response_tolerance_affects_response_counts_only=True,
            weighted_summation='64_input_blocks_then_explicit_pairwise_float64_merging',
            empty_groups='local_means_unavailable_common_denominator_zero_mass_contribution_zero',
            factorial_bounded_is_not_a_pure_relative_factor=True,
            training_and_incident_causal_attribution=False,
            scope='exploratory_validation_checkpoint_sensitivity_no_training_no_test'))


def write_report(output, result, diagnostics=None):
    """Write a self-contained Chinese Markdown report; MAPE renders as percent."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    fmt = lambda value: '不可用' if value is None else format(float(value), '.9f')
    lines = ['# MIN-F1 相对权重：同检查点诊断', '',
        '原 IGSTGNN 仍为论文主 baseline；当前十组预测归档不包含它，本报告不能证明超过原 IGSTGNN。', '',
        '使用同一 EVENT 最佳检查点与原验证批次。新增精确中心化和幅度受限中心化两种推理状态；不训练、不评价测试集。',
        '这是已用于设计和选模的验证集上的探索性敏感性，不证明中心化后重新训练的效果。', '',
        '两种新状态均保留报告输入，比较衡量系数变换作用，不能标为事故内容增益。', '',
        '精确中心化：r = delta − mean(delta)。受限中心化：必要时对 r 统一缩小以保持原系数增量界限。',
        '系数零和不保证预测输出不偏移；十个预测分量并不相同。精确状态可能超过原幅度界限，应结合机制诊断判断。', '',
        '“精确”指不额外收缩幅度，不指无限精度；缓存统一权重与新中心化的计算精度可能略有差异。交互量依据实际四组预测计算，不宣称无限精度正交分解。', '',
        '## 十种状态', '',
        '| 状态 | MAE pooled | MAE macro | RMSE pooled | RMSE macro | MAPE (%) |',
        '|---|---:|---:|---:|---:|---:|']
    for case, metrics in result['cases'].items():
        lines.append('| '+case+' | '+' | '.join(fmt(metrics[k]) for k in
            ('mae_pooled', 'mae_macro', 'rmse_pooled', 'rmse_macro'))+' | '
            +fmt(None if metrics['mape_pooled'] is None else 100*metrics['mape_pooled'])+' |')
    lines += ['', '## 配对收益与固定困难窗口', '',
        '收益为对照误差减候选误差；正值表示候选更好。困难集直接复用源 MIN 保存的样本 ID，绝不重选。', '',
        '| 比较 | MAE pooled | MAE macro | RMSE pooled | MAPE收益（百分点） | 固定窗口收益 |',
        '|---|---:|---:|---:|---:|---:|']
    for key, comparison in result['comparisons'].items():
        row = comparison['groups']['all_nodes']
        lines.append('| '+key+' | '+' | '.join(fmt(row[k]) for k in
            ('mae_gain_pooled', 'mae_gain_macro', 'rmse_gain_pooled'))+' | '
            +fmt(None if row['mape_gain_pooled'] is None else 100*row['mape_gain_pooled'])+' | '
            +fmt(result['fixed_hard_windows']['comparisons'][key]['window_equal_weight_mean_gain'])+' |')
    primary = ('EXACT_vs_FUSION_OFF', 'EXACT_vs_ON', 'BOUNDED_vs_FUSION_OFF', 'BOUNDED_vs_ON')
    for title, selected in [('流量与当前 MIN 偏差', lambda name, definition: definition['family'] in ('flow', 'current_MIN_bias')),
                            ('道路与事故三分区', lambda name, definition: name in PARTITION or definition['family'] == 'road')]:
        lines += ['', '## '+title, '', '| 分组 | 有效点数 | '+' | '.join(primary)+' |',
                  '|---|---:|'+'---:|'*len(primary)]
        for name, definition in result['group_definitions'].items():
            if selected(name, definition):
                rows = [result['comparisons'][key]['groups'][name] for key in primary]
                lines.append('| '+name+' | '+str(rows[0]['valid_count'])+' | '
                             +' | '.join(fmt(row['mae_gain_pooled']) for row in rows)+' |')
    lines += ['', '## 逐预测时距', '', '| 步 | '+' | '.join(primary)+' |', '|---:|'+'---:|'*len(primary)]
    for h in range(12):
        lines.append('| '+str(h+1)+' | '+' | '.join(fmt(result['comparisons'][key]['groups']['all_nodes']['per_horizon_mae_gain'][h])
                                                               for key in primary)+' |')
    lines += ['', '## 修正方向', '', '各比较按自己的对照预测判断方向，当前 MIN 偏差分组则始终由 FUSION_OFF 定义。', '',
        '| 比较 | 方向 | 点数 | 组内MAE收益 | 全网MAE贡献 |', '|---|---|---:|---:|---:|']
    for key, item in result['directions'].items():
        for direction, row in item['groups']['all_nodes']['directions'].items():
            lines.append('| '+key+' | '+direction+' | '+str(row['valid_count'])+' | '+fmt(row['gain_pooled'])+' | '
                         +fmt(row['global_mae_contribution_pooled'])+' |')
    lines += ['', '## 精确中心化的四状态交互', '',
        'F为固定求和，U为共同缩放，R为精确相对权重，E为完整融合。逐点交互 = error(U)+error(R)−error(E)−error(F)。',
        '这里 error 指逐点绝对误差；RMSE 不能套用这个线性分解。',
        '正交互表示联合降低误差超过两个单独作用之和；它是同检查点描述，不是训练或事故因果。受限中心化因另有缩放，不属于这个纯四状态分解。', '',
        '| 分组 | 交互MAE pooled | 交互MAE macro | 全网贡献 |', '|---|---:|---:|---:|']
    for name, row in result['factorial']['groups'].items():
        if name == 'all_nodes' or result['group_definitions'][name]['family'] in ('flow', 'current_MIN_bias') or name in PARTITION:
            lines.append('| '+name+' | '+fmt(row['interaction_mae_pooled'])+' | '+fmt(row['interaction_mae_macro'])
                         +' | '+fmt(row['global_mae_contribution_pooled'])+' |')
    lines += ['', '同点与分组恒等式检查：'+result['factorial']['identity_status']+'。', '',
        '## 解释与后续选择', '',
        '应联合查看相对固定求和与相对完整融合的结果：优于完整融合不等于优于当前 MIN 固定求和。',
        '优先检查低流量 MAE/MAPE、长时距、原固定困难集，以及精确和受限状态是否一致；不只按一个全网均值决定结构。',
        '真值流量、偏差及修正方向仅用于事后诊断，不可作为在线输入；事故关联包络不是认证影响范围。',
        '窗口与预测点可能重叠，不视为独立事故重复。均值权重、零和系数及交互量均不等于模块重要性或真实物理容量。',
        '仅本轮新增两种推理状态；已有八组预测继续沿用保存制品。此报告不重新认证历史源检查点。']
    if diagnostics:
        import json
        lines += ['', '## 系数变换与保护诊断', '', '```json', json.dumps(diagnostics, ensure_ascii=False, indent=2), '```']
    lines += ['', '数值制品：analysis.json、cases.csv、comparisons.csv、flow.csv、current_bias.csv、source_bias.csv、roads.csv、regions.csv、cross.csv、group_horizons.csv、directions.csv、factorial.csv、fixed_hard.csv、paired_predictions.npz。', '']
    path = output/'report.md'
    path.write_text('\n'.join(lines), encoding='utf-8')
    return path
