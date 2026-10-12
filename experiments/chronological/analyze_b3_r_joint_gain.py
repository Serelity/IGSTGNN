"""Offline region x OFF-residual x horizon decomposition of a passed B3-R probe.

Only saved validation arrays are read. This module does not import Torch or run
inference; residual categories use evaluation targets and are not gate inputs.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import platform
import uuid

import numpy as np

try:
    from . import analyze_b3_r_path_probe as base
except ImportError:
    import analyze_b3_r_path_probe as base


SCHEMA = 'b3r_joint_gain_analysis_v1'
REGIONS = base.PARTITIONS
BIASES = ('over', 'under', 'exact')
ADDITIVE_FIELDS = ('global_gain_contribution', 'global_gain_positive', 'global_gain_negative',
                   'global_valid_weight_fraction', 'global_prediction_abs_change_contribution',
                   'global_prediction_signed_change_contribution')
HORIZON_ADDITIVE_FIELDS = ('per_horizon_global_gain_contribution', 'per_horizon_global_gain_positive',
                           'per_horizon_global_gain_negative', 'per_horizon_global_valid_weight_fraction',
                           'per_horizon_global_prediction_abs_change_contribution',
                           'per_horizon_global_prediction_signed_change_contribution')


def extended_aggregate(raw, global_counts):
    metric = base.aggregate(raw, global_counts)
    counts = raw['count'].sum(0)
    signed = raw['signed_change'].sum(0)
    change = raw['change'].sum(0)
    metric.update(
        global_prediction_signed_change_contribution=float(np.mean(signed/global_counts)),
        per_horizon_prediction_signed_change_mean=[float(s/n) if n else None for s, n in zip(signed, counts)],
        per_horizon_global_prediction_abs_change_contribution=(change/global_counts).tolist(),
        per_horizon_global_prediction_signed_change_contribution=(signed/global_counts).tolist())
    return metric


def verify_partition(parts, total, label):
    """Check additive global contributions, without adding group-normalized MAEs."""
    for field in ADDITIVE_FIELDS:
        base.close(sum(part[field] for part in parts), total[field], label+': '+field)
    for field in HORIZON_ADDITIVE_FIELDS:
        base.close(np.sum([part[field] for part in parts], axis=0), total[field], label+': '+field)
    base.require(sum(part['valid_count'] for part in parts) == total['valid_count'], label+': valid count')
    base.require(np.array_equal(np.sum([part['per_horizon_valid_count'] for part in parts], axis=0),
                                total['per_horizon_valid_count']), label+': horizon counts')


def decomposition_rows(joint):
    rows = []
    for region in REGIONS:
        for bias in BIASES:
            metric = joint[region][bias]
            for horizon in ('macro', *range(1, 13)):
                macro = horizon == 'macro'
                h = None if macro else horizon-1
                value = lambda macro_key, horizon_key: metric[macro_key] if macro else metric[horizon_key][h]
                positive = value('gain_positive', 'per_horizon_gain_positive')
                negative = value('gain_negative', 'per_horizon_gain_negative')
                gain = value('off_minus_on_mae', 'per_horizon_off_minus_on')
                gross = None if positive is None else positive+negative
                contribution = value('global_gain_contribution', 'per_horizon_global_gain_contribution')
                global_positive = value('global_gain_positive', 'per_horizon_global_gain_positive')
                global_negative = value('global_gain_negative', 'per_horizon_global_gain_negative')
                signed_contribution = value('global_prediction_signed_change_contribution',
                                            'per_horizon_global_prediction_signed_change_contribution')
                absolute_contribution = value('global_prediction_abs_change_contribution',
                                              'per_horizon_global_prediction_abs_change_contribution')
                rows.append(dict(
                    region=region, off_residual_bias=bias, horizon=horizon,
                    metric_available=metric['region_metric_available'] if macro else metric['per_horizon_valid_count'][h] > 0,
                    valid_count=value('valid_count', 'per_horizon_valid_count'),
                    on_mae=value('on_mae_macro', 'per_horizon_on_mae'),
                    off_mae=value('off_mae_macro', 'per_horizon_off_mae'), off_minus_on=gain,
                    gain_positive=positive, gain_negative=negative, gross_error_gain=gross,
                    cancellation_fraction=None if gross is None or gross == 0 else max(0., 1.-abs(gain)/gross),
                    global_horizon_gain_contribution=None if macro else contribution,
                    contribution_to_global_macro_mae=contribution if macro else contribution/12.,
                    global_gain_positive=global_positive, global_gain_negative=global_negative,
                    positive_contribution_to_global_macro_mae=global_positive if macro else global_positive/12.,
                    negative_contribution_to_global_macro_mae=global_negative if macro else global_negative/12.,
                    global_valid_weight_fraction=value('global_valid_weight_fraction', 'per_horizon_global_valid_weight_fraction'),
                    prediction_signed_shift=value('prediction_signed_change_horizon_macro_mean', 'per_horizon_prediction_signed_change_mean'),
                    prediction_abs_change=value('prediction_abs_change_horizon_macro_mean', 'per_horizon_prediction_abs_change_mean'),
                    global_prediction_signed_shift_contribution=signed_contribution,
                    global_prediction_abs_change_contribution=absolute_contribution,
                    signed_shift_contribution_to_global_macro=signed_contribution if macro else signed_contribution/12.,
                    abs_change_contribution_to_global_macro=absolute_contribution if macro else absolute_contribution/12.))
    base.require(len(rows) == 117, 'Expected nine joint groups x thirteen horizon rows')
    return rows


def explain(result):
    all_nodes = result['metrics']['all_nodes']
    return '\n'.join([
        '# B3-R 区域×OFF偏差×时距联合分解', '',
        f'源范围：{result["source_scope"]}；最佳轮次：{result["source_best_epoch"]}。',
        f'全网 MAE_OFF−MAE_ON={all_nodes["off_minus_on_mae"]:.12g}。', '',
        '仅重算已有验证预测，无新推理、训练或Test读取。原TIID保持开启；这是整条D路径的检查点敏感性。', '',
        'A=direct_report_nodes，含算子外的直接报告节点；B=potential_propagation_only_nodes，即候选包络扣除A；',
        'C=remaining_nodes，即A与候选包络之外。A/B/C互斥完备，候选包络是算子内连通分量，不等于实测传播范围。',
        'OFF偏差以OFF−target定义：over>0、under<0、exact=0；这些是依赖验证目标的事后分析轴，不能作为未来门控输入。', '',
        '逐元素收益g=|OFF−target|−|ON−target|；G+=max(g,0)，G−=max(−g,0)，net=G+−G−。',
        '预测有符号变化统一为OFF−ON，绝对变化为|OFF−ON|，不把预测变化符号误称为收益符号。',
        '每个时距先除以自己的有效目标数，再对12个时距等权平均。组内任一时距为空时macro为不可用，禁止nanmean。',
        '空格的组内指标为null/CSV空白，全网贡献为0。h行的global_horizon_gain_contribution以全网该时距计数为分母；',
        'contribution_to_global_macro_mae再除以12。macro行贡献是12个时距贡献的均值，组内MAE不能直接相加。',
        'global_valid_weight_fraction是全网各时距有效计数占比，macro为12时距均值，不是所有元素的简单占比。', '',
        'joint_decomposition.csv含9个联合格×13行=117行；analysis.json保留全部向量及每一时距/宏平均的可加性核对。',
        '源文件在分析前后重新计算哈希并核对一致。这证明本次读取保全，不补充源NPZ生成时缺失的认证。',
        '联合结果只用于定位收益/损失集中处，不自动选择架构，不宣称事故因果、真实物理机制或跨种子稳定收益。', ''])


def analyze(probe, output, allow_engineering=False):
    probe, output = Path(probe).resolve(), Path(output).resolve()
    base.require(probe.is_dir(), 'Probe directory does not exist')
    base.require(output != probe and not output.is_relative_to(probe) and not probe.is_relative_to(output),
                 'Output overlaps the protected probe directory')
    output.mkdir(parents=True, exist_ok=False)
    before = {}
    try:
        paths = sorted(path for path in probe.iterdir() if path.is_file())
        before = base.snapshot(paths)
        report, provenance, replay = base.verify_source(probe, allow_engineering)
        arrays = base.load_arrays(probe, report, provenance)
        valid = arrays['valid']
        target = np.where(valid, arrays['target'], 0.).astype(np.float64)
        on_residual = arrays['on'].astype(np.float64)-target
        off_residual = arrays['off'].astype(np.float64)-target
        on_error, off_error = np.abs(on_residual), np.abs(off_residual)
        gain = off_error-on_error
        shift = arrays['off'].astype(np.float64)-arrays['on']
        errors = dict(on_error=on_error, off_error=off_error, gain=gain,
                      positive=np.maximum(gain, 0.), negative=np.maximum(-gain, 0.),
                      change=np.abs(shift), signed_change=shift)
        global_counts = valid.sum((0, 2, 3))
        regions = base.region_masks(arrays)
        region_raw = {name: base.raw_statistics(errors, valid, mask) for name, mask in regions.items()}
        metrics = {name: extended_aggregate(raw, global_counts) for name, raw in region_raw.items()}
        base.verify_metrics(report, replay, metrics)
        bias_masks = dict(over=off_residual > 0, under=off_residual < 0, exact=off_residual == 0)
        base.require(np.all(sum(mask.astype(np.int8) for mask in bias_masks.values())[valid] == 1),
                     'OFF residual groups do not partition valid targets')
        bias_metrics = {bias: extended_aggregate(base.raw_statistics(errors, valid, mask), global_counts)
                        for bias, mask in bias_masks.items()}
        joint = {region: {bias: extended_aggregate(base.raw_statistics(
                    errors, valid, regions[region][:, None, :, None] & bias_masks[bias]), global_counts)
                          for bias in BIASES} for region in REGIONS}
        verify_partition([metric for row in joint.values() for metric in row.values()], metrics['all_nodes'], 'Nine joint cells')
        for region in REGIONS:
            verify_partition(list(joint[region].values()), metrics[region], region+' bias marginal')
        for bias in BIASES:
            verify_partition([joint[region][bias] for region in REGIONS], bias_metrics[bias], bias+' region marginal')
        rows = decomposition_rows(joint)
        result = dict(
            status='B3R_JOINT_GAIN_ANALYSIS_PASS', schema=SCHEMA, source_probe=str(probe),
            source_scope=report['scope'], source_best_epoch=report['best_epoch'],
            validation_rows=len(arrays['sample_indices']), nodes=arrays['on'].shape[2],
            region_axis=list(REGIONS), off_residual_bias_axis=list(BIASES), horizons=12, csv_rows=len(rows),
            off_residual_definition='OFF_minus_target; over_gt_0_under_lt_0_exact_eq_0',
            signed_prediction_shift_definition='OFF_minus_ON',
            metric_sign='MAE_OFF_minus_MAE_ON_positive_means_D_helpful_at_this_checkpoint',
            weighting='original_rows; own_and_global_horizon_counts; equal_1_over_12_macro',
            unavailable_macro_rule='null_if_any_joint_horizon_has_no_valid_targets; never_nanmean',
            analysis_axes_are_evaluation_only_not_gate_inputs=True,
            metrics=metrics, bias_metrics=bias_metrics, joint_metrics=joint,
            partition_checks=dict(nine_cells=True, each_horizon=True, horizon_macro=True,
                                  region_marginals=True, off_bias_marginals=True),
            new_inference_calls=0, optimizer_updates=0, test_accessed=False,
            predictive_gain_claim=False, accident_only_attribution=False, architecture_selected=False,
            source_artifacts_preserved=True, original_artifact_hash_manifest_available=False,
            source_sha256_before=before, analysis_source_sha256=base.sha256(Path(__file__)),
            reused_analysis_source_sha256=base.sha256(Path(base.__file__)),
            environment=dict(python=platform.python_version(), numpy=np.__version__))
        base.write_csv(output/'joint_decomposition.csv', rows)
        (output/'README.md').write_text(explain(result), encoding='utf-8')
        after_paths = sorted(path for path in probe.iterdir() if path.is_file())
        after = base.snapshot(after_paths)
        base.require(before == after, 'Source artifacts changed during analysis; completion blocked')
        result['source_sha256_after'] = after
        base.write_json(output/'preservation.json', dict(before=before, after=after, source_artifacts_preserved=True))
        result['outputs_sha256'] = {p.name: base.sha256(p) for p in sorted(output.iterdir())}
        base.write_json(output/'analysis.json', result)
        return result
    except BaseException as error:
        base.write_json(output/'failure.json', dict(status='B3R_JOINT_GAIN_ANALYSIS_FAILED', completion_claim=False,
            error_type=type(error).__name__, message=str(error), new_inference_calls=0,
            optimizer_updates=0, test_accessed=False, source_sha256_before=before))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--allow-engineering', action='store_true')
    args = parser.parse_args(argv)
    probe = args.probe_dir.resolve()
    output = args.output_dir or probe.parent/('b3r_joint_gain_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6])
    print('Source probe: '+str(probe), flush=True)
    print('Analysis directory: '+str(output.resolve()), flush=True)
    result = analyze(probe, output, args.allow_engineering)
    print(json.dumps(dict(status=result['status'], source_best_epoch=result['source_best_epoch'],
                          source_scope=result['source_scope'], csv_rows=result['csv_rows'],
                          joint_macro_contributions={region: {bias: metric['global_gain_contribution']
                              for bias, metric in row.items()} for region, row in result['joint_metrics'].items()},
                          all_nodes_net_gain=result['metrics']['all_nodes']['off_minus_on_mae'],
                          new_inference_calls=0, source_artifacts_preserved=True,
                          analysis=str(output.resolve()/'analysis.json')), ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
