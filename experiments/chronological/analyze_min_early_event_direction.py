"""Audit correction directions in saved early-event predictions with NumPy only.

Reuses the completed evaluation archive and its read-only certificate checks.
No raw data, checkpoint, model, accelerator, training or inference is used.
"""
import argparse
from contextlib import nullcontext
from datetime import datetime
import json
from pathlib import Path
import sys
import uuid

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.chronological import analyze_min_early_event_gain as archive
from experiments.chronological.min_early_event_direction_metrics import (
    DIRECTIONS, FLOW_EDGES, analyze_direction, direction_rows, group_rows,
)

HERE = Path(__file__).resolve().parent
SCHEMA = 'min_early_event_saved_OFF_direction_analysis_v1'
STATUS = 'MIN_EARLY_EVENT_OFF_DIRECTION_ANALYSIS_PASS'


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--run-dir', type=Path, help='Use latest completed early_event_evaluation_* archive')
    source.add_argument('--evaluation-dir', type=Path, help='Use this completed archive, including an archived copy')
    parser.add_argument('--output-dir', type=Path, help='Fresh directory outside the source archive')
    parser.add_argument('--flow-edges', nargs='+', type=float, default=FLOW_EDGES)
    parser.add_argument('--response-tol', type=float, default=.001)
    args = parser.parse_args(argv)
    for name in ('run_dir', 'evaluation_dir', 'output_dir'):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    edges = np.asarray(args.flow_edges)
    archive.require(len(edges) >= 2 and np.isfinite(edges).all() and edges[0] == 0. and (np.diff(edges) > 0.).all(),
                    'Flow edges must increase from zero')
    archive.require(np.isfinite(args.response_tol) and args.response_tol >= 0., 'Response tolerance must be finite nonnegative')
    return args


def write_tables(output, result):
    rows = group_rows(result)
    archive.write_csv(output/'groups.csv', [row for row in rows if row['horizon'] == 0])
    archive.write_csv(output/'horizons.csv', [row for row in rows if row['group'] == 'all_nodes' and row['horizon'] > 0])
    for filename, families in [('OFF_bias.csv', {'OFF_bias'}), ('STATE_OFF_bias.csv', {'STATE_OFF_bias'}),
        ('flow.csv', {'flow', 'flow_OFF_bias'}), ('regions.csv', {'region', 'partition_OFF_bias'}),
        ('roads.csv', {'road'}), ('directions.csv', {'direction', 'prediction_delta', 'P_sign'}),
        ('fixed_hard.csv', {'fixed_STATE_hard_windows'})]:
        archive.write_csv(output/filename, [row for row in rows if row['family'] in families])
    archive.write_csv(output/'group_directions.csv', direction_rows(result))


def write_report(output, result, info):
    def fmt(value):
        return '不可用' if value is None else f'{value:.9f}'
    overall = result['groups']['all_nodes']
    lines = ['# EVENT_OFF 基准：事故路径修正方向审计', '',
        '本报告只读已保存的五种配对预测；没有加载原始数据或检查点，没有重新推理、训练或评价测试集。', '',
        '工程检查结果，不能解释为正式预测效果。' if info['summary']['engineering_only'] else
        '范围：正式验证集上的探索性检查点诊断，窗口和预测点不是独立事故重复。', '',
        '**本轮高估／低估由 EVENT_OFF − 真值定义。**STATE 仅保留在 STATE × OFF 偏差交叉表，避免把训练组间差异当作当前修正基准。', '',
        'P = |EVENT_OFF − 真值| − |EVENT_ON − 真值|；正数表示开启事故路径误差更低。预测改变量为 ON − OFF，正数表示上调。', '',
        '| 指标 | EVENT_OFF | EVENT_ON | OFF−ON收益 |', '|---|---:|---:|---:|']
    for label, key, gain in [('MAE pooled', 'mae_pooled', 'mae_gain_pooled'), ('MAE horizon macro', 'mae_macro', 'mae_gain_macro'),
                            ('RMSE pooled', 'rmse_pooled', 'rmse_gain_pooled'), ('MAPE pooled (%)', 'mape_pooled', 'mape_gain_pooled')]:
        factor = 100. if 'MAPE' in label else 1.
        scale = lambda value: None if value is None else factor*value
        lines.append('| '+label+' | '+fmt(scale(overall['control'][key]))+' | '+fmt(scale(overall['candidate'][key]))+' | '+fmt(scale(overall[gain]))+' |')
    lines += ['', '## 修正方向与收益', '',
        '“朝向真值”描述初始改动方向，包括越过真值的情形；是否改善仍须比较改动后的绝对误差。', '',
        '| 方向 | 有效点数 | 全网点数占比 | 组内P pooled | P全网贡献 pooled |', '|---|---:|---:|---:|---:|']
    for name in DIRECTIONS:
        row = overall['directions'][name]
        lines.append('| '+name+' | '+str(row['valid_count'])+' | '+fmt(row['global_point_fraction'])+' | '
                     +fmt(row['P_gain_pooled'])+' | '+fmt(row['global_P_contribution_pooled'])+' |')
    lines += ['', '## EVENT_OFF 高估／低估', '',
        '| OFF偏差 | 点数 | P pooled | P macro | P全网贡献 pooled | 上调比例 | 下调比例 |', '|---|---:|---:|---:|---:|---:|---:|']
    for bias in ('over', 'under', 'exact'):
        row = result['groups']['off_bias:'+bias]
        lines.append('| '+bias+' | '+str(row['valid_count'])+' | '+' | '.join(fmt(row[key]) for key in
            ('mae_gain_pooled', 'mae_gain_macro', 'global_mae_contribution_pooled'))+' | '
            +fmt(row['prediction_change']['up_fraction'])+' | '+fmt(row['prediction_change']['down_fraction'])+' |')
    lines += ['', '## STATE × OFF 偏差交叉', '',
        '| STATE偏差 | OFF偏差 | 点数 | P pooled | P全网贡献 pooled |', '|---|---|---:|---:|---:|']
    for name, definition in result['group_definitions'].items():
        if definition['family'] != 'STATE_OFF_bias':
            continue
        row = result['groups'][name]
        lines.append('| '+definition['STATE_bias']+' | '+definition['OFF_bias']+' | '+str(row['valid_count'])+' | '
                     +fmt(row['mae_gain_pooled'])+' | '+fmt(row['global_mae_contribution_pooled'])+' |')
    lines += ['', '## 流量、事故关联区域和道路', '',
        '| 分组 | 点数 | P pooled | P全网贡献 pooled |', '|---|---:|---:|---:|']
    for name, definition in result['group_definitions'].items():
        if definition['family'] not in ('flow', 'road') and name not in ('direct_report_nodes', 'potential_only', 'remaining_nodes'):
            continue
        row = result['groups'][name]
        lines.append('| '+name+' | '+str(row['valid_count'])+' | '+fmt(row['mae_gain_pooled'])+' | '
                     +fmt(row['global_mae_contribution_pooled'])+' |')
    lines += ['', '## 逐预测时距', '', '| 步 | 有效点数 | P收益 | 正收益 | 负收益 | 上调比例 | 下调比例 |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for h in range(12):
        lines.append('| '+str(h+1)+' | '+str(overall['valid_count_per_horizon'][h])+' | '
            +' | '.join(fmt(overall[key][h]) for key in ('per_horizon_mae_gain', 'per_horizon_gain_positive', 'per_horizon_gain_negative'))
            +' | '+fmt(overall['prediction_change']['per_horizon_up_fraction'][h])+' | '+fmt(overall['prediction_change']['per_horizon_down_fraction'][h])+' |')
    lines += ['', '## 手算含义', '',
        '真值为 10，OFF 为 12：ON 为 11 属于朝向且未越过，P=1；ON 为 9 属于越过但改善，P=1；ON 为 8 属于越过且误差相同，P=0；ON 为 7 属于越过后恶化，P=−1。',
        '同样的真值与 OFF 下，ON 为 13 属于远离真值，P=−1。若 OFF 已为 10，任何非零改动都会增大绝对误差；不变单独统计。', '',
        '## 一致性与解释边界', '',
        '八种修正方向、三种预测变化方向、三种P符号、OFF偏差、STATE×OFF偏差、流量×OFF偏差、事故分区×OFF偏差各自互斥完备；各family之间不能相加。',
        '同一family的点数与共同分母P贡献逐时距核对守恒。道路和候选区域可能重叠，不能当作互斥分区相加。',
        '组内 macro 让各时距等权，global macro 贡献先除各时距全网点数；两者可能异号。报告主分解使用 pooled，JSON/CSV同时保存macro。',
        'response_tol 仅影响预测变化响应比例，不改变方向、P正负或高估/低估定义。零目标沿用原有效掩码，未重新纳入；MAPE在JSON/CSV中为比例。空组局部均值为不可用，共同分母贡献为零。',
        '固定困难窗口复用原 STATE 保存的样本ID；本轮没有重新挑选。真值与误差符号只用于事后分析，不作线上门控。',
        '本轮核对已保存数组和历史回放证书，不是新增模型回放或历史文件独立签名认证。事故关联区域不等于真实事故影响，其余区域不等于自然无事故。',
        '制品：analysis.json、summary.json、OFF_bias.csv、STATE_OFF_bias.csv、flow.csv、regions.csv、roads.csv、directions.csv、horizons.csv、group_directions.csv、fixed_hard.csv、provenance.json、preservation.json。', '']
    (output/'report.md').write_text('\n'.join(lines), encoding='utf-8')


def analyze_run(args):
    source = archive.select_source(args)
    if args.output_dir is None:
        root = args.run_dir if args.run_dir is not None else source.parent
        args.output_dir = root/('early_event_direction_analysis_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6])
    output = archive.output_path(args, source)
    lock_root = args.run_dir if args.run_dir is not None else source.parent if (source.parent/'active.lock').is_file() else None
    with archive.readonly_run_lock(lock_root) if lock_root is not None else nullcontext(None) as lock_state:
        def source_paths():
            files = {str(path.relative_to(source)): path for path in source.rglob('*') if path.is_file()}
            archive.require(all(source == path.resolve().parent or source in path.resolve().parents for path in files.values()),
                            'Source archive contains an external file link')
            return files
        def protected_snapshot():
            values = archive.snapshot(source_paths())
            if lock_state is not None:
                values['<existing_source_run_lock>'] = archive.locked_stream_hash(lock_state[1])
            return values
        before = protected_snapshot()
        output.mkdir(parents=True)
        print('Source evaluation: '+str(source), flush=True)
        print('Direction analysis directory: '+str(output), flush=True)
        try:
            info = archive.load_and_verify(source)
            result = analyze_direction(info['predictions'], info['target'], info['valid'], info['regions'],
                info['indices'], info['stations'], flow_edges=tuple(args.flow_edges), response_tol=args.response_tol,
                fixed_hard_sample_ids=info['fixed_hard_ids'])
            archive.assert_close(result['groups']['all_nodes']['mae_gain_pooled'],
                info['analysis']['comparisons']['EVENT_dependency']['metrics']['all_nodes']['mae_gain_pooled'], 'original_EVENT_dependency')
            result.update(source_evaluation=str(source), source_scope=info['comparable']['whole_model_scope'],
                engineering_only=info['summary']['engineering_only'], source_best_epochs=info['summary']['best_epochs'],
                training_comparability=info['comparable'], archive_axis_scope=info['axis_scope'],
                new_inference_calls=0, optimizer_updates=0, test_evaluated=False)
            archive.write_json(output/'analysis.json', result)
            write_tables(output, result)
            write_report(output, result, info)
            source_files = (Path(__file__).resolve(), HERE/'min_early_event_direction_metrics.py',
                HERE/'analyze_min_early_event_gain.py', HERE/'min_early_event_gain_metrics.py', HERE/'min_event_budget_probe_metrics.py')
            archive.write_json(output/'provenance.json', dict(schema=SCHEMA, source_evaluation=str(source),
                source_files_sha256_before=before, analysis_source_sha256={str(path.relative_to(REPO)): archive.sha256(path) for path in source_files},
                flow_edges=list(args.flow_edges), response_tol=args.response_tol, original_fixed_hard_sample_ids=info['fixed_hard_ids'].tolist(),
                imported_runtime='numpy_and_standard_library_only', raw_data_read=False, checkpoint_loaded=False,
                new_checkpoint_replay=False, historical_signature_verified=False, new_inference_calls=0, optimizer_updates=0, test_evaluated=False))
            after = protected_snapshot()
            archive.require(before == after, 'Source artifacts changed during read-only direction analysis')
            archive.write_json(output/'preservation.json', dict(status='SOURCE_ARTIFACTS_PRESERVED', source_artifacts_preserved=True,
                files=len(before), before=before, after=after, scope='this_analysis_read_interval_byte_hashes_not_original_archive_signature'))
            summary = dict(schema=SCHEMA, status=STATUS, source_evaluation=str(source), validation_rows=len(info['indices']),
                source_best_epochs=info['summary']['best_epochs'], engineering_only=info['summary']['engineering_only'],
                all_nodes=result['groups']['all_nodes'], OFF_bias={bias: result['groups']['off_bias:'+bias] for bias in ('over', 'under', 'exact')},
                new_inference_calls=0, optimizer_updates=0, test_evaluated=False, raw_data_read=False, checkpoint_loaded=False,
                source_artifacts_preserved=True, source_certificate_check='archived_internal_consistency_not_new_checkpoint_replay', output_dir=str(output))
            archive.write_json(output/'summary.json', summary)
            return summary
        except BaseException as error:
            after = protected_snapshot()
            archive.write_json(output/'preservation.json', dict(status='SOURCE_ARTIFACTS_PRESERVED' if before == after else 'SOURCE_ARTIFACTS_CHANGED',
                source_artifacts_preserved=before == after, files=len(before), before=before, after=after, analysis_completed=False))
            archive.write_json(output/'failure.json', dict(schema=SCHEMA, status='MIN_EARLY_EVENT_OFF_DIRECTION_ANALYSIS_FAILED',
                error_type=type(error).__name__, error=str(error), source_evaluation=str(source),
                new_inference_calls=0, optimizer_updates=0, test_evaluated=False))
            raise


def main(argv=None):
    result = analyze_run(parse_args(argv))
    concise = {key: value for key, value in result.items() if key not in ('all_nodes', 'OFF_bias')}
    concise['all_nodes'] = {key: result['all_nodes'][key] for key in ('mae_gain_pooled', 'mae_gain_macro', 'gain_positive_pooled', 'gain_negative_pooled')}
    print(json.dumps(concise, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
    return result


if __name__ == '__main__':
    main()
