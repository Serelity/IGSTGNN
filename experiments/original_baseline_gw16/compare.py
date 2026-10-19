"""Pair a frozen original IGSTGNN validation archive with one GW16 archive.

No model inference, training, test loading, screening override or new selection.
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from experiments.chronological.original_igstgnn_runtime import sha256, write_json
from experiments.position_state_gw16.evaluate import read_archive as read_gw16_archive
from experiments.position_state_gw16.objective import numpy_metrics
from .evaluate import native_summary, read_archive as read_original_archive


def require(condition, message):
    if not condition:
        raise ValueError(message)


def paired_scores(original, candidate, original_metadata, candidate_metadata):
    """Refuse matching by shape alone, then score both arrays under each mask."""
    for key in ('sample_ids', 'station_ids', 'target'):
        require(original[key].dtype == candidate[key].dtype
                and np.array_equal(original[key], candidate[key]),
                'Original/GW16 validation axes or targets differ: '+key)
    require(original_metadata['data_identity']['hashes']
            == candidate_metadata['identity']['data']['hashes'],
            'Original/GW16 published data fingerprints differ')
    require(original_metadata['engineering_only'] == candidate_metadata['engineering_only'],
            'Cannot pair engineering and full validation archives')
    config = candidate_metadata['identity']['config']
    target = original['target']
    rows = {}
    threads = original_metadata['runtime']['threads']
    require(type(threads) is int and threads > 0, 'Original metric thread count is invalid')
    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(threads)
        for name, arrays in (('ORIGINAL_IGSTGNN', original),
                             (candidate_metadata['arm'], candidate)):
            rows[name] = dict(
                author_metrics=native_summary(arrays['prediction'], target),
                gw16_common_mask_pooled=numpy_metrics(arrays['prediction'], target, config))
    finally:
        torch.set_num_threads(previous)
    original_row, candidate_row = rows['ORIGINAL_IGSTGNN'], rows[candidate_metadata['arm']]
    base = original_row['author_metrics']['horizon_macro']['mae']
    actual = candidate_row['author_metrics']['horizon_macro']['mae']
    require(base > 0, 'Original MAE must be positive to report relative gain')
    original_complete = bool(original_metadata['source']['training_complete'])
    candidate_complete = bool(candidate_metadata['checkpoint_role']['full_protocol_reached'])
    return dict(rows=rows, candidate_arm=candidate_metadata['arm'], split='val',
        relative_mae_gain_vs_original_horizon_macro_percent=100*(base-actual)/base,
        original_source=original_metadata['source'],
        candidate_checkpoint_role=candidate_metadata['checkpoint_role'],
        original_training_complete=original_complete,
        gw16_training_complete=candidate_complete,
        full_training_validation_comparison=original_complete and candidate_complete
            and not original_metadata['engineering_only'],
        original_and_gw16_training_recipes_identical=False,
        native_pos_is_original=False, test_evaluated=False,
        metric_cpu_threads=threads,
        paper_test_mae_directly_comparable_to_this_validation=False,
        interpretation='paired_development_validation_not_independent_confirmation',
        screening_overridden=False, candidate_for_confirmation=False)


def safe_output(output, sources, data_dir=None):
    output = Path(output).resolve()
    if data_dir is not None:
        dataset = Path(data_dir).resolve()
        require(output != dataset and dataset not in output.parents
                and output not in dataset.parents, 'Comparison output must be outside the dataset')
    for source in sources:
        source = Path(source).resolve()
        require(output != source and source not in output.parents
                and output not in source.parents,
                'Comparison output must be separate from source archives')
    for parent in (output, *output.parents):
        require(not (parent/'identity.json').is_file(),
                'Comparison output must be outside training run directories')
    require(not output.exists(), 'Use a fresh comparison output directory')
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-eval', type=Path, required=True)
    parser.add_argument('--gw16-eval', type=Path, required=True,
                        help='One arm directory containing predictions.npz and metadata.json')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    output = safe_output(args.output_dir, (args.original_eval, args.gw16_eval))
    source_files = [root/name for root in (args.original_eval, args.gw16_eval)
                    for name in ('predictions.npz', 'metadata.json')]
    before = {str(path.resolve()): sha256(path) for path in source_files}
    original, original_metadata = read_original_archive(args.original_eval)
    candidate, candidate_metadata = read_gw16_archive(args.gw16_eval)
    safe_output(output, (args.original_eval, args.gw16_eval), original_metadata['data_directory'])
    result = paired_scores(original, candidate, original_metadata, candidate_metadata)
    require(all(sha256(path) == expected for path, expected in before.items()),
            'Source archive changed during comparison')
    result.update(schema='original_baseline_gw16_paired_validation_v1',
                  source_files=before, comparison_script_sha256=sha256(Path(__file__)))
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'comparison.json', result)
    lines = ['# 原版 IGSTGNN 与 GW16：同一验证点比较', '',
        'ORIGINAL_IGSTGNN 使用冻结原版源码；NATIVE_POS 只属于位置增强补充对照。',
        '以下全部是验证集结果。原文测试 MAE 13.43 不参与本表提升率计算。',
        '两种模型保留各自训练配方，本表不将差异全部归因于新增结构。', '',
        '原版训练完整：{}；GW16训练完整：{}。部分预算与工程结果只供观察。'.format(
            result['original_training_complete'], result['gw16_training_complete']), '',
        '| 方法 | 原生逐步平均 MAE | 原生逐步平均 RMSE | 原生逐步平均 MAPE % | GW16共同掩码 pooled MAE |',
        '|---|---:|---:|---:|---:|']
    for name, row in result['rows'].items():
        macro = row['author_metrics']['horizon_macro']
        lines.append('| {} | {:.9f} | {:.9f} | {:.9f} | {:.9f} |'.format(
            name, macro['mae'], macro['rmse'], macro['mape_percent'],
            row['gw16_common_mask_pooled']['mae']))
    lines += ['', '候选相对原版的验证逐步平均 MAE 改善：{:.6f}%。'.format(
        result['relative_mae_gain_vs_original_horizon_macro_percent']), '',
        '原版训练是否完整、历史来源边界及候选完成轮数见 comparison.json。',
        '本报告不改变既有 GW16 筛查、选模、训练预算或测试集边界。']
    (output/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('Read '+str(output/'report.md'), flush=True)
    return result


if __name__ == '__main__':
    main()
