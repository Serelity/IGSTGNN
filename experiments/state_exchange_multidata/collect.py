"""Collect three completed paired tests; never construct a model or data loader."""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from experiments.final_test_gw16.state import cache_read, payload_hash, verify_files
from experiments.final_test_gw16_recovery.recover import repair_cached_arrays
from experiments.original_baseline_gw16.evaluate import native_summary
from experiments.position_state_gw16.objective import numpy_metrics
from .common import exclusive_run, read_json, require, sha256, spec, write_json

CASES = ('ORIGINAL_IGSTGNN', 'STATE_EXCHANGE')


def load_contra(directory):
    """Reuse the exact already-reported Contra result, including its layout fix."""
    directory = Path(directory).resolve()
    files = [directory/'comparison.json', directory/'freeze.json']
    files += [directory/case/name for case in CASES
              for name in ('metadata.json', 'predictions.npz')]
    before = {str(path):sha256(path) for path in files}
    frozen, result = read_json(directory/'freeze.json'), read_json(directory/'comparison.json')
    require(frozen.get('schema') == 'final_test_gw16_freeze_v1'
            and frozen.get('freeze_id') == payload_hash(frozen['payload'])
            and result.get('freeze_id') == frozen['freeze_id'], 'Contra test freeze differs')
    require(result.get('schema') == 'final_test_gw16_paired_test_v1'
            and result.get('status') == 'GW16_FINAL_TEST_COMPLETE'
            and result.get('split') == 'test' and result.get('test_evaluated') is True
            and result.get('samples') == 839
            and result.get('candidate_selection_used_test') is False,
            'Contra must be the completed existing paired test')
    require(result['candidate_checkpoint_role'] == frozen['payload']['candidate_checkpoint_role']
            and result['original_source'] == frozen['payload']['original_source'],
            'Contra reported model selection differs from its existing freeze')
    verify_files(frozen['payload']['source_files'])
    previous = torch.get_num_threads()
    cached = {}
    audits = {}
    try:
        torch.set_num_threads(frozen['payload']['runtime']['threads'])
        for case in CASES:
            arrays, metadata = cache_read(directory/case, frozen['freeze_id'], case)
            require(metadata['runtime'] == frozen['payload']['runtime']
                    and metadata['metric_contract'] == result['metric_contract'],
                    'Contra cached metric/runtime contract differs')
            audit = metadata['audit']
            weight_key = 'original_weights_sha256' if case == CASES[0] else 'candidate_weights_sha256'
            require(audit['model_tensor_sha256'] == frozen['payload'][weight_key]
                    and audit['optimizer_updates'] == 0 and audit['future_target_passed_to_forward'] is False
                    and all(audit[key] is True for key in
                            ('rng_unchanged', 'registered_tensors_unchanged', 'raw_inputs_unchanged')),
                    'Contra cache is not the frozen model inference')
            material = metadata['data_identity']
            require(material['hashes'] == spec('Contra_Costa')['hashes']
                    and material['bounds_half_open'] == [4746, 5585]
                    and material['count'] == 839 and material['batch_size'] == 48
                    and material['station_ids'] == frozen['payload']['test_protocol']['station_ids'],
                    'Contra cached data axes differ from the existing frozen release')
            arrays, audit = repair_cached_arrays(arrays, metadata, case, native_summary)
            common = numpy_metrics(arrays['prediction'], arrays['target'],
                                   result['metric_contract']['common_mask'])
            require(common == metadata['common_mask_pooled']
                    and result['rows'][case]['author_metrics'] == metadata['author_metrics']
                    and result['rows'][case]['common_mask_pooled'] == common,
                    'Contra frozen comparison differs from its immutable caches')
            cached[case], audits[case] = arrays, audit
        for key in ('target', 'station_ids', 'sample_ids'):
            require(cached[CASES[0]][key].dtype == cached[CASES[1]][key].dtype
                    and np.array_equal(cached[CASES[0]][key], cached[CASES[1]][key]),
                    'Contra paired axes differ: '+key)
    finally:
        torch.set_num_threads(previous)
    require(all(sha256(path) == digest for path, digest in before.items()),
            'Contra source cache changed during collection')
    parameters = dict(ORIGINAL_IGSTGNN=455220,
        STATE_EXCHANGE=result['candidate_checkpoint_role']['registered_parameters'])
    return dict(dataset='Contra_Costa', split='test', samples=839, rows=result['rows'],
        parameter_counts=parameters, candidate_checkpoint_role=result['candidate_checkpoint_role'],
        original_source=result['original_source'], existing_test_result_reused=True,
        new_model_forward_calls=0, new_training_updates=0,
        source_files=before, layout_replay_audits=audits)


def comparison_rows(results):
    rows = []
    for dataset in ('Alameda', 'Contra_Costa', 'Orange'):
        result = results[dataset]
        require(result['dataset'] == dataset and result['split'] == 'test',
                'Cannot mix datasets or validation and test rows')
        base = result['rows'][CASES[0]]['author_metrics']['horizon_macro']
        actual = result['rows'][CASES[1]]['author_metrics']['horizon_macro']
        require(all(np.isfinite(base[key]) and base[key] > 0 for key in
                    ('mae', 'rmse', 'mape_percent')), 'Invalid original denominator')
        for case in CASES:
            metric = result['rows'][case]['author_metrics']['horizon_macro']
            require(all(np.isfinite(metric[key]) for key in ('mae', 'rmse', 'mape_percent')),
                    'Nonfinite metric in completed result')
            rows.append(dict(dataset=dataset, case=case, samples=result['samples'],
                parameters=result['parameter_counts'][case], **metric,
                mae_gain_percent_vs_original=100*(base['mae']-actual['mae'])/base['mae']
                    if case == CASES[1] else 0.))
    return rows


def render(directory, results):
    rows = comparison_rows(results)
    lines = ['# GW16 STATE_EXCHANGE：三个作者发布数据集', '',
        '主表为各数据集保留测试的12步等权平均；使用同样本回放原版作分母。',
        'Contra Costa复用已冻结并报告的测试缓存，没有新训练或模型推理。',
        'Alameda和Orange各自从头训练；GW16沿用固定100轮配方，原版保留原生课程学习、调度和patience20。',
        '三个数据集没有合并训练；词表按本地发布ID及类别数构建，没有共享类别编号。', '',
        '| 数据集 | 方法 | 参数量 | MAE | RMSE | MAPE % | MAE改善 % |',
        '|---|---|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append('| {dataset} | {case} | {parameters} | {mae:.6f} | {rmse:.6f} | '
                     '{mape_percent:.6f} | {mae_gain_percent_vs_original:.4f} |'.format(**row))
    lines += ['', 'RMSE逐步计算再取平均，不是pooled RMSE；common_mask_pooled仅为补充口径。',
        '各数据集均为单seed整套方法比较，输入与训练配方有差异，尚不独立识别状态交换机制收益。',
        '发布窗口重叠，绝对时间/事故独立性和报告首次可用时刻未获独立认证。',
        '没有多seed显著性结论；不按这些测试结果换模型、校准、延长训练或挑选数据集。', '']
    (directory/'report.md').write_text('\n'.join(lines), encoding='utf-8')
    with (directory/'metrics.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (directory/'per_horizon.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['dataset', 'case', 'lead', 'mae', 'rmse', 'mape_percent'])
        writer.writeheader()
        for dataset in ('Alameda', 'Contra_Costa', 'Orange'):
            for case in CASES:
                for row in results[dataset]['rows'][case]['author_metrics']['per_horizon']:
                    writer.writerow(dict(dataset=dataset, case=case, **row))
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('alameda-result', 'orange-result', 'contra-test', 'output-dir'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args(argv)
    sources = [args.alameda_result.resolve(), args.orange_result.resolve(), args.contra_test.resolve()]
    output = args.output_dir.resolve()
    require(all(output != source and source not in output.parents and output not in source.parents
                for source in sources), 'Collection output intersects a source result')
    from .evaluate import read_result
    results = {}
    for dataset, directory in zip(('Alameda', 'Orange'), sources[:2]):
        result = read_result(directory)
        require(result['dataset'] == dataset and result['split'] == 'test'
                and result['engineering_only'] is False, 'A new dataset test is incomplete')
        results[dataset] = result
    results['Contra_Costa'] = load_contra(sources[2])
    output.mkdir(parents=True, exist_ok=True)
    with exclusive_run(output):
        record = dict(schema='gw16_three_dataset_test_collection_v1',
            datasets=results, primary_metric='native_12_lead_equal_weight_macro',
            single_seed=2025, statistical_confirmation=False,
            source_collection_new_model_forward_calls=0, source_collection_training_updates=0)
        if (output/'comparison.json').is_file():
            require(read_json(output/'comparison.json') == record,
                    'Collection sources changed; preserve the existing report')
        rows = render(output, results)
        write_json(output/'comparison.json', record)
    print('Read '+str(output/'report.md'), flush=True)
    return rows


if __name__ == '__main__':
    main()
