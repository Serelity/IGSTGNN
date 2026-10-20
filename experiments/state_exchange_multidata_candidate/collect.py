"""Collect candidate-only three-dataset results without constructing any model."""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from experiments.final_test_gw16.state import cache_read, payload_hash, verify_files
from experiments.original_baseline_gw16.evaluate import native_summary
from experiments.position_state_gw16.objective import numpy_metrics
from experiments.state_exchange_multidata.common import (
    exclusive_run, read_json, require, sha256, spec, write_json)

CASE = 'STATE_EXCHANGE'
DATASETS = ('Alameda', 'Contra_Costa', 'Orange')
METRICS = ('mae', 'rmse', 'mape_percent')
CONTRA_CONTRACT = dict(
    primary='unchanged_author_native_metrics_each_lead_then_equal_average_over_12',
    native_null='min_of_entire_test_target_if_less_than_1_else_0; same_for_both_models_and_all_leads',
    native_accumulation='frozen_author_pytorch_float32',
    rmse_macro='mean_of_12_per_lead_RMSE_not_sqrt_of_pooled_MSE',
    mape_units='percent', horizons_reported=[3, 6, 12],
    common_mask=dict(target_minimum=.001, mape_minimum=1., accumulation='numpy_float64_pooled'),
    prediction_clipping=False, fitted_calibration=False, sample_reweighting=False)


def load_contra(directory):
    """Read only the completed candidate cache from the already-reported test.

    The historical freeze is an immutable provenance certificate. Its source-file
    map is authenticated by its payload digest; reopening original checkpoints,
    validation caches or test predictions is unnecessary for this collection.
    """
    directory = Path(directory).resolve()
    files = [directory/'comparison.json', directory/'freeze.json',
             directory/CASE/'metadata.json', directory/CASE/'predictions.npz']
    require(all(path.is_file() for path in files), 'Contra candidate test cache is incomplete')
    before = {str(path):sha256(path) for path in files}
    frozen = read_json(directory/'freeze.json')
    result = read_json(directory/'comparison.json')
    require(frozen.get('schema') == 'final_test_gw16_freeze_v1'
            and frozen.get('freeze_id') == payload_hash(frozen['payload'])
            and result.get('freeze_id') == frozen['freeze_id'], 'Contra test freeze differs')
    payload = frozen['payload']
    require(result.get('schema') == 'final_test_gw16_paired_test_v1'
            and result.get('status') == 'GW16_FINAL_TEST_COMPLETE'
            and result.get('split') == 'test' and result.get('test_evaluated') is True
            and result.get('samples') == 839
            and result.get('candidate_selection_used_test') is False,
            'Contra must be the completed existing test')
    role = payload['candidate_checkpoint_role']
    require(result['candidate_checkpoint_role'] == role and role['selection'] == 'best'
            and role['completed_training_epochs'] == 100 and role['full_protocol_reached'] is True
            and type(role['registered_parameters']) is int and role['registered_parameters'] > 0,
            'Contra candidate selection differs from the full frozen validation BEST')
    protocol = payload['test_protocol']
    require(protocol['bounds_half_open'] == [4746, 5585] and protocol['count'] == 839
            and protocol['nodes'] == 496 and protocol['history'] == protocol['horizon'] == 12
            and protocol['batch_size'] == 48
            and protocol['split'] == 'author_order_70_15_15', 'Contra frozen test protocol differs')
    require(result['metric_contract'] == payload['metric_contract'] == CONTRA_CONTRACT,
            'Contra native metric contract differs')
    arrays, metadata = cache_read(directory/CASE, frozen['freeze_id'], CASE)
    require(metadata['runtime'] == payload['runtime']
            and metadata['metric_contract'] == CONTRA_CONTRACT,
            'Contra cached metric/runtime contract differs')
    audit = metadata['audit']
    require(audit['model_tensor_sha256'] == payload['candidate_weights_sha256']
            and audit['optimizer_updates'] == 0 and audit['forward_batches'] == 18
            and audit['future_target_passed_to_forward'] is False
            and all(audit[key] is True for key in
                    ('rng_unchanged', 'registered_tensors_unchanged', 'raw_inputs_unchanged')),
            'Contra cache is not the frozen candidate inference')
    material = metadata['data_identity']
    require(material['hashes'] == spec('Contra_Costa')['hashes']
            and material['bounds_half_open'] == [4746, 5585]
            and material['count'] == 839 and material['batch_size'] == 48
            and material['station_ids'] == protocol['station_ids']
            and np.array_equal(arrays['station_ids'], np.asarray(protocol['station_ids'], dtype=np.int64)),
            'Contra cached data axes differ from the frozen release')
    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(payload['runtime']['threads'])
        native = native_summary(arrays['prediction'], arrays['target'])
        common = numpy_metrics(arrays['prediction'], arrays['target'], CONTRA_CONTRACT['common_mask'])
    finally:
        torch.set_num_threads(previous)
    require(native == metadata['author_metrics'] == result['rows'][CASE]['author_metrics']
            and common == metadata['common_mask_pooled'] == result['rows'][CASE]['common_mask_pooled'],
            'Contra candidate metrics do not reproduce exactly; preserve the existing cache')
    verify_files(before)
    return dict(dataset='Contra_Costa', split='test', samples=839,
        rows={CASE:dict(author_metrics=native, common_mask_pooled=common)},
        parameter_counts={CASE:role['registered_parameters']}, checkpoint_roles={CASE:role},
        candidate_checkpoint_role=role, engineering_only=False, test_evaluated=True,
        existing_test_result_reused=True, new_model_forward_calls=0, new_training_updates=0,
        source_files=before, historical_source_certificate=payload_hash(payload['source_files']),
        historical_source_artifacts_reopened=False, original_prediction_cache_read=False)


def comparison_rows(results):
    require(set(results) == set(DATASETS), 'Collection requires all three datasets')
    rows = []
    for dataset in DATASETS:
        result = results[dataset]
        require(result['dataset'] == dataset and result['split'] == 'test'
                and result['test_evaluated'] is True and result['engineering_only'] is False,
                'Cannot mix datasets, engineering, validation and test rows')
        require(result['samples'] == spec(dataset)['test_samples']
                and set(result['rows']) == {CASE} and set(result['parameter_counts']) == {CASE},
                'Candidate-only result axis or cases differ')
        parameters = result['parameter_counts'][CASE]
        require(type(parameters) is int and parameters > 0, 'Invalid registered parameter count')
        native = result['rows'][CASE]['author_metrics']
        macro, leads = native['horizon_macro'], native['per_horizon']
        require(set(macro) == set(METRICS) and [row['lead'] for row in leads] == list(range(1, 13)),
                'Candidate metrics must preserve all twelve ordered horizons')
        require(all(np.isfinite(macro[key]) and macro[key] >= 0
                    and all(np.isfinite(row[key]) and row[key] >= 0 for row in leads)
                    and macro[key] == float(np.mean([row[key] for row in leads])) for key in METRICS),
                'Invalid native equal-horizon macro metrics')
        rows.append(dict(dataset=dataset, case=CASE, samples=result['samples'],
                         parameters=parameters, **macro))
    return rows


def render(directory, results):
    rows = comparison_rows(results)
    lines = ['# GW16 STATE_EXCHANGE：三个作者发布数据集', '',
        '这里只运行和报告改进模型；各数据集分别训练，没有合并训练或共享类别编号。',
        'Contra Costa复用已完成的STATE_EXCHANGE冻结测试缓存，没有新训练或模型推理。',
        'Alameda和Orange使用同一GW16结构及固定100轮配方，各自在验证集选择BEST后测试。',
        '主表逐时距计算作者指标，再对12个时距等权平均；RMSE不是pooled RMSE。', '',
        '| 数据集 | 方法 | 参数量 | 12步平均 MAE | 12步平均 RMSE | 12步平均 MAPE % |',
        '|---|---|---:|---:|---:|---:|']
    for row in rows:
        lines.append('| {dataset} | {case} | {parameters} | {mae:.6f} | {rmse:.6f} | '
                     '{mape_percent:.6f} |'.format(**row))
    lines += ['', '所有12个时距见per_horizon.csv；common_mask_pooled仅作为补充口径。',
        '本表没有新训练的原版对照，不计算相对原版的提升率；不同数据集的绝对误差不代表泛化难度排序。',
        '这是单seed整套方法的跨数据集评估，不能独立归因于某个机制，也没有多seed显著性结论。',
        '发布窗口存在重叠，绝对时间/事故独立性和报告首次可用时刻未获独立认证。',
        '不按测试结果换模型、校准、延长训练或挑选数据集。', '']
    (directory/'report.md').write_text('\n'.join(lines), encoding='utf-8')
    with (directory/'metrics.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (directory/'per_horizon.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['dataset', 'case', 'lead', *METRICS])
        writer.writeheader()
        for dataset in DATASETS:
            for row in results[dataset]['rows'][CASE]['author_metrics']['per_horizon']:
                writer.writerow(dict(dataset=dataset, case=CASE, **row))
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
                and result['engineering_only'] is False and result['test_evaluated'] is True,
                'A new dataset candidate test is incomplete')
        results[dataset] = result
    results['Contra_Costa'] = load_contra(sources[2])
    comparison_rows(results)
    output.mkdir(parents=True, exist_ok=True)
    with exclusive_run(output):
        record = dict(schema='state_exchange_candidate_three_dataset_collection_v1',
            datasets=results, primary_metric='native_12_lead_equal_weight_macro',
            single_seed=2025, statistical_confirmation=False, original_models_run=False,
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
