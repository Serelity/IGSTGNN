"""Certified GW17 validation archives; no test loader or test evaluation."""
import argparse
import json
import logging
from pathlib import Path
import time
import uuid

import numpy as np
import torch

from . import ARMS
from .common import (SCHEMA, exclusive_run, load_state, protected_output,
    read_json, require, runtime, sha256, source_hashes, synchronize, tensor_tree_hash, write_json)
from .data import device_batch, load_data, validation_loader
from .model import build_model
from .objective import archive_diagnostics, metric_contract
from .reference import read_reference, verify_reference
from .train import validate_identity, validate_runtime, validate_saved

ARCHIVE_SCHEMA = 'observed_state_gw17_prediction_archive_v1'
ARCHIVE_FIELDS = {'prediction', 'target', 'reference', 'regional', 'alpha', 'beta',
    'support', 'available', 'station_ids', 'sample_ids', 'assignment', 'state_rms_batch',
    'history_prediction', 'history_target', 'history_valid'}
FLOAT_FIELDS = {'prediction', 'target', 'reference', 'regional', 'alpha', 'beta',
    'state_rms_batch', 'history_prediction', 'history_target'}


def canonical_arrays(arrays):
    """Freeze layout before metrics and serialization, with no precision change."""
    return {key:np.array(value, copy=True, order='C') for key, value in arrays.items()}


def require_reference_axes(candidate, reference):
    """A historical forecast is comparable only on the exact same raw labels."""
    for key in ('target', 'sample_ids', 'station_ids', 'assignment', 'support', 'available'):
        require(candidate[key].dtype == reference[key].dtype
            and np.array_equal(candidate[key], reference[key]),
            'GW16 reference differs from candidate '+key)


def read_archive(directory):
    """Authenticate every load and independently recompute all reported metrics."""
    directory = Path(directory)
    metadata = read_json(directory/'metadata.json')
    require(metadata.get('schema') == ARCHIVE_SCHEMA, 'Not a GW17 prediction archive')
    identity = metadata['identity']
    validate_identity(identity)
    require(source_hashes() == identity['execution_source'], 'Archive execution source changed')
    verify_reference(identity['gw16_reference'])
    require(metadata.get('identity_sha256') == identity['identity_sha256']
        and metadata['metric_contract'] == metric_contract(identity['config'])
        and metadata.get('gw16_reference') == identity['gw16_reference'],
        'Archive identity or metric contract differs')
    require(metadata.get('engineering_only') == identity['engineering_only']
        and metadata.get('split') == 'val' and metadata.get('test_evaluated') is False,
        'Archive scope differs from the validation-only protocol')
    require(metadata.get('arm') in ARMS, 'Unknown archived arm')
    path = directory/'predictions.npz'
    before = sha256(path)
    require(before == metadata.get('archive_sha256'), 'Prediction archive checksum differs')
    with np.load(path, allow_pickle=False) as payload:
        arrays = dict(payload)
    require(sha256(path) == before, 'Prediction archive changed during read')
    require(set(arrays) == ARCHIVE_FIELDS, 'Archive fields differ')
    require(all(arrays[key].dtype == np.dtype('float32') for key in FLOAT_FIELDS),
        'Archive floating fields must retain model float32 values')
    require(all(value.flags.c_contiguous for value in arrays.values()),
        'Archive fields must use canonical C layout')
    p, y = arrays['prediction'], arrays['target']
    config = identity['config']
    require(p.shape == y.shape == arrays['reference'].shape and p.ndim == 4
        and p.shape[1:] == (config['horizon'], config['nodes'], 1),
        'Archive prediction/target shape differs')
    expected_samples = identity['data']['loader_batching']['val_loader']['samples']
    require(len(p) == expected_samples
        and np.array_equal(arrays['sample_ids'], np.arange(3909, 3909+len(p), dtype=np.int64))
        and np.array_equal(arrays['station_ids'], np.asarray(identity['data']['station_ids']))
        and np.array_equal(arrays['assignment'], np.asarray(identity['data']['partition']['assignment'])),
        'Archive sample, station or region axis differs from its identity')
    regions = int(arrays['assignment'].max())+1
    batches = identity['data']['loader_batching']['val_loader']['batches']
    require(arrays['support'].shape == (len(p), p.shape[2])
        and arrays['available'].shape == (len(p),) and arrays['beta'].shape == p.shape
        and arrays['alpha'].shape == arrays['regional'].shape == (len(p), p.shape[1], regions, 1)
        and arrays['state_rms_batch'].shape == (batches, p.shape[1], 2)
        and arrays['history_prediction'].shape == arrays['history_target'].shape
            == arrays['history_valid'].shape == (len(p), config['history']-1, config['nodes'], 1),
        'Archive auxiliary shapes differ')
    require(arrays['history_valid'].dtype == np.dtype('bool'), 'Historical update mask must remain boolean')
    require(all(np.isfinite(value).all() for value in arrays.values()),
        'Nonfinite values in prediction archive')
    require(np.isin(arrays['support'], [0, 1]).all() and np.isin(arrays['available'], [0, 1]).all(),
        'Support/availability must be binary')
    require(metadata['diagnostics'] == archive_diagnostics(arrays, config),
        'Archived diagnostics do not match its predictions')
    require(metadata['validation_replay']['passed'] is True, 'Archive lacks a passed validation replay')
    return arrays, metadata


def selected_checkpoint(run, identity, arm, selection, batches):
    directory = run/arm
    last_path = directory/'last.pt'
    last_digest = sha256(last_path)
    last = load_state(last_path)
    require(sha256(last_path) == last_digest, 'LAST changed during load')
    validate_saved(last, identity, arm, directory, batches)
    if selection == 'best':
        path = directory/last['best_checkpoint_file']
        require(sha256(path) == last['best_checkpoint_sha256'], 'Selected BEST changed')
        selected = load_state(path)
        require(selected['identity_sha256'] == identity['identity_sha256']
            and selected['arm'] == arm and selected['epoch'] == last['best_epoch']
            and selected['validation'] == last['best_metrics'], 'BEST role mismatch')
        epoch, expected = selected['epoch'], selected['validation']
        expected_digest = last['best_checkpoint_sha256']
    else:
        path, selected = last_path, last
        epoch, expected = last['completed_epoch'], last['history'][-1]['validation']
        expected_digest = last_digest
    require(sha256(path) == expected_digest, 'Selected checkpoint changed during load')
    require(selected['stage'] == 'joint', 'GW17 has a single joint training phase')
    require(last['registered_parameters'] == identity['initializations'][arm]['registered_parameters'],
        'Parameter count differs from initialization identity')
    role = dict(selection=selection, epoch=epoch, stage='joint',
        completed_training_epochs=last['completed_epoch'], global_updates=last['global_updates'],
        training_batches_per_epoch=batches, checkpoint_sha256=expected_digest,
        model_tensor_sha256=tensor_tree_hash(selected['model_state']),
        registered_parameters=last['registered_parameters'],
        full_protocol_reached=last['completed_epoch'] == identity['config']['max_epochs'],
        best_includes_stage_zero=True)
    return selected['model_state'], role, expected, path


def predict(model, loader, data, device, offset):
    model.eval()
    fields = {k:[] for k in ('prediction', 'target', 'reference', 'regional', 'alpha',
                             'beta', 'support', 'available', 'state_rms_batch',
                             'history_prediction', 'history_target', 'history_valid')}
    forward_seconds, forward_batches = 0., 0
    if torch.device(device).type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    started = time.perf_counter()
    with torch.no_grad():
        for raw in loader.get_iterator():
            x, target, event = device_batch(raw, data, device)
            synchronize(device)
            tick = time.perf_counter()
            output = model(x, event)
            synchronize(device)
            forward_seconds += time.perf_counter()-tick
            forward_batches += 1
            for key in fields:
                value = target if key == 'target' else output['state_rms'] if key == 'state_rms_batch' else output[key]
                fields[key].append(np.array(value.detach().cpu().numpy(), copy=True, order='C'))
    synchronize(device)
    total_seconds = time.perf_counter()-started
    archive = {k:np.stack(v) if k == 'state_rms_batch' else np.concatenate(v) for k, v in fields.items()}
    require(all(np.isfinite(value).all() for value in archive.values()), 'Nonfinite inference output')
    archive.update(station_ids=np.asarray(data.station_ids, dtype=np.int64),
        sample_ids=np.arange(offset, offset+loader.size, dtype=np.int64), assignment=data.assignment.copy())
    archive = canonical_arrays(archive)
    timing = dict(full_collection_seconds=total_seconds, synchronized_forward_seconds=forward_seconds,
        forward_batches=forward_batches, samples=loader.size,
        batched_forward_seconds_per_sample=forward_seconds/loader.size,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if torch.device(device).type == 'cuda' else 0,
        warmup_batches=0, cold_first_batch_included=True,
        forward_scope='history_and_event_to_all_model_outputs; excludes_H2D_D2H_and_serialization',
        collection_scope='loader_H2D_forward_D2H; excludes_npz_writing_and_numpy_analysis')
    return archive, timing


def validation_replay(diagnostics, expected):
    observed = dict(diagnostics['overall'], region_mae=diagnostics['region_mae'],
        reference_mae=diagnostics['reference']['mae'],
        history_preupdate_mae=diagnostics['history_preupdate']['mae'])
    differences = {}
    for key in ('mae', 'rmse', 'mape_percent', 'signed_bias', 'negative_fraction',
                'region_mae', 'reference_mae', 'history_preupdate_mae'):
        if key not in expected:
            continue
        left, right = observed[key], expected[key]
        require((left is None) == (right is None), 'Validation replay missing metric: '+key)
        differences[key] = abs(left-right) if left is not None else 0.
    for key in ('valid_count', 'mape_valid_count'):
        require(observed[key] == expected[key], 'Validation replay target count differs: '+key)
    if 'region_valid_count' in expected:
        require(sum(row['valid_aggregate_cells'] for row in diagnostics['per_region'].values())
            == expected['region_valid_count'], 'Validation replay region count differs')
    if 'history_valid_count' in expected:
        require(diagnostics['history_preupdate']['valid_count'] == expected['history_valid_count'],
            'Validation replay historical target count differs')
    return dict(passed=all(v <= 1e-5 for v in differences.values()),
        absolute_differences=differences, tolerance=1e-5, stored_validation=expected)


def evaluate_arm(args, identity, data, arm, environment, logger, reference_arrays):
    directory = args.output_dir/arm
    weights, role, expected, checkpoint_path = selected_checkpoint(args.run_dir, identity, arm,
        args.selection, data.loaders['train_loader'].num_batch)
    if directory.exists():
        arrays, metadata = read_archive(directory)
        require(metadata['identity_sha256'] == identity['identity_sha256']
            and metadata['checkpoint_role'] == role and metadata['arm'] == arm,
            'Existing evaluation belongs to another checkpoint; choose a fresh --output-dir')
        require(validation_replay(metadata['diagnostics'], expected)['passed'],
            'Cached validation replay differs from the selected training checkpoint')
        if not identity['engineering_only']:
            require_reference_axes(arrays, reference_arrays)
        logger.info('%s certified archive reused; zero new inference calls', arm)
        return metadata
    model = build_model(data, identity['config'], arm, args.device)
    model.load_state_dict(weights, strict=True)
    model.set_stage('joint')
    loader, offset = validation_loader(data, 'val', identity['config']['batch_size'], logger)
    archive, timing = predict(model, loader, data, args.device, offset)
    if not identity['engineering_only']:
        require_reference_axes(archive, reference_arrays)
    diagnostics = archive_diagnostics(archive, identity['config'])
    replay = validation_replay(diagnostics, expected)
    require(replay['passed'], 'Validation replay differs from training: '+arm)
    require(sha256(checkpoint_path) == role['checkpoint_sha256'], 'Checkpoint changed during inference')
    require(source_hashes() == identity['execution_source'], 'Execution source changed during evaluation')
    verify_reference(identity['gw16_reference'])
    # A failed serialization leaves a distinct audit directory, never a seemingly
    # reusable arm archive. The rename commits all files together.
    temporary = args.output_dir/('.'+arm+'.writing_'+uuid.uuid4().hex)
    temporary.mkdir()
    np.savez_compressed(temporary/'predictions.npz', **archive)
    metadata = dict(schema=ARCHIVE_SCHEMA, identity_sha256=identity['identity_sha256'],
        identity=identity, checkpoint_role=role, arm=arm, split='val',
        archive_sha256=sha256(temporary/'predictions.npz'), diagnostics=diagnostics,
        validation_replay=replay, inference_timing=timing, runtime=environment,
        metric_contract=metric_contract(identity['config']), engineering_only=identity['engineering_only'],
        test_evaluated=False, independent_confirmation=False,
        historical_runs_reused_as_training_sources=False,
        canonical_array_layout='C_float32_model_values_no_precision_change',
        gw16_reference=identity['gw16_reference'],
        old_baseline_new_training_updates=0, old_baseline_new_inference_calls=0)
    write_json(temporary/'metadata.json', metadata)
    write_json(temporary/'analysis.json', diagnostics)
    read_archive(temporary)
    temporary.rename(directory)
    logger.info('%s archived MAE=%.8f P95=%.8f', arm, diagnostics['overall']['mae'], diagnostics['window_p95'])
    del model, weights, archive
    return metadata


def render_report(directory, records, selection, reference_metadata, *, engineering_only=False):
    lines = ['# GW17 历史观测校正与连续状态演化：'+selection+' / val', '',
        'OBSERVATION_CONTROL与INNOVATION_STATE从头训练；两组参数量完全相同，正式预算各100轮。',
        '前者注入观测变化，后者注入更新前预测偏差；两组接受同样的历史一步预测辅助监督。',
        'GW16_REFERENCE只复用此前STATE_EXCHANGE BEST的837窗验证缓存，新增推理次数为0。',
        'GW16不是本轮等预算训练对照；新历史递推及辅助损失带来额外计算，不能只按参数量声称成本相同。',
        '发布整数组的反序列化包含test载荷；本入口只构建验证loader，不加载历史测试缓存。',
        'BEST包含训练前第0轮；BEST和LAST同权重不能当成两次独立实验。',
        '未来主指标为原单位float64 pooled口径；历史MAE在观测更新之前计算，不参与BEST选择。', '',
        '| 组 | 已完成轮数 | 选中轮数 | 参数量 | 未来MAE | RMSE | MAPE % | 窗口P95 | 偏差 | 历史更新前MAE |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    number = lambda value: 'NA' if value is None else format(value, '.8f')
    displayed = {} if engineering_only else {'GW16_REFERENCE':reference_metadata}
    displayed.update(records)
    for arm, m in displayed.items():
        d, role = m['diagnostics'], m['checkpoint_role']
        point = d['overall']
        values = [arm, str(role['completed_training_epochs']), str(role['epoch']), str(role['registered_parameters']),
            *(number(v) for v in (point['mae'], point['rmse'], point['mape_percent'], d['window_p95'],
                                 point.get('signed_bias'), d.get('history_preupdate', {}).get('mae')))]
        lines.append('| '+' | '.join(values)+' |')
    lines.extend(['', '每时距、区域、事故关系支持、低流量及预定分组见各组analysis.json。',
        '窗口存在历史重叠；这是重复使用的单seed开发验证，不能形成多seed显著性或严格时间独立性结论。',
        '完整100轮后运行screen；候选相对CONTROL和GW16缓存各需至少0.5% MAE收益，',
        '并保持RMSE、MAPE、窗口P95不退化；LAST相对自身BEST退化不超过1%。',
        '工程小样本或部分预算不可成为确认候选；筛查失败不授权延长轮数或修改标准。',
        'GW16测试已经公开，本轮不自动打开或反复使用该测试；通过也只是开发候选资格，不是独立确认。',
        '计时含首次冷批，不能与历史异口径耗时直接解释成加速比。', ''])
    if engineering_only:
        lines.extend(['工程小样本仅核对自身流程，未与完整837窗GW16缓存计算配对收益。', ''])
    (directory/'report.md').write_text('\n'.join(lines), encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--arms', nargs='+', choices=ARMS, default=list(ARMS))
    parser.add_argument('--selection', choices=('best', 'last'), default='best')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--threads', type=int, default=3)
    args = parser.parse_args(argv)
    args.run_dir, args.data_dir = args.run_dir.resolve(), args.data_dir.resolve()
    require(len(set(args.arms)) == len(args.arms), 'Duplicate evaluation arms')
    logger = logging.getLogger('gw17-evaluate')
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    logger.addHandler(handler)
    try:
        with exclusive_run(args.run_dir):
            identity = read_json(args.run_dir/'identity.json')
            validate_identity(identity)
            require(identity['schema'] == SCHEMA and source_hashes() == identity['execution_source'],
                'Source differs from training identity')
            args.output_dir = protected_output(args.output_dir or args.run_dir/('eval_'+args.selection+'_val'),
                args.data_dir, allow_run_children=True)
            require(args.output_dir != args.run_dir, 'Evaluation needs its own subdirectory')
            args.output_dir.mkdir(parents=True, exist_ok=True)
            environment = runtime(args.device, args.threads, identity['seed'])
            validate_runtime(read_json(args.run_dir/'runtime.json'), environment)
            data = load_data(args.data_dir, identity['config'], logger, engineering=identity['engineering_only'])
            require(data.identity == identity['data'], 'Dataset, position or partition differs from training')
            reference_arrays, reference_metadata, reference_record = read_reference(
                identity['gw16_reference']['directory'], data=None if identity['engineering_only'] else data)
            require(reference_record == identity['gw16_reference'], 'GW16 reference changed after training')
            records = {arm:evaluate_arm(args, identity, data, arm, environment, logger,
                reference_arrays) for arm in args.arms}
            verify_reference(identity['gw16_reference'])
            require(source_hashes() == identity['execution_source'], 'Execution source changed during evaluation')
            render_report(args.output_dir, records, args.selection, reference_metadata,
                engineering_only=identity['engineering_only'])
            write_json(args.output_dir/'summary.json', dict(schema='observed_state_gw17_evaluation_summary_v1',
                identity_sha256=identity['identity_sha256'], selection=args.selection, split='val',
                arms={arm:dict(checkpoint_role=m['checkpoint_role'], diagnostics=m['diagnostics']) for arm, m in records.items()},
                gw16_reference=identity['gw16_reference'],
                gw16_historical_diagnostics=None if identity['engineering_only'] else reference_metadata['diagnostics'],
                historical_reference_is_equal_budget_control=False,
                engineering_only=identity['engineering_only'], test_evaluated=False, independent_confirmation=False))
            print(json.dumps(dict(status='GW17_EVALUATION_COMPLETE', report=str(args.output_dir/'report.md'),
                arms=list(records), test_evaluated=False), ensure_ascii=False))
            return records
    finally:
        logger.removeHandler(handler)
        handler.close()


if __name__ == '__main__':
    main()
