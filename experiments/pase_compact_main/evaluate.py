"""BEST evaluation of compact PASE and its two matched directional ablations.

Validation is the default. Test requires an explicit flag, completed formal
100-epoch training for each requested arm, and frozen validation BESTs.
This entry performs no fitting, checkpoint selection on test, or plotting.
"""
import argparse
from contextlib import ExitStack, nullcontext
import csv
import gc
import logging
from pathlib import Path
import uuid

import numpy as np
import torch

from experiments.chronological.original_igstgnn_runtime import rng_fingerprint
from experiments.original_baseline_gw16.evaluate import native_summary
from experiments.position_state_gw16.common import (
    digest_json, exclusive_run, load_state, read_json, require, runtime,
    sha256, synchronize, tensor_tree_hash, write_json)
from experiments.state_exchange_multidata.data import device_batch, load_data, validation_loader
from experiments.position_state_gw16.objective import PooledStatistics, numpy_metrics
from .model import build_model, capture_exchange_diagnostics, parameter_accounting
from .protocol import CORE_ARMS
from experiments.pase_compact_v3.promote import load_promoted
from .state import expected_batches, source_hashes, validate_identity, validate_runtime, validate_saved


SCHEMA = 'pase_compact_main_best_evaluation_v1'
CONVERSION_REPLAY_TOLERANCE = 1e-3
METRIC_CONTRACT = dict(
    primary='unchanged_author_native_FP32_each_lead_then_equal_average_over_12',
    rmse='mean_of_12_per_lead_RMSE_not_pooled_RMSE',
    native_mask='minimum_of_entire_split_target_if_below_1_else_zero',
    mape_units='percent', reported_leads=[3, 6, 12], selection='validation_pooled_MAE',
    clipping=False, calibration=False, future_target_passed_to_forward=False)
REPO = Path(__file__).resolve().parents[2]


def evaluation_sources():
    """Include metric implementation and the evaluation entry in cache identity."""
    names = ['experiments/original_baseline_gw16/evaluate.py',
             'experiments/chronological/original_igstgnn_source.py',
             'experiments/chronological/original_igstgnn_runtime.py']
    paths = [*(Path(__file__).parent.glob('*.py')),
             *(REPO/'experiments/position_state_gw16').glob('*.py'),
             *(REPO/'experiments/chronological/original_igstgnn_558ab01').glob('*.py'),
             *(REPO/name for name in names)]
    result = source_hashes()
    result.update({path.relative_to(REPO).as_posix(): sha256(path) for path in sorted(set(paths))})
    return result


def selected_best(run, identity, arm):
    """Authenticate LAST history and its immutable validation-selected BEST."""
    if arm == 'FULL' and identity.get('external_full_reference'):
        reference = identity['external_full_reference']
        directory = Path(reference['directory'])
        require(sha256(directory/'metadata.json') == reference['metadata_sha256']
                and sha256(directory/'model.pt') == reference['model_sha256'],
                'Promoted FULL reference changed')
        weights, role, metadata = load_promoted(directory, expected_dataset=identity['dataset'])
        role = dict(role, full_protocol_reached=role['completed_training_epochs'] == 100)
        return weights, role, metadata['selection_metrics']
    directory = Path(run)/arm
    last_path = directory/'last.pt'
    before = sha256(last_path)
    last = load_state(last_path)
    require(sha256(last_path) == before, 'LAST changed while reading: '+arm)
    batches = expected_batches(identity, 'train')
    validate_saved(last, identity, arm, directory, batches)
    filename = last['best_checkpoint_file']
    require(isinstance(filename, str) and Path(filename).name == filename,
            'BEST must be a local immutable filename: '+arm)
    path = directory/filename
    checksum = sha256(path)
    require(checksum == last['best_checkpoint_sha256'], 'BEST checksum changed: '+arm)
    saved = load_state(path)
    require(sha256(path) == checksum and saved.get('schema') == identity['schema']
            and saved.get('stage') == 'joint'
            and saved['identity_sha256'] == identity['identity_sha256']
            and saved['arm'] == arm and saved['epoch'] == last['best_epoch']
            and saved['validation'] == last['best_metrics']
            and saved['model_tensor_sha256'] == tensor_tree_hash(saved['model_state']),
            'BEST role differs from validation history: '+arm)
    role = dict(selection='best', epoch=saved['epoch'],
        completed_training_epochs=last['completed_epoch'],
        checkpoint_file=filename, checkpoint_sha256=checksum,
        model_tensor_sha256=tensor_tree_hash(saved['model_state']),
        best_includes_stage_zero=True, selected_on='validation_pooled_mae',
        full_protocol_reached=last['completed_epoch'] == 100)
    return saved['model_state'], role, saved['validation']


def converted_source_validation(identity, arm):
    """Return source references only; never reuse the old 128-input predictions."""
    if arm != 'FULL' or not identity.get('external_full_reference'):
        return None
    reference = identity['external_full_reference']
    directory = Path(reference['directory'])
    require(sha256(directory/'metadata.json') == reference['metadata_sha256']
            and sha256(directory/'model.pt') == reference['model_sha256'],
            'Promoted FULL reference changed')
    _, _, metadata = load_promoted(directory, expected_dataset=identity['dataset'])
    require(metadata.get('conversion_validation_required') is True
            and metadata.get('source_metrics_are_reference_only') is True
            and metadata.get('copied_prediction_cache') is False,
            'Compact FULL must require actual converted-model validation')
    axes = metadata['validation_axes']
    require(set(axes) == {'target_tensor_sha256', 'sample_ids_sha256', 'station_ids_sha256'},
            'Converted-model reference axes are incomplete')
    return dict(validation_axes=axes, native_metrics=metadata['native_metrics'],
                selection_metrics=metadata['selection_metrics'])


def validate_converted_forward(arrays, pooled, native, reference):
    """Certify the fixed converted BEST by a fresh complete validation forward.

    Slicing the zero-input columns changes the FP32 GEMM shape. Tensor algebra
    agrees, but bit-identical predictions are deliberately not promised. The
    fixed 1e-3 absolute metric bound is a conversion check, not a selection rule.
    """
    axes = {name: tensor_tree_hash(arrays[field]) for name, field in (
        ('target_tensor_sha256', 'target'), ('sample_ids_sha256', 'sample_ids'),
        ('station_ids_sha256', 'station_ids'))}
    require(axes == reference['validation_axes'],
            'Converted FULL target, sample order or station axis differs from its source')
    pooled_replay = replay_validation(pooled, reference['selection_metrics'],
                                      CONVERSION_REPLAY_TOLERANCE)
    expected = reference['native_metrics']
    require(native['mask_value'] == expected['mask_value']
            and [row['lead'] for row in native['per_horizon']]
                == [row['lead'] for row in expected['per_horizon']],
            'Converted FULL native mask or lead axis differs from its source')
    differences = {}
    for key in ('mae', 'rmse', 'mape_percent'):
        differences['macro_'+key] = abs(native['horizon_macro'][key]
                                       - expected['horizon_macro'][key])
        for row, old in zip(native['per_horizon'], expected['per_horizon']):
            differences['h'+str(row['lead'])+'_'+key] = abs(row[key]-old[key])
    require(all(value <= CONVERSION_REPLAY_TOLERANCE for value in differences.values()),
            'Converted FULL native metrics exceed the fixed numerical replay tolerance')
    return dict(passed=True, actual_compact_model_forward=True,
        source_prediction_cache_reused=False, axes=axes,
        tolerance=CONVERSION_REPLAY_TOLERANCE,
        native_absolute_differences=differences, selection_metric_replay=pooled_replay,
        numerical_contract='FP32_metric_tolerance_not_prediction_bit_equality',
        new_training_updates=0, new_checkpoint_selection=False)


def test_guard(identity, roles, requested_arms, engineering=False):
    """Before any test loader, authenticate every requested completed BEST.

    This is a versioned development comparison, never a new independent holdout.
    Optional component experiments need not be trained to evaluate FULL.
    """
    require(not engineering and identity.get('engineering_only') is False,
            'Engineering runs cannot evaluate test')
    require(identity['config']['max_epochs'] == 100
            and set(identity['comparison_arms']) == set(CORE_ARMS),
            'Test requires the fixed compact PASE protocol')
    require(set(roles) == set(requested_arms) and set(requested_arms) <= set(CORE_ARMS),
            'Test must authenticate every requested arm')
    for arm in requested_arms:
        role = roles[arm]
        require(role['completed_training_epochs'] == 100
                and role['full_protocol_reached'] is True
                and role['selection'] == 'best' and role['selected_on'] == 'validation_pooled_mae'
                and role['best_includes_stage_zero'] is True,
                'Test requires a completed validation-selected BEST for '+arm)


def replay_validation(observed, expected, tolerance=1e-5):
    differences = {}
    for key in ('mae', 'rmse', 'mape_percent', 'signed_bias', 'negative_fraction',
                'region_mae', 'reference_mae'):
        if key not in expected:
            continue
        left, right = observed[key], expected[key]
        require((left is None) == (right is None), 'Validation replay missing '+key)
        differences[key] = abs(left-right) if left is not None else 0.
    for key in ('valid_count', 'mape_valid_count', 'region_valid_count'):
        if key in expected:
            require(observed[key] == expected[key], 'Validation replay count differs: '+key)
    require(all(value <= tolerance for value in differences.values()),
            'BEST validation replay differs from the training selection metrics')
    return dict(passed=True, absolute_differences=differences, tolerance=tolerance)


def collect(model, data, config, split, device, logger, max_batches=None, diagnostics=False):
    """Collect ordered predictions without feeding future labels to forward."""
    loader, offset = validation_loader(data, split, config['batch_size'], logger)
    model.eval()
    before = tensor_tree_hash(model.state_dict())
    random_before = rng_fingerprint()
    predictions, targets = [], []
    exchange_records = []
    statistics = PooledStatistics(config)
    with torch.no_grad():
        for index, raw in enumerate(loader.get_iterator(), 1):
            if max_batches is not None and index > max_batches:
                break
            raw_before = tensor_tree_hash(raw)
            history, target, event = device_batch(raw, data, device)
            context = capture_exchange_diagnostics(model) if diagnostics else nullcontext()
            with context as exchange:
                output = model(history, event)
            if diagnostics:
                exchange_records.append(exchange)
            require(tensor_tree_hash(raw) == raw_before, 'Inference modified raw inputs')
            statistics.update(output, target, model.geometry)
            predictions.append(output['prediction'].detach().cpu().numpy())
            targets.append(target.detach().cpu().numpy())
            if index in (1, loader.num_batch, max_batches):
                logger.info('%s %s batch=%d/%d', getattr(model, 'paper_arm', model.arm),
                            split, index, loader.num_batch)
    synchronize(device)
    require(tensor_tree_hash(model.state_dict()) == before,
            'Inference modified registered model tensors')
    require(rng_fingerprint() == random_before, 'Evaluation consumed RNG state')
    count = sum(len(value) for value in targets)
    expected_forward_batches = min(loader.num_batch, max_batches) if max_batches else loader.num_batch
    require(len(predictions) == expected_forward_batches,
            'Evaluation loader ended before the declared number of batches')
    if max_batches is None or max_batches >= loader.num_batch:
        require(count == loader.size, 'Evaluation loader sample count differs from its complete split')
    arrays = dict(prediction=np.concatenate(predictions), target=np.concatenate(targets),
        sample_ids=np.arange(offset, offset+count, dtype=np.int64),
        station_ids=np.asarray(data.station_ids, dtype=np.int64))
    p, y = arrays['prediction'], arrays['target']
    require(count > 0 and p.shape == y.shape == (count, config['horizon'], config['nodes'], 1)
            and p.dtype == y.dtype == np.float32
            and np.isfinite(p).all() and np.isfinite(y).all(),
            'Unexpected prediction shape, dtype, or nonfinite value')
    audit = dict(model_tensor_sha256=before,
        registered_tensors_unchanged=True, rng_unchanged=True, raw_inputs_unchanged=True,
        future_target_passed_to_forward=False, optimizer_updates=0,
        forward_batches=expected_forward_batches,
        samples=count)
    if diagnostics:
        audit['exchange_diagnostics'] = dict(
            coefficient_rms_mean_over_batches=np.mean(
                [item['coefficient_rms'] for item in exchange_records], axis=0).tolist(),
            message_rms_mean_over_batches=np.mean(
                [item['message_rms'] for item in exchange_records], axis=0).tolist(),
            active_directions=exchange_records[0]['active_directions'],
            aggregation='arithmetic_mean_of_batch_RMS_not_population_RMS',
            no_additional_model_forwards=True, not_a_timing_measurement=True)
    return arrays, statistics.result(), audit


def cache_digest(metadata):
    return digest_json({key:value for key, value in metadata.items() if key != 'payload_sha256'})


def read_cache(directory, frozen, key):
    directory = Path(directory)
    metadata = read_json(directory/'metadata.json')
    require(metadata.get('schema') == SCHEMA and metadata.get('payload_sha256') == cache_digest(metadata),
            'Evaluation cache metadata checksum differs')
    require(metadata.get('frozen_sha256') == frozen['frozen_sha256']
            and metadata.get('record_key') == key and metadata.get('split') == frozen['split']
            and metadata.get('metric_contract') == METRIC_CONTRACT,
            'Evaluation cache belongs to another checkpoint, split, runtime, or protocol')
    require(metadata['checkpoint_role'] == frozen['records'][key]['role'],
            'Evaluation cache BEST role changed')
    audit = metadata['audit']
    require(audit['model_tensor_sha256'] == metadata['checkpoint_role']['model_tensor_sha256']
            and all(audit[name] is True for name in
                    ('registered_tensors_unchanged', 'rng_unchanged', 'raw_inputs_unchanged'))
            and audit['future_target_passed_to_forward'] is False and audit['optimizer_updates'] == 0,
            'Cached inference audit differs from frozen BEST')
    source = frozen['records'][key].get('source_validation')
    if source is not None and metadata.get('split') == 'val':
        conversion = metadata.get('conversion_validation_replay', {})
        require(metadata.get('converted_model_replayed') is True
                and conversion.get('passed') is True
                and conversion.get('actual_compact_model_forward') is True
                and conversion.get('source_prediction_cache_reused') is False
                and conversion.get('new_training_updates') == 0
                and conversion.get('new_checkpoint_selection') is False
                and conversion.get('tolerance') == CONVERSION_REPLAY_TOLERANCE
                and conversion.get('axes') == source['validation_axes']
                and {name: metadata[name] for name in source['validation_axes']}
                    == source['validation_axes'],
                'Converted FULL cache lacks a certified actual compact-model replay')
    prediction_file = metadata.get('prediction_file')
    if prediction_file:
        path = directory/prediction_file
        require(Path(prediction_file).name == prediction_file
                and sha256(path) == metadata['prediction_sha256'], 'Prediction cache checksum differs')
        with np.load(path, allow_pickle=False) as saved:
            arrays = dict(saved)
        require(set(arrays) == {'prediction', 'target', 'sample_ids', 'station_ids'}
                and tensor_tree_hash(arrays['target']) == metadata['target_tensor_sha256']
                and tensor_tree_hash(arrays['sample_ids']) == metadata['sample_ids_sha256']
                and tensor_tree_hash(arrays['station_ids']) == metadata['station_ids_sha256'],
                'Cached prediction axes or target differ')
        require(native_summary(arrays['prediction'], arrays['target']) == metadata['native_metrics'],
                'Cached author metric values differ')
        require(numpy_metrics(arrays['prediction'], arrays['target'], metadata['config'])
                == metadata['common_mask_pooled'], 'Cached pooled metrics differ')
    return metadata


def commit_cache(directory, arrays, metadata, save_predictions, frozen, key):
    """Rename commits the whole arm atomically; interrupted writes stay distinct."""
    directory = Path(directory)
    require(not directory.exists(), 'Refusing to overwrite a completed arm cache')
    temporary = directory.with_name('.'+directory.name+'.writing_'+uuid.uuid4().hex)
    temporary.mkdir(parents=True)
    metadata = dict(metadata, prediction_file=None, prediction_sha256=None)
    if save_predictions:
        np.savez_compressed(temporary/'predictions.npz', **arrays)
        metadata.update(prediction_file='predictions.npz',
                        prediction_sha256=sha256(temporary/'predictions.npz'))
    metadata['payload_sha256'] = cache_digest(metadata)
    write_json(temporary/'metadata.json', metadata)
    read_cache(temporary, frozen, key)
    temporary.rename(directory)
    return metadata


def import_conversion_cache(source_directory, destination, frozen, key, save_predictions=False):
    """Reuse an already executed compact FULL replay across evaluation scopes.

    A v2 zero-input cache is never accepted here. Only this evaluator's completed
    FULL-only compact-model validation certificate can populate the larger
    matched-ablation report without repeating its forward pass.
    """
    source_directory = Path(source_directory)
    require(key == 's2025__FULL' and frozen['split'] == 'val'
            and frozen['engineering_only'] is False,
            'Conversion-cache import is only for the formal compact FULL validation')
    source_frozen = read_json(source_directory/'frozen.json')
    source_payload = {name:value for name, value in source_frozen.items() if name != 'frozen_sha256'}
    require(source_frozen.get('frozen_sha256') == digest_json(source_payload)
            and source_frozen.get('schema') == SCHEMA
            and source_frozen.get('arms') == ['FULL']
            and source_frozen.get('seeds') == [2025]
            and source_frozen.get('split') == 'val'
            and source_frozen.get('engineering_only') is False,
            'Expected a completed FULL-only compact validation certificate')
    for name in ('dataset', 'runtime', 'metric_contract', 'evaluation_source'):
        require(source_frozen[name] == frozen[name],
                'Compact conversion certificate changed '+name)
    require(source_frozen['evaluation_source'] == evaluation_sources(),
            'Compact conversion certificate numerical source changed')
    require(source_frozen['records'].get(key) == frozen['records'].get(key)
            and frozen['records'][key].get('source_validation') is not None,
            'Compact conversion certificate BEST role, identity or source validation changed')
    source_cache = source_directory/key
    metadata = read_cache(source_cache, source_frozen, key)
    require(metadata.get('converted_model_replayed') is True
            and metadata.get('arm') == 'FULL' and metadata.get('seed') == 2025
            and metadata.get('identity_sha256') == frozen['records'][key]['identity_sha256'],
            'Compact conversion certificate identity or replay scope changed')
    checksum = sha256(source_cache/'metadata.json')
    arrays = None
    if save_predictions:
        filename = metadata.get('prediction_file')
        require(filename is not None,
                'Compact conversion certificate has no prediction arrays; '
                'create the FULL replay with --save-predictions before requesting their import')
        with np.load(source_cache/filename, allow_pickle=False) as saved:
            arrays = dict(saved)
    require(sha256(source_cache/'metadata.json') == checksum
            and read_json(source_directory/'frozen.json') == source_frozen,
            'Compact conversion certificate changed while being imported')
    imported = dict(metadata, frozen_sha256=frozen['frozen_sha256'], record_key=key,
        imported_from=dict(directory=str(source_directory.resolve()),
            frozen_sha256=source_frozen['frozen_sha256'], metadata_sha256=checksum,
            model_tensor_sha256=metadata['checkpoint_role']['model_tensor_sha256'],
            already_executed_compact_model=True, original_v2_prediction_cache=False,
            new_forward_batches=0, new_training_updates=0))
    return commit_cache(destination, arrays, imported, save_predictions, frozen, key)


def aggregate(records, arms):
    result = {}
    for arm in arms:
        selected = [item for item in records if item['arm'] == arm]
        require(selected and len({item['seed'] for item in selected}) == len(selected),
                'Aggregation requires one result per requested arm and seed')
        require(all(item['parameters'] == selected[0]['parameters'] for item in selected),
                'Parameter accounting differs across seeds: '+arm)
        rows = {}
        for metric in ('mae', 'rmse', 'mape_percent'):
            values = [item['native_metrics']['horizon_macro'][metric] for item in selected]
            rows[metric] = dict(mean=float(np.mean(values)),
                sample_std=float(np.std(values, ddof=1)) if len(values) > 1 else None,
                seed_values={str(item['seed']):value for item, value in zip(selected, values)})
        result[arm] = dict(seeds=[item['seed'] for item in selected], metrics=rows,
                          parameters=selected[0]['parameters'])
    return result


def render_report(directory, records, frozen):
    directory = Path(directory)
    summary = aggregate(records, frozen['arms'])
    labels = {'FULL':'PASE-Net (compact MLP)',
        'UP_ONLY':'仅局域→区域', 'DOWN_ONLY':'仅区域→局域'}
    number = lambda value: '—' if value is None else format(value, '.6f')
    lines = ['# PASE-Net 紧凑模型主结果', '',
        '数据集：'+frozen['dataset']+'；选择：各臂验证集 pooled MAE 最优的 BEST（含第0轮）。',
        '评价划分：'+frozen['split']+'；平均指标逐时距按原生 FP32 计算，再对12步等权平均。',
        'RMSE 为12个逐步 RMSE 的平均；未使用测试结果选择模型、调整预算或校准。', '',
        '工程检查结果不能作为论文精度实验。' if frozen['engineering_only'] else
        '当前紧凑结构固定100轮、验证BEST；已有公开测试划分不构成新的独立保留集确认。', '']
    groups = [('完整模型预测结果', frozen['arms'])]
    for title, arms in groups:
        lines.extend(['## '+title, '', '| 方法 | Seed | BEST轮数 | 已训练轮数 | MAE | RMSE | MAPE % |',
                      '|---|---:|---:|---:|---:|---:|---:|'])
        for arm in arms:
            for record in records:
                if record['arm'] != arm:
                    continue
                m, r = record['native_metrics']['horizon_macro'], record['checkpoint_role']
                lines.append('| '+' | '.join([labels.get(arm, arm), str(record['seed']),
                    str(r['epoch']), str(r['completed_training_epochs']),
                    *[number(m[key]) for key in ('mae', 'rmse', 'mape_percent')]])+' |')
        lines.append('')
        if len(frozen['seeds']) > 1:
            lines.extend(['| 方法 | Seeds | MAE 均值±样本标准差 | RMSE 均值±样本标准差 | MAPE % 均值±样本标准差 |',
                          '|---|---:|---:|---:|---:|'])
            for arm in arms:
                rows = summary[arm]['metrics']
                values = [number(rows[key]['mean'])+' ± '+number(rows[key]['sample_std'])
                          for key in ('mae', 'rmse', 'mape_percent')]
                lines.append('| '+' | '.join([labels.get(arm, arm), str(len(frozen['seeds'])), *values])+' |')
            lines.append('')
    lines.extend(['## 参数统计', '',
        '| 方法 | 注册参数 | 结构活跃参数 | 系数生成器注册参数 |',
        '|---|---:|---:|---:|'])
    for arm in frozen['arms']:
        p = summary[arm]['parameters']
        lines.append('| '+' | '.join([labels.get(arm, arm), str(p['registered']),
            str(p['structurally_active']), str(p['coefficient_registered'])])+' |')
    conversion_note = [
        'FULL 由已认证的MLP v2 NO_POSITION_INPUT验证BEST转换；原训练100轮、原BEST轮数保持不变，不新增训练或重新选择检查点。',
        '转换后的96维交换器必须实际执行完整验证前向；源128维模型指标仅用于固定1e-3绝对误差容限的数值核对，不冒称预测位完全一致。'
        ] if any(item.get('converted_model_replayed') is True for item in records) else []
    lines.extend(['',
        'parameters.csv 记录全部注册参数和结构活跃参数；关闭通路保留的注册参数不冒称等容量参与训练。',
        'H3/H6/H12 见 selected_horizons.csv；全部12步见 per_horizon.csv。',
        *conversion_note,
        '本表仅包含当前紧凑FULL模型，各数据集从头训练，不运行其他基线或消融。',
        '道路位置仍参与历史编码、条件构建和区域划分；此处只移除交换器额外的位置输入块。',
        '本入口不测推理速度；缓存读取不重复执行模型前向。', ''])
    (directory/'report.md').write_text('\n'.join(lines), encoding='utf-8')
    fields = ['arm', 'seed', 'lead', 'mae', 'rmse', 'mape_percent']
    for filename, leads in (('per_horizon.csv', None), ('selected_horizons.csv', {3, 6, 12})):
        with (directory/filename).open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for record in records:
                for row in record['native_metrics']['per_horizon']:
                    if leads is None or row['lead'] in leads:
                        writer.writerow(dict(arm=record['arm'], seed=record['seed'], **row))
    with (directory/'parameters.csv').open('w', encoding='utf-8', newline='') as stream:
        parameter_fields = ['registered', 'trainable', 'structurally_active', 'coefficient_registered',
                            'coefficient_active', 'inactive_registered', 'buffer_bytes', 'parameter_bytes']
        writer = csv.DictWriter(stream, fieldnames=['arm', 'seed', *parameter_fields,
                                                    'parameter_accounting_json'])
        writer.writeheader()
        import json
        for record in records:
            writer.writerow(dict(arm=record['arm'], seed=record['seed'],
                **{name:record['parameters'].get(name) for name in parameter_fields},
                parameter_accounting_json=json.dumps(record['parameters'], ensure_ascii=False, sort_keys=True)))
    write_json(directory/'summary.json', dict(schema=SCHEMA, split=frozen['split'],
        engineering_only=frozen['engineering_only'], test_evaluated=frozen['split'] == 'test',
        frozen_sha256=frozen['frozen_sha256'], primary_metric_contract=METRIC_CONTRACT,
        results=summary, records=records, independent_confirmation=False))


def safe_output(output, root, data_dir):
    output, root, data_dir = Path(output).resolve(), Path(root).resolve(), Path(data_dir).resolve()
    require(output != root and output not in root.parents,
            'Evaluation output must not overwrite the run root or its parents')
    require(output != data_dir and data_dir not in output.parents and output not in data_dir.parents,
            'Evaluation output must not intersect raw data')
    for parent in (output, *output.parents):
        require(not (parent/'identity.json').is_file(), 'Evaluation output cannot be inside a seed training run')
    return output


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--seeds', type=int, nargs='+', choices=(2025,), default=[2025])
    parser.add_argument('--arms', nargs='+', choices=CORE_ARMS, default=['FULL'])
    parser.add_argument('--split', choices=('val', 'test'), default='val')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--threads', type=int, default=3)
    parser.add_argument('--save-predictions', action='store_true')
    parser.add_argument('--conversion-evaluation', type=Path,
        help='Reuse this evaluator\'s completed FULL-only compact validation; never a v2 source cache')
    parser.add_argument('--engineering', action='store_true')
    args = parser.parse_args(argv)
    require(len(set(args.seeds)) == len(args.seeds) and len(set(args.arms)) == len(args.arms),
            'Seeds and arms must be unique')
    require(args.threads >= 1 and not (args.engineering and args.split == 'test'),
            'Use positive threads; engineering cannot evaluate test')
    require(args.conversion_evaluation is None or (not args.engineering and args.split == 'val'
            and 'FULL' in args.arms),
            'Conversion replay import requires formal validation including FULL')
    args.output_dir = safe_output(args.output_dir or args.run_root/('evaluation_'+args.split),
                                  args.run_root, args.data_dir)
    return args


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger('pase-compact-evaluate')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        stack.enter_context(exclusive_run(args.output_dir))
        for seed in sorted(args.seeds):
            stack.enter_context(exclusive_run(args.run_root/('s'+str(seed))))
        runs, records, roles = {}, {}, {}
        for seed in args.seeds:
            run = args.run_root/('s'+str(seed))
            identity = read_json(run/'identity.json')
            validate_identity(identity)
            require(identity['execution_source'] == source_hashes(),
                    'Numerical execution source differs from training identity')
            require(identity['seed'] == seed and identity['engineering_only'] == args.engineering,
                    'Seed or engineering scope differs from identity')
            for name, checksum in identity['data']['hashes'].items():
                require(sha256(args.data_dir/name) == checksum,
                        'Dataset fingerprint differs from training identity: '+name)
            require(set(args.arms) <= set(identity['comparison_arms']), 'Unregistered requested arm')
            # Freeze and authenticate every requested BEST before any test loader.
            wanted = args.arms
            seed_roles = {}
            for arm in wanted:
                weights, role, expected = selected_best(run, identity, arm)
                seed_roles[arm] = role
                if arm in args.arms:
                    key = 's'+str(seed)+'__'+arm
                    records[key] = dict(seed=seed, arm=arm, role=role,
                        expected_validation=expected, identity_sha256=identity['identity_sha256'],
                        source_validation=converted_source_validation(identity, arm))
                del weights
            if args.split == 'test':
                test_guard(identity, seed_roles, args.arms, args.engineering)
            runs[seed] = identity
            roles[str(seed)] = seed_roles
        require(len({identity['dataset'] for identity in runs.values()}) == 1,
                'One evaluation cannot mix datasets')
        environments = {}
        for seed in args.seeds:
            environment = runtime(args.device, args.threads, seed)
            validate_runtime(read_json(args.run_root/('s'+str(seed))/'runtime.json'), environment)
            environments[str(seed)] = environment
        frozen = dict(schema=SCHEMA, dataset=next(iter(runs.values()))['dataset'],
            split=args.split, seeds=args.seeds, arms=args.arms,
            engineering_only=args.engineering, metric_contract=METRIC_CONTRACT,
            runtime=environments, evaluation_source=evaluation_sources(), records=records,
            all_authenticated_roles=roles, save_predictions=args.save_predictions)
        frozen['frozen_sha256'] = digest_json(frozen)
        freeze_path = args.output_dir/'frozen.json'
        if freeze_path.exists():
            require(read_json(freeze_path) == frozen,
                    'Frozen selection/runtime differs; use a fresh --output-dir, do not overwrite caches')
        else:
            write_json(freeze_path, frozen)
        result = []
        paired_axes = None
        for seed in args.seeds:
            identity = runs[seed]
            data = None
            for arm in args.arms:
                key = 's'+str(seed)+'__'+arm
                directory = args.output_dir/key
                if directory.exists():
                    metadata = read_cache(directory, frozen, key)
                    logger.info('%s certified cache reused; zero new forwards', key)
                elif arm == 'FULL' and args.conversion_evaluation is not None:
                    metadata = import_conversion_cache(args.conversion_evaluation, directory,
                        frozen, key, args.save_predictions)
                    logger.info('%s certified compact FULL replay imported; zero new forwards', key)
                else:
                    if data is None:
                        data = load_data(args.data_dir, identity['config'], logger,
                                         engineering=args.engineering)
                        require(data.identity == identity['data'], 'Dataset or partition differs from training identity')
                    weights, role, expected = selected_best(args.run_root/('s'+str(seed)), identity, arm)
                    require(role == records[key]['role'], 'BEST changed after freezing')
                    model = build_model(data, identity['config'], arm, args.device)
                    model.load_state_dict(weights, strict=True)
                    model.set_stage('joint')
                    require(parameter_accounting(model) == identity['initializations'][arm]['parameters'],
                            'Model parameter accounting differs from its initialization identity')
                    arrays, pooled, audit = collect(model, data, identity['config'], args.split,
                        args.device, logger, identity['phase_batch_limits']['val']
                        if args.engineering else None, diagnostics=True)
                    source_validation = records[key]['source_validation']
                    native = native_summary(arrays['prediction'], arrays['target'])
                    converted = source_validation is not None and args.split == 'val'
                    replay = replay_validation(pooled, expected,
                        CONVERSION_REPLAY_TOLERANCE if converted else 1e-5
                        ) if args.split == 'val' else None
                    conversion = validate_converted_forward(arrays, pooled, native,
                        source_validation) if converted else None
                    metadata = dict(schema=SCHEMA, frozen_sha256=frozen['frozen_sha256'],
                        record_key=key, seed=seed, arm=arm, split=args.split,
                        identity_sha256=identity['identity_sha256'], checkpoint_role=role,
                        config=identity['config'], metric_contract=METRIC_CONTRACT,
                        native_metrics=native,
                        common_mask_pooled=numpy_metrics(arrays['prediction'], arrays['target'], identity['config']),
                        selection_metric_replay=replay, parameters=parameter_accounting(model), audit=audit,
                        converted_model_replayed=converted, conversion_validation_replay=conversion,
                        target_tensor_sha256=tensor_tree_hash(arrays['target']),
                        sample_ids_sha256=tensor_tree_hash(arrays['sample_ids']),
                        station_ids_sha256=tensor_tree_hash(arrays['station_ids']),
                        test_evaluated=args.split == 'test', engineering_only=args.engineering,
                        independent_confirmation=False)
                    require(evaluation_sources() == frozen['evaluation_source'],
                            'Evaluation source changed during inference')
                    metadata = commit_cache(directory, arrays, metadata, args.save_predictions, frozen, key)
                    del model, weights, arrays
                    gc.collect()
                    if torch.device(args.device).type == 'cuda':
                        torch.cuda.empty_cache()
                axes = tuple(metadata[name] for name in ('target_tensor_sha256',
                    'sample_ids_sha256', 'station_ids_sha256'))
                require(paired_axes is None or paired_axes == axes,
                        'Arms differ in target, sample order or station axis')
                paired_axes = axes
                result.append(metadata)
            del data
            gc.collect()
        render_report(args.output_dir, result, frozen)
    logger.info('PASE_COMPACT_MAIN_EVALUATION_COMPLETE: %s', args.output_dir/'report.md')


if __name__ == '__main__':
    main()
