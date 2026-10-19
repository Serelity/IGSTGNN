"""Freeze validation-selected models, then evaluate the published test slice once.

No optimizer, fitting, epoch option, calibration, or test-dependent selection.
Completed immutable arm caches are reused when this command is resumed.
"""
import argparse
from contextlib import ExitStack
import csv
import gc
import logging
import os
from pathlib import Path
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch

from experiments.chronological.original_igstgnn_source import get_original_modules, load_archive
from experiments.chronological.original_igstgnn_runtime import (
    RELEASE, build_model as build_original, device_batch as original_batch,
    rng_fingerprint, run_lock, tensor_tree_hash, write_json)
from experiments.original_baseline_gw16.evaluate import (
    archive_reference, native_summary, read_archive as read_original_validation)
from experiments.position_state_gw16.common import read_json, runtime, source_hashes, synchronize
from experiments.position_state_gw16.data import load_data, device_batch as candidate_batch
from experiments.position_state_gw16.evaluate import (
    read_archive as read_candidate_validation, selected_checkpoint)
from experiments.position_state_gw16.model import build_model as build_candidate
from experiments.position_state_gw16.objective import numpy_metrics
from experiments.position_state_gw16.train import validate_identity, validate_runtime, validate_artifacts
from .data import load_test_data, TEST_START, TEST_STOP
from .state import (require, snapshot_files, verify_files,
                    freeze_output, validate_arrays, cache_read, cache_write)

REPO = Path(__file__).resolve().parents[2]
CASES = ('ORIGINAL_IGSTGNN', 'STATE_EXCHANGE')
METRIC_CONTRACT = dict(
    primary='unchanged_author_native_metrics_each_lead_then_equal_average_over_12',
    native_null='min_of_entire_test_target_if_less_than_1_else_0; same_for_both_models_and_all_leads',
    native_accumulation='frozen_author_pytorch_float32',
    rmse_macro='mean_of_12_per_lead_RMSE_not_sqrt_of_pooled_MSE',
    mape_units='percent', horizons_reported=[3, 6, 12],
    common_mask=dict(target_minimum=.001, mape_minimum=1., accumulation='numpy_float64_pooled'),
    prediction_clipping=False, fitted_calibration=False, sample_reweighting=False)


def selection_guard(identity, candidate_metadata, original_metadata, role):
    """Validate full training and validation-only selection before opening test."""
    validate_identity(identity)
    require(identity['engineering_only'] is False and identity['config']['max_epochs'] == 100,
            'Final test requires the full, non-engineering 100-epoch GW16 protocol')
    require(candidate_metadata['identity'] == identity
            and candidate_metadata['arm'] == 'STATE_EXCHANGE'
            and candidate_metadata['engineering_only'] is False
            and candidate_metadata['split'] == 'val'
            and candidate_metadata['test_evaluated'] is False
            and candidate_metadata['validation_replay']['passed'] is True,
            'Expected a certified STATE_EXCHANGE validation archive')
    require(role['selection'] == 'best' and role['completed_training_epochs'] == 100
            and role['full_protocol_reached'] is True
            and candidate_metadata['checkpoint_role'] == role,
            'Only the completed run validation BEST can enter this final test')
    require(original_metadata['engineering_only'] is False
            and original_metadata['original_full_model'] is True
            and original_metadata['native_pos'] is False
            and original_metadata['test_evaluated'] is False
            and original_metadata['native_replay']['passed'] is True
            and original_metadata['source']['training_complete'] is True,
            'Expected the fully trained unmodified original validation replay')
    require(original_metadata['data_identity']['hashes'] == identity['data']['hashes'] == RELEASE,
            'The original and candidate must use the same certified author release')


def original_weights(metadata):
    """Recheck the actual legacy checkpoint/log, not a user-supplied completion flag."""
    source = metadata['source']
    require(source['kind'] == 'checkpoint_and_log',
            'This entry locks the certified legacy checkpoint-and-log baseline')
    archive = load_archive(checkpoint=Path(source['path']), train_log=Path(source['log_path']))
    completion = archive_reference(archive)
    require(completion['training_complete'] is True and all(
        completion[key] == source[key] for key in
        ('training_complete', 'completed_epoch', 'best_epoch', 'reference_level')),
        'Original log no longer proves the recorded complete training and BEST epoch')
    require(archive.config == metadata['training_config']
            and archive.metadata['parameter_count'] == metadata['registered_parameters'] == 455220
            and archive.metadata['counts'] == dict(train=3909, val=837, node_num=496)
            and archive.best_validation_mae == metadata['expected_validation_mae'],
            'Original checkpoint/log configuration differs from its certified validation replay')
    require(archive.provenance['checkpoint_sha256'] == source['checkpoint_sha256']
            and archive.provenance['training_log_sha256'] == source['training_log_sha256'],
            'Original source fingerprints differ from certified validation')
    require(tensor_tree_hash(archive.state_dict) == metadata['native_replay']['model_state_sha256'],
            'Original model tensors differ from the certified validation forward')
    return archive.state_dict, archive.config


def safe_output(output, inputs, data_dir):
    output = Path(output).resolve()
    for source in (*inputs, data_dir):
        source = Path(source).resolve()
        require(output != source and source not in output.parents and output not in source.parents,
                'Final-test output must be separate from training, validation and raw data')
    for parent in (output, *output.parents):
        require(not (parent/'identity.json').is_file(), 'Final-test output cannot be inside a training run')
    return output


def execution_files(identity, original_metadata):
    names = set(identity['execution_source']) | set(original_metadata['execution_source'])
    # Frozen imports authenticate their own source. Include every package module
    # used by the new entry, including indirect runtime and native metrics code.
    for package in ('experiments/final_test_gw16', 'experiments/original_baseline_gw16',
                    'experiments/chronological/original_igstgnn_558ab01'):
        names.update(p.relative_to(REPO).as_posix() for p in (REPO/package).glob('*.py'))
    names.update(('experiments/chronological/original_igstgnn_source.py',
                  'experiments/chronological/original_igstgnn_558ab01/manifest.json',
                  'experiments/chronological/original_igstgnn_checkpoint_compat.py',
                  'experiments/chronological/evaluate_original_igstgnn_modules.py'))
    return [REPO/name for name in sorted(names)]


def paired_scores(original, candidate, config, modules=None):
    validate_arrays(original)
    validate_arrays(candidate)
    for key in ('sample_ids', 'station_ids', 'target'):
        require(original[key].dtype == candidate[key].dtype
                and np.array_equal(original[key], candidate[key]),
                'Original/GW16 final-test axes or targets differ: '+key)
    rows = {}
    for case, arrays in zip(CASES, (original, candidate)):
        rows[case] = dict(author_metrics=native_summary(arrays['prediction'], original['target'], modules),
            common_mask_pooled=numpy_metrics(arrays['prediction'], original['target'], config))
    base, actual = (rows[case]['author_metrics']['horizon_macro'] for case in CASES)
    require(all(base[key] > 0 for key in ('mae', 'rmse', 'mape_percent')),
            'Original native metrics must be positive for relative improvements')
    gains = {key:100*(base[key]-actual[key])/base[key] for key in base}
    return dict(rows=rows, relative_gain_percent_vs_replayed_original=gains,
        mape_percentage_points_reduction=base['mape_percent']-actual['mape_percent'])


def validate_cached(arrays, metadata, frozen, case, data, modules):
    expected_hash = frozen['original_weights_sha256' if case == CASES[0]
                           else 'candidate_weights_sha256']
    audit = metadata['audit']
    require(audit['model_tensor_sha256'] == expected_hash
            and all(audit[key] is True for key in
                    ('rng_unchanged', 'registered_tensors_unchanged', 'raw_inputs_unchanged'))
            and audit['future_target_passed_to_forward'] is False
            and audit['optimizer_updates'] == 0 and audit['forward_batches'] == 18,
            'Cached final test has different weights or inference scope')
    material = metadata['data_identity']
    require(material['hashes'] == RELEASE
            and material['bounds_half_open'] == [TEST_START, TEST_STOP]
            and material['count'] == 839 and material['batch_size'] == 48
            and material['normalization'] == dict(mean=data.mean, std=data.std, refitted=False)
            and material['station_ids'] == frozen['test_protocol']['station_ids']
            and np.array_equal(arrays['station_ids'], data.station_ids),
            'Cached final-test data or station axis differs from the frozen author release')
    require(metadata['metric_contract'] == METRIC_CONTRACT
            and metadata['runtime'] == frozen['runtime']
            and metadata['author_metrics'] == native_summary(
                arrays['prediction'], arrays['target'], modules)
            and metadata['common_mask_pooled'] == numpy_metrics(
                arrays['prediction'], arrays['target'], METRIC_CONTRACT['common_mask']),
            'Final-test cached metrics or runtime changed')


def collect(case, model, loader, data, device, logger):
    """Neither forward receives a future target; certify no state/RNG/input update."""
    model.eval()
    model_before, random_before = tensor_tree_hash(model.state_dict()), rng_fingerprint()
    predictions, targets = [], []
    synchronize(device)
    tick = time.perf_counter()
    with torch.no_grad():
        for index, raw in enumerate(loader.get_iterator(), 1):
            raw_before = tensor_tree_hash(raw)
            if case == 'ORIGINAL_IGSTGNN':
                x, y, event, sensors = original_batch(raw, device)
                output = model(x, label=None, incident_data=event, sensor_data=sensors)
                prediction = data.scaler.inverse_transform(output)
                target = data.scaler.inverse_transform(y)
            else:
                x, target, event = candidate_batch(raw, data, device)
                prediction = model(x, event)['prediction']
            require(tensor_tree_hash(raw) == raw_before, 'Inference modified raw inputs: '+case)
            predictions.append(prediction.detach().cpu().numpy())
            targets.append(target.detach().cpu().numpy())
            if index in (1, loader.num_batch):
                logger.info('%s test batch=%d/%d', case, index, loader.num_batch)
    synchronize(device)
    seconds = time.perf_counter()-tick
    require(tensor_tree_hash(model.state_dict()) == model_before,
            'Inference modified registered model tensors: '+case)
    require(rng_fingerprint() == random_before, 'Evaluation consumed a random stream: '+case)
    arrays = dict(prediction=np.concatenate(predictions), target=np.concatenate(targets),
        sample_ids=np.arange(TEST_START, TEST_STOP, dtype=np.int64),
        station_ids=np.asarray(data.station_ids, dtype=np.int64))
    validate_arrays(arrays)
    return arrays, dict(model_tensor_sha256=model_before, rng_unchanged=True,
        registered_tensors_unchanged=True, raw_inputs_unchanged=True,
        future_target_passed_to_forward=False, optimizer_updates=0,
        forward_batches=loader.num_batch, full_collection_seconds=seconds,
        timing_scope='cold_loader_H2D_forward_D2H_and_input_hashes; not_a_speed_benchmark')


def render_report(directory, result):
    lines = ['# 原版 IGSTGNN 与 GW16：冻结模型后的配对测试', '',
        '839个作者顺序测试样本；两者目标、站点和样本轴完全相同。',
        'STATE_EXCHANGE只使用完整100轮训练后、验证集选出的BEST（第{}轮）。'.format(
            result['candidate_checkpoint_role']['epoch']),
        '主指标沿用原版：每步分别计算，再将12步等权平均。RMSE不是pooled RMSE。', '',
        '| 方法 | 12步平均 MAE | 12步平均 RMSE | 12步平均 MAPE % |', '|---|---:|---:|---:|']
    for case in CASES:
        row = result['rows'][case]['author_metrics']['horizon_macro']
        lines.append('| {} | {:.9f} | {:.9f} | {:.9f} |'.format(
            case, row['mae'], row['rmse'], row['mape_percent']))
    gains = result['relative_gain_percent_vs_replayed_original']
    lines += ['', '相对本次回放原版：MAE改善{:.4f}%，RMSE改善{:.4f}%，MAPE相对改善{:.4f}%。'.format(
        gains['mae'], gains['rmse'], gains['mape_percent']), '',
        '| 方法 | 时距 | MAE | RMSE | MAPE % |', '|---|---:|---:|---:|---:|']
    for case in CASES:
        for row in result['rows'][case]['author_metrics']['per_horizon']:
            if row['lead'] in (3, 6, 12):
                lines.append('| {} | {} | {:.9f} | {:.9f} | {:.9f} |'.format(
                    case, row['lead'], row['mae'], row['rmse'], row['mape_percent']))
    lines += ['', '论文报告的Contra Costa 12步平均为13.43 / 22.50 / 15.84%，仅作外部参考；',
        '本表正式提升率以相同发布样本上回放的原版权重为分母。全部12个时距见per_horizon.csv。',
        'common_mask_pooled为补充口径，见comparison.json，不与上表混算。', '',
        '这是单seed整套方法的保留测试结果；输入和训练配方也有差异，尚不能将收益独立归因于状态交换。',
        '发布窗口存在历史重叠，绝对时间/事件独立性和事故首次可用时刻未获独立认证。',
        '已知历史原版测试记录；当前候选在冻结前没有经本入口测试。没有多seed显著性结论。',
        '不按测试集更换BEST/LAST、追加训练预算、校准或挑选分组。',
        '再次运行同一命令只复用已完整认证的模型缓存，并可补齐中断时缺少的模型结果。', '']
    (directory/'report.md').write_text('\n'.join(lines), encoding='utf-8')
    with (directory/'per_horizon.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['case', 'lead', 'mae', 'rmse', 'mape_percent'])
        writer.writeheader()
        for case in CASES:
            for row in result['rows'][case]['author_metrics']['per_horizon']:
                writer.writerow(dict(case=case, **row))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('gw16-run', 'gw16-val-eval', 'original-eval', 'data-dir', 'output-dir'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--threads', type=int, default=3)
    args = parser.parse_args(argv)
    for key in ('gw16_run', 'gw16_val_eval', 'original_eval', 'data_dir', 'output_dir'):
        setattr(args, key, getattr(args, key).resolve())
    safe_output(args.output_dir, (args.gw16_run, args.gw16_val_eval, args.original_eval), args.data_dir)
    logger = logging.getLogger('final-test-gw16')
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    logger.addHandler(handler)
    test_started = False
    freeze_id = None
    try:
        with ExitStack() as locks:
            require((args.gw16_run/'active.lock').is_file(), 'Completed GW16 run lacks its original lock')
            locks.enter_context(run_lock(args.gw16_run))
            entry_files = [args.gw16_run/name for name in ('identity.json', 'runtime.json', 'STATE_EXCHANGE/last.pt')]
            entry_files += [root/name for root in (args.original_eval, args.gw16_val_eval)
                            for name in ('metadata.json', 'predictions.npz')]
            entry_before = snapshot_files(entry_files)
            identity = read_json(args.gw16_run/'identity.json')
            validate_identity(identity)
            require(source_hashes() == identity['execution_source'], 'GW16 training source changed')
            environment = runtime(args.device, args.threads, identity['seed'])
            validate_runtime(read_json(args.gw16_run/'runtime.json'), environment)
            # Readers recompute cached validation metrics; no validation forward.
            _, original_meta = read_original_validation(args.original_eval)
            _, candidate_meta = read_candidate_validation(args.gw16_val_eval)
            batches = identity['data']['loader_batching']['train_loader']['batches']
            weights, role, _, checkpoint = selected_checkpoint(
                args.gw16_run, identity, 'STATE_EXCHANGE', 'best', batches)
            selection_guard(identity, candidate_meta, original_meta, role)
            original_state, original_config = original_weights(original_meta)
            require(original_config['seed'] == identity['seed'], 'Paired test seed provenance differs')
            modules = get_original_modules()
            require(original_meta['source_fingerprint'] == modules.source_fingerprint,
                    'Original frozen author code differs from validated source')
            files = [args.gw16_run/name for name in
                     ('identity.json', 'runtime.json', 'config.json', 'partition.json',
                      'position.json', 'metric_contract.json', 'STATE_EXCHANGE/last.pt')]
            files += [checkpoint, *map(Path, original_meta['source']['source_files'])]
            files += [root/name for root in (args.original_eval, args.gw16_val_eval)
                      for name in ('metadata.json', 'predictions.npz')]
            files += execution_files(identity, original_meta)
            files += [args.data_dir/name for name in RELEASE]
            before = snapshot_files(files)
            verify_files(entry_before)
            require(all(before[str((args.data_dir/name).resolve())] == digest
                        for name, digest in RELEASE.items()), 'Raw data differs from certified release')
            frozen = dict(
                schema='final_test_gw16_method_v1', source_files=before, runtime=environment,
                candidate_training_identity_sha256=identity['identity_sha256'],
                candidate_checkpoint_role=role,
                candidate_weights_sha256=tensor_tree_hash(weights),
                original_source=original_meta['source'], original_config=original_config,
                original_weights_sha256=tensor_tree_hash(original_state),
                selection='GW16_validation_BEST_after_full_100_epochs; original_logged_validation_BEST',
                test_protocol=dict(bounds_half_open=[TEST_START, TEST_STOP], count=839,
                    nodes=496, history=12, horizon=12, batch_size=48,
                    station_ids=identity['data']['station_ids'], split='author_order_70_15_15'),
                metric_contract=METRIC_CONTRACT, no_updates=True,
                test_targets_inspected_before_freeze=False,
                published_object_array_contains_all_splits=True,
                historical_original_test_metrics_previously_known=True,
                development_validation_reused=True, strict_time_or_event_independent=False)
            # No test loader has been constructed or test target examined above.
            # A fixed sibling lock serializes first freeze creation and resumes.
            args.output_dir.parent.mkdir(parents=True, exist_ok=True)
            # run_lock needs a directory; hold a dedicated non-result lock folder.
            lock_dir = args.output_dir.parent/('.'+args.output_dir.name+'.locks')
            lock_dir.mkdir(exist_ok=True)
            locks.enter_context(run_lock(lock_dir))
            freeze_id = freeze_output(args.output_dir, frozen)
            logger.info('Frozen paired test identity=%s; STATE_EXCHANGE BEST epoch=%d', freeze_id, role['epoch'])
            data = load_data(args.data_dir, identity['config'], logger, engineering=False)
            require(data.identity == identity['data'], 'Frozen GW16 dataset/transforms/partition changed')
            validate_artifacts(args.gw16_run, identity, data)
            cached = {}
            for case in CASES:
                directory = args.output_dir/case
                if directory.exists():
                    arrays, metadata = cache_read(directory, freeze_id, case)
                    validate_cached(arrays, metadata, frozen, case, data, modules)
                    cached[case] = arrays
                    logger.info('%s immutable final-test cache reused; zero new model forwards', case)
            if len(cached) != len(CASES):
                test_started = True
                write_json(args.output_dir/'test_exposure.json', dict(freeze_id=freeze_id,
                    test_targets_may_be_inspected=True, new_model_selection_permitted=False))
                test_data = load_test_data(args.data_dir, original_config, data, modules, logger)
                for case in CASES:
                    if case in cached:
                        continue
                    verify_files(before)
                    if case == 'ORIGINAL_IGSTGNN':
                        model = build_original(test_data.native_data, modules, args.device)
                        state, loader, material = original_state, test_data.native_loader, test_data.native_data
                    else:
                        model = build_candidate(data, identity['config'], case, args.device)
                        model.set_stage('joint')
                        state, loader, material = weights, test_data.candidate_loader, data
                    model.load_state_dict(state, strict=True)
                    require(sum(parameter.numel() for parameter in model.parameters()) ==
                            (455220 if case == CASES[0] else role['registered_parameters']),
                            'Final-test registered parameter count differs from validation')
                    require(tensor_tree_hash(model.state_dict()) == tensor_tree_hash(state),
                            'Strict model load changed selected weights')
                    arrays, audit = collect(case, model, loader, material, args.device, logger)
                    require(np.array_equal(arrays['sample_ids'], test_data.sample_ids)
                            and np.array_equal(arrays['station_ids'], test_data.station_ids),
                            'Collected final-test sample/station axes changed')
                    if cached:
                        other = next(iter(cached.values()))
                        require(np.array_equal(arrays['target'], other['target']),
                                'The two frozen model loaders produced different test targets')
                    verify_files(before)
                    metadata = dict(audit=audit, data_identity=test_data.identity,
                        metric_contract=METRIC_CONTRACT, runtime=environment,
                        author_metrics=native_summary(arrays['prediction'], arrays['target'], modules),
                        common_mask_pooled=numpy_metrics(arrays['prediction'], arrays['target'], identity['config']))
                    cache_write(args.output_dir/case, freeze_id, case, arrays, metadata)
                    cached[case], saved = cache_read(args.output_dir/case, freeze_id, case)
                    validate_cached(cached[case], saved, frozen, case, data, modules)
                    del model, arrays
                    gc.collect()
                    if torch.device(args.device).type == 'cuda':
                        torch.cuda.empty_cache()
            result = paired_scores(cached[CASES[0]], cached[CASES[1]], identity['config'], modules)
            verify_files(before)
            result.update(schema='final_test_gw16_paired_test_v1', status='GW16_FINAL_TEST_COMPLETE',
                freeze_id=freeze_id, candidate_checkpoint_role=role,
                original_source=original_meta['source'], metric_contract=METRIC_CONTRACT,
                split='test', samples=839, test_evaluated=True, optimizer_updates=0,
                candidate_selection_used_test=False, original_and_candidate_recipes_identical=False,
                statistical_confirmation=False, isolated_mechanism_confirmed=False,
                strict_time_or_event_independent=False,
                paper_external_reference=dict(mae=13.43, rmse=22.50, mape_percent=15.84,
                    source='https://arxiv.org/html/2602.02528v2', role='reported_reference_not_paired_denominator'))
            write_json(args.output_dir/'comparison.json', result)
            render_report(args.output_dir, result)
            logger.info('GW16_FINAL_TEST_COMPLETE: %s', args.output_dir/'report.md')
            return result
    except BaseException as error:
        if freeze_id is not None:
            write_json(args.output_dir/'failure.json', dict(error_type=type(error).__name__,
                error=str(error), freeze_id=freeze_id, test_loader_started_this_invocation=test_started,
                preserve_completed_caches=True, new_model_selection_permitted=False))
        raise
    finally:
        logger.removeHandler(handler)
        handler.close()


if __name__ == '__main__':
    main()
