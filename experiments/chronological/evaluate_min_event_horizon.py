"""Read-only best validation replay and horizon-gate mechanism evaluation.

All interventions follow both complete best-checkpoint ON replays. Source E1
predictions are reused from the immutable epoch-zero reference, not retrained.
"""
import argparse
from contextlib import ExitStack
import csv
from datetime import datetime
import gc
import json
import logging
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import uuid

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.chronological import train_min_event_horizon as training
from experiments.chronological import evaluate_min_early_event as early_evaluation
from experiments.chronological import min_early_event_metrics as old_metrics
from experiments.chronological import probe_min_event_budget as probe
from experiments.chronological.smoke import sha256
from src.models.incident_spatial import require
from src.models.incident_spatial_residual import tensor_state_hash
from src.utils.igstgnn_published_events import EVENT_KEYS, encode_event
from src.utils.metrics import compute_all_metrics

SCHEMA = 'min_event_horizon_evaluation_v1'
CASES = ('CONTROL_ON', 'HORIZON_ON', 'HORIZON_OFF', 'HORIZON_GATE1',
         'HORIZON_SEMANTIC_UNK', 'HORIZON_LOCATION_SHUFFLE', 'SOURCE_EVENT_ON')
ALIASES = {'STATE_ON': 'CONTROL_ON', 'EVENT_ON': 'HORIZON_ON', 'EVENT_OFF': 'HORIZON_OFF',
           'EVENT_SEMANTIC_UNK': 'HORIZON_SEMANTIC_UNK', 'EVENT_LOCATION_SHUFFLE': 'HORIZON_LOCATION_SHUFFLE',
           'HORIZON_GATE1': 'HORIZON_GATE1', 'SOURCE_EVENT_ON': 'SOURCE_EVENT_ON'}
COMPARISONS = {
    'HORIZON_vs_CONTROL': ('STATE_ON', 'EVENT_ON', 'paired_warmstart_best_stage_checkpoint_comparison'),
    'report_path_dependency': ('EVENT_OFF', 'EVENT_ON', 'same_checkpoint_full_report_path_dependency'),
    'horizon_attenuation_effect': ('HORIZON_GATE1', 'EVENT_ON', 'same_checkpoint_gate_one_vs_learned_horizon_attenuation'),
    'semantic_UNK_sensitivity': ('EVENT_SEMANTIC_UNK', 'EVENT_ON', 'same_checkpoint_semantic_UNK_potential_OOD'),
    'location_shuffle_sensitivity': ('EVENT_LOCATION_SHUFFLE', 'EVENT_ON', 'same_checkpoint_full_station_shuffle_potential_OOD'),
    'HORIZON_vs_SOURCE': ('SOURCE_EVENT_ON', 'EVENT_ON', 'source_epoch_zero_vs_updated_candidate_not_a_paired_architecture_effect'),
    'CONTROL_vs_SOURCE': ('SOURCE_EVENT_ON', 'STATE_ON', 'source_epoch_zero_vs_continued_original_structure'),
}


def selected_metrics(saved, baseline):
    if saved['best_epoch'] == 0:
        return baseline['actual']
    row = saved['history'][saved['best_epoch']-1]
    require(row['validation_mae'] == saved['best_metric'], 'Selected stage checkpoint metric changed')
    return [row[key] for key in ('validation_mae', 'validation_mape', 'validation_rmse')]


def forward_case(model, batch, scaler, case, events_override=None):
    before = probe.input_digest(batch, events_override)
    try:
        device = next(model.parameters()).device
        x = torch.as_tensor(batch['x_data'], device=device, dtype=torch.float32)
        y = torch.as_tensor(batch['y_data'], device=device, dtype=torch.float32)
        event = training.event_tensors(batch['event_data'] if events_override is None else events_override, device)
        with torch.inference_mode():
            result = model(x, incident_data=event, report_enabled=case != 'HORIZON_OFF',
                semantic_enabled=case != 'HORIZON_SEMANTIC_UNK', horizon_enabled=case != 'HORIZON_GATE1',
                return_spatial_details=True)
            prediction = scaler.inverse_transform(result['prediction']).squeeze(-1).cpu().numpy().copy()
            target = scaler.inverse_transform(y).squeeze(-1).cpu().numpy().copy()
        require(np.isfinite(prediction).all() and np.isfinite(target).all(), 'Nonfinite horizon validation output')
        details = result['horizon']
        mechanism = early_evaluation.small_mechanisms(result['spatial'])
        mechanism['paired_core_calls'] = details['paired_core_calls']
        mechanism['response_measured_calls'] = int(details['response_measured'])
        for name in ('raw_delta', 'applied_delta'):
            value = details[name]
            if value is not None:
                mechanism[name+'_abs_sum'] = float(value.abs().sum().cpu())
                mechanism[name+'_signed_sum'] = float(value.sum().cpu())
                mechanism[name+'_abs_max'] = float(value.abs().max().cpu())
        return dict(prediction=prediction, target=target, mechanisms=mechanism)
    finally:
        require(probe.input_digest(batch, events_override) == before, 'Horizon evaluation changed input arrays')


def summarize(predictions, target, valid, regions, indices):
    aliases = {name: predictions[actual] for name, actual in ALIASES.items()}
    result = old_metrics.summarize(aliases, target, valid, regions, indices, comparisons=COMPARISONS)
    result['schema'] = 'min_event_horizon_metrics_v1'
    result['cases'] = {ALIASES[name]: row for name, row in result['cases'].items()}
    fixed = result['fixed_hard_windows']
    fixed['selector'] = 'CONTROL_ON'
    fixed['cases'] = {ALIASES[name]: row for name, row in fixed['cases'].items()}
    for row in result['comparisons'].values():
        row['control'], row['candidate'] = ALIASES[row['control']], ALIASES[row['candidate']]
    result['bias_reference'] = 'CONTROL_ON_for_all_comparisons'
    result['scope'] = 'exploratory_validation_checkpoint_sensitivity_not_incident_causality'
    result['forecast_scope'] = 'no_forecast_readout_interventions_in_this_stage'
    return result


def write_report(output, result, comparable, weights):
    fmt = lambda value: '不可用' if value is None else f'{value:.9f}'
    lines = ['# MIN EVENT 时距响应控制：配对暖启动评估', '',
        '工程检查：'+str(comparable['engineering_only'])+'；验证窗口：'+str(comparable['validation_rows'])+'。', '',
        ('本报告只覆盖少量工程检查窗口，用于验证训练、回放、干预和恢复流程，不构成正式预测效果或改进收益结论。'
         if comparable['engineering_only'] else '本报告覆盖本次源身份记录的完整验证集，仍属于用于模型设计和选模的探索性评估。'), '',
        'CONTROL保持原E1结构；HORIZON使用同一核心权重的ON/OFF差，以12个[0,1]标量调节事故路径响应。两组从同一源EVENT最佳权重暖启动，并对称重置优化器和调度器。', '',
        '本轮并非源实验精确续跑；原MIN没有重新训练。epoch0源模型纳入最佳选模候选，训练退化时可以保留源模型。', '',
        '| 状态 | MAE pooled | RMSE pooled | MAPE比例 |', '|---|---:|---:|---:|']
    for case, row in result['cases'].items():
        metric = row['metrics']['all_nodes']
        lines.append('| '+case+' | '+' | '.join(fmt(metric[key]) for key in ('mae_pooled', 'rmse_pooled', 'mape_pooled'))+' |')
    lines += ['', '| 比较 | MAE收益 horizon macro | 正收益 | 负收益 | 抵消比例 | 固定困难窗口收益 |', '|---|---:|---:|---:|---:|---:|']
    for name, row in result['comparisons'].items():
        metric = row['metrics']['all_nodes']
        lines.append('| '+name+' | '+' | '.join(fmt(metric[key]) for key in ('mae_gain_macro', 'gain_positive', 'gain_negative', 'cancellation_fraction'))+
                     ' | '+fmt(row['fixed_hard_window_gain'])+' |')
    lines += ['', '## 所选最佳检查点的12步权重', '', str(weights), '',
        '权重1表示原E1事故响应，权重0表示同一当前核心的事故路径OFF，不能解释为真实道路容量比例。',
        '完成阶段轮次：'+str(comparable['completed_epochs'])+'；所选阶段轮次：'+str(comparable['best_stage_epochs'])+'。',
        '源原始最佳轮次：'+str(comparable['source_best_epoch'])+'；完成预算是否一致：'+str(comparable['same_completed_training_budget'])+'。',
        'SOURCE_EVENT_ON复用训练开始前完整回放保存的源预测。HORIZON_GATE1衡量同一候选权重关闭时距衰减后的预测，不等同于独立继续训练的CONTROL。',
        '语义UNK和全站位置shuffle可能产生分布外输入；事故关联区域不等于真实事故影响，现有报告属性仍是conditional offline范围。',
        '固定困难窗口和偏差符号由CONTROL_ON统一定义。验证窗口可能重叠，预测点不视为独立事故重复。',
        '未计算的CONTROL/OFF/GATE1内部ON/OFF响应记为不可用，不把缺少测量写成零。',
        '未训练、未评价测试集、未重复12组forecast读出干预。完整数值见analysis.json、paired_predictions.npz、mechanisms.json、replay.json。', '']
    (output/'report.md').write_text('\n'.join(lines), encoding='utf-8')
    with (output/'horizons.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['case', 'horizon', 'mae', 'rmse', 'mape'])
        for case, row in result['cases'].items():
            metric = row['metrics']['all_nodes']
            for horizon in range(12):
                writer.writerow([case, horizon+1]+[metric[key][horizon] for key in ('per_horizon_mae', 'per_horizon_rmse', 'per_horizon_mape')])
    for filename, family in (('regions.csv', 'metrics'), ('strata.csv', 'strata')):
        with (output/filename).open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['comparison', 'group', 'valid_count', 'mae_gain_macro', 'global_gain_contribution', 'gain_positive', 'gain_negative'])
            for name, row in result['comparisons'].items():
                for group, metric in row[family].items():
                    writer.writerow([name, group]+[metric[key] for key in ('valid_count', 'mae_gain_macro', 'global_gain_contribution', 'gain_positive', 'gain_negative')])


def run_probe(models, data, saved, graph, roads, output, baseline, baseline_arrays, protected_paths):
    output = Path(output)
    require(not output.exists(), 'Use a fresh horizon evaluation directory')
    output.mkdir(parents=True)
    before = probe.snapshot_files(protected_paths)
    predictions, replay, mechanisms, preservation, calls = {}, {}, {}, {}, {}
    try:
        require(models.keys() == saved.keys() == {'CONTROL', 'HORIZON'}, 'Both CONTROL and HORIZON best models are required')
        histories = {arm: value['history'] for arm, value in saved.items()}
        common = min(len(value) for value in histories.values())
        require(all(len({histories[arm][epoch]['train_order_sha256'] for arm in histories}) == 1 for epoch in range(common)),
                'Paired common-prefix sample order differs')
        comparable = dict(completed_epochs={arm: value['completed_epoch'] for arm, value in saved.items()},
            engineering_only=baseline['engineering_only'], validation_rows=baseline['validation_rows'],
            global_updates={arm: value['global_updates'] for arm, value in saved.items()},
            best_stage_epochs={arm: value['best_epoch'] for arm, value in saved.items()},
            common_stage_epoch=common, source_best_epoch=baseline['source_best_epoch'],
            same_completed_training_budget=len({value['completed_epoch'] for value in saved.values()}) == 1
                and len({value['global_updates'] for value in saved.values()}) == 1,
            source_exact_resume=False, optimizer_and_scheduler_reset_at_stage_start=True, existing_MIN_retrained=False)
        target, valid, indices = baseline_arrays['target'], baseline_arrays['valid'], baseline_arrays['sample_indices']
        require(len(target) == data.loaders['val_loader'].size and np.array_equal(baseline_arrays['station_ids'], data.station_ids),
                'Saved source-reference sample/station axes differ')
        rows = [encode_event(sample, len(data.station_ids)) for sample in data.loaders['val_loader'].original_samples]
        events = {key: np.asarray([row[key] for row in rows], dtype=np.float32 if key == 'relation'
            else np.int64 if key in ('type', 'description') else bool) for key in EVENT_KEYS}
        shuffled, donors, shuffle_audit = early_evaluation.shuffled_locations(events, indices, 20261015)
        regions = probe.regions_for(events, graph, roads)
        model_hashes = {arm: tensor_state_hash(model.state_dict()) for arm, model in models.items()}
        with ExitStack() as stack:
            for arm, model in models.items():
                require(model_hashes[arm] == tensor_state_hash(saved[arm]['best_model_state']), 'Model is not saved selected best')
                stack.enter_context(probe.preserve_model(model.core, preservation, arm))
            # First gate: both independent full ON replays must pass.
            for arm, model in models.items():
                case, position, partitions = arm+'_ON', 0, []
                predictions[case] = np.empty_like(target)
                for batch in data.loaders['val_loader'].get_iterator():
                    row = forward_case(model, batch, data.scaler, case)
                    end = position+len(row['prediction'])
                    require(np.array_equal(row['target'], target[position:end]), 'Targets differ from immutable source reference')
                    predictions[case][position:end] = row['prediction']
                    probe.merge_mechanisms(mechanisms.setdefault(case, {}), row['mechanisms'])
                    calls[case] = calls.get(case, 0)+1
                    partitions.append(end-position)
                    position = end
                require(position == len(target) and partitions == baseline['batch_partition'], 'Original validation batch partition changed')
                actual = compute_all_metrics(torch.from_numpy(predictions[case]), torch.from_numpy(target), torch.tensor(baseline['mask_value']))
                expected = selected_metrics(saved[arm], baseline)
                require(np.allclose(actual, expected, atol=1e-5, rtol=1e-6), arm+' selected best ON replay failed')
                replay[arm] = dict(status='BEST_ON_REPLAY_PASS', best_stage_epoch=saved[arm]['best_epoch'],
                    actual=actual, expected=expected, mask_value=baseline['mask_value'], batch_partition=partitions, atol=1e-5, rtol=1e-6)
            position, model = 0, models['HORIZON']
            for batch in data.loaders['val_loader'].get_iterator():
                on = forward_case(model, batch, data.scaler, 'HORIZON_ON')
                end = position+len(on['prediction'])
                calls['HORIZON_ON'] += 1
                require(np.array_equal(on['prediction'], predictions['HORIZON_ON'][position:end]), 'Paired HORIZON ON differs from verified replay')
                for case in CASES[2:-1]:
                    changed_events = {key: value[position:end] for key, value in shuffled.items()} if case.endswith('SHUFFLE') else None
                    row = forward_case(model, batch, data.scaler, case, changed_events)
                    predictions.setdefault(case, np.empty_like(target))[position:end] = row['prediction']
                    probe.merge_mechanisms(mechanisms.setdefault(case, {}), row['mechanisms'])
                    calls[case] = calls.get(case, 0)+1
                    restored = forward_case(model, batch, data.scaler, 'HORIZON_ON')
                    calls['HORIZON_ON'] += 1
                    require(np.array_equal(restored['prediction'], on['prediction']), 'HORIZON ON restoration failed')
                position = end
            require(position == len(target), 'Incomplete horizon interventions')
        require(all(tensor_state_hash(model.state_dict()) == model_hashes[arm] for arm, model in models.items()),
                'Horizon wrapper or offsets changed during evaluation')
        predictions['SOURCE_EVENT_ON'] = baseline_arrays['prediction']
        result = summarize(predictions, target, valid, regions, indices)
        result['training_comparability'] = comparable
        result['comparisons']['HORIZON_vs_CONTROL']['scope'] = ('paired_warmstart_equal_completed_stage_budget_independent_best'
            if comparable['same_completed_training_budget'] else 'unequal_completed_stage_budget_exploratory_independent_best')
        weights = {arm: model.horizon_weights().detach().cpu().tolist() for arm, model in models.items()}
        region_keys = {name: 'region_'+str(i) for i, name in enumerate(regions)}
        np.savez_compressed(output/'paired_predictions.npz', **predictions, target=target, valid=valid,
            sample_indices=indices, station_ids=np.asarray(data.station_ids, np.int64), location_donor_station_indices=donors,
            **{region_keys[name]: mask for name, mask in regions.items()})
        training.legacy.write_json(output/'region_definitions.json', dict(masks=region_keys,
            direct='original_author_relation_support_not_certified_incident_effect',
            potential='weak_component_envelope_not_certified_physical_propagation'))
        for name, value in (('analysis.json', result), ('comparability.json', comparable), ('replay.json', replay),
                            ('mechanisms.json', dict(cases=mechanisms, selected_best_horizon_weights=weights,
                                response_scale='normalized_prediction_difference', location_shuffle=shuffle_audit,
                                unmeasured_response='unavailable_not_zero'))):
            training.legacy.write_json(output/name, value)
        write_report(output, result, comparable, weights)
        files = probe.require_preserved(before, protected_paths)
        training.legacy.write_json(output/'preservation.json', dict(files=files, models=preservation, wrapper_offsets_preserved=True))
        summary = dict(schema=SCHEMA, status='MIN_EVENT_HORIZON_EVALUATION_PASS', validation_rows=len(target),
            engineering_only=baseline['engineering_only'], best_stage_epochs=comparable['best_stage_epochs'],
            source_best_epoch=baseline['source_best_epoch'], cases=list(predictions), selected_best_horizon_weights=weights,
            all_nodes={name: row['metrics']['all_nodes'] for name, row in result['cases'].items()},
            comparisons={name: row['metrics']['all_nodes'] for name, row in result['comparisons'].items()},
            replay_passed=True, on_exactly_restored=True, training_comparability=comparable,
            source_artifacts_preserved=True, new_inference_calls=calls, optimizer_updates=0, test_evaluated=False,
            source_scope='conditional_offline_validation_warmstart_checkpoint_sensitivity', output_dir=str(output))
        training.legacy.write_json(output/'summary.json', summary)
        return summary
    except BaseException as error:
        training.legacy.write_json(output/'failure.json', dict(status='MIN_EVENT_HORIZON_EVALUATION_FAILED',
            error_type=type(error).__name__, error=str(error), new_inference_calls=calls, optimizer_updates=0, test_evaluated=False))
        after = probe.snapshot_files(protected_paths)
        training.legacy.write_json(output/'preservation.json', dict(source_artifacts_preserved=before == after, before=before, after=after,
            models=preservation, evaluation_completed=False))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--source-run', type=Path, default=training.DEFAULT_SOURCE)
    parser.add_argument('--data-dir', type=Path, default=training.paper.default_data())
    parser.add_argument('--graph-dir', type=Path, default=training.HERE/'internal_spatial_graph_is11')
    parser.add_argument('--protocol', type=Path, default=training.PROTOCOL)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--threads', type=int)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args(argv)
    for name in ('run_dir', 'source_run', 'data_dir', 'graph_dir', 'protocol'):
        setattr(args, name, getattr(args, name).resolve())
    identity = training.legacy.read_json(args.run_dir/'identity.json')
    require(identity.get('schema') == training.SCHEMA, 'Expected a horizon warmstart run')
    require(args.check is identity['engineering_check'], 'Engineering marker differs from source run')
    args.threads = identity['threads'] if args.threads is None else args.threads
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    configuration = dict(vars(args), arms=list(identity['arm_modes']), batch_size=identity['protocol']['batch_size'],
                         seed=identity['seed'], epochs=1, only=None, worker=None, evaluate_after=False)
    setup = SimpleNamespace(**configuration)
    logger = logging.getLogger('min_event_horizon_evaluation')
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler(sys.stdout))
    with training.legacy.run_lock(args.run_dir), training.readonly_source_lock(args.source_run):
        protocol, data, graph, initialization, current, source = training.prepare(setup, logger)
        require(current == identity, 'Warmstart training identity changed')
        baseline = training.verify_source_reference(setup, identity)
        with np.load(args.run_dir/'source_reference.npz', allow_pickle=False) as archive:
            baseline_arrays = {name: archive[name] for name in archive.files}
        _, roads, _ = training.legacy.load_graph(args.graph_dir, data.station_ids, protocol['graph_sha256'])
        saved, models = {}, {}
        for arm in ('CONTROL', 'HORIZON'):
            path = args.run_dir/arm/'last.pt'
            require(path.is_file(), 'Both completed CONTROL and HORIZON arms are required')
            saved[arm] = torch.load(path, map_location='cpu', weights_only=False)
            training.validate_saved(saved[arm], identity, arm)
            require(training.legacy.read_json(args.run_dir/arm/'history.json') == saved[arm]['history'], 'History sidecar differs from committed checkpoint')
            companion = training.legacy.read_json(args.run_dir/arm/'summary.json')
            require(all(companion.get(key) == value for key, value in dict(arm=arm, mode=training.ARMS[arm],
                completed_epoch=saved[arm]['completed_epoch'], best_epoch=saved[arm]['best_epoch'], best_metric=saved[arm]['best_metric']).items()),
                'Summary sidecar differs from committed checkpoint')
            models[arm], _ = training.build_arm(setup, arm, len(data.station_ids), graph, torch.device(args.device), source)
            models[arm].load_state_dict(saved[arm]['best_model_state'], strict=True)
            models[arm].eval()
        paths = {'run/'+name: args.run_dir/name for name in ('identity.json', 'initialization.json', 'source_reference.json', 'source_reference.npz')}
        paths.update({'source/'+name: args.source_run/name for name in training.SOURCE_FILES})
        for arm in saved:
            paths.update({arm+'/'+name: args.run_dir/arm/name for name in ('last.pt', 'history.json', 'summary.json')})
        for key, directory in (('source_sha256', REPO), ('package', args.data_dir), ('graph_inputs_sha256', args.graph_dir)):
            paths.update({key+'/'+name: directory/name for name in identity[key]})
        paths['evaluation_source'] = Path(__file__).resolve()
        paths['protocol'] = args.protocol
        args.output_dir = args.output_dir.resolve() if args.output_dir else args.run_dir/('event_horizon_evaluation_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6])
        require(args.output_dir != args.run_dir and args.output_dir not in args.run_dir.parents
                and args.output_dir != args.source_run and args.source_run not in args.output_dir.parents,
                'Evaluation output must be a fresh directory outside the original source run')
        require(all(args.output_dir != args.run_dir/arm and args.run_dir/arm not in args.output_dir.parents
                    for arm in training.ARMS), 'Evaluation output must not be inside a checkpoint directory')
        print('Evaluation directory: '+str(args.output_dir), flush=True)
        result = run_probe(models, data, saved, graph, roads, args.output_dir, baseline, baseline_arrays, paths)
        training.legacy.write_json(args.output_dir/'provenance.json', dict(training_identity=identity,
            evaluation_source_sha256={str(Path(__file__).relative_to(REPO)): sha256(__file__)},
            source_reference_reused=True, existing_MIN_retrained=False, test_evaluated=False))
        print(json.dumps({key: result[key] for key in ('status', 'validation_rows', 'best_stage_epochs', 'selected_best_horizon_weights',
                                                      'optimizer_updates', 'output_dir')}, indent=2, allow_nan=False), flush=True)
    gc.collect()
    return result


if __name__ == '__main__':
    main()
