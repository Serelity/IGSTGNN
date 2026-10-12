"""Same-checkpoint TIID x D factorial screening of B3-R incident paths.

All four states use the original validation batch partition and unchanged
weights, report encoding, P, road graph and exchange operator. A saved full-ON
replay must pass before any path is disabled. These interventions measure
checkpoint dependence, not the performance of a retrained smaller model.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime
import gc
import json
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
from experiments.chronological import probe_b3_r_path as previous
from experiments.chronological import train_b3_r as base
from experiments.chronological import train_b3_r_strength as strength
from experiments.chronological import train_internal_spatial as legacy
from experiments.chronological.b3_r_module_metrics import summarize_factorial
from experiments.chronological.smoke import device_batch, sha256, verify_package
from experiments.chronological.train import batches, validate_protocol
from src.models.incident_spatial import SpatialFeatureExchange, require
from src.models.incident_spatial_residual import ReportCapacityResidual, tensor_state_hash
from src.models.incident_spatial_strength import ReportCapacityStrength
from src.utils.chronological import ChronologicalDataset

SCHEMA = 'b3r_tiid_d_factorial_probe_v1'
CASES = {
    'TIID_ON_D_ON': (True, True),
    'TIID_ON_D_OFF': (True, False),
    'TIID_OFF_D_ON': (False, True),
    'TIID_OFF_D_OFF': (False, False),
}
ON = 'TIID_ON_D_ON'
SUPPORTED_ARMS = ('B3R', 'B3R_HALF', 'B3R_GATE')
CONTEXT_FIELDS = ('base_logit', 'base_node_capacity', 'base_edge_capacity',
                  'node_capacity', 'capacity', 'boundary', 'report_mask',
                  'effective_logit_delta')


def validate_model(model):
    require(not any(module.training for module in model.modules()), 'Probe requires eval mode')
    require(all(parameter.grad is None for parameter in model.parameters()), 'Probe found parameter gradients')
    require(type(model.report_spatial_state) in (ReportCapacityResidual, ReportCapacityStrength)
            and model.report_spatial_state.report_path_enabled, 'Expected a trained D-ON model; NO_D cannot be enabled post hoc')
    require(getattr(model, '_internal_spatial_mode', None) == 'state_restricted'
            and model.icsf_module.state_spatial_mode and model._incident_routing == 'none'
            and model._time_response_mode == 'fixed', 'Unsupported incident/spatial backbone')
    require(not any(hasattr(model.icsf_module, name) for name in ('q_proj', 'v_proj', 'icsf_fusion_mlp')),
            'Original ICSF spatial injection is unexpectedly active')
    expected = {f'layers.{i}.dif_layer.localized_st_conv.incident_exchange' for i in range(5)}
    found = {name for name, module in model.named_modules() if isinstance(module, SpatialFeatureExchange)}
    require(found == expected, 'Incomplete or misplaced five-layer exchange coverage')
    require(all(model.get_submodule(name).restricted and model.get_submodule(name).steps == 2
                and model.get_submodule(name).eta == .5 for name in found), 'Wrong exchange configuration')


@contextmanager
def execution_counts(model):
    """Passive local hooks, always removed, including when forward fails."""
    counts, handles = {'D': 0, 'TIID': 0}, []
    def count(name):
        def hook(module, args, output):
            counts[name] += 1
        return hook
    try:
        handles.append(model.report_spatial_state.D.register_forward_hook(count('D')))
        handles.append(model.tiid_module.register_forward_hook(count('TIID')))
        yield counts
    finally:
        for handle in handles:
            handle.remove()


def forward(model, batch, scaler, case):
    require(case in CASES, 'Unknown factorial case')
    require(not any(module.training for module in model.modules()), 'Probe requires eval mode')
    require(all(parameter.grad is None for parameter in model.parameters()), 'Probe found parameter gradients')
    tiid, d = CASES[case]
    with execution_counts(model) as calls, torch.inference_mode():
        result = model(batch['x'], incident_data=batch['incident'], spatial_report_enabled=d,
                       tiid_report_enabled=tiid, return_spatial_details=True)
        prediction = result['prediction']*scaler['std']+scaler['mean']
    require(calls == {'D': int(d), 'TIID': int(tiid)}, 'D/TIID execution coverage mismatch')
    context = result['spatial']
    require(context['report_path_executed'] is d, 'D switch did not take effect')
    if not d:
        require(context['report_logit_delta'] is None and not bool(context['effective_logit_delta'].any())
                and torch.equal(context['node_capacity'], context['base_node_capacity']), 'OFF did not skip the whole D path')
    require(bool(torch.isfinite(prediction).all()), 'Non-finite prediction')
    return prediction.detach().cpu().numpy().copy(), context


def check_contexts(contexts, restored):
    full, d_off = contexts[ON], contexts['TIID_ON_D_OFF']
    for context in contexts.values():
        for name in ('base_logit', 'base_node_capacity', 'base_edge_capacity', 'boundary', 'report_mask'):
            require(torch.equal(full[name], context[name]), 'Intervention changed P/boundary/support: '+name)
    for first, second in ((ON, 'TIID_OFF_D_ON'), ('TIID_ON_D_OFF', 'TIID_OFF_D_OFF')):
        for name in CONTEXT_FIELDS:
            require(torch.equal(contexts[first][name], contexts[second][name]), 'TIID switch changed spatial context: '+name)
    previous.capacity_changes(full, d_off, previous.ChangeTotals())
    for name in CONTEXT_FIELDS:
        require(torch.equal(full[name], restored[name]), 'Final ON context restoration failed: '+name)


def check_prediction_support(predictions, associated):
    require(associated.dtype == bool and associated.ndim == 2, 'Invalid report support axes')
    outside = np.broadcast_to(~associated[:, None, :, None], predictions[ON].shape)
    for first, second in ((ON, 'TIID_OFF_D_ON'), ('TIID_ON_D_OFF', 'TIID_OFF_D_OFF')):
        require(np.array_equal(predictions[first][outside], predictions[second][outside]),
                'TIID intervention changed predictions outside direct report support')
    no_report = ~associated.any(1)
    for prediction in predictions.values():
        require(np.array_equal(prediction[no_report], predictions[ON][no_report]),
                'A report path intervention changed a no-report sample')


def regions_for(associated, graph, roads):
    # Reuse the trainer's operation and group semantics, with no second BLAS runtime.
    with torch.inference_mode():
        potential = (torch.from_numpy(associated).float() @ legacy.component_reach(graph).cpu() > 0).numpy()
    operator = graph.operator_mask.cpu().numpy()
    regions = dict(all_nodes=np.ones_like(associated), report_associated=associated,
                   candidate_potential=potential, potential_only=potential & ~associated,
                   outside_operator=np.broadcast_to(~operator, associated.shape),
                   direct_outside_operator=associated & ~operator[None],
                   no_report_samples=np.broadcast_to(~associated.any(1)[:, None], associated.shape))
    for name, nodes in roads.items():
        mask = np.zeros_like(associated)
        mask[:, nodes] = True
        regions['road:'+name] = mask
    return regions


def run_probe(model, dataset, indices, batch_size, device, graph, roads, saved_windows, expected_mae, output):
    validate_model(model)
    require(indices and list(indices) == [int(row['sample_index']) for row in dataset.rows[:len(indices)]],
            'Probe must preserve the original validation manifest order')
    require(isinstance(batch_size, int) and batch_size > 0, 'Invalid original batch size')
    state_before = tensor_state_hash(model.state_dict())
    switch_before = model.report_spatial_state.report_path_enabled
    parts, targets, valids, associations = [], [], [], []
    position = 0
    for raw in batches(dataset, indices, batch_size):
        batch = device_batch(raw, device)
        input_hash = previous.tree_hash(batch)
        prediction, _ = forward(model, batch, dataset.scaler, ON)
        require(input_hash == previous.tree_hash(batch), 'Replay changed inputs')
        parts.append(prediction)
        targets.append(batch['y_flow'].cpu().numpy().copy())
        valids.append(batch['y_valid'].cpu().numpy().copy())
        associations.append((batch['incident']['distances'].abs().sum(-1) > 0).cpu().numpy())
        position += len(prediction)
        print(f'Verified full ON replay: {position}/{len(indices)} rows', flush=True)
    on, target, valid, associated = [np.concatenate(values) for values in (parts, targets, valids, associations)]
    del parts, targets, valids, associations, batch, raw
    t0 = np.asarray([row['t0'] for row in dataset.rows[:len(indices)]])
    replay = previous.verify_saved_windows(on, target, valid, np.asarray(indices), t0, saved_windows, expected_mae)
    require(tensor_state_hash(model.state_dict()) == state_before, 'Replay changed model state; OFF blocked')
    legacy.write_json(output/'replay.json', replay)
    print('Full ON replay passed; permitting TIID x D interventions', flush=True)

    predictions = {case: on if case == ON else np.empty_like(on) for case in CASES}
    diagnostics = {case: strength.GateDiagnostics() for case in CASES}
    capacity_changes = previous.ChangeTotals()
    partition, position = [], 0
    for raw in batches(dataset, indices, batch_size):
        batch = device_batch(raw, device)
        input_hash = previous.tree_hash(batch)
        paired_on, on_context = forward(model, batch, dataset.scaler, ON)
        end = position+len(paired_on)
        require(np.array_equal(paired_on, on[position:end]), 'Paired ON differs from verified replay')
        contexts = {ON: on_context}
        diagnostics[ON].update(on_context)
        for case in CASES:
            if case == ON:
                continue
            prediction, contexts[case] = forward(model, batch, dataset.scaler, case)
            predictions[case][position:end] = prediction
            diagnostics[case].update(contexts[case])
        restored_on, restored_context = forward(model, batch, dataset.scaler, ON)
        require(np.array_equal(paired_on, restored_on), 'Final ON prediction restoration failed')
        check_contexts(contexts, restored_context)
        check_prediction_support({case: value[position:end] for case, value in predictions.items()}, associated[position:end])
        previous.capacity_changes(on_context, contexts['TIID_ON_D_OFF'], capacity_changes)
        require(input_hash == previous.tree_hash(batch), 'Paired inference changed inputs')
        require(tensor_state_hash(model.state_dict()) == state_before
                and model.report_spatial_state.report_path_enabled is switch_before
                and all(parameter.grad is None for parameter in model.parameters()), 'Intervention changed model state/switch/gradients')
        partition.append(end-position)
        position = end
        print(f'Paired four states + ON restoration: {position}/{len(indices)} rows', flush=True)
        del contexts, on_context, restored_context, batch, raw, paired_on, restored_on, prediction
        gc.collect()
    require(position == len(indices), 'Incomplete factorial validation')
    masks = regions_for(associated, graph, roads)
    analysis = summarize_factorial({name: array[..., 0] for name, array in predictions.items()}, target[..., 0],
                                   masks, valid=valid[..., 0], sample_indices=np.asarray(indices))
    legacy.write_json(output/'analysis.json', analysis)
    legacy.write_json(output/'capacity_changes.json', capacity_changes.result())
    legacy.write_json(output/'mechanisms.json', {case: value.result() for case, value in diagnostics.items()})
    np.savez_compressed(output/'paired_predictions.npz', **predictions, target=target, valid=valid,
                        sample_indices=np.asarray(indices), t0=t0, station_ids=np.asarray(dataset.station_ids),
                        **{'region__'+name: value for name, value in masks.items()})
    return dict(replay=replay, analysis=analysis, batch_partition=partition, inference_calls=6*len(partition),
                model_state_sha256=state_before, final_on_exactly_restored=True, inputs_preserved=True,
                inference_only=True, training_updates=0, test_read=False,
                original_graph_convolution_enabled=True, exchange_enabled_in_all_four_states=True,
                group_scope='candidate_potential_is_a_component_envelope; outside_operator_is_not_no_incident; groups_overlap',
                D_scope='whole_history_and_report_conditioned_path; not_pure_report_information',
                all_off_scope='TIID_and_D_off; shared_encoding_and_backbone_P_exchange_retained')


def select_best(checkpoint, identity, arm, history, summary):
    require(checkpoint.get('identity') == identity and checkpoint.get('arm') == arm, 'Checkpoint identity/arm mismatch')
    require(history and checkpoint['history'] == history and len(history) == checkpoint['completed_epoch']
            and [row['epoch'] for row in history] == list(range(1, len(history)+1)), 'Checkpoint/history mismatch')
    metrics = [row['validation_mae_macro'] for row in history]
    require(np.isfinite(metrics).all(), 'Non-finite selection metric')
    best = int(np.argmin(metrics))+1
    require(history[-1]['best_epoch'] == best and checkpoint['best_metric'] == metrics[best-1]
            and checkpoint['best_model_state'] is not None, 'Best checkpoint selection mismatch')
    require(summary['arm'] == arm and summary['completed_epoch'] == len(history)
            and summary['best_epoch'] == best and summary['best_metric'] == metrics[best-1], 'Summary/checkpoint mismatch')
    return best, metrics[best-1]


def source_paths():
    return [Path(__file__), Path(previous.__file__), REPO/'experiments/chronological/b3_r_module_metrics.py',
            Path(strength.__file__), REPO/'experiments/chronological/model_evaluation_metrics.py']


def execute(args, output):
    identity = legacy.read_json(args.run_dir/'identity.json')
    require(identity.get('schema') in ('b3r_run_v1', strength.SCHEMA) and identity.get('tiid_report_enabled') is True
            and identity.get('test_read') is False, 'Expected a supported B3-R development run')
    trainer = strength if identity['schema'] == strength.SCHEMA else base
    require(args.arm in identity['arm_modes'] and args.arm in SUPPORTED_ARMS, 'Selected arm was not trained D-ON in this run')
    protocol_path = trainer.HERE/('internal_spatial_b3_r_strength_g11.json' if trainer is strength else 'internal_spatial_b3_r_v11.json')
    require(trainer.source_fingerprints() == identity['source_sha256'] and sha256(protocol_path) == identity['protocol_sha256'],
            'Frozen training source/protocol fingerprint changed')
    protocol = legacy.read_json(protocol_path)
    trainer.validate_architecture_protocol(protocol)
    require(protocol == identity['protocol'], 'Recorded protocol differs')
    threads = identity['threads'] if args.threads is None else args.threads
    require(threads == identity['threads'], 'Use the original inference thread count')
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    require(device.type in ('cpu', 'cuda') and (device.type != 'cuda' or torch.cuda.is_available()), 'Requested device unavailable')
    for filename, key in (('summary.json', 'files'), ('context_manifest.json', 'outputs')):
        names = legacy.read_json(args.data_dir/filename)[key]
        require(all(Path(name).name == name and not name.lower().startswith('test') for name in names),
                'Manifest attempts to read a non-development path')
    hashes = verify_package(args.data_dir)
    train, val = ChronologicalDataset(args.data_dir, 'train'), ChronologicalDataset(args.data_dir, 'val')
    validate_protocol(protocol, hashes, len(train), len(val), len(train.station_ids))
    graph, roads, graph_info = legacy.load_graph(args.graph_dir, train.station_ids, protocol['graph_sha256'])
    original_args = SimpleNamespace(**dict(vars(args), seed=identity['seed'], check=identity['engineering_check'],
                                          arms=list(identity['arm_modes']), protocol=protocol_path, threads=threads))
    initialization = trainer.preflight(original_args, train, graph)
    require(trainer.make_identity(original_args, protocol, hashes, graph_info, initialization) == identity,
            'Reconstructed identity differs (data, graph, runtime, batch or initialization)')
    recorded = legacy.read_json(args.run_dir/'initialization.json')
    require((recorded == initialization if trainer is strength else
             base.initialization_identity(recorded) == base.initialization_identity(initialization)), 'Initialization certificate differs')
    del train
    paths = [args.run_dir/name for name in ('identity.json', 'initialization.json', 'initial_environment.json',
                                           args.arm+'/last.pt', args.arm+'/history.json', args.arm+'/summary.json')]
    before = {str(path.relative_to(args.run_dir)): sha256(path) for path in paths}
    probe_sources = {str(path.relative_to(REPO)): sha256(path) for path in source_paths()}
    checkpoint = torch.load(args.run_dir/args.arm/'last.pt', map_location='cpu', weights_only=False)
    history = legacy.read_json(args.run_dir/args.arm/'history.json')
    best, expected_mae = select_best(checkpoint, identity, args.arm, history, legacy.read_json(args.run_dir/args.arm/'summary.json'))
    windows_path = args.run_dir/args.arm/f'validation_windows_epoch_{best:03d}.npz'
    paths.append(windows_path)
    before[str(windows_path.relative_to(args.run_dir))] = sha256(windows_path)
    with np.load(windows_path, allow_pickle=False) as archive:
        saved_windows = {key: archive[key].copy() for key in ('sample_indices', 't0', 'absolute', 'counts')}
    model = trainer.build_arm(original_args, args.arm, len(val.station_ids), graph, device).eval()
    model.load_state_dict(checkpoint['best_model_state'], strict=True)
    best_hash = tensor_state_hash(checkpoint['best_model_state'])
    completed = checkpoint['completed_epoch']
    del checkpoint, history
    gc.collect()
    legacy.write_json(output/'provenance.json', dict(schema=SCHEMA, arm=args.arm, best_epoch=best,
                      completed_training_epoch=completed, run_dir=str(args.run_dir), data_dir=str(args.data_dir),
                      graph_dir=str(args.graph_dir), weights=args.arm+'/last.pt:best_model_state',
                      best_model_state_sha256=best_hash, saved_files_sha256=before, identity=identity,
                      probe_source_sha256=probe_sources, environment=legacy.environment(device),
                      original_environment=legacy.read_json(args.run_dir/'initial_environment.json'), test_read=False,
                      cases={name: dict(tiid_enabled=t, D_enabled=d) for name, (t, d) in CASES.items()}))
    indices = [int(row['sample_index']) for row in val.rows[:4 if identity['engineering_check'] else len(val)]]
    print(f'Verifying {args.arm} best epoch {best}, {len(indices)} rows, original batch {identity["batch_size"]}', flush=True)
    result = run_probe(model, val, indices, identity['batch_size'], device, graph, roads, saved_windows, expected_mae, output)
    after = {str(path.relative_to(args.run_dir)): sha256(path) for path in paths}
    graph_after = {name: sha256(args.graph_dir/name) for name in identity['graph_inputs_sha256']}
    source_preserved = (trainer.source_fingerprints() == identity['source_sha256']
                        and sha256(protocol_path) == identity['protocol_sha256']
                        and {str(path.relative_to(REPO)): sha256(path) for path in source_paths()} == probe_sources)
    data_preserved = verify_package(args.data_dir) == hashes
    legacy.write_json(output/'preservation.json', dict(before=before, after=after, source_preserved=source_preserved,
                      data_preserved=data_preserved, graph_before=identity['graph_inputs_sha256'], graph_after=graph_after))
    require(before == after and source_preserved and data_preserved and graph_after == identity['graph_inputs_sha256']
            and tensor_state_hash(model.state_dict()) == best_hash, 'Source/input/model artifact changed; completion blocked')
    return dict(status='B3R_MODULE_FACTORIAL_PROBE_PASS', schema=SCHEMA, arm=args.arm, best_epoch=best,
                completed_training_epoch=completed, source_artifacts_preserved=True,
                scope='engineering_only' if identity['engineering_check'] else 'conditional_offline_checkpoint_sensitivity',
                predictive_gain_claim=False, retrained_redundancy_claim=False, accident_only_attribution=False, **result)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--arm', choices=SUPPORTED_ARMS, default='B3R')
    parser.add_argument('--data-dir', type=Path, default=legacy.default_data())
    parser.add_argument('--graph-dir', type=Path, default=base.HERE/'internal_spatial_graph_is11')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--threads', type=int, help='Default: original run thread count; changes are rejected')
    args = parser.parse_args(argv)
    require(args.threads is None or args.threads >= 1, 'Threads must be positive')
    for name in ('run_dir', 'data_dir', 'graph_dir'):
        setattr(args, name, getattr(args, name).resolve())
    return args


def main(argv=None):
    args = parse_args(argv)
    require(args.run_dir.is_dir(), 'Run directory does not exist')
    output = args.run_dir/('module_factorial_'+args.arm+'_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6])
    output.mkdir()
    print('Probe directory: '+str(output), flush=True)
    try:
        with legacy.run_lock(args.run_dir):
            result = execute(args, output)
        write_report(output, result)
        legacy.write_json(output/'summary.json', result)
        analysis = result['analysis']
        print(json.dumps(dict(status=result['status'], arm=result['arm'], best_epoch=result['best_epoch'],
                              cases={case: dict(mae=value['metrics']['all_nodes']['mae_macro'],
                                                rmse=value['metrics']['all_nodes']['rmse_macro'],
                                                window_p95=value['metrics']['all_nodes']['window_mae']['p95'],
                                                worst_5_percent=value['metrics']['all_nodes']['window_mae']['top_5_percent']['mean'])
                                     for case, value in analysis['cases'].items()},
                              effects={name: {key: value['metrics']['all_nodes'][key] for key in
                                               ('baseline_minus_candidate_mae', 'gain_positive', 'gain_negative',
                                                'cancellation_fraction', 'prediction_absolute_change_macro')}
                                       for name, value in analysis['comparisons'].items()},
                              interaction=analysis['interaction']['metrics']['all_nodes'],
                              output=str(output/'summary.json')), indent=2), flush=True)
    except BaseException as error:
        legacy.write_json(output/'failure.json', dict(status='B3R_MODULE_FACTORIAL_PROBE_FAILED',
                          error_type=type(error).__name__, message=str(error), completed=False, test_read=False))
        raise


def write_report(output, result):
    # Expanded human-readable results are written after all preservation checks.
    lines = ['# B3-R TIID × D 模块必要性筛查', '',
             f"模型：{result['arm']}；最佳轮次：{result['best_epoch']}；范围：{result['scope']}。", '',
             '四种状态使用同一检查点与原验证批次。全开回放通过，逐批恢复全开预测完全一致。', '',
             '本实验衡量已训练权重对路径的依赖。不能据此直接断言删除后重新训练仍不退化。',
             'D同时处理历史与报告；D关闭不等于仅移除事故内容。两者全关仍保留P、共享编码及原图卷积和受限交换。', '',
             '正收益表示候选误差更低。交互项为 M10 + M01 − M11 − M00（M为MAE），正数表示两路联合降低误差的幅度超过各自单独作用之和。',
             '交互与正负收益分解是检查点上的描述性结果，不构成真实事故因果或物理容量识别证据。',
             '窗口可能重叠；区域也可能重叠，不能视为独立重复或直接相加。', '']
    analysis = result['analysis']
    def number(value):
        return '不可用' if value is None else f'{value:.9f}'
    lines += ['## 四组合全网结果', '', '| TIID | D | MAE | RMSE macro | 窗口P95 | 各自最差5%窗口均值 |',
              '|---|---|---:|---:|---:|---:|']
    for case, (tiid, d) in CASES.items():
        row = analysis['cases'][case]['metrics']['all_nodes']
        values = (row['mae_macro'], row['rmse_macro'], row['window_mae']['p95'], row['window_mae']['top_5_percent']['mean'])
        lines.append('| '+('开' if tiid else '关')+' | '+('开' if d else '关')+' | '+' | '.join(map(number, values))+' |')
    labels = dict(D_given_TIID_ON='TIID开启时D收益', D_given_TIID_OFF='TIID关闭时D收益',
                  TIID_given_D_ON='D开启时TIID收益', TIID_given_D_OFF='D关闭时TIID收益',
                  combined_vs_both_off='两路联合收益', TIID_only_vs_D_only='仅TIID相对仅D收益')
    lines += ['', '## 配对作用与抵消', '', '| 比较 | MAE收益 | 正收益 | 负收益 | 抵消比例 | 预测绝对变化 |',
              '|---|---:|---:|---:|---:|---:|']
    for name, value in analysis['comparisons'].items():
        row = value['metrics']['all_nodes']
        values = [row[key] for key in ('baseline_minus_candidate_mae', 'gain_positive', 'gain_negative',
                                       'cancellation_fraction', 'prediction_absolute_change_macro')]
        lines.append('| '+labels[name]+' | '+' | '.join(map(number, values))+' |')
    lines += ['', 'MAE交互量：'+number(analysis['interaction']['metrics']['all_nodes']['mae_interaction_macro'])+'。',
              '正负收益来自逐点误差差值；预测变化大并不保证净收益大。', '',
              '## 分预测时距', '', '| 步 | 全开MAE | TIID开时D收益 | TIID关时D收益 | D开时TIID收益 | D关时TIID收益 | 交互量 |',
              '|---|---:|---:|---:|---:|---:|---:|']
    effects = ('D_given_TIID_ON', 'D_given_TIID_OFF', 'TIID_given_D_ON', 'TIID_given_D_OFF')
    for horizon in range(analysis['horizon_count']):
        values = [analysis['cases'][ON]['metrics']['all_nodes']['per_horizon_mae'][horizon]]
        values += [analysis['comparisons'][name]['metrics']['all_nodes']['per_horizon_baseline_minus_candidate_mae'][horizon]
                   for name in effects]
        values.append(analysis['interaction']['metrics']['all_nodes']['per_horizon_mae_interaction'][horizon])
        lines.append('| '+str(horizon+1)+' | '+' | '.join(map(number, values))+' |')
    lines += ['', '## 道路与区域', '', '| 区域 | 全开MAE | TIID开时D收益 | TIID关时D收益 | D开时TIID收益 | D关时TIID收益 | 交互量 |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for region, row in analysis['cases'][ON]['metrics'].items():
        values = [row['mae_macro']]
        values += [analysis['comparisons'][name]['metrics'][region]['baseline_minus_candidate_mae'] for name in effects]
        values.append(analysis['interaction']['metrics'][region]['mae_interaction_macro'])
        lines.append('| '+region+' | '+' | '.join(map(number, values))+' |')
    hard = analysis['fixed_hard_windows']
    lines += ['', '## 固定困难窗口', '', f"完整全开状态选出的最差10%完整窗口，共{hard['count']}个。四组合使用同一集合。", '',
              '| 状态 | 固定集MAE | 固定集RMSE macro | 相对全开窗口平均收益 |', '|---|---:|---:|---:|']
    for case, value in hard['cases'].items():
        row = value['metrics']['all_nodes']
        lines.append('| '+case+' | '+' | '.join(number(row[key]) for key in
                     ('mae_macro', 'rmse_macro', 'full_on_minus_case_window_mae_mean'))+' |')
    lines += ['', '每种状态各自最差5%用于比较尾部分布；固定困难集用于比较同一批窗口。窗口等权收益不必等于按有效点数汇总的全网收益。', '',
              '模块是否可删除，须再用相同输入与训练预算从头训练精简组确认。此报告不自动推荐删模块。', '',
              '数值明细：analysis.json；预测与显式有效掩码：paired_predictions.npz；容量诊断：mechanisms.json；回放：replay.json；保护检查：preservation.json。', '']
    (output/'report.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()
