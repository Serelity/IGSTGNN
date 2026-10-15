"""Read-only best-checkpoint validation for the replacement DIF architecture.

Reports three separately trained arms and five same-checkpoint interventions.
Historical SOURCE_MIN predictions are reused only when a matching archive
exists. This evaluator never runs the source model or a test loader.
"""
import argparse
from contextlib import ExitStack
from datetime import datetime
import gc
import hashlib
import json
import logging
import math
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
from experiments.chronological import train_min_conditional_propagation as training
from experiments.chronological.evaluate_min_early_event import shuffled_locations
from experiments.chronological.smoke import sha256
from src.models.incident_spatial import require
from src.models.incident_spatial_residual import tensor_state_hash
from src.utils.metrics import compute_all_metrics

SCHEMA = 'min_conditional_propagation_evaluation_v1'
CASES = ('HISTORY_ON', 'M_ONLY_ON', 'EVENT_ON', 'EVENT_REPORT_OFF', 'EVENT_LOGIT_OFF',
         'EVENT_VALUE_OFF', 'EVENT_SEMANTIC_UNK', 'EVENT_LOCATION_SHUFFLE')
COMPARISONS = {
    'EVENT_vs_HISTORY': ('HISTORY_ON', 'EVENT_ON'),
    'M_ONLY_vs_HISTORY': ('HISTORY_ON', 'M_ONLY_ON'),
    'EVENT_vs_M_ONLY': ('M_ONLY_ON', 'EVENT_ON'),
    'report_dependency': ('EVENT_REPORT_OFF', 'EVENT_ON'),
    'logit_dependency': ('EVENT_LOGIT_OFF', 'EVENT_ON'),
    'value_dependency': ('EVENT_VALUE_OFF', 'EVENT_ON'),
    'semantic_UNK_sensitivity': ('EVENT_SEMANTIC_UNK', 'EVENT_ON'),
    'location_shuffle_sensitivity': ('EVENT_LOCATION_SHUFFLE', 'EVENT_ON'),
    'EVENT_vs_SOURCE': ('SOURCE_MIN', 'EVENT_ON'),
    'HISTORY_vs_SOURCE': ('SOURCE_MIN', 'HISTORY_ON'),
}


def semantic_unknown(events):
    result = {key: np.asarray(value).copy() for key, value in events.items()}
    result['type'].fill(6)
    result['description'].fill(29)
    result['type_valid'].fill(False)
    result['description_valid'].fill(False)
    return result


def input_digest(batch, override=None):
    digest = hashlib.sha256()
    values = [('x', batch['x_data']), ('y', batch['y_data'])]
    values += list(sorted(batch['event_data'].items()))
    if override is not None:
        values += [('override_'+k, v) for k, v in sorted(override.items())]
    for key, value in values:
        array = np.ascontiguousarray(value)
        digest.update(key.encode())
        digest.update(str((array.shape, array.dtype.str)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def forward_case(model, batch, scaler, case, events_override=None):
    require(case in CASES, 'Unknown conditional-propagation inference case')
    expected = {'HISTORY_ON': 'history', 'M_ONLY_ON': 'mask_only'}.get(case, 'event')
    require(model._conditional_propagation_mode == expected, 'Case/model arm mismatch')
    before = input_digest(batch, events_override)
    try:
        device = next(model.parameters()).device
        events = batch['event_data'] if events_override is None else events_override
        if case == 'EVENT_SEMANTIC_UNK':
            events = semantic_unknown(events)
        event = None if case == 'HISTORY_ON' else training.event_tensors(events, device)
        x = torch.as_tensor(batch['x_data'], device=device, dtype=torch.float32)
        with torch.inference_mode():
            result = model(x, incident_data=event, report_enabled=case != 'EVENT_REPORT_OFF',
                logit_enabled=case != 'EVENT_LOGIT_OFF', value_enabled=case != 'EVENT_VALUE_OFF',
                semantic_enabled=True, return_spatial_details=True)
            prediction = scaler.inverse_transform(result['prediction']).squeeze(-1).cpu().numpy().copy()
        detail = result['propagation']
        require(detail['mode'] == expected and detail['logit_enabled'] is (case != 'EVENT_LOGIT_OFF')
            and detail['value_enabled'] is (case != 'EVENT_VALUE_OFF'), 'Wrong propagation intervention executed')
        support = detail['support'].detach().cpu().numpy()
        require(support.dtype == np.bool_ and support.shape == (len(x), x.shape[2]), 'Propagation support axis mismatch')
        statistics = dict(detail['statistics'])
        statistics.update(forward_calls=1, supported_node_positions=int(support.sum()))
        if case in ('HISTORY_ON', 'EVENT_REPORT_OFF'):
            require(not support.any(), 'History/report-OFF retained event support')
        require(prediction.shape == (len(x), 12, x.shape[2]) and np.isfinite(prediction).all(), 'Nonfinite/mismatched candidate output')
        return prediction, statistics
    finally:
        require(input_digest(batch, events_override) == before, 'Evaluation mutated input arrays')


def validation_axes(data):
    batches = list(data.loaders['val_loader'].get_iterator())
    require(bool(batches) and all(len(b['y_data']) > 0 for b in batches), 'Validation loader must contain nonempty batches')
    target = np.concatenate([data.scaler.inverse_transform(torch.as_tensor(b['y_data'], dtype=torch.float32)).squeeze(-1).numpy() for b in batches])
    mask_value = float(target.min()) if target.min() < 1 else 0.
    valid = target != mask_value
    require(np.isfinite(target).all() and valid.any() and (target[valid] > 0).all(),
        'Validation requires finite targets and at least one positive valid traffic target; nonpositive valid targets are unsupported')
    return batches, dict(target=target, valid=valid,
        sample_indices=data.audit['counts']['train_samples']+np.arange(len(target), dtype=np.int64),
        station_ids=np.asarray(data.station_ids, np.int64)), mask_value


def load_source_archive(args, identity, axes, partitions, mask_value):
    names = ('source_reference.json', 'source_reference.npz')
    present = [(args.reference_run/name).is_file() for name in names]
    if not any(present):
        return None, dict(status='SOURCE_MIN_ARCHIVE_UNAVAILABLE', new_inference_calls=0)
    require(all(present), 'Incomplete source reference archive; no fallback inference allowed')
    before = {name: sha256(args.reference_run/name) for name in names}
    certificate = training.legacy.read_json(args.reference_run/names[0])
    require(certificate.get('status') == 'SOURCE_MIN_A_BEST_FULL_ON_REPLAY_PASS'
        and certificate.get('source_best_epoch') == identity['source']['best_epoch']
        and certificate.get('source_best_model_sha256') == identity['source']['best_model_sha256']
        and certificate.get('engineering_only') is args.check
        and certificate.get('archive_sha256') == before[names[1]], 'Source archive selected checkpoint/scope mismatch')
    require(certificate.get('batch_partition') == partitions and certificate.get('mask_value') == mask_value, 'Source archive batches/mask differ')
    with np.load(args.reference_run/names[1], allow_pickle=False) as archive:
        require(set(archive.files) == {'prediction', *axes}, 'Unexpected source archive axes')
        require(all(np.array_equal(archive[name], value) for name, value in axes.items()), 'Source archive validation axes/targets differ')
        prediction = archive['prediction'].copy()
    require(prediction.shape == axes['target'].shape and np.isfinite(prediction).all(), 'Invalid source archived predictions')
    actual = compute_all_metrics(torch.from_numpy(prediction), torch.from_numpy(axes['target']), torch.tensor(mask_value))
    require(np.allclose(actual, certificate['actual'], atol=1e-5, rtol=1e-6)
        and np.allclose(actual, identity['source']['best_metrics'], atol=1e-5, rtol=1e-6), 'Source archived metrics disagree')
    require(before == {name: sha256(args.reference_run/name) for name in names}, 'Source archive changed while reading')
    return prediction, dict(status='SOURCE_MIN_ARCHIVED_INTERNAL_CONSISTENCY_PASS', origin=str(args.reference_run),
        files_sha256=before, new_inference_calls=0, new_checkpoint_replay=False, actual=list(actual))


def metrics(prediction, target, valid):
    require(prediction.shape == target.shape == valid.shape and valid.dtype == np.bool_, 'Metric axes differ')
    require(target.ndim == 3 and target.shape[1] == 12 and len(target) > 0 and valid.any()
        and np.isfinite(target).all() and np.isfinite(prediction).all() and (target[valid] > 0).all(),
        'Metrics require nonempty finite arrays and positive valid traffic targets')
    counts = valid.sum((0, 2))
    absolute = np.abs(prediction.astype(np.float64)-target.astype(np.float64))
    sums = np.where(valid, absolute, 0.).sum((0, 2))
    squares = np.where(valid, absolute**2, 0.).sum((0, 2))
    percentage = np.divide(absolute, target, out=np.zeros_like(absolute), where=valid & (target != 0))
    mape = np.where(valid, percentage, 0.).sum((0, 2))
    divide = lambda values: np.divide(values, counts, out=np.full(12, np.nan), where=counts > 0)
    mae_h, mse_h, mape_h = divide(sums), divide(squares), divide(mape)
    windows = window_scores(absolute, valid)
    pooled = lambda value: float(value.sum()/counts.sum()) if counts.sum() else None
    return dict(count=int(counts.sum()), mae_pooled=pooled(sums), rmse_pooled=math.sqrt(pooled(squares)) if counts.sum() else None,
        mape_pooled=pooled(mape), mae_macro=float(np.mean(mae_h)) if (counts > 0).all() else None,
        rmse_macro=float(np.sqrt(mse_h).mean()) if (counts > 0).all() else None,
        mape_macro=float(mape_h.mean()) if (counts > 0).all() else None,
        horizons=[dict(step=h+1, count=int(counts[h]), mae=None if not counts[h] else float(mae_h[h]),
            rmse=None if not counts[h] else float(np.sqrt(mse_h[h])), mape=None if not counts[h] else float(mape_h[h])) for h in range(12)],
        window_p95=float(np.quantile(windows[np.isfinite(windows)], .95)) if np.isfinite(windows).any() else None)


def window_scores(absolute, valid):
    """Original SOURCE hard-window rule: twelve horizon means, equal weight."""
    counts = valid.sum(2)
    values = np.divide(np.where(valid, absolute, 0.).sum(2), counts,
        out=np.zeros_like(counts, dtype=np.float64), where=counts > 0)
    scores = np.full(len(valid), np.nan)
    complete = (counts > 0).all(1)
    scores[complete] = values[complete].mean(1)
    return scores


def region_masks(events, adjacency, roads):
    direct = (np.abs(events['relation']).sum(-1) > 0) & events['report_present'][:, None] & events['relation_available'][:, None]
    edges = np.asarray(adjacency) > 0
    support = edges | edges.T | (edges.astype(np.int32) @ edges.astype(np.int32) > 0) | (edges.T.astype(np.int32) @ edges.T.astype(np.int32) > 0)
    np.fill_diagonal(support, True)
    potential = direct.astype(np.int32) @ support.T.astype(np.int32) > 0
    result = dict(all_nodes=np.ones_like(direct), direct_report_nodes=direct,
        hard_neighbor_only=potential & ~direct, remaining_nodes=~potential)
    for name, nodes in roads.items():
        mask = np.zeros(direct.shape[1], bool)
        mask[np.asarray(nodes, np.int64)] = True
        result['road:'+name] = np.broadcast_to(mask, direct.shape)
    require(np.all(result['direct_report_nodes'].astype(int)+result['hard_neighbor_only']+result['remaining_nodes'] == 1), 'Region partition failed')
    return result


def summarize(predictions, target, valid, regions, indices):
    result = dict(cases={case: metrics(value, target, valid) for case, value in predictions.items()}, comparisons={})
    flow_edges = (0., 5., 20., 50., 100., 200., float('inf'))
    groups = {name: np.broadcast_to(mask[:, None, :], target.shape) for name, mask in regions.items()}
    for low, high in zip(flow_edges, flow_edges[1:]):
        groups[f'flow:[{low:g},{high:g})'] = (target >= low) & (target < high)
    hard = None
    if 'SOURCE_MIN' in predictions:
        error = np.abs(predictions['SOURCE_MIN'].astype(np.float64)-target)
        scores = window_scores(error, valid)
        eligible = np.flatnonzero(np.isfinite(scores))
        chosen = eligible[np.lexsort((indices[eligible], -scores[eligible]))[:max(1, math.ceil(.1*len(eligible)))]]
        hard = np.zeros(len(target), bool)
        hard[chosen] = True
        groups['fixed_SOURCE_MIN_hard'] = np.broadcast_to(hard[:, None, None], target.shape)
        result['fixed_hard_sample_indices'] = indices[chosen].tolist()
    else:
        result['fixed_hard_sample_indices'] = None
    denominator = int(valid.sum())
    for name, (left, right) in COMPARISONS.items():
        if left not in predictions or right not in predictions:
            continue
        gain = np.abs(predictions[left].astype(np.float64)-target)-np.abs(predictions[right].astype(np.float64)-target)
        rows = {}
        for group, mask in groups.items():
            selected = valid & mask
            count = int(selected.sum())
            total = float(gain[selected].sum())
            positive, negative = float(np.maximum(gain[selected], 0).sum()), float(np.maximum(-gain[selected], 0).sum())
            horizons = [float(gain[:, h][selected[:, h]].mean()) if selected[:, h].any() else None for h in range(12)]
            rows[group] = dict(count=count, mae_gain_pooled=total/count if count else None,
                mae_gain_macro=float(np.mean(horizons)) if all(v is not None for v in horizons) else None,
                global_mae_contribution=total/denominator if denominator else None,
                gain_positive=positive/count if count else None, gain_negative=negative/count if count else None,
                horizon_mae_gains=horizons)
        result['comparisons'][name] = dict(control=left, candidate=right, groups=rows,
            rmse_gain_pooled=result['cases'][left]['rmse_pooled']-result['cases'][right]['rmse_pooled'],
            mape_gain_percentage_points=100*(result['cases'][left]['mape_pooled']-result['cases'][right]['mape_pooled']),
            fixed_hard_window_gain=float((window_scores(np.abs(predictions[left].astype(np.float64)-target), valid)[hard]-
                window_scores(np.abs(predictions[right].astype(np.float64)-target), valid)[hard]).mean())
                if hard is not None and hard.any() else None)
    result.update(schema=SCHEMA, validation_rows=len(target), primary_baseline='original_IGSTGNN_not_present_in_this_archive',
        mape_unit='fraction', group_scope='true_flow_for_posthoc_diagnosis_only',
        positive_target_requirement='valid_targets_strictly_positive_preserve_author_abs_error_over_target_MAPE',
        region_scope='direct_report_and_fixed_hard_neighbor_envelope_not_certified_incident_impact',
        fixed_hard_scope='archived_SOURCE_MIN_top10_percent_no_reselection_by_candidate' if hard is not None else 'unavailable_without_SOURCE_MIN_archive')
    return result


def write_report(output, result, comparability, source):
    fmt = lambda value: '不可用' if value is None else f'{value:.9f}'
    lines = ['# MIN 条件消息传播：三组配对验证', '',
        '原始 IGSTGNN 为论文主 baseline；本归档不能证明超过它。开发验证集已用于设计和选模，结果为探索性。', '',
        '工程检查：'+str(comparability['engineering_only'])+'；验证窗口：'+str(result['validation_rows'])+'。',
        'HISTORY / M_ONLY / EVENT 使用相同结构、相同初始化规则和完成预算；各自新架构阶段0参加选模，阶段0不等同历史 MIN。',
        'M_ONLY保留原支持M，屏蔽数值关系及语义。REPORT/LOGIT/VALUE关闭均为同检查点依赖检查，不等于删除后重训。', '',
        '| 状态 | MAE pooled | MAE macro | RMSE pooled | MAPE (%) | 窗口P95 |', '|---|---:|---:|---:|---:|---:|']
    for case, row in result['cases'].items():
        lines.append('| '+' | '.join([case, fmt(row['mae_pooled']), fmt(row['mae_macro']), fmt(row['rmse_pooled']),
            fmt(None if row['mape_pooled'] is None else 100*row['mape_pooled']), fmt(row['window_p95'])])+' |')
    lines += ['', '| 比较 | MAE收益 pooled | MAE收益 macro | RMSE收益 pooled | MAPE收益百分点 | 固定困难窗口收益 |', '|---|---:|---:|---:|---:|---:|']
    for name, row in result['comparisons'].items():
        all_nodes = row['groups']['all_nodes']
        lines.append('| '+' | '.join([name, fmt(all_nodes['mae_gain_pooled']), fmt(all_nodes['mae_gain_macro']),
            fmt(row['rmse_gain_pooled']), fmt(row['mape_gain_percentage_points']), fmt(row['fixed_hard_window_gain'])])+' |')
    for comparison in ('EVENT_vs_HISTORY', 'EVENT_vs_M_ONLY', 'report_dependency', 'logit_dependency', 'value_dependency'):
        if comparison not in result['comparisons']:
            continue
        lines += ['', '## '+comparison, '', '| 分组 | 点数 | MAE收益 pooled | 全网贡献 |', '|---|---:|---:|---:|']
        for name, row in result['comparisons'][comparison]['groups'].items():
            lines.append('| '+' | '.join([name, str(row['count']), fmt(row['mae_gain_pooled']), fmt(row['global_mae_contribution'])])+' |')
    lines += ['', '逐时距指标及配对收益见analysis.json；完整预测、真值、原掩码、样本和站点轴见paired_predictions.npz。',
        'MAPE沿用作者abs(pred−target)/target；有效真值须严格为正，否则拒绝评价，不偷偷替换指标定义。',
        '报告可沿内部传播路径扩展作用范围，不沿用F2“原支持外作用严格为零”的断言。硬图限制是结构取舍，不认证道路因果方向。',
        '潜块编号不是物理时延；归一化消息不等于车辆守恒、队列或真实容量。',
        'SOURCE_MIN状态：'+source['status']+'；新增源推理调用为0。未训练、未评价测试集。',
        '窗口可能重叠，不视为独立事故重复。语义UNK/位置重排可能分布外；同检查点依赖不等于真实事故因果。',
        '相同注册参数量不等于相同活跃报告参数。训练数据加载器反序列化发布全样本对象，但不迭代测试loader。']
    (output/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--threads', type=int, default=3)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args(argv)
    require(args.threads > 0, 'Invalid threads')
    args.run_dir = args.run_dir.resolve()
    if args.output_dir is None:
        args.output_dir = args.run_dir/('conditional_propagation_evaluation_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6])
    args.output_dir = args.output_dir.resolve()
    require(args.output_dir != args.run_dir and args.run_dir not in args.output_dir.parents[1:]
        and not args.output_dir.exists(), 'Evaluation requires a new direct-child or separate output directory')
    return args


def evaluate_run(args):
    identity = training.legacy.read_json(args.run_dir/'identity.json')
    require(identity.get('schema') == training.SCHEMA and identity.get('engineering_check') is args.check,
            'Wrong run identity/check scope')
    require(identity['source_sha256'] == training.source_fingerprints(), 'Candidate code changed since training')
    require(set(identity['arm_modes']) == set(training.ARMS), 'Evaluation requires all three trained arms')
    runtime = dict(torch=str(torch.__version__), cuda=torch.version.cuda, device_type=torch.device(args.device).type,
        cublas_workspace_config=os.environ['CUBLAS_WORKSPACE_CONFIG'])
    require(identity['runtime'] == runtime, 'Runtime changed since training')
    request = SimpleNamespace(**vars(args), data_dir=Path(identity['data_dir']) if 'data_dir' in identity else None)
    # All reconstruction paths are immutable training metadata, never guessed
    # from current defaults or another run's similarly named directory.
    for key in ('data_dir', 'graph_dir', 'source_run', 'reference_run', 'protocol'):
        setattr(request, key, Path(identity['paths'][key]))
    for protected_root in (request.source_run, request.reference_run):
        require(args.output_dir != protected_root and protected_root not in args.output_dir.parents
            and args.output_dir not in protected_root.parents, 'Output must be separate from source/reference roots')
    request.arms, request.only = list(training.ARMS), None
    request.seed, request.batch_size = identity['seed'], identity['protocol']['batch_size']
    request.receiver_chunk_size, request.checkpoint_messages = identity['receiver_chunk_size'], identity['checkpoint_messages']
    request.epochs, request.worker = 1, None
    protected = [args.run_dir/'identity.json', args.run_dir/'initialization.json']
    protected += [args.run_dir/arm/name for arm in training.ARMS for name in ('last.pt', 'history.json', 'summary.json', 'stage_zero.json')]
    require(all(p.is_file() for p in protected), 'Missing committed candidate artifacts')
    before = {str(p): sha256(p) for p in protected}
    counts = {case: 0 for case in CASES}
    logger = logging.getLogger('conditional_propagation_evaluation')
    logger.addHandler(logging.NullHandler())
    args.output_dir.mkdir(parents=True)
    try:
        with ExitStack() as stack:
            stack.enter_context(training.legacy.run_lock(args.run_dir))
            require(before == {str(p): sha256(p) for p in protected}, 'Candidate artifacts changed before evaluation lock')
            stack.enter_context(training.readonly_source_lock(request.source_run))
            source_before = training.source_snapshot(request)
            protocol, data, graph, adjacency, initialization, rebuilt, source = training.prepare(request, logger, skip_preflight=True)
            require(rebuilt == identity, 'Rebuilt candidate identity changed')
            batches, axes, mask_value = validation_axes(data)
            partitions = [len(b['x_data']) for b in batches]
            events = {key: np.concatenate([b['event_data'][key] for b in batches]) for key in batches[0]['event_data']}
            shuffled, donors, shuffle_audit = shuffled_locations(events, axes['sample_indices'], 20261015)
            _, roads, _ = training.legacy.load_graph(request.graph_dir, data.station_ids, protocol['graph_sha256'])
            regions = region_masks(events, adjacency, roads)
            predictions, mechanisms, replay, saved_by_arm = {}, {}, {}, {}
            for arm, on_case in zip(training.ARMS, CASES[:3]):
                saved = torch.load(args.run_dir/arm/'last.pt', map_location='cpu', weights_only=False)
                training.validate_saved(saved, identity, arm)
                require(training.legacy.read_json(args.run_dir/arm/'history.json') == saved['history'], 'Checkpoint/history mismatch')
                require(training.legacy.read_json(args.run_dir/arm/'stage_zero.json') == saved['stage_zero'], 'Stage-zero sidecar differs from checkpoint')
                companion = training.legacy.read_json(args.run_dir/arm/'summary.json')
                require(all(companion.get(k) == saved[k] for k in ('arm', 'completed_epoch', 'best_epoch', 'best_metric', 'global_updates')),
                    'Summary counters differ from checkpoint')
                model, _ = training.build_arm(request, arm, len(data.station_ids), graph, args.device, source, adjacency)
                model.load_state_dict(saved['best_model_state'], strict=True)
                model.eval()
                state_hash = tensor_state_hash(model.state_dict())
                cases = (on_case,)+CASES[3:] if arm == 'EVENT' else (on_case,)
                for case in cases:
                    pieces, statistics, position = [], {}, 0
                    for batch in batches:
                        end = position+len(batch['x_data'])
                        override = {key: value[position:end] for key, value in shuffled.items()} if case.endswith('SHUFFLE') else None
                        counts[case] += 1
                        value, row = forward_case(model, batch, data.scaler, case, override)
                        pieces.append(value)
                        training.merge_statistics(statistics, row)
                        position = end
                    predictions[case], mechanisms[case] = np.concatenate(pieces), statistics
                expected = [saved['stage_zero'][k] for k in ('validation_mae', 'validation_mape', 'validation_rmse')] if saved['best_epoch'] == 0 else [saved['history'][saved['best_epoch']-1][k] for k in ('validation_mae', 'validation_mape', 'validation_rmse')]
                actual = compute_all_metrics(torch.from_numpy(predictions[on_case]), torch.from_numpy(axes['target']), torch.tensor(mask_value))
                require(np.allclose(actual, expected, atol=1e-5, rtol=1e-6), 'Best checkpoint replay failed: '+arm)
                if arm == 'EVENT':
                    restored = []
                    for batch in batches:
                        counts['EVENT_ON'] += 1
                        value, _ = forward_case(model, batch, data.scaler, 'EVENT_ON')
                        restored.append(value)
                    require(np.array_equal(np.concatenate(restored), predictions['EVENT_ON']), 'EVENT_ON did not restore exactly after interventions')
                require(tensor_state_hash(model.state_dict()) == state_hash, 'Evaluation mutated model state')
                replay[arm] = dict(best_stage_epoch=saved['best_epoch'], actual=list(actual), expected=expected,
                    status='BEST_CHECKPOINT_VALIDATION_REPLAY_PASS', best_model_state_sha256=state_hash)
                saved_by_arm[arm] = saved
                del model
                gc.collect()
            budgets = {a: s['completed_epoch'] for a, s in saved_by_arm.items()}
            updates = {a: s['global_updates'] for a, s in saved_by_arm.items()}
            require(len(set(budgets.values())) == len(set(updates.values())) == 1, 'Complete identical budgets required')
            common = next(iter(budgets.values()))
            require(all(len({s['history'][e]['train_order_sha256'] for s in saved_by_arm.values()}) == 1 for e in range(common)), 'Paired training order differs')
            archived, source_certificate = load_source_archive(request, identity, axes, partitions, mask_value)
            if archived is not None:
                predictions['SOURCE_MIN'] = archived
            result = summarize(predictions, axes['target'], axes['valid'], regions, axes['sample_indices'])
            comparable = dict(completed_epochs=budgets, global_updates=updates,
                best_stage_epochs={a: s['best_epoch'] for a, s in saved_by_arm.items()}, same_completed_training_budget=True,
                own_stage_zero_eligible=True, validation_rows=len(axes['target']), engineering_only=args.check,
                scope='same_architecture_equal_completed_budget_independent_best_validation')
            require(training.source_snapshot(request) == source_before, 'Source MIN changed during evaluation')
            require(before == {str(p): sha256(p) for p in protected}, 'Candidate artifacts changed during evaluation')
            np.savez_compressed(args.output_dir/'paired_predictions.npz', **predictions, **axes,
                **{'region_'+name: value for name, value in regions.items()}, location_shuffle_donors=donors)
            summary = dict(schema=SCHEMA, status='MIN_CONDITIONAL_PROPAGATION_EVALUATION_PASS',
                engineering_only=args.check, validation_rows=len(axes['target']), cases=list(predictions),
                new_inference_calls=dict(counts, SOURCE_MIN=0), optimizer_updates=0, test_evaluated=False,
                inference_count_scope='forward_case_attempts_including_an_interrupted_attempt_failure_counts_not_exact_completed_model_calls',
                source_artifacts_preserved=True, candidate_artifacts_preserved=True, on_exactly_restored=True,
                primary_baseline='original_IGSTGNN_not_present_in_this_archive', output_dir=str(args.output_dir),
                all_nodes={name: row['groups']['all_nodes'] for name, row in result['comparisons'].items()})
            for name, value in (('analysis.json', result), ('comparability.json', comparable), ('replay.json', replay),
                ('source_reference.json', source_certificate), ('mechanisms.json', dict(cases=mechanisms,
                    aggregation='sum_except_suffix_max_and_min', location_shuffle=shuffle_audit,
                    no_legacy_latent_exchange_audit=True)), ('summary.json', summary),
                ('preservation.json', dict(candidate_before=before, source_before=source_before,
                    candidate_artifacts_preserved=True, source_artifacts_preserved=True)),
                ('provenance.json', dict(evaluator_sha256=sha256(Path(__file__)), training_identity_sha256=before[str(args.run_dir/'identity.json')],
                    original_batch_partition=partitions, mask_value=mask_value,
                    predictions_sha256=sha256(args.output_dir/'paired_predictions.npz')))):
                training.write_json(args.output_dir/name, value)
            write_report(args.output_dir, result, comparable, source_certificate)
            return summary
    except BaseException as error:
        training.write_json(args.output_dir/'failure.json', dict(status='MIN_CONDITIONAL_PROPAGATION_EVALUATION_FAILED',
            error_type=type(error).__name__, error=str(error), new_inference_calls=counts,
            inference_count_scope='forward_case_attempts_not_exact_completed_model_calls', optimizer_updates=0, test_evaluated=False))
        raise


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    require(torch.device(args.device).type in ('cpu', 'cuda') and (not args.device.startswith('cuda') or torch.cuda.is_available()), 'Device unavailable')
    print(json.dumps(evaluate_run(args), ensure_ascii=False, indent=2, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
