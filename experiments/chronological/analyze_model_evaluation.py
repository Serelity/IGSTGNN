"""E0/E1: read saved development JSON/NPZ without Torch, inference or training.

The report describes a fixed common training prefix.  Existing overlapping
region summaries are not added as if they were disjoint, and window totals
are never presented as point predictions or independent incident replicates.
"""
import argparse
import csv
from datetime import datetime
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import uuid

import numpy as np

try:
    from .model_evaluation_metrics import validate_windows, window_summary, compare_windows
except ImportError:
    from model_evaluation_metrics import validate_windows, window_summary, compare_windows


SCHEMA = 'model_evaluation_e01_v1'
MINIMAL_SCHEMA = 'b3r_minimal_run_v1'
MINIMAL_CERTIFICATE = 'B3RMIN_INITIALIZATION_PAIR_PASS'
SUPPORTED = {'internal_spatial_is11_run_v1', 'b3r_run_v1', 'b3r_gate_run_v1', 'b3r_strength_run_g11_v1', MINIMAL_SCHEMA}
POLICY_FIELDS = ('train_samples', 'val_samples', 'stations', 'batch_size', 'max_epochs', 'patience',
                 'learning_rate', 'weight_decay', 'adam_eps', 'clip_grad_norm', 'lr_milestones',
                 'lr_gamma', 'supervised_horizons', 'selection_metric', 'min_delta')
INPUT_SOURCES = ('src/utils/chronological.py', 'experiments/chronological/train.py',
                 'experiments/chronological/smoke.py', 'src/models/igstgnn.py',
                 'src/base/model.py', 'src/models/incident_response.py', 'src/models/incident_spatial.py')
SNAPSHOTS = ('common_epoch', 'common_prefix_best')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value, label):
    require(isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value), label+' must be finite numeric')
    return float(value)


def close(left, right, label):
    require(np.allclose(left, right, rtol=1e-10, atol=1e-9), label+' differs from saved evidence')


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        if keys:
            writer = csv.DictWriter(stream, fieldnames=keys)
            writer.writeheader()
            writer.writerows({key: (json.dumps(value, ensure_ascii=False, allow_nan=False)
                                    if isinstance(value, (dict, list)) else value)
                             for key, value in row.items()} for row in rows)


class Sources:
    def __init__(self):
        self.hashes = {}

    def register(self, path, digest):
        key = str(Path(path).resolve())
        require(key not in self.hashes or self.hashes[key] == digest, 'Source changed while reading: '+key)
        self.hashes[key] = digest

    def read_json(self, path):
        path = Path(path).resolve()
        data = path.read_bytes()
        self.register(path, hashlib.sha256(data).hexdigest())
        def reject(value):
            raise ValueError('Non-finite JSON constant in '+str(path)+': '+value)
        return json.loads(data.decode('utf-8-sig'), parse_constant=reject)

    def verify(self):
        for name, digest in self.hashes.items():
            require(hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest,
                    'Source changed during evaluation; retry from a stable epoch snapshot: '+name)


def validate_metric(metric, label, all_nodes=False):
    require(isinstance(metric, dict), 'Missing metric object: '+label)
    if metric.get('status') == 'unavailable':
        require(not all_nodes, 'Global metric unavailable: '+label)
        return
    mae = finite(metric.get('mae_macro'), label+'.mae_macro')
    vector = np.asarray(metric.get('per_horizon_mae'), dtype=np.float64)
    counts = np.asarray(metric.get('valid_count_per_horizon'))
    require(vector.shape == counts.shape == (12,), 'Expected 12 horizon values: '+label)
    require(np.isfinite(vector).all() and (vector >= 0).all() and mae >= 0, 'Invalid MAE: '+label)
    require(counts.dtype.kind in 'iu' and (counts > 0).all(), 'Invalid horizon counts: '+label)
    close(mae, vector.mean(), label+' macro')
    if metric.get('mae_pooled') is not None:
        close(metric['mae_pooled'], np.average(vector, weights=counts), label+' pooled')
    rmse = np.asarray(metric.get('per_horizon_rmse'), dtype=np.float64)
    require(rmse.shape == (12,) and np.isfinite(rmse).all() and (rmse >= 0).all(), 'Invalid RMSE horizons: '+label)
    close(finite(metric.get('rmse_macro'), label+' RMSE macro'), rmse.mean(), label+' RMSE macro')
    scale = float(rmse.max())
    pooled_rmse = scale * math.sqrt(float(np.average((rmse/scale)**2, weights=counts))) if scale else 0.
    close(finite(metric.get('rmse_pooled'), label+' RMSE pooled'), pooled_rmse, label+' RMSE pooled')
    positive = np.asarray(metric.get('positive_target_count_per_horizon'))
    require(positive.shape == (12,) and positive.dtype.kind in 'iu' and (positive >= 0).all()
            and (positive <= counts).all(), 'Invalid positive target counts: '+label)
    mape = metric.get('per_horizon_mape')
    require(isinstance(mape, list) and len(mape) == 12, 'Invalid MAPE horizon axes: '+label)
    for h, value in enumerate(mape):
        if positive[h] == 0:
            require(value is None, 'MAPE without positive targets must be unavailable: '+label)
        else:
            require(finite(value, label+' MAPE') >= 0, 'Negative MAPE: '+label)
    if (positive > 0).all():
        close(finite(metric.get('mape_macro'), label+' MAPE macro'), np.mean(mape), label+' MAPE macro')
    else:
        require(metric.get('mape_macro') is None, 'Incomplete-horizon MAPE macro must be unavailable: '+label)
    if (positive > 0).any():
        selected = positive > 0
        expected = np.average(np.array([0. if v is None else v for v in mape])[selected], weights=positive[selected])
        close(finite(metric.get('mape_pooled'), label+' MAPE pooled'), expected, label+' MAPE pooled')
    else:
        require(metric.get('mape_pooled') is None, 'MAPE pooled without positive targets must be unavailable: '+label)


def initialization_evidence(initialization, arms):
    if initialization is None:
        return dict(status='missing_evidence', common_initial_sha256=None)
    certificates = {'IS11_INITIALIZATION_PAIR_PASS', 'B3R_INITIALIZATION_PAIR_PASS',
                    'B3RGATE_INITIALIZATION_PAIR_PASS', 'B3RSTRENGTH_INITIALIZATION_PAIR_PASS', MINIMAL_CERTIFICATE}
    require(initialization.get('status') in certificates, 'Initialization certificate did not pass')
    hashes = [initialization.get('arms', {}).get(arm, {}).get('common_initial_sha256') for arm in arms]
    require(all(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) for value in hashes),
            'Initialization certificate lacks common parameter fingerprints')
    require(len(set(hashes)) == 1, 'Common parameter initialization differs across arms')
    return dict(status='recorded_certificate_pass', certificate_status=initialization['status'],
                common_initial_sha256=hashes[0], reexecuted=False)


def load_run(root, sources, allow_engineering, selected_arms=None):
    root = Path(root).resolve()
    identity = sources.read_json(root/'identity.json')
    require(identity.get('schema') in SUPPORTED, 'Unsupported run identity schema: '+str(root))
    protocol = identity.get('protocol', {})
    require(identity.get('test_read') is False and protocol.get('test_read') is False,
            'Only development runs with test_read=false are supported')
    require(protocol.get('selection_metric') == 'all_nodes.mae_macro'
            and protocol.get('supervised_horizons') == 12, 'Unexpected selection/horizon protocol')
    require(type(identity.get('engineering_check')) is bool, 'Missing engineering scope')
    require(allow_engineering or not identity['engineering_check'],
            'Engineering run is not a model-quality evaluation; use --allow-engineering only for verification')
    for key in ('train_samples', 'val_samples', 'stations'):
        require(type(protocol.get(key)) is int and protocol[key] > 0, 'Missing positive protocol '+key)
    require(isinstance(identity.get('package'), dict) and identity['package'], 'Missing package fingerprint')
    require(type(identity.get('seed')) is int, 'Missing integer training seed')
    require(type(identity.get('batch_size')) is int and identity['batch_size'] > 0, 'Missing actual batch size')
    require(identity['batch_size'] == (2 if identity['engineering_check'] else protocol.get('batch_size')),
            'Actual batch size differs from protocol/check scope')
    require(isinstance(identity.get('graph_sha256'), str) and re.fullmatch('[0-9a-f]{64}', identity['graph_sha256']),
            'Missing graph fingerprint')
    arms = selected_arms if selected_arms is not None else protocol.get('arms')
    require(isinstance(arms, (list, tuple)) and len(arms) > 0 and len(set(arms)) == len(arms), 'Invalid arms')
    require(all(isinstance(arm, str) and re.fullmatch('[A-Z][A-Z0-9_]*', arm) for arm in arms), 'Unsafe arm name')
    require(set(arms) <= set(protocol.get('arms', [])), 'Requested reference arm outside recorded protocol')
    initialization = sources.read_json(root/'initialization.json') if (root/'initialization.json').is_file() else None
    initial_evidence = initialization_evidence(initialization, arms)
    if initialization is not None and identity.get('initialization_sha256') is not None:
        excluded = {'canonical_spatial_off_capacity_abs_max', 'canonical_spatial_off_prediction_abs_max'} if identity['schema'] == 'b3r_run_v1' else set()
        canonical = {key: value for key, value in initialization.items() if key not in excluded}
        digest = hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        require(identity['initialization_sha256'] == digest,
                'Initialization certificate fingerprint differs from identity')
    if identity['schema'] == MINIMAL_SCHEMA:
        require(initialization is not None and identity.get('initialization_sha256') is not None,
                'Minimal run requires a fingerprinted retained-state mapping certificate')
        require(initialization.get('status') == MINIMAL_CERTIFICATE
                and initialization.get('retained_state_exact_verified') is True
                and initialization.get('reference_seed') == identity['seed'],
                'Minimal retained-state mapping certificate did not pass')
        bridge = initialization.get('reference_common_initial_sha256')
        require(isinstance(bridge, str) and re.fullmatch('[0-9a-f]{64}', bridge),
                'Minimal certificate lacks reference initialization fingerprint')
        reference_sources = initialization.get('reference_source_sha256')
        require(isinstance(reference_sources, dict) and set(INPUT_SOURCES) <= set(reference_sources)
                and all(identity.get('source_sha256', {}).get(name) == digest
                        for name, digest in reference_sources.items()),
                'Minimal reference source mapping differs from recorded source')
        initial_evidence.update(reference_common_initial_sha256=bridge,
                                initialization_scope='exact_retained_state_from_fresh_reference',
                                full_parameter_equality_claim=False)
    run = dict(root=root, identity=identity, protocol=protocol, initialization=initialization,
               initialization_evidence=initial_evidence, variants={})
    for arm in arms:
        history = sources.read_json(root/arm/'history.json')
        summary = sources.read_json(root/arm/'summary.json')
        require(isinstance(history, list) and history, 'Missing completed history: '+arm)
        require(summary.get('arm') == arm and summary.get('test_read') is False, 'Summary identity mismatch: '+arm)
        require(summary.get('completed_epoch') == len(history), 'History/summary epoch mismatch: '+arm)
        best, best_epoch, previous_updates = math.inf, None, 0
        for epoch, row in enumerate(history, 1):
            require(row.get('epoch') == epoch, 'History epochs must be contiguous from one: '+arm)
            metrics = row.get('validation', {}).get('metrics', {})
            require('all_nodes' in metrics, 'Missing global validation metrics: '+arm)
            for name, metric in metrics.items():
                validate_metric(metric, arm+'/'+name, name == 'all_nodes')
            value = finite(row.get('validation_mae_macro'), arm+' validation MAE')
            close(value, metrics['all_nodes']['mae_macro'], arm+' history metric')
            if value < best:
                best, best_epoch = value, epoch
            if 'best_metric' in row:
                close(row['best_metric'], best, arm+' running best')
            if 'best_epoch' in row:
                require(row['best_epoch'] == best_epoch, 'Running best epoch mismatch: '+arm)
            updates = row.get('global_updates')
            require(type(updates) is int and updates > previous_updates, 'Invalid update budget: '+arm)
            require(type(row.get('epoch_updates')) is int and row['epoch_updates'] == updates-previous_updates,
                    'Missing or inconsistent epoch update budget: '+arm)
            require(finite(row.get('learning_rate_used'), arm+' learning_rate_used') >= 0,
                    'Negative training learning rate: '+arm)
            require(isinstance(row.get('order_sha256'), str) and re.fullmatch('[0-9a-f]{64}', row['order_sha256']),
                    'Missing or invalid training order_sha256: '+arm)
            previous_updates = updates
        close(summary.get('best_metric'), best, arm+' summary best')
        require(summary.get('best_epoch') == best_epoch, 'Summary best epoch mismatch: '+arm)
        require(summary.get('global_updates') == previous_updates, 'Summary update budget mismatch: '+arm)
        run['variants'][arm] = dict(arm=arm, run=run, history=history, summary=summary)
    return run


def compatibility(primary, reference):
    differences, missing = [], []
    left, right = primary['identity'], reference['identity']
    initial_a = primary['initialization_evidence']['common_initial_sha256']
    initial_b = reference['initialization_evidence']['common_initial_sha256']
    initialization_comparison = 'identical_recorded_common_parameter_fingerprint'
    if (left['schema'] == MINIMAL_SCHEMA) != (right['schema'] == MINIMAL_SCHEMA):
        minimal = primary if left['schema'] == MINIMAL_SCHEMA else reference
        bridge = minimal['initialization_evidence']['reference_common_initial_sha256']
        if left['schema'] == MINIMAL_SCHEMA:
            initial_a = bridge
        else:
            initial_b = bridge
        initialization_comparison = 'fresh_reference_fingerprint_and_exact_retained_state_mapping'
    if initial_a is None or initial_b is None:
        missing.append('common_initial_sha256')
    elif initial_a != initial_b:
        differences.append(dict(field='common_initial_sha256', primary=initial_a, reference=initial_b))
    for key in ('package', 'graph_sha256', 'graph_inputs_sha256', 'seed', 'batch_size', 'engineering_check'):
        if key not in left or key not in right:
            missing.append(key)
        elif left[key] != right[key]:
            differences.append(dict(field=key, primary=left[key], reference=right[key]))
    for key in POLICY_FIELDS:
        a, b = primary['protocol'].get(key), reference['protocol'].get(key)
        if a is None or b is None:
            missing.append('protocol.'+key)
        elif a != b:
            differences.append(dict(field='protocol.'+key, primary=a, reference=b))
    for name in INPUT_SOURCES:
        a, b = left.get('source_sha256', {}).get(name), right.get('source_sha256', {}).get(name)
        if a is None or b is None:
            missing.append('source_sha256.'+name)
        elif a != b:
            differences.append(dict(field='input_source:'+name, primary=a, reference=b))
    runtime_changes = {key: dict(primary=left.get('runtime', {}).get(key), reference=right.get('runtime', {}).get(key))
                       for key in ('torch', 'cuda', 'device_type', 'cublas_workspace_config')
                       if left.get('runtime', {}).get(key) != right.get('runtime', {}).get(key)}
    return dict(status='incompatible' if differences else 'missing_evidence' if missing else 'conditional_comparable',
                paired_allowed=not differences and not missing, differences=differences, missing_evidence=missing,
                environment_changes=runtime_changes,
                initialization_comparison=initialization_comparison,
                interpretation='Recorded package/protocol compatibility; point targets and masks not reread; initialization not re-executed')


def chosen_epoch(variant, snapshot, epoch):
    if snapshot == 'common_epoch':
        return epoch
    if snapshot == 'latest_best_descriptive':
        return int(variant['summary']['best_epoch'])
    prefix = variant['history'][:epoch]
    return min(range(1, len(prefix)+1), key=lambda index: prefix[index-1]['validation_mae_macro'])


def group_rows(variant, snapshot, epoch):
    row = variant['history'][epoch-1]
    metrics = row['validation']['metrics']
    result = []
    for group, metric in metrics.items():
        unavailable = metric.get('status') == 'unavailable'
        counts = metric.get('valid_count_per_horizon', metric.get('counts', []))
        result.append(dict(model=variant['key'], arm=variant['arm'], snapshot=snapshot, epoch=epoch, group=group,
                           status='unavailable' if unavailable else 'available',
                           paired_allowed=variant['eligible'], reference_run=variant['reference'],
                           **{key: metric.get(key) for key in ('mae_macro', 'rmse_macro', 'mape_macro',
                                                               'mae_pooled', 'rmse_pooled', 'mape_pooled')},
                           valid_count_per_horizon=counts,
                           registered_parameters=variant['summary'].get('parameters', {}).get('total_parameters'),
                           epoch_seconds=row.get('seconds'), peak_cuda_allocated_bytes=row.get('peak_cuda_allocated_bytes'),
                           aggregation='mean of 12 horizon metrics; MAPE fraction on positive targets only',
                           region_warning='Existing regions overlap; no additive spatial attribution' if group != 'all_nodes' and not group.startswith('road:') else None))
    return result


def compare_metric_groups(control, candidate, metadata):
    rows = []
    left = control['history'][metadata['control_epoch']-1]['validation']['metrics']
    right = candidate['history'][metadata['candidate_epoch']-1]['validation']['metrics']
    for group in sorted(set(left) | set(right)):
        a, b = left.get(group), right.get(group)
        prefix = dict(metadata, group=group, region_warning='Overlapping regions; no additive attribution')
        if a is None or b is None or a.get('status') == 'unavailable' or b.get('status') == 'unavailable':
            rows.append(dict(prefix, status='unavailable', reason='Group absent or has an empty horizon'))
            continue
        if a['valid_count_per_horizon'] != b['valid_count_per_horizon']:
            rows.append(dict(prefix, status='unavailable', reason='Group valid counts differ'))
            continue
        for metric in ('mae', 'rmse', 'mape'):
            if metric == 'mape' and a['positive_target_count_per_horizon'] != b['positive_target_count_per_horizon']:
                rows.append(dict(prefix, metric=metric, status='unavailable', reason='Positive-target counts differ'))
                continue
            for aggregation in ('macro', 'pooled'):
                av, bv = a.get(metric+'_'+aggregation), b.get(metric+'_'+aggregation)
                rows.append(dict(prefix, metric=metric, aggregation=aggregation, horizon=None,
                    status='available' if av is not None and bv is not None else 'unavailable',
                    control_value=av, candidate_value=bv, gain=av-bv if av is not None and bv is not None else None))
            for h, (av, bv) in enumerate(zip(a['per_horizon_'+metric], b['per_horizon_'+metric]), 1):
                rows.append(dict(prefix, metric=metric, aggregation='per_horizon', horizon=h,
                    status='available' if av is not None and bv is not None else 'unavailable',
                    control_value=av, candidate_value=bv, gain=av-bv if av is not None and bv is not None else None))
    return rows


def mechanism_rows(variant, snapshot, epoch):
    mechanism = variant['history'][epoch-1]['validation'].get('mechanisms')
    prefix = dict(model=variant['key'], snapshot=snapshot, epoch=epoch)
    if not isinstance(mechanism, dict):
        return [dict(prefix, kind='availability', name='mechanisms', status='unavailable')]
    rows = []
    for name, value in mechanism.get('report_state', {}).items():
        rows.append(dict(prefix, kind='legacy_report_state', name=name, value=value,
                         schema_warning='Legacy diagnostic definitions/denominators differ; no direct G1.1 subtraction'))
    for name, values in mechanism.get('capacity_statistics', {}).items():
        rows.append(dict(prefix, kind='capacity_statistic', name=name, **values))
    for values in mechanism.get('rows', []):
        rows.append(dict(prefix, kind='spatial_call_substep', **values))
    for name in ('d_strength_population_std', 'd_magnitude_weighted_strength', 'd_strength_status',
                 'raw_delta_status', 'restricted_stability', 'b2_stability', 'budget_counts_semantics',
                 'd_strength_denominator', 'flux_units', 'bottlenecks', 'bottleneck_denominator'):
        if name in mechanism:
            rows.append(dict(prefix, kind='summary', name=name, value=mechanism[name]))
    return rows


def time_evidence(windows):
    dates, invalid = [], 0
    for cutoff in windows['t0']:
        try:
            dates.append(datetime.fromisoformat(str(cutoff)).date().isoformat())
        except ValueError:
            invalid += 1
    return dict(unique_date_count=len(set(dates)), first_date=min(dates) if dates else None,
                last_date=max(dates) if dates else None, invalid_iso_cutoffs=invalid,
                event_count=None, event_id_verified=False, temporal_independence_verified=False,
                note='No event manifest/point targets read in E0/E1; t0 and window count are not independent incident counts')


def make_report(summary, global_rows, windows, comparability, metric_comparisons):
    fmt = lambda value: '不可用' if value is None else f'{value:.6f}'
    lines = ['# IGSTGNN E0/E1 模型评估', '',
             '范围：'+summary['scope']+'；共同预算：epoch '+str(summary['common_epoch'])+'。', '',
             '仅读取已保存JSON/NPZ，零新推理、零训练更新、未读取test或checkpoint张量。',
             '产物核验完成不等于模型科学有效；本次仍是开发集评价。', '',
             '## 共同epoch的全网与窗口表现', '',
             '| 模型 | MAE | RMSE macro | 完整/排除窗口 | 窗口P95 | 窗口最差5%均值 |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    for row in global_rows:
        if row['snapshot'] != 'common_epoch':
            continue
        window = windows.get(row['model']+'|common_epoch', {}).get('window_mae', {})
        lines.append('| '+row['model']+' | '+fmt(row.get('mae_macro'))+' | '+fmt(row.get('rmse_macro'))
                     +' | '+str(window.get('available_count', '不可用'))+'/'+str(window.get('excluded_count', '不可用'))
                     +' | '+fmt(window.get('p95'))+' | '+fmt(window.get('top_5_percent', {}).get('mean'))+' |')
    lines.extend(['', '## 共同训练预算内各自最佳', '',
                  '| 模型 | 选中epoch | MAE | RMSE macro |', '| --- | ---: | ---: | ---: |'])
    for row in global_rows:
        if row['snapshot'] == 'common_prefix_best':
            lines.append('| '+row['model']+' | '+str(row['epoch'])+' | '+fmt(row.get('mae_macro'))+' | '+fmt(row.get('rmse_macro'))+' |')
    lines.extend(['', '## 同一窗口的配对结果', '',
                  '正收益表示候选误差更低。窗口胜率使用窗口等权评分，全网收益使用各时距全网有效数，两者分母不同。', ''])
    for pair in summary['comparisons']:
        if pair['snapshot'] != 'common_epoch':
            continue
        report = pair['window_comparison']
        outcome, tail = report['window_outcomes'], report['fixed_control_worst_10_percent']
        lines.append(f'- {pair["candidate"]} vs {pair["control"]}：全网MAE收益 {fmt(report["mae_macro"]["gain"])}；'
                     f'窗口改善/持平/恶化 {outcome["wins"]}/{outcome["ties"]}/{outcome["losses"]}；'
                     f'固定控制最差10%窗口平均收益 {fmt(tail["window_mae_gain_mean"])}。')
    main_pairs = [pair for pair in summary['comparisons'] if pair['snapshot'] == 'common_epoch'
                  and (pair['control'] == 'reference:B0' or pair['control'] == summary['control'])
                  and not pair['candidate'].startswith('reference:')]
    lines.extend(['', '## 共同epoch的时距与空间收益', '',
                  '以下均为控制组减候选组；正数表示候选误差更低。分组结果不相加。', ''])
    for pair in main_pairs:
        rows = [row for row in metric_comparisons if row['control'] == pair['control']
                and row['candidate'] == pair['candidate'] and row['snapshot'] == 'common_epoch']
        lines.extend(['### '+pair['candidate']+' vs '+pair['control'], '',
                      '| 时距 | 全网MAE收益 | 全网RMSE收益 |', '| --- | ---: | ---: |'])
        lookup = {(row.get('horizon'), row.get('metric')): row.get('gain') for row in rows
                  if row['group'] == 'all_nodes' and row.get('aggregation') == 'per_horizon'}
        for h in range(1, 13):
            lines.append('| '+str(h)+' | '+fmt(lookup.get((h, 'mae')))+' | '+fmt(lookup.get((h, 'rmse')))+' |')
        lines.extend(['', '| 区域/道路 | MAE收益 | RMSE收益 |', '| --- | ---: | ---: |'])
        groups = sorted(set(row['group'] for row in rows) - {'all_nodes'})
        lookup = {(row['group'], row.get('metric')): row.get('gain') for row in rows if row.get('aggregation') == 'macro'}
        for group in groups:
            lines.append('| '+group+' | '+fmt(lookup.get((group, 'mae')))+' | '+fmt(lookup.get((group, 'rmse')))+' |')
        lines.append('')
    lines.extend(['', '## 解释与后续', '',
                  '- 全网、各道路、已有区域和12个时距见CSV；所有分组完整保留，既有区域重叠，不相加。',
                  '- common_prefix_best按共同预算前缀分别选最佳数值，common_epoch在同轮比较；latest_best_descriptive可能预算不同。',
                  '- 窗口尾部不是逐点尾部；当前不能得到点级正负收益抵消、signed bias或真实事故影响范围。',
                  '- 门控、D、容量、预算摘要见mechanism_chain.csv；alpha需与raw/applied D一起读。汇总不能反演状态联合关系。',
                  '- 多次使用的验证集、重叠窗口和单seed不支持独立事故泛化或显著性结论。',
                  '- 下一阶段适配G1.1同checkpoint配对推理，定位收益/伤害与D→容量→交换→读出的关系。'])
    if not summary['b0_comparison_available']:
        lines.append('- 本次没有可比B0的有效配对，不能宣称优于原IGSTGNN。可用--reference-run-dir加入旧B0/B2/B3，先做兼容性核验。')
    for name, entry in comparability.items():
        if name != 'primary':
            lines.append('- 参考 '+name+'：'+entry['status']+'；差异和缺失证据见comparability.json。')
    if summary['scope'] == 'engineering_only':
        lines.append('- **本次为小样本工程验收，以上数值不得用于模型排名或预测收益主张。**')
    return '\n'.join(lines)+'\n'


def analyze(run_dir, output_dir, control='B3R', epoch=None, reference_run_dir=None,
            reference_arms=('B0', 'B2', 'B3'), allow_engineering=False):
    sources = Sources()
    run_dir, output = Path(run_dir).resolve(), Path(output_dir).resolve()
    roots = [run_dir] + ([Path(reference_run_dir).resolve()] if reference_run_dir is not None else [])
    require(all(output != root and not root.is_relative_to(output) for root in roots),
            'Output cannot replace a source run or contain a source run')
    require(all(not output.is_relative_to(root/arm) for root in roots for arm in ('B0', 'B1', 'B2', 'B3', 'B3R', 'B3R_HALF', 'B3R_GATE', 'B3R_NO_D', 'B3R_OFF')),
            'Output must not be placed inside a source arm directory')
    require(not output.exists(), 'Use a fresh output directory; existing results are preserved')
    primary = load_run(run_dir, sources, allow_engineering)
    require(primary['initialization_evidence']['status'] == 'recorded_certificate_pass',
            'Primary comparison requires a recorded initialization certificate')
    require(control in primary['variants'], 'Control is not present in primary run')
    available = min(len(v['history']) for v in primary['variants'].values())
    epoch = available if epoch is None else epoch
    require(type(epoch) is int and 1 <= epoch <= available, 'Requested epoch exceeds common completed primary budget')
    variants, comparability, reference = {}, {'primary': dict(status='conditional_comparable', paired_allowed=True,
        initialization_evidence=primary['initialization_evidence'])}, None
    for arm, variant in primary['variants'].items():
        variant.update(key=arm, eligible=True, reference=False)
        variants[arm] = variant
    if reference_run_dir is not None:
        require(Path(reference_run_dir).resolve() != run_dir, 'Reference run duplicates primary run')
        reference = load_run(reference_run_dir, sources, True, reference_arms)
        result = compatibility(primary, reference)
        comparability['reference'] = result
        for arm, variant in reference['variants'].items():
            variant.update(key='reference:'+arm, eligible=result['paired_allowed'] and len(variant['history']) >= epoch, reference=True)
            variants[variant['key']] = variant
    output.mkdir(parents=True)
    try:
        global_rows, horizon_rows, regions, roads, selections, mechanisms, curves = [], [], [], [], [], [], []
        window_reports, window_cache, evidence, picked = {}, {}, [], {}
        for key, variant in variants.items():
            history = variant['history']
            for row in history:
                curves.append(dict(model=key, epoch=row['epoch'], validation_mae_macro=row['validation_mae_macro'],
                                   train_mae_macro=row.get('train', {}).get('metrics', {}).get('all_nodes', {}).get('mae_macro'),
                                   global_updates=row['global_updates'], seconds=row.get('seconds'),
                                   learning_rate_used=row.get('learning_rate_used'), peak_cuda_allocated_bytes=row.get('peak_cuda_allocated_bytes')))
            snapshots = (*SNAPSHOTS, 'latest_best_descriptive') if len(history) >= epoch else ('latest_best_descriptive',)
            for snapshot in snapshots:
                selected = chosen_epoch(variant, snapshot, epoch)
                picked[key+'|'+snapshot] = selected
                row = history[selected-1]
                rows = group_rows(variant, snapshot, selected)
                for metric in rows:
                    (global_rows if metric['group'] == 'all_nodes' else roads if metric['group'].startswith('road:') else regions).append(metric)
                for group, metric in row['validation']['metrics'].items():
                    counts = metric.get('valid_count_per_horizon', metric.get('counts', [0]*12))
                    for h in range(12):
                        horizon_rows.append(dict(model=key, snapshot=snapshot, epoch=selected, group=group, horizon=h+1,
                            status=metric.get('status', 'available'), valid_count=counts[h],
                            mae=metric.get('per_horizon_mae', [None]*12)[h],
                            rmse=metric.get('per_horizon_rmse', [None]*12)[h], mape_fraction=metric.get('per_horizon_mape', [None]*12)[h]))
                selections.append(dict(model=key, snapshot=snapshot, selected_epoch=selected,
                    common_budget_epoch=epoch, actual_completed_epoch=len(history), selected_global_updates=row['global_updates'],
                    current_best_epoch=variant['summary']['best_epoch'], checkpoint_exists=(variant['run']['root']/variant['arm']/'last.pt').is_file(),
                    weight_evidence='Current-best metadata only; checkpoint tensor not loaded' if selected == variant['summary']['best_epoch'] else 'Historical weight availability not guaranteed',
                    paired_allowed=variant['eligible'], registered_parameters=variant['summary'].get('parameters', {}).get('total_parameters')))
                mechanisms.extend(mechanism_rows(variant, snapshot, selected))
                if snapshot == 'latest_best_descriptive' or not variant['eligible']:
                    continue
                cache_key = key+'|'+str(selected)
                if cache_key not in window_cache:
                    path = variant['run']['root']/variant['arm']/f'validation_windows_epoch_{selected:03d}.npz'
                    expected = min(4, variant['run']['protocol']['val_samples']) if variant['run']['identity']['engineering_check'] else variant['run']['protocol']['val_samples']
                    stored = validate_windows(path, expected_rows=expected)
                    require((stored['counts'] <= variant['run']['protocol']['stations']).all(), 'Window counts exceed station count')
                    sources.register(path, stored['sha256'])
                    window_cache[cache_key] = stored
                stored = window_cache[cache_key]
                report = window_summary(stored)
                metric = row['validation']['metrics']['all_nodes']
                close(report['mae_macro'], metric['mae_macro'], key+' reconstructed MAE')
                close(report['per_horizon_mae'], metric['per_horizon_mae'], key+' reconstructed horizon MAE')
                require(report['valid_count_per_horizon'] == metric['valid_count_per_horizon'], 'Saved window/metric counts differ: '+key)
                window_reports[key+'|'+snapshot] = report
                dependence = dict(report['time_dependence'])
                dependence['window_dependence_note'] = dependence.pop('note')
                evidence.append(dict(model=key, snapshot=snapshot, epoch=selected, **dependence, **time_evidence(stored)))
        # Check common update/LR/RNG budgets before comparing predictions across arms/runs.
        eligible = [v for v in variants.values() if v['eligible']]
        control_variant = variants[control]
        for variant in eligible:
            for i in range(epoch):
                for field in ('global_updates', 'epoch_updates', 'learning_rate_used', 'order_sha256'):
                    require(control_variant['history'][i].get(field) == variant['history'][i].get(field),
                            'Paired training budget/order mismatch: '+variant['key']+' '+field)
        pairs = [(control, v['key']) for v in eligible if v['key'] != control]
        # Complement the primary-control table with within-four-group and B0 controls.
        for left, right in itertools.combinations([v['key'] for v in eligible if not v['reference']], 2):
            if (left, right) not in pairs and (right, left) not in pairs:
                pairs.append((left, right))
        if 'reference:B0' in variants and variants['reference:B0']['eligible']:
            pairs.extend(('reference:B0', v['key']) for v in eligible if not v['reference'])
        comparisons, paired_rows, difficult, metric_comparisons = [], [], [], []
        for left, right in pairs:
            for snapshot in SNAPSHOTS:
                a, b = picked[left+'|'+snapshot], picked[right+'|'+snapshot]
                # Same data/counts are verified for different selected epochs as well.
                report, rows = compare_windows(window_cache[left+'|'+str(a)], window_cache[right+'|'+str(b)])
                metadata = dict(control=left, candidate=right, snapshot=snapshot, control_epoch=a, candidate_epoch=b)
                comparisons.append(dict(metadata, window_comparison=report))
                metric_comparisons.extend(compare_metric_groups(variants[left], variants[right], metadata))
                for row in rows:
                    entry = dict(metadata, **row)
                    paired_rows.append(entry)
                    if row['in_control_worst_10_percent']:
                        difficult.append(entry)
        sources.verify()
        b0_available = any(p['control'] == 'reference:B0' or p['candidate'] == 'B0' or p['control'] == 'B0' for p in comparisons)
        scope = 'engineering_only' if primary['identity']['engineering_check'] else 'exploratory_validation'
        summary = dict(schema=SCHEMA, status='MODEL_EVALUATION_E01_COMPLETE', scope=scope,
            run_dir=str(run_dir), evaluation_dir=str(output), common_epoch=epoch, control=control,
            models=list(variants), comparisons=comparisons, b0_comparison_available=b0_available,
            comparability=comparability,
            source_artifacts_preserved=True, source_file_count=len(sources.hashes), new_inference_calls=0,
            training_updates=0, checkpoint_tensors_loaded=False, raw_data_read=False, test_read=False,
            pointwise_gain_decomposition_available=False, scientific_gain_claim=False,
            point_targets_and_masks_verified=False, independent_incident_count=None,
            outstanding=['G1.1 ON/intervention/ON inference', 'Pointwise benefit/harm and state joint analysis',
                         'Paired seeds and audited dependent-time uncertainty', 'Frozen-design independent test'])
        manifest = dict(schema=SCHEMA, scope=scope, primary_identity=primary['identity'],
            reference_identity=reference['identity'] if reference else None, selected_epochs=picked,
            source_files=sources.hashes, input_source_verification='Recorded fingerprints and saved arrays only; no checkpoint/raw-data tensors read',
            initialization_evidence=dict(primary=primary['initialization_evidence'],
                reference=reference['initialization_evidence'] if reference else None),
            evaluation_source_sha256={name: hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                                      for name in ('analyze_model_evaluation.py', 'model_evaluation_metrics.py')},
            gain_sign='control_minus_candidate', split='validation', test_read=False)
        for name, rows in (('metrics_global.csv', global_rows), ('metrics_horizon.csv', horizon_rows),
                           ('metrics_region.csv', regions), ('metrics_road.csv', roads),
                           ('checkpoint_selection.csv', selections), ('mechanism_chain.csv', mechanisms),
                           ('training_curves.csv', curves), ('data_evidence.csv', evidence),
                           ('metric_comparisons.csv', metric_comparisons),
                           ('window_pairs.csv', paired_rows), ('fixed_difficult_windows.csv', difficult)):
            write_csv(output/name, rows)
        write_json(output/'evaluation_manifest.json', manifest)
        write_json(output/'comparability.json', comparability)
        write_json(output/'window_statistics.json', window_reports)
        write_json(output/'summary.json', summary)
        (output/'report.md').write_text(make_report(summary, global_rows, window_reports, comparability, metric_comparisons), encoding='utf-8')
        return summary
    except Exception as error:
        write_json(output/'failure.json', dict(status='MODEL_EVALUATION_INCOMPLETE', error=str(error),
                                              new_inference_calls=0, training_updates=0, test_read=False))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--control', default='B3R')
    parser.add_argument('--epoch', type=int, help='Fixed common completed epoch; default minimum across primary arms')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--reference-run-dir', type=Path)
    parser.add_argument('--reference-arms', nargs='+', default=['B0', 'B2', 'B3'])
    parser.add_argument('--allow-engineering', action='store_true', help='Only verify small engineering artifacts; no quality claim')
    args = parser.parse_args(argv)
    output = args.output_dir or args.run_dir/('evaluation_e01_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6])
    print('Evaluation directory: '+str(output.resolve()), flush=True)
    summary = analyze(args.run_dir, output, args.control, args.epoch, args.reference_run_dir,
                      args.reference_arms, args.allow_engineering)
    compact = {key: summary[key] for key in ('status', 'scope', 'common_epoch', 'models', 'b0_comparison_available',
               'new_inference_calls', 'training_updates', 'checkpoint_tensors_loaded', 'source_artifacts_preserved')}
    compact['comparisons'] = [dict(control=p['control'], candidate=p['candidate'], snapshot=p['snapshot'],
        mae_gain=p['window_comparison']['mae_macro']['gain'], window_outcomes=p['window_comparison']['window_outcomes'],
        fixed_control_worst_10_percent_gain=p['window_comparison']['fixed_control_worst_10_percent']['window_mae_gain_mean'])
        for p in summary['comparisons'] if p['snapshot'] == 'common_epoch']
    compact['report'] = str(output.resolve()/'report.md')
    print(json.dumps(compact, ensure_ascii=False, indent=2, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
