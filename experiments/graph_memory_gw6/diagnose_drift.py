"""Read certified GW6 BEST/LAST prediction archives without model inference.

This diagnostic imports only the standard library and NumPy. It never loads a
checkpoint, constructs a model, accesses a GPU, or changes an input archive.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


ARMS = ('RECURRENT_CONTROL', 'POOLED_MEMORY', 'ADDITIVE_MEMORY', 'JOINT_MEMORY')
AXIS_KEYS = ('y', 'valid', 'sample_ids', 'node_ids', 'road_labels', 'direct_support')
ARCHIVE_SCHEMA = 'original_igstgnn_graph_memory_gw6_archive_v1'
PRODUCER_SCHEMA = 'gw6_graph_memory_producer_v1'
TRAINING_SCHEMA = 'original_igstgnn_graph_memory_gw6_training_v1'
LIMITATIONS = [
    'Only existing development-validation predictions are read; no new inference, training, test evaluation or GPU work.',
    'BEST is independently selected within each arm; its epoch may differ. This is an endpoint comparison, not an observed epoch-19 trajectory.',
    'Archive hashes, run identity, producer metadata and current summary checkpoint references are checked; checkpoints and the original producer are not re-executed or independently recertified.',
    'Squared mean shift / mean squared prediction change is a prediction-change energy fraction, never a fraction of MAE deterioration explained.',
    'Centered LAST = LAST minus the global mean(LAST minus BEST), fitted on these same validation predictions. It is an exploratory diagnostic, not a new model score, test result or deployment correction.',
    'Centered removed fraction is (LAST MAE minus centered LAST MAE) / (LAST MAE minus BEST MAE), only when the denominator is positive; it is not clipped and can lie outside [0,1].',
    'Group boundaries use shared truth and mask. Target units are not inferred. Signed contributions may cancel; no contribution is divided by a near-zero net MAE change.',
    'Node concentration is descriptive: top ceil(10% of valid nodes) are selected separately by each nonnegative quantity, with their valid-point fraction reported. No root-cause threshold is imposed.',
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    with Path(path).open('w', encoding='utf-8', newline='\n') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def _vectors(y, best, last):
    arrays = tuple(np.asarray(value, dtype=np.float64) for value in (y, best, last))
    require(all(value.ndim == 1 for value in arrays), 'Expected one-dimensional valid-point vectors')
    require(arrays[0].size > 0 and len({value.shape for value in arrays}) == 1,
        'Valid-point vectors must have the same nonempty shape')
    require(all(np.isfinite(value).all() for value in arrays), 'Nonfinite valid-point values')
    require((arrays[0] > 0).all(), 'Positive valid targets required for MAPE')
    return arrays


def _metrics(error, y):
    return dict(mae=float(np.mean(np.abs(error))), rmse=float(np.sqrt(np.mean(error * error))),
        bias=float(np.mean(error)), mape=float(np.mean(np.abs(error) / y)))


def pair_statistics(y, best, last):
    """Global endpoint and constant-translation diagnostic on valid points.

    MAPE values are fractions. Centering always uses one global mean prediction
    change, rather than fitting truth residuals or separately fitting groups.
    """
    y, best, last = _vectors(y, best, last)
    delta = last - best
    mu = float(delta.mean())
    variance = float(np.var(delta))
    mse = float(np.mean(delta * delta))
    result = dict(count=int(y.size), shift_mean=mu, shift_std=float(np.sqrt(variance)),
        shift_negative_fraction=float(np.mean(delta < 0)), shift_mse=mse,
        shift_mean_square=mu * mu, shift_variance=variance,
        global_shift_energy_fraction=(mu * mu / mse if mse > 0 else None))
    for name, value in zip(('q01', 'q05', 'q25', 'q50', 'q75', 'q95', 'q99'),
            np.quantile(delta, (.01, .05, .25, .50, .75, .95, .99))):
        result['shift_' + name] = float(value)
    best_error, last_error = best - y, last - y
    for name, error in (('best', best_error), ('last', last_error), ('centered_last', last_error - mu)):
        result.update({name + '_' + key: value for key, value in _metrics(error, y).items()})
    worsening = np.abs(last_error) - np.abs(best_error)
    deterioration = result['last_mae'] - result['best_mae']
    removed = result['last_mae'] - result['centered_last_mae']
    result.update(mae_deterioration=deterioration, centered_removed_mae=removed,
        centered_removed_fraction=removed / deterioration if deterioration > 0 else None,
        gross_mae_worsening=float(np.maximum(worsening, 0).mean()),
        gross_mae_improvement=float(np.maximum(-worsening, 0).mean()))
    return result


def group_statistics(labels, n_groups, y, best, last, global_shift):
    """Vectorized grouped sums, normalized contributions use all valid points."""
    y, best, last = _vectors(y, best, last)
    labels = np.asarray(labels)
    require(labels.shape == y.shape and labels.dtype.kind in 'iu', 'Invalid group labels')
    require(isinstance(n_groups, (int, np.integer)) and n_groups > 0,
        'Positive integer group count required')
    require(labels.min() >= 0 and labels.max() < n_groups, 'Group label outside registered groups')
    if labels.dtype.kind == 'u':
        labels = labels.astype(np.int64)
    require(np.isfinite(global_shift), 'Finite global shift required')
    count = np.bincount(labels, minlength=n_groups)
    rows = [dict(group=index, count=int(value), point_fraction=float(value / y.size))
        for index, value in enumerate(count)]

    def add_mean(name, values, square_root=False):
        sums = np.bincount(labels, weights=values, minlength=n_groups)
        for index, row in enumerate(rows):
            value = float(sums[index] / count[index]) if count[index] else None
            row[name] = float(np.sqrt(max(0., value))) if square_root and value is not None else value
        return sums

    error_best, error_last = best - y, last - y
    for name, error in (('best', error_best), ('last', error_last),
            ('centered_last', error_last - float(global_shift))):
        add_mean(name + '_mae', np.abs(error))
        add_mean(name + '_bias', error)
        add_mean(name + '_rmse', error * error, square_root=True)
    delta = last - best
    add_mean('shift_mean', delta)
    add_mean('shift_mse', delta * delta)
    for row in rows:
        row['shift_std'] = (float(np.sqrt(max(0., row['shift_mse'] - row['shift_mean'] ** 2)))
            if row['count'] else None)
    change = np.abs(error_last) - np.abs(error_best)
    for name, values in (('net_mae_contribution', change),
            ('gross_mae_worsening_contribution', np.maximum(change, 0)),
            ('gross_mae_improvement_contribution', np.maximum(-change, 0))):
        sums = np.bincount(labels, weights=values, minlength=n_groups)
        for index, row in enumerate(rows):
            row[name] = float(sums[index] / y.size)
    return rows


def target_bins(y, max_bins=5):
    """Shared truth quantiles; repeated edges merge, a constant target has one bin."""
    y = np.asarray(y, dtype=np.float64)
    require(y.ndim == 1 and y.size > 0 and np.isfinite(y).all(), 'Finite nonempty target vector required')
    require(isinstance(max_bins, (int, np.integer)) and max_bins > 0, 'Positive integer bin count required')
    bounds = np.unique(np.quantile(y, np.linspace(0., 1., max_bins + 1)))
    if bounds.size == 1:
        return np.zeros(y.size, dtype=np.int64), np.repeat(bounds, 2)
    return np.searchsorted(bounds[1:-1], y, side='right').astype(np.int64), bounds


def require_same_axes(best_axes, last_axes):
    for key in AXIS_KEYS:
        a, b = np.asarray(best_axes[key]), np.asarray(last_axes[key])
        require(a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b),
            'BEST/LAST shared axis differs in dtype, shape or values: ' + key)


def verify_run(run, epochs):
    paths = {name: run / (name + '.json') for name in ('identity', 'summary', 'budget', 'source_reference')}
    before = {name: sha256(path) for name, path in paths.items()}
    documents = {name: read_json(path) for name, path in paths.items()}
    identity, summary = documents['identity'], documents['summary']
    digest_content = {key: value for key, value in identity.items() if key != 'identity_sha256'}
    require(hashlib.sha256(json.dumps(digest_content, sort_keys=True).encode()).hexdigest()
        == identity.get('identity_sha256'), 'Run identity digest differs')
    require(identity.get('schema') == TRAINING_SCHEMA and identity.get('engineering_only') is False
        and identity.get('stage') == 'joint', 'A certified joint training identity is required')
    require(identity.get('arms') == list(ARMS), 'Expected the four registered default arms in canonical order')
    require(summary.get('schema') == TRAINING_SCHEMA and summary.get('status') == 'GW6_TRAINING_COMPLETE'
        and summary.get('stage') == 'joint' and summary.get('requested_new_epochs') == epochs
        and summary.get('same_completed_new_training_budget') is True,
        'Complete the four arms at the requested common training budget')
    require(documents['budget'].get('target_completed_epochs') == epochs, 'Current budget differs')
    require(summary.get('registered_arms') == list(ARMS) and set(summary.get('arms', {})) == set(ARMS),
        'Summary registered arms differ')
    require(summary.get('source_reference') == identity.get('source_reference')
        == documents['source_reference'], 'Source reference identity differs')
    for arm in ARMS:
        row = summary['arms'][arm]
        require(row.get('arm') == arm and row.get('completed_epoch') == epochs
            and row.get('target_completed_epochs') == epochs and row.get('target_reached') is True
            and row.get('engineering_only') is False,
            'Incomplete or engineering-only arm: ' + arm)
        require(isinstance(row.get('best_epoch'), int) and 1 <= row['best_epoch'] <= epochs,
            'Invalid BEST epoch: ' + arm)
    require(all(sha256(path) == before[name] for name, path in paths.items()),
        'Run documents changed during authentication')
    records = {name: dict(path=str(path), sha256=before[name]) for name, path in paths.items()}
    return identity, summary, records


def verify_eval(directory, role, identity, summary, epochs):
    metadata_path, analysis_path = directory / 'metadata.json', directory / 'analysis.json'
    hashes = {name: sha256(path) for name, path in (('metadata', metadata_path), ('analysis', analysis_path))}
    metadata, analysis = read_json(metadata_path), read_json(analysis_path)
    require(metadata.get('schema') == PRODUCER_SCHEMA and metadata.get('status') == 'GW6_PREDICTION_PASS'
        and metadata.get('engineering_only') is False and metadata.get('stage') == 'joint',
        'Not a completed certified joint producer directory: ' + str(directory))
    require(metadata.get('common_completed_new_epochs') == epochs
        and metadata.get('checkpoint_selection') == role,
        'Evaluation role or common completed budget differs: ' + str(directory))
    require(metadata.get('run_identity_sha256') == identity['identity_sha256']
        and metadata.get('source_reference') == identity['source_reference']
        and metadata.get('registered_arms') == identity['arms'],
        'Evaluation identity, source reference or arms differ: ' + str(directory))
    require(analysis.get('schema') == 'original_igstgnn_graph_memory_gw6_evaluation_v1'
        and analysis.get('status') == 'GW6_EVALUATION_COMPLETE'
        and analysis.get('engineering_only') is False and analysis.get('stage') == 'joint'
        and analysis.get('registered_arms') == identity['arms'], 'Evaluation analysis is incomplete or differs')
    for arm in ARMS:
        selected = metadata.get('checkpoint_roles', {}).get(arm, {})
        saved = summary['arms'][arm]
        require(selected.get('selection') == role and selected.get('policy') == arm
            and selected.get('historical') is False, 'Checkpoint role differs: ' + arm)
        expected_epoch = saved['best_epoch'] if role == 'best' else saved['completed_epoch']
        require(selected.get('epoch') == expected_epoch, 'Checkpoint epoch differs: ' + arm)
        if role == 'best':
            require(selected.get('checkpoint_sha256') == saved.get('best_checkpoint_sha256')
                and isinstance(saved.get('best_checkpoint_sha256'), str), 'Current BEST checkpoint hash differs: ' + arm)
        else:
            require(selected.get('model_tensor_sha256') == saved.get('last_model_tensor_sha256')
                and isinstance(saved.get('last_model_tensor_sha256'), str), 'Current LAST tensor hash differs: ' + arm)
        require(metadata.get('validation_replay', {}).get(arm, {}).get('passed') is True,
            'Native validation replay was not certified: ' + arm)
    archive_path = directory / 'paired_predictions.npz'
    require(archive_path.is_file(), 'Missing paired_predictions.npz')
    require(isinstance(metadata.get('archive_sha256'), str)
        and sha256(archive_path) == metadata['archive_sha256'], 'Prediction archive hash differs')
    require(sha256(metadata_path) == hashes['metadata'] and sha256(analysis_path) == hashes['analysis'],
        'Producer documents changed during authentication')
    return metadata, dict(directory=str(directory), metadata_sha256=hashes['metadata'],
        analysis_sha256=hashes['analysis'], archive_path=str(archive_path),
        archive_sha256=metadata['archive_sha256'], checkpoint_selection=role,
        checkpoint_roles={arm: metadata['checkpoint_roles'][arm] for arm in ARMS})


def select_eval(run, explicit, role, identity, summary, epochs):
    if explicit is not None:
        directory = explicit.resolve()
        metadata, record = verify_eval(directory, role, identity, summary, epochs)
        return directory, metadata, record
    rejected = []
    candidates = sorted((path for path in run.glob('eval_' + role + '_*') if path.is_dir()),
        key=lambda path: (path.stat().st_mtime_ns, path.name), reverse=True)
    for directory in candidates:
        try:
            metadata, record = verify_eval(directory, role, identity, summary, epochs)
            record['skipped_newer_directories'] = rejected
            return directory, metadata, record
        except (ValueError, KeyError, OSError, json.JSONDecodeError) as error:
            rejected.append(dict(directory=str(directory), reason=str(error)))
    raise ValueError('No complete matching ' + role.upper() + ' evaluation: ' + json.dumps(rejected))


def load_axes(payload, identity):
    for key, expected in (('schema', ARCHIVE_SCHEMA), ('axis_order', 'sample,horizon,node'), ('stage', 'joint')):
        value = np.asarray(payload[key])
        require(value.shape == () and value.dtype.kind == 'U' and value.item() == expected,
            'Invalid explicit archive scalar: ' + key)
    arms = np.asarray(payload['registered_arms'])
    require(arms.ndim == 1 and arms.dtype.kind == 'U' and arms.tolist() == identity['arms'],
        'Archive registered arms differ')
    axes = {key: np.asarray(payload[key]) for key in AXIS_KEYS}
    y, valid = axes['y'], axes['valid']
    require(y.ndim == 3 and y.shape[1] == 12 and min(y.shape) > 0 and y.dtype.kind == 'f',
        'Expected floating [sample,12,node] targets')
    require(valid.shape == y.shape and valid.dtype == np.bool_ and valid.any(), 'Invalid shared target mask')
    require(np.isfinite(y).all() and (y[valid] > 0).all(), 'Finite targets and positive valid MAPE targets required')
    null = float(y.min()) if y.min() < 1 else 0.
    require(np.array_equal(valid, y != null), 'Mask differs from original global-null rule')
    for key, size in (('sample_ids', y.shape[0]), ('node_ids', y.shape[2])):
        ids = axes[key]
        require(ids.shape == (size,) and ids.dtype.kind in 'iu' and len(np.unique(ids)) == size,
            'Invalid unique integer axis: ' + key)
    require(axes['road_labels'].shape == (y.shape[2],) and axes['road_labels'].dtype.kind in 'US',
        'Invalid road axis')
    require(axes['direct_support'].shape == (y.shape[0], y.shape[2])
        and axes['direct_support'].dtype == np.bool_, 'Invalid direct support axis')
    return axes


def node_concentration(rows):
    valid_rows = [row for row in rows if row['count']]
    top_count = max(1, int(np.ceil(len(valid_rows) * .1)))
    result = dict(valid_nodes=len(valid_rows), top_node_count=top_count,
        actual_top_node_fraction=float(top_count / len(valid_rows)))
    totals = dict(energy=sum(row['shift_mse'] * row['count'] for row in valid_rows),
        worsening=sum(row['gross_mae_worsening_contribution'] for row in valid_rows))
    for name in ('energy', 'worsening'):
        value = (lambda row: row['shift_mse'] * row['count']) if name == 'energy' else (
            lambda row: row['gross_mae_worsening_contribution'])
        selected = sorted(valid_rows, key=lambda row: (-value(row), row['group']))[:top_count]
        prefix = 'top10pct_nodes_by_' + name
        result[prefix + '_share'] = sum(value(row) for row in selected) / totals[name] if totals[name] > 0 else None
        result[prefix + '_point_fraction'] = sum(row['point_fraction'] for row in selected)
        result[prefix + '_node_ids'] = [row['node_id'] for row in selected]
    return result


def write_csv(path, rows):
    require(bool(rows), 'Cannot write empty diagnostic table')
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value, digits=5):
    return 'NA' if value is None else f'{value:.{digits}f}'


def make_report(summary, tables):
    lines = ['# GW6 BEST/LAST 预测漂移诊断', '',
        f'共同新增训练预算：{summary["epochs"]} 轮；有效点：{summary["valid_count"]}；仅分析已有开发验证集预测。',
        '差值方向：LAST − BEST；signed bias = 预测 − 真实值。目标数值不推断物理单位。', '',
        '| 模型 | BEST轮 | BEST MAE | LAST MAE | MAE变化 | BEST偏差 | LAST偏差 | 平均预测变化 | 变化标准差 | 平移能量占比 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for arm, row in summary['arms'].items():
        values = [_fmt(row[key]) for key in (
            'best_mae', 'last_mae', 'mae_deterioration', 'best_bias', 'last_bias',
            'shift_mean', 'shift_std', 'global_shift_energy_fraction')]
        if row['shift_mse'] == 0:
            values[-1] = '无预测变化'
        lines.append('| ' + arm + ' | ' + str(row['best_epoch']) + ' | ' + ' | '.join(values) + ' |')
    lines += ['', '变化平方均值 = 平均变化的平方 + 变化方差；“平移能量占比”是前者在变化平方均值中的占比，绝不是 MAE 退化解释比例。越接近 1，预测变化越接近常量；它本身不证明根因。', '',
        '| 模型 | 同集均值对齐后 MAE | 对齐后偏差 | 对齐移除的MAE增量 | 对齐移除比例 | 负向预测变化点占比 | 点上MAE恶化总幅度 | 点上MAE改善总幅度 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm, row in summary['arms'].items():
        lines.append('| ' + arm + ' | ' + ' | '.join(_fmt(row[key]) for key in (
            'centered_last_mae', 'centered_last_bias', 'centered_removed_mae',
            'centered_removed_fraction', 'shift_negative_fraction', 'gross_mae_worsening',
            'gross_mae_improvement')) + ' |')
    lines += ['', '均值对齐仅将 LAST 减去总体 mean(LAST−BEST)，使用同一验证集拟合，不能作新模型成绩或部署修正。移除比例仅在原 MAE 变差时定义，保留超出 [0,1] 的值。', '']
    joint = summary['arms']['JOINT_MEMORY']
    lines += ['JOINT 预测变化分位数：' + '；'.join('P' + name[1:] + '=' + _fmt(joint['shift_' + name])
        for name in ('q01', 'q05', 'q25', 'q50', 'q75', 'q95', 'q99')) + '。', '',
        '| 模型 | 变化能量前10%节点的能量占比 | 这些节点的点占比 | MAE恶化前10%节点的恶化幅度占比 | 这些节点的点占比 |',
        '|---|---:|---:|---:|---:|']
    for arm, row in summary['arms'].items():
        concentration = row['node_concentration']
        lines.append('| ' + arm + ' | ' + ' | '.join(_fmt(concentration[key]) for key in (
            'top10pct_nodes_by_energy_share', 'top10pct_nodes_by_energy_point_fraction',
            'top10pct_nodes_by_worsening_share', 'top10pct_nodes_by_worsening_point_fraction')) + ' |')
    lines += ['', '两列前10%节点分别按变化能量和点上MAE恶化总幅度选择；同时参考点数占比，不设“已定位根因”的阈值。', '']

    def group_table(title, rows, label):
        lines.extend([title, '', '| 组 | 有效点 | BEST MAE | LAST MAE | 对齐后MAE | 平均变化 | 变化标准差 | BEST偏差 | LAST偏差 | 净MAE贡献 | 恶化幅度贡献 | 改善幅度贡献 |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|'])
        for row in rows:
            lines.append('| ' + str(row[label]) + ' | ' + str(row['count']) + ' | ' + ' | '.join(_fmt(row[key]) for key in (
                'best_mae', 'last_mae', 'centered_last_mae', 'shift_mean', 'shift_std',
                'best_bias', 'last_bias', 'net_mae_contribution',
                'gross_mae_worsening_contribution', 'gross_mae_improvement_contribution')) + ' |')
        lines.append('')

    group_table('JOINT：逐时距', [row for row in tables['by_horizon'] if row['arm'] == 'JOINT_MEMORY'], 'step')
    group_table('JOINT：gap 输出通道（(step−1)%3，0 起始）',
        [row for row in tables['by_gap_channel'] if row['arm'] == 'JOINT_MEMORY'], 'channel')
    group_table('JOINT：真实值分位数组（重复边界已合并）',
        [row for row in tables['by_target_bin'] if row['arm'] == 'JOINT_MEMORY'], 'interval')
    nodes = [row for row in tables['by_node'] if row['arm'] == 'JOINT_MEMORY' and row['count']]
    nodes.sort(key=lambda row: (-row['gross_mae_worsening_contribution'], row['group']))
    group_table('JOINT：点上MAE恶化总幅度贡献最高的10个节点', nodes[:10], 'node_id')
    lines += ['净MAE贡献 = 本组点上(|LAST−真实值|−|BEST−真实值|)之和 / 全部有效点数；各互斥分组可求和到总体MAE变化。恶化/改善幅度分开累计，不用净差作比例分母。', '',
        '解读时同时查看平均变化、变化标准差、均值对齐后的误差和分组贡献；全局能量占比与局部集中可以并存。这些端点统计不能确定是哪一层、哪个参数或第19轮何时导致漂移。', '',
        '限制：']
    lines.extend('- ' + item for item in LIMITATIONS)
    lines.extend(['', '完整四组明细：by_horizon.csv、by_gap_channel.csv、by_node.csv、by_target_bin.csv；全局统计见 summary.json，认证输入及哈希见 input_manifest.json。', ''])
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--best-dir', type=Path)
    parser.add_argument('--last-dir', type=Path)
    args = parser.parse_args(argv)
    run, output = args.run_dir.resolve(), args.output_dir.resolve()
    require(args.epochs > 0, 'Positive epoch budget required')
    require(output.parent == run and not output.exists(),
        'Use a new direct child output directory of --run-dir; existing inputs are preserved')
    identity, saved_summary, run_records = verify_run(run, args.epochs)
    best_dir, best_meta, best_record = select_eval(run, args.best_dir, 'best', identity, saved_summary, args.epochs)
    last_dir, last_meta, last_record = select_eval(run, args.last_dir, 'last', identity, saved_summary, args.epochs)
    endpoint_analysis = {'best': read_json(best_dir / 'analysis.json'),
        'last': read_json(last_dir / 'analysis.json')}
    require(output not in (best_dir, last_dir), 'Output cannot be an input evaluation directory')
    tables = {name: [] for name in ('by_horizon', 'by_gap_channel', 'by_node', 'by_target_bin')}
    result = dict(schema='gw6_paired_endpoint_drift_v1', status='GW6_DRIFT_ANALYSIS_COMPLETE',
        epochs=args.epochs, stage='joint', registered_arms=list(ARMS), arms={},
        optimizer_updates=0, new_inference_calls=0, test_evaluated=False, limitations=LIMITATIONS)
    best_path, last_path = best_dir / 'paired_predictions.npz', last_dir / 'paired_predictions.npz'
    with np.load(best_path, allow_pickle=False) as best_payload, np.load(last_path, allow_pickle=False) as last_payload:
        axes = load_axes(best_payload, identity)
        other_axes = load_axes(last_payload, identity)
        require_same_axes(axes, other_axes)
        del other_axes
        y_all, valid = axes['y'], axes['valid']
        shape = y_all.shape
        y = y_all[valid].astype(np.float64)
        result.update(valid_count=int(y.size), validation_rows=shape[0], horizon=shape[1], node_count=shape[2])
        labels_target, bounds = target_bins(y)
        result['target_bin_bounds'] = bounds.tolist()
        for arm in ARMS:
            # NpzFile remains lazy: historical predictions and models are never loaded.
            predictions = []
            for payload in (best_payload, last_payload):
                prediction = np.asarray(payload['pred__' + arm])
                require(prediction.shape == shape and prediction.dtype.kind == 'f'
                    and np.isfinite(prediction).all(), 'Invalid prediction values/axes: ' + arm)
                predictions.append(prediction[valid].astype(np.float64))
                del prediction
            best, last = predictions
            stats = pair_statistics(y, best, last)
            for role in ('best', 'last'):
                certified = endpoint_analysis[role].get('cases', {}).get(arm, {})
                for recomputed, recorded in (('mae', 'mae_pooled'), ('rmse', 'rmse_pooled'),
                        ('mape', 'mape_pooled'), ('bias', 'signed_bias')):
                    expected = certified.get(recorded)
                    require(isinstance(expected, (int, float)) and np.isfinite(expected)
                        and np.isclose(stats[role + '_' + recomputed], expected, atol=2e-5, rtol=1e-7),
                        'Recomputed archive metric differs from certified analysis: '
                        + role + '/' + arm + '/' + recomputed)
            stats.update(best_epoch=best_meta['checkpoint_roles'][arm]['epoch'],
                last_epoch=last_meta['checkpoint_roles'][arm]['epoch'])
            mu = stats['shift_mean']
            labels = np.broadcast_to(np.arange(12, dtype=np.int16)[None, :, None], shape)[valid]
            rows = group_statistics(labels, 12, y, best, last, mu)
            tables['by_horizon'].extend(dict(arm=arm, step=row['group'] + 1, **row) for row in rows)
            rows = group_statistics(labels % 3, 3, y, best, last, mu)
            tables['by_gap_channel'].extend(dict(arm=arm, channel=row['group'], **row) for row in rows)
            labels = np.broadcast_to(np.arange(shape[2], dtype=np.int32)[None, None, :], shape)[valid]
            rows = group_statistics(labels, shape[2], y, best, last, mu)
            for row in rows:
                index = row['group']
                row.update(node_id=int(axes['node_ids'][index]), road_label=str(axes['road_labels'][index]))
            stats['node_concentration'] = node_concentration(rows)
            tables['by_node'].extend(dict(arm=arm, **row) for row in rows)
            rows = group_statistics(labels_target, len(bounds) - 1, y, best, last, mu)
            for row in rows:
                index = row['group']
                row.update(lower=float(bounds[index]), upper=float(bounds[index + 1]),
                    upper_inclusive=index == len(bounds) - 2,
                    interval=f'[{bounds[index]:.8g}, {bounds[index + 1]:.8g}' + (']' if index == len(bounds) - 2 else ')'))
            tables['by_target_bin'].extend(dict(arm=arm, **row) for row in rows)
            result['arms'][arm] = stats
            del predictions, best, last, labels
    require(sha256(best_path) == best_meta['archive_sha256']
        and sha256(last_path) == last_meta['archive_sha256'], 'Prediction archive changed during analysis')
    for record in run_records.values():
        require(sha256(record['path']) == record['sha256'], 'Run document changed during analysis')
    for record in (best_record, last_record):
        for name in ('metadata', 'analysis'):
            require(sha256(Path(record['directory']) / (name + '.json')) == record[name + '_sha256'],
                'Producer document changed during analysis')
    manifest = dict(schema='gw6_paired_endpoint_drift_input_v1',
        created_utc=datetime.now(timezone.utc).isoformat(), run_dir=str(run), requested_epochs=args.epochs,
        run_identity_sha256=identity['identity_sha256'], inputs=run_records,
        best=best_record, last=last_record, shared_axes_exact_including_dtype=True,
        loaded_prediction_arms=list(ARMS), historical_predictions_loaded=False,
        optimizer_updates=0, new_inference_calls=0, test_evaluated=False, limitations=LIMITATIONS,
        diagnostic_script_sha256=sha256(Path(__file__)))
    report = make_report(result, tables)
    output.mkdir()  # All input checks and calculations finish before any output is created.
    write_json(output / 'input_manifest.json', manifest)
    write_json(output / 'summary.json', result)
    for name, rows in tables.items():
        write_csv(output / (name + '.csv'), rows)
    (output / 'report.md').write_text(report, encoding='utf-8')
    print(report)
    print(json.dumps(dict(status=result['status'], output_dir=str(output), report=str(output / 'report.md')),
        ensure_ascii=False), flush=True)
    return result


if __name__ == '__main__':
    main()
