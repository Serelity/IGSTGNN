"""M4.5: read completed P1 ON/OFF artifacts; no model, inference or training.

Original-row horizon-macro pooled MAE remains primary. Region-normalized gains
and contributions with the whole-network denominator are separate quantities.
"""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import platform
import uuid

import numpy as np


ATOL, RTOL = 1e-9, 1e-10  # Float64 artifact arithmetic; not M4.4 replay tolerances.
THRESHOLDS = (1e-4, 1e-3, 1e-2, .1, 1.)
QUANTILES = (0., .25, .5, .75, .9, .95, .99, 1.)
PARTITIONS = ('direct_report_nodes', 'potential_propagation_only_nodes',
              'outside_potential_supported_nodes', 'no_report_supported_samples')
ARRAY_KEYS = {'on_prediction', 'off_prediction', 'target', 'valid', 'sample_indices',
              'station_ids', 'report_association', 'direct_report_node_mask',
              'potential_component_mask', 'unique_cutoff_weights', 'target_windows_minutes'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    require(bool(rows), 'Empty output table')
    with Path(path).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def close(actual, expected, label):
    if actual is None or expected is None:
        require(actual is None and expected is None, 'Unavailable metric differs: '+label)
        return
    a, b = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
    require(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
            and np.allclose(a, b, atol=ATOL, rtol=RTOL), 'Artifact metric mismatch: '+label)


def snapshot(paths):
    return {str(path): sha256(path) for path in paths}


def verify_source(probe, allow_engineering):
    report = read_json(probe/'report.json')
    require(report.get('status') == 'P1_REPORT_CAPACITY_PROBE_PASS', 'Source probe did not complete successfully')
    for key in ('model_state_preserved', 'final_on_restored', 'original_IGSTGNN_incident_inputs_preserved'):
        require(report.get(key) is True, 'Source invariant absent/failed: '+key)
    require(report.get('optimizer_updates') == 0 and report.get('test_accessed') is False,
            'Source was not inference-only validation')
    manifest = report.get('outputs_sha256')
    required = {'paired_predictions.npz', 'samples.csv', 'replay.json', 'pathway_curves.npz'}
    if manifest:
        require(isinstance(manifest, dict) and required | {'invocation.json', 'checkpoint_audit.json'} <= set(manifest),
                'Incomplete completed-probe hash manifest')
        for name in manifest:
            require(Path(name).name == name and '/' not in name and '\\' not in name,
                    'Invalid source manifest path')
        paths = [probe/'report.json']+[probe/name for name in sorted(manifest)]
    else:
        require(allow_engineering and report.get('engineering_only') is True,
                'Source lacks a completed hash manifest; only explicitly marked engineering fixtures may opt in')
        paths = [probe/name for name in sorted(required | {'report.json'})]
    before = snapshot(paths)
    if manifest:
        for name, expected in manifest.items():
            require(before[str(probe/name)] == expected, 'Source artifact hash mismatch: '+name)
    replay = read_json(probe/'replay.json')
    require(replay.get('matching') is True and report.get('replay') == replay,
            'Completed source replay differs from replay.json')
    invocation = read_json(probe/'invocation.json') if (probe/'invocation.json').is_file() else None
    if invocation is not None:
        require(invocation.get('test_accessed') is False and invocation.get('optimizer_updates') == 0,
                'Source invocation is not validation inference-only')
        require(report.get('protected_artifacts_unchanged') is True, 'Source protected-artifact audit failed')
    return report, invocation, before


def load_arrays(probe, report):
    with np.load(probe/'paired_predictions.npz', allow_pickle=False) as packed:
        require(set(packed.files) == ARRAY_KEYS, 'Unexpected paired-prediction schema')
        arrays = {key: packed[key] for key in packed.files}
    on, off, target, valid = (arrays[k] for k in ('on_prediction', 'off_prediction', 'target', 'valid'))
    require(on.ndim == 4 and on.shape == off.shape == target.shape == valid.shape
            and on.shape[0] > 0 and on.shape[1] == 12 and on.shape[2] > 0 and on.shape[3] == 1,
            'Expected paired [rows,12,stations,1] axes')
    require(valid.dtype == np.bool_ and all(a.dtype.kind == 'f' for a in (on, off, target))
            and np.isfinite(on).all() and np.isfinite(off).all() and np.isfinite(target[valid]).all(),
            'Invalid paired predictions/targets/mask')
    rows, _, nodes, _ = on.shape
    require(report.get('validation_rows') == rows, 'Source validation-row count differs')
    engineering = bool(report.get('engineering_only') or report.get('check_subset'))
    require(engineering or (rows == 917 and nodes == 496), 'Full source must preserve 917 rows and 496 stations')
    for name, size in (('sample_indices', rows), ('station_ids', nodes)):
        value = arrays[name]
        require(value.shape == (size,) and value.dtype.kind in 'iu' and len(np.unique(value)) == size,
                'Invalid identity axis: '+name)
    expected_windows = np.stack((np.arange(5., 65., 5.), np.arange(10., 70., 5.)), -1)
    require(np.array_equal(arrays['target_windows_minutes'], expected_windows), 'Target windows changed')
    association = arrays['report_association']
    require(association.ndim == 2 and association.shape[0] == rows and np.isfinite(association).all()
            and ((association >= 0) & (association <= 1)).all(), 'Invalid report association')
    for key in ('direct_report_node_mask', 'potential_component_mask'):
        require(arrays[key].shape == (rows, nodes) and arrays[key].dtype == np.bool_, 'Invalid region mask: '+key)
    with np.load(probe/'pathway_curves.npz', allow_pickle=False) as curves:
        edges = curves['edge_index']
        require(np.array_equal(curves['station_ids'], arrays['station_ids']), 'Pathway station axis differs')
        require(np.array_equal(curves['target_windows_minutes'], expected_windows), 'Pathway windows differ')
    require(edges.shape == (2, association.shape[1]) and edges.dtype.kind in 'iu'
            and ((edges >= -1) & (edges < nodes)).all(), 'Invalid pathway edge identity')
    internal = (edges >= 0).all(0)
    require(not (association[:, ~internal] > 0).any(), 'Report association targets an external edge')
    direct = np.zeros((rows, nodes), dtype=bool)
    adjacency = [[] for _ in range(nodes)]
    for j in np.flatnonzero(internal):
        src, dst = edges[:, j]
        direct[:, src] |= association[:, j] > 0
        direct[:, dst] |= association[:, j] > 0
        adjacency[src].append(dst)
        adjacency[dst].append(src)
    potential = np.zeros_like(direct)
    visited = set()
    for node in range(nodes):
        if node in visited:
            continue
        component, pending = [], [node]
        visited.add(node)
        while pending:
            member = pending.pop()
            component.append(member)
            for other in adjacency[member]:
                if other not in visited:
                    visited.add(other)
                    pending.append(other)
        potential[:, component] = direct[:, component].any(1)[:, None]
    require(np.array_equal(direct, arrays['direct_report_node_mask'])
            and np.array_equal(potential, arrays['potential_component_mask']), 'Saved graph-derived region masks differ')
    supported = association.any(1)
    require(np.array_equal(on[~supported], off[~supported]), 'Unsupported samples changed under report intervention')
    arrays['edge_index'] = edges
    return arrays


def network_regions(network, invocation, arrays):
    require(network is not None, 'Network metadata unavailable; supply --network-dir')
    network = Path(network).resolve()
    summary = read_json(network/'summary.json')
    identity = {} if invocation is None else invocation['origin_identity']
    expected = identity.get('inputs_sha256', {}).get('network/summary.json')
    if invocation is not None:
        require(expected == sha256(network/'summary.json'), 'Network summary differs from source invocation')
    candidates = (['graph.json' if identity['ramp_exchanges'] else 'conservative_graph.json']
                  if 'ramp_exchanges' in identity else ['graph.json', 'conservative_graph.json'])
    chosen = None
    for name in candidates:
        if not (network/name).is_file():
            continue
        graph = read_json(network/name)
        edges = np.array([[row['source'], row['destination']] for row in graph['edges']], dtype=np.int64).T
        if np.array_equal(edges, arrays['edge_index']) and np.array_equal(graph['station_ids'], arrays['station_ids']):
            chosen = name, graph
            break
    require(chosen is not None, 'Network graph does not match saved edge/station axes')
    graph_name, graph = chosen
    names = ('summary.json', graph_name, 'common_structure_mask.npy', 'stations.csv', 'roads.csv')
    hashes = snapshot([network/name for name in names])
    for name in names[1:]:
        require(summary.get('outputs_sha256', {}).get(name) == hashes[str(network/name)],
                'Network metadata hash mismatch: '+name)
    nodes = len(arrays['station_ids'])
    candidate, boundary = (np.asarray(graph[k], dtype=bool) for k in ('operator_mask', 'boundary_nodes'))
    common = np.load(network/'common_structure_mask.npy', allow_pickle=False)
    require(candidate.shape == boundary.shape == common.shape == (nodes,) and common.dtype == np.bool_
            and not (common & ~candidate).any(), 'Network region masks differ')
    if identity:
        require(int(candidate.sum()) == identity['candidate_structure_nodes'], 'Network candidate coverage changed')
    rows, roads = read_csv(network/'stations.csv'), read_csv(network/'roads.csv')
    require([int(row['station_id']) for row in rows] == arrays['station_ids'].tolist()
            and [int(row['node_index']) for row in rows] == list(range(nodes)), 'Station metadata axes differ')
    regions = dict(all_nodes=np.ones(nodes, bool), common_structure=common,
                   added_structure=candidate & ~common, candidate_structure=candidate,
                   outside_candidate=~candidate, candidate_boundary=boundary)
    require(len({row['road'] for row in roads}) == len(roads)
            and {row['road'] for row in roads} == {row['road'] for row in rows}, 'Road metadata groups differ')
    for road in roads:
        mask = np.array([row['road'] == road['road'] for row in rows])
        require(mask.sum() == int(road['stations']), 'Road metadata count differs')
        regions['road_'+road['road']] = mask
    return regions, hashes


def load_samples(probe, arrays, report):
    rows = read_csv(probe/'samples.csv')
    require(len(rows) == len(arrays['sample_indices']) and
            [int(row['sample_index']) for row in rows] == arrays['sample_indices'].tolist(), 'Sample CSV order/identity differs')
    ages = []
    direct, potential, association = (arrays[k] for k in
        ('direct_report_node_mask', 'potential_component_mask', 'report_association'))
    for i, row in enumerate(rows):
        require(bool(row['incident_id']) and bool(row['t0']), 'Missing trigger/cutoff identity')
        datetime.fromisoformat(row['t0'])
        matched, count = int(row['matched_report_entities']), int(row['report_entities'])
        require(0 <= matched <= count and bool(matched) == bool(association[i].any()), 'Report support counts differ')
        require(int(row['associated_edges']) == int((association[i] > 0).sum())
                and int(row['associated_nodes']) == int(direct[i].sum())
                and int(row['potential_component_nodes']) == int(potential[i].sum()), 'Sample region counts differ')
        value = row['youngest_matched_recorded_report_age_minutes']
        age = float(value) if value else np.nan
        require((matched > 0 and np.isfinite(age) and 0 <= age <= 60)
                or (matched == 0 and np.isnan(age)), 'Invalid recorded report age/support')
        ages.append(age)
    cutoff_counts = Counter(row['t0'] for row in rows)
    weights = np.array([1./cutoff_counts[row['t0']] for row in rows])
    close(arrays['unique_cutoff_weights'], weights, 'NPZ cutoff weights')
    close([float(row['unique_cutoff_sensitivity_weight']) for row in rows], weights, 'CSV cutoff weights')
    require(report.get('distinct_cutoffs') == len(cutoff_counts), 'Distinct cutoff count differs')
    return rows, np.asarray(ages), weights


def raw_statistics(errors, valid, mask):
    """Keep row x horizon sufficient statistics for fast event/cutoff grouping."""
    mask = np.broadcast_to(mask, (valid.shape[0], valid.shape[2]))
    selected = valid & mask[:, None, :, None]
    result = {'count': selected.sum((2, 3)).astype(np.float64)}
    for name, value in errors.items():
        result[name] = np.where(selected, value, 0).sum((2, 3))
    change = errors['change']
    result['maximum'] = np.where(selected, change, 0).max((1, 2, 3))
    for threshold in THRESHOLDS:
        result['above_'+str(threshold)] = (selected & (change > threshold)).sum((2, 3)).astype(np.float64)
    return result


def aggregate(raw, global_counts=None, indices=None, weights=None):
    indices = np.arange(len(raw['count'])) if indices is None else np.asarray(indices)
    w = np.ones(len(indices)) if weights is None else np.asarray(weights)[indices]
    require(w.shape == (len(indices),) and np.isfinite(w).all() and (w > 0).all(), 'Invalid aggregation weights')
    sums = {name: (value[indices]*w[:, None]).sum(0) for name, value in raw.items() if name != 'maximum'}
    count = sums['count']
    available = bool((count > 0).all())
    ratio = lambda x: np.divide(x, count, out=np.zeros_like(x), where=count > 0)
    def per_horizon(value):
        return [float(x) if c > 0 else None for x, c in zip(ratio(value), count)]
    def macro(value):
        return float(ratio(value).mean()) if available else None
    positive, negative, gain = (macro(sums[key]) for key in ('positive', 'negative', 'gain'))
    response = macro(sums['change'])
    require(np.all(ratio(sums['positive']+sums['negative']) <= ratio(sums['change'])+ATOL),
            'Absolute-error triangle bound failed')
    if available:
        close(gain, positive-negative, 'positive/negative gain identity')
    contribution = per_contribution = None
    if global_counts is not None and (np.asarray(global_counts) > 0).all():
        per_contribution = (sums['gain']/global_counts).tolist()
        contribution = float(np.mean(per_contribution))
    gross = None if positive is None else positive+negative
    return dict(region_metric_available=available, on_mae_macro=macro(sums['on_error']),
        off_mae_macro=macro(sums['off_error']), off_minus_on_mae=gain,
        relative_gain_percent=None if not available or macro(sums['off_error']) == 0 else
            100.*gain/macro(sums['off_error']),
        gain_positive=positive, gain_negative=negative,
        cancellation_fraction=None if gross is None or gross == 0 else max(0., 1.-abs(gain)/gross),
        global_gain_contribution=contribution, per_horizon_global_gain_contribution=per_contribution,
        valid_count=int(raw['count'][indices].sum()), weighted_valid_count=float(count.sum()),
        per_horizon_valid_count=raw['count'][indices].sum(0).astype(int).tolist(),
        per_horizon_weighted_count=count.tolist(),
        per_horizon_on_mae=per_horizon(sums['on_error']), per_horizon_off_mae=per_horizon(sums['off_error']),
        per_horizon_off_minus_on=per_horizon(sums['gain']),
        per_horizon_gain_positive=per_horizon(sums['positive']),
        per_horizon_gain_negative=per_horizon(sums['negative']),
        per_horizon_prediction_abs_change_mean=per_horizon(sums['change']),
        prediction_abs_change_horizon_macro_mean=response,
        prediction_abs_change_pooled_mean=None if count.sum() == 0 else float(sums['change'].sum()/count.sum()),
        prediction_abs_change_max=None if raw['count'][indices].sum() == 0 else float(raw['maximum'][indices].max()),
        response_fraction_above={str(t): macro(sums['above_'+str(t)]) for t in THRESHOLDS})


def compare_saved_metric(actual, expected, label):
    if expected is None:
        require(not actual['region_metric_available'], 'Source empty-region metric changed: '+label)
        return
    require(actual['region_metric_available'], 'Source available-region metric became unavailable: '+label)
    keys = ('on_mae_macro', 'off_mae_macro', 'off_minus_on_mae', 'valid_count',
            'per_horizon_on_mae', 'per_horizon_off_mae', 'per_horizon_off_minus_on',
            'per_horizon_valid_count', 'per_horizon_weighted_count', 'prediction_abs_change_max')
    for key in keys:
        close(actual[key], expected[key], label+'.'+key)
    close(actual['prediction_abs_change_pooled_mean'], expected['prediction_abs_change_mean'],
          label+'.prediction_abs_change_mean (legacy pooled)')


def group_table(samples, key, raw, global_counts):
    grouped = defaultdict(list)
    for index, row in enumerate(samples):
        grouped[row[key]].append(index)
    table = []
    for value, indices in sorted(grouped.items()):
        metric = aggregate(raw['all_nodes'], global_counts, indices)
        row = dict(group=value, rows=len(indices), distinct_cutoffs=len({samples[i]['t0'] for i in indices}),
            distinct_trigger_ids=len({samples[i]['incident_id'] for i in indices}),
            report_supported_rows=sum(int(samples[i]['matched_report_entities']) > 0 for i in indices),
            exact_prediction_noop_on_valid_targets_rows=int((raw['all_nodes']['maximum'][indices] == 0).sum()),
            rows_with_change_above_1e_4=int((raw['all_nodes']['maximum'][indices] > 1e-4).sum()))
        for name in ('on_mae_macro', 'off_mae_macro', 'off_minus_on_mae', 'relative_gain_percent',
                     'gain_positive', 'gain_negative', 'cancellation_fraction', 'global_gain_contribution',
                     'valid_count', 'prediction_abs_change_horizon_macro_mean', 'prediction_abs_change_max'):
            row[name] = metric[name]
        for region in PARTITIONS:
            row[region+'_global_contribution'] = aggregate(raw[region], global_counts, indices)['global_gain_contribution']
        table.append(row)
    return table


def decomposition_rows(metrics):
    rows = []
    for name, metric in metrics.items():
        category = 'partition' if name in PARTITIONS else ('road' if name.startswith('road_') else 'diagnostic')
        rows.append(dict(category=category, region=name, horizon='macro', window_start_minutes=None,
            window_end_minutes=None, valid_count=metric['valid_count'], on_mae=metric['on_mae_macro'],
            off_mae=metric['off_mae_macro'], off_minus_on=metric['off_minus_on_mae'],
            gain_positive=metric['gain_positive'], gain_negative=metric['gain_negative'],
            global_gain_contribution=metric['global_gain_contribution'],
            response_mean=metric['prediction_abs_change_horizon_macro_mean']))
        for h in range(12):
            rows.append(dict(category=category, region=name, horizon=h+1, window_start_minutes=5*(h+1),
                window_end_minutes=5*(h+2), valid_count=metric['per_horizon_valid_count'][h],
                on_mae=metric['per_horizon_on_mae'][h], off_mae=metric['per_horizon_off_mae'][h],
                off_minus_on=metric['per_horizon_off_minus_on'][h],
                gain_positive=metric['per_horizon_gain_positive'][h],
                gain_negative=metric['per_horizon_gain_negative'][h],
                global_gain_contribution=metric['per_horizon_global_gain_contribution'][h],
                response_mean=metric['per_horizon_prediction_abs_change_mean'][h]))
    return rows


def explain(result, events):
    metric = result['metrics']['all_nodes']
    lines = ['# M4.5 P1报告预测增益分解', '',
        '源结果范围：'+result['source_scientific_scope']+'。本工具只读已有产物，没有训练或推理。', '',
        f"全网 ON MAE={metric['on_mae_macro']:.12g}；OFF MAE={metric['off_mae_macro']:.12g}。",
        f"OFF−ON={metric['off_minus_on_mae']:.12g}；正向项={metric['gain_positive']:.12g}；负向项={metric['gain_negative']:.12g}。",
        f"预测绝对变化的时距宏平均={metric['prediction_abs_change_horizon_macro_mean']:.12g}；最大值={metric['prediction_abs_change_max']:.12g}。", '',
        '| 互斥区域 | 区域内OFF−ON | 对全网收益的贡献K |', '|---|---:|---:|']
    for name in PARTITIONS:
        row = result['metrics'][name]
        gain = '不可用' if row['off_minus_on_mae'] is None else f"{row['off_minus_on_mae']:.12g}"
        lines.append(f"| {name} | {gain} | {row['global_gain_contribution']:.12g} |")
    lines += ['', '区域贡献使用全网分母，四区域可加和；道路、年龄等交叉表不能再相加。',
        '区域没有有效目标时区域MAE为null；全网分母有效时空区域贡献为0。',
        '抵消率在总正负收益为0时为null；正负改善是预测误差变化，不是事故真实因果效果。',
        '触发事件与截止点汇总保留全部原预测条件；并发报告不逐条归因，重叠窗口及共源站点不视为独立重复。',
        '唯一截止点/触发事件逆行数加权是敏感性结果，缺测不一致时不等于先算每组MAE再等权。',
        '绝对变化阈值只是原目标数值尺度上的描述；没有显著性检验、自动架构选择或增益成功声明。', '',
        '## 对称案例（按触发事件全网净MAE变化排序，各最多5个）', '']
    for title, positive in (('改善', True), ('恶化', False)):
        selected = [r for r in events if r['off_minus_on_mae'] is not None and
                    (r['off_minus_on_mae'] > 0 if positive else r['off_minus_on_mae'] < 0)]
        selected.sort(key=lambda r: r['off_minus_on_mae'], reverse=positive)
        lines.append(title+'：'+('; '.join(f"{r['group']} ({r['off_minus_on_mae']:.8g})" for r in selected[:5]) or '无'))
    return '\n'.join(lines)+'\n'


def analyze(probe, output, network=None, allow_engineering=False):
    probe, output = Path(probe).resolve(), Path(output).resolve()
    require(output != probe and not output.is_relative_to(probe) and not probe.is_relative_to(output),
            'Output overlaps the protected probe directory')
    output.mkdir(parents=True, exist_ok=False)
    before = {}
    try:
        report, invocation, before = verify_source(probe, allow_engineering)
        arrays = load_arrays(probe, report)
        if network is None and invocation is not None:
            network = invocation.get('directories', {}).get('network')
        regions, network_hashes = network_regions(network, invocation, arrays)
        before.update(network_hashes)
        samples, ages, cutoff_weights = load_samples(probe, arrays, report)
        direct, potential = arrays['direct_report_node_mask'], arrays['potential_component_mask']
        supported = arrays['report_association'].any(1)
        shape = direct.shape
        partitions = dict(direct_report_nodes=direct, potential_propagation_only_nodes=potential & ~direct,
            outside_potential_supported_nodes=~potential & supported[:, None],
            no_report_supported_samples=np.broadcast_to(~supported[:, None], shape))
        require(np.all(sum(mask.astype(np.int8) for mask in partitions.values()) == 1), 'Regions do not form a partition')
        regions.update(partitions)
        regions.update(report_associated_nodes=direct, no_direct_report_association_nodes=~direct,
            outside_potential_propagation_nodes=~potential,
            report_supported_samples=np.broadcast_to(supported[:, None], shape))
        for name, low, high in (('matched_report_age_0_5', 0, 5), ('matched_report_age_5_15', 5, 15),
                                ('matched_report_age_15_60', 15, 61)):
            regions[name] = np.broadcast_to(((ages >= low) & (ages < high))[:, None], shape)
        target, valid = arrays['target'], arrays['valid']
        on_error = np.abs(arrays['on_prediction'].astype(np.float64)-target)
        off_error = np.abs(arrays['off_prediction'].astype(np.float64)-target)
        gain = off_error-on_error
        change = np.abs(arrays['off_prediction'].astype(np.float64)-arrays['on_prediction'])
        require(np.all(np.abs(gain[valid]) <= change[valid]+ATOL), 'Pointwise triangle bound failed')
        errors = dict(on_error=on_error, off_error=off_error, gain=gain,
                      positive=np.maximum(gain, 0), negative=np.maximum(-gain, 0), change=change)
        raw = {name: raw_statistics(errors, valid, mask) for name, mask in regions.items()}
        global_counts = raw['all_nodes']['count'].sum(0)
        require((global_counts > 0).all(), 'Source all-node main metric unavailable')
        metrics = {name: aggregate(value, global_counts) for name, value in raw.items()}
        for name, metric in metrics.items():
            mask = np.broadcast_to(regions[name], shape)
            values = change[valid & mask[:, None, :, None]]
            metric['prediction_abs_change_quantiles_unweighted'] = (None if not len(values) else
                dict(zip(map(str, QUANTILES), map(float, np.quantile(values, QUANTILES)))))
        weighted_counts = (raw['all_nodes']['count']*cutoff_weights[:, None]).sum(0)
        cutoff_sensitivity = {name: aggregate(value, weighted_counts, weights=cutoff_weights) for name, value in raw.items()}
        for name, expected in report['metrics'].items():
            require(name in metrics, 'Source metric region cannot be reconstructed: '+name)
            compare_saved_metric(metrics[name], expected, name)
            compare_saved_metric(cutoff_sensitivity[name], report['unique_cutoff_sensitivity_metrics'][name], 'cutoff:'+name)
        for i, row in enumerate(samples):
            metric = aggregate(raw['all_nodes'], indices=[i])
            for column, key in (('on_mae', 'on_mae_macro'), ('off_mae', 'off_mae_macro'), ('off_minus_on_mae', 'off_minus_on_mae')):
                close(metric[key], float(row[column]) if row[column] else None, f'sample {i}:'+column)
        close(sum(metrics[name]['global_gain_contribution'] for name in PARTITIONS),
              metrics['all_nodes']['off_minus_on_mae'], 'partition contribution sum')
        close(np.sum([metrics[name]['per_horizon_global_gain_contribution'] for name in PARTITIONS], 0),
              metrics['all_nodes']['per_horizon_off_minus_on'], 'per-horizon contribution sum')
        coverage = report['coverage']
        for key, value in dict(rows_with_matched_reports=int(supported.sum()),
                directly_associated_sample_node_positions=int(direct.sum()), sample_node_positions=int(direct.size),
                supported_sample_edge_positions=int((arrays['report_association'] > 0).sum()),
                matched_report_row_occurrences=sum(int(row['matched_report_entities']) for row in samples),
                report_row_occurrences=sum(int(row['report_entities']) for row in samples)).items():
            require(coverage[key] == value, 'Source coverage mismatch: '+key)
        events = group_table(samples, 'incident_id', raw, global_counts)
        cutoffs = group_table(samples, 't0', raw, global_counts)
        for table in (events, cutoffs):
            close(sum(row['global_gain_contribution'] for row in table),
                  metrics['all_nodes']['off_minus_on_mae'], 'group contribution sum')
        trigger_counts = Counter(row['incident_id'] for row in samples)
        trigger_weights = np.array([1./trigger_counts[row['incident_id']] for row in samples])
        trigger_global_counts = (raw['all_nodes']['count']*trigger_weights[:, None]).sum(0)
        engineering = bool(report.get('engineering_only') or report.get('check_subset'))
        result = dict(status='P1_REPORT_GAIN_ANALYSIS_PASS', source_probe=str(probe),
            source_scientific_scope='ENGINEERING_ONLY' if engineering else 'EXPLORATORY_CHECKPOINT_SENSITIVITY',
            source_best_epoch=report.get('best_epoch'), validation_rows=len(samples), stations=shape[1],
            distinct_trigger_ids=len(events), distinct_cutoffs=len(cutoffs), independent_incident_count=None,
            primary_weighting='original_rows_equal_weight_horizon_macro_pooled_MAE',
            relative_gain_denominator='OFF_MAE_for_same_checkpoint_intervention',
            difference_sign='OFF_MAE_minus_ON_MAE; positive_means_ON_better',
            metrics=metrics, unique_cutoff_sensitivity_metrics=cutoff_sensitivity,
            inverse_trigger_row_count_sensitivity_metrics={name: aggregate(value, trigger_global_counts, weights=trigger_weights)
                for name, value in raw.items()},
            additive_partition=list(PARTITIONS), source_metrics_reproduced=True, sample_csv_metrics_reproduced=True,
            partition_contributions_add_up=True, triangle_bounds_passed=True,
            input_sha256=before, source_artifacts_preserved=True,
            source_verification='completed_hash_manifest' if report.get('outputs_sha256') else 'explicit_engineering_fixture',
            statistical_independence_claim=False, predictive_gain_claim=False, causal_claim=False,
            true_physical_capacity_claim=False, test_accessed=False, optimizer_updates=0, new_inference_calls=0,
            report_age_semantics=report['report_age_semantics'], potential_region_semantics=report['potential_region_semantics'],
            matched_report_event_attribution_available=False,
            numerical_tolerances=dict(atol=ATOL, rtol=RTOL), response_thresholds_descriptive_only=list(THRESHOLDS),
            runtime=dict(python=platform.python_version(), numpy=np.__version__, host=platform.node()),
            script_sha256=sha256(Path(__file__)))
        write_csv(output/'gain_decomposition.csv', decomposition_rows(metrics))
        write_csv(output/'trigger_event_summary.csv', events)
        write_csv(output/'cutoff_summary.csv', cutoffs)
        (output/'README.md').write_text(explain(result, events), encoding='utf-8')
        require(snapshot([Path(path) for path in before]) == before, 'Read-only input artifacts changed during analysis')
        result['outputs_sha256'] = {p.name: sha256(p) for p in output.iterdir() if p.is_file()}
        write_json(output/'analysis.json', result)
        return result
    except BaseException as error:
        try:
            unchanged = snapshot([Path(path) for path in before]) == before if before else None
        except OSError:
            unchanged = False
        write_json(output/'failure.json', dict(status='P1_REPORT_GAIN_ANALYSIS_FAILED', error_type=type(error).__name__,
            error=str(error), source_artifacts_preserved=unchanged, completion_claim=False,
            test_accessed=False, optimizer_updates=0, new_inference_calls=0))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe-dir', type=Path, required=True)
    parser.add_argument('--network-dir', type=Path, help='Default: network directory recorded by the completed probe')
    parser.add_argument('--output-dir', type=Path, help='New directory outside the protected source probe')
    parser.add_argument('--allow-engineering-input', action='store_true',
                        help='Allow explicitly marked local engineering fixtures lacking a completed hash manifest')
    args = parser.parse_args(argv)
    probe = args.probe_dir.resolve()
    output = args.output_dir or probe.parent/('report_gain_analysis_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6])
    print('Source probe: '+str(probe)+'\nAnalysis directory: '+str(output.resolve()), flush=True)
    result = analyze(probe, output, args.network_dir, args.allow_engineering_input)
    print(json.dumps(dict(status=result['status'], source_scope=result['source_scientific_scope'],
        validation_rows=result['validation_rows'], source_best_epoch=result['source_best_epoch'],
        all_nodes={key: result['metrics']['all_nodes'][key] for key in ('on_mae_macro', 'off_mae_macro',
            'off_minus_on_mae', 'gain_positive', 'gain_negative', 'cancellation_fraction',
            'prediction_abs_change_horizon_macro_mean', 'prediction_abs_change_max', 'response_fraction_above')},
        partition_contributions={name: result['metrics'][name]['global_gain_contribution'] for name in PARTITIONS},
        source_artifacts_preserved=True, new_inference_calls=0, analysis=str(output.resolve()/'analysis.json')),
        indent=2, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
