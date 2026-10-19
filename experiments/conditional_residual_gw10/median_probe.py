"""Train-only conditional-median residual tables; NumPy and stdlib only.

All thresholds and corrections are fitted from the supplied training arrays.
Prediction accepts origin features only. Author-order sample blocks are a
support safeguard, not independent observations or certified calendar blocks.
"""
import hashlib
import json
from pathlib import Path

import numpy as np


SCHEMA = 'igstgnn_gw10_conditional_median_probe_v1'
ARMS = ('GLOBAL', 'HORIZON', 'NODE_HORIZON', 'ORIGIN',
    'OWN2', 'NEIGHBOR2', 'OWN4', 'JOINT')
FLOW_EDGES = (-float('inf'), 0., 5., 20., 50., 100., 200., 314., float('inf'))
MIN_POINTS = 64
MIN_BLOCKS = 3
SHRINKAGE = 128
BLOCK_SIZE = 128
FEATURES = ('last_flow', 'mean_abs_change', 'neighbor_change_disagreement')
CHILD_STATES = {'OWN2': 2, 'NEIGHBOR2': 2, 'OWN4': 4, 'JOINT': 4}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _features(features, nodes=None):
    _require(isinstance(features, dict) and all(name in features for name in FEATURES),
        'Three origin feature arrays are required')
    _require(all(np.asarray(features[name]).dtype.kind in 'fiu' for name in FEATURES),
        'Origin features must be real numeric arrays')
    arrays = {name: np.asarray(features[name], dtype=np.float64) for name in FEATURES}
    shape = arrays['last_flow'].shape
    _require(len(shape) == 2 and min(shape) > 0
        and all(value.shape == shape for value in arrays.values()),
        'Expected matching [sample,node] origin features')
    _require(nodes is None or shape[1] == nodes, 'Origin node axis differs from fitted model')
    _require(np.isfinite(arrays['last_flow']).all()
        and np.isfinite(arrays['mean_abs_change']).all(),
        'last_flow and mean_abs_change must be finite')
    return arrays


def _cutpoints(values):
    finite = np.asarray(values)[np.isfinite(values)]
    if not finite.size:
        return dict(quartiles=None, median=None, finite_count=0)
    quartiles = [float(value) for value in np.quantile(finite, [.25, .5, .75])]
    return dict(quartiles=quartiles, median=quartiles[1], finite_count=int(finite.size))


def _bins(features, cutpoints):
    flow = np.searchsorted(np.asarray(FLOW_EDGES[1:-1]), features['last_flow'], side='right')
    own = features['mean_abs_change']
    own2 = (own >= cutpoints['mean_abs_change']['median']).astype(np.int8)
    own4 = np.searchsorted(cutpoints['mean_abs_change']['quartiles'], own, side='right')
    neighbor = features['neighbor_change_disagreement']
    available = np.isfinite(neighbor)
    neighbor_median = cutpoints['neighbor_change_disagreement']['median']
    neighbor2 = np.zeros(neighbor.shape, dtype=np.int8)
    if neighbor_median is None:
        available = np.zeros(neighbor.shape, dtype=np.bool_)
    else:
        neighbor2[available] = neighbor[available] >= neighbor_median
    return flow, dict(OWN2=own2, NEIGHBOR2=neighbor2,
        OWN4=own4, JOINT=own2*2+neighbor2), available


def _cell(values, blocks, parent=None):
    count = int(values.size)
    block_count = int(np.unique(blocks).size)
    eligible = count >= MIN_POINTS and block_count >= MIN_BLOCKS
    if not eligible:
        _require(parent is not None, 'GLOBAL requires 64 valid points from at least three author-order blocks')
        return float(parent), count, block_count
    median = float(np.median(values))
    if parent is None:
        return median, count, block_count
    weight = count/(count+SHRINKAGE)
    # Equivalent shrinkage, preserving a constant parent/median exactly.
    return float(parent)+weight*(median-float(parent)), count, block_count


def _groups(values, blocks, keys, parents):
    """Sort one node/horizon shard once; do not scan the whole cube per cell."""
    corrections = np.asarray(parents, dtype=np.float64).copy()
    counts = np.zeros(corrections.shape, dtype=np.int64)
    block_counts = np.zeros(corrections.shape, dtype=np.int64)
    selected = keys >= 0
    if not np.any(selected):
        return corrections, counts, block_counts
    local_keys = keys[selected]
    order = np.argsort(local_keys, kind='stable')
    sorted_keys = local_keys[order]
    sorted_values, sorted_blocks = values[selected][order], blocks[selected][order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_keys))+1]
    stops = np.r_[starts[1:], len(order)]
    for start, stop in zip(starts, stops):
        key = int(sorted_keys[start])
        corrections[key], counts[key], block_counts[key] = _cell(
            sorted_values[start:stop], sorted_blocks[start:stop], corrections[key])
    return corrections, counts, block_counts


def _stats(counts, blocks, total_valid, total_blocks):
    counts, blocks = np.asarray(counts), np.asarray(blocks)
    eligible = (counts >= MIN_POINTS) & (blocks >= MIN_BLOCKS)
    assigned = int(counts.sum())
    fallback = int(counts[~eligible].sum())+(total_valid-assigned)
    observed_blocks = blocks[counts > 0]
    return dict(total_cells=int(counts.size), observed_cells=int(np.count_nonzero(counts)),
        eligible_cells=int(np.count_nonzero(eligible)),
        fallback_cells=int(counts.size-np.count_nonzero(eligible)),
        valid_count=assigned, block_count=int(total_blocks),
        cell_block_count_min=int(observed_blocks.min()) if observed_blocks.size else None,
        cell_block_count_max=int(observed_blocks.max()) if observed_blocks.size else None,
        missing_feature_valid_count=int(total_valid-assigned),
        fallback_valid_count=fallback,
        block_count_interpretation='global_training_valid_blocks; per_cell_counts_in_support; not_calendar_or_independent')


def fit_probe(pred, y, valid, features, sample_ids):
    """Fit all fixed arms using *training* labels only; return JSON-safe data.

    Inputs pred/y/valid are [sample,horizon,node]. Residual sign is y-pred,
    hence the returned correction is added to the unchanged prediction.
    No validation arrays, label-derived cutpoints or tuning arguments exist.
    """
    prediction, truth, mask = np.asarray(pred), np.asarray(y), np.asarray(valid)
    _require(prediction.shape == truth.shape == mask.shape and truth.ndim == 3
        and min(truth.shape) > 0 and mask.dtype == np.bool_ and mask.any(),
        'Expected matching nonempty [sample,horizon,node] arrays and Boolean validity mask')
    for name, value in (('prediction', prediction), ('truth', truth)):
        _require(value.dtype.kind in 'fiu' and np.isfinite(value[mask]).all(),
            'Selected '+name+' values must be finite real numbers')
    samples, horizon, nodes = truth.shape
    arrays = _features(features, nodes)
    _require(arrays['last_flow'].shape[0] == samples, 'Training feature sample axis differs')
    ids = np.asarray(sample_ids)
    _require(ids.shape == (samples,) and ids.dtype.kind in 'iu' and (ids >= 0).all()
        and np.unique(ids).size == samples, 'Unique nonnegative author-order sample IDs required')
    blocks = ids//BLOCK_SIZE
    broadcast_blocks = np.broadcast_to(blocks[:, None, None], truth.shape)
    residual = truth.astype(np.float64)-prediction.astype(np.float64)
    selected_blocks = broadcast_blocks[mask]
    total_valid, total_blocks = int(mask.sum()), int(np.unique(selected_blocks).size)
    cutpoints = {name: _cutpoints(arrays[name]) for name in FEATURES[1:]}
    flow, states, neighbor_available = _bins(arrays, cutpoints)
    global_delta, global_n, global_b = _cell(residual[mask], selected_blocks)
    tables = {'GLOBAL': np.asarray(global_delta)}
    counts = {'GLOBAL': np.asarray(global_n, dtype=np.int64)}
    block_counts = {'GLOBAL': np.asarray(global_b, dtype=np.int64)}
    tables['HORIZON'] = np.empty(horizon, dtype=np.float64)
    counts['HORIZON'] = np.zeros(horizon, dtype=np.int64)
    block_counts['HORIZON'] = np.zeros(horizon, dtype=np.int64)
    for h in range(horizon):
        keep = mask[:, h, :]
        tables['HORIZON'][h], counts['HORIZON'][h], block_counts['HORIZON'][h] = _cell(
            residual[:, h, :][keep], np.broadcast_to(blocks[:, None], (samples, nodes))[keep], global_delta)
    shapes = {'NODE_HORIZON': (nodes, horizon), 'ORIGIN': (nodes, horizon, 8)}
    shapes.update({name: (nodes, horizon, 8, size) for name, size in CHILD_STATES.items()})
    for name, shape in shapes.items():
        tables[name] = np.empty(shape, dtype=np.float64)
        counts[name] = np.zeros(shape, dtype=np.int64)
        block_counts[name] = np.zeros(shape, dtype=np.int64)
    for node in range(nodes):
        for h in range(horizon):
            keep = mask[:, h, node]
            values, local_blocks = residual[:, h, node][keep], blocks[keep]
            nh, counts['NODE_HORIZON'][node, h], block_counts['NODE_HORIZON'][node, h] = _cell(
                values, local_blocks, tables['HORIZON'][h])
            tables['NODE_HORIZON'][node, h] = nh
            origin, n, b = _groups(values, local_blocks, flow[:, node][keep], np.full(8, nh))
            tables['ORIGIN'][node, h], counts['ORIGIN'][node, h], block_counts['ORIGIN'][node, h] = origin, n, b
            for name, size in CHILD_STATES.items():
                keys = flow[:, node]*size+states[name][:, node]
                if name in ('NEIGHBOR2', 'JOINT'):
                    keys = np.where(neighbor_available[:, node], keys, -1)
                fitted, n, b = _groups(values, local_blocks, keys[keep], np.repeat(origin, size))
                shape = (8, size)
                tables[name][node, h] = fitted.reshape(shape)
                counts[name][node, h], block_counts[name][node, h] = n.reshape(shape), b.reshape(shape)
    statistics = {name: _stats(counts[name], block_counts[name], total_valid, total_blocks) for name in ARMS}
    return dict(schema=SCHEMA, arms=list(ARMS), sole_candidate='JOINT',
        node_count=nodes, horizon=horizon,
        protocol=dict(min_points=MIN_POINTS, min_blocks=MIN_BLOCKS, shrinkage=SHRINKAGE,
            block_size=BLOCK_SIZE, flow_edges=['-inf', 0., 5., 20., 50., 100., 200., 314., 'inf'],
            residual='truth_minus_prediction', correction='added_without_clipping',
            feature_cutpoints='all_training_origin_inputs_only_no_target_conditioning',
            tie_policy='searchsorted_right', child_parent='ORIGIN',
            missing_neighbor='ORIGIN', own2_median=True, neighbor2_median=True,
            joint_state='own2_times_2_plus_neighbor2'),
        training=dict(sample_count=samples, valid_count=total_valid, block_count=total_blocks,
            sample_ids_sha256=hashlib.sha256(ids.astype('<u8').tobytes()).hexdigest(),
            split='caller_supplied_training_only', validation_labels_used=False),
        cutpoints=cutpoints,
        tables={name: value.tolist() for name, value in tables.items()},
        support={name: dict(valid_count=counts[name].tolist(), block_count=block_counts[name].tolist())
            for name in ARMS},
        statistics=statistics, stats=statistics)


def _validated_tables(model):
    _require(isinstance(model, dict) and model.get('schema') == SCHEMA
        and model.get('arms') == list(ARMS), 'Unsupported conditional median model')
    protocol = model.get('protocol', {})
    _require(all(protocol.get(name) == expected for name, expected in
        (('min_points', MIN_POINTS), ('min_blocks', MIN_BLOCKS),
         ('shrinkage', SHRINKAGE), ('block_size', BLOCK_SIZE))), 'Fixed support protocol differs')
    _require(protocol.get('flow_edges') == ['-inf', 0., 5., 20., 50., 100., 200., 314., 'inf']
        and protocol.get('residual') == 'truth_minus_prediction'
        and protocol.get('child_parent') == 'ORIGIN'
        and protocol.get('missing_neighbor') == 'ORIGIN'
        and protocol.get('joint_state') == 'own2_times_2_plus_neighbor2', 'Fixed binning or residual protocol differs')
    nodes, horizon = model.get('node_count'), model.get('horizon')
    _require(isinstance(nodes, int) and nodes > 0 and isinstance(horizon, int) and horizon > 0,
        'Positive fitted node and horizon axes required')
    shapes = {'GLOBAL': (), 'HORIZON': (horizon,), 'NODE_HORIZON': (nodes, horizon),
        'ORIGIN': (nodes, horizon, 8)}
    shapes.update({name: (nodes, horizon, 8, size) for name, size in CHILD_STATES.items()})
    tables = {}
    for name, shape in shapes.items():
        tables[name] = np.asarray(model.get('tables', {}).get(name), dtype=np.float64)
        _require(tables[name].shape == shape and np.isfinite(tables[name]).all(),
            'Invalid fitted table: '+name)
    cuts = model.get('cutpoints', {})
    for name in FEATURES[1:]:
        record = cuts.get(name, {})
        quartiles, median = record.get('quartiles'), record.get('median')
        if quartiles is None:
            _require(name == 'neighbor_change_disagreement' and median is None,
                'Only unavailable neighbor cutpoints may be absent')
        else:
            values = np.asarray(quartiles, dtype=np.float64)
            _require(values.shape == (3,) and np.isfinite(values).all()
                and (np.diff(values) >= 0).all() and median == float(values[1]),
                'Invalid training cutpoints: '+name)
    return tables


def predict_corrections(model, features, horizon=None):
    """Return eight float64 [sample,horizon,node] corrections without labels."""
    tables = _validated_tables(model)
    fitted_horizon, nodes = model['horizon'], model['node_count']
    _require(horizon is None or horizon == fitted_horizon, 'Prediction horizon differs from fitted model')
    arrays = _features(features, nodes)
    flow, states, neighbor_available = _bins(arrays, model['cutpoints'])
    shape = (arrays['last_flow'].shape[0], fitted_horizon, nodes)
    node = np.arange(nodes)[None, None, :]
    h = np.arange(fitted_horizon)[None, :, None]
    origin = tables['ORIGIN'][node, h, flow[:, None, :]]
    result = dict(GLOBAL=np.full(shape, float(tables['GLOBAL']), dtype=np.float64),
        HORIZON=np.broadcast_to(tables['HORIZON'][None, :, None], shape).copy(),
        NODE_HORIZON=np.broadcast_to(tables['NODE_HORIZON'].T[None, :, :], shape).copy(),
        ORIGIN=origin)
    for name in CHILD_STATES:
        correction = tables[name][node, h, flow[:, None, :], states[name][:, None, :]]
        if name in ('NEIGHBOR2', 'JOINT'):
            correction = np.where(neighbor_available[:, None, :], correction, origin)
        result[name] = correction
    return result


def save_model(model, path):
    """Write fitted JSON; callers bind the resulting bytes in their run identity."""
    _validated_tables(model)
    target = Path(path)
    _require(not target.exists(), 'Refusing to overwrite an existing fitted model')
    payload = json.dumps(model, sort_keys=True, allow_nan=False, separators=(',', ':'))+'\n'
    with target.open('x', encoding='utf-8') as stream:
        stream.write(payload)


def load_model(path):
    def reject_constant(value):
        raise ValueError('Nonfinite JSON constant: '+value)
    model = json.loads(Path(path).read_text(encoding='utf-8'), parse_constant=reject_constant)
    _validated_tables(model)
    return model
