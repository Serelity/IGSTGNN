"""CPU conditional medians, with a node-specific OWN4 parent and shared children.

This is an information probe, not a neural architecture. The shared context is
lead x last-flow-bin x amplitude-quartile. It does not condition on origin
clocks, and is not a replication of the GW12 seven-feature readout comparison.
"""
import hashlib
import json

import numpy as np

from experiments.conditional_residual_gw10 import median_probe as old


SCHEMA = 'gw13_shared_trajectory_median_v1'
CHILDREN = ('MAGNITUDE16', 'DIRECTION16', 'TRAJECTORY16')
ARMS = ('SUM', 'ORIGIN', 'OWN4', 'CONTEXT', *CHILDREN)
PROTOCOL = dict(parent='train_only_node_lead_flow_OWN4', shared_context='lead_flow8_amplitude4',
    child_states=4, min_points=64, min_blocks=3, block_size=128, shrinkage=128,
    target='y_minus_SUM_minus_OWN4', prediction='SUM_plus_OWN4_plus_one_shared_table',
    shared_context_parent=0., child_parent='CONTEXT',
    magnitude_subcuts='training_quartiles_within_each_amplitude_quartile',
    direction_subcuts='training_signed_direction_quartiles_within_each_amplitude_quartile',
    trajectory='two_times_nonnegative_early_trend_plus_nonnegative_recent_trend',
    cutoff_ties='searchsorted_right_no_jitter', clocks_in_new_lookup=False,
    node_pooling=True, validation_labels_used_for_fit=False)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _arrays(features, nodes=None):
    names = ('last_flow', 'mean_abs_change', 'signed_direction', 'shape')
    require(isinstance(features, dict) and all(k in features for k in names), 'Missing trajectory features')
    arrays = {k: np.asarray(features[k]) for k in names}
    shape = arrays['last_flow'].shape
    require(len(shape) == 2 and min(shape) > 0 and (nodes is None or shape[1] == nodes)
        and all(v.shape == shape and v.dtype.kind in 'fiu' and np.isfinite(v).all()
                for v in arrays.values()), 'Invalid [sample,node] trajectory features')
    require((arrays['mean_abs_change'] >= 0).all()
        and (np.abs(arrays['signed_direction']) <= 1.000001).all(), 'Invalid amplitude or direction')
    require(arrays['shape'].dtype.kind in 'iu' and (arrays['shape'] >= 0).all()
        and (arrays['shape'] < 4).all(), 'Shape codes must be integers in 0..3')
    return arrays


def _bins(features, cuts):
    arrays = _arrays(features)
    flow = np.searchsorted(np.asarray(old.FLOW_EDGES[1:-1]), arrays['last_flow'], side='right')
    amplitude = np.searchsorted(cuts['amplitude'], arrays['mean_abs_change'], side='right')
    states = {name: np.zeros(flow.shape, dtype=np.int64) for name in CHILDREN}
    for group in range(4):
        selected = amplitude == group
        for name, key in (('MAGNITUDE16', 'mean_abs_change'), ('DIRECTION16', 'signed_direction')):
            boundary = cuts[name][group]
            if boundary is not None:
                states[name][selected] = np.searchsorted(boundary, arrays[key][selected], side='right')
    states['TRAJECTORY16'] = arrays['shape'].astype(np.int64)
    return flow, amplitude, flow*4+amplitude, states


def fit_model(data):
    """Fit every parent, cut and child using only caller-supplied fit labels."""
    arrays = _arrays(data['features'])
    prediction, truth, valid = (np.asarray(data[k]) for k in ('pred', 'y', 'valid'))
    require(prediction.shape == truth.shape == valid.shape and truth.ndim == 3
        and valid.dtype == np.bool_ and valid.any() and np.isfinite(prediction).all()
        and np.isfinite(truth).all(), 'Finite paired prediction/label cubes and a Boolean mask required')
    samples, horizon, nodes = truth.shape
    require(arrays['last_flow'].shape == (samples, nodes), 'Feature axes differ from fit cubes')
    require('neighbor_change_disagreement' in data['features'],
            'Keep all three original cache descriptors when fitting the GW10 parent')
    # The parent is refitted within each audit prefix. No full-train lookup is
    # allowed to see the labels of that prefix's held-out scoring segment.
    base = old.fit_probe(prediction, truth, valid, data['features'], data['sample_ids'])
    cuts = dict(amplitude=base['cutpoints']['mean_abs_change']['quartiles'],
                MAGNITUDE16=[], DIRECTION16=[])
    amplitude = np.searchsorted(cuts['amplitude'], arrays['mean_abs_change'], side='right')
    for group in range(4):
        selected = amplitude == group
        for name, key in (('MAGNITUDE16', 'mean_abs_change'), ('DIRECTION16', 'signed_direction')):
            cuts[name].append(np.quantile(arrays[key][selected], [.25, .5, .75]).tolist()
                              if selected.any() else None)
    parents = {name: base['tables'][name] for name in ('ORIGIN', 'OWN4')}
    parent_support = base['statistics']['OWN4']
    del base
    flow, amplitude, context, states = _bins(data['features'], cuts)
    own = np.asarray(parents['OWN4'], dtype=np.float64)
    tables, support = {}, {}
    for name in ('CONTEXT', *CHILDREN):
        size = 32 if name == 'CONTEXT' else 128
        tables[name] = np.zeros((horizon, size), dtype=np.float64)
        support[name] = {k: np.zeros((horizon, size), dtype=np.int64)
                         for k in ('valid_count', 'block_count')}
    ids = np.asarray(data['sample_ids'])
    # fit_probe has already checked uniqueness, integer type and nonnegative IDs.
    blocks = np.broadcast_to((ids//old.BLOCK_SIZE)[:, None], (samples, nodes)).ravel()
    node = np.arange(nodes)[None, :]
    for h in range(horizon):
        keep = valid[:, h, :].ravel()
        residual = (truth[:, h, :].astype(np.float64)-prediction[:, h, :]
                    -own[node, h, flow, amplitude]).ravel()[keep]
        fitted, counts, coverage = old._groups(residual, blocks[keep], context.ravel()[keep], np.zeros(32))
        tables['CONTEXT'][h] = fitted
        support['CONTEXT']['valid_count'][h], support['CONTEXT']['block_count'][h] = counts, coverage
        for name in CHILDREN:
            keys = (context*4+states[name]).ravel()[keep]
            values, counts, coverage = old._groups(residual, blocks[keep], keys, np.repeat(fitted, 4))
            tables[name][h] = values
            support[name]['valid_count'][h], support[name]['block_count'][h] = counts, coverage
    return dict(schema=SCHEMA, protocol=dict(PROTOCOL), arms=list(ARMS), nodes=nodes, horizon=horizon,
        fit_sample_count=samples, fit_valid_count=int(valid.sum()),
        fit_sample_ids_sha256=hashlib.sha256(ids.astype('<u8').tobytes()).hexdigest(),
        cuts=cuts, parent_tables=parents, parent_support=parent_support,
        tables={k:v.tolist() for k,v in tables.items()},
        support={k:{s:v.tolist() for s,v in row.items()} for k,row in support.items()})


def validate_model(model):
    require(isinstance(model, dict) and model.get('schema') == SCHEMA
        and model.get('protocol') == PROTOCOL and model.get('arms') == list(ARMS), 'Unknown trajectory model')
    nodes, horizon = model.get('nodes'), model.get('horizon')
    require(type(nodes) is int and nodes > 0 and type(horizon) is int and horizon > 0, 'Invalid fitted axes')
    samples, count = model.get('fit_sample_count'), model.get('fit_valid_count')
    digest = model.get('fit_sample_ids_sha256')
    require(type(samples) is int and samples > 0 and type(count) is int
        and 0 < count <= samples*horizon*nodes and isinstance(digest, str)
        and len(digest) == 64 and all(c in '0123456789abcdef' for c in digest),
        'Invalid fitting count or sample identity')
    require(isinstance(model.get('parent_support'), dict), 'Missing original parent support')
    for name, size in (('ORIGIN', (nodes, horizon, 8)), ('OWN4', (nodes, horizon, 8, 4))):
        value = np.asarray(model.get('parent_tables', {}).get(name), dtype=np.float64)
        require(value.shape == size and np.isfinite(value).all(), 'Invalid parent table '+name)
    cuts = model.get('cuts', {})
    amplitude = np.asarray(cuts.get('amplitude'), dtype=np.float64)
    require(amplitude.shape == (3,) and np.isfinite(amplitude).all() and (np.diff(amplitude) >= 0).all(),
        'Invalid training amplitude boundaries')
    for name in CHILDREN[:2]:
        require(isinstance(cuts.get(name), list) and len(cuts[name]) == 4, 'Four within-amplitude cuts required')
        for values in cuts[name]:
            if values is not None:
                values = np.asarray(values, dtype=np.float64)
                require(values.shape == (3,) and np.isfinite(values).all() and (np.diff(values) >= 0).all(),
                        'Invalid within-amplitude boundaries')
    for name in ('CONTEXT', *CHILDREN):
        shape = (horizon, 32 if name == 'CONTEXT' else 128)
        values = np.asarray(model.get('tables', {}).get(name), dtype=np.float64)
        require(values.shape == shape and np.isfinite(values).all(), 'Invalid shared table '+name)
        for key in ('valid_count', 'block_count'):
            counts = np.asarray(model.get('support', {}).get(name, {}).get(key))
            require(counts.shape == shape and counts.dtype.kind in 'iu' and (counts >= 0).all(),
                    'Invalid support '+name+'/'+key)
        require(int(np.asarray(model['support'][name]['valid_count']).sum()) == count,
                'Each shared table must account for every valid fitting point')
    for name in CHILDREN:
        counts = np.asarray(model['support'][name]['valid_count']).reshape(horizon, 32, 4)
        blocks = np.asarray(model['support'][name]['block_count']).reshape(horizon, 32, 4)
        require(np.array_equal(counts.sum(axis=-1), model['support']['CONTEXT']['valid_count'])
            and (blocks <= np.asarray(model['support']['CONTEXT']['block_count'])[..., None]).all(),
            'Child support must partition its common context')
    return model


def predictions(model, data):
    """Predict one arm at a time; no labels or validity mask are accessed."""
    validate_model(model)
    arrays = _arrays(data['features'], model['nodes'])
    reference = np.asarray(data['pred'], dtype=np.float64)
    require(reference.shape == (arrays['last_flow'].shape[0], model['horizon'], model['nodes'])
        and np.isfinite(reference).all(), 'Prediction cube differs from fitted axes')
    flow, amplitude, context, states = _bins(arrays, model['cuts'])
    node = np.arange(model['nodes'])[None, None, :]
    h = np.arange(model['horizon'])[None, :, None]
    origin = np.asarray(model['parent_tables']['ORIGIN'])[node, h, flow[:, None, :]]
    own = np.asarray(model['parent_tables']['OWN4'])[node, h, flow[:, None, :], amplitude[:, None, :]]
    yield 'SUM', reference
    yield 'ORIGIN', reference+origin
    own_prediction = reference+own
    yield 'OWN4', own_prediction
    for name in ('CONTEXT', *CHILDREN):
        keys = context if name == 'CONTEXT' else context*4+states[name]
        table = np.asarray(model['tables'][name], dtype=np.float64)
        yield name, own_prediction+table[np.arange(model['horizon'])[None, :, None], keys[:, None, :]]


def coverage(model, features, valid):
    """Fitted support and held-out coverage, not independent sample counts."""
    validate_model(model)
    _arrays(features, model['nodes'])
    flow, amplitude, context, states = _bins(features, model['cuts'])
    valid = np.asarray(valid)
    require(valid.dtype == np.bool_ and valid.shape == (flow.shape[0], model['horizon'], model['nodes'])
        and valid.any(), 'Coverage needs a matching nonempty Boolean scoring mask')
    result = {}
    for name in ('CONTEXT', *CHILDREN):
        counts = np.asarray(model['support'][name]['valid_count'])
        blocks = np.asarray(model['support'][name]['block_count'])
        eligible = (counts >= old.MIN_POINTS) & (blocks >= old.MIN_BLOCKS)
        keys = context if name == 'CONTEXT' else context*4+states[name]
        active = eligible[np.arange(model['horizon'])[None, :, None], keys[:, None, :]]
        observed = counts > 0
        result[name] = dict(total_cells=int(counts.size), observed_cells=int(observed.sum()),
            eligible_cells=int(eligible.sum()), valid_count=int(valid.sum()),
            active_valid_count=int((active & valid).sum()),
            active_fraction=float((active & valid).sum()/valid.sum()),
            fit_active_fraction=float(counts[eligible].sum()/model['fit_valid_count']),
            min_observed_blocks=int(blocks[observed].min()) if observed.any() else None,
            max_observed_blocks=int(blocks[observed].max()) if observed.any() else None,
            meaning='eligible_current_level_cells_not_parent_fallback_not_independent_points')
    return result


def save_model(model, path):
    validate_model(model)
    with open(path, 'x', encoding='utf-8', newline='\n') as stream:
        json.dump(model, stream, sort_keys=True, allow_nan=False, separators=(',', ':'))
        stream.write('\n')
