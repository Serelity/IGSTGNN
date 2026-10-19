"""CPU support-matched child tables; frozen GW13 parents and cuts are reused.

Only caller-supplied fitting labels enter the three new child medians. Selection
uses origin features, the native validity mask and author-order blocks. Equal
point counts, block histograms and shrinkage do not equate sample identities or
the information represented by different partitions. No neural model is fitted.
"""
from copy import deepcopy
import hashlib
import json

import numpy as np

from experiments.trajectory_diagnostic_gw13 import model as frozen


SCHEMA = 'gw14_support_matched_trajectory_median_v1'
SEED = 2025
CHILDREN = frozen.CHILDREN
ARMS = ('OWN4', 'CONTEXT', *CHILDREN)
PROTOCOL = dict(frozen_parent_cuts_context=True, child_states=4,
    context='lead_flow8_amplitude4', min_points=64, min_blocks=3,
    block_size=128, shrinkage=128, seed=SEED,
    state_selection='eligible_training_count_descending_then_state_ascending',
    k='minimum_eligible_state_count_across_three_children_within_context',
    blocks='intersection_across_all_selected_cells_of_all_three_children',
    per_block_quota='minimum_point_count_across_all_selected_cells',
    point_selection='PCG64_SHA256_seed_lead_context_arm_state_author_block',
    fallback='frozen_CONTEXT_for_all_arms_in_failed_context_or_unretained_state',
    fitted_target='y_minus_SUM_minus_frozen_OWN4',
    fitted_value='frozen_CONTEXT_plus_n_over_n_plus_128_times_median_minus_CONTEXT',
    heldout_common_mask='native_valid_and_all_three_children_hit_retained_cell',
    interpretation='robustness_under_matched_budget_not_equal_sample_identity_or_architecture_authorization')


def _digest(value):
    raw = json.dumps(value, sort_keys=True, allow_nan=False, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()


def _ids_digest(ids):
    return hashlib.sha256(ids.astype('<u8').tobytes()).hexdigest()


def _rng(h, context, arm, state, block):
    token = f'{SEED}:{h}:{context}:{arm}:{state}:{int(block)}'.encode('ascii')
    seed = int.from_bytes(hashlib.sha256(token).digest()[:16], 'little')
    return np.random.Generator(np.random.PCG64(seed))


def _fit_inputs(parent, data):
    frozen.validate_model(parent)
    arrays = frozen._arrays(data['features'], parent['nodes'])
    pred, truth, valid = (np.asarray(data[key]) for key in ('pred', 'y', 'valid'))
    expected = (parent['fit_sample_count'], parent['horizon'], parent['nodes'])
    frozen.require(pred.shape == truth.shape == valid.shape == expected
        and valid.dtype == np.bool_ and valid.any()
        and pred.dtype.kind in 'fiu' and truth.dtype.kind in 'fiu'
        and np.isfinite(pred).all() and np.isfinite(truth).all(),
        'Matching requires the finite native fitting cubes used by frozen GW13')
    frozen.require(arrays['last_flow'].shape[0] == expected[0], 'Fitting feature sample axis differs')
    ids = np.asarray(data['sample_ids'])
    frozen.require(ids.shape == (expected[0],) and ids.dtype.kind in 'iu'
        and (ids >= 0).all() and np.unique(ids).size == ids.size
        and _ids_digest(ids) == parent['fit_sample_ids_sha256']
        and int(valid.sum()) == parent['fit_valid_count'],
        'Matching fit identity or validity differs from frozen GW13')
    return arrays, pred, truth, valid, ids


def fit_matched_model(frozen_model, fit_data):
    """Refit three child tables only; no configurable seed or fitted cut exists.

    The supplied GW13 model must be the authenticated model for exactly this
    fitting prefix. Support is recomputed and checked against its frozen rows.
    """
    arrays, pred, truth, valid, ids = _fit_inputs(frozen_model, fit_data)
    samples, horizon, nodes = truth.shape
    flow, amplitude, context, states = frozen._bins(arrays, frozen_model['cuts'])
    context_flat = context.ravel()
    state_flat = {name: state.ravel() for name, state in states.items()}
    blocks = np.broadcast_to((ids//128)[:, None], (samples, nodes)).ravel()
    context_values = np.asarray(frozen_model['tables']['CONTEXT'], dtype=np.float64)
    tables = {name: np.repeat(context_values, 4, axis=1).copy() for name in CHILDREN}
    retained = {name: np.zeros((horizon, 128), dtype=np.bool_) for name in CHILDREN}
    support = {name: {key: np.zeros((horizon, 128), dtype=np.int64)
                     for key in ('valid_count', 'block_count', 'author_sample_count')}
               for name in CHILDREN}
    point_digests = {name: [['']*128 for _ in range(horizon)] for name in CHILDREN}
    seen_samples = {name: np.zeros(samples, dtype=np.bool_) for name in CHILDREN}
    seen_blocks = {name: set() for name in CHILDREN}
    own = np.asarray(frozen_model['parent_tables']['OWN4'], dtype=np.float64)
    rows = []
    for h in range(horizon):
        point_ids = np.flatnonzero(valid[:, h, :].ravel())
        order = np.argsort(context_flat[point_ids], kind='stable')
        point_ids = point_ids[order]
        sorted_context = context_flat[point_ids]
        edges = np.searchsorted(sorted_context, np.arange(33))
        residual = (truth[:, h, :].astype(np.float64)-pred[:, h, :]
            -own[np.arange(nodes)[None, :], h, flow, amplitude]).ravel()
        for c in range(32):
            points = point_ids[edges[c]:edges[c+1]]
            frozen.require(len(points) == frozen_model['support']['CONTEXT']['valid_count'][h][c]
                and np.unique(blocks[points]).size == frozen_model['support']['CONTEXT']['block_count'][h][c],
                'Fitting context support differs from frozen GW13')
            cells, block_sets, eligible, proposed = {}, {}, {}, {}
            for name in CHILDREN:
                cells[name], block_sets[name], eligible[name] = {}, {}, []
                for state in range(4):
                    chosen = points[state_flat[name][points] == state]
                    cell_blocks = np.unique(blocks[chosen])
                    key = c*4+state
                    frozen.require(len(chosen) == frozen_model['support'][name]['valid_count'][h][key]
                        and cell_blocks.size == frozen_model['support'][name]['block_count'][h][key],
                        'Fitting child support differs from frozen GW13')
                    cells[name][state], block_sets[name][state] = chosen, cell_blocks
                    if len(chosen) >= 64 and cell_blocks.size >= 3:
                        eligible[name].append(state)
            requested = min(len(eligible[name]) for name in CHILDREN)
            for name in CHILDREN:
                proposed[name] = sorted(eligible[name], key=lambda s: (-len(cells[name][s]), s))[:requested]
            shared = None
            for name in CHILDREN:
                for state in proposed[name]:
                    values = block_sets[name][state]
                    shared = values.copy() if shared is None else np.intersect1d(shared, values)
            shared = np.asarray([] if shared is None else shared, dtype=blocks.dtype)
            quotas = [min(int(np.count_nonzero(blocks[cells[name][state]] == block))
                          for name in CHILDREN for state in proposed[name])
                      for block in shared]
            candidate_n = sum(quotas)
            reason = ('no_common_eligible_state' if requested == 0 else
                'fewer_than_three_shared_blocks' if shared.size < 3 else
                'fewer_than_64_matched_points' if candidate_n < 64 else 'retained')
            active = reason == 'retained'
            n, k = (candidate_n, requested) if active else (0, 0)
            weight = n/(n+128) if active else 0.
            row = dict(horizon=h+1, context=c, flow_bin=c//4, amplitude_bin=c%4,
                requested_k=requested, k=k, common_block_count=int(shared.size),
                common_block_ids=shared.tolist(), per_block_quota=quotas,
                candidate_n_per_cell=candidate_n, n_per_cell=n, w=weight,
                retained=active, reason=reason)
            for name in CHILDREN:
                row[name+'_eligible_count'] = len(eligible[name])
                row[name+'_proposed_states'] = proposed[name]
                row[name+'_retained_states'] = proposed[name] if active else []
                row[name+'_retained_author_sample_counts'] = []
                if not active:
                    continue
                for state in proposed[name]:
                    chosen = cells[name][state]
                    chunks = []
                    for block, quota in zip(shared, quotas):
                        available = chosen[blocks[chosen] == block]
                        chunks.append(_rng(h, c, name, state, block).choice(available, quota, replace=False))
                    selected = np.sort(np.concatenate(chunks))
                    histogram = [int(np.count_nonzero(blocks[selected] == block)) for block in shared]
                    frozen.require(len(selected) == n and histogram == quotas,
                        'Selected training points do not obey the common block budget')
                    key = c*4+state
                    retained[name][h, key] = True
                    tables[name][h, key] = context_values[h, c]+weight*(
                        float(np.median(residual[selected]))-context_values[h, c])
                    support[name]['valid_count'][h, key] = n
                    support[name]['block_count'][h, key] = shared.size
                    selected_samples = np.unique(selected//nodes)
                    support[name]['author_sample_count'][h, key] = selected_samples.size
                    row[name+'_retained_author_sample_counts'].append(int(selected_samples.size))
                    seen_samples[name][selected_samples] = True
                    seen_blocks[name].update(int(block) for block in shared)
                    identity = np.column_stack((ids[selected//nodes], selected % nodes)).astype('<u8')
                    point_digests[name][h][key] = hashlib.sha256(identity.tobytes()).hexdigest()
            rows.append(row)
    stage = dict(original_valid_points=int(valid.sum()), original_author_samples=samples,
        inherited_parent_context_refits=0, child_tables_refitted=3,
        per_arm={name: dict(retained_cells=int(retained[name].sum()),
            selected_valid_points=int(support[name]['valid_count'].sum()),
            distinct_author_samples=int(seen_samples[name].sum()),
            distinct_author_blocks=len(seen_blocks[name])) for name in CHILDREN})
    result = dict(schema=SCHEMA, protocol=deepcopy(PROTOCOL), arms=list(ARMS),
        frozen_parent_model=deepcopy(frozen_model), frozen_parent_model_sha256=_digest(frozen_model),
        tables={name: value.tolist() for name, value in tables.items()},
        retained={name: value.tolist() for name, value in retained.items()},
        support={name: {key: value.tolist() for key, value in row.items()} for name, row in support.items()},
        selected_point_ids_sha256=point_digests, contexts=rows,
        fit_stage_counts=stage, match_exact=True)
    validate_model(result)
    return result


def validate_model(model):
    """Check serialized inherited identity and every equal-budget invariant."""
    frozen.require(isinstance(model, dict) and model.get('schema') == SCHEMA
        and model.get('protocol') == PROTOCOL and model.get('arms') == list(ARMS)
        and model.get('match_exact') is True, 'Unknown matched trajectory model')
    parent = frozen.validate_model(model.get('frozen_parent_model'))
    frozen.require(model.get('frozen_parent_model_sha256') == _digest(parent), 'Frozen parent snapshot changed')
    horizon = parent['horizon']
    masks, supports = {}, {}
    for name in CHILDREN:
        table = np.asarray(model.get('tables', {}).get(name), dtype=np.float64)
        mask = np.asarray(model.get('retained', {}).get(name))
        frozen.require(table.shape == mask.shape == (horizon, 128)
            and np.isfinite(table).all() and mask.dtype == np.bool_, 'Invalid matched table or mask')
        masks[name], supports[name] = mask, {}
        for key in ('valid_count', 'block_count', 'author_sample_count'):
            value = np.asarray(model.get('support', {}).get(name, {}).get(key))
            frozen.require(value.shape == mask.shape and value.dtype.kind in 'iu'
                and (value >= 0).all() and (value[~mask] == 0).all(), 'Invalid matched support')
            supports[name][key] = value
        counts, blocks = supports[name]['valid_count'], supports[name]['block_count']
        samples = supports[name]['author_sample_count']
        frozen.require((counts[mask] >= 64).all() and (blocks[mask] >= 3).all()
            and (samples[mask] >= blocks[mask]).all() and (samples <= counts).all(),
            'Matched cells need native minimum support')
        baseline = np.repeat(np.asarray(parent['tables']['CONTEXT']), 4, axis=1)
        frozen.require(np.array_equal(table[~mask], baseline[~mask]), 'Unretained cells must use frozen CONTEXT')
        digests = model.get('selected_point_ids_sha256', {}).get(name)
        frozen.require(isinstance(digests, list) and len(digests) == horizon
            and all(isinstance(row, list) and len(row) == 128 for row in digests), 'Missing selected point identities')
        for h in range(horizon):
            for key in range(128):
                digest = digests[h][key]
                frozen.require(isinstance(digest, str) and
                    ((mask[h, key] and len(digest) == 64 and all(ch in '0123456789abcdef' for ch in digest))
                     or (not mask[h, key] and digest == '')), 'Invalid selected point identity')
    rows = model.get('contexts')
    frozen.require(isinstance(rows, list) and len(rows) == horizon*32, 'One row per lead/context required')
    for index, row in enumerate(rows):
        h, c = divmod(index, 32)
        frozen.require(isinstance(row, dict) and row.get('horizon') == h+1 and row.get('context') == c
            and row.get('flow_bin') == c//4 and row.get('amplitude_bin') == c%4,
            'Matched context order or identity differs')
        k, n, b, weight = (row.get(key) for key in ('k', 'n_per_cell', 'common_block_count', 'w'))
        frozen.require(type(k) is int and 0 <= k <= 4 and type(n) is int and n >= 0
            and type(b) is int and b >= 0 and type(row.get('retained')) is bool,
            'Invalid matched context budget')
        block_ids, quotas = row.get('common_block_ids'), row.get('per_block_quota')
        frozen.require(isinstance(block_ids, list) and isinstance(quotas, list)
            and len(block_ids) == len(quotas) == b
            and all(type(x) is int and x >= 0 for x in block_ids)
            and block_ids == sorted(set(block_ids))
            and all(type(x) is int and x > 0 for x in quotas)
            and row.get('candidate_n_per_cell') == sum(quotas), 'Invalid shared block histogram')
        eligible_counts = []
        for name in CHILDREN:
            original_n = np.asarray(parent['support'][name]['valid_count'])[h, c*4:c*4+4]
            original_b = np.asarray(parent['support'][name]['block_count'])[h, c*4:c*4+4]
            eligible = [state for state in range(4) if original_n[state] >= 64 and original_b[state] >= 3]
            eligible_counts.append(len(eligible))
            frozen.require(row.get(name+'_eligible_count') == len(eligible), 'Original eligibility differs')
        requested = min(eligible_counts)
        frozen.require(row.get('requested_k') == requested, 'Unequal proposed cell budget')
        expected_reason = ('no_common_eligible_state' if requested == 0 else
            'fewer_than_three_shared_blocks' if b < 3 else
            'fewer_than_64_matched_points' if sum(quotas) < 64 else 'retained')
        active = expected_reason == 'retained'
        frozen.require(row['reason'] == expected_reason and row['retained'] == active
            and k == (requested if active else 0) and n == (sum(quotas) if active else 0)
            and weight == (n/(n+128) if active else 0.), 'Matched budget or fallback rule differs')
        for name in CHILDREN:
            original_n = np.asarray(parent['support'][name]['valid_count'])[h, c*4:c*4+4]
            original_b = np.asarray(parent['support'][name]['block_count'])[h, c*4:c*4+4]
            eligible = [state for state in range(4) if original_n[state] >= 64 and original_b[state] >= 3]
            proposed = sorted(eligible, key=lambda s: (-original_n[s], s))[:requested]
            kept = proposed if active else []
            frozen.require(row.get(name+'_proposed_states') == proposed
                and row.get(name+'_retained_states') == kept, 'State selection differs from fixed support rule')
            expected_mask = np.zeros(4, dtype=np.bool_)
            expected_mask[kept] = True
            local_mask = masks[name][h, c*4:c*4+4]
            frozen.require(np.array_equal(local_mask, expected_mask), 'Retained cell mask differs')
            for state in kept:
                key = c*4+state
                frozen.require(supports[name]['valid_count'][h, key] == n
                    and supports[name]['block_count'][h, key] == b, 'Unequal retained point or block budget')
            expected_samples = [int(supports[name]['author_sample_count'][h, c*4+state]) for state in kept]
            frozen.require(row.get(name+'_retained_author_sample_counts') == expected_samples,
                'Retained author-sample counts differ from cell support')
    stage = model.get('fit_stage_counts', {})
    frozen.require(stage.get('original_valid_points') == parent['fit_valid_count']
        and stage.get('original_author_samples') == parent['fit_sample_count']
        and stage.get('inherited_parent_context_refits') == 0 and stage.get('child_tables_refitted') == 3,
        'Unexpected fitting stages')
    for name in CHILDREN:
        summary = stage.get('per_arm', {}).get(name, {})
        block_union = {block for row in rows if row['retained'] for block in row['common_block_ids']}
        selected_count = int(supports[name]['valid_count'].sum())
        frozen.require(summary.get('retained_cells') == int(masks[name].sum())
            and summary.get('selected_valid_points') == selected_count
            and type(summary.get('distinct_author_samples')) is int
            and len(block_union) <= summary['distinct_author_samples'] <= min(parent['fit_sample_count'], selected_count)
            and type(summary.get('distinct_author_blocks')) is int
            and summary['distinct_author_blocks'] == len(block_union),
            'Fitting summary differs from selected support')
    return model


def predictions(model, data):
    """Yield only the five paired diagnostic arms; no scoring labels are read."""
    validate_model(model)
    adapted = dict(model['frozen_parent_model'])
    adapted['tables'] = dict(adapted['tables'])
    adapted['tables'].update(model['tables'])
    for name, value in frozen.predictions(adapted, data):
        if name in ARMS:
            yield name, value


def score_masks(model, features, valid):
    """Return native-valid common-support mask and point-coverage summaries."""
    validate_model(model)
    parent = model['frozen_parent_model']
    arrays = frozen._arrays(features, parent['nodes'])
    _, _, context, states = frozen._bins(arrays, parent['cuts'])
    valid = np.asarray(valid)
    frozen.require(valid.dtype == np.bool_ and valid.shape ==
        (context.shape[0], parent['horizon'], parent['nodes']) and valid.any(),
        'Matching coverage needs a nonempty native Boolean scoring mask')
    common = valid.copy()
    per_arm = {}
    h = np.arange(parent['horizon'])[None, :, None]
    total = int(valid.sum())
    for name in CHILDREN:
        keys = context*4+states[name]
        active = np.asarray(model['retained'][name])[h, keys[:, None, :]] & valid
        count = int(active.sum())
        per_arm[name] = dict(active_valid_count=count, active_fraction=float(count/total),
            retained_cells=int(np.asarray(model['retained'][name]).sum()))
        common &= active
    count = int(common.sum())
    report = dict(total_valid_count=total, common_valid_count=count,
        common_fraction=float(count/total), per_arm_active=per_arm,
        contexts_total=len(model['contexts']),
        contexts_retained=sum(int(row['retained']) for row in model['contexts']),
        meaning='native_valid_points_hitting_retained_cells_in_all_three_arms_not_independent_samples',
        match_exact=True)
    return common, report


def save_model(model, path):
    validate_model(model)
    with open(path, 'x', encoding='utf-8', newline='\n') as stream:
        json.dump(model, stream, sort_keys=True, allow_nan=False, separators=(',', ':'))
        stream.write('\n')
