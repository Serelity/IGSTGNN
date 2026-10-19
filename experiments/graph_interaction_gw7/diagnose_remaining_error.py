"""CPU-only R_CONTINUE BEST residual/history analysis; no model imports.

Reads an authenticated existing GW7 prediction archive and the published input
array. Fits feature scales/bins on training X only. No inference, training,
checkpoint deserialization, test evaluation or prediction correction.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


REPO = Path(__file__).resolve().parents[2]
ARM = 'R_CONTINUE'
ARMS = ('R_CONTINUE', 'R_MLP', 'R_MEMORY', 'R_INTERACTION_MEMORY')
TRAIN_SCHEMA = 'original_igstgnn_graph_interaction_gw7_training_v1'
ARCHIVE_SCHEMA = 'original_igstgnn_graph_interaction_gw7_archive_v1'
RELEASE = {
    'incident_all.npy': 'bc6c5608457f88864625d683f9a66d17bc0fc31fe5acba366b6033708a099279',
    'incident_stats.npz': '9f6590a3d1df5a0cd59e00717e3157520a4f5b99996976a4e41ddc6fc28b224c',
    'adj_matrix.npy': 'de2a74cc981853ff81dd494d6e7d212b7ae524fb9d6079c91323e56bcb22254f',
    'sensors.csv': '682f3cdf75e643f0b37356ab69cbabb27389be5089f41d3b2cbc4bede3332094',
}
LIMITATIONS = [
    'Descriptive analysis of repeatedly used development validation; no causal or independent-confirmation claim.',
    'Windows and nodes are dependent. Counts are not independent sample sizes; no p-values or confidence intervals are computed.',
    'Training input windows can repeat observations. Scales and bins describe window exposures, not deduplicated calendar times.',
    'Conditional contrasts use coarse bins and common support, not exact covariate matching or causal adjustment.',
    'Published A directions are prediction dependencies, not certified traffic upstream/downstream directions.',
    'All 5585 trusted published objects are deserialized, including unused test payload; only train X and validation X/Y are indexed. No test evaluation.',
    'Checkpoint bytes, archive and certificates are checked; checkpoints/models are never loaded or independently replayed.',
    'Six contiguous author-order blocks are sensitivity descriptions; actual calendar order and block independence are not certified.',
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    with Path(path).open('w', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def write_csv(path, rows):
    if not rows:
        Path(path).write_text('', encoding='utf-8')
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fit_bins(values, quantiles=(1 / 3, 2 / 3)):
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    require(finite.size > 0, 'No finite training feature values')
    return np.unique(np.quantile(finite, quantiles))


def assign_bins(values, cuts):
    values = np.asarray(values, dtype=np.float64)
    cuts = np.asarray(cuts, dtype=np.float64)
    require(cuts.ndim == 1 and np.isfinite(cuts).all()
        and (np.diff(cuts) > 0).all(), 'Sorted unique training cutpoints required')
    # Ties stay in the lower interval. In particular, cut=[0] gives a usable
    # zero-versus-positive comparison instead of putting all zeros in HIGH.
    labels = np.searchsorted(cuts, values, side='left').astype(np.int64)
    return np.where(np.isfinite(values), labels, -1)


def graph_weights(adjacency):
    a = np.asarray(adjacency, dtype=np.float64).copy()
    require(a.ndim == 2 and a.shape[0] == a.shape[1]
        and np.isfinite(a).all() and (a >= 0).all(), 'Finite nonnegative square graph required')
    np.fill_diagonal(a, 0.)
    def normalize(matrix):
        mass = matrix.sum(axis=1, keepdims=True)
        return np.divide(matrix, mass, out=np.zeros_like(matrix), where=mass > 0)
    return {'symmetric': normalize(a + a.T), 'row_A': normalize(a), 'row_AT': normalize(a.T)}


def temporal_features(flow, scale):
    x, scale = np.asarray(flow, dtype=np.float64), np.asarray(scale, dtype=np.float64)
    require(x.ndim == 3 and x.shape[1] >= 6 and scale.shape == (x.shape[2],)
        and np.isfinite(x).all() and np.isfinite(scale).all() and (scale > 0).all(),
        'Finite [window,time,node] history and positive per-node scales required')
    return dict(level=x.mean(axis=1), volatility=np.abs(np.diff(x, axis=1) / scale).mean(axis=1),
        trend=(x[:, -3:].mean(axis=1) - x[:, :3].mean(axis=1)) / scale)


def neighbor_features(flow, scale, weights, chunk_size=16, progress=None):
    x, weights = np.asarray(flow), np.asarray(weights, dtype=np.float64)
    local = temporal_features(x, scale)['volatility']
    n = x.shape[2]
    require(chunk_size > 0 and weights.shape == (n, n) and np.isfinite(weights).all()
        and (weights >= 0).all() and np.allclose(np.diag(weights), 0.), 'Invalid neighbor weights')
    mass = weights.sum(axis=1)
    require(np.allclose(mass[mass > 0], 1.), 'Neighbor rows must sum to one, or be empty')
    receivers, senders = np.nonzero(weights)
    edge_weight = weights[receivers, senders]
    disagreement = np.zeros(local.shape, dtype=np.float64)
    # Never materialize [all_windows,time,edges]. Absolute value is taken BEFORE
    # summing over neighbors, so opposite neighbor trends cannot cancel.
    for first in range(0, len(x), chunk_size):
        stop = min(first + chunk_size, len(x))
        delta = np.diff(x[first:stop].astype(np.float64), axis=1) / scale
        for edge_first in range(0, len(receivers), 2048):
            edge_stop = edge_first + 2048
            dst, src = receivers[edge_first:edge_stop], senders[edge_first:edge_stop]
            gap = np.abs(delta[:, :, src] - delta[:, :, dst]).mean(axis=1)
            for row in range(stop - first):
                disagreement[first + row] += np.bincount(dst,
                    weights=gap[row] * edge_weight[edge_first:edge_stop], minlength=n)
        if progress is not None and (first == 0 or stop == len(x) or stop % 256 == 0):
            progress(stop, len(x))
    neighbor = local @ weights.T
    denominator = local + neighbor
    ratio = np.divide(disagreement, denominator, out=np.zeros_like(disagreement), where=denominator > 0)
    require((ratio <= 1. + 1e-10).all(), 'Discordance ratio violates triangle inequality')
    ratio = np.minimum(ratio, 1.)
    for value in (disagreement, neighbor, ratio):
        value[:, mass == 0] = np.nan
    return dict(disagreement=disagreement, neighbor_volatility=neighbor, discordance_ratio=ratio)


def conditional_contrast(scores, normalized_scores, bias, labels, controls,
                         min_count=5, high_label=2, control_values=None):
    """Within-node/binned-control high-minus-low, weighted by min(n_low,n_high).

    Each observation is ONE window x node score, already averaged over valid
    horizons. The endpoints come from TRAINING bins, never validation maxima.
    """
    scores, normalized_scores, bias = map(lambda x: np.asarray(x, dtype=np.float64),
        (scores, normalized_scores, bias))
    labels = np.asarray(labels)
    require(scores.ndim == 2 and all(v.shape == scores.shape for v in
        (normalized_scores, bias, labels, *controls)) and min_count >= 1 and high_label >= 1,
        'Invalid conditional contrast axes/options')
    require(labels.dtype.kind in 'iu', 'Integer exposure bins required')
    eligible = np.isfinite(scores) & np.isfinite(normalized_scores) & np.isfinite(bias) & (labels >= 0)
    control_values = control_values or {}
    for name, value in control_values.items():
        require(np.asarray(value).shape == scores.shape, 'Continuous control axes differ: ' + name)
        eligible &= np.isfinite(value)
    node = np.broadcast_to(np.arange(scores.shape[1]), scores.shape)
    strata, radix = node.copy(), []
    for values in controls:
        values = np.asarray(values)
        require(values.dtype.kind in 'iu', 'Integer control bins required')
        eligible &= values >= 0
        base = max(1, int(values.max()) + 1)
        radix.append(base)
        strata = strata * base + np.maximum(values, 0)
    total_strata = scores.shape[1] * int(np.prod(radix, dtype=np.int64))
    require(total_strata < 1000000, 'Too many registered conditional strata')
    extreme = eligible & ((labels == 0) | (labels == high_label))
    ids = (strata[extreme] * 2 + (labels[extreme] == high_label)).astype(np.int64)
    counts = np.bincount(ids, minlength=2 * total_strata).reshape(-1, 2)
    present = counts.sum(axis=1) > 0
    matched = (counts >= min_count).all(axis=1)
    means = {}
    outcomes = dict(mae=scores, scaled_mae=normalized_scores, bias=bias)
    outcomes.update({'control_' + name: np.asarray(value) for name, value in control_values.items()})
    for name, value in outcomes.items():
        sums = np.bincount(ids, weights=value[extreme], minlength=2 * total_strata).reshape(-1, 2)
        means[name] = np.divide(sums, counts, out=np.full(sums.shape, np.nan), where=counts > 0)
    weights = np.minimum(counts[:, 0], counts[:, 1])[matched].astype(float)
    cells = []
    matched_node_ids, node_effects = [], []
    for index in np.flatnonzero(present):
        rem, bins = int(index), []
        for base in reversed(radix):
            bins.append(rem % base); rem //= base
        row = dict(node_index=rem, control_bins='|'.join(map(str, reversed(bins))),
            low_units=int(counts[index, 0]), high_units=int(counts[index, 1]), matched=bool(matched[index]))
        for name, value in means.items():
            row['low_' + name] = float(value[index, 0]) if counts[index, 0] else None
            row['high_' + name] = float(value[index, 1]) if counts[index, 1] else None
            row[name + '_high_minus_low'] = float(value[index, 1] - value[index, 0]) if matched[index] else None
        cells.append(row)
        if matched[index]:
            matched_node_ids.append(rem)
            node_effects.append(row['mae_high_minus_low'])
    summary = dict(min_units_per_extreme_per_stratum=min_count, high_training_bin=high_label,
        eligible_units=int(eligible.sum()), low_units=int(counts[:, 0].sum()), high_units=int(counts[:, 1].sum()),
        matched_strata=int(matched.sum()), matched_nodes=len(set(matched_node_ids)),
        matched_low_units=int(counts[matched, 0].sum()), matched_high_units=int(counts[matched, 1].sum()),
        matched_extreme_fraction=float(counts[matched].sum() / max(1, extreme.sum())),
        matched_all_fraction=float(counts[matched].sum() / max(1, np.isfinite(scores).sum())),
        weighting='min(low_units,high_units) across matched same-node/control strata')
    for name, value in means.items():
        summary['weighted_' + name + '_high_minus_low'] = (float(np.average(
            value[matched, 1] - value[matched, 0], weights=weights)) if weights.size else None)
        if name.startswith('control_'):
            for endpoint, column in (('low', 0), ('high', 1)):
                summary['weighted_' + name + '_' + endpoint] = float(np.average(
                    value[matched, column], weights=weights)) if weights.size else None
    summary['positive_matched_strata_fraction'] = float(np.mean(np.asarray(node_effects) > 0)) if weights.size else None
    if weights.size:
        node_ids = np.asarray(matched_node_ids)
        sums = np.bincount(node_ids, weights=np.asarray(node_effects) * weights, minlength=scores.shape[1])
        mass = np.bincount(node_ids, weights=weights, minlength=scores.shape[1])
        effects = sums[mass > 0] / mass[mass > 0]
        summary['equal_node_mae_high_minus_low'] = float(effects.mean())
        summary['positive_matched_nodes_fraction'] = float((effects > 0).mean())
    else:
        summary.update(equal_node_mae_high_minus_low=None, positive_matched_nodes_fraction=None)
    return summary, cells


def verify_run(run):
    identity, summary = read_json(run / 'identity.json'), read_json(run / 'summary.json')
    budget = read_json(run / 'budget.json')
    require(budget.get('target_completed_epochs') == 5 and budget.get('total_not_additional') is True
        and budget.get('maximum') == 5 and budget.get('extension_allowed') is False,
        'Original fixed five-epoch screening budget required')
    unsigned = {k: v for k, v in identity.items() if k != 'identity_sha256'}
    require(hashlib.sha256(json.dumps(unsigned, sort_keys=True).encode()).hexdigest()
        == identity.get('identity_sha256'), 'GW7 identity digest differs')
    require(identity.get('schema') == TRAIN_SCHEMA and identity.get('arms') == list(ARMS)
        and identity.get('engineering_only') is False, 'Full four-arm GW7 run required')
    require(summary.get('status') == 'GW7_TRAINING_COMPLETE' and summary.get('requested_new_epochs') == 5
        and summary.get('same_completed_new_training_budget') is True and summary.get('test_evaluated') is False,
        'Completed fixed five-epoch GW7 summary required')
    require(summary['source_reference'] == identity['source_reference'], 'Summary source differs')
    for arm in ARMS:
        row = summary['arms'][arm]
        require(row.get('completed_epoch') == 5 and row.get('target_completed_epochs') == 5
            and row.get('target_reached') is True and row.get('global_updates') == 410,
            'Finish the fixed five-epoch, 410-update budget: ' + arm)
    for split, bounds in (('train', [0, 3909]), ('val', [3909, 4746])):
        saved = identity['data']['splits'][split]
        require(saved['bounds_half_open'] == bounds and saved['selected_indices'] == list(range(*bounds))
            and saved['count'] == bounds[1] - bounds[0], 'Full author-order split differs: ' + split)
    for name, digest in identity['execution_source'].items():
        path = (REPO / name).resolve()
        require(REPO in path.parents and sha256(path) == digest, 'GW7 execution bytes differ: ' + name)
    return identity, summary


def select_evaluation(run, explicit, identity, summary):
    paths = [explicit.resolve() / 'metadata.json'] if explicit else sorted(run.glob('*/metadata.json'))
    matches, skipped = [], []
    saved = summary['arms'][ARM]
    for path in paths:
        try:
            meta = read_json(path)
            require(meta.get('schema') == 'gw7_graph_interaction_producer_v1'
                and meta.get('status') == 'GW7_PREDICTION_PASS' and meta.get('checkpoint_selection') == 'best'
                and meta.get('run_identity_sha256') == identity['identity_sha256']
                and meta.get('registered_arms') == list(ARMS) and meta.get('common_completed_new_epochs') == 5
                and meta.get('engineering_only') is False and meta.get('test_evaluated') is False
                and meta.get('optimizer_updates') == 0 and meta.get('source_reference') == identity['source_reference'],
                'Not a matching completed GW7 BEST certificate')
            role = meta['checkpoint_roles'][ARM]
            require(role.get('selection') == 'best' and role.get('policy') == ARM
                and role.get('historical') is False and role.get('epoch') == saved['best_epoch']
                and role.get('checkpoint_sha256') == saved['best_checkpoint_sha256']
                and meta['validation_replay'][ARM]['passed'] is True, 'R_CONTINUE BEST role/replay differs')
            require(meta.get('producer_source'), 'Missing evaluation source hashes')
            for name, digest in meta['producer_source'].items():
                source = (REPO / name).resolve()
                require(REPO in source.parents and sha256(source) == digest, 'Evaluation source bytes differ')
            require(sha256(path.parent / 'paired_predictions.npz') == meta['archive_sha256'], 'Archive digest differs')
            matches.append((path.parent, meta))
        except (ValueError, OSError, KeyError, TypeError) as error:
            if explicit:
                raise
            skipped.append(dict(path=str(path), reason=str(error)))
    require(matches, 'No certified GW7 BEST evaluation found; pass --best-dir explicitly')
    checkpoint_name = saved['best_checkpoint_file']
    require(Path(checkpoint_name).name == checkpoint_name, 'Unsafe checkpoint filename')
    checkpoint = run / ARM / checkpoint_name
    require(sha256(checkpoint) == saved['best_checkpoint_sha256'], 'Current R_CONTINUE BEST checkpoint bytes differ')
    directory, meta = matches[-1]
    return directory, meta, checkpoint, skipped


def load_archive(path, identity, node_ids):
    with np.load(path, allow_pickle=False) as payload:
        require(str(payload['schema'].item()) == ARCHIVE_SCHEMA
            and str(payload['axis_order'].item()) == 'sample,horizon,node'
            and np.array_equal(payload['registered_arms'], np.asarray(ARMS)), 'GW7 archive registration differs')
        result = {key: payload[key] for key in ('y', 'valid', 'sample_ids', 'node_ids', 'pred__' + ARM)}
    y, valid, p = result['y'], result['valid'], result['pred__' + ARM]
    require(y.shape == (837, 12, 496) and y.dtype == np.float32 and p.shape == y.shape
        and p.dtype.kind == 'f' and np.isfinite(y).all() and np.isfinite(p).all(), 'Prediction/target axes differ')
    require(valid.shape == y.shape and valid.dtype == np.bool_ and valid.any()
        and (y[valid] > 0).all()
        and np.array_equal(valid, y != (float(y.min()) if y.min() < 1 else 0.)), 'Shared original mask differs')
    require(np.array_equal(result['sample_ids'], identity['data']['splits']['val']['selected_indices'])
        and np.array_equal(result['node_ids'], node_ids), 'Sample/node order differs')
    return result


def trusted_array(path):
    # Called only AFTER the exact published file digest is authenticated.
    try:
        return np.load(path, allow_pickle=True)
    except ModuleNotFoundError as error:
        if error.name not in ('numpy._core', 'numpy._core.multiarray'):
            raise
        import numpy.core.multiarray as multiarray
        sys.modules.setdefault('numpy._core.multiarray', multiarray)
        return np.load(path, allow_pickle=True)


def inverse_float32(values, mean, std):
    values = np.asarray(values, dtype=np.float32)
    return np.add(np.multiply(values, np.float32(std), dtype=np.float32), np.float32(mean), dtype=np.float32)


def load_histories(directory, identity, archive, mean, std):
    data = trusted_array(directory / 'incident_all.npy')
    require(len(data) == 5585, 'Published sample count differs')
    histories = []
    for split, ids in (('train', range(3909)), ('val', archive['sample_ids'])):
        rows = []
        for offset, index in enumerate(ids):
            sample = data[int(index)]
            x = np.asarray(sample['x_data'])
            require(x.shape == (12, 496, 3) and x.dtype == np.float32 and np.isfinite(x).all(),
                'Invalid published X in ' + split)
            rows.append(x[..., 0])
            if split == 'val':
                y = np.asarray(sample['y_data'])
                require(y.shape == (12, 496, 3) and y.dtype == np.float32
                    and np.array_equal(inverse_float32(y[..., 0], mean, std), archive['y'][offset]),
                    'Published validation truth does not exactly replay archive, index ' + str(index))
        histories.append(inverse_float32(np.stack(rows), mean, std))
    del data
    return histories


def score_units(prediction, truth, valid):
    error = prediction.astype(np.float64) - truth.astype(np.float64)
    count = valid.sum(axis=1)
    mae = np.divide(np.where(valid, np.abs(error), 0.).sum(axis=1), count,
        out=np.full(count.shape, np.nan, dtype=np.float64), where=count > 0)
    bias = np.divide(np.where(valid, error, 0.).sum(axis=1), count,
        out=np.full(count.shape, np.nan, dtype=np.float64), where=count > 0)
    return mae, bias


def grouped_metrics(prediction, truth, valid, labels, scale, feature, high_label):
    error = prediction.astype(np.float64) - truth.astype(np.float64)
    rows = []
    for level in range(high_label + 1):
        for h in (None, *range(12)):
            mask = valid & (labels[:, None, :] == level)
            if h is not None:
                mask[:, :h] = False; mask[:, h + 1:] = False
            count = int(mask.sum())
            e, y = error[mask], truth[mask]
            scaled = np.broadcast_to(scale[None, None, :], truth.shape)[mask]
            rows.append(dict(feature=feature, bin=level, horizon='all' if h is None else h + 1,
                valid_points=count, windows=int(mask.any(axis=(1, 2)).sum()), nodes=int(mask.any(axis=(0, 1)).sum()),
                mae=float(np.abs(e).mean()) if count else None,
                scaled_mae=float((np.abs(e) / scaled).mean()) if count else None,
                rmse=float(np.sqrt(np.mean(e * e))) if count else None,
                signed_bias=float(e.mean()) if count else None,
                mape_fraction=float((np.abs(e) / y).mean()) if count else None))
    return rows


def build_analysis(train_x, val_x, archive, adjacency, output, chunk_size):
    # Fixed floor is ONE release flow unit. Scales use TRAINING X only, with
    # overlapping window exposures retained; historical zeros are not masked.
    quartiles = np.quantile(train_x.astype(np.float64), (.25, .75), axis=(0, 1))
    iqr = quartiles[1] - quartiles[0]
    scale = np.maximum(iqr, 1.)
    local_train, local_val = temporal_features(train_x, scale), temporal_features(val_x, scale)
    level_cuts = np.quantile(local_train['level'], (1 / 3, 2 / 3), axis=0).T
    level_bins = np.column_stack([assign_bins(local_val['level'][:, i], np.unique(level_cuts[i]))
        for i in range(len(scale))])
    fitted = dict(level_per_node_cutpoints=level_cuts.tolist(), scale=scale.tolist(),
        scale_floor_nodes=int((iqr < 1.).sum()), scale_floor=1., node_training_iqr=iqr.tolist(), features={})
    features, bins, high = {}, {}, {}
    for name in ('volatility', 'trend'):
        cuts = fit_bins(local_train[name])
        occupied = np.unique(assign_bins(local_train[name], cuts)).tolist()
        fitted['features'][name] = dict(cutpoints=cuts.tolist(), training_bin_count=len(cuts) + 1,
            training_occupied_bins=occupied, training_degenerate=len(occupied) < 2, cut_equality='lower_bin')
        features[name] = local_val[name]; bins[name] = assign_bins(features[name], cuts); high[name] = len(cuts)
    graphs = graph_weights(adjacency)
    graph_info = {}
    for graph_name, weights in graphs.items():
        graph_info[graph_name] = dict(nonself_edges=int((weights > 0).sum()), empty_nodes=int((weights.sum(1) == 0).sum()))
        neighbor = []
        for split, x in (('train', train_x), ('val', val_x)):
            def progress(done, total, split=split, graph_name=graph_name):
                print(json.dumps(dict(phase='history_features', graph=graph_name, split=split,
                    windows=done, total=total)), flush=True)
            neighbor.append(neighbor_features(x, scale, weights, chunk_size, progress))
        for name in ('disagreement', 'neighbor_volatility', 'discordance_ratio'):
            key = graph_name + '__' + name
            cuts = fit_bins(neighbor[0][name])
            occupied = np.unique(assign_bins(neighbor[0][name], cuts))
            occupied = occupied[occupied >= 0].tolist()
            fitted['features'][key] = dict(cutpoints=cuts.tolist(), training_bin_count=len(cuts) + 1,
                training_occupied_bins=occupied, training_degenerate=len(occupied) < 2, cut_equality='lower_bin')
            features[key] = neighbor[1][name]; bins[key] = assign_bins(features[key], cuts); high[key] = len(cuts)
        # Ratio zero at an entirely inactive history is well-defined, but is
        # excluded from conditional ratio contrasts, not called alignment.
        ratio_key = graph_name + '__discordance_ratio'
        activity = local_val['volatility'] + neighbor[1]['neighbor_volatility']
        bins[ratio_key] = np.where(activity > 0, bins[ratio_key], -1)
    p, y, valid = archive['pred__' + ARM], archive['y'], archive['valid']
    scores, biases = score_units(p, y, valid)
    marginal = grouped_metrics(p, y, valid, level_bins, scale, 'within_node_history_level', 2)
    for key, labels in bins.items():
        marginal.extend(grouped_metrics(p, y, valid, labels, scale, key, high[key]))
    contrasts, cells, blocks, horizons = {}, [], [], []
    level_values = dict(level_over_scale=local_val['level'] / scale)
    specs = [('volatility_given_level', 'volatility', [level_bins], level_values)]
    for g in graphs:
        d, nv, q = (g + '__' + name for name in ('disagreement', 'neighbor_volatility', 'discordance_ratio'))
        loose_values = dict(level_values, local_volatility=features['volatility'])
        strict_values = dict(loose_values, neighbor_volatility=features[nv])
        specs += [(g + '__D_given_level_local', d, [level_bins, bins['volatility']], loose_values),
            (g + '__D_given_level_local_neighbor', d, [level_bins, bins['volatility'], bins[nv]], strict_values),
            (g + '__Q_given_level_local_neighbor', q, [level_bins, bins['volatility'], bins[nv]], strict_values)]
    for name, feature, controls, control_values in specs:
        common = dict(min_count=5, high_label=high[feature])
        result, rows = conditional_contrast(scores, scores / scale, biases, bins[feature], controls,
            control_values=control_values, **common)
        contrasts[name] = result
        cells.extend(dict(contrast=name, **row) for row in rows)
        for block, indices in enumerate(np.array_split(np.arange(len(y)), 6), 1):
            item, _ = conditional_contrast(scores[indices], (scores / scale)[indices], biases[indices],
                bins[feature][indices], [v[indices] for v in controls],
                control_values={k: v[indices] for k, v in control_values.items()}, **common)
            blocks.append(dict(contrast=name, block=block, first_sample_id=int(archive['sample_ids'][indices[0]]),
                last_sample_id=int(archive['sample_ids'][indices[-1]]), **item))
        for h in range(12):
            hs, hb = score_units(p[:, h:h + 1], y[:, h:h + 1], valid[:, h:h + 1])
            item, _ = conditional_contrast(hs, hs / scale, hb, bins[feature], controls,
                control_values=control_values, **common)
            horizons.append(dict(contrast=name, horizon=h + 1, **item))
    write_csv(output / 'marginal_groups.csv', marginal)
    write_csv(output / 'conditional_cells.csv', cells)
    write_csv(output / 'conditional_contrasts.csv', [dict(contrast=k, **v) for k, v in contrasts.items()])
    write_csv(output / 'by_order_block.csv', blocks)
    write_csv(output / 'by_horizon.csv', horizons)
    node_rows = [dict(node_index=i, station_id=int(archive['node_ids'][i]), scale=float(scale[i]),
        training_iqr=float(iqr[i]), level_q33=float(level_cuts[i, 0]), level_q67=float(level_cuts[i, 1])) for i in range(len(scale))]
    write_csv(output / 'node_scales.csv', node_rows)
    np.savez_compressed(output / 'history_features.npz', sample_ids=archive['sample_ids'], node_ids=archive['node_ids'],
        history_level=local_val['level'], level_bin=level_bins, scale=scale,
        **features, **{'bin__' + k: v for k, v in bins.items()})
    write_json(output / 'fitted_training_parameters.json', fitted)
    error = p[valid].astype(np.float64) - y[valid].astype(np.float64)
    return dict(pooled_mae=float(np.abs(error).mean()), pooled_rmse=float(np.sqrt(np.mean(error ** 2))),
        signed_bias=float(error.mean()), mape_fraction=float((np.abs(error) / y[valid]).mean()),
        valid_points=int(valid.sum()), window_node_macro_mae=float(np.nanmean(scores)),
        validation_windows=len(y), nodes=y.shape[2], graphs=graph_info, conditional_contrasts=contrasts)


def make_report(result):
    fmt = lambda x: 'NA' if x is None else f'{x:.6f}'
    lines = ['# R_CONTINUE BEST remaining-error diagnostic', '',
        'CPU-only cached-prediction analysis. No new inference, training or test evaluation.', '',
        f"BEST epoch: {result['checkpoint_role']['epoch']}; pooled MAE: {result['pooled_mae']:.9f}.",
        f"Validation windows: {result['validation_windows']}; valid target points: {result['valid_points']}.", '',
        'Primary graph: row-normalized (A + A.T), self edges removed. row_A/row_AT are direction sensitivity checks.',
        'V = mean absolute historical increments / per-node training IQR scale.',
        'D = weighted mean absolute neighbor-minus-receiver standardized increments (absolute value before neighbor sum).',
        'Q = D / (V_self + V_neighbor); entirely inactive histories excluded from conditional Q comparisons.', '',
        'High-minus-low contrasts below use training-bin endpoints within the SAME NODE and listed control bins.',
        'One observation is a window x node mean over its valid horizons. Strata require >=5 observations in EACH endpoint.',
        'Stratum weights are min(n_low,n_high); coverage is matched endpoint units / all scored window-node units.', '',
        '| Contrast | MAE high-low | Scaled MAE high-low | Bias high-low | Coverage | Matched nodes |',
        '|---|---:|---:|---:|---:|---:|']
    for name, row in result['conditional_contrasts'].items():
        lines.append('| ' + name + ' | ' + ' | '.join((fmt(row['weighted_mae_high_minus_low']),
            fmt(row['weighted_scaled_mae_high_minus_low']), fmt(row['weighted_bias_high_minus_low']),
            fmt(row['matched_all_fraction']), str(row['matched_nodes']))) + ' |')
    lines += ['', '## Interpretation', '',
        'A positive contrast means higher historical exposure is associated with larger remaining error, within available binned common support.',
        'If D contrasts shrink after controlling neighbor volatility, neighbor activity is a competing explanation.',
        'If D and Q patterns persist across directions, nodes and horizons, this motivates a controlled relative-change message experiment; it does not prove a mechanism.',
        'Inspect by_order_block.csv for sign/coverage consistency. Blocks may be dependent and are not calendar-certified.',
        'Each order block rebuilds common support with the same >=5 rule. Populations can differ and sparse blocks can be NA.',
        'Continuous-control endpoint means/differences are in conditional tables; residual within-bin imbalance can remain.',
        'Sparse/empty matched strata are inconclusive; a zero pooled bias does not imply small local error.',
        'marginal_groups.csv reports pooled target-point metrics; conditional tables average horizons first and have different weighting.',
        'No automatic model-selection pass, budget extension or causal conclusion is produced.', '', '## Limitations', '']
    lines.extend('- ' + item for item in LIMITATIONS)
    return '\n'.join(lines) + '\n'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--best-dir', type=Path, help='Optional; otherwise newest matching certified BEST child is selected')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--chunk-size', type=int, default=16, help='CPU feature memory chunk; does not change bins or model predictions')
    args = parser.parse_args(argv)
    run, data_dir, output = (getattr(args, name).resolve() for name in ('run_dir', 'data_dir', 'output_dir'))
    require(args.chunk_size > 0 and output.parent == run and not output.exists(),
        'Use positive chunk size and a NEW direct child output directory of --run-dir')
    inputs = {str(run / name): sha256(run / name) for name in ('identity.json', 'summary.json', 'budget.json')}
    identity, saved = verify_run(run)
    best_dir, metadata, checkpoint, skipped = select_evaluation(run, args.best_dir, identity, saved)
    for path in (best_dir / 'metadata.json', best_dir / 'analysis.json', best_dir / 'paired_predictions.npz', checkpoint):
        inputs[str(path)] = sha256(path)
    require(inputs[str(best_dir / 'paired_predictions.npz')] == metadata['archive_sha256'],
        'Archive changed after certificate selection')
    require(inputs[str(checkpoint)] == saved['arms'][ARM]['best_checkpoint_sha256'],
        'Checkpoint changed after selection')
    for name, digest in RELEASE.items():
        path = data_dir / name
        require(identity['data']['hashes'].get(name) == digest and sha256(path) == digest,
            'Exact trusted published input digest required: ' + name)
        inputs[str(path)] = digest
    with (data_dir / 'sensors.csv').open(encoding='utf-8-sig', newline='') as stream:
        node_ids = np.asarray([int(row['station_id']) for row in csv.DictReader(stream)], dtype=np.int64)
    require(len(node_ids) == 496 and len(np.unique(node_ids)) == 496, 'Unique 496-node axis required')
    archive = load_archive(best_dir / 'paired_predictions.npz', identity, node_ids)
    require(sha256(best_dir / 'paired_predictions.npz') == metadata['archive_sha256'],
        'Archive changed while loading')
    with np.load(data_dir / 'incident_stats.npz', allow_pickle=False) as stats:
        mean, std = float(stats['mean']), float(stats['std'])
        require(bool(stats['normalized']) and np.isfinite([mean, std]).all() and std > 0, 'Original normalization required')
    require(identity['data']['normalization']['mean'] == mean and identity['data']['normalization']['std'] == std,
        'Saved normalization differs')
    output.mkdir()
    try:
        manifest = dict(schema='gw7_remaining_error_inputs_v1', input_sha256=inputs,
            script_sha256=sha256(Path(__file__)), best_directory=str(best_dir), skipped_evaluation_metadata=skipped)
        write_json(output / 'input_manifest.json', manifest)
        print(json.dumps(dict(phase='loading_authenticated_history', best_dir=str(best_dir), arm=ARM,
            checkpoint_epoch=metadata['checkpoint_roles'][ARM]['epoch'])), flush=True)
        train_x, val_x = load_histories(data_dir, identity, archive, mean, std)
        adjacency = np.load(data_dir / 'adj_matrix.npy', allow_pickle=False)
        result = build_analysis(train_x, val_x, archive, adjacency, output, args.chunk_size)
        certified = read_json(best_dir / 'analysis.json')['cases'][ARM]
        for actual, key in (('pooled_mae', 'mae_pooled'), ('pooled_rmse', 'rmse_pooled'),
                            ('signed_bias', 'signed_bias'), ('mape_fraction', 'mape_pooled')):
            require(np.isclose(result[actual], certified[key], atol=2e-5, rtol=1e-7),
                'Cached metric replay differs: ' + actual)
        require(np.isclose(result['pooled_mae'], saved['arms'][ARM]['best_validation_mae'], atol=2e-5, rtol=1e-6),
            'MAE differs from saved BEST selection')
        for path, digest in inputs.items():
            require(sha256(path) == digest, 'Input changed during analysis: ' + path)
        require(sha256(Path(__file__)) == manifest['script_sha256'], 'Diagnostic source changed during analysis')
        result.update(schema='gw7_recurrent_remaining_error_v1', status='GW7_REMAINING_ERROR_COMPLETE',
            created_utc=datetime.now(timezone.utc).isoformat(), run_identity_sha256=identity['identity_sha256'],
            arm=ARM, selection='best', checkpoint_role=metadata['checkpoint_roles'][ARM],
            input_manifest_sha256=sha256(output / 'input_manifest.json'), exact_target_replay=True,
            cached_metric_replay=True, parameter_fitting='training_X_only_for_scales_and_bins',
            inference_calls=0, optimizer_updates=0, checkpoint_deserializations=0, gpu_used=False,
            test_evaluated=False, test_payload_deserialized=True, limitations=LIMITATIONS)
        (output / 'report.md').write_text(make_report(result), encoding='utf-8')
        # Completion marker comes last; interrupted/failed tables are partial.
        write_json(output / 'summary.json', result)
        print(json.dumps(dict(status=result['status'], pooled_mae=result['pooled_mae'], output_dir=str(output))), flush=True)
        return 0
    except BaseException as error:
        write_json(output / 'failure.json', dict(status='GW7_REMAINING_ERROR_FAILED',
            error_type=type(error).__name__, error=str(error), completed=False))
        raise


if __name__ == '__main__':
    sys.exit(main())
