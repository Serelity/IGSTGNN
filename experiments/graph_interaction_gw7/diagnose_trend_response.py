"""Final CPU-only, history-matched signed-trend audit of cached R_CONTINUE BEST.

No Torch/model imports, checkpoint deserialization, inference or training.
Old cached local features are reconstructed from authenticated published X;
no neighbor features are recomputed and no prediction correction is fitted.
"""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

from experiments.graph_interaction_gw7 import diagnose_remaining_error as old


require, sha256, read_json, write_json, write_csv = (
    old.require, old.sha256, old.read_json, old.write_json, old.write_csv)
DEADBAND = .05
MIN_PAIRS = 5
PROFILE_MULTIPLIERS = {'primary': 1., 'double_caliper_sensitivity': 2.}
FEATURE_NAMES = ('mean_level', 'recent3_level', 'last_level', 'volatility', 'abs_trend')
LIMITATIONS = [
    'Repeatedly used development validation, overlapping windows and shared nodes; no causal or independent-confirmation claims.',
    'Six author-order blocks are matching strata, not certified calendar periods or independent replicates.',
    'Pairing reads history features only; matching is maximum cardinality under fixed calipers, not a global minimum-distance optimum.',
    'Finite calipers permit residual imbalance. Uncontrolled history features and future reversal remain competing explanations.',
    'An UP-negative/DOWN-positive error pattern is compatible with underresponse to historical trend, not proof of lag or a faulty encoder.',
    'Directional matching controls the latest observed flow level; mean and recent-three levels are reported but not forced equal.',
    'Both caliper profiles are descriptive. Do not select the more favorable profile or relax support after seeing errors.',
    'No new predictions or corrections are generated. Checkpoints are only byte-hashed, never deserialized.',
    'Trusted full published object array is deserialized, including unused test payload. Only train X and validation X/Y are indexed; no test evaluation.',
]


def classify_trend(trend, deadband=DEADBAND):
    values = np.asarray(trend, dtype=np.float64)
    require(np.isfinite(values).all() and np.isfinite(deadband) and deadband >= 0,
        'Finite historical trend and nonnegative deadband required')
    return np.where(values > deadband, 1, np.where(values < -deadband, -1, 0)).astype(np.int8)


def caliper_pairs(features, left, right, calipers):
    """Deterministic maximum-cardinality bipartite matching, no replacement.

    Augmenting paths can reassign a previously used right endpoint. Candidate
    preferences use normalized L1 distance then original row index; this does
    NOT claim globally minimum total distance. Call separately per node/block.
    """
    x = np.asarray(features, dtype=np.float64)
    left, right, tol = np.asarray(left), np.asarray(right), np.asarray(calipers, dtype=np.float64)
    require(x.ndim == 2 and x.shape[1] > 0 and np.isfinite(x).all()
        and tol.shape == (x.shape[1],) and np.isfinite(tol).all() and (tol > 0).all(),
        'Finite feature matrix and positive per-column calipers required')
    for ids in (left, right):
        require(ids.ndim == 1 and ids.dtype.kind in 'iu' and len(np.unique(ids)) == len(ids)
            and (ids >= 0).all() and (ids < len(x)).all(), 'Unique in-range integer endpoint indices required')
    require(not np.intersect1d(left, right).size, 'Endpoint groups must be disjoint')
    left, right = np.sort(left.astype(np.int64)), np.sort(right.astype(np.int64))
    if not left.size or not right.size:
        return np.empty((0, 2), dtype=np.int64)
    gaps = np.abs(x[left, None, :] - x[right][None, :, :])
    allowed = (gaps <= tol).all(axis=2)
    distance = (gaps / tol).sum(axis=2)
    neighbors = []
    for i in range(len(left)):
        candidates = np.flatnonzero(allowed[i])
        neighbors.append(candidates[np.lexsort((right[candidates], distance[i, candidates]))])
    owner = np.full(len(right), -1, dtype=np.int64)
    def augment(i, seen):
        for j in neighbors[i]:
            if seen[j]:
                continue
            seen[j] = True
            if owner[j] < 0 or augment(int(owner[j]), seen):
                owner[j] = i
                return True
        return False
    for i in range(len(left)):
        augment(i, np.zeros(len(right), dtype=bool))
    pairs = [(int(left[i]), int(right[j])) for j, i in enumerate(owner) if i >= 0]
    return np.asarray(sorted(pairs), dtype=np.int64).reshape(-1, 2)


def pair_statistics(left_error, right_error, common_valid, scale, directional=False):
    a, b = np.asarray(left_error, dtype=np.float64), np.asarray(right_error, dtype=np.float64)
    mask, scale = np.asarray(common_valid), np.asarray(scale, dtype=np.float64)
    require(a.ndim == 2 and b.shape == a.shape and mask.shape == a.shape and mask.dtype == np.bool_
        and scale.shape == (len(a),) and np.isfinite(scale).all() and (scale > 0).all()
        and np.isfinite(a[mask]).all() and np.isfinite(b[mask]).all(), 'Invalid paired errors/mask/scales')
    counts = mask.sum(axis=1)
    keep = counts > 0
    def per_pair(values):
        return np.divide(np.where(mask, values, 0.).sum(axis=1), counts,
            out=np.full(len(a), np.nan), where=keep)
    ab, bb = per_pair(a), per_pair(b)
    am, bm = per_pair(np.abs(a)), per_pair(np.abs(b))
    average = lambda value: float(np.mean(value[keep])) if keep.any() else None
    result = dict(pairs_proposed=len(a), pairs_scored=int(keep.sum()), common_valid_points=int(mask.sum()),
        left_mae=average(am), right_mae=average(bm), mae_left_minus_right=average(am - bm),
        left_bias=average(ab), right_bias=average(bb), scaled_left_mae=average(am / scale),
        scaled_right_mae=average(bm / scale), scaled_mae_left_minus_right=average((am - bm) / scale),
        aligned_error=average((-ab + bb) / 2) if directional else None,
        scaled_aligned_error=average((-ab + bb) / (2 * scale)) if directional else None,
        underreaction_pair_fraction=float(np.mean((ab[keep] < 0) & (bb[keep] > 0)))
            if directional and keep.any() else None)
    result['underresponse_score'] = result['aligned_error']
    result['scaled_underresponse_score'] = result['scaled_aligned_error']
    return result


def select_history(run, explicit, identity, saved):
    paths = [explicit.resolve() / 'summary.json'] if explicit else sorted(run.glob('*/summary.json'))
    matches = []
    for path in paths:
        try:
            s = read_json(path)
            require(path.parent.parent == run and s.get('schema') == 'gw7_recurrent_remaining_error_v1'
                and s.get('status') == 'GW7_REMAINING_ERROR_COMPLETE' and s.get('arm') == old.ARM
                and s.get('selection') == 'best' and s.get('run_identity_sha256') == identity['identity_sha256']
                and s.get('exact_target_replay') is True and s.get('cached_metric_replay') is True
                and s.get('test_evaluated') is False and s.get('gpu_used') is False
                and s.get('optimizer_updates') == 0 and s.get('inference_calls') == 0
                and s.get('checkpoint_deserializations') == 0, 'Matching completed history diagnostic required')
            role = s['checkpoint_role']
            require(role['epoch'] == saved['arms'][old.ARM]['best_epoch'] and role['selection'] == 'best'
                and role['checkpoint_sha256'] == saved['arms'][old.ARM]['best_checkpoint_sha256'],
                'History diagnostic BEST role differs')
            manifest_path = path.parent / 'input_manifest.json'
            require(sha256(manifest_path) == s['input_manifest_sha256'], 'Old input manifest changed')
            manifest = read_json(manifest_path)
            require(manifest.get('schema') == 'gw7_remaining_error_inputs_v1'
                and manifest.get('script_sha256') == sha256(Path(old.__file__)), 'Old diagnostic producer differs')
            matches.append((s.get('created_utc', ''), str(path.parent), s, manifest))
        except (OSError, ValueError, KeyError, TypeError):
            if explicit:
                raise
    require(matches, 'No completed matching remaining-error diagnostic; supply --history-dir')
    _, directory, summary, manifest = sorted(matches, key=lambda v: (v[0], v[1]))[-1]
    return Path(directory), summary, manifest


def load_inputs(run, directory, history_arg):
    inputs = {str(run / n): sha256(run / n) for n in ('identity.json', 'summary.json', 'budget.json')}
    identity, saved = old.verify_run(run)
    history_dir, prior_summary, prior_manifest = select_history(run, history_arg, identity, saved)
    for path, digest in prior_manifest['input_sha256'].items():
        require(sha256(path) == digest, 'Old diagnostic input changed: ' + path)
        inputs[path] = digest
    best_dir = Path(prior_manifest['best_directory']).resolve()
    require(best_dir.parent == run, 'BEST evaluation must be a direct child of this run')
    best_dir, meta, checkpoint, _ = old.select_evaluation(run, best_dir, identity, saved)
    require(meta['checkpoint_roles'][old.ARM] == prior_summary['checkpoint_role'], 'History/current BEST roles differ')
    for path in (history_dir / 'summary.json', history_dir / 'input_manifest.json',
                 history_dir / 'history_features.npz', history_dir / 'fitted_training_parameters.json',
                 best_dir / 'metadata.json', best_dir / 'analysis.json', best_dir / 'paired_predictions.npz',
                 checkpoint, Path(old.__file__)):
        digest = sha256(path)
        require(str(path) not in inputs or inputs[str(path)] == digest, 'Input changed between certificates')
        inputs[str(path)] = digest
    require(read_json(history_dir / 'summary.json') == prior_summary
        and read_json(history_dir / 'input_manifest.json') == prior_manifest
        and inputs[str(history_dir / 'input_manifest.json')] == prior_summary['input_manifest_sha256']
        and inputs[str(Path(old.__file__))] == prior_manifest['script_sha256'],
        'Old diagnostic certificate/source changed during selection')
    require(inputs[str(best_dir / 'paired_predictions.npz')] == meta['archive_sha256'], 'Archive differs from certificate')
    require(inputs[str(checkpoint)] == meta['checkpoint_roles'][old.ARM]['checkpoint_sha256'],
        'Selected checkpoint changed during certification')
    for name, expected in old.RELEASE.items():
        path = directory / name
        require(identity['data']['hashes'].get(name) == expected and sha256(path) == expected,
            'Trusted published data bytes required: ' + name)
        inputs[str(path)] = expected
    with (directory / 'sensors.csv').open(encoding='utf-8-sig', newline='') as stream:
        node_ids = np.asarray([int(row['station_id']) for row in csv.DictReader(stream)], dtype=np.int64)
    archive = old.load_archive(best_dir / 'paired_predictions.npz', identity, node_ids)
    require(sha256(best_dir / 'paired_predictions.npz') == meta['archive_sha256'], 'Archive changed during loading')
    with np.load(directory / 'incident_stats.npz', allow_pickle=False) as stats:
        mean, std = float(stats['mean']), float(stats['std'])
        require(bool(stats['normalized']) and np.isfinite([mean, std]).all() and std > 0, 'Original scaler required')
    require(identity['data']['normalization']['mean'] == mean and identity['data']['normalization']['std'] == std,
        'Normalization differs from run')
    print('Reconstructing authenticated LOCAL history features; no graph or model forward.', flush=True)
    train_x, val_x = old.load_histories(directory, identity, archive, mean, std)
    quartiles = np.quantile(train_x.astype(np.float64), (.25, .75), axis=(0, 1))
    iqr = quartiles[1] - quartiles[0]
    scale = np.maximum(iqr, 1.)
    train_local, local = old.temporal_features(train_x, scale), old.temporal_features(val_x, scale)
    level_cuts = np.quantile(train_local['level'], (1 / 3, 2 / 3), axis=0).T
    level_bin = np.column_stack([old.assign_bins(local['level'][:, i], np.unique(level_cuts[i]))
        for i in range(len(scale))])
    fitted = read_json(history_dir / 'fitted_training_parameters.json')
    for key, value in (('scale', scale), ('node_training_iqr', iqr), ('level_per_node_cutpoints', level_cuts)):
        require(np.array_equal(np.asarray(fitted[key]), value), 'Training parameter replay differs: ' + key)
    expected = dict(sample_ids=archive['sample_ids'], node_ids=archive['node_ids'], scale=scale,
        history_level=local['level'], level_bin=level_bin, volatility=local['volatility'], trend=local['trend'])
    high = None
    for name in ('volatility', 'trend'):
        cuts = old.fit_bins(train_local[name])
        require(np.array_equal(fitted['features'][name]['cutpoints'], cuts)
            and fitted['features'][name]['training_bin_count'] == len(cuts) + 1,
            'Training cutpoints replay differs: ' + name)
        expected['bin__' + name] = old.assign_bins(local[name], cuts)
        if name == 'volatility':
            high = len(cuts)
    with np.load(history_dir / 'history_features.npz', allow_pickle=False) as cache:
        for key, value in expected.items():
            require(cache[key].shape == value.shape and np.array_equal(cache[key], value),
                'Cached local feature does not exactly replay published X: ' + key)
    local.update(recent3_level=val_x[:, -3:].astype(np.float64).mean(axis=1) / scale,
        last_level=val_x[:, -1].astype(np.float64) / scale, mean_level=local['level'] / scale,
        abs_trend=np.abs(local['trend']), volatility_bin=expected['bin__volatility'])
    y, p, valid = archive['y'], archive['pred__' + old.ARM], archive['valid']
    e = p[valid].astype(np.float64) - y[valid].astype(np.float64)
    replay = dict(pooled_mae=float(np.abs(e).mean()), pooled_rmse=float(np.sqrt(np.mean(e * e))),
        mape_fraction=float((np.abs(e) / y[valid]).mean()), signed_bias=float(e.mean()))
    certified = read_json(best_dir / 'analysis.json')['cases'][old.ARM]
    for key, metric in (('pooled_mae', 'mae_pooled'), ('pooled_rmse', 'rmse_pooled'),
                        ('mape_fraction', 'mape_pooled'), ('signed_bias', 'signed_bias')):
        require(np.isclose(replay[key], certified[metric], atol=2e-5, rtol=1e-7), 'Cached metric replay differs: ' + key)
    require(np.isclose(replay['pooled_mae'], saved['arms'][old.ARM]['best_validation_mae'], atol=2e-5, rtol=1e-6),
        'Cached MAE differs from selected BEST')
    del train_x, val_x, train_local
    return archive, local, scale, high, inputs, dict(run_identity_sha256=identity['identity_sha256'],
        history_directory=str(history_dir), best_directory=str(best_dir),
        checkpoint_role=meta['checkpoint_roles'][old.ARM], replay=replay)


def make_pairs(local, block_ids, high_bin, multiplier):
    direction = classify_trend(local['trend'])
    vb = local['volatility_bin']
    contrasts = {
        'volatility_high_vs_low': (vb == high_bin, vb == 0, (0,), np.array([.05]), False),
        'trend_up_vs_down_all_V': (direction == 1, direction == -1, (2, 3, 4), np.array([.05, .02, .05]), True),
        'trend_up_vs_down_high_V': ((direction == 1) & (vb == high_bin),
            (direction == -1) & (vb == high_bin), (2, 3, 4), np.array([.05, .02, .05]), True),
    }
    feature = np.stack([local[key] for key in FEATURE_NAMES], axis=-1)
    result = {}
    for name, (left_mask, right_mask, columns, base_tol, directional) in contrasts.items():
        records = []
        tol = base_tol * multiplier
        for node in range(feature.shape[1]):
            for block in range(6):
                here = block_ids == block
                pairs = caliper_pairs(feature[:, node][:, columns],
                    np.flatnonzero(here & left_mask[:, node]), np.flatnonzero(here & right_mask[:, node]), tol)
                records.extend((int(a), int(b), node, block) for a, b in pairs)
            if (node + 1) % 128 == 0:
                print(json.dumps(dict(phase='matching_progress', contrast=name,
                    caliper_multiplier=multiplier, nodes=node + 1, total_nodes=feature.shape[1])), flush=True)
        pairs = np.asarray(records, dtype=np.int64).reshape(-1, 4)
        result[name] = dict(pairs=pairs, left_mask=left_mask, right_mask=right_mask,
            columns=columns, calipers=tol, directional=directional, feature=feature)
        print(json.dumps(dict(phase='matching', contrast=name, caliper_multiplier=multiplier,
            constructed_pairs=len(pairs))), flush=True)
    return result


def summarize(spec, archive, scale, block_ids, horizon=None, block=None):
    pairs = spec['pairs']
    if block is not None:
        pairs = pairs[pairs[:, 3] == block]
    left, right, nodes, blocks = pairs.T
    y, p, valid = archive['y'], archive['pred__' + old.ARM], archive['valid']
    ea = p[left, :, nodes].astype(np.float64) - y[left, :, nodes].astype(np.float64)
    eb = p[right, :, nodes].astype(np.float64) - y[right, :, nodes].astype(np.float64)
    common = valid[left, :, nodes] & valid[right, :, nodes]
    if horizon is not None:
        ea, eb, common = ea[:, horizon:horizon + 1], eb[:, horizon:horizon + 1], common[:, horizon:horizon + 1]
    # Pair identities never change. Only the shared valid target mask determines
    # whether a pair/slice can be scored, with the declared >=5 node-block rule.
    ids = nodes * 6 + blocks
    before_min = common.any(axis=1)
    counts = np.bincount(ids[before_min], minlength=len(scale) * 6)
    common = common & (counts[ids] >= MIN_PAIRS)[:, None]
    scored = common.any(axis=1)
    result = pair_statistics(ea, eb, common, scale[nodes], spec['directional'])
    valid_unit = valid.any(axis=1) if horizon is None else valid[:, horizon]
    in_block = np.ones(len(y), bool) if block is None else block_ids == block
    eligible_left = int((spec['left_mask'] & valid_unit & in_block[:, None]).sum())
    eligible_right = int((spec['right_mask'] & valid_unit & in_block[:, None]).sum())
    result.update(nodes=int(len(np.unique(nodes[scored]))), node_blocks=int(len(np.unique(ids[scored]))),
        pairs_with_common_targets_before_minimum=int(before_min.sum()), min_pairs_per_scored_node_block=MIN_PAIRS,
        excluded_node_blocks_below_minimum=int(((counts > 0) & (counts < MIN_PAIRS)).sum()),
        history_eligible_left_units=int((spec['left_mask'] & in_block[:, None]).sum()),
        history_eligible_right_units=int((spec['right_mask'] & in_block[:, None]).sum()),
        eligible_left_units=eligible_left, eligible_right_units=eligible_right,
        matched_endpoint_fraction=2 * result['pairs_scored'] / max(1, eligible_left + eligible_right),
        matched_all_units_fraction=2 * result['pairs_scored'] / max(1, int(valid_unit[in_block].sum())),
        left_group='UP' if spec['directional'] else 'high_V', right_group='DOWN' if spec['directional'] else 'low_V')
    for index, name in enumerate(FEATURE_NAMES):
        a = spec['feature'][left, nodes, index][scored]
        b = spec['feature'][right, nodes, index][scored]
        gap = np.abs(a - b)
        result[name + '_left_mean'] = float(a.mean()) if len(a) else None
        result[name + '_right_mean'] = float(b.mean()) if len(b) else None
        result[name + '_mean_absolute_pair_gap'] = float(gap.mean()) if len(gap) else None
        result[name + '_max_absolute_pair_gap'] = float(gap.max()) if len(gap) else None
        if index in spec['columns']:
            tol = float(spec['calipers'][spec['columns'].index(index)])
            require(not len(gap) or gap.max() <= tol + 1e-12, 'Matched pair violates fixed caliper')
            result[name + '_caliper'] = tol
    node_rows = []
    for node in np.unique(nodes[scored]):
        here = nodes == node
        row = pair_statistics(ea[here], eb[here], common[here], scale[nodes[here]], spec['directional'])
        node_rows.append(dict(node_index=int(node), station_id=int(archive['node_ids'][node]), **row))
    for name in ('left_bias', 'right_bias', 'mae_left_minus_right', 'aligned_error',
                 'scaled_mae_left_minus_right', 'scaled_aligned_error'):
        values = [row[name] for row in node_rows if row[name] is not None]
        result['equal_node_' + name] = float(np.mean(values)) if values else None
    result['positive_node_mae_difference_fraction'] = float(np.mean([
        row['mae_left_minus_right'] > 0 for row in node_rows])) if node_rows else None
    result['underreaction_node_fraction'] = float(np.mean([
        row['left_bias'] < 0 and row['right_bias'] > 0 for row in node_rows])) if node_rows and spec['directional'] else None
    return result, node_rows


def trend_group_rows(archive, local, scale, high_bin):
    """Actual errors by historical direction. Unmatched descriptive context."""
    error = archive['pred__' + old.ARM].astype(np.float64) - archive['y'].astype(np.float64)
    labels, valid = classify_trend(local['trend']), archive['valid']
    counts = valid.sum(axis=1)
    scored = counts > 0
    def unit(values):
        return np.divide(np.where(valid, values, 0.).sum(axis=1), counts,
            out=np.full(counts.shape, np.nan), where=scored)
    mae, bias = unit(np.abs(error)), unit(error)
    rows = []
    for v_name, v_mask in (('all_V', np.ones(labels.shape, bool)), ('high_V', local['volatility_bin'] == high_bin),
                           ('low_V', local['volatility_bin'] == 0)):
        for direction, name in ((1, 'UP'), (-1, 'DOWN'), (0, 'FLAT')):
            base = v_mask & (labels == direction)
            for h in (None, *range(12)):
                mask = base & scored if h is None else base & valid[:, h]
                values_mae = mae[mask] if h is None else np.abs(error[:, h][mask])
                values_bias = bias[mask] if h is None else error[:, h][mask]
                units = int(mask.sum())
                scale_grid = np.broadcast_to(scale, mask.shape)[mask]
                rows.append(dict(volatility_group=v_name, trend_group=name, horizon='all' if h is None else h + 1,
                    scored_window_node_units=units, nodes=int(mask.any(axis=0).sum()),
                    actual_mae=float(values_mae.mean()) if units else None,
                    actual_bias=float(values_bias.mean()) if units else None,
                    scaled_mae=float((values_mae / scale_grid).mean()) if units else None,
                    interpretation='unmatched_description_not_adjusted_effect'))
    return rows


def analyze(archive, local, scale, high_bin, output):
    block_ids = np.empty(len(archive['y']), dtype=np.int8)
    for index, rows in enumerate(np.array_split(np.arange(len(block_ids)), 6)):
        block_ids[rows] = index
    tables = {k: [] for k in ('matched_overall', 'by_horizon', 'by_order_block', 'by_node', 'constructed_pairs')}
    for profile, multiplier in PROFILE_MULTIPLIERS.items():
        for contrast, spec in make_pairs(local, block_ids, high_bin, multiplier).items():
            common = dict(profile=profile, contrast=contrast)
            result, node_rows = summarize(spec, archive, scale, block_ids)
            tables['matched_overall'].append(dict(common, **result))
            tables['by_node'].extend(dict(common, **row) for row in node_rows)
            for h in range(12):
                row, _ = summarize(spec, archive, scale, block_ids, horizon=h)
                tables['by_horizon'].append(dict(common, horizon=h + 1, **row))
            for b in range(6):
                row, _ = summarize(spec, archive, scale, block_ids, block=b)
                tables['by_order_block'].append(dict(common, block=b + 1, **row))
            for left, right, node, block in spec['pairs']:
                row = dict(common, left_sample_id=int(archive['sample_ids'][left]),
                    right_sample_id=int(archive['sample_ids'][right]), station_id=int(archive['node_ids'][node]),
                    node_index=int(node), block=int(block) + 1)
                for index, name in enumerate(FEATURE_NAMES):
                    row[name + '_left'] = float(spec['feature'][left, node, index])
                    row[name + '_right'] = float(spec['feature'][right, node, index])
                tables['constructed_pairs'].append(row)
    for name, rows in tables.items():
        write_csv(output / (name + '.csv'), rows)
    write_csv(output / 'unmatched_trend_groups.csv', trend_group_rows(archive, local, scale, high_bin))
    return tables['matched_overall']


def report(result):
    fmt = lambda x: 'NA' if x is None else f'{x:.6f}'
    lines = ['# R_CONTINUE BEST matched trend-response audit', '',
        f"Cached pooled MAE: {result['replay']['pooled_mae']:.9f}; no inference or optimizer updates.",
        f"Selected history diagnostic: {result['history_directory']}", '',
        'Pairs are constructed from history only within the SAME NODE and SAME AUTHOR-ORDER BLOCK, without replacement.',
        'V high/low: match full-window mean/scale within 0.05. UP/DOWN: match latest observed flow/scale, V, |T| within (0.05,0.02,0.05).',
        'T = (last-three mean minus first-three mean)/training node scale; UP >0.05, DOWN <-0.05, equality is FLAT.',
        'The sensitivity profile doubles all calipers; it is not an alternative selected for better errors.',
        'Pairs stay fixed across horizons. Scoring uses shared valid horizons and >=5 scored pairs per node/block slice.',
        'First average common horizons within a pair, then weight pairs equally. Equal-node results are separate sensitivity summaries.', '',
        '| Profile / contrast | Scored pairs | Nodes | Endpoint coverage | Left bias | Right bias | MAE left-right | Underresponse score |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in result['matched_overall']:
        lines.append('| ' + row['profile'] + ' / ' + row['contrast'] + ' | ' + ' | '.join((
            str(row['pairs_scored']), str(row['nodes']), fmt(row['matched_endpoint_fraction']),
            fmt(row['left_bias']), fmt(row['right_bias']), fmt(row['mae_left_minus_right']), fmt(row['aligned_error']))) + ' |')
    lines += ['', '## Interpretation', '',
        'For directional comparisons LEFT=UP, RIGHT=DOWN. These are ACTUAL group biases, not a high-minus-low bias difference.',
        'Underresponse score (also named aligned_error in CSV) = (-UP_error + DOWN_error)/2. Positive values are underresponse-consistent; also inspect each actual bias, node breadth and coverage.',
        'For V comparisons LEFT=high V, RIGHT=low V. Positive MAE difference marks remaining difficulty after bounded level matching, not recoverable model gain.',
        'Check every continuous balance column, including uncontrolled mean/recent levels. Latest-level matching does not balance all history.',
        'Sparse support is NA. No p-values, correction fitting or automatic permission to train/extend a model is produced.',
        'Use by_horizon.csv, by_order_block.csv and by_node.csv to assess heterogeneity; the primary and doubled profiles compare different matched populations.',
        'After this audit, make a continue/stop decision on the local-response hypothesis; do not keep expanding old GW7 training or searching calipers.', '',
        '## Limitations', '']
    lines += ['- ' + text for text in LIMITATIONS]
    return '\n'.join(lines) + '\n'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--history-dir', type=Path, help='Completed old remaining-error child; auto-select latest matching completion if omitted')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    run, directory, output = (getattr(args, key).resolve() for key in ('run_dir', 'data_dir', 'output_dir'))
    require(output.parent == run and not output.exists(), 'Use a NEW direct child output directory of --run-dir')
    script_digest = sha256(Path(__file__))
    archive, local, scale, high_bin, inputs, metadata = load_inputs(run, directory, args.history_dir)
    require(sha256(Path(__file__)) == script_digest, 'Diagnostic source changed during input reconstruction')
    output.mkdir()
    try:
        manifest = dict(schema='gw7_trend_response_inputs_v1', input_sha256=inputs,
            script_sha256=script_digest, prior_source_sha256=sha256(Path(old.__file__)),
            feature_cache_reconstructed_exactly=True, protocol=dict(trend_deadband=DEADBAND,
                min_scored_pairs_per_node_block=MIN_PAIRS, profile_multipliers=PROFILE_MULTIPLIERS,
                volatility_level_caliper=.05, directional_calipers=[.05, .02, .05],
                directional_controls=['last_level', 'volatility', 'abs_trend'], history_only_pairing=True,
                no_replacement=True, same_node=True, same_author_order_block=True))
        write_json(output / 'input_manifest.json', manifest)
        overall = analyze(archive, local, scale, high_bin, output)
        for path, digest in inputs.items():
            require(sha256(path) == digest, 'Input changed during analysis: ' + path)
        require(sha256(Path(__file__)) == manifest['script_sha256'], 'Diagnostic source changed')
        result = dict(schema='gw7_matched_trend_response_v1', status='GW7_TREND_RESPONSE_COMPLETE',
            created_utc=datetime.now(timezone.utc).isoformat(), **metadata,
            matched_overall=overall, input_manifest_sha256=sha256(output / 'input_manifest.json'),
            feature_cache_reconstructed_exactly=True, exact_target_replay=True, cached_metric_replay=True,
            optimizer_updates=0, inference_calls=0, checkpoint_deserializations=0, gpu_used=False,
            test_evaluated=False, test_payload_deserialized=True, limitations=LIMITATIONS)
        (output / 'report.md').write_text(report(result), encoding='utf-8')
        write_json(output / 'summary.json', result)
        print(json.dumps(dict(status=result['status'], output_dir=str(output))), flush=True)
        return 0
    except BaseException as error:
        write_json(output / 'failure.json', dict(status='GW7_TREND_RESPONSE_FAILED',
            error_type=type(error).__name__, error=str(error), completed=False))
        raise


if __name__ == '__main__':
    sys.exit(main())
