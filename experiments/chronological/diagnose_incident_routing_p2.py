"""CPU-only descriptive diagnostics of the completed P2 pair's saved predictions."""

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.chronological.continue_incident_routing_p2 import (
    COMPLETE, VARIANTS, digest, load_summaries, require, validate_pair,
)


def read_predictions(path, expected_hash):
    require(digest(path) == expected_hash, f'Prediction checksum mismatch: {path}')
    with np.load(path, allow_pickle=False) as stored:
        arrays = {name: stored[name] for name in (
            'prediction', 'target', 'valid', 'associated', 'sample_indices', 'station_ids')}
    shape = arrays['prediction'].shape
    require(len(shape) == 4 and shape[1] == 12 and shape[3] == 1, 'Expected [samples,12,nodes,1]')
    for name in ('target', 'valid', 'associated'):
        require(arrays[name].shape == shape, f'Wrong shape: {name}')
    for name in ('valid', 'associated'):
        require(arrays[name].dtype == np.bool_, f'Expected boolean mask: {name}')
    for name, count in (('sample_indices', shape[0]), ('station_ids', shape[2])):
        require(arrays[name].shape == (count,) and
                len(np.unique(arrays[name])) == count, f'Invalid axis: {name}')
    require(np.isfinite(arrays['prediction']).all(), 'Nonfinite predictions')
    target = arrays['target'][arrays['valid']]
    require(np.isfinite(target).all() and (target >= 0).all(), 'Invalid observed targets')
    return arrays


def paired_predictions(left, right):
    for name in ('target', 'valid', 'associated', 'sample_indices', 'station_ids'):
        require(np.array_equal(left[name], right[name], equal_nan=True),
                f'Paired prediction axes/labels differ: {name}')


def event_sums(arrays, region):
    valid = arrays['valid'].copy()
    if region != 'all_nodes':
        associated = arrays['associated']
        valid &= associated if region == 'associated_nodes' else ~associated
    # Cast before subtraction, matching the original trainer's double precision metrics.
    error = np.abs(arrays['prediction'].astype(np.float64) - arrays['target'].astype(np.float64))
    return np.where(valid, error, 0.).sum(axis=(2, 3)), valid.sum(axis=(2, 3))


def aggregate(sums, counts):
    total, count = sums.sum(axis=0), counts.sum(axis=0)
    per = np.divide(total, count, out=np.full(12, np.nan), where=count > 0)
    def finite_mean(values):
        return float(values.mean()) if np.isfinite(values).all() else None
    return {
        'mae_macro': finite_mean(per),
        'mae_pooled': float(total.sum() / count.sum()) if count.sum() else None,
        'per_horizon_mae': [float(v) if np.isfinite(v) else None for v in per],
        'valid_count_per_horizon': count.tolist(),
        'h1_h3_mae_macro': finite_mean(per[:3]),
        'h4_h6_mae_macro': finite_mean(per[3:6]),
        'h7_h12_mae_macro': finite_mean(per[6:]),
    }


def compare(left, right):
    a, b = left['mae_macro'], right['mae_macro']
    gain = a - b if a is not None and b is not None else None
    return {'fixed': left, 'acdg': right, 'gain_fixed_minus_acdg': gain,
            'relative_gain_percent': 100 * gain / a if gain is not None and a else None}


def curve_diagnostics(summaries):
    rows, best, selected = [], {}, {}
    for variant, summary in summaries.items():
        require(summary['status'] == COMPLETE, 'Both runs must be complete')
        history = summary['history']
        epochs = summary['completed_epoch']
        require([row['epoch'] for row in history] == list(range(1, epochs + 1)),
                'Incomplete or unordered training history')
        require(summary['global_updates'] == epochs * 76, 'Unexpected optimizer update count')
        require((summary['stop_reason'] == 'max_epochs' and epochs == 100) or
                (summary['stop_reason'] == 'early_stopping' and summary['wait'] == 20),
                'Unexpected training termination')
        values = np.array([row['validation']['all_nodes']['mae_macro'] for row in history])
        require(np.isfinite(values).all(), 'Nonfinite learning curve')
        require(int(values.argmin()) + 1 == summary['best_epoch'] and
                np.isclose(values.min(), summary['best_metric'], rtol=0, atol=1e-10),
                'Best epoch/metric does not match history')
        best[variant] = np.minimum.accumulate(values)
        selected[variant] = {'best_epoch': summary['best_epoch'], 'completed_epoch': epochs,
                             'best_validation_mae': summary['best_metric'],
                             'stop_reason': summary['stop_reason']}
        for row, cumulative in zip(history, best[variant]):
            rows.append({'variant': variant, 'epoch': row['epoch'],
                         'train_mae': row['train']['mae_macro'],
                         'validation_mae': row['validation']['all_nodes']['mae_macro'],
                         'best_validation_mae_so_far': float(cumulative),
                         'learning_rate': row['learning_rate']})
    common = min(len(v) for v in best.values())
    # These comparisons describe already observed histories, never reselect the final models.
    budgets = sorted({common, *(s['best_epoch'] for s in summaries.values()
                               if s['best_epoch'] <= common)})
    comparisons = []
    for epoch in budgets:
        a, b = (float(best[v][epoch - 1]) for v in VARIANTS)
        comparisons.append({'through_epoch': epoch, 'fixed_best_so_far': a,
                            'acdg_best_so_far': b, 'gain_fixed_minus_acdg': a - b})
    return {'selected_models': selected, 'common_completed_epoch': common,
            'common_budget_best_so_far': comparisons,
            'status': 'DESCRIPTIVE_ONLY_NO_RESELECTION'}, rows


def read_metadata(data_dir, summary, sample_indices):
    package = data_dir / 'summary.json'
    require(digest(package) == summary['identity']['package_sha256']['summary.json'],
            'Data package manifest changed')
    manifest = data_dir / 'val_manifest.csv'
    meta = json.loads(package.read_text())
    require(digest(manifest) == meta['files']['val_manifest.csv'], 'Validation manifest changed')
    with manifest.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    require([int(row['sample_index']) for row in rows] == sample_indices.tolist(),
            'Validation manifest order differs from predictions')
    for row in rows:
        require(row['split'] == 'val', 'Non-validation sample in manifest')
        year, week, _ = datetime.fromisoformat(row['t0']).isocalendar()
        row['week'] = f'{year}-W{week:02d}'
    return rows, {str(package): digest(package), str(manifest): digest(manifest)}


def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def diagnose(root, data_dir, output):
    require(not output.exists(), 'Use a new diagnostics output directory')
    summaries = load_summaries(root)
    validate_pair(summaries)
    curves, curve_rows = curve_diagnostics(summaries)
    hashes = {str(root / v / 'summary.json'): digest(root / v / 'summary.json') for v in VARIANTS}
    arrays = {}
    for variant in VARIANTS:
        path = root / variant / 'best_validation_predictions.npz'
        checksum = summaries[variant]['best_validation_predictions_sha256']
        arrays[variant] = read_predictions(path, checksum)
        hashes[str(path)] = checksum
    paired_predictions(arrays['fixed'], arrays['acdg'])
    metadata, meta_hashes = read_metadata(data_dir, summaries['fixed'], arrays['fixed']['sample_indices'])
    hashes.update(meta_hashes)
    require(arrays['fixed']['prediction'].shape == (917, 12, 496, 1), 'Not the full P2 validation set')
    regions, horizon_rows, event_rows, weekly_rows, statistics = {}, [], [], [], {}
    for region in ('all_nodes', 'associated_nodes', 'nonassociated_nodes'):
        stats = {v: event_sums(arrays[v], region) for v in VARIANTS}
        statistics[region] = stats
        metrics = {v: aggregate(*stats[v]) for v in VARIANTS}
        for variant in VARIANTS:
            if region != 'nonassociated_nodes':
                original = summaries[variant]['best_validation_metrics'][region]
                require(np.allclose(metrics[variant]['per_horizon_mae'], original['per_horizon_mae'],
                                    rtol=0, atol=1e-8), f'Saved MAE mismatch: {variant}/{region}')
                require(np.isclose(metrics[variant]['mae_macro'], original['mae_macro'],
                                   rtol=0, atol=1e-8), 'Saved macro MAE mismatch')
                if region == 'all_nodes':
                    require(np.isclose(metrics[variant]['mae_macro'], summaries[variant]['best_metric'],
                                       rtol=0, atol=1e-8), 'Predictions differ from selected metric')
        regions[region] = compare(metrics['fixed'], metrics['acdg'])
        for h in range(12):
            a, b = (metrics[v]['per_horizon_mae'][h] for v in VARIANTS)
            horizon_rows.append({'region': region, 'horizon': h + 1,
                                 'fixed_mae': a, 'acdg_mae': b,
                                 'gain_fixed_minus_acdg': a - b if a is not None else None,
                                 'valid_count': metrics['fixed']['valid_count_per_horizon'][h]})
        for i, row in enumerate(metadata):
            values = {v: aggregate(stats[v][0][i:i + 1], stats[v][1][i:i + 1]) for v in VARIANTS}
            # Per-window macro uses the same 12-horizon definition; incomplete windows are null.
            pair = compare(values['fixed'], values['acdg'])
            event_rows.append({'sample_index': row['sample_index'], 't0': row['t0'],
                               'week': row['week'], 'region': region,
                               'fixed_mae_macro': values['fixed']['mae_macro'],
                               'acdg_mae_macro': values['acdg']['mae_macro'],
                               'gain_fixed_minus_acdg': pair['gain_fixed_minus_acdg']})
        for week in sorted({r['week'] for r in metadata}):
            mask = np.array([r['week'] == week for r in metadata])
            values = {v: aggregate(stats[v][0][mask], stats[v][1][mask]) for v in VARIANTS}
            weekly_rows.append({'week': week, 'region': region, 'windows': int(mask.sum()),
                                'fixed_mae_macro': values['fixed']['mae_macro'],
                                'acdg_mae_macro': values['acdg']['mae_macro'],
                                'gain_fixed_minus_acdg': compare(values['fixed'], values['acdg'])['gain_fixed_minus_acdg']})
    # Regional contributions sum to the original all-node horizon-macro gain, despite unequal masks.
    total_counts = statistics['all_nodes']['fixed'][1].sum(axis=0)
    contribution = {}
    for region in ('associated_nodes', 'nonassociated_nodes'):
        stats = statistics[region]
        difference = (stats['fixed'][0] - stats['acdg'][0]).sum(axis=0)
        contribution[region] = float((difference / total_counts).mean())
    require(np.isclose(sum(contribution.values()), regions['all_nodes']['gain_fixed_minus_acdg'],
                       rtol=0, atol=1e-8), 'Regional contribution decomposition failed')
    for path, expected in hashes.items():
        require(digest(path) == expected, f'Input changed during analysis: {path}')
    window_distribution = {}
    for region in regions:
        gains = np.array([r['gain_fixed_minus_acdg'] for r in event_rows
                          if r['region'] == region and r['gain_fixed_minus_acdg'] is not None])
        window_distribution[region] = {
            'windows_with_12_valid_horizon_metrics': int(len(gains)),
            'improved_windows': int((gains > 1e-10).sum()),
            'worsened_windows': int((gains < -1e-10).sum()),
            'tied_windows': int((np.abs(gains) <= 1e-10).sum()),
            'gain_quantiles_0_25_50_75_100': np.quantile(gains, [0, .25, .5, .75, 1]).tolist() if len(gains) else None,
        }
    report = {
        'status': 'P2_SAVED_PREDICTION_DIAGNOSTICS_COMPLETE',
        'scientific_status': 'POSTHOC_SINGLE_SEED_REUSED_VALIDATION_DESCRIPTIVE_ONLY',
        'pair_directory': str(root), 'regions': regions, 'learning_curves': curves,
        'regional_contribution_to_all_node_mae_gain': contribution,
        'window_gain_distribution': window_distribution, 'weekly_comparisons': weekly_rows,
        'input_sha256': hashes, 'diagnostic_source_sha256': digest(Path(__file__)),
        'interpretation': [
            'Positive gains favor ACDG; final models remain those selected by the original protocol.',
            'Nonassociated nodes are within incident windows, not a no-incident control cohort.',
            'Weekly/window breakdowns are descriptive; no independent-test, significance or equivalence claim.',
            'Calendar groups use manifest nominal t0; this does not certify timezone/DST provenance.',
        ],
    }
    output.mkdir(parents=True)
    for name, rows in (('learning_curves.csv', curve_rows), ('per_horizon.csv', horizon_rows),
                       ('per_window.csv', event_rows), ('per_week.csv', weekly_rows)):
        write_csv(output / name, rows)
    report['output_sha256'] = {p.name: digest(p) for p in output.glob('*.csv')}
    (output / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('status', 'scientific_status', 'regions',
                                           'learning_curves', 'regional_contribution_to_all_node_mae_gain',
                                           'window_gain_distribution', 'weekly_comparisons')},
                     indent=2, allow_nan=False))
    print(f'Saved diagnostics: {output}', flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    diagnose(args.run_dir.resolve(), args.data_dir.resolve(), args.output_dir.resolve())


if __name__ == '__main__':
    main()
