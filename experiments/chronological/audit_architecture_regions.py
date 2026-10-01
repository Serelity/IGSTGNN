"""Analyze saved v12a regional errors with shared paired week bootstraps (NumPy only)."""

import argparse
import csv
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import socket
import sys
import time
import traceback

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.chronological.audit_matched_controls import read_csv, sha256

PROTOCOL = Path(__file__).with_name('architecture_region_audit_v12b.json')
PROTOCOL_SHA256 = 'b7d5792496c549237886a620aa6b0bffe6633ff49f743656c41263b3dbd1e4ce'


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def load_protocol(path):
    if sha256(path) != PROTOCOL_SHA256:
        raise ValueError('Frozen v12b protocol changed')
    return json.loads(Path(path).read_text(encoding='utf-8'))


def validate_source_summary(summary, protocol):
    expected = {'status': 'ARCHITECTURE_MECHANISM_AUDIT_COMPLETE',
                'protocol_id': 'contra_v8_architecture_mechanism_audit_v12a',
                'protocol_sha256': protocol['v12a_protocol_sha256'],
                'engineering_check': False, 'full_cohort_evaluated': True,
                'checkpoint_state_unchanged': True, 'model_training_performed': False,
                'real_data_gradient_computation_performed': False,
                'validation_arrays_read': False, 'test_split_read': False}
    for key, value in expected.items():
        if summary.get(key) != value or type(summary.get(key)) is not type(value):
            raise ValueError(f'v12a source identity/boundary mismatch: {key}')
    if summary.get('checkpoint', {}).get('best_model_sha256') != protocol['checkpoint_sha256']:
        raise ValueError('v12a checkpoint identity changed')
    if summary.get('code_sha256', {}).get('experiments/chronological/audit_architecture_mechanisms.py') != protocol['v12a_auditor_sha256']:
        raise ValueError('v12a source implementation changed')
    if set(summary.get('results', {})) != set(protocol['cohort_samples']):
        raise ValueError('v12a cohort set changed')


def load_artifact(path, identities, source_indices, protocol, source_result):
    regions = protocol['base_regions'] + [f'all_h{h}' for h in range(1, 13)]
    n, p, r = len(identities), len(protocol['paths']), len(regions)
    with np.load(path, allow_pickle=False) as stored:
        if stored['paths'].tolist() != protocol['paths'] or stored['regions'].tolist() != regions:
            raise ValueError('v12a path or region axes changed')
        ids, indices = stored['positive_sample_index'], stored['source_index']
        if (ids.dtype.kind not in 'iu' or indices.dtype.kind not in 'iu' or
                ids.shape != (n,) or indices.shape != (n,) or
                not np.array_equal(ids, identities) or not np.array_equal(indices, source_indices)):
            raise ValueError('v12a sample identities/order/source indices changed')
        arrays = {name: stored[name].copy() for name in (
            'absolute_error_sums', 'valid_counts', 'prediction_counts')}
    errors, counts, predicted = (arrays[k] for k in ('absolute_error_sums', 'valid_counts', 'prediction_counts'))
    if errors.shape != (n, p, r) or counts.shape != (n, r) or predicted.shape != counts.shape:
        raise ValueError('v12a aggregate shape mismatch')
    if counts.dtype.kind not in 'iu' or predicted.dtype.kind not in 'iu' or errors.dtype.kind != 'f':
        raise ValueError('v12a counts/errors dtype mismatch')
    if (not np.isfinite(errors).all() or (errors < 0).any() or (counts < 0).any() or
            (counts > predicted).any() or (predicted < 0).any()):
        raise ValueError('Invalid v12a error totals or counts')
    if not np.all(predicted[:, 0] == protocol['horizon_count'] * protocol['node_count']):
        raise ValueError('v12a prediction cell count changed')
    if np.any(np.where(counts[:, None, :] == 0, errors, 0.) != 0):
        raise ValueError('Empty target regions have nonzero errors')
    # All regions must partition identically; derived regions never double-count a cell.
    for columns in (list(range(1, 6)), list(range(6, 18))):
        for values in (counts, predicted):
            if not np.array_equal(values[:, columns].sum(-1), values[:, 0]):
                raise ValueError('v12a region count partitions disagree')
        if not np.allclose(errors[:, :, columns].sum(-1), errors[:, :, 0], rtol=1e-10, atol=1e-7):
            raise ValueError('v12a regional error partitions disagree')
    for j, region in enumerate(regions):
        declared = source_result['regions'][region]
        if int(counts[:, j].sum()) != declared['valid_target_cells'] or int(predicted[:, j].sum()) != declared['prediction_cells']:
            raise ValueError('v12a summary and stored counts disagree')
        for i, name in enumerate(protocol['paths']):
            observed = errors[:, i, j].sum() / counts[:, j].sum() if counts[:, j].sum() else None
            expected = declared['descriptive_mae'][name]
            if (observed is None) != (expected is None) or (
                    observed is not None and not np.isclose(observed, expected, rtol=1e-10, atol=1e-10)):
                raise ValueError('v12a summary and stored MAE disagree')
    for name, parts in protocol['derived_regions'].items():
        columns = [regions.index(part) for part in parts]
        errors = np.concatenate([errors, errors[:, :, columns].sum(-1, keepdims=True)], axis=-1)
        counts = np.concatenate([counts, counts[:, columns].sum(-1, keepdims=True)], axis=-1)
        predicted = np.concatenate([predicted, predicted[:, columns].sum(-1, keepdims=True)], axis=-1)
        regions.append(name)
    return {'ids': np.asarray(identities), 'source_indices': np.asarray(source_indices),
            'errors': errors, 'counts': counts, 'prediction_counts': predicted, 'regions': regions}


def load_inputs(data_dir, primary_dir, secondary_dir, v12a_dir, protocol):
    hashes = {}

    def verify(path, expected):
        path = Path(path)
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f'Input fingerprint mismatch: {path}')
        hashes[str(path.resolve())] = actual

    baseline_path = PROTOCOL.with_name('incident_branch_materialize_v6a.json')
    verify(baseline_path, protocol['baseline_protocol_sha256'])
    baseline = json.loads(baseline_path.read_text())
    data_dir, primary_dir, secondary_dir, v12a_dir = map(Path, (data_dir, primary_dir, secondary_dir, v12a_dir))
    verify(data_dir / 'summary.json', baseline['positive_package']['summary_sha256'])
    metadata = json.loads((data_dir / 'summary.json').read_text())
    verify(data_dir / 'train_manifest.csv', metadata['files']['train_manifest.csv'])
    verify(primary_dir / 'train_control_manifest.csv', baseline['primary_control_inputs']['train_control_manifest.csv'])
    verify(secondary_dir / 'train_second_control_manifest.csv', baseline['secondary_control_inputs']['train_second_control_manifest.csv'])
    summary_path = v12a_dir / 'summary.json'
    summary = json.loads(summary_path.read_text())
    hashes[str(summary_path.resolve())] = sha256(summary_path)
    validate_source_summary(summary, protocol)
    positive = read_csv(data_dir / 'train_manifest.csv')
    primary = read_csv(primary_dir / 'train_control_manifest.csv')
    secondary = read_csv(secondary_dir / 'train_second_control_manifest.csv')
    if any(row['split'] != 'train' for rows in (positive, primary, secondary) for row in rows):
        raise ValueError('Only train manifest rows are allowed')
    full_ids = [int(row['sample_index']) for row in positive]
    common_ids = [int(row['positive_sample_index']) for row in secondary]
    full_positions = {sample: i for i, sample in enumerate(full_ids)}
    primary_by_id = {int(row['positive_sample_index']): row for row in primary}
    if len(full_positions) != len(full_ids) or len(set(common_ids)) != len(common_ids) or len(primary_by_id) != len(primary):
        raise ValueError('Duplicate manifest sample identities')
    if len(full_ids) != protocol['cohort_samples']['incident_full'] or len(common_ids) != protocol['cohort_samples']['incident']:
        raise ValueError('Frozen cohort sample counts changed')
    expected_sources = {'incident_full': list(range(len(full_ids))), 'incident': [],
                        'primary_control': [], 'secondary_control': list(range(len(common_ids)))}
    for j, (sample, row) in enumerate(zip(common_ids, secondary)):
        if sample not in full_positions or sample not in primary_by_id or int(row['control_index']) != j:
            raise ValueError('Matched identity missing or secondary index changed')
        pos, c1 = positive[full_positions[sample]], primary_by_id[sample]
        if len({pos['incident_id'], row['incident_id'], c1['incident_id']}) != 1 or len({pos['t0'], row['positive_t0'], c1['positive_t0']}) != 1:
            raise ValueError('Matched event or positive clock mismatch')
        expected_sources['incident'].append(full_positions[sample])
        expected_sources['primary_control'].append(int(c1['control_index']))
    records = {}
    for cohort, count in protocol['cohort_samples'].items():
        filename = f'train_{cohort}_mechanisms.npz'
        specification = summary['outputs'][filename]
        if specification['samples'] != count or summary['results'][cohort]['samples'] != count:
            raise ValueError('v12a source sample count mismatch')
        verify(v12a_dir / filename, specification['sha256'])
        ids = full_ids if cohort == 'incident_full' else common_ids
        records[cohort] = load_artifact(v12a_dir / filename, ids, expected_sources[cohort], protocol,
                                       summary['results'][cohort])
    selected = np.asarray(expected_sources['incident'])
    for key in ('counts', 'prediction_counts'):
        if not np.array_equal(records['incident'][key], records['incident_full'][key][selected]):
            raise ValueError('Full and common incident counts disagree')
    if not np.allclose(records['incident']['errors'], records['incident_full']['errors'][selected],
                       rtol=1e-5, atol=1e-3):
        raise ValueError('Full and common incident errors disagree beyond replay tolerance')
    for cohort in ('primary_control', 'secondary_control'):
        if not np.array_equal(records[cohort]['prediction_counts'], records['incident']['prediction_counts']):
            raise ValueError('Matched cohort geometric support changed')
    times = {int(row['sample_index']): row['t0'] for row in positive}
    return records, times, hashes


def week_grid(times):
    mondays = {}
    for sample, value in times.items():
        day = datetime.fromisoformat(value).date()
        mondays[sample] = day - timedelta(days=day.weekday())
    first, last = min(mondays.values()), max(mondays.values())
    grid = [first + timedelta(weeks=i) for i in range((last - first).days // 7 + 1)]
    if len(grid) < 4:
        raise ValueError('Need at least four calendar weeks for the block sensitivity')
    positions = {sample: (monday - first).days // 7 for sample, monday in mondays.items()}
    return [f'{day.isocalendar().year}-W{day.isocalendar().week:02d}' for day in grid], positions


def bootstrap_weights(week_count, draws, seed, block_weeks=1):
    if not 1 <= block_weeks <= week_count or draws < 1:
        raise ValueError('Invalid bootstrap dimensions')
    rng = np.random.default_rng(seed)
    blocks = (week_count + block_weeks - 1) // block_weeks
    starts = rng.integers(0, week_count, size=(draws, blocks))
    selected = ((starts[..., None] + np.arange(block_weeks)) % week_count).reshape(draws, -1)[:, :week_count]
    weights = np.zeros((draws, week_count), dtype=np.int64)
    np.add.at(weights, (np.arange(draws)[:, None], selected), 1)
    return weights


def ratio(numerator, denominator):
    numerator, denominator = np.asarray(numerator), np.asarray(denominator)
    shape = np.broadcast_shapes(numerator.shape, denominator.shape)
    return np.divide(numerator, denominator, out=np.full(shape, np.nan), where=denominator > 0)


def interval(draws, confidence, minimum_fraction):
    valid = np.isfinite(draws)
    count = int(valid.sum())
    if count / len(draws) < minimum_fraction:
        return {'status': 'INSUFFICIENT_VALID_DRAWS', 'valid_draws': count,
                'ci_low': None, 'ci_high': None}
    low, high = np.quantile(draws[valid], [(1 - confidence) / 2, (1 + confidence) / 2])
    return {'status': 'OK', 'valid_draws': count, 'ci_low': float(low), 'ci_high': float(high)}


def finite_number(value):
    return float(value) if np.isfinite(value) else None


def analyze(records, times, protocol, progress=lambda *args, **kwargs: None):
    weeks, positions = week_grid(times)
    bspec = protocol['bootstrap']
    resampling = {
        'week': bootstrap_weights(len(weeks), bspec['draws'], bspec['seed']),
        'four_week_block': bootstrap_weights(len(weeks), bspec['draws'], bspec['seed'] + 1,
                                             bspec['sensitivity_circular_block_weeks']),
    }
    results, stored_draws, weekly_output, csv_rows = {}, {}, {}, []
    comparisons = list(protocol['comparisons'])
    regions = records['incident_full']['regions']

    def ci(values):
        return interval(values, bspec['confidence'], bspec['minimum_valid_draw_fraction'])

    for cohort, record in records.items():
        index = np.asarray([positions[int(sample)] for sample in record['ids']], dtype=np.int64)
        errors, counts = record['errors'], record['counts']
        gains = np.stack([errors[:, protocol['paths'].index(left)] - errors[:, protocol['paths'].index(right)]
                          for left, right in protocol['comparisons'].values()], axis=1)
        wg = np.zeros((len(weeks), len(comparisons), len(regions)))
        wc = np.zeros((len(weeks), len(regions)))
        np.add.at(wg, index, gains)
        np.add.at(wc, index, counts)
        # Keep the event-weighted estimand separate from pooled cell weighting.
        event_gains = ratio(gains, counts[:, None, :])
        we = np.zeros_like(wg)
        wn = np.zeros_like(wc)
        np.add.at(we, index, np.nan_to_num(event_gains, nan=0.))
        np.add.at(wn, index, counts > 0)
        weekly_output[cohort] = {'gain_sums': wg, 'valid_counts': wc,
                                 'event_gain_sums': we, 'evaluable_events': wn}
        sampled = {}
        for method, weights in resampling.items():
            sampled_gain = (weights @ wg.reshape(len(weeks), -1)).reshape(-1, len(comparisons), len(regions))
            sampled_counts = weights @ wc
            sampled_events = (weights @ we.reshape(len(weeks), -1)).reshape(sampled_gain.shape)
            sampled[method] = {
                'pooled': ratio(sampled_gain, sampled_counts[:, None, :]),
                'equal_event': ratio(sampled_events, (weights @ wn)[:, None, :]),
                'global_units': ratio(sampled_gain, sampled_counts[:, None, :1]),
            }
        stored_draws[cohort] = sampled
        pooled = ratio(wg.sum(0), wc.sum(0)[None, :])
        event = ratio(we.sum(0), wn.sum(0)[None, :])
        global_units = ratio(wg.sum(0), wc.sum(0)[None, :1])
        results[cohort] = {'samples': len(index), 'nonempty_positive_weeks': len(set(index)),
                           'regions': {}}
        for r, region in enumerate(regions):
            total_count = int(counts[:, r].sum())
            item = {'valid_cells': total_count, 'evaluable_samples': int((counts[:, r] > 0).sum()),
                    'mae': {name: finite_number(ratio(errors[:, p, r].sum(), np.asarray(total_count)))
                            for p, name in enumerate(protocol['paths'])}, 'comparisons': {}}
            for c, comparison in enumerate(comparisons):
                effect = {'gain_raw_mae': finite_number(pooled[c, r]),
                          'equal_event_gain_raw_mae': finite_number(event[c, r]),
                          'regional_gain_in_global_mae_units': finite_number(global_units[c, r]),
                          'intervals': {method: {name: ci(value[:, c, r]) for name, value in values.items()}
                                        for method, values in sampled.items()}}
                item['comparisons'][comparison] = effect
                csv_rows.append({'cohort': cohort, 'region': region, 'comparison': comparison,
                    'gain_raw_mae': effect['gain_raw_mae'], 'valid_cells': total_count,
                    'equal_event_gain_raw_mae': effect['equal_event_gain_raw_mae'],
                    'regional_gain_in_global_mae_units': effect['regional_gain_in_global_mae_units'],
                    'week_ci_low': effect['intervals']['week']['pooled']['ci_low'],
                    'week_ci_high': effect['intervals']['week']['pooled']['ci_high'],
                    'block_ci_low': effect['intervals']['four_week_block']['pooled']['ci_low'],
                    'block_ci_high': effect['intervals']['four_week_block']['pooled']['ci_high']})
            results[cohort]['regions'][region] = item
        progress('cohort_analyzed', cohort=cohort, samples=len(index))
    matched = {}
    for r, region in enumerate(regions):
        matched[region] = {}
        for c, comparison in enumerate(comparisons):
            effects = [results[name]['regions'][region]['comparisons'][comparison]
                       for name in ('incident', 'primary_control', 'secondary_control')]
            def point(field):
                values = [effect[field] for effect in effects]
                return values[0] - (values[1] + values[2]) / 2 if all(v is not None for v in values) else None
            matched[region][comparison] = {
                'incident_minus_mean_control_gain': point('gain_raw_mae'),
                'equal_event_incident_minus_mean_control_gain': point('equal_event_gain_raw_mae'),
                'intervals': {method: {estimand: ci(
                    stored_draws['incident'][method][estimand][:, c, r] -
                    (stored_draws['primary_control'][method][estimand][:, c, r] +
                     stored_draws['secondary_control'][method][estimand][:, c, r]) / 2)
                    for estimand in ('pooled', 'equal_event')} for method in resampling},
            }
    output_arrays = {'weeks': np.asarray(weeks), 'regions': np.asarray(regions),
                     'comparisons': np.asarray(comparisons),
                     **{f'bootstrap_{method}_weights': value for method, value in resampling.items()}}
    for cohort, values in weekly_output.items():
        output_arrays.update({f'{cohort}_{name}': value for name, value in values.items()})
    return {'weeks': weeks, 'results': results, 'matched_benefit_difference': matched}, csv_rows, output_arrays


def report(summary):
    print('status:', summary['status'])
    print('protocol_sha256:', summary['protocol_sha256'])
    print('recommendation:', summary['recommendation'])
    print('TRAIN-ONLY / POST-HOC / POINTWISE CIs / NO MODEL LOADING')
    for cohort, result in summary['results'].items():
        print(f'\n[{cohort}] samples={result["samples"]} weeks={result["nonempty_positive_weeks"]}')
        for region in ('all', 'candidate_h1_h6', 'candidate_h7_h12', 'noncandidate_all'):
            effect = result['regions'][region]['comparisons']['icsf_given_normalization']
            week = effect['intervals']['week']['pooled']
            block = effect['intervals']['four_week_block']['pooled']
            print(region, 'ICSF gain=', effect['gain_raw_mae'],
                  'week_CI=', [week['ci_low'], week['ci_high']],
                  'block_CI=', [block['ci_low'], block['ci_high']],
                  'global_units=', effect['regional_gain_in_global_mae_units'])
        for name in ('full_given_normalization', 'tiid_given_icsf', 'native_graph_vs_replay'):
            effect = result['regions']['candidate_h1_h6']['comparisons'][name]
            interval_ = effect['intervals']['week']['pooled']
            print('candidate_h1_h6', name, 'gain=', effect['gain_raw_mae'],
                  'week_CI=', [interval_['ci_low'], interval_['ci_high']])
    print('\nMATCHED INCIDENT MINUS MEAN CONTROL BENEFIT (not causal)')
    for region in ('candidate_h1_h6', 'candidate_h7_h12', 'noncandidate_all'):
        effect = summary['matched_benefit_difference'][region]['icsf_given_normalization']
        print(region, json.dumps(effect))


def run(data_dir, primary_dir, secondary_dir, v12a_dir, output, protocol_path=PROTOCOL):
    protocol = load_protocol(protocol_path)
    output = Path(output)
    partial = output.with_name(output.name + '.partial')
    if output.exists() or partial.exists():
        raise FileExistsError('v12b output or .partial exists; use a new name')
    partial.mkdir(parents=True)
    start = time.monotonic()
    identity = {'host': socket.gethostname(), 'pid': os.getpid(),
                'started_utc': datetime.now(timezone.utc).isoformat()}
    def progress(stage, **fields):
        value = {**identity, 'stage': stage, 'elapsed_seconds': time.monotonic() - start, **fields}
        write_json(partial / 'progress.json', value)
        print(json.dumps(value), flush=True)
    try:
        progress('verifying_saved_statistics')
        records, times, hashes = load_inputs(data_dir, primary_dir, secondary_dir, v12a_dir, protocol)
        progress('inputs_verified')
        analysis, csv_rows, arrays = analyze(records, times, protocol, progress)
        with (partial / 'weekly_statistics.npz').open('wb') as stream:
            np.savez_compressed(stream, **arrays)
        with (partial / 'regional_comparisons.csv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
            writer.writeheader()
            writer.writerows(csv_rows)
        summary = {**analysis, 'status': 'ARCHITECTURE_REGION_AUDIT_COMPLETE',
            'protocol_id': protocol['protocol_id'], 'protocol_sha256': sha256(protocol_path),
            'analysis_type': 'posthoc_exploratory', 'validation_arrays_read': False,
            'test_split_read': False, 'raw_traffic_arrays_read': False, 'model_loaded': False,
            'model_training_performed': False, 'independent_confirmation': False,
            'main_training_ready': False, 'recommendation': protocol['decision'],
            'primary_descriptive_endpoint': protocol['primary_descriptive_endpoint'],
            'bootstrap': protocol['bootstrap'], 'interpretation': protocol['interpretation'],
            'inputs': hashes, 'code_sha256': sha256(__file__),
            'environment': {**identity, 'python': sys.version, 'numpy': np.__version__},
            'outputs': {name: sha256(partial / name) for name in ('weekly_statistics.npz', 'regional_comparisons.csv')}}
        write_json(partial / 'summary.json', summary)
        if output.exists():
            raise FileExistsError('Final output appeared; refusing overwrite')
        partial.rename(output)
    except BaseException as error:
        write_json(partial / 'failure.json', {**identity, 'error': str(error), 'traceback': traceback.format_exc()})
        raise
    report(summary)
    print(f'Saved v12b regional audit: {output / "summary.json"}', flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    execute = commands.add_parser('run')
    for name in ('data-dir', 'primary-control-dir', 'secondary-control-dir', 'v12a-dir', 'output'):
        execute.add_argument('--' + name, type=Path, required=True)
    execute.add_argument('--protocol', type=Path, default=PROTOCOL)
    view = commands.add_parser('report')
    view.add_argument('summary', type=Path)
    args = parser.parse_args()
    if args.command == 'report':
        report(json.loads(args.summary.read_text(encoding='utf-8')))
    else:
        run(args.data_dir, args.primary_control_dir, args.secondary_control_dir, args.v12a_dir,
            args.output, args.protocol)


if __name__ == '__main__':
    main()
