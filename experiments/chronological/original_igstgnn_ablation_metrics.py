"""Descriptive metrics for twelve original-IGSTGNN checkpoint interventions.

The comparison sign is always intervention error minus FULL error: a positive
utility means FULL is better. These are paired checkpoint sensitivities, not
additive forecast contributions, retrained deletion results, or causal effects.
"""
import csv
import json
import math
from pathlib import Path

import numpy as np


SCHEMA = 'original_igstgnn_checkpoint_screen_metrics_v1'
CASE_NAMES = (
    'FULL', 'DIF_FORECAST_READOUT_OFF', 'INH_FORECAST_READOUT_OFF', 'ICSF_ONLY',
    'NORM_ONLY', 'TIID_ONLY_ON_NORM', 'RAW_ALL_REPORT_OFF', 'PRE_BANKS_ZERO',
    'DYN_BANKS_ZERO', 'STA_BANKS_ZERO', 'GRU_TRANSITION_IDENTITY', 'ATTENTION_QUERY_IDENTITY',
)
FLOW_EDGES = (0., 5., 20., 50., 100., 200., float('inf'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _plain(value):
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        require(math.isfinite(value), 'Nonfinite report value')
    return value


def _ratio(numerator, denominator):
    return float(numerator/denominator) if denominator else None


def _macro(values, counts):
    return float(np.mean(values)) if bool((np.asarray(counts) > 0).all()) else None


def _roads(roads, nodes):
    if roads is None:
        return {}
    if not isinstance(roads, dict):
        labels = np.asarray(roads)
        require(labels.shape == (nodes,), 'Road labels must follow the station axis')
        roads = {str(label): np.flatnonzero(labels == label) for label in np.unique(labels)}
    result = {}
    for name, values in roads.items():
        name = str(name)
        require(name and name not in result, 'Road group names must be unique and nonempty')
        values = np.asarray(values)
        if values.dtype == np.bool_:
            require(values.shape == (nodes,), 'Road mask must have one value per station')
            mask = values.copy()
        else:
            if values.ndim == 1 and values.size == 0:
                values = values.astype(np.int64)
            require(values.ndim == 1 and values.dtype.kind in 'iu' and ((values >= 0) & (values < nodes)).all()
                and len(np.unique(values)) == len(values), 'Road members must be unique station-axis positions')
            mask = np.zeros(nodes, bool)
            mask[values.astype(np.int64)] = True
        result[name] = mask
    return result


def _validate(predictions, target, valid, sample_indices, station_ids, case_order, full_case):
    require(isinstance(predictions, dict), 'Predictions must be a case dictionary')
    order = tuple(CASE_NAMES if case_order is None else case_order)
    require(len(order) == 12 and len(set(order)) == 12 and set(order) == set(CASE_NAMES)
        and set(predictions) == set(order) and full_case == 'FULL', 'Expected all twelve original IGSTGNN cases including FULL')
    target, valid = np.asarray(target), np.asarray(valid)
    require(target.ndim == 3 and all(size > 0 for size in target.shape) and target.dtype.kind in 'fiu'
        and valid.shape == target.shape and valid.dtype == np.bool_, 'Expected target [sample,horizon,station] and matching Boolean original mask')
    require(np.isfinite(target).all() and valid.any() and (target[valid] > 0).all(),
        'Original masked MAPE requires finite arrays and positive valid traffic targets')
    arrays = {name: np.asarray(predictions[name]) for name in order}
    require(all(value.shape == target.shape and value.dtype.kind in 'fiu' and np.isfinite(value).all()
        for value in arrays.values()), 'All prediction cases must have identical finite sample/horizon/station axes')
    indices = np.asarray(sample_indices)
    require(indices.shape == (len(target),) and indices.dtype.kind in 'iu' and (indices >= 0).all()
        and len(np.unique(indices)) == len(indices), 'Sample indices must be unique nonnegative integers')
    stations = np.arange(target.shape[2], dtype=np.int64) if station_ids is None else np.asarray(station_ids)
    require(stations.shape == (target.shape[2],) and stations.dtype.kind in 'iu'
        and len(np.unique(stations)) == len(stations), 'Station IDs must be a unique integer station axis')
    return arrays, target, valid, indices, stations, order


def _window_scores(prediction, target, valid):
    rows, horizons, _ = target.shape
    sums = np.zeros((rows, horizons), np.float64)
    counts = valid.sum(2)
    for h in range(horizons):
        error = np.abs(prediction[:, h].astype(np.float64)-target[:, h].astype(np.float64))
        sums[:, h] = np.where(valid[:, h], error, 0.).sum(1)
    scores = np.full(rows, np.nan)
    complete = (counts > 0).all(1)
    scores[complete] = (sums[complete]/counts[complete]).mean(1)
    return scores


def _case_summary(sums, counts):
    absolute, squared, percentage = (sums[name] for name in ('absolute', 'squared', 'percentage'))
    counts = np.asarray(counts, np.int64)
    ratio = lambda values: np.divide(values, counts, out=np.zeros_like(values), where=counts > 0)
    mae, mse, mape = ratio(absolute), ratio(squared), ratio(percentage)
    count = int(counts.sum())
    pooled_mse = _ratio(squared.sum(), count)
    return dict(valid_count=count, valid_count_per_horizon=counts.tolist(),
        mae_pooled=_ratio(absolute.sum(), count), rmse_pooled=None if pooled_mse is None else math.sqrt(pooled_mse),
        mape_pooled=_ratio(percentage.sum(), count), mae_macro=_macro(mae, counts),
        rmse_macro=_macro(np.sqrt(mse), counts), mape_macro=_macro(mape, counts),
        horizons=[dict(step=h+1, valid_count=int(counts[h]), mae=float(mae[h]) if counts[h] else None,
            rmse=float(np.sqrt(mse[h])) if counts[h] else None, mape=float(mape[h]) if counts[h] else None)
            for h in range(len(counts))])


def _comparison_summary(sums, counts, global_counts):
    counts = np.asarray(counts, np.int64)
    count, global_count = int(counts.sum()), int(global_counts.sum())
    net, positive, negative = (sums[key] for key in ('net', 'positive', 'negative'))
    ratio = lambda values: np.divide(values, counts, out=np.zeros_like(values), where=counts > 0)
    denominator = float(positive.sum()+negative.sum())
    return dict(valid_count=count, valid_count_per_horizon=counts.tolist(),
        mae_gain_pooled=_ratio(net.sum(), count), mae_gain_macro=_macro(ratio(net), counts),
        gain_positive=_ratio(positive.sum(), count), gain_negative=_ratio(negative.sum(), count),
        cancellation_fraction=float(np.clip(1.-abs(float(net.sum()))/denominator, 0., 1.)) if denominator else (0. if count else None),
        global_mae_contribution=_ratio(net.sum(), global_count) if count else None,
        prediction_signed_mean=_ratio(sums['signed'].sum(), count),
        prediction_absolute_mean=_ratio(sums['absolute'].sum(), count),
        prediction_absolute_max=float(sums['absolute_max'].max()) if count else None,
        prediction_up_fraction=_ratio(sums['up'].sum(), count), prediction_down_fraction=_ratio(sums['down'].sum(), count),
        prediction_response_fraction=_ratio(sums['response'].sum(), count),
        improved_fraction=_ratio(sums['improved'].sum(), count), worsened_fraction=_ratio(sums['worsened'].sum(), count),
        horizons=[dict(step=h+1, valid_count=int(counts[h]), mae_gain=float(ratio(net)[h]) if counts[h] else None,
            gain_positive=float(ratio(positive)[h]) if counts[h] else None,
            gain_negative=float(ratio(negative)[h]) if counts[h] else None,
            global_mae_contribution=_ratio(net[h], int(global_counts[h])) if counts[h] else None) for h in range(len(counts))])


def summarize_predictions(predictions, target, valid, sample_indices, station_ids=None, roads=None, *,
                          case_order=None, full_case='FULL', response_tol=.001, hard_fraction=.10):
    """Summarize already saved paired predictions without any model calls.

    ``roads`` is a mapping to integer station-axis positions or Boolean masks,
    or one road label per station. Masks are supplied by the original engine;
    this module never derives a new validity mask from predictions or labels.
    True-flow groups are posthoc diagnostics, never inputs to prediction.
    """
    arrays, target, valid, indices, stations, order = _validate(
        predictions, target, valid, sample_indices, station_ids, case_order, full_case)
    require(np.isfinite(response_tol) and response_tol >= 0 and 0 < hard_fraction <= 1, 'Invalid response tolerance/hard fraction')
    rows, horizons, nodes = target.shape
    road_masks = _roads(roads, nodes)
    full_window = _window_scores(arrays[full_case], target, valid)
    eligible = np.flatnonzero(np.isfinite(full_window))
    number = max(1, math.ceil(hard_fraction*len(eligible))) if len(eligible) else 0
    chosen = eligible[np.lexsort((indices[eligible], -full_window[eligible]))[:number]]
    hard_rows = np.zeros(rows, bool)
    hard_rows[chosen] = True
    definitions = {'all_nodes': dict(family='all'), 'fixed_FULL_hard': dict(family='fixed_hard')}
    flow_names = []
    for lower, upper in zip(FLOW_EDGES, FLOW_EDGES[1:]):
        name = f'flow:[{lower:g},{upper:g})'
        definitions[name] = dict(family='flow', lower=lower, upper=None if np.isinf(upper) else upper)
        flow_names.append(name)
    for name in road_masks:
        definitions['road:'+name] = dict(family='road', road=name)
    group_names = tuple(definitions)
    counts = {name: np.zeros(horizons, np.int64) for name in group_names}
    case_totals = {case: {name: {field: np.zeros(horizons, np.float64) for field in
        ('absolute', 'squared', 'percentage')} for name in group_names} for case in order}
    comparison_fields = ('net', 'positive', 'negative', 'signed', 'absolute', 'absolute_max', 'up', 'down', 'response', 'improved', 'worsened')
    comparison_totals = {case: {name: {field: np.zeros(horizons, np.float64) for field in comparison_fields}
        for name in group_names} for case in order if case != full_case}
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        for h in range(horizons):
            y = target[:, h].astype(np.float64)
            active = valid[:, h]
            masks = {'all_nodes': active, 'fixed_FULL_hard': active & hard_rows[:, None]}
            for name in flow_names:
                group = definitions[name]
                masks[name] = active & (y >= group['lower'])
                if group['upper'] is not None:
                    masks[name] &= y < group['upper']
            masks.update({'road:'+name: active & mask[None, :] for name, mask in road_masks.items()})
            for name, mask in masks.items():
                counts[name][h] = int(mask.sum())
            full = arrays[full_case][:, h].astype(np.float64)
            full_error = np.abs(full-y)
            for case in order:
                value = arrays[case][:, h].astype(np.float64)
                error = np.abs(value-y)
                percentage = np.divide(error, y, out=np.zeros_like(error), where=active)
                for name, mask in masks.items():
                    total = case_totals[case][name]
                    selected = error[mask]
                    total['absolute'][h] = selected.sum()
                    total['squared'][h] = np.square(selected).sum()
                    total['percentage'][h] = percentage[mask].sum()
                if case == full_case:
                    continue
                gain, change = error-full_error, full-value
                for name, mask in masks.items():
                    total = comparison_totals[case][name]
                    selected, changed = gain[mask], change[mask]
                    for key, values in (('net', selected), ('positive', np.maximum(selected, 0)),
                        ('negative', np.maximum(-selected, 0)), ('signed', changed), ('absolute', np.abs(changed))):
                        total[key][h] = values.sum()
                    total['absolute_max'][h] = np.max(np.abs(changed), initial=0.)
                    for key, values in (('up', changed > 0), ('down', changed < 0), ('response', np.abs(changed) > response_tol),
                        ('improved', selected > 0), ('worsened', selected < 0)):
                        total[key][h] = int(values.sum())
    require(np.array_equal(sum(counts[name] for name in flow_names), counts['all_nodes']), 'Flow buckets fail count conservation')
    cases = {}
    for case in order:
        window = _window_scores(arrays[case], target, valid)
        available = window[np.isfinite(window)]
        cases[case] = dict(groups={name: _case_summary(case_totals[case][name], counts[name]) for name in group_names},
            window_mae=dict(complete_windows=len(available), mean=float(available.mean()) if len(available) else None,
                p95=float(np.quantile(available, .95)) if len(available) else None),
            fixed_hard_window_mean=float(window[chosen].mean()) if len(chosen) else None)
    comparisons = {}
    for case, totals in comparison_totals.items():
        groups = {name: _comparison_summary(totals[name], counts[name], counts['all_nodes']) for name in group_names}
        for field in ('net', 'positive', 'negative', 'signed', 'absolute', 'up', 'down', 'response', 'improved', 'worsened'):
            observed, expected = sum(totals[name][field] for name in flow_names), totals['all_nodes'][field]
            scale = totals['all_nodes']['positive']+totals['all_nodes']['negative']+totals['all_nodes']['absolute']+1.
            require(np.all(np.abs(observed-expected) <= 128*np.finfo(np.float64).eps*scale), 'Flow buckets fail paired conservation: '+case+' '+field)
        full_metrics, other_metrics = cases[full_case]['groups']['all_nodes'], cases[case]['groups']['all_nodes']
        comparison = dict(intervention=case, full_case=full_case, groups=groups,
            rmse_gain_pooled=other_metrics['rmse_pooled']-full_metrics['rmse_pooled'],
            rmse_gain_macro=None if other_metrics['rmse_macro'] is None or full_metrics['rmse_macro'] is None else
                other_metrics['rmse_macro']-full_metrics['rmse_macro'],
            mape_gain_pooled=other_metrics['mape_pooled']-full_metrics['mape_pooled'],
            mape_gain_macro=None if other_metrics['mape_macro'] is None or full_metrics['mape_macro'] is None else
                other_metrics['mape_macro']-full_metrics['mape_macro'],
            fixed_hard_window_gain=None if not len(chosen) else
                cases[case]['fixed_hard_window_mean']-cases[full_case]['fixed_hard_window_mean'])
        comparisons[case] = comparison
    ranking = sorted(comparisons, key=lambda case: (-comparisons[case]['groups']['all_nodes']['mae_gain_pooled'], order.index(case)))
    return dict(schema=SCHEMA, status='ORIGINAL_IGSTGNN_ABLATION_ANALYSIS_PASS', case_order=list(order), full_case=full_case,
        cases=cases, comparisons=comparisons, utility_ranking=ranking, group_definitions=definitions,
        validation_rows=rows, horizons=horizons, stations=nodes, valid_points=int(valid.sum()),
        sample_indices=indices.tolist(), station_ids=stations.tolist(), response_tolerance=float(response_tol),
        gain_sign='intervention_absolute_error_minus_FULL_absolute_error_positive_means_FULL_better',
        prediction_response_sign='FULL_prediction_minus_intervention_prediction_positive_means_FULL_higher',
        mape_unit='fraction', positive_valid_target_requirement=True,
        fixed_hard_windows=dict(selector='FULL', fraction=float(hard_fraction), complete_windows=len(eligible), selected_count=len(chosen),
            sample_indices=indices[chosen].tolist(), selection_rule='largest_original_mask_window_horizon_equal_MAE',
            tie_break='ascending_original_sample_index', reused_for_every_case=True),
        contribution_scope='same_checkpoint_descriptive_sensitivity_not_independent_additive_or_causal_contribution',
        validation_unit_scope='windows_and_forecast_points_may_overlap_not_independent_incident_replicates',
        true_flow_scope='posthoc_diagnostic_not_prediction_input_or_gate',
        mask_policy='caller_supplied_original_mask_unchanged')


def _timing_sample_summary(records, expected_samples, expected_passes):
    by_size = {}
    for row in records:
        size = row.get('batch_size')
        require(size is None or (type(size) is int and size > 0), 'Timing batch_size must be a positive integer')
        for field in ('pass_index', 'batch_index'):
            value = row.get(field)
            require(value is None or (type(value) is int and value >= 0), 'Timing '+field+' must be a nonnegative integer')
        by_size.setdefault('unknown' if size is None else str(size), []).append(row)
    batches = {}
    for size, rows in by_size.items():
        wall = [row['wall_seconds'] for row in rows]
        samples = None if size == 'unknown' else int(size)*len(rows)
        batches[size] = dict(batch_size=None if size == 'unknown' else int(size), measured_calls=len(rows),
            measured_samples=samples, wall_seconds_mean=float(np.mean(wall)), wall_seconds_median=float(np.median(wall)),
            wall_seconds_total=float(sum(wall)), ms_per_sample=None if samples is None else 1000*sum(wall)/samples)
    samples_known = bool(records) and all(row.get('batch_size') is not None for row in records)
    measured_samples = sum(row['batch_size'] for row in records) if samples_known else None
    measured_ms = 1000*sum(row['wall_seconds'] for row in records)/measured_samples if measured_samples else None
    passes_known = bool(records) and all(row.get('pass_index') is not None for row in records)
    by_pass = {}
    if passes_known:
        for row in records:
            by_pass.setdefault(row['pass_index'], []).append(row)
    observed = sorted(by_pass)
    required = list(range(expected_passes)) if expected_passes is not None else list(range(len(observed)))
    pass_rows = []
    for index in sorted(set(required) | set(observed)):
        rows = by_pass.get(index, [])
        batch_indices = [row.get('batch_index') for row in rows]
        indices_known = bool(rows) and all(value is not None for value in batch_indices)
        if indices_known:
            require(len(set(batch_indices)) == len(batch_indices), 'Repeated timing batch_index within pass '+str(index))
        indices_valid = sorted(batch_indices) == list(range(len(rows))) if indices_known else None
        samples = sum(row['batch_size'] for row in rows) if rows and all(row.get('batch_size') is not None for row in rows) else None
        count_matches = None if expected_samples is None or samples is None else samples == expected_samples
        complete = bool(rows) and count_matches is True and indices_valid is not False
        status = ('missing_pass' if not rows else 'sample_metadata_unavailable' if samples is None else
            'expected_sample_count_unavailable' if expected_samples is None else
            'sample_count_mismatch' if not count_matches else 'batch_indices_incomplete' if indices_valid is False else 'complete')
        pass_rows.append(dict(pass_index=index, measured_calls=len(rows), measured_samples=samples,
            expected_samples=expected_samples, sample_count_matches_expected=count_matches,
            batch_indices_checked=indices_known, batch_indices_complete=indices_valid, complete=complete, status=status,
            wall_seconds_total=float(sum(row['wall_seconds'] for row in rows)) if rows else None))
    all_complete = passes_known and observed == required and bool(required) and all(row['complete'] for row in pass_rows)
    validation_mean = float(np.mean([row['wall_seconds_total'] for row in pass_rows])) if all_complete else None
    return dict(batch_sizes=batches, validation_passes=pass_rows, measured_samples=measured_samples,
        measured_ms_per_sample=measured_ms, expected_validation_samples=expected_samples, expected_validation_passes=expected_passes,
        observed_validation_passes=len(observed) if passes_known else None,
        complete_validation_passes=sum(row['complete'] for row in pass_rows),
        validation_sample_counts_verified=bool(all_complete),
        validation_seconds_per_pass_mean=validation_mean,
        ms_per_sample=None if validation_mean is None else 1000*validation_mean/expected_samples)


def summarize_timings(timings, case_order=CASE_NAMES, *, expected_validation_samples=None, expected_passes=None):
    """Keep raw synchronized full-forward records and summarize measured cases.

    This module does not measure or certify synchronization. The runner supplies
    wall_seconds, optional cuda_seconds, peak_allocated_bytes and
    incremental_peak_bytes. Warmup records are preserved but excluded.
    """
    timings = {} if timings is None else timings
    require(isinstance(timings, dict) and set(timings) <= set(case_order), 'Unknown timing case')
    require(expected_validation_samples is None or (type(expected_validation_samples) is int and expected_validation_samples > 0),
        'Expected validation samples must be a positive integer')
    require(expected_passes is None or (type(expected_passes) is int and expected_passes > 0), 'Expected timing passes must be a positive integer')
    result = {}
    for case in case_order:
        raw = timings.get(case, [])
        if isinstance(raw, dict):
            raw = raw.get('records', [raw] if raw else [])
        require(isinstance(raw, (list, tuple)) and all(isinstance(row, dict) for row in raw), 'Timing records must be dictionaries')
        records, included = _plain(raw), []
        for row in records:
            warmup = row.get('warmup', False)
            require(type(warmup) is bool and type(row.get('included', True)) is bool, 'Timing inclusion flags must be Boolean')
            wall = next((row[key] for key in ('wall_seconds', 'elapsed_seconds', 'seconds', 'inference_seconds') if key in row), None)
            require(wall is not None and isinstance(wall, (int, float)) and not isinstance(wall, bool)
                and math.isfinite(wall) and wall >= 0, 'Each timing record needs a finite nonnegative full-forward wall time')
            normalized = dict(row, wall_seconds=float(wall))
            for field in ('cuda_seconds', 'peak_allocated_bytes', 'cuda_peak_allocated_bytes', 'incremental_peak_bytes'):
                value = row.get(field)
                require(value is None or (isinstance(value, (int, float)) and not isinstance(value, bool)
                    and math.isfinite(value) and value >= 0), 'Invalid timing or memory statistic: '+field)
            if not warmup and row.get('included', True):
                included.append(normalized)
        wall = [row['wall_seconds'] for row in included]
        gpu = [row['cuda_seconds'] for row in included if row.get('cuda_seconds') is not None]
        peak = [row.get('peak_allocated_bytes', row.get('cuda_peak_allocated_bytes')) for row in included
            if row.get('peak_allocated_bytes', row.get('cuda_peak_allocated_bytes')) is not None]
        incremental = [row['incremental_peak_bytes'] for row in included if row.get('incremental_peak_bytes') is not None]
        result[case] = dict(status='available' if wall else 'unmeasured', raw_records=records,
            recorded_calls=len(records), measured_calls=len(wall), excluded_calls=len(records)-len(wall),
            wall_seconds_total=float(sum(wall)) if wall else None, wall_seconds_mean=float(np.mean(wall)) if wall else None,
            wall_seconds_median=float(np.median(wall)) if wall else None,
            wall_seconds_min=float(min(wall)) if wall else None, wall_seconds_max=float(max(wall)) if wall else None,
            cuda_seconds_measured_calls=len(gpu), cuda_seconds_total=float(sum(gpu)) if gpu else None,
            cuda_seconds_mean=float(np.mean(gpu)) if gpu else None,
            peak_allocated_bytes_max=max(peak) if peak else None, incremental_peak_bytes_max=max(incremental) if incremental else None,
            **_timing_sample_summary(included, expected_validation_samples, expected_passes))
    return dict(cases=result, requested_measurement_scope='complete_model_inference_CUDA_synchronized_wall_and_peak',
        excluded_from_measurement='data_loading_H2D_CPU_statistics_hashing',
        ms_per_sample_scope='amortized_wall_time_per_batch_sample_not_single_sample_latency',
        certification_scope='raw_runner_measurements_not_independently_certified_by_statistics_module',
        speed_interpretation='zeroed_banks_and_forecast_readouts_still_execute_not_retrained_deleted_model_speed',
        training_time_measured=False)


def _write_csv(path, rows):
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    columns = list(rows[0])
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_report(output, analysis, metadata, timings=None):
    """Write new report files beside the runner's immutable prediction archive."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    require(analysis.get('schema') == SCHEMA and set(analysis['case_order']) == set(CASE_NAMES), 'Wrong checkpoint screen analysis')
    require(isinstance(metadata, dict), 'Report metadata must be a dictionary supplied by the runner')
    timing_summary = summarize_timings(timings, analysis['case_order'],
        expected_validation_samples=analysis['validation_rows'], expected_passes=metadata.get('timing_passes'))
    filenames = ('analysis.json', 'metadata.json', 'timings.json', 'cases.csv', 'comparisons.csv', 'horizons.csv',
        'flow.csv', 'roads.csv', 'fixed_hard.csv', 'timings.csv', 'report.md')
    require(not any((output/name).exists() for name in filenames), 'Report outputs already exist; refusing overwrite')
    json_values = {'analysis.json': analysis, 'metadata.json': metadata, 'timings.json': timing_summary}
    for name, value in json_values.items():
        (output/name).write_text(json.dumps(_plain(value), ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    case_rows, comparison_rows, horizon_rows, flow_rows, road_rows, hard_rows = [], [], [], [], [], []
    for case in analysis['case_order']:
        row = analysis['cases'][case]
        all_nodes = row['groups']['all_nodes']
        case_rows.append(dict(case=case, **{name: all_nodes[name] for name in
            ('valid_count', 'mae_pooled', 'mae_macro', 'rmse_pooled', 'rmse_macro', 'mape_pooled', 'mape_macro')},
            window_p95=row['window_mae']['p95']))
        for h in all_nodes['horizons']:
            horizon_rows.append(dict(kind='case', case=case, step=h['step'], valid_count=h['valid_count'],
                mae=h['mae'], rmse=h['rmse'], mape=h['mape'], mae_gain=None, gain_positive=None, gain_negative=None))
    for case in analysis['utility_ranking']:
        comparison = analysis['comparisons'][case]
        for name, group in comparison['groups'].items():
            row = dict(intervention=case, full_case=analysis['full_case'], group=name,
                **{key: group[key] for key in ('valid_count', 'mae_gain_pooled', 'mae_gain_macro', 'gain_positive', 'gain_negative',
                    'cancellation_fraction', 'global_mae_contribution', 'prediction_signed_mean', 'prediction_absolute_mean',
                    'prediction_absolute_max', 'prediction_up_fraction', 'prediction_down_fraction', 'prediction_response_fraction')})
            family = analysis['group_definitions'][name]['family']
            if family == 'all':
                comparison_rows.append(dict(row, rmse_gain_pooled=comparison['rmse_gain_pooled'],
                    rmse_gain_macro=comparison['rmse_gain_macro'], mape_gain_pooled=comparison['mape_gain_pooled'],
                    mape_gain_macro=comparison['mape_gain_macro'], fixed_hard_window_gain=comparison['fixed_hard_window_gain']))
            elif family == 'flow':
                flow_rows.append(row)
            elif family == 'road':
                road_rows.append(row)
            elif family == 'fixed_hard':
                hard_rows.append(dict(row, window_equal_gain=comparison['fixed_hard_window_gain'],
                    FULL_window_mean=analysis['cases'][analysis['full_case']]['fixed_hard_window_mean'],
                    intervention_window_mean=analysis['cases'][case]['fixed_hard_window_mean']))
        for h in comparison['groups']['all_nodes']['horizons']:
            horizon_rows.append(dict(kind='comparison', case=case, step=h['step'], valid_count=h['valid_count'],
                mae=None, rmse=None, mape=None, mae_gain=h['mae_gain'], gain_positive=h['gain_positive'], gain_negative=h['gain_negative']))
    for name, rows in (('cases.csv', case_rows), ('comparisons.csv', comparison_rows), ('horizons.csv', horizon_rows),
        ('flow.csv', flow_rows), ('roads.csv', road_rows), ('fixed_hard.csv', hard_rows)):
        _write_csv(output/name, rows)
    timing_rows = [dict(case=case, **{name: value for name, value in row.items() if not isinstance(value, (dict, list, tuple))})
        for case, row in timing_summary['cases'].items()]
    _write_csv(output/'timings.csv', timing_rows)
    fmt = lambda value: '不可用' if value is None else f'{value:.9f}'
    full_case = analysis['full_case']
    engineering = metadata.get('engineering_only', metadata.get('engineering_check', '未提供'))
    primary = metadata.get('primary_baseline', 'original_IGSTGNN')
    lines = ['# 原始 IGSTGNN：十二状态固定检查点筛查', '',
        '工程检查：'+str(engineering)+'；窗口：'+str(analysis['validation_rows'])+'；主 baseline 身份：'+str(primary)+'。',
        '注册参数量：'+str(metadata.get('registered_parameters', '未提供'))+'；不参与当前计算的局部BN参数：'+
            str(metadata.get('inactive_local_bn_parameters', '未提供'))+'。注册量不等于活跃计算量。',
        '源检查点、训练日志、运行时和回放证据由 runner 提供，完整内容见 metadata.json；本统计模块不独立认证这些身份。', '',
        '**效用 = 干预状态误差 − FULL误差。正数表示完整路径更好，负数表示该干预在本检查点更好。**',
        '所有干预共享固定检查点与原验证掩码。本筛查不训练模型；同检查点屏蔽不等于删除模块后重训。',
        '多个状态改变的路径可能重叠，不能把效用相加或解释为独立贡献百分比。验证窗口与预测点可能重叠，不视为独立事故重复。', '',
        '| 状态 | MAE pooled | MAE时距macro | RMSE pooled | RMSE时距macro | MAPE (%) | 窗口P95 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for row in case_rows:
        lines.append('| '+' | '.join([row['case']]+[fmt(row[key]) for key in ('mae_pooled', 'mae_macro', 'rmse_pooled', 'rmse_macro')]
            +[fmt(None if row['mape_pooled'] is None else 100*row['mape_pooled']), fmt(row['window_p95'])])+' |')
    lines += ['', '| 干预 | FULL的MAE效用pooled | MAE效用macro | RMSE效用pooled | MAPE效用百分点 | 固定困难窗口效用 |',
        '|---|---:|---:|---:|---:|---:|']
    for row in comparison_rows:
        lines.append('| '+' | '.join([row['intervention'], fmt(row['mae_gain_pooled']), fmt(row['mae_gain_macro']),
            fmt(row['rmse_gain_pooled']), fmt(None if row['mape_gain_pooled'] is None else 100*row['mape_gain_pooled']),
            fmt(row['fixed_hard_window_gain'])])+' |')
    lines += ['', '## 正负抵消与预测响应', '',
        '正/负效用来自各比较自己的逐点MAE差。抵消比例=1−|净效用和|/(正效用和+负效用和)，零响应时为0；空组不可用。',
        '预测变化统一为FULL−干预；正数表示完整路径上调预测。响应阈值：'+str(analysis['response_tolerance'])+'（作者原尺度，未新增物理单位认证）。', '',
        '| 干预 | 正效用均值 | 负效用均值 | 抵消比例 | 预测变化均值 | 预测绝对变化均值 | 上调占比 | 下调占比 | 响应占比 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in comparison_rows:
        lines.append('| '+' | '.join([row['intervention']]+[fmt(row[key]) for key in
            ('gain_positive', 'gain_negative', 'cancellation_fraction', 'prediction_signed_mean', 'prediction_absolute_mean',
             'prediction_up_fraction', 'prediction_down_fraction', 'prediction_response_fraction')])+' |')
    lines += ['', '## 流量与道路', '',
        '流量桶使用真实目标，只用于事后诊断，不能作为预测输入或门控。组内均值不能直接相加；全网贡献统一使用全网有效点数分母。',
        '逐状态/比较的时距、流量、道路与固定困难集数值分别见 horizons.csv、flow.csv、roads.csv、fixed_hard.csv；空组为不可用。',
        '道路分组可能重叠；仅流量桶校验互斥完备与贡献守恒。RMSE不能按MAE逐点线性关系相加。', '',
        '## 固定困难窗口', '',
        '由FULL各窗口的原掩码节点MAE先按时距计算，再对全部时距等权；仅使用每个时距都有效的完整窗口。',
        '选取最差'+str(100*analysis['fixed_hard_windows']['fraction'])+'%，向上取整，原样本ID升序打破同分；所有状态复用同一集合。',
        '完整窗口数：'+str(analysis['fixed_hard_windows']['complete_windows'])+'；固定集合数：'+str(analysis['fixed_hard_windows']['selected_count'])+'。',
        '固定ID：'+str(analysis['fixed_hard_windows']['sample_indices'])+'。', '',
        '## 完整推理耗时', '',
        '计时范围应为完整模型一次推理的CUDA同步wall时间与峰值显存，不包含数据加载、H2D、CPU统计或哈希；同步与计时边界由runner记录。',
        'runner声明的实际计时范围：'+str(metadata.get('timing_scope', '未提供'))+'。',
        '这里只汇总提供的原始记录，不能独立认证计时边界；warmup记录保留但不参与汇总。不同batch大小或顺序的记录不据此宣称公平速度优劣。',
        '**置零图bank和forecast读出仍照常计算，不代表真正删除模块后的速度；本报告没有训练耗时结论。**', '',
        '| 状态 | 测量调用 | wall均值秒 | wall中位数秒 | wall总秒 | CUDA均值秒 | 峰值allocated bytes | 增量峰值bytes |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for case, row in timing_summary['cases'].items():
        lines.append('| '+' | '.join([case, str(row['measured_calls'])]+[fmt(row[key]) for key in
            ('wall_seconds_mean', 'wall_seconds_median', 'wall_seconds_total', 'cuda_seconds_mean',
             'peak_allocated_bytes_max', 'incremental_peak_bytes_max')])+' |')
    lines += ['', '### 按批大小与完整验证pass核对', '',
        '批大小分组的均值仅比较同样本数调用，避免把48行批与尾部21行批混为同一耗时。',
        '完整验证均值要求每个期望pass都存在、样本数与验证集一致，且已提供的batch编号连续且无重复。缺pass或样本信息时为不可用；原始记录仍保留。',
        'measured_ms_per_sample仅描述已记录调用；下表完整验证ms/样本仅在全部期望pass通过样本数核对后给出。', '',
        'ms/样本是批处理wall时间按样本分摊的吞吐口径，不是单样本推理延迟。', '',
        '| 状态 | 批大小 | 调用数 | wall均值秒 | wall总秒 | 已测ms/样本 |', '|---|---:|---:|---:|---:|---:|']
    for case, row in timing_summary['cases'].items():
        for batch in row['batch_sizes'].values():
            lines.append('| '+' | '.join([case, '未知' if batch['batch_size'] is None else str(batch['batch_size']),
                str(batch['measured_calls']), fmt(batch['wall_seconds_mean']), fmt(batch['wall_seconds_total']), fmt(batch['ms_per_sample'])])+' |')
    lines += ['', '| 状态 | 已测样本总数 | 完整pass数 | 样本数核对通过 | 完整验证每pass均值秒 | 完整验证ms/样本 |',
        '|---|---:|---:|---|---:|---:|']
    for case, row in timing_summary['cases'].items():
        lines.append('| '+' | '.join([case, '不可用' if row['measured_samples'] is None else str(row['measured_samples']),
            str(row['complete_validation_passes']), str(row['validation_sample_counts_verified']),
            fmt(row['validation_seconds_per_pass_mean']), fmt(row['ms_per_sample'])])+' |')
    lines += ['', 'JSON/CSV中MAPE为比例，本报告换算为百分数；收益表换算为百分点。',
        'MAE/RMSE pooled按有效点等权；时距macro对全部时距等权，缺失时距则不可用。有效真值须严格为正，保留作者abs差/target的MAPE定义。',
        '本报告不提供独立样本显著性、真实事故因果、物理容量/时延或独立模块贡献认证。']
    (output/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return dict(files=list(filenames), timings=timing_summary)
