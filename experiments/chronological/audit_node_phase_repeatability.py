"""Audit out-of-time repeatability of node and node-phase signed residuals."""

import argparse
from collections import defaultdict
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.chronological.audit_expert_benefit import verify_package
from experiments.chronological.audit_matched_controls import (
    read_csv, sha256, write_csv,
)
from experiments.chronological.audit_signed_residual_probe import (
    PHASES, correction_metrics, load_residuals, target_for, uncertainty,
)


LEVELS = ('node', 'node_phase')
FAMILIES = ('incident_median', 'zero_anchored_mean')
COHORTS = ('incident_full', 'incident', 'primary_control', 'secondary_control')


def load_protocol(path):
    protocol = json.loads(Path(path).read_text(encoding='utf-8'))
    if (protocol.get('protocol_id') != 'contra_v8_node_phase_repeatability_v7c' or
            protocol.get('scope') !=
            'train_only_rolling_origin_node_phase_residual_repeatability' or
            protocol.get('main_training_ready') is not False):
        raise ValueError('v7c protocol identity or scope changed')
    if [protocol.get(key) for key in ('source_year', 'source_version',
                                      'expected_sensor_count')] != [2023, 8, 496]:
        raise ValueError('v7c is frozen to the 2023 source-v8 Contra496 package')
    if (protocol.get('expected_common_train_samples') != 3106 or
            protocol.get('expected_full_positive_train_samples') != 3604):
        raise ValueError('v7c training cohorts changed')
    v7a = protocol.get('v7a_input', {})
    if v7a != {
            'protocol_id': 'contra_v8_signed_residual_materialize_v7a',
            'protocol_sha256':
            'c3d547679b9c6c97ad29720578ddc46a2460d7e9ba336114c59aba429ac03541',
            'status': 'SIGNED_RESIDUAL_MATERIALIZATION_COMPLETE'}:
        raise ValueError('v7c v7a input changed')
    context = protocol.get('v7b_context', {})
    if (context.get('protocol_id') != 'contra_v8_signed_residual_probe_v7b' or
            context.get('protocol_sha256') !=
            '3413cf4a8360f9376e93ef72a51bb32b0be79b52cce74555ce5ce85a5d17d1de' or
            context.get('observed_recommendation') !=
            'SIGNED_RESIDUAL_PROBE_NOT_VALIDATED' or
            context.get('role') != 'motivation_only_not_an_input'):
        raise ValueError('v7c v7b context changed')
    rolling = protocol.get('rolling_origin', {})
    if rolling != {
            'time_source': 'positive_incident_t0_iso_week',
            'expected_unique_train_iso_weeks': 35,
            'initial_fit_week_count': 18,
            'audit_fold_count': 3,
            'audit_week_block_sizes': [6, 6, 5],
            'fit_rule': 'all_train_iso_weeks_strictly_before_each_audit_block',
            'audit_rule': 'each_of_the_final_17_train_iso_weeks_exactly_once',
            'expected_full_positive_audit_counts': [602, 731, 514],
            'expected_common_audit_counts': [536, 618, 338]}:
        raise ValueError('v7c rolling-origin design changed')
    support = protocol.get('support', {})
    if (support.get('candidate_source') !=
            'paired_positive_report_location_distances_nonzero' or
            support.get('active_horizons_zero_based_half_open') != [0, 6] or
            support.get('phase_slices_zero_based_half_open') != [[0, 3], [3, 6]] or
            support.get('levels') != list(LEVELS) or
            support.get('protected_horizons_zero_based_half_open') != [6, 12] or
            support.get('protected_noncandidate_nodes') is not True or
            support.get('correction_exactly_zero_outside_support') is not True):
        raise ValueError('v7c correction support changed')
    families = protocol.get('lookup_families', {})
    expected_families = {
        'incident_median': {
            'role': 'diagnostic_repeatability_lookup',
            'incident_target': 'median_signed_residual_on_level_support',
            'control_target': 'not_used',
            'estimator': 'weighted_median_per_node_or_node_phase',
            'row_weighting': 'equal_incident_event_weight',
        },
        'zero_anchored_mean': {
            'role': 'primary_routine_protected_lookup',
            'incident_target': 'median_signed_residual_on_level_support',
            'control_target': 'exact_zero_without_reading_control_future_Y',
            'estimator': 'weighted_mean_per_node_or_node_phase',
            'row_weighting': 'equal_event_and_equal_cohort_weight',
        },
    }
    if (families != expected_families or
            protocol.get('primary_family') != 'zero_anchored_mean' or
            protocol.get('primary_level') != 'node_phase' or
            protocol.get('prediction_clip') !=
            'symmetric_fit_incident_target_abs_q99_per_fold_and_level'):
        raise ValueError('v7c lookup design changed')
    if protocol.get('uncertainty') != {
            'method': 'audit_iso_week_cluster_bootstrap', 'draws': 2000,
            'confidence_level': 0.95, 'seed': 2025}:
        raise ValueError('v7c uncertainty design changed')
    expected_gate = {
        'require_positive_full_positive_global_point_improvement': True,
        'maximum_global_harm_fraction_of_A_mae': 0.001,
        'require_global_noninferiority_ci_lower_within_margin': True,
        'require_full_positive_candidate_H1_H6_improvement_ci_lower_above_zero': True,
        'require_common_incident_candidate_H1_H6_point_improvement': True,
        'maximum_each_control_H1_H6_harm_fraction_of_A_mae': 0.005,
        'require_nonzero_correction_fraction': 0.05,
        'require_exact_A_equality_outside_support': True,
    }
    if protocol.get('development_gate') != expected_gate:
        raise ValueError('v7c prospective development gate changed')
    boundary = protocol.get('information_boundary', {})
    required = (
        'train_residual_arrays_only', 'validation_residual_arrays_prohibited',
        'test_split_prohibited', 'audit_weeks_prohibited_from_lookup_fit',
        'future_weeks_prohibited_from_each_fold_fit',
        'control_future_Y_prohibited_from_lookup_fit',
        'folds_levels_families_clip_and_gate_may_not_change_after_result',
        'neural_expert_training_prohibited',
    )
    if not all(boundary.get(key) is True for key in required):
        raise ValueError('v7c information boundary changed')
    return protocol


def verify_inputs(data_dir, primary_dir, secondary_dir, materialized_dir, protocol):
    data_dir, materialized_dir = Path(data_dir), Path(materialized_dir)
    summary_path = materialized_dir / 'summary.json'
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    expected = protocol['v7a_input']
    if (summary.get('status') != expected['status'] or
            summary.get('protocol_id') != expected['protocol_id'] or
            summary.get('protocol_sha256') != expected['protocol_sha256'] or
            summary.get('engineering_check') is not False or
            summary.get('test_split_read') is not False or
            summary.get('model_training_performed') is not False):
        raise ValueError('v7a materialization identity or boundary changed')
    expected_outputs = {
        'train_incident_full_signed_residuals.npz':
            protocol['expected_full_positive_train_samples'],
        'train_incident_signed_residuals.npz':
            protocol['expected_common_train_samples'],
        'train_primary_control_signed_residuals.npz':
            protocol['expected_common_train_samples'],
        'train_secondary_control_signed_residuals.npz':
            protocol['expected_common_train_samples'],
    }
    for filename, samples in expected_outputs.items():
        metadata = summary.get('outputs', {}).get(filename, {})
        if (metadata.get('samples') != samples or
                sha256(materialized_dir / filename) != metadata.get('sha256')):
            raise ValueError(f'v7a training output differs: {filename}')
    package = verify_package(data_dir)
    if package != summary['inputs']['positive_package']:
        raise ValueError('Positive package differs from v7a materialization')
    train_names = {
        'primary_controls': (
            primary_dir,
            ('summary.json', 'train_control_flow.npy', 'train_affected_mask.npy',
             'train_control_manifest.csv')),
        'secondary_controls': (
            secondary_dir,
            ('summary.json', 'train_second_control_flow.npy',
             'train_second_affected_mask.npy',
             'train_second_control_manifest.csv')),
    }
    verified = {}
    for key, (directory, names) in train_names.items():
        expected_hashes = summary['inputs'][key]
        actual = {name: sha256(Path(directory) / name) for name in names}
        if any(actual[name] != expected_hashes.get(name) for name in names):
            raise ValueError(f'{key} training inputs differ from v7a')
        verified[key] = actual
    return summary, package, verified


def load_control_identity(materialized_dir, cohort, expected_samples, expected_nodes):
    path = Path(materialized_dir) / f'train_{cohort}_signed_residuals.npz'
    with np.load(path, allow_pickle=False) as stored:
        required = {
            'signed_residual', 'baseline_prediction', 'valid', 'candidate_mask',
            'positive_sample_index', 'baseline_all_absolute_sum',
            'baseline_all_valid_count',
        }
        if set(stored.files) != required:
            raise ValueError(f'Unexpected v7a array schema: {path.name}')
        candidate = stored['candidate_mask'].copy()
        sample = stored['positive_sample_index'].copy()
    if (candidate.shape != (expected_samples, expected_nodes) or
            sample.shape != (expected_samples,) or
            candidate.dtype != np.bool_ or sample.dtype != np.int64):
        raise ValueError(f'Invalid control identity arrays: {path.name}')
    return {'candidate_mask': candidate, 'positive_sample_index': sample}


def iso_week(value):
    calendar = datetime.fromisoformat(value).isocalendar()
    return f'{calendar.year:04d}-W{calendar.week:02d}'


def manifest_weeks(data_dir, secondary_dir, full_ids, common_ids):
    full_rows = read_csv(Path(data_dir) / 'train_manifest.csv')
    common_rows = read_csv(Path(secondary_dir) / 'train_second_control_manifest.csv')
    observed_full = np.asarray([int(row['sample_index']) for row in full_rows])
    observed_common = np.asarray([int(row['positive_sample_index']) for row in common_rows])
    if (not np.array_equal(observed_full, full_ids) or
            not np.array_equal(observed_common, common_ids)):
        raise ValueError('Training manifests and v7a arrays have different identities')
    full_weeks = np.asarray([iso_week(row['t0']) for row in full_rows])
    common_weeks = np.asarray([iso_week(row['positive_t0']) for row in common_rows])
    return full_weeks, common_weeks


def rolling_origin_folds(full_weeks, common_weeks, protocol):
    specification = protocol['rolling_origin']
    unique = sorted(set(full_weeks))
    if len(unique) != specification['expected_unique_train_iso_weeks']:
        raise ValueError('Unique training ISO-week count changed')
    initial = specification['initial_fit_week_count']
    sizes = specification['audit_week_block_sizes']
    if initial + sum(sizes) != len(unique):
        raise ValueError('Rolling-origin week partition is incomplete')
    folds, position = [], initial
    all_audit_weeks = []
    for number, size in enumerate(sizes, start=1):
        audit_weeks = unique[position:position + size]
        fit_weeks = unique[:position]
        full_audit = np.flatnonzero(np.isin(full_weeks, audit_weeks))
        common_audit = np.flatnonzero(np.isin(common_weeks, audit_weeks))
        common_fit = np.flatnonzero(np.isin(common_weeks, fit_weeks))
        if (len(full_audit) !=
                specification['expected_full_positive_audit_counts'][number - 1] or
                len(common_audit) !=
                specification['expected_common_audit_counts'][number - 1] or
                not len(common_fit) or
                np.intersect1d(common_fit, common_audit).size):
            raise ValueError('Rolling-origin fold counts or separation changed')
        folds.append({
            'fold': number, 'fit_weeks': fit_weeks,
            'audit_weeks': audit_weeks, 'common_fit_indices': common_fit,
            'full_audit_indices': full_audit,
            'common_audit_indices': common_audit,
        })
        all_audit_weeks.extend(audit_weeks)
        position += size
    if sorted(all_audit_weeks) != unique[initial:]:
        raise ValueError('Final train weeks are not covered exactly once')
    return folds


def level_rows(residual, valid, candidate, level, use_residual):
    internal = 'event_node' if level == 'node' else 'event_node_phase'
    nodes = np.flatnonzero(candidate)
    return target_for(
        residual if use_residual else None, valid if use_residual else None,
        nodes, internal, use_residual)


def weighted_median(values, weights):
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if (values.ndim != 1 or weights.shape != values.shape or not len(values) or
            not np.isfinite(values).all() or not np.isfinite(weights).all() or
            np.any(weights <= 0)):
        raise ValueError('Invalid weighted-median input')
    order = np.argsort(values, kind='stable')
    cumulative = np.cumsum(weights[order])
    index = int(np.searchsorted(cumulative, weights.sum() / 2, side='left'))
    return float(values[order[index]])


def fit_lookup(incident, controls, indices, level, family, nodes):
    shape = (nodes,) if level == 'node' else (nodes, len(PHASES))
    values_by_key = defaultdict(lambda: [[], []])
    numerator = np.zeros(shape, dtype=np.float64)
    denominator = np.zeros(shape, dtype=np.float64)
    incident_targets = []
    for position in indices:
        rows = level_rows(
            incident['signed_residual'][position], incident['valid'][position],
            incident['candidate_mask'][position], level, True)
        if not rows:
            raise ValueError('Fit incident has no valid node-level target')
        weight = 1.0 / len(rows)
        for node, phase, target in rows:
            key = node if phase is None else (node, phase)
            incident_targets.append(target)
            if family == 'incident_median':
                values_by_key[key][0].append(target)
                values_by_key[key][1].append(weight)
            else:
                numerator[key] += weight * target
                denominator[key] += weight
    clip = float(np.quantile(np.abs(incident_targets), .99))
    if not np.isfinite(clip) or clip <= 0:
        raise ValueError('Rolling-origin train-only clip is invalid')
    if family == 'zero_anchored_mean':
        for control in controls:
            if not np.array_equal(
                    control['positive_sample_index'],
                    incident['positive_sample_index']):
                raise ValueError('Control and incident fit identities differ')
            for position in indices:
                rows = level_rows(
                    None, None, control['candidate_mask'][position], level, False)
                if not rows:
                    raise ValueError('Fit control has no candidate lookup rows')
                weight = 1.0 / len(rows)
                for node, phase, _ in rows:
                    key = node if phase is None else (node, phase)
                    denominator[key] += weight
    lookup = np.zeros(shape, dtype=np.float64)
    known = np.zeros(shape, dtype=bool)
    if family == 'incident_median':
        for key, (current_values, current_weights) in values_by_key.items():
            lookup[key] = weighted_median(current_values, current_weights)
            known[key] = True
    elif family == 'zero_anchored_mean':
        known = denominator > 0
        lookup[known] = numerator[known] / denominator[known]
    else:
        raise ValueError('Unknown lookup family')
    lookup = np.clip(lookup, -clip, clip)
    return lookup, known, {
        'fit_common_incident_events': int(len(indices)),
        'fit_control_cohorts': 0 if family == 'incident_median' else len(controls),
        'known_keys': int(known.sum()), 'total_keys': int(known.size),
        'known_key_fraction': float(known.mean()), 'clip_abs': clip,
        'lookup_abs_mean_known': float(np.abs(lookup[known]).mean()),
        'lookup_abs_max': float(np.abs(lookup[known]).max()),
        'incident_target_rows': int(len(incident_targets)),
    }


def correction_from_lookup(arrays, lookup, known, level):
    correction = np.zeros_like(arrays['signed_residual'], dtype=np.float64)
    known_active, active_total = 0, 0
    for position, candidate in enumerate(arrays['candidate_mask']):
        nodes = np.flatnonzero(candidate)
        if level == 'node':
            event = correction[position]
            event[:, nodes, 0] = lookup[nodes]
            active = arrays['valid'][position][:, nodes, 0]
            active_total += int(active.sum())
            known_active += int((active & known[nodes][None, :]).sum())
        elif level == 'node_phase':
            event = correction[position]
            for phase, (first, last) in enumerate(PHASES):
                event[first:last, nodes, 0] = lookup[nodes, phase][None, :]
                active = arrays['valid'][position][first:last, nodes, 0]
                active_total += int(active.sum())
                known_active += int((active & known[nodes, phase][None, :]).sum())
        else:
            raise ValueError('Unknown lookup level')
    return correction, known_active, active_total


def subset_arrays(arrays, indices):
    return {key: value[indices] for key, value in arrays.items()}


def concatenate_arrays(parts):
    return {key: np.concatenate([part[key] for part in parts], axis=0)
            for key in parts[0]}


def metric_arrays(materialized_dir, cohort, samples, nodes):
    arrays = load_residuals(materialized_dir, 'train', cohort, samples, nodes)
    arrays.pop('baseline_prediction')
    return arrays


def write_event_metrics(path, family, level, cohort, metrics, sample_ids, weeks,
                        fold_numbers):
    event = metrics['_per_event']
    rows = []
    for position, sample in enumerate(sample_ids):
        rows.append({
            'family': family, 'level': level, 'cohort': cohort,
            'fold': int(fold_numbers[position]),
            'positive_sample_index': int(sample), 'positive_iso_week': weeks[position],
            'all_improvement_vs_A': float(event['all_improvement_vs_A'][position]),
            'candidate_h1_h6_improvement_vs_A': float(
                event['candidate_h1_h6_improvement_vs_A'][position]),
        })
    write_csv(path, rows, [
        'family', 'level', 'cohort', 'fold', 'positive_sample_index',
        'positive_iso_week', 'all_improvement_vs_A',
        'candidate_h1_h6_improvement_vs_A',
    ])


def decision(results, protocol):
    gate = protocol['development_gate']
    primary = results[protocol['primary_family']][protocol['primary_level']]
    full, incident = primary['incident_full'], primary['incident']
    margin = gate['maximum_global_harm_fraction_of_A_mae'] * full['all']['mae_A']
    checks = {
        'positive_full_positive_global_point_improvement':
            full['all']['improvement_vs_A'] > 0,
        'full_positive_global_noninferiority':
            full['uncertainty']['all_improvement_vs_A']['ci_low'] >= -margin,
        'full_positive_candidate_h1_h6_improvement':
            full['uncertainty']['candidate_h1_h6_improvement_vs_A']['ci_low'] > 0,
        'common_incident_candidate_h1_h6_point_improvement':
            incident['candidate_h1_h6']['improvement_vs_A'] > 0,
        'nonzero_correction_fraction':
            full['correction']['nonzero_fraction'] >=
            gate['require_nonzero_correction_fraction'],
        'protected_h7_h12_exact_A': full['protected_h7_h12_exact_A'],
        'protected_noncandidate_exact_A': full['protected_noncandidate_exact_A'],
    }
    control_margin = gate['maximum_each_control_H1_H6_harm_fraction_of_A_mae']
    for cohort in ('primary_control', 'secondary_control'):
        current = primary[cohort]
        checks[f'{cohort}_routine_harm_bound'] = (
            current['uncertainty']['candidate_h1_h6_harm_vs_A']['ci_high'] <=
            control_margin * current['candidate_h1_h6']['mae_A'])
    passed = all(checks.values())
    diagnostic = results['incident_median']['node_phase']['incident_full']
    diagnostic_repeats = (
        diagnostic['all']['improvement_vs_A'] > 0 and
        diagnostic['uncertainty']['candidate_h1_h6_improvement_vs_A']['ci_low'] > 0)
    if passed:
        recommendation = 'NODE_PHASE_EMBEDDED_EXPERT_DEVELOPMENT_ALLOWED'
    elif diagnostic_repeats:
        recommendation = 'NODE_PHASE_REPEATS_BUT_ZERO_ANCHOR_NOT_VALIDATED'
    else:
        recommendation = 'STOP_NODE_PHASE_REPEATABILITY_NOT_ESTABLISHED'
    return {
        'checks': checks, 'repeatability_gate_passed': passed,
        'incident_only_node_phase_repeatability_detected': diagnostic_repeats,
        'maximum_global_harm_raw_mae': margin,
        'recommendation': recommendation,
    }


def audit(data_dir, primary_dir, secondary_dir, materialized_dir, protocol_path,
          output):
    output = Path(output)
    if output.exists():
        raise FileExistsError('v7c output exists; preserve it and use a new directory')
    protocol = load_protocol(protocol_path)
    v7a_summary, package_hashes, input_hashes = verify_inputs(
        data_dir, primary_dir, secondary_dir, materialized_dir, protocol)
    nodes = protocol['expected_sensor_count']
    common_count = protocol['expected_common_train_samples']
    full_count = protocol['expected_full_positive_train_samples']
    full = metric_arrays(materialized_dir, 'incident_full', full_count, nodes)
    incident = metric_arrays(materialized_dir, 'incident', common_count, nodes)
    control_identities = [
        load_control_identity(materialized_dir, cohort, common_count, nodes)
        for cohort in ('primary_control', 'secondary_control')
    ]
    if any(not np.array_equal(identity['candidate_mask'], incident['candidate_mask'])
           for identity in control_identities):
        raise ValueError('Control and incident candidate masks differ')
    full_weeks, common_weeks = manifest_weeks(
        data_dir, secondary_dir, full['positive_sample_index'],
        incident['positive_sample_index'])
    folds = rolling_origin_folds(full_weeks, common_weeks, protocol)

    models, model_arrays = {}, {}
    fold_summaries = []
    for fold in folds:
        current = {
            'fold': fold['fold'], 'fit_first_week': fold['fit_weeks'][0],
            'fit_last_week': fold['fit_weeks'][-1],
            'audit_first_week': fold['audit_weeks'][0],
            'audit_last_week': fold['audit_weeks'][-1],
            'fit_common_events': int(len(fold['common_fit_indices'])),
            'audit_full_positive_events': int(len(fold['full_audit_indices'])),
            'audit_common_events': int(len(fold['common_audit_indices'])),
            'models': {},
        }
        for level in LEVELS:
            for family in FAMILIES:
                lookup, known, fit_summary = fit_lookup(
                    incident, control_identities, fold['common_fit_indices'],
                    level, family, nodes)
                key = (fold['fold'], family, level)
                models[key] = (lookup, known)
                prefix = f'fold_{fold["fold"]}_{family}_{level}'
                model_arrays[f'{prefix}_lookup'] = lookup
                model_arrays[f'{prefix}_known'] = known
                current['models'][f'{family}_{level}'] = fit_summary
        fold_summaries.append(current)

    controls = {
        cohort: metric_arrays(materialized_dir, cohort, common_count, nodes)
        for cohort in ('primary_control', 'secondary_control')
    }
    all_arrays = {'incident_full': full, 'incident': incident, **controls}
    output.mkdir(parents=True, exist_ok=False)
    results = {family: {level: {} for level in LEVELS} for family in FAMILIES}
    event_files = []
    for family_number, family in enumerate(FAMILIES):
        for level_number, level in enumerate(LEVELS):
            for cohort_number, cohort in enumerate(COHORTS):
                source = all_arrays[cohort]
                parts, corrections, clusters, fold_numbers = [], [], [], []
                known_active = active_total = 0
                for fold in folds:
                    indices = (fold['full_audit_indices'] if cohort == 'incident_full'
                               else fold['common_audit_indices'])
                    part = subset_arrays(source, indices)
                    lookup, known = models[(fold['fold'], family, level)]
                    correction, current_known, current_total = correction_from_lookup(
                        part, lookup, known, level)
                    parts.append(part)
                    corrections.append(correction)
                    weeks = full_weeks[indices] if cohort == 'incident_full' \
                        else common_weeks[indices]
                    clusters.append(weeks)
                    fold_numbers.append(np.full(len(indices), fold['fold'], dtype=np.int64))
                    known_active += current_known
                    active_total += current_total
                combined = concatenate_arrays(parts)
                combined_correction = np.concatenate(corrections, axis=0)
                combined_weeks = np.concatenate(clusters)
                combined_folds = np.concatenate(fold_numbers)
                metrics = correction_metrics(combined, combined_correction)
                metrics['uncertainty'] = uncertainty(
                    metrics, combined_weeks, protocol,
                    int(protocol['uncertainty']['seed']) +
                    family_number * 100 + level_number * 10 + cohort_number)
                metrics['lookup_coverage'] = {
                    'known_valid_support_cells': int(known_active),
                    'valid_support_cells': int(active_total),
                    'fraction': float(known_active / active_total),
                }
                event_path = output / f'{family}_{level}_{cohort}_events.csv'
                write_event_metrics(
                    event_path, family, level, cohort, metrics,
                    combined['positive_sample_index'], combined_weeks, combined_folds)
                event_files.append(event_path)
                metrics.pop('_per_event')
                results[family][level][cohort] = metrics

    model_path = output / 'lookup_models.npz'
    with model_path.open('wb') as stream:
        np.savez_compressed(stream, **model_arrays)
    gate = decision(results, protocol)
    summary = {
        'status': 'NODE_PHASE_REPEATABILITY_AUDIT_COMPLETE',
        'scope': protocol['scope'], 'protocol_id': protocol['protocol_id'],
        'protocol_sha256': sha256(protocol_path), 'main_training_ready': False,
        'train_residual_arrays_only': True,
        'validation_residual_arrays_read': False, 'test_split_read': False,
        'control_future_Y_used_for_lookup_fit': False,
        'neural_expert_trained': False, 'rolling_origin_fit': True,
        'folds': fold_summaries, 'results': results,
        'development_gate': gate, 'support': protocol['support'],
        'uncertainty': protocol['uncertainty'],
        'limitations': [
            'This audit tests only stable node identity and two fixed onset phases.',
            'The incident-median lookup is diagnostic and has no routine zero anchor.',
            'The zero-anchored lookup is intentionally simple and is not a neural expert.',
            'A passing result authorizes model development, not validation or test claims.',
        ],
        'inputs': {
            'v7a_summary_sha256': sha256(Path(materialized_dir) / 'summary.json'),
            'v7a_protocol_sha256': v7a_summary['protocol_sha256'],
            'positive_package': package_hashes,
            **input_hashes,
            'protocol_sha256': sha256(protocol_path),
            'code_sha256': sha256(__file__),
        },
        'environment': {
            'python_version': sys.version, 'numpy_version': np.__version__,
            'threads_requested': {
                name: os.environ.get(name)
                for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')
            },
            'git_head': subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
        },
        'interpretation': protocol['interpretation'],
    }
    outputs = [model_path, *event_files]
    summary['outputs'] = {
        path.name: {'sha256': sha256(path), 'bytes': path.stat().st_size}
        for path in outputs
    }
    summary_path = output / 'summary.json'
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
        encoding='utf-8')
    print(json.dumps({
        'status': summary['status'], 'development_gate': gate,
        'primary_result': results['zero_anchored_mean']['node_phase'],
    }, indent=2, ensure_ascii=False), flush=True)
    print(f'Saved v7c node-phase repeatability audit: {summary_path}', flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--primary-control-dir', type=Path, required=True)
    parser.add_argument('--secondary-control-dir', type=Path, required=True)
    parser.add_argument('--materialized-dir', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, default=Path(__file__).with_name(
        'node_phase_repeatability_v7c.json'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    audit(args.data_dir, args.primary_control_dir, args.secondary_control_dir,
          args.materialized_dir, args.protocol, args.output)


if __name__ == '__main__':
    main()
