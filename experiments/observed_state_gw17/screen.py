"""Apply the predeclared GW17 development screen to authenticated val caches."""
import argparse
import json
from pathlib import Path

import numpy as np

from . import ARMS
from .common import exclusive_run, read_json, require, sha256, source_hashes, write_json
from .evaluate import read_archive, require_reference_axes, selected_checkpoint, validation_replay
from .reference import read_reference, verify_reference
from .train import validate_identity, validate_runtime

CANDIDATE = 'INNOVATION_STATE'
CONTROL = 'OBSERVATION_CONTROL'
REFERENCE = 'GW16_REFERENCE'
RISK_METRICS = ('rmse', 'mape_percent', 'window_p95')


def require_same_axes(left, right):
    for key in ('target', 'reference', 'sample_ids', 'station_ids', 'assignment',
                'support', 'available', 'history_target', 'history_valid'):
        require(left[key].dtype == right[key].dtype and np.array_equal(left[key], right[key]),
            'Paired GW17 validation archives differ in '+key)


def finite_positive(value, name):
    require(value is not None and np.isfinite(value) and value > 0,
        'Invalid screening metric: '+name)
    return float(value)


def risk_value(diagnostics, name):
    value = diagnostics[name] if name == 'window_p95' else diagnostics['overall'][name]
    return finite_positive(value, name)


def screen_records(best, last, identity, reference_metadata):
    """Pure decision rule; collect_records separately verifies files and axes."""
    require(set(best) == set(last) == set(ARMS), 'Both GW17 BEST and LAST arms are required')
    batches = identity['data']['loader_batching']['train_loader']['batches']
    require(reference_metadata['arm'] == 'STATE_EXCHANGE'
        and reference_metadata['split'] == 'val'
        and reference_metadata['test_evaluated'] is False
        and reference_metadata['engineering_only'] is False,
        'Reference must be the historical formal GW16 STATE_EXCHANGE validation archive')
    reference_role = reference_metadata['checkpoint_role']
    require(reference_role['selection'] == 'best' and reference_role['full_protocol_reached'] is True
        and reference_role['completed_training_epochs'] == 100
        and reference_role['best_includes_stage_zero'] is True,
        'Historical GW16 reference must be stage-zero-inclusive full-100 BEST')
    for selection, records in (('best', best), ('last', last)):
        for arm, record in records.items():
            require(record['identity_sha256'] == identity['identity_sha256']
                and record['arm'] == arm and record['split'] == 'val'
                and record['test_evaluated'] is False
                and record['gw16_reference'] == identity['gw16_reference'],
                'Screen archive identity, split or reference mismatch')
            role = record['checkpoint_role']
            require(role['selection'] == selection and role['stage'] == 'joint'
                and role['best_includes_stage_zero'] is True,
                'GW17 requires stage-zero-inclusive BEST and joint training')
            require(role['registered_parameters'] == identity['initializations'][arm]['registered_parameters'],
                'Screen capacity differs from initialization identity')
            require(role['training_batches_per_epoch'] == batches, 'Screen training batch budget differs')
            require(0 <= role['epoch'] <= role['completed_training_epochs'], 'Invalid selected checkpoint epoch')
            finite_positive(record['diagnostics']['overall']['mae'], arm+' MAE')
    for arm in ARMS:
        br, lr = best[arm]['checkpoint_role'], last[arm]['checkpoint_role']
        require(br['completed_training_epochs'] == lr['completed_training_epochs']
            and br['global_updates'] == lr['global_updates'],
            'BEST and LAST were collected at different training budgets')
        require(lr['epoch'] == lr['completed_training_epochs'], 'LAST is not the final completed epoch')

    init_control, init_candidate = (identity['initializations'][arm] for arm in (CONTROL, CANDIDATE))
    counts_equal = (init_control['registered_parameters'] == init_candidate['registered_parameters']
        and init_candidate['registered_parameters'] > 0)
    tensors_paired = init_control['model_tensor_sha256'] == init_candidate['model_tensor_sha256']
    budget = (identity['config']['max_epochs'] == 100 and all(
        record['checkpoint_role']['completed_training_epochs'] == 100
        and record['checkpoint_role']['full_protocol_reached'] is True
        and record['checkpoint_role']['global_updates'] == 100*batches
        for records in (best, last) for record in records.values()))
    candidate = best[CANDIDATE]['diagnostics']
    comparisons = {CONTROL:best[CONTROL]['diagnostics'], REFERENCE:reference_metadata['diagnostics']}
    candidate_mae = finite_positive(candidate['overall']['mae'], 'candidate MAE')
    relative_gains = {name:1-candidate_mae/finite_positive(d['overall']['mae'], name+' MAE')
        for name, d in comparisons.items()}
    last_mae = finite_positive(last[CANDIDATE]['diagnostics']['overall']['mae'], 'LAST MAE')
    degradation = last_mae/candidate_mae-1
    risk_checks, risk_results = {}, {}
    for comparator, diagnostics in comparisons.items():
        risk_results[comparator] = {}
        for name in RISK_METRICS:
            current, baseline = risk_value(candidate, name), risk_value(diagnostics, name)
            passed = current <= baseline
            risk_checks['BEST_'+name+'_no_worse_than_'+comparator] = passed
            risk_results[comparator][name] = dict(candidate=current, reference=baseline, passed=passed)
    checks = dict(full_two_arm_100_epoch_budget=budget,
        research_scope_not_engineering=identity['engineering_only'] is False,
        exact_equal_parameter_counts=counts_equal,
        exact_paired_initialization=tensors_paired,
        BEST_gain_vs_OBSERVATION_CONTROL_at_least_0_005=relative_gains[CONTROL] >= .005,
        BEST_gain_vs_GW16_REFERENCE_at_least_0_005=relative_gains[REFERENCE] >= .005,
        LAST_own_degradation_at_most_0_01=degradation <= .01,
        **risk_checks)
    checks = {name:bool(passed) for name, passed in checks.items()}
    eligible = all(checks.values())
    return dict(schema='observed_state_gw17_screening_v1', status='GW17_SCREENING_COMPLETE',
        checks=checks, candidate_for_confirmation=eligible, development_eligible=eligible,
        sole_candidate=CANDIDATE, BEST_candidate_MAE=candidate_mae, LAST_candidate_MAE=last_mae,
        BEST_relative_gains=relative_gains, LAST_own_relative_degradation=degradation,
        risk_results=risk_results,
        parameter_counts={arm:identity['initializations'][arm]['registered_parameters'] for arm in ARMS},
        run_identity_sha256=identity['identity_sha256'], gw16_reference=identity['gw16_reference'],
        engineering_only=identity['engineering_only'],
        historical_reference_is_equal_budget_control=False,
        historical_reference_new_model_forwards=0,
        matched_capacity_does_not_claim_equal_compute=True,
        interpretation='predeclared_single_seed_development_screen_not_statistical_confirmation',
        development_validation_has_informed_design=True,
        previous_GW16_test_has_been_disclosed=True,
        screening_pass_does_not_open_test=True,
        screening_failure_does_not_authorize_budget_extension=True,
        test_evaluated=False, independent_confirmation=False)


def collect_records(run, best_dir, last_dir, identity):
    reference_arrays, reference_metadata, record = read_reference(identity['gw16_reference']['directory'])
    require(record == identity['gw16_reference'], 'Historical GW16 reference changed')
    best, last, artifacts = {}, {}, {}
    first = None
    batches = identity['data']['loader_batching']['train_loader']['batches']
    for selection, directory, records in (('best', best_dir, best), ('last', last_dir, last)):
        artifacts[selection] = {}
        for arm in ARMS:
            arrays, metadata = read_archive(directory/arm)
            require(metadata['identity'] == identity and metadata['arm'] == arm,
                'Screen archive does not belong to this run/arm')
            weights, role, expected, checkpoint_path = selected_checkpoint(run, identity, arm, selection, batches)
            del weights
            require(role == metadata['checkpoint_role'], 'Screen selected checkpoint differs from archive')
            require(sha256(checkpoint_path) == role['checkpoint_sha256'], 'Screen checkpoint changed')
            require(validation_replay(metadata['diagnostics'], expected)['passed'],
                'Screen metrics differ from the selected training checkpoint')
            validate_runtime(read_json(run/'runtime.json'), metadata['runtime'])
            if first is None:
                first = arrays
            else:
                require_same_axes(first, arrays)
            if not identity['engineering_only']:
                require_reference_axes(arrays, reference_arrays)
            records[arm] = metadata
            artifacts[selection][arm] = dict(archive_sha256=metadata['archive_sha256'],
                metadata_sha256=sha256(directory/arm/'metadata.json'), checkpoint_role=role)
    verify_reference(identity['gw16_reference'])
    return best, last, artifacts, reference_metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--best-dir', type=Path)
    parser.add_argument('--last-dir', type=Path)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args(argv)
    run = args.run_dir.resolve()
    best_dir, last_dir = ((path or run/default).resolve()
        for path, default in ((args.best_dir, 'eval_best_val'), (args.last_dir, 'eval_last_val')))
    output = (args.output_dir or run).resolve()
    with exclusive_run(run):
        identity = read_json(run/'identity.json')
        validate_identity(identity)
        require(source_hashes() == identity['execution_source'], 'Screen source differs from training identity')
        require(output == run or run in output.parents, 'Screen output must stay in this run directory')
        best, last, artifacts, reference_metadata = collect_records(run, best_dir, last_dir, identity)
        result = screen_records(best, last, identity, reference_metadata)
        result['evaluations'] = artifacts
        require(source_hashes() == identity['execution_source'], 'Execution source changed during screen')
        output.mkdir(parents=True, exist_ok=True)
        write_json(output/'screening.json', result)
        print(json.dumps(dict(status=result['status'], candidate_for_confirmation=result['candidate_for_confirmation'],
            screening=str(output/'screening.json'), test_evaluated=False), ensure_ascii=False))
        return result


if __name__ == '__main__':
    main()
