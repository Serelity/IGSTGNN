"""Read authenticated BEST/LAST validation caches; apply the locked GW16 screen."""
import argparse
import json
from pathlib import Path

import numpy as np

from . import ARMS
from .common import (exclusive_run, read_json, require,
                     sha256, source_hashes, write_json)
from .evaluate import read_archive
from .train import validate_identity

CANDIDATE = 'STATE_EXCHANGE'
MECHANISM_CONTROLS = ('DIRECT', 'FREE_RECURRENT', 'ORIGIN_EXCHANGE')
GUARDS = ('high_value_ge314', 'long_horizon_7_12', 'flow_50_200')


def require_same_axes(left, right):
    # Predictions and latent states may differ. Input-derived relation masks,
    # fixed persistence references, labels and all identities must be identical.
    for key in ('target', 'reference', 'sample_ids', 'station_ids', 'assignment', 'support', 'available'):
        require(left[key].dtype == right[key].dtype and np.array_equal(left[key], right[key]),
                'Paired validation archives differ in '+key)


def screen_records(best, last, identity):
    """Pure decision rule; loading/authentication is performed by collect_records."""
    require(set(best) == set(last) == set(ARMS), 'The screen requires all five BEST and LAST arms')
    for selection, records in (('best', best), ('last', last)):
        for arm, record in records.items():
            require(record['identity_sha256'] == identity['identity_sha256'] and record['arm'] == arm
                and record['split'] == 'val' and record['test_evaluated'] is False
                and record['checkpoint_role']['selection'] == selection,
                'Screen archive identity, selection or split mismatch')
            role = record['checkpoint_role']
            require(role['registered_parameters'] == identity['initializations'][arm]['registered_parameters'],
                'Screen capacity differs from initialization identity')
            require(role['training_batches_per_epoch'] == identity['data']['loader_batching']['train_loader']['batches'],
                'Screen training batch budget differs')
            require(role['best_includes_stage_zero'] is True and role['stage'] == 'joint',
                'GW16 requires stage-zero-inclusive BEST and a single joint phase')
            require(0 <= role['epoch'] <= role['completed_training_epochs'], 'Invalid selected checkpoint epoch')
            require(np.isfinite(record['diagnostics']['overall']['mae'])
                and record['diagnostics']['overall']['mae'] > 0, 'Invalid screening MAE')
    for arm in ARMS:
        require(best[arm]['checkpoint_role']['completed_training_epochs']
                == last[arm]['checkpoint_role']['completed_training_epochs']
            and best[arm]['checkpoint_role']['global_updates'] == last[arm]['checkpoint_role']['global_updates'],
            'BEST and LAST were collected at different training budgets')
        require(last[arm]['checkpoint_role']['epoch'] == last[arm]['checkpoint_role']['completed_training_epochs'],
            'LAST does not identify the final completed epoch')

    candidate_parameters = identity['initializations'][CANDIDATE]['registered_parameters']
    require(candidate_parameters > 0, 'Invalid candidate parameter count')
    capacity_deviations = {arm:abs(identity['initializations'][arm]['registered_parameters']-candidate_parameters)
        /candidate_parameters for arm in ('DIRECT', 'FREE_RECURRENT')}
    matched_capacity = all(value <= .02 for value in capacity_deviations.values())
    # These two arms have the same tensors, inputs and update rules; their only
    # intended difference is origin-fixed versus evolving-state exchange weights.
    origin_init = identity['initializations']['ORIGIN_EXCHANGE']
    state_init = identity['initializations'][CANDIDATE]
    paired_initialization = (origin_init['registered_parameters'] == state_init['registered_parameters']
        and origin_init['model_tensor_sha256'] == state_init['model_tensor_sha256'])
    require(paired_initialization, 'Origin/state exchange initialization is not an exact tensor pair')
    full_budget = (identity['config']['max_epochs'] == 100 and all(
        record['checkpoint_role']['completed_training_epochs'] == 100
        and record['checkpoint_role']['full_protocol_reached'] is True
        and record['checkpoint_role']['global_updates'] == 100*record['checkpoint_role']['training_batches_per_epoch']
        for records in (best, last) for record in records.values()))
    candidate_mae = best[CANDIDATE]['diagnostics']['overall']['mae']
    relative_gains = {arm:1-candidate_mae/best[arm]['diagnostics']['overall']['mae']
                      for arm in (*MECHANISM_CONTROLS, 'NATIVE_POS')}
    # One comparator is chosen by overall BEST MAE and used for every guard.
    # No per-group comparator switching; declared tuple order resolves ties.
    guard_reference = min(MECHANISM_CONTROLS,
        key=lambda arm:(best[arm]['diagnostics']['overall']['mae'], MECHANISM_CONTROLS.index(arm)))
    group_results = {}
    for group in GUARDS:
        candidate = best[CANDIDATE]['diagnostics']['groups'][group]
        control = best[guard_reference]['diagnostics']['groups'][group]
        require(candidate['valid_count'] == control['valid_count'], 'Paired group target count differs: '+group)
        valid = (candidate['valid_count'] > 0 and candidate['mae'] is not None and control['mae'] is not None
            and np.isfinite(candidate['mae']) and np.isfinite(control['mae']))
        group_results[group] = dict(reference=guard_reference, candidate_mae=candidate['mae'],
            reference_mae=control['mae'], valid_count=candidate['valid_count'],
            passed=bool(valid and candidate['mae'] <= control['mae']))
    last_mae = last[CANDIDATE]['diagnostics']['overall']['mae']
    degradation = last_mae/candidate_mae-1
    checks = dict(full_five_arm_100_epoch_budget=bool(full_budget),
        research_scope_not_engineering=identity['engineering_only'] is False,
        matched_capacity_with_DIRECT_and_FREE_within_2_percent=bool(matched_capacity),
        exact_ORIGIN_STATE_initialization=bool(paired_initialization),
        BEST_gain_vs_DIRECT_at_least_0_005=relative_gains['DIRECT'] >= .005,
        BEST_gain_vs_FREE_at_least_0_005=relative_gains['FREE_RECURRENT'] >= .005,
        BEST_gain_vs_ORIGIN_at_least_0_005=relative_gains['ORIGIN_EXCHANGE'] >= .005,
        BEST_no_worse_than_NATIVE_POS=relative_gains['NATIVE_POS'] >= 0,
        LAST_own_degradation_at_most_0_01=degradation <= .01,
        **{'BEST_'+name+'_no_worse_than_fixed_guard_reference':result['passed']
           for name, result in group_results.items()})
    checks = {key:bool(value) for key, value in checks.items()}
    return dict(schema='position_state_gw16_screening_v1', status='GW16_SCREENING_COMPLETE',
        checks=checks, candidate_for_confirmation=all(checks.values()),
        sole_candidate=CANDIDATE, BEST_candidate_MAE=candidate_mae, LAST_candidate_MAE=last_mae,
        BEST_relative_gains=relative_gains, LAST_own_relative_degradation=degradation,
        fixed_guard_reference=guard_reference, guard_results=group_results,
        gain_pct_vs_strongest_mechanism_control=100*relative_gains[guard_reference],
        practical_target_3pct_met=relative_gains[guard_reference] >= .03,
        practical_target_is_report_only_not_screen_gate=True,
        minimum_screen_gain_is_not_paper_level_confirmation=True,
        matched_capacity_claim=bool(matched_capacity), capacity_relative_absolute_deviations=capacity_deviations,
        capacity_denominator='STATE_EXCHANGE registered parameters; both directions bounded by 2%',
        parameter_counts={arm:identity['initializations'][arm]['registered_parameters'] for arm in ARMS},
        native_control_capacity_matched_claimed=False,
        run_identity_sha256=identity['identity_sha256'], engineering_only=identity['engineering_only'],
        interpretation='predeclared_single_seed_development_screen_not_statistical_confirmation',
        development_validation_has_informed_design=True,
        screening_failure_does_not_authorize_budget_extension=True,
        historical_results_are_equal_budget_controls=False,
        test_evaluated=False, independent_confirmation=False)


def collect_records(run, best_dir, last_dir, identity):
    best, last, artifacts = {}, {}, {}
    first = None
    for selection, directory, records in (('best', best_dir, best), ('last', last_dir, last)):
        artifacts[selection] = {}
        for arm in ARMS:
            arrays, record = read_archive(directory/arm)
            require(record['identity'] == identity and record['arm'] == arm,
                'Screen archive does not belong to this run/arm')
            if first is None:
                first = arrays
            else:
                require_same_axes(first, arrays)
            records[arm] = record
            artifacts[selection][arm] = dict(archive_sha256=record['archive_sha256'],
                metadata_sha256=sha256(directory/arm/'metadata.json'),
                checkpoint_role=record['checkpoint_role'])
    return best, last, artifacts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--best-dir', type=Path)
    parser.add_argument('--last-dir', type=Path)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args(argv)
    run = args.run_dir.resolve()
    best_dir = (args.best_dir or run/'eval_best_val').resolve()
    last_dir = (args.last_dir or run/'eval_last_val').resolve()
    output = (args.output_dir or run).resolve()
    with exclusive_run(run):
        identity = read_json(run/'identity.json')
        validate_identity(identity)
        require(source_hashes() == identity['execution_source'], 'Screen source differs from the training identity')
        # Outputs stay in this run. Reading archives never loads models, opens a
        # dataset, refits a correction or changes a saved checkpoint.
        require(output == run or run in output.parents, 'Screen output must stay in this run directory')
        best, last, artifacts = collect_records(run, best_dir, last_dir, identity)
        result = screen_records(best, last, identity)
        result['evaluations'] = artifacts
        output.mkdir(parents=True, exist_ok=True)
        write_json(output/'screening.json', result)
        print(json.dumps(dict(status=result['status'], candidate_for_confirmation=result['candidate_for_confirmation'],
            screening=str(output/'screening.json'), test_evaluated=False), ensure_ascii=False))
        return result


if __name__ == '__main__':
    main()
