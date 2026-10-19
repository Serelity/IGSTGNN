"""Certified-cache CPU diagnostic; an audit failure never opens validation scoring."""
import argparse
import gc
import json
from pathlib import Path

import numpy as np

from experiments.branch_readout_gw9 import readout_auth as auth
from experiments.conditional_residual_gw10 import run_probe as cache_reader
from experiments.conditional_residual_gw10 import screen as old_metrics
from experiments.state_gate_gw11.reference import _export_repo_from_cache
from experiments.trajectory_diagnostic_gw13 import features, model, protocol


SCHEMA = 'gw13_trajectory_information_diagnostic_v1'
FIT_STOP, SCORE_START, TRAIN_COUNT = 1826, 1954, 3909
GROUPS = ('high_value_ge314', 'long_horizon_7_12', 'flow_50_200')
REPO = Path(__file__).resolve().parents[2]
SOURCE_FILES = ('experiments/trajectory_diagnostic_gw13/__init__.py',
    'experiments/trajectory_diagnostic_gw13/features.py', 'experiments/trajectory_diagnostic_gw13/model.py',
    'experiments/trajectory_diagnostic_gw13/protocol.py', 'experiments/trajectory_diagnostic_gw13/run.py',
    'GW13_SOURCE_MANIFEST.json')
METHOD = dict(candidate='TRAJECTORY16', audit_fit_bounds=[0, FIT_STOP],
    audit_excluded_bounds=[FIT_STOP, SCORE_START], audit_score_bounds=[SCORE_START, TRAIN_COUNT],
    author_order_is_certified_calendar_order=False, window_nonoverlap_certified=False,
    audit_scope='new_parent_and_shared_lookup_only_SUM_previously_trained_on_all_train',
    all_parent_tables_and_cuts_refitted_within_audit_prefix=True,
    audit_failure_skips_full_train_fit_and_validation_score=True,
    train_only_cutpoints=True, validation_used_to_select_candidate=False,
    neural_inference_calls=0, neural_optimizer_steps=0,
    clocks_conditioned_in_new_lookup=False, replicates_GW12_clock_control=False,
    turn_position_used_for_fitting=False, turn_position_used_for_descriptive_groups_only=True,
    child_scoring_activity_minimum=.25, activity_is_an_operational_not_statistical_threshold=True,
    validation_relative_gain_minimum=.005,
    validation_gain_controls=['SUM', 'OWN4', 'CONTEXT'],
    strict_shape_controls=['MAGNITUDE16', 'DIRECTION16'],
    entire_author_object_array_deserialized=True, test_samples_accessed=False,
    test_evaluated=False, prior_failed_screens_unchanged=True)


def slice_data(data, start, stop):
    return dict(pred=data['pred'][start:stop], y=data['y'][start:stop], valid=data['valid'][start:stop],
        sample_ids=data['sample_ids'][start:stop],
        features={key:value[start:stop] for key,value in data['features'].items()})


def empty_metrics():
    return dict(valid_count=0, mae=None, rmse=None, mape_fraction=None, signed_bias=None)


def score(fitted, data):
    """Stream arms; output bounded summaries instead of retaining seven cubes."""
    valid, y = data['valid'], data['y']
    h = np.arange(y.shape[1])[None, :, None]
    masks = dict(high_value_ge314=valid & (y >= 314),
        long_horizon_7_12=valid & (h >= 6), flow_50_200=valid & (y >= 50) & (y < 200))
    records, groups, rows, residual_rows, gains = {}, {}, [], [], []
    own_reference = None
    shape = data['features']['shape']
    turn = data['features']['turn_position']
    strata = [('shape', name, shape == index) for index,name in enumerate(
        ('early_down_recent_down', 'early_down_recent_nondecreasing',
         'early_nondecreasing_recent_down', 'early_nondecreasing_recent_nondecreasing'))]
    strata += [('turn_position', 'none', turn == -1), ('turn_position', 'early_2_4', (turn >= 2) & (turn <= 4)),
        ('turn_position', 'middle_5_6', (turn >= 5) & (turn <= 6)),
        ('turn_position', 'late_7_9', (turn >= 7) & (turn <= 9))]
    for arm, prediction in model.predictions(fitted, data):
        records[arm] = old_metrics.metrics(prediction, y, valid)
        groups[arm] = {name: old_metrics.metrics(prediction, y, selected) if selected.any()
                       else empty_metrics() for name,selected in masks.items()}
        if arm == 'OWN4':
            own_reference = prediction
        for dimension, name, selection in strata:
            selected = valid & selection[:, None, :]
            row = old_metrics.metrics(prediction, y, selected) if selected.any() else empty_metrics()
            rows.append(dict(dimension=dimension, group=name, arm=arm, **row))
            if arm == 'SUM':
                residual = y[selected].astype(np.float64)-prediction[selected]
                residual_rows.append(dict(dimension=dimension, group=name, valid_count=int(selected.sum()),
                    desired_correction_median=float(np.median(residual)) if residual.size else None,
                    desired_correction_positive_fraction=float((residual > 0).mean()) if residual.size else None,
                    interpretation='descriptive_association_not_incremental_predictive_evidence'))
        if arm == 'TRAJECTORY16':
            auth.require(own_reference is not None, 'OWN4 reference must precede the candidate')
            for index in range(y.shape[1]):
                selected = valid & (h == index)
                gains.append(dict(dimension='horizon', group=str(index+1), **old_metrics.paired(
                    own_reference, prediction, y, selected)))
            for start in range(0, y.shape[0], 128):
                selected = valid.copy()
                selected[:start] = False
                selected[start+128:] = False
                gains.append(dict(dimension='author_order_block', group=str(int(data['sample_ids'][start])),
                    **old_metrics.paired(own_reference, prediction, y, selected)))
    auth.require(tuple(records) == model.ARMS, 'Unexpected diagnostic arms or order')
    return dict(records=records, groups=groups, strata=rows, residual_associations=residual_rows,
                paired_candidate_vs_OWN4=gains)


def report(result):
    lines = ['# GW13 history trajectory information diagnostic', '',
        'Fixed certified SUM predictions; no new neural inference or optimizer steps.',
        'The author-order audit holds out only the new lookup, not the previously trained SUM.',
        'Shared lookup context controls lead, last-flow bin and amplitude quartile; no clock conditioning.',
        'The coarse shape code is not the full trajectory, an exact turning-time model, or an architecture.', '',
        '| Case | Author-order audit MAE | Development MAE |', '|---|---:|---:|']
    validation = result.get('validation')
    for arm in model.ARMS:
        later = '{:.9f}'.format(validation['records'][arm]['mae']) if validation else 'not scored'
        lines.append('| {} | {:.9f} | {} |'.format(arm, result['audit']['records'][arm]['mae'], later))
    lines += ['', '| Fixed audit check | Passed |', '|---|---|']
    for name,value in result['audit_decision']['checks'].items():
        lines.append('| {} | {} |'.format(name, value))
    if validation:
        lines += ['', '| Development check | Passed |', '|---|---|']
        for name,value in result['screening']['checks'].items():
            lines.append('| {} | {} |'.format(name, value))
    lines += ['', 'audit_passed: '+str(result['audit_decision']['audit_passed']),
        'validation_scored: '+str(validation is not None),
        'trajectory_information_screen_passed: '+str(result['screening']['trajectory_information_screen_passed']),
        'Check coverage.csv before interpreting a negative result. Coverage is eligible current-level cells, not independent samples.',
        'Turn-position strata and sign associations are descriptive only; they do not select a different candidate.',
        'Same nominal table sizes do not ensure equal effective capacity or occupied cells.',
        'This is reused development exploration. No statistical significance, causal mechanism, novelty or independent confirmation is established.',
        'A failure does not authorize retuning, extending GW12, or automatically starting a neural model.', '']
    return '\n'.join(lines)


def write_outputs(output, result):
    cache_reader.write_json(output/'analysis.json', result)
    cache_reader.write_json(output/'screening.json', result['screening'])
    coverage_rows = []
    for phase in ('audit', 'validation'):
        if result.get(phase):
            value = result[phase]
            cache_reader.write_csv(output/(phase+'_cases.csv'), (dict(arm=k, **v) for k,v in value['records'].items()))
            cache_reader.write_csv(output/(phase+'_groups.csv'), (dict(arm=arm,group=name,**row)
                for arm,rows in value['groups'].items() for name,row in rows.items()))
            cache_reader.write_csv(output/(phase+'_strata.csv'), value['strata'])
            cache_reader.write_csv(output/(phase+'_residual_associations.csv'), value['residual_associations'])
            cache_reader.write_csv(output/(phase+'_paired.csv'), value['paired_candidate_vs_OWN4'])
            coverage_rows += [dict(phase=phase, arm=arm, **row) for arm,row in value['coverage'].items()]
    cache_reader.write_csv(output/'coverage.csv', coverage_rows)
    (output/'report.md').write_text(report(result), encoding='utf-8')


def run(probe_dir, cache_dir, data_dir, output_dir):
    auth.require(model.ARMS == protocol.ARMS and protocol.MIN_ACTIVE_FRACTION == .25
        and protocol.MIN_RELATIVE_GAIN == .005, 'Model and locked screening protocol differ')
    probe_dir, cache_dir, data_dir, output = map(lambda p:Path(p).resolve(),
                                               (probe_dir, cache_dir, data_dir, output_dir))
    archive, last, certificate = auth.load_authenticated_run(probe_dir, repo=_export_repo_from_cache(cache_dir))
    del last
    auth.require_output_outside_sources(output, probe_dir, certificate)
    for path in (cache_dir, data_dir):
        auth.require(output != path and path not in output.parents and output not in path.parents,
                     'Output must be a fresh directory outside cache and original data')
    cached, cache_meta, digests = cache_reader.load_cache(cache_dir, archive, certificate)
    del archive
    execution = {str(REPO/name):auth.sha256(REPO/name) for name in SOURCE_FILES}
    digests.update(execution)
    output.mkdir(parents=True, exist_ok=False)
    cache_reader.write_json(output/'protocol.json', dict(schema=SCHEMA, method=METHOD, model=model.PROTOCOL,
        source_identity_sha256=certificate['source_identity_sha256'], cache_digests=digests,
        execution_source=execution, expected_data_identity=certificate['data_identity']))
    try:
        trajectories, history_meta = features.load_features(data_dir, certificate['data_identity'], cached)
        digests.update(history_meta['file_digests'])
        for split in ('train', 'val'):
            cached[split]['features'].update(trajectories[split])
        del trajectories
        train = cached['train']
        auth.require(len(train['sample_ids']) == TRAIN_COUNT, 'Original train count differs')
        print(json.dumps(dict(phase='audit_fit', samples=FIT_STOP, validation_scoring=False)), flush=True)
        audit_model = model.fit_model(slice_data(train, 0, FIT_STOP))
        model.save_model(audit_model, output/'audit_model.json')
        audit_digest = auth.sha256(output/'audit_model.json')
        audit_data = slice_data(train, SCORE_START, TRAIN_COUNT)
        audit = score(audit_model, audit_data)
        audit['coverage'] = model.coverage(audit_model, audit_data['features'], audit_data['valid'])
        audit_decision = protocol.audit_screen(audit['records'], audit['coverage'])
        cache_reader.write_json(output/'audit_decision.json', audit_decision)
        cache_reader.write_csv(output/'audit_cases.csv', (dict(arm=k, **v) for k,v in audit['records'].items()))
        cache_reader.write_csv(output/'audit_strata.csv', audit['strata'])
        cache_reader.write_csv(output/'audit_residual_associations.csv', audit['residual_associations'])
        del audit_model, audit_data
        gc.collect()
        result = dict(schema=SCHEMA, method=METHOD, model_protocol=model.PROTOCOL,
            source_identity_sha256=certificate['source_identity_sha256'], cache_metadata_sha256=auth.sha256(cache_dir/'cache.json'),
            source_certificate=certificate, input_digests=digests, history_metadata=history_meta,
            audit=audit, audit_decision=audit_decision, validation=None,
            neural_inference_calls=0, neural_optimizer_steps=0, test_evaluated=False)
        if audit_decision['validation_scoring_permitted']:
            print(json.dumps(dict(phase='audit_passed_fit_all_train', samples=TRAIN_COUNT)), flush=True)
            fitted = model.fit_model(train)
            model.save_model(fitted, output/'model.json')
            fitted_digest = auth.sha256(output/'model.json')
            cache_reader.write_json(output/'fitted_before_validation.json', dict(model_sha256=fitted_digest,
                protocol_sha256=auth.sha256(output/'protocol.json'), validation_labels_used_for_fit=False))
            validation = score(fitted, cached['val'])
            validation['coverage'] = model.coverage(fitted, cached['val']['features'], cached['val']['valid'])
            result['validation'] = validation
            result['screening'] = protocol.validation_screen(validation['records'], validation['groups'], audit_decision)
            auth.require(auth.sha256(output/'model.json') == fitted_digest, 'Fitted model changed during scoring')
            result['model_sha256'] = fitted_digest
        else:
            result['screening'] = dict(status='GW13_AUDIT_STOPPED_VALIDATION_NOT_SCORED',
                reason=audit_decision['outcome'],
                candidate='TRAJECTORY16', checks=audit_decision['checks'],
                trajectory_information_screen_passed=False, eligible_for_architecture_design=False,
                validation_scored=False, candidate_for_confirmation=False,
                independent_confirmation_authorized=False, test_evaluated=False,
                prior_failed_screens_unchanged=True, retuning_authorized=False)
        auth.assert_unchanged(certificate)
        cache_reader.rehash(digests)
        auth.require(auth.sha256(output/'audit_model.json') == audit_digest, 'Audit model changed')
        result['audit_model_sha256'] = audit_digest
        write_outputs(output, result)
        print(json.dumps(dict(status=result['screening']['status'], output_dir=str(output),
            audit_passed=audit_decision['audit_passed'], validation_scored=result['validation'] is not None)), flush=True)
        return result
    except BaseException as error:
        cache_reader.write_json(output/'failure.json', dict(status='GW13_DIAGNOSTIC_FAILED',
            error_type=type(error).__name__, error=str(error), source_artifacts_written=False))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('probe-dir', 'cache-dir', 'data-dir', 'output-dir'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args(argv)
    return run(args.probe_dir, args.cache_dir, args.data_dir, args.output_dir)


if __name__ == '__main__':
    main()
