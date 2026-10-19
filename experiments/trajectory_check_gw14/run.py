"""CPU follow-up checks: immutable GW13 replay and train-budget-matched medians."""
import argparse
import gc
from numbers import Integral, Real
from pathlib import Path

import numpy as np

from experiments.branch_readout_gw9 import readout_auth as auth
from experiments.conditional_residual_gw10 import run_probe as io
from experiments.conditional_residual_gw10.screen import metrics, paired
from experiments.state_gate_gw11.reference import _export_repo_from_cache
from experiments.trajectory_diagnostic_gw13 import features, model as old_model, run as old_run
from experiments.trajectory_check_gw14 import cross, matching, source


SCHEMA = 'gw14_cross_and_matched_budget_checks_v1'
ARMS = ('OWN4', 'CONTEXT', 'MAGNITUDE16', 'DIRECTION16', 'TRAJECTORY16')
GROUPS = ('high_value_ge314', 'long_horizon_7_12', 'flow_50_200')
MIN_COMMON_FRACTION = .25
REPO = Path(__file__).resolve().parents[2]
SOURCE_FILES = ('experiments/trajectory_check_gw14/__init__.py',
    'experiments/trajectory_check_gw14/source.py', 'experiments/trajectory_check_gw14/cross.py',
    'experiments/trajectory_check_gw14/matching.py', 'experiments/trajectory_check_gw14/run.py',
    'GW14_SOURCE_MANIFEST.json')
METHOD = dict(source='returned_failed_GW13', candidate='TRAJECTORY16',
    old_models_refitted=False, old_cutpoints_changed=False,
    matched_new_statistical_fits_per_stage=3, matched_seed=2025,
    matching_uses_heldout_truth=False, min_points=64, min_author_blocks=3,
    author_block_size=128, shrinkage_pseudocount=128,
    common_scoring_activity_minimum=MIN_COMMON_FRACTION,
    audit_fit_bounds=[0,1826], audit_excluded_bounds=[1826,1954], audit_score_bounds=[1954,3909],
    matching_stage_order='audit_then_full_train_only_if_audit_passes',
    raw_historical_development_replayed_even_if_new_matched_audit_fails=True,
    matched_author_blocks_are_not_independent_calendar_blocks=True,
    SUM_already_trained_on_all_training_labels=True, test_evaluated=False,
    controls='same_retained_cells_per_context_and_same_per_cell_block_histogram_and_shrinkage',
    remaining_limits=['different_cell_partitions_and_sample_identities', 'single_fixed_sampling_seed',
        'common_support_population_may_differ_from_all_points', 'no_new_clock_conditioning',
        'historical_turn_detection_is_not_a_physical_delay_model'],
    source_GW13_failed_screen_unchanged=True,
    architecture_training_authorized=False, independent_confirmation_authorized=False)


def _empty():
    return dict(valid_count=0, mae=None, rmse=None, mape_fraction=None, signed_bias=None)


def evaluate_matched(fitted, data, phase):
    matching.validate_model(fitted)
    common, coverage = matching.score_masks(fitted, data['features'], data['valid'])
    require = source.require
    require(common.shape == data['valid'].shape and common.dtype == np.bool_
        and not (common & ~data['valid']).any(), 'Matched scoring must use a common subset of the native mask')
    require(coverage['common_valid_count'] == int(common.sum())
        and coverage['total_valid_count'] == int(data['valid'].sum())
        and np.isclose(coverage['common_fraction'], common.sum()/data['valid'].sum(), rtol=0., atol=1e-15),
        'Matched coverage must describe the common scoring population')
    predictions = dict(matching.predictions(fitted, data))
    require(tuple(predictions) == ARMS, 'Matched predictor arms or order differ')
    records = {arm:metrics(value,data['y'],common) if common.any() else _empty()
               for arm,value in predictions.items()}
    full = {arm:metrics(value,data['y'],data['valid']) for arm,value in predictions.items()}
    y = data['y']
    h = np.arange(y.shape[1])[None,:,None]
    masks = dict(high_value_ge314=common & (y>=314), long_horizon_7_12=common & (h>=6),
                 flow_50_200=common & (y>=50) & (y<200))
    groups = {arm:{name:metrics(value,y,mask) if mask.any() else _empty()
                   for name,mask in masks.items()} for arm,value in predictions.items()}
    comparisons = {control:paired(value,predictions['TRAJECTORY16'],y,common)
        for control,value in predictions.items() if control != 'TRAJECTORY16'}
    summary = None
    if common.any():
        summary = cross.summarize({arm:predictions[arm] for arm in cross.ARMS},
                                 dict(data,valid=common),phase=phase)
    return dict(records=records, full_native_mask_records=full, coverage=coverage, groups=groups,
        comparisons=comparisons, cross=summary, match_exact=fitted.get('match_exact') is True,
        common_mask_uses_heldout_labels_only_for_native_null_validity=True)


def matched_decision(result, *, development=False):
    """A robustness diagnostic; positive findings never promote the failed GW13."""
    coverage = result['coverage']
    records = result['records']
    source.require(set(records) == set(ARMS), 'All matched scoring arms are required')
    count, total, fraction = (coverage.get(key) for key in
        ('common_valid_count','total_valid_count','common_fraction'))
    source.require(all(isinstance(value,Integral) and not isinstance(value,(bool,np.bool_))
        for value in (count,total)) and total > 0 and 0 <= count <= total
        and isinstance(fraction,Real) and not isinstance(fraction,(bool,np.bool_)) and np.isfinite(fraction)
        and 0 <= fraction <= 1 and np.isclose(fraction,count/total,rtol=0.,atol=1e-15),
        'Matched common coverage is inconsistent or nonfinite')
    source.require(all(isinstance(row.get('valid_count'),Integral)
        and not isinstance(row['valid_count'],(bool,np.bool_)) and row['valid_count'] == count
        for row in records.values()),
                   'Matched metrics must describe the same scoring mask')
    candidate = records['TRAJECTORY16']['mae']
    for arm, row in records.items():
        if count:
            source.require(all(isinstance(row.get(key),Real) and not isinstance(row[key],(bool,np.bool_))
                and np.isfinite(row[key]) for key in ('mae','rmse','mape_fraction','signed_bias'))
                and row['mae'] >= 0 and row['rmse'] >= 0 and row['mape_fraction'] >= 0,
                'Finite matched metrics required: '+arm)
        else:
            source.require(all(row.get(key) is None for key in ('mae','rmse','mape_fraction','signed_bias')),
                           'Empty common support must not report a score')
    checks = dict(exact_training_budget_invariants=result['match_exact'] is True,
        common_activity_at_least_0_25=coverage['common_fraction'] >= MIN_COMMON_FRACTION and count > 0)
    for control in ('CONTEXT','MAGNITUDE16','DIRECTION16'):
        checks['candidate_beats_'+control] = count > 0 and candidate < records[control]['mae']
    if development:
        for name in GROUPS:
            candidate_group = result['groups']['TRAJECTORY16'][name]
            baseline_group = result['groups']['OWN4'][name]
            group_count = baseline_group.get('valid_count')
            source.require(isinstance(group_count,Integral) and not isinstance(group_count,(bool,np.bool_))
                and 0 <= group_count <= count and all(
                    isinstance(result['groups'][arm][name].get('valid_count'),Integral)
                    and not isinstance(result['groups'][arm][name]['valid_count'],(bool,np.bool_))
                    and result['groups'][arm][name]['valid_count'] == group_count for arm in ARMS),
                'Matched guard mask counts differ across arms: '+name)
            if group_count:
                source.require(all(isinstance(result['groups'][arm][name].get('mae'),Real)
                    and not isinstance(result['groups'][arm][name]['mae'],(bool,np.bool_))
                    and np.isfinite(result['groups'][arm][name]['mae'])
                    and result['groups'][arm][name]['mae'] >= 0 for arm in ARMS),
                    'Matched guard MAE must be finite: '+name)
            checks[name+'_available_and_no_worse_than_OWN4_on_common_mask'] = (
                candidate_group['valid_count'] > 0
                and candidate_group['valid_count'] == baseline_group['valid_count']
                and candidate_group['mae'] <= baseline_group['mae'])
    checks = {key:bool(value) for key,value in checks.items()}
    passed = all(checks.values())
    support = checks['common_activity_at_least_0_25']
    return dict(status='GW14_MATCHED_DEVELOPMENT_CHECK_COMPLETE' if development else 'GW14_MATCHED_AUDIT_COMPLETE',
        checks=checks, common_support_sufficient=bool(support),
        matched_information_advantage_observed=passed,
        new_matched_development_scoring_permitted=passed if not development else False,
        interpretation=('inconclusive_common_support' if not support else
            'positive_robustness_signal_under_declared_matching' if passed else 'advantage_not_retained_under_fixed_matching'),
        original_GW13_screen_passed=False, original_GW13_failed_screen_unchanged=True,
        candidate_for_confirmation=False, architecture_training_authorized=False,
        independent_confirmation_authorized=False, statistical_significance_claimed=False,
        test_evaluated=False, retuning_authorized=False)


def write_cross(output, prefix, summary):
    if summary is not None:
        for key, suffix in (('cross_rows','cross'),('phase_lead_rows','phase_lead'),('block_rows','blocks')):
            io.write_csv(output/(prefix+'_'+suffix+'.csv'), summary[key])


def coverage_row(phase, stage):
    row = dict(phase=phase,**{key:value for key,value in stage['coverage'].items()
                            if not isinstance(value,dict)})
    for arm,values in stage['coverage'].get('per_arm_active',{}).items():
        for key,value in values.items():
            row[arm+'_'+key] = value
    for arm,values in stage.get('fit_stage_counts',{}).get('per_arm',{}).items():
        for key,value in values.items():
            row[arm+'_fit_'+key] = value
    return row


def compact_cross(summary, phase):
    """Two readable views of historical cross cells; never select a router."""
    if not summary.get('overall'):
        return []
    groups = {}
    for row in summary['cross_rows']:
        keys = [('shape_turn',row['shape'],row['turn'],'all')]
        if row['turn'] == 'none':
            keys.append(('none_flow','all','none',row['flow']))
        for key in keys:
            groups.setdefault(key,[]).append(row)
    rows = []
    for (view,shape,turn,flow),cells in groups.items():
        n = sum(cell['n'] for cell in cells)
        weighted = lambda name: sum(cell['n']*cell[name] for cell in cells)/n
        rows.append(dict(phase=phase,view=view,shape=shape,turn=turn,flow=flow,n=n,
            fraction_of_all_valid=sum(cell['valid_point_fraction'] for cell in cells),
            trajectory_vs_direction_mae_gain=weighted('trajectory_vs_DIRECTION16_mae_gain'),
            trajectory_vs_own4_mae_gain=weighted('trajectory_vs_OWN4_mae_gain'),
            contribution_vs_direction=sum(cell['trajectory_vs_DIRECTION16_overall_contribution'] for cell in cells),
            contribution_vs_own4=sum(cell['trajectory_vs_OWN4_overall_contribution'] for cell in cells),
            exact_constant_history_fraction=weighted('exact_constant_history_fraction'),
            net_direction_zero_fraction=weighted('net_direction_zero_fraction')))
    return rows


def report(result):
    lines = ['# GW14 trajectory cross-strata and matched-budget checks', '',
        'Historical GW13 models replayed without changing their tables, cuts, or failed screen.',
        'Only three matched shared median tables per permitted stage are newly fitted; no neural inference or updates.',
        'Matched comparisons use the same held-out points for every arm. This common-support population can differ from the full dataset.', '',
        '| Stage | Common valid points | Common coverage % | TRAJECTORY MAE | Direction MAE | Magnitude MAE |',
        '|---|---:|---:|---:|---:|---:|']
    for name in ('audit','development'):
        stage = result['matched'].get(name)
        if stage:
            score = lambda arm: '{:.9f}'.format(stage['records'][arm]['mae']) if stage['records'][arm]['mae'] is not None else 'unavailable'
            lines.append('| {} | {} | {:.4f} | {} | {} | {} |'.format(name,
                stage['coverage']['common_valid_count'],100*stage['coverage']['common_fraction'],
                score('TRAJECTORY16'),score('DIRECTION16'),score('MAGNITUDE16')))
    for name in ('audit','development'):
        decision = result['decisions'].get(name)
        if decision:
            lines += ['', name+': '+decision['interpretation'], '', '| Fixed check | Passed |','|---|---|']
            lines += ['| {} | {} |'.format(key,value) for key,value in decision['checks'].items()]
    raw_summary = {phase:compact_cross(value,phase) for phase,value in result['raw'].items()}
    validation = raw_summary.get('validation',[])
    if validation:
        lines += ['', 'Historical raw shape x turn replay; gain is TRAJECTORY minus DIRECTION error reduction.', '',
            '| Shape | Turn | Validation valid % | Audit MAE gain | Validation MAE gain |',
            '|---|---|---:|---:|---:|']
        for row in validation:
            if row['view'] != 'shape_turn':
                continue
            earlier = next((r for r in raw_summary.get('audit',[]) if r['view']=='shape_turn'
                and r['shape']==row['shape'] and r['turn']==row['turn']),None)
            audit_gain = '{:.7f}'.format(earlier['trajectory_vs_direction_mae_gain']) if earlier else 'unavailable'
            lines.append('| {} | {} | {:.3f} | {} | {:.7f} |'.format(row['shape'],row['turn'],
                100*row['fraction_of_all_valid'],audit_gain,row['trajectory_vs_direction_mae_gain']))
        lines += ['', 'Historical raw none group by origin flow; these are descriptive replays, not selected methods.', '',
            '| Origin flow | All valid % | Exact-constant history % within cell | Gain vs OWN4 | Gain vs DIRECTION |',
            '|---|---:|---:|---:|---:|']
        for row in validation:
            if row['view']=='none_flow':
                lines.append('| {} | {:.3f} | {:.3f} | {:.7f} | {:.7f} |'.format(row['flow'],
                    100*row['fraction_of_all_valid'],100*row['exact_constant_history_fraction'],
                    row['trajectory_vs_own4_mae_gain'],row['trajectory_vs_direction_mae_gain']))
    lines += ['', 'new_matched_development_scored: '+str('development' in result['matched']),
        'new_statistical_lookup_fits: '+str(result['new_statistical_lookup_fits']),
        'original_GW13_screen_passed: False', 'architecture_training_authorized: False',
        'Raw cross tables are descriptive replays of already-seen development predictions, not new validation.',
        'Cross dimensions overlap across files; do not add contributions from different partition tables.',
        'None means no strict reversal detected after smoothing, not a guarantee of monotonic or noise-free history.',
        'Matching controls cell counts, per-cell sample counts, author-block histograms and shrinkage; it does not equalize partitions or sample identities.',
        'One fixed sampling seed is used. New lookups omit clock conditioning. These are limitations, not proof of a turning or delay mechanism.',
        'The SUM source has already seen all train labels; the internal audit holds out only the correction mapping.',
        'No test-set evaluation, statistical confirmation, automatic architecture training, or changes to the old GW13 screen are authorized.', '']
    return '\n'.join(lines)


def run(gw13_dir, probe_dir, cache_dir, data_dir, output_dir):
    gw13_dir, probe_dir, cache_dir, data_dir, output = map(lambda p:Path(p).resolve(),
        (gw13_dir,probe_dir,cache_dir,data_dir,output_dir))
    archive, last, certificate = auth.load_authenticated_run(probe_dir,repo=_export_repo_from_cache(cache_dir))
    del last
    auth.require_output_outside_sources(output,probe_dir,certificate)
    for path in (gw13_dir,cache_dir,data_dir):
        source.require(output != path and path not in output.parents and output not in path.parents,
                       'New output must be outside immutable sources and must not contain them')
    cached, _, cache_digests = io.load_cache(cache_dir,archive,certificate)
    del archive
    output.mkdir(parents=True,exist_ok=False)
    execution = {str(REPO/name):auth.sha256(REPO/name) for name in SOURCE_FILES}
    io.write_json(output/'protocol.json',dict(schema=SCHEMA,method=METHOD,execution_source=execution,
        source_directory=str(gw13_dir),source_identity_sha256=auth.SOURCE_IDENTITY_SHA256))
    try:
        trajectories, history_metadata = features.load_features(data_dir,certificate['data_identity'],cached)
        for split in ('train','val'):
            cached[split]['features'].update(trajectories[split])
        del trajectories
        reference = source.authenticate(gw13_dir,certificate,cache_digests,cached,history_metadata)
        raw = {}
        for phase in ('audit','validation'):
            print('Replaying historical cross strata: '+phase,flush=True)
            values = {arm:value for arm,value in old_model.predictions(reference['models'][phase],reference['datasets'][phase])
                      if arm in cross.ARMS}
            raw[phase] = cross.summarize(values,reference['datasets'][phase],phase=phase)
            write_cross(output,'raw_'+phase,raw[phase])
            del values
        result = dict(schema=SCHEMA,method=METHOD,source_directory=str(gw13_dir),source_digests=reference['digests'],
            raw=raw,matched={},decisions={},new_statistical_lookup_fits=0,
            original_models_or_cuts_refitted=0,new_neural_inference_calls=0,new_neural_optimizer_steps=0,
            original_GW13_failed_screen_unchanged=True,test_evaluated=False)
        fit_data = old_run.slice_data(cached['train'],0,old_run.FIT_STOP)
        stages = [('audit',reference['models']['audit'],fit_data,reference['datasets']['audit'])]
        for phase,parent,fit_data,score_data in stages:
            print('Fitting matched statistical tables: '+phase,flush=True)
            fitted = matching.fit_matched_model(parent,fit_data)
            filename = 'matched_audit_model.json' if phase == 'audit' else 'matched_full_model.json'
            matching.save_model(fitted,output/filename)
            fitted_hash = auth.sha256(output/filename)
            io.write_json(output/(phase+'_fitted_before_score.json'),dict(model_sha256=fitted_hash,
                fit_uses_scoring_labels=False,source_model_unchanged=True))
            result['new_statistical_lookup_fits'] += 3
            cells_name = 'audit_matching_cells.csv' if phase == 'audit' else 'full_matching_cells.csv'
            io.write_csv(output/cells_name,fitted['contexts'])
            evaluated = evaluate_matched(fitted,score_data,phase)
            evaluated['fit_stage_counts'] = fitted['fit_stage_counts']
            decision = matched_decision(evaluated,development=phase == 'development')
            result['matched'][phase] = evaluated
            result['decisions'][phase] = decision
            io.write_json(output/(phase+'_checks.json'),decision)
            write_cross(output,'matched_'+phase,evaluated['cross'])
            source.require(auth.sha256(output/filename) == fitted_hash,'Matched table changed while scoring')
            execution[str(output/filename)] = fitted_hash
            if phase == 'audit' and decision['new_matched_development_scoring_permitted']:
                stages.append(('development',reference['models']['validation'],cached['train'],cached['val']))
            del fitted
            gc.collect()
        source.assert_unchanged(reference)
        auth.assert_unchanged(certificate)
        io.rehash(execution)
        checks = dict(status='GW14_CHECKS_COMPLETE',decisions=result['decisions'],
            historical_source_replay_passed=True,new_statistical_lookup_fits=result['new_statistical_lookup_fits'],
            new_matched_development_scored='development' in result['matched'],
            original_GW13_failed_screen_unchanged=True,original_GW13_screen_passed=False,
            architecture_training_authorized=False,independent_confirmation_authorized=False,
            candidate_for_confirmation=False,statistical_significance_claimed=False,test_evaluated=False)
        io.write_json(output/'analysis.json',result)
        io.write_json(output/'checks.json',checks)
        io.write_csv(output/'cross_summary.csv',(row for phase,value in raw.items()
            for row in compact_cross(value,phase)))
        io.write_csv(output/'matching_summary.csv',(coverage_row(phase,stage)
            for phase,stage in result['matched'].items()))
        (output/'report.md').write_text(report(result),encoding='utf-8')
        print('GW14_CHECKS_COMPLETE: '+str(output),flush=True)
        return result
    except BaseException as error:
        io.write_json(output/'failure.json',dict(status='GW14_CHECKS_FAILED',error_type=type(error).__name__,
            error=str(error),source_artifacts_written=False,test_evaluated=False))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('gw13-dir','probe-dir','cache-dir','data-dir','output-dir'):
        parser.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args(argv)
    return run(args.gw13_dir,args.probe_dir,args.cache_dir,args.data_dir,args.output_dir)


if __name__ == '__main__':
    main()
