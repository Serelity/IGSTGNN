"""Audit the eight archived MIN forecast-fusion predictions on CPU with NumPy.

No raw release, checkpoint, model, torch, accelerator, training or inference is
used. Checks concern saved arrays and recorded certificates, not a new replay
or an independently authenticated historical archive.
"""
import argparse
from contextlib import nullcontext
from datetime import datetime
import json
from pathlib import Path
import sys
import uuid

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.chronological import analyze_min_early_event_gain as archive
from experiments.chronological import min_early_event_metrics as original_metrics
from experiments.chronological.min_forecast_fusion_gain_metrics import analyze_predictions, flat_rows

HERE = Path(__file__).resolve().parent
SCHEMA = 'min_forecast_fusion_saved_gain_analysis_v1'
STATUS = 'MIN_FORECAST_FUSION_GAIN_ANALYSIS_PASS'
SOURCE_SCHEMA = 'min_forecast_fusion_evaluation_v1'
SOURCE_STATUS = 'MIN_FORECAST_FUSION_EVALUATION_PASS'
DEFAULT_RUN = REPO/'experiments/chronological_runs/contra_min_forecast_fusion_f1_bs64_seed2025'
CASES = ('STATE_ON', 'EVENT_ON', 'EVENT_REPORT_OFF', 'EVENT_FUSION_OFF', 'EVENT_UNIFORM',
         'EVENT_SEMANTIC_UNK', 'EVENT_LOCATION_SHUFFLE', 'SOURCE_MIN')
COMPONENT_ORDER = tuple('DIF'+str(i) for i in range(5))+tuple('INH'+str(i) for i in range(5))
SOURCE_FILES = ('summary.json', 'analysis.json', 'replay.json', 'preservation.json',
                'comparability.json', 'region_definitions.json', 'paired_predictions.npz',
                'mechanisms.json', 'provenance.json')
ALIASES = {'STATE_ON': 'SOURCE_MIN', 'FUSION_STATE_ON': 'STATE_ON', 'EVENT_ON': 'EVENT_ON',
    'EVENT_OFF': 'EVENT_REPORT_OFF', 'EVENT_FUSION_OFF': 'EVENT_FUSION_OFF',
    'EVENT_UNIFORM': 'EVENT_UNIFORM', 'EVENT_SEMANTIC_UNK': 'EVENT_SEMANTIC_UNK',
    'EVENT_LOCATION_SHUFFLE': 'EVENT_LOCATION_SHUFFLE'}
ORIGINAL_COMPARISONS = {
    'EVENT_vs_STATE': ('FUSION_STATE_ON', 'EVENT_ON', 'paired_warmstart_independent_best_validation'),
    'report_content_dependency': ('EVENT_OFF', 'EVENT_ON', 'same_checkpoint_report_content_OFF_state_fusion_preserved'),
    'fusion_dependency': ('EVENT_FUSION_OFF', 'EVENT_ON', 'same_checkpoint_fusion_OFF_current_MIN_not_SOURCE_MIN'),
    'component_differentiation': ('EVENT_UNIFORM', 'EVENT_ON', 'same_checkpoint_component_mean_weights_not_independently_trained_calibration'),
    'semantic_UNK_sensitivity': ('EVENT_SEMANTIC_UNK', 'EVENT_ON', 'same_checkpoint_semantic_UNK_potential_OOD'),
    'location_shuffle_sensitivity': ('EVENT_LOCATION_SHUFFLE', 'EVENT_ON', 'same_checkpoint_full_station_shuffle_potential_OOD'),
    'EVENT_vs_SOURCE': ('STATE_ON', 'EVENT_ON', 'source_best_vs_warmstart_candidate_including_extra_training'),
    'STATE_vs_SOURCE': ('STATE_ON', 'FUSION_STATE_ON', 'source_best_vs_warmstart_candidate_including_extra_training'),
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--run-dir', type=Path, help='Latest completed forecast_fusion_evaluation_* under this run')
    source.add_argument('--evaluation-dir', type=Path, help='Explicit completed archive; no raw data required')
    parser.add_argument('--output-dir', type=Path, help='Fresh analysis directory outside the source archive')
    parser.add_argument('--flow-edges', nargs='+', type=float, default=(0., 5., 20., 50., 100., 200.))
    parser.add_argument('--response-tol', type=float, default=.001)
    args = parser.parse_args(argv)
    if args.run_dir is None and args.evaluation_dir is None:
        args.run_dir = DEFAULT_RUN
    for name in ('run_dir', 'evaluation_dir', 'output_dir'):
        if getattr(args, name) is not None:
            setattr(args, name, getattr(args, name).resolve())
    edges = np.asarray(args.flow_edges)
    archive.require(len(edges) >= 2 and np.isfinite(edges).all() and edges[0] == 0. and (np.diff(edges) > 0.).all(),
                    'Flow edges must increase from zero')
    archive.require(np.isfinite(args.response_tol) and args.response_tol >= 0., 'Response tolerance must be finite nonnegative')
    return args


def select_source(args):
    if args.evaluation_dir is not None:
        archive.require(args.evaluation_dir.is_dir(), 'Evaluation directory does not exist')
        return args.evaluation_dir
    archive.require(args.run_dir.is_dir(), 'Run directory does not exist')
    candidates = []
    for path in args.run_dir.glob('forecast_fusion_evaluation_*'):
        if not path.is_dir() or not (path/'summary.json').is_file():
            continue
        try:
            summary = archive.read_json(path/'summary.json')
        except (ValueError, UnicodeDecodeError):
            continue
        if summary.get('schema') == SOURCE_SCHEMA and summary.get('status') == SOURCE_STATUS:
            candidates.append(path)
    archive.require(candidates, 'No completed PASS forecast-fusion evaluation archive found')
    # A broken newest PASS is rejected below, never silently replaced by an older result.
    return max(candidates, key=lambda path: path.name).resolve()


def output_path(args, source):
    root = args.run_dir if args.run_dir is not None else source.parent
    output = (args.output_dir or root/('forecast_fusion_gain_analysis_'+datetime.now().strftime('%Y%m%d_%H%M%S')
                                     +'_'+uuid.uuid4().hex[:6])).resolve()
    archive.require(output != source and source not in output.parents and output not in source.parents,
                    'Output must not replace the source archive, a descendant, or an ancestor')
    for arm in ('STATE', 'EVENT', 'A', 'L', 'S'):
        candidate = source.parent/arm
        archive.require(output != candidate and candidate not in output.parents, 'Output must not be inside a checkpoint directory')
    archive.require(not output.exists(), 'Output directory already exists; refusing overwrite')
    return output


def verify_budget(summary, analysis, replay, comparable):
    archive.require(isinstance(comparable, dict), 'Missing training comparability')
    archive.assert_close(summary.get('training_comparability'), comparable, 'summary.training_comparability')
    archive.assert_close(analysis.get('training_comparability'), comparable, 'analysis.training_comparability')
    axes = {'STATE', 'EVENT'}
    completed, updates, best = (comparable.get(key) for key in ('completed_epochs', 'global_updates', 'best_stage_epochs'))
    archive.require(all(isinstance(value, dict) and value.keys() == axes for value in (completed, updates, best)), 'Training-budget arm axes differ')
    archive.require(all(type(completed[a]) is int and completed[a] > 0 and type(updates[a]) is int and updates[a] > 0
                        and type(best[a]) is int and 0 <= best[a] <= completed[a]
                        and replay[a].get('best_stage_epoch') == best[a] for a in axes), 'Invalid completed budgets or selected-stage epochs')
    equal = len(set(completed.values())) == 1 and len(set(updates.values())) == 1
    archive.require(comparable.get('same_completed_training_budget') is equal
                    and comparable.get('common_stage_epoch') == min(completed.values()), 'Archived budget flags disagree with counters')
    archive.require(summary.get('best_stage_epochs') == best and comparable.get('engineering_only') is summary['engineering_only']
                    and comparable.get('validation_rows') == summary['validation_rows'], 'Archived budget/summary axes disagree')
    archive.require(comparable.get('source_exact_resume') is False
                    and comparable.get('optimizer_and_scheduler_reset_at_stage_start') is True
                    and comparable.get('existing_MIN_retrained') is False, 'Warmstart-stage scope certificate changed')
    source_epoch = comparable.get('source_best_epoch')
    archive.require(type(source_epoch) is int and source_epoch > 0 and source_epoch == summary.get('source_best_epoch'),
                    'Source selected epoch differs')
    return 'equal_completed_stage_budget_independent_best_validation' if equal else 'unequal_completed_stage_budget_exploratory'


def recompute_original(predictions, target, valid, regions, indices, comparable):
    result = original_metrics.summarize({alias: predictions[name] for alias, name in ALIASES.items()},
        target, valid, regions, indices, comparisons=ORIGINAL_COMPARISONS)
    result['schema'], result['status'] = 'min_forecast_fusion_metrics_v1', 'MIN_FORECAST_FUSION_ANALYSIS_PASS'
    result['cases'] = {ALIASES[name]: row for name, row in result['cases'].items()}
    fixed = result['fixed_hard_windows']
    fixed['selector'] = 'SOURCE_MIN'
    fixed['cases'] = {ALIASES[name]: row for name, row in fixed['cases'].items()}
    for row in result['comparisons'].values():
        row['control'], row['candidate'] = ALIASES[row['control']], ALIASES[row['candidate']]
    result['bias_reference'] = 'SOURCE_MIN_for_all_comparisons'
    result['forecast_scope'] = 'one_fusion_module_no_repeated_forecast_component_ablation'
    result['training_comparability'] = comparable
    result['comparisons']['EVENT_vs_STATE']['scope'] = ('paired_warmstart_equal_completed_stage_budget_independent_best'
        if comparable['same_completed_training_budget'] else 'unequal_completed_stage_budget_exploratory_independent_best')
    return result


def verify_mechanisms(mechanisms, predictions, regions, donors, indices, rows, nodes, partitions, identity):
    archive.require(mechanisms.get('component_order') == list(COMPONENT_ORDER), 'Archived forecast component order differs')
    weights = mechanisms.get('fusion_weights', {})
    measured = set(CASES)-{'SOURCE_MIN'}
    archive.require(isinstance(weights, dict) and weights.keys() == measured
                    and mechanisms.get('cases', {}).keys() == measured, 'Archived mechanism case axes differ')
    bound = identity['protocol']['fusion_spec']['max_delta']
    archive.require(isinstance(bound, (float, int)) and 0 < bound < 1, 'Invalid archived fusion bound')
    for case, row in weights.items():
        arrays = {key: np.asarray(row.get(key), np.float64) for key in ('mean', 'absolute_delta_from_one_mean', 'minimum', 'maximum')}
        archive.require(all(value.shape == (12, 10) and np.isfinite(value).all() for value in arrays.values())
                        and row.get('count_per_horizon_component') == rows*nodes, 'Fusion weight axes/count differ: '+case)
        archive.require((arrays['minimum'] <= arrays['mean']+1e-12).all()
                        and (arrays['maximum'] >= arrays['mean']-1e-12).all()
                        and (arrays['minimum'] >= 1-bound-2e-7).all() and (arrays['maximum'] <= 1+bound+2e-7).all()
                        and (arrays['absolute_delta_from_one_mean']+1e-12 >= np.abs(arrays['mean']-1)).all(), 'Invalid fusion weight bounds/mean: '+case)
        for key, shape in (('component_increment_abs_mean_by_horizon_component', (12, 10)),
                           ('weighted_hidden_change_abs_mean_by_horizon', (12,))):
            value = np.asarray(row.get(key), np.float64)
            archive.require(value.shape == shape and np.isfinite(value).all() and (value >= 0).all(), 'Invalid archived hidden increment diagnostics')
        if case == 'EVENT_FUSION_OFF':
            archive.require(np.array_equal(arrays['mean'], np.ones((12, 10)))
                            and np.array_equal(arrays['minimum'], arrays['mean'])
                            and np.array_equal(arrays['maximum'], arrays['mean'])
                            and not arrays['absolute_delta_from_one_mean'].any(), 'Fusion OFF weights did not restore one')
        if case == 'EVENT_UNIFORM':
            archive.require(all(np.array_equal(value, np.broadcast_to(value[:, :1], value.shape)) for value in arrays.values()),
                            'Uniform intervention does not share component weights')
    direct = np.broadcast_to(regions['direct_report_nodes'], (rows, nodes))
    outside = ~direct[:, None, :]
    archive.require(np.array_equal(predictions['EVENT_ON'][np.broadcast_to(outside, predictions['EVENT_ON'].shape)],
                                   predictions['EVENT_REPORT_OFF'][np.broadcast_to(outside, predictions['EVENT_ON'].shape)]),
                    'Report content changed prediction outside original supported nodes')
    shuffle = mechanisms.get('location_shuffle', {})
    seed = shuffle.get('seed')
    archive.require(type(seed) is int and seed >= 0, 'Invalid recorded shuffle seed')
    expected_donors = np.empty_like(donors)
    for row, sample in enumerate(indices):
        order = np.random.default_rng(np.random.SeedSequence([seed, int(sample)])).permutation(nodes)
        expected_donors[row, order] = np.roll(order, 1)
    archive.require(np.array_equal(donors, expected_donors), 'Recorded station donors differ from seed/sample-index cycle')
    expected_changed = direct != np.take_along_axis(direct, donors, axis=1)
    archive.require(shuffle.get('schema') == 'full_station_relation_cycle_v1' and shuffle.get('uses_targets') is False
                    and shuffle.get('zero_relation_rows_permuted') is True and shuffle.get('fixed_semantic_categories') is True
                    and shuffle.get('rows') == rows
                    and shuffle.get('moved_vector_positions') == int((donors != np.arange(nodes)).sum())
                    and shuffle.get('support_changed_positions') == int(expected_changed.sum())
                    and shuffle.get('support_changed_rows') == int(expected_changed.any(1).sum()), 'Archived shuffle coverage differs from donors/support')
    on, off = weights['EVENT_ON'], weights['EVENT_REPORT_OFF']
    change = np.asarray(on['mean'])-np.asarray(off['mean'])
    return dict(scope='difference_of_archived_global_mean_weights_not_mean_absolute_pointwise_weight_difference_or_importance',
                component_order=list(COMPONENT_ORDER), on_minus_report_off_mean=change.tolist(),
                max_absolute_global_mean_difference=float(np.abs(change).max()),
                averaged_signed_global_mean_difference=float(change.mean()),
                sample_node_count_per_horizon_component=rows*nodes,
                raw_per_node_weights_archived=False, source_weights_unmeasured=True)


def load_and_verify(source):
    archive.require(all((source/name).is_file() for name in SOURCE_FILES), 'Missing required forecast-fusion archive artifacts')
    archive.require(not (source/'failure.json').exists(), 'Evaluation archive contains a failure record')
    summary, analysis, replay, preservation, definitions, comparable, mechanisms, provenance = [archive.read_json(source/name)
        for name in ('summary.json', 'analysis.json', 'replay.json', 'preservation.json', 'region_definitions.json',
                     'comparability.json', 'mechanisms.json', 'provenance.json')]
    archive.require(summary.get('schema') == SOURCE_SCHEMA and summary.get('status') == SOURCE_STATUS, 'Unsupported evaluation schema or non-PASS status')
    archive.require(analysis.get('schema') == 'min_forecast_fusion_metrics_v1'
                    and analysis.get('status') == 'MIN_FORECAST_FUSION_ANALYSIS_PASS', 'Unsupported archived analysis schema/status')
    archive.require(summary.get('replay_passed') is True and summary.get('on_exactly_restored') is True
                    and summary.get('source_artifacts_preserved') is True and summary.get('optimizer_updates') == 0
                    and summary.get('test_evaluated') is False and type(summary.get('engineering_only')) is bool,
                    'Incomplete recorded replay/protection/engineering certificate')
    archive.require(isinstance(replay, dict) and replay.keys() == {'STATE', 'EVENT'}, 'Replay arm axes differ')
    for arm, row in replay.items():
        archive.require(row.get('status') == 'BEST_ON_REPLAY_PASS', 'Recorded best ON replay did not pass: '+arm)
    scope = verify_budget(summary, analysis, replay, comparable)
    identity = provenance.get('training_identity')
    archive.require(isinstance(identity, dict) and identity.get('schema') == 'min_forecast_fusion_warmstart_run_v1'
                    and identity.get('engineering_check') is summary['engineering_only'], 'Missing/invalid archived training identity')
    archive.require(identity.get('arm_modes') == {'STATE': 'state', 'EVENT': 'event'}
                    and identity.get('source_exact_resume') is False and identity.get('source_optimizer_restored') is False
                    and identity.get('source_scheduler_restored') is False and identity.get('full_horizon_loss_every_update') is True,
                    'Archived architecture/warmstart identity differs')
    source_info = identity.get('source', {})
    archive.require(source_info.get('best_epoch') == summary['source_best_epoch']
                    and type(source_info.get('completed_epoch')) is int and source_info['completed_epoch'] >= source_info['best_epoch']
                    and archive.is_hash(source_info.get('best_model_sha256'))
                    and isinstance(source_info.get('files_sha256'), dict) and source_info['files_sha256']
                    and all(archive.is_hash(v) for v in source_info['files_sha256'].values()), 'Invalid archived MIN source certificate')
    files, models = preservation.get('files', {}), preservation.get('models', {})
    archive.require(files.get('status') == 'SOURCE_ARTIFACTS_PRESERVED' and isinstance(files.get('before'), dict)
                    and files['before'] and files.get('after') == files['before'] and files.get('files') == len(files['before'])
                    and all(archive.is_hash(value) for value in files['before'].values()), 'Archived file preservation failed')
    archive.require(isinstance(models, dict) and models.keys() == replay.keys()
                    and all(row.get('status') == 'MODEL_STATE_PRESERVED' and row.get('mode_gradients_hooks_graph_rng_preserved') is True
                            and archive.is_hash(row.get('model_state_sha256')) for row in models.values()), 'Archived model preservation failed')
    cases = summary.get('cases')
    archive.require(isinstance(cases, list) and len(cases) == len(set(cases)) and set(cases) == set(CASES), 'Archived prediction cases differ')
    mapping = definitions.get('masks', {})
    archive.require(isinstance(mapping, dict) and mapping and len(set(mapping.values())) == len(mapping)
                    and all(isinstance(name, str) and name and isinstance(key, str) and key.startswith('region_') for name, key in mapping.items()),
                    'Invalid region definitions mapping')
    with np.load(source/'paired_predictions.npz', allow_pickle=False) as saved:
        metadata = {'target', 'valid', 'sample_indices', 'station_ids', 'location_donor_station_indices'}
        archive.require(set(saved.files) == set(CASES) | set(mapping.values()) | metadata, 'NPZ archive fields disagree with cases/regions')
        predictions = {case: saved[case] for case in CASES}
        target, valid, indices, stations, donors = [saved[name] for name in ('target', 'valid', 'sample_indices', 'station_ids', 'location_donor_station_indices')]
        regions = {name: saved[key] for name, key in mapping.items()}
    archive.require(target.ndim == 3 and target.shape[1] == 12 and target.shape[0] > 0 and target.shape[2] > 0
                    and target.dtype.kind in 'fiu' and np.isfinite(target).all(), 'Invalid target numeric axes/values')
    rows, horizons, nodes = target.shape
    archive.require(summary.get('validation_rows') == analysis.get('validation_rows') == rows
                    and analysis.get('horizons') == 12 and analysis.get('stations') == nodes, 'Archived sample/station/horizon counts differ')
    archive.require(valid.shape == target.shape and valid.dtype == np.bool_ and valid.any(), 'Invalid explicit original validity mask')
    archive.require(indices.shape == (rows,) and indices.dtype.kind in 'iu' and (indices >= 0).all()
                    and np.unique(indices).size == rows and (rows == 1 or (np.diff(indices) == 1).all()), 'Invalid original sample indices')
    archive.require(stations.shape == (nodes,) and stations.dtype.kind in 'iu' and np.unique(stations).size == nodes, 'Invalid unique station axis')
    archive.require(donors.shape == (rows, nodes) and donors.dtype.kind in 'iu'
                    and np.array_equal(np.sort(donors, axis=1), np.broadcast_to(np.arange(nodes), donors.shape)), 'Invalid within-row station donor permutations')
    archive.require(all(value.shape == target.shape and value.dtype.kind in 'fiu' and np.isfinite(value).all() for value in predictions.values()),
                    'Invalid paired prediction numeric axes/values')
    normalized = {}
    for name, mask in regions.items():
        archive.require(mask.dtype == np.bool_ and mask.shape in ((nodes,), (rows, nodes)), 'Invalid Boolean region axes: '+name)
        normalized[name] = np.broadcast_to(mask, (rows, nodes))
    partition = ('direct_report_nodes', 'potential_only', 'remaining_nodes')
    archive.require('all_nodes' in normalized and normalized['all_nodes'].all() and all(name in normalized for name in partition)
                    and (sum(normalized[name].astype(np.uint8) for name in partition) == 1).all(), 'Report regions are not disjoint/exhaustive')
    if 'candidate_potential' in normalized:
        archive.require(np.array_equal(normalized['candidate_potential'], normalized[partition[0]] | normalized[partition[1]]), 'Candidate region disagrees with partitions')
    protocol, batch = identity.get('protocol', {}), identity.get('batch_size')
    archive.require(type(batch) is int and batch > 0 and type(protocol.get('train_samples')) is int
                    and protocol.get('stations') == nodes and indices[0] == protocol['train_samples']
                    and (rows <= protocol.get('val_samples', -1) if summary['engineering_only'] else rows == protocol.get('val_samples')),
                    'Saved release identity split/station/batch axes differ')
    partitions = [min(batch, rows-start) for start in range(0, rows, batch)]
    masks = []
    for arm, row in replay.items():
        archive.require(row.get('batch_partition') == partitions, 'Recorded original validation batch partitions differ')
        archive.require(isinstance(row.get('mask_value'), (float, int)) and np.isfinite(row['mask_value']), 'Invalid archived mask value')
        masks.append(row['mask_value'])
        for key in ('expected', 'actual'):
            archive.require(isinstance(row.get(key), list) and len(row[key]) == 3 and np.isfinite(row[key]).all(), 'Invalid recorded replay metric vector')
        archive.require(isinstance(row.get('atol'), (int, float)) and 0 <= row['atol'] <= 1e-5
                        and isinstance(row.get('rtol'), (int, float)) and 0 <= row['rtol'] <= 1e-6
                        and np.allclose(row['actual'], row['expected'], atol=row['atol'], rtol=row['rtol']), 'Replay tolerance or selected metric certificate changed')
    native_mask = float(target.min()) if target.min() < 1. else 0.
    archive.require(len(set(masks)) == 1 and masks[0] == native_mask and np.array_equal(valid, target != masks[0])
                    and (target[valid] >= 0).all(), 'Saved validity differs from original native mask convention')
    calls = summary.get('new_inference_calls', {})
    archive.require(calls.keys() == set(CASES) and calls.get('SOURCE_MIN') == 0
                    and calls.get('STATE_ON') == len(partitions) and calls.get('EVENT_ON') == 6*len(partitions)
                    and all(calls.get(case) == len(partitions) for case in CASES[2:-1]), 'Recorded original inference/batch counts differ')
    # Recompute the original entire analysis: all eight cases, all regions,
    # every archived comparison/stratum and the original SOURCE hard-set IDs.
    recomputed = recompute_original(predictions, target, valid, normalized, indices, comparable)
    archive.assert_close(recomputed, analysis, 'archived_analysis_all_cases_regions_comparisons')
    archive.assert_close(summary.get('all_nodes'), {case: row['metrics']['all_nodes'] for case, row in recomputed['cases'].items()}, 'summary.all_nodes')
    archive.assert_close(summary.get('comparisons'), {name: row['metrics']['all_nodes'] for name, row in recomputed['comparisons'].items()}, 'summary.comparisons')
    for arm, row in replay.items():
        metrics = recomputed['cases'][arm+'_ON']['metrics']['all_nodes']
        actual = [metrics[key] for key in ('mae_pooled', 'mape_pooled', 'rmse_pooled')]
        archive.require(all(np.allclose(actual, row[key], atol=row['atol'], rtol=row['rtol']) for key in ('expected', 'actual')),
                        'Saved ON arrays disagree with selected metric certificate: '+arm)
        if comparable['best_stage_epochs'][arm] == 0:
            archive.require(np.array_equal(predictions[arm+'_ON'], predictions['SOURCE_MIN']), 'Selected stage zero is not the source prediction')
    source_metric = recomputed['cases']['SOURCE_MIN']['metrics']['all_nodes']
    actual_source = [source_metric[key] for key in ('mae_pooled', 'mape_pooled', 'rmse_pooled')]
    archive.require(all(isinstance(source_info.get(key), list) and len(source_info[key]) == 3
                        and np.allclose(actual_source, source_info[key], atol=1e-5, rtol=1e-6) for key in ('best_metrics', 'reference_metrics')),
                    'SOURCE_MIN arrays disagree with archived source-best certificate')
    hard = np.asarray(recomputed['fixed_hard_windows']['sample_indices'], dtype=np.int64)
    archive.require(hard.size > 0 and np.unique(hard).size == len(hard) and np.isin(hard, indices).all(), 'Invalid original fixed hard-window IDs')
    weight_difference = verify_mechanisms(mechanisms, predictions, normalized, donors, indices, rows, nodes, partitions, identity)
    return dict(summary=summary, analysis=analysis, replay=replay, comparable=comparable, identity=identity,
                predictions=predictions, target=target, valid=valid, indices=indices, stations=stations,
                regions=normalized, fixed_hard_ids=hard, mechanisms=mechanisms, weight_difference=weight_difference,
                source_scope=scope, axis_scope='archived_release_identity_and_arrays_verified_without_raw_data_read')


def write_tables(output, result):
    rows = list(flat_rows(result))
    selectors = {
        'all_nodes.csv': lambda r: r.get('group') == 'all_nodes' and r.get('horizon') == 0,
        'flow.csv': lambda r: r.get('family') == 'flow' and r.get('horizon') == 0,
        'current_bias.csv': lambda r: r.get('family') == 'current_MIN_bias' and r.get('horizon') == 0,
        'source_bias.csv': lambda r: r.get('family') == 'source_MIN_bias' and r.get('horizon') == 0,
        'regions.csv': lambda r: r.get('family') in ('region', 'report_partition') and r.get('horizon') == 0,
        'roads.csv': lambda r: r.get('family') == 'road' and r.get('horizon') == 0,
        'cross.csv': lambda r: r.get('family') == 'flow_current_MIN_bias_report_partition' and r.get('horizon') == 0,
        'horizons.csv': lambda r: r.get('group') == 'all_nodes' and r.get('horizon', 0) > 0,
        'group_horizons.csv': lambda r: r.get('horizon', 0) > 0,
        'fixed_hard.csv': lambda r: r.get('group') == 'fixed_hard_windows' and r.get('horizon') == 0,
    }
    for name, selector in selectors.items():
        archive.write_csv(output/name, (row for row in rows if selector(row)))
    archive.write_csv(output/'groups.csv', rows)
    directions = []
    for comparison, item in result.get('directions', {}).items():
        for group, row in item['groups'].items():
            for direction, values in row['directions'].items():
                directions.append(dict(comparison=comparison, group=group, direction=direction, **values))
    archive.write_csv(output/'directions.csv', directions)
    overlap = []
    for name, item in result.get('overlap', {}).items():
        for quadrant, values in item.get('quadrants', {}).items():
            overlap.append(dict(decomposition=name, quadrant=quadrant, total=item['total'],
                                first=item['first'], second=item['second'], **values))
    archive.write_csv(output/'overlap.csv', overlap)


def write_report(output, result, info):
    def fmt(value):
        return '不可用' if value is None else f'{value:.9f}'
    lines = ['# MIN-F1 已有预测：论文机制分析', '',
        '只读已有八组验证预测；没有加载原始数据、检查点或模型，没有新增推理、训练或测试集评价。', '',
        '**论文主基线为原始 IGSTGNN。当前归档没有它的预测；SOURCE_MIN 只是中间参照，本报告不能证明超过原 IGSTGNN。**', '',
        ('本次为四窗口等小规模工程检查，不能解释为正式预测效果。' if info['summary']['engineering_only'] else
         '本报告为已用于设计和选模的验证集上的探索性诊断；窗口及预测点不视为独立事故重复。'), '',
        'EVENT_REPORT_OFF 保留当前状态融合；EVENT_FUSION_OFF 保留 EVENT 联合训练后的当前 MIN 主干，两者均不是历史 SOURCE_MIN。',
        'SOURCE_MIN 与候选的差异包含额外微调、优化策略及联合训练差异，不能因果归因为融合训练或事故训练收益。', '',
        '## 八种状态', '', '| 状态 | MAE pooled | MAE horizon macro | RMSE pooled | MAPE (%) |', '|---|---:|---:|---:|---:|']
    for case in CASES:
        m = info['analysis']['cases'][case]['metrics']['all_nodes']
        lines.append('| '+case+' | '+' | '.join(fmt(m[key]) for key in ('mae_pooled', 'mae_macro', 'rmse_pooled'))
                     +' | '+fmt(None if m['mape_pooled'] is None else 100*m['mape_pooled'])+' |')
    lines += ['', '## 同点收益与模块取舍', '',
        '收益为对照绝对误差减候选绝对误差；pooled按有效点等权，macro按十二时距等权。', '',
        '| 比较 | 对照 | 候选 | MAE收益 pooled | MAE收益 macro |', '|---|---|---|---:|---:|']
    for name, item in result['comparisons'].items():
        row = item['groups']['all_nodes']
        lines.append('| '+name+' | '+item.get('control', '')+' | '+item.get('candidate', '')+' | '
                     +fmt(row.get('mae_gain_pooled'))+' | '+fmt(row.get('mae_gain_macro'))+' |')
    key_comparisons = ('fusion_dependency', 'common_scaling_dependency', 'component_differentiation', 'report_content_dependency')
    comparison_labels = dict(fusion_dependency='完整融合', common_scaling_dependency='共同缩放',
        component_differentiation='分量差异化', report_content_dependency='报告内容')
    lines += ['', '## 四种作用的正负抵消与预测响应', '',
        '正收益和负收益均为逐点MAE变化量的非负部分；净收益为正减负。预测有符号变化为候选减对照，正数表示上调。占比均为0到1的比例；响应占比使用本次保存的response_tol阈值。', '',
        '| 比较 | 正收益pooled | 负收益pooled | 抵消比例 | 预测有符号变化均值 | 预测绝对变化均值 | 上调占比 | 下调占比 | 响应占比 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for comparison in key_comparisons:
        row = result['comparisons'][comparison]['groups']['all_nodes']
        change = row['prediction_change']
        values = [row['gain_positive_pooled'], row['gain_negative_pooled'], row['cancellation_pooled'],
            change['signed_mean_candidate_minus_control_pooled'], change['absolute_mean_pooled'],
            change['up_fraction'], change['down_fraction'], change['response_fraction']]
        lines.append('| '+comparison_labels[comparison]+' | '+' | '.join(fmt(value) for value in values)+' |')
    lines += ['', '## 当前融合与报告的修正方向', '',
        '每种比较按自己的对照预测判定方向；远离真值与越过后恶化分别统计。', '',
        '| 比较 | 方向 | 有效点数 | 组内收益 pooled | 全网MAE贡献 |', '|---|---|---:|---:|---:|']
    for comparison, item in result.get('directions', {}).items():
        for direction, row in item['groups']['all_nodes']['directions'].items():
            lines.append('| '+comparison+' | '+direction+' | '+str(row['valid_count'])+' | '
                         +fmt(row.get('gain_pooled'))+' | '+fmt(row.get('global_mae_contribution_pooled'))+' |')
    lines += ['', '## 当前MIN偏差与流量', '',
        '| 比较 | 分组 | 有效点数 | MAE收益 pooled | MAPE收益（百分点） |', '|---|---|---:|---:|---:|']
    for comparison in ('fusion_dependency', 'common_scaling_dependency', 'component_differentiation', 'report_content_dependency'):
        if comparison not in result['comparisons']:
            continue
        for group, row in result['comparisons'][comparison]['groups'].items():
            if result['group_definitions'][group]['family'] in ('flow', 'current_MIN_bias'):
                mape = row.get('mape_gain_pooled')
                lines.append('| '+comparison+' | '+group+' | '+str(row['valid_count'])+' | '+fmt(row.get('mae_gain_pooled'))
                             +' | '+fmt(None if mape is None else 100*mape)+' |')
    lines += ['', '## 道路、事故分区与预测时距并列', '',
        '以下均列各比较的组内MAE收益pooled；正数表示开启/差异化候选更好。', '']
    for title, groups in (
        ('道路', [name for name, definition in result['group_definitions'].items() if definition['family'] == 'road']),
        ('事故关联三分区', ['direct_report_nodes', 'potential_only', 'remaining_nodes'])):
        lines += ['### '+title, '', '| 分组 | 有效点数 | 完整融合收益 | 共同缩放收益 | 分量差异化收益 | 报告内容收益 |', '|---|---:|---:|---:|---:|---:|']
        for group in groups:
            count = result['comparisons']['fusion_dependency']['groups'][group]['valid_count']
            gains = [result['comparisons'][name]['groups'][group]['mae_gain_pooled'] for name in key_comparisons]
            lines.append('| '+group+' | '+str(count)+' | '+' | '.join(fmt(value) for value in gains)+' |')
        lines.append('')
    lines += ['### 逐预测时距', '', '| 步 | 有效点数 | 完整融合收益 | 共同缩放收益 | 分量差异化收益 | 报告内容收益 |', '|---|---:|---:|---:|---:|---:|']
    for h in range(12):
        overall = result['comparisons']['fusion_dependency']['groups']['all_nodes']
        gains = [result['comparisons'][name]['groups']['all_nodes']['per_horizon_mae_gain'][h] for name in key_comparisons]
        lines.append('| '+str(h+1)+' | '+str(overall['valid_count_per_horizon'][h])+' | '+' | '.join(fmt(value) for value in gains)+' |')
    lines += ['', '## 最大正负交叉贡献', '',
        '按共同全网有效点数分母的MAE贡献排序；每个比较分别列正、负前三个非空组，不能直接相加各组局部均值。', '',
        '| 比较 | 方向 | 流量×当前MIN偏差×事故分区 | 点数 | 组内收益 | 全网贡献 |', '|---|---|---|---:|---:|---:|']
    for comparison in key_comparisons:
        groups = [(group, row) for group, row in result['comparisons'][comparison]['groups'].items()
            if result['group_definitions'][group]['family'] == 'flow_current_MIN_bias_report_partition' and row['valid_count'] > 0]
        for direction, sign in (('正', 1), ('负', -1)):
            ranked = sorted(((group, row) for group, row in groups if sign*row['global_mae_contribution_pooled'] > 0),
                            key=lambda item: sign*item[1]['global_mae_contribution_pooled'], reverse=True)[:3]
            for group, row in ranked:
                lines.append('| '+comparison+' | '+direction+' | '+group+' | '+str(row['valid_count'])+' | '
                             +fmt(row['mae_gain_pooled'])+' | '+fmt(row['global_mae_contribution_pooled'])+' |')
    fixed = result['fixed_hard_windows']
    lines += ['', '## 原评估固定困难窗口', '',
        '保存ID数：'+str(fixed['selected_count'])+'；完整窗口数：'+str(fixed['complete_window_count'])+'。',
        'point_metrics汇总所有保存ID中的有效预测点；窗口等权只使用十二时距都有效的完整窗口。因此两者可能不同，不重新挑选困难集。', '',
        '| 比较 | 窗口等权收益 | 保存ID逐点收益pooled |', '|---|---:|---:|']
    for comparison in key_comparisons:
        row = fixed['comparisons'][comparison]
        lines.append('| '+comparison+' | '+fmt(row['window_equal_weight_mean_gain'])+' | '+fmt(row['point_metrics']['mae_gain_pooled'])+' |')
    lines += ['', '## 分解证据', '',
        '以下为同一预测点上的误差差值恒等式；正负部分及RMSE不能套用线性相加，也不计算事故净收益占总体收益的因果比例。', '',
        '```json', json.dumps(result.get('decompositions', {}), ensure_ascii=False, indent=2, allow_nan=False), '```', '',
        '## 后续改进应回答的问题', '',
        '1. 当前融合在什么流量、当前 MIN 高估/低估位置、道路和预测时距上损害 MAE/MAPE？高低估以 EVENT_FUSION_OFF 定义，SOURCE_MIN 偏差另列。',
        '2. 均值权重与分量差异化是否分别造成损害与补偿？UNIFORM 是同检查点敏感性，不是独立训练的校准对照。',
        '3. 事故内容在原直接支持区域的正负收益、响应方向及抵消程度如何？小全网净收益不能直接证明不读取事故。',
        '4. 融合OFF的负消融须如实保留；不得仅因验证集最好就自动替换论文主模型，删除后重新训练效果仍未知。', '',
        '## 权重描述', '',
        'ON与REPORT_OFF全局均值权重最大绝对差：'+fmt(info['weight_difference']['max_absolute_global_mean_difference'])+'。',
        '该值是两组均值之差，不是逐点权重绝对差均值；不能解释为重要性、精确误差贡献或真实事故容量。', '',
        '## 证据与可比性边界', '',
        '最佳阶段：'+str(info['summary']['best_stage_epochs'])+'；完成轮次：'+str(info['comparable']['completed_epochs'])+'。',
        '固定困难窗口直接复用原 SOURCE_MIN 的样本ID；本次不更换集合。所有真值分层和方向标签仅用于事后诊断，不能作为在线输入。',
        '事故关联区域及弱连通候选包络不是认证事故影响范围；全站重排与语义UNK可能产生分布外输入。',
        '已核对八组预测的全区域指标、比较、摘要、原掩码/轴、选模与原批次证书。此为归档内部一致性，不是新增检查点回放或历史独立签名认证。',
        '原训练顺序未重新读取或独立认证；预算依据保存的评估比较证书。缺少同预算纯MIN微调控制时，不能单独归因结构改进。',
        '制品：analysis.json、all_nodes.csv、flow.csv、current_bias.csv、source_bias.csv、regions.csv、roads.csv、cross.csv、horizons.csv、group_horizons.csv、overlap.csv、directions.csv、fixed_hard.csv、mechanisms.json、paper_evidence_manifest.json、provenance.json、preservation.json。', '']
    (output/'report.md').write_text('\n'.join(lines), encoding='utf-8')


def analyze_run(args):
    source = select_source(args)
    output = output_path(args, source)
    lock_root = args.run_dir if args.run_dir is not None else source.parent if (source.parent/'active.lock').is_file() else None
    with archive.readonly_run_lock(lock_root) if lock_root is not None else nullcontext(None) as lock_state:
        def protected_snapshot():
            paths = {str(path.relative_to(source)): path for path in source.rglob('*') if path.is_file()}
            archive.require(all(source in path.resolve().parents for path in paths.values()), 'Source archive contains an external file link')
            values = archive.snapshot(paths)
            if lock_state is not None:
                values['<existing_source_run_lock>'] = archive.locked_stream_hash(lock_state[1])
            return values
        before = protected_snapshot()
        output.mkdir(parents=True)
        print('Source evaluation: '+str(source), flush=True)
        print('Analysis directory: '+str(output), flush=True)
        try:
            info = load_and_verify(source)
            result = analyze_predictions(info['predictions'], info['target'], info['valid'], info['regions'],
                info['indices'], info['stations'], fixed_hard_sample_ids=info['fixed_hard_ids'],
                flow_edges=tuple(args.flow_edges), response_tol=args.response_tol)
            result.update(source_evaluation=str(source), source_scope=info['source_scope'], engineering_only=info['summary']['engineering_only'],
                source_best_stage_epochs=info['summary']['best_stage_epochs'], source_best_epoch=info['summary']['source_best_epoch'],
                training_comparability=info['comparable'], archive_axis_scope=info['axis_scope'],
                new_inference_calls=0, optimizer_updates=0, test_evaluated=False,
                source_certificate_check='saved_arrays_and_recorded_certificates_not_new_checkpoint_replay')
            archive.write_json(output/'analysis.json', result)
            write_tables(output, result)
            write_report(output, result, info)
            archive.write_json(output/'mechanisms.json', dict(weight_difference=info['weight_difference'],
                source_mechanism_schema='archived_global_weight_statistics_no_new_model_calls',
                component_order=list(COMPONENT_ORDER), fusion_weights=info['mechanisms']['fusion_weights']))
            archive.write_json(output/'paper_evidence_manifest.json', dict(
                primary_baseline=dict(name='original_IGSTGNN', predictions_in_this_archive=False, evaluated_in_this_analysis=False),
                intermediate_reference=dict(name='SOURCE_MIN', source_best_epoch=info['summary']['source_best_epoch']),
                evaluated_cases=list(CASES), source_release_identity=info['identity']['package'],
                source_graph_identity=info['identity']['graph_inputs_sha256'],
                required_before_primary_baseline_gain_claim=['same_split_original_IGSTGNN_predictions_and_metrics',
                    'matched_training_protocol_and_budget', 'validation_selection_and_untouched_test_policy'],
                source_MIN_not_relabelled_as_original_IGSTGNN=True, evidence_scope='exploratory_saved_validation_predictions'))
            code_paths = [Path(__file__).resolve(), HERE/'min_forecast_fusion_gain_metrics.py',
                          HERE/'analyze_min_early_event_gain.py', HERE/'min_early_event_metrics.py',
                          HERE/'min_early_event_gain_metrics.py', HERE/'min_early_event_direction_metrics.py',
                          HERE/'min_event_budget_probe_metrics.py']
            archive.write_json(output/'provenance.json', dict(schema=SCHEMA, source_evaluation=str(source), source_files_sha256_before=before,
                analysis_source_sha256={str(path.relative_to(REPO)): archive.sha256(path) for path in code_paths},
                original_fixed_hard_sample_ids=info['fixed_hard_ids'].tolist(), flow_edges=list(args.flow_edges), response_tol=args.response_tol,
                imported_runtime='numpy_and_standard_library_only', raw_data_read=False, checkpoint_loaded=False,
                new_checkpoint_replay=False, historical_signature_verified=False, new_inference_calls=0, optimizer_updates=0, test_evaluated=False))
            after = protected_snapshot()
            archive.require(before == after, 'Source archive changed during read-only analysis')
            archive.write_json(output/'preservation.json', dict(status='SOURCE_ARTIFACTS_PRESERVED', source_artifacts_preserved=True,
                files=len(before), before=before, after=after, scope='this_analysis_read_interval_byte_hashes_not_historical_signature'))
            summary = dict(schema=SCHEMA, status=STATUS, source_evaluation=str(source), validation_rows=len(info['indices']),
                engineering_only=info['summary']['engineering_only'], source_best_stage_epochs=info['summary']['best_stage_epochs'],
                source_best_epoch=info['summary']['source_best_epoch'], source_scope=info['source_scope'],
                all_nodes={name: item['groups']['all_nodes'] for name, item in result['comparisons'].items()},
                primary_baseline='original_IGSTGNN_not_present_in_source_archive',
                new_inference_calls=0, optimizer_updates=0, test_evaluated=False, source_artifacts_preserved=True,
                source_certificate_check='archived_internal_consistency_not_new_checkpoint_replay', output_dir=str(output))
            archive.write_json(output/'summary.json', summary)
            return summary
        except BaseException as error:
            after = protected_snapshot()
            archive.write_json(output/'preservation.json', dict(status='SOURCE_ARTIFACTS_PRESERVED' if before == after else 'SOURCE_ARTIFACTS_CHANGED',
                source_artifacts_preserved=before == after, before=before, after=after, analysis_completed=False))
            archive.write_json(output/'failure.json', dict(schema=SCHEMA, status='MIN_FORECAST_FUSION_GAIN_ANALYSIS_FAILED',
                error_type=type(error).__name__, error=str(error), source_evaluation=str(source), new_inference_calls=0, optimizer_updates=0, test_evaluated=False))
            raise


def main(argv=None):
    result = analyze_run(parse_args(argv))
    concise = {key: value for key, value in result.items() if key != 'all_nodes'}
    concise['all_nodes'] = {name: {key: row.get(key) for key in ('mae_gain_pooled', 'mae_gain_macro')}
                            for name, row in result['all_nodes'].items()}
    print(json.dumps(concise, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
    return result


if __name__ == '__main__':
    main()
