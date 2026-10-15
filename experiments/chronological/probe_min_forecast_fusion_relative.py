"""Replay the saved EVENT best once, then isolate relative component weights.

The eight original cases remain immutable cached references. Only EVENT ON and
two temporary relative-weight interventions execute. No optimizer, training,
test evaluation, source-MIN forward or STATE forward is used.
"""
import argparse
from datetime import datetime
import gc
import json
import logging
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import uuid

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.chronological import analyze_min_forecast_fusion_gain as archived
from experiments.chronological import evaluate_min_forecast_fusion as evaluation
from experiments.chronological import min_forecast_fusion_relative_metrics as metrics
from experiments.chronological import probe_min_event_budget as probe
from experiments.chronological import train_min_forecast_fusion as training
from experiments.chronological.smoke import sha256
from src.models.incident_spatial import require
from src.models.incident_spatial_fusion_relative import relative_fusion
from src.models.incident_spatial_residual import tensor_state_hash
from src.utils.igstgnn_published_events import EVENT_KEYS, encode_event
from src.utils.metrics import compute_all_metrics

SCHEMA = 'min_forecast_fusion_relative_probe_v1'
STATUS = 'MIN_FORECAST_FUSION_RELATIVE_PROBE_PASS'
RELATIVE_CASES = {'EVENT_RELATIVE_EXACT': 'exact', 'EVENT_RELATIVE_BOUNDED': 'bounded'}
CASES = evaluation.CASES+tuple(RELATIVE_CASES)
CODE_SOURCES = (
    'experiments/chronological/probe_min_forecast_fusion_relative.py',
    'src/models/incident_spatial_fusion_relative.py',
    'experiments/chronological/min_forecast_fusion_relative_metrics.py',
    'experiments/chronological/evaluate_min_forecast_fusion.py',
    'experiments/chronological/probe_min_event_budget.py',
    'experiments/chronological/analyze_min_forecast_fusion_gain.py',
    'experiments/chronological/min_forecast_fusion_gain_metrics.py',
    'experiments/chronological/min_early_event_gain_metrics.py',
    'experiments/chronological/min_early_event_direction_metrics.py',
    'experiments/chronological/min_early_event_metrics.py',
    'experiments/chronological/min_event_budget_probe_metrics.py',
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--evaluation-dir', type=Path)
    parser.add_argument('--source-run', type=Path, default=training.DEFAULT_SOURCE)
    parser.add_argument('--data-dir', type=Path, default=training.paper.default_data())
    parser.add_argument('--graph-dir', type=Path, default=training.HERE/'internal_spatial_graph_is11')
    parser.add_argument('--protocol', type=Path, default=training.PROTOCOL)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--threads', type=int)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args(argv)
    if args.run_dir is None:
        args.run_dir = args.evaluation_dir.parent if args.evaluation_dir is not None else archived.DEFAULT_RUN
    for name in ('run_dir', 'evaluation_dir', 'source_run', 'data_dir', 'graph_dir', 'protocol', 'output_dir'):
        if getattr(args, name) is not None:
            setattr(args, name, getattr(args, name).resolve())
    require(args.threads is None or args.threads >= 1, 'Threads must be positive')
    return args


def select_source(args):
    source = archived.select_source(args)
    require(source.parent == args.run_dir, 'Evaluation archive must belong to the supplied run directory')
    return source


def output_path(args, source):
    output = (args.output_dir or args.run_dir/('forecast_fusion_relative_probe_'+
        datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6])).resolve()
    protected_roots = (source, args.source_run, args.run_dir/'STATE', args.run_dir/'EVENT')
    require(output != args.run_dir and output not in args.run_dir.parents,
            'Output must not replace the run directory or an ancestor')
    require(all(output != root and root not in output.parents and output not in root.parents for root in protected_roots),
            'Output must be separate from source archives and checkpoint directories')
    require(not output.exists(), 'Use a fresh relative-probe directory')
    return output


def merge_relative(total, row):
    """Counters/sums add; extrema retain their original aggregation meaning."""
    for key, value in row.items():
        if key in ('schema', 'mode', 'max_delta'):
            require(key not in total or total[key] == value, 'Relative intervention specification changed')
            total[key] = value
        elif key.endswith('_max'):
            total[key] = max(total.get(key, value), value)
        elif key.endswith('_min'):
            total[key] = min(total.get(key, value), value)
        else:
            total[key] = total.get(key, 0)+value


def verify_axes(data, info, graph, roads):
    target, valid, indices, stations = (info[name] for name in ('target', 'valid', 'indices', 'stations'))
    require(target.shape == valid.shape and target.ndim == 3 and target.shape[1:] == (12, len(data.station_ids)),
            'Original validation target/mask axes changed')
    require(len(target) == data.loaders['val_loader'].size and np.array_equal(stations, data.station_ids),
            'Original validation sample/station axes changed')
    expected = data.audit['counts']['train_samples']+np.arange(len(target), dtype=np.int64)
    require(np.array_equal(indices, expected), 'Original validation sample indices changed')
    mask = info['replay']['EVENT']['mask_value']
    require(valid.dtype == np.bool_ and np.array_equal(valid, target != mask), 'Original validation mask changed')
    rows = [encode_event(sample, len(stations)) for sample in data.loaders['val_loader'].original_samples]
    events = {key: np.asarray([row[key] for row in rows], dtype=np.float32 if key == 'relation'
        else np.int64 if key in ('type', 'description') else bool) for key in EVENT_KEYS}
    regions = probe.regions_for(events, graph, roads)
    require(regions.keys() == info['regions'].keys() and all(np.array_equal(
        np.broadcast_to(value, (len(target), len(stations))), info['regions'][name]) for name, value in regions.items()),
        'Current event/road regions differ from the original archive')


def run_probe(model, data, saved, info, graph, roads, output, protected_paths):
    """Full ON gate, two interventions and exact per-batch restoration."""
    output = Path(output)
    require(not output.exists(), 'Use a fresh relative-probe directory')
    output.mkdir(parents=True)
    before = probe.snapshot_files(protected_paths)
    calls = {case: 0 for case in CASES}
    preservation, replay, mechanisms, relative_audits, weight_totals = {}, {}, {}, {}, {}
    try:
        verify_axes(data, info, graph, roads)
        model_hash = tensor_state_hash(model.state_dict())
        require(model_hash == tensor_state_hash(saved['best_model_state']), 'EVENT model is not the saved selected best')
        archived_hash = info.get('model_state_sha256')
        require(archived.archive.is_hash(archived_hash) and model_hash == archived_hash,
                'EVENT selected model differs from original evaluation')
        require(saved['best_epoch'] == info['comparable']['best_stage_epochs']['EVENT'], 'EVENT selected stage epoch changed')
        expected = evaluation.selected_metrics(saved, dict(actual=info['identity']['source']['reference_metrics']))
        require(np.allclose(expected, info['replay']['EVENT']['expected'], atol=1e-5, rtol=1e-6),
                'EVENT selected metric certificate changed')
        target, valid, indices, stations = (info[name] for name in ('target', 'valid', 'indices', 'stations'))
        predictions = {case: np.asarray(value) for case, value in info['predictions'].items()}
        require(set(predictions) == set(evaluation.CASES), 'Expected all eight cached original cases')
        replay_on = np.empty_like(target)
        partitions, position = [], 0
        print('Verifying saved EVENT best stage '+str(saved['best_epoch'])+' ON ('+str(len(target))+' rows)', flush=True)
        with probe.preserve_model(model, preservation, 'EVENT'):
            for batch in data.loaders['val_loader'].get_iterator():
                row = evaluation.forward_case(model, batch, data.scaler, 'EVENT_ON')
                calls['EVENT_ON'] += 1
                end = position+len(row['prediction'])
                require(end <= len(target) and np.array_equal(row['target'], target[position:end]),
                        'ON replay target differs from the original archive')
                replay_on[position:end] = row['prediction']
                evaluation.update_weights(weight_totals.setdefault('EVENT_ON_REPLAY', {}), row['weights'], row['fusion_means'])
                probe.merge_mechanisms(mechanisms.setdefault('EVENT_ON_REPLAY', {}), row['mechanisms'])
                partitions.append(end-position)
                position = end
                print('Verified EVENT ON: '+str(position)+'/'+str(len(target))+' rows', flush=True)
            require(position == len(target) and partitions == info['replay']['EVENT']['batch_partition'],
                    'Original validation batch partition changed')
            actual = compute_all_metrics(torch.from_numpy(replay_on), torch.from_numpy(target),
                                         torch.tensor(info['replay']['EVENT']['mask_value']))
            metric_passed = bool(np.allclose(actual, expected, atol=1e-5, rtol=1e-6))
            exact = np.array_equal(replay_on, predictions['EVENT_ON'])
            replay.update(status='BEST_EVENT_ON_REPLAY_PASS' if exact and metric_passed else 'BEST_EVENT_ON_REPLAY_FAILED',
                best_stage_epoch=saved['best_epoch'], actual=actual, expected=expected, batch_partition=partitions,
                metric_passed=metric_passed,
                archived_prediction_exact=exact, archived_prediction_abs_max=float(np.abs(
                    replay_on.astype(np.float64)-predictions['EVENT_ON'].astype(np.float64)).max()),
                metric_atol=1e-5, metric_rtol=1e-6, model_state_sha256=model_hash)
            training.legacy.write_json(output/'replay.json', replay)
            require(metric_passed, 'EVENT best ON metric replay failed; see replay.json actual/expected')
            require(exact, 'EVENT ON replay differs from the saved archive; relative interventions blocked')
            print('Full EVENT ON replay passed; permitting exact and bounded relative-weight interventions', flush=True)
            position = 0
            for batch in data.loaders['val_loader'].get_iterator():
                end = position+len(batch['x_data'])
                for case, mode in RELATIVE_CASES.items():
                    with relative_fusion(model, mode=mode, expected_calls=1) as audit:
                        row = evaluation.forward_case(model, batch, data.scaler, 'EVENT_ON')
                        calls[case] += 1
                    require(np.array_equal(row['target'], target[position:end]), 'Relative intervention target axes changed')
                    predictions.setdefault(case, np.empty_like(target))[position:end] = row['prediction']
                    merge_relative(relative_audits.setdefault(case, {}), audit)
                    probe.merge_mechanisms(mechanisms.setdefault(case, {}), row['mechanisms'])
                    evaluation.update_weights(weight_totals.setdefault(case, {}), row['weights'], row['fusion_means'])
                    restored = evaluation.forward_case(model, batch, data.scaler, 'EVENT_ON')
                    calls['EVENT_ON'] += 1
                    require(np.array_equal(restored['target'], target[position:end])
                            and np.array_equal(restored['prediction'], replay_on[position:end]), 'EVENT ON restoration failed')
                position = end
                print('Paired EXACT/ON/BOUNDED/ON: '+str(position)+'/'+str(len(target))+' rows', flush=True)
            require(position == len(target), 'Incomplete relative interventions')
        require(tensor_state_hash(model.state_dict()) == model_hash, 'EVENT tensors changed during relative probe')
        result = metrics.summarize(predictions, target, valid, info['regions'], indices, stations, info['fixed_hard_ids'])
        diagnostics = dict(relative_transforms=relative_audits, replay=replay, comparability=info['comparable'],
            engineering_only=info['summary']['engineering_only'])
        weights = evaluation.weight_results(weight_totals)
        mapping = {name: 'region_'+str(i) for i, name in enumerate(info['regions'])}
        np.savez_compressed(output/'paired_predictions.npz', **predictions, target=target, valid=valid,
            sample_indices=indices, station_ids=stations, fixed_hard_sample_indices=info['fixed_hard_ids'],
            **{mapping[name]: value for name, value in info['regions'].items()})
        training.legacy.write_json(output/'region_definitions.json', dict(masks=mapping,
            scope='unchanged_original_archive_regions_not_certified_accident_effect'))
        training.legacy.write_json(output/'analysis.json', result)
        training.legacy.write_json(output/'mechanisms.json', dict(cases=mechanisms, relative_transforms=relative_audits,
            fusion_weights=weights, component_order=evaluation.COMPONENT_ORDER,
            original_mechanisms_reused=True, weights_not_importance=True))
        training.legacy.write_json(output/'comparability.json', dict(info['comparable'],
            source_scope='same_EVENT_best_checkpoint_relative_weight_inference_sensitivity',
            cached_original_cases=list(evaluation.CASES), newly_inferred_cases=['EVENT_ON']+list(RELATIVE_CASES),
            primary_baseline='original_IGSTGNN_not_present_in_source_archive',
            no_source_MIN_or_STATE_forward=True, no_training=True, no_test_loader_iteration=True,
            test_payload_deserialized=info['identity'].get('test_payload_deserialized', True)))
        rows = list(metrics.flat_rows(result))
        archived.archive.write_csv(output/'groups.csv', rows)
        archived.archive.write_csv(output/'cases.csv', (dict(case=case, **row) for case, row in result['cases'].items()))
        archived.archive.write_csv(output/'comparisons.csv', (row for row in rows if row.get('group') == 'all_nodes'
            and row.get('horizon') == 0))
        archived.archive.write_csv(output/'group_horizons.csv', (row for row in rows if row.get('horizon', 0) > 0))
        archived.archive.write_csv(output/'directions.csv', (dict(comparison=name, group=group, direction=direction, **values)
            for name, item in result['directions'].items() for group, row in item['groups'].items()
            for direction, values in row['directions'].items()))
        archived.archive.write_csv(output/'factorial.csv', (dict(group=group, **row)
            for group, row in result['factorial']['groups'].items()))
        for filename, predicate in (
            ('all_nodes.csv', lambda row: row.get('group') == 'all_nodes' and row.get('horizon') == 0),
            ('horizons.csv', lambda row: row.get('group') == 'all_nodes' and row.get('horizon', 0) > 0),
            ('flow.csv', lambda row: row.get('family') == 'flow' and row.get('horizon') == 0),
            ('current_bias.csv', lambda row: row.get('family') == 'current_MIN_bias' and row.get('horizon') == 0),
            ('source_bias.csv', lambda row: row.get('family') == 'source_MIN_bias' and row.get('horizon') == 0),
            ('fixed_hard.csv', lambda row: row.get('group') == 'fixed_hard_windows' and row.get('horizon') == 0),
            ('roads.csv', lambda row: row.get('family') == 'road' and row.get('horizon') == 0),
            ('regions.csv', lambda row: row.get('family') in ('region', 'report_partition') and row.get('horizon') == 0),
            ('cross.csv', lambda row: row.get('family') == 'flow_current_MIN_bias_report_partition' and row.get('horizon') == 0)):
            archived.archive.write_csv(output/filename, (row for row in rows if predicate(row)))
        metrics.write_report(output, result, diagnostics)
        files = probe.require_preserved(before, protected_paths)
        training.legacy.write_json(output/'preservation.json', dict(files=files, models=preservation))
        summary = dict(schema=SCHEMA, status=STATUS, validation_rows=len(target),
            engineering_only=info['summary']['engineering_only'], best_stage_epoch=saved['best_epoch'],
            source_best_epoch=info['summary']['source_best_epoch'], cases=list(predictions), replay_passed=True,
            on_exactly_restored=True, source_artifacts_preserved=True, new_inference_calls=calls,
            total_new_inference_calls=sum(calls.values()), optimizer_updates=0, test_evaluated=False,
            primary_baseline='original_IGSTGNN_not_present_in_source_archive',
            all_nodes={name: {key: row['groups']['all_nodes'][key] for key in ('mae_gain_pooled', 'mae_gain_macro')}
                       for name, row in result['comparisons'].items()}, output_dir=str(output))
        training.legacy.write_json(output/'summary.json', summary)
        return summary
    except BaseException as error:
        after = probe.snapshot_files(protected_paths)
        training.legacy.write_json(output/'failure.json', dict(schema=SCHEMA, status='MIN_FORECAST_FUSION_RELATIVE_PROBE_FAILED',
            error_type=type(error).__name__, error=str(error), new_inference_calls=calls,
            optimizer_updates=0, test_evaluated=False))
        training.legacy.write_json(output/'preservation.json', dict(source_artifacts_preserved=before == after,
            before=before, after=after, models=preservation, probe_completed=False))
        raise


def load_event(args, info, logger):
    """Validate original preparation inputs without its inference preflight."""
    identity = training.legacy.read_json(args.run_dir/'identity.json')
    require(identity == info['identity'], 'Current run identity differs from the evaluation archive')
    require(args.check is identity['engineering_check'], 'Engineering marker differs from the original run')
    args.threads = identity['threads'] if args.threads is None else args.threads
    require(args.threads == identity['threads'], 'Replay CPU threads differ from the original run')
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    setup = SimpleNamespace(**dict(vars(args), arms=list(identity['arm_modes']),
        batch_size=identity['protocol']['batch_size'], seed=identity['seed'], epochs=1, only=None, worker=None,
        evaluate_after=False))
    protocol = training.load_protocol(setup)
    hashes = training.paper.verify_release(args.data_dir, protocol)
    data = training.EventPublishedData(args.data_dir, protocol, logger, check=args.check)
    graph, roads, graph_info = training.legacy.load_graph(args.graph_dir, data.station_ids, protocol['graph_sha256'])
    source = training.load_source(setup, hashes)
    require(data.loaders['val_loader'].bs == data.loaders['train_loader'].bs == source['identity']['batch_size'],
            'Replay must retain the original batch size')
    initialization = training.legacy.read_json(args.run_dir/'initialization.json')
    current = training.make_identity(setup, protocol, hashes, graph_info, initialization, source)
    require(current == identity, 'Fusion training identity changed; no preflight or training was run')
    baseline, baseline_arrays = training.load_source_reference(setup, identity)
    for name in ('target', 'valid', 'sample_indices', 'station_ids'):
        key = {'sample_indices': 'indices', 'station_ids': 'stations'}.get(name, name)
        require(np.array_equal(baseline_arrays[name], info[key]), 'Source reference '+name+' differs from evaluation')
    require(np.array_equal(baseline_arrays['prediction'], info['predictions']['SOURCE_MIN']), 'Cached source MIN predictions changed')
    saved = torch.load(args.run_dir/'EVENT/last.pt', map_location='cpu', weights_only=False)
    training.validate_saved(saved, identity, 'EVENT')
    require(training.legacy.read_json(args.run_dir/'EVENT/history.json') == saved['history'], 'EVENT history sidecar differs')
    companion = training.legacy.read_json(args.run_dir/'EVENT/summary.json')
    require(all(companion.get(key) == saved[key] for key in ('completed_epoch', 'best_epoch', 'best_metric'))
            and companion.get('arm') == 'EVENT' and companion.get('mode') == 'event', 'EVENT summary sidecar differs')
    require(saved['best_epoch'] == info['comparable']['best_stage_epochs']['EVENT'], 'Selected EVENT stage changed since evaluation')
    model, _ = training.build_arm(setup, 'EVENT', len(data.station_ids), graph, torch.device(args.device), source)
    model.load_state_dict(saved['best_model_state'], strict=True)
    model.eval()
    archived_preservation = archived.archive.read_json(args.evaluation_dir/'preservation.json')
    info['model_state_sha256'] = archived_preservation['models']['EVENT']['model_state_sha256']
    require(tensor_state_hash(model.state_dict()) == info['model_state_sha256'], 'Original evaluated EVENT best tensors changed')
    return model, data, saved, graph, roads


def protected_files(args, info):
    paths = {'run/'+name: args.run_dir/name for name in ('identity.json', 'initialization.json', 'source_reference.json', 'source_reference.npz')}
    paths.update({'source/'+name: args.source_run/name for name in training.SOURCE_FILES})
    for path in args.evaluation_dir.rglob('*'):
        if path.is_file():
            require(args.evaluation_dir in path.resolve().parents, 'Evaluation archive contains an external file link')
            paths['evaluation/'+path.relative_to(args.evaluation_dir).as_posix()] = path
    for arm in ('STATE', 'EVENT'):
        paths.update({arm+'/'+name: args.run_dir/arm/name for name in ('last.pt', 'history.json', 'summary.json')})
    for key, directory in (('source_sha256', REPO), ('package', args.data_dir), ('graph_inputs_sha256', args.graph_dir)):
        paths.update({key+'/'+name: directory/name for name in info['identity'][key]})
    paths['protocol'] = args.protocol
    paths.update({'probe_code/'+name: REPO/name for name in CODE_SOURCES})
    require(all(path.is_file() for path in paths.values()), 'A protected relative-probe input is missing')
    return paths


def main(argv=None):
    args = parse_args(argv)
    args.evaluation_dir = select_source(args)
    args.output_dir = output_path(args, args.evaluation_dir)
    logger = logging.getLogger('min_forecast_fusion_relative_probe')
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler(sys.stdout))
    with archived.archive.readonly_run_lock(args.run_dir), training.readonly_source_lock(args.source_run):
        print('Source evaluation: '+str(args.evaluation_dir), flush=True)
        info = archived.load_and_verify(args.evaluation_dir)
        paths = protected_files(args, info)
        before = probe.snapshot_files(paths)
        model, data, saved, graph, roads = load_event(args, info, logger)
        probe.require_preserved(before, paths)
        print('Probe directory: '+str(args.output_dir), flush=True)
        result = run_probe(model, data, saved, info, graph, roads, args.output_dir, paths)
        training.legacy.write_json(args.output_dir/'provenance.json', dict(training_identity=info['identity'],
            original_evaluation=str(args.evaluation_dir), original_evaluation_sha256={name: sha256(args.evaluation_dir/name) for name in archived.SOURCE_FILES},
            code_sha256={name: sha256(REPO/name) for name in CODE_SOURCES},
            original_archive_cases_reused=True, test_payload_deserialized=True, test_evaluated=False,
            no_test_loader_iteration=True, existing_MIN_retrained=False, preflight_rerun=False))
        print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    gc.collect()
    return result


if __name__ == '__main__':
    main()
