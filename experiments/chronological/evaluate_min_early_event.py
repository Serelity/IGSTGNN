"""Read-only best-validation replay and dataflow diagnostics for early events."""
import argparse
from contextlib import contextmanager, ExitStack
from datetime import datetime
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
from experiments.chronological import train_internal_spatial as legacy
from experiments.chronological import probe_min_event_budget as old_probe
from experiments.chronological.min_early_event_metrics import BASE_CASES, summarize, write_report
from experiments.chronological.smoke import sha256
from src.models.incident_spatial import require
from src.models.incident_spatial_residual import tensor_state_hash
from src.utils.igstgnn_published_events import EVENT_KEYS, encode_event
from src.utils.metrics import compute_all_metrics

SCHEMA = 'min_early_event_evaluation_v1'
REPLAY_ATOL, REPLAY_RTOL = 1e-5, 1e-6


def training_comparability(saved_by_arm):
    """Audit committed histories, rather than trusting a previous launcher report."""
    histories = {arm: saved['history'] for arm, saved in saved_by_arm.items()}
    completed = {arm: saved['completed_epoch'] for arm, saved in saved_by_arm.items()}
    require(all(type(completed[arm]) is int and completed[arm] == len(history) and completed[arm] > 0
                for arm, history in histories.items()), 'Invalid committed training budgets')
    common = min(completed.values())
    for epoch in range(common):
        hashes = [history[epoch].get('train_order_sha256') for history in histories.values()]
        require(all(isinstance(value, str) and len(value) == 64 for value in hashes), 'Missing training sample-order evidence')
        require(len(set(hashes)) == 1, 'Paired common-prefix training sample order differs')
    updates = {arm: saved.get('global_updates') for arm, saved in saved_by_arm.items()}
    require(all(type(value) is int and value > 0 for value in updates.values()), 'Missing optimizer-update evidence')
    equal = len(set(completed.values())) == 1 and len(set(updates.values())) == 1
    return dict(status='COMMON_PREFIX_TRAIN_ORDER_VERIFIED', common_epoch=common, completed_epochs=completed,
        global_updates=updates, best_epochs={arm: old_probe.best_row(saved)['epoch'] for arm, saved in saved_by_arm.items()},
        same_completed_training_budget=equal, common_prefix_training_order_verified=True,
        whole_model_scope='same_protocol_equal_completed_budget_independent_best_validation' if equal else
                          'unequal_completed_budget_exploratory_independent_best_validation')


def shuffled_locations(events, sample_indices, seed):
    """Permute complete relation rows across ALL stations, including zero rows.

    The cycle is deterministic per global sample ID and consumes no global RNG.
    Semantic categories, report presence and relation availability stay fixed.
    """
    relation = np.asarray(events['relation'])
    indices = np.asarray(sample_indices)
    require(relation.ndim == 3 and relation.shape[-1] == 3 and relation.shape[1] >= 2,
            'Expected relation [sample,station,3] with at least two stations')
    require(indices.shape == (len(relation),) and indices.dtype.kind in 'iu' and (indices >= 0).all(), 'Invalid sample IDs')
    result = {key: np.asarray(value).copy() for key, value in events.items()}
    donors = np.empty(relation.shape[:2], np.int64)
    for row, sample in enumerate(indices):
        order = np.random.default_rng(np.random.SeedSequence([int(seed), int(sample)])).permutation(relation.shape[1])
        donors[row, order] = np.roll(order, 1)
        result['relation'][row] = relation[row, donors[row]]
    original_support = np.abs(relation).sum(-1) > 0
    changed_support = np.abs(result['relation']).sum(-1) > 0
    audit = dict(seed=int(seed), schema='full_station_relation_cycle_v1', uses_targets=False,
        zero_relation_rows_permuted=True, fixed_semantic_categories=True,
        scope='full_station_support_location_sensitivity_potential_OOD',
        rows=len(indices), moved_vector_positions=int(donors.size),
        numerically_changed_vectors=int(np.any(relation != result['relation'], axis=-1).sum()),
        numerically_changed_rows=int(np.any(relation != result['relation'], axis=(1, 2)).sum()),
        support_changed_positions=int((original_support != changed_support).sum()),
        support_changed_rows=int(np.any(original_support != changed_support, axis=1).sum()))
    return result, donors, audit


@contextmanager
def capture_forecasts(model):
    forecasts, handles = {}, []
    def hook(index):
        def record(module, args, output):
            require(index not in forecasts and isinstance(output, tuple) and len(output) == 3,
                    'Unexpected layer forecast output or repeated layer call')
            forecasts[index] = (output[1].detach(), output[2].detach())
        return record
    try:
        require(len(model.layers) == 5, 'Forecast audit requires five MIN layers')
        for i, layer in enumerate(model.layers):
            handles.append(layer.register_forward_hook(hook(i)))
        yield forecasts
    finally:
        for handle in handles:
            handle.remove()


def forecast_cases(model, forecasts, expected):
    require(set(forecasts) == set(range(5)), 'Incomplete forecast feature coverage')
    dif = [forecasts[i][0] for i in range(5)]
    inh = [forecasts[i][1] for i in range(5)]
    identity = model._decode_forecast(sum(dif)+sum(inh), None)
    require(torch.equal(identity, expected), 'Native forecast-head identity replay failed')
    result = {}
    for branch, features in [('DIF', dif), ('INH', inh)]:
        for index in range(5):
            changed = [torch.zeros_like(value) if i == index else value for i, value in enumerate(features)]
            hidden = sum(changed)+sum(inh) if branch == 'DIF' else sum(dif)+sum(changed)
            result[f'EVENT_FORECAST_{branch}_{index}_OFF'] = model._decode_forecast(hidden, None)
        zeros = [torch.zeros_like(value) for value in features]
        hidden = sum(zeros)+sum(inh) if branch == 'DIF' else sum(dif)+sum(zeros)
        result[f'EVENT_FORECAST_{branch}_ALL_OFF'] = model._decode_forecast(hidden, None)
    return result


def small_mechanisms(context):
    result = old_probe.small_mechanisms(context)
    records = context.get('records', [])
    result['exchange_calls'] = len(records)
    result['local_substeps'] = sum(int(group.shape[0]) for group in records)
    for key, value in context.get('early_event_diagnostics', {}).items():
        if torch.is_tensor(value) and value.numel() == 1:
            result['early_'+key] = float(value.detach().cpu())
        elif isinstance(value, (int, float, np.integer, np.floating)):
            result['early_'+key] = float(value)
        elif torch.is_tensor(value):
            # Detached full node tensors are reduced within the current batch.
            tensor = value.detach()
            result['early_'+key+'_abs_sum'] = float(tensor.abs().sum().cpu())
            result['early_'+key+'_abs_max'] = float(tensor.abs().max().cpu()) if tensor.numel() else 0.
            result['early_'+key+'_signed_sum'] = float(tensor.sum().cpu())
            result['early_'+key+'_positions'] = tensor.numel()
    require(all(np.isfinite(value) for value in result.values()), 'Nonfinite mechanism diagnostics')
    return result


def forward_case(model, batch, scaler, case, *, events_override=None, audit_forecasts=False):
    before = old_probe.input_digest(batch, events_override)
    try:
        device = next(model.parameters()).device
        x = torch.as_tensor(batch['x_data'], device=device, dtype=torch.float32)
        y = torch.as_tensor(batch['y_data'], device=device, dtype=torch.float32)
        payload = batch['event_data'] if events_override is None else events_override
        events = {key: torch.as_tensor(value, device=device,
                  dtype=torch.long if key in ('type', 'description') else torch.float32 if key == 'relation' else torch.bool)
                  for key, value in payload.items()}
        kwargs = dict(incident_data=events, return_spatial_details=True,
                      report_enabled=case != 'EVENT_OFF', semantic_enabled=case != 'EVENT_SEMANTIC_UNK')
        with torch.inference_mode():
            if audit_forecasts:
                with capture_forecasts(model) as forecasts:
                    output = model(x, **kwargs)
                head_cases = forecast_cases(model, forecasts, output['prediction'])
            else:
                output, head_cases = model(x, **kwargs), {}
            def inverse(value):
                array = scaler.inverse_transform(value).squeeze(-1).cpu().numpy().copy()
                require(np.isfinite(array).all(), 'Nonfinite inference output or target')
                return array
            return dict(prediction=inverse(output['prediction']), target=inverse(y),
                        mechanisms=small_mechanisms(output['spatial']),
                        forecasts={name: inverse(value) for name, value in head_cases.items()})
    finally:
        require(old_probe.input_digest(batch, events_override) == before, 'Inference changed input arrays')


def run_probe(models, data, saved_by_arm, graph, roads, output, *, seed=20261014,
              forecast_audit=True, protected_paths=None):
    output = Path(output)
    require(not output.exists(), 'Use a fresh evaluation output directory')
    output.mkdir(parents=True)
    paths = protected_paths or {}
    before_files = old_probe.snapshot_files(paths)
    loader, nodes = data.loaders['val_loader'], len(data.station_ids)
    indices = data.audit['counts']['train_samples']+np.arange(loader.size, dtype=np.int64)
    rows = [encode_event(sample, nodes) for sample in loader.original_samples]
    events = {key: np.asarray([row[key] for row in rows], dtype=np.float32 if key == 'relation'
              else np.int64 if key in ('type', 'description') else bool) for key in EVENT_KEYS}
    shuffled, donors, shuffle_audit = shuffled_locations(events, indices, seed)
    regions = old_probe.regions_for(events, graph, roads)
    predictions, mechanisms, replays, preservation, calls = {}, {}, {}, {}, {}
    target = np.empty((loader.size, 12, nodes), np.float32)
    expected_partition = [min(loader.bs, loader.size-i) for i in range(0, loader.size, loader.bs)]
    try:
        require(all(value != '<missing>' for value in before_files.values()), 'Protected source artifact is missing')
        require({'STATE', 'EVENT'} <= models.keys() and models.keys() == saved_by_arm.keys(), 'STATE and EVENT best models required')
        comparable = training_comparability(saved_by_arm)
        with ExitStack() as stack:
            for arm, model in models.items():
                require(tensor_state_hash(model.state_dict()) == tensor_state_hash(saved_by_arm[arm]['best_model_state']),
                        'Model is not saved best checkpoint: '+arm)
                stack.enter_context(old_probe.preserve_model(model, preservation, arm))
            # Do not permit a single OFF/content intervention until EVERY full ON replay passes.
            for arm in ('STATE', 'EVENT', 'LOCATION'):
                if arm not in models:
                    continue
                model = models[arm]
                case, position, partitions = arm+'_ON', 0, []
                predictions[case] = np.empty_like(target)
                selected = old_probe.best_row(saved_by_arm[arm])
                print(f'Replaying {arm} best epoch {selected["epoch"]}: {loader.size} rows, batch {loader.bs}', flush=True)
                for batch in loader.get_iterator():
                    row = forward_case(model, batch, data.scaler, case)
                    calls[case] = calls.get(case, 0)+1
                    end = position+len(row['prediction'])
                    partitions.append(end-position)
                    predictions[case][position:end] = row['prediction']
                    if arm == 'STATE':
                        target[position:end] = row['target']
                    else:
                        require(np.array_equal(target[position:end], row['target']), 'Different targets across ON arms')
                    old_probe.merge_mechanisms(mechanisms.setdefault(case, {}), row['mechanisms'])
                    position = end
                    print(f'{arm} ON: {position}/{loader.size}', flush=True)
                require(position == loader.size and partitions == expected_partition, 'Original validation batch partition changed')
                labels = torch.from_numpy(target)
                null = labels.min() if labels.min() < 1 else torch.tensor(0.)
                actual = compute_all_metrics(torch.from_numpy(predictions[case]), labels, null)
                expected = [selected[key] for key in ('validation_mae', 'validation_mape', 'validation_rmse')]
                require(np.allclose(actual, expected, atol=REPLAY_ATOL, rtol=REPLAY_RTOL),
                        f'{arm} best ON replay failed: actual={actual}, saved={expected}')
                replays[arm] = dict(status='BEST_ON_REPLAY_PASS', best_epoch=selected['epoch'], expected=expected, actual=actual,
                    mask_value=float(null), batch_partition=partitions, atol=REPLAY_ATOL, rtol=REPLAY_RTOL)
            legacy.write_json(output/'replay.json', replays)
            print('All best ON replays passed; permitting EVENT interventions and direct forecast readouts', flush=True)
            position, model = 0, models['EVENT']
            for batch in loader.get_iterator():
                on = forward_case(model, batch, data.scaler, 'EVENT_ON', audit_forecasts=forecast_audit)
                calls['EVENT_ON'] += 1
                end = position+len(on['prediction'])
                require(np.array_equal(on['prediction'], predictions['EVENT_ON'][position:end]), 'Paired ON differs from verified replay')
                require(np.array_equal(on['target'], target[position:end]), 'Paired ON target changed')
                for case, value in on['forecasts'].items():
                    predictions.setdefault(case, np.empty_like(target))[position:end] = value
                for case in BASE_CASES[2:]:
                    payload = {key: value[position:end] for key, value in shuffled.items()} if case.endswith('SHUFFLE') else None
                    changed = forward_case(model, batch, data.scaler, case, events_override=payload)
                    calls[case] = calls.get(case, 0)+1
                    require(np.array_equal(changed['target'], target[position:end]), 'Intervention target changed')
                    predictions.setdefault(case, np.empty_like(target))[position:end] = changed['prediction']
                    old_probe.merge_mechanisms(mechanisms.setdefault(case, {}), changed['mechanisms'])
                    restored = forward_case(model, batch, data.scaler, 'EVENT_ON')
                    calls['EVENT_ON'] += 1
                    require(np.array_equal(restored['prediction'], on['prediction']), 'EVENT ON restoration failed')
                position = end
                print(f'Paired EVENT interventions + restoration: {position}/{loader.size}', flush=True)
            require(position == loader.size, 'Incomplete interventions')
        require(len({row['mask_value'] for row in replays.values()}) == 1, 'Arm validity conventions differ')
        valid = np.isfinite(target) & (target != replays['STATE']['mask_value'])
        analysis = summarize(predictions, target, valid, regions, indices)
        analysis['training_comparability'] = comparable
        for row in analysis['comparisons'].values():
            if row['scope'] == 'independently_trained_best_validation':
                row['scope'] = comparable['whole_model_scope']
        legacy.write_json(output/'analysis.json', analysis)
        legacy.write_json(output/'comparability.json', comparable)
        # Cross strata share STATE error signs; the primary comparison can reuse the audited pure NumPy helper.
        from experiments.chronological.min_frozen_gain_metrics import summarize_strata
        cross = summarize_strata(predictions['EVENT_ON'], predictions['STATE_ON'], target, valid,
                                 {name: regions[name] for name in ('direct_report_nodes', 'potential_only', 'remaining_nodes')})
        legacy.write_json(output/'event_state_cross_strata.json', cross)
        legacy.write_json(output/'mechanisms.json', dict(cases=mechanisms,
            aggregation='sums_over_original_batches_except_fields_ending_max',
            capacity_scope='dimensionless_latent_exchange_budget_not_vehicle_capacity',
            forecast_audit=dict(enabled=forecast_audit, cases=sum(name.startswith('EVENT_FORECAST_') for name in predictions),
                scope='remove_only_selected_final_forecast_features_backcasts_and_later_layers_remain_ON'),
            location_shuffle=shuffle_audit))
        region_keys = {name: f'region_{i}' for i, name in enumerate(regions)}
        legacy.write_json(output/'region_definitions.json', dict(masks=region_keys,
            direct='author_relation_support_and_report_presence_not_real_incident_effect',
            potential='weak_component_envelope_not_physical_propagation', natural_no_incident_group_verified=False))
        np.savez_compressed(output/'paired_predictions.npz', **predictions, target=target, valid=valid,
            sample_indices=indices, station_ids=np.asarray(data.station_ids, np.int64), location_donor_station_indices=donors,
            **{region_keys[name]: value for name, value in regions.items()})
        write_report(output, analysis)
        files = old_probe.require_preserved(before_files, paths)
        legacy.write_json(output/'preservation.json', dict(files=files if paths else dict(status='NOT_REQUESTED'), models=preservation))
        result = dict(schema=SCHEMA, status='MIN_EARLY_EVENT_EVALUATION_PASS', validation_rows=loader.size, batch_size=loader.bs,
            best_epochs={name: row['best_epoch'] for name, row in replays.items()}, cases=list(predictions),
            forecast_audit=forecast_audit, replay_passed=True, on_exactly_restored=True, training_updates=0,
            test_evaluated=False, source_artifacts_preserved=bool(paths), inference_calls=calls,
            all_nodes={name: row['metrics']['all_nodes'] for name, row in analysis['cases'].items() if not name.startswith('EVENT_FORECAST_')},
            comparisons={name: row['metrics']['all_nodes'] for name, row in analysis['comparisons'].items() if not name.startswith('forecast_')},
            output_dir=str(output))
        result['training_comparability'] = comparable
        result['engineering_only'] = bool(saved_by_arm['EVENT'].get('identity', {}).get('engineering_check', False))
        legacy.write_json(output/'summary.json', result)
        return result
    except BaseException as error:
        legacy.write_json(output/'failure.json', dict(status='MIN_EARLY_EVENT_EVALUATION_FAILED', error_type=type(error).__name__,
            error=str(error), inference_calls=calls, training_updates=0, test_evaluated=False))
        files = old_probe.require_preserved(before_files, paths)
        legacy.write_json(output/'preservation.json', dict(files=files if paths else dict(status='NOT_REQUESTED'),
            models=preservation, evaluation_completed=False))
        raise


def main(argv=None):
    from experiments.chronological import train_min_early_event as training
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, default=training.paper.default_data())
    parser.add_argument('--graph-dir', type=Path, default=training.HERE/'internal_spatial_graph_is11')
    parser.add_argument('--protocol', type=Path, default=training.PROTOCOL)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--threads', type=int)
    parser.add_argument('--shuffle-seed', type=int, default=20261014)
    parser.add_argument('--skip-forecast-audit', action='store_true')
    parser.add_argument('--check', action='store_true', help='Require an engineering-check source run')
    args = parser.parse_args(argv)
    require(0 <= args.shuffle_seed < 2**32, 'Invalid shuffle seed')
    for name in ('run_dir', 'data_dir', 'graph_dir', 'protocol'):
        setattr(args, name, getattr(args, name).resolve())
    identity = legacy.read_json(args.run_dir/'identity.json')
    require(identity['schema'] == training.SCHEMA, 'Expected early-event run schema')
    require(not args.check or identity['engineering_check'], '--check requires engineering source')
    require({'STATE', 'EVENT'} <= identity['arm_modes'].keys(), 'Source run must contain STATE and EVENT')
    args.threads = identity['threads'] if args.threads is None else args.threads
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    old_probe.verify_runtime(identity, args.device)
    configuration = dict(vars(args))
    configuration.update(seed=identity['seed'], check=identity['engineering_check'],
        batch_size=identity['protocol']['batch_size'], arms=list(identity['arm_modes']), epochs=1,
        only=None, worker=None, evaluate_test=False)
    setup = SimpleNamespace(**configuration)
    logger = logging.getLogger('min_early_event_evaluation')
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.setLevel(logging.INFO)
    with legacy.run_lock(args.run_dir):
        protocol, data, graph, initialization, current = training.prepare(setup, logger)
        require(current == identity, 'Training source/data/protocol identity changed; evaluation blocked')
        require(legacy.read_json(args.run_dir/'initialization.json') == initialization, 'Initialization certificate changed')
        _, roads, _ = legacy.load_graph(args.graph_dir, data.station_ids, protocol['graph_sha256'])
        models, saved = {}, {}
        # Preserve STATE-first order; targets are established only from STATE.
        for arm in ('STATE', 'EVENT', 'LOCATION'):
            if arm not in identity['arm_modes'] or not (args.run_dir/arm/'last.pt').is_file():
                continue
            saved[arm] = torch.load(args.run_dir/arm/'last.pt', map_location='cpu', weights_only=False)
            training.validate_saved(saved[arm], identity, arm)
            require(legacy.read_json(args.run_dir/arm/'history.json') == saved[arm]['history'], 'JSON history differs from checkpoint')
            companion = legacy.read_json(args.run_dir/arm/'summary.json')
            selected = old_probe.best_row(saved[arm])
            require(all(companion.get(key) == value for key, value in dict(arm=arm, mode=training.ARMS[arm],
                completed_epoch=saved[arm]['completed_epoch'], best_epoch=selected['epoch'],
                best_metric=saved[arm]['best_metric']).items()), 'JSON summary differs from checkpoint')
            models[arm], _ = training.build_arm(setup, arm, len(data.station_ids), graph, torch.device(args.device))
            models[arm].load_state_dict(saved[arm]['best_model_state'], strict=True)
            models[arm].eval()
        paths = {'run/'+name: args.run_dir/name for name in ('identity.json', 'initialization.json', 'data_audit.json')}
        for arm in saved:
            paths.update({arm+'/'+name: args.run_dir/arm/name for name in ('last.pt', 'history.json', 'summary.json')})
        for key, directory in [('source_sha256', REPO), ('package', args.data_dir), ('graph_inputs_sha256', args.graph_dir)]:
            paths.update({key+'/'+name: directory/name for name in identity[key]})
        evaluation_sources = ('experiments/chronological/evaluate_min_early_event.py',
            'experiments/chronological/min_early_event_metrics.py', 'experiments/chronological/probe_min_event_budget.py',
            'experiments/chronological/min_event_budget_probe_metrics.py', 'experiments/chronological/min_frozen_gain_metrics.py')
        paths.update({'evaluation_source/'+name: REPO/name for name in evaluation_sources})
        paths['protocol'] = args.protocol
        args.output_dir = args.output_dir.resolve() if args.output_dir else args.run_dir/('early_event_evaluation_'+
            datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6])
        print('Evaluation directory: '+str(args.output_dir), flush=True)
        result = run_probe(models, data, saved, graph, roads, args.output_dir, seed=args.shuffle_seed,
                           forecast_audit=not args.skip_forecast_audit, protected_paths=paths)
        legacy.write_json(args.output_dir/'comparability.json', dict(result['training_comparability'], identity=identity,
            original_batch_size=identity['batch_size'], test_evaluated=False, existing_MIN_retrained=False))
        legacy.write_json(args.output_dir/'provenance.json', dict(training_source_sha256=identity['source_sha256'],
            evaluation_source_sha256={name: sha256(REPO/name) for name in evaluation_sources},
            data_sha256=identity['package'], graph_inputs_sha256=identity['graph_inputs_sha256'],
            runtime=identity['runtime'], original_sample_axis='train_sample_count_plus_validation_offset',
            station_ids=data.station_ids, checkpoint_weights='best_model_state', training_updates=0))
        print(json.dumps({key: result[key] for key in ('status', 'validation_rows', 'best_epochs', 'cases', 'training_updates', 'output_dir')},
                         indent=2, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
