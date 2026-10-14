"""Read-only validation replay and interventions on MIN-I best checkpoints.

Original batches and pooled validation masks are preserved. Independently
trained model comparisons and same-checkpoint dependencies remain separate.
"""
import argparse
from contextlib import contextmanager, ExitStack
from datetime import datetime
import json
import hashlib
import os
from pathlib import Path
import sys
import uuid

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.chronological import train_min_event_budget as training
from experiments.chronological import train_internal_spatial as legacy
from experiments.chronological.min_event_budget_probe_metrics import summarize, write_report
from experiments.chronological.smoke import sha256
from src.models.incident_spatial import STAT_NAMES, require
from src.models.incident_spatial_minimal import MinimalIGSTGNN
from src.models.incident_spatial_event_budget import EventBudgetIGSTGNN
from src.models.incident_spatial_residual import tensor_state_hash
from src.utils.igstgnn_published_events import EventPublishedData, EVENT_KEYS, encode_event
from src.utils.metrics import compute_all_metrics

CASES = ('A_ON', 'L_ON', 'L_OFF', 'S_ON', 'S_OFF', 'S_SEMANTIC_UNK', 'S_SEMANTIC_SHUFFLE')
SEMANTIC_KEYS = ('type', 'description', 'type_valid', 'description_valid')
REPLAY_ATOL, REPLAY_RTOL = 1e-5, 1e-6


def semantic_permutation(n, seed):
    require(n >= 2, 'Semantic shuffle requires at least two validation windows')
    order = np.random.default_rng(seed).permutation(n)
    # A randomized cycle is a derangement and never uses targets to choose donors.
    result = np.empty(n, np.int64)
    result[order] = np.roll(order, 1)
    return result


def joint_shuffled_payload(full_event_data, indices):
    indices = np.asarray(indices)
    size = len(full_event_data['type'])
    require(indices.shape == (size,) and np.issubdtype(indices.dtype, np.integer)
            and np.array_equal(np.sort(indices), np.arange(size)), 'Invalid semantic donor permutation')
    return {key: np.asarray(value)[indices].copy() if key in SEMANTIC_KEYS else value
            for key, value in full_event_data.items()}


def best_row(saved):
    history = saved['history']
    require(history and len(history) == saved['completed_epoch'], 'Invalid checkpoint history')
    index = min(range(len(history)), key=lambda i: history[i]['validation_mae'])
    row = history[index]
    require(row['epoch'] == index+1 and history[-1]['best_epoch'] == row['epoch']
            and saved['best_metric'] == row['validation_mae'], 'Checkpoint best selection disagrees with history')
    return row


def verify_saved_sidecars(root, saved):
    """Inspect companion records without invoking the trainer's repair path."""
    require(legacy.read_json(root/'history.json') == saved['history'],
            'JSON history differs from checkpoint; probe does not repair training records')
    summary = legacy.read_json(root/'summary.json')
    expected = dict(arm=saved['arm'], mode=training.ARMS[saved['arm']],
                    completed_epoch=saved['completed_epoch'], best_metric=saved['best_metric'],
                    best_epoch=best_row(saved)['epoch'], global_updates=saved['global_updates'],
                    curriculum_length=saved['curriculum_length'], parameters=saved['parameters'])
    require(all(summary.get(key) == value for key, value in expected.items()),
            'JSON summary differs from checkpoint; probe does not repair training records')
    return dict(status='CHECKPOINT_COMPANIONS_MATCH', **expected)


def snapshot_files(paths):
    return {name: sha256(path) if Path(path).is_file() else '<missing>' for name, path in paths.items()}


def require_preserved(before, paths):
    after = snapshot_files(paths)
    changed = [name for name in before if before[name] != after.get(name)]
    require(not changed, 'Source artifacts changed during probe: '+', '.join(changed))
    return dict(status='SOURCE_ARTIFACTS_PRESERVED', files=len(before), before=before, after=after)


def protected_files(args, identity):
    paths = {'run/'+name: args.run_dir/name for name in ('identity.json', 'initialization.json', 'data_audit.json')}
    for name in ('latest_stage.json',):
        if (args.run_dir/name).is_file():
            paths['run/'+name] = args.run_dir/name
    for arm in ('A', 'L', 'S'):
        for name in ('last.pt', 'history.json', 'summary.json'):
            paths[arm+'/'+name] = args.run_dir/arm/name
        if (args.run_dir/arm/'test_metrics.json').is_file():
            paths[arm+'/test_metrics.json'] = args.run_dir/arm/'test_metrics.json'
    paths.update({'source/'+name: REPO/name for name in identity['source_sha256']})
    paths.update({'package/'+name: args.data_dir/name for name in identity['package']})
    paths.update({'graph/'+name: args.graph_dir/name for name in identity['graph_inputs_sha256']})
    paths['protocol'] = args.protocol
    require(all(Path(path).is_file() for path in paths.values()), 'A protected input artifact is missing')
    return paths


def verify_runtime(identity, device):
    current = dict(torch=str(torch.__version__), cuda=torch.version.cuda, device_type=torch.device(device).type,
                   cublas_workspace_config=os.environ['CUBLAS_WORKSPACE_CONFIG'])
    require(current == identity['runtime'], 'Replay software/device identity differs from training: '+str(current))
    require(torch.get_num_threads() == identity['threads'], 'Replay CPU thread count differs from training')
    require(torch.are_deterministic_algorithms_enabled() and not torch.backends.cuda.matmul.allow_tf32
            and not torch.backends.cudnn.allow_tf32, 'Replay requires original deterministic/TF32 policy')


def verify_identity_sources(args, identity):
    require(identity.get('schema') == training.SCHEMA
            and identity.get('arm_modes') == training.ARMS, 'Expected the original three-arm MIN-I run')
    require(training.source_fingerprints() == identity['source_sha256'], 'Training source fingerprints changed')
    require(sha256(args.protocol) == identity['protocol_sha256'], 'Frozen protocol fingerprint changed')
    protocol_args = argparse.Namespace(protocol=args.protocol, batch_size=identity['protocol']['batch_size'])
    require(training.load_protocol(protocol_args) == identity['protocol'], 'Frozen protocol contents changed')
    require(training.paper.verify_release(args.data_dir, identity['protocol']) == identity['package'],
            'Published release differs from training')
    for name, expected in identity['graph_inputs_sha256'].items():
        require(sha256(args.graph_dir/name) == expected, 'Candidate graph input changed: '+name)
    require(legacy.read_json(args.run_dir/'initialization.json') == identity['initialization'],
            'Original initialization certificate changed')


def rng_digest(device):
    import hashlib
    state = np.random.get_state()
    digest = hashlib.sha256(repr((state[0], *state[2:])).encode()+state[1].tobytes())
    digest.update(torch.get_rng_state().numpy().tobytes())
    if torch.device(device).type == 'cuda':
        for value in torch.cuda.get_rng_state_all():
            digest.update(value.cpu().numpy().tobytes())
    return digest.hexdigest()


@contextmanager
def preserve_model(model, records, arm):
    device = next(model.parameters()).device
    before = tensor_state_hash(model.state_dict())
    modes = {name: module.training for name, module in model.named_modules()}
    hooks = {name: (tuple(module._forward_hooks), tuple(module._forward_pre_hooks))
             for name, module in model.named_modules()}
    gradients = {name: parameter.requires_grad for name, parameter in model.named_parameters()}
    require(not any(modes.values()) and all(p.grad is None for p in model.parameters()), 'Probe requires eval and no gradients')
    rng = rng_digest(device)
    graph = model.internal_spatial_graph
    edges, supported = graph.edge_index.clone(), graph.operator_mask.clone()
    try:
        yield
    finally:
        require(tensor_state_hash(model.state_dict()) == before, 'Inference changed model tensors: '+arm)
        require(modes == {name: m.training for name, m in model.named_modules()}, 'Inference changed model mode')
        require(hooks == {name: (tuple(m._forward_hooks), tuple(m._forward_pre_hooks))
                         for name, m in model.named_modules()}, 'Probe left hooks registered')
        require(gradients == {name: p.requires_grad for name, p in model.named_parameters()}
                and all(p.grad is None for p in model.parameters()), 'Inference changed gradients')
        require(torch.equal(edges, graph.edge_index) and torch.equal(supported, graph.operator_mask),
                'Inference changed candidate graph')
        require(rng_digest(device) == rng, 'Inference consumed global random state')
        records[arm] = dict(status='MODEL_STATE_PRESERVED', model_state_sha256=before,
                            mode_gradients_hooks_graph_rng_preserved=True)


@contextmanager
def execution_counts(model):
    counts, handles = {'adapter': 0, 'semantic_encoder': 0}, []
    def hook(key):
        def count(module, args, output):
            counts[key] += 1
        return count
    try:
        if hasattr(model, 'event_budget_adapter'):
            handles.append(model.event_budget_adapter.register_forward_hook(hook('adapter')))
            if hasattr(model.event_budget_adapter, 'semantic_encoder'):
                handles.append(model.event_budget_adapter.semantic_encoder.register_forward_hook(hook('semantic_encoder')))
        yield counts
    finally:
        for handle in handles:
            handle.remove()


def small_mechanisms(context):
    result = dict(local_substeps=0)
    for row in context.get('event_budget_records', []):
        result['local_substeps'] += 1
        for name, value in row.items():
            if name == 'substep':
                continue
            value = float(value.detach().cpu()) if torch.is_tensor(value) else float(value)
            require(np.isfinite(value), 'Non-finite local mechanism statistic')
            result[name] = max(result.get(name, 0.), value) if name.endswith('_max') else result.get(name, 0.)+value
    for group in context.get('records', []):
        for name in ('balance_abs_max', 'float64_balance_abs_max', 'capacity_violation_max',
                     'sending_violation_max', 'receiving_violation_max'):
            value = float(group[:, STAT_NAMES.index(name)].max().cpu())
            result[name] = max(result.get(name, 0.), value)
    for name in ('capacity_violation_max', 'sending_violation_max', 'receiving_violation_max'):
        require(result.get(name, 0.) <= 1e-5, 'Latent exchange constraint violation: '+name)
    return result


def input_digest(batch, semantic_payload=None):
    digest = hashlib.sha256()
    values = dict(x_data=batch['x_data'], y_data=batch['y_data'],
                  **{'event/'+key: value for key, value in batch['event_data'].items()})
    if semantic_payload is not None:
        values.update({'donor/'+key: value for key, value in semantic_payload.items()})
    for name, value in sorted(values.items()):
        array = np.asarray(value)
        digest.update(repr((name, array.shape, str(array.dtype))).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def forward_case(model, arm, batch, scaler, case, semantic_payload=None):
    before_input = input_digest(batch, semantic_payload)
    try:
        return _forward_case(model, arm, batch, scaler, case, semantic_payload)
    finally:
        require(input_digest(batch, semantic_payload) == before_input,
                'Inference changed traffic/target/event input payload')


def _forward_case(model, arm, batch, scaler, case, semantic_payload=None):
    require(case in CASES and case.startswith(arm+'_'), 'Unknown case/arm combination')
    require(not any(m.training for m in model.modules()), 'Probe requires eval mode')
    require((case == 'S_SEMANTIC_SHUFFLE') == (semantic_payload is not None), 'Unexpected/missing semantic donor payload')
    if semantic_payload is not None:
        require(set(semantic_payload) == set(batch['event_data']), 'Semantic donor payload fields differ')
        require(all(np.array_equal(semantic_payload[key], batch['event_data'][key])
                    for key in batch['event_data'] if key not in SEMANTIC_KEYS),
                'Semantic shuffle changed relation/presence/availability values')
    device = next(model.parameters()).device
    x = torch.as_tensor(batch['x_data'], dtype=torch.float32, device=device)
    y = torch.as_tensor(batch['y_data'], dtype=torch.float32, device=device)
    event = training.event_tensors(semantic_payload if semantic_payload is not None else batch['event_data'], device)
    report_enabled = not case.endswith('_OFF')
    kwargs = dict(return_spatial_details=True)
    if arm != 'A':
        kwargs.update(incident_data=event, report_enabled=report_enabled,
                      semantic_enabled=case != 'S_SEMANTIC_UNK')
    with execution_counts(model) as calls, torch.inference_mode():
        result = model(x, y, **kwargs)
        context = result['spatial']
        support = (event['relation'].abs().sum(-1) > 0) & event['report_present'][:, None] & event['relation_available'][:, None]
        if arm != 'A':
            expected = support if report_enabled else torch.zeros_like(support)
            require(torch.equal(context['report_mask'][..., 0].bool(), expected), 'Intervention report support mismatch')
            src, dst = model.internal_spatial_graph.edge_index
            edges = ((expected[:, src.clamp_min(0)] | expected[:, dst.clamp_min(0)])
                     & ((src >= 0) & (dst >= 0))[None])
            require(torch.equal(context['event_edge_support'][..., 0], edges), 'Intervention edge support mismatch')
            require(calls['adapter'] == (40 if bool(edges.any()) else 0), 'Incomplete adapter execution/disable coverage')
            require(calls['semantic_encoder'] == int(arm == 'S' and report_enabled), 'Semantic encoder execution mismatch')
        else:
            require(calls == {'adapter': 0, 'semantic_encoder': 0}, 'MIN unexpectedly executed an event path')
        require(len(context['records']) == 20, 'Incomplete five-layer spatial diagnostic coverage')
        prediction = result['prediction']*scaler.std.to(device)+scaler.mean.to(device)
        target = y*scaler.std.to(device)+scaler.mean.to(device)
        require(prediction.shape == target.shape and prediction.shape[1:] == (12, model.node_num, 1)
                and bool(torch.isfinite(prediction).all() & torch.isfinite(target).all()), 'Invalid prediction/target axes or values')
        return dict(prediction=prediction[..., 0].cpu().numpy(), target=target[..., 0].cpu().numpy(),
                    report_support=support.cpu().numpy(), mechanisms=small_mechanisms(context), execution=calls.copy())


def regions_for(events, graph, roads):
    relation = events['relation']
    direct = (np.abs(relation).sum(-1) > 0) & events['report_present'][:, None] & events['relation_available'][:, None]
    n, nodes = direct.shape
    edge = graph.edge_index.cpu().numpy()
    src, dst = edge[:, (edge >= 0).all(0)]
    neighbors = [set() for _ in range(nodes)]
    for u, v in zip(src, dst):
        neighbors[u].add(int(v))
        neighbors[v].add(int(u))
    components, unseen = [], set(range(nodes))
    while unseen:
        seed = min(unseen)
        found, queue = {seed}, [seed]
        unseen.remove(seed)
        while queue:
            for v in neighbors[queue.pop()]:
                if v not in found:
                    found.add(v)
                    unseen.discard(v)
                    queue.append(v)
        components.append(np.array(sorted(found), np.int64))
    endpoints = np.zeros_like(direct)
    for u, v in zip(src, dst):
        linked = direct[:, u] | direct[:, v]
        endpoints[:, u] |= linked
        endpoints[:, v] |= linked
    potential = direct.copy()
    seeds = direct | endpoints
    for component in components:
        potential[:, component] |= seeds[:, component].any(1)[:, None]
    operator = graph.operator_mask.cpu().numpy().astype(bool)
    regions = dict(all_nodes=np.ones((n, nodes), bool), direct_report_nodes=direct,
                   eligible_edge_endpoints=endpoints, candidate_potential=potential,
                   potential_only=potential & ~direct, remaining_nodes=~potential,
                   outside_operator=np.broadcast_to(~operator, (n, nodes)).copy(),
                   no_report_samples=np.broadcast_to(~events['report_present'][:, None], (n, nodes)).copy(),
                   unavailable_relation_samples=np.broadcast_to((events['report_present'] & ~events['relation_available'])[:, None], (n, nodes)).copy(),
                   zero_supported_report_samples=np.broadcast_to((events['report_present'] & events['relation_available'] & ~direct.any(1))[:, None], (n, nodes)).copy())
    for name, indices in roads.items():
        mask = np.zeros(nodes, bool)
        mask[np.asarray(indices, np.int64)] = True
        regions['road:'+name] = mask
    require(np.all(direct.astype(int)+regions['potential_only']+regions['remaining_nodes'] == 1),
            'Contribution partitions do not cover each sample-node exactly once')
    return regions


def merge_mechanisms(total, row):
    for name, value in row.items():
        total[name] = max(total.get(name, 0.), value) if name.endswith('_max') else total.get(name, 0.)+value


def run_probe(models, data, saved_by_arm, graph, roads, output, *, semantic_shuffle_seed=20261013,
              protected_paths=None, protected_before=None):
    output = Path(output)
    require(not output.exists(), 'Use a fresh probe output directory')
    output.mkdir(parents=True)
    paths = protected_paths or {}
    before_files = snapshot_files(paths) if protected_before is None else protected_before
    model_preservation, replays, mechanisms, calls = {}, {}, {}, {case: 0 for case in CASES}
    try:
        require(all(value != '<missing>' for value in before_files.values()), 'A protected input is missing')
        require_preserved(before_files, paths)
        loader = data.loaders['val_loader']
        size, nodes = loader.size, len(data.station_ids)
        rows = [encode_event(sample, nodes) for sample in loader.original_samples]
        events = {key: np.asarray([row[key] for row in rows], dtype=np.float32 if key == 'relation'
                  else np.int64 if key in ('type', 'description') else bool) for key in EVENT_KEYS}
        donors = semantic_permutation(size, semantic_shuffle_seed)
        shuffled = joint_shuffled_payload(events, donors)
        indices = data.audit['counts']['train_samples']+np.arange(size, dtype=np.int64)
        regions = regions_for(events, graph, roads)
        predictions = {case: np.empty((size, 12, nodes), np.float32) for case in CASES}
        target = np.empty((size, 12, nodes), np.float32)
        partitions = []
        expected_partitions = [min(loader.bs, size-i) for i in range(0, size, loader.bs)]
        selected = {arm: best_row(saved_by_arm[arm]) for arm in ('A', 'L', 'S')}
        for arm in ('A', 'L', 'S'):
            require(type(models[arm]) is (MinimalIGSTGNN if arm == 'A' else EventBudgetIGSTGNN),
                    'Unexpected model architecture: '+arm)
            if arm != 'A':
                require(models[arm].event_budget_adapter.mode == training.ARMS[arm], 'Event adapter mode differs from arm')
            require(tensor_state_hash(models[arm].state_dict()) == tensor_state_hash(saved_by_arm[arm]['best_model_state']),
                    'Model is not the saved best checkpoint: '+arm)
        with ExitStack() as stack:
            for arm, model in models.items():
                stack.enter_context(preserve_model(model, model_preservation, arm))
            # Every complete ON replay must pass before the first intervention.
            for arm in ('A', 'L', 'S'):
                position, case, actual_partition = 0, arm+'_ON', []
                print(f'Replaying {arm} best epoch {selected[arm]["epoch"]}: {size} validation rows, batch {loader.bs}', flush=True)
                for batch in loader.get_iterator():
                    row = forward_case(models[arm], arm, batch, data.scaler, case)
                    calls[case] += 1
                    end = position+len(row['prediction'])
                    actual_partition.append(end-position)
                    predictions[case][position:end] = row['prediction']
                    if arm == 'A':
                        target[position:end] = row['target']
                        partitions.append(end-position)
                    else:
                        require(np.array_equal(target[position:end], row['target']), 'ON replay targets differ between arms')
                    mechanisms.setdefault(case, {})
                    merge_mechanisms(mechanisms[case], row['mechanisms'])
                    position = end
                    if position == size or calls[case] % 4 == 0:
                        print(f'{arm} ON replay: {position}/{size}', flush=True)
                require(position == size, 'Incomplete validation replay')
                require(actual_partition == expected_partitions, 'Validation replay changed original batch partition')
                labels = torch.from_numpy(target)
                null = labels.min() if labels.min() < 1 else torch.tensor(0.)
                actual = compute_all_metrics(torch.from_numpy(predictions[case]), labels, null)
                expected = [selected[arm][name] for name in ('validation_mae', 'validation_mape', 'validation_rmse')]
                require(np.allclose(actual, expected, atol=REPLAY_ATOL, rtol=REPLAY_RTOL),
                        f'{arm} saved best ON pooled validation replay failed: expected={expected}, actual={actual}')
                replays[arm] = dict(status='SAVED_POOLED_VALIDATION_REPLAY_PASS', best_epoch=selected[arm]['epoch'],
                    expected=dict(zip(('mae', 'mape', 'rmse'), expected)), actual=dict(zip(('mae', 'mape', 'rmse'), actual)),
                    mask_value=float(null), samples=size, batch_partition=partitions.copy(), atol=REPLAY_ATOL, rtol=REPLAY_RTOL,
                    scope='saved_aggregate_metrics_no_training_time_pointwise_prediction_archive')
            legacy.write_json(output/'replay.json', replays)
            print('All three complete ON replays passed; permitting interventions', flush=True)
            for arm in ('L', 'S'):
                position = 0
                cases = [arm+'_OFF'] if arm == 'L' else ['S_OFF', 'S_SEMANTIC_UNK', 'S_SEMANTIC_SHUFFLE']
                for batch_index, batch in enumerate(loader.get_iterator()):
                    end = position+len(batch['x_data'])
                    require(batch_index < len(expected_partitions)
                            and end-position == expected_partitions[batch_index], 'Intervention changed original batch partition')
                    on = forward_case(models[arm], arm, batch, data.scaler, arm+'_ON')
                    calls[arm+'_ON'] += 1
                    require(np.array_equal(on['prediction'], predictions[arm+'_ON'][position:end]), 'Paired ON differs from verified replay')
                    require(np.array_equal(on['target'], target[position:end]), 'Paired target differs from replay')
                    for case in cases:
                        payload = {key: value[position:end] for key, value in shuffled.items()} if case.endswith('SHUFFLE') else None
                        changed = forward_case(models[arm], arm, batch, data.scaler, case, payload)
                        calls[case] += 1
                        require(np.array_equal(changed['target'], target[position:end]), 'Intervention changed target')
                        require(np.array_equal(changed['report_support'], on['report_support']), 'Intervention changed input relation/presence')
                        predictions[case][position:end] = changed['prediction']
                        mechanisms.setdefault(case, {})
                        merge_mechanisms(mechanisms[case], changed['mechanisms'])
                        restored = forward_case(models[arm], arm, batch, data.scaler, arm+'_ON')
                        calls[arm+'_ON'] += 1
                        require(np.array_equal(restored['prediction'], on['prediction']), 'Final ON restoration differs from paired ON')
                    position = end
                    if position == size or (batch_index+1) % 4 == 0:
                        print(f'{arm} paired interventions + ON restoration: {position}/{size}', flush=True)
                require(position == size, 'Incomplete paired interventions')
        null_value = replays['A']['mask_value']
        require(all(row['mask_value'] == null_value for row in replays.values()), 'Validation mask changed between arms')
        valid = np.isfinite(target) & (target != null_value)
        analysis = summarize(predictions, target, valid, regions, indices)
        legacy.write_json(output/'analysis.json', analysis)
        products = write_report(output, analysis)
        region_definitions = dict(
            direct_report_nodes='float32 abs(D).sum>0 AND report_present AND relation_available; association not actual causal impact',
            eligible_edge_endpoints='both endpoints of internal directed edges with at least one direct-report endpoint',
            candidate_potential='weakly connected internal-graph component envelope seeded by direct nodes and eligible endpoints; not directed downstream reachability or strict prediction support',
            contribution_partitions=['direct_report_nodes', 'potential_only', 'remaining_nodes'],
            overlapping_regions='candidate_potential, eligible_edge_endpoints, roads, outside_operator and sample groups may overlap',
            unavailable_groups='empty groups are unavailable; no_report_samples uses report_present=false, not D zeros',
            masks={name: f'region_{i}' for i, name in enumerate(regions)})
        legacy.write_json(output/'region_definitions.json', region_definitions)
        mechanism_report = dict(scope='local_same_current_ON_state_C_vs_C0_each_own_proposal_not_whole_OFF_trajectory',
            definitions=dict(sums='over_original_validation_batches_spatial_calls_substeps_and_temporal_positions',
                capacity='dimensionless_latent_exchange_budget_not_observed_road_capacity',
                flux='constrained_latent_feature_exchange_not_vehicle_flow',
                eligible_edge_positions='associated_internal_edges_times_batch_temporal_positions_and_latent_channels'),
            cases=mechanisms)
        legacy.write_json(output/'mechanisms.json', mechanism_report)
        np.savez_compressed(output/'paired_predictions.npz', **predictions, target=target, valid=valid,
            sample_indices=indices, station_ids=np.asarray(data.station_ids, np.int64),
            semantic_donor_sample_indices=indices[donors], **{f'region_{i}': value for i, value in enumerate(regions.values())})
        files = require_preserved(before_files, paths)
        legacy.write_json(output/'preservation.json', dict(files=files if paths else dict(status='NOT_REQUESTED'), models=model_preservation))
        engineering = bool(saved_by_arm['A'].get('identity', {}).get('engineering_check', False))
        summary = dict(status='MIN_I_ENGINEERING_PROBE_PASS' if engineering else 'MIN_I_VALIDATION_PROBE_PASS',
            engineering_only=engineering, validation_rows=size, batch_size=loader.bs, batch_partition=partitions,
            best_epochs={arm: row['epoch'] for arm, row in selected.items()},
            saved_pooled_replays_passed=True, final_on_exactly_restored=True, inference_calls=calls,
            semantic_shuffle_seed=semantic_shuffle_seed, semantic_shuffle_scope='joint_type_description_validity_permutation_D_and_targets_unchanged',
            semantic_UNK_scope='unknown_token_sensitivity_may_be_out_of_distribution_not_retrained_L',
            mask_value=null_value, source_artifacts_preserved=bool(paths), model_state_preserved=True,
            test_evaluated=False, optimizer_updates=0, predictive_gain_claim=False,
            all_nodes={case: {name: analysis['cases'][case]['metrics']['all_nodes'][name]
                             for name in ('mae_macro', 'mae_pooled', 'rmse_macro', 'rmse_pooled', 'mape_macro', 'mape_pooled')} for case in CASES},
            comparisons={name: {key: item['metrics']['all_nodes'][key] for key in
                               ('mae_gain_macro', 'mae_gain_pooled', 'gain_positive', 'gain_negative', 'cancellation_fraction')}
                         for name, item in analysis['comparisons'].items()}, products=products,
            output_dir=str(output))
        legacy.write_json(output/'summary.json', summary)
        return summary
    except BaseException as error:
        failed = dict(status='MIN_I_PROBE_FAILED', error_type=type(error).__name__, error=str(error),
                      inference_calls=calls, optimizer_updates=0, test_evaluated=False)
        legacy.write_json(output/'failure.json', failed)
        legacy.write_json(output/'summary.json', failed)
        raise


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, default=training.paper.default_data())
    parser.add_argument('--graph-dir', type=Path, default=training.HERE/'internal_spatial_graph_is11')
    parser.add_argument('--protocol', type=Path, default=training.PROTOCOL)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--device', help='Default: saved device type, cuda:0 or cpu')
    parser.add_argument('--semantic-shuffle-seed', type=int, default=20261013)
    parser.add_argument('--check', action='store_true', help='Existing engineering-check run only; never subsets formal validation')
    args = parser.parse_args(argv)
    require(0 <= args.semantic_shuffle_seed < 2**32, 'Invalid semantic shuffle seed')
    for key in ('run_dir', 'data_dir', 'graph_dir', 'protocol'):
        setattr(args, key, getattr(args, key).resolve())
    args.output_dir = args.output_dir.resolve() if args.output_dir else args.run_dir/('min_i_probe_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6])
    return args


def main(argv=None):
    import logging
    args = parse_args(argv)
    require(args.run_dir.is_dir(), 'Training run directory does not exist')
    identity = legacy.read_json(args.run_dir/'identity.json')
    require(identity.get('schema') == training.SCHEMA, 'Expected a MIN-I published-data training run')
    require(not args.check or identity['engineering_check'], '--check cannot subset a formal validation run')
    args.device = args.device or ('cuda:0' if identity['runtime']['device_type'] == 'cuda' else 'cpu')
    device = torch.device(args.device)
    args.seed, args.threads, args.check = identity['seed'], identity['threads'], identity['engineering_check']
    args.batch_size, args.arms = identity['protocol']['batch_size'], list(training.ARMS)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    require(device.type != 'cuda' or torch.cuda.is_available(), 'CUDA unavailable')
    verify_runtime(identity, device)
    logger = logging.getLogger('min_i_probe')
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler(sys.stdout))
    with ExitStack() as locks:
        locks.enter_context(legacy.run_lock(args.run_dir))
        for arm in ('A', 'L', 'S'):
            require((args.run_dir/arm).is_dir(), 'Required training arm is missing: '+arm)
            locks.enter_context(legacy.run_lock(args.run_dir/arm))
        verify_identity_sources(args, identity)
        paths = protected_files(args, identity)
        before = snapshot_files(paths)
        data = EventPublishedData(args.data_dir, identity['protocol'], logger, check=args.check)
        require(data.audit == legacy.read_json(args.run_dir/'data_audit.json'), 'Published data audit differs from training')
        graph, roads, graph_info = legacy.load_graph(args.graph_dir, data.station_ids, identity['protocol']['graph_sha256'])
        require(graph_info == identity['graph'], 'Candidate graph audit differs from training')
        models, saved, companions = {}, {}, {}
        for arm in ('A', 'L', 'S'):
            saved[arm] = torch.load(args.run_dir/arm/'last.pt', map_location='cpu', weights_only=False)
            training.validate_saved(saved[arm], identity, arm)
            companions[arm] = verify_saved_sidecars(args.run_dir/arm, saved[arm])
            model, _ = training.build_arm(args, arm, len(data.station_ids), graph, device)
            require(tensor_state_hash(model.state_dict()) == identity['initialization']['arms'][arm]['initial_state_sha256'],
                    'Canonical initialization differs from saved certificate: '+arm)
            model.load_state_dict(saved[arm]['best_model_state'], strict=True)
            models[arm] = model.eval()
        print('Source run: '+str(args.run_dir), flush=True)
        print('Probe directory: '+str(args.output_dir), flush=True)
        result = None
        try:
            result = run_probe(models, data, saved, graph, roads, args.output_dir,
                               semantic_shuffle_seed=args.semantic_shuffle_seed, protected_paths=paths, protected_before=before)
            require_preserved(before, paths)
            legacy.write_json(args.output_dir/'environment.json', legacy.environment(device))
            legacy.write_json(args.output_dir/'provenance.json', dict(
                source_run=str(args.run_dir), checkpoint_selection=companions,
                training_runtime=identity['runtime'], best_epoch_gpu_name='not_recorded_in_checkpoint',
                probe_source_sha256={name: sha256(training.HERE/name) for name in (
                    'probe_min_event_budget.py', 'min_event_budget_probe_metrics.py',
                    'run_min_event_budget_probe.sh')},
                semantic_shuffle_seed=args.semantic_shuffle_seed,
                release_and_training_inputs='preservation.json',
                training_identity_unchanged=True))
            result['source_artifacts_preserved'] = True
            legacy.write_json(args.output_dir/'summary.json', result)
        except BaseException as error:
            if result is not None:
                legacy.write_json(args.output_dir/'summary.json', dict(status='MIN_I_PROBE_FAILED', error=str(error)))
            raise
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
