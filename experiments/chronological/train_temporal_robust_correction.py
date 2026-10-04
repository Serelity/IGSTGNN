"""v12o paired fresh GPU adapter training with fit-only temporal excess weights."""

import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time
import traceback

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.chronological import temporal_robust_correction as method
from experiments.chronological import audit_vector_correction_geometry as export

original, geometry = method.original, method.geometry
base, alignment, scope, vector = original.base, original.alignment, original.scope, original.vector
require, identical = geometry.require, export.identical
PROTOCOL = Path(__file__).with_name('temporal_robust_correction_v12o.json')
PROTOCOL_SHA256 = '8cbfe2c038f0a78b107981fb3b83a03ebd8ae05e097016506f9b51df5c8c6854'
FORMAT = 'v12o_temporal_excess_epoch_v1'
POLICY = 'candidate_early_only'


def load_protocol():
    require(base.sha256(PROTOCOL) == PROTOCOL_SHA256, 'Frozen v12o protocol changed')
    protocol = export.decode(PROTOCOL.read_bytes())
    for name, digest in protocol['source_code_sha256'].items():
        require(base.sha256(REPO / name) == digest, f'Frozen producer changed: {name}')
    original.load_protocol()
    require(tuple(protocol['arms']) == method.ARMS and tuple(protocol['objectives']) == method.OBJECTIVES,
            'Paired factors differ from protocol')
    return protocol


def code_hashes(protocol):
    names = list(protocol['source_code_sha256']) + [str(p.relative_to(REPO)) for p in (
        Path(__file__), Path(method.__file__), PROTOCOL, PROTOCOL.with_name('run_temporal_robust_correction.sh'))]
    return {name: base.sha256(REPO / name) for name in names}


def tree_hash(value):
    def canonical(item):
        if isinstance(item, torch.Tensor):
            return {'tensor_sha256': alignment.tensor_hash({'value': item})}
        if isinstance(item, dict):
            return [[type(k).__name__, str(k), canonical(v)] for k, v in sorted(item.items(), key=lambda kv: str(kv[0]))]
        if isinstance(item, (tuple, list)):
            return [type(item).__name__, [canonical(v) for v in item]]
        return item
    return hashlib.sha256(json.dumps(canonical(value), sort_keys=True, allow_nan=False).encode()).hexdigest()


def restore_fit(path, expected, initial, baseline, protocol, reference):
    saved = torch.load(export.ordinary(path), map_location='cpu', weights_only=True)
    require(saved.get('format_version') == FORMAT and identical(saved.get('identity'), expected),
            'Recovery identity mismatch')
    epoch, history = saved['epoch'], saved['history']
    require(type(epoch) is int and 0 <= epoch <= protocol['training']['epochs'] and len(history) == epoch,
            'Recovery epoch/history mismatch')
    pi, _ = method.reference_weights(reference, protocol['weighting'])
    q = pi.copy()
    best = {'epoch': 0, 'selection_metrics': baseline}
    steps = (expected['fit_samples'] + protocol['training']['batch_size'] - 1) // protocol['training']['batch_size']
    for number, row in enumerate(history, 1):
        require(type(row['epoch']) is int and row['epoch'] == number and identical(row['q_before'], q.tolist()),
                'Recovery contiguous epoch/q history mismatch')
        updated = method.update_weights(q, pi, row['fit_groups'], reference, protocol['weighting'])
        q = updated if expected['objective'] == 'temporal_excess' else pi.copy()
        require(identical(row['q_after'], q.tolist()), 'Recovery weight update mismatch')
        decision = scope.selection_decision(row['selection'], baseline, best['selection_metrics'], protocol)
        if decision['replace_best']:
            best = {'epoch': number, 'selection_metrics': row['selection']}
        require(identical(row['decision'], {**decision, 'best_epoch': best['epoch']}), 'Recovery selection mismatch')
        require(type(row['training']['optimizer_steps']) is int and row['training']['optimizer_steps'] == steps,
                'Recovery optimizer step budget mismatch')
    require(identical(saved['q_next'], q.tolist()), 'Recovery next q mismatch')
    original.previous.validate_state(saved['adapter_state'], initial)
    require(alignment.tensor_hash(saved['adapter_state']) ==
            (history[-1]['adapter_state_sha256'] if history else alignment.tensor_hash(initial)), 'Recovery current state mismatch')
    require(identical({k: saved['best'][k] for k in best}, best), 'Recovery best selection metadata mismatch')
    original.previous.validate_state(saved['best']['state'], initial)
    best_hash = history[best['epoch'] - 1]['adapter_state_sha256'] if best['epoch'] else alignment.tensor_hash(initial)
    require(alignment.tensor_hash(saved['best']['state']) == best_hash, 'Recovery best state mismatch')
    original.validate_adam_recovery(saved['optimizer_state'], initial, protocol['training'], epoch * steps)
    require(tree_hash(saved['optimizer_state']) == (history[-1]['optimizer_sha256'] if history else saved['identity']['initial_optimizer_sha256']),
            'Recovery optimizer content mismatch')
    for key, template in (('rng_cpu', torch.get_rng_state()), ('rng_cuda', torch.cuda.get_rng_state(expected['device'])
                         if torch.device(expected['device']).type == 'cuda' else None)):
        value = saved[key]
        require((template is None and value is None) or (isinstance(value, torch.Tensor) and template is not None
                and value.dtype == template.dtype and value.shape == template.shape), 'Recovery RNG shape/device mismatch')
    require(tree_hash([saved['rng_cpu'], saved['rng_cuda']]) == saved['rng_sha256'], 'Recovery RNG content mismatch')
    return saved


def fit(model, arm, objective, seed, datasets, plan, baseline, protocol, groups, reference,
        device, directory, progress, identity_hash, resume_from=None):
    base.set_seed(seed)
    native_hash = base.backbone_hash(model)
    # Preserve original RNG consumption: DataLoader probe BEFORE adapter creation.
    with torch.no_grad():
        raw = next(iter(base.loader(datasets['incident_full'], plan['fit']['indices']['incident_full'], 2)))
        batch = base.device_batch(raw, device)
        a = model(batch['x'], incident_data=batch['incident'])
        adapter = vector.attach_adapter(model, arm, protocol['training']['node_hidden_width'])
        require(torch.equal(a, model(batch['x'], incident_data=batch['incident']))
                and torch.equal(a, original.native_prediction(model, batch)), 'Initial adapter must equal A exactly')
    model.icsf_module.clear_observations()
    initial = base.cpu_tree(adapter.state_dict())
    settings = protocol['training']
    optimizer = torch.optim.Adam(adapter.parameters(), lr=settings['learning_rate'], eps=settings['adam_eps'],
                                 weight_decay=settings['weight_decay'])
    pi, scale = method.reference_weights(reference, protocol['weighting'])
    q = pi.copy()
    identity = {'arm': arm, 'objective': objective, 'seed': seed, 'protocol_sha256': PROTOCOL_SHA256,
        'run_identity_sha256': identity_hash, 'backbone_sha256': native_hash, 'device': str(device),
        'initial_adapter_sha256': alignment.tensor_hash(initial), 'initial_optimizer_sha256': tree_hash(optimizer.state_dict()),
        'fit_samples': len(groups), 'groups': [[i, g] for i, g in groups.items()], 'reference': reference,
        'pi': pi.tolist(), 'baseline_scale': scale, 'training': settings, 'policy': POLICY}
    best = {'epoch': 0, 'selection_metrics': copy.deepcopy(baseline), 'state': copy.deepcopy(initial)}
    history, start, recovery = [], 1, {'method': 'fresh_fit'}
    if resume_from is not None:
        source = resume_from / 'last_adapter.pt'
        if source.exists():
            saved = restore_fit(source, identity, initial, baseline, protocol, reference)
            adapter.load_state_dict(saved['adapter_state'], strict=True)
            optimizer.load_state_dict(saved['optimizer_state'])
            history, best, start = saved['history'], saved['best'], saved['epoch'] + 1
            q = np.asarray(saved['q_next'])
            torch.set_rng_state(saved['rng_cpu'])
            if device.type == 'cuda':
                torch.cuda.set_rng_state(saved['rng_cuda'], device)
            recovery = {'method': 'epoch_boundary', 'epoch': saved['epoch'], 'source_sha256': base.sha256(source)}
            progress('fit_restored', epoch=saved['epoch'], selected_epoch=best['epoch'])
        else:
            require(not (resume_from / 'fit_summary.json').exists(), 'Completed recovery trajectory is missing checkpoint')
            recovery = {'method': 'fresh_fit_no_committed_epoch'}

    def publish(epoch):
        cpu, cuda = torch.get_rng_state(), torch.cuda.get_rng_state(device) if device.type == 'cuda' else None
        base.save_checkpoint(directory / 'last_adapter.pt', base.cpu_tree({
            'format_version': FORMAT, 'identity': identity, 'epoch': epoch, 'adapter_state': adapter.state_dict(),
            'optimizer_state': optimizer.state_dict(), 'q_next': q.tolist(), 'best': best, 'history': history,
            'rng_cpu': cpu, 'rng_cuda': cuda, 'rng_sha256': tree_hash([cpu, cuda])}))
        export.write_json(directory / 'history.json', history)

    publish(start - 1)
    runtime = {}
    with method.checked_device(model, device, progress) as device_checks:
        for epoch in range(start, settings['epochs'] + 1):
            before = q.copy()
            training = method.train_epoch(model, adapter, optimizer, datasets['incident_full'],
                plan['fit']['indices']['incident_full'], settings, device, seed, epoch, progress,
                objective, groups, q, pi)
            # The q update accepts only this entire fit-cohort record.
            fit_record = original.evaluate_policies(model, datasets['incident_full'], plan['fit']['indices']['incident_full'],
                settings['evaluation_batch_size'], device,
                lambda stage, **fields: progress(stage, phase='fit', epoch=epoch, cohort='incident_full', **fields),
                policies=(POLICY,), telemetry=runtime)[POLICY]
            current_groups = method.group_statistics(fit_record, groups)
            updated = method.update_weights(q, pi, current_groups, reference, protocol['weighting'])
            q = updated if objective == 'temporal_excess' else pi.copy()
            current = {}
            for cohort in base.COHORTS:
                record = original.evaluate_policies(model, datasets[cohort], plan['selection']['indices'][cohort],
                    settings['evaluation_batch_size'], device,
                    lambda stage, **fields: progress(stage, phase='selection', epoch=epoch, cohort=cohort, **fields),
                    policies=(POLICY,), telemetry=runtime)[POLICY]
                current[cohort] = original.metrics(record)
            decision = scope.selection_decision(current, baseline, best['selection_metrics'], protocol)
            if decision['replace_best']:
                best = {'epoch': epoch, 'selection_metrics': copy.deepcopy(current), 'state': base.cpu_tree(adapter.state_dict())}
            base.assert_backbone(model, native_hash)
            history.append({'epoch': epoch, 'q_before': before.tolist(), 'q_after': q.tolist(),
                'fit_groups': current_groups, 'training': training, 'selection': current,
                'decision': {**decision, 'best_epoch': best['epoch']},
                'adapter_state_sha256': alignment.tensor_hash(adapter.state_dict()),
                'optimizer_sha256': tree_hash(optimizer.state_dict())})
            publish(epoch)
            progress('epoch_complete', epoch=epoch, selected_epoch=best['epoch'], q_next=q.tolist(),
                     cell_multipliers=(q / pi).tolist(), fit_group_mae=current_groups['mae'],
                     selection_early_mae=current['incident_full']['mae']['candidate_h1_h6'],
                     optimizer_steps_this_epoch=training['optimizer_steps'])
    base.assert_backbone(model, native_hash)
    state_hash = alignment.tensor_hash(best['state'])
    base.save_checkpoint(directory / 'selected_adapter.pt', {'identity': identity, 'epoch': best['epoch'],
        'adapter_state': best['state'], 'adapter_state_sha256': state_hash, 'selection_metrics': best['selection_metrics']})
    detail = {'arm': arm, 'objective': objective, 'seed': seed, 'epochs': len(history),
        'optimizer_steps': sum(row['training']['optimizer_steps'] for row in history),
        'optimizer_steps_this_invocation': sum(row['training']['optimizer_steps'] for row in history[start - 1:]),
        'trainable_parameters': sum(p.numel() for p in adapter.parameters()),
        'selected_epoch': best['epoch'], 'selection_metrics': best['selection_metrics'],
        'adapter_state_sha256': state_hash, 'initial_adapter_sha256': alignment.tensor_hash(initial),
        'epoch1_adapter_sha256': history[0]['adapter_state_sha256'], 'epoch1_optimizer_sha256': history[0]['optimizer_sha256'],
        'checkpoint_sha256': base.sha256(directory / 'selected_adapter.pt'),
        'backbone_unchanged': True, 'initial_prediction_exactly_A': True, 'recovery': recovery,
        'pi': pi.tolist(), 'baseline_scale': scale, 'q_next': q.tolist(), 'device_checks': device_checks,
        'selection_accounting': {'trajectory_epochs': len(history), 'fallback_to_A': best['epoch'] == 0,
            'eligible_epochs': sum(row['decision']['eligible'] for row in history),
            'best_updates': sum(row['decision']['replace_best'] for row in history),
            'guard_failures': {key: sum(not row['decision']['protection_checks'][key] for row in history)
                               for key in history[0]['decision']['protection_checks']}},
        'evaluation_runtime': runtime}
    export.write_json(directory / 'fit_summary.json', detail)
    return detail


def metadata_for(record, cohort, manifests):
    positive = {int(row['sample_index']): row for row in manifests[0]}
    controls = {int(row['positive_sample_index']): row for row in manifests[1 if cohort == 'primary_control' else 2]}
    rows = []
    for sample in record['ids']:
        pos = positive[int(sample)]
        row = pos if cohort in ('incident_full', 'incident') else controls[int(sample)]
        rows.append({'incident_ids': pos['incident_id'], 'positive_t0': pos['t0'],
            'cohort_t0': row['t0'] if cohort in ('incident_full', 'incident') else row['candidate_t0'],
            'support_start': row['support_start'], 'support_end_exclusive': row['support_end_exclusive']})
    return {key: np.asarray([row[key] for row in rows]) for key in geometry.META[2:]}


def evaluate_endpoint(model, dataset, indices, batch_size, device, cohort, manifests, progress):
    """Stream exact protected predictions and signed geometry; no raw full graph cache."""
    pieces, windows = {}, {}
    with torch.no_grad():
        for step, raw in enumerate(base.loader(dataset, indices, batch_size)):
            batch = base.device_batch(raw, device)
            a = original.native_prediction(model, batch) * dataset.scaler['std'] + dataset.scaler['mean']
            p = model(batch['x'], incident_data=batch['incident']) * dataset.scaler['std'] + dataset.scaler['mean']
            p = scope.project_prediction(p, a, batch['candidate_mask'])
            active = base.masks_for(p, batch['candidate_mask'])[base.REGIONS.index('candidate_h1_h6')]
            require(torch.equal(p[~active], a[~active]), 'Protected outputs differ from A')
            record = {**base.statistics(p, batch), 'ids': raw['positive_sample_index'].numpy(),
                'source_indices': raw['source_index'].numpy(), 'candidate_mask': raw['candidate_mask'].numpy()}
            packed = export.pack_predictions(a.cpu().numpy(), p.cpu().numpy(), raw['y_flow'].numpy(),
                raw['y_valid'].numpy(), record, metadata_for(record, cohort, manifests))
            window = geometry.window_geometry(packed)
            column = base.REGIONS.index('candidate_h1_h6')
            require(np.array_equal(window['valid_counts'], record['counts'][:, column]), 'Geometry support mismatch')
            geometry.close(window['p_error_sums'], record['errors'][:, column], 'Geometry P errors mismatch')
            for key, value in record.items():
                pieces.setdefault(key, []).append(value)
            for key, value in window.items():
                if key != 'category_names':
                    windows.setdefault(key, []).append(value)
            model.icsf_module.clear_observations()
            if step % 10 == 0:
                progress('endpoint_evaluation_progress', cohort=cohort, batches_completed=step + 1)
    return ({**{k: np.concatenate(v) for k, v in pieces.items()}, 'regions': base.REGIONS},
            {**{k: np.concatenate(v) for k, v in windows.items()}, 'category_names': np.asarray(geometry.CATEGORIES)})


def name_for(arm, objective, seed):
    return f'{arm}_{objective}_s{seed}'


def run(data_dir, primary_dir, secondary_dir, checkpoint, output, device='cuda:0', check=False, resume_from=None):
    frozen, inherited = load_protocol(), base.load_protocol()
    effective = copy.deepcopy(frozen)
    effective['selection'] = inherited['selection']
    if check:
        effective['seeds'] = frozen['check']['seeds']
        effective['training']['epochs'] = frozen['check']['epochs']
    readonly = [data_dir, primary_dir, secondary_dir, checkpoint]
    if resume_from is not None:
        resume_from = export.ordinary(resume_from)
        require(not check and resume_from.is_dir() and not any(p.is_symlink() for p in resume_from.rglob('*')),
                'Recovery requires a regular full-run source; source must contain no symlinks')
        readonly.append(resume_from)
    output, partial = export.new_output(output, readonly)
    started = time.monotonic()
    device = torch.device(device)
    # Normalize cuda shorthand to the actual device so checks are unambiguous.
    if device.type == 'cuda' and device.index is None:
        device = torch.device('cuda', torch.cuda.current_device())
    def progress(stage, **fields):
        event = {'stage': stage, 'elapsed_seconds': time.monotonic() - started,
                 'memory': base.memory_snapshot(device), **fields}
        export.write_json(partial / 'progress.json', event)
        print(json.dumps(event, allow_nan=False), flush=True)
    try:
        torch.set_num_threads(3)
        base.configure_determinism(device)
        if device.type == 'cuda':
            require(torch.cuda.is_available(), 'CUDA unavailable: allocate V100 and activate igstgnn')
            torch.empty(1, device=device)
            torch.cuda.reset_peak_memory_stats(device)
        codes = code_hashes(frozen)
        progress('verifying_inputs', device=str(device))
        baseline_info, hashes = base.mechanisms.verify_inputs(data_dir, primary_dir, secondary_dir, checkpoint,
            base.mechanisms.load_protocol(base.mechanisms.PROTOCOL))
        manifests = [base.read_csv(Path(root) / name) for root, name in (
            (data_dir, 'train_manifest.csv'), (primary_dir, 'train_control_manifest.csv'),
            (secondary_dir, 'train_second_control_manifest.csv'))]
        plan = base.make_plan(*manifests, inherited)
        samples = {p: {c: len(v) for c, v in x['indices'].items()} for p, x in plan.items()}
        require(samples == frozen['expected_phase_samples'], 'Frozen phase sample budget changed')
        groups = method.fit_groups(manifests[0], plan['fit']['indices']['incident_full'], frozen)
        require(np.bincount(list(groups.values()), minlength=4).tolist() == frozen['fit_groups']['expected_windows'],
                'Frozen fit block window counts changed')
        export.write_json(partial / 'eligibility.json', plan)
        if check:
            selected = {g: [i for i, v in groups.items() if v == g][:frozen['check']['fit_windows_per_group']] for g in range(4)}
            plan['fit']['indices']['incident_full'] = sorted(i for values in selected.values() for i in values)
            for phase in ('selection', 'audit'):
                plan[phase]['indices'] = {c: v[:frozen['check']['other_windows_per_cohort']] for c, v in plan[phase]['indices'].items()}
            groups = method.fit_groups(manifests[0], plan['fit']['indices']['incident_full'], frozen)
        effective_indices = {p: x['indices'] for p, x in plan.items()}
        export.write_json(partial / 'effective_indices.json', effective_indices)
        identity = {'protocol_sha256': PROTOCOL_SHA256, 'engineering_check': check, 'inputs': hashes,
            'checkpoint_sha256': base.sha256(checkpoint), 'code_sha256': codes, 'indices': effective_indices,
            'groups': [[i, g] for i, g in groups.items()], 'training': effective['training'], 'seeds': effective['seeds'],
            'device': str(device), 'torch': str(torch.__version__), 'numpy': np.__version__, 'python': sys.version,
            'cuda_runtime': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
            'gpu_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else None}
        if resume_from:
            require(identical(export.decode((resume_from / 'run_identity.json').read_bytes()), identity),
                    'Recovery run input/code/device/protocol identity changed')
        export.write_json(partial / 'run_identity.json', identity)
        identity_hash = base.sha256(partial / 'run_identity.json')
        datasets = base.make_datasets(data_dir, primary_dir, secondary_dir, baseline_info)
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        def native():
            model = base.make_model(Path(data_dir), len(datasets['incident_full'].station_ids), device, 'fixed')
            model.load_state_dict(state, strict=True)
            require(sum(p.numel() for p in model.parameters()) == baseline_info['checkpoint']['parameters'], 'A size changed')
            return model.eval().requires_grad_(False)
        def release(model):
            if hasattr(model.icsf_module, 'clear_observations'):
                model.icsf_module.clear_observations()
            model.cpu()
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        (partial / 'A').mkdir()
        def evaluate_reference(model, phase):
            if phase == 'audit':
                require((partial / 'selected_endpoints_frozen.json').is_file(), 'Audit access before all endpoints froze')
            records = {}
            with method.checked_device(model, device, progress):
                for cohort, indices in plan[phase]['indices'].items():
                    record = original.evaluate(model, datasets[cohort], indices, effective['training']['evaluation_batch_size'],
                        device, lambda stage, **fields: progress(stage, phase=phase, cohort=cohort, endpoint='A', **fields))
                    base.save_arrays(partial / 'A' / f'{phase}_{cohort}.npz', record)
                    records[cohort] = record
            return records
        model = native()
        backbone = base.backbone_hash(model)
        reference = {p: evaluate_reference(model, p) for p in ('fit', 'selection')}
        base.assert_backbone(model, backbone)
        release(model)
        del model
        reference_groups = method.group_statistics(reference['fit']['incident_full'], groups)
        export.write_json(partial / 'fit_group_reference.json', reference_groups)
        baseline_selection = {c: original.metrics(r) for c, r in reference['selection'].items()}
        runs, endpoints = {}, {}
        for seed in effective['seeds']:
            for arm in method.ARMS:
                paired = []
                for objective in method.OBJECTIVES:
                    name = name_for(arm, objective, seed)
                    directory = partial / name
                    directory.mkdir()
                    model = native()
                    progress('fit_started', endpoint=name, epochs=effective['training']['epochs'])
                    detail = fit(model, arm, objective, seed, datasets, plan, baseline_selection, effective, groups,
                        reference_groups, device, directory, lambda stage, **fields: progress(stage, endpoint=name, **fields),
                        identity_hash, resume_from / name if resume_from else None)
                    runs[name] = detail
                    paired.append(detail)
                    endpoints[f'{name}/selected_adapter.pt'] = detail['checkpoint_sha256']
                    release(model)
                    del model
                for key in ('initial_adapter_sha256', 'epoch1_adapter_sha256', 'epoch1_optimizer_sha256'):
                    require(paired[0][key] == paired[1][key], f'Paired initialization/first epoch mismatch: {key}')
        fits = len(effective['seeds']) * len(method.ARMS) * len(method.OBJECTIVES)
        steps_per_epoch = (len(groups) + effective['training']['batch_size'] - 1) // effective['training']['batch_size']
        budget = {'fits': len(runs), 'trajectory_epochs': sum(r['epochs'] for r in runs.values()),
            'selected_endpoints': len(endpoints), 'optimizer_steps': sum(r['optimizer_steps'] for r in runs.values()),
            'optimizer_steps_this_invocation': sum(r['optimizer_steps_this_invocation'] for r in runs.values())}
        require(budget['fits'] == budget['selected_endpoints'] == fits
                and budget['trajectory_epochs'] == fits * effective['training']['epochs']
                and budget['optimizer_steps'] == fits * effective['training']['epochs'] * steps_per_epoch,
                'Incomplete paired training budget')
        for path, digest in endpoints.items():
            require(base.sha256(partial / path) == digest, 'Selected endpoint changed before freezing')
        export.write_json(partial / 'selected_endpoints_frozen.json', endpoints)
        frozen_hash = base.sha256(partial / 'selected_endpoints_frozen.json')
        progress('all_endpoints_frozen_before_audit', **budget)
        model = native()
        reference['audit'] = evaluate_reference(model, 'audit')
        base.assert_backbone(model, backbone)
        release(model)
        del model
        comparisons, rows, diagnostic = {}, [], {}
        times = {int(row['sample_index']): row['t0'] for row in manifests[0]}
        # Reuse v12n's reporting band only; weights/calendars follow the new v12o protocol.
        geo_protocol = {**frozen, 'replay': export.load_protocol()['replay']}
        for seed in effective['seeds']:
            learned = {p: {} for p in plan}
            for arm in method.ARMS:
                for objective in method.OBJECTIVES:
                    name = name_for(arm, objective, seed)
                    detail = runs[name]
                    path = f'{name}/selected_adapter.pt'
                    require(base.sha256(partial / path) == endpoints[path], 'Frozen endpoint changed')
                    saved = torch.load(partial / path, map_location='cpu', weights_only=True)
                    require(saved['epoch'] == detail['selected_epoch'] and saved['identity']['run_identity_sha256'] == identity_hash
                            and identical(saved['selection_metrics'], detail['selection_metrics']), 'Selected metadata mismatch')
                    model = native()
                    adapter = vector.attach_adapter(model, arm, effective['training']['node_hidden_width'])
                    original.previous.validate_state(saved['adapter_state'], adapter.state_dict())
                    require(alignment.tensor_hash(saved['adapter_state']) == saved['adapter_state_sha256'] == detail['adapter_state_sha256'],
                            'Selected tensor hash mismatch')
                    adapter.load_state_dict(saved['adapter_state'], strict=True)
                    model.requires_grad_(False).eval()
                    full_hash = alignment.tensor_hash(model.state_dict())
                    detail['phase_metrics'], diagnostic[name] = {}, {}
                    with method.checked_device(model, device, progress):
                        for phase in plan:
                            records, windows = {}, {}
                            for cohort, indices in plan[phase]['indices'].items():
                                record, window = evaluate_endpoint(model, datasets[cohort], indices,
                                    effective['training']['evaluation_batch_size'], device, cohort, manifests,
                                    lambda stage, **fields: progress(stage, endpoint=name, phase=phase, **fields))
                                anchor = reference[phase][cohort]
                                for key in ('ids', 'source_indices', 'counts', 'prediction_counts', 'candidate_mask', 'regions'):
                                    require(np.array_equal(record[key], anchor[key]), 'Endpoint identity/support mismatch')
                                for region in ('candidate_h7_h12', 'noncandidate_all'):
                                    j = base.REGIONS.index(region)
                                    require(np.array_equal(record['errors'][:, j], anchor['errors'][:, j]), 'Protected error changed')
                                if saved['epoch'] == 0:
                                    require(np.array_equal(record['errors'], anchor['errors']), 'Epoch-zero endpoint must equal A')
                                geometry.close(window['a_error_sums'], anchor['errors'][:, base.REGIONS.index('candidate_h1_h6')],
                                               'Geometry A errors mismatch')
                                base.save_arrays(partial / name / f'{phase}_{cohort}.npz', record)
                                base.save_arrays(partial / name / f'{phase}_{cohort}_geometry.npz', window)
                                records[cohort], windows[cohort] = record, window
                            learned[phase][method.endpoint(arm, objective)] = records
                            detail['phase_metrics'][phase] = {c: original.metrics(r) for c, r in records.items()}
                            if not check:
                                weeks, weights = geometry.calendar(frozen['periods'][phase], frozen['bootstrap'])
                                diagnostic[name][phase] = {c: geometry.analyze_windows(w, weeks, weights, geo_protocol)[0] for c, w in windows.items()}
                                if phase != 'fit':
                                    matched, deletions = geometry.analyze_matched(windows, weeks, weights, geo_protocol)
                                    diagnostic[name][phase]['matched'] = matched
                                    export.write_csv(partial / name / f'{phase}_matched_deletion.csv', deletions)
                    require(identical(detail['phase_metrics']['selection'], detail['selection_metrics']), 'Selection replay changed')
                    require(alignment.tensor_hash(model.state_dict()) == full_hash, 'Endpoint evaluation changed model')
                    base.assert_backbone(model, backbone)
                    release(model)
                    del model
            comparisons[str(seed)] = {}
            if not check:
                for phase in plan:
                    analysis, lines, weekly = method.compare_phase(reference[phase], learned[phase], times, phase, frozen)
                    comparisons[str(seed)][phase] = analysis
                    rows.extend({'seed': seed, **row} for row in lines)
                    base.save_arrays(partial / f'{phase}_weekly_s{seed}.npz', weekly)
        if rows:
            export.write_csv(partial / 'comparisons.csv', rows)
        require(code_hashes(frozen) == codes, 'Code changed during run')
        require(base.sha256(partial / 'selected_endpoints_frozen.json') == frozen_hash
                and all(base.sha256(partial / name) == digest for name, digest in endpoints.items()), 'Frozen endpoint changed during evaluation')
        summary = {'status': 'ENGINEERING_CHECK_PASS' if check else 'TEMPORAL_EXCESS_PAIRED_COMPARISON_COMPLETE',
            'scientific_status': 'NOT_EVALUATED_ENGINEERING_ONLY' if check else 'REUSED_DEVELOPMENT_COMPARISON_REQUIRES_REVIEW',
            'protocol_id': frozen['protocol_id'], 'protocol_sha256': PROTOCOL_SHA256, 'engineering_check': check,
            'frozen_protocol': frozen, 'effective_training': effective['training'], **frozen['information_boundary'],
            'all_endpoints_frozen_before_audit': True, 'paired_initialization_and_epoch1_exact': True,
            'main_training_ready': False, 'budget': budget, 'runs': runs, 'selected_endpoints_frozen': endpoints,
            'baseline': {p: {c: original.metrics(r) for c, r in values.items()} for p, values in reference.items()},
            'phase_samples': {p: {c: len(v) for c, v in x['indices'].items()} for p, x in plan.items()},
            'fit_group_reference': reference_groups, 'fit_group_windows': np.bincount(list(groups.values()), minlength=4).tolist(),
            'phase_comparisons': comparisons, 'geometry': diagnostic, 'inputs': hashes, 'code_sha256': codes,
            'environment': {**{k: identity[k] for k in ('device', 'torch', 'numpy', 'python', 'cuda_runtime', 'cudnn', 'gpu_name')},
                'host': socket.gethostname(), 'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
                'git_head': base.subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()},
            'memory_at_completion': base.memory_snapshot(device), 'elapsed_seconds': time.monotonic() - started,
            'recovery_source': str(resume_from) if resume_from else None,
            'outputs': {str(p.relative_to(partial)): base.sha256(p) for p in partial.rglob('*') if p.is_file()
                        and p.name not in ('progress.json', 'failure.json')}}
        export.write_json(partial / 'summary.json', summary)
        require(not output.exists() and not output.is_symlink(), 'Final destination appeared; preserve partial')
        partial.rename(output)
    except BaseException as error:
        export.write_json(partial / 'failure.json', {'error': str(error), 'traceback': traceback.format_exc()})
        raise
    report(summary)
    print('Saved v12o result:', output / 'summary.json', flush=True)
    return summary


def report(summary):
    print('status:', summary['status'])
    print('scientific_status:', summary['scientific_status'])
    print('protocol_sha256:', summary['protocol_sha256'])
    print('budget:', json.dumps(summary['budget']))
    print('environment:', json.dumps(summary['environment']))
    print('Training: fresh adapter gradients and Adam updates; frozen original backbone.')
    print('Positive gain = reference MAE - compared MAE. Audit is reused development, not independent confirmation.')
    for name, detail in summary['runs'].items():
        print('\nendpoint:', name, 'selected_epoch:', detail['selected_epoch'], 'q_next:', detail['q_next'])
        print('selection_accounting:', json.dumps(detail['selection_accounting']))
        for phase in ('fit', 'selection', 'audit'):
            a = summary['baseline'][phase]['incident_full']['mae']['candidate_h1_h6']
            p = detail['phase_metrics'][phase]['incident_full']['mae']['candidate_h1_h6']
            print(phase, 'early_gain_vs_A:', a - p if a is not None and p is not None else None)
            if not summary['engineering_check']:
                item = summary['geometry'][name][phase]['incident_full']['estimands']
                for estimand, value in item.items():
                    print(phase, estimand, 'G/S/O:', json.dumps(value))
        if not summary['engineering_check']:
            print('audit_matched:', json.dumps(summary['geometry'][name]['audit']['matched']))
    for seed, phases in summary['phase_comparisons'].items():
        if 'audit' in phases:
            for arm in method.ARMS:
                effect = phases['audit']['results']['incident_full']['candidate_h1_h6']['comparisons'][arm + '__temporal_vs_erm']
                print('PRIMARY', seed, arm, 'temporal_vs_erm:', json.dumps(effect))
    print('Memory:', json.dumps(summary['memory_at_completion']))
    print('No automatic promotion, no held-out test access, no weighting-rate search.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    run_parser = sub.add_parser('run')
    for name in ('data-dir', 'primary-control-dir', 'secondary-control-dir', 'checkpoint', 'output'):
        run_parser.add_argument('--' + name, type=Path, required=True)
    run_parser.add_argument('--device', default='cuda:0')
    run_parser.add_argument('--check', action='store_true')
    run_parser.add_argument('--resume-from', type=Path)
    sub.add_parser('report').add_argument('summary', type=Path)
    args = parser.parse_args()
    if args.action == 'report':
        report(export.decode(args.summary.read_bytes()))
    else:
        run(args.data_dir, args.primary_control_dir, args.secondary_control_dir, args.checkpoint,
            args.output, args.device, args.check, args.resume_from)


if __name__ == '__main__':
    main()
