"""v12m: one early-loss trajectory, two output policies and three frozen output paths."""

import argparse
import copy
import csv
import gc
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
from experiments.chronological import train_vector_objective_alignment as previous
from experiments.chronological import vector_output_scope as scope

alignment, vector, base = previous.alignment, previous.vector, previous.base
evaluate, metrics = previous.evaluate, previous.metrics
PROTOCOL = Path(__file__).with_name('vector_output_scope_v12m.json')
PROTOCOL_SHA256 = 'ac3478f320938a2e61af9529704e76b32cc68d3981d3a935e8851174a7782998'
RECOVERY_FORMAT = 'v12m_shared_output_scope_v1'


def load_protocol():
    if base.sha256(PROTOCOL) != PROTOCOL_SHA256:
        raise ValueError('Frozen v12m protocol changed')
    frozen = json.loads(PROTOCOL.read_text())
    for path, key in ((previous.PROTOCOL, 'v12k_protocol_sha256'),
                      (base.PROTOCOL, 'v12c_protocol_sha256'),
                      (vector.PROTOCOL, 'v12f_protocol_sha256')):
        if base.sha256(path) != frozen[key]:
            raise ValueError('Inherited v12m protocol changed')
    previous.load_protocol()
    if (tuple(frozen['arms']) != scope.ARMS or tuple(frozen['policies']) != scope.POLICIES
            or tuple(frozen['output_paths']) != scope.OUTPUT_PATHS
            or frozen['numerical_checks']['additive_error_rtol'] != scope.ADDITIVITY_RTOL
            or frozen['numerical_checks']['additive_error_atol'] != scope.ADDITIVITY_ATOL):
        raise ValueError('Output scope implementation disagrees with frozen protocol')
    return frozen


def native_prediction(model, batch):
    """Same checkpoint and incident context, temporarily bypassing only the adapter."""
    wrapper = model.icsf_module
    if not isinstance(wrapper, vector.StateInteractionICSF):
        raise ValueError('Native bypass requires a v12f vector wrapper')
    try:
        model.icsf_module = wrapper.base
        with torch.no_grad():
            return model(batch['x'], incident_data=batch['incident'])
    finally:
        model.icsf_module = wrapper


def evaluate_policies(model, dataset, indices, batch_size, device, progress=lambda *a, **k: None,
                      policies=scope.POLICIES, telemetry=None):
    """Compute A and the vector once per batch; score only the requested outputs."""
    policies = tuple(policies)
    if not policies or len(set(policies)) != len(policies) or not set(policies) <= set(scope.POLICIES):
        raise ValueError('Unknown/duplicate/empty output policy')
    model.eval()
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)
    started = time.monotonic()
    pieces = {p: {} for p in policies}
    batches, cells = 0, 0
    with torch.no_grad():
        for step, raw in enumerate(base.loader(dataset, indices, batch_size)):
            batch = base.device_batch(raw, device)
            anchor = native_prediction(model, batch)
            prediction = model(batch['x'], incident_data=batch['incident'])
            anchor = anchor * dataset.scaler['std'] + dataset.scaler['mean']
            prediction = prediction * dataset.scaler['std'] + dataset.scaler['mean']
            projected = scope.project_prediction(prediction, anchor, batch['candidate_mask'])
            mask = base.masks_for(prediction, batch['candidate_mask'])[base.REGIONS.index('candidate_h1_h6')]
            if not torch.equal(projected[mask], prediction[mask]) or not torch.equal(projected[~mask], anchor[~mask]):
                raise ValueError('Output projection violated exact prediction support')
            for policy in policies:
                # Never calculate the unrequested unrestricted(P) error against Y.
                values = base.statistics(prediction if policy == 'unrestricted' else projected, batch)
                values.update(ids=raw['positive_sample_index'].numpy(),
                              source_indices=raw['source_index'].numpy(),
                              candidate_mask=raw['candidate_mask'].numpy())
                for key, value in values.items():
                    pieces[policy].setdefault(key, []).append(value)
            batches += 1
            cells += prediction.numel()
            model.icsf_module.clear_observations()
            if step % 10 == 0:
                progress('evaluation_progress', batches_completed=step + 1, samples_total=len(indices))
    if not batches:
        raise ValueError('Cannot evaluate an empty output-scope dataset')
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)
    if telemetry is not None:
        for key, value in {'native_forward_batches': batches, 'modified_forward_batches': batches,
                           'checked_prediction_cells': cells, 'elapsed_seconds': time.monotonic() - started}.items():
            telemetry[key] = telemetry.get(key, 0) + value
    return {policy: {**{key: np.concatenate(values) for key, values in pieces[policy].items()},
                     'regions': base.REGIONS} for policy in policies}


def _same_optimizer_setting(left, right):
    """Compare saved settings without accepting bool/int or sequence substitutions."""
    if type(left) is not type(right):
        return False
    if isinstance(right, dict):
        return set(left) == set(right) and all(_same_optimizer_setting(left[k], right[k]) for k in right)
    if isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(_same_optimizer_setting(a, b) for a, b in zip(left, right))
    return left == right


def validate_adam_recovery(state, initial, settings, total_steps):
    """Validate the actual optimizer state before load_state_dict can replace settings.

    Both v12f vector adapters contain only Linear parameters, in state_dict order.
    A fresh Adam built from those tensors supplies this Torch version's full default
    group configuration as well as the exact parameter-index order.
    """
    parameters = [torch.nn.Parameter(value.detach().clone()) for value in initial.values()]
    reference = torch.optim.Adam(parameters, lr=settings['learning_rate'],
        eps=settings['adam_eps'], weight_decay=settings['weight_decay']).state_dict()
    if (not isinstance(state, dict) or set(state) != set(reference)
            or not _same_optimizer_setting(state['param_groups'], reference['param_groups'])):
        raise ValueError('Recovery Adam parameter groups/settings changed')
    parameter_ids = reference['param_groups'][0]['params']
    states = state['state']
    expected_ids = set(parameter_ids) if total_steps else set()
    if (not isinstance(states, dict) or any(type(key) is not int for key in states)
            or set(states) != expected_ids):
        raise ValueError('Recovery Adam parameter-state keys disagree with completed updates')
    if total_steps == 0:
        return
    for key, parameter in zip(parameter_ids, parameters):
        entry = states[key]
        if not isinstance(entry, dict) or set(entry) != {'step', 'exp_avg', 'exp_avg_sq'}:
            raise ValueError('Recovery Adam moment fields changed')
        step = entry['step']
        if (not isinstance(step, torch.Tensor) or step.ndim != 0 or not step.is_floating_point()
                or not torch.isfinite(step) or float(step) != total_steps):
            raise ValueError('Recovery Adam step disagrees with completed updates')
        for name in ('exp_avg', 'exp_avg_sq'):
            value = entry[name]
            if (not isinstance(value, torch.Tensor) or value.shape != parameter.shape
                    or value.dtype != parameter.dtype or not torch.isfinite(value).all()):
                raise ValueError('Recovery Adam moment shape/dtype/finite check failed')
            if name == 'exp_avg_sq' and (value < 0).any():
                raise ValueError('Recovery Adam second moment cannot be negative')


def restore_fit(path, expected, initial, baseline, protocol):
    if path.is_symlink():
        raise ValueError('Recovery checkpoint cannot be a symlink')
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if saved.get('format_version') != RECOVERY_FORMAT or saved.get('identity') != expected:
        raise ValueError('Recovery checkpoint identity mismatch')
    epoch, history = saved['epoch'], saved['history']
    if type(epoch) is not int or not 0 <= epoch <= protocol['training']['epochs'] or len(history) != epoch:
        raise ValueError('Recovery epoch/history mismatch')
    selected = scope.replay_selection(history, baseline, protocol)
    previous.validate_state(saved['adapter_state'], initial)
    state_hash = alignment.tensor_hash(saved['adapter_state'])
    if state_hash != (history[-1]['adapter_state_sha256'] if history else alignment.tensor_hash(initial)):
        raise ValueError('Recovered current state hash disagrees with history')
    if set(saved['best']) != set(scope.POLICIES):
        raise ValueError('Recovery requires both output-policy best states')
    for policy, choice in selected.items():
        best = saved['best'][policy]
        if any(best[k] != choice[k] for k in ('epoch', 'selection_metrics')):
            raise ValueError('Recovered best metadata disagrees with selection history')
        previous.validate_state(best['state'], initial)
        expected_hash = history[choice['epoch'] - 1]['adapter_state_sha256'] if choice['epoch'] else alignment.tensor_hash(initial)
        if alignment.tensor_hash(best['state']) != expected_hash:
            raise ValueError('Recovered best state hash disagrees with selected epoch')
    steps = (expected['fit_samples'] + protocol['training']['batch_size'] - 1) // protocol['training']['batch_size']
    for row in history:
        if (type(row['training']['optimizer_steps']) is not int or row['training']['optimizer_steps'] != steps
                or row['training']['loss_region'] != 'candidate_early'):
            raise ValueError('Recovery optimizer step budget/loss changed')
    validate_adam_recovery(saved['optimizer_state'], initial, protocol['training'], epoch * steps)
    return saved


def fit(model, arm, seed, datasets, plan, baseline, protocol, device, directory, progress,
        run_identity_sha256, resume_from=None):
    if arm not in scope.ARMS:
        raise ValueError('Unknown v12m architecture')
    base.set_seed(seed)
    native_hash = base.backbone_hash(model)
    with torch.no_grad():
        raw = next(iter(base.loader(datasets['incident_full'], plan['fit']['indices']['incident_full'], 2)))
        batch = base.device_batch(raw, device)
        original = model(batch['x'], incident_data=batch['incident'])
        adapter = vector.attach_adapter(model, arm, protocol['training']['node_hidden_width'])
        initial_prediction = model(batch['x'], incident_data=batch['incident'])
        if not torch.equal(original, initial_prediction) or not torch.equal(original, native_prediction(model, batch)):
            raise ValueError('Initial vector/native bypass does not exactly reproduce A')
    model.icsf_module.clear_observations()
    initial = base.cpu_tree(adapter.state_dict())
    identity = {'arm': arm, 'loss': 'candidate_early', 'seed': seed, 'protocol_sha256': PROTOCOL_SHA256,
        'run_identity_sha256': run_identity_sha256, 'backbone_state_sha256': native_hash,
        'initial_adapter_sha256': alignment.tensor_hash(initial), 'output_policies': list(scope.POLICIES),
        'fit_samples': len(plan['fit']['indices']['incident_full']), 'training': protocol['training']}
    settings = protocol['training']
    optimizer = torch.optim.Adam(adapter.parameters(), lr=settings['learning_rate'],
        eps=settings['adam_eps'], weight_decay=settings['weight_decay'])
    best = {policy: {'epoch': 0, 'selection_metrics': copy.deepcopy(baseline), 'state': copy.deepcopy(initial)}
            for policy in scope.POLICIES}
    history, start = [], 1
    recovery = {'method': 'fresh_fit'}
    if resume_from is not None:
        source = Path(resume_from) / 'last_adapter.pt'
        if source.is_file() or source.is_symlink():
            saved = restore_fit(source, identity, initial, baseline, protocol)
            adapter.load_state_dict(saved['adapter_state'], strict=True)
            optimizer.load_state_dict(saved['optimizer_state'])
            best, history, start = saved['best'], saved['history'], saved['epoch'] + 1
            torch.set_rng_state(saved['rng_cpu'])
            if device.type == 'cuda':
                torch.cuda.set_rng_state(saved['rng_cuda'], device)
            recovery = {'method': 'epoch_boundary', 'epoch': saved['epoch'], 'source_sha256': base.sha256(source)}
            progress('fit_restored', epoch=saved['epoch'], selected_epochs={k: v['epoch'] for k, v in best.items()})
        else:
            recovery = {'method': 'fresh_fit_no_committed_epoch'}
    base.assert_backbone(model, native_hash)

    def publish(epoch):
        base.save_checkpoint(directory / 'last_adapter.pt', base.cpu_tree({
            'format_version': RECOVERY_FORMAT, 'identity': identity, 'epoch': epoch,
            'adapter_state': adapter.state_dict(), 'optimizer_state': optimizer.state_dict(),
            'best': best, 'history': history, 'rng_cpu': torch.get_rng_state(),
            'rng_cuda': torch.cuda.get_rng_state(device) if device.type == 'cuda' else None}))
        base.write_json(directory / 'history.json', history)

    publish(start - 1)
    evaluation_runtime = {}
    for epoch in range(start, settings['epochs'] + 1):
        training = alignment.early_train_epoch(model, adapter, optimizer, datasets['incident_full'],
            plan['fit']['indices']['incident_full'], settings, device, seed, epoch, progress)
        training['loss_region'] = 'candidate_early'
        current = {policy: {} for policy in scope.POLICIES}
        for cohort in base.COHORTS:
            records = evaluate_policies(model, datasets[cohort], plan['selection']['indices'][cohort],
                settings['evaluation_batch_size'], device,
                lambda stage, **fields: progress(stage, phase='selection', epoch=epoch, cohort=cohort, **fields),
                telemetry=evaluation_runtime)
            for policy in scope.POLICIES:
                current[policy][cohort] = metrics(records[policy])
        decisions = {}
        for policy in scope.POLICIES:
            decision = scope.selection_decision(current[policy], baseline, best[policy]['selection_metrics'], protocol)
            if decision['replace_best']:
                best[policy] = {'epoch': epoch, 'selection_metrics': copy.deepcopy(current[policy]),
                                'state': base.cpu_tree(adapter.state_dict())}
            decisions[policy] = {**decision, 'best_epoch': best[policy]['epoch']}
        base.assert_backbone(model, native_hash)
        history.append({'epoch': epoch, 'training': training, 'selection': current, 'decisions': decisions,
                        'adapter_state_sha256': alignment.tensor_hash(adapter.state_dict())})
        publish(epoch)
        progress('epoch_complete', epoch=epoch, selected_epochs={k: v['epoch'] for k, v in best.items()},
                 selection_mae={p: current[p]['incident_full']['mae'] for p in scope.POLICIES})
    for policy, choice in best.items():
        base.save_checkpoint(directory / f'selected_{policy}.pt', {
            'identity': identity, 'policy': policy, 'epoch': choice['epoch'], 'adapter_state': choice['state'],
            'adapter_state_sha256': alignment.tensor_hash(choice['state']),
            'selection_metrics': choice['selection_metrics']})
    detail = {'arm': arm, 'loss': 'candidate_early', 'seed': seed, 'epochs': len(history),
        'optimizer_steps': sum(row['training']['optimizer_steps'] for row in history),
        'trainable_parameters': sum(p.numel() for p in adapter.parameters()),
        'initial_adapter_sha256': alignment.tensor_hash(initial), 'backbone_state_sha256': native_hash,
        'initial_prediction_exactly_A': True, 'backbone_state_unchanged': True, 'recovery': recovery,
        'evaluation_runtime_this_invocation': evaluation_runtime,
        'selection_accounting': summarize_selection_history(history, protocol,
            {policy: choice['epoch'] for policy, choice in best.items()}),
        'policies': {p: {'selected_epoch': c['epoch'], 'selection_metrics': c['selection_metrics'],
                        'adapter_state_sha256': alignment.tensor_hash(c['state']),
                        'checkpoint_sha256': base.sha256(directory / f'selected_{p}.pt')}
                     for p, c in best.items()}}
    base.write_json(directory / 'fit_summary.json', detail)
    return detail


def fit_name(arm, seed):
    return f'{arm}_s{seed}'


def summarize_selection_history(history, protocol, selected_epochs):
    """Count each policy once per shared epoch, including recovered history."""
    checks = [f'{cohort}/{region}' for cohort in protocol['selection']['protected_cohorts']
              for region in protocol['selection']['protected_regions']]
    if set(selected_epochs) != set(scope.POLICIES):
        raise ValueError('Selection accounting requires both output policies')
    result = {}
    for policy in scope.POLICIES:
        selected = selected_epochs[policy]
        if type(selected) is not int or not 0 <= selected <= len(history):
            raise ValueError('Invalid selected epoch for selection accounting')
        result[policy] = {'trajectory_epochs': len(history), 'protection_rejected_epochs': 0,
            'eligible_epochs': 0, 'best_updates': 0, 'fallback_to_A': selected == 0,
            'protection_failure_counts': dict.fromkeys(checks, 0)}
    for epoch, row in enumerate(history, 1):
        if type(row['epoch']) is not int or row['epoch'] != epoch or set(row['decisions']) != set(scope.POLICIES):
            raise ValueError('Selection accounting requires complete contiguous shared history')
        for policy in scope.POLICIES:
            decision = row['decisions'][policy]
            guards = decision['protection_checks']
            if set(guards) != set(checks) or any(type(value) is not bool for value in guards.values()):
                raise ValueError('Selection accounting requires every Boolean protection check')
            item = result[policy]
            item['protection_rejected_epochs'] += int(not all(guards.values()))
            item['eligible_epochs'] += int(decision['eligible'])
            item['best_updates'] += int(decision['replace_best'])
            for name in checks:
                item['protection_failure_counts'][name] += int(not guards[name])
    return result


def aggregate_selection_accounting(runs, protocol):
    """Endpoint denominators exclude the derived P(U) outputs and duplicate policies."""
    checks = [f'{cohort}/{region}' for cohort in protocol['selection']['protected_cohorts']
              for region in protocol['selection']['protected_regions']]
    details = [detail for fits in runs.values() for detail in fits.values()]

    def group(items, policy):
        result = {'total_endpoints': len(items), 'fallback_endpoints': 0, 'fallback_rate': None,
                  'trajectory_epochs': 0, 'protection_rejected_epochs': 0, 'eligible_epochs': 0,
                  'best_updates': 0, 'protection_failure_counts': dict.fromkeys(checks, 0)}
        for detail in items:
            counts = detail['selection_accounting'][policy]
            result['fallback_endpoints'] += int(counts['fallback_to_A'])
            for key in ('trajectory_epochs', 'protection_rejected_epochs', 'eligible_epochs', 'best_updates'):
                result[key] += counts[key]
            for name in checks:
                result['protection_failure_counts'][name] += counts['protection_failure_counts'][name]
        if items:
            result['fallback_rate'] = result['fallback_endpoints'] / result['total_endpoints']
        return result

    return {'by_policy': {policy: group(details, policy) for policy in scope.POLICIES},
            'by_arm': {arm: {policy: group([d for d in details if d['arm'] == arm], policy)
                             for policy in scope.POLICIES} for arm in scope.ARMS},
            'counting_note': 'Each policy counts each shared trajectory epoch once; guard failures overlap. '
                             'Fallback denominators are selected endpoints per policy, excluding derived P(U) outputs. '
                             'Policies and epochs are not independent replicates.'}


def code_paths():
    return list(dict.fromkeys([Path(__file__), PROTOCOL, Path(scope.__file__),
                              PROTOCOL.with_name('run_vector_output_scope.sh'), *previous.code_paths()]))


def run(data_dir, primary_dir, secondary_dir, checkpoint, output, device='cuda:0', check=False, resume_from=None):
    frozen, inherited = load_protocol(), base.load_protocol()
    effective = copy.deepcopy(inherited)
    effective['training']['objective'] = 'v12m shared candidate_early trajectory; unchanged vector energy penalty'
    if check:
        effective['seeds'] = inherited['check']['seeds']
        effective['training']['epochs'] = inherited['check']['epochs']
    output = Path(os.path.abspath(output))
    partial = output.with_name(output.name + '.partial')
    for target in (output, partial):
        if target.exists() or target.is_symlink():
            raise FileExistsError('Preserve output/partial and choose a new run name')
    if resume_from is not None:
        resume_from = Path(os.path.abspath(resume_from))
        if (check or not resume_from.is_dir() or output.resolve().is_relative_to(resume_from.resolve()) or
                resume_from.is_symlink() or any(p.is_symlink() for p in resume_from.rglob('*'))):
            raise ValueError('Recovery requires a separate, nonsymlink full-run source')
    partial.mkdir(parents=True)
    started = time.monotonic()

    def progress(stage, **fields):
        event = {'stage': stage, 'elapsed_seconds': time.monotonic() - started,
                 'memory': base.memory_snapshot(device), **fields}
        base.write_json(partial / 'progress.json', event)
        print(json.dumps(event), flush=True)

    try:
        device = torch.device(device)
        torch.set_num_threads(3)
        base.configure_determinism(device)
        architecture_protocol = base.mechanisms.load_protocol(base.mechanisms.PROTOCOL)
        if base.sha256(base.mechanisms.PROTOCOL) != inherited['v12a_protocol_sha256']:
            raise ValueError('Inherited backbone protocol changed')
        progress('verifying_inputs')
        baseline, hashes = base.mechanisms.verify_inputs(data_dir, primary_dir, secondary_dir, checkpoint, architecture_protocol)
        positive = base.read_csv(Path(data_dir) / 'train_manifest.csv')
        plan = base.make_plan(positive, base.read_csv(Path(primary_dir) / 'train_control_manifest.csv'),
            base.read_csv(Path(secondary_dir) / 'train_second_control_manifest.csv'), inherited)
        phase_samples = {p: {c: len(v) for c, v in x['indices'].items()} for p, x in plan.items()}
        if phase_samples != frozen['expected_phase_samples']:
            raise ValueError('Frozen phase sample budget changed')
        base.write_json(partial / 'eligibility.json', plan)
        if check:
            plan = copy.deepcopy(plan)
            for period in plan.values():
                period['indices'] = {c: v[:inherited['check']['samples_per_cohort_period']] for c, v in period['indices'].items()}
        code_hashes = {str(p.relative_to(REPO)): base.sha256(p) for p in code_paths()}
        identity = {'protocol_sha256': PROTOCOL_SHA256, 'engineering_check': check,
            'inputs': hashes, 'code_sha256': code_hashes, 'checkpoint_sha256': base.sha256(checkpoint),
            'indices': {p: x['indices'] for p, x in plan.items()}, 'effective_training': effective['training'],
            'seeds': effective['seeds'], 'output_policies': list(scope.POLICIES),
            'output_paths': list(scope.OUTPUT_PATHS), 'device': str(device),
            'torch': str(torch.__version__), 'numpy': np.__version__}
        if resume_from and json.loads((resume_from / 'run_identity.json').read_text()) != identity:
            raise ValueError('Recovery run input/code/protocol/sample identity changed')
        base.write_json(partial / 'run_identity.json', identity)
        identity_hash = base.sha256(partial / 'run_identity.json')
        datasets = base.make_datasets(data_dir, primary_dir, secondary_dir, baseline)
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        baseline_runtime, endpoint_runtime = {}, {}

        def native():
            model = base.make_model(Path(data_dir), len(datasets['incident_full'].station_ids), device, 'fixed')
            model.load_state_dict(state, strict=True)
            if sum(p.numel() for p in model.parameters()) != baseline['checkpoint']['parameters']:
                raise ValueError('Backbone size changed')
            return model.eval().requires_grad_(False)

        def evaluate_reference(model, phase, destination):
            if phase == 'audit' and not all((partial / name).is_file() for name in
                    ('selected_endpoints_frozen.json', 'evaluation_paths_frozen.json')):
                raise ValueError('All endpoint and output identities must freeze before audit access')
            records = {}
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            evaluation_started = time.monotonic()
            batch_size = effective['training']['evaluation_batch_size']
            for cohort, indices in plan[phase]['indices'].items():
                record = evaluate(model, datasets[cohort], indices, batch_size, device,
                    lambda stage, **fields: progress(stage, phase=phase, cohort=cohort, endpoint='A', **fields))
                base.save_arrays(destination / f'{phase}_{cohort}.npz', record)
                records[cohort] = record
                baseline_runtime['native_forward_batches'] = baseline_runtime.get('native_forward_batches', 0) + (len(indices) + batch_size - 1) // batch_size
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            baseline_runtime['elapsed_seconds'] = baseline_runtime.get('elapsed_seconds', 0.) + time.monotonic() - evaluation_started
            return records

        def release(model):
            if hasattr(model.icsf_module, 'clear_observations'):
                model.icsf_module.clear_observations()
            model.cpu()
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        reference_dir = partial / 'A'
        reference_dir.mkdir()
        model = native()
        backbone_hash = base.backbone_hash(model)
        reference = {'selection': evaluate_reference(model, 'selection', reference_dir)}
        selection = {c: metrics(r) for c, r in reference['selection'].items()}
        base.assert_backbone(model, backbone_hash)
        release(model)
        del model
        runs, frozen_endpoints, frozen_paths = {}, {}, {}
        for seed in effective['seeds']:
            runs[str(seed)] = {}
            initial_hashes = set()
            for arm in scope.ARMS:
                name = fit_name(arm, seed)
                directory = partial / name
                directory.mkdir()
                model = native()
                progress('fit_started', seed=seed, arm=arm, loss='candidate_early')
                detail = fit(model, arm, seed, datasets, plan, selection, effective, device, directory,
                    lambda stage, **fields: progress(stage, seed=seed, arm=arm, **fields), identity_hash,
                    resume_from / name if resume_from else None)
                initial_hashes.add(detail['initial_adapter_sha256'])
                if len(initial_hashes) != 1:
                    raise ValueError('Paired vector initializations differ')
                runs[str(seed)][name] = detail
                for policy in scope.POLICIES:
                    key = f'{name}/selected_{policy}.pt'
                    digest = base.sha256(partial / key)
                    if digest != detail['policies'][policy]['checkpoint_sha256']:
                        raise ValueError('Selected checkpoint changed before freezing')
                    frozen_endpoints[key] = digest
                for path, selected_policy, output_policy in (
                        (scope.OUTPUT_PATHS[0], scope.POLICIES[0], scope.POLICIES[0]),
                        (scope.OUTPUT_PATHS[1], scope.POLICIES[0], scope.POLICIES[1]),
                        (scope.OUTPUT_PATHS[2], scope.POLICIES[1], scope.POLICIES[1])):
                    chosen = detail['policies'][selected_policy]
                    key = f'{name}/selected_{selected_policy}.pt'
                    frozen_paths[f'{name}/{path}'] = {
                        'source_checkpoint': key, 'source_checkpoint_sha256': frozen_endpoints[key],
                        'selected_policy': selected_policy, 'output_policy': output_policy,
                        'epoch': chosen['selected_epoch'], 'adapter_state_sha256': chosen['adapter_state_sha256']}
                release(model)
                del model
        expected_fits = len(effective['seeds']) * len(scope.ARMS)
        budget = {'fits': sum(len(values) for values in runs.values()),
                  'trajectory_epochs': sum(detail['epochs'] for values in runs.values() for detail in values.values()),
                  'selected_endpoints': len(frozen_endpoints), 'output_paths': len(frozen_paths)}
        expected_budget = {'fits': expected_fits, 'trajectory_epochs': expected_fits * effective['training']['epochs'],
                           'selected_endpoints': expected_fits * len(scope.POLICIES),
                           'output_paths': expected_fits * len(scope.OUTPUT_PATHS)}
        if budget != expected_budget:
            raise ValueError('Incomplete shared-trajectory/endpoint/output budget')
        base.write_json(partial / 'selected_endpoints_frozen.json', frozen_endpoints)
        base.write_json(partial / 'evaluation_paths_frozen.json', frozen_paths)
        manifest_hashes = {name: base.sha256(partial / name) for name in
                           ('selected_endpoints_frozen.json', 'evaluation_paths_frozen.json')}
        progress('all_selectors_frozen', **budget)

        model = native()
        for phase in ('fit', 'audit'):
            reference[phase] = evaluate_reference(model, phase, reference_dir)
        base.assert_backbone(model, backbone_hash)
        release(model)
        del model
        comparisons, csv_rows = {}, []
        times = {int(row['sample_index']): row['t0'] for row in positive}
        for seed in effective['seeds']:
            learned = {phase: {} for phase in ('fit', 'selection', 'audit')}
            for arm in scope.ARMS:
                name = fit_name(arm, seed)
                detail = runs[str(seed)][name]
                detail['output_paths'] = {}
                model = native()
                adapter = vector.attach_adapter(model, arm, effective['training']['node_hidden_width'])
                for selected_policy in scope.POLICIES:
                    key = f'{name}/selected_{selected_policy}.pt'
                    if base.sha256(partial / key) != frozen_endpoints[key]:
                        raise ValueError('Frozen selected endpoint changed')
                    saved = torch.load(partial / key, map_location='cpu', weights_only=True)
                    chosen = detail['policies'][selected_policy]
                    if (saved['policy'] != selected_policy or saved['epoch'] != chosen['selected_epoch'] or
                            saved['selection_metrics'] != chosen['selection_metrics']):
                        raise ValueError('Selected endpoint metadata disagrees with frozen policy')
                    adapter.load_state_dict(saved['adapter_state'], strict=True)
                    expected_hash = chosen['adapter_state_sha256']
                    if alignment.tensor_hash(adapter.state_dict()) != expected_hash or saved['adapter_state_sha256'] != expected_hash:
                        raise ValueError('Selected adapter state mismatch')
                    requested = scope.POLICIES if selected_policy == scope.POLICIES[0] else (scope.POLICIES[1],)
                    paths = {scope.POLICIES[0]: scope.OUTPUT_PATHS[0], scope.POLICIES[1]: scope.OUTPUT_PATHS[1]} if selected_policy == scope.POLICIES[0] else {scope.POLICIES[1]: scope.OUTPUT_PATHS[2]}
                    for policy in requested:
                        path = paths[policy]
                        frozen_path = frozen_paths[f'{name}/{path}']
                        if (frozen_path['source_checkpoint'] != key or frozen_path['output_policy'] != policy or
                                frozen_path['epoch'] != saved['epoch'] or frozen_path['adapter_state_sha256'] != expected_hash):
                            raise ValueError('Frozen evaluation path disagrees with loaded endpoint')
                        (partial / name / path).mkdir()
                        detail['output_paths'][path] = {**frozen_path, 'phase_metrics': {}}
                    for phase in ('fit', 'selection', 'audit'):
                        records = {policy: {} for policy in requested}
                        for cohort, indices in plan[phase]['indices'].items():
                            batch_records = evaluate_policies(model, datasets[cohort], indices,
                                effective['training']['evaluation_batch_size'], device,
                                lambda stage, **fields: progress(stage, phase=phase, cohort=cohort,
                                    seed=seed, arm=arm, selected_policy=selected_policy, **fields),
                                policies=requested, telemetry=endpoint_runtime)
                            if set(batch_records) != set(requested):
                                raise ValueError('Evaluator returned an unrequested output policy')
                            for policy, record in batch_records.items():
                                anchor = reference[phase][cohort]
                                for field in ('ids', 'source_indices', 'candidate_mask', 'counts', 'prediction_counts', 'regions'):
                                    if not np.array_equal(record[field], anchor[field]):
                                        raise ValueError('Evaluation sample/support changed')
                                if saved['epoch'] == 0 and not np.array_equal(record['errors'], anchor['errors']):
                                    raise ValueError('Epoch-0 endpoint must be exactly A')
                                if policy == scope.POLICIES[1]:
                                    columns = [list(record['regions']).index(region) for region in ('candidate_h7_h12', 'noncandidate_all')]
                                    if not np.array_equal(record['errors'][:, columns], anchor['errors'][:, columns]):
                                        raise ValueError('Protected late/noncandidate errors differ from A')
                                base.save_arrays(partial / name / paths[policy] / f'{phase}_{cohort}.npz', record)
                                records[policy][cohort] = record
                            if len(requested) == 2:
                                column = list(batch_records[scope.POLICIES[0]]['regions']).index('candidate_h1_h6')
                                if not np.array_equal(batch_records[scope.POLICIES[0]]['errors'][:, column],
                                                      batch_records[scope.POLICIES[1]]['errors'][:, column]):
                                    raise ValueError('Same-weight output scope changed candidate early errors')
                        for policy in requested:
                            path = paths[policy]
                            learned[phase][scope.endpoint(arm, path)] = records[policy]
                            detail['output_paths'][path]['phase_metrics'][phase] = {c: metrics(r) for c, r in records[policy].items()}
                    primary_path = scope.OUTPUT_PATHS[0] if selected_policy == scope.POLICIES[0] else scope.OUTPUT_PATHS[2]
                    endpoint_metrics = detail['output_paths'][primary_path]['phase_metrics']
                    if endpoint_metrics['selection'] != chosen['selection_metrics']:
                        raise ValueError('Selected selection metrics failed exact same-batch replay')
                    if alignment.tensor_hash(adapter.state_dict()) != expected_hash:
                        raise ValueError('Evaluation changed adapter state')
                    base.assert_backbone(model, backbone_hash)
                    chosen['phase_metrics'] = endpoint_metrics
                release(model)
                del model
            comparisons[str(seed)] = {}
            if not check:
                for phase in ('fit', 'selection', 'audit'):
                    progress('phase_statistics_started', seed=seed, phase=phase)
                    analysis, rows, weekly = scope.compare_phase(reference[phase], learned[phase], times, inherited)
                    comparisons[str(seed)][phase] = analysis
                    csv_rows.extend({'seed': seed, 'phase': phase, **row} for row in rows)
                    base.save_arrays(partial / f'{phase}_weekly_s{seed}.npz', weekly)
            progress('seed_evaluation_complete', seed=seed)
        if csv_rows:
            with (partial / 'comparisons.csv').open('w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
                writer.writeheader()
                writer.writerows(csv_rows)
        if code_hashes != {str(p.relative_to(REPO)): base.sha256(p) for p in code_paths()}:
            raise ValueError('Source code changed during v12m run')
        if manifest_hashes != {name: base.sha256(partial / name) for name in manifest_hashes}:
            raise ValueError('Frozen endpoint/output path manifests changed during evaluation')
        if any(base.sha256(partial / name) != digest for name, digest in frozen_endpoints.items()):
            raise ValueError('Frozen selected checkpoint changed during evaluation')
        fit_runtime = {name: detail['evaluation_runtime_this_invocation']
                       for values in runs.values() for name, detail in values.items()}
        policy_runtime = {key: endpoint_runtime.get(key, 0) + sum(value.get(key, 0) for value in fit_runtime.values())
                          for key in ('native_forward_batches', 'modified_forward_batches', 'checked_prediction_cells', 'elapsed_seconds')}
        summary = {'status': 'ENGINEERING_CHECK_PASS' if check else 'VECTOR_OUTPUT_SCOPE_COMPARISON_COMPLETE',
            'protocol_id': frozen['protocol_id'], 'protocol_sha256': PROTOCOL_SHA256,
            'engineering_check': check, 'frozen_protocol': frozen, 'inherited_protocol': inherited,
            'effective_training': effective['training'], **frozen['information_boundary'],
            'model_training_performed': True, 'training_scope': 'icsf_vector_adapter_shared_candidate_early_only',
            'recommendation': 'ENGINEERING_ONLY' if check else frozen['decision'],
            'main_training_ready': False, 'all_selectors_frozen_before_audit_evaluation': True,
            'all_output_paths_frozen_before_audit_evaluation': True, 'paired_initialization_exact': True,
            'unrestricted_at_protected_evaluated': False, 'runs': runs, 'phase_comparisons': comparisons,
            'budget': budget, 'selected_endpoints_frozen': frozen_endpoints, 'evaluation_paths_frozen': frozen_paths,
            'selection_accounting': aggregate_selection_accounting(runs, effective),
            'baseline': {p: {c: metrics(r) for c, r in values.items()} for p, values in reference.items()},
            'phase_samples': {p: {c: len(v) for c, v in x['indices'].items()} for p, x in plan.items()},
            'evaluation_runtime_this_invocation': {'baseline_reference': baseline_runtime,
                'fit_selection_by_trajectory': fit_runtime, 'selected_outputs': endpoint_runtime,
                'policy_evaluation_total': policy_runtime, 'memory_at_completion': base.memory_snapshot(device)},
            'inputs': hashes, 'code_sha256': code_hashes, 'recovery_source': str(resume_from) if resume_from else None,
            'environment': {'host': socket.gethostname(), 'python': sys.version, 'device': str(device),
                'torch': str(torch.__version__), 'numpy': np.__version__, 'threads': torch.get_num_threads(),
                'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
                'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
                'tf32': torch.backends.cuda.matmul.allow_tf32,
                'git_head': base.subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()},
            'elapsed_seconds': time.monotonic() - started,
            'outputs': {str(p.relative_to(partial)): base.sha256(p) for p in partial.rglob('*')
                        if p.is_file() and p.name not in ('progress.json', 'failure.json')}}
        base.write_json(partial / 'summary.json', summary)
        if output.exists() or output.is_symlink():
            raise FileExistsError('Final output appeared; preserve partial')
        partial.rename(output)
    except BaseException as error:
        base.write_json(partial / 'failure.json', {'error': str(error), 'traceback': traceback.format_exc()})
        raise
    report(summary)
    print('Saved v12m output-scope comparison:', output / 'summary.json', flush=True)
    return summary


def report(summary):
    print('status:', summary['status'])
    print('protocol_sha256:', summary['protocol_sha256'])
    print('recommendation:', summary['recommendation'])
    print('budget:', json.dumps(summary['budget']))
    print('All selected endpoints frozen before audit:', summary['all_selectors_frozen_before_audit_evaluation'])
    print('All output paths frozen before audit:', summary['all_output_paths_frozen_before_audit_evaluation'])
    print('GAIN = reference MAE - compared MAE; raw units. Shared early-loss trajectories, two output policies.')
    print('Late/noncandidate preservation is structural; early differences reflect endpoint selection, not new early learning.')
    print('Fit/selection are conditional; audit is repeatedly reused development, not independent confirmation.')
    accounting = summary['selection_accounting']
    print('Selection accounting:', accounting['counting_note'])
    for policy, counts in accounting['by_policy'].items():
        print('all_architectures', policy, json.dumps(counts))
    for arm, policies in accounting['by_arm'].items():
        for policy, counts in policies.items():
            print('architecture', arm, policy, json.dumps(counts))
    for seed, fits in summary['runs'].items():
        for name, detail in fits.items():
            for policy, selected in detail['policies'].items():
                print(f'\n[{name} policy={policy}] epoch={selected["selected_epoch"]}')
                print('selection_counts:', json.dumps(detail['selection_accounting'][policy]))
                for region in scope.REGIONS:
                    gains = {}
                    for phase in ('fit', 'selection', 'audit'):
                        anchor = summary['baseline'][phase]['incident_full']['mae'][region]
                        value = selected['phase_metrics'][phase]['incident_full']['mae'][region]
                        gains[phase] = anchor - value if anchor is not None and value is not None else None
                    print(region, 'full_positive_gain=', gains)
                if not summary['engineering_check']:
                    path = scope.OUTPUT_PATHS[0] if policy == scope.POLICIES[0] else scope.OUTPUT_PATHS[2]
                    comparison = scope.endpoint(detail['arm'], path) + '_vs_A'
                    audit = summary['phase_comparisons'][seed]['audit']['results']
                    for cohort in base.COHORTS:
                        effect = audit[cohort]['regions']['candidate_h1_h6']['comparisons'][comparison]
                        ci = effect['intervals']['four_week_block']['pooled']
                        print(cohort, 'early_gain=', effect['gain_raw_mae'],
                              'equal_window_gain=', effect['equal_forecast_window_gain_raw_mae'],
                              'early_block_CI=', [ci['ci_low'], ci['ci_high']])
            if not summary['engineering_check']:
                audit = summary['phase_comparisons'][seed]['audit']['results']['incident_full']['regions']
                for region in scope.REGIONS:
                    for effect_name in ('output_scope_effect', 'selection_effect', 'total_policy_effect'):
                        comparison = f'{detail["arm"]}__{effect_name}'
                        effect = audit[region]['comparisons'][comparison]
                        ci = effect['intervals']['four_week_block']['pooled']
                        print('decomposition', region, comparison, 'gain=', effect['gain_raw_mae'],
                              'block_CI=', [ci['ci_low'], ci['ci_high']])
        if not summary['engineering_check']:
            effect = summary['phase_comparisons'][seed]['audit']['results']['incident_full']['regions']['candidate_h1_h6']['comparisons']['protected_at_protected__context_effect']
            ci = effect['intervals']['four_week_block']['pooled']
            print('protected_context_contrast', seed, 'early_gain=', effect['gain_raw_mae'],
                  'block_CI=', [ci['ci_low'], ci['ci_high']])
    print('Evaluation runtime this invocation:', json.dumps(summary['evaluation_runtime_this_invocation']))
    print('No unrestricted(P) errors were evaluated. Total-effect intervals are computed directly, not by adding interval bounds.')


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
        if not args.summary.is_file():
            parser.exit(1, 'INCOMPLETE: use status to inspect .job/run.log and preserve .partial.\n')
        report(json.loads(args.summary.read_text()))
    else:
        run(args.data_dir, args.primary_control_dir, args.secondary_control_dir, args.checkpoint,
            args.output, args.device, args.check, args.resume_from)


if __name__ == '__main__':
    main()
