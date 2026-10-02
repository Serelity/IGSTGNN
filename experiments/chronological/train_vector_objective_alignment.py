"""v12k: paired vector losses and two independent selectors on each fit trajectory."""

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
from experiments.chronological import vector_objective_alignment as alignment

vector, base = alignment.vector, alignment.base
PROTOCOL = Path(__file__).with_name('vector_objective_alignment_v12k.json')
PROTOCOL_SHA256 = '896a2969389c262aaf7b8dea78f969ab1d10337946bc4c5d6a5ba4a15e9620be'


def load_protocol():
    if base.sha256(PROTOCOL) != PROTOCOL_SHA256:
        raise ValueError('Frozen v12k protocol changed')
    frozen = json.loads(PROTOCOL.read_text())
    if (base.sha256(base.PROTOCOL) != frozen['v12c_protocol_sha256'] or
            base.sha256(vector.PROTOCOL) != frozen['v12f_protocol_sha256']):
        raise ValueError('Inherited vector/temporal protocol changed')
    vector.load_protocol()
    return frozen


def evaluate(model, dataset, indices, batch_size, device, progress=lambda *a, **k: None):
    # Keep per-window errors/support, not large per-node representation diagnostics.
    model.eval()
    pieces = {}
    with torch.no_grad():
        for step, raw in enumerate(base.loader(dataset, indices, batch_size)):
            batch = base.device_batch(raw, device)
            prediction = model(batch['x'], incident_data=batch['incident'])
            prediction = prediction * dataset.scaler['std'] + dataset.scaler['mean']
            values = base.statistics(prediction, batch)
            values.update(ids=raw['positive_sample_index'].numpy(), source_indices=raw['source_index'].numpy(),
                          candidate_mask=raw['candidate_mask'].numpy())
            for key, value in values.items():
                pieces.setdefault(key, []).append(value)
            if hasattr(model.icsf_module, 'clear_observations'):
                model.icsf_module.clear_observations()
            if step % 10 == 0:
                progress('evaluation_progress', batches_completed=step + 1, samples_total=len(indices))
    return {**{k: np.concatenate(v) for k, v in pieces.items()}, 'regions': base.REGIONS}


def metrics(record):
    counts, errors = record['counts'].sum(0), record['errors'].sum(0)
    return {'samples': len(record['ids']), 'mae': {
        r: float(errors[i] / counts[i]) if counts[i] else None for i, r in enumerate(record['regions'])}}


def validate_state(state, initial):
    if set(state) != set(initial):
        raise ValueError('Adapter state keys changed')
    for key, value in state.items():
        if (not isinstance(value, torch.Tensor) or value.shape != initial[key].shape or
                value.dtype != initial[key].dtype or not torch.isfinite(value).all()):
            raise ValueError('Invalid adapter tensor')


def validate_finite_tree(value):
    if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
        raise ValueError('Nonfinite optimizer state')
    if isinstance(value, dict):
        for item in value.values():
            validate_finite_tree(item)
    if isinstance(value, (tuple, list)):
        for item in value:
            validate_finite_tree(item)


def restore_fit(path, expected, initial, baseline, protocol):
    if path.is_symlink():
        raise ValueError('Recovery checkpoint cannot be a symlink')
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if saved.get('format_version') != 1 or saved.get('identity') != expected:
        raise ValueError('Recovery checkpoint identity mismatch')
    epoch, history = saved['epoch'], saved['history']
    if type(epoch) is not int or not 0 <= epoch <= protocol['training']['epochs'] or len(history) != epoch:
        raise ValueError('Recovery epoch/history mismatch')
    selected = alignment.replay_selection(history, baseline, protocol)
    validate_state(saved['adapter_state'], initial)
    state_hash = alignment.tensor_hash(saved['adapter_state'])
    if state_hash != (history[-1]['adapter_state_sha256'] if history else alignment.tensor_hash(initial)):
        raise ValueError('Recovered current state hash disagrees with history')
    if set(saved['best']) != set(alignment.SELECTORS):
        raise ValueError('Recovery requires both selected states')
    for selector, choice in selected.items():
        best = saved['best'][selector]
        if any(best[key] != choice[key] for key in ('epoch', 'selection_metrics')):
            raise ValueError('Recovered best metadata disagrees with selection history')
        validate_state(best['state'], initial)
        expected_hash = history[choice['epoch'] - 1]['adapter_state_sha256'] if choice['epoch'] else alignment.tensor_hash(initial)
        if alignment.tensor_hash(best['state']) != expected_hash:
            raise ValueError('Recovered best state hash disagrees with selected epoch')
    expected_steps = (expected['fit_samples'] + protocol['training']['batch_size'] - 1) // protocol['training']['batch_size']
    if any(row['training']['optimizer_steps'] != expected_steps for row in history):
        raise ValueError('Recovery optimizer step budget changed')
    validate_finite_tree(saved['optimizer_state'])
    return saved


def fit(model, arm, loss, seed, datasets, plan, baseline, protocol, device, directory, progress,
        run_identity_sha256, resume_from=None):
    if arm not in alignment.ARMS or loss not in alignment.LOSSES:
        raise ValueError('Unknown v12k architecture/loss')
    base.set_seed(seed)
    native_hash = base.backbone_hash(model)
    with torch.no_grad():
        raw = next(iter(base.loader(datasets['incident_full'], plan['fit']['indices']['incident_full'], 2)))
        batch = base.device_batch(raw, device)
        original = model(batch['x'], incident_data=batch['incident'])
        adapter = vector.attach_adapter(model, arm, protocol['training']['node_hidden_width'])
        initial_prediction = model(batch['x'], incident_data=batch['incident'])
        if not torch.equal(original, initial_prediction):
            raise ValueError('Initial vector does not exactly reproduce A')
    model.icsf_module.clear_observations()
    initial = base.cpu_tree(adapter.state_dict())
    identity = {'arm': arm, 'loss': loss, 'seed': seed, 'protocol_sha256': PROTOCOL_SHA256,
        'run_identity_sha256': run_identity_sha256, 'backbone_state_sha256': native_hash,
        'initial_adapter_sha256': alignment.tensor_hash(initial),
        'fit_samples': len(plan['fit']['indices']['incident_full']), 'training': protocol['training']}
    settings = protocol['training']
    optimizer = torch.optim.Adam(adapter.parameters(), lr=settings['learning_rate'],
        eps=settings['adam_eps'], weight_decay=settings['weight_decay'])
    best = {selector: {'epoch': 0, 'selection_metrics': copy.deepcopy(baseline), 'state': copy.deepcopy(initial)}
            for selector in alignment.SELECTORS}
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
            recovery = {'method': 'epoch_boundary', 'epoch': saved['epoch'],
                        'source_sha256': base.sha256(source)}
            progress('fit_restored', epoch=saved['epoch'], selected_epochs={k: v['epoch'] for k, v in best.items()})
        else:
            recovery = {'method': 'fresh_fit_no_committed_epoch'}
    base.assert_backbone(model, native_hash)

    def publish(epoch):
        base.save_checkpoint(directory / 'last_adapter.pt', base.cpu_tree({
            'format_version': 1, 'identity': identity, 'epoch': epoch,
            'adapter_state': adapter.state_dict(), 'optimizer_state': optimizer.state_dict(),
            'best': best, 'history': history, 'rng_cpu': torch.get_rng_state(),
            'rng_cuda': torch.cuda.get_rng_state(device) if device.type == 'cuda' else None}))
        base.write_json(directory / 'history.json', history)

    publish(start - 1)
    trainer = vector.vector_train_epoch if loss == 'global' else alignment.early_train_epoch
    for epoch in range(start, settings['epochs'] + 1):
        training = trainer(model, adapter, optimizer, datasets['incident_full'],
            plan['fit']['indices']['incident_full'], settings, device, seed, epoch, progress)
        training['loss_region'] = loss
        current = {}
        for cohort in base.COHORTS:
            current[cohort] = metrics(evaluate(model, datasets[cohort], plan['selection']['indices'][cohort],
                settings['evaluation_batch_size'], device,
                lambda stage, **fields: progress(stage, phase='selection', epoch=epoch, cohort=cohort, **fields)))
        decisions = {}
        for selector in alignment.SELECTORS:
            decision = alignment.selection_decision(current, baseline, best[selector]['selection_metrics'], selector, protocol)
            if decision['replace_best']:
                best[selector] = {'epoch': epoch, 'selection_metrics': copy.deepcopy(current),
                                  'state': base.cpu_tree(adapter.state_dict())}
            decisions[selector] = {**decision, 'best_epoch': best[selector]['epoch']}
        base.assert_backbone(model, native_hash)
        history.append({'epoch': epoch, 'training': training, 'selection': current, 'decisions': decisions,
                        'adapter_state_sha256': alignment.tensor_hash(adapter.state_dict())})
        publish(epoch)
        progress('epoch_complete', epoch=epoch, selected_epochs={k: v['epoch'] for k, v in best.items()},
                 selection_global_mae=current['incident_full']['mae']['all'],
                 selection_early_mae=current['incident_full']['mae']['candidate_h1_h6'])
    for selector, choice in best.items():
        selected = directory / f'selected_{selector}.pt'
        base.save_checkpoint(selected, {'identity': identity, 'selector': selector,
            'epoch': choice['epoch'], 'adapter_state': choice['state'],
            'adapter_state_sha256': alignment.tensor_hash(choice['state']),
            'selection_metrics': choice['selection_metrics']})
    detail = {'arm': arm, 'loss': loss, 'seed': seed, 'epochs': len(history),
        'optimizer_steps': sum(row['training']['optimizer_steps'] for row in history),
        'trainable_parameters': sum(p.numel() for p in adapter.parameters()),
        'initial_adapter_sha256': alignment.tensor_hash(initial), 'backbone_state_sha256': native_hash,
        'initial_prediction_exactly_A': True, 'backbone_state_unchanged': True, 'recovery': recovery,
        'selectors': {s: {'selected_epoch': c['epoch'], 'selection_metrics': c['selection_metrics'],
                         'adapter_state_sha256': alignment.tensor_hash(c['state']),
                         'checkpoint_sha256': base.sha256(directory / f'selected_{s}.pt')}
                      for s, c in best.items()}}
    base.write_json(directory / 'fit_summary.json', detail)
    return detail


def fit_name(arm, loss, seed):
    return f'{arm}__loss_{loss}_s{seed}'


def code_paths():
    paths = [Path(__file__), PROTOCOL, Path(alignment.__file__), Path(vector.__file__), vector.PROTOCOL,
        Path(base.__file__), base.PROTOCOL, Path(base.mechanisms.__file__), base.mechanisms.PROTOCOL,
        Path(alignment.regional.__file__), Path(__file__).with_name('run_vector_objective_alignment.sh')]
    paths += [REPO / 'experiments/chronological' / name for name in
        ('gate_recovery.py', 'audit_matched_controls.py', 'materialize_incident_branch.py', 'smoke.py', 'train.py')]
    paths += [REPO / 'src/utils/chronological.py', *sorted((REPO / 'src/models').rglob('*.py'))]
    return paths


def run(data_dir, primary_dir, secondary_dir, checkpoint, output, device='cuda:0', check=False, resume_from=None):
    frozen, inherited = load_protocol(), base.load_protocol()
    effective = copy.deepcopy(inherited)
    effective['training']['objective'] = 'v12k paired global/candidate_early objectives; unchanged vector energy penalty'
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
            'seeds': effective['seeds'], 'device': str(device), 'torch': str(torch.__version__), 'numpy': np.__version__}
        if resume_from and json.loads((resume_from / 'run_identity.json').read_text()) != identity:
            raise ValueError('Recovery run input/code/protocol/sample identity changed')
        base.write_json(partial / 'run_identity.json', identity)
        identity_hash = base.sha256(partial / 'run_identity.json')
        datasets = base.make_datasets(data_dir, primary_dir, secondary_dir, baseline)
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)

        def native():
            model = base.make_model(Path(data_dir), len(datasets['incident_full'].station_ids), device, 'fixed')
            model.load_state_dict(state, strict=True)
            if sum(p.numel() for p in model.parameters()) != baseline['checkpoint']['parameters']:
                raise ValueError('Backbone size changed')
            return model.eval().requires_grad_(False)

        def evaluate_period(model, phase, destination, label):
            records = {}
            for cohort, indices in plan[phase]['indices'].items():
                record = evaluate(model, datasets[cohort], indices, effective['training']['evaluation_batch_size'], device,
                    lambda stage, **fields: progress(stage, phase=phase, cohort=cohort, endpoint=label, **fields))
                base.save_arrays(destination / f'{phase}_{cohort}.npz', record)
                records[cohort] = record
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
        reference = {'selection': evaluate_period(model, 'selection', reference_dir, 'A')}
        selection = {c: metrics(r) for c, r in reference['selection'].items()}
        base.assert_backbone(model, backbone_hash)
        release(model)
        del model
        runs, frozen_endpoints = {}, {}
        # Do not inspect ANY audit target until every optimizer trajectory and both selectors finish.
        for seed in effective['seeds']:
            runs[str(seed)] = {}
            initial_hashes = set()
            for arm in alignment.ARMS:
                for loss in alignment.LOSSES:
                    name = fit_name(arm, loss, seed)
                    directory = partial / name
                    directory.mkdir()
                    model = native()
                    progress('fit_started', seed=seed, arm=arm, loss=loss)
                    detail = fit(model, arm, loss, seed, datasets, plan, selection, effective, device, directory,
                        lambda stage, **fields: progress(stage, seed=seed, arm=arm, loss=loss, **fields), identity_hash,
                        resume_from / name if resume_from else None)
                    initial_hashes.add(detail['initial_adapter_sha256'])
                    if len(initial_hashes) != 1:
                        raise ValueError('Paired arm/loss initializations differ')
                    runs[str(seed)][name] = detail
                    for selector in alignment.SELECTORS:
                        key = f'{name}/selected_{selector}.pt'
                        frozen_endpoints[key] = base.sha256(partial / key)
                    release(model)
                    del model
        base.write_json(partial / 'selected_endpoints_frozen.json', frozen_endpoints)
        progress('all_selectors_frozen', fits=sum(len(v) for v in runs.values()), endpoints=len(frozen_endpoints))

        model = native()
        for phase in ('fit', 'audit'):
            reference[phase] = evaluate_period(model, phase, reference_dir, 'A')
        base.assert_backbone(model, backbone_hash)
        release(model)
        del model
        comparisons, csv_rows = {}, []
        times = {int(row['sample_index']): row['t0'] for row in positive}
        for seed in effective['seeds']:
            learned = {p: {} for p in ('fit', 'selection', 'audit')}
            for arm in alignment.ARMS:
                for loss in alignment.LOSSES:
                    name = fit_name(arm, loss, seed)
                    detail = runs[str(seed)][name]
                    model = native()
                    adapter = vector.attach_adapter(model, arm, effective['training']['node_hidden_width'])
                    for selector in alignment.SELECTORS:
                        key = f'{name}/selected_{selector}.pt'
                        if base.sha256(partial / key) != frozen_endpoints[key]:
                            raise ValueError('Frozen selected endpoint changed')
                        saved = torch.load(partial / key, map_location='cpu', weights_only=True)
                        adapter.load_state_dict(saved['adapter_state'], strict=True)
                        expected_hash = detail['selectors'][selector]['adapter_state_sha256']
                        if alignment.tensor_hash(adapter.state_dict()) != expected_hash:
                            raise ValueError('Selected adapter state mismatch')
                        label = alignment.endpoint(arm, loss, selector)
                        destination = partial / name / f'select_{selector}'
                        destination.mkdir()
                        endpoint_metrics = {}
                        for phase in ('fit', 'selection', 'audit'):
                            records = evaluate_period(model, phase, destination, label)
                            learned[phase][label] = records
                            endpoint_metrics[phase] = {c: metrics(r) for c, r in records.items()}
                            for cohort, record in records.items():
                                a = reference[phase][cohort]
                                for field in ('ids', 'source_indices', 'candidate_mask', 'counts', 'prediction_counts', 'regions'):
                                    if not np.array_equal(record[field], a[field]):
                                        raise ValueError('Evaluation sample/support changed')
                                if saved['epoch'] == 0 and not np.array_equal(record['errors'], a['errors']):
                                    raise ValueError('Epoch-0 endpoint must be exactly A')
                        if endpoint_metrics['selection'] != detail['selectors'][selector]['selection_metrics']:
                            raise ValueError('Selected selection metrics failed exact same-batch replay')
                        if alignment.tensor_hash(adapter.state_dict()) != expected_hash:
                            raise ValueError('Evaluation changed adapter state')
                        base.assert_backbone(model, backbone_hash)
                        detail['selectors'][selector]['phase_metrics'] = endpoint_metrics
                    release(model)
                    del model
            comparisons[str(seed)] = {}
            if not check:
                for phase in ('fit', 'selection', 'audit'):
                    progress('phase_statistics_started', seed=seed, phase=phase)
                    analysis, rows, weekly = alignment.compare_phase(reference[phase], learned[phase], times, inherited)
                    comparisons[str(seed)][phase] = analysis
                    csv_rows.extend({'seed': seed, 'phase': phase, **row} for row in rows)
                    base.save_arrays(partial / f'{phase}_weekly_s{seed}.npz', weekly)
            progress('seed_evaluation_complete', seed=seed)
        if csv_rows:
            with (partial / 'comparisons.csv').open('w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
                writer.writeheader()
                writer.writerows(csv_rows)
        # Code changes during a long run invalidate its claimed identity.
        if code_hashes != {str(p.relative_to(REPO)): base.sha256(p) for p in code_paths()}:
            raise ValueError('Source code changed during v12k run')
        summary = {'status': 'ENGINEERING_CHECK_PASS' if check else 'VECTOR_OBJECTIVE_ALIGNMENT_COMPARISON_COMPLETE',
            'protocol_id': frozen['protocol_id'], 'protocol_sha256': PROTOCOL_SHA256,
            'engineering_check': check, 'frozen_protocol': frozen, 'inherited_protocol': inherited,
            'effective_training': effective['training'], **frozen['information_boundary'],
            'model_training_performed': True, 'training_scope': 'icsf_vector_adapter_only',
            'recommendation': 'ENGINEERING_ONLY' if check else frozen['decision'],
            'main_training_ready': False, 'all_selectors_frozen_before_audit_evaluation': True,
            'paired_initialization_exact': True, 'runs': runs, 'phase_comparisons': comparisons,
            'baseline': {p: {c: metrics(r) for c, r in v.items()} for p, v in reference.items()},
            'phase_samples': {p: {c: len(v) for c, v in x['indices'].items()} for p, x in plan.items()},
            'inputs': hashes, 'code_sha256': code_hashes, 'recovery_source': str(resume_from) if resume_from else None,
            'environment': {'host': socket.gethostname(), 'python': sys.version, 'device': str(device),
                'torch': str(torch.__version__), 'numpy': np.__version__, 'threads': torch.get_num_threads(),
                'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
                'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
                'tf32': torch.backends.cuda.matmul.allow_tf32,
                'git_head': base.subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()},
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
    print('Saved v12k objective alignment:', output / 'summary.json', flush=True)
    return summary


def report(summary):
    print('status:', summary['status'])
    print('protocol_sha256:', summary['protocol_sha256'])
    print('recommendation:', summary['recommendation'])
    print('GAIN = A MAE - endpoint MAE; raw units. Fit/selection are conditional; audit is reused development.')
    for seed, fits in summary['runs'].items():
        for name, detail in fits.items():
            for selector, selected in detail['selectors'].items():
                print(f'\n[{name} selector={selector}] epoch={selected["selected_epoch"]}')
                for region in alignment.REGIONS:
                    gains = {phase: summary['baseline'][phase]['incident_full']['mae'][region]
                        - selected['phase_metrics'][phase]['incident_full']['mae'][region]
                        for phase in ('fit', 'selection', 'audit')}
                    print(region, 'full_positive_gain=', gains)
                if not summary['engineering_check']:
                    comparison = alignment.endpoint(detail['arm'], detail['loss'], selector) + '_vs_A'
                    audit = summary['phase_comparisons'][seed]['audit']['results']
                    for cohort in base.COHORTS:
                        effect = audit[cohort]['regions']['candidate_h1_h6']['comparisons'][comparison]
                        ci = effect['intervals']['four_week_block']['pooled']
                        print(cohort, 'early_gain=', effect['gain_raw_mae'],
                              'equal_window_gain=', effect['equal_forecast_window_gain_raw_mae'],
                              'early_block_CI=', [ci['ci_low'], ci['ci_high']])
        if not summary['engineering_check']:
            for name, effect in summary['phase_comparisons'][seed]['audit']['results']['incident_full']['regions']['candidate_h1_h6']['comparisons'].items():
                if not name.endswith('_vs_A'):
                    ci = effect['intervals']['four_week_block']['pooled']
                    print('paired_factor_contrast', name, 'gain=', effect['gain_raw_mae'],
                          'block_CI=', [ci['ci_low'], ci['ci_high']])


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
