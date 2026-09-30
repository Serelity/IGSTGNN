"""v12c: frozen-backbone scalar/node ICSF adapters with temporal development splits."""

import argparse
import copy
import gc
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.chronological import audit_architecture_mechanisms as mechanisms
from experiments.chronological.audit_architecture_regions import analyze, write_json
from experiments.chronological.audit_matched_controls import read_csv, sha256
from experiments.chronological.materialize_incident_branch import FullPositiveDataset, MatchedCounterfactualDataset
from experiments.chronological.smoke import make_model, device_batch, set_seed
from experiments.chronological.train import configure_determinism, save_checkpoint
from experiments.chronological.gate_recovery import cpu_tree, memory_snapshot, partial_report, recover
from src.models.incident_strength_gate import attach_gate
from src.utils.chronological import masked_flow_mae

PROTOCOL = Path(__file__).with_name('incident_strength_gate_v12c.json')
PROTOCOL_SHA256 = '99a9ee432d858f9568e32132e4e2e5388524184efea5da4d16c99abd7b875def'
COHORTS = ('incident_full', 'incident', 'primary_control', 'secondary_control')
REGIONS = list(mechanisms.REGIONS) + ['candidate_h1_h6', 'candidate_all', 'noncandidate_all']


def load_protocol(path=PROTOCOL):
    if sha256(path) != PROTOCOL_SHA256:
        raise ValueError('Frozen v12c protocol changed')
    return json.loads(Path(path).read_text())


def inside(row, bounds, clock):
    start, end = map(datetime.fromisoformat, bounds)
    t0 = datetime.fromisoformat(row[clock])
    first = datetime.fromisoformat(row['support_start'])
    last = datetime.fromisoformat(row['support_end_exclusive'])
    return start <= first < last <= end and start <= t0 < end


def make_plan(positive, primary, secondary, protocol):
    """Time-only eligibility, with whole paired traffic supports inside each period."""
    ids = [int(row['sample_index']) for row in positive]
    common = [int(row['positive_sample_index']) for row in secondary]
    full_pos = {sample: i for i, sample in enumerate(ids)}
    c1 = {int(row['positive_sample_index']): row for row in primary}
    if len(full_pos) != len(ids) or len(set(common)) != len(common) or len(c1) != len(primary):
        raise ValueError('Duplicate source identities')
    if any(row['split'] != 'train' for rows in (positive, primary, secondary) for row in rows):
        raise ValueError('v12c only uses train manifests')
    for i, row in enumerate(secondary):
        sample = int(row['positive_sample_index'])
        if sample not in full_pos or sample not in c1 or int(row['control_index']) != i:
            raise ValueError('Matched identity or source order changed')
        pos, first = positive[full_pos[sample]], c1[sample]
        if len({pos['incident_id'], first['incident_id'], row['incident_id']}) != 1 or len(
                {pos['t0'], first['positive_t0'], row['positive_t0']}) != 1:
            raise ValueError('Matched positive identity/clock changed')
    periods = list(protocol['periods'].values())
    if any(datetime.fromisoformat(a[1]) >= datetime.fromisoformat(b[0])
           for a, b in zip(periods, periods[1:])):
        raise ValueError('Periods must be chronological with an embargo gap')
    plan = {}
    for phase, bounds in protocol['periods'].items():
        retained = [i for i, row in enumerate(positive) if inside(row, bounds, 't0')]
        if not retained:
            raise ValueError(f'Empty full-positive period: {phase}')
        allowed = {ids[i] for i in retained}
        matched = [i for i, row in enumerate(secondary)
                   if int(row['positive_sample_index']) in allowed
                   and inside(row, bounds, 'candidate_t0')
                   and inside(c1[int(row['positive_sample_index'])], bounds, 'candidate_t0')]
        if phase != 'fit' and not matched:
            raise ValueError(f'Empty matched period: {phase}')
        indices = {'incident_full': retained}
        if phase != 'fit':
            indices.update({cohort: matched for cohort in COHORTS[1:]})
        plan[phase] = {
            'bounds': bounds, 'indices': indices,
            'positive_ids': [ids[i] for i in retained],
            'matched_ids': [common[i] for i in matched] if phase != 'fit' else [],
            'excluded_positive_ids': [sample for sample in ids if sample not in allowed],
            'excluded_common_ids': [sample for i, sample in enumerate(common) if i not in set(matched)]
                                   if phase != 'fit' else [],
        }
    return plan


def backbone_hash(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if name.startswith('icsf_module.gate.'):
            continue
        # Normalize adapter wrapping so this equals the native checkpoint hash.
        name = name.replace('icsf_module.base.', 'icsf_module.')
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def make_datasets(data_dir, primary_dir, secondary_dir, baseline):
    datasets = {'incident_full': FullPositiveDataset(data_dir, 'train',
        baseline['expected_positive_samples']['train'], baseline['expected_sensor_count'])}
    for cohort in COHORTS[1:]:
        datasets[cohort] = MatchedCounterfactualDataset(data_dir, primary_dir, secondary_dir,
            'train', cohort, baseline['expected_common_triples']['train'],
            baseline['expected_sensor_count'], expected_frozen_only_pairs=
            baseline['candidate_mask_compatibility']['expected_frozen_only_pairs']['train'])
    return datasets


def loader(dataset, indices, batch_size):
    return DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False, num_workers=0)


def masks_for(prediction, candidate):
    masks = mechanisms.region_masks(candidate, prediction)
    return masks + [masks[1] | masks[2], masks[1] | masks[2] | masks[3], masks[4] | masks[5]]


def statistics(prediction, batch):
    masks = masks_for(prediction, batch['candidate_mask'])
    valid = batch['y_valid']
    if not torch.isfinite(prediction).all() or not torch.isfinite(batch['y_flow'][valid]).all():
        raise ValueError('Nonfinite prediction or valid target')
    error = (prediction.double() - batch['y_flow'].double()).abs()
    counts = torch.stack([(mask & valid).sum((1, 2, 3)) for mask in masks], -1)
    totals = torch.stack([torch.where(mask & valid, error, 0.).sum((1, 2, 3)) for mask in masks], -1)
    predicted = torch.stack([mask.sum((1, 2, 3)) for mask in masks], -1)
    return {'errors': totals.cpu().numpy(), 'counts': counts.cpu().numpy(),
            'prediction_counts': predicted.cpu().numpy()}


def evaluate(model, dataset, indices, batch_size, device, progress=lambda *a, **k: None):
    model.eval()
    pieces = {}
    with torch.no_grad():
        for step, raw in enumerate(loader(dataset, indices, batch_size)):
            batch = device_batch(raw, device)
            prediction = model(batch['x'], incident_data=batch['incident'])
            prediction = prediction * dataset.scaler['std'] + dataset.scaler['mean']
            values = statistics(prediction, batch)
            gates = (model.icsf_module.last_gate.squeeze(-1) if hasattr(model.icsf_module, 'last_gate')
                     else torch.ones_like(batch['candidate_mask'], dtype=torch.float32))
            values.update(ids=raw['positive_sample_index'].numpy(),
                          source_indices=raw['source_index'].numpy(),
                          gates=gates.detach().cpu().numpy(),
                          candidate_mask=raw['candidate_mask'].numpy())
            for key, value in values.items():
                pieces.setdefault(key, []).append(value)
            if step % 10 == 0:
                progress('evaluation_progress', batches_completed=step + 1, samples_total=len(indices))
    return {**{key: np.concatenate(value) for key, value in pieces.items()}, 'regions': REGIONS}


def metric_summary(record):
    counts = record['counts'].sum(0)
    totals = record['errors'].sum(0)
    gates = record['gates'][record['candidate_mask']]
    return {'samples': len(record['ids']),
            'mae': {name: float(totals[i] / counts[i]) if counts[i] else None
                    for i, name in enumerate(REGIONS)},
            'candidate_gate': {'mean': float(gates.mean()), 'std': float(gates.std()),
                               'q05_q50_q95': np.quantile(gates, [.05, .5, .95]).tolist()}
                              if len(gates) else None}


def selection_eligible(current, baseline, protocol):
    threshold = protocol['selection']['maximum_relative_harm']
    checks = {}
    for cohort in protocol['selection']['protected_cohorts']:
        for region in protocol['selection']['protected_regions']:
            a, b = current[cohort]['mae'][region], baseline[cohort]['mae'][region]
            checks[f'{cohort}/{region}'] = bool(a is not None and b is not None
                and np.isfinite(a) and np.isfinite(b) and a <= b * (1 + threshold) + 1e-12)
    return all(checks.values()), checks


def train_epoch(model, gate, optimizer, dataset, indices, settings, device, seed, epoch, progress):
    model.eval()  # Dropout/buffers of the frozen backbone never enter training mode.
    ordered = np.random.default_rng(seed * 1000 + epoch).permutation(indices).tolist()
    before = {name: p.detach().clone() for name, p in gate.named_parameters()}
    error_sum = 0.
    valid_count = 0
    max_gradient = 0.
    for step, raw in enumerate(loader(dataset, ordered, settings['batch_size'])):
        batch = device_batch(raw, device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(batch['x'], incident_data=batch['incident'])
        target = (batch['y_flow'] - dataset.scaler['mean']) / dataset.scaler['std']
        loss = masked_flow_mae(prediction, target, batch['y_valid'])
        strength = model.icsf_module.last_gate.squeeze(-1)
        candidate = batch['candidate_mask']
        if not candidate.any():
            raise ValueError('Fit batch has no incident support')
        objective = loss + settings['gate_identity_penalty'] * (strength[candidate] - 1).square().mean()
        objective.backward()
        grads = [p.grad for p in gate.parameters()]
        if any(g is None or not torch.isfinite(g).all() for g in grads):
            raise ValueError('Absent/nonfinite gate gradient')
        norm = torch.nn.utils.clip_grad_norm_(gate.parameters(), settings['clip_grad_norm'])
        if not torch.isfinite(norm):
            raise ValueError('Nonfinite gate gradient norm')
        max_gradient = max(max_gradient, float(norm))
        optimizer.step()
        if any(not torch.isfinite(p).all() for p in gate.parameters()):
            raise ValueError('Nonfinite gate parameter')
        n = int(batch['y_valid'].sum())
        error_sum += float(loss.detach()) * n
        valid_count += n
        if step % settings['progress_every_batches'] == 0:
            progress('training_progress', epoch=epoch, batches_completed=step + 1,
                     samples_total=len(indices), loss_standardized=float(loss.detach()))
    changed = any(not torch.equal(before[name], p) for name, p in gate.named_parameters())
    if epoch == 1 and (max_gradient == 0 or not changed):
        raise ValueError('Gate had no effective learning update')
    return {'mae_standardized': error_sum / valid_count, 'maximum_gradient_norm': max_gradient,
            'gate_parameters_changed': changed, 'optimizer_steps': step + 1}


def assert_backbone(model, expected):
    if backbone_hash(model) != expected:
        raise ValueError('Frozen backbone state changed')
    for name, p in model.named_parameters():
        if not name.startswith('icsf_module.gate.') and (p.requires_grad or p.grad is not None):
            raise ValueError(f'Backbone gradient/parameter enabled: {name}')


def fit_variant(model, variant, seed, datasets, plan, baseline_selection, protocol, device,
                output, progress, resume_from=None, *, epoch_trainer=None, protocol_hash=None,
                diagnostic=None, adapter_factory=None, evaluator=None, summarizer=None,
                diagnostic_key='parameter_gradient_diagnostics'):
    # Default v12c behavior is unchanged; later controlled comparisons can reuse
    # identical fitting/selection/recovery machinery with an explicit objective.
    epoch_trainer = train_epoch if epoch_trainer is None else epoch_trainer
    protocol_hash = PROTOCOL_SHA256 if protocol_hash is None else protocol_hash
    adapter_factory = attach_gate if adapter_factory is None else adapter_factory
    evaluator = evaluate if evaluator is None else evaluator
    summarizer = metric_summary if summarizer is None else summarizer
    set_seed(seed)
    original_hash = backbone_hash(model)
    first = next(iter(loader(datasets['incident_full'], plan['fit']['indices']['incident_full'], 2)))
    batch = device_batch(first, device)
    model.eval()
    with torch.no_grad():
        original = model(batch['x'], incident_data=batch['incident'])
    gate = adapter_factory(model, variant, protocol['training']['node_hidden_width'])
    with torch.no_grad():
        initial = model(batch['x'], incident_data=batch['incident'])
    if not torch.equal(original, initial):
        raise ValueError('Identity gate did not exactly reproduce native A')
    assert_backbone(model, original_hash)
    settings = protocol['training']
    optimizer = torch.optim.Adam(gate.parameters(), lr=settings['learning_rate'],
        eps=settings['adam_eps'], weight_decay=settings['weight_decay'])
    best = {'epoch': 0, 'selection_metrics': baseline_selection,
            'state': copy.deepcopy(gate.state_dict())}
    diagnostics = {}
    def diagnose(label):
        if diagnostic is not None:
            # Even DataLoader construction consumes RNG state. Observations must
            # leave the fitting sequence untouched.
            devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
            with torch.random.fork_rng(devices=devices):
                diagnostics[label] = diagnostic(model, gate, label)
            assert_backbone(model, original_hash)
    diagnose('initial')
    history = []
    steps = 0
    first_epoch = 1
    recovery = {'method': 'fresh_fit'}
    if resume_from is not None:
        recovered = recover(resume_from, variant, seed, protocol, protocol_hash, original_hash,
                            baseline_selection, best['state'], selection_eligible)
        if recovered is not None:
            if recovered['restart_required']:
                recovery = {k: v for k, v in recovered.items()}
                progress('legacy_fit_restart_required', reason=recovered['reason'],
                         last_epoch=recovered['last_epoch'], best_epoch=recovered['best_epoch'])
            else:
                gate.load_state_dict(recovered['gate_state'], strict=True)
                optimizer.load_state_dict(recovered['optimizer_state'])
                best, history, steps = recovered['best'], recovered['history'], recovered['steps']
                first_epoch = recovered['epoch'] + 1
                recovery = {'method': recovered['method'], 'epoch': recovered['epoch'],
                            'source_hashes': recovered['source_hashes']}
                progress('fit_restored', epoch=recovered['epoch'], best_epoch=best['epoch'])
    write_json(output / 'recovery.json', recovery)

    def publish_epoch(epoch):
        # One atomic bundle is authoritative, including the historical best state.
        save_checkpoint(output / 'last_gate.pt', cpu_tree({
            'format_version': 2, 'variant': variant, 'seed': seed, 'epoch': epoch,
            'gate_state': gate.state_dict(), 'optimizer_state': optimizer.state_dict(),
            'best': best, 'history': history, 'training_settings': settings,
            'protocol_sha256': protocol_hash, 'backbone_state_sha256': original_hash}))
        write_json(output / 'history.json', history)

    if first_epoch > 1:
        publish_epoch(first_epoch - 1)
    for epoch in range(first_epoch, settings['epochs'] + 1):
        training = epoch_trainer(model, gate, optimizer, datasets['incident_full'],
            plan['fit']['indices']['incident_full'], settings, device, seed, epoch, progress)
        steps += training['optimizer_steps']
        current = {}
        for cohort in COHORTS:
            current[cohort] = summarizer(evaluator(model, datasets[cohort],
                plan['selection']['indices'][cohort], settings['evaluation_batch_size'], device,
                lambda stage, **fields: progress(stage, epoch=epoch, phase='selection', cohort=cohort, **fields)))
        eligible, checks = selection_eligible(current, baseline_selection, protocol)
        improved = current['incident_full']['mae']['all'] < best['selection_metrics']['incident_full']['mae']['all']
        if eligible and improved:
            best = {'epoch': epoch, 'selection_metrics': current, 'state': copy.deepcopy(gate.state_dict())}
        assert_backbone(model, original_hash)
        history.append({'epoch': epoch, 'training': training, 'selection': current,
                        'eligible': eligible, 'protection_checks': checks, 'best_epoch': best['epoch']})
        publish_epoch(epoch)
        progress('epoch_complete', epoch=epoch, eligible=eligible, best_epoch=best['epoch'],
                 selection_all_mae=current['incident_full']['mae']['all'])
    diagnose('last')
    gate.load_state_dict(best['state'], strict=True)
    assert_backbone(model, original_hash)
    diagnose('selected')
    save_checkpoint(output / 'selected_gate.pt', {'variant': variant, 'seed': seed,
        'epoch': best['epoch'], 'gate_state': best['state'], 'protocol_sha256': protocol_hash,
        'backbone_state_sha256': original_hash})
    return {'selected_epoch': best['epoch'], 'selection_metrics': best['selection_metrics'],
            'optimizer_steps': steps, 'trainable_parameters': sum(p.numel() for p in gate.parameters()),
            'initial_prediction_exactly_A': True, 'backbone_state_unchanged': True,
            'recovery': recovery, **({diagnostic_key: diagnostics} if diagnostic is not None else {})}


def comparison_records(reference, variants):
    result = {}
    for cohort, baseline in reference.items():
        ordered = [baseline] + [variants[name][cohort] for name in ('scalar', 'node')]
        for other in ordered[1:]:
            for key in ('ids', 'source_indices', 'counts', 'prediction_counts', 'candidate_mask'):
                if not np.array_equal(other[key], baseline[key]):
                    raise ValueError(f'Comparison sample/support changed: {cohort}/{key}')
        result[cohort] = {**{key: baseline[key] for key in ('ids', 'counts', 'prediction_counts', 'regions')},
                          'errors': np.stack([item['errors'] for item in ordered], axis=1)}
    return result


def audit_comparisons(reference, variants, times, protocol):
    settings = {'paths': ['A', 'scalar', 'node'], 'bootstrap': protocol['bootstrap'],
                'comparisons': {'scalar_vs_A': ['A', 'scalar'], 'node_vs_A': ['A', 'node'],
                                'node_vs_scalar': ['scalar', 'node']}}
    return analyze(comparison_records(reference, variants), times, settings)


def save_arrays(path, record):
    np.savez_compressed(path, **{k: np.asarray(v) for k, v in record.items()})


def run(data_dir, primary_dir, secondary_dir, checkpoint, output, device='cuda:0', check=False,
        resume_from=None):
    protocol = load_protocol()
    identity_protocol = mechanisms.load_protocol(mechanisms.PROTOCOL)
    if sha256(mechanisms.PROTOCOL) != protocol['v12a_protocol_sha256']:
        raise ValueError('v12a identity protocol changed')
    output = Path(output)
    if resume_from is not None:
        resume_from = Path(resume_from).resolve()
        if check or not resume_from.is_dir() or output.resolve().is_relative_to(resume_from):
            raise ValueError('Recovery needs an existing full-run directory and a separate new output')
    partial = output.with_name(output.name + '.partial')
    if output.exists() or partial.exists():
        raise FileExistsError('Output/.partial exists; preserve it and choose a new run name')
    partial.mkdir(parents=True)
    started = time.monotonic()
    identity = {'host': socket.gethostname(), 'pid': os.getpid(),
                'started_utc': datetime.now(timezone.utc).isoformat()}

    def progress(stage, **fields):
        value = {**identity, 'stage': stage, 'elapsed_seconds': time.monotonic() - started,
                 'memory': memory_snapshot(device), **fields}
        write_json(partial / 'progress.json', value)
        print(json.dumps(value), flush=True)

    try:
        device = torch.device(device)
        torch.set_num_threads(3)
        configure_determinism(device)
        progress('verifying_train_inputs')
        baseline, hashes = mechanisms.verify_inputs(data_dir, primary_dir, secondary_dir,
                                                    checkpoint, identity_protocol)
        positive = read_csv(Path(data_dir) / 'train_manifest.csv')
        primary = read_csv(Path(primary_dir) / 'train_control_manifest.csv')
        secondary = read_csv(Path(secondary_dir) / 'train_second_control_manifest.csv')
        plan = make_plan(positive, primary, secondary, protocol)
        recovery_source = None
        if resume_from is not None:
            old_plan = resume_from / 'effective_plan.json'
            if json.loads(old_plan.read_text()) != plan:
                raise ValueError('Recovery source uses different sample/period eligibility')
            recovery_source = {'path': str(resume_from), 'effective_plan_sha256': sha256(old_plan)}
        write_json(partial / 'eligibility.json', plan)
        datasets = make_datasets(data_dir, primary_dir, secondary_dir, baseline)
        if check:
            protocol = copy.deepcopy(protocol)
            protocol['training']['epochs'] = protocol['check']['epochs']
            protocol['seeds'] = protocol['check']['seeds']
            for phase in plan.values():
                for cohort, indices in phase['indices'].items():
                    phase['indices'][cohort] = indices[:protocol['check']['samples_per_cohort_period']]
                phase['positive_ids'] = [int(positive[i]['sample_index'])
                                         for i in phase['indices']['incident_full']]
                phase['matched_ids'] = [int(secondary[i]['positive_sample_index'])
                                        for i in phase['indices'].get('incident', [])]
                phase['excluded_positive_ids'] = [int(row['sample_index']) for row in positive
                                                  if int(row['sample_index']) not in phase['positive_ids']]
                phase['excluded_common_ids'] = [int(row['positive_sample_index']) for row in secondary
                                                if int(row['positive_sample_index']) not in phase['matched_ids']]
                phase['engineering_subsample'] = True
        write_json(partial / 'effective_plan.json', plan)
        progress('inputs_verified', phase_samples={phase: {c: len(v) for c, v in item['indices'].items()}
                                                   for phase, item in plan.items()})
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)

        def native():
            model = make_model(Path(data_dir), len(datasets['incident_full'].station_ids), device, 'fixed')
            model.load_state_dict(state, strict=True)
            if sum(p.numel() for p in model.parameters()) != baseline['checkpoint']['parameters']:
                raise ValueError('Backbone parameter count changed')
            return model.eval().requires_grad_(False)

        reference = {}
        model = native()
        for phase in ('selection', 'audit'):
            reference[phase] = {}
            for cohort in COHORTS:
                record = evaluate(model, datasets[cohort], plan[phase]['indices'][cohort],
                    protocol['training']['evaluation_batch_size'], device,
                    lambda stage, **fields: progress(stage, variant='A', phase=phase, cohort=cohort, **fields))
                reference[phase][cohort] = record
                save_arrays(partial / f'{phase}_A_{cohort}.npz', record)
        baseline_selection = {c: metric_summary(r) for c, r in reference['selection'].items()}
        del model
        runs, comparisons = {}, {}
        audit_ids = set(reference['audit']['incident_full']['ids'].tolist())
        times = {int(row['sample_index']): row['t0'] for row in positive if int(row['sample_index']) in audit_ids}
        for seed in protocol['seeds']:
            runs[str(seed)], learned = {}, {}
            for variant in protocol['variants']:
                directory = partial / f'{variant}_s{seed}'
                directory.mkdir()
                model = native()
                def update(stage, **fields):
                    progress(stage, variant=variant, seed=seed, **fields)
                update('adapter_started')
                detail = fit_variant(model, variant, seed, datasets, plan, baseline_selection,
                                     protocol, device, directory, update,
                                     resume_from / directory.name if resume_from is not None else None)
                learned[variant] = {}
                detail['audit'] = {}
                for cohort in COHORTS:
                    record = evaluate(model, datasets[cohort], plan['audit']['indices'][cohort],
                        protocol['training']['evaluation_batch_size'], device,
                        lambda stage, **fields: update(stage, phase='audit', cohort=cohort, **fields))
                    learned[variant][cohort] = record
                    detail['audit'][cohort] = metric_summary(record)
                    save_arrays(directory / f'audit_{cohort}.npz', record)
                runs[str(seed)][variant] = detail
                write_json(directory / 'summary.json', detail)
                model.icsf_module.last_gate = None
                del model
                gc.collect()
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
            if not check:
                analysis, _, weekly = audit_comparisons(reference['audit'], learned, times, protocol)
                comparisons[str(seed)] = analysis
                save_arrays(partial / f'audit_weekly_s{seed}.npz', weekly)
                write_json(partial / f'audit_comparisons_s{seed}.json', analysis)
            progress('seed_complete', seed=seed)
        sources = [Path(__file__), PROTOCOL, REPO / 'src/models/incident_strength_gate.py',
                   Path(mechanisms.__file__), REPO / 'experiments/chronological/audit_architecture_regions.py',
                   REPO / 'experiments/chronological/materialize_incident_branch.py',
                   REPO / 'experiments/chronological/smoke.py', REPO / 'experiments/chronological/train.py',
                   REPO / 'experiments/chronological/gate_recovery.py',
                   REPO / 'src/utils/chronological.py']
        summary = {'status': 'ENGINEERING_CHECK_PASS' if check else 'ICSF_STRENGTH_GATE_EXPERIMENT_COMPLETE',
            'protocol_id': protocol['protocol_id'], 'protocol_sha256': PROTOCOL_SHA256,
            'frozen_protocol': json.loads(PROTOCOL.read_text()), 'effective_training': protocol['training'],
            'engineering_check': check, **protocol['information_boundary'],
            'main_training_ready': False, 'model_training_performed': True, 'training_scope': 'gate_only',
            'recommendation': 'ENGINEERING_ONLY' if check else protocol['decision'],
            'interpretation': protocol['interpretation'], 'runs': runs,
            'audit_comparisons': comparisons,
            'baseline_audit': {c: metric_summary(r) for c, r in reference['audit'].items()},
            'phase_samples': {phase: {c: len(v) for c, v in item['indices'].items()} for phase, item in plan.items()},
            'checkpoint': baseline['checkpoint'], 'inputs': hashes,
            'recovery_source': recovery_source,
            'code_sha256': {str(p.relative_to(REPO)): sha256(p) for p in sources},
            'environment': {**identity, 'python': sys.version, 'numpy': np.__version__,
                'torch': torch.__version__, 'device': str(device),
                'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                'scheduler': {key: os.environ.get(key) for key in
                              ('SLURM_JOB_ID', 'SLURM_STEP_ID', 'PBS_JOBID', 'LSB_JOBID')},
                'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()},
            'outputs': {str(p.relative_to(partial)): sha256(p) for p in sorted(partial.rglob('*'))
                        if p.is_file() and p.name != 'progress.json'}}
        write_json(partial / 'summary.json', summary)
        if output.exists():
            raise FileExistsError('Final output appeared during run')
        partial.rename(output)
    except BaseException as error:
        write_json(partial / 'failure.json', {**identity, 'error': str(error), 'traceback': traceback.format_exc()})
        raise
    report(summary)
    print(f'Saved v12c gate experiment: {output / "summary.json"}', flush=True)
    return summary


def report(summary):
    print('status:', summary['status'])
    print('protocol_sha256:', summary['protocol_sha256'])
    print('phase_samples:', json.dumps(summary['phase_samples']))
    print('TRAIN-PACKAGE ADAPTER DEVELOPMENT; BACKBONE ALREADY SAW THESE PERIODS; NO INDEPENDENT CONFIRMATION')
    for seed, runs in summary['runs'].items():
        for variant, details in runs.items():
            print(f'\n[{variant} seed={seed}] selected_epoch={details["selected_epoch"]} '
                  f'gate_parameters={details["trainable_parameters"]} initial_exact_A={details["initial_prediction_exactly_A"]}')
            for cohort, result in details['audit'].items():
                baseline = summary['baseline_audit'][cohort]
                gains = {}
                for region in ('all', 'candidate_h1_h6', 'candidate_h7_h12', 'noncandidate_all'):
                    a, b = baseline['mae'][region], result['mae'][region]
                    gains[region] = a - b if a is not None and b is not None else None
                print(cohort, 'gains_vs_A=', json.dumps(gains), 'gate=', result['candidate_gate'])
        if seed in summary['audit_comparisons']:
            analysis = summary['audit_comparisons'][seed]
            print(f'PAIRED AUDIT CIs seed={seed} cohort=incident_full (all cohorts stored in summary.json)')
            for region in ('all', 'candidate_h1_h6', 'candidate_h7_h12', 'noncandidate_all'):
                for comparison, value in analysis['results']['incident_full']['regions'][region]['comparisons'].items():
                    print(region, comparison, 'gain=', value['gain_raw_mae'],
                          'week=', value['intervals']['week']['pooled'],
                          'block=', value['intervals']['four_week_block']['pooled'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    execute = commands.add_parser('run')
    for name in ('data-dir', 'primary-control-dir', 'secondary-control-dir', 'checkpoint', 'output'):
        execute.add_argument('--' + name, required=True, type=Path)
    execute.add_argument('--device', default='cuda:0')
    execute.add_argument('--check', action='store_true')
    execute.add_argument('--resume-from', type=Path)
    view = commands.add_parser('report')
    view.add_argument('summary', type=Path)
    args = parser.parse_args()
    if args.command == 'report':
        if args.summary.is_file():
            report(json.loads(args.summary.read_text()))
        else:
            partial_report(args.summary.parent)
    else:
        run(args.data_dir, args.primary_control_dir, args.secondary_control_dir, args.checkpoint,
            args.output, args.device, args.check, args.resume_from)


if __name__ == '__main__':
    main()
