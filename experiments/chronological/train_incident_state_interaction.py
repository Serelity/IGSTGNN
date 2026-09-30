"""v12f: frozen-A strength, local-state vector and context-conditioned vector adapters."""

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
from experiments.chronological import train_incident_strength_gate as base
from src.models.incident_state_interaction import ARMS, StateInteractionICSF, attach_adapter

PROTOCOL = Path(__file__).with_name('incident_state_interaction_v12f.json')
PROTOCOL_SHA256 = 'd782ed11d2044583d7dda2d0025a6fe24c06ac512ab9465258ccfb894ee0379c'
DIAGNOSTICS = ('correction_rms', 'relative_correction_norm', 'relative_perpendicular_norm',
               'postnorm_change_rms', 'context_modifier_channel_mean', 'context_modifier_channel_std')


def load_protocol():
    if base.sha256(PROTOCOL) != PROTOCOL_SHA256:
        raise ValueError('Frozen v12f protocol changed')
    protocol = json.loads(PROTOCOL.read_text())
    if base.sha256(base.PROTOCOL) != protocol['v12c_protocol_sha256']:
        raise ValueError('Inherited v12c protocol changed')
    return protocol


def state_hash(module):
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def representation_arrays(adapter):
    delta = adapter.last_delta.double()
    injection = adapter.last_injection.double()
    energy = injection.square().sum(-1)
    defined = energy > 0
    denominator = torch.where(defined, energy, 1.)
    projection = (delta * injection).sum(-1) / denominator
    perpendicular = delta - projection[..., None] * injection
    def relative(vector):
        ratio = (vector.square().sum(-1) / denominator).sqrt()
        return torch.where(defined, ratio, torch.nan).cpu().numpy()
    modifier = adapter.last_context_modifier.double()
    return {
        'correction_rms': delta.square().mean(-1).sqrt().cpu().numpy(),
        'relative_correction_norm': relative(delta),
        'relative_perpendicular_norm': relative(perpendicular),
        'postnorm_change_rms': adapter.last_postnorm_delta.double().square().mean(-1).sqrt().cpu().numpy(),
        'context_modifier_channel_mean': modifier.mean(-1).cpu().numpy(),
        'context_modifier_channel_std': modifier.std(-1, unbiased=False).cpu().numpy(),
        'zero_injection': (~defined).cpu().numpy(),
    }


def evaluate(model, dataset, indices, batch_size, device, progress=lambda *a, **k: None):
    model.eval()
    pieces = {}
    adapter = model.icsf_module if isinstance(model.icsf_module, StateInteractionICSF) else None
    with torch.no_grad():
        for step, raw in enumerate(base.loader(dataset, indices, batch_size)):
            batch = base.device_batch(raw, device)
            prediction = model(batch['x'], incident_data=batch['incident'])
            prediction = prediction * dataset.scaler['std'] + dataset.scaler['mean']
            values = base.statistics(prediction, batch)
            values.update(ids=raw['positive_sample_index'].numpy(),
                          source_indices=raw['source_index'].numpy(),
                          candidate_mask=raw['candidate_mask'].numpy())
            if adapter is not None:
                values.update(representation_arrays(adapter))
                if adapter.last_gate is not None:
                    values['gates'] = adapter.last_gate.squeeze(-1).cpu().numpy()
                adapter.clear_observations()
            for key, value in values.items():
                pieces.setdefault(key, []).append(value)
            if step % 10 == 0:
                progress('evaluation_progress', batches_completed=step + 1, samples_total=len(indices))
    return {**{key: np.concatenate(value) for key, value in pieces.items()}, 'regions': base.REGIONS}


def describe(values):
    finite = values[np.isfinite(values)]
    return {'defined_nodes': int(len(finite)), 'undefined_nodes': int(len(values) - len(finite)),
            'mean': float(finite.mean()) if len(finite) else None,
            'std': float(finite.std()) if len(finite) else None,
            'q05_q50_q95': np.quantile(finite, [.05, .5, .95]).tolist() if len(finite) else None}


def metric_summary(record):
    totals, counts = record['errors'].sum(0), record['counts'].sum(0)
    candidate = record['candidate_mask']
    observations = ({k: describe(record[k][candidate]) for k in DIAGNOSTICS}
                    if 'correction_rms' in record else None)
    if observations is not None:
        observations['zero_injection_candidate_nodes'] = int(record['zero_injection'][candidate].sum())
    return {'samples': len(record['ids']),
            'mae': {name: float(totals[i] / counts[i]) if counts[i] else None
                    for i, name in enumerate(base.REGIONS)},
            'candidate_gate': describe(record['gates'][candidate]) if 'gates' in record else None,
            'candidate_representation': observations}


def vector_train_epoch(model, adapter, optimizer, dataset, indices, settings, device, seed, epoch, progress):
    model.eval()
    ordered = np.random.default_rng(seed * 1000 + epoch).permutation(indices).tolist()
    before = {name: p.detach().clone() for name, p in adapter.named_parameters()}
    error_sum, valid_count, maximum_gradient, penalty_sum = 0., 0, 0., 0.
    for step, raw in enumerate(base.loader(dataset, ordered, settings['batch_size'])):
        batch = base.device_batch(raw, device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(batch['x'], incident_data=batch['incident'])
        target = (batch['y_flow'] - dataset.scaler['mean']) / dataset.scaler['std']
        loss = base.masked_flow_mae(prediction, target, batch['y_valid'])
        candidate = batch['candidate_mask']
        if not candidate.any():
            raise ValueError('Fit batch has no incident support')
        penalty = model.icsf_module.last_unit_residual[candidate].square().mean()
        objective = loss + settings['gate_identity_penalty'] * penalty
        objective.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in adapter.parameters()):
            raise ValueError('Absent/nonfinite vector adapter gradient')
        norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), settings['clip_grad_norm'])
        if not torch.isfinite(norm):
            raise ValueError('Nonfinite vector adapter gradient norm')
        maximum_gradient = max(maximum_gradient, float(norm))
        optimizer.step()
        if any(not torch.isfinite(p).all() for p in adapter.parameters()):
            raise ValueError('Nonfinite vector adapter parameters')
        cells = int(batch['y_valid'].sum())
        error_sum += float(loss.detach()) * cells
        valid_count += cells
        penalty_sum += float(penalty.detach())
        model.icsf_module.clear_observations()
        if step % settings['progress_every_batches'] == 0:
            progress('training_progress', epoch=epoch, batches_completed=step + 1,
                     samples_total=len(indices), loss_standardized=float(loss.detach()))
    changed = any(not torch.equal(before[name], p) for name, p in adapter.named_parameters())
    if epoch == 1 and (not changed or maximum_gradient == 0):
        raise ValueError('Vector adapter had no effective learning update')
    return {'mae_standardized': error_sum / valid_count, 'maximum_gradient_norm': maximum_gradient,
            'mean_minibatch_identity_penalty': penalty_sum / (step + 1),
            'gate_parameters_changed': changed, 'optimizer_steps': step + 1}


def compare(reference, arms, times, protocol):
    records = {}
    for cohort, a in reference.items():
        ordered = [a] + [arms[arm][cohort] for arm in ARMS]
        for other in ordered[1:]:
            for key in ('ids', 'source_indices', 'counts', 'prediction_counts', 'candidate_mask'):
                if not np.array_equal(a[key], other[key]):
                    raise ValueError('Adapter-arm sample/support mismatch')
        records[cohort] = {**{k: a[k] for k in ('ids', 'counts', 'prediction_counts', 'regions')},
                           'errors': np.stack([r['errors'] for r in ordered], axis=1)}
    comparisons = {arm + '_vs_A': ['A', arm] for arm in ARMS}
    comparisons.update({arm + '_vs_strength': ['strength', arm] for arm in ARMS[1:]})
    comparisons['interaction_vector_vs_state_vector'] = ['state_vector', 'interaction_vector']
    return base.analyze(records, times, {'paths': ['A', *ARMS], 'bootstrap': protocol['bootstrap'],
                                        'comparisons': comparisons})


def run(data_dir, primary_dir, secondary_dir, checkpoint, output, device='cuda:0', check=False, resume_from=None):
    frozen, inherited = load_protocol(), base.load_protocol()
    effective = copy.deepcopy(inherited)
    if check:
        effective['seeds'] = inherited['check']['seeds']
        effective['training']['epochs'] = inherited['check']['epochs']
    output = Path(output).resolve()
    partial = output.with_name(output.name + '.partial')
    if output.exists() or partial.exists():
        raise FileExistsError('Preserve output/partial; choose a new run name')
    if resume_from is not None:
        resume_from = Path(resume_from).resolve()
        if check or not resume_from.is_dir() or output.is_relative_to(resume_from):
            raise ValueError('Recovery requires a full-run source and a separate destination')
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
        identity_protocol = base.mechanisms.load_protocol(base.mechanisms.PROTOCOL)
        if base.sha256(base.mechanisms.PROTOCOL) != inherited['v12a_protocol_sha256']:
            raise ValueError('Inherited backbone protocol changed')
        progress('verifying_inputs')
        baseline, hashes = base.mechanisms.verify_inputs(data_dir, primary_dir, secondary_dir, checkpoint, identity_protocol)
        positive = base.read_csv(Path(data_dir) / 'train_manifest.csv')
        plan = base.make_plan(positive, base.read_csv(Path(primary_dir) / 'train_control_manifest.csv'),
                             base.read_csv(Path(secondary_dir) / 'train_second_control_manifest.csv'), inherited)
        base.write_json(partial / 'eligibility.json', plan)
        if check:
            plan = copy.deepcopy(plan)
            for phase in plan.values():
                phase['indices'] = {c: v[:inherited['check']['samples_per_cohort_period']] for c, v in phase['indices'].items()}
        fit_indices = plan['fit']['indices']['incident_full']
        probe = [fit_indices[i] for i in np.linspace(0, len(fit_indices) - 1,
                 min(len(fit_indices), frozen['representation_probe']['samples']), dtype=int)]
        source_files = [Path(__file__), PROTOCOL, Path(base.__file__), base.PROTOCOL,
            Path(base.mechanisms.__file__), base.mechanisms.PROTOCOL,
            REPO / 'experiments/chronological/gate_recovery.py',
            REPO / 'experiments/chronological/audit_architecture_regions.py',
            REPO / 'experiments/chronological/materialize_incident_branch.py',
            REPO / 'experiments/chronological/smoke.py', REPO / 'experiments/chronological/train.py',
            REPO / 'src/utils/chronological.py', Path(__file__).with_name('run_incident_state_interaction.sh')]
        source_files += sorted((REPO / 'src/models').rglob('*.py'))
        code_hashes = {str(p.relative_to(REPO)): base.sha256(p) for p in source_files}
        identity = {'protocol_sha256': PROTOCOL_SHA256, 'engineering_check': check, 'inputs': hashes,
                    'checkpoint_sha256': base.sha256(checkpoint), 'code_sha256': code_hashes,
                    'indices': {p: v['indices'] for p, v in plan.items()}, 'probe_indices': probe}
        if resume_from and json.loads((resume_from / 'run_identity.json').read_text()) != identity:
            raise ValueError('Recovery source input/code/protocol/sample identity changed')
        base.write_json(partial / 'run_identity.json', identity)
        datasets = base.make_datasets(data_dir, primary_dir, secondary_dir, baseline)
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        def native():
            model = base.make_model(Path(data_dir), len(datasets['incident_full'].station_ids), device, 'fixed')
            model.load_state_dict(state, strict=True)
            if sum(p.numel() for p in model.parameters()) != baseline['checkpoint']['parameters']:
                raise ValueError('Backbone size changed')
            return model.eval().requires_grad_(False)
        reference = {}
        model = native()
        for phase in ('selection', 'audit'):
            reference[phase] = {}
            for cohort in base.COHORTS:
                record = evaluate(model, datasets[cohort], plan[phase]['indices'][cohort],
                    effective['training']['evaluation_batch_size'], device,
                    lambda stage, **fields: progress(stage, arm='A', phase=phase, cohort=cohort, **fields))
                reference[phase][cohort] = record
                base.save_arrays(partial / f'{phase}_A_{cohort}.npz', record)
        del model
        selection = {c: metric_summary(r) for c, r in reference['selection'].items()}
        audit_ids = set(reference['audit']['incident_full']['ids'].tolist())
        times = {int(row['sample_index']): row['t0'] for row in positive if int(row['sample_index']) in audit_ids}
        runs, comparisons = {}, {}
        for seed in effective['seeds']:
            runs[str(seed)], learned = {}, {}
            vector_initializations = []
            for arm in ARMS:
                directory = partial / f'{arm}_s{seed}'
                directory.mkdir()
                arm_protocol = copy.deepcopy(effective)
                arm_protocol['training']['adapter_arm'] = arm
                arm_protocol['training']['objective'] = 'Inherited global MAE plus 0.001 normalized adapter energy'
                model = native()
                def update(stage, **fields):
                    progress(stage, arm=arm, seed=seed, **fields)
                def diagnostic(current, adapter, label):
                    before = state_hash(adapter)
                    record = evaluate(current, datasets['incident_full'], probe,
                        frozen['representation_probe']['batch_size'], device,
                        lambda stage, **fields: update(stage, snapshot=label, **fields))
                    if state_hash(adapter) != before:
                        raise ValueError('Representation probe changed adapter state')
                    base.save_arrays(directory / f'representation_{label}.npz', record)
                    return {'adapter_state_sha256': before, 'sample_ids': record['ids'].tolist(),
                            **metric_summary(record)}
                update('adapter_started')
                detail = base.fit_variant(model, arm, seed, datasets, plan, selection, arm_protocol, device,
                    directory, update, resume_from / directory.name if resume_from else None,
                    epoch_trainer=base.train_epoch if arm == 'strength' else vector_train_epoch,
                    protocol_hash=PROTOCOL_SHA256, diagnostic=diagnostic, adapter_factory=attach_adapter,
                    evaluator=evaluate, summarizer=metric_summary, diagnostic_key='representation_diagnostics')
                if arm != 'strength':
                    vector_initializations.append(detail['representation_diagnostics']['initial']['adapter_state_sha256'])
                    if len(set(vector_initializations)) != 1:
                        raise ValueError('Vector arms did not share exact initialization')
                learned[arm], detail['audit'] = {}, {}
                for cohort in base.COHORTS:
                    record = evaluate(model, datasets[cohort], plan['audit']['indices'][cohort],
                        effective['training']['evaluation_batch_size'], device,
                        lambda stage, **fields: update(stage, phase='audit', cohort=cohort, **fields))
                    learned[arm][cohort] = record
                    detail['audit'][cohort] = metric_summary(record)
                    base.save_arrays(directory / f'audit_{cohort}.npz', record)
                runs[str(seed)][arm] = detail
                base.write_json(directory / 'summary.json', detail)
                model.icsf_module.clear_observations()
                del model
                gc.collect()
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
            if not check:
                analysis, _, weekly = compare(reference['audit'], learned, times, inherited)
                comparisons[str(seed)] = analysis
                base.save_arrays(partial / f'audit_weekly_s{seed}.npz', weekly)
                base.write_json(partial / f'audit_comparisons_s{seed}.json', analysis)
            progress('seed_complete', seed=seed)
        summary = {'status': 'ENGINEERING_CHECK_PASS' if check else 'INCIDENT_STATE_INTERACTION_COMPARISON_COMPLETE',
            'protocol_id': frozen['protocol_id'], 'protocol_sha256': PROTOCOL_SHA256,
            'engineering_check': check, 'frozen_protocol': frozen, 'inherited_protocol': inherited,
            'effective_training': effective['training'], **frozen['information_boundary'],
            'model_training_performed': True, 'main_training_ready': False,
            'recommendation': 'ENGINEERING_ONLY' if check else frozen['decision'],
            'training_scope': 'icsf_adapter_only', 'runs': runs, 'audit_comparisons': comparisons,
            'baseline_audit': {c: metric_summary(r) for c, r in reference['audit'].items()},
            'phase_samples': {p: {c: len(v) for c, v in x['indices'].items()} for p, x in plan.items()},
            'vector_paired_initialization_exact': True, 'inputs': hashes, 'code_sha256': code_hashes,
            'recovery_source': str(resume_from) if resume_from else None,
            'environment': {'host': socket.gethostname(), 'device': str(device), 'python': sys.version,
                'torch': torch.__version__, 'numpy': np.__version__, 'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
                'threads': torch.get_num_threads(), 'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
                'tf32': torch.backends.cuda.matmul.allow_tf32,
                'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                'git_head': base.subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()},
            'outputs': {str(p.relative_to(partial)): base.sha256(p) for p in partial.rglob('*')
                        if p.is_file() and p.name != 'progress.json'}}
        base.write_json(partial / 'summary.json', summary)
        if output.exists():
            raise FileExistsError('Final output appeared; preserve partial')
        partial.rename(output)
    except BaseException as error:
        if partial.exists():
            base.write_json(partial / 'failure.json', {'error': str(error), 'traceback': traceback.format_exc()})
        raise
    report(summary)
    print('Saved v12f comparison:', output / 'summary.json', flush=True)
    return summary


def report(summary):
    base.report(summary)
    for seed, arms in summary['runs'].items():
        for arm, detail in arms.items():
            print(f'\nREPRESENTATION PROBE seed={seed} arm={arm}; fixed fit subset only')
            for label, result in detail['representation_diagnostics'].items():
                print(label, 'mae=', result['mae'], 'candidate_representation=', result['candidate_representation'])


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
        if args.summary.is_file():
            report(json.loads(args.summary.read_text()))
        else:
            print('INCOMPLETE: no final summary. Check .job/run.log and preserve .partial.')
    else:
        run(args.data_dir, args.primary_control_dir, args.secondary_control_dir, args.checkpoint,
            args.output, args.device, args.check, args.resume_from)


if __name__ == '__main__':
    main()
