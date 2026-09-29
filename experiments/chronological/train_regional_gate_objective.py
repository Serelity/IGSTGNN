"""v12e: matched node gates differing only in the fit loss's regional weights."""

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

PROTOCOL = Path(__file__).with_name('regional_gate_objective_v12e.json')
PROTOCOL_SHA256 = '2c0ca289ea3da9a6b8b6eb2e9487731f849243ef6330e6be9284ec7a3fd1e364'
REGIONS = ('candidate_h1_h6', 'candidate_h7_h12', 'noncandidate_all')
ARMS = ('global', 'regional')


def load_protocol():
    if base.sha256(PROTOCOL) != PROTOCOL_SHA256:
        raise ValueError('Frozen v12e protocol changed')
    p = json.loads(PROTOCOL.read_text())
    if base.sha256(base.PROTOCOL) != p['v12c_protocol_sha256']:
        raise ValueError('Inherited v12c protocol changed')
    return p


def state_hash(module):
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def region_sums(prediction, target, valid, candidate):
    masks = base.masks_for(prediction, candidate)
    masks = [masks[base.REGIONS.index(r)] & valid for r in REGIONS]
    counts = torch.stack([m.sum() for m in masks])
    if (counts <= 0).any() or counts.sum() != valid.sum():
        raise ValueError('Three nonempty disjoint regions must partition each fit/probe batch')
    if not torch.isfinite(prediction).all() or not torch.isfinite(target[valid]).all():
        raise ValueError('Nonfinite prediction or valid target')
    errors = (prediction - target).abs()
    sums = torch.stack([torch.where(m, errors, 0.).sum() for m in masks])
    return sums, counts


def regional_train_epoch(model, adapter, optimizer, dataset, indices, settings, device, seed, epoch, progress):
    model.eval()
    ordered = np.random.default_rng(seed * 1000 + epoch).permutation(indices).tolist()
    before = {n: p.detach().clone() for n, p in adapter.named_parameters()}
    totals, cells = np.zeros(3), np.zeros(3)
    objective_sum, penalty_sum, maximum_gradient = 0., 0., 0.
    for step, raw in enumerate(base.loader(dataset, ordered, settings['batch_size'])):
        batch = base.device_batch(raw, device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(batch['x'], incident_data=batch['incident'])
        target = (batch['y_flow'] - dataset.scaler['mean']) / dataset.scaler['std']
        sums, counts = region_sums(prediction, target, batch['y_valid'], batch['candidate_mask'])
        loss = (sums / counts).mean()
        strength = model.icsf_module.last_gate.squeeze(-1)[batch['candidate_mask']]
        if not strength.numel():
            raise ValueError('No candidate support in fit batch')
        penalty = (strength - 1).square().mean()
        objective = loss + settings['gate_identity_penalty'] * penalty
        objective.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in adapter.parameters()):
            raise ValueError('Absent/nonfinite regional gate gradient')
        norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), settings['clip_grad_norm'])
        if not torch.isfinite(norm):
            raise ValueError('Nonfinite gradient norm')
        maximum_gradient = max(maximum_gradient, float(norm))
        optimizer.step()
        if any(not torch.isfinite(p).all() for p in adapter.parameters()):
            raise ValueError('Nonfinite gate parameters')
        totals += sums.detach().double().cpu().numpy()
        cells += counts.cpu().numpy()
        objective_sum += float(objective.detach())
        penalty_sum += float(penalty.detach())
        if step % settings['progress_every_batches'] == 0:
            progress('training_progress', epoch=epoch, batches_completed=step + 1,
                     samples_total=len(indices), regional_loss_standardized=float(loss.detach()))
    changed = any(not torch.equal(before[n], p) for n, p in adapter.named_parameters())
    if epoch == 1 and (not changed or maximum_gradient == 0):
        raise ValueError('Regional gate had no effective learning update')
    return {'mae_standardized': float(totals.sum() / cells.sum()),
            'online_region_mae_standardized': dict(zip(REGIONS, (totals / cells).tolist())),
            'mean_minibatch_objective': objective_sum / (step + 1),
            'mean_minibatch_identity_penalty': penalty_sum / (step + 1),
            'maximum_gradient_norm': maximum_gradient, 'gate_parameters_changed': changed,
            'optimizer_steps': step + 1}


def probe_indices(indices, count):
    positions = np.linspace(0, len(indices) - 1, min(len(indices), count), dtype=int)
    return [indices[i] for i in positions]


def parameter_probe(model, adapter, dataset, indices, settings, device, destination, progress):
    """Pooled fixed-subset gradients in shared MLP parameter coordinates, no updates."""
    before = state_hash(adapter)
    parameters = list(adapter.named_parameters())
    values = [p for _, p in parameters]
    gradients = np.zeros((4, sum(p.numel() for p in values)), dtype=np.float64)
    sums_total, counts_total = np.zeros(4), np.zeros(4)
    ids = []
    for step, raw in enumerate(base.loader(dataset, indices, settings['batch_size'])):
        batch = base.device_batch(raw, device)
        prediction = model(batch['x'], incident_data=batch['incident'])
        target = (batch['y_flow'] - dataset.scaler['mean']) / dataset.scaler['std']
        sums, counts = region_sums(prediction, target, batch['y_valid'], batch['candidate_mask'])
        strength = model.icsf_module.last_gate.squeeze(-1)[batch['candidate_mask']]
        losses = list(sums.unbind()) + [(strength - 1).square().sum()]
        # Free the full backbone graph on the last regional derivative, rather
        # than ending with a penalty derivative that visits only the gate.
        for i in (3, 0, 1, 2):
            grads = torch.autograd.grad(losses[i], values, retain_graph=i != 2)
            vector = torch.cat([g.reshape(-1) for g in grads]).detach().double().cpu().numpy()
            if not np.isfinite(vector).all():
                raise ValueError('Nonfinite parameter probe gradient')
            gradients[i] += vector
        sums_total += [float(x.detach()) for x in losses]
        counts_total += counts.tolist() + [strength.numel()]
        ids.extend(raw['positive_sample_index'].tolist())
        model.icsf_module.last_gate = None
        progress('parameter_probe_progress', batches=step + 1, samples=len(indices))
    means = gradients / counts_total[:, None]
    vectors = dict(zip(REGIONS + ('identity_penalty_mean',), means))
    fractions = counts_total[:3] / counts_total[:3].sum()
    vectors['global_data'] = fractions @ means[:3]
    vectors['regional_data'] = means[:3].mean(0)
    for arm in ARMS:
        vectors[arm + '_total'] = vectors[arm + '_data'] + settings['penalty'] * means[3]
    norms = {k: float(np.linalg.norm(v)) for k, v in vectors.items()}
    def comparison(a, b):
        dot = float(vectors[a] @ vectors[b])
        return {'dot': dot, 'cosine': dot / (norms[a] * norms[b]) if norms[a] * norms[b] else None}
    result = {
        'samples': len(ids), 'sample_ids': ids, 'gate_state_sha256': before,
        'mean_losses': dict(zip(REGIONS + ('identity_penalty_mean',), (sums_total / counts_total).tolist())),
        'valid_cell_fractions': dict(zip(REGIONS, fractions.tolist())),
        'parameter_names': [n for n, _ in parameters], 'parameter_shapes': [list(p.shape) for p in values],
        'gradient_l2': norms,
        'region_pairs': {a + '__' + b: comparison(a, b) for i, a in enumerate(REGIONS) for b in REGIONS[i + 1:]},
        'region_objective_alignment': {r: {key: comparison(r, key) for key in
            ('global_data', 'regional_data', 'global_total', 'regional_total')} for r in REGIONS},
        'interpretation': 'Positive alignment means infinitesimal descent of this objective lowers the region loss on this fixed fit subset; not an Adam update or a generalization guarantee.'}
    if state_hash(adapter) != before:
        raise ValueError('Gradient probe changed gate weights')
    base.save_arrays(destination, {'gradients': np.stack(list(vectors.values())),
                                  'gradient_names': list(vectors), 'sample_ids': ids,
                                  'loss_sums': sums_total, 'counts': counts_total})
    return result


def compare(reference, arms, times, protocol):
    records = {}
    for cohort, a in reference.items():
        order = [a] + [arms[arm][cohort] for arm in ARMS]
        for other in order[1:]:
            for key in ('ids', 'source_indices', 'counts', 'prediction_counts', 'candidate_mask'):
                if not np.array_equal(a[key], other[key]):
                    raise ValueError('Objective-arm sample/support mismatch')
        records[cohort] = {**{k: a[k] for k in ('ids', 'counts', 'prediction_counts', 'regions')},
                           'errors': np.stack([v['errors'] for v in order], axis=1)}
    return base.analyze(records, times, {'paths': ['A', *ARMS], 'bootstrap': protocol['bootstrap'],
        'comparisons': {'global_vs_A': ['A', 'global'], 'regional_vs_A': ['A', 'regional'],
                        'regional_vs_global': ['global', 'regional']}})


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
            raise ValueError('Recovery requires a completed/partial full run and a separate destination')
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
        base.write_json(partial / 'effective_indices.json', {p: v['indices'] for p, v in plan.items()})
        probe = probe_indices(plan['fit']['indices']['incident_full'], frozen['parameter_gradient_probe']['samples'])
        base.write_json(partial / 'probe_indices.json', probe)
        source_files = [Path(__file__), PROTOCOL, Path(base.__file__), base.PROTOCOL,
            Path(base.mechanisms.__file__), base.mechanisms.PROTOCOL,
            REPO / 'src/models/incident_strength_gate.py', REPO / 'experiments/chronological/gate_recovery.py',
            REPO / 'experiments/chronological/audit_architecture_regions.py',
            REPO / 'experiments/chronological/materialize_incident_branch.py',
            REPO / 'experiments/chronological/smoke.py', REPO / 'experiments/chronological/train.py',
            REPO / 'src/utils/chronological.py', Path(__file__).with_name('run_regional_gate_objective.sh')]
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
                r = base.evaluate(model, datasets[cohort], plan[phase]['indices'][cohort],
                    effective['training']['evaluation_batch_size'], device,
                    lambda stage, **fields: progress(stage, arm='A', phase=phase, cohort=cohort, **fields))
                reference[phase][cohort] = r
                base.save_arrays(partial / f'{phase}_A_{cohort}.npz', r)
        del model
        selection = {c: base.metric_summary(r) for c, r in reference['selection'].items()}
        audit_ids = set(reference['audit']['incident_full']['ids'].tolist())
        times = {int(row['sample_index']): row['t0'] for row in positive if int(row['sample_index']) in audit_ids}
        runs, comparisons = {}, {}
        for seed in effective['seeds']:
            runs[str(seed)], learned = {}, {}
            initial_fingerprints = []
            for arm in ARMS:
                directory = partial / f'{arm}_s{seed}'
                directory.mkdir()
                arm_protocol = copy.deepcopy(effective)
                arm_protocol['training']['loss_arm'] = arm
                arm_protocol['training']['objective'] = frozen['objectives'][arm]
                model = native()
                def update(stage, **fields):
                    progress(stage, arm=arm, seed=seed, **fields)
                def diagnostic(current, adapter, label):
                    return parameter_probe(current, adapter, datasets['incident_full'], probe,
                        {'batch_size': frozen['parameter_gradient_probe']['batch_size'],
                         'penalty': effective['training']['gate_identity_penalty']}, device,
                        directory / f'parameter_gradients_{label}.npz',
                        lambda stage, **fields: update(stage, snapshot=label, **fields))
                detail = base.fit_variant(model, 'node', seed, datasets, plan, selection, arm_protocol, device,
                    directory, update, resume_from / directory.name if resume_from else None,
                    epoch_trainer=base.train_epoch if arm == 'global' else regional_train_epoch,
                    protocol_hash=PROTOCOL_SHA256, diagnostic=diagnostic)
                initial_fingerprints.append(detail['parameter_gradient_diagnostics']['initial']['gate_state_sha256'])
                if len(set(initial_fingerprints)) != 1:
                    raise ValueError('Paired arms did not share identical initialization')
                learned[arm], detail['audit'] = {}, {}
                for cohort in base.COHORTS:
                    r = base.evaluate(model, datasets[cohort], plan['audit']['indices'][cohort],
                        effective['training']['evaluation_batch_size'], device,
                        lambda stage, **fields: update(stage, phase='audit', cohort=cohort, **fields))
                    learned[arm][cohort] = r
                    detail['audit'][cohort] = base.metric_summary(r)
                    base.save_arrays(directory / f'audit_{cohort}.npz', r)
                runs[str(seed)][arm] = detail
                base.write_json(directory / 'summary.json', detail)
                model.icsf_module.last_gate = None
                del model
                gc.collect()
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
            if not check:
                analysis, _, weekly = compare(reference['audit'], learned, times, inherited)
                comparisons[str(seed)] = analysis
                base.save_arrays(partial / f'audit_weekly_s{seed}.npz', weekly)
            progress('seed_complete', seed=seed)
        summary = {'status': 'ENGINEERING_CHECK_PASS' if check else 'REGIONAL_GATE_OBJECTIVE_COMPARISON_COMPLETE',
            'engineering_check': check, 'protocol_sha256': PROTOCOL_SHA256, 'frozen_protocol': frozen,
            'inherited_protocol': inherited, 'effective_training': effective['training'],
            **frozen['information_boundary'], 'model_training_performed': True,
            'main_training_ready': False,
            'recommendation': 'ENGINEERING_ONLY' if check else frozen['decision'],
            'training_scope': 'node_gate_only', 'runs': runs, 'audit_comparisons': comparisons,
            'baseline_audit': {c: base.metric_summary(r) for c, r in reference['audit'].items()},
            'phase_samples': {p: {c: len(v) for c, v in x['indices'].items()} for p, x in plan.items()},
            'paired_initialization_exact': True, 'inputs': hashes, 'code_sha256': code_hashes,
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
    except BaseException as exc:
        if partial.exists():
            base.write_json(partial / 'failure.json', {'error': str(exc), 'traceback': traceback.format_exc()})
        raise
    report(summary)
    print('Saved v12e comparison:', output / 'summary.json', flush=True)
    return summary


def report(summary):
    # Existing reporter's labels are taken from keys, not restricted to scalar/node.
    base.report(summary)
    for seed, arms in summary['runs'].items():
        for arm, detail in arms.items():
            print(f'\nPARAMETER PROBE seed={seed} arm={arm}; fixed fit subset only')
            for label, result in detail['parameter_gradient_diagnostics'].items():
                print(label, 'losses=', result['mean_losses'], 'norms=', result['gradient_l2'])
                print('region_pair_cosines=', {k: v['cosine'] for k, v in result['region_pairs'].items()})
                print('region_alignment_with_current_objective=', {r: v[arm + '_total']['cosine']
                    for r, v in result['region_objective_alignment'].items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    r = sub.add_parser('run')
    for name in ('data-dir', 'primary-control-dir', 'secondary-control-dir', 'checkpoint', 'output'):
        r.add_argument('--' + name, type=Path, required=True)
    r.add_argument('--device', default='cuda:0')
    r.add_argument('--check', action='store_true')
    r.add_argument('--resume-from', type=Path)
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
