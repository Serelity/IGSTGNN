"""v12d: fixed ICSF injection curves and fit-only input-coordinate derivatives."""

import argparse
import copy
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import sys
import time
import traceback

import numpy as np
import torch
from torch import nn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.chronological import train_incident_strength_gate as gate

PROTOCOL = Path(__file__).with_name('icsf_strength_response_v12d.json')
PROTOCOL_SHA256 = '2d78447df846b6fd4e7b3cf631362611b6309f3120615ee913ad39d7e598070f'
REGIONS = ('all', 'candidate_h1_h6', 'candidate_h7_h12', 'noncandidate_all')
PARTS = REGIONS[1:]


def load_protocol():
    if gate.sha256(PROTOCOL) != PROTOCOL_SHA256:
        raise ValueError('Frozen v12d protocol changed')
    return json.loads(PROTOCOL.read_text())


class DirectStrength(nn.Module):
    """External intervention, including exact endpoints; no trainable parameters."""
    def __init__(self):
        super().__init__()
        self.value = 1.
        self.coordinates = None

    def forward(self, history, incident):
        shape = (history.shape[0], history.shape[2], 1)
        if self.coordinates is not None:
            if tuple(self.coordinates.shape) != shape:
                raise ValueError('Strength coordinate shape differs from current batch')
            return self.coordinates
        return history.new_full(shape, self.value)


def attach(model):
    gate.attach_gate(model, 'scalar')
    control = DirectStrength()
    model.icsf_module.gate = control
    return control


def evaluate(model, control, dataset, indices, strength, batch_size, device, progress):
    control.value, control.coordinates = float(strength), None
    parts = {k: [] for k in ('ids', 'errors', 'counts')}
    columns = [gate.REGIONS.index(r) for r in REGIONS]
    with torch.no_grad():
        for step, raw in enumerate(gate.loader(dataset, indices, batch_size)):
            batch = gate.device_batch(raw, device)
            pred = model(batch['x'], incident_data=batch['incident'])
            if strength == 1.:
                adapter = model.icsf_module
                model.icsf_module = adapter.base
                try:
                    native = model(batch['x'], incident_data=batch['incident'])
                finally:
                    model.icsf_module = adapter
                if not torch.equal(pred, native):
                    raise ValueError('g=1 does not exactly reproduce native full predictions')
            record = gate.statistics(pred * dataset.scaler['std'] + dataset.scaler['mean'], batch)
            parts['ids'].append(raw['positive_sample_index'].numpy())
            for key in ('errors', 'counts'):
                parts[key].append(record[key][:, columns])
            if step % 10 == 0:
                progress('scan_progress', strength=strength, batches=step + 1, samples=len(indices))
    return {**{k: np.concatenate(v) for k, v in parts.items()}, 'regions': np.asarray(REGIONS)}


def gradient_batch(model, control, batch, scaler):
    """Differentiate regional standardized error sums, retaining shared coordinates."""
    if not np.isfinite(scaler['std']) or scaler['std'] <= 0:
        raise ValueError('Invalid frozen flow scale')
    leaf = batch['x'].new_ones((len(batch['x']), batch['x'].shape[2], 1), requires_grad=True)
    control.coordinates = leaf
    prediction = model(batch['x'], incident_data=batch['incident'])
    masks = [gate.masks_for(prediction, batch['candidate_mask'])[gate.REGIONS.index(r)]
             & batch['y_valid'] for r in REGIONS]
    # Match the raw-response evaluation's error, then scale by the frozen flow std.
    raw_prediction = prediction * scaler['std'] + scaler['mean']
    if not torch.isfinite(raw_prediction).all() or not torch.isfinite(batch['y_flow'][batch['y_valid']]).all():
        raise ValueError('Nonfinite prediction/valid target in gradient diagnostic')
    error = (raw_prediction.double() - batch['y_flow'].double()).abs() / scaler['std']
    sums = [torch.where(mask, error, 0.).sum() for mask in masks]
    derivatives = [torch.autograd.grad(value, leaf, retain_graph=i < len(sums) - 1)[0]
                   for i, value in enumerate(sums)]
    gradients = torch.stack(derivatives, -1).squeeze(2).detach().double()
    if not torch.isfinite(gradients).all():
        raise ValueError('Nonfinite strength derivatives')
    if not torch.allclose(gradients[..., 0], gradients[..., 1:].sum(-1), atol=1e-5, rtol=1e-4):
        raise ValueError('Regional derivatives do not sum to full-loss derivative')
    connected = batch['incident']['distances'].abs().sum(-1) > 0
    if torch.count_nonzero(gradients[~connected]):
        raise ValueError('Unconnected injection coordinate has a nonzero derivative')
    counts = torch.stack([m.sum((1, 2, 3)) for m in masks], -1)
    if not torch.equal(counts[:, 0], counts[:, 1:].sum(-1)):
        raise ValueError('Regional masks do not partition valid cells')
    control.coordinates = None
    model.icsf_module.last_gate = None
    return gradients.cpu().numpy(), counts.cpu().numpy(), connected.cpu().numpy()


def summarize_gradients(record):
    counts = record['counts'].sum(0).astype(np.float64)
    if (counts <= 0).any():
        raise ValueError('Empty full-fit gradient region')
    # Same denominator is essential: these vectors add to the global-loss gradient.
    vectors = record['error_sum_gradients'].reshape(-1, len(REGIONS)) / counts[0]
    norms = np.linalg.norm(vectors, axis=0)
    slopes = vectors.sum(0)
    paired = {}
    for i in range(1, len(REGIONS)):
        for j in range(i + 1, len(REGIONS)):
            dot = float(np.dot(vectors[:, i], vectors[:, j]))
            paired[f'{REGIONS[i]}__{REGIONS[j]}'] = {
                'dot': dot, 'cosine': dot / float(norms[i] * norms[j]) if norms[i] * norms[j] else None}
    return {
        'units': 'standardized MAE per direct unit of ICSF injection strength',
        'coordinate_system': 'one independent strength per fit sample and station; no shared MLP parameters',
        'regions': {r: {
            'valid_cells': int(counts[i]), 'valid_cell_fraction': float(counts[i] / counts[0]),
            'shared_scalar_global_contribution_derivative': float(slopes[i]),
            'shared_scalar_own_region_mean_derivative': float(slopes[i] * counts[0] / counts[i]),
            'input_coordinate_global_contribution_l2': float(norms[i])}
            for i, r in enumerate(REGIONS)},
        'v12c_scalar_logit_global_derivative_at_identity': float(.5 * slopes[0]),
        'region_pairs': paired,
        'input_coordinate_cancellation_ratio': float(np.linalg.norm(vectors[:, 1:].sum(1)) / norms[1:].sum())
            if norms[1:].sum() else None,
        'shared_scalar_region_cancellation_ratio': float(abs(slopes[1:].sum()) / np.abs(slopes[1:]).sum())
            if np.abs(slopes[1:]).sum() else None,
        'maximum_region_sum_derivative_discrepancy': float(np.abs(vectors[:, 0] - vectors[:, 1:].sum(1)).max()),
        'identity_penalty_derivative_at_g1': 0.,
    }


def report(s):
    print('status:', s['status'])
    print('Fixed-point exploratory diagnostic; no trained/selected winner or independent confirmation.')
    for phase, cohorts in s['response'].items():
        for cohort, points in cohorts.items():
            print(f'\n[{phase}/{cohort}] raw MAE gains vs g=1; positive is better')
            for p in points:
                print('g=', p['strength'], json.dumps(p['gain_vs_identity']))
    print('\nFIT-ONLY GRADIENTS AT g=1; positive slope favors decreasing strength locally')
    print(json.dumps(s['fit_gradients'], indent=2))
    print('\nFIT SCALAR SECANT (0.99 to 1.01) vs local derivative; nonsmoothness/rounding may differ')
    print(json.dumps(s['fit_scalar_slope_comparison'], indent=2))


def run(data_dir, primary_dir, secondary_dir, checkpoint, output, device='cuda:0', check=False):
    protocol, old = load_protocol(), gate.load_protocol()
    if gate.sha256(gate.PROTOCOL) != protocol['v12c_protocol_sha256']:
        raise ValueError('Inherited v12c identity changed')
    output = Path(output)
    partial = output.with_name(output.name + '.partial')
    if output.exists() or partial.exists():
        raise FileExistsError('Preserve existing output/partial; use a new run name')
    partial.mkdir(parents=True)
    started = time.monotonic()
    def progress(stage, **fields):
        value = {'stage': stage, 'elapsed_seconds': time.monotonic() - started,
                 'memory': gate.memory_snapshot(device), **fields}
        gate.write_json(partial / 'progress.json', value)
        print(json.dumps(value), flush=True)
    try:
        device = torch.device(device)
        torch.set_num_threads(protocol['threads'])
        gate.configure_determinism(device)
        identity = gate.mechanisms.load_protocol(gate.mechanisms.PROTOCOL)
        if gate.sha256(gate.mechanisms.PROTOCOL) != old['v12a_protocol_sha256']:
            raise ValueError('Inherited v12a identity changed')
        baseline, hashes = gate.mechanisms.verify_inputs(data_dir, primary_dir, secondary_dir, checkpoint, identity)
        schedule = copy.deepcopy(old)
        schedule['periods'] = {p: old['periods'][p] for p in protocol['phases']}
        plan = gate.make_plan(gate.read_csv(Path(data_dir) / 'train_manifest.csv'),
                             gate.read_csv(Path(primary_dir) / 'train_control_manifest.csv'),
                             gate.read_csv(Path(secondary_dir) / 'train_second_control_manifest.csv'), schedule)
        gate.write_json(partial / 'eligibility.json', plan)
        effective = copy.deepcopy(plan)
        if check:
            for phase in effective.values():
                phase['indices'] = {c: v[:protocol['check_samples_per_cohort_phase']]
                                    for c, v in phase['indices'].items()}
            # Full eligibility stays in eligibility.json; only effective indices are subsampled.
        gate.write_json(partial / 'effective_indices.json', {p: v['indices'] for p, v in effective.items()})
        datasets = gate.make_datasets(data_dir, primary_dir, secondary_dir, baseline)
        model = gate.make_model(Path(data_dir), len(datasets['incident_full'].station_ids), device, 'fixed')
        model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
        if sum(p.numel() for p in model.parameters()) != baseline['checkpoint']['parameters']:
            raise ValueError('Backbone parameter count changed')
        model.eval().requires_grad_(False)
        original_hash = gate.backbone_hash(model)
        control = attach(model)
        responses = {}
        rows = []
        # Evaluate identity first to compare all points against exactly reproduced A.
        strengths = [1.] + [g for g in protocol['strengths'] if g != 1.]
        for phase, item in effective.items():
            responses[phase] = {}
            for cohort, indices in item['indices'].items():
                points = {}
                for strength in strengths:
                    record = evaluate(model, control, datasets[cohort], indices, strength,
                        protocol['evaluation_batch_size'], device,
                        lambda stage, **fields: progress(stage, phase=phase, cohort=cohort, **fields))
                    counts = record['counts'].sum(0)
                    if (counts <= 0).any():
                        raise ValueError('Empty response region')
                    mae = record['errors'].sum(0) / counts
                    if strength == 1.:
                        reference, reference_record = mae, record
                    elif not (np.array_equal(record['ids'], reference_record['ids']) and
                              np.array_equal(record['counts'], reference_record['counts'])):
                        raise ValueError('Strength comparison changed samples or valid support')
                    gains = reference - mae
                    point = {'strength': strength, 'samples': len(indices),
                             'mae': dict(zip(REGIONS, mae.tolist())),
                             'gain_vs_identity': dict(zip(REGIONS, gains.tolist()))}
                    points[strength] = point
                    rows.append({'phase': phase, 'cohort': cohort, 'strength': strength,
                                 **{f'{r}_mae': float(mae[i]) for i, r in enumerate(REGIONS)},
                                 **{f'{r}_gain': float(gains[i]) for i, r in enumerate(REGIONS)}})
                    gate.save_arrays(partial / f'{phase}_{cohort}_g{strength:g}.npz', record)
                    gate.assert_backbone(model, original_hash)
                responses[phase][cohort] = [points[g] for g in protocol['strengths']]
        ds = datasets['incident_full']
        chunks = {k: [] for k in ('ids', 'error_sum_gradients', 'counts', 'connected_mask')}
        for step, raw in enumerate(gate.loader(ds, effective['fit']['indices']['incident_full'], protocol['gradient_batch_size'])):
            grad, counts, connected = gradient_batch(model, control, gate.device_batch(raw, device), ds.scaler)
            chunks['ids'].append(raw['positive_sample_index'].numpy())
            chunks['error_sum_gradients'].append(grad)
            chunks['counts'].append(counts)
            chunks['connected_mask'].append(connected)
            if step % 10 == 0:
                progress('fit_gradient_progress', batches=step + 1)
        record = {k: np.concatenate(v) for k, v in chunks.items()}
        record.update(regions=np.asarray(REGIONS), station_ids=np.asarray(ds.station_ids))
        gradient_summary = summarize_gradients(record)
        fit_points = {p['strength']: p for p in responses['fit']['incident_full']}
        slope_comparison = {r: {
            'secant_raw_mae_per_strength': (fit_points[1.01]['mae'][r] - fit_points[.99]['mae'][r]) / .02,
            'local_raw_mae_derivative_at_one': gradient_summary['regions'][r][
                'shared_scalar_own_region_mean_derivative'] * ds.scaler['std']}
            for r in REGIONS}
        gate.save_arrays(partial / 'fit_strength_gradients.npz', record)
        gate.assert_backbone(model, original_hash)
        with (partial / 'response.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        source_files = [Path(__file__), PROTOCOL, Path(gate.__file__), gate.PROTOCOL,
                        Path(gate.mechanisms.__file__), REPO / 'src/models/incident_strength_gate.py',
                        REPO / 'experiments/chronological/materialize_incident_branch.py',
                        REPO / 'experiments/chronological/smoke.py', REPO / 'src/utils/chronological.py',
                        REPO / 'experiments/chronological/train.py',
                        REPO / 'experiments/chronological/gate_recovery.py',
                        REPO / 'experiments/chronological/audit_architecture_regions.py',
                        Path(__file__).with_name('run_icsf_strength_response.sh')]
        summary = {'status': 'ENGINEERING_CHECK_PASS' if check else 'ICSF_STRENGTH_RESPONSE_DIAGNOSTIC_COMPLETE',
                   'engineering_check': check, 'protocol_sha256': PROTOCOL_SHA256, 'frozen_protocol': protocol,
                   **protocol['information_boundary'], 'gradient_computation_performed': True,
                   'native_identity_exact_all_evaluated_batches': True, 'backbone_state_unchanged': True,
                   'phase_samples': {p: {c: len(v) for c, v in x['indices'].items()} for p, x in effective.items()},
                   'response': responses, 'fit_gradients': gradient_summary,
                   'fit_scalar_slope_comparison': slope_comparison, 'inputs': hashes,
                   'code_sha256': {str(p.relative_to(REPO)): gate.sha256(p) for p in source_files},
                   'environment': {'host': socket.gethostname(), 'pid': os.getpid(), 'device': str(device),
                                   'python': sys.version, 'torch': torch.__version__, 'numpy': np.__version__,
                                   'threads': torch.get_num_threads(),
                                   'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
                                   'tf32': torch.backends.cuda.matmul.allow_tf32,
                                   'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
                                   'git_head': gate.subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
                                   'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                                   'finished_utc': datetime.now(timezone.utc).isoformat()},
                   'outputs': {p.name: gate.sha256(p) for p in partial.iterdir() if p.is_file()}}
        gate.write_json(partial / 'summary.json', summary)
        if output.exists():
            raise FileExistsError('Final output appeared; preserve partial')
        partial.rename(output)
        report(summary)
        print('Saved v12d diagnostic:', output / 'summary.json')
        return summary
    except BaseException as exc:
        if partial.exists():
            gate.write_json(partial / 'failure.json', {'error': str(exc), 'traceback': traceback.format_exc()})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    r = sub.add_parser('run')
    for name in ('data-dir', 'primary-control-dir', 'secondary-control-dir', 'checkpoint', 'output'):
        r.add_argument('--' + name, type=Path, required=True)
    r.add_argument('--device', default='cuda:0')
    r.add_argument('--check', action='store_true')
    sub.add_parser('report').add_argument('summary', type=Path)
    args = parser.parse_args()
    if args.action == 'report':
        report(json.loads(args.summary.read_text()))
    else:
        run(args.data_dir, args.primary_control_dir, args.secondary_control_dir, args.checkpoint,
            args.output, args.device, args.check)


if __name__ == '__main__':
    main()
