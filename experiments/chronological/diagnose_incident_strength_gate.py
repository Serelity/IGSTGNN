"""Read-only v12c trajectory diagnostics from saved selection statistics.

No model loading, optimizer, audit-period arrays, validation or test inputs.
"""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np


PROTOCOL = Path(__file__).with_name('incident_strength_gate_v12c.json')
PROTOCOL_SHA256 = '99a9ee432d858f9568e32132e4e2e5388524184efea5da4d16c99abd7b875def'
COHORTS = ('incident_full', 'incident', 'primary_control', 'secondary_control')
PARTS = ('candidate_h1_h6', 'candidate_h7_h12', 'noncandidate_all')
REGIONS = ('all',) + PARTS


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def finite(value):
    value = float(value)
    if not np.isfinite(value):
        raise ValueError('Nonfinite diagnostic input')
    return value


class Source:
    def __init__(self, root):
        self.root = Path(root).resolve()
        payload = (self.root / 'summary.json').read_bytes()
        self.hashes = {'summary.json': digest(payload)}
        self.summary = json.loads(payload)
        frozen = PROTOCOL.read_bytes()
        if digest(frozen) != PROTOCOL_SHA256:
            raise ValueError('Local frozen protocol changed')
        self.protocol = json.loads(frozen)
        if (self.summary.get('status') != 'ICSF_STRENGTH_GATE_EXPERIMENT_COMPLETE'
                or self.summary.get('engineering_check') is not False
                or self.summary.get('protocol_sha256') != PROTOCOL_SHA256
                or self.summary.get('frozen_protocol') != self.protocol
                or self.summary.get('effective_training') != self.protocol['training']):
            raise ValueError('Require complete full-budget frozen v12c result')

    def read(self, name):
        payload = (self.root / name).read_bytes()
        sha = digest(payload)
        if self.summary['outputs'].get(name) != sha:
            raise ValueError(f'Source hash mismatch: {name}')
        self.hashes[name] = sha
        return payload


def baselines(source):
    result = {}
    for cohort in COHORTS:
        with np.load(io.BytesIO(source.read(f'selection_A_{cohort}.npz')),
                     allow_pickle=False) as arrays:
            regions = arrays['regions'].tolist()
            errors, counts = arrays['errors'], arrays['counts']
            if (len(set(regions)) != len(regions) or errors.shape != counts.shape
                    or errors.ndim != 2 or errors.shape[1] != len(regions)
                    or errors.shape[0] != source.summary['phase_samples']['selection'][cohort]
                    or not np.isfinite(errors).all() or not np.isfinite(counts).all()
                    or (errors < 0).any() or (counts < 0).any()):
                raise ValueError(f'Invalid selection sufficient statistics: {cohort}')
            totals, cells = errors.sum(0), counts.sum(0)
            index = {region: regions.index(region) for region in REGIONS}
            if any(cells[i] <= 0 for i in index.values()):
                raise ValueError('Empty protected selection region')
            if not np.array_equal(counts[:, index['all']],
                                  sum(counts[:, index[r]] for r in PARTS)):
                raise ValueError('Selection regions do not partition all valid cells')
            if not np.allclose(errors[:, index['all']],
                               sum(errors[:, index[r]] for r in PARTS), rtol=1e-12, atol=1e-8):
                raise ValueError('Selection error sums do not partition all cells')
            result[cohort] = {
                'mae': {r: float(totals[i] / cells[i]) for r, i in index.items()},
                'valid_cell_fraction': {r: float(cells[i] / cells[index['all']])
                                        for r, i in index.items()}}
    return result


def analyze_history(history, detail, baseline, protocol):
    if [h['epoch'] for h in history] != list(range(1, protocol['training']['epochs'] + 1)):
        raise ValueError('Missing, duplicate or unordered epochs')
    rows, gradients, losses, changes = [], [], [], []
    improving, eligible_improving, rejected_improving = [], [], []
    best_epoch, best_mae = 0, baseline['incident_full']['mae']['all']
    best_metrics = baseline
    total_steps = 0
    for h in history:
        epoch, current = h['epoch'], h['selection']
        checks = {}
        for cohort in COHORTS:
            metrics = current[cohort]
            for region in REGIONS:
                a, b = baseline[cohort]['mae'][region], finite(metrics['mae'][region])
                checks[f'{cohort}/{region}'] = b <= a * (1 + protocol['selection']['maximum_relative_harm']) + 1e-12
            gate = metrics['candidate_gate']
            mean, std = finite(gate['mean']), finite(gate['std'])
            q05, q50, q95 = map(finite, gate['q05_q50_q95'])
            if not (0 <= mean <= 2 and std >= 0 and 0 <= q05 <= q50 <= q95 <= 2):
                raise ValueError('Invalid saved gate distribution')
            row = {'epoch': epoch, 'cohort': cohort,
                   'selection_global_gain_raw_mae': baseline[cohort]['mae']['all'] - metrics['mae']['all'],
                   'gate_mean': mean, 'gate_std': std, 'gate_q05': q05, 'gate_q50': q50,
                   'gate_q95': q95, 'selection_gate_rms_distance_from_one': float(np.hypot(std, mean - 1))}
            for region in PARTS:
                gain = baseline[cohort]['mae'][region] - metrics['mae'][region]
                row[f'{region}_gain_raw_mae'] = gain
                row[f'{region}_global_contribution'] = gain * baseline[cohort]['valid_cell_fraction'][region]
            if not np.isclose(sum(row[f'{r}_global_contribution'] for r in PARTS),
                              row['selection_global_gain_raw_mae'], rtol=1e-7, atol=1e-9):
                raise ValueError('Epoch regional contributions do not sum to global gain')
            rows.append(row)
        eligible = all(checks.values())
        if h['eligible'] != eligible or h['protection_checks'] != checks:
            raise ValueError('Saved protection decision disagrees with frozen rules')
        score = finite(current['incident_full']['mae']['all'])
        if score < baseline['incident_full']['mae']['all']:
            improving.append(epoch)
            (eligible_improving if eligible else rejected_improving).append(epoch)
        if eligible and score < best_mae:
            best_epoch, best_mae, best_metrics = epoch, score, current
        if h['best_epoch'] != best_epoch:
            raise ValueError('Saved best epoch disagrees with frozen rules')
        training = h['training']
        loss, gradient = finite(training['mae_standardized']), finite(training['maximum_gradient_norm'])
        if loss < 0 or gradient < 0 or training['optimizer_steps'] <= 0:
            raise ValueError('Invalid training diagnostics')
        gradients.append(gradient)
        losses.append(loss)
        changes.append(bool(training['gate_parameters_changed']))
        total_steps += training['optimizer_steps']
        for row in rows[-len(COHORTS):]:
            row.update(eligible=eligible, selected_so_far=best_epoch,
                       online_fit_mae_standardized=loss, maximum_preclip_gradient_norm=gradient,
                       parameters_changed=changes[-1])
    if (detail['selected_epoch'] != best_epoch or detail['optimizer_steps'] != total_steps
            or not detail['initial_prediction_exactly_A'] or not detail['backbone_state_unchanged']):
        raise ValueError('Final selection/training identity disagrees with history')
    for cohort in COHORTS:
        for region in REGIONS:
            if detail['selection_metrics'][cohort]['mae'][region] != best_metrics[cohort]['mae'][region]:
                raise ValueError('Selected metrics disagree with replayed selection')
    full = [r for r in rows if r['cohort'] == 'incident_full']
    reason = ('TRAINED_EPOCH_SELECTED' if best_epoch else
              'NO_EPOCH_IMPROVED_SELECTION_GLOBAL_MAE' if not improving else
              'IMPROVING_EPOCHS_FAILED_PROTECTION')
    result = {
        'selected_epoch': best_epoch, 'selection_reason': reason,
        'epochs': len(history), 'eligible_epochs': sum(h['eligible'] for h in history),
        'epochs_improving_over_A': improving, 'eligible_improving_epochs': eligible_improving,
        'improving_but_rejected_epochs': rejected_improving,
        'optimizer_steps': total_steps, 'parameter_change_epochs': sum(changes),
        'zero_maximum_gradient_epochs': [h['epoch'] for h, g in zip(history, gradients) if g == 0],
        'epoch_maximum_preclip_gradient_norm_range': [min(gradients), max(gradients)],
        'epochs_with_any_gradient_clipping': sum(g > protocol['training']['clip_grad_norm'] for g in gradients),
        'online_fit_mae_standardized_first_last': [losses[0], losses[-1]],
        'online_fit_mae_standardized_first_minus_last': losses[0] - losses[-1],
        'selection_global_gain_range_raw_mae': [min(r['selection_global_gain_raw_mae'] for r in full),
                                               max(r['selection_global_gain_raw_mae'] for r in full)],
        'selected_selection_global_gain_raw_mae': baseline['incident_full']['mae']['all'] - best_mae,
        'selection_gate_mean_range': [min(r['gate_mean'] for r in full), max(r['gate_mean'] for r in full)],
        'selection_gate_rms_distance_from_one_range': [min(r['selection_gate_rms_distance_from_one'] for r in full),
                                                      max(r['selection_gate_rms_distance_from_one'] for r in full)],
        'first_epoch_selection': full[0], 'last_epoch_selection': full[-1],
        'selected_epoch_selection': next((r for r in full if r['epoch'] == best_epoch), None)}
    return result, rows


def report(summary):
    print('status:', summary['status'])
    print('Saved selection history only; no retraining, new checkpoint selection or audit-array reads.')
    print('selection A:', summary['baseline_selection']['incident_full']['mae']['all'])
    for name, result in summary['runs'].items():
        print(f'\n[{name}] selected_epoch={result["selected_epoch"]} reason={result["selection_reason"]}')
        for key in ('eligible_epochs', 'epochs_improving_over_A', 'parameter_change_epochs',
                    'epoch_maximum_preclip_gradient_norm_range', 'online_fit_mae_standardized_first_last',
                    'selection_global_gain_range_raw_mae', 'selection_gate_mean_range',
                    'selection_gate_rms_distance_from_one_range'):
            print(f'{key}: {result[key]}')
        for label in ('last_epoch_selection', 'selected_epoch_selection'):
            row = result[label]
            if row is not None:
                print(label, 'regional_global_contributions:',
                      {r: row[f'{r}_global_contribution'] for r in PARTS})
    print('\nLimits: online fit losses are not fixed-checkpoint fit evaluations; gradient maxima '
          'do not identify cancellation, penalty dominance or causality. Reused development data.')


def diagnose(source_dir, output):
    source = Source(source_dir)
    output = Path(output).resolve()
    partial = output.with_name(output.name + '.partial')
    if output == source.root or source.root in output.parents:
        raise ValueError('Write diagnostics outside the source experiment')
    if output.exists() or partial.exists():
        raise FileExistsError('Preserve existing output; use a new diagnostic directory')
    baseline = baselines(source)
    runs, rows = {}, []
    for seed in source.protocol['seeds']:
        for variant in source.protocol['variants']:
            name = f'{variant}_s{seed}'
            history = json.loads(source.read(f'{name}/history.json'))
            result, epoch_rows = analyze_history(history, source.summary['runs'][str(seed)][variant],
                                                baseline, source.protocol)
            expected_steps = (source.summary['phase_samples']['fit']['incident_full']
                              + source.protocol['training']['batch_size'] - 1) // source.protocol['training']['batch_size']
            if any(h['training']['optimizer_steps'] != expected_steps for h in history):
                raise ValueError('Optimizer step count disagrees with fixed sample/batch budget')
            runs[name] = result
            rows.extend({'variant': variant, 'seed': seed, **row} for row in epoch_rows)
    summary = {'status': 'V12C_TRAJECTORY_DIAGNOSTIC_COMPLETE',
               'source_directory': str(source.root), 'source_protocol_sha256': PROTOCOL_SHA256,
               'source_git_head': source.summary['environment']['git_head'],
               'diagnostic_code_sha256': digest(Path(__file__).read_bytes()),
               'input_sha256': source.hashes, 'model_training_performed': False,
               'checkpoint_loaded': False, 'audit_arrays_read': False, 'new_model_selection_performed': False,
               'validation_arrays_read': False, 'test_split_read': False, 'independent_confirmation': False,
               'baseline_selection': baseline, 'runs': runs,
               'limitations': [
                   'Online fit MAE is averaged while weights change, in standardized units; selection MAE is raw.',
                   'No fixed-checkpoint fit evaluation: cannot directly measure a generalization gap.',
                   'Saved gradients are epoch maxima of total-objective parameter norms before clipping.',
                   'Per-term/per-region gradients, penalty losses and initial hidden weights were not saved.',
                   'Selection gate RMS is not the training penalty or an optimizer update norm.',
                   'Regional accounting is not causal attribution or proof of gradient dilution.',
                   'Summary metadata contains earlier audit results; only selection statistics are analyzed.',
                   'No new thresholds, winner selection or independent confirmation.']}
    partial.mkdir(parents=True)
    with (partial / 'epochs.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (partial / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    # mkdir already reserves this run's partial directory; never replace an existing final directory.
    if output.exists():
        raise FileExistsError('Final diagnostic output appeared during analysis')
    partial.rename(output)
    report(summary)
    print('Saved diagnostic:', output / 'summary.json')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    diagnose(args.source_dir, args.output)


if __name__ == '__main__':
    main()
