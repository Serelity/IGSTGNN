"""v12i: full-fit inference of fixed v12f selected adapters, without training."""

import argparse
import csv
import gc
import hashlib
import io
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
from experiments.chronological import audit_state_interaction_transfer as transfer
from experiments.chronological import train_incident_state_interaction as inference
from experiments.chronological.state_interaction_fit import analyze_phases, ESTIMANDS, REGIONS

base = inference.base
PROTOCOL = Path(__file__).with_name('state_interaction_fit_v12i.json')
PROTOCOL_SHA256 = '7c5e7090fed6e1b28660accc76331db43b5b42fa306c0c5b6cdcaa23b8bf08ea'
UNAVAILABLE = 'UNAVAILABLE_SAVED_AGGREGATES_ONLY'


def load_protocol():
    if base.sha256(PROTOCOL) != PROTOCOL_SHA256:
        raise ValueError('Frozen v12i protocol changed')
    protocol = json.loads(PROTOCOL.read_text(encoding='utf-8'))
    if transfer.PROTOCOL_SHA256 != protocol['source_reader_protocol_sha256']:
        raise ValueError('Frozen source reader protocol changed')
    return protocol


def verify_runtime(source, protocol):
    if source.summary['environment']['git_head'] != protocol['source_v12f_git_head']:
        raise ValueError('Require the frozen v12f source implementation')
    for name in protocol['runtime_source_identity_files']:
        if base.sha256(REPO / name) != source.summary['code_sha256'][name]:
            raise ValueError(f'Inference implementation differs from saved source: {name}')


def verify_input_identity(hashes, source, checkpoint):
    recorded = {(Path(name).name, digest) for name, digest in source.summary['inputs'].items()}
    current = {(Path(name).name, digest) for name, digest in hashes.items()}
    if recorded != current:
        raise ValueError('Train input fingerprints disagree with original v12f provenance')
    payload = Path(checkpoint).read_bytes()
    if hashlib.sha256(payload).hexdigest() != source.identity['checkpoint_sha256']:
        raise ValueError('Original A checkpoint hash differs from saved source')
    return torch.load(io.BytesIO(payload), map_location='cpu', weights_only=True)


def compact(record):
    columns = [record['regions'].index(region) for region in REGIONS]
    return {**{key: record[key] for key in ('ids', 'source_indices', 'candidate_mask')},
            **{key: record[key][:, columns] for key in ('errors', 'counts', 'prediction_counts')},
            'regions': list(REGIONS)}


def assert_frozen(model, expected):
    if inference.state_hash(model) != expected:
        raise ValueError('Fixed model tensor state changed during inference')
    if model.training or any(p.requires_grad or p.grad is not None for p in model.parameters()):
        raise ValueError('Inference model must be eval with all parameters/gradients disabled')


def evaluate(model, dataset, indices, batch_size, device, progress):
    before = inference.state_hash(model)
    assert_frozen(model, before)
    record = compact(inference.evaluate(model, dataset, indices, batch_size, device, progress))
    assert_frozen(model, before)
    return record


def load_selected(model, source, arm, seed, protocol):
    detail = source.summary['runs'][str(seed)][arm]
    checkpoint = torch.load(io.BytesIO(source.read(f'{arm}_s{seed}/selected_gate.pt')),
                            map_location='cpu', weights_only=True)
    expected = {'variant': arm, 'seed': seed, 'epoch': detail['selected_epoch'],
                'protocol_sha256': source.summary['protocol_sha256'],
                'backbone_state_sha256': base.backbone_hash(model)}
    if (not isinstance(checkpoint, dict) or set(checkpoint) != {*expected, 'gate_state'}
            or type(detail['selected_epoch']) is not int
            or not 0 <= detail['selected_epoch'] <= source.inherited['training']['epochs']
            or any(checkpoint.get(key) != value or type(checkpoint.get(key)) is not type(value)
                   for key, value in expected.items())):
        raise ValueError('Selected adapter checkpoint identity disagrees with source')
    gate = inference.attach_adapter(model, arm, source.inherited['training']['node_hidden_width'])
    state = checkpoint['gate_state']
    if (not isinstance(state, dict) or not state
            or any(not isinstance(value, torch.Tensor) or not torch.isfinite(value).all()
                   for value in state.values())):
        raise ValueError('Selected adapter state contains invalid/nonfinite tensors')
    gate.load_state_dict(state, strict=True)
    digest = inference.state_hash(gate)
    declared = detail['representation_diagnostics']['selected']['adapter_state_sha256']
    if (digest != declared
            or sum(p.numel() for p in gate.parameters()) != detail['trainable_parameters']):
        raise ValueError('Selected adapter state or parameter budget disagrees with saved probe')
    if detail['selected_epoch'] == 0 and digest != detail['representation_diagnostics']['initial']['adapter_state_sha256']:
        raise ValueError('Epoch-zero adapter differs from original identity initialization')
    model.requires_grad_(False).eval()
    base.assert_backbone(model, expected['backbone_state_sha256'])
    return {'selected_epoch': detail['selected_epoch'], 'adapter_state_sha256': digest,
            'backbone_state_sha256': expected['backbone_state_sha256'],
            'adapter_parameters': detail['trainable_parameters']}


def probe_replay(full, probe, tolerance):
    positions = {int(sample): i for i, sample in enumerate(full['ids'])}
    try:
        selected = transfer.restrict(full, [positions[int(sample)] for sample in probe['ids']])
    except KeyError as error:
        raise ValueError('Saved probe lies outside complete fit evaluation') from error
    transfer.aligned(selected, probe)
    transfer.close(selected['errors'], probe['errors'], 'Complete-fit/saved-probe replay', **tolerance)
    delta = selected['errors'] - probe['errors']
    cells = probe['counts'].sum(0)
    return {'samples': len(probe['ids']),
            'maximum_absolute_sample_region_error_sum_difference': float(np.abs(delta).max()),
            'pooled_mae_difference_full_minus_saved_probe': {
                region: float(delta[:, r].sum() / cells[r]) if cells[r] else None
                for r, region in enumerate(REGIONS)},
            'error_sum_tolerance': tolerance,
            'full_fit_batch_size': 16, 'saved_probe_batch_size': 8}


def stack(reference, learned):
    ordered = [reference] + [learned[arm] for arm in transfer.ARMS]
    for record in ordered[1:]:
        transfer.aligned(reference, record)
    return {**reference, 'errors': np.stack([record['errors'] for record in ordered], axis=1)}


def selection_context(source, seed, reference, fit_analysis):
    baseline = transfer.metrics(reference)
    maes = {'A': baseline['mae']}
    for arm in transfer.ARMS:
        selected = source.summary['runs'][str(seed)][arm]['selection_metrics']['incident_full']
        if type(selected['samples']) is not int or selected['samples'] != baseline['samples']:
            raise ValueError('Saved selection sample budget changed')
        maes[arm] = {region: selected['mae'][region] for region in REGIONS}
    fractions = baseline['valid_cell_fraction']
    for path, values in maes.items():
        for region, value in values.items():
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                      or not np.isfinite(value) or value < 0):
                raise ValueError('Invalid saved selection MAE')
            if (value is None) != (reference['counts'][:, REGIONS.index(region)].sum() == 0):
                raise ValueError('Saved selection MAE availability disagrees with target support')
        reconstructed = sum(fractions[r] * values[r] for r in REGIONS[1:] if fractions[r])
        transfer.close(reconstructed, values['all'], f'{path} saved selection regional partition')
    result = {'samples': baseline['samples'], 'statistics_source': 'saved_aggregate_selection_metrics',
              'equal_forecast_window_status': UNAVAILABLE, 'weekly_intervals_status': UNAVAILABLE,
              'regions': {}}
    rows, gaps = [], []
    for r, region in enumerate(REGIONS):
        item = {'mae': {path: values[region] for path, values in maes.items()},
                'valid_cells': int(reference['counts'][:, r].sum()),
                'valid_cell_fraction': fractions[region], 'comparisons': {}}
        for name, (left, right) in transfer.comparisons().items():
            a, b = item['mae'][left], item['mae'][right]
            gain = a - b if a is not None and b is not None else None
            contribution = gain * fractions[region] if gain is not None else (0. if fractions[region] == 0 else None)
            effect = {'gain_raw_mae': gain, 'equal_forecast_window_gain_raw_mae': None,
                      'regional_gain_in_global_mae_units': contribution}
            item['comparisons'][name] = effect
            row = {'phase': 'selection', 'region': region, 'comparison': name,
                   'left_path': left, 'right_path': right, 'mae_left': a, 'mae_right': b,
                   'valid_cells': item['valid_cells'], 'valid_cell_fraction': fractions[region],
                   'statistics_source': result['statistics_source'],
                   'equal_forecast_window_status': UNAVAILABLE, **effect}
            for method in ('week', 'four_week_block'):
                for estimand in ESTIMANDS:
                    row.update({f'{method}_{estimand}_status': UNAVAILABLE,
                                **{f'{method}_{estimand}_{key}': None
                                   for key in ('valid_draws', 'ci_low', 'ci_high')}})
            rows.append(row)
            for estimand in ('pooled', 'global_units'):
                field = ESTIMANDS[estimand]
                fit = fit_analysis['regions'][region]['comparisons'][name][field]
                current = effect[field]
                gap = current - fit if current is not None and fit is not None else None
                gaps.append({'phase': 'selection', 'reference_phase': 'fit', 'region': region,
                    'comparison': name, 'estimand': estimand, 'fit_gain_raw_mae': fit,
                    'phase_gain_raw_mae': current, 'phase_minus_fit_gain_raw_mae': gap,
                    'status': 'DEFINED' if gap is not None else 'UNDEFINED_EMPTY_SUPPORT',
                    'paired_windows': False, 'gap_confidence_interval_constructed': False})
        result['regions'][region] = item
    return result, rows, gaps


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(source_dir, data_dir, primary_dir, secondary_dir, checkpoint, output, device='cuda:0'):
    protocol = load_protocol()
    requested = Path(output)
    requested_partial = requested.with_name(requested.name + '.partial')
    if any(path.exists() or path.is_symlink() for path in (requested, requested_partial)):
        raise FileExistsError('Preserve output/partial; choose a new evaluation run')
    output, source_root = requested.resolve(), Path(source_dir).resolve()
    partial = output.with_name(output.name + '.partial')
    if source_root.name.endswith('.partial'):
        raise ValueError('Require a completed final v12f source, not partial')
    for root in (source_root, Path(data_dir).resolve(), Path(primary_dir).resolve(),
                 Path(secondary_dir).resolve(), Path(checkpoint).resolve().parent):
        if output.is_relative_to(root) or partial.is_relative_to(root):
            raise ValueError('Write evaluation output outside all read-only input directories')
    partial.mkdir(parents=True)
    started = time.monotonic()
    def progress(stage, **fields):
        event = {'stage': stage, 'elapsed_seconds': time.monotonic() - started,
                 'memory': base.memory_snapshot(device), **fields}
        base.write_json(partial / 'progress.json', event)
        print(json.dumps(event), flush=True)
    try:
        torch.set_num_threads(3)
        device = torch.device(device)
        base.configure_determinism(device)
        progress('verifying_completed_source')
        source = transfer.Source(source_root, transfer.load_protocol())
        verify_runtime(source, protocol)
        backbone_protocol = base.mechanisms.load_protocol(base.mechanisms.PROTOCOL)
        if base.sha256(base.mechanisms.PROTOCOL) != source.inherited['v12a_protocol_sha256']:
            raise ValueError('Original backbone protocol changed')
        baseline, hashes = base.mechanisms.verify_inputs(data_dir, primary_dir, secondary_dir,
                                                        checkpoint, backbone_protocol)
        state = verify_input_identity(hashes, source, checkpoint)
        manifest, _ = transfer.read_manifest(data_dir, source)
        plan = base.make_plan(base.read_csv(Path(data_dir) / 'train_manifest.csv'),
                             base.read_csv(Path(primary_dir) / 'train_control_manifest.csv'),
                             base.read_csv(Path(secondary_dir) / 'train_second_control_manifest.csv'), source.inherited)
        for phase in plan:
            for key in ('bounds', 'indices', 'positive_ids', 'matched_ids'):
                if plan[phase][key] != source.plan[phase][key]:
                    raise ValueError('Recomputed manifest eligibility differs from original source')
        fit = plan['fit']['indices']['incident_full']
        if len(fit) != protocol['fit_samples'] or protocol['evaluation_batch_size'] != source.inherited['training']['evaluation_batch_size']:
            raise ValueError('Complete fit budget or evaluation batching changed')
        probe_indices = [fit[i] for i in np.linspace(0, len(fit) - 1,
            min(len(fit), source.summary['frozen_protocol']['representation_probe']['samples']), dtype=int)]
        if source.identity['probe_indices'] != probe_indices:
            raise ValueError('Saved fit probe does not follow original manifest-only rule')
        probe_ids = [list(manifest)[i] for i in probe_indices]
        dataset = base.FullPositiveDataset(data_dir, 'train', baseline['expected_positive_samples']['train'],
                                           baseline['expected_sensor_count'])
        def native():
            model = base.make_model(Path(data_dir), len(dataset.station_ids), device, 'fixed')
            model.load_state_dict(state, strict=True)
            if sum(p.numel() for p in model.parameters()) != baseline['checkpoint']['parameters']:
                raise ValueError('Original A parameter budget changed')
            return model.requires_grad_(False).eval()
        def full_fit(model, arm, seed=None):
            record = evaluate(model, dataset, fit, protocol['evaluation_batch_size'], device,
                lambda stage, **fields: progress(stage, arm=arm, seed=seed, phase='fit', **fields))
            if (not np.array_equal(record['ids'], plan['fit']['positive_ids'])
                    or not np.array_equal(record['source_indices'], fit)):
                raise ValueError('Full-fit evaluation changed original ID/source order')
            return record
        progress('train_inputs_and_fit_eligibility_verified', full_fit_samples=len(fit),
                 saved_probe_samples=len(probe_ids))
        model = native()
        reference_fit = full_fit(model, 'A')
        base.save_arrays(partial / 'fit_A.npz', reference_fit)
        del model
        reference = {}
        for phase in ('selection', 'audit'):
            record = transfer.read_record(source, f'{phase}_A_incident_full.npz', plan[phase]['positive_ids'])
            if not np.array_equal(record['source_indices'], plan[phase]['indices']['incident_full']):
                raise ValueError('Saved full-positive source positions changed')
            reference[phase] = record
        transfer.reconcile_metrics(reference['audit'], source.summary['baseline_audit']['incident_full'], 'A audit')
        times = {phase: {sample: manifest[sample]['t0'] for sample in plan[phase]['positive_ids']}
                 for phase in ('fit', 'audit')}
        results, selected_models, replays, changes = {}, {}, {}, []
        phase_rows, week_rows = [], []
        for seed in protocol['seeds']:
            fit_arms, audit_arms = {}, {}
            selected_models[str(seed)], replays[str(seed)] = {}, {}
            for arm in protocol['arms']:
                directory = f'{arm}_s{seed}'
                detail = source.summary['runs'][str(seed)][arm]
                progress('selected_adapter_inference_started', arm=arm, seed=seed,
                         selected_epoch=detail['selected_epoch'])
                model = native()
                metadata = load_selected(model, source, arm, seed, protocol)
                fit_record = full_fit(model, arm, seed)
                if metadata['selected_epoch'] == 0:
                    transfer.aligned(reference_fit, fit_record)
                    transfer.close(reference_fit['errors'], fit_record['errors'], 'Epoch-zero full-fit identity', rtol=0, atol=0)
                base.save_arrays(partial / f'fit_{arm}_s{seed}.npz', fit_record)
                audit_record = transfer.read_record(source, f'{directory}/audit_incident_full.npz', plan['audit']['positive_ids'])
                transfer.aligned(reference['audit'], audit_record)
                transfer.reconcile_metrics(audit_record, detail['audit']['incident_full'], f'{directory} audit')
                replay = {}
                for label, full in (('initial', reference_fit), ('selected', fit_record)):
                    probe = transfer.read_record(source, f'{directory}/representation_{label}.npz', probe_ids)
                    declared = detail['representation_diagnostics'][label]
                    transfer.reconcile_metrics(probe, declared, f'{directory} probe {label}')
                    if declared['sample_ids'] != probe_ids or not np.array_equal(probe['source_indices'], probe_indices):
                        raise ValueError('Saved probe source positions/order changed')
                    replay[label] = probe_replay(full, probe, protocol['probe_replay_error_tolerance'])
                fit_arms[arm], audit_arms[arm] = fit_record, audit_record
                selected_models[str(seed)][arm] = {**metadata, 'model_state_unchanged': True,
                    'full_fit_samples': len(fit_record['ids']),
                    'epoch_zero_full_fit_error_sums_exact_A': True if metadata['selected_epoch'] == 0 else None}
                replays[str(seed)][arm] = replay
                del model
                gc.collect()
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                progress('selected_adapter_inference_complete', arm=arm, seed=seed)
            analysis, rows, weeks, gaps = analyze_phases(
                {'fit': stack(reference_fit, fit_arms), 'audit': stack(reference['audit'], audit_arms)},
                transfer.comparisons(), source.inherited['bootstrap'], times)
            selection, selection_rows, selection_gaps = selection_context(source, seed, reference['selection'], analysis['phases']['fit'])
            analysis['selection_point_context'] = selection
            results[str(seed)] = analysis
            phase_rows.extend({'seed': seed, 'statistics_source': 'new_full_fit_inference' if row['phase'] == 'fit'
                               else 'saved_per_window_audit', **row} for row in rows)
            phase_rows.extend({'seed': seed, **row} for row in selection_rows)
            week_rows.extend({'seed': seed, **row} for row in weeks)
            changes.extend({'seed': seed, **row} for row in [*gaps, *selection_gaps])
            progress('full_fit_phase_statistics_complete', seed=seed)
        write_csv(partial / 'phase_metrics.csv', phase_rows)
        write_csv(partial / 'weekly_metrics.csv', week_rows)
        write_csv(partial / 'phase_gain_changes.csv', changes)
        code_files = [Path(__file__), PROTOCOL, Path(__file__).with_name('state_interaction_fit.py'),
                      Path(transfer.__file__), transfer.PROTOCOL,
                      transfer.SOURCE_PROTOCOL, transfer.INHERITED_PROTOCOL, base.mechanisms.PROTOCOL,
                      Path(__file__).with_name('state_interaction_trajectory.py'),
                      Path(__file__).with_name('audit_matched_controls.py'),
                      Path(__file__).with_name('run_state_interaction_fit_audit.sh'),
                      Path(__file__).with_name('audit_architecture_regions.py')]
        code_files.extend(REPO / name for name in protocol['runtime_source_identity_files'])
        summary = {'status': 'STATE_INTERACTION_FULL_FIT_AUDIT_COMPLETE',
            'protocol_id': protocol['protocol_id'], 'protocol_sha256': PROTOCOL_SHA256,
            'frozen_protocol': protocol, **protocol['information_boundary'], 'main_training_ready': False,
            'recommendation': protocol['decision'], 'interpretation': protocol['interpretation'],
            'source_directory': str(source_root), 'source_protocol_sha256': source.summary['protocol_sha256'],
            'source_git_head': source.summary['environment']['git_head'],
            'phase_samples': source.summary['phase_samples'], 'results': results,
            'selected_models': selected_models, 'saved_probe_replay': replays, 'phase_gain_changes': changes,
            'inputs': {**hashes, **source.hashes},
            'code_sha256': {str(path.relative_to(REPO)): base.sha256(path) for path in code_files},
            'environment': {'host': socket.gethostname(), 'device': str(device), 'python': sys.version,
                'torch': torch.__version__, 'numpy': np.__version__, 'threads': torch.get_num_threads(),
                'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
                'tf32': torch.backends.cuda.matmul.allow_tf32,
                'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
                'git_head': base.subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()},
            'outputs': {path.name: base.sha256(path) for path in partial.iterdir()
                        if path.is_file() and path.name != 'progress.json'}}
        base.write_json(partial / 'summary.json', summary)
        if output.exists() or output.is_symlink():
            raise FileExistsError('Final output appeared; preserve partial')
        partial.rename(output)
    except BaseException as error:
        if partial.exists():
            base.write_json(partial / 'failure.json', {'error': str(error), 'traceback': traceback.format_exc()})
        raise
    report(summary)
    print('Saved v12i complete-fit audit:', output / 'summary.json', flush=True)
    return summary


def report(summary):
    print('status:', summary['status'])
    print('protocol_sha256:', summary['protocol_sha256'])
    print('FIXED SELECTED MODELS; FULL FIT IS IN-SAMPLE; NO NEW TRAINING OR MODEL SELECTION')
    for seed, analysis in summary['results'].items():
        for arm in summary['frozen_protocol']['arms']:
            comparison = arm + '_vs_A'
            print(f'\n[{arm} seed={seed}] saved_selected_epoch={summary["selected_models"][seed][arm]["selected_epoch"]}')
            for region in REGIONS:
                f = analysis['phases']['fit']['regions'][region]['comparisons'][comparison]
                a = analysis['phases']['audit']['regions'][region]['comparisons'][comparison]
                s = analysis['selection_point_context']['regions'][region]['comparisons'][comparison]
                print(region, 'pooled_gain_fit_selection_audit=', [f['gain_raw_mae'], s['gain_raw_mae'], a['gain_raw_mae']],
                      'equal_window_gain_fit_audit=', [f['equal_forecast_window_gain_raw_mae'], a['equal_forecast_window_gain_raw_mae']])
            f = analysis['phases']['fit']['regions']['candidate_h1_h6']['comparisons'][comparison]
            print('fit_early_block_CI_in_sample=', f['intervals']['four_week_block']['pooled'])
            print('early_phase_changes=', [row for row in summary['phase_gain_changes'] if
                row['seed'] == int(seed) and row['comparison'] == comparison
                and row['region'] == 'candidate_h1_h6' and row['estimand'] == 'pooled'])
            print('probe_replay=', summary['saved_probe_replay'][seed][arm])
    print('Selection has aggregate pooled points only. Phase changes are unpaired descriptive points, not causal or independent confirmation.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    execute = commands.add_parser('run')
    for name in ('source-dir', 'data-dir', 'primary-control-dir', 'secondary-control-dir', 'checkpoint', 'output'):
        execute.add_argument('--' + name, type=Path, required=True)
    execute.add_argument('--device', default='cuda:0')
    commands.add_parser('report').add_argument('summary', type=Path)
    args = parser.parse_args()
    if args.action == 'report':
        if args.summary.is_file():
            report(json.loads(args.summary.read_text(encoding='utf-8')))
        else:
            print('INCOMPLETE: no final summary; inspect .job/run.log and preserve .partial.')
    else:
        run(args.source_dir, args.data_dir, args.primary_control_dir, args.secondary_control_dir,
            args.checkpoint, args.output, args.device)


if __name__ == '__main__':
    main()
