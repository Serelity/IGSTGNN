"""Five independent single-layer gate interventions at the selected P2 checkpoint."""

import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.chronological import probe_incident_routing_p2 as probe

METRICS = ('mae_macro', 'mae_pooled', 'h1_h3_mae_macro', 'h4_h6_mae_macro', 'h7_h12_mae_macro')
STATUS = 'P2_FIXED_CHECKPOINT_LAYER_GATE_PROBE_COMPLETE'
INTERPRETATION = [
    'Each case disables only one incident_condition output; the other four branches stay ON.',
    'layer_off_minus_all_on > 0 means removing that branch raises error at these fixed weights.',
    'layer_off_minus_all_on < 0 means removing that branch lowers error at these fixed weights.',
    'Effects are conditional on the other four branches being ON and are not additive.',
    'OFF retains the ACDG-trained backbone, ICSF and TIID; it is not the trained fixed baseline.',
    'These reused-validation diagnostics do not select a new architecture or establish retrained gains.',
    'Gate statistics count hidden positions; incident effects are not causally identified.',
]


def case_plan(model):
    names = [f'layers.{i}.estimation_gate' for i in range(5)]
    probe.require(list(probe.GateIntervention(model, 'on').gates) == names,
                  'Expected exactly the original five ordered DecoupleLayer gates')
    return [{'case': f'layer_{i + 1}_off', 'layer_number': i + 1, 'disabled_layer': name}
            for i, name in enumerate(names)]


def verify_gate_statistics(rows, disabled_layer):
    expected = {f'layers.{i}.estimation_gate' for i in range(5)}
    observed = [(row['layer'], row['region']) for row in rows]
    probe.require(len(observed) == 10 and set(observed) == {
        (name, region) for name in expected for region in ('associated_nodes', 'nonassociated_nodes')
    }, 'Incomplete layer/region gate statistics')
    for row in rows:
        disabled = row['layer'] == disabled_layer
        probe.require(row['branch_disabled'] == disabled, 'Unexpected layer intervention')
        if row['gate_positions']:
            probe.require(row['applied_delta_abs_max'] == (0. if disabled else row['proposed_delta_abs_max']),
                          'An untargeted gate changed or the selected residual was not zeroed')
            if disabled or row['region'] == 'nonassociated_nodes':
                probe.require(row['gate_change_abs_max'] == 0.,
                              f"Expected zero direct gate change: layer={row['layer']}, "
                              f"region={row['region']}, disabled={disabled}, "
                              f"applied_delta_abs_max={row['applied_delta_abs_max']}, "
                              f"gate_change_abs_max={row['gate_change_abs_max']}")


def compare_case(reference, on, off, case):
    values, horizon_rows = probe.comparisons(reference, on, off)
    regions, rows = {}, []
    for region, result in values.items():
        metrics = {label: result[old] for label, old in (
            ('fixed_saved', 'fixed_saved'), ('all_on', 'acdg_on'), ('layer_off', 'acdg_off'))}
        differences = {key: (metrics['layer_off'][key] - metrics['all_on'][key]
                             if metrics['all_on'][key] is not None else None) for key in METRICS}
        regions[region] = {**metrics, 'layer_off_minus_all_on': differences,
                           'layer_off_minus_fixed_mae': result['off_minus_fixed_mae']}
        rows.append({**case, 'region': region, 'fixed_saved_mae': metrics['fixed_saved']['mae_macro'],
                     'all_on_mae': metrics['all_on']['mae_macro'],
                     'layer_off_mae': metrics['layer_off']['mae_macro'],
                     **{f'layer_off_minus_all_on_{key}': value for key, value in differences.items()},
                     'layer_off_minus_fixed_mae': result['off_minus_fixed_mae']})
    horizons = [{**case, 'region': row['region'], 'horizon': row['horizon'],
                 'fixed_saved_mae': row['fixed_saved_mae'], 'all_on_mae': row['acdg_on_mae'],
                 'layer_off_mae': row['acdg_off_mae'],
                 'layer_off_minus_all_on_mae': row['off_minus_on_mae'], 'valid_count': row['valid_count']}
                for row in horizon_rows]
    return regions, rows, horizons


def layerwise_inference(model, dataset, batch_size, device, saved, output):
    plan = case_plan(model)
    print('Replaying the selected checkpoint with all five branches ON', flush=True)
    on, rows, elapsed = probe.infer(model, dataset, batch_size, device, 'on')
    replay = probe.replay_check(on, saved['acdg'])
    probe.train.atomic_json(output / 'on_replay_check.json', replay)
    print(json.dumps({'on_replay': replay}), flush=True)
    probe.require(replay['passed'], 'ON replay failed; no layer-off cases were run')
    verify_gate_statistics(rows, None)
    probe.train.atomic_npz(output / 'all_on_predictions.npz', **on)
    gate_rows = [{'case': 'all_on', **row} for row in rows]
    seconds = {'all_on': elapsed}
    cases, comparison_rows, horizon_rows = [], [], []
    for case in plan:
        print(f"Running {case['case']}: only {case['disabled_layer']}.incident_condition is OFF", flush=True)
        off, rows, elapsed = probe.infer(model, dataset, batch_size, device, 'layer_off',
                                         disabled_layer=case['disabled_layer'])
        verify_gate_statistics(rows, case['disabled_layer'])
        regions, compared, horizons = compare_case(saved['fixed'], on, off, case)
        prediction_file = f"{case['case']}_predictions.npz"
        probe.train.atomic_npz(output / prediction_file, **off)
        del off  # Keep only one intervention's prediction array in memory at a time.
        cases.append({**case, 'regions': regions, 'prediction_file': prediction_file})
        comparison_rows.extend(compared)
        horizon_rows.extend(horizons)
        gate_rows.extend({'case': case['case'], **row} for row in rows)
        seconds[case['case']] = elapsed

    print('Replaying all branches ON again to verify intervention restoration', flush=True)
    restored, rows, elapsed = probe.infer(model, dataset, batch_size, device, 'on')
    restoration = probe.replay_check(restored, on)
    # Identical inputs, weights, deterministic evaluator and device within this process.
    probe.require(restoration['passed'] and restoration['max_abs_prediction_difference'] == 0.,
                  'Final all-ON predictions did not exactly restore the first all-ON pass')
    verify_gate_statistics(rows, None)
    probe.train.atomic_json(output / 'on_restoration_check.json', restoration)
    gate_rows.extend({'case': 'all_on_restored', **row} for row in rows)
    seconds['all_on_restored'] = elapsed
    probe.write_csv(output / 'layer_comparisons.csv', comparison_rows)
    probe.write_csv(output / 'per_horizon.csv', horizon_rows)
    probe.write_csv(output / 'gate_statistics.csv', gate_rows)
    return {'cases': cases, 'gate_statistics': gate_rows, 'seconds': seconds,
            'on_replay': replay, 'on_restoration': restoration, 'inference_passes': len(seconds)}


def run_probe(root, data_dir, output):
    probe.require(not output.exists(), 'Use a new output directory')
    source = probe.load_probe_source(root, data_dir)
    plan = case_plan(source.model)
    output.mkdir(parents=True)
    probe.train.atomic_json(output / 'probe_config.json', {
        'probe': 'P2_BEST_CHECKPOINT_SINGLE_LAYER_GATE_RESIDUAL_OFF',
        'cases': plan, 'inference_passes': 7, 'cumulative': False,
        'selected_epoch': source.summary['best_epoch'], 'model_state_sha256': source.weight_hash,
        'batch_size': source.batch_size, 'samples': len(source.dataset),
        'seed': source.summary['identity']['seed'], 'optimizer_steps': 0, 'parameter_updates': 0,
        'prediction_replay_atol': probe.REPLAY_PREDICTION_ATOL,
        'mae_replay_atol': probe.REPLAY_MAE_ATOL, 'restoration_prediction_atol': 0.,
        'saturation_edge': probe.SATURATION_EDGE, 'input_sha256': source.hashes,
    })
    print(json.dumps({'best_epoch': source.summary['best_epoch'], 'cases': plan,
                      'environment': probe.probe_environment(source.device)}), flush=True)
    result = layerwise_inference(source.model, source.dataset, source.batch_size, source.device,
                                 source.saved, output)
    probe.verify_probe_source(source)
    report = {
        'status': STATUS, 'scientific_status': 'POSTHOC_FIXED_WEIGHT_INTERVENTION_NOT_RETRAINED_ABLATION',
        'pair_directory': str(root), 'selected_epoch': source.summary['best_epoch'],
        'model_state_sha256': source.weight_hash, 'model_state_unchanged': True,
        'input_files_unchanged': True, 'optimizer_steps': 0, 'parameter_updates': 0,
        'cumulative': False, **result, 'environment': probe.probe_environment(source.device),
        'input_sha256': source.hashes,
        'analysis_source_sha256': {str(path.relative_to(REPO)): probe.digest(path) for path in (
            Path(__file__), Path(probe.__file__),
            Path(__file__).with_name('continue_incident_routing_p2.py'),
            Path(__file__).with_name('diagnose_incident_routing_p2.py'))},
        'interpretation': INTERPRETATION,
        'output_sha256': {path.name: probe.digest(path) for path in output.iterdir() if path.is_file()},
    }
    probe.train.atomic_json(output / 'summary.json', report)
    compact = {key: report[key] for key in (
        'status', 'scientific_status', 'selected_epoch', 'model_state_unchanged',
        'input_files_unchanged', 'optimizer_steps', 'cumulative', 'inference_passes',
        'on_replay', 'on_restoration', 'seconds', 'interpretation')}
    compact['cases'] = [{**{key: case[key] for key in ('case', 'layer_number', 'disabled_layer')},
                         'regions': {region: {
                             'fixed_saved_mae': values['fixed_saved']['mae_macro'],
                             'all_on_mae': values['all_on']['mae_macro'],
                             'layer_off_mae': values['layer_off']['mae_macro'],
                             'layer_off_minus_all_on': values['layer_off_minus_all_on'],
                         } for region, values in case['regions'].items()}} for case in report['cases']]
    print(json.dumps(compact, indent=2, allow_nan=False), flush=True)
    print(f'Saved layer gate probe: {output}', flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    run_probe(args.run_dir.resolve(), args.data_dir.resolve(), args.output_dir.resolve())


if __name__ == '__main__':
    main()
