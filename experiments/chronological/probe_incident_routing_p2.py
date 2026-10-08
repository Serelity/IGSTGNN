"""Fixed-checkpoint inference with the five ACDG residual gates on versus off."""

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.chronological import train
from experiments.chronological.continue_incident_routing_p2 import (
    COMPLETE, PROTOCOL, checkpoint_action, digest, load_summaries, require, validate_pair,
)
from experiments.chronological.diagnose_incident_routing_p2 import (
    aggregate, curve_diagnostics, event_sums, paired_predictions, read_metadata,
    read_predictions, write_csv,
)

REPLAY_PREDICTION_ATOL = 1e-5  # Absolute original flow units; fixed before running the probe.
REPLAY_MAE_ATOL = 1e-8
SATURATION_EDGE = .01  # Descriptive threshold, never a model-selection rule.
REGIONS = ('all_nodes', 'associated_nodes', 'nonassociated_nodes')


class GateStats:
    def __init__(self):
        self.count = 0
        self.sums = np.zeros(7, dtype=np.float64)
        self.maximum = np.zeros(3, dtype=np.float64)

    def update(self, raw_delta, applied_delta, native_gate, actual_gate, mask):
        raw, applied = raw_delta[mask].double(), applied_delta[mask].double()
        native, actual = native_gate[mask].double(), actual_gate[mask].double()
        if not raw.numel():
            return
        shift = (actual - native).abs()
        saturated_native = (native <= SATURATION_EDGE) | (native >= 1 - SATURATION_EDGE)
        saturated_actual = (actual <= SATURATION_EDGE) | (actual >= 1 - SATURATION_EDGE)
        self.count += raw.numel()
        self.sums += torch.stack([
            raw.abs().sum(), applied.abs().sum(), native.sum(), actual.sum(), shift.sum(),
            saturated_native.double().sum(), saturated_actual.double().sum(),
        ]).cpu().numpy()
        self.maximum = np.maximum(self.maximum, torch.stack([
            raw.abs().max(), applied.abs().max(), shift.max(),
        ]).cpu().numpy())

    def result(self):
        names = ('proposed_delta_abs_mean', 'applied_delta_abs_mean', 'native_gate_mean',
                 'actual_gate_mean', 'gate_change_abs_mean', 'native_saturation_fraction',
                 'actual_saturation_fraction')
        result = dict(zip(names, (self.sums / self.count).tolist() if self.count else [None] * len(names)))
        result.update(zip(('proposed_delta_abs_max', 'applied_delta_abs_max', 'gate_change_abs_max'),
                          self.maximum.tolist() if self.count else [None] * 3))
        return {'gate_positions': self.count, **result}


class GateIntervention:
    """Observe the native gate and replace only incident_condition's output in off mode.

    The original modules and state_dict are never edited. Hooks are removed even on failure.
    Off-mode proposed deltas are evaluated on off-mode hidden states; they are descriptive.
    """
    def __init__(self, model, mode, disabled_layer=None):
        require(mode in ('on', 'off', 'layer_off'), 'Unknown gate intervention mode')
        require(not model.training, 'Inference probe requires model.eval()')
        self.mode = mode
        self.gates = {name: module for name, module in model.named_modules()
                      if name.endswith('estimation_gate') and module.incident_condition is not None}
        require(bool(self.gates), 'No ACDG gates found')
        if mode == 'layer_off':
            require(disabled_layer in self.gates, 'Select exactly one existing ACDG gate by module name')
            self.disabled_gates = frozenset((disabled_layer,))
        else:
            require(disabled_layer is None, 'A selected layer is only valid in layer_off mode')
            self.disabled_gates = frozenset(self.gates if mode == 'off' else ())
        self.handles, self.base = [], {}
        self.calls = {name: 0 for name in self.gates}
        self.samples = {name: 0 for name in self.gates}
        self.stats = {name: {region: GateStats() for region in ('associated_nodes', 'nonassociated_nodes')}
                      for name in self.gates}

    def __enter__(self):
        try:
            for name, gate in self.gates.items():
                def capture_base(module, args, output, name=name):
                    require(name not in self.base, 'Native gate output was not consumed')
                    self.base[name] = output.detach()

                def intervene(module, args, output, name=name):
                    history, incident = args
                    base = self.base.pop(name)[:, -history.shape[1]:]
                    require(base.shape == output.shape, 'Native and conditional gate shapes differ')
                    require(torch.isfinite(output).all().item(), 'Nonfinite conditional gate')
                    support = (incident['distances'].abs().sum(-1) > 0)[:, None, :, None].expand_as(output)
                    require(torch.count_nonzero(output[~support]).item() == 0,
                            'Conditional gate is nonzero outside its support')
                    disabled = name in self.disabled_gates
                    applied = torch.zeros_like(output) if disabled else output
                    native, actual = torch.sigmoid(base), torch.sigmoid(base + applied)
                    require(torch.isfinite(actual).all().item(), 'Nonfinite effective gate')
                    for region, mask in (('associated_nodes', support), ('nonassociated_nodes', ~support)):
                        self.stats[name][region].update(output, applied, native, actual, mask)
                    self.calls[name] += 1
                    self.samples[name] += history.shape[0]
                    # Untargeted branches preserve the original tensor and forward path.
                    return applied if disabled else None

                self.handles.append(gate.fully_connected_layer_2.register_forward_hook(capture_base))
                self.handles.append(gate.incident_condition.register_forward_hook(intervene))
        except Exception:
            self.close()
            raise
        return self

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.base.clear()

    def __exit__(self, *_):
        self.close()

    def results(self, samples, batches):
        require(not self.base, 'Unconsumed native gate output')
        rows = []
        for name in self.gates:
            require(self.calls[name] == batches and self.samples[name] == samples,
                    f'Not all validation batches visited gate: {name}')
            for region, stats in self.stats[name].items():
                rows.append({'mode': self.mode, 'layer': name, 'region': region,
                             'branch_disabled': name in self.disabled_gates,
                             'batches': self.calls[name], 'samples': self.samples[name], **stats.result()})
        return rows


def infer(model, dataset, batch_size, device, mode, disabled_layer=None):
    indices = [int(row['sample_index']) for row in dataset.rows]
    before = train.state_sha256(model.state_dict())
    started = time.perf_counter()
    with GateIntervention(model, mode, disabled_layer) as intervention:
        with torch.inference_mode():
            _, arrays = train.evaluate(model, dataset, indices, batch_size, device, dataset.scaler, collect=True)
        rows = intervention.results(len(indices), math.ceil(len(indices) / batch_size))
    require(train.state_sha256(model.state_dict()) == before, 'Inference changed model weights or buffers')
    require(np.isfinite(arrays['prediction']).all(), 'Nonfinite inference predictions')
    arrays['sample_indices'] = np.asarray(indices, dtype=np.int64)
    arrays['station_ids'] = np.asarray(dataset.station_ids, dtype=np.int64)
    return arrays, rows, time.perf_counter() - started


def replay_check(actual, saved):
    paired_predictions(actual, saved)
    require(np.isfinite(actual['prediction']).all(), 'Nonfinite on-mode predictions')
    maximum = float(np.abs(actual['prediction'].astype(np.float64) - saved['prediction']).max())
    metric_differences = {}
    for region in REGIONS:
        a, b = (aggregate(*event_sums(arrays, region))['mae_macro'] for arrays in (actual, saved))
        require((a is None) == (b is None), 'On replay has different valid metric support')
        metric_differences[region] = abs(a - b) if a is not None else 0.
    passed = maximum <= REPLAY_PREDICTION_ATOL and max(metric_differences.values()) <= REPLAY_MAE_ATOL
    return {'passed': passed, 'max_abs_prediction_difference': maximum,
            'mae_absolute_differences': metric_differences,
            'prediction_atol': REPLAY_PREDICTION_ATOL, 'mae_atol': REPLAY_MAE_ATOL}


def inference_pair(model, dataset, batch_size, device, saved, output):
    print('Replaying ACDG best checkpoint with added gates ON', flush=True)
    on, on_stats, on_seconds = infer(model, dataset, batch_size, device, 'on')
    replay = replay_check(on, saved)
    train.atomic_json(output / 'on_replay_check.json', replay)
    print(json.dumps({'on_replay': replay}), flush=True)
    require(replay['passed'], 'ON predictions do not reproduce the saved best model; OFF was not run')
    print('Running the same checkpoint with all five added gate residuals OFF', flush=True)
    off, off_stats, off_seconds = infer(model, dataset, batch_size, device, 'off')
    return on, off, on_stats + off_stats, {'on': on_seconds, 'off': off_seconds}, replay


def comparisons(reference, on, off):
    paired_predictions(on, off)
    paired_predictions(on, reference)
    result, rows = {}, []
    for region in REGIONS:
        values = {label: aggregate(*event_sums(arrays, region)) for label, arrays in
                  (('fixed_saved', reference), ('acdg_on', on), ('acdg_off', off))}
        a, b, c = (values[label]['mae_macro'] for label in ('fixed_saved', 'acdg_on', 'acdg_off'))
        result[region] = {**values, 'off_minus_on_mae': c - b if b is not None else None,
                          'on_minus_fixed_mae': b - a if a is not None else None,
                          'off_minus_fixed_mae': c - a if a is not None else None}
        for h in range(12):
            a, b, c = (values[label]['per_horizon_mae'][h] for label in
                       ('fixed_saved', 'acdg_on', 'acdg_off'))
            rows.append({'region': region, 'horizon': h + 1, 'fixed_saved_mae': a,
                         'acdg_on_mae': b, 'acdg_off_mae': c,
                         'off_minus_on_mae': c - b if b is not None else None,
                         'valid_count': values['acdg_on']['valid_count_per_horizon'][h]})
    return result, rows


@dataclass
class ProbeSource:
    root: Path
    data_dir: Path
    summary: dict
    dataset: object
    device: torch.device
    batch_size: int
    package: dict
    saved: dict
    hashes: dict
    weight_hash: str
    model: torch.nn.Module


def load_probe_source(root, data_dir):
    """Validate the frozen experiment and load its selected weights without training."""
    summaries = load_summaries(root)
    validate_pair(summaries)
    curve_diagnostics(summaries)
    summary = summaries['acdg']
    require(all(s['status'] == COMPLETE for s in summaries.values()), 'Both source runs must be complete')
    require(torch.__version__ == summary['identity']['torch_version'], 'PyTorch version changed')
    require(np.__version__ == summary['identity']['numpy_version'], 'NumPy version changed')
    require(torch.cuda.is_available() and 'V100' in torch.cuda.get_device_name(0),
            'Use the original V100 environment for the saved-prediction replay')
    require(torch.version.cuda == summary['environment']['cuda_version'], 'CUDA build changed')
    require(int(os.environ.get('OMP_NUM_THREADS', '3')) == summary['environment']['threads'] == 3,
            'Use the original three CPU threads')
    torch.set_num_threads(3)
    device = torch.device('cuda:0')
    train.configure_determinism(device)
    protocol = json.loads(PROTOCOL.read_text())
    package = train.verify_package(data_dir)
    require(package == summary['identity']['package_sha256'], 'Data package changed')
    dataset = train.ChronologicalDataset(data_dir, 'val')
    require(len(dataset) == protocol['val_samples'] == 917 and len(dataset.station_ids) == 496,
            'Expected the complete original validation set')
    batch_size = summary['identity']['batch_size']
    require(batch_size == 48, 'Replay must preserve the original evaluation batch size')

    inputs = [root / variant / 'summary.json' for variant in ('fixed', 'acdg')]
    saved = {}
    for variant in ('fixed', 'acdg'):
        path = root / variant / 'best_validation_predictions.npz'
        saved[variant] = read_predictions(path, summaries[variant]['best_validation_predictions_sha256'])
        inputs.append(path)
    paired_predictions(saved['fixed'], saved['acdg'])
    _, manifest_hashes = read_metadata(data_dir, summary, saved['acdg']['sample_indices'])
    last_path, best_path = (root / 'acdg' / name for name in ('last_checkpoint.pt', 'best_model.pt'))
    inputs.extend((last_path, best_path, PROTOCOL))
    hashes = {str(path): digest(path) for path in inputs}
    hashes.update(manifest_hashes)
    last = torch.load(last_path, map_location='cpu', weights_only=False)
    require(checkpoint_action(summary, last, root / 'acdg') == 'skip', 'Incomplete source run')
    weights = torch.load(best_path, map_location='cpu', weights_only=True)
    weight_hash = train.state_sha256(weights)
    require(weight_hash == train.state_sha256(last['best_model_state']),
            'Exported best model differs from the selected state in last_checkpoint.pt')
    del last
    train.set_seed(summary['identity']['seed'])
    model = train.make_model(data_dir, len(dataset.station_ids), device, 'fixed', incident_routing='acdg')
    model.load_state_dict(weights, strict=True)
    del weights
    model.requires_grad_(False)
    model.eval()
    require(train.state_sha256(model.state_dict()) == weight_hash, 'Loaded model state changed')
    require(len(GateIntervention(model, 'on').gates) == 5, 'Expected the original five ACDG gates')
    return ProbeSource(root, data_dir, summary, dataset, device, batch_size, package,
                       saved, hashes, weight_hash, model)


def verify_probe_source(source):
    require(train.state_sha256(source.model.state_dict()) == source.weight_hash, 'Final weights/buffers changed')
    require(train.verify_package(source.data_dir) == source.package, 'Data package changed during the probe')
    for path, expected in source.hashes.items():
        require(digest(path) == expected, f'Input artifact changed during the probe: {path}')
    validate_pair(load_summaries(source.root))


def probe_environment(device):
    return {
        'python': sys.version, 'torch': str(torch.__version__), 'numpy': np.__version__,
        'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0), 'device': str(device),
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'threads': torch.get_num_threads(), 'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
        'cudnn_deterministic': torch.backends.cudnn.deterministic,
        'tf32': bool(torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32),
    }


def run_probe(root, data_dir, output):
    require(not output.exists(), 'Use a new output directory')
    source = load_probe_source(root, data_dir)
    summary, dataset, device = source.summary, source.dataset, source.device
    batch_size, saved, hashes = source.batch_size, source.saved, source.hashes
    weight_hash, model = source.weight_hash, source.model
    output.mkdir(parents=True)
    config = {
        'probe': 'P2_BEST_CHECKPOINT_ALL_FIVE_GATE_RESIDUALS_ON_OFF', 'modes': ['on', 'off'],
        'selected_epoch': summary['best_epoch'], 'model_state_sha256': weight_hash,
        'batch_size': batch_size, 'samples': len(dataset), 'seed': summary['identity']['seed'],
        'optimizer_steps': 0, 'parameter_updates': 0, 'saturation_edge': SATURATION_EDGE,
        'prediction_replay_atol': REPLAY_PREDICTION_ATOL, 'mae_replay_atol': REPLAY_MAE_ATOL,
        'input_sha256': hashes,
    }
    train.atomic_json(output / 'probe_config.json', config)
    print(json.dumps({'best_epoch': summary['best_epoch'], 'gpu': torch.cuda.get_device_name(0),
                      'torch': torch.__version__, 'numpy': np.__version__, 'batch_size': batch_size}), flush=True)
    on, off, gate_stats, seconds, replay = inference_pair(
        model, dataset, batch_size, device, saved['acdg'], output)
    regions, horizon_rows = comparisons(saved['fixed'], on, off)
    verify_probe_source(source)
    train.atomic_npz(output / 'paired_predictions.npz',
                     prediction_on=on['prediction'], prediction_off=off['prediction'],
                     **{name: on[name] for name in ('target', 'valid', 'associated', 'sample_indices', 'station_ids')})
    write_csv(output / 'per_horizon.csv', horizon_rows)
    write_csv(output / 'gate_statistics.csv', gate_stats)
    report = {
        'status': 'P2_FIXED_CHECKPOINT_GATE_ON_OFF_COMPLETE',
        'scientific_status': 'POSTHOC_FIXED_WEIGHT_INTERVENTION_NOT_RETRAINED_ABLATION',
        'pair_directory': str(root), 'selected_epoch': summary['best_epoch'],
        'model_state_sha256': weight_hash, 'model_state_unchanged': True, 'input_files_unchanged': True,
        'optimizer_steps': 0, 'on_replay': replay, 'regions': regions,
        'gate_statistics': gate_stats, 'seconds': seconds,
        'environment': probe_environment(device),
        'input_sha256': hashes,
        'analysis_source_sha256': {str(path.relative_to(REPO)): digest(path) for path in (
            Path(__file__), Path(__file__).with_name('continue_incident_routing_p2.py'),
            Path(__file__).with_name('diagnose_incident_routing_p2.py'))},
        'interpretation': [
            'off_minus_on_mae > 0 means the added branch lowers error at this fixed checkpoint.',
            'OFF keeps the ACDG-trained backbone, ICSF and TIID; it is not the fixed baseline.',
            'All five conditional residuals are disabled jointly; no layer attribution or re-selection.',
            'Gate summaries count hidden time/node positions, not independent samples.',
            'Ablation is a posthoc computational intervention on reused validation data, not causal incident evidence.',
        ],
    }
    report['output_sha256'] = {path.name: digest(path) for path in output.iterdir() if path.is_file()}
    train.atomic_json(output / 'summary.json', report)
    compact_regions = {region: {
        **{label: {key: values[label][key] for key in ('mae_macro', 'h1_h3_mae_macro',
                                                     'h4_h6_mae_macro', 'h7_h12_mae_macro')}
           for label in ('fixed_saved', 'acdg_on', 'acdg_off')},
        **{key: values[key] for key in ('off_minus_on_mae', 'on_minus_fixed_mae', 'off_minus_fixed_mae')},
    } for region, values in regions.items()}
    compact_gates = [{key: row[key] for key in ('mode', 'layer', 'region', 'proposed_delta_abs_mean',
                     'applied_delta_abs_max', 'gate_change_abs_mean', 'native_saturation_fraction',
                     'actual_saturation_fraction')} for row in gate_stats]
    print(json.dumps({'status': report['status'], 'scientific_status': report['scientific_status'],
                      'selected_epoch': report['selected_epoch'], 'on_replay': replay,
                      'model_state_unchanged': True, 'input_files_unchanged': True,
                      'regions': compact_regions, 'gate_statistics': compact_gates,
                      'interpretation': report['interpretation']}, indent=2, allow_nan=False), flush=True)
    print(f'Saved gate probe: {output}', flush=True)
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
