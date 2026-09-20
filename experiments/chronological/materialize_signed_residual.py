"""Materialize frozen-A signed residuals for a hard-supported residual probe."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.chronological.audit_matched_controls import sha256
from experiments.chronological.materialize_incident_branch import (
    COHORTS, SPLITS, FullPositiveDataset, MatchedCounterfactualDataset, atomic_npz,
    load_protocol as load_v6a_protocol, verify_inputs,
)
from experiments.chronological.smoke import make_model
from experiments.chronological.train import configure_determinism
from src.utils.chronological import ChronologicalDataset


def load_protocol(path, base_protocol_path):
    protocol = json.loads(Path(path).read_text(encoding='utf-8'))
    if (protocol.get('protocol_id') != 'contra_v8_signed_residual_materialize_v7a' or
            protocol.get('scope') !=
            'frozen_A_incident_on_signed_residual_materialization' or
            protocol.get('main_training_ready') is not False):
        raise ValueError('v7a protocol identity or scope changed')
    if [protocol.get(key) for key in ('source_year', 'source_version',
                                      'expected_sensor_count')] != [2023, 8, 496]:
        raise ValueError('v7a is frozen to the 2023 source-v8 Contra496 package')
    if protocol.get('expected_common_triples') != {'train': 3106, 'val': 618}:
        raise ValueError('v7a common-triple cohort changed')
    if protocol.get('expected_positive_samples') != {'train': 3604, 'val': 917}:
        raise ValueError('v7a full-positive cohort changed')
    base = protocol.get('base_input_contract', {})
    if (base.get('protocol_id') != 'contra_v8_incident_branch_materialize_v6a' or
            base.get('protocol_sha256') != sha256(base_protocol_path)):
        raise ValueError('v7a base input contract changed')
    base_protocol = load_v6a_protocol(base_protocol_path)
    if (protocol.get('checkpoint') != base_protocol.get('checkpoint') or
            protocol['expected_common_triples'] !=
            base_protocol['expected_common_triples'] or
            protocol['expected_positive_samples'] !=
            base_protocol['expected_positive_samples']):
        raise ValueError('v7a checkpoint or cohort differs from the v6a input contract')
    inference = protocol.get('inference', {})
    if (inference.get('mode') != 'incident_on' or
            inference.get('signed_residual_definition') !=
            'target_minus_frozen_A_prediction' or
            inference.get('baseline_prediction_available_at_inference') is not True or
            inference.get('gradient_computation_prohibited') is not True or
            inference.get('optimizer_step_prohibited') is not True):
        raise ValueError('v7a inference estimand changed')
    support = protocol.get('support', {})
    if (support.get('candidate_source') !=
            'paired_positive_report_location_distances_nonzero' or
            support.get('stored_horizons_zero_based_half_open') != [0, 6] or
            support.get('downstream_correction_support') !=
            'candidate_nodes_H1_H6_only' or
            support.get('protected_horizons_zero_based_half_open') != [6, 12] or
            support.get('protected_noncandidate_nodes') is not True):
        raise ValueError('v7a correction support changed')
    expected_arrays = [
        'signed_residual', 'baseline_prediction', 'valid', 'candidate_mask',
        'positive_sample_index', 'baseline_all_absolute_sum',
        'baseline_all_valid_count',
    ]
    output = protocol.get('output_schema', {})
    if (output.get('cohorts') != list(COHORTS) or
            output.get('full_positive_cohort') != 'incident_full' or
            output.get('stored_arrays') != expected_arrays or
            output.get('early_array_shape') != 'samples_by_6_by_496_by_1' or
            output.get('float_dtype') != 'float32' or
            output.get('sum_dtype') != 'float64' or
            output.get('count_dtype') != 'int64'):
        raise ValueError('v7a output schema changed')
    boundary = protocol.get('information_boundary', {})
    required = (
        'train_and_validation_only', 'test_split_prohibited', 'checkpoint_frozen',
        'model_training_prohibited', 'common_triples_may_not_change',
        'future_Y_may_not_change_inputs_or_cohort',
        'future_Y_used_only_to_form_signed_residual',
    )
    if not all(boundary.get(key) is True for key in required):
        raise ValueError('v7a information boundary changed')
    if not isinstance(protocol.get('batch_size'), int) or protocol['batch_size'] < 1:
        raise ValueError('v7a batch size must be positive')
    return protocol, base_protocol


class ResidualTotals:
    def __init__(self):
        self.all_absolute = 0.0
        self.all_count = 0
        self.candidate_early_absolute = 0.0
        self.candidate_early_count = 0

    def update(self, residual, valid, candidate):
        absolute = residual.double().abs()
        active = valid[:, :6] & candidate[:, None, :, None]
        self.all_absolute += float(absolute[valid].sum())
        self.all_count += int(valid.sum())
        self.candidate_early_absolute += float(absolute[:, :6][active].sum())
        self.candidate_early_count += int(active.sum())

    def result(self):
        if self.all_count < 1 or self.candidate_early_count < 1:
            raise ValueError('v7a diagnostic population is empty')
        return {
            'all': {
                'valid_cells': self.all_count,
                'mae_A': self.all_absolute / self.all_count,
            },
            'candidate_h1_h6': {
                'valid_cells': self.candidate_early_count,
                'mae_A': self.candidate_early_absolute / self.candidate_early_count,
            },
        }


def materialize_cohort(model, dataset, batch_size, device, output_path):
    keys = (
        'signed_residual', 'baseline_prediction', 'valid', 'candidate_mask',
        'positive_sample_index', 'baseline_all_absolute_sum',
        'baseline_all_valid_count',
    )
    collected = {key: [] for key in keys}
    totals = ResidualTotals()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            x = batch['x'].to(device)
            incident = {key: value.to(device) for key, value in batch['incident'].items()}
            target = batch['y_flow'].to(device)
            valid = batch['y_valid'].to(device).bool()
            candidate = batch['candidate_mask'].to(device).bool()
            prediction = model(x, incident_data=incident) * dataset.scaler['std'] + \
                dataset.scaler['mean']
            if not torch.isfinite(prediction).all():
                raise ValueError('Frozen-A residual inference produced non-finite predictions')
            residual = target - prediction
            if not torch.isfinite(residual[valid]).all():
                raise ValueError('Frozen-A valid signed residual is non-finite')
            totals.update(residual, valid, candidate)
            valid_residual = torch.where(valid, residual, torch.zeros_like(residual))
            absolute = valid_residual.double().abs()
            collected['signed_residual'].append(
                valid_residual[:, :6].cpu().float().numpy())
            collected['baseline_prediction'].append(
                prediction[:, :6].cpu().float().numpy())
            collected['valid'].append(valid[:, :6].cpu().numpy())
            collected['candidate_mask'].append(candidate.cpu().numpy())
            collected['positive_sample_index'].append(
                batch['positive_sample_index'].numpy())
            collected['baseline_all_absolute_sum'].append(
                absolute.sum(dim=(1, 2, 3)).cpu().numpy())
            collected['baseline_all_valid_count'].append(
                valid.sum(dim=(1, 2, 3)).cpu().numpy().astype(np.int64, copy=False))
    arrays = {key: np.concatenate(values, axis=0) for key, values in collected.items()}
    early_shape = (len(dataset), 6, len(dataset.station_ids), 1)
    if (arrays['signed_residual'].shape != early_shape or
            arrays['baseline_prediction'].shape != early_shape or
            arrays['valid'].shape != early_shape or
            arrays['candidate_mask'].shape != (len(dataset), len(dataset.station_ids)) or
            arrays['positive_sample_index'].shape != (len(dataset),) or
            arrays['baseline_all_absolute_sum'].shape != (len(dataset),) or
            arrays['baseline_all_valid_count'].shape != (len(dataset),) or
            arrays['signed_residual'].dtype != np.float32 or
            arrays['baseline_prediction'].dtype != np.float32 or
            arrays['valid'].dtype != np.bool_ or
            arrays['candidate_mask'].dtype != np.bool_ or
            arrays['baseline_all_absolute_sum'].dtype != np.float64 or
            arrays['baseline_all_valid_count'].dtype != np.int64):
        raise ValueError('v7a output arrays have unexpected shapes or dtypes')
    atomic_npz(output_path, **arrays)
    return totals.result()


def run(data_dir, primary_dir, secondary_dir, checkpoint, output, protocol_path,
        base_protocol_path, device_name, batch_size=None, check=False):
    output = Path(output)
    if output.exists():
        raise FileExistsError('v7a output exists; preserve it and use a new directory')
    protocol, base_protocol = load_protocol(protocol_path, base_protocol_path)
    input_hashes = verify_inputs(
        data_dir, primary_dir, secondary_dir, checkpoint, base_protocol)
    device = torch.device(device_name)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but not available')
    configure_determinism(device)
    torch.set_num_threads(max(1, int(os.environ.get('OMP_NUM_THREADS', '3'))))
    positive_train = ChronologicalDataset(data_dir, 'train')
    model = make_model(
        Path(data_dir), len(positive_train.station_ids), device, 'fixed')
    state = torch.load(Path(checkpoint), map_location='cpu', weights_only=True)
    if not isinstance(state, dict) or not state or not all(
            isinstance(key, str) and torch.is_tensor(value) for key, value in state.items()):
        raise ValueError('Expected a plain tensor state_dict checkpoint')
    model.load_state_dict(state, strict=True)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    if parameters != protocol['checkpoint']['parameters']:
        raise ValueError('Fixed-A parameter count differs from the v7a protocol')
    del positive_train, state
    actual_batch_size = batch_size or int(protocol['batch_size'])
    sample_limit = 2 if check else None
    if check:
        actual_batch_size = min(actual_batch_size, sample_limit)
    if actual_batch_size < 1:
        raise ValueError('Batch size must be positive')
    output.mkdir(parents=True, exist_ok=False)
    results, output_files, mask_compatibility = {}, {}, {}
    for split in SPLITS:
        results[split] = {}
        full_dataset = FullPositiveDataset(
            data_dir, split, expected_count=protocol['expected_positive_samples'][split],
            expected_nodes=protocol['expected_sensor_count'], sample_limit=sample_limit)
        full_path = output / f'{split}_incident_full_signed_residuals.npz'
        results[split]['incident_full'] = materialize_cohort(
            model, full_dataset, actual_batch_size, device, full_path)
        output_files[full_path.name] = {
            'sha256': sha256(full_path), 'bytes': full_path.stat().st_size,
            'samples': len(full_dataset),
        }
        print(json.dumps({
            'split': split, 'cohort': 'incident_full', 'samples': len(full_dataset),
            'all_mae_A': results[split]['incident_full']['all']['mae_A'],
        }), flush=True)
        for cohort in COHORTS:
            dataset = MatchedCounterfactualDataset(
                data_dir, primary_dir, secondary_dir, split, cohort,
                expected_count=protocol['expected_common_triples'][split],
                expected_nodes=protocol['expected_sensor_count'],
                sample_limit=sample_limit,
                expected_frozen_only_pairs=base_protocol['candidate_mask_compatibility'][
                    'expected_frozen_only_pairs'][split])
            if split not in mask_compatibility:
                mask_compatibility[split] = dataset.candidate_mask_compatibility
            elif mask_compatibility[split] != dataset.candidate_mask_compatibility:
                raise ValueError('Candidate-mask compatibility differs across cohorts')
            path = output / f'{split}_{cohort}_signed_residuals.npz'
            results[split][cohort] = materialize_cohort(
                model, dataset, actual_batch_size, device, path)
            output_files[path.name] = {
                'sha256': sha256(path), 'bytes': path.stat().st_size,
                'samples': len(dataset),
            }
            print(json.dumps({
                'split': split, 'cohort': cohort, 'samples': len(dataset),
                'all_mae_A': results[split][cohort]['all']['mae_A'],
            }), flush=True)
    if not check:
        observed = results['val']['incident_full']['all']['mae_A']
        expected = float(protocol['checkpoint']['best_validation_mae'])
        tolerance = float(protocol['checkpoint']['reproduction_absolute_tolerance'])
        if abs(observed - expected) > tolerance:
            raise ValueError(
                f'Full fixed-A validation MAE reproduction differs by {abs(observed - expected)}')
    summary = {
        'status': ('SIGNED_RESIDUAL_ENGINEERING_CHECK_PASS' if check else
                   'SIGNED_RESIDUAL_MATERIALIZATION_COMPLETE'),
        'scope': protocol['scope'], 'protocol_id': protocol['protocol_id'],
        'protocol_sha256': sha256(protocol_path), 'main_training_ready': False,
        'model_training_performed': False, 'gradient_computation_performed': False,
        'test_split_read': False, 'common_triples_changed': False,
        'engineering_check': bool(check),
        'checkpoint': {
            **protocol['checkpoint'], 'path': str(Path(checkpoint).resolve()),
            'loaded_strictly': True,
        },
        'inference': protocol['inference'], 'support': protocol['support'],
        'candidate_mask_compatibility': mask_compatibility,
        'split_results': results,
        'inputs': {
            'positive_package': input_hashes[0],
            'primary_controls': input_hashes[1],
            'secondary_controls': input_hashes[2],
            'checkpoint_sha256': sha256(checkpoint),
            'base_protocol_sha256': sha256(base_protocol_path),
            'protocol_sha256': sha256(protocol_path),
            'code_sha256': sha256(__file__),
        },
        'outputs': output_files,
        'environment': {
            'python_version': sys.version, 'torch_version': torch.__version__,
            'numpy_version': np.__version__, 'device': str(device),
            'gpu_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
            'threads': torch.get_num_threads(),
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'tf32': bool(torch.backends.cuda.matmul.allow_tf32 or
                         torch.backends.cudnn.allow_tf32),
            'git_head': subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
        },
        'interpretation': protocol['interpretation'],
    }
    summary_path = output / 'summary.json'
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
        encoding='utf-8')
    print(f'Saved v7a signed residual materialization: {summary_path}', flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--primary-control-dir', type=Path, required=True)
    parser.add_argument('--secondary-control-dir', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, default=Path(__file__).with_name(
        'signed_residual_materialize_v7a.json'))
    parser.add_argument('--base-protocol', type=Path, default=Path(__file__).with_name(
        'incident_branch_materialize_v6a.json'))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    run(args.data_dir, args.primary_control_dir, args.secondary_control_dir,
        args.checkpoint, args.output, args.protocol, args.base_protocol,
        args.device, args.batch_size, args.check)


if __name__ == '__main__':
    main()
