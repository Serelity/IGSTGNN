"""Materialize same-checkpoint incident-branch on/off errors on matched triples."""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.chronological.audit_matched_controls import read_csv, sha256
from experiments.chronological.smoke import make_model, verify_package
from experiments.chronological.train import configure_determinism
from src.utils.chronological import ChronologicalDataset, encode_window


SPLITS = ('train', 'val')
COHORTS = ('incident', 'primary_control', 'secondary_control')


def load_protocol(path):
    protocol = json.loads(Path(path).read_text(encoding='utf-8'))
    if (protocol.get('scope') != 'frozen_A_incident_branch_counterfactual_materialization'
            or protocol.get('protocol_id') != 'contra_v8_incident_branch_materialize_v6a'
            or protocol.get('main_training_ready') is not False):
        raise ValueError('v6a protocol identity or scope changed')
    if [protocol.get(key) for key in ('source_year', 'source_version',
                                      'expected_sensor_count')] != [2023, 8, 496]:
        raise ValueError('v6a is frozen to the 2023 source-v8 Contra496 package')
    if protocol.get('expected_common_triples') != {'train': 3106, 'val': 618}:
        raise ValueError('v6a common-triple cohort changed')
    if protocol.get('expected_positive_samples') != {'train': 3604, 'val': 917}:
        raise ValueError('v6a full positive cohort changed')
    checkpoint = protocol.get('checkpoint', {})
    if (checkpoint.get('variant') != 'fixed' or checkpoint.get('seed') != 2025 or
            checkpoint.get('best_epoch') != 99 or checkpoint.get('parameters') != 443645 or
            checkpoint.get('best_validation_mae') != 22.686666155323852 or
            checkpoint.get('reproduction_absolute_tolerance') != 0.00001 or
            checkpoint.get('state_format') != 'plain_state_dict' or
            len(checkpoint.get('best_model_sha256', '')) != 64):
        raise ValueError('v6a fixed-A checkpoint identity changed')
    modes = protocol.get('counterfactual_modes', {})
    pseudo = modes.get('control_pseudo_event', {})
    if (modes.get('published_A_anchor') != 'incident_on for positive incident windows' or
            pseudo.get('forecast_clock') != 'from each matched control candidate_t0' or
            pseudo.get('report_age_and_distances') != 'from the paired positive incident'):
        raise ValueError('v6a counterfactual construction changed')
    compatibility = protocol.get('candidate_mask_compatibility', {})
    expected_pairs = compatibility.get('expected_frozen_only_pairs', {})
    if (compatibility.get('model_candidate_source') !=
            'paired_positive_report_location_distances_nonzero' or
            compatibility.get('frozen_control_mask_source') !=
            'same_freeway_direction_within_inclusive_10_miles' or
            compatibility.get('required_relationship') !=
            'model_connected_is_subset_of_frozen_control_mask' or
            expected_pairs != {
                'train': [{
                    'positive_sample_index': 1542,
                    'station_id': 402510,
                    'postmile_delta_miles': 10.0,
                    'reason': 'positive_10_mile_boundary_has_zero_normalized_similarity',
                }],
                'val': [],
            }):
        raise ValueError('v6a candidate-mask compatibility changed')
    boundary = protocol.get('information_boundary', {})
    required = (
        'train_and_validation_only', 'test_split_prohibited', 'checkpoint_frozen',
        'gradient_computation_prohibited', 'optimizer_step_prohibited',
        'common_triples_may_not_change', 'future_Y_may_not_change_inputs_or_cohort',
    )
    if not all(boundary.get(key) is True for key in required):
        raise ValueError('v6a information boundary changed')
    if (protocol.get('window') != {
            'steps': 26, 'X_slice': [0, 12], 'Y_slice': [14, 26],
            'early_horizons': [0, 6], 'late_horizons': [6, 12]} or
            protocol.get('output_schema', {}).get('cohorts') != list(COHORTS)):
        raise ValueError('v6a window or output schema changed')
    if protocol['output_schema'].get('full_positive_cohort') != 'incident_full':
        raise ValueError('v6a full positive output schema changed')
    if not isinstance(protocol.get('batch_size'), int) or protocol['batch_size'] < 1:
        raise ValueError('v6a batch size must be positive')
    return protocol


def _verify_hash_group(directory, specification):
    excluded = {'protocol_id'}
    actual = {
        name: sha256(Path(directory) / name)
        for name in specification if name not in excluded
    }
    expected = {key: value for key, value in specification.items() if key not in excluded}
    if actual != expected:
        raise ValueError(f'Input fingerprints differ in {Path(directory)}')
    return actual


def verify_inputs(data_dir, primary_dir, secondary_dir, checkpoint, protocol):
    package = verify_package(Path(data_dir))
    positive_summary = json.loads(
        (Path(data_dir) / 'summary.json').read_text(encoding='utf-8'))
    if (positive_summary.get('test_flow_built') is not False or
            positive_summary.get('split_counts', {}).get('test') is not None):
        raise ValueError('Positive package violates the development-only boundary')
    expected_package = protocol['positive_package']
    if (package.get('summary.json') != expected_package['summary_sha256'] or
            package.get('context_manifest.json') !=
            expected_package['context_manifest_sha256']):
        raise ValueError('Positive package differs from the v6a protocol')
    primary = _verify_hash_group(primary_dir, protocol['primary_control_inputs'])
    secondary = _verify_hash_group(secondary_dir, protocol['secondary_control_inputs'])
    if sha256(checkpoint) != protocol['checkpoint']['best_model_sha256']:
        raise ValueError('Fixed-A checkpoint differs from the v6a protocol')
    for directory, specification, status in (
            (primary_dir, protocol['primary_control_inputs'],
             'MATCHED_NONINCIDENT_MATERIALIZATION_PASS'),
            (secondary_dir, protocol['secondary_control_inputs'],
             'SECOND_MATCHED_CONTROL_MATERIALIZATION_PASS')):
        summary = json.loads((Path(directory) / 'summary.json').read_text(encoding='utf-8'))
        if (summary.get('protocol_id') != specification['protocol_id'] or
                summary.get('status') != status or summary.get('test_split_read') is not False):
            raise ValueError('Control materialization identity or test boundary changed')
    return package, primary, secondary


def forecast_clock(t0):
    timestamp = datetime.fromisoformat(t0)
    return np.int64(timestamp.hour * 12 + timestamp.minute // 5), np.int64(
        (timestamp.weekday() + 1) % 7)


class MatchedCounterfactualDataset(Dataset):
    """One common-triple cohort with pseudo-event inputs fixed before Y inspection."""

    def __init__(self, data_dir, primary_dir, secondary_dir, split, cohort,
                 expected_count=None, expected_nodes=None, sample_limit=None,
                 expected_frozen_only_pairs=None):
        if split not in SPLITS or cohort not in COHORTS:
            raise ValueError('Unknown split or matched cohort')
        self.split, self.cohort = split, cohort
        self.positive = ChronologicalDataset(data_dir, split)
        self.scaler = self.positive.scaler
        self.station_ids = self.positive.station_ids
        self.positive_positions = {
            int(row['sample_index']): index for index, row in enumerate(self.positive.rows)
        }
        if len(self.positive_positions) != len(self.positive.rows):
            raise ValueError('Positive manifest contains duplicate sample indices')
        self.primary_rows = read_csv(Path(primary_dir) / f'{split}_control_manifest.csv')
        self.primary_by_sample = {
            int(row['positive_sample_index']): row for row in self.primary_rows
        }
        self.secondary_rows = read_csv(
            Path(secondary_dir) / f'{split}_second_control_manifest.csv')
        self.primary_flow = np.load(
            Path(primary_dir) / f'{split}_control_flow.npy', mmap_mode='r', allow_pickle=False)
        self.secondary_flow = np.load(
            Path(secondary_dir) / f'{split}_second_control_flow.npy', mmap_mode='r',
            allow_pickle=False)
        self.primary_masks = np.load(
            Path(primary_dir) / f'{split}_affected_mask.npy', allow_pickle=False)
        self.secondary_masks = np.load(
            Path(secondary_dir) / f'{split}_second_affected_mask.npy', allow_pickle=False)
        if expected_count is not None and len(self.secondary_rows) != expected_count:
            raise ValueError('Common-triple count differs from the v6a protocol')
        node_count = len(self.station_ids)
        if expected_nodes is not None and node_count != expected_nodes:
            raise ValueError('Node count differs from the v6a protocol')
        if (self.primary_flow.shape[1:] != (26, node_count) or
                self.secondary_flow.shape != (len(self.secondary_rows), 26, node_count) or
                self.primary_masks.shape != (len(self.primary_rows), node_count) or
                self.secondary_masks.shape != (len(self.secondary_rows), node_count) or
                self.primary_masks.dtype != np.bool_ or
                self.secondary_masks.dtype != np.bool_):
            raise ValueError('Matched-control arrays have inconsistent shapes or dtypes')
        self._validate_rows(expected_frozen_only_pairs or [])
        if sample_limit is not None:
            if not isinstance(sample_limit, int) or not 0 < sample_limit <= len(
                    self.secondary_rows):
                raise ValueError('Sample limit must retain at least one common triple')
            self.secondary_rows = self.secondary_rows[:sample_limit]

    def _validate_rows(self, expected_frozen_only_pairs):
        seen = set()
        frozen_only_pairs = []
        for position, row in enumerate(self.secondary_rows):
            sample = int(row['positive_sample_index'])
            if (int(row['control_index']) != position or sample in seen or
                    sample not in self.positive_positions or sample not in self.primary_by_sample):
                raise ValueError('Secondary common-triple order or identity is inconsistent')
            seen.add(sample)
            primary = self.primary_by_sample[sample]
            positive = self.positive.rows[self.positive_positions[sample]]
            primary_index = int(primary['control_index'])
            if (row['split'] != self.split or primary['split'] != self.split or
                    positive['split'] != self.split or
                    len({row['incident_id'], primary['incident_id'],
                         positive['incident_id']}) != 1 or
                    len({row['positive_t0'], primary['positive_t0'], positive['t0']}) != 1 or
                    not np.array_equal(self.primary_masks[primary_index],
                                       self.secondary_masks[position])):
                raise ValueError('Matched triple identities or candidate masks differ')
            context_mask = np.any(
                self.positive.context['distances'][self.positive_positions[sample]] != 0,
                axis=-1)
            frozen_mask = self.primary_masks[primary_index]
            outside = np.flatnonzero(context_mask & ~frozen_mask)
            if outside.size:
                raise ValueError(
                    'Model-connected candidate lies outside the frozen affected mask')
            for node_index in np.flatnonzero(frozen_mask & ~context_mask):
                frozen_only_pairs.append({
                    'positive_sample_index': sample,
                    'station_id': int(self.station_ids[node_index]),
                })
        expected = [{
            'positive_sample_index': int(item['positive_sample_index']),
            'station_id': int(item['station_id']),
        } for item in expected_frozen_only_pairs]
        if frozen_only_pairs != expected:
            raise ValueError('Frozen-only candidate-mask boundary cases changed')
        self.candidate_mask_compatibility = {
            'model_connected_outside_frozen_node_references': 0,
            'frozen_only_node_references': len(frozen_only_pairs),
            'frozen_only_pairs': frozen_only_pairs,
        }

    def __len__(self):
        return len(self.secondary_rows)

    def __getitem__(self, index):
        secondary = self.secondary_rows[index]
        sample = int(secondary['positive_sample_index'])
        positive_position = self.positive_positions[sample]
        positive_row = self.positive.rows[positive_position]
        primary = self.primary_by_sample[sample]
        if self.cohort == 'incident':
            raw, row = self.positive.flow[positive_position], positive_row
            source_index = positive_position
        elif self.cohort == 'primary_control':
            source_index = int(primary['control_index'])
            raw, row = self.primary_flow[source_index], primary
        else:
            source_index = index
            raw, row = self.secondary_flow[source_index], secondary
        x, y, x_valid, y_valid = encode_window(
            raw, datetime.fromisoformat(row['x_start']), self.scaler)
        forecast_tod, forecast_dow = forecast_clock(
            positive_row['t0'] if self.cohort == 'incident' else row['candidate_t0'])
        distances = self.positive.context['distances'][positive_position].astype(
            np.float32, copy=True)
        return {
            'x': x, 'y_flow': y, 'x_valid': x_valid, 'y_valid': y_valid,
            'incident': {
                'report_age_minutes': np.float32(
                    self.positive.context['report_age_minutes'][positive_position]),
                'forecast_tod': forecast_tod, 'forecast_dow': forecast_dow,
                'distances': distances,
            },
            'candidate_mask': np.any(distances != 0, axis=-1),
            'positive_sample_index': np.int64(sample),
            'source_index': np.int64(source_index),
        }


class FullPositiveDataset(Dataset):
    """All incident-centered samples, used for the full validation non-inferiority gate."""

    def __init__(self, data_dir, split, expected_count=None, expected_nodes=None,
                 sample_limit=None):
        self.dataset = ChronologicalDataset(data_dir, split)
        self.scaler = self.dataset.scaler
        self.station_ids = self.dataset.station_ids
        if expected_count is not None and len(self.dataset) != expected_count:
            raise ValueError('Full positive count differs from the v6a protocol')
        if expected_nodes is not None and len(self.station_ids) != expected_nodes:
            raise ValueError('Full positive node count differs from the v6a protocol')
        self.length = len(self.dataset)
        if sample_limit is not None:
            if not isinstance(sample_limit, int) or not 0 < sample_limit <= self.length:
                raise ValueError('Sample limit must retain at least one positive incident')
            self.length = sample_limit

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        sample = self.dataset[index]
        distances = sample['incident']['distances']
        sample['candidate_mask'] = np.any(distances != 0, axis=-1)
        sample['positive_sample_index'] = np.int64(
            self.dataset.rows[index]['sample_index'])
        sample['source_index'] = np.int64(index)
        return sample


class BranchTotals:
    def __init__(self):
        names = ('all', 'candidate_h1_h6', 'candidate_h7_h12', 'noncandidate')
        self.values = {
            name: {'count': 0, 'on_absolute': 0., 'off_absolute': 0.,
                   'prediction_delta_absolute': 0., 'prediction_delta_max': 0.}
            for name in names
        }

    def update(self, prediction_on, prediction_off, target, valid, candidate):
        error_on = (prediction_on.double() - target.double()).abs()
        error_off = (prediction_off.double() - target.double()).abs()
        delta = (prediction_on.double() - prediction_off.double()).abs()
        candidate_cells = candidate[:, None, :, None].expand_as(valid)
        early = torch.zeros_like(valid)
        early[:, :6] = True
        selections = {
            'all': valid,
            'candidate_h1_h6': valid & candidate_cells & early,
            'candidate_h7_h12': valid & candidate_cells & ~early,
            'noncandidate': valid & ~candidate_cells,
        }
        for name, selected in selections.items():
            if not selected.any():
                continue
            current = self.values[name]
            current['count'] += int(selected.sum())
            current['on_absolute'] += float(error_on[selected].sum())
            current['off_absolute'] += float(error_off[selected].sum())
            current['prediction_delta_absolute'] += float(delta[selected].sum())
            current['prediction_delta_max'] = max(
                current['prediction_delta_max'], float(delta[selected].max()))

    def result(self):
        result = {}
        for name, values in self.values.items():
            count = values['count']
            if count < 1:
                raise ValueError(f'Empty diagnostic population: {name}')
            result[name] = {
                'valid_cells': count,
                'mae_on': values['on_absolute'] / count,
                'mae_off': values['off_absolute'] / count,
                'activation_gain_off_minus_on':
                    (values['off_absolute'] - values['on_absolute']) / count,
                'prediction_delta_abs_mean': values['prediction_delta_absolute'] / count,
                'prediction_delta_abs_max': values['prediction_delta_max'],
            }
        return result


def atomic_npz(path, **arrays):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.partial')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def materialize_cohort(model, dataset, batch_size, device, output_path):
    collected = {key: [] for key in (
        'absolute_error_on', 'absolute_error_off', 'valid', 'candidate_mask',
        'positive_sample_index')}
    totals = BranchTotals()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            x = batch['x'].to(device)
            incident = {key: value.to(device) for key, value in batch['incident'].items()}
            target = batch['y_flow'].to(device)
            valid = batch['y_valid'].to(device).bool()
            candidate = batch['candidate_mask'].to(device).bool()
            prediction_on = model(x, incident_data=incident) * dataset.scaler['std'] + \
                dataset.scaler['mean']
            prediction_off = model(x, incident_data=None) * dataset.scaler['std'] + \
                dataset.scaler['mean']
            if (not torch.isfinite(prediction_on).all() or
                    not torch.isfinite(prediction_off).all()):
                raise ValueError('Frozen branch inference produced non-finite predictions')
            totals.update(prediction_on, prediction_off, target, valid, candidate)
            error_on = torch.where(
                valid, (prediction_on - target).abs(), torch.zeros_like(prediction_on))
            error_off = torch.where(
                valid, (prediction_off - target).abs(), torch.zeros_like(prediction_off))
            collected['absolute_error_on'].append(error_on.cpu().float().numpy())
            collected['absolute_error_off'].append(error_off.cpu().float().numpy())
            collected['valid'].append(valid.cpu().numpy())
            collected['candidate_mask'].append(candidate.cpu().numpy())
            collected['positive_sample_index'].append(
                batch['positive_sample_index'].numpy())
    arrays = {key: np.concatenate(values, axis=0) for key, values in collected.items()}
    expected_shape = (len(dataset), 12, len(dataset.station_ids), 1)
    if (arrays['absolute_error_on'].shape != expected_shape or
            arrays['absolute_error_off'].shape != expected_shape or
            arrays['valid'].shape != expected_shape or
            arrays['candidate_mask'].shape != expected_shape[::2] or
            arrays['absolute_error_on'].dtype != np.float32 or
            arrays['absolute_error_off'].dtype != np.float32 or
            arrays['valid'].dtype != np.bool_ or
            arrays['candidate_mask'].dtype != np.bool_):
        raise ValueError('v6a output arrays have unexpected shapes or dtypes')
    atomic_npz(output_path, **arrays)
    return totals.result()


def run(data_dir, primary_dir, secondary_dir, checkpoint, output, protocol_path,
        device_name, batch_size=None, check=False):
    output = Path(output)
    if output.exists():
        raise FileExistsError('v6a output exists; preserve it and use a new directory')
    protocol = load_protocol(protocol_path)
    input_hashes = verify_inputs(
        data_dir, primary_dir, secondary_dir, checkpoint, protocol)
    device = torch.device(device_name)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but not available')
    configure_determinism(device)
    threads = max(1, int(os.environ.get('OMP_NUM_THREADS', '3')))
    torch.set_num_threads(threads)
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
        raise ValueError('Fixed-A parameter count differs from the v6a protocol')
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
        full_path = output / f'{split}_incident_full_branch_errors.npz'
        results[split]['incident_full'] = materialize_cohort(
            model, full_dataset, actual_batch_size, device, full_path)
        output_files[full_path.name] = {
            'sha256': sha256(full_path), 'bytes': full_path.stat().st_size,
            'samples': len(full_dataset),
        }
        print(json.dumps({
            'split': split, 'cohort': 'incident_full', 'samples': len(full_dataset),
            'all_mae_on': results[split]['incident_full']['all']['mae_on'],
            'all_mae_off': results[split]['incident_full']['all']['mae_off'],
        }), flush=True)
        for cohort in COHORTS:
            dataset = MatchedCounterfactualDataset(
                data_dir, primary_dir, secondary_dir, split, cohort,
                expected_count=protocol['expected_common_triples'][split],
                expected_nodes=protocol['expected_sensor_count'],
                sample_limit=sample_limit,
                expected_frozen_only_pairs=protocol['candidate_mask_compatibility'][
                    'expected_frozen_only_pairs'][split])
            if split not in mask_compatibility:
                mask_compatibility[split] = dataset.candidate_mask_compatibility
            elif mask_compatibility[split] != dataset.candidate_mask_compatibility:
                raise ValueError('Candidate-mask compatibility differs across cohorts')
            path = output / f'{split}_{cohort}_branch_errors.npz'
            results[split][cohort] = materialize_cohort(
                model, dataset, actual_batch_size, device, path)
            output_files[path.name] = {
                'sha256': sha256(path), 'bytes': path.stat().st_size,
                'samples': len(dataset),
            }
            print(json.dumps({
                'split': split, 'cohort': cohort, 'samples': len(dataset),
                'all_mae_on': results[split][cohort]['all']['mae_on'],
                'all_mae_off': results[split][cohort]['all']['mae_off'],
            }), flush=True)
    if not check:
        observed = results['val']['incident_full']['all']['mae_on']
        expected = float(protocol['checkpoint']['best_validation_mae'])
        tolerance = float(protocol['checkpoint']['reproduction_absolute_tolerance'])
        if abs(observed - expected) > tolerance:
            raise ValueError(
                f'Full fixed-A validation MAE reproduction differs by {abs(observed - expected)}')
    summary = {
        'status': ('ENGINEERING_CHECK_PASS' if check else
                   'INCIDENT_BRANCH_MATERIALIZATION_COMPLETE'),
        'scope': protocol['scope'], 'protocol_id': protocol['protocol_id'],
        'protocol_sha256': sha256(protocol_path), 'main_training_ready': False,
        'model_training_performed': False, 'gradient_computation_performed': False,
        'test_split_read': False, 'common_triples_changed': False,
        'engineering_check': bool(check),
        'checkpoint': {
            **protocol['checkpoint'], 'path': str(Path(checkpoint).resolve()),
            'loaded_strictly': True,
        },
        'counterfactual_modes': protocol['counterfactual_modes'],
        'candidate_mask_compatibility': mask_compatibility,
        'split_results': results,
        'inputs': {
            'positive_package': input_hashes[0],
            'primary_controls': input_hashes[1],
            'secondary_controls': input_hashes[2],
            'checkpoint_sha256': sha256(checkpoint),
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
    print(f'Saved v6a branch materialization: {summary_path}', flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--primary-control-dir', type=Path, required=True)
    parser.add_argument('--secondary-control-dir', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, default=Path(__file__).with_name(
        'incident_branch_materialize_v6a.json'))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    run(args.data_dir, args.primary_control_dir, args.secondary_control_dir,
        args.checkpoint, args.output, args.protocol, args.device, args.batch_size,
        args.check)


if __name__ == '__main__':
    main()
