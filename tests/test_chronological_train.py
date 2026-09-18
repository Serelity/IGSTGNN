import copy
import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import numpy as np

from experiments.chronological import train
from src.utils.chronological import flow_metrics


REPO = Path(__file__).resolve().parents[1]


class ChronologicalTrainTests(unittest.TestCase):
    @staticmethod
    def _write_tiny_package(directory):
        directory = Path(directory)
        stations = np.array([11, 22], dtype=np.int64)
        np.save(directory / 'station_ids.npy', stations)
        np.save(directory / 'adjacency.npy', np.array([[0, 1], [1, 0]], dtype=np.float32))
        rows = {
            'train': [
                {'sample_index': '1', 'split': 'train', 'source_version': '8',
                 'report_time': '2023-01-01T01:02:00', 't0': '2023-01-01T01:05:00',
                 'x_start': '2023-01-01T00:00:00'},
                {'sample_index': '2', 'split': 'train', 'source_version': '8',
                 'report_time': '2023-01-02T01:02:00', 't0': '2023-01-02T01:05:00',
                 'x_start': '2023-01-02T00:00:00'}],
            'val': [
                {'sample_index': '3', 'split': 'val', 'source_version': '8',
                 'report_time': '2023-09-01T01:02:00', 't0': '2023-09-01T01:05:00',
                 'x_start': '2023-09-01T00:00:00'},
                {'sample_index': '4', 'split': 'val', 'source_version': '8',
                 'report_time': '2023-09-02T01:02:00', 't0': '2023-09-02T01:05:00',
                 'x_start': '2023-09-02T00:00:00'}]}
        for split, split_rows in rows.items():
            with (directory / f'{split}_manifest.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(split_rows[0]))
                writer.writeheader()
                writer.writerows(split_rows)
            flow = np.arange(2 * 26 * 2, dtype=np.float32).reshape(2, 26, 2) + 10
            if split == 'val':
                flow += 3
            np.save(directory / f'{split}_flow.npy', flow)
            distances = np.zeros((2, 2, 3), dtype=np.float32)
            distances[..., 1] = 1
            dates = [row['t0'] for row in split_rows]
            tod = np.array([13, 13], dtype=np.int64)
            dow = np.array([0, 1] if split == 'train' else [5, 6], dtype=np.int64)
            np.savez(directory / f'{split}_context.npz', distances=distances,
                     report_age_minutes=np.array([3, 3], dtype=np.float32),
                     forecast_tod=tod, forecast_dow=dow,
                     sample_indices=np.array([int(row['sample_index']) for row in split_rows]),
                     station_ids=stations)
        scaler = {
            'source_version': 8,
            'fit_scope': 'train_X_0:12_unique_station_nominal_slot_finite_nonnegative',
            'station_ids': [11, 22], 'mean': 35., 'std': 15.,
            'node_fill_mean': [34., 36.], 'fitted_sample_indices': [1, 2]}
        (directory / 'scaler.json').write_text(json.dumps(scaler))

        def digest(path):
            return hashlib.sha256(Path(path).read_bytes()).hexdigest()

        raw_names = ['train_flow.npy', 'val_flow.npy', 'train_manifest.csv',
                     'val_manifest.csv', 'station_ids.npy', 'scaler.json']
        summary = {'source_version': 8, 'build_complete': True,
                   'files': {name: digest(directory / name) for name in raw_names}}
        (directory / 'summary.json').write_text(json.dumps(summary))
        context_names = ['train_context.npz', 'val_context.npz', 'adjacency.npy']
        context = {'schema': 'report_location_v1', 'scope': 'conditional_development',
                   'outputs': {name: digest(directory / name) for name in context_names}}
        (directory / 'context_manifest.json').write_text(json.dumps(context))
        protocol = {
            'protocol_id': 'tiny-screen-v1', 'scope': 'conditional_offline_screening',
            'main_training_ready': False,
            'data_summary_sha256': digest(directory / 'summary.json'),
            'context_summary_sha256': digest(directory / 'context_manifest.json'),
            'train_samples': 2, 'val_samples': 2, 'stations': 2, 'batch_size': 2,
            'max_epochs': 100, 'patience': 20, 'learning_rate': .002,
            'weight_decay': 1e-5, 'adam_eps': 1e-8, 'clip_grad_norm': 5.,
            'phase_learning_rate': .002, 'phase_weight_decay': 0.,
            'lr_milestones': [1, 38], 'lr_gamma': .5, 'supervised_horizons': 12,
            'selection_metric': 'all_nodes.mae_macro', 'min_delta': 0.,
            'first_screen_seed': 2025, 'free_test_session_hours': 6.}
        protocol_path = directory / 'protocol.json'
        protocol_path.write_text(json.dumps(protocol))
        return protocol_path

    def test_cli_exposes_resume_and_bounded_pause(self):
        result = subprocess.run(
            [sys.executable, 'experiments/chronological/train.py', '--help'],
            cwd=REPO, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--resume', result.stdout)
        self.assertIn('--stop-after-epoch', result.stdout)
        self.assertIn('--check', result.stdout)
        self.assertIn('phase', result.stdout)
        self.assertIn('phase_residual', result.stdout)

    def test_cuda_determinism_requires_cublas_workspace_configuration(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop('CUBLAS_WORKSPACE_CONFIG', None)
            with self.assertRaisesRegex(RuntimeError, 'CUBLAS_WORKSPACE_CONFIG'):
                train.configure_determinism(torch.device('cuda:0'))
        with patch.dict(os.environ, {'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}):
            train.configure_determinism(torch.device('cuda:0'))
            self.assertTrue(torch.are_deterministic_algorithms_enabled())

    def test_epoch_plan_is_paired_complete_and_keeps_tail(self):
        make_plan = getattr(train, 'epoch_plan', lambda *args, **kwargs: None)
        first = make_plan([101, 102, 103, 104, 105], batch_size=2, seed=7, epoch=1)
        self.assertIsNotNone(first)
        self.assertEqual(first['order'], [101, 103, 104, 105, 102])
        self.assertEqual(first['batches'], [[101, 103], [104, 105], [102]])
        self.assertEqual(first['tail_batch_size'], 1)
        self.assertEqual(first, make_plan([101, 102, 103, 104, 105], 2, 7, 1))
        second = make_plan([101, 102, 103, 104, 105], 2, 7, 2)
        self.assertEqual(second['order'], [102, 105, 101, 103, 104])
        self.assertEqual(sorted(second['order']), [101, 102, 103, 104, 105])

    def test_protocol_rejects_scope_or_package_drift(self):
        validate = getattr(train, 'validate_protocol', None)
        self.assertIsNotNone(validate)
        protocol = {
            'protocol_id': 'screen-v1', 'scope': 'conditional_offline_screening',
            'main_training_ready': False, 'data_summary_sha256': 'raw-ok',
            'context_summary_sha256': 'context-ok', 'train_samples': 5,
            'val_samples': 3, 'stations': 2, 'batch_size': 2, 'max_epochs': 2,
            'patience': 2, 'learning_rate': .002, 'weight_decay': 1e-5,
            'adam_eps': 1e-8, 'clip_grad_norm': 5., 'lr_milestones': [1],
            'lr_gamma': .5, 'supervised_horizons': 12,
            'selection_metric': 'all_nodes.mae_macro', 'min_delta': 0.,
            'first_screen_seed': 2025, 'free_test_session_hours': 6.
        }
        package = {'summary.json': 'raw-ok', 'context_manifest.json': 'context-ok'}
        self.assertEqual(validate(protocol, package, 5, 3, 2)['protocol_id'], 'screen-v1')
        changed = dict(protocol, main_training_ready=True)
        with self.assertRaisesRegex(ValueError, 'conditional'):
            validate(changed, package, 5, 3, 2)
        with self.assertRaisesRegex(ValueError, 'fingerprint'):
            validate(protocol, dict(package, **{'summary.json': 'other'}), 5, 3, 2)
        with self.assertRaisesRegex(ValueError, 'shape'):
            validate(protocol, package, 6, 3, 2)
        residual = dict(protocol, candidate_variant='phase_residual',
                        phase_learning_rate=.002, phase_weight_decay=0.)
        self.assertEqual(validate(residual, package, 5, 3, 2)['phase_weight_decay'], 0.)
        with self.assertRaisesRegex(ValueError, 'weight decay'):
            validate(dict(residual, phase_weight_decay=1e-5), package, 5, 3, 2)

    def test_residual_phase_optimizer_uses_separate_zero_decay_group(self):
        from src.models.igstgnn import IGSTGNN

        torch.manual_seed(1234)
        model = IGSTGNN(
            model_args=dict(num_feat=1, num_hidden=8, node_hidden=4, time_emb_dim=4,
                            layer=5, k_s=2, k_t=3, tpd=288, dropout=0., gap=3,
                            adjs=[torch.eye(3)] * 2, incident_schema='report_location_v1',
                            time_response='phase_residual', use_sensor_info=False),
            node_num=3, input_dim=3, output_dim=1, seq_len=12, horizon=12,
            dataset='unused', data_path='/unused-report-schema-path')
        protocol = {'learning_rate': .002, 'phase_learning_rate': .004,
                    'weight_decay': 1e-5, 'phase_weight_decay': 0., 'adam_eps': 1e-8}
        optimizer = train.build_optimizer(model, protocol, 'phase_residual')
        self.assertEqual([group['group_name'] for group in optimizer.param_groups],
                         ['common', 'phase_response'])
        self.assertEqual([group['lr'] for group in optimizer.param_groups], [.002, .004])
        self.assertEqual([group['weight_decay'] for group in optimizer.param_groups], [1e-5, 0.])
        expected = {id(parameter) for name, parameter in model.named_parameters()
                    if name.startswith('tiid_module.time_response.')}
        actual = {id(parameter) for parameter in optimizer.param_groups[1]['params']}
        self.assertEqual(actual, expected)

    def test_metric_totals_matches_independent_float64_metrics(self):
        prediction = np.array(
            [[[[100_000_000.], [2.], [2.], [2.]],
              [[3.], [4.], [5.], [6.]]]], dtype=np.float32)
        target = np.ones_like(prediction)
        valid = np.ones_like(prediction, dtype=bool)
        expected = flow_metrics(prediction, target, valid)
        totals = train.MetricTotals(horizons=2)
        totals.update(torch.from_numpy(prediction[:, :, :2]),
                      torch.from_numpy(target[:, :, :2]),
                      torch.from_numpy(valid[:, :, :2]))
        totals.update(torch.from_numpy(prediction[:, :, 2:]),
                      torch.from_numpy(target[:, :, 2:]),
                      torch.from_numpy(valid[:, :, 2:]))
        actual = totals.result()
        self.assertEqual(actual['valid_count_per_horizon'], [4, 4])
        self.assertAlmostEqual(actual['mae_macro'], expected['mae_macro'], places=12)
        self.assertAlmostEqual(actual['rmse_macro'], expected['rmse_macro'], places=12)
        self.assertAlmostEqual(actual['mape_macro'], expected['mape_macro'], places=12)

    def test_epoch_checkpoint_resume_matches_uninterrupted_adam_and_scheduler(self):
        save = getattr(train, 'save_checkpoint', None)
        restore = getattr(train, 'restore_checkpoint', None)
        self.assertIsNotNone(save)
        self.assertIsNotNone(restore)

        def objects():
            model = torch.nn.Linear(1, 1, bias=False)
            with torch.no_grad():
                model.weight.fill_(1.)
            optimizer = torch.optim.Adam(model.parameters(), lr=.002, weight_decay=1e-5)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1], gamma=.5)
            return model, optimizer, scheduler

        def update(model, optimizer, scheduler):
            optimizer.zero_grad(set_to_none=True)
            (model(torch.tensor([[2.]])) - 1).abs().mean().backward()
            optimizer.step()
            scheduler.step()

        identity = {'variant': 'conditioned', 'seed': 2025, 'protocol_sha256': 'abc'}
        model, optimizer, scheduler = objects()
        update(model, optimizer, scheduler)
        self.assertEqual(optimizer.param_groups[0]['lr'], .001)
        payload = {
            'format_version': 1, 'identity': identity, 'completed_epoch': 1,
            'model_state': copy.deepcopy(model.state_dict()),
            'optimizer_state': copy.deepcopy(optimizer.state_dict()),
            'scheduler_state': copy.deepcopy(scheduler.state_dict()),
            'best_model_state': copy.deepcopy(model.state_dict()),
            'best_epoch': 1, 'best_metric': 3., 'wait': 0,
            'global_updates': 1, 'history': [{'epoch': 1}]
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'last_checkpoint.pt'
            save(path, payload)
            self.assertTrue(path.exists())
            self.assertFalse(path.with_suffix(path.suffix + '.partial').exists())
            resumed_model, resumed_optimizer, resumed_scheduler = objects()
            restored = restore(path, identity, resumed_model, resumed_optimizer, resumed_scheduler)
            self.assertEqual(restored['completed_epoch'], 1)
            self.assertEqual(resumed_optimizer.param_groups[0]['lr'], .001)
            update(model, optimizer, scheduler)
            update(resumed_model, resumed_optimizer, resumed_scheduler)
            torch.testing.assert_close(resumed_model.weight, model.weight, rtol=0, atol=0)
            self.assertEqual(resumed_optimizer.state_dict(), optimizer.state_dict())
            self.assertEqual(resumed_scheduler.state_dict(), scheduler.state_dict())
            with self.assertRaisesRegex(ValueError, 'identity'):
                restore(path, dict(identity, seed=2026), *objects())

    def test_real_model_check_resume_matches_uninterrupted(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            data = base / 'data'
            data.mkdir()
            protocol = self._write_tiny_package(data)

            def run(output, *extra):
                command = [sys.executable, 'experiments/chronological/train.py',
                           '--data-dir', str(data), '--output-dir', str(output),
                           '--protocol', str(protocol), '--variant', 'phase_residual',
                           '--device', 'cpu', '--seed', '2025', '--check', *extra]
                result = subprocess.run(command, cwd=REPO, text=True, capture_output=True,
                                        env=dict(os.environ, OMP_NUM_THREADS='1'), timeout=90)
                self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
                self.assertTrue((output / 'summary.json').exists(), result.stdout)
                return json.loads((output / 'summary.json').read_text())

            uninterrupted = base / 'uninterrupted'
            direct = run(uninterrupted)
            self.assertEqual(direct['status'], 'ENGINEERING_CHECK_PASS')
            self.assertEqual(direct['completed_epoch'], 2)
            self.assertEqual(direct['initial_max_abs_difference_from_A'], 0.)
            self.assertEqual(direct['initial_phase_response_diagnostics'][
                'max_abs_response_difference_from_A'], 0.)
            self.assertGreater(direct['history'][0]['time_response_gradient_l1_before_clipping'][
                'tiid_module.time_response.phase_encoder.2.weight']['mean_l1'], 0.)
            self.assertGreater(direct['best_phase_response_diagnostics'][
                'event_fraction_different_from_A_gt_1e-6'], 0.)
            self.assertEqual(direct['selection_metric'], 'all_nodes.mae_macro')
            self.assertEqual(direct['best_metric'],
                             direct['best_validation_metrics']['all_nodes']['mae_macro'])
            self.assertIn('h7_h12_mae_macro',
                          direct['best_validation_metrics']['associated_nodes'])
            self.assertEqual(direct['environment']['device'], 'cpu')
            self.assertEqual(direct['environment']['threads'], 1)
            self.assertFalse(direct['environment']['tf32'])
            self.assertTrue(direct['environment']['deterministic_algorithms'])
            self.assertIn('python_version', direct['environment'])
            self.assertTrue((uninterrupted / 'best_model.pt').exists())
            self.assertTrue((uninterrupted / 'best_validation_predictions.npz').exists())
            paused = base / 'resumed'
            first = run(paused, '--stop-after-epoch', '1')
            self.assertEqual(first['status'], 'PAUSED_AT_EPOCH_BOUNDARY')
            self.assertEqual(first['completed_epoch'], 1)
            self.assertFalse((paused / 'best_validation_predictions.npz').exists())
            resumed = run(paused, '--resume')
            self.assertEqual(resumed['status'], 'ENGINEERING_CHECK_PASS')
            self.assertEqual(resumed['completed_epoch'], 2)
            self.assertEqual(resumed['history'], direct['history'])
            self.assertEqual(resumed['best_metric'], direct['best_metric'])
            self.assertEqual(resumed['best_validation_metrics'], direct['best_validation_metrics'])
            left = torch.load(uninterrupted / 'last_checkpoint.pt', map_location='cpu', weights_only=False)
            right = torch.load(paused / 'last_checkpoint.pt', map_location='cpu', weights_only=False)
            self.assertEqual(left['scheduler_state'], right['scheduler_state'])
            self.assertEqual(left['global_updates'], right['global_updates'])
            for key in left['model_state']:
                torch.testing.assert_close(left['model_state'][key], right['model_state'][key], rtol=0, atol=0)
            with np.load(uninterrupted / 'best_validation_predictions.npz') as left_prediction, \
                    np.load(paused / 'best_validation_predictions.npz') as right_prediction:
                self.assertEqual(set(left_prediction.files),
                                 {'prediction', 'target', 'valid', 'associated',
                                  'sample_indices', 'station_ids'})
                for key in left_prediction.files:
                    np.testing.assert_array_equal(left_prediction[key], right_prediction[key])


if __name__ == '__main__':
    unittest.main()
