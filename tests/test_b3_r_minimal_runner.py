"""Meaningful contracts for pruned training, fresh identity and exact resume."""
import copy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.chronological import train_b3_r as base
from experiments.chronological import train_b3_r_minimal as training
from experiments.chronological import train_internal_spatial as legacy
from src.models.incident_spatial import configure_spatial_mode
from src.models.incident_spatial_minimal import from_residual
from src.models.incident_spatial_residual import from_canonical_b3
from tests.test_architecture_mechanisms import tiny_model
from tests.test_incident_spatial import graph
from tests.test_internal_spatial_training import Dataset


PROTOCOL = training.HERE/'internal_spatial_b3_r_minimal_v1.json'


def clock_dataset():
    dataset = Dataset()
    for sample in dataset.samples:
        incident = sample['incident']
        forecast_slot, day = int(incident['forecast_tod']), int(incident['forecast_dow'])
        for step in range(12):
            absolute = forecast_slot-13+step
            sample['x'][step, :, 1] = (absolute % 288)/288
            sample['x'][step, :, 2] = ((day+absolute//288) % 7)/7
    return dataset


def tiny_reference(args, arm, nodes, g, device):
    if arm != 'B3R':
        raise AssertionError('Expected fresh B3R reference')
    canonical = configure_spatial_mode(tiny_model().requires_grad_(True), 'state_restricted', copy.deepcopy(g))
    return from_canonical_b3(canonical)[0].to(device)


def tiny_candidate(args, arm, nodes, g, device):
    if arm != training.ARM:
        raise AssertionError('Expected minimal arm')
    return from_residual(tiny_reference(args, 'B3R', nodes, g, device))[0]


def worker_args(root, epochs=1):
    return SimpleNamespace(worker=training.ARM, run_dir=root, device='cpu', seed=2025,
                           check=True, epochs=epochs, arms=[training.ARM], threads=1)


class MinimalRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)

    def assert_nested_equal(self, actual, expected):
        if isinstance(actual, torch.Tensor):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        elif isinstance(actual, dict):
            self.assertEqual(actual.keys(), expected.keys())
            for key in actual:
                self.assert_nested_equal(actual[key], expected[key])
        elif isinstance(actual, (list, tuple)):
            self.assertEqual(type(actual), type(expected))
            self.assertEqual(len(actual), len(expected))
            for a, b in zip(actual, expected):
                self.assert_nested_equal(a, b)
        else:
            self.assertEqual(actual, expected)

    def test_frozen_protocol_new_identity_and_absolute_epoch_semantics(self):
        protocol = legacy.read_json(PROTOCOL)
        training.validate_architecture_protocol(protocol)
        for change in ({'batch_size': 64}, {'max_epochs': 200}, {'learning_rate': .003},
                       {'graph_sha256': 'changed'}, {'test_read': True},
                       {'arms': ['B3R']}, {'report_control': 'TIID_on'},
                       {'minimal_spec': {**protocol['minimal_spec'], 'prediction_inputs': 'incidents'}}):
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, 'protocol'):
                training.validate_architecture_protocol({**protocol, **change})
        self.assertEqual(training.parse_args([]).epochs, 10)
        self.assertEqual(training.parse_args(['--check']).epochs, 1)
        with self.assertRaisesRegex(ValueError, 'Engineering check'):
            training.parse_args(['--check', '--epochs', '10'])
        with tempfile.TemporaryDirectory(dir=training.REPO) as directory:
            args = training.parse_args(['--run-dir', directory, '--device', 'cpu'])
            certificate = {'fixture': 'minimal certificate'}
            first = training.make_identity(args, protocol, {'data': 'same'}, {'graph': 'same'}, certificate)
            args.epochs = 20
            second = training.make_identity(args, protocol, {'data': 'same'}, {'graph': 'same'}, certificate)
            self.assertEqual(first, second)
            self.assertEqual(first['schema'], training.SCHEMA)
            self.assertFalse(first['tiid_report_enabled'])
            self.assertFalse(first['D_module_present'])
            self.assertFalse(first['incident_inputs_used'])
            self.assertNotIn('gpu', first['runtime'])
            self.assertEqual(first['batch_size'], 48)
            training.verify_identity(args, first, certificate, True)
            before = {name: (args.run_dir/name).read_bytes() for name in ('identity.json', 'initialization.json')}
            with self.assertRaisesRegex(ValueError, 'identity changed'):
                training.verify_identity(args, {**first, 'schema': 'b3r_run_v1'}, certificate, False)
            with self.assertRaisesRegex(ValueError, 'certificate changed'):
                training.verify_identity(args, first, {'fixture': 'changed'}, False)
            self.assertEqual(before, {name: (args.run_dir/name).read_bytes() for name in before})
        self.assertTrue(base.source_fingerprints().items() <= training.source_fingerprints().items())

    def test_preflight_actual_pruning_exact_retained_state_and_separate_reference_hash(self):
        args, dataset, g = worker_args(Path('unused')), clock_dataset(), graph(4, ((0, 1), (1, 2), (2, 3)))
        with patch.object(base, 'build_arm', side_effect=tiny_reference):
            certificate = training.preflight(args, dataset, g)
        self.assertEqual(certificate['status'], 'B3RMIN_INITIALIZATION_PAIR_PASS')
        self.assertTrue(certificate['retained_state_exact_verified'])
        self.assertTrue(certificate['incident_payload_ignored_exactly'])
        self.assertTrue(certificate['history_clock_matches_metadata'])
        self.assertEqual(certificate['reference_seed'], args.seed)
        self.assertEqual(certificate['reference_source_sha256'], base.source_fingerprints())
        self.assertEqual(certificate['initial_both_off_prediction_abs_max'], 0.)
        manifest = certificate['arms'][training.ARM]
        self.assertEqual(manifest['total_parameters'], manifest['trainable_parameters'])
        self.assertGreater(certificate['removed_parameters'], 0)
        self.assertNotEqual(certificate['reference_common_initial_sha256'], manifest['common_initial_sha256'])
        self.assertFalse(any(name.startswith(('tiid_module.', 'report_spatial_state.D.'))
                             for name in manifest['specification']))
        dataset.samples[0]['incident']['forecast_tod'] += 1
        with patch.object(base, 'build_arm', side_effect=tiny_reference):
            with self.assertRaisesRegex(ValueError, 'forecast clock'):
                training.preflight(args, dataset, g)

    def test_isolated_loop_trains_retained_P_and_exchange_without_dead_incident_parameters(self):
        before = (legacy.ARMS, legacy.build_arm, legacy.Diagnostics, base.ARMS, base.build_arm)
        loop = training.training_loop()
        self.assertEqual(before, (legacy.ARMS, legacy.build_arm, legacy.Diagnostics, base.ARMS, base.build_arm))
        self.assertEqual(loop.ARMS, training.ARMS)
        self.assertIs(loop.worker.__globals__['Diagnostics'], training.MinimalDiagnostics)
        args, dataset, g = worker_args(Path('unused')), clock_dataset(), graph(4, ((0, 1), (1, 2), (2, 3)))
        model = tiny_candidate(args, training.ARM, 4, g, 'cpu')
        protocol = legacy.read_json(PROTOCOL)
        optimizer = loop.optimizer_for(model, protocol)
        self.assertEqual(sum(p.numel() for p in model.parameters()),
                         sum(p.numel() for group in optimizer.param_groups for p in group['params']))
        train, _ = loop.run_epoch(model, optimizer, dataset, [0, 1, 2, 3], 2,
                                  args, 1, protocol, g, {'fixture': list(range(4))}, True)
        self.assertTrue(all(p.requires_grad for p in model.parameters()))
        gradients = train['first_batch_gradient_l1']
        self.assertFalse(any(name.startswith(('tiid_module.', 'report_spatial_state.D.')) for name in gradients))
        self.assertGreater(sum(value for name, value in gradients.items()
                               if name.startswith('report_spatial_state.P.') and value is not None), 0.)
        self.assertGreater(sum(value for name, value in gradients.items()
                               if '.incident_exchange.' in name and value is not None), 0.)
        mechanism = train['mechanisms']
        self.assertEqual(mechanism['schema'], training.DIAGNOSTIC_SCHEMA)
        self.assertFalse(mechanism['incident_inputs_used'])
        self.assertEqual(mechanism['raw_delta_status'], 'not_computed')
        self.assertEqual(mechanism['capacity_statistics']['report_logit_delta:all_nodes']['count']
                         if 'report_logit_delta:all_nodes' in mechanism['capacity_statistics'] else 0, 0)

    def test_resume_matches_continuous_including_optimizer_scheduler_order_and_noop(self):
        protocol, dataset, g = legacy.read_json(PROTOCOL), clock_dataset(), graph(4, ((0, 1), (1, 2), (2, 3)))
        identity = {'fixture': 'minimal resume', 'batch_size': 2, 'arm_modes': dict(training.ARMS)}
        loop = training.training_loop()
        with tempfile.TemporaryDirectory(dir=training.REPO) as directory:
            roots = [Path(directory)/name for name in ('continuous', 'resumed')]
            for root in roots:
                root.mkdir()
            args = worker_args(roots[0], 2)
            with patch.object(loop, 'build_arm', side_effect=tiny_candidate):
                loop.worker(args, identity, protocol, dataset, dataset, g, {'fixture': list(range(4))})
                args.run_dir, args.epochs = roots[1], 1
                loop.worker(args, identity, protocol, dataset, dataset, g, {'fixture': list(range(4))})
                args.epochs = 2
                loop.worker(args, identity, protocol, dataset, dataset, g, {'fixture': list(range(4))})
                checkpoint = roots[1]/training.ARM/'last.pt'
                snapshot = checkpoint.read_bytes()
                loop.worker(args, identity, protocol, dataset, dataset, g, {'fixture': list(range(4))})
                self.assertEqual(snapshot, checkpoint.read_bytes())
            a, b = [torch.load(root/training.ARM/'last.pt', map_location='cpu', weights_only=False) for root in roots]
            for name in ('model_state', 'best_model_state', 'optimizer_state', 'scheduler_state',
                         'global_updates', 'best_metric', 'wait', 'completed_epoch'):
                self.assert_nested_equal(a[name], b[name])
            self.assertEqual(a['global_updates'], 4)
            self.assertEqual(a['completed_epoch'], 2)
            for left, right in zip(a['history'], b['history']):
                for name in ('validation_mae_macro', 'order_sha256', 'learning_rate_used',
                             'next_learning_rate', 'global_updates', 'epoch_updates', 'tail_batch_size'):
                    self.assertEqual(left[name], right[name])
            for root in roots:
                with np.load(root/training.ARM/'validation_windows_epoch_002.npz', allow_pickle=False) as packed:
                    self.assertEqual(packed['absolute'].shape, (4, 12))
                    np.testing.assert_array_equal(packed['sample_indices'], np.arange(4))

    def test_sidecar_recovery_preserves_interrupted_bytes_and_rejects_conflicting_history(self):
        protocol, dataset, g = legacy.read_json(PROTOCOL), clock_dataset(), graph(4, ((0, 1), (1, 2), (2, 3)))
        identity = {'fixture': 'minimal repair', 'batch_size': 2}
        with tempfile.TemporaryDirectory(dir=training.REPO) as directory:
            root = Path(directory)
            args = worker_args(root)
            model = tiny_candidate(args, training.ARM, 4, g, 'cpu')
            legacy.write_json(root/'initialization.json', {'arms': {training.ARM: legacy.parameter_manifest(model)}})
            loop = training.training_loop()
            with patch.object(loop, 'build_arm', side_effect=tiny_candidate):
                loop.worker(args, identity, protocol, dataset, dataset, g, {'fixture': list(range(4))})
            checkpoint_path = root/training.ARM/'last.pt'
            checkpoint_bytes = checkpoint_path.read_bytes()
            saved = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            for name in ('history.json', 'summary.json'):
                (root/training.ARM/name).write_text('{interrupted', encoding='utf-8')
            self.assertTrue(training.repair_sidecars(args, identity, protocol, torch.device('cpu')))
            self.assertEqual(legacy.read_json(root/training.ARM/'history.json'), saved['history'])
            self.assertEqual(legacy.read_json(root/training.ARM/'summary.json')['completed_epoch'], 1)
            self.assertEqual(legacy.read_json(root/training.ARM/'summary.json')['environment_scope'],
                             'sidecar_repair_worker_not_recovered_training_provenance')
            self.assertEqual(len(list((root/training.ARM).glob('*.interrupted_*'))), 2)
            self.assertEqual(checkpoint_bytes, checkpoint_path.read_bytes())
            legacy.write_json(root/training.ARM/'history.json', [{'epoch': 99}])
            with self.assertRaisesRegex(ValueError, 'conflicts'):
                training.repair_sidecars(args, identity, protocol, torch.device('cpu'))
            self.assertEqual(legacy.read_json(root/training.ARM/'history.json'), [{'epoch': 99}])
            with self.assertRaisesRegex(ValueError, 'identity/arm mismatch'):
                training.repair_sidecars(args, {'changed': True}, protocol, torch.device('cpu'))

    def test_worker_guard_precedes_data_access_and_rejects_legacy_token(self):
        with tempfile.TemporaryDirectory(dir=training.REPO) as directory:
            root = Path(directory).resolve()
            with patch.dict(os.environ, {'B3R_MIN_INTERNAL_WORKER': '',
                                         'B3R_STRENGTH_INTERNAL_WORKER': str(root)+'|'+training.ARM}):
                with patch.object(training, 'prepare', side_effect=AssertionError('Data accessed before guard')):
                    with self.assertRaisesRegex(ValueError, 'direct internal workers'):
                        training.main(['--data-source', 'chronological', '--run-dir', str(root), '--device', 'cpu', '--worker', training.ARM])


if __name__ == '__main__':
    unittest.main()
