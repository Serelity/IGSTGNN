"""Frozen-loop and checkpoint contract checks for the independent gate runner."""

import copy
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.chronological import train_b3_r as base
from experiments.chronological import train_b3_r_gate as training
from experiments.chronological import train_internal_spatial as legacy
from experiments.chronological.smoke import sha256
from experiments.chronological.train import save_checkpoint
from src.models.incident_spatial import configure_spatial_mode
from src.models.incident_spatial_attenuation import from_residual
from src.models.incident_spatial_residual import from_canonical_b3
from tests.test_architecture_mechanisms import inputs, tiny_model
from tests.test_incident_spatial import graph
from tests.test_internal_spatial_training import Dataset


PROTOCOL = training.HERE/'internal_spatial_b3_r_gate_v1.json'


def tiny_candidate(args, arm, nodes, g, device):
    canonical = configure_spatial_mode(tiny_model().requires_grad_(True), 'state_restricted', copy.deepcopy(g))
    model = from_canonical_b3(canonical)[0]
    if arm != 'B3R':
        model = from_residual(model, training.ARMS[arm])[0]
    return model.to(device)


def worker_args(root, epochs=1):
    return SimpleNamespace(worker='B3R_GATE', run_dir=root, device='cpu', seed=2025,
                           check=True, epochs=epochs, arms=list(training.ARMS),
                           only=['B3R_GATE'], threads=1)


class GateRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)

    def assert_nested_equal(self, actual, expected):
        if isinstance(actual, torch.Tensor):
            self.assertIsInstance(expected, torch.Tensor)
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

    def test_protocol_rejects_training_data_and_gate_contract_changes(self):
        protocol = legacy.read_json(PROTOCOL)
        training.validate_architecture_protocol(protocol)
        changes = ({'batch_size': 24}, {'learning_rate': .003}, {'max_epochs': 200},
                   {'graph_sha256': 'changed'}, {'test_read': True},
                   {'readout': 'different'}, {'architecture_version': 'other'},
                   {'gate_spec': {**protocol['gate_spec'], 'hidden': 32}})
        for changed in changes:
            with self.subTest(changed=changed), self.assertRaisesRegex(ValueError, 'protocol'):
                training.validate_architecture_protocol({**protocol, **changed})

    def test_only_and_epoch_target_do_not_change_resume_identity_or_old_sources(self):
        protocol = legacy.read_json(PROTOCOL)
        with tempfile.TemporaryDirectory(dir=training.REPO) as directory:
            root = Path(directory)
            args = training.parse_args(['--run-dir', str(root), '--device', 'cpu', '--only', 'B3R_GATE'])
            initial = {'fixture': 'same common initialization'}
            first = training.make_identity(args, protocol, {'data': 'same'}, {'graph': 'same'}, initial)
            args.only, args.epochs = ['B3R_HALF'], 10
            second = training.make_identity(args, protocol, {'data': 'same'}, {'graph': 'same'}, initial)
            self.assertEqual(first, second)
            self.assertEqual(set(first['arm_modes']), set(training.ARMS))
            self.assertNotIn('gpu', first['runtime'])
            self.assertFalse(first['future_bias_labels_used_as_inputs'])
            training.verify_identity(args, first, initial, True)
            before = {name: (root/name).read_bytes() for name in ('identity.json', 'initialization.json')}
            training.verify_identity(args, second, initial, False)
            with self.assertRaisesRegex(ValueError, 'identity changed'):
                training.verify_identity(args, {**second, 'gate_target': 'whole_readout'}, initial, False)
            with self.assertRaisesRegex(ValueError, 'certificate changed'):
                training.verify_identity(args, second, {'fixture': 'different initialization'}, False)
            self.assertEqual(before, {name: (root/name).read_bytes() for name in before})
        frozen = base.source_fingerprints()
        current = training.source_fingerprints()
        self.assertTrue(frozen.keys() <= current.keys())
        self.assertEqual(frozen, {name: current[name] for name in frozen})
        with self.assertRaisesRegex(ValueError, 'only'):
            training.parse_args(['--arms', 'B3R', '--only', 'B3R_GATE'])

    def test_loop_isolation_and_initialization_keep_all_shared_model_state(self):
        old = (legacy.ARMS, legacy.build_arm, legacy.Diagnostics, base.ARMS, base.build_arm)
        loop = training.training_loop(list(training.ARMS))
        self.assertEqual(old, (legacy.ARMS, legacy.build_arm, legacy.Diagnostics, base.ARMS, base.build_arm))
        self.assertIs(loop.worker.__globals__['Diagnostics'], training.GateDiagnostics)
        self.assertEqual(loop.ARMS, training.ARMS)
        dataset, g = Dataset(), graph(4, ((0, 1),))
        args = worker_args(Path('unused'))
        with patch.object(training, 'build_arm', side_effect=tiny_candidate):
            certificate = training.preflight(args, dataset, g)
        self.assertEqual(certificate['status'], 'B3RGATE_INITIALIZATION_PAIR_PASS')
        counts = {arm: row['total_parameters'] for arm, row in certificate['arms'].items()}
        self.assertEqual(counts['B3R'], counts['B3R_HALF'])
        self.assertEqual(counts['B3R_GATE']-counts['B3R'], 737)
        self.assertEqual(certificate['initial_prediction_abs_max'], 0.)

    def test_real_gate_worker_resume_matches_continuous_model_optimizer_scheduler_and_updates(self):
        protocol = legacy.read_json(PROTOCOL)
        dataset, g = Dataset(), graph(4, ((0, 1),))
        identity = {'fixture': 'gate resume', 'batch_size': 2,
                    'arm_modes': dict(training.ARMS)}
        frozen_before = base.source_fingerprints()
        loop = training.training_loop(list(training.ARMS))
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
                snapshot = (roots[1]/'B3R_GATE'/'last.pt').read_bytes()
                loop.worker(args, identity, protocol, dataset, dataset, g, {'fixture': list(range(4))})
                self.assertEqual(snapshot, (roots[1]/'B3R_GATE'/'last.pt').read_bytes())
            a, b = [torch.load(root/'B3R_GATE'/'last.pt', map_location='cpu', weights_only=False) for root in roots]
            for key in ('model_state', 'best_model_state', 'optimizer_state', 'scheduler_state',
                        'global_updates', 'best_metric', 'wait', 'completed_epoch'):
                self.assert_nested_equal(a[key], b[key])
            self.assertEqual(a['global_updates'], 4)
            for left, right in zip(a['history'], b['history']):
                for key in ('validation_mae_macro', 'order_sha256', 'learning_rate_used',
                            'next_learning_rate', 'global_updates', 'epoch_updates', 'tail_batch_size'):
                    self.assertEqual(left[key], right[key])
            gradients = a['history'][0]['train']['first_batch_gradient_l1']
            gate_keys = [key for key in gradients if key.startswith('report_spatial_state.gate.')]
            self.assertEqual(len(gate_keys), 4)
            self.assertTrue(all(gradients[key] == 0. for key in gate_keys))
            for root in roots:
                with np.load(root/'B3R_GATE'/'validation_windows_epoch_002.npz', allow_pickle=False) as packed:
                    self.assertEqual(packed['absolute'].shape, (4, 12))
                    np.testing.assert_array_equal(packed['sample_indices'], np.arange(4))
        self.assertEqual(frozen_before, base.source_fingerprints())

    def test_diagnostics_gate_counts_nodes_and_retains_raw_and_applied_d_magnitudes(self):
        g = graph(4, ((0, 1),))
        args = worker_args(Path('unused'))
        model = tiny_candidate(args, 'B3R_GATE', 4, g, 'cpu').eval()
        with torch.no_grad():
            model.report_spatial_state.D[-1].bias.copy_(torch.tensor([.8, -.6, 1.2, -1.]))
        x, report = inputs()
        with torch.inference_mode():
            context = model(x, incident_data=report, return_spatial_details=True)['spatial']
        collector = training.GateDiagnostics()
        collector.update(context)
        collector.update(context)
        result = collector.result()
        stats = result['capacity_statistics']
        node_count = 2*int(context['report_mask'].sum())
        self.assertEqual(stats['d_strength:report_nodes']['count'], node_count)
        self.assertEqual(stats['node_capacity:report_nodes']['count'], node_count*4)
        self.assertEqual(stats['d_strength:report_nodes']['mean'], .5)
        self.assertEqual(result['d_strength_population_std'], 0.)
        self.assertEqual(stats['d_strength_below_0_1:report_nodes']['mean'], 0.)
        self.assertEqual(stats['d_strength_above_0_9:report_nodes']['mean'], 0.)
        raw, applied = [stats[key+':report_nodes'] for key in
                        ('raw_effective_logit_delta_abs', 'applied_effective_logit_delta_abs')]
        self.assertEqual(raw['count'], node_count*4)
        self.assertEqual(applied['count'], node_count*4)
        self.assertAlmostEqual(applied['mean'], .5*raw['mean'], places=12)
        self.assertFalse(result['calibrated_reliability_claim'])
        report['distances'].zero_()
        with torch.inference_mode():
            empty = model(x, incident_data=report, return_spatial_details=True)['spatial']
        collector = training.GateDiagnostics()
        collector.update(empty)
        self.assertIsNone(collector.result()['d_strength_population_std'])
        self.assertEqual(collector.result()['capacity_statistics']['d_strength:report_nodes']['count'], 0)

    def test_paired_report_uses_common_prefix_and_reports_unstarted_arms(self):
        values = {'B3R': [35., 34., 32., 31., 30., 29., 28., 27.],
                  'B3R_HALF': [34., 33., 31., 30., 29., 28., 27., 26., 1., 2.],
                  'B3R_GATE': [36., 32., 30., 29., 28., 27., 26., 25., 1., .8, .6, .4]}
        with tempfile.TemporaryDirectory(dir=training.REPO) as directory:
            root = Path(directory)
            legacy.write_json(root/'identity.json', {'engineering_check': False})
            partial = training.paired_report(root, list(training.ARMS))
            self.assertEqual(partial['status'], 'B3RGATE_PARTIAL_STAGE_COMPLETE')
            self.assertEqual(partial['common_epoch'], 0)
            self.assertEqual(partial['comparisons'], {})
            self.assertEqual(partial['missing_arms'], list(training.ARMS))
            for arm, history_values in values.items():
                (root/arm).mkdir()
                legacy.write_json(root/arm/'history.json', [dict(epoch=i+1, validation_mae_macro=value)
                                                           for i, value in enumerate(history_values)])
                legacy.write_json(root/arm/'summary.json', dict(arm=arm, completed_epoch=len(history_values),
                                                               best_metric=min(history_values)))
            result = training.paired_report(root, list(training.ARMS))
            self.assertEqual(result['common_epoch'], 8)
            self.assertEqual(result['status'], 'B3RGATE_PAIRED_STAGE_COMPLETE')
            gate = result['comparisons']['B3R_GATE_vs_B3R']
            self.assertEqual(gate['common_epoch_mae_gain'], 2.)
            self.assertEqual(gate['common_prefix_best_mae_gain'], 2.)
            self.assertEqual(gate['common_last_five_mean_gain'], 2.)
            self.assertEqual(gate['candidate_common_last_five'], values['B3R_GATE'][3:8])
            self.assertEqual(gate['control_common_last_five'], values['B3R'][3:8])
            self.assertNotIn('best_mae_gain', gate)

    def test_checkpoint_authority_repairs_sidecars_and_rejects_identity_and_model_mismatch(self):
        protocol = legacy.read_json(PROTOCOL)
        dataset, g = Dataset(), graph(4, ((0, 1),))
        identity = {'fixture': 'repair', 'batch_size': 2, 'arm_modes': dict(training.ARMS)}
        loop = training.training_loop(list(training.ARMS))
        with tempfile.TemporaryDirectory(dir=training.REPO) as directory:
            args = worker_args(Path(directory))
            with patch.object(loop, 'build_arm', side_effect=tiny_candidate):
                loop.worker(args, identity, protocol, dataset, dataset, g, {'fixture': list(range(4))})
            root = args.run_dir/'B3R_GATE'
            checkpoint = root/'last.pt'
            saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
            legacy.write_json(root/'history.json', [])
            legacy.write_json(root/'summary.json', {'completed_epoch': 0, 'sentinel': 'stale'})
            checkpoint_hash = sha256(checkpoint)
            with patch.object(training, 'build_arm', side_effect=lambda *unused: tiny_candidate(args, args.worker, 4, g, 'cpu')):
                training.repair_sidecars(args, identity, protocol, torch.device('cpu'))
            self.assertEqual(sha256(checkpoint), checkpoint_hash)
            self.assertEqual(legacy.read_json(root/'history.json'), saved['history'])
            summary = legacy.read_json(root/'summary.json')
            self.assertEqual(summary['completed_epoch'], 1)
            self.assertEqual(summary['best_metric'], saved['best_metric'])
            self.assertEqual(summary['arm'], args.worker)
            before = {name: (root/name).read_bytes() for name in ('history.json', 'summary.json')}
            with self.assertRaisesRegex(ValueError, 'identity'):
                training.repair_sidecars(args, {**identity, 'fixture': 'different'}, protocol, torch.device('cpu'))
            self.assertEqual(before, {name: (root/name).read_bytes() for name in before})
            key = next(name for name, value in saved['model_state'].items() if value.numel())
            broken = copy.deepcopy(saved)
            broken['model_state'][key] = torch.empty(0)
            save_checkpoint(checkpoint, broken)
            with patch.object(training, 'build_arm', side_effect=lambda *unused: tiny_candidate(args, args.worker, 4, g, 'cpu')):
                with self.assertRaises((ValueError, RuntimeError)):
                    training.repair_sidecars(args, identity, protocol, torch.device('cpu'))
            self.assertEqual(before, {name: (root/name).read_bytes() for name in before})

    def test_direct_worker_is_blocked_before_loading_data_or_checkpoints(self):
        with tempfile.TemporaryDirectory(dir=training.REPO) as directory:
            with (patch.dict(os.environ, {'B3R_GATE_INTERNAL_WORKER': ''}),
                  patch.object(training, 'prepare', side_effect=AssertionError('Data loaded before worker guard'))):
                with self.assertRaisesRegex(ValueError, 'direct internal workers'):
                    training.main(['--run-dir', directory, '--device', 'cpu', '--worker', 'B3R_GATE'])

    def test_checkpoint_recovers_truncated_sidecars_but_not_conflicting_valid_history(self):
        protocol = legacy.read_json(PROTOCOL)
        dataset, g = Dataset(), graph(4, ((0, 1),))
        identity = {'fixture': 'interrupted JSON write', 'batch_size': 2,
                    'arm_modes': dict(training.ARMS)}
        loop = training.training_loop(list(training.ARMS))
        with tempfile.TemporaryDirectory(dir=training.REPO) as directory:
            args = worker_args(Path(directory))
            with patch.object(loop, 'build_arm', side_effect=tiny_candidate):
                loop.worker(args, identity, protocol, dataset, dataset, g, {'fixture': list(range(4))})
            root = args.run_dir/'B3R_GATE'
            checkpoint_hash = sha256(root/'last.pt')
            saved = torch.load(root/'last.pt', map_location='cpu', weights_only=False)
            (root/'history.json').write_text('[{"epoch":', encoding='utf-8')
            (root/'summary.json').write_text('{"completed_epoch":', encoding='utf-8')
            training.repair_sidecars(args, identity, protocol, torch.device('cpu'))
            self.assertEqual(legacy.read_json(root/'history.json'), saved['history'])
            self.assertEqual(legacy.read_json(root/'summary.json')['completed_epoch'], 1)
            self.assertEqual(sha256(root/'last.pt'), checkpoint_hash)
            legacy.write_json(root/'history.json', [{'epoch': 1, 'validation_mae_macro': 99.}])
            before = {name: (root/name).read_bytes() for name in ('history.json', 'summary.json')}
            with self.assertRaisesRegex(ValueError, 'conflicts'):
                training.repair_sidecars(args, identity, protocol, torch.device('cpu'))
            self.assertEqual(before, {name: (root/name).read_bytes() for name in before})
            self.assertEqual(sha256(root/'last.pt'), checkpoint_hash)


if __name__ == '__main__':
    unittest.main()
