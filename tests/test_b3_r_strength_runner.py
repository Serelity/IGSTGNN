"""G1.1 runner contracts: identity strength, independent no-D and fair budgets."""

import copy
import math
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.chronological import train_b3_r as base
from experiments.chronological import train_b3_r_strength as training
from experiments.chronological import train_internal_spatial as legacy
from src.models.incident_spatial import configure_spatial_mode
from src.models.incident_spatial_residual import from_canonical_b3
from src.models.incident_spatial_strength import from_residual
from tests.test_architecture_mechanisms import inputs, tiny_model
from tests.test_incident_spatial import graph
from tests.test_internal_spatial_training import Dataset


PROTOCOL = training.HERE/'internal_spatial_b3_r_strength_g11.json'


def tiny_candidate(args, arm, nodes, g, device):
    canonical = configure_spatial_mode(tiny_model().requires_grad_(True), 'state_restricted', copy.deepcopy(g))
    model = from_canonical_b3(canonical, report_path_enabled=arm != 'B3R_NO_D')[0]
    if arm in ('B3R_HALF', 'B3R_GATE'):
        model = from_residual(model, training.ARMS[arm])[0]
    return model.to(device)


def worker_args(root, epochs=1, arm='B3R_GATE'):
    return SimpleNamespace(worker=arm, run_dir=root, device='cpu', seed=2025,
                           check=True, epochs=epochs, arms=list(training.ARMS),
                           only=[arm], threads=1)


class StrengthRunnerTests(unittest.TestCase):
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

    def test_protocol_and_new_identity_preserve_frozen_training_and_only_resume_semantics(self):
        protocol = legacy.read_json(PROTOCOL)
        training.validate_architecture_protocol(protocol)
        changes = ({'batch_size': 24}, {'learning_rate': .003}, {'max_epochs': 200},
                   {'graph_sha256': 'changed'}, {'test_read': True},
                   {'readout': 'different'}, {'architecture_version': 'B3-R-G1.0'},
                   {'arms': list(training.ARMS)[:-1]},
                   {'report_control': 'TIID_off'},
                   {'gate_spec': {**protocol['gate_spec'], 'range': [0., 1.]}},
                   {'gate_spec': {**protocol['gate_spec'], 'initial_strength': .5}})
        for changed in changes:
            with self.subTest(changed=changed), self.assertRaisesRegex(ValueError, 'protocol'):
                training.validate_architecture_protocol({**protocol, **changed})
        with tempfile.TemporaryDirectory(dir=training.REPO) as directory:
            root = Path(directory)
            args = training.parse_args(['--run-dir', str(root), '--device', 'cpu', '--only', 'B3R_GATE'])
            initial = {'fixture': 'same shared initialization'}
            first = training.make_identity(args, protocol, {'data': 'same'}, {'graph': 'same'}, initial)
            args.only, args.epochs = ['B3R_NO_D'], 10
            second = training.make_identity(args, protocol, {'data': 'same'}, {'graph': 'same'}, initial)
            self.assertEqual(first, second)
            self.assertEqual(first['schema'], 'b3r_strength_run_g11_v1')
            self.assertEqual(first['arm_modes'], training.ARMS)
            self.assertTrue(first['tiid_report_enabled'])
            self.assertNotIn('gpu', first['runtime'])
            self.assertFalse(first['future_bias_labels_used_as_inputs'])
            training.verify_identity(args, first, initial, True)
            before = {name: (root/name).read_bytes() for name in ('identity.json', 'initialization.json')}
            training.verify_identity(args, second, initial, False)
            with self.assertRaisesRegex(ValueError, 'identity changed'):
                training.verify_identity(args, {**second, 'schema': 'b3r_gate_run_v1'}, initial, False)
            with self.assertRaisesRegex(ValueError, 'certificate changed'):
                training.verify_identity(args, second, {'fixture': 'different'}, False)
            self.assertEqual(before, {name: (root/name).read_bytes() for name in before})
        frozen, current = base.source_fingerprints(), training.source_fingerprints()
        self.assertTrue(frozen.keys() <= current.keys())
        self.assertEqual(frozen, {name: current[name] for name in frozen})
        with self.assertRaisesRegex(ValueError, 'only'):
            training.parse_args(['--arms', 'B3R', '--only', 'B3R_NO_D'])

    def test_four_arm_preflight_and_isolated_loop_certify_identity_center_not_only_zero_d(self):
        old = (legacy.ARMS, legacy.build_arm, legacy.Diagnostics, base.ARMS, base.build_arm)
        loop = training.training_loop(list(training.ARMS))
        self.assertEqual(old, (legacy.ARMS, legacy.build_arm, legacy.Diagnostics, base.ARMS, base.build_arm))
        self.assertEqual(loop.ARMS, training.ARMS)
        self.assertIs(loop.worker.__globals__['Diagnostics'], training.GateDiagnostics)
        args, dataset, g = worker_args(Path('unused')), Dataset(), graph(4, ((0, 1),))
        with patch.object(training, 'build_arm', side_effect=tiny_candidate):
            certificate = training.preflight(args, dataset, g)
        self.assertEqual(certificate['status'], 'B3RSTRENGTH_INITIALIZATION_PAIR_PASS')
        self.assertEqual(certificate['initial_prediction_abs_max'], 0.)
        self.assertEqual(certificate['half_initial_strength'], .5)
        self.assertEqual(certificate['gate_initial_strength'], 1.)
        self.assertTrue(certificate['nonzero_D_gate_prediction_and_capacity_exact'])
        self.assertTrue(certificate['no_D_training_path_disabled'])
        self.assertTrue(certificate['original_tiid_on'])
        counts = {arm: row['total_parameters'] for arm, row in certificate['arms'].items()}
        self.assertEqual(counts['B3R'], counts['B3R_HALF'])
        self.assertEqual(counts['B3R'], counts['B3R_NO_D'])
        self.assertEqual(counts['B3R_GATE']-counts['B3R'], 737)
        for manifest in certificate['arms'].values():
            self.assertEqual(manifest['total_parameters'], manifest['trainable_parameters'])

    def test_no_d_training_skips_d_without_disabling_tiid_or_fabricating_alpha_zero(self):
        args, dataset, g = worker_args(Path('unused'), arm='B3R_NO_D'), Dataset(), graph(4, ((0, 1),))
        sentinel = object()
        with patch.object(base, 'build_arm', return_value=sentinel) as build:
            self.assertIs(training.build_arm(args, 'B3R_NO_D', 4, g, 'cpu'), sentinel)
        self.assertEqual(build.call_args.args[1], 'B3R_OFF')
        model = tiny_candidate(args, args.worker, 4, g, 'cpu')
        self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))
        before = copy.deepcopy(model.report_spatial_state.D.state_dict())
        protocol = legacy.read_json(PROTOCOL)
        loop = training.training_loop(list(training.ARMS))
        optimizer = loop.optimizer_for(model, protocol)
        calls = []
        handle = model.tiid_module.register_forward_hook(lambda *unused: calls.append(True))
        try:
            with patch.object(model.report_spatial_state.D, 'forward', side_effect=AssertionError('NO_D executed D')):
                train, _ = loop.run_epoch(model, optimizer, dataset, [0, 1, 2, 3], 2,
                                          args, 1, protocol, g, {'fixture': list(range(4))}, True)
        finally:
            handle.remove()
        self.assertEqual(len(calls), 2)
        self.assert_nested_equal(model.report_spatial_state.D.state_dict(), before)
        d_gradients = {name: value for name, value in train['first_batch_gradient_l1'].items()
                       if name.startswith('report_spatial_state.D.')}
        self.assertEqual(len(d_gradients), 4)
        self.assertTrue(all(value is None for value in d_gradients.values()))
        mechanism = train['mechanisms']
        self.assertEqual(mechanism['raw_delta_status'], 'not_computed')
        self.assertEqual(mechanism['d_strength_status'], 'not_computed')
        self.assertIsNone(mechanism['d_strength_population_std'])
        self.assertIsNone(mechanism['d_magnitude_weighted_strength'])
        self.assertNotIn('d_strength:report_nodes', mechanism['capacity_statistics'])
        x, incident = inputs()
        model.eval()
        with torch.inference_mode():
            with_tiid = model(x, incident_data=incident)
            no_tiid = model(x, incident_data=incident, tiid_report_enabled=False)
        self.assertGreater(float((with_tiid-no_tiid).abs().max()), 0.)

    def test_strength_diagnostics_accept_amplification_and_keep_node_and_latent_denominators_separate(self):
        args, g = worker_args(Path('unused')), graph(4, ((0, 1),))
        model = tiny_candidate(args, 'B3R_GATE', 4, g, 'cpu').eval()
        with torch.no_grad():
            model.report_spatial_state.D[-1].bias.copy_(torch.tensor([.8, -.6, 1.2, -1.]))
            model.report_spatial_state.gate[-1].bias.fill_(math.log(3.))
        x, incident = inputs()
        with torch.inference_mode():
            context = model(x, incident_data=incident, return_spatial_details=True)['spatial']
        collector = training.GateDiagnostics()
        collector.update(context)
        collector.update(context)
        result, count = collector.result(), 2*int(context['report_mask'].sum())
        stats = result['capacity_statistics']
        self.assertEqual(result['schema'], training.DIAGNOSTIC_SCHEMA)
        self.assertEqual(result['d_strength_status'], 'available')
        self.assertEqual(stats['d_strength:report_nodes']['count'], count)
        self.assertAlmostEqual(stats['d_strength:report_nodes']['mean'], 1.5, places=6)
        self.assertEqual(stats['d_strength_above_one:report_nodes']['mean'], 1.)
        for name in ('d_strength_below_one', 'd_strength_below_0_1', 'd_strength_above_1_9'):
            self.assertEqual(stats[name+':report_nodes']['mean'], 0.)
        raw, applied = [stats[name+':report_nodes'] for name in
                        ('raw_effective_logit_delta_abs', 'applied_effective_logit_delta_abs')]
        self.assertEqual(raw['count'], count*4)
        self.assertEqual(applied['count'], count*4)
        self.assertEqual(stats['node_capacity:report_nodes']['count'], count*4)
        self.assertAlmostEqual(applied['mean'], 1.5*raw['mean'], places=6)
        self.assertAlmostEqual(result['d_magnitude_weighted_strength'], 1.5, places=6)
        self.assertFalse(result['calibrated_reliability_claim'])
        for bad in (-.1, 2.1, float('nan')):
            with self.subTest(strength=bad), self.assertRaisesRegex(ValueError, 'strength'):
                training.GateDiagnostics().update({**context, 'd_strength': torch.full_like(context['d_strength'], bad)})
        incident['distances'].zero_()
        with torch.inference_mode():
            empty = model(x, incident_data=incident, return_spatial_details=True)['spatial']
        collector = training.GateDiagnostics()
        collector.update(empty)
        result = collector.result()
        self.assertEqual(result['d_strength_status'], 'no_associated_nodes')
        self.assertIsNone(result['d_magnitude_weighted_strength'])
        self.assertEqual(result['capacity_statistics']['d_strength:report_nodes']['count'], 0)

    def test_gate_and_independent_no_d_resume_match_continuous_optimizer_scheduler_and_updates(self):
        protocol, dataset, g = legacy.read_json(PROTOCOL), Dataset(), graph(4, ((0, 1),))
        identity = {'fixture': 'strength resume', 'batch_size': 2, 'arm_modes': dict(training.ARMS)}
        frozen_before = base.source_fingerprints()
        loop = training.training_loop(list(training.ARMS))
        with tempfile.TemporaryDirectory(dir=training.REPO) as directory:
            for arm in ('B3R_GATE', 'B3R_NO_D'):
                with self.subTest(arm=arm):
                    roots = [Path(directory)/(arm+'_'+name) for name in ('continuous', 'resumed')]
                    for root in roots:
                        root.mkdir()
                    args = worker_args(roots[0], 2, arm)
                    with patch.object(loop, 'build_arm', side_effect=tiny_candidate):
                        loop.worker(args, identity, protocol, dataset, dataset, g, {'fixture': list(range(4))})
                        args.run_dir, args.epochs = roots[1], 1
                        loop.worker(args, identity, protocol, dataset, dataset, g, {'fixture': list(range(4))})
                        args.epochs = 2
                        loop.worker(args, identity, protocol, dataset, dataset, g, {'fixture': list(range(4))})
                        checkpoint = roots[1]/arm/'last.pt'
                        snapshot = checkpoint.read_bytes()
                        loop.worker(args, identity, protocol, dataset, dataset, g, {'fixture': list(range(4))})
                        self.assertEqual(snapshot, checkpoint.read_bytes())
                    a, b = [torch.load(root/arm/'last.pt', map_location='cpu', weights_only=False) for root in roots]
                    for name in ('model_state', 'best_model_state', 'optimizer_state', 'scheduler_state',
                                 'global_updates', 'best_metric', 'wait', 'completed_epoch'):
                        self.assert_nested_equal(a[name], b[name])
                    self.assertEqual(a['global_updates'], 4)
                    self.assertEqual(a['completed_epoch'], 2)
                    for left, right in zip(a['history'], b['history']):
                        for name in ('validation_mae_macro', 'order_sha256', 'learning_rate_used',
                                     'next_learning_rate', 'global_updates', 'epoch_updates', 'tail_batch_size'):
                            self.assertEqual(left[name], right[name])
                    if arm == 'B3R_NO_D':
                        for row in a['history']:
                            self.assertEqual(row['train']['mechanisms']['d_strength_status'], 'not_computed')
                            self.assertEqual(row['validation']['mechanisms']['d_strength_status'], 'not_computed')
                    for root in roots:
                        with np.load(root/arm/'validation_windows_epoch_002.npz', allow_pickle=False) as packed:
                            self.assertEqual(packed['absolute'].shape, (4, 12))
                            np.testing.assert_array_equal(packed['sample_indices'], np.arange(4))
        self.assertEqual(frozen_before, base.source_fingerprints())

    def test_four_arm_report_uses_shortest_common_budget_and_waits_for_no_d(self):
        values = {'B3R': [35., 34., 32., 31., 30., 29., 28., 27.],
                  'B3R_HALF': [34., 33., 31., 30., 29., 28., 27., 26., 1., 2.],
                  'B3R_GATE': [36., 32., 30., 29., 28., 27., 26., 25., 1., .8, .6, .4],
                  'B3R_NO_D': [38., 37., 35., 34., 33., 32.]}
        with tempfile.TemporaryDirectory(dir=training.REPO) as directory:
            root = Path(directory)
            legacy.write_json(root/'identity.json', {'engineering_check': False})
            for arm, history_values in values.items():
                if arm == 'B3R_NO_D':
                    partial = training.paired_report(root, list(training.ARMS))
                    self.assertEqual(partial['status'], 'B3RSTRENGTH_PARTIAL_STAGE_COMPLETE')
                    self.assertEqual(partial['missing_arms'], ['B3R_NO_D'])
                    self.assertEqual(partial['common_epoch'], 0)
                    self.assertEqual(partial['comparisons'], {})
                (root/arm).mkdir()
                legacy.write_json(root/arm/'history.json', [dict(epoch=i+1, validation_mae_macro=value)
                                                           for i, value in enumerate(history_values)])
                legacy.write_json(root/arm/'summary.json', dict(arm=arm, completed_epoch=len(history_values),
                                                               best_metric=min(history_values)))
            result = training.paired_report(root, list(training.ARMS))
            self.assertEqual(result['status'], 'B3RSTRENGTH_PAIRED_STAGE_COMPLETE')
            self.assertEqual(result['common_epoch'], 6)
            self.assertEqual(len(result['comparisons']), 6)
            gate = result['comparisons']['B3R_GATE_vs_B3R']
            self.assertEqual(gate['common_epoch_mae_gain'], 2.)
            self.assertEqual(gate['common_prefix_best_mae_gain'], 2.)
            self.assertEqual(gate['common_last_five_mean_gain'], 2.)
            self.assertEqual(gate['candidate_common_last_five'], values['B3R_GATE'][1:6])
            self.assertEqual(gate['control_common_last_five'], values['B3R'][1:6])
            self.assertEqual(result['comparisons']['B3R_GATE_vs_B3R_NO_D']['common_epoch_mae_gain'], 5.)
            self.assertNotIn('best_mae_gain', gate)
            self.assertFalse(result['predictive_gain_claim'])
            self.assertFalse(result['test_read'])

    def test_legacy_worker_token_cannot_bypass_new_launcher_lock(self):
        with tempfile.TemporaryDirectory(dir=training.REPO) as directory:
            root = Path(directory).resolve()
            token = str(root)+'|B3R_NO_D'
            with (patch.dict(os.environ, {'B3R_STRENGTH_INTERNAL_WORKER': '', 'B3R_GATE_INTERNAL_WORKER': token}),
                  patch.object(training, 'prepare', side_effect=AssertionError('Data loaded before worker guard'))):
                with self.assertRaisesRegex(ValueError, 'direct internal workers'):
                    training.main(['--run-dir', str(root), '--device', 'cpu', '--worker', 'B3R_NO_D'])


if __name__ == '__main__':
    unittest.main()
