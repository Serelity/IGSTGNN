"""Server-only contracts for GW9; synthetic checks are not accuracy evidence.

Run on server: python -m unittest tests.test_branch_readout_gw9_model -v
Do not execute these tests, imports, or syntax checks on the local workstation.
"""
import copy
import unittest
from unittest.mock import patch

import torch

from experiments.branch_readout_gw9.model import (
    ARMS, PROBE_ARMS, BranchReadoutIGSTGNN, ReadoutHead, reshape_blocks)
from experiments.chronological.original_graph_message import GraphMessageIGSTGNN
from experiments.chronological.original_receiver_residual import ReceiverResidualIGSTGNN
from experiments.chronological.original_igstgnn_runtime import rng_fingerprint, tensor_tree_hash
from experiments.chronological.original_igstgnn_source import get_original_modules
from experiments.graph_memory_gw6.model import GraphMemoryIGSTGNN
from experiments.lag_transport_gw8.model import LagTransportIGSTGNN
from tests import test_original_graph_window as fixtures


class BranchReadoutGW9Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.modules = get_original_modules()

    native_model = fixtures.GraphWindowTests.model
    batch = staticmethod(fixtures.GraphWindowTests.batch)

    def anchor(self):
        receiver = ReceiverResidualIGSTGNN(
            GraphMessageIGSTGNN(self.native_model(), 'POST_MESSAGE'), 'RECEIVER_RESIDUAL')
        with torch.no_grad():
            receiver.selector.router.static_logits.uniform_(-.4, .4)
            receiver.selector.router.score_head.weight.normal_(std=.1)
            receiver.selector.residual.key.weight.normal_(std=.15)
            receiver.selector.residual.bias.normal_(std=.1)
        recurrent = GraphMemoryIGSTGNN(receiver, 'RECURRENT_CONTROL', stage='joint')
        return LagTransportIGSTGNN(recurrent, 'R_CONTINUE').eval()

    def test_registered_arms_anchor_validation_and_dimensions(self):
        self.assertEqual(ARMS, ('R_CONTINUE', 'SUM_CONTROL', 'BRANCH_READOUT'))
        self.assertEqual(PROBE_ARMS, ARMS[1:])
        anchor = self.anchor()
        for invalid in ('OTHER', 'R_MEMORY'):
            with self.assertRaises(ValueError):
                BranchReadoutIGSTGNN(copy.deepcopy(anchor), invalid)
        with self.assertRaises(ValueError):
            BranchReadoutIGSTGNN(anchor.anchor, 'BRANCH_READOUT')
        with self.assertRaises(ValueError):
            BranchReadoutIGSTGNN(copy.deepcopy(anchor), 'BRANCH_READOUT', stage='unknown')
        with self.assertRaises(ValueError):
            BranchReadoutIGSTGNN(copy.deepcopy(anchor), 'BRANCH_READOUT', hidden=0)
        for channels, gap, arm, hidden in ((0, 3, 'SUM_CONTROL', 16),
                (256, 0, 'SUM_CONTROL', 16), (256, 3, 'R_CONTINUE', 16),
                (256, 3, 'BRANCH_READOUT', 0)):
            with self.assertRaises(ValueError):
                ReadoutHead(channels, gap, arm, hidden)

    def test_construction_preserves_anchor_names_tensors_rng_and_joint_modes(self):
        anchor = self.anchor().train()
        anchor.original.embedding.eval()  # Deliberately heterogeneous modes.
        names = list(anchor.state_dict())
        tensors, rng = tensor_tree_hash(anchor.state_dict()), rng_fingerprint()
        modes = {name: module.training for name, module in anchor.named_modules()}
        for arm in ARMS:
            model = BranchReadoutIGSTGNN(copy.deepcopy(anchor), arm)
            self.assertEqual(rng, rng_fingerprint())
            self.assertEqual(tensors, tensor_tree_hash(model.anchor.state_dict()))
            self.assertEqual(names, list(model.anchor.state_dict()))
            self.assertEqual(modes,
                {name: module.training for name, module in model.anchor.named_modules()})
            self.assertTrue(model.training)
            model.assert_cache_clear()

    def test_equal_capacity_identical_initial_heads_and_strict_state_roundtrip(self):
        anchor = self.anchor()
        variants = [BranchReadoutIGSTGNN(copy.deepcopy(anchor), arm) for arm in PROBE_ARMS]
        first, second = variants
        self.assertEqual(tensor_tree_hash(first.head.state_dict()),
            tensor_tree_hash(second.head.state_dict()))
        self.assertEqual(first.trainable_parameter_counts(), second.trainable_parameter_counts())
        self.assertEqual(sum(p.numel() for p in first.head.parameters()), 8256)
        self.assertEqual(list(first.head.state_dict()), ['input.weight', 'input.bias', 'output.weight'])
        self.assertFalse(bool(first.head.output.weight.any()))
        self.assertTrue(all(p.requires_grad for model in variants for p in model.parameters()))
        control = BranchReadoutIGSTGNN(copy.deepcopy(anchor), 'R_CONTINUE')
        self.assertIsNone(control.head)
        self.assertEqual(control.trainable_parameter_counts()['added'], 0)
        self.assertEqual(first.param_num()-control.param_num(), 8256)
        for model in (control, first, second):
            restored = BranchReadoutIGSTGNN(copy.deepcopy(anchor), model.arm)
            restored.load_state_dict(model.state_dict(), strict=True)
            self.assertEqual(tensor_tree_hash(model.state_dict()), tensor_tree_hash(restored.state_dict()))

    def test_zero_readout_recovers_exact_r_and_rng_train_eval_grad_and_no_grad(self):
        anchor = self.anchor()
        x, report = self.batch()
        for arm in ARMS:
            for training in (False, True):
                for grad_enabled in (False, True):
                    with self.subTest(arm=arm, training=training, grad_enabled=grad_enabled):
                        reference = copy.deepcopy(anchor).train(training)
                        model = BranchReadoutIGSTGNN(copy.deepcopy(anchor), arm).train(training)
                        with torch.set_grad_enabled(grad_enabled):
                            torch.manual_seed(843)
                            expected = reference(x, incident_data=report)
                            expected_rng = rng_fingerprint()
                            torch.manual_seed(843)
                            actual = model(x, incident_data=report)
                        self.assertTrue(torch.equal(expected, actual))
                        self.assertEqual(expected_rng, rng_fingerprint())
                        self.assertEqual(tensor_tree_hash(reference.state_dict()),
                            tensor_tree_hash(model.anchor.state_dict()))
                        model.assert_cache_clear()

    def test_capture_reads_native_layer_sums_in_order_and_executed_tiid_only_once(self):
        model = BranchReadoutIGSTGNN(self.anchor(), 'BRANCH_READOUT').eval()
        x, report = self.batch()
        layers, tiid, graphs = [], [], []
        handles = [layer.register_forward_hook(
            lambda module, inputs, output: layers.append((output[1], output[2])))
            for layer in model.original.layers]
        handles.append(model.original.tiid_module.register_forward_hook(
            lambda module, inputs, output: tiid.append(output)))
        handles.append(model.original.dynamic_graph_constructor.register_forward_pre_hook(
            lambda module, inputs: graphs.append(True)))
        model.start_diagnostics()
        try:
            features = model.capture_features(x, incident_data=report)
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(set(features), {'base', 'S', 'Q'})
        self.assertEqual((len(layers), len(tiid), len(graphs)), (5, 1, 1))
        D, I = sum([pair[0] for pair in layers]), sum([pair[1] for pair in layers])
        self.assertTrue(torch.equal(features['Q'], D-I))
        self.assertTrue(torch.equal(features['S'], tiid[0]))
        self.assertFalse(torch.equal(features['S'], D+I))
        self.assertEqual(tuple(features['S'].shape), (2, 4, 4, 256))
        self.assertEqual(tuple(features['base'].shape), (2, 12, 4, 1))
        self.assertTrue(all(value.requires_grad for value in features.values()))
        stats = model.diagnostics()
        self.assertEqual(stats['complete_forwards'], 1)
        self.assertEqual(stats['captured_forwards'], 1)
        self.assertEqual(stats['layer_observations'], 5)
        self.assertEqual(stats['tiid_observations'], 1)
        self.assertEqual(stats['readout_calls'], 0)
        model.assert_cache_clear()

    def test_capture_without_incidents_uses_ordered_native_sum_and_no_tiid_call(self):
        model = BranchReadoutIGSTGNN(self.anchor(), 'R_CONTINUE').eval()
        x, _ = self.batch()
        layers = []
        handles = [layer.register_forward_hook(
            lambda module, inputs, output: layers.append((output[1], output[2])))
            for layer in model.original.layers]
        try:
            with patch.object(model.original.tiid_module, 'forward',
                    side_effect=AssertionError('TIID must not run without incidents')):
                features = model.capture_features(x)
        finally:
            for handle in handles:
                handle.remove()
        D, I = sum([pair[0] for pair in layers]), sum([pair[1] for pair in layers])
        self.assertTrue(torch.equal(features['S'], D+I))
        self.assertTrue(torch.equal(features['Q'], D-I))
        model.assert_cache_clear()

    def test_s_is_pre_relu_actual_tiid_output_and_is_not_recomputed(self):
        model = BranchReadoutIGSTGNN(self.anchor(), 'BRANCH_READOUT').eval()
        x, report = self.batch()
        observed = []
        def negative_tiid(forecast, context):
            actual = forecast-(forecast.detach().abs().max()+1.)
            observed.append(actual)
            return actual
        with patch.object(model.original.tiid_module, 'forward', side_effect=negative_tiid):
            features = model.capture_features(x, incident_data=report)
        self.assertEqual(len(observed), 1)
        self.assertTrue(bool((features['S'] < 0).all()))
        self.assertTrue(torch.equal(features['S'], observed[0]))
        self.assertFalse(torch.equal(features['S'], torch.relu(observed[0])))
        model.assert_cache_clear()

    def test_head_reads_exact_prespecified_information_and_block_gap_mapping(self):
        S = torch.arange(48., dtype=torch.float64).reshape(2, 2, 4, 3)/20.
        Q = torch.flip(S, dims=(-1,))-.9
        for arm in PROBE_ARMS:
            head = ReadoutHead(3, 3, arm, hidden=2).double()
            with torch.no_grad():
                head.input.weight.copy_(torch.arange(12., dtype=torch.float64).reshape(2, 6)/50.)
                head.input.bias.copy_(torch.tensor([.2, -.3], dtype=torch.float64))
                head.output.weight.copy_(torch.tensor([[1., -.5], [.3, .8], [-.7, .4]],
                    dtype=torch.float64))
            joined = torch.cat((S, torch.tanh(S if arm == 'SUM_CONTROL' else Q)), -1)
            reference = torch.tanh(joined@head.input.weight.T+head.input.bias)@head.output.weight.T
            blocks = head.output(head.activation(head.input(joined)))
            torch.testing.assert_close(blocks, reference, atol=1e-14, rtol=1e-14)
            actual = head(S, Q)
            self.assertEqual(tuple(actual.shape), (2, 6, 4, 1))
            for batch in range(2):
                for block in range(2):
                    for node in range(4):
                        for gap in range(3):
                            self.assertEqual(float(actual[batch, block*3+gap, node, 0]),
                                float(blocks[batch, block, node, gap]))
            changed = head(S, Q+17.)
            if arm == 'SUM_CONTROL':
                self.assertTrue(torch.equal(actual, changed))
            else:
                self.assertGreater(float((actual-changed).abs().sum()), 0.)

    def test_readout_rejects_malformed_feature_and_block_shapes(self):
        head = ReadoutHead(3, 3, 'BRANCH_READOUT')
        valid = torch.ones(1, 4, 2, 3)
        for S, Q in ((valid.squeeze(0), valid), (valid, valid[..., :2]),
                (valid[..., :2], valid[..., :2]), (valid, valid.double()),
                (valid[:, :0], valid[:, :0])):
            with self.assertRaises(ValueError):
                head(S, Q)
        for invalid in (torch.ones(1, 2, 3), torch.ones(1, 0, 2, 3), None):
            with self.assertRaises(ValueError):
                reshape_blocks(invalid)

    def test_probe_freezes_anchor_eval_but_trains_head_without_changing_native_state(self):
        x, report = self.batch()
        for arm in ARMS:
            model = BranchReadoutIGSTGNN(self.anchor().train(), arm, stage='probe').train()
            before = tensor_tree_hash(model.anchor.state_dict())
            self.assertTrue(model.training)
            self.assertTrue(all(not module.training for module in model.anchor.modules()))
            self.assertTrue(all(not p.requires_grad for p in model.anchor.parameters()))
            self.assertEqual(model.trainable_parameter_counts()['anchor'], 0)
            features = model.capture_features(x, incident_data=report)
            self.assertTrue(all(not value.requires_grad for value in features.values()))
            if model.head is not None:
                self.assertTrue(model.head.training)
                before_head = tensor_tree_hash(model.head.state_dict())
                optimizer = torch.optim.Adam(model.head.parameters(), lr=.01)
                optimizer.zero_grad(set_to_none=True)
                prediction = model(x, incident_data=report)
                (prediction-.3).square().mean().backward()
                self.assertTrue(all(p.grad is None for p in model.anchor.parameters()))
                self.assertGreater(float(model.head.output.weight.grad.abs().sum()), 0.)
                optimizer.step()
                self.assertNotEqual(before_head, tensor_tree_hash(model.head.state_dict()))
            self.assertEqual(before, tensor_tree_hash(model.anchor.state_dict()))
            model.eval().train()
            self.assertTrue(all(not module.training for module in model.anchor.modules()))
            model.assert_cache_clear()

    def test_probe_anchor_call_matches_source_flags_without_enabling_autograd(self):
        anchor = self.anchor()
        reference = copy.deepcopy(anchor).requires_grad_(True).eval()
        model = BranchReadoutIGSTGNN(copy.deepcopy(anchor), 'R_CONTINUE', stage='probe').train()
        x, report = self.batch()
        x = x.detach().requires_grad_(True)
        observed = []
        def observe(module, inputs, output):
            self.assertTrue(all(parameter.requires_grad for parameter in module.parameters()))
            self.assertTrue(all(parameter.grad is None for parameter in module.parameters()))
            self.assertTrue(all(not child.training for child in module.modules()))
            self.assertFalse(torch.is_grad_enabled())
            self.assertFalse(output.requires_grad)
            observed.append(True)
        handle = model.anchor.register_forward_hook(observe)
        before, rng = tensor_tree_hash(model.anchor.state_dict()), rng_fingerprint()
        try:
            for caller_grad_enabled in (False, True):
                with self.subTest(caller_grad_enabled=caller_grad_enabled):
                    with torch.no_grad():
                        expected = reference(x, incident_data=report)
                    with torch.set_grad_enabled(caller_grad_enabled):
                        features = model.capture_features(x, incident_data=report)
                        self.assertEqual(torch.is_grad_enabled(), caller_grad_enabled)
                    self.assertTrue(torch.equal(expected, features['base']))
                    self.assertTrue(all(not value.requires_grad for value in features.values()))
                    self.assertTrue(all(not parameter.requires_grad and parameter.grad is None
                        for parameter in model.anchor.parameters()))
                    self.assertTrue(all(not child.training for child in model.anchor.modules()))
                    self.assertEqual(before, tensor_tree_hash(model.anchor.state_dict()))
                    self.assertEqual(rng, rng_fingerprint())
                    self.assertIsNone(x.grad)
                    model.assert_cache_clear()
        finally:
            handle.remove()
        self.assertEqual(observed, [True, True])

    def test_probe_anchor_failure_restores_flags_grad_context_and_observation_hooks(self):
        model = BranchReadoutIGSTGNN(self.anchor(), 'BRANCH_READOUT', stage='probe').train()
        x, report = self.batch()
        before, rng = tensor_tree_hash(model.anchor.state_dict()), rng_fingerprint()
        observed = []
        def fail(*args, **kwargs):
            self.assertTrue(all(parameter.requires_grad for parameter in model.anchor.parameters()))
            self.assertFalse(torch.is_grad_enabled())
            self.assertTrue(all(not child.training for child in model.anchor.modules()))
            observed.append(True)
            raise RuntimeError('probe anchor deliberate failure')
        for caller_grad_enabled in (False, True):
            with self.subTest(caller_grad_enabled=caller_grad_enabled):
                with torch.set_grad_enabled(caller_grad_enabled):
                    with patch.object(model.original.embedding, 'forward', side_effect=fail):
                        with self.assertRaisesRegex(RuntimeError, 'probe anchor deliberate failure'):
                            model.capture_features(x, incident_data=report)
                    self.assertEqual(torch.is_grad_enabled(), caller_grad_enabled)
                self.assertTrue(all(not parameter.requires_grad and parameter.grad is None
                    for parameter in model.anchor.parameters()))
                self.assertTrue(all(not child.training for child in model.anchor.modules()))
                self.assertEqual(before, tensor_tree_hash(model.anchor.state_dict()))
                self.assertEqual(rng, rng_fingerprint())
                model.assert_cache_clear()
        self.assertEqual(observed, [True, True])
        self.assertEqual(tuple(model(x, incident_data=report).shape), (2, 12, 4, 1))
        model.assert_cache_clear()

    def test_probe_anchor_rejects_trainability_gradients_and_training_modes_at_boundary(self):
        model = BranchReadoutIGSTGNN(self.anchor(), 'BRANCH_READOUT', stage='probe').eval()
        x, report = self.batch()
        parameter = next(model.anchor.parameters())
        parameter.requires_grad_(True)
        with self.assertRaisesRegex(RuntimeError, 'must be frozen without gradients'):
            model.capture_features(x, incident_data=report)
        self.assertTrue(parameter.requires_grad)
        model.assert_cache_clear()
        parameter.requires_grad_(False)
        parameter.grad = torch.ones_like(parameter)
        with self.assertRaisesRegex(RuntimeError, 'must be frozen without gradients'):
            model.capture_features(x, incident_data=report)
        self.assertTrue(bool((parameter.grad == 1).all()))
        model.assert_cache_clear()
        parameter.grad = None
        model.original.embedding.train()
        with self.assertRaisesRegex(RuntimeError, 'entirely in evaluation mode'):
            model.capture_features(x, incident_data=report)
        model.assert_cache_clear()
        model.eval()
        self.assertEqual(tuple(model(x, incident_data=report).shape), (2, 12, 4, 1))

    def test_probe_source_compatible_forward_restores_freezing_before_head_backward(self):
        model = BranchReadoutIGSTGNN(self.anchor(), 'BRANCH_READOUT', stage='probe').train()
        x, report = self.batch()
        x = x.detach().requires_grad_(True)
        before, rng = tensor_tree_hash(model.anchor.state_dict()), rng_fingerprint()
        observed = []
        def observe_head(module, inputs, output):
            self.assertTrue(torch.is_grad_enabled())
            self.assertTrue(all(not parameter.requires_grad and parameter.grad is None
                for parameter in model.anchor.parameters()))
            self.assertTrue(all(not feature.requires_grad for feature in inputs))
            self.assertTrue(output.requires_grad)
            observed.append(True)
        handle = model.head.register_forward_hook(observe_head)
        try:
            with torch.enable_grad():
                prediction = model(x, incident_data=report)
                (prediction-.3).square().mean().backward()
        finally:
            handle.remove()
        self.assertEqual(observed, [True])
        self.assertTrue(all(not parameter.requires_grad and parameter.grad is None
            for parameter in model.anchor.parameters()))
        self.assertGreater(float(model.head.output.weight.grad.abs().sum()), 0.)
        self.assertTrue(all(bool(torch.isfinite(parameter.grad).all())
            for parameter in model.head.parameters() if parameter.grad is not None))
        self.assertEqual(before, tensor_tree_hash(model.anchor.state_dict()))
        self.assertEqual(rng, rng_fingerprint())
        self.assertIsNone(x.grad)
        model.assert_cache_clear()

    def test_probe_shared_features_head_matches_model_prediction(self):
        model = BranchReadoutIGSTGNN(self.anchor(), 'BRANCH_READOUT', stage='probe').eval()
        with torch.no_grad():
            model.head.output.weight.normal_(std=.1)
        x, report = self.batch()
        features = model.capture_features(x, incident_data=report)
        expected = features['base']+model.head(features['S'], features['Q'])
        self.assertTrue(torch.equal(expected, model(x, incident_data=report)))
        model.assert_cache_clear()

    def test_joint_native_forecasts_and_new_output_then_interior_receive_gradients(self):
        x, report = self.batch()
        target = torch.linspace(.1, 1., 12)[None, :, None, None]
        for arm in ARMS:
            model = BranchReadoutIGSTGNN(self.anchor(), arm).eval()
            (model(x, incident_data=report)-target).square().mean().backward()
            for layer in model.original.layers:
                branch = list(layer.dif_layer.forecast_branch.parameters())
                self.assertTrue(all(p.requires_grad for p in branch))
                self.assertGreater(sum(float(p.grad.abs().sum()) for p in branch
                    if p.grad is not None), 0.)
            if model.head is not None:
                self.assertGreater(float(model.head.output.weight.grad.abs().sum()), 0.)
                with torch.no_grad():
                    model.head.output.weight.normal_(std=.1)
                model.zero_grad(set_to_none=True)
                (model(x, incident_data=report)-target).square().mean().backward()
                self.assertGreater(float(model.head.input.weight.grad.abs().sum()), 0.)
                self.assertGreater(float(model.head.input.bias.grad.abs().sum()), 0.)
            self.assertTrue(all(bool(torch.isfinite(p.grad).all()) for p in model.parameters()
                if p.grad is not None))
            model.assert_cache_clear()

    def test_nonzero_readout_forward_equals_one_base_plus_mapped_correction(self):
        model = BranchReadoutIGSTGNN(self.anchor(), 'BRANCH_READOUT').eval()
        with torch.no_grad():
            model.head.output.weight.normal_(std=.1)
        x, report = self.batch()
        features = model.capture_features(x, incident_data=report)
        expected = features['base']+model.head(features['S'], features['Q'])
        model.start_diagnostics()
        actual = model(x, incident_data=report)
        self.assertTrue(torch.equal(actual, expected))
        stats = model.diagnostics()
        self.assertEqual(stats['complete_forwards'], 1)
        self.assertEqual(stats['readout_calls'], 1)
        self.assertGreater(stats['correction_abs_max'], 0.)
        self.assertTrue(stats['cache_clear'])

    def test_labels_are_ignored_and_history_incidents_are_not_mutated(self):
        x, report = self.batch()
        label = torch.randn(2, 12, 4, 1)
        before = tensor_tree_hash((x, report, label))
        for arm in ARMS:
            model = BranchReadoutIGSTGNN(self.anchor(), arm).eval()
            if model.head is not None:
                with torch.no_grad():
                    model.head.output.weight.normal_(std=.1)
            first = model(x, label=label, incident_data=report)
            self.assertTrue(torch.equal(first, model(x, label=label*1e9, incident_data=report)))
            a = model.capture_features(x, label=label, incident_data=report)
            b = model.capture_features(x, label=label*1e9, incident_data=report)
            self.assertTrue(all(torch.equal(a[key], b[key]) for key in a))
            self.assertEqual(before, tensor_tree_hash((x, report, label)))
            model.assert_cache_clear()

    def test_errors_remove_all_temporary_hooks_and_next_forward_is_fresh(self):
        model = BranchReadoutIGSTGNN(self.anchor(), 'BRANCH_READOUT').eval()
        x, report = self.batch()
        watched = list(model.original.layers)+[model.original.tiid_module]
        initial = [(len(module._forward_pre_hooks), len(module._forward_hooks)) for module in watched]
        for module in (model.original.layers[2], model.original.tiid_module,
                model.original.out_fc_2, model.head):
            with patch.object(module, 'forward', side_effect=RuntimeError('deliberate exception')):
                with self.assertRaisesRegex(RuntimeError, 'deliberate exception'):
                    model(x, incident_data=report)
            model.assert_cache_clear()
            self.assertEqual(initial, [(len(module._forward_pre_hooks), len(module._forward_hooks))
                for module in watched])
        expected = model(x, incident_data=report)
        model(x*2., incident_data=report)
        self.assertTrue(torch.equal(expected, model(x, incident_data=report)))
        model.assert_cache_clear()

    def test_reentry_capture_and_forward_do_not_destroy_enclosing_observations(self):
        model = BranchReadoutIGSTGNN(self.anchor(), 'BRANCH_READOUT').eval()
        x, report = self.batch()
        observed = []
        def nested(module, inputs):
            if observed:
                return
            observed.append(True)
            for call in (model, model.capture_features):
                with self.assertRaisesRegex(RuntimeError, 'Reentrant'):
                    call(x, incident_data=report)
        handle = model.original.embedding.register_forward_pre_hook(nested)
        try:
            prediction = model(x, incident_data=report)
        finally:
            handle.remove()
        self.assertEqual(tuple(prediction.shape), (2, 12, 4, 1))
        self.assertEqual(observed, [True])
        model.assert_cache_clear()

    def test_repeated_forward_without_capture_control_adds_no_observation_hooks(self):
        model = BranchReadoutIGSTGNN(self.anchor(), 'R_CONTINUE').eval()
        x, report = self.batch()
        calls = []
        with patch.object(model.original.tiid_module, 'register_forward_hook',
                side_effect=AssertionError('R control must not install observation hooks')):
            calls.append(model(x, incident_data=report))
            calls.append(model(x, incident_data=report))
        self.assertTrue(torch.equal(calls[0], calls[1]))
        model.assert_cache_clear()


if __name__ == '__main__':
    unittest.main()
