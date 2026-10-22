"""Remote/server numerical contracts for the selected MLP model and ablations."""
import unittest
from unittest.mock import patch

import torch

from experiments.pase_paper_experiments.model import build_model as legacy_model
from experiments.pase_mlp_v2.model import (
    build_model, capture_exchange_diagnostics, parameter_accounting,
)
from experiments.pase_mlp_v2.protocol import (
    CORE_ARMS, PREDECLARED_SEEDS, arm_spec, configuration, protocol_record,
)
from tests.test_position_state_gw16_model import fixture


class PASEMLPv2ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(53)
        self.data, self.config, self.x, self.event = fixture()
        self.config.update(exchange_rank=8, horizon=3)

    def model(self, arm='FULL', seed=2025):
        torch.manual_seed(seed)
        return build_model(self.data, self.config, arm, 'cpu').eval()

    def legacy(self, arm='CONCAT_MLP', seed=2025):
        torch.manual_seed(seed)
        return legacy_model(self.data, self.config, arm, 'cpu').eval()

    def assert_outputs_equal(self, actual, expected):
        self.assertEqual(set(actual), set(expected))
        for key in expected:
            self.assertTrue(torch.equal(actual[key], expected[key]), key)

    def assert_gradient_equal(self, actual, expected, *, exclude_exchange=False):
        expected = dict(expected.named_parameters())
        for name, parameter in actual.named_parameters():
            if exclude_exchange and name.startswith('head.exchange.'):
                continue
            gradient, reference = parameter.grad, expected[name].grad
            if gradient is None or reference is None:
                self.assertIs(gradient, reference, name)
            else:
                self.assertTrue(torch.equal(gradient, reference), name)

    def nonzero_dropout(self, *models):
        for model in models:
            model.train()
            for module in model.modules():
                if isinstance(module, torch.nn.Dropout):
                    module.p = .15

    def test_full_keeps_legacy_type_forward_tensors_and_following_rng(self):
        original = self.legacy()
        following_rng = torch.get_rng_state().clone()
        full = self.model()
        self.assertTrue(torch.equal(torch.get_rng_state(), following_rng))
        self.assertIs(type(full), type(original))
        self.assertIs(type(full.head), type(original.head))
        self.assertEqual(full.arm, 'CONCAT_MLP')
        self.assertEqual(full.paper_arm, 'FULL')
        self.assertEqual(full.config, original.config)
        self.assertEqual(full.capacity, original.capacity)
        self.assertEqual(set(full.state_dict()), set(original.state_dict()))
        for name, expected in original.state_dict().items():
            self.assertTrue(torch.equal(full.state_dict()[name], expected), name)
        self.assert_outputs_equal(full(self.x, self.event), original(self.x, self.event))

    def test_full_training_dropout_gradients_optimizer_and_rng_match_legacy(self):
        full, original = self.model(), self.legacy()
        self.nonzero_dropout(full, original)
        optimizers = [torch.optim.Adam(model.parameters(), lr=.002) for model in (full, original)]
        outputs, states = [], []
        for model, optimizer in zip((full, original), optimizers):
            torch.manual_seed(942)
            output = model(self.x, self.event)
            loss = output['prediction'].square().mean()+.2*output['regional'].square().mean()
            loss.backward()
            optimizer.step()
            outputs.append(output)
            states.append(torch.get_rng_state().clone())
        self.assert_outputs_equal(outputs[0], outputs[1])
        self.assert_gradient_equal(full, original)
        self.assertTrue(torch.equal(states[0], states[1]))
        for name, parameter in full.state_dict().items():
            self.assertTrue(torch.equal(parameter, original.state_dict()[name]), name)

    def test_all_arms_start_with_identical_full_tensors_and_following_rng(self):
        full = self.model()
        following_rng = torch.get_rng_state().clone()
        for arm in CORE_ARMS:
            with self.subTest(arm=arm):
                model = self.model(arm)
                self.assertTrue(torch.equal(torch.get_rng_state(), following_rng))
                self.assertEqual(set(model.state_dict()), set(full.state_dict()))
                for name, expected in full.state_dict().items():
                    self.assertTrue(torch.equal(model.state_dict()[name], expected), name)

    def test_no_exchange_matches_legacy_active_function_initialization_and_training(self):
        model, original = self.model('NO_EXCHANGE'), self.legacy('NO_EXCHANGE')
        for name, expected in original.state_dict().items():
            if not name.startswith('head.exchange.'):
                self.assertTrue(torch.equal(model.state_dict()[name], expected), name)
        self.nonzero_dropout(model, original)
        outputs, rngs = [], []
        for value in (model, original):
            torch.manual_seed(112)
            output = value(self.x, self.event)
            (output['prediction'].square().mean()+.2*output['regional'].square().mean()).backward()
            outputs.append(output)
            rngs.append(torch.get_rng_state().clone())
        self.assert_outputs_equal(outputs[0], outputs[1])
        self.assert_gradient_equal(model, original, exclude_exchange=True)
        self.assertTrue(torch.equal(rngs[0], rngs[1]))
        self.assertTrue(all(not p.requires_grad and p.grad is None
                            for p in model.head.exchange.parameters()))

    def test_disabled_directions_equal_zero_message_intervention_on_full(self):
        for arm, disabled in (('NO_EXCHANGE', ('pool_map', 'lift_map')),
                              ('UP_ONLY', ('lift_map',)), ('DOWN_ONLY', ('pool_map',))):
            with self.subTest(arm=arm):
                full, model = self.model(), self.model(arm)
                handles = [getattr(full.head, name).register_forward_hook(
                    lambda module, args, output: torch.zeros_like(output)) for name in disabled]
                try:
                    expected = full(self.x, self.event)
                finally:
                    for handle in handles:
                        handle.remove()
                self.assert_outputs_equal(model(self.x, self.event), expected)
                model(self.x, self.event)['prediction'].square().mean().backward()
                for name in disabled:
                    self.assertTrue(all(not p.requires_grad and p.grad is None
                                        for p in getattr(model.head, name).parameters()))
                if arm != 'NO_EXCHANGE':
                    final_grad = model.head.exchange.network[-1].weight.grad
                    inactive = final_grad[1] if arm == 'UP_ONLY' else final_grad[0]
                    active = final_grad[0] if arm == 'UP_ONLY' else final_grad[1]
                    self.assertTrue(torch.equal(inactive, torch.zeros_like(inactive)))
                    self.assertGreater(active.abs().sum().item(), 0)
                    self.assertGreater(model.head.exchange.network[0].weight.grad.abs().sum().item(), 0)

    def test_no_position_zeros_only_p_input_and_keeps_same_registered_weights(self):
        full, model = self.model(), self.model('NO_POSITION_INPUT')
        width = self.config['hidden']
        position = torch.randn(6, width)
        values = [torch.randn(2, 6, width) for _ in range(3)]
        first = model.head.exchange(position, *values, model.geometry)
        second = model.head.exchange(position*100, *values, model.geometry)
        expected = full.head.exchange(torch.zeros_like(position), *values, full.geometry)
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(first, expected))
        first.square().sum().backward()
        gradient = model.head.exchange.network[0].weight.grad
        self.assertTrue(torch.equal(gradient[:, :width], torch.zeros_like(gradient[:, :width])))
        self.assertGreater(gradient[:, width:].abs().sum().item(), 0)
        # The intervention does not remove position from the original encoder or
        # origin condition; those complete modules remain present and identical.
        for name in ('position', 'encoder', 'origin_condition'):
            for key, tensor in getattr(full, name).state_dict().items():
                self.assertTrue(torch.equal(getattr(model, name).state_dict()[key], tensor))

    def test_origin_freezes_only_coefficient_state_views_not_current_messages(self):
        full, model = self.model(), self.model('ORIGIN_EXCHANGE')
        views, messages = [], []
        actual_exchange = model.head.exchange.forward

        def scorer(position, condition, coarse, local, geometry):
            views.append((coarse.detach().clone(), local.detach().clone(), condition.detach().clone()))
            return actual_exchange(position, condition, coarse, local, geometry)

        import experiments.pase_mlp_v2.model as implementation
        actual_pool = implementation.pool_contrast

        def message(geometry, local, weights):
            messages.append(local.detach().clone())
            return actual_pool(geometry, local, weights)

        with patch.object(model.head.exchange, 'forward', side_effect=scorer), \
                patch.object(implementation, 'pool_contrast', side_effect=message):
            actual = model(self.x, self.event)
        expected = full(self.x, self.event)
        self.assertTrue(torch.equal(actual['prediction'][:, 0], expected['prediction'][:, 0]))
        self.assertFalse(torch.equal(actual['prediction'][:, 1:], expected['prediction'][:, 1:]))
        for coarse, local, condition in views[1:]:
            self.assertTrue(torch.equal(coarse, views[0][0]))
            self.assertTrue(torch.equal(local, views[0][1]))
        self.assertFalse(torch.equal(views[1][2], views[0][2]))
        self.assertFalse(torch.equal(messages[0], messages[1]))

    def test_no_interior_center_retains_initial_and_final_projection(self):
        model = self.model('NO_INTERIOR_CENTER')
        local_states = []
        handle = model.head.local_readout.register_forward_hook(
            lambda module, args, output: local_states.append(args[0].detach().clone()))
        try:
            output = model(self.x, self.event)
        finally:
            handle.remove()
        self.assertGreater(model.geometry.pool(local_states[-1]).abs().max().item(), 1e-5)
        torch.testing.assert_close(model.geometry.pool(output['beta']),
            torch.zeros(2, 3, 2, 1), atol=1e-6, rtol=0)
        torch.testing.assert_close(model.geometry.pool(output['prediction']), output['regional'],
                                   atol=2e-5, rtol=1e-6)

    def test_parameter_accounting_handles_shared_hidden_and_zero_position_columns(self):
        full = parameter_accounting(self.model())
        self.assertEqual(full['coefficient_registered'], 2098)
        for arm in CORE_ARMS:
            with self.subTest(arm=arm):
                model = self.model(arm)
                record = parameter_accounting(model)
                self.assertEqual(record['registered'], full['registered'])
                self.assertEqual(record['coefficient_registered'], 2098)
                if arm in ('UP_ONLY', 'DOWN_ONLY'):
                    self.assertEqual(record['coefficient_active'], 2081)
                    self.assertEqual(record['inactive_registered'], 17+32*32)
                elif arm == 'NO_POSITION_INPUT':
                    self.assertEqual(record['coefficient_active'], 2098-16*32)
                    self.assertEqual(record['inactive_registered'], 16*32)
                elif arm == 'NO_EXCHANGE':
                    self.assertEqual(record['coefficient_active'], 0)
                    self.assertEqual(record['inactive_registered'], 2098+2*32*32)
                    self.assertEqual(record['trainable'], record['structurally_active'])
                else:
                    self.assertEqual(record['structurally_active'], record['registered'])

    def test_diagnostics_preserve_predictions_and_cleanup_even_on_failure(self):
        for arm in ('FULL', 'NO_EXCHANGE', 'UP_ONLY', 'DOWN_ONLY'):
            model = self.model(arm)
            expected = model(self.x, self.event)
            with capture_exchange_diagnostics(model) as record:
                actual = model(self.x, self.event)
            self.assert_outputs_equal(actual, expected)
            self.assertEqual(len(record['coefficient_rms']), 3)
            for direction, enabled in enumerate(record['active_directions']):
                if not enabled:
                    self.assertTrue(all(row[direction] == 0 for row in record['message_rms']))
                    self.assertTrue(all(row[direction] == 0 for row in record['coefficient_rms']))
            with self.assertRaisesRegex(RuntimeError, 'sentinel'):
                with capture_exchange_diagnostics(model):
                    raise RuntimeError('sentinel')
            self.assertEqual(len(model.head.exchange._forward_hooks), 0)
            self.assertEqual(len(model.head.pool_map._forward_hooks), 0)

    def test_dataset_vocabulary_extension_still_matches_selected_legacy_full(self):
        self.data.event_counts = dict(type=9, description=41)
        full, original = self.model(), self.legacy()
        self.assertEqual(full.incident.type_embedding.num_embeddings, 10)
        self.assertEqual(full.incident.description_embedding.num_embeddings, 42)
        self.assert_outputs_equal(full(self.x, self.event), original(self.x, self.event))

    def test_fixed_single_seed_protocol_and_unchanged_dataset_recipes(self):
        self.assertEqual(PREDECLARED_SEEDS, (2025,))
        record = protocol_record()
        self.assertEqual(record['predeclared_seeds'], [2025])
        self.assertEqual(record['epochs'], 100)
        self.assertEqual(record['legacy_reuse_arms'],
                         {'FULL': 'CONCAT_MLP', 'NO_EXCHANGE': 'NO_EXCHANGE'})
        from experiments.pase_paper_experiments.protocol import configuration as old_config
        for dataset in ('Contra_Costa', 'Alameda', 'Orange'):
            for arm in CORE_ARMS:
                self.assertEqual(configuration(arm, dataset), old_config('FULL', dataset))
        for invalid in ('CONCAT_MLP', 'NO_POSITION_BRANCH', 'invented'):
            with self.assertRaises(ValueError):
                arm_spec(invalid)
        with self.assertRaises(ValueError):
            configuration(nodes=495)
        with self.assertRaises(ValueError):
            self.model('invented')


if __name__ == '__main__':
    unittest.main()
