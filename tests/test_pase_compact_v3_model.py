"""Server/CI contracts for physically compact PASE and its matched controls."""
import unittest

import torch

from experiments.pase_mlp_v2.model import build_model as v2_model
from experiments.pase_mlp_v2.model import parameter_accounting as v2_parameters
from experiments.pase_compact_v3.model import (
    FIRST_WEIGHT, build_model, compact_state_dict, capture_exchange_diagnostics,
    parameter_accounting,
)
from experiments.pase_compact_v3.protocol import (
    CORE_ARMS, PREDECLARED_SEEDS, arm_spec, configuration, protocol_record,
)
from tests.test_position_state_gw16_model import fixture


class PASECompactV3ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(53)
        self.data, self.config, self.history, self.event = fixture()
        self.config.update(exchange_rank=8, horizon=3)

    def model(self, arm='FULL', seed=2025):
        torch.manual_seed(seed)
        return build_model(self.data, self.config, arm, 'cpu').eval()

    def previous(self, arm='NO_POSITION_INPUT', seed=2025):
        torch.manual_seed(seed)
        return v2_model(self.data, self.config, arm, 'cpu').eval()

    def assert_outputs_close(self, actual, expected):
        self.assertEqual(set(actual), set(expected))
        for key in actual:
            if actual[key].is_floating_point():
                torch.testing.assert_close(actual[key], expected[key], rtol=2e-5, atol=2e-5,
                                           msg=lambda message: key+': '+message)
            else:
                self.assertTrue(torch.equal(actual[key], expected[key]), key)

    def test_conversion_slices_only_p_columns_retains_all_other_tensor_bits(self):
        previous = self.previous()
        source = previous.state_dict()
        converted = compact_state_dict(source)
        self.assertEqual(set(source), set(converted))
        for key in source:
            expected = source[key][:, 32:] if key == FIRST_WEIGHT else source[key]
            self.assertTrue(torch.equal(converted[key], expected), key)
            self.assertNotEqual(converted[key].data_ptr(), source[key].data_ptr(), key)
            self.assertFalse(converted[key].requires_grad)
        self.assertEqual(converted[FIRST_WEIGHT].shape, (16, 96))
        model = self.model()
        model.load_state_dict(converted, strict=True)
        self.assert_outputs_close(model(self.history, self.event), previous(self.history, self.event))

    def test_conversion_rejects_wrong_envelopes_shapes_and_extra_scorer_keys(self):
        source = self.previous().state_dict()
        bad_states = [dict(model=source), {key: value for key, value in source.items()
                                        if key != 'head.exchange.network.2.bias'}]
        for key, value in ((FIRST_WEIGHT, torch.zeros(16, 96)),
                           ('head.exchange.network.0.bias', torch.zeros(15)),
                           ('head.exchange.network.2.weight', torch.zeros(2, 15)),
                           ('head.exchange.network.2.bias', torch.zeros(2, dtype=torch.long)),
                           ('head.exchange.extra', torch.zeros(1))):
            bad_states.append(dict(source, **{key: value}))
        for state in bad_states:
            with self.subTest(keys=list(state)[-2:]), self.assertRaises(ValueError):
                compact_state_dict(state)
        with self.assertRaises(ValueError):
            compact_state_dict(source, width=True)
        # The consumer must also reject unrelated tensors outside the scorer.
        model = self.model()
        converted = compact_state_dict(source)
        converted['unrelated'] = torch.zeros(1)
        with self.assertRaises(RuntimeError):
            model.load_state_dict(converted, strict=True)

    def test_all_arms_share_sliced_initial_tensors_and_legacy_following_rng(self):
        previous = self.previous()
        following_rng = torch.get_rng_state().clone()
        expected = compact_state_dict(previous.state_dict())
        for arm in CORE_ARMS:
            with self.subTest(arm=arm):
                model = self.model(arm)
                self.assertTrue(torch.equal(torch.get_rng_state(), following_rng))
                self.assertEqual(set(model.state_dict()), set(expected))
                for name, value in expected.items():
                    self.assertTrue(torch.equal(model.state_dict()[name], value), name)
                self.assertEqual(model.head.exchange.network[0].in_features, 96)
                self.assertEqual(model.config, previous.config)

    def test_compaction_keeps_training_dropout_stream_and_close_gradients(self):
        compact, previous = self.model(), self.previous()
        for model in (compact, previous):
            model.train()
            for module in model.modules():
                if isinstance(module, torch.nn.Dropout):
                    module.p = .15
        outputs, following_rngs = [], []
        for model in (compact, previous):
            torch.manual_seed(941)
            output = model(self.history, self.event)
            (output['prediction'].square().mean()+.2*output['regional'].square().mean()).backward()
            outputs.append(output)
            following_rngs.append(torch.get_rng_state().clone())
        self.assert_outputs_close(*outputs)
        self.assertTrue(torch.equal(*following_rngs))
        original = dict(previous.named_parameters())
        for name, value in compact.named_parameters():
            reference = original[name].grad
            if value.grad is None or reference is None:
                self.assertIs(value.grad, reference, name)
            else:
                if name == FIRST_WEIGHT:
                    reference = reference[:, 32:]
                torch.testing.assert_close(value.grad, reference, rtol=2e-4, atol=2e-4,
                                           msg=lambda message: name+': '+message)

    def test_disabled_direction_matches_zero_message_intervention_and_gradients(self):
        for arm, disabled, inactive_row in (('UP_ONLY', 'lift_map', 1),
                                           ('DOWN_ONLY', 'pool_map', 0)):
            with self.subTest(arm=arm):
                full, control = self.model(), self.model(arm)
                handle = getattr(full.head, disabled).register_forward_hook(
                    lambda module, args, output: torch.zeros_like(output))
                try:
                    expected = full(self.history, self.event)
                finally:
                    handle.remove()
                actual = control(self.history, self.event)
                for key in expected:
                    self.assertTrue(torch.equal(actual[key], expected[key]), key)
                (actual['prediction'].square().mean()+.2*actual['regional'].square().mean()).backward()
                self.assertTrue(all(not parameter.requires_grad and parameter.grad is None
                                    for parameter in getattr(control.head, disabled).parameters()))
                final = control.head.exchange.network[-1]
                self.assertTrue(torch.equal(final.weight.grad[inactive_row],
                                            torch.zeros_like(final.weight.grad[inactive_row])))
                self.assertEqual(final.bias.grad[inactive_row].item(), 0.)
                self.assertGreater(final.weight.grad[1-inactive_row].abs().sum().item(), 0.)
                self.assertGreater(control.head.exchange.network[0].weight.grad.abs().sum().item(), 0.)

    def test_scorer_ignores_direct_position_but_reads_conditions_and_current_states(self):
        model = self.model()
        position = torch.randn(6, 32, requires_grad=True)
        inputs = [torch.randn(2, 6, 32, requires_grad=True) for _ in range(3)]
        expected = model.head.exchange(position, *inputs, model.geometry)
        actual = model.head.exchange(position*100, *inputs, model.geometry)
        self.assertTrue(torch.equal(actual, expected))
        expected.square().sum().backward()
        self.assertIsNone(position.grad)
        for value in inputs:
            self.assertGreater(value.grad.abs().sum().item(), 0.)
        torch.testing.assert_close(model.geometry.pool(actual),
                                   torch.zeros(2, 2, 2), rtol=0, atol=1e-6)

    def test_other_position_paths_incident_and_centering_are_retained(self):
        model, previous = self.model(), self.previous()
        for name in ('position', 'encoder', 'origin_condition', 'incident'):
            for key, expected in getattr(previous, name).state_dict().items():
                self.assertTrue(torch.equal(getattr(model, name).state_dict()[key], expected))
        output = model(self.history, self.event)
        handle = model.position.register_forward_hook(
            lambda module, args, value: torch.zeros_like(value))
        try:
            removed_position = model(self.history, self.event)
        finally:
            handle.remove()
        self.assertFalse(torch.equal(output['prediction'], removed_position['prediction']))
        no_report = dict(self.event, report_present=torch.zeros_like(self.event['report_present']))
        self.assertFalse(torch.equal(output['prediction'], model(self.history, no_report)['prediction']))
        torch.testing.assert_close(model.geometry.pool(output['beta']),
                                   torch.zeros_like(output['alpha']), rtol=0, atol=1e-6)
        local_updates = []
        handle = model.head.local_readout.register_forward_hook(
            lambda module, args, value: local_updates.append(args[0].detach().clone()))
        try:
            model(self.history, self.event)
        finally:
            handle.remove()
        for value in local_updates:
            torch.testing.assert_close(model.geometry.pool(value), torch.zeros(2, 2, 32),
                                       rtol=0, atol=1e-6)

    def test_parameter_counts_physically_remove_512_and_keep_shared_hidden_rows(self):
        old = v2_parameters(self.previous())
        full = parameter_accounting(self.model())
        self.assertEqual(full['registered'], old['registered']-512)
        self.assertEqual(full['coefficient_registered'], 1586)
        self.assertEqual(full['coefficient_active'], 1586)
        self.assertEqual(full['structurally_active'], full['registered'])
        for arm in ('UP_ONLY', 'DOWN_ONLY'):
            record = parameter_accounting(self.model(arm))
            self.assertEqual(record['registered'], full['registered'])
            self.assertEqual(record['coefficient_active'], 1569)
            self.assertEqual(record['inactive_registered'], 17+32*32)

    def test_diagnostics_do_not_modify_forward_and_report_disabled_directions(self):
        for arm in CORE_ARMS:
            model = self.model(arm)
            expected = model(self.history, self.event)
            with capture_exchange_diagnostics(model) as record:
                actual = model(self.history, self.event)
            for key in expected:
                self.assertTrue(torch.equal(actual[key], expected[key]), key)
            self.assertEqual(len(record['coefficient_rms']), 3)
            self.assertEqual(record['active_directions'], [arm != 'DOWN_ONLY', arm != 'UP_ONLY'])
            for direction, enabled in enumerate(record['active_directions']):
                if not enabled:
                    self.assertTrue(all(row[direction] == 0 for row in record['message_rms']))
                    self.assertTrue(all(row[direction] == 0 for row in record['coefficient_rms']))
            self.assertEqual(len(model.head.exchange._forward_hooks), 0)

    def test_registry_has_only_two_new_training_controls_and_original_recipes(self):
        self.assertEqual(CORE_ARMS, ('FULL', 'UP_ONLY', 'DOWN_ONLY'))
        self.assertEqual(PREDECLARED_SEEDS, (2025,))
        self.assertEqual(protocol_record()['new_training_arms'], ['UP_ONLY', 'DOWN_ONLY'])
        self.assertEqual(protocol_record()['legacy_reuse_arms'], {'FULL': 'NO_POSITION_INPUT'})
        from experiments.pase_mlp_v2.protocol import configuration as old_configuration
        for dataset in ('Contra_Costa', 'Alameda', 'Orange'):
            for arm in CORE_ARMS:
                self.assertEqual(configuration(arm, dataset), old_configuration('FULL', dataset))
                self.assertFalse(arm_spec(arm)['explicit_position'])
                self.assertTrue(arm_spec(arm)['interior_center'])
        for invalid in ('NO_EXCHANGE', 'NO_POSITION_INPUT', 'ORIGIN_EXCHANGE', 'CONCAT_MLP'):
            with self.assertRaises(ValueError):
                arm_spec(invalid)
        with self.assertRaises(ValueError):
            configuration(nodes=495)


if __name__ == '__main__':
    unittest.main()
