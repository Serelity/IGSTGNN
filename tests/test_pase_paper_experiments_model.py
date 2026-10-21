"""Server numerical checks for paper variants; no local Python execution needed."""
import math
import unittest
from unittest.mock import patch

import torch

from experiments.position_state_gw16.model import build_model as build_gw16
from experiments.position_state_gw16.geometry import RegionGeometry
from experiments.pase_paper_experiments.model import (
    ConcatMLPExchange, NoPositionExchange, RegionalSoftmaxExchange,
    build_model, capture_exchange_diagnostics, parameter_accounting,
)
from experiments.pase_paper_experiments.protocol import (
    CORE_ARMS, SUPPLEMENTARY_ARMS, arm_spec, configuration, protocol_record,
)
from tests.test_position_state_gw16_model import fixture


class PaperModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(67)
        self.data, self.config, self.x, self.event = fixture()
        self.config.update(exchange_rank=8, horizon=3)

    def model(self, arm='FULL', seed=2025):
        torch.manual_seed(seed)
        return build_model(self.data, self.config, arm, 'cpu').eval()

    def test_full_preserves_every_tensor_and_original_forward_exactly(self):
        torch.manual_seed(2025)
        original = build_gw16(self.data, self.config, 'STATE_EXCHANGE', 'cpu').eval()
        full = self.model()
        self.assertIs(type(full), type(original))
        self.assertEqual(full.arm, 'STATE_EXCHANGE')
        self.assertEqual(set(full.state_dict()), set(original.state_dict()))
        for name, expected in original.state_dict().items():
            self.assertTrue(torch.equal(full.state_dict()[name], expected), name)
        actual, expected = full(self.x, self.event), original(self.x, self.event)
        for name in expected:
            self.assertTrue(torch.equal(actual[name], expected[name]), name)

    def test_common_initial_tensors_and_following_rng_identical_across_nine_arms(self):
        full = self.model()
        expected = {name: value.clone() for name, value in full.state_dict().items()
                    if not name.startswith('head.exchange.')}
        expected_rng = torch.get_rng_state().clone()
        for arm in CORE_ARMS:
            with self.subTest(arm=arm):
                model = self.model(arm)
                self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
                for name, value in expected.items():
                    self.assertTrue(torch.equal(model.state_dict()[name], value), name)
                if arm not in ('CONCAT_MLP', 'NO_POSITION_BRANCH'):
                    for name, value in full.head.exchange.state_dict().items():
                        self.assertTrue(torch.equal(model.head.exchange.state_dict()[name], value))
                elif arm == 'NO_POSITION_BRANCH':
                    for name, value in full.head.exchange.dynamic.state_dict().items():
                        self.assertTrue(torch.equal(model.head.exchange.dynamic.state_dict()[name], value))

    def test_full_training_dropout_and_gradients_equal_original(self):
        torch.manual_seed(19)
        original = build_gw16(self.data, self.config, 'STATE_EXCHANGE', 'cpu').train()
        full = self.model(seed=19).train()
        original.config['dropout'] = full.config['dropout'] = .1
        # The fixture builds zero-dropout modules; set the same nonzero value in
        # both models to exercise the real training RNG stream as well.
        for model in (original, full):
            for module in model.modules():
                if isinstance(module, torch.nn.Dropout):
                    module.p = .1
        torch.manual_seed(888)
        expected = original(self.x, self.event)['prediction']
        expected.square().mean().backward()
        torch.manual_seed(888)
        actual = full(self.x, self.event)['prediction']
        actual.square().mean().backward()
        self.assertTrue(torch.equal(actual, expected))
        for (name, a), (other, b) in zip(original.named_parameters(), full.named_parameters()):
            self.assertEqual(name, other)
            self.assertTrue(torch.equal(a.grad, b.grad), name)

    def test_disabled_directions_equal_zero_message_intervention(self):
        for arm, disabled in (('NO_EXCHANGE', ('pool_map', 'lift_map')),
                              ('UP_ONLY', ('lift_map',)), ('DOWN_ONLY', ('pool_map',))):
            with self.subTest(arm=arm):
                reference, model = self.model(), self.model(arm)
                handles = [getattr(reference.head, name).register_forward_hook(
                    lambda module, args, value: torch.zeros_like(value)) for name in disabled]
                try:
                    expected = reference(self.x, self.event)
                finally:
                    for handle in handles:
                        handle.remove()
                actual = model(self.x, self.event)
                self.assertTrue(torch.equal(actual['prediction'], expected['prediction']))
                actual['prediction'].square().mean().backward()
                for name in disabled:
                    self.assertTrue(all(p.grad is None and not p.requires_grad
                                        for p in getattr(model.head, name).parameters()))
                if arm == 'NO_EXCHANGE':
                    self.assertTrue(all(p.grad is None for p in model.head.exchange.parameters()))
                else:
                    rank = self.config['exchange_rank']
                    # Both scalar directions occupy independent output-row blocks.
                    grad = model.head.exchange.dynamic.weight.grad
                    inactive = grad[rank:] if arm == 'UP_ONLY' else grad[:rank]
                    self.assertTrue(torch.equal(inactive, torch.zeros_like(inactive)))

    def test_origin_changes_only_coefficient_states_not_message_values(self):
        full, origin = self.model(), self.model('ORIGIN_EXCHANGE')
        origin_states, messages = [], []
        original_exchange = origin.head.exchange.forward

        def exchange(position, condition, coarse, local, geometry):
            origin_states.append((coarse.detach().clone(), local.detach().clone(),
                                  condition.detach().clone()))
            return original_exchange(position, condition, coarse, local, geometry)

        import experiments.pase_paper_experiments.model as paper
        real_pool = paper.pool_contrast

        def pool(geometry, local, weights):
            messages.append(local.detach().clone())
            return real_pool(geometry, local, weights)

        with patch.object(origin.head.exchange, 'forward', side_effect=exchange), \
                patch.object(paper, 'pool_contrast', side_effect=pool):
            actual = origin(self.x, self.event)
        expected = full(self.x, self.event)
        self.assertTrue(torch.equal(actual['prediction'][:, 0], expected['prediction'][:, 0]))
        self.assertFalse(torch.equal(actual['prediction'][:, 1:], expected['prediction'][:, 1:]))
        for coarse, local, condition in origin_states[1:]:
            self.assertTrue(torch.equal(coarse, origin_states[0][0]))
            self.assertTrue(torch.equal(local, origin_states[0][1]))
        self.assertFalse(torch.equal(origin_states[1][2], origin_states[0][2]))
        self.assertFalse(torch.equal(messages[0], messages[1]))

    def test_no_interior_center_keeps_initial_and_final_projection(self):
        model = self.model('NO_INTERIOR_CENTER')
        raw_states = []
        def observe(module, args, output):
            raw_states.append(args[0].detach().clone())
        handle = model.head.local_readout.register_forward_hook(observe)
        try:
            output = model(self.x, self.event)
        finally:
            handle.remove()
        self.assertGreater(model.geometry.pool(raw_states[-1]).abs().max().item(), 1e-5)
        torch.testing.assert_close(model.geometry.pool(output['beta']),
            torch.zeros(2, 3, 2, 1), atol=1e-6, rtol=0)
        torch.testing.assert_close(model.geometry.pool(output['prediction']), output['regional'],
                                   atol=2e-5, rtol=1e-6)
        full = self.model()
        self.assertTrue(torch.equal(model.encoder.state_dict()['node.weight'],
                                    full.encoder.state_dict()['node.weight']))

    def test_no_position_branch_is_only_explicit_coefficient_path(self):
        model = self.model('NO_POSITION_BRANCH')
        geometry, width = model.geometry, self.config['hidden']
        position = torch.randn(6, width)
        values = [torch.randn(2, 6, width) for _ in range(3)]
        one = model.head.exchange(position, *values, geometry)
        two = model.head.exchange(position*100, *values, geometry)
        self.assertTrue(torch.equal(one, two))
        expected = torch.full((2, 8), 1/math.sqrt(8))
        self.assertTrue(torch.equal(model.head.exchange.shared_position.detach(), expected))
        full = self.model()
        for name in ('position', 'encoder', 'origin_condition'):
            for key, tensor in getattr(full, name).state_dict().items():
                self.assertTrue(torch.equal(getattr(model, name).state_dict()[key], tensor))

    def test_regional_softmax_matches_explicit_group_formula_and_gradients(self):
        geometry = RegionGeometry([0, 0, 1, 1, 1, 2])
        exchange = RegionalSoftmaxExchange(4, 2)
        position, condition = torch.randn(6, 4), torch.randn(2, 6, 4)
        coarse, local = torch.randn(2, 6, 4), torch.randn(2, 6, 4)
        actual = exchange(position, condition, coarse, local, geometry)
        phi = exchange.position(position).reshape(6, 2, 2)
        psi = exchange.dynamic(torch.cat([condition, coarse, local], -1)).reshape(2, 6, 2, 2)
        score = (phi[None]*psi).sum(-1)/math.sqrt(2)
        expected = torch.zeros_like(actual)
        for region in range(3):
            mask = geometry.assignment == region
            expected[:, mask] = mask.sum()*torch.softmax(score[:, mask], dim=1)
        expected = geometry.center(expected)
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(actual[:, -1], torch.zeros_like(actual[:, -1])))
        torch.testing.assert_close(geometry.pool(actual), torch.zeros(2, 3, 2), atol=1e-6, rtol=0)
        actual.square().sum().backward()
        self.assertGreater(exchange.position.weight.grad.abs().sum().item(), 0)
        self.assertGreater(exchange.dynamic.weight.grad.abs().sum().item(), 0)

    def test_replacement_counts_and_active_partial_rows_are_explicit(self):
        expected = {'FULL':2064, 'CONCAT_MLP':2098, 'REGIONAL_SOFTMAX':2064,
                    'NO_POSITION_BRANCH':1568}
        full_count = parameter_accounting(self.model())
        for arm, count in expected.items():
            model = self.model(arm)
            record = parameter_accounting(model)
            self.assertEqual(record['coefficient_registered'], count)
            self.assertEqual(record['registered']-full_count['registered'], count-2064)
            self.assertEqual(record['structurally_active'], record['registered'])
            self.assertEqual(model.capacity['actual_head_parameters'],
                             sum(parameter.numel() for parameter in model.head.parameters()))
        for arm in ('UP_ONLY', 'DOWN_ONLY'):
            record = parameter_accounting(self.model(arm))
            self.assertEqual(record['registered'], full_count['registered'])
            self.assertEqual(record['coefficient_active'], 1032)
            self.assertEqual(record['registered']-record['structurally_active'], 1032+32*32)
        record = parameter_accounting(self.model('NO_EXCHANGE'))
        self.assertEqual(record['coefficient_active'], 0)
        self.assertEqual(record['registered']-record['structurally_active'], 2064+2*32*32)

    def test_diagnostics_do_not_change_predictions_and_mark_disabled_paths(self):
        for arm in ('FULL', 'NO_EXCHANGE', 'UP_ONLY', 'DOWN_ONLY'):
            with self.subTest(arm=arm):
                model = self.model(arm)
                expected = model(self.x, self.event)['prediction']
                with capture_exchange_diagnostics(model) as record:
                    actual = model(self.x, self.event)['prediction']
                self.assertTrue(torch.equal(actual, expected))
                self.assertEqual(len(record['coefficient_rms']), 3)
                self.assertEqual(len(record['message_rms']), 3)
                for direction, enabled in enumerate(record['active_directions']):
                    if not enabled:
                        self.assertTrue(all(row[direction] == 0 for row in record['message_rms']))
                        self.assertTrue(all(row[direction] == 0 for row in record['coefficient_rms']))
                self.assertEqual(len(model.head.exchange._forward_hooks), 0)

    def test_all_arms_finite_differentiable_and_preserve_output_contract(self):
        for arm in CORE_ARMS:
            with self.subTest(arm=arm):
                model = self.model(arm)
                output = model(self.x, self.event)
                self.assertEqual(output['prediction'].shape, (2, 3, 6, 1))
                self.assertTrue(all(torch.isfinite(value).all() for value in output.values()))
                output['prediction'].abs().mean().backward()
                self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                                    for p in model.encoder.parameters()))
                torch.testing.assert_close(model.geometry.pool(output['beta']),
                    torch.zeros(2, 3, 2, 1), atol=1e-6, rtol=0)

    def test_protocol_is_fixed_and_supplementary_arms_are_plans_only(self):
        self.assertEqual(len(CORE_ARMS), 9)
        self.assertEqual(protocol_record()['predeclared_seeds'], [2025, 2026, 2027])
        config = configuration('FULL')
        self.assertEqual(config['max_epochs'], 100)
        self.assertEqual(config['nodes'], 496)
        for arm in CORE_ARMS:
            self.assertEqual(configuration(arm), config)
        for arm in SUPPLEMENTARY_ARMS:
            self.assertFalse(arm_spec(arm)['implemented'])
            with self.assertRaises(ValueError):
                configuration(arm)
        with self.assertRaises(ValueError):
            self.model('UNKNOWN')
        with self.assertRaises(ValueError):
            configuration('FULL', nodes=495)
        with self.assertRaises(ValueError):
            configuration('FULL', dataset='invented')


if __name__ == '__main__':
    unittest.main()
