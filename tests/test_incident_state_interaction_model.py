"""v12f identity, bounded vector adaptation and frozen-backbone checks."""

import copy
import unittest

import torch

from experiments.chronological import train_incident_strength_gate as base
from src.models.incident_state_interaction import (
    ARMS, IncidentStateVector, attach_adapter,
)
from src.models.incident_strength_gate import attach_gate
from test_architecture_mechanisms import tiny_model, inputs


class StateInteractionModelTests(unittest.TestCase):
    def test_every_arm_replays_native_including_no_support_and_off(self):
        for arm in ARMS:
            for supported in (True, False):
                with self.subTest(arm=arm, supported=supported):
                    model = tiny_model()
                    x, incident = inputs()
                    if not supported:
                        incident['distances'].zero_()
                    before = base.backbone_hash(model)
                    with torch.no_grad():
                        expected = model(x, incident_data=incident)
                        expected_off = model(x, incident_data=None)
                    attach_adapter(model, arm)
                    with torch.no_grad():
                        torch.testing.assert_close(model(x, incident_data=incident), expected,
                                                   atol=0, rtol=0)
                        torch.testing.assert_close(model(x, incident_data=None), expected_off,
                                                   atol=0, rtol=0)
                    base.assert_backbone(model, before)
                    self.assertTrue(torch.equal(model.icsf_module.last_delta,
                                                torch.zeros_like(model.icsf_module.last_delta)))
                    if arm == 'strength':
                        self.assertIsNotNone(model.icsf_module.last_gate)
                    else:
                        self.assertIsNone(model.icsf_module.last_gate)

    def test_only_adapter_learns_and_hidden_gradient_starts_after_first_update(self):
        for arm in ARMS:
            with self.subTest(arm=arm):
                model = tiny_model()
                before = base.backbone_hash(model)
                adapter = attach_adapter(model, arm)
                initial = copy.deepcopy(adapter.state_dict())
                optimizer = torch.optim.Adam(adapter.parameters(), lr=.01)
                x, incident = inputs()
                projection = torch.randn(2, 12, 4, 1)
                hidden = adapter.network[0] if arm == 'strength' else adapter.state
                for step in range(2):
                    optimizer.zero_grad(set_to_none=True)
                    (model(x, incident_data=incident) * projection).sum().backward()
                    self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                        for p in adapter.parameters()))
                    if step == 0:
                        self.assertEqual(hidden.weight.grad.abs().sum().item(), 0.)
                    else:
                        self.assertGreater(hidden.weight.grad.abs().sum().item(), 0.)
                    optimizer.step()
                    base.assert_backbone(model, before)
                self.assertTrue(any(not torch.equal(initial[k], value)
                                    for k, value in adapter.state_dict().items()))

    def test_latent_support_history_and_tiid_context_preserved_after_updates(self):
        for arm in ARMS:
            with self.subTest(arm=arm):
                model = tiny_model()
                native = model.icsf_module
                _, incident = inputs()
                history = torch.randn(2, 12, 4, native.q_proj.in_features)
                tod, dow = torch.randn(2, 4), torch.randn(2, 4)
                with torch.no_grad():
                    expected, context = native(history, incident, None, tod, dow)
                adapter = attach_adapter(model, arm)
                with torch.no_grad():
                    output = adapter.network[-1] if arm == 'strength' else adapter.output
                    output.weight.fill_(.2)
                    output.bias.fill_(.1)
                    actual, actual_context = model.icsf_module(history, incident, None, tod, dow)
                torch.testing.assert_close(actual[:, :-1], history[:, :-1], atol=0, rtol=0)
                torch.testing.assert_close(actual[:, -1, 2:], expected[:, -1, 2:], atol=0, rtol=0)
                self.assertFalse(torch.equal(actual[:, -1, :2], expected[:, -1, :2]))
                for key in context:
                    torch.testing.assert_close(actual_context[key], context[key], atol=0, rtol=0)
                self.assertEqual(model.icsf_module.last_delta[:, 2:].abs().sum().item(), 0.)
                self.assertEqual(model.icsf_module.last_postnorm_delta[:, 2:].abs().sum().item(), 0.)
                self.assertFalse(model.icsf_module.last_delta.requires_grad)
                self.assertFalse(model.icsf_module.last_postnorm_delta.requires_grad)

    def test_strength_matches_original_initialization_and_adam_updates_exactly(self):
        original = tiny_model()
        current = copy.deepcopy(original)
        torch.manual_seed(2025)
        old_gate = attach_gate(original, 'node')
        torch.manual_seed(2025)
        new_gate = attach_adapter(current, 'strength')
        optimizers = [torch.optim.Adam(gate.parameters(), lr=.001)
                      for gate in (old_gate, new_gate)]
        x, incident = inputs()
        target = torch.randn(2, 12, 4, 1)
        support = incident['distances'].abs().sum(-1) > 0
        for _ in range(3):
            predictions = []
            for model, optimizer in zip((original, current), optimizers):
                optimizer.zero_grad(set_to_none=True)
                prediction = model(x, incident_data=incident)
                predictions.append(prediction.detach())
                strength = model.icsf_module.last_gate.squeeze(-1)
                loss = (prediction - target).abs().mean() + .001 * (strength[support] - 1).square().mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.icsf_module.gate.parameters(), 5.)
                optimizer.step()
            torch.testing.assert_close(*predictions, atol=0, rtol=0)
            for name, parameter in old_gate.state_dict().items():
                torch.testing.assert_close(parameter, new_gate.state_dict()[name], atol=0, rtol=0)

    def test_vector_arms_have_paired_initialization_and_identical_capacity(self):
        for hidden in (8, 32):
            torch.manual_seed(2025)
            state = IncidentStateVector(hidden, False)
            torch.manual_seed(2025)
            interaction = IncidentStateVector(hidden, True)
            for name, value in state.state_dict().items():
                torch.testing.assert_close(value, interaction.state_dict()[name], atol=0, rtol=0)
            expected = hidden * (3 * hidden + 5) + hidden * (hidden + 1)
            self.assertEqual(sum(p.numel() for p in state.parameters()), expected)
            self.assertEqual(sum(p.numel() for p in interaction.parameters()), expected)
            if hidden == 32:
                self.assertEqual(expected, 4288)

    def test_only_interaction_unit_residual_responds_to_additional_context_value(self):
        _, incident = inputs()
        hidden = 8
        history = torch.randn(2, 12, 4, hidden)
        value_a = torch.zeros(2, 1, hidden)
        value_b = torch.full_like(value_a, .5)
        for interaction in (False, True):
            adapter = IncidentStateVector(hidden, interaction)
            with torch.no_grad():
                adapter.state.weight.zero_()
                adapter.state.bias.fill_(.4)
                adapter.output.weight.copy_(torch.eye(hidden))
                adapter.output.bias.zero_()
                a = adapter(history, incident, value_a)
                b = adapter(history, incident, value_b)
            self.assertEqual(torch.equal(a, b), not interaction)

    def test_vector_residual_can_change_direction_and_obeys_energy_bound(self):
        for arm in ('state_vector', 'interaction_vector'):
            with self.subTest(arm=arm):
                model = tiny_model()
                adapter = attach_adapter(model, arm)
                x, incident = inputs()
                with torch.no_grad():
                    adapter.output.weight.zero_()
                    adapter.output.bias.copy_(torch.linspace(-1.5, 1.5, adapter.output.out_features))
                    model(x, incident_data=incident)
                wrapper = model.icsf_module
                value, delta = wrapper.last_injection, wrapper.last_delta
                support = incident['distances'].abs().sum(-1) > 0
                norm_squared = value.square().sum(-1)
                self.assertTrue((delta.square().sum(-1) <= norm_squared + 1e-7).all())
                projection = ((delta[support] * value[support]).sum(-1)
                              / norm_squared[support]).unsqueeze(-1) * value[support]
                self.assertGreater((delta[support] - projection).square().sum().item(), 1e-5)
                # The regularizer is relative injection energy, not raw delta magnitude.
                relative_energy = delta[support].square().sum(-1) / norm_squared[support]
                expected = wrapper.last_unit_residual[support].square().mean(-1)
                torch.testing.assert_close(relative_energy, expected)

    def test_zero_value_forces_exact_zero_delta_even_with_nonzero_unit_residual(self):
        for arm in ('state_vector', 'interaction_vector'):
            with self.subTest(arm=arm):
                model = tiny_model()
                adapter = attach_adapter(model, arm)
                x, incident = inputs()
                with torch.no_grad():
                    model.icsf_module.base.v_proj.weight.zero_()
                    adapter.output.bias.fill_(1.)
                    model(x, incident_data=incident)
                wrapper = model.icsf_module
                self.assertGreater(wrapper.last_unit_residual.abs().sum().item(), 0.)
                self.assertEqual(wrapper.last_injection.abs().sum().item(), 0.)
                self.assertEqual(wrapper.last_delta.abs().sum().item(), 0.)
                self.assertEqual(wrapper.last_postnorm_delta.abs().sum().item(), 0.)
                wrapper.clear_observations()
                for key in ('last_gate', 'last_unit_residual', 'last_delta', 'last_injection',
                            'last_postnorm_delta', 'last_context_modifier'):
                    self.assertIsNone(getattr(wrapper, key))


if __name__ == '__main__':
    unittest.main()
