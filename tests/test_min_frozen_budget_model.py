"""Frozen MIN invariants and a real forecast-loss gradient to its adapter."""
import copy
import unittest

import torch
from torch import nn

from src.models.incident_spatial_event_budget import from_minimal
from src.models.incident_spatial_frozen_budget import (
    ADAPTER_PREFIX, assert_frozen, build_frozen_location, frozen_state_hash,
    set_frozen_train_mode, trainable_named_parameters,
)
from src.models.incident_spatial_minimal import from_residual
from src.models.incident_spatial_residual import tensor_state_hash
from tests.test_b3_r import controlled_forecast
from tests.test_min_event_budget_model import event_inputs, minimal, nonzero_adapter


class FrozenLocationBudgetModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_initial_on_and_off_exactly_match_source_and_preserve_source_rng(self):
        base = minimal()
        x, report = event_inputs()
        source, rng = tensor_state_hash(base.state_dict()), torch.get_rng_state().clone()
        source_flags = [(name, module.training) for name, module in base.named_modules()]
        model, certificate = build_frozen_location(base)
        reference_model = copy.deepcopy(base).requires_grad_(False).eval()
        with torch.inference_mode():
            reference = reference_model(x)
            # Freezing alone can select a different CPU floating-point kernel.
            # The exact intervention reference must have matching freeze flags.
            torch.testing.assert_close(reference, base(x), atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(model(x, incident_data=report), reference, atol=0, rtol=0)
            torch.testing.assert_close(model(x, incident_data=report, report_enabled=False),
                                       reference, atol=0, rtol=0)
        self.assertEqual(source, tensor_state_hash(base.state_dict()))
        self.assertEqual(source_flags, [(name, module.training) for name, module in base.named_modules()])
        torch.testing.assert_close(rng, torch.get_rng_state(), atol=0, rtol=0)
        self.assertEqual(certificate['frozen_state_sha256'], source)
        self.assertEqual(certificate['frozen_parameters'], sum(p.numel() for p in base.parameters()))
        self.assertEqual(certificate['trainable_parameters'], sum(p.numel() for _, p in trainable_named_parameters(model)))
        self.assertTrue(assert_frozen(model, source))

    def test_nonzero_adapter_off_still_matches_source_exactly(self):
        base = minimal()
        model, certificate = build_frozen_location(base)
        nonzero_adapter(model)
        x, report = event_inputs()
        reference_model = copy.deepcopy(base).requires_grad_(False).eval()
        with torch.inference_mode():
            reference = reference_model(x)
            actual = model(x, incident_data=report, report_enabled=False)
        torch.testing.assert_close(actual, reference, atol=0, rtol=0)
        self.assertTrue(assert_frozen(model, certificate['frozen_state_sha256']))

    def test_forecast_loss_updates_only_adapter_and_backbone_buffers_stay_identical(self):
        with controlled_forecast() as (residual, _, _):
            base = from_residual(residual)[0].eval()
            model, certificate = build_frozen_location(base)
            adapter = model.event_budget_adapter
            with torch.no_grad():
                adapter.A[0].weight.fill_(.005)
                adapter.A[0].bias.fill_(1.)
            before = tensor_state_hash(adapter.state_dict())
            optimizer = torch.optim.SGD([p for _, p in trainable_named_parameters(model)], lr=.005)
            x, report = event_inputs()
            for step in range(2):
                set_frozen_train_mode(model)
                optimizer.zero_grad(set_to_none=True)
                prediction = model(x, incident_data=report)
                self.assertTrue(prediction.requires_grad)
                (prediction-prediction.detach()-1).abs().mean().backward()
                self.assertGreater(float(adapter.A[-1].weight.grad.abs().sum()), 0.)
                if step == 0:
                    self.assertEqual(float(adapter.A[0].weight.grad.abs().sum()), 0.)
                else:
                    self.assertGreater(float(adapter.A[0].weight.grad.abs().sum()), 0.)
                self.assertTrue(all(p.grad is None for name, p in model.named_parameters()
                                    if not name.startswith(ADAPTER_PREFIX)))
                optimizer.step()
                self.assertTrue(assert_frozen(model, certificate['frozen_state_sha256']))
            self.assertNotEqual(tensor_state_hash(adapter.state_dict()), before)
            self.assertEqual(frozen_state_hash(model), tensor_state_hash(base.state_dict()))
            self.assertTrue(any(isinstance(module, nn.modules.batchnorm._BatchNorm)
                                for module in model.modules()))

    def test_optimizer_allowlist_is_only_complete_location_adapter_and_has_no_semantics(self):
        model, _ = build_frozen_location(minimal())
        allowed = trainable_named_parameters(model)
        self.assertFalse(hasattr(model.event_budget_adapter, 'semantic_encoder'))
        self.assertEqual({id(p) for _, p in allowed},
                         {id(p) for p in model.event_budget_adapter.parameters()})
        self.assertTrue(all(name.startswith(ADAPTER_PREFIX) for name, _ in allowed))
        model.event_budget_adapter.A[-1].bias.requires_grad_(False)
        with self.assertRaisesRegex(ValueError, 'Only the complete'):
            trainable_named_parameters(model)

    def test_mode_guard_rejects_root_and_nested_backbone_training(self):
        model, certificate = build_frozen_location(minimal())
        for module in (model, model.layers[0]):
            with self.subTest(module=type(module).__name__):
                module.train()
                with self.assertRaisesRegex(ValueError, 'evaluation mode'):
                    assert_frozen(model, certificate['frozen_state_sha256'])
                set_frozen_train_mode(model)
                self.assertTrue(model.event_budget_adapter.training)
                self.assertFalse(model.training)
                self.assertTrue(assert_frozen(model, certificate['frozen_state_sha256']))
        model.eval()
        self.assertTrue(assert_frozen(model, certificate['frozen_state_sha256']))

    def test_guard_rejects_backbone_grad_flag_and_unexpected_stale_gradient(self):
        model, certificate = build_frozen_location(minimal())
        parameter = next(p for name, p in model.named_parameters() if not name.startswith(ADAPTER_PREFIX))
        parameter.requires_grad_(True)
        with self.assertRaisesRegex(ValueError, 'Only the complete'):
            assert_frozen(model, certificate['frozen_state_sha256'])
        parameter.requires_grad_(False)
        parameter.grad = torch.zeros_like(parameter)
        with self.assertRaisesRegex(ValueError, 'unexpected gradient'):
            assert_frozen(model, certificate['frozen_state_sha256'])
        with self.assertRaisesRegex(ValueError, 'unexpected gradient'):
            set_frozen_train_mode(model)
            assert_frozen(model, certificate['frozen_state_sha256'])

    def test_guard_rejects_backbone_parameter_and_running_buffer_mutation(self):
        original, certificate = build_frozen_location(minimal())
        parameter_model = copy.deepcopy(original)
        with torch.no_grad():
            next(p for name, p in parameter_model.named_parameters()
                 if not name.startswith(ADAPTER_PREFIX)).view(-1)[0].add_(1.)
        with self.assertRaisesRegex(ValueError, 'parameters or buffers changed'):
            assert_frozen(parameter_model, certificate['frozen_state_sha256'])
        buffer_model = copy.deepcopy(original)
        bn = next(module for module in buffer_model.modules()
                  if isinstance(module, nn.modules.batchnorm._BatchNorm))
        bn.running_mean.add_(.1)
        with self.assertRaisesRegex(ValueError, 'parameters or buffers changed'):
            assert_frozen(buffer_model, certificate['frozen_state_sha256'])

    def test_guard_hash_contains_graph_and_source_conversion_rejects_already_adapted_model(self):
        model, certificate = build_frozen_location(minimal())
        model.internal_spatial_graph.operator_mask[3] = False
        with self.assertRaisesRegex(ValueError, 'parameters or buffers changed'):
            assert_frozen(model, certificate['frozen_state_sha256'])
        with self.assertRaisesRegex(ValueError, 'unadapted restricted MIN'):
            build_frozen_location(model)

    def test_semantic_architecture_is_rejected_by_frozen_helpers(self):
        semantic, _ = from_minimal(minimal(), 'semantic')
        for action in (lambda: frozen_state_hash(semantic),
                       lambda: set_frozen_train_mode(semantic),
                       lambda: trainable_named_parameters(semantic)):
            with self.assertRaisesRegex(ValueError, 'location-only'):
                action()


if __name__ == '__main__':
    unittest.main()
