"""CPU contracts for standalone score interactions; no quality claims/data."""
import inspect
import unittest

import torch

from src.models.incident_component_interaction import InteractionScoreResidual


def tiny_inputs(*, requires_grad=False):
    generator = torch.Generator().manual_seed(93)
    shapes = ((2, 4, 3, 10, 6), (2, 12, 3, 1), (2, 12, 3, 1),
              (12, 4), (2, 3, 19))
    tensors = [torch.randn(shape, generator=generator, requires_grad=requires_grad)
               for shape in shapes]
    support = torch.tensor([[[True], [False], [True]], [[True], [False], [False]]])
    return (*tensors, support)


def activate(module):
    with torch.no_grad():
        module.history_readout.weight.fill_(.3)
        module.report_readout.weight.fill_(.4)


class InteractionScoreResidualTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(17)
        self.module = InteractionScoreResidual(component_dim=6)
        self.inputs = tiny_inputs()

    @torch.no_grad()
    def test_exact_zero_initialization_random_factors_and_parameter_count(self):
        result = self.module(*self.inputs, collect=True)
        for key in ('history_scores', 'report_scores'):
            self.assertEqual(result[key].shape, (2, 12, 3, 10))
            self.assertTrue(torch.equal(result[key], torch.zeros_like(result[key])))
        self.assertEqual(sum(p.numel() for p in self.module.parameters()), 16*6+344)
        self.assertIsNone(self.module(*self.inputs)['diagnostics'])
        self.assertIsNone(self.module.report_projection.bias)
        self.assertIsNone(self.module.history_readout.bias)
        self.assertIsNone(self.module.report_readout.bias)
        for name, parameter in self.module.named_parameters():
            if not name.endswith('readout.weight'):
                self.assertGreater(float(parameter.detach().abs().sum()), 0., name)
        for value in result['diagnostics'].values():
            if isinstance(value, torch.Tensor):
                self.assertFalse(value.requires_grad)

    def test_first_step_readouts_and_second_step_factor_and_input_gradients(self):
        values = tiny_inputs(requires_grad=True)
        target = torch.randn(2, 12, 3, 10, generator=torch.Generator().manual_seed(66))
        result = self.module(*values)
        scores = sum(result[name]-result[name].mean(-1, keepdim=True)
                     for name in ('history_scores', 'report_scores'))
        loss = (scores-target).square().mean()
        loss.backward()
        for name, parameter in self.module.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()), name)
            if name.endswith('readout.weight'):
                self.assertGreater(float(parameter.grad.abs().sum()), 0., name)
            else:
                self.assertEqual(float(parameter.grad.abs().sum()), 0., name)
        with torch.no_grad():
            self.module.history_readout.weight.add_(-.1*self.module.history_readout.weight.grad)
            self.module.report_readout.weight.add_(-.1*self.module.report_readout.weight.grad)
        self.module.zero_grad(set_to_none=True)
        for value in values[:-1]:
            value.grad = None
        result = self.module(*values)
        scores = sum(result[name]-result[name].mean(-1, keepdim=True)
                     for name in ('history_scores', 'report_scores'))
        (scores-target).square().mean().backward()
        for name, parameter in self.module.named_parameters():
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()), name)
            self.assertGreater(float(parameter.grad.abs().sum()), 0., name)
        for name, value in zip(('components', 'history', 'prediction', 'horizon', 'report'), values[:-1]):
            self.assertIsNotNone(value.grad, name)
            self.assertTrue(bool(torch.isfinite(value.grad).all()), name)
            self.assertGreater(float(value.grad.abs().sum()), 0., name)
        self.assertEqual(float(values[4].grad[~values[5].squeeze(-1)].abs().sum()), 0.)

    @torch.no_grad()
    def test_report_support_missing_and_disabled_are_exact_zero(self):
        activate(self.module)
        on = self.module(*self.inputs)
        supported = self.inputs[-1][:, None, :, :].expand_as(on['report_scores'])
        self.assertTrue(torch.equal(on['report_scores'][~supported],
                                    torch.zeros_like(on['report_scores'][~supported])))
        self.assertGreater(float(on['report_scores'][supported].abs().max()), 0.)
        absent = self.module(*self.inputs[:4])
        off = self.module(*self.inputs[:4], report_features='ignored', support='ignored', report_enabled=False)
        zero_report = self.module(*self.inputs[:4], torch.zeros_like(self.inputs[4]), self.inputs[5])
        zero_support = self.module(*self.inputs[:-1], torch.zeros_like(self.inputs[-1]))
        for result in (absent, off, zero_report, zero_support):
            self.assertTrue(torch.equal(result['report_scores'], torch.zeros_like(result['report_scores'])))
            self.assertTrue(torch.equal(result['history_scores'], on['history_scores']))
        history_off = self.module(*self.inputs, history_enabled=False)
        self.assertTrue(torch.equal(history_off['history_scores'], torch.zeros_like(on['history_scores'])))
        self.assertTrue(torch.equal(history_off['report_scores'], on['report_scores']))

    @torch.no_grad()
    def test_history_and_full_prediction_each_change_history_scores(self):
        activate(self.module)
        expected = self.module(*self.inputs)['history_scores']
        for index in (1, 2):
            changed = list(self.inputs)
            changed[index] = changed[index].clone()
            changed[index][:, -1] += .8
            actual = self.module(*changed)['history_scores']
            self.assertGreater(float((actual-expected).abs().max()), 1e-6)
        changed = list(self.inputs)
        changed[4] = changed[4]*3+1
        self.assertTrue(torch.equal(self.module(*changed)['history_scores'], expected))

    @torch.no_grad()
    def test_evidence_exact_definition_and_twelve_slot_prediction_not_averaged(self):
        activate(self.module)
        components = self.inputs[0][:, :1].expand(-1, 4, -1, -1, -1).clone()
        history = torch.arange(12.).view(1, 12, 1, 1).expand(2, -1, 3, -1).clone()
        prediction = torch.full_like(history, 12.)
        horizon = torch.zeros_like(self.inputs[3])
        args = (components, history, prediction, horizon, *self.inputs[4:])
        baseline = self.module(*args, collect=True)
        evidence = baseline['diagnostics']['history_features']
        expected = torch.tensor([11., 5.5, 11., 1., 6.5])
        torch.testing.assert_close(evidence, expected.view(1, 1, 1, 5).expand_as(evidence), rtol=0, atol=0)
        changed = prediction.clone()
        changed[:, 1] += 1
        changed[:, 2] -= 1  # block mean is unchanged, individual slots are not.
        result = self.module(components, history, changed, horizon, *self.inputs[4:])
        difference = result['history_scores']-baseline['history_scores']
        self.assertGreater(float(difference[:, 1:3].abs().max()), 1e-6)
        self.assertTrue(torch.equal(difference[:, 0], torch.zeros_like(difference[:, 0])))
        self.assertTrue(torch.equal(difference[:, 3:], torch.zeros_like(difference[:, 3:])))

    @torch.no_grad()
    def test_report_component_interaction_survives_centering_and_identical_boundary(self):
        activate(self.module)
        on = self.module(*self.inputs)['report_scores']
        centered = on-on.mean(-1, keepdim=True)
        self.assertGreater(float(centered.abs().max()), 1e-6)
        self.assertLess(float(centered.sum(-1).abs().max()), 2e-7)
        changed = list(self.inputs)
        changed[0] = changed[0].clone()
        changed[0][..., 3, :] += 2
        delta = self.module(*changed)['report_scores']-on
        self.assertGreater(float(delta[..., 3].abs().max()), 1e-6)
        other = [i for i in range(10) if i != 3]
        self.assertTrue(torch.equal(delta[..., other], torch.zeros_like(delta[..., other])))
        identical = list(self.inputs)
        identical[0] = identical[0][..., :1, :].expand(-1, -1, -1, 10, -1).clone()
        identical_scores = self.module(*identical)['report_scores']
        self.assertTrue(torch.equal(identical_scores, identical_scores[..., :1].expand_as(identical_scores)))
        torch.testing.assert_close(identical_scores-identical_scores.mean(-1, keepdim=True),
                                   torch.zeros_like(identical_scores), atol=1e-7, rtol=0)

    @torch.no_grad()
    def test_labels_absent_from_api_and_no_input_or_state_mutation(self):
        names = inspect.signature(self.module.forward).parameters
        self.assertNotIn('label', names)
        self.assertNotIn('labels', names)
        self.assertNotIn('target', names)
        with self.assertRaises(TypeError):
            self.module(*self.inputs, label=torch.zeros(1))
        state = {name: value.clone() for name, value in self.module.state_dict().items()}
        values = [value.clone() for value in self.inputs]
        self.module(*self.inputs, collect=True)
        for name, value in self.module.state_dict().items():
            self.assertTrue(torch.equal(value, state[name]))
        for value, before in zip(self.inputs, values):
            self.assertTrue(torch.equal(value, before))

    @torch.no_grad()
    def test_invalid_shapes_dtypes_finite_values_flags_and_gap(self):
        for component_dim in (0, -1, True, 1.5):
            with self.subTest(component_dim=component_dim), self.assertRaises(ValueError):
                InteractionScoreResidual(component_dim)
        for kwargs in ({'rank': 4}, {'rank': True}, {'horizon_dim': 5},
                       {'horizon_dim': 4.0}, {'report_dim': 18}, {'report_dim': True}):
            with self.subTest(constructor_kwargs=kwargs), self.assertRaises(ValueError):
                InteractionScoreResidual(6, **kwargs)
        replacements = {0: self.inputs[0][:, :3], 1: self.inputs[1][:, :11],
                        2: self.inputs[2][..., 0], 3: self.inputs[3][:, :3],
                        4: self.inputs[4][..., :18], 5: self.inputs[5].float()}
        for index, value in replacements.items():
            changed = list(self.inputs)
            changed[index] = value
            with self.subTest(shape_index=index), self.assertRaises(ValueError):
                self.module(*changed)
        for index in range(5):
            for value in (float('nan'), float('inf')):
                changed = list(self.inputs)
                changed[index] = changed[index].clone()
                changed[index].reshape(-1)[0] = value
                with self.subTest(finite_index=index, value=value), self.assertRaisesRegex(ValueError, 'finite'):
                    self.module(*changed)
        for index in range(5):
            changed = list(self.inputs)
            changed[index] = changed[index].double() if index else changed[index].long()
            with self.subTest(dtype_index=index), self.assertRaises(ValueError):
                self.module(*changed)
        for kwargs in ({'history_enabled': 1}, {'report_enabled': None}, {'collect': 'yes'},
                       {'gap': 2}, {'gap': True}, {'gap': 3.0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.module(*self.inputs, **kwargs)
        with self.assertRaisesRegex(ValueError, 'explicit support'):
            self.module(*self.inputs[:5])
        with self.assertRaisesRegex(ValueError, 'missing'):
            self.module(*self.inputs[:4], support=self.inputs[5])
        absent = self.module(*self.inputs[:4], support=torch.zeros_like(self.inputs[5]))
        self.assertTrue(torch.equal(absent['report_scores'], torch.zeros_like(absent['report_scores'])))


if __name__ == '__main__':
    unittest.main()
