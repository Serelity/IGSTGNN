"""Server-only numerical tests for GW17 causality and shared-state invariants."""
import unittest

import torch

from experiments.observed_state_gw17.model import (
    ARMS, advance_calendar, build_model, calendar_indices, parameter_count)
from tests.test_position_state_gw16_model import fixture


class ObservedStateModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2025)
        self.data, self.config, self.history, self.event = fixture()
        self.config.update(observation_step_initial=.1, history_loss_weight=.1)
        # Distinct nodes, regions, samples and historical changes are intentional.
        ramp = torch.arange(12).view(1, 12, 1)*.07
        self.history[..., 0] += ramp+torch.arange(6).view(1, 1, 6)*.11

    def model(self, arm='INNOVATION_STATE', nonzero_readout=False):
        model = build_model(self.data, self.config, arm, 'cpu').eval()
        if nonzero_readout:
            with torch.no_grad():
                model.core.coarse_readout.weight.copy_(torch.linspace(-.03, .04, self.config['hidden'])[None])
                model.core.local_readout.weight.copy_(torch.linspace(.04, -.02, self.config['hidden'])[None])
        return model

    def test_initial_all_tensors_and_active_parameter_counts_match(self):
        torch.manual_seed(91)
        one = self.model('OBSERVATION_CONTROL')
        torch.manual_seed(91)
        two = self.model('INNOVATION_STATE')
        self.assertEqual(set(one.state_dict()), set(two.state_dict()))
        self.assertTrue(all(torch.equal(value, two.state_dict()[key])
                            for key, value in one.state_dict().items()))
        self.assertEqual(parameter_count(one), parameter_count(two))
        self.assertEqual(one.capacity, two.capacity)
        self.assertEqual(parameter_count(one), one.capacity['actual_parameters'])
        self.assertEqual(one.capacity['match_fraction'], 0.)

    def test_stage_zero_exact_future_persistence_and_history_previous_value(self):
        for arm in ARMS:
            with self.subTest(arm=arm):
                model = self.model(arm)
                output = model(self.history, self.event)
                reference = (100+20*self.history[:, -1:, :, :1]).expand(2, 12, 6, 1)
                expected_history = 100+20*self.history[:, :-1, :, :1]
                self.assertTrue(torch.equal(output['prediction'], reference))
                self.assertTrue(torch.equal(output['reference'], reference))
                self.assertTrue(torch.equal(output['history_prediction'], expected_history))
                self.assertTrue(torch.equal(output['history_target'], 100+20*self.history[:, 1:, :, :1]))
                self.assertEqual(output['history_prediction'].shape, (2, 11, 6, 1))
                self.assertFalse(output['history_target'].requires_grad)
                self.assertIsNone(model.core.coarse_readout.bias)
                self.assertIsNone(model.core.local_readout.bias)

    def test_one_core_used_for_history_and_future_without_lead_embedding(self):
        model = self.model(nonzero_readout=True)
        calls = []
        handle = model.core.register_forward_hook(lambda module, args, output: calls.append(args[3].detach().clone()))
        try:
            output = model(self.history, self.event)
        finally:
            handle.remove()
        self.assertEqual(len(calls), 11+12)
        self.assertTrue(all(torch.count_nonzero(source).item() == 0 for source in calls[:11]))
        self.assertTrue(any(torch.count_nonzero(source).item() > 0 for source in calls[11:]))
        self.assertFalse(any('lead' in name for name, _ in model.named_parameters()))
        self.assertEqual(output['state_rms'].shape, (12, 2))
        self.assertEqual(output['history_state_rms'].shape, (11, 2))

    def test_nonuniform_local_states_zero_mean_and_regional_readout_consistent(self):
        model = self.model(nonzero_readout=True)
        local_states = []
        handles = [module.register_forward_hook(lambda module, args, output:
                    local_states.append(output[1].detach())) for module in (model.core, model.correction)]
        try:
            output = model(self.history, self.event)
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(len(local_states), 34)
        self.assertTrue(any(local.abs().max().item() > 1e-3 for local in local_states))
        for local in local_states:
            torch.testing.assert_close(model.geometry.pool(local),
                torch.zeros_like(model.geometry.pool(local)), atol=3e-6, rtol=0)
        torch.testing.assert_close(model.geometry.pool(output['beta']),
            torch.zeros_like(output['alpha']), atol=1e-6, rtol=0)
        torch.testing.assert_close(model.geometry.pool(output['prediction']), output['regional'], atol=3e-5, rtol=1e-6)
        torch.testing.assert_close(output['prediction'], output['reference']+model.std*(
            model.geometry.lift(output['alpha'])+output['beta']), atol=0, rtol=0)
        for value in output.values():
            self.assertTrue(torch.isfinite(value).all())

    def test_current_and_later_flow_cannot_change_preupdate_prediction(self):
        for arm in ARMS:
            with self.subTest(arm=arm):
                model = self.model(arm, nonzero_readout=True)
                first = model(self.history, self.event)['history_prediction']
                changed = self.history.clone()
                changed[:, 5:, :, 0] += torch.arange(7).view(1, 7, 1)*2+7
                second = model(changed, self.event)['history_prediction']
                # Entries 0..4 predict observations 1..5 before observation 5 is read.
                self.assertTrue(torch.equal(first[:, :5], second[:, :5]))
                self.assertGreater((first[:, 5:]-second[:, 5:]).abs().max().item(), 1e-4)

    def test_history_predictions_and_updates_do_not_read_incident(self):
        model = self.model(nonzero_readout=True)
        changed = dict(self.event, relation=self.event['relation']*-8,
                       type=self.event['type'].flip(0), description=self.event['description'].flip(0))
        one, two = model(self.history, self.event), model(self.history, changed)
        for key in ('history_prediction', 'history_target', 'history_valid', 'history_state_rms', 'innovation_rms'):
            self.assertTrue(torch.equal(one[key], two[key]), key)
        self.assertGreater((one['prediction']-two['prediction']).abs().max().item(), 1e-5)

    def test_control_changes_only_the_observation_residual_rule(self):
        torch.manual_seed(71)
        control = self.model('OBSERVATION_CONTROL', nonzero_readout=True)
        candidate = self.model('INNOVATION_STATE')
        candidate.load_state_dict(control.state_dict(), strict=True)
        control_inputs, candidate_inputs = [], []
        first = control.correction.register_forward_pre_hook(lambda module, args: control_inputs.append(args[2].detach()))
        second = candidate.correction.register_forward_pre_hook(lambda module, args: candidate_inputs.append(args[2].detach()))
        try:
            out_control = control(self.history, self.event)
            out_candidate = candidate(self.history, self.event)
        finally:
            first.remove()
            second.remove()
        expected = self.history[:, 1, :, :1]-self.history[:, 0, :, :1]
        self.assertTrue(torch.equal(control_inputs[0], expected))
        self.assertGreater((control_inputs[0]-candidate_inputs[0]).abs().max().item(), 1e-7)
        self.assertTrue(torch.equal(out_control['history_prediction'][:, :1], out_candidate['history_prediction'][:, :1]))
        self.assertGreater((out_control['prediction']-out_candidate['prediction']).abs().max().item(), 1e-7)

    def test_readouts_learn_first_then_all_state_branches_receive_gradients(self):
        model = self.model()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
        target = 110+torch.arange(6).view(1, 1, 6, 1)*3+torch.arange(12).view(1, 12, 1, 1)*.4
        output = model(self.history, self.event)
        loss = (output['prediction']-target).square().mean()+.1*(
            output['history_prediction']-output['history_target']).square().mean()
        loss.backward()
        for module in (model.core.coarse_readout, model.core.local_readout):
            self.assertGreater(module.weight.grad.abs().sum().item(), 0.)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        output = model(self.history, self.event)
        loss = (output['prediction']-target).square().mean()+.1*(
            output['history_prediction']-output['history_target']).square().mean()
        loss.backward()
        for module in (model.encoder, model.position, model.incident, model.core.exchange,
                       model.core.coarse_update, model.core.local_update, model.correction.gate,
                       model.correction.coarse_map, model.correction.local_map):
            self.assertTrue(any(parameter.grad is not None and parameter.grad.abs().sum().item() > 0
                                for parameter in module.parameters()), type(module).__name__)

    def test_known_clock_advances_across_midnight_and_week_boundary(self):
        tod = torch.tensor([[287, 286], [0, 280]])
        dow = torch.tensor([[6, 2], [0, 5]])
        actual_tod, actual_dow = advance_calendar(tod, dow, 3)
        self.assertTrue(torch.equal(actual_tod, torch.tensor([[2, 1], [3, 283]])))
        self.assertTrue(torch.equal(actual_dow, torch.tensor([[0, 3], [0, 5]])))
        values = torch.zeros(1, 2, 3)
        values[..., 1] = torch.tensor([287, 0])/288
        values[..., 2] = torch.tensor([6, 0])/7
        decoded_tod, decoded_dow = calendar_indices(values)
        self.assertTrue(torch.equal(decoded_tod, torch.tensor([[287, 0]])))
        self.assertTrue(torch.equal(decoded_dow, torch.tensor([[6, 0]])))

    def test_rollout_condition_clock_matches_known_future_calendar(self):
        model = self.model()
        history = self.history.clone()
        history[:, -1, :, 1], history[:, -1, :, 2] = 287/288, 6/7
        clocks = []
        handle = model.encoder.tod.register_forward_pre_hook(lambda module, args: clocks.append(args[0].detach().clone()))
        try:
            model(history, self.event)
        finally:
            handle.remove()
        # One initial state, eleven pre-observation steps, then twelve future steps.
        self.assertEqual(len(clocks), 24)
        for h, clock in enumerate(clocks[-12:]):
            self.assertTrue(torch.equal(clock, torch.full((2, 6), h, dtype=torch.long)))

    def test_absent_report_clears_all_future_event_fields(self):
        model = self.model(nonzero_readout=True)
        absent = dict(self.event, report_present=torch.zeros(2, dtype=torch.bool))
        changed = dict(absent, relation=self.event['relation']*99,
            type=self.event['type'].flip(0), description=self.event['description'].flip(0))
        one, two = model(self.history, absent), model(self.history, changed)
        self.assertTrue(torch.equal(one['prediction'], two['prediction']))
        self.assertFalse(one['support'].any())
        self.assertFalse(one['available'].any())

    def test_invalid_observation_pair_masks_correction_in_both_arms(self):
        history = self.history.clone()
        history[:, 4, 0, 0] = -5.  # raw zero; affects pairs entering and leaving slot 4.
        history[:, 7, 3, 0] = -6.  # raw negative; distinct region and time.
        raw = 100+20*history[..., :1]
        expected_valid = (torch.isfinite(raw[:, 1:]) & torch.isfinite(raw[:, :-1])
                          & (raw[:, 1:] > .001) & (raw[:, :-1] > .001))
        for arm in ARMS:
            with self.subTest(arm=arm):
                model = self.model(arm, nonzero_readout=True)
                residuals = []
                handle = model.correction.register_forward_pre_hook(
                    lambda module, args: residuals.append(args[2].detach().clone()))
                try:
                    output = model(history, self.event)
                finally:
                    handle.remove()
                self.assertEqual(output['history_valid'].dtype, torch.bool)
                self.assertTrue(torch.equal(output['history_valid'], expected_valid))
                stacked = torch.stack(residuals, 1)
                self.assertTrue(torch.equal(stacked[~expected_valid], torch.zeros_like(stacked[~expected_valid])))
                self.assertGreater(stacked[expected_valid].abs().max().item(), 0.)

    def test_zero_residual_is_exactly_no_observation_update_apart_from_projection(self):
        model = self.model(nonzero_readout=True)
        position = model.position()
        static = model.encoder.static(position)
        tod, dow = calendar_indices(self.history[:, 3])
        condition = model.encoder.condition(static, tod, dow)
        coarse = torch.randn(2, 2, 32)
        local = model.geometry.center(torch.randn(2, 6, 32))
        actual_c, actual_l = model.correction(coarse, local, torch.zeros(2, 6, 1),
                                             condition, position, model.geometry)
        self.assertTrue(torch.equal(actual_c, coarse))
        torch.testing.assert_close(actual_l, local, atol=2e-7, rtol=0)

    def test_history_threshold_matches_float64_metric_comparison(self):
        self.data.mean, self.data.std = 0., 1.
        self.config['target_minimum'] = .001
        history = self.history.clone()
        edge = torch.tensor(.001, dtype=torch.float32)
        below = torch.nextafter(edge, torch.tensor(0., dtype=torch.float32))
        history[..., 0] = edge
        history[:, 4, 0, 0] = below
        output = self.model()(history, self.event)
        raw = history[..., :1]
        expected = (torch.isfinite(raw[:, 1:]) & torch.isfinite(raw[:, :-1])
                    & (raw[:, 1:].double() > .001) & (raw[:, :-1].double() > .001))
        self.assertTrue(torch.equal(output['history_valid'], expected))
        self.assertTrue(output['history_valid'][:, :, 1:].all())
        self.assertFalse(output['history_valid'][:, 3:5, 0].any())

    def test_no_report_history_and_future_condition_are_exactly_equal(self):
        model = self.model()
        static = model.encoder.static(model.position())
        tod, dow = calendar_indices(self.history[:, 6])
        historical = model.encoder.condition(static, tod, dow)
        future_no_report = model.encoder.condition(static, tod, dow,
            torch.zeros(2, self.config['hidden']), torch.zeros(2, 6, self.config['hidden']))
        self.assertTrue(torch.equal(historical, future_no_report))

    def test_stages_shapes_and_arms_fail_closed(self):
        model = self.model()
        model.set_stage('joint')
        with self.assertRaises(ValueError):
            model.set_stage('warmup')
        with self.assertRaises(ValueError):
            self.model('UNKNOWN')
        with self.assertRaises(ValueError):
            model(self.history[:, :-1], self.event)
        self.config['observation_step_initial'] = 0.
        with self.assertRaises(ValueError):
            self.model()


if __name__ == '__main__':
    unittest.main()
