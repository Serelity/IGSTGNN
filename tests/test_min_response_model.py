"""Persistent response identities, condition routing and complete-branch gradients."""
import copy
import json
import unittest
from unittest.mock import patch

import torch

from src.models.incident_spatial import exchange_step
from src.models.incident_spatial_residual import tensor_state_hash
from src.models.incident_spatial_response import build_frozen_response, response_common_hash
from tests.test_min_event_budget_model import event_inputs, minimal


def active_response(model):
    """Controlled capacity-limited example, without altering its frozen MIN."""
    with torch.no_grad():
        model.response.capacity_prior[-1].weight.fill_(.02)
        model.response.capacity_prior[-1].bias.fill_(-3.)
        model.response.proposal[-1].weight.fill_(.02)
        model.response.proposal[-1].bias.fill_(2.)
        model.response.condition_output.weight.fill_(.08)
        model.response.condition_output.bias.copy_(torch.linspace(-.2, .2, model.response.latent))
        model.response.readout.weight.copy_(torch.linspace(-.2, .3, 3*model.response.latent)[None])


class PersistentResponseModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_zero_start_one_MIN_call_exact_on_off_and_frozen_source(self):
        base, (x, report) = minimal(), event_inputs()
        before, rng = tensor_state_hash(base.state_dict()), torch.get_rng_state().clone()
        frozen = copy.deepcopy(base).requires_grad_(False).eval()
        for mode in ('location', 'state'):
            model, cert = build_frozen_response(base, mode)
            with torch.inference_mode(), patch.object(model.backbone, 'forward', wraps=model.backbone.forward) as calls:
                expected = frozen(x)
                on = model(x, incident_data=report, return_details=True)
                self.assertEqual(calls.call_count, 1)
                off = model(x, incident_data=report, report_enabled=False)
            torch.testing.assert_close(on['prediction'], expected, rtol=0, atol=0)
            torch.testing.assert_close(on['baseline_prediction'], expected, rtol=0, atol=0)
            torch.testing.assert_close(off, expected, rtol=0, atol=0)
            self.assertFalse(on['baseline_prediction'].requires_grad)
            self.assertEqual(on['response']['state_delta_abs_mean'], 0.)
            self.assertEqual(len(on['response']['by_slot']), 12)
            json.dumps(on['response'])
            self.assertEqual(cert['frozen_state_sha256'], before)
            self.assertTrue(model.assert_backbone_frozen())
            self.assertEqual(len(model.backbone.out_fc_1._forward_pre_hooks), 0)
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        self.assertEqual(before, tensor_state_hash(base.state_dict()))

    def test_common_initialization_exact_and_relation_parameters_active(self):
        base = minimal()
        location, left = build_frozen_response(base, 'location')
        state, right = build_frozen_response(base, 'state')
        self.assertEqual(left['common_initial_sha256'], right['common_initial_sha256'])
        self.assertEqual(response_common_hash(location.response), response_common_hash(state.response))
        self.assertEqual(left['trainable_parameters']-right['trainable_parameters'], 6*8)
        self.assertTrue(all(name.startswith('response.') for name in left['trainable_names']))
        self.assertNotEqual(left['response_initial_sha256'], right['response_initial_sha256'])
        self.assertFalse(hasattr(state.response, 'relation_projection'))

    def test_first_condition_gradient_then_complete_branch_gradients(self):
        x, report = event_inputs()
        model, _ = build_frozen_response(minimal(((0, 1), (1, 2), (2, 0))))
        # A task with active service constraints gives the zero-start output a
        # gradient; upstream shared state modules open after that output moves.
        with torch.no_grad():
            model.response.capacity_prior[-1].weight.fill_(.02)
            model.response.capacity_prior[-1].bias.fill_(-3.)
            model.response.proposal[-1].bias.fill_(2.)
        result = model(x, incident_data=report, return_details=True)
        result['prediction'].sum().backward()
        self.assertGreater(float(model.response.condition_output.weight.grad.abs().sum()), 0.)
        model.zero_grad(set_to_none=True)
        active_response(model)
        with torch.no_grad():
            # Some reference proposals remain below C0, while their event
            # budgets are lower still. A fully capacity-limited example would
            # correctly give every proposal parameter zero gradient.
            model.response.proposal[-1].bias.copy_(torch.tensor([-3.2, 2., -3.2, 2.]))
            model.response.condition_output.bias.copy_(torch.tensor([-.8, .1, -.8, .2]))
        result = model(x, incident_data=report, return_details=True)
        weights = torch.linspace(.3, 2., 12).reshape(1, 12, 1, 1)
        (result['prediction']*weights).sum().backward()
        for name in ('origin_encoder', 'feature_projection', 'capacity_prior', 'proposal',
                     'condition_state', 'condition_output', 'relation_projection', 'readout'):
            module = getattr(model.response, name)
            total = sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None)
            self.assertGreater(total, 0., name)
        self.assertTrue(model.assert_backbone_frozen())

    def test_unmodified_default_initialization_has_task_gradient_at_condition_output(self):
        x, report = event_inputs()
        model, _ = build_frozen_response(minimal(((0, 1), (1, 2), (2, 0))))
        model(x, incident_data=report).square().mean().backward()
        gradient = model.response.condition_output.weight.grad
        self.assertTrue(bool(torch.isfinite(gradient).all()))
        self.assertGreater(float(gradient.abs().sum()), 0.)
        self.assertTrue(model.assert_backbone_frozen())

    def test_joint_phase_retains_feature_gradients_and_uses_same_checkpoint_OFF(self):
        x, report = event_inputs()
        model, _ = build_frozen_response(minimal(((0, 1), (1, 2), (2, 0))))
        source_digest = tensor_state_hash(model.backbone.state_dict())
        model.set_joint_training(True).train()
        self.assertTrue(all(p.requires_grad for p in model.backbone.parameters()))
        self.assertTrue(model.backbone.training)
        with self.assertRaisesRegex(ValueError, 'joint'):
            model.assert_backbone_frozen()
        model.eval()
        # At zero condition the joint checkpoint ON and OFF agree in eval.
        on = model(x, incident_data=report, return_details=True)
        off = model(x, incident_data=report, report_enabled=False)
        torch.testing.assert_close(on['prediction'], off, rtol=0, atol=0)
        self.assertFalse(on['baseline_prediction'].requires_grad)
        self.assertTrue(on['prediction'].requires_grad)
        self.assertEqual(on['response']['off_reference'], 'same_joint_checkpoint_MIN')
        active_response(model)
        features = []
        def capture(module, args):
            args[0].retain_grad()
            features.append(args[0])
        handle = model.response.feature_projection.register_forward_pre_hook(capture)
        try:
            result = model(x, incident_data=report, return_details=True)
            result['prediction'].square().mean().backward()
        finally:
            handle.remove()
        self.assertEqual(len(features), 12)
        self.assertTrue(all(value.requires_grad for value in features))
        self.assertGreater(sum(float(value.grad.abs().sum()) for value in features if value.grad is not None), 0.)
        self.assertGreater(float(model.backbone.embedding.weight.grad.abs().sum()), 0.)
        with torch.no_grad():
            model.backbone.out_fc_2.bias.add_(.01)
        self.assertNotEqual(tensor_state_hash(model.backbone.state_dict()), source_digest)
        model.set_joint_training(False)
        with self.assertRaisesRegex(ValueError, 'changed'):
            model.assert_backbone_frozen()

    def test_state_control_learns_nonzero_without_numeric_D(self):
        x, report = event_inputs()
        model, _ = build_frozen_response(minimal(), 'state')
        active_response(model)
        alternate = dict(report, relation=report['relation']*7.)
        with torch.no_grad():
            first = model(x, incident_data=report, return_details=True)
            second = model(x, incident_data=alternate)
        self.assertGreater(first['response']['delta_prediction_abs_mean'], 0.)
        torch.testing.assert_close(first['prediction'], second, rtol=0, atol=0)

    def test_location_uses_relation_detail_with_support_fixed(self):
        x, report = event_inputs()
        model, _ = build_frozen_response(minimal())
        active_response(model)
        fixed_support = report['relation'].abs().sum(-1, keepdim=True)>0
        with torch.no_grad():
            model.response.relation_projection.weight.fill_(.5)
            actual = model(x, incident_data=report, support_override=fixed_support)
            changed = model(x, incident_data=dict(report, relation=report['relation']*0.), support_override=fixed_support)
        self.assertGreater(float((actual-changed).abs().max()), 0.)

    def test_state_is_carried_across_slots_and_origin_runs_once(self):
        x, report = event_inputs()
        model, _ = build_frozen_response(minimal(((0, 1), (1, 2))))
        active_response(model)
        calls = []
        def record(state, proposal, capacity, graph, **kwargs):
            output = exchange_step(state, proposal, capacity, graph, **kwargs)
            calls.append((state.detach().clone(), output['state'].detach().clone()))
            return output
        with patch('src.models.incident_spatial_response.exchange_step', side_effect=record), \
             patch.object(model.response, 'origin', wraps=model.response.origin) as origin:
            model(x, incident_data=report)
        self.assertEqual(origin.call_count, 1)
        self.assertEqual(len(calls), 12*2*2)
        # The last substep of slot 1 is followed by slot 2 without reset.
        torch.testing.assert_close(calls[4][0], calls[2][1], atol=0, rtol=0)
        torch.testing.assert_close(calls[5][0], calls[3][1], atol=0, rtol=0)
        self.assertGreater(float((calls[4][0]-calls[0][0]).abs().max()), 0.)

    def test_missing_report_or_relation_and_outside_operator_fall_back_exactly(self):
        x, report = event_inputs()
        base = minimal()
        base.internal_spatial_graph.operator_mask[3] = False
        model, _ = build_frozen_response(base)
        active_response(model)
        with torch.no_grad():
            reference = model(x, report_enabled=False)
            for incident in (None, dict(report, report_present=torch.zeros(2, dtype=torch.bool)),
                             dict(report, relation_available=torch.zeros(2, dtype=torch.bool)),
                             dict(report, relation=torch.zeros_like(report['relation']))):
                torch.testing.assert_close(model(x, incident_data=incident), reference, rtol=0, atol=0)
            actual = model(x, incident_data=report)
            torch.testing.assert_close(actual[:, :, 3], reference[:, :, 3], rtol=0, atol=0)
        empty, _ = build_frozen_response(minimal(()))
        active_response(empty)
        with torch.no_grad():
            torch.testing.assert_close(empty(x, incident_data=report), empty(x, report_enabled=False), rtol=0, atol=0)

    def test_internal_exchange_balance_bounds_zero_boundary_and_plain_diagnostics(self):
        x, report = event_inputs()
        model, _ = build_frozen_response(minimal(((-1, 0), (0, 1), (1, 2), (2, -1))))
        active_response(model)
        seen = []
        def record(state, proposal, capacity, graph, **kwargs):
            result = exchange_step(state, proposal, capacity, graph, **kwargs)
            seen.append((proposal.detach(), capacity.detach(), result))
            return result
        with torch.no_grad(), patch('src.models.incident_spatial_response.exchange_step', side_effect=record):
            result = model(x, incident_data=report, return_details=True)
        for proposal, capacity, record in seen:
            self.assertEqual(float(proposal[:, [0, 3]].abs().max()), 0.)
            self.assertEqual(float(capacity[:, [0, 3]].abs().max()), 0.)
            self.assertTrue(bool(((record['state'] >= -1e-5) & (record['state'] <= 1+1e-5)).all()))
            self.assertLess(float(record['balance'].abs().max()), 1e-5)
        for name in ('balance_abs_max', 'capacity_violation_max', 'sending_violation_max', 'receiving_violation_max'):
            self.assertLess(result['response'][name], 1e-5)
        diagnostic = result['response']
        self.assertNotIn('capacity_limited_fraction', diagnostic)
        self.assertGreater(diagnostic['eligible_edge_positions'], 0)
        for name in ('internal_proposal_exceeds_budget_fraction_all_edges',
                     'eligible_proposal_exceeds_budget_fraction', 'eligible_final_flux_at_budget_fraction'):
            self.assertGreaterEqual(diagnostic[name], 0.)
            self.assertLessEqual(diagnostic[name], 1.)
        self.assertLessEqual(diagnostic['eligible_final_flux_at_budget_fraction'],
                             diagnostic['eligible_proposal_exceeds_budget_fraction'])
        json.dumps(result['response'])

    def test_proposal_exceedance_denominators_and_zero_eligible_are_explicit(self):
        x, report = event_inputs()
        model, _ = build_frozen_response(minimal(((-1, 0), (0, 1), (1, 2), (2, -1))))
        active_response(model)
        with torch.no_grad():
            result = model(x, incident_data=report, return_details=True)
            off = model(x, report_enabled=False, return_details=True)
        details = result['response']
        self.assertEqual(details['edge_positions'], 2*4*4)
        self.assertEqual(off['response']['eligible_edge_positions'], 0)
        self.assertEqual(off['response']['eligible_proposal_exceeds_budget_fraction'], 0.)
        self.assertEqual(off['response']['eligible_final_flux_at_budget_fraction'], 0.)
        # Controlled high proposals exceed every internal budget: denominator
        # includes two boundary edges only in the all-edge statistic.
        self.assertEqual(details['internal_proposal_exceeds_budget_fraction_all_edges'], .5)
        self.assertEqual(details['eligible_proposal_exceeds_budget_fraction'], 1.)

    def test_exception_removes_hook_and_train_keeps_all_backbone_modules_eval(self):
        x, report = event_inputs()
        model, _ = build_frozen_response(minimal())
        model.eval().train()
        self.assertTrue(model.response.training)
        self.assertTrue(all(not module.training for module in model.backbone.modules()))
        with patch.object(model.backbone, '_decode_forecast', side_effect=RuntimeError('injected')):
            with self.assertRaisesRegex(RuntimeError, 'injected'):
                model(x, incident_data=report)
        self.assertEqual(len(model.backbone.out_fc_1._forward_pre_hooks), 0)
        self.assertTrue(model.assert_backbone_frozen())
        parameter = next(model.backbone.parameters())
        parameter.requires_grad_(True)
        with self.assertRaisesRegex(ValueError, 'flag'):
            model.assert_backbone_frozen()

    def test_invalid_dimensions_flags_support_and_inputs_rejected(self):
        base, (x, report) = minimal(), event_inputs()
        for kwargs in ({'mode': 'semantic'}, {'latent': True}, {'steps': 0}, {'eta': 0.}):
            with self.assertRaises(ValueError):
                build_frozen_response(base, **kwargs)
        model, _ = build_frozen_response(base)
        for kwargs in ({'report_enabled': 1}, {'support_override': torch.zeros(2, 4, 1)},
                       {'incident_data': dict(report, relation=report['relation'].double())},
                       {'incident_data': dict(report, report_present=torch.ones(2))}):
            with self.assertRaises(ValueError):
                model(x, **kwargs)


if __name__ == '__main__':
    unittest.main()
