import copy
import unittest

import torch

from src.models.incident_capacity_exchange import ExchangeGraph
from src.models.incident_capacity_fusion import (
    CapacityAugmentedIGSTGNN, IncidentCapacityBranch, report_summary,
)
from test_acdg import make_model, incident_batch


torch.set_num_threads(3)


def fixture(mode='capacity', device='cpu'):
    graph = ExchangeGraph(3, torch.tensor([[-1, 0, 1, 2], [0, 1, 2, -1]]),
                          torch.ones(3, dtype=torch.bool), torch.tensor([True, False, True]),
                          evidence_scope='synthetic_only')
    branch = IncidentCapacityBranch(graph, torch.tensor([0., 1., 1., 1.]), mode=mode).to(device)
    inputs = dict(history=torch.rand(2, 12, 3, 3, device=device),
                  valid=torch.ones(2, 12, 3, 3, dtype=torch.bool, device=device),
                  references=torch.ones(3, 3, device=device), labels=torch.arange(-65., -5., 5., device=device))
    inputs['reports'] = dict(weights=torch.tensor([[[0., 1., 0., 0.]], [[0., 0., .8, 0.]]], device=device),
                            ages=torch.tensor([[3.], [4.]], device=device),
                            present=torch.ones(2, 1, dtype=torch.bool, device=device),
                            distance=torch.zeros(2, 1, 4, device=device), confidence=torch.ones(2, 1, device=device))
    return branch, inputs


class CapacityFusionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4010)

    def test_report_summary_independent_numbers_empty_set_and_permutation(self):
        weights = torch.tensor([[[.5, 0.], [.2, .8]]])
        ages = torch.tensor([[2., 8.]])
        present = torch.ones(1, 2, dtype=torch.bool)
        distance = torch.tensor([[[1., 0.], [3., 2.]]])
        confidence = torch.tensor([[.8, .5]])
        summary, association = report_summary(weights, ages, present, distance, confidence)
        torch.testing.assert_close(association, torch.tensor([[.6, .8]]))
        self.assertAlmostEqual(float(summary[0, 0, 3]), float((.5*torch.log1p(ages[0, 0])+.2*torch.log1p(ages[0, 1]))/.7/5))
        permuted = report_summary(weights.flip(1), ages.flip(1), present.flip(1), distance.flip(1), confidence.flip(1))
        for actual, expected in zip(permuted, (summary, association)):
            torch.testing.assert_close(actual, expected)
        empty, support = report_summary(weights[:, :0], ages[:, :0], present[:, :0], distance[:, :0], confidence[:, :0])
        self.assertEqual(empty.shape, (1, 2, 8))
        self.assertEqual(float(empty.abs().sum()+support.sum()), 0)

    def test_report_errors_reject_future_unknown_fields_inlet_and_nonlocal_weights(self):
        branch, inputs = fixture()
        for mutation in ('future', 'nonlocal', 'inlet', 'absent', 'unknown'):
            bad = copy.deepcopy(inputs)
            if mutation == 'future':
                bad['reports']['ages'][0, 0] = -1
            elif mutation == 'nonlocal':
                bad['reports']['weights'][0, 0, 2] = 1
            elif mutation == 'inlet':
                bad['reports']['weights'][0, 0] = torch.tensor([1., 0., 0., 0.])
            elif mutation == 'absent':
                bad['reports']['present'][0, 0] = False
            else:
                bad['reports']['duration'] = torch.ones(2)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                branch(**bad)

    def test_history_clock_masking_blind_lag_and_all_missing_prior(self):
        branch, inputs = fixture()
        inputs['valid'][0, :, 1] = False
        inputs['history'][0, :, 1] = torch.nan
        result = branch(**inputs)
        self.assertTrue(torch.equal(result['initial'][0, 1], torch.full((4,), .5)))
        self.assertFalse(bool(result['available'][0, 1]))
        self.assertEqual(float(result['lag_intervals'][1, 0]), 1)
        self.assertTrue(torch.isfinite(result['features']).all())
        changed = copy.deepcopy(inputs)
        changed['history'][~changed['valid']] = 1e20
        torch.testing.assert_close(branch(**changed)['features'], result['features'], atol=0, rtol=0)
        for label in (inputs['labels']+5, inputs['labels'].flip(0)):
            with self.assertRaises(ValueError):
                branch(**dict(inputs, labels=label))

    def test_condition_off_preserves_initial_boundary_and_history_and_restores_on(self):
        branch, inputs = fixture()
        on = branch(**inputs)
        off = branch(**inputs, incident_enabled=False)
        for key in ('initial', 'hidden', 'summary', 'boundary_demand', 'boundary_coefficients', 'history_coefficients'):
            self.assertTrue(torch.equal(on[key], off[key]), key)
        self.assertTrue(torch.equal(off['coefficients'], off['history_coefficients']))
        self.assertGreater(float((on['coefficients']-off['coefficients']).detach().abs().max()), 0)
        association = on['report_association'] == 0
        self.assertTrue(torch.equal(on['coefficients'][association], off['coefficients'][association]))
        self.assertTrue(torch.equal(branch(**inputs)['features'], on['features']))

    def test_empty_reports_exactly_match_disabled_condition(self):
        branch, inputs = fixture()
        empty = {k: v[:, :0] for k, v in inputs['reports'].items()}
        a = branch(**dict(inputs, reports=empty))
        b = branch(**inputs, incident_enabled=False)
        self.assertTrue(torch.equal(a['features'], b['features']))

    def test_initialization_matches_full_native_and_preserves_icsf_tiid(self):
        native = make_model().eval()
        branch, inputs = fixture()
        model = CapacityAugmentedIGSTGNN(copy.deepcopy(native), branch).eval()
        x = torch.rand(2, 12, 3, 3)
        incident = incident_batch()
        with torch.no_grad():
            original = native(x, incident_data=incident)
            details = model(x, incident_data=incident, capacity_inputs=inputs, return_details=True)
            self.assertTrue(torch.equal(original, details['prediction']))
            self.assertEqual(float(details['forecast_delta'].abs().max()), 0)
            without = native(x, incident_data=None)
            self.assertGreater(float((without-original).abs().max()), 0)
            disabled = model(x, incident_data=incident, capacity_inputs=inputs, incident_enabled=False)
            self.assertTrue(torch.equal(original, disabled))
        with self.assertRaises(ValueError):
            model(x, capacity_inputs=dict(inputs, target=torch.zeros(1)))
        with self.assertRaises(ValueError):
            native(x, incident_data=incident, forecast_delta=torch.zeros(1))

    def test_auxiliary_head_and_projection_train_then_capacity_reaches_main_forecast(self):
        branch, inputs = fixture()
        # Engineer a binding capacity. Low-demand zero gradients are valid;
        # connectivity must be checked where the minimum actually selects C.
        with torch.no_grad():
            branch.coefficients[-1].bias.fill_(.8)
        model = CapacityAugmentedIGSTGNN(make_model().eval(), branch)
        x, incident = torch.rand(2, 12, 3, 3), incident_batch()
        optimizer = torch.optim.Adam(branch.parameters(), lr=.01)
        # Synthetic response target only; not a real-data performance claim.
        first = model(x, incident_data=incident, capacity_inputs=inputs, return_details=True)
        loss = (first['prediction']-2).square().mean() + .1*(first['auxiliary_prediction']-1).square().mean()
        loss.backward()
        self.assertGreater(float(branch.projection.weight.grad.abs().sum()), 0)
        self.assertGreater(float(branch.coefficients[-1].weight.grad.abs().sum()), 0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        second = model(x, incident_data=incident, capacity_inputs=inputs, return_details=True)
        (second['prediction']-2).square().mean().backward()
        for name in ('history.gru.weight_ih_l0', 'coefficients.0.weight', 'coefficients.2.weight', 'boundary.2.weight'):
            grad = dict(branch.named_parameters())[name].grad
            self.assertIsNotNone(grad, name)
            self.assertGreater(float(grad.abs().sum()), 0, name)
        self.assertGreater(float(second['forecast_delta'].detach().abs().sum()), 0)

    def test_auxiliary_and_fusion_consume_only_rollout_features(self):
        branch, inputs = fixture()
        with torch.no_grad():
            branch.projection.weight.normal_(0, .1)
        result = branch(**inputs)
        torch.testing.assert_close(result['forecast_delta'], branch.projection(result['features']))
        torch.testing.assert_close(result['auxiliary_prediction'], branch.observation(result['features']))
        self.assertEqual(result['features'].shape, (2, 12, 3, 16))
        self.assertEqual(result['rollout']['times'][-1], 65)

    def test_budget_common_weights_and_effective_ordinary_gradients(self):
        capacity, inputs = fixture()
        ordinary, _ = fixture('ordinary')
        common = capacity.state_dict()
        missing, unexpected = ordinary.load_state_dict(common, strict=False)
        self.assertFalse(unexpected)
        self.assertTrue(all(k.startswith(('operator.message.', 'operator.transition.', 'operator.gate.')) for k in missing))
        for k, value in common.items():
            self.assertTrue(torch.equal(value, ordinary.state_dict()[k]), k)
        counts = [sum(p.numel() for p in model.parameters()) for model in (capacity, ordinary)]
        self.assertEqual(counts[1]-counts[0], 244)
        result = ordinary(**inputs)
        (result['auxiliary_prediction']-1).square().mean().backward()
        for name, param in ordinary.operator.named_parameters():
            self.assertIsNotNone(param.grad, name)
            self.assertGreater(float(param.grad.abs().sum()), 0, name)

    def test_ineligible_nodes_have_zero_increment_even_after_training(self):
        graph = ExchangeGraph(3, torch.empty(2, 0, dtype=torch.long), torch.zeros(3, dtype=torch.bool),
                              torch.zeros(3, dtype=torch.bool), evidence_scope='candidate_unverified')
        branch = IncidentCapacityBranch(graph, torch.empty(0))
        _, inputs = fixture()
        inputs.pop('reports')
        with torch.no_grad():
            branch.projection.weight.fill_(.3)
            branch.observation.bias.fill_(.7)
        result = branch(**inputs)
        for key in ('features', 'forecast_delta', 'auxiliary_prediction'):
            self.assertEqual(float(result[key].abs().sum()), 0)

    def test_checkpoint_and_optimizer_next_update_match(self):
        model, inputs = fixture()
        optimizer = torch.optim.Adam(model.parameters(), lr=.002)
        def step(m, opt):
            opt.zero_grad(set_to_none=True)
            r = m(**inputs)
            ((r['auxiliary_prediction']-1).square().mean()+r['forecast_delta'].square().mean()).backward()
            opt.step()
        step(model, optimizer)
        restored, _ = fixture()
        restored.load_state_dict(copy.deepcopy(model.state_dict()))
        other = torch.optim.Adam(restored.parameters(), lr=.002)
        other.load_state_dict(copy.deepcopy(optimizer.state_dict()))
        self.assertTrue(torch.equal(model(**inputs)['features'], restored(**inputs)['features']))
        step(model, optimizer)
        step(restored, other)
        for k, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, restored.state_dict()[k]), k)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_matches_cpu_and_has_shared_encoder_gradients(self):
        cpu, inputs = fixture()
        with torch.no_grad():
            cpu.coefficients[-1].bias.fill_(.8)
        gpu = copy.deepcopy(cpu).cuda()
        cuda_inputs = {k: ({q: v.cuda() for q, v in value.items()} if isinstance(value, dict) else value.cuda())
                       for k, value in inputs.items()}
        a, b = cpu(**inputs), gpu(**cuda_inputs)
        torch.testing.assert_close(a['features'], b['features'].cpu(), atol=1e-5, rtol=1e-5)
        b['auxiliary_prediction'].square().mean().backward()
        self.assertGreater(float(gpu.coefficients[-1].weight.grad.abs().sum()), 0)


if __name__ == '__main__':
    unittest.main()
