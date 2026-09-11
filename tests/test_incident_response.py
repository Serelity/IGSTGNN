import hashlib
import tempfile
import unittest
from pathlib import Path

import torch

from src.models.igstgnn import IGSTGNN, TemporalIncidentImpactDecay
try:
    from src.models.incident_response import IncidentTimeResponse, ReportLocationEncoder
except ModuleNotFoundError:
    IncidentTimeResponse = ReportLocationEncoder = None


def model_args():
    return dict(num_feat=1, num_hidden=8, node_hidden=4, time_emb_dim=4,
                layer=5, k_s=2, k_t=3, tpd=288, dropout=0.0, gap=3,
                adjs=[torch.eye(3)] * 2)


def make_research(mode):
    torch.manual_seed(1234)
    return IGSTGNN(model_args=dict(model_args(), incident_schema='report_location_v1',
                                  time_response=mode, use_sensor_info=False),
                    node_num=3, input_dim=3, output_dim=1, seq_len=12, horizon=12,
                    dataset='unused', data_path='/unused-report-schema-path').eval()


def inputs():
    x = torch.zeros(2, 12, 3, 3)
    x[..., 0] = torch.arange(12).view(1, 12, 1) + torch.arange(3).view(1, 1, 3)
    x[..., 1] = .25
    x[..., 2] = 2 / 7
    incident = {'report_age_minutes': torch.tensor([6., 9.]),
                'forecast_tod': torch.tensor([100, 200]), 'forecast_dow': torch.tensor([0, 6]),
                'distances': torch.tensor([[[0., .8, 1.], [0., .5, 0.], [0., 0., 0.]]]).expand(2, -1, -1)}
    return x, incident


class IncidentResponseTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(3)

    def require_response(self):
        self.assertIsNotNone(IncidentTimeResponse, 'Research time-response implementation is missing')

    def test_default_legacy_numerics_and_checkpoint_keys_remain_unchanged(self):
        torch.manual_seed(1234)
        with tempfile.TemporaryDirectory() as directory:
            for name in ('desc_mapping.json', 'type_mapping.json'):
                (Path(directory) / name).write_text('{"test":0}')
            model = IGSTGNN(model_args=model_args(), node_num=3, input_dim=3, output_dim=1,
                            seq_len=12, horizon=12, dataset='test', data_path=directory,
                            use_sensor_info=False).eval()
        x = torch.arange(216, dtype=torch.float32).reshape(2, 12, 3, 3) / 216
        incident = {'incident': torch.zeros(2, 4), 'position': torch.tensor([11, 11]),
                    'distances': torch.tensor([[[1., 0., 0.], [.5, .2, 1.], [0., 0., 0.]]]).expand(2, -1, -1)}
        with torch.no_grad():
            actual = model(x, incident_data=incident)
        expected = torch.tensor([.1454806924, .2248780727, .2371357083, .1457182616,
                                 .2840448320, .3318867087, .1447663605, .3082248867,
                                 .4047656357, .0929655805, .3526356220, .4430113435])
        torch.testing.assert_close(actual[0, :, 0, 0], expected, atol=1e-6, rtol=1e-6)
        self.assertAlmostEqual(actual.sum().item(), 18.791061401367188, places=5)
        self.assertEqual(hashlib.sha256('\n'.join(model.state_dict()).encode()).hexdigest(),
                         '46241f15766ed4a063be4411796a36fe69c12f8f23cc9920ac5c3f7505fe4ca2')
        model.load_state_dict(model.state_dict(), strict=True)

    def test_abc_share_common_initial_weights_and_predictions(self):
        self.require_response()
        models = [make_research(mode) for mode in ('fixed', 'shared', 'conditioned')]
        common = models[0].state_dict()
        for model in models[1:]:
            for key, value in common.items():
                torch.testing.assert_close(model.state_dict()[key], value, atol=0, rtol=0)
        x, incident = inputs()
        with torch.no_grad():
            predictions = [model(x, incident_data=incident) for model in models]
        for prediction in predictions[1:]:
            torch.testing.assert_close(prediction, predictions[0], atol=0, rtol=0)
        counts = [sum(p.numel() for p in model.parameters()) for model in models]
        self.assertEqual(counts[1] - counts[0], 12)
        self.assertEqual(counts[2] - counts[0], 268)
        self.assertIsNone(models[2].tiid_module.time_response.mlp[-1].bias)

    def test_report_encoder_uses_age_and_shared_forecast_clock_embeddings(self):
        self.require_response()
        model = make_research('fixed')
        x, incident = inputs()
        seen = []
        handle = model.icsf_module.report_encoder.fusion[0].register_forward_pre_hook(
            lambda module, args: seen.append(args[0].detach().clone()))
        model(x, incident_data=incident)
        handle.remove()
        expected = torch.cat([incident['report_age_minutes'].unsqueeze(-1) / 5,
                              model.T_i_D_emb[incident['forecast_tod']],
                              model.D_i_W_emb[incident['forecast_dow']]], dim=-1)
        torch.testing.assert_close(seen[0], expected, atol=0, rtol=0)
        self.assertFalse(any(word in name for name in model.icsf_module.state_dict()
                             for word in ('position_embedding', 'desc_embedding', 'incident_type_embedding', 'holiday_embedding')))

    def test_conditioned_state_is_exact_history_flow_summary_and_ignores_y(self):
        self.require_response()
        model = make_research('conditioned')
        with torch.no_grad():
            model.tiid_module.time_response.mlp[-1].weight.fill_(.01)
        x, incident = inputs()
        seen = []
        handle = model.tiid_module.time_response.register_forward_pre_hook(
            lambda module, args: seen.append(args[1].detach().clone()))
        with torch.no_grad():
            first = model(x, torch.zeros(2, 12, 3, 1), incident_data=incident)
            second = model(x, torch.full((2, 12, 3, 1), 12345.), incident_data=incident)
        handle.remove()
        expected = torch.tensor([[[5.5, 11., 9.], [6.5, 12., 9.], [7.5, 13., 9.]]]).expand(2, -1, -1)
        torch.testing.assert_close(seen[0], expected, atol=0, rtol=0)
        torch.testing.assert_close(seen[1], expected, atol=0, rtol=0)
        torch.testing.assert_close(first, second, atol=0, rtol=0)

    def positive_context(self, mode):
        module = TemporalIncidentImpactDecay(2, 2)
        module.time_response = IncidentTimeResponse(mode, 12)
        with torch.no_grad():
            for parameter in module.context_fusion.parameters():
                parameter.zero_()
            module.context_fusion[-1].bias.fill_(1)
            module.context_projection.weight.copy_(torch.eye(2))
        return module

    def test_disconnected_nodes_have_no_tiid_effect_after_response_changes(self):
        self.require_response()
        for mode in ('shared', 'conditioned'):
            module = self.positive_context(mode)
            with torch.no_grad():
                module.time_response.b.fill_(-.5)
            hidden = torch.ones(1, 12, 2, 2)
            actual = module(hidden, torch.ones(1, 2), torch.empty(1, 2, 0),
                            torch.tensor([[[0., 1., 0.], [0., 0., 0.]]]),
                            history_state=torch.ones(1, 2, 3))
            torch.testing.assert_close(actual[:, :, 1], hidden[:, :, 1], atol=0, rtol=0)
            self.assertLess(actual[0, -1, 0, 0].item(), hidden[0, -1, 0, 0].item())

    def test_horizon_twelve_offset_gradient_does_not_vanish_with_gaussian(self):
        self.require_response()
        for mode in ('shared', 'conditioned'):
            module = self.positive_context(mode)
            actual = module(torch.zeros(1, 12, 1, 2), torch.ones(1, 2), torch.empty(1, 1, 0),
                            torch.tensor([[[0., 1., 0.]]]), history_state=torch.ones(1, 1, 3))
            actual[:, -1].sum().backward()
            self.assertAlmostEqual(module.time_response.b.grad[-1].item(), 2.0)
            if mode == 'conditioned':
                self.assertGreater(module.time_response.mlp[-1].weight.grad[-1].abs().sum().item(), 0.)

    def test_conditioned_early_layer_gradient_opens_after_first_update(self):
        self.require_response()
        torch.manual_seed(12)
        response = IncidentTimeResponse('conditioned', 12)
        optimizer = torch.optim.SGD(response.parameters(), lr=.1)
        gaussian = torch.exp(-torch.arange(1, 13, dtype=torch.float32).square() / 2).view(1, 12, 1, 1)
        early_gradients = []
        before = response.mlp[0].weight.detach().clone()
        for _ in range(2):
            optimizer.zero_grad()
            loss = (response(gaussian, torch.ones(1, 2, 3))[:, -1] - 1).square().mean()
            loss.backward()
            early_gradients.append(response.mlp[0].weight.grad.abs().sum().item())
            optimizer.step()
        self.assertEqual(early_gradients[0], 0.0)
        self.assertGreater(early_gradients[1], 0.0)
        self.assertGreater((response.mlp[0].weight - before).abs().sum().item(), 0.)


if __name__ == '__main__':
    unittest.main()
