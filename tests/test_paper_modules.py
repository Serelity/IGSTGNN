import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from src.models.igstgnn import (
    IGSTGNN, IncidentContextSpatialFusion, TemporalIncidentImpactDecay,
)


def make_model(horizon=12):
    with tempfile.TemporaryDirectory() as directory:
        for name in ("desc_mapping.json", "type_mapping.json"):
            (Path(directory) / name).write_text('{"test": 0}', encoding="utf-8")
        return IGSTGNN(
            model_args=dict(
                num_feat=1, num_hidden=8, node_hidden=4, time_emb_dim=4,
                layer=5, k_s=2, k_t=3, tpd=288, dropout=0.0, gap=3,
                adjs=[torch.eye(3)] * 2,
            ),
            node_num=3, input_dim=3, output_dim=1, seq_len=12, horizon=horizon,
            dataset="test", data_path=directory, use_sensor_info=False,
        )


class PaperModulesTest(unittest.TestCase):
    def test_single_incident_aggregation_does_not_compete_across_nodes(self):
        module = IncidentContextSpatialFusion(4, 2, False, 1, 1).eval()
        with torch.no_grad():
            module.incident_fusion[-1].weight.zero_()
            module.incident_fusion[-1].bias.copy_(torch.tensor([1., 2., 0., 0.]))
            module.v_proj.weight.copy_(torch.eye(4))
        history = torch.tensor([0., 1., 2., 3.]).expand(1, 12, 3, 4).clone()
        incidents = {
            "incident": torch.zeros(1, 4), "position": torch.tensor([11]),
            "distances": torch.tensor([[[1., 0., 0.], [.5, .2, 1.], [0., 0., 0.]]]),
        }
        actual, _ = module(history, incidents)
        # Eq. 6-8 with one incident: weight 1 at each connected node, 0 elsewhere.
        expected = F.layer_norm(torch.tensor([[1., 3., 2., 3.], [1., 3., 2., 3.], [0., 1., 2., 3.]]), (4,))
        torch.testing.assert_close(actual[0, -1], expected)
        torch.testing.assert_close(actual[:, :-1], history[:, :-1])

    def test_tiid_independent_context_is_masked_and_decays_each_future_step(self):
        module = TemporalIncidentImpactDecay(incident_dim=2, forecast_dim=2)
        self.assertTrue(hasattr(module, "context_fusion"), "Eq.13 needs its own context MLP")
        with torch.no_grad():
            for parameter in module.context_fusion.parameters():
                parameter.zero_()
            module.context_fusion[-1].bias.fill_(1)
            module.context_projection.weight.copy_(torch.eye(2))
        hidden = torch.zeros(1, 12, 2, 2)
        actual = module(
            hidden, incident_key=torch.tensor([[2., 3.]]),
            sensor_features=torch.empty(1, 2, 0),
            distances=torch.tensor([[[1., 0., 0.], [0., 0., 0.]]]),
        )
        expected = torch.tensor([
            .6065306597, .1353352832, .01110899654, .000335462628,
            3.726653172e-6, 1.522997974e-8, 2.289734846e-11, 1.266416555e-14,
            2.576757109e-18, 1.928749848e-22, 5.311092250e-27, 5.380186160e-32,
        ])
        torch.testing.assert_close(actual[0, :, 0, 0], expected, atol=0, rtol=1e-5)
        torch.testing.assert_close(actual[:, :, 1], torch.zeros(1, 12, 2))

    def test_tiid_context_uses_incident_key_sensor_and_distance(self):
        module = TemporalIncidentImpactDecay(2, 2, sensor_dim=2)
        self.assertTrue(hasattr(module, "context_fusion"), "Missing Eq.13 context construction")
        with torch.no_grad():
            for parameter in module.context_fusion.parameters():
                parameter.fill_(.1)
            module.context_projection.weight.fill_(.2)
        hidden = torch.zeros(1, 12, 1, 2)
        key = torch.ones(1, 2, requires_grad=True)
        sensors = torch.ones(1, 1, 2, requires_grad=True)
        distances = torch.ones(1, 1, 3, requires_grad=True)
        output = module(hidden, key, sensors, distances)
        output.sum().backward()
        self.assertTrue(torch.all(key.grad > 0))
        self.assertTrue(torch.all(sensors.grad > 0))
        self.assertTrue(torch.all(distances.grad > 0))

    def test_step_expansion_preserves_original_gap_channel_order_without_tiid(self):
        model = make_model().eval()
        self.assertTrue(hasattr(model, "_decode_forecast"))
        hidden = torch.randn(2, 4, 3, 256)
        grouped = model.out_fc_2(F.relu(model.out_fc_1(F.relu(hidden))))
        expected = grouped.transpose(1, 2).reshape(2, 3, 12).transpose(1, 2).unsqueeze(-1)
        actual = model._decode_forecast(hidden, None)
        torch.testing.assert_close(actual, expected)

    def test_full_model_tiid_sees_twelve_steps_and_backward_is_finite(self):
        torch.manual_seed(2025)
        model = make_model()
        seen_lengths = []
        handle = model.tiid_module.register_forward_pre_hook(
            lambda module, args: seen_lengths.append(args[0].shape[1]))
        history = torch.rand(2, 12, 3, 3)
        incidents = {
            "incident": torch.zeros(2, 4), "position": torch.tensor([11, 11]),
            "distances": torch.ones(2, 3, 3),
        }
        output = model(history, incident_data=incidents)
        handle.remove()
        self.assertEqual(seen_lengths, [12])
        self.assertEqual(output.shape, (2, 12, 3, 1))
        output.square().mean().backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(g).all() for g in gradients))

    def test_forecast_uses_requested_horizon_instead_of_history_length(self):
        model = make_model(horizon=5).eval()
        with torch.no_grad():
            output = model(torch.rand(2, 12, 3, 3))
        self.assertEqual(output.shape, (2, 5, 3, 1))


if __name__ == "__main__":
    unittest.main()
