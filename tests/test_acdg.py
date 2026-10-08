import unittest

import torch

from src.models.igstgnn import ExplicitIncidentCondition, IGSTGNN


def make_model(incident_routing='none'):
    supports = [torch.eye(3), torch.eye(3)]
    return IGSTGNN(
        model_args=dict(
            num_feat=1,
            num_hidden=8,
            node_hidden=4,
            time_emb_dim=4,
            layer=2,
            k_s=1,
            k_t=2,
            tpd=288,
            dropout=0.0,
            gap=3,
            sigma_t=1.0,
            lambda_incident=1.0,
            adjs=supports,
            incident_schema='report_location_v1',
            time_response='fixed',
            incident_routing=incident_routing,
        ),
        node_num=3,
        input_dim=3,
        output_dim=1,
        seq_len=12,
        horizon=12,
        dataset='test',
        use_sensor_info=False,
    )


def incident_batch():
    return {
        'report_age_minutes': torch.tensor([10.0, 15.0]),
        'forecast_tod': torch.tensor([10, 11], dtype=torch.long),
        'forecast_dow': torch.tensor([1, 2], dtype=torch.long),
        'distances': torch.tensor([
            [[1.0, 0.2, 0.0], [0.0, 0.0, 0.0], [0.4, 0.1, 1.0]],
            [[0.0, 0.0, 0.0], [0.3, 0.1, 0.0], [0.2, 0.2, 0.0]],
        ]),
    }


class ACDGTest(unittest.TestCase):
    def test_acdg_common_initialization_and_output_match_native_gate(self):
        torch.manual_seed(20261008)
        native = make_model('none').eval()
        torch.manual_seed(20261008)
        acdg = make_model('acdg').eval()

        common = set(native.state_dict()) & set(acdg.state_dict())
        self.assertTrue(common)
        for name in common:
            torch.testing.assert_close(
                native.state_dict()[name], acdg.state_dict()[name], atol=0, rtol=0)

        history = torch.rand(2, 12, 3, 3)
        with torch.inference_mode():
            native_prediction = native(history, incident_data=incident_batch())
            acdg_prediction = acdg(history, incident_data=incident_batch())
            native_without_incident = native(history)
            acdg_without_incident = acdg(history)
        torch.testing.assert_close(acdg_prediction, native_prediction, atol=0, rtol=0)
        torch.testing.assert_close(acdg_without_incident, native_without_incident, atol=0, rtol=0)

    def test_condition_is_zero_on_disconnected_nodes_after_training(self):
        module = ExplicitIncidentCondition(hidden_dim=4, condition_hidden=8)
        with torch.no_grad():
            module.mlp[-1].weight.fill_(1.0)
            module.mlp[-1].bias.fill_(0.25)
        history = torch.ones(1, 3, 2, 4)
        inputs = {
            'incident_embedding': torch.ones(1, 4),
            'distances': torch.tensor([[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]),
        }
        delta = module(history, inputs)
        self.assertTrue(torch.isfinite(delta).all())
        self.assertGreater(float(delta[:, :, 0].abs().sum()), 0.0)
        torch.testing.assert_close(delta[:, :, 1], torch.zeros_like(delta[:, :, 1]))

    def test_acdg_branch_is_connected_and_updates(self):
        torch.manual_seed(9)
        model = make_model('acdg')
        history = torch.rand(2, 12, 3, 3)
        output = model(history, incident_data=incident_batch())
        loss = output.square().mean()
        loss.backward()
        route_parameters = [
            parameter for name, parameter in model.named_parameters()
            if '.incident_condition.' in name
        ]
        self.assertTrue(route_parameters)
        self.assertTrue(all(parameter.grad is not None for parameter in route_parameters))
        self.assertTrue(all(torch.isfinite(parameter.grad).all() for parameter in route_parameters))
        final_gradients = [
            parameter.grad.abs().sum()
            for name, parameter in model.named_parameters()
            if '.incident_condition.mlp.2.' in name
        ]
        self.assertTrue(final_gradients)
        self.assertGreater(float(sum(final_gradients)), 0.0)


if __name__ == '__main__':
    unittest.main()
