import copy
import tempfile
import types
import unittest
from pathlib import Path

import torch

from src.models.igstgnn import IGSTGNN, STLocalizedConv


def original_gconv(conv, support, x_k, x_0):
    outputs = [x_0]
    for graph in support:
        if graph.ndim != 2:
            graph = graph.unsqueeze(1)
        outputs.append(torch.matmul(graph, x_k))
    return conv.dropout(conv.gcn_updt(torch.cat(outputs, dim=-1)))


class STLocalizedConvTest(unittest.TestCase):
    def make_conv(self, nodes, device="cpu", dtype=torch.float64):
        return STLocalizedConv(
            hidden_dim=4,
            pre_defined_graph=[torch.eye(nodes, device=device, dtype=dtype)] * 2,
            k_s=2,
            k_t=3,
            dropout=0.0,
            use_pre=True,
            dy_graph=True,
            sta_graph=True,
        ).to(device=device, dtype=dtype)

    def test_gconv_matches_original_outputs_and_gradients(self):
        configurations = [("cpu", torch.float64)]
        if torch.cuda.is_available():
            configurations.append(("cuda", torch.float32))
        for device, dtype in configurations:
            for batch, steps, contiguous in [(1, 1, True), (2, 6, True), (2, 6, False)]:
                with self.subTest(device=device, batch=batch, steps=steps, contiguous=contiguous):
                    torch.manual_seed(2025)
                    nodes = 5
                    conv = self.make_conv(nodes, device, dtype)
                    support = list(conv.pre_defined_graph)
                    support += [
                        torch.randn(batch, nodes, 3 * nodes, device=device, dtype=dtype, requires_grad=True)
                        for _ in range(4)
                    ]
                    support += [
                        torch.randn(nodes, 3 * nodes, device=device, dtype=dtype, requires_grad=True)
                        for _ in range(2)
                    ]
                    x_k = torch.randn(batch, 3 * nodes, steps, 4, device=device, dtype=dtype)
                    x_k = x_k.transpose(1, 2)
                    if contiguous:
                        x_k = x_k.contiguous()
                    x_k.requires_grad_()
                    x_0 = torch.randn(batch, steps, nodes, 4, device=device, dtype=dtype, requires_grad=True)
                    variables = [x_k, x_0] + support[4:] + list(conv.gcn_updt.parameters())

                    expected = original_gconv(conv, support, x_k, x_0)
                    expected_grads = torch.autograd.grad(expected.square().sum(), variables)
                    actual = conv.gconv(support, x_k, x_0)
                    actual_grads = torch.autograd.grad(actual.square().sum(), variables)

                    tolerance = dict(rtol=1e-4, atol=1e-5) if dtype == torch.float32 else dict(rtol=1e-10, atol=1e-10)
                    torch.testing.assert_close(actual, expected, **tolerance)
                    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
                        torch.testing.assert_close(actual_grad, expected_grad, **tolerance)

    def test_backward_does_not_retain_a_time_expanded_dense_graph(self):
        torch.manual_seed(2025)
        batch, steps, nodes = 2, 6, 64
        conv = self.make_conv(nodes, dtype=torch.float32)
        support = list(conv.pre_defined_graph)
        support += [torch.randn(batch, nodes, 3 * nodes, requires_grad=True) for _ in range(4)]
        support += [torch.randn(nodes, 3 * nodes, requires_grad=True) for _ in range(2)]
        x_k = torch.randn(batch, steps, 3 * nodes, 4, requires_grad=True)
        x_0 = torch.randn(batch, steps, nodes, 4, requires_grad=True)
        saved_storage_sizes = []

        def record_storage(tensor):
            saved_storage_sizes.append(tensor.untyped_storage().nbytes())
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(record_storage, lambda tensor: tensor):
            conv.gconv(support, x_k, x_0).square().mean().backward()

        # Retaining a [batch, time, nodes, 3 * nodes] graph is the OOM regression.
        expanded_graph_bytes = batch * steps * nodes * (3 * nodes) * x_k.element_size()
        self.assertTrue(saved_storage_sizes)
        self.assertLess(max(saved_storage_sizes), expanded_graph_bytes)

    def test_full_localized_convolution_matches_original(self):
        torch.manual_seed(2025)
        batch, nodes = 2, 5
        conv = self.make_conv(nodes)
        reference = copy.deepcopy(conv)
        reference.gconv = types.MethodType(original_gconv, reference)
        x = torch.randn(batch, 12, nodes, 4, dtype=torch.float64, requires_grad=True)
        dynamic = [torch.randn(batch, nodes, 3 * nodes, dtype=torch.float64, requires_grad=True) for _ in range(4)]
        static = [torch.randn(nodes, nodes, dtype=torch.float64, requires_grad=True)]
        inputs = [x] + dynamic + static

        expected = reference(x, dynamic, static)
        expected_variables = inputs + list(reference.fc_list_updt.parameters()) + list(reference.gcn_updt.parameters())
        expected_grads = torch.autograd.grad(expected.square().sum(), expected_variables)
        actual = conv(x, dynamic, static)
        actual_variables = inputs + list(conv.fc_list_updt.parameters()) + list(conv.gcn_updt.parameters())
        actual_grads = torch.autograd.grad(actual.square().sum(), actual_variables)

        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-10, atol=1e-10)

    def test_full_model_training_matches_original_with_incidents_and_sensors(self):
        torch.manual_seed(2025)
        batch, nodes = 2, 8
        with tempfile.TemporaryDirectory() as data_dir:
            for name in ("desc_mapping.json", "type_mapping.json"):
                (Path(data_dir) / name).write_text('{"test": 0}', encoding="utf-8")
            model = IGSTGNN(
                model_args=dict(
                    num_feat=1, num_hidden=8, node_hidden=4, time_emb_dim=4,
                    layer=5, k_s=2, k_t=3, tpd=288, dropout=0.1, gap=3,
                    adjs=[torch.eye(nodes)] * 2,
                ),
                node_num=nodes, input_dim=3, output_dim=1, seq_len=12, horizon=12,
                dataset="test", data_path=data_dir, use_sensor_info=True,
            )
        reference = copy.deepcopy(model)
        for module in reference.modules():
            if isinstance(module, STLocalizedConv):
                module.gconv = types.MethodType(original_gconv, module)
        model.load_state_dict(reference.state_dict(), strict=True)

        history = torch.rand(batch, 12, nodes, 3)
        incidents = {
            "incident": torch.zeros(batch, 4),
            "position": torch.zeros(batch, dtype=torch.long),
            "distances": torch.rand(batch, nodes, 3),
        }
        sensors = {
            key: torch.zeros(batch, nodes)
            for key in ("sensor_type", "surface", "roadway_use", "road_width", "speed_limit")
        }
        torch.manual_seed(17)
        expected = reference(history, incident_data=incidents, sensor_data=sensors)
        expected.square().mean().backward()
        torch.manual_seed(17)
        actual = model(history, incident_data=incidents, sensor_data=sensors)
        actual.square().mean().backward()

        self.assertEqual(actual.shape, (batch, 12, nodes, 1))
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
        for (name, parameter), (_, expected_parameter) in zip(model.named_parameters(), reference.named_parameters()):
            with self.subTest(parameter=name):
                if expected_parameter.grad is None:
                    self.assertIsNone(parameter.grad)
                else:
                    self.assertIsNotNone(parameter.grad)
                    torch.testing.assert_close(parameter.grad, expected_parameter.grad, rtol=1e-4, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
