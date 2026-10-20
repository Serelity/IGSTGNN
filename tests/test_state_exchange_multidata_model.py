"""Server-only parity of the selected GW16 architecture across vocabularies."""
import unittest

import torch

from experiments.position_state_gw16.model import (
    PositionStateModel, build_model as build_gw16)
from experiments.state_exchange_multidata.model import DatasetStateExchange, build_model
from tests.test_position_state_gw16_model import fixture


class MultiDatasetModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2025)
        self.data, self.config, self.history, self.event = fixture()
        self.data.event_counts = dict(type=6, description=29)

    def test_contra_constructor_matches_every_original_tensor_and_forward_bit(self):
        torch.manual_seed(2025)
        original = build_gw16(self.data, self.config, 'STATE_EXCHANGE', 'cpu').eval()
        torch.manual_seed(2025)
        adapted = build_model(self.data, self.config, 'STATE_EXCHANGE', 'cpu').eval()
        self.assertEqual(set(original.state_dict()), set(adapted.state_dict()))
        for name, value in original.state_dict().items():
            self.assertTrue(torch.equal(value, adapted.state_dict()[name]), name)
        with torch.no_grad():
            left, right = original(self.history, self.event), adapted(self.history, self.event)
        for key in left:
            self.assertTrue(torch.equal(left[key], right[key]), key)

    def test_history_exchange_and_forecast_forward_are_inherited(self):
        self.assertIs(DatasetStateExchange.forward, PositionStateModel.forward)
        self.assertIs(DatasetStateExchange.set_stage, PositionStateModel.set_stage)

    def test_each_description_vocabulary_accepts_its_last_known_and_unknown_id(self):
        for count in (29, 30, 32):
            with self.subTest(count=count):
                self.data.event_counts = dict(type=6, description=count)
                model = build_model(self.data, self.config, 'STATE_EXCHANGE', 'cpu').eval()
                event = dict(self.event, description=torch.tensor([count-1, count]),
                             description_valid=torch.tensor([True, False]))
                output = model(self.history, event)
                self.assertTrue(torch.isfinite(output['prediction']).all())
                self.assertEqual(model.incident.description_embedding.num_embeddings, count+1)

    def test_unknown_description_uses_its_own_release_token(self):
        self.data.event_counts = dict(type=6, description=32)
        model = build_model(self.data, self.config, 'STATE_EXCHANGE', 'cpu').eval()
        event = dict(self.event, description_valid=torch.zeros(2, dtype=torch.bool))
        event2 = dict(event, description=torch.tensor([-999, 999]))
        with torch.no_grad():
            left, right = model(self.history, event), model(self.history, event2)
        self.assertTrue(torch.equal(left['prediction'], right['prediction']))

    def test_only_embedding_rows_change_parameter_count(self):
        counts = {}
        for count in (29, 30, 32):
            self.data.event_counts = dict(type=6, description=count)
            model = build_model(self.data, self.config, 'STATE_EXCHANGE', 'cpu')
            counts[count] = sum(p.numel() for p in model.parameters())
        self.assertEqual(counts[30]-counts[29], 16)
        self.assertEqual(counts[32]-counts[29], 48)

    def test_highest_description_row_receives_task_gradient(self):
        self.data.event_counts = dict(type=6, description=32)
        model = build_model(self.data, self.config, 'STATE_EXCHANGE', 'cpu')
        event = dict(self.event, description=torch.full((2,), 31, dtype=torch.long))
        model(self.history, event)['prediction'].square().mean().backward()
        gradient = model.incident.description_embedding.weight.grad[31]
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient.abs().sum()), 0.)

    def test_other_mechanism_arms_cannot_replace_the_selected_method(self):
        with self.assertRaises(ValueError):
            build_model(self.data, self.config, 'ORIGIN_EXCHANGE', 'cpu')


if __name__ == '__main__':
    unittest.main()
