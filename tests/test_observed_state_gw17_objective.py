"""Server-only tests of pre-correction X supervision and masked region loss."""
import unittest

import numpy as np
import torch

from experiments.position_state_gw16.geometry import RegionGeometry
from experiments.observed_state_gw17 import objective as entry


class GW17ObjectiveTests(unittest.TestCase):
    def setUp(self):
        self.config = dict(target_minimum=.001, mape_minimum=1., minimum_region_coverage=.5,
                           region_loss_weight=.2, history_loss_weight=.1,
                           observation_validity='both_adjacent_raw_flows_finite_and_gt_target_minimum')
        self.geometry = RegionGeometry([0, 0, 1])
        self.future = torch.full((1, 1, 3, 1), 10.)

    def output(self):
        return dict(prediction=self.future.clone().requires_grad_(True), reference=self.future,
            history_prediction=torch.tensor([[[[12.], [999.], [17.]]]], requires_grad=True),
            history_target=torch.tensor([[[[10.], [1000.], [20.]]]], requires_grad=True),
            history_valid=torch.tensor([[[[True], [False], [True]]]]))

    def test_same_observed_subset_defines_history_node_and_region_errors(self):
        output = self.output()
        result = entry.loss_terms(output, self.future, self.geometry, self.config)
        self.assertAlmostEqual(float(result['history_node']), 2.5)
        self.assertAlmostEqual(float(result['history_region']), 2.5)
        self.assertAlmostEqual(float(result['history_loss']), 3.)
        self.assertAlmostEqual(float(result['forecast_loss']), 0.)
        self.assertAlmostEqual(float(result['loss']), .3, places=6)

    def test_history_targets_are_detached_and_masked_cells_receive_no_gradient(self):
        output = self.output()
        entry.loss_terms(output, self.future, self.geometry, self.config)['loss'].backward()
        self.assertIsNone(output['history_target'].grad)
        self.assertEqual(float(output['history_prediction'].grad[0, 0, 1, 0]), 0.)
        self.assertGreater(float(output['history_prediction'].grad.abs().sum()), 0.)

    def test_future_Y_cannot_change_the_historical_auxiliary_terms(self):
        output = self.output()
        first = entry.loss_terms(output, self.future, self.geometry, self.config)
        second = entry.loss_terms(output, self.future+50, self.geometry, self.config)
        self.assertEqual(float(first['history_loss']), float(second['history_loss']))
        self.assertNotEqual(float(first['forecast_loss']), float(second['forecast_loss']))

    def test_no_adjacent_valid_history_contributes_differentiable_zero(self):
        output = self.output()
        output['history_valid'].fill_(False)
        result = entry.loss_terms(output, self.future, self.geometry, self.config)
        self.assertEqual(float(result['history_loss']), 0.)
        result['loss'].backward()
        self.assertTrue(torch.equal(output['history_prediction'].grad,
                                    torch.zeros_like(output['history_prediction'])))

    def test_invalid_prediction_and_mismatched_history_axes_are_rejected(self):
        output = self.output()
        output['history_prediction'] = torch.full((1, 1, 3, 1), float('nan'))
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            entry.loss_terms(output, self.future, self.geometry, self.config)
        output = self.output()
        output['history_valid'] = output['history_valid'][:, :, :2]
        with self.assertRaises((ValueError, RuntimeError)):
            entry.loss_terms(output, self.future, self.geometry, self.config)

    def test_history_metric_uses_adjacent_valid_mask_not_future_masks(self):
        p = np.asarray([1., 999., 5.], dtype=np.float32)
        y = np.asarray([2., 1., 3.], dtype=np.float32)
        valid = np.asarray([True, False, True])
        score = entry.numpy_metrics(p, y, self.config, valid)
        self.assertEqual(score['valid_count'], 2)
        self.assertAlmostEqual(score['mae'], 1.5)


if __name__ == '__main__':
    unittest.main()
