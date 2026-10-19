"""Run only on the server: mask, units and aggregation contracts."""
import unittest

import numpy as np
import torch

from experiments.position_state_gw16.geometry import RegionGeometry
from experiments.position_state_gw16.objective import (
    PooledStatistics, loss_terms, numpy_metrics, region_residual)


class GW16ObjectiveTests(unittest.TestCase):
    def setUp(self):
        self.c = dict(target_minimum=.001, mape_minimum=1., minimum_region_coverage=.5,
                      region_loss_weight=.2)
        self.g = RegionGeometry([0,0,1])

    def test_partial_region_uses_same_observed_subset(self):
        y = torch.tensor([[[[10.],[0.],[20.]]]])
        p = torch.tensor([[[[12.],[500.],[17.]]]])
        error, mask = region_residual(p, y, self.g, self.c)
        self.assertTrue(torch.equal(mask, torch.ones_like(mask)))
        self.assertTrue(torch.equal(error, torch.tensor([[[[2.],[-3.]]]])))

    def test_anchor_is_never_an_extra_training_loss(self):
        y = torch.ones(1,1,3,1)*10
        p = y.clone().requires_grad_(True)
        result = loss_terms(dict(prediction=p, reference=y+100), y, self.g, self.c)
        self.assertEqual(float(result['loss']), 0.)
        result['loss'].backward()
        self.assertIsNotNone(p.grad)

    def test_pooled_statistics_equal_numpy_with_negative_predictions(self):
        y = torch.tensor([[[[10.],[0.],[20.]]]], dtype=torch.float64)
        p = torch.tensor([[[[-2.],[99.],[25.]]]], dtype=torch.float64)
        stats = PooledStatistics(self.c)
        stats.update(dict(prediction=p, reference=y), y, self.g)
        got, expected = stats.result(), numpy_metrics(p.numpy(), y.numpy(), self.c)
        for key in ('mae','rmse','signed_bias','negative_fraction','mape_percent','valid_count'):
            self.assertAlmostEqual(got[key], expected[key])
        self.assertEqual(got['negative_fraction'], .5)

    def test_empty_group_reports_none_without_fake_zero(self):
        got = numpy_metrics(np.zeros((2,)), np.zeros((2,)), self.c)
        self.assertIsNone(got['mae'])
        self.assertEqual(got['valid_count'], 0)

    def test_no_valid_targets_fails(self):
        y = torch.zeros(1,1,3,1)
        with self.assertRaises(ValueError):
            loss_terms(dict(prediction=y, reference=y), y, self.g, self.c)


if __name__ == '__main__':
    unittest.main()
