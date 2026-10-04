"""Temporal weighting math, paired bootstrap, frozen protocol and real gradient checks."""

import copy
import unittest

import numpy as np
import torch

from experiments.chronological import train_temporal_robust_correction as e
from test_incident_strength_gate import manifests, SyntheticDataset, tiny_model


class MathTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.p = e.load_protocol()
        self.m = e.method

    def test_protocol_and_original_sources_are_frozen(self):
        self.assertEqual(e.base.sha256(e.PROTOCOL), e.PROTOCOL_SHA256)
        self.assertEqual(self.p['budget']['trajectory_epochs'], 144)
        self.assertEqual(self.p['fit_groups']['expected_windows'], [353, 364, 414, 389])

    def test_half_open_groups_use_only_t0_and_require_all_groups(self):
        rows = [{'t0': s, 'y': float('nan')} for s in [
            '2023-01-02T00:00:00', '2023-01-30T00:00:00', '2023-02-27T00:00:00', '2023-03-27T00:00:00']]
        self.assertEqual(self.m.fit_groups(rows, list(range(4)), self.p), dict(enumerate(range(4))))
        rows[3]['t0'] = '2023-04-24T00:00:00'
        with self.assertRaises(ValueError):
            self.m.fit_groups(rows, list(range(4)), self.p)
        with self.assertRaises(ValueError):
            self.m.fit_groups(rows, [0, 1, 2], self.p)

    def test_common_scale_excess_and_coverage(self):
        baseline = {'counts': [1, 2, 3, 4], 'errors': [1., 4., 9., 16.], 'mae': [1., 2., 3., 4.]}
        pi, scale = self.m.reference_weights(baseline, self.p['weighting'])
        current = {'counts': baseline['counts'], 'errors': [2., 6., 6., 16.], 'mae': [2., 3., 2., 4.]}
        actual = self.m.update_weights(pi, pi, current, baseline, self.p['weighting'])
        logits = np.log(pi) + 20 * (np.array([1, 1, -1, 0]) / scale)
        v = np.exp(logits - logits.max()); v /= v.sum()
        np.testing.assert_array_equal(actual, .05 * pi + .95 * v)
        self.assertAlmostEqual((actual[0] - .05 * pi[0]) / (actual[1] - .05 * pi[1]), pi[0] / pi[1])
        self.assertTrue(np.all(actual / pi >= .05))
        np.testing.assert_allclose(self.m.update_weights(pi, pi, baseline, baseline, self.p['weighting']), pi, atol=1e-16)
        for bad in ([0, .2, .3, .5], [float('nan'), .2, .3, .5], [1, 2, 3, 4]):
            with self.assertRaises(ValueError):
                self.m.distribution(bad)
        baseline['errors'] = [0.] * 4
        with self.assertRaises(ValueError):
            self.m.reference_weights(baseline, self.p['weighting'])

    def test_weighted_loss_original_denominator_masking_and_gradient(self):
        prediction = torch.arange(48, dtype=torch.float32).reshape(4, 12, 1, 1).requires_grad_()
        target = prediction.detach() - 1
        valid = torch.ones_like(prediction, dtype=torch.bool)
        valid[1, :3] = False
        target[~valid] = float('nan')
        candidate = torch.ones((4, 1), dtype=torch.bool)
        groups = torch.arange(4)
        pi, q = np.ones(4) / 4, np.array([.1, .2, .3, .4])
        loss, count = self.m.weighted_loss(prediction, target, valid, candidate, groups, q, pi)
        self.assertEqual(count, 21)
        self.assertAlmostEqual(float(loss), (6 * .4 + 3 * .8 + 6 * 1.2 + 6 * 1.6) / 21, places=6)
        loss.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertEqual(float(prediction.grad[~valid].sum()), 0)
        original, _ = e.alignment.early_loss(prediction, target, valid, candidate)
        unit, _ = self.m.weighted_loss(prediction, target, valid, candidate, groups, pi, pi)
        self.assertTrue(torch.equal(original, unit))

    def test_unit_weights_match_original_gradients_exactly(self):
        a = torch.randn(4, 12, 3, 1, requires_grad=True)
        b = a.detach().clone().requires_grad_()
        target, valid = torch.randn_like(a), torch.ones_like(a, dtype=torch.bool)
        candidate = torch.tensor([[True, False, True]] * 4)
        loss, _ = e.alignment.early_loss(a, target, valid, candidate)
        paired, _ = self.m.weighted_loss(b, target, valid, candidate, torch.arange(4), np.ones(4)/4, np.ones(4)/4)
        loss.backward(); paired.backward()
        self.assertTrue(torch.equal(loss, paired))
        self.assertTrue(torch.equal(a.grad, b.grad))

    def test_bootstrap_direct_pairing_and_missing_support(self):
        ds = SyntheticDataset()
        record = e.original.evaluate(tiny_model(), ds, [0, 1], 16, torch.device('cpu'))
        # Deterministic regional gains on a shared two-window subset.
        endpoints = {}
        for arm in self.m.ARMS:
            for objective, reduction in (('erm', .1), ('temporal_excess', .3)):
                other = copy.deepcopy(record)
                early = e.base.REGIONS.index('candidate_h1_h6')
                # Change all candidate early component regions consistently.
                for region in ('candidate_h1_h3', 'candidate_h4_h6'):
                    self.assertIn(region, e.base.REGIONS)
                for j, region in enumerate(e.base.REGIONS):
                    if region in ('candidate_h1_h3', 'candidate_h4_h6', 'candidate_h1_h6', 'candidate_all', 'all'):
                        factor = record['counts'][:, j] if region not in ('candidate_all', 'all') else record['counts'][:, early]
                        other['errors'][:, j] -= reduction * factor
                endpoints[self.m.endpoint(arm, objective)] = {'incident_full': other}
        p = copy.deepcopy(self.p); p['bootstrap']['draws'] = 100
        result, _, arrays = self.m.compare_phase({'incident_full': record}, endpoints,
            {11: '2023-01-10T12:00:00', 73: '2023-01-17T12:00:00'}, 'fit', p)
        effect = result['results']['incident_full']['candidate_h1_h6']['comparisons']['state_vector__temporal_vs_erm']
        self.assertAlmostEqual(effect['gain'], .2)
        self.assertAlmostEqual(effect['equal_forecast_window_gain'], .2)
        self.assertEqual(len(arrays['weeks']), 16)  # Empty calendar weeks retained.
        for method in effect['intervals'].values():
            for ci in method.values():
                if ci['status'] == 'OK':
                    self.assertAlmostEqual(ci['ci_low'], .2)
                    self.assertAlmostEqual(ci['ci_high'], .2)
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            self.m.compare_phase({'incident_full': record}, {}, {}, 'fit', p)


if __name__ == '__main__':
    unittest.main()
