"""Server-only numerical contracts; no PyTorch/model imports or model execution."""
import json
import unittest

import numpy as np

from experiments.graph_memory_gw6 import diagnose_layers as layers


class LayerDiagnosticTests(unittest.TestCase):
    def test_history_amplitude_uses_numerical_history_and_saved_scale(self):
        x = np.zeros((1, 2, 2, 3), dtype=np.float32)
        x[0, :, 0, 0] = [-2., 1.]
        x[0, :, 1, 0] = [0., 2.]
        x[..., 1:] = 1000.  # Clock features must not enter amplitude bins.
        np.testing.assert_allclose(layers.historical_amplitude(x, 2., 2.), [[3., 4.]])

    def test_history_amplitude_rejects_invalid_axis_scale_and_nonfinite(self):
        x = np.ones((2, 3, 4, 1))
        for invalid in (0., -1., float('nan')):
            with self.assertRaises(ValueError):
                layers.historical_amplitude(x, 0., invalid)
        with self.assertRaises(ValueError):
            layers.historical_amplitude(x[:, 0], 0., 1.)
        with self.assertRaises(ValueError):
            layers.historical_amplitude(x, 0., 1., num_feat=2)
        x[0, 0, 0, 0] = np.inf
        with self.assertRaises(ValueError):
            layers.historical_amplitude(x, 0., 1.)

    def test_activation_moments_and_negative_fraction_have_known_values(self):
        statistics = layers.ActivationStatistics()
        value = np.array([-3., -1., 1., 3.]).reshape(1, 2, 2, 1)
        statistics.add('best', 'known', value, np.array([[0, 1]]), 2)
        overall = next(row for row in statistics.rows() if row['axis'] == 'all')
        self.assertEqual(overall['count'], 4)
        self.assertEqual(overall['mean'], 0.)
        self.assertAlmostEqual(overall['rms'], np.sqrt(5.))
        self.assertEqual(overall['negative_fraction'], .5)
        groups = [row for row in statistics.rows() if row['axis'] == 'historical_amplitude']
        self.assertEqual([row['mean'] for row in groups], [-1., 1.])

    def test_streamed_batches_combine_by_element_exposure(self):
        statistics = layers.ActivationStatistics()
        statistics.add('best', 'known', np.full((2, 1, 1, 1), 2.), np.zeros((2, 1), dtype=int), 1)
        statistics.add('best', 'known', np.full((1, 1, 1, 1), -1.), np.zeros((1, 1), dtype=int), 1)
        overall = next(row for row in statistics.rows() if row['axis'] == 'all')
        self.assertEqual(overall['count'], 3)
        self.assertEqual(overall['mean'], 1.)
        self.assertAlmostEqual(overall['rms'], np.sqrt(3.))
        self.assertAlmostEqual(overall['negative_fraction'], 1. / 3.)

    def test_empty_history_groups_are_not_fabricated(self):
        statistics = layers.ActivationStatistics()
        statistics.add('best', 'known', np.ones((1, 1, 1, 1)), np.array([[0]]), 3)
        groups = [row['group'] for row in statistics.rows() if row['axis'] == 'historical_amplitude']
        self.assertEqual(groups, [0])
        json.dumps(statistics.rows(), allow_nan=False)

    def test_activation_axes_and_nonfinite_values_are_rejected(self):
        statistics = layers.ActivationStatistics()
        with self.assertRaises(ValueError):
            statistics.add('best', 'known', np.ones((1, 2, 3)), np.zeros((1, 3), dtype=int), 1)
        with self.assertRaises(ValueError):
            statistics.add('best', 'known', np.ones((1, 2, 3, 4)), np.zeros((3, 1), dtype=int), 1)
        with self.assertRaises(ValueError):
            statistics.add('best', 'known', np.ones((1, 2, 3, 4)), np.full((1, 3), 2), 1)
        with self.assertRaises(ValueError):
            statistics.add('best', 'known', np.full((1, 1, 1, 1), np.nan), np.array([[0]]), 1)

    def test_endpoint_rms_difference_is_not_pointwise_change_rms(self):
        statistics = layers.ActivationStatistics()
        labels = np.array([[0, 0]])
        statistics.add('best', 'known', np.array([-1., 1.]).reshape(1, 1, 2, 1), labels, 1)
        statistics.add('last', 'known', np.array([1., -1.]).reshape(1, 1, 2, 1), labels, 1)
        row = next(row for row in layers.statistics_changes(statistics.rows()) if row['axis'] == 'all')
        self.assertEqual(row['rms_change'], 0.)
        self.assertEqual(row['mean_change'], 0.)
        # Each actual activation changed by magnitude two; endpoints have equal RMS.
        self.assertNotEqual(row['rms_change'], 2.)

    def test_missing_endpoint_and_exposure_mismatch_are_rejected(self):
        statistics = layers.ActivationStatistics()
        statistics.add('best', 'known', np.ones((1, 1, 1, 1)), np.array([[0]]), 1)
        with self.assertRaises(ValueError):
            layers.statistics_changes(statistics.rows())
        statistics.add('last', 'known', np.ones((2, 1, 1, 1)), np.zeros((2, 1), dtype=int), 1)
        with self.assertRaises(ValueError):
            layers.statistics_changes(statistics.rows())

    def test_replay_tolerance_is_explicit_and_nonfinite_is_not_ignored(self):
        reference = np.array([1., 100.])
        result = layers.compare_arrays(reference + np.array([1e-6, 1e-4]), reference)
        self.assertTrue(result['passed'])
        self.assertFalse(layers.compare_arrays(reference + .1, reference)['passed'])
        self.assertEqual(result['shape'], [2])
        for value in (np.array([1., np.nan]), np.array([1., np.inf])):
            with self.assertRaises(ValueError):
                layers.compare_arrays(value, reference)
        with self.assertRaises(ValueError):
            layers.compare_arrays(reference.reshape(2, 1), reference)

    def test_gap_selection_uses_lead_modulo_three_without_reordering_nodes(self):
        channels = np.arange(1 * 12 * 2 * 3).reshape(1, 12, 2, 3)
        actual = layers.select_gap_numpy(channels, 3)
        self.assertEqual(actual.shape, (1, 12, 2, 1))
        np.testing.assert_array_equal(actual[0, :, 0, 0],
            [0, 7, 14, 18, 25, 32, 36, 43, 50, 54, 61, 68])
        np.testing.assert_array_equal(actual[0, :, 1, 0], actual[0, :, 0, 0] + 3)
        with self.assertRaises(ValueError):
            layers.select_gap_numpy(channels, 2)

    def test_crossed_heads_separate_latent_and_head_conditionals(self):
        latent_best = np.array([[[[-2., 2.]], [[3., -1.]], [[1., 4.]]]])
        latent_last = latent_best * 2.
        head_best = dict(w1=np.eye(2), b1=np.zeros(2),
            w2=np.array([[1., 0.], [0., 1.], [1., 1.]]), b2=np.zeros(3))
        head_last = dict(head_best, b2=np.full(3, 5.))
        bb = layers.head_predictions_numpy(latent_best, head_best, 3)
        bl = layers.head_predictions_numpy(latent_best, head_last, 3)
        lb = layers.head_predictions_numpy(latent_last, head_best, 3)
        ll = layers.head_predictions_numpy(latent_last, head_last, 3)
        np.testing.assert_array_equal(bb.reshape(-1), [0., 0., 5.])
        np.testing.assert_array_equal(bl, bb + 5.)
        np.testing.assert_array_equal(lb, bb * 2.)
        np.testing.assert_array_equal(ll, bb * 2. + 5.)

    def test_zero_latent_retains_learned_head_biases_and_relu_order(self):
        latent = np.zeros((1, 3, 1, 2))
        head = dict(w1=np.eye(2), b1=np.array([-1., 2.]),
            w2=np.array([[10., 1.], [10., 2.], [10., 3.]]), b2=np.array([1., 2., 3.]))
        result = layers.head_predictions_numpy(latent, head, 3)
        np.testing.assert_array_equal(result.reshape(-1), [3., 6., 9.])


if __name__ == '__main__':
    unittest.main()
