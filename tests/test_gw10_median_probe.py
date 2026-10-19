"""Server-only numerical and isolation checks for fixed GW10 median tables."""
from copy import deepcopy
import inspect
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.conditional_residual_gw10 import median_probe as probe


def fixture(samples=384, horizon=2, nodes=2):
    prediction = np.full((samples, horizon, nodes), 20., dtype=np.float64)
    features = dict(last_flow=np.full((samples, nodes), 20.),
        mean_abs_change=np.ones((samples, nodes)),
        neighbor_change_disagreement=np.ones((samples, nodes)))
    return prediction, prediction.copy(), np.ones(prediction.shape, dtype=np.bool_), features, np.arange(samples)


class MedianProbeTests(unittest.TestCase):
    def test_residual_sign_is_direct_median_not_mean_or_negative(self):
        pred, truth, valid, features, ids = fixture()
        residual = np.full(pred.shape, 3.); residual[0] = 1000.
        model = probe.fit_probe(pred, truth+residual, valid, features, ids)
        corrections = probe.predict_corrections(model, features, horizon=2)
        self.assertEqual(model['tables']['GLOBAL'], 3.)
        self.assertTrue(all(np.all(value == 3.) for value in corrections.values()))
        self.assertTrue(np.all(pred+corrections['JOINT'] > pred))

    def test_zero_residual_recovers_predictions_exactly_all_arms(self):
        pred, truth, valid, features, ids = fixture()
        model = probe.fit_probe(pred, truth, valid, features, ids)
        for name, correction in probe.predict_corrections(model, features).items():
            self.assertEqual(correction.dtype, np.float64, name)
            np.testing.assert_array_equal(correction, np.zeros(pred.shape))
            np.testing.assert_array_equal(pred+correction, pred)

    def test_cutpoints_use_all_train_inputs_and_do_not_move_at_prediction(self):
        pred, truth, valid, features, ids = fixture()
        features['mean_abs_change'] = np.arange(768).reshape(384, 2).astype(float)
        features['neighbor_change_disagreement'] = features['mean_abs_change']*2
        valid[300:] = False
        model = probe.fit_probe(pred, truth, valid, features, ids)
        expected = np.quantile(features['mean_abs_change'], [.25, .5, .75])
        np.testing.assert_array_equal(model['cutpoints']['mean_abs_change']['quartiles'], expected)
        serialized = json.dumps(model, allow_nan=False, sort_keys=True)
        shifted = {name: value+10000 for name, value in features.items()}
        probe.predict_corrections(model, shifted)
        self.assertEqual(serialized, json.dumps(model, allow_nan=False, sort_keys=True))

    def test_support_requires_three_distinct_author_order_blocks(self):
        pred, truth, valid, features, ids = fixture()
        features['mean_abs_change'][:128] = 0.
        features['mean_abs_change'][128:256] = 1.
        features['mean_abs_change'][256:] = 2.
        truth[:128] += 9.
        model = probe.fit_probe(pred, truth, valid, features, ids)
        correction = probe.predict_corrections(model, features)
        np.testing.assert_array_equal(correction['OWN2'], correction['ORIGIN'])
        self.assertEqual(model['stats']['OWN2']['eligible_cells'], 0)
        self.assertGreater(model['stats']['OWN2']['fallback_valid_count'], 0)
        self.assertEqual(model['training']['block_count'], 3)

    def test_sparse_origin_cells_back_off_to_node_horizon(self):
        pred, truth, valid, features, ids = fixture()
        features['last_flow'][0] = 314.
        truth[0] += 100.
        model = probe.fit_probe(pred, truth, valid, features, ids)
        correction = probe.predict_corrections(model, features)
        np.testing.assert_array_equal(correction['ORIGIN'][0], correction['NODE_HORIZON'][0])
        self.assertEqual(model['support']['ORIGIN']['valid_count'][0][0][7], 1)
        self.assertEqual(model['support']['ORIGIN']['block_count'][0][0][7], 1)

    def test_sixty_four_points_are_required_even_when_three_blocks_exist(self):
        pred, truth, valid, features, ids = fixture(horizon=1, nodes=1)
        selected = np.r_[np.arange(22), np.arange(128, 149), np.arange(256, 277)]
        features['last_flow'][selected] = 5.
        truth[selected] += 10.
        model = probe.fit_probe(pred, truth, valid, features, ids)
        self.assertEqual(model['support']['ORIGIN']['valid_count'][0][0][2], 64)
        self.assertEqual(model['support']['ORIGIN']['block_count'][0][0][2], 3)
        self.assertAlmostEqual(model['tables']['ORIGIN'][0][0][2], 64*10./(64+128))
        features['last_flow'][selected[-1]] = 20.
        sparse = probe.fit_probe(pred, truth, valid, features, ids)
        self.assertEqual(sparse['support']['ORIGIN']['valid_count'][0][0][2], 63)
        self.assertEqual(sparse['tables']['ORIGIN'][0][0][2], sparse['tables']['NODE_HORIZON'][0][0])

    def test_node_horizon_and_flow_controls_follow_fixed_shrinkage(self):
        pred, truth, valid, features, ids = fixture(samples=768)
        truth[:, 0, 0] += 2.; truth[:, 1, 0] += 4.
        truth[:, 0, 1] += 6.; truth[:, 1, 1] += 8.
        features['last_flow'][np.arange(768) % 2 == 0] = 5.
        truth[np.arange(768) % 2 == 0] += 2.
        model = probe.fit_probe(pred, truth, valid, features, ids)
        residual = truth-pred
        global_median = np.median(residual)
        expected_h = (1536*np.median(residual[:, 0, :])+128*global_median)/(1536+128)
        expected_nh = (768*np.median(residual[:, 0, 0])+128*expected_h)/(768+128)
        expected_origin = (384*np.median(residual[::2, 0, 0])+128*expected_nh)/(384+128)
        self.assertAlmostEqual(model['tables']['HORIZON'][0], expected_h)
        self.assertAlmostEqual(model['tables']['NODE_HORIZON'][0][0], expected_nh)
        self.assertAlmostEqual(model['tables']['ORIGIN'][0][0][2], expected_origin)
        self.assertNotEqual(model['tables']['NODE_HORIZON'][0][0], model['tables']['NODE_HORIZON'][1][0])

    def test_four_joint_states_use_own_times_two_plus_neighbor(self):
        pred, truth, valid, features, ids = fixture(samples=768, horizon=1, nodes=1)
        state = np.arange(768) % 4
        features['mean_abs_change'][:, 0] = state//2
        features['neighbor_change_disagreement'][:, 0] = state % 2
        truth[:, 0, 0] += state*4.
        model = probe.fit_probe(pred, truth, valid, features, ids)
        correction = probe.predict_corrections(model, features)['JOINT']
        origin = model['tables']['ORIGIN'][0][0][3]
        for value in range(4):
            expected = (192*(value*4.)+128*origin)/(192+128)
            self.assertAlmostEqual(model['tables']['JOINT'][0][0][3][value], expected)
            self.assertAlmostEqual(correction[value, 0, 0], expected)
        self.assertEqual(model['stats']['JOINT']['eligible_cells'], 4)

    def test_own_four_uses_training_quartiles_beyond_two_median_states(self):
        pred, truth, valid, features, ids = fixture(samples=768, horizon=1, nodes=1)
        state = np.arange(768) % 4
        features['mean_abs_change'][:, 0] = state
        truth[:, 0, 0] += state*4.
        model = probe.fit_probe(pred, truth, valid, features, ids)
        corrections = probe.predict_corrections(model, features)
        np.testing.assert_array_equal(model['cutpoints']['mean_abs_change']['quartiles'], [.75, 1.5, 2.25])
        self.assertEqual(model['statistics']['OWN4']['eligible_cells'], 4)
        origin = model['tables']['ORIGIN'][0][0][3]
        for value in range(4):
            expected = (192*(value*4.)+128*origin)/(192+128)
            self.assertAlmostEqual(corrections['OWN4'][value, 0, 0], expected)
        self.assertEqual(corrections['OWN2'][0, 0, 0], corrections['OWN2'][1, 0, 0])
        self.assertNotEqual(corrections['OWN4'][0, 0, 0], corrections['OWN4'][1, 0, 0])

    def test_nonfinite_neighbor_falls_back_to_origin_without_becoming_zero(self):
        pred, truth, valid, features, ids = fixture()
        features['neighbor_change_disagreement'][::3] = np.nan
        features['neighbor_change_disagreement'][1::3] = np.inf
        truth[2::3] += 5.
        model = probe.fit_probe(pred, truth, valid, features, ids)
        correction = probe.predict_corrections(model, features)
        unavailable = ~np.isfinite(features['neighbor_change_disagreement'])
        for name in ('NEIGHBOR2', 'JOINT'):
            np.testing.assert_array_equal(correction[name].transpose(0, 2, 1)[unavailable],
                correction['ORIGIN'].transpose(0, 2, 1)[unavailable])
            self.assertEqual(model['stats'][name]['missing_feature_valid_count'], int(unavailable.sum())*2)
        features['neighbor_change_disagreement'][:] = np.nan
        absent_model = probe.fit_probe(pred, truth, valid, features, ids)
        absent = probe.predict_corrections(absent_model, features)
        self.assertIsNone(absent_model['cutpoints']['neighbor_change_disagreement']['median'])
        np.testing.assert_array_equal(absent['JOINT'], absent['ORIGIN'])

    def test_prediction_is_label_free_and_does_not_change_inputs(self):
        pred, truth, valid, features, ids = fixture()
        model = probe.fit_probe(pred, truth+2., valid, features, ids)
        self.assertEqual(list(inspect.signature(probe.predict_corrections).parameters),
            ['model', 'features', 'horizon'])
        before = {name: value.tobytes() for name, value in features.items()}
        with self.assertRaises(TypeError): probe.predict_corrections(model, features, y=truth)
        output = probe.predict_corrections(model, features)
        self.assertEqual(tuple(output), probe.ARMS)
        self.assertEqual(before, {name: value.tobytes() for name, value in features.items()})
        self.assertFalse(model['training']['validation_labels_used'])

    def test_json_round_trip_preserves_all_corrections_and_rejects_overwrite(self):
        pred, truth, valid, features, ids = fixture()
        model = probe.fit_probe(pred, truth+3., valid, features, ids)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'fit.json'
            probe.save_model(model, path)
            restored = probe.load_model(path)
            for name, value in probe.predict_corrections(model, features).items():
                np.testing.assert_array_equal(value, probe.predict_corrections(restored, features)[name])
            with self.assertRaises(ValueError): probe.save_model(model, path)

    def test_invalid_axes_masks_ids_protocol_and_support_are_rejected(self):
        pred, truth, valid, features, ids = fixture()
        with self.assertRaises(ValueError): probe.fit_probe(pred, truth, valid.astype(int), features, ids)
        with self.assertRaises(ValueError): probe.fit_probe(pred, truth, valid, features, np.zeros(len(ids), dtype=int))
        with self.assertRaises(ValueError): probe.fit_probe(pred[:128], truth[:128], valid[:128],
            {name: value[:128] for name, value in features.items()}, ids[:128])
        model = probe.fit_probe(pred, truth, valid, features, ids)
        with self.assertRaises(ValueError): probe.predict_corrections(model, features, horizon=1)
        changed = deepcopy(model); changed['protocol']['shrinkage'] = 0
        with self.assertRaises(ValueError): probe.predict_corrections(changed, features)
        changed = deepcopy(features); changed['mean_abs_change'][0, 0] = np.nan
        with self.assertRaises(ValueError): probe.predict_corrections(model, changed)


if __name__ == '__main__': unittest.main()
