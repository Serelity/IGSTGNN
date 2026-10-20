"""Small numerical contracts for cached-prediction diagnosis; no model imports."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / 'diagnose_igstgnn_local_problem.py'
SPEC = importlib.util.spec_from_file_location('igstgnn_problem_diagnosis_under_test', SCRIPT)
DIAGNOSIS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DIAGNOSIS)


def changing_target():
    common = np.asarray([20., 22., 25.])[:, None]
    activity = np.asarray([1., 2., 4.])[:, None]
    contrast = np.asarray([-1., 0., 0., 1.])[None, :]
    return common + activity * contrast


class PredictionDiagnosisTests(unittest.TestCase):
    def test_perfect_prediction_has_no_error_and_does_not_match_pattern(self):
        target = changing_target()
        result, _ = DIAGNOSIS.decompose(target.copy(), target)
        for key in ('node_mae', 'region_mae', 'local_mae', 'node_mse',
                    'region_mse', 'local_mse', 'mse_identity_error'):
            self.assertEqual(result[key], 0.)
        self.assertEqual(result['region_change_skill'], 1.)
        self.assertEqual(result['local_change_skill'], 1.)
        self.assertFalse(result['pattern_observed'])

    def test_uniform_bias_is_entirely_regional(self):
        target = changing_target()
        result, _ = DIAGNOSIS.decompose(target + 3.5, target)
        self.assertEqual(result['region_mae'], 3.5)
        self.assertEqual(result['node_mae'], 3.5)
        self.assertEqual(result['local_mae'], 0.)
        self.assertEqual(result['region_mse'], 3.5 ** 2)
        self.assertEqual(result['local_mse'], 0.)
        # Accurate changes do not imply an accurate level: this bias persists.
        self.assertEqual(result['region_change_skill'], 1.)
        self.assertFalse(result['pattern_observed'])

    def test_reversed_local_changes_have_minus_three_skill(self):
        target = changing_target()
        common = target.mean(axis=1, keepdims=True)
        prediction = common - (target - common)
        result, _ = DIAGNOSIS.decompose(prediction, target)
        self.assertEqual(result['region_mae'], 0.)
        self.assertEqual(result['region_change_skill'], 1.)
        self.assertAlmostEqual(result['local_change_skill'], -3., places=12)
        self.assertGreater(result['local_mae'], result['region_mae'])
        self.assertTrue(result['pattern_observed'])

    def test_no_true_temporal_change_has_undefined_skills(self):
        target = np.tile([10., 12., 14., 16.], (3, 1))
        prediction = np.tile([11., 12., 13., 14.], (3, 1))
        result, _ = DIAGNOSIS.decompose(prediction, target)
        self.assertIsNone(result['region_change_skill'])
        self.assertIsNone(result['local_change_skill'])
        self.assertFalse(result['pattern_observed'])

    def test_squared_error_identity_does_not_make_maes_additive(self):
        target = np.asarray([[10., 10.], [12., 12.]])
        prediction = target + np.asarray([[3., -1.], [3., -1.]])
        result, _ = DIAGNOSIS.decompose(prediction, target)
        self.assertEqual(result['node_mse'], 5.)
        self.assertEqual(result['region_mse'], 1.)
        self.assertEqual(result['local_mse'], 4.)
        self.assertEqual(result['mse_identity_error'], 0.)
        self.assertEqual(result['node_mae'], 2.)
        self.assertEqual(result['region_mae'] + result['local_mae'], 3.)
        self.assertNotEqual(result['node_mae'], result['region_mae'] + result['local_mae'])

    def test_fixed_partition_sorts_finite_postmiles_first_and_breaks_ties_by_id(self):
        rows = [
            {'Fwy': '80', 'Direction': 'E', 'Abs PM': '2', 'station_id': '40'},
            {'Fwy': '80', 'Direction': 'E', 'Abs PM': '', 'station_id': '30'},
            {'Fwy': '80', 'Direction': 'E', 'Abs PM': '1', 'station_id': '20'},
            {'Fwy': '80', 'Direction': 'E', 'Abs PM': '1', 'station_id': '10'},
            {'Fwy': '80', 'Direction': 'E', 'Abs PM': 'NaN', 'station_id': '5'},
            {'Fwy': '80', 'Direction': 'W', 'Abs PM': '.5', 'station_id': '90'},
        ]
        regions = DIAGNOSIS.partition(rows)
        self.assertEqual(regions, [
            {'name': '80/E:0', 'members': [3, 2, 0, 4, 1]},
            {'name': '80/W:0', 'members': [5]},
        ])
        large_rows = [dict(Fwy='80', Direction='E', station_id=str(100+i),
                           **{'Abs PM': str(65-i)}) for i in range(65)]
        chunks = DIAGNOSIS.partition(large_rows)
        self.assertEqual([len(r['members']) for r in chunks], [22, 22, 21])
        self.assertEqual([i for r in chunks for i in r['members']], list(reversed(range(65))))


if __name__ == '__main__':
    unittest.main()
