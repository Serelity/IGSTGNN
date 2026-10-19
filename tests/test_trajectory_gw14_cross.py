"""Server-only hand-checkable contracts for frozen GW14 cross diagnostics.

Authored and statically reviewed locally. Do not run local Python or tests.
"""
from copy import deepcopy
import json
import unittest
from unittest import mock

import numpy as np

from experiments.conditional_residual_gw10 import screen as old_metrics
from experiments.trajectory_check_gw14 import cross
from experiments.trajectory_diagnostic_gw13 import features as history_features


def fixture(samples=4, horizon=1, nodes=1):
    truth = np.full((samples, horizon, nodes), 10., dtype=np.float64)
    data = dict(y=truth, valid=np.ones(truth.shape, dtype=np.bool_),
        sample_ids=np.arange(samples, dtype=np.int64),
        features=dict(shape=np.zeros((samples, nodes), dtype=np.int8),
            turn_position=np.full((samples, nodes), -1, dtype=np.int8),
            last_flow=np.full((samples, nodes), 20.),
            mean_abs_change=np.ones((samples, nodes)),
            signed_direction=np.ones((samples, nodes))))
    predictions = {arm: truth.copy() for arm in cross.ARMS}
    return predictions, data


class CrossContributionsTests(unittest.TestCase):
    def test_hand_calculated_gain_help_harm_cancellation_and_contributions(self):
        predictions, data = fixture()
        # Individual OWN4->TRAJ error gains are [4, -1, -1, 0]. The helped
        # fraction is 1/4; gross help is 4/4=1, so count ratios cannot pass.
        predictions['OWN4'][:, 0, 0] = [14., 11., 11., 10.]
        predictions['TRAJECTORY16'][:, 0, 0] = [10., 12., 12., 10.]
        predictions['MAGNITUDE16'][:, 0, 0] = [12., 12., 12., 12.]
        predictions['DIRECTION16'][:, 0, 0] = [11., 11., 11., 11.]
        data['features']['shape'][:, 0] = [0, 1, 1, 3]
        result = cross.summarize(predictions, data, phase='validation')
        row = result['overall']
        self.assertEqual(row['n'], 4)
        self.assertEqual(row['OWN4_mae'], 1.5)
        self.assertEqual(row['TRAJECTORY16_mae'], 1.)
        self.assertEqual(row['trajectory_vs_OWN4_mae_gain'], .5)
        self.assertEqual(row['trajectory_vs_OWN4_gross_help'], 1.)
        self.assertEqual(row['trajectory_vs_OWN4_gross_harm'], .5)
        self.assertAlmostEqual(row['trajectory_vs_OWN4_cancellation'], 2. / 3.)
        self.assertEqual(row['trajectory_vs_MAGNITUDE16_mae_gain'], 1.)
        self.assertEqual(row['trajectory_vs_DIRECTION16_mae_gain'], 0.)
        by_shape = {cell['shape_code']: cell for cell in result['cross_rows']}
        self.assertEqual(by_shape[0]['trajectory_vs_OWN4_overall_contribution'], 1.)
        self.assertEqual(by_shape[1]['trajectory_vs_OWN4_overall_contribution'], -.5)
        self.assertEqual(by_shape[3]['trajectory_vs_OWN4_overall_contribution'], 0.)
        self.assertIsNone(by_shape[3]['trajectory_vs_OWN4_cancellation'])
        self.assertIsNone(by_shape[3]['trajectory_vs_OWN4_relative_gain'])
        for table in ('cross_rows', 'phase_lead_rows', 'block_rows'):
            self.assertEqual(sum(cell['n'] for cell in result[table]), 4)
            self.assertEqual(sum(cell['valid_point_fraction'] for cell in result[table]), 1.)
            for control in cross.CONTROLS:
                prefix = 'trajectory_vs_' + control + '_'
                self.assertAlmostEqual(sum(cell[prefix + 'overall_contribution'] for cell in result[table]),
                                       row[prefix + 'mae_gain'])
                for cell in result[table]:
                    self.assertAlmostEqual(cell[prefix + 'gross_help'] - cell[prefix + 'gross_harm'],
                                           cell[prefix + 'mae_gain'])
                    self.assertAlmostEqual(cell[prefix + 'mae_gain'] * cell['valid_point_fraction'],
                                           cell[prefix + 'overall_contribution'])

    def test_every_arm_uses_same_valid_mask_and_nonzero_mape_denominator(self):
        predictions, data = fixture(samples=4)
        data['y'][:, 0, 0] = [10., 0., -5., 10.]
        data['valid'][:, 0, 0] = [True, True, True, False]
        for arm in cross.ARMS:
            predictions[arm][:, 0, 0] = [12., 3., -4., 1.e10]
        row = cross.summarize(predictions, data, phase='audit')['overall']
        self.assertEqual(row['n'], 3)
        self.assertEqual(row['mape_nonzero_truth_count'], 2)
        for arm in cross.ARMS:
            self.assertEqual(row[arm + '_mae'], 2.)
            self.assertEqual(row[arm + '_signed_bias'], 2.)
            self.assertAlmostEqual(row[arm + '_rmse'], np.sqrt(14. / 3.))
            self.assertAlmostEqual(row[arm + '_mape_fraction'], .2)
            old = old_metrics.metrics(predictions[arm], data['y'], data['valid'])
            for metric in ('mae', 'signed_bias', 'rmse', 'mape_fraction'):
                self.assertAlmostEqual(row[arm + '_' + metric], old[metric])

    def test_all_zero_truth_has_null_mape_but_valid_mae_and_gain(self):
        predictions, data = fixture(samples=2)
        data['y'].fill(0.)
        for arm in cross.ARMS:
            predictions[arm].fill(2.)
        predictions['TRAJECTORY16'].fill(1.)
        row = cross.summarize(predictions, data, phase='validation')['overall']
        self.assertEqual(row['n'], 2)
        self.assertEqual(row['mape_nonzero_truth_count'], 0)
        self.assertIsNone(row['TRAJECTORY16_mape_fraction'])
        self.assertEqual(row['trajectory_vs_OWN4_mae_gain'], 1.)

    def test_small_paired_vectors_reproduce_inherited_statistics(self):
        predictions, data = fixture(samples=4, horizon=2, nodes=2)
        truth = data['y']
        predictions['OWN4'] = truth + np.arange(16).reshape(truth.shape) / 4.
        predictions['TRAJECTORY16'] = truth + np.asarray([1., -1., 0., 2.] * 4).reshape(truth.shape)
        data['valid'][0, 1, 0] = False
        row = cross.summarize(predictions, data, phase='audit')['overall']
        expected = old_metrics.paired(predictions['OWN4'], predictions['TRAJECTORY16'], truth, data['valid'])
        for metric in ('mae_gain', 'relative_gain', 'gross_help', 'gross_harm', 'cancellation'):
            self.assertAlmostEqual(row['trajectory_vs_OWN4_' + metric], expected[metric])


class CrossPartitionsTests(unittest.TestCase):
    def test_complete_shape_turn_flow_cartesian_product_and_lead_partition(self):
        predictions, data = fixture(samples=128, horizon=2)
        index = np.arange(128)
        data['features']['shape'][:, 0] = index // 32
        data['features']['turn_position'][:, 0] = np.asarray([-1, 2, 5, 7])[(index // 8) % 4]
        data['features']['last_flow'][:, 0] = np.asarray([-1., 0., 5., 20., 50., 100., 200., 314.])[index % 8]
        predictions['OWN4'] += 2.
        result = cross.summarize(predictions, data, phase='validation')
        self.assertEqual(result['overall']['cross_total_combinations'], 128)
        self.assertEqual(result['overall']['cross_observed_combinations'], 128)
        self.assertEqual(result['overall']['phase_lead_total_combinations'], 32)
        self.assertEqual(result['overall']['phase_lead_observed_combinations'], 32)
        self.assertEqual({(row['shape_code'], row['turn_code'], row['flow_bin'])
                          for row in result['cross_rows']},
                         {(shape, turn, flow) for shape in range(4) for turn in range(4) for flow in range(8)})
        self.assertEqual({(row['shape_code'], row['turn_code'], row['lead'])
                          for row in result['phase_lead_rows']},
                         {(shape, turn, lead) for shape in range(4) for turn in range(4) for lead in (1, 2)})
        self.assertTrue(all(row['n'] == 2 for row in result['cross_rows']))
        self.assertTrue(all(row['n'] == 8 for row in result['phase_lead_rows']))
        self.assertTrue(all(row['trajectory_vs_OWN4_overall_contribution'] == 1. / 64.
                            for row in result['cross_rows']))
        self.assertTrue(all(row['trajectory_vs_OWN4_overall_contribution'] == 1. / 16.
                            for row in result['phase_lead_rows']))

    def test_all_turn_positions_map_once_and_no_future_truth_can_change_membership(self):
        predictions, data = fixture(samples=9, horizon=2)
        data['features']['turn_position'][:, 0] = [-1, 2, 3, 4, 5, 6, 7, 8, 9]
        before = cross.summarize(predictions, data, phase='audit')
        changed = deepcopy(data)
        changed['y'] = changed['y'] * -100. + np.arange(18).reshape(9, 2, 1)
        # Changing all future labels changes scores while origins/routes stay fixed.
        after = cross.summarize(predictions, changed, phase='audit')
        for table, dimensions in (('cross_rows', ('shape_code', 'turn_code', 'flow_bin')),
                                  ('phase_lead_rows', ('shape_code', 'turn_code', 'lead'))):
            keys_before = [tuple(row[key] for key in dimensions) + (row['n'],) for row in before[table]]
            keys_after = [tuple(row[key] for key in dimensions) + (row['n'],) for row in after[table]]
            self.assertEqual(keys_before, keys_after)
        self.assertEqual([row['n'] for row in before['cross_rows']], [2, 6, 4, 6])
        self.assertNotEqual(before['overall']['OWN4_mae'], after['overall']['OWN4_mae'])

    def test_blocks_preserve_score_order_and_use_positions_not_sample_id_division(self):
        predictions, data = fixture(samples=260)
        data['sample_ids'] = 1954 + 2 * np.arange(260, dtype=np.int64)
        predictions['OWN4'][:128] += 4.
        predictions['TRAJECTORY16'][128:256] += 2.
        predictions['OWN4'][256:] += 1.
        result = cross.summarize(predictions, data, phase='audit')
        rows = result['block_rows']
        self.assertEqual([row['sample_count'] for row in rows], [128, 128, 4])
        self.assertEqual([row['n'] for row in rows], [128, 128, 4])
        self.assertEqual([row['score_position_start'] for row in rows], [0, 128, 256])
        self.assertEqual([row['sample_id_first'] for row in rows], [1954, 2210, 2466])
        self.assertEqual([row['sample_id_last'] for row in rows], [2208, 2464, 2472])
        self.assertEqual([row['trajectory_vs_OWN4_mae_gain'] for row in rows], [4., -2., 1.])
        self.assertAlmostEqual(sum(row['trajectory_vs_OWN4_overall_contribution'] for row in rows),
                               260. / 260.)
        self.assertFalse(result['overall']['blocks_are_independent_calendar_intervals'])

    def test_masked_and_empty_cells_are_not_reported_as_zero_gain(self):
        predictions, data = fixture(samples=2)
        data['features']['shape'][:, 0] = [0, 3]
        data['valid'][1] = False
        result = cross.summarize(predictions, data, phase='audit')
        self.assertEqual(len(result['cross_rows']), 1)
        self.assertEqual(result['cross_rows'][0]['shape_code'], 0)
        self.assertEqual(result['overall']['cross_observed_combinations'], 1)
        data['valid'].fill(False)
        empty = cross.summarize(predictions, data, phase='audit')
        self.assertEqual(empty['cross_rows'], [])
        self.assertEqual(empty['phase_lead_rows'], [])
        self.assertEqual(empty['overall']['n'], 0)
        self.assertIsNone(empty['overall']['trajectory_vs_OWN4_mae_gain'])
        self.assertIsNone(empty['block_rows'][0]['trajectory_vs_OWN4_overall_contribution'])
        json.dumps(empty, allow_nan=False)

    def test_none_and_nondecreasing_shape_do_not_claim_strict_monotonicity(self):
        predictions, data = fixture(samples=1, nodes=2)
        # A raw dip disappears into zero slopes after smoothing. Both block
        # trends are nonnegative, despite the first history not being monotone.
        # The second history is exactly constant. Both belong to "none".
        raw = np.asarray([[0, 3, 0, 3, 3, 3, 3, 3, 4, 4, 4, 4], [20.] * 12],
                         dtype=np.float64).T[None, :, :]
        cached = history_features.features_from_history(raw, 0., 1.)
        self.assertFalse(np.all(np.diff(raw[0, :, 0]) >= 0.))
        self.assertFalse(np.all(np.diff(raw[0, :, 0]) <= 0.))
        np.testing.assert_array_equal(cached['turn_position'], [[-1, -1]])
        np.testing.assert_array_equal(cached['shape'], [[3, 3]])
        data['features'] = cached
        result = cross.summarize(predictions, data, phase='validation')
        self.assertEqual(result['overall']['exact_constant_history_fraction'], .5)
        self.assertEqual(result['overall']['net_direction_zero_fraction'], .5)
        self.assertFalse(result['overall']['shape_is_strict_monotonicity'])
        constant = next(row for row in result['cross_rows'] if row['flow_bin'] == 3)
        self.assertEqual(constant['exact_constant_history_fraction'], 1.)
        self.assertTrue(all(row['turn'] == 'none' for row in result['cross_rows']))

    def test_exact_constant_does_not_sweep_small_positive_amplitude_into_plateau(self):
        predictions, data = fixture(samples=3)
        data['features']['mean_abs_change'][:, 0] = [0., 1.e-20, 1.]
        data['features']['signed_direction'][:, 0] = [0., 0., 1.]
        row = cross.summarize(predictions, data, phase='validation')['overall']
        self.assertAlmostEqual(row['exact_constant_history_fraction'], 1. / 3.)
        self.assertAlmostEqual(row['net_direction_zero_fraction'], 2. / 3.)


class CrossInputContractsTests(unittest.TestCase):
    def test_rejects_out_of_contract_turn_indices_instead_of_folding_into_none(self):
        predictions, data = fixture(samples=1)
        for turn in (-2, 0, 1, 10, 127):
            data['features']['turn_position'] = np.asarray([[turn]], dtype=np.int64)
            with self.subTest(turn=turn), self.assertRaisesRegex(ValueError, 'turn_position'):
                cross.summarize(predictions, data, phase='audit')
        for turn in (-1., 2.5, np.nan):
            data['features']['turn_position'] = np.asarray([[turn]])
            with self.subTest(turn=turn), self.assertRaises(ValueError):
                cross.summarize(predictions, data, phase='audit')

    def test_shape_must_be_integer_code_not_arbitrary_nonnegative_value(self):
        predictions, data = fixture(samples=1)
        for value in (np.asarray([[4]]), np.asarray([[-1]]), np.asarray([[1.5]]),
                      np.asarray([[True]])):
            data['features']['shape'] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                cross.summarize(predictions, data, phase='audit')

    def test_requires_exact_four_arms_and_finite_matching_cubes(self):
        predictions, data = fixture(samples=1)
        for mutated in ({key: value for key, value in predictions.items() if key != 'OWN4'},
                        dict(predictions, SUM=data['y'].copy()),
                        dict(predictions, OWN4=np.ones((1, 1, 2))),
                        dict(predictions, OWN4=np.full((1, 1, 1), np.inf))):
            with self.subTest(keys=list(mutated)), self.assertRaises(ValueError):
                cross.summarize(mutated, data, phase='audit')

    def test_rejects_wrong_mask_axes_invalid_features_and_duplicate_sample_ids(self):
        predictions, data = fixture(samples=2)
        for field, replacement in (('valid', np.ones((2, 1, 1), dtype=np.int8)),
                                    ('valid', np.ones((2, 1), dtype=np.bool_)),
                                    ('sample_ids', np.asarray([4, 4])),
                                    ('sample_ids', np.asarray([-1, 0])),
                                    ('sample_ids', np.asarray([0., 1.]))):
            modified = deepcopy(data)
            modified[field] = replacement
            with self.subTest(field=field), self.assertRaises(ValueError):
                cross.summarize(predictions, modified, phase='audit')
        for name, value in (('last_flow', np.nan), ('mean_abs_change', -1.),
                            ('signed_direction', 2.)):
            modified = deepcopy(data)
            modified['features'][name][0, 0] = value
            with self.subTest(name=name), self.assertRaises(ValueError):
                cross.summarize(predictions, modified, phase='audit')

    def test_no_model_fit_prediction_reroute_or_input_mutation(self):
        predictions, data = fixture(samples=3, horizon=2)
        before_predictions, before_data = deepcopy(predictions), deepcopy(data)
        from experiments.trajectory_diagnostic_gw13 import model
        with mock.patch.object(model, 'fit_model', side_effect=AssertionError('Unexpected fit')), \
             mock.patch.object(model, 'predictions', side_effect=AssertionError('Unexpected prediction route')):
            result = cross.summarize(predictions, data, phase='audit')
        for name in cross.ARMS:
            np.testing.assert_array_equal(predictions[name], before_predictions[name])
        for name in ('y', 'valid', 'sample_ids'):
            np.testing.assert_array_equal(data[name], before_data[name])
        for name in data['features']:
            np.testing.assert_array_equal(data['features'][name], before_data['features'][name])
        self.assertFalse(result['overall']['grouping_uses_future_truth'])
        self.assertFalse(result['overall']['predictions_rerouted'])
        self.assertEqual(result['overall']['model_fits'], 0)
        json.dumps(result, allow_nan=False)


if __name__ == '__main__':
    unittest.main()
