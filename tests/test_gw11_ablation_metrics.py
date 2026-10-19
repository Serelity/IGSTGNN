"""Server-only synthetic tests; never run locally by the publishing workflow."""
import copy
import json
import unittest

import numpy as np

from experiments.state_gate_gw11.ablation_metrics import ARMS, analyze, render_report


def fixture(samples=2, nodes=2):
    y = np.full((samples, 12, nodes), 100., dtype=np.float32)
    archive = dict(y=y, valid=np.ones_like(y, dtype=np.bool_),
        node_ids=np.arange(nodes, dtype=np.int64)+500,
        road_labels=np.asarray(['road_'+str(n) for n in range(nodes)]),
        sample_ids=np.arange(samples, dtype=np.int64)+20,
        direct_support=np.zeros((samples, nodes), dtype=np.bool_),
        axis_order=np.asarray('sample,horizon,node'), registered_arms=np.asarray(ARMS),
        pred__FROZEN_SUM=y+3.)
    off = {}
    for arm in ARMS:
        off[arm] = y+5.
        archive['pred__'+arm] = off[arm].copy() if arm == 'SUM_CONTINUE' else y+4.
    return archive, off


def global_row(result, arm='STATE_GATE'):
    return next(row for row in result['decomposition'] if row['arm'] == arm and row['group'] == 'all')


class GW11AblationMetricsTests(unittest.TestCase):
    def test_mae_and_bias_decomposition_telescope_in_original_units(self):
        result = analyze(*fixture())
        row = global_row(result)
        self.assertEqual(row['frozen_mae'], 3.)
        self.assertEqual(row['source_only_mae'], 5.)
        self.assertEqual(row['complete_mae'], 4.)
        self.assertEqual(row['source_update_mae_delta'], 2.)
        self.assertEqual(row['adapter_presence_mae_delta'], -1.)
        self.assertEqual(row['total_mae_delta'], 1.)
        self.assertEqual(row['source_update_bias_delta'], 2.)
        self.assertEqual(row['adapter_presence_bias_delta'], -1.)
        self.assertEqual(row['total_bias_delta'], 1.)
        self.assertTrue(row['telescoping_checked'])
        self.assertEqual(result['metrics']['STATE_GATE__ON']['rmse'], 4.)
        self.assertAlmostEqual(result['metrics']['STATE_GATE__ON']['mape_fraction'], .04)

    def test_mae_and_bias_deltas_differ_when_errors_cross_zero(self):
        archive, off = fixture()
        archive['pred__FROZEN_SUM'][:] = 97.
        archive['pred__STATE_GATE'][:] = 99.
        row = global_row(analyze(archive, off))
        self.assertEqual(row['source_update_mae_delta'], 2.)
        self.assertEqual(row['adapter_presence_mae_delta'], -4.)
        self.assertEqual(row['total_mae_delta'], -2.)
        self.assertEqual(row['source_update_bias_delta'], 8.)
        self.assertEqual(row['adapter_presence_bias_delta'], -6.)
        self.assertEqual(row['total_bias_delta'], 2.)

    def test_mask_pools_points_instead_of_averaging_node_means(self):
        archive, off = fixture(samples=1)
        archive['valid'][0, 1:, 1] = False
        archive['pred__STATE_GATE'][:, :, 0] = 102.
        archive['pred__STATE_GATE'][:, :, 1] = 110.
        result = analyze(archive, off)
        row = global_row(result)
        self.assertEqual(row['valid_count'], 13)
        self.assertAlmostEqual(row['complete_mae'], 34./13.)
        self.assertNotAlmostEqual(row['complete_mae'], 6.)
        self.assertAlmostEqual(result['metrics']['STATE_GATE__ON']['mae'], 34./13.)

    def test_flow_bins_use_exact_half_open_truth_boundaries(self):
        archive, off = fixture(samples=1, nodes=1)
        values = np.asarray([1., 49., 50., 199., 200., 313., 314., 315., 49., 50., 200., 314.], dtype=np.float32)
        archive['y'][0, :, 0] = values
        result = analyze(archive, off)
        rows = {row['group']:row for row in result['by_flow'] if row['arm'] == 'STATE_GATE'}
        self.assertEqual({name:row['valid_count'] for name, row in rows.items()},
            {'<50':3, '50_200':3, '200_314':3, 'ge314':3})
        self.assertTrue(all(row['group_input'] == 'future_truth_evaluation_only' for row in rows.values()))
        self.assertEqual(sum(row['valid_count'] for row in rows.values()), 12)

    def test_constant_prediction_movement_has_zero_centered_sd(self):
        row = global_row(analyze(*fixture()))
        self.assertEqual(row['adapter_presence_effect_mean'], -1.)
        self.assertEqual(row['adapter_presence_effect_mean_absolute'], 1.)
        self.assertEqual(row['adapter_presence_effect_rms'], 1.)
        self.assertEqual(row['adapter_presence_effect_centered_std'], 0.)

    def test_heterogeneous_movement_can_cancel_mean_without_zero_effect(self):
        archive, off = fixture()
        archive['pred__STATE_GATE'][:, :6, :] = off['STATE_GATE'][:, :6, :]+2.
        archive['pred__STATE_GATE'][:, 6:, :] = off['STATE_GATE'][:, 6:, :]-2.
        row = global_row(analyze(archive, off))
        self.assertEqual(row['adapter_presence_effect_mean'], 0.)
        self.assertEqual(row['adapter_presence_effect_mean_absolute'], 2.)
        self.assertEqual(row['adapter_presence_effect_rms'], 2.)
        self.assertEqual(row['adapter_presence_effect_centered_std'], 2.)
        self.assertEqual(row['adapter_presence_bias_delta'], 0.)

    def test_sum_control_requires_exactly_equal_prediction_values(self):
        archive, off = fixture()
        off['SUM_CONTINUE'][0, 0, 0] = np.nextafter(off['SUM_CONTINUE'][0, 0, 0], np.float32(np.inf))
        with self.assertRaisesRegex(ValueError, 'SUM_CONTINUE ON/OFF'):
            analyze(archive, off)

    def test_sum_control_requires_identical_float32_dtype(self):
        archive, off = fixture()
        off['SUM_CONTINUE'] = off['SUM_CONTINUE'].astype(np.float64)
        with self.assertRaisesRegex(ValueError, 'float32'):
            analyze(archive, off)

    def test_empty_guard_groups_use_null_metrics_and_no_nan(self):
        result = analyze(*fixture())
        row = next(row for row in result['decomposition']
            if row['arm'] == 'STATE_GATE' and row['group'] == 'high_value_ge314')
        self.assertEqual(row['valid_count'], 0)
        self.assertIsNone(row['source_update_mae_delta'])
        self.assertIsNone(row['adapter_presence_bias_delta'])
        self.assertIsNone(row['adapter_presence_effect_rms'])
        self.assertFalse(row['telescoping_checked'])
        self.assertIsNone(result['metrics']['FROZEN_SUM']['groups']['high_value_ge314']['rmse'])
        json.dumps(result, allow_nan=False)

    def test_global_mask_must_be_boolean_and_nonempty(self):
        for mode in ('integer', 'empty'):
            with self.subTest(mode=mode):
                archive, off = fixture()
                archive['valid'] = archive['valid'].astype(np.int8) if mode == 'integer' else ~archive['valid']
                with self.assertRaisesRegex(ValueError, 'Boolean'):
                    analyze(archive, off)

    def test_unselected_points_do_not_change_metrics(self):
        archive, off = fixture(samples=1)
        archive['valid'][0, 0, 0] = False
        first = analyze(archive, off)
        archive['y'][0, 0, 0] = 0.
        archive['pred__STATE_GATE'][0, 0, 0] = 1e10
        off['STATE_GATE'][0, 0, 0] = -1e10
        second = analyze(archive, off)
        self.assertEqual(global_row(first), global_row(second))

    def test_finite_checks_cover_masked_predictions_too(self):
        archive, off = fixture()
        archive['valid'][0, 0, 0] = False
        off['STATE_GATE'][0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, 'Finite'):
            analyze(archive, off)

    def test_wrong_forecast_horizon_or_prediction_shape_is_rejected(self):
        archive, off = fixture()
        archive['y'] = archive['y'][:, :11]
        with self.assertRaisesRegex(ValueError, 'sample,12,node'):
            analyze(archive, off)
        archive, off = fixture()
        off['STATE_GATE'] = off['STATE_GATE'][:, :, :1]
        with self.assertRaisesRegex(ValueError, 'axes'):
            analyze(archive, off)

    def test_axes_keep_real_identifiers_and_reject_duplicates(self):
        archive, off = fixture()
        result = analyze(archive, off)
        node = next(row for row in result['by_node'] if row['arm'] == 'STATE_GATE')
        window = next(row for row in result['by_window'] if row['arm'] == 'STATE_GATE')
        self.assertEqual((node['node_id'], node['road_label']), (500, 'road_0'))
        self.assertEqual(window['sample_id'], 20)
        archive['node_ids'][:] = 500
        with self.assertRaisesRegex(ValueError, 'Unique integer'):
            analyze(archive, off)

    def test_support_road_and_explicit_axis_order_must_match(self):
        for name, value in (('direct_support', np.ones((2, 12, 2), dtype=np.bool_)),
                ('road_labels', np.asarray([1, 2])), ('axis_order', np.asarray('node,horizon,sample'))):
            with self.subTest(name=name):
                archive, off = fixture()
                archive[name] = value
                with self.assertRaises(ValueError):
                    analyze(archive, off)

    def test_exact_off_arm_set_is_required(self):
        for operation in ('missing', 'extra'):
            with self.subTest(operation=operation):
                archive, off = fixture()
                if operation == 'missing':
                    del off['STATE_GATE']
                else:
                    off['OTHER'] = off['STATE_GATE']
                with self.assertRaisesRegex(ValueError, 'exactly the four'):
                    analyze(archive, off)

    def test_all_nine_cases_fixed_groups_and_partition_counts_are_reported(self):
        result = analyze(*fixture())
        self.assertEqual(len(result['metrics']), 9)
        self.assertEqual(len(result['decomposition']), 16)
        self.assertEqual(len(result['by_horizon']), 48)
        self.assertEqual(len(result['by_node']), 8)
        self.assertEqual(len(result['by_window']), 8)
        for table in ('by_horizon', 'by_node', 'by_window', 'by_flow'):
            count = sum(row['valid_count'] for row in result[table] if row['arm'] == 'STATE_GATE')
            self.assertEqual(count, result['metrics']['FROZEN_SUM']['valid_count'])
            contributions = sum(row['total_mae_global_contribution'] or 0.
                for row in result[table] if row['arm'] == 'STATE_GATE')
            self.assertAlmostEqual(contributions, global_row(result)['total_mae_delta'])

    def test_arrays_are_not_mutated_and_interpretation_does_not_select_winner(self):
        archive, off = fixture()
        original_archive, original_off = copy.deepcopy(archive), copy.deepcopy(off)
        result = analyze(archive, off)
        for key in archive:
            self.assertTrue(np.array_equal(archive[key], original_archive[key]), key)
        for key in off:
            self.assertTrue(np.array_equal(off[key], original_off[key]), key)
        meaning = result['interpretation']
        self.assertFalse(meaning['winner_selection'])
        self.assertFalse(meaning['causal_training_claim'])
        self.assertFalse(meaning['screening_changed'])
        self.assertEqual(meaning['additional_training_updates'], 0)
        self.assertFalse(meaning['test_evaluated'])
        self.assertIn('coadapted', meaning['limitation'])
        self.assertIn('STATE_GATE', render_report(result))


if __name__ == '__main__':
    unittest.main()
