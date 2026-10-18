"""Server-only numeric contracts for CPU archive diagnostics; no model imports."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.graph_memory_gw6 import diagnose_drift as drift


class PredictionDriftTests(unittest.TestCase):
    def test_constant_translation_is_removed_without_changing_reference_error(self):
        y = np.array([10., 20., 30., 40.])
        best = y+.5
        result = drift.pair_statistics(y, best, best-3.)
        self.assertAlmostEqual(result['shift_mean'], -3.)
        self.assertAlmostEqual(result['shift_std'], 0.)
        self.assertAlmostEqual(result['shift_mse'], 9.)
        self.assertAlmostEqual(result['global_shift_energy_fraction'], 1.)
        self.assertAlmostEqual(result['best_mae'], .5)
        self.assertAlmostEqual(result['last_mae'], 2.5)
        self.assertAlmostEqual(result['centered_last_mae'], .5)
        self.assertAlmostEqual(result['centered_removed_fraction'], 1.)
        self.assertAlmostEqual(result['last_bias']-result['best_bias'], result['shift_mean'])

    def test_zero_mean_changes_have_no_global_shift_energy(self):
        y = np.full(4, 10.)
        result = drift.pair_statistics(y, y, np.array([8., 8., 12., 12.]))
        self.assertAlmostEqual(result['shift_mean'], 0.)
        self.assertAlmostEqual(result['shift_variance'], 4.)
        self.assertAlmostEqual(result['shift_mse'], 4.)
        self.assertAlmostEqual(result['global_shift_energy_fraction'], 0.)
        self.assertAlmostEqual(result['last_mae'], 2.)
        self.assertAlmostEqual(result['centered_last_mae'], 2.)

    def test_prediction_energy_fraction_is_not_mae_recovery_fraction(self):
        y = np.full(2, 10.)
        result = drift.pair_statistics(y, y, np.array([9., 5.]))
        self.assertAlmostEqual(result['shift_mean'], -3.)
        self.assertAlmostEqual(result['shift_variance'], 4.)
        self.assertAlmostEqual(result['shift_mse'], 13.)
        self.assertAlmostEqual(result['global_shift_energy_fraction'], 9./13.)
        self.assertAlmostEqual(result['last_mae'], 3.)
        self.assertAlmostEqual(result['centered_last_mae'], 2.)
        self.assertAlmostEqual(result['centered_removed_fraction'], 1./3.)

    def test_localized_node_shift_uses_one_global_centering_constant(self):
        labels = np.tile(np.arange(4), 6)
        y = np.arange(1., 25.)
        best = y.copy()
        last = y+4.*(labels == 0)
        result = drift.pair_statistics(y, best, last)
        self.assertAlmostEqual(result['shift_mean'], 1.)
        self.assertAlmostEqual(result['shift_variance'], 3.)
        self.assertAlmostEqual(result['global_shift_energy_fraction'], .25)
        self.assertAlmostEqual(result['centered_last_mae'], 1.5)
        self.assertAlmostEqual(result['centered_removed_fraction'], -.5)
        rows = drift.group_statistics(labels, 4, y, best, last, result['shift_mean'])
        self.assertEqual([row['count'] for row in rows], [6, 6, 6, 6])
        self.assertAlmostEqual(rows[0]['shift_mean'], 4.)
        self.assertAlmostEqual(rows[0]['centered_last_mae'], 3.)
        self.assertAlmostEqual(rows[1]['centered_last_mae'], 1.)
        self.assertAlmostEqual(rows[0]['net_mae_contribution'], 1.)
        self.assertAlmostEqual(rows[0]['gross_mae_worsening_contribution'], 1.)
        self.assertAlmostEqual(sum(row['net_mae_contribution'] for row in rows), 1.)

    def test_cancellation_keeps_both_positive_and_negative_contributions(self):
        y = np.full(2, 10.)
        best = np.array([9., 10.])
        last = np.array([10., 11.])
        result = drift.pair_statistics(y, best, last)
        self.assertAlmostEqual(result['mae_deterioration'], 0.)
        self.assertAlmostEqual(result['gross_mae_worsening'], .5)
        self.assertAlmostEqual(result['gross_mae_improvement'], .5)
        self.assertIsNone(result['centered_removed_fraction'])
        rows = drift.group_statistics(np.array([0, 1]), 2, y, best, last, result['shift_mean'])
        self.assertAlmostEqual(rows[0]['net_mae_contribution'], -.5)
        self.assertAlmostEqual(rows[1]['net_mae_contribution'], .5)

    def test_unchanged_predictions_and_empty_groups_serialize_without_nan(self):
        y = np.array([1., 2.])
        result = drift.pair_statistics(y, y+.1, y+.1)
        self.assertEqual(result['shift_mse'], 0.)
        self.assertIsNone(result['global_shift_energy_fraction'])
        self.assertIsNone(result['centered_removed_fraction'])
        rows = drift.group_statistics(np.array([0, 0]), 3, y, y, y, 0.)
        self.assertEqual(rows[1]['count'], 0)
        self.assertIsNone(rows[1]['last_mae'])
        self.assertEqual(rows[1]['net_mae_contribution'], 0.)
        json.dumps({'overall': result, 'groups': rows}, allow_nan=False)

    def test_repeated_quantiles_and_constant_targets_cover_every_point(self):
        constant = np.full(8, 10.)
        labels, bounds = drift.target_bins(constant)
        np.testing.assert_array_equal(labels, np.zeros(8, dtype=np.int64))
        np.testing.assert_array_equal(bounds, np.array([10., 10.]))
        y = np.array([1.]*8+[2.]*2+[3.]*2)
        labels, bounds = drift.target_bins(y)
        self.assertEqual(labels.shape, y.shape)
        self.assertTrue(bool((np.diff(bounds) > 0).all()))
        self.assertTrue(bool((labels >= 0).all()))
        self.assertTrue(bool((labels < len(bounds)-1).all()))
        self.assertEqual(labels[-1], len(bounds)-2)
        self.assertEqual(int(np.bincount(labels).sum()), len(y))

    def test_invalid_numeric_inputs_are_rejected(self):
        y = np.array([1., 2.])
        with self.assertRaises(ValueError):
            drift.pair_statistics(y, y, np.array([1., np.nan]))
        with self.assertRaises(ValueError):
            drift.pair_statistics(y, y[:1], y)
        with self.assertRaises(ValueError):
            drift.pair_statistics(np.array([0., 2.]), y, y)
        with self.assertRaises(ValueError):
            drift.pair_statistics(np.array([]), np.array([]), np.array([]))

    def test_shared_axis_values_and_dtypes_must_match(self):
        axes = dict(y=np.ones((2, 3, 4)), valid=np.ones((2, 3, 4), dtype=bool),
            sample_ids=np.array([101, 102]), node_ids=np.array([11, 12, 13, 14]),
            road_labels=np.array(['r1', 'r1', 'r2', 'r2']),
            direct_support=np.zeros((2, 4), dtype=bool))
        duplicate = {key: value.copy() for key, value in axes.items()}
        drift.require_same_axes(axes, duplicate)
        duplicate['node_ids'][0] = 999
        with self.assertRaises(ValueError):
            drift.require_same_axes(axes, duplicate)
        duplicate = {key: value.copy() for key, value in axes.items()}
        duplicate['valid'] = duplicate['valid'].astype(np.uint8)
        with self.assertRaises(ValueError):
            drift.require_same_axes(axes, duplicate)

    def test_node_concentration_uses_valid_nodes_and_handles_zero_changes(self):
        y = np.full(3, 10.)
        labels = np.arange(3)
        last = np.array([14., 10., 10.])
        rows = drift.group_statistics(labels, 4, y, y, last, 4./3.)
        for row in rows:
            row['node_id'] = 100+row['group']
        concentration = drift.node_concentration(rows)
        self.assertEqual(concentration['valid_nodes'], 3)
        self.assertEqual(concentration['top_node_count'], 1)
        self.assertAlmostEqual(concentration['top10pct_nodes_by_energy_share'], 1.)
        self.assertAlmostEqual(concentration['top10pct_nodes_by_energy_point_fraction'], 1./3.)
        rows = drift.group_statistics(labels, 4, y, y, y, 0.)
        for row in rows:
            row['node_id'] = 100+row['group']
        concentration = drift.node_concentration(rows)
        self.assertIsNone(concentration['top10pct_nodes_by_energy_share'])
        self.assertIsNone(concentration['top10pct_nodes_by_worsening_share'])
        json.dumps(concentration, allow_nan=False)

    def test_archive_mask_cannot_silently_drop_valid_targets(self):
        y = np.ones((2, 12, 2), dtype=np.float32)
        y[0, 0, 0] = 0.
        payload = dict(schema=np.array(drift.ARCHIVE_SCHEMA),
            axis_order=np.array('sample,horizon,node'), stage=np.array('joint'),
            registered_arms=np.asarray(drift.ARMS), y=y, valid=y != 0.,
            sample_ids=np.array([1, 2]), node_ids=np.array([11, 12]),
            road_labels=np.array(['r1', 'r2']), direct_support=np.zeros((2, 2), dtype=bool))
        axes = drift.load_axes(payload, {'arms': list(drift.ARMS)})
        self.assertEqual(int(axes['valid'].sum()), 47)
        payload['valid'] = payload['valid'].copy()
        payload['valid'][1, 1, 1] = False
        with self.assertRaises(ValueError):
            drift.load_axes(payload, {'arms': list(drift.ARMS)})

    def test_old_five_epoch_evaluation_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            metadata = dict(schema=drift.PRODUCER_SCHEMA, status='GW6_PREDICTION_PASS',
                engineering_only=False, stage='joint', common_completed_new_epochs=5,
                checkpoint_selection='best')
            (directory/'metadata.json').write_text(json.dumps(metadata), encoding='utf-8')
            (directory/'analysis.json').write_text('{}', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'budget'):
                drift.verify_eval(directory, 'best', {}, {}, 20)


if __name__ == '__main__':
    unittest.main()
