from datetime import datetime
import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from src.utils.chronological import ChronologicalDataset, encode_window, masked_flow_mae, flow_metrics
from experiments.chronological.prepare_context import spatial_features


class ChronologicalTests(unittest.TestCase):
    def test_loader_rejects_reordered_scaler_and_stale_forecast_clock(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            row = {'sample_index': '42', 'split': 'train', 'source_version': '8',
                   'report_time': '2023-02-02T04:06:00', 't0': '2023-02-02T04:10:00',
                   'x_start': '2023-02-02T03:05:00'}
            with (base / 'train_manifest.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)
            np.save(base / 'station_ids.npy', [10, 20])
            np.save(base / 'train_flow.npy', np.ones((1, 26, 2), dtype=np.float32))
            context = dict(sample_indices=[42], station_ids=[10, 20], distances=np.zeros((1, 2, 3)),
                           report_age_minutes=[4.], forecast_tod=[50], forecast_dow=[4])
            np.savez(base / 'train_context.npz', **context)
            scaler = dict(source_version=8, fit_scope='train_X_0:12_unique_station_nominal_slot_finite_nonnegative',
                          station_ids=[10, 20], mean=1, std=1, node_fill_mean=[1, 1], fitted_sample_indices=[42])
            (base / 'scaler.json').write_text(json.dumps(scaler))
            self.assertEqual(len(ChronologicalDataset(base, 'train')), 1)
            scaler['station_ids'] = [20, 10]
            (base / 'scaler.json').write_text(json.dumps(scaler))
            with self.assertRaisesRegex(ValueError, 'Scaler.*station order'):
                ChronologicalDataset(base, 'train')
            scaler['station_ids'] = [10, 20]
            (base / 'scaler.json').write_text(json.dumps(scaler))
            context['forecast_tod'] = [48]
            np.savez(base / 'train_context.npz', **context)
            with self.assertRaisesRegex(ValueError, 'forecast clock'):
                ChronologicalDataset(base, 'train')

    def test_gap_missing_input_and_true_zero_target(self):
        raw = np.ones((26, 2), dtype=np.float32) * 8
        raw[0, 0] = np.nan
        raw[12:14] = 9999  # Deliberately unavailable gap, never exposed as X/Y.
        raw[14, 0], raw[14, 1], raw[15, 1] = 0, np.nan, -1
        scaler = {'mean': 5, 'std': 2, 'node_fill_mean': [4, 6]}
        x, y, xm, ym = encode_window(raw, datetime(2023, 8, 1, 23, 30), scaler)
        self.assertEqual(x[0, 0, 0], -.5)
        self.assertFalse(xm[0, 0, 0])
        self.assertTrue(ym[0, 0, 0])
        self.assertFalse(ym[0, 1, 0])
        self.assertFalse(ym[1, 1, 0])
        self.assertEqual(y[0, 0, 0], 0)
        self.assertLess(x[..., 0].max(), 2)
        self.assertEqual(x[6, 0, 1], 0)  # Midnight calendar rollover.
        self.assertAlmostEqual(float(x[6, 0, 2]), 3 / 7)

    def test_explicit_mask_excludes_nan_but_counts_real_zero_and_gradient(self):
        pred = torch.tensor([2., 20., 6.], requires_grad=True)
        target = torch.tensor([0., float('nan'), 4.])
        mask = torch.tensor([True, False, True])
        loss = masked_flow_mae(pred, target, mask)
        self.assertEqual(float(loss), 2)
        loss.backward()
        torch.testing.assert_close(pred.grad, torch.tensor([.5, 0., .5]))
        with self.assertRaises(ValueError):
            masked_flow_mae(pred, target, torch.zeros(3, dtype=torch.bool))

    def test_macro_and_pooled_differ_with_unequal_horizon_counts(self):
        target = np.ones((1, 2, 2, 1))
        prediction = np.array([[[[2.], [2.]], [[10.], [999.]]]])
        valid = np.array([[[[True], [True]], [[True], [False]]]])
        result = flow_metrics(prediction, target, valid)
        self.assertEqual(result['mae_macro'], 5.)
        self.assertAlmostEqual(result['mae_pooled'], 11 / 3)
        self.assertEqual(result['rmse_macro'], 5.)
        self.assertAlmostEqual(result['rmse_pooled'], np.sqrt(83 / 3))

    def test_mape_excludes_zero_without_excluding_it_from_mae(self):
        target = np.array([[[[0.], [2.]]]])
        prediction = np.array([[[[4.], [3.]]]])
        result = flow_metrics(prediction, target, np.ones_like(target, dtype=bool))
        self.assertEqual(result['valid_count_per_horizon'], [2])
        self.assertEqual(result['positive_target_count_per_horizon'], [1])
        self.assertEqual(result['mae_macro'], 2.5)
        self.assertEqual(result['mape_macro'], .5)

    def test_spatial_inputs_require_same_road_and_direction_and_ignore_description(self):
        sensors = [{'Fwy': road, 'Direction': direction, 'Abs PM': str(pm)}
                   for road, direction, pm in [('I80-E', 'E', 9), ('I80-W', 'W', 9),
                                                ('I680-E', 'E', 9), ('I80-E', 'E', 21)]]
        event = {'postmile': 10, 'freeway': 80, 'direction': 'E', 'description': 'not an input'}
        actual = spatial_features(event, sensors)
        self.assertEqual(actual[:, 0].tolist(), [0, 0, 0, 0])
        self.assertAlmostEqual(float(actual[0, 1]), np.exp(-.5))
        self.assertEqual(float(actual[0, 2]), 1)
        np.testing.assert_array_equal(actual[1:], 0)
        event['description'] = 'changed after the report'
        np.testing.assert_array_equal(spatial_features(event, sensors), actual)


if __name__ == '__main__':
    unittest.main()
