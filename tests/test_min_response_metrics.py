"""Independent contracts for the complete MIN response branch objective.

History-defined safeguards must not use future target-flow buckets as routing
inputs. The future targets are supervision only, and the frozen OFF branch is
always detached from the objective.
"""
import json
import unittest

import numpy as np
import torch

from experiments.chronological.min_response_metrics import (
    paired_training_loss, summarize_response_pair,
)


class ResponseTrainingLossTests(unittest.TestCase):
    def fixture(self, *, mean=0., std=1.):
        target = torch.tensor([200., 10.], dtype=torch.float64).reshape(1, 1, 2, 1).repeat(1, 12, 1, 1)
        off = target+1.
        on = target+torch.tensor([3., 0.], dtype=torch.float64).reshape(1, 1, 2, 1)
        history = torch.zeros(1, 12, 2, 3, dtype=torch.float64)
        history[..., 0] = torch.tensor([20., 200.], dtype=torch.float64)
        on, off, target = [(value-mean)/std for value in (on, off, target)]
        history[..., 0] = (history[..., 0]-mean)/std
        return on, off, target, history

    def test_original_scale_mae_plus_history_group_excess(self):
        on, off, target, history = self.fixture(mean=137., std=159.)
        loss, row = paired_training_loss(on, off, target, history, 137., 159.)
        self.assertAlmostEqual(float(loss), 2.)
        self.assertAlmostEqual(row['mae_on'], 1.5)
        self.assertAlmostEqual(row['mae_off'], 1.)
        self.assertAlmostEqual(row['low_history_mae_on'], 3.)
        self.assertAlmostEqual(row['low_history_mae_off'], 1.)
        self.assertAlmostEqual(row['low_history_excess_mae'], 2.)
        self.assertAlmostEqual(row['risk_penalty'], .5)
        self.assertEqual(row['valid_count'], 24)
        self.assertEqual(row['low_history_valid_count'], 12)
        json.dumps(row, allow_nan=False)

    def test_future_target_flow_does_not_select_safeguard_group(self):
        on, off, target, history = self.fixture()
        _, first = paired_training_loss(on, off, target, history, 0., 1.)
        # Reverse future flow regimes but retain the same observed history and
        # the same per-node ON/OFF errors. The protected group must be unchanged.
        changed_target = target.flip(2)
        changed_on = changed_target+(on-target)
        changed_off = changed_target+(off-target)
        _, second = paired_training_loss(changed_on, changed_off, changed_target, history, 0., 1.)
        for name in ('low_history_valid_count', 'low_history_mae_on',
                     'low_history_mae_off', 'risk_penalty'):
            self.assertEqual(first[name], second[name], name)
        self.assertEqual(second['low_history_mae_on'], 3.)

    def test_only_on_prediction_acquires_gradients(self):
        on, off, target, history = [value.requires_grad_(True) for value in self.fixture()]
        loss, _ = paired_training_loss(on, off, target, history, 0., 1.)
        loss.backward()
        self.assertIsNotNone(on.grad)
        self.assertTrue(torch.isfinite(on.grad).all())
        self.assertGreater(float(on.grad[..., 0, :].abs().sum()), 0.)
        self.assertIsNone(off.grad)
        self.assertIsNone(target.grad)
        self.assertIsNone(history.grad)

    def test_improving_protected_group_has_no_negative_penalty(self):
        _, off, target, history = self.fixture()
        on = target+.5
        loss, row = paired_training_loss(on, off, target, history, 0., 1.)
        self.assertAlmostEqual(float(loss), .5)
        self.assertEqual(row['risk_penalty'], 0.)
        self.assertAlmostEqual(row['low_history_excess_mae'], -.5)

    def test_zero_risk_weight_recovers_full_horizon_original_mae(self):
        on, off, target, history = self.fixture()
        loss, row = paired_training_loss(on, off, target, history, 0., 1., risk_weight=0.)
        self.assertAlmostEqual(float(loss), 1.5)
        self.assertEqual(row['risk_penalty'], 0.)

    def test_history_mean_and_threshold_boundary_not_latest_frame(self):
        on, off, target, history = self.fixture()
        history[..., 0] = 200.
        history[:, -1, 0, 0] = 0.
        history[:, :, 1, 0] = 100.
        loss, row = paired_training_loss(on, off, target, history, 0., 1.)
        self.assertAlmostEqual(float(loss), 1.5)
        self.assertEqual(row['low_history_valid_count'], 0)
        self.assertIsNone(row['low_history_mae_on'])
        self.assertIsNone(row['low_history_mae_off'])
        self.assertIsNone(row['low_history_excess_mae'])
        self.assertEqual(row['risk_penalty'], 0.)

    def test_author_minimum_null_mask_is_shared_by_on_and_off(self):
        on, off, target, history = self.fixture()
        target[..., 0, :] = .5
        on[..., 0, :] = 1000.
        off[..., 0, :] = 500.
        on[..., 1, :] = target[..., 1, :]+2.
        off[..., 1, :] = target[..., 1, :]+1.
        loss, row = paired_training_loss(on, off, target, history, 0., 1.)
        self.assertEqual(row['valid_count'], 12)
        self.assertEqual(row['low_history_valid_count'], 0)
        self.assertAlmostEqual(float(loss), 2.)

    def test_malformed_shapes_bad_scaler_or_weights_fail_closed(self):
        on, off, target, history = self.fixture()
        cases = [
            ((on[:, :11], off, target, history, 0., 1.), {}),
            ((on, off, target, history[:, :11], 0., 1.), {}),
            ((on, off, target, history[..., :1], 0., 1.), {}),
            ((on, off, target, history, 0., 0.), {}),
            ((on, off, target, history, float('nan'), 1.), {}),
            ((on, off, target, history, 0., 1.), {'risk_weight': -.1}),
            ((on, off, target, history, 0., 1.), {'risk_weight': float('inf')}),
            ((on, off, target, history, 0., 1.), {'low_history_threshold': float('nan')}),
        ]
        for args, kwargs in cases:
            with self.subTest(kwargs=kwargs, shape=tuple(args[0].shape)):
                with self.assertRaises(ValueError):
                    paired_training_loss(*args, **kwargs)

    def test_nonfinite_valid_prediction_is_rejected(self):
        on, off, target, history = self.fixture()
        on[0, 0, 0, 0] = float('nan')
        with self.assertRaises(ValueError):
            paired_training_loss(on, off, target, history, 0., 1.)

    def test_all_masked_batch_returns_differentiable_zero_without_nan(self):
        on, off, target, history = self.fixture()
        on.fill_(float('nan'))
        on.requires_grad_(True)
        off.fill_(float('nan'))
        target.zero_()
        loss, row = paired_training_loss(on, off, target, history, 0., 1.)
        self.assertEqual(float(loss.detach()), 0.)
        self.assertEqual(row['valid_count'], 0)
        self.assertEqual(row['low_history_valid_count'], 0)
        self.assertIsNone(row['low_history_mae_on'])
        loss.backward()
        self.assertTrue(torch.equal(on.grad, torch.zeros_like(on)))
        json.dumps(row, allow_nan=False)

    def test_nan_target_and_invalid_prediction_are_excluded_with_shared_mask(self):
        on, off, target, history = self.fixture()
        target[..., 0, :] = float('nan')
        on[..., 0, :] = float('nan')
        off[..., 0, :] = float('nan')
        on.requires_grad_(True)
        loss, row = paired_training_loss(on, off, target, history, 0., 1.)
        self.assertEqual(row['valid_count'], 12)
        self.assertEqual(row['low_history_valid_count'], 0)
        self.assertEqual(float(loss.detach()), 0.)
        loss.backward()
        self.assertTrue(torch.isfinite(on.grad).all())


class ResponsePairMetricsTests(unittest.TestCase):
    def fixture(self):
        target = np.tile(np.array([2., 300.])[None, None], (2, 12, 1))
        off = target+1.
        on = target+np.array([3., 0.])[None, None]
        valid = np.ones_like(target, dtype=bool)
        history_low = np.array([[True, False], [False, True]])
        return on, off, target, valid, history_low

    def test_history_group_and_retrospective_target_bins_remain_distinct(self):
        on, off, target, valid, history_low = self.fixture()
        report = summarize_response_pair(on, off, target, valid, np.array([0]), history_low)
        self.assertAlmostEqual(report['on']['mae_pooled'], 1.5)
        self.assertAlmostEqual(report['mae_gain_pooled'], -.5)
        protected = report['history_low_group']
        self.assertEqual(protected['valid_count'], 24)
        self.assertAlmostEqual(protected['on']['mae_pooled'], 1.5)
        self.assertAlmostEqual(protected['off']['mae_pooled'], 1.)
        self.assertAlmostEqual(protected['gain']['mae_gain_pooled'], -.5)
        risk = report['risks']['groups']
        self.assertEqual(risk['[0,5)']['valid_count'], 24)
        self.assertAlmostEqual(risk['[0,5)']['on']['mae'], 3.)
        self.assertAlmostEqual(risk['[200,inf)']['on']['mae'], 0.)
        self.assertAlmostEqual(risk['all_nodes']['on']['mape'], .75)
        self.assertEqual(report['fixed_hard_windows']['positional_indices'], [0])
        json.dumps(report, allow_nan=False)

    def test_history_group_inherits_valid_mask_and_reports_empty_as_unavailable(self):
        on, off, target, valid, history_low = self.fixture()
        valid[:] = ~history_low[:, None, :]
        on[~valid] = np.nan
        report = summarize_response_pair(on, off, target, valid, np.array([0]), history_low)
        group = report['history_low_group']
        self.assertEqual(group['status'], 'unavailable')
        self.assertEqual(group['valid_count'], 0)
        self.assertIsNone(group['gain']['mae_gain_pooled'])
        self.assertEqual(report['on']['valid_count'], 24)
        json.dumps(report, allow_nan=False)

    def test_history_group_requires_explicit_boolean_station_mask(self):
        on, off, target, valid, history_low = self.fixture()
        for mask in (history_low.astype(np.int8), history_low[:, :1],
                     np.broadcast_to(history_low[:, None], target.shape)):
            with self.subTest(shape=mask.shape), self.assertRaises(ValueError):
                summarize_response_pair(on, off, target, valid, np.array([0]), mask)


if __name__ == '__main__':
    unittest.main()
