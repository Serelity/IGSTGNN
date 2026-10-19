"""Server-only numerical diagnostic tests; no model/data execution locally."""
import json
import unittest

import numpy as np

from experiments.branch_readout_gw9 import replay


class BranchReadoutReplayTests(unittest.TestCase):
    def test_exact_equal_requires_dtype_shape_finite_and_values(self):
        value = np.arange(12, dtype=np.float32).reshape(2, 3, 2)
        result = replay.compare_arrays(value.copy(), value)
        self.assertTrue(result['exact_equal'])
        self.assertEqual(result['mismatch_count'], 0)
        self.assertEqual((result['max_abs'], result['mean_abs'], result['rms']), (0., 0., 0.))
        self.assertIsNone(result['first_mismatch'])
        dtype_only = replay.compare_arrays(value.astype(np.float64), value)
        self.assertFalse(dtype_only['exact_equal'])
        self.assertEqual(dtype_only['mismatch_count'], 0)
        shape_only = replay.compare_arrays(value.reshape(3, 2, 2), value)
        self.assertFalse(shape_only['exact_equal'])
        self.assertIsNone(shape_only['mismatch_count'])
        self.assertIsNone(shape_only['max_abs'])

    def test_difference_statistics_and_coordinates_are_exact(self):
        expected = np.zeros((2, 2, 2), dtype=np.float32)
        actual = expected.copy(); actual[0, 1, 1] = 3.; actual[1, 0, 0] = -4.
        result = replay.compare_arrays(actual, expected)
        self.assertFalse(result['exact_equal'])
        self.assertEqual(result['mismatch_count'], 2)
        self.assertEqual(result['max_abs'], 4.)
        self.assertEqual(result['mean_abs'], 7./8)
        self.assertAlmostEqual(result['rms'], np.sqrt(25./8))
        self.assertEqual(result['first_mismatch'], dict(index=[0, 1, 1], actual=3., expected=0.))
        replay._annotate_first(result, [3909, 3910], [100, 200])
        self.assertEqual(result['first_mismatch']['sample_id'], 3909)
        self.assertEqual(result['first_mismatch']['horizon'], 2)
        self.assertEqual(result['first_mismatch']['node_id'], 200)

    def test_nonfinite_is_not_exact_and_serializes_without_nan(self):
        actual = np.array([np.nan, np.inf, -np.inf, 1.], dtype=np.float32)
        expected = np.array([np.nan, np.inf, 0., 0.], dtype=np.float32)
        result = replay.compare_arrays(actual, expected)
        self.assertFalse(result['exact_equal'])
        self.assertEqual(result['nonfinite_counts'], dict(actual=3, expected=2))
        self.assertEqual(result['mismatch_count'], 3)
        self.assertEqual(result['first_mismatch'], dict(index=[0], actual=None, expected=None))
        self.assertEqual((result['max_abs'], result['mean_abs'], result['rms']), (1., 1., 1.))
        json.dumps(result, allow_nan=False)
        entirely_nonfinite = replay.compare_arrays(np.array([np.inf]), np.array([np.inf]))
        self.assertFalse(entirely_nonfinite['exact_equal'])
        self.assertIsNone(entirely_nonfinite['max_abs'])
        json.dumps(entirely_nonfinite, allow_nan=False)

    def test_empty_arrays_and_shape_mismatch_remain_serializable(self):
        empty = replay.compare_arrays(np.empty((0, 12, 2), dtype=np.float32),
            np.empty((0, 12, 2), dtype=np.float32))
        self.assertTrue(empty['exact_equal'])
        self.assertEqual(empty['mismatch_count'], 0)
        self.assertIsNone(empty['max_abs'])
        other = replay.compare_arrays(np.ones((1, 2), dtype=np.float32), np.ones((2,), dtype=np.float64))
        self.assertFalse(other['exact_equal'])
        json.dumps([empty, other], allow_nan=False)

    def test_partial_or_full_diagnostic_never_grants_a_screening_gate(self):
        partial = replay.diagnostic_scope(1, 18)
        full = replay.diagnostic_scope(18, 18)
        self.assertTrue(partial['partial_validation_replay'])
        self.assertFalse(partial['full_validation_replayed'])
        self.assertTrue(full['full_validation_replayed'])
        self.assertFalse(full['partial_validation_replay'])
        for result in (partial, full):
            self.assertFalse(result['screening_performed'])
            self.assertFalse(result['candidate_for_confirmation'])
            self.assertFalse(result['joint_eligibility_changed'])
            self.assertFalse(result['test_evaluated'])
            self.assertFalse(result['predictive_accuracy_evidence'])
            self.assertEqual(result['optimizer_updates'], 0)
            self.assertNotIn('eligible_for_joint_screen', result)
        for selected, total in ((0, 18), (19, 18), (1, 0), (True, 18)):
            with self.assertRaises(ValueError): replay.diagnostic_scope(selected, total)

    def test_parser_rejects_budget_or_capacity_controls(self):
        common = ['--gw8-run', 'old', '--output-dir', 'diagnostics']
        args = replay.parse_args(common)
        self.assertEqual(args.batches, 1)
        self.assertEqual(args.hidden, 16)
        self.assertEqual(replay.parse_args(common+['--batches', '0']).batches, 0)
        self.assertTrue(args.output_dir.is_absolute())
        for extra in (['--batches', '-1'], ['--hidden', '32'], ['--epochs', '3'],
                ['--resume'], ['--threads', '0']):
            with self.subTest(extra=extra), self.assertRaises(SystemExit): replay.parse_args(common+extra)


if __name__ == '__main__': unittest.main()
