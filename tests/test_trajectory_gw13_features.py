"""Server-only tests for origin descriptors and authenticated input boundaries."""
from contextlib import ExitStack
import copy
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from experiments.branch_readout_gw9 import readout_history
from experiments.branch_readout_gw9.readout_auth import sha256
from experiments.trajectory_diagnostic_gw13 import features as subject


def history(*trajectories):
    return np.asarray([list(values) for values in trajectories], dtype=np.float32).T[None]


class TrajectoryFeatureTests(unittest.TestCase):
    def test_legacy_fields_replay_exact_gw9_float64_operation_order(self):
        raw = (np.arange(2*12*3, dtype=np.float32).reshape(2, 12, 3) / 7.1 - 3.9)
        expected = readout_history.feature_arrays(raw, 132.7139, 84.2817, np.ones((3, 3)))
        actual = subject.features_from_history(raw, 132.7139, 84.2817)
        for name in ('last_flow', 'mean_abs_change'):
            self.assertEqual(actual[name].dtype, np.float32)
            np.testing.assert_array_equal(actual[name], expected[name].astype(np.float32))

    def test_output_names_shapes_and_dtypes(self):
        actual = subject.features_from_history(np.ones((2, 12, 3), dtype=np.float64), 0., 1.)
        self.assertEqual(set(actual), set(subject.FEATURES))
        for name, value in actual.items():
            self.assertEqual(value.shape, (2, 3))
            self.assertEqual(value.dtype, np.int8 if name in ('shape', 'turn_position') else np.float32)

    def test_direction_reverses_sign_and_constant_history_is_zero(self):
        raw = history(range(12), reversed(range(12)), [4.]*12)
        actual = subject.features_from_history(raw, 19., 8.)
        np.testing.assert_array_equal(actual['signed_direction'], [[1., -1., 0.]])
        np.testing.assert_array_equal(actual['turn_position'], [[-1, -1, -1]])

    def test_same_endpoints_and_change_amplitude_can_hide_different_shapes(self):
        raw = history([5, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5],
            [5, 6, 7, 8, 9, 10, 10, 9, 8, 7, 6, 5])
        actual = subject.features_from_history(raw, 0., 1.)
        for name in ('last_flow', 'mean_abs_change', 'signed_direction'):
            self.assertEqual(actual[name][0, 0], actual[name][0, 1])
        np.testing.assert_array_equal(actual['shape'], [[1, 2]])

    def test_block_shape_covers_all_four_codes(self):
        raw = history([3.]*4+[2.]*4+[1.]*4, [3.]*4+[1.]*4+[2.]*4,
            [1.]*4+[3.]*4+[2.]*4, [1.]*4+[2.]*4+[3.]*4)
        actual = subject.features_from_history(raw, 0., 1.)
        np.testing.assert_array_equal(actual['shape'], [[0, 1, 2, 3]])

    def test_block_shape_places_exact_zero_trends_on_nonnegative_side(self):
        raw = history([2.]*12, [1.]*8+[2.]*4, [2.]*4+[1.]*8)
        actual = subject.features_from_history(raw, 0., 1.)
        np.testing.assert_array_equal(actual['shape'], [[3, 3, 1]])

    def test_single_peak_and_trough_use_original_zero_based_center(self):
        peak = [0, 1, 2, 3, 4, 9, 4, 3, 2, 1, 0, -1]
        raw = history(peak, [-value for value in peak])
        actual = subject.features_from_history(raw, 0., 1.)
        np.testing.assert_array_equal(actual['turn_position'], [[5, 5]])

    def test_equal_reversal_strength_chooses_earliest(self):
        raw = history([0, 0, 0, 3, 3, 3, 0, 0, 0, 3, 3, 3])
        actual = subject.features_from_history(raw, 0., 1.)
        np.testing.assert_array_equal(actual['turn_position'], [[4]])

    def test_stronger_later_reversal_wins_over_earlier(self):
        raw = history([0, 0, 0, 3, 3, 3, 0, 0, 0, 6, 6, 6])
        actual = subject.features_from_history(raw, 0., 1.)
        np.testing.assert_array_equal(actual['turn_position'], [[7]])

    def test_flat_intermediate_slopes_do_not_count_as_strict_reversal(self):
        raw = history([0, 0, 0, 3, 3, 3, 3, 3, 3, 0, 0, 0])
        actual = subject.features_from_history(raw, 0., 1.)
        np.testing.assert_array_equal(actual['turn_position'], [[-1]])

    def test_shape_turn_and_direction_are_positive_scale_translation_invariant(self):
        raw = history([0, 1, 2, 3, 4, 9, 4, 3, 2, 1, 0, -1], range(12))
        first = subject.features_from_history(raw, 0., 1.)
        second = subject.features_from_history(raw, 19., 8.)
        for name in ('signed_direction', 'shape', 'turn_position'):
            np.testing.assert_array_equal(first[name], second[name])
        np.testing.assert_array_equal(second['last_flow'], first['last_flow']*8.+19.)
        np.testing.assert_array_equal(second['mean_abs_change'], first['mean_abs_change']*8.)

    def test_input_is_not_mutated(self):
        raw = np.arange(24, dtype=np.float32).reshape(2, 12, 1)
        before = raw.copy()
        subject.features_from_history(raw, 13., 8.)
        np.testing.assert_array_equal(raw, before)

    def test_rejects_wrong_dimensions_counts_and_nonfloating_inputs(self):
        for raw in (np.ones((12, 3)), np.ones((1, 11, 3)), np.ones((0, 12, 3)),
                np.ones((1, 12, 0)), np.ones((1, 12, 3), dtype=np.int64)):
            with self.subTest(shape=raw.shape, dtype=raw.dtype), self.assertRaises(ValueError):
                subject.features_from_history(raw, 0., 1.)

    def test_rejects_nonfinite_history_and_invalid_normalization(self):
        for value in (np.nan, np.inf, -np.inf):
            raw = np.ones((1, 12, 1), dtype=np.float32)
            raw[0, 4, 0] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                subject.features_from_history(raw, 0., 1.)
        raw = np.ones((1, 12, 1), dtype=np.float32)
        for mean, std in ((0., 0.), (0., -1.), (np.nan, 1.), (0., np.inf), (True, 1.)):
            with self.subTest(mean=mean, std=std), self.assertRaises(ValueError):
                subject.features_from_history(raw, mean, std)


class _OnlyHistorySample:
    def __init__(self, value):
        self.value = value

    def __getitem__(self, key):
        if key != 'x_data':
            raise AssertionError('Target or unexpected sample field accessed: '+key)
        return self.value


class _GuardedAuthorArray:
    def __init__(self, values):
        self.values = values
        self.accesses = []

    def __len__(self):
        return 4

    def __getitem__(self, index):
        if index >= 3:
            raise AssertionError('Test sample accessed')
        self.accesses.append(index)
        return _OnlyHistorySample(self.values[index])


class TrajectoryLoadTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        for name in ('incident_all.npy', 'adj_matrix.npy'):
            (self.directory/name).write_bytes(('trusted fixture '+name).encode())
        self.identity = dict(hashes={name: sha256(self.directory/name)
            for name in ('incident_all.npy', 'adj_matrix.npy')},
            normalization=dict(mean=132.7139, std=84.2817, refitted=False),
            splits={name: dict(bounds_half_open=bounds, selected_indices=list(range(*bounds)),
                count=bounds[1]-bounds[0]) for name, bounds in {'train': [0, 2], 'val': [2, 3]}.items()})
        self.values = []
        for index in range(3):
            value = np.ones((12, 2, 3), dtype=np.float32)
            value[..., 0] = np.arange(24, dtype=np.float32).reshape(12, 2)/7.1+index
            # Other input channels cannot accidentally stand in for traffic.
            value[..., 1] = -500.
            value[..., 2] = 900.
            self.values.append(value)
        self.array = _GuardedAuthorArray(self.values)
        self.cache = {}
        for split, bounds in {'train': [0, 2], 'val': [2, 3]}.items():
            raw = np.stack(self.values[bounds[0]:bounds[1]])[..., 0]
            result = readout_history.feature_arrays(raw, 132.7139, 84.2817, np.ones((2, 2)))
            self.cache[split] = dict(features={name: result[name].astype(np.float32)
                for name in ('last_flow', 'mean_abs_change')})
        patches = ExitStack()
        self.addCleanup(patches.close)
        # Small input fixtures exercise the production guard implementation;
        # command-line execution exposes no way to change these fixed constants.
        patches.enter_context(mock.patch.object(subject, 'AUTHOR_COUNT', 4))
        patches.enter_context(mock.patch.object(subject, 'SPLIT_BOUNDS', {'train': [0, 2], 'val': [2, 3]}))
        patches.enter_context(mock.patch.object(subject, 'NODE_COUNT', 2))
        patches.enter_context(mock.patch.object(subject, 'CHUNK_SIZE', 1))
        self.loader = patches.enter_context(mock.patch.object(subject.readout_history,
            '_load_array', return_value=self.array))

    def test_only_train_validation_x_data_zero_is_used_and_cache_replays(self):
        values, metadata = subject.load_features(self.directory, self.identity, self.cache)
        self.assertEqual(self.array.accesses, [0, 1, 2])
        self.assertEqual(set(values), {'train', 'val'})
        for split in ('train', 'val'):
            for name in ('last_flow', 'mean_abs_change'):
                np.testing.assert_array_equal(values[split][name], self.cache[split]['features'][name])
        self.assertEqual(set(metadata['file_digests']),
            {str((self.directory/name).resolve()) for name in self.identity['hashes']})
        self.assertTrue(metadata['test_objects_deserialized'])
        self.assertFalse(metadata['test_samples_accessed'])
        self.assertFalse(metadata['training_targets_used'])
        self.assertTrue(metadata['inherited_feature_replay']['passed'])

    def test_any_fingerprint_failure_prevents_object_deserialization(self):
        (self.directory/'adj_matrix.npy').write_bytes(b'changed graph despite not computing neighbors')
        with self.assertRaisesRegex(ValueError, 'fingerprint'):
            subject.load_features(self.directory, self.identity, self.cache)
        self.loader.assert_not_called()

    def test_rejects_parent_path_fingerprint_before_deserialization(self):
        identity = copy.deepcopy(self.identity)
        identity['hashes']['../outside.npy'] = '0'*64
        with self.assertRaisesRegex(ValueError, 'fingerprint'):
            subject.load_features(self.directory, identity, self.cache)
        self.loader.assert_not_called()

    def test_rejects_refitted_normalization_before_deserialization(self):
        identity = copy.deepcopy(self.identity)
        identity['normalization']['refitted'] = True
        with self.assertRaisesRegex(ValueError, 'Refitted'):
            subject.load_features(self.directory, identity, self.cache)
        self.loader.assert_not_called()

    def test_rejects_missing_history_fingerprint_before_deserialization(self):
        identity = copy.deepcopy(self.identity)
        del identity['hashes']['incident_all.npy']
        with self.assertRaisesRegex(ValueError, 'history fingerprint'):
            subject.load_features(self.directory, identity, self.cache)
        self.loader.assert_not_called()

    def test_rejects_changed_complete_author_array_count(self):
        self.loader.return_value = []
        with self.assertRaisesRegex(ValueError, 'author history count'):
            subject.load_features(self.directory, self.identity, self.cache)

    def test_rejects_changed_bounds_or_shuffled_indices_before_deserialization(self):
        for field, value in (('bounds_half_open', [0, 3]), ('selected_indices', [1, 0]), ('count', 3)):
            identity = copy.deepcopy(self.identity)
            identity['splits']['train'][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'split/order'):
                subject.load_features(self.directory, identity, self.cache)
        self.loader.assert_not_called()

    def test_cached_feature_replay_rejects_one_float32_ulp_difference(self):
        cache = copy.deepcopy(self.cache)
        old = cache['val']['features']['mean_abs_change'][0, 0]
        cache['val']['features']['mean_abs_change'][0, 0] = np.nextafter(old, np.float32(np.inf))
        with self.assertRaisesRegex(ValueError, 'Exact GW10 cached val/mean_abs_change'):
            subject.load_features(self.directory, self.identity, cache)

    def test_cached_feature_replay_rejects_dtype_and_shape_difference(self):
        for replacement in (self.cache['train']['features']['last_flow'].astype(np.float64),
                self.cache['train']['features']['last_flow'][:1]):
            cache = copy.deepcopy(self.cache)
            cache['train']['features']['last_flow'] = replacement
            with self.subTest(shape=replacement.shape, dtype=replacement.dtype), self.assertRaisesRegex(
                    ValueError, 'Exact GW10 cached train/last_flow'):
                subject.load_features(self.directory, self.identity, cache)

    def test_rejects_bad_production_x_shape_dtype_or_nonfinite_values(self):
        original = self.values[0]
        for replacement in (original[:11], original.astype(np.float64), original.copy()):
            if replacement.shape == original.shape and replacement.dtype == np.float32:
                replacement[0, 0, 0] = np.nan
            self.values[0] = replacement
            with self.subTest(shape=replacement.shape, dtype=replacement.dtype), self.assertRaisesRegex(
                    ValueError, 'Invalid authenticated historical x_data'):
                subject.load_features(self.directory, self.identity, self.cache)
        self.values[0] = original

    def test_final_rehash_detects_input_changed_during_deserialization(self):
        def changed(_path):
            (self.directory/'adj_matrix.npy').write_bytes(b'changed while processing')
            return self.array
        self.loader.side_effect = changed
        with self.assertRaisesRegex(ValueError, 'changed during analysis'):
            subject.load_features(self.directory, self.identity, self.cache)


if __name__ == '__main__':
    unittest.main()
