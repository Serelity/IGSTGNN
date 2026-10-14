"""Independent preservation, hand-calculated, and batching checks for shuffling."""
import copy
import unittest

import numpy as np

from experiments.chronological.min_frozen_location_shuffle import (
    SCOPE, assert_shuffle_preserved, make_location_shuffle,
)


def event_payload():
    relation = np.zeros((5, 5, 3), dtype=np.float32)
    relation[0, 1] = [1, 2, 3]
    relation[0, 4] = [4, 5, 6]
    # A duplicate-vector eligible row still moves donors but not values.
    relation[1, [0, 2, 3]] = [0, .5, 1]
    relation[2, 2] = [0, .25, 0]
    relation[3, [0, 4]] = [[1, 1, 1], [2, 2, 2]]
    relation[4, [0, 4]] = [[3, 3, 3], [4, 4, 4]]
    return dict(relation=relation,
                type=np.arange(5, dtype=np.int64),
                description=np.arange(5, dtype=np.int64),
                type_valid=np.array([True, False, True, True, True]),
                description_valid=np.array([False, True, True, True, True]),
                report_present=np.array([True, True, True, False, True]),
                relation_available=np.array([True, True, True, True, False]))


class LocationShuffleTests(unittest.TestCase):
    def test_two_support_hand_swap_and_duplicate_vector_audit(self):
        events = event_payload()
        changed, donors, audit = make_location_shuffle(events, np.arange(5)+100)
        np.testing.assert_array_equal(donors[0], [0, 4, 2, 3, 1])
        np.testing.assert_array_equal(changed['relation'][0, 1], [4, 5, 6])
        np.testing.assert_array_equal(changed['relation'][0, 4], [1, 2, 3])
        np.testing.assert_array_equal(changed['relation'][1], events['relation'][1])
        self.assertEqual(audit['scope'], SCOPE)
        self.assertEqual(audit['eligible_rows'], 2)
        self.assertEqual(audit['effective_changed_rows'], 1)
        self.assertEqual(audit['unchanged_due_single_or_zero_rows'], 3)
        self.assertEqual(audit['eligible_rows_with_no_numeric_change'], 1)
        self.assertEqual(audit['rows_with_duplicate_supported_vectors'], 1)
        self.assertEqual(audit['moved_station_count'], 5)
        self.assertEqual(audit['changed_vector_count'], 2)
        self.assertEqual(audit['unchanged_moved_vector_count'], 3)
        self.assertEqual(audit['vector_abs_change']['sum'], 18.)
        self.assertAlmostEqual(audit['vector_abs_change']['mean'], 18/75)
        self.assertEqual(audit['vector_abs_change']['max'], 3.)
        self.assertEqual(audit['per_sample'][0]['sample_index'], 100)

    def test_single_support_absent_report_unavailable_relation_and_zero_support_are_identity(self):
        events = event_payload()
        events['relation'][1] = 0
        result, donors, audit = make_location_shuffle(events, np.arange(5))
        for row in [1, 2, 3, 4]:
            np.testing.assert_array_equal(donors[row], np.arange(5))
            np.testing.assert_array_equal(result['relation'][row], events['relation'][row])
        self.assertEqual(audit['per_sample'][2]['supported_station_count'], 1)
        self.assertEqual(audit['per_sample'][3]['supported_station_count'], 0)
        self.assertEqual(audit['per_sample'][4]['supported_station_count'], 0)

    def test_batch_boundaries_and_row_order_do_not_change_shuffle(self):
        events = event_payload()
        events['relation'] = np.arange(75, dtype=np.float32).reshape(5, 5, 3)
        events['report_present'][:] = True
        events['relation_available'][:] = True
        ids = np.array([14, 57, 92, 8, 302])
        full, donors, _ = make_location_shuffle(events, ids, seed=17)
        for selection in [np.array([0, 1]), np.array([2, 3, 4]), np.array([4, 2, 0])]:
            batch = {key: value[selection] for key, value in events.items()}
            actual, actual_donors, _ = make_location_shuffle(batch, ids[selection], seed=17)
            np.testing.assert_array_equal(actual['relation'], full['relation'][selection])
            np.testing.assert_array_equal(actual_donors, donors[selection])

    def test_global_rng_and_all_original_fields_unchanged_and_no_aliases(self):
        events, before = event_payload(), event_payload()
        state = np.random.get_state()
        actual, _, _ = make_location_shuffle(events, np.arange(5))
        after = np.random.get_state()
        self.assertEqual(state[0], after[0])
        np.testing.assert_array_equal(state[1], after[1])
        self.assertEqual(state[2:], after[2:])
        for key in events:
            np.testing.assert_array_equal(events[key], before[key])
            self.assertFalse(np.shares_memory(events[key], actual[key]))
            if key != 'relation':
                np.testing.assert_array_equal(actual[key], before[key])

    def test_whole_vectors_are_copied_exactly_including_signed_zero(self):
        events = event_payload()
        events['relation'][0, 1] = [0., -0., 1.]
        events['relation'][0, 4] = [-0., 0., 2.]
        result, _, _ = make_location_shuffle(events, np.arange(5))
        np.testing.assert_array_equal(result['relation'][0, 4].view(np.uint32),
                                      events['relation'][0, 1].view(np.uint32))

    def test_rejects_new_support_injection_or_non_whole_vector_copy(self):
        events = event_payload()
        result, donors, _ = make_location_shuffle(events, np.arange(5))
        result['relation'][0, 0, 0] = 1.
        with self.assertRaisesRegex(ValueError, 'mask changed'):
            assert_shuffle_preserved(events, result, donors)
        result, donors, _ = make_location_shuffle(events, np.arange(5))
        result['relation'][0, 1, 0] += .1
        with self.assertRaisesRegex(ValueError, 'whole-vector'):
            assert_shuffle_preserved(events, result, donors)

    def test_rejects_mutated_donors_non_permutation_and_self_donors(self):
        events = event_payload()
        result, donors, _ = make_location_shuffle(events, np.arange(5))
        bad = donors.copy()
        bad[0, 0] = 4
        with self.assertRaisesRegex(ValueError, 'permutation'):
            assert_shuffle_preserved(events, result, bad)
        bad = donors.copy()
        bad[0] = np.arange(5)
        with self.assertRaisesRegex(ValueError, 'no self donor'):
            assert_shuffle_preserved(events, result, bad)
        bad = donors.copy()
        bad[0, [0, 2]] = [2, 0]
        with self.assertRaisesRegex(ValueError, 'outside'):
            assert_shuffle_preserved(events, result, bad)

    def test_rejects_flag_category_changes_and_ineligible_donor_moves(self):
        events = event_payload()
        for key in ['report_present', 'relation_available', 'type_valid',
                    'description_valid', 'type', 'description']:
            result, donors, _ = make_location_shuffle(events, np.arange(5))
            result[key][0] = not result[key][0] if result[key].dtype == np.bool_ else 6
            with self.subTest(field=key), self.assertRaises(ValueError):
                assert_shuffle_preserved(events, result, donors)
        result, donors, _ = make_location_shuffle(events, np.arange(5))
        donors[3, [0, 4]] = [4, 0]
        with self.assertRaises(ValueError):
            assert_shuffle_preserved(events, result, donors)

    def test_rejects_invalid_schema_types_axes_nonfinite_and_sample_ids(self):
        events = event_payload()
        for key, value in [('relation', events['relation'].astype(np.float64)),
                           ('relation', events['relation'][:, :, :2]),
                           ('relation', np.full((5, 5, 3), np.nan, np.float32)),
                           ('type', np.arange(5, dtype=np.float32)),
                           ('type_valid', np.ones(5, dtype=np.int64)),
                           ('report_present', np.ones((5, 1), dtype=bool))]:
            bad = copy.deepcopy(events)
            bad[key] = value
            with self.subTest(field=key, shape=value.shape), self.assertRaises(ValueError):
                make_location_shuffle(bad, np.arange(5))
        for ids in [[1, 1, 2, 3, 4], [-1, 1, 2, 3, 4], [0., 1., 2., 3., 4.],
                    [0, 1], np.ones(5, dtype=bool), np.arange(5).reshape(1, 5)]:
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                make_location_shuffle(events, ids)
        for seed in [-1, 2**32, True, 1.5]:
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                make_location_shuffle(events, np.arange(5), seed=seed)

    def test_empty_sample_axis_is_valid_and_outputs_empty_audit(self):
        events = {key: value[:0] for key, value in event_payload().items()}
        result, donors, audit = make_location_shuffle(events, np.array([], dtype=np.int64))
        self.assertEqual(result['relation'].shape, (0, 5, 3))
        self.assertEqual(donors.shape, (0, 5))
        self.assertEqual(audit['vector_abs_change']['sum'], 0.)
        self.assertEqual(audit['per_sample'], [])


if __name__ == '__main__':
    unittest.main()
