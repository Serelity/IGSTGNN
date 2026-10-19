"""Server-only checks for metadata alignment, missingness and fixed graph support."""
import unittest
import numpy as np

from experiments.position_state_gw16.data import (
    aligned_rows, build_position_metadata, build_partition, graph_support)


def rows():
    return [dict(station_id='30', Fwy='I80', Direction='E', Lat='38', Lng='-122', **{'Abs PM':'10'}),
            dict(station_id='10', Fwy='I80', Direction='E', Lat='39', Lng='-121', **{'Abs PM':'12'}),
            dict(station_id='20', Fwy='I80', Direction='W', Lat='', Lng='nan', **{'Abs PM':''}),
            dict(station_id='40', Fwy='', Direction='', Lat='40', Lng='-120', **{'Abs PM':'5'})]


class PositionStateDataTests(unittest.TestCase):
    def test_explicit_station_id_join_is_order_invariant(self):
        source, axis = rows(), [10, 20, 30, 40]
        original = build_position_metadata(source, axis)
        reversed_rows = build_position_metadata(list(reversed(source)), axis)
        self.assertTrue(np.array_equal(original[0], reversed_rows[0]))
        self.assertTrue(np.array_equal(original[1], reversed_rows[1]))
        self.assertEqual(original[3], reversed_rows[3])
        self.assertEqual([int(row['station_id']) for row in aligned_rows(source, axis)], axis)

    def test_duplicate_missing_extra_station_ids_rejected(self):
        for source, axis in ((rows()+[rows()[0]], [10, 20, 30, 40]),
                             (rows(), [10, 20, 30]), (rows(), [10, 20, 30, 50]),
                             (rows(), [10, 20, 30, 30])):
            with self.subTest(axis=axis):
                with self.assertRaises(ValueError):
                    aligned_rows(source, axis)

    def test_missing_numeric_has_independent_masks(self):
        features, categories, sizes, identity = build_position_metadata(rows(), [10, 20, 30, 40])
        self.assertEqual(features.shape, (4, 6))
        np.testing.assert_array_equal(features[1], np.zeros(6))
        np.testing.assert_array_equal(features[0, 3:], np.ones(3))
        np.testing.assert_array_equal(categories[3], np.zeros(2))
        self.assertEqual(sizes, [2, 3])
        self.assertTrue(np.isfinite(features).all())
        self.assertEqual(identity['fitting_scope'], 'fixed_metadata_no_traffic_no_targets')

    def test_postmile_relative_only_within_road_direction(self):
        features, _, _, identity = build_position_metadata(rows(), [10, 20, 30, 40])
        self.assertEqual(float(features[0, 2]), 1.)
        self.assertEqual(float(features[2, 2]), 0.)
        self.assertEqual(float(features[3, 2]), 0.)
        self.assertEqual(len(identity['postmile_groups']), 3)

    def test_metadata_transform_ignores_unapproved_fields(self):
        source = rows()
        expected = build_position_metadata(source, [10, 20, 30, 40])
        changed = [dict(row, future_error='999999', incident_position='7', Holiday='1') for row in source]
        actual = build_position_metadata(changed, [10, 20, 30, 40])
        np.testing.assert_array_equal(expected[0], actual[0])
        self.assertEqual(expected[3], actual[3])

    def test_partition_respects_direction_size_and_postmile_order(self):
        assignment, identity = build_partition(rows(), 1)
        self.assertEqual(len(identity['sizes']), 4)
        self.assertTrue(all(size == 1 for size in identity['sizes']))
        self.assertEqual(sorted(assignment.tolist()), list(range(4)))
        _, grouped = build_partition(rows(), 2)
        self.assertIn([0, 1], grouped['members'])
        self.assertFalse(grouped['physical_conservation_certified'])

    def test_graph_topk_is_deterministic_and_preserves_positive_support(self):
        adjacency = np.array([[0, 4, 4, 0], [1, 0, 2, 0], [0, 0, 0, 0], [1, 0, 0, 0]], dtype=np.float32)
        support = graph_support(adjacency, 1)
        np.testing.assert_allclose(support.sum(1), np.ones(4))
        self.assertGreater(support[0, 1], 0)
        self.assertEqual(support[0, 2], 0)
        self.assertEqual(support[2, 2], 1)
        self.assertTrue(np.array_equal(support, graph_support(adjacency, 1)))

    def test_negative_graph_and_invalid_partition_size_rejected(self):
        with self.assertRaises(ValueError):
            graph_support(np.array([[0, -1], [1, 0]]), 1)
        with self.assertRaises(ValueError):
            build_partition(rows(), 0)


if __name__ == '__main__':
    unittest.main()
