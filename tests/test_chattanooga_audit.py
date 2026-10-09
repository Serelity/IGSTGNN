"""Small independent checks for audit conclusions that affect experiment design."""
import importlib.util
import io
import unittest
from pathlib import Path
from zipfile import ZipFile

import numpy as np

SPEC = importlib.util.spec_from_file_location('chattanooga_audit', Path(__file__).parents[1] / 'experiments/chronological/audit_chattanooga.py')
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


class ChattanoogaAuditTests(unittest.TestCase):
    def test_sensor_alias_uses_explicit_metadata_not_raw_name_equality(self):
        row = {'Road': '00I75', 'Heading': 'S', 'Mile': '013.7', 'Name': 'RDG-00I75-013.7S'}
        self.assertEqual(audit.sensor_alias(row), '00I75S13.7')

    def test_midnight_is_unwrapped_about_report_not_sorted_by_clock(self):
        np.testing.assert_array_equal(audit.relative_times(['23:59:30', '00:00:00', '00:00:30'], 0), [-30, 0, 30])
        np.testing.assert_array_equal(audit.relative_times(['23:59:30', '00:00:00'], 86370), [0, 30])

    def test_duplicate_times_remain_visible(self):
        t = audit.relative_times(['12:00:00', '12:00:00', '12:01:00'], 43200)
        np.testing.assert_array_equal(np.diff(t), [0, 60])

    def test_zero_missing_and_unmapped_neighbour_are_distinct(self):
        a = np.ones((2, 11, 3))
        a[:, 0] = np.nan
        a[0, 5, 0] = np.nan
        a[0, 5, 1] = 0
        p = audit.Profile()
        p.add(a, np.isnan(a), [None] + [str(i) for i in range(10)], {})
        r = p.result()
        self.assertEqual(r['missing_on_unmapped_hops_cells'], 6)
        self.assertEqual(r['missing_on_mapped_hops_cells'], 1)
        self.assertEqual(r['metrics']['volume']['zero'], 1)
        self.assertEqual(r['metrics']['volume']['nonfinite'], 2)

    def test_overlap_matches_independent_sensor_time_enumeration(self):
        records = []
        for ids, times, label, split in [(['a', 'b'], [0, 30], 1, 'train'),
                                         (['b', 'c'], [30, 60], 0, 'val'),
                                         (['a', 'b'], [86400, 86430], 0, 'val')]:
            records.append({'_ids': ids, '_times': np.array(times), '_array': np.ones((2, 2, 3)),
                            'label': label, 'best': True, 'candidate_calendar_split': split})
        keys = [{(s, t) for s in r['_ids'] for t in r['_times']} for r in records]
        pairs = [(i, j) for i in range(3) for j in range(i) if keys[i] & keys[j]]
        r = audit.overlap_audit(records)
        self.assertEqual(r['window_pairs_with_shared_sensor_timestamp'], len(pairs))
        self.assertEqual(r['shared_sensor_timestamps_summed_over_pairs'], sum(len(keys[i] & keys[j]) for i, j in pairs))
        self.assertEqual(r['pairs_with_different_window_labels'], 1)
        self.assertEqual(r['pairs_crossing_candidate_calendar_split'], 1)
        self.assertEqual(r['finite_value_disagreements_absolute_gt_1e_6'], 0)

    def test_invalid_archive_path_is_rejected_without_extraction(self):
        buffer = io.BytesIO()
        with ZipFile(buffer, 'w') as z:
            z.writestr('../escape.csv', 'x\n1\n')
        buffer.seek(0)
        with ZipFile(buffer) as z, self.assertRaises(ValueError):
            audit.preflight(z)

    def test_nonrectangular_csv_is_not_silently_skipped(self):
        with self.assertRaises(ValueError):
            audit.table(b'x,y\n1,2\n3\n')


if __name__ == '__main__':
    unittest.main()
