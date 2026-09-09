import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


class CheckOverlapTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.data_root = Path(self.temp_dir.name)
        self.dataset_dir = self.data_root / 'Fixture'
        self.dataset_dir.mkdir()
        self.script = Path(__file__).resolve().parents[1] / 'data/xtraffic/check_overlap.py'
        self.rows = {
            'train': [self.sample([1, 2], [3, 4])],
            'val': [self.sample([11, 12], [13, 14])],
            'test': [self.sample([21, 22], [23, 24])],
        }
        np.savez(self.dataset_dir / 'incident_stats.npz', mean=0.0, std=1.0,
                 normalized=True)

    @staticmethod
    def sample(x_values, y_values):
        def frames(values):
            # All frames share their clock features; traffic distinguishes them.
            return np.asarray([[[v, 0.25, 0.5], [v + 100, 0.25, 0.5]]
                               for v in values], dtype=np.float32)
        return {'x_data': frames(x_values), 'y_data': frames(y_values)}

    def save_splits(self):
        for split, rows in self.rows.items():
            np.save(self.dataset_dir / f'incident_{split}.npy',
                    np.asarray(rows, dtype=object))

    def run_check(self, *args):
        return subprocess.run(
            [sys.executable, str(self.script), '--dataset', 'Fixture',
             '--data_root', str(self.data_root), *map(str, args)],
            capture_output=True, text=True, timeout=30,
        )

    def test_same_clock_with_different_traffic_is_not_overlap(self):
        self.save_splits()
        result = self.run_check('--fail-on-overlap')
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertFalse(report['has_cross_split_frame_overlap'])
        self.assertEqual(report['samples'], {'train': 1, 'val': 1, 'test': 1})
        self.assertEqual(report['pairs']['train_vs_test']['overlapping_unique_xy_frames'], 0)

    def test_all_four_cross_split_xy_directions_are_detected(self):
        for left_part, right_part in [('x', 'x'), ('x', 'y'), ('y', 'x'), ('y', 'y')]:
            with self.subTest(left_part=left_part, right_part=right_part):
                self.rows['test'] = [self.sample([21, 22], [23, 24])]
                self.rows['test'][0][right_part + '_data'][0] = (
                    self.rows['train'][0][left_part + '_data'][0])
                self.save_splits()
                result = self.run_check('--fail-on-overlap')
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertTrue(result.stdout.strip(), result.stderr)
                report = json.loads(result.stdout)
                pair = report['pairs']['train_vs_test']
                self.assertTrue(report['has_cross_split_frame_overlap'])
                self.assertEqual(pair['overlapping_unique_xy_frames'], 1)
                self.assertEqual(pair['overlapping_unique_frames_by_part'][
                    left_part + '_vs_' + right_part], 1)

    def test_val_test_overlap_is_checked_without_train_overlap(self):
        self.rows['test'][0]['y_data'][0] = self.rows['val'][0]['x_data'][0]
        self.save_splits()
        result = self.run_check('--fail-on-overlap')
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertTrue(result.stdout.strip(), result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report['pairs']['val_vs_test']['overlapping_unique_xy_frames'], 1)
        self.assertEqual(report['pairs']['train_vs_test']['overlapping_unique_xy_frames'], 0)

    def test_repeated_targets_count_occurrences_separately_from_unique_frames(self):
        self.rows['train'] = [self.sample([1, 2], [1, 4])]
        self.rows['test'] = [self.sample([21, 22], [1, 1])]
        self.save_splits()
        result = self.run_check()
        self.assertEqual(result.returncode, 0, result.stderr)
        pair = json.loads(result.stdout)['pairs']['train_vs_test']
        self.assertEqual(pair['overlapping_unique_xy_frames'], 1)
        self.assertEqual(pair['right_y_frame_occurrences'], 2)
        self.assertEqual(pair['right_y_matches_left_xy_occurrences'], 2)
        self.assertEqual(pair['right_y_matches_left_x_occurrences'], 2)
        self.assertEqual(pair['right_y_matches_left_y_occurrences'], 2)

    def test_duplicates_only_within_one_split_do_not_fail_gate(self):
        self.rows['train'] *= 2
        self.save_splits()
        result = self.run_check('--fail-on-overlap')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(json.loads(result.stdout)['has_cross_split_frame_overlap'])

    def test_reads_actual_splits_without_needing_incident_all(self):
        self.rows['train'][0]['x_data'][0] = self.rows['test'][0]['y_data'][0]
        self.save_splits()
        result = self.run_check()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(json.loads(result.stdout)['has_cross_split_frame_overlap'])
        self.assertFalse((self.dataset_dir / 'incident_all.npy').exists())

    def test_default_check_and_explicit_json_output_preserve_input_files(self):
        self.save_splits()
        before = {path.name: path.read_bytes() for path in self.dataset_dir.iterdir()}
        result = self.run_check()
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        for key in ('train', 'val', 'test', 'stats'):
            suffix = 'npz' if key == 'stats' else 'npy'
            filename = f'incident_{key}.{suffix}'
            self.assertEqual(report['fingerprints'][key]['sha256'],
                             hashlib.sha256(before[filename]).hexdigest())
            self.assertEqual(Path(report['fingerprints'][key]['path']),
                             self.dataset_dir / filename)
        self.assertEqual(before, {path.name: path.read_bytes()
                                  for path in self.dataset_dir.iterdir()})
        output = self.data_root / 'overlap.json'
        result = self.run_check('--output', output)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(output.read_text()), report)
        self.assertEqual(before, {path.name: path.read_bytes()
                                  for path in self.dataset_dir.iterdir()})

    def test_output_cannot_overwrite_an_input_or_existing_report(self):
        self.save_splits()
        result = self.run_check()
        self.assertEqual(result.returncode, 0, result.stderr)
        output = self.data_root / 'existing.json'
        output.write_text('keep this report', encoding='utf-8')
        for target in (self.dataset_dir / 'incident_test.npy', output):
            with self.subTest(target=target.name):
                before = target.read_bytes()
                result = self.run_check('--output', target)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(target.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
