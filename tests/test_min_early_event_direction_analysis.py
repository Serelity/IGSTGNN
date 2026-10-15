"""OFF-baseline correction directions and read-only archive protection."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experiments.chronological import analyze_min_early_event_direction as analysis
from experiments.chronological import min_early_event_direction_metrics as metrics
from tests.test_min_early_event_gain_analysis import fixture, edit_json


def point_fixture():
    target = np.full((1, 12, 10), 10., np.float32)
    off = np.broadcast_to(np.array([12, 12, 12, 12, 12, 12, 12, 10, 8, 8], np.float32), target.shape).copy()
    on = np.broadcast_to(np.array([12, 11, 10, 9, 8, 7, 13, 11, 9, 13], np.float32), target.shape).copy()
    predictions = dict(STATE_ON=np.full_like(target, 8.), EVENT_ON=on, EVENT_OFF=off,
                       EVENT_SEMANTIC_UNK=on.copy(), EVENT_LOCATION_SHUFFLE=on.copy())
    direct, potential = np.zeros((1, 10), bool), np.zeros((1, 10), bool)
    direct[:, 0], potential[:, 1] = True, True
    regions = dict(all_nodes=np.ones((1, 10), bool), direct_report_nodes=direct, potential_only=potential,
                   remaining_nodes=~(direct | potential), empty=np.zeros((1, 10), bool),
                   **{'road:WEST': np.arange(10) < 5})
    return predictions, target, np.ones_like(target, bool), regions, np.array([100]), np.arange(1000, 1010)


class DirectionMetricTests(unittest.TestCase):
    def test_hand_calculated_eight_directions_include_toward_but_harmful_overshoot(self):
        args = point_fixture()
        actual = metrics.direction_codes(args[0]['EVENT_OFF'][0, 0], args[0]['EVENT_ON'][0, 0], args[1][0, 0])
        np.testing.assert_array_equal(actual, [0, 1, 2, 3, 4, 5, 6, 7, 1, 5])
        result = metrics.analyze_direction(*args)
        global_metric = result['groups']['all_nodes']
        self.assertAlmostEqual(global_metric['mae_gain_pooled'], .1)
        self.assertAlmostEqual(global_metric['gain_positive_pooled'], .5)
        self.assertAlmostEqual(global_metric['gain_negative_pooled'], .4)
        self.assertEqual(global_metric['directions']['overshoot_worsened']['valid_count'], 24)
        self.assertEqual(global_metric['directions']['overshoot_worsened']['P_gain_pooled'], -1.)
        self.assertEqual(global_metric['directions']['overshoot_equal_error']['P_gain_pooled'], 0.)
        self.assertEqual(global_metric['directions']['exact_OFF_changed']['P_gain_pooled'], -1.)
        self.assertEqual(global_metric['directions']['reached_target']['P_gain_pooled'], 2.)
        self.assertAlmostEqual(global_metric['prediction_change']['up_fraction'], .4)
        self.assertAlmostEqual(global_metric['prediction_change']['down_fraction'], .5)
        self.assertAlmostEqual(global_metric['prediction_change']['unchanged_fraction'], .1)
        json.dumps(result, allow_nan=False)

    def test_OFF_not_STATE_defines_bias_and_every_partition_conserves_contributions(self):
        result = metrics.analyze_direction(*point_fixture())
        groups = result['groups']
        self.assertEqual(groups['off_bias:over']['valid_count'], 84)
        self.assertEqual(groups['off_bias:under']['valid_count'], 24)
        self.assertEqual(groups['off_bias:exact']['valid_count'], 12)
        # STATE is under for every point; OFF is over at seven stations.
        self.assertEqual(groups['state_off_bias:STATE_under|OFF_over']['valid_count'], 84)
        self.assertEqual(groups['state_off_bias:STATE_over|OFF_over']['valid_count'], 0)
        for family in ('flow', 'OFF_bias', 'STATE_OFF_bias', 'flow_OFF_bias', 'partition_OFF_bias', 'direction', 'prediction_delta', 'P_sign'):
            selected = [groups[name] for name, definition in result['group_definitions'].items() if definition['family'] == family]
            self.assertEqual(sum(row['valid_count'] for row in selected), groups['all_nodes']['valid_count'])
            for field in ('global_mae_contribution_pooled', 'global_mae_contribution_macro'):
                self.assertAlmostEqual(sum(row[field] for row in selected), groups['all_nodes'][field])
        self.assertIsNone(groups['empty']['mae_gain_pooled'])
        self.assertEqual(groups['empty']['global_mae_contribution_pooled'], 0.)
        self.assertTrue(all(value['status'] == 'PASS' for value in result['reconciliation'].values()))
        for group in groups.values():
            self.assertEqual(sum(row['valid_count'] for row in group['directions'].values()), group['valid_count'])

    def test_original_mask_excludes_zero_and_tolerance_does_not_change_direction(self):
        args = list(point_fixture())
        args[1][0, 0, 0], args[2][0, 0, 0] = 0., False
        low = metrics.analyze_direction(*args, response_tol=.001)
        high = metrics.analyze_direction(*args, response_tol=100.)
        self.assertEqual(low['groups']['all_nodes']['valid_count'], 119)
        self.assertEqual(low['groups']['all_nodes']['directions'], high['groups']['all_nodes']['directions'])
        self.assertGreater(low['groups']['all_nodes']['prediction_change']['response_fraction'], 0.)
        self.assertEqual(high['groups']['all_nodes']['prediction_change']['response_fraction'], 0.)
        self.assertEqual(len(metrics.group_rows(low)), len(low['groups'])*13)
        self.assertEqual(len(metrics.direction_rows(low)), len(low['groups'])*8*13)
        rows = metrics.group_rows(low)
        self.assertTrue(all(set(row) == set(rows[0]) for row in rows))


class DirectionArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=analysis.REPO)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def run_source(self, source, output=None):
        with redirect_stdout(io.StringIO()):
            return analysis.main(['--evaluation-dir', str(source), '--output-dir', str(output or self.root/'direction')])

    def test_complete_archive_produces_direction_tables_without_changing_source(self):
        source = fixture(self.root)
        before = {path.name: path.read_bytes() for path in source.iterdir()}
        result = self.run_source(source)
        self.assertEqual(result['status'], analysis.STATUS)
        self.assertEqual(result['new_inference_calls'], 0)
        self.assertEqual(result['optimizer_updates'], 0)
        self.assertFalse(result['checkpoint_loaded'] or result['raw_data_read'] or result['test_evaluated'])
        self.assertEqual(before, {path.name: path.read_bytes() for path in source.iterdir()})
        for name in ('analysis.json', 'report.md', 'summary.json', 'provenance.json', 'preservation.json', 'groups.csv',
                     'OFF_bias.csv', 'STATE_OFF_bias.csv', 'flow.csv', 'regions.csv', 'roads.csv', 'directions.csv',
                     'horizons.csv', 'group_directions.csv', 'fixed_hard.csv'):
            self.assertTrue((self.root/'direction'/name).is_file(), name)
        report = (self.root/'direction'/'report.md').read_text(encoding='utf-8')
        for text in ('EVENT_OFF 基准', 'STATE × OFF', '越过后恶化', 'P=−1', '未重新纳入', '不是新增模型回放'):
            self.assertIn(text, report)

    def test_stale_or_failed_replay_certificate_blocks_new_success(self):
        source = fixture(self.root)
        edit_json(source, 'replay.json', lambda value: value['EVENT'].update(status='FAILED'))
        with self.assertRaisesRegex(ValueError, 'replay did not pass'):
            self.run_source(source)
        self.assertFalse((self.root/'direction'/'summary.json').exists())
        self.assertTrue((self.root/'direction'/'failure.json').is_file())
        preserved = analysis.archive.read_json(self.root/'direction'/'preservation.json')
        self.assertTrue(preserved['source_artifacts_preserved'])

    def test_latest_completed_selection_and_fresh_output_guards(self):
        (self.root/'active.lock').write_bytes(b'0')
        source = fixture(self.root, name='early_event_evaluation_20261015_100000_valid')
        unfinished = self.root/'early_event_evaluation_20261015_110000_failed'
        unfinished.mkdir()
        analysis.archive.write_json(unfinished/'summary.json', {'status': 'FAILED'})
        with redirect_stdout(io.StringIO()):
            result = analysis.main(['--run-dir', str(self.root)])
        self.assertEqual(result['source_evaluation'], str(source))
        self.assertTrue(Path(result['output_dir']).name.startswith('early_event_direction_analysis_'))
        with self.assertRaisesRegex(ValueError, 'already exists'):
            self.run_source(source, Path(result['output_dir']))
        with self.assertRaisesRegex(ValueError, 'not replace'):
            self.run_source(source, source/'inside')
        self.assertEqual((self.root/'active.lock').read_bytes(), b'0')

    def test_source_mutation_during_output_blocks_completion(self):
        source = fixture(self.root)
        writer = analysis.write_tables
        def change_source(output, result):
            writer(output, result)
            (source/'late_change.txt').write_text('changed', encoding='utf-8')
        with patch.object(analysis, 'write_tables', side_effect=change_source):
            with self.assertRaisesRegex(ValueError, 'Source artifacts changed'):
                self.run_source(source)
        self.assertFalse((self.root/'direction'/'summary.json').exists())
        preserved = analysis.archive.read_json(self.root/'direction'/'preservation.json')
        self.assertFalse(preserved['source_artifacts_preserved'])

    def test_entrypoint_stays_numpy_only_and_launcher_has_no_scheduler_or_git(self):
        source = fixture(self.root)
        script = ('import sys; from experiments.chronological.analyze_min_early_event_direction import main; '
                  +'main(["--evaluation-dir", '+repr(str(source))+', "--output-dir", '+repr(str(self.root/'subprocess'))+']); '
                  +'assert "torch" not in sys.modules; print("NUMPY_ONLY_DIRECTION_PASS")')
        process = subprocess.run([sys.executable, '-X', 'utf8', '-c', script], cwd=analysis.REPO,
                                 capture_output=True, text=True, encoding='utf-8', timeout=30)
        self.assertEqual(process.returncode, 0, process.stdout+process.stderr)
        self.assertIn('NUMPY_ONLY_DIRECTION_PASS', process.stdout)
        launcher = (analysis.HERE/'run_min_early_event_direction_analysis.sh').read_text(encoding='utf-8').lower()
        for forbidden in ('sbatch', '#sbatch', '--mem', 'git ', 'gpu', 'cuda'):
            self.assertNotIn(forbidden, launcher)


if __name__ == '__main__':
    unittest.main()
