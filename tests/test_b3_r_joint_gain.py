"""Independent scalar checks for offline region x OFF bias x horizon analysis."""
import csv
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

from experiments.chronological import analyze_b3_r_joint_gain as joint
from tests.test_b3_r_gain_analysis import main_arrays, small_arrays, source_fixture, source_hashes, read_json, write_json


REPO = Path(__file__).resolve().parents[1]


def scalar_joint_oracle(arrays, region, bias):
    """Classify and sum each valid observation without the implementation's masks."""
    shape = arrays['on'].shape
    counts, totals = [0]*12, [0]*12
    values = {key: [0.]*12 for key in ('on', 'off', 'positive', 'negative', 'signed', 'change')}
    for b in range(shape[0]):
        for h in range(shape[1]):
            for node in range(shape[2]):
                if not bool(arrays['valid'][b, h, node, 0]):
                    continue
                totals[h] += 1
                target = float(arrays['target'][b, h, node, 0])
                on = float(arrays['on'][b, h, node, 0])
                off = float(arrays['off'][b, h, node, 0])
                observed_region = ('direct_report_nodes' if bool(arrays['report_associated'][b, node])
                                   else 'potential_propagation_only_nodes' if bool(arrays['candidate_potential'][b, node])
                                   else 'remaining_nodes')
                observed_bias = 'over' if off > target else 'under' if off < target else 'exact'
                if observed_region != region or observed_bias != bias:
                    continue
                counts[h] += 1
                error_on, error_off = abs(on-target), abs(off-target)
                values['on'][h] += error_on
                values['off'][h] += error_off
                values['positive'][h] += max(error_off-error_on, 0.)
                values['negative'][h] += max(error_on-error_off, 0.)
                values['signed'][h] += off-on
                values['change'][h] += abs(off-on)
    own = lambda key: [v/n if n else None for v, n in zip(values[key], counts)]
    macro = lambda key: sum(own(key))/12 if all(counts) else None
    global_h = lambda key: [v/n for v, n in zip(values[key], totals)]
    result = dict(
        region_metric_available=all(counts), valid_count=sum(counts), per_horizon_valid_count=counts,
        on_mae_macro=macro('on'), off_mae_macro=macro('off'),
        gain_positive=macro('positive'), gain_negative=macro('negative'),
        prediction_signed_change_horizon_macro_mean=macro('signed'),
        prediction_abs_change_horizon_macro_mean=macro('change'),
        per_horizon_on_mae=own('on'), per_horizon_off_mae=own('off'),
        per_horizon_gain_positive=own('positive'), per_horizon_gain_negative=own('negative'),
        per_horizon_prediction_signed_change_mean=own('signed'),
        per_horizon_prediction_abs_change_mean=own('change'),
        per_horizon_global_gain_positive=global_h('positive'), per_horizon_global_gain_negative=global_h('negative'),
        per_horizon_global_prediction_signed_change_contribution=global_h('signed'),
        per_horizon_global_prediction_abs_change_contribution=global_h('change'),
        per_horizon_global_valid_weight_fraction=[n/total for n, total in zip(counts, totals)])
    result['per_horizon_off_minus_on'] = [b-a if a is not None else None
                                         for a, b in zip(own('on'), own('off'))]
    result['off_minus_on_mae'] = (result['off_mae_macro']-result['on_mae_macro']) if all(counts) else None
    result['per_horizon_global_gain_contribution'] = [p-n for p, n in zip(global_h('positive'), global_h('negative'))]
    for short, long_name in (
        ('gain_contribution', 'gain_contribution'), ('gain_positive', 'gain_positive'), ('gain_negative', 'gain_negative'),
        ('valid_weight_fraction', 'valid_weight_fraction'),
        ('prediction_signed_change_contribution', 'prediction_signed_change_contribution'),
        ('prediction_abs_change_contribution', 'prediction_abs_change_contribution')):
        result['global_'+short] = sum(result['per_horizon_global_'+long_name])/12.
    return result


class JointGainTests(unittest.TestCase):
    def assert_oracle(self, metric, expected):
        for field, value in expected.items():
            if isinstance(value, list):
                self.assertEqual(len(metric[field]), len(value))
                for i, item in enumerate(value):
                    if item is None:
                        self.assertIsNone(metric[field][i], (field, i))
                    else:
                        self.assertAlmostEqual(metric[field][i], item, places=12, msg=(field, i))
            elif value is None:
                self.assertIsNone(metric[field], field)
            elif isinstance(value, bool):
                self.assertEqual(metric[field], value, field)
            else:
                self.assertAlmostEqual(metric[field], value, places=12, msg=field)

    def test_all_nine_cells_match_scalar_classification_with_unequal_horizon_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arrays = source_fixture(root/'source')
            result = joint.analyze(root/'source', root/'analysis', allow_engineering=True)
            self.assertEqual(result['status'], 'B3R_JOINT_GAIN_ANALYSIS_PASS')
            self.assertEqual(result['metrics']['all_nodes']['per_horizon_valid_count'], [12, 4, 6]*4)
            for region in joint.REGIONS:
                for bias in joint.BIASES:
                    with self.subTest(region=region, bias=bias):
                        self.assert_oracle(result['joint_metrics'][region][bias], scalar_joint_oracle(arrays, region, bias))
            for field in joint.ADDITIVE_FIELDS:
                self.assertAlmostEqual(sum(m[field] for row in result['joint_metrics'].values() for m in row.values()),
                                       result['metrics']['all_nodes'][field], places=12)
            for field in joint.HORIZON_ADDITIVE_FIELDS:
                np.testing.assert_allclose(np.sum([m[field] for row in result['joint_metrics'].values()
                                                  for m in row.values()], axis=0),
                                           result['metrics']['all_nodes'][field], atol=1e-12, rtol=0.)
            self.assertTrue(all(result['partition_checks'].values()))

    def test_csv_has_117_rows_and_horizon_contribution_to_macro_is_divided_by_twelve(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_fixture(root/'source')
            result = joint.analyze(root/'source', root/'analysis', allow_engineering=True)
            with (root/'analysis'/'joint_decomposition.csv').open(encoding='utf-8', newline='') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 117)
            self.assertEqual(sum(row['horizon'] == 'macro' for row in rows), 9)
            for row in rows:
                metric = result['joint_metrics'][row['region']][row['off_residual_bias']]
                if row['horizon'] == 'macro':
                    self.assertEqual(row['global_horizon_gain_contribution'], '')
                    self.assertAlmostEqual(float(row['contribution_to_global_macro_mae']), metric['global_gain_contribution'])
                    continue
                h = int(row['horizon'])-1
                for column, field in (
                    ('contribution_to_global_macro_mae', 'per_horizon_global_gain_contribution'),
                    ('positive_contribution_to_global_macro_mae', 'per_horizon_global_gain_positive'),
                    ('negative_contribution_to_global_macro_mae', 'per_horizon_global_gain_negative'),
                    ('signed_shift_contribution_to_global_macro', 'per_horizon_global_prediction_signed_change_contribution'),
                    ('abs_change_contribution_to_global_macro', 'per_horizon_global_prediction_abs_change_contribution')):
                    self.assertAlmostEqual(float(row[column]), metric[field][h]/12., places=12)
            for horizon in range(1, 13):
                total = sum(float(row['contribution_to_global_macro_mae']) for row in rows if row['horizon'] == str(horizon))
                self.assertAlmostEqual(total, result['metrics']['all_nodes']['per_horizon_global_gain_contribution'][horizon-1]/12., places=12)
            self.assertAlmostEqual(sum(float(row['contribution_to_global_macro_mae']) for row in rows
                                       if row['horizon'] != 'macro'),
                                   result['metrics']['all_nodes']['off_minus_on_mae'], places=12)

    def test_partial_horizon_cell_has_unavailable_macro_without_losing_global_contribution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arrays = main_arrays()
            arrays['report_associated'][:] = False
            arrays['report_associated'][2, 1] = True
            arrays['candidate_potential'][:] = False
            arrays['candidate_potential'][2, :2] = True
            source_fixture(root/'source', arrays)
            result = joint.analyze(root/'source', root/'analysis', allow_engineering=True)
            cell = result['joint_metrics']['direct_report_nodes']['under']
            self.assertEqual(cell['per_horizon_valid_count'], [1, 0, 0]*4)
            self.assertIsNone(cell['off_minus_on_mae'])
            self.assertIsNone(cell['prediction_signed_change_horizon_macro_mean'])
            self.assertGreater(cell['global_gain_contribution'], 0.)
            self.assert_oracle(cell, scalar_joint_oracle(arrays, 'direct_report_nodes', 'under'))

    def test_empty_joint_cells_are_none_locally_but_zero_for_every_global_contribution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_fixture(root/'source', small_arrays([-2.], [2.]))
            result = joint.analyze(root/'source', root/'analysis', allow_engineering=True)
            for region in ('direct_report_nodes', 'potential_propagation_only_nodes'):
                for metric in result['joint_metrics'][region].values():
                    self.assertEqual(metric['valid_count'], 0)
                    self.assertIsNone(metric['on_mae_macro'])
                    self.assertIsNone(metric['gain_positive'])
                    self.assertIsNone(metric['prediction_signed_change_horizon_macro_mean'])
                    for field in joint.ADDITIVE_FIELDS:
                        self.assertEqual(metric[field], 0., field)
                    for field in joint.HORIZON_ADDITIVE_FIELDS:
                        self.assertEqual(metric[field], [0.]*12, field)
            populated = result['joint_metrics']['remaining_nodes']['over']
            self.assertEqual(populated['off_minus_on_mae'], 0.)
            self.assertEqual(populated['prediction_abs_change_horizon_macro_mean'], 4.)
            self.assertIsNone(populated['cancellation_fraction'])

    def test_joint_cell_retains_positive_negative_cancellation_and_exact_off_bias(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_fixture(root/'source', small_arrays([1., 3.], [2., 2.]))
            result = joint.analyze(root/'source', root/'analysis', allow_engineering=True)
            cell = result['joint_metrics']['remaining_nodes']['over']
            self.assertEqual(cell['gain_positive'], .5)
            self.assertEqual(cell['gain_negative'], .5)
            self.assertEqual(cell['off_minus_on_mae'], 0.)
            self.assertEqual(cell['cancellation_fraction'], 1.)
            self.assertEqual(cell['prediction_signed_change_horizon_macro_mean'], 0.)
            source_fixture(root/'exact_source', small_arrays([2., -2.], [0., 0.], [True, False]))
            exact = joint.analyze(root/'exact_source', root/'exact_analysis', allow_engineering=True)
            for region in ('direct_report_nodes', 'remaining_nodes'):
                cell = exact['joint_metrics'][region]['exact']
                self.assertEqual(cell['off_minus_on_mae'], -2.)
                self.assertEqual(cell['global_gain_negative'], 1.)

    def test_scope_failed_probe_and_inconsistent_source_metrics_reject_without_pass_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for case in ('engineering', 'failed', 'metrics'):
                source = root/case
                source_fixture(source)
                if case != 'engineering':
                    doc = read_json(source/'summary.json')
                    if case == 'failed':
                        doc['status'] = 'B3R_PATH_PROBE_FAILED'
                    else:
                        doc['metrics']['all_nodes']['off_minus_on_mae'] = 100.
                    write_json(source/'summary.json', doc)
                before = source_hashes(source)
                with self.subTest(case=case), self.assertRaises(ValueError):
                    joint.analyze(source, root/(case+'_output'), allow_engineering=case != 'engineering')
                self.assertEqual(before, source_hashes(source))
                self.assertFalse((root/(case+'_output')/'analysis.json').exists())

    def test_success_preserves_every_source_file_and_records_snapshots(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_fixture(root/'source')
            (root/'source'/'extra_audit.txt').write_text('protected\n', encoding='utf-8')
            before = source_hashes(root/'source')
            result = joint.analyze(root/'source', root/'analysis', allow_engineering=True)
            self.assertEqual(before, source_hashes(root/'source'))
            self.assertEqual(result['source_sha256_before'], result['source_sha256_after'])
            self.assertIn(str(root/'source'/'extra_audit.txt'), result['source_sha256_before'])
            self.assertEqual(read_json(root/'analysis'/'analysis.json'), result)
            for name in ('README.md', 'joint_decomposition.csv', 'preservation.json'):
                self.assertTrue((root/'analysis'/name).is_file())
            self.assertEqual(result['new_inference_calls'], 0)
            self.assertFalse(result['test_accessed'])
            self.assertFalse(result['architecture_selected'])

    def test_source_change_during_analysis_blocks_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_fixture(root/'source')
            original = joint.base.write_csv
            def changed_write(path, rows):
                original(path, rows)
                (root/'source'/'concurrent.txt').write_text('changed', encoding='utf-8')
            with mock.patch.object(joint.base, 'write_csv', side_effect=changed_write):
                with self.assertRaisesRegex(ValueError, 'Source artifacts changed'):
                    joint.analyze(root/'source', root/'analysis', allow_engineering=True)
            self.assertFalse((root/'analysis'/'analysis.json').exists())
            self.assertEqual(read_json(root/'analysis'/'failure.json')['status'], 'B3R_JOINT_GAIN_ANALYSIS_FAILED')

    def test_cli_runs_without_torch_and_preserves_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_fixture(root/'source')
            before = source_hashes(root/'source')
            script = REPO/'experiments'/'chronological'/'analyze_b3_r_joint_gain.py'
            code = (
                "import builtins, runpy, sys\n"
                "from pathlib import Path\n"
                "original = builtins.__import__\n"
                "def guarded(name, *args, **kwargs):\n"
                "    if name == 'torch' or name.startswith('torch.'):\n"
                "        raise AssertionError('Offline joint analysis attempted to import Torch')\n"
                "    return original(name, *args, **kwargs)\n"
                "builtins.__import__ = guarded\n"
                "script = sys.argv[1]\n"
                "sys.path.insert(0, str(Path(script).parent))\n"
                "sys.argv = [script] + sys.argv[2:]\n"
                "runpy.run_path(script, run_name='__main__')\n")
            completed = subprocess.run([sys.executable, '-X', 'utf8', '-c', code, str(script),
                '--probe-dir', str(root/'source'), '--output-dir', str(root/'analysis'), '--allow-engineering'],
                cwd=REPO, capture_output=True, text=True, encoding='utf-8', timeout=60)
            self.assertEqual(completed.returncode, 0, completed.stdout+'\n'+completed.stderr)
            result = read_json(root/'analysis'/'analysis.json')
            self.assertEqual(result['status'], 'B3R_JOINT_GAIN_ANALYSIS_PASS')
            self.assertEqual(result['csv_rows'], 117)
            self.assertEqual(before, source_hashes(root/'source'))


if __name__ == '__main__':
    unittest.main()
