"""Saved-statistic, paired inference and failure-boundary tests; no torch required."""

from contextlib import redirect_stdout
import copy
import csv
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experiments.chronological import audit_architecture_regions as audit


def fixture(root):
    protocol = copy.deepcopy(audit.load_protocol(audit.PROTOCOL))
    protocol['cohort_samples'] = dict.fromkeys(protocol['cohort_samples'], 4)
    protocol['node_count'] = 2
    protocol['bootstrap']['draws'] = 80
    # Noncontiguous identities, chronological full order and different matched order.
    ids = [41, 7, 93, 12]
    matched = [93, 41, 12, 7]
    times = {sample: f'2023-01-{2 + 7 * i:02d}T12:00:00' for i, sample in enumerate(ids)}
    dirs = {name: root / name for name in ('data', 'primary', 'secondary', 'v12a')}
    for directory in dirs.values():
        directory.mkdir()

    def csv_file(path, rows):
        with path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    csv_file(dirs['data'] / 'train_manifest.csv', [
        {'sample_index': sample, 'split': 'train', 'incident_id': str(sample), 't0': times[sample]}
        for sample in ids])
    for directory, filename, order in (
        ('primary', 'train_control_manifest.csv', ids),
        ('secondary', 'train_second_control_manifest.csv', matched),
    ):
        csv_file(dirs[directory] / filename, [
            {'positive_sample_index': sample, 'control_index': i, 'split': 'train',
             'incident_id': str(sample), 'positive_t0': times[sample]}
            for i, sample in enumerate(order)])
    audit.write_json(dirs['data'] / 'summary.json', {'files': {
        'train_manifest.csv': audit.sha256(dirs['data'] / 'train_manifest.csv')}})
    baseline = {
        'positive_package': {'summary_sha256': audit.sha256(dirs['data'] / 'summary.json')},
        'primary_control_inputs': {'train_control_manifest.csv': audit.sha256(dirs['primary'] / 'train_control_manifest.csv')},
        'secondary_control_inputs': {'train_second_control_manifest.csv': audit.sha256(dirs['secondary'] / 'train_second_control_manifest.csv')},
    }
    audit.write_json(root / 'incident_branch_materialize_v6a.json', baseline)
    protocol['baseline_protocol_sha256'] = audit.sha256(root / 'incident_branch_materialize_v6a.json')
    summary = {
        'status': 'ARCHITECTURE_MECHANISM_AUDIT_COMPLETE',
        'protocol_id': 'contra_v8_architecture_mechanism_audit_v12a',
        'protocol_sha256': protocol['v12a_protocol_sha256'],
        'engineering_check': False, 'full_cohort_evaluated': True,
        'checkpoint_state_unchanged': True, 'model_training_performed': False,
        'real_data_gradient_computation_performed': False,
        'validation_arrays_read': False, 'test_split_read': False,
        'checkpoint': {'best_model_sha256': protocol['checkpoint_sha256']},
        'code_sha256': {'experiments/chronological/audit_architecture_mechanisms.py': protocol['v12a_auditor_sha256']},
        'results': {}, 'outputs': {},
    }
    regions = protocol['base_regions'] + [f'all_h{h}' for h in range(1, 13)]
    masks = np.zeros((18, 12, 2), dtype=bool)
    masks[0] = True
    masks[1, :3, 0] = True
    masks[2, 3:6, 0] = True
    masks[3, 6:, 0] = True
    masks[4, :6, 1] = True
    masks[5, 6:, 1] = True
    for h in range(12):
        masks[6 + h, h, :] = True
    valid = np.ones((4, 12, 2), dtype=bool)
    valid[0, 1:, :] = False  # Unequal cell weighting is intentional.
    predicted = np.broadcast_to(masks.sum((1, 2)), (4, 18)).copy()
    counts = (valid[:, None] & masks[None]).sum((2, 3))
    # Norm-minus-ICSF benefit varies by event; identical across matched cohorts.
    errors = np.broadcast_to(counts[:, None, :] * 10., (4, 6, 18)).copy()
    errors[:, 2, :] -= counts * np.asarray([1., 2., 3., 4.])[:, None]
    for cohort in protocol['cohort_samples']:
        order = ids if cohort == 'incident_full' else matched
        selection = np.asarray([ids.index(sample) for sample in order])
        source = list(range(4)) if cohort in ('incident_full', 'secondary_control') else selection
        values = {'paths': np.asarray(protocol['paths']), 'regions': np.asarray(regions),
                  'positive_sample_index': np.asarray(order), 'source_index': np.asarray(source),
                  'absolute_error_sums': errors[selection], 'valid_counts': counts[selection],
                  'prediction_counts': predicted[selection]}
        filename = f'train_{cohort}_mechanisms.npz'
        np.savez_compressed(dirs['v12a'] / filename, **values)
        summary['outputs'][filename] = {'samples': 4, 'sha256': audit.sha256(dirs['v12a'] / filename)}
        summary['results'][cohort] = {'samples': 4, 'regions': {
            region: {'valid_target_cells': int(counts[:, r].sum()),
                     'prediction_cells': int(predicted[:, r].sum()),
                     'descriptive_mae': {name: float(errors[:, p, r].sum() / counts[:, r].sum())
                                         for p, name in enumerate(protocol['paths'])}}
            for r, region in enumerate(regions)}}
    audit.write_json(dirs['v12a'] / 'summary.json', summary)
    return protocol, dirs, times


class RegionAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.protocol, self.dirs, self.times = fixture(self.root)

    def load(self):
        with patch.object(audit, 'PROTOCOL', self.root / 'protocol.json'):
            return audit.load_inputs(*self.dirs.values(), self.protocol)

    def test_frozen_protocol_and_rejection_of_check_or_validation_source(self):
        summary = json.loads((self.dirs['v12a'] / 'summary.json').read_text())
        audit.validate_source_summary(summary, self.protocol)
        for key, value in (('engineering_check', True), ('validation_arrays_read', True),
                           ('test_split_read', True), ('checkpoint_state_unchanged', False)):
            changed = copy.deepcopy(summary)
            changed[key] = value
            with self.assertRaises(ValueError):
                audit.validate_source_summary(changed, self.protocol)
        changed_protocol = self.root / 'changed.json'
        audit.write_json(changed_protocol, self.protocol)
        with self.assertRaisesRegex(ValueError, 'protocol changed'):
            audit.load_protocol(changed_protocol)

    def test_saved_only_inputs_noncontiguous_identities_and_region_accounting(self):
        records, times, hashes = self.load()
        self.assertEqual(records['incident']['ids'].tolist(), [93, 41, 12, 7])
        self.assertEqual(records['incident']['source_indices'].tolist(), [2, 0, 3, 1])
        self.assertEqual(times, self.times)
        self.assertTrue(all(not path.endswith('.npy') for path in hashes))
        record = records['incident_full']
        for field in ('counts', 'prediction_counts', 'errors'):
            v = record[field]
            np.testing.assert_array_equal(v[..., -3], v[..., 1] + v[..., 2])
            np.testing.assert_array_equal(v[..., -2] + v[..., -1], v[..., 0])

    def test_unequal_cell_and_equal_event_estimands_and_paired_cancellation(self):
        records, times, _ = self.load()
        result, _, arrays = audit.analyze(records, times, self.protocol)
        effect = result['results']['incident_full']['regions']['all']['comparisons']['icsf_given_normalization']
        self.assertAlmostEqual(effect['gain_raw_mae'], (2 + 24 * (2 + 3 + 4)) / 74)
        self.assertAlmostEqual(effect['equal_event_gain_raw_mae'], 2.5)
        paired = result['matched_benefit_difference']['all']['icsf_given_normalization']
        self.assertEqual(paired['incident_minus_mean_control_gain'], 0.)
        for method in ('week', 'four_week_block'):
            self.assertEqual(paired['intervals'][method]['pooled']['ci_low'], 0.)
            self.assertEqual(paired['intervals'][method]['pooled']['ci_high'], 0.)
        regional = result['results']['incident_full']['regions']
        total = sum(regional[name]['comparisons']['icsf_given_normalization']['regional_gain_in_global_mae_units']
                    for name in self.protocol['base_regions'][1:])
        self.assertAlmostEqual(total, effect['gain_raw_mae'])
        np.testing.assert_array_equal(arrays['bootstrap_week_weights'].sum(1), 4)

    def test_bootstrap_wraparound_empty_weeks_and_reproducibility(self):
        # One four-week block on four weeks must include each week exactly once.
        np.testing.assert_array_equal(audit.bootstrap_weights(4, 50, 5, 4), np.ones((50, 4)))
        weights = audit.bootstrap_weights(7, 50, 5, 4)
        np.testing.assert_array_equal(weights.sum(1), 7)
        np.testing.assert_array_equal(weights, audit.bootstrap_weights(7, 50, 5, 4))
        weeks, positions = audit.week_grid({1: '2022-12-26', 2: '2023-01-23'})
        self.assertEqual(weeks, ['2022-W52', '2023-W01', '2023-W02', '2023-W03', '2023-W04'])
        self.assertEqual(positions, {1: 0, 2: 4})

    def test_empty_regions_remain_undefined_and_serialize_without_nan(self):
        records, times, _ = self.load()
        for record in records.values():
            record['counts'][:, -3] = 0
            record['errors'][:, :, -3] = 0
        result, _, _ = audit.analyze(records, times, self.protocol)
        effect = result['results']['incident_full']['regions']['candidate_h1_h6']['comparisons']['normalization']
        self.assertIsNone(effect['gain_raw_mae'])
        self.assertEqual(effect['intervals']['week']['pooled']['status'], 'INSUFFICIENT_VALID_DRAWS')
        json.dumps(result, allow_nan=False)

    def test_corrupted_hash_rejected_before_analysis(self):
        path = self.dirs['v12a'] / 'train_incident_mechanisms.npz'
        with path.open('ab') as stream:
            stream.write(b'changed')
        with self.assertRaisesRegex(ValueError, 'fingerprint'):
            self.load()

    def test_artifact_identity_partitions_and_mae_mismatch_rejected(self):
        path = self.dirs['v12a'] / 'train_incident_full_mechanisms.npz'
        with np.load(path) as source:
            original = {key: source[key].copy() for key in source.files}
        result = json.loads((self.dirs['v12a'] / 'summary.json').read_text())['results']['incident_full']
        for key in ('positive_sample_index', 'valid_counts', 'absolute_error_sums'):
            values = {k: v.copy() for k, v in original.items()}
            values[key].flat[0] += 1
            np.savez(path, **values)
            with self.assertRaises(ValueError):
                audit.load_artifact(path, [41, 7, 93, 12], range(4), self.protocol, result)
        np.savez(path, **original)
        result['regions']['all']['descriptive_mae']['off'] += 1
        with self.assertRaisesRegex(ValueError, 'MAE disagree'):
            audit.load_artifact(path, [41, 7, 93, 12], range(4), self.protocol, result)

    def test_end_to_end_atomic_publication_report_and_refuse_overwrite(self):
        protocol_path = self.root / 'protocol.json'
        audit.write_json(protocol_path, self.protocol)
        out = self.root / 'completed'
        with patch.object(audit, 'PROTOCOL', protocol_path), patch.object(
                audit, 'PROTOCOL_SHA256', audit.sha256(protocol_path)), redirect_stdout(io.StringIO()):
            summary = audit.run(*self.dirs.values(), out, protocol_path)
            with self.assertRaises(FileExistsError):
                audit.run(*self.dirs.values(), out, protocol_path)
        self.assertTrue((out / 'summary.json').is_file())
        self.assertFalse(out.with_name('completed.partial').exists())
        self.assertFalse(summary['model_loaded'])
        self.assertFalse(summary['test_split_read'])
        for name, digest in summary['outputs'].items():
            self.assertEqual(audit.sha256(out / name), digest)
        with (out / 'regional_comparisons.csv').open() as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 4 * 21 * 7)

    def test_failure_preserves_diagnostic_without_final_output(self):
        out = self.root / 'failed'
        with patch.object(audit, 'load_inputs', side_effect=ValueError('bad source')), redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, 'bad source'):
                audit.run(*self.dirs.values(), out)
        self.assertFalse(out.exists())
        failure = json.loads((self.root / 'failed.partial' / 'failure.json').read_text())
        self.assertEqual(failure['error'], 'bad source')
        with self.assertRaises(FileExistsError):
            audit.run(*self.dirs.values(), out)

    def test_import_and_cli_do_not_load_torch(self):
        import sys
        code = 'from experiments.chronological import audit_architecture_regions; import sys; assert "torch" not in sys.modules'
        subprocess.run([sys.executable, '-c', code], cwd=audit.REPO, check=True)

    def test_launcher_stops_after_failed_tests_and_records_exit_status(self):
        scripts = self.root / 'repo' / 'experiments' / 'chronological'
        scripts.mkdir(parents=True)
        launcher = scripts / 'run_architecture_region_audit.sh'
        shutil.copyfile(audit.REPO / 'experiments/chronological' / launcher.name, launcher)
        job = scripts.parent / 'chronological_runs' / 'contra_v12b_regions_test.job'
        job.mkdir(parents=True)
        fake_bin = self.root / 'bin'
        fake_bin.mkdir()
        fake_git = fake_bin / 'git'
        fake_git.write_text('#!/bin/bash\nexit 0\n')
        fake_git.chmod(0o755)
        fake_python = fake_bin / 'python with spaces'
        fake_python.write_text(
            '#!/bin/bash\n'
            'if [[ "$1" == "-c" ]]; then exit 0; fi\n'
            'if [[ "$1" == "-m" ]]; then exit 9; fi\n'
            'echo "ANALYSIS_MUST_NOT_RUN"\nexit 77\n')
        fake_python.chmod(0o755)
        result = subprocess.run(['bash', str(launcher), '_worker', 'contra_v12b_regions_test',
                                 str(fake_python), 'source'], capture_output=True, text=True,
                                env={**os.environ, 'PATH': str(fake_bin) + os.pathsep + os.environ['PATH']})
        self.assertEqual(result.returncode, 9, result.stderr)
        self.assertEqual((job / 'exit_code').read_text().strip(), '9')
        self.assertNotIn('ANALYSIS_MUST_NOT_RUN', result.stdout)
        invalid = subprocess.run(['bash', str(launcher), 'start', '../escape'], capture_output=True)
        self.assertEqual(invalid.returncode, 2)


if __name__ == '__main__':
    unittest.main()
