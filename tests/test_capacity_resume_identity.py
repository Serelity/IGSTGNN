"""Exact identity diagnostics must preserve frozen sources and original artifacts."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.chronological import continue_incident_capacity as continuation
from experiments.chronological import resume_incident_capacity_checked as checked
from experiments.chronological import train_incident_capacity as training
from src.utils.incident_corridor import read_json, sha256, write_json


class CapacityResumeIdentityTests(unittest.TestCase):
    def setUp(self):
        scratch = training.REPO/'experiments/chronological_runs'
        scratch.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.saved = dict(seed=11, common_backbone_sha256='original',
                          environment=dict(gpu='Tesla V100-SXM2-32GB', python='3.10.21'))
        write_json(self.root/'identity.json', self.saved)
        (self.root/'last_checkpoint.pt').write_bytes(b'original checkpoint marker')
        self.hashes = {p.name: sha256(p) for p in self.root.iterdir()}
        self.report = self.root/'diagnostic.json'
        self.arguments = ['--identity-report', str(self.report), '--resume', '--output-dir', str(self.root)]

    def fake_main(self, current, calls):
        def run(argv):
            identity = current
            training.require(self.saved == identity, 'Run identity changed')
            calls.append('training reached')
        return run

    def assert_preserved(self):
        self.assertEqual({name: sha256(self.root/name) for name in self.hashes}, self.hashes)

    def test_reports_nested_gpu_version_initialization_and_missing_fields(self):
        current = dict(self.saved, common_backbone_sha256='different',
                       environment=dict(gpu='different GPU', python='3.10.21', cuda='12.1'))
        rows = checked.identity_differences(self.saved, current)
        self.assertEqual({r['field'] for r in rows},
                         {'common_backbone_sha256', 'environment.gpu', 'environment.cuda'})
        self.assertEqual(checked.identity_differences(self.saved, self.saved), [])

    def test_normal_mismatch_is_still_rejected_before_training_and_preserves_files(self):
        calls = []
        current = dict(self.saved, environment=dict(gpu='other', python='3.10.21'))
        old_require = training.require
        with patch.object(training, 'main', self.fake_main(current, calls)):
            with self.assertRaisesRegex(ValueError, 'Run identity changed'):
                checked.main(self.arguments)
        self.assertEqual(calls, [])
        report = read_json(self.report)
        self.assertFalse(report['matching'])
        self.assertEqual(report['differences'][0]['field'], 'environment.gpu')
        self.assertIs(training.require, old_require)
        self.assert_preserved()

    def test_normal_matching_identity_continues_through_original_check(self):
        calls = []
        with patch.object(training, 'main', self.fake_main(self.saved, calls)):
            checked.main(self.arguments)
        self.assertEqual(calls, ['training reached'])
        self.assertTrue(read_json(self.report)['matching'])
        self.assert_preserved()

    def test_audit_mode_stops_at_identity_gate_for_match_and_mismatch(self):
        for changed in (False, True):
            with self.subTest(changed=changed):
                current = dict(self.saved, common_backbone_sha256='different') if changed else self.saved
                calls = []
                report = self.root/f'audit_{changed}.json'
                arguments = ['--identity-report', str(report), '--identity-audit-only',
                             '--resume', '--output-dir', str(self.root)]
                with patch.object(training, 'main', self.fake_main(current, calls)):
                    checked.main(arguments)
                self.assertEqual(calls, [])
                self.assertEqual(read_json(report)['matching'], not changed)
                self.assertTrue(read_json(report)['audit_only'])
                self.assert_preserved()

    def test_audit_guard_rejects_training_if_expected_gate_is_not_reached(self):
        original = training.train_arm
        def unexpected_main(argv):
            training.train_arm()
        with patch.object(training, 'main', unexpected_main):
            with self.assertRaisesRegex(ValueError, 'before training'):
                checked.main(self.arguments+['--identity-audit-only'])
        self.assertIs(training.train_arm, original)
        self.assert_preserved()

    def test_existing_report_and_missing_resume_cannot_overwrite_any_artifact(self):
        with self.assertRaisesRegex(ValueError, 'explicit --resume'):
            checked.main([arg for arg in self.arguments if arg != '--resume'])
        arguments = ['--identity-report', str(self.root/'identity.json'), '--resume', '--output-dir', str(self.root)]
        with self.assertRaisesRegex(ValueError, 'report exists'):
            checked.main(arguments)
        self.assert_preserved()

    def test_parent_command_passes_diagnostic_mode_and_original_check_scope(self):
        directories = {key: self.root/key for key in ('data', 'history', 'network', 'reports')}
        identity = dict(seed=11, check=True, device='cuda:0', ramp_exchanges=True)
        command = continuation.trainer_command(self.root, identity, directories, self.root/'sensors', 10,
                                               identity_report=self.report, identity_audit_only=True)
        self.assertIn('--identity-audit-only', command)
        self.assertIn('--check', command)
        self.assertIn('--resume', command)
        self.assertEqual(command[command.index('--identity-report')+1], str(self.report))
        self.assertEqual(command[command.index('--epochs')+1], '10')


if __name__ == '__main__':
    unittest.main()
