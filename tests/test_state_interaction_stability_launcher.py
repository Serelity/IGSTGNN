"""Foreground saved-only stability diagnostic launch and preservation checks."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        scripts = root / 'repo with spaces/experiments/chronological'
        scripts.mkdir(parents=True)
        self.launcher = scripts / 'run_state_interaction_stability_audit.sh'
        shutil.copyfile(Path(__file__).resolve().parents[1] / 'experiments/chronological' /
                        self.launcher.name, self.launcher)
        self.runs = scripts.parent / 'chronological_runs'
        self.runs.mkdir()
        self.default_source = self.create_source('contra_v12g_transfer_audit_01')
        binary = root / 'bin with spaces'
        binary.mkdir()
        python = binary / 'python'
        python.write_text('''#!/usr/bin/env bash
if [[ "$1" == -c ]]; then
  if [[ "$2" == *sys.executable* ]]; then printf '%s\\n' "$0"; fi
  if [[ "$2" == *torch* || "$2" == *cuda* ]]; then exit 90; fi
  if [[ "$2" == *numpy* && "$FAIL_STAGE" == preflight ]]; then exit 8; fi
  exit 0
fi
if [[ "$1" == -m ]]; then
  if [[ "$*" != *test_state_interaction_stability\*.py* ]]; then exit 91; fi
  if [[ "$FAIL_STAGE" == tests ]]; then exit 9; else exit 0; fi
fi
if [[ "$2" == report ]]; then
  printf 'REPORT: %s\\n' "$3"
  exit 0
fi
if [[ "$1" != -u || "$3" != run ]]; then exit 92; fi
OUTPUT=''
SOURCE=''
PREVIOUS=''
for arg in "$@"; do
  case "$arg" in --check|--device|--resume-from|--data-dir|--checkpoint) exit 93 ;; esac
  if [[ "$PREVIOUS" == --output ]]; then OUTPUT=$arg; fi
  if [[ "$PREVIOUS" == --source-dir ]]; then SOURCE=$arg; fi
  PREVIOUS=$arg
done
if [[ "$SOURCE" != "$EXPECT_SOURCE" ]]; then exit 94; fi
printf 'AUDIT_STARTED: %s\\n' "$SOURCE"
if [[ "$FAIL_STAGE" == audit ]]; then exit 23; fi
mkdir -p -- "$OUTPUT"
printf '{}\\n' > "$OUTPUT/summary.json"
exit 0
''')
        python.chmod(0o755)
        git = binary / 'git'
        git.write_text('#!/usr/bin/env bash\nexit 0\n')
        git.chmod(0o755)
        self.env = {**os.environ, 'PATH': str(binary) + os.pathsep + os.environ['PATH'],
                    'FAIL_STAGE': 'success', 'EXPECT_SOURCE': str(self.default_source),
                    'SLURM_JOB_ID': '12345'}

    def create_source(self, name):
        source = self.runs / name
        source.mkdir()
        (source / 'summary.json').write_text('{"preserve": true}')
        for seed in (2025, 2026, 2027):
            (source / f'audit_weekly_s{seed}.npz').write_bytes(f'preserve:{seed}'.encode())
        return source

    def snapshot(self, source):
        return {path.name: path.read_bytes() for path in source.iterdir()}

    def invoke(self, *args, **env):
        return subprocess.run(['bash', str(self.launcher), *args],
                              env={**self.env, **env}, capture_output=True,
                              text=True, timeout=30)

    def test_default_run_source_and_completed_report(self):
        before = self.snapshot(self.default_source)
        result = self.invoke('run')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('Device: CPU (NumPy)', result.stdout)
        name = 'contra_v12h_stability_audit_01'
        job = self.runs / (name + '.job')
        self.assertEqual((job / 'exit_code').read_text().strip(), '0')
        self.assertEqual((job / 'slurm_job_id').read_text().strip(), '12345')
        self.assertTrue((job / 'pid').read_text().strip().isdigit())
        self.assertTrue((job / 'host').read_text().strip())
        self.assertIn('AUDIT_STARTED', (job / 'run.log').read_text())
        self.assertIn('Summary:', self.invoke('status', name).stdout)
        report = self.invoke('report', name)
        self.assertEqual(report.returncode, 0, report.stderr)
        self.assertIn(str(self.runs / name / 'summary.json'), report.stdout)
        self.assertEqual(self.snapshot(self.default_source), before)

    def test_source_override_is_read_only(self):
        source_name = 'contra_v12g_transfer_audit_03'
        source = self.create_source(source_name)
        before = self.snapshot(source)
        result = self.invoke('run', 'contra_v12h_stability_audit_other', source_name,
                             EXPECT_SOURCE=str(source))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(str(source), result.stdout)
        self.assertEqual(self.snapshot(source), before)

    def test_failure_propagation_through_tee_and_existing_job_guard(self):
        for stage, code in (('preflight', 8), ('tests', 9), ('audit', 23), ('success', 0)):
            with self.subTest(stage=stage):
                name = 'contra_v12h_stability_audit_' + stage
                result = self.invoke('run', name, FAIL_STAGE=stage)
                self.assertEqual(result.returncode, code, result.stderr)
                self.assertEqual('AUDIT_STARTED' in result.stdout,
                                 stage in ('audit', 'success'))
                job = self.runs / (name + '.job')
                self.assertEqual((job / 'exit_code').read_text().strip(), str(code))
                log = (job / 'run.log').read_bytes()
                self.assertEqual(self.invoke('run', name).returncode, 1)
                self.assertEqual((job / 'run.log').read_bytes(), log)

    def test_existing_artifacts_are_preserved(self):
        for suffix in ('', '.partial', '.job'):
            with self.subTest(suffix=suffix):
                name = 'contra_v12h_stability_audit_preserve' + str(len(suffix))
                target = self.runs / (name + suffix)
                target.mkdir()
                sentinel = target / 'saved'
                sentinel.write_text('keep')
                result = self.invoke('run', name)
                self.assertEqual(result.returncode, 1)
                self.assertEqual(sentinel.read_text(), 'keep')
                self.assertNotIn('AUDIT_STARTED', result.stdout)
                self.assertEqual(list(target.iterdir()), [sentinel])

    def test_dangling_symlink_artifacts_are_preserved(self):
        for index, suffix in enumerate(('', '.partial', '.job')):
            with self.subTest(suffix=suffix):
                name = 'contra_v12h_stability_audit_symlink' + str(index)
                target = self.runs / (name + suffix)
                destination = self.runs / ('absent' + str(index))
                target.symlink_to(destination)
                result = self.invoke('run', name)
                self.assertEqual(result.returncode, 1)
                self.assertTrue(target.is_symlink())
                self.assertEqual(target.readlink(), destination)
                self.assertNotIn('AUDIT_STARTED', result.stdout)

    def test_invalid_names_missing_source_and_no_resume(self):
        name = 'contra_v12h_stability_audit_new'
        for args in (('run', '../escape'), ('run', name, '../escape'),
                     ('run', name, 'contra_v12f_state_interaction_01'),
                     ('resume', name)):
            with self.subTest(args=args):
                self.assertEqual(self.invoke(*args).returncode, 2)
        result = self.invoke('run', name, 'contra_v12g_transfer_audit_absent')
        self.assertEqual(result.returncode, 1)
        self.assertIn('summary.json', result.stderr)
        self.assertEqual(list(self.runs.iterdir()), [self.default_source])

    def test_missing_weekly_seed_fails_before_job_creation(self):
        source = self.create_source('contra_v12g_transfer_audit_incomplete')
        (source / 'audit_weekly_s2027.npz').unlink()
        before = self.snapshot(source)
        name = 'contra_v12h_stability_audit_missing_seed'
        result = self.invoke('run', name, source.name)
        self.assertEqual(result.returncode, 1)
        self.assertIn('audit_weekly_s2027.npz', result.stderr)
        self.assertFalse((self.runs / (name + '.job')).exists())
        self.assertEqual(self.snapshot(source), before)

    def test_missing_summary_reports_incomplete_without_python_traceback(self):
        name = 'contra_v12h_stability_audit_interrupted'
        result = self.invoke('run', name, FAIL_STAGE='audit')
        self.assertEqual(result.returncode, 23)
        partial = self.runs / (name + '.partial')
        partial.mkdir()
        status = self.invoke('status', name)
        self.assertEqual(status.returncode, 0, status.stderr)
        self.assertIn('Workflow exit code: 23', status.stdout)
        self.assertIn(str(partial), status.stdout)
        report = self.invoke('report', name)
        self.assertEqual(report.returncode, 1)
        self.assertIn('status', report.stderr)
        self.assertNotIn('Traceback', report.stderr)


if __name__ == '__main__':
    unittest.main()
