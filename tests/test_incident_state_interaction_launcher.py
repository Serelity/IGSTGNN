"""Launcher failure boundaries, fresh recovery, and explicit incomplete reports."""

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
        self.launcher = scripts / 'run_incident_state_interaction.sh'
        shutil.copyfile(Path(__file__).resolve().parents[1] / 'experiments/chronological' /
                        self.launcher.name, self.launcher)
        self.runs = scripts.parent / 'chronological_runs'
        self.runs.mkdir()
        binary = root / 'bin with spaces'
        binary.mkdir()
        python = binary / 'python'
        python.write_text('''#!/usr/bin/env bash
if [[ "$1" == -c ]]; then
  if [[ "$2" == *sys.executable* ]]; then printf '%s\\n' "$0"; fi
  exit 0
fi
if [[ "$1" == -m ]]; then
  if [[ "$FAIL_STAGE" == tests ]]; then exit 9; else exit 0; fi
fi
if [[ "$2" == report ]]; then
  printf 'REPORT: %s\\n' "$3"
  exit 0
fi
IS_CHECK=0
HAS_RESUME=0
HAS_CPU=0
OUTPUT=''
PREVIOUS=''
for arg in "$@"; do
  if [[ "$arg" == --check ]]; then IS_CHECK=1; fi
  if [[ "$arg" == --resume-from ]]; then HAS_RESUME=1; fi
  if [[ "$PREVIOUS" == --device && "$arg" == cpu ]]; then HAS_CPU=1; fi
  if [[ "$PREVIOUS" == --output ]]; then OUTPUT=$arg; fi
  PREVIOUS=$arg
done
if [[ "$HAS_CPU" != 1 ]]; then exit 90; fi
if [[ "$IS_CHECK" == 1 ]]; then
  if [[ "$HAS_RESUME" == 1 ]]; then exit 88; fi
  if [[ "$OUTPUT" != *.job/check ]]; then exit 91; fi
  if [[ "$FAIL_STAGE" == check ]]; then exit 17; else exit 0; fi
fi
if [[ "$HAS_RESUME" != "$EXPECT_RESUME" ]]; then exit 89; fi
echo FULL_FIT_STARTED
if [[ "$FAIL_STAGE" == full ]]; then exit 23; fi
mkdir -p -- "$OUTPUT"
printf '{}\\n' > "$OUTPUT/summary.json"
exit 0
''')
        python.chmod(0o755)
        git = binary / 'git'
        git.write_text('#!/usr/bin/env bash\nexit 0\n')
        git.chmod(0o755)
        self.env = {**os.environ, 'PATH': str(binary) + os.pathsep + os.environ['PATH'],
                    'V12F_DEVICE': 'cpu', 'FAIL_STAGE': 'success', 'EXPECT_RESUME': '0'}

    def invoke(self, *args, **env):
        return subprocess.run(['bash', str(self.launcher), *args],
                              env={**self.env, **env}, capture_output=True,
                              text=True, timeout=30)

    def test_failure_propagation_and_resume_never_enters_engineering_check(self):
        source_name = 'contra_v12f_state_interaction_old'
        source = self.runs / (source_name + '.partial')
        source.mkdir()
        identity = source / 'run_identity.json'
        identity.write_text('{"preserve": true}')
        for stage, code in (('tests', 9), ('check', 17), ('full', 23), ('success', 0)):
            with self.subTest(stage=stage):
                name = 'contra_v12f_state_interaction_' + stage
                args = ('resume', name, source_name)
                result = self.invoke(*args, FAIL_STAGE=stage, EXPECT_RESUME='1')
                self.assertEqual(result.returncode, code, result.stderr)
                self.assertEqual('FULL_FIT_STARTED' in result.stdout,
                                 stage in ('full', 'success'))
                job = self.runs / (name + '.job')
                self.assertEqual((job / 'exit_code').read_text().strip(), str(code))
                log = (job / 'run.log').read_bytes()
                self.assertNotEqual(self.invoke(*args).returncode, 0)
                self.assertEqual((job / 'run.log').read_bytes(), log)
                self.assertEqual(identity.read_text(), '{"preserve": true}')
                self.assertEqual(list(source.iterdir()), [identity])

    def test_fresh_run_and_completed_report(self):
        name = 'contra_v12f_state_interaction_fresh'
        result = self.invoke('run', name)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('Device: cpu', result.stdout)
        self.assertIn('nine fits', result.stdout)
        self.assertIn('Summary:', self.invoke('status', name).stdout)
        report = self.invoke('report', name)
        self.assertEqual(report.returncode, 0, report.stderr)
        self.assertIn(str(self.runs / name / 'summary.json'), report.stdout)

    def test_existing_artifacts_are_preserved(self):
        for suffix in ('', '.partial', '.job'):
            with self.subTest(suffix=suffix):
                name = 'contra_v12f_state_interaction_preserve' + str(len(suffix))
                target = self.runs / (name + suffix)
                target.mkdir()
                sentinel = target / 'saved'
                sentinel.write_text('keep')
                result = self.invoke('run', name)
                self.assertEqual(result.returncode, 1)
                self.assertEqual(sentinel.read_text(), 'keep')
                self.assertNotIn('FULL_FIT_STARTED', result.stdout)
                self.assertEqual(list(target.iterdir()), [sentinel])

    def test_invalid_names_and_missing_recovery_identity(self):
        name = 'contra_v12f_state_interaction_new'
        for args in (('run', '../escape'), ('resume', name, name),
                     ('resume', name, '../escape'), ('resume', name)):
            with self.subTest(args=args):
                self.assertEqual(self.invoke(*args).returncode, 2)
        result = self.invoke('resume', name, 'contra_v12f_state_interaction_absent')
        self.assertEqual(result.returncode, 1)
        self.assertIn('run_identity.json', result.stderr)
        self.assertEqual(list(self.runs.iterdir()), [])

    def test_missing_summary_reports_incomplete_without_python_traceback(self):
        name = 'contra_v12f_state_interaction_interrupted'
        result = self.invoke('run', name, FAIL_STAGE='full')
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
