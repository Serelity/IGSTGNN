"""Foreground launcher failures, endpoint recovery and preservation of existing runs."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        scripts = root / 'repo with spaces/experiments/chronological'
        scripts.mkdir(parents=True)
        self.launcher = scripts / 'run_vector_objective_alignment.sh'
        shutil.copyfile(Path(__file__).resolve().parents[1] / 'experiments/chronological' / self.launcher.name, self.launcher)
        self.runs = scripts.parent / 'chronological_runs'
        self.runs.mkdir()
        binary = root / 'bin with spaces'
        binary.mkdir()
        python = binary / 'python'
        python.write_text('''#!/usr/bin/env bash
if [[ "$1" == -c ]]; then
  if [[ "$2" == *sys.executable* ]]; then printf '%s\\n' "$0"; exit 0; fi
  if [[ "$FAIL_STAGE" == device ]]; then exit 13; fi
  exit 0
fi
if [[ "$1" == -m ]]; then
  if [[ "$FAIL_STAGE" == tests ]]; then exit 9; else exit 0; fi
fi
if [[ "$2" == report ]]; then printf 'REPORT: %s\\n' "$3"; exit 0; fi
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
  if [[ "$HAS_RESUME" == 1 || "$OUTPUT" != *.job/check ]]; then exit 88; fi
  if [[ "$FAIL_STAGE" == check ]]; then exit 17; else exit 0; fi
fi
if [[ "$HAS_RESUME" != "$EXPECT_RESUME" ]]; then exit 89; fi
echo FULL_FIT_STARTED
if [[ "$FAIL_STAGE" == full ]]; then exit 23; fi
mkdir -p -- "$OUTPUT"
printf '{}\\n' > "$OUTPUT/summary.json"
''')
        python.chmod(0o755)
        git = binary / 'git'
        git.write_text('#!/usr/bin/env bash\nexit 0\n')
        git.chmod(0o755)
        self.env = {**os.environ, 'PATH': str(binary) + os.pathsep + os.environ['PATH'],
                    'V12K_DEVICE': 'cpu', 'FAIL_STAGE': 'success', 'EXPECT_RESUME': '0'}

    def invoke(self, *args, **env):
        return subprocess.run(['bash', str(self.launcher), *args], env={**self.env, **env},
                              capture_output=True, text=True, timeout=30)

    def test_all_failure_stages_record_exit_and_never_continue(self):
        for stage, code in (('device', 13), ('tests', 9), ('check', 17), ('full', 23), ('success', 0)):
            with self.subTest(stage=stage):
                name = 'contra_v12k_objective_alignment_' + stage
                result = self.invoke('run', name, FAIL_STAGE=stage)
                self.assertEqual(result.returncode, code, result.stderr)
                self.assertEqual('FULL_FIT_STARTED' in result.stdout, stage in ('full', 'success'))
                job = self.runs / (name + '.job')
                self.assertEqual((job / 'exit_code').read_text().strip(), str(code))
                before = (job / 'run.log').read_bytes()
                self.assertEqual(self.invoke('run', name).returncode, 1)
                self.assertEqual((job / 'run.log').read_bytes(), before)

    def test_resume_has_fresh_engineering_check_and_preserves_source(self):
        source_name = 'contra_v12k_objective_alignment_old'
        source = self.runs / (source_name + '.partial')
        source.mkdir()
        (source / 'run_identity.json').write_text('{"preserve": true}')
        result = self.invoke('resume', 'contra_v12k_objective_alignment_new', source_name, EXPECT_RESUME='1')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((source / 'run_identity.json').read_text(), '{"preserve": true}')
        self.assertEqual(len(list(source.iterdir())), 1)

    def test_existing_artifacts_and_dangling_links_are_preserved(self):
        for index, suffix in enumerate(('', '.partial', '.job')):
            for link in (False, True):
                name = f'contra_v12k_objective_alignment_saved_{index}_{link}'
                target = self.runs / (name + suffix)
                if link:
                    target.symlink_to(self.runs / 'absent', target_is_directory=True)
                else:
                    target.mkdir()
                result = self.invoke('run', name)
                self.assertEqual(result.returncode, 1)
                self.assertTrue(target.is_symlink() if link else target.is_dir())

    def test_invalid_names_and_sources_do_not_create_runs(self):
        name = 'contra_v12k_objective_alignment_new'
        for args in (('run', '../escape'), ('resume', name, name), ('resume', name), ('resume', name, '../escape')):
            self.assertEqual(self.invoke(*args).returncode, 2)
        self.assertEqual(self.invoke('resume', name, 'contra_v12k_objective_alignment_absent').returncode, 1)
        self.assertEqual(list(self.runs.iterdir()), [])

    def test_symlinked_recovery_source_is_rejected(self):
        real = self.runs / 'real'
        real.mkdir()
        (real / 'run_identity.json').write_text('{}')
        source_name = 'contra_v12k_objective_alignment_source'
        (self.runs / (source_name + '.partial')).symlink_to(real, target_is_directory=True)
        result = self.invoke('resume', 'contra_v12k_objective_alignment_new', source_name)
        self.assertEqual(result.returncode, 1)

    def test_report_status_and_missing_summary(self):
        name = 'contra_v12k_objective_alignment_done'
        result = self.invoke('run', name)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('twelve fits', result.stdout)
        self.assertIn('24 endpoints', result.stdout)
        self.assertIn('Summary:', self.invoke('status', name).stdout)
        self.assertIn(str(self.runs / name / 'summary.json'), self.invoke('report', name).stdout)
        other = 'contra_v12k_objective_alignment_failed'
        self.invoke('run', other, FAIL_STAGE='full')
        result = self.invoke('report', other)
        self.assertEqual(result.returncode, 1)
        self.assertNotIn('Traceback', result.stderr)
        self.assertIn('Workflow exit code: 23', self.invoke('status', other).stdout)


if __name__ == '__main__':
    unittest.main()
