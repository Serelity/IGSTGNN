"""Exercise the platform entry point without a GPU or research data."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class ForegroundLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        scripts = root / 'repo/experiments/chronological'
        scripts.mkdir(parents=True)
        self.launcher = scripts / 'run_incident_strength_gate.sh'
        source = Path(__file__).resolve().parents[1] / 'experiments/chronological' / self.launcher.name
        shutil.copyfile(source, self.launcher)
        self.name = 'contra_v12c_strength_gate_restart_test'
        self.job = scripts.parent / 'chronological_runs' / (self.name + '.job')
        fake_bin = root / 'bin with spaces'
        fake_bin.mkdir()
        python = fake_bin / 'python'
        python.write_text('''#!/usr/bin/env bash
if [[ "$1" == -c ]]; then
  if [[ "$2" == *sys.executable* ]]; then printf '%s\\n' "$0"; fi
  exit 0
fi
if [[ "$1" == -m ]]; then echo TESTS_PASSED; exit 0; fi
for arg in "$@"; do
  if [[ "$arg" == --resume-from ]]; then exit 88; fi
  if [[ "$arg" == --check ]]; then echo ENGINEERING_CHECK; exit "${CHECK_EXIT:-0}"; fi
done
echo FULL_TRAINING
exit "${TRAIN_EXIT:-0}"
''')
        python.chmod(0o755)
        git = fake_bin / 'git'
        git.write_text('#!/usr/bin/env bash\nexit 0\n')
        git.chmod(0o755)
        self.env = {**os.environ, 'PATH': str(fake_bin) + os.pathsep + os.environ['PATH'],
                    'V12C_DEVICE': 'cpu', 'SLURM_JOB_ID': '12345', 'SLURM_CPUS_PER_TASK': '3',
                    'CHECK_EXIT': '0', 'TRAIN_EXIT': '0'}

    def run_launcher(self, **env):
        return subprocess.run(['bash', str(self.launcher), 'run', self.name],
                              env={**self.env, **env}, capture_output=True, text=True, timeout=30)

    def test_foreground_waits_records_resources_and_rejects_overwrite(self):
        result = self.run_launcher()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('FULL_TRAINING', result.stdout)
        self.assertEqual((self.job / 'exit_code').read_text().strip(), '0')
        self.assertEqual((self.job / 'slurm_job_id').read_text().strip(), '12345')
        for name in ('resources_started.txt', 'resources_finished.txt'):
            self.assertIn('SLURM_JOB_ID=12345', (self.job / name).read_text())
        log = (self.job / 'run.log').read_bytes()
        self.assertNotEqual(self.run_launcher().returncode, 0)
        self.assertEqual((self.job / 'run.log').read_bytes(), log)

    def test_training_failure_reaches_platform_through_tee(self):
        result = self.run_launcher(TRAIN_EXIT='23')
        self.assertEqual(result.returncode, 23, result.stderr)
        self.assertEqual((self.job / 'exit_code').read_text().strip(), '23')

    def test_engineering_failure_prevents_full_training(self):
        result = self.run_launcher(CHECK_EXIT='17')
        self.assertEqual(result.returncode, 17, result.stderr)
        self.assertNotIn('FULL_TRAINING', result.stdout)
        self.assertEqual((self.job / 'exit_code').read_text().strip(), '17')


if __name__ == '__main__':
    unittest.main()
