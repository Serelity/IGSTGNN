"""Launch safety, CUDA routing and failure propagation without an actual cluster."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / 'experiments/chronological/run_temporal_robust_correction.sh'


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.script = self.root / 'experiments/chronological/run_temporal_robust_correction.sh'
        self.script.parent.mkdir(parents=True)
        shutil.copyfile(SCRIPT, self.script)
        self.name = 'contra_v12o_temporal_robust_test'
        self.out = self.root / 'experiments/chronological_runs' / self.name
        fake = self.root / 'bin'; fake.mkdir()
        self.calls = self.root / 'calls'
        (fake / 'python').write_text('''#!/usr/bin/env bash
set -eu
if [[ "$1" == -c && "$2" == *sys.executable* ]]; then
  printf '%s\\n' "$0"
  exit 0
fi
printf '%s\\n' "$*" >> "$FAKE_CALLS"
if [[ "$1" == -m && "${FAIL_TESTS:-0}" == 1 ]]; then exit 19; fi
if [[ " $* " == *" --check "* && "${FAIL_CHECK:-0}" == 1 ]]; then exit 23; fi
exit 0
''')
        (fake / 'python').chmod(0o755)
        (fake / 'git').write_text('#!/usr/bin/env bash\nprintf "commit=synthetic-launcher-test\\n"\n')
        (fake / 'git').chmod(0o755)
        self.env = {**os.environ, 'PATH': str(fake) + ':' + os.environ['PATH'], 'FAKE_CALLS': str(self.calls)}
        self.env.pop('V12O_DEVICE', None)

    def launch(self, *args, **env):
        return subprocess.run(['bash', str(self.script), *args], env={**self.env, **env},
                              capture_output=True, text=True, timeout=20)

    def test_check_then_full_both_route_to_cuda_by_default(self):
        result = self.launch('run', self.name)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        calls = self.calls.read_text().splitlines()
        runs = [s for s in calls if 'train_temporal_robust_correction.py run ' in s]
        self.assertEqual(len(runs), 2)
        self.assertTrue(all('--device cuda:0' in s for s in runs))
        self.assertIn('--check', runs[0].split()); self.assertNotIn('--check', runs[1].split())
        self.assertEqual(Path(str(self.out) + '.job/exit_code').read_text().strip(), '0')

    def test_failure_stops_before_formal_run_and_records_exit(self):
        for variable, status in (('FAIL_TESTS', 19), ('FAIL_CHECK', 23)):
            name = self.name + variable
            result = self.launch('run', name, **{variable: '1'})
            self.assertEqual(result.returncode, status)
            job = self.out.parent / (name + '.job')
            self.assertEqual((job / 'exit_code').read_text().strip(), str(status))
        calls = self.calls.read_text().splitlines()
        self.assertFalse(any('train_temporal_robust_correction.py run ' in s and '--check' not in s.split() for s in calls))

    def test_check_only_stops_after_real_check_command(self):
        result = self.launch('check', self.name)
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.calls.read_text().splitlines()
        self.assertEqual(sum('train_temporal_robust_correction.py run ' in s for s in calls), 1)

    def test_existing_artifacts_and_unsafe_names_refused(self):
        for suffix in ('', '.partial', '.job'):
            target = Path(str(self.out) + suffix)
            target.mkdir(parents=True)
            (target / 'sentinel').write_text('keep')
            result = self.launch('run', self.name)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual((target / 'sentinel').read_text(), 'keep')
            shutil.rmtree(target)
        for name in ('../escape', self.name + '/escape', 'wrong_prefix'):
            self.assertEqual(self.launch('run', name).returncode, 2)
        self.assertFalse(self.calls.exists())
        self.assertNotEqual(self.launch('resume', self.name, self.name).returncode, 0)


if __name__ == '__main__':
    unittest.main()
