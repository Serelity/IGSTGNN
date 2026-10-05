"""Server handoff behavior without a GPU, external messages or campus access."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'experiments/chronological/run_effective_input_audit.sh'


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.script = self.root / 'experiments/chronological/run_effective_input_audit.sh'
        self.script.parent.mkdir(parents=True)
        shutil.copyfile(SCRIPT, self.script)
        binary = self.root / 'bin'
        binary.mkdir()
        (binary / 'python').write_text('''#!/usr/bin/env bash
set -eu
if [[ "$1" == -c && "$2" == *sys.executable* ]]; then printf '%s\\n' "$0"; exit 0; fi
printf '%s\\n' "$*" >> "$FAKE_CALLS"
if [[ "$1" == -m && "${FAIL_TESTS:-0}" == 1 ]]; then exit 19; fi
if [[ " $* " == *" --check "* && "${FAIL_CHECK:-0}" == 1 ]]; then exit 23; fi
exit 0
''')
        (binary / 'git').write_text('#!/usr/bin/env bash\nprintf "commit=fixture\\n"\n')
        for path in binary.iterdir(): path.chmod(0o755)
        self.calls = self.root / 'calls'
        self.env = {**os.environ, 'PATH': str(binary) + ':' + os.environ['PATH'], 'FAKE_CALLS': str(self.calls)}
        self.env.pop('V13A_DEVICE', None)
        self.name = 'contra_v13a_inputs_fixture'
        self.out = self.root / 'experiments/chronological_runs' / self.name

    def launch(self, action, name=None, **env):
        return subprocess.run(['bash', str(self.script), action, name or self.name],
                              capture_output=True, text=True, env={**self.env, **env}, timeout=20)

    def test_default_cuda_check_precedes_full_and_exit_is_recorded(self):
        result = self.launch('run')
        self.assertEqual(result.returncode, 0, result.stderr)
        runs = [line for line in self.calls.read_text().splitlines() if 'audit_effective_inputs.py run' in line]
        self.assertEqual(len(runs), 2)
        self.assertTrue(all('--device cuda:0' in line and '--checkpoint' in line for line in runs))
        self.assertIn('--check', runs[0].split())
        self.assertNotIn('--check', runs[1].split())
        self.assertEqual(Path(str(self.out) + '.job/exit_code').read_text().strip(), '0')

    def test_failed_tests_or_check_never_launch_full_audit(self):
        for flag, code in [('FAIL_TESTS', 19), ('FAIL_CHECK', 23)]:
            name = self.name + flag
            result = self.launch('run', name, **{flag: '1'})
            self.assertEqual(result.returncode, code)
            self.assertEqual((self.out.parent / (name + '.job/exit_code')).read_text().strip(), str(code))
        self.assertFalse(any('audit_effective_inputs.py run' in line and '--check' not in line.split()
                             for line in self.calls.read_text().splitlines()))

    def test_inventory_is_cpu_without_checkpoint_and_check_is_not_full(self):
        for mode in ('inventory', 'check'):
            result = self.launch(mode, self.name + mode)
            self.assertEqual(result.returncode, 0, result.stderr)
        runs = [line for line in self.calls.read_text().splitlines() if 'audit_effective_inputs.py run' in line]
        self.assertEqual(len(runs), 2)
        self.assertIn('--inventory-only', runs[0])
        self.assertIn('--device cpu', runs[0])
        self.assertNotIn('--checkpoint', runs[0])
        self.assertIn('--check', runs[1].split())

    def test_existing_artifact_or_unsafe_name_is_refused(self):
        for suffix in ('', '.partial', '.job'):
            path = Path(str(self.out) + suffix)
            path.mkdir(parents=True)
            self.assertNotEqual(self.launch('run').returncode, 0)
            path.rmdir()
        self.assertEqual(self.launch('run', '../escape').returncode, 2)
        self.assertEqual(self.launch('unknown').returncode, 2)
        self.assertFalse(self.calls.exists())


if __name__ == '__main__':
    unittest.main()
