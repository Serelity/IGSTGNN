"""Exercise the campus handoff without contacting a server or using a GPU."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'experiments/chronological/run_minimal_information.sh'


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.script = self.root / 'experiments/chronological/run_minimal_information.sh'
        self.script.parent.mkdir(parents=True)
        shutil.copyfile(SCRIPT, self.script)
        binary = self.root / 'bin'
        binary.mkdir()
        (binary / 'python').write_text('''#!/usr/bin/env bash
set -eu
if [[ "$1" == -c && "$2" == *sys.executable* ]]; then printf '%s\\n' "$0"; exit 0; fi
printf '%s\\n' "$*" >> "$FAKE_CALLS"
if [[ "$1" == -m && "${FAIL_TESTS:-0}" == 1 ]]; then exit 19; fi
if [[ " $* " == *" --mode check "* && "${FAIL_CHECK:-0}" == 1 ]]; then exit 23; fi
if [[ " $* " == *" --mode run "* && "${FAIL_RUN:-0}" == 1 ]]; then exit 27; fi
exit 0
''')
        (binary / 'git').write_text('#!/usr/bin/env bash\nprintf "commit=fixture\\n"\n')
        for path in binary.iterdir():
            path.chmod(0o755)
        self.calls = self.root / 'calls'
        self.env = {**os.environ, 'PATH': str(binary) + ':' + os.environ['PATH'], 'FAKE_CALLS': str(self.calls)}
        self.env.pop('V13B_DEVICE', None)
        self.name = 'contra_v13b_information_fixture'
        self.out = self.root / 'experiments/chronological_runs' / self.name

    def launch(self, action, name=None, source=None, **env):
        command = ['bash', str(self.script), action, name or self.name]
        if source:
            command.append(source)
        return subprocess.run(command, capture_output=True, text=True, env={**self.env, **env}, timeout=20)

    def runs(self):
        return [line for line in self.calls.read_text().splitlines() if 'train_minimal_information.py run' in line]

    def test_cuda_training_routes_check_before_pilot_or_full_no_checkpoint(self):
        for mode in ('pilot', 'run'):
            result = self.launch(mode, self.name + mode)
            self.assertEqual(result.returncode, 0, result.stderr)
        runs = self.runs()
        self.assertEqual(len(runs), 4)
        self.assertIn('--mode check', runs[0])
        self.assertIn('--mode pilot', runs[1])
        self.assertIn('--mode check', runs[2])
        self.assertIn('--mode run', runs[3])
        self.assertTrue(all('--device cuda:0' in line and '--checkpoint' not in line for line in runs))

    def test_failures_propagate_and_do_not_start_dependent_work(self):
        for flag, code in (('FAIL_TESTS', 19), ('FAIL_CHECK', 23), ('FAIL_RUN', 27)):
            name = self.name + flag
            result = self.launch('run', name, **{flag: '1'})
            self.assertEqual(result.returncode, code)
            self.assertEqual((self.out.parent / (name + '.job/exit_code')).read_text().strip(), str(code))
        self.assertEqual(sum('--mode run' in line for line in self.runs()), 1)

    def test_preflight_cpu_and_check_are_bounded(self):
        for mode in ('preflight', 'check'):
            result = self.launch(mode, self.name + mode)
            self.assertEqual(result.returncode, 0, result.stderr)
        runs = self.runs()
        self.assertEqual(len(runs), 2)
        self.assertIn('--mode preflight', runs[0])
        self.assertIn('--device cpu', runs[0])
        self.assertIn('--mode check', runs[1])

    def test_resume_requires_readonly_source_and_new_destination(self):
        source_name = self.name + 'source'
        source = self.out.parent / (source_name + '.partial')
        self.assertNotEqual(self.launch('resume', source=source_name).returncode, 0)
        source.mkdir(parents=True)
        (source / 'run_identity.json').write_text('{}')
        result = self.launch('resume', source=source_name)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--resume-from ' + str(source), self.runs()[-1])
        self.assertEqual((source / 'run_identity.json').read_text(), '{}')

    def test_output_reuse_unsafe_names_and_incomplete_report_rejected(self):
        for suffix in ('', '.partial', '.job'):
            path = Path(str(self.out) + suffix)
            path.mkdir(parents=True)
            self.assertNotEqual(self.launch('run').returncode, 0)
            path.rmdir()
        self.assertEqual(self.launch('run', '../escape').returncode, 2)
        self.assertEqual(self.launch('unknown').returncode, 2)
        self.assertNotEqual(self.launch('report').returncode, 0)
        self.assertFalse(self.calls.exists())


if __name__ == '__main__':
    unittest.main()
