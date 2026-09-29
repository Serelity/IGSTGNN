"""The platform must see failure, and checks must precede the full diagnostic."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class LauncherTests(unittest.TestCase):
    def test_checks_failure_propagation_and_no_overwrite(self):
        for stage, code in (('tests', 9), ('check', 17), ('full', 23), ('success', 0)):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                scripts = root / 'repo/experiments/chronological'
                scripts.mkdir(parents=True)
                launcher = scripts / 'run_icsf_strength_response.sh'
                shutil.copyfile(Path(__file__).resolve().parents[1] / 'experiments/chronological' / launcher.name, launcher)
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
for arg in "$@"; do
  if [[ "$arg" == --check ]]; then
    if [[ "$FAIL_STAGE" == check ]]; then exit 17; else exit 0; fi
  fi
done
echo FULL_DIAGNOSTIC_STARTED
if [[ "$FAIL_STAGE" == full ]]; then exit 23; fi
exit 0
''')
                python.chmod(0o755)
                git = binary / 'git'
                git.write_text('#!/usr/bin/env bash\nexit 0\n')
                git.chmod(0o755)
                env = {**os.environ, 'PATH': str(binary) + os.pathsep + os.environ['PATH'],
                       'V12D_DEVICE': 'cpu', 'FAIL_STAGE': stage}
                name = 'contra_v12d_strength_response_test'
                args = ['bash', str(launcher), 'run', name]
                result = subprocess.run(args, env=env, capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, code, result.stderr)
                self.assertEqual('FULL_DIAGNOSTIC_STARTED' in result.stdout, stage in ('full', 'success'))
                job = scripts.parent / 'chronological_runs' / (name + '.job')
                self.assertEqual((job / 'exit_code').read_text().strip(), str(code))
                log = (job / 'run.log').read_bytes()
                retry = subprocess.run(args, env=env, capture_output=True, timeout=30)
                self.assertNotEqual(retry.returncode, 0)
                self.assertEqual((job / 'run.log').read_bytes(), log)


if __name__ == '__main__':
    unittest.main()
