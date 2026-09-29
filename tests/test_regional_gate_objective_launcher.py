"""Foreground failure codes and engineering/full separation for recovery."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class LauncherTests(unittest.TestCase):
    def test_failure_propagation_and_resume_never_enters_engineering_check(self):
        for stage, code in (('tests', 9), ('check', 17), ('full', 23), ('success', 0)):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                scripts = root / 'repo/experiments/chronological'
                scripts.mkdir(parents=True)
                launcher = scripts / 'run_regional_gate_objective.sh'
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
IS_CHECK=0
HAS_RESUME=0
for arg in "$@"; do
  if [[ "$arg" == --check ]]; then IS_CHECK=1; fi
  if [[ "$arg" == --resume-from ]]; then HAS_RESUME=1; fi
done
if [[ "$IS_CHECK" == 1 ]]; then
  if [[ "$HAS_RESUME" == 1 ]]; then exit 88; fi
  if [[ "$FAIL_STAGE" == check ]]; then exit 17; else exit 0; fi
fi
if [[ "$HAS_RESUME" != 1 ]]; then exit 89; fi
echo FULL_FIT_STARTED
if [[ "$FAIL_STAGE" == full ]]; then exit 23; fi
exit 0
''')
                python.chmod(0o755)
                git = binary / 'git'
                git.write_text('#!/usr/bin/env bash\nexit 0\n')
                git.chmod(0o755)
                source_name = 'contra_v12e_regional_gate_old'
                source = scripts.parent / 'chronological_runs' / (source_name + '.partial')
                source.mkdir(parents=True)
                (source / 'run_identity.json').write_text('{}')
                env = {**os.environ, 'PATH': str(binary) + os.pathsep + os.environ['PATH'],
                       'V12E_DEVICE': 'cpu', 'FAIL_STAGE': stage}
                name = 'contra_v12e_regional_gate_new'
                args = ['bash', str(launcher), 'resume', name, source_name]
                result = subprocess.run(args, env=env, capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, code, result.stderr)
                self.assertEqual('FULL_FIT_STARTED' in result.stdout, stage in ('full', 'success'))
                job = scripts.parent / 'chronological_runs' / (name + '.job')
                self.assertEqual((job / 'exit_code').read_text().strip(), str(code))
                log = (job / 'run.log').read_bytes()
                self.assertNotEqual(subprocess.run(args, env=env, capture_output=True, timeout=30).returncode, 0)
                self.assertEqual((job / 'run.log').read_bytes(), log)


if __name__ == '__main__':
    unittest.main()
