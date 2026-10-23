"""Run the real Bash launcher with a stub, proving no baseline/CC retraining."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


@unittest.skipUnless(shutil.which('bash'), 'Linux Bash required')
class MainLauncherTests(unittest.TestCase):
    def test_three_datasets_full_only_cc_reuse_and_auto_resume(self):
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for dataset in ('Contra_Costa', 'Alameda', 'Orange'):
                (root/'data'/dataset).mkdir(parents=True)
            (root/'cc'/'s2025').mkdir(parents=True)
            (root/'cc'/'s2025'/'identity.json').write_text('{}')
            exe = root/'fake-python'
            exe.write_text('''#!/usr/bin/env bash
set -euo pipefail
printf '%s\\t' "$@" >> "$PASE_FAKE_LOG"
printf '\\n' >> "$PASE_FAKE_LOG"
if [ "$2" = experiments.pase_compact_main.train ]; then
  while [ "$#" -gt 0 ]; do
    if [ "$1" = --run-dir ]; then mkdir -p "$2"; printf '{}' > "$2/identity.json"; break; fi
    shift
  done
fi
''')
            exe.chmod(0o755)
            log = root/'calls'
            env = dict(os.environ, CONDA_DEFAULT_ENV='igstgnn', PASE_FAKE_LOG=str(log),
                PASE_MAIN_PYTHON=str(exe), PASE_MAIN_DATA_ROOT=str(root/'data'),
                PASE_MAIN_CC_ROOT=str(root/'cc'), PASE_MAIN_RUN_ROOT=str(root/'runs'))
            command = ['bash', str(repo/'scripts/run_pase_compact_main.sh')]
            for attempt in range(2):
                log.write_text('')
                subprocess.run(command, env=env, cwd=root, check=True, capture_output=True, text=True)
                calls = [line.rstrip('\t').split('\t') for line in log.read_text().splitlines()]
                training = [x for x in calls if x[:2] == ['-m','experiments.pase_compact_main.train']]
                self.assertEqual(len(training), 2)
                self.assertEqual([x[x.index('--dataset')+1] for x in training], ['Alameda', 'Orange'])
                for call in training:
                    self.assertEqual(call[call.index('--arms')+1], 'FULL')
                    self.assertEqual('--resume' in call, attempt == 1)
                self.assertFalse(any('experiments.pase_compact_v3.train' in x for x in calls))
                evaluations = [x for x in calls if len(x)>1 and x[1].endswith('.evaluate')]
                self.assertEqual(len(evaluations), 5)
                self.assertEqual(sum(x[x.index('--split')+1] == 'test' for x in evaluations), 3)
                for call in evaluations:
                    self.assertEqual(call[call.index('--arms')+1], 'FULL')


if __name__ == '__main__':
    unittest.main()
