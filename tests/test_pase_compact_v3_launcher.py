"""Exercise the long-running Bash orchestration with a harmless executable stub."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


@unittest.skipUnless(shutil.which('bash'), 'Bash is required on the server/CI')
class CompactLauncherTests(unittest.TestCase):
    def test_only_two_new_training_arms_and_auto_resume(self):
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root/'data'
            data.mkdir()
            executable = root/'fake-python'
            executable.write_text('''#!/usr/bin/env bash
set -euo pipefail
printf '%s\\t' "$@" >> "$PASE_FAKE_LOG"
printf '\\n' >> "$PASE_FAKE_LOG"
if [ "$2" = experiments.pase_compact_v3.train ]; then
  while [ "$#" -gt 0 ]; do
    if [ "$1" = --run-dir ]; then mkdir -p "$2"; printf '{}' > "$2/identity.json"; break; fi
    shift
  done
fi
''', encoding='utf-8')
            executable.chmod(0o755)
            log = root/'calls.log'
            environment = dict(os.environ, CONDA_DEFAULT_ENV='igstgnn', PASE_FAKE_LOG=str(log),
                PASE_COMPACT_PYTHON=str(executable), PASE_COMPACT_RUN_ROOT=str(root/'run'),
                PASE_COMPACT_SOURCE_ROOT=str(root/'old'), PASE_COMPACT_DATA_DIR=str(data),
                PASE_V2_ARMS='FULL ORIGIN_EXCHANGE NO_INTERIOR_CENTER', PASE_DATASET='Orange')
            command = ['bash', str(repo/'scripts/run_pase_compact_v3.sh')]
            for attempt in range(2):
                log.write_text('', encoding='utf-8')
                subprocess.run(command, env=environment, cwd=root, check=True, capture_output=True, text=True)
                calls = [line.rstrip('\t').split('\t') for line in log.read_text().splitlines()]
                fitting = [call for call in calls if call[:2] == ['-m','experiments.pase_compact_v3.train']]
                self.assertEqual(len(fitting), 2)
                for call, expected in zip(fitting, (['FULL'], ['UP_ONLY','DOWN_ONLY'])):
                    index = call.index('--arms')+1
                    end = next((i for i in range(index,len(call)) if call[i].startswith('--')),len(call))
                    self.assertEqual(call[index:end], expected)
                    self.assertIn('--full-reference',call)
                self.assertEqual('--resume' in fitting[0], attempt == 1)
                self.assertIn('--resume', fitting[1])
                self.assertIn('--conversion-evaluation', fitting[1])
                evaluation = [call for call in calls if call[:2] == ['-m','experiments.pase_compact_v3.evaluate']]
                self.assertEqual(len(evaluation),2)
                self.assertIn('--conversion-evaluation', evaluation[-1])
                self.assertNotIn('test', evaluation[-1])


if __name__ == '__main__':
    unittest.main()
