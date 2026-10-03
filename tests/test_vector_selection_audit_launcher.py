"""Launcher source checks, foreground exit propagation and artifact preservation."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[1]


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        scripts = self.root / 'experiments/chronological'
        scripts.mkdir(parents=True)
        self.runs = self.root / 'experiments/chronological_runs'
        self.runs.mkdir()
        self.launcher = scripts / 'run_vector_selection_audit.sh'
        shutil.copyfile(REPO / 'experiments/chronological/run_vector_selection_audit.sh', self.launcher)
        self.source = self.runs / 'contra_v12k_objective_alignment_01'
        self.source.mkdir()
        for name in ('summary.json', 'run_identity.json', 'selected_endpoints_frozen.json'):
            (self.source / name).write_text('{}')
        for arm in ('state_vector', 'interaction_vector'):
            for loss in ('global', 'candidate_early'):
                for seed in (2025, 2026, 2027):
                    directory = self.source / f'{arm}__loss_{loss}_s{seed}'
                    directory.mkdir()
                    for name in ('history.json', 'fit_summary.json'):
                        (directory / name).write_text('{}')
        binary = self.root / 'bin with spaces'
        binary.mkdir()
        python = binary / 'python'
        python.write_text('''#!/usr/bin/env bash
if [[ "$1" == -c ]]; then printf '%s\\n' "$0"; exit 0; fi
if [[ "$1" == -m ]]; then
  if [[ "$FAIL_STAGE" == tests ]]; then exit 9; else exit 0; fi
fi
if [[ "$2" == report ]]; then echo REPORT; exit 0; fi
echo AUDIT_STARTED
if [[ "$FAIL_STAGE" == audit ]]; then exit 23; fi
OUTPUT=''
PREVIOUS=''
for arg in "$@"; do
  if [[ "$PREVIOUS" == --output ]]; then OUTPUT=$arg; fi
  PREVIOUS=$arg
done
mkdir -p -- "$OUTPUT"
printf '{}\\n' > "$OUTPUT/summary.json"
''')
        python.chmod(0o755)
        git = binary / 'git'
        git.write_text('#!/usr/bin/env bash\nexit 0\n')
        git.chmod(0o755)
        self.env = {**os.environ, 'PATH': str(binary) + os.pathsep + os.environ['PATH'], 'FAIL_STAGE': 'success'}

    def invoke(self, *args, **env):
        return subprocess.run(['bash', str(self.launcher), *args], env={**self.env, **env},
                              capture_output=True, text=True, timeout=30)

    def test_preflight_and_analysis_exit_codes_are_preserved(self):
        for stage, code in (('tests', 9), ('audit', 23), ('success', 0)):
            name = 'contra_v12l_selection_audit_' + stage
            result = self.invoke('run', name, FAIL_STAGE=stage)
            self.assertEqual(result.returncode, code, result.stderr)
            self.assertEqual('AUDIT_STARTED' in result.stdout, stage != 'tests')
            job = self.runs / (name + '.job')
            self.assertEqual((job / 'exit_code').read_text().strip(), str(code))
            log = (job / 'run.log').read_bytes()
            self.assertEqual(self.invoke('run', name).returncode, 1)
            self.assertEqual((job / 'run.log').read_bytes(), log)

    def test_missing_invalid_or_symlinked_source_does_not_start(self):
        name = 'contra_v12l_selection_audit_new'
        for args, code in ((('run', '../escape'), 2), (('run', name, '../escape'), 2),
                           (('run', name, self.source.name, 'extra'), 2),
                           (('run', name, 'contra_v12k_objective_alignment_absent'), 1)):
            self.assertEqual(self.invoke(*args).returncode, code)
        path = self.source / 'state_vector__loss_global_s2025/history.json'
        path.unlink()
        self.assertEqual(self.invoke('run', name).returncode, 1)
        path.symlink_to(self.source / 'summary.json')
        self.assertEqual(self.invoke('run', name).returncode, 1)
        self.assertFalse((self.runs / (name + '.job')).exists())

    def test_existing_and_dangling_output_paths_are_preserved(self):
        for index, suffix in enumerate(('', '.partial', '.job')):
            name = f'contra_v12l_selection_audit_old{index}'
            path = self.runs / (name + suffix)
            path.symlink_to(self.root / 'absent')
            self.assertEqual(self.invoke('run', name).returncode, 1)
            self.assertTrue(path.is_symlink())

    def test_status_and_report_distinguish_complete_and_failed_runs(self):
        name = 'contra_v12l_selection_audit_complete'
        self.assertEqual(self.invoke('run', name).returncode, 0)
        self.assertIn('Summary:', self.invoke('status', name).stdout)
        self.assertIn('REPORT', self.invoke('report', name).stdout)
        failed = 'contra_v12l_selection_audit_failed'
        self.invoke('run', failed, FAIL_STAGE='audit')
        self.assertIn('Workflow exit code: 23', self.invoke('status', failed).stdout)
        result = self.invoke('report', failed)
        self.assertEqual(result.returncode, 1)
        self.assertNotIn('Traceback', result.stderr)


if __name__ == '__main__':
    unittest.main()
