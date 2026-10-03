"""v12m foreground startup, exact argument routing and recovery preservation."""

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
        self.root = Path(self.temp.name)
        scripts = self.root / 'repo with spaces/experiments/chronological'
        scripts.mkdir(parents=True)
        self.launcher = scripts / 'run_vector_output_scope.sh'
        shutil.copyfile(Path(__file__).resolve().parents[1] / 'experiments/chronological' / self.launcher.name,
                        self.launcher)
        self.runs = scripts.parent / 'chronological_runs'
        self.runs.mkdir()
        self.trace = self.root / 'stages.log'
        binary = self.root / 'bin with spaces'
        binary.mkdir()
        python = binary / 'python'
        python.write_text('''#!/usr/bin/env bash
set -euo pipefail
if [[ "$1" == -c ]]; then
  if [[ "$2" == *sys.executable* ]]; then printf '%s\\n' "$0"; exit 0; fi
  printf 'device\\n' >> "$STAGE_TRACE"
  if [[ "$3" != "$EXPECTED_DEVICE" ]]; then exit 90; fi
  if [[ "$FAIL_STAGE" == device ]]; then exit 13; fi
  exit 0
fi
if [[ "$1" == -m ]]; then
  if [[ "$*" != '-m unittest discover -s tests -p test_vector_output_scope*.py -v' ]]; then exit 91; fi
  printf 'tests\\n' >> "$STAGE_TRACE"
  if [[ "$FAIL_STAGE" == tests ]]; then exit 9; else exit 0; fi
fi
if [[ "$2" == report ]]; then
  if [[ "$1" != */train_vector_output_scope.py ]]; then exit 92; fi
  printf 'REPORT: %s\\n' "$3"
  exit 0
fi
if [[ "$1" != -u || "$2" != */train_vector_output_scope.py || "$3" != run ]]; then exit 93; fi
shift 3
IS_CHECK=0
OUTPUT=''
SOURCE=''
DEVICE=''
DATA=''
PRIMARY=''
SECONDARY=''
CHECKPOINT=''
while [[ $# -gt 0 ]]; do
  case "$1" in
    --check) IS_CHECK=1; shift ;;
    --output) OUTPUT=$2; shift 2 ;;
    --resume-from) SOURCE=$2; shift 2 ;;
    --device) DEVICE=$2; shift 2 ;;
    --data-dir) DATA=$2; shift 2 ;;
    --primary-control-dir) PRIMARY=$2; shift 2 ;;
    --secondary-control-dir) SECONDARY=$2; shift 2 ;;
    --checkpoint) CHECKPOINT=$2; shift 2 ;;
    *) exit 94 ;;
  esac
done
if [[ "$DEVICE" != "$EXPECTED_DEVICE" || "$DATA" != ../data/chronological/Contra_Costa_v8_dev ||
      "$PRIMARY" != ../research_artifacts/v6_inputs_20260920/v3_materialized_01 ||
      "$SECONDARY" != ../research_artifacts/v6_inputs_20260920/v5b_second_materialized_01 ||
      "$CHECKPOINT" != experiments/chronological_runs/contra_fixed_s2025_full_01/best_model.pt ]]; then exit 95; fi
if [[ "$IS_CHECK" == 1 ]]; then
  if [[ -n "$SOURCE" || "$OUTPUT" != *.job/check ]]; then exit 88; fi
  printf 'check\\n' >> "$STAGE_TRACE"
  if [[ "$FAIL_STAGE" == check ]]; then exit 17; else exit 0; fi
fi
if [[ "$SOURCE" != "$EXPECTED_SOURCE" ]]; then exit 89; fi
printf 'full\\n' >> "$STAGE_TRACE"
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
                    'V12M_DEVICE': 'cpu', 'EXPECTED_DEVICE': 'cpu', 'FAIL_STAGE': 'success',
                    'EXPECTED_SOURCE': '', 'STAGE_TRACE': str(self.trace), 'SLURM_JOB_ID': 'v12m-test-job'}

    def invoke(self, *args, **env):
        return subprocess.run(['bash', str(self.launcher), *args], env={**self.env, **env},
                              capture_output=True, text=True, timeout=30)

    def test_failure_stages_keep_exit_status_and_stop_before_later_work(self):
        stages = ['device', 'tests', 'check', 'full']
        for stage, code in (('device', 13), ('tests', 9), ('check', 17), ('full', 23), ('success', 0)):
            with self.subTest(stage=stage):
                self.trace.unlink(missing_ok=True)
                name = 'contra_v12m_output_scope_' + stage
                result = self.invoke('run', name, FAIL_STAGE=stage)
                self.assertEqual(result.returncode, code, result.stderr)
                self.assertEqual(self.trace.read_text().splitlines(),
                                 stages if stage == 'success' else stages[:stages.index(stage) + 1])
                self.assertEqual('FULL_FIT_STARTED' in result.stdout, stage in ('full', 'success'))
                job = self.runs / (name + '.job')
                self.assertEqual((job / 'exit_code').read_text().strip(), str(code))
                self.assertEqual((job / 'slurm_job_id').read_text().strip(), 'v12m-test-job')
                self.assertTrue((job / 'host').read_text().strip())
                self.assertTrue((job / 'pid').read_text().strip().isdigit())
                before = (job / 'run.log').read_bytes()
                self.assertEqual(self.invoke('run', name).returncode, 1)
                self.assertEqual((job / 'run.log').read_bytes(), before)

    def test_resume_checks_fresh_package_and_passes_exact_readonly_partial_path(self):
        source_name = 'contra_v12m_output_scope_old'
        source = self.runs / (source_name + '.partial')
        source.mkdir()
        (source / 'run_identity.json').write_text('{"preserve": true}')
        (source / 'last_adapter.pt').write_bytes(b'preserved checkpoint bytes')
        before = {p.name: p.read_bytes() for p in source.iterdir()}
        result = self.invoke('resume', 'contra_v12m_output_scope_new', source_name,
                             EXPECTED_SOURCE=str(source))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.trace.read_text().splitlines(), ['device', 'tests', 'check', 'full'])
        self.assertEqual(before, {p.name: p.read_bytes() for p in source.iterdir()})
        self.assertIn(str(source), result.stdout)

    def test_existing_directories_files_and_dangling_links_are_preserved(self):
        for index, suffix in enumerate(('', '.partial', '.job')):
            for kind in ('directory', 'file', 'link'):
                name = f'contra_v12m_output_scope_saved_{index}_{kind}'
                target = self.runs / (name + suffix)
                if kind == 'link':
                    target.symlink_to(self.runs / 'absent', target_is_directory=True)
                elif kind == 'file':
                    target.write_bytes(b'preserve')
                else:
                    target.mkdir()
                result = self.invoke('run', name)
                self.assertEqual(result.returncode, 1)
                if kind == 'link':
                    self.assertTrue(target.is_symlink())
                elif kind == 'file':
                    self.assertEqual(target.read_bytes(), b'preserve')
                else:
                    self.assertTrue(target.is_dir())
        self.assertFalse(self.trace.exists())

    def test_invalid_names_actions_and_argument_counts_do_not_create_runs(self):
        name = 'contra_v12m_output_scope_new'
        for args in (('run', '../escape'), ('run', 'contra_v12k_objective_alignment_old'),
                     ('run', name, 'extra'), ('resume', name, name), ('resume', name),
                     ('resume', name, '../escape'), ('resume', name, name + '_old', 'extra'),
                     ('status', name, 'extra'), ('report', name, 'extra'), ('help', 'extra'),
                     ('unknown',), ('_worker', name)):
            with self.subTest(args=args):
                self.assertEqual(self.invoke(*args).returncode, 2)
        self.assertEqual(self.invoke('resume', name, 'contra_v12m_output_scope_absent').returncode, 1)
        self.assertEqual(list(self.runs.iterdir()), [])
        self.assertFalse(self.trace.exists())

    def test_symlinked_or_missing_recovery_identity_is_rejected(self):
        real = self.root / 'real'
        real.mkdir()
        (real / 'run_identity.json').write_text('{}')
        linked_name = 'contra_v12m_output_scope_linked'
        (self.runs / (linked_name + '.partial')).symlink_to(real, target_is_directory=True)
        identity_name = 'contra_v12m_output_scope_identity_link'
        identity_source = self.runs / (identity_name + '.partial')
        identity_source.mkdir()
        (identity_source / 'run_identity.json').symlink_to(real / 'run_identity.json')
        missing_name = 'contra_v12m_output_scope_missing_identity'
        (self.runs / (missing_name + '.partial')).mkdir()
        for name in (linked_name, identity_name, missing_name):
            result = self.invoke('resume', 'contra_v12m_output_scope_new', name)
            self.assertEqual(result.returncode, 1)
        self.assertFalse((self.runs / 'contra_v12m_output_scope_new.job').exists())

    def test_status_report_and_missing_results_expose_completion_state(self):
        name = 'contra_v12m_output_scope_done'
        result = self.invoke('run', name)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('six shared early-loss fits', result.stdout)
        self.assertIn('72 epochs', result.stdout)
        self.assertIn('12 primary endpoints + 6 derived P(U) outputs', result.stdout)
        self.assertIn('Summary:', self.invoke('status', name).stdout)
        self.assertIn(str(self.runs / name / 'summary.json'), self.invoke('report', name).stdout)
        failed = 'contra_v12m_output_scope_failed'
        self.invoke('run', failed, FAIL_STAGE='full')
        partial = self.runs / (failed + '.partial')
        partial.mkdir()
        result = self.invoke('report', failed)
        self.assertEqual(result.returncode, 1)
        self.assertNotIn('Traceback', result.stderr)
        status = self.invoke('status', failed)
        self.assertIn('Workflow exit code: 23', status.stdout)
        self.assertIn(str(partial), status.stdout)
        self.assertEqual(self.invoke('status', 'contra_v12m_output_scope_absent').returncode, 1)

    def test_help_default_run_and_device_override_use_v12m_contract(self):
        for args in ((), ('help',)):
            result = self.invoke(*args)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('V12M_DEVICE', result.stdout)
            self.assertIn('resume', result.stdout)
        self.assertFalse(self.trace.exists())
        result = self.invoke('run', V12M_DEVICE='cuda:2', EXPECTED_DEVICE='cuda:2')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((self.runs / 'contra_v12m_output_scope_01/summary.json').is_file())


if __name__ == '__main__':
    unittest.main()
