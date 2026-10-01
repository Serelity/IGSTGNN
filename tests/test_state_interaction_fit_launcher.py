"""Inference-only full-fit launch, error propagation and artifact preservation."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        scripts = root / 'repo with spaces/experiments/chronological'
        scripts.mkdir(parents=True)
        self.launcher = scripts / 'run_state_interaction_fit_audit.sh'
        shutil.copyfile(Path(__file__).resolve().parents[1] / 'experiments/chronological' /
                        self.launcher.name, self.launcher)
        self.runs = scripts.parent / 'chronological_runs'
        self.runs.mkdir()
        self.default_source = self.create_source('contra_v12f_state_interaction_01')
        binary = root / 'bin with spaces'
        binary.mkdir()
        python = binary / 'python'
        python.write_text('''#!/usr/bin/env bash
if [[ "$1" == -c ]]; then
  if [[ "$2" == *sys.executable* ]]; then printf '%s\\n' "$0"; fi
  if [[ "$2" == *torch* && "$FAIL_STAGE" == preflight ]]; then exit 8; fi
  exit 0
fi
if [[ "$1" == -m ]]; then
  if [[ "$*" != *test_state_interaction_fit\\*.py* ]]; then exit 91; fi
  if [[ "$FAIL_STAGE" == tests ]]; then exit 9; else exit 0; fi
fi
if [[ "$2" == report ]]; then
  printf 'REPORT: %s\\n' "$3"
  exit 0
fi
if [[ "$1" != -u || "$3" != run ]]; then exit 92; fi
OUTPUT=''
SOURCE=''
DEVICE=''
DATA=''
PRIMARY=''
SECONDARY=''
CHECKPOINT=''
PREVIOUS=''
for arg in "$@"; do
  case "$arg" in --check|--resume-from|--epochs|--seed|--optimizer) exit 93 ;; esac
  case "$PREVIOUS" in
    --output) OUTPUT=$arg ;;
    --source-dir) SOURCE=$arg ;;
    --device) DEVICE=$arg ;;
    --data-dir) DATA=$arg ;;
    --primary-control-dir) PRIMARY=$arg ;;
    --secondary-control-dir) SECONDARY=$arg ;;
    --checkpoint) CHECKPOINT=$arg ;;
  esac
  PREVIOUS=$arg
done
if [[ "$SOURCE" != "$EXPECT_SOURCE" || "$DEVICE" != "$EXPECT_DEVICE" ]]; then exit 94; fi
if [[ "$DATA" != ../data/chronological/Contra_Costa_v8_dev ||
      "$PRIMARY" != ../research_artifacts/v6_inputs_20260920/v3_materialized_01 ||
      "$SECONDARY" != ../research_artifacts/v6_inputs_20260920/v5b_second_materialized_01 ||
      "$CHECKPOINT" != experiments/chronological_runs/contra_fixed_s2025_full_01/best_model.pt ]]; then exit 95; fi
if [[ "$CUBLAS_WORKSPACE_CONFIG" != :4096:8 || "$OMP_NUM_THREADS" != 3 ||
      "$OPENBLAS_NUM_THREADS" != 3 || "$MKL_NUM_THREADS" != 3 || "$NUMEXPR_NUM_THREADS" != 3 ]]; then exit 96; fi
printf 'AUDIT_STARTED: %s\\n' "$SOURCE"
if [[ "$FAIL_STAGE" == audit ]]; then exit 23; fi
mkdir -p -- "$OUTPUT"
printf '{}\\n' > "$OUTPUT/summary.json"
exit 0
''')
        python.chmod(0o755)
        git = binary / 'git'
        git.write_text('#!/usr/bin/env bash\nexit 0\n')
        git.chmod(0o755)
        self.env = {**os.environ, 'PATH': str(binary) + os.pathsep + os.environ['PATH'],
                    'FAIL_STAGE': 'success', 'EXPECT_SOURCE': str(self.default_source),
                    'EXPECT_DEVICE': 'cuda:0', 'V12I_DEVICE': 'cuda:0',
                    'SLURM_JOB_ID': '12345'}

    def create_source(self, name):
        source = self.runs / name
        source.mkdir()
        for filename in ('summary.json', 'run_identity.json', 'eligibility.json'):
            (source / filename).write_text('{"preserve": true}')
        for arm in ('strength', 'state_vector', 'interaction_vector'):
            for seed in (2025, 2026, 2027):
                directory = source / f'{arm}_s{seed}'
                directory.mkdir()
                (directory / 'selected_gate.pt').write_bytes(f'preserve:{arm}:{seed}'.encode())
        return source

    def snapshot(self, source):
        return {str(path.relative_to(source)): path.read_bytes()
                for path in source.rglob('*') if path.is_file()}

    def invoke(self, *args, **env):
        return subprocess.run(['bash', str(self.launcher), *args],
                              env={**self.env, **env}, capture_output=True,
                              text=True, timeout=30)

    def test_default_run_source_and_completed_report(self):
        before = self.snapshot(self.default_source)
        result = self.invoke('run')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('Device: cuda:0', result.stdout)
        self.assertIn('No training or epoch reselection', result.stdout)
        name = 'contra_v12i_full_fit_audit_01'
        job = self.runs / (name + '.job')
        self.assertEqual((job / 'exit_code').read_text().strip(), '0')
        self.assertEqual((job / 'slurm_job_id').read_text().strip(), '12345')
        self.assertTrue((job / 'pid').read_text().strip().isdigit())
        self.assertTrue((job / 'host').read_text().strip())
        self.assertIn('AUDIT_STARTED', (job / 'run.log').read_text())
        self.assertIn('Summary:', self.invoke('status', name).stdout)
        report = self.invoke('report', name)
        self.assertEqual(report.returncode, 0, report.stderr)
        self.assertIn(str(self.runs / name / 'summary.json'), report.stdout)
        self.assertEqual(self.snapshot(self.default_source), before)

    def test_source_and_device_override_are_read_only(self):
        source = self.create_source('contra_v12f_state_interaction_03')
        before = self.snapshot(source)
        result = self.invoke('run', 'contra_v12i_full_fit_audit_other', source.name,
                             EXPECT_SOURCE=str(source), V12I_DEVICE='cuda:1', EXPECT_DEVICE='cuda:1')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(str(source), result.stdout)
        self.assertIn('Device: cuda:1', result.stdout)
        self.assertEqual(self.snapshot(source), before)

    def test_explicit_cpu_override_remains_inference_only(self):
        result = self.invoke('run', 'contra_v12i_full_fit_audit_cpu',
                             V12I_DEVICE='cpu', EXPECT_DEVICE='cpu')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('Device: cpu', result.stdout)
        self.assertIn('No training or epoch reselection', result.stdout)

    def test_failure_propagation_through_tee_and_existing_job_guard(self):
        for stage, code in (('preflight', 8), ('tests', 9), ('audit', 23), ('success', 0)):
            with self.subTest(stage=stage):
                name = 'contra_v12i_full_fit_audit_' + stage
                result = self.invoke('run', name, FAIL_STAGE=stage)
                self.assertEqual(result.returncode, code, result.stderr)
                self.assertEqual('AUDIT_STARTED' in result.stdout,
                                 stage in ('audit', 'success'))
                job = self.runs / (name + '.job')
                self.assertEqual((job / 'exit_code').read_text().strip(), str(code))
                log = (job / 'run.log').read_bytes()
                self.assertEqual(self.invoke('run', name).returncode, 1)
                self.assertEqual((job / 'run.log').read_bytes(), log)

    def test_existing_artifacts_are_preserved(self):
        for index, suffix in enumerate(('', '.partial', '.job')):
            with self.subTest(suffix=suffix):
                name = 'contra_v12i_full_fit_audit_preserve' + str(index)
                target = self.runs / (name + suffix)
                target.mkdir()
                sentinel = target / 'saved'
                sentinel.write_text('keep')
                result = self.invoke('run', name)
                self.assertEqual(result.returncode, 1)
                self.assertEqual(sentinel.read_text(), 'keep')
                self.assertNotIn('AUDIT_STARTED', result.stdout)
                self.assertEqual(list(target.iterdir()), [sentinel])

    def test_dangling_symlink_artifacts_are_preserved(self):
        for index, suffix in enumerate(('', '.partial', '.job')):
            with self.subTest(suffix=suffix):
                name = 'contra_v12i_full_fit_audit_symlink' + str(index)
                target = self.runs / (name + suffix)
                destination = self.runs / ('absent' + str(index))
                target.symlink_to(destination)
                result = self.invoke('run', name)
                self.assertEqual(result.returncode, 1)
                self.assertTrue(target.is_symlink())
                self.assertEqual(target.readlink(), destination)
                self.assertNotIn('AUDIT_STARTED', result.stdout)

    def test_invalid_names_missing_source_and_no_training_recovery(self):
        name = 'contra_v12i_full_fit_audit_new'
        for args in (('run', '../escape'), ('run', name, '../escape'),
                     ('run', name, 'contra_v12f_state_interaction_01.partial'),
                     ('resume', name), ('check', name)):
            with self.subTest(args=args):
                self.assertEqual(self.invoke(*args).returncode, 2)
        result = self.invoke('run', name, 'contra_v12f_state_interaction_absent')
        self.assertEqual(result.returncode, 1)
        self.assertEqual(list(self.runs.iterdir()), [self.default_source])

    def test_each_required_source_artifact_fails_before_job_creation(self):
        required = ['summary.json', 'run_identity.json', 'eligibility.json']
        required += [f'{arm}_s{seed}/selected_gate.pt'
                     for arm in ('strength', 'state_vector', 'interaction_vector')
                     for seed in (2025, 2026, 2027)]
        for index, filename in enumerate(required):
            with self.subTest(filename=filename):
                source = self.create_source('contra_v12f_state_interaction_missing' + str(index))
                (source / filename).unlink()
                before = self.snapshot(source)
                name = 'contra_v12i_full_fit_audit_missing' + str(index)
                result = self.invoke('run', name, source.name)
                self.assertEqual(result.returncode, 1)
                self.assertIn(filename, result.stderr)
                self.assertFalse((self.runs / (name + '.job')).exists())
                self.assertEqual(self.snapshot(source), before)

    def test_source_symlink_does_not_promote_partial_to_completed(self):
        partial = self.create_source('contra_v12f_state_interaction_incomplete.partial')
        source = self.runs / 'contra_v12f_state_interaction_linked'
        source.symlink_to(partial)
        name = 'contra_v12i_full_fit_audit_linked'
        before = self.snapshot(partial)
        result = self.invoke('run', name, source.name)
        self.assertEqual(result.returncode, 1)
        self.assertFalse((self.runs / (name + '.job')).exists())
        self.assertEqual(self.snapshot(partial), before)

    def test_missing_summary_reports_incomplete_without_python_traceback(self):
        name = 'contra_v12i_full_fit_audit_interrupted'
        result = self.invoke('run', name, FAIL_STAGE='audit')
        self.assertEqual(result.returncode, 23)
        partial = self.runs / (name + '.partial')
        partial.mkdir()
        status = self.invoke('status', name)
        self.assertEqual(status.returncode, 0, status.stderr)
        self.assertIn('Workflow exit code: 23', status.stdout)
        self.assertIn(str(partial), status.stdout)
        report = self.invoke('report', name)
        self.assertEqual(report.returncode, 1)
        self.assertIn('status', report.stderr)
        self.assertNotIn('Traceback', report.stderr)


if __name__ == '__main__':
    unittest.main()
