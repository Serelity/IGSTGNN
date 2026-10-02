"""Saved-only support launch, error propagation and source preservation."""

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
        self.launcher = scripts / 'run_state_interaction_support_audit.sh'
        shutil.copyfile(Path(__file__).resolve().parents[1] / 'experiments/chronological' /
                        self.launcher.name, self.launcher)
        self.runs = scripts.parent / 'chronological_runs'
        self.runs.mkdir()
        self.default_fit = self.create_source('contra_v12i_full_fit_audit_01', fit=True)
        self.default_source = self.create_source('contra_v12f_state_interaction_01')
        binary = root / 'bin with spaces'
        binary.mkdir()
        python = binary / 'python'
        python.write_text('''#!/usr/bin/env bash
if [[ "$1" == -c ]]; then
  if [[ "$2" == *sys.executable* ]]; then printf '%s\\n' "$0"; fi
  if [[ "$2" == *torch* || "$2" == *cuda* ]]; then exit 90; fi
  if [[ "$2" == *numpy* && "$FAIL_STAGE" == preflight ]]; then exit 8; fi
  exit 0
fi
if [[ "$1" == -m ]]; then
  if [[ "$*" != *test_state_interaction_support\\*.py* ]]; then exit 91; fi
  if [[ "$FAIL_STAGE" == tests ]]; then exit 9; else exit 0; fi
fi
if [[ "$2" == report ]]; then
  printf 'REPORT: %s\\n' "$3"
  exit 0
fi
if [[ "$1" != -u || "$3" != run ]]; then exit 92; fi
OUTPUT=''
FIT=''
SOURCE=''
DATA=''
PREVIOUS=''
for arg in "$@"; do
  case "$arg" in --check|--device|--resume-from|--checkpoint|--epochs|--seed|--optimizer) exit 93 ;; esac
  case "$PREVIOUS" in
    --output) OUTPUT=$arg ;;
    --fit-dir) FIT=$arg ;;
    --source-dir) SOURCE=$arg ;;
    --data-dir) DATA=$arg ;;
  esac
  PREVIOUS=$arg
done
if [[ "$SOURCE" != "$EXPECT_SOURCE" || "$FIT" != "$EXPECT_FIT" ]]; then exit 94; fi
if [[ "$DATA" != ../data/chronological/Contra_Costa_v8_dev ]]; then exit 95; fi
if [[ "$OMP_NUM_THREADS" != 3 || "$OPENBLAS_NUM_THREADS" != 3 ||
      "$MKL_NUM_THREADS" != 3 || "$NUMEXPR_NUM_THREADS" != 3 ]]; then exit 96; fi
printf 'AUDIT_STARTED: %s %s\\n' "$FIT" "$SOURCE"
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
                    'EXPECT_FIT': str(self.default_fit), 'SLURM_JOB_ID': '12345'}

    @staticmethod
    def required_files(fit):
        if fit:
            return ['summary.json', 'fit_A.npz'] + [
                f'fit_{arm}_s{seed}.npz'
                for arm in ('strength', 'state_vector', 'interaction_vector')
                for seed in (2025, 2026, 2027)]
        return ['summary.json', 'run_identity.json', 'eligibility.json',
                'audit_A_incident_full.npz'] + [
                    f'{arm}_s{seed}/audit_incident_full.npz'
                    for arm in ('strength', 'state_vector', 'interaction_vector')
                    for seed in (2025, 2026, 2027)]

    def create_source(self, name, fit=False):
        source = self.runs / name
        source.mkdir()
        for filename in self.required_files(fit):
            path = source / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f'preserve:{filename}'.encode())
        return source

    def snapshot(self, source):
        return {str(path.relative_to(source)): path.read_bytes()
                for path in source.rglob('*') if path.is_file()}

    def invoke(self, *args, **env):
        return subprocess.run(['bash', str(self.launcher), *args],
                              env={**self.env, **env}, capture_output=True,
                              text=True, timeout=30)

    def test_default_run_sources_and_completed_report(self):
        before = [self.snapshot(source) for source in (self.default_fit, self.default_source)]
        result = self.invoke('run')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('Device: CPU (NumPy)', result.stdout)
        self.assertIn('No training, GPU inference or model selection', result.stdout)
        name = 'contra_v12j_support_audit_01'
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
        self.assertEqual([self.snapshot(source) for source in
                          (self.default_fit, self.default_source)], before)

    def test_both_source_overrides_are_read_only(self):
        fit = self.create_source('contra_v12i_full_fit_audit_03', fit=True)
        source = self.create_source('contra_v12f_state_interaction_03')
        before = [self.snapshot(directory) for directory in (fit, source)]
        result = self.invoke('run', 'contra_v12j_support_audit_other', fit.name,
                             source.name, EXPECT_FIT=str(fit), EXPECT_SOURCE=str(source))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(str(fit), result.stdout)
        self.assertIn(str(source), result.stdout)
        self.assertEqual([self.snapshot(directory) for directory in (fit, source)], before)

    def test_fit_override_retains_default_model_source(self):
        fit = self.create_source('contra_v12i_full_fit_audit_02', fit=True)
        result = self.invoke('run', 'contra_v12j_support_audit_fit_only', fit.name,
                             EXPECT_FIT=str(fit))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(str(self.default_source), result.stdout)

    def test_failure_propagation_through_tee_and_existing_job_guard(self):
        for stage, code in (('preflight', 8), ('tests', 9), ('audit', 23), ('success', 0)):
            with self.subTest(stage=stage):
                name = 'contra_v12j_support_audit_' + stage
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
                name = 'contra_v12j_support_audit_preserve' + str(index)
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
                name = 'contra_v12j_support_audit_symlink' + str(index)
                target = self.runs / (name + suffix)
                destination = self.runs / ('absent' + str(index))
                target.symlink_to(destination)
                result = self.invoke('run', name)
                self.assertEqual(result.returncode, 1)
                self.assertTrue(target.is_symlink())
                self.assertEqual(target.readlink(), destination)
                self.assertNotIn('AUDIT_STARTED', result.stdout)

    def test_invalid_names_missing_sources_and_no_training_recovery(self):
        name = 'contra_v12j_support_audit_new'
        for args in (('run', '../escape'), ('run', name, '../escape'),
                     ('run', name, self.default_fit.name, '../escape'),
                     ('run', name, self.default_fit.name + '.partial'),
                     ('run', name, self.default_fit.name, self.default_source.name + '.partial'),
                     ('run', name, self.default_fit.name, self.default_source.name, '--device'),
                     ('resume', name), ('check', name)):
            with self.subTest(args=args):
                self.assertEqual(self.invoke(*args).returncode, 2)
        for fit_name, source_name in (
                ('contra_v12i_full_fit_audit_absent', self.default_source.name),
                (self.default_fit.name, 'contra_v12f_state_interaction_absent')):
            with self.subTest(fit=fit_name, source=source_name):
                self.assertEqual(self.invoke('run', name, fit_name, source_name).returncode, 1)
        self.assertEqual(set(self.runs.iterdir()), {self.default_fit, self.default_source})

    def test_each_required_source_artifact_fails_before_job_creation(self):
        for fit in (True, False):
            for index, filename in enumerate(self.required_files(fit)):
                with self.subTest(fit=fit, filename=filename):
                    prefix = ('contra_v12i_full_fit_audit_' if fit else
                              'contra_v12f_state_interaction_')
                    source = self.create_source(prefix + 'missing' + str(index), fit=fit)
                    (source / filename).unlink()
                    before = self.snapshot(source)
                    name = 'contra_v12j_support_audit_missing' + str(fit) + str(index)
                    result = self.invoke('run', name,
                                         source.name if fit else self.default_fit.name,
                                         self.default_source.name if fit else source.name)
                    self.assertEqual(result.returncode, 1)
                    self.assertIn(filename, result.stderr)
                    self.assertFalse((self.runs / (name + '.job')).exists())
                    self.assertEqual(self.snapshot(source), before)

    def test_source_symlinks_do_not_promote_partial_to_completed(self):
        for fit in (True, False):
            with self.subTest(fit=fit):
                prefix = ('contra_v12i_full_fit_audit_' if fit else
                          'contra_v12f_state_interaction_')
                partial = self.create_source(prefix + 'incomplete.partial', fit=fit)
                source = self.runs / (prefix + 'linked')
                source.symlink_to(partial)
                name = 'contra_v12j_support_audit_linked' + str(fit)
                before = self.snapshot(partial)
                result = self.invoke('run', name,
                                     source.name if fit else self.default_fit.name,
                                     self.default_source.name if fit else source.name)
                self.assertEqual(result.returncode, 1)
                self.assertFalse((self.runs / (name + '.job')).exists())
                self.assertEqual(self.snapshot(partial), before)

    def test_symlinked_source_file_and_model_directory_are_rejected(self):
        for index, (fit, filename) in enumerate(((True, 'fit_A.npz'),
                                               (False, 'summary.json'),
                                               (False, 'state_vector_s2025'))):
            with self.subTest(fit=fit, filename=filename):
                prefix = ('contra_v12i_full_fit_audit_' if fit else
                          'contra_v12f_state_interaction_')
                source = self.create_source(prefix + 'linkfile' + str(index), fit=fit)
                target = source / filename
                saved = source / ('saved' + str(index))
                target.rename(saved)
                target.symlink_to(saved)
                name = 'contra_v12j_support_audit_linkfile' + str(index)
                before = self.snapshot(source)
                result = self.invoke('run', name,
                                     source.name if fit else self.default_fit.name,
                                     self.default_source.name if fit else source.name)
                self.assertEqual(result.returncode, 1)
                self.assertIn(filename, result.stderr)
                self.assertFalse((self.runs / (name + '.job')).exists())
                self.assertTrue(target.is_symlink())
                self.assertEqual(self.snapshot(source), before)

    def test_missing_summary_reports_incomplete_without_python_traceback(self):
        name = 'contra_v12j_support_audit_interrupted'
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

    def test_status_without_job_and_help_are_graceful(self):
        result = self.invoke('status')
        self.assertEqual(result.returncode, 1)
        self.assertIn('No run record', result.stderr)
        self.assertNotIn('Traceback', result.stderr)
        result = self.invoke('help')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('CPU NumPy diagnostic', result.stdout)
        self.assertIn('No training, GPU, check or resume', result.stdout)


if __name__ == '__main__':
    unittest.main()
