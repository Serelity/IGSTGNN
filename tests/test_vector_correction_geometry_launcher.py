"""Foreground launcher: quoted paths, stage failures, preserved runs and routing."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        directory = self.root / 'repo with spaces/experiments/chronological'; directory.mkdir(parents=True)
        self.script = directory / 'run_vector_correction_geometry.sh'
        shutil.copyfile(Path(__file__).resolve().parents[1] / 'experiments/chronological' / self.script.name, self.script)
        self.runs = directory.parent / 'chronological_runs'; self.runs.mkdir()
        binary = self.root / 'binary with spaces'; binary.mkdir()
        self.trace = self.root / 'trace'
        python = binary / 'python'
        python.write_text('''#!/usr/bin/env bash
set -euo pipefail
if [[ "$1" == -c ]]; then
  if [[ "$2" == *sys.executable* ]]; then printf '%s\\n' "$0"; exit 0; fi
  if [[ "$2" == *'Symlinked output parent'* ]]; then exit 0; fi
  if [[ "$3" != cpu ]]; then exit 92; fi
  printf 'device\\n' >> "$TRACE"
  if [[ "$FAIL_STAGE" == device ]]; then exit 13; fi
  exit 0
fi
if [[ "$1" == -m ]]; then
  if [[ "$*" != '-m unittest discover -s tests -p test_vector_correction_geometry*.py -v' ]]; then exit 93; fi
  printf 'tests\\n' >> "$TRACE"
  if [[ "$FAIL_STAGE" == tests ]]; then exit 17; fi
  exit 0
fi
if [[ "$1" == -u ]]; then shift; fi
if [[ "$1" != */audit_vector_correction_geometry.py ]]; then exit 94; fi
MODE=$2
shift 2
if [[ "$MODE" == report ]]; then echo "REPORT: $1"; exit 0; fi
OUT=''
SOURCE=''
CHECK=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --source) SOURCE=$2; shift 2 ;;
    --data-dir|--primary-control-dir|--secondary-control-dir|--checkpoint|--device) shift 2 ;;
    --output) OUT=$2; shift 2 ;;
    --check) CHECK=1; shift ;;
    *) exit 95 ;;
  esac
done
if [[ "$SOURCE" != "$EXPECTED_SOURCE" ]]; then exit 96; fi
if [[ "$MODE" == preflight ]]; then
  printf 'preflight\\n' >> "$TRACE"
  if [[ "$FAIL_STAGE" == preflight ]]; then exit 19; fi
  exit 0
fi
if [[ "$MODE" != run ]]; then exit 97; fi
if [[ "$CHECK" == 1 ]]; then
  printf 'check\\n' >> "$TRACE"
  if [[ "$FAIL_STAGE" == check ]]; then exit 23; fi
else
  printf 'formal\\n' >> "$TRACE"
  if [[ "$FAIL_STAGE" == formal ]]; then exit 29; fi
fi
mkdir -p -- "$OUT"
printf '{}\\n' > "$OUT/summary.json"
''')
        python.chmod(0o755)
        git = binary / 'git'; git.write_text('#!/usr/bin/env bash\nexit 0\n'); git.chmod(0o755)
        self.env = {**os.environ, 'PATH': str(binary)+os.pathsep+os.environ['PATH'], 'V12N_DEVICE': 'cpu',
            'TRACE': str(self.trace), 'EXPECTED_SOURCE': str(self.runs/'contra_v12m_output_scope_01'),
            'FAIL_STAGE': 'success', 'SLURM_JOB_ID': 'synthetic-v12n-job'}

    def invoke(self, *args, **env):
        return subprocess.run(['bash', str(self.script), *args], env={**self.env, **env},
                              capture_output=True, text=True, timeout=30)

    def test_all_stage_failures_stop_and_record_original_exit_code(self):
        stages = ['device', 'tests', 'preflight', 'check', 'formal']
        for stage, code in (('device',13),('tests',17),('preflight',19),('check',23),('formal',29),('success',0)):
            with self.subTest(stage=stage):
                self.trace.unlink(missing_ok=True)
                name = 'contra_v12n_geometry_'+stage
                result = self.invoke('run', name, FAIL_STAGE=stage)
                self.assertEqual(result.returncode, code, result.stderr)
                expected = stages if stage=='success' else stages[:stages.index(stage)+1]
                self.assertEqual(self.trace.read_text().splitlines(), expected)
                job = self.runs/(name+'.job')
                self.assertEqual((job/'exit_code').read_text().strip(), str(code))
                self.assertEqual((job/'slurm_job_id').read_text().strip(), 'synthetic-v12n-job')
                previous = (job/'run.log').read_bytes()
                self.assertEqual(self.invoke('run',name).returncode, 1)
                self.assertEqual((job/'run.log').read_bytes(), previous)

    def test_check_mode_never_starts_formal_run(self):
        result = self.invoke('check', 'contra_v12n_geometry_subset')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.trace.read_text().splitlines(), ['device','tests','preflight','check'])
        self.assertTrue((self.runs/'contra_v12n_geometry_subset/summary.json').is_file())

    def test_preflight_reads_custom_source_without_creating_output(self):
        source = 'contra_v12m_output_scope_custom'
        result = self.invoke('preflight','contra_v12n_geometry_pre',source,
                             EXPECTED_SOURCE=str(self.runs/source))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.trace.read_text().splitlines(), ['preflight'])
        self.assertFalse((self.runs/'contra_v12n_geometry_pre.job').exists())

    def test_existing_partial_job_file_and_broken_link_are_preserved(self):
        for i, suffix in enumerate(('', '.partial', '.job')):
            for kind in ('directory','file','link'):
                name=f'contra_v12n_geometry_keep_{i}_{kind}'; target=self.runs/(name+suffix)
                if kind=='directory': target.mkdir()
                elif kind=='file': target.write_bytes(b'preserve')
                else: target.symlink_to(self.runs/'missing')
                self.assertEqual(self.invoke('run',name).returncode,1)
                if kind=='link': self.assertTrue(target.is_symlink())
                else: self.assertTrue(target.exists())

    def test_invalid_actions_names_and_extra_arguments(self):
        for args in (('other',),('run','../../escape'),('run','contra_v12n_geometry_a;echo'),
                     ('run','contra_v12n_geometry_a','../../source'),('status','contra_v12n_geometry_a','extra'),
                     ('run','contra_v12n_geometry_a','contra_v12m_output_scope_01','extra')):
            with self.subTest(args=args): self.assertEqual(self.invoke(*args).returncode,2)

    def test_report_requires_completion_and_help_describes_server_environment(self):
        name='contra_v12n_geometry_report'
        self.assertEqual(self.invoke('report',name).returncode,1)
        out=self.runs/name;out.mkdir();(out/'summary.json').write_text('{}')
        self.assertIn('REPORT:',self.invoke('report',name).stdout)
        help_text=self.invoke('help').stdout
        self.assertIn('igstgnn',help_text);self.assertIn('V100',help_text)


if __name__=='__main__':unittest.main()
