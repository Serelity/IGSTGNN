"""Launcher contracts and subprocess orchestration with a shell-only fake Python.

No model, CUDA allocation or real trainer executes in these tests. Linux-only
subprocess checks exercise process groups used by the actual server launcher.
"""
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile
import time
import unittest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / 'experiments' / 'original_multidata_only' / 'run.sh'
POSIX_TOOLS = os.name == 'posix' and all(shutil.which(name) for name in ('bash', 'setsid', 'flock', 'tee'))
FAKE_PYTHON = r'''#!/usr/bin/env bash
set -euo pipefail
if [ "${1:-}" = - ]; then cat >/dev/null; exit "${FAKE_GPU_EXIT:-0}"; fi
if [ "${1:-}" != -m ]; then exit 92; fi
module="$2"
shift 2
if [ "$module" = unittest ]; then printf 'UNITTEST\n' >> "$FAKE_EVENTS"; exit 0; fi
dataset= directory= output= split= engineering=0 resume=0
while [ "$#" -gt 0 ]; do
  case "$1" in
    --dataset) dataset="$2"; shift 2 ;;
    --run-dir) directory="$2"; shift 2 ;;
    --output-dir) output="$2"; shift 2 ;;
    --split) split="$2"; shift 2 ;;
    --check|--engineering) engineering=1; shift ;;
    --resume) resume=1; shift ;;
    *) shift ;;
  esac
done
case "$module" in
  experiments.state_exchange_multidata.baseline)
    phase=formal
    if [ "$engineering" -eq 1 ]; then phase=engineering; fi
    printf 'TRAIN|%s|%s|resume=%s\n' "$dataset" "$phase" "$resume" >> "$FAKE_EVENTS"
    mkdir -p "$directory"
    printf '{}\n' > "$directory/identity.json"
    if [ "$phase" = engineering ]; then
      touch "$FAKE_TMP/started_$dataset"
      other=Orange
      if [ "$dataset" = Orange ]; then other=Alameda; fi
      for count in $(seq 1 100); do
        if [ -f "$FAKE_TMP/started_$other" ]; then break; fi
        sleep 0.05
      done
      test -f "$FAKE_TMP/started_$other"
    else
      printf '%s\n' "$$" > "$FAKE_TMP/formal_pid_$dataset"
      if [ "${FAKE_FORMAL_WAIT:-0}" = 1 ]; then sleep 60; fi
      if [ "${FAKE_FAIL_DATASET:-}" = "$dataset" ]; then exit 7; fi
    fi
    printf 'DONE|%s|%s\n' "$dataset" "$phase" >> "$FAKE_EVENTS"
    ;;
  experiments.original_multidata_only.evaluate)
    mkdir -p "$output"
    printf '%s original %s\n' "$dataset" "$split" > "$output/report.md"
    printf 'EVAL|%s|%s|engineering=%s\n' "$dataset" "$split" "$engineering" >> "$FAKE_EVENTS"
    ;;
  experiments.original_multidata_only.batch_check)
    printf 'BATCH|%s\n' "$dataset" >> "$FAKE_EVENTS"
    ;;
  *) printf 'Unexpected module: %s\n' "$module" >&2; exit 93 ;;
esac
'''


class OriginalMultidataLauncherStaticTests(unittest.TestCase):
    def test_only_original_training_and_new_original_evaluation_are_invoked(self):
        text = LAUNCHER.read_text(encoding='utf-8')
        self.assertIn('python -m experiments.state_exchange_multidata.baseline', text)
        self.assertNotIn('python -m experiments.state_exchange_multidata.train', text)
        self.assertNotIn('state_exchange_multidata_candidate', text)
        self.assertNotIn('CONTRA_TEST', text)
        self.assertIn('--split test --validation-dir "$run/original_val"', text)

    def test_visible_devices_and_registered_output_paths_are_explicit(self):
        text = LAUNCHER.read_text(encoding='utf-8')
        self.assertIn('IGSTGNN_ORIGINAL_ALAMEDA_DEVICE:-cuda:0', text)
        self.assertIn('IGSTGNN_ORIGINAL_ORANGE_DEVICE:-cuda:1', text)
        self.assertNotIn('export CUDA_VISIBLE_DEVICES=', text)
        self.assertIn('original_alameda_orange_s2025', text)
        self.assertIn('original_multidata_paths.env', text)
        self.assertIn("'test_state_exchange_multidata_baseline.py'", text)
        self.assertIn("'test_original_multidata_only*.py'", text)


@unittest.skipUnless(POSIX_TOOLS, 'Server orchestration requires Linux bash, setsid, flock and tee')
class OriginalMultidataLauncherProcessTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='original-launcher-test-')
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)
        code = self.path / 'code'
        package = code / 'experiments' / 'original_multidata_only'
        package.mkdir(parents=True)
        self.script = package / 'run.sh'
        shutil.copyfile(LAUNCHER, self.script)
        (code / 'tests').mkdir()
        for dataset in ('Alameda', 'Orange'):
            data = self.path / 'data' / dataset
            data.mkdir(parents=True)
            for name in ('incident_all.npy', 'sensors.csv'):
                (data / name).write_bytes(b'dummy')
        binary = self.path / 'bin'
        binary.mkdir()
        python = binary / 'python'
        python.write_text(FAKE_PYTHON, encoding='utf-8')
        python.chmod(0o755)
        self.events_path = self.path / 'events.log'
        self.events_path.touch()
        self.env = dict(os.environ, PATH=str(binary) + os.pathsep + os.environ.get('PATH', ''),
            IGSTGNN_ORIGINAL_CODE=str(code), IGSTGNN_ORIGINAL_DATA=str(self.path / 'data'),
            IGSTGNN_ORIGINAL_RUN=str(self.path / 'run'), IGSTGNN_ORIGINAL_THREADS='1',
            IGSTGNN_ORIGINAL_EPOCHS='100', IGSTGNN_ORIGINAL_ALAMEDA_DEVICE='cuda:0',
            IGSTGNN_ORIGINAL_ORANGE_DEVICE='cuda:1', FAKE_EVENTS=str(self.events_path),
            FAKE_TMP=str(self.path), FAKE_GPU_EXIT='0', FAKE_FAIL_DATASET='', FAKE_FORMAL_WAIT='0')

    def run_launcher(self, **updates):
        return subprocess.run(['bash', str(self.script)], env=dict(self.env, **updates),
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=20)

    def events(self):
        return self.events_path.read_text(encoding='utf-8')

    def test_workers_run_concurrently_and_both_finish_original_test(self):
        result = self.run_launcher()
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertEqual(self.events().count('UNITTEST'), 2)
        for dataset in ('Alameda', 'Orange'):
            self.assertIn(f'EVAL|{dataset}|test|engineering=0', self.events())
            self.assertIn(f'DONE|{dataset}|formal', self.events())
            log = self.path / 'run' / f'{dataset}.launcher.log'
            self.assertTrue(log.is_file())
            self.assertIn(f'{dataset} original test', log.read_text(encoding='utf-8'))

    def test_one_failed_worker_does_not_cancel_the_other(self):
        result = self.run_launcher(FAKE_FAIL_DATASET='Alameda')
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn('Pipeline failed (exit 7)', result.stdout)
        self.assertIn('EVAL|Orange|test|engineering=0', self.events())
        self.assertNotIn('EVAL|Alameda|test', self.events())

    def test_repeated_launcher_resumes_each_existing_training_directory(self):
        first = self.run_launcher()
        self.assertEqual(first.returncode, 0, first.stdout)
        self.events_path.write_text('', encoding='utf-8')
        second = self.run_launcher()
        self.assertEqual(second.returncode, 0, second.stdout)
        for dataset in ('Alameda', 'Orange'):
            for phase in ('engineering', 'formal'):
                self.assertIn(f'TRAIN|{dataset}|{phase}|resume=1', self.events())

    def test_partial_initialization_is_preserved_and_rejected(self):
        partial = self.path / 'run' / 'Alameda' / 'original'
        partial.mkdir(parents=True)
        marker = partial / 'keep.txt'
        marker.write_text('preserve', encoding='utf-8')
        result = self.run_launcher()
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn('Incomplete initialization:', result.stdout)
        self.assertEqual(marker.read_text(encoding='utf-8'), 'preserve')
        self.assertIn('EVAL|Orange|test|engineering=0', self.events())
        self.assertNotIn('TRAIN|Alameda|formal', self.events())

    def test_gpu_preflight_failure_runs_no_training_or_tests(self):
        result = self.run_launcher(FAKE_GPU_EXIT='6')
        self.assertEqual(result.returncode, 6, result.stdout)
        self.assertEqual(self.events(), '')

    def test_partial_epoch_target_does_not_run_formal_evaluations(self):
        result = self.run_launcher(IGSTGNN_ORIGINAL_EPOCHS='1')
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertNotIn('|test|', self.events())
        self.assertNotIn('|val|engineering=0', self.events())
        self.assertIn('Repeat with IGSTGNN_ORIGINAL_EPOCHS=100', result.stdout)

    def test_sigterm_stops_both_python_process_groups(self):
        process = subprocess.Popen(['bash', str(self.script)], env=dict(self.env, FAKE_FORMAL_WAIT='1'),
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        def cleanup():
            if process.poll() is None:
                process.terminate()
                try:
                    process.communicate(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.communicate(timeout=5)
        self.addCleanup(cleanup)
        deadline = time.monotonic() + 10
        pids = []
        while time.monotonic() < deadline:
            files = [self.path / f'formal_pid_{name}' for name in ('Alameda', 'Orange')]
            if all(path.is_file() for path in files):
                pids = [int(path.read_text(encoding='utf-8')) for path in files]
                break
            time.sleep(0.05)
        self.assertEqual(len(pids), 2, 'Both fake trainers must start before interruption')
        process.send_signal(signal.SIGTERM)
        output, _ = process.communicate(timeout=10)
        self.assertEqual(process.returncode, 143, output)
        for pid in pids:
            state_file = Path(f'/proc/{pid}/stat')
            if state_file.exists():
                self.assertEqual(state_file.read_text(encoding='utf-8').split()[2], 'Z')
        self.assertNotIn('|test|', self.events())


if __name__ == '__main__':
    unittest.main()
