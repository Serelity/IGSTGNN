"""Check launcher sequencing and paired-report gates without GPU training."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / 'experiments/chronological/run_incident_routing_p2.sh'


def find_bash():
    if os.name == 'nt':
        git = shutil.which('git')
        if git:
            bundled = Path(git).resolve().parents[1] / 'bin/bash.exe'
            if bundled.is_file():
                return str(bundled)
        return None  # Do not invoke an unconfigured WSL bash.exe.
    return shutil.which('bash')


def shell_path(path):
    value = Path(path).resolve().as_posix()
    return '/' + value[0].lower() + value[2:] if os.name == 'nt' else value


BASH = find_bash()


@unittest.skipUnless(BASH, 'Bash is required for launcher checks')
class P2LauncherTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='p2 launcher ')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.script = self.root / 'experiments/chronological' / SCRIPT.name
        self.script.parent.mkdir(parents=True)
        shutil.copyfile(SCRIPT, self.script)
        self.data = self.root / 'data'
        self.data.mkdir()
        self.binary = self.root / 'env/bin'
        self.binary.mkdir(parents=True)
        self.calls = self.root / 'calls.jsonl'
        helper = self.root / 'fake_python.py'
        helper.write_text('''import json, os, sys
from pathlib import Path
args = sys.argv[1:]
with Path(os.environ['P2_TEST_CALLS']).open('a') as stream:
    stream.write(json.dumps(args) + '\\n')
if args[:3] == ['-u', '-', 'preflight']:
    sys.stdin.read()
    sys.exit(int(os.environ.get('P2_TEST_GPU_EXIT', '0')))
if args[:2] == ['-m', 'unittest']:
    sys.exit(int(os.environ.get('P2_TEST_UNIT_EXIT', '0')))
if args[:2] == ['-u', '-']:
    source = sys.stdin.read()
    sys.argv = ['-', args[2]]
    exec(compile(source, '<paired-report>', 'exec'), {'__name__': '__main__'})
    sys.exit(0)
if 'experiments/chronological/train.py' in args:
    variant = args[args.index('--variant') + 1]
    if os.environ.get('P2_TEST_TRAIN_FAIL') == variant:
        sys.exit(23)
    directory = Path(args[args.index('--output-dir') + 1])
    directory.mkdir(parents=True, exist_ok=False)
    identity = {'check': False, 'seed': 2025, 'batch_size': 48,
                'common_initialization_sha256': 'common',
                'protocol_sha256': 'protocol', 'package_sha256': {'data': 'same'},
                'source_sha256': {'model': 'same'}}
    if variant == 'acdg' and os.environ.get('P2_TEST_PAIR_MISMATCH'):
        identity['common_initialization_sha256'] = 'different'
    summary = {'variant': variant, 'identity': identity,
               'status': 'PAUSED_AT_EPOCH_BOUNDARY', 'completed_epoch': 1,
               'global_updates': 76, 'initial_max_abs_difference_from_A': 0,
               'parameters': 443645 if variant == 'fixed' else 465730,
               'runtime_epochs': [{'seconds': 12.5}], 'best_metric': 30.0,
               'history': [{'train_order_sha256': 'order', 'train': {'mae_macro': 31.0}}]}
    (directory / 'summary.json').write_text(json.dumps(summary))
    (directory / 'last_checkpoint.pt').write_bytes(b'fixture only')
    sys.exit(0)
raise SystemExit('Unexpected Python invocation: ' + repr(args))
''', encoding='utf-8')
        (self.binary / 'python').write_text(
            '#!/usr/bin/env bash\nexec "$P2_TEST_PYTHON" "$P2_TEST_HELPER" "$@"\n',
            encoding='utf-8', newline='\n')
        (self.binary / 'git').write_text(
            '#!/usr/bin/env bash\n'
            'if [[ "$1" == diff ]]; then exit "${P2_TEST_SOURCE_EXIT:-0}"; fi\n'
            'printf "commit=fixture\\n"\n', encoding='utf-8', newline='\n')
        for executable in self.binary.iterdir():
            executable.chmod(0o755)
        self.env = dict(os.environ,
                        CONDA_DEFAULT_ENV='igstgnn',
                        CONDA_PREFIX=shell_path(self.binary.parent),
                        P2_DATA_DIR=shell_path(self.data),
                        P2_TEST_PYTHON=Path(sys.executable).as_posix(),
                        P2_TEST_HELPER=helper.as_posix(),
                        P2_TEST_CALLS=str(self.calls))

    def run_launcher(self, *args, **overrides):
        return subprocess.run([BASH, self.script.as_posix(), *args],
                              cwd=self.root.parent, env={**self.env, **overrides},
                              capture_output=True, text=True, timeout=30)

    def recorded_calls(self):
        if not self.calls.exists():
            return []
        return [json.loads(line) for line in self.calls.read_text().splitlines()]

    def run_root(self):
        return next((self.root / 'experiments/chronological_runs').iterdir())

    def test_success_runs_two_full_epochs_and_checks_actual_pair_report(self):
        result = self.run_launcher()
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        calls = self.recorded_calls()
        self.assertEqual(calls[0], ['-u', '-', 'preflight'])
        self.assertEqual(calls[1][:2], ['-m', 'unittest'])
        runs = [args for args in calls if 'experiments/chronological/train.py' in args]
        self.assertEqual([args[args.index('--variant') + 1] for args in runs], ['fixed', 'acdg'])
        for args in runs:
            self.assertNotIn('--check', args)
            self.assertNotIn('--resume', args)
            self.assertEqual(args[args.index('--stop-after-epoch') + 1], '1')
            self.assertEqual(args[args.index('--device') + 1], 'cuda:0')
        root = self.run_root()
        report = json.loads((root / 'pair_report.json').read_text())
        self.assertEqual(report['status'], 'PAIR_IDENTITY_CHECK_PASS')
        self.assertEqual(report['additional_parameters'], 22085)
        self.assertEqual((root / 'exit_code').read_text().strip(), '0')
        self.assertIn('PAIR_IDENTITY_CHECK_PASS', (root / 'run.log').read_text())

    def test_failed_gpu_check_never_starts_tests_or_training(self):
        result = self.run_launcher(P2_TEST_GPU_EXIT='17')
        self.assertEqual(result.returncode, 17)
        self.assertEqual(self.recorded_calls(), [['-u', '-', 'preflight']])
        self.assertEqual((self.run_root() / 'exit_code').read_text().strip(), '17')

    def test_failed_unit_test_never_starts_training(self):
        result = self.run_launcher(P2_TEST_UNIT_EXIT='19')
        self.assertEqual(result.returncode, 19)
        self.assertFalse(any('experiments/chronological/train.py' in args for args in self.recorded_calls()))

    def test_failed_baseline_preserves_run_and_does_not_start_candidate(self):
        result = self.run_launcher(P2_TEST_TRAIN_FAIL='fixed')
        self.assertEqual(result.returncode, 23)
        runs = [args for args in self.recorded_calls() if 'experiments/chronological/train.py' in args]
        self.assertEqual(len(runs), 1)
        self.assertIn('fixed', runs[0])
        self.assertFalse((self.run_root() / 'pair_report.json').exists())
        self.assertEqual((self.run_root() / 'exit_code').read_text().strip(), '23')

    def test_source_drift_stops_before_python_or_run_creation(self):
        result = self.run_launcher(P2_TEST_SOURCE_EXIT='1')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('differ from the validated version', result.stderr)
        self.assertFalse(self.calls.exists())
        self.assertFalse((self.root / 'experiments/chronological_runs').exists())

    def test_pair_identity_mismatch_is_not_reported_as_success(self):
        result = self.run_launcher(P2_TEST_PAIR_MISMATCH='1')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Paired identity mismatch', result.stdout + result.stderr)
        self.assertFalse((self.run_root() / 'pair_report.json').exists())

    def test_help_and_bad_arguments_do_not_start_work(self):
        self.assertEqual(self.run_launcher('--help').returncode, 0)
        self.assertEqual(self.run_launcher('--resume').returncode, 2)
        self.assertFalse(self.calls.exists())


if __name__ == '__main__':
    unittest.main()
