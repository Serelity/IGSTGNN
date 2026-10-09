"""Check the CPU entry point and safe discovery of existing history packages."""

from pathlib import Path
import re
import subprocess
import tempfile
import unittest

from experiments.chronological.prepare_incident_corridors import discover_history
from src.utils.incident_corridor import write_json
from test_incident_physics_launcher import BASH, shell_path


REPO = Path(__file__).absolute().parents[1]
SCRIPT = REPO / 'experiments/chronological/run_incident_corridors.sh'


class CorridorLauncherTests(unittest.TestCase):
    def test_no_source_control_download_install_or_submission(self):
        text = SCRIPT.read_text(encoding='utf-8')
        self.assertIsNone(re.search(r'\b(git|curl|wget|pip|sbatch|srun)\b', text))
        self.assertNotIn(b'\r', SCRIPT.read_bytes())

    @unittest.skipUnless(BASH, 'Bash is required')
    def test_syntax_help_and_bad_arguments(self):
        for args in (['-n', shell_path(SCRIPT)], [shell_path(SCRIPT), '--help']):
            result = subprocess.run([BASH, *args], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('CPU only', result.stdout)
        for args in (['a', 'b'], ['--unknown'], ['nonexistent_corridor_history_directory']):
            result = subprocess.run([BASH, shell_path(SCRIPT), *args], capture_output=True, text=True)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertNotIn('Started:', result.stdout)

    def test_discovery_reuses_identical_content_and_rejects_conflicting_packages(self):
        scratch = REPO / 'experiments/chronological_runs'
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='corridor_discovery_', dir=scratch) as temp:
            root = Path(temp)
            self.assertIsNone(discover_history([root], 'source-hash'))
            for name in ('a', 'b'):
                directory = root / name
                directory.mkdir()
                for filename in ('train_history.npy', 'train_multichannel_scaler.json'):
                    (directory / filename).write_bytes(b'fixture')
                meta = {'status': 'MULTICHANNEL_HISTORY_MATERIALIZATION_COMPLETE',
                        'engineering_check': False, 'acceptance': {'gate_passed': True},
                        'inputs': {'data_summary_sha256': 'source-hash'},
                        'outputs': {'train_history.npy': {'sha256': 'same-history'}}}
                write_json(directory / 'summary.json', meta)
            self.assertEqual(discover_history([root], 'source-hash'), root / 'a')
            self.assertIsNone(discover_history([root], 'another-source'))
            meta['outputs']['train_history.npy']['sha256'] = 'different-history'
            write_json(root / 'b/summary.json', meta)
            with self.assertRaisesRegex(ValueError, 'Multiple different'):
                discover_history([root], 'source-hash')


if __name__ == '__main__':
    unittest.main()
