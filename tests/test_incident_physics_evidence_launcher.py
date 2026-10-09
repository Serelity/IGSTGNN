"""Check the server evidence entry point without scheduling a job."""

from pathlib import Path
import re
import subprocess
import unittest

from test_incident_physics_launcher import BASH, shell_path


SCRIPT = Path(__file__).resolve().parents[1] / 'experiments/chronological/run_incident_physics_evidence.sh'


class EvidenceLauncherTests(unittest.TestCase):
    def test_no_git_network_install_or_scheduler_submission(self):
        text = SCRIPT.read_text(encoding='utf-8')
        self.assertIsNone(re.search(r'\b(git|curl|wget|pip|sbatch|srun)\b', text))
        self.assertNotIn('\r', SCRIPT.read_bytes().decode('utf-8'))
        self.assertNotIn('check_incident_physics.py', text)

    @unittest.skipUnless(BASH, 'Bash is required')
    def test_shell_syntax_and_help(self):
        for args in (['-n', shell_path(SCRIPT)], [shell_path(SCRIPT), '--help']):
            result = subprocess.run([BASH, *args], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('CPU only', result.stdout)

    @unittest.skipUnless(BASH, 'Bash is required')
    def test_invalid_or_missing_source_fails_before_environment_or_data_work(self):
        for args in (['a', 'b'], ['--unknown'], ['source_file_that_does_not_exist.txt.gz']):
            result = subprocess.run([BASH, shell_path(SCRIPT), *args], capture_output=True, text=True)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertNotIn('Started:', result.stdout)


if __name__ == '__main__':
    unittest.main()
