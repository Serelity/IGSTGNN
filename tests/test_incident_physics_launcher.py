"""Exercise actual Bash parsing/help without needing Slurm or an environment."""

from pathlib import Path
import os
import re
import shutil
import subprocess
import unittest


def find_bash():
    if os.name != 'nt':
        return shutil.which('bash')
    candidates = []
    for name in ('bash', 'git'):
        found = shutil.which(name)
        if found:
            location = Path(found).resolve()
            candidates.append(location if name == 'bash' else location.parent / 'bash.exe')
            for parent in location.parents:
                candidates += [parent / 'bin/bash.exe', parent / 'usr/bin/bash.exe']
    for candidate in candidates:
        if candidate.is_file() and ((candidate.parent / 'msys-2.0.dll').is_file()
                                    or (candidate.parent.parent / 'usr/bin/msys-2.0.dll').is_file()):
            return str(candidate)
    return None


def shell_path(path):
    value = Path(path).resolve().as_posix()
    return '/' + value[0].lower() + value[2:] if os.name == 'nt' else value


BASH = find_bash()

SCRIPT = Path(__file__).resolve().parents[1] / 'experiments/chronological/run_incident_physics_h1.sh'


class PhysicsLauncherTests(unittest.TestCase):
    def test_no_git_network_install_or_scheduler_submission(self):
        text = SCRIPT.read_text(encoding='utf-8')
        self.assertIsNone(re.search(r'\b(git|curl|wget|pip|sbatch|srun)\b', text))
        self.assertNotIn('\r', SCRIPT.read_bytes().decode('utf-8'))

    @unittest.skipUnless(BASH, 'Bash is required')
    def test_shell_syntax_and_help(self):
        result = subprocess.run([BASH, '-n', shell_path(SCRIPT)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        result = subprocess.run([BASH, shell_path(SCRIPT), '--help'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('No full training', result.stdout)

    @unittest.skipUnless(BASH, 'Bash is required')
    def test_extra_arguments_fail_before_any_execution(self):
        result = subprocess.run([BASH, shell_path(SCRIPT), 'a', 'b'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)


if __name__ == '__main__':
    unittest.main()
