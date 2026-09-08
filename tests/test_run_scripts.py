import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


@unittest.skipUnless(os.name == "posix", "The launch scripts require Linux Bash.")
class RunScriptsTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.script_dir = self.root / "experiments" / "IGSTGNN"
        self.script_dir.mkdir(parents=True)
        source_dir = Path(__file__).resolve().parents[1] / "experiments" / "IGSTGNN"
        for name in ["run.sh", "run.slurm"]:
            if (source_dir / name).exists():
                shutil.copy2(source_dir / name, self.script_dir / name)

        self.bin_dir = self.root / "env" / "bin"
        self.bin_dir.mkdir(parents=True)
        self.calls_file = self.root / "calls.jsonl"
        # Stand in only for GPU checks and costly training at the process boundary.
        python = self.bin_dir / "python"
        python.write_text(
            f"#!{sys.executable}\n"
            "import json, os, sys\n"
            "from pathlib import Path\n"
            "args = sys.argv[1:]\n"
            "with open(os.environ['CALLS_FILE'], 'a') as f:\n"
            "    f.write(json.dumps(args) + '\\n')\n"
            "if args == ['-u', '-']:\n"
            "    sys.stdin.read()\n"
            "    sys.exit(int(os.environ.get('CUDA_CHECK_EXIT', '0')))\n"
            "if 'data/xtraffic/prepare_splits.py' in args:\n"
            "    if os.environ.get('PREPARE_EXIT'):\n"
            "        sys.exit(int(os.environ['PREPARE_EXIT']))\n"
            "    dataset = args[args.index('--dataset') + 1]\n"
            "    for split in ['train', 'val', 'test']:\n"
            "        Path('data/xtraffic', dataset, 'incident_' + split + '.npy').write_bytes(b'prepared')\n"
            "if 'experiments/IGSTGNN/main.py' in args:\n"
            "    sys.exit(int(os.environ.get('TRAIN_EXIT', '0')))\n",
            encoding="utf-8",
        )
        python.chmod(0o755)
        self.env = os.environ.copy()
        self.env.update(
            PATH=str(self.bin_dir) + os.pathsep + self.env["PATH"],
            CALLS_FILE=str(self.calls_file),
        )

    def make_dataset(self, dataset, splits=True):
        directory = self.root / "data" / "xtraffic" / dataset
        directory.mkdir(parents=True)
        for name in [
            "adj_matrix.npy", "desc_mapping.json", "type_mapping.json",
            "incident_stats.npz", "sensors.csv", "incident_all.npy",
        ]:
            (directory / name).write_bytes(b"fixture")
        if splits:
            for split in ["train", "val", "test"]:
                (directory / f"incident_{split}.npy").write_bytes(b"existing")
        return directory

    def run_script(self, *args, script="run.sh"):
        return subprocess.run(
            ["bash", str(self.script_dir / script), *args],
            cwd=self.root.parent,
            env=self.env,
            capture_output=True,
            text=True,
            timeout=10,
        )

    def calls(self):
        if not self.calls_file.exists():
            return []
        return [json.loads(line) for line in self.calls_file.read_text().splitlines()]

    def test_city_presets_keep_the_reproduction_configuration(self):
        for dataset, batch_size in [("Alameda", "48"), ("Contra_Costa", "48"), ("Orange", "24")]:
            with self.subTest(dataset=dataset):
                directory = self.make_dataset(dataset)
                result = self.run_script(dataset)
                self.assertEqual(result.returncode, 0, result.stderr)
                train = self.calls()[-1]
                self.assertEqual(train[:2], ["-u", "experiments/IGSTGNN/main.py"])
                expected = {
                    "--dataset": dataset, "--bs": batch_size, "--seed": "2025",
                    "--device": "cuda:0", "--model_name": "igstgnn",
                    "--max_epochs": "100", "--patience": "20",
                    "--warm_epoch": "30", "--cl_epoch": "3",
                }
                for flag, value in expected.items():
                    self.assertIn(flag, train)
                    self.assertEqual(train[train.index(flag) + 1], value)
                self.assertIn("--incident", train)
                self.assertIn("--use_sensor_info", train)
                self.assertFalse(any("data/xtraffic/prepare_splits.py" in call for call in self.calls()))
                self.assertEqual((directory / "incident_train.npy").read_bytes(), b"existing")

    def test_prepares_missing_splits_before_training(self):
        directory = self.make_dataset("Contra_Costa", splits=False)
        result = self.run_script("Contra_Costa")
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.calls()
        self.assertEqual(calls[0], ["-u", "-"])
        self.assertEqual(calls[1], ["-u", "data/xtraffic/prepare_splits.py", "--dataset", "Contra_Costa"])
        self.assertIn("experiments/IGSTGNN/main.py", calls[2])
        self.assertTrue((directory / "incident_test.npy").exists())

    def test_rejects_partial_splits_without_overwriting(self):
        directory = self.make_dataset("Orange", splits=False)
        train_file = directory / "incident_train.npy"
        train_file.write_bytes(b"preserve")
        result = self.run_script("Orange")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(train_file.read_bytes(), b"preserve")
        self.assertEqual(self.calls(), [])

    def test_invalid_arguments_do_not_start_python(self):
        for args in [("Unknown",), ("Orange", "--bs", "48")]:
            with self.subTest(args=args):
                result = self.run_script(*args)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(self.calls(), [])

    def test_missing_metadata_does_not_start_python(self):
        directory = self.make_dataset("Contra_Costa")
        (directory / "sensors.csv").unlink()
        result = self.run_script("Contra_Costa")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("sensors.csv", result.stderr)
        self.assertEqual(self.calls(), [])

    def test_failed_cuda_check_does_not_prepare_or_train(self):
        self.make_dataset("Orange", splits=False)
        self.env["CUDA_CHECK_EXIT"] = "7"
        result = self.run_script("Orange")
        self.assertEqual(result.returncode, 7)
        self.assertEqual(self.calls(), [["-u", "-"]])

    def test_failed_preparation_does_not_train(self):
        self.make_dataset("Contra_Costa", splits=False)
        self.env["PREPARE_EXIT"] = "9"
        result = self.run_script("Contra_Costa")
        self.assertEqual(result.returncode, 9)
        self.assertFalse(any("experiments/IGSTGNN/main.py" in call for call in self.calls()))

    def test_training_failure_is_not_reported_as_success(self):
        self.make_dataset("Orange")
        self.env["TRAIN_EXIT"] = "23"
        result = self.run_script("Orange")
        self.assertEqual(result.returncode, 23)

    def test_slurm_uses_activated_environment_and_submission_directory(self):
        self.make_dataset("Contra_Costa")
        self.env["PATH"] = os.environ["PATH"]
        self.env.update(
            CONDA_PREFIX=str(self.bin_dir.parent),
            SLURM_SUBMIT_DIR=str(self.root),
            SLURM_JOB_ID="123",
        )
        result = self.run_script("Contra_Costa", script="run.slurm")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--dataset", self.calls()[-1])
        self.assertIn("Contra_Costa", self.calls()[-1])

    def test_slurm_requires_an_activated_conda_environment(self):
        self.env.pop("CONDA_PREFIX", None)
        self.env.update(SLURM_SUBMIT_DIR=str(self.root), SLURM_JOB_ID="123")
        result = self.run_script("Orange", script="run.slurm")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("conda activate igstgnn", result.stderr)
        self.assertEqual(self.calls(), [])


if __name__ == "__main__":
    unittest.main()
