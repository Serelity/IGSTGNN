import base64
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


NUMPY_2_OBJECT_ARRAY = base64.b64decode(
    "k05VTVBZAQB2AHsnZGVzY3InOiAnfE8nLCAnZm9ydHJhbl9vcmRlcic6IEZhbHNlLCAnc2hhcGUnOiAoNCwpLCB9ICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAqABJVQAQAAAAAAAIwWbnVtcHkuX2NvcmUubXVsdGlhcnJheZSMDF9yZWNvbnN0cnVjdJSTlIwFbnVtcHmUjAduZGFycmF5lJOUSwCFlEMBYpSHlFKUKEsBSwSFlGgDjAVkdHlwZZSTlIwCTziUiYiHlFKUKEsDjAF8lE5OTkr/////Sv////9LP3SUYoldlCh9lIwFdmFsdWWUaAJoBUsAhZRoB4eUUpQoSwFLAYWUaAyMAmY0lImIh5RSlChLA4wBPJROTk5K/////0r/////SwB0lGKJQwQAAIA/lHSUYnN9lGgUaAJoBUsAhZRoB4eUUpQoSwFLAYWUaBuJQwQAAABAlHSUYnN9lGgUaAJoBUsAhZRoB4eUUpQoSwFLAYWUaBuJQwQAAEBAlHSUYnN9lGgUaAJoBUsAhZRoB4eUUpQoSwFLAYWUaBuJQwQAAIBAlHSUYnNldJRiLg=="
)


class PrepareSplitsTest(unittest.TestCase):
    def test_loads_object_array_saved_by_numpy_2(self):
        repo_root = Path(__file__).resolve().parents[1]

        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = Path(temp_dir)
            dataset_dir = data_root / "Alameda"
            dataset_dir.mkdir()
            (dataset_dir / "incident_all.npy").write_bytes(NUMPY_2_OBJECT_ARRAY)
            np.savez(
                dataset_dir / "incident_stats.npz",
                mean=np.float32(0.0),
                std=np.float32(1.0),
                normalized=np.bool_(True),
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(repo_root / "data" / "xtraffic" / "prepare_splits.py"),
                    "--dataset",
                    "Alameda",
                    "--data_root",
                    str(data_root),
                    "--train_ratio",
                    "0.5",
                    "--val_ratio",
                    "0.25",
                    "--test_ratio",
                    "0.25",
                ],
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            expected = {"train": [1.0, 2.0], "val": [3.0], "test": [4.0]}
            for split, expected_values in expected.items():
                samples = np.load(
                    dataset_dir / f"incident_{split}.npy", allow_pickle=True
                )
                actual_values = [sample["value"].item() for sample in samples]
                self.assertEqual(actual_values, expected_values)


if __name__ == "__main__":
    unittest.main()
