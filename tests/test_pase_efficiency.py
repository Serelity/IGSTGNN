import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

PATH = Path(__file__).resolve().parents[1]/'scripts/benchmark_pase_compact.py'
spec = importlib.util.spec_from_file_location('pase_efficiency', PATH)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


class EfficiencyTests(unittest.TestCase):
    def test_latency_and_throughput_use_complete_batch(self):
        result = benchmark.timing_summary([10., 20., 30.], 48)
        self.assertEqual(result['mean_ms'], 20.)
        self.assertEqual(result['median_ms'], 20.)
        self.assertAlmostEqual(result['p95_ms'], 29.)
        self.assertEqual(result['windows_per_second'], 2400.)
        self.assertEqual(result['measured_forwards'], 3)
        for bad in ([], [0.], [-1.], [float('nan')], [float('inf')]):
            with self.assertRaises(ValueError):
                benchmark.timing_summary(bad, 48)

    def test_original_tensor_sensor_fields_use_clone(self):
        class TensorLike:
            def __getitem__(self, value):
                self.stop = value.stop
                return self
            def clone(self):
                self.cloned = True
                return self
            def contiguous(self):
                return self
        value = TensorLike()
        benchmark.slice_batch({'sensor_data': {'road_width': value}}, 1)
        self.assertTrue(value.cloned)
        self.assertEqual(value.stop, 1)

    def rows(self):
        return [dict(model=model, batch_size=batch, history_sha256='x',
            station_ids_sha256='s', sample_ids=list(range(batch)),
            runtime={'device_name': 'V100', 'dtype': 'float32'}, warmup=20,
            repeats=50, rounds=3, output_shape=[batch, 12, 496, 1])
            for batch in (1, 48) for model in ('IGSTGNN', 'PASE-Net')]

    def test_pairing_rejects_different_inputs_or_settings(self):
        rows = self.rows()
        benchmark.paired_check(rows)
        for field in ('history_sha256', 'station_ids_sha256', 'sample_ids', 'runtime',
                      'warmup', 'repeats', 'rounds', 'output_shape'):
            changed = copy.deepcopy(rows)
            changed[1][field] = None
            with self.assertRaisesRegex(ValueError, field):
                benchmark.paired_check(changed)
        with self.assertRaises(ValueError):
            benchmark.paired_check(rows[:3])

    def test_parent_uses_four_separate_processes_and_writes_report(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ('run/identity.json', 'model.pt', 'training.log',
                         'code/experiments/pase_compact_v3/model.py'):
                p = root/name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.touch()
            output = root/'output'
            calls = []
            def fake_process(command, **kwargs):
                calls.append(command)
                model = command[command.index('--worker')+1]
                batch = int(command[command.index('--batch-size')+1])
                row = next(r for r in self.rows() if r['model'] == model and r['batch_size'] == batch)
                row.update(parameters=73357 if model == 'PASE-Net' else 455220,
                    **benchmark.timing_summary([20., 20.], batch),
                    peak_allocated_mib=100., additional_forward_mib=20.)
                benchmark.save_json(output/(model.replace('-', '_')+f'_b{batch}.json'), row)
            with patch.object(benchmark.subprocess, 'run', side_effect=fake_process):
                benchmark.main(['--code-dir', str(root/'code'), '--data-dir', str(root/'data'),
                    '--pase-run', str(root/'run'), '--original-checkpoint', str(root/'model.pt'),
                    '--original-log', str(root/'training.log'), '--output-dir', str(output)])
            self.assertEqual(len(calls), 4)
            self.assertTrue((output/'summary.json').is_file())
            self.assertIn('峰值显存', (output/'report.md').read_text(encoding='utf-8'))
            self.assertEqual(len((output/'results.csv').read_text().splitlines()), 5)


if __name__ == '__main__':
    unittest.main()
