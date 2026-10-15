"""CPU-only passive timing checks; no model-speed or training claims."""
import csv
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import torch

from experiments.chronological.original_igstgnn_module_timing import profile_full_modules, write_module_timing
from tests import test_original_igstgnn_interventions as fixture


class OriginalModuleTimingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.OriginalInterventionTests.setUpClass.__func__(cls)

    @classmethod
    def tearDownClass(cls):
        fixture.OriginalInterventionTests.tearDownClass.__func__(cls)

    def setUp(self):
        fixture.OriginalInterventionTests.setUp(self)

    assertTreeEqual = fixture.OriginalInterventionTests.assertTreeEqual
    hook_inventory = fixture.OriginalInterventionTests.hook_inventory

    def test_default_all_modules_cpu_inclusive_counts_and_native_prediction(self):
        with torch.inference_mode():
            expected = self.model(self.x, incident_data=self.incident)
        prediction, report = profile_full_modules(self.model, self.x, self.incident, None, 'cpu')
        self.assertTrue(torch.equal(prediction, expected))
        self.assertTrue(prediction.is_inference())
        self.assertEqual(report['case'], 'FULL')
        self.assertEqual(report['repetitions'], 2)
        self.assertEqual(report['clock'], 'perf_counter_ns')
        self.assertEqual(report['cuda_synchronizations'], 0)
        rows = {row['path']: row for row in report['modules']}
        self.assertEqual(set(rows), set(dict(self.model.named_modules())))
        self.assertEqual(rows['']['calls'], 2)
        self.assertEqual(rows['icsf_module']['calls_per_repetition'], [1, 1])
        self.assertEqual(rows['tiid_module']['calls'], 2)
        self.assertEqual(rows['layers.0.dif_layer.localized_st_conv']['calls'], 8)
        self.assertEqual(rows['layers.0.inh_layer.rnn_layer.gru_cell']['calls'], 26)
        for layer in range(5):
            unused = rows['layers.{}.dif_layer.localized_st_conv.bn'.format(layer)]
            self.assertEqual(unused['calls'], 0)
            self.assertEqual(unused['inclusive_seconds_total'], 0.)
            self.assertIsNone(unused['inclusive_seconds_mean_per_call'])
        # MultiheadAttention uses out_proj weights functionally; no out_proj
        # module call is observed, which does not prove its weights are unused.
        self.assertEqual(rows['layers.0.inh_layer.transformer_layer.multi_head_self_attention.out_proj']['calls'], 0)
        self.assertTrue(report['zero_calls_does_not_prove_parameters_unused'])
        self.assertTrue(report['parent_child_inclusive_times_overlap'])
        self.assertTrue(report['module_times_must_not_be_summed'])
        for row in report['modules']:
            self.assertEqual(row['calls'], sum(row['calls_per_repetition']))
            self.assertAlmostEqual(row['inclusive_seconds_total'], sum(row['inclusive_seconds_per_repetition']))
            if row['calls']:
                self.assertGreater(row['inclusive_seconds_mean_per_call'], 0.)
                self.assertLessEqual(row['inclusive_seconds_total'], rows['']['inclusive_seconds_total'])
        json.dumps(report, allow_nan=False)

    def test_state_inputs_rng_modes_gradients_and_existing_hooks_are_unchanged(self):
        before_state = fixture.clone_tree(self.model.state_dict())
        before_inputs = fixture.clone_tree((self.x, self.incident))
        before_graphs = [fixture.clone_tree(layer.dif_layer.localized_st_conv.pre_defined_graph)
                         for layer in self.model.layers]
        before_requires_grad = [parameter.requires_grad for parameter in self.model.parameters()]
        rng = torch.get_rng_state().clone()
        observed = []
        handle = self.model.embedding.register_forward_hook(lambda _m, _a, _o: observed.append(1))
        hooks = self.hook_inventory()
        try:
            profile_full_modules(self.model, self.x, self.incident, None, 'cpu', repetitions=2)
            self.assertEqual(self.hook_inventory(), hooks)
            self.assertEqual(observed, [1, 1])
        finally:
            handle.remove()
        self.assertTreeEqual(self.model.state_dict(), before_state)
        self.assertTreeEqual((self.x, self.incident), before_inputs)
        self.assertTreeEqual([layer.dif_layer.localized_st_conv.pre_defined_graph
                              for layer in self.model.layers], before_graphs)
        self.assertEqual([parameter.requires_grad for parameter in self.model.parameters()], before_requires_grad)
        self.assertTrue(all(parameter.grad is None for parameter in self.model.parameters()))
        self.assertTrue(all(not module.training for module in self.model.modules()))
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))

    def test_selected_paths_and_empty_selection_are_supported(self):
        paths = ['icsf_module', 'layers.0.dif_layer.localized_st_conv.bn', 'out_fc_2']
        _, report = profile_full_modules(self.model, self.x, self.incident, None, 'cpu:0',
                                         repetitions=1, selected_paths=paths)
        self.assertEqual([row['path'] for row in report['modules']], paths)
        self.assertEqual([row['calls'] for row in report['modules']], [1, 0, 1])
        self.assertEqual(report['device'], 'cpu')
        prediction, empty = profile_full_modules(self.model, self.x, self.incident, None, 'cpu',
                                                 repetitions=1, selected_paths=[])
        self.assertEqual(empty['modules'], [])
        self.assertEqual(len(empty['forward_seconds']), 1)
        self.assertEqual(prediction.shape, (2, 12, 4, 1))

    def test_all_timing_hooks_restore_after_forward_exception(self):
        before = self.hook_inventory()
        mode = torch.is_inference_mode_enabled()
        with patch.object(self.model.out_fc_2, 'forward', side_effect=RuntimeError('synthetic timing failure')):
            with self.assertRaisesRegex(RuntimeError, 'synthetic timing failure'):
                profile_full_modules(self.model, self.x, self.incident, None, 'cpu')
        self.assertEqual(self.hook_inventory(), before)
        self.assertEqual(torch.is_inference_mode_enabled(), mode)

    def test_invalid_repetitions_paths_mode_and_report_fail_before_hooks(self):
        before = self.hook_inventory()
        for repetitions in (0, -1, 1.5, True):
            with self.assertRaisesRegex(ValueError, 'positive integer'):
                profile_full_modules(self.model, self.x, self.incident, None, 'cpu', repetitions)
        for paths in ('embedding', ['not.a.module'], ['embedding', 'embedding'], [1]):
            with self.assertRaises(ValueError):
                profile_full_modules(self.model, self.x, self.incident, None, 'cpu', selected_paths=paths)
        with self.assertRaisesRegex(ValueError, 'supplied original report'):
            profile_full_modules(self.model, self.x, None, None, 'cpu')
        with self.assertRaisesRegex(ValueError, 'CPU or CUDA'):
            profile_full_modules(self.model, self.x, self.incident, None, 'meta')
        self.model.layers[0].train()
        with self.assertRaisesRegex(ValueError, 'eval mode'):
            profile_full_modules(self.model, self.x, self.incident, None, 'cpu')
        self.model.eval()
        parameter = next(self.model.parameters())
        parameter.grad = torch.ones_like(parameter)
        with self.assertRaisesRegex(ValueError, 'absent parameter gradients'):
            profile_full_modules(self.model, self.x, self.incident, None, 'cpu')
        parameter.grad = None
        self.assertEqual(self.hook_inventory(), before)

    def test_mock_cuda_syncs_before_first_and_after_each_complete_forward(self):
        # Exercise event bookkeeping without initializing CUDA or running a
        # GPU model; actual CUDA timings remain the runner's responsibility.
        recorded_streams = []
        recorded_sync_counts = []

        class Event:
            def __init__(self, enable_timing):
                self.stamp = None

            def record(self, stream):
                recorded_streams.append(stream)
                recorded_sync_counts.append(stream.synchronizations)
                self.stamp = time.perf_counter_ns()

            def elapsed_time(self, other):
                return (other.stamp-self.stamp)/1e6

        class Stream:
            cuda_stream = 123

            def __init__(self):
                self.synchronizations = 0

            def synchronize(self):
                self.synchronizations += 1

        stream = Stream()
        target = torch.device('cuda:1')
        with patch('experiments.chronological.original_igstgnn_module_timing._device_for', return_value=target), \
             patch('torch.cuda.current_stream', return_value=stream) as current, \
             patch('torch.cuda.Event', side_effect=Event):
            prediction, report = profile_full_modules(self.model, self.x, self.incident, None, target,
                                                      repetitions=2, selected_paths=['', 'embedding'])
        current.assert_called_once_with(device=target)
        self.assertEqual(stream.synchronizations, 3)
        self.assertEqual(report['cuda_synchronizations'], 3)
        self.assertEqual(report['cuda_synchronizations_before_first_forward'], 1)
        self.assertEqual(report['cuda_synchronizations_after_complete_forward'], 2)
        self.assertEqual(report['cuda_stream'], 123)
        self.assertEqual(len(recorded_streams), 12)
        self.assertTrue(all(value is stream for value in recorded_streams))
        self.assertEqual(recorded_sync_counts, [1]*6+[2]*6)
        self.assertEqual(prediction.device.type, 'cpu')

    def test_writer_preserves_report_and_writes_all_rows_with_actual_class_summary(self):
        _, report = profile_full_modules(self.model, self.x, self.incident, None, 'cpu', repetitions=1)
        with tempfile.TemporaryDirectory(dir=self.data.name) as temporary:
            directory = Path(temporary)
            existing = directory/'report.md'
            original_text = 'Existing audit results must remain intact.\n'
            existing.write_text(original_text, encoding='utf-8')
            section = write_module_timing(directory, report)
            self.assertEqual(existing.read_text(encoding='utf-8'), original_text)
            self.assertEqual(json.loads((directory/'module_timings.json').read_text(encoding='utf-8')), report)
            with (directory/'module_timings.csv').open(encoding='utf-8', newline='') as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), len(report['modules']))
            indexed = {row['path']: row for row in rows}
            unused = indexed['layers.0.dif_layer.localized_st_conv.bn']
            self.assertEqual(unused['calls'], '0')
            self.assertEqual(unused['inclusive_seconds_mean_per_call'], '')
            self.assertEqual(json.loads(unused['calls_per_repetition']), [0])
            self.assertIn('父模块包含子模块', section)
            self.assertIn('不能相加', section)
            self.assertIn('DecoupleLayer', section)
            self.assertIn('IncidentContextSpatialFusion', section)
            self.assertIn('TemporalIncidentImpactDecay', section)
            self.assertNotIn('DecoupleBlock', section)
            self.assertNotIn('ICStateFusion', section)
            self.assertNotIn('| BatchNorm', section)
            before = [(directory/name).read_bytes() for name in ('module_timings.json', 'module_timings.csv')]
            with self.assertRaises(FileExistsError):
                write_module_timing(directory, report)
            self.assertEqual(before, [(directory/name).read_bytes() for name in ('module_timings.json', 'module_timings.csv')])

    def test_writer_refuses_partial_prior_artifacts_and_nonfinite_json(self):
        _, report = profile_full_modules(self.model, self.x, self.incident, None, 'cpu',
                                         repetitions=1, selected_paths=[''])
        with tempfile.TemporaryDirectory(dir=self.data.name) as temporary:
            directory = Path(temporary)
            csv_path = directory/'module_timings.csv'
            csv_path.write_text('prior partial artifact', encoding='utf-8')
            with self.assertRaises(FileExistsError):
                write_module_timing(directory, report)
            self.assertFalse((directory/'module_timings.json').exists())
            self.assertEqual(csv_path.read_text(encoding='utf-8'), 'prior partial artifact')
        with tempfile.TemporaryDirectory(dir=self.data.name) as temporary:
            invalid = dict(report, forward_seconds_mean=float('nan'))
            with self.assertRaises(ValueError):
                write_module_timing(temporary, invalid)
            self.assertEqual(list(Path(temporary).iterdir()), [])


if __name__ == '__main__':
    unittest.main()
