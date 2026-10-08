"""CPU model checks for independent layer interventions, replay guards and artifacts."""

from contextlib import redirect_stdout
import copy
import csv
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.chronological import probe_incident_routing_p2 as probe
from experiments.chronological import probe_incident_routing_p2_layers as layers
from test_incident_routing_p2_probe import tiny_model, TinyValidation


class LayerGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(321)
        self.model = tiny_model(layers=5).requires_grad_(False)
        self.data = TinyValidation()
        self.device = torch.device('cpu')

    def forward(self, model):
        with torch.inference_mode():
            return model(self.data.history, incident_data=self.data.incident)

    def saved(self):
        on, _, _ = probe.infer(self.model, self.data, 1, self.device, 'on')
        return {'fixed': copy.deepcopy(on), 'acdg': on}

    def execute(self, saved, output):
        return layers.layerwise_inference(self.model, self.data, 1, self.device, saved, output)

    def test_each_of_five_cases_matches_independent_zero_projection_and_restores_all_on(self):
        before = probe.train.state_sha256(self.model.state_dict())
        original = self.forward(self.model)
        for case in layers.case_plan(self.model):
            with self.subTest(case=case['case']):
                expected = copy.deepcopy(self.model)
                projection = expected.layers[case['layer_number'] - 1].estimation_gate.incident_condition.mlp[-1]
                with torch.no_grad():
                    projection.weight.zero_()
                    projection.bias.zero_()
                with probe.GateIntervention(self.model, 'layer_off', case['disabled_layer']) as intervention:
                    actual = self.forward(self.model)
                    rows = intervention.results(2, 1)
                torch.testing.assert_close(actual, self.forward(expected), rtol=0, atol=0)
                layers.verify_gate_statistics(rows, case['disabled_layer'])
                associated = [r for r in rows if r['region'] == 'associated_nodes']
                self.assertEqual(sum(r['branch_disabled'] for r in associated), 1)
                self.assertTrue(all(r['applied_delta_abs_mean'] > 0 for r in associated if not r['branch_disabled']))
                torch.testing.assert_close(self.forward(self.model), original, rtol=0, atol=0)
                self.assertEqual(before, probe.train.state_sha256(self.model.state_dict()))

    def test_rejects_missing_unknown_or_misplaced_layer_and_wrong_architecture(self):
        for mode, target in [('layer_off', None), ('layer_off', 'layers.5.estimation_gate'),
                             ('on', 'layers.0.estimation_gate'), ('off', 'layers.0.estimation_gate')]:
            with self.subTest(mode=mode, target=target), self.assertRaises(ValueError):
                probe.GateIntervention(self.model, mode, target)
        with self.assertRaisesRegex(ValueError, 'five ordered'):
            layers.case_plan(tiny_model())

    def test_single_layer_exception_removes_hooks(self):
        original = self.forward(self.model)
        with self.assertRaisesRegex(RuntimeError, 'intentional'):
            with probe.GateIntervention(self.model, 'layer_off', 'layers.3.estimation_gate'):
                self.forward(self.model)
                raise RuntimeError('intentional')
        for module in self.model.modules():
            self.assertFalse(module._forward_hooks)
        torch.testing.assert_close(self.forward(self.model), original, rtol=0, atol=0)

    def test_all_seven_passes_write_readable_paired_artifacts_without_training(self):
        saved = self.saved()
        before = probe.train.state_sha256(self.model.state_dict())
        with tempfile.TemporaryDirectory(prefix='p2-layers-') as directory, \
                patch.object(torch.optim, 'Adam', side_effect=AssertionError('Optimizer forbidden')), \
                patch.object(probe, 'infer', wraps=probe.infer) as infer, redirect_stdout(io.StringIO()):
            output = Path(directory)
            result = self.execute(saved, output)
            self.assertEqual(result['inference_passes'], 7)
            self.assertEqual([call.args[4] for call in infer.call_args_list], ['on'] + ['layer_off'] * 5 + ['on'])
            self.assertEqual([call.kwargs['disabled_layer'] for call in infer.call_args_list[1:6]],
                             [case['disabled_layer'] for case in layers.case_plan(self.model)])
            self.assertTrue(result['on_replay']['passed'])
            self.assertEqual(result['on_restoration']['max_abs_prediction_difference'], 0.)
            self.assertEqual(len(result['gate_statistics']), 70)
            for case in result['cases']:
                path = output / case['prediction_file']
                stored = probe.read_predictions(path, probe.digest(path))
                probe.paired_predictions(saved['acdg'], stored)
                self.assertEqual(stored['prediction'].shape, (2, 12, 3, 1))
            for filename, count in [('layer_comparisons.csv', 15), ('per_horizon.csv', 180),
                                    ('gate_statistics.csv', 70)]:
                with (output / filename).open(newline='', encoding='utf-8') as stream:
                    self.assertEqual(len(list(csv.DictReader(stream))), count)
        self.assertEqual(before, probe.train.state_sha256(self.model.state_dict()))
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))

    def test_failed_initial_replay_blocks_every_layer_off_case(self):
        saved = self.saved()
        changed = copy.deepcopy(saved['acdg'])
        changed['prediction'] += 1.
        with patch.object(probe, 'infer', return_value=(changed, [], 0.)) as infer, \
                patch.object(probe.train, 'atomic_json'), patch.object(probe.train, 'atomic_npz') as writer, \
                redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, 'no layer-off cases'):
                self.execute(saved, Path('unused-probe-output'))
        self.assertEqual(infer.call_count, 1)
        writer.assert_not_called()

    def test_failed_final_restoration_blocks_completion(self):
        saved = self.saved()
        real_infer = probe.infer
        on_calls = 0

        def changed_restoration(*args, **kwargs):
            nonlocal on_calls
            arrays, rows, seconds = real_infer(*args, **kwargs)
            if args[4] == 'on':
                on_calls += 1
                if on_calls == 2:
                    arrays['prediction'] += .01
            return arrays, rows, seconds

        with patch.object(probe, 'infer', side_effect=changed_restoration), \
                patch.object(probe.train, 'atomic_json'), patch.object(probe.train, 'atomic_npz'), \
                patch.object(probe, 'write_csv') as writer, redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, 'exactly restore'):
                self.execute(saved, Path('unused-probe-output'))
        self.assertEqual(on_calls, 2)
        writer.assert_not_called()

    def test_region_and_horizon_signs_are_off_minus_on_and_axes_are_checked(self):
        on = self.saved()['acdg']
        on['prediction'].fill(2.)
        off = copy.deepcopy(on)
        off['prediction'][on['associated']] = 3.
        off['prediction'][~on['associated']] = 1.
        case = layers.case_plan(self.model)[0]
        regions, rows, horizons = layers.compare_case(on, on, off, case)
        for key in layers.METRICS:
            self.assertEqual(regions['associated_nodes']['layer_off_minus_all_on'][key], 1.)
            self.assertEqual(regions['nonassociated_nodes']['layer_off_minus_all_on'][key], -1.)
        self.assertEqual(len(rows), 3)
        self.assertEqual(len(horizons), 36)
        off['station_ids'] = off['station_ids'][::-1].copy()
        with self.assertRaisesRegex(ValueError, 'station_ids'):
            layers.compare_case(on, on, off, case)

    def test_empty_region_is_reported_as_unavailable_not_zero(self):
        on = self.saved()['acdg']
        on['associated'].fill(False)
        result, _, _ = layers.compare_case(on, on, copy.deepcopy(on), layers.case_plan(self.model)[0])
        self.assertTrue(all(value is None for value in result['associated_nodes']['layer_off_minus_all_on'].values()))

    def test_gate_audit_rejects_an_additional_disabled_branch(self):
        _, rows, _ = probe.infer(self.model, self.data, 1, self.device, 'off')
        with self.assertRaisesRegex(ValueError, 'Unexpected layer'):
            layers.verify_gate_statistics(rows, 'layers.0.estimation_gate')


if __name__ == '__main__':
    unittest.main()
