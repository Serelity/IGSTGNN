"""Actual tiny EVENT checkpoint: replay gates, hook restoration and protection."""
import copy
import json
import logging
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.chronological import probe_min_forecast_fusion_relative as relative
from experiments.chronological import evaluate_min_forecast_fusion as evaluation
from src.models.incident_spatial_residual import tensor_state_hash
from src.utils.igstgnn_published_events import EVENT_KEYS, encode_event
from tests import test_min_forecast_fusion_evaluation as fixtures


class RelativeProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)

    def setUp(self):
        # Reuse the real five-layer MIN fixture without running old evaluations.
        self.fixture = fixtures.ForecastFusionEvaluationTests('test_actual_nonzero_model_writes_eight_cases_and_preserves_all_state')
        self.fixture.logger = logging.getLogger('relative_probe_tests')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.data, self.graph = self.fixture.data, self.fixture.graph
        self.model = self.fixture.models['EVENT']
        self.saved = self.fixture.saved['EVENT']
        # The reused fixture has antisymmetric logits and nearly zero common
        # gain. Add a shared offset so centering changes an actual prediction.
        with torch.no_grad():
            self.model.forecast_fusion.condition[-1].bias.add_(.35)
        self.root, self.paths = self.fixture.root, self.fixture.paths
        self.roads = {'road-A': [0, 1], 'road-B': [2, 3]}
        predictions = {}
        for case in evaluation.CASES[:-1]:
            model = self.fixture.models['STATE'] if case == 'STATE_ON' else self.model
            predictions[case] = np.concatenate([evaluation.forward_case(model, batch, self.data.scaler, case)['prediction']
                for batch in self.data.loaders['val_loader'].get_iterator()])
        predictions['SOURCE_MIN'] = self.fixture.arrays['prediction'].copy()
        actual = evaluation.compute_all_metrics(torch.from_numpy(predictions['EVENT_ON']),
            torch.from_numpy(self.fixture.arrays['target']), torch.tensor(self.fixture.baseline['mask_value']))
        self.saved['best_model_state'] = copy.deepcopy(self.model.state_dict())
        self.saved['best_metric'] = actual[0]
        self.saved['history'][0].update(validation_mae=actual[0], validation_mape=actual[1], validation_rmse=actual[2])
        events = [encode_event(sample, 4) for sample in self.data.loaders['val_loader'].original_samples]
        event_arrays = {key: np.asarray([row[key] for row in events]) for key in EVENT_KEYS}
        regions = evaluation.probe.regions_for(event_arrays, self.graph, self.roads)
        normalized = {name: np.broadcast_to(mask, (len(events), 4)) for name, mask in regions.items()}
        old_analysis = evaluation.summarize(predictions, self.fixture.arrays['target'], self.fixture.arrays['valid'],
                                            regions, self.fixture.arrays['sample_indices'])
        self.info = dict(predictions=predictions, target=self.fixture.arrays['target'].copy(),
            valid=self.fixture.arrays['valid'].copy(), indices=self.fixture.arrays['sample_indices'].copy(),
            stations=np.asarray(self.data.station_ids), regions=normalized,
            fixed_hard_ids=np.asarray(old_analysis['fixed_hard_windows']['sample_indices']),
            summary=dict(engineering_only=True, source_best_epoch=28),
            comparable=dict(best_stage_epochs={'STATE': 1, 'EVENT': 1}),
            identity=dict(source=dict(reference_metrics=self.fixture.baseline['actual']), test_payload_deserialized=True),
            replay=dict(EVENT=dict(mask_value=self.fixture.baseline['mask_value'],
                expected=evaluation.selected_metrics(self.saved, self.fixture.baseline),
                batch_partition=self.fixture.baseline['batch_partition'])),
            model_state_sha256=tensor_state_hash(self.model.state_dict()))

    def run_probe(self, name='relative', info=None, saved=None):
        return relative.run_probe(self.model, self.data, saved or self.saved, info or self.info,
            self.graph, self.roads, self.root/name, self.paths)

    def test_nonzero_actual_model_writes_ten_cases_and_preserves_state(self):
        before = tensor_state_hash(self.model.state_dict())
        inputs = copy.deepcopy(next(self.data.loaders['val_loader'].get_iterator()))
        summary = self.run_probe()
        self.assertEqual(summary['status'], relative.STATUS)
        self.assertEqual(summary['new_inference_calls']['EVENT_ON'], 6)
        self.assertEqual(summary['total_new_inference_calls'], 10)
        self.assertEqual(summary['new_inference_calls']['SOURCE_MIN'], 0)
        self.assertEqual(summary['new_inference_calls']['STATE_ON'], 0)
        self.assertEqual(summary['optimizer_updates'], 0)
        self.assertFalse(summary['test_evaluated'])
        self.assertTrue(summary['on_exactly_restored'])
        self.assertEqual(tensor_state_hash(self.model.state_dict()), before)
        self.assertFalse(self.model.forecast_fusion._forward_hooks)
        output = self.root/'relative'
        for name in ('summary.json', 'analysis.json', 'mechanisms.json', 'replay.json', 'preservation.json',
                     'comparability.json', 'region_definitions.json', 'paired_predictions.npz', 'report.md',
                     'all_nodes.csv', 'horizons.csv', 'flow.csv', 'roads.csv', 'regions.csv', 'groups.csv'):
            self.assertTrue((output/name).is_file(), name)
        with np.load(output/'paired_predictions.npz', allow_pickle=False) as arrays:
            for case in evaluation.CASES:
                np.testing.assert_array_equal(arrays[case], self.info['predictions'][case])
            self.assertFalse(np.array_equal(arrays['EVENT_RELATIVE_EXACT'], arrays['EVENT_ON']))
            np.testing.assert_array_equal(arrays['fixed_hard_sample_indices'], self.info['fixed_hard_ids'])
        mechanism = json.loads((output/'mechanisms.json').read_text())
        for case in relative.RELATIVE_CASES:
            self.assertEqual(mechanism['relative_transforms'][case]['hook_calls'], 2)
            self.assertEqual(mechanism['relative_transforms'][case]['positions'], len(self.info['target'])*12*4)
        report = (output/'report.md').read_text(encoding='utf-8')
        self.assertIn('IGSTGNN', report)
        batch = next(self.data.loaders['val_loader'].get_iterator())
        self.assertEqual(evaluation.probe.input_digest(batch), evaluation.probe.input_digest(inputs))

    def test_complete_archive_replay_failure_blocks_every_relative_call(self):
        info = copy.deepcopy(self.info)
        info['predictions']['EVENT_ON'][0, 0, 0] += .01
        with patch.object(relative, 'relative_fusion') as intervene, self.assertRaisesRegex(ValueError, 'relative interventions blocked'):
            self.run_probe('bad_replay', info=info)
        intervene.assert_not_called()
        output = self.root/'bad_replay'
        self.assertFalse((output/'summary.json').exists())
        replay = json.loads((output/'replay.json').read_text())
        self.assertFalse(replay['archived_prediction_exact'])
        self.assertGreater(replay['archived_prediction_abs_max'], 0)
        self.assertEqual(json.loads((output/'failure.json').read_text())['new_inference_calls']['EVENT_ON'], 2)

    def test_selected_identity_and_metric_change_are_blocked_before_intervention(self):
        for name, mutate in (
            ('model', lambda info: info.update(model_state_sha256='a'*64)),
            ('stage', lambda info: info['comparable']['best_stage_epochs'].update(EVENT=2)),
            ('certificate', lambda info: info['replay']['EVENT']['expected'].__setitem__(0, 123.))):
            with self.subTest(name=name):
                info = copy.deepcopy(self.info)
                mutate(info)
                with patch.object(relative, 'relative_fusion') as intervene, self.assertRaises(ValueError):
                    self.run_probe('bad_'+name, info=info)
                intervene.assert_not_called()

    def test_station_mask_and_region_changes_are_rejected(self):
        for name, mutate in (
            ('station', lambda info: info.update(stations=info['stations'][::-1])),
            ('mask', lambda info: info.update(valid=~info['valid'])),
            ('region', lambda info: info['regions'].update(direct_report_nodes=~info['regions']['direct_report_nodes']))):
            with self.subTest(name=name):
                info = copy.deepcopy(self.info)
                mutate(info)
                with patch.object(relative, 'relative_fusion') as intervene, self.assertRaises(ValueError):
                    self.run_probe('bad_'+name, info=info)
                intervene.assert_not_called()

    def test_restoration_failure_removes_hook_and_blocks_completion(self):
        calls, original = [0], evaluation.forward_case
        def faulty(*args, **kwargs):
            row = original(*args, **kwargs)
            calls[0] += 1
            if calls[0] == 4:  # Two complete replay batches, exact, then restored ON.
                row['prediction'] += 1.
            return row
        with patch.object(evaluation, 'forward_case', side_effect=faulty), self.assertRaisesRegex(ValueError, 'ON restoration failed'):
            self.run_probe('bad_restore')
        self.assertFalse(self.model.forecast_fusion._forward_hooks)
        self.assertFalse((self.root/'bad_restore/summary.json').exists())

    def test_source_mutation_is_detected_and_failure_keeps_diagnostic(self):
        original = evaluation.forward_case
        def faulty(*args, **kwargs):
            row = original(*args, **kwargs)
            if self.model.forecast_fusion._forward_hooks:
                self.fixture.source_file.write_text('mutated source', encoding='utf-8')
            return row
        with patch.object(evaluation, 'forward_case', side_effect=faulty), self.assertRaisesRegex(ValueError, 'Source artifacts changed'):
            self.run_probe('bad_source')
        output = self.root/'bad_source'
        self.assertFalse((output/'summary.json').exists())
        self.assertFalse(json.loads((output/'preservation.json').read_text())['source_artifacts_preserved'])
        self.assertTrue((output/'failure.json').is_file())

    def test_cli_infers_run_from_archive_and_rejects_dangerous_output(self):
        source = self.root/'archive'
        args = relative.parse_args(['--evaluation-dir', str(source), '--device', 'cpu'])
        self.assertEqual(args.run_dir, self.root.resolve())
        for path in (source, source/'child', self.root, self.root/'EVENT/probe', self.root.parent):
            args.output_dir = path
            with self.subTest(path=path), self.assertRaises(ValueError):
                relative.output_path(args, source)

    def test_aggregation_preserves_static_fields_and_extrema(self):
        total = {}
        relative.merge_relative(total, dict(schema='v1', mode='bounded', max_delta=.1,
            hook_calls=1, positions=10, output_delta_abs_max=.08, bounded_scale_min=.8))
        relative.merge_relative(total, dict(schema='v1', mode='bounded', max_delta=.1,
            hook_calls=1, positions=20, output_delta_abs_max=.09, bounded_scale_min=.9))
        self.assertEqual(total['hook_calls'], 2)
        self.assertEqual(total['positions'], 30)
        self.assertEqual(total['max_delta'], .1)
        self.assertEqual(total['output_delta_abs_max'], .09)
        self.assertEqual(total['bounded_scale_min'], .8)


if __name__ == '__main__':
    unittest.main()
