"""Actual small five-layer MIN fixture: paired replay, scopes and protection."""
import copy
import csv
import json
import logging
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.chronological import evaluate_min_forecast_fusion as evaluation
from experiments.chronological import train_min_forecast_fusion as training
from experiments.chronological import train_internal_spatial as legacy
from src.models.incident_spatial_forecast_fusion import from_minimal
from src.models.incident_spatial_residual import tensor_state_hash
from src.utils.igstgnn_published_events import EventPublishedData
from src.utils.metrics import compute_all_metrics
from tests.test_incident_spatial import graph
from tests.test_min_event_budget_data import event_fixture
from tests.test_min_event_budget_runner import tiny_minimal


class ForecastFusionEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        cls.logger = logging.getLogger('forecast_fusion_evaluation_test')
        cls.logger.addHandler(logging.NullHandler())

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=evaluation.REPO)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        raw = self.root/'raw'
        raw.mkdir()
        _, protocol = event_fixture(raw)
        self.data = EventPublishedData(raw, protocol, self.logger, check=True)
        self.graph = graph(4)
        self.core, _ = tiny_minimal(SimpleNamespace(seed=2025), 4, self.graph, 'cpu')
        self.core.eval()
        self.arrays, reference = training.validation_predictions(self.core, self.data, torch.device('cpu'))
        self.baseline = dict(reference, engineering_only=True, validation_rows=len(self.arrays['target']), source_best_epoch=28)
        self.models, self.saved = {}, {}
        for arm, mode in (('STATE', 'state'), ('EVENT', 'event')):
            model, _ = from_minimal(self.core, mode=mode, width=12)
            with torch.no_grad():
                last = model.forecast_fusion.condition[-1]
                scales = torch.linspace(-.8, .8, 10)
                last.weight.copy_(scales[:, None]*torch.linspace(-.3, .5, last.weight.shape[1])[None])
                last.bias.copy_(scales)
            model.eval()
            self.models[arm] = model
            predictions = []
            for batch in self.data.loaders['val_loader'].get_iterator():
                predictions.append(evaluation.forward_case(model, batch, self.data.scaler, arm+'_ON')['prediction'])
            prediction = np.concatenate(predictions)
            actual = compute_all_metrics(torch.from_numpy(prediction), torch.from_numpy(self.arrays['target']), torch.tensor(self.baseline['mask_value']))
            self.saved[arm] = dict(completed_epoch=1, best_epoch=1, best_metric=actual[0], global_updates=2,
                history=[dict(epoch=1, validation_mae=actual[0], validation_mape=actual[1], validation_rmse=actual[2],
                              train_order_sha256='a'*64)], best_model_state=copy.deepcopy(model.state_dict()))
        self.source_file = self.root/'protected_source.txt'
        self.source_file.write_text('immutable source', encoding='utf-8')
        self.paths = {'source': self.source_file}

    def run_probe(self, name='evaluation', *, saved=None, arrays=None, baseline=None):
        return evaluation.run_probe(self.models, self.data, self.saved if saved is None else saved,
            self.graph, {'road-A': [0, 1], 'road-B': [2, 3]}, self.root/name,
            self.baseline if baseline is None else baseline, self.arrays if arrays is None else arrays, self.paths)

    def test_actual_nonzero_model_writes_eight_cases_and_preserves_all_state(self):
        hashes = {arm: tensor_state_hash(model.state_dict()) for arm, model in self.models.items()}
        inputs = copy.deepcopy(next(self.data.loaders['val_loader'].get_iterator()))
        source_bytes = self.source_file.read_bytes()
        summary = self.run_probe()
        self.assertEqual(summary['status'], 'MIN_FORECAST_FUSION_EVALUATION_PASS')
        self.assertEqual(set(summary['cases']), set(evaluation.CASES))
        self.assertEqual(summary['new_inference_calls']['SOURCE_MIN'], 0)
        self.assertEqual(summary['optimizer_updates'], 0)
        self.assertFalse(summary['test_evaluated'])
        self.assertTrue(summary['on_exactly_restored'])
        self.assertTrue(summary['training_comparability']['same_completed_training_budget'])
        self.assertEqual(self.source_file.read_bytes(), source_bytes)
        self.assertEqual({arm: tensor_state_hash(model.state_dict()) for arm, model in self.models.items()}, hashes)
        output = self.root/'evaluation'
        for filename in ('analysis.json', 'comparability.json', 'replay.json', 'mechanisms.json', 'preservation.json', 'report.md',
                         'paired_predictions.npz', 'weights.csv', 'regions.csv', 'strata.csv', 'horizons.csv', 'region_definitions.json'):
            self.assertTrue((output/filename).is_file(), filename)
        with np.load(output/'paired_predictions.npz', allow_pickle=False) as archive:
            self.assertFalse(np.array_equal(archive['EVENT_ON'], archive['EVENT_REPORT_OFF']))
            self.assertFalse(np.array_equal(archive['EVENT_ON'], archive['EVENT_FUSION_OFF']))
            self.assertFalse(np.array_equal(archive['EVENT_ON'], archive['EVENT_UNIFORM']))
            np.testing.assert_array_equal(archive['SOURCE_MIN'], self.arrays['prediction'])
            np.testing.assert_array_equal(archive['station_ids'], self.data.station_ids)
            # Report effects are final-node-local in this architecture.
            supported = np.broadcast_to(np.array([True, False, True, False])[None, None], archive['target'].shape)
            np.testing.assert_array_equal(archive['EVENT_ON'][~supported], archive['EVENT_REPORT_OFF'][~supported])
        with (output/'weights.csv').open(encoding='utf-8-sig', newline='') as stream:
            weights = list(csv.DictReader(stream))
        self.assertEqual(len(weights), 7*12*10)
        self.assertEqual({row['component'] for row in weights}, set(evaluation.COMPONENT_ORDER))
        self.assertEqual({int(row['horizon']) for row in weights}, set(range(1, 13)))
        report = (output/'report.md').read_text(encoding='utf-8')
        self.assertIn('工程检查：True', report)
        self.assertIn('不构成正式预测效果', report)
        self.assertIn('未新增事故空间传播', report)
        self.assertIn('权重不等于重要性', report)
        self.assertIn('独立训练的通用校准', report)
        for key in inputs:
            current = next(self.data.loaders['val_loader'].get_iterator())[key]
            if isinstance(current, dict):
                for name in current:
                    np.testing.assert_array_equal(inputs[key][name], current[name])
            else:
                np.testing.assert_array_equal(inputs[key], current)

    def test_bad_complete_replay_blocks_all_interventions(self):
        saved = copy.deepcopy(self.saved)
        saved['EVENT']['best_metric'] += 1.
        saved['EVENT']['history'][0]['validation_mae'] += 1.
        seen, original = [], evaluation.forward_case
        def record(*args, **kwargs):
            seen.append(args[3])
            return original(*args, **kwargs)
        with patch.object(evaluation, 'forward_case', side_effect=record), self.assertRaisesRegex(ValueError, 'ON replay failed'):
            self.run_probe('bad_replay', saved=saved)
        self.assertTrue(seen and all(case in ('STATE_ON', 'EVENT_ON') for case in seen))
        self.assertFalse((self.root/'bad_replay/summary.json').exists())
        self.assertTrue((self.root/'bad_replay/failure.json').is_file())

    def test_restore_failure_blocks_completion(self):
        original, intervened = evaluation.forward_case, [False]
        def faulty(*args, **kwargs):
            row = original(*args, **kwargs)
            if args[3] not in ('STATE_ON', 'EVENT_ON'):
                intervened[0] = True
            elif args[3] == 'EVENT_ON' and intervened[0]:
                row['prediction'] += 1.
            return row
        with patch.object(evaluation, 'forward_case', side_effect=faulty), self.assertRaisesRegex(ValueError, 'ON restoration failed'):
            self.run_probe('bad_restore')
        self.assertFalse((self.root/'bad_restore/summary.json').exists())
        self.assertEqual(legacy.read_json(self.root/'bad_restore/preservation.json')['source_artifacts_preserved'], True)

    def test_source_file_change_and_model_buffer_change_are_detected(self):
        original = evaluation.forward_case
        def source_changed(*args, **kwargs):
            row = original(*args, **kwargs)
            if args[3] == 'EVENT_UNIFORM':
                self.source_file.write_text('changed', encoding='utf-8')
            return row
        with patch.object(evaluation, 'forward_case', side_effect=source_changed), self.assertRaisesRegex(ValueError, 'Source artifacts changed'):
            self.run_probe('bad_source')
        self.assertFalse(legacy.read_json(self.root/'bad_source/preservation.json')['source_artifacts_preserved'])
        self.assertFalse((self.root/'bad_source/summary.json').exists())
        self.source_file.write_text('immutable source', encoding='utf-8')
        def model_changed(*args, **kwargs):
            row = original(*args, **kwargs)
            if args[3] == 'EVENT_REPORT_OFF':
                with torch.no_grad():
                    args[0].T_i_D_emb.add_(.01)
            return row
        with patch.object(evaluation, 'forward_case', side_effect=model_changed), self.assertRaisesRegex(ValueError, 'Inference changed model tensors'):
            self.run_probe('bad_model')
        self.assertFalse((self.root/'bad_model/summary.json').exists())

    def test_source_axes_mask_and_batch_partition_cannot_drift(self):
        for name, mutate in (
            ('station', lambda arrays: arrays.update(station_ids=arrays['station_ids'][::-1])),
            ('sample', lambda arrays: arrays.update(sample_indices=arrays['sample_indices'][::-1])),
            ('mask', lambda arrays: arrays.update(valid=~arrays['valid']))):
            with self.subTest(name=name):
                arrays = copy.deepcopy(self.arrays)
                mutate(arrays)
                with self.assertRaises(ValueError):
                    self.run_probe('bad_axes_'+name, arrays=arrays)
        baseline = dict(self.baseline, batch_partition=[1]*len(self.arrays['target']))
        with self.assertRaisesRegex(ValueError, 'batch partition changed'):
            self.run_probe('bad_batch', baseline=baseline)

    def test_semantic_UNK_preserves_support_and_is_distinct_from_zero_context(self):
        batch = next(self.data.loaders['val_loader'].get_iterator())
        events = batch['event_data']
        before = probe_digest = evaluation.probe.input_digest(batch)
        unknown = evaluation.semantic_unknown(events)
        np.testing.assert_array_equal(unknown['relation'], events['relation'])
        np.testing.assert_array_equal(unknown['report_present'], events['report_present'])
        np.testing.assert_array_equal(unknown['type'], events['type'])
        self.assertFalse(unknown['type_valid'].any() or unknown['description_valid'].any())
        observed, original = [], self.models['EVENT'].forward
        def record(*args, **kwargs):
            observed.append(kwargs)
            return original(*args, **kwargs)
        with patch.object(self.models['EVENT'], 'forward', side_effect=record):
            row = evaluation.forward_case(self.models['EVENT'], batch, self.data.scaler, 'EVENT_SEMANTIC_UNK')
        self.assertTrue(observed[0]['semantic_enabled'])
        self.assertFalse(observed[0]['incident_data']['type_valid'].any())
        self.assertEqual(row['mechanisms']['fusion_supported_node_positions'], 2*len(batch['x_data']))
        off = evaluation.forward_case(self.models['EVENT'], batch, self.data.scaler, 'EVENT_REPORT_OFF')
        self.assertFalse(np.array_equal(row['prediction'], off['prediction']))
        self.assertEqual(evaluation.probe.input_digest(batch), before)

    def test_source_really_selects_fixed_hard_windows_and_bias(self):
        target = np.full((3, 12, 4), 10., np.float32)
        valid = np.ones_like(target, dtype=bool)
        predictions = {name: target.copy()+1 for name in evaluation.CASES}
        predictions['SOURCE_MIN'][0] = 20.
        predictions['SOURCE_MIN'][1:] = 9.
        predictions['STATE_ON'][0] = 10.
        predictions['STATE_ON'][2] = 50.
        regions = dict(all_nodes=np.ones(4, bool), direct_report_nodes=np.array([1, 0, 0, 0], bool),
            potential_only=np.array([0, 1, 0, 0], bool), remaining_nodes=np.array([0, 0, 1, 1], bool))
        result = evaluation.summarize(predictions, target, valid, regions, np.array([100, 101, 102]))
        self.assertEqual(result['fixed_hard_windows']['selector'], 'SOURCE_MIN')
        self.assertEqual(result['fixed_hard_windows']['sample_indices'], [100])
        self.assertEqual(result['comparisons']['EVENT_vs_STATE']['control'], 'STATE_ON')
        self.assertEqual(result['comparisons']['STATE_vs_SOURCE']['control'], 'SOURCE_MIN')
        strata = result['comparisons']['EVENT_vs_STATE']['strata']
        self.assertEqual(strata['bias:over']['valid_count'], 48)
        self.assertEqual(strata['bias:under']['valid_count'], 96)
        with self.assertRaisesRegex(ValueError, 'all eight'):
            evaluation.summarize({k: v for k, v in predictions.items() if k != 'SOURCE_MIN'}, target, valid, regions, np.arange(3))

    def test_weight_means_use_actual_batch_node_counts_not_mean_of_batch_means(self):
        totals = {}
        evaluation.update_weights(totals, np.ones((2, 12, 4, 10)))
        evaluation.update_weights(totals, np.full((1, 12, 4, 10), 1.3))
        result = evaluation.weight_results({'EVENT_ON': totals})['EVENT_ON']
        np.testing.assert_allclose(result['mean'], np.full((12, 10), 1.1))
        np.testing.assert_allclose(result['absolute_delta_from_one_mean'], np.full((12, 10), .1))
        self.assertEqual(result['count_per_horizon_component'], 12)


if __name__ == '__main__':
    unittest.main()
