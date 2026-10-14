"""Best replay gates interventions; direct forecast readouts preserve all state."""
import copy
from contextlib import redirect_stdout
import io
import logging
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.chronological import evaluate_min_early_event as evaluation
from experiments.chronological import min_early_event_metrics as metrics
from src.models.incident_spatial_early_event import from_minimal
from src.models.incident_spatial_residual import tensor_state_hash
from src.utils.igstgnn_published_events import EventPublishedData
from src.utils.metrics import compute_all_metrics
from tests.test_min_event_budget_data import event_fixture
from tests.test_min_event_budget_runner import tiny_minimal
from tests.test_incident_spatial import graph


class EarlyEventMetricsTests(unittest.TestCase):
    def fixture(self):
        target = np.full((2, 12, 4), 10., np.float32)
        values = {'STATE_ON': 12., 'EVENT_ON': 11., 'EVENT_OFF': 13.,
                  'EVENT_SEMANTIC_UNK': 12., 'EVENT_LOCATION_SHUFFLE': 14.}
        predictions = {key: np.full_like(target, value) for key, value in values.items()}
        direct = np.array([[True, False, False, False]]*2)
        potential = np.array([[False, True, False, False]]*2)
        regions = dict(all_nodes=np.ones((2, 4), bool), direct_report_nodes=direct,
                       potential_only=potential, remaining_nodes=~(direct | potential),
                       empty=np.zeros((2, 4), bool), **{'road:west': np.array([True, True, False, False])})
        return predictions, target, np.ones_like(target, bool), regions, np.array([100, 200])

    def test_hand_calculated_signs_weights_empty_regions_and_fixed_windows(self):
        result = metrics.summarize(*self.fixture())
        gain = result['comparisons']['EVENT_vs_STATE']['metrics']
        self.assertEqual(gain['all_nodes']['mae_gain_pooled'], 1.)
        self.assertEqual(gain['all_nodes']['gain_positive'], 1.)
        self.assertEqual(gain['all_nodes']['gain_negative'], 0.)
        self.assertEqual(gain['direct_report_nodes']['global_gain_contribution'], .25)
        self.assertEqual(gain['remaining_nodes']['global_gain_contribution'], .5)
        self.assertIsNone(gain['empty']['mae_gain_pooled'])
        self.assertEqual(result['fixed_hard_windows']['sample_indices'], [100])
        self.assertEqual(result['comparisons']['EVENT_dependency']['metrics']['all_nodes']['mae_gain_pooled'], 2.)
        self.assertAlmostEqual(result['cases']['EVENT_ON']['metrics']['all_nodes']['mape_pooled'], .1)

    def test_bias_reference_is_STATE_for_every_comparison_not_OFF(self):
        args = self.fixture()
        args[0]['STATE_ON'][:] = 8.  # STATE under, although EVENT_OFF over.
        result = metrics.summarize(*args)
        groups = result['comparisons']['EVENT_dependency']['strata']
        self.assertEqual(groups['bias:under']['mae_gain_pooled'], 2.)
        self.assertIsNone(groups['bias:over']['mae_gain_pooled'])

    def test_invalid_axes_and_partition_are_rejected(self):
        args = self.fixture()
        args[3]['potential_only'][:] = True
        with self.assertRaisesRegex(ValueError, 'disjoint'):
            metrics.summarize(*args)
        args = self.fixture()
        args[0]['EVENT_OFF'] = args[0]['EVENT_OFF'][:, :, :3]
        with self.assertRaisesRegex(ValueError, 'axes'):
            metrics.summarize(*args)


class EarlyEventEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        cls.temp = tempfile.TemporaryDirectory(dir=evaluation.REPO)
        cls.root = Path(cls.temp.name)
        cls.raw = cls.root/'data'
        cls.raw.mkdir()
        _, cls.protocol = event_fixture(cls.raw)
        cls.logger = logging.getLogger('early_event_eval_tests')
        cls.logger.addHandler(logging.NullHandler())
        cls.candidate_graph = graph(4)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def data(self):
        return EventPublishedData(self.raw, self.protocol, self.logger)

    def models(self):
        models = {}
        for arm, mode in [('STATE', 'state'), ('EVENT', 'event')]:
            minimal, _ = tiny_minimal(SimpleNamespace(seed=2025), 4, self.candidate_graph, 'cpu')
            model, _ = from_minimal(minimal, mode=mode, width=8)
            with torch.no_grad():
                model.early_event_condition.condition[-1].weight.fill_(.02)
                model.early_event_condition.condition[-1].bias.fill_(.01)
            models[arm] = model.eval()
        return models

    def saved(self, models, data):
        result = {}
        for arm, model in models.items():
            prediction, target = [], []
            for batch in data.loaders['val_loader'].get_iterator():
                row = evaluation.forward_case(model, batch, data.scaler, arm+'_ON')
                prediction.append(row['prediction'])
                target.append(row['target'])
            pred, y = torch.from_numpy(np.concatenate(prediction)), torch.from_numpy(np.concatenate(target))
            null = y.min() if y.min() < 1 else torch.tensor(0.)
            mae, mape, rmse = compute_all_metrics(pred, y, null)
            result[arm] = dict(completed_epoch=1, best_metric=mae, global_updates=7,
                best_model_state={key: value.detach().clone() for key, value in model.state_dict().items()},
                history=[dict(epoch=1, best_epoch=1, validation_mae=mae, validation_mape=mape, validation_rmse=rmse,
                              train_order_sha256='a'*64)])
        return result

    def output(self):
        return self.root/self._testMethodName

    def test_full_station_shuffle_is_deterministic_moves_support_and_preserves_other_fields(self):
        data = self.data()
        batch = next(data.loaders['val_loader'].get_iterator())
        original = copy.deepcopy(batch['event_data'])
        before = np.random.get_state()
        shuffled, donors, audit = evaluation.shuffled_locations(original, np.array([14, 15]), 17)
        after = np.random.get_state()
        self.assertEqual(repr(before), repr(after))
        self.assertTrue((donors != np.arange(4)).all())
        for i in range(2):
            np.testing.assert_array_equal(shuffled['relation'][i], original['relation'][i, donors[i]])
        second, same, other_audit = evaluation.shuffled_locations(original, np.array([14, 15]), 17)
        np.testing.assert_array_equal(donors, same)
        self.assertEqual(audit, other_audit)
        reversed_data = {key: value[::-1] for key, value in original.items()}
        reverse, reverse_donor, _ = evaluation.shuffled_locations(reversed_data, np.array([15, 14]), 17)
        np.testing.assert_array_equal(reverse_donor[::-1], donors)
        np.testing.assert_array_equal(reverse['relation'][::-1], shuffled['relation'])
        self.assertGreater(audit['support_changed_positions'], 0)
        for key in original:
            if key != 'relation':
                np.testing.assert_array_equal(original[key], shuffled[key])

    def test_native_forecast_identity_and_twelve_direct_readouts_preserve_hooks(self):
        model, data = self.models()['EVENT'], self.data()
        batch = next(data.loaders['val_loader'].get_iterator())
        before = tensor_state_hash(model.state_dict())
        row = evaluation.forward_case(model, batch, data.scaler, 'EVENT_ON', audit_forecasts=True)
        self.assertEqual(len(row['forecasts']), 12)
        self.assertEqual(row['mechanisms']['exchange_calls'], 20)
        self.assertEqual(row['mechanisms']['local_substeps'], 40)
        self.assertTrue(any(not np.array_equal(pred, row['prediction']) for pred in row['forecasts'].values()))
        self.assertEqual(tensor_state_hash(model.state_dict()), before)
        self.assertTrue(all(not module._forward_hooks for module in model.modules()))
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            evaluation.forecast_cases(model, {}, torch.empty(1))

    def test_success_writes_readable_predictions_metrics_and_preservation(self):
        models, data = self.models(), self.data()
        saved = self.saved(models, data)
        with redirect_stdout(io.StringIO()):
            result = evaluation.run_probe(models, data, saved, self.candidate_graph, {'west': [0, 1]}, self.output())
        self.assertEqual(result['status'], 'MIN_EARLY_EVENT_EVALUATION_PASS')
        self.assertEqual(len(result['cases']), 17)
        self.assertEqual(result['training_updates'], 0)
        self.assertTrue(result['on_exactly_restored'])
        for name in ('paired_predictions.npz', 'analysis.json', 'event_state_cross_strata.json', 'report.md', 'mechanisms.json',
                     'preservation.json', 'replay.json', 'horizons.csv', 'regions.csv', 'strata.csv'):
            self.assertTrue((self.output()/name).is_file(), name)
        with np.load(self.output()/'paired_predictions.npz') as archive:
            self.assertEqual(archive['EVENT_ON'].shape, (3, 12, 4))
            np.testing.assert_array_equal(archive['sample_indices'], [14, 15, 16])
        self.assertTrue(all(not module._forward_hooks for model in models.values() for module in model.modules()))

    def test_failed_best_replay_blocks_all_interventions(self):
        models, data = self.models(), self.data()
        saved = self.saved(models, data)
        saved['EVENT']['history'][0]['validation_mae'] += 1.
        saved['EVENT']['best_metric'] += 1.
        original, calls = evaluation.forward_case, []
        def recording(*args, **kwargs):
            calls.append(args[3])
            return original(*args, **kwargs)
        with patch.object(evaluation, 'forward_case', side_effect=recording), redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, 'replay failed'):
                evaluation.run_probe(models, data, saved, self.candidate_graph, {}, self.output())
        self.assertTrue(all(case.endswith('_ON') for case in calls))
        self.assertFalse((self.output()/'summary.json').exists())
        self.assertTrue((self.output()/'failure.json').is_file())

    def test_training_budget_and_common_prefix_orders_are_actually_checked(self):
        models, data = self.models(), self.data()
        saved = self.saved(models, data)
        self.assertTrue(evaluation.training_comparability(saved)['same_completed_training_budget'])
        saved['EVENT']['global_updates'] += 1
        result = evaluation.training_comparability(saved)
        self.assertFalse(result['same_completed_training_budget'])
        self.assertIn('unequal_completed_budget', result['whole_model_scope'])
        saved['EVENT']['history'][0]['train_order_sha256'] = 'b'*64
        with self.assertRaisesRegex(ValueError, 'sample order differs'):
            evaluation.training_comparability(saved)

    def test_failed_restoration_blocks_completion_and_forecast_exception_removes_hooks(self):
        models, data = self.models(), self.data()
        saved = self.saved(models, data)
        original, intervention_seen = evaluation.forward_case, False
        def broken(*args, **kwargs):
            nonlocal intervention_seen
            row = original(*args, **kwargs)
            if args[3] == 'EVENT_OFF':
                intervention_seen = True
            elif intervention_seen and args[3] == 'EVENT_ON':
                row['prediction'][0, 0, 0] += 1.
            return row
        with patch.object(evaluation, 'forward_case', side_effect=broken), redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, 'restoration failed'):
                evaluation.run_probe(models, data, saved, self.candidate_graph, {}, self.output())
        self.assertFalse((self.output()/'summary.json').exists())
        batch = next(data.loaders['val_loader'].get_iterator())
        with patch.object(models['EVENT'], '_decode_forecast', side_effect=RuntimeError('head failure')):
            with self.assertRaisesRegex(RuntimeError, 'head failure'):
                evaluation.forward_case(models['EVENT'], batch, data.scaler, 'EVENT_ON', audit_forecasts=True)
        self.assertTrue(all(not module._forward_hooks for model in models.values() for module in model.modules()))


if __name__ == '__main__':
    unittest.main()
