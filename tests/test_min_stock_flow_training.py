"""New-candidate lifecycle, train-only scaling and exact epoch resume."""
import copy
from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.chronological import train_min_stock_flow as training
from experiments.chronological import train_min_event_budget as original
from experiments.chronological import train_b3_r_minimal_paper as paper
from experiments.chronological import train_internal_spatial as legacy
from experiments.chronological import probe_min_event_budget as probe
from experiments.chronological.smoke import set_seed
from src.models.incident_spatial_residual import tensor_state_hash
from tests import test_min_frozen_budget_runner as source_fixture
from tests.test_min_event_budget_runner import tiny_minimal


class StockFlowTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source_fixture.FrozenBudgetRunnerTests.setUpClass()
        cls.fixture = source_fixture.FrozenBudgetRunnerTests
        cls.root = cls.fixture.root
        cls.recipe = legacy.read_json(training.RECIPE)

    @classmethod
    def tearDownClass(cls):
        cls.fixture.tearDownClass()

    def base(self):
        with patch.object(paper, 'build_model', side_effect=tiny_minimal):
            base, _ = original.build_arm(self.fixture.args, 'A', 4, self.fixture.graph, 'cpu')
        base.load_state_dict(self.fixture.saved['best_model_state'], strict=True)
        return base.requires_grad_(True).eval()

    def models(self, arm='location'):
        base, data = self.base(), self.fixture.data()
        scale, model_recipe, provenance = training.training_scale(data, self.recipe)
        set_seed(2025)
        model, certificate = training.build_joint(base, arm, model_recipe, scale)
        identity = dict(schema=training.SCHEMA, training_stage=training.STAGE,
            arm=arm, recipe=copy.deepcopy(self.recipe), engineering_check=True, seed=2025,
            source_MIN_training_updates=0, source_best_epoch=probe.best_row(self.fixture.saved)['epoch'],
            source_state_sha256=tensor_state_hash(base.state_dict()),
            flow_scale_provenance=provenance, model_initial_sha256=tensor_state_hash(model.state_dict()))
        return model, base, data, identity, certificate

    def output(self, suffix=''):
        return self.root/(self._testMethodName+suffix)

    def fit(self, model, base, data, identity, output, epochs, **kwargs):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return training.fit(model, base, data, self.fixture.saved, identity,
                output, epochs, self.recipe, **kwargs)

    def load(self, output):
        return torch.load(output/'last.pt', map_location='cpu', weights_only=False)

    def equal_nested(self, left, right):
        self.fixture.equal_nested(self, left, right)

    def test_only_new_candidates_allowed_and_no_test_or_batch_override(self):
        args = training.parse_args(['--source-run', str(self.fixture.source)])
        self.assertEqual(args.arms, ['state', 'location'])
        self.assertEqual(self.recipe['source_MIN_training_updates'], 0)
        for option in (['--arms', 'A'], ['--evaluate-test'], ['--batch-size', '128'], ['--epochs', '11']):
            with redirect_stderr(io.StringIO()), self.assertRaises((SystemExit, ValueError)):
                training.parse_args(['--source-run', str(self.fixture.source), *option])

    def test_flow_scale_reads_only_training_history_not_Y_or_validation(self):
        data = self.fixture.data()
        expected, _, provenance = training.training_scale(data, self.recipe)
        for sample in data.loaders['val_loader'].original_samples:
            sample['x_data'][:] = 9999
            sample['y_data'][:] = -9999
        for sample in data.loaders['train_loader'].original_samples:
            sample['y_data'][:] = 7777
        actual, _, changed_provenance = training.training_scale(data, self.recipe)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(provenance, changed_provenance)
        self.assertEqual(provenance['source_scope'], 'published_training_X_only_no_Y_validation_or_test')

    def test_training_peak_floor_prevents_median_range_clipping(self):
        data = self.fixture.data()
        mean, std = float(data.scaler.mean), float(data.scaler.std)
        for sample in data.loaders['train_loader'].original_samples:
            sample['x_data'][..., 0] = (1.-mean)/std
        data.loaders['train_loader'].original_samples[0]['x_data'][0, 0, 0] = (1000.-mean)/std
        scale, _, provenance = training.training_scale(data, self.recipe)
        self.assertGreaterEqual(float(scale[0])*self.recipe['flow_scale_max_rate_factor'], 1250.-1e-3)
        self.assertTrue(provenance['training_peak_range_assertion_passed'])
        self.assertEqual(provenance['training_peak_margin'], 1.25)
        self.assertFalse(provenance['future_targets_used'])

    def test_common_initialization_is_equal_without_claiming_source_function_equality(self):
        common, predictions = [], []
        for arm in training.ARMS:
            model, base, data, _, certificate = self.models(arm)
            common.append(certificate['common_initial_sha256'])
            self.assertTrue(certificate['compatible_tensors_exact'])
            self.assertFalse(certificate['exact_source_prediction_initialization'])
            self.assertTrue(certificate['old_history_P_removed'])
            batch = next(data.loaders['val_loader'].get_iterator())
            on = training.forward_case(model, arm, batch, data.scaler, True)
            off = training.forward_case(model, arm, batch, data.scaler, False)
            np.testing.assert_array_equal(on['prediction'], off['prediction'])
            predictions.append(on['prediction'])
            self.assertEqual(on['mechanisms']['output_scope'], 'sum_actual_node_outflow_dt_no_native_prediction_residual')
        self.assertEqual(common[0], common[1])
        np.testing.assert_array_equal(predictions[0], predictions[1])

    def test_native_source_reference_replayed_once_then_cache_reused(self):
        base, data = self.base(), self.fixture.data()
        cache = self.output(); cache.mkdir()
        first = training.baseline_reference(base, data, self.fixture.saved, self.recipe, cache)
        self.assertTrue(first['original_source_metrics_replay_passed'])
        with patch.object(base, 'forward', side_effect=AssertionError('Repeated source replay')):
            second = training.baseline_reference(base, data, self.fixture.saved, self.recipe, cache)
        for key in training.BASELINE_KEYS:
            np.testing.assert_array_equal(first[key], second[key])
        (cache/'source_reference.npz').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, '[Cc]ache'):
            training.baseline_reference(base, data, self.fixture.saved, self.recipe, cache)

    def test_optimizer_excludes_source_and_has_no_duplicate_parameters(self):
        model, base, _, _, _ = self.models()
        optimizer = training.optimizer_for(model, 'location', self.recipe)
        ids = [id(p) for group in optimizer.param_groups for p in group['params']]
        self.assertEqual(set(ids), {id(p) for p in model.parameters() if p.requires_grad})
        self.assertEqual(len(ids), len(set(ids)))
        self.assertFalse(set(ids).intersection(id(p) for p in base.parameters()))
        self.assertEqual([group['name'] for group in optimizer.param_groups], ['backbone', 'decoder'])

    def test_joint_updates_preserve_source_and_pair_archives_original_batches(self):
        model, base, data, identity, _ = self.models()
        source_before = tensor_state_hash(base.state_dict())
        before = tensor_state_hash(model.state_dict())
        paths = {str(p.relative_to(self.fixture.source)): p for p in self.fixture.source.rglob('*') if p.is_file()}
        identity['source_artifacts_sha256'] = probe.snapshot_files(paths)
        with patch.object(data, 'test_loader', side_effect=AssertionError('Test evaluation forbidden')):
            result = self.fit(model, base, data, identity, self.output(), 1, protected_paths=paths)
        saved = self.load(self.output())
        self.assertNotEqual(tensor_state_hash(saved['model_state']), before)
        self.assertEqual(tensor_state_hash(base.state_dict()), source_before)
        self.assertEqual(saved['history'][1]['training']['source_teacher_forward_calls'], 0)
        self.assertEqual(result['source_MIN_training_updates'], 0)
        self.assertTrue(result['source_reference_preserved'])
        self.assertFalse(result['test_evaluated'])
        self.assertEqual(result['best_validation']['batch_partition'], [2, 1])
        self.assertEqual(probe.snapshot_files(paths), identity['source_artifacts_sha256'])
        with np.load(self.output()/'best_predictions.npz', allow_pickle=False) as best, \
                np.load(self.output()/'baseline_predictions.npz', allow_pickle=False) as baseline:
            np.testing.assert_array_equal(best['original_off'], baseline['prediction'])
            for key in ('target', 'valid', 'sample_indices', 'fixed_hard_indices', 'history_low_mask'):
                np.testing.assert_array_equal(best[key], baseline[key])
            self.assertEqual(best['on'].shape, (3, 12, 4))
            self.assertEqual(best['current_off'].shape, (3, 12, 4))

    def test_two_epochs_equal_split_resume_model_optimizer_rng_and_order(self):
        model, base, data, identity, _ = self.models('state')
        self.fit(model, base, data, identity, self.output('_whole'), 2)
        expected = self.load(self.output('_whole'))
        model, base, data, identity, _ = self.models('state')
        self.fit(model, base, data, identity, self.output('_split'), 1)
        model, base, data, identity, _ = self.models('state')
        result = self.fit(model, base, data, identity, self.output('_split'), 2)
        actual = self.load(self.output('_split'))
        for key in ('model_state', 'best_model_state', 'optimizer_state', 'train_order',
                    'numpy_rng', 'torch_rng', 'cuda_rng', 'completed_epoch', 'best_epoch', 'best_metric'):
            self.equal_nested(expected[key], actual[key])
        for left, right in zip(expected['history'], actual['history']):
            self.equal_nested({k: v for k, v in left.items() if k != 'seconds'},
                              {k: v for k, v in right.items() if k != 'seconds'})
        self.assertEqual(result['optimizer_updates_this_invocation'], 2)
        model, base, data, identity, _ = self.models('state')
        self.assertEqual(self.fit(model, base, data, identity, self.output('_split'), 2)
            ['optimizer_updates_this_invocation'], 0)

    def test_resume_rejects_other_schema_arm_and_wrong_best(self):
        model, base, data, identity, _ = self.models('state')
        self.fit(model, base, data, identity, self.output(), 1)
        saved = self.load(self.output())
        for change in ('schema', 'arm', 'selection'):
            bad = copy.deepcopy(saved)
            if change == 'schema': bad['schema'] = 'MIN_conditional_response_joint_pair_v1'
            elif change == 'arm': bad['identity']['arm'] = 'location'
            else: bad['best_metric'] = -1.
            with self.assertRaises(ValueError):
                training.validate_checkpoint(bad, identity, model, saved['baseline_sha256'])

    def test_two_arm_report_uses_same_source_and_committed_best(self):
        root = self.output(); root.mkdir()
        for arm in training.ARMS:
            model, base, data, identity, _ = self.models(arm)
            self.fit(model, base, data, identity, root/arm, 1)
        report = training.write_diagnostics(root)
        self.assertEqual(report['status'], 'COMPLETE')
        self.assertEqual(report['source_MIN_training_updates'], 0)
        self.assertEqual(report['selected_best_location_vs_state']['completed_training_budget'], 1)
        self.assertFalse(report['predictive_gain_claim'])
        for arm in training.ARMS:
            row = report['runs'][arm]
            self.assertEqual(row['validation_rows'], 3)
            self.assertEqual(row['original_reference_pair']['paired'],
                             self.load(root/arm)['history'][row['best_epoch']]['validation']['paired'])


if __name__ == '__main__':
    unittest.main()
