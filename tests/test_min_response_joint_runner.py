"""Joint-stage training, source preservation and exact resume contracts.

Current-checkpoint OFF is a fine-tuned MIN; the fixed original MIN is a second,
immutable reference. The existing MIN is replayed only, never retrained.
"""
import copy
from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.chronological import train_min_response_joint as training
from experiments.chronological import train_min_event_budget as original
from experiments.chronological import train_b3_r_minimal_paper as paper
from experiments.chronological import train_internal_spatial as legacy
from experiments.chronological import probe_min_event_budget as probe
from experiments.chronological.min_response_joint_metrics import write_diagnostics
from experiments.chronological.smoke import set_seed
from src.models.incident_spatial_residual import tensor_state_hash
from tests import test_min_frozen_budget_runner as source_fixture
from tests.test_min_event_budget_runner import tiny_minimal


class JointResponseRunnerTests(unittest.TestCase):
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
        base = self.base()
        set_seed(2025)
        model, certificate = training.build_joint(base, arm, self.recipe)
        identity = dict(schema=training.SCHEMA, training_stage=self.recipe['training_stage'],
            arm=arm, recipe=copy.deepcopy(self.recipe), engineering_check=True, seed=2025,
            source_MIN_training_updates=0,
            source_state_sha256=tensor_state_hash(base.state_dict()),
            model_initial_sha256=tensor_state_hash(model.state_dict()))
        return model, base, self.fixture.data(), identity, certificate

    def fit(self, model, base, data, identity, output, epochs, recipe=None, **kwargs):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return training.fit(model, base, data, self.fixture.saved, identity,
                                output, epochs, recipe or self.recipe, **kwargs)

    def output(self, suffix=''):
        return self.root/(self._testMethodName+suffix)

    def load(self, output):
        return torch.load(output/'last.pt', map_location='cpu', weights_only=False)

    def equal_nested(self, left, right):
        self.fixture.equal_nested(self, left, right)

    def test_two_candidates_have_identical_epoch_zero_predictions_and_original_batches(self):
        references = []
        root = self.output()
        for arm in training.ARMS:
            model, base, data, identity, _ = self.models(arm)
            result = self.fit(model, base, data, identity, root/arm, 1)
            saved = self.load(root/arm)
            self.assertEqual(saved['history'][0]['epoch'], 0)
            references.append(saved['history'][0]['validation']['on']['mae_pooled'])
            self.assertEqual(result['completed_epoch'], 1)
            with np.load(root/arm/'baseline_predictions.npz', allow_pickle=False) as baseline:
                self.assertEqual(baseline['prediction'].shape, (3, 12, 4))
                self.assertEqual(baseline['history_low_mask'].shape, (3, 4))
                self.assertEqual(baseline['history_low_mask'].dtype, np.bool_)
        self.assertEqual(references, [references[0]]*len(training.ARMS))
        orders = [self.load(root/arm)['history'][1]['training']['train_order_sha256']
                  for arm in training.ARMS]
        self.assertEqual(len(set(orders)), 1)
        report = write_diagnostics(root)
        self.assertEqual(report['status'], 'COMPLETE')
        self.assertEqual(set(report['runs']), {'state', 'location'})
        self.assertEqual(report['source_MIN_training_updates'], 0)

    def test_joint_training_updates_backbone_and_response_preserving_readonly_source(self):
        model, base, data, identity, _ = self.models('location')
        source_before = tensor_state_hash(base.state_dict())
        backbone_before = tensor_state_hash(model.backbone.state_dict())
        response_before = tensor_state_hash(model.response.state_dict())
        paths = {str(path.relative_to(self.fixture.source)): path
                 for path in self.fixture.source.rglob('*') if path.is_file()}
        protected = probe.snapshot_files(paths)
        identity['source_artifacts_sha256'] = protected
        with patch.object(data, 'test_loader', side_effect=AssertionError('Unexpected test evaluation')):
            summary = self.fit(model, base, data, identity, self.output(), 1, protected_paths=paths)
        saved = self.load(self.output())
        model.load_state_dict(saved['model_state'], strict=True)
        self.assertNotEqual(tensor_state_hash(model.backbone.state_dict()), backbone_before)
        self.assertNotEqual(tensor_state_hash(model.response.state_dict()), response_before)
        self.assertEqual(tensor_state_hash(base.state_dict()), source_before)
        self.assertTrue(all(parameter.grad is None for parameter in base.parameters()))
        self.assertEqual(probe.snapshot_files(paths), protected)
        self.assertFalse(summary['test_evaluated'])
        self.assertEqual(summary['training_stage'], 'joint_finetune_stage2')
        self.assertEqual(saved['history'][1]['training']['full_supervised_horizons'], 12)

    def test_joint_optimizer_contains_candidate_copies_and_excludes_fixed_reference(self):
        for arm in ('location', 'state'):
            model, base, _, _, _ = self.models(arm)
            optimizer = training.optimizer_for(model, arm, self.recipe)
            included = [id(parameter) for group in optimizer.param_groups for parameter in group['params']]
            source = {id(parameter) for parameter in base.parameters()}
            self.assertFalse(source.intersection(included))
            self.assertEqual(set(included), {id(parameter) for parameter in model.parameters()
                                             if parameter.requires_grad})
            self.assertEqual(len(included), len(set(included)))
            self.assertEqual([(group['name'], group['lr']) for group in optimizer.param_groups],
                             [('backbone', self.recipe['backbone_learning_rate']),
                              ('response', self.recipe['response_learning_rate'])])

    def test_current_off_is_finetuned_backbone_and_original_off_stays_fixed(self):
        model, base, data, identity, _ = self.models('state')
        self.fit(model, base, data, identity, self.output(), 1)
        saved = self.load(self.output())
        model.load_state_dict(saved['model_state'], strict=True)
        model.eval()
        batch = next(data.loaders['val_loader'].get_iterator())
        history = torch.as_tensor(batch['x_data'], dtype=torch.float32)
        event = original.event_tensors(batch['event_data'], 'cpu')
        with torch.no_grad():
            current_off = model(history, incident_data=event, report_enabled=False)
            current_min = model.backbone(history)
            original_min = base(history)
        torch.testing.assert_close(current_off, current_min, rtol=0, atol=0)
        self.assertFalse(torch.equal(current_off, original_min))
        with np.load(self.output()/'best_predictions.npz', allow_pickle=False) as best, \
                np.load(self.output()/'baseline_predictions.npz', allow_pickle=False) as baseline:
            for key in ('target', 'valid', 'sample_indices', 'fixed_hard_indices', 'history_low_mask'):
                np.testing.assert_array_equal(best[key], baseline[key])
            np.testing.assert_array_equal(best['original_off'], baseline['prediction'])
            np.testing.assert_array_equal(best['off'], best['original_off'])
            self.assertEqual(best['current_off'].shape, best['original_off'].shape)

    def test_full_joint_model_optimizer_rng_and_order_resume_exactly(self):
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
            with self.subTest(key=key):
                self.equal_nested(expected[key], actual[key])
        for left, right in zip(expected['history'], actual['history']):
            self.equal_nested({key: value for key, value in left.items() if key != 'seconds'},
                              {key: value for key, value in right.items() if key != 'seconds'})
        self.assertEqual(result['optimizer_updates_this_invocation'], 2)
        model, base, data, identity, _ = self.models('state')
        self.assertEqual(self.fit(model, base, data, identity, self.output('_split'), 2)
                         ['optimizer_updates_this_invocation'], 0)

    def test_resume_rejects_other_stage_identity_and_invalid_best_certificate(self):
        model, base, data, identity, _ = self.models('state')
        self.fit(model, base, data, identity, self.output(), 1)
        saved = self.load(self.output())
        for change in ('schema', 'stage', 'arm', 'selection'):
            bad = copy.deepcopy(saved)
            if change == 'schema':
                bad['schema'] = 'frozen_MIN_conditional_response_pair_v1'
            elif change == 'stage':
                bad['identity']['training_stage'] = 'frozen_diagnostic_stage1'
            elif change == 'arm':
                bad['identity']['arm'] = 'location'
            else:
                bad['best_metric'] = -1.
            with self.subTest(change=change), self.assertRaises(ValueError):
                training.validate_checkpoint(bad, identity, model, saved['baseline_sha256'])

    def test_joint_cli_reuses_existing_MIN_and_rejects_a_baseline_training_arm(self):
        args = training.parse_args(['--source-run', str(self.fixture.source)])
        self.assertEqual(set(args.arms), {'location', 'state'})
        self.assertEqual(len(args.arms), 2)
        self.assertEqual(self.recipe['training_stage'], 'joint_finetune_stage2')
        for command in (['--batch-size', '128'], ['--evaluate-test'], ['--arms', 'minimal']):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                training.parse_args(['--source-run', str(self.fixture.source), *command])


if __name__ == '__main__':
    unittest.main()
