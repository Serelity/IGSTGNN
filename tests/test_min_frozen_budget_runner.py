"""Frozen MIN replay, adapter-only optimization and epoch-boundary resume."""
import copy
from contextlib import redirect_stderr, redirect_stdout
import io
import logging
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.chronological import train_min_frozen_budget as frozen
from experiments.chronological import train_min_event_budget as original
from experiments.chronological import train_b3_r_minimal_paper as paper
from experiments.chronological import probe_min_event_budget as probe
from experiments.chronological import train_internal_spatial as legacy
from experiments.chronological.smoke import set_seed, sha256
from src.models.incident_spatial_frozen_budget import build_frozen_location, frozen_state_hash
from src.models.incident_spatial_residual import tensor_state_hash
from src.utils.igstgnn_published_events import EventPublishedData
from tests.test_incident_spatial import graph
from tests.test_min_event_budget_data import event_fixture
from tests.test_min_event_budget_runner import args_for, tiny_minimal


class FrozenBudgetRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        cls.temp = tempfile.TemporaryDirectory(dir=original.REPO)
        cls.root = Path(cls.temp.name)
        cls.raw, cls.source = cls.root/'data', cls.root/'source'
        cls.raw.mkdir()
        cls.source.mkdir()
        samples, fixture = event_fixture(cls.raw)
        # Include the author's zero-target mask and a one-row validation tail.
        samples[14]['y_data'][0, 0, 0] = -5.
        np.save(cls.raw/'incident_all.npy', np.array(samples, dtype=object))
        cls.protocol = original.load_protocol(original.parse_args([]))
        for key in ('total_samples', 'train_samples', 'val_samples', 'test_samples',
                    'stations', 'batch_size', 'warm_epoch', 'cl_epoch'):
            cls.protocol[key] = fixture[key]
        cls.protocol['data_sha256'] = {name: sha256(cls.raw/name)
                                     for name in cls.protocol['data_sha256'] if (cls.raw/name).is_file()}
        cls.args = args_for(cls.source)
        cls.args.arms = ['A']
        cls.args.data_dir = cls.raw
        cls.graph = graph(4)
        cls.logger = logging.getLogger('frozen_budget_runner_tests')
        cls.logger.addHandler(logging.NullHandler())
        cls.recipe = legacy.read_json(frozen.RECIPE)
        with patch.object(paper, 'build_model', side_effect=tiny_minimal), \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            data = cls.data()
            initialization = original.preflight(cls.args, data, cls.graph)
            cls.source_identity = original.make_identity(cls.args, cls.protocol,
                cls.protocol['data_sha256'], {'fixture': 'four_stations'}, initialization)
            original.verify_identity(cls.args, cls.source_identity, initialization, allow_new=True)
            legacy.write_json(cls.source/'data_audit.json', data.audit)
            original.train_arm(cls.args, 'A', cls.protocol, data, cls.source_identity, cls.graph, cls.logger)
        cls.saved = torch.load(cls.source/'A/last.pt', map_location='cpu', weights_only=False)
        original.validate_saved(cls.saved, cls.source_identity, 'A')

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    @classmethod
    def data(cls):
        return EventPublishedData(cls.raw, cls.protocol, cls.logger, check=True)

    def setup_models(self):
        with patch.object(paper, 'build_model', side_effect=tiny_minimal):
            base, _ = original.build_arm(self.args, 'A', 4, self.graph, 'cpu')
        base.load_state_dict(self.saved['best_model_state'], strict=True)
        base.eval()
        set_seed(2025)
        model, certificate = build_frozen_location(base)
        identity = dict(schema=frozen.SCHEMA, recipe=copy.deepcopy(self.recipe),
            engineering_check=True, seed=2025,
            frozen_state_sha256=certificate['frozen_state_sha256'],
            adapter_initial_sha256=certificate['adapter_initial_sha256'])
        return model, base, self.data(), identity

    def output(self, suffix=''):
        return self.root/(self._testMethodName+suffix)

    def fit(self, model, base, data, identity, output, epoch, **kwargs):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return frozen.fit(model, base, data, self.saved, identity, output, epoch,
                              self.recipe, **kwargs)

    def equal_nested(self, left, right):
        if torch.is_tensor(left):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        elif isinstance(left, np.ndarray):
            np.testing.assert_array_equal(left, right)
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.equal_nested(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(type(left), type(right))
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.equal_nested(a, b)
        else:
            self.assertEqual(left, right)

    def load(self, output):
        return torch.load(output/'last.pt', map_location='cpu', weights_only=False)

    def test_two_epochs_equal_one_plus_resume_in_adapter_optimizer_rng_and_order(self):
        model, base, data, identity = self.setup_models()
        whole = self.output('_whole')
        self.fit(model, base, data, identity, whole, 2)
        expected = self.load(whole)
        model, base, data, identity = self.setup_models()
        split = self.output('_split')
        self.fit(model, base, data, identity, split, 1)
        # A reconstructed process starts with fresh objects before restoring RNG/order.
        model, base, data, identity = self.setup_models()
        result = self.fit(model, base, data, identity, split, 2)
        actual = self.load(split)
        for key in ('adapter_state', 'best_adapter_state', 'optimizer_state', 'train_order',
                    'numpy_rng', 'torch_rng', 'cuda_rng', 'best_epoch', 'best_metric', 'completed_epoch'):
            with self.subTest(key=key):
                self.equal_nested(expected[key], actual[key])
        for a, b in zip(expected['history'], actual['history']):
            self.equal_nested({k: v for k, v in a.items() if k != 'seconds'},
                              {k: v for k, v in b.items() if k != 'seconds'})
        self.assertEqual(result['optimizer_updates_this_invocation'], 2)

    def test_training_changes_only_adapter_and_preserves_source_and_fixed_tail(self):
        model, base, data, identity = self.setup_models()
        before_frozen = frozen_state_hash(model)
        before_adapter = tensor_state_hash(model.event_budget_adapter.state_dict())
        paths = {str(p.relative_to(self.source)): p for p in self.source.rglob('*') if p.is_file()}
        identity['source_artifacts_sha256'] = probe.snapshot_files(paths)
        with patch.object(data, 'test_loader', side_effect=AssertionError('Implicit test evaluation')):
            result = self.fit(model, base, data, identity, self.output(), 1, protected_paths=paths)
        self.assertEqual(frozen_state_hash(model), before_frozen)
        self.assertNotEqual(tensor_state_hash(model.event_budget_adapter.state_dict()), before_adapter)
        self.assertEqual(probe.snapshot_files(paths), identity['source_artifacts_sha256'])
        self.assertEqual(result['best_validation']['batch_partition'], [2, 1])
        self.assertTrue(result['off_exactly_matches_source'])
        self.assertFalse(result['test_evaluated'])
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        saved = self.load(self.output())
        self.assertEqual(saved['history'][1]['training']['full_supervised_horizons'], 12)
        self.assertGreater(saved['history'][1]['training']['batches_with_nonzero_adapter_gradient'], 0)
        self.assertNotIn('model_state', saved)

    def test_epoch_zero_wins_when_training_does_not_improve(self):
        model, base, data, identity = self.setup_models()
        zero_state = tensor_state_hash(model.event_budget_adapter.state_dict())
        no_update = dict(train_mae_batch_mean=1., full_supervised_horizons=12,
            batches=data.loaders['train_loader'].num_batch, optimizer_steps=0,
            batches_with_nonzero_adapter_gradient=0, adapter_gradient_abs_max=0.,
            adapter_state_changed=False, train_order_sha256='unchanged_fixture')
        with patch.object(frozen, 'train_epoch', return_value=no_update):
            result = self.fit(model, base, data, identity, self.output(), 2)
        self.assertEqual(result['best_adapter_epoch'], 0)
        self.assertEqual(result['baseline_minus_best_mae'], 0.)
        self.assertFalse(result['best_adapter_selected'])
        self.assertFalse(result['exploratory_validation_increment'])
        saved = self.load(self.output())
        self.assertEqual(tensor_state_hash(saved['best_adapter_state']), zero_state)

    def test_source_replay_failure_prevents_optimizer_creation_and_training(self):
        model, base, data, identity = self.setup_models()
        broken = copy.deepcopy(self.saved)
        broken['history'][0]['validation_mae'] += 1.
        broken['history'][0]['best_metric'] = broken['best_metric'] = broken['history'][0]['validation_mae']
        with patch.object(frozen, 'optimizer_for', side_effect=AssertionError('Optimized before replay')), \
                patch.object(frozen, 'train_epoch', side_effect=AssertionError('Trained before replay')):
            with self.assertRaisesRegex(ValueError, 'Source A best validation replay failed'):
                frozen.fit(model, base, data, broken, identity, self.output(), 1, self.recipe)
        self.assertFalse((self.output()/'last.pt').exists())

    def test_off_tail_drift_is_rejected_before_any_training(self):
        model, base, data, identity = self.setup_models()
        forward = probe.forward_case

        def altered(model, arm, batch, scaler, case, semantic_payload=None):
            result = forward(model, arm, batch, scaler, case, semantic_payload)
            if case == 'L_OFF' and len(batch['x_data']) == 1:
                result['prediction'] = result['prediction']+1.
            return result

        with patch.object(probe, 'forward_case', side_effect=altered), \
                patch.object(frozen, 'train_epoch', side_effect=AssertionError('Trained before OFF verification')):
            with self.assertRaisesRegex(ValueError, 'OFF differs'):
                self.fit(model, base, data, identity, self.output(), 1)
        self.assertFalse((self.output()/'last.pt').exists())

    def test_frozen_buffer_drift_blocks_committing_trained_epoch(self):
        model, base, data, identity = self.setup_models()
        train = frozen.train_epoch

        def changed(*args, **kwargs):
            row = train(*args, **kwargs)
            name, buffer = next((n, p) for n, p in model.named_buffers()
                                if not n.startswith('event_budget_adapter.') and p.numel())
            with torch.no_grad():
                buffer.reshape(-1)[0].add_(1)
            return row

        with patch.object(frozen, 'train_epoch', side_effect=changed):
            with self.assertRaisesRegex(ValueError, 'parameters or buffers changed'):
                self.fit(model, base, data, identity, self.output(), 1)
        self.assertEqual(self.load(self.output())['completed_epoch'], 0)

    def test_optimizer_rejects_missing_duplicate_or_extra_parameters(self):
        model, _, _, _ = self.setup_models()
        adapter = list(model.event_budget_adapter.parameters())
        backbone = next(p for name, p in model.named_parameters()
                        if not name.startswith('event_budget_adapter.'))
        for parameters in (adapter[:-1], adapter+[backbone]):
            with self.subTest(size=len(parameters)):
                optimizer = torch.optim.Adam(parameters)
                with self.assertRaisesRegex(ValueError, 'only and all adapter'):
                    frozen.verify_optimizer(model, optimizer)
        good = frozen.optimizer_for(model, self.recipe)
        good.param_groups[0]['params'].append(adapter[0])
        with self.assertRaisesRegex(ValueError, 'only and all adapter'):
            frozen.verify_optimizer(model, good)

    def test_new_checkpoint_rejects_identity_or_adapter_corruption(self):
        model, base, data, identity = self.setup_models()
        self.fit(model, base, data, identity, self.output(), 1)
        saved = self.load(self.output())
        for changed in ('identity', 'tensor'):
            corrupt = copy.deepcopy(saved)
            if changed == 'identity':
                corrupt['identity']['seed'] += 1
            else:
                key = next(iter(corrupt['adapter_state']))
                corrupt['adapter_state'][key].reshape(-1)[0] = float('nan')
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                frozen.validate_checkpoint(corrupt, identity, model, saved['baseline_sha256'])

    def test_resume_repairs_only_lagging_new_history_and_rejects_conflict(self):
        model, base, data, identity = self.setup_models()
        self.fit(model, base, data, identity, self.output(), 1)
        saved = self.load(self.output())
        legacy.write_json(self.output()/'history.json', saved['history'][:1])
        model, base, data, identity = self.setup_models()
        result = self.fit(model, base, data, identity, self.output(), 1)
        self.assertEqual(result['optimizer_updates_this_invocation'], 0)
        self.assertEqual(legacy.read_json(self.output()/'history.json'), saved['history'])
        conflict = copy.deepcopy(saved['history'])
        conflict[0]['global_updates'] = 100
        legacy.write_json(self.output()/'history.json', conflict)
        model, base, data, identity = self.setup_models()
        with self.assertRaisesRegex(ValueError, 'history conflicts'):
            self.fit(model, base, data, identity, self.output(), 2)
        self.assertEqual(self.load(self.output())['completed_epoch'], 1)

    def test_source_companions_are_checked_without_repair(self):
        root = self.output()
        root.mkdir()
        history = copy.deepcopy(self.saved['history'])
        history[0]['validation_mae'] += 1.
        legacy.write_json(root/'history.json', history)
        legacy.write_json(root/'summary.json', legacy.read_json(self.source/'A/summary.json'))
        before = {p.name: p.read_bytes() for p in root.iterdir()}
        with self.assertRaisesRegex(ValueError, 'does not repair'):
            probe.verify_saved_sidecars(root, self.saved)
        self.assertEqual(before, {p.name: p.read_bytes() for p in root.iterdir()})

    def test_fresh_run_resets_external_rng_and_training_order(self):
        model, base, data, identity = self.setup_models()
        expected_output = self.output('_canonical')
        self.fit(model, base, data, identity, expected_output, 1)
        expected = self.load(expected_output)
        model, base, data, identity = self.setup_models()
        loader = data.loaders['train_loader']
        loader.set_order(np.arange(loader.size)[::-1])
        np.random.seed(81)
        torch.manual_seed(93)
        self.fit(model, base, data, identity, self.output('_perturbed'), 1)
        actual = self.load(self.output('_perturbed'))
        for key in ('adapter_state', 'optimizer_state', 'train_order', 'numpy_rng', 'torch_rng'):
            with self.subTest(key=key):
                self.equal_nested(expected[key], actual[key])

    def test_missing_or_corrupt_committed_baseline_blocks_resume_before_training(self):
        model, base, data, identity = self.setup_models()
        self.fit(model, base, data, identity, self.output(), 1)
        path = self.output()/'baseline_predictions.npz'
        archive_bytes = path.read_bytes()
        path.unlink()
        model, base, data, identity = self.setup_models()
        with patch.object(frozen, 'train_epoch', side_effect=AssertionError('Training after missing baseline')):
            with self.assertRaisesRegex(ValueError, 'missing baseline predictions'):
                self.fit(model, base, data, identity, self.output(), 2)
        path.write_bytes(archive_bytes)
        with np.load(path, allow_pickle=False) as archive:
            values = {name: archive[name].copy() for name in archive.files}
        values['prediction'][0, 0, 0] += 1.
        np.savez_compressed(path, **values)
        model, base, data, identity = self.setup_models()
        with patch.object(frozen, 'train_epoch', side_effect=AssertionError('Training after baseline drift')):
            with self.assertRaisesRegex(ValueError, 'Stored source baseline replay differs'):
                self.fit(model, base, data, identity, self.output(), 2)
        self.assertEqual(self.load(self.output())['completed_epoch'], 1)

    def test_protected_source_change_blocks_optimizer_before_training(self):
        model, base, data, identity = self.setup_models()
        sentinel = self.output('_source.txt')
        sentinel.write_text('original immutable source', encoding='utf-8')
        paths = {'source_sentinel': sentinel}
        identity['source_artifacts_sha256'] = probe.snapshot_files(paths)
        sentinel.write_text('changed source', encoding='utf-8')
        with patch.object(frozen, 'optimizer_for', side_effect=AssertionError('Optimizer after source drift')):
            with self.assertRaisesRegex(ValueError, 'Source artifacts changed'):
                self.fit(model, base, data, identity, self.output(), 1, protected_paths=paths)
        self.assertFalse((self.output()/'last.pt').exists())

    def test_fresh_adapter_initial_state_must_match_canonical_certificate(self):
        model, base, data, identity = self.setup_models()
        # Alter a first-layer column without touching the zero final projection.
        # Predictions remain exactly MIN, but this is no longer canonical init.
        with torch.no_grad():
            model.event_budget_adapter.A[0].weight[0, 0].add_(1.)
        with patch.object(frozen, 'source_baseline', side_effect=AssertionError('Read baseline after invalid init')), \
                patch.object(frozen, 'train_epoch', side_effect=AssertionError('Trained noncanonical adapter')):
            with self.assertRaisesRegex(ValueError, 'canonical zero-start adapter'):
                self.fit(model, base, data, identity, self.output(), 1)
        self.assertFalse((self.output()/'last.pt').exists())

    def test_valid_but_changed_training_permutation_and_epoch_zero_order_are_rejected(self):
        model, base, data, identity = self.setup_models()
        self.fit(model, base, data, identity, self.output(), 1)
        saved = self.load(self.output())
        changed = copy.deepcopy(saved)
        changed['train_order'] = np.roll(changed['train_order'], 1)
        with self.assertRaisesRegex(ValueError, 'committed digest'):
            frozen.validate_checkpoint(changed, identity, model, saved['baseline_sha256'])
        zero = copy.deepcopy(saved)
        zero.update(completed_epoch=0, history=zero['history'][:1], best_epoch=0,
                    best_metric=zero['history'][0]['validation']['on']['mae_pooled'])
        zero['train_order'] = np.arange(len(saved['train_order']), dtype=np.int64)[::-1]
        with self.assertRaisesRegex(ValueError, 'not canonical'):
            frozen.validate_checkpoint(zero, identity, model, saved['baseline_sha256'])


    def test_interrupted_epoch_zero_commit_has_baseline_and_can_restart_safely(self):
        model, base, data, identity = self.setup_models()
        root = self.output()
        with patch.object(frozen, 'save_checkpoint', side_effect=RuntimeError('injected initial commit interruption')):
            with self.assertRaisesRegex(RuntimeError, 'initial commit interruption'):
                self.fit(model, base, data, identity, root, 1)
        self.assertTrue((root/'baseline.json').is_file())
        self.assertTrue((root/'baseline_predictions.npz').is_file())
        self.assertFalse((root/'last.pt').exists())
        model, base, data, identity = self.setup_models()
        result = self.fit(model, base, data, identity, root, 1)
        self.assertEqual(result['completed_epoch'], 1)
        self.assertEqual(self.load(root)['history'][0]['epoch'], 0)
        self.assertTrue(result['off_exactly_matches_source'])


if __name__ == '__main__':
    unittest.main()
