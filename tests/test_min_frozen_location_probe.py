"""A fixed-support location intervention must follow exact full validation replay."""
from contextlib import redirect_stderr, redirect_stdout
import io
import logging
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.chronological import probe_min_frozen_location as location
from experiments.chronological import probe_min_event_budget as original_probe
from experiments.chronological import train_min_event_budget as training
from experiments.chronological import train_internal_spatial as legacy
from experiments.chronological.min_frozen_budget_metrics import select_hard_windows
from src.models.incident_spatial_frozen_budget import build_frozen_location, frozen_state_hash
from src.models.incident_spatial_residual import tensor_state_hash
from src.utils.igstgnn_published_events import EventPublishedData
from tests.test_incident_spatial import graph
from tests.test_min_event_budget_data import event_fixture
from tests.test_min_event_budget_runner import args_for, tiny_minimal


class FrozenLocationProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        cls.temp = tempfile.TemporaryDirectory(dir=training.REPO)
        cls.root = Path(cls.temp.name)
        cls.raw = cls.root/'data'
        cls.raw.mkdir()
        samples, fixture_protocol = event_fixture(cls.raw)
        # Original valid-mask behavior and the unpadded one-row tail both matter.
        samples[14]['y_data'][0, 0, 0] = -5.
        # The absent-report tail retains nonzero raw relations. Its support is
        # empty because presence is false, so those relations may not be moved.
        samples[16]['report_present'] = False
        np.save(cls.raw/'incident_all.npy', np.array(samples, dtype=object))
        cls.protocol = training.load_protocol(training.parse_args([]))
        for key in ('total_samples', 'train_samples', 'val_samples', 'test_samples',
                    'stations', 'batch_size', 'warm_epoch', 'cl_epoch'):
            cls.protocol[key] = fixture_protocol[key]
        cls.logger = logging.getLogger('frozen_location_probe_tests')
        cls.logger.addHandler(logging.NullHandler())

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def data(self):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return EventPublishedData(self.raw, self.protocol, self.logger, check=True)

    def fixture(self):
        args = args_for(self.root)
        base, _ = tiny_minimal(args, 4, graph(4), 'cpu')
        model, _ = build_frozen_location(base.eval())
        # Fixed nonzero saved adapter, not a claimed predictive training result.
        with torch.no_grad():
            model.event_budget_adapter.A[-1].weight.fill_(.03)
            model.event_budget_adapter.A[-1].bias.fill_(.01)
        model.eval()
        data = self.data()
        on, off, target, mechanisms = [], [], [], {}
        for batch in data.loaders['val_loader'].get_iterator():
            real = original_probe.forward_case(model, 'L', batch, data.scaler, 'L_ON')
            disabled = original_probe.forward_case(model, 'L', batch, data.scaler, 'L_OFF')
            on.append(real['prediction'])
            off.append(disabled['prediction'])
            target.append(real['target'])
            original_probe.merge_mechanisms(mechanisms, real['mechanisms'])
        on, off, target = map(np.concatenate, (on, off, target))
        null = float(target.min()) if float(target.min()) < 1 else 0.
        valid = np.isfinite(target) & (target != null)
        indices = data.audit['counts']['train_samples']+np.arange(len(target), dtype=np.int64)
        hard = select_hard_windows(off, target, valid, indices, fraction=.10)
        best = dict(on=on, off=off, target=target, valid=valid,
                    sample_indices=indices, fixed_hard_indices=hard)
        # Internal 0--1--2 component; station 3 is isolated. This is a fixture
        # candidate envelope, never a certified physical accident impact area.
        direct = np.array([[True, False, True, False],
                           [True, False, True, False],
                           [False, False, False, False]])
        potential = np.array([[False, True, False, False],
                              [False, True, False, False],
                              [False, False, False, False]])
        regions = dict(direct_report_nodes=direct, potential_only=potential,
                       remaining_nodes=~(direct | potential))
        return model, data, best, regions, mechanisms

    def output(self, suffix=''):
        path = self.root/(self._testMethodName+suffix)
        path.mkdir()
        return path

    def run_case(self, model, data, best, regions, output, **kwargs):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return location.run_probe(model, data, best, regions, output,
                frozen_hash=frozen_state_hash(model), **kwargs)

    def assert_no_success(self, output):
        for name in ('summary.json',):
            if (output/name).is_file():
                status = legacy.read_json(output/name).get('status', '')
                self.assertNotIn('PASS', status)
                self.assertNotIn('COMPLETE', status)

    def test_complete_probe_has_exact_axes_original_tail_and_no_training_or_test(self):
        model, data, best, regions, mechanisms = self.fixture()
        before = tensor_state_hash(model.state_dict())
        rng = original_probe.rng_digest('cpu')
        batch_digests = [original_probe.input_digest(batch)
                         for batch in data.loaders['val_loader'].get_iterator()]
        calls = []
        forward = original_probe.forward_case

        def observe(*args, **kwargs):
            calls.append((args[4], len(args[2]['x_data'])))
            return forward(*args, **kwargs)

        output = self.output()
        with patch.object(original_probe, 'forward_case', side_effect=observe), \
                patch.object(data, 'test_loader', side_effect=AssertionError('Implicit test evaluation')), \
                patch.object(torch.optim, 'Adam', side_effect=AssertionError('Constructed optimizer')):
            result = self.run_case(model, data, best, regions, output, expected_mechanisms=mechanisms)
        self.assertIn('PASS', result['status'])
        self.assertEqual(sum(result['inference_calls'].values()), 10)
        self.assertEqual(len(calls), 10)
        self.assertEqual(calls[:4], [('L_ON', 2), ('L_OFF', 2), ('L_ON', 1), ('L_OFF', 1)])
        self.assertEqual(tensor_state_hash(model.state_dict()), before)
        self.assertEqual(original_probe.rng_digest('cpu'), rng)
        self.assertEqual(batch_digests, [original_probe.input_digest(batch)
                         for batch in data.loaders['val_loader'].get_iterator()])
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        self.assertTrue(all(not module.training for module in model.modules()))
        with np.load(output/'paired_predictions.npz', allow_pickle=False) as archive:
            for key in ('REAL', 'SHUFFLED', 'MIN'):
                self.assertEqual(archive[key].shape, (3, 12, 4))
            for key in ('target', 'valid', 'sample_indices', 'fixed_hard_indices'):
                np.testing.assert_array_equal(archive[key], best[key])
            np.testing.assert_array_equal(archive['REAL'], best['on'])
            np.testing.assert_array_equal(archive['MIN'], best['off'])
            np.testing.assert_array_equal(archive['station_ids'], data.station_ids)
            np.testing.assert_array_equal(archive['REAL'][2], archive['SHUFFLED'][2])
            np.testing.assert_array_equal(archive['REAL'][2], archive['MIN'][2])
        with np.load(output/'shuffle_mapping.npz', allow_pickle=False) as archive:
            self.assertEqual(archive['donor_station_indices'].shape, (3, 4))
            np.testing.assert_array_equal(archive['sample_indices'], best['sample_indices'])
            np.testing.assert_array_equal(archive['station_ids'], data.station_ids)
        for name in ('analysis.json', 'replay.json', 'shuffle_audit.json', 'mechanisms.json',
                     'model_preservation.json', 'report.md'):
            self.assertTrue((output/name).is_file(), name)
        # Only main may publish success after its protected-file audit.
        self.assertFalse((output/'summary.json').exists())

    def test_late_initial_real_replay_failure_blocks_shuffle_construction(self):
        model, data, best, regions, _ = self.fixture()
        forward = original_probe.forward_case

        def corrupt(*args, **kwargs):
            result = forward(*args, **kwargs)
            if args[4] == 'L_ON' and len(args[2]['x_data']) == 1:
                result['prediction'] = result['prediction']+1.
            return result

        output = self.output()
        with patch.object(original_probe, 'forward_case', side_effect=corrupt), \
                patch.object(location, 'make_location_shuffle', side_effect=AssertionError('Shuffled before full real replay')):
            with self.assertRaises(ValueError):
                self.run_case(model, data, best, regions, output)
        self.assert_no_success(output)

    def test_late_initial_off_replay_failure_blocks_shuffle_construction(self):
        model, data, best, regions, _ = self.fixture()
        forward = original_probe.forward_case

        def corrupt(*args, **kwargs):
            result = forward(*args, **kwargs)
            if args[4] == 'L_OFF' and len(args[2]['x_data']) == 1:
                result['prediction'] = result['prediction']+1.
            return result

        output = self.output()
        with patch.object(original_probe, 'forward_case', side_effect=corrupt), \
                patch.object(location, 'make_location_shuffle', side_effect=AssertionError('Shuffled before full OFF replay')):
            with self.assertRaises(ValueError):
                self.run_case(model, data, best, regions, output)
        self.assert_no_success(output)

    def test_paired_real_drift_blocks_first_shuffle_forward(self):
        model, data, best, regions, _ = self.fixture()
        forward, on_calls = original_probe.forward_case, []

        def corrupt(*args, **kwargs):
            result = forward(*args, **kwargs)
            if args[4] == 'L_ON':
                on_calls.append(original_probe.input_digest(args[2]))
                if len(on_calls) == 3:
                    result['prediction'] = result['prediction']+1.
            return result

        output = self.output()
        with patch.object(original_probe, 'forward_case', side_effect=corrupt):
            with self.assertRaises(ValueError):
                self.run_case(model, data, best, regions, output)
        self.assertEqual(len(on_calls), 3)
        self.assert_no_success(output)

    def test_failed_restoration_never_emits_success_and_preserves_model(self):
        model, data, best, regions, _ = self.fixture()
        before, rng = tensor_state_hash(model.state_dict()), original_probe.rng_digest('cpu')
        forward, on_calls = original_probe.forward_case, 0

        def corrupt(*args, **kwargs):
            nonlocal on_calls
            result = forward(*args, **kwargs)
            if args[4] == 'L_ON':
                on_calls += 1
                if on_calls == 5:
                    result['prediction'] = result['prediction']+1.
            return result

        output = self.output()
        with patch.object(original_probe, 'forward_case', side_effect=corrupt):
            with self.assertRaises(ValueError):
                self.run_case(model, data, best, regions, output)
        self.assert_no_success(output)
        self.assertEqual(tensor_state_hash(model.state_dict()), before)
        self.assertEqual(original_probe.rng_digest('cpu'), rng)
        self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules()))

    def test_mechanism_certificate_mismatch_blocks_shuffle_construction(self):
        model, data, best, regions, mechanisms = self.fixture()
        mechanisms['capacity_change_abs_sum'] += 1.
        output = self.output()
        with patch.object(location, 'make_location_shuffle', side_effect=AssertionError('Shuffled before mechanism check')):
            with self.assertRaises(ValueError):
                self.run_case(model, data, best, regions, output, expected_mechanisms=mechanisms)
        self.assert_no_success(output)

    def test_shuffle_exception_leaves_no_hooks_modes_gradients_or_random_changes(self):
        model, data, best, regions, _ = self.fixture()
        before, rng = tensor_state_hash(model.state_dict()), original_probe.rng_digest('cpu')
        forward, on_calls = original_probe.forward_case, 0

        def explode(*args, **kwargs):
            nonlocal on_calls
            if args[4] == 'L_ON':
                on_calls += 1
                if on_calls == 4:
                    raise RuntimeError('Synthetic shuffle inference failure')
            return forward(*args, **kwargs)

        output = self.output()
        with patch.object(original_probe, 'forward_case', side_effect=explode):
            with self.assertRaisesRegex(RuntimeError, 'Synthetic shuffle'):
                self.run_case(model, data, best, regions, output)
        self.assert_no_success(output)
        self.assertEqual(tensor_state_hash(model.state_dict()), before)
        self.assertEqual(original_probe.rng_digest('cpu'), rng)
        self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules()))
        self.assertTrue(all(not m.training for m in model.modules()))
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_mutating_original_relation_input_is_rejected(self):
        model, data, best, regions, _ = self.fixture()
        forward = original_probe.forward_case

        def corrupt(*args, **kwargs):
            result = forward(*args, **kwargs)
            args[2]['event_data']['relation'][0, 0, 1] += .125
            return result

        output = self.output()
        with patch.object(original_probe, 'forward_case', side_effect=corrupt):
            with self.assertRaises(ValueError):
                self.run_case(model, data, best, regions, output)
        self.assert_no_success(output)

    def test_mutating_plain_graph_evidence_is_rejected(self):
        model, data, best, regions, _ = self.fixture()
        forward = original_probe.forward_case

        def corrupt(*args, **kwargs):
            result = forward(*args, **kwargs)
            model.internal_spatial_graph.evidence_scope = 'candidate_unverified'
            return result

        output = self.output()
        with patch.object(original_probe, 'forward_case', side_effect=corrupt), \
                patch.object(location, 'make_location_shuffle', side_effect=AssertionError('Shuffled after initial graph mutation')):
            with self.assertRaises(ValueError):
                self.run_case(model, data, best, regions, output)
        self.assert_no_success(output)

    def test_initial_global_rng_mutation_blocks_shuffle_construction(self):
        model, data, best, regions, _ = self.fixture()
        forward = original_probe.forward_case
        rng_state = np.random.get_state()

        def corrupt(*args, **kwargs):
            result = forward(*args, **kwargs)
            np.random.random()
            return result

        output = self.output()
        try:
            with patch.object(original_probe, 'forward_case', side_effect=corrupt), \
                    patch.object(location, 'make_location_shuffle', side_effect=AssertionError('Shuffled after initial RNG mutation')):
                with self.assertRaises(ValueError):
                    self.run_case(model, data, best, regions, output)
        finally:
            # The test injects a mutation; rejection must not pollute other tests.
            np.random.set_state(rng_state)
        self.assert_no_success(output)

    def test_training_mode_or_existing_gradients_rejected_before_any_inference(self):
        for bad in ('training', 'gradient'):
            with self.subTest(bad=bad):
                model, data, best, regions, _ = self.fixture()
                if bad == 'training':
                    model.train()
                else:
                    parameter = next(model.event_budget_adapter.parameters())
                    parameter.grad = torch.ones_like(parameter)
                with patch.object(original_probe, 'forward_case', side_effect=AssertionError('Invalid model was inferred')):
                    with self.assertRaises(ValueError):
                        self.run_case(model, data, best, regions, self.output('_'+bad))


if __name__ == '__main__':
    unittest.main()
