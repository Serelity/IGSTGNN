"""Best-validation replay must precede read-only paired incident interventions."""
import copy
from contextlib import redirect_stderr, redirect_stdout
import io
import logging
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.chronological import probe_min_event_budget as probe
from experiments.chronological import train_min_event_budget as training
from experiments.chronological import train_b3_r_minimal_paper as paper
from experiments.chronological import train_internal_spatial as legacy
from experiments.chronological.smoke import sha256
from src.models.incident_spatial_residual import tensor_state_hash
from src.utils.igstgnn_published_events import EventPublishedData
from tests.test_incident_spatial import graph
from tests.test_min_event_budget_data import event_fixture
from tests.test_min_event_budget_runner import args_for, tiny_minimal


CASES = ('A_ON', 'L_ON', 'L_OFF', 'S_ON', 'S_OFF',
         'S_SEMANTIC_UNK', 'S_SEMANTIC_SHUFFLE')
SEMANTIC_KEYS = ('type', 'description', 'type_valid', 'description_valid')


class EventBudgetProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        cls.temp = tempfile.TemporaryDirectory(dir=training.REPO)
        cls.root = Path(cls.temp.name)
        cls.raw, cls.run_directory = cls.root/'data', cls.root/'run'
        cls.raw.mkdir()
        cls.run_directory.mkdir()
        samples, fixture_protocol = event_fixture(cls.raw)
        # One zero target verifies the author's actual masked-metric convention.
        samples[14]['y_data'][0, 0, 0] = -5.  # inverse transform: -5*20+100 = 0
        # This is an artificial missing-report example, never a natural no-incident claim.
        samples[16]['report_present'] = False
        np.save(cls.raw/'incident_all.npy', np.array(samples, dtype=object))
        cls.protocol = training.load_protocol(training.parse_args([]))
        for key in ('total_samples', 'train_samples', 'val_samples', 'test_samples',
                    'stations', 'batch_size', 'warm_epoch', 'cl_epoch'):
            cls.protocol[key] = fixture_protocol[key]
        cls.protocol['data_sha256'] = {name: sha256(cls.raw/name)
                                      for name in cls.protocol['data_sha256'] if (cls.raw/name).is_file()}
        cls.args = args_for(cls.run_directory)
        cls.args.data_dir = cls.raw
        cls.graph = graph(4)
        cls.roads = {'fixture:west': [0, 1], 'fixture:east': [2, 3]}
        cls.logger = logging.getLogger('min_event_probe_test')
        cls.logger.addHandler(logging.NullHandler())
        with patch.object(paper, 'build_model', side_effect=tiny_minimal), \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            data = EventPublishedData(cls.raw, cls.protocol, cls.logger, check=True)
            initialization = training.preflight(cls.args, data, cls.graph)
            cls.identity = training.make_identity(cls.args, cls.protocol,
                cls.protocol['data_sha256'], {'fixture': 'four_stations'}, initialization)
            training.verify_identity(cls.args, cls.identity, initialization, allow_new=True)
            legacy.write_json(cls.run_directory/'data_audit.json', data.audit)
            for arm in training.ARMS:
                data = EventPublishedData(cls.raw, cls.protocol, cls.logger, check=True)
                training.train_arm(cls.args, arm, cls.protocol, data,
                                   cls.identity, cls.graph, cls.logger)
        cls.saved = {arm: torch.load(cls.run_directory/arm/'last.pt', map_location='cpu', weights_only=False)
                     for arm in training.ARMS}

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def models(self):
        models = {}
        with patch.object(paper, 'build_model', side_effect=tiny_minimal):
            for arm in training.ARMS:
                model, _ = training.build_arm(self.args, arm, 4, self.graph, 'cpu')
                model.load_state_dict(self.saved[arm]['best_model_state'], strict=True)
                models[arm] = model.eval()
        return models

    def data(self):
        return EventPublishedData(self.raw, self.protocol, self.logger, check=True)

    def output(self, suffix=''):
        return self.root/(self._testMethodName+suffix)

    def protected_paths(self):
        return {str(path.relative_to(self.run_directory)): path
                for path in self.run_directory.rglob('*') if path.is_file()}

    def assert_no_complete(self, output):
        self.assertTrue((output/'failure.json').is_file())
        if (output/'summary.json').is_file():
            status = legacy.read_json(output/'summary.json').get('status', '')
            self.assertNotIn(status, ('MIN_I_VALIDATION_PROBE_PASS', 'MIN_I_ENGINEERING_PROBE_PASS'))

    def test_complete_all_arm_probe_keeps_original_batches_and_preserves_all_sources(self):
        models, data, calls = self.models(), self.data(), []
        state = {arm: tensor_state_hash(model.state_dict()) for arm, model in models.items()}
        modes = {arm: [module.training for module in model.modules()] for arm, model in models.items()}
        numpy_rng, torch_rng = np.random.get_state(), torch.get_rng_state().clone()
        paths = self.protected_paths()
        files = probe.snapshot_files(paths)
        original = probe.forward_case

        def observe(model, arm, batch, scaler, case, semantic_payload=None):
            calls.append((arm, case, len(batch['x_data'])))
            return original(model, arm, batch, scaler, case, semantic_payload=semantic_payload)

        output = self.output()
        with patch.object(probe, 'forward_case', side_effect=observe), \
                patch.object(data, 'test_loader', side_effect=AssertionError('Implicit test evaluation')), \
                patch.object(legacy, 'optimizer_for', side_effect=AssertionError('Probe constructed optimizer')), \
                redirect_stdout(io.StringIO()):
            report = probe.run_probe(models, data, self.saved, self.graph, self.roads,
                                     output, protected_paths=paths)
        self.assertTrue(report['engineering_only'])
        self.assertEqual(report['status'], 'MIN_I_ENGINEERING_PROBE_PASS')
        self.assertEqual(report['validation_rows'], 3)
        self.assertEqual(report['batch_partition'], [2, 1])
        self.assertEqual(sum(report['inference_calls'].values()), 26)
        self.assertTrue(report['final_on_exactly_restored'])
        self.assertEqual(len(calls), 26)
        replay_calls = calls[:6]
        self.assertEqual(sorted(replay_calls), sorted((arm, arm+'_ON', size)
                         for arm in training.ARMS for size in (2, 1)))
        self.assertEqual({case for _, case, _ in calls}, set(CASES))
        self.assertEqual(files, probe.snapshot_files(paths))
        for arm, model in models.items():
            self.assertEqual(state[arm], tensor_state_hash(model.state_dict()))
            self.assertEqual(modes[arm], [module.training for module in model.modules()])
            self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        np.testing.assert_array_equal(numpy_rng[1], np.random.get_state()[1])
        self.assertEqual(numpy_rng[0], np.random.get_state()[0])
        self.assertEqual(numpy_rng[2:], np.random.get_state()[2:])
        torch.testing.assert_close(torch_rng, torch.get_rng_state(), atol=0, rtol=0)
        with np.load(output/'paired_predictions.npz', allow_pickle=False) as archive:
            for case in CASES:
                self.assertEqual(archive[case].shape, (3, 12, 4))
            self.assertEqual(archive['target'].shape, (3, 12, 4))
            self.assertEqual(archive['valid'].dtype, np.bool_)
            self.assertFalse(archive['valid'][0, 0, 0])
            np.testing.assert_array_equal(archive['station_ids'], data.station_ids)
            np.testing.assert_array_equal(archive['sample_indices'], [14, 15, 16])
            # The artificial absent report is the tail batch, not an omitted row.
            for case in ('S_OFF', 'S_SEMANTIC_UNK', 'S_SEMANTIC_SHUFFLE'):
                np.testing.assert_array_equal(archive['S_ON'][2], archive[case][2])
            np.testing.assert_array_equal(archive['L_ON'][2], archive['L_OFF'][2])
            donors = archive['semantic_donor_sample_indices']
            self.assertEqual(sorted(donors.tolist()), [14, 15, 16])
            self.assertTrue(np.all(donors != archive['sample_indices']))
        for name in ('summary.json', 'replay.json', 'analysis.json', 'mechanisms.json',
                     'preservation.json', 'report.md', 'region_definitions.json'):
            self.assertTrue((output/name).is_file(), name)

    def test_late_failed_replay_of_any_arm_blocks_every_intervention(self):
        models, calls = self.models(), []
        original = probe.forward_case

        def corrupt(model, arm, batch, scaler, case, semantic_payload=None):
            calls.append(case)
            result = original(model, arm, batch, scaler, case, semantic_payload=semantic_payload)
            if case == 'S_ON' and len(batch['x_data']) == 1:
                result['prediction'] = result['prediction']+1.
            return result

        output = self.output()
        with patch.object(probe, 'forward_case', side_effect=corrupt), redirect_stdout(io.StringIO()):
            with self.assertRaises(ValueError):
                probe.run_probe(models, self.data(), self.saved, self.graph, self.roads, output)
        self.assertTrue(calls)
        self.assertTrue(all(case in ('A_ON', 'L_ON', 'S_ON') for case in calls))
        self.assert_no_complete(output)

    def test_first_paired_on_drift_is_rejected_before_off(self):
        models, counts, calls = self.models(), {}, []
        original = probe.forward_case

        def corrupt(model, arm, batch, scaler, case, semantic_payload=None):
            calls.append(case)
            counts[case] = counts.get(case, 0)+1
            result = original(model, arm, batch, scaler, case, semantic_payload=semantic_payload)
            if case == 'L_ON' and counts[case] == 3:
                result['prediction'] = result['prediction']+.5
            return result

        output = self.output()
        with patch.object(probe, 'forward_case', side_effect=corrupt), redirect_stdout(io.StringIO()):
            with self.assertRaises(ValueError):
                probe.run_probe(models, self.data(), self.saved, self.graph, self.roads, output)
        self.assertNotIn('L_OFF', calls)
        self.assertNotIn('S_OFF', calls)
        self.assert_no_complete(output)

    def test_failed_restored_on_never_writes_success(self):
        models, counts = self.models(), {}
        original = probe.forward_case
        state = {arm: tensor_state_hash(model.state_dict()) for arm, model in models.items()}

        def corrupt(model, arm, batch, scaler, case, semantic_payload=None):
            counts[case] = counts.get(case, 0)+1
            result = original(model, arm, batch, scaler, case, semantic_payload=semantic_payload)
            if case == 'L_ON' and counts[case] == 4:
                result['prediction'] = result['prediction']+.5
            return result

        output = self.output()
        with patch.object(probe, 'forward_case', side_effect=corrupt), redirect_stdout(io.StringIO()):
            with self.assertRaises(ValueError):
                probe.run_probe(models, self.data(), self.saved, self.graph, self.roads, output)
        self.assert_no_complete(output)
        for arm, model in models.items():
            self.assertEqual(state[arm], tensor_state_hash(model.state_dict()))

    def test_semantic_unknown_and_joint_shuffle_keep_relation_and_support_unchanged(self):
        models, data = self.models(), self.data()
        original, observed = probe.forward_case, []

        def observe(model, arm, batch, scaler, case, semantic_payload=None):
            result = original(model, arm, batch, scaler, case, semantic_payload=semantic_payload)
            if arm == 'S':
                observed.append((case, len(batch['x_data']), result['report_support'].copy()))
            return result

        with patch.object(probe, 'forward_case', side_effect=observe), redirect_stdout(io.StringIO()):
            probe.run_probe(models, data, self.saved, self.graph, self.roads, self.output())
        for size in (2, 1):
            baseline = next(mask for case, count, mask in observed if case == 'S_ON' and count == size)
            for case, count, mask in observed:
                if count == size and case in ('S_SEMANTIC_UNK', 'S_SEMANTIC_SHUFFLE'):
                    np.testing.assert_array_equal(mask, baseline)

    def test_joint_shuffle_preserves_pairing_validity_flags_and_all_nonsemantic_fields(self):
        payload = dict(type=np.array([0, 1, 2]), description=np.array([4, 5, 6]),
                       type_valid=np.array([True, False, True]),
                       description_valid=np.array([False, True, True]),
                       relation=np.arange(36, dtype=np.float32).reshape(3, 4, 3),
                       relation_available=np.array([True, False, True]),
                       report_present=np.array([True, True, False]))
        before, permutation = copy.deepcopy(payload), np.array([2, 0, 1])
        actual = probe.joint_shuffled_payload(payload, permutation)
        self.assertEqual(actual.keys(), payload.keys())
        for key in payload:
            expected = payload[key][permutation] if key in SEMANTIC_KEYS else payload[key]
            np.testing.assert_array_equal(actual[key], expected)
            np.testing.assert_array_equal(payload[key], before[key])
        actual['type'][0] = 100
        np.testing.assert_array_equal(payload['type'], before['type'])

    def test_permutation_is_seeded_derangement_without_global_rng_mutation(self):
        numpy_rng, torch_rng = np.random.get_state(), torch.get_rng_state().clone()
        for count in (2, 3, 17):
            actual = probe.semantic_permutation(count, 20261013)
            np.testing.assert_array_equal(np.sort(actual), np.arange(count))
            self.assertTrue(np.all(actual != np.arange(count)))
            np.testing.assert_array_equal(actual, probe.semantic_permutation(count, 20261013))
        for count in (0, 1):
            with self.assertRaises(ValueError):
                probe.semantic_permutation(count, 20261013)
        np.testing.assert_array_equal(numpy_rng[1], np.random.get_state()[1])
        self.assertEqual(numpy_rng[2:], np.random.get_state()[2:])
        torch.testing.assert_close(torch_rng, torch.get_rng_state(), atol=0, rtol=0)

    def test_loaded_last_weights_instead_of_best_or_training_mode_are_rejected(self):
        for defect in ('wrong_weights', 'training_mode', 'gradient'):
            with self.subTest(defect=defect):
                models = self.models()
                if defect == 'wrong_weights':
                    with torch.no_grad():
                        next(models['L'].parameters()).add_(1.)
                elif defect == 'training_mode':
                    models['L'].train()
                else:
                    parameter = next(models['L'].parameters())
                    parameter.grad = torch.zeros_like(parameter)
                output = self.output('_'+defect)
                with patch.object(probe, 'forward_case', side_effect=AssertionError('Invalid model reached inference')):
                    with self.assertRaises(ValueError):
                        probe.run_probe(models, self.data(), self.saved, self.graph, self.roads, output)
                self.assert_no_complete(output)

    def test_best_weights_are_used_even_when_checkpoint_last_weights_differ(self):
        saved = copy.deepcopy(self.saved)
        for checkpoint in saved.values():
            with torch.no_grad():
                next(iter(checkpoint['model_state'].values())).add_(1.)
        with redirect_stdout(io.StringIO()):
            result = probe.run_probe(self.models(), self.data(), saved,
                                     self.graph, self.roads, self.output())
        self.assertTrue(result['final_on_exactly_restored'])

    def test_source_snapshot_detects_mutation_and_fails_without_rewriting_source(self):
        sentinel = self.root/'protected_sentinel.txt'
        sentinel.write_text('original', encoding='utf-8')
        paths, original = {'sentinel': sentinel}, probe.forward_case
        before = probe.snapshot_files(paths)

        def mutate(model, arm, batch, scaler, case, semantic_payload=None):
            result = original(model, arm, batch, scaler, case, semantic_payload=semantic_payload)
            if case == 'L_OFF':
                sentinel.write_text('changed by injected failure', encoding='utf-8')
            return result

        output = self.output()
        with patch.object(probe, 'forward_case', side_effect=mutate), redirect_stdout(io.StringIO()):
            with self.assertRaises(ValueError):
                probe.run_probe(self.models(), self.data(), self.saved,
                                self.graph, self.roads, output, protected_paths=paths)
        self.assert_no_complete(output)
        self.assertNotEqual(before, probe.snapshot_files(paths))
        self.assertEqual(sentinel.read_text(encoding='utf-8'), 'changed by injected failure')
        with self.assertRaises(ValueError):
            probe.require_preserved(before, paths)

    def test_existing_probe_directory_is_not_overwritten(self):
        output = self.output()
        output.mkdir()
        sentinel = output/'existing.txt'
        sentinel.write_text('preserve me', encoding='utf-8')
        with self.assertRaises((ValueError, FileExistsError)):
            probe.run_probe(self.models(), self.data(), self.saved, self.graph, self.roads, output)
        self.assertEqual(sentinel.read_text(encoding='utf-8'), 'preserve me')

    def test_checkpoint_companion_conflicts_are_rejected_without_repair(self):
        source = self.run_directory/'L'
        self.assertEqual(probe.verify_saved_sidecars(source, self.saved['L'])['status'],
                         'CHECKPOINT_COMPANIONS_MATCH')
        for defect in ('history', 'completed_epoch', 'global_updates'):
            with self.subTest(defect=defect):
                directory = self.output('_'+defect)
                directory.mkdir()
                history = legacy.read_json(source/'history.json')
                summary = legacy.read_json(source/'summary.json')
                if defect == 'history':
                    history[0]['validation_mae'] += 1.
                else:
                    summary[defect] += 1
                legacy.write_json(directory/'history.json', history)
                legacy.write_json(directory/'summary.json', summary)
                paths = {name: directory/name for name in ('history.json', 'summary.json')}
                before = probe.snapshot_files(paths)
                with self.assertRaisesRegex(ValueError, 'does not repair'):
                    probe.verify_saved_sidecars(directory, self.saved['L'])
                self.assertEqual(before, probe.snapshot_files(paths))

    def test_changed_runtime_thread_count_or_determinism_policy_is_rejected(self):
        identity = copy.deepcopy(self.identity)
        policy = SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False)),
                                 cudnn=SimpleNamespace(allow_tf32=False))
        with patch.object(torch, 'get_num_threads', return_value=identity['threads']), \
                patch.object(torch, 'are_deterministic_algorithms_enabled', return_value=True), \
                patch.object(torch, 'backends', policy):
            probe.verify_runtime(identity, 'cpu')
            changed = copy.deepcopy(identity)
            changed['runtime']['torch'] = 'unrecorded-runtime'
            with self.assertRaisesRegex(ValueError, 'identity differs'):
                probe.verify_runtime(changed, 'cpu')
            with patch.object(torch, 'get_num_threads', return_value=identity['threads']+1):
                with self.assertRaisesRegex(ValueError, 'thread count differs'):
                    probe.verify_runtime(identity, 'cpu')
            with patch.object(torch, 'are_deterministic_algorithms_enabled', return_value=False):
                with self.assertRaisesRegex(ValueError, 'deterministic/TF32 policy'):
                    probe.verify_runtime(identity, 'cpu')
            with patch.object(torch.backends.cuda.matmul, 'allow_tf32', True):
                with self.assertRaisesRegex(ValueError, 'deterministic/TF32 policy'):
                    probe.verify_runtime(identity, 'cpu')

    def test_changed_training_sources_or_protocol_stop_before_release_loading(self):
        with patch.object(training, 'source_fingerprints', return_value={'changed-source': 'different'}), \
                patch.object(paper, 'verify_release', side_effect=AssertionError('Read release after invalid source')):
            with self.assertRaisesRegex(ValueError, 'source fingerprints changed'):
                probe.verify_identity_sources(self.args, self.identity)
        identity = copy.deepcopy(self.identity)
        identity['protocol_sha256'] = 'changed-protocol'
        with patch.object(paper, 'verify_release', side_effect=AssertionError('Read release after invalid protocol')):
            with self.assertRaisesRegex(ValueError, 'protocol fingerprint changed'):
                probe.verify_identity_sources(self.args, identity)

    def test_forward_exception_removes_probe_hooks_and_preserves_models(self):
        models = self.models()
        hooks = {arm: {name: (tuple(module._forward_hooks), tuple(module._forward_pre_hooks))
                       for name, module in model.named_modules()} for arm, model in models.items()}
        states = {arm: tensor_state_hash(model.state_dict()) for arm, model in models.items()}
        output = self.output()
        with patch.object(models['L'].event_budget_adapter, 'forward', side_effect=RuntimeError('probe fixture forward failure')), \
                redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, 'probe fixture forward failure'):
                probe.run_probe(models, self.data(), self.saved, self.graph, self.roads, output)
        self.assert_no_complete(output)
        for arm, model in models.items():
            self.assertEqual(states[arm], tensor_state_hash(model.state_dict()))
            self.assertEqual(hooks[arm], {name: (tuple(module._forward_hooks), tuple(module._forward_pre_hooks))
                                         for name, module in model.named_modules()})

    def test_forward_cannot_mutate_nonzero_relation_even_if_support_mask_stays_the_same(self):
        models = self.models()
        original = models['L'].forward

        def mutate(history, *args, **kwargs):
            result = original(history, *args, **kwargs)
            event = kwargs.get('incident_data')
            if event is not None and kwargs.get('report_enabled') is False:
                event['relation'].mul_(2.)
            return result

        output = self.output()
        with patch.object(models['L'], 'forward', side_effect=mutate), redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, 'input|payload|mutat'):
                probe.run_probe(models, self.data(), self.saved, self.graph, self.roads, output)
        self.assert_no_complete(output)

    def test_input_mutation_is_detected_even_when_model_forward_also_raises(self):
        model, data = self.models()['L'], self.data()
        batch = next(data.loaders['val_loader'].get_iterator())
        before_input = probe.input_digest(batch)
        before_state = tensor_state_hash(model.state_dict())
        before_hooks = {name: (tuple(module._forward_hooks), tuple(module._forward_pre_hooks))
                        for name, module in model.named_modules()}

        def mutate_then_raise(history, *args, **kwargs):
            kwargs['incident_data']['relation'].mul_(2.)
            raise RuntimeError('model failed after mutating input')

        with patch.object(model, 'forward', side_effect=mutate_then_raise):
            with self.assertRaisesRegex(ValueError, 'changed traffic/target/event input') as caught:
                probe.forward_case(model, 'L', batch, data.scaler, 'L_ON')
        self.assertIsInstance(caught.exception.__context__, RuntimeError)
        self.assertNotEqual(before_input, probe.input_digest(batch))
        self.assertEqual(before_state, tensor_state_hash(model.state_dict()))
        self.assertEqual(before_hooks,
                         {name: (tuple(module._forward_hooks), tuple(module._forward_pre_hooks))
                          for name, module in model.named_modules()})


if __name__ == '__main__':
    unittest.main()
