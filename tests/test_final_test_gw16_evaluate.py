"""Server-only tests of selection boundaries and the paired paper metric axes."""
import copy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.final_test_gw16 import evaluate as entry


def identity_fixture():
    identity = dict(engineering_only=False, config=dict(max_epochs=100),
                    data=dict(hashes=entry.RELEASE.copy()))
    role = dict(selection='best', epoch=99, completed_training_epochs=100, full_protocol_reached=True)
    candidate = dict(identity=identity, arm='STATE_EXCHANGE', engineering_only=False,
        split='val', test_evaluated=False, validation_replay=dict(passed=True), checkpoint_role=role)
    original = dict(engineering_only=False, original_full_model=True, native_pos=False,
        test_evaluated=False, native_replay=dict(passed=True), source=dict(training_complete=True),
        data_identity=dict(hashes=entry.RELEASE.copy()))
    return identity, candidate, original, role


def arrays(prediction_offset=0.):
    target = np.full((839, 12, 496, 1), 100., dtype=np.float32)
    return dict(target=target, prediction=target+np.float32(prediction_offset),
                sample_ids=np.arange(4746, 5585, dtype=np.int64),
                station_ids=np.arange(496, dtype=np.int64))


class FinalTestGW16SelectionTests(unittest.TestCase):
    def guard(self, values):
        # Identity hashing itself is covered by existing GW16 tests. Here isolate
        # the newly introduced completion, role, baseline and split boundaries.
        with patch.object(entry, 'validate_identity'):
            entry.selection_guard(*values)

    def test_only_full_validation_best_and_original_are_accepted(self):
        self.guard(identity_fixture())
        for mutate in (
            lambda v: v[0].update(engineering_only=True),
            lambda v: v[0]['config'].update(max_epochs=20),
            lambda v: v[3].update(selection='last'),
            lambda v: v[3].update(completed_training_epochs=99),
            lambda v: v[3].update(full_protocol_reached=False),
            lambda v: v[1].update(arm='NATIVE_POS'),
            lambda v: v[1].update(split='test'),
            lambda v: v[1].update(test_evaluated=True),
            lambda v: v[2].update(native_pos=True),
            lambda v: v[2]['source'].update(training_complete=False),
            lambda v: v[2]['data_identity']['hashes'].update(**{'sensors.csv':'changed'}),
        ):
            with self.subTest(mutate=mutate):
                values = copy.deepcopy(identity_fixture())
                mutate(values)
                with self.assertRaises(ValueError):
                    self.guard(values)

    def test_validation_cache_must_match_the_selected_checkpoint_role(self):
        values = list(copy.deepcopy(identity_fixture()))
        values[1]['checkpoint_role'] = dict(values[3], epoch=98)
        with self.assertRaisesRegex(ValueError, 'validation BEST'):
            self.guard(values)

    def test_output_cannot_modify_inputs_or_their_parents(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            run, validation, data = (root/name for name in ('run', 'val', 'data'))
            for output in (run, run/'test', root, data/'result', validation/'test'):
                with self.subTest(output=output), self.assertRaises(ValueError):
                    entry.safe_output(output, (run, validation), data)
            self.assertEqual(entry.safe_output(root/'paired_test', (run, validation), data),
                             (root/'paired_test').resolve())


class FinalTestGW16PairedMetricsTests(unittest.TestCase):
    def setUp(self):
        self.previous_threads = torch.get_num_threads()
        torch.set_num_threads(3)
        self.addCleanup(torch.set_num_threads, self.previous_threads)

    def test_rmse_is_per_horizon_then_mean_and_both_methods_share_the_target(self):
        original, candidate = arrays(), arrays()
        error = np.arange(1, 13, dtype=np.float32)[None, :, None, None]
        original['prediction'] += error
        candidate['prediction'] += error*.5
        result = entry.paired_scores(original, candidate, dict(target_minimum=.001, mape_minimum=1.))
        native = result['rows']['ORIGINAL_IGSTGNN']['author_metrics']
        self.assertAlmostEqual(native['horizon_macro']['rmse'], 6.5, places=5)
        self.assertGreater(native['pooled']['rmse'], 7.)
        self.assertAlmostEqual(result['relative_gain_percent_vs_replayed_original']['mae'], 50., places=4)
        self.assertEqual(len(native['per_horizon']), 12)
        self.assertEqual(native['mask_value'], 0.)

    def test_matching_shape_cannot_hide_changed_sample_station_or_target_axes(self):
        original = arrays(2.)
        for key in ('sample_ids', 'station_ids', 'target'):
            candidate = arrays(1.)
            if key in ('sample_ids', 'station_ids'):
                candidate[key] = candidate[key][::-1].copy()
            else:
                candidate[key][0, 0, 0, 0] += 1.
            with self.subTest(key=key), self.assertRaises(ValueError):
                entry.paired_scores(original, candidate, dict(target_minimum=.001, mape_minimum=1.))

    def test_original_cached_tensor_must_match_the_forward_that_was_validated(self):
        state = dict(weight=torch.ones(1))
        source = dict(kind='checkpoint_and_log', path='model.pt', log_path='record.log',
            checkpoint_sha256='a', training_log_sha256='b', training_complete=True,
            completed_epoch=47, best_epoch=27, reference_level='fully_trained_original_protocol')
        metadata = dict(source=source, training_config=dict(seed=2025), registered_parameters=455220,
            expected_validation_mae=13.8, native_replay=dict(model_state_sha256='different'))
        archive = SimpleNamespace(config=metadata['training_config'], state_dict=state,
            metadata=dict(parameter_count=455220, counts=dict(train=3909, val=837, node_num=496)),
            best_validation_mae=13.8, provenance=source)
        with patch.object(entry, 'load_archive', return_value=archive), patch.object(
                entry, 'archive_reference', return_value=source), self.assertRaisesRegex(ValueError, 'model tensors'):
            entry.original_weights(metadata)

    def test_cache_cannot_relabel_both_station_axes_together(self):
        saved = arrays(1.)
        saved['station_ids'] = saved['station_ids'][::-1].copy()
        audit = dict(model_tensor_sha256='frozen', rng_unchanged=True,
            registered_tensors_unchanged=True, raw_inputs_unchanged=True,
            future_target_passed_to_forward=False, optimizer_updates=0, forward_batches=18)
        material = dict(hashes=entry.RELEASE.copy(), bounds_half_open=[4746, 5585],
            count=839, batch_size=48, normalization=dict(mean=100., std=20., refitted=False),
            station_ids=list(range(496)))
        frozen = dict(original_weights_sha256='frozen', test_protocol=dict(station_ids=list(range(496))))
        data = SimpleNamespace(mean=100., std=20., station_ids=np.arange(496))
        with self.assertRaisesRegex(ValueError, 'station axis'):
            entry.validate_cached(saved, dict(audit=audit, data_identity=material), frozen,
                                  'ORIGINAL_IGSTGNN', data, None)

    def test_original_forward_never_receives_target_or_enables_gradients(self):
        class Spy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.ones(1))
                self.calls = 0

            def forward(self, x, label=None, incident_data=None, sensor_data=None):
                self.calls += 1
                if label is not None or torch.is_grad_enabled() or self.training:
                    raise AssertionError('Future target or training mode reached native forward')
                return torch.ones((len(x), 12, 496, 1), dtype=torch.float32)

        def batches():
            for count in [48]*17+[23]:
                yield dict(x_data=np.zeros((count, 1), dtype=np.float32),
                           y_data=np.full((count, 12, 496, 1), 100., dtype=np.float32))

        loader = SimpleNamespace(num_batch=18, size=839, get_iterator=batches)
        data = SimpleNamespace(station_ids=np.arange(496), scaler=SimpleNamespace(inverse_transform=lambda x:x))
        model = Spy()
        batch = lambda raw, device:(torch.as_tensor(raw['x_data']), torch.as_tensor(raw['y_data']), {}, {})
        with patch.object(entry, 'original_batch', side_effect=batch):
            result, audit = entry.collect('ORIGINAL_IGSTGNN', model, loader, data, 'cpu',
                                         SimpleNamespace(info=lambda *args:None))
        self.assertEqual(model.calls, 18)
        self.assertEqual(result['prediction'].shape, (839, 12, 496, 1))
        self.assertFalse(audit['future_target_passed_to_forward'])
        self.assertTrue(audit['rng_unchanged'])
        self.assertEqual(audit['optimizer_updates'], 0)


if __name__ == '__main__':
    unittest.main()
