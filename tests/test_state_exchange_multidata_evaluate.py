"""Server-only paired metric, freeze and immutable cache boundary tests."""
import copy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.state_exchange_multidata import evaluate as entry


def material(split='val'):
    return dict(dataset='Alameda', split=split, engineering_only=False,
        count=4, nodes=3, bounds_half_open=[7, 11], station_ids=[101, 103, 107],
        release_hashes={'published.npy':'data'})


def arrays(offset=1.):
    target = np.full((4, 12, 3, 1), 100., dtype=np.float32)
    return dict(prediction=target+np.float32(offset), target=target,
        sample_ids=np.arange(7, 11, dtype=np.int64), station_ids=np.array([101, 103, 107], dtype=np.int64))


def frozen(split='val'):
    payload = dict(protocol=material(split), runtime=dict(threads=3), execution_source={'model.py':'code'},
        source_files={}, metric_contract=entry.METRIC_CONTRACT,
        checkpoint_roles={case:dict(model_tensor_sha256='weights', registered_parameters=100)
                          for case in entry.CASES})
    return dict(schema=entry.LOCK_SCHEMA, freeze_id=entry.digest_json(payload), payload=payload)


def audit_metadata(split='val'):
    return dict(audit=dict(model_tensor_sha256='weights', rng_unchanged=True,
        registered_tensors_unchanged=True, raw_inputs_unchanged=True,
        future_target_passed_to_forward=False, optimizer_updates=0, forward_batches=2),
        validation_replay=dict(passed=True) if split == 'val' else dict(applicable=False))


def identities():
    candidate = dict(schema=entry.SCHEMA, engineering_only=False, seed=2025,
        config=dict(max_epochs=100), data=dict(dataset='Alameda', hashes={'published.npy':'data'}),
        execution_source={'model.py':'code'})
    baseline = dict(schema=entry.BASELINE_SCHEMA, engineering_only=False, seed=2025,
        data=copy.deepcopy(candidate['data']), execution_source={'model.py':'code'})
    candidate_role = dict(selection='best', completed_training_epochs=100, full_protocol_reached=True)
    baseline_role = dict(selection='best', training_complete=True)
    return candidate, baseline, candidate_role, baseline_role


class MultiDatasetSelectionTests(unittest.TestCase):
    def guard(self, values, dataset='Alameda', split='test', engineering=False):
        with patch.object(entry, 'validate_identity'), patch.object(entry, 'spec', return_value={
                'hashes':{'published.npy':'data'}}), patch.object(entry, 'source_hashes', return_value={'model.py':'code'}):
            entry.selection_guard(dataset, split, *values, engineering)

    def test_full_best_is_eligible_for_explicit_test(self):
        self.guard(identities())

    def test_partial_candidate_or_baseline_and_last_are_rejected(self):
        for change in (
            lambda v:v[2].update(completed_training_epochs=99),
            lambda v:v[2].update(full_protocol_reached=False),
            lambda v:v[3].update(training_complete=False),
            lambda v:v[2].update(selection='last'),
            lambda v:v[3].update(selection='last'),
        ):
            values = copy.deepcopy(identities())
            change(values)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.guard(values)

    def test_engineering_test_and_contra_new_forward_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Engineering test'):
            self.guard(identities(), engineering=True)
        with self.assertRaisesRegex(ValueError, 'Existing Contra'):
            self.guard(identities(), dataset='Contra_Costa')

    def test_seed_release_source_and_dataset_cannot_silently_differ(self):
        for change in (
            lambda v:v[1].update(seed=2026),
            lambda v:v[1]['data'].update(dataset='Orange'),
            lambda v:v[1]['data']['hashes'].update(**{'published.npy':'changed'}),
            lambda v:v[1]['execution_source'].update(**{'model.py':'changed'}),
        ):
            values = copy.deepcopy(identities())
            change(values)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.guard(values)

    def test_cli_engineering_has_no_test_escape(self):
        with self.assertRaises(SystemExit):
            entry.parse_args(['--dataset', 'Alameda', '--data-dir', 'data', '--candidate-run', 'candidate',
                '--baseline-run', 'baseline', '--output-dir', 'output', '--split', 'test', '--engineering'])

    def test_output_never_overlaps_training_or_data(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            candidate, baseline, data = (root/name for name in ('candidate', 'baseline', 'data'))
            for output in (root, candidate, candidate/'eval', baseline/'eval', data/'eval'):
                with self.subTest(output=output), self.assertRaises(ValueError):
                    entry.safe_output(output, (candidate, baseline), data)
            self.assertEqual(entry.safe_output(root/'eval', (candidate, baseline), data), root/'eval')

    def test_protocol_uses_registered_rounding_and_explicit_split(self):
        release = dict(train_samples=7, val_samples=4, test_samples=5, nodes=3, hashes={'x':'hash'})
        with patch.object(entry, 'spec', return_value=release):
            self.assertEqual(entry.protocol('Alameda', 'val', False, [1, 2, 3])['bounds_half_open'], [7, 11])
            self.assertEqual(entry.protocol('Alameda', 'test', False, [1, 2, 3])['bounds_half_open'], [11, 16])


class MultiDatasetPairedMetricsTests(unittest.TestCase):
    def setUp(self):
        previous = torch.get_num_threads()
        torch.set_num_threads(3)
        self.addCleanup(torch.set_num_threads, previous)

    def test_native_macro_rmse_is_not_pooled_rmse(self):
        original, candidate = arrays(0.), arrays(0.)
        errors = np.arange(1, 13, dtype=np.float32)[None, :, None, None]
        original['prediction'] += errors
        candidate['prediction'] += errors*.5
        result = entry.paired_scores(original, candidate, material())
        row = result['rows']['ORIGINAL_IGSTGNN']
        self.assertAlmostEqual(row['author_metrics']['horizon_macro']['rmse'], 6.5, places=5)
        self.assertGreater(row['common_mask_pooled']['rmse'], 7.)
        self.assertAlmostEqual(result['relative_gain_percent_vs_replayed_original']['mae'], 50., places=4)

    def test_equal_shapes_cannot_hide_different_target_or_axes(self):
        for key in ('target', 'sample_ids', 'station_ids'):
            candidate = arrays(1.)
            if key == 'target':
                candidate[key][0, 0, 0, 0] += 1.
            else:
                candidate[key] = candidate[key][::-1].copy()
            with self.subTest(key=key), self.assertRaises(ValueError):
                entry.paired_scores(arrays(2.), candidate, material())

    def test_canonicalization_precedes_fp32_reductions(self):
        values = arrays()
        values['prediction'] = np.asfortranarray(values['prediction'])
        values['target'] = np.asfortranarray(values['target'])
        with self.assertRaisesRegex(ValueError, 'layout'):
            entry.validate_arrays(values, material())
        canonical = entry.canonical_arrays(values)
        entry.validate_arrays(canonical, material())
        self.assertTrue(all(value.flags.c_contiguous for value in canonical.values()))
        self.assertTrue(np.array_equal(canonical['prediction'], values['prediction']))

    def test_dtype_or_nonfinite_predictions_are_rejected(self):
        for value in (np.float64, 'nonfinite'):
            changed = arrays()
            if value == 'nonfinite':
                changed['prediction'][0, 0, 0, 0] = np.nan
            else:
                changed['prediction'] = changed['prediction'].astype(value)
            with self.subTest(value=value), self.assertRaises(ValueError):
                entry.validate_arrays(changed, material())

    def test_each_validation_replay_preserves_its_training_metric(self):
        row = dict(author_metrics=dict(pooled=dict(mae=2.)), common_mask_pooled=dict(mae=3.))
        self.assertTrue(entry.validation_replay('ORIGINAL_IGSTGNN', row, 2.)['passed'])
        self.assertTrue(entry.validation_replay('STATE_EXCHANGE', row, dict(mae=3.))['passed'])
        with self.assertRaises(ValueError):
            entry.validation_replay('STATE_EXCHANGE', row, dict(mae=2.))

    def test_report_keeps_native_main_and_common_supplement_distinct(self):
        result = dict(protocol=material(), checkpoint_roles=frozen()['payload']['checkpoint_roles'],
                      **entry.paired_scores(arrays(2.), arrays(1.), material()))
        with TemporaryDirectory() as temp:
            entry.render_report(temp, result)
            report = (Path(temp)/'report.md').read_text(encoding='utf-8')
            self.assertIn('12-lead RMSE', report)
            self.assertIn('common_mask_pooled is supplementary', report)
            self.assertIn('single-seed whole-method', report)
            rows = (Path(temp)/'per_horizon.csv').read_text(encoding='utf-8').splitlines()
            self.assertEqual(len(rows), 25)


class MultiDatasetCacheTests(unittest.TestCase):
    def setUp(self):
        previous = torch.get_num_threads()
        torch.set_num_threads(3)
        self.addCleanup(torch.set_num_threads, previous)

    def test_completed_cache_recomputes_metrics_without_model_forward(self):
        with TemporaryDirectory() as temp:
            path = Path(temp)/'model'
            saved, metadata = entry.write_cache(path, frozen(), 'STATE_EXCHANGE', arrays(), audit_metadata())
            with patch.object(entry, 'collect', side_effect=AssertionError('No forward allowed')):
                actual, replay = entry.read_cache(path, frozen(), 'STATE_EXCHANGE')
            self.assertTrue(np.array_equal(actual['prediction'], saved['prediction']))
            self.assertEqual(replay, metadata)

    def test_cache_overwrite_and_incomplete_cache_are_rejected(self):
        with TemporaryDirectory() as temp:
            path = Path(temp)/'model'
            path.mkdir()
            with self.assertRaisesRegex(ValueError, 'cannot be overwritten'):
                entry.write_cache(path, frozen(), 'STATE_EXCHANGE', arrays(), audit_metadata())
            with self.assertRaisesRegex(ValueError, 'Incomplete cache'):
                entry.read_cache(path, frozen(), 'STATE_EXCHANGE')

    def test_npz_checksum_tampering_is_rejected(self):
        with TemporaryDirectory() as temp:
            path = Path(temp)/'model'
            entry.write_cache(path, frozen(), 'STATE_EXCHANGE', arrays(), audit_metadata())
            with (path/'predictions.npz').open('ab') as stream:
                stream.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                entry.read_cache(path, frozen(), 'STATE_EXCHANGE')

    def test_metadata_cannot_change_metrics_or_model_weights(self):
        for kind in ('metrics', 'weights'):
            with self.subTest(kind=kind), TemporaryDirectory() as temp:
                path = Path(temp)/'model'
                entry.write_cache(path, frozen(), 'STATE_EXCHANGE', arrays(), audit_metadata())
                metadata = entry.read_json(path/'metadata.json')
                if kind == 'metrics':
                    metadata['scores']['author_metrics']['horizon_macro']['mae'] += 1.
                else:
                    metadata['audit']['model_tensor_sha256'] = 'changed'
                entry.write_json(path/'metadata.json', metadata)
                with self.assertRaises(ValueError):
                    entry.read_cache(path, frozen(), 'STATE_EXCHANGE')

    def test_cached_native_metrics_use_recorded_threads_and_restore_current(self):
        with TemporaryDirectory() as temp:
            path = Path(temp)/'model'
            entry.write_cache(path, frozen(), 'STATE_EXCHANGE', arrays(), audit_metadata())
            torch.set_num_threads(1)
            entry.read_cache(path, frozen(), 'STATE_EXCHANGE')
            self.assertEqual(torch.get_num_threads(), 1)

    def test_test_lock_is_immutable_and_precedes_forward(self):
        payload = frozen('test')['payload']
        with TemporaryDirectory() as temp:
            directory = Path(temp)
            record = entry.freeze_output(directory, payload)
            self.assertTrue((directory/'test_lock.json').is_file())
            self.assertEqual(entry.freeze_output(directory, payload), record)
            changed = copy.deepcopy(payload)
            changed['checkpoint_roles']['STATE_EXCHANGE']['model_tensor_sha256'] = 'new'
            with self.assertRaisesRegex(ValueError, 'identity changed'):
                entry.freeze_output(directory, changed)

    def test_residual_output_without_lock_is_never_adopted(self):
        with TemporaryDirectory() as temp:
            directory = Path(temp)
            (directory/'predictions.npz').write_bytes(b'residual')
            with self.assertRaisesRegex(ValueError, 'residual files'):
                entry.freeze_output(directory, frozen('test')['payload'])

    def test_freeze_checks_source_release_and_external_artifact(self):
        with TemporaryDirectory() as temp:
            path = Path(temp)/'checkpoint.pt'
            path.write_bytes(b'fixed checkpoint')
            record = frozen('test')
            record['payload']['source_files'] = {str(path.resolve()):entry.sha256(path)}
            record['freeze_id'] = entry.digest_json(record['payload'])
            with patch.object(entry, 'source_hashes', return_value={'model.py':'code'}), patch.object(
                    entry, 'spec', return_value={'hashes':{'published.npy':'data'}}):
                entry.verify_frozen(record)
                path.write_bytes(b'changed checkpoint')
                with self.assertRaisesRegex(ValueError, 'artifact changed'):
                    entry.verify_frozen(record)

    def test_freeze_payload_checksum_is_verified(self):
        record = frozen()
        record['payload']['protocol']['count'] += 1
        with self.assertRaisesRegex(ValueError, 'checksum'):
            entry.verify_frozen(record)

    def test_result_reader_authenticates_both_caches_and_scope_without_forwards(self):
        record = frozen('test')
        with TemporaryDirectory() as temp:
            directory = Path(temp)
            entry.write_json(directory/'test_lock.json', record)
            for case, offset in zip(entry.CASES, (2., 1.)):
                entry.write_cache(directory/case, record, case, arrays(offset), audit_metadata('test'))
            roles = record['payload']['checkpoint_roles']
            result = dict(schema='state_exchange_multidata_comparison_v1', status='PAIRED_EVALUATION_COMPLETE',
                freeze_id=record['freeze_id'], protocol=record['payload']['protocol'],
                checkpoint_roles=roles, metric_contract=entry.METRIC_CONTRACT,
                runtime=record['payload']['runtime'], dataset='Alameda', split='test', samples=4,
                engineering_only=False, test_evaluated=True,
                candidate_checkpoint_role=roles['STATE_EXCHANGE'], original_checkpoint_role=roles['ORIGINAL_IGSTGNN'],
                parameter_counts={case:100 for case in entry.CASES},
                **entry.paired_scores(arrays(2.), arrays(1.), record['payload']['protocol']))
            entry.write_json(directory/'comparison.json', result)
            with patch.object(entry, 'verify_frozen'), patch.object(
                    entry, 'collect', side_effect=AssertionError('No model forward allowed')):
                self.assertEqual(entry.read_result(directory), result)
                result['parameter_counts']['STATE_EXCHANGE'] = 99
                entry.write_json(directory/'comparison.json', result)
                with self.assertRaisesRegex(ValueError, 'scope'):
                    entry.read_result(directory)


if __name__ == '__main__':
    unittest.main()
