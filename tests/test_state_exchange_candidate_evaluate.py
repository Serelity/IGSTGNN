"""Server-only candidate selection, native metrics and immutable cache tests."""
import copy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.state_exchange_multidata_candidate import evaluate as entry
from experiments.state_exchange_multidata import evaluate as previous


def release():
    return dict(train_samples=7, val_samples=4, test_samples=5, nodes=3,
                hashes={'published.npy':'data'})


def material(split='val'):
    start, count = (7, 4) if split == 'val' else (11, 5)
    return dict(dataset='Alameda', split=split, engineering_only=False,
        count=count, nodes=3, bounds_half_open=[start, start+count],
        station_ids=[101, 103, 107], release_hashes={'published.npy':'data'})


def arrays(offset=1., split='val'):
    scope = material(split)
    target = np.full((scope['count'], 12, 3, 1), 100., dtype=np.float32)
    return dict(prediction=target+np.float32(offset), target=target,
        sample_ids=np.arange(*scope['bounds_half_open'], dtype=np.int64),
        station_ids=np.array([101, 103, 107], dtype=np.int64))


def role():
    return dict(case='STATE_EXCHANGE', selection='best', epoch=99,
        completed_training_epochs=100, full_protocol_reached=True,
        model_tensor_sha256='weights', registered_parameters=100)


def frozen(split='val'):
    payload = dict(protocol=material(split), runtime=dict(threads=3),
        training_source={'old.py':'unchanged'}, evaluation_source={'candidate.py':'new'},
        source_files={}, metric_contract=entry.METRIC_CONTRACT,
        checkpoint_roles={'STATE_EXCHANGE':role()}, future_target_passed_to_forward=False,
        optimizer_updates=0, new_original_training_updates=0, new_original_inference_calls=0)
    return dict(schema=entry.LOCK_SCHEMA, freeze_id=entry.digest_json(payload), payload=payload)


def audit_metadata(split='val'):
    return dict(audit=dict(model_tensor_sha256='weights', rng_unchanged=True,
        registered_tensors_unchanged=True, raw_inputs_unchanged=True,
        future_target_passed_to_forward=False, optimizer_updates=0, forward_batches=2),
        validation_replay=dict(passed=True) if split == 'val' else dict(applicable=False))


def identity():
    return dict(schema=entry.SCHEMA, engineering_only=False, seed=2025,
        config=dict(max_epochs=100), data=dict(dataset='Alameda', hashes={'published.npy':'data'}),
        execution_source={'old.py':'unchanged'})


def result_record(record, metadata):
    roles = record['payload']['checkpoint_roles']
    scope = record['payload']['protocol']
    return dict(schema=entry.RESULT_SCHEMA, status=entry.STATUS,
        freeze_id=record['freeze_id'], protocol=scope, checkpoint_roles=roles,
        metric_contract=entry.METRIC_CONTRACT, runtime=record['payload']['runtime'],
        dataset=scope['dataset'], split=scope['split'], samples=scope['count'],
        engineering_only=scope['engineering_only'], test_evaluated=scope['split'] == 'test',
        candidate_checkpoint_role=roles['STATE_EXCHANGE'], parameter_counts={'STATE_EXCHANGE':100},
        caches={'STATE_EXCHANGE':dict(archive_sha256=metadata['archive_sha256'])},
        new_original_training_updates=0, new_original_inference_calls=0,
        rows={'STATE_EXCHANGE':metadata['scores']})


class CandidateOnlySelectionTests(unittest.TestCase):
    def guard(self, candidate=None, checkpoint=None, dataset='Alameda', split='test', engineering=False):
        with patch.object(entry, 'validate_identity'), patch.object(entry, 'spec', return_value=release()), \
                patch.object(entry, 'training_source_hashes', return_value={'old.py':'unchanged'}):
            entry.selection_guard(dataset, split, candidate or identity(), checkpoint or role(), engineering)

    def test_full_candidate_best_needs_no_original_identity_or_checkpoint(self):
        self.guard()
        self.assertEqual(entry.CASES, ('STATE_EXCHANGE',))

    def test_partial_training_last_or_other_arm_are_rejected(self):
        for changes in (dict(completed_training_epochs=99), dict(full_protocol_reached=False),
                        dict(selection='last'), dict(case='ORIGINAL_IGSTGNN')):
            changed = dict(role(), **changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.guard(checkpoint=changed)

    def test_changed_release_source_and_scope_are_rejected(self):
        for change in (lambda value:value['data'].update(dataset='Orange'),
                       lambda value:value['data']['hashes'].update(**{'published.npy':'changed'}),
                       lambda value:value['execution_source'].update(**{'old.py':'changed'}),
                       lambda value:value.update(engineering_only=True)):
            changed = identity()
            change(changed)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.guard(candidate=changed)

    def test_engineering_test_and_new_contra_inference_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Engineering test'):
            self.guard(engineering=True)
        with self.assertRaisesRegex(ValueError, 'Existing Contra'):
            self.guard(dataset='Contra_Costa')

    def test_cli_requires_candidate_run_but_has_no_original_run_option(self):
        args = entry.parse_args(['--dataset', 'Alameda', '--data-dir', 'data',
            '--run-dir', 'candidate', '--output-dir', 'output'])
        self.assertEqual(args.run_dir, Path('candidate'))
        self.assertFalse(hasattr(args, 'baseline_run'))
        with self.assertRaises(SystemExit):
            entry.parse_args(['--dataset', 'Alameda', '--data-dir', 'data', '--run-dir', 'candidate',
                '--output-dir', 'output', '--split', 'test', '--engineering'])

    def test_output_cannot_intersect_candidate_training_or_data(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            candidate, data = root/'candidate', root/'data'
            for output in (root, candidate, candidate/'eval', data/'eval'):
                with self.subTest(output=output), self.assertRaises(ValueError):
                    entry.safe_output(output, (candidate,), data)
            self.assertEqual(entry.safe_output(root/'eval', (candidate,), data), root/'eval')

    def test_protocol_preserves_exact_author_split_offsets(self):
        with patch.object(previous, 'spec', return_value=release()):
            self.assertEqual(entry.protocol('Alameda', 'val', False, [101, 103, 107]), material())
            self.assertEqual(entry.protocol('Alameda', 'test', False, [101, 103, 107]), material('test'))

    def test_selected_checkpoint_uses_immutable_best_not_last_model_tensors(self):
        with TemporaryDirectory() as temp:
            directory = Path(temp)/'STATE_EXCHANGE'
            directory.mkdir()
            last_path, best_path = directory/'last.pt', directory/'best_epoch_099_fixed.pt'
            last_path.write_bytes(b'last')
            best_path.write_bytes(b'best')
            weights = {'only':torch.tensor([1.])}
            last = dict(completed_epoch=100, best_checkpoint_file=best_path.name,
                best_checkpoint_sha256=entry.sha256(best_path), best_epoch=99,
                best_metrics=dict(mae=2.), global_updates=800, registered_parameters=1)
            best = dict(identity_sha256='identity', arm='STATE_EXCHANGE', epoch=99,
                validation=dict(mae=2.), stage='joint', model_state=weights)
            candidate = dict(identity_sha256='identity', config=dict(max_epochs=100))
            data = SimpleNamespace(loaders={'train_loader':SimpleNamespace(num_batch=8)})
            with patch.object(previous, 'load_state', side_effect=[last, best]), \
                    patch.object(previous, 'validate_saved'):
                selected, selected_role, expected, path = entry.selected_checkpoint(temp, candidate, data)
            self.assertEqual(path, best_path)
            self.assertIs(selected, weights)
            self.assertEqual(selected_role['epoch'], 99)
            self.assertEqual(selected_role['checkpoint_sha256'], entry.sha256(best_path))
            self.assertEqual(expected, best['validation'])


class CandidateOnlyMetricTests(unittest.TestCase):
    def setUp(self):
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(3)
        self.addCleanup(torch.set_num_threads, previous_threads)

    def test_native_macro_and_pooled_rmse_remain_distinct(self):
        values = arrays(0.)
        values['prediction'] += np.arange(1, 13, dtype=np.float32)[None, :, None, None]
        row = entry.scores(values)
        self.assertAlmostEqual(row['author_metrics']['horizon_macro']['rmse'], 6.5, places=5)
        self.assertGreater(row['common_mask_pooled']['rmse'], 7.)

    def test_canonical_layout_is_applied_before_native_metric_reduction(self):
        values = arrays()
        for key in ('prediction', 'target'):
            values[key] = np.asfortranarray(values[key])
        with self.assertRaisesRegex(ValueError, 'layout'):
            entry.validate_arrays(values, material())
        canonical = entry.canonical_arrays(values)
        entry.validate_arrays(canonical, material())
        self.assertTrue(all(value.flags.c_contiguous for value in canonical.values()))

    def test_wrong_axes_nonfinite_or_float64_predictions_are_rejected(self):
        for kind in ('axis', 'nonfinite', 'dtype'):
            values = arrays()
            if kind == 'axis':
                values['sample_ids'] = values['sample_ids'][::-1].copy()
            elif kind == 'nonfinite':
                values['prediction'][0, 0, 0, 0] = np.nan
            else:
                values['prediction'] = values['prediction'].astype(np.float64)
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                entry.validate_arrays(values, material())

    def test_candidate_validation_replay_uses_training_pooled_mae(self):
        row = dict(author_metrics=dict(pooled=dict(mae=2.)), common_mask_pooled=dict(mae=3.))
        self.assertTrue(entry.validation_replay('STATE_EXCHANGE', row, dict(mae=3.))['passed'])
        with self.assertRaises(ValueError):
            entry.validation_replay('STATE_EXCHANGE', row, dict(mae=2.))

    def test_report_contains_only_candidate_rows_and_no_original_gain(self):
        record = frozen()
        with TemporaryDirectory() as temp:
            _, metadata = entry.write_cache(Path(temp)/'STATE_EXCHANGE', record,
                'STATE_EXCHANGE', arrays(), audit_metadata())
            result = result_record(record, metadata)
            entry.render_report(temp, result)
            report = (Path(temp)/'report.md').read_text(encoding='utf-8')
            self.assertIn('12-lead RMSE', report)
            self.assertIn('common_mask_pooled is supplementary', report)
            self.assertNotIn('| ORIGINAL_IGSTGNN |', report)
            self.assertNotIn('relative_gain_percent_vs_replayed_original', result)
            self.assertEqual(len((Path(temp)/'per_horizon.csv').read_text(encoding='utf-8').splitlines()), 13)


class CandidateOnlyCacheTests(unittest.TestCase):
    def setUp(self):
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(3)
        self.addCleanup(torch.set_num_threads, previous_threads)

    def test_candidate_cache_reuses_predictions_with_zero_model_forward(self):
        with TemporaryDirectory() as temp:
            path = Path(temp)/'STATE_EXCHANGE'
            saved, metadata = entry.write_cache(path, frozen(), 'STATE_EXCHANGE', arrays(), audit_metadata())
            with patch.object(entry, 'collect', side_effect=AssertionError('No forward')):
                actual, observed = entry.read_cache(path, frozen())
            self.assertTrue(np.array_equal(saved['prediction'], actual['prediction']))
            self.assertEqual(observed, metadata)

    def test_original_cache_case_is_rejected(self):
        with TemporaryDirectory() as temp, self.assertRaisesRegex(ValueError, 'Only STATE_EXCHANGE'):
            entry.write_cache(Path(temp)/'original', frozen(), 'ORIGINAL_IGSTGNN', arrays(), audit_metadata())

    def test_completed_or_incomplete_cache_is_never_overwritten(self):
        with TemporaryDirectory() as temp:
            path = Path(temp)/'STATE_EXCHANGE'
            path.mkdir()
            with self.assertRaisesRegex(ValueError, 'cannot be overwritten'):
                entry.write_cache(path, frozen(), 'STATE_EXCHANGE', arrays(), audit_metadata())
            with self.assertRaisesRegex(ValueError, 'Incomplete cache'):
                entry.read_cache(path, frozen())

    def test_npz_checksum_tampering_is_rejected(self):
        with TemporaryDirectory() as temp:
            path = Path(temp)/'STATE_EXCHANGE'
            entry.write_cache(path, frozen(), 'STATE_EXCHANGE', arrays(), audit_metadata())
            with (path/'predictions.npz').open('ab') as stream:
                stream.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                entry.read_cache(path, frozen())

    def test_metadata_cannot_change_metrics_weights_or_replay(self):
        for kind in ('metrics', 'weights', 'replay'):
            with self.subTest(kind=kind), TemporaryDirectory() as temp:
                path = Path(temp)/'STATE_EXCHANGE'
                entry.write_cache(path, frozen(), 'STATE_EXCHANGE', arrays(), audit_metadata())
                metadata = entry.read_json(path/'metadata.json')
                if kind == 'metrics':
                    metadata['scores']['author_metrics']['horizon_macro']['mae'] += 1.
                elif kind == 'weights':
                    metadata['audit']['model_tensor_sha256'] = 'changed'
                else:
                    metadata['validation_replay']['passed'] = False
                entry.write_json(path/'metadata.json', metadata)
                with self.assertRaises(ValueError):
                    entry.read_cache(path, frozen())

    def test_recorded_threads_are_restored_after_cache_read(self):
        with TemporaryDirectory() as temp:
            path = Path(temp)/'STATE_EXCHANGE'
            entry.write_cache(path, frozen(), 'STATE_EXCHANGE', arrays(), audit_metadata())
            torch.set_num_threads(1)
            entry.read_cache(path, frozen())
            self.assertEqual(torch.get_num_threads(), 1)

    def test_test_lock_is_committed_before_forward_and_cannot_be_reselected(self):
        with TemporaryDirectory() as temp:
            record = entry.freeze_output(temp, frozen('test')['payload'])
            self.assertTrue((Path(temp)/'test_lock.json').is_file())
            self.assertEqual(entry.freeze_output(temp, record['payload']), record)
            changed = copy.deepcopy(record['payload'])
            changed['checkpoint_roles']['STATE_EXCHANGE']['epoch'] = 98
            with self.assertRaisesRegex(ValueError, 'identity changed'):
                entry.freeze_output(temp, changed)

    def test_orphan_output_without_lock_is_preserved(self):
        with TemporaryDirectory() as temp:
            (Path(temp)/'residual').write_bytes(b'keep')
            with self.assertRaisesRegex(ValueError, 'residual files'):
                entry.freeze_output(temp, frozen()['payload'])

    def test_freeze_checks_old_training_and_new_evaluation_source_separately(self):
        for kind in ('training', 'evaluation'):
            record = frozen()
            with patch.object(entry, 'training_source_hashes', return_value={'old.py':'changed' if kind == 'training' else 'unchanged'}), \
                    patch.object(entry, 'new_source_hashes', return_value={'candidate.py':'changed' if kind == 'evaluation' else 'new'}):
                with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, 'source changed'):
                    entry.verify_frozen(record)

    def test_freeze_checks_checkpoint_and_release_snapshot_without_forward(self):
        with TemporaryDirectory() as temp:
            path = Path(temp)/'best.pt'
            path.write_bytes(b'fixed BEST')
            record = frozen()
            record['payload']['source_files'] = {str(path.resolve()):entry.sha256(path)}
            record['freeze_id'] = entry.digest_json(record['payload'])
            with patch.object(entry, 'training_source_hashes', return_value={'old.py':'unchanged'}), \
                    patch.object(entry, 'new_source_hashes', return_value={'candidate.py':'new'}), \
                    patch.object(previous, 'spec', return_value=release()):
                entry.verify_frozen(record)
                path.write_bytes(b'changed BEST')
                with self.assertRaisesRegex(ValueError, 'artifact changed'):
                    entry.verify_frozen(record)

    def test_added_original_role_or_nonbest_freeze_is_rejected(self):
        for kind in ('original', 'last'):
            record = frozen()
            if kind == 'original':
                record['payload']['checkpoint_roles']['ORIGINAL_IGSTGNN'] = role()
            else:
                record['payload']['checkpoint_roles']['STATE_EXCHANGE']['selection'] = 'last'
            record['freeze_id'] = entry.digest_json(record['payload'])
            with patch.object(entry, 'training_source_hashes', return_value={'old.py':'unchanged'}), \
                    patch.object(entry, 'new_source_hashes', return_value={'candidate.py':'new'}), \
                    patch.object(previous, 'spec', return_value=release()):
                with self.subTest(kind=kind), self.assertRaises(ValueError):
                    entry.verify_frozen(record)

    def test_result_reader_authenticates_one_cache_and_scope_without_forward(self):
        record = frozen('test')
        with TemporaryDirectory() as temp:
            directory = Path(temp)
            entry.write_json(directory/'test_lock.json', record)
            _, metadata = entry.write_cache(directory/'STATE_EXCHANGE', record,
                'STATE_EXCHANGE', arrays(split='test'), audit_metadata('test'))
            result = result_record(record, metadata)
            entry.write_json(directory/'result.json', result)
            with patch.object(entry, 'verify_frozen'), patch.object(
                    entry, 'collect', side_effect=AssertionError('No model forward allowed')):
                self.assertEqual(entry.read_result(directory), result)
                result['rows']['ORIGINAL_IGSTGNN'] = copy.deepcopy(result['rows']['STATE_EXCHANGE'])
                entry.write_json(directory/'result.json', result)
                with self.assertRaisesRegex(ValueError, 'scope'):
                    entry.read_result(directory)


if __name__ == '__main__':
    unittest.main()
