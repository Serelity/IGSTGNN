"""Server-only original BEST selection, native replay and immutable-cache checks."""
import copy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from experiments.original_multidata_only import evaluate as entry
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
    return dict(case='ORIGINAL_IGSTGNN', dataset='Alameda', selection='best', best_epoch=99,
        selected_epoch=99, completed_epoch=100, wait=1, training_complete=True,
        training_budget_epochs=100, engineering_only=False, best_selection_includes_stage_zero=False,
        model_tensor_sha256='weights', model_state_sha256='weights', registered_parameters=100,
        checkpoint_sha256='checkpoint', expected_validation_mae=1.)


def identity():
    return dict(schema=entry.BASELINE_SCHEMA, engineering_only=False, seed=2025,
        data=dict(dataset='Alameda', hashes={'published.npy':'data'}), execution_source={'old.py':'unchanged'})


def frozen(split='val'):
    payload = dict(protocol=material(split), runtime=dict(threads=3),
        training_source={'old.py':'unchanged'}, evaluation_source={'original.py':'new'},
        source_files={}, metric_contract=entry.METRIC_CONTRACT,
        checkpoint_roles={'ORIGINAL_IGSTGNN':role()}, future_target_passed_to_forward=False,
        optimizer_updates=0, candidate_inference_calls=0,
        validation_replay_reference=dict(passed=True, checkpoint_sha256='checkpoint') if split == 'test' else None)
    return dict(schema=entry.LOCK_SCHEMA, freeze_id=entry.digest_json(payload), payload=payload)


def audit_metadata(split='val'):
    return dict(audit=dict(model_tensor_sha256='weights', rng_unchanged=True,
        registered_tensors_unchanged=True, raw_inputs_unchanged=True,
        future_target_passed_to_forward=False, optimizer_updates=0, forward_batches=2),
        validation_replay=entry.validation_replay('ORIGINAL_IGSTGNN', entry.scores(arrays()), 1.)
            if split == 'val' else dict(applicable=False))


def result_record(record, metadata):
    roles, scope = record['payload']['checkpoint_roles'], record['payload']['protocol']
    return dict(schema=entry.RESULT_SCHEMA, status=entry.STATUS,
        freeze_id=record['freeze_id'], protocol=scope, checkpoint_roles=roles,
        metric_contract=entry.METRIC_CONTRACT, runtime=record['payload']['runtime'],
        dataset=scope['dataset'], split=scope['split'], samples=scope['count'],
        engineering_only=scope['engineering_only'], test_evaluated=scope['split'] == 'test',
        original_checkpoint_role=roles['ORIGINAL_IGSTGNN'], parameter_counts={'ORIGINAL_IGSTGNN':100},
        caches={'ORIGINAL_IGSTGNN':dict(archive_sha256=metadata['archive_sha256'])},
        candidate_inference_calls=0, rows={'ORIGINAL_IGSTGNN':metadata['scores']})


class OriginalOnlySelectionTests(unittest.TestCase):
    def guard(self, selected=None, run=None, split='test', dataset='Alameda', engineering=False):
        with patch.object(entry, 'spec', return_value=release()), patch.object(
                entry, 'training_source_hashes', return_value={'old.py':'unchanged'}):
            entry.selection_guard(dataset, split, run or identity(), selected or role(), engineering)

    def test_complete_original_best_requires_no_candidate(self):
        self.guard()
        self.assertEqual(entry.CASES, ('ORIGINAL_IGSTGNN',))

    def test_original_native_patience_twenty_is_valid_completion(self):
        self.guard(dict(role(), completed_epoch=70, best_epoch=50, selected_epoch=50, wait=20))

    def test_incomplete_nonbest_stagezero_and_candidate_roles_are_rejected(self):
        for update in (dict(training_complete=False), dict(selection='last'),
                       dict(completed_epoch=70, selected_epoch=50, best_epoch=50, wait=19),
                       dict(best_selection_includes_stage_zero=True), dict(case='STATE_EXCHANGE')):
            with self.subTest(update=update), self.assertRaises(ValueError):
                self.guard(dict(role(), **update))

    def test_changed_release_source_and_scope_are_rejected(self):
        for mutate in (lambda value:value['data'].update(dataset='Orange'),
                       lambda value:value['data']['hashes'].update(**{'published.npy':'changed'}),
                       lambda value:value['execution_source'].update(**{'old.py':'changed'}),
                       lambda value:value.update(engineering_only=True)):
            run = identity()
            mutate(run)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                self.guard(run=run)

    def test_engineering_three_epoch_val_is_accepted_but_test_forbidden(self):
        run = dict(identity(), engineering_only=True)
        selected = dict(role(), engineering_only=True, training_budget_epochs=3,
                        completed_epoch=3, selected_epoch=2, best_epoch=2)
        self.guard(selected, run, split='val', engineering=True)
        with self.assertRaisesRegex(ValueError, 'Engineering test'):
            self.guard(selected, run, engineering=True)

    def test_cli_test_requires_validation_replay_and_has_no_candidate_run(self):
        common = ['--dataset', 'Alameda', '--data-dir', 'data', '--run-dir', 'original', '--output-dir', 'output']
        self.assertFalse(hasattr(entry.parse_args(common), 'candidate_run'))
        with self.assertRaises(SystemExit):
            entry.parse_args(common+['--split', 'test'])
        args = entry.parse_args(common+['--split', 'test', '--validation-dir', 'original_val'])
        self.assertEqual(args.validation_dir, Path('original_val'))
        with self.assertRaises(SystemExit):
            entry.parse_args(common+['--split', 'test', '--validation-dir', 'original_val', '--engineering'])

    def test_validation_reference_rejects_other_checkpoint_dataset_or_runtime(self):
        result = dict(dataset='Alameda', split='val', engineering_only=False,
                      original_checkpoint_role=role(), runtime=dict(threads=3))
        for update in (dict(split='test'), dict(dataset='Orange'), dict(engineering_only=True),
                       dict(original_checkpoint_role=dict(role(), checkpoint_sha256='changed')),
                       dict(runtime=dict(threads=1))):
            with self.subTest(update=update), patch.object(entry, 'read_result', return_value=dict(result, **update)), \
                    self.assertRaisesRegex(ValueError, 'different BEST or runtime'):
                entry.validation_reference('original_val', 'Alameda', role(), dict(threads=3))

    def test_validation_reference_binds_all_replay_files_without_new_forward(self):
        result = dict(dataset='Alameda', split='val', engineering_only=False,
                      original_checkpoint_role=role(), runtime=dict(threads=3))
        with patch.object(entry, 'read_result', return_value=result), patch.object(entry, 'sha256', return_value='fixed'):
            reference, files = entry.validation_reference('original_val', 'Alameda', role(), dict(threads=3))
        self.assertTrue(reference['passed'])
        self.assertEqual(reference['checkpoint_sha256'], 'checkpoint')
        self.assertEqual({path.name for path in files}, {'result.json', 'validation_lock.json', 'metadata.json', 'predictions.npz'})
        with self.assertRaisesRegex(ValueError, 'requires --validation-dir'):
            entry.validation_reference(None, 'Alameda', role(), dict(threads=3))


class OriginalOnlyMetricCacheTests(unittest.TestCase):
    def setUp(self):
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(3)
        self.addCleanup(torch.set_num_threads, previous_threads)

    def test_native_macro_rmse_is_not_pooled_rmse(self):
        values = arrays(0.)
        values['prediction'] += np.arange(1, 13, dtype=np.float32)[None, :, None, None]
        row = entry.scores(values)
        self.assertAlmostEqual(row['author_metrics']['horizon_macro']['rmse'], 6.5, places=5)
        self.assertGreater(row['common_mask_pooled']['rmse'], 7.)

    def test_original_validation_replay_uses_native_pooled_mae(self):
        row = dict(author_metrics=dict(pooled=dict(mae=2.)), common_mask_pooled=dict(mae=3.))
        self.assertTrue(entry.validation_replay('ORIGINAL_IGSTGNN', row, 2.)['passed'])
        with self.assertRaises(ValueError):
            entry.validation_replay('ORIGINAL_IGSTGNN', row, 3.)

    def test_canonical_layout_is_required_before_native_reduction(self):
        values = arrays()
        values['prediction'] = np.asfortranarray(values['prediction'])
        with self.assertRaisesRegex(ValueError, 'layout'):
            entry.validate_arrays(values, material())
        entry.validate_arrays(entry.canonical_arrays(values), material())

    def test_cache_replays_without_forward_and_restores_threads(self):
        with TemporaryDirectory() as temp:
            path = Path(temp)/'ORIGINAL_IGSTGNN'
            saved, metadata = entry.write_cache(path, frozen(), 'ORIGINAL_IGSTGNN', arrays(), audit_metadata())
            torch.set_num_threads(1)
            with patch.object(entry, 'collect', side_effect=AssertionError('No forward')):
                actual, observed = entry.read_cache(path, frozen())
            self.assertTrue(np.array_equal(saved['prediction'], actual['prediction']))
            self.assertEqual(observed, metadata)
            self.assertEqual(torch.get_num_threads(), 1)

    def test_candidate_cache_case_and_existing_cache_are_rejected(self):
        with TemporaryDirectory() as temp:
            path = Path(temp)/'original'
            with self.assertRaisesRegex(ValueError, 'Only ORIGINAL_IGSTGNN'):
                entry.write_cache(path, frozen(), 'STATE_EXCHANGE', arrays(), audit_metadata())
            path.mkdir()
            with self.assertRaisesRegex(ValueError, 'cannot be overwritten'):
                entry.write_cache(path, frozen(), 'ORIGINAL_IGSTGNN', arrays(), audit_metadata())
            with self.assertRaisesRegex(ValueError, 'Incomplete cache'):
                entry.read_cache(path, frozen())

    def test_archive_checksum_and_native_replay_tampering_are_rejected(self):
        for kind in ('archive', 'metrics', 'weights', 'replay'):
            with self.subTest(kind=kind), TemporaryDirectory() as temp:
                path = Path(temp)/'ORIGINAL_IGSTGNN'
                entry.write_cache(path, frozen(), 'ORIGINAL_IGSTGNN', arrays(), audit_metadata())
                if kind == 'archive':
                    with (path/'predictions.npz').open('ab') as stream:
                        stream.write(b'changed')
                else:
                    metadata = entry.read_json(path/'metadata.json')
                    if kind == 'metrics':
                        metadata['scores']['author_metrics']['horizon_macro']['mae'] += 1.
                    elif kind == 'weights':
                        metadata['audit']['model_tensor_sha256'] = 'changed'
                    else:
                        metadata['validation_replay']['expected_mae'] = 2.
                    entry.write_json(path/'metadata.json', metadata)
                with self.assertRaises(ValueError):
                    entry.read_cache(path, frozen())

    def test_test_lock_cannot_reselect_best_or_lose_validation_replay(self):
        with TemporaryDirectory() as temp:
            record = entry.freeze_output(temp, frozen('test')['payload'])
            self.assertTrue((Path(temp)/'test_lock.json').is_file())
            self.assertEqual(entry.freeze_output(temp, record['payload']), record)
            changed = copy.deepcopy(record['payload'])
            changed['checkpoint_roles']['ORIGINAL_IGSTGNN']['best_epoch'] = 98
            with self.assertRaisesRegex(ValueError, 'identity changed'):
                entry.freeze_output(temp, changed)

    def test_orphan_output_is_preserved(self):
        with TemporaryDirectory() as temp:
            (Path(temp)/'residual').write_bytes(b'keep')
            with self.assertRaisesRegex(ValueError, 'residual files'):
                entry.freeze_output(temp, frozen()['payload'])

    def test_frozen_role_rejects_candidate_or_nonbest_and_failed_replay(self):
        for kind in ('candidate', 'last', 'replay', 'checkpoint'):
            record = frozen('test')
            if kind == 'candidate':
                record['payload']['checkpoint_roles']['STATE_EXCHANGE'] = role()
            elif kind == 'last':
                record['payload']['checkpoint_roles']['ORIGINAL_IGSTGNN']['selection'] = 'last'
            elif kind == 'replay':
                record['payload']['validation_replay_reference']['passed'] = False
            else:
                record['payload']['validation_replay_reference']['checkpoint_sha256'] = 'changed'
            record['freeze_id'] = entry.digest_json(record['payload'])
            with self.subTest(kind=kind), patch.object(entry, 'training_source_hashes', return_value={'old.py':'unchanged'}), \
                    patch.object(entry, 'new_source_hashes', return_value={'original.py':'new'}), \
                    patch.object(previous, 'spec', return_value=release()), self.assertRaises(ValueError):
                entry.verify_frozen(record)

    def test_frozen_artifact_tampering_is_rejected_without_forward(self):
        with TemporaryDirectory() as temp:
            path = Path(temp)/'best.pt'
            path.write_bytes(b'fixed BEST')
            record = frozen()
            record['payload']['source_files'] = {str(path.resolve()):entry.sha256(path)}
            record['freeze_id'] = entry.digest_json(record['payload'])
            with patch.object(entry, 'training_source_hashes', return_value={'old.py':'unchanged'}), \
                    patch.object(entry, 'new_source_hashes', return_value={'original.py':'new'}), \
                    patch.object(previous, 'spec', return_value=release()):
                entry.verify_frozen(record)
                path.write_bytes(b'changed BEST')
                with self.assertRaisesRegex(ValueError, 'artifact changed'):
                    entry.verify_frozen(record)

    def test_report_and_result_have_only_original_and_twelve_leads(self):
        with TemporaryDirectory() as temp:
            directory = Path(temp)
            record = frozen()
            entry.write_json(directory/'validation_lock.json', record)
            _, metadata = entry.write_cache(directory/'ORIGINAL_IGSTGNN', record,
                'ORIGINAL_IGSTGNN', arrays(), audit_metadata())
            result = result_record(record, metadata)
            entry.render_report(directory, result)
            with patch.object(entry, 'verify_frozen'), patch.object(entry, 'collect', side_effect=AssertionError('No forward')):
                self.assertEqual(entry.read_result(directory), result)
                result['rows']['STATE_EXCHANGE'] = copy.deepcopy(result['rows']['ORIGINAL_IGSTGNN'])
                entry.write_json(directory/'result.json', result)
                with self.assertRaisesRegex(ValueError, 'scope'):
                    entry.read_result(directory)
            report = (directory/'report.md').read_text(encoding='utf-8')
            self.assertIn('| ORIGINAL_IGSTGNN |', report)
            self.assertNotIn('| STATE_EXCHANGE |', report)
            self.assertEqual(len((directory/'per_horizon.csv').read_text(encoding='utf-8').splitlines()), 13)

    def test_main_commits_test_lock_after_validation_replay_and_before_test_loader(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            run, data_dir, val_dir, output = (root/name for name in ('original', 'data', 'validation', 'test'))
            for directory in (run, data_dir, val_dir):
                directory.mkdir()
            for path in (run/'identity.json', run/'last.pt', run/'best.pt', data_dir/'published.npy'):
                path.write_bytes(b'fixed')
            environment = dict(threads=3)
            entry.write_json(run/'runtime.json', environment)
            training = dict(identity(), seed=2025, config=dict(bs=48), identity_sha256='identity')
            weights = {'weights':torch.tensor([1., 2.])}
            digest = entry.tensor_tree_hash(weights)
            checkpoint_role = dict(role(), model_state_sha256=digest, model_tensor_sha256=digest,
                                   registered_parameters=2, checkpoint_sha256=entry.sha256(run/'best.pt'))
            model = SimpleNamespace(load_state_dict=Mock(), state_dict=lambda:weights,
                                    parameters=lambda:iter(weights.values()))
            data = SimpleNamespace(station_ids=np.asarray([101, 103, 107], dtype=np.int64))
            phases = []
            def passed_replay(*unused):
                self.assertFalse((output/'test_lock.json').exists())
                phases.append('native_val_replay')
                return dict(passed=True, checkpoint_sha256=checkpoint_role['checkpoint_sha256']), []
            def locked_loader(*unused):
                self.assertTrue((output/'test_lock.json').is_file())
                self.assertEqual(phases, ['native_val_replay'])
                phases.append('test_loader')
                return SimpleNamespace(num_batch=2)
            observed_audit = dict(audit_metadata('test')['audit'], model_tensor_sha256=digest)
            with patch.object(entry, '_read_identity', return_value=training), \
                    patch.object(entry, 'configure_runtime', return_value=environment), \
                    patch.object(entry, 'load_original_data', return_value=data), \
                    patch.object(entry, 'selected_checkpoint', return_value=(weights, checkpoint_role, 1., run/'best.pt')), \
                    patch.object(entry, 'validation_reference', side_effect=passed_replay), \
                    patch.object(entry, 'build_original_model', return_value=model), \
                    patch.object(entry, 'original_loader', side_effect=locked_loader), \
                    patch.object(entry, 'collect', return_value=(arrays(split='test'), observed_audit)), \
                    patch.object(entry, 'training_source_hashes', return_value={'old.py':'unchanged'}), \
                    patch.object(entry, 'new_source_hashes', return_value={'original.py':'new'}), \
                    patch.object(entry, 'spec', return_value=release()), \
                    patch.object(previous, 'spec', return_value=release()):
                result = entry.main(['--dataset', 'Alameda', '--data-dir', str(data_dir),
                    '--run-dir', str(run), '--output-dir', str(output), '--split', 'test',
                    '--validation-dir', str(val_dir), '--device', 'cpu'])
            self.assertEqual(phases, ['native_val_replay', 'test_loader'])
            self.assertTrue(result['test_evaluated'])
            self.assertEqual(set(result['rows']), {'ORIGINAL_IGSTGNN'})


if __name__ == '__main__':
    unittest.main()
