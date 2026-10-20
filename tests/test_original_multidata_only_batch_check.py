"""Server-side original-only memory check scope and cache validation."""
from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.state_exchange_multidata import batch_check as paired
from experiments.state_exchange_multidata.common import digest_json, read_json
from experiments.original_multidata_only import batch_check as entry


def result():
    return dict(case='ORIGINAL_IGSTGNN', nodes=521, batch_size=48, optimizer_updates=2,
        warmup_updates=1, measured_updates=1, validation_forward_calls=0,
        test_forward_calls=0, persistent_model_checkpoint=False)


def record():
    identity = dict(dataset='Alameda', seed=2025, batch_size=48, data_hashes={'a':'hash'},
        execution_source={'source.py':'hash'}, runtime={'device':'cpu'},
        cases=list(entry.CASES), no_adaptive_batch_reduction=True,
        no_candidate_model_forwards=True)
    value = dict(schema=entry.SCHEMA, status=entry.STATUS, identity=identity,
        checks={'ORIGINAL_IGSTGNN':result()}, test_evaluated=False,
        formal_training_updates=0, candidate_model_forward_calls=0, batch_size_changed=False)
    resign(value)
    return value, identity


def resign(value):
    value.pop('record_sha256', None)
    value['record_sha256'] = digest_json(value)


class OriginalBatchCheckTests(unittest.TestCase):
    def test_reuses_exact_original_operation_and_never_constructs_candidate(self):
        self.assertIs(entry.original_check, paired.original_check)
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(dataset='Alameda', data_dir=Path(tmp)/'data',
                output_dir=Path(tmp)/'check', device='cpu', threads=1)
            with patch.object(entry, 'runtime', return_value={'device':'cpu'}), \
                    patch.object(entry, 'authenticate_release', return_value=({}, {'a':'hash'})), \
                    patch.object(entry, 'source_hashes', return_value={'source.py':'hash'}), \
                    patch.object(entry, '_release_device'), \
                    patch.object(entry, 'original_check', return_value=result()) as original, \
                    patch.object(paired, 'candidate_check', side_effect=AssertionError('candidate called')) as candidate, \
                    patch.object(paired, 'build_model', side_effect=AssertionError('candidate built')) as build:
                first, second = entry.run(args), entry.run(args)
            original.assert_called_once()
            candidate.assert_not_called()
            build.assert_not_called()
            self.assertEqual(first, second)
            self.assertEqual(set(first['checks']), {'ORIGINAL_IGSTGNN'})
            self.assertEqual(list(args.output_dir.glob('*.pt')), [])

    def test_cache_rejects_resigned_extra_candidate_or_smaller_batch(self):
        saved, identity = record()
        entry.validate_cached(saved, identity)
        extra = deepcopy(saved)
        extra['checks']['STATE_EXCHANGE'] = dict(result(), case='STATE_EXCHANGE')
        resign(extra)
        with self.assertRaisesRegex(ValueError, 'runtime changed'):
            entry.validate_cached(extra, identity)
        saved['checks']['ORIGINAL_IGSTGNN']['batch_size'] = 24
        resign(saved)
        with self.assertRaisesRegex(ValueError, 'dimensions'):
            entry.validate_cached(saved, identity)

    def test_cache_authenticates_source_data_and_runtime(self):
        saved, identity = record()
        for key, value in (('data_hashes', {'a':'new'}),
                ('execution_source', {'source.py':'new'}), ('runtime', {'device':'different'})):
            with self.assertRaisesRegex(ValueError, 'runtime changed'):
                entry.validate_cached(saved, dict(identity, **{key:value}))
        saved['status'] = 'partial'
        with self.assertRaisesRegex(ValueError, 'record changed'):
            entry.validate_cached(saved, identity)

    def test_cache_rejects_resigned_validation_test_or_formal_training(self):
        saved, identity = record()
        for key in ('validation_forward_calls', 'test_forward_calls'):
            changed = deepcopy(saved)
            changed['checks']['ORIGINAL_IGSTGNN'][key] = 1
            resign(changed)
            with self.assertRaisesRegex(ValueError, 'scope changed'):
                entry.validate_cached(changed, identity)
        for key, value in (('formal_training_updates', 1),
                ('candidate_model_forward_calls', 1), ('test_evaluated', True)):
            changed = dict(saved, **{key:value})
            resign(changed)
            with self.assertRaisesRegex(ValueError, 'scope changed'):
                entry.validate_cached(changed, identity)

    def test_oom_records_failure_without_candidate_or_smaller_batch_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(dataset='Alameda', data_dir=Path(tmp)/'data',
                output_dir=Path(tmp)/'check', device='cpu', threads=1)
            with patch.object(entry, 'runtime', return_value={'device':'cpu'}), \
                    patch.object(entry, 'authenticate_release', return_value=({}, {'a':'hash'})), \
                    patch.object(entry, 'source_hashes', return_value={'source.py':'hash'}), \
                    patch.object(entry, '_release_device'), \
                    patch.object(entry, 'original_check', side_effect=RuntimeError('CUDA out of memory')) as original, \
                    patch.object(paired, 'candidate_check') as candidate:
                with self.assertRaisesRegex(RuntimeError, 'out of memory'):
                    entry.run(args)
            original.assert_called_once()
            candidate.assert_not_called()
            self.assertFalse((args.output_dir/'metadata.json').exists())
            failures = list(args.output_dir.glob('failure_*.json'))
            self.assertEqual(len(failures), 1)
            failure = read_json(failures[0])
            self.assertEqual(failure['batch_size'], 48)
            self.assertFalse(failure['batch_size_changed'])
            self.assertEqual(failure['candidate_model_forward_calls'], 0)
            self.assertEqual(failure['formal_training_updates'], 0)

    def test_cli_restricts_dataset_and_positive_threads(self):
        args = entry.parse_args(['--dataset', 'Orange', '--data-dir', 'data', '--output-dir', 'check'])
        self.assertEqual((args.dataset, args.device, args.threads), ('Orange', 'cuda:0', 3))
        with self.assertRaises(SystemExit):
            entry.parse_args(['--dataset', 'Contra_Costa', '--data-dir', 'data', '--output-dir', 'check'])
        with self.assertRaisesRegex(ValueError, 'Threads'):
            entry.parse_args(['--dataset', 'Orange', '--data-dir', 'data', '--output-dir', 'check', '--threads', '0'])


if __name__ == '__main__':
    unittest.main()
