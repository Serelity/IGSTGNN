"""Server-side checks for the disposable full-batch training preflight."""
from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from experiments.state_exchange_multidata import batch_check as entry
from experiments.state_exchange_multidata.common import digest_json, read_json


def result(case, nodes=521):
    return dict(case=case, nodes=nodes, batch_size=48, optimizer_updates=2,
        warmup_updates=1, measured_updates=1,
        validation_forward_calls=0, test_forward_calls=0, persistent_model_checkpoint=False)


def record():
    identity = dict(dataset='Alameda', seed=2025, batch_size=48, data_hashes={'a':'hash'},
                    execution_source={'source.py':'hash'}, runtime={'device':'cpu'},
                    cases=list(entry.CASES), no_adaptive_batch_reduction=True)
    value = dict(schema=entry.SCHEMA, status='FULL_BATCH_TRAINING_CHECK_COMPLETE',
                 identity=identity, checks={case:result(case) for case in entry.CASES})
    value['record_sha256'] = digest_json(value)
    return value, identity


class BatchCheckTests(unittest.TestCase):
    def test_adam_moments_remain_resident_during_second_training_forward(self):
        torch.manual_seed(2025)
        model = torch.nn.Linear(2, 1)
        previous = model.weight.detach().clone()
        optimizer = torch.optim.Adam(model.parameters(), lr=.002)
        seen_state = []
        def loss_builder():
            seen_state.append(len(optimizer.state))
            return model(torch.ones(48, 2)).square().mean()
        report = entry.profile_step(model, optimizer, loss_builder, 'cpu')
        self.assertEqual(report['batch_size'], 48)
        self.assertEqual(report['optimizer_updates'], 2)
        self.assertEqual((report['warmup_updates'], report['measured_updates']), (1, 1))
        self.assertEqual(seen_state, [0, 2])
        self.assertEqual(report['optimizer_state_entries'], 2)
        self.assertFalse(torch.equal(previous, model.weight))
        self.assertFalse(report['persistent_model_checkpoint'])
        self.assertEqual(report['validation_forward_calls'], 0)
        self.assertEqual(report['test_forward_calls'], 0)

    def test_nonfinite_loss_stops_before_optimizer_update(self):
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=.002)
        with self.assertRaisesRegex(ValueError, 'finite'):
            entry.profile_step(model, optimizer,
                lambda: model(torch.ones(48, 1)).mean()*float('nan'), 'cpu')
        self.assertEqual(len(optimizer.state), 0)

    def test_cache_rejects_smaller_batch_even_if_record_resigned(self):
        saved, identity = record()
        entry.validate_cached(saved, identity)
        saved['checks']['ORIGINAL_IGSTGNN']['batch_size'] = 24
        saved.pop('record_sha256')
        saved['record_sha256'] = digest_json(saved)
        with self.assertRaisesRegex(ValueError, 'dimensions'):
            entry.validate_cached(saved, identity)

    def test_cache_authenticates_source_data_and_runtime_identity(self):
        saved, identity = record()
        for key, value in (('data_hashes', {'a':'new'}), ('execution_source', {'source.py':'new'}),
                           ('runtime', {'device':'different'})):
            with self.assertRaisesRegex(ValueError, 'runtime changed'):
                entry.validate_cached(saved, dict(identity, **{key:value}))
        saved['status'] = 'partial'
        with self.assertRaisesRegex(ValueError, 'record changed'):
            entry.validate_cached(saved, identity)

    def test_preflight_runs_methods_sequentially_then_reuses_certified_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(dataset='Alameda', data_dir=Path(tmp)/'data',
                output_dir=Path(tmp)/'batch_check', device='cpu', threads=1)
            calls = []
            def candidate(*args):
                calls.append('STATE_EXCHANGE')
                return result('STATE_EXCHANGE')
            def original(*args):
                calls.append('ORIGINAL_IGSTGNN')
                return result('ORIGINAL_IGSTGNN')
            with patch.object(entry, 'runtime', return_value={'device':'cpu'}), \
                    patch.object(entry, 'authenticate_release', return_value=({}, {'a':'hash'})), \
                    patch.object(entry, 'source_hashes', return_value={'source.py':'hash'}), \
                    patch.object(entry, 'candidate_check', side_effect=candidate), \
                    patch.object(entry, 'original_check', side_effect=original):
                first = entry.run(args)
                second = entry.run(args)
            self.assertEqual(calls, ['STATE_EXCHANGE', 'ORIGINAL_IGSTGNN'])
            self.assertEqual(first, second)
            self.assertEqual(read_json(args.output_dir/'metadata.json')['status'],
                             'FULL_BATCH_TRAINING_CHECK_COMPLETE')
            self.assertEqual(list(args.output_dir.glob('*.pt')), [])

    def test_oom_is_recorded_and_never_retried_with_smaller_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(dataset='Orange', data_dir=Path(tmp)/'data',
                output_dir=Path(tmp)/'batch_check', device='cpu', threads=1)
            with patch.object(entry, 'runtime', return_value={'device':'cpu'}), \
                    patch.object(entry, 'authenticate_release', return_value=({}, {'a':'hash'})), \
                    patch.object(entry, 'source_hashes', return_value={'source.py':'hash'}), \
                    patch.object(entry, 'candidate_check', side_effect=RuntimeError('CUDA out of memory')) as candidate, \
                    patch.object(entry, 'original_check') as original:
                with self.assertRaisesRegex(RuntimeError, 'out of memory'):
                    entry.run(args)
            candidate.assert_called_once()
            original.assert_not_called()
            self.assertFalse((args.output_dir/'metadata.json').exists())
            failures = list(args.output_dir.glob('failure_*.json'))
            self.assertEqual(len(failures), 1)
            self.assertEqual(read_json(failures[0])['batch_size'], 48)
            self.assertFalse(read_json(failures[0])['batch_size_changed'])


if __name__ == '__main__':
    unittest.main()
