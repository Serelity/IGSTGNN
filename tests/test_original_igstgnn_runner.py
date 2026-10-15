"""Replay gates, paired screening, and exact baseline epoch-boundary resume."""
from contextlib import redirect_stdout
import io
import hashlib
import json
import logging
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.chronological import evaluate_original_igstgnn_modules as evaluate
from experiments.chronological import train_original_igstgnn_baseline as train
from experiments.chronological.original_igstgnn_source import get_original_modules
from experiments.chronological.original_igstgnn_runtime import CONFIG, sha256, tensor_tree_hash


class FakeEngine:
    def __init__(self):
        self._device = 'cpu'
        self.model = torch.nn.Linear(1, 1)
        self._optimizer = torch.optim.Adam(self.model.parameters(), lr=.002)
        self._lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self._optimizer, [1, 3], gamma=.5)
        self._dataloader = dict(train_loader=SimpleNamespace(samples=[0, 1, 2, 3], num_batch=2),
                                val_loader=SimpleNamespace(num_batch=1))
        self._logger = logging.getLogger('fake_original_engine')
        self._patience = 20
        self._iter_cnt = self._cl_len = 0

    def train_batch(self):
        self.model.train()
        self._dataloader['train_loader'].shuffle()
        total = 0.
        for value in self._dataloader['train_loader'].samples:
            self._optimizer.zero_grad()
            x = torch.tensor([[value+1.]])*torch.rand(1)
            loss = (self.model(x)-1).square().mean()
            loss.backward()
            self._optimizer.step()
            self._iter_cnt += 1
            self._cl_len = min(12, self._iter_cnt)
            total += float(loss.detach())
        return total, .1, total**.5

    def evaluate(self, mode):
        if mode != 'val':
            raise AssertionError('test must never execute')
        self.model.eval()
        with torch.no_grad():
            value = float((self.model(torch.ones(1, 1))-1).abs().mean())
        return value, .1, value+.1


class OriginalRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def temp(self):
        temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(temp.cleanup)
        return Path(temp.name)

    def identity(self):
        return dict(identity_sha256='fixture', engineering_only=True, training_config=CONFIG,
                    source_fingerprint='fixture-source')

    def test_replay_mismatch_blocks_all_off_cases(self):
        with self.assertRaisesRegex(ValueError, 'interventions blocked'):
            evaluate.verify_replay(dict(mae=13.9), 13.8)
        result = evaluate.verify_replay(dict(mae=13.800001), 13.8)
        self.assertTrue(result['full_metric_gate_applied'])
        self.assertFalse(evaluate.verify_replay(dict(mae=99.), 13.8, engineering_only=True)['full_metric_gate_applied'])

    def test_epoch_boundary_resume_preserves_weights_rng_order_curriculum_optimizer(self):
        run_a, run_b = self.temp(), self.temp()
        def seed():
            torch.manual_seed(41)
            np.random.seed(42)
        with redirect_stdout(io.StringIO()):
            seed()
            a = train.run_epochs(FakeEngine(), run_a, self.identity(), 3)
            seed()
            train.run_epochs(FakeEngine(), run_b, self.identity(), 1)
            state = torch.load(run_b/'last.pt', map_location='cpu', weights_only=True)
            b = train.run_epochs(FakeEngine(), run_b, self.identity(), 3, state)
        for key in ('model_state', 'optimizer_state', 'scheduler_state', 'train_order',
                    'rng_state', 'global_updates', 'curriculum_length', 'best_epoch', 'best_validation_mae'):
            self.assertEqual(tensor_tree_hash(a[key]), tensor_tree_hash(b[key]), key)
        self.assertFalse(json.loads((run_b/'summary.json').read_text())['test_evaluated'])
        self.assertFalse(json.loads((run_b/'summary.json').read_text())['best_selection_includes_stage_zero'])
        a_rows = [{k: v for k, v in r.items() if not k.endswith('_seconds')} for r in a['history']]
        b_rows = [{k: v for k, v in r.items() if not k.endswith('_seconds')} for r in b['history']]
        self.assertEqual(a_rows, b_rows)

    def test_twelve_states_replay_restore_and_timing_on_original_tiny_model(self):
        directory = self.temp()
        for name, count in (('type_mapping.json', 6), ('desc_mapping.json', 29)):
            (directory/name).write_text(json.dumps({str(i): i for i in range(count)}))
        modules = get_original_modules()
        torch.manual_seed(7)
        model = modules.model.IGSTGNN(dict(num_feat=1, num_hidden=8, node_hidden=4, time_emb_dim=4,
            tpd=288, k_s=2, k_t=3, layer=5, gap=3, dropout=.1, dataset='fixture',
            data_path=str(directory), use_sensor_info=False, adjs=[torch.eye(4), torch.eye(4)]),
            node_num=4, input_dim=3, output_dim=1, seq_len=12, horizon=12).eval()
        x = np.ones((2, 12, 4, 3), dtype=np.float32)*.3
        raw = dict(x_data=x, y_data=np.ones((2, 12, 4, 1), dtype=np.float32),
                   incident_features=np.asarray([[0, 1, 1, 0]]*2, dtype=np.float32),
                   incident_position=np.asarray([0, 1]), incident_distances=np.ones((2, 4, 3), dtype=np.float32),
                   sensor_data={})
        loader = SimpleNamespace(num_batch=1, get_iterator=lambda: iter([raw]))
        scaler = modules.loader.StandardScaler(0., 1.)
        calls = {case: 0 for case in evaluate.CASE_NAMES}
        with redirect_stdout(io.StringIO()):
            values, target, valid, replay, times = evaluate.screen(model, loader, scaler, modules,
                [3909, 3910], 'cpu', directory, expected_mae=99., engineering_only=True,
                passes=2, warmup=1, calls=calls)
        self.assertEqual(set(values), set(evaluate.CASE_NAMES))
        self.assertEqual(replay['batch_partition'], [2])
        self.assertTrue(all(len(records) == 2 for records in times.values()))
        self.assertTrue(all(record['wall_seconds'] >= 0 for records in times.values() for record in records))
        self.assertEqual(calls['FULL'], 6)  # replay, warmup, two timed, two restores
        self.assertTrue(all(calls[case] == 3 for case in evaluate.CASE_NAMES if case != 'FULL'))
        self.assertFalse(any(p.grad is not None for p in model.parameters()))
        self.assertTrue(json.loads((directory/'interventions.json').read_text())['rng_preserved'])
        for module in model.modules():
            self.assertFalse(module._forward_hooks)
            self.assertFalse(module._forward_pre_hooks)

    def test_screen_does_not_call_off_after_failed_full_metric_gate(self):
        class Loader:
            num_batch = 1
            def get_iterator(self):
                return iter([None])
        module = torch.nn.Linear(1, 1).eval()
        batch = (torch.ones(1, 1, 1, 1), torch.ones(1, 1, 1, 1), {}, {})
        calls = {case: 0 for case in evaluate.CASE_NAMES}
        with patch.object(evaluate, 'device_batch', return_value=batch), patch.object(evaluate,
                'run_case_with_audit', return_value=(batch[0], {})) as case, redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, 'interventions blocked'):
                evaluate.screen(module, Loader(), SimpleNamespace(inverse_transform=lambda x: x), get_original_modules(),
                                [3909], 'cpu', self.temp(), expected_mae=2., calls=calls)
        self.assertEqual(case.call_count, 1)
        self.assertEqual(calls['FULL'], 1)
        self.assertTrue(all(calls[name] == 0 for name in calls if name != 'FULL'))

    def test_resume_rejects_contradictory_selection_budget_and_curriculum(self):
        directory = self.temp()
        (directory/'best.pt').write_bytes(b'fixture')
        identity = dict(identity_sha256='fixture', training_config=CONFIG,
                        data=dict(splits=dict(train=dict(count=4))))
        state = dict(identity_sha256='fixture', completed_epoch=2, best_epoch=1,
            best_validation_mae=1., wait=1, global_updates=4, registered_parameters=455220,
            history=[dict(epoch=1, validation_mae=1., global_updates=2),
                     dict(epoch=2, validation_mae=2., global_updates=4)],
            train_order=[0, 2, 1, 3], curriculum_length=12,
            scheduler_state=dict(last_epoch=2), best_checkpoint_file='best.pt',
            best_checkpoint_sha256=sha256(directory/'best.pt'))
        train.validate_saved(state, identity, directory, 2)
        for field, changed in (('best_epoch', 2), ('wait', 0), ('global_updates', 3),
                               ('curriculum_length', 1), ('train_order', [0, 0, 1, 3]),
                               ('best_checkpoint_sha256', 'wrong')):
            with self.subTest(field=field), self.assertRaises(ValueError):
                train.validate_saved(dict(state, **{field: changed}), identity, directory, 2)

    def test_screen_source_rejects_false_complete_and_shared_wrong_budget(self):
        directory = self.temp()
        identity = dict(schema=train.SCHEMA, training_config=CONFIG, source_fingerprint='fixture-source',
            engineering_only=True, runtime={}, data=dict(splits=dict(train=dict(count=4, partition=[2, 2]))))
        identity['identity_sha256'] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        (directory/'identity.json').write_text(json.dumps(identity), encoding='utf-8')
        torch.save({}, directory/'best.pt')
        state = dict(identity_sha256=identity['identity_sha256'], completed_epoch=2, best_epoch=1,
            best_validation_mae=1., wait=1, global_updates=4, registered_parameters=455220,
            history=[dict(epoch=1, validation_mae=1., global_updates=2),
                     dict(epoch=2, validation_mae=2., global_updates=4)],
            train_order=[0, 2, 1, 3], curriculum_length=12,
            scheduler_state=dict(last_epoch=2), best_checkpoint_file='best.pt',
            best_checkpoint_sha256=sha256(directory/'best.pt'))
        def save(state, summary):
            torch.save(state, directory/'last.pt')
            (directory/'summary.json').write_text(json.dumps(summary), encoding='utf-8')
        summary = train.summary_for(state, identity)
        save(state, summary)
        self.assertEqual(evaluate.source_from_run(directory, check=True)[2], 1.)
        save(state, dict(summary, status='ORIGINAL_IGSTGNN_BASELINE_COMPLETE'))
        with self.assertRaisesRegex(ValueError, 'validated committed'):
            evaluate.source_from_run(directory, check=True)
        save(dict(state, global_updates=3), dict(summary, global_updates=3))
        with self.assertRaisesRegex(ValueError, 'update budget'):
            evaluate.source_from_run(directory, check=True)


if __name__ == '__main__':
    unittest.main()
