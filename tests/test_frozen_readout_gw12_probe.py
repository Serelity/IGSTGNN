"""Server-only guards for source freezing, paired snapshots and resume modes.

No native author model or released arrays are used by these tiny doubles.
Publishing must not execute Python, imports, syntax checks or tests locally.
"""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from experiments.frozen_readout_gw12 import probe
from experiments.frozen_readout_gw12.protocol import ARMS


class SumSource(nn.Module):
    def __init__(self):
        super().__init__()
        self.arm, self.stage = 'SUM_CONTROL', 'joint'
        self.node_emb_u = nn.Parameter(torch.ones(2, 3))
        self.node_emb_d = nn.Parameter(torch.full((2, 3), 2.))
        self.body = nn.Parameter(torch.tensor(2.))
        self.head = nn.Linear(1, 1)
        with torch.no_grad():
            self.head.weight.zero_(); self.head.bias.fill_(3.)
        self.calls = 0

    @property
    def original(self):
        return self

    def assert_cache_clear(self):
        return True

    def forward(self, history, incident_data=None, sensor_data=None):
        self.calls += 1
        return history[..., :1]*self.body+self.head(history[..., :1])


def engine_fixture():
    source = SumSource()
    data = SimpleNamespace(config=dict(lrate=.002, wdecay=1e-5, clip_grad_value=5.))
    engine = probe.Engine(source, data, object(), 'cpu', {}, {})
    x = torch.zeros(2, 12, 2, 3); x[..., 0] = 1.; x[..., 1] = .25; x[..., 2] = .5
    return engine, x


class GW12ProbeTests(unittest.TestCase):
    def test_both_optimizers_exclude_complete_source_and_own_exact_heads(self):
        engine, _ = engine_fixture()
        source = {id(value) for value in engine.source.parameters()}
        previous = set()
        for arm in ARMS:
            optimizer = {id(value) for group in engine.optimizers[arm].param_groups for value in group['params']}
            self.assertEqual(optimizer, {id(value) for value in engine.heads[arm].parameters()})
            self.assertFalse(optimizer & source)
            self.assertFalse(optimizer & previous)
            previous |= optimizer
        initial = probe.head_initializations(engine.heads)
        self.assertEqual({row['registered_parameters'] for row in initial.values()}, {725})
        self.assertEqual(len({row['head_tensor_sha256'] for row in initial.values()}), 1)

    def test_shared_source_forward_is_detached_and_keeps_fitted_sum_head(self):
        engine, x = engine_fixture()
        frozen, features = engine.frozen_forward(x, {}, {})
        self.assertEqual(engine.source.calls, 1)
        self.assertFalse(frozen.requires_grad)
        self.assertTrue(all(not value.requires_grad for value in features.values()))
        torch.testing.assert_close(frozen, torch.full((2, 12, 2, 1), 5.))
        self.assertTrue(all(value.requires_grad for value in engine.source.parameters()))
        self.assertFalse(any(module.training for module in engine.source.modules()))

    def test_head_training_updates_both_heads_without_source_gradients_or_tensor_changes(self):
        engine, x = engine_fixture()
        before = probe.base.tensor_tree_hash(engine.source.state_dict())
        initial = probe.head_initializations(engine.heads)
        engine.heads.train()
        frozen, features = engine.frozen_forward(x, {}, {})
        for arm in ARMS:
            optimizer = engine.optimizers[arm]; optimizer.zero_grad(set_to_none=True)
            loss = (frozen+engine.heads[arm](features[arm])).abs().mean()
            loss.backward(); optimizer.step()
            self.assertNotEqual(probe.base.tensor_tree_hash(engine.heads[arm].state_dict()), initial[arm]['head_tensor_sha256'])
        engine.check_source()
        self.assertEqual(before, probe.base.tensor_tree_hash(engine.source.state_dict()))
        self.assertTrue(all(value.grad is None for value in engine.source.parameters()))

    def test_source_guard_rejects_mode_flags_gradients_and_buffer_or_tensor_changes(self):
        for mutation in ('mode', 'flag', 'gradient', 'tensor'):
            engine, _ = engine_fixture()
            if mutation == 'mode': engine.source.head.train()
            elif mutation == 'flag': engine.source.body.requires_grad_(False)
            elif mutation == 'gradient': engine.source.body.grad = torch.ones_like(engine.source.body)
            else:
                with torch.no_grad(): engine.source.head.bias.add_(1.)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                engine.check_source()

    def test_commit_owns_last_head_and_optimizer_tensors_despite_later_best_loads(self):
        engine, x = engine_fixture()
        frozen, features = engine.frozen_forward(x, {}, {})
        for arm in ARMS:
            (frozen+engine.heads[arm](features[arm])).mean().backward()
            engine.optimizers[arm].step()
        identity = dict(identity_sha256='a'*64, initializations=probe.head_initializations(engine.heads))
        saved = dict(history=[])
        with tempfile.TemporaryDirectory() as directory, patch.object(probe.epoch_tools, 'atomic_save') as writer:
            probe.commit(engine, saved, Path(directory), identity)
            self.assertEqual(writer.call_count, 1)
            self.assertEqual(Path(writer.call_args.args[0]).name, 'last.pt')
            original_hashes = dict(saved['heads_tensor_sha256'])
            optimizer_hash = probe.base.tensor_tree_hash(saved['optimizers_state'])
            with torch.no_grad():
                for arm in ARMS:
                    engine.heads[arm].adapters[0].output.bias.add_(100.)
                    for state in engine.optimizers[arm].state.values():
                        for value in state.values():
                            if torch.is_tensor(value): value.add_(100.)
            self.assertEqual(optimizer_hash, probe.base.tensor_tree_hash(saved['optimizers_state']))
            for arm in ARMS:
                self.assertEqual(original_hashes[arm], probe.base.tensor_tree_hash(saved['heads_state'][arm]))

    def test_marker_cannot_be_created_after_committed_training_or_rebound(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); identity = {'identity_sha256': 'a'*64}
            with self.assertRaisesRegex(ValueError, 'lost its original'):
                probe.marker(root, identity, 1)
            probe.marker(root, identity, 0)
            before = (root/'training_started.json').read_bytes()
            probe.marker(root, identity, 3)
            self.assertEqual(before, (root/'training_started.json').read_bytes())
            with self.assertRaisesRegex(ValueError, 'marker changed'):
                probe.marker(root, {'identity_sha256': 'b'*64}, 0)

    def test_explicit_fixed_epoch_rng_and_target_order_do_not_need_loader_shuffle(self):
        order = probe.epoch_order(2025, 2, 3909)
        np.testing.assert_array_equal(np.sort(order), np.arange(3909))
        np.testing.assert_array_equal(order, np.random.RandomState(2027).permutation(3909))
        self.assertEqual(probe.epoch_seed(2025, 2), 22025)

    def test_cli_has_only_fixed_probe_modes_and_no_new_budget_or_seed_switch(self):
        with tempfile.TemporaryDirectory() as directory:
            required = ['--gw11-run', directory, '--data-dir', directory, '--run-dir', directory]
            args = probe.parse_args(required)
            self.assertTrue(args.run_dir.is_absolute())
            self.assertFalse(args.resume or args.evaluate_only)
            for extra in (['--epochs', '4'], ['--seed', '7'], ['--resume', '--evaluate-only'], ['--threads', '0']):
                with self.subTest(extra=extra), self.assertRaises(SystemExit):
                    probe.parse_args(required+extra)


if __name__ == '__main__':
    unittest.main()
