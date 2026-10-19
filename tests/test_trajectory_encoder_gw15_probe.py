"""Server-only checks for the new internal encoder gradient boundary and resume.

Tiny sources exercise differentiable embedding injection, not author datasets.
The publication process must not run Python locally.
"""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from experiments.trajectory_encoder_gw15 import probe
from experiments.trajectory_encoder_gw15.protocol import ARMS


class SumSource(nn.Module):
    def __init__(self):
        super().__init__()
        self.arm, self.stage = 'SUM_CONTROL', 'joint'
        self.embedding = nn.Linear(1, 2)
        self.head = nn.Linear(2, 1)
        with torch.no_grad():
            self.embedding.weight.fill_(.1); self.embedding.bias.fill_(.2)
            self.head.weight.fill_(.5); self.head.bias.fill_(.3)
        self.calls = 0

    @property
    def original(self):
        return self

    def assert_cache_clear(self):
        return True

    def start_diagnostics(self):
        self.calls = 0

    def diagnostics(self):
        return dict(complete_forwards=self.calls)

    def forward(self, history, incident_data=None, sensor_data=None):
        self.calls += 1
        return self.head(torch.tanh(self.embedding(history[..., :1])))


def fixture():
    source = SumSource()
    data = SimpleNamespace(config=dict(lrate=.002, wdecay=1e-5, clip_grad_value=5.),
        scaler=SimpleNamespace(inverse_transform=lambda value: value))
    metrics = SimpleNamespace(masked_mae=lambda pred, target, null: (pred-target).abs().mean())
    engine = probe.Engine(source, data, SimpleNamespace(metrics=metrics), 'cpu', {}, {})
    x = torch.zeros(2, 12, 2, 3)
    x[..., 0] = torch.tensor([0., 1., 2., 4., 3., 2., 1., 3., 5., 4., 2., 1.])[None, :, None]
    return engine, x


class GW15ProbeTests(unittest.TestCase):
    def test_both_optimizers_own_only_their_identically_initialized_encoder(self):
        engine, _ = fixture(); previous = set()
        source = {id(value) for value in engine.source.parameters()}
        for arm in ARMS:
            params = {id(value) for group in engine.optimizers[arm].param_groups for value in group['params']}
            self.assertEqual(params, {id(value) for value in engine.heads[arm].parameters()})
            self.assertFalse(params & source or params & previous)
            previous |= params
        initial = probe.head_initializations(engine.heads)
        self.assertEqual(len({row['head_tensor_sha256'] for row in initial.values()}), 1)
        self.assertEqual({row['registered_parameters'] for row in initial.values()}, {576})

    def test_stage_zero_is_exact_and_source_keeps_fitted_decoder(self):
        engine, x = fixture()
        frozen = engine.frozen_forward(x, {}, {})
        self.assertFalse(frozen.requires_grad)
        with torch.no_grad():
            for arm in ARMS:
                self.assertTrue(torch.equal(frozen, engine.encoded_forward(arm, x, {}, {})))
        self.assertTrue(all(parameter.requires_grad for parameter in engine.source.parameters()))
        engine.check_source()

    def test_encoder_step_backpropagates_through_source_without_accumulating_source_grad(self):
        engine, x = fixture()
        before = probe.base.tensor_tree_hash(engine.source.state_dict())
        initial = probe.head_initializations(engine.heads)
        for arm in ARMS:
            loss = engine.encoder_step(arm, x, {}, {}, torch.zeros(2, 12, 2, 1), 0.)
            self.assertTrue(np.isfinite(loss))
            self.assertNotEqual(probe.base.tensor_tree_hash(engine.heads[arm].state_dict()),
                initial[arm]['head_tensor_sha256'])
        engine.check_source()
        self.assertEqual(before, probe.base.tensor_tree_hash(engine.source.state_dict()))
        self.assertTrue(all(value.grad is None for value in engine.source.parameters()))
        with torch.no_grad():
            frozen = engine.frozen_forward(x, {}, {})
            self.assertFalse(torch.equal(frozen, engine.encoded_forward('ORDERED_DELTA', x, {}, {})))

    def test_guard_rejects_source_modes_flags_gradients_and_tensor_mutations(self):
        for kind in ('mode', 'flag', 'gradient', 'tensor'):
            engine, _ = fixture()
            if kind == 'mode': engine.source.head.train()
            elif kind == 'flag': engine.source.head.weight.requires_grad_(False)
            elif kind == 'gradient': engine.source.head.weight.grad = torch.ones_like(engine.source.head.weight)
            else:
                with torch.no_grad(): engine.source.head.bias.add_(1.)
            with self.subTest(kind=kind), self.assertRaises(ValueError): engine.check_source()

    def test_commit_clones_last_encoder_and_optimizer_tensors_before_best_loading(self):
        engine, x = fixture()
        for arm in ARMS: engine.encoder_step(arm, x, {}, {}, torch.zeros(2, 12, 2, 1), 0.)
        identity = dict(identity_sha256='a'*64, initializations=probe.head_initializations(engine.heads))
        saved = dict(history=[])
        with tempfile.TemporaryDirectory() as directory, patch.object(probe.epoch_tools, 'atomic_save') as writer:
            probe.commit(engine, saved, Path(directory), identity)
            self.assertEqual(writer.call_count, 1)
            hashes = dict(saved['heads_tensor_sha256'])
            optimizer_hash = probe.base.tensor_tree_hash(saved['optimizers_state'])
            with torch.no_grad():
                for arm in ARMS:
                    for value in engine.heads[arm].parameters(): value.add_(100.)
                    for state in engine.optimizers[arm].state.values():
                        for value in state.values():
                            if torch.is_tensor(value): value.add_(100.)
            self.assertEqual(optimizer_hash, probe.base.tensor_tree_hash(saved['optimizers_state']))
            for arm in ARMS:
                self.assertEqual(hashes[arm], probe.base.tensor_tree_hash(saved['heads_state'][arm]))

    def test_marker_cannot_be_recreated_after_training_or_rebound_to_another_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); identity = {'identity_sha256': 'a'*64}
            with self.assertRaisesRegex(ValueError, 'lost its original'): probe.marker(root, identity, 1)
            probe.marker(root, identity, 0)
            before = (root/'training_started.json').read_bytes()
            probe.marker(root, identity, 3)
            self.assertEqual(before, (root/'training_started.json').read_bytes())
            with self.assertRaisesRegex(ValueError, 'marker changed'):
                probe.marker(root, {'identity_sha256': 'b'*64}, 0)

    def test_fixed_epoch_order_is_independent_of_loader_shuffle(self):
        order = probe.epoch_order(2025, 2, 3909)
        np.testing.assert_array_equal(order, np.random.RandomState(2027).permutation(3909))
        self.assertEqual(probe.epoch_seed(2025, 2), 22025)

    def test_cli_exposes_resume_and_evaluation_but_no_budget_search(self):
        with tempfile.TemporaryDirectory() as directory:
            required = ['--gw11-run', directory, '--data-dir', directory, '--run-dir', directory]
            self.assertTrue(probe.parse_args(required).run_dir.is_absolute())
            for extra in (['--epochs', '4'], ['--seed', '7'], ['--resume', '--evaluate-only'], ['--threads', '0']):
                with self.subTest(extra=extra), self.assertRaises(SystemExit): probe.parse_args(required+extra)


if __name__ == '__main__':
    unittest.main()
