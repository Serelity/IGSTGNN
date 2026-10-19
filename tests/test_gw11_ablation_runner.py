"""Server-only tests of replay order, source preservation and read-only guards.

These tests use tiny source doubles, not the published arrays/native model.
Do not execute Python, imports, syntax checks or tests on the workstation.
"""
from copy import deepcopy
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from experiments.state_gate_gw11 import ablate_last as runner


class CompleteSumDouble(nn.Module):
    def __init__(self):
        super().__init__()
        self.arm, self.stage = 'SUM_CONTROL', 'joint'
        self.body = nn.Parameter(torch.tensor(2.))
        self.head = nn.Linear(1, 1)
        with torch.no_grad():
            self.head.weight.fill_(0.)
            self.head.bias.fill_(7.)

    def forward(self, value):
        return value*self.body+self.head(value)


class OuterDouble(nn.Module):
    def __init__(self):
        super().__init__()
        self.source = CompleteSumDouble()
        self.adapter = nn.Parameter(torch.tensor(3.))
        self.stale = False

    def assert_cache_clear(self):
        if self.stale:
            raise AssertionError('Stale source double')

    def forward(self, value):
        return self.source(value)+self.adapter


def prediction():
    return np.full((2, 12, 3), 10., dtype=np.float32)


class GW11AblationRunnerTests(unittest.TestCase):
    def test_source_only_bypasses_adapter_and_keeps_trained_sum_head_and_flags(self):
        model = OuterDouble()
        original_hash = runner.base.tensor_tree_hash(model.state_dict())
        flags = runner.parameter_flags(model)
        source = runner.source_only(model)
        self.assertIs(source, model.source)
        with torch.no_grad():
            torch.testing.assert_close(source(torch.ones(1, 1)), torch.tensor([[9.]]))
            torch.testing.assert_close(model(torch.ones(1, 1)), torch.tensor([[12.]]))
        runner.assert_model_unchanged(model, original_hash, flags)

    def test_source_only_rejects_missing_head_wrong_source_and_frozen_flags(self):
        for mutation in ('head', 'arm', 'stage', 'flags', 'cache'):
            model = OuterDouble()
            if mutation == 'head': model.source.head = None
            elif mutation == 'arm': model.source.arm = 'R_CONTINUE'
            elif mutation == 'stage': model.source.stage = 'probe'
            elif mutation == 'flags': model.source.body.requires_grad_(False)
            else: model.stale = True
            with self.subTest(mutation=mutation), self.assertRaises((ValueError, AssertionError)):
                runner.source_only(model)

    def test_exact_replay_rejects_even_tiny_differences_dtype_shape_and_nonfinite(self):
        original = prediction()
        certificate = runner.exact_replay(original.copy(), original, 'fixture')
        self.assertTrue(certificate['passed'])
        self.assertEqual(certificate['differing_points'], 0)
        changed = original.copy()
        changed[0, 0, 0] = np.nextafter(changed[0, 0, 0], np.float32(100.))
        for value in (changed, original.astype(np.float64), original[:, :11], original*np.nan):
            with self.subTest(dtype=value.dtype, shape=value.shape), self.assertRaisesRegex(ValueError, 'Exact'):
                runner.exact_replay(value, original, 'fixture')

    def test_validation_identity_allows_only_the_validation_subset(self):
        saved = dict(hashes={'release': 'a'}, normalization={'mean': 1., 'std': 2.},
            splits={'train': {'count': 3909}, 'val': {'count': 837}}, test_evaluated=False)
        observed = deepcopy(saved)
        del observed['splits']['train']
        self.assertTrue(runner.validation_identity(observed, saved))
        for key in ('normalization', 'split', 'extra_loader', 'test'):
            changed = deepcopy(observed)
            if key == 'normalization': changed['normalization']['std'] = 3.
            elif key == 'split': changed['splits']['val']['count'] = 838
            elif key == 'extra_loader': changed['splits']['train'] = {'count': 3909}
            else: changed['test_evaluated'] = True
            with self.subTest(key=key), self.assertRaises(ValueError):
                runner.validation_identity(changed, saved)

    def test_failed_on_cache_replay_stops_before_off_inference(self):
        model = OuterDouble(); correct = prediction(); changed = correct+1.
        with patch.object(runner.gw11, 'evaluate', return_value=({}, changed, {}, {})) as evaluate:
            with self.assertRaisesRegex(ValueError, 'Exact prediction replay'):
                runner.evaluate_pair(model, object(), 'cpu', {}, {'pred__STATE_GATE': correct}, {}, 'STATE_GATE')
        self.assertEqual(evaluate.call_count, 1)

    def test_pair_runs_on_then_complete_source_without_changing_parameters(self):
        model = OuterDouble(); on = prediction(); off = on-2.; seen = []
        def evaluate(which, *args):
            seen.append(which)
            return {'record': 'matched'}, (on if which is model else off), {}, {}
        with patch.object(runner.gw11, 'evaluate', side_effect=evaluate), \
                patch.object(runner.gw11, 'same_record', return_value=True):
            actual_on, actual_off, evidence = runner.evaluate_pair(model, object(), 'cpu', {},
                {'pred__STATE_GATE': on}, {'record': 'matched'}, 'STATE_GATE')
        self.assertEqual(seen, [model, model.source])
        np.testing.assert_array_equal(actual_on, on)
        np.testing.assert_array_equal(actual_off, off)
        self.assertEqual(evidence['optimizer_updates'], 0)
        self.assertTrue(evidence['trainability_flags_unchanged'])

    def test_sum_continue_identity_control_rejects_any_off_difference(self):
        model = OuterDouble(); on = prediction()
        replies = [({}, on, {}, {}), ({}, on+1., {}, {})]
        with patch.object(runner.gw11, 'evaluate', side_effect=replies), \
                patch.object(runner.gw11, 'same_record', return_value=True):
            with self.assertRaisesRegex(ValueError, 'SUM_CONTINUE ON/OFF'):
                runner.evaluate_pair(model, object(), 'cpu', {}, {'pred__SUM_CONTINUE': on}, {}, 'SUM_CONTINUE')

    def test_readonly_guard_detects_tensor_flag_and_gradient_mutation(self):
        for mutation in ('tensor', 'flag', 'gradient'):
            model = OuterDouble(); before = runner.base.tensor_tree_hash(model.state_dict())
            flags = runner.parameter_flags(model)
            if mutation == 'tensor':
                with torch.no_grad(): model.source.body.add_(1.)
            elif mutation == 'flag': model.source.body.requires_grad_(False)
            else: model.source.body.grad = torch.ones_like(model.source.body)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                runner.assert_model_unchanged(model, before, flags)

    def test_bound_checkpoint_requires_hash_and_rechecks_after_deserialization(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'last.pt'; path.write_bytes(b'opaque checkpoint')
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            certificate = {'file_digests': {str(path.resolve()): digest}}
            with patch.object(runner.base, 'load_checkpoint', return_value={'fixture': True}):
                self.assertEqual(runner.load_bound_checkpoint(path, certificate), {'fixture': True})
            with patch.object(runner.base, 'load_checkpoint') as loader:
                with self.assertRaisesRegex(ValueError, 'Unbound'):
                    runner.load_bound_checkpoint(path, {'file_digests': {}})
                loader.assert_not_called()
            def mutate(*_):
                path.write_bytes(b'changed')
                return {}
            with patch.object(runner.base, 'load_checkpoint', side_effect=mutate):
                with self.assertRaisesRegex(ValueError, 'changed while loading'):
                    runner.load_bound_checkpoint(path, certificate)

    def test_cli_has_no_training_resume_or_checkpoint_selection_switch(self):
        with tempfile.TemporaryDirectory() as directory:
            required = ['--run-dir', directory, '--data-dir', directory, '--output-dir', directory]
            args = runner.parse_args(required)
            self.assertEqual(args.threads, 3)
            self.assertEqual(args.device, 'cuda:0')
            self.assertTrue(args.output_dir.is_absolute())
            for forbidden in (['--resume'], ['--epochs', '1'], ['--selection', 'best'],
                    ['--threads', '0'], ['--train']):
                with self.subTest(forbidden=forbidden), self.assertRaises(SystemExit):
                    runner.parse_args(required+forbidden)


if __name__ == '__main__':
    unittest.main()
