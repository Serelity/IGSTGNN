"""Counter-only restricted checkpoint compatibility, with CPU resume evidence."""
from collections import Counter
from contextlib import contextmanager, redirect_stdout
import copy
import io
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch
import warnings

import numpy as np
import torch
import torch._weights_only_unpickler as weights_unpickler

from experiments.chronological import original_igstgnn_checkpoint_compat as compat
from experiments.chronological import train_original_igstgnn_baseline as train
from experiments.chronological.original_igstgnn_runtime import CONFIG
from tests.test_original_igstgnn_runner import FakeEngine


class UnapprovedEval:
    def __reduce__(self):
        return eval, ('1 + 1',)


@contextmanager
def simulate_torch_2_3():
    """Emulate the missing Counter whitelist/public API without changing torch files."""
    allowed = dict(weights_unpickler._get_allowed_globals())
    allowed.pop('collections.Counter', None)
    with patch.object(weights_unpickler, '_get_allowed_globals', return_value=allowed), \
            patch.object(torch.serialization, 'safe_globals', None, create=True), \
            patch.object(torch, '__version__', '2.3.1+cu121'):
        yield allowed


def serialized(value):
    stream = io.BytesIO()
    torch.save(value, stream, pickle_protocol=2)
    return stream.getvalue()


def restricted_load(payload):
    return torch.load(io.BytesIO(payload), weights_only=True, map_location='cpu')


def cpu_rng_state():
    value = np.random.get_state()
    return dict(torch=torch.get_rng_state(), cuda=[],
                numpy=[value[0], value[1].tolist(), value[2], value[3], value[4]],
                python=random.getstate())


class OriginalCheckpointCompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def assert_tree_equal(self, expected, actual, path='root'):
        self.assertIs(type(actual), type(expected), path)
        if torch.is_tensor(expected):
            self.assertEqual(expected.dtype, actual.dtype, path)
            self.assertEqual(expected.device, actual.device, path)
            self.assertTrue(torch.equal(expected, actual), path)
        elif isinstance(expected, dict):
            self.assertEqual(expected.keys(), actual.keys(), path)
            for key in expected:
                self.assert_tree_equal(expected[key], actual[key], '{}[{}]'.format(path, key))
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(len(expected), len(actual), path)
            for index, (left, right) in enumerate(zip(expected, actual)):
                self.assert_tree_equal(left, right, '{}[{}]'.format(path, index))
        else:
            self.assertEqual(expected, actual, path)

    def fixture(self):
        parameter = torch.nn.Parameter(torch.tensor([1., 2.]))
        optimizer = torch.optim.Adam([parameter], lr=.002)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [1, 3, 3, 6], gamma=.5)
        for _ in range(2):
            optimizer.zero_grad()
            parameter.square().sum().backward()
            optimizer.step()
            scheduler.step()
        return dict(model_state={'parameter': parameter.detach().clone()},
                    optimizer_state=copy.deepcopy(optimizer.state_dict()),
                    scheduler_state=copy.deepcopy(scheduler.state_dict()),
                    rng_state=cpu_rng_state(), completed_epoch=2,
                    train_order=[2, 0, 3, 1], identity_sha256='unchanged-fixture')

    def test_legacy_counter_only_load_preserves_complete_state_and_restores_whitelist(self):
        original = self.fixture()
        payload = serialized(original)
        with simulate_torch_2_3() as allowed:
            before = dict(allowed)
            with self.assertRaises(Exception):
                restricted_load(payload)
            with compat.counter_weights_only_compat() as mode:
                self.assertEqual(mode, 'legacy_2_3_Counter_only')
                self.assertEqual(set(allowed)-set(before), {'collections.Counter'})
                self.assertIs(allowed['collections.Counter'], Counter)
                restored = restricted_load(payload)
                self.assert_tree_equal(original, restored)
                self.assertIs(type(restored['scheduler_state']['milestones']), Counter)
            self.assertEqual(allowed, before)
            with self.assertRaises(Exception):
                restricted_load(payload)

    def test_unapproved_eval_is_rejected_and_keyboard_interrupt_restores_whitelist(self):
        payload = serialized(UnapprovedEval())
        with simulate_torch_2_3() as allowed:
            before = dict(allowed)
            with self.assertRaisesRegex(KeyboardInterrupt, 'fixture interruption'):
                with compat.counter_weights_only_compat() as mode:
                    self.assertEqual(mode, 'legacy_2_3_Counter_only')
                    with patch('builtins.eval', side_effect=AssertionError('eval must never execute')) as forbidden:
                        with self.assertRaises(Exception) as rejected:
                            restricted_load(payload)
                        # Torch 2.3 and 2.8 format the rejected builtin differently.
                        self.assertIn('eval', str(rejected.exception))
                        self.assertEqual(forbidden.call_count, 0)
                    raise KeyboardInterrupt('fixture interruption')
            self.assertEqual(allowed, before)

    def test_public_safe_globals_path_is_scoped(self):
        public = getattr(torch.serialization, 'safe_globals', None)
        if not callable(public):
            self.skipTest('Installed torch has no public scoped safe globals API')
        allowed = dict(weights_unpickler._get_allowed_globals())
        allowed.pop('collections.Counter', None)
        before_user = list(torch.serialization.get_safe_globals())
        payload = serialized(self.fixture())
        with patch.object(weights_unpickler, '_get_allowed_globals', return_value=allowed):
            with compat.counter_weights_only_compat() as mode:
                self.assertEqual(mode, 'public_safe_globals')
                self.assertIs(type(restricted_load(payload)['scheduler_state']['milestones']), Counter)
            self.assertEqual(torch.serialization.get_safe_globals(), before_user)
            self.assertNotIn('collections.Counter', allowed)

    def test_public_pre_registered_counter_plain_and_alias_are_preserved(self):
        if not callable(getattr(torch.serialization, 'safe_globals', None)):
            self.skipTest('Installed torch has no public scoped safe globals API')
        saved_user = list(torch.serialization.get_safe_globals())
        allowed = dict(weights_unpickler._get_allowed_globals())
        allowed.pop('collections.Counter', None)
        payload = serialized(self.fixture())
        try:
            for registration in (Counter, (Counter, 'collections.Counter')):
                with self.subTest(registration=registration):
                    torch.serialization.clear_safe_globals()
                    torch.serialization.add_safe_globals([registration])
                    before_user = list(torch.serialization.get_safe_globals())
                    before_native = dict(allowed)
                    with patch.object(weights_unpickler, '_get_allowed_globals', return_value=allowed):
                        with compat.counter_weights_only_compat() as mode:
                            self.assertEqual(mode, 'public_existing_Counter')
                            self.assertIs(type(restricted_load(payload)['scheduler_state']['milestones']), Counter)
                            self.assertEqual(torch.serialization.get_safe_globals(), before_user)
                        self.assertEqual(torch.serialization.get_safe_globals(), before_user)
                        self.assertIs(type(restricted_load(payload)['scheduler_state']['milestones']), Counter)
                    self.assertEqual(allowed, before_native)
        finally:
            torch.serialization.clear_safe_globals()
            torch.serialization.add_safe_globals(saved_user)

    def test_public_counter_alias_to_another_class_rejects_without_mutation(self):
        if not callable(getattr(torch.serialization, 'safe_globals', None)):
            self.skipTest('Installed torch has no public scoped safe globals API')
        saved_user = list(torch.serialization.get_safe_globals())
        allowed = dict(weights_unpickler._get_allowed_globals())
        allowed.pop('collections.Counter', None)
        before_native = dict(allowed)
        try:
            torch.serialization.clear_safe_globals()
            torch.serialization.add_safe_globals([(dict, 'collections.Counter')])
            before_user = list(torch.serialization.get_safe_globals())
            with patch.object(weights_unpickler, '_get_allowed_globals', return_value=allowed):
                with self.assertRaisesRegex(ValueError, 'not stdlib Counter'):
                    with compat.counter_weights_only_compat():
                        self.fail('A conflicting public Counter alias must reject before entry')
            self.assertEqual(torch.serialization.get_safe_globals(), before_user)
            self.assertEqual(allowed, before_native)
        finally:
            torch.serialization.clear_safe_globals()
            torch.serialization.add_safe_globals(saved_user)

    def test_native_counter_whitelist_needs_no_mutation(self):
        allowed = dict(weights_unpickler._get_allowed_globals())
        allowed['collections.Counter'] = Counter
        before = dict(allowed)
        with patch.object(weights_unpickler, '_get_allowed_globals', return_value=allowed):
            with compat.counter_weights_only_compat() as mode:
                self.assertEqual(mode, 'native')
                self.assertIs(type(restricted_load(serialized(self.fixture()))['scheduler_state']['milestones']), Counter)
            self.assertEqual(allowed, before)

    def test_nested_legacy_context_preserves_outer_counter_until_outer_exit(self):
        payload = serialized(self.fixture())
        with simulate_torch_2_3() as allowed:
            before = dict(allowed)
            with compat.counter_weights_only_compat() as outer_mode:
                self.assertEqual(outer_mode, 'legacy_2_3_Counter_only')
                outer_allowed = dict(allowed)
                with compat.counter_weights_only_compat() as inner_mode:
                    self.assertEqual(inner_mode, 'native')
                    self.assertIs(type(restricted_load(payload)['scheduler_state']['milestones']), Counter)
                self.assertEqual(allowed, outer_allowed)
                self.assertIs(allowed['collections.Counter'], Counter)
                self.assertIs(type(restricted_load(payload)['scheduler_state']['milestones']), Counter)
            self.assertEqual(allowed, before)

    def test_counter_name_mapped_to_another_class_is_rejected_without_mutation(self):
        with simulate_torch_2_3() as allowed:
            allowed['collections.Counter'] = dict
            before = dict(allowed)
            with self.assertRaisesRegex(ValueError, 'not stdlib Counter'):
                with compat.counter_weights_only_compat():
                    self.fail('A conflicting Counter allowlist entry must reject before entry')
            self.assertEqual(allowed, before)
            self.assertIs(allowed['collections.Counter'], dict)

    def test_scheduler_counter_semantics_include_duplicate_and_explicit_epoch_steps(self):
        original = self.fixture()
        with simulate_torch_2_3():
            with compat.counter_weights_only_compat():
                restored = restricted_load(serialized(original))

        def trace(state, explicit):
            parameter = torch.nn.Parameter(state['model_state']['parameter'].clone())
            optimizer = torch.optim.Adam([parameter], lr=.002)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [1, 3, 3, 6], gamma=.5)
            optimizer.load_state_dict(copy.deepcopy(state['optimizer_state']))
            scheduler.load_state_dict(copy.deepcopy(state['scheduler_state']))
            self.assertIs(type(scheduler.milestones), Counter)
            values = []
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                for epoch in range(3, 9):
                    optimizer.zero_grad()
                    parameter.square().sum().backward()
                    optimizer.step()
                    if explicit:
                        scheduler.step(epoch)
                    else:
                        scheduler.step()
                    values.append((scheduler.last_epoch, scheduler.get_last_lr()[0], parameter.detach().clone()))
            return values

        for explicit in (False, True):
            with self.subTest(explicit_epoch=explicit):
                expected, actual = trace(original, explicit), trace(restored, explicit)
                self.assert_tree_equal(expected, actual)
                self.assertEqual([row[1] for row in actual], [.00025, .00025, .00025, .000125, .000125, .000125])

    def test_original_training_loop_exact_resume_with_legacy_restricted_loader(self):
        def directory():
            temporary = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
            self.addCleanup(temporary.cleanup)
            return Path(temporary.name)

        def seed():
            torch.random.default_generator.manual_seed(41)
            np.random.seed(42)
            random.seed(43)

        identity = dict(identity_sha256='fixture', engineering_only=True,
                        training_config=CONFIG, source_fingerprint='fixture-source')
        run_a, run_b = directory(), directory()
        with patch.object(train, 'rng_state', side_effect=cpu_rng_state), redirect_stdout(io.StringIO()):
            seed()
            continuous = train.run_epochs(FakeEngine(), run_a, identity, 4)
            seed()
            train.run_epochs(FakeEngine(), run_b, identity, 1)
            with simulate_torch_2_3() as allowed:
                before = dict(allowed)
                with compat.counter_weights_only_compat():
                    state = torch.load(run_b/'last.pt', map_location='cpu', weights_only=True)
                    resumed = train.run_epochs(FakeEngine(), run_b, identity, 4, state)
                self.assertEqual(allowed, before)
        for key in ('model_state', 'optimizer_state', 'scheduler_state', 'train_order', 'rng_state',
                    'global_updates', 'curriculum_length', 'best_epoch', 'best_validation_mae'):
            self.assert_tree_equal(continuous[key], resumed[key], key)
        rows = lambda state: [{key: value for key, value in row.items() if not key.endswith('_seconds')}
                              for row in state['history']]
        self.assertEqual(rows(continuous), rows(resumed))
        self.assertFalse(json.loads((run_b/'summary.json').read_text())['test_evaluated'])
        with simulate_torch_2_3():
            with compat.counter_weights_only_compat():
                committed = torch.load(run_b/'last.pt', map_location='cpu', weights_only=True)
        self.assert_tree_equal(resumed, committed)


if __name__ == '__main__':
    unittest.main()
