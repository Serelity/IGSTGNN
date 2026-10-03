"""Frozen Adam settings, complete moments and update counts during v12m recovery."""

import copy
from pathlib import Path
import tempfile
import unittest

import torch

from experiments.chronological import train_vector_output_scope as experiment


class AdamRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'last_adapter.pt'
        self.protocol = copy.deepcopy(experiment.base.load_protocol())
        self.protocol['training']['epochs'] = 2
        self.protocol['training']['batch_size'] = 2
        self.settings = self.protocol['training']
        self.baseline = {cohort: {'mae': dict.fromkeys(experiment.base.REGIONS, 10.)}
                         for cohort in experiment.base.COHORTS}

    def adam(self, parameters):
        return torch.optim.Adam(parameters, lr=self.settings['learning_rate'],
                                eps=self.settings['adam_eps'], weight_decay=self.settings['weight_decay'])

    @staticmethod
    def update(module, optimizer):
        optimizer.zero_grad(set_to_none=True)
        sum((parameter.square() + .25 * parameter).sum() for parameter in module.parameters()).backward()
        optimizer.step()

    def fixture(self, epochs=2):
        torch.manual_seed(45)
        module = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 3))
        optimizer = self.adam(module.parameters())
        initial = experiment.base.cpu_tree(module.state_dict())
        expected = {'fit_samples': 5, 'training': self.settings, 'loss': 'candidate_early',
                    'protocol_sha256': experiment.PROTOCOL_SHA256, 'run_identity_sha256': 'synthetic_identity'}
        history = []
        for epoch in range(1, epochs + 1):
            for _ in range(3):  # ceil(5 fit samples / batch size 2)
                self.update(module, optimizer)
            decisions = {policy: {**experiment.scope.selection_decision(
                self.baseline, self.baseline, self.baseline, self.protocol), 'best_epoch': 0}
                for policy in experiment.scope.POLICIES}
            history.append({'epoch': epoch, 'training': {'optimizer_steps': 3, 'loss_region': 'candidate_early'},
                'selection': {policy: copy.deepcopy(self.baseline) for policy in experiment.scope.POLICIES},
                'decisions': decisions, 'adapter_state_sha256': experiment.alignment.tensor_hash(module.state_dict())})
        saved = {'format_version': experiment.RECOVERY_FORMAT, 'identity': expected,
            'epoch': epochs, 'history': history, 'adapter_state': experiment.base.cpu_tree(module.state_dict()),
            'best': {policy: {'epoch': 0, 'selection_metrics': self.baseline, 'state': initial}
                     for policy in experiment.scope.POLICIES},
            'optimizer_state': experiment.base.cpu_tree(optimizer.state_dict()),
            'rng_cpu': torch.get_rng_state(), 'rng_cuda': None}
        return saved, expected, initial, module, optimizer

    def restore(self, saved, expected, initial):
        torch.save(saved, self.path)
        before = self.path.read_bytes()
        try:
            return experiment.restore_fit(self.path, expected, initial, self.baseline, self.protocol)
        finally:
            self.assertEqual(self.path.read_bytes(), before, 'Recovery must preserve the source checkpoint')

    def test_valid_recovery_preserves_adam_and_next_update_exactly(self):
        saved, expected, initial, uninterrupted, optimizer = self.fixture()
        rng = torch.get_rng_state().clone()
        restored = self.restore(saved, expected, initial)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        resumed = copy.deepcopy(uninterrupted)
        resumed.load_state_dict(restored['adapter_state'])
        resumed_optimizer = self.adam(resumed.parameters())
        resumed_optimizer.load_state_dict(restored['optimizer_state'])
        self.update(uninterrupted, optimizer)
        self.update(resumed, resumed_optimizer)
        self.assertEqual(experiment.alignment.tensor_hash(uninterrupted.state_dict()),
                         experiment.alignment.tensor_hash(resumed.state_dict()))
        for parameter_id, state in optimizer.state_dict()['state'].items():
            for field, value in state.items():
                self.assertTrue(torch.equal(value, resumed_optimizer.state_dict()['state'][parameter_id][field]))

    def test_group_configuration_changes_are_rejected_before_load(self):
        saved, expected, initial, _, _ = self.fixture()
        changes = {'lr': .5, 'weight_decay': .9, 'eps': 1e-4, 'betas': (.8, .999),
                   'amsgrad': True, 'maximize': True, 'foreach': True, 'capturable': True,
                   'differentiable': True, 'fused': True, 'unexpected_setting': 1}
        for field, value in changes.items():
            with self.subTest(field=field):
                corrupted = copy.deepcopy(saved)
                corrupted['optimizer_state']['param_groups'][0][field] = value
                with self.assertRaisesRegex(ValueError, 'Adam parameter groups/settings'):
                    self.restore(corrupted, expected, initial)
        for value in (float('nan'), float('inf'), torch.tensor(.001)):
            with self.subTest(invalid_lr=value):
                corrupted = copy.deepcopy(saved)
                corrupted['optimizer_state']['param_groups'][0]['lr'] = value
                with self.assertRaisesRegex(ValueError, 'Adam parameter groups/settings'):
                    self.restore(corrupted, expected, initial)

    def test_parameter_index_order_and_state_keys_are_required(self):
        saved, expected, initial, _, _ = self.fixture()
        for change in ('reordered', 'duplicate', 'empty', 'missing', 'extra', 'bool_key'):
            with self.subTest(change=change):
                corrupted = copy.deepcopy(saved)
                state = corrupted['optimizer_state']
                if change == 'reordered':
                    state['param_groups'][0]['params'].reverse()
                elif change == 'duplicate':
                    state['param_groups'][0]['params'][1] = state['param_groups'][0]['params'][0]
                elif change == 'empty':
                    state['state'].clear()
                elif change == 'missing':
                    del state['state'][0]
                elif change == 'extra':
                    state['state'][999] = copy.deepcopy(state['state'][0])
                else:
                    state['state'][False] = state['state'].pop(0)
                with self.assertRaisesRegex(ValueError, 'Adam parameter'):
                    self.restore(corrupted, expected, initial)

    def test_every_parameter_step_must_equal_completed_total_updates(self):
        saved, expected, initial, _, _ = self.fixture()
        for step in (torch.tensor(0.), torch.tensor(5.), torch.tensor(7.), torch.tensor(-1.),
                     torch.tensor(float('nan')), torch.tensor(float('inf')), torch.tensor([6.]), 6):
            with self.subTest(step=step):
                corrupted = copy.deepcopy(saved)
                corrupted['optimizer_state']['state'][3]['step'] = step
                with self.assertRaisesRegex(ValueError, 'Adam step'):
                    self.restore(corrupted, expected, initial)

    def test_moment_shape_dtype_finiteness_fields_and_variance_are_checked(self):
        saved, expected, initial, _, _ = self.fixture()
        for change in ('shape', 'dtype', 'nan', 'negative_variance', 'missing_field', 'extra_field'):
            with self.subTest(change=change):
                corrupted = copy.deepcopy(saved)
                state = corrupted['optimizer_state']['state'][0]
                if change == 'shape':
                    state['exp_avg'] = state['exp_avg'].flatten()
                elif change == 'dtype':
                    state['exp_avg'] = state['exp_avg'].double()
                elif change == 'nan':
                    state['exp_avg'][0, 0] = float('nan')
                elif change == 'negative_variance':
                    state['exp_avg_sq'][0, 0] = -1.
                elif change == 'missing_field':
                    del state['exp_avg_sq']
                else:
                    state['max_exp_avg_sq'] = state['exp_avg_sq'].clone()
                with self.assertRaisesRegex(ValueError, 'Adam moment|Adam second moment'):
                    self.restore(corrupted, expected, initial)

    def test_epoch_zero_accepts_empty_state_and_rejects_nonempty_state(self):
        zero, expected, initial, _, _ = self.fixture(epochs=0)
        restored = self.restore(zero, expected, initial)
        self.assertEqual(restored['optimizer_state']['state'], {})
        trained, _, _, _, _ = self.fixture()
        zero['optimizer_state']['state'] = trained['optimizer_state']['state']
        with self.assertRaisesRegex(ValueError, 'Adam parameter-state keys'):
            self.restore(zero, expected, initial)


if __name__ == '__main__':
    unittest.main()
