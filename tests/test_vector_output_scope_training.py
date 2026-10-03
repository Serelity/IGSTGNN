"""Real v12m early-loss trajectories, independent policies and exact recovery."""

from contextlib import ExitStack, redirect_stdout
import copy
from datetime import datetime, timedelta
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.chronological import train_vector_output_scope as e
from experiments.chronological import train_vector_objective_alignment as legacy
from test_incident_strength_gate import SyntheticDataset, manifests, tiny_model
from test_vector_objective_alignment import selection_metrics


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ds = SyntheticDataset()
        self.protocol = copy.deepcopy(e.base.load_protocol())
        self.protocol['seeds'] = [2025]
        self.protocol['training']['epochs'] = 2
        self.protocol['bootstrap']['draws'] = 40
        self.frozen = e.load_protocol()
        self.manifests = manifests()
        # Four calendar weeks per tiny phase retain the noncontiguous sample IDs.
        for index, day in ((1, '2023-02-07'), (3, '2023-06-13')):
            t0 = datetime.fromisoformat(day + 'T12:00:00')
            for rows in self.manifests:
                row = rows[index]
                for key in ('t0', 'positive_t0', 'candidate_t0'):
                    if key in row:
                        row[key] = t0.isoformat()
                row['support_start'] = (t0 - timedelta(hours=1)).isoformat()
                row['support_end_exclusive'] = (t0 + timedelta(hours=2)).isoformat()
        self.plan = e.base.make_plan(*self.manifests, self.protocol)
        self.frozen['expected_phase_samples'] = {
            phase: {cohort: len(indices) for cohort, indices in period['indices'].items()}
            for phase, period in self.plan.items()}
        model = tiny_model()
        self.checkpoint = self.root / 'A.pt'
        torch.save(model.state_dict(), self.checkpoint)
        self.baseline = {'checkpoint': {'parameters': sum(p.numel() for p in model.parameters())}}

    def pipeline(self, output, resume=None, check=False):
        partial = output.with_name(output.name + '.partial')
        dataset = self.ds

        class Guarded:
            scaler = dataset.scaler
            station_ids = dataset.station_ids

            def __getitem__(inner, index):
                if index >= 2 and torch.is_grad_enabled():
                    raise AssertionError('Selection/audit target entered optimization')
                if index >= 4 and not all((partial / name).is_file() for name in
                                         ('selected_endpoints_frozen.json', 'evaluation_paths_frozen.json')):
                    raise AssertionError('Audit target read before all policies and output paths were frozen')
                return dataset[index]

        class Controls(Guarded):
            def __getitem__(inner, index):
                if torch.is_grad_enabled():
                    raise AssertionError('Control Y entered optimization')
                return super().__getitem__(index)

        datasets = {'incident_full': Guarded(), **{c: Controls() for c in e.base.COHORTS[1:]}}
        with ExitStack() as stack:
            for owner, name, kwargs in (
                (e, 'load_protocol', {'return_value': self.frozen}),
                (e.base, 'load_protocol', {'return_value': self.protocol}),
                (e.base.mechanisms, 'verify_inputs', {'return_value': (self.baseline, {})}),
                (e.base, 'read_csv', {'side_effect': self.manifests}),
                (e.base, 'make_datasets', {'return_value': datasets}),
                (e.base, 'make_model', {'side_effect': lambda *a, **k: tiny_model()})):
                stack.enter_context(patch.object(owner, name, **kwargs))
            stack.enter_context(redirect_stdout(io.StringIO()))
            return e.run(self.root, self.root, self.root, self.checkpoint, output, 'cpu', check, resume)

    def selection_baseline(self):
        return {cohort: e.metrics(e.evaluate(tiny_model(), self.ds, [2, 3], 16, torch.device('cpu')))
                for cohort in e.base.COHORTS}

    def fit_direct(self, directory, arm='state_vector', resume=None,
                   progress=lambda *a, **k: None, baseline=None, identity='test_run_identity'):
        directory.mkdir()
        if baseline is None:
            baseline = self.selection_baseline()
        return e.fit(tiny_model(), arm, 2025, dict.fromkeys(e.base.COHORTS, self.ds),
                     self.plan, baseline, self.protocol, torch.device('cpu'), directory,
                     progress, identity, resume)

    def test_complete_workflow_has_one_fit_per_arm_and_completed_recovery_does_not_refit(self):
        original = self.root / 'original'
        with patch.object(e.alignment, 'early_train_epoch', wraps=e.alignment.early_train_epoch) as trainer:
            summary = self.pipeline(original)
        self.assertEqual(summary['status'], 'VECTOR_OUTPUT_SCOPE_COMPARISON_COMPLETE')
        self.assertTrue(summary['all_selectors_frozen_before_audit_evaluation'])
        self.assertTrue(summary['all_output_paths_frozen_before_audit_evaluation'])
        self.assertTrue(summary['paired_initialization_exact'])
        self.assertFalse(summary['unrestricted_at_protected_evaluated'])
        self.assertEqual(summary['budget'], {'fits': 2, 'trajectory_epochs': 4,
                                            'selected_endpoints': 4, 'output_paths': 6})
        self.assertEqual(len(summary['runs']['2025']), 2)
        self.assertEqual(trainer.call_count, 4)  # Two epochs, two arms; never per output policy.
        self.assertEqual(len({d['initial_adapter_sha256'] for d in summary['runs']['2025'].values()}), 1)
        self.assertEqual(set(summary['phase_comparisons']['2025']), {'fit', 'selection', 'audit'})
        endpoints = json.loads((original / 'selected_endpoints_frozen.json').read_text())
        self.assertEqual(len(endpoints), 4)
        output_paths = json.loads((original / 'evaluation_paths_frozen.json').read_text())
        self.assertEqual(len(output_paths), 6)
        self.assertEqual(endpoints, summary['selected_endpoints_frozen'])
        self.assertEqual(output_paths, summary['evaluation_paths_frozen'])
        for name, detail in summary['runs']['2025'].items():
            self.assertEqual(detail['epochs'], 2)
            self.assertEqual(detail['optimizer_steps'], 2)
            self.assertEqual(set(detail['policies']), set(e.scope.POLICIES))
            self.assertTrue(detail['initial_prediction_exactly_A'])
            self.assertTrue(detail['backbone_state_unchanged'])
            self.assertEqual(set(detail['output_paths']), set(e.scope.OUTPUT_PATHS))
            for output_path, selected_policy, output_policy in (
                    ('unrestricted_at_unrestricted', 'unrestricted', 'unrestricted'),
                    ('protected_at_unrestricted', 'unrestricted', 'candidate_early_only'),
                    ('protected_at_protected', 'candidate_early_only', 'candidate_early_only')):
                frozen_path = output_paths[f'{name}/{output_path}']
                checkpoint = f'{name}/selected_{selected_policy}.pt'
                self.assertEqual(frozen_path['source_checkpoint'], checkpoint)
                self.assertEqual(frozen_path['source_checkpoint_sha256'], endpoints[checkpoint])
                self.assertEqual(frozen_path['selected_policy'], selected_policy)
                self.assertEqual(frozen_path['output_policy'], output_policy)
                self.assertEqual(frozen_path['epoch'], detail['policies'][selected_policy]['selected_epoch'])
                self.assertEqual(frozen_path['adapter_state_sha256'],
                                 detail['policies'][selected_policy]['adapter_state_sha256'])
            history = json.loads((original / name / 'history.json').read_text())
            self.assertTrue(all(row['training']['loss_region'] == 'candidate_early' for row in history))
            self.assertTrue(all(set(row['selection']) == set(e.scope.POLICIES) for row in history))
            for phase in ('fit', 'selection', 'audit'):
                paths = list((original / name).rglob(phase + '_incident_full.npz'))
                self.assertEqual(len(paths), 3)
                for path in paths:
                    with np.load(path) as records:
                        np.testing.assert_array_equal(records['ids'], self.plan[phase]['positive_ids'])
                for cohort in self.plan[phase]['indices']:
                    with np.load(original / 'A' / f'{phase}_{cohort}.npz') as reference:
                        records = {}
                        for output_path in e.scope.OUTPUT_PATHS:
                            with np.load(original / name / output_path / f'{phase}_{cohort}.npz') as archive:
                                records[output_path] = {key: archive[key] for key in archive.files}
                        for record in records.values():
                            for field in ('ids', 'source_indices', 'counts', 'prediction_counts', 'candidate_mask'):
                                np.testing.assert_array_equal(record[field], reference[field])
                        for region in ('candidate_h7_h12', 'noncandidate_all'):
                            column = list(reference['regions']).index(region)
                            for output_path in ('protected_at_unrestricted', 'protected_at_protected'):
                                np.testing.assert_array_equal(records[output_path]['errors'][:, column],
                                                              reference['errors'][:, column])
                        early = list(reference['regions']).index('candidate_h1_h6')
                        np.testing.assert_array_equal(records['unrestricted_at_unrestricted']['errors'][:, early],
                                                      records['protected_at_unrestricted']['errors'][:, early])
        for name, digest in summary['outputs'].items():
            self.assertEqual(e.base.sha256(original / name), digest)
        before = {str(path): e.base.sha256(path) for path in original.rglob('*') if path.is_file()}
        with patch.object(e.alignment, 'early_train_epoch', side_effect=AssertionError('unnecessary refit')):
            recovered = self.pipeline(self.root / 'recovered', original)
        self.assertEqual(summary['phase_comparisons'], recovered['phase_comparisons'])
        self.assertEqual(before, {str(path): e.base.sha256(path) for path in original.rglob('*') if path.is_file()})

    def test_check_is_engineering_only_and_preserves_both_shared_trajectories(self):
        summary = self.pipeline(self.root / 'check', check=True)
        self.assertEqual(summary['status'], 'ENGINEERING_CHECK_PASS')
        self.assertEqual(summary['recommendation'], 'ENGINEERING_ONLY')
        self.assertEqual(summary['phase_comparisons'], {'2025': {}})
        self.assertEqual(len(summary['runs']['2025']), 2)

    def test_evaluator_uses_exact_native_report_path_and_only_requested_error_outputs(self):
        native = e.evaluate(tiny_model(), self.ds, [2, 3], 16, torch.device('cpu'))
        model = tiny_model()
        original_hash = e.base.backbone_hash(model)
        adapter = e.vector.attach_adapter(model, 'state_vector')
        wrapper = model.icsf_module
        with torch.no_grad():
            adapter.output.weight.fill_(.08)
            adapter.output.bias.fill_(.02)
        record = e.evaluate_policies(model, self.ds, [2, 3], 16, torch.device('cpu'))
        self.assertIs(model.icsf_module, wrapper)
        self.assertEqual(set(record), set(e.scope.POLICIES))
        self.assertFalse(np.array_equal(record['unrestricted']['errors'], native['errors']))
        for region in ('candidate_h7_h12', 'noncandidate_all'):
            column = list(native['regions']).index(region)
            np.testing.assert_array_equal(record['candidate_early_only']['errors'][:, column],
                                          native['errors'][:, column])
        early = list(native['regions']).index('candidate_h1_h6')
        np.testing.assert_array_equal(record['candidate_early_only']['errors'][:, early],
                                      record['unrestricted']['errors'][:, early])
        e.base.assert_backbone(model, original_hash)
        with patch.object(e.base, 'statistics', wraps=e.base.statistics) as summarize:
            protected = e.evaluate_policies(model, self.ds, [2, 3], 16, torch.device('cpu'),
                                           policies=('candidate_early_only',))
        self.assertEqual(set(protected), {'candidate_early_only'})
        self.assertEqual(summarize.call_count, 1)  # Never compute unrestricted target errors at P.

    def test_native_forward_failure_restores_the_live_vector_module(self):
        model = tiny_model()
        e.vector.attach_adapter(model, 'interaction_vector')
        wrapper = model.icsf_module
        with patch.object(wrapper.base, 'forward', side_effect=RuntimeError('native forward interrupted')):
            with self.assertRaisesRegex(RuntimeError, 'native forward interrupted'):
                e.evaluate_policies(model, self.ds, [2, 3], 16, torch.device('cpu'))
        self.assertIs(model.icsf_module, wrapper)

    def test_interrupted_recovery_preserves_both_policies_and_exact_training(self):
        for arm in e.alignment.ARMS:
            full, stopped, resumed = [self.root / (arm + '_' + part) for part in ('full', 'stopped', 'resumed')]
            self.fit_direct(full, arm=arm)

            def interrupt(stage, **fields):
                if stage == 'epoch_complete' and fields['epoch'] == 1:
                    raise RuntimeError('interrupted')

            with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                self.fit_direct(stopped, arm=arm, progress=interrupt)
            before = {path.name: e.base.sha256(path) for path in stopped.iterdir()}
            self.fit_direct(resumed, arm=arm, resume=stopped)
            self.assertEqual((full / 'history.json').read_text(), (resumed / 'history.json').read_text())
            for filename in ('last_adapter.pt', *[f'selected_{policy}.pt' for policy in e.scope.POLICIES]):
                left = torch.load(full / filename, weights_only=True)['adapter_state']
                right = torch.load(resumed / filename, weights_only=True)['adapter_state']
                self.assertEqual(e.alignment.tensor_hash(left), e.alignment.tensor_hash(right))
            self.assertEqual(before, {path.name: e.base.sha256(path) for path in stopped.iterdir()})

    def test_epoch_zero_saved_for_both_policies_when_no_global_improvement(self):
        baseline = {cohort: {'mae': dict.fromkeys(e.base.REGIONS, 0.)} for cohort in e.base.COHORTS}
        directory = self.root / 'fallback'
        detail = self.fit_direct(directory, baseline=baseline)
        self.assertEqual([item['selected_epoch'] for item in detail['policies'].values()], [0, 0])
        for policy in e.scope.POLICIES:
            self.assertEqual(detail['policies'][policy]['adapter_state_sha256'], detail['initial_adapter_sha256'])
            saved = torch.load(directory / f'selected_{policy}.pt', weights_only=True)
            self.assertEqual(saved['epoch'], 0)
            self.assertEqual(e.alignment.tensor_hash(saved['adapter_state']), detail['initial_adapter_sha256'])

    def test_recovery_uses_atomic_checkpoint_when_history_publication_is_interrupted(self):
        full, stopped, resumed = [self.root / part for part in ('full', 'stopped', 'resumed')]
        self.fit_direct(full)
        write_json = e.base.write_json

        def interrupt_history(path, value):
            if Path(path) == stopped / 'history.json' and value:
                raise RuntimeError('history publication interrupted')
            return write_json(path, value)

        with patch.object(e.base, 'write_json', side_effect=interrupt_history):
            with self.assertRaisesRegex(RuntimeError, 'history publication interrupted'):
                self.fit_direct(stopped)
        self.assertEqual(json.loads((stopped / 'history.json').read_text()), [])
        self.assertEqual(torch.load(stopped / 'last_adapter.pt', weights_only=True)['epoch'], 1)
        self.fit_direct(resumed, resume=stopped)
        self.assertEqual((full / 'history.json').read_text(), (resumed / 'history.json').read_text())
        for policy in e.scope.POLICIES:
            left = torch.load(full / f'selected_{policy}.pt', weights_only=True)['adapter_state']
            right = torch.load(resumed / f'selected_{policy}.pt', weights_only=True)['adapter_state']
            self.assertEqual(e.alignment.tensor_hash(left), e.alignment.tensor_hash(right))

    def test_shared_trajectory_exactly_matches_original_early_training(self):
        current, old = self.root / 'current', self.root / 'legacy'
        current_detail = self.fit_direct(current)
        old.mkdir()
        original = legacy.fit(tiny_model(), 'state_vector', 'candidate_early', 2025,
            dict.fromkeys(e.base.COHORTS, self.ds), self.plan, self.selection_baseline(), self.protocol,
            torch.device('cpu'), old, lambda *a, **k: None, 'legacy_test_identity')
        self.assertEqual(current_detail['policies']['unrestricted']['selected_epoch'],
                         original['selectors']['candidate_early']['selected_epoch'])
        for new_file, old_file in (('last_adapter.pt', 'last_adapter.pt'),
                                  ('selected_unrestricted.pt', 'selected_candidate_early.pt')):
            new = torch.load(current / new_file, weights_only=True)['adapter_state']
            previous = torch.load(old / old_file, weights_only=True)['adapter_state']
            self.assertEqual(e.alignment.tensor_hash(new), e.alignment.tensor_hash(previous))

    def test_two_policies_save_distinct_historical_states_and_recover_them(self):
        source = self.root / 'distinct'
        baseline = selection_metrics()
        first = selection_metrics(9.8, 9.9)
        second_unrestricted = selection_metrics(9.9, 9.5)
        second_unrestricted['incident']['mae']['candidate_h7_h12'] = 10.1
        second_protected = selection_metrics(9.9, 9.5)
        # Only selection outcomes are prescribed; real Torch optimizers still update both epochs.
        scripted = [current[cohort]
                    for epoch in ((first, first), (second_unrestricted, second_protected))
                    for cohort in e.base.COHORTS for current in epoch]
        with patch.object(e, 'metrics', side_effect=scripted):
            detail = self.fit_direct(source, baseline=baseline)
        self.assertEqual([detail['policies'][policy]['selected_epoch'] for policy in e.scope.POLICIES], [1, 2])
        history = json.loads((source / 'history.json').read_text())
        for policy, epoch in zip(e.scope.POLICIES, (1, 2)):
            saved = torch.load(source / f'selected_{policy}.pt', weights_only=True)
            self.assertEqual(e.alignment.tensor_hash(saved['adapter_state']), history[epoch - 1]['adapter_state_sha256'])
        with patch.object(e.alignment, 'early_train_epoch', side_effect=AssertionError('unnecessary refit')):
            restored = self.fit_direct(self.root / 'distinct_resumed', baseline=baseline, resume=source)
        for policy in e.scope.POLICIES:
            for key in ('selected_epoch', 'selection_metrics', 'adapter_state_sha256'):
                self.assertEqual(detail['policies'][policy][key], restored['policies'][policy][key])

    def test_wrong_arm_run_identity_and_corrupt_best_are_rejected(self):
        source = self.root / 'source'
        self.fit_direct(source)
        for name, kwargs in (('wrong_arm', {'arm': 'interaction_vector'}),
                             ('wrong_identity', {'identity': 'different_source'})):
            with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                self.fit_direct(self.root / name, resume=source, **kwargs)
        saved = torch.load(source / 'last_adapter.pt', weights_only=True)
        next(iter(saved['best']['candidate_early_only']['state'].values())).add_(1.)
        torch.save(saved, source / 'last_adapter.pt')
        with self.assertRaisesRegex(ValueError, 'best state hash'):
            self.fit_direct(self.root / 'corrupt', resume=source)

    def test_old_v12k_recovery_is_rejected(self):
        source = self.root / 'v12k_source'
        source.mkdir()
        legacy.fit(tiny_model(), 'state_vector', 'candidate_early', 2025,
            dict.fromkeys(e.base.COHORTS, self.ds), self.plan, self.selection_baseline(), self.protocol,
            torch.device('cpu'), source, lambda *a, **k: None, 'test_run_identity')
        with self.assertRaisesRegex(ValueError, 'identity mismatch'):
            self.fit_direct(self.root / 'reject_v12k', resume=source)

    def test_changed_protocol_or_missing_policy_in_recovery_is_rejected(self):
        source = self.root / 'source'
        self.fit_direct(source)
        saved = torch.load(source / 'last_adapter.pt', weights_only=True)
        for name, key, value in (('wrong_protocol', 'protocol_sha256', legacy.PROTOCOL_SHA256),
                                 ('wrong_scope', 'output_policies', ['unrestricted']),
                                 ('wrong_loss', 'loss', 'global')):
            changed = copy.deepcopy(saved)
            changed['identity'][key] = value
            torch.save(changed, source / 'last_adapter.pt')
            with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                self.fit_direct(self.root / name, resume=source)
        wrong_history = copy.deepcopy(saved)
        wrong_history['history'][0]['training']['loss_region'] = 'global'
        torch.save(wrong_history, source / 'last_adapter.pt')
        with self.assertRaisesRegex(ValueError, 'step budget/loss'):
            self.fit_direct(self.root / 'wrong_history_loss', resume=source)
        del saved['best']['candidate_early_only']
        torch.save(saved, source / 'last_adapter.pt')
        with self.assertRaisesRegex(ValueError, 'both|polic'):
            self.fit_direct(self.root / 'missing_policy', resume=source)

    def test_output_symlinks_and_changed_run_identity_are_rejected(self):
        for name in ('final', 'partial'):
            output = self.root / ('dangling_' + name)
            target = output if name == 'final' else output.with_name(output.name + '.partial')
            target.symlink_to(self.root / 'absent', target_is_directory=True)
            with self.assertRaises(FileExistsError):
                self.pipeline(output)
        source = self.root / 'source'
        source.mkdir()
        (source / 'run_identity.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'run input/code/protocol/sample identity'):
            self.pipeline(self.root / 'wrong', source)
        self.assertTrue((self.root / 'wrong.partial/failure.json').is_file())


if __name__ == '__main__':
    unittest.main()
