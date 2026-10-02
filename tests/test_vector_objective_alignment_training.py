"""Real tiny-network v12k workflow, delayed audit access and atomic two-selector recovery."""

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

from experiments.chronological import train_vector_objective_alignment as e
from test_incident_strength_gate import SyntheticDataset, manifests, tiny_model
from test_vector_objective_alignment import selection_metrics


class WorkflowTests(unittest.TestCase):
    def setUp(self):
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
        # Four calendar weeks in each tiny phase, preserving eight noncontiguous IDs.
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
        self.frozen['expected_phase_samples'] = {p: {c: len(v) for c, v in x['indices'].items()} for p, x in self.plan.items()}
        model = tiny_model()
        self.checkpoint = self.root / 'A.pt'
        torch.save(model.state_dict(), self.checkpoint)
        self.baseline = {'checkpoint': {'parameters': sum(p.numel() for p in model.parameters())}}

    def pipeline(self, output, resume=None, check=False):
        partial = output.with_name(output.name + '.partial')
        class Guarded:
            scaler = self.ds.scaler
            station_ids = self.ds.station_ids
            def __getitem__(inner, index):
                if index >= 2 and torch.is_grad_enabled():
                    raise AssertionError('Selection/audit target entered optimization')
                if index >= 4 and not (partial / 'selected_endpoints_frozen.json').is_file():
                    raise AssertionError('Audit target read before all selectors were frozen')
                return self.ds[index]
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

    def test_complete_workflow_and_recovery_without_any_refit(self):
        original = self.root / 'original'
        s = self.pipeline(original)
        self.assertEqual(s['status'], 'VECTOR_OBJECTIVE_ALIGNMENT_COMPARISON_COMPLETE')
        self.assertTrue(s['all_selectors_frozen_before_audit_evaluation'])
        self.assertTrue(s['paired_initialization_exact'])
        self.assertEqual(len(s['runs']['2025']), 4)
        initial = {d['initial_adapter_sha256'] for d in s['runs']['2025'].values()}
        self.assertEqual(len(initial), 1)
        self.assertEqual(set(s['phase_comparisons']['2025']), {'fit', 'selection', 'audit'})
        for name, d in s['runs']['2025'].items():
            self.assertEqual(d['epochs'], 2)
            self.assertEqual(d['optimizer_steps'], 2)
            self.assertEqual(set(d['selectors']), {'global', 'candidate_early'})
            for selector in d['selectors']:
                for phase in ('fit', 'selection', 'audit'):
                    path = original / name / ('select_' + selector) / (phase + '_incident_full.npz')
                    with np.load(path) as records:
                        expected = self.plan[phase]['positive_ids']
                        np.testing.assert_array_equal(records['ids'], expected)
        for name, digest in s['outputs'].items():
            self.assertEqual(e.base.sha256(original / name), digest)
        before = {str(p): e.base.sha256(p) for p in original.rglob('*') if p.is_file()}
        with patch.object(e.vector, 'vector_train_epoch', side_effect=AssertionError('global refit')), patch.object(
                e.alignment, 'early_train_epoch', side_effect=AssertionError('early refit')):
            recovered = self.pipeline(self.root / 'recovered', original)
        self.assertEqual(s['phase_comparisons'], recovered['phase_comparisons'])
        self.assertEqual(before, {str(p): e.base.sha256(p) for p in original.rglob('*') if p.is_file()})

    def test_check_is_engineering_only_and_preserves_all_four_trajectories(self):
        summary = self.pipeline(self.root / 'check', check=True)
        self.assertEqual(summary['status'], 'ENGINEERING_CHECK_PASS')
        self.assertEqual(summary['recommendation'], 'ENGINEERING_ONLY')
        self.assertEqual(summary['phase_comparisons'], {'2025': {}})
        self.assertEqual(len(summary['runs']['2025']), 4)

    def fit_direct(self, directory, arm='state_vector', loss='global', resume=None, progress=lambda *a, **k: None, baseline=None):
        directory.mkdir()
        baseline = baseline or {c: e.metrics(e.evaluate(tiny_model(), self.ds, [2, 3], 16, torch.device('cpu'))) for c in e.base.COHORTS}
        return e.fit(tiny_model(), arm, loss, 2025, dict.fromkeys(e.base.COHORTS, self.ds),
            self.plan, baseline, self.protocol, torch.device('cpu'), directory, progress, 'test_run_identity', resume)

    def test_interrupted_recovery_preserves_both_selectors_and_exact_training(self):
        for loss in e.alignment.LOSSES:
            full, stopped, resumed = [self.root / (loss + '_' + x) for x in ('full', 'stopped', 'resumed')]
            self.fit_direct(full, loss=loss)
            def interrupt(stage, **fields):
                if stage == 'epoch_complete' and fields['epoch'] == 1:
                    raise RuntimeError('interrupted')
            with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                self.fit_direct(stopped, loss=loss, progress=interrupt)
            before = {p.name: e.base.sha256(p) for p in stopped.iterdir()}
            self.fit_direct(resumed, loss=loss, resume=stopped)
            self.assertEqual((full / 'history.json').read_text(), (resumed / 'history.json').read_text())
            for filename in ('last_adapter.pt', 'selected_global.pt', 'selected_candidate_early.pt'):
                left = torch.load(full / filename, weights_only=True)['adapter_state']
                right = torch.load(resumed / filename, weights_only=True)['adapter_state']
                self.assertEqual(e.alignment.tensor_hash(left), e.alignment.tensor_hash(right))
            self.assertEqual(before, {p.name: e.base.sha256(p) for p in stopped.iterdir()})

    def test_epoch_zero_is_saved_for_both_selectors_when_no_global_improvement(self):
        baseline = {c: {'mae': dict.fromkeys(e.base.REGIONS, 0.)} for c in e.base.COHORTS}
        detail = self.fit_direct(self.root / 'fallback', baseline=baseline)
        self.assertEqual([x['selected_epoch'] for x in detail['selectors'].values()], [0, 0])
        for selector in e.alignment.SELECTORS:
            self.assertEqual(detail['selectors'][selector]['adapter_state_sha256'], detail['initial_adapter_sha256'])

    def test_global_trajectory_exactly_matches_original_vector_fit_and_selection(self):
        current, legacy = self.root / 'current', self.root / 'legacy'
        current_detail = self.fit_direct(current)
        legacy.mkdir()
        baseline = {c: e.metrics(e.evaluate(tiny_model(), self.ds, [2, 3], 16, torch.device('cpu'))) for c in e.base.COHORTS}
        old = e.base.fit_variant(tiny_model(), 'state_vector', 2025,
            dict.fromkeys(e.base.COHORTS, self.ds), self.plan, baseline, self.protocol,
            torch.device('cpu'), legacy, lambda *a, **k: None,
            epoch_trainer=e.vector.vector_train_epoch, adapter_factory=e.vector.attach_adapter,
            evaluator=e.evaluate, summarizer=e.metrics)
        self.assertEqual(current_detail['selectors']['global']['selected_epoch'], old['selected_epoch'])
        for new_file, old_file in (('last_adapter.pt', 'last_gate.pt'), ('selected_global.pt', 'selected_gate.pt')):
            new = torch.load(current / new_file, weights_only=True)['adapter_state']
            original = torch.load(legacy / old_file, weights_only=True)['gate_state']
            self.assertEqual(e.alignment.tensor_hash(new), e.alignment.tensor_hash(original))

    def test_two_selectors_save_different_historical_states_and_recover_them(self):
        source = self.root / 'distinct'
        baseline = selection_metrics()
        first, second = selection_metrics(9.8, 9.9), selection_metrics(9.9, 9.5)
        # Prescribe selection outcomes only; the tiny network still performs real optimizer steps.
        scripted = [m[c] for m in (first, second) for c in e.base.COHORTS]
        with patch.object(e, 'metrics', side_effect=scripted):
            detail = self.fit_direct(source, baseline=baseline)
        self.assertEqual([detail['selectors'][s]['selected_epoch'] for s in e.alignment.SELECTORS], [1, 2])
        history = json.loads((source / 'history.json').read_text())
        for selector, epoch in (('global', 1), ('candidate_early', 2)):
            saved = torch.load(source / f'selected_{selector}.pt', weights_only=True)
            self.assertEqual(e.alignment.tensor_hash(saved['adapter_state']), history[epoch - 1]['adapter_state_sha256'])
        with patch.object(e.vector, 'vector_train_epoch', side_effect=AssertionError('unnecessary refit')):
            restored = self.fit_direct(self.root / 'distinct_resumed', baseline=baseline, resume=source)
        # Re-serialization can change storage aliasing/bytes without changing any tensor.
        for selector in e.alignment.SELECTORS:
            for key in ('selected_epoch', 'selection_metrics', 'adapter_state_sha256'):
                self.assertEqual(detail['selectors'][selector][key], restored['selectors'][selector][key])
            saved = torch.load(self.root / 'distinct_resumed' / f'selected_{selector}.pt', weights_only=True)
            self.assertEqual(e.alignment.tensor_hash(saved['adapter_state']), detail['selectors'][selector]['adapter_state_sha256'])

    def test_cross_loss_recovery_and_corrupt_best_are_rejected(self):
        source = self.root / 'source'
        self.fit_direct(source)
        with self.assertRaisesRegex(ValueError, 'identity mismatch'):
            self.fit_direct(self.root / 'wrong_loss', loss='candidate_early', resume=source)
        saved = torch.load(source / 'last_adapter.pt', weights_only=True)
        next(iter(saved['best']['candidate_early']['state'].values())).add_(1.)
        torch.save(saved, source / 'last_adapter.pt')
        with self.assertRaisesRegex(ValueError, 'best state hash'):
            self.fit_direct(self.root / 'corrupt', resume=source)

    def test_output_symlinks_and_changed_run_identity_are_rejected(self):
        target = self.root / 'dangling'
        target.symlink_to(self.root / 'absent', target_is_directory=True)
        with self.assertRaises(FileExistsError):
            self.pipeline(target)
        source = self.root / 'source'
        source.mkdir()
        (source / 'run_identity.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'run input/code/protocol/sample identity'):
            self.pipeline(self.root / 'wrong', source)
        self.assertTrue((self.root / 'wrong.partial/failure.json').is_file())


if __name__ == '__main__':
    unittest.main()
