"""Recovery must preserve selection, not merely restore the latest adapter weights."""

import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from experiments.chronological import train_incident_strength_gate as e
from experiments.chronological.gate_recovery import recover, partial_report, memory_snapshot
from test_incident_strength_gate import SyntheticDataset, manifests, tiny_model


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.protocol = copy.deepcopy(e.load_protocol())
        self.protocol['training']['epochs'] = 3
        self.dataset = SyntheticDataset()
        self.datasets = dict.fromkeys(e.COHORTS, self.dataset)
        self.plan = e.make_plan(*manifests(), self.protocol)
        self.baseline = {c: e.metric_summary(e.evaluate(tiny_model(), self.dataset,
                         self.plan['selection']['indices'][c], 16, torch.device('cpu')))
                         for c in e.COHORTS}

    def fit(self, directory, resume=None, progress=lambda *a, **k: None):
        directory.mkdir()
        return e.fit_variant(tiny_model(), 'node', 2025, self.datasets, self.plan,
            self.baseline, self.protocol, torch.device('cpu'), directory, progress, resume)

    def test_interrupted_epoch_recovery_matches_uninterrupted_training_exactly(self):
        full, interrupted, resumed = (self.root / name for name in ('full', 'interrupted', 'resumed'))
        expected = self.fit(full)
        def stop(stage, **fields):
            if stage == 'epoch_complete' and fields['epoch'] == 1:
                raise RuntimeError('simulated platform interruption')
        with self.assertRaisesRegex(RuntimeError, 'platform interruption'):
            self.fit(interrupted, progress=stop)
        before = {p.name: e.sha256(p) for p in interrupted.iterdir()}
        actual = self.fit(resumed, resume=interrupted)
        self.assertEqual(actual['selected_epoch'], expected['selected_epoch'])
        self.assertEqual(actual['selection_metrics'], expected['selection_metrics'])
        self.assertEqual(actual['optimizer_steps'], expected['optimizer_steps'])
        self.assertEqual(actual['recovery']['epoch'], 1)
        self.assertEqual(json.loads((full / 'history.json').read_text()),
                         json.loads((resumed / 'history.json').read_text()))
        for filename in ('last_gate.pt', 'selected_gate.pt'):
            a = torch.load(full / filename, weights_only=True)
            b = torch.load(resumed / filename, weights_only=True)
            for key in a['gate_state']:
                torch.testing.assert_close(a['gate_state'][key], b['gate_state'][key], atol=0, rtol=0)
        self.assertEqual(before, {p.name: e.sha256(p) for p in interrupted.iterdir()})

    def test_completed_fit_recovery_does_not_run_optimizer(self):
        source, target = self.root / 'source', self.root / 'target'
        expected = self.fit(source)
        with patch.object(e, 'train_epoch') as train:
            actual = self.fit(target, resume=source)
            train.assert_not_called()
        self.assertEqual(actual['selected_epoch'], expected['selected_epoch'])

    def legacy(self, scores, saved_selected=False):
        directory = self.root / 'legacy'
        directory.mkdir()
        baseline = {c: {'mae': dict.fromkeys(e.REGIONS, 10.)} for c in e.COHORTS}
        history, best_epoch, best_score = [], 0, 10.
        for epoch, score in enumerate(scores, 1):
            metrics = {c: {'mae': dict.fromkeys(e.REGIONS, score)} for c in e.COHORTS}
            allowed, checks = e.selection_eligible(metrics, baseline, self.protocol)
            if allowed and score < best_score:
                best_epoch, best_score = epoch, score
            history.append({'epoch': epoch, 'selection': metrics, 'eligible': allowed,
                            'protection_checks': checks, 'best_epoch': best_epoch,
                            'training': {'optimizer_steps': 1}})
        state = {'variant': 'scalar', 'seed': 2025, 'epoch': len(scores),
                 'protocol_sha256': e.PROTOCOL_SHA256, 'backbone_state_sha256': 'backbone',
                 'gate_state': {'logit': torch.tensor(2.)}, 'optimizer_state': {}}
        torch.save(state, directory / 'last_gate.pt')
        e.write_json(directory / 'history.json', history)
        if saved_selected:
            torch.save({**state, 'epoch': best_epoch, 'gate_state': {'logit': torch.tensor(.5)}},
                       directory / 'selected_gate.pt')
        return directory, baseline

    def load_legacy(self, directory, baseline):
        return recover(directory, 'scalar', 2025, self.protocol, e.PROTOCOL_SHA256,
                       'backbone', baseline, {'logit': torch.tensor(0.)}, e.selection_eligible)

    def test_legacy_missing_historical_best_requires_restart(self):
        directory, baseline = self.legacy([9., 9.5])
        state = self.load_legacy(directory, baseline)
        self.assertTrue(state['restart_required'])
        self.assertEqual(state['best_epoch'], 1)

    def test_legacy_current_best_can_resume(self):
        directory, baseline = self.legacy([9.5, 9.])
        state = self.load_legacy(directory, baseline)
        self.assertFalse(state['restart_required'])
        self.assertEqual(state['epoch'], 2)
        self.assertEqual(state['best']['state']['logit'].item(), 2.)

    def test_legacy_identity_best_can_resume(self):
        directory, baseline = self.legacy([10.1, 10.2])
        state = self.load_legacy(directory, baseline)
        self.assertEqual(state['best']['epoch'], 0)
        self.assertEqual(state['best']['state']['logit'].item(), 0.)

    def test_completed_legacy_uses_selected_not_last_weights(self):
        directory, baseline = self.legacy([9., 9.5, 9.8], saved_selected=True)
        state = self.load_legacy(directory, baseline)
        self.assertEqual(state['method'], 'completed_legacy_fit')
        self.assertEqual(state['best']['epoch'], 1)
        self.assertEqual(state['best']['state']['logit'].item(), .5)

    def test_wrong_protocol_or_tampered_history_rejected(self):
        directory, baseline = self.legacy([9.])
        with self.assertRaisesRegex(ValueError, 'identity mismatch'):
            recover(directory, 'scalar', 2025, self.protocol, 'wrong', 'backbone',
                    baseline, {}, e.selection_eligible)
        history = json.loads((directory / 'history.json').read_text())
        history[0]['best_epoch'] = 0
        e.write_json(directory / 'history.json', history)
        with self.assertRaisesRegex(ValueError, 'selection rules'):
            self.load_legacy(directory, baseline)

    def test_atomic_bundle_history_wins_over_stale_json(self):
        source, target = self.root / 'source', self.root / 'target'
        self.fit(source)
        e.write_json(source / 'history.json', [])
        with patch.object(e, 'train_epoch') as train:
            actual = self.fit(target, resume=source)
        train.assert_not_called()
        self.assertEqual(actual['recovery']['epoch'], 3)

    def test_missing_summary_reports_partial_and_memory_has_rss(self):
        output = self.root / 'run'
        partial = self.root / 'run.partial'
        partial.mkdir()
        directory = partial / 'node_s2025'
        self.fit(directory)
        result = io.StringIO()
        with redirect_stdout(result):
            partial_report(output)
        self.assertIn('INCOMPLETE', result.getvalue())
        self.assertIn('last saved epoch= 3', result.getvalue())
        self.assertGreater(memory_snapshot(torch.device('cpu'))['VmRSS_KiB'], 0)

    def test_resume_pipeline_recomputes_audit_without_refitting(self):
        protocol = copy.deepcopy(self.protocol)
        protocol['seeds'] = [2025]
        protocol['bootstrap']['draws'] = 40
        source, target = self.root / 'source', self.root / 'target'
        model = tiny_model()
        checkpoint = self.root / 'A.pt'
        torch.save(model.state_dict(), checkpoint)
        baseline = {'checkpoint': {'parameters': sum(p.numel() for p in model.parameters())}}
        def run(output, resume=None):
            with patch.object(e, 'load_protocol', return_value=protocol), patch.object(
                    e.mechanisms, 'verify_inputs', return_value=(baseline, {})), patch.object(
                    e, 'read_csv', side_effect=list(manifests())), patch.object(
                    e, 'make_datasets', return_value=self.datasets), patch.object(
                    e, 'make_model', side_effect=lambda *a, **k: tiny_model()), redirect_stdout(io.StringIO()):
                return e.run(self.root, self.root, self.root, checkpoint, output, 'cpu', resume_from=resume)
        expected = run(source)
        before = {str(p.relative_to(source)): e.sha256(p) for p in source.rglob('*') if p.is_file()}
        with patch.object(e, 'train_epoch') as train:
            actual = run(target, source)
            train.assert_not_called()
        self.assertEqual(actual['audit_comparisons'], expected['audit_comparisons'])
        self.assertEqual(before, {str(p.relative_to(source)): e.sha256(p) for p in source.rglob('*') if p.is_file()})
        plan = json.loads((source / 'effective_plan.json').read_text())
        plan['fit']['indices']['incident_full'] = [99]
        e.write_json(source / 'effective_plan.json', plan)
        with self.assertRaisesRegex(ValueError, 'sample/period eligibility'):
            run(self.root / 'bad_plan', source)


if __name__ == '__main__':
    unittest.main()
