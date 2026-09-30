"""Fit/evaluate information boundaries, paired comparisons and recovery for v12f."""

from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.chronological import train_incident_state_interaction as e
from test_incident_strength_gate import SyntheticDataset, manifests, tiny_model


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
        model = tiny_model()
        self.checkpoint = self.root / 'A.pt'
        torch.save(model.state_dict(), self.checkpoint)
        self.baseline = {'checkpoint': {'parameters': sum(p.numel() for p in model.parameters())}}

    def run_pipeline(self, output, resume=None, check=False):
        # Audit/selection Y may only be evaluated under no_grad, never optimized.
        class GuardedDataset:
            scaler = self.ds.scaler
            station_ids = self.ds.station_ids
            def __getitem__(inner, index):
                if index >= 2 and torch.is_grad_enabled():
                    raise AssertionError('Non-fit target accessed with gradients enabled')
                return self.ds[index]
        class Controls(GuardedDataset):
            def __getitem__(inner, index):
                if torch.is_grad_enabled():
                    raise AssertionError('Control Y accessed with gradients enabled')
                return self.ds[index]
        datasets = {'incident_full': GuardedDataset(), **{c: Controls() for c in e.base.COHORTS[1:]}}
        with patch.object(e.base, 'load_protocol', return_value=self.protocol), patch.object(
                e.base.mechanisms, 'verify_inputs', return_value=(self.baseline, {})), patch.object(
                e.base, 'read_csv', side_effect=list(manifests())), patch.object(
                e.base, 'make_datasets', return_value=datasets), patch.object(
                e.base, 'make_model', side_effect=lambda *a, **k: tiny_model()), redirect_stdout(io.StringIO()):
            return e.run(self.root, self.root, self.root, self.checkpoint, output,
                         'cpu', check=check, resume_from=resume)

    def test_protocol_boundaries_and_capacity_comparison_are_frozen(self):
        p = e.load_protocol()
        self.assertEqual(tuple(p['arms']), e.ARMS)
        self.assertFalse(p['information_boundary']['test_split_read'])
        self.assertFalse(p['information_boundary']['high_impact_router_used'])
        self.assertFalse(p['information_boundary']['audit_Y_used_for_selection'])
        self.assertEqual(self.protocol['training']['gate_identity_penalty'], .001)
        self.assertEqual(self.protocol['selection']['maximum_relative_harm'], .001)

    def test_observations_distinguish_vector_from_scalar_and_zero_value_is_undefined(self):
        model = tiny_model()
        e.attach_adapter(model, 'state_vector')
        with torch.no_grad():
            model.icsf_module.base.v_proj.weight.zero_()
        record = e.evaluate(model, self.ds, [0, 1], 2, torch.device('cpu'))
        result = e.metric_summary(record)
        self.assertIsNone(result['candidate_gate'])
        self.assertNotIn('gates', record)
        obs = result['candidate_representation']
        self.assertEqual(obs['relative_correction_norm']['defined_nodes'], 0)
        self.assertEqual(obs['relative_correction_norm']['undefined_nodes'], 4)
        self.assertIsNone(obs['relative_perpendicular_norm']['mean'])
        self.assertEqual(obs['zero_injection_candidate_nodes'], 4)
        self.assertEqual(obs['correction_rms']['mean'], 0.)
        self.assertIsNone(model.icsf_module.last_unit_residual)
        # The JSON summary must not contain NaN even though NPZ angles can be undefined.
        json.dumps(result, allow_nan=False)

    def test_full_three_arm_workflow_and_completed_recovery_without_refitting(self):
        original = self.root / 'original'
        summary = self.run_pipeline(original)
        self.assertEqual(summary['status'], 'INCIDENT_STATE_INTERACTION_COMPARISON_COMPLETE')
        self.assertTrue(summary['vector_paired_initialization_exact'])
        arms = summary['runs']['2025']
        self.assertEqual(set(arms), set(e.ARMS))
        for arm, detail in arms.items():
            self.assertTrue(detail['initial_prediction_exactly_A'])
            self.assertTrue(detail['backbone_state_unchanged'])
            self.assertEqual(set(detail['representation_diagnostics']), {'initial', 'last', 'selected'})
            self.assertEqual(detail['representation_diagnostics']['initial']['sample_ids'], self.ds.ids[:2])
            self.assertEqual(detail['representation_diagnostics']['initial']['candidate_representation']['correction_rms']['mean'], 0.)
            if arm != 'strength':
                self.assertIsNone(detail['audit']['incident_full']['candidate_gate'])
        self.assertEqual(arms['state_vector']['representation_diagnostics']['initial']['adapter_state_sha256'],
                         arms['interaction_vector']['representation_diagnostics']['initial']['adapter_state_sha256'])
        contrasts = summary['audit_comparisons']['2025']['results']['incident_full']['regions']['all']['comparisons']
        self.assertIn('interaction_vector_vs_state_vector', contrasts)
        self.assertEqual(len(contrasts), 6)
        for name, digest in summary['outputs'].items():
            self.assertEqual(e.base.sha256(original / name), digest)
        before = {str(p): e.base.sha256(p) for p in original.rglob('*') if p.is_file()}
        with patch.object(e.base, 'train_epoch', side_effect=AssertionError('strength refit')), patch.object(
                e, 'vector_train_epoch', side_effect=AssertionError('vector refit')):
            recovered = self.run_pipeline(self.root / 'recovered', original)
        self.assertEqual(summary['audit_comparisons'], recovered['audit_comparisons'])
        self.assertEqual(before, {str(p): e.base.sha256(p) for p in original.rglob('*') if p.is_file()})
        with self.assertRaises(FileExistsError):
            self.run_pipeline(original)
        identity = json.loads((original / 'run_identity.json').read_text())
        identity['probe_indices'] = [999]
        (original / 'run_identity.json').write_text(json.dumps(identity))
        with self.assertRaisesRegex(ValueError, 'identity changed'):
            self.run_pipeline(self.root / 'wrong_identity', original)

    def test_engineering_check_has_all_arms_but_no_scientific_comparison(self):
        summary = self.run_pipeline(self.root / 'check', check=True)
        self.assertEqual(summary['status'], 'ENGINEERING_CHECK_PASS')
        self.assertEqual(summary['recommendation'], 'ENGINEERING_ONLY')
        self.assertEqual(summary['audit_comparisons'], {})
        self.assertEqual(set(summary['runs']['2025']), set(e.ARMS))
        self.assertEqual(summary['phase_samples']['audit']['incident_full'], 2)

    def fit(self, directory, arm='interaction_vector', resume=None, progress=lambda *a, **k: None,
            diagnostic=None):
        directory.mkdir()
        protocol = copy.deepcopy(self.protocol)
        protocol['training']['adapter_arm'] = arm
        plan = e.base.make_plan(*manifests(), protocol)
        baseline = {c: e.metric_summary(e.evaluate(tiny_model(), self.ds,
                    plan['selection']['indices'][c], 16, torch.device('cpu'))) for c in e.base.COHORTS}
        return e.base.fit_variant(tiny_model(), arm, 2025, dict.fromkeys(e.base.COHORTS, self.ds),
            plan, baseline, protocol, torch.device('cpu'), directory, progress, resume,
            epoch_trainer=e.vector_train_epoch, protocol_hash=e.PROTOCOL_SHA256,
            adapter_factory=e.attach_adapter, evaluator=e.evaluate, summarizer=e.metric_summary,
            diagnostic=diagnostic, diagnostic_key='representation_diagnostics')

    def test_interrupted_vector_recovery_and_rng_isolation(self):
        full, stopped, resumed = [self.root / n for n in ('full', 'stopped', 'resumed')]
        self.fit(full)
        def stop(stage, **fields):
            if stage == 'epoch_complete' and fields['epoch'] == 1:
                raise RuntimeError('interrupted')
        with self.assertRaisesRegex(RuntimeError, 'interrupted'):
            self.fit(stopped, progress=stop)
        def diagnostic(model, adapter, label):
            torch.rand(100)
            return {'state': e.state_hash(adapter)}
        before = {str(p): e.base.sha256(p) for p in stopped.iterdir()}
        self.fit(resumed, resume=stopped, diagnostic=diagnostic)
        self.assertEqual(json.loads((full / 'history.json').read_text()),
                         json.loads((resumed / 'history.json').read_text()))
        for name in ('last_gate.pt', 'selected_gate.pt'):
            expected = torch.load(full / name, weights_only=True)['gate_state']
            actual = torch.load(resumed / name, weights_only=True)['gate_state']
            for key in expected:
                torch.testing.assert_close(expected[key], actual[key], atol=0, rtol=0)
        self.assertEqual(before, {str(p): e.base.sha256(p) for p in stopped.iterdir()})
        with self.assertRaisesRegex(ValueError, 'identity mismatch'):
            self.fit(self.root / 'wrong_arm', arm='state_vector', resume=stopped)

    def test_comparison_rejects_misaligned_candidate_support(self):
        record = e.evaluate(tiny_model(), self.ds, [4, 5, 6, 7], 2, torch.device('cpu'))
        arms = {arm: {c: copy.deepcopy(record) for c in e.base.COHORTS} for arm in e.ARMS}
        arms['interaction_vector']['incident']['candidate_mask'][0, 0] = False
        with self.assertRaisesRegex(ValueError, 'support mismatch'):
            e.compare(dict.fromkeys(e.base.COHORTS, record), arms, {}, self.protocol)


if __name__ == '__main__':
    unittest.main()
