"""Objective contrast, observation isolation and full/recovered paired workflow."""

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

from experiments.chronological import train_regional_gate_objective as e
from test_incident_strength_gate import SyntheticDataset, manifests, tiny_model


class ObjectiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_protocol_and_probe_choice_are_frozen(self):
        p = e.load_protocol()
        self.assertEqual(p['arms'], ['global', 'regional'])
        self.assertEqual(p['region_weights'], [1/3] * 3)
        self.assertEqual(e.probe_indices([1, 4, 7, 12, 19], 3), [1, 7, 19])
        self.assertEqual(e.probe_indices([1, 4], 64), [1, 4])
        self.assertFalse(p['information_boundary']['audit_Y_used_for_selection'])

    def test_regional_loss_is_equal_region_mean_not_global_cell_mean(self):
        prediction = torch.zeros(1, 12, 4, 1, requires_grad=True)
        target = torch.full_like(prediction, 10.)
        target[:, :6, 0] = 1.
        target[:, 6:, 0] = 2.
        valid = torch.ones_like(prediction, dtype=torch.bool)
        candidate = torch.tensor([[True, False, False, False]])
        sums, counts = e.region_sums(prediction, target, valid, candidate)
        self.assertEqual(counts.tolist(), [6, 6, 36])
        self.assertAlmostEqual(float((sums / counts).mean()), 13/3, places=6)
        self.assertAlmostEqual(float(sums.sum() / counts.sum()), 7.875)
        (sums / counts).mean().backward()
        self.assertAlmostEqual(float(prediction.grad[0, 0, 0, 0]), -1/18, places=6)
        self.assertAlmostEqual(float(prediction.grad[0, 0, 1, 0]), -1/108, places=6)
        valid[:, :6, 0] = False
        with self.assertRaisesRegex(ValueError, 'nonempty'):
            e.region_sums(prediction, target, valid, candidate)

    def test_probe_actual_parameter_gradient_matches_finite_difference_and_preserves_state(self):
        model, ds = tiny_model(), SyntheticDataset()
        adapter = e.base.attach_gate(model, 'node')
        before = e.state_hash(adapter)
        output = self.root / 'probe.npz'
        result = e.parameter_probe(model, adapter, ds, [0, 1], {'batch_size': 2, 'penalty': .001},
                                   torch.device('cpu'), output, lambda *a, **k: None)
        self.assertEqual(e.state_hash(adapter), before)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        self.assertEqual(result['gradient_l2']['identity_penalty_mean'], 0.)
        self.assertEqual(result['parameter_names'], [n for n, _ in adapter.named_parameters()])
        arrays = np.load(output)
        analytic = arrays['gradients'][list(arrays['gradient_names']).index('regional_data'), -1]
        batch = e.base.device_batch(next(iter(e.base.loader(ds, [0, 1], 2))), torch.device('cpu'))
        parameter = list(adapter.parameters())[-1]
        def loss(value):
            with torch.no_grad():
                parameter.fill_(value)
                prediction = model(batch['x'], incident_data=batch['incident'])
                target = (batch['y_flow'] - ds.scaler['mean']) / ds.scaler['std']
                sums, counts = e.region_sums(prediction, target, batch['y_valid'], batch['candidate_mask'])
                return float((sums / counts).mean())
        numeric = (loss(.002) - loss(-.002)) / .004
        np.testing.assert_allclose(analytic, numeric, rtol=.05, atol=2e-5)

    def fit(self, path, arm='global', diagnostic=None, progress=lambda *a, **k: None, resume=None):
        protocol = copy.deepcopy(e.base.load_protocol())
        protocol['training']['epochs'] = 2
        protocol['training']['loss_arm'] = arm
        ds = SyntheticDataset()
        datasets = dict.fromkeys(e.base.COHORTS, ds)
        plan = e.base.make_plan(*manifests(), protocol)
        baseline = {c: e.base.metric_summary(e.base.evaluate(tiny_model(), ds,
            plan['selection']['indices'][c], 16, torch.device('cpu'))) for c in e.base.COHORTS}
        path.mkdir()
        return e.base.fit_variant(tiny_model(), 'node', 2025, datasets, plan, baseline, protocol,
            torch.device('cpu'), path, progress, resume,
            epoch_trainer=e.base.train_epoch if arm == 'global' else e.regional_train_epoch,
            protocol_hash=e.PROTOCOL_SHA256, diagnostic=diagnostic)

    def test_diagnostic_does_not_change_original_global_fit_or_rng(self):
        def observe(model, adapter, label):
            # A deliberately RNG-consuming observation must have no fitting side effects.
            torch.rand(200)
            return {'state': e.state_hash(adapter)}
        a, b = self.root / 'plain', self.root / 'observed'
        expected = self.fit(a)
        actual = self.fit(b, diagnostic=observe)
        self.assertEqual(expected['selection_metrics'], actual['selection_metrics'])
        self.assertEqual(json.loads((a / 'history.json').read_text()), json.loads((b / 'history.json').read_text()))
        for name in ('last_gate.pt', 'selected_gate.pt'):
            left = torch.load(a / name, weights_only=True)['gate_state']
            right = torch.load(b / name, weights_only=True)['gate_state']
            for k in left:
                torch.testing.assert_close(left[k], right[k], rtol=0, atol=0)

    def test_regional_recovery_matches_uninterrupted_and_rejects_other_arm(self):
        a, b, c = [self.root / name for name in ('whole', 'stopped', 'resumed')]
        self.fit(a, arm='regional')
        def stop(stage, **fields):
            if stage == 'epoch_complete' and fields['epoch'] == 1:
                raise RuntimeError('interrupted')
        with self.assertRaisesRegex(RuntimeError, 'interrupted'):
            self.fit(b, arm='regional', progress=stop)
        before = {p.name: e.base.sha256(p) for p in b.iterdir()}
        self.fit(c, arm='regional', resume=b)
        self.assertEqual(before, {p.name: e.base.sha256(p) for p in b.iterdir()})
        self.assertEqual(json.loads((a / 'history.json').read_text()), json.loads((c / 'history.json').read_text()))
        for k, v in torch.load(a / 'last_gate.pt', weights_only=True)['gate_state'].items():
            torch.testing.assert_close(v, torch.load(c / 'last_gate.pt', weights_only=True)['gate_state'][k], rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, 'settings changed'):
            self.fit(self.root / 'wrong_arm', arm='global', resume=b)

    def test_full_workflow_matching_arms_probes_and_recovery_without_optimizer(self):
        ds, model = SyntheticDataset(), tiny_model()
        baseline = {'checkpoint': {'parameters': sum(p.numel() for p in model.parameters())}}
        checkpoint = self.root / 'A.pt'
        torch.save(model.state_dict(), checkpoint)
        protocol = copy.deepcopy(e.base.load_protocol())
        protocol['seeds'] = [2025]
        protocol['training']['epochs'] = 2
        protocol['bootstrap']['draws'] = 40
        def execute(output, resume=None, check=False):
            with patch.object(e.base, 'load_protocol', return_value=protocol), patch.object(
                    e.base.mechanisms, 'verify_inputs', return_value=(baseline, {})), patch.object(
                    e.base, 'read_csv', side_effect=list(manifests())), patch.object(
                    e.base, 'make_datasets', return_value=dict.fromkeys(e.base.COHORTS, ds)), patch.object(
                    e.base, 'make_model', side_effect=lambda *a, **k: tiny_model()), redirect_stdout(io.StringIO()):
                return e.run(self.root, self.root, self.root, checkpoint, output, 'cpu', check=check, resume_from=resume)
        original = self.root / 'run'
        s = execute(original)
        self.assertEqual(s['status'], 'REGIONAL_GATE_OBJECTIVE_COMPARISON_COMPLETE')
        self.assertTrue(s['paired_initialization_exact'])
        for arm in e.ARMS:
            p = s['runs']['2025'][arm]['parameter_gradient_diagnostics']
            self.assertEqual(set(p), {'initial', 'last', 'selected'})
            self.assertEqual(p['initial']['sample_ids'], ds.ids[:2])
        self.assertEqual(s['runs']['2025']['global']['parameter_gradient_diagnostics']['initial'],
                         s['runs']['2025']['regional']['parameter_gradient_diagnostics']['initial'])
        for name, sha in s['outputs'].items():
            self.assertEqual(e.base.sha256(original / name), sha)
        before = {str(p): e.base.sha256(p) for p in original.rglob('*') if p.is_file()}
        with patch.object(e.base, 'train_epoch', side_effect=AssertionError('global refit')), patch.object(
                e, 'regional_train_epoch', side_effect=AssertionError('regional refit')):
            recovered = execute(self.root / 'resumed', original)
        self.assertEqual(s['audit_comparisons'], recovered['audit_comparisons'])
        self.assertEqual(before, {str(p): e.base.sha256(p) for p in original.rglob('*') if p.is_file()})
        with self.assertRaises(FileExistsError):
            execute(original)
        engineering = execute(self.root / 'engineering', check=True)
        self.assertEqual(engineering['status'], 'ENGINEERING_CHECK_PASS')
        self.assertEqual(engineering['audit_comparisons'], {})
        self.assertEqual(engineering['recommendation'], 'ENGINEERING_ONLY')


if __name__ == '__main__':
    unittest.main()
