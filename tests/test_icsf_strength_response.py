"""Behavioral tests for a diagnostic that must never become a training run."""

from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.chronological import audit_icsf_strength_response as d
from test_incident_strength_gate import SyntheticDataset, manifests, tiny_model
from test_architecture_mechanisms import inputs


class StrengthResponseTests(unittest.TestCase):
    def test_protocol_freezes_grid_and_information_boundary(self):
        p = d.load_protocol()
        self.assertEqual(p['strengths'], [0., .5, .8, .95, .99, 1., 1.01, 1.05, 1.2, 1.5, 2.])
        self.assertEqual(p['phases'], ['fit', 'selection'])
        self.assertEqual(p['gradient_strength'], 1.)
        self.assertFalse(any(p['information_boundary'].values()))

    def test_direct_strength_identity_endpoints_and_context(self):
        model = tiny_model()
        x, incident = inputs()
        with torch.no_grad():
            native = model(x, incident_data=incident)
        before = d.gate.backbone_hash(model)
        control = d.attach(model)
        with torch.no_grad():
            torch.testing.assert_close(model(x, incident_data=incident), native, rtol=0, atol=0)
        base = model.icsf_module.base
        history = torch.randn(2, 12, 4, base.q_proj.in_features)
        tod = torch.randn(2, model.T_i_D_emb.shape[1])
        dow = torch.randn_like(tod)
        contexts = []
        for g in (0., 1., 2.):
            control.value = g
            enhanced, context = model.icsf_module(history, incident, incident_tod_feat=tod, incident_day_feat=dow)
            contexts.append(context)
            if g == 0:
                torch.testing.assert_close(enhanced[:, -1], base.output_norm(history[:, -1]), rtol=0, atol=0)
            self.assertTrue(torch.isfinite(enhanced).all())
        for context in contexts[1:]:
            for key in context:
                torch.testing.assert_close(context[key], contexts[0][key], rtol=0, atol=0)
        d.gate.assert_backbone(model, before)

    def test_gradient_partition_and_shared_scalar_finite_difference(self):
        ds, model = SyntheticDataset(), tiny_model()
        raw = next(iter(d.gate.loader(ds, [0, 1], 2)))
        batch = d.gate.device_batch(raw, torch.device('cpu'))
        before = d.gate.backbone_hash(model)
        control = d.attach(model)
        gradients, counts, mask = d.gradient_batch(model, control, batch, ds.scaler)
        np.testing.assert_allclose(gradients[..., 0], gradients[..., 1:].sum(-1), atol=1e-5, rtol=1e-4)
        self.assertEqual(np.count_nonzero(gradients[~mask]), 0)
        self.assertGreater(np.linalg.norm(gradients), 0.)
        def loss(g):
            control.value = g
            with torch.no_grad():
                y = model(batch['x'], incident_data=batch['incident']) * ds.scaler['std'] + ds.scaler['mean']
                return float((y.double() - batch['y_flow'].double()).abs().mean() / ds.scaler['std'])
        numerical = (loss(1.001) - loss(.999)) / .002
        actual = gradients[..., 0].sum() / counts[:, 0].sum()
        np.testing.assert_allclose(actual, numerical, atol=2e-5, rtol=.05)
        d.gate.assert_backbone(model, before)
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_gradients_invariant_to_batch_partition(self):
        ds, model = SyntheticDataset(), tiny_model()
        control = d.attach(model)
        def compute(size):
            results = [d.gradient_batch(model, control, d.gate.device_batch(b, torch.device('cpu')), ds.scaler)
                       for b in d.gate.loader(ds, [0, 1], size)]
            return np.concatenate([r[0] for r in results])
        np.testing.assert_allclose(compute(1), compute(2), rtol=1e-4, atol=1e-5)

    def test_changed_identity_is_rejected_before_publishing_metrics(self):
        ds, model = SyntheticDataset(), tiny_model()
        control = d.attach(model)
        def wrong_strength(history, incident):
            return history.new_full((history.shape[0], history.shape[2], 1), 1.2)
        with patch.object(control, 'forward', side_effect=wrong_strength):
            with self.assertRaisesRegex(ValueError, 'exactly reproduce'):
                d.evaluate(model, control, ds, [0, 1], 1., 2, torch.device('cpu'), lambda *a, **k: None)

    def test_global_denominator_and_opposing_region_vectors(self):
        # Own-region means differ, but global contributions must add on one denominator.
        record = {'counts': np.array([[100, 10, 30, 60]]),
                  'error_sum_gradients': np.array([[[1., 3., -1., -1.], [1., 1., -2., 2.]]])}
        s = d.summarize_gradients(record)
        self.assertAlmostEqual(s['regions']['all']['shared_scalar_global_contribution_derivative'], .02)
        self.assertAlmostEqual(s['regions']['candidate_h1_h6']['shared_scalar_own_region_mean_derivative'], .4)
        self.assertLess(s['region_pairs']['candidate_h1_h6__candidate_h7_h12']['cosine'], 0)
        self.assertLess(s['input_coordinate_cancellation_ratio'], 1.)
        self.assertAlmostEqual(s['v12c_scalar_logit_global_derivative_at_identity'], .01)

    def test_full_pipeline_never_fetches_audit_samples_or_updates_parameters(self):
        class GuardedDataset(SyntheticDataset):
            def __getitem__(self, index):
                if index >= 4:
                    raise AssertionError('Audit-period sample fetched')
                return super().__getitem__(index)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ds, model = GuardedDataset(), tiny_model()
            checkpoint = root / 'A.pt'
            torch.save(model.state_dict(), checkpoint)
            baseline = {'checkpoint': {'parameters': sum(p.numel() for p in model.parameters())}}
            with patch.object(d.gate.mechanisms, 'verify_inputs', return_value=(baseline, {})), patch.object(
                    d.gate, 'read_csv', side_effect=list(manifests())), patch.object(
                    d.gate, 'make_datasets', return_value=dict.fromkeys(d.gate.COHORTS, ds)), patch.object(
                    d.gate, 'make_model', return_value=model), patch.object(
                    torch.optim.Adam, 'step', side_effect=AssertionError('Optimizer used')), redirect_stdout(io.StringIO()):
                s = d.run(root, root, root, checkpoint, root / 'result', 'cpu')
            self.assertEqual(set(s['response']), {'fit', 'selection'})
            self.assertEqual(set(s['response']['fit']), {'incident_full'})
            self.assertEqual(len(s['response']['selection']), 4)
            self.assertEqual(s['status'], 'ICSF_STRENGTH_RESPONSE_DIAGNOSTIC_COMPLETE')
            for cohorts in s['response'].values():
                for points in cohorts.values():
                    self.assertEqual(len(points), 11)
                    self.assertEqual(points[5]['gain_vs_identity'], dict.fromkeys(d.REGIONS, 0.))
            for name, digest in s['outputs'].items():
                self.assertEqual(d.gate.sha256(root / 'result' / name), digest)
            with self.assertRaises(FileExistsError):
                d.run(root, root, root, checkpoint, root / 'result', 'cpu')

    def test_invalid_inputs_leave_failure_and_never_build_model(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'result'
            with patch.object(d.gate.mechanisms, 'verify_inputs', side_effect=ValueError('input mismatch')), patch.object(
                    d.gate, 'make_model') as model:
                with self.assertRaisesRegex(ValueError, 'input mismatch'):
                    d.run('.', '.', '.', '.', output, 'cpu')
                model.assert_not_called()
            self.assertFalse(output.exists())
            self.assertTrue((Path(str(output) + '.partial') / 'failure.json').is_file())


if __name__ == '__main__':
    unittest.main()
