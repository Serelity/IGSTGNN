"""Identity, gradients, temporal boundaries and full workflow with synthetic targets."""

from contextlib import redirect_stdout
import copy
from datetime import datetime, timedelta
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.chronological import train_incident_strength_gate as experiment
from src.models.incident_strength_gate import attach_gate
from test_architecture_mechanisms import tiny_model, inputs


class GateTests(unittest.TestCase):
    def test_both_variants_exact_identity_including_zero_support_and_off(self):
        for variant in ('scalar', 'node'):
            for support in (True, False):
                model = tiny_model()
                x, incident = inputs()
                if not support:
                    incident['distances'].zero_()
                with torch.no_grad():
                    native = model(x, incident_data=incident)
                    off = model(x, incident_data=None)
                before = experiment.backbone_hash(model)
                attach_gate(model, variant)
                with torch.no_grad():
                    torch.testing.assert_close(model(x, incident_data=incident), native, atol=0, rtol=0)
                    torch.testing.assert_close(model(x, incident_data=None), off, atol=0, rtol=0)
                experiment.assert_backbone(model, before)

    def test_gradients_reach_only_gate_and_hidden_layer_after_first_update(self):
        for variant in ('scalar', 'node'):
            model = tiny_model()
            before = experiment.backbone_hash(model)
            gate = attach_gate(model, variant)
            optimizer = torch.optim.Adam(gate.parameters(), lr=.01)
            x, incident = inputs()
            projection = torch.randn(2, 12, 4, 1)
            first_gradient = None
            for step in range(2):
                optimizer.zero_grad(set_to_none=True)
                (model(x, incident_data=incident) * projection).sum().backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in gate.parameters()))
                if variant == 'node':
                    gradient = gate.network[0].weight.grad.abs().sum().item()
                    if step == 0:
                        first_gradient = gradient
                    else:
                        self.assertGreater(gradient, 0)
                else:
                    self.assertGreater(abs(gate.logit.grad.item()), 0)
                optimizer.step()
                experiment.assert_backbone(model, before)
            if variant == 'node':
                self.assertEqual(first_gradient, 0.)
            self.assertFalse(torch.equal(model.icsf_module.last_gate, torch.ones_like(model.icsf_module.last_gate)))

    def test_gate_acts_before_normalization_and_leaves_tiid_context_exact(self):
        model = tiny_model()
        raw = model.icsf_module
        x, incident = inputs()
        history = torch.randn(2, 12, 4, raw.q_proj.in_features)
        tod = torch.randn(2, 4)
        dow = torch.randn(2, 4)
        native, context = raw(history, incident, None, tod, dow)
        gate = attach_gate(model, 'scalar')
        with torch.no_grad():
            gate.logit.fill_(np.log(3))  # g=1.5, no change to K/TIID context.
        adapted, other = model.icsf_module(history, incident, None, tod, dow)
        embedding = raw.embed_incident_features(incident, tod, dow)
        mask = (incident['distances'].abs().sum(-1, keepdim=True) > 0)
        expected = raw.output_norm(history[:, -1] + 1.5 * mask * raw.v_proj(embedding)[:, None])
        torch.testing.assert_close(adapted[:, -1], expected)
        torch.testing.assert_close(adapted[:, :-1], history[:, :-1], atol=0, rtol=0)
        torch.testing.assert_close(adapted[:, -1, 2:], native[:, -1, 2:], atol=0, rtol=0)
        for name in context:
            torch.testing.assert_close(other[name], context[name], atol=0, rtol=0)


def manifests():
    days = ['2023-01-10', '2023-01-17', '2023-05-16', '2023-05-23',
            '2023-07-04', '2023-07-11', '2023-07-18', '2023-07-25']
    ids = [11, 73, 22, 98, 15, 123, 4, 56]
    positive, primary, secondary = [], [], []
    for index, (day, sample) in enumerate(zip(days, ids)):
        t0 = datetime.fromisoformat(day + 'T12:00:00')
        interval = {'support_start': (t0 - timedelta(hours=1)).isoformat(),
                    'support_end_exclusive': (t0 + timedelta(hours=2)).isoformat()}
        positive.append({'sample_index': str(sample), 'incident_id': str(sample), 'split': 'train',
                         't0': t0.isoformat(), **interval})
        control = {'positive_sample_index': str(sample), 'control_index': str(index),
                   'incident_id': str(sample), 'split': 'train', 'positive_t0': t0.isoformat(),
                   'candidate_t0': t0.isoformat(), **interval}
        primary.append(control.copy())
        secondary.append(control.copy())
    return positive, primary, secondary


class SyntheticDataset:
    scaler = {'mean': 20., 'std': 5.}
    station_ids = [1, 2, 3, 4]

    def __init__(self):
        x, incident = inputs()
        self.x, self.incident = x.numpy(), {k: v.numpy() for k, v in incident.items()}
        self.ids = [int(row['sample_index']) for row in manifests()[0]]
        with torch.no_grad():
            self.targets = (tiny_model()(x, incident_data=incident) * 5 + 20 + 1).numpy()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        j = i % 2
        return {'x': self.x[j].copy(), 'incident': {k: v[j].copy() for k, v in self.incident.items()},
                'y_flow': self.targets[j].copy(), 'y_valid': np.ones_like(self.targets[j], dtype=bool),
                'x_valid': np.ones((12, 4, 1), dtype=bool),
                'candidate_mask': np.asarray([True, True, False, False]),
                'positive_sample_index': np.int64(self.ids[i]), 'source_index': np.int64(i)}


class WorkflowTests(unittest.TestCase):
    def test_periods_purge_crossing_controls_without_changing_positive_cohort(self):
        protocol = experiment.load_protocol()
        positive, primary, secondary = manifests()
        secondary[2]['support_start'] = '2023-04-01T00:00:00'
        plan = experiment.make_plan(positive, primary, secondary, protocol)
        self.assertEqual(plan['selection']['positive_ids'], [22, 98])
        self.assertEqual(plan['selection']['matched_ids'], [98])
        self.assertEqual(plan['fit']['indices']['incident_full'], [0, 1])
        self.assertEqual(plan['audit']['positive_ids'], [15, 123, 4, 56])
        changed = copy.deepcopy(protocol)
        changed['periods']['selection'][0] = '2023-01-01T00:00:00'
        with self.assertRaisesRegex(ValueError, 'embargo'):
            experiment.make_plan(positive, primary, secondary, changed)

    def test_selection_rejects_local_control_harm_even_if_global_improves(self):
        protocol = experiment.load_protocol()
        baseline = {c: {'mae': dict.fromkeys(experiment.REGIONS, 10.)} for c in experiment.COHORTS}
        current = copy.deepcopy(baseline)
        current['incident_full']['mae']['all'] = 9.
        self.assertTrue(experiment.selection_eligible(current, baseline, protocol)[0])
        current['primary_control']['mae']['candidate_h7_h12'] = 10.02
        self.assertFalse(experiment.selection_eligible(current, baseline, protocol)[0])
        current['primary_control']['mae']['candidate_h7_h12'] = None
        self.assertFalse(experiment.selection_eligible(current, baseline, protocol)[0])

    def test_fallback_to_A_and_no_control_targets_in_optimizer(self):
        protocol = copy.deepcopy(experiment.load_protocol())
        protocol['training']['epochs'] = 2
        dataset = SyntheticDataset()
        plan = experiment.make_plan(*manifests(), protocol)
        baseline = {c: {'mae': dict.fromkeys(experiment.REGIONS, 0.)} for c in experiment.COHORTS}
        # Control and selection datasets must never be accessed under enabled autograd.
        class SelectionOnly:
            scaler = dataset.scaler
            def __getitem__(self, index):
                if torch.is_grad_enabled():
                    raise AssertionError('Control future Y entered optimizer')
                return dataset[index]
        datasets = {'incident_full': dataset, **{c: SelectionOnly() for c in experiment.COHORTS[1:]}}
        with tempfile.TemporaryDirectory() as directory:
            result = experiment.fit_variant(tiny_model(), 'node', 2025, datasets, plan, baseline,
                protocol, torch.device('cpu'), Path(directory), lambda *a, **k: None)
            self.assertEqual(result['selected_epoch'], 0)
            self.assertEqual(result['optimizer_steps'], 2)
            checkpoint = torch.load(Path(directory) / 'selected_gate.pt', weights_only=True)
            self.assertTrue(torch.equal(checkpoint['gate_state']['network.2.weight'],
                                        torch.zeros_like(checkpoint['gate_state']['network.2.weight'])))

    def test_future_targets_do_not_enter_gate_inputs(self):
        model = tiny_model()
        attach_gate(model, 'node')
        sample = SyntheticDataset()[0]
        x = torch.from_numpy(sample['x'])[None]
        incident = {k: torch.as_tensor(v)[None] for k, v in sample['incident'].items()}
        with torch.no_grad():
            a = model(x, incident_data=incident)
            sample['y_flow'][:] = -100000
            b = model(x, incident_data=incident)
        torch.testing.assert_close(a, b, atol=0, rtol=0)

    def test_zero_gradient_after_first_epoch_is_recorded_as_stagnation(self):
        protocol = experiment.load_protocol()
        dataset = SyntheticDataset()
        dataset.scaler = {'mean': 0., 'std': 1.}
        x, incident = inputs()
        model = tiny_model()
        with torch.no_grad():
            dataset.targets = model(x, incident_data=incident).numpy()
        gate = attach_gate(model, 'scalar')
        optimizer = torch.optim.Adam(gate.parameters(), lr=.001)
        result = experiment.train_epoch(model, gate, optimizer, dataset, [0, 1],
            protocol['training'], torch.device('cpu'), 2025, 2, lambda *a, **k: None)
        self.assertFalse(result['gate_parameters_changed'])
        self.assertEqual(result['maximum_gradient_norm'], 0.)

    def test_engineering_check_reports_only_actual_subsample_and_no_bootstrap(self):
        protocol = experiment.load_protocol()
        dataset = SyntheticDataset()
        model = tiny_model()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / 'A.pt'
            torch.save(model.state_dict(), checkpoint)
            baseline = {'checkpoint': {'parameters': sum(p.numel() for p in model.parameters())}}
            with patch.object(experiment.mechanisms, 'verify_inputs', return_value=(baseline, {})), patch.object(
                    experiment, 'read_csv', side_effect=list(manifests())), patch.object(
                    experiment, 'make_datasets', return_value=dict.fromkeys(experiment.COHORTS, dataset)), patch.object(
                    experiment, 'make_model', side_effect=lambda *a, **k: tiny_model()), redirect_stdout(io.StringIO()):
                summary = experiment.run(root, root, root, checkpoint, root / 'check', 'cpu', check=True)
            self.assertEqual(summary['status'], 'ENGINEERING_CHECK_PASS')
            self.assertEqual(summary['audit_comparisons'], {})
            self.assertEqual(list(summary['runs']), ['2025'])
            plan = json.loads((root / 'check/effective_plan.json').read_text())
            self.assertEqual(plan['audit']['positive_ids'], [15, 123])
            self.assertEqual(len(plan['audit']['indices']['incident_full']), 2)

    def test_full_synthetic_workflow_publication_and_paired_statistics(self):
        protocol = copy.deepcopy(experiment.load_protocol())
        protocol['training']['epochs'] = 2
        protocol['seeds'] = [2025]
        protocol['bootstrap']['draws'] = 40
        dataset = SyntheticDataset()
        model = tiny_model()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / 'A.pt'
            torch.save(model.state_dict(), checkpoint)
            baseline = {'checkpoint': {'parameters': sum(p.numel() for p in model.parameters())}}
            with patch.object(experiment, 'load_protocol', return_value=protocol), patch.object(
                    experiment.mechanisms, 'verify_inputs', return_value=(baseline, {})), patch.object(
                    experiment, 'read_csv', side_effect=list(manifests())), patch.object(
                    experiment, 'make_datasets', return_value=dict.fromkeys(experiment.COHORTS, dataset)), patch.object(
                    experiment, 'make_model', side_effect=lambda *a, **k: tiny_model()), redirect_stdout(io.StringIO()):
                summary = experiment.run(root, root, root, checkpoint, root / 'result', 'cpu')
            self.assertEqual(summary['status'], 'ICSF_STRENGTH_GATE_EXPERIMENT_COMPLETE')
            self.assertFalse(summary['validation_arrays_read'])
            self.assertFalse(summary['control_Y_used_for_optimizer'])
            self.assertFalse(summary['independent_confirmation'])
            self.assertEqual(set(summary['runs']['2025']), {'scalar', 'node'})
            self.assertIn('node_vs_scalar', summary['audit_comparisons']['2025']['results']['incident_full']['regions']['all']['comparisons'])
            self.assertFalse((root / 'result.partial').exists())
            for name, digest in summary['outputs'].items():
                self.assertEqual(experiment.sha256(root / 'result' / name), digest)
            with self.assertRaises(FileExistsError):
                experiment.run(root, root, root, checkpoint, root / 'result', 'cpu')

    def test_bad_inputs_leave_failure_record_and_never_train(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'bad'
            with patch.object(experiment.mechanisms, 'verify_inputs', side_effect=ValueError('bad hash')), patch.object(
                    experiment, 'fit_variant') as fit, redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError, 'bad hash'):
                    experiment.run('.', '.', '.', '.', output, 'cpu')
                fit.assert_not_called()
            self.assertFalse(output.exists())
            self.assertEqual(json.loads((Path(directory) / 'bad.partial/failure.json').read_text())['error'], 'bad hash')

    def test_launcher_stops_on_test_or_engineering_failure(self):
        for failing_stage, expected_status in (('tests', 9), ('check', 17)):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                scripts = root / 'repo/experiments/chronological'
                scripts.mkdir(parents=True)
                launcher = scripts / 'run_incident_strength_gate.sh'
                shutil.copyfile(experiment.REPO / 'experiments/chronological' / launcher.name, launcher)
                job = scripts.parent / 'chronological_runs/contra_v12c_strength_gate_test.job'
                job.mkdir(parents=True)
                fake_bin = root / 'bin'
                fake_bin.mkdir()
                git = fake_bin / 'git'
                git.write_text('#!/bin/bash\nexit 0\n')
                git.chmod(0o755)
                python = fake_bin / 'python with spaces'
                python.write_text('#!/bin/bash\n'
                    'if [[ "$1" == "-c" ]]; then exit 0; fi\n'
                    'if [[ "$1" == "-m" ]]; then\n'
                    '  if [[ "$FAIL_STAGE" == tests ]]; then exit 9; else exit 0; fi\nfi\n'
                    'for arg in "$@"; do if [[ "$arg" == --check ]]; then exit 17; fi; done\n'
                    'echo FULL_TRAINING_MUST_NOT_START\nexit 77\n')
                python.chmod(0o755)
                result = subprocess.run(['bash', str(launcher), '_worker', 'contra_v12c_strength_gate_test',
                    str(python), 'cpu'], capture_output=True, text=True,
                    env={**os.environ, 'FAIL_STAGE': failing_stage,
                         'PATH': str(fake_bin) + os.pathsep + os.environ['PATH']})
                self.assertEqual(result.returncode, expected_status, result.stderr)
                self.assertEqual((job / 'exit_code').read_text().strip(), str(expected_status))
                self.assertNotIn('FULL_TRAINING_MUST_NOT_START', result.stdout)


if __name__ == '__main__':
    unittest.main()
