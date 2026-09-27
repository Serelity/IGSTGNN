"""Behavioral checks for the v12a mechanism audit; no scientific outcomes."""

import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.chronological import audit_architecture_mechanisms as audit
from experiments.chronological.report_architecture_mechanisms import report
from src.models.igstgnn import IGSTGNN


def tiny_model():
    torch.manual_seed(11)
    torch.set_num_threads(1)
    return IGSTGNN(
        model_args=dict(num_feat=1, num_hidden=8, node_hidden=4, time_emb_dim=4,
                        layer=5, k_s=2, k_t=3, tpd=288, dropout=.1, gap=3,
                        sigma_t=1., lambda_incident=1.,
                        adjs=[torch.ones(4, 4) / 4] * 2,
                        incident_schema='report_location_v1', time_response='fixed'),
        node_num=4, input_dim=3, output_dim=1, seq_len=12, horizon=12,
        dataset='fixture', use_sensor_info=False).eval().requires_grad_(False)


def inputs():
    generator = torch.Generator().manual_seed(12)
    x = torch.rand(2, 12, 4, 3, generator=generator)
    incident = {
        'report_age_minutes': torch.tensor([1., 4.]),
        'forecast_tod': torch.tensor([11, 120]), 'forecast_dow': torch.tensor([1, 2]),
        'distances': torch.tensor([[[0., .9, 0.], [0., .2, 1.],
                                    [0., 0., 0.], [0., 0., 0.]]] * 2),
    }
    return x, incident


class PathTests(unittest.TestCase):
    def test_native_replay_uses_real_backbone_without_state_changes(self):
        model = tiny_model()
        before = audit.state_hash(model)
        x, incident = inputs()
        predictions, diagnostics = audit.path_predictions(model, x, incident)
        self.assertEqual(tuple(predictions), audit.PATHS)
        self.assertEqual(diagnostics['native_full_replay']['abs_max'], 0.)
        self.assertEqual(diagnostics['native_off_replay']['abs_max'], 0.)
        self.assertGreater(diagnostics['normalization_non_candidate']['abs_mean'], 0.)
        self.assertEqual(diagnostics['icsf_vs_norm_non_candidate']['abs_max'], 0.)
        self.assertGreater(diagnostics['icsf_vs_norm_candidate']['abs_mean'], 0.)
        self.assertEqual(before, audit.state_hash(model))
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        self.assertTrue(all(not value.requires_grad for value in predictions.values()))

    def test_zero_incident_support_collapses_to_normalization_not_off(self):
        model = tiny_model()
        x, incident = inputs()
        incident['distances'].zero_()
        predictions, _ = audit.path_predictions(model, x, incident)
        for name in ('icsf_only', 'tiid_only', 'full', 'full_norm_graph'):
            torch.testing.assert_close(predictions[name], predictions['norm_only'], atol=0, rtol=0)
        self.assertFalse(torch.equal(predictions['norm_only'], predictions['off']))

    def test_disabling_tiid_has_no_effect_on_icsf_or_normalization_paths(self):
        model = tiny_model()
        x, incident = inputs()
        model.tiid_module.incident_scale = 0.
        predictions, _ = audit.path_predictions(model, x, incident)
        torch.testing.assert_close(predictions['tiid_only'], predictions['norm_only'], atol=0, rtol=0)
        torch.testing.assert_close(predictions['full'], predictions['icsf_only'], atol=0, rtol=0)

    def test_graph_replay_equals_full_when_graph_cannot_respond_to_history(self):
        model = tiny_model()
        # Hold all dynamic graph supports fixed independently of the injected history.
        x, incident = inputs()
        real_constructor = model._graph_constructor
        stored = []

        def fixed_graph(**kwargs):
            if not stored:
                stored.append(real_constructor(**kwargs))
            return stored[0]

        with patch.object(model, '_graph_constructor', side_effect=fixed_graph):
            predictions, diagnostics = audit.path_predictions(model, x, incident)
        torch.testing.assert_close(predictions['full_norm_graph'], predictions['full'], atol=0, rtol=0)
        self.assertEqual(diagnostics['dynamic_graph_icsf_vs_norm']['abs_max'], 0.)

    def test_gradient_probe_has_positive_control_and_does_not_touch_model(self):
        model = tiny_model()
        before = audit.state_hash(model)
        result = audit.synthetic_probe(model, 12025)
        gradients = result['gradient_abs_max']
        self.assertEqual(gradients['q_proj.weight'], 0.)
        self.assertEqual(gradients['k_proj.weight'], 0.)
        for name, magnitude in gradients.items():
            if name.startswith('icsf_fusion_mlp.'):
                self.assertEqual(magnitude, 0.)
        self.assertGreater(gradients['v_proj.weight'], 0.)
        self.assertEqual(result['semantic_attention_vs_mask']['abs_max'], 0.)
        self.assertLess(result['pre_norm_injection_vs_mask_times_V']['abs_max'], 1e-6)
        self.assertLess(result['injection_change_after_history_perturbation']['abs_max'], 1e-6)
        self.assertEqual(result['injection_change_after_same_support_distance_perturbation']['abs_max'], 0.)
        self.assertEqual(result['zero_support_vs_normalization_only']['abs_max'], 0.)
        self.assertGreater(result['zero_support_latest_state_change']['abs_mean'], 0.)
        self.assertEqual(before, audit.state_hash(model))
        self.assertTrue(all(p.grad is None for p in model.parameters()))


class StatisticsTests(unittest.TestCase):
    def test_counts_missing_targets_and_interaction_are_separate(self):
        values = {'off': 1., 'norm_only': 2., 'icsf_only': 4.,
                  'tiid_only': 5., 'full': 8., 'full_norm_graph': 7.}
        predictions = {key: torch.full((2, 12, 2, 1), value) for key, value in values.items()}
        target = torch.zeros(2, 12, 2, 1)
        valid = torch.ones_like(target, dtype=torch.bool)
        valid[0, :, 0] = False
        target[~valid] = float('nan')
        candidate = torch.tensor([[True, False], [True, False]])
        arrays = audit.sample_statistics(predictions, target, valid, candidate)
        summary = audit.summarize(arrays)
        self.assertEqual(summary['all']['prediction_cells'], 48)
        self.assertEqual(summary['all']['valid_target_cells'], 36)
        self.assertEqual(summary['candidate_h1_h3']['prediction_cells'], 6)
        self.assertEqual(summary['candidate_h1_h3']['valid_target_cells'], 3)
        self.assertEqual(summary['all']['descriptive_mae']['full'], 8.)
        self.assertEqual(summary['all']['prediction_sensitivity']['icsf_tiid_interaction']['abs_mean'], 1.)
        self.assertEqual(summary['all']['prediction_sensitivity']['full_vs_off']['signed_mean'], 7.)
        # Replacing future targets cannot alter any prediction-sensitivity statistic.
        other = audit.sample_statistics(predictions, target + 900, valid, candidate)
        for key in ('delta_abs_sums', 'delta_signed_sums', 'delta_abs_max', 'prediction_counts'):
            np.testing.assert_array_equal(arrays[key], other[key])

    def test_empty_strata_serialize_null_instead_of_nan(self):
        predictions = {key: torch.zeros(1, 12, 2, 1) for key in audit.PATHS}
        arrays = audit.sample_statistics(predictions, predictions['full'],
            torch.zeros_like(predictions['full'], dtype=torch.bool), torch.zeros(1, 2, dtype=torch.bool))
        summary = audit.summarize(arrays)
        self.assertIsNone(summary['all']['descriptive_mae']['full'])
        self.assertIsNone(summary['candidate_h1_h3']['prediction_sensitivity']['normalization']['abs_max'])
        json.dumps(summary, allow_nan=False)


class BoundaryAndRunTests(unittest.TestCase):
    def test_protocol_rejects_validation_and_path_changes(self):
        spec = audit.load_protocol(audit.PROTOCOL)
        self.assertEqual(spec['split'], 'train')
        for field, replacement in (('split', 'val'), ('paths', ['full'])):
            changed = {**spec, field: replacement}
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'changed.json'
                path.write_text(json.dumps(changed))
                with self.assertRaisesRegex(ValueError, 'fingerprint'):
                    audit.load_protocol(path)

    def test_input_verification_never_reads_validation_or_test_payloads(self):
        spec = audit.load_protocol(audit.PROTOCOL)
        baseline = audit.load_baseline_protocol(audit.PROTOCOL.with_name(spec['baseline_protocol']))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            positive = root / 'data'
            positive.mkdir()
            train_names = ('train_flow.npy', 'train_manifest.csv', 'station_ids.npy', 'scaler.json')
            payloads = {name: 'fake-' + name for name in train_names}
            context_hashes = {name: 'fake-' + name for name in ('train_context.npz', 'adjacency.npy')}
            (positive / 'summary.json').write_text(json.dumps({'files': payloads}))
            (positive / 'context_manifest.json').write_text(json.dumps({'outputs': context_hashes}))
            expected = {audit.PROTOCOL.with_name(spec['baseline_protocol']): spec['baseline_protocol_sha256']}
            expected.update({audit.REPO / name: digest for name, digest in spec['model_source_sha256'].items()})
            expected[root / 'model.pt'] = baseline['checkpoint']['best_model_sha256']
            expected[positive / 'summary.json'] = baseline['positive_package']['summary_sha256']
            expected[positive / 'context_manifest.json'] = baseline['positive_package']['context_manifest_sha256']
            expected.update({positive / name: digest for name, digest in {**payloads, **context_hashes}.items()})
            for dirname, group in (('c1', 'primary_control_inputs'), ('c2', 'secondary_control_inputs')):
                expected.update({root / dirname / name: digest for name, digest in baseline[group].items()
                                 if name == 'summary.json' or name.startswith('train_')})
            accessed = []

            def guarded_hash(path):
                path = Path(path)
                self.assertFalse(path.name.startswith(('val_', 'test_')))
                accessed.append(path)
                return expected[path]

            with patch.object(audit, 'sha256', side_effect=guarded_hash):
                audit.verify_inputs(positive, root / 'c1', root / 'c2', root / 'model.pt', spec)
            self.assertEqual(set(accessed), set(expected))

    def test_all_cohorts_publish_and_failed_runs_preserve_partial(self):
        model = tiny_model()
        x, incident = inputs()
        samples = []
        for index in range(2):
            samples.append({'x': x[index].numpy(), 'incident': {
                key: value[index].numpy() for key, value in incident.items()},
                'y_flow': np.ones((12, 4, 1), dtype=np.float32),
                'y_valid': np.ones((12, 4, 1), dtype=bool),
                'candidate_mask': np.array([True, True, False, False]),
                'positive_sample_index': np.int64(10 + 3 * index), 'source_index': np.int64(index)})

        class FixtureDataset:
            scaler = {'std': 2., 'mean': 5.}
            candidate_mask_compatibility = {'fixture': True}

            def __len__(self):
                return 2

            def __getitem__(self, index):
                return samples[index]

        baseline = audit.load_baseline_protocol(audit.PROTOCOL.with_name('incident_branch_materialize_v6a.json'))
        baseline['expected_sensor_count'] = 4
        baseline['checkpoint']['parameters'] = sum(p.numel() for p in model.parameters())
        with tempfile.TemporaryDirectory() as directory, contextlib.ExitStack() as stack:
            root = Path(directory)
            checkpoint = root / 'model.pt'
            torch.save(model.state_dict(), checkpoint)
            stack.enter_context(patch.object(audit, 'verify_inputs', return_value=(baseline, {})))
            stack.enter_context(patch.object(audit, 'make_model', return_value=model))
            stack.enter_context(patch.object(audit, 'FullPositiveDataset', return_value=FixtureDataset()))
            stack.enter_context(patch.object(audit, 'MatchedCounterfactualDataset', return_value=FixtureDataset()))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            result = audit.run(root, root, root, checkpoint, root / 'output', device_name='cpu', check=True)
            self.assertEqual(result['status'], 'ENGINEERING_CHECK_PASS')
            self.assertFalse(result['full_cohort_evaluated'])
            self.assertFalse(result['validation_arrays_read'])
            self.assertEqual(set(result['results']), {'incident_full', 'incident', 'primary_control', 'secondary_control'})
            self.assertFalse((root / 'output.partial').exists())
            report(root / 'output' / 'summary.json')
            for cohort in result['results']:
                with np.load(root / 'output' / f'train_{cohort}_mechanisms.npz', allow_pickle=False) as stored:
                    np.testing.assert_array_equal(stored['positive_sample_index'], [10, 13])
                    self.assertEqual(stored['absolute_error_sums'].shape, (2, 6, 18))
            with self.assertRaises(FileExistsError):
                audit.run(root, root, root, checkpoint, root / 'output', device_name='cpu')
            with patch.object(audit, 'path_predictions', side_effect=RuntimeError('fixture failure')):
                with self.assertRaisesRegex(RuntimeError, 'fixture failure'):
                    audit.run(root, root, root, checkpoint, root / 'failed', device_name='cpu', check=True)
            self.assertTrue((root / 'failed.partial' / 'failure.json').is_file())
            self.assertFalse((root / 'failed').exists())
            with self.assertRaises(FileExistsError):
                audit.run(root, root, root, checkpoint, root / 'failed', device_name='cpu')


@unittest.skipUnless(os.name == 'posix', 'Server launcher requires Bash')
class LauncherTests(unittest.TestCase):
    def test_engineering_failure_stops_full_audit_and_records_exit_status(self):
        self.exercise_worker(check_status=17)

    def test_successful_engineering_check_runs_full_audit_then_report(self):
        self.exercise_worker(check_status=0)

    def exercise_worker(self, check_status):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scripts = root / 'experiments' / 'chronological'
            scripts.mkdir(parents=True)
            source = audit.REPO / 'experiments/chronological/run_architecture_audit.sh'
            runner = scripts / source.name
            shutil.copy2(source, runner)
            job = root / 'experiments/chronological_runs/contra_v12a_architecture_fixture.job'
            job.mkdir(parents=True)
            stub_dir = root / 'bin'
            stub_dir.mkdir()
            fake_git = stub_dir / 'git'
            fake_git.write_text('#!/bin/sh\nexit 0\n')
            fake_git.chmod(0o755)
            fake_python = stub_dir / 'python'
            fake_python.write_text(
                f'#!{sys.executable}\n'
                'import json, os, sys\n'
                'from pathlib import Path\n'
                'with Path(os.environ["CALLS_FILE"]).open("a") as stream:\n'
                '    stream.write(json.dumps(sys.argv[1:]) + "\\n")\n'
                'sys.exit(int(os.environ["CHECK_STATUS"]) if "--check" in sys.argv else 0)\n')
            fake_python.chmod(0o755)
            calls_path = root / 'calls.jsonl'
            env = {**os.environ, 'PATH': str(stub_dir) + os.pathsep + os.environ['PATH'],
                   'CHECK_STATUS': str(check_status), 'CALLS_FILE': str(calls_path)}
            result = subprocess.run(['bash', str(runner), '_worker',
                'contra_v12a_architecture_fixture', str(fake_python), 'cpu'],
                env=env, capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, check_status, result.stderr)
            self.assertEqual((job / 'exit_code').read_text().strip(), str(check_status))
            calls = [json.loads(line) for line in calls_path.read_text().splitlines()]
            audits = [args for args in calls if any('audit_architecture_mechanisms.py' in a for a in args)]
            self.assertEqual(len(audits), 1 if check_status else 2)
            self.assertIn('--check', audits[0])
            if not check_status:
                self.assertNotIn('--check', audits[1])
                self.assertTrue(any('report_architecture_mechanisms.py' in a for a in calls[-1]))


if __name__ == '__main__':
    unittest.main()
