"""Read-only selected-checkpoint replay with actual tiny frozen IGSTGNN models."""

from contextlib import ExitStack, redirect_stdout
import copy
import csv
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.chronological import audit_state_interaction_fit as e
from test_incident_strength_gate import manifests, SyntheticDataset, tiny_model


def fixture(root):
    source, data, primary, secondary, checkpoints = [root / name for name in
        ('source', 'data', 'primary', 'secondary', 'checkpoints')]
    for directory in (source, data, primary, secondary, checkpoints):
        directory.mkdir()
    protocol = copy.deepcopy(e.load_protocol())
    protocol['fit_samples'] = 2
    reader = copy.deepcopy(e.transfer.load_protocol())
    reader['node_count'] = 4
    reader['source_phase_samples'] = {'fit': {'incident_full': 2},
        'selection': dict.fromkeys(e.transfer.COHORTS, 2),
        'audit': dict.fromkeys(e.transfer.COHORTS, 4)}
    positive, first, second = manifests()
    # Keep the original chronological period bounds and a >=4-week fit grid.
    for rows in (positive, first, second):
        for key in ('t0', 'positive_t0', 'candidate_t0', 'support_start', 'support_end_exclusive'):
            if key in rows[1]:
                rows[1][key] = rows[1][key].replace('2023-01-17', '2023-02-07')
    for directory, name, rows in ((data, 'train_manifest.csv', positive),
                                  (primary, 'train_control_manifest.csv', first),
                                  (secondary, 'train_second_control_manifest.csv', second)):
        with (directory / name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (data / 'val_flow.npy').write_bytes(b'Forbidden old validation sentinel')
    (data / 'test_flow.npy').write_bytes(b'Forbidden test sentinel')
    inherited = e.base.load_protocol()
    plan = e.base.make_plan(positive, first, second, inherited)
    ds = SyntheticDataset()
    native = tiny_model()
    checkpoint = checkpoints / 'A.pt'
    torch.save(native.state_dict(), checkpoint)
    hashes = {str(directory / name): e.base.sha256(directory / name) for directory, name in
              ((data, 'train_manifest.csv'), (primary, 'train_control_manifest.csv'),
               (secondary, 'train_second_control_manifest.csv'))}
    hashes[str(checkpoint)] = e.base.sha256(checkpoint)
    baseline = {'checkpoint': {'parameters': sum(p.numel() for p in native.parameters())},
                'expected_positive_samples': {'train': len(ds)}, 'expected_sensor_count': 4}
    summary = {'status': 'INCIDENT_STATE_INTERACTION_COMPARISON_COMPLETE',
        'protocol_sha256': reader['source_protocol_sha256'], 'engineering_check': False,
        'model_training_performed': True, 'validation_arrays_read': False, 'test_split_read': False,
        'independent_confirmation': False, 'vector_paired_initialization_exact': True,
        'frozen_protocol': json.loads(e.transfer.SOURCE_PROTOCOL.read_text()),
        'inherited_protocol': inherited, 'effective_training': inherited['training'],
        'phase_samples': reader['source_phase_samples'], 'inputs': hashes,
        'code_sha256': reader['source_code_sha256'],
        'environment': {'git_head': protocol['source_v12f_git_head']}, 'runs': {}}
    identity = {'protocol_sha256': reader['source_protocol_sha256'], 'engineering_check': False,
        'inputs': hashes, 'checkpoint_sha256': e.base.sha256(checkpoint),
        'code_sha256': reader['source_code_sha256'],
        'indices': {phase: item['indices'] for phase, item in plan.items()}, 'probe_indices': [0, 1]}
    def save_json(name, payload):
        (source / name).parent.mkdir(parents=True, exist_ok=True)
        (source / name).write_text(json.dumps(payload, allow_nan=False))
    save_json('run_identity.json', identity)
    save_json('eligibility.json', plan)
    initial_record = e.inference.evaluate(native, ds, [0, 1], 8, torch.device('cpu'))
    summary['baseline_audit'] = {}
    for phase in ('selection', 'audit'):
        indices = plan[phase]['indices']['incident_full']
        record = e.inference.evaluate(native, ds, indices, 16, torch.device('cpu'))
        e.base.save_arrays(source / f'{phase}_A_incident_full.npz', record)
        if phase == 'audit':
            summary['baseline_audit']['incident_full'] = e.inference.metric_summary(record)
    for seed in protocol['seeds']:
        summary['runs'][str(seed)] = {}
        for arm in protocol['arms']:
            directory = source / f'{arm}_s{seed}'
            directory.mkdir()
            model = tiny_model()
            backbone_digest = e.base.backbone_hash(model)
            gate = e.inference.attach_adapter(model, arm, inherited['training']['node_hidden_width'])
            initial_digest = e.inference.state_hash(gate)
            epoch = 0 if arm == 'strength' and seed != 2027 else 1
            if epoch:
                with torch.no_grad():
                    if arm == 'strength':
                        gate.network[2].bias.fill_(.025)
                    else:
                        gate.output.bias.fill_(.01)
            model.requires_grad_(False).eval()
            selected_digest = e.inference.state_hash(gate)
            torch.save({'variant': arm, 'seed': seed, 'epoch': epoch,
                'protocol_sha256': reader['source_protocol_sha256'],
                'backbone_state_sha256': backbone_digest, 'gate_state': gate.state_dict()},
                directory / 'selected_gate.pt')
            (directory / 'last_gate.pt').write_bytes(b'Forbidden optimizer/last-state sentinel')
            selected_probe = e.inference.evaluate(model, ds, [0, 1], 8, torch.device('cpu'))
            e.base.save_arrays(directory / 'representation_initial.npz', initial_record)
            e.base.save_arrays(directory / 'representation_selected.npz', selected_probe)
            selection = e.inference.metric_summary(e.inference.evaluate(model, ds,
                plan['selection']['indices']['incident_full'], 16, torch.device('cpu')))
            audit_record = e.inference.evaluate(model, ds, plan['audit']['indices']['incident_full'],
                                                 16, torch.device('cpu'))
            e.base.save_arrays(directory / 'audit_incident_full.npz', audit_record)
            summary['runs'][str(seed)][arm] = {
                'selected_epoch': epoch, 'trainable_parameters': sum(p.numel() for p in gate.parameters()),
                'selection_metrics': {'incident_full': selection},
                'audit': {'incident_full': e.inference.metric_summary(audit_record)},
                'representation_diagnostics': {
                    'initial': {**e.inference.metric_summary(initial_record), 'sample_ids': ds.ids[:2],
                                'adapter_state_sha256': initial_digest},
                    'selected': {**e.inference.metric_summary(selected_probe), 'sample_ids': ds.ids[:2],
                                 'adapter_state_sha256': selected_digest}}}
    def publish():
        summary['outputs'] = {str(path.relative_to(source)): e.base.sha256(path)
                              for path in source.rglob('*') if path.is_file() and path.name != 'summary.json'}
        save_json('summary.json', summary)
    publish()
    return protocol, reader, source, data, primary, secondary, checkpoint, ds, baseline, hashes, summary, publish


class FitAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.protocol, self.reader, self.source, self.data, self.primary, self.secondary,
         self.checkpoint, self.ds, self.baseline, self.hashes, self.summary, self.publish) = fixture(self.root)

    def run_audit(self, output):
        ds = self.ds
        class FitOnly:
            scaler, station_ids = ds.scaler, ds.station_ids
            def __getitem__(self, index):
                if index not in (0, 1) or torch.is_grad_enabled():
                    raise AssertionError('Only fit positives may enter no-grad new inference')
                return ds[index]
        with ExitStack() as context:
            context.enter_context(patch.object(e, 'load_protocol', return_value=self.protocol))
            context.enter_context(patch.object(e.transfer, 'load_protocol', return_value=self.reader))
            context.enter_context(patch.object(e.base.mechanisms, 'verify_inputs', return_value=(self.baseline, self.hashes)))
            context.enter_context(patch.object(e.base, 'FullPositiveDataset', return_value=FitOnly()))
            context.enter_context(patch.object(e.base, 'make_model', side_effect=lambda *a, **k: tiny_model()))
            context.enter_context(patch.object(e.base, 'make_datasets', side_effect=AssertionError('Control dataset loaded')))
            context.enter_context(patch.object(torch.optim, 'Adam', side_effect=AssertionError('Optimizer created')))
            context.enter_context(patch.object(torch.Tensor, 'backward', side_effect=AssertionError('Gradient computed')))
            context.enter_context(redirect_stdout(io.StringIO()))
            return e.run(self.source, self.data, self.primary, self.secondary, self.checkpoint, output, 'cpu')

    def test_complete_readonly_inference_all_selected_arms_and_original_fallbacks(self):
        before = {str(p): e.base.sha256(p) for p in self.root.rglob('*') if p.is_file()}
        output = self.root / 'output'
        summary = self.run_audit(output)
        self.assertEqual(summary['status'], 'STATE_INTERACTION_FULL_FIT_AUDIT_COMPLETE')
        self.assertFalse(summary['model_training_performed'])
        self.assertFalse(summary['gradient_computation_performed'])
        self.assertFalse(summary['optimizer_created'])
        self.assertFalse(summary['new_model_selection_performed'])
        self.assertFalse(summary['new_selection_period_inference_performed'])
        self.assertFalse(summary['new_audit_period_inference_performed'])
        self.assertFalse(summary['validation_arrays_read'])
        self.assertFalse(summary['test_split_read'])
        self.assertEqual(before, {str(p): e.base.sha256(p) for p in self.root.rglob('*') if str(p) in before})
        self.assertFalse(output.with_name('output.partial').exists())
        self.assertEqual(set(summary['results']), {'2025', '2026', '2027'})
        self.assertEqual(len(list(output.glob('fit_*.npz'))), 10)
        for seed in ('2025', '2026', '2027'):
            result = summary['results'][seed]
            self.assertEqual(result['phases']['fit']['samples'], 2)
            self.assertEqual(result['phases']['audit']['samples'], 4)
            self.assertEqual(result['selection_point_context']['weekly_intervals_status'], e.UNAVAILABLE)
            for arm in self.protocol['arms']:
                self.assertTrue(summary['selected_models'][seed][arm]['model_state_unchanged'])
                self.assertEqual(summary['saved_probe_replay'][seed][arm]['selected']['samples'], 2)
            if seed != '2027':
                self.assertEqual(result['phases']['fit']['regions']['all']['comparisons']['strength_vs_A']['gain_raw_mae'], 0.)
                self.assertTrue(summary['selected_models'][seed]['strength']['epoch_zero_full_fit_error_sums_exact_A'])
        self.assertFalse(any('last_gate' in name or 'val_flow' in name or 'test_flow' in name for name in summary['inputs']))
        for name, digest in summary['outputs'].items():
            self.assertEqual(e.base.sha256(output / name), digest)
        json.dumps(summary, allow_nan=False)
        capture = io.StringIO()
        with redirect_stdout(capture):
            e.report(summary)
        self.assertIn('NO NEW TRAINING', capture.getvalue())

    def test_selected_checkpoint_header_state_hash_and_strict_shapes(self):
        path = self.source / 'state_vector_s2025/selected_gate.pt'
        original = torch.load(path, weights_only=True)
        changes = [('seed', 2026), ('epoch', True), ('variant', 'interaction_vector'),
                   ('protocol_sha256', 'wrong'), ('backbone_state_sha256', 'wrong')]
        for i, (key, value) in enumerate(changes):
            with self.subTest(key=key):
                checkpoint = copy.deepcopy(original)
                checkpoint[key] = value
                torch.save(checkpoint, path)
                self.publish()
                with self.assertRaisesRegex(ValueError, 'checkpoint identity'):
                    self.run_audit(self.root / f'bad_header_{i}')
        checkpoint = copy.deepcopy(original)
        checkpoint['gate_state']['output.bias'] += .02
        torch.save(checkpoint, path)
        self.publish()
        with self.assertRaisesRegex(ValueError, 'state or parameter budget'):
            self.run_audit(self.root / 'bad_state')
        checkpoint = copy.deepcopy(original)
        checkpoint['gate_state']['output.bias'] = torch.ones(1)
        torch.save(checkpoint, path)
        self.publish()
        with self.assertRaisesRegex(RuntimeError, 'size mismatch'):
            self.run_audit(self.root / 'bad_shape')
        checkpoint = copy.deepcopy(original)
        checkpoint['gate_state']['output.bias'][0] = torch.nan
        torch.save(checkpoint, path)
        self.publish()
        with self.assertRaisesRegex(ValueError, 'invalid/nonfinite tensors'):
            self.run_audit(self.root / 'bad_nonfinite')

    def test_original_checkpoint_payload_is_rechecked_before_loading(self):
        self.checkpoint.write_bytes(b'Changed original backbone payload')
        with self.assertRaisesRegex(ValueError, 'Original A checkpoint hash'):
            self.run_audit(self.root / 'bad_A')

    def test_hash_mismatch_failure_is_atomic(self):
        (self.source / 'strength_s2025/selected_gate.pt').write_bytes(b'corrupted checkpoint')
        output = self.root / 'bad_hash'
        with self.assertRaisesRegex(ValueError, 'artifact hash mismatch'):
            self.run_audit(output)
        self.assertFalse(output.exists())
        self.assertTrue((self.root / 'bad_hash.partial/failure.json').is_file())

    def test_probe_replay_reports_tolerance_and_rejects_support_or_large_errors(self):
        source = e.transfer.Source(self.source, self.reader)
        probe = e.transfer.read_record(source, 'strength_s2025/representation_initial.npz', self.ds.ids[:2])
        full = copy.deepcopy(probe)
        full['errors'] += 1e-5
        result = e.probe_replay(full, probe, self.protocol['probe_replay_error_tolerance'])
        self.assertGreater(result['maximum_absolute_sample_region_error_sum_difference'], 0)
        self.assertEqual(result['full_fit_batch_size'], 16)
        self.assertEqual(result['saved_probe_batch_size'], 8)
        full['candidate_mask'][0, 0] = False
        with self.assertRaisesRegex(ValueError, 'alignment mismatch'):
            e.probe_replay(full, probe, self.protocol['probe_replay_error_tolerance'])
        full = copy.deepcopy(probe)
        full['errors'] += 1.
        with self.assertRaisesRegex(ValueError, 'saved-probe replay'):
            e.probe_replay(full, probe, self.protocol['probe_replay_error_tolerance'])

    def test_source_budget_runtime_and_input_fingerprints_are_rejected(self):
        self.summary['engineering_check'] = True
        self.publish()
        with self.assertRaisesRegex(ValueError, 'full-budget'):
            self.run_audit(self.root / 'check_source')
        self.summary['engineering_check'] = False
        self.publish()
        changed = copy.deepcopy(self.protocol)
        changed['runtime_source_identity_files'] += ['experiments/chronological/audit_architecture_regions.py']
        with self.assertRaisesRegex(ValueError, 'Inference implementation'):
            source = e.transfer.Source(self.source, self.reader)
            e.verify_runtime(source, changed)
        self.hashes[str(self.checkpoint)] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'input fingerprints'):
            self.run_audit(self.root / 'bad_input')

    def test_output_guards_and_incomplete_report(self):
        with self.assertRaisesRegex(ValueError, 'outside all read-only'):
            self.run_audit(self.data / 'nested')
        link = self.root / 'dangling'
        link.symlink_to(self.root / 'missing')
        with self.assertRaises(FileExistsError):
            self.run_audit(link)
        partial = self.root / 'old.partial'
        partial.mkdir()
        with self.assertRaises(FileExistsError):
            self.run_audit(self.root / 'old')
        result = subprocess.run([sys.executable, str(Path(e.__file__)), 'report', str(self.root / 'none.json')],
                                 cwd=e.REPO, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('INCOMPLETE', result.stdout)


if __name__ == '__main__':
    unittest.main()
