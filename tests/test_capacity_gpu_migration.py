"""GPU-name-only migration on controlled CPU fixtures with synthetic GPU metadata."""
import copy
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import torch

from experiments.chronological import continue_incident_capacity as continuation
from experiments.chronological import resume_incident_capacity_checked as checked
from experiments.chronological import train_incident_capacity as training
from src.utils.incident_corridor import read_json, sha256, write_json
from test_acdg import incident_batch, make_model
from test_incident_capacity_fusion import fixture
import test_capacity_training as training_tests


class CapacityGPUMigrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(3)
        cls.scratch = training.REPO/'experiments/chronological_runs'
        cls.scratch.mkdir(exist_ok=True)
        cls.temporary = tempfile.TemporaryDirectory(dir=cls.scratch)
        cls.template = Path(cls.temporary.name)
        # Only the metadata are synthetic; training/resume/state loading are real.
        cls.origin = dict(schema='capacity_training_m42_v1', device='cuda:0', seed=11,
                          check=True, batch_size=2, common_branch_sha256='fixed',
                          torch_version='fixed', inputs_sha256={'data': 'fixed'}, source_sha256={'source': 'fixed'},
                          environment=dict(gpu='Tesla V100-SXM2-32GB', python='fixed', cuda='fixed'))
        cls.current = dict(cls.origin, environment=dict(cls.origin['environment'], gpu='Tesla V100-PCIE-32GB'))
        cls.protocol = dict(read_json(continuation.PROTOCOL), train_samples=4, val_samples=4, stations=3, batch_size=2)
        write_json(cls.template/'identity.json', cls.origin)
        model, inputs = cls.build()
        training.train_arm(model, inputs, 'P1', cls.template, dict(cls.origin, arm='P1'),
                           cls.protocol, 11, 1, 2, 'cpu', True, False)
        cls.baseline = cls.template/'uninterrupted'
        cls.baseline.mkdir()
        shutil.copytree(cls.template/'P1', cls.baseline/'P1')
        model, inputs = cls.build()
        training.train_arm(model, inputs, 'P1', cls.baseline, dict(cls.origin, arm='P1'),
                           cls.protocol, 11, 2, 2, 'cpu', True, True)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    @classmethod
    def build(cls):
        torch.manual_seed(31337)
        branch, capacity = fixture()
        backbone = make_model()
        model = training.build_arm(backbone, branch.state_dict(), branch.graph, branch.outgoing_weights, 'P1', 11)
        batch = dict(x=torch.rand(2, 12, 3, 3), incident=incident_batch(), capacity_inputs=capacity)
        inputs = training_tests.CapacityTrainingTests().fake_inputs(batch)
        return model, inputs

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=self.scratch)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        shutil.copyfile(self.template/'identity.json', self.root/'identity.json')
        shutil.copytree(self.template/'P1', self.root/'P1')

    def hashes(self):
        return {str(p.relative_to(self.root)): sha256(p) for p in self.root.rglob('*')
                if p.is_file() and (p.parent.name == 'P1' or p.name == 'identity.json')}

    def run_checked(self, current=None, *, allow=True, audit=False, epochs=2, train=True, name='diagnostic.json'):
        current = self.current if current is None else current
        def main(argv):
            identity = copy.deepcopy(current)
            training.require(read_json(self.root/'identity.json') == identity, 'Run identity changed')
            if train:
                model, inputs = self.build()
                training.train_arm(model, inputs, 'P1', self.root, dict(identity, arm='P1'),
                                   self.protocol, 11, epochs, 2, 'cpu', True, True)
        args = ['--identity-report', str(self.root/name), '--resume', '--output-dir', str(self.root)]
        if allow:
            args.append('--allow-gpu-name-change')
        if audit:
            args.append('--identity-audit-only')
        with patch.object(training, 'main', main):
            checked.main(args)
        return read_json(self.root/name)

    def test_explicit_gpu_only_resume_preserves_origin_and_restores_optimizer_scheduler(self):
        origin_hash = sha256(self.root/'identity.json')
        before_hash = sha256(self.root/'P1/last_checkpoint.pt')
        report = self.run_checked()
        self.assertEqual(report['status'], 'RESUME_GPU_NAME_CHANGE_ACCEPTED')
        self.assertTrue(report['resume_accepted'])
        self.assertFalse(report['matching'])
        self.assertEqual(sha256(self.root/'identity.json'), origin_hash)
        last = torch.load(self.root/'P1/last_checkpoint.pt', weights_only=False)
        baseline = torch.load(self.baseline/'P1/last_checkpoint.pt', weights_only=False)
        self.assertEqual(last['identity'], dict(self.origin, arm='P1'))
        self.assertEqual((last['completed_epoch'], last['global_updates']), (2, 4))
        self.assertEqual(training.state_digest(last['model_state']), training.state_digest(baseline['model_state']))
        self.assertEqual(last['scheduler_state'], baseline['scheduler_state'])
        for key, value in last['optimizer_state']['state'].items():
            for name, tensor in value.items():
                self.assertTrue(torch.equal(tensor, baseline['optimizer_state']['state'][key][name]))
        segments = last['runtime_segments']
        self.assertEqual([(s['first_epoch'], s['last_epoch']) for s in segments], [(1, 1), (2, 2)])
        self.assertEqual(segments[-1]['environment']['gpu'], self.current['environment']['gpu'])
        self.assertEqual(segments[-1]['resume_checkpoint_sha256'], before_hash)
        self.assertEqual(read_json(self.root/'P1/summary.json')['runtime_segments'], segments)

    def test_gpu_name_change_remains_rejected_without_explicit_flag(self):
        before = self.hashes()
        with self.assertRaisesRegex(ValueError, 'Run identity changed'):
            self.run_checked(allow=False)
        self.assertEqual(before, self.hashes())

    def test_explicit_flag_cannot_admit_other_identity_differences(self):
        before = self.hashes()
        candidates = [dict(self.current, **{key: value}) for key, value in (
            ('seed', 12), ('common_branch_sha256', 'changed'), ('torch_version', 'changed'),
            ('inputs_sha256', {'data': 'changed'}), ('source_sha256', {'source': 'changed'}), ('device', 'cpu'))]
        candidates.append(dict(self.current, environment=dict(self.current['environment'], cuda='changed')))
        candidates.append(dict(self.current, environment=dict(self.current['environment'], gpu=None)))
        for i, candidate in enumerate(candidates):
            with self.subTest(i=i), self.assertRaisesRegex(ValueError, 'Run identity changed'):
                self.run_checked(candidate, train=False, name=f'reject_{i}.json')
        self.assertEqual(before, self.hashes())

    def test_audit_with_migration_flag_preserves_every_original_artifact(self):
        before = self.hashes()
        report = self.run_checked(audit=True)
        self.assertTrue(report['resume_accepted'])
        self.assertTrue(report['audit_only'])
        self.assertEqual(before, self.hashes())

    def test_hardware_segments_survive_repeated_resume_and_return_to_origin_gpu(self):
        self.run_checked()
        self.run_checked(epochs=3, name='second.json')
        self.run_checked(self.origin, allow=False, epochs=4, name='return.json')
        last = torch.load(self.root/'P1/last_checkpoint.pt', weights_only=False)
        self.assertEqual([(s['first_epoch'], s['last_epoch']) for s in last['runtime_segments']],
                         [(1, 1), (2, 2), (3, 3), (4, 4)])
        self.assertEqual([s['environment']['gpu'] for s in last['runtime_segments']],
                         [self.origin['environment']['gpu'], self.current['environment']['gpu'],
                          self.current['environment']['gpu'], self.origin['environment']['gpu']])
        checked.validate_runtime_segments(last['runtime_segments'], 4, self.origin)

    def test_gpu_flag_does_not_bypass_saved_arm_identity_or_runtime_corruption(self):
        path = self.root/'P1/last_checkpoint.pt'
        saved = torch.load(path, weights_only=False)
        saved['identity'] = dict(saved['identity'], seed=12)
        torch.save(saved, path)
        before = self.hashes()
        with self.assertRaisesRegex(ValueError, 'Saved arm identity changed'):
            self.run_checked()
        self.assertEqual(before, self.hashes())

    def test_runtime_validator_rejects_overlap_missing_epochs_and_version_change(self):
        origin = dict(first_epoch=1, last_epoch=1, environment=self.origin['environment'],
                      source='origin_identity', gpu_name_change_accepted=False)
        current = dict(first_epoch=2, last_epoch=2, environment=self.current['environment'],
                       source='checked_resume', gpu_name_change_accepted=True)
        for changed in (dict(current, first_epoch=1), dict(current, first_epoch=3),
                        dict(current, environment=dict(self.current['environment'], python='changed'))):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                checked.validate_runtime_segments([origin, changed], 2, self.origin)

    def test_parent_propagates_gpu_permission_only_when_explicit(self):
        directories = {key: self.root/key for key in ('data', 'history', 'network', 'reports')}
        args = (self.root, dict(self.origin, ramp_exchanges=True), directories, self.root/'sensors', 10)
        self.assertNotIn('--allow-gpu-name-change', continuation.trainer_command(*args))
        self.assertIn('--allow-gpu-name-change', continuation.trainer_command(*args, allow_gpu_name_change=True))

    def test_missing_gpu_metadata_never_becomes_an_allowed_name_change(self):
        saved = copy.deepcopy(self.origin)
        del saved['environment']['gpu']
        differences = checked.identity_differences(saved, self.current)
        self.assertEqual([r['field'] for r in differences], ['environment.gpu'])
        self.assertFalse(checked.gpu_name_only_change(saved, self.current, differences))

    def test_all_in_memory_hooks_are_restored_after_arm_failure(self):
        before = self.hashes()
        functions = (training.require, training.save_checkpoint, training.write_json)
        with patch.object(training, 'train_arm', side_effect=RuntimeError('controlled arm failure')) as arm:
            with self.assertRaisesRegex(RuntimeError, 'controlled arm failure'):
                self.run_checked()
            self.assertIs(training.train_arm, arm)
        self.assertEqual((training.require, training.save_checkpoint, training.write_json), functions)
        self.assertEqual(before, self.hashes())


if __name__ == '__main__':
    unittest.main()
