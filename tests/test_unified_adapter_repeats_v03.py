"""Small artifact fixtures; no model fitting, inference, or GPU access."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import torch

from experiments.chronological.original_igstgnn_runtime import CONFIG, RELEASE, sha256, tensor_tree_hash
from experiments.chronological.original_igstgnn_source import MANIFEST_SHA256
from experiments.chronological.train_unified_adapter_v03 import EXECUTION_FILES, PROTOCOL
from experiments.chronological.unified_incident_adapter_v03 import ADAPTER_CONFIG, DEFAULT_ARMS
from experiments.chronological.summarize_unified_adapter_v03 import (
    REPO, SCHEMA, load_run, main, summarize_runs)


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True), encoding='utf-8')


class RepeatedSeedTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)

    def make_run(self, seed, epochs=2, *, engineering=False, name=None, data_change=False):
        run = self.root/(name or str(seed))
        run.mkdir()
        train_count, val_count = (4, 4) if engineering else (3909, 837)
        partition = [min(48, train_count-i) for i in range(0, train_count, 48)]
        splits = dict(train=dict(count=train_count, selected_indices=list(range(train_count)), partition=partition),
                      val=dict(count=val_count, selected_indices=list(range(3909, 3909+val_count))))
        data = dict(hashes=RELEASE, splits=splits, normalization=dict(mean=1., std=2.))
        if data_change:
            data['normalization']['mean'] = 1.01
        initializations = {arm: dict(registered_parameters=1, marker=seed,
            original_initial_tensor_sha256=str(seed)*16, traffic_tensor_sha256=str(seed)*16,
            post_construction_rng_sha256=str(seed)*16,
            adapter_tensor_sha256=None if arm in ('ORIGINAL', 'TRAFFIC_CORE') else str(seed+1)*16)
            for arm in DEFAULT_ARMS}
        identity = dict(schema='unified_adapter_matched_training_v03',
            arms=list(DEFAULT_ARMS), training_config=dict(CONFIG, seed=seed),
            runtime=dict(python_device='cpu', threads=1, seed=seed), data=data,
            adapter_config=ADAPTER_CONFIG, source_fingerprint=MANIFEST_SHA256,
            execution_source={name: sha256(REPO/name) for name in EXECUTION_FILES},
            initializations=initializations, engineering_only=engineering, curriculum='original',
            holdout_rows=0, protocol=PROTOCOL, scheduler=dict(milestones=[1], gamma=.5),
            early_stopping_enabled=False, from_scratch=True, test_evaluated=False)
        identity['identity_sha256'] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        write_json(run/'identity.json', identity)
        write_json(run/'budget.json', dict(target_completed_epochs=epochs))
        gain = {2025: -.03, 2026: .02, 2027: .04}.get(seed, .01)
        offsets = dict(ORIGINAL=0., TRAFFIC_CORE=.2, H_ONLY=.1, R_ONLY=.08, BOTH=-gain, DIRECT=.05)
        for arm in DEFAULT_ARMS:
            directory = run/arm
            directory.mkdir()
            weights = {'fixture_weight': torch.tensor([float(seed)])}
            best_name = 'best.pt'
            torch.save(weights, directory/best_name)
            history = [dict(epoch=epoch, global_updates=epoch*len(partition),
                validation_mae=10.-.01*epoch+offsets[arm],
                validation_rmse=14.-.01*epoch+offsets[arm],
                train_order_sha256=hashlib.sha256((str(seed)+':'+str(epoch)).encode()).hexdigest(),
                validation_mape_fraction=(10.-.01*epoch+offsets[arm])/100) for epoch in range(1, epochs+1)]
            updates = epochs*len(partition)
            warm, step = CONFIG['warm_epoch']*len(partition), CONFIG['cl_epoch']*len(partition)
            length = 12 if updates < warm else min(12, 1+(updates-warm)//step)
            state = dict(identity_sha256=identity['identity_sha256'], arm=arm, completed_epoch=epochs,
                history=history, global_updates=updates, initialization=initializations[arm],
                train_order=list(range(train_count)), curriculum_length=length,
                scheduler_state=dict(last_epoch=epochs), model_state=weights,
                model_tensor_sha256=tensor_tree_hash(weights),
                best_epoch=epochs, best_validation_mae=history[-1]['validation_mae'], wait=0,
                best_checkpoint_file=best_name, best_checkpoint_sha256=sha256(directory/best_name))
            torch.save(state, directory/'last.pt')
        # A forged CLI summary must have no influence on computed gains.
        write_json(run/'summary.json', dict(status='PASS', claimed_gain=99999))
        return run

    def test_same_seed_rejected_even_in_different_directories(self):
        a = self.make_run(2025, name='first')
        b = self.make_run(2025, name='second')
        with self.assertRaisesRegex(ValueError, 'Repeated seed'):
            summarize_runs([a, b])

    def test_data_difference_rejected(self):
        a = self.make_run(2025)
        b = self.make_run(2026, data_change=True)
        with self.assertRaisesRegex(ValueError, 'data'):
            summarize_runs([a, b])

    def test_two_seeds_insufficient_and_negative_values_preserved(self):
        report = summarize_runs([self.make_run(2025), self.make_run(2026)])
        self.assertEqual(report['status'], 'V03_PAIRED_REPEATS_INSUFFICIENT_SEEDS')
        self.assertFalse(report['formal_budget_complete'])
        result = report['comparisons']['BOTH_vs_ORIGINAL']['independent_best']['mae_gain']
        self.assertEqual(result['positive_count'], 1)
        self.assertEqual(result['negative_count'], 1)
        self.assertAlmostEqual(result['mean'], -.005)
        self.assertLess(result['values'][0], 0)
        self.assertGreater(result['sample_std'], 0)
        self.assertTrue(all(not r['summary_metrics_trusted'] for r in report['runs']))

    def test_three_short_runs_descriptive_not_formal(self):
        report = summarize_runs([self.make_run(seed) for seed in (2025, 2026, 2027)])
        self.assertEqual(report['status'], 'V03_PAIRED_REPEATS_DESCRIPTIVE_ONLY')
        self.assertFalse(report['formal_budget_complete'])
        self.assertEqual(report['common_compared_completed_epochs'], 2)
        self.assertEqual(report['comparisons']['BOTH_vs_DIRECT']['common_last']['mae_gain']['count'], 3)

    def test_engineering_only_never_formal_even_with_100_artifact_epochs(self):
        report = summarize_runs([self.make_run(seed, 100, engineering=True) for seed in (2025, 2026, 2027)])
        self.assertFalse(report['formal_budget_complete'])
        self.assertIn('engineering_only_or_missing_engineering_declaration', report['formal_ineligibility_reasons'])

    def test_three_full_registered_budgets_qualify_only_as_budget_complete(self):
        report = summarize_runs([self.make_run(seed, 100) for seed in (2025, 2026, 2027)])
        self.assertTrue(report['formal_budget_complete'])
        self.assertEqual(report['status'], 'V03_PAIRED_REPEATS_FORMAL_BUDGET_COMPLETE')
        self.assertFalse(report['exceeds_baseline_confirmed'])
        self.assertFalse(report['independent_event_generalization_certified'])
        self.assertFalse(report['statistical_significance_claimed'])

    def test_common_budget_prefix_prevents_comparing_unequal_winners(self):
        report = summarize_runs([self.make_run(2025, 2), self.make_run(2026, 3), self.make_run(2027, 4)])
        self.assertEqual(report['common_compared_completed_epochs'], 2)
        rows = report['comparisons']['BOTH_vs_ORIGINAL']['paired']
        self.assertTrue(all(row['compared_completed_epochs'] == 2 for row in rows))
        self.assertTrue(all(set(row['best_epochs'].values()) == {2} for row in rows))

    def test_identity_checksum_and_source_hash_are_checked(self):
        run = self.make_run(2025)
        path = run/'identity.json'
        identity = json.loads(path.read_text())
        identity['training_config']['seed'] = 99
        write_json(path, identity)
        with self.assertRaisesRegex(ValueError, 'checksum'):
            load_run(run)
        identity['training_config']['seed'] = 2025
        identity['execution_source'][EXECUTION_FILES[0]] = '0'*64
        identity['identity_sha256'] = hashlib.sha256(json.dumps(
            {k:v for k,v in identity.items() if k != 'identity_sha256'}, sort_keys=True).encode()).hexdigest()
        write_json(path, identity)
        with self.assertRaisesRegex(ValueError, 'source hash'):
            load_run(run)

    def test_best_role_checksum_and_last_tensor_checksum_are_checked(self):
        run = self.make_run(2025)
        (run/'BOTH'/'best.pt').write_bytes(b'corrupt checkpoint')
        with self.assertRaisesRegex(ValueError, 'best checkpoint'):
            load_run(run)
        run = self.make_run(2026)
        path = run/'BOTH'/'last.pt'
        state = torch.load(path, weights_only=True)
        state['model_state']['fixture_weight'][0] += 1
        torch.save(state, path)
        with self.assertRaisesRegex(ValueError, 'model tensors'):
            load_run(run)

    def test_missing_registered_arm_cannot_qualify(self):
        dirs = [self.make_run(seed, 100) for seed in (2025, 2026, 2027)]
        (dirs[-1]/'DIRECT'/'last.pt').unlink()
        report = summarize_runs(dirs)
        self.assertFalse(report['formal_budget_complete'])
        self.assertEqual(report['common_compared_completed_epochs'], 0)

    def test_same_seed_arm_training_order_must_be_paired(self):
        run = self.make_run(2025)
        path = run/'BOTH'/'last.pt'
        state = torch.load(path, weights_only=True)
        state['history'][0]['train_order_sha256'] = 'f'*64
        torch.save(state, path)
        with self.assertRaisesRegex(ValueError, 'training order'):
            load_run(run)

    def test_negative_error_metric_is_rejected_but_negative_gains_are_allowed(self):
        run = self.make_run(2025)
        path = run/'BOTH'/'last.pt'
        state = torch.load(path, weights_only=True)
        state['history'][0]['validation_rmse'] = -1
        torch.save(state, path)
        with self.assertRaisesRegex(ValueError, 'cannot be negative'):
            load_run(run)

    def test_three_unregistered_seeds_are_not_the_prespecified_replication(self):
        report = summarize_runs([self.make_run(seed) for seed in (2025, 2026, 2030)])
        self.assertFalse(report['formal_budget_complete'])
        self.assertIn('requires_exactly_preregistered_seeds_2025_2026_2027', report['formal_ineligibility_reasons'])

    def test_cli_outputs_only_to_fresh_directory(self):
        dirs = [self.make_run(2025)]
        output = self.root/'output'
        result = main(['--run-dirs', str(dirs[0]), '--output-dir', str(output)])
        self.assertEqual(result['schema'], SCHEMA)
        self.assertTrue((output/'summary.json').is_file())
        self.assertTrue((output/'report.md').is_file())
        with self.assertRaisesRegex(ValueError, 'fresh output'):
            main(['--run-dirs', str(dirs[0]), '--output-dir', str(output)])


if __name__ == '__main__':
    unittest.main()
