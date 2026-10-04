"""Actual paired gradients, temporal information firewall, recovery and artifact safety."""

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

from experiments.chronological import train_temporal_robust_correction as e
from test_incident_strength_gate import SyntheticDataset, manifests, tiny_model


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ds = SyntheticDataset()
        self.manifests = manifests()
        days = ['2023-01-10', '2023-02-07', '2023-03-07', '2023-04-04',
                '2023-05-16', '2023-06-13', '2023-07-04', '2023-08-01']
        for i, day in enumerate(days):
            t0 = datetime.fromisoformat(day + 'T12:00:00')
            for rows in self.manifests:
                for key in ('t0', 'positive_t0', 'candidate_t0'):
                    if key in rows[i]:
                        rows[i][key] = t0.isoformat()
                rows[i]['support_start'] = (t0 - timedelta(hours=1)).isoformat()
                rows[i]['support_end_exclusive'] = (t0 + timedelta(hours=2)).isoformat()
        self.frozen = e.load_protocol()
        self.frozen['seeds'] = [2025]
        self.frozen['training']['epochs'] = 2
        self.frozen['bootstrap']['draws'] = 40
        self.frozen['fit_groups']['expected_windows'] = [1] * 4
        self.protocol = e.base.load_protocol()
        self.plan = e.base.make_plan(*self.manifests, self.protocol)
        self.frozen['expected_phase_samples'] = {p: {c: len(v) for c, v in x['indices'].items()} for p, x in self.plan.items()}
        model = tiny_model()
        self.checkpoint = self.root / 'A.pt'
        torch.save(model.state_dict(), self.checkpoint)
        self.baseline = {'checkpoint': {'parameters': sum(p.numel() for p in model.parameters())}}
        self.future_delta = 0.

    def pipeline(self, output, resume=None, check=False):
        partial, owner = output.with_name(output.name + '.partial'), self
        class Guarded:
            scaler, station_ids = owner.ds.scaler, owner.ds.station_ids
            def __getitem__(self, index):
                if index >= 4 and torch.is_grad_enabled():
                    raise AssertionError('Nonfit Y entered optimization')
                if index >= 6:
                    manifest = partial / 'selected_endpoints_frozen.json'
                    if not manifest.is_file() or len(json.loads(manifest.read_text())) != 4:
                        raise AssertionError('Audit accessed before all paired endpoints froze')
                item = owner.ds[index]
                # Heterogeneous block targets to exercise nonuniform q and a real second-epoch difference.
                item['y_flow'] += [-3., 0., 2., -.5][index] if index < 4 else owner.future_delta
                return item
        class Control(Guarded):
            def __getitem__(self, index):
                if torch.is_grad_enabled():
                    raise AssertionError('Control Y entered optimization')
                return super().__getitem__(index)
        datasets = {'incident_full': Guarded(), **{c: Control() for c in e.base.COHORTS[1:]}}
        with ExitStack() as stack:
            for module, name, kwargs in (
                (e, 'load_protocol', {'return_value': self.frozen}),
                (e.base, 'load_protocol', {'return_value': self.protocol}),
                (e.base.mechanisms, 'verify_inputs', {'return_value': (self.baseline, {})}),
                (e.base, 'read_csv', {'side_effect': self.manifests}),
                (e.base, 'make_datasets', {'return_value': datasets}),
                (e.base, 'make_model', {'side_effect': lambda *a, **k: tiny_model()})):
                stack.enter_context(patch.object(module, name, **kwargs))
            stack.enter_context(redirect_stdout(io.StringIO()))
            return e.run(self.root / 'data', self.root / 'primary', self.root / 'secondary', self.checkpoint,
                         output, 'cpu', check, resume)

    def checkpoint_at(self, root, objective='temporal_excess', arm='state_vector'):
        return torch.load(root / f'{arm}_{objective}_s2025/last_adapter.pt', weights_only=True)

    def test_engineering_runs_four_groups_real_training_and_exact_pairs(self):
        result = self.pipeline(self.root / 'check', check=True)
        self.assertEqual(result['status'], 'ENGINEERING_CHECK_PASS')
        self.assertEqual(result['scientific_status'], 'NOT_EVALUATED_ENGINEERING_ONLY')
        self.assertEqual(result['fit_group_windows'], [1, 1, 1, 1])
        self.assertEqual(result['budget']['optimizer_steps'], 8)
        self.assertTrue(result['new_model_training_performed'])
        self.assertEqual(result['phase_comparisons'], {'2025': {}})
        for arm in e.method.ARMS:
            first = self.checkpoint_at(self.root / 'check', 'erm', arm)
            second = self.checkpoint_at(self.root / 'check', arm=arm)
            self.assertEqual(first['history'][0]['adapter_state_sha256'], second['history'][0]['adapter_state_sha256'])
            self.assertEqual(first['history'][0]['optimizer_sha256'], second['history'][0]['optimizer_sha256'])
            self.assertNotEqual(first['history'][1]['adapter_state_sha256'], second['history'][1]['adapter_state_sha256'])
            self.assertNotEqual(second['q_next'], second['identity']['pi'])
            self.assertEqual(first['q_next'], first['identity']['pi'])
            self.assertGreater(result['runs'][f'{arm}_temporal_excess_s2025']['device_checks']['gradient_forwards'], 0)

    def test_formal_synthetic_outputs_have_shared_statistics_and_geometry(self):
        result = self.pipeline(self.root / 'formal')
        self.assertFalse(result['engineering_check'])
        self.assertEqual(result['budget']['trajectory_epochs'], 8)
        self.assertTrue(result['all_endpoints_frozen_before_audit'])
        self.assertTrue(all(r['selected_epoch'] > 0 for r in result['runs'].values()))
        audit = result['phase_comparisons']['2025']['audit']
        self.assertEqual(len(audit['weeks']), 9)
        self.assertIn('state_vector__temporal_vs_erm', audit['results']['incident_full']['candidate_h1_h6']['comparisons'])
        for phases in result['geometry'].values():
            self.assertEqual(phases['audit']['matched']['triplets'], 2)
            for cohort in e.base.COHORTS:
                point = phases['audit'][cohort]['estimands']['pooled_valid_cells']['point']
                self.assertAlmostEqual(point['gain'], point['slope'] - point['crossing_penalty'])
        self.assertTrue((self.root / 'formal/comparisons.csv').is_file())

    def test_selection_and_audit_target_changes_do_not_change_training_weights(self):
        self.pipeline(self.root / 'first', check=True)
        self.future_delta = 1000.
        self.pipeline(self.root / 'changed', check=True)
        for objective in e.method.OBJECTIVES:
            a, b = [self.checkpoint_at(self.root / name, objective) for name in ('first', 'changed')]
            self.assertEqual(a['q_next'], b['q_next'])
            self.assertEqual(e.tree_hash(a['adapter_state']), e.tree_hash(b['adapter_state']))
            self.assertEqual(e.tree_hash(a['optimizer_state']), e.tree_hash(b['optimizer_state']))

    def interrupt_run(self, destination, epoch=1):
        actual = e.fit
        def interrupted(*args, **kwargs):
            args = list(args)
            progress = args[12]
            def fail(stage, **fields):
                progress(stage, **fields)
                if stage == 'epoch_complete' and fields['epoch'] == epoch and args[2] == 'temporal_excess':
                    raise RuntimeError('simulated interruption')
            args[12] = fail
            return actual(*args, **kwargs)
        with patch.object(e, 'fit', side_effect=interrupted), self.assertRaisesRegex(RuntimeError, 'simulated'):
            self.pipeline(destination)
        return destination.with_name(destination.name + '.partial')

    def test_resume_identical_current_best_optimizer_q_and_completed_trajectory(self):
        complete = self.pipeline(self.root / 'continuous')
        source = self.interrupt_run(self.root / 'broken')
        source_hashes = {str(p.relative_to(source)): e.base.sha256(p) for p in source.rglob('*') if p.is_file()}
        resumed = self.pipeline(self.root / 'resumed', resume=source)
        self.assertEqual(source_hashes, {str(p.relative_to(source)): e.base.sha256(p) for p in source.rglob('*') if p.is_file()})
        self.assertEqual(resumed['budget']['optimizer_steps_this_invocation'], 5)
        for arm in e.method.ARMS:
            for objective in e.method.OBJECTIVES:
                a = self.checkpoint_at(self.root / 'continuous', objective, arm)
                b = self.checkpoint_at(self.root / 'resumed', objective, arm)
                for key in ('adapter_state', 'optimizer_state', 'best', 'q_next', 'history', 'rng_cpu'):
                    self.assertEqual(e.tree_hash(a[key]), e.tree_hash(b[key]), key)
        self.assertEqual(complete['phase_comparisons'], resumed['phase_comparisons'])

    def test_epoch_zero_recovery_and_tampering_rejected(self):
        source = self.interrupt_run(self.root / 'zero_source')
        path = source / 'state_vector_temporal_excess_s2025/last_adapter.pt'
        original = torch.load(path, weights_only=True)
        effective = copy.deepcopy(self.frozen); effective['selection'] = self.protocol['selection']
        # Original initialization reconstructed with the exact DataLoader RNG consumption.
        e.base.set_seed(2025)
        model = tiny_model()  # tiny_model resets RNG; reseed again before probe as production does.
        e.base.set_seed(2025)
        raw = next(iter(e.base.loader(self.ds, [0,1,2,3], 2)))
        with torch.no_grad():
            model(raw['x'], incident_data=raw['incident'])
            adapter = e.vector.attach_adapter(model, 'state_vector', 16)
        initial = e.base.cpu_tree(adapter.state_dict())
        baseline = {c: e.original.metrics(e.original.evaluate(tiny_model(), self.ds, [4,5], 16, torch.device('cpu'))) for c in e.base.COHORTS}
        for tamper in ('q_next', 'history', 'optimizer', 'best', 'identity'):
            saved = copy.deepcopy(original)
            if tamper == 'q_next': saved['q_next'][0] += .01
            if tamper == 'history': saved['history'][0]['q_after'][0] += .01
            if tamper == 'optimizer': saved['optimizer_state']['state'][0]['exp_avg'].add_(.01)
            if tamper == 'best': saved['best']['state'][next(iter(initial))].add_(1)
            if tamper == 'identity': saved['identity']['objective'] = 'erm'
            torch.save(saved, path)
            with self.assertRaises(ValueError):
                e.restore_fit(path, original['identity'], initial, baseline, effective, original['identity']['reference'])
        # A checkpoint interrupted before any optimizer step is recoverable too.
        saved = copy.deepcopy(original)
        saved.update(epoch=0, history=[], adapter_state=initial, q_next=original['identity']['pi'])
        optimizer = torch.optim.Adam(adapter.parameters(), lr=.001, eps=1e-8, weight_decay=0.)
        saved['optimizer_state'] = optimizer.state_dict()
        saved['best'] = {'epoch': 0, 'selection_metrics': baseline, 'state': initial}
        torch.save(saved, path)
        restored = e.restore_fit(path, original['identity'], initial, baseline, effective, original['identity']['reference'])
        self.assertEqual(restored['epoch'], 0)

    def test_existing_output_partial_and_symlink_preserved(self):
        for suffix in ('', '.partial'):
            target = self.root / ('keep' + suffix)
            target.mkdir()
            (target / 'sentinel').write_text('preserve')
            with self.assertRaises(FileExistsError):
                self.pipeline(self.root / 'keep')
            self.assertEqual((target / 'sentinel').read_text(), 'preserve')
            (target / 'sentinel').unlink(); target.rmdir()
        (self.root / 'link').symlink_to(self.root / 'missing')
        with self.assertRaises(ValueError):
            self.pipeline(self.root / 'link')


if __name__ == '__main__':
    unittest.main()
