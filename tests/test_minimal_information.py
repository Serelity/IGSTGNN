import copy
import csv
import json
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.chronological import train_minimal_information as train
from experiments.chronological.minimal_information_data import DevelopmentData, json_hash, verify_sources
from experiments.chronological.minimal_information_evaluation import compare_all, region_statistics
from src.models.minimal_information import MODELS, forward_inputs


def tiny_package(root):
    """Synthetic chronological raw files with matched controls, no legacy scaler."""
    dirs = [root / name for name in ('positive', 'primary', 'secondary')]
    for path in dirs:
        path.mkdir()
    protocol = copy.deepcopy(train.load_protocol())
    protocol.update(nodes=3, source_counts=dict(positive=9, primary=9, secondary=9),
        frozen_only_candidate_pairs=[], seeds=[7],
        architecture=dict(hidden=8, node_hidden=4, time_hidden=4, layers=5, dropout=.1),
        periods={'fit': ['2023-01-02T00:00:00', '2023-01-30T00:00:00'],
                 'selection': ['2023-02-06T00:00:00', '2023-03-06T00:00:00'],
                 'audit': ['2023-03-13T00:00:00', '2023-05-15T00:00:00']},
        expected_phase_samples={'fit': {'incident_full': 3},
            'selection': {name: 3 for name in train.COHORTS}, 'audit': {name: 3 for name in train.COHORTS}})
    protocol['training'].update(epochs=2, batch_size=2, evaluation_batch_size=2, lr_milestones=[1], progress_every_batches=100)
    protocol['check'].update(seed=7, samples_per_phase=2, epochs=2)
    protocol['pilot'].update(seed=7)
    protocol['evaluation'].update(minimum_windows=2, minimum_nonempty_weeks=2)
    protocol['bootstrap'].update(draws=100, block_weeks=2, minimum_valid_fraction=.8)
    positive, primary, secondary = [], [], []
    times = [datetime.fromisoformat(protocol['periods'][phase][0]) + timedelta(days=1 + 7 * i, hours=12)
             for phase in ('fit', 'selection', 'audit') for i in range(3)]

    def window(t):
        return {key: (t + timedelta(minutes=offset)).isoformat() for key, offset in
                (('x_start', -65), ('x_end', -10), ('y_start', 5), ('y_end', 60),
                 ('support_start', -70), ('support_end_exclusive', 65))}

    for i, stamp in enumerate(times):
        row = dict(sample_index=str(i + 10), incident_id=str(i + 100), split='train', source_version='8',
                   t0=stamp.isoformat(), report_time=(stamp - timedelta(minutes=i % 5 + 1)).isoformat(), **window(stamp))
        positive.append(row)
        for collection, delta in ((primary, 1), (secondary, 2)):
            t = stamp + timedelta(days=delta)
            collection.append(dict(control_index=str(i), split='train', source_version='8', positive_sample_index=str(i + 10),
                incident_id=row['incident_id'], positive_t0=row['t0'], candidate_t0=t.isoformat(), **window(t)))
    for directory, filename, rows in ((dirs[0], 'train_manifest.csv', positive),
        (dirs[1], 'train_control_manifest.csv', primary), (dirs[2], 'train_second_control_manifest.csv', secondary)):
        with (directory / filename).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    rng = np.random.default_rng(19)
    values = rng.integers(0, 100, size=(9, 26, 3)).astype(np.float32)
    for directory, filename, shift in ((dirs[0], 'train_flow.npy', 0), (dirs[1], 'train_control_flow.npy', 2),
                                      (dirs[2], 'train_second_control_flow.npy', 4)):
        np.save(directory / filename, values + shift)
    stations = np.array([11, 22, 33])
    distances = np.zeros((9, 3, 3), dtype=np.float32)
    distances[:, :2, 1] = [.7, .4]
    distances[:, 0, 2] = 1
    np.save(dirs[0] / 'station_ids.npy', stations)
    np.save(dirs[0] / 'adjacency.npy', np.ones((3, 3), dtype=np.float32) - np.eye(3, dtype=np.float32))
    np.savez(dirs[0] / 'train_context.npz', station_ids=stations, sample_indices=np.arange(10, 19),
             distances=distances, report_age_minutes=np.array([i % 5 + 1 for i in range(9)], np.float32),
             forecast_tod=np.array([144] * 9), forecast_dow=np.array([(t.weekday() + 1) % 7 for t in times]))
    for directory, filename in ((dirs[1], 'train_affected_mask.npy'), (dirs[2], 'train_second_affected_mask.npy')):
        np.save(directory / filename, np.any(distances != 0, -1))
    return dirs, protocol


def silence(*args, **kwargs):
    pass


def unverified_synthetic(*args):
    return {'fixture': 'synthetic_test_only'}


class DataBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.dirs, self.protocol = tiny_package(self.root)
        self.data = DevelopmentData(*self.dirs, self.protocol)

    def tearDown(self):
        self.temp.cleanup()

    def test_scaler_unique_slots_real_zero_and_no_future_fallback(self):
        data = self.data
        data.flow = np.array(data.flow)
        data.flow[0, 0, 0] = 0
        data.flow[0, 1, 0] = np.nan
        data.rows[1]['x_start'] = data.rows[0]['x_start']
        data.flow[1, :12] = data.flow[0, :12]
        first = data.build_scaler()
        self.assertEqual(first['unique_slots'], 24)
        unique = np.concatenate([data.flow[0, :12], data.flow[2, :12]])
        valid = np.isfinite(unique) & (unique >= 0)
        self.assertAlmostEqual(first['mean'], float(unique[valid].astype(np.float64).mean()))
        self.assertGreater(first['true_zero_count'], 0)
        data.flow[:, 12:] = 999999
        data.flow[3:, :12] = 888888
        self.assertEqual(first, data.build_scaler())
        data.flow[1, 0, 1] += 1
        with self.assertRaisesRegex(ValueError, 'Conflicting duplicate'):
            data.build_scaler()
        data.flow[1, :12] = data.flow[0, :12]
        data.flow[:3, :12, 2] = np.nan
        with self.assertRaisesRegex(ValueError, 'No valid fit X'):
            data.build_scaler()

    def test_m0_inputs_exclude_reports_targets_and_candidate_mask(self):
        data = self.data
        data.build_scaler()
        first = data.batch('fit', 'incident_full', [0, 1], 'M0', targets=False)
        self.assertEqual(set(first), {'x', 'clock', 'sample_id', 'source_index'})
        data.context['distances'][:] = 999
        data.context['report_age_minutes'][:] = 999
        data.context['forecast_tod'][:] = 0
        data.flow = np.array(data.flow)
        data.flow[:, 12:] = -99
        second = data.batch('fit', 'incident_full', [0, 1], 'M0', targets=False)
        for key in first:
            np.testing.assert_array_equal(first[key], second[key])
        self.assertEqual(len(forward_inputs('M0', second, 'cpu')), 2)

    def test_phase_qualification_full_support_and_audit_lock(self):
        self.data.build_scaler()
        with self.assertRaisesRegex(ValueError, 'outside declared phase'):
            self.data.batch('fit', 'incident_full', [3], 'M2')
        with self.assertRaisesRegex(ValueError, 'Audit locked'):
            self.data.batch('audit', 'incident_full', [6], 'M0')
        from experiments.chronological.minimal_information_data import inside
        row = self.data.rows[0]
        self.assertFalse(inside(row, [row['t0'], '2023-01-30T00:00:00']))
        control = self.data.secondary_rows[3]
        self.assertFalse(inside(control, [control['candidate_t0'], '2023-03-06T00:00:00'], True))

    def test_controls_use_new_scaler_own_clock_and_paired_location(self):
        self.data.build_scaler()
        first = self.data.batch('selection', 'incident', [3], 'M2')
        second = self.data.batch('selection', 'secondary_control', [3], 'M2')
        np.testing.assert_array_equal(first['distances'], second['distances'])
        np.testing.assert_array_equal(first['age'], second['age'])
        self.assertNotEqual(first['clock'][0, 1], second['clock'][0, 1])
        self.assertFalse(np.array_equal(first['x'], second['x']))
        self.assertFalse((self.dirs[0] / 'scaler.json').exists())

    def test_source_allowlist_never_opens_old_scaler_val_test_or_checkpoint(self):
        from experiments.chronological import minimal_information_data as source
        base_path = source.REPO / 'experiments/chronological/incident_branch_materialize_v6a.json'
        base = json.loads(base_path.read_text())
        positive = {'build_complete': True, 'test_flow_built': False,
                    'files': {name: name for name in ('train_flow.npy', 'train_manifest.csv', 'station_ids.npy')}}
        context = {'schema': 'report_location_v1', 'outputs': {name: name for name in ('train_context.npz', 'adjacency.npy')}}
        (self.dirs[0] / 'summary.json').write_text(json.dumps(positive))
        (self.dirs[0] / 'context_manifest.json').write_text(json.dumps(context))
        seen = []

        def fake_sha(path):
            path = Path(path)
            seen.append(path.name)
            if path == base_path:
                return self.protocol['data_contract_sha256']
            if path.parent == self.dirs[0]:
                if path.name == 'summary.json':
                    return base['positive_package']['summary_sha256']
                if path.name == 'context_manifest.json':
                    return base['positive_package']['context_manifest_sha256']
                return path.name
            return base['primary_control_inputs' if path.parent == self.dirs[1] else 'secondary_control_inputs'][path.name]

        with patch.object(source, 'sha256', side_effect=fake_sha):
            verify_sources(*self.dirs, self.protocol)
        self.assertFalse(any(name.startswith(('val_', 'test_')) or name in ('scaler.json', 'best_model.pt') for name in seen))
        with patch.object(source, 'sha256', return_value='corrupted'):
            with self.assertRaisesRegex(ValueError, 'Data contract changed'):
                verify_sources(*self.dirs, self.protocol)


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.arch = dict(hidden=8, node_hidden=4, time_hidden=4, layers=5, dropout=.1)
        self.adj = np.ones((3, 3), np.float32) - np.eye(3, dtype=np.float32)
        self.batch = {'x': np.ones((2, 12, 3, 3), np.float32) * .2,
                      'clock': np.array([[2, 1], [3, 2]], np.int64),
                      'distances': np.ones((2, 3, 3), np.float32), 'age': np.array([1, 5], np.float32)}

    def test_equal_initial_weights_and_zero_input_equivalence(self):
        hashes, outputs, counts = [], [], []
        zero = dict(self.batch, distances=np.zeros((2, 3, 3), np.float32), age=np.zeros(2, np.float32))
        for arm in MODELS:
            train.seed_all(7)
            model = MODELS[arm](self.adj, self.arch).eval()
            self.assertFalse(hasattr(model, 'icsf_module'))
            self.assertFalse(hasattr(model, 'tiid_module'))
            hashes.append(train.tree_hash(model.state_dict()))
            counts.append(model.capacity(arm)['nominal_trainable_parameters'])
            with torch.no_grad():
                outputs.append(model(*forward_inputs(arm, zero, 'cpu')))
        self.assertEqual(len(set(hashes)), 1)
        self.assertEqual(len(set(counts)), 1)
        self.assertTrue(all(torch.equal(outputs[0], value) for value in outputs[1:]))

    def test_report_perturbations_cannot_reach_m0_or_age_reach_m1(self):
        for arm, field in (('M0', 'distances'), ('M0', 'age'), ('M1', 'age')):
            model = MODELS[arm](self.adj, self.arch).eval()
            changed = copy.deepcopy(self.batch)
            changed[field] *= 100
            changed['y'] = np.zeros((2, 12, 3))
            changed['candidate'] = np.zeros((2, 3), bool)
            with torch.no_grad():
                a = model(*forward_inputs(arm, self.batch, 'cpu'))
                b = model(*forward_inputs(arm, changed, 'cpu'))
            self.assertTrue(torch.equal(a, b))
        m0 = MODELS['M0'](self.adj, self.arch)
        with self.assertRaises(TypeError):
            m0(*forward_inputs('M2', self.batch, 'cpu'))

    def test_information_inputs_have_gradients_and_graph_caches_move(self):
        model = MODELS['M2'](self.adj, self.arch).eval()
        inputs = list(forward_inputs('M2', self.batch, 'cpu'))
        inputs[2].requires_grad_(True)
        inputs[3].requires_grad_(True)
        model(*inputs).square().mean().backward()
        self.assertGreater(float(inputs[2].grad.abs().sum()), 0)
        self.assertGreater(float(inputs[3].grad.abs().sum()), 0)
        model.double()
        for layer in model.layers:
            self.assertTrue(all(graph.dtype == torch.float64 for graph in layer.dif_layer.localized_st_conv.pre_defined_graph))
        self.assertTrue(all(graph.dtype == torch.float64 for graph in model._model_args['adjs']))


class EvaluationTests(unittest.TestCase):
    def test_regions_partition_real_zero_is_valid_and_nan_predictions_fail(self):
        batch = {'y': np.zeros((2, 12, 3)), 'valid': np.ones((2, 12, 3), bool),
                 'candidate': np.array([[True, False, False], [False, True, False]])}
        error, count = region_statistics(np.ones_like(batch['y']), batch)
        np.testing.assert_array_equal(count[:, 0], count[:, 1:].sum(1))
        np.testing.assert_array_equal(error, count)
        batch['valid'][0, 0, 0] = False
        error, count = region_statistics(np.ones_like(batch['y']), batch)
        self.assertEqual(count[0, 0], 35)
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            region_statistics(np.full_like(batch['y'], np.nan), batch)

    def test_paired_ci_gain_direction_and_insufficient_support(self):
        p = copy.deepcopy(train.load_protocol())
        p['seeds'] = [7, 8, 9]
        p['bootstrap']['draws'] = 100
        times = {i: (datetime(2023, 7, 3) + timedelta(days=i)).isoformat() for i in range(63)}
        records = {}
        for seed in p['seeds']:
            records[seed] = {}
            for arm, error in (('M0', 10.), ('M1', 9.), ('M2', 8.)):
                records[seed][arm] = {cohort: dict(ids=np.arange(63), source_indices=np.arange(63),
                    counts=np.ones((63, 4), np.int64), errors=np.full((63, 4), error)) for cohort in train.COHORTS}
        result = compare_all(records, times, p)
        self.assertTrue(all(value['status'] == 'SUPPORTED_DEVELOPMENT_INCREMENT' for value in result['comparisons'].values()))
        for row in result['rows']:
            self.assertEqual(row['gain'], 1)
            self.assertEqual(row['week_low'], 1)
            self.assertEqual(row['four_week_block_low'], 1)
        records[9]['M2']['primary_control']['errors'] *= 2
        result = compare_all(records, times, p)
        self.assertEqual(result['comparisons']['M2-M1']['status'], 'NOT_SUPPORTED')
        p['evaluation']['minimum_windows'] = 100
        result = compare_all(records, times, p)
        self.assertEqual(result['comparisons']['M1-M0']['status'], 'INSUFFICIENT_SUPPORT')
        records[7]['M1']['incident_full']['ids'][0] = 999
        with self.assertRaisesRegex(ValueError, 'Paired ids'):
            compare_all(records, times, p)


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.dirs, self.protocol = tiny_package(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def run_tiny(self, output, **kwargs):
        return train.run(*self.dirs, self.root / output, device='cpu', protocol=self.protocol,
                         verifier=unverified_synthetic, producer=unverified_synthetic,
                         progress=kwargs.pop('progress', silence), **kwargs)

    def test_check_and_pilot_train_backbone_without_audit(self):
        for mode in ('check', 'pilot'):
            with patch.object(train, 'evaluate_selected', side_effect=AssertionError('Audit must stay closed')):
                result = self.run_tiny(mode, mode=mode)
            self.assertTrue(result['backbone_retrained'])
            self.assertGreater(result['optimizer_steps'], 0)
            self.assertFalse(result['audit_targets_evaluated'])
            self.assertEqual(len({e['initial_state_sha256'] for e in result['endpoints'].values()}), 1)
            self.assertEqual(len(result['endpoints']), 3)
            for key in result['endpoints']:
                history = json.loads((self.root / mode / key / 'history.json').read_text())
                self.assertTrue(all(row['training']['backbone_embedding_changed'] for row in history))
        with self.assertRaisesRegex(ValueError, 'Preserve existing'):
            self.run_tiny('check', mode='check')

    def test_full_workflow_freezes_every_endpoint_before_audit(self):
        original = train.evaluate_selected
        observed = []

        def guarded(output, data, endpoints, *args):
            self.assertTrue(data.audit_unlocked)
            self.assertEqual(len(json.loads((output / 'frozen_endpoints.json').read_text())), 3)
            observed.append(True)
            return original(output, data, endpoints, *args)

        with patch.object(train, 'evaluate_selected', side_effect=guarded):
            result = self.run_tiny('full', mode='run')
        self.assertEqual(result['status'], 'V13B_DEVELOPMENT_COMPARISON_COMPLETE')
        self.assertEqual(result['optimizer_steps'], 12)
        self.assertEqual(observed, [True])
        self.assertTrue((self.root / 'full/paired_comparisons.csv').is_file())
        self.assertTrue((self.root / 'full/weekly_statistics.csv').is_file())

    def test_epoch_resume_matches_uninterrupted_and_source_unchanged(self):
        expected = self.run_tiny('reference', mode='run')

        def interrupt(stage, **fields):
            if stage == 'epoch_complete' and fields['arm'] == 'M1' and fields['epoch'] == 1:
                raise RuntimeError('simulated job interruption')

        with self.assertRaisesRegex(RuntimeError, 'simulated job interruption'):
            self.run_tiny('interrupted', mode='run', progress=interrupt)
        source = self.root / 'interrupted.partial'
        hashes = {str(p): train.sha256(p) for p in source.rglob('*') if p.is_file()}
        resumed = self.run_tiny('resumed', mode='run', resume_from=source)
        self.assertEqual(hashes, {str(p): train.sha256(p) for p in source.rglob('*') if p.is_file()})
        self.assertEqual(resumed['new_optimizer_steps'], 6)
        for key in expected['endpoints']:
            self.assertEqual(expected['endpoints'][key]['selected_state_sha256'], resumed['endpoints'][key]['selected_state_sha256'])
            for filename in ('audit_incident_full.csv', 'audit_primary_control.csv'):
                self.assertEqual((self.root / 'reference' / key / filename).read_bytes(),
                                 (self.root / 'resumed' / key / filename).read_bytes())
        self.assertEqual((self.root / 'reference/paired_comparisons.csv').read_bytes(),
                         (self.root / 'resumed/paired_comparisons.csv').read_bytes())
        identity = json.loads((source / 'run_identity.json').read_text())
        identity['mode'] = 'check'
        (source / 'run_identity.json').write_text(json.dumps(identity))
        with self.assertRaisesRegex(ValueError, 'Recovery run identity'):
            self.run_tiny('wrong_identity', mode='run', resume_from=source)

    def test_seal_and_historical_selection_corruption_rejected(self):
        result = self.run_tiny('source', mode='check')
        key = next(iter(result['endpoints']))
        path = self.root / 'source' / key / 'last.pt'
        saved = torch.load(path, weights_only=True)
        saved['payload']['epoch'] = 500
        torch.save(saved, path)
        with self.assertRaisesRegex(ValueError, 'integrity'):
            train.load_sealed(path)
        saved['payload']['epoch'] = 2
        saved['payload']['best']['epoch'] = 100
        train.save_sealed(path, saved['payload'])
        checked = train.load_sealed(path)
        with self.assertRaisesRegex(ValueError, 'selected state/history'):
            train.restore_epoch(checked, checked['identity'], [0, 1], checked['identity']['settings'])


if __name__ == '__main__':
    unittest.main()
