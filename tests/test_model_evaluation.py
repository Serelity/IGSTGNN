"""Offline evaluation contracts using synthetic artifacts only.

These tests distinguish the frozen horizon-macro metric from equally weighted
windows and verify that comparisons retain the same observations and counts.
No training data, model, accelerator, or sealed test split is needed.
"""

from datetime import datetime, timedelta
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experiments.chronological import model_evaluation_metrics as metrics


REPO = Path(__file__).resolve().parents[1]


def horizon_rows(values, *, dtype=np.float64):
    return np.repeat(np.asarray(values, dtype=dtype)[:, None], 12, axis=1)


def times_for(indices):
    origin = datetime(2026, 10, 1)
    return np.asarray([(origin + timedelta(minutes=5*int(i))).isoformat()
                       for i in indices])


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False), encoding='utf-8')


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode('utf-8')).hexdigest()


def synthetic_run(root, *, engineering=False, arms=None, package_tag='a'):
    """Build auditable 12-horizon JSON/NPZ fixtures, never real checkpoints."""
    arms = list(arms or ('B3R', 'B3R_HALF', 'B3R_GATE', 'B3R_NO_D'))
    root.mkdir(parents=True, exist_ok=True)
    data_hash, context_hash, graph_hash = package_tag*64, 'b'*64, 'c'*64
    protocol = dict(selection_metric='all_nodes.mae_macro', supervised_horizons=12,
                    arms=arms, train_samples=4, val_samples=3, stations=2,
                    batch_size=2, max_epochs=5, patience=20, min_delta=0.,
                    learning_rate=.002, weight_decay=1e-5, adam_eps=1e-8,
                    clip_grad_norm=5., lr_milestones=[1], lr_gamma=.5,
                    data_summary_sha256=data_hash, context_summary_sha256=context_hash,
                    graph_sha256=graph_hash, latent_channels=4, spatial_steps=2,
                    eta=.5, test_read=False, main_training_ready=False,
                    architecture_version='B3-R-G1.1',
                    scope='conditional_offline_screening')
    initialization = dict(status='B3RSTRENGTH_INITIALIZATION_PAIR_PASS',
                          arms={arm: {'common_initial_sha256': 'a'*64} for arm in arms},
                          shared_state_sha256='a'*64)
    write_json(root/'initialization.json', initialization)
    identity = dict(schema='b3r_strength_run_g11_v1', protocol=protocol,
                    protocol_sha256='d'*64, package={'summary.json': data_hash,
                                                    'context_manifest.json': context_hash},
                    seed=2025, engineering_check=engineering, batch_size=2,
                    threads=1, graph_sha256=graph_hash,
                    graph=dict(nodes=2, operator_nodes=2, edges=1, road_direction_groups=1,
                               evidence_scope='candidate_unverified'),
                    graph_inputs_sha256={'graph.json': graph_hash},
                    source_sha256={name: 'e'*64 for name in
                                   ('src/models/igstgnn.py', 'src/utils/chronological.py',
                                    'experiments/chronological/train.py', 'experiments/chronological/smoke.py',
                                    'src/base/model.py', 'src/models/incident_response.py',
                                    'src/models/incident_spatial.py')},
                    runtime=dict(torch='fixture', cuda=None, device_type='cpu'),
                    initialization_sha256=canonical_sha256(initialization),
                    test_read=False)
    write_json(root/'identity.json', identity)
    errors = {'B3R': ([5., 1., 3.], [2., 1., 3.]),
              'B3R_GATE': ([4., 1., 1.], [3., 2., 4.]),
              'B3R_HALF': ([5., 2., 2.], [3., 2., 1.]),
              'B3R_NO_D': ([5., 1., 3.], [3., 2., 4.])}
    indices = np.asarray([11, 7, 20], dtype=np.int64)
    t0 = times_for(indices)
    counts = np.full((3, 12), 2, dtype=np.int64)
    for arm in arms:
        directory = root/arm
        directory.mkdir()
        history, best, best_epoch = [], float('inf'), None
        for epoch, window_errors in enumerate(errors.get(arm, errors['B3R']), 1):
            absolute = horizon_rows(window_errors)*counts
            mae = float(absolute.sum(axis=0).mean()/counts.sum(axis=0)[0])
            improved = mae < best
            if improved:
                best, best_epoch = mae, epoch
            values = dict(mae_macro=mae, mae_pooled=mae, rmse_macro=mae+1.,
                          rmse_pooled=mae+1., mape_macro=.1, mape_pooled=.1,
                          per_horizon_mae=[mae]*12, per_horizon_rmse=[mae+1.]*12,
                          per_horizon_mape=[.1]*12, valid_count_per_horizon=[6]*12,
                          positive_target_count_per_horizon=[6]*12)
            history.append(dict(epoch=epoch, train={'metrics': {'all_nodes': values}},
                                validation={'metrics': {'all_nodes': values,
                                                        'report_associated': values,
                                                        'road:fixture': values},
                                            'mechanisms': {'d_strength_status': 'fixture'}},
                                validation_mae_macro=mae, best_metric=best,
                                best_epoch=best_epoch, improved=improved,
                                global_updates=2*epoch, epoch_updates=2,
                                learning_rate_used=.002 if epoch == 1 else .001,
                                order_sha256=str(epoch)*64,
                                seconds=1., peak_cuda_allocated_bytes=0))
            np.savez_compressed(directory/f'validation_windows_epoch_{epoch:03d}.npz',
                                absolute=absolute, counts=counts, sample_indices=indices, t0=t0)
        write_json(directory/'history.json', history)
        write_json(directory/'summary.json', dict(status='ENGINEERING_CHECK_PASS' if engineering
                                                  else 'PAUSED_AT_EPOCH_BOUNDARY', arm=arm,
                    completed_epoch=2, best_epoch=best_epoch, best_metric=best,
                    global_updates=4, early_stopped=False,
                    parameters={'total_parameters': 100, 'trainable_parameters': 100},
                    checkpoint='last.pt', best_model_state_key='best_model_state', test_read=False))
        (directory/'last.pt').write_bytes(b'Not a Torch checkpoint: evaluation must not load this file.')
    (root/'test_manifest.csv').write_text('SEALED TEST MUST NOT BE OPENED', encoding='utf-8')
    return root


class WindowMetricTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=REPO)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def save(self, name, absolute, counts, indices=None, t0=None):
        absolute, counts = np.asarray(absolute), np.asarray(counts)
        indices = np.arange(len(absolute), dtype=np.int64) if indices is None else np.asarray(indices)
        t0 = times_for(indices) if t0 is None else np.asarray(t0)
        path = self.root/name
        np.savez_compressed(path, absolute=absolute, counts=counts,
                            sample_indices=indices, t0=t0)
        return path

    def load(self, name, absolute, counts, indices=None, t0=None):
        return metrics.validate_windows(self.save(name, absolute, counts, indices, t0))

    def test_unequal_counts_reverse_window_and_main_metric_ranking(self):
        counts = horizon_rows([1, 9, 5], dtype=np.int64)
        control = self.load('control.npz', horizon_rows([10, 0, 10]), counts)
        candidate = self.load('candidate.npz', horizon_rows([8, 9, 5]), counts)
        summary = metrics.window_summary(control)
        self.assertAlmostEqual(summary['mae_macro'], 20/15)
        self.assertAlmostEqual(summary['window_mae']['mean'], 4.)
        self.assertNotAlmostEqual(summary['mae_macro'], summary['window_mae']['mean'])

        comparison, rows = metrics.compare_windows(control, candidate)
        self.assertAlmostEqual(comparison['mae_macro']['gain'], -2/15)
        self.assertGreater(comparison['window_gain_distribution']['mean'], 0.)
        expected = {0: 2/15, 1: -9/15, 2: 5/15}
        for row in rows:
            self.assertAlmostEqual(row['global_mae_gain_contribution'], expected[row['sample_index']])
        self.assertAlmostEqual(sum(r['global_mae_gain_contribution'] for r in rows), -2/15)
        self.assertAlmostEqual(comparison['global_contribution_sum'], -2/15)
        self.assertLessEqual(abs(comparison['additivity_error']), 1e-12)

    def test_horizon_macro_uses_separate_horizon_denominators(self):
        counts = np.ones((2, 12), dtype=np.int64)
        counts[:, 0] = [10, 1]
        absolute = horizon_rows([2., 6.])
        absolute[:, 0] = [10., 0.]
        summary = metrics.window_summary(self.load('different_h.npz', absolute, counts))
        expected_horizon = [10/11] + [4.]*11
        np.testing.assert_allclose(summary['per_horizon_mae'], expected_horizon)
        self.assertAlmostEqual(summary['mae_macro'], np.mean(expected_horizon))
        self.assertNotAlmostEqual(summary['mae_macro'], float(absolute.sum()/counts.sum()))

    def test_candidate_is_aligned_by_sample_id_instead_of_array_position(self):
        indices = np.asarray([7, 3, 10], dtype=np.int64)
        counts = horizon_rows([1, 2, 3], dtype=np.int64)
        control = self.load('control.npz', horizon_rows([2, 8, 18]), counts, indices)
        permutation = [2, 0, 1]
        candidate_values = horizon_rows([1, 6, 15])
        candidate = self.load('candidate.npz', candidate_values[permutation], counts[permutation],
                              indices[permutation])
        comparison, rows = metrics.compare_windows(control, candidate)
        self.assertAlmostEqual(comparison['mae_macro']['gain'], 1.)
        by_id = {row['sample_index']: row for row in rows}
        self.assertEqual(set(by_id), {3, 7, 10})
        for sample_index in indices:
            self.assertAlmostEqual(by_id[int(sample_index)]['window_mae_gain'], 1.)

    def test_counts_time_and_sample_set_mismatch_cannot_be_paired(self):
        counts = horizon_rows([1, 2], dtype=np.int64)
        absolute = horizon_rows([2, 4])
        control = self.load('control.npz', absolute, counts, [2, 4])
        changed_counts = counts.copy()
        changed_counts[0, 5] += 1
        changed_time = times_for([2, 4])
        changed_time[0] = '2026-10-01T01:00:00'
        variants = [
            self.load('changed_counts.npz', absolute, changed_counts, [2, 4]),
            self.load('changed_time.npz', absolute, counts, [2, 4], changed_time),
            self.load('changed_set.npz', absolute, counts, [2, 5]),
        ]
        for candidate in variants:
            with self.subTest(path=candidate['path']), self.assertRaises(ValueError):
                metrics.compare_windows(control, candidate)

    def test_fixed_difficult_windows_use_control_set_even_when_candidate_tail_differs(self):
        indices = np.arange(1, 21, dtype=np.int64)
        counts = np.ones((20, 12), dtype=np.int64)
        control = self.load('control.npz', horizon_rows(indices), counts, indices)
        candidate_errors = np.zeros(20)
        candidate_errors[:2] = [100., 90.]
        candidate = self.load('candidate.npz', horizon_rows(candidate_errors), counts, indices)
        comparison, rows = metrics.compare_windows(control, candidate)
        difficult = comparison['fixed_control_worst_10_percent']
        self.assertEqual(difficult['count'], 2)
        self.assertEqual(difficult['sample_indices'], [20, 19])
        self.assertAlmostEqual(difficult['control_window_mae_mean'], 19.5)
        self.assertAlmostEqual(difficult['candidate_window_mae_mean'], 0.)
        self.assertAlmostEqual(difficult['window_mae_gain_mean'], 19.5)
        self.assertEqual({r['sample_index'] for r in rows if r['in_control_worst_10_percent']}, {19, 20})
        self.assertEqual(metrics.window_summary(candidate)['window_mae']['top_5_percent']['sample_indices'], [1])

    def test_empty_window_is_unavailable_and_is_not_a_zero_error_win(self):
        counts = horizon_rows([0, 2, 1], dtype=np.int64)
        control = self.load('control.npz', horizon_rows([0, 4, 3]), counts)
        candidate = self.load('candidate.npz', horizon_rows([0, 2, 4]), counts)
        summary = metrics.window_summary(control)
        self.assertEqual(summary['window_mae']['available_count'], 2)
        self.assertEqual(summary['window_mae']['excluded_count'], 1)
        self.assertEqual(summary['window_mae']['empty_count'], 1)
        self.assertEqual(summary['window_mae']['partial_count'], 0)
        self.assertAlmostEqual(summary['window_mae']['mean'], 2.5)
        comparison, rows = metrics.compare_windows(control, candidate)
        empty = next(row for row in rows if row['sample_index'] == 0)
        self.assertFalse(empty['available'])
        self.assertIsNone(empty['control_window_mae'])
        self.assertIsNone(empty['candidate_window_mae'])
        self.assertIsNone(empty['window_mae_gain'])
        self.assertIsNone(empty['outcome'])
        self.assertEqual(empty['global_mae_gain_contribution'], 0.)
        self.assertEqual(comparison['window_outcomes']['available_count'], 2)
        self.assertEqual(comparison['window_outcomes']['excluded_count'], 1)
        self.assertEqual(comparison['window_outcomes']['wins'], 1)
        self.assertEqual(comparison['window_outcomes']['losses'], 1)
        json.dumps({'summary': summary, 'comparison': comparison, 'rows': rows}, allow_nan=False)

    def test_partial_window_keeps_global_contribution_but_has_no_window_score(self):
        counts = np.ones((2, 12), dtype=np.int64)
        counts[0, 0] = 0
        control_error, candidate_error = horizon_rows([3, 4]), horizon_rows([1, 4])
        control_error[0, 0] = candidate_error[0, 0] = 0
        control = self.load('partial_control.npz', control_error, counts)
        candidate = self.load('partial_candidate.npz', candidate_error, counts)
        comparison, rows = metrics.compare_windows(control, candidate)
        partial = next(row for row in rows if row['sample_index'] == 0)
        self.assertFalse(partial['available'])
        self.assertIsNone(partial['window_mae_gain'])
        self.assertAlmostEqual(partial['global_mae_gain_contribution'], 11/12)
        self.assertAlmostEqual(comparison['mae_macro']['gain'], 11/12)
        self.assertEqual(comparison['window_outcomes']['available_count'], 1)
        self.assertEqual(comparison['window_outcomes']['partial_count'], 1)
        self.assertEqual(comparison['window_outcomes']['empty_count'], 0)
        self.assertEqual(comparison['window_outcomes']['ties'], 1)

    def test_duplicate_times_are_recorded_without_claiming_independence(self):
        same_time = np.asarray(['2026-10-01T00:00:00']*3)
        windows = self.load('repeated_time.npz', horizon_rows([1, 2, 3]),
                            np.ones((3, 12), dtype=np.int64), t0=same_time)
        dependence = metrics.window_summary(windows)['time_dependence']
        self.assertEqual(dependence['unique_t0_count'], 1)
        self.assertEqual(dependence['duplicate_t0_rows'], 2)
        self.assertEqual(dependence['repeated_t0_count'], 1)
        self.assertEqual(dependence['maximum_windows_per_t0'], 3)
        self.assertFalse(dependence['independence_claim'])

    def test_validation_preserves_artifact_and_checks_expected_count(self):
        path = self.save('windows.npz', horizon_rows([1, 2]), np.ones((2, 12), dtype=np.int64))
        before = path.read_bytes()
        windows = metrics.validate_windows(path, expected_rows=2)
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(len(windows['sha256']), 64)
        with self.assertRaises(ValueError):
            metrics.validate_windows(path, expected_rows=3)

    def test_invalid_numeric_artifacts_are_rejected(self):
        absolute = np.ones((2, 12), dtype=np.float64)
        counts = np.ones((2, 12), dtype=np.int64)
        variants = []
        for value in (float('nan'), float('inf'), -1.):
            broken = absolute.copy()
            broken[0, 0] = value
            variants.append((f'absolute_{len(variants)}.npz', broken, counts))
        for value in (-1., 1.25, float('nan'), float('inf')):
            broken = counts.astype(np.float64)
            broken[0, 0] = value
            variants.append((f'counts_{len(variants)}.npz', absolute, broken))
        zero_horizon = counts.copy()
        zero_horizon[:, 3] = 0
        no_error_horizon = absolute.copy()
        no_error_horizon[:, 3] = 0
        variants.append(('empty_horizon.npz', no_error_horizon, zero_horizon))
        impossible_counts = counts.copy()
        impossible_counts[0, 0] = 0
        variants.append(('error_without_count.npz', absolute, impossible_counts))
        for name, errors, selected_counts in variants:
            with self.subTest(name=name), self.assertRaises(ValueError):
                metrics.validate_windows(self.save(name, errors, selected_counts))

    def test_object_arrays_duplicate_ids_and_wrong_axes_are_rejected(self):
        counts = np.ones((2, 12), dtype=np.int64)
        absolute = np.ones((2, 12), dtype=np.float64)
        paths = [
            self.save('object_errors.npz', absolute.astype(object), counts),
            self.save('object_counts.npz', absolute, counts.astype(object)),
            self.save('object_times.npz', absolute, counts, t0=times_for([0, 1]).astype(object)),
            self.save('duplicate_ids.npz', absolute, counts, [7, 7]),
            self.save('wrong_horizon.npz', absolute[:, :11], counts[:, :11]),
            self.save('wrong_counts_axis.npz', absolute, counts[:, :, None]),
        ]
        for path in paths:
            with self.subTest(path=path.name), self.assertRaises(ValueError):
                metrics.validate_windows(path)


class OfflineEvaluationRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=REPO)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def analyzer(self):
        from experiments.chronological import analyze_model_evaluation
        return analyze_model_evaluation

    def source_snapshot(self, directory):
        return {str(path.relative_to(directory)): path.read_bytes()
                for path in directory.rglob('*')
                if path.is_file() and path.name != 'test_manifest.csv'}

    def rewrite_initialization(self, run, certificate, *, update_identity_hash=True):
        write_json(run/'initialization.json', certificate)
        if update_identity_hash:
            path = run/'identity.json'
            identity = json.loads(path.read_text(encoding='utf-8'))
            identity['initialization_sha256'] = canonical_sha256(certificate)
            write_json(path, identity)

    def test_analysis_is_read_only_and_does_not_open_checkpoints_or_sealed_test(self):
        run = synthetic_run(self.root/'run')
        before = self.source_snapshot(run)
        original_open = Path.open

        def reject_model_and_test_reads(path, *args, **kwargs):
            if path.suffix == '.pt' or path.name == 'test_manifest.csv':
                raise AssertionError('Offline artifact evaluation opened a model or sealed test file')
            return original_open(path, *args, **kwargs)

        with patch.object(Path, 'open', reject_model_and_test_reads):
            report = self.analyzer().analyze(run, self.root/'evaluation')
        self.assertEqual(report['status'], 'MODEL_EVALUATION_E01_COMPLETE')
        self.assertEqual(report['scope'], 'exploratory_validation')
        self.assertEqual(report['common_epoch'], 2)
        self.assertEqual(report['control'], 'B3R')
        self.assertEqual(report['new_inference_calls'], 0)
        self.assertEqual(report['training_updates'], 0)
        self.assertIs(report['test_read'], False)
        self.assertIs(report['b0_comparison_available'], False)
        self.assertIs(report['source_artifacts_preserved'], True)
        self.assertEqual(self.source_snapshot(run), before)
        self.assertTrue((self.root/'evaluation'/'report.md').is_file())
        self.assertTrue((self.root/'evaluation'/'summary.json').is_file())
        json.dumps(report, allow_nan=False)

    def test_fixed_epoch_and_prefix_best_are_separate_snapshots(self):
        report = self.analyzer().analyze(synthetic_run(self.root/'run'), self.root/'evaluation')
        candidates = [row for row in report['comparisons']
                      if row['control'] == 'B3R' and row['candidate'] == 'B3R_GATE']
        self.assertEqual({row['snapshot'] for row in candidates}, {'common_epoch', 'common_prefix_best'})
        # This fixture intentionally gives GATE its best at epoch1 and B3R at2.
        # No latest-state model or inference may be substituted for those NPZs.
        for row in candidates:
            expected = -1. if row['snapshot'] == 'common_epoch' else 0.
            self.assertAlmostEqual(row['window_comparison']['mae_macro']['gain'], expected)

    def test_engineering_runs_require_explicit_opt_in_and_remain_engineering_only(self):
        run = synthetic_run(self.root/'run', engineering=True)
        with self.assertRaises(ValueError):
            self.analyzer().analyze(run, self.root/'default_rejected')
        report = self.analyzer().analyze(run, self.root/'explicit_engineering', allow_engineering=True)
        self.assertEqual(report['scope'], 'engineering_only')
        self.assertEqual(report['new_inference_calls'], 0)

    def test_incompatible_reference_never_enters_paired_results(self):
        run = synthetic_run(self.root/'run')
        reference = synthetic_run(self.root/'reference', arms=['B0'], package_tag='9')
        report = self.analyzer().analyze(run, self.root/'evaluation', reference_run_dir=reference,
                                         reference_arms=('B0',))
        self.assertIs(report['b0_comparison_available'], False)
        self.assertTrue(all(not row['candidate'].endswith('B0') and not row['control'].endswith('B0')
                            for row in report['comparisons']))
        comparability = json.loads((self.root/'evaluation'/'comparability.json').read_text(encoding='utf-8'))
        self.assertEqual(comparability['reference']['status'], 'incompatible')
        self.assertIs(comparability['reference']['paired_allowed'], False)

    def test_compatible_b0_reference_is_kept_as_a_real_original_model_control(self):
        run = synthetic_run(self.root/'run')
        reference = synthetic_run(self.root/'reference', arms=['B0'])
        identity_path = reference/'identity.json'
        identity = json.loads(identity_path.read_text(encoding='utf-8'))
        identity['schema'] = 'internal_spatial_is11_run_v1'
        identity['protocol']['architecture_version'] = 'IS1.1'
        write_json(identity_path, identity)
        report = self.analyzer().analyze(run, self.root/'evaluation', reference_run_dir=reference,
                                         reference_arms=('B0',))
        self.assertIs(report['b0_comparison_available'], True)
        b0_pairs = [row for row in report['comparisons'] if row['control'] == 'reference:B0']
        self.assertEqual({row['candidate'] for row in b0_pairs},
                         {'B3R', 'B3R_HALF', 'B3R_GATE', 'B3R_NO_D'})
        self.assertTrue(all(row['snapshot'] in ('common_epoch', 'common_prefix_best') for row in b0_pairs))

    def test_changed_shared_backbone_disqualifies_reference_even_with_same_data_and_budget(self):
        for index, changed_source in enumerate(('src/models/igstgnn.py', 'src/base/model.py',
                                                'src/models/incident_response.py',
                                                'src/models/incident_spatial.py')):
            with self.subTest(source=changed_source):
                run = synthetic_run(self.root/f'run_{index}')
                reference = synthetic_run(self.root/f'reference_{index}', arms=['B0'])
                path = reference/'identity.json'
                identity = json.loads(path.read_text(encoding='utf-8'))
                identity['source_sha256'][changed_source] = '9'*64
                write_json(path, identity)
                output = self.root/f'evaluation_{index}'
                report = self.analyzer().analyze(run, output, reference_run_dir=reference,
                                                 reference_arms=('B0',))
                self.assertIs(report['b0_comparison_available'], False)
                self.assertTrue(all(not row['control'].endswith('B0') and not row['candidate'].endswith('B0')
                                    for row in report['comparisons']))
                comparability = json.loads((output/'comparability.json').read_text(encoding='utf-8'))
                self.assertEqual(comparability['reference']['status'], 'incompatible')

    def test_changed_or_missing_reference_initialization_is_not_certified_as_paired(self):
        for variant in ('changed', 'missing'):
            with self.subTest(variant=variant):
                run = synthetic_run(self.root/('run_'+variant))
                reference = synthetic_run(self.root/('reference_'+variant), arms=['B0'])
                certificate_path = reference/'initialization.json'
                if variant == 'changed':
                    certificate = json.loads(certificate_path.read_text(encoding='utf-8'))
                    certificate['arms']['B0']['common_initial_sha256'] = '9'*64
                    certificate['shared_state_sha256'] = '9'*64
                    self.rewrite_initialization(reference, certificate)
                else:
                    certificate_path.unlink()
                output = self.root/('evaluation_'+variant)
                report = self.analyzer().analyze(run, output, reference_run_dir=reference,
                                                 reference_arms=('B0',))
                self.assertIs(report['b0_comparison_available'], False)
                comparability = json.loads((output/'comparability.json').read_text(encoding='utf-8'))
                self.assertIs(comparability['reference']['paired_allowed'], False)
                self.assertIn(comparability['reference']['status'], ('incompatible', 'missing_evidence'))

    def test_primary_initialization_must_be_present_passed_consistent_and_hash_verified(self):
        for variant in ('missing', 'failed', 'different_common', 'hash_conflict'):
            with self.subTest(variant=variant):
                run = synthetic_run(self.root/('run_'+variant))
                path = run/'initialization.json'
                certificate = json.loads(path.read_text(encoding='utf-8'))
                if variant == 'missing':
                    path.unlink()
                else:
                    if variant == 'failed':
                        certificate['status'] = 'INITIALIZATION_FAILED'
                    else:
                        certificate['arms']['B3R_GATE']['common_initial_sha256'] = '9'*64
                    self.rewrite_initialization(run, certificate,
                                                update_identity_hash=variant != 'hash_conflict')
                with self.assertRaises(ValueError):
                    self.analyzer().analyze(run, self.root/('evaluation_'+variant))

    def test_rmse_macro_and_pooled_must_match_saved_horizon_values(self):
        for field in ('rmse_macro', 'rmse_pooled'):
            with self.subTest(field=field):
                run = synthetic_run(self.root/('run_'+field))
                path = run/'B3R'/'history.json'
                history = json.loads(path.read_text(encoding='utf-8'))
                history[0]['validation']['metrics']['all_nodes'][field] += 1.
                write_json(path, history)
                with self.assertRaises(ValueError):
                    self.analyzer().analyze(run, self.root/('evaluation_'+field))

    def test_positive_target_counts_and_mape_aggregation_are_verified(self):
        for variant in ('too_many_positive', 'wrong_macro', 'wrong_pooled', 'missing_positive_mape'):
            with self.subTest(variant=variant):
                run = synthetic_run(self.root/('run_'+variant))
                path = run/'B3R'/'history.json'
                history = json.loads(path.read_text(encoding='utf-8'))
                metric = history[0]['validation']['metrics']['all_nodes']
                if variant == 'too_many_positive':
                    metric['positive_target_count_per_horizon'][0] = 7
                elif variant == 'wrong_macro':
                    metric['mape_macro'] = .2
                elif variant == 'wrong_pooled':
                    metric['mape_pooled'] = .2
                else:
                    metric['per_horizon_mape'][0] = None
                write_json(path, history)
                with self.assertRaises(ValueError):
                    self.analyzer().analyze(run, self.root/('evaluation_'+variant))

    def test_mape_none_is_legal_when_a_horizon_has_no_positive_targets(self):
        run = synthetic_run(self.root/'run')
        for arm in ('B3R', 'B3R_HALF', 'B3R_GATE', 'B3R_NO_D'):
            path = run/arm/'history.json'
            history = json.loads(path.read_text(encoding='utf-8'))
            for row in history:
                metric = row['validation']['metrics']['all_nodes']
                metric['positive_target_count_per_horizon'][0] = 0
                metric['per_horizon_mape'][0] = None
                metric['mape_macro'] = None
                metric['mape_pooled'] = .1
            write_json(path, history)
        report = self.analyzer().analyze(run, self.root/'evaluation')
        self.assertEqual(report['status'], 'MODEL_EVALUATION_E01_COMPLETE')
        json.dumps(report, allow_nan=False)

    def test_metric_comparison_csv_preserves_global_road_and_horizon_gains(self):
        run = synthetic_run(self.root/'run')
        for arm, road_mae in (('B3R', 1.25), ('B3R_GATE', 2.5)):
            path = run/arm/'history.json'
            history = json.loads(path.read_text(encoding='utf-8'))
            metric = history[1]['validation']['metrics']['road:fixture']
            metric.update(mae_macro=road_mae, mae_pooled=road_mae,
                          rmse_macro=road_mae+1., rmse_pooled=road_mae+1.,
                          per_horizon_mae=[road_mae]*12,
                          per_horizon_rmse=[road_mae+1.]*12,
                          valid_count_per_horizon=[3]*12,
                          positive_target_count_per_horizon=[3]*12)
            write_json(path, history)
        output = self.root/'evaluation'
        self.analyzer().analyze(run, output)
        with (output/'metric_comparisons.csv').open(encoding='utf-8-sig', newline='') as stream:
            rows = [row for row in csv.DictReader(stream)
                    if row['control'] == 'B3R' and row['candidate'] == 'B3R_GATE'
                    and row['snapshot'] == 'common_epoch' and row['metric'] == 'mae']
        for group, aggregation, horizon, expected in (
                ('all_nodes', 'macro', '', -1.),
                ('road:fixture', 'macro', '', -1.25),
                ('all_nodes', 'per_horizon', '7', -1.)):
            with self.subTest(group=group, aggregation=aggregation, horizon=horizon):
                selected = [row for row in rows if row['group'] == group
                            and row['aggregation'] == aggregation and row['horizon'] == horizon]
                self.assertEqual(len(selected), 1)
                self.assertEqual(selected[0]['status'], 'available')
                self.assertAlmostEqual(float(selected[0]['gain']), expected)

    def test_legacy_report_state_is_preserved_without_claiming_shared_diagnostic_semantics(self):
        run = synthetic_run(self.root/'run')
        reference = synthetic_run(self.root/'reference', arms=['B0'])
        path = reference/'B0'/'history.json'
        history = json.loads(path.read_text(encoding='utf-8'))
        legacy = dict(prior_mean=.4, posterior_mean=.3, adoption_mean=.2)
        history[1]['validation']['mechanisms']['report_state'] = legacy
        write_json(path, history)
        output = self.root/'evaluation'
        self.analyzer().analyze(run, output, reference_run_dir=reference, reference_arms=('B0',))
        with (output/'mechanism_chain.csv').open(encoding='utf-8-sig', newline='') as stream:
            rows = [row for row in csv.DictReader(stream)
                    if row['model'] == 'reference:B0' and row['snapshot'] == 'common_epoch'
                    and row['kind'] == 'legacy_report_state']
        self.assertEqual({row['name'] for row in rows}, set(legacy))
        for row in rows:
            self.assertAlmostEqual(float(row['value']), legacy[row['name']])
            self.assertIn('Legacy', row['schema_warning'])
            self.assertIn('no direct G1.1 subtraction', row['schema_warning'])

    def test_source_change_during_evaluation_is_incomplete_instead_of_preservation_success(self):
        run = synthetic_run(self.root/'run')
        analyzer = self.analyzer()
        original_verify = analyzer.Sources.verify
        source = run/'B3R'/'history.json'

        def external_writer_then_verify(sources):
            # Simulate another process committing a new artifact after its read.
            source.write_bytes(source.read_bytes()+b'\n')
            return original_verify(sources)

        output = self.root/'evaluation'
        with patch.object(analyzer.Sources, 'verify', external_writer_then_verify):
            with self.assertRaisesRegex(ValueError, 'Source changed'):
                analyzer.analyze(run, output)
        failure = json.loads((output/'failure.json').read_text(encoding='utf-8'))
        self.assertEqual(failure['status'], 'MODEL_EVALUATION_INCOMPLETE')
        self.assertIsNot(failure.get('source_artifacts_preserved'), True)
        self.assertFalse((output/'summary.json').exists())

    def test_history_scalar_cannot_disagree_with_saved_horizon_metrics(self):
        run = synthetic_run(self.root/'run')
        path = run/'B3R'/'history.json'
        history = json.loads(path.read_text(encoding='utf-8'))
        history[0]['validation_mae_macro'] += .25
        write_json(path, history)
        with self.assertRaises(ValueError):
            self.analyzer().analyze(run, self.root/'evaluation')

    def test_selected_window_artifact_must_reconstruct_history_metric(self):
        run = synthetic_run(self.root/'run')
        path = run/'B3R'/'validation_windows_epoch_002.npz'
        with np.load(path, allow_pickle=False) as original:
            arrays = {key: original[key].copy() for key in original.files}
        arrays['absolute'][0, 0] += 1.
        np.savez_compressed(path, **arrays)
        with self.assertRaises(ValueError):
            self.analyzer().analyze(run, self.root/'evaluation')

    def test_missing_training_order_evidence_cannot_silently_certify_all_arms(self):
        run = synthetic_run(self.root/'run')
        for arm in ('B3R', 'B3R_HALF', 'B3R_GATE', 'B3R_NO_D'):
            path = run/arm/'history.json'
            history = json.loads(path.read_text(encoding='utf-8'))
            for row in history:
                del row['order_sha256']
            write_json(path, history)
        with self.assertRaises(ValueError):
            self.analyzer().analyze(run, self.root/'evaluation')

    def test_missing_update_or_learning_rate_evidence_cannot_certify_all_arms(self):
        for field in ('epoch_updates', 'learning_rate_used'):
            with self.subTest(field=field):
                run = synthetic_run(self.root/('run_'+field))
                for arm in ('B3R', 'B3R_HALF', 'B3R_GATE', 'B3R_NO_D'):
                    path = run/arm/'history.json'
                    history = json.loads(path.read_text(encoding='utf-8'))
                    for row in history:
                        del row[field]
                    write_json(path, history)
                with self.assertRaises(ValueError):
                    self.analyzer().analyze(run, self.root/('evaluation_'+field))

    def test_epoch_updates_must_equal_the_cumulative_update_difference(self):
        run = synthetic_run(self.root/'run')
        path = run/'B3R_GATE'/'history.json'
        history = json.loads(path.read_text(encoding='utf-8'))
        history[1]['epoch_updates'] += 1
        write_json(path, history)
        with self.assertRaises(ValueError):
            self.analyzer().analyze(run, self.root/'evaluation')

    def test_internally_consistent_but_different_arm_budget_is_not_paired(self):
        run = synthetic_run(self.root/'run')
        path = run/'B3R_GATE'/'history.json'
        history = json.loads(path.read_text(encoding='utf-8'))
        history[0]['epoch_updates'] += 1
        for row in history:
            row['global_updates'] += 1
        write_json(path, history)
        summary_path = run/'B3R_GATE'/'summary.json'
        summary = json.loads(summary_path.read_text(encoding='utf-8'))
        summary['global_updates'] += 1
        write_json(summary_path, summary)
        with self.assertRaises(ValueError):
            self.analyzer().analyze(run, self.root/'evaluation')

    def test_different_training_order_rejects_pair_even_when_metrics_match(self):
        run = synthetic_run(self.root/'run')
        path = run/'B3R_GATE'/'history.json'
        history = json.loads(path.read_text(encoding='utf-8'))
        history[0]['order_sha256'] = '9'*64
        write_json(path, history)
        with self.assertRaises(ValueError):
            self.analyzer().analyze(run, self.root/'evaluation')

    def test_run_recording_test_access_is_rejected(self):
        run = synthetic_run(self.root/'run')
        path = run/'identity.json'
        identity = json.loads(path.read_text(encoding='utf-8'))
        identity['test_read'] = True
        write_json(path, identity)
        with self.assertRaises(ValueError):
            self.analyzer().analyze(run, self.root/'evaluation')


if __name__ == '__main__':
    unittest.main()
