"""Saved frozen-MIN diagnostics validate artifacts without any model runtime."""
import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experiments.chronological import analyze_min_frozen_gain as runner
from experiments.chronological.min_frozen_budget_metrics import select_hard_windows, summarize_pair


class FrozenGainRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=runner.REPO)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run, self.data, self.graph = (self.root/name for name in ('run', 'data', 'graph'))
        for directory in (self.run, self.data, self.graph):
            directory.mkdir()
        self.out = self.run/'analysis'
        self.stations = [101, 102, 103, 104]
        samples = []
        flows = np.resize(np.array([0., 2., 10., 30., 70., 150., 250.], np.float32), (12, 4))
        for index in range(20):
            y = np.zeros((12, 4, 3), np.float32)
            y[..., 0] = (flows-10.)/2.
            relation = np.zeros((4, 3), np.float32)
            relation[index % 4, 1] = 1.
            samples.append(dict(x_data=np.zeros_like(y), y_data=y,
                                incident_distances=relation, incident_features={}))
        # Validation is 14,15,16. A very small real relation rounds to zero in
        # the model's original float32 conversion; a false report has no seed.
        samples[14]['incident_distances'][:] = 0.
        samples[14]['incident_distances'][0, 1] = 1.
        samples[15]['incident_distances'] = np.zeros((4, 3), np.float64)
        samples[15]['incident_distances'][2, 0] = 1e-60
        samples[16]['report_present'] = False
        np.save(self.data/'incident_all.npy', np.array(samples, dtype=object))
        np.savez(self.data/'incident_stats.npz', normalized=True,
                 mean=np.float32(10.), std=np.float32(2.))
        (self.data/'sensors.csv').write_text('station_id\n101\n102\n103\n104\n', encoding='utf-8')
        runner.write_json(self.graph/'graph.json', dict(station_ids=self.stations,
            edges=[dict(source=0, destination=1), dict(source=-1, destination=2),
                   dict(source=3, destination=-1)], evidence_scope='candidate_unverified'))
        self.target = np.stack([sample['y_data'][..., 0]*np.float32(2.)+np.float32(10.)
                                for sample in samples[14:17]])
        offsets = np.resize(np.array([2., -2., 0., 1.], np.float32), self.target.shape)
        self.off = self.target+offsets
        self.on = self.target+offsets*np.float32(.5)
        self.valid = self.target != 0.
        self.indices = np.arange(14, 17, dtype=np.int64)
        self.hard = select_hard_windows(self.off, self.target, self.valid, self.indices, fraction=.1)
        self.pair = self.report(self.on)
        self.zero = self.report(self.off)
        self.identity = dict(schema=runner.RUN_SCHEMA, source_best_epoch=28,
            engineering_check=True, frozen_state_sha256='frozen_fixture_hash', batch_size=2,
            recipe=dict(max_epochs=10, fixed_hard_fraction=.1),
            source_artifacts_sha256={'source/fixture': 'unchanged_fixture_digest'},
            experiment_sources_sha256={'experiments/chronological/min_frozen_budget_metrics.py':
                runner.sha256(runner.REPO/'experiments/chronological/min_frozen_budget_metrics.py')},
            source_identity=dict(schema='min_event_budget_published_run_v1',
                protocol=dict(stations=4, total_samples=20, train_samples=14,
                    val_samples=3, test_samples=3, graph_sha256=runner.sha256(self.graph/'graph.json')),
                package={path.name: runner.sha256(path) for path in self.data.iterdir()},
                graph_inputs_sha256={'graph.json': runner.sha256(self.graph/'graph.json')}))
        self.identity['source_identity']['protocol']['data_sha256'] = copy.deepcopy(self.identity['source_identity']['package'])
        self.history = [dict(epoch=epoch, frozen_state_sha256=self.identity['frozen_state_sha256'],
            validation=copy.deepcopy(self.zero if epoch == 0 else self.pair)) for epoch in range(3)]
        self.summary = dict(status='FROZEN_MIN_LOCATION_ENGINEERING_PASS', completed_epoch=2,
            source_best_epoch=28, best_adapter_epoch=1, frozen_backbone_preserved=True,
            off_exactly_matches_source=True, source_artifacts_preserved=True,
            selection_includes_epoch_zero=True, test_evaluated=False, engineering_only=True,
            frozen_state_sha256=self.identity['frozen_state_sha256'], fixed_hard_windows=len(self.hard),
            best_validation=copy.deepcopy(self.pair), baseline_metrics=copy.deepcopy(self.zero['on']),
            best_metrics=copy.deepcopy(self.pair['on']), baseline_mae=self.zero['on']['mae_pooled'],
            best_mae=self.pair['on']['mae_pooled'], baseline_minus_best_mae=self.pair['mae_gain_pooled'])
        self.baseline_info = dict(source_best_epoch=28, zero_ON_and_OFF_exactly_match_frozen_MIN=True,
            original_source_metrics_replay_passed=True, mask_value=0., validation_rows=3,
            batch_partition=[2, 1], fixed_hard_sample_indices=self.indices[self.hard].tolist(),
            validation=copy.deepcopy(self.zero))
        self.baseline = dict(prediction=self.off.copy(), target=self.target.copy(), valid=self.valid.copy(),
            sample_indices=self.indices.copy(), fixed_hard_indices=self.hard.copy())
        self.best = dict(on=self.on.copy(), off=self.off.copy(), target=self.target.copy(), valid=self.valid.copy(),
            sample_indices=self.indices.copy(), fixed_hard_indices=self.hard.copy())
        for name, value in (('identity', self.identity), ('history', self.history),
                            ('summary', self.summary), ('baseline', self.baseline_info), ('initialization', {})):
            runner.write_json(self.run/(name+'.json'), value)
        runner.write_json(self.run/'source_preservation.json', dict(status='SOURCE_ARTIFACTS_PRESERVED',
            before=self.identity['source_artifacts_sha256'], after=self.identity['source_artifacts_sha256']))
        (self.run/'last.pt').write_bytes(b'opaque checkpoint bytes deliberately never deserialized')
        (self.run/'active.lock').write_bytes(b'1')
        self.write_archives()

    def report(self, on):
        result = summarize_pair(on, self.off, self.target, self.valid, self.hard)
        result.update(off_exactly_matches_source=True, on_exactly_restored=True,
                      validation_rows=3, batch_partition=[2, 1], mechanisms={})
        return result

    def write_archives(self):
        np.savez_compressed(self.run/'baseline_predictions.npz', **self.baseline)
        np.savez_compressed(self.run/'best_predictions.npz', **self.best)

    def args(self, output=None):
        return runner.parse_args(['--run-dir', str(self.run), '--data-dir', str(self.data),
                                  '--graph-dir', str(self.graph), '--output-dir', str(output or self.out)])

    def analyze(self):
        with redirect_stdout(io.StringIO()):
            return runner.analyze_run(self.args())

    def source_snapshot(self):
        return {str(path.relative_to(self.root)): path.read_bytes()
                for directory in (self.run, self.data, self.graph) for path in directory.iterdir() if path.is_file()}

    def assert_rejected(self, message=None):
        before = self.source_snapshot()
        context = self.assertRaisesRegex(ValueError, message) if message else self.assertRaises(ValueError)
        with context:
            self.analyze()
        self.assertEqual(before, self.source_snapshot())
        self.assertFalse(self.out.exists())

    def refresh_release_hashes(self):
        self.identity['source_identity']['package'] = {path.name: runner.sha256(path) for path in self.data.iterdir()}
        self.identity['source_identity']['protocol']['data_sha256'] = copy.deepcopy(self.identity['source_identity']['package'])
        runner.write_json(self.run/'identity.json', self.identity)

    def test_success_writes_readable_diagnostics_and_preserves_every_source_byte(self):
        before = self.source_snapshot()
        result = self.analyze()
        self.assertEqual(result['status'], 'FROZEN_MIN_GAIN_ANALYSIS_PASS')
        self.assertEqual(result['best_adapter_epoch'], 1)
        self.assertEqual(result['new_inference_calls'], 0)
        self.assertEqual(result['training_updates'], 0)
        self.assertFalse(result['new_checkpoint_replay'])
        self.assertTrue(result['engineering_only'])
        self.assertEqual(before, self.source_snapshot())
        expected = {'analysis.json', 'report.md', 'strata.csv', 'horizons.csv',
                    'region_masks.npz', 'provenance.json', 'preservation.json', 'summary.json'}
        self.assertEqual({path.name for path in self.out.iterdir()}, expected)
        for name in ('analysis', 'provenance', 'preservation', 'summary'):
            self.assertIsInstance(runner.read_json(self.out/(name+'.json')), dict)
        self.assertIn('没有加载模型', (self.out/'report.md').read_text(encoding='utf-8'))
        with np.load(self.out/'region_masks.npz', allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved['sample_indices'], self.indices)
            np.testing.assert_array_equal(saved['station_ids'], self.stations)
            np.testing.assert_array_equal(saved['direct_report_nodes'][0], [True, False, False, False])
            np.testing.assert_array_equal(saved['potential_only'][0], [False, True, False, False])
            self.assertFalse(saved['direct_report_nodes'][1:].any())
            self.assertTrue(saved['remaining_nodes'][1:].all())

    def test_standalone_analysis_refuses_any_torch_import(self):
        code = '''import builtins, runpy, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] == 'torch':
        raise AssertionError('Analysis attempted to import torch')
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')
assert 'torch' not in sys.modules
'''
        command = [sys.executable, '-X', 'utf8', '-c', code, str(Path(runner.__file__).resolve()),
                   '--run-dir', str(self.run), '--data-dir', str(self.data),
                   '--graph-dir', str(self.graph), '--output-dir', str(self.out)]
        result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('FROZEN_MIN_GAIN_ANALYSIS_PASS', result.stdout)

    def test_each_mismatched_shared_archive_axis_or_field_is_rejected(self):
        original = copy.deepcopy(self.best)
        changes = {'target': lambda a: a+1., 'valid': lambda a: ~a,
                   'sample_indices': lambda a: a[::-1], 'fixed_hard_indices': lambda a: np.array([2], np.int64),
                   'off': lambda a: a+1.}
        for name, changed in changes.items():
            with self.subTest(field=name):
                self.best = copy.deepcopy(original)
                self.best[name] = changed(self.best[name])
                self.write_archives()
                self.assert_rejected()

    def test_matching_but_wrong_archive_shape_or_dtype_is_rejected(self):
        original_best, original_baseline = copy.deepcopy(self.best), copy.deepcopy(self.baseline)
        for mode in ('horizon', 'station', 'float64', 'integer_valid'):
            with self.subTest(mode=mode):
                self.best, self.baseline = copy.deepcopy(original_best), copy.deepcopy(original_baseline)
                if mode in ('horizon', 'station'):
                    for archive in (self.best, self.baseline):
                        for key in tuple(archive):
                            if archive[key].ndim == 3:
                                archive[key] = archive[key][:, :11] if mode == 'horizon' else archive[key][..., :3]
                elif mode == 'float64':
                    self.best['on'] = self.best['on'].astype(np.float64)
                else:
                    self.best['valid'] = self.baseline['valid'] = self.valid.astype(np.uint8)
                self.write_archives()
                self.assert_rejected()

    def test_invalid_sample_axis_and_wrong_full_validation_order_are_rejected(self):
        for ids in (np.array([14, 14, 16]), np.array([16, 15, 14]), np.array([15, 16, 17]),
                    np.array([14., 15., 16.])):
            with self.subTest(indices=ids.tolist()):
                self.best['sample_indices'] = self.baseline['sample_indices'] = ids
                self.write_archives()
                self.assert_rejected()

    def test_matching_valid_mask_and_fixed_hard_tampering_are_rejected(self):
        self.best['valid'] = self.baseline['valid'] = ~self.valid
        self.write_archives()
        self.assert_rejected('validity mask')
        self.best['valid'] = self.baseline['valid'] = self.valid
        wrong = np.array([(int(self.hard[0])+1) % 3], dtype=np.int64)
        self.best['fixed_hard_indices'] = self.baseline['fixed_hard_indices'] = wrong
        self.write_archives()
        self.assert_rejected('difficult-window selection')

    def test_nonfinite_valid_predictions_and_extra_or_object_fields_are_rejected(self):
        self.best['on'][0, 0, 1] = np.nan
        self.write_archives()
        self.assert_rejected()
        self.best['on'] = self.on.copy()
        self.best['unexpected'] = np.array([1])
        self.write_archives()
        self.assert_rejected('Unexpected archive fields')
        del self.best['unexpected']
        self.best['on'] = self.on.astype(object)
        self.write_archives()
        self.assert_rejected('Object arrays cannot be loaded')

    def test_history_earliest_tie_and_complete_epoch_zero_are_required(self):
        self.summary['best_adapter_epoch'] = 2
        runner.write_json(self.run/'summary.json', self.summary)
        self.assert_rejected('earliest history minimum')
        self.summary['best_adapter_epoch'] = 1
        runner.write_json(self.run/'summary.json', self.summary)
        runner.write_json(self.run/'history.json', self.history[1:])
        self.assert_rejected('epoch-zero history')

    def test_saved_metrics_or_training_certificate_tampering_is_rejected(self):
        self.summary['best_metrics']['mae_pooled'] += .01
        runner.write_json(self.run/'summary.json', self.summary)
        self.assert_rejected('Saved metric mismatch')
        self.summary['best_metrics'] = self.pair['on']
        self.summary['off_exactly_matches_source'] = False
        runner.write_json(self.run/'summary.json', self.summary)
        self.assert_rejected('successful training certificate')

    def test_source_preservation_and_original_metric_hash_failure_are_rejected(self):
        preservation = runner.read_json(self.run/'source_preservation.json')
        preservation['after']['source/fixture'] = 'changed'
        runner.write_json(self.run/'source_preservation.json', preservation)
        self.assert_rejected('source-preservation certificate')
        preservation['after'] = copy.deepcopy(preservation['before'])
        runner.write_json(self.run/'source_preservation.json', preservation)
        self.identity['experiment_sources_sha256']['experiments/chronological/min_frozen_budget_metrics.py'] = 'changed'
        runner.write_json(self.run/'identity.json', self.identity)
        self.assert_rejected('metric implementation changed')

    def test_release_and_graph_hashes_are_checked_before_saved_data_analysis(self):
        original = (self.data/'sensors.csv').read_bytes()
        (self.data/'sensors.csv').write_bytes(original+b'999\n')
        self.assert_rejected('Recorded release input differs')
        (self.data/'sensors.csv').write_bytes(original)
        (self.graph/'graph.json').write_text('{}', encoding='utf-8')
        self.assert_rejected('Recorded graph input differs')

    def test_hash_consistent_release_target_misalignment_is_still_rejected(self):
        samples = np.load(self.data/'incident_all.npy', allow_pickle=True)
        samples[14]['y_data'][0, 1, 0] += np.float32(1.)
        np.save(self.data/'incident_all.npy', samples)
        self.refresh_release_hashes()
        self.assert_rejected('targets differ from original validation targets')

    def test_metadata_float32_flags_and_weak_components_match_model_policy(self):
        regions, ids, audit = runner.read_metadata(self.data, self.graph, self.identity, self.best, self.baseline_info)
        np.testing.assert_array_equal(ids, self.stations)
        np.testing.assert_array_equal(regions['direct_report_nodes'],
                                     [[True, False, False, False], [False]*4, [False]*4])
        np.testing.assert_array_equal(regions['potential_only'],
                                     [[False, True, False, False], [False]*4, [False]*4])
        coverage = sum(value.astype(int) for value in regions.values())
        self.assertTrue((coverage == 1).all())
        self.assertTrue(audit['selected_original_targets_exact'])
        self.assertFalse(audit['test_records_inspected'])
        # Direction reversal does not change a weak-component envelope;
        # boundary edges must not connect two unrelated station components.
        graph = dict(station_ids=self.stations, edges=[dict(source=1, destination=0),
            dict(source=-1, destination=2), dict(source=3, destination=-1)])
        other = runner.partition_regions(regions['direct_report_nodes'], graph, self.stations)
        for key in regions:
            np.testing.assert_array_equal(other[key], regions[key])

    def test_metadata_requires_complete_author_validation_axis_and_matching_station_ids(self):
        best = copy.deepcopy(self.best)
        best['sample_indices'] = np.array([16, 15, 14], np.int64)
        with self.assertRaisesRegex(ValueError, 'full original validation axis'):
            runner.read_metadata(self.data, self.graph, self.identity, best, self.baseline_info)
        graph = runner.read_json(self.graph/'graph.json')
        graph['station_ids'] = self.stations[::-1]
        with self.assertRaisesRegex(ValueError, 'Graph station axis'):
            runner.partition_regions(np.zeros((3, 4), bool), graph, self.stations)

    def test_missing_relation_and_invalid_report_flags_are_not_accident_support(self):
        np.testing.assert_array_equal(runner.report_support({'incident_features': {}}, 4), [False]*4)
        relation = np.zeros((4, 3), np.float32)
        relation[0, 1] = 1.
        for flag in ('yes', 1, None):
            with self.subTest(flag=flag), self.assertRaisesRegex(ValueError, 'Non-Boolean'):
                runner.report_support(dict(incident_distances=relation, report_present=flag), 4)

    def test_existing_output_and_unsafe_output_locations_are_rejected(self):
        self.out.mkdir()
        sentinel = self.out/'keep.txt'
        sentinel.write_text('unchanged', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'already exists'):
            self.args()
        self.assertEqual(sentinel.read_text(encoding='utf-8'), 'unchanged')
        for output in (self.run, self.root, self.data/'analysis', self.graph/'analysis'):
            with self.subTest(output=str(output)), self.assertRaises(ValueError):
                self.args(output)

    def test_active_lock_conflict_and_missing_lock_do_not_modify_source(self):
        original = (self.run/'active.lock').read_bytes()
        before = self.source_snapshot()
        with runner.readonly_run_lock(self.run):
            with self.assertRaisesRegex(ValueError, 'Source run is active'):
                self.analyze()
        self.assertEqual(before, self.source_snapshot())
        self.assertFalse(self.out.exists())
        self.assertEqual((self.run/'active.lock').read_bytes(), original)
        (self.run/'active.lock').unlink()
        self.assert_rejected('Missing existing run lock')

    def test_source_change_during_analysis_prevents_completed_output(self):
        snapshot = runner.snapshot
        calls = 0

        def drifting(paths):
            nonlocal calls
            calls += 1
            if calls == 2:
                (self.run/'last.pt').write_bytes(b'external concurrent change')
            return snapshot(paths)

        with patch.object(runner, 'snapshot', side_effect=drifting), redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, 'Source changed during analysis'):
                runner.analyze_run(self.args())
        self.assertFalse(self.out.exists())


if __name__ == '__main__':
    unittest.main()
