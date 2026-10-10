"""Artifact-only M4.5 checks, including unequal horizon counts and cancellation."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1]/'experiments/chronological/analyze_incident_capacity_report_probe.py'
spec = importlib.util.spec_from_file_location('capacity_gain_analysis', SCRIPT)
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def oracle(on, off, target, valid, mask, weights=None):
    """Independent per-horizon selection oracle for the legacy saved schema."""
    mask = np.broadcast_to(mask, (len(on), on.shape[2]))
    weights = np.ones(len(on)) if weights is None else weights
    on_h, off_h, counts, wcounts = [], [], [], []
    total_change = total_weight = maximum = 0.
    for h in range(on.shape[1]):
        selected = valid[:, h, :, 0] & mask
        counts.append(int(selected.sum()))
        w = np.broadcast_to(weights[:, None], selected.shape)[selected]
        if not len(w):
            return None
        a, b, y = on[:, h, :, 0][selected], off[:, h, :, 0][selected], target[:, h, :, 0][selected]
        on_h.append(float(np.average(np.abs(a-y), weights=w)))
        off_h.append(float(np.average(np.abs(b-y), weights=w)))
        wcounts.append(float(w.sum()))
        total_change += float(np.dot(np.abs(b-a), w))
        total_weight += float(w.sum())
        maximum = max(maximum, float(np.abs(b-a).max()))
    return dict(on_mae_macro=float(np.mean(on_h)), off_mae_macro=float(np.mean(off_h)),
        off_minus_on_mae=float(np.mean(np.asarray(off_h)-on_h)),
        per_horizon_on_mae=on_h, per_horizon_off_mae=off_h,
        per_horizon_off_minus_on=(np.asarray(off_h)-on_h).tolist(),
        valid_count=sum(counts), per_horizon_valid_count=counts, per_horizon_weighted_count=wcounts,
        prediction_abs_change_mean=total_change/total_weight, prediction_abs_change_max=maximum)


class GainAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='m45_', dir=Path(__file__).resolve().parent)
        self.root = Path(self.temp.name)
        self.probe, self.network = self.root/'probe', self.root/'network'
        self.probe.mkdir()
        self.network.mkdir()
        self.on = np.full((4, 12, 6, 1), 10., dtype=np.float64)
        self.off = self.on.copy()
        self.off[0, :, :, 0] += [1., -.5, .25, 0, 0, 0]
        self.off[1, :, :, 0] += [.1, 0, 0, -1., .5, 0]
        self.off[3, :, :, 0] += [.5, 1., -.5, 0, 0, 0]
        self.target = np.zeros_like(self.on)
        self.valid = np.ones_like(self.on, dtype=bool)
        self.valid[0, 0, :2] = False
        self.target[~self.valid] = np.nan
        self.edges = np.array([[0, 1, 3, -1], [1, 2, 4, 0]])
        self.ids = np.arange(100, 106)
        self.association = np.array([[.5, 0, 0, 0], [0, 0, 1., 0], [0, 0, 0, 0], [.2, 0, 0, 0]])
        self.direct = np.array([[1, 1, 0, 0, 0, 0], [0, 0, 0, 1, 1, 0],
                               [0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0]], dtype=bool)
        self.potential = np.array([[1, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 0],
                                  [0, 0, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0]], dtype=bool)
        self.weights = np.array([.5, .5, 1., 1.])
        self.windows = np.stack((np.arange(5., 65., 5.), np.arange(10., 70., 5.)), -1)
        self.arrays = dict(on_prediction=self.on, off_prediction=self.off, target=self.target, valid=self.valid,
            sample_indices=np.arange(9, 13), station_ids=self.ids, report_association=self.association,
            direct_report_node_mask=self.direct, potential_component_mask=self.potential,
            unique_cutoff_weights=self.weights, target_windows_minutes=self.windows)
        self.save_arrays()
        np.savez_compressed(self.probe/'pathway_curves.npz', edge_index=self.edges,
                            station_ids=self.ids, target_windows_minutes=self.windows)
        common = np.array([1, 1, 1, 0, 0, 0], bool)
        candidate = np.array([1, 1, 1, 1, 1, 0], bool)
        boundary = np.array([1, 0, 0, 0, 1, 0], bool)
        graph = dict(station_ids=self.ids.tolist(), edges=[dict(source=int(a), destination=int(b)) for a, b in self.edges.T],
                     operator_mask=candidate.tolist(), boundary_nodes=boundary.tolist())
        analysis.write_json(self.network/'graph.json', graph)
        np.save(self.network/'common_structure_mask.npy', common)
        analysis.write_csv(self.network/'stations.csv', [dict(station_id=int(station), node_index=i,
            road='R1' if i < 3 else 'R2') for i, station in enumerate(self.ids)])
        analysis.write_csv(self.network/'roads.csv', [dict(road='R1', stations=3), dict(road='R2', stations=3)])
        summary = dict(outputs_sha256={p.name: analysis.sha256(p) for p in self.network.iterdir()})
        analysis.write_json(self.network/'summary.json', summary)
        invocation = dict(directories=dict(network=str(self.network)), optimizer_updates=0, test_accessed=False,
            origin_identity=dict(check=True, ramp_exchanges=True, candidate_structure_nodes=5,
                inputs_sha256={'network/summary.json': analysis.sha256(self.network/'summary.json')}))
        analysis.write_json(self.probe/'invocation.json', invocation)
        analysis.write_json(self.probe/'checkpoint_audit.json', dict(protected_artifacts_unchanged=True))
        supported = self.association.any(1)
        self.masks = dict(all_nodes=np.ones(6, bool), common_structure=common, added_structure=candidate & ~common,
            candidate_structure=candidate, outside_candidate=~candidate, candidate_boundary=boundary,
            road_R1=np.arange(6) < 3, road_R2=np.arange(6) >= 3,
            report_associated_nodes=self.direct, no_direct_report_association_nodes=~self.direct,
            potential_propagation_only_nodes=self.potential & ~self.direct,
            outside_potential_propagation_nodes=~self.potential,
            report_supported_samples=np.broadcast_to(supported[:, None], self.direct.shape),
            no_report_supported_samples=np.broadcast_to(~supported[:, None], self.direct.shape),
            matched_report_age_0_5=np.broadcast_to(np.array([1, 0, 0, 0], bool)[:, None], self.direct.shape),
            matched_report_age_5_15=np.broadcast_to(np.array([0, 1, 0, 0], bool)[:, None], self.direct.shape),
            matched_report_age_15_60=np.broadcast_to(np.array([0, 0, 0, 1], bool)[:, None], self.direct.shape))
        replay = dict(matching=True, prediction_abs_max=0., atol=1e-4, rtol=1e-6)
        analysis.write_json(self.probe/'replay.json', replay)
        samples = []
        for i in range(4):
            metric = oracle(self.on[i:i+1], self.off[i:i+1], self.target[i:i+1], self.valid[i:i+1], np.ones(6, bool))
            samples.append(dict(sample_index=9+i, incident_id=['A', 'A', 'B', 'C'][i],
                t0=['2023-09-01T00:00:00', '2023-09-01T00:00:00', '2023-09-01T01:00:00', '2023-09-01T02:00:00'][i],
                report_entities=1, matched_report_entities=int(supported[i]), associated_edges=int(supported[i]),
                associated_nodes=int(self.direct[i].sum()), potential_component_nodes=int(self.potential[i].sum()),
                youngest_matched_recorded_report_age_minutes=[0, 5, None, 60][i],
                unique_cutoff_sensitivity_weight=float(self.weights[i]), on_mae=metric['on_mae_macro'],
                off_mae=metric['off_mae_macro'], off_minus_on_mae=metric['off_minus_on_mae']))
        analysis.write_csv(self.probe/'samples.csv', samples)
        self.report = dict(status='P1_REPORT_CAPACITY_PROBE_PASS', check_subset=True, validation_rows=4,
            best_epoch=9, distinct_cutoffs=3, replay=replay, optimizer_updates=0, test_accessed=False,
            model_state_preserved=True, final_on_restored=True, original_IGSTGNN_incident_inputs_preserved=True,
            protected_artifacts_unchanged=True, report_age_semantics='recorded_age', potential_region_semantics='envelope',
            metrics={name: oracle(self.on, self.off, self.target, self.valid, mask) for name, mask in self.masks.items()},
            unique_cutoff_sensitivity_metrics={name: oracle(self.on, self.off, self.target, self.valid, mask, self.weights)
                                               for name, mask in self.masks.items()},
            coverage=dict(rows_with_matched_reports=3, directly_associated_sample_node_positions=6,
                sample_node_positions=24, supported_sample_edge_positions=3, matched_report_row_occurrences=3,
                report_row_occurrences=4))
        self.refresh_manifest()

    def tearDown(self):
        assert self.root.resolve().is_relative_to(Path(__file__).resolve().parent)
        self.temp.cleanup()

    def save_arrays(self):
        np.savez_compressed(self.probe/'paired_predictions.npz', **self.arrays)

    def refresh_manifest(self):
        self.report['outputs_sha256'] = {p.name: analysis.sha256(p) for p in self.probe.iterdir() if p.name != 'report.json'}
        analysis.write_json(self.probe/'report.json', self.report)

    def run_analysis(self, **kwargs):
        return analysis.analyze(self.probe, self.root/'output', **kwargs)

    def assert_rejected(self, pattern, **kwargs):
        with self.assertRaisesRegex(ValueError, pattern):
            self.run_analysis(**kwargs)
        self.assertFalse((self.root/'output/analysis.json').exists())
        self.assertEqual(analysis.read_json(self.root/'output/failure.json')['completion_claim'], False)

    def test_completed_artifacts_reproduce_oracle_and_remain_unchanged(self):
        before = analysis.snapshot([p for folder in (self.probe, self.network) for p in folder.iterdir()])
        result = self.run_analysis()
        self.assertEqual(result['status'], 'P1_REPORT_GAIN_ANALYSIS_PASS')
        self.assertEqual(result['source_scientific_scope'], 'ENGINEERING_ONLY')
        self.assertEqual(result['distinct_trigger_ids'], 3)
        self.assertEqual(result['new_inference_calls'], 0)
        self.assertEqual(result['optimizer_updates'], 0)
        self.assertFalse(result['test_accessed'])
        self.assertEqual(before, analysis.snapshot([Path(p) for p in before]))
        self.assertEqual(set(result['outputs_sha256']), {'README.md', 'gain_decomposition.csv',
            'trigger_event_summary.csv', 'cutoff_summary.csv'})
        for name in self.masks:
            self.assertAlmostEqual(result['metrics'][name]['off_minus_on_mae'], self.report['metrics'][name]['off_minus_on_mae'])
        all_nodes = result['metrics']['all_nodes']
        contributions = [result['metrics'][name]['global_gain_contribution'] for name in analysis.PARTITIONS]
        self.assertAlmostEqual(sum(contributions), all_nodes['off_minus_on_mae'])
        self.assertGreater(all_nodes['gain_positive'], 0)
        self.assertGreater(all_nodes['gain_negative'], 0)
        self.assertGreater(all_nodes['cancellation_fraction'], 0)
        self.assertNotAlmostEqual(all_nodes['prediction_abs_change_pooled_mean'],
                                  all_nodes['prediction_abs_change_horizon_macro_mean'])
        weighted = result['unique_cutoff_sensitivity_metrics']['all_nodes']
        self.assertAlmostEqual(weighted['off_minus_on_mae'], self.report['unique_cutoff_sensitivity_metrics']['all_nodes']['off_minus_on_mae'])
        for filename in ('trigger_event_summary.csv', 'cutoff_summary.csv'):
            groups = analysis.read_csv(self.root/'output'/filename)
            self.assertAlmostEqual(sum(float(g['global_gain_contribution']) for g in groups), all_nodes['off_minus_on_mae'])

    def test_exact_cancellation_and_stronger_bound_use_horizon_denominators(self):
        # Two observations per horizon, errors change by +2 and -2: zero net, full cancellation.
        on = np.full((1, 12, 2, 1), 3.)
        off = on.copy()
        off[:, :, 0] += 2
        off[:, :, 1] -= 2
        valid = np.ones_like(on, bool)
        errors = dict(on_error=on, off_error=off, gain=off-on,
                      positive=np.maximum(off-on, 0), negative=np.maximum(on-off, 0), change=np.abs(off-on))
        metric = analysis.aggregate(analysis.raw_statistics(errors, valid, np.ones(2, bool)))
        self.assertEqual(metric['off_minus_on_mae'], 0)
        self.assertEqual(metric['gain_positive'], 1)
        self.assertEqual(metric['gain_negative'], 1)
        self.assertEqual(metric['cancellation_fraction'], 1)
        self.assertEqual(metric['prediction_abs_change_horizon_macro_mean'], 2)
        errors['change'] *= .5  # Net bound passes, positive+negative bound must fail.
        with self.assertRaisesRegex(ValueError, 'triangle bound'):
            analysis.aggregate(analysis.raw_statistics(errors, valid, np.ones(2, bool)))

    def test_empty_region_is_unavailable_but_global_contribution_is_zero(self):
        errors = {name: np.zeros_like(self.on) for name in ('on_error', 'off_error', 'gain', 'positive', 'negative', 'change')}
        metric = analysis.aggregate(analysis.raw_statistics(errors, self.valid, np.zeros(6, bool)), np.ones(12))
        self.assertIsNone(metric['on_mae_macro'])
        self.assertIsNone(metric['off_minus_on_mae'])
        self.assertIsNone(metric['gain_positive'])
        self.assertIsNone(metric['prediction_abs_change_max'])
        self.assertEqual(metric['global_gain_contribution'], 0)
        self.assertEqual(metric['per_horizon_global_gain_contribution'], [0.]*12)

    def test_partial_horizon_is_unavailable_but_keeps_available_horizons(self):
        errors = {name: np.zeros_like(self.on) for name in ('on_error', 'off_error', 'gain', 'positive', 'negative', 'change')}
        valid = self.valid.copy()
        valid[:, 2] = False
        metric = analysis.aggregate(analysis.raw_statistics(errors, valid, np.ones(6, bool)), np.ones(12))
        self.assertIsNone(metric['off_minus_on_mae'])
        self.assertIsNone(metric['per_horizon_off_minus_on'][2])
        self.assertEqual(metric['per_horizon_off_minus_on'][3], 0)
        self.assertEqual(metric['global_gain_contribution'], 0)

    def test_rejects_hash_mismatch_without_touching_source(self):
        with (self.probe/'samples.csv').open('a') as stream:
            stream.write('\n')
        self.assert_rejected('hash mismatch')

    def test_rejects_reordered_samples_even_after_rehash(self):
        self.arrays['sample_indices'] = self.arrays['sample_indices'][::-1]
        self.save_arrays()
        self.refresh_manifest()
        self.assert_rejected('Sample CSV order')

    def test_rejects_station_axis_mismatch_even_after_rehash(self):
        self.arrays['station_ids'] = self.ids[::-1]
        self.save_arrays()
        self.refresh_manifest()
        self.assert_rejected('Pathway station axis')

    def test_rejects_graph_region_mask_corruption_even_after_rehash(self):
        self.arrays['potential_component_mask'][0, 5] = True
        self.save_arrays()
        self.refresh_manifest()
        self.assert_rejected('graph-derived region masks')

    def test_rejects_invalid_target_window_and_mask(self):
        self.arrays['target_windows_minutes'] = self.windows+5
        self.save_arrays()
        self.refresh_manifest()
        self.assert_rejected('Target windows')

    def test_rejects_nonfinite_valid_target(self):
        self.arrays['valid'][0, 0, 0] = True
        self.save_arrays()
        self.refresh_manifest()
        self.assert_rejected('Invalid paired predictions/targets/mask')

    def test_rejects_no_report_prediction_response(self):
        self.arrays['off_prediction'][2, 1, 2] += 1.
        self.save_arrays()
        self.refresh_manifest()
        self.assert_rejected('Unsupported samples changed')

    def test_rejects_changed_source_metric(self):
        self.report['metrics']['all_nodes']['off_minus_on_mae'] += .01
        analysis.write_json(self.probe/'report.json', self.report)
        self.assert_rejected('Artifact metric mismatch')

    def test_rejects_incomplete_replay(self):
        self.report['replay']['matching'] = False
        analysis.write_json(self.probe/'replay.json', self.report['replay'])
        self.refresh_manifest()
        self.assert_rejected('Completed source replay')

    def test_unfinished_probe_cannot_produce_completion(self):
        self.report['status'] = 'FAILED'
        analysis.write_json(self.probe/'report.json', self.report)
        self.assert_rejected('did not complete')

    def test_engineering_fixture_requires_explicit_mark_and_opt_in(self):
        self.report.pop('outputs_sha256')
        self.report['engineering_only'] = True
        analysis.write_json(self.probe/'report.json', self.report)
        with self.assertRaisesRegex(ValueError, 'lacks a completed hash manifest'):
            analysis.analyze(self.probe, self.root/'rejected')
        result = self.run_analysis(allow_engineering=True)
        self.assertEqual(result['source_verification'], 'explicit_engineering_fixture')
        self.assertEqual(result['source_scientific_scope'], 'ENGINEERING_ONLY')

    def test_manifest_opt_in_cannot_bypass_unmarked_source(self):
        self.report.pop('outputs_sha256')
        analysis.write_json(self.probe/'report.json', self.report)
        self.assert_rejected('lacks a completed hash manifest', allow_engineering=True)

    def test_output_directory_cannot_overwrite_or_enter_source(self):
        for output in (self.probe, self.probe/'new', self.root):
            with self.assertRaisesRegex(ValueError, 'overlaps'):
                analysis.analyze(self.probe, output)
        self.assertFalse((self.probe/'new').exists())

    def test_cli_uses_no_torch_or_training_imports(self):
        code = "import runpy,sys; sys.argv=['analyzer','--help']; " \
               "\ntry: runpy.run_path(SCRIPT,run_name='__main__')" \
               "\nexcept SystemExit as e: assert e.code == 0" \
               "\nassert 'torch' not in sys.modules" \
               "\nassert not any('train_incident' in m for m in sys.modules)"
        code = 'SCRIPT='+repr(str(SCRIPT))+'\n'+code
        result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--probe-dir', result.stdout)


if __name__ == '__main__':
    unittest.main()
