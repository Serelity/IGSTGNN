"""Pure-NumPy F1 archive checks, analysis artifacts and read-only protection."""
import copy
from contextlib import redirect_stdout, redirect_stderr
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experiments.chronological import analyze_min_forecast_fusion_gain as analysis


def fixture(root, name='forecast_fusion_evaluation_20261015_120000_fixture', *, event_epoch=1):
    source = root/name
    source.mkdir()
    target = np.broadcast_to(np.array([1., 10., 80., 220.], np.float32), (4, 12, 4)).copy()
    target += np.arange(4, dtype=np.float32)[:, None, None]
    target[0, 0, 0] = 0.
    row = np.arange(4, dtype=np.float32)[:, None, None]
    current = target+np.array([2., -1., 0., 3.], np.float32)+row*.2
    report_off = current+np.array([.3, -.2, .1, -.1], np.float32)
    on = report_off+np.array([-.1, 0., 0., 0.], np.float32)
    source_pred = target+np.array([3., -2., .1, 4.], np.float32)
    predictions = dict(STATE_ON=target+np.array([1., -.5, .2, 2.], np.float32)+row*.1,
        EVENT_ON=on, EVENT_REPORT_OFF=report_off, EVENT_FUSION_OFF=current, EVENT_UNIFORM=current+.5,
        EVENT_SEMANTIC_UNK=on+np.array([.02, 0., 0., 0.], np.float32),
        EVENT_LOCATION_SHUFFLE=on+.3, SOURCE_MIN=source_pred)
    if event_epoch == 0:
        for case in analysis.CASES[1:-1]:
            predictions[case] = source_pred.copy()
    valid = target != 0.
    indices, stations = np.arange(100, 104, dtype=np.int64), np.arange(1001, 1005, dtype=np.int64)
    direct = np.broadcast_to([True, False, False, False], (4, 4)).copy()
    potential = np.broadcast_to([False, True, False, False], (4, 4)).copy()
    regions = dict(all_nodes=np.ones((4, 4), bool), direct_report_nodes=direct, potential_only=potential,
        remaining_nodes=~(direct | potential), candidate_potential=direct | potential, empty=np.zeros((4, 4), bool),
        **{'road:WEST': np.array([True, True, False, False]), 'road:EAST': np.array([False, False, True, True])})
    comparable = dict(completed_epochs={'STATE': 2, 'EVENT': 2}, engineering_only=False, validation_rows=4,
        global_updates={'STATE': 4, 'EVENT': 4}, best_stage_epochs={'STATE': 1, 'EVENT': event_epoch},
        common_stage_epoch=2, source_best_epoch=28, same_completed_training_budget=True, source_exact_resume=False,
        optimizer_and_scheduler_reset_at_stage_start=True, existing_MIN_retrained=False,
        registered_parameter_caveat='equal_registered_size_does_not_imply_equal_active_semantic_parameters_or_support')
    original = analysis.recompute_original(predictions, target, valid, regions, indices, comparable)
    replay = {}
    for arm in ('STATE', 'EVENT'):
        metrics = original['cases'][arm+'_ON']['metrics']['all_nodes']
        values = [metrics[key] for key in ('mae_pooled', 'mape_pooled', 'rmse_pooled')]
        replay[arm] = dict(status='BEST_ON_REPLAY_PASS', best_stage_epoch=comparable['best_stage_epochs'][arm],
            actual=values, expected=values, mask_value=0., batch_partition=[2, 2], atol=1e-5, rtol=1e-6)
    source_metrics = original['cases']['SOURCE_MIN']['metrics']['all_nodes']
    source_values = [source_metrics[key] for key in ('mae_pooled', 'mape_pooled', 'rmse_pooled')]
    identity = dict(schema='min_forecast_fusion_warmstart_run_v1', engineering_check=False,
        arm_modes={'STATE': 'state', 'EVENT': 'event'}, batch_size=2,
        source_exact_resume=False, source_optimizer_restored=False, source_scheduler_restored=False,
        full_horizon_loss_every_update=True,
        protocol=dict(train_samples=100, val_samples=4, stations=4, fusion_spec={'max_delta': .1}),
        source=dict(best_epoch=28, completed_epoch=30, best_model_sha256='a'*64,
                    files_sha256={'A/last.pt': 'b'*64}, best_metrics=source_values, reference_metrics=source_values),
        package={'incident_all.npy': 'c'*64}, graph_inputs_sha256={'graph.json': 'd'*64})
    protected = {'old/checkpoint': 'e'*64}
    preservation = dict(files=dict(status='SOURCE_ARTIFACTS_PRESERVED', files=1, before=protected, after=protected),
        models={arm: dict(status='MODEL_STATE_PRESERVED', model_state_sha256='f'*64,
                         mode_gradients_hooks_graph_rng_preserved=True) for arm in replay})
    summary = dict(schema=analysis.SOURCE_SCHEMA, status=analysis.SOURCE_STATUS, validation_rows=4, engineering_only=False,
        best_stage_epochs=comparable['best_stage_epochs'], source_best_epoch=28, cases=list(analysis.CASES),
        all_nodes={case: row['metrics']['all_nodes'] for case, row in original['cases'].items()},
        comparisons={name: row['metrics']['all_nodes'] for name, row in original['comparisons'].items()},
        replay_passed=True, on_exactly_restored=True, training_comparability=comparable,
        source_artifacts_preserved=True, new_inference_calls={case: 0 if case == 'SOURCE_MIN' else 12 if case == 'EVENT_ON' else 2 for case in analysis.CASES},
        optimizer_updates=0, test_evaluated=False, source_scope='conditional_offline_validation_warmstart_checkpoint_sensitivity')
    donors = np.empty((4, 4), np.int64)
    for r, sample in enumerate(indices):
        order = np.random.default_rng(np.random.SeedSequence([20261015, int(sample)])).permutation(4)
        donors[r, order] = np.roll(order, 1)
    changed = direct != np.take_along_axis(direct, donors, 1)
    weights = {}
    for case in analysis.CASES[:-1]:
        weight = 1. if case == 'EVENT_FUSION_OFF' or event_epoch == 0 and case != 'STATE_ON' else 1.01
        weights[case] = dict(mean=np.full((12, 10), weight).tolist(),
            absolute_delta_from_one_mean=np.full((12, 10), abs(weight-1)).tolist(),
            minimum=np.full((12, 10), weight).tolist(), maximum=np.full((12, 10), weight).tolist(),
            count_per_horizon_component=16, component_increment_abs_mean_by_horizon_component=np.zeros((12, 10)).tolist(),
            weighted_hidden_change_abs_mean_by_horizon=np.zeros(12).tolist())
    mechanisms = dict(cases={case: {} for case in analysis.CASES[:-1]}, component_order=list(analysis.COMPONENT_ORDER),
        fusion_weights=weights, location_shuffle=dict(seed=20261015, schema='full_station_relation_cycle_v1', uses_targets=False,
            zero_relation_rows_permuted=True, fixed_semantic_categories=True, rows=4, moved_vector_positions=16,
            numerically_changed_vectors=int(changed.sum()), numerically_changed_rows=4,
            support_changed_positions=int(changed.sum()), support_changed_rows=int(changed.any(1).sum())))
    region_keys = {name: 'region_'+str(i) for i, name in enumerate(regions)}
    for name, value in (('summary.json', summary), ('analysis.json', original), ('replay.json', replay),
        ('comparability.json', comparable), ('preservation.json', preservation), ('mechanisms.json', mechanisms),
        ('provenance.json', {'training_identity': identity}), ('region_definitions.json', {'masks': region_keys})):
        analysis.archive.write_json(source/name, value)
    np.savez_compressed(source/'paired_predictions.npz', **predictions, target=target, valid=valid,
        sample_indices=indices, station_ids=stations, location_donor_station_indices=donors,
        **{region_keys[name]: mask for name, mask in regions.items()})
    return source


def edit_json(source, name, mutate):
    value = analysis.archive.read_json(source/name)
    mutate(value)
    analysis.archive.write_json(source/name, value)


def edit_archive(source, mutate):
    with np.load(source/'paired_predictions.npz', allow_pickle=False) as saved:
        values = {name: saved[name] for name in saved.files}
    mutate(values)
    np.savez_compressed(source/'paired_predictions.npz', **values)


class ForecastFusionGainAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=analysis.REPO)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def run_source(self, source, name='output'):
        with redirect_stdout(io.StringIO()):
            return analysis.main(['--evaluation-dir', str(source), '--output-dir', str(self.root/name)])

    def test_complete_archive_analysis_preserves_bytes_and_paper_baseline_scope(self):
        source = fixture(self.root)
        before = {p.name: p.read_bytes() for p in source.iterdir()}
        summary = self.run_source(source)
        self.assertEqual(summary['status'], analysis.STATUS)
        self.assertEqual(summary['new_inference_calls'], 0)
        self.assertEqual(summary['optimizer_updates'], 0)
        self.assertFalse(summary['test_evaluated'])
        self.assertTrue(summary['source_artifacts_preserved'])
        self.assertEqual(before, {p.name: p.read_bytes() for p in source.iterdir()})
        output = self.root/'output'
        for name in ('analysis.json', 'summary.json', 'report.md', 'provenance.json', 'preservation.json',
                     'paper_evidence_manifest.json', 'mechanisms.json', 'flow.csv', 'current_bias.csv', 'source_bias.csv',
                     'regions.csv', 'roads.csv', 'cross.csv', 'horizons.csv', 'group_horizons.csv', 'directions.csv', 'overlap.csv'):
            self.assertTrue((output/name).is_file(), name)
        for name in ('current_bias.csv', 'source_bias.csv', 'cross.csv', 'directions.csv', 'overlap.csv'):
            self.assertGreater(len((output/name).read_text(encoding='utf-8-sig').splitlines()), 1, name)
        provenance = analysis.archive.read_json(output/'provenance.json')
        self.assertFalse(provenance['checkpoint_loaded'])
        self.assertFalse(provenance['raw_data_read'])
        self.assertFalse(provenance['historical_signature_verified'])
        manifest = analysis.archive.read_json(output/'paper_evidence_manifest.json')
        self.assertEqual(manifest['primary_baseline']['name'], 'original_IGSTGNN')
        self.assertFalse(manifest['primary_baseline']['predictions_in_this_archive'])
        report = (output/'report.md').read_text(encoding='utf-8')
        for phrase in ('当前MIN偏差与流量', '当前归档没有', '不能因果归因', '不得仅因验证集最好', '原训练顺序未重新读取',
                       '四种作用的正负抵消与预测响应', '预测有符号变化均值', '上调占比', '响应占比',
                       '完整融合收益', '共同缩放收益', '分量差异化收益', '报告内容收益'):
            self.assertIn(phrase, report)

    def test_no_torch_model_or_training_import_in_fresh_process(self):
        code = ('import sys; import experiments.chronological.analyze_min_forecast_fusion_gain; '
                'assert "torch" not in sys.modules; '
                'assert not any(k.startswith("src.models") for k in sys.modules); '
                'assert "experiments.chronological.train_min_forecast_fusion" not in sys.modules')
        process = subprocess.run([sys.executable, '-X', 'utf8', '-c', code], cwd=analysis.REPO, capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stdout+process.stderr)

    def test_default_latest_completed_selection_and_broken_newest_PASS_not_silently_skipped(self):
        self.assertEqual(analysis.parse_args([]).run_dir, analysis.DEFAULT_RUN.resolve())
        (self.root/'active.lock').write_bytes(b'0')
        old = fixture(self.root, 'forecast_fusion_evaluation_20261015_110000_old')
        new = fixture(self.root, 'forecast_fusion_evaluation_20261015_120000_new')
        bad = self.root/'forecast_fusion_evaluation_20261015_130000_incomplete'
        bad.mkdir()
        analysis.archive.write_json(bad/'summary.json', {'status': 'FAILED'})
        args = analysis.parse_args(['--run-dir', str(self.root)])
        self.assertEqual(analysis.select_source(args), new)
        edit_json(new, 'replay.json', lambda r: r['EVENT'].update(status='FAILED'))
        with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'replay did not pass'):
            analysis.main(['--run-dir', str(self.root), '--output-dir', str(self.root/'failed')])
        self.assertTrue((old/'summary.json').exists())
        self.assertEqual((self.root/'active.lock').read_bytes(), b'0')

    def test_stage_zero_selected_source_is_legal_but_wrong_selected_epoch_is_rejected(self):
        source = fixture(self.root, event_epoch=0)
        info = analysis.load_and_verify(source)
        self.assertEqual(info['summary']['best_stage_epochs']['EVENT'], 0)
        edit_json(source, 'replay.json', lambda r: r['EVENT'].update(best_stage_epoch=99))
        with self.assertRaisesRegex(ValueError, 'selected-stage epochs'):
            analysis.load_and_verify(source)

    def test_prediction_mask_axis_and_donor_corruption_are_rejected(self):
        mutations = (
            lambda r: r['EVENT_ON'].__setitem__((0, 1, 0), np.nan),
            lambda r: r.update(valid=r['valid'].astype(np.float32)),
            lambda r: r['valid'].__setitem__((0, 1, 0), False),
            lambda r: r['station_ids'].__setitem__(0, r['station_ids'][1]),
            lambda r: r['sample_indices'].__setitem__(0, r['sample_indices'][0]+8),
            lambda r: r['location_donor_station_indices'].__setitem__((0, 0), 100),
            lambda r: r.update(location_donor_station_indices=np.roll(r['location_donor_station_indices'], 1, 0)),
        )
        for index, mutation in enumerate(mutations):
            with self.subTest(index=index):
                source = fixture(self.root, 'forecast_fusion_evaluation_bad_'+str(index))
                edit_archive(source, mutation)
                with self.assertRaises(ValueError):
                    analysis.load_and_verify(source)

    def test_legal_road_mask_swap_and_saved_metric_summary_drift_are_rejected(self):
        source = fixture(self.root)
        mapping = analysis.archive.read_json(source/'region_definitions.json')['masks']
        edit_archive(source, lambda r: r.update({mapping['road:WEST']: r[mapping['road:EAST']], mapping['road:EAST']: r[mapping['road:WEST']]}))
        with self.assertRaisesRegex(ValueError, 'Saved metric mismatch'):
            analysis.load_and_verify(source)
        for index, filename in enumerate(('analysis.json', 'summary.json')):
            source = fixture(self.root, 'forecast_fusion_evaluation_metric_'+str(index))
            if filename == 'analysis.json':
                edit_json(source, filename, lambda r: r['cases']['EVENT_ON']['metrics']['all_nodes'].update(mae_pooled=999.))
            else:
                edit_json(source, filename, lambda r: r['comparisons']['fusion_dependency'].update(mae_gain_pooled=999.))
            with self.assertRaisesRegex(ValueError, 'Saved metric mismatch'):
                analysis.load_and_verify(source)

    def test_batch_source_budget_preservation_weight_and_hard_certificates_rejected(self):
        mutations = (
            ('replay.json', lambda r: r['STATE'].update(batch_partition=[4])),
            ('preservation.json', lambda r: r['files'].update(after={'changed': 'a'*64})),
            ('analysis.json', lambda r: r['fixed_hard_windows'].update(sample_indices=[9999])),
            ('provenance.json', lambda r: r['training_identity']['source'].update(best_epoch=999)),
            ('mechanisms.json', lambda r: r['fusion_weights']['EVENT_ON']['minimum'][0].__setitem__(0, .1)),
            ('comparability.json', lambda r: r.update(same_completed_training_budget=False)),
            ('replay.json', lambda r: r['EVENT'].update(atol=.1)),
        )
        for index, (filename, mutate) in enumerate(mutations):
            with self.subTest(filename=filename):
                source = fixture(self.root, 'forecast_fusion_evaluation_certificate_'+str(index))
                edit_json(source, filename, mutate)
                with self.assertRaises(ValueError):
                    analysis.load_and_verify(source)

    def test_existing_output_source_descendant_and_missing_archive_are_rejected(self):
        source = fixture(self.root)
        for output in (source, source/'child', self.root):
            with self.subTest(output=output), self.assertRaises(ValueError):
                self.run_source(source, str(output))
        (source/'failure.json').write_text('{}', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'failure record'):
            analysis.load_and_verify(source)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            analysis.parse_args(['--run-dir', str(self.root), '--evaluation-dir', str(source)])

    def test_detects_source_change_during_analysis_and_keeps_failure_evidence(self):
        source = fixture(self.root)
        original = analysis.analyze_predictions
        def mutate(*args, **kwargs):
            result = original(*args, **kwargs)
            (source/'external_change.txt').write_text('changed', encoding='utf-8')
            return result
        with patch.object(analysis, 'analyze_predictions', side_effect=mutate):
            with self.assertRaisesRegex(ValueError, 'changed during read-only'):
                self.run_source(source)
        preserved = analysis.archive.read_json(self.root/'output'/'preservation.json')
        self.assertFalse(preserved['source_artifacts_preserved'])
        self.assertTrue((self.root/'output'/'failure.json').exists())


if __name__ == '__main__':
    unittest.main()
