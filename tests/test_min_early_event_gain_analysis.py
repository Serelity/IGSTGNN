"""Archived prediction consistency and source protection without model imports."""
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

from experiments.chronological import analyze_min_early_event_gain as analysis
from experiments.chronological.min_early_event_metrics import summarize


def fixture(root, *, name='early_event_evaluation_20261014_120000_fixture', engineering=False, identity=True):
    source = root/name
    source.mkdir()
    target = np.broadcast_to(np.array([1., 10., 80., 220.], np.float32), (4, 12, 4)).copy()
    target += np.arange(4, dtype=np.float32)[:, None, None]
    target[0, 0, 0] = 0.
    row = np.arange(4, dtype=np.float32)[:, None, None]
    state = target+np.array([2., -1., 0., 3.], np.float32)+row*.2
    on = target+np.array([1., -2., 0., 2.], np.float32)+row*.1
    off = target+np.array([4., -1., 1., 1.], np.float32)+row*.15
    predictions = dict(STATE_ON=state, EVENT_ON=on, EVENT_OFF=off,
                       EVENT_SEMANTIC_UNK=on+.1, EVENT_LOCATION_SHUFFLE=on+.3)
    valid, indices, stations = target != 0., np.arange(100, 104, dtype=np.int64), np.arange(1001, 1005, dtype=np.int64)
    direct = np.broadcast_to([True, False, False, False], (4, 4)).copy()
    potential = np.broadcast_to([False, True, False, False], (4, 4)).copy()
    regions = dict(all_nodes=np.ones((4, 4), bool), direct_report_nodes=direct, potential_only=potential,
                   remaining_nodes=~(direct | potential), candidate_potential=direct | potential,
                   empty=np.zeros((4, 4), bool), **{'road:WEST': np.array([True, True, False, False])})
    result = summarize(predictions, target, valid, regions, indices)
    budget = dict(status='COMMON_PREFIX_TRAIN_ORDER_VERIFIED', common_epoch=2,
        completed_epochs={'STATE': 2, 'EVENT': 2}, global_updates={'STATE': 4, 'EVENT': 4},
        best_epochs={'STATE': 1, 'EVENT': 1}, same_completed_training_budget=True,
        common_prefix_training_order_verified=True,
        whole_model_scope='same_protocol_equal_completed_budget_independent_best_validation')
    result['training_comparability'] = copy.deepcopy(budget)
    for comparison in result['comparisons'].values():
        if comparison['scope'] == 'independently_trained_best_validation':
            comparison['scope'] = budget['whole_model_scope']
    summary = dict(schema=analysis.SOURCE_SCHEMA, status=analysis.SOURCE_STATUS, validation_rows=4, batch_size=2,
        best_epochs=budget['best_epochs'], cases=list(predictions), forecast_audit=False,
        replay_passed=True, on_exactly_restored=True, source_artifacts_preserved=True,
        training_updates=0, test_evaluated=False, engineering_only=engineering,
        training_comparability=copy.deepcopy(budget),
        all_nodes={case: value['metrics']['all_nodes'] for case, value in result['cases'].items()},
        comparisons={key: value['metrics']['all_nodes'] for key, value in result['comparisons'].items()})
    replay = {}
    for arm in ('STATE', 'EVENT'):
        metric = summary['all_nodes'][arm+'_ON']
        values = [metric[key] for key in ('mae_pooled', 'mape_pooled', 'rmse_pooled')]
        replay[arm] = dict(status='BEST_ON_REPLAY_PASS', best_epoch=1, actual=values, expected=values,
            mask_value=0., batch_partition=[2, 2], atol=1e-5, rtol=1e-6)
    comparable = dict(budget, test_evaluated=False)
    if identity:
        comparable['identity'] = dict(schema='min_early_event_published_run_v1', engineering_check=engineering,
            batch_size=2, protocol=dict(train_samples=100, val_samples=837 if engineering else 4, stations=4))
    protected = {'old/checkpoint': 'a'*64}
    preservation = dict(files=dict(status='SOURCE_ARTIFACTS_PRESERVED', files=1, before=protected, after=protected),
        models={arm: dict(status='MODEL_STATE_PRESERVED', model_state_sha256='b'*64,
                         mode_gradients_hooks_graph_rng_preserved=True) for arm in replay})
    region_keys = {name: 'region_'+str(i) for i, name in enumerate(regions)}
    for filename, value in (('summary.json', summary), ('analysis.json', result), ('replay.json', replay),
                            ('comparability.json', comparable), ('preservation.json', preservation),
                            ('region_definitions.json', {'masks': region_keys}),
                            ('provenance.json', {'station_ids': stations.tolist()})):
        analysis.write_json(source/filename, value)
    np.savez_compressed(source/'paired_predictions.npz', **predictions, target=target, valid=valid,
        sample_indices=indices, station_ids=stations, location_donor_station_indices=np.tile([1, 2, 3, 0], (4, 1)),
        **{region_keys[name]: value for name, value in regions.items()})
    return source


def edit_json(source, filename, mutate):
    value = analysis.read_json(source/filename)
    mutate(value)
    analysis.write_json(source/filename, value)


def edit_archive(source, mutate):
    with np.load(source/'paired_predictions.npz', allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    mutate(arrays)
    np.savez_compressed(source/'paired_predictions.npz', **arrays)


class EarlyEventGainAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=analysis.REPO)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def run_source(self, source, *, output=None):
        args = ['--evaluation-dir', str(source), '--output-dir', str(output or self.root/'analysis')]
        with redirect_stdout(io.StringIO()):
            return analysis.main(args)

    def test_complete_archive_produces_readable_decomposition_and_preserves_every_source_byte(self):
        source = fixture(self.root)
        before = {path.name: path.read_bytes() for path in source.iterdir()}
        result = self.run_source(source)
        self.assertEqual(result['status'], 'MIN_EARLY_EVENT_GAIN_ANALYSIS_PASS')
        self.assertFalse(result['engineering_only'])
        self.assertEqual(result['validation_rows'], 4)
        self.assertEqual(result['new_inference_calls'], 0)
        self.assertEqual(result['optimizer_updates'], 0)
        self.assertFalse(result['test_evaluated'])
        self.assertTrue(result['source_artifacts_preserved'])
        self.assertAlmostEqual(result['all_nodes']['T']['mae_gain_pooled'],
            result['all_nodes']['P']['mae_gain_pooled']+result['all_nodes']['R']['mae_gain_pooled'])
        self.assertEqual(before, {path.name: path.read_bytes() for path in source.iterdir()})
        for name in ('analysis.json', 'summary.json', 'provenance.json', 'preservation.json', 'report.md',
                     'flow.csv', 'bias.csv', 'regions.csv', 'roads.csv', 'horizons.csv', 'cross.csv',
                     'gain_overlap.csv', 'fixed_hard.csv', 'cross_horizons.csv', 'road_horizons.csv', 'region_horizons.csv'):
            self.assertTrue((self.root/'analysis'/name).is_file(), name)
        provenance = analysis.read_json(self.root/'analysis'/'provenance.json')
        self.assertFalse(provenance['historical_signature_verified'])
        self.assertFalse(provenance['new_checkpoint_replay'])
        report = (self.root/'analysis'/'report.md').read_text(encoding='utf-8')
        for text in ('T = P + R', '真实流量区间', '逐预测时距', '原评估固定困难窗口', '总体收益与路径收益是否重合'):
            self.assertIn(text, report)

    def test_latest_completed_selection_ignores_incomplete_but_does_not_fallback_on_invalid_PASS(self):
        (self.root/'active.lock').write_bytes(b'0')
        older = fixture(self.root, name='early_event_evaluation_20261014_110000_old')
        newer = fixture(self.root, name='early_event_evaluation_20261014_120000_new')
        unfinished = self.root/'early_event_evaluation_20261014_130000_incomplete'
        unfinished.mkdir()
        analysis.write_json(unfinished/'summary.json', {'status': 'FAILED'})
        args = analysis.parse_args(['--run-dir', str(self.root)])
        self.assertEqual(analysis.select_source(args), newer)
        edit_json(newer, 'replay.json', lambda value: value['EVENT'].update(status='FAILED'))
        with self.assertRaisesRegex(ValueError, 'replay did not pass'):
            with redirect_stdout(io.StringIO()):
                analysis.main(['--run-dir', str(self.root), '--output-dir', str(self.root/'rejected')])
        self.assertTrue((older/'summary.json').is_file())
        self.assertEqual((self.root/'active.lock').read_bytes(), b'0')

    def test_no_completed_archive_is_rejected_without_creating_prediction_outputs(self):
        (self.root/'active.lock').write_bytes(b'0')
        with self.assertRaisesRegex(ValueError, 'No completed PASS'):
            analysis.main(['--run-dir', str(self.root)])
        self.assertEqual({path.name for path in self.root.iterdir()}, {'active.lock'})
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            analysis.parse_args(['--run-dir', str(self.root), '--evaluation-dir', str(self.root)])

    def test_bad_status_schema_replay_protection_and_stale_PASS_are_blocked(self):
        mutations = [
            ('summary.json', lambda value: value.update(schema='old_schema'), 'schema'),
            ('summary.json', lambda value: value.update(status='FAILED'), 'non-PASS'),
            ('summary.json', lambda value: value.update(replay_passed=False), 'certificate'),
            ('summary.json', lambda value: value.update(source_artifacts_preserved=False), 'certificate'),
            ('summary.json', lambda value: value.update(validation_rows=837), 'counts'),
            ('replay.json', lambda value: value['EVENT'].update(status='FAILED'), 'replay did not pass'),
            ('replay.json', lambda value: value['EVENT'].update(best_epoch=2), 'best epochs'),
            ('replay.json', lambda value: value['EVENT'].update(batch_partition=[4]), 'batch partition'),
            ('replay.json', lambda value: value['EVENT'].update(atol=1.), 'tolerances'),
            ('preservation.json', lambda value: value['files'].update(after={'bad': 'c'*64}), 'preservation'),
            ('preservation.json', lambda value: value['models']['EVENT'].update(status='FAILED'), 'preservation'),
        ]
        for i, (filename, mutate, error) in enumerate(mutations):
            with self.subTest(filename=filename, i=i):
                source = fixture(self.root, name='case_'+str(i))
                edit_json(source, filename, mutate)
                output = self.root/('rejected_'+str(i))
                with self.assertRaisesRegex(ValueError, error):
                    self.run_source(source, output=output)
                self.assertFalse((output/'summary.json').exists())
                self.assertTrue((output/'failure.json').is_file())
                self.assertTrue(analysis.read_json(output/'preservation.json')['source_artifacts_preserved'])
        source = fixture(self.root, name='stale')
        analysis.write_json(source/'failure.json', {'status': 'FAILED'})
        with self.assertRaisesRegex(ValueError, 'stale PASS'):
            self.run_source(source, output=self.root/'stale_output')

    def test_bad_npz_arrays_ids_masks_and_saved_metric_drift_are_blocked(self):
        mutations = [
            (lambda arrays: arrays.pop('EVENT_OFF'), 'NPZ fields'),
            (lambda arrays: arrays.update(EVENT_ON=arrays['EVENT_ON'][:, :, :3]), 'prediction numeric axes'),
            (lambda arrays: arrays['EVENT_ON'].__setitem__((0, 0, 0), np.nan), 'prediction numeric axes'),
            (lambda arrays: arrays['target'].__setitem__((0, 1, 0), np.nan), 'numeric target'),
            (lambda arrays: arrays.update(valid=arrays['valid'].astype(np.float32)), 'validity mask'),
            (lambda arrays: arrays['valid'].__setitem__((0, 0, 0), True), 'validity differs'),
            (lambda arrays: arrays['sample_indices'].__setitem__(1, arrays['sample_indices'][0]), 'sample IDs'),
            (lambda arrays: arrays['station_ids'].__setitem__(1, arrays['station_ids'][0]), 'station IDs'),
            (lambda arrays: arrays['region_2'].__setitem__(slice(None), True), 'disjoint'),
            (lambda arrays: arrays['EVENT_ON'].__iadd__(.01), 'Saved metric mismatch'),
        ]
        for i, (mutate, error) in enumerate(mutations):
            with self.subTest(i=i):
                source = fixture(self.root, name='bad_array_'+str(i))
                edit_archive(source, mutate)
                with self.assertRaisesRegex(ValueError, error):
                    self.run_source(source, output=self.root/('bad_output_'+str(i)))
        source = fixture(self.root, name='bad_metric')
        edit_json(source, 'analysis.json', lambda value: value['cases']['EVENT_ON']['metrics']['all_nodes'].update(mae_pooled=999.))
        with self.assertRaisesRegex(ValueError, 'Saved metric mismatch'):
            self.run_source(source, output=self.root/'bad_metric_output')

    def test_legal_disjoint_region_swap_is_detected_by_original_region_metrics(self):
        source = fixture(self.root)
        def swap(arrays):
            arrays['region_1'], arrays['region_2'] = arrays['region_2'].copy(), arrays['region_1'].copy()
        edit_archive(source, swap)
        with self.assertRaisesRegex(ValueError, 'region.direct_report_nodes'):
            self.run_source(source)

    def test_fixed_hard_ids_are_reused_and_mismatched_saved_certificate_is_blocked(self):
        source = fixture(self.root)
        original = analysis.read_json(source/'analysis.json')['fixed_hard_windows']['sample_indices']
        result = self.run_source(source)
        saved = analysis.read_json(Path(result['output_dir'])/'analysis.json')
        self.assertEqual(saved['fixed_hard_windows']['sample_indices'], original)
        edit_json(source, 'analysis.json', lambda value: value['fixed_hard_windows'].update(sample_indices=[100]))
        with self.assertRaisesRegex(ValueError, 'fixed_hard_windows'):
            self.run_source(source, output=self.root/'bad_hard')

    def test_unequal_budget_downgrades_and_missing_identity_does_not_claim_raw_axis_verification(self):
        source = fixture(self.root, identity=False)
        def change_budget(value):
            budget = value['training_comparability'] if 'training_comparability' in value else value
            budget['completed_epochs']['EVENT'] = 3
            budget['global_updates']['EVENT'] = 6
            budget['same_completed_training_budget'] = False
            budget['whole_model_scope'] = 'unequal_completed_budget_exploratory_independent_best_validation'
        for name in ('summary.json', 'analysis.json', 'comparability.json'):
            edit_json(source, name, change_budget)
        result = self.run_source(source)
        self.assertIn('unequal_completed_budget', result['source_scope'])
        saved = analysis.read_json(Path(result['output_dir'])/'analysis.json')
        self.assertEqual(saved['archive_axis_scope'], 'archived_axes_only_no_new_release_axis_certification')
        edit_json(source, 'comparability.json', lambda value: value.update(same_completed_training_budget=True))
        with self.assertRaisesRegex(ValueError, 'certificate mismatch|training_comparability|budget'):
            self.run_source(source, output=self.root/'lying_budget')

    def test_engineering_scope_is_inherited_and_identity_mismatch_is_rejected(self):
        source = fixture(self.root, engineering=True)
        result = self.run_source(source)
        self.assertTrue(result['engineering_only'])
        self.assertEqual(result['validation_rows'], 4)
        edit_json(source, 'summary.json', lambda value: value.update(engineering_only=False))
        with self.assertRaisesRegex(ValueError, 'Engineering flag'):
            self.run_source(source, output=self.root/'mislabelled')

    def test_output_cannot_overwrite_source_ancestors_or_checkpoint_directories(self):
        source = fixture(self.root)
        checkpoint = self.root/'STATE'
        checkpoint.mkdir()
        (checkpoint/'last.pt').write_bytes(b'do not load')
        candidates = (source, source/'nested', self.root, self.root.parent, checkpoint/'analysis', self.root/'already_exists')
        candidates[-1].mkdir()
        before = {str(path.relative_to(self.root)): path.read_bytes() for path in self.root.rglob('*') if path.is_file()}
        for output in candidates:
            with self.subTest(output=output), self.assertRaisesRegex(ValueError, 'Output'):
                self.run_source(source, output=output)
        self.assertEqual(before, {str(path.relative_to(self.root)): path.read_bytes() for path in self.root.rglob('*') if path.is_file()})

    def test_source_change_during_analysis_prevents_PASS_and_records_failed_preservation(self):
        source = fixture(self.root)
        original = analysis.analyze_predictions
        def changing(*args, **kwargs):
            result = original(*args, **kwargs)
            (source/'provenance.json').write_text('{"external_change": true}', encoding='utf-8')
            return result
        with patch.object(analysis, 'analyze_predictions', side_effect=changing):
            with self.assertRaisesRegex(ValueError, 'Source artifacts changed'):
                self.run_source(source)
        self.assertFalse((self.root/'analysis'/'summary.json').exists())
        self.assertEqual(analysis.read_json(self.root/'analysis'/'preservation.json')['status'], 'SOURCE_ARTIFACTS_CHANGED')

    def test_existing_run_lock_is_readonly_and_concurrent_analysis_is_rejected(self):
        source = fixture(self.root)
        lock = self.root/'active.lock'
        lock.write_bytes(b'0')
        with analysis.readonly_run_lock(self.root):
            with self.assertRaisesRegex(ValueError, 'Source run is active'):
                with redirect_stdout(io.StringIO()):
                    analysis.main(['--run-dir', str(self.root)])
        before = lock.read_bytes()
        with redirect_stdout(io.StringIO()):
            result = analysis.main(['--run-dir', str(self.root), '--output-dir', str(self.root/'locked_success')])
        self.assertEqual(before, lock.read_bytes())
        self.assertTrue(result['source_artifacts_preserved'])

    def test_subprocess_import_guard_proves_no_torch_model_training_or_raw_loader_imports(self):
        source = fixture(self.root)
        code = '''import importlib.abc, sys
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        blocked = ('torch', 'src.models', 'src.utils.igstgnn_published', 'data.xtraffic.prepare_splits',
                   'experiments.chronological.train', 'experiments.chronological.evaluate',
                   'experiments.chronological.smoke', 'experiments.chronological.probe_min')
        if any(fullname == key or fullname.startswith(key+'.') or fullname.startswith(key+'_') for key in blocked):
            raise AssertionError('Forbidden analysis import: '+fullname)
sys.meta_path.insert(0, Guard())
from experiments.chronological import analyze_min_early_event_gain as module
module.main(sys.argv[1:])
assert 'torch' not in sys.modules
assert not any(key.startswith('src.models') for key in sys.modules)
'''
        result = subprocess.run([sys.executable, '-X', 'utf8', '-c', code, '--evaluation-dir', str(source),
                                 '--output-dir', str(self.root/'subprocess_analysis')],
                                cwd=analysis.REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                encoding='utf-8', timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout)


if __name__ == '__main__':
    unittest.main()
