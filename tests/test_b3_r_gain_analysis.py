"""Independent scalar-oracle tests for offline B3-R prediction gain analysis."""

import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from experiments.chronological import analyze_b3_r_path_probe as analysis


REPO = Path(__file__).resolve().parents[1]
SCHEMA = 'b3r_d_path_probe_v1'


def scalar_oracle(on, off, target, valid, region):
    """Sum individual observations with 1/(H * valid_count_at_horizon)."""
    batch, horizons, nodes, channels = on.shape
    region = np.broadcast_to(region[:, None, :, None], on.shape) if region.ndim == 2 else region
    global_counts, counts = [0]*horizons, [0]*horizons
    on_error, off_error, positive, negative, pred_change = [[0.]*horizons for _ in range(5)]
    for b in range(batch):
        for h in range(horizons):
            for n in range(nodes):
                for c in range(channels):
                    if not valid[b, h, n, c]:
                        continue
                    global_counts[h] += 1
                    if not region[b, h, n, c]:
                        continue
                    counts[h] += 1
                    a = abs(float(on[b, h, n, c])-float(target[b, h, n, c]))
                    z = abs(float(off[b, h, n, c])-float(target[b, h, n, c]))
                    on_error[h] += a
                    off_error[h] += z
                    positive[h] += max(z-a, 0.)
                    negative[h] += max(a-z, 0.)
                    pred_change[h] += abs(float(on[b, h, n, c])-float(off[b, h, n, c]))
    def own_macro(values):
        return sum(value/count for value, count in zip(values, counts))/horizons if all(counts) else None
    on_mae, off_mae = own_macro(on_error), own_macro(off_error)
    gain_positive, gain_negative = own_macro(positive), own_macro(negative)
    gross = None if gain_positive is None else gain_positive+gain_negative
    net = None if on_mae is None else off_mae-on_mae
    return dict(on_mae_macro=on_mae, off_mae_macro=off_mae, off_minus_on_mae=net,
                gain_positive=gain_positive, gain_negative=gain_negative, gross_error_gain=gross,
                cancellation_fraction=1.-abs(net)/gross if gross else None,
                prediction_abs_change_horizon_macro_mean=own_macro(pred_change),
                global_gain_contribution=sum((z-a)/(horizons*count)
                                             for a, z, count in zip(on_error, off_error, global_counts)),
                global_gain_positive=sum(value/(horizons*count) for value, count in zip(positive, global_counts)),
                global_gain_negative=sum(value/(horizons*count) for value, count in zip(negative, global_counts)),
                global_valid_weight_fraction=sum(count/(horizons*total) for count, total in zip(counts, global_counts)),
                per_horizon_valid_count=counts,
                per_horizon_on_mae=[value/count if count else None for value, count in zip(on_error, counts)],
                per_horizon_off_mae=[value/count if count else None for value, count in zip(off_error, counts)])


def source_metric(on, off, target, valid, mask):
    oracle = scalar_oracle(on, off, target, valid, mask)
    chosen = valid & mask[:, None, :, None]
    delta = np.abs(off.astype(np.float64)-on.astype(np.float64))[chosen]
    count = int(chosen.sum())
    on_h, off_h = oracle['per_horizon_on_mae'], oracle['per_horizon_off_mae']
    return dict(status='available' if all(oracle['per_horizon_valid_count']) else 'unavailable',
                on_mae_macro=oracle['on_mae_macro'], off_mae_macro=oracle['off_mae_macro'],
                off_minus_on_mae=oracle['off_minus_on_mae'],
                counts=oracle['per_horizon_valid_count'], on_mae_by_horizon=on_h, off_mae_by_horizon=off_h,
                off_minus_on_by_horizon=[b-a if a is not None else None for a, b in zip(on_h, off_h)],
                prediction_change=dict(status='available' if count else 'unavailable', count=count,
                                       absolute_mean=float(delta.mean()) if count else None,
                                       absolute_max=float(delta.max()) if count else None))


def main_arrays():
    target = np.full((3, 12, 4, 1), 10., dtype=np.float32)
    off_residual = np.array([[2., -2., 0., 4.], [-3., 3., 5., 0.], [2., -4., 0., 1.]])
    on_residual = np.array([[1., -3., 0., -2.], [1., 0., 4., 2.], [-2., -1., 1., 1.]])
    on = target+on_residual[:, None, :, None].astype(np.float32)
    off = target+off_residual[:, None, :, None].astype(np.float32)
    valid = np.zeros_like(target, dtype=bool)
    for h in range(12):
        if h % 3 == 0:
            valid[:, h] = True
        elif h % 3 == 1:
            valid[0, h, :3] = True
            valid[1, h, 3] = True
        else:
            valid[0, h] = True
            valid[1, h, :2] = True
    return dict(on=on, off=off, target=target, valid=valid,
                report_associated=np.array([[True, False, False, True],
                                            [False, True, False, False], [False]*4]),
                candidate_potential=np.array([[True, True, False, False],
                                              [True, True, True, False], [False]*4]),
                operator_mask=np.array([True, True, True, False]),
                sample_indices=np.array([10, 20, 30], dtype=np.int64),
                t0=np.array(['2020-01-01T12:00:00', '2020-01-01T12:00:00', '2020-01-01T12:05:00']),
                final_on_exactly_restored=np.array(True))


def small_arrays(on_values, off_values, associated=None):
    nodes = len(on_values)
    shape = (1, 12, nodes, 1)
    return dict(on=np.broadcast_to(np.array(on_values, dtype=np.float32)[None, None, :, None], shape).copy(),
                off=np.broadcast_to(np.array(off_values, dtype=np.float32)[None, None, :, None], shape).copy(),
                target=np.zeros(shape, dtype=np.float32), valid=np.ones(shape, dtype=bool),
                report_associated=np.array([associated if associated is not None else [False]*nodes]),
                candidate_potential=np.array([associated if associated is not None else [False]*nodes]),
                operator_mask=np.ones(nodes, dtype=bool), sample_indices=np.array([10], dtype=np.int64),
                t0=np.array(['2020-01-01T12:00:00']), final_on_exactly_restored=np.array(True))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def source_fixture(root, arrays=None):
    root.mkdir()
    arrays = main_arrays() if arrays is None else arrays
    np.savez_compressed(root/'paired_predictions.npz', **arrays)
    all_nodes = source_metric(arrays['on'], arrays['off'], arrays['target'], arrays['valid'],
                              np.ones(arrays['report_associated'].shape, dtype=bool))
    replay = dict(status='BEST_ON_WINDOW_REPLAY_PASS', on_mae_macro=all_nodes['on_mae_macro'],
                  saved_mae_macro=all_nodes['on_mae_macro'], atol=1e-4, rtol=1e-6,
                  window_mae_abs_difference_max=0.)
    rows = len(arrays['sample_indices'])
    direct, potential = arrays['report_associated'], arrays['candidate_potential']
    outside = np.broadcast_to(~arrays['operator_mask'][None], direct.shape)
    masks = dict(all_nodes=np.ones_like(direct), report_associated=direct,
                 candidate_potential=potential, potential_only=potential & ~direct,
                 direct_outside_operator=direct & outside, outside_operator=outside,
                 no_report_samples=np.broadcast_to(~direct.any(1)[:, None], direct.shape))
    metrics = {name: source_metric(arrays['on'], arrays['off'], arrays['target'], arrays['valid'], mask)
               for name, mask in masks.items()}
    summary = dict(status='B3R_PATH_PROBE_PASS', schema=SCHEMA, best_epoch=84,
                   scope='engineering_only', final_on_exactly_restored=True, inputs_preserved=True,
                   inference_only=True, tiid_report_enabled=True, source_artifacts_preserved=True, test_read=False,
                   replay=replay, model_state_sha256='c'*64,
                   batch_partition=[min(2, rows)] + ([rows-2] if rows > 2 else []),
                   metrics=metrics)
    provenance = dict(schema=SCHEMA, best_epoch=84, test_read=False,
                      best_model_state_sha256='c'*64, saved_files_sha256={'B3R/last.pt': 'a'*64},
                      identity=dict(engineering_check=True, batch_size=2))
    preservation = dict(before={'B3R/last.pt': 'a'*64}, after={'B3R/last.pt': 'a'*64},
                        source_preserved=True, graph_before={'graph.npz': 'b'*64}, graph_after={'graph.npz': 'b'*64})
    for name, value in [('summary.json', summary), ('provenance.json', provenance),
                        ('replay.json', replay), ('preservation.json', preservation)]:
        write_json(root/name, value)
    return arrays


def source_hashes(root):
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in root.iterdir() if path.is_file()}


class GainAnalysisTests(unittest.TestCase):
    def assert_metric(self, actual, expected):
        for key in ('on_mae_macro', 'off_mae_macro', 'off_minus_on_mae',
                    'prediction_abs_change_horizon_macro_mean', 'global_gain_contribution',
                    'global_gain_positive', 'global_gain_negative', 'global_valid_weight_fraction'):
            if expected[key] is None:
                self.assertIsNone(actual[key], key)
            else:
                self.assertAlmostEqual(actual[key], expected[key], places=12, msg=key)
        self.assertEqual(actual['per_horizon_valid_count'], expected['per_horizon_valid_count'])

    def test_independent_scalar_oracle_checks_horizon_macro_and_three_complete_partitions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arrays = source_fixture(root/'source')
            result = analysis.analyze(root/'source', root/'output', allow_engineering=True)
            self.assertEqual(result['status'], 'B3R_GAIN_ANALYSIS_PASS')
            a, b = arrays['report_associated'], arrays['candidate_potential']
            masks = dict(all_nodes=np.ones_like(a), direct_report_nodes=a,
                         potential_propagation_only_nodes=b & ~a, remaining_nodes=~(a | b))
            for name, mask in masks.items():
                with self.subTest(region=name):
                    expected = scalar_oracle(arrays['on'], arrays['off'], arrays['target'], arrays['valid'], mask)
                    self.assert_metric(result['metrics'][name], expected)
            whole = result['metrics']['all_nodes']
            self.assertEqual(whole['per_horizon_valid_count'], [12, 4, 6]*4)
            self.assertNotAlmostEqual(whole['off_minus_on_mae'],
                                     float((np.abs(arrays['off']-arrays['target'])-
                                            np.abs(arrays['on']-arrays['target']))[arrays['valid']].mean()))
            for key in ('global_gain_contribution', 'global_gain_positive', 'global_gain_negative',
                        'global_valid_weight_fraction'):
                self.assertAlmostEqual(sum(result['metrics'][name][key] for name in list(masks)[1:]),
                                       whole[key], places=12)
            outside = np.broadcast_to(~arrays['operator_mask'], a.shape)
            for name, mask in [('direct_outside_operator', a & outside), ('outside_operator', outside),
                               ('no_report_samples', np.broadcast_to(~a.any(1)[:, None], a.shape))]:
                self.assert_metric(result['metrics'][name],
                                   scalar_oracle(arrays['on'], arrays['off'], arrays['target'], arrays['valid'], mask))

    def test_symmetric_crossing_changes_predictions_without_error_gain_or_cancellation_ratio(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_fixture(root/'source', small_arrays([-2.], [2.]))
            result = analysis.analyze(root/'source', root/'output', allow_engineering=True)
            whole = result['metrics']['all_nodes']
            self.assertEqual(whole['prediction_abs_change_horizon_macro_mean'], 4.)
            self.assertEqual(whole['off_minus_on_mae'], 0.)
            self.assertEqual(whole['gross_error_gain'], 0.)
            self.assertIsNone(whole['cancellation_fraction'])
            for name in ('direct_report_nodes', 'potential_propagation_only_nodes', 'direct_outside_operator'):
                self.assertIsNone(result['metrics'][name]['on_mae_macro'])
                self.assertEqual(result['metrics'][name]['global_gain_contribution'], 0.)
            self.assertEqual(result['movement_metrics']['crossing']['global_valid_weight_fraction'], 1.)

    def test_partial_horizon_region_has_unavailable_local_mae_and_retains_nonzero_global_contribution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arrays = main_arrays()
            arrays['report_associated'][:] = False
            arrays['report_associated'][2, 1] = True
            arrays['candidate_potential'][:] = False
            arrays['candidate_potential'][2, :2] = True
            source_fixture(root/'source', arrays)
            result = analysis.analyze(root/'source', root/'output', allow_engineering=True)
            expected = scalar_oracle(arrays['on'], arrays['off'], arrays['target'], arrays['valid'],
                                     arrays['report_associated'])
            self.assert_metric(result['metrics']['direct_report_nodes'], expected)
            direct = result['metrics']['direct_report_nodes']
            self.assertEqual(direct['per_horizon_valid_count'], [1, 0, 0]*4)
            self.assertIsNone(direct['on_mae_macro'])
            self.assertIsNone(direct['off_minus_on_mae'])
            self.assertGreater(direct['global_gain_contribution'], 0.)

    def test_positive_and_negative_error_gains_survive_complete_net_cancellation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_fixture(root/'source', small_arrays([1., 2.], [2., 1.], [True, False]))
            result = analysis.analyze(root/'source', root/'output', allow_engineering=True)
            row = result['metrics']['all_nodes']
            self.assertEqual(row['off_minus_on_mae'], 0.)
            self.assertEqual(row['gain_positive'], .5)
            self.assertEqual(row['gain_negative'], .5)
            self.assertEqual(row['gross_error_gain'], 1.)
            self.assertEqual(row['cancellation_fraction'], 1.)
            self.assertEqual(result['metrics']['direct_report_nodes']['global_gain_contribution'], .5)
            self.assertEqual(result['metrics']['remaining_nodes']['global_gain_contribution'], -.5)

    def test_bias_and_movement_masks_use_off_residual_and_strict_crossing_exhaustively(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arrays = source_fixture(root/'source')
            result = analysis.analyze(root/'source', root/'output', allow_engineering=True)
            r_on, r_off = arrays['on']-arrays['target'], arrays['off']-arrays['target']
            bias = dict(over=r_off > 0, under=r_off < 0, exact=r_off == 0)
            movement = dict(same_side=r_on*r_off > 0, crossing=r_on*r_off < 0,
                            off_at_target=(r_off == 0) & (r_on != 0),
                            on_at_target=(r_on == 0) & (r_off != 0),
                            both_at_target=(r_on == 0) & (r_off == 0))
            for field, masks in [('bias_metrics', bias), ('movement_metrics', movement)]:
                np.testing.assert_array_equal(sum(mask.astype(int) for mask in masks.values()),
                                              np.ones_like(r_on, dtype=int))
                self.assertEqual(set(result[field]), set(masks))
                for name, mask in masks.items():
                    with self.subTest(field=field, subgroup=name):
                        self.assert_metric(result[field][name],
                                           scalar_oracle(arrays['on'], arrays['off'], arrays['target'], arrays['valid'], mask))
                for key in ('global_gain_contribution', 'global_gain_positive', 'global_gain_negative',
                            'global_valid_weight_fraction'):
                    self.assertAlmostEqual(sum(row[key] for row in result[field].values()),
                                           result['metrics']['all_nodes'][key], places=12)

    def test_duplicate_cutoffs_aggregate_additive_global_contributions_and_source_stays_exact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arrays = source_fixture(root/'source')
            before = source_hashes(root/'source')
            result = analysis.analyze(root/'source', root/'output', allow_engineering=True)
            self.assertEqual(before, source_hashes(root/'source'))
            self.assertEqual(read_json(root/'output'/'analysis.json'), result)
            for name in ('README.md', 'gain_decomposition.csv', 'bias_decomposition.csv',
                         'movement_decomposition.csv', 'window_summary.csv', 'cutoff_summary.csv'):
                self.assertTrue((root/'output'/name).is_file(), name)
            with (root/'output'/'window_summary.csv').open(encoding='utf-8-sig', newline='') as stream:
                windows = list(csv.DictReader(stream))
            with (root/'output'/'cutoff_summary.csv').open(encoding='utf-8-sig', newline='') as stream:
                cutoffs = list(csv.DictReader(stream))
            self.assertEqual(len(windows), 3)
            self.assertEqual(len(cutoffs), 2)
            self.assertEqual([int(row['sample_index']) for row in windows], arrays['sample_indices'].tolist())
            by_time = {row['t0']: float(row['global_gain_contribution']) for row in cutoffs}
            for stamp in set(arrays['t0']):
                self.assertAlmostEqual(by_time[stamp], sum(float(row['global_gain_contribution'])
                                                         for row in windows if row['t0'] == stamp), places=12)
            for rows in (windows, cutoffs):
                self.assertAlmostEqual(sum(float(row['global_gain_contribution']) for row in rows),
                                       result['metrics']['all_nodes']['off_minus_on_mae'], places=12)
            # The optional unique-cutoff sensitivity reweights duplicate rows and recomputes denominators.
            weighted_gain, weighted_on, weighted_off = [], [], []
            for horizon in range(12):
                numerator_gain, numerator_on, numerator_off, denominator = 0., 0., 0., 0.
                for b in range(3):
                    weight = .5 if b < 2 else 1.
                    for node in range(4):
                        if arrays['valid'][b, horizon, node, 0]:
                            on_error = abs(float(arrays['on'][b, horizon, node, 0]-arrays['target'][b, horizon, node, 0]))
                            off_error = abs(float(arrays['off'][b, horizon, node, 0]-arrays['target'][b, horizon, node, 0]))
                            numerator_gain += weight*(off_error-on_error)
                            numerator_on += weight*on_error
                            numerator_off += weight*off_error
                            denominator += weight
                weighted_gain.append(numerator_gain/denominator)
                weighted_on.append(numerator_on/denominator)
                weighted_off.append(numerator_off/denominator)
            sensitivity = result['unique_cutoff_sensitivity_metrics']['all_nodes']
            for field, values in [('off_minus_on_mae', weighted_gain), ('on_mae_macro', weighted_on),
                                  ('off_mae_macro', weighted_off)]:
                self.assertAlmostEqual(sensitivity[field], sum(values)/12., places=12)

    def test_engineering_probe_requires_explicit_opt_in(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_fixture(root/'source')
            with self.assertRaises(ValueError):
                analysis.analyze(root/'source', root/'output')

    def test_inconsistent_source_aggregate_incomplete_or_unrestored_probe_is_rejected(self):
        alterations = [lambda doc: doc.update(status='B3R_PATH_PROBE_FAILED'),
                       lambda doc: doc.update(final_on_exactly_restored=False),
                       lambda doc: doc['metrics']['all_nodes'].update(off_minus_on_mae=100.),
                       lambda doc: doc['metrics']['all_nodes'].update(counts=[1]*12),
                       lambda doc: doc['metrics']['report_associated']['on_mae_by_horizon'].__setitem__(0, 100.),
                       lambda doc: doc['metrics']['candidate_potential']['prediction_change'].update(absolute_mean=100.),
                       lambda doc: doc['metrics']['direct_outside_operator'].update(status='available')]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for i, change in enumerate(alterations):
                with self.subTest(case=i):
                    source = root/f'source_{i}'
                    source_fixture(source)
                    summary = read_json(source/'summary.json')
                    change(summary)
                    write_json(source/'summary.json', summary)
                    with self.assertRaises(ValueError):
                        analysis.analyze(source, root/f'output_{i}', allow_engineering=True)
                    self.assertFalse((root/f'output_{i}'/'analysis.json').exists())

    def test_inconsistent_provenance_preservation_scope_and_failure_record_are_rejected(self):
        edits = [('provenance.json', lambda doc: doc.update(saved_files_sha256={'B3R/last.pt': 'different'})),
                 ('provenance.json', lambda doc: doc.update(best_model_state_sha256='different')),
                 ('preservation.json', lambda doc: doc.update(graph_after={'graph.npz': 'changed'})),
                 ('summary.json', lambda doc: doc.update(test_read=True)),
                 ('summary.json', lambda doc: doc.update(scope='conditional_offline_checkpoint_sensitivity'))]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for i, (filename, change) in enumerate(edits):
                with self.subTest(case=i):
                    source = root/f'source_{i}'
                    source_fixture(source)
                    doc = read_json(source/filename)
                    change(doc)
                    write_json(source/filename, doc)
                    before = source_hashes(source)
                    with self.assertRaises(ValueError):
                        analysis.analyze(source, root/f'output_{i}', allow_engineering=True)
                    self.assertEqual(before, source_hashes(source))
                    self.assertFalse((root/f'output_{i}'/'analysis.json').exists())
            source = root/'has_failure'
            source_fixture(source)
            write_json(source/'failure.json', {'status': 'B3R_PATH_PROBE_FAILED'})
            with self.assertRaises(ValueError):
                analysis.analyze(source, root/'failure_output', allow_engineering=True)

    def test_wrong_axes_and_nonfinite_valid_target_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for i, field in enumerate(('off', 'valid', 'report_associated', 'target', 'final_on_exactly_restored')):
                with self.subTest(field=field):
                    source = root/f'source_{i}'
                    arrays = source_fixture(source)
                    altered = {name: value.copy() for name, value in arrays.items()}
                    if field == 'off':
                        altered[field] = altered[field][:, :, :, 0]
                    elif field == 'valid':
                        altered[field] = altered[field].astype(np.int8)
                    elif field == 'report_associated':
                        altered[field] = altered[field][:, :1]
                    elif field == 'target':
                        altered[field][0, 0, 0, 0] = np.nan
                    else:
                        altered[field] = np.array(False)
                    np.savez_compressed(source/'paired_predictions.npz', **altered)
                    with self.assertRaises(ValueError):
                        analysis.analyze(source, root/f'output_{i}', allow_engineering=True)

    def test_cli_runs_without_importing_torch_and_preserves_every_source_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_fixture(root/'source')
            before = source_hashes(root/'source')
            script = REPO/'experiments'/'chronological'/'analyze_b3_r_path_probe.py'
            code = (
                "import builtins, runpy, sys\n"
                "original = builtins.__import__\n"
                "def guarded(name, *args, **kwargs):\n"
                "    if name == 'torch' or name.startswith('torch.'):\n"
                "        raise AssertionError('Offline analysis attempted to import torch')\n"
                "    return original(name, *args, **kwargs)\n"
                "builtins.__import__ = guarded\n"
                "script = sys.argv[1]\n"
                "sys.argv = [script] + sys.argv[2:]\n"
                "runpy.run_path(script, run_name='__main__')\n"
            )
            completed = subprocess.run([sys.executable, '-X', 'utf8', '-c', code, str(script),
                                        '--probe-dir', str(root/'source'), '--output-dir', str(root/'output'),
                                        '--allow-engineering'], cwd=REPO, capture_output=True, text=True,
                                       encoding='utf-8', timeout=60)
            self.assertEqual(completed.returncode, 0, completed.stdout+'\n'+completed.stderr)
            self.assertEqual(read_json(root/'output'/'analysis.json')['status'], 'B3R_GAIN_ANALYSIS_PASS')
            self.assertEqual(before, source_hashes(root/'source'))


if __name__ == '__main__':
    unittest.main()
