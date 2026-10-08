"""Numerical and saved-artifact checks for the CPU-only P2 diagnostic."""

from contextlib import redirect_stdout
import csv
from datetime import datetime, timedelta
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.chronological import continue_incident_routing_p2 as continuation
from experiments.chronological import diagnose_incident_routing_p2 as diagnostic


class NumericalTests(unittest.TestCase):
    def test_macro_keeps_equal_horizon_weight_and_counts_real_zeros(self):
        sums = np.ones((2, 12)) * 2
        counts = np.ones((2, 12), dtype=np.int64)
        sums[:, 0], counts[:, 0] = [8, 0], [4, 0]
        result = diagnostic.aggregate(sums, counts)
        self.assertEqual(result['mae_macro'], 2.)
        sums[:, 0] = [16, 0]
        result = diagnostic.aggregate(sums, counts)
        self.assertAlmostEqual(result['mae_macro'], (4 + 11 * 2) / 12)
        self.assertAlmostEqual(result['mae_pooled'], 60 / 26)
        self.assertNotEqual(result['mae_macro'], result['mae_pooled'])

    def test_invalid_nan_is_masked_and_empty_region_is_null(self):
        shape = (1, 12, 2, 1)
        arrays = {'prediction': np.ones(shape, dtype=np.float32) * 3,
                  'target': np.zeros(shape, dtype=np.float32),
                  'valid': np.ones(shape, dtype=bool), 'associated': np.ones(shape, dtype=bool)}
        arrays['valid'][:, :, 1] = False
        arrays['target'][:, :, 1] = np.nan
        valid = diagnostic.aggregate(*diagnostic.event_sums(arrays, 'all_nodes'))
        self.assertEqual(valid['mae_macro'], 3.)  # Target zero is a valid observation.
        empty = diagnostic.aggregate(*diagnostic.event_sums(arrays, 'nonassociated_nodes'))
        self.assertIsNone(empty['mae_macro'])
        self.assertEqual(empty['per_horizon_mae'], [None] * 12)

    def test_axis_permutation_cannot_be_silently_paired(self):
        left = {key: np.array([1, 2]) for key in
                ('target', 'valid', 'associated', 'sample_indices', 'station_ids')}
        right = {key: value.copy() for key, value in left.items()}
        right['sample_indices'] = np.array([2, 1])
        with self.assertRaisesRegex(ValueError, 'sample_indices'):
            diagnostic.paired_predictions(left, right)

    def test_prediction_hash_is_checked_before_loading(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'predictions.npz'
            path.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                diagnostic.read_predictions(path, 'original')


class CompletedPairTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / 'pair'
        self.data = Path(self.tmp.name) / 'data'
        self.data.mkdir()
        manifest = self.data / 'val_manifest.csv'
        with manifest.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['sample_index', 'split', 't0'])
            writer.writeheader()
            for i in range(917):
                writer.writerow({'sample_index': i, 'split': 'val',
                                 't0': (datetime(2023, 9, 1) + timedelta(hours=i)).isoformat()})
        package = self.data / 'summary.json'
        package.write_text(json.dumps({'files': {'val_manifest.csv': diagnostic.digest(manifest)}}))
        self.summaries = {}
        shape = (917, 12, 496, 1)
        support = np.zeros(shape, dtype=bool)
        support[:, :, 0] = True
        for variant, epochs, metric in (('fixed', 100, 2.), ('acdg', 89, 3.)):
            directory = self.root / variant
            directory.mkdir(parents=True)
            predictions = directory / 'best_validation_predictions.npz'
            np.savez_compressed(predictions, prediction=np.full(shape, metric, dtype=np.float32),
                                target=np.zeros(shape, dtype=np.float32), valid=np.ones(shape, dtype=bool),
                                associated=support, sample_indices=np.arange(917), station_ids=np.arange(496))
            selected = 99 if variant == 'fixed' else 69
            history = [{'epoch': i, 'learning_rate': .002, 'train_order_sha256': f'order-{i}',
                        'train': {'mae_macro': 1.},
                        'validation': {'all_nodes': {'mae_macro': metric if i == selected else metric + 1}}}
                       for i in range(1, epochs + 1)]
            identity = {'variant': variant, 'seed': 2025, 'check': False, 'device': 'cuda:0',
                        'batch_size': 48, 'max_epochs': 100,
                        'protocol_sha256': diagnostic.digest(continuation.PROTOCOL),
                        'package_sha256': {'summary.json': diagnostic.digest(package)},
                        'source_sha256': {name: diagnostic.digest(continuation.REPO / name)
                                          for name in continuation.SOURCE_NAMES},
                        'common_initialization_sha256': 'common', 'torch_version': 'fixture',
                        'numpy_version': 'fixture'}
            self.summaries[variant] = {
                'variant': variant, 'identity': identity, 'status': continuation.COMPLETE,
                'completed_epoch': epochs, 'global_updates': epochs * 76, 'best_epoch': selected,
                'best_metric': metric, 'wait': epochs - selected, 'history': history,
                'stop_reason': 'max_epochs' if variant == 'fixed' else 'early_stopping',
                'selection_metric': 'all_nodes.mae_macro', 'initial_max_abs_difference_from_A': 0.,
                'best_validation_predictions_sha256': diagnostic.digest(predictions),
                'best_validation_metrics': {region: {'mae_macro': metric, 'per_horizon_mae': [metric] * 12}
                                            for region in ('all_nodes', 'associated_nodes')},
            }
        self.save()

    def save(self):
        for variant, summary in self.summaries.items():
            (self.root / variant / 'summary.json').write_text(json.dumps(summary))

    def test_complete_diagnostic_preserves_inputs_and_decomposes_gain(self):
        original = {path: diagnostic.digest(path) for path in self.root.rglob('*') if path.is_file()}
        output = self.root / 'diagnostics/report'
        with redirect_stdout(io.StringIO()):
            report = diagnostic.diagnose(self.root, self.data, output)
        self.assertEqual(report['status'], 'P2_SAVED_PREDICTION_DIAGNOSTICS_COMPLETE')
        self.assertEqual(report['regions']['all_nodes']['gain_fixed_minus_acdg'], -1.)
        self.assertAlmostEqual(report['regional_contribution_to_all_node_mae_gain']['associated_nodes'], -1 / 496)
        self.assertEqual(report['window_gain_distribution']['all_nodes']['worsened_windows'], 917)
        self.assertEqual(report['learning_curves']['common_completed_epoch'], 89)
        self.assertEqual(report['learning_curves']['common_budget_best_so_far'][0]['through_epoch'], 69)
        self.assertEqual(len(report['output_sha256']), 4)
        self.assertTrue(report['weekly_comparisons'])
        self.assertEqual(original, {path: diagnostic.digest(path) for path in original})
        with self.assertRaisesRegex(ValueError, 'new diagnostics'):
            diagnostic.diagnose(self.root, self.data, output)

    def test_exported_predictions_cannot_disagree_with_selected_metric(self):
        self.summaries['fixed']['best_validation_metrics']['all_nodes']['per_horizon_mae'][0] = 7.
        self.save()
        with self.assertRaisesRegex(ValueError, 'Saved MAE mismatch'):
            diagnostic.diagnose(self.root, self.data, self.root / 'bad_report')
        self.assertFalse((self.root / 'bad_report').exists())

    def test_manifest_tampering_is_rejected(self):
        with (self.data / 'val_manifest.csv').open('a') as stream:
            stream.write('999,val,2023-09-01T00:00:00\n')
        with self.assertRaisesRegex(ValueError, 'manifest changed'):
            diagnostic.diagnose(self.root, self.data, self.root / 'bad_report')

    def test_history_must_identify_the_selected_epoch(self):
        self.summaries['acdg']['best_epoch'] = 70
        with self.assertRaisesRegex(ValueError, 'Best epoch'):
            diagnostic.curve_diagnostics(self.summaries)


if __name__ == '__main__':
    unittest.main()
