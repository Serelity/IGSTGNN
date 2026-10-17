"""Shared-mask metrics, archive preservation, and raw mechanism aggregation."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.chronological import evaluate_unified_adapter_v03 as evaluation


def archive(rows=10, offset=0):
    y = np.broadcast_to(np.arange(rows)[:, None, None]+10., (rows, 12, 2)).copy()
    original = y+np.arange(rows)[:, None, None]
    return dict(schema=np.array(evaluation.ARCHIVE_SCHEMA), axis_order=np.array(evaluation.AXIS_ORDER),
                y=y, valid=np.ones(y.shape, bool), sample_ids=np.arange(offset, rows+offset),
                node_ids=np.array([11, 12]), direct_support=np.tile([True, False], (rows, 1)),
                road_labels=np.array(['R1', 'R2']), group__candidate_potential=np.ones((rows, 2), bool),
                pred__ORIGINAL=original, pred__TRAFFIC_CORE=original+.3,
                pred__H_ONLY=original+.2, pred__R_ONLY=original+.1, pred__BOTH=original-.2,
                pred__DIRECT=original+.4, pred__BOTH_REPORT_OFF=original+.1)


class ArchiveEvaluationTests(unittest.TestCase):
    def test_strict_shape_and_mask_contract(self):
        for mutate in (lambda a: a.update(axis_order=np.array('sample,node,horizon')),
                       lambda a: a.update(valid=a['valid'].astype(float)),
                       lambda a: a.update(sample_ids=np.zeros(len(a['y']), int)),
                       lambda a: a.update(pred__BOTH=a['pred__BOTH'][..., None]),
                       lambda a: a.update(schema=np.array('old')),
                       lambda a: a.pop('pred__ORIGINAL')):
            data = archive()
            mutate(data)
            with self.assertRaises(ValueError):
                evaluation.validate_archive(data)

    def test_invalid_prediction_cannot_silently_change_shared_mask(self):
        data = archive()
        data['pred__BOTH'][0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, 'finite'):
            evaluation.validate_archive(data)
        data['valid'][0, 0, 0] = False
        checked = evaluation.validate_archive(data)
        self.assertEqual(checked['valid'].sum(), data['valid'].size-1)

    def test_positive_target_contract_and_empty_horizon(self):
        data = archive()
        data['y'][0, 0, 0] = 0
        with self.assertRaisesRegex(ValueError, 'positive'):
            evaluation.validate_archive(data)
        data['valid'][0, 0, 0] = False
        data['valid'][:, 2] = False
        checked = evaluation.validate_archive(data)
        result = evaluation.point_metrics(checked['predictions']['BOTH'], checked['y'], checked['valid'])
        self.assertIsNone(result['mae_macro'])
        self.assertGreater(result['mae_pooled'], 0)

    def test_pooled_and_macro_do_not_confuse_unequal_counts(self):
        y = np.ones((2, 12, 1))
        pred = y.copy()
        pred[:, 0] += 10
        valid = np.ones(y.shape, bool)
        valid[1, 0] = False
        values = evaluation.point_metrics(pred, y, valid)
        self.assertAlmostEqual(values['mae_pooled'], 10/23)
        self.assertAlmostEqual(values['mae_macro'], 10/12)
        self.assertAlmostEqual(values['rmse_pooled'], np.sqrt(100/23))
        self.assertAlmostEqual(values['rmse_macro'], 10/12)

    def test_hard_set_selected_once_by_original_and_id_ties(self):
        data = archive()
        checked = evaluation.validate_archive(data)
        hard = evaluation.fixed_hard_indices(checked['predictions']['ORIGINAL'], checked['y'], checked['valid'], checked['sample_ids'])
        self.assertEqual(hard.tolist(), [9])
        data['pred__ORIGINAL'] = data['y']+1
        data['sample_ids'] = np.arange(10)[::-1]
        hard = evaluation.fixed_hard_indices(data['pred__ORIGINAL'], data['y'], data['valid'], data['sample_ids'])
        self.assertEqual(hard.tolist(), [9])
        data['valid'][9, 4] = False
        hard = evaluation.fixed_hard_indices(data['pred__ORIGINAL'], data['y'], data['valid'], data['sample_ids'])
        self.assertEqual(hard.tolist(), [8])

    def test_direction_contributions_and_positive_negative_net(self):
        target = np.full((1, 12, 1), 10.)
        control = np.array([5, 10, 8, 8, 8, 8, 8, 8, 12, 12, 12, 12.]).reshape(target.shape)
        candidate = np.array([5, 11, 9, 10, 11, 12, 13, 7, 11, 10, 9, 7.]).reshape(target.shape)
        values = evaluation.paired_metrics(control, candidate, target, np.ones(target.shape, bool))
        self.assertAlmostEqual(values['positive_gain']-values['negative_gain'], values['mae_gain_pooled'])
        self.assertEqual(sum(value['valid_count'] for value in values['directions'].values()), 12)
        self.assertAlmostEqual(sum(value['global_mae_contribution'] for value in values['directions'].values()), values['mae_gain_pooled'])
        self.assertEqual(values['directions']['overshoot_equal_error']['valid_count'], 1)
        self.assertEqual(values['directions']['exact_control_changed']['valid_count'], 1)

    def test_quantiles_pinball_and_interval_score(self):
        target = np.array([1., 5., 9.]).reshape(1, 3, 1)
        q = np.broadcast_to(np.array([2., 5., 8.]), (*target.shape, 3))
        result = evaluation.quantile_metrics(q, target, np.ones(target.shape, bool))
        self.assertAlmostEqual(result['coverage80'], 1/3)
        self.assertEqual(result['width80'], 6.)
        self.assertAlmostEqual(result['interval_score80'], (16+6+16)/3)
        self.assertEqual(len(result['pinball_by_quantile']), 3)
        self.assertNotIn('crps', result)

    def test_quantile_crossing_or_median_mismatch_refused(self):
        data = archive()
        data['quantile_levels'] = np.array([.1, .5, .9])
        data['q__BOTH'] = np.stack([data['pred__BOTH']-1, data['pred__BOTH'], data['pred__BOTH']+1], -1)
        evaluation.validate_archive(data)
        data['q__BOTH'][0, 0, 0, 0] += 3
        with self.assertRaisesRegex(ValueError, 'crossing'):
            evaluation.validate_archive(data)
        data['q__BOTH'][0, 0, 0] = data['pred__BOTH'][0, 0, 0]+np.array([-.9, .1, 1.1])
        with self.assertRaisesRegex(ValueError, 'median'):
            evaluation.validate_archive(data)

    def test_mechanisms_aggregate_all_batches_with_count_weighting(self):
        result = evaluation.aggregate_mechanisms([dict(delta_sum=10, delta_count=2, balance_max=.1, scale_min=.7),
                                                   dict(delta_sum=3, delta_count=1, balance_max=.03, scale_min=.9)])
        self.assertEqual(result['raw']['delta_count'], 3)
        self.assertAlmostEqual(result['means']['delta_mean'], 13/3)
        self.assertEqual(result['raw']['balance_max'], .1)
        self.assertEqual(result['raw']['scale_min'], .7)
        for values in ([dict(batch_mean=2)], [dict(counter=np.nan)]):
            with self.assertRaises(ValueError):
                evaluation.aggregate_mechanisms(values)

    def test_cli_style_archive_preserved_and_all_tables_written(self):
        data = archive()
        before = deepcopy(data)
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            source, output = Path(tmp)/'source.npz', Path(tmp)/'analysis'
            np.savez_compressed(source, **data)
            snapshot = source.read_bytes()
            result = evaluation.analyze_archive(source, output, dict(split='validation', engineering_only=True, test_evaluated=False))
            self.assertEqual(source.read_bytes(), snapshot)
            self.assertEqual(result['hard_set']['sample_ids'], [9])
            self.assertEqual(result['new_inference_calls'], 0)
            self.assertFalse(result['test_evaluated'])
            self.assertIn('report_dependency', result['comparisons'])
            for file in ('report.md', 'cases.csv', 'comparisons.csv', 'groups.csv', 'horizons.csv', 'directions.csv', 'analysis.json'):
                self.assertTrue((output/file).is_file())
            self.assertEqual(json.loads((output/'analysis.json').read_text(encoding='utf-8'))['status'], result['status'])
            with self.assertRaisesRegex(ValueError, 'preserved'):
                evaluation.analyze_archive(source, output)
        for key, value in before.items():
            np.testing.assert_array_equal(value, data[key])

    def test_test_split_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            with self.assertRaisesRegex(ValueError, 'test split'):
                evaluation.analyze_archive(archive(), Path(tmp)/'out', dict(split='test'))


if __name__ == '__main__':
    unittest.main()
