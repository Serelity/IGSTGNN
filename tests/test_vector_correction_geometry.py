"""Independent arithmetic examples and support/paired-statistics boundaries."""

import copy
from datetime import datetime, timedelta
import json
from pathlib import Path
import unittest

import numpy as np

from experiments.chronological import vector_correction_geometry as g


def protocol():
    return json.loads((Path(__file__).resolve().parents[1] / 'experiments/chronological/vector_correction_geometry_v12n.json').read_text())


def packed(n=4):
    dates = [datetime(2023, 7, 4) + timedelta(weeks=i) for i in range(n)]
    return {'ids': np.arange(11, 11+n, dtype=np.int64), 'source_indices': np.arange(n, dtype=np.int64),
        'incident_ids': np.asarray(['same-incident'] * n),
        'positive_t0': np.asarray([d.isoformat() for d in dates]),
        'cohort_t0': np.asarray([d.isoformat() for d in dates]),
        'support_start': np.asarray([(d-timedelta(hours=1)).isoformat() for d in dates]),
        'support_end_exclusive': np.asarray([(d+timedelta(hours=2)).isoformat() for d in dates]),
        'window_offsets': np.arange(n+1, dtype=np.int64) * 6,
        'candidate_mask': np.ones((n, 1), dtype=bool),
        'candidate_node_indices': np.zeros(6*n, dtype=np.int64),
        'horizon_indices': np.tile(np.arange(1, 7, dtype=np.int64), n),
        'A': np.zeros(6*n, dtype=np.float32), 'Y': np.ones(6*n, dtype=np.float32),
        'd': np.full(6*n, .25, dtype=np.float64), 'valid': np.ones(6*n, dtype=bool)}


class GeometryTests(unittest.TestCase):
    def test_six_categories_and_zero_residual_derivative(self):
        r = np.array([0, 0, 1, 1, 1, 1, -1, -1], np.float32)
        d = np.array([0, 1, -1, 1.5, 2, 3, -.5, 1], np.float64)
        result = g.cell_geometry(np.zeros_like(r), r, d, np.ones(len(r), bool))
        np.testing.assert_array_equal(result['category'], [0, 1, 2, 3, 4, 5, 3, 2])
        np.testing.assert_allclose(result['gain'], [0, -1, -1, .5, 0, -1, .5, -1])
        self.assertEqual(result['slope'][1], -1)
        np.testing.assert_allclose(result['gain'], result['slope'] - result['crossing_penalty'])

    def test_positive_slope_negative_gain_without_harmful_overshoot(self):
        result = g.cell_geometry(np.zeros(2, np.float32), np.ones(2, np.float32),
                                 np.array([1.5, -1], np.float64), np.ones(2, bool))
        self.assertEqual(result['gain'].mean(), -.25)
        self.assertEqual(result['slope'].mean(), .25)
        self.assertEqual(result['crossing_penalty'].mean(), .5)
        self.assertNotIn(5, result['category'])

    def test_concavity_bound_for_both_weightings(self):
        rng = np.random.default_rng(23)
        r = rng.integers(-5, 6, 200).astype(np.float32)
        d = rng.normal(size=200).astype(np.float64)
        result = g.cell_geometry(np.zeros_like(r), r, d, np.ones(200, bool))
        for weights in (np.ones(200)/200, np.repeat([1/2/50, 1/2/150], [50, 150])):
            slope = result['slope'] @ weights
            for lam in (0., .01, .4, 1., 5.):
                gain = (np.abs(r.astype(float))-np.abs(r.astype(float)-lam*d)) @ weights
                self.assertLessEqual(gain, lam*slope+1e-12)

    def test_invalid_targets_ignored_without_dropping_predictions(self):
        p = packed(1); p['Y'][0] = np.nan; p['valid'][0] = False
        result = g.window_geometry(p)
        self.assertEqual(p['window_offsets'][-1], 6)
        self.assertEqual(result['valid_counts'][0], 5)
        self.assertEqual(result['gain_sums'][0], 1.25)
        p['valid'][0] = True
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            g.window_geometry(p)

    def test_zero_support_and_no_candidates_are_undefined(self):
        p = packed(); p['valid'][:] = False
        record = g.window_geometry(p)
        spec = protocol(); weeks, weights = g.calendar(spec['periods']['audit'], spec['bootstrap'])
        result, categories = g.analyze_windows(record, weeks, weights, spec)
        self.assertEqual(result['evaluable_windows'], 0)
        for value in result['estimands'].values():
            self.assertIsNone(value['point']['gain'])
            self.assertEqual(value['intervals']['four_week_block']['slope']['status'], 'INSUFFICIENT_VALID_DRAWS')
        self.assertTrue(all(row['gain_contribution'] is None for row in categories))
        p = packed(1); p['candidate_mask'][:] = False; p['window_offsets'][:] = 0
        for key in ('A','Y','d','valid','candidate_node_indices','horizon_indices'):
            p[key] = p[key][:0]
        self.assertEqual(g.window_geometry(p)['valid_counts'][0], 0)

    def test_weighting_reversal_and_category_contribution_closure(self):
        p = packed(2); p['d'][:6] = -1; p['d'][6:] = 2; p['valid'][7:] = False
        record = g.window_geometry(p)
        spec = protocol(); weeks, weights = g.calendar(spec['periods']['audit'], spec['bootstrap'])
        result, rows = g.analyze_windows(record, weeks, weights, spec)
        self.assertLess(result['estimands']['pooled_valid_cells']['point']['slope'], 0)
        self.assertGreater(result['estimands']['equal_forecast_window']['point']['slope'], 0)
        for estimand in g.ESTIMANDS:
            for key in g.QUANTITIES:
                self.assertAlmostEqual(sum(row[key+'_contribution'] for row in rows if row['estimand']==estimand),
                                       result['estimands'][estimand]['point'][key])
        self.assertEqual(result['unique_incidents'], 1)

    def test_packing_omission_duplicate_and_order_are_rejected(self):
        for key in ('window_offsets','horizon_indices','candidate_node_indices','ids'):
            p = packed()
            p[key][1] = p[key][0]
            if key == 'candidate_node_indices': p[key][1] = 2
            with self.subTest(key=key), self.assertRaises(ValueError):
                g.window_geometry(p)

    def test_near_zero_report_does_not_round_cell_values(self):
        p = packed(1); p['d'][:] = 1e-9
        result = g.window_geometry(p)
        self.assertGreater(result['slope_sums'][0], 0)
        self.assertEqual(result['category_counts'][0,3], 6)
        self.assertEqual(g.slope_status(1e-9, 1e-9, 1e-6), 'NEAR_ZERO_UNRESOLVED')

    def test_calendar_keeps_empty_weeks_and_shared_original_draws(self):
        spec = protocol(); weeks, weights = g.calendar(spec['periods']['audit'], spec['bootstrap'])
        self.assertEqual(weeks, [f'2023-W{i:02d}' for i in range(27,36)])
        for value in weights.values():
            np.testing.assert_array_equal(value.sum(1), np.full(2000,9))
        np.testing.assert_array_equal(weights['four_week_block'], g.regional.bootstrap_weights(9,2000,12029,4))


class MatchedTests(unittest.TestCase):
    def make_records(self):
        records = {}
        for j, cohort in enumerate(g.COHORTS):
            p = packed(); p['d'][:] = [.1,.3,-.2][j]
            p['source_indices'] += j*100
            records[cohort] = g.window_geometry(p)
        return records

    def test_direct_paired_difference_and_reordering(self):
        records = self.make_records(); spec = protocol()
        for key in (*g.META,'valid_counts','gain_sums'):
            records['secondary_control'][key] = records['secondary_control'][key][::-1]
        weeks, weights = g.calendar(spec['periods']['audit'], spec['bootstrap'])
        result, _ = g.analyze_matched(records,weeks,weights,spec)
        self.assertAlmostEqual(result['points']['incident_minus_mean_controls']['equal_triad'], .05)
        ci = result['intervals']['incident_minus_primary']['four_week_block']['pooled_valid_cells']
        self.assertAlmostEqual(ci['ci_low'], -.2)
        self.assertAlmostEqual(ci['ci_high'], -.2)

    def test_complete_triplets_have_common_denominator(self):
        records = self.make_records(); spec = protocol()
        records['primary_control']['valid_counts'][0] = 0
        records['primary_control']['gain_sums'][0] = 0
        records['incident']['gain_sums'][0] = 60
        weeks, weights = g.calendar(spec['periods']['audit'], spec['bootstrap'])
        result, _ = g.analyze_matched(records,weeks,weights,spec)
        self.assertEqual(result['complete_triplets'], 3)
        self.assertAlmostEqual(result['points']['incident_minus_mean_controls']['equal_triad'], .05)
        self.assertGreater(result['points']['incident_minus_mean_controls']['pooled_valid_cells'], 2)
        self.assertEqual(result['coverage']['incident']['retained_valid_cells'],18)

    def test_inactive_control_does_not_poison_single_control_contrast(self):
        records = self.make_records(); spec = protocol()
        records['secondary_control']['valid_counts'][:] = 0; records['secondary_control']['gain_sums'][:] = 0
        weeks, weights = g.calendar(spec['periods']['audit'], spec['bootstrap'])
        result, _ = g.analyze_matched(records,weeks,weights,spec)
        self.assertAlmostEqual(result['points']['incident_minus_primary']['pooled_valid_cells'], -.2)
        self.assertIsNone(result['points']['incident_minus_mean_controls']['pooled_valid_cells'])
        self.assertIsNone(result['points']['incident_minus_primary']['equal_triad'])

    def test_delete_week_when_any_member_touches_half_open_support(self):
        records = self.make_records(); spec = protocol()
        first = records['primary_control']
        first['support_start'][0] = '2023-07-09T23:00:00'
        first['support_end_exclusive'][0] = '2023-07-10T00:00:00'
        weeks, weights = g.calendar(spec['periods']['audit'], spec['bootstrap'])
        _, rows = g.analyze_matched(records,weeks,weights,spec)
        row = next(x for x in rows if x['deleted_week']=='2023-W28')
        self.assertEqual(row['removed_triplets'], 1) # only original week-28 triplet
        first['support_end_exclusive'][0] = '2023-07-10T00:00:01'
        _, rows = g.analyze_matched(records,weeks,weights,spec)
        row = next(x for x in rows if x['deleted_week']=='2023-W28')
        self.assertEqual(row['removed_triplets'], 2)

    def test_mismatched_ids_and_incident_clock_fail(self):
        records = self.make_records(); records['primary_control']['ids'][0] = 99
        with self.assertRaisesRegex(ValueError,'IDs'):
            g.matched_arrays(records)
        records = self.make_records(); records['primary_control']['positive_t0'][0] = '2023-08-01T00:00:00'
        with self.assertRaisesRegex(ValueError,'clock'):
            g.matched_arrays(records)


if __name__ == '__main__':
    unittest.main()
