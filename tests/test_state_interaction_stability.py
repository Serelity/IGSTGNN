"""Saved-week v12h stability tests use fabricated NumPy statistics only."""

import copy
import unittest

import numpy as np

from experiments.chronological.audit_architecture_regions import analyze, interval, ratio
from experiments.chronological.state_interaction_stability import (
    analyze_weekly, COMPARISONS, COHORTS, GROUPS, REGIONS, _validate,
)


SPEC = {'draws': 2000, 'seed': 12028, 'confidence': 0.95,
        'sensitivity_circular_block_weeks': 4, 'minimum_valid_draw_fraction': 0.95}
PATHS = ['A', 'strength', 'state_vector', 'interaction_vector']
PAIRS = dict(zip(COMPARISONS, (['A', 'strength'], ['A', 'state_vector'],
    ['A', 'interaction_vector'], ['strength', 'state_vector'],
    ['strength', 'interaction_vector'], ['state_vector', 'interaction_vector'])))


def fixture(sparse=False, empty_common=False, covariance=False, empty_full_week=False):
    # First common window is larger; cell pooling and equal-window weighting differ.
    records, times = {}, {}
    dates = ['2023-05-01', '2023-05-08', '2023-05-15', '2023-05-22', '2023-05-29']
    common, complement = [], []
    for week, date in enumerate(dates):
        if empty_full_week and week == 2:
            continue
        for group, container in ((0, common), (1, complement)):
            if group == 0 and (empty_common or (sparse and week != 0)):
                continue
            if group == 1 and sparse and week == 0:
                continue
            sample = 100 + week * 10 + group
            cells = np.array([100 if week == 0 and (group == 0 or covariance) else 2, 3, 5], dtype=np.int64)
            # Common's first week dominates pooled candidate gain and LOO influence.
            early = (0.3 if week == 0 else -1.) if group == 0 else -0.2
            if covariance and group == 1:
                early = (0.3 if week == 0 else -1.) - 0.2
            state_gain = cells * np.array([early, 0.2 + week * 0.1, 0.1])
            arm_gains = np.stack([np.zeros(3), state_gain * 0.1, state_gain, state_gain * 0.5])
            region_errors = cells[None, :] * 100 - arm_gains
            errors = np.concatenate([region_errors.sum(1, keepdims=True), region_errors], axis=1)
            container.append((sample, errors, np.r_[cells.sum(), cells]))
            times[sample] = date + 'T00:00:00'

    def make(rows):
        n = len(rows)
        return {'ids': np.asarray([row[0] for row in rows], dtype=np.int64),
                'errors': np.asarray([row[1] for row in rows], dtype=np.float64).reshape(n, 4, 4),
                'counts': np.asarray([row[2] for row in rows], dtype=np.int64).reshape(n, 4),
                'regions': list(REGIONS)}

    records['incident_full'] = make(common + complement)
    records['incident'] = make(common)
    for cohort in ('primary_control', 'secondary_control'):
        record = make(common)
        # Controls need not share valid target cells with positive windows.
        record['counts'] = record['counts'] * 2
        record['errors'] = record['errors'] * 2
        records[cohort] = record
    records[GROUPS[0]], records[GROUPS[1]] = make(common), make(complement)
    protocol = {'bootstrap': SPEC, 'comparisons': PAIRS, 'paths': PATHS}
    analysis, _, arrays = analyze(records, times, protocol)
    return arrays, analysis


class StabilityTests(unittest.TestCase):
    def test_reconciles_original_intervals_and_distinguishes_weighting(self):
        arrays, source = fixture()
        result, contrasts, weekly, contributions = analyze_weekly(arrays, source, PAIRS, SPEC)
        effects = result['cohort_stability'][GROUPS[0]]['regions']['candidate_h1_h6']['comparisons']['state_vector_vs_A']
        self.assertGreater(effects['pooled']['point_gain_raw_mae'], 0)
        self.assertLess(effects['equal_forecast_window']['point_gain_raw_mae'], 0)
        self.assertEqual(effects['pooled']['intervals']['week'],
                         source['results'][GROUPS[0]]['regions']['candidate_h1_h6']['comparisons']['state_vector_vs_A']['intervals']['week']['pooled'])
        self.assertTrue(result['source_statistics_reconciled'])
        self.assertEqual(result['equal_weight_unit'], 'forecast_window_not_unique_accident')
        self.assertEqual(len(contrasts), 4 * 6 * 2)
        self.assertEqual(len(weekly), 5 * 6 * 4 * 6)
        self.assertEqual(len(contributions), (5 + 1) * 3 * 4 * 6)

    def test_paired_contrast_resampling_uses_covariance_not_endpoint_subtraction(self):
        arrays, source = fixture(covariance=True)
        result, rows, _, _ = analyze_weekly(arrays, source, PAIRS, SPEC)
        weights = arrays['bootstrap_week_weights']
        c, r = COMPARISONS.index('state_vector_vs_A'), REGIONS.index('candidate_h1_h6')
        draws = []
        for group in GROUPS:
            draws.append(ratio(weights @ arrays[f'{group}_gain_sums'][:, c, r],
                               weights @ arrays[f'{group}_valid_counts'][:, r]))
        expected = interval(draws[0] - draws[1], .95, .95)
        observed = result['common_minus_complement']['candidate_h1_h6']['state_vector_vs_A']['pooled']['intervals']['week']
        self.assertEqual(observed, expected)
        # Shared weekly variation cancels in the paired contrast, not endpoint differences.
        common_ci = source['results'][GROUPS[0]]['regions']['candidate_h1_h6']['comparisons']['state_vector_vs_A']['intervals']['week']['pooled']
        complement_ci = source['results'][GROUPS[1]]['regions']['candidate_h1_h6']['comparisons']['state_vector_vs_A']['intervals']['week']['pooled']
        self.assertAlmostEqual(observed['ci_low'], .2, places=12)
        self.assertAlmostEqual(observed['ci_high'], .2, places=12)
        self.assertLess(common_ci['ci_low'] - complement_ci['ci_high'], observed['ci_low'] - .1)
        record = next(row for row in rows if row['region'] == 'candidate_h1_h6' and
                      row['comparison'] == 'state_vector_vs_A' and row['estimand'] == 'pooled')
        self.assertGreater(record['common_minus_complement_gain_raw_mae'], 0)

    def test_exact_global_contributions_and_weekly_denominators(self):
        arrays, source = fixture()
        result, _, _, rows = analyze_weekly(arrays, source, PAIRS, SPEC)
        for comparison in COMPARISONS:
            totals = result['full_positive_contributions'][comparison]['contribution_to_full_global_gain']
            for region in REGIONS:
                self.assertAlmostEqual(totals['incident_full'][region], sum(totals[group][region] for group in GROUPS))
            for cohort in ('incident_full', *GROUPS):
                self.assertAlmostEqual(totals[cohort]['all'], sum(totals[cohort][region] for region in REGIONS[1:]))
        c = COMPARISONS.index('state_vector_vs_A')
        for row in rows:
            if row['comparison'] != COMPARISONS[c] or row['region'] != 'candidate_h1_h6':
                continue
            gain = arrays[row['cohort'] + '_gain_sums'][:, c, 1]
            full = arrays['incident_full_valid_counts'][:, 0]
            if row['period'] == 'overall':
                expected = gain.sum() / full.sum()
            else:
                w = source['weeks'].index(row['positive_week'])
                expected = gain[w] / full[w]
            self.assertAlmostEqual(row['contribution_to_full_global_gain'], expected)

    def test_leave_one_week_out_exposes_sign_flip(self):
        arrays, source = fixture()
        result, _, _, _ = analyze_weekly(arrays, source, PAIRS, SPEC)
        effects = result['cohort_stability'][GROUPS[0]]['regions']['candidate_h1_h6']['comparisons']['state_vector_vs_A']['pooled']
        self.assertEqual(effects['weekly_sign_counts'], {'positive': 1, 'negative': 4, 'zero': 0, 'undefined': 0})
        omitted = effects['leave_one_week_out']
        self.assertEqual(omitted['sign_flip_from_full_point_count'], 1)
        self.assertLess(omitted['weeks'][0]['gain_raw_mae'], 0)
        self.assertGreater(omitted['minimum_gain_raw_mae'] * -1, 0)
        self.assertEqual(len(omitted['weeks']), 5)

    def test_sparse_groups_keep_empty_weeks_and_insufficient_paired_draw_status(self):
        arrays, source = fixture(sparse=True)
        result, _, rows, _ = analyze_weekly(arrays, source, PAIRS, SPEC)
        effect = result['common_minus_complement']['candidate_h1_h6']['state_vector_vs_A']['pooled']
        self.assertEqual(effect['intervals']['week']['status'], 'INSUFFICIENT_VALID_DRAWS')
        self.assertIsNone(effect['intervals']['week']['ci_low'])
        self.assertEqual(effect['weekly_sign_counts']['undefined'], 5)
        empty = [row for row in rows if row['cohort'] == GROUPS[0] and row['positive_week'] != source['weeks'][0]]
        self.assertTrue(all(row['pooled_gain_raw_mae'] is None and row['status'] == 'UNDEFINED_EMPTY_SUPPORT' for row in empty))

    def test_empty_group_is_explicit_and_partition_still_holds(self):
        arrays, source = fixture(empty_common=True)
        result, _, _, contributions = analyze_weekly(arrays, source, PAIRS, SPEC)
        effect = result['common_minus_complement']['all']['state_vector_vs_A']['pooled']
        self.assertIsNone(effect['point_gain_raw_mae'])
        self.assertEqual(effect['intervals']['week']['valid_draws'], 0)
        self.assertEqual(effect['leave_one_week_out']['sign_counts']['undefined'], 5)
        common = [row for row in contributions if row['cohort'] == GROUPS[0]]
        self.assertTrue(all(row['contribution_to_full_global_gain'] == 0 for row in common))

    def test_empty_full_week_has_undefined_weekly_global_contributions(self):
        arrays, source = fixture(empty_full_week=True)
        result, _, weekly, contributions = analyze_weekly(arrays, source, PAIRS, SPEC)
        self.assertEqual(len(result['weeks']), 5)
        gap_week = source['weeks'][2]
        empty_rows = [row for row in weekly if row['positive_week'] == gap_week]
        self.assertTrue(all(row['valid_cells'] == 0 and row['pooled_gain_raw_mae'] is None for row in empty_rows))
        empty_contributions = [row for row in contributions if row['positive_week'] == gap_week]
        self.assertTrue(all(row['contribution_to_full_global_gain'] is None and
                            row['status'] == 'UNDEFINED_EMPTY_FULL_POSITIVE_WEEK' for row in empty_contributions))

    def test_rejects_corrupt_weights_axes_counts_partitions_and_summary(self):
        arrays, source = fixture()
        cases = []
        bad = copy.deepcopy(arrays); bad['bootstrap_week_weights'][0, 0] += 1; cases.append((bad, source))
        bad = copy.deepcopy(arrays); bad['comparisons'] = bad['comparisons'][::-1]; cases.append((bad, source))
        bad = copy.deepcopy(arrays); bad[GROUPS[0] + '_valid_counts'][0, 1] += .1; cases.append((bad, source))
        bad = copy.deepcopy(arrays); bad[GROUPS[0] + '_gain_sums'][0, 1, 1] += .1; cases.append((bad, source))
        bad = copy.deepcopy(arrays); bad['incident_full_event_gain_sums'][0, 1, 1] += .1; cases.append((bad, source))
        bad = copy.deepcopy(arrays); bad['incident_full_gain_sums'][0, 1, 1] = np.nan; cases.append((bad, source))
        bad_summary = copy.deepcopy(source)
        bad_summary['results']['incident_full']['regions']['all']['comparisons']['state_vector_vs_A']['intervals']['week']['pooled']['ci_low'] += .1
        cases.append((arrays, bad_summary))
        for number, (bad_arrays, bad_summary) in enumerate(cases):
            with self.subTest(number=number), self.assertRaises(ValueError):
                analyze_weekly(bad_arrays, bad_summary, PAIRS, SPEC)

    def test_rejects_noninteger_support_and_changed_bootstrap(self):
        arrays, source = fixture()
        bad = copy.deepcopy(arrays)
        bad['incident_full_evaluable_events'][0, 1] = .5
        with self.assertRaisesRegex(ValueError, 'integers'):
            analyze_weekly(bad, source, PAIRS, SPEC)
        bad_spec = {**SPEC, 'draws': 1000}
        with self.assertRaisesRegex(ValueError, 'settings'):
            analyze_weekly(arrays, source, PAIRS, bad_spec)

    def test_calendar_grid_cannot_drop_empty_weeks(self):
        arrays, source = fixture()
        bad = copy.deepcopy(arrays)
        bad['weeks'][2] = '2023-W23'
        changed = copy.deepcopy(source)
        changed['weeks'] = bad['weeks'].tolist()
        with self.assertRaisesRegex(ValueError, 'consecutive'):
            analyze_weekly(bad, changed, PAIRS, SPEC)

    def test_evaluable_forecast_window_totals_cannot_exceed_source_budget(self):
        arrays, source = fixture()
        bad = copy.deepcopy(arrays)
        bad[GROUPS[0] + '_evaluable_events'][0] += 1
        with self.assertRaisesRegex(ValueError, 'exceed.*sample budget'):
            analyze_weekly(bad, source, PAIRS, SPEC)
        smaller_budget = copy.deepcopy(source)
        smaller_budget['results'][GROUPS[0]]['samples'] -= 1
        with self.assertRaisesRegex(ValueError, 'exceed.*sample budget'):
            analyze_weekly(arrays, smaller_budget, PAIRS, SPEC)

    def test_source_sample_budget_rejects_bool_float_negative_and_missing(self):
        arrays, source = fixture()
        for value in (True, 5., -1, None):
            bad = copy.deepcopy(source)
            bad['results'][GROUPS[0]]['samples'] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'sample budget.*integer'):
                analyze_weekly(arrays, bad, PAIRS, SPEC)

    def test_raw_gain_partition_tolerance_matches_saved_producer(self):
        arrays, source = fixture()
        bad = copy.deepcopy(arrays)
        bad['incident_full_gain_sums'][1, 1, 0] += 5e-8
        _validate(bad, source, PAIRS, SPEC)
        bad['incident_full_gain_sums'][1, 1, 0] += 1e-5
        with self.assertRaisesRegex(ValueError, 'algebra|partition'):
            _validate(bad, source, PAIRS, SPEC)

    def test_matched_and_full_positive_common_replay_support_must_match_exactly(self):
        arrays, source = fixture()
        changed_cells = copy.deepcopy(arrays)
        changed_cells['incident_valid_counts'][0, :2] += 1
        with self.assertRaisesRegex(ValueError, 'valid_counts replay support'):
            analyze_weekly(changed_cells, source, PAIRS, SPEC)
        changed_windows = copy.deepcopy(arrays)
        changed_windows['incident_evaluable_events'][0] += 1
        larger_budget = copy.deepcopy(source)
        larger_budget['results']['incident']['samples'] += 1
        with self.assertRaisesRegex(ValueError, 'evaluable_events replay support'):
            analyze_weekly(changed_windows, larger_budget, PAIRS, SPEC)


if __name__ == '__main__':
    unittest.main()
