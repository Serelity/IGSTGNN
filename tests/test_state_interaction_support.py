"""Observable-state diagnostics use fabricated, fixed-path prediction errors."""

import copy
import json
import unittest

import numpy as np

from experiments.chronological import audit_architecture_regions as regional
from experiments.chronological.state_interaction_support import (
    analyze_support, FEATURE_NAMES, REGIONS, STATE_SPEC, SUPPORT_GROUPS,
)


BOOTSTRAP = {'draws': 400, 'seed': 12030, 'confidence': .95,
             'sensitivity_circular_block_weeks': 4, 'minimum_valid_draw_fraction': .95}


def fixture():
    records, features, times = {}, {}, {}
    for phase, n, prefix, start in (('fit', 80, 100, '2023-01-02'),
                                   ('audit', 64, 300, '2023-07-03')):
        ids = np.arange(prefix, prefix + n)
        mean = (np.arange(n) % 2).astype(float) if phase == 'fit' else np.r_[np.zeros(48), np.ones(16)]
        week = np.arange(n) // (n // 8)
        dates = np.datetime64(start) + week * np.timedelta64(7, 'D')
        times[phase] = {int(sample): str(day) + 'T00:00:00' for sample, day in zip(ids, dates)}
        masks = np.tile(np.array([True, True, False, False, False]), (n, 1))
        components = np.stack((6 + 6 * mean.astype(int), np.full(n, 6), np.full(n, 12)), axis=1)
        counts = np.concatenate((components.sum(1, keepdims=True), components), axis=1)
        early_gain = 1. + 2. * mean if phase == 'fit' else -2. + 3. * mean
        gains = np.stack((early_gain, np.full(n, .1), np.full(n, .2)), axis=1)
        all_gains = np.stack((np.zeros_like(gains), np.zeros_like(gains), gains, gains * .5), axis=1)
        error_components = components[:, None, :] * (100. - all_gains)
        errors = np.concatenate((error_components.sum(2, keepdims=True), error_components), axis=2)
        records[phase] = {'ids': ids, 'source_indices': ids.copy(), 'candidate_mask': masks,
                         'counts': counts, 'prediction_counts': np.tile([60, 12, 12, 36], (n, 1)),
                         'errors': errors, 'regions': list(REGIONS)}
        features[phase] = {'history_mean': mean, 'history_trend': np.zeros(n),
                           'history_volatility': np.full(n, .2),
                           'history_missing_fraction': np.zeros(n), 'report_age_minutes': np.full(n, 3.),
                           'candidate_node_count': np.full(n, 2.)}
    return records, features, times


def group(analysis, phase, label, region='candidate_h1_h6', feature=''):
    return next(item for item in analysis['phases'][phase]['groups']
                if item['feature'] == feature and item['group'] == label)['regions'][region]


class SupportTests(unittest.TestCase):
    def analyze(self, records=None, features=None, times=None, bspec=None):
        if records is None:
            records, features, times = fixture()
        return analyze_support(records, features, times, bspec or BOOTSTRAP, STATE_SPEC)

    def test_fit_only_cuts_ranges_and_membership_do_not_use_audit_values_or_errors(self):
        records, features, times = fixture()
        before = self.analyze(records, features, times)
        shifted = copy.deepcopy(features)
        shifted['audit']['history_mean'][:] = 1e6
        records['audit']['errors'] *= 2
        after = self.analyze(records, shifted, times)
        for name in ('marginal_bins', 'joint_bins', 'joint_cells'):
            self.assertEqual(before[0][name], after[0][name])
        self.assertEqual(before[0]['marginal_bins']['history_mean']['cuts'], [.5, 1.])
        self.assertTrue(all(row['support_group'] == 'outside_fit_range'
                            for row in after[1] if row['phase'] == 'audit'))
        self.assertFalse(after[0]['state_definition_uses_errors'])

    def test_ties_constant_features_collapse_and_nonconstant_maximum_is_retained(self):
        records, features, times = fixture()
        result, rows, _, _, _ = self.analyze(records, features, times)
        for name in ('history_trend', 'candidate_node_count'):
            self.assertEqual(result['joint_bins'][name]['cuts'], [])
            self.assertEqual(result['joint_bins'][name]['in_range_bins'], 1)
        self.assertEqual(len(result['joint_cells']), 2)
        self.assertTrue(all(row['support_group'] == 'supported' for row in rows))
        # Interior cut ties go to the right; observed maximum remains in range.
        features['audit']['history_mean'][0] = .5
        result, rows, _, _, _ = self.analyze(records, features, times)
        first = next(row for row in rows if row['phase'] == 'audit')
        self.assertEqual(first['history_mean_bin'], 'in_range_1')
        self.assertEqual(first['joint_cell'], '1:0:0')

    def test_missing_priority_and_extreme_values_are_explicit_not_clipped(self):
        records, features, times = fixture()
        features['audit']['history_mean'][0:4] = [np.nan, -1., 2., 2.]
        features['audit']['history_volatility'][3] = np.nan
        result, rows, gains, _, _ = self.analyze(records, features, times)
        audit = [row for row in rows if row['phase'] == 'audit']
        self.assertEqual([row['support_group'] for row in audit[:4]],
                         ['missing_state', 'outside_fit_range', 'outside_fit_range', 'missing_state'])
        self.assertEqual(audit[1]['history_mean_bin'], 'below_fit_range')
        self.assertEqual(audit[2]['history_mean_bin'], 'above_fit_range')
        self.assertIsNone(audit[0]['history_mean'])
        self.assertTrue(all(row['joint_cell'] is None for row in audit[:4]))
        self.assertEqual(sum(item['forecast_windows'] for item in result['phases']['audit']['groups']
                             if item['grouping'] == 'support'), 64)
        json.dumps(result, allow_nan=False)
        json.dumps(gains, allow_nan=False)

    def test_support_requires_both_fit_window_and_nonempty_week_counts(self):
        records, features, times = fixture()
        # A cell can contain 40 fit windows but all in the first four weeks.
        features['fit']['history_mean'][:] = np.r_[np.zeros(40), np.ones(40)]
        result, rows, _, _, _ = self.analyze(records, features, times)
        self.assertTrue(all(item['supported'] for item in result['joint_cells'].values()))
        features['fit']['history_mean'][:] = np.r_[np.zeros(30), np.ones(50)]
        result, rows, _, _, _ = self.analyze(records, features, times)
        zero = result['joint_cells']['0:0:0']
        self.assertEqual((zero['fit_windows'], zero['fit_nonempty_weeks']), (30, 3))
        self.assertFalse(zero['supported'])
        self.assertTrue(any(row['support_group'] == 'low_joint_support' for row in rows))
        features['fit']['history_mean'][:] = np.r_[np.zeros(29), np.ones(51)]
        # Extend the minority cell across calendar weeks without adding windows.
        minority = np.r_[np.arange(25), [30, 40, 50, 60]]
        features['fit']['history_mean'][:] = 1.
        features['fit']['history_mean'][minority] = 0.
        result, _, _, _, _ = self.analyze(records, features, times)
        self.assertEqual(result['joint_cells']['0:0:0']['fit_windows'], 29)
        self.assertGreaterEqual(result['joint_cells']['0:0:0']['fit_nonempty_weeks'], 4)
        self.assertFalse(result['joint_cells']['0:0:0']['supported'])

    def test_joint_fit_counts_exclude_missing_nonjoint_state(self):
        records, features, times = fixture()
        features['fit']['history_volatility'][0] = np.nan
        result, rows, _, _, _ = self.analyze(records, features, times)
        self.assertEqual(sum(cell['fit_windows'] for cell in result['joint_cells'].values()), 79)
        self.assertEqual(rows[0]['support_group'], 'missing_state')

    def test_absent_finite_fit_reference_is_undefined_and_cannot_create_supported_cell(self):
        records, features, times = fixture()
        features['fit']['history_mean'][:] = np.nan
        result, rows, gains, _, composition = self.analyze(records, features, times)
        self.assertFalse(result['marginal_bins']['history_mean']['finite_reference_available'])
        self.assertEqual(result['joint_cells'], {})
        self.assertTrue(all(row['support_group'] == 'missing_state' for row in rows if row['phase'] == 'fit'))
        self.assertTrue(all(row['support_group'] == 'outside_fit_range' for row in rows if row['phase'] == 'audit'))
        self.assertTrue(all(row['status'] == 'UNDEFINED_NO_SHARED_SUPPORTED_REGION_CELLS' for row in composition))
        self.assertTrue(all(row['fit_common_gain_raw_mae'] is None for row in composition))
        json.dumps(result, allow_nan=False)

    def test_gain_weights_and_partition_contributions_use_cells_and_evaluable_windows(self):
        result, _, _, _, _ = self.analyze()
        expected = {'fit': (7. / 3., 2.), 'audit': (-.8, -1.25)}
        for phase, (pooled, equal) in expected.items():
            supported = group(result, phase, 'supported')
            effect = supported['comparisons']['state_vector_vs_A']
            self.assertAlmostEqual(effect['gain_raw_mae'], pooled)
            self.assertAlmostEqual(effect['equal_forecast_window_gain_raw_mae'], equal)
            self.assertAlmostEqual(effect['gain_raw_mae'], supported['mae']['A'] - supported['mae']['state_vector'])
            self.assertEqual(supported['phase_region_valid_cell_fraction'], 1.)
            for row in result['phases'][phase]['partition_accounting']:
                self.assertTrue(row['reconstruction_passed'])
                if row['region'] == 'candidate_h1_h6' and row['comparison'] == 'state_vector_vs_A':
                    self.assertAlmostEqual(row['sum_group_gain_contributions_raw_mae'], pooled)
                    self.assertAlmostEqual(row['sum_group_equal_forecast_window_gain_contributions_raw_mae'], equal)
        # Audit window mix is 3:1 but valid-cell mix is 3:2, not window fraction.
        first = group(result, 'audit', 'in_range_0', feature='history_mean')
        self.assertAlmostEqual(first['phase_region_valid_cell_fraction'], .6)
        self.assertAlmostEqual(first['phase_region_evaluable_window_fraction'], .75)

    def test_shared_bootstrap_matches_direct_calendar_resampling(self):
        records, features, times = fixture()
        result, _, _, _, _ = self.analyze(records, features, times)
        for phase in ('fit', 'audit'):
            for label, mask in (('supported', np.ones(len(records[phase]['ids']), dtype=bool)),
                                ('in_range_0', features[phase]['history_mean'] == 0.)):
                item = group(result, phase, label, feature='' if label == 'supported' else 'history_mean')
                weeks, positions = regional.week_grid(times[phase])
                wi = np.array([positions[int(sample)] for sample in records[phase]['ids']])
                num = records[phase]['errors'][:, 0, 1] - records[phase]['errors'][:, 2, 1]
                den = records[phase]['counts'][:, 1]
                wg, wc, we, wn = (np.zeros(len(weeks)) for _ in range(4))
                for index in np.flatnonzero(mask):
                    wg[wi[index]] += num[index]
                    wc[wi[index]] += den[index]
                    we[wi[index]] += num[index] / den[index]
                    wn[wi[index]] += 1
                for method, block in (('week', 1), ('four_week_block', 4)):
                    weights = regional.bootstrap_weights(len(weeks), BOOTSTRAP['draws'],
                                                          BOOTSTRAP['seed'] + (method != 'week'), block)
                    for estimand, numerator, denominator in (('pooled', wg, wc), ('equal_forecast_window', we, wn)):
                        expected = regional.interval(regional.ratio(weights @ numerator, weights @ denominator), .95, .95)
                        actual = item['comparisons']['state_vector_vs_A']['intervals'][method][estimand]
                        self.assertEqual(actual, expected)

    def test_empty_group_and_sparse_week_intervals_are_not_zero_gain(self):
        records, features, times = fixture()
        features['audit']['history_mean'][0] = 2.
        result, _, _, weekly, _ = self.analyze(records, features, times)
        empty = group(result, 'fit', 'outside_fit_range')
        sparse = group(result, 'audit', 'outside_fit_range')
        for estimand in ('gain_raw_mae', 'equal_forecast_window_gain_raw_mae'):
            self.assertIsNone(empty['comparisons']['state_vector_vs_A'][estimand])
        self.assertEqual(empty['comparisons']['state_vector_vs_A']['gain_contribution_to_phase_region_raw_mae'], 0.)
        for method in ('week', 'four_week_block'):
            self.assertEqual(empty['comparisons']['state_vector_vs_A']['intervals'][method]['pooled']['valid_draws'], 0)
            self.assertEqual(sparse['comparisons']['state_vector_vs_A']['intervals'][method]['pooled']['status'], 'INSUFFICIENT_VALID_DRAWS')
        empty_week = [row for row in weekly if row['phase'] == 'fit' and row['grouping'] == 'support'
                      and row['group'] == 'outside_fit_range']
        self.assertTrue(all(row['gain_sign'] == 'UNDEFINED' and row['gain_raw_mae'] is None for row in empty_week))

    def test_empty_region_is_undefined_and_excluded_from_composition(self):
        records, features, times = fixture()
        for record in records.values():
            record['counts'][:, 0] -= record['counts'][:, 1]
            record['counts'][:, 1] = 0
            record['errors'][:, :, 1] = 0.
            record['errors'][:, :, 0] = record['errors'][:, :, 1:].sum(2)
        result, _, _, _, rows = self.analyze(records, features, times)
        self.assertTrue(all(row['status'] == 'UNDEFINED_NO_SHARED_SUPPORTED_REGION_CELLS'
                            for row in rows if row['region'] == 'candidate_h1_h6'))
        item = group(result, 'fit', 'supported')
        self.assertIsNone(item['comparisons']['state_vector_vs_A']['gain_raw_mae'])
        self.assertIsNone(item['phase_region_valid_cell_fraction'])
        json.dumps(result, allow_nan=False)

    def test_composition_algebra_uses_correct_mass_and_only_shared_supported_cells(self):
        result, _, _, _, rows = self.analyze()
        selected = {row['estimand']: row for row in rows if row['region'] == 'candidate_h1_h6'
                    and row['comparison'] == 'state_vector_vs_A'}
        pooled, equal = selected['pooled'], selected['equal_forecast_window']
        self.assertAlmostEqual(pooled['fit_common_gain_raw_mae'], 7. / 3.)
        self.assertAlmostEqual(pooled['audit_common_gain_raw_mae'], -.8)
        self.assertAlmostEqual(pooled['audit_fit_mass_standardized_gain_raw_mae'], 0.)
        self.assertAlmostEqual(pooled['composition_component_raw_mae'], -.8)
        self.assertAlmostEqual(pooled['within_cell_component_raw_mae'], -7. / 3.)
        self.assertAlmostEqual(equal['fit_common_gain_raw_mae'], 2.)
        self.assertAlmostEqual(equal['audit_common_gain_raw_mae'], -1.25)
        self.assertAlmostEqual(equal['audit_fit_mass_standardized_gain_raw_mae'], -.5)
        self.assertAlmostEqual(equal['composition_component_raw_mae'], -.75)
        self.assertAlmostEqual(equal['within_cell_component_raw_mae'], -2.5)
        for row in rows:
            self.assertAlmostEqual(row['audit_minus_fit_common_gain_raw_mae'],
                                   row['composition_component_raw_mae'] + row['within_cell_component_raw_mae'])
            self.assertAlmostEqual(row['decomposition_residual_raw_mae'], 0.)
            self.assertEqual(row['fit_shared_supported_fraction_of_all_region_mass'], 1.)
            self.assertEqual(row['audit_shared_supported_fraction_of_all_region_mass'], 1.)
            self.assertFalse(row['paired_windows'])
            self.assertFalse(row['confidence_interval_constructed'])
            self.assertFalse(any('ci_low' in key for key in row))
        self.assertFalse(result['automatic_development_gate'])
        self.assertFalse(result['deployable_router_constructed'])

    def test_composition_coverage_excludes_unsupported_and_supported_unshared_mass(self):
        records, features, times = fixture()
        features['audit']['history_mean'][:] = 0.
        features['audit']['history_mean'][0] = 2.
        _, _, _, _, rows = self.analyze(records, features, times)
        row = next(row for row in rows if row['region'] == 'candidate_h1_h6'
                   and row['comparison'] == 'state_vector_vs_A' and row['estimand'] == 'pooled')
        self.assertEqual(row['shared_supported_cells'], 1)
        self.assertEqual(row['fit_excluded_supported_unshared_region_mass'], 480)
        self.assertEqual(row['audit_excluded_unsupported_region_mass'], 6)
        self.assertAlmostEqual(row['fit_shared_supported_fraction_of_all_region_mass'], 1. / 3.)
        self.assertAlmostEqual(row['audit_shared_supported_fraction_of_all_region_mass'], 474. / 480.)

    def test_rejects_wrong_features_shapes_infinity_and_candidate_alignment(self):
        records, features, times = fixture()
        cases = []
        for name, value in (('history_mean', np.zeros(79)), ('history_mean', np.zeros(80, dtype=int)),
                            ('history_mean', np.full(80, np.inf)), ('candidate_node_count', np.full(80, 3.)),
                            ('history_missing_fraction', np.full(80, 1.1)), ('history_volatility', np.full(80, -.1)),
                            ('report_age_minutes', np.full(80, -1.)),
                            ('report_age_minutes', np.zeros(80)), ('report_age_minutes', np.full(80, 5.1))):
            bad = copy.deepcopy(features)
            bad['fit'][name] = value
            cases.append(bad)
        bad = copy.deepcopy(features)
        bad['fit'].pop('history_trend')
        cases.append(bad)
        for index, bad in enumerate(cases):
            with self.subTest(index=index), self.assertRaises(ValueError):
                self.analyze(records, bad, times)

    def test_rejects_corrupt_geometry_phases_and_timestamps(self):
        records, features, times = fixture()
        for key in ('counts', 'prediction_counts', 'errors'):
            bad = copy.deepcopy(records)
            bad['fit'][key][0, 0] += 1
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.analyze(bad, features, times)
        with self.assertRaisesRegex(ValueError, 'exactly fit and audit'):
            analyze_support({'fit': records['fit']}, features, times, BOOTSTRAP, STATE_SPEC)
        bad_times = copy.deepcopy(times)
        bad_times['audit'] = {key: value.replace('2023', '2022') for key, value in bad_times['audit'].items()}
        with self.assertRaisesRegex(ValueError, 'chronologically disjoint'):
            self.analyze(records, features, bad_times)
        bad_times = copy.deepcopy(times)
        bad_times['audit'][300] += '+00:00'
        with self.assertRaisesRegex(ValueError, 'timezone awareness'):
            self.analyze(records, features, bad_times)
        bad = copy.deepcopy(records)
        bad['audit']['source_indices'][0] = 100
        with self.assertRaisesRegex(ValueError, 'disjoint across phases'):
            self.analyze(bad, features, times)

    def test_rejects_post_result_support_threshold_or_feature_changes(self):
        records, features, times = fixture()
        for key, value in (('minimum_fit_windows', 29), ('minimum_fit_weeks', 3),
                           ('quantiles', [.5]), ('joint_features', ['history_mean']),
                           ('joint_quantiles', [.25])):
            spec = {**STATE_SPEC, key: value}
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'frozen v12j'):
                analyze_support(records, features, times, BOOTSTRAP, spec)
        with self.assertRaisesRegex(ValueError, 'five fields'):
            analyze_support(records, features, times, BOOTSTRAP, {**STATE_SPEC, 'new_cut': 1})


if __name__ == '__main__':
    unittest.main()
