"""Full-fit diagnostics use fabricated fixed-path error sums, never training data."""

import copy
import unittest

import numpy as np

from experiments.chronological import audit_architecture_regions as regions_audit
from experiments.chronological.state_interaction_fit import (
    analyze_phases, COMPARISONS, ESTIMANDS, PATHS, REGIONS,
)


SPEC = {'draws': 2000, 'seed': 12028, 'confidence': .95,
        'sensitivity_circular_block_weeks': 4, 'minimum_valid_draw_fraction': .95}


def fixture(empty_early=False, empty_all=False, missing_week=False):
    records, times = {}, {}
    phase_dates = {
        'fit': ['2023-01-02', '2023-01-09', '2023-01-16', '2023-01-23', '2023-01-30'],
        'audit': ['2023-07-03', '2023-07-10', '2023-07-17', '2023-07-24'],
    }
    for phase, dates in phase_dates.items():
        rows, clocks = [], {}
        for week, date in enumerate(dates):
            if missing_week and phase == 'fit' and week == 2:
                continue
            sample = (100 if phase == 'fit' else 200) + week
            # First fit window dominates cell pooling but is just one window.
            cells = np.array([6 if phase == 'fit' and week == 0 else 1, 2, 3], dtype=np.int64)
            if empty_early:
                cells[0] = 0
            if empty_all:
                cells[:] = 0
            early = .5 if phase == 'fit' and week == 0 else -1. if phase == 'fit' else -.25
            state = cells * np.array([early, .1 + week * .05, .2])
            arm_gain = np.stack((np.zeros(3), state * .1, state, state * .5))
            components = cells[None, :] * 100. - arm_gain
            errors = np.concatenate((components.sum(1, keepdims=True), components), axis=1)
            rows.append((sample, np.r_[cells.sum(), cells], errors))
            clocks[sample] = date + 'T00:00:00'
        n = len(rows)
        mask = np.tile(np.array([True, False]), (n, 1))
        records[phase] = {
            'ids': np.array([row[0] for row in rows], dtype=np.int64),
            'source_indices': np.array([row[0] for row in rows], dtype=np.int64),
            'candidate_mask': mask,
            'counts': np.array([row[1] for row in rows], dtype=np.int64),
            'prediction_counts': np.tile(np.array([24, 6, 6, 12], dtype=np.int64), (n, 1)),
            'errors': np.array([row[2] for row in rows], dtype=np.float64),
            'regions': list(REGIONS),
        }
        times[phase] = clocks
    return records, times


class FitTests(unittest.TestCase):
    def analyze(self, records=None, times=None):
        if records is None:
            records, times = fixture()
        return analyze_phases(records, COMPARISONS, SPEC, times)

    def test_gain_direction_and_distinct_cell_window_weighting(self):
        result, rows, weekly, gaps = self.analyze()
        effect = result['phases']['fit']['regions']['candidate_h1_h6']['comparisons']['state_vector_vs_A']
        self.assertAlmostEqual(effect['gain_raw_mae'], -.1)
        self.assertAlmostEqual(effect['equal_forecast_window_gain_raw_mae'], -.7)
        self.assertEqual(result['equal_weight_unit'], 'forecast_window_not_unique_accident')
        self.assertEqual(len(rows), 2 * 4 * 6)
        self.assertEqual(len(weekly), (5 + 4) * 4 * 6)
        self.assertEqual(len(gaps), 4 * 6 * 3)
        for phase in result['phases'].values():
            for region in phase['regions'].values():
                for name, pair in COMPARISONS.items():
                    expected = region['mae'][pair[0]] - region['mae'][pair[1]]
                    self.assertAlmostEqual(region['comparisons'][name]['gain_raw_mae'], expected)

    def test_intervals_match_existing_shared_week_bootstrap_producer(self):
        records, times = fixture()
        observed, _, _, _ = self.analyze(records, times)
        for phase in ('fit', 'audit'):
            # The old producer requires controls; these aliases are fixture-only.
            source, _, _ = regions_audit.analyze(
                {cohort: records[phase] for cohort in
                 ('incident_full', 'incident', 'primary_control', 'secondary_control')}, times[phase],
                {'paths': list(PATHS), 'comparisons': COMPARISONS, 'bootstrap': SPEC})
            for region in REGIONS:
                actual = observed['phases'][phase]['regions'][region]
                old = source['results']['incident_full']['regions'][region]
                self.assertEqual(actual['mae'], old['mae'])
                for name in COMPARISONS:
                    new, expected = actual['comparisons'][name], old['comparisons'][name]
                    self.assertEqual(new['gain_raw_mae'], expected['gain_raw_mae'])
                    self.assertEqual(new['equal_forecast_window_gain_raw_mae'], expected['equal_event_gain_raw_mae'])
                    for method in ('week', 'four_week_block'):
                        for estimand, old_key in (('pooled', 'pooled'),
                             ('equal_forecast_window', 'equal_event'), ('global_units', 'global_units')):
                            self.assertEqual(new['intervals'][method][estimand],
                                             expected['intervals'][method][old_key])

    def test_gap_algebra_is_unpaired_point_difference_without_intervals(self):
        result, _, _, rows = self.analyze()
        for row in rows:
            region, comparison, estimand = row['region'], row['comparison'], row['estimand']
            field = ESTIMANDS[estimand]
            expected = (result['phases']['audit']['regions'][region]['comparisons'][comparison][field]
                        - result['phases']['fit']['regions'][region]['comparisons'][comparison][field])
            self.assertAlmostEqual(row['phase_minus_fit_gain_raw_mae'], expected)
            self.assertFalse(row['paired_windows'])
            self.assertFalse(row['gap_confidence_interval_constructed'])
            self.assertFalse(any('ci_low' in key for key in row))
        self.assertIn('in_sample', result['phases']['fit']['interval_scope'])
        self.assertIn('posthoc_reused', result['phases']['audit']['interval_scope'])
        self.assertFalse(result['automatic_development_gate'])
        self.assertFalse(result['new_model_selection_performed'])

    def test_regional_valid_cell_contributions_partition_phase_global_gain(self):
        result, _, rows, _ = self.analyze()
        for phase in result['phases'].values():
            fractions = [phase['regions'][region]['valid_cell_fraction'] for region in REGIONS[1:]]
            self.assertAlmostEqual(sum(fractions), 1.)
            for name in COMPARISONS:
                parts = [phase['regions'][region]['comparisons'][name]['regional_gain_in_global_mae_units']
                         for region in REGIONS[1:]]
                self.assertAlmostEqual(sum(parts), phase['regions']['all']['comparisons'][name]['gain_raw_mae'])
        for phase in ('fit', 'audit'):
            weeks = result['phases'][phase]['weeks']
            for week in weeks:
                selected = [row for row in rows if row['phase'] == phase and row['positive_week'] == week
                            and row['comparison'] == 'state_vector_vs_A']
                overall = next(row for row in selected if row['region'] == 'all')
                self.assertAlmostEqual(overall['gain_raw_mae'],
                    sum(row['regional_gain_in_global_mae_units'] for row in selected if row['region'] != 'all'))

    def test_different_phase_lengths_and_empty_calendar_week_are_not_paired(self):
        records, times = fixture(missing_week=True)
        result, _, weekly, gaps = self.analyze(records, times)
        self.assertEqual(len(result['phases']['fit']['weeks']), 5)
        self.assertEqual(len(result['phases']['audit']['weeks']), 4)
        empty = [row for row in weekly if row['phase'] == 'fit' and row['positive_week'] == '2023-W03']
        self.assertEqual(len(empty), 24)
        self.assertTrue(all(row['forecast_windows'] == 0 and row['gain_raw_mae'] is None for row in empty))
        self.assertTrue(all(row['status'] == 'UNDEFINED_EMPTY_SUPPORT' for row in empty))
        self.assertTrue(all(not row['paired_windows'] for row in gaps))

    def test_empty_region_is_explicit_but_contributes_zero_to_nonempty_global(self):
        records, times = fixture(empty_early=True)
        result, _, _, gaps = self.analyze(records, times)
        early = result['phases']['fit']['regions']['candidate_h1_h6']
        self.assertEqual(early['valid_cells'], 0)
        self.assertEqual(early['valid_cell_fraction'], 0.)
        for effect in early['comparisons'].values():
            self.assertIsNone(effect['gain_raw_mae'])
            self.assertIsNone(effect['equal_forecast_window_gain_raw_mae'])
            self.assertEqual(effect['regional_gain_in_global_mae_units'], 0.)
            self.assertEqual(effect['intervals']['week']['pooled']['valid_draws'], 0)
            self.assertEqual(effect['intervals']['week']['pooled']['status'], 'INSUFFICIENT_VALID_DRAWS')
        early_gaps = [row for row in gaps if row['region'] == 'candidate_h1_h6']
        self.assertTrue(all(row['phase_minus_fit_gain_raw_mae'] is None
                            for row in early_gaps if row['estimand'] != 'global_units'))

    def test_rejects_empty_full_positive_global_support_in_either_phase(self):
        records, times = fixture()
        empty, _ = fixture(empty_all=True)
        for phase in ('fit', 'audit'):
            bad = copy.deepcopy(records)
            bad[phase] = empty[phase]
            with self.subTest(phase=phase), self.assertRaisesRegex(ValueError, 'global target support is empty'):
                self.analyze(bad, times)

    def test_rejects_different_node_axis_even_when_each_geometry_is_consistent(self):
        records, times = fixture()
        records['audit']['candidate_mask'] = np.concatenate((records['audit']['candidate_mask'],
            np.zeros((len(records['audit']['ids']), 1), dtype=bool)), axis=1)
        records['audit']['prediction_counts'][:, 0] += 12
        records['audit']['prediction_counts'][:, 3] += 12
        with self.assertRaisesRegex(ValueError, 'same node axis'):
            self.analyze(records, times)

    def test_empty_region_global_contribution_interval_and_weekly_status_are_not_undefined(self):
        records, times = fixture(empty_early=True)
        result, _, weekly, _ = self.analyze(records, times)
        for phase in ('fit', 'audit'):
            early = result['phases'][phase]['regions']['candidate_h1_h6']
            for effect in early['comparisons'].values():
                for method in ('week', 'four_week_block'):
                    confidence = effect['intervals'][method]['global_units']
                    self.assertEqual(confidence['status'], 'OK')
                    self.assertEqual(confidence['ci_low'], 0.)
                    self.assertEqual(confidence['ci_high'], 0.)
            selected = [row for row in weekly if row['phase'] == phase and row['region'] == 'candidate_h1_h6']
            self.assertTrue(all(row['gain_raw_mae'] is None
                and row['regional_gain_in_global_mae_units'] == 0. for row in selected))

    def test_rejects_corrupt_identity_mask_counts_errors_and_partitions(self):
        records, times = fixture()
        cases = []
        def changed(key, value):
            bad = copy.deepcopy(records)
            bad['fit'][key] = value
            cases.append(bad)
        changed('ids', np.array([100, 100, 102, 103, 104]))
        changed('source_indices', np.array([100, 100, 102, 103, 104]))
        changed('candidate_mask', records['fit']['candidate_mask'].astype(np.int64))
        changed('counts', records['fit']['counts'].astype(float))
        changed('errors', records['fit']['errors'].astype(np.int64))
        changed('regions', list(REGIONS)[::-1])
        for key, index, value in (
            ('counts', (0, 0), 25), ('counts', (0, 0), -1),
            ('prediction_counts', (0, 1), 5), ('errors', (0, 0, 1), np.nan),
            ('errors', (0, 0, 1), -1.), ('errors', (0, 0, 0), 2000.)):
            bad = copy.deepcopy(records)
            bad['fit'][key][index] = value
            cases.append(bad)
        for number, bad in enumerate(cases):
            with self.subTest(case=number), self.assertRaises(ValueError):
                self.analyze(bad, times)

    def test_rejects_nonzero_error_on_empty_support(self):
        records, times = fixture(empty_early=True)
        records['fit']['errors'][0, 0, 1] = 1.
        with self.assertRaisesRegex(ValueError, 'target support'):
            self.analyze(records, times)

    def test_rejects_phase_overlap_and_wrong_timestamp_membership(self):
        records, times = fixture()
        bad = copy.deepcopy(records)
        bad['audit']['ids'][0] = bad['fit']['ids'][0]
        clocks = copy.deepcopy(times)
        clocks['audit'][100] = clocks['audit'].pop(200)
        with self.assertRaisesRegex(ValueError, 'disjoint across phases'):
            self.analyze(bad, clocks)
        clocks = copy.deepcopy(times)
        clocks['fit'][999] = '2023-01-16T00:00:00'
        with self.assertRaisesRegex(ValueError, 'exact sample identity set'):
            self.analyze(records, clocks)
        clocks = copy.deepcopy(times)
        clocks['audit'] = {sample: '2022-07-03T00:00:00' for sample in clocks['audit']}
        with self.assertRaisesRegex(ValueError, 'chronologically disjoint'):
            self.analyze(records, clocks)
        clocks = copy.deepcopy(times)
        clocks['audit'][200] = '2023-07-03T00:00:00+00:00'
        with self.assertRaisesRegex(ValueError, 'timezone awareness'):
            self.analyze(records, clocks)

    def test_rejects_missing_extra_or_misdirected_path_comparisons(self):
        records, times = fixture()
        for comparisons in (
            {key: value for key, value in COMPARISONS.items() if key != 'strength_vs_A'},
            {**COMPARISONS, 'strength_vs_A': ['strength', 'A']},
            dict(reversed(list(COMPARISONS.items())))):
            with self.assertRaisesRegex(ValueError, 'six fixed directed'):
                analyze_phases(records, comparisons, SPEC, times)

    def test_rejects_missing_extra_phases_and_bad_bootstrap_settings(self):
        records, times = fixture()
        with self.assertRaisesRegex(ValueError, 'exactly fit and audit'):
            self.analyze({'fit': records['fit']}, {'fit': times['fit']})
        with self.assertRaisesRegex(ValueError, 'exactly fit and audit'):
            self.analyze({**records, 'selection': records['fit']}, {**times, 'selection': times['fit']})
        for key, value in (('draws', True), ('seed', -1), ('confidence', 1.),
                           ('minimum_valid_draw_fraction', 0.), ('sensitivity_circular_block_weeks', 5)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                analyze_phases(records, COMPARISONS, {**SPEC, key: value}, times)


if __name__ == '__main__':
    unittest.main()
