"""Hand-calculated pure NumPy F1 decompositions; no model or torch imports."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np

from experiments.chronological import min_forecast_fusion_gain_metrics as metrics


CASES = ('STATE_ON', 'EVENT_ON', 'EVENT_REPORT_OFF', 'EVENT_FUSION_OFF', 'EVENT_UNIFORM',
         'EVENT_SEMANTIC_UNK', 'EVENT_LOCATION_SHUFFLE', 'SOURCE_MIN')
EXPECTED = {'EVENT_vs_STATE': 2., 'report_content_dependency': 1., 'fusion_dependency': 4.,
    'state_fusion_dependency': 3., 'component_differentiation': .5, 'common_scaling_dependency': 3.5,
    'semantic_UNK_sensitivity': .25, 'location_shuffle_sensitivity': .75,
    'EVENT_vs_SOURCE': 3., 'STATE_vs_SOURCE': 1., 'current_backbone_vs_SOURCE': -1.,
    'remaining_model_difference': 1.}


def fixture(rows=2, nodes=4):
    target = np.full((rows, 12, nodes), 10., np.float64)
    values = dict(STATE_ON=13., EVENT_ON=11., EVENT_REPORT_OFF=12., EVENT_FUSION_OFF=15.,
        EVENT_UNIFORM=11.5, EVENT_SEMANTIC_UNK=11.25, EVENT_LOCATION_SHUFFLE=11.75, SOURCE_MIN=14.)
    predictions = {case: np.full_like(target, value) for case, value in values.items()}
    direct, potential = np.zeros((rows, nodes), bool), np.zeros((rows, nodes), bool)
    direct[:, 0] = True
    if nodes > 1:
        potential[:, 1] = True
    regions = dict(all_nodes=np.ones((rows, nodes), bool), direct_report_nodes=direct,
        potential_only=potential, remaining_nodes=~(direct | potential), empty=np.zeros((rows, nodes), bool),
        **{'road:WEST': np.arange(nodes) < max(1, nodes//2)})
    samples, stations = np.arange(rows, dtype=np.int64)*100+100, np.arange(nodes, dtype=np.int64)+500
    return [predictions, target, np.ones_like(target, bool), regions, samples, stations, samples[-1:].copy()]


class ForecastFusionGainMetricTests(unittest.TestCase):
    def analyze(self, args, **kwargs):
        return metrics.analyze_predictions(*args, **kwargs)

    def group(self, result, comparison, name='all_nodes'):
        return result['comparisons'][comparison]['groups'][name]

    def test_twelve_comparisons_and_four_arithmetic_identities_hand_calculated(self):
        result = self.analyze(fixture())
        self.assertEqual(set(result['comparisons']), set(EXPECTED))
        for name, expected in EXPECTED.items():
            m = self.group(result, name)
            self.assertEqual(m['mae_gain_pooled'], expected)
            self.assertEqual(m['mae_gain_macro'], expected)
            self.assertEqual(m['global_mae_contribution_pooled'], expected)
            self.assertEqual(m['global_mae_contribution_macro'], expected)
            self.assertAlmostEqual(m['mape_gain_pooled'], expected/10)
        identities = [
            ('EVENT_vs_SOURCE', 'current_backbone_vs_SOURCE', 'fusion_dependency'),
            ('fusion_dependency', 'state_fusion_dependency', 'report_content_dependency'),
            ('fusion_dependency', 'common_scaling_dependency', 'component_differentiation'),
            ('EVENT_vs_STATE', 'remaining_model_difference', 'report_content_dependency')]
        for left, first, second in identities:
            for field in ('mae_gain_pooled', 'mae_gain_macro', 'global_mae_contribution_pooled', 'global_mae_contribution_macro'):
                self.assertEqual(self.group(result, left)[field], self.group(result, first)[field]+self.group(result, second)[field])
        certificates = result['identity_check']['decompositions']
        self.assertEqual(len(certificates), 4)
        self.assertEqual({(row['total'], frozenset(row['parts'])) for row in certificates.values()},
                         {(total, frozenset((first, second))) for total, first, second in identities})
        for row in certificates.values():
            self.assertEqual(row['status'], 'POINTWISE_AND_GROUPED_IDENTITY_PASS')
            self.assertEqual(row['max_pointwise_residual'], 0.)
            self.assertEqual(row['max_grouped_residual'], 0.)
            for group in result['group_definitions']:
                for field in ('global_mae_contribution_pooled', 'global_mae_contribution_macro'):
                    self.assertEqual(self.group(result, row['total'], group)[field],
                                     sum(self.group(result, part, group)[field] for part in row['parts']))
        # Signed identities never imply additivity of positive or negative parts.
        positive = lambda name: self.group(result, name)['gain_positive_pooled']
        self.assertNotEqual(positive('EVENT_vs_SOURCE'), positive('current_backbone_vs_SOURCE')+positive('fusion_dependency'))
        self.assertEqual(self.group(result, 'current_backbone_vs_SOURCE')['gain_negative_pooled'], 1.)
        json.dumps(result, allow_nan=False)

    def test_prediction_up_down_and_cancellation_are_distinct_from_error_gain(self):
        args = fixture(rows=1)
        # Two rising predictions help and two rising predictions hurt.
        args[0]['EVENT_FUSION_OFF'][:] = [8., 12., 8., 12.]
        args[0]['EVENT_ON'][:] = [9., 13., 9., 13.]
        result = self.analyze(args)
        m = self.group(result, 'fusion_dependency')
        self.assertEqual(m['mae_gain_pooled'], 0.)
        self.assertEqual(m['gain_positive_pooled'], .5)
        self.assertEqual(m['gain_negative_pooled'], .5)
        self.assertEqual(m['cancellation_pooled'], 1.)
        self.assertEqual(m['prediction_change']['up_fraction'], 1.)
        self.assertEqual(m['prediction_change']['down_fraction'], 0.)
        self.assertEqual(m['prediction_change']['signed_mean_candidate_minus_control_pooled'], 1.)
        args[0]['EVENT_ON'][:] = [9., 11., 9., 11.]
        m = self.group(self.analyze(args), 'fusion_dependency')
        self.assertEqual(m['prediction_change']['up_fraction'], .5)
        self.assertEqual(m['prediction_change']['down_fraction'], .5)
        self.assertEqual(m['mae_gain_pooled'], 1.)

    def test_current_MIN_and_source_bias_are_separately_fixed_for_all_comparisons(self):
        args = fixture()
        args[0]['SOURCE_MIN'][:] = 8.
        args[0]['STATE_ON'][:] = 8.
        args[0]['EVENT_FUSION_OFF'][:] = 15.
        result = self.analyze(args)
        for comparison in result['comparisons']:
            self.assertEqual(self.group(result, comparison, 'bias:over')['valid_count'], 96)
            self.assertEqual(self.group(result, comparison, 'bias:under')['valid_count'], 0)
            self.assertEqual(self.group(result, comparison, 'source_bias:under')['valid_count'], 96)
            self.assertEqual(self.group(result, comparison, 'source_bias:over')['valid_count'], 0)
        self.assertEqual(self.group(result, 'report_content_dependency', 'bias:over')['mae_gain_pooled'], 1.)

    def test_all_exhaustive_families_reconcile_pooled_macro_and_MAPE(self):
        args = fixture(rows=3)
        rng = np.random.default_rng(241)
        args[1][:] = np.broadcast_to(np.array([2., 15., 75., 300.]), args[1].shape)
        for case in args[0]:
            args[0][case][:] = args[1]+rng.normal(0., 3., args[1].shape)
        args[2][0, ::2, -1] = False
        result = self.analyze(args)
        definitions = result['group_definitions']
        cross = [name for name in definitions if name.startswith('cross:')]
        self.assertEqual(len(cross), 54)
        families = [[name for name in definitions if name.startswith('flow:')],
            [name for name in definitions if name.startswith('bias:')],
            [name for name in definitions if name.startswith('source_bias:')], cross,
            ['direct_report_nodes', 'potential_only', 'remaining_nodes']]
        for comparison in result['comparisons']:
            global_m = self.group(result, comparison)
            for names in families:
                self.assertEqual(sum(self.group(result, comparison, name)['valid_count'] for name in names), global_m['valid_count'])
                for field in ('global_mae_contribution_pooled', 'global_mae_contribution_macro',
                              'global_mape_contribution_pooled', 'global_mape_contribution_macro'):
                    self.assertAlmostEqual(sum(self.group(result, comparison, name)[field] for name in names), global_m[field], places=12)
            empty = self.group(result, comparison, 'empty')
            self.assertEqual(empty['status'], 'unavailable')
            self.assertIsNone(empty['mae_gain_pooled'])
            self.assertIsNone(empty['mae_gain_macro'])
            self.assertEqual(empty['global_mae_contribution_pooled'], 0.)
            self.assertEqual(empty['global_mae_contribution_macro'], 0.)
        rows = metrics.flat_rows(result)
        self.assertEqual(len(rows), len(definitions)*12*13)
        self.assertEqual({row['horizon'] for row in rows}, set(range(13)))
        self.assertTrue(all(set(row) == set(rows[0]) for row in rows))

    def test_horizon_group_macro_can_oppose_global_contribution(self):
        args = fixture(rows=1, nodes=100)
        args[0]['EVENT_FUSION_OFF'][:] = 8.
        args[0]['EVENT_ON'][:] = 8.
        for h in range(12):
            count, candidate = (10, 13.) if h < 6 else (90, 11.5)
            args[0]['EVENT_FUSION_OFF'][0, h, :count] = 12.
            args[0]['EVENT_ON'][0, h, :count] = candidate
        m = self.group(self.analyze(args), 'fusion_dependency', 'bias:over')
        self.assertEqual(m['mae_gain_macro'], -.25)
        self.assertAlmostEqual(m['mae_gain_pooled'], .35)
        self.assertAlmostEqual(m['global_mae_contribution_macro'], .175)
        self.assertEqual(m['valid_count_per_horizon'], [10]*6+[90]*6)
        self.assertEqual(m['per_horizon_mae_gain'], [-1.]*6+[.5]*6)

    def test_fixed_hard_saved_ids_are_not_reselected_and_window_weights_are_explicit(self):
        args = fixture(rows=3)
        args[0]['STATE_ON'][0] = 12.
        args[0]['STATE_ON'][1] = 16.
        args[0]['SOURCE_MIN'][2] = 1000.  # Would win a new hard-window search; must remain excluded.
        args[2][1, :, 1:] = False
        args[6] = args[4][[1, 0]].copy()
        fixed = self.analyze(args)['fixed_hard_windows']
        self.assertEqual(fixed['sample_indices'], [200, 100])
        self.assertEqual(fixed['complete_window_count'], 2)
        group = fixed['comparisons']['EVENT_vs_STATE']
        self.assertEqual(group['window_equal_weight_mean_gain'], 3.)
        self.assertAlmostEqual(group['point_metrics']['mae_gain_pooled'], 1.8)

    def test_RMSE_and_MAPE_use_saved_scale_and_zero_targets_not_wrong_denominators(self):
        args = fixture(rows=1)
        args[0]['EVENT_REPORT_OFF'][:] = 12.
        args[0]['EVENT_ON'][:] = 11.
        args[1][0, 0, 0] = 0.
        m = self.group(self.analyze(args), 'report_content_dependency')
        self.assertEqual(m['valid_count'], 48)
        self.assertEqual(m['positive_target_count'], 47)
        self.assertEqual(m['mae_gain_pooled'], 1.)
        self.assertAlmostEqual(m['control']['rmse_pooled'], np.sqrt(332/48))
        self.assertAlmostEqual(m['candidate']['rmse_pooled'], np.sqrt(168/48))
        self.assertAlmostEqual(m['rmse_gain_pooled'], np.sqrt(332/48)-np.sqrt(168/48))
        self.assertAlmostEqual(m['control']['mape_pooled'], .2)
        self.assertAlmostEqual(m['candidate']['mape_pooled'], .1)
        self.assertAlmostEqual(m['mape_gain_pooled'], .1)  # fraction, not 10 or 1000.
        args[2][0, 0, 0] = False
        m = self.group(self.analyze(args), 'report_content_dependency')
        self.assertEqual(m['valid_count'], 47)
        self.assertEqual(m['positive_target_count'], 47)
        self.assertEqual(m['control']['rmse_pooled'], 2.)
        self.assertEqual(m['candidate']['rmse_pooled'], 1.)

    def test_missing_horizon_is_unavailable_macro_and_response_tol_only_counts_motion(self):
        args = fixture(rows=1)
        args[2][0, 0] = False
        result = self.analyze(args)
        m = self.group(result, 'fusion_dependency')
        self.assertEqual(m['status'], 'partial_horizons')
        self.assertIsNone(m['mae_gain_macro'])
        self.assertIsNone(m['global_mae_contribution_macro'])
        self.assertIsNotNone(m['mae_gain_pooled'])
        self.assertIsNone(m['per_horizon_mae_gain'][0])
        self.assertEqual(result['fixed_hard_windows']['complete_window_count'], 0)
        args = fixture(rows=1)
        args[0]['EVENT_FUSION_OFF'][:] = 11.0005
        low = self.group(self.analyze(args, response_tol=.0001), 'fusion_dependency')
        high = self.group(self.analyze(args, response_tol=.001), 'fusion_dependency')
        self.assertEqual(low['prediction_change']['response_fraction'], 1.)
        self.assertEqual(high['prediction_change']['response_fraction'], 0.)
        self.assertEqual(low['mae_gain_pooled'], high['mae_gain_pooled'])
        self.assertEqual(low['improved_fraction'], high['improved_fraction'])

    def test_eight_directions_separate_harmful_overshoot_from_away_for_each_own_control(self):
        args = fixture(rows=1, nodes=8)
        args[0]['EVENT_FUSION_OFF'][:] = [12., 8., 8., 8., 8., 8., 8., 10.]
        args[0]['EVENT_ON'][:] = [12., 9., 10., 11., 12., 13., 7., 11.]
        args[0]['EVENT_REPORT_OFF'][:] = 12.
        args[0]['SOURCE_MIN'][:] = 8.
        result = self.analyze(args)
        directions = result['directions']['fusion_dependency']['groups']['all_nodes']['directions']
        expected = {'unchanged': 12, 'toward_without_crossing': 12, 'reached_target': 12,
            'overshoot_improved': 12, 'overshoot_equal_error': 12, 'overshoot_worsened': 12,
            'away_from_target': 12, 'exact_OFF_changed': 12}
        self.assertEqual({key: value['valid_count'] for key, value in directions.items()}, expected)
        self.assertEqual(directions['overshoot_worsened']['gain_pooled'], -1.)
        self.assertEqual(directions['away_from_target']['gain_pooled'], -1.)
        self.assertEqual(directions['reached_target']['gain_pooled'], 2.)
        self.assertEqual(directions['overshoot_equal_error']['gain_pooled'], 0.)
        # Report sensitivity uses C=12, not F or SOURCE, as its direction baseline.
        report = result['directions']['report_content_dependency']['groups']['all_nodes']['directions']
        self.assertEqual(report['unchanged']['valid_count'], 24)
        self.assertEqual(report['toward_without_crossing']['valid_count'], 24)
        self.assertEqual(report['reached_target']['valid_count'], 12)
        self.assertEqual(report['overshoot_improved']['valid_count'], 12)
        self.assertEqual(report['overshoot_worsened']['valid_count'], 12)
        self.assertEqual(report['away_from_target']['valid_count'], 12)
        self.assertEqual(report['exact_OFF_changed']['valid_count'], 0)
        for name in ('fusion_dependency', 'report_content_dependency'):
            direction_groups = result['directions'][name]['groups']
            m = self.group(result, name)
            self.assertEqual(sum(row['valid_count'] for row in direction_groups['all_nodes']['directions'].values()), m['valid_count'])
            self.assertAlmostEqual(sum(row['global_mae_contribution_pooled'] for row in direction_groups['all_nodes']['directions'].values()), m['mae_gain_pooled'])
        high = self.analyze(args, response_tol=100.)
        for name in ('fusion_dependency', 'report_content_dependency'):
            self.assertEqual(result['directions'][name]['groups']['all_nodes']['directions'],
                             high['directions'][name]['groups']['all_nodes']['directions'])

    def test_input_axes_masks_ids_fields_and_tolerance_are_validated(self):
        for case in CASES:
            with self.subTest(missing=case):
                args = fixture()
                del args[0][case]
                with self.assertRaises(ValueError):
                    self.analyze(args)
            with self.subTest(shape=case):
                args = fixture()
                args[0][case] = args[0][case][:, :, :3]
                with self.assertRaises(ValueError):
                    self.analyze(args)
            with self.subTest(nonfinite=case):
                args = fixture()
                args[0][case][0, 0, 0] = np.inf
                with self.assertRaises(ValueError):
                    self.analyze(args)
        mutations = [lambda a: a.__setitem__(1, a[1][:, :11]), lambda a: a.__setitem__(2, a[2].astype(np.uint8)),
            lambda a: a[2].fill(False), lambda a: a[1].__setitem__((0, 0, 0), -1.),
            lambda a: a.__setitem__(4, np.array([100, 100])), lambda a: a.__setitem__(4, np.array([-1, 100])),
            lambda a: a.__setitem__(5, np.array([1, 1, 2, 3])), lambda a: a.__setitem__(6, np.array([999])),
            lambda a: a.__setitem__(6, np.array([100, 100])), lambda a: a[3]['remaining_nodes'].fill(True),
            lambda a: a[3].__setitem__('all_nodes', np.zeros((2, 4), bool)),
            lambda a: a[3].__setitem__('empty', np.ones((12, 4), bool))]
        for number, mutate in enumerate(mutations):
            with self.subTest(mutation=number):
                args = fixture()
                mutate(args)
                with self.assertRaises(ValueError):
                    self.analyze(args)
        for tol in (-1., np.inf, np.nan, True):
            with self.subTest(tol=tol), self.assertRaises(ValueError):
                self.analyze(fixture(), response_tol=tol)
        for edges in ((0., 5., 5.), (1., 5.), (0., np.inf), (0.,)):
            with self.subTest(edges=edges), self.assertRaises(ValueError):
                self.analyze(fixture(), flow_edges=edges)

    def test_fractional_roundoff_and_all_sources_are_preserved_and_no_torch_import(self):
        args = fixture(rows=3)
        rng = np.random.default_rng(112)
        for case in args[0]:
            args[0][case][:] = rng.normal(10., 5., args[1].shape)
        original = copy.deepcopy(args)
        result = self.analyze(args)
        json.dumps(result, allow_nan=False)
        for actual, before in zip(args, original):
            if isinstance(actual, dict):
                for name in actual:
                    np.testing.assert_array_equal(actual[name], before[name])
            else:
                np.testing.assert_array_equal(actual, before)
        command = 'import sys; from experiments.chronological import min_forecast_fusion_gain_metrics; assert "torch" not in sys.modules'
        subprocess.run([sys.executable, '-c', command], cwd=Path(__file__).resolve().parents[1], check=True, capture_output=True, text=True)


if __name__ == '__main__':
    unittest.main()
