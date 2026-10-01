"""Saved v12f selection replay, regional accounting and integrity checks."""

import copy
import unittest

from experiments.chronological import state_interaction_trajectory as trajectory


def fixture(epochs=2):
    protocol = {'training': {'epochs': epochs, 'clip_grad_norm': 5.},
                'selection': {'maximum_relative_harm': .001,
                              'protected_cohorts': list(trajectory.COHORTS),
                              'protected_regions': list(trajectory.REGIONS)}}
    fractions = dict(zip(trajectory.REGIONS, [1., .1, .1, .8]))
    baseline = {c: {'mae': dict.fromkeys(trajectory.REGIONS, 10.),
                    'valid_cell_fraction': fractions.copy()}
                for c in trajectory.COHORTS}
    history = []
    for epoch in range(1, epochs + 1):
        selection = {c: {'samples': 3, 'mae': dict.fromkeys(trajectory.REGIONS, 10.),
                         'candidate_gate': None, 'candidate_representation': {'mean': None}}
                     for c in trajectory.COHORTS}
        history.append({'epoch': epoch,
                        'training': {'mae_standardized': .1, 'maximum_gradient_norm': .01,
                                     'gate_parameters_changed': True, 'optimizer_steps': 2},
                        'selection': selection, 'eligible': True,
                        'protection_checks': {f'{c}/{r}': True for c in trajectory.COHORTS for r in trajectory.REGIONS},
                        'best_epoch': 0})
    detail = {'selected_epoch': 0, 'optimizer_steps': epochs * 2,
              'selection_metrics': {c: {'mae': v['mae'].copy()} for c, v in baseline.items()},
              'initial_prediction_exactly_A': True, 'backbone_state_unchanged': True}
    return history, detail, baseline, protocol


def set_regions(entry, cohort, early, late, noncandidate):
    entry['selection'][cohort]['mae'] = dict(zip(trajectory.REGIONS,
        [.1 * early + .1 * late + .8 * noncandidate, early, late, noncandidate]))


class TrajectoryTests(unittest.TestCase):
    def analyze(self, values):
        return trajectory.analyze_history(*values, expected_steps=2)

    def test_epoch_zero_wins_ties_and_metadata_does_not_require_scalar_gate(self):
        values = fixture()
        for entry in values[0]:
            for metrics in entry['selection'].values():
                metrics['mae'].update(candidate_h1_h3=10., all_h1=10.)
        for metrics in values[1]['selection_metrics'].values():
            metrics['mae'].update(candidate_h1_h3=10., all_h1=10.)
        before = copy.deepcopy(values)
        result, rows = self.analyze(values)
        self.assertEqual(result['selected_epoch'], 0)
        self.assertEqual(result['eligible_epochs'], 2)
        self.assertIn('NO_STRICT', result['fallback_reason'])
        self.assertEqual(len(rows), 32)
        self.assertFalse(result['candidate_gate_used_for_selection'])
        self.assertEqual(values, before)

    def test_strict_selection_and_region_contributions_detect_local_harm(self):
        values = fixture()
        history, detail, _, _ = values
        for entry in history:
            set_regions(entry, 'incident_full', 10.005, 10., 9.999)
            entry['best_epoch'] = 1
        detail['selected_epoch'] = 1
        detail['selection_metrics'] = copy.deepcopy(history[0]['selection'])
        result, rows = self.analyze(values)
        self.assertEqual(result['selection_update_epochs'], [1])
        self.assertEqual(result['cohorts']['incident_full']['global_improved_while_candidate_h1_h6_worse_epochs'], [1, 2])
        self.assertEqual(result['cohorts']['incident_full']['candidate_h1_h6_worsened_epochs'], [1, 2])
        grouped = [r for r in rows if r['epoch'] == 1 and r['cohort'] == 'incident_full']
        whole = next(r['gain_vs_A_raw'] for r in grouped if r['region'] == 'all')
        contributions = sum(r['weighted_global_contribution_raw'] for r in grouped if r['region'] != 'all')
        self.assertAlmostEqual(whole, contributions)
        self.assertAlmostEqual(contributions, .0003)
        self.assertLess(next(r['weighted_global_contribution_raw'] for r in grouped if r['region'] == 'candidate_h1_h6'), 0)

    def test_improved_global_does_not_override_a_failed_local_protection(self):
        values = fixture()
        history = values[0]
        set_regions(history[0], 'incident_full', 10.02, 10., 9.99)
        history[0]['eligible'] = False
        history[0]['protection_checks']['incident_full/candidate_h1_h6'] = False
        result, _ = self.analyze(values)
        self.assertEqual(result['selected_epoch'], 0)
        self.assertEqual(result['rejected_epochs'][0]['failed_constraints'], ['incident_full/candidate_h1_h6'])
        self.assertEqual(result['rejected_epochs'][0]['failures'][0]['reason'], 'harm_bound_exceeded')

    def test_undefined_metric_rejects_epoch_without_inventing_zero_gain(self):
        values = fixture()
        first = values[0][0]
        first['selection']['primary_control']['mae']['candidate_h1_h6'] = None
        first['eligible'] = False
        first['protection_checks']['primary_control/candidate_h1_h6'] = False
        result, rows = self.analyze(values)
        self.assertFalse(result['epoch_summary'][0]['partition_reconciliation']['primary_control']['defined'])
        null_row = next(r for r in rows if r['epoch'] == 1 and r['cohort'] == 'primary_control' and r['region'] == 'candidate_h1_h6')
        self.assertIsNone(null_row['gain_vs_A_raw'])
        self.assertIsNone(null_row['weighted_global_contribution_raw'])
        self.assertEqual(result['rejected_epochs'][0]['failures'][0]['reason'], 'undefined_metric')

    def test_no_eligible_epochs_has_distinct_fallback_reason(self):
        values = fixture()
        for entry in values[0]:
            entry['selection']['primary_control']['mae']['all'] = None
            entry['eligible'] = False
            entry['protection_checks']['primary_control/all'] = False
        result, _ = self.analyze(values)
        self.assertEqual(result['fallback_reason'], 'NO_ELIGIBLE_EPOCH')

    def test_saved_flag_best_epoch_selected_metrics_and_steps_are_checked(self):
        mutations = [
            (lambda v: v[0][0].update(eligible=False), 'eligibility'),
            (lambda v: v[0][0]['protection_checks'].update({'incident/all': False}), 'protection checks'),
            (lambda v: v[0][0].update(best_epoch=1), 'best epoch'),
            (lambda v: v[1].update(selected_epoch=1), 'selected epoch'),
            (lambda v: v[1]['selection_metrics']['incident']['mae'].update(all=9.), 'selected metrics'),
            (lambda v: v[0][0]['training'].update(optimizer_steps=3), 'fixed budget'),
            (lambda v: v[1].update(optimizer_steps=5), 'total optimizer'),
            (lambda v: v[1].update(backbone_state_unchanged=False), 'frozen backbone'),
            (lambda v: v[0][0].update(epoch=2), 'contiguous'),
        ]
        for mutate, message in mutations:
            with self.subTest(message=message):
                values = fixture()
                mutate(values)
                with self.assertRaisesRegex(ValueError, message):
                    self.analyze(values)

    def test_history_budget_original_constraints_and_accounting_cannot_be_relaxed(self):
        mutations = [
            (lambda v: v[0].pop(), 'every configured epoch'),
            (lambda v: v[3]['selection'].update(maximum_relative_harm=.01), '16 protection'),
            (lambda v: v[3]['selection']['protected_regions'].pop(), '16 protection'),
            (lambda v: v[2]['incident']['valid_cell_fraction'].update(noncandidate_all=.9), 'partition all cells'),
            (lambda v: v[0][0]['selection']['incident']['mae'].update(all=9.), 'do not reconcile'),
        ]
        for mutate, message in mutations:
            with self.subTest(message=message):
                values = fixture()
                mutate(values)
                with self.assertRaisesRegex(ValueError, message):
                    self.analyze(values)

    def test_gradient_norms_are_preclip_and_invalid_training_is_rejected(self):
        values = fixture()
        values[0][1]['training']['maximum_gradient_norm'] = 6.
        result, _ = self.analyze(values)
        self.assertEqual(result['training']['epochs_with_some_preclip_gradient_norm_above_limit'], [2])
        self.assertIn('pre-clipping', result['training']['gradient_norm_interpretation'])
        for key, bad in [('mae_standardized', float('nan')), ('maximum_gradient_norm', float('inf')),
                         ('maximum_gradient_norm', -1), ('mean_minibatch_identity_penalty', float('nan'))]:
            with self.subTest(key=key, bad=bad):
                values = fixture()
                values[0][0]['training'][key] = bad
                with self.assertRaisesRegex(ValueError, 'finite'):
                    self.analyze(values)
        for key, bad in [('maximum_gradient_norm', 0.), ('gate_parameters_changed', False)]:
            values = fixture()
            values[0][0]['training'][key] = bad
            with self.assertRaisesRegex(ValueError, 'effective learning'):
                self.analyze(values)


if __name__ == '__main__':
    unittest.main()
