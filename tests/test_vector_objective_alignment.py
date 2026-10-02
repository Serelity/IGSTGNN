"""Objective/selector factor isolation, shared weighting and protected fallback."""

import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.chronological import vector_objective_alignment as a
from experiments.chronological import train_vector_objective_alignment as e
from test_incident_strength_gate import SyntheticDataset, tiny_model


def selection_metrics(global_mae=10., early_mae=10.):
    result = {cohort: {'mae': dict.fromkeys(a.base.REGIONS, 10.)} for cohort in a.base.COHORTS}
    result['incident_full']['mae'].update(all=global_mae, candidate_h1_h6=early_mae)
    return result


class ObjectiveTests(unittest.TestCase):
    def test_protocol_freezes_two_by_two_design_and_protection(self):
        p = e.load_protocol()
        self.assertEqual(p['arms'], list(a.ARMS))
        self.assertEqual(p['losses'], list(a.LOSSES))
        self.assertEqual(p['selectors'], list(a.SELECTORS))
        self.assertFalse(p['information_boundary']['audit_Y_used_for_selection'])
        self.assertFalse(p['information_boundary']['control_Y_used_for_optimizer'])
        self.assertFalse(p['information_boundary']['v12j_groups_used_for_training_or_selection'])
        self.assertEqual(p['expected_phase_samples']['fit']['incident_full'], 1520)
        with tempfile.TemporaryDirectory() as root:
            changed = Path(root) / 'protocol.json'
            changed.write_text(e.PROTOCOL.read_text().replace('strictly improve', 'weakly improve'))
            with patch.object(e, 'PROTOCOL', changed), self.assertRaisesRegex(ValueError, 'Frozen v12k'):
                e.load_protocol()

    def test_selectors_disagree_on_early_harm_without_changing_global_guard(self):
        baseline, p = selection_metrics(), a.base.load_protocol()
        harmed_early = selection_metrics(9.9, 10.005)
        self.assertTrue(a.selection_decision(harmed_early, baseline, baseline, 'global', p)['replace_best'])
        self.assertFalse(a.selection_decision(harmed_early, baseline, baseline, 'candidate_early', p)['eligible'])
        for global_mae in (10., 10.001):
            current = selection_metrics(global_mae, 9.)
            self.assertFalse(a.selection_decision(current, baseline, baseline, 'candidate_early', p)['eligible'])

    def test_each_selector_ranks_its_own_objective_and_keeps_earlier_ties(self):
        baseline, p = selection_metrics(), a.base.load_protocol()
        first, second = selection_metrics(9.8, 9.9), selection_metrics(9.9, 9.5)
        self.assertFalse(a.selection_decision(second, baseline, first, 'global', p)['replace_best'])
        self.assertTrue(a.selection_decision(second, baseline, first, 'candidate_early', p)['replace_best'])
        for selector in a.SELECTORS:
            self.assertFalse(a.selection_decision(first, baseline, first, selector, p)['replace_best'])

    def test_control_late_harm_and_missing_metrics_reject_both_selectors(self):
        baseline, p = selection_metrics(), a.base.load_protocol()
        for invalid in (10.02, None, float('nan')):
            current = selection_metrics(9., 8.)
            current['secondary_control']['mae']['candidate_h7_h12'] = invalid
            for selector in a.SELECTORS:
                self.assertFalse(a.selection_decision(current, baseline, baseline, selector, p)['eligible'])

    def test_history_replay_preserves_two_distinct_best_epochs_and_fallback(self):
        baseline, p = selection_metrics(), a.base.load_protocol()
        best = dict.fromkeys(a.SELECTORS, baseline)
        epochs = dict.fromkeys(a.SELECTORS, 0)
        history = []
        for epoch, current in enumerate((selection_metrics(9.8, 9.9), selection_metrics(9.9, 9.5)), 1):
            decisions = {}
            for selector in a.SELECTORS:
                d = a.selection_decision(current, baseline, best[selector], selector, p)
                if d['replace_best']:
                    best[selector], epochs[selector] = current, epoch
                decisions[selector] = {**d, 'best_epoch': epochs[selector]}
            history.append({'epoch': epoch, 'selection': current, 'decisions': decisions})
        result = a.replay_selection(history, baseline, p)
        self.assertEqual({k: v['epoch'] for k, v in result.items()}, {'global': 1, 'candidate_early': 2})
        self.assertEqual(a.replay_selection([], baseline, p)['candidate_early']['epoch'], 0)
        history[-1]['decisions']['global']['best_epoch'] = 2
        with self.assertRaisesRegex(ValueError, 'selectors'):
            a.replay_selection(history, baseline, p)

    def test_early_loss_gradients_exclude_late_noncandidate_and_invalid_cells(self):
        prediction = torch.ones((2, 12, 4, 1), requires_grad=True)
        target = torch.zeros_like(prediction)
        candidate = torch.tensor([[1, 1, 0, 0], [1, 0, 0, 0]], dtype=torch.bool)
        valid = torch.ones_like(prediction, dtype=torch.bool)
        valid[0, 0, 0, 0] = False
        target[0, 0, 0, 0] = float('nan')
        loss, count = a.early_loss(prediction, target, valid, candidate)
        self.assertEqual(count, 17)
        self.assertEqual(float(loss), 1.)
        loss.backward()
        expected = torch.zeros_like(prediction)
        expected[:, :6] = candidate[:, None, :, None] / 17.
        expected[~valid] = 0.
        torch.testing.assert_close(prediction.grad, expected)
        with self.assertRaisesRegex(ValueError, 'no valid candidate'):
            a.early_loss(prediction, target, valid, torch.zeros_like(candidate))

    def test_both_trainers_update_only_the_vector_adapter(self):
        torch.set_num_threads(1)
        ds, p = SyntheticDataset(), a.base.load_protocol()
        settings = p['training']
        for trainer in (a.early_train_epoch, a.vector.vector_train_epoch):
            model = tiny_model()
            a.base.set_seed(2025)
            original = a.base.backbone_hash(model)
            adapter = a.vector.attach_adapter(model, 'state_vector')
            optimizer = torch.optim.Adam(adapter.parameters(), lr=settings['learning_rate'])
            result = trainer(model, adapter, optimizer, ds, [0, 1], settings, torch.device('cpu'), 2025, 1, lambda *a, **k: None)
            self.assertTrue(result['gate_parameters_changed'])
            self.assertEqual(result['optimizer_steps'], 1)
            a.base.assert_backbone(model, original)


class StatisticsTests(unittest.TestCase):
    def fixture(self):
        # Different candidate counts make cell/window weighting distinguishable.
        record = {'ids': np.arange(4), 'source_indices': np.arange(4), 'regions': list(a.REGIONS),
            'candidate_mask': np.ones((4, 2), dtype=bool),
            'counts': np.asarray([[12, 2, 2, 8], [12, 4, 4, 4], [12, 2, 2, 8], [12, 4, 4, 4]])}
        record['prediction_counts'] = record['counts'].copy()
        record['errors'] = record['counts'] * 10.
        variants = {}
        for arm in a.ARMS:
            for loss in a.LOSSES:
                for selector in a.SELECTORS:
                    r = copy.deepcopy(record)
                    benefit = (1 if loss == 'candidate_early' else 0) + (2 if selector == 'candidate_early' else 0)
                    raw_gain = np.asarray([1., 3., 1., 3.]) * benefit
                    r['errors'][:, 1] -= raw_gain * r['counts'][:, 1]
                    r['errors'][:, 0] -= raw_gain * r['counts'][:, 1]
                    variants[a.endpoint(arm, loss, selector)] = {'incident_full': r}
        times = dict(enumerate(['2023-01-02', '2023-01-09', '2023-01-16', '2023-01-23']))
        p = copy.deepcopy(a.base.load_protocol())
        p['bootstrap']['draws'] = 40
        return {'incident_full': record}, variants, times, p

    def test_factor_contrasts_shared_draws_and_region_contribution(self):
        reference, variants, times, p = self.fixture()
        result, rows, weekly = a.compare_phase(reference, variants, times, p)
        r = result['results']['incident_full']['regions']
        name = 'state_vector__loss_global__selector_effect'
        early = r['candidate_h1_h6']['comparisons'][name]
        self.assertAlmostEqual(early['gain_raw_mae'], 14 / 3)
        self.assertEqual(early['equal_forecast_window_gain_raw_mae'], 4.)
        self.assertAlmostEqual(early['regional_gain_in_global_mae_units'], r['all']['comparisons'][name]['gain_raw_mae'])
        context = r['candidate_h1_h6']['comparisons']['loss_global__select_global__context_effect']
        self.assertEqual(context['intervals']['week']['pooled']['ci_low'], 0.)
        self.assertEqual(context['intervals']['four_week_block']['pooled']['ci_high'], 0.)
        self.assertEqual(weekly['week_weights'].shape, (40, 4))
        self.assertEqual(len(rows), 20 * 4)

    def test_support_misalignment_is_rejected(self):
        ref, variants, times, p = self.fixture()
        next(iter(variants.values()))['incident_full']['candidate_mask'][0, 0] = False
        with self.assertRaisesRegex(ValueError, 'sample/support'):
            a.compare_phase(ref, variants, times, p)

    def test_empty_region_keeps_undefined_gain_and_ci(self):
        ref, variants, times, p = self.fixture()
        for record in [ref['incident_full']] + [v['incident_full'] for v in variants.values()]:
            for key in ('counts', 'prediction_counts', 'errors'):
                record[key][:, 2] = 0
        result, _, _ = a.compare_phase(ref, variants, times, p)
        effect = next(iter(result['results']['incident_full']['regions']['candidate_h7_h12']['comparisons'].values()))
        self.assertIsNone(effect['gain_raw_mae'])
        self.assertEqual(effect['intervals']['week']['pooled']['status'], 'INSUFFICIENT_VALID_DRAWS')


if __name__ == '__main__':
    unittest.main()
