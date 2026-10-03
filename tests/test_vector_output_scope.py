"""Output boundaries, shared gradients, policy replay and paired scope decomposition."""

import copy
import unittest

import numpy as np
import torch

from experiments.chronological import vector_output_scope as scope
from test_architecture_mechanisms import inputs, tiny_model


def metrics(global_mae=10., early_mae=10.):
    result = {cohort: {'mae': dict.fromkeys(scope.base.REGIONS, 10.)} for cohort in scope.base.COHORTS}
    result['incident_full']['mae'].update(all=global_mae, candidate_h1_h6=early_mae)
    return result


def make_history(candidates, baseline, protocol):
    best = {policy: baseline for policy in scope.POLICIES}
    epochs = dict.fromkeys(scope.POLICIES, 0)
    history = []
    for epoch, selection in enumerate(candidates, 1):
        decisions = {}
        for policy in scope.POLICIES:
            decision = scope.selection_decision(selection[policy], baseline, best[policy], protocol)
            if decision['replace_best']:
                best[policy], epochs[policy] = selection[policy], epoch
            decisions[policy] = {**decision, 'best_epoch': epochs[policy]}
        history.append({'epoch': epoch, 'selection': copy.deepcopy(selection), 'decisions': decisions})
    return history


class ProjectionTests(unittest.TestCase):
    def test_direct_choice_is_exact_even_when_residual_arithmetic_would_cancel(self):
        prediction = torch.ones((2, 12, 3, 1), dtype=torch.float32, requires_grad=True)
        baseline = torch.full_like(prediction, 1e20)
        candidate = torch.tensor([[True, False, True], [False, False, False]])
        projected = scope.project_prediction(prediction, baseline, candidate)
        mask = (candidate[:, None, :, None] & (torch.arange(12)[None, :, None, None] < 6)).expand_as(prediction)
        self.assertTrue(torch.equal(projected[mask], prediction[mask]))
        self.assertTrue(torch.equal(projected[~mask], baseline[~mask]))
        self.assertTrue(torch.equal(projected[1], baseline[1]))
        self.assertFalse(torch.equal((baseline + (prediction - baseline))[mask], prediction[mask]))
        projected.sum().backward()
        torch.testing.assert_close(prediction.grad, mask.to(prediction.dtype), atol=0, rtol=0)

    def test_invalid_shapes_nonfinite_baseline_gradient_and_non_node_masks_rejected(self):
        prediction, baseline = torch.ones(2, 12, 3, 1), torch.zeros(2, 12, 3, 1)
        candidate = torch.ones(2, 3, dtype=torch.bool)
        for bad_mask in (torch.ones_like(prediction, dtype=torch.bool), torch.ones(2, dtype=torch.bool),
                         candidate.float(), torch.ones(2, 1, 3, 1, dtype=torch.bool)):
            with self.assertRaisesRegex(ValueError, 'Boolean.*report-location'):
                scope.project_prediction(prediction, baseline, bad_mask)
        for bad_baseline in (baseline.double(), baseline[:, :, :2], baseline.requires_grad_()):
            with self.assertRaises(ValueError):
                scope.project_prediction(prediction, bad_baseline, candidate)
        baseline = baseline.detach()
        for source in ('prediction', 'baseline'):
            p, b = prediction.clone(), baseline.clone()
            (p if source == 'prediction' else b)[0, 11, 2, 0] = float('nan')
            with self.assertRaisesRegex(ValueError, 'Nonfinite'):
                scope.project_prediction(p, b, candidate)
        with self.assertRaisesRegex(ValueError, 'Prediction shape'):
            scope.project_prediction(prediction[:, :6], baseline[:, :6], candidate)
        with self.assertRaisesRegex(ValueError, 'floating point'):
            scope.project_prediction(prediction.long(), baseline.long(), candidate)

    def test_nonzero_real_adapters_have_identical_early_objective_and_gradients(self):
        torch.set_num_threads(1)
        x, incident = inputs()
        candidate = incident['distances'].abs().sum(-1) > 0
        for arm in scope.ARMS:
            model = tiny_model().eval()
            with torch.no_grad():
                native = model(x, incident_data=incident)
            original = scope.base.backbone_hash(model)
            adapter = scope.alignment.vector.attach_adapter(model, arm)
            with torch.no_grad():
                adapter.output.weight.fill_(0.08)
                adapter.output.bias.fill_(0.02)
            target = native + .23
            valid = torch.ones_like(target, dtype=torch.bool)
            valid[0, 0, 0, 0] = False
            target[~valid] = float('nan')
            losses, gradients = [], []
            for policy in scope.POLICIES:
                prediction = model(x, incident_data=incident)
                self.assertFalse(torch.equal(prediction, native))
                penalty = model.icsf_module.last_unit_residual[candidate].square().mean()
                if policy == 'candidate_early_only':
                    projected = scope.project_prediction(prediction, native, candidate)
                    # Invalid targets do not remove a report-time predicted candidate cell.
                    self.assertTrue(torch.equal(projected[~valid], prediction[~valid]))
                    prediction = projected
                loss, count = scope.alignment.early_loss(prediction, target, valid, candidate)
                objective = loss + .001 * penalty
                losses.append(objective.detach())
                gradients.append(torch.autograd.grad(objective, tuple(adapter.parameters())))
                self.assertGreater(count, 0)
            self.assertTrue(torch.equal(losses[0], losses[1]))
            self.assertGreater(sum(float(g.abs().sum()) for g in gradients[0]), 0.)
            for unbounded, protected in zip(*gradients):
                torch.testing.assert_close(unbounded, protected, atol=0, rtol=0)
            scope.base.assert_backbone(model, original)

    def test_projection_has_no_target_validity_or_cohort_argument(self):
        p, a, mask = torch.ones(2, 12, 3, 1), torch.zeros(2, 12, 3, 1), torch.ones(2, 3, dtype=torch.bool)
        for future in ({'y_valid': torch.zeros_like(p, dtype=torch.bool)}, {'cohort': 'incident'}):
            with self.assertRaises(TypeError):
                scope.project_prediction(p, a, mask, **future)


class PolicyReplayTests(unittest.TestCase):
    def setUp(self):
        self.protocol = scope.base.load_protocol()
        self.baseline = metrics()

    def test_policy_specific_rejection_best_epochs_and_earlier_ties(self):
        initial_u, first_p = metrics(9.99, 9.9), metrics(9.99, 9.9)
        initial_u['incident']['mae']['candidate_h7_h12'] = 10.02
        next_u, next_p = metrics(9.98, 9.95), metrics(9.98, 9.8)
        history = make_history([
            dict(zip(scope.POLICIES, (initial_u, first_p))),
            dict(zip(scope.POLICIES, (next_u, next_p))),
            dict(zip(scope.POLICIES, (next_u, next_p)))], self.baseline, self.protocol)
        restored = scope.replay_selection(history, self.baseline, self.protocol)
        self.assertFalse(history[0]['decisions']['unrestricted']['eligible'])
        self.assertTrue(history[0]['decisions']['candidate_early_only']['eligible'])
        self.assertEqual({p: r['epoch'] for p, r in restored.items()}, dict.fromkeys(scope.POLICIES, 2))
        self.assertFalse(history[-1]['decisions']['candidate_early_only']['replace_best'])
        history[1]['decisions']['candidate_early_only']['best_epoch'] = 1
        with self.assertRaisesRegex(ValueError, 'output-policy selectors'):
            scope.replay_selection(history, self.baseline, self.protocol)

    def test_different_best_epochs_and_zero_fallback_are_not_substituted(self):
        history = make_history([
            {'unrestricted': metrics(9.9, 9.7), 'candidate_early_only': metrics(9.99, 9.9)},
            {'unrestricted': metrics(9.8, 9.8), 'candidate_early_only': metrics(9.98, 9.8)}],
            self.baseline, self.protocol)
        selected = scope.replay_selection(history, self.baseline, self.protocol)
        self.assertEqual(selected['unrestricted']['epoch'], 1)
        self.assertEqual(selected['candidate_early_only']['epoch'], 2)
        empty = scope.replay_selection([], self.baseline, self.protocol)
        self.assertEqual({p: r['epoch'] for p, r in empty.items()}, dict.fromkeys(scope.POLICIES, 0))
        history = make_history([dict.fromkeys(scope.POLICIES, self.baseline)], self.baseline, self.protocol)
        self.assertEqual(scope.replay_selection(history, self.baseline, self.protocol), empty)

    def test_original_global_and_matched_early_guards_remain_required(self):
        for changed in ('global', 'incident_early', 'control_early'):
            current = metrics(9.99, 9.8)
            if changed == 'global':
                current['incident_full']['mae']['all'] = 10.
            else:
                cohort = 'incident' if changed == 'incident_early' else 'primary_control'
                current[cohort]['mae']['candidate_h1_h6'] = 10.02
            self.assertFalse(scope.selection_decision(current, self.baseline, self.baseline, self.protocol)['eligible'])

    def test_recovery_rejects_gaps_extra_policies_and_boolean_decisions(self):
        history = make_history([dict.fromkeys(scope.POLICIES, metrics(9.99, 9.9))], self.baseline, self.protocol)
        for mutate in ('gap', 'extra', 'boolean'):
            changed = copy.deepcopy(history)
            if mutate == 'gap':
                changed[0]['epoch'] = 2
            elif mutate == 'extra':
                changed[0]['selection']['unexpected'] = self.baseline
            else:
                changed[0]['decisions']['unrestricted']['eligible'] = 1
            with self.assertRaises(ValueError):
                scope.replay_selection(changed, self.baseline, self.protocol)


class PairedStatisticsTests(unittest.TestCase):
    def fixture(self):
        candidate = np.asarray([[1, 0, 0], [1, 1, 0], [1, 0, 0], [1, 1, 0]], dtype=bool)
        n = candidate.sum(1)
        counts = np.stack((np.full(4, 36), 6*n, 6*n, 12*(3-n)), 1)
        record = {'ids': np.arange(4), 'source_indices': np.arange(4), 'regions': list(scope.REGIONS),
            'candidate_mask': candidate, 'counts': counts, 'prediction_counts': counts.copy(),
            'errors': counts * 10.}
        variants = {}
        for arm in scope.ARMS:
            u, pu, pp = (copy.deepcopy(record) for _ in range(3))
            early_gain = np.asarray([1., 3., 1., 3.])
            u['errors'][:, 1] -= counts[:, 1] * early_gain
            u['errors'][:, 2] += counts[:, 2] * np.asarray([.1, .4, .2, .3])
            u['errors'][:, 3] -= counts[:, 3] * .2
            pu['errors'][:, 1] = u['errors'][:, 1]
            pp['errors'][:, 1] -= counts[:, 1] * (early_gain + np.asarray([1., -1., 2., -1.]))
            if arm == 'interaction_vector':
                pp['errors'][:, 1] -= counts[:, 1] * .1
            for path, value in zip(scope.OUTPUT_PATHS, (u, pu, pp)):
                value['errors'][:, 0] = value['errors'][:, 1:].sum(-1)
                variants[scope.endpoint(arm, path)] = {'incident_full': value}
        times = dict(enumerate(['2023-01-02', '2023-01-09', '2023-01-23', '2023-01-30']))
        protocol = copy.deepcopy(scope.base.load_protocol())
        protocol['bootstrap']['draws'] = 80
        return {'incident_full': record}, variants, times, protocol

    def test_paired_decomposition_direct_intervals_and_shared_empty_calendar_week(self):
        ref, variants, times, protocol = self.fixture()
        result, rows, weekly = scope.compare_phase(ref, variants, times, protocol)
        self.assertEqual(len(result['paths']), 7)  # A plus exactly six prespecified outputs.
        self.assertEqual(len(rows), 13 * 4)
        self.assertEqual(weekly['week_weights'].shape, (80, 5))
        self.assertEqual(weekly['incident_full_valid_counts'][2].sum(), 0)
        regions = result['results']['incident_full']['regions']
        for arm in scope.ARMS:
            for region in scope.REGIONS:
                effects = regions[region]['comparisons']
                for estimand in ('gain_raw_mae', 'equal_forecast_window_gain_raw_mae'):
                    self.assertAlmostEqual(effects[f'{arm}__total_policy_effect'][estimand],
                        effects[f'{arm}__output_scope_effect'][estimand] + effects[f'{arm}__selection_effect'][estimand])
            early = regions['candidate_h1_h6']['comparisons'][f'{arm}__output_scope_effect']
            self.assertEqual(early['gain_raw_mae'], 0.)
            self.assertEqual(early['intervals']['four_week_block']['equal_forecast_window']['ci_high'], 0.)
            for region in ('candidate_h7_h12', 'noncandidate_all'):
                self.assertEqual(regions[region]['comparisons'][f'{arm}__selection_effect']['gain_raw_mae'], 0.)
            protected = scope.endpoint(arm, 'protected_at_protected') + '_vs_A'
            self.assertAlmostEqual(regions['all']['comparisons'][protected]['gain_raw_mae'],
                regions['candidate_h1_h6']['comparisons'][protected]['regional_gain_in_global_mae_units'])
        early = regions['candidate_h1_h6']['comparisons'][scope.endpoint('state_vector', 'protected_at_unrestricted') + '_vs_A']
        self.assertNotEqual(early['gain_raw_mae'], early['equal_forecast_window_gain_raw_mae'])
        # Independently rebuild the direct total-policy interval using shared draws.
        name = 'state_vector__total_policy_effect'
        column = list(result['contrasts']).index(name)
        gains = weekly['incident_full_gain_sums'][:, column, 0]
        total = regions['all']['comparisons'][name]
        for method in ('week', 'four_week_block'):
            for weighting, numerator, denominator in (
                ('pooled', gains, weekly['incident_full_valid_counts'][:, 0]),
                ('equal_forecast_window', weekly['incident_full_window_gain_sums'][:, column, 0],
                 weekly['incident_full_evaluable_windows'][:, 0])):
                w = weekly[f'{method}_weights']
                direct = scope.regional.interval(scope.regional.ratio(w @ numerator, w @ denominator),
                    protocol['bootstrap']['confidence'], protocol['bootstrap']['minimum_valid_draw_fraction'])
                self.assertEqual(total['intervals'][method][weighting], direct)

    def test_misaligned_support_extra_up_path_or_unmatched_cohort_is_rejected(self):
        for problem in ('mask', 'count', 'ids', 'extra_path', 'cohort'):
            ref, variants, times, protocol = self.fixture()
            first = next(iter(variants.values()))
            if problem == 'mask':
                first['incident_full']['candidate_mask'][0, 0] = False
            elif problem == 'count':
                first['incident_full']['counts'][0, [0, 1]] -= 1
            elif problem == 'ids':
                first['incident_full']['ids'][0] = 99
            elif problem == 'extra_path':
                variants['state_vector__unrestricted_at_protected'] = copy.deepcopy(first)
            else:
                first['incident'] = copy.deepcopy(first['incident_full'])
            with self.assertRaises(ValueError):
                scope.compare_phase(ref, variants, times, protocol)

    def test_changed_early_or_protected_late_errors_fail_even_when_support_matches(self):
        for path, column in (('protected_at_unrestricted', 1), ('protected_at_unrestricted', 2),
                             ('protected_at_protected', 3)):
            ref, variants, times, protocol = self.fixture()
            errors = variants[scope.endpoint('state_vector', path)]['incident_full']['errors']
            errors[0, column] += .001
            errors[0, 0] += .001
            with self.assertRaisesRegex(ValueError, 'early errors|late/noncandidate errors'):
                scope.compare_phase(ref, variants, times, protocol)

    def test_empty_cohort_and_empty_target_region_remain_undefined(self):
        ref, variants, times, protocol = self.fixture()
        all_records = [ref['incident_full'], *[v['incident_full'] for v in variants.values()]]
        for record in all_records:
            for key in ('counts', 'errors'):
                record[key][:, 0] -= record[key][:, 2]
                record[key][:, 2] = 0
        empty = {key: (value if key == 'regions' else value[:0].copy()) for key, value in ref['incident_full'].items()}
        ref['incident'] = empty
        for value in variants.values():
            value['incident'] = copy.deepcopy(empty)
        result, _, weekly = scope.compare_phase(ref, variants, times, protocol)
        self.assertEqual(result['results']['incident']['forecast_windows'], 0)
        self.assertEqual(weekly['incident_valid_counts'].sum(), 0)
        for cohort, region in (('incident_full', 'candidate_h7_h12'), ('incident', 'all')):
            effect = next(iter(result['results'][cohort]['regions'][region]['comparisons'].values()))
            self.assertIsNone(effect['gain_raw_mae'])
            self.assertIsNone(effect['equal_forecast_window_gain_raw_mae'])
            self.assertEqual(effect['intervals']['week']['pooled']['status'], 'INSUFFICIENT_VALID_DRAWS')

    def test_corrupt_region_partitions_and_zero_count_errors_are_rejected(self):
        for problem in ('partition', 'zero_count_error'):
            ref, variants, times, protocol = self.fixture()
            if problem == 'partition':
                ref['incident_full']['errors'][0, 0] += 1.
            else:
                ref['incident_full']['counts'][0, 1] = 0
            with self.assertRaisesRegex(ValueError, 'partition|empty-region'):
                scope.compare_phase(ref, variants, times, protocol)

    def test_common_prediction_support_cannot_be_redefined_by_target_validity(self):
        ref, variants, times, protocol = self.fixture()
        for record in [ref['incident_full'], *[v['incident_full'] for v in variants.values()]]:
            # Even identical changes in every endpoint cannot turn target availability
            # into the report-time prediction mask.
            record['prediction_counts'][:, 1] += 1
            record['prediction_counts'][:, 0] += 1
        with self.assertRaisesRegex(ValueError, 'Prediction support'):
            scope.compare_phase(ref, variants, times, protocol)


if __name__ == '__main__':
    unittest.main()
