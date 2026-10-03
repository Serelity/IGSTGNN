"""Complete guard counts, shared-trajectory denominators and recovered-history reporting."""

from contextlib import redirect_stdout
import copy
import io
import unittest

from experiments.chronological import train_vector_output_scope as e


class ReportingTests(unittest.TestCase):
    def setUp(self):
        self.protocol = e.base.load_protocol()
        self.guards = [f'{cohort}/{region}' for cohort in self.protocol['selection']['protected_cohorts']
                       for region in self.protocol['selection']['protected_regions']]

    def decision(self, failed=(), eligible=False, replace=False):
        guards = dict.fromkeys(self.guards, True)
        for key in failed:
            guards[key] = False
        return {'protection_checks': guards, 'eligible': eligible, 'replace_best': replace}

    def history(self):
        return [
            {'epoch': 1, 'decisions': {
                'unrestricted': self.decision(self.guards[:2]),
                'candidate_early_only': self.decision(eligible=True, replace=True)}},
            {'epoch': 2, 'decisions': {
                'unrestricted': self.decision(self.guards[:1]),
                'candidate_early_only': self.decision(self.guards[2:3])}},
            {'epoch': 3, 'decisions': {
                'unrestricted': self.decision(),
                'candidate_early_only': self.decision(eligible=True)}}]

    def counts(self, selected=None, history=None):
        return e.summarize_selection_history(self.history() if history is None else history, self.protocol,
            selected or {'unrestricted': 0, 'candidate_early_only': 1})

    def test_all_sixteen_guards_include_zeros_and_failures_overlap_per_epoch(self):
        result = self.counts()
        u, p = result['unrestricted'], result['candidate_early_only']
        self.assertEqual(len(u['protection_failure_counts']), 16)
        self.assertEqual(set(u['protection_failure_counts']), set(self.guards))
        self.assertEqual(u['protection_rejected_epochs'], 2)
        self.assertEqual(sum(u['protection_failure_counts'].values()), 3)
        self.assertEqual(u['protection_failure_counts'][self.guards[0]], 2)
        self.assertEqual(u['protection_failure_counts'][self.guards[-1]], 0)
        self.assertEqual((u['eligible_epochs'], u['best_updates'], u['fallback_to_A']), (0, 0, True))
        self.assertEqual((p['protection_rejected_epochs'], p['eligible_epochs'], p['best_updates']), (1, 2, 1))
        self.assertFalse(p['fallback_to_A'])

    def test_policy_and_architecture_fallback_denominators_exclude_derived_outputs(self):
        runs = {}
        for seed in (2025, 2026, 2027):
            runs[str(seed)] = {}
            for arm in e.scope.ARMS:
                selected = {'unrestricted': 0 if seed != 2027 else 1,
                            'candidate_early_only': 0 if arm == 'state_vector' and seed == 2025 else 1}
                runs[str(seed)][e.fit_name(arm, seed)] = {
                    'arm': arm, 'selection_accounting': self.counts(selected),
                    'output_paths': dict.fromkeys(e.scope.OUTPUT_PATHS, {})}
        result = e.aggregate_selection_accounting(runs, self.protocol)
        u, p = (result['by_policy'][policy] for policy in e.scope.POLICIES)
        self.assertEqual((u['total_endpoints'], u['fallback_endpoints'], u['fallback_rate']), (6, 4, 4 / 6))
        self.assertEqual((p['total_endpoints'], p['fallback_endpoints'], p['fallback_rate']), (6, 1, 1 / 6))
        self.assertEqual(u['trajectory_epochs'], 18)
        self.assertEqual(p['trajectory_epochs'], 18)
        self.assertEqual(u['protection_rejected_epochs'], 12)
        self.assertEqual(u['protection_failure_counts'][self.guards[-1]], 0)
        state = result['by_arm']['state_vector']['candidate_early_only']
        self.assertEqual((state['total_endpoints'], state['fallback_endpoints'], state['fallback_rate']), (3, 1, 1 / 3))
        self.assertEqual(result['by_arm']['interaction_vector']['candidate_early_only']['fallback_rate'], 0.)

    def test_complete_recovered_history_is_counted_instead_of_only_new_epochs(self):
        recovered = self.history()
        before = copy.deepcopy(recovered)
        completed = recovered + [{'epoch': 4, 'decisions': {
            'unrestricted': self.decision(eligible=True, replace=True),
            'candidate_early_only': self.decision(eligible=True, replace=True)}}]
        result = self.counts({'unrestricted': 4, 'candidate_early_only': 4}, completed)
        self.assertEqual(recovered, before)
        self.assertEqual(result['unrestricted']['trajectory_epochs'], 4)
        self.assertEqual(result['unrestricted']['protection_rejected_epochs'], 2)
        self.assertEqual(result['candidate_early_only']['eligible_epochs'], 3)
        self.assertEqual(result['candidate_early_only']['best_updates'], 2)
        self.assertFalse(result['unrestricted']['fallback_to_A'])
        self.assertEqual(self.counts(history=recovered), self.counts(history=copy.deepcopy(recovered)))

    def test_incomplete_guard_or_epoch_history_cannot_produce_misleading_counts(self):
        history = self.history()
        del history[0]['decisions']['unrestricted']['protection_checks'][self.guards[-1]]
        with self.assertRaisesRegex(ValueError, 'every Boolean protection'):
            self.counts(history=history)
        history = self.history()[1:]
        with self.assertRaisesRegex(ValueError, 'complete contiguous'):
            self.counts(history=history)

    def test_report_prints_freeze_flags_fallback_rates_and_per_policy_guards(self):
        phase_metrics = {phase: {'incident_full': {'mae': dict.fromkeys(e.scope.REGIONS, 1.)}}
                         for phase in ('fit', 'selection', 'audit')}
        detail = {'arm': 'state_vector', 'selection_accounting': self.counts(),
                  'policies': {policy: {'selected_epoch': 0 if policy == 'unrestricted' else 1,
                                       'phase_metrics': phase_metrics} for policy in e.scope.POLICIES}}
        runs = {'2025': {'state_vector_s2025': detail}}
        summary = {'status': 'ENGINEERING_CHECK_PASS', 'protocol_sha256': 'test',
                   'recommendation': 'ENGINEERING_ONLY', 'budget': {}, 'engineering_check': True,
                   'all_selectors_frozen_before_audit_evaluation': True,
                   'all_output_paths_frozen_before_audit_evaluation': True,
                   'selection_accounting': e.aggregate_selection_accounting(runs, self.protocol),
                   'runs': runs, 'baseline': phase_metrics, 'evaluation_runtime_this_invocation': {}}
        output = io.StringIO()
        with redirect_stdout(output):
            e.report(summary)
        text = output.getvalue()
        self.assertIn('All selected endpoints frozen before audit: True', text)
        self.assertIn('All output paths frozen before audit: True', text)
        self.assertIn('all_architectures unrestricted', text)
        self.assertIn('all_architectures candidate_early_only', text)
        self.assertIn('"fallback_rate": 1.0', text)
        self.assertIn('selection_counts:', text)
        self.assertIn(self.guards[-1] + '\": 0', text)


if __name__ == '__main__':
    unittest.main()
