"""Local parity check against the unchanged Torch-based v12k selector implementation.

Kept outside the JSON-only server preflight pattern, which needs no third-party packages.
"""

import copy
import random
import unittest

from experiments.chronological import vector_objective_alignment as original
from experiments.chronological import vector_selection_audit as audit


class ReferenceParityTests(unittest.TestCase):
    def test_all_guards_tolerance_nulls_ties_and_fallback_match_original(self):
        protocol = original.base.load_protocol()
        spec = protocol['selection']
        baseline = {c: {'mae': dict.fromkeys(spec['protected_regions'], 10.)} for c in spec['protected_cohorts']}
        rng = random.Random(1212)
        for trial in range(8):
            history, best = [], {s: {'epoch': 0, 'metrics': baseline} for s in audit.SELECTORS}
            for epoch in range(1, 101):
                current = copy.deepcopy(baseline)
                full = current['incident_full']['mae']
                full['all'] = rng.choice((9.8, 9.9, 10., 10.005))
                full['candidate_h1_h6'] = rng.choice((9.7, 9.9, 10., 10.005))
                if epoch % 3:
                    cohort = rng.choice(spec['protected_cohorts'])
                    region = rng.choice(spec['protected_regions'])
                    current[cohort]['mae'][region] = rng.choice((None, 10.02, 10.*1.001+1e-12,
                                                               10.*1.001+2e-12))
                decisions = {}
                for selector in audit.SELECTORS:
                    d = original.selection_decision(current, baseline, best[selector]['metrics'], selector, protocol)
                    if d['replace_best']:
                        best[selector] = {'epoch': epoch, 'metrics': current}
                    decisions[selector] = {**d, 'best_epoch': best[selector]['epoch']}
                history.append({'epoch': epoch, 'selection': current, 'decisions': decisions})
            result = audit.audit_history(history, baseline, spec)
            self.assertEqual({s: r['selected_epoch'] for s, r in result['replayed_selectors'].items()},
                             {s: r['epoch'] for s, r in best.items()}, trial)
            self.assertEqual(original.replay_selection(history, baseline, protocol),
                {s: {'epoch': r['selected_epoch'], 'selection_metrics': r['selection_metrics']}
                 for s, r in result['replayed_selectors'].items()})


if __name__ == '__main__':
    unittest.main()
