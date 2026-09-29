"""Saved-result diagnosis must not turn fallback or regional accounting into claims."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experiments.chronological import diagnose_incident_strength_gate as d


def metrics(gain=0., gate=1.01):
    return {'mae': {r: 10. - gain for r in d.REGIONS},
            'candidate_gate': {'mean': gate, 'std': .02, 'q05_q50_q95': [.98, gate, 1.05]}}


def fixture_history(protocol, gains, bad_control=False):
    history, best, best_gain = [], 0, 0.
    selected = {c: metrics(0.) for c in d.COHORTS}
    for epoch, gain in enumerate(gains, 1):
        selection = {c: metrics(gain) for c in d.COHORTS}
        if bad_control:
            selection['primary_control'] = metrics(-.1)
        checks = {f'{c}/{r}': m['mae'][r] <= 10. * 1.001 + 1e-12
                  for c, m in selection.items() for r in d.REGIONS}
        eligible = all(checks.values())
        if eligible and gain > best_gain:
            best, best_gain, selected = epoch, gain, selection
        history.append({'epoch': epoch, 'eligible': eligible, 'protection_checks': checks,
                        'best_epoch': best, 'selection': selection,
                        'training': {'mae_standardized': .15 - .001 * epoch,
                                     'maximum_gradient_norm': .005 * epoch,
                                     'gate_parameters_changed': True, 'optimizer_steps': 1}})
    detail = {'selected_epoch': best, 'selection_metrics': selected,
              'optimizer_steps': len(gains), 'initial_prediction_exactly_A': True,
              'backbone_state_unchanged': True}
    return history, detail


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'source'
        self.source.mkdir()
        self.protocol = json.loads(d.PROTOCOL.read_text())
        self.baseline = {c: {'mae': dict.fromkeys(d.REGIONS, 10.),
                            'valid_cell_fraction': dict(zip(d.REGIONS, [1., .2, .3, .5]))}
                         for c in d.COHORTS}

    def analyze(self, gains, bad_control=False):
        h, detail = fixture_history(self.protocol, gains, bad_control)
        return d.analyze_history(h, detail, self.baseline, self.protocol)

    def make_source(self):
        s = {'status': 'ICSF_STRENGTH_GATE_EXPERIMENT_COMPLETE', 'engineering_check': False,
             'protocol_sha256': d.PROTOCOL_SHA256, 'frozen_protocol': self.protocol,
             'effective_training': self.protocol['training'], 'environment': {'git_head': 'fixture'},
             'phase_samples': {'fit': {'incident_full': 1}, 'selection': dict.fromkeys(d.COHORTS, 1)},
             'outputs': {}, 'runs': {}}
        for c in d.COHORTS:
            np.savez(self.source / f'selection_A_{c}.npz', regions=d.REGIONS,
                     counts=np.array([[10, 2, 3, 5]]), errors=np.array([[100., 20., 30., 50.]]))
        for seed in self.protocol['seeds']:
            s['runs'][str(seed)] = {}
            for variant in self.protocol['variants']:
                gains = [-.001] * 12
                if variant == 'node' and seed == 2027:
                    gains[10] = .003
                history, detail = fixture_history(self.protocol, gains)
                directory = self.source / f'{variant}_s{seed}'
                directory.mkdir()
                (directory / 'history.json').write_text(json.dumps(history))
                s['runs'][str(seed)][variant] = detail
        for p in self.source.rglob('*'):
            if p.is_file():
                s['outputs'][str(p.relative_to(self.source))] = d.digest(p.read_bytes())
        (self.source / 'summary.json').write_text(json.dumps(s))
        return s

    def test_fallback_is_not_mistaken_for_no_training_or_constraint_failure(self):
        result, rows = self.analyze([-.001] * 12)
        self.assertEqual(result['selection_reason'], 'NO_EPOCH_IMPROVED_SELECTION_GLOBAL_MAE')
        self.assertEqual(result['parameter_change_epochs'], 12)
        self.assertEqual(result['eligible_epochs'], 12)
        self.assertIsNone(result['selected_epoch_selection'])
        self.assertAlmostEqual(rows[0]['selection_gate_rms_distance_from_one'], np.hypot(.02, .01))

    def test_real_protection_failure_is_distinguished(self):
        result, _ = self.analyze([.001] * 12, bad_control=True)
        self.assertEqual(result['selection_reason'], 'IMPROVING_EPOCHS_FAILED_PROTECTION')
        self.assertEqual(result['improving_but_rejected_epochs'], list(range(1, 13)))

    def test_replay_selects_earlier_tie_and_last_epoch_is_not_best(self):
        gains = [-.001] * 12
        gains[3] = gains[7] = .005
        result, rows = self.analyze(gains)
        self.assertEqual(result['selected_epoch'], 4)
        self.assertEqual(result['selected_epoch_selection']['epoch'], 4)
        for row in rows:
            self.assertAlmostEqual(sum(row[f'{r}_global_contribution'] for r in d.PARTS),
                                   row['selection_global_gain_raw_mae'])

    def test_inconsistent_best_or_protection_or_nonfinite_gradient_rejected(self):
        for change in ('best', 'protection', 'gradient'):
            history, detail = fixture_history(self.protocol, [-.001] * 12)
            if change == 'best':
                history[0]['best_epoch'] = 1
            elif change == 'protection':
                history[0]['eligible'] = False
            else:
                history[0]['training']['maximum_gradient_norm'] = float('nan')
            with self.assertRaises(ValueError):
                d.analyze_history(history, detail, self.baseline, self.protocol)

    def test_nonpartitioning_region_mae_is_rejected(self):
        history, detail = fixture_history(self.protocol, [-.001] * 12)
        history[0]['selection']['incident_full']['mae']['candidate_h1_h6'] = 1.
        with self.assertRaisesRegex(ValueError, 'contributions'):
            d.analyze_history(history, detail, self.baseline, self.protocol)

    def test_end_to_end_preserves_source_and_needs_no_audit_or_checkpoint_files(self):
        self.make_source()
        before = {str(p): d.digest(p.read_bytes()) for p in self.source.rglob('*') if p.is_file()}
        with redirect_stdout(io.StringIO()):
            result = d.diagnose(self.source, self.root / 'diagnostic')
        self.assertEqual(result['runs']['node_s2027']['selected_epoch'], 11)
        self.assertEqual(len(result['input_sha256']), 11)
        self.assertFalse(result['audit_arrays_read'])
        self.assertFalse(result['checkpoint_loaded'])
        self.assertEqual(len((self.root / 'diagnostic/epochs.csv').read_text().splitlines()), 289)
        self.assertEqual(before, {str(p): d.digest(p.read_bytes()) for p in self.source.rglob('*') if p.is_file()})
        with self.assertRaises(FileExistsError):
            d.diagnose(self.source, self.root / 'diagnostic')
        with self.assertRaises(ValueError):
            d.diagnose(self.source, self.source / 'diagnostic')

    def test_tampered_input_and_engineering_only_result_rejected(self):
        s = self.make_source()
        history = self.source / 'scalar_s2025/history.json'
        history.write_text(history.read_text() + '\n')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            d.diagnose(self.source, self.root / 'diagnostic')
        s['engineering_check'] = True
        (self.source / 'summary.json').write_text(json.dumps(s))
        with self.assertRaisesRegex(ValueError, 'complete full-budget'):
            d.diagnose(self.source, self.root / 'other')

    def test_reads_actual_trainer_output_schema_with_synthetic_model_and_data(self):
        import torch
        from experiments.chronological import train_incident_strength_gate as trainer
        from test_incident_strength_gate import SyntheticDataset, manifests, tiny_model

        dataset, model = SyntheticDataset(), tiny_model()
        checkpoint = self.root / 'A.pt'
        torch.save(model.state_dict(), checkpoint)
        baseline = {'checkpoint': {'parameters': sum(p.numel() for p in model.parameters())}}
        output = self.root / 'trainer_result'
        with patch.object(trainer.mechanisms, 'verify_inputs', return_value=(baseline, {})), patch.object(
                trainer, 'read_csv', side_effect=list(manifests())), patch.object(
                trainer, 'make_datasets', return_value=dict.fromkeys(trainer.COHORTS, dataset)), patch.object(
                trainer, 'make_model', side_effect=lambda *a, **k: tiny_model()), patch.object(
                trainer, 'audit_comparisons', return_value=({}, {}, {})), patch.object(
                trainer, 'report'), redirect_stdout(io.StringIO()):
            trainer.run(self.root, self.root, self.root, checkpoint, output, 'cpu')
            result = d.diagnose(output, self.root / 'diagnostic')
        self.assertEqual(len(result['runs']), 6)
        self.assertTrue(all(r['epochs'] == 12 for r in result['runs'].values()))


if __name__ == '__main__':
    unittest.main()
