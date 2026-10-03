"""Frozen guard boundaries, overlapping blockers and JSON-only source replay."""

from contextlib import redirect_stdout
import copy
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from experiments.chronological import audit_vector_selection as e

a = e.analysis


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def metrics(global_mae=10., early_mae=10.):
    p = json.loads(e.INHERITED_PROTOCOL.read_text())
    counts = json.loads(e.SOURCE_PROTOCOL.read_text())['expected_phase_samples']['selection']
    result = {c: {'samples': n, 'mae': dict.fromkeys(p['selection']['protected_regions'], 10.)} for c, n in counts.items()}
    result['incident_full']['mae'].update(all=global_mae, candidate_h1_h6=early_mae)
    return result


def history_fixture():
    values = [metrics(9.9, 9.8), metrics(9.8, 9.6), metrics(9.85, 9.7), metrics(9.7, 10.005),
              metrics(9.75, 9.5), metrics(9.7, 9.5), metrics(10., 9.), metrics(10.002, 10.001),
              metrics(9.4, 9.4), metrics(9.85, 9.7), metrics(9.75, 9.5), metrics()]
    values[1]['primary_control']['mae']['candidate_h7_h12'] = 10.03
    values[1]['secondary_control']['mae']['noncandidate_all'] = 10.04
    values[8]['primary_control']['mae']['all'] = None
    # Hand-specified outcomes, including ties, non-ranking eligibility and overlapping blockers.
    best_global = [1, 1, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4]
    best_early = [1, 1, 3, 3, 5, 5, 5, 5, 5, 5, 5, 5]
    result = []
    for index, value in enumerate(values):
        epoch = index + 1
        protected = epoch not in (2, 9)
        checks = {f'{c}/{r}': True for c in value for r in value[c]['mae']}
        if epoch == 2:
            checks['primary_control/candidate_h7_h12'] = False
            checks['secondary_control/noncandidate_all'] = False
        if epoch == 9:
            checks['primary_control/all'] = False
        decisions = {}
        for selector, best in (('global', best_global), ('candidate_early', best_early)):
            decisions[selector] = {'protected': protected, 'protection_checks': checks,
                'full_global_strictly_better_than_A': protected and epoch not in (7, 8, 12),
                'full_early_strictly_better_than_A': protected and epoch not in (4, 8, 12),
                'eligible': protected and epoch not in (7, 8, 12) and (selector == 'global' or epoch != 4),
                'replace_best': epoch == best[index], 'best_epoch': best[index]}
        result.append({'epoch': epoch, 'selection': value, 'decisions': decisions,
            'training': {'optimizer_steps': 190, 'loss_region': 'global'}, 'adapter_state_sha256': digest(str(epoch))})
    return result


def source_fixture(root):
    source = root / 'source'
    source.mkdir()
    protocol, inherited = e.load_protocol(), json.loads(e.INHERITED_PROTOCOL.read_text())
    frozen = json.loads(e.SOURCE_PROTOCOL.read_text())
    training = copy.deepcopy(inherited['training'])
    training['objective'] = 'v12k paired global/candidate_early objectives; unchanged vector energy penalty'
    code = {**protocol['producer_code_sha256'],
        str(e.SOURCE_PROTOCOL.relative_to(e.REPO)): protocol['source_protocol_sha256'],
        str(e.INHERITED_PROTOCOL.relative_to(e.REPO)): protocol['inherited_protocol_sha256']}
    summary = {'status': 'VECTOR_OBJECTIVE_ALIGNMENT_COMPARISON_COMPLETE',
        'protocol_id': frozen['protocol_id'], 'protocol_sha256': protocol['source_protocol_sha256'],
        'engineering_check': False, 'main_training_ready': False,
        'all_selectors_frozen_before_audit_evaluation': True, 'paired_initialization_exact': True,
        'model_training_performed': True, 'training_scope': 'icsf_vector_adapter_only',
        'recommendation': frozen['decision'], **frozen['information_boundary'],
        'frozen_protocol': frozen, 'inherited_protocol': inherited, 'effective_training': training,
        'phase_samples': frozen['expected_phase_samples'], 'code_sha256': code, 'inputs': {},
        'baseline': {'selection': metrics(), 'audit': 'NOT_FOR_THIS_ANALYSIS'},
        'environment': {'git_head': 'synthetic_fixture'}, 'runs': {}, 'outputs': {}}
    identity = {'protocol_sha256': summary['protocol_sha256'], 'engineering_check': False,
        'code_sha256': code, 'inputs': {}, 'effective_training': training, 'seeds': protocol['seeds'],
        'indices': {p: {c: list(range(n)) for c, n in cs.items()} for p, cs in summary['phase_samples'].items()}}
    e.write_json(source / 'run_identity.json', identity)
    endpoints = {}
    for seed in protocol['seeds']:
        summary['runs'][str(seed)] = {}
        for arm in protocol['arms']:
            for loss in protocol['losses']:
                name = f'{arm}__loss_{loss}_s{seed}'
                directory = source / name
                directory.mkdir()
                history = history_fixture()
                for row in history:
                    row['training']['loss_region'] = loss
                selectors = {}
                for selector, epoch in (('global', 4), ('candidate_early', 5)):
                    key = f'{name}/selected_{selector}.pt'
                    endpoints[key] = digest(key)
                    selectors[selector] = {'selected_epoch': epoch, 'selection_metrics': history[epoch - 1]['selection'],
                        'adapter_state_sha256': digest(str(epoch)), 'checkpoint_sha256': endpoints[key]}
                detail = {'arm': arm, 'loss': loss, 'seed': seed, 'epochs': 12, 'optimizer_steps': 2280,
                    'trainable_parameters': 4288, 'initial_adapter_sha256': digest(f'initial{seed}'),
                    'backbone_state_sha256': digest('backbone'), 'initial_prediction_exactly_A': True,
                    'backbone_state_unchanged': True, 'recovery': {'method': 'fresh_fit'}, 'selectors': selectors}
                e.write_json(directory / 'history.json', history)
                e.write_json(directory / 'fit_summary.json', detail)
                reported = copy.deepcopy(detail)
                for choice in reported['selectors'].values():
                    choice['phase_metrics'] = {'selection': choice['selection_metrics'], 'audit': 'NOT_FOR_THIS_ANALYSIS'}
                summary['runs'][str(seed)][name] = reported
    e.write_json(source / 'selected_endpoints_frozen.json', endpoints)
    def publish():
        summary['outputs'] = {**endpoints,
            **{str(p.relative_to(source)): e.sha256(p) for p in source.rglob('*.json') if p.name != 'summary.json'}}
        e.write_json(source / 'summary.json', summary)
    publish()
    return source, summary, publish


class AccountingTests(unittest.TestCase):
    def setUp(self):
        self.spec = json.loads(e.INHERITED_PROTOCOL.read_text())['selection']

    def test_raw_gains_survive_protection_failure_and_replay_preserves_ties(self):
        r = a.audit_history(history_fixture(), metrics(), self.spec)
        self.assertEqual({s: v['selected_epoch'] for s, v in r['replayed_selectors'].items()}, {'global': 4, 'candidate_early': 5})
        self.assertTrue(r['epochs'][1]['joint_positive_protection_rejected'])
        self.assertEqual(len(r['epochs'][1]['failed_checks']), 2)
        d = [v for v in r['decisions'] if v['epoch'] == 6]
        self.assertTrue(all(v['reason'] == 'eligible_not_better_than_incumbent' for v in d))
        self.assertFalse(history_fixture()[1]['decisions']['global']['full_early_strictly_better_than_A'])

    def test_overlapping_counts_have_epoch_denominators_and_complete_reason_partition(self):
        r = a.audit_history(history_fixture(), metrics(), self.spec)
        totals = a.accounting(r['epochs'], r['checks'], r['decisions'], self.spec)
        self.assertEqual(totals['trajectory_epochs'], 12)
        self.assertEqual(totals['selector_decisions'], 24)
        self.assertEqual(totals['protection_rejected_epochs'], 2)
        self.assertEqual(totals['joint_positive_protection_rejected_epochs'], 2)
        failures = totals['protection_failure_counts']
        self.assertEqual(sum(v['failed_epochs'] for v in failures.values()), 3)
        self.assertEqual(failures['primary_control/all']['joint_positive_sole_failed_protection_epochs'], 1)
        self.assertEqual(failures['primary_control/candidate_h7_h12']['sole_failed_protection_epochs'], 0)
        pair = next(v for v in totals['cofailures'] if v['epochs'])
        self.assertEqual(pair['epochs'], 1)
        for counts in totals['selector_reason_counts'].values():
            self.assertEqual(sum(counts.values()), 12)

    def test_exact_tolerance_zero_anchor_and_null_support(self):
        current, baseline = metrics(), metrics()
        bound = 10. * 1.001 + 1e-12
        current['primary_control']['mae']['all'] = bound
        rows = a.metric_rows(current, baseline, self.spec)
        row = next(r for r in rows if r['check'] == 'primary_control/all')
        self.assertTrue(row['passed'])
        self.assertEqual(row['excess_over_decision_limit'], 0.)
        current['primary_control']['mae']['all'] = bound + 1e-11
        self.assertFalse(next(r for r in a.metric_rows(current, baseline, self.spec) if r['check'] == 'primary_control/all')['passed'])
        baseline['primary_control']['mae']['all'] = 0.
        current['primary_control']['mae']['all'] = 0.
        zero = next(r for r in a.metric_rows(current, baseline, self.spec) if r['check'] == 'primary_control/all')
        self.assertTrue(zero['passed'])
        self.assertIsNone(zero['relative_harm_percent'])
        current['primary_control']['mae']['all'] = None
        invalid = next(r for r in a.metric_rows(current, baseline, self.spec) if r['check'] == 'primary_control/all')
        self.assertFalse(invalid['passed'])
        self.assertIsNone(invalid['excess_over_decision_limit'])

    def test_history_decision_corruption_and_boolean_substitution_fail(self):
        for field, value in (('eligible', 1), ('best_epoch', 12), ('replace_best', False)):
            history = history_fixture()
            history[0]['decisions']['global'][field] = value
            with self.assertRaisesRegex(ValueError, 'Stored decision mismatch'):
                a.audit_history(history, metrics(), self.spec)
        history = history_fixture()
        history[1]['epoch'] = 4
        with self.assertRaisesRegex(ValueError, 'Noncontiguous'):
            a.audit_history(history, metrics(), self.spec)


class SourceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source, self.summary, self.publish = source_fixture(self.root)

    def audit(self, name='result'):
        with redirect_stdout(io.StringIO()):
            return e.run(self.source, self.root / name)

    def test_complete_workflow_preserves_source_and_opens_only_its_json(self):
        for name in ('unused.pt', 'train.npy', 'val.npy', 'test.npz'):
            (self.source / name).write_bytes(b'NEVER READ')
        before = {str(p): e.sha256(p) for p in self.source.rglob('*') if p.is_file()}
        opened, real_open = [], Path.open
        def guard(path, *args, **kwargs):
            if path.is_relative_to(self.source):
                self.assertEqual(path.suffix, '.json')
                opened.append(str(path))
            return real_open(path, *args, **kwargs)
        with patch.object(Path, 'open', guard):
            result = self.audit()
        self.assertEqual(before, {str(p): e.sha256(p) for p in self.source.rglob('*') if p.is_file()})
        self.assertEqual(result['status'], 'VECTOR_SELECTION_FAILURE_AUDIT_COMPLETE')
        self.assertEqual(len(result['source_json_sha256']), 27)
        self.assertEqual(len(set(opened)), 27)
        self.assertEqual(result['totals']['trajectory_epochs'], 144)
        self.assertEqual(result['totals']['selector_decisions'], 288)
        self.assertEqual(len(result['joint_positive_rejections']), 24)
        self.assertTrue(result['all_final_selection_references_match'])
        self.assertFalse(result['source_audit_metrics_used_for_analysis'])
        self.assertFalse(result['checkpoint_files_read'])
        self.assertFalse((self.root / 'result.partial').exists())
        for name, digest_value in result['outputs'].items():
            self.assertEqual(e.sha256(self.root / 'result' / name), digest_value)
        capture = io.StringIO()
        with redirect_stdout(capture):
            e.report(result)
        self.assertIn('primary_control/candidate_h7_h12', capture.getvalue())
        self.assertIn('excess_over_limit=', capture.getvalue())

    def test_epoch_zero_fallback_reconciles_to_initial_state(self):
        name = 'state_vector__loss_global_s2025'
        history = history_fixture()
        baseline = metrics()
        checks = {f'{c}/{r}': True for c in baseline for r in baseline[c]['mae']}
        for row in history:
            row['selection'] = copy.deepcopy(baseline)
            row['decisions'] = {s: {'protected': True, 'protection_checks': checks,
                'full_global_strictly_better_than_A': False, 'full_early_strictly_better_than_A': False,
                'eligible': False, 'replace_best': False, 'best_epoch': 0} for s in a.SELECTORS}
        e.write_json(self.source / name / 'history.json', history)
        reported = self.summary['runs']['2025'][name]
        for choice in reported['selectors'].values():
            choice.update(selected_epoch=0, selection_metrics=baseline, adapter_state_sha256=reported['initial_adapter_sha256'])
            choice['phase_metrics']['selection'] = baseline
        detail = copy.deepcopy(reported)
        for choice in detail['selectors'].values():
            del choice['phase_metrics']
        e.write_json(self.source / name / 'fit_summary.json', detail)
        self.publish()
        result = self.audit()
        self.assertEqual(result['totals']['fallback_endpoints'], 2)

    def test_unhashed_and_rehashed_decision_tampering_both_fail(self):
        path = self.source / 'state_vector__loss_global_s2025/history.json'
        history = json.loads(path.read_text())
        history[0]['decisions']['global']['best_epoch'] = 0
        e.write_json(path, history)
        with self.assertRaisesRegex(ValueError, 'artifact hash mismatch'):
            self.audit('unhashed')
        self.publish()
        with self.assertRaisesRegex(ValueError, 'Stored decision mismatch'):
            self.audit('rehashed')
        self.assertFalse((self.root / 'rehashed').exists())
        self.assertTrue((self.root / 'rehashed.partial/failure.json').is_file())

    def test_changed_final_metrics_and_state_references_are_rejected(self):
        name = 'state_vector__loss_global_s2025'
        detail_path = self.source / name / 'fit_summary.json'
        original = copy.deepcopy(self.summary['runs']['2025'][name])
        for index, field in enumerate(('state', 'metrics', 'epoch')):
            reported = copy.deepcopy(original)
            selected = reported['selectors']['global']
            if field == 'state':
                selected['adapter_state_sha256'] = digest('wrong_state')
            elif field == 'metrics':
                selected['selection_metrics']['incident_full']['mae']['all'] = 9.2
                selected['phase_metrics']['selection'] = selected['selection_metrics']
            else:
                selected['selected_epoch'] = 3
                selected['adapter_state_sha256'] = digest('3')
            self.summary['runs']['2025'][name] = reported
            detail = copy.deepcopy(reported)
            for choice in detail['selectors'].values():
                del choice['phase_metrics']
            e.write_json(detail_path, detail)
            self.publish()
            with self.assertRaisesRegex(ValueError, 'state reference|Final selector replay'):
                self.audit(f'bad_{index}')

    def test_incomplete_source_protocol_code_and_identity_are_rejected(self):
        pristine = copy.deepcopy(self.summary)
        changes = [('engineering_check', True), ('protocol_sha256', '0'*64),
                   ('all_selectors_frozen_before_audit_evaluation', False)]
        for index, (key, value) in enumerate(changes):
            self.summary.clear()
            self.summary.update(copy.deepcopy(pristine))
            self.summary[key] = value
            self.publish()
            with self.assertRaisesRegex(ValueError, 'complete frozen'):
                self.audit(f'bad_source_{index}')
        self.summary.clear()
        self.summary.update(copy.deepcopy(pristine))
        self.summary['code_sha256'][next(iter(e.load_protocol()['producer_code_sha256']))] = '0'*64
        self.publish()
        with self.assertRaisesRegex(ValueError, 'Producer code'):
            self.audit('bad_code')

    def test_null_metric_rejects_but_missing_or_negative_metric_is_corruption(self):
        path = self.source / 'state_vector__loss_global_s2025/history.json'
        original = json.loads(path.read_text())
        for index, value in enumerate((-1., '10.0')):
            history = copy.deepcopy(original)
            history[0]['selection']['primary_control']['mae']['all'] = value
            e.write_json(path, history)
            self.publish()
            with self.assertRaisesRegex(ValueError, 'Selection MAE'):
                self.audit(f'bad_metric_{index}')
        history = copy.deepcopy(original)
        del history[0]['selection']['primary_control']['mae']['all']
        e.write_json(path, history)
        self.publish()
        with self.assertRaisesRegex(ValueError, 'Missing selection metric'):
            self.audit('missing_metric')

    def test_missing_epoch_wrong_step_budget_and_sample_count_are_rejected(self):
        path = self.source / 'state_vector__loss_global_s2025/history.json'
        pristine = json.loads(path.read_text())
        for index, change in enumerate(('missing_epoch', 'steps', 'samples')):
            history = copy.deepcopy(pristine)
            if change == 'missing_epoch':
                history.pop()
            elif change == 'steps':
                history[0]['training']['optimizer_steps'] = 189
            else:
                history[0]['selection']['incident']['samples'] = 205
            e.write_json(path, history)
            self.publish()
            with self.assertRaisesRegex(ValueError, 'Incomplete epoch|budget/loss|sample count'):
                self.audit(f'budget_{index}')

    def test_rehashed_run_identity_and_endpoint_manifest_mismatches_fail(self):
        path = self.source / 'run_identity.json'
        pristine = json.loads(path.read_text())
        identity = copy.deepcopy(pristine)
        identity['seeds'] = [2025]
        e.write_json(path, identity)
        self.publish()
        with self.assertRaisesRegex(ValueError, 'run identity mismatch'):
            self.audit('identity')
        e.write_json(path, pristine)
        manifest = self.source / 'selected_endpoints_frozen.json'
        endpoints = json.loads(manifest.read_text())
        endpoints[next(iter(endpoints))] = digest('changed_checkpoint_reference')
        e.write_json(manifest, endpoints)
        self.publish()
        with self.assertRaisesRegex(ValueError, 'Frozen checkpoint hash references'):
            self.audit('checkpoint_reference')

    def test_prior_audit_outcomes_do_not_change_selection_accounting(self):
        first = self.audit('before_audit_change')
        self.summary['baseline']['audit'] = {'all': 999999., 'candidate_h1_h6': 0.}
        for fits in self.summary['runs'].values():
            for detail in fits.values():
                for selected in detail['selectors'].values():
                    selected['phase_metrics']['audit'] = {'all': 0., 'candidate_h1_h6': 999999.}
        self.publish()
        second = self.audit('after_audit_change')
        self.assertEqual(first['totals'], second['totals'])
        self.assertEqual(first['fits'], second['fits'])
        self.assertEqual(first['joint_positive_rejections'], second['joint_positive_rejections'])
        self.assertEqual(first['outputs'], second['outputs'])

    def test_source_mutation_during_analysis_never_publishes_complete_result(self):
        original = e.Source.recheck
        def changed(reader):
            path = reader.root / 'run_identity.json'
            path.write_text(path.read_text() + '\n')
            original(reader)
        with patch.object(e.Source, 'recheck', changed), self.assertRaisesRegex(ValueError, 'Source changed'):
            self.audit()
        self.assertFalse((self.root / 'result').exists())
        self.assertTrue((self.root / 'result.partial/failure.json').exists())

    def test_overlap_existing_outputs_and_symlinked_inputs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'separate'):
            e.run(self.source, self.source / 'nested')
        for suffix in ('', '.partial'):
            path = self.root / ('existing' + suffix)
            path.mkdir()
            with self.assertRaises(FileExistsError):
                e.run(self.source, self.root / 'existing')
            path.rmdir()
            path.symlink_to(self.root / 'missing')
            with self.assertRaisesRegex(ValueError, 'Symlink'):
                e.run(self.source, self.root / 'existing')
            path.unlink()
        path = self.source / 'state_vector__loss_global_s2025/history.json'
        moved = self.root / 'moved.json'
        path.rename(moved)
        path.symlink_to(moved)
        with self.assertRaisesRegex(ValueError, 'Symlink'):
            self.audit('symlink')

    def test_standard_library_only_cli_and_protocol_integrity(self):
        result = subprocess.run([sys.executable, '-S', str(Path(e.__file__)), 'run',
            '--source-dir', str(self.source), '--output', str(self.root / 'nosite')],
            text=True, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('VECTOR_SELECTION_FAILURE_AUDIT_COMPLETE', result.stdout)
        changed = self.root / 'changed_protocol.json'
        changed.write_text(e.PROTOCOL.read_text().replace('0.001', '0.1'))
        with patch.object(e, 'PROTOCOL', changed), self.assertRaisesRegex(ValueError, 'Frozen v12l'):
            e.load_protocol()
        for payload in ('{"a": 1, "a": 2}', '{"a": NaN}'):
            with self.assertRaises(ValueError):
                e.decode(payload)


if __name__ == '__main__':
    unittest.main()
