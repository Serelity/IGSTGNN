"""Server-only scoring and stage-gate checks; no source model is fitted here."""
from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from experiments.trajectory_check_gw14 import run as runner


def small_data(samples=2, nodes=2):
    y = np.broadcast_to(np.where(np.arange(nodes) % 2, 400., 100.)[None, None, :],
                        (samples, 12, nodes)).copy()
    return dict(pred=y+10., y=y, valid=np.ones(y.shape, dtype=np.bool_),
        sample_ids=np.arange(samples, dtype=np.int64),
        features=dict(last_flow=np.full((samples, nodes), 100., dtype=np.float32),
            mean_abs_change=np.ones((samples, nodes), dtype=np.float32),
            neighbor_change_disagreement=np.zeros((samples, nodes), dtype=np.float32),
            signed_direction=np.zeros((samples, nodes), dtype=np.float32),
            shape=(np.arange(samples*nodes).reshape(samples, nodes) % 4).astype(np.int8),
            turn_position=np.resize(np.array([-1, 2, 5, 7], dtype=np.int8),
                                    samples*nodes).reshape(samples, nodes)))


def metric_record(count, mae):
    return dict(valid_count=count, mae=mae, rmse=mae, mape_fraction=mae/100., signed_bias=mae)


def decision_fixture(common_count=25, total_count=100):
    errors = dict(OWN4=6., CONTEXT=5., MAGNITUDE16=4., DIRECTION16=3., TRAJECTORY16=2.)
    records = {arm:metric_record(common_count, value) if common_count else runner._empty()
               for arm,value in errors.items()}
    groups = {arm:{name:metric_record(1, 2.) for name in runner.GROUPS} for arm in runner.ARMS}
    return dict(records=records, full_native_mask_records=deepcopy(records), groups=groups,
        coverage=dict(common_valid_count=common_count, total_valid_count=total_count,
                      common_fraction=common_count/total_count),
        comparisons={}, cross=None, match_exact=True)


def cross_fixture(phase):
    return {name:[dict(phase=phase, fixture='bounded')] for name in
            ('cross_rows', 'phase_lead_rows', 'block_rows')}


class MatchedEvaluationTests(unittest.TestCase):
    def test_compact_cross_views_keep_contributions_and_none_population_separate(self):
        data = small_data()
        predictions = {arm:data['y']+error for arm,error in
            (('OWN4',4.),('MAGNITUDE16',3.),('DIRECTION16',2.),('TRAJECTORY16',1.))}
        summary = runner.cross.summarize(predictions,data,phase='validation')
        rows = runner.compact_cross(summary,'validation')
        shape_rows = [row for row in rows if row['view']=='shape_turn']
        none_rows = [row for row in rows if row['view']=='none_flow']
        self.assertEqual(sum(row['n'] for row in shape_rows),48)
        self.assertAlmostEqual(sum(row['contribution_vs_direction'] for row in shape_rows),1.)
        self.assertAlmostEqual(sum(row['contribution_vs_own4'] for row in shape_rows),3.)
        self.assertEqual(len(none_rows),1)
        self.assertEqual(none_rows[0]['n'],12)
        self.assertEqual(none_rows[0]['fraction_of_all_valid'],.25)
        self.assertEqual(none_rows[0]['trajectory_vs_direction_mae_gain'],1.)
        self.assertEqual(none_rows[0]['exact_constant_history_fraction'],0.)

    def _evaluate(self, fitted, data, common, predictions, **coverage_updates):
        coverage = dict(common_valid_count=int(common.sum()),
                        total_valid_count=int(data['valid'].sum()),
                        common_fraction=float(common.sum()/data['valid'].sum()))
        coverage.update(coverage_updates)
        with ExitStack() as stack:
            validate = stack.enter_context(patch.object(runner.matching, 'validate_model'))
            masks = stack.enter_context(patch.object(runner.matching, 'score_masks',
                                                    return_value=(common, coverage)))
            predict = stack.enter_context(patch.object(runner.matching, 'predictions',
                                                      return_value=iter(predictions.items())))
            summarize = stack.enter_context(patch.object(runner.cross, 'summarize',
                                                        return_value=cross_fixture('audit')))
            result = runner.evaluate_matched(fitted, data, 'audit')
        validate.assert_called_once_with(fitted)
        masks.assert_called_once_with(fitted, data['features'], data['valid'])
        predict.assert_called_once_with(fitted, data)
        return result, summarize

    def test_every_arm_uses_identical_common_points_and_guards_use_their_intersections(self):
        data = small_data()
        data['valid'][0, 1, 0] = False
        native_before = data['valid'].copy()
        common = np.zeros(data['valid'].shape, dtype=np.bool_)
        common[0, 0, 0] = True
        common[1, 6, 1] = True
        errors = dict(OWN4=9., CONTEXT=7., MAGNITUDE16=5., DIRECTION16=4., TRAJECTORY16=1.)
        predictions = {}
        for arm, error in errors.items():
            # Huge candidate errors outside common support must not leak into
            # its matched score or be silently removed from its native score.
            prediction = data['y'] + (1000. if arm == 'TRAJECTORY16' else 0.)
            prediction[common] = data['y'][common]+error
            predictions[arm] = prediction
        result, summarize = self._evaluate(dict(match_exact=True), data, common, predictions)
        for arm, error in errors.items():
            self.assertEqual(result['records'][arm]['valid_count'], 2)
            self.assertEqual(result['records'][arm]['mae'], error)
            self.assertEqual(result['full_native_mask_records'][arm]['valid_count'], 47)
            for name in runner.GROUPS:
                self.assertEqual(result['groups'][arm][name]['valid_count'], 1)
                self.assertEqual(result['groups'][arm][name]['mae'], error)
        self.assertGreater(result['full_native_mask_records']['TRAJECTORY16']['mae'], 900.)
        for arm in runner.ARMS[:-1]:
            self.assertEqual(result['comparisons'][arm]['valid_count'], 2)
            self.assertEqual(result['comparisons'][arm]['mae_gain'], errors[arm]-1.)
        summarize.assert_called_once()
        summarized_predictions, summarized_data = summarize.call_args.args
        self.assertEqual(set(summarized_predictions), set(runner.cross.ARMS))
        self.assertIs(summarized_data['valid'], common)
        self.assertIs(summarized_data['y'], data['y'])
        self.assertEqual(summarize.call_args.kwargs, dict(phase='audit'))
        np.testing.assert_array_equal(data['valid'], native_before)

    def test_common_mask_cannot_include_native_invalid_points_or_change_axes_or_dtype(self):
        data = small_data()
        data['valid'][0, 0, 0] = False
        valid_common = data['valid'].copy()
        predictions = {arm:data['pred'] for arm in runner.ARMS}
        invalid_masks = [np.ones(valid_common.shape, dtype=np.bool_),
                         valid_common[:1], valid_common.astype(np.int8)]
        for common in invalid_masks:
            with self.subTest(shape=common.shape, dtype=str(common.dtype)):
                with self.assertRaisesRegex(ValueError, 'common subset'):
                    self._evaluate(dict(match_exact=True), data, common, predictions)

    def test_misreported_common_coverage_fails_before_scoring(self):
        data = small_data()
        common = data['valid'].copy()
        predictions = {arm:data['pred'] for arm in runner.ARMS}
        for changed in (dict(common_valid_count=47), dict(total_valid_count=49),
                        dict(common_fraction=.5), dict(common_fraction=float('nan'))):
            with self.subTest(changed=changed):
                with self.assertRaisesRegex(ValueError, 'common scoring population'):
                    self._evaluate(dict(match_exact=True), data, common, predictions, **changed)

    def test_empty_common_support_has_no_scores_no_cross_and_no_permission_to_continue(self):
        data = small_data()
        common = np.zeros(data['valid'].shape, dtype=np.bool_)
        predictions = {arm:data['pred'] for arm in runner.ARMS}
        result, summarize = self._evaluate(dict(match_exact=True), data, common, predictions)
        summarize.assert_not_called()
        self.assertIsNone(result['cross'])
        for arm in runner.ARMS:
            self.assertEqual(result['records'][arm]['valid_count'], 0)
            self.assertIsNone(result['records'][arm]['mae'])
            for name in runner.GROUPS:
                self.assertEqual(result['groups'][arm][name]['valid_count'], 0)
                self.assertIsNone(result['groups'][arm][name]['mae'])
        for row in result['comparisons'].values():
            self.assertEqual(row['valid_count'], 0)
            self.assertIsNone(row['mae_gain'])
        decision = runner.matched_decision(result)
        self.assertEqual(decision['interpretation'], 'inconclusive_common_support')
        self.assertFalse(decision['matched_information_advantage_observed'])
        self.assertFalse(decision['new_matched_development_scoring_permitted'])

    def test_missing_or_reordered_arm_does_not_silently_change_the_comparison(self):
        data = small_data()
        predictions = {arm:data['pred'] for arm in reversed(runner.ARMS)}
        with self.assertRaisesRegex(ValueError, 'arms or order'):
            self._evaluate(dict(match_exact=True), data, data['valid'], predictions)
        predictions.pop('DIRECTION16')
        with self.assertRaisesRegex(ValueError, 'arms or order'):
            self._evaluate(dict(match_exact=True), data, data['valid'], predictions)


class MatchedDecisionTests(unittest.TestCase):
    def test_quarter_coverage_is_inclusive_but_less_than_quarter_does_not_continue(self):
        at_boundary = runner.matched_decision(decision_fixture())
        self.assertTrue(at_boundary['common_support_sufficient'])
        self.assertTrue(at_boundary['matched_information_advantage_observed'])
        self.assertTrue(at_boundary['new_matched_development_scoring_permitted'])
        below_boundary = runner.matched_decision(decision_fixture(common_count=24))
        self.assertFalse(below_boundary['common_support_sufficient'])
        self.assertFalse(below_boundary['new_matched_development_scoring_permitted'])

    def test_tie_with_any_control_or_unequal_training_budget_blocks_new_development(self):
        for control in ('CONTEXT', 'MAGNITUDE16', 'DIRECTION16'):
            result = decision_fixture()
            result['records'][control] = metric_record(25, 2.)
            with self.subTest(control=control):
                decision = runner.matched_decision(result)
                self.assertFalse(decision['checks']['candidate_beats_'+control])
                self.assertFalse(decision['new_matched_development_scoring_permitted'])
        result = decision_fixture()
        result['match_exact'] = False
        decision = runner.matched_decision(result)
        self.assertFalse(decision['checks']['exact_training_budget_invariants'])
        self.assertFalse(decision['new_matched_development_scoring_permitted'])

    def test_empty_high_value_guard_is_unavailable_and_cannot_pass_development(self):
        result = decision_fixture()
        passed = runner.matched_decision(result, development=True)
        self.assertTrue(passed['matched_information_advantage_observed'])
        self.assertFalse(passed['new_matched_development_scoring_permitted'])
        high = 'high_value_ge314'
        for arm in runner.ARMS:
            result['groups'][arm][high] = runner._empty()
        stopped = runner.matched_decision(result, development=True)
        self.assertFalse(stopped['checks'][high+'_available_and_no_worse_than_OWN4_on_common_mask'])
        self.assertFalse(stopped['matched_information_advantage_observed'])
        self.assertFalse(stopped['new_matched_development_scoring_permitted'])

    def test_each_guard_uses_same_count_and_allows_equality_but_not_harm(self):
        for name in runner.GROUPS:
            for mismatch in ('worse', 'different_count'):
                result = decision_fixture()
                if mismatch == 'worse':
                    result['groups']['TRAJECTORY16'][name] = metric_record(1, 2.01)
                else:
                    result['groups']['TRAJECTORY16'][name] = metric_record(2, 2.)
                with self.subTest(group=name, mismatch=mismatch):
                    if mismatch == 'different_count':
                        with self.assertRaisesRegex(ValueError, 'guard mask counts differ'):
                            runner.matched_decision(result, development=True)
                    else:
                        decision = runner.matched_decision(result, development=True)
                        self.assertFalse(decision['checks'][name+'_available_and_no_worse_than_OWN4_on_common_mask'])
                        self.assertFalse(decision['matched_information_advantage_observed'])

    def test_positive_and_negative_diagnostics_never_promote_failed_gw13_or_authorize_training(self):
        for result, development in ((decision_fixture(), False),
                                    (decision_fixture(), True),
                                    (decision_fixture(common_count=24), False)):
            with self.subTest(development=development, coverage=result['coverage']['common_fraction']):
                decision = runner.matched_decision(result, development=development)
                self.assertTrue(decision['original_GW13_failed_screen_unchanged'])
                for name in ('original_GW13_screen_passed', 'candidate_for_confirmation',
                             'architecture_training_authorized', 'independent_confirmation_authorized',
                             'statistical_significance_claimed', 'test_evaluated', 'retuning_authorized'):
                    self.assertIs(decision[name], False)

    def test_metrics_must_cover_the_same_points_and_be_finite(self):
        for name, value in (('valid_count', 24), ('valid_count', 25.5),
                            ('valid_count', True), ('mae', float('nan')),
                            ('rmse', float('inf')), ('mape_fraction', -.1),
                            ('signed_bias', True)):
            result = decision_fixture()
            result['records']['DIRECTION16'][name] = value
            with self.subTest(field=name, value=value):
                with self.assertRaises(ValueError):
                    runner.matched_decision(result)

    def test_coverage_cannot_be_forged_with_coercible_counts_or_nonfinite_fractions(self):
        invalid = [('common_valid_count', True), ('common_valid_count', 25.5),
                   ('common_valid_count', -1), ('total_valid_count', True),
                   ('total_valid_count', 100.5), ('total_valid_count', 0),
                   ('total_valid_count', 24), ('common_fraction', True),
                   ('common_fraction', float('nan')), ('common_fraction', float('inf')),
                   ('common_fraction', -.1), ('common_fraction', 1.1),
                   ('common_fraction', .5)]
        for name, value in invalid:
            result = decision_fixture()
            result['coverage'][name] = value
            with self.subTest(field=name, value=value):
                with self.assertRaises(ValueError):
                    runner.matched_decision(result)


class TrajectoryCheckRunTests(unittest.TestCase):
    def _mock_failed_audit_run(self, directory, source_changed=False):
        root = Path(directory)
        paths = {name:root/name for name in ('gw13', 'probe', 'cache', 'data', 'output')}
        for name in ('gw13', 'probe', 'cache', 'data'):
            paths[name].mkdir()
        cached = dict(train=small_data(runner.old_run.TRAIN_COUNT, nodes=1),
                      val=small_data(2, nodes=1))
        certificate = dict(data_identity={'authenticated': True})
        cache_digests = {'authenticated_cache': 'b'*64}
        extra = {split:{name:value['features'][name] for name in
                         ('shape', 'signed_direction', 'turn_position')}
                 for split,value in cached.items()}
        history_meta = dict(file_digests={'authenticated_history': 'c'*64})
        parents = dict(audit=dict(source_stage='audit'), validation=dict(source_stage='validation'))
        datasets = dict(audit=runner.old_run.slice_data(cached['train'],
                           runner.old_run.SCORE_START, runner.old_run.TRAIN_COUNT),
                        validation=cached['val'])
        reference = dict(models=parents, datasets=datasets, digests={'immutable_gw13': 'e'*64})
        fitted = dict(match_exact=True, contexts=[dict(context=0, retained_cells_per_arm=1)],
                      fit_stage_counts=dict(child_tables_refitted=3,per_arm={}))
        audit_result = decision_fixture(common_count=int(datasets['audit']['valid'].sum())//4,
                                        total_count=int(datasets['audit']['valid'].sum()))
        # Adequate common support does not rescue a candidate tied with DIR.
        audit_result['records']['DIRECTION16']['mae'] = audit_result['records']['TRAJECTORY16']['mae']
        audit_result['cross'] = cross_fixture('audit')

        def predictions(selected_model, selected_data):
            phase = selected_model['source_stage']
            self.assertIs(selected_data, datasets[phase])
            for arm in runner.ARMS:
                yield arm, selected_data['pred']

        def summarize(selected_predictions, selected_data, *, phase):
            self.assertIs(selected_data, datasets[phase])
            self.assertEqual(set(selected_predictions), set(runner.cross.ARMS))
            return cross_fixture(phase)

        def save_model(value, path):
            self.assertIs(value, fitted)
            Path(path).write_text(json.dumps(value)+'\n', encoding='utf-8')

        with ExitStack() as stack:
            stack.enter_context(patch.object(runner, '_export_repo_from_cache', return_value=root))
            authenticated_run = stack.enter_context(patch.object(runner.auth, 'load_authenticated_run',
                return_value=(object(), object(), certificate)))
            stack.enter_context(patch.object(runner.auth, 'require_output_outside_sources'))
            stack.enter_context(patch.object(runner.auth, 'sha256', return_value='d'*64))
            stack.enter_context(patch.object(runner.io, 'load_cache',
                return_value=(cached, {}, cache_digests)))
            stack.enter_context(patch.object(runner.features, 'load_features',
                return_value=(extra, history_meta)))
            authenticate = stack.enter_context(patch.object(runner.source, 'authenticate', return_value=reference))
            old_fit = stack.enter_context(patch.object(runner.old_model, 'fit_model',
                side_effect=AssertionError('GW14 must replay saved GW13 tables')))
            old_predict = stack.enter_context(patch.object(runner.old_model, 'predictions', side_effect=predictions))
            raw_summary = stack.enter_context(patch.object(runner.cross, 'summarize', side_effect=summarize))
            fit = stack.enter_context(patch.object(runner.matching, 'fit_matched_model', return_value=fitted))
            stack.enter_context(patch.object(runner.matching, 'save_model', side_effect=save_model))
            evaluate = stack.enter_context(patch.object(runner, 'evaluate_matched', return_value=audit_result))
            final_source_guard = stack.enter_context(patch.object(runner.source, 'assert_unchanged',
                side_effect=ValueError('immutable GW13 source changed') if source_changed else None))
            final_probe_guard = stack.enter_context(patch.object(runner.auth, 'assert_unchanged'))
            final_execution_guard = stack.enter_context(patch.object(runner.io, 'rehash'))
            args = (paths['gw13'], paths['probe'], paths['cache'], paths['data'], paths['output'])
            if source_changed:
                with self.assertRaisesRegex(ValueError, 'immutable GW13 source changed'):
                    runner.run(*args)
                result = None
            else:
                result = runner.run(*args)

            authenticated_run.assert_called_once()
            authenticate.assert_called_once_with(paths['gw13'].resolve(), certificate,
                                                  cache_digests, cached, history_meta)
            old_fit.assert_not_called()
            self.assertEqual(old_predict.call_count, 2)
            self.assertEqual(raw_summary.call_count, 2)
            self.assertEqual([call.kwargs['phase'] for call in raw_summary.call_args_list],
                             ['audit', 'validation'])
            fit.assert_called_once()
            self.assertIs(fit.call_args.args[0], parents['audit'])
            fitted_data = fit.call_args.args[1]
            np.testing.assert_array_equal(fitted_data['sample_ids'], np.arange(runner.old_run.FIT_STOP))
            evaluate.assert_called_once_with(fitted, datasets['audit'], 'audit')
            final_source_guard.assert_called_once_with(reference)
            self.assertFalse((paths['output']/'matched_full_model.json').exists())
            self.assertFalse((paths['output']/'full_matching_cells.csv').exists())
            self.assertFalse((paths['output']/'development_checks.json').exists())
            self.assertFalse((paths['output']/'development_fitted_before_score.json').exists())
            for phase in ('audit', 'validation'):
                for suffix in ('cross', 'phase_lead', 'blocks'):
                    self.assertTrue((paths['output']/('raw_'+phase+'_'+suffix+'.csv')).is_file())
            if source_changed:
                final_probe_guard.assert_not_called()
                final_execution_guard.assert_not_called()
                self.assertFalse((paths['output']/'checks.json').exists())
                self.assertFalse((paths['output']/'analysis.json').exists())
                self.assertFalse((paths['output']/'matching_summary.csv').exists())
                self.assertFalse((paths['output']/'report.md').exists())
                failure = json.loads((paths['output']/'failure.json').read_text(encoding='utf-8'))
                self.assertEqual(failure['status'], 'GW14_CHECKS_FAILED')
                self.assertEqual(failure['error'], 'immutable GW13 source changed')
                self.assertFalse(failure['source_artifacts_written'])
            else:
                final_probe_guard.assert_called_once_with(certificate)
                final_execution_guard.assert_called_once()
                self.assertIn(str(paths['output'].resolve()/'matched_audit_model.json'),
                              final_execution_guard.call_args.args[0])
                checks = json.loads((paths['output']/'checks.json').read_text(encoding='utf-8'))
                self.assertEqual(checks['status'], 'GW14_CHECKS_COMPLETE')
                self.assertEqual(checks['new_statistical_lookup_fits'], 3)
                self.assertFalse(checks['new_matched_development_scored'])
                self.assertTrue(checks['historical_source_replay_passed'])
                self.assertTrue(checks['original_GW13_failed_screen_unchanged'])
                for name in ('original_GW13_screen_passed', 'candidate_for_confirmation',
                             'architecture_training_authorized', 'independent_confirmation_authorized',
                             'statistical_significance_claimed', 'test_evaluated'):
                    self.assertIs(checks[name], False)
                self.assertTrue((paths['output']/'matched_audit_model.json').is_file())
                self.assertTrue((paths['output']/'matching_summary.csv').is_file())
                self.assertTrue((paths['output']/'report.md').is_file())
            return result

    def test_failed_matched_audit_keeps_both_raw_replays_and_skips_new_development_fit_and_score(self):
        with TemporaryDirectory() as directory:
            result = self._mock_failed_audit_run(directory)
        self.assertEqual(set(result['raw']), {'audit', 'validation'})
        self.assertEqual(set(result['matched']), {'audit'})
        self.assertEqual(result['new_statistical_lookup_fits'], 3)
        self.assertEqual(result['original_models_or_cuts_refitted'], 0)
        self.assertFalse(result['decisions']['audit']['new_matched_development_scoring_permitted'])
        self.assertFalse(result['test_evaluated'])

    def test_source_drift_aborts_before_any_completed_result_is_published(self):
        with TemporaryDirectory() as directory:
            self._mock_failed_audit_run(directory, source_changed=True)


if __name__ == '__main__':
    unittest.main()
