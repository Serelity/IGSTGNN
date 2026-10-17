"""No validation fit, explicit information channels, and selective controls."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.chronological import unified_adapter_decision_v03 as decision
from experiments.chronological import unified_adapter_risk_v03 as risk
from tests.test_unified_adapter_evaluation_v03 import archive


def fixture():
    cal, val = archive(8, 100), archive(5, 200)
    for data in (cal, val):
        shape = data['y'].shape
        group = np.broadcast_to(np.arange(len(data['y']))[:, None, None] % 2, shape)
        data['pred__H_ONLY'] = np.full(shape, 15.)
        data['pred__BOTH'] = data['pred__H_ONLY']+np.where(group, 2., -2.)
        data['feature__history'] = np.zeros(shape)
        data['feature__relation'] = group.astype(float)
        data['feature__category'] = np.stack([group, 1-group], -1).astype(int)
    manifest = dict(history=dict(source='history', available_at_cutoff=True, information_channel='H'),
                    relation=dict(source='report', available_at_cutoff=True, information_channel='R'),
                    category=dict(source='report', available_at_cutoff=True, information_channel='C', categorical_cardinalities=[3, 3]))
    common = dict(producer_status='PASS', sample_identity_verified=True, eligible_for_gain_fit=True, candidate_frozen_before_calibration=True,
                  calibration_used_for_candidate_fit=False, calibration_used_for_candidate_selection=False,
                  test_evaluated=False, candidate_checkpoint_sha256='candidate', base_checkpoint_sha256='base',
                  candidate_training_sample_ids=[1, 2, 3], base_training_sample_ids=[1, 2, 3],
                  prediction_case_roles=dict(H_ONLY=dict(checkpoint_sha256='base', epoch=1, selection='best'),
                                             BOTH=dict(checkpoint_sha256='candidate', epoch=1, selection='best')),
                  feature_manifest=manifest)
    return cal, val, dict(common, split='calibration'), dict(common, split='validation')


class GainHeadTests(unittest.TestCase):
    def test_producer_must_finish_before_auxiliary_fit(self):
        cal, val, cm, vm = fixture()
        cm['producer_status'] = 'REPLAY_IN_PROGRESS'
        with self.assertRaisesRegex(ValueError, 'Producer'):
            decision.fit_gain_head(cal, val, cm, vm)
        with self.assertRaisesRegex(ValueError, 'Producer'):
            risk.fit_risk_head(cal, val, cm, vm, steps=1)

    def test_group_claim_checks_fallback_training_and_is_not_independent_certification(self):
        cal, val, cm, vm = fixture()
        for metadata in (cm, vm):
            metadata.update(group_identity_verified=True, candidate_training_group_ids=['train'],
                            base_training_group_ids=['train'], calibration_group_ids=['cal'], validation_group_ids=['val'])
        model = decision.fit_gain_head(cal, val, cm, vm)
        self.assertTrue(model['identity']['declared_groups_disjoint'])
        self.assertFalse(model['identity']['independent_events_certified'])
        for metadata in (cm, vm):
            metadata['base_training_group_ids'] = ['cal']
        with self.assertRaisesRegex(ValueError, 'base training groups'):
            decision.fit_gain_head(cal, val, cm, vm)
        for metadata in (cm, vm):
            metadata.pop('base_training_group_ids')
        with self.assertRaisesRegex(ValueError, 'candidate/base training'):
            decision.fit_gain_head(cal, val, cm, vm)

    def test_concrete_role_hash_cannot_bind_other_case_or_changed_role(self):
        cal, val, cm, vm = fixture()
        with self.assertRaisesRegex(ValueError, 'binds the fallback'):
            decision.fit_gain_head(cal, val, cm, vm, base_case='TRAFFIC_CORE')
        model = decision.fit_gain_head(cal, val, cm, vm)
        changed = deepcopy(vm)
        changed['prediction_case_roles']['H_ONLY']['checkpoint_sha256'] = 'candidate'
        with self.assertRaisesRegex(ValueError, 'H_ONLY role checkpoint'):
            decision.predict_gain_head(model, val, changed)
        with self.assertRaisesRegex(ValueError, 'H_ONLY role checkpoint'):
            decision.fit_gain_head(cal, val, cm, changed)
        changed = deepcopy(vm)
        changed['prediction_case_roles']['BOTH']['selection'] = 'different'
        with self.assertRaisesRegex(ValueError, 'roles changed'):
            decision.predict_gain_head(model, val, changed)

    def test_file_backed_holdout_must_match_published_archive_bytes(self):
        cal, val, cm, vm = fixture()
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            cp, vp = Path(tmp)/'cal.npz', Path(tmp)/'val.npz'
            np.savez_compressed(cp, **cal)
            np.savez_compressed(vp, **val)
            for path, metadata in ((cp, cm), (vp, vm)):
                metadata['archive_sha256'] = sha256(path.read_bytes()).hexdigest()
            model = decision.fit_gain_head(cp, vp, cm, vm)
            decision.predict_gain_head(model, vp, vm)
            vm['archive_sha256'] = 'another-archive'
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                decision.predict_gain_head(model, vp, vm)
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                risk.fit_risk_head(cp, vp, cm, vm, steps=1)

    def test_fit_validation_truth_changes_do_not_change_gain_model(self):
        cal, val, cm, vm = fixture()
        one = decision.fit_gain_head(cal, val, cm, vm)
        val['y'] += 100
        two = decision.fit_gain_head(cal, val, cm, vm)
        self.assertEqual(one, two)
        self.assertFalse(one['validation_labels_used_for_fitting'])
        self.assertFalse(one['identity']['whole_pipeline_oof'])
        self.assertIn('category:0=2', one['feature_names'])

    def test_strict_holdout_identity_and_unknown_provenance(self):
        for key, bad in [('sample_identity_verified', False), ('split', 'validation'),
                         ('candidate_frozen_before_calibration', False), ('eligible_for_gain_fit', False),
                         ('calibration_used_for_candidate_selection', True), ('calibration_used_for_candidate_fit', True),
                         ('candidate_training_sample_ids', [100, 1]), ('base_training_sample_ids', [100, 1])]:
            cal, val, cm, vm = fixture()
            cm[key] = bad
            with self.subTest(key=key), self.assertRaises(ValueError):
                decision.fit_gain_head(cal, val, cm, vm)
        cal, val, cm, vm = fixture()
        val['sample_ids'][0] = cal['sample_ids'][0]
        with self.assertRaisesRegex(ValueError, 'disjoint'):
            decision.fit_gain_head(cal, val, cm, vm)

    def test_checkpoint_change_requires_refit_and_future_features_refused(self):
        cal, val, cm, vm = fixture()
        model = decision.fit_gain_head(cal, val, cm, vm)
        changed = dict(vm, candidate_checkpoint_sha256='new')
        with self.assertRaisesRegex(ValueError, 'checkpoint'):
            decision.predict_gain_head(model, val, changed)
        for metadata in (cm, vm):
            metadata['feature_manifest']['category']['source'] = 'future_target'
        with self.assertRaisesRegex(ValueError, 'cutoff|source'):
            decision.fit_gain_head(cal, val, cm, vm)

    def test_category_ids_not_ordinal_and_illegal_ids_refused(self):
        cal, val, cm, vm = fixture()
        model = decision.fit_gain_head(cal, val, cm, vm)
        self.assertNotIn('category:0', model['feature_names'])
        val['feature__category'] = val['feature__category'].astype(float)
        val['feature__category'][0, 0, 0, 0] = 1.5
        with self.assertRaisesRegex(ValueError, 'integers'):
            decision.predict_gain_head(model, val, vm)

    def test_gain_predictor_does_not_read_prediction_targets(self):
        cal, val, cm, vm = fixture()
        model = decision.fit_gain_head(cal, val, cm, vm)
        first = decision.predict_gain_head(model, val, vm)
        val['y'] += 999
        second = decision.predict_gain_head(model, val, vm)
        np.testing.assert_array_equal(first, second)

    def test_policy_random_matches_valid_acceptance_and_oracle_labeled(self):
        cal, val, cm, vm = fixture()
        model = decision.fit_gain_head(cal, val, cm, vm)
        gain = decision.predict_gain_head(model, val, vm)
        gains = np.where(np.indices(gain.shape)[0] % 2, 1., -1.)
        scores, predictions = decision.evaluate_policies(val['pred__H_ONLY'], val['pred__BOTH'], val['y'], val['valid'], gains)
        self.assertEqual(scores['LEARNED']['acceptance_fraction'], scores['RANDOM_MATCHED']['acceptance_fraction'])
        self.assertFalse(scores['ORACLE_UNDEPLOYABLE']['deployable'])
        self.assertFalse(scores['RANDOM_MATCHED']['deployable'])
        np.testing.assert_array_equal(predictions['HALF_CORRECTION'], (val['pred__H_ONLY']+val['pred__BOTH'])/2)
        self.assertLessEqual(scores['ORACLE_UNDEPLOYABLE']['mae_pooled'], scores['ALWAYS_BASE']['mae_pooled'])

    def test_completed_decision_analysis_preserves_sources(self):
        cal, val, cm, vm = fixture()
        before = deepcopy(val)
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            report = decision.run_decision_analysis(cal, val, cm, vm, Path(tmp)/'out')
            self.assertFalse(report['test_evaluated'])
            self.assertTrue((Path(tmp)/'out'/'decision_predictions.npz').is_file())
            analysis = json.loads((Path(tmp)/'out'/'analysis'/'analysis.json').read_text(encoding='utf-8'))
            self.assertIn('LEARNED_vs_ALWAYS_BASE', analysis['comparisons'])
            self.assertIn('LEARNED_vs_HALF_CORRECTION', analysis['comparisons'])
            self.assertTrue((Path(tmp)/'out'/'analysis'/'groups.csv').is_file())
            self.assertEqual(report['runtime']['deployment_required_source_model_forwards'], 2)
            self.assertFalse(report['runtime']['complete_deployment_latency_measured'])
            self.assertGreaterEqual(report['runtime']['fit']['cpu_process_seconds'], 0.)
            self.assertEqual(report['runtime']['predict']['processed_points'], val['y'].size)
        for key in val:
            np.testing.assert_array_equal(val[key], before[key])


class RiskHeadTests(unittest.TestCase):
    def test_h_r_rc_read_declared_channels_and_same_frozen_center(self):
        cal, val, cm, vm = fixture()
        models = {mode: risk.fit_risk_head(cal, val, cm, vm, mode, steps=20, batch_size=64) for mode in risk.MODES}
        self.assertFalse(any('relation' in label or 'category' in label for label in models['H']['feature_names']))
        self.assertTrue(any('relation' in label for label in models['H_R']['feature_names']))
        self.assertFalse(any('category' in label for label in models['H_R']['feature_names']))
        self.assertTrue(any('category:0=2' == label for label in models['H_R_C']['feature_names']))
        for model in models.values():
            quantile = risk.predict_risk_head(model, val, vm)
            np.testing.assert_array_equal(quantile[..., 1], val['pred__H_ONLY'])
            self.assertTrue((np.diff(quantile, axis=-1) > 0).all())

    def test_risk_fit_and_prediction_do_not_use_validation_truth(self):
        cal, val, cm, vm = fixture()
        first = risk.fit_risk_head(cal, val, cm, vm, 'H_R_C', steps=20, batch_size=64)
        q1 = risk.predict_risk_head(first, val, vm)
        val['y'] *= 5
        second = risk.fit_risk_head(cal, val, cm, vm, 'H_R_C', steps=20, batch_size=64)
        self.assertEqual(first, second)
        np.testing.assert_array_equal(q1, risk.predict_risk_head(second, val, vm))

    def test_risk_absent_channel_and_mislabeled_report_refused(self):
        cal, val, cm, vm = fixture()
        cal.pop('feature__category')
        val.pop('feature__category')
        with self.assertRaisesRegex(ValueError, 'absent'):
            risk.fit_risk_head(cal, val, cm, vm, 'H_R_C', steps=2)
        for metadata in (cm, vm):
            metadata['feature_manifest']['relation']['information_channel'] = 'H'
        with self.assertRaisesRegex(ValueError, 'relabeled'):
            risk.fit_risk_head(cal, val, cm, vm, 'H', steps=2)

    def test_risk_prediction_is_refitted_after_center_change(self):
        cal, val, cm, vm = fixture()
        model = risk.fit_risk_head(cal, val, cm, vm, steps=2)
        with self.assertRaisesRegex(ValueError, 'checkpoint|refitted'):
            risk.predict_risk_head(model, val, dict(vm, base_checkpoint_sha256='new'))

    def test_conditional_width_learns_real_heteroscedastic_residuals(self):
        cal, val, cm, vm = fixture()
        for data in (cal, val):
            group = data['feature__relation']
            sign = np.where(np.indices(data['y'].shape)[1] % 2, 1., -1.)
            data['y'] = data['pred__H_ONLY']+sign*np.where(group, 6., 1.)
        model = risk.fit_risk_head(cal, val, cm, vm, 'H_R', steps=250, batch_size=192)
        q = risk.predict_risk_head(model, val, vm)
        width = q[..., 2]-q[..., 0]
        group = val['feature__relation'].astype(bool)
        self.assertGreater(width[group].mean(), width[~group].mean()+3.)

    def test_ordinal_semantic_regression_is_rejected(self):
        cal, val, cm, vm = fixture()
        for metadata in (cm, vm):
            metadata['feature_manifest']['category'].pop('categorical_cardinalities', None)
        with self.assertRaisesRegex(ValueError, 'ordinal'):
            risk.fit_risk_head(cal, val, cm, vm, 'H_R_C', steps=2)

    def test_complete_risk_analysis_archive_is_compatible_with_evaluator(self):
        from experiments.chronological.evaluate_unified_adapter_v03 import validate_archive
        cal, val, cm, vm = fixture()
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            output = Path(tmp)/'out'
            report = risk.run_risk_analysis(cal, val, cm, vm, output)
            self.assertEqual(set(report['scores']), {'H', 'H_R', 'H_R_C'})
            self.assertTrue(report['same_center_for_all_information_conditions'])
            checked = validate_archive(output/'risk_predictions.npz')
            self.assertEqual(len(checked['quantiles']), 3)
            self.assertTrue((output/'analysis'/'quantiles.csv').is_file())
            self.assertTrue((output/'analysis'/'groups.csv').is_file())
            self.assertTrue((output/'analysis'/'directions.csv').is_file())
            analysis = json.loads((output/'analysis'/'analysis.json').read_text(encoding='utf-8'))
            self.assertEqual(analysis['comparisons']['RISK_H_R_vs_RISK_H_same_center']['mae_gain_pooled'], 0.)
            self.assertEqual(report['runtime']['H_R_C']['fit']['iterations'], 300)
            self.assertEqual(report['runtime']['H_R_C']['predict']['processed_points'], val['y'].size)
            self.assertFalse(report['complete_deployment_latency_measured'])

    def test_risk_predict_rechecks_the_concrete_center_role(self):
        cal, val, cm, vm = fixture()
        model = risk.fit_risk_head(cal, val, cm, vm, steps=1)
        changed = deepcopy(vm)
        changed['prediction_case_roles']['H_ONLY']['checkpoint_sha256'] = 'different'
        with self.assertRaisesRegex(ValueError, 'H_ONLY role checkpoint'):
            risk.predict_risk_head(model, val, changed)


if __name__ == '__main__':
    unittest.main()
