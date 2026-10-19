"""Server-only checks of the predeclared decision and leakage boundaries."""
import unittest
from unittest.mock import patch

import numpy as np

from experiments.conditional_residual_gw10.screen import REQUIRED_ARMS, metrics, paired, screen
from experiments.conditional_residual_gw10.run_probe import fit, predictions, slice_data


def fixture():
    validation = {arm:dict(mae=10.,valid_count=100) for arm in REQUIRED_ARMS}
    validation['JOINT']['mae'] = 9.8
    audit = {arm:dict(mae=11.,valid_count=100) for arm in REQUIRED_ARMS}
    audit['JOINT']['mae'] = 10.9
    groups = {name:{arm:dict(mae=12.,valid_count=10) for arm in REQUIRED_ARMS}
        for name in ('high_value_ge314','long_horizon_7_12','flow_50_200')}
    return validation,audit,groups


class FixedGateTests(unittest.TestCase):
    def test_pass_is_design_evidence_and_never_joint_authorization(self):
        result = screen(*fixture())
        self.assertTrue(result['eligible_for_new_model_design'])
        self.assertFalse(result['GW9_joint_training_authorized'])
        self.assertFalse(result['candidate_for_confirmation'])
        self.assertEqual(result['candidate'],'JOINT')

    def test_losing_candidate_does_not_choose_winning_other_arm(self):
        validation,audit,groups = fixture()
        validation['OWN4']['mae'] = 8.
        result = screen(validation,audit,groups)
        self.assertFalse(result['eligible_for_new_model_design'])
        self.assertEqual(result['candidate'],'JOINT')
        self.assertFalse(result['winner_selection'])

    def test_training_audit_failure_blocks_a_validation_gain(self):
        validation,audit,groups = fixture()
        audit['ORIGIN']['mae'] = audit['JOINT']['mae']
        self.assertFalse(screen(validation,audit,groups)['eligible_for_new_model_design'])

    def test_each_fixed_safety_group_is_required(self):
        for name in fixture()[2]:
            validation,audit,groups = fixture()
            groups[name]['JOINT']['mae'] += .1
            self.assertFalse(screen(validation,audit,groups)['eligible_for_new_model_design'])

    def test_empty_guard_is_not_an_automatic_pass(self):
        validation,audit,groups = fixture()
        groups['high_value_ge314']['JOINT'] = dict(valid_count=0,mae=None)
        self.assertFalse(screen(validation,audit,groups)['eligible_for_new_model_design'])

    def test_small_improvement_fails_locked_threshold(self):
        validation,audit,groups = fixture(); validation['JOINT']['mae'] = 9.99
        self.assertFalse(screen(validation,audit,groups)['eligible_for_new_model_design'])

    def test_missing_arm_is_an_error(self):
        validation,audit,groups = fixture(); del validation['OWN4']
        with self.assertRaises(ValueError): screen(validation,audit,groups)


class ResidualInterfaceTests(unittest.TestCase):
    def test_paired_help_minus_harm_equals_net(self):
        before = np.array([5.,8.,2.]); after = np.array([4.,10.,2.]); y = np.array([3.,9.,2.])
        row = paired(before,after,y,np.ones(3,dtype=bool))
        self.assertAlmostEqual(row['gross_help']-row['gross_harm'],row['mae_gain'])

    def test_positive_correction_is_added_not_subtracted(self):
        data = dict(pred=np.ones((2,1,1)),features={})
        with patch('experiments.conditional_residual_gw10.run_probe.ARMS',('JOINT',)), patch(
                'experiments.conditional_residual_gw10.run_probe.predict_corrections',
                return_value={'JOINT':np.full((2,1,1),2.)}):
            values = predictions({},data)
        np.testing.assert_array_equal(values['JOINT'],3.)
        np.testing.assert_array_equal(values['SUM'],1.)

    def test_fit_receives_only_its_selected_training_slice(self):
        data = dict(pred=np.arange(10.)[:,None,None],y=np.arange(10.)[:,None,None]+1.,
            valid=np.ones((10,1,1),bool),sample_ids=np.arange(10),
            features={'last_flow':np.arange(10.)[:,None]})
        subset = slice_data(data,0,4)
        with patch('experiments.conditional_residual_gw10.run_probe.fit_probe',return_value={}) as mocked:
            fit(subset)
        arguments = mocked.call_args.args
        self.assertEqual(len(arguments[0]),4)
        np.testing.assert_array_equal(arguments[4],np.arange(4))
        self.assertNotIn('val',arguments[3])

    def test_metrics_ignore_invalid_target_points(self):
        row = metrics(np.array([3.,999.]),np.array([2.,0.]),np.array([True,False]))
        self.assertEqual(row['valid_count'],1); self.assertEqual(row['mae'],1.)
        self.assertEqual(row['signed_bias'],1.)

    def test_nonboolean_or_empty_mask_is_rejected(self):
        for mask in (np.array([1]),np.array([False])):
            with self.assertRaises(ValueError): metrics(np.array([1.]),np.array([2.]),mask)


if __name__ == '__main__': unittest.main()
