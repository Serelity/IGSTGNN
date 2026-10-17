"""Training identities, held-out rows, curriculum, and paired archive inputs."""
from copy import deepcopy
import logging
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from experiments.chronological import train_unified_adapter_v03 as runner
from experiments.chronological.predict_unified_adapter_v03 import (
    archive_features, matched_semantic_donors, available_semantic_pair)
from experiments.chronological.original_igstgnn_source import get_original_modules
from tests import test_original_isp_runner as fixtures


class V03RunnerTests(unittest.TestCase):
    def test_default_controls_and_protocol_fix_primary_before_results(self):
        args = runner.parse_args(['--run-dir', 'fixture'])
        self.assertEqual(args.arms, ['ORIGINAL', 'TRAFFIC_CORE', 'H_ONLY', 'R_ONLY', 'BOTH', 'DIRECT'])
        self.assertEqual(args.curriculum, 'original')
        self.assertEqual(args.holdout_rows, 0)
        self.assertEqual(runner.PROTOCOL['primary_comparison'], 'BOTH_vs_ORIGINAL')
        self.assertEqual(runner.PROTOCOL['paired_seeds'], [2025, 2026, 2027])
        self.assertFalse(runner.PROTOCOL['independent_final_confirmation_completed'])

    def test_invalid_arm_budget_and_holdout_rejected(self):
        for tail in (['--arms','BOTH','BOTH'], ['--only','STACKED'], ['--epochs','0'], ['--holdout-rows','-1']):
            with self.subTest(tail=tail), patch('sys.stderr'), self.assertRaises(SystemExit):
                runner.parse_args(['--run-dir','fixture',*tail])

    def test_validation_mechanisms_use_counts_not_batch_means(self):
        acc = runner.DiagnosticsAccumulator()
        acc.add(dict(state_abs_sum=12, state_elements=4, supported_positions=2, node_positions=4,
                     output_delta_abs_max=.03))
        acc.add(dict(state_abs_sum=3, state_elements=1, supported_positions=1, node_positions=1,
                     output_delta_abs_max=.01))
        result = acc.result()
        self.assertEqual(result['batches'],2)
        self.assertEqual(result['state_abs_mean'],3)
        self.assertEqual(result['supported_fraction'],.6)
        self.assertEqual(result['output_delta_abs_max'],.03)

    @staticmethod
    def samples():
        result=[]
        for i in range(5):
            x=np.zeros((12,4,3),np.float32)
            x[...,0]=i*.01
            x[...,1]=.25
            x[...,2]=1/7
            result.append(dict(x_data=x,y_data=np.ones((12,4,3),np.float32),
                incident_features={'Description':i%2,'Type':i%2,'Holiday':0},
                incident_position=0,incident_distances=np.ones((4,3),np.float32)))
        return result

    def test_holdout_reserved_before_all_arm_fitting_and_keeps_original_ids(self):
        modules=get_original_modules()
        samples=self.samples()
        logger=logging.getLogger('v03-holdout-test')
        loader=modules.loader.IncidentDataLoader(samples,True,2,logger,input_dim=3,sensor_info=None)
        identity={'splits':{'train':{'selected_indices':[0,1,2,3,4],'count':5}}}
        data=SimpleNamespace(loaders={'train_loader':loader},identity=identity)
        result=runner.partition_training(data,modules,logger,2)
        self.assertEqual(len(result.loaders['train_loader'].samples),3)
        self.assertEqual(result.identity['splits']['train']['selected_indices'],[0,1,2])
        self.assertEqual(result.identity['splits']['calibration']['selected_indices'],[3,4])
        self.assertTrue(result.identity['calibration']['eligible_for_auxiliary_fit'])
        self.assertEqual(identity['splits']['train']['count'],5)
        self.assertEqual(len(loader.samples),5)

    def test_no_holdout_cannot_be_declared_eligible_for_policy_fit(self):
        data=SimpleNamespace(identity={'splits':{}},loaders={})
        result=runner.partition_training(data,None,None,0)
        self.assertFalse(result.identity['calibration']['eligible_for_auxiliary_fit'])

    def test_resume_replays_same_real_optimizer_rng_order_and_selection(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            whole, split=Path(temp)/'whole',Path(temp)/'split'
            whole.mkdir();split.mkdir()
            engine=fixtures.Engine()
            identity=fixtures.identity_for(engine,'BOTH')
            identity['curriculum']='original'
            with patch('builtins.print'):
                expected=runner.run_epochs(engine,whole,identity,'BOTH',4)
                runner.run_epochs(fixtures.Engine(),split,identity,'BOTH',2)
                state=runner.load_checkpoint(split/'last.pt')
                runner.validate_saved(state,identity,'BOTH',split,2)
                actual=runner.run_epochs(fixtures.Engine(99),split,identity,'BOTH',4,state)
            self.assertEqual(fixtures.comparable_history(expected),fixtures.comparable_history(actual))
            for key in ('model_state','optimizer_state','scheduler_state','rng_state','train_order'):
                self.assertEqual(runner.tensor_tree_hash(expected[key]),runner.tensor_tree_hash(actual[key]))

    def test_full_curriculum_resume_checks_own_contract(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            run=Path(temp)
            engine=fixtures.Engine()
            engine._warm_step=100000
            identity=fixtures.identity_for(engine,'BOTH')
            identity['curriculum']='full'
            with patch('builtins.print'):
                state=runner.run_epochs(engine,run,identity,'BOTH',3)
            self.assertEqual(state['curriculum_length'],12)
            runner.validate_saved(state,identity,'BOTH',run,2)
            bad=dict(state,curriculum_length=2)
            with self.assertRaisesRegex(ValueError,'Full-horizon'):
                runner.validate_saved(bad,identity,'BOTH',run,2)

    def test_summary_never_labels_single_pilot_as_confirmed_improvement(self):
        def state(arm):
            history=[dict(epoch=1,validation_mae=3.,validation_rmse=4.,validation_mape_fraction=.1)]
            return dict(arm=arm,completed_epoch=1,history=history,best_epoch=1,best_validation_mae=3.,
                best_checkpoint_file='best.pt',best_checkpoint_sha256='x',model_tensor_sha256='y',
                global_updates=2,initialization={})
        states={arm:state(arm) for arm in runner.DEFAULT_ARMS}
        identity=dict(arms=list(runner.DEFAULT_ARMS),engineering_only=True,curriculum='original',
                      data={'calibration':{'rows':0}},training_config={'seed':2025})
        result=runner.paired_summary(states,identity,1)
        self.assertTrue(result['same_completed_training_budget'])
        self.assertFalse(result['formal_replication_completed'])
        self.assertFalse(result['exceeds_baseline_confirmed'])
        self.assertIn('BOTH_vs_R_ONLY',result['comparisons'])

    def test_features_and_semantic_matching_never_read_targets(self):
        class NoTarget(dict):
            def __getitem__(self,key):
                if key=='y_data':
                    raise AssertionError('Future target read')
                return super().__getitem__(key)
        samples=[NoTarget(sample) for sample in self.samples()]
        features,manifest=archive_features(samples)
        self.assertEqual(features['feature__H_HISTORY'].shape,(5,12,4,6))
        self.assertEqual(manifest['C_CATEGORY']['categorical_cardinalities'],[30,7,13])
        self.assertEqual(manifest['R_RELATION']['information_channel'],'R')
        donors,report=matched_semantic_donors(samples,np.array(['road']*4))
        self.assertEqual(sorted(donors.tolist()),list(range(5)))
        self.assertFalse(report['uses_targets'])
        self.assertTrue(report['pair_preserved'])

    def test_semantic_shuffle_never_resurrects_unavailable_final_categories(self):
        samples=self.samples()
        samples[0]['available']=False
        samples[0]['incident_features'].update(Description=27,Type=5)
        self.assertEqual(available_semantic_pair(samples[0],4).tolist(),[-1,-1])
        donors,_=matched_semantic_donors(samples,np.array(['road']*4))
        self.assertEqual(donors[0],0)
        self.assertNotIn(0,donors[1:])


if __name__=='__main__':
    unittest.main()
