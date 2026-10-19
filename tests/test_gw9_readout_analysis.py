"""Server-only end-to-end numerical/CLI checks for the read-only analysis."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experiments.branch_readout_gw9 import analyze_readouts as analysis
from experiments.branch_readout_gw9 import readout_auth as auth
from experiments.branch_readout_gw9 import readout_history as history


def archive_fixture():
    y=(10.+np.arange(108,dtype=np.float32).reshape(3,12,3)%6).astype(np.float32)
    y[0,0,0]=0.
    r=y+3.; s=r-.5; b=s+np.array([-.25,.25,0.],dtype=np.float32)[None,None,:]
    return dict(schema=np.array(auth.ARCHIVE_SCHEMA),axis_order=np.array('sample,horizon,node'),
        registered_arms=np.array(auth.ARMS),y=y,valid=y!=0,
        sample_ids=np.array([3909,3910,3911]),node_ids=np.array([10,20,30]),
        road_labels=np.array(['A','B','A']),
        direct_support=np.array([[True,False,False],[False,True,False],[False,False,False]]),
        pred__FROZEN_R=r,pred__SUM_CONTROL=s,pred__BRANCH_READOUT=b)


class ReadoutAnalysisTests(unittest.TestCase):
    def test_all_disjoint_group_contributions_recover_global_gain(self):
        archive=archive_fixture()
        before={key:value.tobytes() for key,value in archive.items()}
        result,tables=analysis.build_analysis(archive)
        self.assertEqual(result['valid_count'],107)
        self.assertFalse(result['history']['available'])
        self.assertEqual(len(tables['by_horizon']),12*6)
        self.assertEqual(len(tables['by_node']),3*6)
        self.assertEqual(len(tables['by_flow']),(len(analysis.FLOW_EDGES)-1)*6)
        for row in tables['partition_cancellation']:
            expected=result['global_metrics']['comparisons'][row['comparison']]['mae_gain']
            self.assertAlmostEqual(row['net_global_contribution'],expected)
            self.assertEqual(row['valid_count'],107)
        self.assertFalse(result['joint_eligibility_changed'])
        self.assertFalse(result['candidate_for_confirmation'])
        self.assertFalse(result['fits_or_oracles_applied'])
        self.assertEqual(result['inference_calls'],0)
        self.assertEqual(result['optimizer_updates'],0)
        self.assertEqual(before,{key:value.tobytes() for key,value in archive.items()})
        json.dumps(result,allow_nan=False)

    def test_fixed_flow_masks_partition_negative_boundary_and_extreme_values(self):
        values=np.array([-1.,0.,4.99,5.,20.,50.,100.,200.,314.,1000.])
        masks=[mask for _,mask in analysis.flow_masks(values)]
        np.testing.assert_array_equal(np.sum(masks,axis=0),np.ones(len(values)))
        self.assertTrue(masks[0][0])
        self.assertTrue(masks[1][1])
        self.assertTrue(masks[2][3])
        self.assertTrue(masks[-1][-2])

    def test_history_correlations_and_matched_controls_use_origin_inputs(self):
        archive=archive_fixture()
        train={name:np.arange(12,dtype=np.float32).reshape(4,3) for name in history.FEATURES}
        features={name:np.arange(9,dtype=np.float32).reshape(3,3) for name in history.FEATURES}
        features['neighbor_change_disagreement'][:,2]=np.nan
        metadata=dict(train_input_cutpoints=history.train_cutpoints(train),file_digests={})
        result,tables=analysis.build_analysis(archive,features,metadata)
        self.assertTrue(result['history']['available'])
        self.assertFalse(result['history']['predictive_skill_validated'])
        self.assertEqual(result['history']['matched_controls'],
            ['forecast_horizon','origin_last_flow_fixed_bins'])
        self.assertEqual(len(tables['history_features']),len(history.FEATURES)*13)
        self.assertEqual({row['feature'] for row in tables['matched_history']},set(history.MATCHED_FEATURES))
        matched_cases=[row for row in tables['matched_history']
            if row['record_kind']=='case' and row['case']=='SUM_CONTROL']
        for name in history.MATCHED_FEATURES:
            self.assertEqual(sum(row['count'] for row in matched_cases if row['feature']==name),107)
        self.assertIn('开发集描述性关联',analysis.report_text(result))
        json.dumps(result,allow_nan=False)

    def test_empty_or_constant_correlations_are_undefined(self):
        self.assertIsNone(analysis.safe_corr([],[]))
        self.assertIsNone(analysis.safe_corr([1.,1.],[2.,3.]))
        self.assertAlmostEqual(analysis.safe_corr([1.,2.,3.],[3.,2.,1.]),-1.)

    def test_cli_writes_new_artifacts_and_preserves_all_source_files(self):
        archive=archive_fixture()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); probe=root/'probe'; probe.mkdir()
            evaluation=probe/'evaluation_demo'
            for role in ('best','last'):
                (evaluation/role).mkdir(parents=True)
                (evaluation/role/'paired_predictions.npz').write_bytes(b'same archived bytes')
            source=probe/'screening.json'; source.write_text('failed gate remains unchanged',encoding='utf-8')
            cert=dict(source_identity_sha256='fixture',evaluation_directory=str(evaluation),
                file_digests={str(source):auth.sha256(source)},protected_directories=[str(probe)],data_identity={})
            output=root/'analysis'
            with patch.object(auth,'load_authenticated_run',return_value=(archive,deepcopy(archive),cert)), \
                    patch('builtins.print'):
                result=analysis.main(['--probe-dir',str(probe),'--output-dir',str(output)])
            self.assertEqual(result['status'],'GW9_READOUT_ANALYSIS_COMPLETE')
            self.assertTrue(result['provenance']['best_last_predictions_equal'])
            self.assertTrue(result['provenance']['best_last_archive_same_bytes'])
            for name in ('report.md','analysis.json','by_horizon.csv','by_node.csv',
                    'by_flow.csv','by_predicted_flow.csv','by_sample_block.csv','partition_cancellation.csv'):
                self.assertTrue((output/name).is_file(),name)
            self.assertEqual(source.read_text(encoding='utf-8'),'failed gate remains unchanged')
            self.assertFalse((probe/'analysis.json').exists())
            with patch.object(auth,'load_authenticated_run',return_value=(archive,deepcopy(archive),cert)), \
                    self.assertRaises(ValueError):
                analysis.main(['--probe-dir',str(probe),'--output-dir',str(output)])

    def test_parser_rejects_model_or_training_budget_options(self):
        common=['--probe-dir','old','--output-dir','new']
        for extra in (['--epochs','3'],['--device','cuda:0'],['--fit'],['--resume']):
            with self.subTest(extra=extra),self.assertRaises(SystemExit):analysis.parse_args(common+extra)

    def test_cli_optional_history_path_exports_conditional_contrasts(self):
        archive=archive_fixture()
        features={name:np.arange(9,dtype=np.float32).reshape(3,3) for name in history.FEATURES}
        metadata=dict(train_input_cutpoints=history.train_cutpoints(features),file_digests={})
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); probe=root/'probe'; probe.mkdir()
            data=root/'data'; data.mkdir()
            evaluation=probe/'evaluation_demo'
            for role in ('best','last'):
                (evaluation/role).mkdir(parents=True)
                (evaluation/role/'paired_predictions.npz').write_bytes(b'identical archives')
            identity=dict(synthetic=True)
            cert=dict(source_identity_sha256='fixture',evaluation_directory=str(evaluation),
                file_digests={},protected_directories=[str(probe)],data_identity=identity)
            output=root/'with_history'
            with patch.object(auth,'load_authenticated_run',return_value=(archive,deepcopy(archive),cert)), \
                    patch.object(history,'load_origin_features',return_value=(features,metadata)) as loader, \
                    patch('builtins.print'):
                result=analysis.main(['--probe-dir',str(probe),'--output-dir',str(output),'--data-dir',str(data)])
            loader.assert_called_once_with(data.resolve(),identity)
            self.assertTrue(result['history']['available'])
            self.assertFalse(result['history']['predictive_skill_validated'])
            self.assertFalse(result['joint_eligibility_changed'])
            self.assertGreater((output/'matched_contrasts.csv').stat().st_size,0)
            self.assertGreater((output/'history_features.csv').stat().st_size,0)
            text=(output/'report.md').read_text(encoding='utf-8')
            self.assertIn('同一时距与起点流量',text)
            self.assertIn('不是显著性检验',text)

    def test_fresh_import_does_not_load_torch_or_any_training_module(self):
        code=('import sys; from experiments.branch_readout_gw9 import analyze_readouts; '
            'assert not any(n=="torch" or n.startswith("torch.") for n in sys.modules); '
            'assert "experiments.branch_readout_gw9.train" not in sys.modules; print("PURE_NUMPY_ANALYSIS")')
        completed=subprocess.run([sys.executable,'-c',code],cwd=analysis.REPO,
            capture_output=True,text=True,check=True)
        self.assertIn('PURE_NUMPY_ANALYSIS',completed.stdout)


if __name__=='__main__': unittest.main()
