"""Actual small MIN/F2 models: replay blocking, interventions and preservation."""
import copy
import csv
import json
import logging
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.chronological import evaluate_min_component_fusion as evaluation
from experiments.chronological import train_min_forecast_fusion as reference
from src.models.incident_spatial_component_fusion import from_minimal
from src.models.incident_spatial_residual import tensor_state_hash
from src.utils.igstgnn_published_events import EventPublishedData
from src.utils.metrics import compute_all_metrics
from tests.test_incident_spatial import graph
from tests.test_min_event_budget_data import event_fixture
from tests.test_min_event_budget_runner import tiny_minimal


class ComponentFusionEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        cls.logger=logging.getLogger('F2_evaluation_test')
        cls.logger.addHandler(logging.NullHandler())

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=evaluation.REPO)
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        raw=self.root/'raw';raw.mkdir()
        _,protocol=event_fixture(raw)
        self.data=EventPublishedData(raw,protocol,self.logger,check=True)
        self.graph=graph(4)
        core,_=tiny_minimal(SimpleNamespace(seed=2025),4,self.graph,'cpu')
        core.eval()
        self.arrays,base=reference.validation_predictions(core,self.data,torch.device('cpu'))
        self.baseline=dict(base,engineering_only=True,validation_rows=len(self.arrays['target']),source_best_epoch=28)
        self.models={'CONTROL':copy.deepcopy(core)}
        for arm,mode in (('STATE','state'),('EVENT','event')):
            model,_=from_minimal(core,mode=mode,width=12)
            with torch.no_grad():
                # Break zero initialization so all interventions have an actual path.
                for name,parameter in model.forecast_fusion.named_parameters():
                    if name.endswith('weight') and parameter.ndim==2:
                        parameter.add_(torch.linspace(-.05,.08,parameter.numel()).reshape_as(parameter))
            model.eval();self.models[arm]=model
        self.saved={}
        for arm,model in self.models.items():
            rows=[evaluation.forward_case(model,batch,self.data.scaler,arm+'_ON')['prediction']
                for batch in self.data.loaders['val_loader'].get_iterator()]
            actual=compute_all_metrics(torch.from_numpy(np.concatenate(rows)),torch.from_numpy(self.arrays['target']),torch.tensor(self.baseline['mask_value']))
            self.saved[arm]=dict(completed_epoch=1,best_epoch=1,best_metric=actual[0],global_updates=2,
                history=[dict(epoch=1,validation_mae=actual[0],validation_mape=actual[1],validation_rmse=actual[2],train_order_sha256='a'*64)],
                best_model_state=copy.deepcopy(model.state_dict()))
        self.source=self.root/'source.txt';self.source.write_text('immutable',encoding='utf-8')
        self.paths={'source':self.source}

    def run_probe(self,name='evaluation',saved=None,arrays=None,baseline=None):
        return evaluation.run_probe(self.models,self.data,self.saved if saved is None else saved,self.graph,
            {'road-A':[0,1],'road-B':[2,3]},self.root/name,self.baseline if baseline is None else baseline,
            self.arrays if arrays is None else arrays,self.paths)

    def test_nine_cases_original_masks_replay_restore_and_all_artifacts(self):
        hashes={arm:tensor_state_hash(model.state_dict()) for arm,model in self.models.items()}
        source=self.source.read_bytes()
        result=self.run_probe()
        self.assertEqual(result['status'],'MIN_COMPONENT_FUSION_EVALUATION_PASS')
        self.assertEqual(set(result['cases']),set(evaluation.CASES))
        self.assertEqual(result['new_inference_calls']['SOURCE_MIN'],0)
        batches=len(self.baseline['batch_partition'])
        self.assertEqual(result['total_new_inference_calls'],13*batches)
        self.assertTrue(result['on_exactly_restored'])
        self.assertEqual(result['optimizer_updates'],0)
        self.assertFalse(result['test_evaluated'])
        self.assertEqual({arm:tensor_state_hash(model.state_dict()) for arm,model in self.models.items()},hashes)
        self.assertEqual(self.source.read_bytes(),source)
        output=self.root/'evaluation'
        for name in ('analysis.json','comparability.json','replay.json','mechanisms.json','preservation.json','report.md',
                     'paired_predictions.npz','region_definitions.json','weights.csv','horizons.csv','regions.csv','strata.csv'):
            self.assertTrue((output/name).is_file(),name)
        with np.load(output/'paired_predictions.npz',allow_pickle=False) as archive:
            np.testing.assert_array_equal(archive['SOURCE_MIN'],self.arrays['prediction'])
            np.testing.assert_array_equal(archive['target'],self.arrays['target'])
            np.testing.assert_array_equal(archive['valid'],self.arrays['valid'])
            np.testing.assert_array_equal(archive['station_ids'],self.data.station_ids)
            self.assertFalse(np.array_equal(archive['EVENT_ON'],archive['EVENT_CONTENT_OFF']))
            supported=np.broadcast_to(np.array([1,0,1,0],bool)[None,None],archive['target'].shape)
            np.testing.assert_array_equal(archive['EVENT_ON'][~supported],archive['EVENT_REPORT_OFF'][~supported])
        with (output/'weights.csv').open(encoding='utf-8-sig',newline='') as stream:
            weights=list(csv.DictReader(stream))
        self.assertEqual(len(weights),7*12*10)
        self.assertNotIn('CONTROL_ON',{row['case'] for row in weights})
        report=(output/'report.md').read_text(encoding='utf-8')
        for text in ('工程检查：True','不构成预测效果','不能单独归因为分量内容感知','未新增事故空间传播','可能分布外','原 IGSTGNN','新增源模型调用为0'):
            self.assertIn(text,report)

    def test_bad_complete_replay_blocks_every_intervention(self):
        saved=copy.deepcopy(self.saved)
        saved['EVENT']['best_metric']+=1
        saved['EVENT']['history'][0]['validation_mae']+=1
        seen=[];original=evaluation.forward_case
        def record(*args,**kwargs):
            seen.append(args[3]);return original(*args,**kwargs)
        with patch.object(evaluation,'forward_case',side_effect=record),self.assertRaisesRegex(ValueError,'ON replay failed'):
            self.run_probe('bad_replay',saved=saved)
        self.assertTrue(all(case in evaluation.ON_CASES for case in seen))
        self.assertTrue((self.root/'bad_replay/failure.json').is_file())
        self.assertFalse((self.root/'bad_replay/summary.json').exists())

    def test_failed_restoration_cannot_publish_success(self):
        original=evaluation.forward_case;intervened=[False]
        def faulty(*args,**kwargs):
            result=original(*args,**kwargs)
            if args[3] in evaluation.INTERVENTIONS:
                intervened[0]=True
            elif args[3]=='EVENT_ON' and intervened[0]:
                result['prediction']+=1
            return result
        with patch.object(evaluation,'forward_case',side_effect=faulty),self.assertRaisesRegex(ValueError,'ON restoration failed'):
            self.run_probe('bad_restore')
        self.assertFalse((self.root/'bad_restore/summary.json').exists())
        self.assertTrue(json.loads((self.root/'bad_restore/preservation.json').read_text())['source_artifacts_preserved'])

    def test_case_arm_mismatch_and_content_off_execution_are_checked(self):
        batch=next(self.data.loaders['val_loader'].get_iterator())
        for arm,case in (('CONTROL','EVENT_REPORT_OFF'),('STATE','EVENT_CONTENT_OFF'),('EVENT','STATE_ON')):
            with self.subTest(arm=arm,case=case),self.assertRaisesRegex(ValueError,'another model arm'):
                evaluation.forward_case(self.models[arm],batch,self.data.scaler,case)
        observed=[];original=self.models['EVENT'].forward
        def record(*args,**kwargs):
            observed.append(kwargs);return original(*args,**kwargs)
        with patch.object(self.models['EVENT'],'forward',side_effect=record):
            row=evaluation.forward_case(self.models['EVENT'],batch,self.data.scaler,'EVENT_CONTENT_OFF')
        self.assertFalse(observed[0]['content_enabled'])
        self.assertTrue(observed[0]['report_enabled'] and observed[0]['fusion_enabled'])
        self.assertEqual(row['mechanisms']['fusion_component_content_calls'],0)

    def test_protected_source_or_model_mutation_blocks_success(self):
        original=evaluation.forward_case
        def mutate(*args,**kwargs):
            result=original(*args,**kwargs)
            if args[3]=='EVENT_CONTENT_OFF':
                self.source.write_text('changed',encoding='utf-8')
            return result
        with patch.object(evaluation,'forward_case',side_effect=mutate),self.assertRaisesRegex(ValueError,'Source artifacts changed'):
            self.run_probe('bad_source')
        self.assertFalse((self.root/'bad_source/summary.json').exists())
        self.assertFalse(json.loads((self.root/'bad_source/preservation.json').read_text())['source_artifacts_preserved'])
        self.source.write_text('immutable',encoding='utf-8')
        def mutate_model(*args,**kwargs):
            result=original(*args,**kwargs)
            if args[3]=='EVENT_CONTENT_OFF':
                with torch.no_grad():args[0].T_i_D_emb.add_(.01)
            return result
        with patch.object(evaluation,'forward_case',side_effect=mutate_model),self.assertRaisesRegex(ValueError,'Inference changed model tensors'):
            self.run_probe('bad_model')
        self.assertFalse((self.root/'bad_model/summary.json').exists())

    def test_source_axes_mask_and_batch_partition_cannot_drift(self):
        for key,change in (('station_ids',lambda v:v[::-1]),('sample_indices',lambda v:v[::-1]),('valid',lambda v:~v)):
            arrays=copy.deepcopy(self.arrays);arrays[key]=change(arrays[key])
            with self.subTest(key=key),self.assertRaises(ValueError):self.run_probe('bad_'+key,arrays=arrays)
        with self.assertRaisesRegex(ValueError,'batch partition changed'):
            self.run_probe('bad_batches',baseline=dict(self.baseline,batch_partition=[1]*len(self.arrays['target'])))

    def test_saved_source_selects_hard_set_but_current_backbone_selects_bias(self):
        target=np.full((3,12,4),10.,np.float32);valid=np.ones_like(target,bool)
        predictions={case:target+1 for case in evaluation.CASES}
        predictions['SOURCE_MIN']=target.copy();predictions['SOURCE_MIN'][0]=30
        predictions['CONTROL_ON']=target.copy();predictions['CONTROL_ON'][2]=50
        predictions['EVENT_FUSION_OFF']=target-1;predictions['EVENT_FUSION_OFF'][1]=target[1]+2
        regions=dict(all_nodes=np.ones(4,bool),direct_report_nodes=np.array([1,0,0,0],bool),
            potential_only=np.array([0,1,0,0],bool),remaining_nodes=np.array([0,0,1,1],bool))
        result=evaluation.summarize(predictions,target,valid,regions,np.array([100,101,102]),np.arange(4))
        self.assertEqual(result['fixed_hard_windows']['sample_indices'],[100])
        self.assertEqual(result['fixed_hard_windows']['selector'],'SOURCE_MIN')
        self.assertEqual(result['bias_reference'],'EVENT_FUSION_OFF_for_all_comparisons')
        strata=result['comparisons']['EVENT_vs_STATE']['strata']
        self.assertEqual(strata['bias:over']['valid_count'],48)
        self.assertEqual(strata['bias:under']['valid_count'],96)
        self.assertEqual(result['comparisons']['STATE_vs_CONTROL']['control'],'CONTROL_ON')
        with self.assertRaisesRegex(ValueError,'all nine'):
            evaluation.summarize({k:v for k,v in predictions.items() if k!='SOURCE_MIN'},target,valid,regions,np.arange(3),np.arange(4))

    def test_mechanism_minimum_uses_min_and_counts_sum(self):
        total={}
        evaluation.merge_mechanisms(total,dict(bounded_scale_min=.8,delta_abs_max=.3,calls=2))
        evaluation.merge_mechanisms(total,dict(bounded_scale_min=.9,delta_abs_max=.2,calls=3))
        self.assertEqual(total,dict(bounded_scale_min=.8,delta_abs_max=.3,calls=5))

    def test_unequal_completed_budgets_are_labeled_per_pair(self):
        saved=copy.deepcopy(self.saved)
        row=dict(saved['EVENT']['history'][0],epoch=2)
        saved['EVENT']['history'].append(row)
        saved['EVENT']['completed_epoch']=2
        saved['EVENT']['global_updates']=4
        result=self.run_probe('unequal_budget',saved=saved)
        self.assertFalse(result['training_comparability']['same_completed_training_budget'])
        analysis=json.loads((self.root/'unequal_budget/analysis.json').read_text())
        self.assertTrue(analysis['comparisons']['STATE_vs_CONTROL']['same_completed_training_budget'])
        for name in ('EVENT_vs_STATE','EVENT_vs_CONTROL'):
            self.assertFalse(analysis['comparisons'][name]['same_completed_training_budget'])
            self.assertIn('unequal_completed_budget',analysis['comparisons'][name]['scope'])
        self.assertIn('三组完成训练预算不相等',(self.root/'unequal_budget/report.md').read_text(encoding='utf-8'))

    def test_STATE_passes_no_incident_tensors(self):
        batch=next(self.data.loaders['val_loader'].get_iterator())
        observed=[];original=self.models['STATE'].forward
        def record(*args,**kwargs):
            observed.append(kwargs);return original(*args,**kwargs)
        with patch.object(self.models['STATE'],'forward',side_effect=record),patch.object(evaluation.training,'event_tensors',side_effect=AssertionError('STATE read event tensors')):
            evaluation.forward_case(self.models['STATE'],batch,self.data.scaler,'STATE_ON')
        self.assertIsNone(observed[0]['incident_data'])


if __name__=='__main__':
    unittest.main()
