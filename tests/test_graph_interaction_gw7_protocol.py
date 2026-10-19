"""Server-only GW7 budget, stage-zero, resume and screening contracts.

Tiny stochastic optimizations test the protocol, not traffic-model accuracy.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from experiments.graph_interaction_gw7 import train, evaluate, reference
from tests.test_original_isp_runner import Engine as PreviousEngine, comparable_history


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor=nn.Sequential(nn.Linear(1,3),nn.Dropout(.2),nn.Linear(3,1))
        self.residual=nn.Linear(1,1,bias=False)
        nn.init.zeros_(self.residual.weight)

    def forward(self,x):
        return self.anchor(x)+self.residual(x)

    def trainable_parameter_counts(self):
        return dict(total=sum(p.numel() for p in self.parameters()))

    def diagnostics(self):
        return {}


class Engine(PreviousEngine):
    def __init__(self,seed=2025,constant_validation=None):
        super().__init__(seed)
        self.model=ToyModel()
        self._optimizer=torch.optim.Adam(self.model.parameters(),lr=.01)
        self._lr_scheduler=torch.optim.lr_scheduler.MultiStepLR(self._optimizer,milestones=[1,3],gamma=.5)
        self._warm_step,self._cl_step=202,6
        self.constant_validation=constant_validation
        train.base.configure_runtime('cpu',1,seed)

    def evaluate(self,mode):
        values=super().evaluate(mode)
        return values if self.constant_validation is None else (self.constant_validation,.1,.5)


def fixed_components(engine):
    # Finite, zero-increment model-domain reference. No RNG consumption.
    base=torch.ones(2,12,1,1)*next(engine.model.anchor.parameters()).detach().mean()
    return 'fixed-input',dict(base=base,correction=torch.zeros_like(base),total=base.clone())


def identity_for(engine):
    identity=dict(schema=train.SCHEMA,design=train.DESIGN,arms=list(train.ARMS),
        engineering_only=False,initializations={arm:train.initialization(engine.model) for arm in train.ARMS},
        data=dict(splits=dict(train=dict(count=4))),training_config=dict(warm_epoch=101,cl_epoch=3),
        source_reference={},model_options=dict(hidden=32,node_chunk=32))
    identity['identity_sha256']=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    return identity


def screen_inputs():
    cases={arm:dict(mae_pooled=value) for arm,value in zip(train.ARMS,(10.,9.99,10.01,9.9))}
    groups={arm:{name:dict(mae=value) for name in ('high_value_ge314','long_horizon_7_12')}
        for arm,value in zip(train.ARMS,(20.,19.,21.,19.5))}
    return dict(cases=cases,groups=groups)


class GraphInteractionProtocolTests(unittest.TestCase):
    def test_parser_requires_diagnostic_and_exact_five_epoch_budget(self):
        argv=['--gw6-run','old','--run-dir','new','--diagnostic-dir','diag']
        self.assertEqual(train.parse_args(argv).epochs,5)
        for extra in (['--epochs','20'],['--epochs','0'],['--hidden','0'],
            ['--only','R_MLP','R_MLP']):
            with self.subTest(extra=extra),self.assertRaises(SystemExit): train.parse_args(argv+extra)
        with self.assertRaises(SystemExit): train.parse_args(['--gw6-run','old','--run-dir','new'])

    def test_stage_zero_participates_in_best_and_ties_keep_zero(self):
        with tempfile.TemporaryDirectory() as directory,patch('builtins.print'),patch.object(train,'fixed_components',fixed_components):
            engine=Engine(constant_validation=1.); identity=identity_for(engine)
            state=train.run_epochs(engine,Path(directory),identity,'R_CONTINUE',2)
            train.validate_saved(state,identity,'R_CONTINUE',directory,2)
            self.assertEqual(state['best_epoch'],0)
            self.assertEqual(state['wait'],2)
            self.assertTrue(train.arm_summary(state)['best_selection_includes_stage_zero'])
            self.assertFalse(train.arm_summary(state)['exceeds_stage_zero'])
            self.assertEqual(train.arm_summary(state)['best_metrics']['mae'],1.)

    def test_exact_resume_preserves_optimizer_rng_order_and_component_reference(self):
        with tempfile.TemporaryDirectory() as directory,patch('builtins.print'),patch.object(train,'fixed_components',fixed_components):
            whole,split=Path(directory)/'whole',Path(directory)/'split'; whole.mkdir(); split.mkdir()
            identity=identity_for(Engine())
            expected=train.run_epochs(Engine(),whole,identity,'R_CONTINUE',4)
            train.run_epochs(Engine(),split,identity,'R_CONTINUE',2)
            saved=train.base.load_checkpoint(split/'last.pt')
            train.validate_saved(saved,identity,'R_CONTINUE',split,2)
            resumed=train.run_epochs(Engine(97),split,identity,'R_CONTINUE',4,saved)
            train.validate_saved(resumed,identity,'R_CONTINUE',split,2)
            self.assertEqual(comparable_history(expected),comparable_history(resumed))
            for key in ('model_state','optimizer_state','scheduler_state','rng_state','train_order','fixed_reference_components'):
                self.assertEqual(train.base.tensor_tree_hash(expected[key]),train.base.tensor_tree_hash(resumed[key]),key)
            self.assertEqual(resumed['global_updates'],8)
            self.assertTrue(all(row['curriculum_length']==12 for row in resumed['history']))

    def test_first_epoch_kill_restores_stage_zero_and_replays_incomplete_epoch(self):
        with tempfile.TemporaryDirectory() as directory,patch('builtins.print'),patch.object(train,'fixed_components',fixed_components):
            whole,killed=Path(directory)/'whole',Path(directory)/'killed'; whole.mkdir(); killed.mkdir()
            identity=identity_for(Engine())
            expected=train.run_epochs(Engine(),whole,identity,'R_CONTINUE',1)
            interrupted=Engine(); interrupted.fail_at_update=1
            with self.assertRaises(KeyboardInterrupt): train.run_epochs(interrupted,killed,identity,'R_CONTINUE',1)
            saved=train.base.load_checkpoint(killed/'last.pt')
            self.assertEqual(saved['completed_epoch'],0); self.assertEqual(saved['best_epoch'],0)
            train.validate_saved(saved,identity,'R_CONTINUE',killed,2)
            resumed=train.run_epochs(Engine(99),killed,identity,'R_CONTINUE',1,saved)
            self.assertEqual(comparable_history(expected),comparable_history(resumed))
            self.assertEqual(expected['model_tensor_sha256'],resumed['model_tensor_sha256'])

    def test_saved_state_rejects_budget_identity_zero_selection_and_components_tampering(self):
        with tempfile.TemporaryDirectory() as directory,patch('builtins.print'),patch.object(train,'fixed_components',fixed_components):
            engine=Engine(); identity=identity_for(engine)
            state=train.run_epochs(engine,Path(directory),identity,'R_CONTINUE',1)
            mutations=(lambda s:s.update(global_updates=3),lambda s:s.update(identity_sha256='bad'),
                lambda s:s.update(train_order=[0,0,1,2]),lambda s:s.update(best_epoch=4),
                lambda s:s['fixed_reference_components']['correction'].fill_(1.),
                lambda s:s.update(stage_zero_model_tensor_sha256='bad'))
            for mutation in mutations:
                bad=deepcopy(state); mutation(bad)
                with self.assertRaises(ValueError): train.validate_saved(bad,identity,'R_CONTINUE',directory,2)

    def test_loop_refuses_unbounded_extension(self):
        with self.assertRaises(ValueError): train.run_epochs(None,Path('unused'),{},'R_CONTINUE',20)

    def test_common_stage_zero_replays_certified_source(self):
        with tempfile.TemporaryDirectory() as directory,patch('builtins.print'),patch.object(train,'fixed_components',fixed_components):
            engine=Engine(); identity=identity_for(engine)
            identity['source_reference']={'source_validation_metrics':dict(mae=999.,rmse=999.,mape_fraction=999.)}
            with self.assertRaisesRegex(ValueError,'certified R20'): train.run_epochs(engine,Path(directory),identity,'R_CONTINUE',0)

    def test_screen_all_conditions_pass_and_never_claim_significance(self):
        best=screen_inputs(); last=deepcopy(best); last['cases']['R_INTERACTION_MEMORY']['mae_pooled']=9.95
        result=evaluate.screening(best,last)
        self.assertTrue(result['candidate_for_confirmation'])
        self.assertFalse(result['test_evaluated'])
        self.assertIn('not_statistical_significance',result['interpretation'])

    def test_screen_rejects_weak_gain_late_drift_or_regime_regression(self):
        for mutate in (lambda b,l:b['cases']['R_INTERACTION_MEMORY'].update(mae_pooled=9.99),
            lambda b,l:l['cases']['R_INTERACTION_MEMORY'].update(mae_pooled=10.2),
            lambda b,l:b['groups']['R_INTERACTION_MEMORY']['high_value_ge314'].update(mae=20.1),
            lambda b,l:b['groups']['R_INTERACTION_MEMORY']['long_horizon_7_12'].update(mae=None)):
            best=screen_inputs(); last=deepcopy(best); mutate(best,last)
            self.assertFalse(evaluate.screening(best,last)['candidate_for_confirmation'])

    def test_groups_keep_original_mask_and_fixed_horizon_and_threshold(self):
        y=np.ones((2,12,2))*314.; prediction=y+1.; valid=np.ones_like(y,dtype=bool)
        y[0,0,0]=313.; valid[1,11,1]=False; prediction[1,11,1]=9999.
        groups=evaluate.group_metrics(prediction,y,valid)
        self.assertEqual(groups['high_value_ge314']['valid_count'],46)
        self.assertEqual(groups['high_value_ge314']['mae'],1.)
        self.assertEqual(groups['long_horizon_7_12']['valid_count'],23)
        self.assertEqual(groups['long_horizon_7_12']['mae'],1.)

    def test_epoch_zero_native_replay_uses_zero_record_not_last_history_row(self):
        state=dict(stage_zero=dict(validation_mae=1.,validation_rmse=2.,validation_mape_fraction=.1),
            history=[dict(validation_mae=4.,validation_rmse=5.,validation_mape_fraction=.4)])
        with patch.object(evaluate.metrics,'native_metrics',return_value=dict(mae=1.,rmse=2.,mape_fraction=.1)):
            result=evaluate.replay_metrics(None,None,state,0,None)
        self.assertTrue(result['passed']); self.assertEqual(result['expected']['mae'],1.)

    def test_paired_axes_reject_dtype_mask_and_node_order_changes(self):
        arrays={key:np.arange(4,dtype=np.int64) for key in evaluate.AXIS_KEYS}
        evaluate.require_matching_axes(arrays,deepcopy(arrays))
        for key,value in (('node_ids',np.arange(4,dtype=np.int64)[::-1]),
            ('y',np.arange(4,dtype=np.float32)),('valid',np.ones(4,dtype=np.int64))):
            wrong=deepcopy(arrays); wrong[key]=value
            with self.assertRaisesRegex(ValueError,key): evaluate.require_matching_axes(arrays,wrong)

    def test_diagnostic_gate_authenticates_roles_budget_and_producer(self):
        role=lambda selection,epoch:dict(selection=selection,epoch=epoch,policy='JOINT_MEMORY',
            checkpoint_sha256='weights',model_tensor_sha256='tensors')
        roles=dict(best=role('best',18),last=role('last',20))
        certificate=dict(source_identity_sha256='identity',source_run='source',
            checkpoint_roles=dict(JOINT_MEMORY=roles))
        paths=('experiments/graph_memory_gw6/diagnose_layers.py',
            'experiments/graph_memory_gw6/diagnose_drift.py',
            'experiments/graph_memory_gw6/evaluate.py','experiments/graph_memory_gw6/model.py')
        value=dict(status='GW6_LAYER_DIAGNOSTIC_COMPLETE',source_run_identity_sha256='identity',
            source_run='source',reconstruction_pass=True,test_evaluated=False,optimizer_updates=0,
            source_budget_epochs=20,archive_prediction_replay_pass=True,model_tensors_unchanged=True,
            checkpoint_writes=0,bn_recalibration=False,input_manifest_sha256='digest',
            producer_source={path:'digest' for path in paths},checkpoint_roles=deepcopy(roles))
        manifest=dict(source_run_identity_sha256='identity',diagnostic_script_sha256='digest',
            execution_source={name:'digest' for name in reference.gw6.EXECUTION_FILES})
        def reader(path): return manifest if Path(path).name=='input_manifest.json' else value
        with patch.object(reference,'read_json',side_effect=reader),patch.object(reference.gw6.base,'sha256',return_value='digest'):
            self.assertTrue(reference.validate_diagnostic(Path('diag'),certificate)['reconstruction_pass'])
            for mutation in (lambda d:d.update(source_budget_epochs=5),
                lambda d:d.update(optimizer_updates=1),
                lambda d:d['checkpoint_roles']['best'].update(epoch=5),
                lambda d:d['producer_source'].update({paths[0]:'wrong'})):
                original=deepcopy(value)
                mutation(value)
                with self.assertRaises(ValueError): reference.validate_diagnostic(Path('diag'),certificate)
                value.clear(); value.update(original)


if __name__=='__main__': unittest.main()
