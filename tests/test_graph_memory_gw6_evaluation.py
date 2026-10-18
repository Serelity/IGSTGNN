"""Server-side GW6 provenance, matched-arm, cache and numerical contracts.

These tests are shipped for execution in the target PyTorch environment.
They do not fit on validation data and do not evaluate a test loader.
"""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.graph_memory_gw6 import evaluate as evaluation
from experiments.graph_memory_gw6 import reference
from experiments.graph_memory_gw6 import train


def archive_fixture(arms=None, stage='joint'):
    arms = tuple(arms or train.DEFAULT_ARMS)
    y = np.full((5, 12, 4), 10., np.float32)
    archive = dict(schema=np.array(evaluation.SCHEMA), axis_order=np.array('sample,horizon,node'),
        stage=np.array(stage), registered_arms=np.asarray(arms, dtype=str),
        y=y, valid=np.ones_like(y, dtype=bool), sample_ids=np.arange(100,105),
        node_ids=np.arange(4), road_labels=np.array(['A','A','B','B']),
        direct_support=np.array([[True,False,False,False]]*5))
    for index,case in enumerate((*evaluation.HISTORICAL_CASES,*arms)):
        archive['pred__'+case] = y+float(index+1)
    return archive


class GraphMemoryEvaluationTests(unittest.TestCase):
    def test_archive_registration_accepts_optional_arms_and_rejects_missing_or_extra_predictions(self):
        archive = archive_fixture(('JOINT_MEMORY','FACTORIZED_MEMORY'))
        evaluation.validate_archive(archive)
        broken = deepcopy(archive)
        broken['pred__RECURRENT_CONTROL'] = broken['y'].copy()
        with self.assertRaises(ValueError):
            evaluation.validate_archive(broken)
        broken = deepcopy(archive)
        broken['registered_arms'] = np.array(['JOINT_MEMORY','JOINT_MEMORY'])
        with self.assertRaises(ValueError):
            evaluation.validate_archive(broken)
        broken = deepcopy(archive)
        broken.pop('pred__FACTORIZED_MEMORY')
        with self.assertRaises(ValueError):
            evaluation.validate_archive(broken)
        broken = deepcopy(archive)
        broken['stage'] = np.array('unknown')
        with self.assertRaises(ValueError):
            evaluation.validate_archive(broken)
        with self.assertRaises(ValueError):
            evaluation.validate_archive(archive_fixture(stage='probe'))

    def test_report_preserves_negative_gains_distinct_fixed_hard_sets_and_all_twelve_leads(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)/'report'
            archive = archive_fixture()
            archive['pred__GW2_BASE'][4] += 100.
            archive['pred__RECEIVER_RESIDUAL'][3] += 100.
            result = evaluation.report(archive, output, dict(engineering_only=True))
            self.assertEqual(result['status'],'GW6_EVALUATION_COMPLETE')
            self.assertLess(result['comparisons']['JOINT_MEMORY_vs_RECURRENT_CONTROL']['mae_gain_pooled'],0.)
            self.assertEqual(result['additional_fixed_hard_ids']['GW2_BASE'],[104])
            self.assertEqual(result['additional_fixed_hard_ids']['RECEIVER_RESIDUAL'],[103])
            self.assertEqual(len(result['cases']['JOINT_MEMORY']['horizons']),12)
            self.assertEqual(result['comparisons']['JOINT_MEMORY_vs_RECURRENT_CONTROL']['comparison_kind'],
                'matched_new_training')
            historical = result['comparisons']['JOINT_MEMORY_vs_historical_RECEIVER_RESIDUAL']
            self.assertEqual(historical['comparison_kind'],'historical_reference')
            self.assertFalse(result['historical_GW4_GW5_are_equal_budget_controls'])
            self.assertTrue((output/'cancellation_and_bias.csv').is_file())
            self.assertIn('普通加法 query', (output/'report.md').read_text(encoding='utf-8'))

    def test_positive_and_negative_gain_cancellation_and_signed_bias_remain_visible(self):
        with tempfile.TemporaryDirectory() as temporary:
            archive = archive_fixture()
            archive['pred__RECURRENT_CONTROL'] = archive['y']+1.
            archive['pred__JOINT_MEMORY'] = archive['y']+np.array([1.5,1.5,.5,.5],np.float32)
            result = evaluation.report(archive, Path(temporary)/'out', dict(engineering_only=True))
            pair = result['comparisons']['JOINT_MEMORY_vs_RECURRENT_CONTROL']
            self.assertAlmostEqual(pair['mae_gain_pooled'],0.)
            self.assertAlmostEqual(pair['positive_gain'],.25)
            self.assertAlmostEqual(pair['negative_gain'],.25)
            self.assertAlmostEqual(pair['cancellation_fraction'],1.)
            self.assertAlmostEqual(result['cases']['JOINT_MEMORY']['signed_bias'],1.)
            self.assertAlmostEqual(pair['prediction_change_mean'],0.)

    def test_archive_only_reanalysis_cannot_certify_confirmation_or_run_inference(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = archive_fixture()
            archive['pred__JOINT_MEMORY'] = archive['y'].copy()
            path = root/'saved.npz'; np.savez(path, **archive)
            before = evaluation.metrics.sha256(path)
            with patch.object(evaluation,'produce',side_effect=AssertionError('No inference')):
                result = evaluation.main(['--archive',str(path),'--output-dir',str(root/'out')])
            self.assertEqual(evaluation.metrics.sha256(path),before)
            self.assertEqual(result['inference_calls_this_analysis'],0)
            self.assertTrue(result['graph_memory_screening']['beats_all_primary_controls_MAE'])
            self.assertFalse(result['graph_memory_screening']['producer_independently_certified'])
            self.assertFalse(result['graph_memory_screening']['candidate_for_confirmation'])
            self.assertFalse((root/'out'/'paired_predictions.npz').exists())

    def test_cache_reuse_requires_identical_architecture_stage_source_and_tensor_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); output = root/'eval_best'; output.mkdir()
            archive = archive_fixture()
            path = output/'paired_predictions.npz'; np.savez(path, **archive)
            identity = dict(identity_sha256='run',source_reference={'GW4_BEST':'certified'},
                stage='joint',arms=list(train.DEFAULT_ARMS))
            role = dict(policy='JOINT_MEMORY',model_tensor_sha256='same-tensors')
            metadata = dict(schema=evaluation.PRODUCER_SCHEMA,status='GW6_PREDICTION_PASS',
                run_identity_sha256='run',source_reference=identity['source_reference'],
                stage='joint',registered_arms=identity['arms'],
                checkpoint_roles={'JOINT_MEMORY':role},validation_replay={'JOINT_MEMORY':dict(passed=True)},
                producer_source=evaluation.producer_source(),archive_sha256=evaluation.metrics.sha256(path))
            metadata_path = output/'metadata.json'
            def save(value):
                metadata_path.write_text(json.dumps(value),encoding='utf-8')
            save(metadata)
            with patch.object(train,'model_for_arm',side_effect=AssertionError('No repeated forward')):
                self.assertIsNotNone(evaluation.existing_cache(root,identity,'JOINT_MEMORY',role,archive))
                self.assertIsNone(evaluation.existing_cache(root,identity,'ADDITIVE_MEMORY',role,archive))
                for key,value in (('stage','probe'),('source_reference',{'GW4_BEST':'changed'})):
                    changed = deepcopy(metadata); changed[key] = value; save(changed)
                    self.assertIsNone(evaluation.existing_cache(root,identity,'JOINT_MEMORY',role,archive))
                changed = deepcopy(metadata)
                changed['checkpoint_roles']['JOINT_MEMORY']['policy'] = 'ADDITIVE_MEMORY'; save(changed)
                self.assertIsNone(evaluation.existing_cache(root,identity,'JOINT_MEMORY',role,archive))
                save(metadata)
                changed_shared = deepcopy(archive); changed_shared['sample_ids'][0] = 999
                with self.assertRaisesRegex(ValueError,'axes'):
                    evaluation.existing_cache(root,identity,'JOINT_MEMORY',role,changed_shared)

    def test_GW5_certificate_rejects_checkpoint_policy_epoch_source_and_replay_tampering(self):
        identity = dict(identity_sha256='GW5',reference={'GW2':'cert'},gw4_reference={'GW4':'cert'})
        states = {arm:dict(best_epoch=3,completed_epoch=5,best_checkpoint_sha256=arm+'best',
            _last_file_sha256=arm+'last',_best_model_sha256=arm+'tensor',
            _last_model_sha256=arm+'lasttensor') for arm in reference.gw5.ARMS}
        metadata = dict(schema='gw5_history_anchor_producer_v1',status='GW5_PREDICTION_PASS',
            run_identity_sha256='GW5',reference=identity['reference'],gw4_reference=identity['gw4_reference'],
            common_completed_adaptation_epochs=5,engineering_only=False,
            checkpoint_roles={arm:dict(selection='best',epoch=3,policy=arm,
                checkpoint_sha256=arm+'best',model_tensor_sha256=arm+'tensor') for arm in states},
            validation_replay={arm:dict(passed=True) for arm in states},
            producer_source={name:reference.gw5.base.sha256(reference.gw5.REPO/'experiments/chronological'/name)
                for name in reference.evaluation.PRODUCERS})
        reference.validate_certificate(metadata,identity,states,'best',5)
        arm = reference.gw5.ARMS[0]
        for field,value in (('policy','wrong-arm'),('selection','last'),('epoch',4),
                            ('checkpoint_sha256','changed'),('model_tensor_sha256','changed')):
            changed = deepcopy(metadata); changed['checkpoint_roles'][arm][field] = value
            with self.assertRaises(ValueError):
                reference.validate_certificate(changed,identity,states,'best',5)
        changed = deepcopy(metadata); changed['validation_replay'][arm]['passed'] = False
        with self.assertRaises(ValueError):
            reference.validate_certificate(changed,identity,states,'best',5)
        changed = deepcopy(metadata); changed['producer_source'][reference.evaluation.PRODUCERS[0]] = 'bad'
        with self.assertRaises(ValueError):
            reference.validate_certificate(changed,identity,states,'best',5)

    def test_certification_returns_receiver_BEST_tensors_instead_of_GW2_initialization(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); gw5run = root/'GW5'; gw4run = root/'GW4'
            def digest(path):
                return 'digest:'+str(Path(path).resolve())
            old_identity = dict(identity_sha256='GW4-id')
            identity = dict(schema=reference.gw5.SCHEMA,design=reference.gw5.DESIGN,engineering_only=False,
                identity_sha256='GW5-id',execution_source={},
                gw4_reference=dict(source_run=str(gw4run),completed_adaptation_epochs=5),
                data=dict(splits=dict(train=dict(partition=[0,1]))))
            receiver_path = gw4run/'RECEIVER_RESIDUAL'/'best.pt'
            receiver_weights = {'receiver':torch.tensor([7.])}
            discarded_GW2_weights = {'GW2':torch.tensor([1.])}
            gw4_states = {arm:dict(completed_epoch=5,best_epoch=2,best_checkpoint_file='best.pt',
                best_checkpoint_sha256=digest(gw4run/arm/'best.pt'),_last_file_sha256=digest(gw4run/arm/'last.pt'),
                _best_model_sha256=reference.gw5.base.tensor_tree_hash(receiver_weights)) for arm in train.gw4.ARMS}
            states = {arm:dict(completed_epoch=5,best_epoch=3,best_checkpoint_file='best.pt',
                best_checkpoint_sha256=digest(gw5run/arm/'best.pt'),model_state={'new':torch.tensor([2.])})
                for arm in reference.gw5.ARMS}
            def read(path):
                path = Path(path)
                if path.name == 'identity.json':
                    return identity
                if path.name == 'budget.json':
                    return dict(target_completed_epochs=5)
                return identity['gw4_reference']
            def load(path):
                path = Path(path)
                if path.resolve() == receiver_path.resolve():
                    return receiver_weights
                if path.name == 'last.pt':
                    return deepcopy(states[path.parent.name])
                return {'newbest':torch.tensor([3.])}
            with patch.object(reference,'read_json',side_effect=read), \
                patch.object(reference,'verify_identity_digest'), \
                patch.object(reference,'compatible'), \
                patch.object(reference,'recertify_gw4',return_value=(old_identity,discarded_GW2_weights,gw4_states)), \
                patch.object(reference,'find_evaluation',return_value={'directory':'certified-eval'}), \
                patch.object(reference.gw5,'EXECUTION_FILES',()), \
                patch.object(reference.gw5,'validate_saved'), \
                patch.object(reference.gw5.base,'sha256',side_effect=digest), \
                patch.object(reference.gw5.base,'load_checkpoint',side_effect=load):
                certificate, _, weights, _ = reference.certify(gw5run)
            self.assertIs(weights,receiver_weights)
            self.assertEqual(certificate['source_initialization']['selection'],'best')
            self.assertEqual(certificate['source_initialization']['arm'],'RECEIVER_RESIDUAL')
            self.assertEqual(certificate['source_initialization']['checkpoint_sha256'],digest(receiver_path))
            self.assertEqual(certificate['source_initialization']['model_tensor_sha256'],
                reference.gw5.base.tensor_tree_hash(receiver_weights))
            self.assertFalse(certificate['historical_controls_are_new_budget_matched'])

    def test_recertification_rejects_replaced_GW4_BEST_initialization(self):
        saved = dict(source_run='GW5',archives={key:dict(directory=key) for key in ('best','last')},
            source_initialization=dict(model_tensor_sha256='original'))
        changed = deepcopy(saved); changed['source_initialization']['model_tensor_sha256'] = 'replaced'
        with patch.object(reference,'certify',return_value=(changed,{}, {}, {})):
            with self.assertRaisesRegex(ValueError,'GW4 BEST initialization changed'):
                reference.recertify(saved)


if __name__ == '__main__':
    unittest.main()
