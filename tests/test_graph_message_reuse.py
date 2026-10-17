"""Server-run tests for cached references and omission of baseline reruns."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experiments.chronological import graph_message_reuse as reuse
from experiments.chronological import evaluate_original_graph_message_reuse as evaluation
from tests.test_original_graph_window_evaluation import fixture as gw1_fixture


class GraphMessageReuseTests(unittest.TestCase):
    def identity(self):
        return dict(training_config={'seed': 2025, 'bs': 48}, data={'sample_ids': [1, 2]},
            source_fingerprint='native', engineering_only=False,
            runtime={key:'same' for key in reuse.RUNTIME_KEYS},
            initializations={arm:dict(original_tensor_sha256='native_init',
                post_construction_rng_sha256='rng') for arm in ('ORIGINAL','STATIC')})

    def test_reuse_requires_matching_recipe_data_runtime_and_native_initialization(self):
        source = self.identity(); target = deepcopy(source)
        reuse.compatible(source, target)
        mutations = [lambda x:x['training_config'].update(seed=2026),
            lambda x:x['data'].update(sample_ids=[2,1]),
            lambda x:x['runtime'].update(torch='different'),
            lambda x:x.update(source_fingerprint='different'),
            lambda x:x.update(engineering_only=True),
            lambda x:x['initializations']['ORIGINAL'].update(original_tensor_sha256='different')]
        for mutate in mutations:
            target = deepcopy(source); mutate(target)
            with self.assertRaises(ValueError):
                reuse.compatible(source, target)

    def test_cache_certificate_rejects_wrong_selection_budget_and_checkpoint(self):
        identity = self.identity(); identity['identity_sha256']='identity'
        states = {a:dict(completed_epoch=5,best_epoch=3,best_checkpoint_sha256=a+'best',
            _last_file_sha256=a+'last') for a in reuse.REUSED_ARMS}
        metadata = dict(schema='original_igstgnn_graph_window_producer_v1',status='GW1_PREDICTION_PASS',
            run_identity_sha256='identity',engineering_only=False,source_fingerprint='native',
            common_completed_epochs=5,
            producer_source={name:reuse.sha256(reuse.gw1.REPO/'experiments/chronological'/name)
                for name in ('evaluate_original_graph_window.py','evaluate_unified_adapter_v03.py')},
            checkpoint_roles={a:dict(selection='best',epoch=3,checkpoint_sha256=a+'best') for a in states},
            validation_replay={a:dict(passed=True) for a in states})
        reuse.validate_archive_certificate(metadata,identity,states,'best')
        for mutate in (lambda m:m.update(common_completed_epochs=100),
            lambda m:m['checkpoint_roles']['ORIGINAL'].update(epoch=2),
            lambda m:m['validation_replay']['STATIC'].update(passed=False)):
            changed = deepcopy(metadata); mutate(changed)
            with self.assertRaises(ValueError):
                reuse.validate_archive_certificate(changed,identity,states,'best')
        with self.assertRaises(ValueError):
            reuse.validate_archive_certificate(metadata,identity,states,'last')

    def test_source_identity_digest_is_verified_without_model_execution(self):
        import hashlib, json
        identity = self.identity()
        identity['identity_sha256']=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
        reuse.verify_identity_digest(identity)
        identity['training_config']['seed']=0
        with self.assertRaises(ValueError):
            reuse.verify_identity_digest(identity)

    def test_reuse_report_seven_cases_does_not_invent_static_uniform(self):
        with tempfile.TemporaryDirectory() as temp:
            source = gw1_fixture()
            # Keep the inherited numerical fixture axes and explicit mask.
            archive = {k:v for k,v in source.items() if not k.startswith('pred__')}
            archive['schema']=np.array(evaluation.SCHEMA)
            for index, case in enumerate(evaluation.CASES):
                archive['pred__'+case]=archive['y']+index+1.
            result=evaluation.base.report(archive,Path(temp)/'out',
                dict(engineering_only=True,reuse_reference={'source_run':'fixture'}),
                evaluation.CASES,evaluation.COMPARISONS,evaluation.SCHEMA)
            self.assertEqual(len(result['cases']),7)
            self.assertNotIn('STATIC_UNIFORM',result['cases'])
            self.assertEqual(result['comparisons']['POST_MESSAGE_vs_ORIGINAL']['mae_gain_pooled'],-3.)
            self.assertEqual(result['fixed_hard_ids'],[100])

    def test_cached_metric_path_never_builds_or_calls_a_model(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'paired_predictions.npz'
            source=gw1_fixture(); np.savez(path,**source)
            reference={'archives':{'best':{'path':str(path),'sha256':reuse.sha256(path)}}}
            identity={'data':{'splits':{'val':{'selected_indices':source['sample_ids'].tolist()}}}}
            modules=evaluation.base.get_original_modules()
            states={a:dict(best_epoch=1,completed_epoch=1,history=[{
                'validation_'+k:v for k,v in evaluation.base.native_metrics(source['pred__'+a],source['y'],modules).items()}])
                for a in reuse.REUSED_ARMS}
            with patch.object(evaluation.train,'model_for_arm',side_effect=AssertionError('baseline inference forbidden')):
                archive, checks=evaluation.cached_predictions(reference,'best',identity,states,modules)
            for arm in reuse.REUSED_ARMS:
                self.assertTrue(np.array_equal(archive['pred__'+arm],source['pred__'+arm]))
                self.assertTrue(checks[arm]['passed'])
                self.assertEqual(checks[arm]['new_inference_calls'],0)
            identity['data']['splits']['val']['selected_indices'].reverse()
            with self.assertRaises(ValueError):
                evaluation.cached_predictions(reference,'best',identity,states,modules)

    def test_completed_new_predictions_are_reused_without_model_construction(self):
        import json
        with tempfile.TemporaryDirectory() as temp:
            run=Path(temp); directory=run/'eval_best_old'; directory.mkdir()
            source=gw1_fixture()
            archive={k:v for k,v in source.items() if not k.startswith('pred__')}
            archive['schema']=np.array(evaluation.SCHEMA)
            for case in evaluation.CASES:
                archive['pred__'+case]=source['y']+1
            path=directory/'paired_predictions.npz';np.savez(path,**archive)
            metadata=dict(status='GW2_REUSE_PREDICTION_PASS',run_identity_sha256='runid',
                archive_sha256=reuse.sha256(path),
                checkpoint_roles={'POST_MESSAGE':dict(selection='best',epoch=3,checkpoint_sha256='checkpoint')},
                validation_replay={'POST_MESSAGE':dict(passed=True)})
            (directory/'metadata.json').write_text(json.dumps(metadata),encoding='utf-8')
            state=dict(best_epoch=3,best_checkpoint_sha256='checkpoint')
            with patch.object(evaluation.train,'model_for_arm',side_effect=AssertionError('no inference')):
                cached=evaluation.existing_new_cache(run,{'identity_sha256':'runid'},state,'POST_MESSAGE','best',archive)
            self.assertIsNotNone(cached)
            metadata['checkpoint_roles']['POST_MESSAGE']['epoch']=2
            (directory/'metadata.json').write_text(json.dumps(metadata),encoding='utf-8')
            self.assertIsNone(evaluation.existing_new_cache(run,{'identity_sha256':'runid'},state,'POST_MESSAGE','best',archive))


if __name__ == '__main__':
    unittest.main()
