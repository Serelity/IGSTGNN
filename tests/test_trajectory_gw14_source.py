"""Server-only provenance rejection tests; no source model is executed."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from experiments.branch_readout_gw9 import readout_auth as auth
from experiments.trajectory_diagnostic_gw13 import model, protocol, run
from experiments.trajectory_check_gw14 import source


def records(phase):
    return {arm:dict(valid_count=100,mae=value,rmse=value+1.,mape_fraction=.2,signed_bias=-.1)
            for arm,value in source.PINNED_MAES[phase].items()}


class SourceTests(unittest.TestCase):
    def test_reduction_replay_accepts_only_small_numeric_reduction_differences(self):
        saved = dict(records=dict(valid_count=100,mae=13.,bias=-.2,empty=None))
        actual = dict(records=dict(valid_count=np.int64(100),mae=np.float64(13.+1e-11),bias=-.2,empty=None))
        source.assert_metrics(actual,saved,'fixture')
        actual['records']['mae'] = 13.+1e-7
        with self.assertRaises(ValueError): source.assert_metrics(actual,saved,'fixture')

    def test_counts_keys_nonfinite_and_boolean_metrics_cannot_replay(self):
        good = dict(valid_count=10,mae=13.,empty=None)
        for key,value in (('valid_count',11),('valid_count',10.),('valid_count',True),
                          ('mae',True),('mae',np.nan),('mae',np.inf),('empty',0)):
            changed = dict(good,**{key:value})
            with self.subTest(key=key,value=value), self.assertRaises(ValueError):
                source.assert_metrics(changed,good,'fixture')
        with self.assertRaises(ValueError): source.assert_metrics(dict(mae=13.),good,'fixture')

    def test_nonfinite_json_and_indirect_source_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root/'bad.json').write_text('{"mae":NaN}',encoding='utf-8')
            with self.assertRaises(ValueError): source.read_json(root/'bad.json')
            with self.assertRaises(ValueError): source._child(root,'missing.json')
            nested = root/'nested'
            nested.mkdir()
            (nested/'value.json').write_text('{}',encoding='utf-8')
            with self.assertRaises(ValueError): source._child(root,'nested/value.json')

    def test_immutable_hash_guard_detects_a_source_edit(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name)/'old_model.json'
            path.write_text('{"unchanged":true}',encoding='utf-8')
            reference = dict(digests={str(path):auth.sha256(path)})
            source.assert_unchanged(reference)
            path.write_text('{"unchanged":false}',encoding='utf-8')
            with self.assertRaises(ValueError): source.assert_unchanged(reference)

    def _fixture(self, root):
        certificate = dict(data_identity={'fixture':True})
        cache_digests = {str(root/'cache'/'cache.json'):'c'*64}
        history = dict(file_digests={})
        execution = {str(run.REPO/name):auth.sha256(run.REPO/name) for name in run.SOURCE_FILES}
        inputs = dict(cache_digests,**execution)
        for name in ('audit_model.json','model.json'):
            (root/name).write_text('{}\n',encoding='utf-8')
        declared = dict(schema=run.SCHEMA,method=run.METHOD,model=model.PROTOCOL,
            source_identity_sha256=auth.SOURCE_IDENTITY_SHA256,cache_digests=inputs,
            execution_source=execution,expected_data_identity=certificate['data_identity'])
        (root/'protocol.json').write_text(json.dumps(declared),encoding='utf-8')
        audit_records, validation_records = records('audit'), records('validation')
        coverage = {arm:dict(active_fraction=.5,eligible_cells=10,observed_cells=20)
                    for arm in protocol.COVERAGE_ARMS}
        groups = {arm:{group:dict(valid_count=20,mae=row['mae'],signed_bias=-.1)
                       for group in protocol.GROUPS} for arm,row in validation_records.items()}
        audit_decision = protocol.audit_screen(audit_records,coverage)
        failed = protocol.validation_screen(validation_records,groups,audit_decision)
        self.assertFalse(failed['trajectory_information_screen_passed'])
        analysis = dict(schema=run.SCHEMA,method=run.METHOD,model_protocol=model.PROTOCOL,
            source_identity_sha256=auth.SOURCE_IDENTITY_SHA256,source_certificate=certificate,
            history_metadata=history,neural_inference_calls=0,neural_optimizer_steps=0,test_evaluated=False,
            input_digests=inputs,cache_metadata_sha256='c'*64,
            audit_model_sha256=auth.sha256(root/'audit_model.json'),model_sha256=auth.sha256(root/'model.json'),
            audit=dict(records=audit_records,coverage=coverage),audit_decision=audit_decision,
            validation=dict(records=validation_records,groups=groups),screening=failed)
        files = {'analysis.json':analysis,'audit_decision.json':audit_decision,'screening.json':failed,
            'fitted_before_validation.json':dict(model_sha256=analysis['model_sha256'],
                protocol_sha256=auth.sha256(root/'protocol.json'),validation_labels_used_for_fit=False)}
        for filename,value in files.items():
            (root/filename).write_text(json.dumps(value),encoding='utf-8')
        return certificate,cache_digests,history,analysis

    def test_changed_fitted_table_bytes_are_rejected_before_metric_replay(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            certificate,digests,history,_ = self._fixture(root)
            (root/'model.json').write_text('{"changed":true}',encoding='utf-8')
            with mock.patch.object(run,'score',side_effect=AssertionError('must not replay')):
                with self.assertRaisesRegex(ValueError,'model bytes'):
                    source.authenticate(root,certificate,digests,{},history)

    def test_source_failure_cannot_be_relabelled_as_passed(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            certificate,digests,history,analysis = self._fixture(root)
            changed = deepcopy(analysis['screening'])
            changed['trajectory_information_screen_passed'] = True
            (root/'screening.json').write_text(json.dumps(changed),encoding='utf-8')
            with mock.patch.object(run,'score',side_effect=AssertionError('must not replay')):
                with self.assertRaisesRegex(ValueError,'failed GW13 screen'):
                    source.authenticate(root,certificate,digests,{},history)

    def test_changed_history_or_source_identity_is_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            certificate,digests,history,analysis = self._fixture(root)
            for key,value in (('source_identity_sha256','0'*64),('history_metadata',{}),
                              ('test_evaluated',True),('method',{})):
                modified = dict(analysis,**{key:value})
                (root/'analysis.json').write_text(json.dumps(modified),encoding='utf-8')
                with self.subTest(key=key), self.assertRaisesRegex(ValueError,'immutable GW13'):
                    source.authenticate(root,certificate,digests,{},history)

    def test_changed_initial_cache_bindings_are_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            certificate,digests,history,_ = self._fixture(root)
            declared = source.read_json(root/'protocol.json')
            declared['cache_digests'] = {}
            (root/'protocol.json').write_text(json.dumps(declared),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'cache execution provenance'):
                source.authenticate(root,certificate,digests,{},history)


if __name__ == '__main__':
    unittest.main()
