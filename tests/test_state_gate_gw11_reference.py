"""Server-only provenance tests; synthetic lookup records are not accuracy evidence."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experiments.state_gate_gw11 import reference as ref


def write(path,value):
    Path(path).write_text(json.dumps(value,allow_nan=False),encoding='utf-8')


class GW10Fixture:
    def __init__(self,root):
        root=Path(root);self.directory=root/'gw10';self.cache_dir=root/'cache'
        self.directory.mkdir();self.cache_dir.mkdir()
        self.certificate=dict(source_identity_sha256='synthetic-source')
        self.metadata=dict(runtime={'numpy':'synthetic'},_directory=str(self.cache_dir))
        self.digests={}
        for name in ('cache.json','train.npz','val.npz'):
            path=self.cache_dir/name;path.write_bytes(name.encode());self.digests[str(path)]=ref.auth.sha256(path)
        y=np.full((3,12,2),400.,dtype=np.float32)
        self.validation=dict(y=y,pred=y+10.,valid=np.ones(y.shape,dtype=bool),features={})
        self.corrections={arm:np.full(y.shape,-1.,dtype=np.float64) for arm in ref.median_probe.ARMS}
        self.corrections.update(ORIGIN=np.full(y.shape,-2.),OWN4=np.full(y.shape,-3.),JOINT=np.full(y.shape,-2.5))
        predictions={'SUM':self.validation['pred'].astype(np.float64)}
        predictions.update({arm:predictions['SUM']+value for arm,value in self.corrections.items()})
        self.records=ref.gw10.records(predictions,self.validation)
        masks=dict(high_value_ge314=y>=314,long_horizon_7_12=np.arange(12)[None,:,None]>=6,
            flow_50_200=(y>=50)&(y<200))
        groups={name:ref.gw10.records(predictions,self.validation,mask) for name,mask in masks.items()}
        decision=ref.screening.screen(self.records,self.records,groups)
        protocol=dict(schema=ref.gw10.SCHEMA,protocol=ref.gw10.PROTOCOL,
            source_identity_sha256=self.certificate['source_identity_sha256'],input_digests=self.digests)
        write(self.directory/'protocol.json',protocol)
        write(self.directory/'model.json',dict(synthetic=True));write(self.directory/'audit_model.json',dict(audit=True))
        model_hash=ref.auth.sha256(self.directory/'model.json')
        self.analysis=dict(schema=ref.gw10.SCHEMA,protocol=ref.gw10.PROTOCOL,source_certificate=self.certificate,
            cache_metadata_sha256=self.digests[str(self.cache_dir/'cache.json')],cache_runtime=self.metadata['runtime'],
            model_updates=0,SUM_head_updates=0,validation_fits=0,test_evaluated=False,screening=decision,
            model_sha256=model_hash,audit_model_sha256=ref.auth.sha256(self.directory/'audit_model.json'),
            validation=self.records,training_order_audit=self.records,groups=groups)
        write(self.directory/'analysis.json',self.analysis);write(self.directory/'screening.json',decision)
        write(self.directory/'fitted_before_validation.json',dict(model_sha256=model_hash,
            protocol_sha256=ref.auth.sha256(self.directory/'protocol.json'),validation_labels_used_for_fit=False))

    def certify(self):
        pins={arm:self.records[arm]['mae'] for arm in ('ORIGIN','OWN4','JOINT')}
        with patch.object(ref,'PINNED_VALIDATION_MAE',pins),patch.object(ref.median_probe,'load_model',return_value={}), \
                patch.object(ref.median_probe,'predict_corrections',return_value=self.corrections):
            return ref.certify_gw10(self.directory,{},self.certificate,{'val':self.validation},self.metadata,self.digests)


class GW11ReferenceTests(unittest.TestCase):
    def test_completed_failed_gw10_replays_without_promoting_own4_to_source(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture=GW10Fixture(temp);result=fixture.certify()
            self.assertEqual(result['source_base'],'GW9_SUM_CONTROL_BEST')
            self.assertFalse(result['historical_controls_promoted_to_base'])
            self.assertFalse(result['eligible_for_new_model_design'])
            self.assertEqual(result['validation_records'],fixture.records)
            self.assertEqual(result['new_lookup_fits'],0)
            self.assertIn(str(fixture.directory/'model.json'),result['file_digests'])

    def test_model_bytes_or_reported_validation_change_is_rejected(self):
        for kind in ('model','reported_mae'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as temp:
                fixture=GW10Fixture(temp)
                if kind=='model':(fixture.directory/'model.json').write_bytes(b'changed')
                else:
                    altered=deepcopy(fixture.analysis);altered['validation']['OWN4']['mae']+=1.
                    write(fixture.directory/'analysis.json',altered)
                with self.assertRaises(ValueError):fixture.certify()

    def test_failed_gate_and_bound_protocol_cannot_be_replaced(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture=GW10Fixture(temp);screen_path=fixture.directory/'screening.json'
            decision=json.loads(screen_path.read_text());decision['eligible_for_new_model_design']=True;write(screen_path,decision)
            with self.assertRaisesRegex(ValueError,'failed fixed'):fixture.certify()

    def test_pins_are_applied_to_actual_recomputed_validation_mae(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture=GW10Fixture(temp)
            with patch.object(ref.median_probe,'load_model',return_value={}), \
                    patch.object(ref.median_probe,'predict_corrections',return_value=fixture.corrections), \
                    self.assertRaisesRegex(ValueError,'returned GW10'):
                ref.certify_gw10(fixture.directory,{},fixture.certificate,{'val':fixture.validation},fixture.metadata,fixture.digests)

    def test_changed_source_after_certification_is_detected(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture=GW10Fixture(temp);result=fixture.certify()
            (fixture.directory/'protocol.json').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'source changed'):ref.assert_unchanged(result)

    def test_identical_exporter_in_another_checkout_is_allowed_with_actual_hash_checks(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);relative=ref.gw10.export_cache.SOURCE_FILES[0]
            old=root/'old'/relative;new=root/'new'/relative
            old.parent.mkdir(parents=True);new.parent.mkdir(parents=True)
            old.write_bytes(b'identical code');new.write_bytes(b'identical code')
            digest=ref.auth.sha256(new);digests={str(new):digest}
            ref.verify_cache_bindings({str(old):digest},{str(new):digest},digests)
            self.assertIn(str(old.resolve()),digests)
            old.write_bytes(b'altered old code')
            with self.assertRaisesRegex(ValueError,'digest differs'):
                ref.verify_cache_bindings({str(old):digest},{str(new):digest},{})

    def test_cache_artifact_paths_cannot_use_checkout_relocation_exception(self):
        with self.assertRaises(ValueError):ref.verify_cache_bindings({'/old/train.npz':'a'},{'/new/train.npz':'a'},{})

    def test_fresh_output_must_protect_all_source_roots_and_resolved_symlinks(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);roots=[root/name for name in ('probe','cache','gw10','data')]
            for path in roots:path.mkdir()
            certificate=dict(protected_directories=[str(roots[0])])
            for source in roots:
                with self.assertRaises(ValueError):
                    ref.require_output_outside_sources(source/'new',roots[0],certificate,roots[1],roots[2],roots[3])
            result=ref.require_output_outside_sources(root/'new',roots[0],certificate,roots[1],roots[2],roots[3])
            self.assertFalse(result.exists())

    def test_load_inputs_returns_six_values_and_keeps_cache_metadata_unchanged(self):
        metadata=dict(runtime={});certificate={'source_identity_sha256':'x'}
        with patch.object(ref,'_export_repo_from_cache',return_value=Path('old_repo')), \
                patch.object(ref.auth,'load_authenticated_run',return_value=({}, {}, certificate)) as authenticating, \
                patch.object(ref.gw10,'load_cache',return_value=({'train':{},'val':{}},metadata,{})), \
                patch.object(ref,'certify_gw10',return_value={'schema':ref.SCHEMA}), \
                patch.object(ref.auth,'assert_unchanged'):
            result=ref.load_inputs('probe','cache','gw10')
        self.assertEqual(len(result),6);self.assertIs(result[3],metadata);self.assertNotIn('_directory',metadata)
        authenticating.assert_called_once_with('probe',repo=Path('old_repo'))

    def test_recorded_execution_checkout_is_located_without_rewriting_certificate(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);cache=root/'cache';cache.mkdir()
            source=root/'old_repo'/'experiments'/'branch_readout_gw9'/'probe.py'
            write(cache/'cache.json',dict(source_certificate=dict(file_digests={str(source):'pinned-by-auth'})))
            self.assertEqual(ref._export_repo_from_cache(cache),(root/'old_repo').resolve())
            write(cache/'cache.json',dict(source_certificate=dict(file_digests={})))
            with self.assertRaises(ValueError):ref._export_repo_from_cache(cache)


if __name__=='__main__':unittest.main()
