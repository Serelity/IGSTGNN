"""Frozen selected-checkpoint inference, provenance and export recovery tests."""

from contextlib import ExitStack, redirect_stdout
import copy
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.chronological import audit_vector_correction_geometry as e
from experiments.chronological import train_vector_output_scope as producer
from test_incident_strength_gate import SyntheticDataset, manifests, tiny_model


def fixture(root):
    """Synthetic frozen outputs; creates no optimizer and performs no training."""
    torch.set_num_threads(1)
    protocol = e.load_protocol()
    inherited, frozen = producer.base.load_protocol(), producer.load_protocol()
    source, data, first, second, weights = [root / name for name in ('source', 'data', 'first', 'second', 'weights')]
    for path in (source, data, first, second, weights): path.mkdir()
    rows = manifests(); plan = producer.base.make_plan(*rows, inherited)
    protocol['expected_phase_samples'] = {p: {c: len(v) for c, v in item['indices'].items()} for p, item in plan.items()}
    for directory, filename, values in zip((data, first, second),
            ('train_manifest.csv','train_control_manifest.csv','train_second_control_manifest.csv'), rows):
        with (directory/filename).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0])); writer.writeheader(); writer.writerows(values)
    model = tiny_model(); checkpoint = weights / 'best_model.pt'; torch.save(model.state_dict(), checkpoint)
    hashes = {str(p): e.sha256(p) for p in (checkpoint, data/'train_manifest.csv', first/'train_control_manifest.csv',
                                          second/'train_second_control_manifest.csv')}
    for i in range(15):
        p = data / f'input_{i}.bin'; p.write_bytes(str(i).encode()); hashes[str(p)] = e.sha256(p)
    baseline = {'checkpoint': {'parameters': sum(p.numel() for p in model.parameters())}}
    ds = SyntheticDataset()
    with torch.no_grad():
        ds.targets = (tiny_model()(torch.from_numpy(ds.x), incident_data={k:torch.from_numpy(v) for k,v in ds.incident.items()})*5+20).numpy()
    datasets = {c:ds for c in producer.base.COHORTS}
    training = copy.deepcopy(inherited['training'])
    training['objective'] = 'v12m shared candidate_early trajectory; unchanged vector energy penalty'
    identity = {'inputs': hashes, 'code_sha256': protocol['source']['producer_code_sha256'],
        'protocol_sha256': producer.PROTOCOL_SHA256, 'engineering_check': False,
        'indices': {p:item['indices'] for p,item in plan.items()}, 'seeds':protocol['seeds'], 'effective_training':training,
        'output_policies': frozen['policies'], 'output_paths':frozen['output_paths'], 'checkpoint_sha256':e.sha256(checkpoint)}
    e.write_json(source/'run_identity.json',identity); e.write_json(source/'eligibility.json',plan)
    (source/'A').mkdir(); references={}
    for phase,item in plan.items():
        references[phase]={}
        for cohort,indices in item['indices'].items():
            record=producer.evaluate(tiny_model(),ds,indices,16,'cpu')
            references[phase][cohort]=record
            producer.base.save_arrays(source/'A'/f'{phase}_{cohort}.npz',record)
    summary={'status':protocol['source']['required_status'], 'engineering_check':False,
        'protocol_sha256':producer.PROTOCOL_SHA256,'frozen_protocol':frozen,'inherited_protocol':inherited,
        'code_sha256':protocol['source']['producer_code_sha256'],'inputs':hashes,'effective_training':training,
        'all_selectors_frozen_before_audit_evaluation':True,'all_output_paths_frozen_before_audit_evaluation':True,
        'paired_initialization_exact':True,'unrestricted_at_protected_evaluated':False,'main_training_ready':False,
        **frozen['information_boundary'],'environment':{'git_head':protocol['source']['git_head']},
        'phase_samples':protocol['expected_phase_samples'],
        'budget':{'fits':6,'trajectory_epochs':72,'selected_endpoints':12,'output_paths':18},
        'runs':{},'baseline':{p:{c:producer.metrics(r) for c,r in rs.items()} for p,rs in references.items()}}
    endpoints,paths={},{}
    for seed in protocol['seeds']:
        summary['runs'][str(seed)]={}
        for arm in protocol['arms']:
            name=f'{arm}_s{seed}'; directory=source/name; directory.mkdir(); (directory/e.PATH).mkdir()
            model=tiny_model(); native_hash=producer.base.backbone_hash(model)
            adapter=producer.vector.attach_adapter(model,arm,16)
            initial=producer.alignment.tensor_hash(adapter.state_dict())
            with torch.no_grad():
                adapter.output.bias.copy_(torch.linspace(-.05,.07,adapter.output.bias.numel()))
            model.requires_grad_(False).eval()
            state=producer.base.cpu_tree(adapter.state_dict()); digest=producer.alignment.tensor_hash(state)
            metrics={}
            for phase,item in plan.items():
                metrics[phase]={}
                for cohort,indices in item['indices'].items():
                    record=producer.evaluate_policies(model,ds,indices,16,'cpu',policies=(e.POLICY,))[e.POLICY]
                    producer.base.save_arrays(directory/e.PATH/f'{phase}_{cohort}.npz',record)
                    metrics[phase][cohort]=producer.metrics(record)
            header={'arm':arm,'loss':'candidate_early','seed':seed,'protocol_sha256':producer.PROTOCOL_SHA256,
                'run_identity_sha256':e.sha256(source/'run_identity.json'),'backbone_state_sha256':native_hash,
                'initial_adapter_sha256':initial,'output_policies':frozen['policies'],
                'fit_samples':len(plan['fit']['indices']['incident_full']),'training':training}
            epoch=protocol['source']['selected_epochs'][arm][str(seed)]
            policies={}
            for policy in frozen['policies']:
                key=f'{name}/selected_{policy}.pt'
                torch.save({'identity':header,'policy':policy,'epoch':epoch,'adapter_state':state,
                    'adapter_state_sha256':digest,'selection_metrics':metrics['selection']},source/key)
                endpoints[key]=e.sha256(source/key)
                policies[policy]={'selected_epoch':epoch,'selection_metrics':metrics['selection'],
                    'adapter_state_sha256':digest,'checkpoint_sha256':endpoints[key]}
            detail={'arm':arm,'loss':'candidate_early','seed':seed,'epochs':12,
                'trainable_parameters':sum(p.numel() for p in adapter.parameters()),'backbone_state_sha256':native_hash,
                'initial_adapter_sha256':initial,'policies':policies}
            e.write_json(directory/'fit_summary.json',detail)
            summary['runs'][str(seed)][name]=copy.deepcopy(detail)
            output_paths={}
            for path,policy in zip(frozen['output_paths'],('unrestricted','unrestricted',e.POLICY)):
                key=f'{name}/selected_{policy}.pt'
                item={'source_checkpoint':key,'source_checkpoint_sha256':endpoints[key],
                    'selected_policy':policy,'output_policy':'unrestricted' if path=='unrestricted_at_unrestricted' else e.POLICY,
                    'epoch':epoch,'adapter_state_sha256':digest}
                paths[f'{name}/{path}']=item
                output_paths[path]={**item,'phase_metrics':metrics}
            summary['runs'][str(seed)][name]['output_paths']=output_paths
    for phase,bounds in protocol['periods'].items():
        weeks, draws=e.geometry.calendar(bounds,protocol['bootstrap'])
        for seed in protocol['seeds']:
            np.savez_compressed(source/f'{phase}_weekly_s{seed}.npz',weeks=weeks,**{k+'_weights':v for k,v in draws.items()})
    e.write_json(source/'selected_endpoints_frozen.json',endpoints)
    e.write_json(source/'evaluation_paths_frozen.json',paths)
    summary.update(selected_endpoints_frozen=endpoints,evaluation_paths_frozen=paths)
    def publish():
        summary['outputs']={str(p.relative_to(source)):e.sha256(p) for p in source.rglob('*') if p.is_file() and p.name!='summary.json'}
        e.write_json(source/'summary.json',summary)
    publish()
    return protocol,source,data,first,second,checkpoint,ds,baseline,hashes,summary,publish


class AuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.template=tempfile.TemporaryDirectory()
        cls.values=fixture(Path(cls.template.name))

    @classmethod
    def tearDownClass(cls): cls.template.cleanup()

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        (self.protocol,self.source,self.data,self.first,self.second,self.checkpoint,self.ds,
         self.baseline,self.hashes,self.summary,self.publish)=self.values

    def context(self):
        stack=ExitStack()
        stack.enter_context(patch.object(e,'load_protocol',return_value=self.protocol))
        stack.enter_context(patch.object(producer.base.mechanisms,'verify_inputs',return_value=(self.baseline,self.hashes)))
        ds=self.ds
        class Guarded:
            scaler,station_ids=ds.scaler,ds.station_ids
            def __getitem__(inner,index):
                if torch.is_grad_enabled(): raise AssertionError('Y loaded with gradients enabled')
                return ds[index]
        stack.enter_context(patch.object(producer.base,'make_datasets',return_value={c:Guarded() for c in producer.base.COHORTS}))
        stack.enter_context(patch.object(producer.base,'make_model',side_effect=lambda *a,**k:tiny_model()))
        stack.enter_context(patch.object(torch.optim,'Adam',side_effect=AssertionError('Optimizer created')))
        stack.enter_context(patch.object(torch.Tensor,'backward',side_effect=AssertionError('Backward called')))
        stack.enter_context(patch.object(producer.alignment,'early_train_epoch',side_effect=AssertionError('Training called')))
        stack.enter_context(redirect_stdout(io.StringIO()))
        return stack

    def run_audit(self,output,**kwargs):
        with self.context():
            return e.run(self.source,self.data,self.first,self.second,self.checkpoint,output,'cpu',**kwargs)

    def test_full_frozen_workflow_and_statistics_only_recovery(self):
        before={str(p):e.sha256(p) for p in self.source.rglob('*') if p.is_file()}
        output=self.root/'complete'; summary=self.run_audit(output)
        self.assertEqual(summary['status'],'VECTOR_CORRECTION_GEOMETRY_AUDIT_COMPLETE')
        self.assertEqual(len(summary['results']),6)
        self.assertEqual(len(list((output/'signed_cells').glob('*.npz'))),54)
        self.assertEqual(summary['inference_telemetry']['native_forward_batches'],9)
        self.assertEqual(summary['inference_telemetry']['adapted_forward_batches'],54)
        for flag in ('model_training_performed','gradient_computation_performed','optimizer_created',
                     'new_model_selection_performed','lambda_search_performed','validation_arrays_read','test_split_read'):
            self.assertFalse(summary[flag])
        self.assertEqual(before,{str(p):e.sha256(p) for p in self.source.rglob('*') if p.is_file()})
        for name,digest in summary['outputs'].items(): self.assertEqual(e.sha256(output/name),digest)
        with patch.object(e,'load_protocol',return_value=self.protocol),patch.object(torch,'load',side_effect=AssertionError('Model loaded')),redirect_stdout(io.StringIO()):
            resumed=e.analyze_export(output,self.root/'statistics')
        self.assertEqual(resumed['results'],summary['results']); self.assertEqual(resumed['matched'],summary['matched'])
        self.assertFalse(resumed['new_model_inference_this_invocation'])

    def test_source_arrays_and_weekly_draws_tampering(self):
        source=e.Source(self.source,self.protocol);e.verify_arrays(source)
        source.records['audit_weekly_s2025.npz']['week_weights'][0,0]+=1
        with self.assertRaisesRegex(ValueError,'draws'): e.verify_arrays(source)
        source=e.Source(self.source,self.protocol)
        source.records['A/audit_incident_full.npz']=source.array('A/audit_incident_full.npz').copy()
        source.records['A/audit_incident_full.npz']['counts']=np.zeros_like(source.records['A/audit_incident_full.npz']['counts'])
        with self.assertRaises(ValueError): e.verify_arrays(source)

    def test_engineering_source_rejected_before_inference(self):
        summary=copy.deepcopy(self.summary);summary['engineering_check']=True
        original_read=e.Source.read
        def read(source,name,listed=True):
            return json.dumps(summary).encode() if name=='summary.json' else original_read(source,name,listed)
        with patch.object(e.Source,'read',read),self.assertRaisesRegex(ValueError,'Engineering source'):
            e.Source(self.source,self.protocol)
        with self.assertRaisesRegex(ValueError,'only with --check'):
            self.run_audit(self.root/'out',allow_engineering_source=True)

    def test_checkpoint_header_and_nonfinite_tensor_rejected(self):
        source=e.Source(self.source,self.protocol); key='state_vector_s2025/selected_candidate_early_only.pt'
        saved=torch.load(io.BytesIO(source.read(key)),map_location='cpu',weights_only=True)
        for kind in ('seed','shape','nan','epoch'):
            changed=copy.deepcopy(saved)
            if kind=='seed':changed['identity']['seed']=2027
            elif kind=='epoch':changed['epoch']=12
            else:
                tensor=next(iter(changed['adapter_state']))
                if kind=='shape':changed['adapter_state'][tensor]=torch.zeros(1)
                else:changed['adapter_state'][tensor].flatten()[0]=float('nan')
            stream=io.BytesIO();torch.save(changed,stream)
            with patch.object(source,'read',return_value=stream.getvalue()),self.subTest(kind=kind),self.assertRaises(ValueError):
                e.load_selected(tiny_model(),source,'state_vector',2025,producer)

    def test_output_and_symlink_preserved(self):
        path=self.root/'out';path.mkdir();(path/'keep').write_text('preserve')
        with self.assertRaises(FileExistsError):self.run_audit(path)
        self.assertEqual((path/'keep').read_text(),'preserve')
        symlink=self.root/'link';symlink.symlink_to(self.root/'absent')
        with self.assertRaises(ValueError):self.run_audit(symlink)
        with self.assertRaisesRegex(ValueError,'read-only'):self.run_audit(self.source/'out')

    def test_replay_support_error_and_frozen_tolerances(self):
        source=e.Source(self.source,self.protocol);saved=source.array('A/fit_incident_full.npz')
        for kind in ('support','error'):
            changed={k:v.copy() for k,v in saved.items()}
            if kind=='support':changed['ids'][0]+=1
            else:changed['errors'][0,0]+=1
            with self.subTest(kind=kind),self.assertRaises(ValueError):e.replay(changed,saved,self.protocol,{})

    def test_native_predictions_independent_of_Y(self):
        ds=self.ds;indices=[0,1];model=tiny_model()
        _,a,_,_,_=e.infer_cohort(model,ds,indices,producer,'cpu',lambda *a,**k:None)
        altered=copy.copy(ds);altered.targets=ds.targets+1000
        _,b,_,_,_=e.infer_cohort(tiny_model(),altered,indices,producer,'cpu',lambda *a,**k:None)
        np.testing.assert_array_equal(a,b)

    def test_model_state_or_gradient_mutation_rejected(self):
        model=tiny_model();digest=producer.alignment.tensor_hash(model.state_dict())
        with torch.no_grad():
            next(model.parameters()).add_(.1)
            with self.assertRaisesRegex(ValueError,'state changed'):e.assert_frozen(model,digest,producer)
        with self.assertRaisesRegex(ValueError,'no_grad'):e.assert_frozen(tiny_model(),digest,producer)

    def test_export_cannot_omit_endpoint_or_change_sign_data(self):
        output=self.root/'check';self.run_audit(output,check=True)
        path=output/'signed_export_manifest.json';manifest=json.loads(path.read_text())
        original=copy.deepcopy(manifest);manifest['selected_models'].pop(next(iter(manifest['selected_models'])))
        e.write_json(path,manifest)
        with self.assertRaisesRegex(ValueError,'model set'):e.verified_export(output,self.protocol)
        e.write_json(path,original)
        artifact=output/original['entries'][0]['file'];artifact.write_bytes(b'bad npz')
        with self.assertRaisesRegex(ValueError,'hash mismatch'):e.verified_export(output,self.protocol)

    def test_signed_export_required_before_statistics_recovery(self):
        with patch.object(e,'load_protocol',return_value=self.protocol),self.assertRaises(FileNotFoundError):
            e.analyze_export(self.root,self.root/'new')


if __name__=='__main__':unittest.main()
