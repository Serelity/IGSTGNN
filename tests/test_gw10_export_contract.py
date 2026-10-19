"""Server-only NumPy export contracts; no local execution or forecasting tests."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from experiments.conditional_residual_gw10 import export_cache as export


def replay_fixture():
    y=np.full((2,12,3),20.,dtype=np.float32);y[0,0,0]=0.
    actual=dict(y=y,valid=y!=0.,sample_ids=np.array([3909,3910],dtype=np.int64),
        node_ids=np.array([10,20,30],dtype=np.int64),direct_support=np.zeros((2,3),dtype=bool),
        frozen_R=y+3.,pred=y+2.)
    expected={name:actual[name].copy() for name in ('y','valid','sample_ids','node_ids','direct_support')}
    expected.update(pred__FROZEN_R=actual['frozen_R'].copy(),pred__SUM_CONTROL=actual['pred'].copy())
    return actual,expected


class ExportContractTests(unittest.TestCase):
    def test_one_complete_split_global_null_mask_instead_of_batch_masks(self):
        y=np.full((2,12,3),5.,dtype=np.float32)
        y[0,0,0]=.2;y[1,0,0]=.1
        mask=export.global_valid_mask(y)
        self.assertTrue(mask[0,0,0]);self.assertFalse(mask[1,0,0])
        self.assertFalse(export.global_valid_mask(y[:1])[0,0,0])
        self.assertEqual(mask.dtype,np.bool_)

    def test_mask_zero_null_and_all_positive_targets(self):
        y=np.full((2,12,3),5.,dtype=np.float32)
        self.assertTrue(export.global_valid_mask(y).all())
        y[0,0,0]=0.
        self.assertFalse(export.global_valid_mask(y)[0,0,0])

    def test_mask_rejects_nonfinite_wrong_axes_and_no_valid_targets(self):
        invalid=[np.ones((2,3),dtype=np.float32),np.ones((2,12,3),dtype=np.int64),
            np.zeros((2,12,3),dtype=np.float32),np.full((2,12,3),np.nan,dtype=np.float32),
            np.empty((0,12,3),dtype=np.float32)]
        for value in invalid:
            with self.subTest(shape=value.shape),self.assertRaises(ValueError):export.global_valid_mask(value)

    def test_validation_replay_is_exact_for_every_axis_and_prediction(self):
        actual,expected=replay_fixture()
        before={name:value.tobytes() for name,value in actual.items()}
        result=export.assert_exact_replay(actual,expected)
        self.assertTrue(result['exact_equal'])
        self.assertTrue(all(result['fields'].values()))
        self.assertEqual(result['validation_samples'],2)
        self.assertEqual(before,{name:value.tobytes() for name,value in actual.items()})
        json.dumps(result,allow_nan=False)

    def test_validation_replay_rejects_one_ulp_float_change(self):
        actual,expected=replay_fixture()
        actual['pred'][0,1,1]=np.nextafter(actual['pred'][0,1,1],np.float32(np.inf))
        with self.assertRaisesRegex(ValueError,'values differ'):
            export.assert_exact_replay(actual,expected)

    def test_replay_checks_dtype_shape_sample_node_support_truth_and_mask(self):
        alterations=[lambda a:a.update(pred=a['pred'].astype(np.float64)),
            lambda a:a.update(pred=a['pred'][:1]),
            lambda a:a['sample_ids'].__setitem__(0,3911),
            lambda a:a['node_ids'].__setitem__(0,11),
            lambda a:a['direct_support'].__setitem__((0,0),True),
            lambda a:a['y'].__setitem__((0,1,1),21.),
            lambda a:a['valid'].__setitem__((0,0,0),True),
            lambda a:a['frozen_R'].__setitem__((0,1,1),24.)]
        for alter in alterations:
            actual,expected=replay_fixture();alter(actual)
            with self.subTest(alter=alter),self.assertRaises(ValueError):
                export.assert_exact_replay(actual,expected)

    def test_output_is_fresh_and_outside_probe_source_and_data_roots(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);probe=root/'probe';source=root/'source';data=root/'data'
            for path in (probe,source,data):path.mkdir()
            cert=dict(protected_directories=[str(probe),str(source)])
            for output in (probe/'cache',source/'cache',data/'cache',data,root):
                with self.subTest(output=output),self.assertRaises(ValueError):
                    export.require_cache_output(output,probe,data,cert)
            output=export.require_cache_output(root/'new_cache',probe,data,cert)
            self.assertEqual(output,(root/'new_cache').resolve());self.assertFalse(output.exists())

    def test_data_symlinks_cannot_bypass_output_protection(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);probe=root/'probe';data=root/'data';link=root/'data_link'
            probe.mkdir();data.mkdir()
            try:link.symlink_to(data,target_is_directory=True)
            except (OSError,NotImplementedError):self.skipTest('Directory symlinks unavailable')
            with self.assertRaisesRegex(ValueError,'authenticated data directory'):
                export.require_cache_output(link/'cache',probe,data,dict(protected_directories=[str(probe)]))

    def test_atomic_npz_has_explicit_fields_and_cannot_overwrite_cache(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'train.npz';actual,_=replay_fixture();del actual['frozen_R']
            actual['feature__neighbor_change_disagreement']=np.array([[np.nan,1.,2.],[np.nan,3.,4.]],dtype=np.float32)
            export.atomic_npz(path,actual)
            with np.load(path,allow_pickle=False) as archive:
                self.assertEqual(set(archive.files),set(actual))
                self.assertTrue(np.isnan(archive['feature__neighbor_change_disagreement'][0,0]))
                np.testing.assert_array_equal(archive['pred'],actual['pred'])
            digest=export.auth.sha256(path)
            with self.assertRaisesRegex(ValueError,'immutable'):export.atomic_npz(path,actual)
            self.assertEqual(export.auth.sha256(path),digest)
            self.assertEqual([entry.name for entry in Path(temp).iterdir()],['train.npz'])

    def test_atomic_json_refuses_nonfinite_certificates_without_publishing(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'cache.json'
            with self.assertRaises(ValueError):export.atomic_json(path,dict(invalid=float('nan')))
            self.assertFalse(path.exists());self.assertEqual(list(Path(temp).iterdir()),[])
            export.atomic_json(path,dict(model_updates=0,test_evaluated=False))
            self.assertEqual(json.loads(path.read_text()),dict(model_updates=0,test_evaluated=False))
            with self.assertRaisesRegex(ValueError,'immutable'):export.atomic_json(path,{})

    def test_frozen_checker_rejects_mode_gradients_or_weight_changes(self):
        class Component:
            def __init__(self,parameter,signature):self.parameter=parameter;self.signature=signature;self.training=False
            def parameters(self):return iter([self.parameter])
            def modules(self):return iter([self])
            def state_dict(self):return self.signature
        parameter=SimpleNamespace(requires_grad=False,grad=None)
        anchor=Component(parameter,'anchor');head=Component(SimpleNamespace(requires_grad=True,grad=None),'head')
        extractor=SimpleNamespace(anchor=anchor,modules=anchor.modules,assert_cache_clear=lambda:True)
        base=SimpleNamespace(tensor_tree_hash=lambda value:value)
        export._check_frozen(extractor,head,base,'anchor','head')
        for change in ('mode','gradient','weights','trainability'):
            with self.subTest(change=change):
                anchor.training=False;parameter.grad=None;parameter.requires_grad=False;head.signature='head'
                if change=='mode':anchor.training=True
                elif change=='gradient':parameter.grad='unexpected'
                elif change=='weights':head.signature='changed'
                else:parameter.requires_grad=True
                with self.assertRaises(ValueError):export._check_frozen(extractor,head,base,'anchor','head')

    def test_parser_has_no_training_budget_or_joint_options(self):
        common=['--probe-dir','probe','--data-dir','data','--cache-dir','cache']
        args=export.parse_args(common)
        self.assertEqual((args.device,args.threads),('cuda:0',3))
        for extra in (['--epochs','3'],['--resume'],['--joint'],['--threads','0']):
            with self.subTest(extra=extra),self.assertRaises(SystemExit):export.parse_args(common+extra)

    def test_importing_contract_helpers_does_not_import_torch_models_or_training(self):
        code=('import sys; from experiments.conditional_residual_gw10 import export_cache; '
            'assert not any(n=="torch" or n.startswith("torch.") for n in sys.modules); '
            'assert "experiments.branch_readout_gw9.train" not in sys.modules; print("NUMPY_EXPORT_HELPERS")')
        process=subprocess.run([sys.executable,'-c',code],cwd=export.REPO,capture_output=True,text=True,check=True)
        self.assertIn('NUMPY_EXPORT_HELPERS',process.stdout)


if __name__ == '__main__': unittest.main()
