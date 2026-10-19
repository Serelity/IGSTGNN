"""Server-only historical descriptor contracts; no local execution."""
import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch
import numpy as np

from experiments.branch_readout_gw9 import readout_history as history


class ReadoutHistoryTests(unittest.TestCase):
    def test_origin_history_descriptors_use_only_x_and_fixed_normalization(self):
        values = np.array([[[0.,0.],[1.,2.],[3.,2.]]])
        graph = np.array([[10.,1.],[1.,10.]])
        result = history.feature_arrays(values,10.,2.,graph)
        np.testing.assert_array_equal(result['last_flow'],[[16.,14.]])
        np.testing.assert_array_equal(result['last_change'],[[4.,0.]])
        np.testing.assert_array_equal(result['history_trend'],[[6.,4.]])
        np.testing.assert_array_equal(result['mean_abs_change'],[[3.,2.]])
        np.testing.assert_array_equal(result['neighbor_change_disagreement'],[[4.,4.]])
        np.testing.assert_array_equal(result['neighbor_change_dispersion'],[[0.,0.]])

    def test_row_weights_and_isolated_nodes_have_explicit_semantics(self):
        values = np.array([[[0.,0.,0.],[2.,4.,8.]]])
        graph = np.array([[0.,1.,3.],[0.,9.,0.],[1.,0.,0.]])
        result = history.feature_arrays(values,0.,1.,graph)
        self.assertAlmostEqual(result['neighbor_change_disagreement'][0,0],5.)
        self.assertAlmostEqual(result['neighbor_change_dispersion'][0,0],np.sqrt(3.))
        self.assertTrue(np.isnan(result['neighbor_change_disagreement'][0,1]))
        self.assertTrue(np.isnan(result['neighbor_change_dispersion'][0,1]))
        self.assertEqual(result['neighbor_change_disagreement'][0,2],6.)

    def test_cutpoints_use_training_inputs_with_ties_and_missing_availability(self):
        train = dict(change=np.array([0.,1.,2.,3.,4.]),zero=np.zeros(5),missing=np.full(3,np.nan))
        cuts = history.train_cutpoints(train)
        self.assertEqual(cuts['change']['cutpoints'],[1.,2.,3.])
        self.assertEqual(cuts['change']['training_finite_count'],5)
        np.testing.assert_array_equal(history.assign_quartiles(np.array([0.,1.,2.,3.,9.,np.nan]),
            cuts['change']['cutpoints']),[0,1,2,3,3,-1])
        np.testing.assert_array_equal(history.assign_quartiles(np.zeros(2),cuts['zero']['cutpoints']),[3,3])
        np.testing.assert_array_equal(history.assign_quartiles(np.zeros(2),None),[-1,-1])
        self.assertIsNone(cuts['missing']['cutpoints'])

    def test_invalid_input_graph_normalization_and_cuts_are_rejected(self):
        values = np.ones((1,3,2)); graph = np.eye(2)
        for bad_history,bad_std,bad_graph in ((values[:,0],1.,graph),(values,0.,graph),
            (values,1.,-graph),(values,1.,np.ones((3,3)))):
            with self.assertRaises(ValueError): history.feature_arrays(bad_history,0.,bad_std,bad_graph)
        for cuts in ([1.,0.,2.],[1.,np.nan,2.],[1.,2.]):
            with self.assertRaises(ValueError): history.assign_quartiles(values,cuts)

    def test_dataset_hashes_are_checked_before_any_pickle_load(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'incident_all.npy'; path.write_bytes(b'untrusted')
            identity=dict(hashes={'incident_all.npy':'0'*64})
            with patch.object(history,'_load_array') as loader,self.assertRaises(ValueError):
                history.load_origin_features(directory,identity)
            loader.assert_not_called()

    def test_loader_accesses_only_train_val_histories_and_uses_train_cutpoints(self):
        accessed=[]
        class Sample:
            def __init__(self,index):self.index=index
            def __getitem__(self,key):
                if key!='x_data':raise AssertionError('Target access is forbidden')
                accessed.append(self.index)
                return np.full((12,496,3),self.index,dtype=np.float32)
        class Array:
            def __len__(self):return 6
            def __getitem__(self,index):
                if index>=5:raise AssertionError('Test sample access is forbidden')
                return Sample(index)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); array=root/'incident_all.npy'; array.write_bytes(b'fingerprinted fixture')
            np.save(root/'adj_matrix.npy',np.ones((496,496),dtype=np.float32))
            identity=dict(hashes={path.name:history.sha256(path) for path in root.iterdir()},
                normalization=dict(mean=0.,std=1.,refitted=False),splits={
                    name:dict(bounds_half_open=bounds,selected_indices=list(range(*bounds)),count=bounds[1]-bounds[0])
                    for name,bounds in {'train':[0,3],'val':[3,5]}.items()})
            with patch.object(history,'AUTHOR_COUNT',6), \
                    patch.object(history,'SPLIT_BOUNDS',{'train':[0,3],'val':[3,5]}), \
                    patch.object(history,'_load_array',return_value=Array()):
                features,metadata=history.load_origin_features(root,identity)
            self.assertEqual(accessed,[0,1,2,3,4])
            self.assertEqual(features['last_flow'].shape,(2,496))
            cuts=metadata['train_input_cutpoints']['last_flow']['cutpoints']
            self.assertEqual(cuts,[0.,1.,2.])
            self.assertFalse(metadata['training_targets_used'])
            self.assertFalse(metadata['test_samples_accessed'])
            self.assertFalse(metadata['test_evaluated'])
            self.assertTrue(metadata['test_objects_deserialized'])
            array.write_bytes(b'changed after extraction')
            with self.assertRaises(ValueError):history.assert_unchanged(metadata)


if __name__=='__main__': unittest.main()
