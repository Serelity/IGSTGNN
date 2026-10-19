"""Server-only NumPy masking and CPU audit-gate contracts."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experiments.state_gate_gw11 import mask_audit as audit


def data(samples=96,nodes=3):
    y=np.full((samples,12,nodes),10.,dtype=np.float32)
    return y,np.ones(y.shape,dtype=bool),np.arange(samples,dtype=np.int64)


class GW11MaskTests(unittest.TestCase):
    def test_identical_masks_for_all_positive_targets_in_two_explicit_orders(self):
        y,valid,order=data()
        for permutation in (order,order[::-1]):
            result,batches,nodes=audit.audit_order(y,valid,permutation)
            self.assertTrue(result['all_masks_equal']);self.assertEqual(result['differing_points'],0)
            self.assertEqual(len(batches),2);self.assertEqual(len(nodes),3)
            json.dumps(result,allow_nan=False)

    def test_batch_minimum_can_exclude_a_valid_positive_value(self):
        y,_,order=data();y[0,0,0]=.2;y[48,0,0]=.1;valid=y!=y.min()
        result,batches,nodes=audit.audit_order(y,valid,order)
        self.assertEqual(result['wrongly_excluded'],1);self.assertEqual(result['wrongly_included'],0)
        self.assertFalse(result['all_masks_equal']);self.assertEqual(result['positive_below_one_count'],2)
        self.assertEqual(batches[0]['batch_null'],float(np.float32(.2)))
        self.assertEqual(nodes[0]['wrongly_excluded'],1)

    def test_negative_global_null_can_leave_zero_valid_without_missingness_claim(self):
        # This cannot occur if a single finite split-global minimum defines
        # the rule: assert the audit reports zero inclusions for this boundary.
        y,_,order=data();y[0,0,0]=-.1;y[48,0,0]=0.;valid=y!=y.min()
        result,_,_=audit.audit_order(y,valid,order)
        self.assertEqual(result['wrongly_included'],0)
        self.assertEqual(result['wrongly_excluded'],1)
        self.assertEqual(result['negative_count'],1);self.assertEqual(result['zero_count'],1)

    def test_all_null_batches_and_float32_near_zero_values_are_explicit(self):
        y=np.zeros((96,12,1),dtype=np.float32);valid=np.zeros(y.shape,dtype=bool)
        result,_,_=audit.audit_order(y,valid,np.arange(96))
        self.assertTrue(result['all_masks_equal']);self.assertEqual(result['global_valid_count'],0)
        y[48:]=10.;y[48,0,0]=np.nextafter(np.float32(0.),np.float32(1.));valid=y!=0.
        result,_,_=audit.audit_order(y,valid,np.arange(96))
        self.assertEqual(result['wrongly_excluded'],1)
        self.assertEqual(result['positive_below_one_count'],1)
        self.assertTrue(result['numeric_counts_do_not_identify_missingness'])

    def test_permutation_can_change_mask_agreement(self):
        y,_,order=data();y[0,0,0]=0.;y[48,0,0]=.5;valid=y!=0.
        authored=audit.audit_order(y,valid,order)[0]
        interleaved=np.ravel(np.column_stack((np.arange(48),np.arange(48,96))))
        shuffled=audit.audit_order(y,valid,interleaved)[0]
        self.assertEqual(authored['differing_points'],1)
        self.assertEqual(shuffled['differing_points'],0)
        self.assertNotEqual(authored['order_sha256'],shuffled['order_sha256'])

    def test_invalid_masks_order_dtype_duplicates_and_bounds_are_rejected(self):
        y,valid,order=data()
        cases=[(valid.astype(np.int8),order),(~valid,order),(valid,order.astype(float)),
            (valid,np.zeros(96,dtype=int)),(valid,np.arange(1,97)),(valid,order[:-1])]
        for mask,permutation in cases:
            with self.subTest(permutation=permutation[:2]),self.assertRaises(ValueError):
                audit.audit_order(y,mask,permutation)
        for batch_size in (0,-1,True,1.5):
            with self.assertRaises(ValueError):audit.audit_order(y,valid,order,batch_size)

    def test_nodes_and_value_details_recover_total_difference_counts(self):
        y,_,order=data();y[0,0,0]=0.;y[48,1,1]=.1;y[49,2,2]=.1;valid=y!=0.
        result,batches,nodes=audit.audit_order(y,valid,order)
        self.assertEqual(result['differing_points'],2)
        self.assertEqual(sum(row['differing_points'] for row in nodes),2)
        self.assertEqual(sum(row['differing_points'] for row in batches),2)
        self.assertEqual(result['differing_values'][0]['wrongly_excluded'],2)

    def test_saved_checkpoint_is_hash_checked_before_import_or_loading(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'last.pt';path.write_bytes(b'untrusted')
            certificate=dict(file_digests={str(path.resolve()):'0'*64})
            with self.assertRaisesRegex(ValueError,'pinned GW9'):audit.load_saved_order(temp,certificate)

    def test_run_and_gate_preserve_source_and_recompute_bound_mask_summaries(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);probe=root/'probe';cache_dir=root/'cache';gw10=root/'gw10'
            for path in (probe,cache_dir,gw10):path.mkdir()
            y,valid,order=data();np.savez_compressed(cache_dir/'train.npz',y=y,valid=valid)
            digests={str(cache_dir/'train.npz'):audit.auth.sha256(cache_dir/'train.npz')}
            certificate=dict(source_identity_sha256='synthetic',probe_directory=str(probe),
                file_digests={},protected_directories=[str(probe)])
            gw10ref=dict(file_digests={});metadata=dict(data_file_digests={})
            inputs=({'node_ids':np.array([10,20,30])},certificate,{'train':dict(y=y,valid=valid)},metadata,digests,gw10ref)
            provenance=dict(completed_epoch=3,scope='saved_LAST_epoch_only_not_prior_epochs_or_future_shuffles')
            output=root/'audit'
            with patch.object(audit.reference,'load_inputs',return_value=inputs), \
                    patch.object(audit,'load_saved_order',return_value=(order,provenance)):
                result=audit.run_audit(probe,cache_dir,gw10,output)
                gate=audit.validate_audit(output,certificate,digests,gw10ref)
                self.assertTrue(result['eligible_for_training']);self.assertTrue(gate['eligible_for_training'])
                self.assertFalse(result['future_training_orders_certified'])
                self.assertTrue(gate['runtime_mask_assertions_required'])
                self.assertEqual(len(gate['file_digests']),4)
                edited=deepcopy(result);edited['orders']['author_order']['order_sha256']='edited'
                (output/'mask_audit.json').write_text(json.dumps(edited),encoding='utf-8')
                with self.assertRaisesRegex(ValueError,'do not replay'):
                    audit.validate_audit(output,certificate,digests,gw10ref)
            self.assertEqual(audit.auth.sha256(cache_dir/'train.npz'),digests[str(cache_dir/'train.npz')])
            with patch.object(audit.reference,'load_inputs',return_value=inputs),self.assertRaises(ValueError):
                audit.run_audit(probe,cache_dir,gw10,output)

    def test_failed_mask_gate_cannot_authorize_training(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'mask_audit.json'
            path.write_text(json.dumps(dict(schema=audit.SCHEMA,status='GW11_MASK_AUDIT_COMPLETE',
                eligible_for_training=False)),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'passing two-order'):
                audit.validate_audit(temp,dict(source_identity_sha256='x',file_digests={}), {}, {})


if __name__=='__main__':unittest.main()
