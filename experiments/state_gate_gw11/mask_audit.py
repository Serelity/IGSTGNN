"""CPU-only native batch-null/global-null audit before any GW11 training.

The saved permutation belongs only to GW9's last completed epoch. Neither it
nor the author-order audit establishes historical or future shuffle agreement.
Numeric zero is reported without asserting missing-value semantics.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.branch_readout_gw9 import readout_auth as auth
from experiments.state_gate_gw11 import reference


SCHEMA = 'gw11_mask_audit_v1'
ORDER_NAMES = ('author_order','GW9_saved_last_epoch_order')


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def audit_order(y,valid,order,batch_size=48):
    """Compare masking rules in one explicit complete permutation using NumPy."""
    y,valid,order = map(np.asarray,(y,valid,order))
    auth.require(y.ndim == 3 and y.shape[1] == 12 and min(y.shape)>0
        and y.dtype == np.float32 and np.isfinite(y).all(), 'Finite float32 [sample,12,node] targets required')
    auth.require(valid.dtype == np.bool_ and valid.shape == y.shape, 'Shared Boolean global mask required')
    auth.require(order.shape == (len(y),) and order.dtype.kind in 'iu'
        and np.array_equal(np.sort(order),np.arange(len(y))), 'Complete training permutation required')
    auth.require(type(batch_size) is int and batch_size>0, 'Positive integer batch size required')
    null = float(y.min()) if y.min()<1 else 0.
    auth.require(np.array_equal(valid,y!=null), 'Cache validity differs from complete training global-null rule')
    node_included = np.zeros(y.shape[2],dtype=np.int64)
    node_excluded = np.zeros_like(node_included)
    batches,values = [],{}
    for offset in range(0,len(y),batch_size):
        indices=order[offset:offset+batch_size];target=y[indices];global_mask=valid[indices]
        batch_null=float(target.min()) if target.min()<1 else 0.
        native_mask=target!=batch_null
        included=native_mask & ~global_mask;excluded=~native_mask & global_mask
        node_included+=included.sum(axis=(0,1));node_excluded+=excluded.sum(axis=(0,1))
        for label,selection in (('wrongly_included',included),('wrongly_excluded',excluded)):
            unique,counts=np.unique(target[selection],return_counts=True)
            for value,count in zip(unique,counts):
                key=float(value).hex()
                row=values.setdefault(key,dict(value=float(value),float64_hex=key,
                    wrongly_included=0,wrongly_excluded=0))
                row[label]+=int(count)
        batches.append(dict(batch=offset//batch_size+1,samples=len(indices),
            batch_null=batch_null,global_null=null,null_values_equal=batch_null==null,
            native_valid_count=int(native_mask.sum()),global_valid_count=int(global_mask.sum()),
            wrongly_included=int(included.sum()),wrongly_excluded=int(excluded.sum()),
            differing_points=int(included.sum()+excluded.sum()),
            first_author_sample=int(indices[0]),last_author_sample=int(indices[-1])))
    included_count=int(node_included.sum());excluded_count=int(node_excluded.sum())
    result=dict(samples=len(y),horizon=y.shape[1],nodes=y.shape[2],batch_size=batch_size,
        global_null=null,global_valid_count=int(valid.sum()),negative_count=int((y<0).sum()),
        zero_count=int((y==0).sum()),positive_below_one_count=int(((y>0)&(y<1)).sum()),
        wrongly_included=included_count,wrongly_excluded=excluded_count,
        differing_points=included_count+excluded_count,all_masks_equal=included_count+excluded_count==0,
        distinct_batch_nulls=sorted({row['batch_null'] for row in batches}),
        order_sha256=hashlib.sha256(order.astype('<i8').tobytes()).hexdigest(),
        differing_values=sorted(values.values(),key=lambda row:row['value']),
        numeric_counts_do_not_identify_missingness=True)
    nodes=[dict(node_index=index,wrongly_included=int(node_included[index]),
        wrongly_excluded=int(node_excluded[index]),
        differing_points=int(node_included[index]+node_excluded[index])) for index in range(y.shape[2])]
    return result,batches,nodes


def load_saved_order(probe_dir,certificate):
    """Lazy CPU weights-only deserialization of the already hash-pinned GW9 state."""
    path=Path(probe_dir).resolve()/'last.pt'
    expected=certificate['file_digests'].get(str(path))
    auth.require(expected == auth.LAST_CHECKPOINT_SHA256 and auth.sha256(path)==expected,
        'Only the pinned GW9 paired LAST checkpoint may provide the saved order')
    import torch
    from experiments.chronological.original_igstgnn_checkpoint_compat import counter_weights_only_compat
    with counter_weights_only_compat():
        state=torch.load(path,map_location='cpu',weights_only=True)
    auth.require(auth.sha256(path)==expected and state.get('schema')=='original_igstgnn_branch_readout_gw9_probe_v1'
        and state.get('identity_sha256')==certificate['source_identity_sha256']
        and state.get('completed_epoch')==3 and state.get('head_updates')=={arm:246 for arm in auth.ARMS},
        'GW9 saved order state identity or completed budget differs')
    order=np.asarray(state['train_order']).copy()
    return order,dict(checkpoint_file=str(path),checkpoint_sha256=expected,completed_epoch=3,
        scope='saved_LAST_epoch_only_not_prior_epochs_or_future_shuffles',map_location='cpu',weights_only=True)


def _bindings(certificate,cache_digests,gw10reference):
    return dict(source_identity_sha256=certificate['source_identity_sha256'],
        source_file_digests=certificate['file_digests'],cache_file_digests=cache_digests,
        gw10_reference=gw10reference)


def _write_json(path,value):
    with Path(path).open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')


def _write_csv(path,rows):
    with Path(path).open('x',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def run_audit(probe_dir,cache_dir,gw10_dir,output_dir):
    archive,certificate,cache,metadata,digests,gw10reference=reference.load_inputs(probe_dir,cache_dir,gw10_dir)
    output=reference.require_output_outside_sources(output_dir,probe_dir,certificate,cache_dir,gw10_dir)
    # Cache source data files are hashed in metadata; their parent directories
    # also remain protected even though this audit reads cached arrays only.
    for filename in metadata.get('data_file_digests',{}):
        directory=Path(filename).resolve().parent
        auth.require(output!=directory and directory not in output.parents and output not in directory.parents,
            'Mask audit output must be outside authenticated data directories')
    output.mkdir(parents=True,exist_ok=False)
    try:
        saved,saved_provenance=load_saved_order(probe_dir,certificate)
        orders=dict(author_order=np.arange(len(cache['train']['y']),dtype=np.int64),
            GW9_saved_last_epoch_order=saved)
        results,batch_rows,node_rows={ },[],[]
        for name,order in orders.items():
            result,batches,nodes=audit_order(cache['train']['y'],cache['train']['valid'],order)
            results[name]=result
            batch_rows.extend(dict(order=name,**row) for row in batches)
            node_rows.extend(dict(order=name,node_id=int(archive['node_ids'][row['node_index']]),**row) for row in nodes)
        eligible=all(row['differing_points']==0 for row in results.values())
        result=dict(schema=SCHEMA,status='GW11_MASK_AUDIT_COMPLETE',bindings=_bindings(certificate,digests,gw10reference),
            orders=results,saved_order_provenance=saved_provenance,eligible_for_training=eligible,
            covered_orders=list(ORDER_NAMES),audit_scope='two_explicit_orders_only',
            prior_epochs_certified=False,future_training_orders_certified=False,
            next_training_batches_require_runtime_mask_equality=True,
            difference_does_not_authorize_mask_change=True,
            numeric_zero_is_missingness_claim=False,neural_inference_calls=0,model_updates=0,test_evaluated=False)
        _write_csv(output/'batches.csv',batch_rows);_write_csv(output/'nodes.csv',node_rows)
        lines=['# GW11 training mask audit','',
            'CPU cache-only audit of author order and the single saved GW9 LAST epoch order.',
            'The saved permutation does not certify prior epochs or future shuffles.',
            'Numeric zero and near-zero counts do not establish missing-value semantics.','',
            '| Order | Differing points | Wrongly included | Wrongly excluded |','|---|---:|---:|---:|']
        for name,row in results.items():
            lines.append('| {} | {} | {} | {} |'.format(name,row['differing_points'],row['wrongly_included'],row['wrongly_excluded']))
        lines+=['','eligible_for_training: '+str(eligible).lower(),
            'Any difference stops GW11 before training; the audit never changes the mask.',
            'Every future training batch must still compare native and cached global masks.','']
        (output/'report.md').write_text('\n'.join(lines),encoding='utf-8')
        auth.assert_unchanged(certificate);reference.assert_unchanged(gw10reference)
        for filename,expected in digests.items():
            auth.require(auth.sha256(filename)==expected,'Cache changed during mask audit')
        _write_json(output/'mask_audit.json',result)
        return result
    except BaseException as error:
        _write_json(output/'failure.json',dict(schema=SCHEMA,status='GW11_MASK_AUDIT_FAILED',
            error_type=type(error).__name__,error=str(error),model_updates=0,test_evaluated=False))
        raise


def validate_audit(audit_dir,certificate,cache_digests,gw10reference):
    directory=Path(audit_dir).resolve();path=directory/'mask_audit.json'
    digest=auth.sha256(path);audit=_json(path)
    auth.require(audit.get('schema')==SCHEMA and audit.get('status')=='GW11_MASK_AUDIT_COMPLETE'
        and audit.get('bindings')==_bindings(certificate,cache_digests,gw10reference)
        and audit.get('covered_orders')==list(ORDER_NAMES) and set(audit.get('orders',{}))==set(ORDER_NAMES)
        and audit.get('eligible_for_training') is True
        and all(row.get('differing_points')==0 and row.get('all_masks_equal') is True for row in audit['orders'].values())
        and audit.get('next_training_batches_require_runtime_mask_equality') is True
        and audit.get('prior_epochs_certified') is False and audit.get('future_training_orders_certified') is False
        and audit.get('model_updates')==0 and audit.get('neural_inference_calls')==0
        and audit.get('test_evaluated') is False,'A bound passing two-order mask audit is required before GW11 training')
    training_paths=[Path(name) for name in cache_digests if Path(name).name=='train.npz']
    auth.require(len(training_paths)==1,'One authenticated training cache is required for audit replay')
    training_path=training_paths[0]
    auth.require(auth.sha256(training_path)==cache_digests[str(training_path)],'Training cache changed before audit replay')
    with np.load(training_path,allow_pickle=False) as payload:
        y,valid=payload['y'],payload['valid']
    saved,provenance=load_saved_order(certificate['probe_directory'],certificate)
    recomputed={}
    for name,order in (('author_order',np.arange(len(y),dtype=np.int64)),('GW9_saved_last_epoch_order',saved)):
        recomputed[name]=audit_order(y,valid,order)[0]
    auth.require(recomputed==audit['orders'] and provenance==audit.get('saved_order_provenance'),
        'Mask audit summaries do not replay from their bound training targets and saved order')
    auth.require(auth.sha256(training_path)==cache_digests[str(training_path)],'Training cache changed during audit replay')
    files={str(path):digest}
    for name in ('report.md','batches.csv','nodes.csv'):
        child=directory/name;files[str(child)]=auth.sha256(child)
    auth.assert_unchanged(certificate);reference.assert_unchanged(gw10reference)
    auth.require(auth.sha256(path)==digest,'Mask audit changed during gate certification')
    return dict(schema='gw11_authenticated_mask_audit_gate_v1',directory=str(directory),
        source_identity_sha256=certificate['source_identity_sha256'],audit_sha256=digest,
        file_digests=files,bindings=audit['bindings'],eligible_for_training=True,
        future_training_orders_certified=False,runtime_mask_assertions_required=True,test_evaluated=False)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('probe-dir','cache-dir','gw10-dir','output-dir'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args(argv)
    result=run_audit(args.probe_dir,args.cache_dir,args.gw10_dir,args.output_dir)
    print(json.dumps(dict(status=result['status'],eligible_for_training=result['eligible_for_training'],
        output_dir=str(args.output_dir))),flush=True)
    return result


if __name__=='__main__':main()
