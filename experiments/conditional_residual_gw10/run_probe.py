"""CPU median fitting on certified frozen-SUM caches; no Torch dependency.

An author-order training holdout checks the fitted mapping, not the previously
trained backbone. The final mapping uses all train labels and is committed to
disk before validation scoring. No validation labels enter fitting or cuts.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from experiments.branch_readout_gw9.readout_auth import (
    SOURCE_IDENTITY_SHA256, assert_unchanged, load_authenticated_run,
    require, require_output_outside_sources, sha256)
from experiments.conditional_residual_gw10 import export_cache
from experiments.conditional_residual_gw10.median_probe import (
    ARMS, fit_probe, predict_corrections, save_model)
from experiments.conditional_residual_gw10.screen import (
    AUDIT_FIT_STOP, AUDIT_SCORE_START, CANDIDATE, TRAIN_COUNT, metrics, paired, screen)

FEATURES = ('last_flow','mean_abs_change','neighbor_change_disagreement')
SCHEMA = 'gw10_train_only_residual_direction_probe_v1'
PROTOCOL = dict(candidate='JOINT', minimum_relative_validation_gain=.005,
    minimum_points_per_cell=64,minimum_author_order_blocks=3,author_order_block_size=128,
    shrinkage_pseudocount=128,flow_internal_edges=[0,5,20,50,100,200,314],
    global_parent='none_raw_training_median',horizon_parent='GLOBAL',node_horizon_parent='HORIZON',
    origin_parent='NODE_HORIZON',state_parent='ORIGIN',
    candidate_states='train_median_own_change_x_train_median_neighbor_disagreement',
    four_state_control='OWN4_train_quartiles',
    audit_fit_bounds=[0,AUDIT_FIT_STOP],audit_gap_bounds=[AUDIT_FIT_STOP,AUDIT_SCORE_START],
    audit_score_bounds=[AUDIT_SCORE_START,TRAIN_COUNT],
    fit_targets='training_y_minus_frozen_SUM',prediction='SUM_plus_train_fitted_median',
    corrections_clipped=False,validation_selects_arms_or_parameters=False,
    mean_or_scale_only_correlation_is_not_direction_evidence=True,test_evaluated=False)


def write_json(path, value):
    path = Path(path); temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temporary.replace(path)


def write_csv(path, rows):
    rows = list(rows)
    if not rows: return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w',newline='',encoding='utf-8') as stream:
        writer = csv.DictWriter(stream,fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def load_cache(directory, archive, certificate):
    directory = Path(directory).resolve(); metadata_path = directory/'cache.json'
    metadata_hash = sha256(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    require(metadata.get('schema') == export_cache.SCHEMA
        and metadata.get('status') == 'GW10_FROZEN_SUM_CACHE_COMPLETE'
        and metadata.get('source_identity_sha256') == SOURCE_IDENTITY_SHA256
        and metadata.get('source_certificate') == certificate
        and metadata.get('data_identity') == certificate['data_identity']
        and metadata.get('model_updates') == 0 and metadata.get('optimizer_steps') == 0
        and metadata.get('test_evaluated') is False
        and metadata.get('test_loader_constructed') is False
        and metadata.get('test_samples_evaluated') is False
        and metadata.get('source_artifacts_written') is False
        and metadata.get('failed_GW9_joint_gate_unchanged') is True,
        'Cache source, frozen-update budget or schema differs')
    source_role = certificate['head_roles']['best']['SUM_CONTROL']
    require(metadata.get('selected_head_role') == source_role
        and metadata.get('head_tensor_sha256') == source_role['head_tensor_sha256']
        and metadata.get('backbone_tensor_sha256') == certificate['source_reference'][
            'source_initialization']['model_tensor_sha256'],
        'Cache selected SUM role or frozen tensor identity differs')
    require(metadata.get('axis_order') == 'sample,horizon,node'
        and metadata.get('feature_axis_order') == 'sample,node'
        and metadata.get('features') == list(FEATURES) == list(export_cache.FEATURES)
        and metadata.get('feature_source') == 'x_data_traffic_feature_zero_at_forecast_origin'
        and metadata.get('feature_precision') == 'float64_history_descriptors_then_float32_storage'
        and metadata.get('isolated_neighbor_descriptor') == 'NaN_unavailable_not_imputed_zero'
        and metadata.get('training_sample_order') == 'original_author_order_no_shuffle'
        and metadata.get('gradient_mode') == 'no_grad_preserve_certified_parameter_flags'
        and metadata.get('train_valid_mask') == 'one_complete_training_split_native_global_null_rule'
        and metadata.get('validation_valid_mask') == 'strict_replay_then_copy_authenticated_GW9_BEST_mask',
        'Cache axes, historical feature implementation or execution policy differs')
    replay = metadata.get('frozen_replay',{})
    require(replay.get('exact_equal') is True
        and replay.get('fields') == {name:True for name in
            ('y','valid','sample_ids','node_ids','direct_support','frozen_R','pred')}
        and replay.get('policy') == 'same_dtype_same_shape_all_values_equal_without_tolerance'
        and replay.get('validation_samples') == 837
        and replay.get('inference_before_training_export') is True
        and metadata.get('validation_replayed_before_train_export') is True,
        'Cache must certify exact validation replay before train export')
    loaded, digests = {}, {str(metadata_path):metadata_hash}
    execution = {name:sha256(export_cache.REPO/name) for name in export_cache.SOURCE_FILES}
    require(metadata.get('cache_execution_source') == execution,
        'Cache exporter implementation differs; export a fresh cache with the current source')
    digests.update({str(export_cache.REPO/name):digest for name,digest in execution.items()})
    require(set(metadata.get('archives',{})) == {'train','val'},
        'Exactly the train and validation cache archives are required')
    total_batches = 0
    expected_keys = {'y','pred','valid','sample_ids','node_ids','direct_support',
                     *('feature__'+name for name in FEATURES)}
    for split,bounds in (('train',(0,3909)),('val',(3909,4746))):
        entry = metadata['archives'][split]; path = (directory/entry['path']).resolve()
        split_identity = certificate['data_identity']['splits'][split]
        require(split_identity['bounds_half_open'] == list(bounds)
            and split_identity['selected_indices'] == list(range(*bounds))
            and split_identity['count'] == bounds[1]-bounds[0]
            and entry.get('shape') == [bounds[1]-bounds[0],12,496]
            and entry.get('samples') == bounds[1]-bounds[0]
            and entry.get('inference_batches') == len(split_identity['partition']),
            'Cache split descriptor or batch budget differs: '+split)
        total_batches += len(split_identity['partition'])
        require(path.parent == directory and path.name == split+'.npz'
            and sha256(path) == entry['sha256'],'Cache archive path or bytes differ: '+split)
        with np.load(path,allow_pickle=False) as payload:
            require(set(payload.files) == expected_keys,'Unexpected cache keys: '+split)
            values = {name:payload[name].copy() for name in payload.files}
        shape = (bounds[1]-bounds[0],12,496)
        require(values['pred'].shape == values['y'].shape == values['valid'].shape == shape
            and values['pred'].dtype == values['y'].dtype == np.float32
            and values['valid'].dtype == np.bool_ and values['valid'].any()
            and np.isfinite(values['pred']).all() and np.isfinite(values['y']).all(),
            'Cache prediction, truth or mask differs: '+split)
        null = float(values['y'].min()) if values['y'].min()<1 else 0.
        require(np.array_equal(values['valid'],values['y'] != null)
            and (values['y'][values['valid']]>0).all(),'Global source-null mask differs: '+split)
        require(values['sample_ids'].dtype == np.int64
            and np.array_equal(values['sample_ids'],np.arange(*bounds))
            and values['node_ids'].dtype == archive['node_ids'].dtype
            and np.array_equal(values['node_ids'],archive['node_ids'])
            and values['direct_support'].shape == (shape[0],shape[2])
            and values['direct_support'].dtype == np.bool_,'Cache axes differ: '+split)
        for name in FEATURES:
            feature = values['feature__'+name]
            require(feature.shape == (shape[0],shape[2]) and feature.dtype == np.float32,
                'Cache historical feature axis differs: '+name)
            require(not np.isinf(feature).any() and (name == 'neighbor_change_disagreement'
                or np.isfinite(feature).all()),'Cache input history is nonfinite: '+name)
        values['features'] = {name:values.pop('feature__'+name) for name in FEATURES}
        loaded[split] = values; digests[str(path)] = entry['sha256']
    require(metadata.get('complete_forward_calls') == total_batches,
        'Frozen export complete-forward budget differs')
    for name in ('y','valid','sample_ids','node_ids','direct_support'):
        require(loaded['val'][name].dtype == archive[name].dtype
            and np.array_equal(loaded['val'][name],archive[name]),'Validation cache axis differs: '+name)
    require(np.array_equal(loaded['val']['pred'],archive['pred__SUM_CONTROL']),
        'SUM validation cache differs from certified GW9 BEST')
    rehash(digests)
    return loaded,metadata,digests


def rehash(digests):
    for filename,digest in digests.items():
        require(sha256(filename)==digest,'Input changed while probing: '+filename)


def slice_data(data, start, stop):
    return dict(pred=data['pred'][start:stop],y=data['y'][start:stop],valid=data['valid'][start:stop],
        sample_ids=data['sample_ids'][start:stop],features={key:value[start:stop]
            for key,value in data['features'].items()})


def fit(data):
    return fit_probe(data['pred'],data['y'],data['valid'],data['features'],data['sample_ids'])


def predictions(model, data):
    corrections = predict_corrections(model,data['features'],horizon=data['pred'].shape[1])
    require(set(corrections)==set(ARMS),'Probe returned unexpected arms')
    reference = data['pred'].astype(np.float64)
    result = {'SUM':reference}
    for arm,value in corrections.items():
        require(value.shape==reference.shape and np.isfinite(value).all(),'Invalid correction: '+arm)
        result[arm] = reference+value
    return result


def records(predictions_by_arm, data, mask=None):
    valid = data['valid'] if mask is None else data['valid'] & np.broadcast_to(mask,data['valid'].shape)
    if not valid.any():
        return {arm:dict(valid_count=0,mae=None,rmse=None,signed_bias=None,mape_fraction=None)
            for arm in predictions_by_arm}
    return {arm:metrics(value,data['y'],valid) for arm,value in predictions_by_arm.items()}


def paired_rows(preds, data, mask=None, group='all'):
    valid = data['valid'] if mask is None else data['valid'] & np.broadcast_to(mask,data['valid'].shape)
    for arm in ARMS:
        yield dict(group=group,comparison=arm+'_vs_SUM',**paired(preds['SUM'],preds[arm],data['y'],valid))
    for control in ('ORIGIN','OWN2','NEIGHBOR2','OWN4'):
        yield dict(group=group,comparison='JOINT_vs_'+control,
            **paired(preds[control],preds[CANDIDATE],data['y'],valid))


def direction_rows(preds, data):
    valid = data['valid']; wanted = data['y'].astype(np.float64)-preds['SUM']
    for arm in ARMS:
        correction = preds[arm]-preds['SUM']
        for name,selected in (('up',valid & (correction>0)),('down',valid & (correction<0)),
                ('zero',valid & (correction==0))):
            count = int(selected.sum()); target = wanted[selected]
            gain = paired(preds['SUM'],preds[arm],data['y'],selected)
            yield dict(arm=arm,correction_direction=name,valid_count=count,
                desired_correction_median=float(np.median(target)) if count else None,
                same_sign_fraction=float(np.mean(np.sign(correction[selected])==np.sign(target))) if count else None,
                MAE_gain=gain['mae_gain'],interpretation='descriptive_validation_direction_not_a_router')


def report_text(result):
    lines = ['# GW10 train-only conditional-median residual probe','',
        'Frozen GW9 SUM BEST. No backbone/head updates; lookup tables use training labels only.',
        'JOINT is the sole candidate. All arms are reported without validation winner selection.',
        'Training residuals come from a model already trained on those labels; the audit holds out only the new mapping.',
        'Author-order blocks and the 128-sample audit gap are not certified calendar separation.',
        'This reused development validation is not independent confirmation. No test evaluation.','',
        '| Case | Validation MAE | RMSE | Signed bias | Training-order audit MAE |',
        '|---|---:|---:|---:|---:|']
    for arm,row in result['validation'].items():
        lines.append('| {} | {:.9f} | {:.9f} | {:.9f} | {:.9f} |'.format(
            arm,row['mae'],row['rmse'],row['signed_bias'],result['training_order_audit'][arm]['mae']))
    lines += ['', '| JOINT comparison | MAE gain | Relative gain % |','|---|---:|---:|']
    for row in result['paired_validation']:
        if row['comparison'].startswith('JOINT_vs_'):
            lines.append('| {} | {:.9f} | {:.6f} |'.format(row['comparison'],row['mae_gain'],100*row['relative_gain']))
    lines += ['','| Fixed check | Passed |','|---|---|']
    for name,value in result['screening']['checks'].items(): lines.append('| '+name+' | '+str(value)+' |')
    lines += ['', 'eligible_for_new_model_design: '+str(result['screening']['eligible_for_new_model_design']),
        'This result does not authorize GW9 joint training, lookup retuning, or additional epochs.',
        'See model.json and coverage.csv for sparse-cell fallback and training support; this is not validation routing coverage.',
        'A pass supports further model design under this probe, not a causal neighborhood mechanism or a novel architecture.','']
    return '\n'.join(lines)


def run(probe_dir, cache_dir, output_dir):
    archive,_,certificate = load_authenticated_run(probe_dir)
    output = Path(output_dir).resolve(); cache_dir = Path(cache_dir).resolve()
    require_output_outside_sources(output,probe_dir,certificate)
    require(output != cache_dir and cache_dir not in output.parents and output not in cache_dir.parents,
        'Probe output and immutable cache must be separate sibling directories')
    loaded,cache_meta,digests = load_cache(cache_dir,archive,certificate)
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'protocol.json',dict(schema=SCHEMA,protocol=PROTOCOL,
        source_identity_sha256=SOURCE_IDENTITY_SHA256,input_digests=digests))
    try:
        train = loaded['train']; validation = loaded['val']
        print(json.dumps(dict(phase='fit_training_order_audit',fit_samples=AUDIT_FIT_STOP,
            excluded_gap=AUDIT_SCORE_START-AUDIT_FIT_STOP)),flush=True)
        audit_model = fit(slice_data(train,0,AUDIT_FIT_STOP))
        save_model(audit_model,output/'audit_model.json')
        audit_data = slice_data(train,AUDIT_SCORE_START,TRAIN_COUNT)
        audit_predictions = predictions(audit_model,audit_data)
        audit_records = records(audit_predictions,audit_data)
        write_csv(output/'audit_cases.csv',(dict(arm=arm,**row) for arm,row in audit_records.items()))
        write_csv(output/'audit_comparisons.csv',paired_rows(audit_predictions,audit_data))
        del audit_predictions
        print(json.dumps(dict(phase='fit_all_train',samples=TRAIN_COUNT)),flush=True)
        model = fit(train); save_model(model,output/'model.json')
        fitted_hash = sha256(output/'model.json')
        write_json(output/'fitted_before_validation.json',dict(model_sha256=fitted_hash,
            protocol_sha256=sha256(output/'protocol.json'),validation_labels_used_for_fit=False))
        print(json.dumps(dict(phase='one_fixed_validation_score',model_sha256=fitted_hash)),flush=True)
        preds = predictions(model,validation); validation_records = records(preds,validation)
        masks = dict(high_value_ge314=validation['y']>=314,
            long_horizon_7_12=(np.arange(12)[None,:,None]>=6),
            flow_50_200=(validation['y']>=50)&(validation['y']<200))
        groups = {name:records(preds,validation,mask) for name,mask in masks.items()}
        decision = screen(validation_records,audit_records,groups)
        comparison_rows = list(paired_rows(preds,validation))
        result = dict(schema=SCHEMA,protocol=PROTOCOL,source_certificate=certificate,
            cache_metadata_sha256=digests[str(cache_dir/'cache.json')],cache_runtime=cache_meta['runtime'],
            model_sha256=fitted_hash,audit_model_sha256=sha256(output/'audit_model.json'),
            validation=validation_records,training_order_audit=audit_records,groups=groups,
            paired_validation=comparison_rows,screening=decision,
            model_statistics=model.get('statistics',{}),audit_statistics=audit_model.get('statistics',{}),
            model_updates=0,SUM_head_updates=0,validation_fits=0,test_evaluated=False)
        write_csv(output/'cases.csv',(dict(arm=arm,**row) for arm,row in validation_records.items()))
        write_csv(output/'comparisons.csv',comparison_rows)
        write_csv(output/'directions.csv',direction_rows(preds,validation))
        group_rows = [dict(group=name,arm=arm,**row) for name,rows in groups.items() for arm,row in rows.items()]
        write_csv(output/'groups.csv',group_rows)
        horizon_rows,block_rows,node_rows = [],[],[]
        for index in range(12):
            mask = np.arange(12)[None,:,None]==index
            horizon_rows.extend(paired_rows(preds,validation,mask,'horizon_'+str(index+1)))
        for start in range(0,837,128):
            mask = (np.arange(837)>=start)&(np.arange(837)<start+128)
            block_rows.extend(paired_rows(preds,validation,mask[:,None,None],'author_order_'+str(start)))
        for index,node in enumerate(archive['node_ids']):
            node_rows.append(dict(node_id=int(node),**paired(
                preds['ORIGIN'][:,:,index],preds[CANDIDATE][:,:,index],
                validation['y'][:,:,index],validation['valid'][:,:,index])))
        write_csv(output/'by_horizon.csv',horizon_rows); write_csv(output/'by_sample_block.csv',block_rows)
        write_csv(output/'by_node.csv',node_rows)
        # These origin-feature strata are label-free; target-flow masks above are diagnostic only.
        history_rows = []
        for name in ('mean_abs_change','neighbor_change_disagreement'):
            values = train['features'][name]; finite = values[np.isfinite(values)]
            val_values = validation['features'][name]
            if finite.size:
                cutpoints = np.quantile(finite,[.25,.5,.75])
                labels = np.searchsorted(cutpoints,val_values,side='right')
                labels[~np.isfinite(val_values)] = -1
            else:
                # An unavailable training descriptor has no usable fitted cuts.
                labels = np.full(val_values.shape,-1,dtype=np.int8)
            for group in range(-1,4):
                mask = labels==group
                history_rows.extend(paired_rows(preds,validation,mask[:,None,:],name+':'+str(group)))
        write_csv(output/'by_history.csv',history_rows)
        # These fitted statistics describe training support and fallback only.
        statistics = model.get('statistics',{})
        if isinstance(statistics,dict):
            write_csv(output/'coverage.csv',(dict(arm=arm,**row) for arm,row in statistics.items() if isinstance(row,dict)))
        require(sha256(output/'model.json')==fitted_hash,'Mapping changed during validation scoring')
        rehash(digests); assert_unchanged(certificate)
        write_json(output/'analysis.json',result); write_json(output/'screening.json',decision)
        (output/'report.md').write_text(report_text(result),encoding='utf-8')
        print(json.dumps(dict(status=decision['status'],output_dir=str(output),
            eligible_for_new_model_design=decision['eligible_for_new_model_design'])),flush=True)
        return result
    except BaseException as error:
        write_json(output/'failure.json',dict(status='GW10_PROBE_FAILED',error_type=type(error).__name__,
            error=str(error),source_artifacts_written=False,test_evaluated=False))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe-dir',type=Path,required=True)
    parser.add_argument('--cache-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args = parser.parse_args(argv)
    return run(args.probe_dir,args.cache_dir,args.output_dir)


if __name__ == '__main__': main()
