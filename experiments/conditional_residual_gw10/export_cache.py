"""Export frozen GW9 SUM predictions and origin features without model updates.

Server execution first replays the entire authenticated validation archive
exactly, then visits the training loader in its original author order. Training
targets are exported for the separately invoked residual experiment. No test
loader is constructed or evaluated; the inherited trusted author object array
does deserialize the complete payload, including test objects.
"""
import argparse
from datetime import datetime, timezone
import gc
import json
import logging
import os
from pathlib import Path
import uuid

import numpy as np

from experiments.branch_readout_gw9 import readout_auth as auth
from experiments.branch_readout_gw9 import readout_history as history


SCHEMA = 'gw10_frozen_sum_residual_cache_v1'
FEATURES = ('last_flow', 'mean_abs_change', 'neighbor_change_disagreement')
REPO = Path(__file__).resolve().parents[2]
SOURCE_FILES = ('experiments/conditional_residual_gw10/export_cache.py',
    'experiments/branch_readout_gw9/readout_auth.py',
    'experiments/branch_readout_gw9/readout_history.py')


def global_valid_mask(y):
    """Compute one mask from the complete split, never independently per batch."""
    y = np.asarray(y)
    auth.require(y.ndim == 3 and y.shape[1] == 12 and min(y.shape) > 0
        and y.dtype.kind == 'f' and np.isfinite(y).all(), 'Expected finite floating [sample,12,node] targets')
    null = float(y.min()) if y.min() < 1 else 0.
    mask = y != null
    auth.require(mask.any() and (y[mask] > 0).all(), 'Positive valid targets required by the global-null rule')
    return mask


def assert_exact_array(actual, expected, name):
    """Reject every dtype, shape, or value difference without tolerance changes."""
    actual, expected = np.asarray(actual), np.asarray(expected)
    auth.require(actual.dtype == expected.dtype and actual.shape == expected.shape,
        'Strict validation '+name+' dtype/shape differs: '+str(actual.dtype)+' '+str(actual.shape)
        +' versus '+str(expected.dtype)+' '+str(expected.shape))
    if not np.array_equal(actual, expected):
        mismatch = actual != expected
        first = tuple(int(i) for i in np.unravel_index(int(np.argmax(mismatch)), actual.shape))
        numeric = actual.dtype.kind in 'fiu'
        detail = dict(mismatch_count=int(np.count_nonzero(mismatch)), first_index=list(first))
        if numeric:
            detail['max_abs_difference'] = float(np.max(np.abs(
                actual.astype(np.float64)-expected.astype(np.float64))))
        raise ValueError('Strict validation '+name+' values differ: '+json.dumps(detail))


def assert_exact_replay(actual, certified_best):
    fields = {'y':'y', 'valid':'valid', 'sample_ids':'sample_ids', 'node_ids':'node_ids',
        'direct_support':'direct_support', 'frozen_R':'pred__FROZEN_R', 'pred':'pred__SUM_CONTROL'}
    for name, saved_name in fields.items():
        auth.require(name in actual and saved_name in certified_best, 'Missing strict replay field: '+name)
        assert_exact_array(actual[name], certified_best[saved_name], name)
    return dict(exact_equal=True, fields={name:True for name in fields},
        policy='same_dtype_same_shape_all_values_equal_without_tolerance',
        validation_samples=int(len(actual['y'])), inference_before_training_export=True)


def require_cache_output(cache_dir, probe_dir, data_dir, certificate):
    output = auth.require_output_outside_sources(cache_dir, probe_dir, certificate)
    data_dir = Path(data_dir).resolve()
    auth.require(output != data_dir and data_dir not in output.parents and output not in data_dir.parents,
        'Cache output must be outside and must not contain the authenticated data directory')
    return output


def atomic_npz(path, archive):
    """Publish a completed NPZ in the same fresh output directory."""
    path = Path(path)
    auth.require(not path.exists(), 'Existing cache archives are immutable: '+str(path))
    temporary = path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        with temporary.open('xb') as stream:
            np.savez_compressed(stream, **archive)
            stream.flush(); os.fsync(stream.fileno())
        auth.require(not path.exists(), 'Cache archive appeared while exporting')
        temporary.replace(path)
    finally:
        if temporary.exists(): temporary.unlink()


def atomic_json(path, value):
    path = Path(path)
    auth.require(not path.exists(), 'Existing cache certificate is immutable: '+str(path))
    temporary = path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        with temporary.open('x', encoding='utf-8', newline='\n') as stream:
            stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')
            stream.flush(); os.fsync(stream.fileno())
        auth.require(not path.exists(), 'Cache certificate appeared while exporting')
        temporary.replace(path)
    finally:
        if temporary.exists(): temporary.unlink()


def _data_digests(directory, identity):
    digests = {}
    directory = Path(directory).resolve()
    for name, expected in identity['hashes'].items():
        path = (directory/name).resolve()
        auth.require(path.parent == directory and path.is_file() and auth.sha256(path) == expected,
            'Authenticated export data changed: '+name)
        digests[str(path)] = expected
    return digests


def _assert_digests(digests):
    for path, expected in digests.items():
        auth.require(auth.sha256(path) == expected, 'Input changed during export: '+path)


def _check_frozen(extractor, head, base, expected_backbone, expected_head):
    auth.require(not any(module.training for module in extractor.modules())
        and not any(module.training for module in head.modules()), 'Frozen export left evaluation mode')
    auth.require(not any(parameter.requires_grad or parameter.grad is not None
        for parameter in extractor.anchor.parameters()), 'Frozen source backbone acquired gradients or trainability')
    auth.require(all(parameter.grad is None for parameter in head.parameters()), 'Frozen SUM head acquired gradients')
    auth.require(base.tensor_tree_hash(extractor.anchor.state_dict()) == expected_backbone
        and base.tensor_tree_hash(head.state_dict()) == expected_head, 'Frozen weights or buffers changed during export')
    extractor.assert_cache_clear()


def _collect_split(split, data, extractor, head, adjacency, torch, base, device_batch,
        device, expected_backbone, expected_head, baseline=None):
    auth.require(split in ('train', 'val'), 'Only train and validation splits may be exported')
    descriptor = data.identity['splits'][split]
    start, stop = descriptor['bounds_half_open']
    auth.require(descriptor['selected_indices'] == list(range(start,stop))
        and descriptor['count'] == stop-start, 'Export split must preserve the complete author order')
    loader = data.loaders[split+'_loader']
    auth.require(loader.num_batch == len(descriptor['partition'])
        and len(loader.samples) == descriptor['count'], 'Loader and authenticated batch partition differ')
    predictions, truths, supports = [], [], []
    frozen_predictions = [] if baseline is not None else None
    features = {name:[] for name in FEATURES}
    partitions = []
    normalization = data.identity['normalization']
    auth.require(normalization.get('refitted') is False, 'Source normalization must remain unchanged')
    _check_frozen(extractor,head,base,expected_backbone,expected_head)
    # No shuffle, no optimizer, no backward; constructors retain certified
    # requires_grad flags for the head because they can affect replay kernels.
    with torch.no_grad():
        for batch_index, raw in enumerate(loader.get_iterator(), 1):
            raw_history = np.asarray(raw['x_data'])
            auth.require(raw_history.ndim == 4 and raw_history.shape[1:] == (12,496,3)
                and raw_history.dtype == np.float32 and np.isfinite(raw_history).all(),
                'Invalid complete historical x_data batch')
            partitions.append(len(raw_history))
            origin = history.feature_arrays(raw_history[...,0],normalization['mean'],normalization['std'],adjacency)
            for name in FEATURES:
                value = np.asarray(origin[name],dtype=np.float32)
                auth.require(value.shape == (len(raw_history),496) and not np.isinf(value).any(),
                    'Invalid origin descriptor: '+name)
                if name != 'neighbor_change_disagreement':
                    auth.require(np.isfinite(value).all(), 'Nonfinite origin traffic descriptor: '+name)
                features[name].append(value)
            x, y, incident, sensors = device_batch(raw, device)
            capture = extractor.capture_features(x,incident_data=incident,sensor_data=sensors)
            auth.require(all(bool(torch.isfinite(value).all()) for value in capture.values()),
                'Nonfinite frozen model features')
            normalized = capture['base']+head(capture['S'],capture['Q'])
            prediction = data.scaler.inverse_transform(normalized).squeeze(-1).cpu().numpy()
            truth = data.scaler.inverse_transform(y).squeeze(-1).cpu().numpy()
            auth.require(prediction.shape == truth.shape and prediction.dtype == truth.dtype == np.float32
                and np.isfinite(prediction).all() and np.isfinite(truth).all(), 'Invalid frozen prediction or target')
            predictions.append(prediction); truths.append(truth)
            supports.append((incident['distances'].abs().sum(-1).cpu().numpy() > 0))
            if frozen_predictions is not None:
                frozen_predictions.append(data.scaler.inverse_transform(capture['base']).squeeze(-1).cpu().numpy())
            if batch_index == 1 or batch_index % 10 == 0 or batch_index == loader.num_batch:
                print(json.dumps(dict(phase='GW10_FROZEN_CACHE_EXPORT',split=split,
                    batch=batch_index,total=loader.num_batch,model_updates=0)),flush=True)
    auth.require(partitions == descriptor['partition'], 'Observed export batches differ from authenticated source partition')
    _check_frozen(extractor,head,base,expected_backbone,expected_head)
    truth = np.concatenate(truths)
    result = dict(y=truth,pred=np.concatenate(predictions),valid=global_valid_mask(truth),
        sample_ids=np.asarray(descriptor['selected_indices'],dtype=np.int64),
        node_ids=data.station_ids.copy(),direct_support=np.concatenate(supports),
        **{'feature__'+name:np.concatenate(values) for name,values in features.items()})
    auth.require(result['y'].shape == (descriptor['count'],12,496), 'Incomplete split cache')
    if baseline is not None:
        result['frozen_R'] = np.concatenate(frozen_predictions)
        replay = assert_exact_replay(result,baseline)
        del result['frozen_R']
        # Retain the already-certified mask after proving the global rule agrees.
        result['valid'] = baseline['valid'].copy()
    else:
        replay = None
    return result,replay


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe-dir',type=Path,required=True)
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--cache-dir',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--threads',type=int,default=3)
    args = parser.parse_args(argv)
    if args.threads < 1: parser.error('threads must be positive')
    for name in ('probe_dir','data_dir','cache_dir'): setattr(args,name,getattr(args,name).resolve())
    return args


def main(argv=None):
    args = parse_args(argv)
    best,last,certificate = auth.load_authenticated_run(args.probe_dir,REPO)
    del last
    output = require_cache_output(args.cache_dir,args.probe_dir,args.data_dir,certificate)
    implementation = {name:auth.sha256(REPO/name) for name in SOURCE_FILES}
    source_files = _data_digests(args.data_dir,certificate['data_identity'])
    output.mkdir(parents=True)
    try:
        # Runtime/model imports occur only after the fixed completed probe and
        # output location have been authenticated. Contract tests import NumPy
        # helpers above without importing Torch or any model module.
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        import torch
        from experiments.branch_readout_gw9 import probe, train
        from experiments.branch_readout_gw9.model import ReadoutHead
        from experiments.branch_readout_gw9.reference import recertify, read_json
        from experiments.chronological.graph_message_reuse import RUNTIME_KEYS
        from experiments.chronological.original_igstgnn_runtime import device_batch

        base = train.base
        old_identity,gw4,recurrent = recertify(certificate['source_reference'])
        probe_identity = read_json(args.probe_dir/'identity.json')
        auth.require(probe_identity['identity_sha256'] == certificate['source_identity_sha256'], 'Probe identity changed')
        seed = int(probe_identity['runtime']['seed'])
        runtime = base.configure_runtime(args.device,args.threads,seed)
        for key in RUNTIME_KEYS:
            auth.require(runtime.get(key) == old_identity['runtime'].get(key)
                == probe_identity['runtime'].get(key), 'Export/source runtime differs: '+key)
        modules = base.get_original_modules()
        auth.require(modules.source_fingerprint == old_identity['source_fingerprint']
            == probe_identity['source_fingerprint'], 'Original source fingerprint differs')
        config = dict(base.CONFIG,device=args.device,seed=seed,warm_epoch=101)
        data = base.load_data(args.data_dir,config,modules,logging.getLogger('gw10-export'),check=False,training=True)
        auth.require(data.identity == old_identity['data'] == certificate['data_identity'], 'Export/source data identity differs')
        assert_exact_array(data.station_ids,best['node_ids'],'source_node_ids')
        roads = np.full(len(data.station_ids),'unknown',dtype='<U40')
        for road,indices in data.roads.items(): roads[indices] = road
        assert_exact_array(roads,best['road_labels'],'source_road_labels')
        extractor = train.model_for_arm(data,modules,args.device,'R_CONTINUE',gw4,recurrent,hidden=16,stage='probe')
        del gw4,recurrent; gc.collect()
        head = ReadoutHead(extractor.channels,extractor.gap,'SUM_CONTROL',hidden=16).to(args.device)
        state = base.load_checkpoint(args.probe_dir/'last.pt')
        probe.validate_saved(state,probe_identity,args.probe_dir,len(data.identity['splits']['train']['partition']))
        weights,role = probe.selected_role(state,args.probe_dir,'SUM_CONTROL','best')
        auth.require(role == certificate['head_roles']['best']['SUM_CONTROL'], 'Selected SUM head role differs')
        head.load_state_dict(weights,strict=True)
        del weights,state; gc.collect()
        extractor.eval(); head.eval()
        expected_backbone = certificate['source_reference']['source_initialization']['model_tensor_sha256']
        expected_head = role['head_tensor_sha256']
        _check_frozen(extractor,head,base,expected_backbone,expected_head)
        adjacency = np.load(args.data_dir/'adj_matrix.npy',allow_pickle=False)
        archives = {}
        validation,replay = _collect_split('val',data,extractor,head,adjacency,torch,base,
            device_batch,args.device,expected_backbone,expected_head,baseline=best)
        atomic_npz(output/'val.npz',validation)
        archives['val'] = dict(path='val.npz',sha256=auth.sha256(output/'val.npz'),
            shape=list(validation['y'].shape),samples=len(validation['y']),
            inference_batches=data.loaders['val_loader'].num_batch)
        del validation,best; gc.collect()
        training,_ = _collect_split('train',data,extractor,head,adjacency,torch,base,
            device_batch,args.device,expected_backbone,expected_head)
        atomic_npz(output/'train.npz',training)
        archives['train'] = dict(path='train.npz',sha256=auth.sha256(output/'train.npz'),
            shape=list(training['y'].shape),samples=len(training['y']),
            inference_batches=data.loaders['train_loader'].num_batch)
        del training; gc.collect()
        _check_frozen(extractor,head,base,expected_backbone,expected_head)
        auth.assert_unchanged(certificate); _assert_digests(source_files)
        _assert_digests({str(REPO/name):digest for name,digest in implementation.items()})
        result = dict(schema=SCHEMA,status='GW10_FROZEN_SUM_CACHE_COMPLETE',
            source_identity_sha256=certificate['source_identity_sha256'],source_certificate=certificate,
            data_identity=data.identity,runtime=runtime,selected_head_role=role,
            backbone_tensor_sha256=expected_backbone,head_tensor_sha256=expected_head,
            archives=archives,frozen_replay=replay,features=list(FEATURES),axis_order='sample,horizon,node',
            feature_axis_order='sample,node',feature_source='x_data_traffic_feature_zero_at_forecast_origin',
            feature_precision='float64_history_descriptors_then_float32_storage',
            isolated_neighbor_descriptor='NaN_unavailable_not_imputed_zero',
            training_sample_order='original_author_order_no_shuffle',
            train_valid_mask='one_complete_training_split_native_global_null_rule',
            validation_valid_mask='strict_replay_then_copy_authenticated_GW9_BEST_mask',
            model_updates=0,optimizer_steps=0,gradient_mode='no_grad_preserve_certified_parameter_flags',
            validation_replayed_before_train_export=True,
            complete_forward_calls=sum(row['inference_batches'] for row in archives.values()),
            source_artifacts_written=False,test_evaluated=False,test_loader_constructed=False,
            test_objects_deserialized=True,test_samples_evaluated=False,
            failed_GW9_joint_gate_unchanged=True,cache_execution_source=implementation,
            data_file_digests=source_files,created_utc=datetime.now(timezone.utc).isoformat())
        atomic_json(output/'cache.json',result)
        print(json.dumps(dict(status=result['status'],cache_dir=str(output),
            complete_forward_calls=result['complete_forward_calls'],model_updates=0,test_evaluated=False)),flush=True)
        return result
    except BaseException as error:
        if not (output/'failure.json').exists():
            atomic_json(output/'failure.json',dict(schema=SCHEMA,status='GW10_CACHE_EXPORT_FAILED',
                error_type=type(error).__name__,error=str(error),model_updates=0,test_evaluated=False,
                partial_files_are_not_a_completed_cache=True))
        raise


if __name__ == '__main__': main()
