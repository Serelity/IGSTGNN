"""Archive paired v03 validation/calibration predictions, without training.

Calibration is available only when reserved before fitting every candidate and
fallback. Split row IDs are authenticated author-array identities, not certified
event identities or publication timestamps. Test samples are never evaluated.
"""
import argparse
from contextlib import nullcontext
import gc
import json
import logging
from pathlib import Path
import sys
import time

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.chronological.original_igstgnn_source import get_original_modules
from experiments.chronological.original_igstgnn_runtime import (
    configure_runtime, default_data_dir, device_batch, load_data, sha256, tensor_tree_hash, write_json)
from experiments.chronological.unified_adapter_data_v03 import make_loaders, device_batch_v03, encode_report
from experiments.chronological.train_unified_adapter_v03 import (
    EXECUTION_FILES, DiagnosticsAccumulator, model_for_arm, partition_training,
    load_checkpoint, validate_saved)
from experiments.chronological.evaluate_unified_adapter_v03 import (
    ARCHIVE_SCHEMA, AXIS_ORDER, analyze_archive, validate_archive)


def available_semantic_pair(sample, nodes):
    report = encode_report(sample, nodes)
    return report['incident'][1:3] if report['available'] else np.full(2, -1, np.float32)


def matched_semantic_donors(samples, road_labels, seed=2025):
    """Split-local cyclic neighbors matched by road, known clock, and history.

    Only Type/Description pairs move. Singleton strata remain unchanged and all
    actual change rates are reported. No y_data, errors or future fields read.
    """
    rows = len(samples)
    labels = np.asarray(road_labels)
    groups, histories, pairs = {}, [], []
    for i, sample in enumerate(samples):
        x = np.asarray(sample['x_data'])
        encoded = encode_report(sample, x.shape[1])
        support = np.abs(encoded['distances']).sum(-1) > 0
        unique, counts = np.unique(labels[support], return_counts=True)
        road = str(unique[np.argmax(counts)]) if len(unique) else 'NO_SUPPORT'
        hour, day = int(np.floor(float(x[-1, 0, 1])*24)), int(np.floor(float(x[-1, 0, 2])*7))
        groups.setdefault((road, hour, day, encoded['available']), []).append(i)
        histories.append(float(x[..., 0].mean()))
        pairs.append(available_semantic_pair(sample, x.shape[1]))
    donors = np.arange(rows)
    rng = np.random.default_rng(seed)
    for members in groups.values():
        if len(members) > 1:
            # Seeded tie breaks; neighboring histories share a cycle, with the
            # wraparound mismatch disclosed through the distance statistic.
            members = sorted(members, key=lambda i: (histories[i], rng.random()))
            donors[members] = np.roll(members, 1)
    pairs = np.asarray(pairs)
    return donors, dict(schema='split_local_road_clock_history_semantic_cycle_v03',
        uses_targets=False, pair_preserved=True, position_preserved=True,
        seed=seed, rows=rows, moved_rows=int((donors != np.arange(rows)).sum()),
        numerically_changed_pairs=int(np.any(pairs != pairs[donors], axis=-1).sum()),
        singleton_strata=sum(len(m) == 1 for m in groups.values()),
        historical_mean_distance_abs_mean=float(np.abs(np.asarray(histories)-np.asarray(histories)[donors]).mean()),
        semantic_constancy_can_make_intervention_uninformative=True,
        distribution_shift_still_possible=True)


def archive_features(samples, horizon=12):
    x = np.stack([sample['x_data'] for sample in samples]).astype(np.float32)
    rows, _, nodes, _ = x.shape
    values = x[..., 0]
    common = np.stack((values[:, -1], values.mean(1), values[:, -1]-values[:, 0],
                       values.std(1), x[:, -1, :, 1], x[:, -1, :, 2]), -1)
    h = np.broadcast_to(common[:, None], (rows, horizon, nodes, common.shape[-1]))
    reports = [encode_report(sample, nodes) for sample in samples]
    relations = np.stack([row['distances'] for row in reports])
    flags = np.asarray([[row['available'],row['relation_available']] for row in reports],np.float32)
    relations = np.concatenate((relations,np.broadcast_to(flags[:,None],(rows,nodes,2)),
                               (np.abs(relations).sum(-1)>0)[...,None]),-1)
    r = np.broadcast_to(relations[:, None], (rows, horizon, nodes, 6))
    holiday = np.asarray([row['incident'][3] for row in reports],np.float32)
    public_holiday = np.broadcast_to(holiday[:,None,None],(rows,horizon,nodes))
    known_horizon = np.broadcast_to(np.arange(1,horizon+1,dtype=np.float32)[None,:,None]/horizon,(rows,horizon,nodes))
    cats = np.asarray([[int(row['incident'][1]) if row['incident'][1] >= 0 else 29,
                        int(row['incident'][2]) if row['incident'][2] >= 0 else 6,
                        row['position'] if row['position'] >= 0 else 12] for row in reports], dtype=np.int16)
    # Unavailable report content is unknown even if the archival record has a
    # populated final category; the feature side channel must obey availability.
    absent = np.array([not row['available'] for row in reports])
    cats[absent] = [29, 6, 12]
    c = np.broadcast_to(cats[:, None, None], (rows, horizon, nodes, 3))
    return dict(feature__H_HISTORY=h, feature__H_HORIZON=known_horizon,
                feature__H_HOLIDAY=public_holiday, feature__R_RELATION=r, feature__C_CATEGORY=c), {
        'H_HISTORY': dict(source='history', available_at_cutoff=True, information_channel='H',
            fields=['last_flow_normalized', 'mean_flow_normalized', 'endpoint_change_normalized',
                    'flow_std_normalized', 'known_tod', 'known_dow']),
        'H_HORIZON': dict(source='horizon',available_at_cutoff=True,information_channel='H'),
        'H_HOLIDAY': dict(source='clock',available_at_cutoff=True,information_channel='H'),
        'R_RELATION': dict(source='report', available_at_cutoff=True, information_channel='R',
            fields=['relation_0','relation_1','relation_2','report_available','relation_available','direct_support'],
            cutoff_semantics='assumed_author_release_not_independently_certified'),
        'C_CATEGORY': dict(source='report', available_at_cutoff=True, information_channel='C',
            fields=['Description', 'Type', 'incident_position'], categorical_cardinalities=[30, 7, 13],
            cutoff_semantics='assumed_author_release_not_independently_certified')}


def checkpoint_for(run, state, selection):
    if selection == 'best':
        path = run/state['best_checkpoint_file']
        if path.parent != run or sha256(path) != state['best_checkpoint_sha256']:
            raise ValueError('Best checkpoint identity mismatch')
        return path, load_checkpoint(path), state['best_epoch'], sha256(path)
    path = run/'last.pt'
    return path, state['model_state'], state['completed_epoch'], state['model_tensor_sha256']


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, default=default_data_dir())
    parser.add_argument('--device')
    parser.add_argument('--checkpoint', choices=('best', 'last'), default='best')
    parser.add_argument('--split', choices=('validation', 'calibration'), default='validation')
    parser.add_argument('--no-interventions', action='store_true')
    parser.add_argument('--no-features', action='store_true')
    parser.add_argument('--validation-reference',type=Path,
                        help='Required PASS validation metadata for matching calibration identities')
    args = parser.parse_args(argv)
    args.run_dir, args.output_dir = args.run_dir.resolve(), args.output_dir.resolve()
    if args.output_dir.exists():
        parser.error('Use a fresh output directory')
    if args.split=='calibration' and args.validation_reference is None:
        parser.error('Calibration archives require an already replay-passed --validation-reference')
    return args


def main(argv=None):
    args = parse_args(argv)
    identity = json.loads((args.run_dir/'identity.json').read_text(encoding='utf-8'))
    if identity.get('schema') != 'unified_adapter_matched_training_v03':
        raise ValueError('Only immutable v03 run identities are supported')
    for path in EXECUTION_FILES:
        if sha256(REPO/path) != identity['execution_source'].get(path):
            raise ValueError('Training source changed: '+path)
    device = args.device or identity['runtime']['python_device']
    runtime = configure_runtime(device, identity['runtime']['threads'], identity['training_config']['seed'])
    modules = get_original_modules()
    logger = logging.getLogger('v03-readonly-predict')
    data = load_data(args.data_dir, identity['training_config'], modules, logger,
                     check=identity['engineering_only'], training=True)
    data = partition_training(data, modules, logger, identity['holdout_rows'])
    if data.identity != identity['data']:
        raise ValueError('Prediction data identity differs from training')
    candidates = make_loaders(data, modules, logger)
    if args.split == 'calibration':
        if not identity['holdout_rows']:
            raise ValueError('No calibration rows were reserved before candidate fitting')
        samples = data.loaders['calibration_loader'].samples
        # Build independent candidate calibration batches without modifying the
        # old or fitting loaders. Existing factory intentionally accepts train/val.
        calibration_data = copy_data_with_validation(data, data.loaders['calibration_loader'])
        candidate_loader = make_loaders(calibration_data, modules, logger)['val_loader']
        native_loader = data.loaders['calibration_loader']
        ids = identity['data']['splits']['calibration']['selected_indices']
    else:
        samples, native_loader = data.loaders['val_loader'].samples, data.loaders['val_loader']
        candidate_loader = candidates['val_loader']
        ids = identity['data']['splits']['val']['selected_indices']
    states = {arm: load_checkpoint(args.run_dir/arm/'last.pt') for arm in identity['arms']}
    if 'ORIGINAL' not in states:
        raise ValueError('A shared ORIGINAL reference is required for fixed hard windows')
    budget = json.loads((args.run_dir/'budget.json').read_text())['target_completed_epochs']
    if len({s['completed_epoch'] for s in states.values()}) != 1 or min(s['completed_epoch'] for s in states.values()) < budget:
        raise ValueError('Complete all registered arms to a common budget before paired evaluation')
    for arm, state in states.items():
        validate_saved(state, identity, arm, args.run_dir/arm, data.loaders['train_loader'].num_batch)
    road_labels = np.full(len(data.station_ids), 'unknown', dtype='<U40')
    for road, indices in data.roads.items():
        road_labels[indices] = road
    donors, shuffle = matched_semantic_donors(samples, road_labels, identity['training_config']['seed'])
    semantic_values = np.stack([available_semantic_pair(s, len(data.station_ids)) for s in samples])
    archive = dict(schema=np.array(ARCHIVE_SCHEMA), axis_order=np.array(AXIS_ORDER),
        sample_ids=np.asarray(ids, dtype=np.int64), node_ids=data.station_ids, road_labels=road_labels,
        direct_support=np.stack([np.abs(encode_report(s, len(data.station_ids))['distances']).sum(-1) > 0 for s in samples]))
    adjacency=np.load(data.directory/'adj_matrix.npy',allow_pickle=False)>0
    union=adjacency|adjacency.T
    np.fill_diagonal(union,False)
    direct=archive['direct_support']
    envelope=direct|((direct.astype(np.int32)@union.T.astype(np.int32))>0)
    archive.update(group__OPERATOR_ENVELOPE=envelope,
                   group__POTENTIAL_ONLY=envelope&~direct,group__OUTSIDE_OPERATOR=~envelope)
    roles, mechanics, timings, replay = {}, {}, {}, {}
    expected_y = None
    for arm, state in states.items():
        model = model_for_arm(data, modules, device, arm)
        path, weights, epoch, checkpoint_hash = checkpoint_for(args.run_dir/arm, state, args.checkpoint)
        model.load_state_dict(weights, strict=True)
        model.eval()
        roles[arm] = dict(checkpoint_sha256=checkpoint_hash, epoch=epoch, selection=args.checkpoint)
        conditions = {arm: {}}
        if arm == 'BOTH' and not args.no_interventions:
            conditions.update(BOTH_TIME_OFF={'time': False}, BOTH_GRAPH_OFF={'graph': False},
                BOTH_REPORT_OFF={'report': False}, BOTH_SEMANTIC_UNK={'semantic': False},
                BOTH_SEMANTIC_SHUFFLE={'shuffle': True})
        for case, settings in conditions.items():
            predictions, targets, times = [], [], []
            stats, offset = DiagnosticsAccumulator(), 0
            loader = native_loader if arm == 'ORIGINAL' else candidate_loader
            for raw in loader.get_iterator():
                x,y,incident,sensors = (device_batch(raw, device) if arm == 'ORIGINAL' else device_batch_v03(raw, device))
                if settings.get('shuffle'):
                    incident = dict(incident, incident=incident['incident'].clone())
                    values = semantic_values[donors[offset:offset+len(x)]]
                    replacement = torch.as_tensor(values, device=device)
                    incident['incident'][:, 1:3] = torch.where(incident['available'][:, None], replacement, -1.)
                intervention = {key:value for key,value in settings.items() if key != 'shuffle'}
                context = model.intervention(**intervention) if intervention else nullcontext()
                if torch.device(device).type == 'cuda':
                    torch.cuda.synchronize()
                started = time.perf_counter()
                with torch.no_grad(), context:
                    prediction = model(x, incident_data=incident, sensor_data=sensors)
                if torch.device(device).type == 'cuda':
                    torch.cuda.synchronize()
                times.append(dict(batch_size=len(x), seconds=time.perf_counter()-started))
                predictions.append(data.scaler.inverse_transform(prediction).squeeze(-1).cpu().numpy())
                targets.append(data.scaler.inverse_transform(y).squeeze(-1).cpu().numpy())
                if hasattr(model, 'batch_diagnostics'):
                    stats.add(model.batch_diagnostics())
                offset += len(x)
            prediction, truth = np.concatenate(predictions), np.concatenate(targets)
            if expected_y is None:
                expected_y = truth
            elif not np.array_equal(expected_y, truth):
                raise ValueError('Cases differ in saved targets/order')
            archive['pred__'+case] = prediction
            mechanics[case], timings[case] = stats.result(), dict(records=times,
                scope='synchronized_forward_excludes_H2D_inverse_statistics',
                dedicated_benchmark=False, warmup_not_removed=True)
            print(json.dumps(dict(case=case, rows=len(prediction), split=args.split)), flush=True)
        del model, weights
        gc.collect()
        if torch.device(device).type == 'cuda':
            torch.cuda.empty_cache()
    null = float(expected_y.min()) if expected_y.min() < 1 else 0.
    valid = expected_y != null
    if np.any(expected_y[valid] <= 0):
        raise ValueError('Original null mask includes nonpositive MAPE targets; do not silently change the mask')
    archive.update(y=expected_y, valid=valid)
    manifest = {}
    if not args.no_features:
        features, manifest = archive_features(samples)
        archive.update(features)
    eligible = bool(identity['holdout_rows'])
    metadata = dict(split=args.split, test_evaluated=False, engineering_only=identity['engineering_only'],
        source='authenticated_author_release_v03_run', axis_identity='author_array_row_and_station_id',
        mask_definition='frozen_engine_labels_not_equal_global_min_if_min_below_one_else_zero',
        mask_null_value=null, run_identity_sha256=identity['identity_sha256'],
        sample_identity_verified=True, group_identity_verified=False,
        eligible_for_gain_fit=eligible, candidate_frozen_before_calibration=True,
        calibration_used_for_candidate_fit=False if eligible else None,
        calibration_used_for_candidate_selection=False if eligible else None,
        candidate_training_sample_ids=identity['data']['splits']['train']['selected_indices'],
        base_training_sample_ids=identity['data']['splits']['train']['selected_indices'],
        candidate_checkpoint_sha256=roles.get('BOTH', {}).get('checkpoint_sha256'),
        base_checkpoint_sha256=roles.get('H_ONLY', {}).get('checkpoint_sha256'),
        prediction_case_roles=roles, feature_manifest=manifest, matched_semantic_shuffle=shuffle,
        absolute_order_verified=False, report_publication_cutoff_verified=False,
        auxiliary_holdout_scope='row_holdout_not_event_independent', runtime=runtime,
        code_fingerprint=identity['execution_source'], training_data_identity=identity['data'],
        producer_source={p:sha256(REPO/p) for p in (
            'experiments/chronological/predict_unified_adapter_v03.py',
            'experiments/chronological/evaluate_unified_adapter_v03.py')})
    validate_archive(archive)
    if args.split == 'validation':
        # Authenticate ALL native saved metrics before any consumable archive
        # is published. An interrupted/failed replay has no eligible metadata.
        for arm, state in states.items():
            error = np.abs(expected_y.astype(np.float64)-archive['pred__'+arm])
            actual = dict(mae=float(error[valid].mean()),rmse=float(np.sqrt((error[valid]**2).mean())),
                          mape_fraction=float((error/np.where(valid,expected_y,1))[valid].mean()))
            saved = state['history'][roles[arm]['epoch']-1]
            expected = {key:saved['validation_'+key] for key in actual}
            passed = all(np.isclose(actual[key],expected[key],atol=1e-5,rtol=1e-6) for key in actual)
            replay[arm] = dict(actual=actual,expected=expected,passed=bool(passed))
        if not all(row['passed'] for row in replay.values()):
            args.output_dir.mkdir(parents=True)
            write_json(args.output_dir/'failure.json',dict(status='REPLAY_FAILED',replay=replay))
            raise ValueError('Saved validation replay failed; no eligible archive published')
        metadata['validation_replay']='PASS'
    else:
        reference=json.loads(args.validation_reference.read_text(encoding='utf-8'))
        if reference.get('producer_status')!='PASS' or reference.get('validation_replay')!='PASS':
            raise ValueError('Calibration reference is not a replay-passed validation archive')
        for key in ('run_identity_sha256','prediction_case_roles','feature_manifest','code_fingerprint','producer_source'):
            if metadata[key]!=reference.get(key):
                raise ValueError('Calibration reference differs: '+key)
        reference_archive=args.validation_reference.parent/'paired_predictions.npz'
        if sha256(reference_archive)!=reference.get('archive_sha256'):
            raise ValueError('Validation reference archive hash differs')
        metadata['validation_replay']='PASS'
        metadata['validation_reference_sha256']=sha256(args.validation_reference)
    metadata['producer_status']='PASS'
    args.output_dir.mkdir(parents=True)
    path=args.output_dir/'paired_predictions.npz'
    with path.with_suffix('.npz.tmp').open('wb') as stream:
        np.savez_compressed(stream,**archive)
    path.with_suffix('.npz.tmp').replace(path)
    metadata['archive_sha256']=sha256(path)
    write_json(args.output_dir/'metadata.json',metadata)
    write_json(args.output_dir/'mechanisms.json',mechanics)
    write_json(args.output_dir/'timings.json',timings)
    if args.split=='validation':
        write_json(args.output_dir/'replay.json',replay)
        analyze_archive(path,args.output_dir/'analysis',metadata=metadata)
    else:
        write_json(args.output_dir/'calibration_role.json', dict(never_candidate_fit_or_selection=True,
            only_for_auxiliary_head_fit=True, independent_events_certified=False))
    return metadata


def copy_data_with_validation(data, loader):
    from copy import copy
    result = copy(data)
    result.loaders = {'val_loader': loader}
    return result


if __name__ == '__main__':
    main()
