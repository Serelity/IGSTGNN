"""GW2 dynamic evaluation with zero new GW1 baseline inference calls."""
import argparse
from contextlib import nullcontext
import gc
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import numpy as np
import torch

from experiments.chronological import evaluate_original_graph_message as base
from experiments.chronological import evaluate_original_graph_window as gw1_eval
from experiments.chronological import train_original_graph_message as train
from experiments.chronological.graph_message_reuse import (
    NEW_ARMS, REUSED_ARMS, load_reference, new_states, read_json, require, sidecar)


SCHEMA = 'original_igstgnn_graph_message_reuse_archive_v1'
CASES = ('ORIGINAL', 'STATIC', 'POST_LOCAL', 'POST_MESSAGE', 'POST_LOCAL_UNIFORM',
         'POST_MESSAGE_UNIFORM', 'POST_MESSAGE_STATIC_ONLY')
COMPARISONS = tuple((b+'_vs_'+a, a, b) for a, b in train.PAIRS) + (
    ('local_selector_dependency', 'POST_LOCAL_UNIFORM', 'POST_LOCAL'),
    ('same_checkpoint_selector_dependency', 'POST_MESSAGE_UNIFORM', 'POST_MESSAGE'),
    ('same_checkpoint_content_dependency', 'POST_MESSAGE_STATIC_ONLY', 'POST_MESSAGE'))


def cached_predictions(reference, selection, identity, states, modules):
    saved = reference['archives'][selection]
    path = Path(saved['path'])
    require(base.sha256(path) == saved['sha256'], 'Cached GW1 predictions changed')
    with np.load(path, allow_pickle=False) as npz:
        source = dict(npz)
    gw1_eval.validate_archive(source)
    require(np.array_equal(source['sample_ids'], identity['data']['splits']['val']['selected_indices']),
            'Cached GW1 validation IDs/order differ')
    archive = {key: source[key] for key in ('y', 'valid', 'sample_ids', 'node_ids', 'road_labels', 'direct_support')}
    archive.update(schema=np.array(SCHEMA), axis_order=np.array('sample,horizon,node'))
    checks = {}
    for arm in REUSED_ARMS:
        archive['pred__'+arm] = source['pred__'+arm]
        epoch = states[arm]['best_epoch'] if selection == 'best' else states[arm]['completed_epoch']
        checks[arm] = dict(base.replay_metrics(source['pred__'+arm], source['y'], states[arm], epoch, modules),
            mode='cached_predictions_metric_check_not_new_checkpoint_forward', new_inference_calls=0)
        require(checks[arm]['passed'], 'Cached metrics differ from saved history: '+arm)
    require(base.sha256(path) == saved['sha256'], 'Source prediction archive changed during read')
    return archive, checks


def existing_new_cache(run, identity, state, arm, selection, reference_archive):
    """Reuse completed GW2 prediction states too, when roles and axes match."""
    epoch = state['best_epoch'] if selection == 'best' else state['completed_epoch']
    expected_sha = state['best_checkpoint_sha256'] if selection == 'best' else base.sha256(run/arm/'last.pt')
    needed = [arm, arm+'_UNIFORM'] + (['POST_MESSAGE_STATIC_ONLY'] if arm == 'POST_MESSAGE' else [])
    for path in sorted(run.glob('*/metadata.json'), reverse=True):
        metadata = read_json(path)
        if metadata.get('status') not in ('GW2_PREDICTION_PASS', 'GW2_REUSE_PREDICTION_PASS'):
            continue
        if metadata.get('run_identity_sha256') != identity['identity_sha256']:
            continue
        role = metadata.get('checkpoint_roles', {}).get(arm, {})
        if (role.get('selection'), role.get('epoch'), role.get('checkpoint_sha256')) != (selection, epoch, expected_sha):
            continue
        if metadata.get('validation_replay', {}).get(arm, {}).get('passed') is not True:
            continue
        archive_path = path.parent/'paired_predictions.npz'
        require(base.sha256(archive_path) == metadata.get('archive_sha256'), 'Existing GW2 archive digest differs')
        with np.load(archive_path, allow_pickle=False) as npz:
            archive = dict(npz)
        reused = metadata['status'] == 'GW2_REUSE_PREDICTION_PASS'
        base.validate_archive(archive, CASES if reused else base.CASES, SCHEMA if reused else base.SCHEMA)
        require(all(np.array_equal(archive[key], reference_archive[key]) for key in (
            'y', 'valid', 'sample_ids', 'node_ids', 'road_labels', 'direct_support')), 'Existing GW2 cache axes/mask differ')
        require(all('pred__'+case in archive for case in needed), 'Existing GW2 intervention states missing')
        require(base.sha256(archive_path) == metadata['archive_sha256'], 'Existing GW2 cache changed during read')
        return archive, metadata, path.parent
    return None


def produce(args, output):
    identity = read_json(args.run_dir/'identity.json')
    require(identity.get('schema') == train.SCHEMA and identity.get('design') == train.DESIGN, 'GW2 run identity required')
    for name in train.EXECUTION_FILES:
        require(identity['execution_source'].get(name) == base.sha256(REPO/name), 'Training execution source changed: '+name)
    saved_reference = read_json(sidecar(args.run_dir)/'reference.json')
    epochs = saved_reference['completed_epochs']
    reference, source_identity, old_states = load_reference(saved_reference['source_run'], epochs, target=identity,
        archives={s: saved_reference['archives'][s]['path'] for s in ('best', 'last')})
    require(reference == saved_reference, 'GW1 reuse reference changed')
    states = new_states(args.run_dir, identity, epochs)
    require(read_json(args.run_dir/'budget.json')['target_completed_epochs'] == epochs, 'GW2 target budget differs')
    modules = base.get_original_modules()
    require(modules.source_fingerprint == identity['source_fingerprint'], 'Frozen original identity differs')
    device = args.device or identity['runtime']['python_device']
    runtime = base.configure_runtime(device, identity['runtime']['threads'], identity['training_config']['seed'])
    data = base.load_data(args.data_dir, identity['training_config'], modules,
        base.logging.getLogger('gw2-reuse-eval'), check=identity['engineering_only'], training=True)
    require(data.identity == identity['data'], 'Actual evaluation data differs')
    archive, replay = cached_predictions(reference, args.checkpoint, identity, old_states, modules)
    require(np.array_equal(archive['node_ids'], data.station_ids), 'Cached station axis differs')
    roads = np.full(len(data.station_ids), 'unknown', dtype='<U40')
    for road, indices in data.roads.items():
        roads[indices] = road
    require(np.array_equal(archive['road_labels'], roads), 'Cached road labels differ')
    source_metadata = read_json(reference['archives'][args.checkpoint]['metadata_path'])
    costs = {arm: dict(source_metadata['benchmarks'][arm], measurement_origin='GW1_cached_prior_session',
        measurement_runtime=source_metadata['runtime']) for arm in REUSED_ARMS}
    roles = {arm: dict(source_metadata['checkpoint_roles'][arm], reused=True) for arm in REUSED_ARMS}
    mechanisms = {arm: dict(reused=True, new_inference_calls=0,
        source_mechanisms_path=str(Path(reference['archives'][args.checkpoint]['path']).parent/'mechanisms.json'))
        for arm in REUSED_ARMS}
    calls = {arm: 0 for arm in REUSED_ARMS}
    loader = data.loaders['val_loader']
    for arm, state in states.items():
        cached = existing_new_cache(args.run_dir, identity, state, arm, args.checkpoint, archive)
        if cached:
            prior, prior_metadata, prior_dir = cached
            conditions = [arm, arm+'_UNIFORM'] + (['POST_MESSAGE_STATIC_ONLY'] if arm == 'POST_MESSAGE' else [])
            for case in conditions:
                archive['pred__'+case] = prior['pred__'+case]
                mechanisms[case] = dict(reused=True, new_inference_calls=0, source_mechanisms_path=str(prior_dir/'mechanisms.json'))
            roles[arm] = dict(prior_metadata['checkpoint_roles'][arm], reused_predictions=True)
            costs[arm] = dict(prior_metadata['benchmarks'][arm], measurement_origin='GW2_cached_prior_session')
            epoch = state['best_epoch'] if args.checkpoint == 'best' else state['completed_epoch']
            replay[arm] = dict(base.replay_metrics(prior['pred__'+arm], prior['y'], state, epoch, modules),
                mode='cached_predictions_metric_check_not_new_checkpoint_forward', new_inference_calls=0)
            require(replay[arm]['passed'], 'Existing new-arm cache metrics differ')
            calls[arm] = 0
            continue
        model = train.model_for_arm(data, modules, device, arm)
        if args.checkpoint == 'best':
            path = args.run_dir/arm/state['best_checkpoint_file']
            require(base.sha256(path) == state['best_checkpoint_sha256'], 'GW2 best checkpoint changed')
            weights, epoch = train.load_checkpoint(path), state['best_epoch']
        else:
            path, weights, epoch = args.run_dir/arm/'last.pt', state['model_state'], state['completed_epoch']
        model.load_state_dict(weights, strict=True); model.eval()
        roles[arm] = dict(selection=args.checkpoint, epoch=epoch, checkpoint_sha256=base.sha256(path), reused=False)
        costs[arm] = dict(base.benchmark(model, base.device_batch(next(loader.get_iterator()), device), device),
            measurement_origin='GW2_current_session', measurement_runtime=runtime)
        calls[arm] = 5  # two warmups and three measured complete forwards
        conditions = [(arm, None), (arm+'_UNIFORM', 'UNIFORM')]
        if arm == 'POST_MESSAGE':
            conditions.append(('POST_MESSAGE_STATIC_ONLY', 'STATIC'))
        for case, setting in conditions:
            predictions, truths, supports = [], [], []
            model.start_diagnostics()
            context = model.intervention(setting) if setting else nullcontext()
            with torch.no_grad(), context:
                for index, raw in enumerate(loader.get_iterator(), 1):
                    x, y, incident, sensors = base.device_batch(raw, device)
                    p = model(x, incident_data=incident, sensor_data=sensors)
                    calls[arm] += 1
                    predictions.append(data.scaler.inverse_transform(p).squeeze(-1).cpu().numpy())
                    truths.append(data.scaler.inverse_transform(y).squeeze(-1).cpu().numpy())
                    supports.append(incident['distances'].abs().sum(-1).cpu().numpy() > 0)
                    if index == 1 or index % 10 == 0 or index == loader.num_batch:
                        print(dict(case=case, batch=index, total=loader.num_batch), flush=True)
            prediction, truth, support = map(np.concatenate, (predictions, truths, supports))
            require(np.array_equal(truth, archive['y']) and np.array_equal(support, archive['direct_support']),
                    'Cached baseline and new model targets/support/order differ')
            null = float(truth.min()) if truth.min() < 1 else 0.
            require(np.array_equal(archive['valid'], truth != null), 'Cached mask differs')
            archive['pred__'+case] = prediction
            mechanisms[case] = model.diagnostics()
            if setting is None:
                replay[arm] = dict(base.replay_metrics(prediction, truth, state, epoch, modules), mode='new_checkpoint_forward')
                require(replay[arm]['passed'], 'GW2 native validation replay failed: '+arm)
        del model, weights
        gc.collect()
        if torch.device(device).type == 'cuda':
            torch.cuda.empty_cache()
    base.validate_archive(archive, CASES, SCHEMA)
    metadata = dict(schema='original_igstgnn_graph_message_reuse_producer_v1', status='GW2_REUSE_PREDICTION_PASS',
        engineering_only=identity['engineering_only'], reuse_reference=reference, common_completed_epochs=epochs,
        run_identity_sha256=identity['identity_sha256'], runtime=runtime, checkpoint_roles=roles,
        validation_replay=replay, benchmarks=costs, new_inference_calls=calls, optimizer_updates=0, test_evaluated=False,
        omitted_cases={'STATIC_UNIFORM': 'not_in_source_archive_and_not_repeated'},
        producer_source={name: base.sha256(REPO/'experiments/chronological'/name) for name in (
            'evaluate_original_graph_message_reuse.py', 'graph_message_reuse.py', 'evaluate_original_graph_message.py')})
    path = output/'paired_predictions.npz'
    np.savez_compressed(path, **archive)
    metadata['archive_sha256'] = base.sha256(path)
    base.write_json(output/'metadata.json', metadata)
    base.write_json(output/'mechanisms.json', mechanisms)
    return archive, metadata


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--run-dir', type=Path)
    source.add_argument('--archive', type=Path)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--data-dir', type=Path, default=base.default_data_dir())
    p.add_argument('--checkpoint', choices=('best', 'last'), default='best')
    p.add_argument('--device')
    args = p.parse_args(argv)
    require(not args.output_dir.exists(), 'Use a fresh evaluation directory')
    args.output_dir.mkdir(parents=True)
    try:
        if args.archive:
            before = base.sha256(args.archive)
            with np.load(args.archive, allow_pickle=False) as saved:
                archive = dict(saved)
            metadata = dict(engineering_only=True, source_archive_sha256=before,
                producer_identity_independently_verified=False, new_inference_calls=0, test_evaluated=False)
        else:
            archive, metadata = produce(args, args.output_dir)
        result = base.report(archive, args.output_dir, metadata, CASES, COMPARISONS, SCHEMA)
        if args.archive:
            require(before == base.sha256(args.archive), 'Source archive changed')
        print(dict(status=result['status'], validation_rows=result['validation_rows'],
            engineering_only=result['engineering_only'], output_dir=str(args.output_dir.resolve())), flush=True)
        return result
    except BaseException as error:
        base.write_json(args.output_dir/'failure.json', dict(status='GW2_REUSE_EVALUATION_FAILED',
            error_type=type(error).__name__, error=str(error), optimizer_updates=0, test_evaluated=False))
        raise


if __name__ == '__main__':
    main()
