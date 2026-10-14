"""Replay one frozen location adapter, then shuffle relation vectors in support.

This is one fixed-seed checkpoint sensitivity experiment. The original support
mask remains intact; neither the value of that mask nor real incident causality
is identified. No optimizer or training loop is constructed.
"""
import argparse
from contextlib import ExitStack
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import sys
import uuid

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.chronological import analyze_min_frozen_gain as offline
from experiments.chronological import train_min_frozen_budget as frozen
from experiments.chronological import probe_min_event_budget as probe
from experiments.chronological.min_frozen_location_shuffle import make_location_shuffle, assert_shuffle_preserved
from experiments.chronological.min_frozen_location_probe_metrics import summarize_location_probe, write_report
from experiments.chronological.smoke import set_seed
from src.models.incident_spatial_frozen_budget import build_frozen_location, assert_frozen
from src.models.incident_spatial_residual import tensor_state_hash
from src.utils.igstgnn_published_events import EventPublishedData, EVENT_KEYS, encode_event

SCHEMA = 'frozen_MIN_location_content_probe_v1'
HERE = Path(__file__).resolve().parent
SCOPE = 'conditional_offline_same_checkpoint_supported_station_relation_detail_sensitivity'


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True, help='Existing frozen-MIN location pilot')
    parser.add_argument('--source-run', type=Path, help='Original A run; default is the recorded original location')
    parser.add_argument('--data-dir', type=Path, default=offline.default_data())
    parser.add_argument('--graph-dir', type=Path, default=HERE/'internal_spatial_graph_is11')
    parser.add_argument('--source-protocol', type=Path, default=frozen.original.PROTOCOL)
    parser.add_argument('--output-dir', type=Path, help='New directory only; default is a timestamped run subdirectory')
    parser.add_argument('--shuffle-seed', type=int, default=20261014, help='One prespecified shuffle, never selected by outcomes')
    parser.add_argument('--device', help='Default: source device type; original software/precision policy required')
    parser.add_argument('--check', action='store_true', help='Only accepts an already-existing engineering source')
    args = parser.parse_args(argv)
    offline.require(0 <= args.shuffle_seed < 2**32, 'Expected a 32-bit nonnegative shuffle seed')
    for name in ('run_dir', 'data_dir', 'graph_dir', 'source_protocol'):
        setattr(args, name, getattr(args, name).resolve())
    if args.source_run:
        args.source_run = args.source_run.resolve()
    args.output_dir = (args.output_dir.resolve() if args.output_dir else args.run_dir/(
        'location_content_probe_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6]))
    offline.require(not args.output_dir.exists() and args.output_dir != args.run_dir
                    and args.output_dir not in args.run_dir.parents, 'Use a new independent probe output directory')
    for directory in (args.data_dir, args.graph_dir):
        offline.require(args.output_dir != directory and directory not in args.output_dir.parents
                        and args.output_dir not in directory.parents, 'Probe output overlaps release or graph inputs')
    return args


def full_events(data):
    loader = data.loaders['val_loader']
    rows = [encode_event(sample, len(data.station_ids)) for sample in loader.original_samples]
    return {key: np.asarray([row[key] for row in rows],
            dtype=np.float32 if key == 'relation' else np.int64 if key in ('type', 'description') else np.bool_)
            for key in EVENT_KEYS}


def require_row(row, expected_prediction, target, support, label):
    offline.require(np.array_equal(row['prediction'], expected_prediction), label+' prediction differs from saved replay')
    offline.require(np.array_equal(row['target'], target), label+' target differs from saved validation')
    offline.require(np.array_equal(row['report_support'], support), label+' report support changed')


def run_probe(model, data, best, regions, output, *, seed=20261014, frozen_hash,
              expected_mechanisms=None, metadata=None):
    """Require the entire real/OFF replay before permitting any shuffled forward."""
    output = Path(output)
    offline.require(output.is_dir(), 'Create a new probe output directory before inference')
    loader = data.loaders['val_loader']
    size = loader.size
    offline.require(size == len(best['target']) and np.array_equal(loader.order, np.arange(size)),
                    'Probe requires the original unshuffled validation order')
    expected_indices = data.audit['counts']['train_samples']+np.arange(size, dtype=np.int64)
    offline.require(np.array_equal(best['sample_indices'], expected_indices), 'Saved validation sample IDs differ')
    offline.require(set(regions) == {'direct_report_nodes', 'potential_only', 'remaining_nodes'}, 'Expected the three original regions')
    offline.require(not any(module.training for module in model.modules()), 'Probe requires all modules in evaluation mode')
    assert_frozen(model, frozen_hash)
    graph = model.internal_spatial_graph
    graph_plain = (graph.nodes, graph.evidence_scope)
    graph_hash = tensor_state_hash(graph.state_dict())
    model_hash = tensor_state_hash(model.state_dict())
    rng_hash = probe.rng_digest(next(model.parameters()).device)
    input_events = full_events(data)
    support = (np.abs(input_events['relation']).sum(-1) > 0) & input_events['report_present'][:, None] & input_events['relation_available'][:, None]
    offline.require(np.array_equal(support, regions['direct_report_nodes']), 'Original direct-region mask differs from model support')
    calls = dict(real=0, MIN=0, shuffled=0)
    preservation, mechanisms = {}, {'real': {}, 'MIN': {}, 'shuffled': {}}
    replay = dict(status='REAL_AND_MIN_REPLAY_NOT_COMPLETE', exact_prediction_reference='saved_frozen_execution_float32_arrays',
                  original_batch_size=loader.bs, validation_rows=size, batch_partition=[], shuffled_forward_permitted=False)
    with probe.preserve_model(model, preservation, 'frozen_location'):
        position = 0
        print(f'Verifying saved best REAL and MIN predictions ({size} rows, original batch {loader.bs})', flush=True)
        for batch in loader.get_iterator():
            digest = probe.input_digest(batch)
            end = position+len(batch['x_data'])
            real = probe.forward_case(model, 'L', batch, data.scaler, 'L_ON')
            calls['real'] += 1
            require_row(real, best['on'][position:end], best['target'][position:end], support[position:end], 'Initial REAL')
            off = probe.forward_case(model, 'L', batch, data.scaler, 'L_OFF')
            calls['MIN'] += 1
            # forward_case audits the actual OFF context internally; its
            # returned report_support describes the unchanged input payload.
            require_row(off, best['off'][position:end], best['target'][position:end], support[position:end], 'Initial MIN')
            probe.merge_mechanisms(mechanisms['real'], real['mechanisms'])
            probe.merge_mechanisms(mechanisms['MIN'], off['mechanisms'])
            offline.require(probe.input_digest(batch) == digest, 'Initial replay modified an original batch')
            replay['batch_partition'].append(end-position)
            position = end
            print(f'Verified REAL/MIN: {position}/{size} rows', flush=True)
        offline.require(position == size, 'Initial replay missed validation rows')
        if expected_mechanisms is not None:
            offline.require(mechanisms['real'] == expected_mechanisms, 'Saved REAL mechanism replay differs')
        offline.require((graph.nodes, graph.evidence_scope) == graph_plain
                        and tensor_state_hash(graph.state_dict()) == graph_hash,
                        'Initial replay changed candidate graph')
        offline.require(tensor_state_hash(model.state_dict()) == model_hash
                        and probe.rng_digest(next(model.parameters()).device) == rng_hash,
                        'Initial replay changed model tensors or global RNG')
        assert_frozen(model, frozen_hash)
        replay.update(status='REAL_AND_MIN_EXACT_REPLAY_PASS', shuffled_forward_permitted=True,
                      saved_REAL_exact=True, saved_MIN_exact=True)
        offline.write_json(output/'replay.json', replay)
        # This local generator only assigns donors; it never reads any targets.
        shuffled_events, donors, shuffle_audit = make_location_shuffle(input_events, best['sample_indices'], seed=seed)
        assert_shuffle_preserved(input_events, shuffled_events, donors)
        print('Full replay passed; permitting one within-support relation-vector shuffle', flush=True)
        wrong, position, paired_partition = [], 0, []
        for batch in loader.get_iterator():
            end = position+len(batch['x_data'])
            digest = probe.input_digest(batch)
            real = probe.forward_case(model, 'L', batch, data.scaler, 'L_ON')
            calls['real'] += 1
            require_row(real, best['on'][position:end], best['target'][position:end], support[position:end], 'Paired REAL')
            changed_batch = dict(batch, event_data={key: value[position:end].copy() for key, value in shuffled_events.items()})
            # The intervention modifies only relation assignment, never traffic,
            # targets, event categories, flags, or the supported station set.
            for key in EVENT_KEYS:
                if key != 'relation':
                    offline.require(np.array_equal(changed_batch['event_data'][key], batch['event_data'][key]),
                                    'Shuffle changed a non-relation event field: '+key)
            changed = probe.forward_case(model, 'L', changed_batch, data.scaler, 'L_ON')
            calls['shuffled'] += 1
            offline.require(np.array_equal(changed['target'], best['target'][position:end])
                            and np.array_equal(changed['report_support'], support[position:end]), 'Shuffled target/support changed')
            restored = probe.forward_case(model, 'L', batch, data.scaler, 'L_ON')
            calls['real'] += 1
            require_row(restored, real['prediction'], best['target'][position:end], support[position:end], 'Restored REAL')
            offline.require(probe.input_digest(batch) == digest, 'Intervention modified the original batch')
            wrong.append(changed['prediction'])
            probe.merge_mechanisms(mechanisms['shuffled'], changed['mechanisms'])
            paired_partition.append(end-position)
            position = end
            print(f'Paired REAL/SHUFFLED/REAL: {position}/{size} rows', flush=True)
        offline.require(position == size and paired_partition == replay['batch_partition'], 'Paired validation partition changed')
        offline.require((graph.nodes, graph.evidence_scope) == graph_plain
                        and tensor_state_hash(graph.state_dict()) == graph_hash, 'Intervention changed candidate graph')
        assert_shuffle_preserved(input_events, shuffled_events, donors)
        assert_frozen(model, frozen_hash)
    changed_prediction = np.concatenate(wrong)
    analysis = summarize_location_probe(best['on'], changed_prediction, best['off'], best['target'], best['valid'],
                                        regions, best['fixed_hard_indices'])
    replay.update(paired_REAL_restored_exact=True, paired_batch_partition=paired_partition)
    result = dict(status='FROZEN_MIN_LOCATION_CONTENT_PROBE_PASS', schema=SCHEMA, scope=SCOPE,
                  analysis=analysis, replay=replay, shuffle_audit=shuffle_audit, mechanisms=mechanisms,
                  model_preservation=preservation, inference_calls=calls, new_inference_calls=sum(calls.values()),
                  training_updates=0, test_evaluated=False, shuffle_seed=seed,
                  support_location_effect_tested=False, unique_incident_causality_claim=False,
                  single_shuffle_realization=True, validation_rows=size)
    details = dict(metadata or {}, validation_rows=size, shuffle_seed=seed, shuffle_audit=shuffle_audit,
                   new_inference_calls=result['new_inference_calls'], training_updates=0, test_evaluated=False, scope=SCOPE)
    write_report(output, analysis, details)
    np.savez_compressed(output/'paired_predictions.npz', REAL=best['on'], SHUFFLED=changed_prediction, MIN=best['off'],
                        target=best['target'], valid=best['valid'], sample_indices=best['sample_indices'],
                        fixed_hard_indices=best['fixed_hard_indices'], station_ids=np.asarray(data.station_ids, np.int64))
    np.savez_compressed(output/'shuffle_mapping.npz', donor_station_indices=donors,
                        sample_indices=best['sample_indices'], station_ids=np.asarray(data.station_ids, np.int64))
    for name in ('analysis', 'replay', 'shuffle_audit', 'mechanisms', 'model_preservation'):
        offline.write_json(output/(name+'.json'), result[name])
    return result


def main(argv=None):
    args = parse_args(argv)
    identity = offline.read_json(args.run_dir/'identity.json')
    offline.require(identity.get('schema') == frozen.SCHEMA, 'Expected an existing frozen MIN location pilot')
    offline.require(not args.check or identity['engineering_check'], '--check cannot subset a formal validation run')
    args.source_run = args.source_run or Path(identity['source_run']).resolve()
    offline.require(args.output_dir != args.source_run and args.source_run not in args.output_dir.parents
                    and args.output_dir not in args.source_run.parents, 'Probe output overlaps original source A run')
    device = torch.device(args.device or ('cuda:0' if identity['runtime']['device_type'] == 'cuda' else 'cpu'))
    offline.require(device.type in ('cpu', 'cuda') and (device.type != 'cuda' or torch.cuda.is_available()), 'Source device unavailable')
    torch.set_num_threads(identity['threads'])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    probe.verify_runtime(identity['source_identity'], device)
    with ExitStack() as locks:
        for directory in sorted((args.run_dir, args.source_run, args.source_run/'A'), key=str):
            locks.enter_context(offline.readonly_run_lock(directory))
        source_identity = offline.read_json(args.source_run/'identity.json')
        offline.require(source_identity == identity['source_identity'], 'Original A identity changed')
        source_args = argparse.Namespace(source_run=args.source_run, data_dir=args.data_dir, graph_dir=args.graph_dir,
                                         source_protocol=args.source_protocol, recipe=frozen.RECIPE)
        frozen.verify_source(source_args, source_identity)
        source_paths = frozen.source_paths(source_args, source_identity)
        offline.require(probe.snapshot_files(source_paths) == identity['source_artifacts_sha256'], 'Recorded original A/frozen training inputs changed')
        offline.require(frozen.experiment_sources() == identity['experiment_sources_sha256']
                        and offline.sha256(frozen.RECIPE) == identity['recipe_sha256']
                        and offline.read_json(frozen.RECIPE) == identity['recipe'], 'Original frozen training recipe/code changed')
        paths = {'frozen/'+name: args.run_dir/name for name in offline.RUN_FILES}
        paths.update({'original/'+key: path for key, path in source_paths.items()})
        for name in ('probe_min_frozen_location.py', 'min_frozen_location_shuffle.py', 'min_frozen_location_probe_metrics.py',
                     'run_min_frozen_location_probe.sh', 'analyze_min_frozen_gain.py', 'min_frozen_gain_metrics.py'):
            paths['probe_code/'+name] = HERE/name
        before = probe.snapshot_files(paths)
        offline.require('<missing>' not in before.values(), 'A protected probe input is missing')
        best, baseline = offline.load_archives(args.run_dir)
        summary, history, baseline_info = (offline.read_json(args.run_dir/name) for name in ('summary.json', 'history.json', 'baseline.json'))
        offline.verify_certificates(args.run_dir, identity, summary, history, baseline_info, best, baseline)
        del baseline
        logger = logging.getLogger('frozen_location_probe')
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            logger.addHandler(logging.StreamHandler(sys.stdout))
        data = EventPublishedData(args.data_dir, source_identity['protocol'], logger, check=identity['engineering_check'])
        offline.require(data.audit == offline.read_json(args.source_run/'data_audit.json'), 'Original data audit changed')
        graph, roads, graph_info = frozen.legacy.load_graph(args.graph_dir, data.station_ids, source_identity['protocol']['graph_sha256'])
        offline.require(graph_info == source_identity['graph'], 'Original graph audit changed')
        source_saved = torch.load(args.source_run/'A/last.pt', map_location='cpu', weights_only=False)
        frozen.original.validate_saved(source_saved, source_identity, 'A')
        probe.verify_saved_sidecars(args.source_run/'A', source_saved)
        offline.require(probe.best_row(source_saved)['epoch'] == identity['source_best_epoch']
                        and tensor_state_hash(source_saved['best_model_state']) == identity['source_best_model_sha256'], 'Source best MIN changed')
        base, _ = frozen.original.build_arm(argparse.Namespace(data_dir=args.data_dir, seed=source_identity['seed']),
                                           'A', len(data.station_ids), graph, device)
        base.load_state_dict(source_saved['best_model_state'], strict=True)
        base.eval()
        set_seed(identity['seed'])
        model, certificate = build_frozen_location(base)
        offline.require(certificate == offline.read_json(args.run_dir/'initialization.json'), 'Frozen reconstruction initialization differs')
        saved = torch.load(args.run_dir/'last.pt', map_location='cpu', weights_only=False)
        frozen.validate_checkpoint(saved, identity, model, offline.sha256(args.run_dir/'baseline_predictions.npz'))
        offline.require(saved['history'] == history and saved['completed_epoch'] == summary['completed_epoch']
                        and saved['best_epoch'] == summary['best_adapter_epoch'], 'Frozen checkpoint and JSON companions disagree')
        model.event_budget_adapter.load_state_dict(saved['best_adapter_state'], strict=True)
        model.eval()
        assert_frozen(model, identity['frozen_state_sha256'])
        all_regions = probe.regions_for(full_events(data), graph, roads)
        regions = {name: all_regions[name] for name in offline.PARTITION}
        probe.require_preserved(before, paths)
        args.output_dir.mkdir(parents=True, exist_ok=False)
        print('Source frozen run: '+str(args.run_dir), flush=True)
        print('Probe directory: '+str(args.output_dir), flush=True)
        metadata = dict(source_best_epoch=identity['source_best_epoch'], best_adapter_epoch=saved['best_epoch'],
                        engineering_only=identity['engineering_check'], source_artifacts_preserved=False)
        offline.write_json(args.output_dir/'environment.json', frozen.legacy.environment(device))
        offline.write_json(args.output_dir/'request.json', dict(run_dir=str(args.run_dir), source_run=str(args.source_run),
            shuffle_seed=args.shuffle_seed, scope=SCOPE, batch_size=identity['batch_size'], selection='existing_best_adapter_no_reselection'))
        try:
            result = run_probe(model, data, best, regions, args.output_dir, seed=args.shuffle_seed,
                               frozen_hash=identity['frozen_state_sha256'],
                               expected_mechanisms=summary['best_validation']['mechanisms'], metadata=metadata)
            preserved = probe.require_preserved(before, paths)
            offline.write_json(args.output_dir/'preservation.json', preserved)
            metadata.update(source_artifacts_preserved=True, shuffle_seed=args.shuffle_seed,
                            shuffle_audit=result['shuffle_audit'], new_inference_calls=result['new_inference_calls'])
            write_report(args.output_dir, result['analysis'], metadata)
            compact = {key: value for key, value in result.items() if key not in ('analysis', 'mechanisms', 'model_preservation')}
            compact.update(source_best_epoch=identity['source_best_epoch'], best_adapter_epoch=saved['best_epoch'],
                           engineering_only=identity['engineering_check'], source_artifacts_preserved=True,
                           content_effect=result['analysis']['pairs']['real_vs_shuffle'],
                           real_vs_MIN=result['analysis']['pairs']['real_vs_MIN']['paired'],
                           shuffle_vs_MIN=result['analysis']['pairs']['shuffle_vs_MIN']['paired'], output_dir=str(args.output_dir))
            offline.write_json(args.output_dir/'provenance.json', dict(input_paths={key: str(path) for key, path in paths.items()},
                input_sha256=before, source_best_epoch=identity['source_best_epoch'], best_adapter_epoch=saved['best_epoch'],
                best_adapter_sha256=tensor_state_hash(saved['best_adapter_state']),
                fixed_bias_reference='MIN_prediction_minus_observed_target', scope=SCOPE,
                test_payload_deserialized=True, test_evaluated=False, seed_selected_by_outcomes=False))
            offline.write_json(args.output_dir/'summary.json', compact)
            # Do not print per-sample donor audits or millions of predictions.
            console = {key: compact[key] for key in ('status', 'source_best_epoch', 'best_adapter_epoch', 'validation_rows',
                'engineering_only', 'shuffle_seed', 'new_inference_calls', 'training_updates', 'source_artifacts_preserved', 'output_dir')}
            console.update(content_mae_gain=compact['content_effect']['mae_gain_pooled'],
                content_mape_gain=compact['content_effect']['paired']['mape_gain_pooled'],
                real_mae=compact['content_effect']['on']['mae_pooled'], shuffled_mae=compact['content_effect']['off']['mae_pooled'],
                real_vs_MIN_mae_gain=compact['real_vs_MIN']['mae_gain_pooled'],
                shuffle_vs_MIN_mae_gain=compact['shuffle_vs_MIN']['mae_gain_pooled'],
                shuffle_coverage={key: value for key, value in result['shuffle_audit'].items() if key != 'per_sample'})
            print(json.dumps(console, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
        except BaseException as error:
            offline.write_json(args.output_dir/'failure.json', dict(status='FROZEN_MIN_LOCATION_CONTENT_PROBE_FAILED',
                error=str(error), source_artifacts_preserved=probe.snapshot_files(paths) == before))
            raise


if __name__ == '__main__':
    main()
