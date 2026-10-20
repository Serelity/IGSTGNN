"""Paired original/GW16 evaluation on certified multi-dataset release slices.

Only validation chooses BEST. Test identities are committed before test loaders
or forwards, and each completed model cache is immutable and independently reused.
Numerical execution belongs on the research server, never the development PC.
"""
import argparse
from contextlib import ExitStack
import csv
import gc
import logging
import os
from pathlib import Path
import time
import uuid

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch

from experiments.chronological.original_igstgnn_source import get_original_modules
from experiments.chronological.original_igstgnn_runtime import (
    device_batch as original_batch, rng_fingerprint)
from experiments.original_baseline_gw16.evaluate import native_summary
from experiments.position_state_gw16.objective import numpy_metrics
from .baseline import (load_original_data, build_original_model, original_loader,
                       selected_checkpoint as original_checkpoint)
from .common import (SCHEMA, BASELINE_SCHEMA, digest_json, exclusive_run, load_state,
    read_json, require, runtime, sha256, source_hashes, spec, synchronize,
    tensor_tree_hash, write_json)
from .data import device_batch as candidate_batch, load_data, validation_loader
from .model import build_model as build_candidate
from .train import validate_identity, validate_runtime, validate_artifacts, validate_saved

CASES = ('ORIGINAL_IGSTGNN', 'STATE_EXCHANGE')
ARCHIVE_SCHEMA = 'state_exchange_multidata_paired_archive_v1'
LOCK_SCHEMA = 'state_exchange_multidata_evaluation_lock_v1'
FIELDS = {'prediction', 'target', 'sample_ids', 'station_ids'}
METRIC_CONTRACT = dict(
    primary='unchanged_author_native_metrics_each_lead_then_equal_average_over_12',
    native_null='min_of_entire_selected_target_if_less_than_1_else_0; shared_by_both_models',
    native_accumulation='frozen_author_pytorch_float32',
    rmse_macro='mean_of_12_per_lead_RMSE_not_sqrt_of_pooled_MSE',
    mape_units='percent', horizons_reported=[3, 6, 12],
    common_mask=dict(target_minimum=.001, mape_minimum=1., accumulation='numpy_float64_pooled'),
    prediction_clipping=False, fitted_calibration=False, sample_reweighting=False)


def selected_checkpoint(run, identity, data):
    """Load only the completed candidate's validation-selected immutable BEST."""
    directory = Path(run)/'STATE_EXCHANGE'
    path = directory/'last.pt'
    last_sha = sha256(path)
    last = load_state(path)
    require(sha256(path) == last_sha, 'Candidate LAST changed during load')
    batches = data.loaders['train_loader'].num_batch
    validate_saved(last, identity, 'STATE_EXCHANGE', directory, batches)
    require(last['completed_epoch'] == identity['config']['max_epochs'],
            'Paired evaluation requires completed candidate training')
    path = directory/last['best_checkpoint_file']
    require(sha256(path) == last['best_checkpoint_sha256'], 'Candidate BEST changed')
    best = load_state(path)
    require(sha256(path) == last['best_checkpoint_sha256']
            and best['identity_sha256'] == identity['identity_sha256']
            and best['arm'] == 'STATE_EXCHANGE' and best['epoch'] == last['best_epoch']
            and best['validation'] == last['best_metrics'] and best['stage'] == 'joint',
            'Candidate validation BEST role differs')
    role = dict(case='STATE_EXCHANGE', selection='best', epoch=best['epoch'],
        completed_training_epochs=last['completed_epoch'], full_protocol_reached=True,
        global_updates=last['global_updates'], training_batches_per_epoch=batches,
        registered_parameters=last['registered_parameters'],
        checkpoint_sha256=last['best_checkpoint_sha256'], last_checkpoint_sha256=last_sha,
        model_tensor_sha256=tensor_tree_hash(best['model_state']),
        best_includes_stage_zero=True)
    return best['model_state'], role, best['validation'], path


def selection_guard(dataset, split, candidate, baseline, candidate_role, original_role, engineering):
    validate_identity(candidate)
    require(dataset in ('Alameda', 'Orange'),
            'Existing Contra_Costa frozen results are reused; this entry trains/evaluates only new datasets')
    require(split in ('val', 'test'), 'Evaluation split must be val or test')
    require(type(engineering) is bool and not (engineering and split == 'test'),
            'Engineering test evaluation is forbidden')
    require(candidate['schema'] == SCHEMA and baseline['schema'] == BASELINE_SCHEMA
            and candidate['engineering_only'] is engineering
            and baseline['engineering_only'] is engineering,
            'Run schemas or engineering scope differ')
    require(candidate['data']['dataset'] == baseline['data']['dataset'] == dataset
            and candidate['data']['hashes'] == baseline['data']['hashes'] == spec(dataset)['hashes'],
            'The two runs must use the same certified dataset release')
    require(candidate['seed'] == baseline['seed'], 'Paired runs use different seeds')
    expected = 3 if engineering else 100
    require(candidate['config']['max_epochs'] == expected
            and candidate_role['completed_training_epochs'] == expected
            and candidate_role['selection'] == original_role['selection'] == 'best'
            and candidate_role['full_protocol_reached'] is True
            and original_role['training_complete'] is True,
            'Complete training and validation-selected BEST are required before evaluation')
    require(candidate['execution_source'] == baseline['execution_source'] == source_hashes(),
            'Training and evaluation source fingerprints differ')


def protocol(dataset, split, engineering, station_ids):
    release = spec(dataset)
    start = release['train_samples'] + (release['val_samples'] if split == 'test' else 0)
    count = 4 if engineering else release[split+'_samples']
    return dict(dataset=dataset, split=split, engineering_only=engineering,
        count=count, nodes=release['nodes'], bounds_half_open=[start, start+count],
        station_ids=[int(v) for v in station_ids], release_hashes=release['hashes'])


def validate_arrays(arrays, material):
    require(isinstance(arrays, dict) and set(arrays) == FIELDS, 'Prediction archive fields differ')
    shape = (material['count'], 12, material['nodes'], 1)
    for key in ('prediction', 'target'):
        value = arrays[key]
        require(isinstance(value, np.ndarray) and value.dtype == np.float32
                and value.shape == shape and np.isfinite(value).all()
                and value.flags.c_contiguous, 'Prediction shape/dtype/layout differs: '+key)
    samples, stations = arrays['sample_ids'], arrays['station_ids']
    require(samples.dtype == np.int64 and np.array_equal(samples,
            np.arange(*material['bounds_half_open'], dtype=np.int64)), 'Published sample axis differs')
    require(stations.dtype == np.int64 and stations.shape == (material['nodes'],)
            and len(np.unique(stations)) == material['nodes']
            and np.array_equal(stations, np.asarray(material['station_ids'], dtype=np.int64)),
            'Published station axis differs')


def canonical_arrays(arrays):
    """Choose the serialized C layout BEFORE any FP32 native reduction."""
    return {key:np.array(value, dtype=np.float32 if key in ('prediction', 'target')
                         else np.int64, order='C', copy=True) for key, value in arrays.items()}


def scores(arrays, modules=None):
    return dict(author_metrics=native_summary(arrays['prediction'], arrays['target'], modules),
        common_mask_pooled=numpy_metrics(arrays['prediction'], arrays['target'], METRIC_CONTRACT['common_mask']))


def paired_scores(original, candidate, material, modules=None):
    for value in (original, candidate):
        validate_arrays(value, material)
    for key in ('target', 'sample_ids', 'station_ids'):
        require(original[key].dtype == candidate[key].dtype
                and np.array_equal(original[key], candidate[key]), 'Paired target/axis differs: '+key)
    rows = {case:scores(arrays, modules) for case, arrays in zip(CASES, (original, candidate))}
    base, actual = (rows[case]['author_metrics']['horizon_macro'] for case in CASES)
    require(all(base[key] > 0 for key in base), 'Original native metrics must be positive')
    return dict(rows=rows, relative_gain_percent_vs_replayed_original={
        key:100*(base[key]-actual[key])/base[key] for key in base})


def safe_output(output, inputs, data_dir):
    output = Path(output).resolve()
    for source in (*inputs, data_dir):
        source = Path(source).resolve()
        require(output != source and source not in output.parents and output not in source.parents,
                'Evaluation output must be separate from training and raw data')
    for parent in (output, *output.parents):
        require(not (parent/'identity.json').is_file(), 'Evaluation output cannot be nested in another training run')
    return output


def freeze_output(directory, payload):
    """Durably commit identities before any test loader is constructed."""
    directory = Path(directory)
    name = 'test_lock.json' if payload['protocol']['split'] == 'test' else 'validation_lock.json'
    record = dict(schema=LOCK_SCHEMA, freeze_id=digest_json(payload), payload=payload)
    path = directory/name
    if path.is_file():
        require(read_json(path) == record, 'Frozen evaluation identity changed; do not reselect from results')
    else:
        require(not any(p.name != 'active.lock' for p in directory.iterdir()),
                'Evaluation directory has residual files without a valid lock; preserve it')
        write_json(path, record)
    return record


def verify_frozen(frozen):
    payload = frozen['payload']
    require(frozen['schema'] == LOCK_SCHEMA and frozen['freeze_id'] == digest_json(payload),
            'Evaluation lock checksum differs')
    require(payload['execution_source'] == source_hashes(), 'Frozen execution source changed')
    material = payload['protocol']
    require(material['release_hashes'] == spec(material['dataset'])['hashes'], 'Frozen dataset release changed')
    for name, expected in payload['source_files'].items():
        path = Path(name)
        require(path.is_absolute() and path.is_file() and sha256(path) == expected,
                'Frozen training/data artifact changed: '+name)


def read_result(directory):
    """Authenticate a complete comparison without loading or forwarding a model."""
    directory = Path(directory)
    locks = [directory/name for name in ('test_lock.json', 'validation_lock.json') if (directory/name).is_file()]
    require(len(locks) == 1, 'Expected exactly one evaluation split lock')
    frozen = read_json(locks[0])
    verify_frozen(frozen)
    original, _ = read_cache(directory/CASES[0], frozen, CASES[0])
    candidate, _ = read_cache(directory/CASES[1], frozen, CASES[1])
    result = read_json(directory/'comparison.json')
    require(result['schema'] == 'state_exchange_multidata_comparison_v1'
            and result['status'] == 'PAIRED_EVALUATION_COMPLETE'
            and result['freeze_id'] == frozen['freeze_id']
            and result['protocol'] == frozen['payload']['protocol']
            and result['checkpoint_roles'] == frozen['payload']['checkpoint_roles']
            and result['metric_contract'] == METRIC_CONTRACT
            and result['runtime'] == frozen['payload']['runtime'], 'Comparison identity differs')
    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(result['runtime']['threads'])
        observed = paired_scores(original, candidate, result['protocol'])
    finally:
        torch.set_num_threads(previous)
    require(all(result[key] == value for key, value in observed.items()), 'Comparison metrics differ from immutable caches')
    require(result['dataset'] == result['protocol']['dataset']
            and result['split'] == result['protocol']['split']
            and result['samples'] == result['protocol']['count']
            and result['engineering_only'] is result['protocol']['engineering_only']
            and result['test_evaluated'] is (result['split'] == 'test')
            and result['candidate_checkpoint_role'] == result['checkpoint_roles'][CASES[1]]
            and result['original_checkpoint_role'] == result['checkpoint_roles'][CASES[0]]
            and result['parameter_counts'] == {case:result['checkpoint_roles'][case]['registered_parameters']
                                               for case in CASES},
            'Comparison scope or checkpoint aliases differ')
    return result


def read_cache(directory, frozen, case, modules=None):
    directory = Path(directory)
    metadata_path, path = directory/'metadata.json', directory/'predictions.npz'
    require(metadata_path.is_file() and path.is_file(), 'Incomplete cache must be preserved, never overwritten')
    before = sha256(metadata_path)
    metadata = read_json(metadata_path)
    require(metadata['schema'] == ARCHIVE_SCHEMA and metadata['case'] == case
            and metadata['freeze_id'] == frozen['freeze_id']
            and metadata['metric_contract'] == METRIC_CONTRACT
            and metadata['runtime'] == frozen['payload']['runtime'], 'Cache identity or metric contract differs')
    checksum = sha256(path)
    require(checksum == metadata['archive_sha256'], 'Prediction archive checksum differs')
    with np.load(path, allow_pickle=False) as saved:
        arrays = {key:saved[key] for key in saved.files}
    require(sha256(path) == checksum and sha256(metadata_path) == before, 'Cache changed during read')
    validate_arrays(arrays, frozen['payload']['protocol'])
    audit = metadata['audit']
    require(audit['model_tensor_sha256'] == frozen['payload']['checkpoint_roles'][case]['model_tensor_sha256']
            and all(audit[key] is True for key in ('rng_unchanged', 'registered_tensors_unchanged', 'raw_inputs_unchanged'))
            and audit['future_target_passed_to_forward'] is False and audit['optimizer_updates'] == 0,
            'Cached inference audit differs from frozen model')
    if frozen['payload']['protocol']['split'] == 'val':
        require(metadata['validation_replay']['passed'] is True, 'Validation cache lacks passed training replay')
    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(metadata['runtime']['threads'])
        observed = scores(arrays, modules)
    finally:
        torch.set_num_threads(previous)
    require(observed == metadata['scores'], 'Cached native/common metrics do not reproduce exactly')
    return arrays, metadata


def write_cache(directory, frozen, case, arrays, metadata, modules=None):
    directory = Path(directory)
    require(not directory.exists(), 'Completed cache cannot be overwritten')
    arrays = canonical_arrays(arrays)
    validate_arrays(arrays, frozen['payload']['protocol'])
    stage = directory.parent/('.'+directory.name+'.writing_'+uuid.uuid4().hex)
    stage.mkdir()
    np.savez_compressed(stage/'predictions.npz', **arrays)
    result = dict(metadata, schema=ARCHIVE_SCHEMA, case=case, freeze_id=frozen['freeze_id'],
        metric_contract=METRIC_CONTRACT, runtime=frozen['payload']['runtime'],
        archive_sha256=sha256(stage/'predictions.npz'), scores=scores(arrays, modules))
    write_json(stage/'metadata.json', result)
    read_cache(stage, frozen, case, modules)
    require(not directory.exists(), 'Cache appeared during write; preserve both directories')
    stage.rename(directory)
    return arrays, result


def collect(case, model, loader, data, device, material, logger):
    model.eval()
    before, random_before = tensor_tree_hash(model.state_dict()), rng_fingerprint()
    predictions, targets = [], []
    synchronize(device)
    started = time.perf_counter()
    with torch.no_grad():
        for index, raw in enumerate(loader.get_iterator(), 1):
            raw_before = tensor_tree_hash(raw)
            if case == 'ORIGINAL_IGSTGNN':
                x, y, incident, sensors = original_batch(raw, device)
                prediction = data.scaler.inverse_transform(model(
                    x, label=None, incident_data=incident, sensor_data=sensors))
                target = data.scaler.inverse_transform(y)
            else:
                x, target, incident = candidate_batch(raw, data, device)
                prediction = model(x, incident)['prediction']
            require(tensor_tree_hash(raw) == raw_before, 'Inference modified raw input')
            predictions.append(prediction.detach().cpu().numpy())
            targets.append(target.detach().cpu().numpy())
            if index in (1, loader.num_batch):
                logger.info('%s %s batch=%d/%d', case, material['split'], index, loader.num_batch)
    synchronize(device)
    require(tensor_tree_hash(model.state_dict()) == before and rng_fingerprint() == random_before,
            'Evaluation changed registered model tensors or consumed random streams')
    arrays = canonical_arrays(dict(prediction=np.concatenate(predictions), target=np.concatenate(targets),
        sample_ids=np.arange(*material['bounds_half_open'], dtype=np.int64),
        station_ids=np.asarray(data.station_ids, dtype=np.int64)))
    validate_arrays(arrays, material)
    return arrays, dict(model_tensor_sha256=before, rng_unchanged=True, registered_tensors_unchanged=True,
        raw_inputs_unchanged=True, future_target_passed_to_forward=False, optimizer_updates=0,
        forward_batches=loader.num_batch, full_collection_seconds=time.perf_counter()-started,
        timing_scope='cold_loader_H2D_forward_D2H_and_input_hashes; not_a_speed_benchmark')


def validation_replay(case, row, expected):
    actual = (row['author_metrics']['pooled']['mae'] if case == CASES[0]
              else row['common_mask_pooled']['mae'])
    wanted = float(expected if case == CASES[0] else expected['mae'])
    difference = abs(actual-wanted)
    require(difference <= 1e-5, 'Validation BEST replay differs from training: '+case)
    return dict(passed=True, expected_mae=wanted, observed_mae=actual,
                absolute_difference=difference, tolerance=1e-5)


def render_report(directory, result):
    material = result['protocol']
    lines = ['# Original IGSTGNN / GW16 STATE_EXCHANGE: '+material['dataset']+' / '+material['split'], '',
        '{} published samples; target, station and sample axes are exactly paired.'.format(material['count']),
        'Both models use their own validation-selected BEST, frozen after their registered training protocol.',
        'Primary: per-lead author FP32 metrics, then an equal mean over 12 leads. RMSE is not pooled RMSE.', '',
        '| Method | Parameters | 12-lead MAE | 12-lead RMSE | 12-lead MAPE % |',
        '|---|---:|---:|---:|---:|']
    for case in CASES:
        row = result['rows'][case]['author_metrics']['horizon_macro']
        lines.append('| {} | {} | {:.9f} | {:.9f} | {:.9f} |'.format(case,
            result['checkpoint_roles'][case]['registered_parameters'], row['mae'], row['rmse'], row['mape_percent']))
    gain = result['relative_gain_percent_vs_replayed_original']
    lines += ['', 'Relative gains vs replayed original: MAE {:.4f}%, RMSE {:.4f}%, MAPE {:.4f}%.'.format(
        gain['mae'], gain['rmse'], gain['mape_percent']), '',
        'Per-lead results: per_horizon.csv. common_mask_pooled is supplementary in comparison.json.',
        'The original reuses the project Contra reproduction recipe (batch 48), native curriculum and patience-20 stopping;',
        'STATE_EXCHANGE uses its fixed GW16 recipe, 100 epochs, pooled validation MAE and stage-zero BEST.',
        'Inputs, training recipe and architecture differ; this is a single-seed whole-method comparison.',
        'Published window overlap and incident first availability remain unverified.',
        'Completed caches are immutable; rerunning reuses them with zero new completed-model forwards.']
    if material['engineering_only']:
        lines += ['', 'ENGINEERING ONLY: four windows and three epochs; no research performance conclusion.']
    (Path(directory)/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    with (Path(directory)/'per_horizon.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['case', 'lead', 'mae', 'rmse', 'mape_percent'])
        writer.writeheader()
        for case in CASES:
            for row in result['rows'][case]['author_metrics']['per_horizon']:
                writer.writerow(dict(case=case, **row))
    write_json(Path(directory)/'comparison.json', result)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['Alameda', 'Orange'], required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--candidate-run', type=Path, required=True)
    parser.add_argument('--baseline-run', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--threads', type=int, default=3)
    parser.add_argument('--engineering', action='store_true')
    args = parser.parse_args(argv)
    if args.engineering and args.split == 'test':
        parser.error('Engineering evaluation is validation-only')
    return args


def main(argv=None):
    args = parse_args(argv)
    args.output_dir = safe_output(args.output_dir, (args.candidate_run, args.baseline_run), args.data_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger('state_exchange_multidata.evaluate')
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    with ExitStack() as locks:
        for directory in sorted({args.candidate_run.resolve(), args.baseline_run.resolve(), args.output_dir}):
            require(directory.is_dir(), 'Required run directory is missing: '+str(directory))
            locks.enter_context(exclusive_run(directory))
        identity = read_json(args.candidate_run/'identity.json')
        baseline_identity = read_json(args.baseline_run/'identity.json')
        validate_identity(identity)
        environment = runtime(args.device, args.threads, identity['seed'])
        validate_runtime(read_json(args.candidate_run/'runtime.json'), environment)
        baseline_runtime = read_json(args.baseline_run/'runtime.json')
        require(all(key in environment and environment[key] == value for key, value in baseline_runtime.items()),
                'Baseline numerical runtime, seed, device or threads changed')
        data = load_data(args.data_dir, identity['config'], logger, engineering=args.engineering)
        require(data.identity == identity['data'], 'Candidate training data identity changed')
        validate_artifacts(args.candidate_run, identity, data)
        modules = get_original_modules()
        original = load_original_data(args.data_dir, args.dataset, baseline_identity['config'],
                                      modules, logger, check=args.engineering, training=False)
        candidate_weights, candidate_role, candidate_expected, candidate_path = selected_checkpoint(
            args.candidate_run, identity, data)
        baseline_weights, baseline_role, baseline_expected, baseline_path = original_checkpoint(
            args.baseline_run, original)
        baseline_role = dict(baseline_role, model_tensor_sha256=baseline_role['model_state_sha256'])
        selection_guard(args.dataset, args.split, identity, baseline_identity,
                        candidate_role, baseline_role, args.engineering)
        require(np.array_equal(data.station_ids, original.station_ids), 'Training station axes differ')
        material = protocol(args.dataset, args.split, args.engineering, data.station_ids)
        roles = dict(zip(CASES, (baseline_role, candidate_role)))
        input_paths = [candidate_path, baseline_path, args.candidate_run/'identity.json',
            args.candidate_run/'runtime.json', args.candidate_run/'STATE_EXCHANGE'/'last.pt',
            args.baseline_run/'identity.json', args.baseline_run/'runtime.json', args.baseline_run/'last.pt',
            *(args.data_dir/name for name in material['release_hashes'])]
        source_files = {str(Path(path).resolve()):sha256(path) for path in input_paths}
        frozen = freeze_output(args.output_dir, dict(protocol=material, checkpoint_roles=roles,
            candidate_identity_sha256=identity['identity_sha256'],
            baseline_identity_sha256=baseline_identity['identity_sha256'],
            execution_source=source_hashes(), source_files=source_files,
            runtime=environment, metric_contract=METRIC_CONTRACT,
            future_target_passed_to_forward=False, optimizer_updates=0,
            selection='validation_BEST_only_before_test', independent_confirmation=False))
        verify_frozen(frozen)
        # No test loader has been requested before the immutable test_lock.json.
        records, arrays = {}, {}
        for case, weights, role, expected, checkpoint, active_data in (
            (CASES[0], baseline_weights, baseline_role, baseline_expected, baseline_path, original),
            (CASES[1], candidate_weights, candidate_role, candidate_expected, candidate_path, data)):
            cache = args.output_dir/case
            if cache.exists():
                arrays[case], records[case] = read_cache(cache, frozen, case, modules)
                logger.info('%s immutable cache reused; zero new forwards', case)
            else:
                if case == CASES[0]:
                    model = build_original_model(active_data, modules, args.device)
                    loader = original_loader(active_data, modules, args.split,
                        2 if args.engineering else baseline_identity['config']['bs'], logger)
                    offset = material['bounds_half_open'][0]
                else:
                    model = build_candidate(active_data, identity['config'], 'STATE_EXCHANGE', args.device)
                    model.set_stage('joint')
                    loader, offset = validation_loader(active_data, args.split, identity['config']['batch_size'], logger)
                require(offset == material['bounds_half_open'][0], 'Loader sample offset differs')
                model.load_state_dict(weights, strict=True)
                require(tensor_tree_hash(model.state_dict()) == role['model_tensor_sha256']
                        and sum(p.numel() for p in model.parameters()) == role['registered_parameters'],
                        'Built model differs from the frozen checkpoint')
                observed, audit = collect(case, model, loader, active_data, args.device, material, logger)
                row = scores(observed, modules)
                replay = validation_replay(case, row, expected) if args.split == 'val' else dict(
                    applicable=False, reason='test_is_not_used_for_training_replay_or_selection')
                arrays[case], records[case] = write_cache(cache, frozen, case, observed,
                    dict(audit=audit, validation_replay=replay), modules)
                del model
                gc.collect()
            require(sha256(checkpoint) == role['checkpoint_sha256'], 'Selected checkpoint changed during evaluation')
            require(source_hashes() == frozen['payload']['execution_source'], 'Execution source changed during evaluation')
        result = dict(schema='state_exchange_multidata_comparison_v1', status='PAIRED_EVALUATION_COMPLETE',
            freeze_id=frozen['freeze_id'], protocol=material, checkpoint_roles=roles,
            dataset=args.dataset, split=args.split, samples=material['count'],
            engineering_only=args.engineering,
            candidate_checkpoint_role=candidate_role, original_checkpoint_role=baseline_role,
            parameter_counts={case:roles[case]['registered_parameters'] for case in CASES},
            metric_contract=METRIC_CONTRACT, runtime=environment,
            test_evaluated=args.split == 'test', independent_confirmation=False,
            caches={case:dict(archive_sha256=records[case]['archive_sha256']) for case in CASES},
            **paired_scores(arrays[CASES[0]], arrays[CASES[1]], material, modules))
        render_report(args.output_dir, result)
        verify_frozen(frozen)
        logger.info('PAIRED_EVALUATION_COMPLETE: %s', args.output_dir/'report.md')
        return result


if __name__ == '__main__':
    main()
