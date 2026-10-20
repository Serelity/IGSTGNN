"""Evaluate only completed original IGSTGNN validation-selected BEST.

The original frozen model, loader and native metrics are reused directly.
Test requires an authenticated native validation replay and an immutable lock
committed before a test loader is constructed. No GW16 model is constructed.
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
    configure_runtime, device_batch as original_batch, rng_fingerprint)
from experiments.state_exchange_multidata.baseline import (
    _read_identity, selected_checkpoint, load_original_data, original_loader,
    build_original_model)
from experiments.state_exchange_multidata.common import (
    BASELINE_SCHEMA, digest_json, exclusive_run, read_json, require, sha256, spec,
    synchronize, tensor_tree_hash, write_json)
from experiments.state_exchange_multidata.evaluate import (
    protocol, canonical_arrays, validate_arrays, scores, safe_output,
    validation_replay, METRIC_CONTRACT as PREVIOUS_METRIC_CONTRACT)
from .common import source_hashes as new_source_hashes, training_source_hashes

CASES = ('ORIGINAL_IGSTGNN',)
ARCHIVE_SCHEMA = 'original_multidata_only_archive_v1'
LOCK_SCHEMA = 'original_multidata_only_evaluation_lock_v1'
RESULT_SCHEMA = 'original_multidata_only_result_v1'
STATUS = 'ORIGINAL_MULTIDATA_ONLY_EVALUATION_COMPLETE'
METRIC_CONTRACT = dict(PREVIOUS_METRIC_CONTRACT,
    native_null='min_of_entire_selected_target_if_less_than_1_else_0; original_only')


def selection_guard(dataset, split, identity, role, engineering):
    require(dataset in ('Alameda', 'Orange'), 'Only Alameda and Orange original runs are accepted')
    require(split in ('val', 'test'), 'Evaluation split must be val or test')
    require(type(engineering) is bool and not (engineering and split == 'test'),
            'Engineering test evaluation is forbidden')
    require(identity['schema'] == BASELINE_SCHEMA and identity['engineering_only'] is engineering,
            'Original schema or engineering scope differs')
    require(identity['data']['dataset'] == dataset
            and identity['data']['hashes'] == spec(dataset)['hashes'],
            'Original training must use the certified dataset release')
    require(identity['execution_source'] == training_source_hashes(),
            'Original training source fingerprint changed')
    require(role['case'] == CASES[0] and role['dataset'] == dataset
            and role['selection'] == 'best' and role['training_complete'] is True
            and role['engineering_only'] is engineering
            and role['training_budget_epochs'] == (3 if engineering else 100)
            and role['best_selection_includes_stage_zero'] is False,
            'Complete original protocol and validation-selected BEST are required')
    require(0 < role['selected_epoch'] == role['best_epoch'] <= role['completed_epoch']
            <= (3 if engineering else 100)
            and (role['completed_epoch'] == (3 if engineering else 100) or role['wait'] >= 20),
            'Original completion requires its full budget or native patience-20 stopping')
    require(role['model_tensor_sha256'] == role['model_state_sha256'], 'Original tensor aliases differ')


def freeze_output(directory, payload):
    directory = Path(directory)
    name = 'test_lock.json' if payload['protocol']['split'] == 'test' else 'validation_lock.json'
    record = dict(schema=LOCK_SCHEMA, freeze_id=digest_json(payload), payload=payload)
    path = directory/name
    if path.is_file():
        require(read_json(path) == record, 'Frozen evaluation identity changed; preserve completed results')
    else:
        require(not any(path.name != 'active.lock' for path in directory.iterdir()),
                'Evaluation directory has residual files without a valid lock; preserve it')
        write_json(path, record)
    return record


def verify_frozen(frozen):
    payload = frozen['payload']
    require(frozen['schema'] == LOCK_SCHEMA and frozen['freeze_id'] == digest_json(payload),
            'Evaluation lock checksum differs')
    require(payload['training_source'] == training_source_hashes()
            and payload['evaluation_source'] == new_source_hashes(), 'Frozen execution source changed')
    require(payload['metric_contract'] == METRIC_CONTRACT
            and set(payload['checkpoint_roles']) == set(CASES)
            and payload['future_target_passed_to_forward'] is False
            and payload['optimizer_updates'] == 0 and payload['candidate_inference_calls'] == 0,
            'Frozen original-only scope or metric contract differs')
    material = payload['protocol']
    require(material['dataset'] in ('Alameda', 'Orange') and material['split'] in ('val', 'test')
            and type(material['engineering_only']) is bool
            and not (material['engineering_only'] and material['split'] == 'test'),
            'Frozen original evaluation scope differs')
    require(material == protocol(material['dataset'], material['split'],
            material['engineering_only'], material['station_ids']), 'Frozen published slice changed')
    role = payload['checkpoint_roles'][CASES[0]]
    require(role['case'] == CASES[0] and role['dataset'] == material['dataset']
            and role['selection'] == 'best' and role['training_complete'] is True
            and role['engineering_only'] is material['engineering_only']
            and role['training_budget_epochs'] == (3 if material['engineering_only'] else 100)
            and role['best_selection_includes_stage_zero'] is False
            and role['model_state_sha256'] == role['model_tensor_sha256'],
            'Frozen role must be the completed original validation BEST')
    expected_budget = 3 if material['engineering_only'] else 100
    require(0 < role['selected_epoch'] == role['best_epoch'] <= role['completed_epoch'] <= expected_budget
            and (role['completed_epoch'] == expected_budget or role['wait'] >= 20),
            'Frozen original completion evidence differs')
    if material['split'] == 'test':
        require(payload['validation_replay_reference']['passed'] is True
                and payload['validation_replay_reference']['checkpoint_sha256'] == role['checkpoint_sha256'],
                'Test requires passed native validation replay for the same BEST')
    for name, expected in payload['source_files'].items():
        path = Path(name)
        require(path.is_absolute() and path.is_file() and sha256(path) == expected,
                'Frozen training/data artifact changed: '+name)


def read_cache(directory, frozen, case=CASES[0], modules=None):
    require(case == CASES[0], 'Only ORIGINAL_IGSTGNN caches are accepted')
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
    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(metadata['runtime']['threads'])
        observed = scores(arrays, modules)
    finally:
        torch.set_num_threads(previous)
    require(observed == metadata['scores'], 'Cached native/common metrics do not reproduce exactly')
    if frozen['payload']['protocol']['split'] == 'val':
        expected = frozen['payload']['checkpoint_roles'][case]['expected_validation_mae']
        require(metadata['validation_replay'] == validation_replay(case, observed, expected),
                'Validation cache lacks exact passed native training replay')
    return arrays, metadata


def write_cache(directory, frozen, case, arrays, metadata, modules=None):
    require(case == CASES[0], 'Only ORIGINAL_IGSTGNN caches are accepted')
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


def collect(model, loader, data, device, material, logger):
    model.eval()
    before, random_before = tensor_tree_hash(model.state_dict()), rng_fingerprint()
    predictions, targets = [], []
    synchronize(device)
    started = time.perf_counter()
    with torch.no_grad():
        for index, raw in enumerate(loader.get_iterator(), 1):
            raw_before = tensor_tree_hash(raw)
            x, y, incident, sensors = original_batch(raw, device)
            prediction = data.scaler.inverse_transform(model(
                x, label=None, incident_data=incident, sensor_data=sensors))
            target = data.scaler.inverse_transform(y)
            require(tensor_tree_hash(raw) == raw_before, 'Inference modified raw input')
            predictions.append(prediction.detach().cpu().numpy())
            targets.append(target.detach().cpu().numpy())
            if index in (1, loader.num_batch):
                logger.info('ORIGINAL_IGSTGNN %s batch=%d/%d', material['split'], index, loader.num_batch)
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


def read_result(directory):
    directory = Path(directory)
    locks = [directory/name for name in ('test_lock.json', 'validation_lock.json') if (directory/name).is_file()]
    require(len(locks) == 1, 'Expected exactly one evaluation split lock')
    frozen = read_json(locks[0])
    verify_frozen(frozen)
    _, metadata = read_cache(directory/CASES[0], frozen)
    result = read_json(directory/'result.json')
    roles, material = frozen['payload']['checkpoint_roles'], frozen['payload']['protocol']
    require(result['schema'] == RESULT_SCHEMA and result['status'] == STATUS
            and result['freeze_id'] == frozen['freeze_id'] and result['protocol'] == material
            and result['checkpoint_roles'] == roles and result['metric_contract'] == METRIC_CONTRACT
            and result['runtime'] == frozen['payload']['runtime'], 'Original result identity differs')
    require(result['dataset'] == material['dataset'] and result['split'] == material['split']
            and result['samples'] == material['count'] and result['engineering_only'] is material['engineering_only']
            and result['test_evaluated'] is (material['split'] == 'test')
            and result['original_checkpoint_role'] == roles[CASES[0]]
            and result['parameter_counts'] == {CASES[0]:roles[CASES[0]]['registered_parameters']}
            and set(result['rows']) == set(CASES) and result['rows'][CASES[0]] == metadata['scores']
            and result['caches'] == {CASES[0]:dict(archive_sha256=metadata['archive_sha256'])}
            and result['candidate_inference_calls'] == 0
            and 'candidate_checkpoint_role' not in result
            and 'relative_gain_percent_vs_replayed_original' not in result,
            'Original-only result scope, metrics or checkpoint aliases differ')
    return result


def validation_reference(directory, dataset, role, environment):
    require(directory is not None, 'Test requires --validation-dir with passed original native replay')
    directory = Path(directory).resolve()
    result = read_result(directory)
    require(result['dataset'] == dataset and result['split'] == 'val'
            and result['engineering_only'] is False and result['original_checkpoint_role'] == role
            and result['runtime'] == environment, 'Native validation replay belongs to a different BEST or runtime')
    files = [directory/'result.json', directory/'validation_lock.json',
             directory/CASES[0]/'metadata.json', directory/CASES[0]/'predictions.npz']
    return dict(passed=True, checkpoint_sha256=role['checkpoint_sha256'],
                validation_directory=str(directory), validation_result_sha256=sha256(files[0])), files


def render_report(directory, result):
    material, case = result['protocol'], CASES[0]
    row = result['rows'][case]['author_metrics']['horizon_macro']
    role = result['original_checkpoint_role']
    lines = ['# Original IGSTGNN: '+material['dataset']+' / '+material['split'], '',
        '{} published samples; original IGSTGNN only.'.format(material['count']),
        'Validation BEST epoch {} is frozen after {} completed epochs.'.format(role['best_epoch'], role['completed_epoch']),
        'Primary: per-lead author FP32 metrics, then an equal mean over 12 leads. RMSE is not pooled RMSE.', '',
        '| Method | Parameters | 12-lead MAE | 12-lead RMSE | 12-lead MAPE % |',
        '|---|---:|---:|---:|---:|',
        '| {} | {} | {:.9f} | {:.9f} | {:.9f} |'.format(case, result['parameter_counts'][case],
            row['mae'], row['rmse'], row['mape_percent']), '',
        'Per-lead results: per_horizon.csv. common_mask_pooled is supplementary in result.json.',
        'The frozen original model uses the project reproduction recipe: batch 48, maximum 100 epochs,',
        'native curriculum and patience-20 stopping; BEST excludes stage zero.',
        'No GW16 training or inference is performed; no improvement rate is inferred from an unpaired result.',
        'This is a single-seed result. Published window overlap and incident first availability remain unverified.',
        'Completed caches are immutable; rerunning reuses them with zero new completed-model forwards.']
    if material['engineering_only']:
        lines += ['', 'ENGINEERING ONLY: four windows and three epochs; no research performance conclusion.']
    (Path(directory)/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    with (Path(directory)/'per_horizon.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['case', 'lead', 'mae', 'rmse', 'mape_percent'])
        writer.writeheader()
        for value in result['rows'][case]['author_metrics']['per_horizon']:
            writer.writerow(dict(case=case, **value))
    write_json(Path(directory)/'result.json', result)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['Alameda', 'Orange'], required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    parser.add_argument('--validation-dir', type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--threads', type=int, default=3)
    parser.add_argument('--engineering', action='store_true')
    args = parser.parse_args(argv)
    if args.engineering and args.split == 'test':
        parser.error('Engineering evaluation is validation-only')
    if args.split == 'test' and args.validation_dir is None:
        parser.error('--split test requires --validation-dir for native BEST replay')
    return args


def main(argv=None):
    args = parse_args(argv)
    sources = (args.run_dir,) if args.validation_dir is None else (args.run_dir, args.validation_dir)
    args.output_dir = safe_output(args.output_dir, sources, args.data_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger('original_multidata_only.evaluate')
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    with ExitStack() as locks:
        for directory in sorted({args.run_dir.resolve(), args.output_dir,
                                 *(() if args.validation_dir is None else (args.validation_dir.resolve(),))}):
            require(directory.is_dir(), 'Required original run or replay directory is missing: '+str(directory))
            locks.enter_context(exclusive_run(directory))
        identity = _read_identity(args.run_dir.resolve())
        environment = configure_runtime(args.device, args.threads, identity['seed'])
        require(read_json(args.run_dir/'runtime.json') == environment,
                'Original evaluation requires its recorded numerical runtime, device, threads and seed')
        modules = get_original_modules()
        data = load_original_data(args.data_dir, args.dataset, identity['config'], modules, logger,
                                  check=args.engineering, training=False)
        weights, original_role, expected, checkpoint = selected_checkpoint(args.run_dir, data)
        role = dict(original_role, model_tensor_sha256=original_role['model_state_sha256'])
        selection_guard(args.dataset, args.split, identity, role, args.engineering)
        material = protocol(args.dataset, args.split, args.engineering, data.station_ids)
        roles = {CASES[0]:role}
        input_paths = [checkpoint, args.run_dir/'identity.json', args.run_dir/'runtime.json',
                       args.run_dir/'last.pt', *(args.data_dir/name for name in material['release_hashes'])]
        replay_reference = None
        if args.split == 'test':
            replay_reference, replay_files = validation_reference(args.validation_dir, args.dataset, role, environment)
            input_paths += replay_files
        frozen = freeze_output(args.output_dir, dict(protocol=material, checkpoint_roles=roles,
            original_identity_sha256=identity['identity_sha256'], training_source=training_source_hashes(),
            evaluation_source=new_source_hashes(), source_files={str(Path(path).resolve()):sha256(path) for path in input_paths},
            runtime=environment, metric_contract=METRIC_CONTRACT, validation_replay_reference=replay_reference,
            future_target_passed_to_forward=False, optimizer_updates=0, candidate_inference_calls=0,
            selection='validation_BEST_only_before_test', independent_confirmation=False))
        verify_frozen(frozen)
        # Original test loader construction must remain below the durable test lock.
        cache = args.output_dir/CASES[0]
        if cache.exists():
            _, metadata = read_cache(cache, frozen, modules=modules)
            logger.info('ORIGINAL_IGSTGNN immutable cache reused; zero new forwards')
        else:
            model = build_original_model(data, modules, args.device)
            model.load_state_dict(weights, strict=True)
            require(tensor_tree_hash(model.state_dict()) == role['model_tensor_sha256']
                    and sum(parameter.numel() for parameter in model.parameters()) == role['registered_parameters'],
                    'Built original model differs from frozen checkpoint')
            loader = original_loader(data, modules, args.split, 2 if args.engineering else identity['config']['bs'], logger)
            observed, audit = collect(model, loader, data, args.device, material, logger)
            replay = validation_replay(CASES[0], scores(observed, modules), expected) if args.split == 'val' else dict(
                applicable=False, reason='test_is_not_used_for_training_replay_or_selection')
            _, metadata = write_cache(cache, frozen, CASES[0], observed, dict(audit=audit, validation_replay=replay), modules)
            del model
            gc.collect()
        verify_frozen(frozen)
        result = dict(schema=RESULT_SCHEMA, status=STATUS, freeze_id=frozen['freeze_id'],
            protocol=material, checkpoint_roles=roles, dataset=args.dataset, split=args.split,
            samples=material['count'], engineering_only=args.engineering, original_checkpoint_role=role,
            parameter_counts={CASES[0]:role['registered_parameters']}, metric_contract=METRIC_CONTRACT,
            runtime=environment, test_evaluated=args.split == 'test', independent_confirmation=False,
            candidate_inference_calls=0, caches={CASES[0]:dict(archive_sha256=metadata['archive_sha256'])},
            rows={CASES[0]:metadata['scores']})
        render_report(args.output_dir, result)
        read_result(args.output_dir)
        logger.info('%s: %s', STATUS, args.output_dir/'report.md')
        return result


if __name__ == '__main__':
    main()
