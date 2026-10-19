"""Replay one original FULL checkpoint on validation, with explicit metric axes."""
import argparse
from contextlib import ExitStack
import json
import logging
import os
from pathlib import Path
import re

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch

from experiments.chronological.original_igstgnn_checkpoint_compat import counter_weights_only_compat
from experiments.chronological.evaluate_original_igstgnn_modules import source_from_run, verify_replay
from experiments.chronological.original_igstgnn_source import get_original_modules, load_archive
from experiments.chronological.original_igstgnn_runtime import (
    RELEASE, build_model, configure_runtime, default_data_dir, device_batch, load_data,
    native_metrics, rng_fingerprint, run_lock, sha256, tensor_tree_hash, write_json)
from experiments.position_state_gw16.objective import numpy_metrics

SCHEMA = 'original_baseline_gw16_validation_v1'
REPO = Path(__file__).resolve().parents[2]
COMMON_CONFIG = dict(target_minimum=.001, mape_minimum=1.)


def native_summary(prediction, target, modules=None):
    """Original engine uses one complete-validation mask value for every lead."""
    modules = get_original_modules() if modules is None else modules
    p = torch.as_tensor(prediction[..., 0], dtype=torch.float32)
    y = torch.as_tensor(target[..., 0], dtype=torch.float32)
    pooled = native_metrics(p, y, modules)
    null = y.new_tensor(pooled['mask_value'])
    rows = []
    for h in range(12):
        mae, mape, rmse = modules.metrics.compute_all_metrics(p[:, h], y[:, h], null)
        rows.append(dict(lead=h+1, mae=float(mae), rmse=float(rmse), mape_percent=100*float(mape)))
    macro = {key: float(np.mean([row[key] for row in rows]))
             for key in ('mae', 'rmse', 'mape_percent')}
    return dict(pooled=pooled, per_horizon=rows, horizon_macro=macro, mask_value=pooled['mask_value'])


def archive_reference(archive):
    """A short archived training run never becomes a fully trained baseline."""
    m, c = archive.metadata, archive.config
    if [row['epoch'] for row in m['training_rows']] != list(range(1, m['completed_epochs']+1)):
        raise ValueError('Original archive completed epoch and training rows disagree')
    patience = c.get('patience')
    early = re.search(r'Early stop at epoch\s+(\d+),\s*loss\s*=', archive.log_text)
    original_budget = c.get('max_epochs') == 100 and patience == 20
    early_proven = (early is not None and int(early[1]) == m['completed_epochs']
                    and m['completed_epochs']-m['best_epoch'] >= 20)
    complete = original_budget and (m['completed_epochs'] == 100 or early_proven)
    return dict(training_complete=bool(complete), completed_epoch=m['completed_epochs'],
                best_epoch=m['best_epoch'],
                reference_level='fully_trained_original_protocol' if complete else 'historical_reference',
                completion_evidence='original_100_epoch_or_patience20_log' if complete else 'not_proven')


def protect_output(output, data_dir, source_run=None):
    output, data_dir = Path(output).resolve(), Path(data_dir).resolve()
    if output.exists() or output == data_dir or data_dir in output.parents:
        raise ValueError('Use a fresh output outside the data directory')
    if source_run is not None:
        source_run = Path(source_run).resolve()
        if output == source_run or source_run in output.parents:
            raise ValueError('Output must be outside the read-only source run')
    for parent in output.parents:
        if (parent/'identity.json').is_file() or (parent/'metadata.json').is_file():
            raise ValueError('Output cannot be nested in another run or evaluation')


def recheck_files(source, execution, data_dir, hashes):
    for path, expected in source['source_files'].items():
        if sha256(path) != expected:
            raise ValueError('Read-only original source artifact changed: '+path)
    for name, expected in execution.items():
        if sha256(REPO/name) != expected:
            raise ValueError('Evaluation source changed: '+name)
    for name, expected in hashes.items():
        if sha256(Path(data_dir)/name) != expected:
            raise ValueError('Published data changed: '+name)
    return get_original_modules()


def validate_arrays(arrays, engineering_only):
    required = {'prediction', 'target', 'sample_ids', 'station_ids'}
    if set(arrays) != required:
        raise ValueError('Original prediction archive fields differ')
    p, y, samples, stations = (arrays[k] for k in ('prediction', 'target', 'sample_ids', 'station_ids'))
    count = 4 if engineering_only else 837
    if (p.shape != (count, 12, 496, 1) or y.shape != p.shape
            or p.dtype != np.float32 or y.dtype != np.float32
            or not np.isfinite(p).all() or not np.isfinite(y).all()
            or samples.dtype != np.int64 or not np.array_equal(samples, np.arange(3909, 3909+count))
            or stations.dtype != np.int64 or stations.shape != (496,) or len(np.unique(stations)) != 496):
        raise ValueError('Original prediction values, dtype or explicit axes differ')


def read_archive(directory):
    """Read an authenticated completed replay, recomputing both metric contracts."""
    directory = Path(directory)
    metadata = json.loads((directory/'metadata.json').read_text(encoding='utf-8'))
    if (metadata.get('schema') != SCHEMA or metadata.get('test_evaluated') is not False
            or metadata['native_replay'].get('passed') is not True):
        raise ValueError('Expected authenticated original FULL validation replay')
    path = directory/'predictions.npz'
    if sha256(path) != metadata['archive_sha256']:
        raise ValueError('Original prediction archive fingerprint changed')
    with np.load(path, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    validate_arrays(arrays, metadata['engineering_only'])
    source, data = metadata['source'], metadata['data_identity']
    if data['hashes'] != RELEASE:
        raise ValueError('Original prediction archive is not the certified author release')
    modules = recheck_files(source, metadata['execution_source'], metadata['data_dir'], data['hashes'])
    if metadata['source_fingerprint'] != modules.source_fingerprint or metadata['registered_parameters'] != 455220:
        raise ValueError('Original source or registered parameter identity differs')
    if (not np.array_equal(arrays['station_ids'], metadata['station_ids'])
            or data['splits']['val']['selected_indices'] != arrays['sample_ids'].tolist()):
        raise ValueError('Original archive and metadata sample/station axes disagree')
    # Native reductions are FP32 and depend on CPU thread partitioning.
    # Replay their saved partition; never loosen a changed-cache tolerance.
    threads = metadata['runtime']['threads']
    if type(threads) is not int or threads < 1:
        raise ValueError('Invalid original cached metric thread count')
    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(threads)
        summary = native_summary(arrays['prediction'], arrays['target'], modules)
    finally:
        torch.set_num_threads(previous)
    pooled, rows, macro = (summary[key] for key in ('pooled', 'per_horizon', 'horizon_macro'))
    common = numpy_metrics(arrays['prediction'], arrays['target'], COMMON_CONFIG)
    for actual, name in ((pooled, 'native_metrics'), (rows, 'native_per_horizon'),
                         (macro, 'native_macro_average'), (common, 'gw16_common_metrics')):
        if actual != metadata[name]:
            raise ValueError('Original cached metrics changed: '+name)
    verify_replay(pooled, metadata['expected_validation_mae'], engineering_only=metadata['engineering_only'])
    return arrays, metadata


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--source-run', type=Path)
    group.add_argument('--archive', type=Path)
    group.add_argument('--checkpoint', type=Path)
    parser.add_argument('--training-log', type=Path)
    parser.add_argument('--data-dir', type=Path, default=default_data_dir())
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--threads', type=int, default=3)
    parser.add_argument('--check', action='store_true', help='Only four validation windows; engineering only')
    args = parser.parse_args(argv)
    if bool(args.checkpoint) != bool(args.training_log):
        parser.error('--checkpoint and --training-log are required together')
    return args


def main(argv=None):
    args = parse_args(argv)
    protect_output(args.output_dir, args.data_dir, args.source_run)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with ExitStack() as locks:
        try:
            if args.source_run:
                if not (args.source_run/'active.lock').is_file():
                    raise ValueError('Original source run lacks its training lock')
                locks.enter_context(run_lock(args.source_run))
                with counter_weights_only_compat():
                    state, config, expected, source = source_from_run(args.source_run, check=args.check)
                source['reference_level'] = ('fully_trained_original_protocol' if source['training_complete']
                                             else 'partial_original_training')
            else:
                archive = load_archive(args.archive, checkpoint=args.checkpoint, train_log=args.training_log)
                if archive.metadata['parameter_count'] != 455220 or archive.metadata['counts'] != dict(train=3909, val=837, node_num=496):
                    raise ValueError('Original archive parameter count or release split differs')
                state, config, expected = archive.state_dict, archive.config, archive.best_validation_mae
                source = dict(archive.provenance, training_metadata=archive.metadata, **archive_reference(archive))
                source['source_files'] = {source['path']: sha256(source['path'])}
                if source['kind'] == 'checkpoint_and_log':
                    source['source_files'][source['log_path']] = sha256(source['log_path'])
                if sha256(source['path']) != source.get('sha256', source['checkpoint_sha256']):
                    raise ValueError('Original archive changed while loading')
                if source['kind'] == 'checkpoint_and_log' and sha256(source['log_path']) != source['training_log_sha256']:
                    raise ValueError('Original training log changed while loading')
            runtime = configure_runtime(args.device, args.threads, config['seed'])
            modules = get_original_modules()
            paths = [Path(__file__), REPO/'experiments/position_state_gw16/objective.py',
                     REPO/'experiments/chronological/evaluate_original_igstgnn_modules.py',
                     REPO/'experiments/chronological/train_original_igstgnn_baseline.py',
                     REPO/'experiments/chronological/original_igstgnn_checkpoint_compat.py',
                     *sorted((REPO/'experiments/chronological').glob('original_igstgnn_*.py'))]
            execution = {str(path.relative_to(REPO)): sha256(path) for path in paths}
            data = load_data(args.data_dir, config, modules, logging.getLogger(__name__), check=args.check)
            if args.source_run:
                old = source['input_data_identity']
                if (source['source_fingerprint'] != modules.source_fingerprint
                        or old['hashes'] != data.identity['hashes']
                        or old['normalization'] != data.identity['normalization']
                        or (not args.check and old['splits']['val'] != data.identity['splits']['val'])):
                    raise ValueError('Original training and replay data/source identity differ')
            model = build_model(data, modules, args.device)
            model.load_state_dict(state, strict=True)
            model.eval()
            if sum(p.numel() for p in model.parameters()) != 455220:
                raise ValueError('Original full model parameter count differs')
            before, rng = tensor_tree_hash(model.state_dict()), rng_fingerprint()
            prediction, target, partition = [], [], []
            with torch.no_grad():
                for number, raw in enumerate(data.loaders['val_loader'].get_iterator(), 1):
                    batch = device_batch(raw, args.device)
                    x, y, incident, sensors = batch
                    fingerprint = tensor_tree_hash(batch)
                    output = model(x, y, incident_data=incident, sensor_data=sensors)
                    prediction.append(data.scaler.inverse_transform(output).cpu().numpy().copy())
                    target.append(data.scaler.inverse_transform(y).cpu().numpy().copy())
                    partition.append(len(x))
                    if tensor_tree_hash(batch) != fingerprint:
                        raise ValueError('Original FULL forward changed its inputs')
                    print(json.dumps(dict(phase='original_full_validation', batch=number,
                                          total=data.loaders['val_loader'].num_batch)), flush=True)
            if tensor_tree_hash(model.state_dict()) != before or rng_fingerprint() != rng:
                raise ValueError('Original FULL evaluation changed weights or RNG')
            arrays = dict(prediction=np.concatenate(prediction), target=np.concatenate(target),
                          sample_ids=np.asarray(data.identity['splits']['val']['selected_indices'], dtype=np.int64),
                          station_ids=data.station_ids)
            validate_arrays(arrays, args.check)
            summary = native_summary(arrays['prediction'], arrays['target'], modules)
            pooled, rows, macro = (summary[key] for key in ('pooled', 'per_horizon', 'horizon_macro'))
            replay = verify_replay(pooled, expected, engineering_only=args.check)
            replay.update(passed=True, checkpoint_strict_load=True, model_state_sha256=before,
                          batch_partition=partition)
            recheck_files(source, execution, data.directory, data.identity['hashes'])
            np.savez_compressed(args.output_dir/'predictions.npz', **arrays)
            complete = bool(source['training_complete']) and not args.check
            status = ('ORIGINAL_VALIDATION_COMPLETE' if complete else
                      'ORIGINAL_VALIDATION_PARTIAL' if args.source_run else 'ORIGINAL_VALIDATION_REFERENCE')
            metadata = dict(schema=SCHEMA, status=status,
                source=source, source_fingerprint=modules.source_fingerprint, execution_source=execution,
                data_dir=str(data.directory), data_directory=str(data.directory), training_config=config,
                data_identity=data.identity, station_ids=data.station_ids.tolist(),
                native_metrics=pooled, native_per_horizon=rows, native_macro_average=macro,
                gw16_common_metrics=numpy_metrics(arrays['prediction'], arrays['target'], COMMON_CONFIG),
                gw16_common_metric_config=COMMON_CONFIG, native_replay=replay, expected_validation_mae=expected,
                registered_parameters=455220, runtime=runtime, engineering_only=args.check, test_evaluated=False,
                archive_sha256=sha256(args.output_dir/'predictions.npz'), optimizer_updates=0,
                original_full_model=True, native_pos=False, equal_budget_gw16_training_control=False)
            write_json(args.output_dir/'metadata.json', metadata)
            read_archive(args.output_dir)
            lines = ['# Original IGSTGNN FULL validation', '',
                'Unmodified frozen author model (558ab01), not NATIVE_POS. Validation only; no test evaluation.',
                'This uses the original training protocol and is not a GW16 equal-budget training control.',
                'Reference level: '+source['reference_level']+'. Engineering only: '+str(args.check)+'.', '',
                '| Metric contract | MAE | RMSE | MAPE % |', '|---|---:|---:|---:|']
            for label, values in [('Native pooled validation', pooled), ('Native 12-lead equal average', macro)]:
                mape = values.get('mape_percent', 100*values.get('mape_fraction', 0))
                lines.append(f'| {label} | {values["mae"]:.9f} | {values["rmse"]:.9f} | {mape:.9f} |')
            common = metadata['gw16_common_metrics']
            lines.append(f'| GW16 common target mask | {common["mae"]:.9f} | {common["rmse"]:.9f} | {common["mape_percent"]:.9f} |')
            lines += ['', 'Native per-lead metrics use one global validation null value, as the original engine does.',
                      'The native 12-lead average has the paper table format, but this is validation and cannot establish test improvement.',
                      'No retraining, architecture ablation, parameter updates or source modification occurs here.']
            (args.output_dir/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
            print(json.dumps(dict(status=metadata['status'], output_dir=str(args.output_dir),
                                  training_complete=complete, test_evaluated=False)), flush=True)
            return metadata
        except BaseException as error:
            write_json(args.output_dir/'failure.json', dict(schema=SCHEMA, error_type=type(error).__name__,
                                                          error=str(error), test_evaluated=False))
            raise


if __name__ == '__main__':
    main()
