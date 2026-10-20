"""Server-only full-batch training memory check, without changing either recipe.

One batch of 48 training windows is used twice: the first Adam update allocates
its moments, then the second forward/backward exercises their resident memory.
No validation or test iterator is used. Disposable weights never become a run.
"""
import argparse
from datetime import datetime, timezone
import gc
import logging
import os
from pathlib import Path
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch

from experiments.chronological.original_igstgnn_runtime import device_batch as original_device_batch
from experiments.chronological.original_igstgnn_source import get_original_modules
from experiments.position_state_gw16.objective import loss_terms
from .baseline import build_original_model, load_original_data, original_config
from .common import (digest_json, exclusive_run, load_config, protected_output,
    read_json, require, runtime, source_hashes, spec, synchronize, write_json)
from .data import authenticate_release, device_batch, load_data
from .model import build_model

SCHEMA = 'state_exchange_multidata_batch_check_v1'
BATCH_SIZE = 48
CASES = ('STATE_EXCHANGE', 'ORIGINAL_IGSTGNN')


def profile_step(model, optimizer, loss_builder, device, clip_norm=5.):
    """Measure both Adam allocation and a subsequent forward/backward peak."""
    model.train()
    cuda = torch.device(device).type == 'cuda'
    if cuda:
        torch.cuda.reset_peak_memory_stats(device)
        baseline = torch.cuda.memory_allocated(device)
    else:
        baseline = 0
    synchronize(device)
    started = time.perf_counter()
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_builder()
        require(loss.ndim == 0 and bool(torch.isfinite(loss)), 'Full-batch training loss is not finite')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm, error_if_nonfinite=True)
        optimizer.step()
    synchronize(device)
    seconds = time.perf_counter()-started
    peak = torch.cuda.max_memory_allocated(device) if cuda else 0
    require(all(bool(torch.isfinite(value).all()) for value in model.parameters()),
            'Full-batch optimizer update produced nonfinite weights')
    return dict(loss=float(loss.detach()), preclip_gradient_norm=float(norm), seconds=seconds,
        registered_parameters=sum(value.numel() for value in model.parameters()),
        optimizer_state_entries=len(optimizer.state), peak_allocated_bytes=peak,
        incremental_peak_allocated_bytes=max(0, peak-baseline), device=str(device),
        batch_size=BATCH_SIZE, optimizer_updates=2, warmup_updates=1, measured_updates=1,
        scope='two_updates_same_full_batch_peak_covers_Adam_allocation_and_resident_moments_forward_backward',
        validation_forward_calls=0, test_forward_calls=0, persistent_model_checkpoint=False)


def candidate_check(directory, dataset, device, logger):
    config = load_config(dataset)
    data = load_data(directory, config, logger, engineering=False)
    loader = data.loaders['train_loader']
    raw = next(iter(loader.get_iterator()))
    require(len(raw['x_data']) == BATCH_SIZE and config['batch_size'] == BATCH_SIZE,
            'Candidate check requires the unchanged full batch of 48')
    model = build_model(data, config, 'STATE_EXCHANGE', device)
    model.set_stage('joint')
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'],
        weight_decay=config['weight_decay'], eps=config['adam_eps'])
    def loss_builder():
        history, target, event = device_batch(raw, data, device)
        output = model(history, event)
        return loss_terms(output, target, model.geometry, config, stage='joint')['loss']
    result = profile_step(model, optimizer, loss_builder, device, config['clip_grad_norm'])
    return dict(result, case='STATE_EXCHANGE', nodes=config['nodes'],
                loss_contract='unchanged_GW16_node_MAE_plus_regional_MAE')


def original_check(directory, dataset, device, logger):
    config = original_config(dataset, 2025, device)
    modules = get_original_modules()
    data = load_original_data(directory, dataset, config, modules, logger, training=True)
    raw = next(iter(data.loaders['train_loader'].get_iterator()))
    require(len(raw['x_data']) == BATCH_SIZE and config['bs'] == BATCH_SIZE,
            'Original check requires the unchanged full batch of 48')
    model = build_original_model(data, modules, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lrate'],
        weight_decay=config['wdecay'], eps=1e-8)
    def loss_builder():
        history, target, event, sensors = original_device_batch(raw, device)
        prediction = model(history, target, incident_data=event, sensor_data=sensors)
        prediction, target = data.scaler.inverse_transform(prediction), data.scaler.inverse_transform(target)
        # Frozen engine's first update uses all 12 leads and this batch's null.
        null = target.min() if target.min() < 1 else target.new_tensor(0.)
        return modules.metrics.masked_mae(prediction[:, :12], target[:, :12], null)
    result = profile_step(model, optimizer, loss_builder, device, config['clip_grad_value'])
    return dict(result, case='ORIGINAL_IGSTGNN', nodes=data.config['node_num'],
                loss_contract='unchanged_original_first_update_12_lead_native_masked_MAE')


def _release_device(device):
    gc.collect()
    if torch.device(device).type == 'cuda':
        torch.cuda.empty_cache()


def validate_cached(record, identity):
    value = dict(record)
    saved_hash = value.pop('record_sha256', None)
    require(saved_hash == digest_json(value), 'Full-batch check record changed')
    require(record.get('schema') == SCHEMA and record.get('status') == 'FULL_BATCH_TRAINING_CHECK_COMPLETE'
        and record.get('identity') == identity and set(record.get('checks', {})) == set(CASES),
        'Full-batch check source, dataset or numerical runtime changed')
    for case in CASES:
        row = record['checks'][case]
        require(row['case'] == case and row['batch_size'] == BATCH_SIZE and row['optimizer_updates'] == 2
            and row['warmup_updates'] == row['measured_updates'] == 1
            and row['nodes'] == spec(identity['dataset'])['nodes']
            and row['validation_forward_calls'] == row['test_forward_calls'] == 0
            and row['persistent_model_checkpoint'] is False,
            'Full-batch check dimensions or training scope changed')


def run(args):
    output = protected_output(args.output_dir, args.data_dir)
    output.mkdir(parents=True, exist_ok=True)
    with exclusive_run(output):
        try:
            environment = runtime(args.device, args.threads, 2025)
            _, hashes = authenticate_release(args.data_dir, args.dataset)
            identity = dict(dataset=args.dataset, seed=2025, batch_size=BATCH_SIZE,
                data_hashes=hashes, execution_source=source_hashes(), runtime=environment,
                cases=list(CASES), no_adaptive_batch_reduction=True)
            if (output/'metadata.json').is_file():
                record = read_json(output/'metadata.json')
                validate_cached(record, identity)
                print('Full batch 48 check reused: '+str(output/'metadata.json'), flush=True)
                return record
            checks = {}
            logger = logging.getLogger(__name__)
            for case, operation in (('STATE_EXCHANGE', candidate_check), ('ORIGINAL_IGSTGNN', original_check)):
                _release_device(args.device)
                # Reset initialization and RNG before each disposable method.
                runtime(args.device, args.threads, 2025)
                print('Full batch 48 training memory check: '+args.dataset+' '+case, flush=True)
                checks[case] = operation(args.data_dir, args.dataset, args.device, logger)
                _release_device(args.device)
            require(source_hashes() == identity['execution_source'], 'Source changed during full-batch check')
            record = dict(schema=SCHEMA, status='FULL_BATCH_TRAINING_CHECK_COMPLETE',
                identity=identity, checks=checks, test_evaluated=False,
                formal_training_updates=0, batch_size_changed=False,
                interpretation='engineering_memory_and_finite_update_check_not_predictive_evidence')
            record['record_sha256'] = digest_json(record)
            validate_cached(record, identity)
            write_json(output/'metadata.json', record)
            write_json(output/'summary.json', record)
            print('Full batch 48 check passed: '+str(output/'metadata.json'), flush=True)
            return record
        except BaseException as error:
            write_json(output/('failure_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')+'.json'),
                dict(status='FULL_BATCH_TRAINING_CHECK_FAILED', error_type=type(error).__name__,
                    error=str(error), batch_size=BATCH_SIZE, batch_size_changed=False,
                    interpretation='stop_without_shrinking_batch_or_starting_formal_training', test_evaluated=False))
            raise


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=('Alameda', 'Contra_Costa', 'Orange'), required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--threads', type=int, default=3)
    args = parser.parse_args(argv)
    require(args.threads > 0, 'Threads must be positive')
    args.data_dir, args.output_dir = args.data_dir.resolve(), args.output_dir.resolve()
    return args


if __name__ == '__main__':
    run(parse_args())
