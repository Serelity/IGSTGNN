"""Disposable full-batch training check for STATE_EXCHANGE alone.

Reuse the unchanged candidate preflight: two Adam updates on the first 48
training windows. No original model is constructed and no formal state is saved.
"""
import argparse
from datetime import datetime, timezone
import logging
from pathlib import Path

from experiments.state_exchange_multidata.batch_check import (
    BATCH_SIZE, _release_device, candidate_check)
from experiments.state_exchange_multidata.common import (
    digest_json, exclusive_run, protected_output, read_json, require, runtime,
    spec, write_json)
from experiments.state_exchange_multidata.data import authenticate_release
from .common import source_hashes

SCHEMA = 'state_exchange_candidate_batch_check_v1'
STATUS = 'CANDIDATE_FULL_BATCH_TRAINING_CHECK_COMPLETE'
CASES = ('STATE_EXCHANGE',)


def validate_cached(record, identity):
    value = dict(record)
    saved_hash = value.pop('record_sha256', None)
    require(saved_hash == digest_json(value), 'Candidate full-batch check record changed')
    require(record.get('schema') == SCHEMA and record.get('status') == STATUS
        and record.get('identity') == identity
        and identity.get('cases') == list(CASES)
        and identity.get('no_original_model_forwards') is True
        and set(record.get('checks', {})) == set(CASES),
        'Candidate full-batch check source, dataset or numerical runtime changed')
    row = record['checks']['STATE_EXCHANGE']
    require(row['case'] == 'STATE_EXCHANGE' and row['batch_size'] == BATCH_SIZE
        and row['optimizer_updates'] == 2
        and row['warmup_updates'] == row['measured_updates'] == 1
        and row['nodes'] == spec(identity['dataset'])['nodes']
        and row['validation_forward_calls'] == row['test_forward_calls'] == 0
        and row['persistent_model_checkpoint'] is False
        and record.get('formal_training_updates') == 0
        and record.get('original_model_forward_calls') == 0
        and record.get('batch_size_changed') is False
        and record.get('test_evaluated') is False,
        'Candidate full-batch check dimensions or training scope changed')


def run(args):
    require(args.dataset in ('Alameda', 'Orange'),
            'This preflight only trains Alameda or Orange; Contra Costa reuses its completed result')
    output = protected_output(args.output_dir, args.data_dir)
    output.mkdir(parents=True, exist_ok=True)
    with exclusive_run(output):
        try:
            environment = runtime(args.device, args.threads, 2025)
            _, hashes = authenticate_release(args.data_dir, args.dataset)
            identity = dict(dataset=args.dataset, seed=2025, batch_size=BATCH_SIZE,
                data_hashes=hashes, execution_source=source_hashes(), runtime=environment,
                cases=list(CASES), no_adaptive_batch_reduction=True,
                no_original_model_forwards=True)
            if (output/'metadata.json').is_file():
                record = read_json(output/'metadata.json')
                validate_cached(record, identity)
                print('Candidate batch 48 check reused: '+str(output/'metadata.json'), flush=True)
                return record
            _release_device(args.device)
            runtime(args.device, args.threads, 2025)
            print('Candidate batch 48 training memory check: '+args.dataset+' STATE_EXCHANGE', flush=True)
            checks = {'STATE_EXCHANGE': candidate_check(
                args.data_dir, args.dataset, args.device, logging.getLogger(__name__))}
            _release_device(args.device)
            require(source_hashes() == identity['execution_source'],
                    'Source changed during candidate full-batch check')
            record = dict(schema=SCHEMA, status=STATUS, identity=identity, checks=checks,
                test_evaluated=False, formal_training_updates=0,
                original_model_forward_calls=0, batch_size_changed=False,
                interpretation='engineering_memory_and_finite_update_check_not_predictive_evidence')
            record['record_sha256'] = digest_json(record)
            validate_cached(record, identity)
            write_json(output/'metadata.json', record)
            write_json(output/'summary.json', record)
            print('Candidate batch 48 check passed: '+str(output/'metadata.json'), flush=True)
            return record
        except BaseException as error:
            write_json(output/('failure_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')+'.json'),
                dict(status='CANDIDATE_FULL_BATCH_TRAINING_CHECK_FAILED',
                    error_type=type(error).__name__, error=str(error), batch_size=BATCH_SIZE,
                    batch_size_changed=False, original_model_forward_calls=0,
                    formal_training_updates=0, test_evaluated=False,
                    interpretation='stop_without_shrinking_batch_or_starting_formal_training'))
            raise


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=('Alameda', 'Orange'), required=True)
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
