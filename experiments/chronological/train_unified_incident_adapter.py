"""Six matched scratch arms for the report-free core and one incident adapter.

Uses the frozen original engine/recipe and the existing atomic epoch runner.
Early stopping is explicitly disabled for the common completed-epoch budget.
No test evaluation, warm start, source checkpoint replacement or parameter sweep.
"""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import logging
import os
from pathlib import Path
import sys

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.chronological.original_igstgnn_source import get_original_modules
from experiments.chronological.original_igstgnn_runtime import (
    CONFIG, build_model, configure_runtime, default_data_dir, load_data, run_lock,
    sha256, tensor_tree_hash, write_json)
from experiments.chronological.train_original_isp import (
    MILESTONES, rng_state, load_checkpoint, validate_saved, run_epochs, arm_summary)
from experiments.chronological.unified_incident_adapter import (
    ARMS, ADAPTER_CONFIG, build_variant, common_traffic_state)

SCHEMA = 'unified_incident_adapter_scratch_training_v02'
EXECUTION_FILES = (
    'experiments/chronological/train_unified_incident_adapter.py',
    'experiments/chronological/unified_incident_adapter.py',
    'experiments/chronological/train_original_isp.py',
    'experiments/chronological/original_isp.py',
    'experiments/chronological/original_igstgnn_runtime.py',
    'experiments/chronological/original_igstgnn_source.py',
    'experiments/chronological/train_original_igstgnn_baseline.py',
    'experiments/chronological/original_igstgnn_checkpoint_compat.py')
COMPARISONS = (
    ('ORIGINAL', 'BOTH'), ('TRAFFIC_CORE', 'TIME'), ('TRAFFIC_CORE', 'GRAPH'),
    ('TRAFFIC_CORE', 'BOTH'), ('TIME', 'BOTH'), ('GRAPH', 'BOTH'), ('STACKED', 'BOTH'))


def model_for_arm(data, modules, device, arm):
    original = build_model(data, modules, device)
    original_hash = tensor_tree_hash(original.state_dict())
    model = build_variant(original, arm)
    model._original_initial_tensor_sha256 = original_hash
    return model


def initialization(model, arm):
    adapter = getattr(model, 'adapter', None)
    inactive = 0
    if arm == 'TIME':
        inactive = sum(p.numel() for p in adapter.graph.parameters())
    elif arm == 'GRAPH':
        inactive = sum(p.numel() for layer in (adapter.output_state, adapter.output_node) for p in layer.parameters())
    return dict(original_initial_tensor_sha256=model._original_initial_tensor_sha256,
        traffic_tensor_sha256=tensor_tree_hash(common_traffic_state(model)),
        adapter_tensor_sha256=None if adapter is None else tensor_tree_hash(adapter.state_dict()),
        post_construction_rng_sha256=tensor_tree_hash(rng_state()),
        registered_parameters=sum(p.numel() for p in model.parameters()),
        adapter_registered_parameters=0 if adapter is None else sum(p.numel() for p in adapter.parameters()),
        inactive_head_registered_parameters=inactive,
        zero_adapter_reference=('complete_original' if arm == 'STACKED' else 'report_free_traffic_core')
            if adapter is not None else None,
        native_ICSF_TIID_retained=arm in ('ORIGINAL', 'STACKED'),
        registered_parameter_caveat='registered_size_does_not_imply_equal_active_capacity')


def verify_initializations(initializations):
    for key in ('original_initial_tensor_sha256', 'traffic_tensor_sha256', 'post_construction_rng_sha256'):
        if len({v[key] for v in initializations.values()}) != 1:
            raise ValueError('Unified arms do not share '+key)
    adapters = {v['adapter_tensor_sha256'] for v in initializations.values() if v['adapter_tensor_sha256'] is not None}
    if len(adapters) > 1:
        raise ValueError('Adapter arms do not share initialization')


def paired_summary(states, identity, target_epoch):
    all_present = bool(states) and set(states) == set(identity['arms'])
    common = min((s['completed_epoch'] for s in states.values()), default=0)
    complete = all_present and common >= target_epoch
    comparisons = {}
    for control, candidate in COMPARISONS:
        if not (all_present and common and control in states and candidate in states):
            continue
        a, b = states[control]['history'][:common], states[candidate]['history'][:common]
        best_a, best_b = min(a, key=lambda r: r['validation_mae']), min(b, key=lambda r: r['validation_mae'])
        comparisons[candidate+'_vs_'+control] = dict(
            independent_best_at_common_budget={
                metric+'_gain': best_a['validation_'+metric]-best_b['validation_'+metric]
                for metric in ('mae', 'rmse', 'mape_fraction')},
            best_epochs={control: best_a['epoch'], candidate: best_b['epoch']},
            common_last_epoch={metric+'_gain': a[-1]['validation_'+metric]-b[-1]['validation_'+metric]
                for metric in ('mae', 'rmse', 'mape_fraction')})
    return dict(schema=SCHEMA, status='UNIFIED_ADAPTER_TRAINING_PASS' if complete else 'UNIFIED_ADAPTER_TRAINING_PARTIAL',
        run_kind='matched_scratch_pilot' if target_epoch < 100 else 'matched_scratch_fixed_budget',
        requested_completed_epochs=target_epoch, common_completed_epochs=common if all_present else None,
        same_completed_training_budget=all_present and len({s['completed_epoch'] for s in states.values()}) == 1,
        engineering_only=identity['engineering_only'],
        arms={arm: arm_summary(state, target_epoch, identity['engineering_only']) for arm, state in states.items()},
        comparisons=comparisons, primary_baseline='original_IGSTGNN_558ab01', primary_baseline_present='ORIGINAL' in states,
        common_traffic_initialization=True, common_adapter_initialization=True,
        shared_seed_and_initial_train_order=True, optimizer_reset_on_resume=False,
        early_stopping_enabled=False, validation_used_for_model_selection=True, test_evaluated=False,
        zero_adapter_recovers='TRAFFIC_CORE_except_STACKED_recovers_ORIGINAL',
        interpretation='architecture_comparison_not_incident_causality_or_physical_capacity', data_boundaries=identity['data'])


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, default=default_data_dir())
    parser.add_argument('--epochs', type=int, default=5, help='Total completed epochs, not additional epochs')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--threads', type=int, default=3)
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--arms', nargs='+', choices=ARMS, default=list(ARMS))
    parser.add_argument('--only', nargs='+', choices=ARMS, help='Run a subset of registered arms; all remain in identity')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--check', action='store_true', help='Four train/validation rows, batch two; engineering only')
    args = parser.parse_args(argv)
    if not 1 <= args.epochs <= CONFIG['max_epochs']:
        parser.error('epochs must be in 1..100')
    if args.threads < 1 or len(set(args.arms)) != len(args.arms) or (args.only and (
            len(set(args.only)) != len(args.only) or not set(args.only) <= set(args.arms))):
        parser.error('threads must be positive; arms unique; only a unique subset of arms')
    args.run_dir, args.data_dir = args.run_dir.resolve(), args.data_dir.resolve()
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.run_dir.exists() != args.resume:
        raise ValueError('Use a fresh unified-adapter run directory, or --resume an existing run')
    args.run_dir.mkdir(parents=True, exist_ok=args.resume)
    with run_lock(args.run_dir):
        handler = None
        try:
            runtime = configure_runtime(args.device, args.threads, args.seed)
            modules = get_original_modules()
            logger = logging.getLogger('unified-adapter-'+str(args.run_dir))
            logger.setLevel(logging.INFO)
            handler = logging.FileHandler(args.run_dir/'record.log', encoding='utf-8')
            logger.addHandler(handler)
            config = dict(CONFIG, seed=args.seed, device=args.device)
            if args.check:
                config['bs'] = 2
            data = load_data(args.data_dir, config, modules, logger, check=args.check, training=True)
            original_samples = tuple(data.loaders['train_loader'].samples)
            initializations = {}
            for arm in args.arms:
                configure_runtime(args.device, args.threads, args.seed)
                model = model_for_arm(data, modules, args.device, arm)
                initializations[arm] = initialization(model, arm)
                del model
            verify_initializations(initializations)
            identity = dict(schema=SCHEMA, arms=args.arms, training_config=config, adapter_config=ADAPTER_CONFIG,
                data=data.identity, source_fingerprint=modules.source_fingerprint,
                execution_source={p: sha256(REPO/p) for p in EXECUTION_FILES}, runtime=runtime,
                initializations=initializations, engineering_only=args.check,
                initial_train_order_sha256=hashlib.sha256(np.arange(len(original_samples), dtype=np.int64).tobytes()).hexdigest(),
                scheduler=dict(milestones=MILESTONES, gamma=.5), early_stopping_enabled=False,
                from_scratch=True, test_evaluated=False)
            identity['identity_sha256'] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
            if args.resume:
                if json.loads((args.run_dir/'identity.json').read_text(encoding='utf-8')) != identity:
                    raise ValueError('Run identity differs; keep old source/runtime/data or use a fresh directory')
                if args.epochs < json.loads((args.run_dir/'budget.json').read_text(encoding='utf-8'))['target_completed_epochs']:
                    raise ValueError('Cannot decrease an existing completed-epoch budget')
            else:
                write_json(args.run_dir/'identity.json', identity)
                write_json(args.run_dir/'runtime.json', runtime)
            write_json(args.run_dir/'budget.json', dict(target_completed_epochs=args.epochs,
                total_not_additional=True, early_stopping_enabled=False, test_evaluated=False))
            states, batches = {}, data.loaders['train_loader'].num_batch
            for arm in args.arms:
                run = args.run_dir/arm
                if (run/'last.pt').is_file():
                    states[arm] = load_checkpoint(run/'last.pt')
                    validate_saved(states[arm], identity, arm, run, batches)
            for arm in args.only or args.arms:
                run = args.run_dir/arm
                run.mkdir(exist_ok=True)
                state = states.get(arm)
                if state and state['completed_epoch'] >= args.epochs:
                    write_json(run/'summary.json', arm_summary(state, args.epochs, args.check))
                    continue
                configure_runtime(args.device, args.threads, args.seed)
                data.loaders['train_loader'].samples = list(original_samples)
                model = model_for_arm(data, modules, args.device, arm)
                if initialization(model, arm) != initializations[arm]:
                    raise ValueError('Actual arm initialization differs from preflight')
                optimizer = torch.optim.Adam(model.parameters(), lr=config['lrate'], weight_decay=config['wdecay'], eps=1e-8)
                scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=MILESTONES, gamma=.5)
                engine = modules.engine.IGSTGNN_Engine(device=torch.device(args.device), model=model,
                    dataloader=data.loaders, scaler=data.scaler, sampler=None, loss_fn=modules.metrics.masked_mae,
                    lrate=config['lrate'], optimizer=optimizer, scheduler=scheduler,
                    clip_grad_value=config['clip_grad_value'], max_epochs=CONFIG['max_epochs'],
                    patience=config['patience'], log_dir=str(run), logger=logger, seed=args.seed,
                    cl_step=config['cl_epoch']*batches, warm_step=config['warm_epoch']*batches,
                    horizon=12, incident=True)
                print(json.dumps(dict(arm=arm, phase='arm_start', from_scratch=state is None,
                    completed_epoch=0 if state is None else state['completed_epoch'], initialization=initializations[arm])), flush=True)
                states[arm] = run_epochs(engine, run, identity, arm, args.epochs, state)
                write_json(args.run_dir/'summary.json', paired_summary(states, identity, args.epochs))
                del engine, model, optimizer, scheduler
                gc.collect()
                if torch.device(args.device).type == 'cuda':
                    torch.cuda.empty_cache()
            summary = paired_summary(states, identity, args.epochs)
            write_json(args.run_dir/'summary.json', summary)
            print(json.dumps(summary), flush=True)
            return summary
        except BaseException as error:
            write_json(args.run_dir/('failure_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')+'.json'),
                dict(status='UNIFIED_ADAPTER_TRAINING_FAILED', error_type=type(error).__name__, error=str(error),
                    checkpoint_semantics='resume_last_committed_epoch_replays_incomplete_epoch', test_evaluated=False))
            raise
        finally:
            if handler is not None:
                logger.removeHandler(handler)
                handler.close()


if __name__ == '__main__':
    main()
