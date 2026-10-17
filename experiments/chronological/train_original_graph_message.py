"""GW2 four-arm, from-scratch architecture pilot on the unchanged author recipe.

--epochs is the TOTAL completed budget per arm. No early stopping, test loader,
warmstart, learning-rate search, or validation-based architecture search. Partial
epochs replay from the last atomic epoch checkpoint, including optimizer/RNG.
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
from experiments.chronological.original_graph_message import ARMS, DESIGN, build_variant, split_state
from experiments.chronological.train_original_isp import (
    MILESTONES, arm_summary, load_checkpoint, rng_state, run_epochs,
    validate_saved as validate_epoch_state)

SCHEMA = 'original_igstgnn_graph_message_training_v1'
EXECUTION_FILES = (
    'experiments/chronological/train_original_graph_message.py',
    'experiments/chronological/original_graph_message.py',
    'experiments/chronological/train_original_isp.py',
    'experiments/chronological/original_isp.py',
    'experiments/chronological/original_igstgnn_runtime.py',
    'experiments/chronological/original_igstgnn_source.py',
    'experiments/chronological/train_original_igstgnn_baseline.py',
    'experiments/chronological/original_igstgnn_checkpoint_compat.py')
PAIRS = (('ORIGINAL', 'STATIC'), ('ORIGINAL', 'POST_LOCAL'),
         ('ORIGINAL', 'POST_MESSAGE'), ('STATIC', 'POST_LOCAL'),
         ('STATIC', 'POST_MESSAGE'), ('POST_LOCAL', 'POST_MESSAGE'))


def model_for_arm(data, modules, device, arm):
    return build_variant(build_model(data, modules, device), arm)


def initialization(model):
    native, added = split_state(model)
    return dict(original_tensor_sha256=tensor_tree_hash(native),
        added_tensor_sha256=tensor_tree_hash(added),
        post_construction_rng_sha256=tensor_tree_hash(rng_state()),
        registered_parameters=sum(p.numel() for p in model.parameters()),
        added_registered_parameters=sum(v.numel() for v in added.values()),
        registered_parameter_caveat='candidate_registered_tensors_match_but_active_capacity_differs')


def validate_saved(state, identity, arm, run, train_batches):
    if identity.get('schema') != SCHEMA or identity.get('design') != DESIGN or arm not in ARMS:
        raise ValueError('Not a GW2 run identity')
    try:
        validate_epoch_state(state, identity, arm, run, train_batches)
    except ValueError as error:
        raise ValueError(str(error).replace('ISP', 'GW2')) from error


def make_engine(model, data, modules, device, run, logger, seed):
    class ValidationDiagnosticsEngine(modules.engine.IGSTGNN_Engine):
        def evaluate(self, mode):
            if mode != 'val':
                raise ValueError('GW2 runner evaluates validation only')
            if hasattr(self.model, 'start_diagnostics'):
                self.model.start_diagnostics()
            # Generic epoch runner calls diagnostics after timing this full pass.
            return super().evaluate(mode)
    config, batches = data.config, data.loaders['train_loader'].num_batch
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lrate'], weight_decay=config['wdecay'], eps=1e-8)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=MILESTONES, gamma=.5)
    return ValidationDiagnosticsEngine(device=torch.device(device), model=model,
        dataloader=data.loaders, scaler=data.scaler, sampler=None, loss_fn=modules.metrics.masked_mae,
        lrate=config['lrate'], optimizer=optimizer, scheduler=scheduler,
        clip_grad_value=config['clip_grad_value'], max_epochs=CONFIG['max_epochs'],
        patience=config['patience'], log_dir=str(run), logger=logger, seed=seed,
        cl_step=config['cl_epoch']*batches, warm_step=config['warm_epoch']*batches,
        horizon=12, incident=True)


def paired_summary(states, identity, target_epoch):
    present = set(states) == set(ARMS)
    common = min((s['completed_epoch'] for s in states.values()), default=0) if present else 0
    comparisons = {}
    if common:
        for control, candidate in PAIRS:
            a, b = (states[arm]['history'][:common] for arm in (control, candidate))
            ba, bb = (min(rows, key=lambda row: row['validation_mae']) for rows in (a, b))
            comparisons[candidate+'_vs_'+control] = dict(
                common_completed_epochs=common,
                independent_best_at_common_budget={metric+'_gain': ba['validation_'+metric]-bb['validation_'+metric]
                    for metric in ('mae', 'rmse', 'mape_fraction')},
                best_epochs={control: ba['epoch'], candidate: bb['epoch']},
                common_last_epoch={metric+'_gain': a[-1]['validation_'+metric]-b[-1]['validation_'+metric]
                    for metric in ('mae', 'rmse', 'mape_fraction')})
    complete = present and all(s['completed_epoch'] >= target_epoch for s in states.values())
    return dict(schema=SCHEMA, status='GW2_TRAINING_PASS' if complete else 'GW2_TRAINING_PARTIAL',
        engineering_only=identity['engineering_only'], from_scratch=True,
        requested_completed_epochs=target_epoch, common_completed_epochs=common if present else None,
        same_completed_training_budget=present and len({s['completed_epoch'] for s in states.values()}) == 1,
        arms={arm: arm_summary(state, target_epoch, identity['engineering_only']) for arm, state in states.items()},
        comparisons=comparisons, primary_comparison='POST_MESSAGE_vs_ORIGINAL',
        required_structural_controls=['POST_MESSAGE_vs_STATIC', 'POST_MESSAGE_vs_POST_LOCAL'],
        primary_metric='native_validation_mae_pooled',
        primary_baseline='original_IGSTGNN_558ab01_trained_from_scratch_in_same_run',
        initial_original_tensors_equal=True, candidate_initial_router_tensors_equal=True,
        native_optimizer_scheduler_loss_curriculum=True, early_stopping_enabled=False,
        optimizer_reset_on_resume=False, test_evaluated=False,
        interpretation='architecture_pilot_not_paper_test_reproduction_or_incident_causality',
        pilot_caveat='five_epochs_do_not_establish_converged_or_repeated_seed_advantage',
        data_boundaries=identity['data'])


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--data-dir', type=Path, default=default_data_dir())
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--threads', type=int, default=3)
    p.add_argument('--seed', type=int, default=2025)
    p.add_argument('--only', nargs='+', choices=ARMS)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--check', action='store_true', help='Four train/validation rows, batch two; engineering only')
    a = p.parse_args(argv)
    if not 1 <= a.epochs <= CONFIG['max_epochs']:
        p.error('epochs must be a TOTAL completed budget in 1..100')
    if a.only and len(set(a.only)) != len(a.only):
        p.error('only arms must be unique')
    a.run_dir, a.data_dir = a.run_dir.resolve(), a.data_dir.resolve()
    return a


def main(argv=None):
    args = parse_args(argv)
    if args.run_dir.exists() != args.resume:
        raise ValueError('Use a fresh GW2 run directory, or --resume an existing run')
    args.run_dir.mkdir(parents=True, exist_ok=args.resume)
    with run_lock(args.run_dir):
        logger = logging.getLogger('gw2-'+str(args.run_dir))
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(args.run_dir/'record.log', encoding='utf-8')
        logger.addHandler(handler)
        try:
            runtime = configure_runtime(args.device, args.threads, args.seed)
            modules = get_original_modules()
            config = dict(CONFIG, seed=args.seed, device=args.device)
            if args.check:
                config['bs'] = 2
            data = load_data(args.data_dir, config, modules, logger, check=args.check, training=True)
            samples = tuple(data.loaders['train_loader'].samples)
            initializations = {}
            for arm in ARMS:
                configure_runtime(args.device, args.threads, args.seed)
                model = model_for_arm(data, modules, args.device, arm)
                initializations[arm] = initialization(model)
                del model
            for key in ('original_tensor_sha256', 'post_construction_rng_sha256'):
                if len({row[key] for row in initializations.values()}) != 1:
                    raise ValueError('GW2 arms differ in native initialization or construction RNG')
            if len({initializations[arm]['added_tensor_sha256'] for arm in ARMS[1:]}) != 1:
                raise ValueError('GW2 candidate router initializations differ')
            identity = dict(schema=SCHEMA, arms=list(ARMS), design=DESIGN,
                training_config=config, data=data.identity, runtime=runtime,
                source_fingerprint=modules.source_fingerprint,
                execution_source={name: sha256(REPO/name) for name in EXECUTION_FILES},
                initializations=initializations, engineering_only=args.check,
                initial_train_order_sha256=hashlib.sha256(np.arange(len(samples), dtype=np.int64).tobytes()).hexdigest(),
                scheduler=dict(milestones=MILESTONES, gamma=.5), from_scratch=True,
                early_stopping_enabled=False, stage_zero_participates_in_selection=False, test_evaluated=False)
            identity['identity_sha256'] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
            if args.resume:
                if json.loads((args.run_dir/'identity.json').read_text(encoding='utf-8')) != identity:
                    raise ValueError('GW2 data/source/runtime/initialization identity changed; use a fresh directory')
                if args.epochs < json.loads((args.run_dir/'budget.json').read_text())['target_completed_epochs']:
                    raise ValueError('Cannot decrease a saved GW2 budget')
            else:
                write_json(args.run_dir/'identity.json', identity)
            write_json(args.run_dir/'budget.json', dict(target_completed_epochs=args.epochs,
                total_not_additional=True, test_evaluated=False))
            states, batches = {}, data.loaders['train_loader'].num_batch
            for arm in ARMS:
                run = args.run_dir/arm
                if (run/'last.pt').is_file():
                    states[arm] = load_checkpoint(run/'last.pt')
                    validate_saved(states[arm], identity, arm, run, batches)
            for arm in args.only or ARMS:
                run = args.run_dir/arm
                run.mkdir(exist_ok=True)
                state = states.get(arm)
                if state and state['completed_epoch'] >= args.epochs:
                    continue
                configure_runtime(args.device, args.threads, args.seed)
                data.loaders['train_loader'].samples = list(samples)
                model = model_for_arm(data, modules, args.device, arm)
                if initialization(model) != initializations[arm]:
                    raise ValueError('Actual GW2 initialization changed')
                engine = make_engine(model, data, modules, args.device, run, logger, args.seed)
                print(json.dumps(dict(arm=arm, phase='arm_start', from_scratch=state is None,
                    completed_epoch=0 if state is None else state['completed_epoch'],
                    initialization=initializations[arm])), flush=True)
                states[arm] = run_epochs(engine, run, identity, arm, args.epochs, state)
                write_json(args.run_dir/'summary.json', paired_summary(states, identity, args.epochs))
                del engine, model
                gc.collect()
                if torch.device(args.device).type == 'cuda':
                    torch.cuda.empty_cache()
            summary = paired_summary(states, identity, args.epochs)
            write_json(args.run_dir/'summary.json', summary)
            print(json.dumps(summary), flush=True)
            return summary
        except BaseException as error:
            write_json(args.run_dir/('failure_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')+'.json'),
                dict(status='GW2_TRAINING_FAILED', error_type=type(error).__name__, error=str(error),
                    resume_semantics='last_committed_epoch_replays_incomplete_epoch', test_evaluated=False))
            raise
        finally:
            logger.removeHandler(handler)
            handler.close()


if __name__ == '__main__':
    main()
