"""GW4 frozen-anchor adaptation; reuse all finished baseline artifacts read-only.

--epochs is the TOTAL new adaptation budget (1..5), independent of the source
GW2 budget. A new optimizer/scheduler/curriculum starts at adaptation step zero.
Interrupted epochs replay from the last committed epoch; no test evaluation.
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
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import numpy as np
import torch

from experiments.chronological import train_original_graph_message as base
from experiments.chronological.original_graph_message import GraphMessageIGSTGNN
from experiments.chronological.original_receiver_residual import (
    ARMS, DESIGN, ReceiverResidualIGSTGNN, split_state)
from experiments.chronological.receiver_residual_reference import (
    certify, compatible, recertify, read_json, require)


SCHEMA = 'original_igstgnn_receiver_residual_training_v1'
EXECUTION_FILES = tuple(dict.fromkeys((*base.EXECUTION_FILES,
    'experiments/chronological/receiver_match_reference.py',
    'experiments/chronological/graph_message_reuse.py',
    'experiments/chronological/receiver_residual_reference.py',
    'experiments/chronological/original_receiver_residual.py',
    'experiments/chronological/train_original_receiver_residual.py')))
PAIRS = (('STATIC_RESIDUAL', 'MESSAGE_RESIDUAL'),
    ('STATIC_RESIDUAL', 'RECEIVER_RESIDUAL'), ('MESSAGE_RESIDUAL', 'RECEIVER_RESIDUAL'))


def model_for_arm(data, modules, device, arm, source_weights):
    anchor = GraphMessageIGSTGNN(base.build_model(data, modules, device), 'POST_MESSAGE')
    anchor.load_state_dict(source_weights, strict=True)
    return ReceiverResidualIGSTGNN(anchor, arm)


def initialization(model):
    frozen, added = split_state(model.state_dict())
    return dict(frozen_tensor_sha256=base.tensor_tree_hash(frozen),
        added_tensor_sha256=base.tensor_tree_hash(added),
        post_construction_rng_sha256=base.tensor_tree_hash(base.rng_state()),
        registered_parameters=model.param_num(),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        added_registered_parameters=sum(v.numel() for v in added.values()),
        capacity_caveat='MESSAGE_and_RECEIVER_equal_active_capacity_STATIC_only_bias_active')


def check_frozen(state, identity):
    frozen, _ = split_state(state)
    require(base.tensor_tree_hash(frozen) == identity['reference']['source_model_tensor_sha256'],
            'GW4 frozen anchor tensors changed')


def validate_saved(state, identity, arm, directory, batches):
    require(identity.get('schema') == SCHEMA and identity.get('design') == DESIGN and arm in ARMS,
            'Expected a GW4 adaptation identity')
    base.validate_epoch_state(state, identity, arm, directory, batches)
    check_frozen(state['model_state'], identity)
    if state['best_checkpoint_file']:
        best = base.load_checkpoint(directory/state['best_checkpoint_file'])
        check_frozen(best, identity)


def make_engine(model, data, modules, device, directory, logger, seed, identity):
    class FrozenAnchorEngine(modules.engine.IGSTGNN_Engine):
        def train_batch(self):
            values = super().train_batch()
            check_frozen(self.model.state_dict(), identity)
            return values

        def evaluate(self, mode):
            require(mode == 'val', 'GW4 evaluates validation only')
            self.model.start_diagnostics()
            return super().evaluate(mode)
    active = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    require(bool(active) and all('.residual.' in name for name, _ in active),
            'Only the GW4 residual may receive optimizer updates')
    config, batches = data.config, data.loaders['train_loader'].num_batch
    optimizer = torch.optim.Adam([p for _, p in active], lr=config['lrate'],
        weight_decay=config['wdecay'], eps=1e-8)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=base.MILESTONES, gamma=.5)
    return FrozenAnchorEngine(device=torch.device(device), model=model, dataloader=data.loaders,
        scaler=data.scaler, sampler=None, loss_fn=modules.metrics.masked_mae,
        lrate=config['lrate'], optimizer=optimizer, scheduler=scheduler,
        clip_grad_value=config['clip_grad_value'], max_epochs=base.CONFIG['max_epochs'],
        patience=config['patience'], log_dir=str(directory), logger=logger, seed=seed,
        cl_step=config['cl_epoch']*batches, warm_step=config['warm_epoch']*batches,
        horizon=12, incident=True)


def paired_summary(states, identity, epochs):
    complete = set(states) == set(ARMS) and all(s['completed_epoch'] == epochs for s in states.values())
    comparisons = {}
    for control, candidate in PAIRS:
        if control not in states or candidate not in states:
            continue
        common = min(states[name]['completed_epoch'] for name in (control, candidate))
        if not common:
            continue
        a, b = (states[name]['history'][:common] for name in (control, candidate))
        best_a, best_b = (min(rows, key=lambda r: r['validation_mae']) for rows in (a, b))
        comparisons[candidate+'_vs_'+control] = dict(common_adaptation_epochs=common,
            independent_best_at_common_budget={key+'_gain': best_a['validation_'+key]-best_b['validation_'+key]
                for key in ('mae', 'rmse', 'mape_fraction')},
            common_last_epoch={key+'_gain': a[-1]['validation_'+key]-b[-1]['validation_'+key]
                for key in ('mae', 'rmse', 'mape_fraction')})
    anchor = identity['reference']['source_best_metrics']
    for arm, state in states.items():
        if not state['history']:
            continue
        best = min(state['history'], key=lambda r: r['validation_mae'])
        comparisons[arm+'_vs_GW2_BASE'] = dict(
            best_adaptation_gain={key+'_gain': anchor[key]-best['validation_'+key]
                for key in ('mae', 'rmse', 'mape_fraction')},
            last_adaptation_gain={key+'_gain': anchor[key]-state['history'][-1]['validation_'+key]
                for key in ('mae', 'rmse', 'mape_fraction')},
            caveat='source_GW2_plus_extra_adaptation_budget_not_equal_from_scratch_budget')
    return dict(schema=SCHEMA, status='GW4_ADAPTATION_COMPLETE' if complete else 'GW4_ADAPTATION_PARTIAL',
        engineering_only=False, from_scratch=False, reference=identity['reference'],
        requested_adaptation_epochs=epochs, adaptation_epoch_cap=5,
        same_completed_adaptation_budget=complete,
        arms={arm: base.arm_summary(s, epochs, False) for arm, s in states.items()},
        comparisons=comparisons, primary_comparison='RECEIVER_RESIDUAL_vs_MESSAGE_RESIDUAL',
        required_controls=['RECEIVER_RESIDUAL_vs_STATIC_RESIDUAL', 'RECEIVER_RESIDUAL_vs_GW2_BASE'],
        old_parameters_frozen=True, old_modules_eval=True, baseline_new_training_updates=0,
        baseline_new_inference_calls=0, optimizer_reset_at_adaptation_start=True,
        optimizer_reset_on_resume=False, scheduler_and_curriculum_restart_at_adaptation_zero=True,
        source_optimizer_state_reused=False, stage_zero_in_best_selection=False,
        early_stopping_enabled=False, test_evaluated=False,
        caveat='completion_is_not_evidence_of_improvement_convergence_multiseed_or_independent_test')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--gw2-run', type=Path, required=True)
    parser.add_argument('--gw2-best-eval', type=Path)
    parser.add_argument('--gw2-last-eval', type=Path)
    parser.add_argument('--data-dir', type=Path, default=base.default_data_dir())
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--threads', type=int, default=3)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--only', nargs='+', choices=ARMS)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    if not 1 <= args.epochs <= 5:
        parser.error('GW4 stage one is capped at 1..5 TOTAL adaptation epochs')
    if args.only and len(set(args.only)) != len(args.only):
        parser.error('only arms must be unique')
    args.run_dir, args.gw2_run, args.data_dir = (p.resolve() for p in (args.run_dir, args.gw2_run, args.data_dir))
    return args


def main(argv=None):
    args = parse_args(argv)
    require(args.run_dir.exists() == args.resume, 'Use a fresh GW4 directory, or --resume an existing one')
    require(args.run_dir != args.gw2_run and args.gw2_run not in args.run_dir.parents,
            'Place GW4 outside the immutable GW2 run')
    args.run_dir.mkdir(parents=True, exist_ok=args.resume)
    with base.run_lock(args.run_dir):
        logger = logging.getLogger('gw4-'+str(args.run_dir))
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(args.run_dir/'record.log', encoding='utf-8')
        logger.addHandler(handler)
        try:
            runtime = base.configure_runtime(args.device, args.threads, args.seed)
            source_path = args.run_dir/'reference.json'
            if source_path.exists():
                reference = read_json(source_path)
                require(reference['source_run'] == str(args.gw2_run), 'Cannot replace a GW4 source run')
                source_identity, source_weights = recertify(reference)
            else:
                reference, source_identity, source_weights = certify(args.gw2_run,
                    {key: getattr(args, 'gw2_'+key+'_eval') for key in ('best', 'last')})
            modules = base.get_original_modules()
            config = dict(base.CONFIG, seed=args.seed, device=args.device)
            data = base.load_data(args.data_dir, config, modules, logger, check=False, training=True)
            samples = tuple(data.loaders['train_loader'].samples)
            initializations = {}
            for arm in ARMS:
                base.configure_runtime(args.device, args.threads, args.seed)
                model = model_for_arm(data, modules, args.device, arm, source_weights)
                initializations[arm] = initialization(model)
                require(initializations[arm]['frozen_tensor_sha256'] == reference['source_model_tensor_sha256'],
                        'GW4 construction altered its anchor')
                del model
            for key in ('frozen_tensor_sha256', 'added_tensor_sha256', 'post_construction_rng_sha256'):
                require(len({row[key] for row in initializations.values()}) == 1, 'GW4 initializations differ: '+key)
            require(initializations['MESSAGE_RESIDUAL']['trainable_parameters'] ==
                initializations['RECEIVER_RESIDUAL']['trainable_parameters'], 'Active matching capacity differs')
            identity = dict(schema=SCHEMA, design=DESIGN, arms=list(ARMS), training_config=config,
                data=data.identity, runtime=runtime, source_fingerprint=modules.source_fingerprint,
                engineering_only=False, reference=reference, initializations=initializations,
                execution_source={name: base.sha256(REPO/name) for name in EXECUTION_FILES},
                initial_train_order_sha256=hashlib.sha256(np.arange(len(samples), dtype=np.int64).tobytes()).hexdigest(),
                scheduler=dict(milestones=base.MILESTONES, gamma=.5),
                adaptation_protocol=dict(old_parameters_frozen=True, old_modules_eval=True,
                    optimizer='fresh_Adam_on_active_residual_only', source_optimizer_reused=False,
                    scheduler_and_curriculum_start=0, epoch_cap=5),
                from_scratch=False, early_stopping_enabled=False,
                stage_zero_participates_in_selection=False, test_evaluated=False)
            compatible(source_identity, identity)
            identity['identity_sha256'] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
            identity_path = args.run_dir/'identity.json'
            if identity_path.exists():
                require(read_json(identity_path) == identity, 'GW4 source/data/runtime/initialization identity changed')
            else:
                require(all(p.name in ('active.lock', 'record.log') or
                    (p.is_file() and p.name.startswith('failure_') and p.suffix == '.json')
                    for p in args.run_dir.iterdir()), 'Missing GW4 identity in a nonempty run')
                base.write_json(identity_path, identity)
            if not source_path.exists():
                base.write_json(source_path, reference)
            budget_path = args.run_dir/'budget.json'
            if budget_path.exists():
                require(args.epochs >= read_json(budget_path)['target_completed_epochs'], 'Cannot decrease adaptation budget')
            else:
                require(not any(args.run_dir.glob('*/last.pt')), 'Saved states require their budget file')
            base.write_json(budget_path, dict(target_completed_epochs=args.epochs,
                total_adaptation_not_additional_to_saved_adaptation=True,
                source_GW2_epochs=reference['source_completed_epochs'], test_evaluated=False))
            states, batches = {}, data.loaders['train_loader'].num_batch
            for arm in ARMS:
                path = args.run_dir/arm/'last.pt'
                if path.exists():
                    states[arm] = base.load_checkpoint(path)
                    validate_saved(states[arm], identity, arm, path.parent, batches)
                    require(states[arm]['completed_epoch'] <= args.epochs, 'Saved arm exceeds adaptation budget')
            for arm in args.only or ARMS:
                directory = args.run_dir/arm
                directory.mkdir(exist_ok=True)
                state = states.get(arm)
                if state and state['completed_epoch'] == args.epochs:
                    continue
                base.configure_runtime(args.device, args.threads, args.seed)
                data.loaders['train_loader'].samples = list(samples)
                model = model_for_arm(data, modules, args.device, arm, source_weights)
                require(initialization(model) == initializations[arm], 'Actual GW4 initialization changed')
                engine = make_engine(model, data, modules, args.device, directory, logger, args.seed, identity)
                states[arm] = base.run_epochs(engine, directory, identity, arm, args.epochs, state)
                validate_saved(states[arm], identity, arm, directory, batches)
                base.write_json(args.run_dir/'summary.json', paired_summary(states, identity, args.epochs))
                del engine, model
                gc.collect()
                if torch.device(args.device).type == 'cuda':
                    torch.cuda.empty_cache()
            recertify(reference)
            summary = paired_summary(states, identity, args.epochs)
            base.write_json(args.run_dir/'summary.json', summary)
            print(json.dumps(summary), flush=True)
            return summary
        except BaseException as error:
            base.write_json(args.run_dir/('failure_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')+'.json'),
                dict(status='GW4_ADAPTATION_FAILED', error_type=type(error).__name__, error=str(error),
                    resume_semantics='last_atomic_adaptation_epoch_replays_unfinished_epoch', test_evaluated=False))
            raise
        finally:
            logger.removeHandler(handler)
            handler.close()


if __name__ == '__main__':
    main()
