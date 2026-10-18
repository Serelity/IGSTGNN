"""GW5: four parameter-matched policies from GW2, reusing finished GW4 controls.

Each new arm starts with zero GW4 receiver residuals, not the trained GW4 head.
The new total adaptation budget is capped at five; only that completed budget
is paired to the existing five-epoch GW4 control. No test evaluation.
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

from experiments.chronological import train_original_receiver_residual as gw4
from experiments.chronological.original_graph_message import GraphMessageIGSTGNN
from experiments.chronological.original_history_anchor import ARMS, DESIGN, HistoryAnchorIGSTGNN
from experiments.chronological.history_anchor_reference import certify, compatible, read_json, recertify, require


base = gw4.base
SCHEMA = 'original_igstgnn_history_anchor_training_v1'
EXECUTION_FILES = (*gw4.EXECUTION_FILES, 'experiments/chronological/original_history_anchor.py',
    'experiments/chronological/history_anchor_reference.py', 'experiments/chronological/train_original_history_anchor.py')
PAIRS = (('HISTORY_FIXED', 'HISTORY_ANCHORED'), ('PHASE_SHRINK', 'HISTORY_ANCHORED'),
    ('BIAS_SHRINK', 'HISTORY_ANCHORED'))


def model_for_arm(data, modules, device, arm, source_weights):
    anchor = GraphMessageIGSTGNN(base.build_model(data, modules, device), 'POST_MESSAGE')
    anchor.load_state_dict(source_weights, strict=True)
    return HistoryAnchorIGSTGNN(anchor, arm)


def initialization(model):
    return dict(gw4.initialization(model), capacity_caveat='all_GW5_policies_match_GW4_RECEIVER_active_and_registered_capacity')


def validate_saved(state, identity, arm, directory, batches):
    require(identity.get('schema') == SCHEMA and identity.get('design') == DESIGN and arm in ARMS,
            'Expected a GW5 history-anchor identity')
    base.validate_epoch_state(state, identity, arm, directory, batches)
    gw4.check_frozen(state['model_state'], identity)
    if state['best_checkpoint_file']:
        gw4.check_frozen(base.load_checkpoint(directory/state['best_checkpoint_file']), identity)


def paired_summary(states, identity, epochs, old_states=None):
    comparisons = {}
    old_states = old_states or {}
    for control, candidate in (*PAIRS, *((a, b) for a in gw4.ARMS for b in ARMS)):
        left, right = (states if control in ARMS else old_states).get(control), states.get(candidate)
        if left is None or right is None:
            continue
        common = min(left['completed_epoch'], right['completed_epoch'])
        # Old-prefix BEST predictions were not regenerated or independently
        # certified. Report old/new comparisons only at the completed budget.
        if not common or (control in gw4.ARMS and common != identity['gw4_reference']['completed_adaptation_epochs']):
            continue
        a, b = left['history'][:common], right['history'][:common]
        best_a, best_b = (min(rows, key=lambda row:row['validation_mae']) for rows in (a, b))
        comparisons[candidate+'_vs_'+control] = dict(common_adaptation_epochs=common,
            independent_best_at_common_budget={key+'_gain':best_a['validation_'+key]-best_b['validation_'+key]
                for key in ('mae', 'rmse', 'mape_fraction')},
            common_last_epoch={key+'_gain':a[-1]['validation_'+key]-b[-1]['validation_'+key]
                for key in ('mae', 'rmse', 'mape_fraction')})
    source_metrics = identity['reference']['source_best_metrics']
    for arm, state in states.items():
        if state['history']:
            best = min(state['history'], key=lambda row:row['validation_mae'])
            comparisons[arm+'_vs_GW2_BASE'] = dict(best_adaptation_gain={key+'_gain':source_metrics[key]-best['validation_'+key]
                for key in ('mae', 'rmse', 'mape_fraction')},
                caveat='GW2_plus_extra_adaptation_not_equal_from_scratch_budget')
    complete = set(states) == set(ARMS) and all(s['completed_epoch'] == epochs for s in states.values())
    return dict(schema=SCHEMA, status='GW5_ADAPTATION_COMPLETE' if complete else 'GW5_ADAPTATION_PARTIAL',
        engineering_only=False, from_scratch=False, requested_adaptation_epochs=epochs,
        same_completed_adaptation_budget=complete, GW4_control_budget_matched=complete and epochs == 5,
        reference=identity['reference'], gw4_reference=identity['gw4_reference'],
        arms={arm:base.arm_summary(state, epochs, False) for arm, state in states.items()}, comparisons=comparisons,
        newly_trained_arms=list(ARMS), reused_arms=list(gw4.ARMS),
        primary_comparison='HISTORY_ANCHORED_vs_RECEIVER_RESIDUAL',
        required_controls=['HISTORY_ANCHORED_vs_HISTORY_FIXED', 'HISTORY_ANCHORED_vs_PHASE_SHRINK',
            'HISTORY_ANCHORED_vs_BIAS_SHRINK',
            'HISTORY_ANCHORED_vs_STATIC_RESIDUAL', 'HISTORY_ANCHORED_vs_MESSAGE_RESIDUAL'],
        old_parameters_frozen=True, old_modules_eval=True,
        baseline_new_training_updates=0, baseline_new_inference_calls=0,
        stage_zero_in_best_selection=False, optimizer_reset_at_adaptation_start=True,
        optimizer_reset_on_resume=False, early_stopping_enabled=False, test_evaluated=False,
        caveat='single_anchor_single_seed_exploratory_validation_not_independent_confirmation')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--gw4-run', type=Path, required=True)
    parser.add_argument('--gw4-best-eval', type=Path)
    parser.add_argument('--gw4-last-eval', type=Path)
    parser.add_argument('--data-dir', type=Path, default=base.default_data_dir())
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--threads', type=int, default=3)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--only', nargs='+', choices=ARMS)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    if not 1 <= args.epochs <= 5:
        parser.error('GW5 stage one uses 1..5 TOTAL new adaptation epochs')
    if args.only and len(set(args.only)) != len(args.only):
        parser.error('only policies must be unique')
    args.run_dir, args.gw4_run, args.data_dir = (p.resolve() for p in (args.run_dir, args.gw4_run, args.data_dir))
    return args


def main(argv=None):
    args = parse_args(argv)
    require(args.run_dir.exists() == args.resume, 'Use a fresh GW5 directory, or --resume an existing one')
    require(args.run_dir != args.gw4_run and args.gw4_run not in args.run_dir.parents,
            'Place GW5 outside the immutable GW4 run')
    if (args.gw4_run/'identity.json').is_file():
        protected = Path(read_json(args.gw4_run/'identity.json')['reference']['source_run']).resolve()
        require(args.run_dir != protected and protected not in args.run_dir.parents,
                'Place GW5 outside the immutable GW2 run')
    if args.resume and (args.run_dir/'identity.json').exists():
        require(read_json(args.run_dir/'identity.json').get('schema') == SCHEMA,
                'Cannot resume or write logs into a different experiment')
    args.run_dir.mkdir(parents=True, exist_ok=args.resume)
    with base.run_lock(args.run_dir):
        logger = logging.getLogger('gw5-'+str(args.run_dir)); logger.setLevel(logging.INFO)
        handler = logging.FileHandler(args.run_dir/'record.log', encoding='utf-8'); logger.addHandler(handler)
        try:
            runtime = base.configure_runtime(args.device, args.threads, args.seed)
            reference_path = args.run_dir/'gw4_reference.json'
            if reference_path.exists():
                reference = read_json(reference_path)
                require(reference['source_run'] == str(args.gw4_run), 'Cannot replace the GW4 reference')
                old_identity, weights, old_states = recertify(reference)
            else:
                reference, old_identity, weights, old_states = certify(args.gw4_run,
                    {key:getattr(args, 'gw4_'+key+'_eval') for key in ('best', 'last')})
            modules = base.get_original_modules()
            config = dict(base.CONFIG, seed=args.seed, device=args.device)
            data = base.load_data(args.data_dir, config, modules, logger, check=False, training=True)
            samples, initializations = tuple(data.loaders['train_loader'].samples), {}
            for arm in ARMS:
                base.configure_runtime(args.device, args.threads, args.seed)
                model = model_for_arm(data, modules, args.device, arm, weights)
                initializations[arm] = initialization(model)
                expected = old_identity['initializations']['RECEIVER_RESIDUAL']
                for key in ('frozen_tensor_sha256', 'added_tensor_sha256', 'post_construction_rng_sha256',
                            'registered_parameters', 'trainable_parameters'):
                    require(initializations[arm][key] == expected[key], 'GW5/GW4 paired initialization differs: '+key)
                del model
            identity = dict(schema=SCHEMA, design=DESIGN, arms=list(ARMS), training_config=config,
                data=data.identity, runtime=runtime, source_fingerprint=modules.source_fingerprint,
                engineering_only=False, reference=old_identity['reference'], gw4_reference=reference,
                initializations=initializations, execution_source={name:base.sha256(REPO/name) for name in EXECUTION_FILES},
                initial_train_order_sha256=hashlib.sha256(np.arange(len(samples), dtype=np.int64).tobytes()).hexdigest(),
                scheduler=dict(milestones=base.MILESTONES, gamma=.5), from_scratch=False,
                adaptation_protocol=old_identity['adaptation_protocol'], stage_zero_participates_in_selection=False,
                early_stopping_enabled=False, test_evaluated=False)
            compatible(old_identity, identity)
            require(identity['initial_train_order_sha256'] == old_identity['initial_train_order_sha256'],
                    'GW5/GW4 initial sample order differs')
            identity['identity_sha256'] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
            identity_path = args.run_dir/'identity.json'
            if identity_path.exists():
                require(read_json(identity_path) == identity, 'GW5 identity changed')
            else:
                require(all(p.name in ('active.lock', 'record.log') or
                    (p.is_file() and p.name.startswith('failure_') and p.suffix == '.json')
                    for p in args.run_dir.iterdir()), 'Missing identity in a nonempty GW5 run')
                base.write_json(identity_path, identity)
            if not reference_path.exists():
                base.write_json(reference_path, reference)
            budget_path = args.run_dir/'budget.json'
            if budget_path.exists():
                require(args.epochs >= read_json(budget_path)['target_completed_epochs'], 'Cannot decrease adaptation budget')
            else:
                require(not any(args.run_dir.glob('*/last.pt')), 'Saved GW5 states require their budget file')
            base.write_json(budget_path, dict(target_completed_epochs=args.epochs, total_not_additional=True,
                GW4_reference_adaptation_epochs=5, test_evaluated=False))
            states, batches = {}, data.loaders['train_loader'].num_batch
            for arm in ARMS:
                path = args.run_dir/arm/'last.pt'
                if path.exists():
                    states[arm] = base.load_checkpoint(path)
                    validate_saved(states[arm], identity, arm, path.parent, batches)
                    require(states[arm]['completed_epoch'] <= args.epochs, 'Saved policy exceeds requested budget')
            for arm in args.only or ARMS:
                directory = args.run_dir/arm; directory.mkdir(exist_ok=True)
                state = states.get(arm)
                if state and state['completed_epoch'] == args.epochs:
                    continue
                base.configure_runtime(args.device, args.threads, args.seed)
                data.loaders['train_loader'].samples = list(samples)
                model = model_for_arm(data, modules, args.device, arm, weights)
                require(initialization(model) == initializations[arm], 'Actual GW5 initialization changed')
                engine = gw4.make_engine(model, data, modules, args.device, directory, logger, args.seed, identity)
                states[arm] = base.run_epochs(engine, directory, identity, arm, args.epochs, state)
                validate_saved(states[arm], identity, arm, directory, batches)
                base.write_json(args.run_dir/'summary.json', paired_summary(states, identity, args.epochs, old_states))
                del engine, model
                gc.collect()
                if torch.device(args.device).type == 'cuda':
                    torch.cuda.empty_cache()
            recertify(reference)
            summary = paired_summary(states, identity, args.epochs, old_states)
            base.write_json(args.run_dir/'summary.json', summary)
            print(json.dumps(summary), flush=True)
            return summary
        except BaseException as error:
            base.write_json(args.run_dir/('failure_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')+'.json'),
                dict(status='GW5_ADAPTATION_FAILED', error_type=type(error).__name__, error=str(error),
                    resume_semantics='last_atomic_adaptation_epoch', test_evaluated=False))
            raise
        finally:
            logger.removeHandler(handler); handler.close()


if __name__ == '__main__':
    main()
