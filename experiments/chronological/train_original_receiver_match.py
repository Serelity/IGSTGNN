"""GW3 three new arms only; read-only GW2 references, original training recipe.

--epochs is total, not additional. Epoch checkpoints preserve optimizer,
scheduler, curriculum, sample order and RNG. No early stopping or test batches.
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
from experiments.chronological.original_receiver_match import ARMS, DESIGN, build_variant
from experiments.chronological.receiver_match_reference import certify, read_json, require


SCHEMA = 'original_igstgnn_receiver_match_training_v1'
EXECUTION_FILES = (*base.EXECUTION_FILES,
    'experiments/chronological/original_receiver_match.py',
    'experiments/chronological/train_original_receiver_match.py',
    'experiments/chronological/receiver_match_reference.py')
PAIRS = (('ORIGINAL', 'RECEIVER_MATCH'), ('STATIC', 'RECEIVER_MATCH'),
    ('POST_MESSAGE', 'RECEIVER_MATCH'), ('FIXED_UNIFORM', 'MESSAGE_SELF'),
    ('FIXED_UNIFORM', 'RECEIVER_MATCH'), ('MESSAGE_SELF', 'RECEIVER_MATCH'))


def model_for_arm(data, modules, device, arm):
    return build_variant(base.build_model(data, modules, device), arm)


def initialization(model):
    row = base.initialization(model)
    row['trainable_parameters'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    row['registered_parameter_caveat'] = 'two_matching_arms_equal_active_capacity;FIXED_UNIFORM_router_inactive'
    return row


def validate_saved(state, identity, arm, directory, batches):
    require(identity.get('schema') == SCHEMA and identity.get('design') == DESIGN and arm in ARMS,
            'Not a GW3 run identity')
    base.validate_epoch_state(state, identity, arm, directory, batches)


def paired_summary(states, identity, epochs, reference=None, old_states=None):
    all_states = {**(old_states or {}), **states}
    complete = set(states) == set(ARMS) and all(s['completed_epoch'] == epochs for s in states.values())
    common = min((s['completed_epoch'] for s in all_states.values()), default=0)
    comparisons = {}
    for control, candidate in PAIRS:
        if not common or control not in all_states or candidate not in all_states:
            continue
        a, b = (all_states[arm]['history'][:common] for arm in (control, candidate))
        ba, bb = (min(rows, key=lambda r: r['validation_mae']) for rows in (a, b))
        comparisons[candidate+'_vs_'+control] = dict(common_completed_epochs=common,
            independent_best_at_common_budget={key+'_gain': ba['validation_'+key]-bb['validation_'+key]
                for key in ('mae', 'rmse', 'mape_fraction')},
            common_last_epoch={key+'_gain': a[-1]['validation_'+key]-b[-1]['validation_'+key]
                for key in ('mae', 'rmse', 'mape_fraction')},
            best_epochs={control: ba['epoch'], candidate: bb['epoch']})
    return dict(schema=SCHEMA, status='GW3_TRAINING_PASS' if complete else 'GW3_TRAINING_PARTIAL',
        engineering_only=identity['engineering_only'], from_scratch=True,
        requested_completed_epochs=epochs, common_completed_epochs=common,
        same_completed_training_budget=complete and len({s['completed_epoch'] for s in all_states.values()}) == 1,
        arms={arm: dict(base.arm_summary(state, epochs, identity['engineering_only']),
                reused=arm not in ARMS) for arm, state in all_states.items()},
        comparisons=comparisons, reference=reference,
        newly_trained_arms=list(ARMS), reused_arms=list((old_states or {}).keys()),
        baseline_new_training_updates=0, baseline_new_inference_calls=0,
        primary_comparison='RECEIVER_MATCH_vs_ORIGINAL',
        required_controls=['RECEIVER_MATCH_vs_FIXED_UNIFORM', 'RECEIVER_MATCH_vs_MESSAGE_SELF'],
        native_optimizer_scheduler_loss_curriculum=True, optimizer_reset_on_resume=False,
        best_selection_includes_stage_zero=False, early_stopping_enabled=False, test_evaluated=False,
        caveat='exploratory_validation_not_convergence_multiseed_or_independent_test_evidence')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--gw2-run', type=Path)
    parser.add_argument('--gw2-best-eval', type=Path)
    parser.add_argument('--gw2-last-eval', type=Path)
    parser.add_argument('--data-dir', type=Path, default=base.default_data_dir())
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--threads', type=int, default=3)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--only', nargs='+', choices=ARMS)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--check', action='store_true', help='Four train/val rows, batch two; no historical reference')
    args = parser.parse_args(argv)
    if not 1 <= args.epochs <= base.CONFIG['max_epochs']:
        parser.error('epochs must be a TOTAL completed budget in 1..100')
    if args.only and len(set(args.only)) != len(args.only):
        parser.error('only arms must be unique')
    if not args.check and args.gw2_run is None:
        parser.error('--gw2-run required for a full pilot; old experiments are never retrained')
    if args.check and (args.gw2_run or args.gw2_best_eval or args.gw2_last_eval):
        parser.error('--check must not mix miniature data with full historical references')
    args.run_dir, args.data_dir = args.run_dir.resolve(), args.data_dir.resolve()
    return args


def main(argv=None):
    args = parse_args(argv)
    require(args.run_dir.exists() == args.resume, 'Use a fresh GW3 directory, or --resume an existing one')
    args.run_dir.mkdir(parents=True, exist_ok=args.resume)
    with base.run_lock(args.run_dir):
        logger = logging.getLogger('gw3-'+str(args.run_dir))
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(args.run_dir/'record.log', encoding='utf-8')
        logger.addHandler(handler)
        try:
            runtime = base.configure_runtime(args.device, args.threads, args.seed)
            modules = base.get_original_modules()
            config = dict(base.CONFIG, seed=args.seed, device=args.device)
            if args.check:
                config['bs'] = 2
            data = base.load_data(args.data_dir, config, modules, logger, check=args.check, training=True)
            samples = tuple(data.loaders['train_loader'].samples)
            initializations = {}
            for arm in ('ORIGINAL', *ARMS):
                base.configure_runtime(args.device, args.threads, args.seed)
                model = base.build_model(data, modules, args.device) if arm == 'ORIGINAL' else model_for_arm(data, modules, args.device, arm)
                initializations[arm] = initialization(model)
                del model
            for key in ('original_tensor_sha256', 'post_construction_rng_sha256'):
                require(len({row[key] for row in initializations.values()}) == 1, 'GW3 native initialization/RNG differs')
            require(len({initializations[arm]['added_tensor_sha256'] for arm in ARMS}) == 1, 'GW3 router initializations differ')
            identity = dict(schema=SCHEMA, design=DESIGN, arms=list(ARMS), training_config=config,
                data=data.identity, runtime=runtime, source_fingerprint=modules.source_fingerprint,
                initializations=initializations, engineering_only=args.check,
                reference_run=str(args.gw2_run.resolve()) if args.gw2_run else None,
                execution_source={name: base.sha256(REPO/name) for name in EXECUTION_FILES},
                initial_train_order_sha256=hashlib.sha256(np.arange(len(samples), dtype=np.int64).tobytes()).hexdigest(),
                scheduler=dict(milestones=base.MILESTONES, gamma=.5), from_scratch=True,
                early_stopping_enabled=False, stage_zero_participates_in_selection=False, test_evaluated=False)
            identity['identity_sha256'] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
            if args.resume and (args.run_dir/'identity.json').is_file():
                require(read_json(args.run_dir/'identity.json') == identity, 'GW3 data/source/runtime identity changed')
                if (args.run_dir/'budget.json').is_file():
                    require(args.epochs >= read_json(args.run_dir/'budget.json')['target_completed_epochs'], 'Cannot decrease budget')
                else:
                    require(not any(args.run_dir.glob('*/last.pt')), 'Saved checkpoints require their original budget file')
            else:
                if args.resume:
                    require(all(p.name in ('active.lock', 'record.log')
                        or (p.is_file() and p.name.startswith('failure_') and p.suffix == '.json')
                        for p in args.run_dir.iterdir()),
                        'Missing identity in a nonempty run; do not adopt existing checkpoints')
                base.write_json(args.run_dir/'identity.json', identity)
            # Persist even if reference preflight fails, so correcting an
            # evaluation path can safely --resume without deleting the run.
            base.write_json(args.run_dir/'budget.json', dict(target_completed_epochs=args.epochs, total_not_additional=True))
            reference, old_states = None, {}
            if not args.check:
                reference_path = args.run_dir/('reference_epoch_%03d.json' % args.epochs)
                evaluations = {key: getattr(args, 'gw2_'+key+'_eval') for key in ('best', 'last')}
                if reference_path.exists():
                    saved = read_json(reference_path)
                    evaluations = {key: saved['archives'][key]['directory'] for key in ('best', 'last')}
                reference, _, old_states = certify(args.gw2_run, args.epochs, target=identity, evaluations=evaluations)
                if reference_path.exists():
                    require(reference == saved, 'Saved GW3 reference changed')
                else:
                    base.write_json(reference_path, reference)
            states, batches = {}, data.loaders['train_loader'].num_batch
            for arm in ARMS:
                path = args.run_dir/arm/'last.pt'
                if path.is_file():
                    states[arm] = base.load_checkpoint(path)
                    validate_saved(states[arm], identity, arm, path.parent, batches)
                    require(states[arm]['completed_epoch'] <= args.epochs, 'Saved arm exceeds requested budget')
            for arm in args.only or ARMS:
                directory = args.run_dir/arm
                directory.mkdir(exist_ok=True)
                state = states.get(arm)
                if state and state['completed_epoch'] == args.epochs:
                    continue
                base.configure_runtime(args.device, args.threads, args.seed)
                data.loaders['train_loader'].samples = list(samples)
                model = model_for_arm(data, modules, args.device, arm)
                require(initialization(model) == initializations[arm], 'Actual GW3 initialization changed')
                engine = base.make_engine(model, data, modules, args.device, directory, logger, args.seed)
                states[arm] = base.run_epochs(engine, directory, identity, arm, args.epochs, state)
                base.write_json(args.run_dir/'summary.json', paired_summary(states, identity, args.epochs, reference, old_states))
                del engine, model
                gc.collect()
                if torch.device(args.device).type == 'cuda':
                    torch.cuda.empty_cache()
            if reference:
                fresh, _, _ = certify(args.gw2_run, args.epochs, target=identity,
                    evaluations={key: reference['archives'][key]['directory'] for key in ('best', 'last')})
                require(fresh == reference, 'Reference artifacts changed while training')
            summary = paired_summary(states, identity, args.epochs, reference, old_states)
            base.write_json(args.run_dir/'summary.json', summary)
            print(json.dumps(summary), flush=True)
            return summary
        except BaseException as error:
            base.write_json(args.run_dir/('failure_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')+'.json'),
                dict(status='GW3_TRAINING_FAILED', error_type=type(error).__name__, error=str(error),
                    resume_semantics='last_atomic_epoch_replays_unfinished_epoch', test_evaluated=False))
            raise
        finally:
            logger.removeHandler(handler)
            handler.close()


if __name__ == '__main__':
    main()
