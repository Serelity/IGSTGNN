"""Matched GW6 training from one certified, completed GW4 receiver checkpoint.

Joint is a new equal-budget continuation experiment, not a comparison of a
fully trained candidate to historical five-epoch controls. Probe freezes the
anchor and is a separate experiment identity. No test loader is evaluated.
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
from experiments.chronological.original_receiver_residual import ReceiverResidualIGSTGNN
from experiments.graph_memory_gw6.model import (
    ARMS, DEFAULT_ARMS, DESIGN, GraphMemoryIGSTGNN)
from experiments.graph_memory_gw6.reference import (
    certify, recertify, read_json, require, verify_identity_digest)


base = gw4.base
SCHEMA = 'original_igstgnn_graph_memory_gw6_training_v1'
EXECUTION_FILES = tuple(dict.fromkeys((*gw4.EXECUTION_FILES,
    'experiments/chronological/original_history_anchor.py',
    'experiments/chronological/train_original_history_anchor.py',
    'experiments/chronological/history_anchor_reference.py',
    'experiments/graph_memory_gw6/__init__.py',
    'experiments/graph_memory_gw6/model.py',
    'experiments/graph_memory_gw6/reference.py',
    'experiments/graph_memory_gw6/train.py')))


def model_for_arm(data, modules, device, arm, source_weights, stage='joint',
                  hidden=32, node_chunk=32):
    anchor = ReceiverResidualIGSTGNN(
        GraphMessageIGSTGNN(base.build_model(data, modules, device), 'POST_MESSAGE'),
        'RECEIVER_RESIDUAL')
    anchor.load_state_dict(source_weights, strict=True)
    return GraphMemoryIGSTGNN(anchor, arm, stage=stage, hidden=hidden,
                             node_chunk=node_chunk).to(device)


def anchor_state(model):
    return model.anchor_state() if hasattr(model, 'anchor_state') else model.anchor.state_dict()


def initialization(model):
    source = anchor_state(model)
    all_parameters = dict(model.named_parameters())
    added = {name:value for name,value in model.state_dict().items()
             if not name.startswith('anchor.')}
    return dict(source_tensor_sha256=base.tensor_tree_hash(source),
        added_tensor_sha256=base.tensor_tree_hash(added),
        post_construction_rng_sha256=base.tensor_tree_hash(base.rng_state()),
        registered_parameters=model.param_num(),
        trainable_parameters=sum(p.numel() for p in all_parameters.values() if p.requires_grad),
        frozen_parameters=sum(p.numel() for p in all_parameters.values() if not p.requires_grad),
        added_registered_parameters=sum(v.numel() for v in added.values()),
        architecture_parameter_counts=(model.trainable_parameter_counts()
            if hasattr(model,'trainable_parameter_counts') else None),
        capacity_caveat='report_actual_active_capacity_architecture_arms_need_not_be_parameter_equal')


def check_probe_frozen(model, identity):
    if identity['stage'] == 'probe':
        require(base.tensor_tree_hash(anchor_state(model)) ==
                identity['initializations'][model.arm]['source_tensor_sha256'],
                'Frozen GW4 probe anchor changed')


def validate_saved(state, identity, arm, directory, batches):
    require(identity.get('schema') == SCHEMA and identity.get('design') == DESIGN
            and arm in identity['arms'] and arm in ARMS, 'Expected a GW6 training identity')
    verify_identity_digest(identity)
    base.validate_epoch_state(state, identity, arm, directory, batches)
    if identity['stage'] == 'probe':
        source = {name[len('anchor.'):]:value for name,value in state['model_state'].items()
                  if name.startswith('anchor.')}
        require(base.tensor_tree_hash(source) == identity['initializations'][arm]['source_tensor_sha256'],
                'Saved probe anchor tensors changed')


def make_engine(model, data, modules, device, directory, logger, seed, identity):
    class MemoryEngine(modules.engine.IGSTGNN_Engine):
        def train_batch(self):
            values = super().train_batch()
            check_probe_frozen(self.model, identity)
            return values

        def evaluate(self, mode):
            require(mode == 'val', 'GW6 only evaluates validation')
            self.model.start_diagnostics()
            return super().evaluate(mode)

    active = [p for p in model.parameters() if p.requires_grad]
    require(bool(active), 'The declared arm has no trainable parameters')
    config, batches = data.config, data.loaders['train_loader'].num_batch
    optimizer = torch.optim.Adam(active, lr=config['lrate'],
                                 weight_decay=config['wdecay'], eps=1e-8)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=base.MILESTONES, gamma=.5)
    return MemoryEngine(device=torch.device(device), model=model, dataloader=data.loaders,
        scaler=data.scaler, sampler=None, loss_fn=modules.metrics.masked_mae,
        lrate=config['lrate'], optimizer=optimizer, scheduler=scheduler,
        clip_grad_value=config['clip_grad_value'], max_epochs=100,
        patience=config['patience'], log_dir=str(directory), logger=logger, seed=seed,
        cl_step=config['cl_epoch']*batches, warm_step=config['warm_epoch']*batches,
        horizon=12, incident=True)


def paired_summary(states, identity, epochs):
    arms = identity['arms']
    complete = set(states) == set(arms) and all(s['completed_epoch'] == epochs for s in states.values())
    comparisons = {}
    for control in arms:
        if control == 'JOINT_MEMORY' or control not in states or 'JOINT_MEMORY' not in states:
            continue
        left, right = states[control], states['JOINT_MEMORY']
        common = min(left['completed_epoch'], right['completed_epoch'])
        if not common:
            continue
        a, b = left['history'][:common], right['history'][:common]
        ba, bb = (min(rows, key=lambda r:r['validation_mae']) for rows in (a,b))
        comparisons['JOINT_MEMORY_vs_'+control] = dict(common_new_epochs=common,
            independent_best_at_common_budget={key+'_gain':ba['validation_'+key]-bb['validation_'+key]
                for key in ('mae','rmse','mape_fraction')},
            common_last_epoch={key+'_gain':a[-1]['validation_'+key]-b[-1]['validation_'+key]
                for key in ('mae','rmse','mape_fraction')})
    return dict(schema=SCHEMA, status='GW6_TRAINING_COMPLETE' if complete else 'GW6_TRAINING_PARTIAL',
        stage=identity['stage'], from_scratch=False, requested_new_epochs=epochs,
        same_completed_new_training_budget=complete, arms={arm:base.arm_summary(state, epochs, False)
            for arm,state in states.items()}, comparisons=comparisons,
        source_reference=identity['source_reference'], model_options=identity['model_options'],
        curriculum=identity['curriculum'], registered_arms=arms,
        primary_comparison='JOINT_MEMORY_vs_newly_trained_RECURRENT_CONTROL',
        matched_primary_controls_present=all(arm in arms for arm in DEFAULT_ARMS),
        historical_GW4_GW5_are_equal_budget_controls=False,
        predictive_accuracy_pass=False,
        interpretation='new_equal_budget_continuation_screen_not_independent_confirmation',
        optimizer_reset_at_start=True, optimizer_reset_on_resume=False,
        best_selection_includes_stage_zero=False, early_stopping_enabled=False,
        test_evaluated=False)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gw5-run', type=Path, required=True)
    parser.add_argument('--gw5-best-eval', type=Path)
    parser.add_argument('--gw5-last-eval', type=Path)
    parser.add_argument('--data-dir', type=Path, default=base.default_data_dir())
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=5, help='TOTAL new epochs per registered arm, 1..100')
    parser.add_argument('--stage', choices=('joint','probe'), default='joint')
    parser.add_argument('--curriculum', choices=('full','native'), default='full')
    parser.add_argument('--arms', nargs='+', choices=ARMS, default=list(DEFAULT_ARMS))
    parser.add_argument('--only', nargs='+', choices=ARMS)
    parser.add_argument('--hidden', type=int, default=32)
    parser.add_argument('--node-chunk', type=int, default=32)
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--threads', type=int, default=3)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    if not 1 <= args.epochs <= 100 or min(args.hidden,args.node_chunk,args.threads) < 1:
        parser.error('epochs must be 1..100; hidden, node-chunk and threads must be positive')
    if len(args.arms) != len(set(args.arms)) or (args.only and (
            len(args.only) != len(set(args.only)) or not set(args.only) <= set(args.arms))):
        parser.error('arms must be unique; only must be a unique subset of arms')
    if args.stage == 'probe' and 'RECURRENT_CONTROL' in args.arms:
        parser.error('Probe freezes the GW4 baseline; omit RECURRENT_CONTROL from --arms')
    args.gw5_run, args.run_dir, args.data_dir = (p.resolve() for p in (args.gw5_run,args.run_dir,args.data_dir))
    return args


def protected_run_paths(reference):
    result = {Path(reference['source_run']).resolve()}
    gw4ref = reference.get('gw4_reference', {})
    if gw4ref.get('source_run'):
        result.add(Path(gw4ref['source_run']).resolve())
    gw2ref = gw4ref.get('GW2_reference', {})
    if gw2ref.get('source_run'):
        result.add(Path(gw2ref['source_run']).resolve())
    return result


def main(argv=None):
    args = parse_args(argv)
    require(args.run_dir.exists() == args.resume, 'Use a fresh GW6 directory or --resume an existing one')
    require(args.run_dir != args.gw5_run and args.gw5_run not in args.run_dir.parents,
            'Place GW6 outside the immutable GW5 run')
    # Authenticate before creating or writing a run directory.
    identity_path = args.run_dir/'identity.json'
    if args.resume and identity_path.is_file():
        saved_identity = read_json(args.run_dir/'identity.json')
        require(saved_identity.get('schema') == SCHEMA, 'Cannot resume another experiment')
        reference = saved_identity['source_reference']
        require(reference['source_run'] == str(args.gw5_run), 'Cannot replace GW5 source')
        old_identity, weights, controls = recertify(reference)
    else:
        if args.resume:
            require(all(p.name in ('active.lock','record.log') or
                    (p.is_file() and p.name.startswith('failure_') and p.suffix == '.json')
                    for p in args.run_dir.iterdir()),
                    'Cannot recover a nonempty run without its identity')
        reference, old_identity, weights, controls = certify(args.gw5_run,
            {key:getattr(args,'gw5_'+key+'_eval') for key in ('best','last')})
    for protected in protected_run_paths(reference):
        require(args.run_dir != protected and protected not in args.run_dir.parents,
                'New output must be outside every historical source run')
    args.run_dir.mkdir(parents=True, exist_ok=args.resume)
    with base.run_lock(args.run_dir):
        logger = logging.getLogger('gw6-'+str(args.run_dir)); logger.setLevel(logging.INFO)
        handler = logging.FileHandler(args.run_dir/'record.log', encoding='utf-8'); logger.addHandler(handler)
        try:
            runtime = base.configure_runtime(args.device,args.threads,args.seed)
            modules = base.get_original_modules()
            require(modules.source_fingerprint == old_identity['source_fingerprint'], 'Original source changed')
            config = dict(base.CONFIG, seed=args.seed, device=args.device)
            if args.curriculum == 'full':
                config['warm_epoch'] = 101
            data = base.load_data(args.data_dir, config, modules, logger, check=False, training=True)
            require(data.identity == old_identity['data'], 'GW6 data or row partitions differ from GW5')
            samples = tuple(data.loaders['train_loader'].samples)
            options = dict(hidden=args.hidden,node_chunk=args.node_chunk)
            initializations = {}
            source_tensor_hash = base.tensor_tree_hash(weights)
            for arm in args.arms:
                base.configure_runtime(args.device,args.threads,args.seed)
                model = model_for_arm(data,modules,args.device,arm,weights,args.stage,**options)
                initializations[arm] = initialization(model)
                require(initializations[arm]['source_tensor_sha256'] == source_tensor_hash,
                        'Construction changed source GW4 weights: '+arm)
                require(initializations[arm]['trainable_parameters'] > 0, 'Arm has no active parameters')
                del model
            identity = dict(schema=SCHEMA, design=DESIGN, stage=args.stage, arms=args.arms,
                model_options=options, curriculum=args.curriculum, training_config=config,
                data=data.identity, runtime=runtime, source_fingerprint=modules.source_fingerprint,
                source_reference=reference, reference=old_identity['reference'],
                gw4_reference=old_identity['gw4_reference'], initializations=initializations,
                execution_source={name:base.sha256(REPO/name) for name in EXECUTION_FILES},
                initial_train_order_sha256=hashlib.sha256(np.arange(len(samples),dtype=np.int64).tobytes()).hexdigest(),
                scheduler=dict(milestones=base.MILESTONES,gamma=.5),
                engineering_only=False, from_scratch=False, test_evaluated=False,
                historical_budget_caveat='historical_GW4_GW5_are_not_equal_budget_new_training_controls')
            identity['identity_sha256'] = hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
            if identity_path.is_file():
                require(read_json(identity_path) == identity, 'GW6 identity changed')
                require(read_json(args.run_dir/'source_reference.json') == reference, 'GW6 source certificate changed')
                budget_path = args.run_dir/'budget.json'
                if budget_path.is_file():
                    require(args.epochs >= read_json(budget_path)['target_completed_epochs'],
                            'Cannot decrease saved new-epoch budget')
                else:
                    require(not any(args.run_dir.glob('*/last.pt')),
                            'Existing checkpoints require their saved budget file')
            else:
                base.write_json(args.run_dir/'source_reference.json',reference)
                base.write_json(identity_path,identity)
            base.write_json(args.run_dir/'budget.json',dict(target_completed_epochs=args.epochs,
                total_not_additional=True, maximum=100, test_evaluated=False))
            states, batches = {}, data.loaders['train_loader'].num_batch
            for arm in args.arms:
                path = args.run_dir/arm/'last.pt'
                if path.is_file():
                    states[arm] = base.load_checkpoint(path)
                    validate_saved(states[arm],identity,arm,path.parent,batches)
                    require(states[arm]['completed_epoch'] <= args.epochs, 'Saved arm exceeds budget')
            for arm in args.only or args.arms:
                directory = args.run_dir/arm; directory.mkdir(exist_ok=True)
                state = states.get(arm)
                if state and state['completed_epoch'] == args.epochs:
                    print(json.dumps(dict(arm=arm,status='ALREADY_COMPLETE')),flush=True)
                    continue
                base.configure_runtime(args.device,args.threads,args.seed)
                data.loaders['train_loader'].samples = list(samples)
                model = model_for_arm(data,modules,args.device,arm,weights,args.stage,**options)
                require(initialization(model) == initializations[arm], 'Actual arm initialization changed')
                engine = make_engine(model,data,modules,args.device,directory,logger,args.seed,identity)
                states[arm] = base.run_epochs(engine,directory,identity,arm,args.epochs,state)
                validate_saved(states[arm],identity,arm,directory,batches)
                base.write_json(args.run_dir/'summary.json',paired_summary(states,identity,args.epochs))
                del engine,model
                gc.collect()
                if torch.device(args.device).type == 'cuda':
                    torch.cuda.empty_cache()
            recertify(reference)
            summary = paired_summary(states,identity,args.epochs)
            base.write_json(args.run_dir/'summary.json',summary)
            print(json.dumps(summary),flush=True)
            return summary
        except BaseException as error:
            base.write_json(args.run_dir/('failure_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')+'.json'),
                dict(status='GW6_TRAINING_FAILED',error_type=type(error).__name__,error=str(error),
                    resume_semantics='last_atomic_epoch_incomplete_epoch_replayed',test_evaluated=False))
            raise
        finally:
            logger.removeHandler(handler); handler.close()


if __name__ == '__main__':
    main()
