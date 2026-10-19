"""Train only two small readouts above the immutable complete certified SUM.

Source requires_grad flags retain the certified inference dispatch contract.
Its forward and origin-feature extraction execute under no_grad in eval mode;
only the separate readout parameters enter Adam. Every epoch verifies source
tensors, buffers, flags and absence of gradients. No test loader is created.
"""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import logging
import os
from pathlib import Path
from time import perf_counter
import uuid

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from torch import nn

from experiments.state_gate_gw11 import train as gw11
from experiments.state_gate_gw11 import ablation_auth as auth
from experiments.state_gate_gw11.ablate_last import exact_replay, load_bound_checkpoint
from experiments.state_gate_gw11.reference import load_inputs, assert_unchanged as assert_reference
from experiments.state_gate_gw11.mask_audit import validate_audit
from experiments.frozen_readout_gw12.model import FrozenReadoutHead, origin_features
from experiments.frozen_readout_gw12.protocol import ARMS, EPOCHS, screen
from experiments.frozen_readout_gw12.state import SCHEMA, identity_digest, epoch_order, epoch_seed, validate_progress

base, epoch_tools, REPO = gw11.base, gw11.epoch_tools, gw11.REPO
CASES = ('FROZEN_SUM', *ARMS)
EXECUTION_FILES = tuple(dict.fromkeys((*gw11.EXECUTION_FILES,
    'experiments/state_gate_gw11/ablation_auth.py',
    'experiments/state_gate_gw11/ablate_last.py',
    'experiments/state_gate_gw11/ablation_metrics.py',
    'experiments/frozen_readout_gw12/__init__.py',
    'experiments/frozen_readout_gw12/model.py',
    'experiments/frozen_readout_gw12/protocol.py',
    'experiments/frozen_readout_gw12/state.py',
    'experiments/frozen_readout_gw12/probe.py')))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def source_hashes():
    return {name: base.sha256(REPO/name) for name in EXECUTION_FILES}


def head_initializations(heads):
    return {arm: dict(head_tensor_sha256=base.tensor_tree_hash(heads[arm].state_dict()),
        registered_parameters=sum(p.numel() for p in heads[arm].parameters()),
        trainable_parameters=sum(p.numel() for p in heads[arm].parameters() if p.requires_grad)) for arm in ARMS}


def source_flags(source):
    return {name: bool(value.requires_grad) for name, value in source.named_parameters()}


def check_source(source, expected_hash, expected_flags):
    require(not any(module.training for module in source.modules()), 'Complete SUM must stay in eval mode')
    require(source_flags(source) == expected_flags and all(expected_flags.values()),
        'Certified source inference flags changed')
    require(all(value.grad is None for value in source.parameters()), 'Frozen SUM acquired gradients')
    require(base.tensor_tree_hash(source.state_dict()) == expected_hash, 'Frozen SUM tensor or buffer changed')
    source.assert_cache_clear()


class Engine:
    def __init__(self, source, data, modules, device, baseline, train_cache, seed=2025):
        require(source.arm == 'SUM_CONTROL' and source.stage == 'joint' and source.head is not None,
            'GW12 requires the complete certified SUM including its fitted head')
        self.source, self.data, self.modules = source, data, modules
        self.device, self.baseline, self.train_cache, self.seed = torch.device(device), baseline, train_cache, seed
        self.source.eval()
        self.source_tensor_sha256, self.flags = base.tensor_tree_hash(source.state_dict()), source_flags(source)
        parameter = next(source.parameters())
        self.heads = nn.ModuleDict({arm: FrozenReadoutHead(arm).to(device=parameter.device, dtype=parameter.dtype)
            for arm in ARMS})
        config = data.config
        self.optimizers = {arm: torch.optim.Adam(self.heads[arm].parameters(),
            lr=config['lrate'], weight_decay=config['wdecay'], eps=1e-8) for arm in ARMS}
        self.schedulers = {arm: torch.optim.lr_scheduler.MultiStepLR(self.optimizers[arm],
            milestones=base.MILESTONES, gamma=.5) for arm in ARMS}
        source_ids = {id(value) for value in source.parameters()}
        require(all(not source_ids.intersection(id(value) for group in opt.param_groups for value in group['params'])
            for opt in self.optimizers.values()), 'An optimizer contains a source parameter')
        self.check_source()

    def check_source(self):
        check_source(self.source, self.source_tensor_sha256, self.flags)

    def frozen_forward(self, x, incident, sensors):
        with torch.no_grad():
            frozen = self.source(x, incident_data=incident, sensor_data=sensors)
            features = {arm: origin_features(x, self.source.original.node_emb_u,
                self.source.original.node_emb_d, arm) for arm in ARMS}
        require(not frozen.requires_grad and all(not value.requires_grad for value in features.values()),
            'Source output or origin features retained an autograd graph')
        return frozen, features

    def evaluate(self):
        self.check_source(); self.heads.eval(); self.source.start_diagnostics()
        for head in self.heads.values(): head.start_diagnostics()
        predictions = {case: [] for case in CASES}; truths, supports = [], []
        if self.device.type == 'cuda': torch.cuda.synchronize(self.device)
        start = perf_counter()
        with torch.no_grad():
            for batch, raw in enumerate(self.data.loaders['val_loader'].get_iterator(), 1):
                x, y, incident, sensors = gw11.device_batch(raw, self.device)
                frozen, features = self.frozen_forward(x, incident, sensors)
                values = {'FROZEN_SUM': frozen}
                values.update({arm: frozen+self.heads[arm](features[arm]) for arm in ARMS})
                for case, value in values.items():
                    predictions[case].append(self.data.scaler.inverse_transform(value).squeeze(-1).cpu().numpy())
                truths.append(self.data.scaler.inverse_transform(y).squeeze(-1).cpu().numpy())
                supports.append(incident['distances'].abs().sum(-1).cpu().numpy() > 0)
                if batch == 1 or batch == self.data.loaders['val_loader'].num_batch:
                    print(json.dumps(dict(phase='GW12_SHARED_FROZEN_VALIDATION', batch=batch,
                        total=self.data.loaders['val_loader'].num_batch)), flush=True)
        predictions = {case: np.concatenate(values) for case, values in predictions.items()}
        truth, support = np.concatenate(truths), np.concatenate(supports)
        exact_replay(truth, self.baseline['y'], 'GW12 validation target')
        require(np.array_equal(support, self.baseline['direct_support']), 'GW12 support axis changed')
        replay = exact_replay(predictions['FROZEN_SUM'], self.baseline['pred__SUM_CONTROL'], 'GW12 immutable SUM')
        require(all(value.dtype == np.float32 and value.shape == truth.shape and np.isfinite(value).all()
            for value in predictions.values()), 'Nonfinite or changed GW12 prediction axes')
        self.check_source()
        if self.device.type == 'cuda': torch.cuda.synchronize(self.device)
        details = dict(seconds=perf_counter()-start, complete_source_forwards=self.data.loaders['val_loader'].num_batch,
            shared_source_forward_between_heads=True, frozen_replay=replay,
            source=self.source.diagnostics(), heads={arm: self.heads[arm].diagnostics() for arm in ARMS},
            source_optimizer_updates=0, scope='shared_frozen_SUM_forward_two_heads_inverse_axis_checks')
        return {case: gw11.record(value, truth, self.baseline['valid']) for case, value in predictions.items()}, predictions, details

    def train_epoch(self, epoch):
        self.check_source(); self.heads.train(); self.source.start_diagnostics()
        for head in self.heads.values(): head.start_diagnostics()
        loader = self.data.loaders['train_loader']; original = tuple(loader.samples)
        order = epoch_order(self.seed, epoch, len(original))
        loader.samples = [original[int(index)] for index in order]
        base.configure_runtime(str(self.device), torch.get_num_threads(), epoch_seed(self.seed, epoch))
        losses = {arm: [] for arm in ARMS}; used_lr = {arm: self.optimizers[arm].param_groups[0]['lr'] for arm in ARMS}
        start, position, different_null = perf_counter(), 0, 0
        global_min = float(self.train_cache['y'].min())
        global_null = global_min if global_min < 1 else 0.
        if self.device.type == 'cuda': torch.cuda.synchronize(self.device); torch.cuda.reset_peak_memory_stats(self.device)
        try:
            for batch, raw in enumerate(loader.get_iterator(), 1):
                x, y, incident, sensors = gw11.device_batch(raw, self.device)
                frozen, features = self.frozen_forward(x, incident, sensors)
                target = self.data.scaler.inverse_transform(y)
                observed = target.squeeze(-1).detach().cpu().numpy(); ids = order[position:position+len(observed)]
                exact_replay(observed, self.train_cache['y'][ids], 'GW12 train target/order')
                batch_null = float(target.min()) if target.min() < 1 else 0.
                require(np.array_equal(observed != batch_null, self.train_cache['valid'][ids]),
                    'GW12 actual batch mask differs from audited cache; masks remain unchanged')
                different_null += int(batch_null != global_null); position += len(observed)
                for arm in ARMS:
                    opt = self.optimizers[arm]; opt.zero_grad(set_to_none=True)
                    prediction = self.data.scaler.inverse_transform(frozen+self.heads[arm](features[arm]))
                    require(bool(torch.isfinite(prediction).all()), 'Nonfinite GW12 training prediction')
                    loss = self.modules.metrics.masked_mae(prediction, target, target.new_tensor(batch_null))
                    require(bool(torch.isfinite(loss)), 'Nonfinite GW12 training MAE')
                    loss.backward()
                    require(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in self.heads[arm].parameters()),
                        'Nonfinite GW12 readout gradient')
                    if self.data.config['clip_grad_value']:
                        torch.nn.utils.clip_grad_norm_(self.heads[arm].parameters(), self.data.config['clip_grad_value'])
                    opt.step(); losses[arm].append(float(loss.detach()))
                require(all(p.grad is None for p in self.source.parameters()), 'Source received a training gradient')
                if batch == 1 or batch % 10 == 0 or batch == loader.num_batch:
                    print(json.dumps(dict(phase='GW12_PAIRED_HEAD_TRAINING', epoch=epoch,
                        batch=batch, total=loader.num_batch)), flush=True)
            require(position == len(original) and all(len(value) == 82 for value in losses.values()),
                'Incomplete fixed GW12 epoch')
            self.check_source()
            for scheduler in self.schedulers.values(): scheduler.step()
            if self.device.type == 'cuda': torch.cuda.synchronize(self.device)
            return dict(train_metrics={arm: dict(mean_batch_train_mae=float(np.mean(losses[arm])),
                    lr_used=float(used_lr[arm]), lr_next=float(self.optimizers[arm].param_groups[0]['lr']), updates=82)
                    for arm in ARMS},
                train_order_sha256=hashlib.sha256(order.astype('<i8').tobytes()).hexdigest(),
                dropout_epoch_seed=epoch_seed(self.seed, epoch), mask_difference_points=0,
                different_native_null_with_identical_mask_batches=different_null,
                timing=dict(seconds=perf_counter()-start, complete_source_forwards=82,
                    source_backward_calls=0, source_optimizer_updates=0, head_optimizer_steps=164,
                    peak_allocated_bytes=int(torch.cuda.max_memory_allocated(self.device)) if self.device.type == 'cuda' else None),
                mechanisms=dict(source=self.source.diagnostics(), heads={arm: self.heads[arm].diagnostics() for arm in ARMS}))
        finally:
            loader.samples = list(original)


def best_snapshot(head, directory, arm, epoch, row):
    name = '{}_best_epoch_{:03d}_{}.pt'.format(arm, epoch, uuid.uuid4().hex[:12])
    weights = gw11.cpu_tree(head.state_dict()); epoch_tools.atomic_save(directory/name, weights)
    return dict(epoch=epoch, mae=row['mae'], checkpoint_file=name,
        checkpoint_sha256=base.sha256(directory/name), head_tensor_sha256=base.tensor_tree_hash(weights))


def commit(engine, saved, directory, identity):
    engine.check_source()
    saved.update(schema=SCHEMA, identity_sha256=identity['identity_sha256'],
        source_tensor_sha256=engine.source_tensor_sha256, initializations=identity['initializations'],
        heads_state={arm: gw11.cpu_tree(engine.heads[arm].state_dict()) for arm in ARMS},
        optimizers_state={arm: gw11.cpu_tree(engine.optimizers[arm].state_dict()) for arm in ARMS},
        schedulers_state={arm: gw11.cpu_tree(engine.schedulers[arm].state_dict()) for arm in ARMS},
        rng_state=gw11.cpu_tree(epoch_tools.rng_state()))
    saved['heads_tensor_sha256'] = {arm: base.tensor_tree_hash(saved['heads_state'][arm]) for arm in ARMS}
    for field in ('optimizers_state', 'schedulers_state', 'rng_state'):
        saved[field+'_sha256'] = base.tensor_tree_hash(saved[field])
    epoch_tools.atomic_save(directory/'last.pt', saved)
    base.write_json(directory/'history.json', saved['history'])


def validate_saved(saved, identity, directory):
    validate_progress(saved, identity, 82)
    for arm in ARMS:
        require(base.tensor_tree_hash(saved['heads_state'][arm]) == saved['heads_tensor_sha256'][arm], 'Saved head tensors changed')
        require(saved['schedulers_state'][arm]['last_epoch'] == saved['completed_epoch'], 'Saved scheduler epoch changed')
        best = saved['best'][arm]; path = gw11.checkpoint_child(directory, best['checkpoint_file'])
        require(base.sha256(path) == best['checkpoint_sha256'], 'Saved BEST head bytes changed')
        require(base.tensor_tree_hash(base.load_checkpoint(path)) == best['head_tensor_sha256'], 'Saved BEST head tensors changed')
    for field in ('optimizers_state', 'schedulers_state', 'rng_state'):
        require(base.tensor_tree_hash(saved[field]) == saved[field+'_sha256'], 'Saved paired optimizer/scheduler/RNG changed')


def marker(directory, identity, completed):
    path = directory/'training_started.json'
    expected = dict(schema='gw12_paired_readouts_training_started_v1', identity_sha256=identity['identity_sha256'],
        written_before_first_update=True, source_updates=0)
    if path.exists():
        require(gw11.read_json(path) == expected, 'Training-start marker changed')
    else:
        require(completed == 0, 'A trained run lost its original training-start marker')
        temporary = directory/('.training_started_'+uuid.uuid4().hex+'.tmp')
        with temporary.open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(expected, stream, sort_keys=True, allow_nan=False); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        temporary.replace(path)


def summary(saved, identity):
    return dict(schema=SCHEMA, status='GW12_TRAINING_COMPLETE' if saved['completed_epoch'] == EPOCHS else 'GW12_TRAINING_PARTIAL',
        identity_sha256=identity['identity_sha256'], completed_epoch=saved['completed_epoch'], new_epochs=EPOCHS,
        head_updates=saved['head_updates'], total_head_optimizer_steps=sum(saved['head_updates'].values()),
        best=saved['best'], source_tensor_sha256=saved['source_tensor_sha256'], source_updates=0,
        FROZEN_SUM_is_equal_training_budget_control=False, optimizer_reset_on_resume=False,
        best_selection_includes_stage_zero=True, extension_allowed=False, test_evaluated=False)


def final_report(engine, saved, identity, directory, last_archive):
    require(saved['completed_epoch'] == EPOCHS, 'Complete the fixed paired three-epoch budget before final export')
    output = directory/('evaluation_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'_'+uuid.uuid4().hex[:8]); output.mkdir()
    records, provenance = {}, {}
    for selection in ('best', 'last'):
        child = output/selection; child.mkdir(); roles = {}
        for arm in ARMS:
            if selection == 'best':
                role = saved['best'][arm]; path = gw11.checkpoint_child(directory, role['checkpoint_file'])
                require(base.sha256(path) == role['checkpoint_sha256'], 'BEST bytes changed during export')
                weights = base.load_checkpoint(path)
            else:
                weights = saved['heads_state'][arm]
                role = dict(epoch=EPOCHS, checkpoint_file='last.pt', checkpoint_sha256=base.sha256(directory/'last.pt'),
                    head_tensor_sha256=saved['heads_tensor_sha256'][arm])
            require(base.tensor_tree_hash(weights) == role['head_tensor_sha256'], 'Selected head tensor hash differs')
            engine.heads[arm].load_state_dict(weights, strict=True); roles[arm] = dict(role, selection=selection)
        rows, predictions, details = engine.evaluate()
        for arm in ARMS:
            epoch = roles[arm]['epoch']
            expected = saved['stage_zero_validation'][arm] if epoch == 0 else saved['history'][epoch-1]['validation'][arm]
            require(gw11.same_record(rows[arm], expected), 'Selected head prediction failed metric replay')
            if epoch == 0: exact_replay(predictions[arm], predictions['FROZEN_SUM'], 'GW12 zero BEST '+arm)
        archive = {key: engine.baseline[key] for key in ('y', 'valid', 'sample_ids', 'node_ids', 'road_labels', 'direct_support')}
        archive.update(schema=np.array('gw12_frozen_readout_prediction_archive_v1'), axis_order=np.array('sample,horizon,node'),
            **{'pred__'+case: value for case, value in predictions.items()}, pred__OWN4=last_archive['pred__OWN4'])
        np.savez_compressed(child/'paired_predictions.npz', **archive)
        meta = dict(schema='gw12_frozen_readout_prediction_producer_v1', selection=selection,
            run_identity_sha256=identity['identity_sha256'], checkpoint_roles=roles, records=rows,
            frozen_source_tensor_sha256=engine.source_tensor_sha256, execution=details,
            archive_sha256=base.sha256(child/'paired_predictions.npz'), source_updates=0, test_evaluated=False)
        base.write_json(child/'metadata.json', meta); records[selection] = rows
        provenance[selection] = dict(directory=str(child), metadata_sha256=base.sha256(child/'metadata.json'), archive_sha256=meta['archive_sha256'])
    decision = screen(records['best'], records['last']); decision.update(run_identity_sha256=identity['identity_sha256'], evaluations=provenance)
    own4 = gw11.record(last_archive['pred__OWN4'], engine.baseline['y'], engine.baseline['valid'])
    decision['historical_OWN4_MAE_reference_only'] = own4['mae']
    decision['BEST_no_worse_than_historical_OWN4_report_only'] = records['best']['STATE_READOUT']['mae'] <= own4['mae']
    base.write_json(output/'screening.json', decision)
    lines = ['# GW12 fixed-SUM state information development probe', '',
        'Complete SUM backbone and fitted SUM head stayed in eval mode with zero updates.',
        'Two equal-capacity 725-parameter heads, three new epochs each. BEST includes stage zero.',
        'FROZEN_SUM and historical OWN4 are zero-update references, not trained equal-budget controls. No test evaluation.', '',
        '| Selection | Case | MAE | RMSE | MAPE % | Signed bias |', '|---|---|---:|---:|---:|---:|']
    for selection in ('best', 'last'):
        for case, row in {**records[selection], 'OWN4_HISTORY': own4}.items():
            lines.append('| {} | {} | {:.9f} | {:.9f} | {:.9f} | {:.9f} |'.format(selection, case, row['mae'], row['rmse'], 100*row['mape_fraction'], row['signed_bias']))
    lines += ['', '| Fixed check | Passed |', '|---|---|']
    lines += ['| '+key+' | '+str(value)+' |' for key, value in decision['checks'].items()]
    lines += ['', 'Passing is a development signal for state-conditioned readout utility under this fixed source.',
        'It does not establish novelty or superiority to historical methods.',
        'Independent confirmation is not authorized. The original failed GW11 screen remains unchanged.', '']
    (output/'report.md').write_text('\n'.join(lines), encoding='utf-8')
    return output, decision


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('gw11-run', 'data-dir', 'run-dir'): parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--device', default='cuda:0'); parser.add_argument('--threads', type=int, default=3)
    modes = parser.add_mutually_exclusive_group(); modes.add_argument('--resume', action='store_true'); modes.add_argument('--evaluate-only', action='store_true')
    args = parser.parse_args(argv)
    if args.threads < 1: parser.error('threads must be positive')
    for name in ('gw11_run', 'data_dir', 'run_dir'): setattr(args, name, getattr(args, name).resolve())
    return args


def main(argv=None):
    args = parse_args(argv); existing = args.resume or args.evaluate_only
    last_archive, gw11_identity, certificate = auth.load_run(args.gw11_run)
    paths = certificate['source_paths']
    baseline, old_certificate, cache, _, digests, gw10ref = load_inputs(Path(paths['probe_dir']), Path(paths['cache_dir']), Path(paths['gw10_dir']))
    require(old_certificate == gw11_identity['source_certificate'] and gw10ref == gw11_identity['gw10_reference'], 'Source provenance changed')
    gate = validate_audit(Path(paths['audit_dir']), old_certificate, digests, gw10ref)
    require(gate == gw11_identity['mask_gate'], 'Existing training mask audit changed')
    # Source/ancestor protection applies even when resuming, before a lock can be written.
    for protected in {args.gw11_run, args.data_dir, *(Path(value).resolve() for value in certificate['protected_directories'])}:
        require(args.run_dir != protected and protected not in args.run_dir.parents and args.run_dir not in protected.parents,
            'GW12 run overlaps an immutable input: '+str(protected))
    if existing:
        require((args.run_dir/'identity.json').is_file() and (args.run_dir/'last.pt').is_file(),
            'Resume needs the original identity and complete paired last.pt; keep incomplete initialization and use a new run')
    else:
        auth.require_output_outside_sources(args.run_dir, args.gw11_run, certificate, args.data_dir)
        args.run_dir.mkdir(parents=True, exist_ok=False)
    with base.run_lock(args.run_dir):
        try:
            source_identity, gw4, recurrent = gw11.recertify(old_certificate['source_reference'])
            runtime = base.configure_runtime(args.device, args.threads, 2025)
            require(runtime == gw11_identity['runtime'], 'Use the certified complete runtime configuration')
            modules = base.get_original_modules()
            require(modules.source_fingerprint == source_identity['source_fingerprint'], 'Original source changed')
            config = dict(base.CONFIG, device=args.device, seed=2025, warm_epoch=101)
            data = base.load_data(args.data_dir, config, modules, logging.getLogger('gw12'), check=False, training=True)
            require(data.identity == gw11_identity['data'] == old_certificate['data_identity'], 'Original data identity changed')
            require(np.array_equal(data.station_ids, baseline['node_ids']), 'Node order changed')
            old_state = load_bound_checkpoint(Path(paths['probe_dir'])/'last.pt', old_certificate)
            old_probe_identity = gw11.read_json(Path(paths['probe_dir'])/'identity.json')
            gw11.gw9_probe.validate_saved(old_state, old_probe_identity, Path(paths['probe_dir']), 82)
            head, role = gw11.gw9_probe.selected_role(old_state, Path(paths['probe_dir']), 'SUM_CONTROL', 'best')
            require(role == old_certificate['head_roles']['best']['SUM_CONTROL'], 'Original SUM BEST role changed')
            source = gw11.gw9.model_for_arm(data, modules, args.device, 'SUM_CONTROL', gw4, recurrent, hidden=16, stage='joint')
            source.head.load_state_dict(head, strict=True)
            del old_state, head
            engine = Engine(source, data, modules, args.device, baseline, cache['train'])
            require(engine.source_tensor_sha256 == gw11_identity['initializations']['SUM_CONTINUE']['source_tensor_sha256'],
                'GW12 must retain exactly the certified complete SUM starting tensors')
            initial = head_initializations(engine.heads)
            require(all(row['registered_parameters'] == row['trainable_parameters'] == 725 for row in initial.values())
                and len({row['head_tensor_sha256'] for row in initial.values()}) == 1, 'Added head capacity/initial tensors differ')
            identity = dict(schema=SCHEMA, run_directory=str(args.run_dir), arms=list(ARMS), candidate='STATE_READOUT', new_epochs=EPOCHS,
                seed=2025, train_count=3909, train_batches=82, full_horizon=12, initializations=initial,
                source_tensor_sha256=engine.source_tensor_sha256, source_flags=engine.flags,
                source_certificate=old_certificate, motivating_GW11_certificate=certificate, mask_gate=gate,
                config=config, data=data.identity, runtime=runtime, execution_source=source_hashes(),
                source_update_policy='eval_no_grad_excluded_from_all_optimizers_flags_retained_for_exact_replay',
                source_updates=0, frozen_reference_is_equal_budget_control=False,
                optimizer_reset_at_start=True, optimizer_reset_on_resume=False, best_selection_includes_stage_zero=True,
                extension_allowed=False, test_evaluated=False)
            identity['identity_sha256'] = identity_digest(identity)
            if existing:
                require(gw11.read_json(args.run_dir/'identity.json') == identity, 'Cannot resume changed source/protocol/runtime')
                saved = base.load_checkpoint(args.run_dir/'last.pt'); validate_saved(saved, identity, args.run_dir)
                marker(args.run_dir, identity, saved['completed_epoch'])
                for arm in ARMS:
                    engine.heads[arm].load_state_dict(saved['heads_state'][arm], strict=True)
                    engine.optimizers[arm].load_state_dict(saved['optimizers_state'][arm])
                    engine.schedulers[arm].load_state_dict(saved['schedulers_state'][arm])
                epoch_tools.restore_rng(saved['rng_state'])
            else:
                base.write_json(args.run_dir/'identity.json', identity)
                rows, predictions, details = engine.evaluate()
                for arm in ARMS: exact_replay(predictions[arm], predictions['FROZEN_SUM'], 'GW12 stage zero '+arm)
                saved = dict(schema=SCHEMA, completed_epoch=0, global_updates=0, head_updates={arm: 0 for arm in ARMS},
                    stage_zero_validation=rows, stage_zero_execution=details, history=[],
                    best={arm: best_snapshot(engine.heads[arm], args.run_dir, arm, 0, rows[arm]) for arm in ARMS})
                commit(engine, saved, args.run_dir, identity); marker(args.run_dir, identity, 0)
                base.write_json(args.run_dir/'summary.json', summary(saved, identity))
            if not args.evaluate_only:
                for epoch in range(saved['completed_epoch']+1, EPOCHS+1):
                    trained = engine.train_epoch(epoch); rows, _, details = engine.evaluate()
                    saved['history'].append(dict(epoch=epoch, validation=rows, validation_execution=details,
                        global_updates=epoch*82, head_updates={arm: epoch*82 for arm in ARMS}, **trained))
                    saved.update(completed_epoch=epoch, global_updates=epoch*82, head_updates={arm: epoch*82 for arm in ARMS})
                    for arm in ARMS:
                        if rows[arm]['mae'] < saved['best'][arm]['mae']:
                            saved['best'][arm] = best_snapshot(engine.heads[arm], args.run_dir, arm, epoch, rows[arm])
                    commit(engine, saved, args.run_dir, identity)
                    base.write_json(args.run_dir/'summary.json', summary(saved, identity))
                    print(json.dumps(dict(status='GW12_PAIRED_EPOCH_COMMITTED', epoch=epoch,
                        head_updates=saved['head_updates'], validation=rows, source_updates=0)), flush=True)
            validate_saved(saved, identity, args.run_dir)
            output, decision = final_report(engine, saved, identity, args.run_dir, last_archive)
            engine.check_source(); auth.assert_unchanged(certificate); gw11.assert_source_unchanged(old_certificate); assert_reference(gw10ref)
            require(validate_audit(Path(paths['audit_dir']), old_certificate, digests, gw10ref) == gate, 'Mask gate changed')
            require(source_hashes() == identity['execution_source'], 'GW12 execution source changed during run')
            base.write_json(args.run_dir/'screening.json', decision)
            base.write_json(args.run_dir/'summary.json', summary(saved, identity))
            base.write_json(args.run_dir/'result_pointer.json', dict(directory=str(output),
                report_sha256=base.sha256(output/'report.md'), screening_sha256=base.sha256(output/'screening.json')))
            (args.run_dir/'report.md').write_text((output/'report.md').read_text(encoding='utf-8'), encoding='utf-8')
            print(json.dumps(dict(status='GW12_EVALUATION_COMPLETE', output_dir=str(output), source_updates=0, test_evaluated=False)), flush=True)
            return decision
        except BaseException as error:
            base.write_json(args.run_dir/('failure_'+uuid.uuid4().hex[:12]+'.json'),
                dict(status='GW12_FAILED', error_type=type(error).__name__, error=str(error), test_evaluated=False))
            raise


if __name__ == '__main__':
    main()
