"""IS1.1 paired B0/B1/B2/B3 training; 1 -> 10 -> 30 absolute epochs.

Each arm is a separate process and resumes only from its own epoch checkpoint.
No test split, external forecast correction, auxiliary loss or new traffic channel.
"""
import argparse
import copy
from contextlib import contextmanager
import csv
from datetime import datetime
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

import numpy as np
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.chronological.smoke import device_batch, make_model, set_seed, sha256, verify_package
from experiments.chronological.train import (
    MetricTotals, batches, cpu_state, epoch_plan, save_checkpoint, state_sha256, validate_protocol,
)
from src.models.incident_capacity_exchange import ExchangeGraph
from src.models.incident_spatial import MODES, STAT_NAMES, configure_spatial_mode, require, state_capacity
from src.utils.chronological import ChronologicalDataset, masked_flow_mae

HERE = Path(__file__).resolve().parent
ARMS = dict(zip(('B0', 'B1', 'B2', 'B3'), MODES))


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def default_data():
    candidates = [REPO/'../data/chronological/Contra_Costa_v8_dev',
                  REPO/'../论文学习/研究开发_20260911/data_v8_contra']
    return next((p.resolve() for p in candidates if (p/'context_manifest.json').is_file()), candidates[0].resolve())


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, default=default_data())
    p.add_argument('--run-dir', type=Path)
    p.add_argument('--protocol', type=Path, default=HERE/'internal_spatial_is11.json')
    p.add_argument('--graph-dir', type=Path, default=HERE/'internal_spatial_graph_is11')
    p.add_argument('--epochs', type=int, default=1, help='Absolute completed epoch, including resumed epochs')
    p.add_argument('--seed', type=int, default=2025)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--threads', type=int, default=3)
    p.add_argument('--check', action='store_true', help='First four train/val rows, batch two; engineering only')
    p.add_argument('--worker', choices=ARMS, help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    require(args.epochs >= 1 and args.threads >= 1 and 0 <= args.seed < 2**32, 'Invalid epoch/threads/seed')
    require(not args.check or args.epochs == 1, 'Engineering check is exactly one small epoch')
    if args.run_dir is None:
        args.run_dir = REPO/'experiments/chronological_runs'/(
            'contra_is11_'+('check_' if args.check else '')+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6])
    for name in ('data_dir', 'run_dir', 'protocol', 'graph_dir'):
        setattr(args, name, getattr(args, name).resolve())
    return args


def load_graph(directory, station_ids, expected_hash):
    source = read_json(directory/'source_summary.json')
    require(sha256(directory/'graph.json') == expected_hash == source['outputs_sha256']['graph.json'],
            'Frozen graph fingerprint mismatch')
    require(sha256(directory/'stations.csv') == source['outputs_sha256']['stations.csv'], 'Station metadata fingerprint mismatch')
    meta = read_json(directory/'graph.json')
    require(np.array_equal(meta['station_ids'], station_ids), 'Graph station axis differs from traffic')
    pairs = [(int(e['source']), int(e['destination'])) for e in meta['edges']]
    require(len(set(pairs)) == len(pairs), 'Duplicate candidate edge, including external edges')
    for edge in meta['edges']:
        src, dst = edge['source'], edge['destination']
        require(edge['kind'] == ('inlet' if src < 0 else 'exit' if dst < 0 else 'internal'), 'Edge type mismatch')
        if src < 0 or dst < 0:
            require(bool(edge.get('boundary_basis')), 'Boundary lacks source evidence')
    graph = ExchangeGraph(len(station_ids), torch.tensor(pairs, dtype=torch.long).T.contiguous(),
                          torch.tensor(meta['operator_mask'], dtype=torch.bool),
                          torch.tensor(meta['boundary_nodes'], dtype=torch.bool), evidence_scope='candidate_unverified')
    with (directory/'stations.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    require([int(r['station_id']) for r in rows] == list(station_ids), 'Road group station axis mismatch')
    groups = {}
    for i, row in enumerate(rows):
        require(int(row['node_index']) == i, 'Road group node index mismatch')
        groups.setdefault(row['road'], []).append(i)
    return graph, groups, dict(nodes=graph.nodes, operator_nodes=int(graph.operator_mask.sum()), edges=graph.edges,
                               internal_edges=sum(s >= 0 and d >= 0 for s, d in pairs),
                               boundary_edges=sum(s < 0 or d < 0 for s, d in pairs), road_direction_groups=len(groups),
                               evidence_scope=graph.evidence_scope, jointly_certified_nodes=source['jointly_certified_nodes'])


def source_fingerprints():
    names = ('src/models/igstgnn.py', 'src/models/incident_spatial.py', 'src/models/incident_response.py',
             'src/models/incident_strength_gate.py', 'src/models/incident_capacity_exchange.py',
             'src/models/incident_relative_capacity.py', 'src/base/model.py', 'src/utils/chronological.py',
             'experiments/chronological/smoke.py', 'experiments/chronological/train.py',
             'experiments/chronological/train_internal_spatial.py')
    return {name: sha256(REPO/name) for name in names}


@contextmanager
def run_lock(directory):
    """OS releases the advisory lock on process death, so no stale-lock deletion is needed."""
    stream = (directory/'active.lock').open('a+b')
    if stream.tell() == 0:
        stream.write(b'0')
        stream.flush()
    stream.seek(0)
    try:
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        stream.close()
        raise ValueError('Another launcher owns this run; use its existing job or a separate run directory')
    try:
        yield
    finally:
        stream.close()


def build_arm(args, arm, nodes, graph, device):
    set_seed(args.seed)
    model = make_model(args.data_dir, nodes, device, 'fixed')
    return configure_spatial_mode(model, ARMS[arm], copy.deepcopy(graph).to(device) if arm in ('B2', 'B3') else None)


def parameter_manifest(model):
    state = {k: p.detach().cpu() for k, p in model.named_parameters()}
    common = {k: v for k, v in state.items() if not k.startswith(('report_spatial_state.', 'icsf_module.q_proj.',
              'icsf_module.v_proj.', 'icsf_module.icsf_fusion_mlp.', 'icsf_module.strength_gate.'))
              and '.incident_exchange.' not in k}
    extras = {k: v for k, v in state.items() if k.startswith(('report_spatial_state.', 'icsf_module.strength_gate.'))
              or '.incident_exchange.' in k}
    return dict(total_parameters=sum(p.numel() for p in model.parameters()),
                trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                new_parameters=sum(p.numel() for p in extras.values()),
                original_retained_parameters=sum(p.numel() for p in state.values())-sum(p.numel() for p in extras.values()),
                initial_parameters_sha256=state_sha256(state), common_initial_sha256=state_sha256(common),
                specification={k: list(p.shape) for k, p in model.named_parameters()})


def environment(device):
    return dict(python=sys.version, torch=str(torch.__version__), cuda=torch.version.cuda,
                device=str(device), gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled(), tf32=False,
                cublas_workspace_config=os.environ['CUBLAS_WORKSPACE_CONFIG'])


def make_identity(args, protocol, hashes, graph_info):
    return dict(schema='internal_spatial_is11_run_v1', protocol=protocol, protocol_sha256=sha256(args.protocol),
                package=hashes, graph=graph_info, graph_sha256=sha256(args.graph_dir/'graph.json'),
                graph_inputs_sha256={name: sha256(args.graph_dir/name)
                                     for name in ('graph.json', 'stations.csv', 'source_summary.json')},
                seed=args.seed, engineering_check=args.check, batch_size=2 if args.check else protocol['batch_size'],
                threads=args.threads, source_sha256=source_fingerprints(),
                runtime=dict(torch=str(torch.__version__), cuda=torch.version.cuda, device_type=torch.device(args.device).type,
                             cublas_workspace_config=os.environ['CUBLAS_WORKSPACE_CONFIG']),
                test_read=False)


def restore(path, identity, arm, model, optimizer, scheduler):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    require(saved.get('identity') == identity and saved.get('arm') == arm, 'Checkpoint identity mismatch')
    # Identity is checked before changing any model/optimizer state. GPU name is a separate audit field.
    model.load_state_dict(saved['model_state'], strict=True)
    optimizer.load_state_dict(saved['optimizer_state'])
    scheduler.load_state_dict(saved['scheduler_state'])
    return saved


def optimizer_for(model, protocol):
    return torch.optim.Adam(model.parameters(), lr=protocol['learning_rate'],
                            weight_decay=protocol['weight_decay'], eps=protocol['adam_eps'])


class Diagnostics:
    """Only detached small statistics, indexed by layer/call/substep; no activation cache."""
    def __init__(self):
        self.values, self.calls = None, 0
        self.state = {}

    def update(self, context):
        if context is None:
            return
        values = torch.stack(context['records']).detach().double().cpu().numpy()
        require(values.shape[0] == 20 and values.shape[1:] == (2, len(STAT_NAMES)), 'Incomplete five-layer spatial coverage')
        if self.values is None:
            self.values = values.copy()
        else:
            require(values.shape == self.values.shape, 'Spatial call axes changed')
            for i, name in enumerate(STAT_NAMES):
                if name.endswith('_max') or name == 'state_max':
                    self.values[..., i] = np.maximum(self.values[..., i], values[..., i])
                elif name == 'state_min':
                    self.values[..., i] = np.minimum(self.values[..., i], values[..., i])
                else:
                    self.values[..., i] += values[..., i]
        self.calls += 1
        capacity_difference = (context['capacity'].detach()-state_capacity(context['prior'].detach(), context['graph'])).abs()
        for name, tensor in (('adoption', context['adoption']), ('report_update_abs', (context['posterior']-context['prior']).abs()),
                             ('capacity', context['capacity']), ('report_capacity_abs_change', capacity_difference)):
            a = tensor.detach().double()
            selected = a[context['report_mask'].expand_as(a).bool()] if name in ('adoption', 'report_update_abs') else a.flatten()
            if selected.numel():
                old = self.state.setdefault(name, dict(sum=0., count=0, minimum=float('inf'), maximum=-float('inf')))
                old['sum'] += float(selected.sum())
                old['count'] += selected.numel()
                old['minimum'], old['maximum'] = min(old['minimum'], float(selected.min())), max(old['maximum'], float(selected.max()))

    def result(self):
        if self.values is None:
            return None
        rows = []
        for i, call in enumerate(self.values):
            for step, values in enumerate(call):
                row = dict(zip(STAT_NAMES, map(float, values)))
                row.update(layer=i//4, call_in_layer=i%4, substep=step,
                           clip_fraction=row['clip_count']/max(row['positions'], 1),
                           potential_capacity_limited_fraction=row['capacity_limited_count']/max(row['edge_positions'], 1))
                rows.append(row)
        state = {k: dict(count=v['count'], mean=v['sum']/v['count'], minimum=v['minimum'], maximum=v['maximum'])
                 for k, v in self.state.items()}
        return dict(batches=self.calls, spatial_call_count_per_forward=20, rows=rows, report_state=state,
                    budget_counts_semantics='B2 hypothetical restrictions; B3 applied restrictions',
                    flux_units='dimensionless_per_call_latent_channel',
                    b2_stability=dict(clip_fraction=sum(r['clip_count'] for r in rows)/max(sum(r['positions'] for r in rows), 1),
                                      clip_abs_max=max(r['clip_abs_max'] for r in rows)),
                    inference_gain_claim=False)


def component_reach(graph):
    # Fixed undirected components give a structural potential envelope, not a causal response claim.
    neighbors = [set([i]) for i in range(graph.nodes)]
    for s, d in graph.edge_index.T.tolist():
        if s >= 0 and d >= 0:
            neighbors[s].add(d)
            neighbors[d].add(s)
    reach = torch.zeros(graph.nodes, graph.nodes)
    for i in range(graph.nodes):
        seen, pending = set(), [i]
        while pending:
            node = pending.pop()
            if node not in seen:
                seen.add(node)
                pending.extend(neighbors[node]-seen)
        if graph.operator_mask[i]:
            reach[i, list(seen)] = 1.
    return reach


def metric_result(totals):
    return totals.result() if (totals.counts > 0).all() else dict(status='unavailable', counts=totals.counts.tolist())


def gradient_summary(model):
    values = {}
    for name, parameter in model.named_parameters():
        if name.startswith(('report_spatial_state.', 'icsf_module.strength_gate.')) or '.incident_exchange.' in name:
            values[name] = None if parameter.grad is None else float(parameter.grad.detach().abs().sum())
            if name.endswith('incident_exchange.readout.weight') and parameter.grad is not None:
                for i, block in enumerate(('displacement', 'inflow', 'outflow')):
                    values[name+':'+block] = float(parameter.grad[:, i*4:(i+1)*4].detach().abs().sum())
    return values


def run_epoch(model, optimizer, dataset, indices, batch_size, args, epoch, protocol, graph, roads, training):
    model.train(training)
    all_metrics = {'all_nodes': MetricTotals()}
    if not training:
        all_metrics.update({key: MetricTotals() for key in ('report_associated', 'candidate_potential', 'outside_operator')})
        all_metrics.update({'road:'+k: MetricTotals() for k in roads})
    diagnostics, gradients, windows = Diagnostics(), {}, []
    device = torch.device(args.device)
    reach, operator = component_reach(graph).to(device), graph.operator_mask.to(device)
    for batch_index, batch in enumerate(batches(dataset, indices, batch_size)):
        batch = device_batch(batch, device)
        if training:
            # Identical per-batch RNG in all arms and on resume; candidates add no dropout calls.
            set_seed((args.seed*1_000_003+epoch*10_007+batch_index) % 2**32)
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            result = model(batch['x'], incident_data=batch['incident'], return_spatial_details=True)
            prediction = result['prediction']*dataset.scaler['std']+dataset.scaler['mean']
            loss = masked_flow_mae(prediction, batch['y_flow'], batch['y_valid'])
            if training:
                loss.backward()
                active = [p.grad for p in model.parameters() if p.grad is not None]
                require(active and bool(torch.isfinite(torch.stack([g.detach().abs().max() for g in active])).all()),
                        'Non-finite training gradients')
                if batch_index == 0:
                    gradients = gradient_summary(model)
                torch.nn.utils.clip_grad_norm_(model.parameters(), protocol['clip_grad_norm'], error_if_nonfinite=True)
                optimizer.step()
        diagnostics.update(result['spatial'])
        prediction = prediction.detach()
        target, valid = batch['y_flow'], batch['y_valid']
        all_metrics['all_nodes'].update(prediction, target, valid)
        if not training:
            associated = batch['incident']['distances'].abs().sum(-1) > 0
            masks = dict(report_associated=associated, candidate_potential=(associated.float() @ reach > 0),
                         outside_operator=(~operator)[None].expand(associated.shape[0], -1))
            for key, nodes in roads.items():
                mask = torch.zeros_like(associated)
                mask[:, nodes] = True
                masks['road:'+key] = mask
            for key, mask in masks.items():
                all_metrics[key].update(prediction, target, valid & mask[:, None, :, None])
            error = torch.where(valid, (prediction.double()-target.double()).abs(), 0.)
            windows.append((error.sum((2, 3)).cpu().numpy(), valid.sum((2, 3)).cpu().numpy()))
        del result, prediction, loss, batch
    return dict(metrics={k: metric_result(v) for k, v in all_metrics.items()}, mechanisms=diagnostics.result(),
                first_batch_gradient_l1=gradients), windows


def preflight(args, train, graph):
    manifests, native_prediction = {}, None
    batch = device_batch(next(batches(train, [int(r['sample_index']) for r in train.rows[:2]], 2)), 'cpu')
    for arm in ARMS:
        model = build_arm(args, arm, len(train.station_ids), graph, torch.device('cpu')).eval()
        manifests[arm] = parameter_manifest(model)
        if arm in ('B0', 'B1'):
            with torch.inference_mode():
                pred = model(batch['x'], incident_data=batch['incident'])
            if native_prediction is None:
                native_prediction = pred
            else:
                require(torch.equal(native_prediction, pred), 'B0/B1 initial predictions differ')
        del model
    require(len({v['common_initial_sha256'] for v in manifests.values()}) == 1, 'Common initialization mismatch')
    require(manifests['B2']['initial_parameters_sha256'] == manifests['B3']['initial_parameters_sha256']
            and manifests['B2']['specification'] == manifests['B3']['specification'], 'B2/B3 parameter pairing mismatch')
    require(all(v['total_parameters'] == v['trainable_parameters'] for v in manifests.values()), 'Frozen active parameters')
    return dict(status='IS11_INITIALIZATION_PAIR_PASS', arms=manifests, b0_b1_prediction_abs_max=0.)


def worker(args, identity, protocol, train, val, graph, roads):
    arm = args.worker
    root = args.run_dir/arm
    root.mkdir(exist_ok=True)
    device = torch.device(args.device)
    model = build_arm(args, arm, len(train.station_ids), graph, device)
    optimizer = optimizer_for(model, protocol)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, protocol['lr_milestones'], gamma=protocol['lr_gamma'])
    history, best, wait, updates, best_state = [], float('inf'), 0, 0, None
    if (root/'last.pt').is_file():
        saved = restore(root/'last.pt', identity, arm, model, optimizer, scheduler)
        history, best, wait, updates = saved['history'], saved['best_metric'], saved['wait'], saved['global_updates']
        best_state = saved['best_model_state']
    completed = len(history)
    if completed >= args.epochs or wait >= protocol['patience']:
        print(json.dumps(dict(arm=arm, status='ALREADY_COMPLETE_FOR_REQUEST', completed_epoch=completed)), flush=True)
        return
    parameters = parameter_manifest(model)
    parameters.pop('initial_parameters_sha256')
    parameters.pop('common_initial_sha256')
    parameters['optimizer_parameters'] = sum(p.numel() for group in optimizer.param_groups for p in group['params'])
    train_indices = [int(r['sample_index']) for r in train.rows[:4 if args.check else len(train)]]
    val_indices = [int(r['sample_index']) for r in val.rows[:4 if args.check else len(val)]]
    for epoch in range(completed+1, min(args.epochs, protocol['max_epochs'])+1):
        start = time.perf_counter()
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        plan = epoch_plan(train_indices, identity['batch_size'], args.seed, epoch)
        train_report, _ = run_epoch(model, optimizer, train, plan['order'], identity['batch_size'], args, epoch,
                                    protocol, graph, roads, True)
        val_report, windows = run_epoch(model, optimizer, val, val_indices, identity['batch_size'], args, epoch,
                                        protocol, graph, roads, False)
        metric = val_report['metrics']['all_nodes']['mae_macro']
        updates += len(plan['batches'])
        improved = metric < best
        if improved:
            best, wait, best_state = metric, 0, cpu_state(model.state_dict())
        else:
            wait += 1
        lr_used = optimizer.param_groups[0]['lr']
        scheduler.step()
        row = dict(epoch=epoch, train=train_report, validation=val_report, validation_mae_macro=metric,
                   best_metric=best, best_epoch=epoch if improved else history[-1]['best_epoch'], improved=improved,
                   global_updates=updates, epoch_updates=len(plan['batches']), learning_rate_used=lr_used,
                   next_learning_rate=optimizer.param_groups[0]['lr'], order_sha256=plan['order_sha256'],
                   tail_batch_size=plan['tail_batch_size'], seconds=time.perf_counter()-start,
                   peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None)
        history.append(row)
        # Save window errors BEFORE checkpoint; interrupted writes cannot claim a complete epoch.
        winpath = root/f'validation_windows_epoch_{epoch:03d}.npz'
        with winpath.with_suffix('.npz.partial').open('wb') as stream:
            np.savez_compressed(stream, sample_indices=val_indices,
                                absolute=np.concatenate([v[0] for v in windows]), counts=np.concatenate([v[1] for v in windows]),
                                t0=np.array([r['t0'] for r in val.rows[:len(val_indices)]]))
        winpath.with_suffix('.npz.partial').replace(winpath)
        payload = dict(identity=identity, arm=arm, completed_epoch=epoch, model_state=cpu_state(model.state_dict()),
                       optimizer_state=optimizer.state_dict(), scheduler_state=scheduler.state_dict(), best_metric=best,
                       best_model_state=best_state, wait=wait, global_updates=updates, history=history,
                       rng_policy='per_epoch_batch_seed_v1', torch_rng=torch.get_rng_state(),
                       cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else [])
        save_checkpoint(root/'last.pt', payload)
        write_json(root/'history.json', history)
        write_json(root/'summary.json', dict(status='ENGINEERING_CHECK_PASS' if args.check else 'PAUSED_AT_EPOCH_BOUNDARY',
                   arm=arm, mode=ARMS[arm], completed_epoch=epoch, best_epoch=row['best_epoch'], best_metric=best,
                   global_updates=updates, early_stopped=wait >= protocol['patience'], parameters=parameters,
                   environment=environment(device), checkpoint='last.pt', best_model_state_key='best_model_state',
                   test_read=False, predictive_gain_claim=False))
        print(json.dumps(dict(arm=arm, epoch=epoch, train_mae_macro=train_report['metrics']['all_nodes']['mae_macro'],
                              validation_mae_macro=metric, best_metric=best, seconds=row['seconds'])), flush=True)
        del payload, train_report, val_report, windows
        gc.collect()
        if wait >= protocol['patience']:
            break


def paired_report(run_dir):
    summaries = {arm: read_json(run_dir/arm/'summary.json') for arm in ARMS}
    rows = {arm: read_json(run_dir/arm/'history.json') for arm in ARMS}
    common_epoch = min(s['completed_epoch'] for s in summaries.values())
    comparisons = {}
    # Emit B2 numerical stability before interpreting any candidate ranking.
    stability = {phase: rows['B2'][common_epoch-1][phase]['mechanisms']['b2_stability']
                 for phase in ('train', 'validation')}
    for candidate, control in (('B1', 'B0'), ('B2', 'B1'), ('B3', 'B2'), ('B3', 'B0')):
        comparisons[candidate+'_vs_'+control] = dict(
            best_mae_gain=summaries[control]['best_metric']-summaries[candidate]['best_metric'],
            common_epoch_mae_gain=rows[control][common_epoch-1]['validation_mae_macro']-rows[candidate][common_epoch-1]['validation_mae_macro'],
            candidate_last_five=[r['validation_mae_macro'] for r in rows[candidate][-5:]],
            control_last_five=[r['validation_mae_macro'] for r in rows[control][-5:]])
    return dict(status='IS11_PAIRED_STAGE_COMPLETE', b2_stability_before_ranking=stability, common_epoch=common_epoch,
                arms=summaries, comparisons=comparisons, gain_sign='control_minus_candidate_mae',
                scope='engineering_only' if read_json(run_dir/'identity.json')['engineering_check'] else 'exploratory_validation',
                attribution='B3/B2 same-information operator; B3/B0 system and candidate road structure', test_read=False)


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    require(device.type in ('cpu', 'cuda'), 'Only CPU/CUDA supported')
    require(device.type != 'cuda' or torch.cuda.is_available(), 'CUDA is not available')
    for filename, key in (('summary.json', 'files'), ('context_manifest.json', 'outputs')):
        names = read_json(args.data_dir/filename)[key]
        require(all(Path(name).name == name and not name.lower().startswith('test') for name in names),
                'Development input manifest attempts to read a non-development path')
    hashes, protocol = verify_package(args.data_dir), read_json(args.protocol)
    train, val = ChronologicalDataset(args.data_dir, 'train'), ChronologicalDataset(args.data_dir, 'val')
    validate_protocol(protocol, hashes, len(train), len(val), len(train.station_ids))
    require(args.epochs <= protocol['max_epochs'], 'Requested epochs exceed frozen budget')
    require((protocol.get('architecture_version'), protocol.get('latent_channels'), protocol.get('spatial_steps'),
             protocol.get('eta'), protocol.get('arms'), protocol.get('readout')) ==
            ('IS1.1', 4, 2, .5, list(ARMS), 'displacement_mean_inflow_mean_outflow_bias_free'), 'IS1.1 architecture protocol mismatch')
    graph, roads, graph_info = load_graph(args.graph_dir, train.station_ids, protocol['graph_sha256'])
    identity = make_identity(args, protocol, hashes, graph_info)
    if (args.run_dir/'identity.json').exists():
        require(read_json(args.run_dir/'identity.json') == identity, 'Run identity changed; preserved old checkpoints')
    else:
        require(not args.worker and not args.run_dir.exists(), 'Use a fresh directory or an existing IS1.1 run')
        args.run_dir.mkdir(parents=True)
        write_json(args.run_dir/'identity.json', identity)
    if args.worker:
        worker(args, identity, protocol, train, val, graph, roads)
        return
    with run_lock(args.run_dir):
        launch(args, train, val, graph, device)


def launch(args, train, val, graph, device):
    session = args.run_dir/('session_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6])
    session.mkdir()
    current_environment = environment(device)
    write_json(session/'environment.json', current_environment)
    if (args.run_dir/'initial_environment.json').exists():
        initial_environment = read_json(args.run_dir/'initial_environment.json')
        changes = {k: dict(saved=initial_environment.get(k), current=v) for k, v in current_environment.items()
                   if initial_environment.get(k) != v}
        write_json(session/'environment_changes.json', changes)
        if 'gpu' in changes:
            print('RESUME_GPU_NAME_CHANGE_ACCEPTED '+json.dumps(changes['gpu']), flush=True)
    else:
        write_json(args.run_dir/'initial_environment.json', current_environment)
    print('Run directory: '+str(args.run_dir), flush=True)
    print('Session directory: '+str(session), flush=True)
    initialization = preflight(args, train, graph)
    if (args.run_dir/'initialization.json').exists():
        require(read_json(args.run_dir/'initialization.json') == initialization, 'Reconstructed initialization differs')
    else:
        write_json(args.run_dir/'initialization.json', initialization)
    # Parent stays small; each child releases all model, graph and optimizer allocations on exit.
    del train, val, graph
    gc.collect()
    for arm in ARMS:
        command = [sys.executable, '-X', 'utf8', '-u', str(Path(__file__).resolve()), '--worker', arm, '--run-dir', str(args.run_dir),
                   '--data-dir', str(args.data_dir), '--graph-dir', str(args.graph_dir), '--protocol', str(args.protocol),
                   '--epochs', str(args.epochs), '--seed', str(args.seed), '--device', args.device, '--threads', str(args.threads)]
        if args.check:
            command.append('--check')
        print('Starting '+arm, flush=True)
        with (session/(arm+'.log')).open('w', encoding='utf-8') as log:
            process = subprocess.Popen(command, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, encoding='utf-8', errors='replace')
            for line in process.stdout:
                print(line, end='', flush=True)
                log.write(line)
                log.flush()
            code = process.wait()
        write_json(session/(arm+'_process_result.json'), dict(exit_code=code, arm=arm))
        require(code == 0, f'{arm} process failed ({code}); checkpoints preserved; see {session/(arm+".log")}')
    report = paired_report(args.run_dir)
    write_json(session/'comparison.json', report)
    write_json(args.run_dir/'latest_comparison.json', report)
    print(json.dumps(dict(status=report['status'], common_epoch=report['common_epoch'],
                         b2_stability_before_ranking=report['b2_stability_before_ranking'],
                         comparisons=report['comparisons'], output=str(session/'comparison.json')), indent=2), flush=True)


if __name__ == '__main__':
    main()
