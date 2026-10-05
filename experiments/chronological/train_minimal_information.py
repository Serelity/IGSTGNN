"""v13b: fresh, paired full-backbone information ablations; no pretrained A."""

import argparse
import copy
import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.chronological.minimal_information_data import (
    COHORTS, DevelopmentData, json_hash, require, sha256, verify_sources, write_json,
)
from experiments.chronological.minimal_information_evaluation import REGIONS, compare_all, metrics, region_statistics
from src.models.minimal_information import MODELS, forward_inputs

PROTOCOL = Path(__file__).with_name('minimal_information_v13b.json')
PROTOCOL_SHA256 = 'ac832ff07a049a5bb0705fb2f2a0dcff8ffd97988ac017d4fabb5acd0e58db84'
FORMAT = 'v13b_sealed_epoch_v1'


def load_protocol():
    require(sha256(PROTOCOL) == PROTOCOL_SHA256, 'Frozen v13b protocol changed')
    protocol = json.loads(PROTOCOL.read_text())
    for name, expected in protocol['frozen_legacy_sources'].items():
        require(sha256(REPO / name) == expected, f'Frozen backbone producer changed: {name}')
    return protocol


def code_hashes(protocol):
    names = list(protocol['frozen_legacy_sources']) + [
        'src/models/minimal_information.py', 'experiments/chronological/minimal_information_v13b.json',
        'experiments/chronological/train_minimal_information.py',
        'experiments/chronological/minimal_information_data.py',
        'experiments/chronological/minimal_information_evaluation.py',
        'experiments/chronological/run_minimal_information.sh']
    return {name: sha256(REPO / name) for name in names}


def deterministic(device):
    if device.type == 'cuda':
        require(torch.cuda.is_available(), 'CUDA unavailable; activate igstgnn on an allocated V100 node')
        require(os.environ.get('CUBLAS_WORKSPACE_CONFIG') in (':4096:8', ':16:8'), 'Set CUBLAS_WORKSPACE_CONFIG before Python')
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def seed_all(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [cpu_tree(v) for v in value]
    return value


def tree_hash(value):
    def canonical(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            return ['tensor', str(tensor.dtype), list(tensor.shape), hashlib.sha256(tensor.numpy().tobytes()).hexdigest()]
        if isinstance(item, dict):
            return ['dict', [[type(k).__name__, str(k), canonical(v)] for k, v in sorted(item.items(), key=lambda kv: str(kv[0]))]]
        if isinstance(item, (list, tuple)):
            return ['sequence', [canonical(v) for v in item]]
        return item
    return json_hash(canonical(value))


def save_sealed(path, payload):
    payload = cpu_tree(payload)
    temporary = path.with_suffix('.tmp')
    torch.save({'format': FORMAT, 'sha256': tree_hash(payload), 'payload': payload}, temporary)
    temporary.replace(path)


def load_sealed(path):
    require(path.is_file() and not path.is_symlink(), 'Missing regular recovery checkpoint')
    saved = torch.load(path, map_location='cpu', weights_only=True)
    require(saved.get('format') == FORMAT and saved.get('sha256') == tree_hash(saved['payload']), 'Checkpoint integrity mismatch')
    return saved['payload']


def write_csv(path, rows):
    require(bool(rows), 'Cannot publish empty CSV')
    with Path(path).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def epoch_order(indices, seed, epoch):
    return np.random.default_rng(seed * 1000 + epoch).permutation(indices).tolist()


def evaluate(model, arm, data, phase, cohort, indices, batch_size, device, progress):
    model.eval()
    parts = {key: [] for key in ('errors', 'counts', 'ids', 'source_indices')}
    with torch.inference_mode():
        for offset in range(0, len(indices), batch_size):
            batch = data.batch(phase, cohort, indices[offset:offset + batch_size], arm)
            prediction = model(*forward_inputs(arm, batch, device)).squeeze(-1)
            raw = (prediction * data.scaler['std'] + data.scaler['mean']).cpu().numpy()
            errors, counts = region_statistics(raw, batch)
            for key, value in (('errors', errors), ('counts', counts), ('ids', batch['sample_id']),
                               ('source_indices', batch['source_index'])):
                parts[key].append(value)
            if offset % (batch_size * 10) == 0:
                progress('evaluation', phase=phase, cohort=cohort, completed=min(offset + batch_size, len(indices)), total=len(indices))
    return {key: np.concatenate(value) for key, value in parts.items()}


def train_epoch(model, arm, data, indices, optimizer, settings, seed, epoch, device, progress):
    model.train()
    order = epoch_order(indices, seed, epoch)
    before = model.embedding.weight.detach().clone()
    error_sum, count, max_norm, steps = 0., 0, 0., 0
    no_gradient = []
    synchronize(device)
    started = time.perf_counter()
    for offset in range(0, len(order), settings['batch_size']):
        batch = data.batch('fit', 'incident_full', order[offset:offset + settings['batch_size']], arm)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(*forward_inputs(arm, batch, device)).squeeze(-1)
        valid = torch.as_tensor(batch['valid'], device=device)
        target = (torch.as_tensor(batch['y'], device=device) - data.scaler['mean']) / data.scaler['std']
        require(bool(valid.any()) and bool(torch.isfinite(prediction).all()), 'Invalid training prediction/target support')
        loss = (prediction[valid] - target[valid]).abs().mean()
        require(bool(torch.isfinite(loss)), 'Nonfinite training loss')
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        require(grads and all(bool(torch.isfinite(g).all()) for g in grads), 'Missing/nonfinite gradients')
        require(all(g.device == device for g in grads), 'Gradient device mismatch')
        if offset == 0:
            no_gradient = [name for name, p in model.named_parameters() if p.grad is None]
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings['clip_grad_norm'])
        require(bool(torch.isfinite(norm)), 'Nonfinite gradient norm')
        optimizer.step()
        n = int(valid.sum())
        error_sum += float(loss.detach()) * n
        count += n
        max_norm = max(max_norm, float(norm))
        steps += 1
        if (steps - 1) % settings['progress_every_batches'] == 0:
            progress('training', arm=arm, seed=seed, epoch=epoch, completed=min(offset + settings['batch_size'], len(order)),
                total=len(order), loss_standardized=float(loss.detach()), optimizer_steps=steps, gradient_device=str(grads[0].device))
    synchronize(device)
    changed = not torch.equal(before, model.embedding.weight)
    require(max_norm > 0 and changed, 'No effective backbone learning update')
    require(all(bool(torch.isfinite(p).all()) for p in model.parameters()), 'Nonfinite trained parameters')
    return {'optimizer_steps': steps, 'valid_cells': count, 'mae_standardized': error_sum / count,
            'maximum_gradient_norm': max_norm, 'backbone_embedding_changed': changed,
            'first_batch_no_gradient_parameter_names': no_gradient, 'order_sha256': json_hash(order),
            'seconds': time.perf_counter() - started, 'gradient_device': str(device)}


def restore_epoch(saved, identity, indices, settings):
    require(saved['identity'] == identity, 'Recovery identity mismatch')
    epoch, history = saved['epoch'], saved['history']
    require(type(epoch) is int and 0 <= epoch <= settings['epochs'] and len(history) == epoch, 'Recovery epoch/history mismatch')
    best = None
    steps = math.ceil(len(indices) / settings['batch_size'])
    for number, row in enumerate(history, 1):
        require(row['epoch'] == number and row['training']['optimizer_steps'] == steps
                and row['training']['order_sha256'] == json_hash(epoch_order(indices, identity['seed'], number)),
                'Recovery order/step history mismatch')
        metric = row['selection']['all']['mae_pooled']
        require(metric is not None and np.isfinite(metric), 'Invalid historical selection metric')
        if best is None or metric < best['metric']:
            best = {'epoch': number, 'metric': metric, 'state_sha256': row['state_sha256']}
    require((best is None and saved['best'] is None) or (best is not None and
        {k: saved['best'][k] for k in best} == best and tree_hash(saved['best']['state']) == best['state_sha256']),
        'Recovery selected state/history mismatch')
    require(tree_hash(saved['model']) == (history[-1]['state_sha256'] if history else identity['initial_state_sha256']),
            'Recovery current model mismatch')
    require(saved['scheduler']['last_epoch'] == epoch, 'Recovery scheduler epoch mismatch')
    for state in saved['optimizer']['state'].values():
        require(float(state['step']) == epoch * steps, 'Recovery Adam step mismatch')
    require(epoch == 0 or bool(saved['optimizer']['state']), 'Recovery missing Adam state')
    return saved


def fit_one(arm, seed, data, settings, indices, device, output, identity_hash, architecture, progress, source=None):
    seed_all(seed)
    model = MODELS[arm](data.adjacency, architecture).to(device)
    require(all(p.device == device for p in model.parameters()), 'Model device mismatch')
    initial_hash = tree_hash(model.state_dict())
    identity = {'run_identity_sha256': identity_hash, 'arm': arm, 'seed': seed,
                'initial_state_sha256': initial_hash, 'settings': settings}
    optimizer = torch.optim.Adam(model.parameters(), lr=settings['learning_rate'],
        eps=settings['adam_eps'], weight_decay=settings['weight_decay'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings['lr_milestones'], gamma=settings['lr_gamma'])
    history, best, first = [], None, 1
    if source is not None and (source / 'last.pt').exists():
        saved = restore_epoch(load_sealed(source / 'last.pt'), identity, indices['fit'], settings)
        model.load_state_dict(saved['model'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        torch.set_rng_state(saved['rng_cpu'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state(saved['rng_cuda'], device)
        first, history, best = saved['epoch'] + 1, saved['history'], saved['best']
        progress('restored', arm=arm, seed=seed, epoch=first - 1)
    elif source is not None:
        require(not (source / 'fit_summary.json').exists(), 'Completed source fit missing checkpoint')
    output.mkdir()

    def publish(epoch):
        save_sealed(output / 'last.pt', dict(identity=identity, epoch=epoch, model=model.state_dict(),
            optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), history=history, best=best,
            rng_cpu=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state(device) if device.type == 'cuda' else None))
        write_json(output / 'history.json', history)

    publish(first - 1)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(first, settings['epochs'] + 1):
        started = time.perf_counter()
        training = train_epoch(model, arm, data, indices['fit'], optimizer, settings, seed, epoch, device, progress)
        selected = metrics(evaluate(model, arm, data, 'selection', 'incident_full', indices['selection'],
                                   settings['evaluation_batch_size'], device, progress))
        metric = selected['all']['mae_pooled']
        require(metric is not None and np.isfinite(metric), 'Missing/nonfinite selection metric')
        state_hash = tree_hash(model.state_dict())
        if best is None or metric < best['metric']:
            best = {'epoch': epoch, 'metric': metric, 'state_sha256': state_hash, 'state': cpu_tree(model.state_dict())}
        scheduler.step()
        history.append({'epoch': epoch, 'training': training, 'selection': selected,
                        'state_sha256': state_hash, 'best_epoch': best['epoch'],
                        'train_and_selection_seconds': time.perf_counter() - started})
        publish(epoch)
        progress('epoch_complete', arm=arm, seed=seed, epoch=epoch, total_epochs=settings['epochs'],
                 selection_all_mae=metric, best_epoch=best['epoch'], backbone_trained=True)
    require(best is not None, 'No trained checkpoint selected')
    save_sealed(output / 'best.pt', {'identity': identity, 'best': best})
    result = {'arm': arm, 'seed': seed, 'initial_state_sha256': initial_hash, 'selected_epoch': best['epoch'],
        'selected_state_sha256': best['state_sha256'], 'selected_checkpoint_sha256': sha256(output / 'best.pt'),
        'selection_all_mae': best['metric'], 'epochs_completed': len(history),
        'optimizer_steps': sum(row['training']['optimizer_steps'] for row in history),
        'capacity': model.capacity(arm), 'device': str(device),
        'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
        'train_and_selection_seconds': sum(row['train_and_selection_seconds'] for row in history),
        'recovered_epochs': first - 1, 'new_optimizer_steps': sum(row['training']['optimizer_steps'] for row in history[first - 1:])}
    write_json(output / 'fit_summary.json', result)
    del model, optimizer, scheduler
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return result


def freeze_endpoints(output, endpoints, keys):
    require(set(endpoints) == set(keys), 'All selected endpoints must exist before audit')
    manifest = {}
    for key in keys:
        result = endpoints[key]
        path = output / key / 'best.pt'
        require(sha256(path) == result['selected_checkpoint_sha256'], 'Selected checkpoint changed before audit')
        selected = load_sealed(path)
        require(tree_hash(selected['best']['state']) == result['selected_state_sha256']
                and selected['best']['epoch'] == result['selected_epoch'], 'Selected identity mismatch')
        manifest[key] = {k: result[k] for k in ('selected_epoch', 'selected_state_sha256', 'selected_checkpoint_sha256')}
    write_json(output / 'frozen_endpoints.json', manifest)
    return manifest


def export_record(path, record):
    rows = []
    for i, sample in enumerate(record['ids']):
        row = {'sample_id': int(sample), 'source_index': int(record['source_indices'][i])}
        for j, region in enumerate(REGIONS):
            row[region + '_absolute_error_sum'] = float(record['errors'][i, j])
            row[region + '_valid_cells'] = int(record['counts'][i, j])
        rows.append(row)
    write_csv(path, rows)


def evaluate_selected(output, data, endpoints, protocol, device, progress):
    records = {}
    for key, endpoint in endpoints.items():
        arm, seed = endpoint['arm'], endpoint['seed']
        require(sha256(output / key / 'best.pt') == endpoint['selected_checkpoint_sha256'], 'Frozen checkpoint changed')
        saved = load_sealed(output / key / 'best.pt')
        model = MODELS[arm](data.adjacency, protocol['architecture']).to(device)
        model.load_state_dict(saved['best']['state'], strict=True)
        for phase in ('selection', 'audit'):
            phase_records = {}
            for cohort in COHORTS:
                if cohort == 'incident':
                    full = phase_records['incident_full']
                    positions = {int(sample): i for i, sample in enumerate(full['ids'])}
                    selected = [positions[int(data.secondary_rows[i]['positive_sample_index'])] for i in data.plan[phase][cohort]]
                    record = {name: values[selected] for name, values in full.items()}
                else:
                    record = evaluate(model, arm, data, phase, cohort, data.plan[phase][cohort],
                        protocol['training']['evaluation_batch_size'], device, progress)
                phase_records[cohort] = record
                export_record(output / key / f'{phase}_{cohort}.csv', record)
            write_json(output / key / f'{phase}_metrics.json', {k: metrics(v) for k, v in phase_records.items()})
            if phase == 'audit':
                records.setdefault(seed, {})[arm] = phase_records
        require(tree_hash(model.state_dict()) == endpoint['selected_state_sha256'], 'Evaluation changed frozen state')
        del model
        gc.collect()
    times = {int(row['sample_index']): row['t0'] for row in data.rows}
    comparison = compare_all(records, times, protocol)
    write_csv(output / 'paired_comparisons.csv', comparison.pop('rows'))
    write_csv(output / 'weekly_statistics.csv', comparison.pop('weekly_rows'))
    write_json(output / 'comparison.json', comparison)
    return comparison


def run(data_dir, primary_dir, secondary_dir, output, device='cuda:0', mode='check', resume_from=None,
        *, protocol=None, verifier=verify_sources, producer=code_hashes, progress=None):
    # Injectable dependencies are for synthetic tests only; CLI always loads frozen protocol.
    protocol = load_protocol() if protocol is None else copy.deepcopy(protocol)
    require(mode in ('preflight', 'check', 'pilot', 'run'), 'Unknown execution mode')
    output = Path(output)
    partial = Path(str(output) + '.partial')
    require(not output.exists() and not output.is_symlink() and not partial.exists() and not partial.is_symlink(),
            'Preserve existing output; choose a new run name')
    source = Path(resume_from).resolve() if resume_from else None
    if source is not None:
        require(mode == 'run' and source.is_dir() and not Path(resume_from).is_symlink()
                and source not in (output.resolve(), partial.resolve()), 'Recovery requires separate regular source and full run')
    partial.mkdir(parents=True)
    started = time.perf_counter()
    if progress is None:
        def progress(stage, **fields):
            print(json.dumps({'stage': stage, 'elapsed_seconds': round(time.perf_counter() - started, 3), **fields}, ensure_ascii=False), flush=True)
    try:
        hashes = verifier(data_dir, primary_dir, secondary_dir, protocol)
        code = producer(protocol)
        data = DevelopmentData(data_dir, primary_dir, secondary_dir, protocol)
        scaler = data.build_scaler()
        write_json(partial / 'protocol.json', protocol)
        write_json(partial / 'scaler.json', scaler)
        write_json(partial / 'eligibility.json', data.plan)
        data_identity = {'protocol_sha256': json_hash(protocol), 'input_sha256': hashes,
            'code_sha256': code, 'scaler_sha256': json_hash(scaler), 'plan_sha256': json_hash(data.plan)}
        if mode == 'preflight':
            summary = {'status': 'V13B_INPUT_PREFLIGHT_PASS', 'scientific_status': 'NOT_EVALUATED',
                       'identity': data_identity, 'phase_counts': {phase: {k: len(v) for k, v in cohorts.items()} for phase, cohorts in data.plan.items()},
                       'scaler': scaler, 'training_performed': False, 'audit_targets_evaluated': False}
        else:
            device = torch.device(device)
            if device.type == 'cuda' and device.index is None:
                device = torch.device('cuda:0')
            require(device.type in ('cpu', 'cuda'), 'Only CPU engineering or CUDA supported')
            deterministic(device)
            settings = copy.deepcopy(protocol['training'])
            seeds = protocol['seeds'] if mode == 'run' else [protocol[mode]['seed']]
            indices = {phase: data.plan[phase]['incident_full'] for phase in ('fit', 'selection')}
            if mode in ('check', 'pilot'):
                settings['epochs'] = protocol[mode]['epochs']
            if mode == 'check':
                settings['batch_size'] = protocol['check']['batch_size']
                indices = {phase: values[:protocol['check']['samples_per_phase']] for phase, values in indices.items()}
            identity = dict(data_identity, mode=mode, device=str(device), torch=str(torch.__version__), numpy=np.__version__,
                device_name=torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU',
                runtime={'python': sys.version.split()[0], 'cuda': torch.version.cuda,
                         'cudnn': torch.backends.cudnn.version(), 'cpu_threads': torch.get_num_threads(),
                         'cpu_interop_threads': torch.get_num_interop_threads(),
                         'cublas_workspace': os.environ.get('CUBLAS_WORKSPACE_CONFIG') if device.type == 'cuda' else None},
                seeds=seeds, settings=settings, selected_indices=indices)
            if source is not None:
                require((source / 'run_identity.json').is_file() and not (source / 'run_identity.json').is_symlink(), 'Missing recovery identity')
                require(json.loads((source / 'run_identity.json').read_text()) == identity, 'Recovery run identity mismatch')
            write_json(partial / 'run_identity.json', identity)
            endpoints, initial = {}, {}
            for seed in seeds:
                for arm in protocol['arms']:
                    key = f'{arm}_s{seed}'
                    endpoints[key] = fit_one(arm, seed, data, settings, indices, device, partial / key,
                        json_hash(identity), protocol['architecture'], progress, source / key if source else None)
                    value = endpoints[key]['initial_state_sha256']
                    require(seed not in initial or initial[seed] == value, 'Arms did not share exact initialization')
                    initial[seed] = value
                    write_json(partial / 'completed_fits.json', endpoints)
            comparison = None
            if mode == 'run':
                keys = [f'{arm}_s{seed}' for seed in seeds for arm in protocol['arms']]
                freeze_endpoints(partial, endpoints, keys)
                progress('all_endpoints_frozen', count=len(keys))
                data.audit_unlocked = True
                comparison = evaluate_selected(partial, data, endpoints, protocol, device, progress)
            status = {'check': 'V13B_ENGINEERING_CHECK_PASS', 'pilot': 'V13B_COST_PILOT_COMPLETE',
                      'run': 'V13B_DEVELOPMENT_COMPARISON_COMPLETE'}[mode]
            summary = {'status': status, 'scientific_status': 'REUSED_DEVELOPMENT_ONLY_SEE_INCREMENT_GATES' if mode == 'run' else 'NOT_EVALUATED_ENGINEERING_ONLY',
                'identity': identity, 'training_performed': True, 'backbone_retrained': True,
                'audit_targets_evaluated': mode == 'run', 'old_checkpoint_loaded': False, 'old_scaler_loaded': False,
                'optimizer_steps': sum(e['optimizer_steps'] for e in endpoints.values()),
                'new_optimizer_steps': sum(e['new_optimizer_steps'] for e in endpoints.values()),
                'endpoints': endpoints, 'comparison': comparison}
            if mode == 'pilot':
                measured = sum(e['train_and_selection_seconds'] for e in endpoints.values())
                summary['cost_projection'] = {'measured_three_arm_one_epoch_train_selection_seconds': measured,
                    'projected_nine_fit_60_epoch_train_selection_seconds': measured * len(protocol['seeds']) * protocol['training']['epochs'],
                    'interpretation': 'Linear estimate only; excludes final cohort evaluation, checkpoint/setup overhead and queue time. Full budget fixed before pilot; no metric-based tuning.'}
        require(producer(protocol) == code, 'Producer code changed during run')
        require(verifier(data_dir, primary_dir, secondary_dir, protocol) == hashes, 'Input package changed during run')
        summary.update(host=socket.gethostname(), slurm_job_id=os.environ.get('SLURM_JOB_ID'),
            python=sys.executable, elapsed_seconds=time.perf_counter() - started,
            protocol_file_sha256=sha256(PROTOCOL), boundaries=protocol['boundaries'],
            git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip())
        summary['artifacts_sha256'] = {str(path.relative_to(partial)): sha256(path)
            for path in sorted(partial.rglob('*')) if path.is_file()}
        write_json(partial / 'summary.json', summary)
        partial.rename(output)
        return summary
    except Exception as error:
        write_json(partial / 'failure.json', {'status': 'FAILED', 'error_type': type(error).__name__, 'error': str(error),
            'traceback': traceback.format_exc(), 'elapsed_seconds': time.perf_counter() - started})
        raise


def report(summary):
    print('Status:', summary['status'])
    print('Scientific status:', summary['scientific_status'])
    print('Training performed:', summary['training_performed'])
    print('Audit targets evaluated:', summary['audit_targets_evaluated'])
    print('Optimizer steps:', summary.get('optimizer_steps', 0))
    if 'device' in summary['identity']:
        print('Device:', summary['identity']['device'], summary['identity']['device_name'])
    for name, endpoint in summary.get('endpoints', {}).items():
        print(name + ':', json.dumps(endpoint, ensure_ascii=False))
    if summary.get('comparison'):
        print('Increment gates:', json.dumps(summary['comparison']['comparisons'], ensure_ascii=False))
    if 'cost_projection' in summary:
        print('Cost projection:', json.dumps(summary['cost_projection'], ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    execute = sub.add_parser('run')
    execute.add_argument('--data-dir', type=Path, required=True)
    execute.add_argument('--primary-control-dir', type=Path, required=True)
    execute.add_argument('--secondary-control-dir', type=Path, required=True)
    execute.add_argument('--output', type=Path, required=True)
    execute.add_argument('--device', default='cuda:0')
    execute.add_argument('--mode', choices=('preflight', 'check', 'pilot', 'run'), default='check')
    execute.add_argument('--resume-from', type=Path)
    show = sub.add_parser('report')
    show.add_argument('summary', type=Path)
    args = parser.parse_args()
    if args.action == 'report':
        report(json.loads(args.summary.read_text()))
    else:
        report(run(args.data_dir, args.primary_control_dir, args.secondary_control_dir,
                   args.output, args.device, args.mode, args.resume_from))


if __name__ == '__main__':
    main()
