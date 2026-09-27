"""Checked epoch-boundary recovery for v12c; legacy incomplete best states never guessed."""

import hashlib
import json
from pathlib import Path

import torch


def fingerprint(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return value


def choose_best(history, baseline, protocol, eligible):
    best_epoch, best_metrics = 0, baseline
    for index, row in enumerate(history, 1):
        if row['epoch'] != index:
            raise ValueError('Recovery history is not a contiguous epoch sequence')
        allowed, checks = eligible(row['selection'], baseline, protocol)
        if allowed and row['selection']['incident_full']['mae']['all'] < best_metrics['incident_full']['mae']['all']:
            best_epoch, best_metrics = index, row['selection']
        if (row['eligible'] != allowed or row['protection_checks'] != checks or
                row['best_epoch'] != best_epoch):
            raise ValueError('Recovery history disagrees with frozen selection rules/baseline')
    return best_epoch, best_metrics


def recover(directory, variant, seed, protocol, protocol_hash, backbone_hash, baseline,
            initial_state, eligible):
    """Return None for absent state; a restart record for unrecoverable legacy best state.

    The original source directory is read-only. New-format last_gate.pt is the
    authoritative atomic bundle, including history, optimizer, and selected state.
    """
    directory = Path(directory)
    used = {}

    def checkpoint(name):
        path = directory / name
        used[str(path.resolve())] = fingerprint(path)
        state = torch.load(path, map_location='cpu', weights_only=True)
        expected = {'variant': variant, 'seed': seed, 'protocol_sha256': protocol_hash,
                    'backbone_state_sha256': backbone_hash}
        if any(state.get(key) != value for key, value in expected.items()):
            raise ValueError(f'Recovery checkpoint identity mismatch: {path}')
        for value in state['gate_state'].values():
            if not torch.isfinite(value).all():
                raise ValueError('Nonfinite recovered gate weights')
        return state

    last_path = directory / 'last_gate.pt'
    if not last_path.is_file():
        return None
    last = checkpoint('last_gate.pt')
    if last.get('format_version') not in (None, 2):
        raise ValueError('Unknown recovery checkpoint format')
    epoch = last['epoch']
    if not isinstance(epoch, int) or not 1 <= epoch <= protocol['training']['epochs']:
        raise ValueError('Invalid recovery epoch')
    if last.get('format_version') == 2:
        if last['training_settings'] != protocol['training']:
            raise ValueError('Recovery training settings changed')
        history = last['history']
    else:
        path = directory / 'history.json'
        used[str(path.resolve())] = fingerprint(path)
        # Legacy history was published before last_gate; a newer history row is uncommitted.
        history = json.loads(path.read_text())[:epoch]
    if len(history) != epoch:
        raise ValueError('Recovery checkpoint/history epoch mismatch')
    selected_epoch, selected_metrics = choose_best(history, baseline, protocol, eligible)
    steps = sum(row['training']['optimizer_steps'] for row in history)
    best = {'epoch': selected_epoch, 'selection_metrics': selected_metrics}
    method = 'epoch_boundary'
    if last.get('format_version') == 2:
        saved = last['best']
        if saved['epoch'] != selected_epoch or saved['selection_metrics'] != selected_metrics:
            raise ValueError('Recovered best state metadata disagrees with history')
        best['state'] = saved['state']
    elif epoch == protocol['training']['epochs'] and (directory / 'selected_gate.pt').is_file():
        selected = checkpoint('selected_gate.pt')
        if selected['epoch'] != selected_epoch:
            raise ValueError('Legacy selected checkpoint disagrees with epoch selection')
        best['state'] = selected['gate_state']
        method = 'completed_legacy_fit'
    elif selected_epoch == 0:
        best['state'] = cpu_tree(initial_state)
        method = 'legacy_epoch_with_identity_best'
    elif selected_epoch == epoch:
        best['state'] = last['gate_state']
        method = 'legacy_epoch_with_current_best'
    else:
        return {'restart_required': True, 'reason': 'LEGACY_BEST_WEIGHTS_NOT_SAVED',
                'last_epoch': epoch, 'best_epoch': selected_epoch, 'source_hashes': used}
    if any(not torch.isfinite(value).all() for value in best['state'].values()):
        raise ValueError('Nonfinite recovered best weights')
    return {'restart_required': False, 'method': method, 'epoch': epoch, 'history': history,
            'steps': steps, 'best': best, 'gate_state': last['gate_state'],
            'optimizer_state': last['optimizer_state'], 'source_hashes': used}


def memory_snapshot(device=None):
    """Linux RSS and CUDA allocator observations, not an OOM diagnosis."""
    result = {}
    try:
        for line in Path('/proc/self/status').read_text().splitlines():
            key, _, value = line.partition(':')
            if key in ('VmRSS', 'VmHWM'):
                result[key + '_KiB'] = int(value.split()[0])
    except (OSError, ValueError):
        pass
    if torch.cuda.is_initialized() and (device is None or torch.device(device).type == 'cuda'):
        result.update(cuda_allocated_bytes=torch.cuda.memory_allocated(device),
                      cuda_reserved_bytes=torch.cuda.memory_reserved(device),
                      cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device))
    return result


def partial_report(output):
    """Describe saved progress without treating it as completed research results."""
    output = Path(output)
    partial = output if output.name.endswith('.partial') else output.with_name(output.name + '.partial')
    if not partial.is_dir():
        print(f'No completed summary or partial output found: {output}')
        return
    print('INCOMPLETE — no final result; preserved directory:', partial)
    progress = partial / 'progress.json'
    if progress.is_file():
        print('Last progress:', progress.read_text().strip())
    for directory in sorted(partial.iterdir()):
        if not directory.is_dir() or not directory.name.startswith(('scalar_s', 'node_s')):
            continue
        completed = directory / 'summary.json'
        last = directory / 'last_gate.pt'
        try:
            if completed.is_file():
                data = json.loads(completed.read_text())
                print(directory.name, 'fit and audit saved; selected_epoch=', data['selected_epoch'])
            elif last.is_file():
                state = torch.load(last, map_location='cpu', weights_only=True)
                print(directory.name, 'last saved epoch=', state['epoch'],
                      'complete resume bundle=', state.get('format_version') == 2)
            else:
                print(directory.name, 'no complete epoch checkpoint')
        except (OSError, ValueError, RuntimeError, EOFError) as error:
            print(directory.name, 'cannot inspect checkpoint:', str(error))
