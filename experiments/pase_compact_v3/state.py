"""Isolated identities and atomic state for the PASE compact v3 experiment."""
from pathlib import Path
import numpy as np

from experiments.position_state_gw16.common import (
    DEPENDENCIES, REPO, atomic_save, digest_json, exclusive_run, load_state,
    read_json, require, restore_rng, rng_state, runtime, sha256, synchronize,
    tensor_tree_hash, write_json)
from experiments.position_state_gw16.objective import metric_contract
from experiments.state_exchange_multidata.common import dataset_name
from .protocol import ARMS, arm_spec, configuration, protocol_record

CORE_ARMS = ARMS

SCHEMA = 'pase_compact_v3_training_v1'
PACKAGE = Path(__file__).resolve().parent


def source_hashes():
    # Lock numerical execution dependencies, not README wording or launch paths.
    paths = list(PACKAGE.glob('*.py'))
    original = REPO/'experiments/position_state_gw16'
    paths += [original/name for name in ('__init__.py', 'common.py', 'data.py',
        'geometry.py', 'model.py', 'objective.py', 'train.py')]
    paths += [REPO/name for name in DEPENDENCIES]
    paths += [REPO/'experiments/state_exchange_multidata'/name
              for name in ('common.py', 'data.py', 'model.py', 'registry.json')]
    paths += list((REPO/'experiments/pase_paper_experiments').glob('*.py'))
    # Promotion and initialization use the unchanged v2 implementation.
    paths += list((REPO/'experiments/pase_mlp_v2').glob('*.py'))
    require(all(path.is_file() for path in paths), 'Missing numerical execution dependency')
    return {path.relative_to(REPO).as_posix(): sha256(path) for path in sorted(set(paths))}


def protected_output(path, data_dir, *, allow_run_children=False):
    path, data_dir = Path(path).resolve(), Path(data_dir).resolve()
    require(path != data_dir and data_dir not in path.parents,
            'Output must be outside the dataset')
    for parent in (path, *path.parents):
        if (parent/'identity.json').is_file():
            require((parent == path or allow_run_children)
                    and read_json(parent/'identity.json').get('schema') == SCHEMA,
                    'Output intersects another experiment: '+str(parent))
    return path


def epoch_order(seed, epoch, size):
    require(type(epoch) is int and epoch >= 1, 'Only trained epochs are shuffled')
    return np.random.RandomState((seed+1009*epoch) % (2**32)).permutation(size)


def validate_config(config, engineering=False):
    expected = configuration('FULL', dataset=dataset_name(config))
    if engineering:
        require(config.get('max_epochs') == 3, 'Engineering budget must be exactly three epochs')
        expected.update(max_epochs=3, lr_milestones=[])
    require(config == expected, 'Configuration differs from the fixed PASE compact v3 protocol')


def validate_identity(identity):
    payload = dict(identity)
    digest = payload.pop('identity_sha256', None)
    require(identity.get('schema') == SCHEMA and digest == digest_json(payload),
            'Invalid PASE compact v3 identity')
    validate_config(identity['config'], identity['engineering_only'])
    require(identity['seed'] == 2025 and identity['dataset'] == dataset_name(identity['config'])
            and identity['comparison_arms'] == list(CORE_ARMS)
            and identity['arm_specs'] == {arm: arm_spec(arm) for arm in CORE_ARMS}
            and identity['protocol'] == protocol_record()
            and identity['metric_contract'] == metric_contract(identity['config'])
            and identity['best_selection'] == 'validation_pooled_mae_includes_stage_zero'
            and identity['selection_includes_stage_zero'] is True
            and identity['test_evaluated'] is False,
            'Run roles, seed, metric contract or selection rule changed')
    reference = identity.get('external_full_reference')
    require(reference is None or (not identity['engineering_only']
            and identity['dataset'] == 'Contra_Costa'
            and reference.get('role_kind') == 'external_compact_inference'
            and reference.get('source_arm') == 'NO_POSITION_INPUT'
            and reference.get('completed_training_epochs') == 100),
            'Invalid external FULL reference role')
    limits = identity['phase_batch_limits']
    if identity['engineering_only']:
        require(all(type(v) is int and v > 0 for v in limits.values()),
                'Engineering phase limits must be positive')
    else:
        require(limits == {'train': None, 'val': None}, 'Formal runs require complete loaders')


def state_digest(state):
    return tensor_tree_hash({key: value for key, value in state.items()
                             if key != 'state_payload_sha256'})


def expected_batches(identity, split):
    key = 'train_loader' if split == 'train' else 'val_loader'
    complete = identity['data']['loader_batching'][key]['batches']
    limit = identity['phase_batch_limits'][split]
    return complete if limit is None else min(complete, limit)


def validate_saved(state, identity, arm, directory, batches=None):
    validate_identity(identity)
    require(arm in CORE_ARMS and state.get('schema') == SCHEMA
            and state.get('identity_sha256') == identity['identity_sha256']
            and state.get('arm') == arm, 'Checkpoint identity/arm mismatch')
    require(state.get('state_payload_sha256') == state_digest(state),
            'Checkpoint payload changed')
    actual_batches = expected_batches(identity, 'train')
    require(batches is None or batches == actual_batches, 'Training batch count changed')
    epoch, history = state['completed_epoch'], state['history']
    config = identity['config']
    require(type(epoch) is int and 0 <= epoch <= config['max_epochs']
            and len(history) == epoch+1
            and [row['epoch'] for row in history] == list(range(epoch+1)),
            'Invalid committed epoch history')
    require(state['stage'] == 'joint' and state['global_updates'] == epoch*actual_batches
            and state['scheduler_state']['last_epoch'] == epoch
            and history[0]['training'] is None,
            'Checkpoint updates, phase or scheduler changed')
    expected_lr = config['learning_rate']*config['lr_gamma']**sum(
        value <= epoch for value in config['lr_milestones'])
    require(all(abs(group['lr']-expected_lr) <= 1e-15
                for group in state['optimizer_state']['param_groups']),
            'Optimizer learning rate changed')
    length = identity['data']['loader_batching']['train_loader']['samples']
    order = np.asarray(state['train_order'])
    expected = epoch_order(identity['seed'], epoch, length) if epoch else np.arange(length)
    require(order.shape == (length,) and np.array_equal(order, expected),
            'Saved loader order changed')
    require(state.get('loader_generator_state', 'MISSING') is None,
            'Published loader has no independent generator; unexpected generator state')
    for row in history:
        order = epoch_order(identity['seed'], row['epoch'], length) if row['epoch'] else np.arange(length)
        lr = config['learning_rate']*config['lr_gamma']**sum(
            value < row['epoch'] for value in config['lr_milestones'])
        require(row['global_updates'] == row['epoch']*actual_batches
                and row['stage'] == 'joint'
                and row['train_order_sha256'] == tensor_tree_hash(order)
                and abs(row['learning_rate']-lr) <= 1e-15,
                'Saved sample order, phase, update count or learning rate changed')
    require(tensor_tree_hash(state['model_state']) == state['model_tensor_sha256'],
            'Saved model tensor hash changed')
    require(state['parameters'] == identity['initializations'][arm]['parameters'],
            'Parameter accounting changed')
    best = min(history, key=lambda row: row['validation']['mae'])
    require(state['best_epoch'] == best['epoch'] and state['best_metrics'] == best['validation'],
            'BEST is not validation pooled MAE including stage zero')
    name = state['best_checkpoint_file']
    require(isinstance(name, str) and Path(name).name == name
            and sha256(Path(directory)/name) == state['best_checkpoint_sha256'],
            'BEST checkpoint missing or changed')


def validate_runtime(previous, current):
    keys = ('torch', 'numpy', 'threads', 'dtype', 'cuda', 'device_name', 'python_device',
            'deterministic_algorithms', 'tf32', 'cudnn_benchmark', 'python', 'seed',
            'scipy', 'cudnn', 'cublas_workspace_config', 'default_dtype')
    require(all(key in previous and key in current and previous[key] == current[key]
                for key in keys), 'Resume requires the recorded numerical runtime/device/threads')


def validate_artifacts(directory, identity, data):
    for name, value in {'config.json': identity['config'], 'partition.json': data.partition,
                        'position.json': data.position_identity,
                        'metric_contract.json': identity['metric_contract']}.items():
        require((directory/name).is_file() and read_json(directory/name) == value,
                'Run artifact missing or changed: '+name)
