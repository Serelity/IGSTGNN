"""Configuration, provenance and atomic state for server entry points."""
from contextlib import contextmanager
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import random
import subprocess
import sys

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch

from experiments.chronological.original_igstgnn_runtime import (
    RELEASE, configure_runtime, run_lock, sha256, tensor_tree_hash, write_json)

PACKAGE = Path(__file__).resolve().parent
REPO = PACKAGE.parents[1]
SCHEMA = 'position_state_gw16_training_v1'
DEPENDENCIES = (
    'src/utils/igstgnn_published.py', 'src/utils/igstgnn_published_events.py',
    'src/utils/dataloader.py', 'data/xtraffic/prepare_splits.py',
    'experiments/chronological/original_igstgnn_runtime.py',
    'experiments/chronological/original_igstgnn_558ab01/model.py',
    'experiments/chronological/original_igstgnn_558ab01/graph_algo.py',
    'src/base/model.py',
)

LOCKED_CONFIG = dict(architecture='position_state_gw16_v1', dataset='Contra_Costa_author_release',
    nodes=496, history=12, horizon=12, hidden=32, update_hidden=96, direct_head_width=None,
    free_transition_width=None, spatial_layers=2, dropout=.1, graph_top_k=8,
    max_region_size=32, partition='road_direction_postmile', exchange_rank=8,
    step_initial=.1, capacity_tolerance=.02, coupling='both', event_ablation='none',
    batch_size=48, max_epochs=100, learning_rate=.002, weight_decay=1e-5,
    adam_eps=1e-8, lr_milestones=[40, 70, 90], lr_gamma=.5,
    clip_grad_norm=5., region_loss_weight=.2, minimum_region_coverage=.8,
    target_minimum=.001, mape_minimum=1.,
    selection='validation_pooled_mae_includes_stage_zero', early_stopping=False,
    event_inputs='published_type_description_relation',
    event_scope='offline_archive_attributes_first_availability_unverified')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode('utf-8')).hexdigest()


def load_config(path=None):
    config = read_json(PACKAGE/'config.json')
    validate_config(config)
    if path is not None:
        override = read_json(path)
        require(isinstance(override, dict), 'Configuration override must be an object')
        require(not (set(override)-set(config)),
                'Unknown configuration keys: '+str(set(override)-set(config)))
        require(all(override[key] == config[key] for key in override),
                'GW16 is locked: --config cannot alter the research protocol; use a new registered experiment')
    return config


def validate_config(c, *, engineering=False):
    require(isinstance(c, dict), 'Configuration must be an object')
    expected = dict(LOCKED_CONFIG)
    if engineering:
        expected.update(max_epochs=3, lr_milestones=[])
    require(set(c) == set(expected), 'Configuration fields differ from the registered GW16 protocol')
    require(all(c[key] == value and (type(c[key]) is bool if type(value) is bool
                else not isinstance(c[key], bool)) for key, value in expected.items()),
            'Configuration differs from the registered GW16 protocol')


def phase_for(epoch, c):
    require(type(epoch) is int and 0 <= epoch <= c['max_epochs'], 'Epoch outside the fixed protocol')
    return 'joint'


def source_hashes():
    paths = [*PACKAGE.glob('*.py'), PACKAGE/'config.json', PACKAGE/'run.sh',
             *(PACKAGE/'configs').glob('*.json'), *(REPO/p for p in DEPENDENCIES)]
    require(all(p.is_file() for p in paths), 'An execution dependency is missing')
    return {p.relative_to(REPO).as_posix(): sha256(p) for p in sorted(paths)}


def git_revision():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO,
            text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def atomic_save(path, value):
    path = Path(path)
    temp = path.with_name(path.name+'.tmp')
    with temp.open('wb') as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def load_state(path):
    # Only our own training state. Never deserialize arbitrary remote checkpoints.
    return torch.load(Path(path), map_location='cpu', weights_only=False)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [])


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:
        torch.cuda.set_rng_state_all([v.cpu() for v in state['cuda']])


def synchronize(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


def protected_output(path, data_dir, *, allow_run_children=False):
    path, data_dir = Path(path).resolve(), Path(data_dir).resolve()
    require(path != data_dir and data_dir not in path.parents, 'Output must be outside the dataset')
    # Historical experiment directories are not valid destinations for this model.
    for parent in (path, *path.parents):
        identity = parent/'identity.json'
        if identity.is_file():
            require((parent == path or allow_run_children)
                    and read_json(identity).get('schema') == SCHEMA,
                    'Output intersects another experiment: '+str(parent))
    return path


@contextmanager
def exclusive_run(path):
    with run_lock(path):
        yield


def runtime(device, threads, seed):
    value = configure_runtime(device, threads, seed)
    value.update(python=sys.version, git_revision=git_revision(), scipy=version('scipy'),
                 cudnn=torch.backends.cudnn.version(),
                 cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
                 default_dtype=str(torch.get_default_dtype()))
    return value
