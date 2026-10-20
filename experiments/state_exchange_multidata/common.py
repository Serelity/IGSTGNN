"""Dataset-shape extension of the frozen GW16 architecture and training recipe."""
from copy import deepcopy
from pathlib import Path

from experiments.position_state_gw16 import common as gw16
from experiments.position_state_gw16.common import (
    atomic_save, digest_json, exclusive_run, git_revision, load_state, phase_for,
    read_json, require, restore_rng, rng_state, runtime, sha256, synchronize,
    tensor_tree_hash, write_json)

PACKAGE = Path(__file__).resolve().parent
REPO = PACKAGE.parents[1]
SCHEMA = 'state_exchange_multidata_training_v1'
BASELINE_SCHEMA = 'state_exchange_multidata_original_training_v1'
LOCKED_CONFIG = deepcopy(gw16.LOCKED_CONFIG)
DATASETS = ('Alameda', 'Contra_Costa', 'Orange')


def spec(dataset):
    require(type(dataset) is str and dataset in DATASETS, 'Unknown exact dataset name')
    registry = read_json(PACKAGE/'registry.json')
    require(set(registry) == set(DATASETS), 'Published release registry fields changed')
    result = deepcopy(registry[dataset])
    require(result['train_samples'] == int(result['total_samples']*.70)
            and result['val_samples'] == int(result['total_samples']*.15)
            and result['test_samples'] == result['total_samples']
                -result['train_samples']-result['val_samples'],
            'Registered release split does not match author rounding')
    return result


def dataset_name(config):
    require(isinstance(config, dict), 'Configuration must be an object')
    names = [name for name in DATASETS if config.get('dataset') == name+'_author_release']
    require(len(names) == 1, 'Configuration has an unknown dataset identity')
    return names[0]


def load_config(dataset, path=None):
    record = spec(dataset)
    config = deepcopy(LOCKED_CONFIG)
    config.update(dataset=dataset+'_author_release', nodes=record['nodes'])
    validate_config(config)
    if path is not None:
        override = read_json(path)
        require(isinstance(override, dict) and not (set(override)-set(config)),
                'Unknown configuration override fields')
        require(all(override[key] == config[key] for key in override),
                'The unchanged GW16 recipe is locked; configuration overrides cannot tune a dataset')
    return config


def validate_config(config, *, engineering=False):
    name = dataset_name(config)
    expected = deepcopy(LOCKED_CONFIG)
    expected.update(dataset=name+'_author_release', nodes=spec(name)['nodes'])
    if engineering:
        expected.update(max_epochs=3, lr_milestones=[])
    require(set(config) == set(expected), 'Configuration fields differ from the locked GW16 recipe')
    require(all(config[key] == value and (type(config[key]) is bool if type(value) is bool
                else not isinstance(config[key], bool)) for key, value in expected.items()),
            'Configuration differs from the locked GW16 recipe')


def source_hashes():
    """Freeze both adapters and every original source used by the baseline path."""
    result = gw16.source_hashes()
    vendor = REPO/'experiments/chronological/original_igstgnn_558ab01'
    paths = [*PACKAGE.glob('*.py'), PACKAGE/'registry.json', PACKAGE/'run.sh',
             *PACKAGE.glob('config*.json'), *vendor.glob('*.py'), vendor/'manifest.json',
             REPO/'experiments/chronological/original_igstgnn_source.py',
             REPO/'experiments/chronological/train_original_igstgnn_baseline.py',
             REPO/'experiments/chronological/evaluate_original_igstgnn_modules.py',
             REPO/'experiments/chronological/original_igstgnn_checkpoint_compat.py',
             REPO/'experiments/original_baseline_gw16/evaluate.py',
             REPO/'experiments/final_test_gw16/state.py',
             REPO/'experiments/final_test_gw16_recovery/recover.py']
    require(all(path.is_file() for path in paths), 'An execution dependency is missing')
    result.update({path.relative_to(REPO).as_posix(): sha256(path) for path in sorted(set(paths))})
    return result


def protected_output(path, data_dir, *, allow_run_children=False):
    path, data_dir = Path(path).resolve(), Path(data_dir).resolve()
    require(path != data_dir and data_dir not in path.parents, 'Output must be outside the dataset')
    for parent in (path, *path.parents):
        identity = parent/'identity.json'
        if identity.is_file():
            require((parent == path or allow_run_children)
                    and read_json(identity).get('schema') in (SCHEMA, BASELINE_SCHEMA),
                    'Output intersects another experiment: '+str(parent))
    return path
