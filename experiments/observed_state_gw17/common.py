"""Locked GW17 configuration, source identity and atomic server state."""
from pathlib import Path

from experiments.position_state_gw16.common import (
    atomic_save, digest_json, exclusive_run, load_state, read_json, require,
    restore_rng, rng_state, runtime, sha256, synchronize, tensor_tree_hash, write_json,
    source_hashes as gw16_source_hashes)

PACKAGE = Path(__file__).resolve().parent
REPO = PACKAGE.parents[1]
SCHEMA = 'observed_state_gw17_training_v1'
LOCKED_CONFIG = dict(
    architecture='observed_state_gw17_v1', dataset='Contra_Costa_author_release',
    nodes=496, history=12, horizon=12, hidden=32, update_hidden=96,
    spatial_layers=2, dropout=.1, graph_top_k=8, max_region_size=32,
    partition='road_direction_postmile', exchange_rank=8,
    step_initial=.1, observation_step_initial=.1, init_readout='zero',
    transition_context='position_sensor_clock_no_horizon_specific_embedding',
    batch_size=48, max_epochs=100, learning_rate=.002, weight_decay=1e-5,
    adam_eps=1e-8, lr_milestones=[40, 70, 90], lr_gamma=.5,
    clip_grad_norm=5., region_loss_weight=.2, history_loss_weight=.1,
    minimum_region_coverage=.8, target_minimum=.001, mape_minimum=1.,
    selection='validation_pooled_mae_includes_stage_zero', early_stopping=False,
    event_inputs='published_type_description_relation',
    event_scope='forecast_origin_only_first_availability_unverified',
    historical_target='next_observed_X_before_correction_not_Y',
    observation_validity='both_adjacent_raw_flows_finite_and_gt_target_minimum')


def validate_config(config, *, engineering=False):
    expected = dict(LOCKED_CONFIG)
    if engineering:
        expected.update(max_epochs=3, lr_milestones=[])
    require(isinstance(config, dict) and set(config) == set(expected),
            'Configuration fields differ from the locked GW17 protocol')
    require(all(config[key] == value and (type(config[key]) is bool if type(value) is bool
                else not isinstance(config[key], bool)) for key, value in expected.items()),
            'Configuration differs from the locked GW17 protocol')


def load_config(path=None):
    config = read_json(PACKAGE/'config.json')
    validate_config(config)
    if path is not None:
        override = read_json(path)
        require(isinstance(override, dict) and not (set(override)-set(config))
                and all(override[key] == config[key] for key in override),
                'GW17 configuration is locked; overrides may only restate existing values')
    return config


def source_hashes():
    # Include every imported GW16 dependency without adding files to that
    # frozen package or modifying its existing source identity.
    result = gw16_source_hashes()
    paths = [*PACKAGE.glob('*.py'), PACKAGE/'config.json', PACKAGE/'run.sh']
    require(all(path.is_file() for path in paths), 'Missing GW17 execution dependency')
    result.update({path.relative_to(REPO).as_posix(): sha256(path) for path in sorted(paths)})
    return result


def phase_for(epoch, config):
    require(type(epoch) is int and 0 <= epoch <= config['max_epochs'], 'Epoch outside GW17 budget')
    return 'joint'


def protected_output(path, data_dir, *, allow_run_children=False):
    path, data_dir = Path(path).resolve(), Path(data_dir).resolve()
    require(path != data_dir and data_dir not in path.parents and path not in data_dir.parents,
            'GW17 output must be separate from the dataset')
    for parent in (path, *path.parents):
        if (parent/'identity.json').is_file():
            require((parent == path or allow_run_children)
                    and read_json(parent/'identity.json').get('schema') == SCHEMA,
                    'GW17 output intersects another experiment: '+str(parent))
    return path
