"""Origin-available trajectory descriptors without models or target access.

The trusted author object array is deserialized as a whole, including test
objects. Only authenticated train/validation ``x_data[..., 0]`` is accessed.
No target, test sample, normalization fit, or quantile fit is used here.
"""
from numbers import Real
from pathlib import Path

import numpy as np

from experiments.branch_readout_gw9 import readout_history
from experiments.branch_readout_gw9.readout_auth import require, sha256


FEATURES = ('last_flow', 'mean_abs_change', 'signed_direction', 'shape', 'turn_position')
AUTHOR_COUNT = 5585
SPLIT_BOUNDS = {'train': [0, 3909], 'val': [3909, 4746]}
HISTORY_STEPS = 12
NODE_COUNT = 496
INPUT_CHANNELS = 3
CHUNK_SIZE = 128
DEFINITIONS = {
    'last_flow': 'physical_x[11]; float64 inverse normalization then float32 storage',
    'mean_abs_change': 'mean(abs(diff(physical_x))); same GW9/GW10 float64 calculation then float32 storage',
    'signed_direction': '(physical_x[11]-physical_x[0])/sum(abs(diff(physical_x))); zero if path length is zero',
    'shape': '2*(mean(x[4:8])-mean(x[0:4])>=0)+(mean(x[8:12])-mean(x[4:8])>=0); 0 down/down, 1 down/up, 2 up/down, 3 up/up; ties use the nonnegative side',
    'turn_position': '3-point moving means centered at original indices 1..10; strictly opposite adjacent slopes define reversals centered at indices 2..9; choose maximum abs(right_slope-left_slope), earliest on exact ties; -1 if absent',
}


def _normalization(mean, std):
    require(all(isinstance(value, Real) and not isinstance(value, (bool, np.bool_))
        and np.isfinite(value) for value in (mean, std)) and std > 0,
        'Finite scalar mean and positive unchanged scalar std required')


def features_from_history(history, mean, std):
    """Return ``[sample,node]`` descriptors from finite normalized 12-step inputs.

    ``shape`` is a nonnegative/negative block-trend code, not a claim of strict
    monotonicity. Flat histories therefore have shape 3 and turn position -1.
    Reversal detection is strict: an intermediate zero slope is not skipped.
    """
    raw = np.asarray(history)
    require(raw.ndim == 3 and raw.shape[1] == HISTORY_STEPS
        and min(raw.shape) > 0 and raw.dtype.kind == 'f' and np.isfinite(raw).all(),
        'Expected finite floating [sample,12,node] traffic history')
    _normalization(mean, std)
    # Preserve the established GW9/GW10 operation order exactly, rather than
    # calculating changes in normalized units and scaling after the reduction.
    traffic = np.asarray(raw, dtype=np.float64) * std + mean
    require(np.isfinite(traffic).all(), 'Inverse-normalized history must be finite')
    changes = np.diff(traffic, axis=1)
    path = np.abs(changes).sum(axis=1)
    direction = np.divide(traffic[:, -1] - traffic[:, 0], path,
        out=np.zeros_like(path), where=path > 0.)
    blocks = [traffic[:, start:start+4].mean(axis=1) for start in (0, 4, 8)]
    shape = (2 * (blocks[1] - blocks[0] >= 0.)
        + (blocks[2] - blocks[1] >= 0.)).astype(np.int8)
    smooth = (traffic[:, :-2] + traffic[:, 1:-1] + traffic[:, 2:]) / 3.
    slopes = np.diff(smooth, axis=1)
    left, right = slopes[:, :-1], slopes[:, 1:]
    reversals = ((left > 0.) & (right < 0.)) | ((left < 0.) & (right > 0.))
    scores = np.where(reversals, np.abs(right - left), -np.inf)
    position = np.where(reversals.any(axis=1), scores.argmax(axis=1) + 2, -1).astype(np.int8)
    result = dict(last_flow=traffic[:, -1].astype(np.float32),
        mean_abs_change=np.abs(changes).mean(axis=1).astype(np.float32),
        signed_direction=direction.astype(np.float32), shape=shape, turn_position=position)
    require(all(np.isfinite(values).all() for values in result.values()),
        'Trajectory feature overflow or nonfinite value')
    return result


def _assert_cache(values, cached, split):
    require(isinstance(cached, dict), 'Missing cached origin features: ' + split)
    for name in ('last_flow', 'mean_abs_change'):
        actual = values[name]
        expected = np.asarray(cached.get(name))
        require(actual.dtype == expected.dtype and actual.shape == expected.shape
            and np.array_equal(actual, expected),
            'Exact GW10 cached ' + split + '/' + name + ' dtype/shape/values differ')


def assert_unchanged(metadata):
    """Rehash raw inputs after analysis, including every identity-bound file."""
    for path, expected in metadata['file_digests'].items():
        require(sha256(path) == expected, 'Trajectory input changed during analysis: ' + path)


def load_features(data_dir, identity, cache):
    """Load origin features only after authenticating all original data bytes.

    Production axes, split bounds/order, and the two inherited feature arrays
    must match exactly. This is an input diagnostic and never constructs a
    model or reads y_data, target caches, or test sample fields.
    """
    directory = Path(data_dir).resolve()
    digests = {}
    hashes = identity.get('hashes')
    require(isinstance(hashes, dict) and 'incident_all.npy' in hashes,
        'Authenticated original history fingerprint required')
    for name, expected in hashes.items():
        require(isinstance(name, str), 'Data fingerprint name must be a string')
        path = (directory / name).resolve()
        require(path.parent == directory and path.is_file() and sha256(path) == expected,
            'Trajectory input fingerprint differs: ' + name)
        digests[str(path)] = expected
    normalization = identity.get('normalization', {})
    require(normalization.get('refitted') is False,
        'Refitted history normalization is not permitted')
    mean, std = normalization.get('mean'), normalization.get('std')
    _normalization(mean, std)
    for split in ('train', 'val'):
        descriptor = identity['splits'][split]
        start, stop = SPLIT_BOUNDS[split]
        require(descriptor['bounds_half_open'] == [start, stop]
            and descriptor['selected_indices'] == list(range(start, stop))
            and descriptor['count'] == stop-start,
            'Author trajectory split/order differs: ' + split)
        require(split in cache and isinstance(cache[split].get('features'), dict),
            'GW10 train and validation feature caches required: ' + split)
    all_samples = readout_history._load_array(directory / 'incident_all.npy')
    require(len(all_samples) == AUTHOR_COUNT, 'Original author history count differs')
    collected = {}
    for split in ('train', 'val'):
        start, stop = SPLIT_BOUNDS[split]
        pieces = {name: [] for name in FEATURES}
        for offset in range(start, stop, CHUNK_SIZE):
            histories = []
            for index in range(offset, min(offset + CHUNK_SIZE, stop)):
                # Validate the full original input, then extract traffic only;
                # never access y_data or any test sample.
                sample = np.asarray(all_samples[index]['x_data'])
                require(sample.shape == (HISTORY_STEPS, NODE_COUNT, INPUT_CHANNELS)
                    and sample.dtype == np.float32 and np.isfinite(sample).all(),
                    'Invalid authenticated historical x_data')
                histories.append(sample[..., 0])
            batch = features_from_history(np.stack(histories), mean, std)
            for name in FEATURES:
                pieces[name].append(batch[name])
        values = {name: np.concatenate(parts, axis=0) for name, parts in pieces.items()}
        require(all(value.shape == (stop-start, NODE_COUNT) for value in values.values()),
            'Incomplete production trajectory axes: ' + split)
        _assert_cache(values, cache[split]['features'], split)
        collected[split] = values
    del all_samples
    metadata = dict(features=list(FEATURES), definitions=DEFINITIONS.copy(),
        feature_axis_order='sample,node', history_axis_order='sample,12,node',
        file_digests=digests, normalization=dict(normalization),
        source_split_bounds={name: list(bounds) for name, bounds in SPLIT_BOUNDS.items()},
        chunk_size=CHUNK_SIZE, physical_feature_precision='float64_then_float32',
        categorical_feature_dtype='int8', raw_input_shape=[HISTORY_STEPS, NODE_COUNT, INPUT_CHANNELS],
        raw_traffic_channel=0, origin_available=True, target_values_used_for_features=False,
        training_targets_used=False, test_objects_deserialized=True,
        test_samples_accessed=False, test_evaluated=False, new_training_updates=0,
        model_imported=False, inference_calls=0, normalization_refitted=False,
        inherited_feature_replay=dict(policy='same_dtype_same_shape_array_equal_no_tolerance',
            features=['last_flow', 'mean_abs_change'], splits=['train', 'val'], passed=True))
    assert_unchanged(metadata)
    return collected, metadata
