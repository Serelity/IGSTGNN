"""Origin-available historical descriptors; no model fitting or test evaluation.

Only the fingerprinted author object array may be unpickled. Its entire payload
is deserialized, including test objects, but only train/validation x_data is
accessed. Quantile boundaries use training histories without target labels.
"""
from pathlib import Path
import sys

import numpy as np

from experiments.branch_readout_gw9.readout_auth import require, sha256

FEATURES = ('last_flow', 'mean_abs_change', 'window_std', 'last_change',
    'history_trend', 'neighbor_change_disagreement', 'neighbor_change_dispersion')
MATCHED_FEATURES = ('mean_abs_change', 'neighbor_change_disagreement')
AUTHOR_COUNT = 5585
SPLIT_BOUNDS = {'train':[0,3909],'val':[3909,4746]}


def feature_arrays(history, mean, std, adjacency):
    history = np.asarray(history, dtype=np.float64)
    adjacency = np.asarray(adjacency, dtype=np.float64)
    require(history.ndim == 3 and history.shape[1] >= 2 and min(history.shape) > 0
        and np.isfinite(history).all(), 'Expected finite [sample,history,node] traffic inputs')
    nodes = history.shape[-1]
    require(adjacency.shape == (nodes,nodes) and np.isfinite(adjacency).all()
        and (adjacency >= 0).all(), 'Expected finite nonnegative author adjacency')
    require(np.isfinite([mean,std]).all() and std > 0, 'Invalid unchanged source normalization')
    traffic = history*std+mean
    changes = np.diff(traffic, axis=1)
    last = changes[:,-1]
    weights = adjacency.copy(); np.fill_diagonal(weights,0.)
    mass = weights.sum(axis=1); nonempty = mass > 0
    weights = np.divide(weights,mass[:,None],out=np.zeros_like(weights),where=mass[:,None]>0)
    neighbor_mean = last@weights.T
    neighbor_variance = np.maximum((last*last)@weights.T-neighbor_mean*neighbor_mean,0.)
    disagreement = np.abs(last-neighbor_mean); dispersion = np.sqrt(neighbor_variance)
    disagreement[:,~nonempty] = np.nan; dispersion[:,~nonempty] = np.nan
    return dict(last_flow=traffic[:,-1], mean_abs_change=np.abs(changes).mean(axis=1),
        window_std=traffic.std(axis=1), last_change=last,
        history_trend=traffic[:,-1]-traffic[:,0],
        neighbor_change_disagreement=disagreement, neighbor_change_dispersion=dispersion)


def train_cutpoints(features):
    result = {}
    for name, values in features.items():
        finite = np.asarray(values)[np.isfinite(values)]
        cuts = np.quantile(finite,[.25,.5,.75]).tolist() if finite.size else None
        result[name] = dict(cutpoints=cuts, training_finite_count=int(finite.size),
            source='train_x_data_only_no_targets', tie_policy='searchsorted_right_no_jitter')
    return result


def assign_quartiles(values, cutpoints):
    values = np.asarray(values)
    result = np.full(values.shape,-1,dtype=np.int8)
    if cutpoints is None: return result
    cuts = np.asarray(cutpoints,dtype=np.float64)
    require(cuts.shape == (3,) and np.isfinite(cuts).all() and (np.diff(cuts)>=0).all(),
        'Three ordered training-input cutpoints required')
    finite = np.isfinite(values)
    result[finite] = np.searchsorted(cuts,values[finite],side='right')
    return result


def _load_array(path):
    try:
        return np.load(path,allow_pickle=True)
    except ModuleNotFoundError as error:
        if error.name not in ('numpy._core','numpy._core.multiarray'): raise
        import numpy.core.multiarray as multiarray
        sys.modules.setdefault('numpy._core.multiarray',multiarray)
        return np.load(path,allow_pickle=True)


def load_origin_features(data_dir, identity):
    directory = Path(data_dir).resolve()
    digests = {}
    for name, expected in identity['hashes'].items():
        path = (directory/name).resolve()
        require(path.parent == directory and path.is_file() and sha256(path) == expected,
            'Historical input data fingerprint differs: '+name)
        digests[str(path)] = expected
    require('incident_all.npy' in identity['hashes'] and 'adj_matrix.npy' in identity['hashes'],
        'Authenticated history array and graph are required')
    adjacency = np.load(directory/'adj_matrix.npy',allow_pickle=False)
    all_samples = _load_array(directory/'incident_all.npy')
    require(len(all_samples)==AUTHOR_COUNT,'Original author array size differs')
    normalization = identity['normalization']
    mean, std = normalization['mean'], normalization['std']
    require(normalization.get('refitted') is False,'Refitted history normalization is not permitted')
    collected = {}
    for split in ('train','val'):
        descriptor = identity['splits'][split]
        start,stop = descriptor['bounds_half_open']
        require([start,stop] == SPLIT_BOUNDS[split]
            and descriptor['selected_indices'] == list(range(start,stop))
            and descriptor['count'] == stop-start,'Author history split/axis differs')
        pieces = {name:[] for name in FEATURES}
        for offset in range(start,stop,128):
            histories = []
            for index in range(offset,min(offset+128,stop)):
                # Deliberately never read y_data or any sample >=4746.
                value = np.asarray(all_samples[index]['x_data'])
                require(value.shape == (12,496,3) and value.dtype == np.float32
                    and np.isfinite(value).all(),'Invalid authenticated historical x_data')
                histories.append(value[...,0])
            batch = feature_arrays(np.stack(histories),mean,std,adjacency)
            for name,values in batch.items(): pieces[name].append(values.astype(np.float32))
        collected[split] = {name:np.concatenate(values) for name,values in pieces.items()}
    del all_samples
    cuts = train_cutpoints(collected['train'])
    metadata = dict(features=list(FEATURES),matched_features=list(MATCHED_FEATURES),
        train_input_cutpoints=cuts, file_digests=digests, normalization=normalization,
        origin_available=True, target_values_used_for_features=False,
        training_targets_used=False,test_objects_deserialized=True,test_samples_accessed=False,
        test_evaluated=False,new_training_updates=0,inference_calls=0,
        graph_definition='as_stored_A_ij_row_neighbors_nonnegative_weights_diagonal_removed_row_normalized',
        graph_caveat='Stored row direction is not asserted to be physical upstream or travel delay.',
        isolated_nodes='neighbor_descriptors_nan_not_artificial_zero',
        precision='float64_feature_calculation_then_float32_storage_same_for_train_and_validation',
        quantile_ties='Repeated cutpoints can produce empty groups; no jitter or label-based cuts.')
    assert_unchanged(metadata)
    return collected['val'],metadata


def assert_unchanged(metadata):
    for name,expected in metadata['file_digests'].items():
        require(sha256(name)==expected,'History input changed during analysis: '+name)
