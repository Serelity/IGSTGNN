"""Reassign whole relation vectors within each sample's original support.

This is a checkpoint-content sensitivity control. It does not move the support
of an incident, create a natural no-incident observation, or assign physical
meanings to the relation columns. The only random state used is a local
Generator keyed by the fixed seed and original global sample index.
"""
from collections.abc import Mapping

import numpy as np


SCHEMA = 'frozen_min_within_support_location_shuffle_v1'
SCOPE = 'within_original_supported_station_vector_reassignment_not_support_location_test'
EVENT_KEYS = ('type', 'description', 'type_valid', 'description_valid',
              'relation', 'relation_available', 'report_present')
FLAG_KEYS = ('type_valid', 'description_valid', 'relation_available', 'report_present')


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _validate_events(events):
    _require(isinstance(events, Mapping), 'Event payload must be a mapping')
    _require(set(events) == set(EVENT_KEYS), 'Unexpected or missing event schema keys')
    _require(all(isinstance(value, np.ndarray) for value in events.values()),
             'Event fields must be NumPy arrays')
    relation = events['relation']
    _require(relation.ndim == 3 and relation.shape[-1] == 3 and relation.shape[1] > 0,
             'Relation must have axes [samples, stations, 3]')
    _require(relation.dtype == np.dtype(np.float32), 'Relation must use original float32 dtype')
    _require(np.isfinite(relation).all(), 'Non-finite relation vector')
    rows, nodes, _ = relation.shape
    for key in FLAG_KEYS:
        _require(events[key].shape == (rows,) and events[key].dtype == np.dtype(np.bool_),
                 'Event flag must be Boolean with sample axis: '+key)
    for key in ('type', 'description'):
        value = events[key]
        _require(value.shape == (rows,) and np.issubdtype(value.dtype, np.integer),
                 'Event category must have an integer sample axis: '+key)
        _require((value >= 0).all(), 'Event category must be nonnegative: '+key)
    # For finite float32 vectors, any nonzero component is exactly equivalent
    # to abs(relation).sum(-1) > 0, without float32 summation overflow.
    support = np.any(relation != 0, axis=-1)
    support &= events['report_present'][:, None] & events['relation_available'][:, None]
    return rows, nodes, support


def _same_bits(left, right):
    return (left.dtype == right.dtype and left.shape == right.shape
            and left.tobytes(order='C') == right.tobytes(order='C'))


def assert_shuffle_preserved(original, shuffled, donors):
    """Check the whole-vector intervention and return descriptive counts.

    ``donors[i, j]`` is the source station in sample i copied to station j.
    Every eligible row must be a support-only derangement. Repeated relation
    vectors may make a donor move numerically ineffective; this is reported,
    not repaired by repeatedly searching for a favorable shuffle.
    """
    rows, nodes, support = _validate_events(original)
    other_rows, other_nodes, other_support = _validate_events(shuffled)
    _require((rows, nodes) == (other_rows, other_nodes), 'Event axes changed')
    _require(np.array_equal(support, other_support), 'Original supported-station mask changed')
    for key in EVENT_KEYS:
        if key != 'relation':
            _require(_same_bits(original[key], shuffled[key]), 'Unshuffled event field changed: '+key)
    _require(isinstance(donors, np.ndarray) and donors.shape == (rows, nodes)
             and np.issubdtype(donors.dtype, np.integer), 'Invalid donor station axes or dtype')
    _require(((donors >= 0) & (donors < nodes)).all(), 'Donor station out of bounds')
    station_axis = np.arange(nodes)
    moved = donors != station_axis[None, :]
    counts = support.sum(-1)
    for row in range(rows):
        _require(np.array_equal(np.sort(donors[row]), station_axis),
                 'Donors are not a within-sample station permutation')
        _require(np.array_equal(donors[row, ~support[row]], station_axis[~support[row]]),
                 'Donor changed outside original supported stations')
        if counts[row] < 2:
            _require(not moved[row].any(), 'Zero/single-support sample must stay unchanged')
        else:
            _require(moved[row, support[row]].all(),
                     'Eligible support shuffle must have no self donor')
            _require(support[row, donors[row, support[row]]].all(),
                     'Donor came from outside original supported stations')
    expected = original['relation'][np.arange(rows)[:, None], donors]
    _require(_same_bits(expected, shuffled['relation']),
             'Shuffled relation is not an exact whole-vector donor copy')
    changed = np.any(original['relation'] != shuffled['relation'], axis=-1)
    changed_counts, moved_counts = changed.sum(-1), moved.sum(-1)
    unchanged_moved_counts = (moved & ~changed).sum(-1)
    duplicate_rows = []
    per_sample = []
    for row in range(rows):
        unique_count = len(np.unique(original['relation'][row, support[row]], axis=0))
        duplicate_rows.append(int(counts[row]) > unique_count)
        per_sample.append(dict(row_index=row, supported_station_count=int(counts[row]),
                               moved_station_count=int(moved_counts[row]),
                               changed_vector_count=int(changed_counts[row]),
                               unchanged_moved_vector_count=int(unchanged_moved_counts[row]),
                               unique_supported_vector_count=unique_count))
    difference = np.abs(shuffled['relation'].astype(np.float64)
                        - original['relation'].astype(np.float64))
    eligible = counts >= 2
    return dict(schema=SCHEMA, scope=SCOPE, rows=rows, nodes=nodes,
                per_sample=per_sample, eligible_rows=int(eligible.sum()),
                effective_changed_rows=int((changed_counts > 0).sum()),
                unchanged_due_single_or_zero_rows=int((~eligible).sum()),
                eligible_rows_with_no_numeric_change=int((eligible & (changed_counts == 0)).sum()),
                rows_with_duplicate_supported_vectors=int(sum(duplicate_rows)),
                supported_station_count=int(counts.sum()),
                moved_station_count=int(moved_counts.sum()),
                changed_vector_count=int(changed_counts.sum()),
                unchanged_moved_vector_count=int(unchanged_moved_counts.sum()),
                vector_abs_change=dict(sum=float(difference.sum()),
                                       mean=float(difference.mean()) if difference.size else 0.,
                                       max=float(difference.max()) if difference.size else 0.,
                                       scalar_positions=int(difference.size),
                                       mean_scope='all_samples_stations_relation_columns'))


def make_location_shuffle(events, sample_indices, *, seed=20261014):
    """Return copied events, donor station indices, and preservation audit.

    Sample IDs are the original release indices, not local batch positions.
    A random cycle deranges all supported stations when support has size >=2.
    It is not a uniformly sampled member of all possible derangements.
    """
    rows, nodes, support = _validate_events(events)
    _require(isinstance(seed, (int, np.integer)) and not isinstance(seed, (bool, np.bool_))
             and 0 <= int(seed) <= np.iinfo(np.uint32).max,
             'Seed must be a nonnegative 32-bit integer')
    indices = np.asarray(sample_indices)
    _require(indices.shape == (rows,) and np.issubdtype(indices.dtype, np.integer),
             'Sample indices must be a one-dimensional integer sample axis')
    _require((indices >= 0).all() and len(np.unique(indices)) == rows,
             'Sample indices must be unique and nonnegative')
    shuffled = {key: value.copy() for key, value in events.items()}
    donors = np.broadcast_to(np.arange(nodes, dtype=np.int64), (rows, nodes)).copy()
    for row, sample_index in enumerate(indices):
        supported = np.flatnonzero(support[row])
        if len(supported) >= 2:
            generator = np.random.default_rng(np.random.SeedSequence([int(seed), int(sample_index)]))
            cycle = generator.permutation(supported)
            donors[row, cycle] = np.roll(cycle, -1)
    shuffled['relation'] = events['relation'][np.arange(rows)[:, None], donors].copy()
    audit = assert_shuffle_preserved(events, shuffled, donors)
    audit.update(seed=int(seed), sample_index_scope='original_global_release_indices',
                 randomization='independent_seedsequence_seed_and_sample_index_random_cycle',
                 uses_targets=False, uses_global_numpy_rng=False)
    for row, sample_index in zip(audit['per_sample'], indices):
        row['sample_index'] = int(sample_index)
    return shuffled, donors, audit
