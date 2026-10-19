"""Immutable selection records and authenticated final-test prediction caches.

This module never selects a model, opens a data split or computes a metric.
Its numerical tests are intended for the server, not the development PC.
"""
import hashlib
import json
from pathlib import Path
import uuid

import numpy as np

from experiments.chronological.original_igstgnn_runtime import sha256, write_json

FREEZE_SCHEMA = 'final_test_gw16_freeze_v1'
ARCHIVE_SCHEMA = 'final_test_gw16_archive_v1'
ARRAY_FIELDS = {'prediction', 'target', 'sample_ids', 'station_ids'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def payload_hash(value):
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(',', ':'), allow_nan=False).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def snapshot_files(paths):
    result = {}
    for value in paths:
        path = Path(value).resolve()
        require(path.is_file(), 'Snapshot source is not a file: '+str(path))
        result[str(path)] = sha256(path)
    return result


def verify_files(mapping):
    require(isinstance(mapping, dict), 'File snapshot must be a mapping')
    for name, expected in mapping.items():
        path = Path(name)
        require(path.is_absolute() and path.is_file(), 'Snapshot source is missing: '+str(path))
        require(sha256(path) == expected, 'Frozen source changed: '+str(path))


def freeze_output(directory, payload):
    """Commit a selection record before any test loader or inference is opened."""
    directory = Path(directory)
    freeze_id = payload_hash(payload)
    record = dict(schema=FREEZE_SCHEMA, freeze_id=freeze_id, payload=payload)
    path = directory/'freeze.json'
    if path.is_file():
        saved = json.loads(path.read_text(encoding='utf-8'))
        require(saved == record and saved.get('freeze_id') == payload_hash(saved.get('payload')),
                'Existing final-test freeze differs; do not reselect using test results')
        return freeze_id
    require(not directory.exists() or (directory.is_dir() and not any(directory.iterdir())),
            'Final-test directory has residual files without a valid freeze; preserve it')
    directory.mkdir(parents=True, exist_ok=True)
    write_json(path, record)
    return freeze_id


def validate_arrays(arrays):
    require(isinstance(arrays, dict) and set(arrays) == ARRAY_FIELDS,
            'Final-test archive fields differ')
    for name in ('prediction', 'target'):
        value = arrays[name]
        require(isinstance(value, np.ndarray) and value.dtype == np.float32
                and value.shape == (839, 12, 496, 1) and np.isfinite(value).all(),
                'Final-test shape, dtype or finite values differ: '+name)
    samples, stations = arrays['sample_ids'], arrays['station_ids']
    require(isinstance(samples, np.ndarray) and samples.dtype == np.int64
            and np.array_equal(samples, np.arange(4746, 5585, dtype=np.int64)),
            'Final-test sample_ids must preserve the certified 839-sample axis')
    require(isinstance(stations, np.ndarray) and stations.dtype == np.int64
            and stations.shape == (496,) and len(np.unique(stations)) == 496,
            'Final-test station_ids must contain 496 unique int64 IDs')


def _fixed_metadata(freeze_id, case):
    require(isinstance(freeze_id, str) and bool(freeze_id), 'Missing final-test freeze ID')
    require(isinstance(case, str) and bool(case), 'Missing final-test case')
    return dict(schema=ARCHIVE_SCHEMA, freeze_id=freeze_id, case=case,
                split='test', test_evaluated=True, optimizer_updates=0)


def _validate_metadata(metadata, fixed):
    require(isinstance(metadata, dict), 'Final-test cache metadata must be an object')
    require(all(metadata.get(key) == value and type(metadata.get(key)) is type(value)
                for key, value in fixed.items()),
            'Final-test cache schema, freeze, case or test scope differs')


def cache_read(directory, freeze_id, case):
    """Authenticate cached arrays; the caller independently recomputes metrics."""
    directory = Path(directory)
    metadata_path, archive_path = directory/'metadata.json', directory/'predictions.npz'
    require(metadata_path.is_file() and archive_path.is_file(),
            'Final-test cache is incomplete; preserve it instead of overwriting')
    metadata_digest = sha256(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    _validate_metadata(metadata, _fixed_metadata(freeze_id, case))
    digest = sha256(archive_path)
    require(digest == metadata.get('archive_sha256'), 'Final-test cache checksum differs')
    with np.load(archive_path, allow_pickle=False) as saved:
        arrays = {key:saved[key] for key in saved.files}
    require(sha256(archive_path) == digest and sha256(metadata_path) == metadata_digest,
            'Final-test cache changed during read')
    validate_arrays(arrays)
    return arrays, metadata


def cache_write(directory, freeze_id, case, arrays, metadata):
    """Publish one complete cache with a sibling staging-directory rename."""
    directory = Path(directory)
    require(not directory.exists(), 'Final-test cache already exists; never overwrite')
    validate_arrays(arrays)
    fixed = _fixed_metadata(freeze_id, case)
    require(isinstance(metadata, dict), 'Final-test cache metadata must be an object')
    for key, value in fixed.items():
        require(key not in metadata or (metadata[key] == value and type(metadata[key]) is type(value)),
                'Caller cannot override fixed final-test metadata: '+key)
    require('archive_sha256' not in metadata, 'Archive checksum is generated by cache_write')
    directory.parent.mkdir(parents=True, exist_ok=True)
    staging = directory.parent/('.'+directory.name+'.writing_'+uuid.uuid4().hex)
    staging.mkdir()
    np.savez_compressed(staging/'predictions.npz', **arrays)
    result = dict(metadata, **fixed, archive_sha256=sha256(staging/'predictions.npz'))
    write_json(staging/'metadata.json', result)
    cache_read(staging, freeze_id, case)
    require(not directory.exists(), 'Final-test cache appeared during write; preserve both directories')
    staging.rename(directory)
    return result
