"""Read-only certification of the existing complete GW16 validation reference."""
from pathlib import Path

import numpy as np

from experiments.position_state_gw16.evaluate import read_archive
from experiments.chronological.original_igstgnn_runtime import RELEASE
from .common import require, sha256


def verify_reference(record):
    require(isinstance(record, dict) and record.get('schema') == 'gw17_gw16_validation_reference_v1',
            'Missing certified GW16 validation reference record')
    files = record.get('source_files', {})
    directory = Path(record['directory']).resolve()
    require(set(files) == {str(directory/name) for name in ('metadata.json', 'predictions.npz')},
            'Reference artifact set differs')
    for name, expected in files.items():
        require(Path(name).is_file() and sha256(name) == expected,
                'Read-only GW16 reference changed: '+name)
    return record


def read_reference(directory, data=None):
    directory = Path(directory).resolve()
    files = {str(directory/name):sha256(directory/name) for name in ('metadata.json', 'predictions.npz')}
    arrays, metadata = read_archive(directory)
    role = metadata['checkpoint_role']
    require(metadata['arm'] == 'STATE_EXCHANGE' and metadata['engineering_only'] is False
            and metadata['split'] == 'val' and metadata['test_evaluated'] is False
            and role['selection'] == 'best' and role['full_protocol_reached'] is True
            and role['completed_training_epochs'] == 100
            and metadata['validation_replay']['passed'] is True,
            'GW17 requires the existing full GW16 STATE_EXCHANGE validation BEST')
    require(metadata['identity']['data']['hashes'] == RELEASE
            and len(arrays['prediction']) == 837,
            'GW16 reference must contain all 837 certified validation windows')
    if data is not None:
        require(metadata['identity']['data']['hashes'] == data.identity['hashes']
                and np.array_equal(arrays['station_ids'], data.station_ids)
                and np.array_equal(arrays['assignment'], data.assignment)
                and metadata['identity']['data']['position'] == data.position_identity,
                'GW16/GW17 data, position, station or region identities differ')
    record = dict(schema='gw17_gw16_validation_reference_v1', directory=str(directory),
        source_files=files, identity_sha256=metadata['identity_sha256'],
        checkpoint_role=role, metrics=metadata['diagnostics']['overall'],
        reference_role='historical_full_GW16_validation_cache_not_a_new_training_arm',
        historical_model_forward_calls=0, historical_training_updates=0,
        test_evaluated=False)
    verify_reference(record)
    return arrays, metadata, record
