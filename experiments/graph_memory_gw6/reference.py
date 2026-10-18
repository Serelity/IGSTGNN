"""Read-only certification of completed GW5 evidence and a GW4 BEST start.

Historical predictions are evidence only. All newly trained GW6 architecture
arms start from the identical, certified GW4 RECEIVER_RESIDUAL BEST tensors.
This module never trains or runs a historical model.
"""
from pathlib import Path

import numpy as np

from experiments.chronological import train_original_history_anchor as gw5
from experiments.chronological import evaluate_original_history_anchor as evaluation
from experiments.chronological.history_anchor_reference import (
    compatible, read_json, recertify as recertify_gw4, require,
    verify_identity_digest)


CONTROLS = ('STATIC_RESIDUAL', 'RECEIVER_RESIDUAL', 'PHASE_SHRINK')


def validate_certificate(metadata, identity, states, selection, epochs):
    require(metadata.get('schema') == 'gw5_history_anchor_producer_v1'
        and metadata.get('status') == 'GW5_PREDICTION_PASS',
        'Expected a completed GW5 prediction certificate')
    require(metadata.get('run_identity_sha256') == identity['identity_sha256']
        and metadata.get('reference') == identity['reference']
        and metadata.get('gw4_reference') == identity['gw4_reference'],
        'GW5 evaluation identity or anchor differs')
    require(metadata.get('common_completed_adaptation_epochs') == epochs
        and metadata.get('engineering_only') is False,
        'GW5 evaluation must use its completed full budget')
    for name in evaluation.PRODUCERS:
        require(metadata.get('producer_source', {}).get(name) ==
            gw5.base.sha256(gw5.REPO/'experiments/chronological'/name),
            'GW5 producer source differs: '+name)
    for arm in gw5.ARMS:
        state = states[arm]
        role = metadata.get('checkpoint_roles', {}).get(arm, {})
        epoch = state['best_epoch'] if selection == 'best' else state['completed_epoch']
        digest = state['best_checkpoint_sha256'] if selection == 'best' else state['_last_file_sha256']
        require(role.get('selection') == selection and role.get('epoch') == epoch
            and role.get('policy') == arm and role.get('checkpoint_sha256') == digest
            and role.get('model_tensor_sha256') == state['_'+selection+'_model_sha256'],
            'GW5 cached checkpoint role differs: '+arm)
        require(metadata.get('validation_replay', {}).get(arm, {}).get('passed') is True,
            'GW5 cached validation replay not passed: '+arm)


def find_evaluation(run, identity, states, selection, epochs, explicit=None):
    candidates = [Path(explicit).resolve()/'metadata.json'] if explicit else sorted(run.glob('*/metadata.json'))
    matches = []
    for path in candidates:
        try:
            metadata = read_json(path)
            validate_certificate(metadata, identity, states, selection, epochs)
            archive = path.parent/'paired_predictions.npz'
            require(gw5.base.sha256(archive) == metadata.get('archive_sha256'),
                'GW5 archive digest differs')
            matches.append(dict(directory=str(path.parent.resolve()),
                metadata_sha256=gw5.base.sha256(path), archive_sha256=metadata['archive_sha256']))
        except (OSError, KeyError, ValueError):
            if explicit:
                raise
    require(bool(matches), 'Missing certified GW5 '+selection+' evaluation; point --gw5-'+selection+
        '-eval to its EXISTING directory. Do not retrain or re-infer finished controls')
    return matches[-1]


def certify(run, evaluations=None):
    run = Path(run).resolve()
    identity = read_json(run/'identity.json')
    verify_identity_digest(identity)
    require(identity.get('schema') == gw5.SCHEMA and identity.get('design') == gw5.DESIGN
        and identity.get('engineering_only') is False,
        'Expected a full completed GW5 adaptation run')
    for name in gw5.EXECUTION_FILES:
        require(identity['execution_source'].get(name) == gw5.base.sha256(gw5.REPO/name),
            'GW5 execution source differs: '+name)
    require(read_json(run/'gw4_reference.json') == identity['gw4_reference'],
        'GW5 GW4 reference file changed')
    gw4_identity, _, gw4_states = recertify_gw4(identity['gw4_reference'])
    compatible(gw4_identity, identity)
    epochs = read_json(run/'budget.json')['target_completed_epochs']
    require(epochs == identity['gw4_reference']['completed_adaptation_epochs'] == 5,
        'GW6 requires the completed five-epoch GW4/GW5 historical evidence')
    states, roles = {}, {}
    batches = len(identity['data']['splits']['train']['partition'])
    for arm in gw5.ARMS:
        directory = run/arm
        before = gw5.base.sha256(directory/'last.pt')
        state = gw5.base.load_checkpoint(directory/'last.pt')
        gw5.validate_saved(state, identity, arm, directory, batches)
        require(state['completed_epoch'] == epochs, 'Finish the existing GW5 common budget: '+arm)
        require(gw5.base.sha256(directory/'last.pt') == before, 'GW5 state changed during read')
        state['_last_file_sha256'] = before
        state['_last_model_sha256'] = gw5.base.tensor_tree_hash(state['model_state'])
        best_path = directory/state['best_checkpoint_file']
        require(gw5.base.sha256(best_path) == state['best_checkpoint_sha256'], 'GW5 BEST changed: '+arm)
        state['_best_model_sha256'] = gw5.base.tensor_tree_hash(gw5.base.load_checkpoint(best_path))
        require(gw5.base.sha256(best_path) == state['best_checkpoint_sha256'], 'GW5 BEST changed during read')
        states[arm] = state
        roles[arm] = dict(completed_epoch=epochs, best_epoch=state['best_epoch'],
            last_checkpoint_sha256=before, best_checkpoint_sha256=state['best_checkpoint_sha256'],
            last_model_tensor_sha256=state['_last_model_sha256'],
            best_model_tensor_sha256=state['_best_model_sha256'])
    archives = {selection:find_evaluation(run, identity, states, selection, epochs,
        (evaluations or {}).get(selection)) for selection in ('best', 'last')}
    controls = {arm:(states if arm in states else gw4_states)[arm] for arm in CONTROLS}
    receiver = gw4_states['RECEIVER_RESIDUAL']
    source_path = (Path(identity['gw4_reference']['source_run'])/'RECEIVER_RESIDUAL'/
                   receiver['best_checkpoint_file']).resolve()
    require(gw5.base.sha256(source_path) == receiver['best_checkpoint_sha256'],
        'GW4 receiver BEST source digest differs')
    source_weights = gw5.base.load_checkpoint(source_path)
    source_tensor_hash = gw5.base.tensor_tree_hash(source_weights)
    require(source_tensor_hash == receiver['_best_model_sha256'],
        'GW4 receiver BEST tensors differ from certified evidence')
    require(gw5.base.sha256(source_path) == receiver['best_checkpoint_sha256'],
        'GW4 receiver BEST changed during read')
    certificate = dict(schema='gw6_certified_graph_memory_reference_v1', source_run=str(run),
        source_identity_sha256=identity['identity_sha256'],
        source_identity_file_sha256=gw5.base.sha256(run/'identity.json'),
        completed_adaptation_epochs=epochs, archives=archives, historical_gw5_arms=roles,
        source_initialization=dict(source='GW4', arm='RECEIVER_RESIDUAL', selection='best',
            source_identity_sha256=gw4_identity['identity_sha256'], checkpoint_file=str(source_path),
            checkpoint_sha256=receiver['best_checkpoint_sha256'],
            model_tensor_sha256=source_tensor_hash, epoch=receiver['best_epoch']),
        historical_controls={arm:dict(source='GW5' if arm == 'PHASE_SHRINK' else 'GW4',
            completed_epoch=controls[arm]['completed_epoch'], best_epoch=controls[arm]['best_epoch'],
            best_checkpoint_sha256=controls[arm]['best_checkpoint_sha256'],
            last_checkpoint_sha256=controls[arm]['_last_file_sha256']) for arm in CONTROLS},
        gw4_reference=identity['gw4_reference'], source_artifacts_written=False,
        historical_controls_are_new_budget_matched=False, new_training_updates=0, new_inference_calls=0)
    return certificate, identity, source_weights, controls


def recertify(saved):
    current, identity, weights, states = certify(saved['source_run'],
        {key:saved['archives'][key]['directory'] for key in ('best', 'last')})
    require(current == saved, 'Historical evidence or GW4 BEST initialization changed since GW6 construction')
    return identity, weights, states


def load_cached(reference, selection):
    require(selection in ('best', 'last'), 'Explicit BEST/LAST source role required')
    saved = reference['archives'][selection]
    directory = Path(saved['directory'])
    metadata_path = directory/'metadata.json'
    require(gw5.base.sha256(metadata_path) == saved['metadata_sha256'], 'GW5 cache metadata changed')
    metadata = read_json(metadata_path)
    path = directory/'paired_predictions.npz'
    require(gw5.base.sha256(path) == saved['archive_sha256'], 'GW5 cache archive changed')
    with np.load(path, allow_pickle=False) as payload:
        archive = dict(payload)
    evaluation.validate_archive(archive)
    require(gw5.base.sha256(path) == saved['archive_sha256']
        and gw5.base.sha256(metadata_path) == saved['metadata_sha256'],
        'GW5 cached evidence changed during read')
    return archive, metadata
