"""Read-only certification of completed GW4 controls and their GW2 anchor."""
from pathlib import Path

import numpy as np

from experiments.chronological import train_original_receiver_residual as gw4
from experiments.chronological import evaluate_original_receiver_residual as evaluation
from experiments.chronological.receiver_residual_reference import (
    compatible, read_json, recertify as recertify_gw2, require, verify_identity_digest)


def validate_certificate(metadata, identity, states, selection, epochs):
    require(metadata.get('schema') == 'gw4_receiver_residual_producer_v1'
        and metadata.get('status') == 'GW4_PREDICTION_PASS', 'Expected a completed GW4 producer certificate')
    require(metadata.get('run_identity_sha256') == identity['identity_sha256']
        and metadata.get('reference') == identity['reference'], 'GW4 evaluation identity or GW2 anchor differs')
    require(metadata.get('common_completed_adaptation_epochs') == epochs, 'GW4 evaluation adaptation budget differs')
    require(metadata.get('engineering_only') is False and metadata.get('old_parameters_frozen') is True,
            'GW4 reference must be a full frozen-anchor run')
    for name in evaluation.PRODUCERS:
        require(metadata.get('producer_source', {}).get(name) == gw4.base.sha256(gw4.REPO/'experiments/chronological'/name),
                'GW4 producer source differs: '+name)
    for arm in gw4.ARMS:
        state = states[arm]
        role = metadata.get('checkpoint_roles', {}).get(arm, {})
        epoch = state['best_epoch'] if selection == 'best' else state['completed_epoch']
        digest = state['best_checkpoint_sha256'] if selection == 'best' else state['_last_file_sha256']
        require(role.get('selection') == selection and role.get('epoch') == epoch
            and role.get('checkpoint_sha256') == digest
            and role.get('model_tensor_sha256') == state['_'+selection+'_model_sha256'],
            'GW4 cached checkpoint role differs: '+arm)
        require(metadata.get('validation_replay', {}).get(arm, {}).get('passed') is True,
                'GW4 cached validation replay not passed: '+arm)


def find_evaluation(run, identity, states, selection, epochs, explicit=None):
    candidates = [Path(explicit).resolve()/'metadata.json'] if explicit else sorted(run.glob('*/metadata.json'))
    matches = []
    for path in candidates:
        try:
            metadata = read_json(path)
            validate_certificate(metadata, identity, states, selection, epochs)
            archive = path.parent/'paired_predictions.npz'
            require(gw4.base.sha256(archive) == metadata.get('archive_sha256'), 'GW4 archive digest differs')
            matches.append(dict(directory=str(path.parent.resolve()), metadata_sha256=gw4.base.sha256(path),
                archive_sha256=metadata['archive_sha256']))
        except (OSError, KeyError, ValueError):
            if explicit:
                raise
    require(bool(matches), 'Missing certified GW4 '+selection+' evaluation; point --gw4-'+selection+
        '-eval to its EXISTING directory. Do not retrain or re-infer completed controls')
    return matches[-1]


def certify(run, evaluations=None):
    run = Path(run).resolve()
    identity = read_json(run/'identity.json')
    verify_identity_digest(identity)
    require(identity.get('schema') == gw4.SCHEMA and identity.get('design') == gw4.DESIGN
        and identity.get('engineering_only') is False, 'Expected the full GW4 frozen-adaptation run')
    for name in gw4.EXECUTION_FILES:
        require(identity['execution_source'].get(name) == gw4.base.sha256(gw4.REPO/name), 'GW4 execution source differs: '+name)
    require(read_json(run/'reference.json') == identity['reference'], 'GW4 source reference file changed')
    source_identity, source_weights = recertify_gw2(identity['reference'])
    compatible(source_identity, identity)
    epochs = read_json(run/'budget.json')['target_completed_epochs']
    require(epochs == 5, 'GW5 pilot is paired to the existing five-adaptation-epoch GW4 reference')
    states, roles = {}, {}
    batches = len(identity['data']['splits']['train']['partition'])
    for arm in gw4.ARMS:
        directory = run/arm
        before = gw4.base.sha256(directory/'last.pt')
        state = gw4.base.load_checkpoint(directory/'last.pt')
        gw4.validate_saved(state, identity, arm, directory, batches)
        require(state['completed_epoch'] == epochs, 'Finish the existing GW4 common budget: '+arm)
        require(gw4.base.sha256(directory/'last.pt') == before, 'GW4 state changed during read')
        state['_last_file_sha256'] = before
        state['_last_model_sha256'] = gw4.base.tensor_tree_hash(state['model_state'])
        best = gw4.base.load_checkpoint(directory/state['best_checkpoint_file'])
        state['_best_model_sha256'] = gw4.base.tensor_tree_hash(best)
        states[arm] = state
        roles[arm] = dict(completed_epoch=epochs, best_epoch=state['best_epoch'],
            last_checkpoint_sha256=before, best_checkpoint_sha256=state['best_checkpoint_sha256'],
            last_model_tensor_sha256=state['_last_model_sha256'], best_model_tensor_sha256=state['_best_model_sha256'])
    archives = {selection:find_evaluation(run, identity, states, selection, epochs,
        (evaluations or {}).get(selection)) for selection in ('best', 'last')}
    certificate = dict(schema='gw5_certified_gw4_reference_v1', source_run=str(run),
        source_identity_sha256=identity['identity_sha256'],
        source_identity_file_sha256=gw4.base.sha256(run/'identity.json'),
        completed_adaptation_epochs=epochs, archives=archives, reused_arms=roles,
        GW2_reference=identity['reference'], source_artifacts_written=False,
        new_training_updates=0, new_inference_calls=0)
    return certificate, identity, source_weights, states


def recertify(saved):
    current, identity, weights, states = certify(saved['source_run'],
        {key:saved['archives'][key]['directory'] for key in ('best', 'last')})
    require(current == saved, 'GW4 reference changed since GW5 construction')
    return identity, weights, states


def load_cached(reference, selection):
    saved = reference['archives'][selection]
    directory = Path(saved['directory'])
    require(gw4.base.sha256(directory/'metadata.json') == saved['metadata_sha256'], 'GW4 cache metadata changed')
    path = directory/'paired_predictions.npz'
    require(gw4.base.sha256(path) == saved['archive_sha256'], 'GW4 cache archive changed')
    with np.load(path, allow_pickle=False) as payload:
        archive = dict(payload)
    evaluation.validate_archive(archive)
    require(gw4.base.sha256(path) == saved['archive_sha256'], 'GW4 archive changed during read')
    return archive, read_json(directory/'metadata.json')
